# ==============================================================================
# gradcam_cfp.py
#
# CFP modeli (EfficientNet-B0) için Grad-CAM açıklanabilirlik analizi ve
# kapsamlı görselleştirme scripti.
#
# Üretilen çıktılar:
#   1. Grad-CAM ısı haritaları (6 örnek: 2 TP, 2 TN, 1 FP, 1 FN)
#   2. Training loss & F1 eğrileri
#   3. Test tahminleri JSON
# ==============================================================================

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

from src.cfp_dataset import CFPDataset

DEVICE = torch.device("cpu")


# === GRAD-CAM SINIFI ===
class GradCAM:
    """Grad-CAM ısı haritası üretici (EfficientNet-B0 uyumlu)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, 0]
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def build_model():
    """EfficientNet-B0 model yapısını oluşturur."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def overlay_cam_on_image(pil_img, cam):
    """CAM ısı haritasını orijinal görüntünün üzerine bindirir."""
    img = np.array(pil_img).astype(np.uint8)
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * img + 0.4 * heatmap).astype(np.uint8)
    return overlay


@torch.no_grad()
def collect_predictions(model, dataset):
    """Test setindeki tüm görüntüler için tahmin toplar."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    results = []
    model.eval()

    for batch in loader:
        image = batch["image"].to(DEVICE)
        label = int(batch["label"].item())
        image_id = batch["image_id"][0]
        source_class = batch["source_class"][0]
        source_dataset = batch["source_dataset"][0]
        filepath = batch["filepath"][0]

        logits = model(image)
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob >= 0.5 else 0

        results.append({
            "image_id": image_id,
            "filepath": filepath,
            "source_class": source_class,
            "source_dataset": source_dataset,
            "label": label,
            "pred": pred,
            "prob": round(prob, 4),
        })

    return results


def select_examples(results):
    """Grad-CAM için temsili örnekler: 2 TP + 2 TN + 1 FP + 1 FN."""
    tp = [r for r in results if r["label"] == 1 and r["pred"] == 1]
    tn = [r for r in results if r["label"] == 0 and r["pred"] == 0]
    fp = [r for r in results if r["label"] == 0 and r["pred"] == 1]
    fn = [r for r in results if r["label"] == 1 and r["pred"] == 0]

    selected = []
    selected += tp[:2]
    selected += tn[:2]
    if fp:
        selected += fp[:1]
    if fn:
        selected += fn[:1]

    print(f"\n  Seçilen örnekler: {len(tp)} TP, {len(tn)} TN, {len(fp)} FP, {len(fn)} FN")
    print(f"  Grad-CAM üretilecek: {len(selected)} görüntü")
    return selected


def generate_gradcam_images(model, selected, output_dir):
    """Seçilen örnekler için Grad-CAM görselleri üretir."""
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # EfficientNet-B0'ın son konvolüsyon bloğu: features[-1]
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    for i, item in enumerate(selected, start=1):
        raw_img = Image.open(item["filepath"]).convert("RGB").resize((224, 224))
        input_tensor = eval_transform(raw_img).unsqueeze(0).to(DEVICE)

        model.train()  # Grad hesaplaması için gerekli
        cam = gradcam.generate(input_tensor)
        model.eval()

        overlay = overlay_cam_on_image(raw_img, cam)

        # Tahmin tipini belirle
        if item["label"] == 1 and item["pred"] == 1:
            pred_type = "TP (Doğru Üveit)"
        elif item["label"] == 0 and item["pred"] == 0:
            pred_type = "TN (Doğru Non-Üveit)"
        elif item["label"] == 0 and item["pred"] == 1:
            pred_type = "FP (Yanlış Alarm)"
        else:
            pred_type = "FN (Kaçırılan)"

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(raw_img)
        axes[0].set_title("Orijinal CFP", fontsize=11)
        axes[0].axis("off")

        im = axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM Isı Haritası", fontsize=11)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=11)
        axes[2].axis("off")

        label_name = "Uveitis" if item["label"] == 1 else "Non-Uveitis"
        pred_name = "Uveitis" if item["pred"] == 1 else "Non-Uveitis"

        fig.suptitle(
            f"{pred_type}  |  Gerçek: {label_name}  |  Tahmin: {pred_name}  |  "
            f"Olasılık: {item['prob']:.3f}  |  Kaynak: {item['source_dataset']}  |  "
            f"Sınıf: {item['source_class']}  |  [{item['image_id']}]",
            fontsize=9, fontweight="bold"
        )

        save_path = output_dir / f"cfp_gradcam_{i}_{item['image_id']}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Kaydedildi: {save_path.name}")

    gradcam.remove_hooks()


def generate_training_curves(metrics_dir):
    """CFP eğitim loss ve F1 eğrilerini çizer."""
    history_path = metrics_dir / "cfp_train_history.json"
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_f1 = [h["train_f1"] for h in history]
    val_f1 = [h["val_f1"] for h in history]
    val_auc = [h["val_auc"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss eğrisi
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    axes[0].axvline(x=5, color='gray', linestyle='--', alpha=0.5, label='Unfreeze')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('CFP — Loss Eğrisi')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 eğrisi
    axes[1].plot(epochs, train_f1, 'b-', label='Train F1', linewidth=2)
    axes[1].plot(epochs, val_f1, 'r-', label='Val F1', linewidth=2)
    axes[1].axvline(x=5, color='gray', linestyle='--', alpha=0.5, label='Unfreeze')
    best_epoch = epochs[np.argmax(val_f1)]
    best_f1 = max(val_f1)
    axes[1].scatter([best_epoch], [best_f1], color='gold', s=100, zorder=5,
                    label=f'Best (Epoch {best_epoch})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('CFP — F1 Score Eğrisi')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # AUC eğrisi
    axes[2].plot(epochs, val_auc, 'g-', label='Val AUC', linewidth=2)
    axes[2].axvline(x=5, color='gray', linestyle='--', alpha=0.5, label='Unfreeze')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].set_title('CFP — Validation AUC Eğrisi')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = metrics_dir / "cfp_training_curves.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Eğitim eğrileri kaydedildi: {save_path.name}")


def main():
    print("=" * 60)
    print("CFP GRAD-CAM ANALİZİ")
    print("=" * 60)

    gradcam_dir = PROJECT_ROOT / "outputs/gradcam"
    metrics_dir = PROJECT_ROOT / "outputs/metrics"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model yükle
    model_path = PROJECT_ROOT / "outputs/models/cfp_efficientnetb0_best.pth"
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(str(model_path), map_location=DEVICE,
                                     weights_only=True))
    model.eval()
    print(f"Model yüklendi: {model_path.name}")

    # 2. Test tahminlerini topla
    print("\nTest seti tahminleri toplanıyor...")
    csv_path = str(PROJECT_ROOT / "metadata/cfp_split.csv")
    test_dataset = CFPDataset(csv_path, split="test")
    results = collect_predictions(model, test_dataset)

    # Tahminleri JSON olarak kaydet
    pred_path = gradcam_dir / "cfp_test_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Tahminler kaydedildi: {pred_path.name}")

    # 3. Temsili örnekleri seç
    selected = select_examples(results)

    # 4. Grad-CAM görselleri üret
    print("\nGrad-CAM görselleri üretiliyor...")
    generate_gradcam_images(model, selected, gradcam_dir)

    # 5. Eğitim eğrileri
    print("\nEğitim eğrileri çiziliyor...")
    generate_training_curves(metrics_dir)

    print(f"\n{'='*60}")
    print("TAMAMLANDI!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
