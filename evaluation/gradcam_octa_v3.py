# ==============================================================================
# gradcam_octa_v3.py
#
# OCTA V3 modeli (ResNet-18) için Grad-CAM açıklanabilirlik analizi ve
# kapsamlı görselleştirme scripti.
#
# Üretilen çıktılar:
#   1. Grad-CAM ısı haritaları (6 örnek: 2 TP, 2 TN, 1 FP, 1 FN)
#   2. Training loss & F1 eğrileri
#   3. V1/V2/V3 model karşılaştırma bar grafiği
#   4. Test tahminleri JSON
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

from src.octa_dataset_v2 import OCTADatasetV2

DEVICE = torch.device("cpu")


# === GRAD-CAM SINIFI ===
class GradCAM:
    """Grad-CAM ısı haritası üretici (ResNet-18 uyumlu)."""

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
    """ResNet-18 model yapısını oluşturur."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
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
        filepath = batch["filepath"][0]

        logits = model(image)
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob >= 0.5 else 0

        results.append({
            "image_id": image_id,
            "filepath": filepath,
            "source_class": source_class,
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

    return selected


def generate_gradcam_images(model, selected, output_dir):
    """Seçilen örnekler için Grad-CAM görselleri üretir."""
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ResNet-18'in son konvolüsyon bloğu: layer4
    target_layer = model.layer4[-1]
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
            pred_type = "TN (Doğru Kontrol)"
        elif item["label"] == 0 and item["pred"] == 1:
            pred_type = "FP (Yanlış Alarm)"
        else:
            pred_type = "FN (Kaçırılan)"

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        axes[0].imshow(raw_img)
        axes[0].set_title("Orijinal OCTA", fontsize=11)
        axes[0].axis("off")

        im = axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM Isı Haritası", fontsize=11)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=11)
        axes[2].axis("off")

        label_name = "Uveitis" if item["label"] == 1 else "Control"
        pred_name = "Uveitis" if item["pred"] == 1 else "Control"
        source = "OptoVue" if "UT-OCTA" in item["filepath"] else "Heidelberg"

        fig.suptitle(
            f"{pred_type}  |  Gerçek: {label_name}  |  Tahmin: {pred_name}  |  "
            f"Olasılık: {item['prob']:.3f}  |  Kaynak: {source}  |  [{item['image_id']}]",
            fontsize=10, fontweight="bold"
        )

        save_path = output_dir / f"octa_v3_gradcam_{i}_{item['image_id']}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Kaydedildi: {save_path.name}")

    gradcam.remove_hooks()


def generate_training_curves(metrics_dir):
    """V3 eğitim loss ve F1 eğrilerini çizer."""
    history_path = metrics_dir / "octa_v3_train_history.json"
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
    axes[0].set_title('OCTA V3 — Loss Eğrisi')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 eğrisi
    axes[1].plot(epochs, train_f1, 'b-', label='Train F1', linewidth=2)
    axes[1].plot(epochs, val_f1, 'r-', label='Val F1', linewidth=2)
    axes[1].axvline(x=5, color='gray', linestyle='--', alpha=0.5, label='Unfreeze')
    best_epoch = epochs[np.argmax(val_f1)]
    best_f1 = max(val_f1)
    axes[1].scatter([best_epoch], [best_f1], color='gold', s=100, zorder=5, label=f'Best (Epoch {best_epoch})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('OCTA V3 — F1 Score Eğrisi')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # AUC eğrisi
    axes[2].plot(epochs, val_auc, 'g-', label='Val AUC', linewidth=2)
    axes[2].axvline(x=5, color='gray', linestyle='--', alpha=0.5, label='Unfreeze')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].set_title('OCTA V3 — Validation AUC Eğrisi')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = metrics_dir / "octa_v3_training_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {save_path.name}")


def generate_version_comparison(metrics_dir):
    """V1/V2/V3 model karşılaştırma bar grafiği."""
    v3_path = metrics_dir / "octa_v3_test_metrics.json"
    with open(v3_path, "r") as f:
        v3_data = json.load(f)

    comparison = v3_data.get("version_comparison", {})
    if not comparison:
        print("  ⚠️ Karşılaştırma verisi bulunamadı, atlanıyor.")
        return

    versions = []
    metrics_data = {}
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for ver in ["v1", "v2", "v3"]:
        if ver in comparison:
            versions.append(ver.upper())
            for key in metric_keys:
                if key not in metrics_data:
                    metrics_data[key] = []
                metrics_data[key].append(comparison[ver].get(key, 0))

    x = np.arange(len(metric_keys))
    width = 0.25
    colors = ['#4ECDC4', '#FF6B6B', '#45B7D1']
    labels_display = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (ver, color) in enumerate(zip(versions, colors)):
        values = [metrics_data[k][i] for k in metric_keys]
        bars = ax.bar(x + i * width, values, width, label=ver, color=color, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Metrik', fontsize=12)
    ax.set_ylabel('Skor', fontsize=12)
    ax.set_title('OCTA Model Versiyonları Karşılaştırması (Test Seti)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels_display, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = metrics_dir / "octa_v3_version_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {save_path.name}")


def main():
    gradcam_dir = PROJECT_ROOT / "outputs/gradcam"
    metrics_dir = PROJECT_ROOT / "outputs/metrics"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model yükle
    print("=" * 60)
    print("OCTA V3 — Görselleştirme ve Grad-CAM Analizi")
    print("=" * 60)

    model = build_model().to(DEVICE)
    model.load_state_dict(
        torch.load(str(PROJECT_ROOT / "outputs/models/octa_v3_resnet18_best.pth"), map_location=DEVICE)
    )
    model.eval()
    print("✓ Model yüklendi")

    # 2. Test tahminleri
    print("\n[1/4] Test tahminleri toplanıyor...")
    test_dataset = OCTADatasetV2(
        csv_path=str(PROJECT_ROOT / "metadata/octa_split.csv"),
        split="test"
    )

    results = collect_predictions(model, test_dataset)
    with open(gradcam_dir / "octa_v3_test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    tp = sum(1 for r in results if r["label"]==1 and r["pred"]==1)
    tn = sum(1 for r in results if r["label"]==0 and r["pred"]==0)
    fp = sum(1 for r in results if r["label"]==0 and r["pred"]==1)
    fn = sum(1 for r in results if r["label"]==1 and r["pred"]==0)
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # 3. Grad-CAM
    print("\n[2/4] Grad-CAM görselleri üretiliyor...")
    selected = select_examples(results)
    generate_gradcam_images(model, selected, gradcam_dir)

    # 4. Training curves
    print("\n[3/4] Eğitim eğrileri çiziliyor...")
    generate_training_curves(metrics_dir)

    # 5. Version comparison
    print("\n[4/4] Model karşılaştırma grafiği oluşturuluyor...")
    generate_version_comparison(metrics_dir)

    print(f"\n{'='*60}")
    print("TÜM ÇIKTILAR BAŞARIYLA ÜRETİLDİ ✓")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
