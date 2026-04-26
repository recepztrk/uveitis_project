# ==============================================================================
# gradcam_slitlamp.py
#
# Slit-lamp modeli için Grad-CAM (Gradient-weighted Class Activation Mapping)
# açıklanabilirlik analizi scripti.
#
# Grad-CAM, modelin tahmin sırasında görüntünün HANGİ BÖLGELERİNE odaklandığını
# ısı haritası olarak görselleştirir. Bu sayede modelin klinik açıdan anlamlı
# bölgelere (ör. iltihap, kızarıklık) bakıp bakmadığı doğrulanabilir.
#
# Seçilen örnekler:
#   - 2 True Positive  (doğru üveit tahminleri)
#   - 2 True Negative  (doğru non-üveit tahminleri)
#   - 1 False Positive  (yanlış üveit alarmı)
#   - 1 False Negative  (kaçırılan üveit)
#
# Çıktılar:
#   - outputs/gradcam/gradcam_*.png                    (3 panelli görseller)
#   - outputs/gradcam/slitlamp_test_predictions.json   (tüm test tahminleri)
# ==============================================================================

import sys
from pathlib import Path
import json
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

from src.slitlamp_dataset import SlitLampDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradCAM:
    """Grad-CAM ısı haritası üretici.

    Çalışma prensibi:
    1. Hedef katmana forward hook ve backward hook kaydedilir
    2. Forward pass sırasında aktivasyonlar yakalanır
    3. Backward pass sırasında gradyanlar yakalanır
    4. Gradyanların global ortalaması alınarak her kanal için ağırlık hesaplanır
    5. Ağırlıklı aktivasyonların toplamı ile ısı haritası oluşturulur
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Hook'ları kaydet: ileri ve geri geçiş sırasında çağrılacaklar
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Forward pass sırasında hedef katmanın aktivasyonlarını yakala."""
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        """Backward pass sırasında hedef katmana gelen gradyanları yakala."""
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        """İşlem bitince hook'ları temizle (bellek sızıntısını önlemek için)."""
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate(self, input_tensor, class_idx=None):
        """Verilen girdi için Grad-CAM ısı haritası üretir.

        Returns:
            cam: 0-1 arasında normalize edilmiş 2D numpy array (ısı haritası)
        """
        self.model.zero_grad()

        output = self.model(input_tensor)

        # İkili sınıflandırma: tek çıkış nöronu, index=0
        if class_idx is None:
            class_idx = 0

        # Hedef sınıf skoru üzerinden geri yayılım yap
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        # Her kanal için gradyanların uzamsal ortalamasını al → kanal ağırlıkları
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Ağırlıklı aktivasyon toplamı → ham CAM
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # ReLU: sadece pozitif etkili bölgeleri tut
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        # 0-1 arasında normalize et
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def build_model():
    """EfficientNet-B0 model yapısını oluşturur (ağırlıklar sonra yüklenecek)."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def get_eval_transform():
    """Model beslemek için kullanılan standart değerlendirme transform'u."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_vis_transform():
    """Görselleştirme için sadece resize uygulayan transform (normalize yok)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
    ])


@torch.no_grad()
def collect_predictions(model, dataset):
    """Test setindeki tüm görüntüler için tahmin toplar.
    Her örnek için image_id, gerçek etiket, tahmin ve olasılık kaydedilir."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    results = []

    model.eval()

    for batch in loader:
        image = batch["image"].to(DEVICE)
        label = int(batch["label"].item())
        image_id = batch["image_id"][0]
        raw_class = batch["raw_class"][0]
        filepath = batch["filepath"][0]

        logits = model(image)
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob >= 0.5 else 0

        results.append({
            "image_id": image_id,
            "filepath": filepath,
            "raw_class": raw_class,
            "label": label,
            "pred": pred,
            "prob": prob,
        })

    return results


def overlay_cam_on_image(pil_img, cam):
    """CAM ısı haritasını orijinal görüntünün üzerine bindirir.
    %60 orijinal + %40 ısı haritası oranında karıştırılır."""
    img = np.array(pil_img).astype(np.uint8)
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))

    # CAM değerlerini JET renk haritasına dönüştür
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # OpenCV BGR → RGB

    overlay = (0.6 * img + 0.4 * heatmap).astype(np.uint8)
    return overlay


def select_examples(results):
    """Grad-CAM görselleştirmesi için temsili örnekler seçer.
    2 TP + 2 TN + 1 FP + 1 FN → toplamda 6 örnek."""
    tp = [r for r in results if r["label"] == 1 and r["pred"] == 1]  # True Positive
    tn = [r for r in results if r["label"] == 0 and r["pred"] == 0]  # True Negative
    fp = [r for r in results if r["label"] == 0 and r["pred"] == 1]  # False Positive
    fn = [r for r in results if r["label"] == 1 and r["pred"] == 0]  # False Negative

    selected = []

    selected += tp[:2]   # İlk 2 doğru üveit tahmini
    selected += tn[:2]   # İlk 2 doğru non-üveit tahmini

    if len(fp) > 0:
        selected += fp[:1]   # 1 yanlış alarm
    if len(fn) > 0:
        selected += fn[:1]   # 1 kaçırılan üveit

    return selected


def main():
    output_dir = PROJECT_ROOT / "outputs/gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Eğitilmiş modeli yükle
    model = build_model().to(DEVICE)
    model.load_state_dict(
        torch.load(str(PROJECT_ROOT / "outputs/models/slitlamp_efficientnetb0_best.pth"), map_location=DEVICE)
    )
    model.eval()

    eval_transform = get_eval_transform()

    # Test setini yükle
    test_dataset = SlitLampDataset(
        csv_path=str(PROJECT_ROOT / "metadata/slitlamp_split.csv"),
        split="test",
        transform=eval_transform
    )

    # Tüm test örnekleri için tahmin topla
    results = collect_predictions(model, test_dataset)

    # Tüm tahminleri JSON olarak kaydet (analiz için)
    with open(output_dir / "slitlamp_test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Grad-CAM için temsili örnekleri seç
    selected = select_examples(results)

    # EfficientNet-B0'ın son konvolüsyon bloğunu hedef al
    # Bu katman en yüksek seviye öznitelikleri içerir
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    # --- Seçilen her örnek için Grad-CAM görselleştirmesi üret ---
    for i, item in enumerate(selected, start=1):
        # Orijinal görüntüyü görselleştirme için yükle
        raw_img = Image.open(item["filepath"]).convert("RGB").resize((224, 224))

        # Model için normalize edilmiş tensor oluştur
        input_tensor = eval_transform(raw_img).unsqueeze(0).to(DEVICE)

        # Grad-CAM ısı haritası üret
        cam = gradcam.generate(input_tensor)
        overlay = overlay_cam_on_image(raw_img, cam)

        # 3 panelli görselleştirme: Orijinal | Grad-CAM | Overlay
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(raw_img)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        # Başlık: örnek bilgileri (gerçek etiket, tahmin, olasılık, alt sınıf)
        label_name = "Uveitis" if item["label"] == 1 else "Non-Uveitis"
        pred_name = "Uveitis" if item["pred"] == 1 else "Non-Uveitis"

        fig.suptitle(
            f"{item['image_id']} | true={label_name} | pred={pred_name} | prob={item['prob']:.4f} | raw_class={item['raw_class']}",
            fontsize=10
        )

        save_path = output_dir / f"gradcam_{i}_{item['image_id']}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    # Bellek sızıntısını önlemek için hook'ları temizle
    gradcam.remove_hooks()

    print("Grad-CAM örnekleri kaydedildi:", output_dir)


if __name__ == "__main__":
    main()