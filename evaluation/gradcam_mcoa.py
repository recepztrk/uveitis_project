# ==============================================================================
# gradcam_mcoa.py
#
# AS-OCT (MCOA) modeli için Grad-CAM açıklanabilirlik analizi.
# timm Noisy Student EfficientNet-B0 için target layer: model.conv_head
#
# Üretilen çıktılar (outputs/mcoa/gradcam/):
#   - mcoa_gradcam_{i}_{label}_{pred_type}.png  (6 örnek: 2 TP + 2 TN + 1 FP + 1 FN)
#   - mcoa_gradcam_summary.png                  (6 örnek yan yana özet panel)
# ==============================================================================

import sys
import json
from pathlib import Path
import signal

def handler(signum, frame):
    raise TimeoutError("iCloud timeout")

signal.signal(signal.SIGALRM, handler)

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
from torchvision import transforms
import timm
import pandas as pd

# === AYARLAR ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH  = PROJECT_ROOT / "outputs/mcoa/models/mcoa_efficientnet_best.pth"
SPLIT_CSV   = PROJECT_ROOT / "metadata/mcoa_split.csv"
GRADCAM_DIR = PROJECT_ROOT / "outputs/mcoa/gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
THRESHOLD   = 0.50


# === GRAD-CAM ===
class GradCAM:
    """Grad-CAM ısı haritası üretici (timm EfficientNet-B0 uyumlu).

    timm EfficientNet-B0'ın hedef katmanı: model.conv_head
    Bu katman, son konvolüsyon bloğunun çıkışını temsil eder.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_hook  = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        """Grad-CAM ısı haritası üretir (0-1 arası normalize)."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, 0]
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        if cam.ndim == 0:  # skalar olursa (edge case)
            cam = np.zeros((7, 7))
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


# === MODEL ===
def build_model():
    model = timm.create_model("tf_efficientnet_b0.ns_jft_in1k", pretrained=False, num_classes=1)
    return model


# === YARDIMCI FONKSİYONLAR ===
def overlay_cam_on_image(pil_img, cam):
    """CAM ısı haritasını orijinal görüntünün üzerine bindirir."""
    img = np.array(pil_img).astype(np.uint8)
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * img + 0.4 * heatmap).astype(np.uint8)
    return overlay, cam_resized


@torch.no_grad()
def collect_predictions(model, csv_path, split="test"):
    """Test setindeki tüm görüntüler için tahmin toplar."""
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    df = pd.read_csv(csv_path)
    df = df[df["split"] == split].reset_index(drop=True)

    model.eval()
    results = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        try:
            signal.alarm(2)
            img_cv = cv2.imread(row["filepath"])
            signal.alarm(0)
            if img_cv is None:
                continue
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_cv)
        except Exception:
            signal.alarm(0)
            continue

        input_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)
        prob  = torch.sigmoid(model(input_tensor)).item()
        label = int(row["label"])
        pred  = int(prob >= THRESHOLD)

        results.append({
            "filepath": row["filepath"],
            "label":    label,
            "pred":     pred,
            "prob":     round(prob, 4),
        })

    return results


def select_examples(results):
    """Grad-CAM için temsili örnekler: 2 TP + 2 TN + 1 FP + 1 FN."""
    tp = [r for r in results if r["label"] == 1 and r["pred"] == 1]
    tn = [r for r in results if r["label"] == 0 and r["pred"] == 0]
    fp = [r for r in results if r["label"] == 0 and r["pred"] == 1]
    fn = [r for r in results if r["label"] == 1 and r["pred"] == 0]

    print(f"  Test seti: {len(results)} | TP:{len(tp)} TN:{len(tn)} FP:{len(fp)} FN:{len(fn)}")

    selected = []
    selected += tp[:2]
    selected += tn[:2]
    if fp: selected += fp[:1]
    if fn: selected += fn[:1]

    print(f"  Grad-CAM üretilecek: {len(selected)} görüntü")
    return selected


def pred_type_label(item):
    if item["label"] == 1 and item["pred"] == 1:
        return "TP", "Doğru Pozitif (Patolojik → Patolojik)"
    elif item["label"] == 0 and item["pred"] == 0:
        return "TN", "Doğru Negatif (Normal → Normal)"
    elif item["label"] == 0 and item["pred"] == 1:
        return "FP", "Yanlış Alarm (Normal → Patolojik)"
    else:
        return "FN", "Kaçırılan Vaka (Patolojik → Normal)"


def generate_gradcam_images(model, selected, output_dir):
    """Seçilen örnekler için bireysel Grad-CAM görseli üretir."""
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    target_layer = model.conv_head
    gradcam = GradCAM(model, target_layer)

    saved_paths = []

    for i, item in enumerate(selected, start=1):
        short_code, full_label = pred_type_label(item)
        label_name = "Opaque (Patolojik)" if item["label"] == 1 else "Normal"
        pred_name  = "Opaque (Patolojik)" if item["pred"]  == 1 else "Normal"

        try:
            signal.alarm(2)
            img_cv = cv2.imread(item["filepath"])
            signal.alarm(0)
        except Exception:
            signal.alarm(0)
            img_cv = None

        if img_cv is not None:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            raw_img = Image.fromarray(img_cv).resize((224, 224))
        else:
            raw_img = Image.new("RGB", (224, 224), (0, 0, 0))
        input_tensor = eval_transform(raw_img).unsqueeze(0).to(DEVICE)

        model.train()  # Grad hesaplaması için gerekli
        cam = gradcam.generate(input_tensor)
        model.eval()

        overlay, cam_resized = overlay_cam_on_image(raw_img, cam)

        # Çerçeve rengi: TP/TN yeşil, FP/FN kırmızı
        border_color = "#22C55E" if short_code in ("TP", "TN") else "#EF4444"

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#1a1a2e')

        titles = ["Orijinal AS-OCT", "Grad-CAM Isı Haritası", "Grad-CAM Overlay"]
        imgs   = [np.array(raw_img), cam_resized, overlay]
        cmaps  = [None, 'jet', None]

        for ax, title, img, cmap in zip(axes, titles, imgs, cmaps):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=11, color='white', pad=8)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)

        fig.suptitle(
            f"[{short_code}] {full_label}  |  Gerçek: {label_name}  |  "
            f"Tahmin: {pred_name}  |  Olasılık: {item['prob']:.3f}",
            fontsize=10, fontweight="bold", color='white', y=1.01
        )

        save_path = output_dir / f"mcoa_gradcam_{i:02d}_{short_code}.png"
        plt.tight_layout(pad=1.5)
        plt.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor='#1a1a2e', edgecolor='none')
        plt.close()
        saved_paths.append((item, save_path, short_code))
        print(f"  [{i}] {short_code} — Kaydedildi: {save_path.name}")

    gradcam.remove_hooks()
    return saved_paths


def generate_summary_panel(saved_items, output_dir):
    """Tüm Grad-CAM örneklerini tek bir özet panelde gösterir."""
    n = len(saved_items)
    if n == 0:
        return

    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    fig.patch.set_facecolor('#0f172a')

    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (item, path, short_code) in enumerate(saved_items):
        row_idx = idx // cols
        col_idx = idx % cols
        ax = axes[row_idx][col_idx]

        img = plt.imread(str(path))
        ax.imshow(img)
        ax.axis("off")

        short_code_color = "#22C55E" if short_code in ("TP", "TN") else "#EF4444"
        label_name = "Patolojik" if item["label"] == 1 else "Normal"
        ax.set_title(f"[{short_code}] Gerçek: {label_name} | p={item['prob']:.3f}",
                     fontsize=9, color=short_code_color, fontweight='bold')

    # Boş eksen hücrelerini gizle
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle("AS-OCT (MCOA) — Grad-CAM Özet Paneli", fontsize=16,
                 fontweight='bold', color='white', y=1.01)
    plt.tight_layout(pad=2.0)
    save_path = output_dir / "mcoa_gradcam_summary.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor='#0f172a', edgecolor='none')
    plt.close()
    print(f"  Özet panel kaydedildi: {save_path.name}")


def main():
    print("=" * 65)
    print("AS-OCT (MCOA) — Grad-CAM Analizi")
    print("=" * 65)
    print(f"  Cihaz      : {DEVICE}")
    print(f"  Target Layer: model.conv_head  (timm EfficientNet-B0)")

    # 1. Model yükle
    model = build_model().to(DEVICE)
    state = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("  Model yüklendi.\n")

    # 2. Tahminleri topla
    print(f"{'─'*65}")
    print("Test tahminleri toplanıyor...")
    results = collect_predictions(model, str(SPLIT_CSV), split="test")

    # 3. Örnekleri seç
    selected = select_examples(results)

    # 4. Grad-CAM üret
    print(f"\n{'─'*65}")
    print("Grad-CAM görselleri üretiliyor...")
    saved_items = generate_gradcam_images(model, selected, GRADCAM_DIR)

    # 5. Özet panel
    print(f"\n{'─'*65}")
    print("Özet panel oluşturuluyor...")
    generate_summary_panel(saved_items, GRADCAM_DIR)

    print(f"\n{'='*65}")
    print("TAMAMLANDI!")
    print(f"  Çıktı klasörü: {GRADCAM_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
