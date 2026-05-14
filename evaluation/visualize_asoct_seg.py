import os
import cv2
import numpy as np
import torch
from pathlib import Path
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import signal

def handler(signum, frame):
    raise TimeoutError("iCloud timeout")
signal.signal(signal.SIGALRM, handler)

# --- AYARLAR ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "asoct_seg" / "models" / "asoct_unet_best.pth"
AIDK_DIR = PROJECT_ROOT / "Veri Setleri" / "Ön segment OCT (AS-OCT)" / "AIDK_Dataset_-_asoct" / "Full-frame_Dataset"
IMG_DIR = AIDK_DIR / "Original_AS-OCT_Images"
MASK_DIR = PROJECT_ROOT / "data_raw" / "asoct" / "masks"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "asoct_seg" / "visuals"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NUM_CLASSES = 6

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Renk Paleti (RGB)
# 0: Arka Plan (Siyah - Saydam)
# 1: Cornea (Kırmızı)
# 2: Iris (Mavi)
# 3: Lesion (Sarı)
# 4: Anterior Chamber (Yeşil)
# 5: Lens (Mor)
COLOR_MAP = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (0, 255, 0),
    5: (128, 0, 128)
}

def decode_segmap(mask):
    """Sınıf indekslerini (0-5) RGB renk matrisine çevirir."""
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in COLOR_MAP.items():
        rgb[mask == class_idx] = color
    return rgb

def overlay_mask(image, mask_rgb, alpha=0.5):
    """Orijinal resim üzerine maskeyi alpha-blending ile bindirir."""
    # Sadece arka plan olmayan yerleri kapla
    gray_mask = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
    mask_indices = gray_mask > 0
    
    overlay = image.copy()
    overlay[mask_indices] = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)[mask_indices]
    return overlay

def main():
    print(f"[*] Visualizer Başlatılıyor. Cihaz: {DEVICE}")
    
    if not MODEL_PATH.exists():
        print(f"[!] HATA: Model bulunamadı ({MODEL_PATH}). Lütfen önce eğitimi tamamlayın.")
        return

    # Model Yükleme
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    print("[*] UNet Modeli yüklendi.")

    # Görüntü Dönüşümleri (Sadece Resize + Normalize)
    transform = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Sadece belli sayıda (örn 20 tane) örnek üzerinden gösterim yapalım
    images = sorted(list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png")) + list(IMG_DIR.glob("*.bmp")))
    
    # Rastgele 20 tane seç (tekrarlanabilir olması için seed)
    np.random.seed(42)
    test_images = np.random.choice(images, min(20, len(images)), replace=False)

    for img_path in tqdm(test_images, desc="Görseller Üretiliyor"):
        # iCloud Kalkanı ile okuma
        try:
            signal.alarm(2)
            orig_img = cv2.imread(str(img_path))
            signal.alarm(0)
            if orig_img is None:
                continue
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        except Exception:
            signal.alarm(0)
            continue
            
        # Ön işleme
        augmented = transform(image=orig_img)
        input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
        
        # Orijinal resmi 256x256 olarak görselleştirme için hazırla
        vis_img = cv2.resize(orig_img, (256, 256))

        # Tahmin (Inference)
        with torch.no_grad():
            output = model(input_tensor)
            # Logit'leri sınıflara çevir (Argmax)
            pred_mask = output.argmax(dim=1).squeeze(0).cpu().numpy()

        # Maskeyi Renklendir
        rgb_mask = decode_segmap(pred_mask)
        
        # Orijinal resim ile maskeyi birleştir
        blended = overlay_mask(vis_img, rgb_mask, alpha=0.5)

        # Görselleri yan yana koy: [Orijinal | Renkli Maske | Birleştirilmiş (Overlay)]
        # Fakat BGR yapısına geri dönmeliyiz ki cv2.imwrite doğru kaydetsin
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        rgb_mask_bgr = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        
        combined_vis = np.hstack((vis_img_bgr, rgb_mask_bgr, blended_bgr))
        
        # Başlıkla kaydet (Dosya adı: asoct_seg_236.png vs)
        save_path = OUTPUT_DIR / f"asoct_seg_{img_path.stem}.png"
        cv2.imwrite(str(save_path), combined_vis)

    print(f"\n[*] İşlem tamamlandı. Görseller kaydedildi: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
