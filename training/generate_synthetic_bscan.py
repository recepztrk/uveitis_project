# ==============================================================================
# generate_synthetic_bscan.py
#
# 27 üveit + 28 normal B-scan OCT görüntüsünden sentetik veri üretir.
#
# YÖNTEMLERİ:
#   1. Tıbbi Augmentation: Elastic deformation, grid distortion, CLAHE,
#      gaussian noise — gerçek OCT varyasyonlarını simüle eder
#   2. Mixup: İki görüntüyü karıştırarak yeni görüntü üretir
#   3. CutMix: Bir görüntünün bir bölgesini başka görüntüyle değiştirir
#
# ÇIKTI:
#   data_work/bscan_oct_synthetic/uveitis/     → sentetik üveit
#   data_work/bscan_oct_synthetic/non_uveitis/  → sentetik normal
#
# KULLANIM:
#   python training/generate_synthetic_bscan.py
# ==============================================================================

import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
import copy

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Her orijinal görüntüden kaç sentetik üretilecek
AUGMENTATIONS_PER_IMAGE = 8
# Mixup çifti sayısı (her sınıf için)
MIXUP_COUNT = 15
# CutMix sayısı (her sınıf için)
CUTMIX_COUNT = 10

random.seed(42)
np.random.seed(42)


# =============================================================================
# Augmentation Fonksiyonları
# =============================================================================

def elastic_deform(image, alpha=30, sigma=4):
    """Elastic deformation — doku varyasyonu simüle eder.
    Tıbbi görüntülerde en etkili augmentation yöntemidir.
    """
    img_array = np.array(image, dtype=np.float32)
    shape = img_array.shape[:2]

    # Rastgele yer değiştirme alanları
    dx = np.random.randn(*shape) * alpha
    dy = np.random.randn(*shape) * alpha

    # Gaussian blur ile yumuşat (doğal görünsün)
    from scipy.ndimage import gaussian_filter, map_coordinates
    dx = gaussian_filter(dx, sigma)
    dy = gaussian_filter(dy, sigma)

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x_new = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
    y_new = np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)

    # Her kanal için uygula
    if len(img_array.shape) == 3:
        result = np.zeros_like(img_array)
        for c in range(img_array.shape[2]):
            result[:, :, c] = map_coordinates(
                img_array[:, :, c], [y_new, x_new], order=1, mode='reflect'
            )
    else:
        result = map_coordinates(img_array, [y_new, x_new], order=1, mode='reflect')

    return Image.fromarray(result.astype(np.uint8))


def add_speckle_noise(image, intensity=0.05):
    """Speckle noise — OCT'ye özgü gürültü tipi."""
    img_array = np.array(image, dtype=np.float32) / 255.0
    noise = np.random.randn(*img_array.shape) * intensity
    noisy = img_array + img_array * noise
    noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def adjust_clahe_like(image, factor_range=(0.7, 1.5)):
    """Kontrast ayarı — farklı OCT cihaz ayarlarını simüle eder."""
    enhancer = ImageEnhance.Contrast(image)
    factor = random.uniform(*factor_range)
    return enhancer.enhance(factor)


def random_crop_resize(image, min_scale=0.75):
    """Rastgele bölge kırpıp orijinal boyuta geri büyütür."""
    w, h = image.size
    scale = random.uniform(min_scale, 0.95)
    new_w, new_h = int(w * scale), int(h * scale)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.LANCZOS)


def random_brightness(image, range_=(0.7, 1.3)):
    """Parlaklık ayarı."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(random.uniform(*range_))


def random_blur(image, max_radius=1.5):
    """Hafif bulanıklık — farklı odak durumlarını simüle eder."""
    radius = random.uniform(0, max_radius)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_random_augmentation(image):
    """Rastgele augmentation kombinasyonu uygular."""
    img = image.copy()

    # Her augmentation'ı belirli olasılıkla uygula
    if random.random() < 0.5:
        img = elastic_deform(img, alpha=random.randint(15, 40), sigma=random.randint(3, 6))
    if random.random() < 0.5:
        img = add_speckle_noise(img, intensity=random.uniform(0.02, 0.08))
    if random.random() < 0.6:
        img = adjust_clahe_like(img)
    if random.random() < 0.5:
        img = random_crop_resize(img)
    if random.random() < 0.4:
        img = random_brightness(img)
    if random.random() < 0.3:
        img = random_blur(img)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.3:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=(0, 0, 0))

    return img


# =============================================================================
# Mixup
# =============================================================================
def mixup(img1, img2, alpha=0.3):
    """İki görüntüyü belirli oranda karıştırır.
    alpha=0.3 → %70 img1 + %30 img2
    """
    lam = random.uniform(alpha, 1 - alpha)
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)

    # Boyutları eşitle
    if arr1.shape != arr2.shape:
        img2 = img2.resize(img1.size, Image.LANCZOS)
        arr2 = np.array(img2, dtype=np.float32)

    mixed = (lam * arr1 + (1 - lam) * arr2).astype(np.uint8)
    return Image.fromarray(mixed)


# =============================================================================
# CutMix
# =============================================================================
def cutmix(img1, img2, min_ratio=0.2, max_ratio=0.4):
    """img1'in rastgele bir bölgesini img2'den keserek yapıştırır."""
    w, h = img1.size
    img2_resized = img2.resize((w, h), Image.LANCZOS)

    ratio = random.uniform(min_ratio, max_ratio)
    cut_w = int(w * ratio)
    cut_h = int(h * ratio)

    cx = random.randint(0, w - cut_w)
    cy = random.randint(0, h - cut_h)

    result = img1.copy()
    patch = img2_resized.crop((cx, cy, cx + cut_w, cy + cut_h))
    result.paste(patch, (cx, cy))
    return result


# =============================================================================
# Ana Üretim
# =============================================================================
def generate_for_class(source_dir, output_dir, class_name):
    """Bir sınıf için sentetik veri üretir."""
    os.makedirs(output_dir, exist_ok=True)

    # Orijinal görüntüleri yükle
    images = []
    filenames = []
    for f in sorted(os.listdir(source_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img = Image.open(os.path.join(source_dir, f)).convert("RGB")
            images.append(img)
            filenames.append(f)

    print(f"\n{'='*50}")
    print(f"Sınıf: {class_name} — {len(images)} orijinal görüntü")
    print(f"{'='*50}")

    count = 0

    # 1. Augmentation tabanlı üretim
    print(f"  📐 Augmentation ({AUGMENTATIONS_PER_IMAGE}x her görüntü)...")
    for i, img in enumerate(images):
        for j in range(AUGMENTATIONS_PER_IMAGE):
            synth = apply_random_augmentation(img)
            name = f"synth_aug_{filenames[i].split('.')[0]}_{j}.jpg"
            synth.save(os.path.join(output_dir, name), quality=95)
            count += 1

    # 2. Mixup
    print(f"  🔀 Mixup ({MIXUP_COUNT} çift)...")
    for k in range(MIXUP_COUNT):
        idx1, idx2 = random.sample(range(len(images)), 2)
        mixed = mixup(images[idx1], images[idx2])
        mixed.save(os.path.join(output_dir, f"synth_mixup_{k:03d}.jpg"), quality=95)
        count += 1

    # 3. CutMix
    print(f"  ✂️  CutMix ({CUTMIX_COUNT} çift)...")
    for k in range(CUTMIX_COUNT):
        idx1, idx2 = random.sample(range(len(images)), 2)
        cut = cutmix(images[idx1], images[idx2])
        cut.save(os.path.join(output_dir, f"synth_cutmix_{k:03d}.jpg"), quality=95)
        count += 1

    print(f"  ✅ Toplam: {count} sentetik görüntü üretildi")
    return count


def main():
    uveitis_src = PROJECT_ROOT / "data_work/bscan_oct_clean/uveitis"
    normal_src = PROJECT_ROOT / "data_work/bscan_oct_clean/non_uveitis"

    synth_dir = PROJECT_ROOT / "data_work/bscan_oct_synthetic"

    print("=" * 50)
    print("B-scan OCT Sentetik Veri Üretimi")
    print("=" * 50)

    uv_count = generate_for_class(
        str(uveitis_src),
        str(synth_dir / "uveitis"),
        "Üveit"
    )

    norm_count = generate_for_class(
        str(normal_src),
        str(synth_dir / "non_uveitis"),
        "Normal"
    )

    print(f"\n{'='*50}")
    print(f"ÖZET")
    print(f"{'='*50}")
    print(f"  Üveit:  27 orijinal → +{uv_count} sentetik = {27 + uv_count} toplam")
    print(f"  Normal: 28 orijinal → +{norm_count} sentetik = {28 + norm_count} toplam")
    print(f"  Genel:  55 orijinal → +{uv_count + norm_count} sentetik = {55 + uv_count + norm_count} toplam")
    print(f"\n  📁 Konum: {synth_dir}")
    print(f"\n✅ Sentetik veri üretimi tamamlandı!")


if __name__ == "__main__":
    main()
