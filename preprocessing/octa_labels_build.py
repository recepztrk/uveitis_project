# ==============================================================================
# octa_labels_build.py
#
# OCTA (deep + superficial) görüntüleri için etiket CSV dosyası oluşturur.
# Temizlenmiş görüntülerin bulunduğu data_work/octa_clean/ klasöründeki
# dosyaları tarar ve her görüntü için metadata bilgilerini içeren
# metadata/octa_labels.csv dosyasını üretir.
#
# Dosya isim formatı: <source_class>_<layer_type>_<original_id>.png
# Örnek: active_deep_BH-OCTA-HD-01.png
#
# Üretilen CSV sütunları:
#   image_id, filepath, source_class, layer_type, original_id, group_id, binary_label
#
# Sınıf tanımları:
#   - Pozitif (1): active + inactive → uveitis
#   - Negatif (0): control → non_uveitis
# ==============================================================================

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Temizlenmiş OCTA verilerinin bulunduğu dizinler
base_dir = PROJECT_ROOT / "data_work/octa_clean"
uveitis_dir = base_dir / "uveitis"
non_uveitis_dir = base_dir / "non_uveitis"

# Her görüntünün metadata bilgisini tutacak liste
rows = []

# Desteklenen görüntü formatları
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_valid_image(fp: Path) -> bool:
    """Dosyanın geçerli bir görüntü olup olmadığını kontrol eder."""
    return fp.is_file() and not fp.name.startswith(".") and fp.suffix.lower() in VALID_EXTS


def parse_filename(filename: str):
    """Dosya adından source_class, layer_type ve original_id bilgilerini çıkarır.

    Beklenen format: <source_class>_<layer_type>_<original_id>.png
    Örnek: active_deep_BH-OCTA-HD-01.png → ('active', 'deep', 'BH-OCTA-HD-01')
    """
    stem = Path(filename).stem  # Uzantıyı çıkar

    # İlk parça: source_class (active, inactive, control)
    # İkinci parça: layer_type (deep, sup)
    # Geri kalan: original_id (BH-OCTA-HD-xx)
    parts = stem.split("_", 2)  # Maksimum 2 kez böl → 3 parça

    if len(parts) < 3:
        raise ValueError(f"Beklenmeyen dosya isim formatı: {filename}")

    source_class = parts[0]        # active / inactive / control
    layer_type = parts[1]          # deep / sup
    original_id = parts[2]         # BH-OCTA-HD-xx

    # Doğrulama
    if source_class not in ("active", "inactive", "control"):
        raise ValueError(f"Geçersiz source_class: '{source_class}' (dosya: {filename})")
    if layer_type not in ("deep", "sup"):
        raise ValueError(f"Geçersiz layer_type: '{layer_type}' (dosya: {filename})")

    return source_class, layer_type, original_id


# --- Uveitis klasöründeki görüntüleri tara (Pozitif sınıf, binary_label=1) ---
u_count = 0
for fp in sorted(uveitis_dir.iterdir()):
    if is_valid_image(fp):
        source_class, layer_type, original_id = parse_filename(fp.name)

        # Güvenlik: uveitis klasöründe sadece active veya inactive olmalı
        if source_class not in ("active", "inactive"):
            raise ValueError(f"uveitis klasöründe beklenmeyen source_class: {fp.name}")

        group_id = f"{source_class}_{original_id}"

        rows.append({
            "image_id": f"u_{u_count:04d}",
            "filepath": str(fp.as_posix()),
            "source_class": source_class,
            "layer_type": layer_type,
            "original_id": original_id,
            "group_id": group_id,
            "binary_label": 1,
        })
        u_count += 1

# --- Non-uveitis klasöründeki görüntüleri tara (Negatif sınıf, binary_label=0) ---
n_count = 0
for fp in sorted(non_uveitis_dir.iterdir()):
    if is_valid_image(fp):
        source_class, layer_type, original_id = parse_filename(fp.name)

        # Güvenlik: non_uveitis klasöründe sadece control olmalı
        if source_class != "control":
            raise ValueError(f"non_uveitis klasöründe beklenmeyen source_class: {fp.name}")

        group_id = f"{source_class}_{original_id}"

        rows.append({
            "image_id": f"n_{n_count:04d}",
            "filepath": str(fp.as_posix()),
            "source_class": source_class,
            "layer_type": layer_type,
            "original_id": original_id,
            "group_id": group_id,
            "binary_label": 0,
        })
        n_count += 1


# CSV olarak kaydet
df = pd.DataFrame(rows)
output_csv = PROJECT_ROOT / "metadata/octa_labels.csv"
df.to_csv(output_csv, index=False)

# Özet bilgiler
print(f"Toplam görüntü: {len(df)}")
print(f"\nBinary label dağılımı:")
print(df["binary_label"].value_counts())
print(f"\nSource class dağılımı:")
print(df["source_class"].value_counts())
print(f"\nLayer type dağılımı:")
print(df["layer_type"].value_counts())
print(f"\nBenzersiz group_id sayısı: {df['group_id'].nunique()}")
print(f"\nİlk 5 satır:")
print(df.head())
print(f"\nKaydedildi: {output_csv}")


if __name__ == "__main__":
    pass  # Script doğrudan modül seviyesinde çalışır
