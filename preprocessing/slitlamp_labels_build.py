# ==============================================================================
# slitlamp_labels_build.py
#
# Slit-lamp (ön segment fotoğrafı) görüntüleri için etiket CSV dosyası oluşturur.
# Temizlenmiş görüntülerin bulunduğu data_work/slitlamp_clean/ klasöründeki
# dosyaları tarar ve her görüntü için image_id, filepath, raw_class ve
# binary_label bilgilerini içeren metadata/slitlamp_labels.csv dosyasını üretir.
#
# ÖNEMLİ TASARIM KARARI:
# - "Normal" sınıfı bilinçli olarak dışarıda bırakılmıştır.
#   Normal görüntüler farklı kaynak/çözünürlükte olduğu için modelin
#   hastalık yerine veri kaynağını öğrenme riski vardı.
# - Negatif sınıf: Cataract + Conjunctivitis + Eyelid
# - Pozitif sınıf: Uveitis
# ==============================================================================

from pathlib import Path
import pandas as pd

# Proje kök dizini (bu script preprocessing/ altında)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Temizlenmiş slit-lamp verilerinin bulunduğu dizinler
base_dir = PROJECT_ROOT / "data_work/slitlamp_clean"
uveitis_dir = base_dir / "uveitis"
non_uveitis_dir = base_dir / "non_uveitis"

# Her bir görüntünün metadata bilgisini tutacak liste
rows = []

# Desteklenen görüntü formatları
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_valid_image(fp: Path):
    """Dosyanın geçerli bir görüntü olup olmadığını kontrol eder.
    Gizli dosyalar (. ile başlayanlar) ve desteklenmeyen formatlar elenir."""
    return fp.is_file() and not fp.name.startswith(".") and fp.suffix.lower() in VALID_EXTS


def infer_raw_class(filename: str) -> str:
    """Dosya adından orijinal sınıf etiketini çıkarır.
    Dosya isimleri 'slit_uveitis_001.jpg', 'slit_cataract_001.jpg' vb. formatındadır."""
    name = filename.lower()

    if "uveitis" in name:
        return "Uveitis"
    if "cataract" in name:
        return "Cataract"
    if "conjunctivitis" in name:
        return "Conjunctivitis"
    if "eyelid" in name:
        return "Eyelid"

    return "Unknown"


# --- Uveitis klasöründeki görüntüleri tara (Pozitif sınıf, binary_label=1) ---
u_count = 0
for fp in sorted(uveitis_dir.iterdir()):
    if is_valid_image(fp):
        raw_class = infer_raw_class(fp.name)

        # Güvenlik kontrolü: uveitis klasöründe sadece Uveitis etiketli dosyalar olmalı
        if raw_class != "Uveitis":
            raise ValueError(f"uveitis klasorunde beklenmeyen dosya: {fp.name}")

        rows.append({
            "image_id": f"u_{u_count:04d}",
            "filepath": str(fp.as_posix()),
            "raw_class": raw_class,
            "binary_label": 1,
        })
        u_count += 1

# --- Non-uveitis klasöründeki görüntüleri tara (Negatif sınıf, binary_label=0) ---
n_count = 0
for fp in sorted(non_uveitis_dir.iterdir()):
    if is_valid_image(fp):
        raw_class = infer_raw_class(fp.name)

        # Güvenlik kontrolü: sadece Cataract, Conjunctivitis veya Eyelid olmalı
        if raw_class not in ["Cataract", "Conjunctivitis", "Eyelid"]:
            raise ValueError(f"non_uveitis klasorunde beklenmeyen dosya: {fp.name}")

        rows.append({
            "image_id": f"n_{n_count:04d}",
            "filepath": str(fp.as_posix()),
            "raw_class": raw_class,
            "binary_label": 0,
        })
        n_count += 1


# Oluşturulan metadata'yı CSV olarak kaydet
df = pd.DataFrame(rows)
df.to_csv(PROJECT_ROOT / "metadata/slitlamp_labels.csv", index=False)

# Özet bilgileri yazdır
print("Toplam goruntu:", len(df))
print("\nBinary label dagilimi:")
print(df["binary_label"].value_counts())
print("\nRaw class dagilimi:")
print(df["raw_class"].value_counts())
print("\nIlk 5 satir:")
print(df.head())
