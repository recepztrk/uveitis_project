# ==============================================================================
# bscan_labels_build.py
#
# B-scan OCT görüntüleri için etiket CSV dosyası oluşturur.
#
# Problem tanımı: "Üveit vs Normal"
#   - Pozitif sınıf (1): intermediate_posterior_uveitis, panuveitis
#   - Negatif sınıf (0): normal
#
# ÖNEMLİ DETAY:
# - Diğer modalitelerden farklı olarak image_id 0'dan değil 1'den başlar (bscan_0001).
#
# Çıktı: metadata/bscan_oct_labels.csv
# ==============================================================================

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Temizlenmiş veri yolu
DATA_DIR = PROJECT_ROOT / "data_work/bscan_oct_clean"
METADATA_DIR = PROJECT_ROOT / "metadata"

# Desteklenen görüntü formatları
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Klasör isimlerine karşılık gelen ikili etiketler
CLASS_CONFIG = {
    "uveitis": 1,
    "non_uveitis": 0,
}


def infer_source_class(folder_name: str, filename: str) -> str:
    """Klasör yapısına ve dosya ismine bakarak detaylı alt sınıfı (source_class) belirler."""
    name = filename.lower()

    if folder_name == "uveitis":
        # Üveit alt tipleri
        if name.startswith("intermediate_posterior_uveitis_"):
            return "intermediate_posterior_uveitis"
        if name.startswith("panuveitis_"):
            return "panuveitis"
        return "uveitis"

    if folder_name == "non_uveitis":
        # Kontrol grubu
        if name.startswith("normal_"):
            return "normal"
        return "non_uveitis"

    return "unknown"


def main():
    rows = []
    # NOT: B-scan için id sayacı 1'den başlıyor (diğerleri 0'dan başlamıştı)
    image_counter = 1

    # CLASS_CONFIG üzerinde dönerek uveitis ve non_uveitis klasörlerini sırayla tara
    for folder_name, binary_label in CLASS_CONFIG.items():
        folder_path = DATA_DIR / folder_name

        if not folder_path.exists():
            print(f"[UYARI] Klasör bulunamadı: {folder_path}")
            continue

        # Sadece geçerli resim dosyalarını al ve isme göre sırala
        files = sorted(
            [
                p for p in folder_path.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_EXTS
            ]
        )

        # Bulunan her dosya için metadata sözlüğü (dict) oluştur
        for fp in files:
            rows.append(
                {
                    "image_id": f"bscan_{image_counter:04d}",
                    "filepath": str(fp.as_posix()),
                    "source_class": infer_source_class(folder_name, fp.name),
                    "binary_label": binary_label,
                }
            )
            image_counter += 1

    if not rows:
        raise RuntimeError("Hiç görüntü bulunamadı. Klasör yapısını kontrol et.")

    # CSV verisini oluştur ve kaydet
    df = pd.DataFrame(rows)
    out_path = METADATA_DIR / "bscan_oct_labels.csv"
    df.to_csv(out_path, index=False)

    # Özet bilgileri yazdır
    print("\n=== B-SCAN LABELS ÖZETİ ===")
    print(f"Toplam görüntü: {len(df)}")
    print("\nSource class dağılımı:")
    print(df["source_class"].value_counts())
    print("\nBinary label dağılımı:")
    print(df["binary_label"].value_counts())
    print(f"\nKaydedildi: {out_path}")


if __name__ == "__main__":
    main()