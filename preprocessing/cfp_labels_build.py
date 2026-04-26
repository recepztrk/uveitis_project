# ==============================================================================
# cfp_labels_build.py
#
# CFP / renkli fundus görüntüleri için etiket CSV dosyası oluşturur.
#
# Temizlenmiş görüntülerin bulunduğu dizin:
#   data_work/cfp_clean/
#       ├── non_uveitis/
#       └── uveitis/
#
# Beklenen dosya isim formatı:
#   cfp_<source_dataset>_<target_short>_<source_class>_<sequence>.jpg
#
# Örnek:
#   cfp_cfp1000_uv_vkh_0001.jpg
#   cfp_rfmimd2_uv_crs_0001.jpg
#   cfp_rfmimd2_uv_rs_0001.jpg
#   cfp_cfp1000_nonuv_dr2_0001.jpg
#   cfp_rfmimd2_nonuv_wnl_0001.jpg
#
# Üretilen CSV:
#   metadata/cfp_labels.csv
#
# CSV sütunları:
#   image_id, filepath, source_dataset, source_class, binary_label
#
# Sınıf tanımları:
#   - Pozitif sınıf (1): uveitis
#       VKH, CRS, RS
#   - Negatif sınıf (0): non_uveitis
#       Normal/WNL + üveit dışı retinal hastalıklar
#
# NOT:
# CFP modalitesindeki "uveitis" sınıfı, genel klinik üveit tanısının tamamını
# değil; VKH, chorioretinitis ve retinitis üzerinden temsil edilen üveit ilişkili
# posterior segment inflamatuvar bulguları ifade eder.
# ==============================================================================

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data_work/cfp_clean"
METADATA_DIR = PROJECT_ROOT / "metadata"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CLASS_CONFIG = {
    "uveitis": 1,
    "non_uveitis": 0,
}

ALLOWED_SOURCE_DATASETS = {"cfp1000", "rfmimd2"}

ALLOWED_UVEITIS_CLASSES = {
    "vkh",
    "crs",
    "rs",
}

ALLOWED_NON_UVEITIS_CLASSES = {
    "normal",
    "wnl",
    "dr1",
    "dr2",
    "dr3",
    "brvo",
    "crvo",
    "rao",
    "rd",
    "cscr",
    "maculopathy",
    "erm",
    "mh",
    "pathological_myopia",
    "hypertensive_retinopathy",
    "retinitis_pigmentosa",
    "bietti",
    "peripheral_degeneration_break",
    "laser_spots",
}


def is_valid_image(fp: Path) -> bool:
    """Geçerli görüntü dosyasını kontrol eder."""
    return (
        fp.is_file()
        and not fp.name.startswith(".")
        and fp.suffix.lower() in VALID_EXTS
    )


def parse_cfp_filename(filename: str):
    """
    CFP dosya adından source_dataset, target_short, source_class ve sequence çıkarır.

    Beklenen format:
        cfp_<source_dataset>_<target_short>_<source_class>_<sequence>.jpg

    source_class içinde "_" olabilir.
    Bu yüzden ilk üç parça sabit, son parça sequence kabul edilir.
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 5:
        raise ValueError(f"Beklenmeyen CFP dosya isim formatı: {filename}")

    modality = parts[0]
    source_dataset = parts[1]
    target_short = parts[2]
    sequence = parts[-1]
    source_class = "_".join(parts[3:-1])

    if modality != "cfp":
        raise ValueError(f"Dosya adı 'cfp' ile başlamıyor: {filename}")

    if source_dataset not in ALLOWED_SOURCE_DATASETS:
        raise ValueError(
            f"Geçersiz source_dataset: {source_dataset} | Dosya: {filename}"
        )

    if target_short not in {"uv", "nonuv"}:
        raise ValueError(
            f"Geçersiz target_short: {target_short} | Dosya: {filename}"
        )

    if not sequence.isdigit():
        raise ValueError(f"Sequence numerik değil: {sequence} | Dosya: {filename}")

    return source_dataset, target_short, source_class, sequence


def validate_class(folder_name: str, target_short: str, source_class: str, filename: str):
    """Klasör, dosya hedef kodu ve source_class uyumunu kontrol eder."""

    if folder_name == "uveitis":
        if target_short != "uv":
            raise ValueError(f"uveitis klasöründe nonuv dosya var: {filename}")

        if source_class not in ALLOWED_UVEITIS_CLASSES:
            raise ValueError(
                f"uveitis klasöründe beklenmeyen source_class: "
                f"{source_class} | Dosya: {filename}"
            )

    elif folder_name == "non_uveitis":
        if target_short != "nonuv":
            raise ValueError(f"non_uveitis klasöründe uv dosya var: {filename}")

        if source_class not in ALLOWED_NON_UVEITIS_CLASSES:
            raise ValueError(
                f"non_uveitis klasöründe beklenmeyen source_class: "
                f"{source_class} | Dosya: {filename}"
            )

    else:
        raise ValueError(f"Beklenmeyen klasör adı: {folder_name}")


def main():
    rows = []

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"CFP veri klasörü bulunamadı: {DATA_DIR}")

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    for folder_name, binary_label in CLASS_CONFIG.items():
        folder_path = DATA_DIR / folder_name

        if not folder_path.exists():
            raise FileNotFoundError(f"Klasör bulunamadı: {folder_path}")

        files = sorted(
            [fp for fp in folder_path.iterdir() if is_valid_image(fp)],
            key=lambda x: x.name.lower()
        )

        for fp in files:
            source_dataset, target_short, source_class, sequence = parse_cfp_filename(fp.name)

            validate_class(
                folder_name=folder_name,
                target_short=target_short,
                source_class=source_class,
                filename=fp.name,
            )

            rows.append(
                {
                    "image_id": fp.stem,
                    "filepath": str(fp.as_posix()),
                    "source_dataset": source_dataset,
                    "source_class": source_class,
                    "binary_label": binary_label,
                }
            )

    if not rows:
        raise RuntimeError("Hiç CFP görüntüsü bulunamadı. Klasör yapısını kontrol et.")

    df = pd.DataFrame(rows)

    output_csv = METADATA_DIR / "cfp_labels.csv"
    df.to_csv(output_csv, index=False)

    print("\n=== CFP LABELS ÖZETİ ===")
    print(f"Toplam görüntü: {len(df)}")

    print("\nBinary label dağılımı:")
    print(df["binary_label"].value_counts())

    print("\nSource dataset dağılımı:")
    print(df["source_dataset"].value_counts())

    print("\nSource class dağılımı:")
    print(df["source_class"].value_counts())

    print("\nİlk 5 satır:")
    print(df.head())

    print(f"\nKaydedildi: {output_csv}")


if __name__ == "__main__":
    main()