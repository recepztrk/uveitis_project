# ==============================================================================
# cfp_split_build.py
#
# CFP etiket CSV dosyasını (cfp_labels.csv) okuyarak
# train / validation / test olarak %70 / %15 / %15 oranında böler.
#
# Öncelikli strateji:
#   - Stratified split source_class bazında yapılır.
#   - Böylece vkh, crs, rs, normal, wnl, dr2, brvo vb. alt sınıfların
#     oranı train/val/test içinde korunmaya çalışılır.
#
# Güvenlik stratejisi:
#   - Eğer source_class bazında stratified split hata verirse,
#     binary_label üzerinden stratified split yapılır.
#
# Çıktı:
#   metadata/cfp_split.csv
# ==============================================================================

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RANDOM_STATE = 42

LABELS_CSV = PROJECT_ROOT / "metadata/cfp_labels.csv"
OUTPUT_CSV = PROJECT_ROOT / "metadata/cfp_split.csv"


def make_split(df: pd.DataFrame, stratify_col: str):
    """
    %70 train, %15 val, %15 test split oluşturur.
    Verilen stratify_col üzerinden stratified split yapar.
    """

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df[stratify_col],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df[stratify_col],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    return split_df


def main():
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Labels CSV bulunamadı: {LABELS_CSV}")

    df = pd.read_csv(LABELS_CSV).copy()

    if df.empty:
        raise ValueError("Labels CSV boş.")

    required_cols = {
        "image_id",
        "filepath",
        "source_dataset",
        "source_class",
        "binary_label",
    }

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Eksik kolonlar var: {missing_cols}")

    try:
        split_df = make_split(df, stratify_col="source_class")
        used_stratify = "source_class"

    except ValueError as e:
        print("\n[UYARI] source_class bazlı stratified split başarısız oldu.")
        print(f"Hata: {e}")
        print("binary_label bazlı stratified split deneniyor...\n")

        split_df = make_split(df, stratify_col="binary_label")
        used_stratify = "binary_label"

    split_df = split_df[
        [
            "image_id",
            "filepath",
            "source_dataset",
            "source_class",
            "binary_label",
            "split",
        ]
    ]

    split_df.to_csv(OUTPUT_CSV, index=False)

    print("\n=== CFP SPLIT ÖZETİ ===")
    print(f"Toplam görüntü: {len(split_df)}")
    print(f"Stratify kolonu: {used_stratify}")

    print("\nSplit dağılımı:")
    print(split_df["split"].value_counts())

    print("\nSplit + binary_label dağılımı:")
    print(pd.crosstab(split_df["split"], split_df["binary_label"]))

    print("\nSplit + source_dataset dağılımı:")
    print(pd.crosstab(split_df["split"], split_df["source_dataset"]))

    print("\nSplit + source_class dağılımı:")
    print(pd.crosstab(split_df["split"], split_df["source_class"]))

    print(f"\nKaydedildi: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()