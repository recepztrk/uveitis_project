# ==============================================================================
# bscan_split_build.py
#
# B-scan OCT etiket CSV dosyasını (bscan_oct_labels.csv) okuyarak
# train / validation / test olarak %70 / %15 / %15 oranında böler.
#
# ÖNEMLİ (Slit-lamp ve OCTA'dan farkı):
# - Stratified split burada binary_label (0 veya 1) üzerinden yapılmıştır.
# - Neden source_class (panuveitis vb.) üzerinden yapılmadı?
#   Çünkü B-scan veri setinde alt sınıf dağılımı çok dengesiz olabilir ve
#   bazı alt sınıflarda (ör. belirli bir üveit tipi) çok az örnek bulunabilir.
#   Bu durumda stratify=source_class hata verebileceği için
#   ana sınıflar (binary_label) üzerinden dengeli bölme tercih edilmiştir.
#
# Çıktı: metadata/bscan_oct_split.csv
# ==============================================================================

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RANDOM_STATE = 42

LABELS_CSV = str(PROJECT_ROOT / "metadata/bscan_oct_labels.csv")
OUTPUT_CSV = str(PROJECT_ROOT / "metadata/bscan_oct_split.csv")


def main():
    df = pd.read_csv(LABELS_CSV)

    if df.empty:
        raise ValueError("Labels CSV boş.")

    if "binary_label" not in df.columns:
        raise ValueError("binary_label kolonu bulunamadı.")

    # İlk bölme: %70 train, %30 geçici
    # stratify=binary_label ile sadece üveit/normal oranı korunur
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["binary_label"]
    )

    # İkinci bölme: geçici grubu %50-%50 ile val ve test'e ayır
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["binary_label"]
    )

    # SettingWithCopyWarning önlemek için açık kopya alındı
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # Split etiketlerini ata
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # DF'leri birleştir ve kolonları sırala
    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    split_df = split_df[["image_id", "filepath", "source_class", "binary_label", "split"]]

    # Kaydet
    split_df.to_csv(OUTPUT_CSV, index=False)

    print("\n=== B-SCAN SPLIT ÖZETİ ===")
    print(f"Toplam görüntü: {len(split_df)}")
    print("\nSplit dağılımı:")
    print(split_df["split"].value_counts())

    print("\nSplit + binary_label dağılımı:")
    print(pd.crosstab(split_df["split"], split_df["binary_label"]))

    print("\nSplit + source_class dağılımı:")
    print(pd.crosstab(split_df["split"], split_df["source_class"]))

    print(f"\nKaydedildi: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()