# ==============================================================================
# slitlamp_split_build.py
#
# Slit-lamp etiket CSV dosyasını (slitlamp_labels.csv) okuyarak
# train / validation / test olarak %70 / %15 / %15 oranında böler.
#
# ÖNEMLİ:
# - Stratified split raw_class bazında yapılır (binary_label değil).
#   Böylece Uveitis, Cataract, Conjunctivitis ve Eyelid alt sınıflarının
#   oranı her üç sette de korunmuş olur.
# - random_state=42 ile tekrarlanabilirlik sağlanır.
#
# Çıktı: metadata/slitlamp_split.csv (split kolonu eklenmiş hali)
# ==============================================================================

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RANDOM_STATE = 42
LABELS_CSV = str(PROJECT_ROOT / "metadata/slitlamp_labels.csv")
OUTPUT_CSV = str(PROJECT_ROOT / "metadata/slitlamp_split.csv")


def main():
    df = pd.read_csv(LABELS_CSV).copy()

    # Boş CSV kontrolü
    if df.empty:
        raise ValueError("Labels CSV bos.")

    # Gerekli kolonların varlığını doğrula
    required_cols = {"image_id", "filepath", "raw_class", "binary_label"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Eksik kolonlar var: {missing_cols}")

    # İlk bölme: %70 train, %30 geçici (val + test)
    # stratify=raw_class ile her alt sınıfın oranı korunur
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["raw_class"],
    )

    # İkinci bölme: geçici seti %50-%50 ile val ve test'e ayır
    # Sonuç: toplam verinin %15'i val, %15'i test olur
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["raw_class"],
    )

    # SettingWithCopyWarning önlemek için açık kopya oluştur
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # Her sete ait split etiketini ata
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Üç seti birleştir ve kolon sırasını düzenle
    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    split_df = split_df[["image_id", "filepath", "raw_class", "binary_label", "split"]]

    split_df.to_csv(OUTPUT_CSV, index=False)

    # Özet bilgileri yazdır
    print("\n=== SLIT-LAMP SPLIT OZETI ===")
    print(f"Toplam goruntu: {len(split_df)}")

    print("\nSplit dagilimi:")
    print(split_df["split"].value_counts())

    print("\nSplit + binary_label dagilimi:")
    print(pd.crosstab(split_df["split"], split_df["binary_label"]))

    print("\nSplit + raw_class dagilimi:")
    print(pd.crosstab(split_df["split"], split_df["raw_class"]))

    print(f"\nKaydedildi: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
