# ==============================================================================
# octa_split_build.py
#
# OCTA etiket CSV dosyasını kullanarak train/val/test split işlemi yapar.
# Split, GÖRÜNTÜ bazında değil GROUP_ID bazında yapılır.
# Böylece aynı hastanın deep ve sup görüntüleri aynı split'e düşer.
#
# Data leakage önleme stratejisi:
#   1. Benzersiz group_id'ler çıkarılır
#   2. source_class dağılımı korunarak stratified split yapılır
#   3. Split bilgisi görüntü satırlarına geri yazılır
#
# Çıktı: metadata/octa_split.csv
# ==============================================================================

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RANDOM_STATE = 42

df = pd.read_csv(PROJECT_ROOT / "metadata/octa_labels.csv").copy()

# Benzersiz grupları çıkar (her group_id tek bir satır olacak şekilde)
# Stratification için source_class kullanılır
group_df = df[["group_id", "source_class", "binary_label"]].drop_duplicates().reset_index(drop=True)

print(f"Toplam benzersiz grup: {len(group_df)}")
print(f"Source class dağılımı:\n{group_df['source_class'].value_counts()}\n")

# İlk bölme: train (%70) vs geçici (%30)
train_groups, temp_groups = train_test_split(
    group_df["group_id"],
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=group_df["source_class"],
)

# İkinci bölme: geçici seti val (%50) ve test (%50) olarak böl → her biri %15
temp_df = group_df[group_df["group_id"].isin(temp_groups)]
val_groups, test_groups = train_test_split(
    temp_df["group_id"],
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["source_class"],
)

# Group_id'leri set'lere dönüştür (hızlı arama için)
train_set = set(train_groups)
val_set = set(val_groups)
test_set = set(test_groups)


# Her görüntüye split bilgisi ata
def assign_split(group_id):
    if group_id in train_set:
        return "train"
    elif group_id in val_set:
        return "val"
    elif group_id in test_set:
        return "test"
    else:
        raise ValueError(f"Atanamamış group_id: {group_id}")


df["split"] = df["group_id"].apply(assign_split)

# Çıktı sütun sırası
split_df = df[[
    "image_id", "filepath", "source_class", "layer_type",
    "original_id", "group_id", "binary_label", "split"
]]

# CSV olarak kaydet
output_csv = PROJECT_ROOT / "metadata/octa_split.csv"
split_df.to_csv(output_csv, index=False)

# Özet bilgiler
print("Split dağılımı (görüntü bazında):")
print(split_df["split"].value_counts())
print()

# Her split içindeki sınıf dağılımını kontrol et
for split_name in ["train", "val", "test"]:
    subset = split_df[split_df["split"] == split_name]
    group_count = subset["group_id"].nunique()
    print(f"--- {split_name.upper()} ---")
    print(f"  Görüntü: {len(subset)}, Benzersiz grup: {group_count}")
    print(f"  Source class: {dict(subset['source_class'].value_counts())}")
    print(f"  Layer type:   {dict(subset['layer_type'].value_counts())}")
    print(f"  Binary label: {dict(subset['binary_label'].value_counts())}")
    print()

# Data leakage kontrolü: aynı group_id birden fazla split'te olmamalı
for gid in df["group_id"].unique():
    splits = df[df["group_id"] == gid]["split"].unique()
    if len(splits) > 1:
        raise ValueError(f"DATA LEAKAGE! group_id={gid} birden fazla split'te: {splits}")

print("Data leakage kontrolü: GEÇTI ✓")
print(f"\nKaydedildi: {output_csv}")
