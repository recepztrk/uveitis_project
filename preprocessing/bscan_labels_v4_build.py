# ==============================================================================
# bscan_labels_v4_build.py
#
# B-scan V4 modeli için metadata (labels) CSV dosyasını oluşturur.
# 
# Kaynaklar:
# 1. data_work/bscan_oct_clean/non_uveitis (Gerçek Sağlıklı B-scan'ler)
# 2. data_work/bscan_oct_synthetic_v4/uveitis (Gerçek + Sentetik Üveit B-scan'leri)
#
# Çıktı: metadata/bscan_oct_labels_v4.csv
# ==============================================================================

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NON_UVEITIS_DIR = PROJECT_ROOT / "data_work/bscan_oct_clean/non_uveitis"
UVEITIS_DIR = PROJECT_ROOT / "data_work/bscan_oct_synthetic_v4/uveitis"
OUTPUT_CSV = PROJECT_ROOT / "metadata/bscan_oct_labels_v4.csv"

def main():
    print("B-scan V4 metadata (labels) oluşturuluyor...")
    
    data = []
    
    # 1. Non-Uveitis (Normal) Sınıfı - Tamamı Gerçek
    if NON_UVEITIS_DIR.exists():
        non_uv_files = list(NON_UVEITIS_DIR.glob("*.jpg")) + list(NON_UVEITIS_DIR.glob("*.tif")) + list(NON_UVEITIS_DIR.glob("*.jpeg"))
        for f in tqdm(non_uv_files, desc="Processing Non-Uveitis"):
            if f.is_file() and not f.name.startswith("."):
                data.append({
                    "image_id": f.stem,
                    "filepath": str(f.relative_to(PROJECT_ROOT)),
                    "source_class": "normal",
                    "binary_label": 0,
                    "is_synthetic": False
                })
    
    # 2. Uveitis Sınıfı - Gerçek ve Sentetik Karışık
    if UVEITIS_DIR.exists():
        uv_files = list(UVEITIS_DIR.glob("*.jpg"))
        for f in tqdm(uv_files, desc="Processing Uveitis"):
            if f.is_file() and not f.name.startswith("."):
                # Sentetik veri bayrağını dosya isminden anla
                is_synth = not f.name.startswith("real_")
                data.append({
                    "image_id": f.stem,
                    "filepath": str(f.relative_to(PROJECT_ROOT)),
                    "source_class": "uveitis",
                    "binary_label": 1,
                    "is_synthetic": is_synth
                })

    df = pd.DataFrame(data)
    
    # Çıktı klasörünü kontrol et
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # Kaydet
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n=== METADATA ÖZETİ ===")
    print(f"Toplam Görüntü: {len(df)}")
    print("\nSınıf Dağılımı (Binary Label):")
    print(df["binary_label"].value_counts())
    print("\nGerçek / Sentetik Dağılımı:")
    print(pd.crosstab(df["binary_label"], df["is_synthetic"], rownames=["Label (0:Normal, 1:Uveitis)"], colnames=["Is Synthetic"]))
    
    print(f"\n✅ Başarıyla kaydedildi: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
