import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Paths
DATA_DIR = Path("/Users/recepozturk/Desktop/uveitis_project/Veri Setleri/Ön segment OCT (AS-OCT)/MCOA_ Dataset_-_asoct/Images")
OUTPUT_CSV = Path("/Users/recepozturk/Desktop/uveitis_project/metadata/mcoa_split.csv")

def main():
    print("MCOA Veri Seti Taranıyor...")
    
    data = []
    # 1. Normal Cornea
    normal_dir = DATA_DIR / "Normal Cornea" / "AS-OCT"
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        for img_path in normal_dir.glob(ext):
            data.append({
                "image_id": img_path.stem,
                "filepath": str(img_path),
                "label_str": "Normal",
                "label": 0
            })
            
    # 2. Opaque Cornea
    opaque_dir = DATA_DIR / "Opaque Cornea" / "AS-OCT" / "AS-OCT Original Images"
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        for img_path in opaque_dir.glob(ext):
            data.append({
                "image_id": img_path.stem,
                "filepath": str(img_path),
                "label_str": "Opaque",
                "label": 1
            })
            
    df = pd.DataFrame(data)
    print(f"Toplam Görüntü: {len(df)}")
    print(df['label_str'].value_counts())
    
    if len(df) == 0:
        print("Hata: Hiç görüntü bulunamadı!")
        return

    # Stratified Split (80/10/10)
    train_df, temp_df = train_test_split(df, test_size=0.20, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42)
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    final_df = pd.concat([train_df, val_df, test_df])
    
    # Save
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print("\nSplit Özeti:")
    print(final_df.groupby(['split', 'label_str']).size())
    print(f"\nMetadata başarıyla kaydedildi: {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
