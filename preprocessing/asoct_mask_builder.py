import os
import glob
import json
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# Proje Kök Dizini ve Veri Yolları
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AIDK_DIR = PROJECT_ROOT / "Veri Setleri" / "Ön segment OCT (AS-OCT)" / "AIDK_Dataset_-_asoct" / "Full-frame_Dataset"
JSON_DIR = AIDK_DIR / "Experts_Annotations"
IMG_DIR = AIDK_DIR / "Original_AS-OCT_Images"

# Çıktı Dizini
OUTPUT_MASK_DIR = PROJECT_ROOT / "data_raw" / "asoct" / "masks"
OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

# Sınıf-Renk (Grayscale) Haritası - Segmentasyon Maskesi İçin
CLASS_MAP = {
    "_background_": 0,
    "Cornea": 1,
    "Iris": 2,
    "Lesion": 3,
    "Anterior_Chamber": 4, # Bulunursa diye
    "Lens": 5
}

def create_mask_from_json(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    img_h = data.get('imageHeight')
    img_w = data.get('imageWidth')
    
    # Maske başlangıçta arka plan (0)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    # Poligonları çiz (öncelik sırasına göre veya sıralı - Genelde Iris, sonra Cornea vs.)
    for shape in data.get('shapes', []):
        label = shape['label']
        points = np.array(shape['points'], dtype=np.int32)
        
        # Eğer etiket haritada yoksa dinamik olarak ekle (Yeni bir sınıfsı)
        if label not in CLASS_MAP:
            CLASS_MAP[label] = len(CLASS_MAP)
            
        class_idx = CLASS_MAP[label]
        
        # Maske üzerine poligonu doldur
        cv2.fillPoly(mask, [points], class_idx)
        
    # Maskeyi PNG olarak kaydet
    cv2.imwrite(str(output_path), mask)
    return True

def main():
    print(f"Maske üretim dizini: {OUTPUT_MASK_DIR}")
    json_files = list(JSON_DIR.glob('*.json'))
    print(f"Toplam {len(json_files)} JSON dosyası bulundu.")
    
    success_count = 0
    for i, json_path in enumerate(json_files):
        # Orijinal resim adıyla eşleşen maske adı (örnek: 1.json -> 1.png)
        img_id = json_path.stem
        output_path = OUTPUT_MASK_DIR / f"{img_id}.png"
        
        try:
            create_mask_from_json(json_path, output_path)
            success_count += 1
            if i % 50 == 0 and i > 0:
                print(f"[{i}/{len(json_files)}] Maskeler üretiliyor...")
        except Exception as e:
            print(f"Hata ({json_path.name}): {e}")
            
    print(f"\nMaske üretimi tamamlandı! Toplam başarılı: {success_count}/{len(json_files)}")
    print("\nTespit Edilen Tüm Sınıflar ve ID'leri:")
    for k, v in CLASS_MAP.items():
        print(f" - {k}: {v}")
        
    # Sınıf haritasını da metadata olarak kaydedelim
    with open(OUTPUT_MASK_DIR / "class_map.json", "w") as f:
        json.dump(CLASS_MAP, f, indent=4)

if __name__ == "__main__":
    main()
