import os
import shutil
import random
from pathlib import Path

# Paths
BASE_DIR = Path("/Users/recepozturk/Desktop/uveitis_project")
DEMO_DIR = BASE_DIR / "data_work" / "demo_images"
OUTPUT_DIR = BASE_DIR / "data_work" / "router_data"

MODALITIES = ["slitlamp", "octa", "cfp", "bscan_oct", "as_oct"]

def prepare_dataset():
    print(f"Preparing Router Dataset in {OUTPUT_DIR}...")
    
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ["train", "val"]:
        for mod in MODALITIES:
            (OUTPUT_DIR / split / mod).mkdir(parents=True, exist_ok=True)
            
    for mod in MODALITIES:
        mod_dir = DEMO_DIR / mod
        if not mod_dir.exists():
            print(f"[WARN] {mod_dir} bulunamadi!")
            continue
            
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
            images.extend(list(mod_dir.rglob(ext)))
            images.extend(list(mod_dir.rglob(ext.upper())))
            
        images = list(set(images))
        
        random.seed(42)
        random.shuffle(images)
        
        # Elimizdeki resimlerin %80 Train, %20 Val (Normalde tam 100 resim var -> 80 / 20)
        n_total = len(images)
        n_train = int(n_total * 0.8)
        
        train_images = images[:n_train]
        val_images = images[n_train:]
        
        for i, img_path in enumerate(train_images):
            ext = img_path.suffix.lower()
            shutil.copy(img_path, OUTPUT_DIR / "train" / mod / f"{mod}_{i}{ext}")
            
        for i, img_path in enumerate(val_images):
            ext = img_path.suffix.lower()
            shutil.copy(img_path, OUTPUT_DIR / "val" / mod / f"{mod}_{i}{ext}")
            
        print(f"[{mod}] Copied {len(train_images)} train, {len(val_images)} val images.")

if __name__ == "__main__":
    prepare_dataset()
