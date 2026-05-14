import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import signal
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch import ToTensorV2

def handler(signum, frame):
    raise TimeoutError("iCloud timeout")
signal.signal(signal.SIGALRM, handler)

class AsoctSegmentationDataset(Dataset):
    """
    AS-OCT Segmentasyon Dataset Sinifi.
    Beklenen Klasor Yapisi:
    - data_dir/Original_AS-OCT_Images/ (Görüntüler)
    - mask_dir/ (PNG maskeleri)
    """
    def __init__(self, image_dir, mask_dir, img_size=(256, 256), transform=None, is_train=True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.is_train = is_train
        
        # Valid image files
        self.images = sorted(list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.bmp")))
        
        # Default albumentations transform if none provided
        if transform is None:
            if is_train:
                self.transform = A.Compose([
                    A.Resize(height=img_size[0], width=img_size[1]),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(height=img_size[0], width=img_size[1]),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    ToTensorV2()
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        
        # Resim yukleme (Timeout ile)
        try:
            signal.alarm(2)
            image = cv2.imread(str(img_path))
            signal.alarm(0)
            if image is None:
                raise ValueError("None image")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            signal.alarm(0)
            image = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        
        # Maske yukleme (Timeout ile)
        if mask_path.exists():
            try:
                signal.alarm(2)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                signal.alarm(0)
                if mask is None:
                    raise ValueError("None mask")
            except Exception:
                signal.alarm(0)
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            # Eger maske yoksa (test sirasinda) veya hata almissa bos maske dondur
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
        # Augmentasyonlar
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Maskeyi LongTensor yap
        mask = mask.long()
            
        return image, mask

if __name__ == "__main__":
    # Test dataset class
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    AIDK_DIR = PROJECT_ROOT / "Veri Setleri" / "Ön segment OCT (AS-OCT)" / "AIDK_Dataset_-_asoct" / "Full-frame_Dataset"
    
    img_dir = AIDK_DIR / "Original_AS-OCT_Images"
    mask_dir = PROJECT_ROOT / "data_raw" / "asoct" / "masks"
    
    dataset = AsoctSegmentationDataset(img_dir, mask_dir, is_train=True)
    print(f"Dataset eleman sayisi: {len(dataset)}")
    if len(dataset) > 0:
        img, mask = dataset[0]
        print(f"Resim shape: {img.shape}, dtype: {img.dtype}")
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, unique vals: {torch.unique(mask)}")
