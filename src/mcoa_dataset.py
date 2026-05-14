import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class McoaClassificationDataset(Dataset):
    """
    MCOA Sınıflandırma Dataset Sınıfı (Normal vs Opaque).
    split_file: metadata/mcoa_split.csv
    """
    def __init__(self, split_file, split_name='train', img_size=(224, 224), transform=None):
        self.split_file = Path(split_file)
        self.split_name = split_name
        self.img_size = img_size
        
        # DataFrame'i oku ve split'e gore filtrele
        df = pd.read_csv(self.split_file)
        self.data = df[df['split'] == split_name].reset_index(drop=True)
        
        if transform is None:
            if split_name == 'train':
                self.transform = A.Compose([
                    A.Resize(height=img_size[0], width=img_size[1]),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(height=img_size[0], width=img_size[1]),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['filepath']
        label = row['label']
        
        # Görüntüyü oku
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Albumentations uygula
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        # Label'i float tensor'a cevir (BCEWithLogitsLoss icin)
        label_tensor = torch.tensor([label], dtype=torch.float32)
            
        return image, label_tensor

if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    SPLIT_CSV = PROJECT_ROOT / "metadata" / "mcoa_split.csv"
    
    ds = McoaClassificationDataset(SPLIT_CSV, split_name='train')
    print(f"Train verisi boyutu: {len(ds)}")
    if len(ds) > 0:
        img, lbl = ds[0]
        print(f"Image shape: {img.shape}, Label: {lbl}")
