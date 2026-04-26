# ==============================================================================
# octa_dataset_v2.py
#
# OCTA modalitesi için V2 (Performans İyileştirme) PyTorch Dataset yapısı.
# Aşırı öğrenmeyi (overfitting) engellemek için agresif veri artırma (augmentation)
# tekniklerini barındırır.
#
# Eklenen Agresif Transformlar (sadece train split'i için):
# - RandomHorizontalFlip, RandomVerticalFlip
# - RandomAffine (dönme ±15 derece, kaydırma ±%10, boyutlandırma %90-%110)
# - ColorJitter (parlaklık ve kontrast oynamaları)
# - RandomErasing (piksel bloklarını silme - tensör sonrası)
# ==============================================================================

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def get_octa_v2_transforms(split="train"):
    """Split tipine göre güçlü transform hattını döndürür."""
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            # Normalize sonrası RandomErasing uygulanır (tensör üzerinde)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
        ])
    else:
        # Val ve Test için standard transform (Augmentation yalanmaz)
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class OCTADatasetV2(Dataset):
    """OCTA v2 Dataset (Agresif Augmentation ile Güçlendirilmiş)."""

    def __init__(self, csv_path, split):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.split = split
        self.transform = get_octa_v2_transforms(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Görüntüyü yükle ve RGB'ye dönüştür
        image = Image.open(row["filepath"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["binary_label"])

        return {
            "image": image,
            "label": label,
            "image_id": row["image_id"],
            "source_class": row["source_class"],
            "layer_type": row["layer_type"],
            "group_id": row["group_id"],
            "filepath": row["filepath"],
        }
