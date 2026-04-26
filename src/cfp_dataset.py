# ==============================================================================
# src/cfp_dataset.py
#
# CFP (Color Fundus Photography) modalitesi için PyTorch Dataset sınıfı.
#
# Split CSV dosyasını okur, istenen split'e (train/val/test) göre filtreler
# ve her örnek için görüntü tensörü ile metadata bilgilerini döndürür.
#
# Augmentation stratejisi (sadece train):
#   - RandomHorizontalFlip (fundus görüntüsü simetrik)
#   - RandomRotation (±10°, klinik olarak makul)
#   - ColorJitter (hafif parlaklık/kontrast)
#   - RandomResizedCrop (hafif zoom)
#   NOT: VerticalFlip kullanılmıyor (fundus oryantasyonu bozulur)
#
# Val/Test:
#   - Resize(224) + Normalize (standart)
#
# Döndürdüğü format: dict
#   {"image": tensor, "label": tensor, "image_id": str,
#    "source_class": str, "source_dataset": str, "filepath": str}
# ==============================================================================

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_cfp_transforms(split="train"):
    """Split tipine göre CFP transform hattını döndürür.

    Train: Fundus görüntülerine uygun augmentation + normalize
    Val/Test: Sadece resize + normalize
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            # RandomErasing: küçük bölgeleri silerek overfitting'i azaltır
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


class CFPDataset(Dataset):
    """CFP renkli fundus fotoğrafları için PyTorch Dataset.

    Args:
        csv_path: Split bilgisini içeren CSV dosyasının yolu (cfp_split.csv)
        split: Hangi setin yükleneceği ("train", "val" veya "test")
        transform: Opsiyonel harici transform. None ise split'e uygun
                   varsayılan transform otomatik uygulanır.
    """

    def __init__(self, csv_path, split, transform=None):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.split = split

        if len(self.df) == 0:
            raise ValueError(f"{split} split için CFP verisi bulunamadı.")

        # Harici transform verilmezse, split'e uygun varsayılanı kullan
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_cfp_transforms(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["filepath"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["binary_label"])

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "image_id": row["image_id"],
            "source_class": row["source_class"],
            "source_dataset": row["source_dataset"],
            "filepath": row["filepath"],
        }
