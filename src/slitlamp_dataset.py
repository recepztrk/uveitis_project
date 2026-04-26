# ==============================================================================
# src/slitlamp_dataset.py
#
# Slit-lamp modalitesi için PyTorch Dataset sınıfı.
# Split CSV dosyasını okur, istenen split'e (train/val/test) göre filtreler
# ve her örnek için görüntü tensörü ile metadata bilgilerini döndürür.
#
# Döndürdüğü format: dict
#   {"image": tensor, "label": tensor, "image_id": str,
#    "raw_class": str, "filepath": str}
#
# Label float32 olarak döndürülür (BCEWithLogitsLoss uyumluluğu için).
# ==============================================================================

from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class SlitLampDataset(Dataset):
    """Slit-lamp ön segment fotoğrafları için PyTorch Dataset.

    Args:
        csv_path: Split bilgisini içeren CSV dosyasının yolu (slitlamp_split.csv)
        split: Hangi setin yükleneceği ("train", "val" veya "test")
        transform: Görüntüye uygulanacak torchvision dönüşümleri
    """

    def __init__(self, csv_path, split, transform=None):
        self.csv_path = Path(csv_path)
        self.split = split
        self.transform = transform

        # CSV'yi oku ve sadece istenen split'i filtrele
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"{split} split için veri bulunamadı.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = Path(row["filepath"])
        label = int(row["binary_label"])
        image_id = row["image_id"]
        raw_class = row["raw_class"]

        if not img_path.exists():
            raise FileNotFoundError(f"Görüntü bulunamadı: {img_path}")

        # Görüntüyü RGB formatında aç (grayscale veya RGBA olsa bile 3 kanala dönüştür)
        image = Image.open(img_path).convert("RGB")

        # Eğitim/değerlendirme dönüşümlerini uygula (resize, augmentation, normalize)
        if self.transform:
            image = self.transform(image)

        # Dict olarak döndür — eğitim ve değerlendirme scriptlerinde
        # batch["image"], batch["label"] şeklinde erişilir
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "image_id": image_id,
            "raw_class": raw_class,
            "filepath": str(img_path),
        }