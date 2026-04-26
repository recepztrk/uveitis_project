# ==============================================================================
# src/bscan_oct_dataset.py
#
# B-scan OCT modalitesi için PyTorch Dataset sınıfı.
# Split CSV dosyasını okur, istenen split'e göre veriyi filtreler ve
# (image, label) formatında return eder.
#
# ÖNEMLİ BİLİNEN TUTARSIZLIK (KNOWN ISSUE):
# - Slit-lamp ve OCTA dataset sınıfları dict döndürürken ( {"image": ..., "label": ...} ),
#   bu sınıf tuple döndürmektedir ( image, label ).
# - İleride tüm veri hatlarını standartlaştırmak için bu sınıfın da dict
#   döndürecek şekilde refactor edilmesi önerilir.
# ==============================================================================

from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class BScanOCTDataset(Dataset):
    """B-scan OCT görüntüleri için PyTorch Dataset.

    Args:
        csv_path: Split bilgisini içeren CSV dosyasının yolu (bscan_oct_split.csv)
        split: Hangi setin yükleneceği ("train", "val" veya "test")
        transform: Görüntüye uygulanacak torchvision dönüşümleri
    """

    def __init__(self, csv_path, split, transform=None):
        self.csv_path = Path(csv_path)
        self.split = split
        self.transform = transform

        # CSV'yi oku ve istenen split'e göre filtrele
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"{split} split için veri bulunamadı.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = Path(row["filepath"])
        # Label'ı float olarak alıyoruz (BCEWithLogitsLoss için)
        label = float(row["binary_label"])

        if not img_path.exists():
            raise FileNotFoundError(f"Görüntü bulunamadı: {img_path}")

        # Görüntüyü yükle ve RGB formatına dönüştür (grayscale olsalar bile)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # NOT: Dict yerine (image, label) tuple'ı döndürülüyor.
        # Bu eğitim scriptinde (train_bscan_oct_baseline.py)
        # for images, labels in loader: şeklinde bir paket açılımı (unpacking)
        # yapılmasını gerektirir.
        return image, torch.tensor(label, dtype=torch.float32)