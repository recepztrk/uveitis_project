# ==============================================================================
# src/kermany_dataset.py
#
# Kermany (CellData) B-scan OCT veri seti için PyTorch Dataset sınıfı.
# 109.309 retinal OCT görüntüsünü 4 sınıfa ayırır:
#   - CNV  (Koroidal Neovaskülarizasyon)
#   - DME  (Diyabetik Maküler Ödem)
#   - DRUSEN (Drusen Birikimi)
#   - NORMAL (Sağlıklı Retina)
#
# AMAÇ:
#   B-scan OCT modelinin "göz uzmanı" olmasını sağlamak için domain-specific
#   pre-training'de kullanılır. Bu veri setiyle eğitilen model, retinal katman
#   yapısını ve patolojik desenleri öğrenir; ardından üveit verisine fine-tune edilir.
#
# VERİ YAPISI:
#   new-datasets/CellData/OCT/
#   ├── train/
#   │   ├── CNV/      (37.205 görüntü)
#   │   ├── DME/      (11.348 görüntü)
#   │   ├── DRUSEN/   (8.616 görüntü)
#   │   └── NORMAL/   (51.140 görüntü)
#   └── test/
#       ├── CNV/      (250 görüntü)
#       ├── DME/      (250 görüntü)
#       ├── DRUSEN/   (250 görüntü)
#       └── NORMAL/   (250 görüntü)
#
# NOT:
#   - Orijinal veri setinde validation split yok.
#     Pre-training scriptinde train'den %10 stratified olarak ayrılır.
#   - Sınıf dengesizliği mevcut (NORMAL ~51K vs DRUSEN ~9K).
#     Eğitim sırasında class weights ile dengelenir.
# ==============================================================================

from pathlib import Path
from typing import Optional, Callable, List, Tuple
import os

from PIL import Image
import torch
from torch.utils.data import Dataset


# Sınıf isimlerini indeks ile eşleştirme (alfabetik sıralı)
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Desteklenen görüntü uzantıları
_VALID_EXTENSIONS = {".jpeg", ".jpg", ".png", ".tif", ".bmp"}


class KermanyOCTDataset(Dataset):
    """Kermany (CellData) B-scan OCT veri seti için PyTorch Dataset.

    Klasör yapısından (ImageFolder benzeri) görüntüleri yükler ve
    4 sınıflı etiketler üretir.

    Args:
        root_dir: Veri setinin kök dizini (örn: new-datasets/CellData/OCT)
        split: "train" veya "test"
        transform: Görüntülere uygulanacak torchvision dönüşümleri
        indices: Alt küme kullanmak için indeks listesi
                 (train/val ayırımında kullanılır)

    Returns:
        (image, label) tuple'ı — label: 0-3 arası int (CrossEntropyLoss için)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        indices: Optional[List[int]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # Görüntü yollarını ve etiketlerini topla
        self.samples: List[Tuple[str, int]] = []
        split_dir = self.root_dir / split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split dizini bulunamadı: {split_dir}")

        # Her sınıf klasörünü tara — os.scandir ile hızlı tarama
        # (sorted(glob) 109K dosyada ~20 dk sürerken, scandir ~saniyeler)
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"⚠️  Sınıf dizini bulunamadı, atlanıyor: {class_dir}")
                continue

            label = CLASS_TO_IDX[class_name]

            # os.scandir: stat çağrısı yapmadan dosyaları hızlıca listeler
            for entry in os.scandir(str(class_dir)):
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in _VALID_EXTENSIONS:
                        self.samples.append((entry.path, label))

        if len(self.samples) == 0:
            raise ValueError(f"'{split}' split'inde hiç görüntü bulunamadı: {split_dir}")

        # İndeks alt kümesi (train/val ayırımı için)
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        # Sınıf dağılımını hesapla (loglama ve class weight için)
        self._class_counts = [0] * len(CLASS_NAMES)
        for _, label in self.samples:
            self._class_counts[label] += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Görüntüyü yükle — grayscale olsa bile RGB'ye çevir
        # (ResNet 3 kanallı girdi bekler)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    @property
    def class_counts(self) -> dict:
        """Her sınıftaki görüntü sayısını döndürür."""
        return {name: self._class_counts[idx] for idx, name in enumerate(CLASS_NAMES)}

    @property
    def class_weights(self) -> torch.Tensor:
        """Sınıf dengesizliğini telafi etmek için ters frekans ağırlıkları.
        CrossEntropyLoss(weight=...) parametresinde kullanılır.

        Formül: weight_i = total_samples / (num_classes * count_i)
        """
        total = len(self.samples)
        num_classes = len(CLASS_NAMES)
        weights = []
        for count in self._class_counts:
            if count > 0:
                weights.append(total / (num_classes * count))
            else:
                weights.append(0.0)
        return torch.tensor(weights, dtype=torch.float32)

    def __repr__(self) -> str:
        counts_str = ", ".join(f"{k}: {v}" for k, v in self.class_counts.items())
        return (
            f"KermanyOCTDataset(split={self.split}, "
            f"total={len(self.samples)}, {counts_str})"
        )
