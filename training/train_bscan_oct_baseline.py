# ==============================================================================
# train_bscan_oct_baseline.py
#
# B-scan OCT modalitesi için unimodal baseline model eğitim scripti.
# ResNet-18 backbone üzerinde transfer learning ile "Üveit vs Normal"
# ikili sınıflandırma modeli eğitir.
#
# ÖNEMLİ (Veri Yapısı Farklılığı):
# - BScanOCTDataset dict yerine (image, label) tuple'ı döndürdüğü için, 
#   Eğitim (train_one_epoch) ve Doğrulama (evaluate) döngülerinde 
#   "for images, labels in loader:" şeklinde (tuple unpacking) kullanılmıştır.
#
# BİLİNEN SORUN (Overfitting):
# - B-scan veri setinde validation setinin çok küçük olması veya verideki 
#   genel sınırlılıklar nedeniyle, bu model ciddi bir overfitting yapmaya meyillidir 
#   (Train Loss 0'a yakınsarken Val Loss yükselmektedir).
#   İleride daha güçlü veri artırımı veya daha büyük bir veri seti gerekecektir.
#
# Çıktılar:
#   - outputs/models/bscan_oct_resnet18_best.pth  (model ağırlıkları)
#   - outputs/metrics/bscan_oct_train_history.json  (epoch bazlı metrikler)
#   - outputs/metrics/bscan_oct_test_metrics.json   (test seti sonuçları)
# ==============================================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import copy
import json

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.bscan_oct_dataset import BScanOCTDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(csv_path: str, batch_size: int = 16):
    """Train, validation ve test için DataLoader'ları oluşturur."""
    # Eğitim seti dönüşümleri — augmentation (veri artırma)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Değerlendirme (Validation/Test) transformu — artırma yok
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = BScanOCTDataset(csv_path=csv_path, split="train", transform=train_transform)
    val_dataset = BScanOCTDataset(csv_path=csv_path, split="val", transform=eval_transform)
    test_dataset = BScanOCTDataset(csv_path=csv_path, split="test", transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def build_model():
    """ImageNet pretrained ResNet-18 modelini yükler. 
    OCTA'da olduğu gibi veriseti boyutundan dolayı (ResNet-18) tercih edilmiştir."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


def compute_pos_weight(csv_path: str):
    """Sınıf dengesizliğini telafi etmek için (negatif sınıf / pozitif sınıf) oranını hesaplar."""
    df = pd.read_csv(csv_path)
    train_df = df[df["split"] == "train"]

    pos_count = (train_df["binary_label"] == 1).sum()
    neg_count = (train_df["binary_label"] == 0).sum()

    pos_weight = neg_count / pos_count
    return torch.tensor([pos_weight], dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer):
    """Bir epoch eğitim yapar, ortalama kaybı döndürür."""
    model.train()
    running_loss = 0.0

    # DİKKAT: BScan dataset sınıfı (images, labels) tuple döndürdüğü için unpacking yapıyoruz.
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Model performansını validation veya test setinde değerlendirir."""
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []

    # DİKKAT: Yine (images, labels) tuple unpacking kullanılıyor.
    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= 0.5).long()

        running_loss += loss.item() * images.size(0)

        all_labels.extend(labels.squeeze(1).cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = running_loss / len(loader.dataset)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_probs),
    }

    return metrics


def main():
    csv_path = str(PROJECT_ROOT / "metadata/bscan_oct_split.csv")
    output_model_dir = PROJECT_ROOT / "outputs/models"
    output_metric_dir = PROJECT_ROOT / "outputs/metrics"
    output_model_dir.mkdir(parents=True, exist_ok=True)
    output_metric_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 16
    num_epochs = 15
    learning_rate = 1e-4

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_dataloaders(
        csv_path=csv_path,
        batch_size=batch_size
    )

    print("Train:", len(train_dataset))
    print("Val:", len(val_dataset))
    print("Test:", len(test_dataset))
    print("Device:", DEVICE)

    model = build_model().to(DEVICE)

    pos_weight = compute_pos_weight(csv_path).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_f1 = -1.0
    best_model_state = None
    history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, criterion)

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        history.append(epoch_result)

        print(epoch_result)

        # En iyi validation F1 skoruna göre modeli seç
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_state = copy.deepcopy(model.state_dict())

    # Eğitimin sonunda test değerlendirmesi için en iyi F1 ağırlıklarına sahip modeli geri yükle
    model.load_state_dict(best_model_state)

    test_metrics = evaluate(model, test_loader, criterion)
    print("\nBest model test metrics:")
    print(test_metrics)

    # Model ve sonuç dosyası kayıtları
    model_path = output_model_dir / "bscan_oct_resnet18_best.pth"
    torch.save(model.state_dict(), model_path)

    history_path = output_metric_dir / "bscan_oct_train_history.json"
    test_metric_path = output_metric_dir / "bscan_oct_test_metrics.json"

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(test_metric_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nModel saved to: {model_path}")
    print(f"History saved to: {history_path}")
    print(f"Test metrics saved to: {test_metric_path}")


if __name__ == "__main__":
    main()