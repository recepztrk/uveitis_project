# ==============================================================================
# train_slitlamp_baseline.py
#
# Slit-lamp (ön segment fotoğrafı) modalitesi için unimodal baseline model
# eğitim scripti. EfficientNet-B0 backbone üzerinde transfer learning ile
# "Uveitis vs Non-Uveitis" ikili sınıflandırma modeli eğitir.
#
# Eğitim stratejisi:
#   - Backbone: ImageNet pretrained EfficientNet-B0
#   - Son sınıflandırma katmanı ikili çıkış için yeniden tanımlanır
#   - Sınıf dengesizliği pos_weight ile yönetilir
#   - En iyi model validation F1 skoruna göre seçilir
#
# Çıktılar:
#   - outputs/models/slitlamp_efficientnetb0_best.pth  (model ağırlıkları)
#   - outputs/metrics/slitlamp_train_history.json       (epoch bazlı metrikler)
#   - outputs/metrics/slitlamp_test_metrics.json        (test seti sonuçları)
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

from src.slitlamp_dataset import SlitLampDataset


# GPU varsa kullan, yoksa CPU üzerinde çalış
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(csv_path: str, batch_size: int = 16):
    """Train, validation ve test için DataLoader'ları oluşturur.

    Train seti için veri artırma (augmentation) uygulanır:
    - RandomHorizontalFlip: Yatay çevirme
    - RandomRotation: ±10° döndürme
    - ColorJitter: Hafif parlaklık/kontrast değişimi
    Val/Test setleri için sadece resize ve normalizasyon yapılır.
    """
    # Eğitim seti dönüşümleri — augmentation ile veri çeşitliliği artırılır
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet ortalaması
            std=[0.229, 0.224, 0.225]      # ImageNet standart sapması
        ),
    ])

    # Değerlendirme seti dönüşümleri — augmentation yok, sadece resize + normalize
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = SlitLampDataset(csv_path=csv_path, split="train", transform=train_transform)
    val_dataset = SlitLampDataset(csv_path=csv_path, split="val", transform=eval_transform)
    test_dataset = SlitLampDataset(csv_path=csv_path, split="test", transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def build_model():
    """ImageNet pretrained EfficientNet-B0 modelini yükler ve
    son sınıflandırma katmanını ikili çıkış (1 nöron) için değiştirir."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def compute_pos_weight(csv_path: str):
    """Sınıf dengesizliğini telafi etmek için pos_weight hesaplar.
    pos_weight = negatif_sayısı / pozitif_sayısı
    Bu değer BCEWithLogitsLoss'a verilerek azınlık sınıfına (Uveitis)
    daha yüksek kayıp ağırlığı atanır."""
    df = pd.read_csv(csv_path)
    train_df = df[df["split"] == "train"]

    pos_count = (train_df["binary_label"] == 1).sum()
    neg_count = (train_df["binary_label"] == 0).sum()

    pos_weight = neg_count / pos_count
    return torch.tensor([pos_weight], dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer):
    """Bir epoch boyunca modeli eğitir ve ortalama loss döndürür."""
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE).unsqueeze(1)  # [B] → [B,1] boyutuna çevir

        optimizer.zero_grad()
        logits = model(images)       # Ham çıkış (sigmoid uygulanmamış)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Model performansını değerlendirir ve tüm metrikleri hesaplar.
    Gradient hesaplanmaz (torch.no_grad) — sadece ileri yönlü geçiş yapılır."""
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE).unsqueeze(1)

        logits = model(images)
        loss = criterion(logits, labels)

        # Sigmoid ile olasılığa çevir, 0.5 eşik ile sınıflandır
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= 0.5).long()

        running_loss += loss.item() * images.size(0)

        all_labels.extend(labels.squeeze(1).cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = running_loss / len(loader.dataset)

    # Tüm değerlendirme metriklerini hesapla
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
    csv_path = str(PROJECT_ROOT / "metadata/slitlamp_split.csv")
    output_model_dir = PROJECT_ROOT / "outputs/models"
    output_metric_dir = PROJECT_ROOT / "outputs/metrics"
    output_model_dir.mkdir(parents=True, exist_ok=True)
    output_metric_dir.mkdir(parents=True, exist_ok=True)

    # Hiperparametreler
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-4

    # Veri yükleme
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_dataloaders(
        csv_path=csv_path,
        batch_size=batch_size
    )

    print("Train:", len(train_dataset))
    print("Val:", len(val_dataset))
    print("Test:", len(test_dataset))
    print("Device:", DEVICE)

    # Model, loss fonksiyonu ve optimizer tanımla
    model = build_model().to(DEVICE)

    pos_weight = compute_pos_weight(csv_path).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # En iyi modeli takip etmek için değişkenler
    best_val_f1 = -1.0
    best_model_state = None
    history = []

    # --- Eğitim döngüsü ---
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, criterion)

        # Epoch sonuçlarını kaydet
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

        # Validation F1 iyileştiyse modelin kopyasını sakla
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_state = copy.deepcopy(model.state_dict())

    # --- Eğitim sonrası: en iyi modelle test değerlendirmesi ---
    model.load_state_dict(best_model_state)

    test_metrics = evaluate(model, test_loader, criterion)
    print("\nBest model test metrics:")
    print(test_metrics)

    # Model ağırlıklarını kaydet
    model_path = output_model_dir / "slitlamp_efficientnetb0_best.pth"
    torch.save(model.state_dict(), model_path)

    # Eğitim geçmişi ve test metriklerini JSON olarak kaydet
    history_path = output_metric_dir / "slitlamp_train_history.json"
    test_metric_path = output_metric_dir / "slitlamp_test_metrics.json"

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(test_metric_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nModel saved to: {model_path}")
    print(f"History saved to: {history_path}")
    print(f"Test metrics saved to: {test_metric_path}")


if __name__ == "__main__":
    main()