# ==============================================================================
# train_cfp_baseline.py
#
# CFP (Color Fundus Photography) modalitesi için unimodal baseline model
# eğitim scripti. EfficientNet-B0 backbone üzerinde transfer learning ile
# "Uveitis vs Non-Uveitis" ikili sınıflandırma modeli eğitir.
#
# Eğitim stratejisi:
#   - Backbone: ImageNet pretrained EfficientNet-B0
#   - Progressive Fine-Tuning (5 epoch head → full fine-tuning)
#   - WeightedRandomSampler (ciddi sınıf dengesizliği: 1:12.5)
#   - pos_weight (BCEWithLogitsLoss içinde)
#   - Label Smoothing (0.1)
#   - Cosine Annealing LR
#   - Early Stopping (patience=10)
#   - AdamW optimizer + weight decay
#
# Veri seti:
#   870 CFP görüntüsü (63 uveitis, 807 non_uveitis)
#   Train: 609 | Val: 130 | Test: 131
#
# Çıktılar:
#   - outputs/models/cfp_efficientnetb0_best.pth    (model ağırlıkları)
#   - outputs/metrics/cfp_train_history.json         (epoch bazlı metrikler)
#   - outputs/metrics/cfp_test_metrics.json          (test seti sonuçları)
# ==============================================================================

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score
)
from tqdm import tqdm

from src.cfp_dataset import CFPDataset

# === YAPILANDIRMA ===
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

EPOCHS = 30
UNFREEZE_EPOCH = 5         # İlk 5 epoch sadece head eğitimi
BATCH_SIZE = 16
LABEL_SMOOTHING = 0.1      # 1 -> 0.95, 0 -> 0.05
PATIENCE = 10              # Early stopping: 10 epoch iyileşme yoksa dur
LR_HEAD = 1e-3             # Faz 1: Head eğitimi öğrenme oranı
LR_FULL = 1e-4             # Faz 2: Full fine-tuning öğrenme oranı
WEIGHT_DECAY = 1e-4        # L2 düzenlileştirme


def get_class_weights(dataset):
    """Eğitim setindeki sınıf dengesizliği oranını (pos_weight) hesaplar."""
    labels = dataset.df["binary_label"]
    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    pos_weight = neg / pos if pos > 0 else 1.0
    return torch.tensor([pos_weight], dtype=torch.float32)


def build_weighted_sampler(dataset):
    """Ciddi sınıf dengesizliğini dengelemek için WeightedRandomSampler oluşturur.

    Her epoch'ta pozitif ve negatif örneklerin yaklaşık eşit sıklıkta
    örneklenmesini sağlar. Bu, 1:12.5 gibi dengesiz veri setlerinde
    modelin azınlık sınıfını yeterince görmesini garantiler.
    """
    labels = dataset.df["binary_label"].values
    pos_count = (labels == 1).sum()
    neg_count = (labels == 0).sum()

    # Her örneğe sınıf ağırlığı ata
    weight_per_class = {0: 1.0 / neg_count, 1: 1.0 / pos_count}
    sample_weights = [weight_per_class[int(lbl)] for lbl in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def print_data_summary(dataset, split_name):
    """Veri setinin kaynak ve sınıf dağılımını yazdırır."""
    df = dataset.df
    print(f"\n  [{split_name.upper()}] Toplam: {len(df)}")

    label_counts = df["binary_label"].value_counts().to_dict()
    print(f"    Sınıf: Uveitis={label_counts.get(1, 0)}, Non-uveitis={label_counts.get(0, 0)}")

    source_counts = df["source_dataset"].value_counts().to_dict()
    for src, count in source_counts.items():
        print(f"    Kaynak: {src} = {count}")


def train_epoch(model, loader, optimizer, criterion):
    """Tek bir epoch eğitimi gerçekleştirir."""
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(DEVICE)
        # Label smoothing: smoothed_y = y * (1 - e) + e / 2
        raw_labels = batch["label"].to(DEVICE).unsqueeze(1)
        smooth_labels = raw_labels * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, smooth_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        all_labels.extend(batch["label"].numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    return epoch_loss, epoch_f1


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Modelin validasyon/test performansını metriklerle ölçer."""
    model.eval()
    running_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    for batch in loader:
        images = batch["image"].to(DEVICE)
        raw_labels = batch["label"].to(DEVICE).unsqueeze(1)

        logits = model(images)
        loss = criterion(logits, raw_labels)  # Validation'da smoothing yok

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        all_labels.extend(batch["label"].numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    epoch_acc = accuracy_score(all_labels, all_preds)

    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_auc = 0.0

    epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, zero_division=0)

    return epoch_loss, epoch_f1, epoch_acc, epoch_auc, epoch_precision, epoch_recall


def main():
    print("=" * 60)
    print("CFP BASELINE EĞİTİMİ — EfficientNet-B0")
    print("=" * 60)
    print(f"Cihaz: {DEVICE}")
    print(f"Epoch: {EPOCHS}, Batch: {BATCH_SIZE}, Patience: {PATIENCE}")
    print(f"Label Smoothing: {LABEL_SMOOTHING}")

    # 1. Klasörleri Hazırla
    models_dir = PROJECT_ROOT / "outputs/models"
    metrics_dir = PROJECT_ROOT / "outputs/metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_path = str(PROJECT_ROOT / "metadata/cfp_split.csv")

    # 2. Veri Setlerini Yükle
    train_dataset = CFPDataset(csv_path, split="train")
    val_dataset = CFPDataset(csv_path, split="val")
    test_dataset = CFPDataset(csv_path, split="test")

    print_data_summary(train_dataset, "train")
    print_data_summary(val_dataset, "val")
    print_data_summary(test_dataset, "test")

    # WeightedRandomSampler: Ciddi dengesizliği (1:12.5) dengelemek için
    sampler = build_weighted_sampler(train_dataset)

    # Sampler kullanılırken shuffle=False olmalı (sampler zaten karıştırır)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    pos_weight = get_class_weights(train_dataset).to(DEVICE)
    print(f"\npos_weight: {pos_weight.item():.4f}")

    # 3. Modeli Oluştur (EfficientNet-B0)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, 1)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 4. PROGRESSIVE FINE-TUNING STRATEJİSİ
    # Faz 1: Backbone dondur, sadece Classifier eğit
    print(f"\n{'─'*60}")
    print(f"FAZ 1: Head Fine-Tuning (Epoch 1-{UNFREEZE_EPOCH})")
    print(f"{'─'*60}")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.classifier.parameters(),
                            lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UNFREEZE_EPOCH)

    best_val_f1 = 0.0
    patience_counter = 0
    history = []
    best_model_path = str(models_dir / "cfp_efficientnetb0_best.pth")

    try:
        for epoch in range(1, EPOCHS + 1):

            # FAZ 2 GEÇİŞİ: Tüm ağırlıkları serbest bırak
            if epoch == UNFREEZE_EPOCH + 1:
                print(f"\n{'─'*60}")
                print(f"FAZ 2: Full Fine-Tuning (Epoch {UNFREEZE_EPOCH+1}-{EPOCHS})")
                print(f"{'─'*60}")
                for param in model.parameters():
                    param.requires_grad = True

                optimizer = optim.AdamW(model.parameters(),
                                        lr=LR_FULL, weight_decay=WEIGHT_DECAY)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS - UNFREEZE_EPOCH)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch}/{EPOCHS} (LR: {current_lr:.6f})")

            train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_f1, val_acc, val_auc, val_prec, val_rec = evaluate(
                model, val_loader, criterion)

            scheduler.step()

            print(f"  Train → Loss: {train_loss:.4f} | F1: {train_f1:.4f}")
            print(f"  Val   → Loss: {val_loss:.4f} | F1: {val_f1:.4f} | "
                  f"Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")
            print(f"          Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_f1": round(train_f1, 4),
                "val_loss": round(val_loss, 4),
                "val_f1": round(val_f1, 4),
                "val_acc": round(val_acc, 4),
                "val_auc": round(val_auc, 4),
                "val_precision": round(val_prec, 4),
                "val_recall": round(val_rec, 4),
                "lr": round(current_lr, 8),
            })

            # En iyi modeli seç (Val F1 öncelikli)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  🔥 YENİ EN İYİ MODEL! (Val F1: {best_val_f1:.4f})")
            else:
                patience_counter += 1
                print(f"  ⏳ İyileşme yok ({patience_counter}/{PATIENCE})")

            # Early stopping kontrolü (sadece Faz 2'de aktif)
            if epoch > UNFREEZE_EPOCH and patience_counter >= PATIENCE:
                print(f"\n⛔ Early stopping! {PATIENCE} epoch boyunca iyileşme olmadı.")
                break

    except KeyboardInterrupt:
        print("\n⚠️ Eğitim kullanıcı tarafından yarıda kesildi.")

    # 5. En iyi modelle TEST değerlendirmesi
    print(f"\n{'='*60}")
    print("TEST DEĞERLENDİRMESİ")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE,
                                     weights_only=True))
    test_loss, test_f1, test_acc, test_auc, test_prec, test_rec = evaluate(
        model, test_loader, criterion)

    test_metrics = {
        "test_metrics": {
            "accuracy": round(test_acc, 4),
            "precision": round(test_prec, 4),
            "recall": round(test_rec, 4),
            "f1": round(test_f1, 4),
            "roc_auc": round(test_auc, 4),
            "loss": round(test_loss, 4),
        },
        "best_val_f1": round(best_val_f1, 4),
        "model_path": best_model_path,
    }

    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    print(f"  ROC AUC:   {test_auc:.4f}")

    # 6. Sonuçları kaydet
    history_path = metrics_dir / "cfp_train_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    test_metrics_path = metrics_dir / "cfp_test_metrics.json"
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("EĞİTİM TAMAMLANDI!")
    print(f"  En iyi Val F1: {best_val_f1:.4f}")
    print(f"  Model: {best_model_path}")
    print(f"  Tarihçe: {history_path}")
    print(f"  Test metrikleri: {test_metrics_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
