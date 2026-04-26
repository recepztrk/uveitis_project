# ==============================================================================
# train_octa_v4_efficientnet.py
#
# OCTA modalitesi için V4 eğitim script'i.
# ResNet-18 yerine EfficientNet-B0 backbone kullanır.
# V3'teki tüm teknikler korunur (Progressive Fine-Tuning, Label Smoothing,
# Cosine Annealing, Early Stopping, Agresif Augmentation).
#
# Çıktılar:
#   - outputs/models/octa_v4_efficientnet_best.pth
#   - outputs/metrics/octa_v4_efficientnet_train_history.json
# ==============================================================================

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

from src.octa_dataset_v2 import OCTADatasetV2

# === YAPILANDIRMA ===
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

EPOCHS = 40
UNFREEZE_EPOCH = 5
BATCH_SIZE = 16
LABEL_SMOOTHING = 0.1
PATIENCE = 10
LR_HEAD = 1e-3
LR_FULL = 1e-4
WEIGHT_DECAY = 1e-4


def build_efficientnet():
    """EfficientNet-B0 model yapısını oluşturur.
    Son sınıflandırma katmanı ikili çıkış (1 nöron) için değiştirilir."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def get_class_weights(dataset):
    """Eğitim setindeki sınıf dengesizliği oranını (pos_weight) hesaplar."""
    labels = dataset.df["binary_label"]
    pos = sum(labels == 1)
    neg = sum(labels == 0)
    pos_weight = neg / pos if pos > 0 else 1.0
    return torch.tensor([pos_weight], dtype=torch.float32)


def print_data_summary(dataset, split_name):
    """Veri setinin kaynak ve sınıf dağılımını yazdırır."""
    df = dataset.df
    print(f"\n  [{split_name.upper()}] Toplam: {len(df)}")
    label_counts = df["binary_label"].value_counts().to_dict()
    print(f"    Sınıf: Uveitis={label_counts.get(1, 0)}, Control={label_counts.get(0, 0)}")

    def get_source(filepath):
        return "OptoVue (UT)" if "UT-OCTA" in str(filepath) else "Heidelberg (BH)"

    df_copy = df.copy()
    df_copy["device_source"] = df_copy["filepath"].apply(get_source)
    for src, count in df_copy["device_source"].value_counts().to_dict().items():
        print(f"    Cihaz: {src} = {count}")


def train_epoch(model, loader, optimizer, criterion):
    """Tek bir epoch eğitimi gerçekleştirir."""
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(DEVICE)
        raw_labels = batch["label"].float().to(DEVICE).unsqueeze(1)
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
    """Modelin validasyon performansını ölçer."""
    model.eval()
    running_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    for batch in loader:
        images = batch["image"].to(DEVICE)
        raw_labels = batch["label"].float().to(DEVICE).unsqueeze(1)
        logits = model(images)
        loss = criterion(logits, raw_labels)
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
    epoch_prec = precision_score(all_labels, all_preds, zero_division=0)
    epoch_rec = recall_score(all_labels, all_preds, zero_division=0)

    return epoch_loss, epoch_f1, epoch_acc, epoch_auc, epoch_prec, epoch_rec


def main():
    print("=" * 60)
    print("OCTA V4 — EfficientNet-B0 (525 görüntü)")
    print("=" * 60)
    print(f"Cihaz: {DEVICE}")

    models_dir = PROJECT_ROOT / "outputs/models"
    metrics_dir = PROJECT_ROOT / "outputs/metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_path = PROJECT_ROOT / "metadata/octa_split.csv"

    train_dataset = OCTADatasetV2(str(csv_path), split="train")
    val_dataset = OCTADatasetV2(str(csv_path), split="val")
    print_data_summary(train_dataset, "train")
    print_data_summary(val_dataset, "val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    pos_weight = get_class_weights(train_dataset).to(DEVICE)
    print(f"\npos_weight: {pos_weight.item():.4f}")

    # EfficientNet-B0 modeli
    model = build_efficientnet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Faz 1: Backbone dondur, sadece classifier eğit
    print(f"\n{'─'*60}")
    print(f"FAZ 1: Head Fine-Tuning (Epoch 1-{UNFREEZE_EPOCH})")
    print(f"{'─'*60}")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.classifier.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UNFREEZE_EPOCH)

    best_val_f1 = 0.0
    patience_counter = 0
    history = []
    best_model_path = str(models_dir / "octa_v4_efficientnet_best.pth")

    try:
        for epoch in range(1, EPOCHS + 1):
            if epoch == UNFREEZE_EPOCH + 1:
                print(f"\n{'─'*60}")
                print(f"FAZ 2: Full Fine-Tuning (Epoch {UNFREEZE_EPOCH+1}-{EPOCHS})")
                print(f"{'─'*60}")
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=WEIGHT_DECAY)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - UNFREEZE_EPOCH)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch}/{EPOCHS} (LR: {current_lr:.6f})")

            train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_f1, val_acc, val_auc, val_prec, val_rec = evaluate(model, val_loader, criterion)
            scheduler.step()

            print(f"  Train → Loss: {train_loss:.4f} | F1: {train_f1:.4f}")
            print(f"  Val   → Loss: {val_loss:.4f} | F1: {val_f1:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")
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

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  🔥 YENİ EN İYİ MODEL! (Val F1: {best_val_f1:.4f})")
            else:
                patience_counter += 1
                print(f"  ⏳ İyileşme yok ({patience_counter}/{PATIENCE})")

            if epoch > UNFREEZE_EPOCH and patience_counter >= PATIENCE:
                print(f"\n⛔ Early stopping! {PATIENCE} epoch boyunca iyileşme olmadı.")
                break

    except KeyboardInterrupt:
        print("\n⚠️ Eğitim kullanıcı tarafından yarıda kesildi.")

    history_path = metrics_dir / "octa_v4_efficientnet_train_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"EĞİTİM TAMAMLANDI!")
    print(f"  Backbone: EfficientNet-B0")
    print(f"  En iyi Val F1: {best_val_f1:.4f}")
    print(f"  Model: {best_model_path}")
    print(f"  Tarihçe: {history_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
