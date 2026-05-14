# ==============================================================================
# train_bscan_v3_synthetic.py
#
# B-scan OCT V3 — Sentetik Veri ile Eğitim.
# V2 ile aynı strateji (Kermany pretrained backbone + fine-tuning) ama
# sentetik veri ile genişletilmiş dataset kullanır.
#
# VERİ:
#   Train: 528 (38 orijinal + 490 sentetik)
#   Val:   8 orijinal (değişmedi)
#   Test:  9 orijinal (değişmedi)
#
# ÇIKTILAR:
#   - outputs/models/bscan_v3_synthetic_best.pth
#   - outputs/metrics/bscan_v3_train_history.json
#   - outputs/metrics/bscan_v3_test_metrics.json
#
# KULLANIM:
#   python training/train_bscan_v3_synthetic.py
# ==============================================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import copy
import json
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

from src.bscan_oct_dataset import BScanOCTDataset

# =============================================================================
# Cihaz Ayarı
# =============================================================================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# =============================================================================
# Hiperparametreler
# =============================================================================
BATCH_SIZE = 16          # Daha fazla veri → daha büyük batch
NUM_EPOCHS = 50
HEAD_LR = 1e-3           # Head-only: yüksek LR
BACKBONE_LR = 5e-5       # Backbone: daha düşük LR (sentetik veri ile dikkatli)
HEAD_ONLY_EPOCHS = 8     # İlk 8 epoch head-only
PATIENCE = 15            # Early stopping sabır
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 0


# =============================================================================
# Veri Hazırlama
# =============================================================================
def build_dataloaders(csv_path: str):
    """Sentetik veri dahil DataLoader'lar oluşturur."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = BScanOCTDataset(csv_path=csv_path, split="train", transform=train_transform)
    val_dataset = BScanOCTDataset(csv_path=csv_path, split="val", transform=eval_transform)
    test_dataset = BScanOCTDataset(csv_path=csv_path, split="test", transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


# =============================================================================
# Model — Kermany Pretrained
# =============================================================================
def build_model(pretrained_path: str):
    """Kermany pretrained ResNet-18'den binary model oluşturur."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print(f"✅ Kermany ağırlıklar yüklendi: {pretrained_path}")

    model.fc = nn.Linear(model.fc.in_features, 1)
    print(f"🔄 Son katman binary'ye dönüştürüldü: 512 → 1")
    return model


# =============================================================================
# Progressive Fine-Tuning
# =============================================================================
def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True


# =============================================================================
# Eğitim (Label Smoothing ile)
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, smoothing=0.1):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)
        smoothed = labels * (1 - smoothing) + (1 - labels) * smoothing

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, smoothed)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


# =============================================================================
# Değerlendirme
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    for images, labels in tqdm(loader, desc="  Eval", leave=False):
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

    metrics = {
        "loss": running_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
    }
    return metrics


# =============================================================================
# Ana Fonksiyon
# =============================================================================
def main():
    csv_path = str(PROJECT_ROOT / "metadata/bscan_oct_split_v3.csv")
    pretrained_path = str(PROJECT_ROOT / "outputs/models/bscan_kermany_resnet18_pretrained.pth")
    output_model_dir = PROJECT_ROOT / "outputs/models"
    output_metric_dir = PROJECT_ROOT / "outputs/metrics"
    output_model_dir.mkdir(parents=True, exist_ok=True)
    output_metric_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("B-scan OCT V3 — Sentetik Veri ile Eğitim")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS} (head-only: {HEAD_ONLY_EPOCHS})")
    print(f"Head LR: {HEAD_LR} | Backbone LR: {BACKBONE_LR}")
    print()

    # --- Veri ---
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(csv_path)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # --- Model ---
    model = build_model(pretrained_path).to(DEVICE)

    # --- Loss ---
    df = pd.read_csv(csv_path)
    train_df = df[df["split"] == "train"]
    pos = (train_df["binary_label"] == 1).sum()
    neg = (train_df["binary_label"] == 0).sum()
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(DEVICE)
    print(f"⚖️  Pos weight: {pos_weight.item():.3f} ({neg} neg / {pos} pos)")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Optimizer (head-only) ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=HEAD_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HEAD_ONLY_EPOCHS)

    # --- Backbone dondur ---
    freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"🧊 Backbone donduruldu — Eğitilebilir: {trainable:,} / {total:,}")

    # --- Eğitim ---
    best_val_f1 = -1.0
    best_model_state = None
    patience_counter = 0
    history = []
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        if epoch == HEAD_ONLY_EPOCHS:
            unfreeze_backbone(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=BACKBONE_LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - HEAD_ONLY_EPOCHS
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n🔓 Backbone açıldı — Eğitilebilir: {trainable:,} / {total:,}")

        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, LABEL_SMOOTHING)
        val_metrics = evaluate(model, val_loader, criterion)
        scheduler.step()

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_f1": round(val_metrics["f1"], 4),
            "val_roc_auc": round(val_metrics["roc_auc"], 4),
            "lr": round(current_lr, 8),
        }
        history.append(epoch_result)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1']:.4f} "
              f"AUC: {val_metrics['roc_auc']:.4f} (LR: {current_lr:.6f})")

        val_score = val_metrics["f1"] if val_metrics["f1"] > 0 else val_metrics["roc_auc"] * 0.01
        if val_score > best_val_f1:
            best_val_f1 = val_score
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  ✅ Yeni en iyi! (F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['roc_auc']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⛔ Early stopping — {PATIENCE} epoch iyileşme yok.")
                break

    total_time = time.time() - start_time
    print(f"\n⏱️  Toplam süre: {total_time / 60:.1f} dakika")

    # --- Test ---
    print(f"\n{'='*60}")
    print("TEST DEĞERLENDİRMESİ (V3 — Sentetik Veri)")
    print(f"{'='*60}")

    model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_loader, criterion)

    print(f"\nTest Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1:        {test_metrics['f1']:.4f}")
    print(f"Test AUC:       {test_metrics['roc_auc']:.4f}")

    # --- Kaydet ---
    torch.save(best_model_state, output_model_dir / "bscan_v3_synthetic_best.pth")
    print(f"\n💾 Model: {output_model_dir / 'bscan_v3_synthetic_best.pth'}")

    with open(output_metric_dir / "bscan_v3_train_history.json", "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    with open(output_metric_dir / "bscan_v3_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    print("✅ V3 eğitimi tamamlandı!")


if __name__ == "__main__":
    main()
