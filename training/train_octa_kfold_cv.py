# ==============================================================================
# train_octa_kfold_cv.py
#
# OCTA modalitesi için 5-Fold Cross Validation eğitim script'i.
# EfficientNet-B0 backbone ile tüm veriyi kullanarak güvenilir
# performans tahmini yapar.
#
# Önemli özellikler:
#   - GroupKFold: Aynı hastanın görüntüleri aynı fold'da kalır (data leakage önlemi)
#   - Her fold için ayrı model eğitilir
#   - Sonuçta 5 fold'un ortalaması ve standart sapması raporlanır
#
# Çıktılar:
#   - outputs/metrics/octa_kfold_results.json (fold bazlı + ortalama metrikler)
#   - outputs/models/octa_kfold_best_fold{i}.pth (her fold'un en iyi modeli)
# ==============================================================================

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
from PIL import Image

# === YAPILANDIRMA ===
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

N_FOLDS = 5
EPOCHS = 30
UNFREEZE_EPOCH = 5
BATCH_SIZE = 16
LABEL_SMOOTHING = 0.1
PATIENCE = 8
LR_HEAD = 1e-3
LR_FULL = 1e-4
WEIGHT_DECAY = 1e-4


# === DATASET ===
class OCTAKFoldDataset(torch.utils.data.Dataset):
    """K-Fold CV için OCTA Dataset sınıfı.
    Train/Val ayrımını dışarıdan index ile yapar."""

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

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
            "label": label,
            "image_id": row["image_id"],
            "group_id": row["group_id"],
            "filepath": row["filepath"],
        }


def get_transforms(is_train=True):
    """Train ve val/test için transform'ları döndürür."""
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def build_efficientnet():
    """EfficientNet-B0 modeli oluşturur."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def train_epoch(model, loader, optimizer, criterion):
    """Tek bir epoch eğitimi."""
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []

    for batch in tqdm(loader, desc="  Train", leave=False):
        images = batch["image"].to(DEVICE)
        raw_labels = batch["label"].float().to(DEVICE).unsqueeze(1)
        smooth_labels = raw_labels * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, smooth_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        all_labels.extend(batch["label"].numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    return running_loss / len(loader.dataset), f1_score(all_labels, all_preds, zero_division=0)


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Değerlendirme."""
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

    metrics = {
        "loss": running_loss / len(loader.dataset),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["roc_auc"] = 0.0

    return metrics


def train_one_fold(fold_idx, train_df, val_df, models_dir):
    """Tek bir fold için tam eğitim döngüsü."""
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
    print(f"  Train: {len(train_df)} görüntü, Val: {len(val_df)} görüntü")
    print(f"  Train sınıf: Uveitis={sum(train_df['binary_label']==1)}, Control={sum(train_df['binary_label']==0)}")
    print(f"  Val sınıf:   Uveitis={sum(val_df['binary_label']==1)}, Control={sum(val_df['binary_label']==0)}")
    print(f"{'='*60}")

    train_dataset = OCTAKFoldDataset(train_df, transform=get_transforms(is_train=True))
    val_dataset = OCTAKFoldDataset(val_df, transform=get_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # pos_weight hesapla
    pos = sum(train_df["binary_label"] == 1)
    neg = sum(train_df["binary_label"] == 0)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(DEVICE)

    model = build_efficientnet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Faz 1: Head only
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.classifier.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UNFREEZE_EPOCH)

    best_val_f1 = 0.0
    patience_counter = 0
    best_model_path = str(models_dir / f"octa_kfold_best_fold{fold_idx + 1}.pth")

    for epoch in range(1, EPOCHS + 1):
        if epoch == UNFREEZE_EPOCH + 1:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - UNFREEZE_EPOCH)

        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate(model, val_loader, criterion)
        scheduler.step()

        val_f1 = val_metrics["f1"]
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d} → Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_metrics['roc_auc']:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if epoch > UNFREEZE_EPOCH and patience_counter >= PATIENCE:
            print(f"  ⛔ Early stopping @ epoch {epoch}")
            break

    # En iyi modeli yükle ve val üzerinde son değerlendirme
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    final_metrics = evaluate(model, val_loader, criterion)
    final_metrics["best_val_f1"] = best_val_f1

    print(f"  Fold {fold_idx + 1} Sonuç → F1: {final_metrics['f1']:.4f} | AUC: {final_metrics['roc_auc']:.4f} | Acc: {final_metrics['accuracy']:.4f}")

    return final_metrics


def main():
    print("=" * 60)
    print(f"OCTA {N_FOLDS}-FOLD CROSS VALIDATION — EfficientNet-B0")
    print("=" * 60)
    print(f"Cihaz: {DEVICE}")

    models_dir = PROJECT_ROOT / "outputs/models"
    metrics_dir = PROJECT_ROOT / "outputs/metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Tüm veriyi yükle (split sütununu yoksay, tümünü kullan)
    csv_path = PROJECT_ROOT / "metadata/octa_labels.csv"
    df = pd.read_csv(csv_path)
    print(f"\nToplam veri: {len(df)}")
    print(f"  Uveitis: {sum(df['binary_label']==1)}, Control: {sum(df['binary_label']==0)}")
    print(f"  Benzersiz grup: {df['group_id'].nunique()}")

    # GroupKFold: Aynı hastanın tüm görüntüleri aynı fold'da
    groups = df["group_id"].values
    y = df["binary_label"].values

    gkf = GroupKFold(n_splits=N_FOLDS)

    all_fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, y, groups)):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        # Data leakage kontrolü
        train_groups = set(train_df["group_id"])
        val_groups = set(val_df["group_id"])
        overlap = train_groups & val_groups
        if overlap:
            raise ValueError(f"DATA LEAKAGE! Fold {fold_idx}: {overlap}")

        fold_metrics = train_one_fold(fold_idx, train_df, val_df, models_dir)
        all_fold_results.append(fold_metrics)

    # === ÖZET ===
    print(f"\n{'='*60}")
    print(f"{N_FOLDS}-FOLD CV SONUÇLARI")
    print(f"{'='*60}")

    metric_keys = ["f1", "roc_auc", "accuracy", "precision", "recall"]
    summary = {}

    print(f"\n{'Metrik':<12} | {'Fold 1':>8} | {'Fold 2':>8} | {'Fold 3':>8} | {'Fold 4':>8} | {'Fold 5':>8} | {'Ort.':>8} | {'Std':>8}")
    print("-" * 95)

    for key in metric_keys:
        values = [r[key] for r in all_fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[key] = {"mean": round(float(mean_val), 4), "std": round(float(std_val), 4), "folds": [round(v, 4) for v in values]}
        fold_str = " | ".join(f"{v:8.4f}" for v in values)
        print(f"{key:<12} | {fold_str} | {mean_val:8.4f} | {std_val:8.4f}")

    # JSON kaydet
    output = {
        "n_folds": N_FOLDS,
        "backbone": "EfficientNet-B0",
        "total_samples": len(df),
        "summary": summary,
        "fold_details": [{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in all_fold_results],
    }

    output_path = metrics_dir / "octa_kfold_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nKaydedildi: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
