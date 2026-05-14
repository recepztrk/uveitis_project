# ==============================================================================
# train_bscan_v4_kfold.py
#
# B-scan V4 modeli için 5-Fold Cross Validation ile eğitim scripti.
# ~9600 Normal vs 27 Gerçek Üveit resmini dengeli test edebilmek için
# her fold'un test tahminleri bir havuzda toplanır (Aggregated Metrics).
# ==============================================================================

import sys
from pathlib import Path
import copy
import json
import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
BATCH_SIZE = 32          # Veri seti çok büyük, batch size arttırıldı
NUM_EPOCHS = 15          # Her fold için epoch (Büyük veride hızlı öğrenir)
HEAD_LR = 1e-3
BACKBONE_LR = 5e-5
HEAD_ONLY_EPOCHS = 5     # Sadece head eğitilecek ilk epoch sayısı
PATIENCE = 7
LABEL_SMOOTHING = 0.1
N_SPLITS = 5
NUM_WORKERS = 4 if torch.cuda.is_available() else 0

# =============================================================================
# Özel K-Fold Dataset Sınıfı
# =============================================================================
class KFoldBScanDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = PROJECT_ROOT / row["filepath"]
        label = float(row["binary_label"])
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

# =============================================================================
# Model — Kermany Pretrained Yükleyici
# =============================================================================
def build_model(pretrained_path: str):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4) # Kermany'nin 4 sınıfı vardı
    
    # Pretrained ağırlıkları yükle
    state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    
    # Bizim binary (0-1) işimize göre son katmanı değiştir
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True

# =============================================================================
# Eğitim & Değerlendirme Fonksiyonları
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, smoothing=0.1):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="    Train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        # Label Smoothing (0 -> 0.1, 1 -> 0.9)
        smoothed = labels * (1 - smoothing) + (1 - labels) * smoothing

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, smoothed)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    for images, labels in tqdm(loader, desc="    Test ", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
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
    return metrics, all_labels, all_probs, all_preds

def plot_aggregated_cm(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Uveitis"],
                yticklabels=["Normal", "Uveitis"])
    plt.title("Aggregated Confusion Matrix (Tüm K-Fold Sonuçları)")
    plt.ylabel("Gerçek Sınıf")
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# =============================================================================
# Ana Çalışma Döngüsü (5 Folds)
# =============================================================================
def main():
    csv_path = PROJECT_ROOT / "metadata/bscan_oct_split_v4_kfold.csv"
    pretrained_path = PROJECT_ROOT / "outputs/models/bscan_kermany_resnet18_pretrained.pth"
    output_model_dir = PROJECT_ROOT / "outputs/models"
    output_metric_dir = PROJECT_ROOT / "outputs/metrics"
    
    df = pd.read_csv(csv_path)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("=" * 60)
    print("B-scan V4 — K-Fold Cross Validation Eğitimi Başlıyor")
    print("=" * 60)
    
    aggregated_y_true = []
    aggregated_y_prob = []
    aggregated_y_pred = []
    
    best_overall_f1 = -1.0

    for fold in range(N_SPLITS):
        print(f"\n{'='*20} FOLD {fold+1}/{N_SPLITS} {'='*20}")
        
        # Veriyi hazırla
        train_df = df[df[f"fold_{fold}"] == "train"]
        test_df = df[df[f"fold_{fold}"] == "test"]
        
        train_dataset = KFoldBScanDataset(train_df, transform=train_transform)
        test_dataset = KFoldBScanDataset(test_df, transform=eval_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        # Pos Weight Hesapla
        pos = (train_df["binary_label"] == 1).sum()
        neg = (train_df["binary_label"] == 0).sum()
        pos_weight = torch.tensor([neg / max(1, pos)], dtype=torch.float32).to(DEVICE)
        print(f"⚖️ Sınıf Ağırlığı (Pos Weight): {pos_weight.item():.2f} ({neg} Normal / {pos} Üveit)")

        # Modeli Yükle
        model = build_model(pretrained_path).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Progressive ayarları
        freeze_backbone(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=HEAD_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HEAD_ONLY_EPOCHS)
        
        best_fold_f1 = -1.0
        best_fold_model = None
        patience_counter = 0
        
        for epoch in range(NUM_EPOCHS):
            if epoch == HEAD_ONLY_EPOCHS:
                unfreeze_backbone(model)
                optimizer = torch.optim.AdamW(model.parameters(), lr=BACKBONE_LR, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - HEAD_ONLY_EPOCHS)
                print(f"  🔓 Backbone Açıldı!")

            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, LABEL_SMOOTHING)
            val_metrics, _, _, _ = evaluate(model, test_loader, criterion)
            scheduler.step()
            
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Test F1: {val_metrics['f1']:.4f} | Test AUC: {val_metrics['roc_auc']:.4f}")
            
            if val_metrics["f1"] > best_fold_f1:
                best_fold_f1 = val_metrics["f1"]
                best_fold_model = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("  ⛔ Early Stopping tetiklendi.")
                    break
        
        # Fold sonu testini (en iyi modelle) yapıp havuza ekle
        model.load_state_dict(best_fold_model)
        metrics, labels, probs, preds = evaluate(model, test_loader, criterion)
        
        print(f"🏆 Fold {fold+1} Bitti -> Test F1: {metrics['f1']:.4f}, Test Recall: {metrics['recall']:.4f}")
        
        aggregated_y_true.extend(labels)
        aggregated_y_prob.extend(probs)
        aggregated_y_pred.extend(preds)
        
        # En iyi fold'u sakla (İleride inference için kullanırız)
        if metrics["f1"] > best_overall_f1:
            best_overall_f1 = metrics["f1"]
            torch.save(best_fold_model, output_model_dir / "bscan_v4_kfold_best.pth")

    # =========================================================================
    # Aggregated Sonuçları Raporla
    # =========================================================================
    print(f"\n{'='*60}")
    print("🎯 B-SCAN V4 K-FOLD GENEL SONUÇLARI (AGGREGATED)")
    print(f"{'='*60}")
    
    agg_acc = accuracy_score(aggregated_y_true, aggregated_y_pred)
    agg_prec = precision_score(aggregated_y_true, aggregated_y_pred, zero_division=0)
    agg_rec = recall_score(aggregated_y_true, aggregated_y_pred, zero_division=0)
    agg_f1 = f1_score(aggregated_y_true, aggregated_y_pred, zero_division=0)
    agg_auc = roc_auc_score(aggregated_y_true, aggregated_y_prob)
    
    print(f"Toplam Test Edilen Resim: {len(aggregated_y_true)}")
    print(f"Test Edilen Gerçek Üveit : {sum(aggregated_y_true)}")
    print("-" * 30)
    print(f"Accuracy : {agg_acc:.4f}")
    print(f"Precision: {agg_prec:.4f}")
    print(f"Recall   : {agg_rec:.4f}")
    print(f"F1 Score : {agg_f1:.4f}")
    print(f"ROC AUC  : {agg_auc:.4f}")
    
    # Grafiği Çiz ve Kaydet
    cm_path = output_metric_dir / "bscan_v4_kfold_confusion_matrix.png"
    plot_aggregated_cm(aggregated_y_true, aggregated_y_pred, cm_path)
    print(f"📊 Confusion Matrix kaydedildi: {cm_path}")
    print(f"💾 En iyi Fold Modeli kaydedildi: {output_model_dir / 'bscan_v4_kfold_best.pth'}")

if __name__ == "__main__":
    main()
