import os
import argparse
import time
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import timm

# Local imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.mcoa_dataset import McoaClassificationDataset

# Params
EPOCHS = 15
BATCH_SIZE = 32
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mcoa"
MODEL_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def train_one_epoch(model, loader, criterion, optimizer, scaler=None):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast('cuda'): # works for CUDA, MPS doesn't need scaler usually but let's keep standard
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * images.size(0)
        
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
        
        pbar.set_postfix({"loss": loss.item()})
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    return epoch_loss, epoch_f1

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
    epoch_loss = running_loss / len(loader.dataset)
    
    metrics = {
        "loss": epoch_loss,
        "acc": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0)
    }
    return metrics

def main():
    print(f"Device: {DEVICE}")
    
    # Dataset & Loader
    split_csv = PROJECT_ROOT / "metadata" / "mcoa_split.csv"
    train_ds = McoaClassificationDataset(split_csv, split_name='train')
    val_ds = McoaClassificationDataset(split_csv, split_name='val')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    
    # Model
    print("Model Yükleniyor: tf_efficientnet_b0_ns")
    model = timm.create_model("tf_efficientnet_b0_ns", pretrained=True, num_classes=1)
    model = model.to(DEVICE)
    
    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    
    history = []
    best_f1 = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_metrics = validate(model, val_loader, criterion)
        
        scheduler.step(val_metrics['f1'])
        
        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics['loss'],
            "val_f1": val_metrics['f1'],
            "val_auc": val_metrics['auc']
        })
        
        # Save Best Model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            save_path = MODEL_DIR / "mcoa_efficientnet_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"[*] Yeni En İyi Model Kaydedildi! (F1: {best_f1:.4f})")
            
    # Save History
    with open(METRICS_DIR / "mcoa_train_history.json", "w") as f:
        json.dump(history, f, indent=4)
        
    print("\nEğitim Tamamlandı!")
    print(f"En İyi Val F1: {best_f1:.4f}")

if __name__ == '__main__':
    main()
