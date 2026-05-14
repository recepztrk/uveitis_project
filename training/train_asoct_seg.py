import os
import argparse
import time
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp

# Local imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.asoct_seg_dataset import AsoctSegmentationDataset

# Params
EPOCHS = 15
BATCH_SIZE = 16
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "asoct_seg"
MODEL_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
NUM_CLASSES = 6 # Arka plan(0) + 5 sinif (Cornea, Iris, Lesion, Anterior_Chamber, Lens)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def train_one_epoch(model, loader, criterion, optimizer, scaler=None):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({"loss": loss.item()})
        
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def validate(model, loader, criterion, num_classes):
    model.eval()
    running_loss = 0.0
    
    # Calculate IoU
    iou_metric = smp.metrics.iou_score
    total_iou = 0.0
    batches = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            
            # Predictions (argmax over classes)
            tp, fp, fn, tn = smp.metrics.get_stats(outputs.argmax(dim=1).unsqueeze(1), masks.unsqueeze(1), mode='multiclass', num_classes=num_classes)
            iou = iou_metric(tp, fp, fn, tn, reduction="micro")
            total_iou += iou.item()
            batches += 1
            
    epoch_loss = running_loss / len(loader.dataset)
    mean_iou = total_iou / batches if batches > 0 else 0
    
    return {"loss": epoch_loss, "iou": mean_iou}

def main():
    print(f"Device: {DEVICE}")
    
    # Paths
    AIDK_DIR = PROJECT_ROOT / "Veri Setleri" / "Ön segment OCT (AS-OCT)" / "AIDK_Dataset_-_asoct" / "Full-frame_Dataset"
    img_dir = AIDK_DIR / "Original_AS-OCT_Images"
    mask_dir = PROJECT_ROOT / "data_raw" / "asoct" / "masks"
    
    # Dataset
    full_dataset = AsoctSegmentationDataset(img_dir, mask_dir, is_train=True)
    
    # Train/Val Split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # Set val dataset transform to eval mode (no augs)
    val_ds.dataset.is_train = False
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train Images: {len(train_ds)}, Val Images: {len(val_ds)}")
    
    # Model: U-Net with EfficientNet-B0 backbone
    print("Model Yükleniyor: U-Net (efficientnet-b0)")
    model = smp.Unet(
        encoder_name="efficientnet-b0",        
        encoder_weights="imagenet",     
        in_channels=3,                  
        classes=NUM_CLASSES,                      
    )
    model = model.to(DEVICE)
    
    # Loss & Optimizer
    # CrossEntropyLoss is standard for multi-class segmentation
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    
    history = []
    best_iou = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_metrics = validate(model, val_loader, criterion, NUM_CLASSES)
        
        scheduler.step(val_metrics['iou'])
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val mIoU: {val_metrics['iou']:.4f}")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics['loss'],
            "val_iou": val_metrics['iou']
        })
        
        # Save Best Model
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            save_path = MODEL_DIR / "asoct_unet_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"[*] Yeni En İyi Segmentasyon Modeli Kaydedildi! (mIoU: {best_iou:.4f})")
            
    # Save History
    with open(METRICS_DIR / "asoct_seg_train_history.json", "w") as f:
        json.dump(history, f, indent=4)
        
    print("\nSegmentasyon Eğitimi Tamamlandı!")
    print(f"En İyi Val mIoU: {best_iou:.4f}")

if __name__ == '__main__':
    main()
