# ==============================================================================
# train_bscan_pretrain_kermany.py
#
# B-scan OCT modeli için domain-specific pre-training scripti.
# Kermany veri seti (109K B-scan OCT görüntüsü) üzerinde ResNet-18'i
# 4 sınıflı sınıflandırma (CNV, DME, DRUSEN, NORMAL) ile eğitir.
#
# AMAÇ:
#   Model, ImageNet'ten gelen genel görsel özellikler yerine,
#   retinal katman yapısı, maküler ödem, drusen birikimi gibi
#   göze özgü patolojik desenleri öğrenir → "Göz Uzmanı" olur.
#
# STRATEJİ:
#   1. ImageNet pretrained ResNet-18 backbone
#   2. Progressive Fine-Tuning: İlk 3 epoch sadece head, sonra full
#   3. Cosine Annealing LR scheduler
#   4. Class weights ile sınıf dengesizliği telafisi
#   5. Early stopping (patience=5)
#
# ÇIKTILAR:
#   - outputs/models/bscan_kermany_resnet18_pretrained.pth  (en iyi model)
#   - outputs/metrics/bscan_kermany_train_history.json       (epoch metrikleri)
#   - outputs/metrics/bscan_kermany_test_metrics.json        (test sonuçları)
#
# KULLANIM:
#   python training/train_bscan_pretrain_kermany.py
# ==============================================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.kermany_dataset import KermanyOCTDataset, CLASS_NAMES


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
BATCH_SIZE = 32          # 109K veri için makul batch boyutu
NUM_EPOCHS = 12          # Büyük veri seti — fazla epoch gerekmez
LEARNING_RATE = 1e-4     # Fine-tuning için düşük LR
HEAD_ONLY_EPOCHS = 3     # İlk 3 epoch sadece son katman eğitilir
PATIENCE = 5             # Early stopping sabır değeri
VAL_RATIO = 0.1          # Train'den %10 validation ayrılır
NUM_WORKERS = 0          # macOS uyumluluğu için 0


# =============================================================================
# Veri Hazırlama
# =============================================================================
def build_dataloaders(data_root: str):
    """Train, validation ve test DataLoader'larını oluşturur.
    
    Kermany'de val split yok, bu yüzden train'den %10 stratified ayırıyoruz.
    """
    # --- Transform tanımları ---
    # Eğitim: Agresif augmentation (modelin ezberlememesi için)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Değerlendirme: Sadece resize + normalize
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # --- Tam train veri setini yükle ---
    full_train = KermanyOCTDataset(data_root, split="train", transform=None)
    print(f"📊 Tam train seti: {full_train}")

    # --- Stratified train/val split ---
    # Her sınıftan orantılı olarak %10 val'e ayır
    labels = [label for _, label in full_train.samples]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=42)
    train_indices, val_indices = next(splitter.split(np.zeros(len(labels)), labels))

    print(f"✂️  Train: {len(train_indices)} | Val: {len(val_indices)}")

    # --- Alt kümeler oluştur (farklı transform'larla) ---
    train_dataset = KermanyOCTDataset(
        data_root, split="train", transform=train_transform,
        indices=train_indices.tolist()
    )
    val_dataset = KermanyOCTDataset(
        data_root, split="train", transform=eval_transform,
        indices=val_indices.tolist()
    )
    test_dataset = KermanyOCTDataset(
        data_root, split="test", transform=eval_transform
    )

    print(f"📦 Train dataset: {train_dataset}")
    print(f"📦 Val dataset: {val_dataset}")
    print(f"📦 Test dataset: {test_dataset}")

    # --- DataLoader'lar ---
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


# =============================================================================
# Model Oluşturma
# =============================================================================
def build_model():
    """ImageNet pretrained ResNet-18 modelini 4 sınıflı çıktıya adapte eder.
    
    fc katmanı: 512 → 4 (CNV, DME, DRUSEN, NORMAL)
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features  # 512
    model.fc = nn.Linear(in_features, len(CLASS_NAMES))  # 4 sınıf
    return model


# =============================================================================
# Progressive Fine-Tuning: Backbone Dondurma/Çözme
# =============================================================================
def freeze_backbone(model):
    """Backbone'u dondurur — sadece fc katmanı eğitilir.
    İlk epoch'larda ağırlıkların bozulmasını önler."""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

def unfreeze_backbone(model):
    """Tüm parametreleri eğitime açar (full fine-tuning)."""
    for param in model.parameters():
        param.requires_grad = True


# =============================================================================
# Eğitim Döngüsü (Bir Epoch)
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer):
    """Bir epoch eğitim yapar, ortalama loss ve accuracy döndürür."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# =============================================================================
# Değerlendirme (Validation / Test)
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, criterion):
    """Model performansını değerlendirir. Loss, accuracy ve F1 döndürür."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in tqdm(loader, desc="  Eval", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = running_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "all_labels": all_labels,
        "all_preds": all_preds,
    }


# =============================================================================
# Ana Eğitim Fonksiyonu
# =============================================================================
def main():
    data_root = str(PROJECT_ROOT / "new-datasets/CellData/OCT")
    output_model_dir = PROJECT_ROOT / "outputs/models"
    output_metric_dir = PROJECT_ROOT / "outputs/metrics"
    output_model_dir.mkdir(parents=True, exist_ok=True)
    output_metric_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("B-scan OCT Domain-Specific Pre-training (Kermany)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS} (head-only: {HEAD_ONLY_EPOCHS})")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    # --- Veri ---
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(data_root)

    # --- Model ---
    model = build_model().to(DEVICE)

    # --- Loss: Class weights ile dengelenmiş CrossEntropyLoss ---
    class_weights = train_ds.class_weights.to(DEVICE)
    print(f"\n⚖️  Class weights: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # --- LR Scheduler: Cosine Annealing ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # --- Progressive Fine-Tuning: İlk 3 epoch backbone dondur ---
    freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"🧊 Backbone donduruldu — Eğitilebilir: {trainable:,} / {total:,}")

    # --- Eğitim Döngüsü ---
    best_val_f1 = -1.0
    best_model_state = None
    patience_counter = 0
    history = []

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # Progressive: HEAD_ONLY_EPOCHS sonrası backbone'u aç
        if epoch == HEAD_ONLY_EPOCHS:
            unfreeze_backbone(model)
            # Backbone açıldığında optimizer'ı yeniden oluştur
            # (yeni parametreler eklendi)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE * 0.1, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - HEAD_ONLY_EPOCHS
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n🔓 Backbone açıldı — Eğitilebilir: {trainable:,} / {total:,}")

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}  (LR: {current_lr:.6f})")
        print(f"{'='*50}")

        # Eğitim
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)

        # Validation
        val_metrics = evaluate(model, val_loader, criterion)

        # LR güncelle
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Epoch sonuçları
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_f1_macro": round(val_metrics["f1_macro"], 4),
            "lr": round(current_lr, 7),
            "time_sec": round(epoch_time, 1),
        }
        history.append(epoch_result)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1_macro']:.4f}")
        print(f"  Süre: {epoch_time:.1f}s")

        # En iyi modeli kaydet (Val F1 macro'ya göre)
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  ✅ Yeni en iyi model! (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ İyileşme yok ({patience_counter}/{PATIENCE})")

        # Early stopping kontrolü
        if patience_counter >= PATIENCE:
            print(f"\n⛔ Early stopping — {PATIENCE} epoch boyunca iyileşme olmadı.")
            break

    total_time = time.time() - start_time
    print(f"\n⏱️  Toplam eğitim süresi: {total_time / 60:.1f} dakika")

    # --- Test Değerlendirmesi ---
    print(f"\n{'='*60}")
    print("TEST DEĞERLENDİRMESİ (En İyi Model)")
    print(f"{'='*60}")

    model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_loader, criterion)

    print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Macro: {test_metrics['f1_macro']:.4f}")
    print(f"\nSınıf Bazlı Rapor:")
    print(classification_report(
        test_metrics["all_labels"],
        test_metrics["all_preds"],
        target_names=CLASS_NAMES,
        digits=4
    ))

    # --- Kaydet ---
    # Model ağırlıkları (sadece backbone + fc katmanları)
    model_path = output_model_dir / "bscan_kermany_resnet18_pretrained.pth"
    torch.save(best_model_state, model_path)
    print(f"💾 Model kaydedildi: {model_path}")

    # Eğitim geçmişi
    history_path = output_metric_dir / "bscan_kermany_train_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"📊 Eğitim geçmişi: {history_path}")

    # Test metrikleri
    test_save = {
        "accuracy": test_metrics["accuracy"],
        "f1_macro": test_metrics["f1_macro"],
        "loss": test_metrics["loss"],
    }
    test_path = output_metric_dir / "bscan_kermany_test_metrics.json"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_save, f, indent=2, ensure_ascii=False)
    print(f"📊 Test metrikleri: {test_path}")


if __name__ == "__main__":
    main()
