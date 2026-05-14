# ==============================================================================
# train_bscan_v2_finetuned.py
#
# B-scan OCT V2 modeli — Domain-Specific Pre-training + Fine-tuning.
# Kermany veri seti üzerinde eğitilmiş "göz uzmanı" backbone'u alır,
# son katmanı binary'ye (üveit vs normal) dönüştürür ve 55 üveit
# görüntüsüyle ince ayar yapar.
#
# V1 (baseline) vs V2 (finetuned) FARKI:
#   V1: ImageNet → direkt 55 üveit verisi (büyük domain gap, overfit)
#   V2: ImageNet → 109K Kermany OCT → 55 üveit verisi (kademeli uzmanlaşma)
#
# KULLANILAN TEKNİKLER:
#   - Kermany pretrained backbone yükleme
#   - Progressive Fine-Tuning (head-only → full)
#   - Label Smoothing (0.1) — aşırı güveni azaltır
#   - Agresif data augmentation
#   - BCEWithLogitsLoss + pos_weight (sınıf dengesizliği)
#   - Cosine Annealing LR scheduler
#   - Early stopping (patience=10)
#
# ÇIKTILAR:
#   - outputs/models/bscan_v2_finetuned_best.pth   (en iyi model)
#   - outputs/metrics/bscan_v2_train_history.json    (epoch metrikleri)
#   - outputs/metrics/bscan_v2_test_metrics.json     (test sonuçları)
#
# KULLANIM:
#   python training/train_bscan_v2_finetuned.py
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
    f1_score, roc_auc_score, classification_report, confusion_matrix
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
BATCH_SIZE = 8           # Küçük veri seti — küçük batch
NUM_EPOCHS = 60          # Küçük veri, daha fazla epoch gerekebilir
HEAD_LR = 1e-3           # Head-only: yüksek LR (sıfırdan öğrenecek)
BACKBONE_LR = 1e-4       # Backbone açıkken: düşük LR (öğrenilenleri bozmamak için)
HEAD_ONLY_EPOCHS = 10    # İlk 10 epoch sadece son katman eğitilir
PATIENCE = 20            # Early stopping sabır — küçük veride uzun bekle
LABEL_SMOOTHING = 0.1    # Overconfidence'ı azaltır
NUM_WORKERS = 0


# =============================================================================
# Veri Hazırlama
# =============================================================================
def build_dataloaders(csv_path: str):
    """Mevcut B-scan üveit verisinden DataLoader'lar oluşturur.
    
    Agresif augmentation: Küçük veri setinde modelin farklı varyasyonlar
    görmesini sağlayarak ezberlemeyi azaltır.
    """
    # Agresif eğitim augmentation'ları
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # Rastgele kırpma
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),                 # Dikey flip
        transforms.RandomRotation(degrees=20),                 # Daha geniş rotasyon
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        ),
        transforms.RandomGrayscale(p=0.1),                    # Ara sıra grayscale
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # Rastgele silme
    ])

    # Değerlendirme: augmentation yok
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


# =============================================================================
# Model Oluşturma — Kermany Pretrained Backbone
# =============================================================================
def build_model(pretrained_path: str):
    """Kermany pretrained ResNet-18'den backbone yükler ve binary sınıflandırma
    için son katmanı değiştirir.
    
    Akış:
    1. Boş ResNet-18 oluştur
    2. fc katmanını 4 sınıfa ayarla (Kermany formatı)
    3. Kermany ağırlıklarını yükle
    4. fc katmanını 1 çıkışa değiştir (binary: üveit vs normal)
    """
    pretrained_path = Path(pretrained_path)

    if not pretrained_path.exists():
        raise FileNotFoundError(
            f"Kermany pretrained model bulunamadı: {pretrained_path}\n"
            "Önce 'python training/train_bscan_pretrain_kermany.py' çalıştırın."
        )

    # 1. ResNet-18 oluştur (ImageNet ağırlıkları KULLANILMIYOR)
    model = models.resnet18(weights=None)

    # 2. Kermany formatına ayarla (4 sınıf) ve ağırlıkları yükle
    model.fc = nn.Linear(model.fc.in_features, 4)
    state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print(f"✅ Kermany pretrained ağırlıklar yüklendi: {pretrained_path}")

    # 3. Son katmanı binary'ye değiştir
    # Backbone: Göz anatomisi bilgisi korunur
    # fc: Sıfırdan öğrenilir (üveit vs normal)
    in_features = model.fc.in_features  # 512
    model.fc = nn.Linear(in_features, 1)
    print(f"🔄 Son katman binary'ye dönüştürüldü: {in_features} → 1")

    return model


# =============================================================================
# Progressive Fine-Tuning
# =============================================================================
def freeze_backbone(model):
    """Backbone'u dondur — sadece fc eğitilir."""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

def unfreeze_backbone(model):
    """Tüm parametreleri eğitime aç."""
    for param in model.parameters():
        param.requires_grad = True


# =============================================================================
# Eğitim Döngüsü
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer):
    """Bir epoch eğitim. Loss döndürür."""
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


# =============================================================================
# Değerlendirme
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, criterion):
    """Validation/Test değerlendirmesi. Tüm metrikleri döndürür."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []

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

    avg_loss = running_loss / len(all_labels)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
    }

    return metrics


# =============================================================================
# Pos Weight Hesaplama
# =============================================================================
def compute_pos_weight(csv_path: str):
    """Sınıf dengesizliği telafisi: negatif / pozitif oranı."""
    df = pd.read_csv(csv_path)
    train_df = df[df["split"] == "train"]
    pos_count = (train_df["binary_label"] == 1).sum()
    neg_count = (train_df["binary_label"] == 0).sum()
    return torch.tensor([neg_count / pos_count], dtype=torch.float32)


# =============================================================================
# Ana Fonksiyon
# =============================================================================
def main():
    csv_path = str(PROJECT_ROOT / "metadata/bscan_oct_split.csv")
    pretrained_path = str(PROJECT_ROOT / "outputs/models/bscan_kermany_resnet18_pretrained.pth")
    output_model_dir = PROJECT_ROOT / "outputs/models"
    output_metric_dir = PROJECT_ROOT / "outputs/metrics"
    output_model_dir.mkdir(parents=True, exist_ok=True)
    output_metric_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("B-scan OCT V2 — Kermany Pretrained Fine-tuning")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Pretrained: {pretrained_path}")
    print(f"Epochs: {NUM_EPOCHS} (head-only: {HEAD_ONLY_EPOCHS})")
    print(f"Head LR: {HEAD_LR} | Backbone LR: {BACKBONE_LR}")
    print(f"Label smoothing: {LABEL_SMOOTHING}")
    print()

    # --- Veri ---
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(csv_path)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # --- Model ---
    model = build_model(pretrained_path).to(DEVICE)

    # --- Loss ---
    pos_weight = compute_pos_weight(csv_path).to(DEVICE)
    print(f"⚖️  Pos weight: {pos_weight.item():.3f}")

    # Label smoothing uygulama: hedef etiketleri 0→0.05, 1→0.95 yapar
    # Bu, modelin %100 emin olmasını önler, overfit'i azaltır
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Optimizer (Head-only fazı: yüksek LR) ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=HEAD_LR, weight_decay=1e-4)

    # --- Scheduler ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HEAD_ONLY_EPOCHS)

    # --- Progressive: Backbone dondur ---
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
        # Progressive: Backbone'u aç
        if epoch == HEAD_ONLY_EPOCHS:
            unfreeze_backbone(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=BACKBONE_LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - HEAD_ONLY_EPOCHS
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n🔓 Backbone açıldı — Eğitilebilir: {trainable:,} / {total:,}")

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}  (LR: {current_lr:.7f})")

        # Label smoothing: etiketleri yumuşat
        train_loss = train_one_epoch_with_smoothing(
            model, train_loader, criterion, optimizer, LABEL_SMOOTHING
        )
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

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val — Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['roc_auc']:.4f}")

        # Best model — F1 veya AUC ile seçim (F1=0 kalırsa AUC'ye bak)
        val_score = val_metrics["f1"] if val_metrics["f1"] > 0 else val_metrics["roc_auc"] * 0.01
        if val_score > best_val_f1:
            best_val_f1 = val_score
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  ✅ Yeni en iyi! (F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['roc_auc']:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ İyileşme yok ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f"\n⛔ Early stopping — {PATIENCE} epoch boyunca iyileşme olmadı.")
            break

    total_time = time.time() - start_time
    print(f"\n⏱️  Toplam süre: {total_time / 60:.1f} dakika")

    # --- Test ---
    print(f"\n{'='*60}")
    print("TEST DEĞERLENDİRMESİ (V2 — Kermany Pretrained)")
    print(f"{'='*60}")

    model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_loader, criterion)

    print(f"\nTest Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1:        {test_metrics['f1']:.4f}")
    print(f"Test AUC:       {test_metrics['roc_auc']:.4f}")

    # --- Kaydet ---
    model_path = output_model_dir / "bscan_v2_finetuned_best.pth"
    torch.save(best_model_state, model_path)
    print(f"\n💾 Model: {model_path}")

    history_path = output_metric_dir / "bscan_v2_train_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"📊 Geçmiş: {history_path}")

    test_path = output_metric_dir / "bscan_v2_test_metrics.json"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print(f"📊 Test: {test_path}")


# =============================================================================
# Label Smoothing ile Eğitim
# =============================================================================
def train_one_epoch_with_smoothing(model, loader, criterion, optimizer, smoothing=0.1):
    """Label smoothing uygulayarak eğitim yapar.
    
    Etiketler: 0 → smoothing/2 = 0.05,  1 → 1 - smoothing/2 = 0.95
    Bu, modelin %100 emin olmasını önler ve generalizasyonu artırır.
    """
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        # Label smoothing uygula
        smoothed_labels = labels * (1 - smoothing) + (1 - labels) * smoothing

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, smoothed_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


if __name__ == "__main__":
    main()
