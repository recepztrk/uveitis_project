# ==============================================================================
# evaluate_mcoa_model.py
#
# Eğitilmiş AS-OCT (MCOA) modelinin test seti üzerinde nihai değerlendirmesini
# yapar. Confusion matrix, ROC eğrisi, tüm metrikleri üretir ve JSON olarak
# kaydeder.
#
# Çıktılar:
#   - outputs/mcoa/metrics/mcoa_confusion_matrix.png
#   - outputs/mcoa/metrics/mcoa_roc_curve.png
#   - outputs/mcoa/metrics/mcoa_test_metrics.json
#   - outputs/mcoa/metrics/mcoa_test_predictions.json
# ==============================================================================

import sys
import json
from pathlib import Path
import signal

def handler(signum, frame):
    raise TimeoutError("iCloud timeout")

signal.signal(signal.SIGALRM, handler)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import timm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
)

# === AYARLAR ===
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = PROJECT_ROOT / "outputs/mcoa/models/mcoa_efficientnet_best.pth"
SPLIT_CSV  = PROJECT_ROOT / "metadata/mcoa_split.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/mcoa/metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD  = 0.50
BATCH_SIZE = 64
LABEL_NAMES = {0: "Normal", 1: "Opaque Cornea (Patolojik)"}


# === MODEL ===
def build_model():
    """timm Noisy Student EfficientNet-B0 yapısını oluşturur."""
    model = timm.create_model("tf_efficientnet_b0.ns_jft_in1k", pretrained=False, num_classes=1)
    return model


# === DATASET (DataLoader için) ===
from torch.utils.data import Dataset

class McoaTestDataset(Dataset):
    """Hafif test dataset — sadece torchvision transforms kullanır, albumentations gerekmez."""
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(row["filepath"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.transform(image), int(row["label"]), row["filepath"]


# === DEĞERLENDİRME ===
@torch.no_grad()
def run_inference(model, csv_path, split="test"):
    """Manuel batch inference — DataLoader'ın macOS'ta neden olduğu thread kilitlenmesini engeller."""
    df = pd.read_csv(csv_path)
    df = df[df["split"] == split].reset_index(drop=True)
    print(f"  Test seti: {len(df)} görüntü  |  Batch size: {BATCH_SIZE}")

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model.eval()
    all_labels, all_probs, all_paths = [], [], []

    num_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(num_batches):
        batch_df = df.iloc[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        
        batch_imgs = []
        batch_labels = []
        batch_paths_current = []

        for _, row in batch_df.iterrows():
            try:
                signal.alarm(2)
                img_cv = cv2.imread(row["filepath"])
                signal.alarm(0)
                if img_cv is None:
                    raise ValueError("cv2.imread None döndürdü")
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img_cv)
            except Exception:
                signal.alarm(0)
                img = Image.new("RGB", (224, 224), (0, 0, 0))
            
            batch_imgs.append(eval_transform(img))
            batch_labels.append(int(row["label"]))
            batch_paths_current.append(row["filepath"])

        batch_tensor = torch.stack(batch_imgs).to(DEVICE)
        
        # Inference
        probs = torch.sigmoid(model(batch_tensor)).squeeze(1).cpu().numpy()
        
        all_labels.extend(batch_labels)
        all_probs.extend(probs.tolist())
        all_paths.extend(batch_paths_current)
        
        print(f"  ... Batch {i+1} / {num_batches} işlendi. Toplam: {len(all_labels)}/{len(df)}", end="\r")

    print(f"\n  Tamamlandı: {len(all_labels)} görüntü          ")

    all_records = []
    for label, prob, path in zip(all_labels, all_probs, all_paths):
        all_records.append({
            "filepath": path,
            "label":      label,
            "label_name": LABEL_NAMES.get(label, str(label)),
            "prob":       round(prob, 4),
            "pred":       int(prob >= THRESHOLD),
            "correct":    int(int(prob >= THRESHOLD) == label),
        })

    return all_labels, all_probs, all_records


def compute_metrics(labels, probs, threshold=THRESHOLD):
    preds = [1 if p >= threshold else 0 for p in probs]
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "accuracy":  round(accuracy_score(labels, preds), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall":    round(recall_score(labels, preds, zero_division=0), 4),
        "f1":        round(f1_score(labels, preds, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(labels, probs), 4),
        "specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),
        "npv":         round(tn / (tn + fn) if (tn + fn) > 0 else 0, 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "threshold": threshold,
        "n_test": len(labels),
        "n_positive": int(sum(labels)),
        "n_negative": int(len(labels) - sum(labels)),
    }, [1 if p >= threshold else 0 for p in probs]


# === GÖRSELLEŞTİRME ===
def plot_confusion_matrix(labels, preds, output_dir):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    n = cm.sum()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Normal", "Opaque\nCornea"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes, fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = 100 * count / n
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if count > thresh else "black")

    ax.set_xlabel("Tahmin Edilen Sınıf", fontsize=13)
    ax.set_ylabel("Gerçek Sınıf", fontsize=13)
    ax.set_title("AS-OCT (MCOA) — Test Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    save_path = output_dir / "mcoa_confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {save_path.name}")


def plot_roc_curve(labels, probs, metrics, output_dir):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#2563EB', linewidth=2.5,
            label=f'AS-OCT MCOA (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#2563EB')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.5, label='Rastgele Sınıflandırıcı')

    # Operating point (threshold=0.50)
    from sklearn.metrics import roc_curve as _roc
    _, _, thresholds = _roc(labels, probs)
    closest_idx = np.argmin(np.abs(thresholds - THRESHOLD))
    ax.scatter(fpr[closest_idx], tpr[closest_idx], s=80, color='red', zorder=5,
               label=f'Threshold = {THRESHOLD:.2f}')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('AS-OCT (MCOA) — ROC Eğrisi', fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
    plt.tight_layout()
    save_path = output_dir / "mcoa_roc_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {save_path.name}")


def plot_metrics_bar(metrics, output_dir):
    """Metrik özet bar grafiği."""
    keys   = ["accuracy", "precision", "recall", "f1", "roc_auc", "specificity"]
    labels = ["Accuracy", "Precision", "Recall\n(Sensitivity)", "F1 Score", "ROC AUC", "Specificity"]
    values = [metrics[k] for k in keys]
    colors = ['#3B82F6', '#8B5CF6', '#10B981', '#F59E0B', '#EF4444', '#06B6D4']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, color=colors, alpha=0.85, edgecolor='white', linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', ha='left', fontsize=11, fontweight='bold')

    ax.set_xlim(0, 1.12)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('AS-OCT (MCOA) — Final Test Metrikleri', fontsize=14, fontweight='bold', pad=15)
    ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    save_path = output_dir / "mcoa_metrics_bar.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {save_path.name}")


def main():
    print("=" * 65)
    print("AS-OCT (MCOA) — Final Model Değerlendirmesi")
    print("=" * 65)
    print(f"  Cihaz: {DEVICE}")
    print(f"  Model: {MODEL_PATH.name}")
    print(f"  CSV  : {SPLIT_CSV.name}")

    # 1. Model yükle
    model = build_model().to(DEVICE)
    state = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("  Model başarıyla yüklendi.\n")

    # 2. Inference
    print(f"{'─'*65}")
    print("Test seti üzerinde inference çalışıyor...")
    labels, probs, records = run_inference(model, str(SPLIT_CSV), split="test")

    # 3. Metrikler
    metrics, preds = compute_metrics(labels, probs, THRESHOLD)
    cm = metrics["confusion_matrix"]

    print(f"\n{'─'*65}")
    print("SONUÇLAR")
    print(f"{'─'*65}")
    print(f"  Test seti       : {metrics['n_test']} görüntü "
          f"(Patolojik: {metrics['n_positive']}, Normal: {metrics['n_negative']})")
    print(f"  Accuracy        : {metrics['accuracy']:.4f}")
    print(f"  Precision       : {metrics['precision']:.4f}")
    print(f"  Recall (Sens.)  : {metrics['recall']:.4f}")
    print(f"  F1 Score        : {metrics['f1']:.4f}")
    print(f"  ROC AUC         : {metrics['roc_auc']:.4f}")
    print(f"  Specificity     : {metrics['specificity']:.4f}")
    print(f"  NPV             : {metrics['npv']:.4f}")
    print(f"  Confusion Matrix: TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}, TP={cm['TP']}")

    # 4. Görseller
    print(f"\n{'─'*65}")
    print("Görseller üretiliyor...")
    plot_confusion_matrix(labels, preds, OUTPUT_DIR)
    plot_roc_curve(labels, probs, metrics, OUTPUT_DIR)
    plot_metrics_bar(metrics, OUTPUT_DIR)

    # 5. JSON kaydet
    json_path = OUTPUT_DIR / "mcoa_test_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Kaydedildi: {json_path.name}")

    pred_path = OUTPUT_DIR / "mcoa_test_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  Kaydedildi: {pred_path.name}")

    print(f"\n{'='*65}")
    print("TAMAMLANDI!")
    print(f"{'='*65}")
    print(f"  Çıktı klasörü: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
