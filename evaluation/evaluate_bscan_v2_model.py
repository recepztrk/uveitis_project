# ==============================================================================
# evaluate_bscan_v2_model.py
#
# B-scan OCT V2 (Kermany Pretrained + Fine-tuned) modelinin kapsamlı
# değerlendirmesini yapar.
#
# Çıktılar:
#   - outputs/plots/bscan_v2_confusion_matrix.png
#   - outputs/plots/bscan_v2_roc_curve.png
#   - outputs/plots/bscan_v2_training_history.png
#   - outputs/plots/bscan_v2_per_sample_predictions.png
#   - outputs/metrics/bscan_v2_eval_report.json
#
# Kullanım:
#   python evaluation/evaluate_bscan_v2_model.py
# ==============================================================================

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
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
# Model Oluşturma
# =============================================================================
def build_model(model_path: str):
    """V2 model yapısını oluşturur ve ağırlıkları yükler."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print(f"✅ Model yüklendi: {model_path}")
    return model


# =============================================================================
# Değerlendirme
# =============================================================================
@torch.no_grad()
def evaluate(model, loader):
    """Test seti üzerinde tahmin yapar."""
    all_labels = []
    all_probs = []
    all_preds = []
    all_filenames = []

    for i, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= 0.5).long()

        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    return all_labels, all_probs, all_preds


# =============================================================================
# 1. Confusion Matrix
# =============================================================================
def plot_confusion_matrix(labels, preds, output_path):
    """Confusion matrix çizer ve kaydeder."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Üveit"]
    )
    disp.plot(cmap="Blues", ax=ax, values_format="d")

    ax.set_title("B-scan OCT V2 — Confusion Matrix (Test)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tahmin Edilen", fontsize=12)
    ax.set_ylabel("Gerçek", fontsize=12)

    # Sayıları büyüt
    for text in disp.text_.ravel():
        text.set_fontsize(20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion Matrix: {output_path}")


# =============================================================================
# 2. ROC Eğrisi
# =============================================================================
def plot_roc_curve(labels, probs, output_path):
    """ROC eğrisi çizer ve kaydeder."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2b6cb0", linewidth=2.5, label=f"V2 Model (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#a0aec0", linewidth=1.5, label="Rastgele (AUC = 0.50)")

    # Optimal eşik noktası
    optimal_idx = np.argmax(tpr - fpr)
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color="#e53e3e", s=100, zorder=5,
               label=f"Optimal eşik = {thresholds[optimal_idx]:.3f}")

    ax.set_xlabel("Yanlış Pozitif Oranı (FPR)", fontsize=12)
    ax.set_ylabel("Doğru Pozitif Oranı (TPR)", fontsize=12)
    ax.set_title("B-scan OCT V2 — ROC Eğrisi (Test)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 ROC Eğrisi: {output_path}")


# =============================================================================
# 3. Training History Grafikleri
# =============================================================================
def plot_training_history(history_path, output_path):
    """Eğitim geçmişi grafiklerini çizer (Loss, F1, AUC, LR)."""
    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_f1"] for h in history]
    val_auc = [h["val_roc_auc"] for h in history]
    lr = [h["lr"] for h in history]

    # Head-only epoch sayısını bul
    head_only = 10  # Bilinen değer

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("B-scan OCT V2 — Eğitim Geçmişi", fontsize=16, fontweight="bold", y=0.98)

    # --- Loss ---
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, color="#2b6cb0", linewidth=2, label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs, val_loss, color="#e53e3e", linewidth=2, label="Val Loss", marker="s", markersize=3)
    ax.axvline(x=head_only + 0.5, color="#805ad5", linestyle="--", alpha=0.7, label="Backbone açıldı")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Val Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Val F1 ---
    ax = axes[0, 1]
    ax.plot(epochs, val_f1, color="#2f855a", linewidth=2, marker="o", markersize=4)
    ax.axvline(x=head_only + 0.5, color="#805ad5", linestyle="--", alpha=0.7, label="Backbone açıldı")
    best_f1_idx = np.argmax(val_f1)
    ax.scatter(epochs[best_f1_idx], val_f1[best_f1_idx], color="#e53e3e", s=100, zorder=5,
               label=f"En iyi F1 = {val_f1[best_f1_idx]:.3f} (Epoch {epochs[best_f1_idx]})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title("Validation F1 Score")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

    # --- Val AUC ---
    ax = axes[1, 0]
    ax.plot(epochs, val_auc, color="#d69e2e", linewidth=2, marker="o", markersize=4)
    ax.axvline(x=head_only + 0.5, color="#805ad5", linestyle="--", alpha=0.7, label="Backbone açıldı")
    best_auc_idx = np.argmax(val_auc)
    ax.scatter(epochs[best_auc_idx], val_auc[best_auc_idx], color="#e53e3e", s=100, zorder=5,
               label=f"En iyi AUC = {val_auc[best_auc_idx]:.3f} (Epoch {epochs[best_auc_idx]})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Validation ROC AUC")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

    # --- Learning Rate ---
    ax = axes[1, 1]
    ax.plot(epochs, lr, color="#805ad5", linewidth=2, marker="o", markersize=3)
    ax.axvline(x=head_only + 0.5, color="#805ad5", linestyle="--", alpha=0.7, label="Backbone açıldı")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Training History: {output_path}")


# =============================================================================
# 4. Per-Sample Tahmin Grafiği
# =============================================================================
def plot_per_sample_predictions(labels, probs, preds, output_path):
    """Her test görüntüsü için tahmin olasılıklarını gösterir."""
    n = len(labels)
    indices = np.arange(n)

    colors = []
    for label, pred in zip(labels, preds):
        if label == pred:
            colors.append("#38a169" if label == 1 else "#3182ce")  # Doğru: yeşil/mavi
        else:
            colors.append("#e53e3e")  # Yanlış: kırmızı

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(indices, probs, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    ax.axhline(y=0.5, color="#e53e3e", linestyle="--", alpha=0.6, label="Eşik (0.5)")
    ax.set_xlabel("Test Görüntüsü #", fontsize=12)
    ax.set_ylabel("Üveit Olasılığı", fontsize=12)
    ax.set_title("B-scan OCT V2 — Test Tahminleri (Görüntü Bazında)", fontsize=14, fontweight="bold")
    ax.set_xticks(indices)
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3, axis="y")

    # Renk açıklaması
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3182ce", label="Normal (doğru)"),
        Patch(facecolor="#38a169", label="Üveit (doğru)"),
        Patch(facecolor="#e53e3e", label="Yanlış tahmin"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Gerçek etiketleri üstüne yaz
    for i, (label, prob) in enumerate(zip(labels, probs)):
        label_text = "Ü" if label == 1 else "N"
        ax.text(i, prob + 0.03, label_text, ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Per-Sample Predictions: {output_path}")


# =============================================================================
# Ana Fonksiyon
# =============================================================================
def main():
    model_path = PROJECT_ROOT / "outputs/models/bscan_v2_finetuned_best.pth"
    csv_path = PROJECT_ROOT / "metadata/bscan_oct_split.csv"
    history_path = PROJECT_ROOT / "outputs/metrics/bscan_v2_train_history.json"
    output_plot_dir = PROJECT_ROOT / "outputs/plots"
    output_metric_dir = PROJECT_ROOT / "outputs/metrics"
    output_plot_dir.mkdir(parents=True, exist_ok=True)
    output_metric_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("B-scan OCT V2 — Kapsamlı Değerlendirme")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print()

    # --- Veri ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    test_dataset = BScanOCTDataset(csv_path=str(csv_path), split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"Test seti: {len(test_dataset)} görüntü")

    # --- Model ---
    model = build_model(str(model_path))

    # --- Değerlendirme ---
    labels, probs, preds = evaluate(model, test_loader)

    # --- Metrikler ---
    metrics = {
        "model": "B-scan OCT V2 (Kermany Pretrained + Fine-tuned)",
        "test_size": len(labels),
        "accuracy": round(accuracy_score(labels, preds), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall": round(recall_score(labels, preds, zero_division=0), 4),
        "f1": round(f1_score(labels, preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(labels, probs), 4),
    }

    print("\n" + "=" * 40)
    print("TEST SONUÇLARI")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n" + classification_report(
        labels, preds,
        target_names=["Normal", "Üveit"],
        digits=4
    ))

    # --- Görselleştirmeler ---
    print("\n📊 Görselleştirmeler oluşturuluyor...\n")

    plot_confusion_matrix(labels, preds, output_plot_dir / "bscan_v2_confusion_matrix.png")
    plot_roc_curve(labels, probs, output_plot_dir / "bscan_v2_roc_curve.png")
    plot_training_history(str(history_path), output_plot_dir / "bscan_v2_training_history.png")
    plot_per_sample_predictions(labels, probs, preds, output_plot_dir / "bscan_v2_per_sample_predictions.png")

    # --- Metrikleri Kaydet ---
    report_path = output_metric_dir / "bscan_v2_eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n📊 Metrik Raporu: {report_path}")

    print("\n✅ Değerlendirme tamamlandı!")


if __name__ == "__main__":
    main()
