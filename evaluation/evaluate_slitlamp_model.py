# ==============================================================================
# evaluate_slitlamp_model.py
#
# Eğitilmiş slit-lamp modelinin test seti üzerinde nihai değerlendirmesini yapar.
# Kaydedilmiş en iyi model ağırlıklarını yükleyerek confusion matrix, ROC eğrisi
# ve sınıflandırma metriklerini üretir.
#
# Bu script eğitim tamamlandıktan SONRA çalıştırılır.
#
# Çıktılar:
#   - outputs/metrics/slitlamp_confusion_matrix.png   (karmaşıklık matrisi)
#   - outputs/metrics/slitlamp_roc_curve.png           (ROC eğrisi)
#   - outputs/metrics/slitlamp_test_metrics_eval.json   (nihai test metrikleri)
# ==============================================================================

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
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
)

from src.slitlamp_dataset import SlitLampDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    """EfficientNet-B0 model yapısını oluşturur (ağırlıklar yüklenmez).
    Kaydedilmiş ağırlıklar load_state_dict ile ayrıca yüklenecek."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


@torch.no_grad()
def evaluate(model, loader):
    """Test seti üzerinde tahmin yapar, etiketler ve olasılıkları toplar.
    Döndürür: (gerçek etiketler, olasılıklar, tahminler, metrik dict)"""
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []

    for batch in loader:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE).unsqueeze(1)

        logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= 0.5).long()

        all_labels.extend(labels.squeeze(1).cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_probs),
    }

    return all_labels, all_probs, all_preds, metrics


def main():
    output_dir = PROJECT_ROOT / "outputs/metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Değerlendirme dönüşümü — augmentation yok, sadece resize + normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Test setini yükle
    test_dataset = SlitLampDataset(
        csv_path=str(PROJECT_ROOT / "metadata/slitlamp_split.csv"),
        split="test",
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,     # Test setinde sıralama korunmalı
        num_workers=0
    )

    # Eğitilmiş modeli yükle
    model = build_model().to(DEVICE)
    model.load_state_dict(
        torch.load(str(PROJECT_ROOT / "outputs/models/slitlamp_efficientnetb0_best.pth"), map_location=DEVICE)
    )

    # Test seti üzerinde değerlendirme
    labels, probs, preds, metrics = evaluate(model, test_loader)

    print("Test metrics:")
    print(metrics)

    # --- Confusion Matrix görselleştirmesi ---
    # Satırlar: gerçek sınıf, Sütunlar: tahmin edilen sınıf
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Uveitis", "Uveitis"])
    disp.plot(cmap="Blues")
    plt.title("Slit-Lamp Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "slitlamp_confusion_matrix.png", dpi=300)
    plt.close()

    # --- ROC Eğrisi görselleştirmesi ---
    # FPR vs TPR eğrisi ile modelin farklı eşik değerlerindeki performansı gösterilir
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")   # Rastgele sınıflandırıcı referans çizgisi
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Slit-Lamp Test ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "slitlamp_roc_curve.png", dpi=300)
    plt.close()

    # Test metriklerini JSON olarak kaydet
    with open(output_dir / "slitlamp_test_metrics_eval.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Kaydedildi:")
    print(output_dir / "slitlamp_confusion_matrix.png")
    print(output_dir / "slitlamp_roc_curve.png")
    print(output_dir / "slitlamp_test_metrics_eval.json")


if __name__ == "__main__":
    main()