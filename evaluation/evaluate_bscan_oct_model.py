# ==============================================================================
# evaluate_bscan_oct_model.py
#
# Eğitilmiş B-scan OCT modelinin test seti üzerinde nihai değerlendirmesini yapar.
# Kaydedilmiş en iyi model ağırlıklarını yükleyerek confusion matrix, ROC eğrisi
# ve sınıflandırma metriklerini üretir.
#
# Çıktılar:
#   - outputs/metrics/bscan_oct_confusion_matrix.png
#   - outputs/metrics/bscan_oct_roc_curve.png
#   - outputs/metrics/bscan_oct_test_metrics_eval.json
# ==============================================================================

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
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

from src.bscan_oct_dataset import BScanOCTDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    """Test için ResNet-18 model yapısını oluşturur (ağırlıklar daha sonra yüklenecek)."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    # Sınıflandırma katmanını ikili sınıflandırma için 1 çıktıya ayarla
    model.fc = nn.Linear(in_features, 1)
    return model


@torch.no_grad()
def evaluate(model, loader):
    """Test seti üzerinde tahminler yapar ve değerlendirme metriklerini hesaplar.
    DİKKAT: B-scan dataloader (images, labels) döndürür."""
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []

    # Tuple unpacking (Slitlamp ve OCTA'da dict üzerinden yapılmıştı)
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

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

    # Değerlendirme için standart dönüşüm (veri artırma yapılmaz)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    test_dataset = BScanOCTDataset(
        csv_path=str(PROJECT_ROOT / "metadata/bscan_oct_split.csv"),
        split="test",
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,  # Test aşamasında sıra korunmalı
        num_workers=0
    )

    # Modeli başlat ve eğitilmiş ağırlıkları yükle
    model = build_model().to(DEVICE)
    model.load_state_dict(
        torch.load(str(PROJECT_ROOT / "outputs/models/bscan_oct_resnet18_best.pth"), map_location=DEVICE)
    )

    labels, probs, preds, metrics = evaluate(model, test_loader)

    print("Test metrics:")
    print(metrics)

    # --- Confusion Matrix (Karmaşıklık Matrisi) Görselleştirmesi ---
    cm = confusion_matrix(labels, preds)
    # B-scan sınıf isimleri: Normal (0) ve Uveitis (1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Uveitis"])
    disp.plot(cmap="Blues")
    plt.title("B-scan OCT Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "bscan_oct_confusion_matrix.png", dpi=300)
    plt.close()

    # --- ROC Eğrisi Görselleştirmesi ---
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("B-scan OCT Test ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "bscan_oct_roc_curve.png", dpi=300)
    plt.close()

    # Metrikleri JSON dosyasına kaydet
    with open(output_dir / "bscan_oct_test_metrics_eval.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Kaydedildi:")
    print(output_dir / "bscan_oct_confusion_matrix.png")
    print(output_dir / "bscan_oct_roc_curve.png")
    print(output_dir / "bscan_oct_test_metrics_eval.json")


if __name__ == "__main__":
    main()