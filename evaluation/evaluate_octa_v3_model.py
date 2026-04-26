# ==============================================================================
# evaluate_octa_v3_model.py
#
# Eğitilmiş OCTA V3 modelinin test seti üzerinde değerlendirmesini yapar.
# V2'den farklı olarak cihaz bazlı (Heidelberg vs OptoVue) performans
# ayrımı da raporlanır.
#
# Çıktılar:
#   - outputs/metrics/octa_v3_confusion_matrix.png
#   - outputs/metrics/octa_v3_roc_curve.png
#   - outputs/metrics/octa_v3_test_metrics.json
# ==============================================================================

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
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

from src.octa_dataset_v2 import OCTADatasetV2

DEVICE = torch.device("cpu")  # Değerlendirme için CPU yeterli


def build_model():
    """ResNet-18 model yapısını oluşturur (ağırlıklar sonra yüklenecek)."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


@torch.no_grad()
def evaluate(model, loader):
    """Test seti üzerinde tahmin yapar, metadata ile birlikte döndürür."""
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []
    all_filepaths = []

    for batch in loader:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE).unsqueeze(1)

        logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= 0.5).long()

        all_labels.extend(labels.squeeze(1).cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_filepaths.extend(batch["filepath"])

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_probs),
    }

    return all_labels, all_probs, all_preds, all_filepaths, metrics


def source_analysis(labels, probs, preds, filepaths):
    """Cihaz bazlı (Heidelberg vs OptoVue) performans analizi yapar."""
    bh_labels, bh_probs, bh_preds = [], [], []
    ut_labels, ut_probs, ut_preds = [], [], []

    for label, prob, pred, fp in zip(labels, probs, preds, filepaths):
        if "UT-OCTA" in fp:
            ut_labels.append(label)
            ut_probs.append(prob)
            ut_preds.append(pred)
        else:
            bh_labels.append(label)
            bh_probs.append(prob)
            bh_preds.append(pred)

    results = {}
    for name, lbl, prd in [("Heidelberg (BH)", bh_labels, bh_preds), ("OptoVue (UT)", ut_labels, ut_preds)]:
        if len(lbl) > 0:
            results[name] = {
                "count": len(lbl),
                "accuracy": round(accuracy_score(lbl, prd), 4),
                "f1": round(f1_score(lbl, prd, zero_division=0), 4),
                "precision": round(precision_score(lbl, prd, zero_division=0), 4),
                "recall": round(recall_score(lbl, prd, zero_division=0), 4),
            }
    return results


def main():
    output_dir = PROJECT_ROOT / "outputs/metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test veri setini yükle
    test_dataset = OCTADatasetV2(
        csv_path=str(PROJECT_ROOT / "metadata/octa_split.csv"),
        split="test"
    )

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Modeli yükle
    model_path = PROJECT_ROOT / "outputs/models/octa_v3_resnet18_best.pth"
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(str(model_path), map_location=DEVICE))
    print(f"Model yüklendi: {model_path}")

    # Değerlendirme
    labels, probs, preds, filepaths, metrics = evaluate(model, test_loader)

    print("\n" + "=" * 50)
    print("OCTA V3 — TEST METRİKLERİ")
    print("=" * 50)
    print(f"  Test seti boyutu: {len(labels)}")
    for k, v in metrics.items():
        print(f"  {k:>10s}: {v:.4f}")

    # Cihaz bazlı analiz
    source_results = source_analysis(labels, probs, preds, filepaths)
    print(f"\n{'─'*50}")
    print("CİHAZ BAZLI PERFORMANS (Domain Shift Kontrolü)")
    print(f"{'─'*50}")
    for src_name, src_metrics in source_results.items():
        print(f"\n  {src_name} (n={src_metrics['count']}):")
        for k, v in src_metrics.items():
            if k != "count":
                print(f"    {k}: {v:.4f}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Control", "Uveitis"])
    disp.plot(cmap="Blues")
    plt.title("OCTA V3 Test — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "octa_v3_confusion_matrix.png", dpi=300)
    plt.close()

    # --- ROC Eğrisi ---
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("OCTA V3 Test — ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "octa_v3_roc_curve.png", dpi=300)
    plt.close()

    # --- V1 vs V2 vs V3 Karşılaştırma ---
    all_metrics = {"v3": {k: round(v, 4) for k, v in metrics.items()}}

    # V1 ve V2 metriklerini yükle (varsa)
    for version in ["octa_test_metrics", "octa_v2_test_metrics"]:
        path = output_dir / f"{version}.json"
        if path.exists():
            with open(path, "r") as f:
                tag = "v1" if "v2" not in version else "v2"
                all_metrics[tag] = json.load(f)

    # JSON kaydet
    output_json = {
        "test_metrics": {k: round(v, 4) for k, v in metrics.items()},
        "source_analysis": source_results,
        "test_count": len(labels),
    }
    # Karşılaştırma tablosunu da ekle
    if len(all_metrics) > 1:
        output_json["version_comparison"] = all_metrics

    with open(output_dir / "octa_v3_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print("Kaydedilen dosyalar:")
    print(f"  {output_dir / 'octa_v3_confusion_matrix.png'}")
    print(f"  {output_dir / 'octa_v3_roc_curve.png'}")
    print(f"  {output_dir / 'octa_v3_test_metrics.json'}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
