# ==============================================================================
# evaluate_cfp_model.py
#
# Eğitilmiş CFP modelinin (EfficientNet-B0) test seti üzerinde
# değerlendirmesini yapar.
#
# Ek analiz: Kaynak veri seti bazlı (cfp1000 vs rfmimd2) performans ayrımı.
#
# Çıktılar:
#   - outputs/metrics/cfp_confusion_matrix.png
#   - outputs/metrics/cfp_roc_curve.png
#   - outputs/metrics/cfp_test_metrics.json  (güncellenir: source analizi eklenir)
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

from src.cfp_dataset import CFPDataset

DEVICE = torch.device("cpu")  # Değerlendirme için CPU yeterli


def build_model():
    """EfficientNet-B0 model yapısını oluşturur (ağırlıklar sonra yüklenecek)."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


@torch.no_grad()
def evaluate(model, loader):
    """Test seti üzerinde tahmin yapar, metadata ile birlikte döndürür."""
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []
    all_source_datasets = []
    all_source_classes = []
    all_image_ids = []

    for batch in loader:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE).unsqueeze(1)

        logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= 0.5).long()

        all_labels.extend(labels.squeeze(1).cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_source_datasets.extend(batch["source_dataset"])
        all_source_classes.extend(batch["source_class"])
        all_image_ids.extend(batch["image_id"])

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_probs),
    }

    return (all_labels, all_probs, all_preds,
            all_source_datasets, all_source_classes, all_image_ids, metrics)


def source_dataset_analysis(labels, probs, preds, source_datasets):
    """Kaynak veri seti bazlı (cfp1000 vs rfmimd2) performans analizi."""
    sources = sorted(set(source_datasets))
    results = {}

    for src in sources:
        src_labels = [l for l, s in zip(labels, source_datasets) if s == src]
        src_preds = [p for p, s in zip(preds, source_datasets) if s == src]

        if len(src_labels) == 0:
            continue

        result = {
            "count": len(src_labels),
            "pos_count": sum(1 for l in src_labels if l == 1),
            "neg_count": sum(1 for l in src_labels if l == 0),
            "accuracy": round(accuracy_score(src_labels, src_preds), 4),
            "f1": round(f1_score(src_labels, src_preds, zero_division=0), 4),
            "precision": round(precision_score(src_labels, src_preds, zero_division=0), 4),
            "recall": round(recall_score(src_labels, src_preds, zero_division=0), 4),
        }
        results[src] = result

    return results


def error_analysis(labels, probs, preds, source_classes, image_ids):
    """Yanlış tahminleri (FP ve FN) detaylı listeler."""
    errors = {"false_positives": [], "false_negatives": []}

    for lbl, prob, pred, sc, img_id in zip(labels, probs, preds, source_classes, image_ids):
        if pred == 1 and lbl == 0:  # FP
            errors["false_positives"].append({
                "image_id": img_id,
                "source_class": sc,
                "prob": round(prob, 4),
            })
        elif pred == 0 and lbl == 1:  # FN
            errors["false_negatives"].append({
                "image_id": img_id,
                "source_class": sc,
                "prob": round(prob, 4),
            })

    return errors


def main():
    output_dir = PROJECT_ROOT / "outputs/metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test veri setini yükle
    csv_path = str(PROJECT_ROOT / "metadata/cfp_split.csv")
    test_dataset = CFPDataset(csv_path, split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Modeli yükle
    model_path = PROJECT_ROOT / "outputs/models/cfp_efficientnetb0_best.pth"
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(str(model_path), map_location=DEVICE,
                                     weights_only=True))
    print(f"Model yüklendi: {model_path}")

    # Değerlendirme
    (labels, probs, preds, source_datasets,
     source_classes, image_ids, metrics) = evaluate(model, test_loader)

    print("\n" + "=" * 55)
    print("CFP BASELINE — TEST METRİKLERİ")
    print("=" * 55)
    print(f"  Test seti boyutu: {len(labels)}")
    print(f"    Uveitis:     {sum(1 for l in labels if l == 1)}")
    print(f"    Non-uveitis: {sum(1 for l in labels if l == 0)}")
    for k, v in metrics.items():
        print(f"  {k:>10s}: {v:.4f}")

    # Kaynak veri seti analizi
    source_results = source_dataset_analysis(labels, probs, preds, source_datasets)
    print(f"\n{'─'*55}")
    print("KAYNAK VERİ SETİ BAZLI PERFORMANS")
    print(f"{'─'*55}")
    for src_name, src_metrics in source_results.items():
        print(f"\n  {src_name} (n={src_metrics['count']}, "
              f"pos={src_metrics['pos_count']}, neg={src_metrics['neg_count']}):")
        print(f"    Acc: {src_metrics['accuracy']:.4f} | "
              f"F1: {src_metrics['f1']:.4f} | "
              f"Prec: {src_metrics['precision']:.4f} | "
              f"Rec: {src_metrics['recall']:.4f}")

    # Hata analizi
    errors = error_analysis(labels, probs, preds, source_classes, image_ids)
    print(f"\n{'─'*55}")
    print("HATA ANALİZİ")
    print(f"{'─'*55}")
    print(f"  False Positives (Yanlış Alarm): {len(errors['false_positives'])}")
    for fp in errors["false_positives"]:
        print(f"    {fp['image_id']} (class: {fp['source_class']}, prob: {fp['prob']})")
    print(f"  False Negatives (Kaçırılan):    {len(errors['false_negatives'])}")
    for fn in errors["false_negatives"]:
        print(f"    {fn['image_id']} (class: {fn['source_class']}, prob: {fn['prob']})")

    # --- Confusion Matrix ---
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Uveitis", "Uveitis"]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title("CFP Test — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "cfp_confusion_matrix.png", dpi=300)
    plt.close()
    print(f"\n  Confusion matrix kaydedildi.")

    # --- ROC Eğrisi ---
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2196F3", linewidth=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("CFP Test — ROC Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "cfp_roc_curve.png", dpi=300)
    plt.close()
    print(f"  ROC curve kaydedildi.")

    # --- JSON kaydet ---
    output_json = {
        "test_metrics": {k: round(v, 4) for k, v in metrics.items()},
        "test_count": len(labels),
        "source_analysis": source_results,
        "error_analysis": errors,
    }

    json_path = output_dir / "cfp_test_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print("Kaydedilen dosyalar:")
    print(f"  {output_dir / 'cfp_confusion_matrix.png'}")
    print(f"  {output_dir / 'cfp_roc_curve.png'}")
    print(f"  {json_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
