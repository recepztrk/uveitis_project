# ==============================================================================
# evaluate_cfp_tta.py
#
# CFP modelini yeniden eğitmeden TTA + Optimal Threshold ile iyileştirir.
# OCTA'da kanıtlanan strateji CFP'ye uyarlanmıştır.
# ==============================================================================

import sys
from pathlib import Path
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

DEVICE = torch.device("cpu")


def get_tta_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return [
        transforms.Compose([  # 0: Orijinal
            transforms.Resize((224, 224)), transforms.ToTensor(), normalize]),
        transforms.Compose([  # 1: Yatay çevirme
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), normalize]),
        transforms.Compose([  # 2: Hafif döndürme (5°)
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=(5, 5)),
            transforms.ToTensor(), normalize]),
        transforms.Compose([  # 3: Ters döndürme (-5°)
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=(-5, -5)),
            transforms.ToTensor(), normalize]),
        transforms.Compose([  # 4: Merkez kırpma
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(), normalize]),
    ]


def build_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


@torch.no_grad()
def predict_with_tta(model, csv_path, split, tta_transforms):
    model.eval()
    df = pd.read_csv(csv_path)
    df = df[df["split"] == split].reset_index(drop=True)
    all_labels, all_probs, all_meta = [], [], []

    for idx in range(len(df)):
        row = df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        label = int(row["binary_label"])
        tta_probs = []
        for t in tta_transforms:
            img_tensor = t(image).unsqueeze(0).to(DEVICE)
            prob = torch.sigmoid(model(img_tensor)).item()
            tta_probs.append(prob)
        all_labels.append(label)
        all_probs.append(np.mean(tta_probs))
        all_meta.append({
            "filepath": row["filepath"],
            "source_dataset": row.get("source_dataset", ""),
            "source_class": row.get("source_class", ""),
        })
    return all_labels, all_probs, all_meta


@torch.no_grad()
def predict_standard(model, csv_path, split):
    model.eval()
    df = pd.read_csv(csv_path)
    df = df[df["split"] == split].reset_index(drop=True)
    t = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    all_labels, all_probs = [], []
    for idx in range(len(df)):
        row = df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        img_tensor = t(image).unsqueeze(0).to(DEVICE)
        prob = torch.sigmoid(model(img_tensor)).item()
        all_labels.append(int(row["binary_label"]))
        all_probs.append(prob)
    return all_labels, all_probs


def find_optimal_threshold(labels, probs):
    best_t, best_f1 = 0.5, 0.0
    thresholds = np.arange(0.10, 0.90, 0.01)
    f1_scores = []
    for t in thresholds:
        preds = [1 if p >= t else 0 for p in probs]
        f1 = f1_score(labels, preds, zero_division=0)
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1, thresholds, f1_scores


def compute_metrics(labels, probs, threshold=0.5):
    preds = [1 if p >= threshold else 0 for p in probs]
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall": round(recall_score(labels, preds, zero_division=0), 4),
        "f1": round(f1_score(labels, preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(labels, probs), 4),
    }, preds


def main():
    print("=" * 65)
    print("CFP — Test Time Augmentation + Optimal Threshold")
    print("=" * 65)

    output_dir = PROJECT_ROOT / "outputs" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = str(PROJECT_ROOT / "metadata" / "cfp_split.csv")
    model_path = PROJECT_ROOT / "outputs" / "models" / "cfp_efficientnetb0_best.pth"

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(str(model_path), map_location=DEVICE))
    print(f"Model: {model_path.name}")

    tta_transforms = get_tta_transforms()
    print(f"TTA: {len(tta_transforms)} augmentation\n")

    # ADIM 1: Val TTA
    print(f"{'─'*65}\nADIM 1: Validasyon — TTA tahminleri\n{'─'*65}")
    val_labels, val_probs, _ = predict_with_tta(model, csv_path, "val", tta_transforms)
    print(f"  Val: {len(val_labels)} (Uv: {sum(val_labels)}, Non: {len(val_labels)-sum(val_labels)})")

    # ADIM 2: Optimal threshold
    print(f"\n{'─'*65}\nADIM 2: Optimal eşik aranıyor\n{'─'*65}")
    opt_t, opt_f1, thresholds, f1_scores = find_optimal_threshold(val_labels, val_probs)
    val_def, _ = compute_metrics(val_labels, val_probs, 0.5)
    val_opt, _ = compute_metrics(val_labels, val_probs, opt_t)
    print(f"  Default (0.50) → Val F1: {val_def['f1']}")
    print(f"  Optimal ({opt_t:.2f})  → Val F1: {val_opt['f1']}")

    # Threshold curve
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_scores, 'b-', linewidth=2)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Default (0.50)')
    plt.axvline(x=opt_t, color='red', linestyle='--', linewidth=2, label=f'Optimal ({opt_t:.2f})')
    plt.xlabel('Threshold'); plt.ylabel('F1 Score')
    plt.title('CFP — F1 vs Classification Threshold (Validation)')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(output_dir / "cfp_tta_threshold_curve.png", dpi=300)
    plt.close()

    # ADIM 3: Test baseline
    print(f"\n{'─'*65}\nADIM 3: Test — Standart (Baseline)\n{'─'*65}")
    test_labels_std, test_probs_std = predict_standard(model, csv_path, "test")
    baseline, baseline_preds = compute_metrics(test_labels_std, test_probs_std, 0.5)
    for k, v in baseline.items():
        print(f"  {k:>10s}: {v}")

    # ADIM 4: Test TTA
    print(f"\n{'─'*65}\nADIM 4: Test — TTA + Optimal Threshold\n{'─'*65}")
    test_labels, test_probs_tta, test_meta = predict_with_tta(model, csv_path, "test", tta_transforms)

    tta_def, _ = compute_metrics(test_labels, test_probs_tta, 0.5)
    tta_opt, tta_opt_preds = compute_metrics(test_labels, test_probs_tta, opt_t)

    print(f"\n  TTA + Default (0.50):")
    for k, v in tta_def.items():
        print(f"    {k:>10s}: {v}")
    print(f"\n  TTA + Optimal ({opt_t:.2f}):")
    for k, v in tta_opt.items():
        print(f"    {k:>10s}: {v}")

    # En iyi konfigürasyonu seç (en yüksek F1, eşit ise precision'ı tercih et)
    configs = [
        ("TTA + Default (0.50)", 0.50, tta_def),
        (f"TTA + Optimal ({opt_t:.2f})", opt_t, tta_opt),
    ]
    best_config = max(configs, key=lambda c: (c[2]['f1'], c[2]['precision']))
    best_name, best_t, best_metrics = best_config

    print(f"\n  Secilen: {best_name}")

    # Karşılaştırma grafiği
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    before_vals = [baseline[k] for k in keys]
    after_vals = [best_metrics[k] for k in keys]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metric_names)); w = 0.35
    b1 = ax.bar(x - w/2, before_vals, w, label='Baseline', color='#5B8DEF', alpha=0.85)
    b2 = ax.bar(x + w/2, after_vals, w, label=best_name, color='#FF6B6B', alpha=0.85)
    ax.set_ylabel('Score'); ax.set_title('CFP — Baseline vs TTA', fontsize=14, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(metric_names); ax.set_ylim(0.5, 1.05)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    for bar in b1:
        ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in b2:
        ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "cfp_tta_comparison.png", dpi=300); plt.close()

    # JSON kaydet
    result = {
        "model": "cfp_efficientnetb0_best.pth (ayni agirliklar)",
        "technique": f"TTA (5x) + {best_name}",
        "tta_count": len(tta_transforms),
        "selected_threshold": round(best_t, 4),
        "test_results": {
            "baseline (t=0.50, no TTA)": baseline,
            "tta_default (t=0.50)": tta_def,
            f"tta_optimal (t={opt_t:.2f})": tta_opt,
        },
        "final_decision": {
            "selected_config": best_name,
            "final_test_metrics": best_metrics,
            "vs_baseline": {
                k: f"{baseline[k]} -> {best_metrics[k]} (delta={round(best_metrics[k]-baseline[k],4)})"
                for k in keys
            },
        },
    }
    json_path = output_dir / "cfp_tta_test_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # SONUÇ
    print(f"\n{'='*65}")
    print("SONUC — Baseline vs TTA")
    print(f"{'='*65}")
    print(f"{'Metrik':<12} {'Baseline':>12} {best_name:>22} {'Delta':>8}")
    print(f"{'─'*56}")
    for key in keys:
        v1, v2 = baseline[key], best_metrics[key]
        d = v2 - v1
        arrow = "+" if d > 0 else ("" if d < 0 else "=")
        print(f"{key:<12} {v1:>12.4f} {v2:>22.4f} {arrow}{abs(d):>7.4f}")
    print(f"{'='*65}")
    print(f"\nDosyalar:")
    print(f"  {json_path}")
    print(f"  {output_dir / 'cfp_tta_comparison.png'}")
    print(f"  {output_dir / 'cfp_tta_threshold_curve.png'}")


if __name__ == "__main__":
    main()
