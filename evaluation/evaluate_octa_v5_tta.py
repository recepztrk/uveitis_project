# ==============================================================================
# evaluate_octa_v5_tta.py
#
# OCTA V3 modelini yeniden eğitmeden, iki post-processing tekniği ile
# performansı artırır:
#
#   1. Test Time Augmentation (TTA):
#      Her görüntüyü 5 farklı şekilde modele sunar (orijinal + 4 augmented)
#      ve tahminlerin ortalamasını alarak daha güvenilir olasılıklar üretir.
#
#   2. Optimal Threshold Search:
#      Sabit 0.5 eşiği yerine, validasyon setinde en yüksek F1 skorunu
#      veren optimal eşiği bulur ve test setinde bu eşiği kullanır.
#
# Çıktılar:
#   - outputs/metrics/octa_v5_tta_test_metrics.json
#   - outputs/metrics/octa_v5_tta_confusion_matrix.png
#   - outputs/metrics/octa_v5_tta_roc_curve.png
#   - outputs/metrics/octa_v5_tta_threshold_curve.png
#   - outputs/metrics/octa_v5_tta_comparison.png
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
from torchvision import models, transforms
from PIL import Image
import pandas as pd
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

DEVICE = torch.device("cpu")


# =============================================================================
# TTA TRANSFORM TANIMLARI
# =============================================================================
def get_tta_transforms():
    """
    Test Time Augmentation için 5 farklı transform seti döndürür.
    Her biri farklı bir perspektiften görüntüyü modele sunar.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    tta_list = [
        # 0: Orijinal (standart test transform)
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]),
        # 1: Yatay çevirme
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            normalize,
        ]),
        # 2: Dikey çevirme
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            normalize,
        ]),
        # 3: Hem yatay hem dikey çevirme (180° dönüş eşdeğeri)
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            normalize,
        ]),
        # 4: Hafif merkez kırpma (farklı scale)
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
    ]
    return tta_list


# =============================================================================
# MODEL
# =============================================================================
def build_model():
    """ResNet-18 model yapısını oluşturur."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


# =============================================================================
# TTA TAHMİN FONKSİYONU
# =============================================================================
@torch.no_grad()
def predict_with_tta(model, csv_path, split, tta_transforms):
    """
    Her görüntüyü N farklı augmentation ile modelden geçirip
    olasılıkların ortalamasını döndürür.
    """
    model.eval()
    df = pd.read_csv(csv_path)
    df = df[df["split"] == split].reset_index(drop=True)

    all_labels = []
    all_probs = []
    all_filepaths = []

    n_tta = len(tta_transforms)

    for idx in range(len(df)):
        row = df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        label = int(row["binary_label"])

        # Her TTA transform'u ile tahmin al
        tta_probs = []
        for t in tta_transforms:
            img_tensor = t(image).unsqueeze(0).to(DEVICE)
            logit = model(img_tensor)
            prob = torch.sigmoid(logit).item()
            tta_probs.append(prob)

        # Ortalamasını al
        avg_prob = np.mean(tta_probs)

        all_labels.append(label)
        all_probs.append(avg_prob)
        all_filepaths.append(row["filepath"])

    return all_labels, all_probs, all_filepaths


# =============================================================================
# STANDART TAHMİN (TTA olmadan, karşılaştırma için)
# =============================================================================
@torch.no_grad()
def predict_standard(model, csv_path, split):
    """Standart (TTA'sız) tahmin — karşılaştırma bazı olarak kullanılır."""
    model.eval()
    df = pd.read_csv(csv_path)
    df = df[df["split"] == split].reset_index(drop=True)

    standard_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_labels = []
    all_probs = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        img_tensor = standard_transform(image).unsqueeze(0).to(DEVICE)
        logit = model(img_tensor)
        prob = torch.sigmoid(logit).item()

        all_labels.append(int(row["binary_label"]))
        all_probs.append(prob)

    return all_labels, all_probs


# =============================================================================
# OPTİMAL EŞİK ARAMA
# =============================================================================
def find_optimal_threshold(labels, probs):
    """
    Validasyon setindeki olasılıklara göre en yüksek F1 skorunu
    veren eşik değerini bulur.
    """
    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.arange(0.10, 0.90, 0.01)
    f1_scores = []

    for t in thresholds:
        preds = [1 if p >= t else 0 for p in probs]
        f1 = f1_score(labels, preds, zero_division=0)
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1, thresholds, f1_scores


# =============================================================================
# CİHAZ BAZLI ANALİZ
# =============================================================================
def source_analysis(labels, probs, preds, filepaths, threshold):
    """Cihaz bazlı performans analizi."""
    sources = {}
    for label, prob, pred, fp in zip(labels, probs, preds, filepaths):
        src = "OptoVue (UT)" if "UT-OCTA" in fp else "Heidelberg (BH)"
        sources.setdefault(src, {"labels": [], "probs": [], "preds": []})
        sources[src]["labels"].append(label)
        sources[src]["probs"].append(prob)
        sources[src]["preds"].append(pred)

    results = {}
    for name, data in sources.items():
        lbl, prd = data["labels"], data["preds"]
        results[name] = {
            "count": len(lbl),
            "accuracy": round(accuracy_score(lbl, prd), 4),
            "f1": round(f1_score(lbl, prd, zero_division=0), 4),
            "precision": round(precision_score(lbl, prd, zero_division=0), 4),
            "recall": round(recall_score(lbl, prd, zero_division=0), 4),
        }
    return results


# =============================================================================
# METRİK HESAPLAMA
# =============================================================================
def compute_metrics(labels, probs, threshold=0.5):
    """Verilen eşiğe göre tüm metrikleri hesaplar."""
    preds = [1 if p >= threshold else 0 for p in probs]
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall": round(recall_score(labels, preds, zero_division=0), 4),
        "f1": round(f1_score(labels, preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(labels, probs), 4),
    }, preds


# =============================================================================
# GÖRSELLEŞTİRME
# =============================================================================
def plot_threshold_curve(thresholds, f1_scores, optimal_t, optimal_f1, save_path):
    """F1 vs Threshold grafiği çizer."""
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1 Score')
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Default (0.50)')
    plt.axvline(x=optimal_t, color='red', linestyle='--', linewidth=2,
                label=f'Optimal ({optimal_t:.2f})')
    plt.scatter([optimal_t], [optimal_f1], color='red', s=100, zorder=5)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Validation Set — F1 Score vs Classification Threshold', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✅ {save_path}")


def plot_comparison(before, after, save_path):
    """V3 (önce) vs V5-TTA (sonra) karşılaştırma bar grafiği."""
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
    keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    before_vals = [before[k] for k in keys]
    after_vals = [after[k] for k in keys]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, before_vals, width, label='V3 (Baseline)',
                   color='#5B8DEF', alpha=0.85)
    bars2 = ax.bar(x + width/2, after_vals, width, label='V5-TTA (Optimized)',
                   color='#FF6B6B', alpha=0.85)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('OCTA Model Performance — V3 vs V5-TTA', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Bar üzerine değer yaz
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✅ {save_path}")


def plot_confusion_matrix(labels, preds, save_path, title):
    """Confusion matrix çizer."""
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Control", "Uveitis"])
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✅ {save_path}")


def plot_roc_curve(labels, probs, save_path, title):
    """ROC eğrisi çizer."""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc_val = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc_val:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✅ {save_path}")


# =============================================================================
# ANA FONKSİYON
# =============================================================================
def main():
    print("=" * 65)
    print("OCTA V5 — Test Time Augmentation + Optimal Threshold")
    print("=" * 65)

    output_dir = PROJECT_ROOT / "outputs" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = str(PROJECT_ROOT / "metadata" / "octa_split.csv")
    model_path = PROJECT_ROOT / "outputs" / "models" / "octa_v3_resnet18_best.pth"

    # 1. Modeli yükle
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(str(model_path), map_location=DEVICE))
    print(f"Model yüklendi: {model_path.name}")

    tta_transforms = get_tta_transforms()
    print(f"TTA augmentation sayısı: {len(tta_transforms)}")

    # ─────────────────────────────────────────────────────────────
    # ADIM 1: Validasyon seti üzerinde TTA tahminleri al
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("ADIM 1: Validasyon setinde TTA tahminleri alınıyor...")
    print(f"{'─'*65}")

    val_labels, val_probs_tta, _ = predict_with_tta(
        model, csv_path, "val", tta_transforms
    )
    print(f"  Val örnekleri: {len(val_labels)} "
          f"(Üveit: {sum(val_labels)}, Kontrol: {len(val_labels)-sum(val_labels)})")

    # ─────────────────────────────────────────────────────────────
    # ADIM 2: Optimal eşik bul (Validasyon TTA olasılıkları üzerinde)
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("ADIM 2: Optimal eşik aranıyor (Validasyon seti)...")
    print(f"{'─'*65}")

    optimal_t, optimal_f1, thresholds, f1_scores = find_optimal_threshold(
        val_labels, val_probs_tta
    )
    # Validasyonda sabit 0.5 ile de hesapla
    val_metrics_default, _ = compute_metrics(val_labels, val_probs_tta, threshold=0.5)
    val_metrics_optimal, _ = compute_metrics(val_labels, val_probs_tta, threshold=optimal_t)

    print(f"  Default eşik (0.50) → Val F1: {val_metrics_default['f1']:.4f}")
    print(f"  Optimal eşik ({optimal_t:.2f})  → Val F1: {val_metrics_optimal['f1']:.4f}")
    print(f"  🎯 Seçilen optimal eşik: {optimal_t:.2f}")

    # Threshold curve grafiği
    plot_threshold_curve(
        thresholds, f1_scores, optimal_t, optimal_f1,
        output_dir / "octa_v5_tta_threshold_curve.png"
    )

    # ─────────────────────────────────────────────────────────────
    # ADIM 3: Test seti — Standart tahmin (baseline karşılaştırma)
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("ADIM 3: Test seti — Standart tahmin (V3 Baseline)...")
    print(f"{'─'*65}")

    test_labels_std, test_probs_std = predict_standard(model, csv_path, "test")
    baseline_metrics, baseline_preds = compute_metrics(
        test_labels_std, test_probs_std, threshold=0.5
    )
    print(f"  V3 Baseline (threshold=0.50):")
    for k, v in baseline_metrics.items():
        print(f"    {k:>10s}: {v:.4f}")

    # ─────────────────────────────────────────────────────────────
    # ADIM 4: Test seti — TTA + Optimal Threshold
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("ADIM 4: Test seti — TTA + Optimal Threshold...")
    print(f"{'─'*65}")

    test_labels, test_probs_tta, test_filepaths = predict_with_tta(
        model, csv_path, "test", tta_transforms
    )

    # TTA + default threshold (0.5)
    tta_default_metrics, _ = compute_metrics(test_labels, test_probs_tta, threshold=0.5)

    # TTA + optimal threshold
    tta_optimal_metrics, tta_optimal_preds = compute_metrics(
        test_labels, test_probs_tta, threshold=optimal_t
    )

    print(f"\n  TTA + Default (0.50):")
    for k, v in tta_default_metrics.items():
        print(f"    {k:>10s}: {v:.4f}")

    print(f"\n  TTA + Optimal ({optimal_t:.2f}):")
    for k, v in tta_optimal_metrics.items():
        print(f"    {k:>10s}: {v:.4f}")

    # ─────────────────────────────────────────────────────────────
    # ADIM 5: Cihaz bazlı analiz
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("ADIM 5: Cihaz bazlı analiz (TTA + Optimal Threshold)...")
    print(f"{'─'*65}")

    src_results = source_analysis(
        test_labels, test_probs_tta, tta_optimal_preds, test_filepaths, optimal_t
    )
    for src_name, src_metrics in src_results.items():
        print(f"\n  {src_name} (n={src_metrics['count']}):")
        for k, v in src_metrics.items():
            if k != "count":
                print(f"    {k}: {v:.4f}")

    # ─────────────────────────────────────────────────────────────
    # ADIM 6: Karşılaştırma ve görseller
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("ADIM 6: Görseller oluşturuluyor...")
    print(f"{'─'*65}")

    # Confusion Matrix (TTA + Optimal)
    plot_confusion_matrix(
        test_labels, tta_optimal_preds,
        output_dir / "octa_v5_tta_confusion_matrix.png",
        f"OCTA V5-TTA — Confusion Matrix (threshold={optimal_t:.2f})"
    )

    # ROC Curve (TTA)
    plot_roc_curve(
        test_labels, test_probs_tta,
        output_dir / "octa_v5_tta_roc_curve.png",
        "OCTA V5-TTA — ROC Curve"
    )

    # V3 vs V5-TTA karşılaştırma
    plot_comparison(
        baseline_metrics, tta_optimal_metrics,
        output_dir / "octa_v5_tta_comparison.png"
    )

    # ─────────────────────────────────────────────────────────────
    # ADIM 7: JSON kaydet
    # ─────────────────────────────────────────────────────────────
    output_json = {
        "model": "octa_v3_resnet18_best.pth (aynı ağırlıklar, yeniden eğitim yok)",
        "technique": "Test Time Augmentation (5x) + Optimal Threshold Search",
        "tta_count": len(tta_transforms),
        "optimal_threshold": round(optimal_t, 4),
        "validation_threshold_search": {
            "default_0.50_f1": val_metrics_default["f1"],
            "optimal_f1": val_metrics_optimal["f1"],
        },
        "test_results": {
            "v3_baseline (t=0.50, no TTA)": baseline_metrics,
            "v5_tta_default (t=0.50, TTA)": tta_default_metrics,
            "v5_tta_optimal (t={:.2f}, TTA)".format(optimal_t): tta_optimal_metrics,
        },
        "source_analysis": src_results,
        "improvement": {
            "f1_delta": round(tta_optimal_metrics["f1"] - baseline_metrics["f1"], 4),
            "accuracy_delta": round(tta_optimal_metrics["accuracy"] - baseline_metrics["accuracy"], 4),
            "precision_delta": round(tta_optimal_metrics["precision"] - baseline_metrics["precision"], 4),
            "auc_delta": round(tta_optimal_metrics["roc_auc"] - baseline_metrics["roc_auc"], 4),
        },
    }

    json_path = output_dir / "octa_v5_tta_test_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    print(f"  ✅ {json_path}")

    # ─────────────────────────────────────────────────────────────
    # SONUÇ RAPORU
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SONUÇ RAPORU — V3 Baseline vs V5-TTA")
    print(f"{'='*65}")
    print(f"{'Metrik':<12} {'V3 (t=0.50)':>12} {'V5-TTA (t={:.2f})':>16} {'Δ':>8}".format(optimal_t))
    print(f"{'─'*50}")
    for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        v3 = baseline_metrics[key]
        v5 = tta_optimal_metrics[key]
        delta = v5 - v3
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"{key:<12} {v3:>12.4f} {v5:>16.4f} {arrow:>2}{abs(delta):>6.4f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
