# ==============================================================================
# generate_mcoa_publication_plots.py
#
# AS-OCT (MCOA) modülü için makale/sunum kalitesinde görsel üretimi.
# Diğer modalitelerdeki generate_*_publication_plots.py ile aynı format.
#
# Ön koşul: evaluate_mcoa_model.py önce çalıştırılmış olmalı.
#
# Üretilen çıktılar (outputs/mcoa/plots/):
#   pub_01_data_distribution.png  — Veri seti dağılımı
#   pub_02_confusion_matrix.png   — Yüksek çözünürlüklü CM
#   pub_03_roc_curve.png          — Gölgeli ROC eğrisi
#   pub_04_final_metrics.png      — Final metrik kartı
#   pub_05_model_pipeline.png     — Eğitim pipeline özeti
#   pub_06_all_models_comparison.png — 5 modalite karşılaştırma
# ==============================================================================

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix

# === AYARLAR ===
METRICS_DIR = PROJECT_ROOT / "outputs/mcoa/metrics"
PLOTS_DIR   = PROJECT_ROOT / "outputs/mcoa/plots"
SPLIT_CSV   = PROJECT_ROOT / "metadata/mcoa_split.csv"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Renk paleti (koyu tema)
PALETTE = {
    "primary":    "#3B82F6",
    "secondary":  "#8B5CF6",
    "positive":   "#EF4444",
    "negative":   "#22C55E",
    "accent":     "#F59E0B",
    "background": "#0f172a",
    "card":       "#1e293b",
    "text":       "#F1F5F9",
    "grid":       "#334155",
}


def load_metrics():
    """evaluate_mcoa_model.py tarafından üretilen metrikleri yükler."""
    metrics_path = METRICS_DIR / "mcoa_test_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path.name} bulunamadı. "
            "Önce evaluate_mcoa_model.py çalıştırın."
        )
    with open(metrics_path) as f:
        return json.load(f)


def load_predictions():
    """Tahmin kayıtlarını yükler."""
    pred_path = METRICS_DIR / "mcoa_test_predictions.json"
    if not pred_path.exists():
        return None
    with open(pred_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Veri Seti Dağılımı
# ─────────────────────────────────────────────────────────────────────────────
def plot_data_distribution():
    df = pd.read_csv(SPLIT_CSV)

    # Sınıf dağılımı (tüm veri)
    class_counts = df["label"].value_counts().sort_index()
    class_labels = ["Normal (0)", "Opaque Cornea\n(Patolojik) (1)"]
    class_colors = [PALETTE["negative"], PALETTE["positive"]]
    class_vals   = [class_counts.get(0, 0), class_counts.get(1, 0)]

    # Split dağılımı
    split_counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    splits = ["train", "val", "test"]
    split_order = [s for s in splits if s in split_counts.index]

    fig = plt.figure(figsize=(16, 5), facecolor=PALETTE["background"])
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel 1: Sınıf pasta grafiği
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(PALETTE["card"])
    wedges, texts, autotexts = ax1.pie(
        class_vals, labels=class_labels, colors=class_colors,
        autopct='%1.1f%%', startangle=90,
        textprops=dict(color=PALETTE["text"], fontsize=11),
        wedgeprops=dict(edgecolor=PALETTE["background"], linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(12); at.set_fontweight("bold")
    ax1.set_title("Sınıf Dağılımı\n(Tüm Veri)", color=PALETTE["text"],
                  fontsize=13, fontweight="bold", pad=15)

    # Panel 2: Split bar grafiği
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(PALETTE["card"])
    x = np.arange(len(split_order)); w = 0.35
    b1 = ax2.bar(x - w/2, [split_counts.loc[s, 0] for s in split_order],
                 w, label="Normal", color=PALETTE["negative"], alpha=0.85)
    b2 = ax2.bar(x + w/2, [split_counts.loc[s, 1] for s in split_order],
                 w, label="Patolojik", color=PALETTE["positive"], alpha=0.85)
    for bar in list(b1) + list(b2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(int(bar.get_height())), ha='center', va='bottom',
                 color=PALETTE["text"], fontsize=10, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.capitalize() for s in split_order],
                        color=PALETTE["text"], fontsize=11)
    ax2.set_ylabel("Görüntü Sayısı", color=PALETTE["text"], fontsize=11)
    ax2.set_title("Train / Val / Test Dağılımı", color=PALETTE["text"],
                  fontsize=13, fontweight="bold", pad=15)
    ax2.tick_params(colors=PALETTE["text"])
    ax2.spines[["top","right"]].set_visible(False)
    ax2.spines[["bottom","left"]].set_color(PALETTE["grid"])
    ax2.set_facecolor(PALETTE["card"])
    ax2.yaxis.label.set_color(PALETTE["text"])
    ax2.legend(fontsize=10, labelcolor=PALETTE["text"],
               facecolor=PALETTE["card"], edgecolor=PALETTE["grid"])

    # Panel 3: Veri seti bilgi kutusu
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(PALETTE["card"])
    ax3.axis("off")
    total = len(df)
    n_pos = class_vals[1]; n_neg = class_vals[0]
    info = [
        ("Veri Seti", "MCOA (Corneal OCT)"),
        ("Toplam Görüntü", f"{total:,}"),
        ("Normal", f"{n_neg:,}  (%{100*n_neg/total:.1f})"),
        ("Opaque Cornea", f"{n_pos:,}  (%{100*n_pos/total:.1f})"),
        ("Sınıf Dengesi", f"1:{n_neg/n_pos:.1f}"),
        ("Model", "timm EfficientNet-B0"),
        ("Ön Eğitim", "Noisy Student (JFT)"),
        ("Görev", "Binary Sınıflandırma"),
    ]
    y_pos = 0.92
    for key, val in info:
        ax3.text(0.05, y_pos, f"{key}:", color=PALETTE["accent"],
                 fontsize=10, fontweight='bold', transform=ax3.transAxes)
        ax3.text(0.45, y_pos, val, color=PALETTE["text"],
                 fontsize=10, transform=ax3.transAxes)
        y_pos -= 0.11
    ax3.set_title("Veri Seti Özeti", color=PALETTE["text"],
                  fontsize=13, fontweight="bold", pad=15)

    plt.suptitle("AS-OCT (MCOA) — Veri Seti Dağılımı",
                 color=PALETTE["text"], fontsize=15, fontweight="bold", y=1.02)
    save_path = PLOTS_DIR / "pub_01_data_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor=PALETTE["background"], edgecolor='none')
    plt.close()
    print(f"  ✓ {save_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Confusion Matrix (yüksek çözünürlük)
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(metrics):
    cm_data = metrics["confusion_matrix"]
    tn, fp = cm_data["TN"], cm_data["FP"]
    fn, tp = cm_data["FN"], cm_data["TP"]
    n_total = tn + fp + fn + tp
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["background"])
    ax.set_facecolor(PALETTE["card"])

    im = ax.imshow(cm, interpolation='nearest',
                   cmap=plt.cm.Blues, vmin=0, vmax=cm.max())
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["text"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["text"])

    classes = ["Normal", "Opaque\nCornea"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes, fontsize=12, color=PALETTE["text"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes, fontsize=12, color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])

    thresh = cm.max() / 2.0
    labels_matrix = [
        [f"TN = {tn}\n({100*tn/n_total:.1f}%)", f"FP = {fp}\n({100*fp/n_total:.1f}%)"],
        [f"FN = {fn}\n({100*fn/n_total:.1f}%)", f"TP = {tp}\n({100*tp/n_total:.1f}%)"],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels_matrix[i][j],
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "#1e293b")

    ax.set_xlabel("Tahmin Edilen Sınıf", fontsize=13, color=PALETTE["text"], labelpad=10)
    ax.set_ylabel("Gerçek Sınıf", fontsize=13, color=PALETTE["text"], labelpad=10)
    ax.set_title("AS-OCT (MCOA) — Test Confusion Matrix",
                 fontsize=14, fontweight="bold", color=PALETTE["text"], pad=15)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])

    plt.tight_layout()
    save_path = PLOTS_DIR / "pub_02_confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor=PALETTE["background"], edgecolor='none')
    plt.close()
    print(f"  ✓ {save_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — ROC Eğrisi (gölgeli)
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curve(predictions, metrics):
    if predictions is None:
        print("  ⚠ Tahmin dosyası bulunamadı, ROC atlandı.")
        return

    labels = [r["label"] for r in predictions]
    probs  = [r["prob"]  for r in predictions]

    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["background"])
    ax.set_facecolor(PALETTE["card"])

    ax.plot(fpr, tpr, color=PALETTE["primary"], linewidth=2.5,
            label=f'AS-OCT MCOA  (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["primary"])
    ax.plot([0, 1], [0, 1], '--', color=PALETTE["grid"], linewidth=1.5,
            alpha=0.7, label='Rastgele Sınıflandırıcı')

    # Operating point
    closest_idx = np.argmin(np.abs(thresholds - 0.50))
    ax.scatter(fpr[closest_idx], tpr[closest_idx], s=100, zorder=5,
               color=PALETTE["accent"], edgecolors='white', linewidth=1.5,
               label=f'Threshold = 0.50\n(Sens={tpr[closest_idx]:.3f}, '
                     f'Spec={1-fpr[closest_idx]:.3f})')

    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12,
                  color=PALETTE["text"], labelpad=8)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12,
                  color=PALETTE["text"], labelpad=8)
    ax.set_title('AS-OCT (MCOA) — ROC Eğrisi', fontsize=14,
                 fontweight="bold", color=PALETTE["text"], pad=15)
    ax.legend(fontsize=10, labelcolor=PALETTE["text"],
              facecolor=PALETTE["card"], edgecolor=PALETTE["grid"],
              loc='lower right')
    ax.grid(True, alpha=0.2, color=PALETTE["grid"])
    ax.tick_params(colors=PALETTE["text"])
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])

    plt.tight_layout()
    save_path = PLOTS_DIR / "pub_03_roc_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor=PALETTE["background"], edgecolor='none')
    plt.close()
    print(f"  ✓ {save_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Final Metrik Kartı
# ─────────────────────────────────────────────────────────────────────────────
def plot_final_metrics(metrics):
    metric_items = [
        ("Accuracy",          metrics["accuracy"],    "#3B82F6"),
        ("Precision",         metrics["precision"],   "#8B5CF6"),
        ("Recall (Sens.)",    metrics["recall"],      "#10B981"),
        ("F1 Score",          metrics["f1"],          "#F59E0B"),
        ("ROC AUC",           metrics["roc_auc"],     "#EF4444"),
        ("Specificity",       metrics["specificity"], "#06B6D4"),
        ("NPV",               metrics["npv"],         "#84CC16"),
    ]

    keys   = [m[0] for m in metric_items]
    values = [m[1] for m in metric_items]
    colors = [m[2] for m in metric_items]

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=PALETTE["background"])
    ax.set_facecolor(PALETTE["card"])

    bars = ax.barh(keys, values, color=colors, alpha=0.85,
                   edgecolor=PALETTE["background"], linewidth=1.5, height=0.6)

    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', ha='left',
                color=PALETTE["text"], fontsize=12, fontweight='bold')

    ax.axvline(x=0.90, color=PALETTE["grid"], linestyle='--',
               alpha=0.5, linewidth=1.5, label='0.90 referans')
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Score', fontsize=12, color=PALETTE["text"], labelpad=8)
    ax.set_title('AS-OCT (MCOA) — Final Test Metrikleri',
                 fontsize=14, fontweight='bold', color=PALETTE["text"], pad=15)
    ax.tick_params(colors=PALETTE["text"])
    ax.grid(axis='x', alpha=0.2, color=PALETTE["grid"])
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["bottom","left"]].set_edgecolor(PALETTE["grid"])
    ax.invert_yaxis()

    # Sağ üst köşeye CM özeti
    cm = metrics["confusion_matrix"]
    info_text = (f"n={metrics['n_test']}  |  "
                 f"TP={cm['TP']}  FP={cm['FP']}  "
                 f"FN={cm['FN']}  TN={cm['TN']}")
    ax.text(0.99, 0.01, info_text, transform=ax.transAxes,
            ha='right', va='bottom', color=PALETTE["grid"],
            fontsize=9, style='italic')

    plt.tight_layout()
    save_path = PLOTS_DIR / "pub_04_final_metrics.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor=PALETTE["background"], edgecolor='none')
    plt.close()
    print(f"  ✓ {save_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Model Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_pipeline():
    fig, ax = plt.subplots(figsize=(14, 4), facecolor=PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])
    ax.axis("off")

    steps = [
        ("MCOA\nVeri Seti\n6,272 görüntü",     PALETTE["primary"]),
        ("Ön İşleme\nResize 224×224\nImageNet Norm", PALETTE["secondary"]),
        ("timm\nEfficientNet-B0\n(Noisy Student)", PALETTE["accent"]),
        ("Fine-tuning\n15 Epoch\nAdamW LR=1e-4", PALETTE["positive"]),
        ("Test\nDeğerlendirme\nF1=0.920, AUC=0.950", PALETTE["negative"]),
    ]

    box_w, box_h = 0.16, 0.55
    total = len(steps)
    gap   = (1 - total * box_w) / (total + 1)
    y_ctr = 0.5

    for i, (label, color) in enumerate(steps):
        x = gap + i * (box_w + gap)
        fancy = mpatches.FancyBboxPatch(
            (x, y_ctr - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.02", linewidth=2,
            edgecolor='white', facecolor=color, alpha=0.85,
            transform=ax.transAxes, clip_on=False
        )
        ax.add_patch(fancy)
        ax.text(x + box_w/2, y_ctr, label,
                ha='center', va='center', fontsize=9.5,
                fontweight='bold', color='white',
                transform=ax.transAxes, clip_on=False)

        if i < total - 1:
            ax.annotate("", xy=(x + box_w + gap/2, y_ctr),
                        xytext=(x + box_w, y_ctr),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle="->", color='white',
                                        lw=2.5, mutation_scale=20))

    ax.set_title("AS-OCT (MCOA) — Model Eğitim Pipeline",
                 fontsize=14, fontweight="bold", color=PALETTE["text"],
                 pad=20, y=1.05)

    save_path = PLOTS_DIR / "pub_05_model_pipeline.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor=PALETTE["background"], edgecolor='none')
    plt.close()
    print(f"  ✓ {save_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — Tüm Modalite Karşılaştırma
# ─────────────────────────────────────────────────────────────────────────────
def plot_all_models_comparison():
    models_data = [
        {"name": "Slit-lamp\n(EfficientNet-B0)",     "f1": 0.900, "auc": 0.988, "recall": 0.931},
        {"name": "OCTA V5-TTA\n(ResNet-18)",          "f1": 0.780, "auc": 0.910, "recall": 0.821},
        {"name": "CFP TTA\n(EfficientNet-B0)",         "f1": 0.947, "auc": 0.998, "recall": 1.000},
        {"name": "B-scan OCT\n(ResNet-18 Kermany)",   "f1": 0.900, "auc": 1.000, "recall": 1.000},
        {"name": "AS-OCT MCOA\n(EfficientNet-B0 NS)", "f1": 0.920, "auc": 0.950, "recall": 0.940},
    ]

    x = np.arange(len(models_data))
    f1s     = [m["f1"]     for m in models_data]
    aucs    = [m["auc"]    for m in models_data]
    recalls = [m["recall"] for m in models_data]
    names   = [m["name"]   for m in models_data]
    w = 0.26

    colors_f1  = [PALETTE["primary"],   PALETTE["primary"],   PALETTE["primary"],
                  PALETTE["primary"],   PALETTE["accent"]]
    colors_auc = [PALETTE["secondary"], PALETTE["secondary"], PALETTE["secondary"],
                  PALETTE["secondary"], "#F97316"]
    colors_rec = [PALETTE["negative"],  PALETTE["negative"],  PALETTE["negative"],
                  PALETTE["negative"],  "#84CC16"]

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=PALETTE["background"])
    ax.set_facecolor(PALETTE["card"])

    bars_f1  = ax.bar(x - w,   f1s,     w, label='F1 Score', color=colors_f1,  alpha=0.85)
    bars_auc = ax.bar(x,       aucs,    w, label='AUC',      color=colors_auc, alpha=0.85)
    bars_rec = ax.bar(x + w,   recalls, w, label='Recall',   color=colors_rec, alpha=0.85)

    for bars in [bars_f1, bars_auc, bars_rec]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom',
                    fontsize=8, color=PALETTE["text"], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, color=PALETTE["text"])
    ax.set_ylabel("Score", fontsize=12, color=PALETTE["text"])
    ax.set_ylim(0.6, 1.08)
    ax.set_title("Tüm Modaliteler — F1 / AUC / Recall Karşılaştırması",
                 fontsize=14, fontweight='bold', color=PALETTE["text"], pad=15)
    ax.axhline(y=0.9, color=PALETTE["grid"], linestyle='--',
               alpha=0.5, linewidth=1.2, label='0.90 referans')
    ax.tick_params(colors=PALETTE["text"])
    ax.grid(axis='y', alpha=0.2, color=PALETTE["grid"])
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["bottom","left"]].set_edgecolor(PALETTE["grid"])
    ax.legend(fontsize=10, labelcolor=PALETTE["text"],
              facecolor=PALETTE["card"], edgecolor=PALETTE["grid"])

    # AS-OCT çubuklarını vurgula
    for bar in list(bars_f1)[-1:] + list(bars_auc)[-1:] + list(bars_rec)[-1:]:
        bar.set_edgecolor("white")
        bar.set_linewidth(2)

    plt.tight_layout()
    save_path = PLOTS_DIR / "pub_06_all_models_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor=PALETTE["background"], edgecolor='none')
    plt.close()
    print(f"  ✓ {save_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("AS-OCT (MCOA) — Publication Plots")
    print("=" * 65)

    # Metrikleri yükle
    metrics     = load_metrics()
    predictions = load_predictions()

    print(f"\nMetrikler yüklendi: F1={metrics['f1']}, AUC={metrics['roc_auc']}")
    print(f"Görseller: {PLOTS_DIR}\n")
    print(f"{'─'*65}")

    plot_data_distribution()
    plot_confusion_matrix(metrics)
    plot_roc_curve(predictions, metrics)
    plot_final_metrics(metrics)
    plot_model_pipeline()
    plot_all_models_comparison()

    print(f"\n{'='*65}")
    print(f"TAMAMLANDI! {PLOTS_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
