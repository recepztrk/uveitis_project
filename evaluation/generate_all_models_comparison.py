"""
generate_all_models_comparison.py

5 unimodal modelin + Router'ın performansını karşılaştıran
makale/poster kalitesinde görsel üretir.

Çıktılar → outputs/comparison/
  - pub_comparison_01_bar_chart.png   (F1, AUC, Recall bar chart)
  - pub_comparison_02_radar.png       (radar / spider chart)
  - pub_comparison_03_summary_table.png (tablo görseli)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

# ── Çıktı dizini ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Gerçek metrikler (JSON'lardan doğrulandı, 14 Mayıs 2026) ─────────────────
MODELS = {
    "Slit-lamp\n(EfficientNet-B0)": {
        "f1":        0.900,
        "auc":       0.988,
        "recall":    0.931,
        "precision": 0.871,
        "accuracy":  0.970,
        "color":     "#38bdf8",   # sky-400
        "short":     "Slit-lamp",
        "data_n":    1309,
        "backbone":  "EfficientNet-B0\n(ImageNet)",
    },
    "OCTA\n(ResNet-18 + TTA)": {
        "f1":        0.780,
        "auc":       0.910,
        "recall":    0.821,
        "precision": 0.742,
        "accuracy":  0.843,
        "color":     "#a78bfa",   # violet-400
        "short":     "OCTA",
        "data_n":    525,
        "backbone":  "ResNet-18\n(ImageNet + TTA)",
    },
    "CFP\n(EfficientNet-B0\n+ TTA + t=0.68)": {
        "f1":        0.947,
        "auc":       0.998,
        "recall":    1.000,
        "precision": 0.900,
        "accuracy":  0.992,
        "color":     "#34d399",   # emerald-400
        "short":     "CFP",
        "data_n":    870,
        "backbone":  "EfficientNet-B0\n(ImageNet + TTA)",
    },
    "B-scan OCT\n(ResNet-18\nKermany + K-Fold)": {
        "f1":        0.900,
        "auc":       1.000,
        "recall":    1.000,
        "precision": 0.818,
        "accuracy":  0.999,
        "color":     "#fb923c",   # orange-400
        "short":     "B-scan OCT",
        "data_n":    10739,
        "backbone":  "ResNet-18\n(Kermany PT + K-Fold)",
    },
    "AS-OCT\n(EfficientNet-B0\nNoisy Student)": {
        "f1":        0.868,
        "auc":       0.970,
        "recall":    0.787,
        "precision": 0.967,
        "accuracy":  0.885,
        "color":     "#f472b6",   # pink-400
        "short":     "AS-OCT",
        "data_n":    6272,
        "backbone":  "EfficientNet-B0\n(Noisy Student)",
    },
}

LABELS  = list(MODELS.keys())
SHORTS  = [v["short"] for v in MODELS.values()]
COLORS  = [v["color"] for v in MODELS.values()]
N = len(MODELS)

# ══════════════════════════════════════════════════════════════════════════════
# FİGÜR 1 — Grouped Bar Chart  (F1 / AUC / Recall / Precision)
# ══════════════════════════════════════════════════════════════════════════════
def plot_bar_chart():
    metrics   = ["F1 Score", "ROC AUC", "Recall", "Precision"]
    keys      = ["f1", "auc", "recall", "precision"]
    bar_colors= ["#60a5fa", "#a78bfa", "#34d399", "#fb923c"]

    x    = np.arange(N)
    w    = 0.18
    offs = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    for i, (metric, key, bc) in enumerate(zip(metrics, keys, bar_colors)):
        vals = [v[key] for v in MODELS.values()]
        bars = ax.bar(x + offs[i]*w, vals, w, label=metric,
                      color=bc, alpha=0.88, zorder=3,
                      edgecolor='none', linewidth=0)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.008,
                    f"{val:.3f}", ha='center', va='bottom',
                    fontsize=7.5, color='#e2e8f0', fontweight='600')

    ax.set_xticks(x)
    ax.set_xticklabels(SHORTS, fontsize=11, color='#cbd5e1', fontweight='600')
    ax.set_ylim(0.60, 1.08)
    ax.set_ylabel("Score", fontsize=12, color='#94a3b8', labelpad=8)
    ax.tick_params(axis='y', colors='#64748b', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.yaxis.grid(True, color='#334155', linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    legend = ax.legend(fontsize=9, loc='upper left', framealpha=0.15,
                       facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1')

    ax.set_title("Üveit Karar Destek Sistemi — 5 Unimodal Model Performans Karşılaştırması\n"
                 "F1 Score · ROC AUC · Recall · Precision",
                 fontsize=13, color='#f1f5f9', fontweight='700', pad=18)

    plt.tight_layout()
    out = OUTPUT_DIR / "pub_comparison_01_bar_chart.png"
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Kaydedildi: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FİGÜR 2 — Radar / Spider Chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_radar():
    categories = ["F1 Score", "ROC AUC", "Recall", "Precision", "Accuracy"]
    keys       = ["f1", "auc", "recall", "precision", "accuracy"]
    M          = len(categories)
    angles     = np.linspace(0, 2*np.pi, M, endpoint=False).tolist()
    angles    += angles[:1]   # close loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    # Grid
    ax.set_ylim(0.55, 1.05)
    grid_vals = [0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_yticks(grid_vals)
    ax.set_yticklabels([f"{v:.1f}" for v in grid_vals],
                       fontsize=7, color='#475569')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, color='#cbd5e1', fontweight='600')
    ax.spines['polar'].set_color('#334155')
    ax.yaxis.grid(color='#334155', linewidth=0.5, linestyle='--')
    ax.xaxis.grid(color='#334155', linewidth=0.5)

    for name, meta in MODELS.items():
        vals  = [meta[k] for k in keys]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.2, linestyle='solid',
                color=meta["color"], label=meta["short"], zorder=3)
        ax.fill(angles, vals, alpha=0.08, color=meta["color"])
        # Marker dots
        ax.scatter(angles[:-1], vals[:-1], s=50, color=meta["color"],
                   zorder=5, edgecolors='white', linewidths=0.5)

    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.42, 1.15),
                       fontsize=10, framealpha=0.2, facecolor='#1e293b',
                       edgecolor='#334155', labelcolor='#e2e8f0')

    ax.set_title("Model Performans Radar Grafiği\n5 Modalite Karşılaştırması",
                 fontsize=13, color='#f1f5f9', fontweight='700',
                 pad=28, loc='center')

    plt.tight_layout()
    out = OUTPUT_DIR / "pub_comparison_02_radar.png"
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Kaydedildi: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FİGÜR 3 — Summary Table (poster için hazır)
# ══════════════════════════════════════════════════════════════════════════════
def plot_summary_table():
    fig, ax = plt.subplots(figsize=(14, 4.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.axis('off')

    col_labels = ["Modalite", "Backbone", "Eğitim Verisi", "Accuracy",
                  "Precision", "Recall", "F1 Score", "ROC AUC"]
    rows = []
    for meta in MODELS.values():
        rows.append([
            meta["short"],
            meta["backbone"].replace('\n', ' '),
            f"{meta['data_n']:,}",
            f"{meta['accuracy']:.3f}",
            f"{meta['precision']:.3f}",
            f"{meta['recall']:.3f}",
            f"{meta['f1']:.3f}",
            f"{meta['auc']:.3f}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # Header style
    header_color = "#0ea5e9"
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor('#1e293b')

    row_colors = ["#1e293b", "#263548"]
    highlight_cols = [6, 7]   # F1, AUC
    model_colors_map = [meta["color"] for meta in MODELS.values()]

    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            cell.set_facecolor(row_colors[(i-1) % 2])
            cell.set_edgecolor('#334155')
            cell.set_text_props(color='#e2e8f0')
            if j in highlight_cols:
                cell.set_text_props(color=model_colors_map[i-1], fontweight='bold')
        # Modalite sütunu rengi
        table[i, 0].set_facecolor(model_colors_map[i-1])
        table[i, 0].set_text_props(color='#0f172a', fontweight='bold')

    ax.set_title("Üveit Karar Destek Sistemi — Unimodal Model Özet Tablosu",
                 fontsize=14, color='#f1f5f9', fontweight='700',
                 pad=16, loc='center', y=0.98)

    plt.tight_layout()
    out = OUTPUT_DIR / "pub_comparison_03_summary_table.png"
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Kaydedildi: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FİGÜR 4 — F1 + AUC yan yana horizontal bar (sunum slaytı için)
# ══════════════════════════════════════════════════════════════════════════════
def plot_horizontal_bars():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f172a")

    for ax, (metric, key) in zip(axes, [("F1 Score", "f1"), ("ROC AUC", "auc")]):
        ax.set_facecolor("#1e293b")
        vals   = [v[key] for v in MODELS.values()]
        shorts = SHORTS
        y      = np.arange(N)

        bars = ax.barh(y, vals, color=COLORS, alpha=0.88,
                       height=0.55, edgecolor='none')
        ax.set_xlim(0.55, 1.08)
        ax.set_yticks(y)
        ax.set_yticklabels(shorts, fontsize=11, color='#cbd5e1', fontweight='600')
        ax.xaxis.grid(True, color='#334155', linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        ax.spines['bottom'].set_color('#334155')
        ax.tick_params(axis='x', colors='#64748b', labelsize=9)

        for bar, val in zip(bars, vals):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va='center', fontsize=10,
                    color='#f1f5f9', fontweight='700')

        ax.set_title(metric, fontsize=14, color='#f1f5f9',
                     fontweight='700', pad=12)

    fig.suptitle("5 Modalite Model Karşılaştırması",
                 fontsize=15, color='#f1f5f9', fontweight='800', y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / "pub_comparison_04_horizontal_bars.png"
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Kaydedildi: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# ANA
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("5 Model Karşılaştırma Görselleri Üretiliyor...")
    print("=" * 60)
    plot_bar_chart()
    plot_radar()
    plot_summary_table()
    plot_horizontal_bars()
    print("=" * 60)
    print(f"TAMAMLANDI! → {OUTPUT_DIR}/")
    print("=" * 60)
