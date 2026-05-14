# ==============================================================================
# generate_bscan_publication_plots.py
#
# B-scan OCT modeli için makale/sunum kalitesinde görseller üretir.
# Çıktılar: outputs/bscan/plots/
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

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

PLOT_DIR = PROJECT_ROOT / "outputs" / "bscan" / "plots"
METRICS_DIR = PROJECT_ROOT / "outputs" / "bscan" / "metrics"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'primary': '#2563EB', 'secondary': '#7C3AED', 'success': '#10B981',
    'danger': '#EF4444', 'warning': '#F59E0B', 'info': '#06B6D4',
    'dark': '#1F2937',
}

def load_json(filename):
    with open(METRICS_DIR / filename, 'r') as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# 1. VERSİYON EVRİMİ (Baseline → V2 → V3 → V4-KFold)
# ─────────────────────────────────────────────────────────────
def plot_version_evolution():
    versions = ['V1\n(Baseline)', 'V2\n(Kermany PT)', 'V3\n(+Synthetic)', 'V4\n(K-Fold)']
    f1_scores = [1.000, 0.857, 1.000, 0.900]
    acc_scores = [1.000, 0.889, 1.000, 0.999]
    # V1/V3 overfitting, V2 ilk gerçekçi sonuç, V4 final güvenilir

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, scores, title, color in zip(
        axes, [f1_scores, acc_scores], ['F1 Score', 'Accuracy'],
        [COLORS['primary'], COLORS['success']]
    ):
        bars = ax.bar(versions, scores, color=color, alpha=0.85, width=0.55,
                      edgecolor='white', linewidth=2)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_ylim(0.7, 1.05)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

    # V1 ve V3'e overfitting uyarısı
    axes[0].annotate('⚠️ Overfit\n(n=9 test)', xy=(0, 1.0), fontsize=8,
                     ha='center', va='bottom', color=COLORS['danger'])
    axes[0].annotate('⚠️ Overfit\n(n=9 test)', xy=(2, 1.0), fontsize=8,
                     ha='center', va='bottom', color=COLORS['danger'])
    axes[0].annotate('✅ Güvenilir\n(n=9659 test)', xy=(3, 0.9), fontsize=8,
                     ha='center', va='bottom', color=COLORS['success'])

    fig.suptitle('B-scan OCT Model Evrimi — Veri Stratejisi ile Güvenilirlik Artışı',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "pub_01_version_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ pub_01_version_evolution.png")


# ─────────────────────────────────────────────────────────────
# 2. VERİ STRATEJİSİ GÖRSELİ (Orijinal vs Sentetik vs Kermany)
# ─────────────────────────────────────────────────────────────
def plot_data_strategy():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Sol: Orijinal veri (çok az)
    ax = axes[0]
    vals = [27, 28]
    labels = ['Üveit\n(27)', 'Normal\n(28)']
    colors_pie = [COLORS['danger'], COLORS['primary']]
    ax.pie(vals, labels=labels, colors=colors_pie, autopct='%1.0f%%',
           startangle=90, textprops={'fontsize': 11})
    ax.set_title('Orijinal Klinik Veri\n(n=55)', fontsize=13, fontweight='bold')

    # Orta: Sentetik artırma sonrası
    ax = axes[1]
    vals = [1080, 27, 9632]
    labels = ['Sentetik Üveit\n(1080)', 'Gerçek Üveit\n(27)', 'Kermany Normal\n(9632)']
    colors_bar = [COLORS['warning'], COLORS['danger'], COLORS['primary']]
    bars = ax.bar(range(3), vals, color=colors_bar, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Sentetik\nÜveit', 'Gerçek\nÜveit', 'Kermany\nNormal'], fontsize=10)
    ax.set_title('V4 Eğitim Seti\n(n=10,739)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Görüntü Sayısı')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{val:,}', ha='center', fontsize=11, fontweight='bold')

    # Sağ: K-Fold test garantisi
    ax = axes[2]
    ax.text(0.5, 0.75, '🔒', fontsize=40, ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 0.50, 'Test Bütünlüğü', fontsize=14, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 0.30, 'Sentetik veriler\nSADECE eğitimde\nkullanılır.', fontsize=11,
            ha='center', va='center', transform=ax.transAxes, style='italic')
    ax.text(0.5, 0.10, 'Test setinde yalnızca\ngerçek klinik görüntüler var.', fontsize=10,
            ha='center', va='center', transform=ax.transAxes, color=COLORS['success'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    fig.suptitle('B-scan OCT — Veri Artırma Stratejisi',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "pub_02_data_strategy.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ pub_02_data_strategy.png")


# ─────────────────────────────────────────────────────────────
# 3. V4 K-FOLD SONUÇLARI (Fold bazlı)
# ─────────────────────────────────────────────────────────────
def plot_kfold_results():
    folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    f1_scores = [1.000, 0.909, 0.857, 0.800, 1.000]
    recall_scores = [1.000, 1.000, 1.000, 1.000, 1.000]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(folds))
    w = 0.35

    bars1 = ax.bar(x - w/2, f1_scores, w, label='F1 Score',
                   color=COLORS['primary'], alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + w/2, recall_scores, w, label='Recall',
                   color=COLORS['success'], alpha=0.85, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(folds, fontsize=11)
    ax.set_ylim(0.65, 1.08)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('B-scan V4 — 5-Fold Cross Validation Sonuçları', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=10, fontweight='bold')

    # Ortalama çizgisi
    mean_f1 = np.mean(f1_scores)
    ax.axhline(y=mean_f1, color=COLORS['danger'], linestyle='--', linewidth=1.5,
               label=f'Ortalama F1 = {mean_f1:.3f}')
    ax.legend(fontsize=11)

    # Aggregated sonuçları yaz
    ax.text(0.98, 0.02, 'Aggregated (n=9659):\nF1=0.900 | Recall=1.000 | AUC=1.000',
            transform=ax.transAxes, fontsize=10, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor=COLORS['success'], alpha=0.15))

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "pub_03_kfold_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ pub_03_kfold_results.png")


# ─────────────────────────────────────────────────────────────
# 4. AGGREGATED CONFUSION MATRIX (yüzdeli)
# ─────────────────────────────────────────────────────────────
def plot_hq_confusion_matrix():
    # V4 K-Fold aggregated: 9659 test, 27 gerçek üveit, all recall=1.0
    # Precision=0.8182 → TP=27, FP=6, FN=0, TN=9626
    TP = 27
    FN = 0
    FP = 6
    TN = 9626
    cm = np.array([[TN, FP], [FN, TP]])
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

    labels = ['Normal\n(Negatif)', 'Uveitis\n(Pozitif)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('Tahmin (Predicted)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Gerçek (Actual)', fontsize=13, fontweight='bold')

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / total * 100
            color = 'white' if val > total/4 else 'black'
            text = f'{val:,}\n({pct:.2f}%)' if val > 100 else f'{val}\n({pct:.2f}%)'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color)

    ax.set_title('B-scan V4 K-Fold — Aggregated Confusion Matrix\n(5 Fold, n=9,659)',
                 fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "pub_04_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ pub_04_confusion_matrix.png")


# ─────────────────────────────────────────────────────────────
# 5. EĞİTİM STRATEJİSİ AKIŞ DİYAGRAMI (metin tabanlı)
# ─────────────────────────────────────────────────────────────
def plot_training_pipeline():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')

    steps = [
        ('1. Kermany\nPre-training', 'ResNet-18 omurgası\n4 sınıflı retinal OCT\n(108,312 görüntü)', COLORS['info']),
        ('2. Sentetik Veri\nÜretimi', '27 gerçek üveit →\n1,080 sentetik üveit\n(40× çoğaltma)', COLORS['warning']),
        ('3. Progressive\nFine-Tuning', 'Faz 1: Head (5 ep)\nFaz 2: Full backbone\n(LR: 5×10⁻⁵)', COLORS['primary']),
        ('4. 5-Fold CV\nDoğrulama', 'Gerçek veriler test\nSentetik sadece train\n(Data Leakage ✗)', COLORS['success']),
        ('5. Final\nSonuçlar', 'F1: 0.900\nRecall: 1.000\nAUC: 1.000', COLORS['danger']),
    ]

    for i, (title, desc, color) in enumerate(steps):
        x = 0.1 + i * 0.18
        # Kutu
        rect = plt.Rectangle((x, 0.25), 0.14, 0.5, linewidth=2,
                              edgecolor=color, facecolor=color, alpha=0.15,
                              transform=ax.transAxes)
        ax.add_patch(rect)
        # Başlık
        ax.text(x + 0.07, 0.65, title, fontsize=11, fontweight='bold',
                ha='center', va='center', transform=ax.transAxes, color=color)
        # Açıklama
        ax.text(x + 0.07, 0.42, desc, fontsize=9, ha='center', va='center',
                transform=ax.transAxes, style='italic')
        # Ok
        if i < len(steps) - 1:
            ax.annotate('', xy=(x + 0.18, 0.5), xytext=(x + 0.14, 0.5),
                        arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2),
                        xycoords='axes fraction', textcoords='axes fraction')

    ax.set_title('B-scan OCT — Eğitim Pipeline\'ı (Transfer Learning + Synthetic Augmentation)',
                 fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "pub_05_training_pipeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ pub_05_training_pipeline.png")


# ─────────────────────────────────────────────────────────────
# 6. FINAL METRİK KARTI
# ─────────────────────────────────────────────────────────────
def plot_final_metrics_card():
    metrics = {
        'Accuracy': 0.9994,
        'Precision': 0.8182,
        'Recall': 1.0000,
        'F1 Score': 0.9000,
        'ROC AUC': 1.0000,
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLORS['primary'], COLORS['warning'], COLORS['success'],
              COLORS['danger'], COLORS['secondary']]

    bars = ax.barh(list(metrics.keys()), list(metrics.values()), color=colors,
                   alpha=0.85, edgecolor='white', height=0.6)

    ax.set_xlim(0.7, 1.05)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('B-scan V4 K-Fold — Final Test Metrikleri (Aggregated, n=9,659)',
                 fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', ha='left', va='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "pub_06_final_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ pub_06_final_metrics.png")


# ─────────────────────────────────────────────────────────────
# ANA
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("B-scan OCT — Makale & Sunum Kalitesinde Görseller")
    print("=" * 60)
    print(f"Çıktı klasörü: {PLOT_DIR}\n")

    plot_version_evolution()
    plot_data_strategy()
    plot_kfold_results()
    plot_hq_confusion_matrix()
    plot_training_pipeline()
    plot_final_metrics_card()

    print(f"\n{'='*60}")
    print(f"✅ Toplam 6 yüksek kaliteli görsel üretildi!")
    print(f"📁 Konum: {PLOT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
