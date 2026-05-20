import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Klasör oluşturma
    output_dir = "outputs/poster"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Veri hazırlığı
    data = {
        "Model": ["Slit-lamp", "CFP", "OCTA", "B-scan OCT", "AS-OCT"],
        "Accuracy": [0.9695, 0.9924, 0.8434, 0.9994, 0.9400],
        "Precision": [0.8710, 0.9000, 0.7419, 0.8182, 0.9000],
        "Recall": [0.9310, 1.0000, 0.8214, 1.0000, 0.9200],
        "F1 Score": [0.9000, 0.9474, 0.7797, 0.9000, 0.9200],
        "ROC AUC": [0.9883, 0.9982, 0.9104, 1.0000, 0.9500]
    }
    df = pd.DataFrame(data)
    
    # 3. Görsel stil ayarları
    plt.rcParams['font.family'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['text.color'] = '#111111'
    plt.rcParams['axes.labelcolor'] = '#111111'
    plt.rcParams['xtick.color'] = '#111111'
    plt.rcParams['ytick.color'] = '#111111'
    
    # Ana figür
    fig, ax_main = plt.subplots(figsize=(10, 7), facecolor='white')
    
    # 4. Ana Grafik (Dikey Grouped Bar Chart)
    models = df['Model']
    x = np.arange(len(models))
    width = 0.35
    
    f1_scores = df['F1 Score']
    auc_scores = df['ROC AUC']
    
    # Barları çizme
    bars_f1 = ax_main.bar(x - width/2, f1_scores, width, label='F1 Score', color='#003366')
    bars_auc = ax_main.bar(x + width/2, auc_scores, width, label='ROC AUC', color='#2A9D8F')
    
    # Eksen sınırları ve etiketleri
    ax_main.set_ylim(0.0, 1.15) # Etiketlere ve lejantlara yer açmak için limiti biraz artırıyoruz
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(models, fontsize=12)
    ax_main.set_title("Unimodal Expert Model Performance", fontsize=18, pad=15, weight='bold')
    
    # Grid çizgileri (sadece y ekseninde ve çok açık gri)
    ax_main.grid(axis='y', color='#D9DEE5', linestyle='-', linewidth=0.8)
    ax_main.set_axisbelow(True)
    
    # Kenarlıkları temizleme
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['left'].set_color('#111111')
    ax_main.spines['bottom'].set_color('#111111')
    
    # Barların ucuna değerleri yazdırma
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax_main.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                         f'{height:.3f}', va='bottom', ha='center', fontsize=11, color='#111111')

    add_labels(bars_f1)
    add_labels(bars_auc)
    
    # Legend - üst sağ tarafa konumlandıralım
    ax_main.legend(loc='upper right', fontsize=12, frameon=True, edgecolor='#D9DEE5', facecolor='white')
    
    plt.tight_layout()
    
    # 5. Kaydetme
    png_path = os.path.join(output_dir, "model_performance_chart.png")
    svg_path = os.path.join(output_dir, "model_performance_chart.svg")
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
    
    print("Grafikler başarıyla oluşturuldu:")
    print(f" -> {png_path}")
    print(f" -> {svg_path}")

if __name__ == "__main__":
    main()
