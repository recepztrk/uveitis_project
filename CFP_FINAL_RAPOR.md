# CFP Modülü — Final Teknik Rapor

**Proje:** Üveit Karar Destek Sistemi  
**Modalite:** Color Fundus Photography (CFP / Renkli Fundus Fotoğrafı)  
**Son Güncelleme:** 11 Mayıs 2026  
**Final Model:** EfficientNet-B0 + TTA (5×) + Optimal Threshold (0.68)

---

## 1. Giriş ve Amaç

Color Fundus Photography (CFP), retina ve optik disk yapılarının renkli fotoğraflarını üreten, oftalmolojide en yaygın kullanılan görüntüleme yöntemlerinden biridir. Üveit hastalarında retinal vaskülit, koroidit lezyonları, optik disk ödemi, maküler ödem ve chorioretinal skarlar CFP ile tespit edilebilir.

Bu modülde, CFP görüntülerinden üveit hastalığını otomatik olarak ayırt eden bir ikili sınıflandırma modeli (Üveit vs Non-Üveit) geliştirilmiştir. Modelin en büyük zorluğu, **aşırı sınıf dengesizliği** (Üveit:Non-Üveit = 1:12.8) ve **çok az üveit verisi** (toplam 63 görüntü) ile çalışmaktır. Test Time Augmentation (TTA) ile son optimizasyon yapılarak F1 Score 0.783'ten **0.947'ye** yükseltilmiştir.

---

## 2. Veri Seti

### 2.1 Veri Kaynakları

İki farklı kaynaktan toplanan çok çeşitli retinal patolojiler içermektedir:

| Kaynak | Görüntü Sayısı | İçerik | Üveit |
|--------|:--------------:|--------|:-----:|
| 1000 Fundus Images | 563 | Normal, DR, makulopati, RP, myopia, lazer vb. | 22 (VKH, bietti) |
| RFMiD2.0 | 307 | Normal (WNL), CRS, çeşitli retinal hastalıklar | 41 (CRS üveiti) |
| **TOPLAM** | **870** | **22 farklı retinal sınıf** | **63** |

### 2.2 Sınıf Dağılımı (İkili Problem)

| Sınıf | Sayı | Oran |
|-------|:----:|:----:|
| Üveit | 63 | %7.2 |
| Non-Üveit | 807 | %92.8 |
| **Dengesizlik** | — | **1:12.8** |

Bu, projenizdeki 4 modalite arasındaki **en aşırı dengesizlik** oranıdır.

### 2.3 Non-Üveit Sınıfının Zenginliği

Non-üveit sınıfı, 20'den fazla farklı retinal patolojiyi içermektedir:

| Kategori | Alt Sınıflar | Amaç |
|----------|-------------|------|
| Normal | Normal, WNL (Within Normal Limits) | Sağlıklı kontrol |
| Diyabetik Retinopati | DR1, DR2, DR3 | Vasküler lezyon ayrımı |
| Vasküler Oklüzyon | BRVO, CRVO, RAO | Damar tıkanıklığı ayrımı |
| Dejeneratif | Makulopati, ERM, MH, RP | Dejenerasyon ayrımı |
| Diğer | CSCR, Myopia, Lazer, Periferal dejen. | Çeşitli patoloji ayrımı |

Bu zenginlik, modelin "sadece normal vs üveit" değil, **"üveit vs tüm diğer retinal patolojiler"** ayrımını öğrenmesini sağlar — klinik olarak çok daha gerçekçi bir senaryo.

### 2.4 Train / Validation / Test Bölünmesi

| Split | Toplam | Üveit | Non-Üveit |
|-------|:------:|:-----:|:---------:|
| Train | 609 | 45 (%7.4) | 564 (%92.6) |
| Val | 130 | 9 (%6.9) | 121 (%93.1) |
| Test | 131 | 9 (%6.9) | 122 (%93.1) |

---

## 3. Veri Ön İşleme

### 3.1 Görüntü Hazırlama

| İşlem | Parametre |
|-------|-----------|
| Format | RGB dönüşümü |
| Boyut (Eğitim) | RandomResizedCrop(224, scale=(0.85, 1.0)) |
| Boyut (Test) | Resize(224×224) |
| Normalizasyon | ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

### 3.2 Eğitim Augmentation

| Teknik | Parametre | Amaç |
|--------|-----------|------|
| `RandomResizedCrop` | size=224, scale=(0.85, 1.0) | Hafif zoom invariansı |
| `RandomHorizontalFlip` | p=0.5 | Yatay simetri (fundus için uygun) |
| `RandomRotation` | ±10° | Açısal değişkenlik |
| `ColorJitter` | brightness=0.15, contrast=0.15 | Aydınlatma koşulları |
| `RandomErasing` | p=0.15, scale=(0.02, 0.08) | Kısmi oklüzyon dayanıklılığı |

**Not:** `VerticalFlip` fundus görüntülerinde anatomik oryantasyonu bozacağı için kullanılmamıştır.

### 3.3 Metadata Dosyaları

| Dosya | İçerik |
|-------|--------|
| `metadata/cfp_labels.csv` | 870 satır (image_id, filepath, source_dataset, source_class, binary_label) |
| `metadata/cfp_split.csv` | Train/Val/Test split bilgisi |

---

## 4. Model Mimarisi

### 4.1 Backbone

| Parametre | Değer |
|-----------|-------|
| Model | EfficientNet-B0 |
| Pre-training | ImageNet |
| Son Katman | `classifier[1] → Linear(1280, 1)` |
| Aktivasyon | Sigmoid (inference), BCEWithLogitsLoss (eğitim) |

### 4.2 Hiperparametreler

| Parametre | Değer |
|-----------|-------|
| Optimizer | AdamW (weight_decay=1×10⁻⁴) |
| Learning Rate | 1×10⁻⁴ |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Batch Size | 16 |
| Max Epoch | 30 |
| Early Stopping | Patience = 10 (en iyi epoch: 11, toplam: 21) |
| Label Smoothing | 0.1 |
| Loss | BCEWithLogitsLoss + pos_weight |
| pos_weight | ~12.5 (564/45) |
| Sampling | WeightedRandomSampler |

### 4.3 Çift Dengeleme Stratejisi

CFP'deki 1:12.8 dengesizlik, iki mekanizma ile yönetilmiştir:

1. **WeightedRandomSampler:** Eğitim batch'lerinde azınlık sınıfı (üveit) aşırı örneklenerek, model her batch'te dengeli sayıda örnek görür.
2. **pos_weight:** Loss hesabında pozitif örneklere ~12.5× daha fazla ağırlık verilerek, gradient güncellemelerinde üveit örneklerinin etkisi artırılır.

---

## 5. Eğitim Süreci

### 5.1 Eğitim Özeti

- Model 21 epoch boyunca eğitilmiştir (30 epoch hedeflenmişti, 21'de early stop)
- En iyi model: **Epoch 11** (Val F1 = 0.818, Val AUC = 0.995)
- LR scheduler epoch 14'te öğrenme oranını yarıya düşürmüştür

### 5.2 Önemli Gözlemler

- Val AUC, epoch 3'ten itibaren 0.99+ bandında stabilize olmuştur
- Val F1 daha değişken (0.60-0.82 arası) çünkü sadece 9 val üveit örneği var
- Early stopping, overfitting'i başarıyla önlemiştir

---

## 6. Test Time Augmentation (TTA)

### 6.1 Yöntem

OCTA modelinde başarıyla uygulanan TTA stratejisi CFP'ye uyarlanmıştır:

| # | Augmentation | Açıklama |
|---|-------------|----------|
| 1 | Orijinal | Resize(224×224), normalize |
| 2 | Yatay çevirme | HorizontalFlip(p=1.0) |
| 3 | +5° döndürme | RandomRotation(5, 5) |
| 4 | -5° döndürme | RandomRotation(-5, -5) |
| 5 | Merkez kırpma | Resize(256) → CenterCrop(224) |

**Final olasılık = (p₁ + p₂ + p₃ + p₄ + p₅) / 5**

### 6.2 Optimal Threshold Kalibrasyonu

Validation seti üzerinde F1 Score'u maksimize eden eşik aranmıştır:

- **Default eşik (0.50):** Val F1 = 0.692
- **Optimal eşik (0.68):** Val F1 = **0.947**

Bu eşik daha sonra test setine uygulanmıştır.

### 6.3 Neden Bu Kadar Etkili?

Baseline'da 5 FP'den 3'ünün olasılığı 0.501-0.506 aralığındaydı — yani eşiğin kılpayı üstünde. TTA, bu sınır vakaların olasılıklarını stabilize ederek gerçek değerlerine yaklaştırdı. Eşiğin 0.68'e çekilmesi, geri kalan belirsiz vakaları da eledi.

---

## 7. Final Test Sonuçları

### 7.1 Ana Metrikler (Test Seti, n=131, TTA + t=0.68)

| Metrik | Değer |
|--------|:-----:|
| **Accuracy** | **99.24%** |
| **Precision** | **90.00%** |
| **Recall (Sensitivity)** | **100.00%** |
| **F1 Score** | **0.9474** |
| **ROC AUC** | **0.9982** |

### 7.2 Confusion Matrix (TTA + t=0.68)

|  | Tahmin: Non-Üveit | Tahmin: Üveit |
|--|:---:|:---:|
| **Gerçek: Non-Üveit (n=122)** | TN = 121 (92.4%) | FP = 1 (0.8%) |
| **Gerçek: Üveit (n=9)** | FN = 0 (0.0%) | TP = 9 (6.9%) |

- **Specificity:** 121/122 = **99.18%**
- **NPV:** 121/121 = **100.00%**

### 7.3 Baseline → TTA İyileşme

| Metrik | Baseline | TTA (Final) | Delta |
|--------|:--------:|:-----------:|:-----:|
| Accuracy | 96.18% | 99.24% | **+3.06%** |
| Precision | 64.29% | 90.00% | **+25.71%** |
| Recall | 100.00% | 100.00% | Sabit |
| F1 | 0.7826 | 0.9474 | **+0.1648** |
| AUC | 1.0000 | 0.9982 | -0.0018 |

### 7.4 Hata Analizi

**Baseline'daki 5 FP → TTA sonrası 1 FP:**

| FP | Gerçek Sınıf | Baseline Prob | TTA Sonrası |
|:--:|:---:|:---:|:---:|
| 1 | rd | 0.506 | Elendi (< 0.68) |
| 2 | wnl | 0.503 | Elendi (< 0.68) |
| 3 | dr2 | 0.501 | Elendi (< 0.68) |
| 4 | cscr | 0.561 | Elendi (< 0.68) |
| 5 | wnl | 0.798 | **Kaldı** (> 0.68) |

Kalan tek FP (prob=0.80), model tarafından yüksek güvenle üveit olarak etiketlenmiş bir normal (WNL) görüntüdür. Bu, modelin yapısal olarak öğrenmiş olduğu bir yanlış eşleşme olup, yalnızca yeniden eğitim veya ek veri ile düzeltilebilir.

### 7.5 Kaynak Bazlı Performans (Baseline)

| Kaynak | n Test | Üveit n | F1 | Precision | Recall |
|--------|:------:|:-------:|:--:|:---------:|:------:|
| 1000images | 85 | 2 | 0.571 | 0.400 | 1.000 |
| RFMiD2.0 | 46 | 7 | 0.875 | 0.778 | 1.000 |
| **Toplam** | **131** | **9** | **0.783** | **0.643** | **1.000** |

RFMiD2.0 kaynağında daha yüksek performans gözlenmiştir. Bu, RFMiD2.0'daki üveit görüntülerinin (CRS) daha belirgin patolojik bulgular içermesinden kaynaklanmaktadır.

---

## 8. Yorumlanabilirlik (Grad-CAM)

5 adet Grad-CAM ısı haritası üretilmiştir:

| # | Görüntü | Durum | Gözlem |
|---|---------|:-----:|--------|
| 1 | rfmimd2_uv_crs_0036 | TP | Model, CRS lezyonlarına odaklanmış |
| 2 | rfmimd2_uv_crs_0035 | TP | Peripapiller bölgedeki inflamasyona dikkat |
| 3 | cfp1000_nonuv_normal_0014 | TN | Dağınık, düşük aktivasyon |
| 4 | cfp1000_nonuv_peripheral_degeneration | TN | Periferik bölgeye yanlış odaklanma riski |
| 5 | cfp1000_nonuv_rd_0031 | FP | Retinal dekolmanı üveit olarak yorumlamış |

---

## 9. Sınırlılıklar

1. **Çok Az Üveit Verisi:** Test setinde sadece 9 üveit görüntüsü bulunmaktadır. Her 1 hatalı tahmin metrikleri önemli ölçüde etkiler (1 FP: Precision %90 vs %100).

2. **Kaynak Heterojenitesi:** İki farklı veri seti farklı kamera/çözünürlük/populasyon özelliklerine sahiptir. Model, kaynak bazlı önyargıya sahip olabilir.

3. **Üveit Alt Tipi Çeşitliliği Düşük:** Üveit verileri ağırlıklı olarak CRS (chorioretinal scar) ve VKH'den oluşmaktadır. Anterior üveit veya akut posterior üveit gibi bulgular temsil edilmemektedir.

4. **K-Fold Yapılmadı:** Performans tek split'e dayalıdır. 9 pozitif örnekle K-Fold anlamlı sonuç vermeyeceğinden uygulanmamıştır.

5. **TTA'nın AUC Etkisi:** TTA sonrası AUC 1.000'den 0.998'e düşmüştür (ihmal edilebilir). Bu, olasılık dağılımının hafifçe değişmesinden kaynaklanır.

---

## 10. Üretilen Çıktılar ve Dosya Haritası

### 10.1 Makale/Sunum Görselleri (`outputs/cfp/plots/`)

| Dosya | İçerik |
|-------|--------|
| `pub_01_data_distribution.png` | Kaynak + ikili sınıf + split dağılımı |
| `pub_02_training_curves.png` | Loss, F1, AUC eğitim eğrileri (21 epoch) |
| `pub_03_tta_impact.png` | Baseline vs TTA before/after karşılaştırma |
| `pub_04_confusion_matrix.png` | TTA sonrası yüzdeli CM (n=131) |
| `pub_05_roc_curve.png` | Gölgeli ROC eğrisi (AUC=0.998) |
| `pub_06_source_analysis.png` | Kaynak bazlı performans karşılaştırma |
| `pub_07_final_metrics.png` | Final metrik kartı |

### 10.2 Ham Metrikler (`outputs/cfp/metrics/`)

| Dosya | İçerik |
|-------|--------|
| `cfp_tta_test_metrics.json` | TTA final metrikleri + karşılaştırma |
| `cfp_test_metrics.json` | Baseline test metrikleri + kaynak analizi + hata analizi |
| `cfp_train_history.json` | 21 epoch eğitim tarihçesi |
| `cfp_test_predictions.json` | Örnek bazlı tahmin olasılıkları |
| `cfp_tta_comparison.png` | TTA karşılaştırma grafiği |
| `cfp_tta_threshold_curve.png` | F1 vs Threshold eğrisi |

### 10.3 Grad-CAM (`outputs/cfp/gradcam/`)

5 adet Grad-CAM ısı haritası (2 TP + 2 TN + 1 FP).

### 10.4 Model Ağırlıkları (`outputs/cfp/models/`)

| Dosya | Boyut | Açıklama |
|-------|:-----:|----------|
| `cfp_efficientnetb0_best.pth` | ~16 MB | Final model (TTA bu ağırlıkları kullanır) |

---

## 11. Kaynak Kod Referansları

| Dosya | Amaç |
|-------|------|
| `training/train_cfp_baseline.py` | Ana eğitim scripti |
| `evaluation/evaluate_cfp_tta.py` | TTA + Optimal Threshold değerlendirmesi |
| `evaluation/gradcam_cfp.py` | Grad-CAM ısı haritası üretimi |
| `evaluation/generate_cfp_publication_plots.py` | Makale kalitesinde görsel üretimi |
| `preprocessing/cfp_labels_build.py` | Etiket CSV üretimi |
| `preprocessing/cfp_split_build.py` | Stratified split üretimi |
| `src/cfp_dataset.py` | PyTorch Dataset sınıfı |

---

## 12. Sonuç

CFP modülü, 870 görüntülük çok kaynaklı ve aşırı dengesiz bir veri seti üzerinde eğitilmiş, TTA ve optimal threshold kalibrasyonu ile **F1 Score 0.947** ve **Recall %100** performansına ulaşmış bir üveit sınıflandırma modelidir.

Modelin en dikkat çekici özelliği, **%100 sensitivity** (sıfır false negative) ile hiçbir üveit vakasını kaçırmamasıdır. TTA uygulaması, Precision'ı %64'ten %90'a çıkararak yanlış alarm oranını 5'ten 1'e düşürmüştür. Bu iyileştirme, hiçbir yeniden eğitim gerektirmeden, yalnızca post-processing tekniğiyle elde edilmiştir.

Model, projenin Faz-2 aşamasında (Multimodal Decision Fusion) diğer modalitelerle birleştirilmeye hazırdır.

---

*Bu rapor, CFP modülünün makale yazım sürecinde ilgili bölüm (Materials & Methods, Results) için temel kaynak olarak kullanılmak üzere hazırlanmıştır.*
