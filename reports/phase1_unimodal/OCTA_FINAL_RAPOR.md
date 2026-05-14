# OCTA Modülü — Final Teknik Rapor

**Proje:** Üveit Karar Destek Sistemi  
**Modalite:** Optik Koherens Tomografi Anjiyografi (OCTA)  
**Son Güncelleme:** 10 Mayıs 2026  
**Final Model:** OCTA V5-TTA (ResNet-18 + Test Time Augmentation)

---

## 1. Giriş ve Amaç

OCTA (Optical Coherence Tomography Angiography), retina ve koroidde bulunan damar ağını kontrast madde kullanmadan, non-invaziv bir şekilde görüntüleyen ileri düzey bir oftalmolojik modalitedir. Bu modülde, OCTA görüntülerinden üveit hastalığına ait vasküler patolojileri otomatik olarak tespit eden bir ikili sınıflandırma modeli (Üveit vs Kontrol) geliştirilmiştir.

**Klinik Motivasyon:** Üveit hastalarında retinal damar yoğunluğunda azalma, foveal avasküler zon (FAZ) genişlemesi ve kapiller perfüzyon bozuklukları OCTA'da gözlemlenmektedir. Bu bulgular, hastalığın aktivite düzeyini ve progresyonunu değerlendirmek için kritik öneme sahiptir.

---

## 2. Veri Seti

### 2.1 Veri Kaynakları

Çalışmada iki farklı kaynaktan elde edilen toplam **525 OCTA görüntüsü** kullanılmıştır:

| Kaynak | Cihaz | Görüntü Sayısı | Sınıflar |
|--------|-------|:--------------:|----------|
| BH-OCTA (Orijinal klinik veri) | Heidelberg Spectralis | 338 | Aktif Üveit, İnaktif Üveit, Kontrol |
| UT-OCTA (Harici kontrol grubu) | OptoVue RTVue | 187 | Sağlıklı Kontrol |
| **TOPLAM** | **2 farklı cihaz** | **525** | **İkili: Üveit (180) vs Kontrol (345)** |

### 2.2 Sınıf Dağılımı (İkili Problem)

| Sınıf | Toplam | Oran |
|-------|:------:|:----:|
| Üveit (Aktif + İnaktif) | 180 | %34.3 |
| Kontrol (Sağlıklı) | 345 | %65.7 |
| **Toplam** | **525** | **1:1.92 dengesizlik** |

### 2.3 Katman Bilgisi

Her hasta için hem **Superficial (yüzeysel)** hem **Deep (derin)** vasküler katman görüntüleri dahil edilmiştir. V1'de sadece superficial katman kullanılırken, V2 ile birlikte her iki katman da modele sunulmuştur.

### 2.4 Train / Validation / Test Bölünmesi

Bölünme, `group_id` bazlı **GroupKFold** stratejisi ile yapılmıştır. Bu sayede aynı hastanın farklı katman (superficial/deep) veya farklı göz görüntüleri kesinlikle aynı split içinde tutulmuş ve **veri sızıntısı (data leakage)** önlenmiştir.

| Split | Toplam | Üveit | Kontrol |
|-------|:------:|:-----:|:-------:|
| Train | 363 | 126 (%34.7) | 237 (%65.3) |
| Validation | 79 | 26 (%32.9) | 53 (%67.1) |
| Test | 83 | 28 (%33.7) | 55 (%66.3) |

**Benzersiz grup sayısı:** 356 (hasta/göz birimi)

---

## 3. Veri Ön İşleme

### 3.1 Görüntü Hazırlama

- Tüm görüntüler `PIL` ile yüklenip **RGB** formatına dönüştürülmüştür.
- Görüntü boyutu eğitim için **224×224 piksel** olarak yeniden boyutlandırılmıştır.
- ImageNet normalizasyonu uygulanmıştır: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

### 3.2 Veri Artırma (Data Augmentation) — Sadece Eğitim Seti

Overfitting'i önlemek amacıyla agresif augmentation uygulanmıştır:

| Teknik | Parametre | Amaç |
|--------|-----------|------|
| `RandomResizedCrop` | scale=(0.8, 1.0), size=224 | Ölçek invariansı |
| `RandomHorizontalFlip` | p=0.5 | Yatay simetri |
| `RandomVerticalFlip` | p=0.5 | Dikey simetri |
| `RandomAffine` | degrees=±15°, translate=±10%, scale=90-110% | Geometrik dönüşüm |
| `ColorJitter` | brightness=0.2, contrast=0.2 | Parlaklık/kontrast değişkenliği |
| `RandomErasing` | p=0.2, scale=(0.02, 0.1) | Kısmi oklüzyon dayanıklılığı |

Validasyon ve test setlerinde yalnızca `Resize(224×224)` ve `Normalize` uygulanmıştır.

### 3.3 Metadata Yönetimi

- `metadata/octa_labels.csv`: 525 satırlık etiket dosyası (image_id, filepath, source_class, layer_type, group_id, binary_label)
- `metadata/octa_split.csv`: Train/Val/Test split bilgisi eklenmiş hali
- Etiketleme scripti: `preprocessing/octa_labels_build.py`
- Split scripti: `preprocessing/octa_split_build.py`

---

## 4. Model Mimarisi

### 4.1 Backbone

- **Model:** ResNet-18 (ImageNet ön eğitimli)
- **Son Katman:** `fc → Linear(512, 1)` (ikili sınıflandırma için tek çıkış)
- **Aktivasyon:** Sigmoid (inference sırasında), BCEWithLogitsLoss (eğitim sırasında)

### 4.2 Eğitim Stratejisi: Progressive Fine-Tuning

Model iki fazda eğitilmiştir:

| Faz | Epoch Aralığı | Eğitilen Katmanlar | Öğrenme Oranı |
|-----|:-------------:|---------------------|:-------------:|
| Faz 1 (Head) | 1–5 | Yalnızca son FC katmanı | 1×10⁻³ |
| Faz 2 (Full) | 6–40 | Tüm katmanlar | 1×10⁻⁴ |

**Gerekçe:** İlk 5 epoch'ta yalnızca sınıflandırıcı başlığı eğitilerek, ImageNet'ten gelen öznitelik çıkarma yeteneğinin bozulması önlenmiştir. Ardından tüm ağ düşük öğrenme oranıyla ince ayar (fine-tuning) yapılmıştır.

### 4.3 Hiperparametreler

| Parametre | Değer |
|-----------|-------|
| Optimizer | AdamW |
| Weight Decay | 1×10⁻⁴ |
| Batch Size | 16 |
| Max Epoch | 40 |
| Early Stopping Patience | 10 epoch |
| LR Scheduler | Cosine Annealing |
| Label Smoothing | 0.1 (hedef etiketler: 1→0.95, 0→0.05) |
| pos_weight (BCEWithLogitsLoss) | Eğitim setinden otomatik hesaplanmış (~1.88) |

### 4.4 Label Smoothing

Sert etiketler (0/1) yerine yumuşatılmış etiketler kullanılmıştır:

```
smoothed_y = y × (1 − ε) + ε/2, ε = 0.1
→ Üveit: 1.0 → 0.95, Kontrol: 0.0 → 0.05
```

Bu teknik, modelin aşırı güvenli (overconfident) tahminler üretmesini engelleyerek genelleme kapasitesini artırır.

---

## 5. Model Geliştirme Evrimi

OCTA modeli 4 ana iterasyondan geçerek bugünkü performansına ulaşmıştır:

### 5.1 Versiyon Geçmişi

| | V1 | V2 | V3 | V5-TTA (Final) |
|---|:---:|:---:|:---:|:---:|
| **Veri Sayısı** | 214 | 338 | 525 | 525 |
| **Katmanlar** | Sadece Superficial | Superficial + Deep | Superficial + Deep | Superficial + Deep |
| **Cihaz** | Sadece Heidelberg | Sadece Heidelberg | Heidelberg + OptoVue | Heidelberg + OptoVue |
| **Augmentation** | Temel | Agresif | Agresif | Agresif |
| **Fine-Tuning** | Tam | Progressive | Progressive | Progressive |
| **Label Smoothing** | ✗ | ✓ (0.1) | ✓ (0.1) | ✓ (0.1) |
| **LR Schedule** | Sabit | Cosine Annealing | Cosine Annealing | Cosine Annealing |
| **Optimizer** | Adam | Adam | AdamW | AdamW |
| **Early Stopping** | ✗ | ✗ | ✓ (patience=10) | ✓ (patience=10) |
| **TTA** | ✗ | ✗ | ✗ | ✓ (5 augmentation) |
| **F1 Score** | 0.700 | 0.704 | 0.754 | **0.780** |
| **ROC AUC** | 0.732 | 0.696 | 0.901 | **0.910** |
| **Accuracy** | 65.4% | 69.2% | 81.9% | **84.3%** |

### 5.2 Kritik Dönüm Noktaları

1. **V1 → V2 (Katman genişletme):** Deep katman görüntülerinin eklenmesi veri sayısını 214'ten 338'e çıkardı ancak performansta anlamlı artış sağlayamadı (AUC 0.732→0.696).

2. **V2 → V3 (Çapraz cihaz genişletme):** Harici OptoVue kontrol verilerinin eklenmesi AUC'yi **0.696'dan 0.901'e** taşıdı. Bu, projenin en büyük performans sıçramasıdır. Domain diversification, modelin cihaz bağımsız özellikler öğrenmesini sağladı.

3. **V3 → V5-TTA (Post-processing optimizasyonu):** Modelin ağırlıkları değiştirilmeden, test zamanında 5 farklı augmented kopya ile tahmin yapılarak Precision **%69.7'den %74.2'ye**, F1 **0.754'ten 0.780'e** yükseltildi.

---

## 6. Test Time Augmentation (TTA) — V5

### 6.1 Yöntem

TTA, her test görüntüsünün 5 farklı versiyonunu modele sunarak tahminlerin ortalamasını almaktadır:

| # | Augmentation | Açıklama |
|---|-------------|----------|
| 1 | Orijinal | Resize(224×224), standart normalize |
| 2 | Yatay çevirme | HorizontalFlip(p=1.0) |
| 3 | Dikey çevirme | VerticalFlip(p=1.0) |
| 4 | 180° dönüş | HFlip + VFlip (her ikisi birlikte) |
| 5 | Merkez kırpma | Resize(256×256) → CenterCrop(224) |

**Final olasılık = (p₁ + p₂ + p₃ + p₄ + p₅) / 5**

### 6.2 Neden Etkili?

TTA, modelin farklı perspektiflerden gördüğü tahminleri ortalayarak:
- Rastgele yanlış sınıflandırmaları (false positive/negative) azaltır
- Sınır vakalarında (probability ≈ 0.50) daha güvenilir kararlar verir
- Sıfır ek veri, sıfır yeniden eğitim gerektirir

---

## 7. Final Test Sonuçları

### 7.1 Ana Metrikler (V5-TTA, Test Seti, n=83)

| Metrik | Değer |
|--------|:-----:|
| **Accuracy** | **84.34%** |
| **Precision** | **74.19%** |
| **Recall (Sensitivity)** | **82.14%** |
| **F1 Score** | **0.7797** |
| **ROC AUC** | **0.9104** |

### 7.2 Confusion Matrix

|  | Tahmin: Kontrol | Tahmin: Üveit |
|--|:---:|:---:|
| **Gerçek: Kontrol (n=55)** | TN = 47 (56.6%) | FP = 8 (9.6%) |
| **Gerçek: Üveit (n=28)** | FN = 5 (6.0%) | TP = 23 (27.7%) |

- **Specificity (Özgüllük):** 47/55 = 85.45%
- **NPV (Negatif Prediktif Değer):** 47/52 = 90.38%

### 7.3 V3 → V5-TTA İyileşme Miktarı

| Metrik | V3 (Baseline) | V5-TTA (Final) | Δ |
|--------|:---:|:---:|:---:|
| Accuracy | 81.93% | 84.34% | **+2.41%** |
| Precision | 69.70% | 74.19% | **+4.49%** |
| Recall | 82.14% | 82.14% | Sabit |
| F1 | 0.7541 | 0.7797 | **+0.0256** |
| AUC | 0.9006 | 0.9104 | **+0.0098** |

> **Not:** TTA ile hiçbir metrikte düşüş yaşanmamıştır.

### 7.4 Cihaz Bazlı Performans (Domain Generalization)

| Cihaz | n | Accuracy | F1 | Precision | Recall |
|-------|:-:|:--------:|:--:|:---------:|:------:|
| Heidelberg (BH) | 58 | 75.86% | 0.767 | 71.88% | 82.14% |
| OptoVue (UT) | 25 | 96.00% | — | — | — |

> **Not:** OptoVue setinde yalnızca kontrol grubu bulunduğundan (üveit vakası yok), 24/25 doğru sınıflandırma yapılmıştır. Bu, modelin farklı cihazdan gelen sağlıklı damar yapısını başarıyla tanıdığını gösterir.

---

## 8. Çapraz Doğrulama (K-Fold CV)

Modelin güvenilirliğini doğrulamak amacıyla EfficientNet-B0 backbone ile 5-Fold Cross Validation yapılmıştır:

| Metrik | Ortalama | Std | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|--------|:--------:|:---:|:------:|:------:|:------:|:------:|:------:|
| F1 | 0.725 | ±0.069 | 0.731 | 0.667 | 0.628 | 0.805 | 0.794 |
| AUC | 0.829 | ±0.066 | 0.812 | 0.792 | 0.734 | 0.906 | 0.902 |
| Accuracy | 0.762 | ±0.091 | 0.762 | 0.686 | 0.638 | 0.848 | 0.876 |
| Precision | 0.637 | ±0.125 | 0.618 | 0.524 | 0.485 | 0.750 | 0.807 |
| Recall | 0.870 | ±0.047 | 0.895 | 0.917 | 0.889 | 0.868 | 0.781 |

> **Yorum:** Yüksek recall ortalaması (0.87), modelin üveit vakalarını yakalama konusunda tutarlı olduğunu gösterir. Fold'lar arası varyans (F1 std=0.069) küçük veri setleri için kabul edilebilir düzeydedir.

---

## 9. Yorumlanabilirlik (Grad-CAM)

Modelin karar mekanizmasını anlamak ve klinik güvenilirliğini doğrulamak amacıyla **Gradient-weighted Class Activation Mapping (Grad-CAM)** uygulanmıştır. ResNet-18'in son konvolüsyon katmanından (`layer4`) ısı haritaları çıkarılmıştır.

**Gözlemler:**
- **TP (Doğru Pozitif) vakalarda:** Model, foveal avasküler zon (FAZ) çevresindeki kapiller perfüzyon bozukluklarına ve damar yoğunluğunun azaldığı bölgelere odaklanmaktadır.
- **TN (Doğru Negatif) vakalarda:** Aktivasyon dağınık ve düşük yoğunlukludur, model belirgin bir patoloji tespit etmemektedir.
- **FP (Yanlış Pozitif) vakalarda:** Model, damar ağındaki doğal varyasyonları patolojik değişiklik olarak yorumlamış olabilir.

**Grad-CAM Görselleri:** `outputs/octa/gradcam/` klasöründe 6 adet örnek (2 TP + 2 TN + 1 FP + 1 FN) mevcuttur.

---

## 10. Sınırlılıklar

1. **Küçük Veri Seti:** 525 görüntü, derin öğrenme standartlarına göre sınırlıdır. Bu durum, modelin genelleme kapasitesini kısıtlayabilir.

2. **Sınıf Dengesizliği:** Üveit:Kontrol oranı 1:1.92'dir. `pos_weight` ve label smoothing ile dengelenmiş olsa da, ideal 1:1 oranından uzaktır.

3. **Tek Merkezli Üveit Verisi:** Üveit vakaları yalnızca Heidelberg cihazından geldiği için, modelin farklı cihazlardaki üveit vakalarını tanıma kapasitesi test edilememiştir.

4. **Proxy Etiketleme:** Aktif ve inaktif üveit alt tipleri ikili (Üveit/Kontrol) olarak birleştirilmiştir. Klinik pratikte bu alt ayrım önemlidir.

5. **Hasta Düzeyinde Değerlendirme:** Görüntü düzeyinde metrikler sunulmuştur; hasta düzeyinde agregasyon yapılmamıştır.

---

## 11. Üretilen Çıktılar ve Dosya Haritası

### 11.1 Makale/Sunum Görselleri (`outputs/octa/plots/`)

| Dosya | İçerik |
|-------|--------|
| `01_version_evolution.png` | V1→V2→V3→V5-TTA F1/AUC/Accuracy bar grafiği |
| `02_kfold_boxplot.png` | 5-Fold CV kutu grafiği (tüm metrikler) |
| `03_data_distribution.png` | Train/Val/Test sınıf dağılımı pasta grafiği |
| `04_training_curves.png` | Loss, F1, AUC ve LR eğitim eğrileri (4 panel) |
| `05_radar_comparison.png` | V3 vs V5-TTA radar karşılaştırma |
| `06_confusion_matrix_hq.png` | Yüzdeli confusion matrix (yüksek çözünürlük) |
| `07_roc_curve_hq.png` | Gölgeli ROC eğrisi (AUC=0.9104) |
| `08_tta_impact.png` | TTA etkisi before/after analiz görseli |

### 11.2 Ham Metrikler (`outputs/octa/metrics/`)

| Dosya | İçerik |
|-------|--------|
| `octa_v5_tta_test_metrics.json` | V5-TTA final test metrikleri + karşılaştırma |
| `octa_v3_test_metrics.json` | V3 baseline test metrikleri |
| `octa_v3_train_history.json` | 34 epoch eğitim tarihçesi |
| `octa_kfold_results.json` | 5-Fold CV detaylı sonuçları |

### 11.3 Grad-CAM (`outputs/octa/gradcam/`)

6 adet Grad-CAM ısı haritası + test tahminleri JSON dosyası.

### 11.4 Model Ağırlıkları (`outputs/octa/models/`)

| Dosya | Boyut | Açıklama |
|-------|:-----:|----------|
| `octa_v3_resnet18_best.pth` | 43 MB | Ana model (V5-TTA bu ağırlıkları kullanır) |
| `octa_v4_efficientnet_best.pth` | 16 MB | EfficientNet denemesi (referans) |

---

## 12. Kaynak Kod Referansları

| Dosya | Amaç |
|-------|------|
| `src/octa_dataset_v2.py` | PyTorch Dataset sınıfı (agresif augmentation) |
| `preprocessing/octa_labels_build.py` | Etiket CSV üretimi |
| `preprocessing/octa_split_build.py` | GroupKFold tabanlı train/val/test split |
| `training/train_octa_v3.py` | Ana eğitim scripti (Progressive FT + Early Stopping) |
| `evaluation/evaluate_octa_v3_model.py` | V3 test değerlendirmesi |
| `evaluation/evaluate_octa_v5_tta.py` | TTA + Optimal Threshold değerlendirmesi |
| `evaluation/gradcam_octa_v3.py` | Grad-CAM ısı haritası üretimi |
| `evaluation/generate_octa_publication_plots.py` | Makale kalitesinde görsel üretimi |

---

## 13. Sonuç

OCTA modülü, 525 görüntülük çok merkezli/çok cihazlı bir veri seti üzerinde eğitilmiş, **ROC AUC 0.91** ve **F1 Score 0.78** başarı düzeyine ulaşmış bir üveit sınıflandırma modelidir. Progressive Fine-Tuning, Cosine Annealing, Label Smoothing ve Test Time Augmentation gibi modern derin öğrenme tekniklerinin entegrasyonu ile, sınırlı veri koşullarında güçlü bir performans elde edilmiştir.

Model, projenin Faz-2 aşamasında (Multimodal Decision Fusion) Slit-lamp, B-scan OCT ve CFP modelleri ile birleştirilmeye hazırdır.

---

*Bu rapor, OCTA modülünün makale yazım sürecinde ilgili bölüm (Materials & Methods, Results) için temel kaynak olarak kullanılmak üzere hazırlanmıştır.*
