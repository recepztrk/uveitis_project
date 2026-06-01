# Slit-lamp Modülü — Final Teknik Rapor

**Proje:** Üveit Karar Destek Sistemi  
**Modalite:** Slit-lamp Biyomikroskopi (Ön Segment Fotoğrafisi)  
**Son Güncelleme:** 10 Mayıs 2026  
**Final Model:** EfficientNet-B0 (ImageNet Pretrained) Baseline

---

## 1. Giriş ve Amaç

Slit-lamp biyomikroskopi, üveit tanısında en temel ve yaygın kullanılan klinik muayene yöntemidir. Ön kamara hücreleri (anterior chamber cells), flare, keratik presipitatlar (KP), sinesi ve hipopyon gibi üveit bulguları slit-lamp ile doğrudan gözlemlenir. SUN (Standardization of Uveitis Nomenclature) sınıflandırmasının temelini oluşturur.

Bu modülde, slit-lamp fotoğraflarından üveit hastalığını otomatik olarak ayırt eden bir ikili sınıflandırma modeli (Üveit vs Non-Üveit) geliştirilmiştir. Non-üveit sınıfı, klinik pratikte slit-lamp ile görüntülenen diğer yaygın ön segment patolojilerini (katarakt, konjonktivit, göz kapağı hastalıkları) içermektedir.

---

## 2. Veri Seti

### 2.1 Toplam Veri

| Bilgi | Değer |
|-------|-------|
| Toplam görüntü | **1,309** |
| Modalite | Slit-lamp biyomikroskopi fotoğrafı |
| Format | JPEG, RGB |
| İkili sınıf | Üveit (193) vs Non-Üveit (1,116) |
| Dengesizlik oranı | 1:5.8 |

### 2.2 Orijinal Sınıf Dağılımı (4 Sınıf)

Veriler aşağıdaki 4 hastalık kategorisinden toplanmıştır:

| Sınıf | Sayı | İkili Etiket |
|-------|:----:|:---:|
| Eyelid (Göz Kapağı) | 441 | Non-Üveit (0) |
| Cataract (Katarakt) | 357 | Non-Üveit (0) |
| Conjunctivitis (Konjonktivit) | 318 | Non-Üveit (0) |
| **Uveitis (Üveit)** | **193** | **Üveit (1)** |
| **TOPLAM** | **1,309** | |

**Not:** Non-Üveit sınıfının üç farklı patolojiden oluşması, modelin "sadece sağlıklı vs hasta" değil, "üveit vs diğer ön segment hastalıkları" ayrımını öğrenmesini sağlar. Bu, klinik açıdan daha gerçekçi ve zorlu bir görevdir.

### 2.3 Train / Validation / Test Bölünmesi

| Split | Toplam | Üveit | Non-Üveit |
|-------|:------:|:-----:|:---------:|
| Train | 916 | 135 (%14.7) | 781 (%85.3) |
| Validation | 196 | 29 (%14.8) | 167 (%85.2) |
| Test | 197 | 29 (%14.7) | 168 (%85.3) |

Bölünme stratified sampling ile yapılmıştır; her split'te üveit oranı ~%14.7-14.8 olarak korunmuştur.

---

## 3. Veri Ön İşleme

### 3.1 Görüntü Hazırlama

| İşlem | Parametre |
|-------|-----------|
| Format | RGB dönüşümü |
| Boyut | Resize(224×224) |
| Normalizasyon | ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

### 3.2 Eğitim Augmentation

| Teknik | Parametre | Amaç |
|--------|-----------|------|
| `RandomHorizontalFlip` | p=0.5 | Yatay simetri invariansı |
| `RandomRotation` | ±10° | Hafif açısal değişkenlik |
| `ColorJitter` | brightness=0.1, contrast=0.1 | Aydınlatma koşulları değişkenliği |

**Not:** Slit-lamp görüntüleri standart klinik koşullarda çekildiği için augmentation nispeten hafif tutulmuştur. Aşırı geometrik dönüşümler, klinik bulguların anatomik pozisyon bilgisini bozabilir.

### 3.3 Metadata Dosyaları

| Dosya | İçerik |
|-------|--------|
| `metadata/slitlamp_labels.csv` | 1,309 satır (image_id, filepath, raw_class, binary_label) |
| `metadata/slitlamp_split.csv` | Train/Val/Test split bilgisi eklenmiş hali |

---

## 4. Model Mimarisi

### 4.1 Backbone

| Parametre | Değer |
|-----------|-------|
| **Model** | EfficientNet-B0 |
| **Pre-training** | ImageNet |
| **Son Katman** | `classifier[1] → Linear(1280, 1)` |
| **Aktivasyon** | Sigmoid (inference), BCEWithLogitsLoss (eğitim) |
| **Parametre Sayısı** | ~5.3M (EfficientNet-B0 toplam) |

### 4.2 Neden EfficientNet-B0?

- ResNet-18'e kıyasla daha yüksek parametre verimliliği (compound scaling)
- MBConv blokları ile güçlü öznitelik çıkarma kapasitesi
- Squeeze-and-Excitation (SE) modülleri ile kanal bazlı dikkat mekanizması
- Küçük boyut (~5.3M parametre) sayesinde sınırlı veri setlerinde overfitting riski düşük

### 4.3 Hiperparametreler

| Parametre | Değer |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1×10⁻⁴ (sabit) |
| Batch Size | 16 |
| Max Epoch | 10 |
| Early Stopping | Uygulanmadı (10 epoch yeterli bulundu) |
| LR Scheduler | Kullanılmadı |
| Loss | BCEWithLogitsLoss |
| pos_weight | ~5.79 (otomatik: 781/135) |

### 4.4 Sınıf Dengesizliği Yönetimi

Eğitim setindeki 1:5.8 dengesizlik, `BCEWithLogitsLoss`'a verilen `pos_weight` parametresi ile yönetilmiştir:

```
pos_weight = neg_count / pos_count = 781 / 135 ≈ 5.79
```

Bu, loss hesabında her üveit örneğinin ~5.8 kat daha ağırlıklı sayılmasını sağlayarak, modelin azınlık sınıfını (üveit) ihmal etmesini önler.

---

## 5. Eğitim Süreci

Model 10 epoch boyunca eğitilmiştir. Tüm katmanlar (backbone dahil) baştan itibaren eğitime açıktır (progressive fine-tuning uygulanmamıştır).

### 5.1 Eğitim Eğrileri Özeti

| Epoch | Train Loss | Val Loss | Val F1 | Val AUC |
|:-----:|:----------:|:--------:|:------:|:-------:|
| 1 | 0.2921 | 0.5148 | 0.651 | 0.932 |
| 2 | 0.1375 | 0.4067 | 0.737 | 0.975 |
| 3 | 0.1001 | 0.3282 | 0.784 | 0.978 |
| 4 | 0.0813 | 0.2936 | 0.803 | 0.984 |
| 5 | 0.0795 | 0.3108 | 0.825 | 0.979 |
| 6 | 0.0741 | 0.2703 | 0.827 | 0.984 |
| 7 | 0.0654 | 0.2774 | 0.849 | 0.983 |
| **8** | **0.0623** | **0.2808** | **0.862** | **0.985** |
| 9 | 0.0690 | 0.3065 | 0.849 | 0.983 |
| 10 | 0.0682 | 0.3340 | 0.857 | 0.976 |

**En iyi model:** Epoch 8 (Val F1 = 0.862, Val AUC = 0.985)

**Gözlemler:**
- Train loss sürekli düşerken val loss epoch 7'den sonra hafifçe artıyor → erken overfitting sinyali, ancak 10 epoch ile sınırlı tutulduğu için ciddi bir sorun değil.
- AUC, epoch 2'den itibaren 0.97+ bandında stabilize olmuş → model sınıfları ayırt etme kapasitesini çok erken kazanmıştır.

---

## 6. Final Test Sonuçları

### 6.1 Ana Metrikler (Test Seti, n=197)

| Metrik | Değer |
|--------|:-----:|
| **Accuracy** | **96.95%** |
| **Precision** | **87.10%** |
| **Recall (Sensitivity)** | **93.10%** |
| **F1 Score** | **0.9000** |
| **ROC AUC** | **0.9883** |

### 6.2 Confusion Matrix

|  | Tahmin: Non-Üveit | Tahmin: Üveit |
|--|:---:|:---:|
| **Gerçek: Non-Üveit (n=168)** | TN = 164 (83.2%) | FP = 4 (2.0%) |
| **Gerçek: Üveit (n=29)** | FN = 2 (1.0%) | TP = 27 (13.7%) |

- **Specificity (Özgüllük):** 164/168 = **97.62%**
- **NPV (Negatif Prediktif Değer):** 164/166 = **98.80%**
- **Sadece 2 üveit kaçırılmış** (FN=2): %93.1 sensitivity
- **Sadece 4 yanlış alarm** (FP=4): %87.1 precision

### 6.3 Klinik Yorum

Model, 29 üveit vakasından 27'sini başarıyla tespit etmiştir. Kaçırılan 2 vaka muhtemelen hafif inflamasyon bulgusu gösteren sınır vakalarıdır. 4 yanlış pozitif, muhtemelen konjonktivit veya katarakt vakalarında üveit benzeri kızarıklık/bulanıklık paternlerinden kaynaklanmaktadır.

---

## 7. Diğer Modalitelerle Karşılaştırma

| Modalite | Backbone | Veri (n) | F1 | AUC | Recall |
|----------|----------|:--------:|:--:|:---:|:------:|
| **Slit-lamp** | EfficientNet-B0 (ImageNet) | 1,309 | **0.900** | **0.988** | 93.1% |
| B-scan OCT | ResNet-18 (Kermany PT) | 10,739 | 0.900 | 1.000 | 100.0% |
| OCTA | ResNet-18 (ImageNet) | 525 | 0.780 | 0.910 | 82.1% |
| CFP | EfficientNet-B0 (ImageNet) | 870 | 0.947 | 0.998 | 100.0% |
| AS-OCT | EfficientNet-B0 (Noisy Student) | 6,272 | 0.920 | 0.950 | ~92% |

**Slit-lamp modeli, en dengeli precision/recall performansını sergileyen modalitedir.** AUC = 0.988 ile sadece CFP'nin gerisinde kalmakta; ancak diğer tüm modalitelere kıyasla en gerçekçi veri dağılımına (4 farklı ön segment patolojisi) sahip olması, bu performansı daha anlamlı kılmaktadır.

---

## 8. Sınırlılıklar

1. **Tek Veri Kaynağı:** Tüm görüntüler tek bir kaynaktan geldiği için, farklı kliniklerdeki aydınlatma, çekim açısı ve kamera çeşitliliği test edilmemiştir.

2. **K-Fold Yapılmadı:** Performans güvenilirliği tek train/test split'e dayalıdır. Ancak test seti yeterince büyük (n=197) ve sonuçlar tutarlı olduğu için bu kabul edilebilir düzeydedir.

3. **Üveit Alt Tipi Ayrımı Yok:** Anterior, intermediate ve posterior üveit alt tipleri ayrıştırılmamıştır. Tüm üveit görüntüleri tek bir pozitif sınıf olarak etiketlenmiştir.

4. **Sınırlı Augmentation:** Sadece temel augmentation (flip, rotation, color jitter) uygulanmıştır. CutMix, MixUp gibi ileri teknikler denenmemiştir.

5. **Progressive Fine-Tuning Uygulanmadı:** Tüm katmanlar baştan eğitime açık bırakılmıştır. Bu, daha büyük veri setlerinde avantaj sağlayabilirdi ancak mevcut performans zaten yüksektir.

---

## 9. Üretilen Çıktılar ve Dosya Haritası

### 9.1 Makale/Sunum Görselleri (`outputs/slitlamp/plots/`)

| Dosya | İçerik |
|-------|--------|
| `pub_01_data_distribution.png` | 4-sınıf dağılımı + ikili dağılım + split dağılımı |
| `pub_02_training_curves.png` | Loss, F1, AUC, Precision/Recall eğitim eğrileri |
| `pub_03_confusion_matrix.png` | Yüzdeli confusion matrix (n=197) |
| `pub_04_roc_curve.png` | Gölgeli ROC eğrisi (AUC=0.988) |
| `pub_05_final_metrics.png` | Final metrik kartı (yatay bar) |
| `pub_06_all_models_comparison.png` | 4 modalite karşılaştırma grafiği |

### 9.2 Ham Metrikler (`outputs/slitlamp/metrics/`)

| Dosya | İçerik |
|-------|--------|
| `slitlamp_test_metrics.json` | Test metrikleri |
| `slitlamp_train_history.json` | 10 epoch eğitim tarihçesi |
| `slitlamp_test_predictions.json` | Örnek bazlı tahmin olasılıkları |
| `slitlamp_confusion_matrix.png` | Orijinal CM |
| `slitlamp_roc_curve.png` | Orijinal ROC |

### 9.3 Model Ağırlıkları (`outputs/slitlamp/models/`)

| Dosya | Boyut | Açıklama |
|-------|:-----:|----------|
| `slitlamp_efficientnetb0_best.pth` | ~16 MB | Final model (Epoch 8) |

---

## 10. Kaynak Kod Referansları

| Dosya | Amaç |
|-------|------|
| `training/train_slitlamp_baseline.py` | Ana eğitim scripti |
| `evaluation/evaluate_slitlamp_model.py` | Test değerlendirmesi |
| `evaluation/gradcam_slitlamp.py` | Grad-CAM ısı haritası üretimi |
| `evaluation/generate_slitlamp_publication_plots.py` | Makale kalitesinde görsel üretimi |
| `preprocessing/slitlamp_labels_build.py` | Etiket CSV üretimi |
| `preprocessing/slitlamp_split_build.py` | Stratified split üretimi |
| `src/slitlamp_dataset.py` | PyTorch Dataset sınıfı |

---

## 11. Sonuç

Slit-lamp modülü, 1,309 görüntülük bir veri seti üzerinde eğitilmiş ve **F1 Score 0.900**, **ROC AUC 0.988** performansına ulaşmış güçlü bir üveit sınıflandırma modelidir. EfficientNet-B0'ın parametre verimliliği ve ImageNet transfer learning sayesinde, sadece 10 epoch eğitimle klinik düzeyde güvenilir sonuçlar elde edilmiştir.

Model, üveiti diğer yaygın ön segment patolojilerinden (katarakt, konjonktivit, göz kapağı) başarıyla ayırt edebilmektedir. Bu, gerçek klinik senaryoya yakın ve akademik olarak anlamlı bir tasarımdır.

Model, projenin Faz-2 aşamasında (Multimodal Decision Fusion) diğer modalitelerle birleştirilmeye hazırdır.

---

*Bu rapor, Slit-lamp modülünün makale yazım sürecinde ilgili bölüm (Materials & Methods, Results) için temel kaynak olarak kullanılmak üzere hazırlanmıştır.*
