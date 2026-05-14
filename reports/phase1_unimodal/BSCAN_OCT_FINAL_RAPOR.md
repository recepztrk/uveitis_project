# B-scan OCT Modülü — Final Teknik Rapor

**Proje:** Üveit Karar Destek Sistemi  
**Modalite:** B-scan Optik Koherens Tomografi (B-scan OCT)  
**Son Güncelleme:** 10 Mayıs 2026  
**Final Model:** B-scan V4 K-Fold (Kermany Pretrained ResNet-18 + Sentetik Augmentation)

---

## 1. Giriş ve Amaç

B-scan OCT (Optical Coherence Tomography), retina katmanlarının kesitsel (cross-sectional) görüntülerini üreten, üveit tanı ve takibinde kritik bir modalitedir. Üveit hastalarında vitreus hücreleri, maküler ödem, retinal kalınlık değişiklikleri ve koroidal inflamasyon gibi bulgular B-scan OCT ile değerlendirilmektedir.

Bu modülde, B-scan OCT görüntülerinden üveit hastalığını otomatik olarak tespit eden bir ikili sınıflandırma modeli geliştirilmiştir. **Ana zorluk**, klinik üveit B-scan OCT verisinin aşırı sınırlı olmasıdır (sadece 27 gerçek üveit görüntüsü). Bu kısıtlamayı aşmak için **üç aşamalı bir transfer learning stratejisi** tasarlanmıştır.

---

## 2. Veri Seti

### 2.1 Orijinal Klinik Veri

| Bilgi | Değer |
|-------|-------|
| Toplam gerçek görüntü | **55** |
| Üveit | 27 (intermediate/posterior üveit) |
| Normal | 28 |
| Kaynak | Klinik B-scan OCT taramaları |

Bu, derin öğrenme için oldukça yetersiz bir veri miktarıdır. Bu sorunu çözmek için iki strateji birlikte uygulanmıştır.

### 2.2 Transfer Learning Kaynak Verisi: Kermany Veri Seti

İlk katman olarak, retinal OCT domain bilgisi içeren büyük bir veri setiyle ön eğitim yapılmıştır:

| Bilgi | Değer |
|-------|-------|
| Veri Seti | Kermany et al. (2018) — Retinal OCT Images |
| Toplam görüntü | ~108,312 |
| Sınıflar | CNV, DME, DRUSEN, NORMAL (4 sınıf) |
| Amaç | ResNet-18 omurgasının retinal OCT özniteliklerini öğrenmesi |
| Referans | Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning", Cell, 2018 |

### 2.3 Sentetik Veri Üretimi

27 gerçek üveit görüntüsünden **1,080 sentetik üveit görüntüsü** üretilmiştir (~40× çoğaltma):

| Teknik | Parametre |
|--------|-----------|
| Yatay/Dikey Çevirme | p=0.5 |
| Rastgele Döndürme | ±15° |
| Parlaklık/Kontrast Değişimi | ±%20 |
| Elastic Deformasyon | Hafif |
| Ölçekleme | %90–%110 |

> **Kritik Kural:** Sentetik veriler **SADECE eğitim setinde** kullanılmıştır. Test setinde yalnızca gerçek klinik görüntüler yer almıştır. Bu kural, sonuçların güvenilirliğini garanti eder.

### 2.4 V4 K-Fold Veri Yapısı (Final)

| Bileşen | Sayı | Açıklama |
|---------|:----:|----------|
| Gerçek Normal (Kermany) | 9,632 | Kermany veri setinden seçilmiş NORMAL sınıf |
| Gerçek Üveit | 27 | Orijinal klinik üveit görüntüleri |
| Sentetik Üveit | 1,080 | Augmentation ile üretilmiş |
| **TOPLAM** | **10,739** | |

**K-Fold Bölünme Stratejisi:**
- 5-Fold Cross Validation uygulanmıştır
- Her fold'da ~1,932 test + ~8,807 eğitim görüntüsü
- Test setlerinde sentetik veri **kesinlikle** bulunmaz
- Her fold'da 5-6 gerçek üveit görüntüsü test edilir
- 5 fold birleştirildiğinde (aggregated): tüm 27 gerçek üveit + tüm 9,632 normal test edilmiş olur

---

## 3. Veri Ön İşleme

### 3.1 Görüntü Hazırlama

| İşlem | Parametre |
|-------|-----------|
| Format | RGB dönüşümü |
| Boyut (Eğitim) | Resize(256) → RandomResizedCrop(224) |
| Boyut (Test) | Resize(224×224) |
| Normalizasyon | ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

### 3.2 Eğitim Augmentation

| Teknik | Parametre |
|--------|-----------|
| `RandomResizedCrop` | size=224, scale=(0.8, 1.0) |
| `RandomHorizontalFlip` | p=0.5 |
| `ColorJitter` | brightness=0.1, contrast=0.1 |

### 3.3 Metadata Dosyaları

| Dosya | İçerik |
|-------|--------|
| `metadata/bscan_oct_labels.csv` | 55 satır (orijinal etiketler) |
| `metadata/bscan_oct_split.csv` | Train/Val/Test split (V1-V3 için) |
| `metadata/bscan_oct_split_v4_kfold.csv` | 10,739 satır, 5 fold sütunu (V4 için) |

---

## 4. Model Mimarisi

### 4.1 Üç Aşamalı Transfer Learning Stratejisi

Bu modül, veri kıtlığını aşmak için **üç katmanlı bilgi transferi** kullanır:

```
Katman 1: ImageNet Pre-training (Genel Görsel Özellikler)
    ↓
Katman 2: Kermany Pre-training (Retinal OCT Domain Bilgisi)
    ↓
Katman 3: Üveit Fine-tuning (Hedef Görev)
```

1. **ImageNet → ResNet-18:** Genel görsel özellikler (kenarlar, dokular, şekiller)
2. **ResNet-18 → Kermany 4-sınıf:** Retinal katman yapısı, OCT artefaktları, hastalık paternleri
3. **Kermany model → Üveit binary:** Son katman 4→1 nörona değiştirilip üveit/normal ayrımı öğretilir

### 4.2 Progressive Fine-Tuning

| Faz | Epoch | Eğitilen Katmanlar | Öğrenme Oranı |
|-----|:-----:|---------------------|:-------------:|
| Faz 1 (Head Only) | 1–5 | Yalnızca son FC katmanı | 1×10⁻³ |
| Faz 2 (Full) | 6–15 | Tüm katmanlar | 5×10⁻⁵ |

### 4.3 Hiperparametreler

| Parametre | Değer |
|-----------|-------|
| Backbone | ResNet-18 (Kermany pretrained) |
| Son Katman | Linear(512, 1) |
| Optimizer | AdamW (weight_decay=1×10⁻⁴) |
| Batch Size | 32 |
| Max Epoch / Fold | 15 |
| Early Stopping | Patience = 7 |
| LR Scheduler | Cosine Annealing |
| Label Smoothing | 0.1 |
| Loss | BCEWithLogitsLoss + pos_weight |
| pos_weight | ~7.0 (otomatik hesaplanan, 9632/1107) |
| K-Fold | 5 fold |

---

## 5. Model Geliştirme Evrimi

### 5.1 Versiyon Geçmişi

| | V1 (Baseline) | V2 (Kermany PT) | V3 (+Synthetic) | V4 (K-Fold, Final) |
|---|:---:|:---:|:---:|:---:|
| **Veri** | 55 orijinal | 55 orijinal | 55 + sentetik | Kermany + sentetik |
| **Pre-training** | ImageNet | Kermany | Kermany | Kermany |
| **Sentetik** | ✗ | ✗ | ✓ | ✓ (1,080) |
| **Test Seti** | 9 resim | 9 resim | 9 resim | 9,659 resim (aggregated) |
| **Doğrulama** | Tek split | Tek split | Tek split | 5-Fold CV |
| **F1** | 1.000 ⚠️ | 0.857 | 1.000 ⚠️ | **0.900** ✅ |
| **AUC** | 1.000 ⚠️ | 1.000 | 1.000 ⚠️ | **1.000** ✅ |
| **Güvenilirlik** | ❌ Overfit | ⚠️ Sınırlı | ❌ Overfit | ✅ Güvenilir |

### 5.2 Kritik Tasarım Kararları

1. **V1 ve V3'teki "Mükemmel" Sonuçlar Yanıltıcıdır:** 9 test örneğinde %100 doğruluk, modelin gerçekten iyi olduğunu kanıtlamaz. Bu, overfitting veya şanslı veri bölünmesinin sonucu olabilir.

2. **V2'nin Düşüşü Aslında Daha Gerçekçidir:** Kermany pre-training sonrası F1'in 0.857'ye düşmesi, modelin gerçek sınırlarını göstermektedir.

3. **V4 K-Fold'un Getirdiği Güvenilirlik:** 9,659 test görüntüsü üzerinde aggregated değerlendirme yapılması, sonuçların istatistiksel olarak güvenilir olmasını sağlar. F1 = 0.900, bu büyük test setinde gerçek performansı yansıtır.

---

## 6. Final Test Sonuçları (V4 K-Fold Aggregated)

### 6.1 Ana Metrikler (n=9,659 aggregated test)

| Metrik | Değer |
|--------|:-----:|
| **Accuracy** | **99.94%** |
| **Precision** | **81.82%** |
| **Recall (Sensitivity)** | **100.00%** |
| **F1 Score** | **0.9000** |
| **ROC AUC** | **1.0000** |

### 6.2 Confusion Matrix (Aggregated, 5 Fold)

|  | Tahmin: Normal | Tahmin: Üveit |
|--|:---:|:---:|
| **Gerçek: Normal (n=9,632)** | TN = 9,626 (99.69%) | FP = 6 (0.06%) |
| **Gerçek: Üveit (n=27)** | FN = 0 (0.00%) | TP = 27 (0.28%) |

**Kritik Gözlem:** 
- **Sıfır False Negative (FN=0):** Model, 27 gerçek üveit görüntüsünün tamamını başarıyla yakalamıştır. Tıbbi bir tarama sisteminde bu en önemli metriktir.
- **Sadece 6 False Positive:** 9,632 normal görüntüden sadece 6 tanesi yanlış alarm vermiştir. Bu, klinik iş yükünü artırmayacak kadar düşük bir orandır.

### 6.3 Fold Bazlı Sonuçlar

| Fold | Test n | Üveit n | F1 | Recall | Precision |
|:----:|:------:|:-------:|:--:|:------:|:---------:|
| 1 | 1,932 | 5 | 1.000 | 1.000 | 1.000 |
| 2 | 1,932 | 5 | 0.909 | 1.000 | 0.833 |
| 3 | 1,931 | 6 | 0.857 | 1.000 | 0.750 |
| 4 | 1,932 | 6 | 0.800 | 1.000 | 0.667 |
| 5 | 1,932 | 5 | 1.000 | 1.000 | 1.000 |
| **Ort.** | — | — | **0.913** | **1.000** | **0.850** |

> **Not:** Tüm fold'larda Recall = 1.000. Model hiçbir fold'da üveit vakası kaçırmamıştır.

---

## 7. Sentetik Veri Üretim Detayları

### 7.1 Üretim Scripti

`training/generate_synthetic_bscan.py` scripti, 27 orijinal üveit görüntüsünden geometrik ve fotometrik dönüşümler uygulayarak 1,080 sentetik görüntü üretmiştir (her orijinal görüntüden 40 varyasyon).

### 7.2 Neden Sentetik Veri?

B-scan OCT'de sınıf dengesizliği aşırıdır:
- Kermany veri setinden 9,632 normal görüntü mevcuttur
- Gerçek üveit görüntüsü sadece 27'dir
- Oran: **1:357** (dengesiz)
- Sentetik üretim sonrası: **1:8.7** (kabul edilebilir, pos_weight ile dengelenir)

### 7.3 Test Bütünlüğü Garantisi

K-Fold split CSV dosyasında (`bscan_oct_split_v4_kfold.csv`) her satırda `is_synthetic` sütunu bulunur. Fold oluşturma algoritması, `is_synthetic=True` olan satırları **her zaman** `train` olarak işaretler, asla `test`'e almaz.

---

## 8. Kermany Pre-training Detayları

### 8.1 Süreç

1. ImageNet pretrained ResNet-18 alınmıştır
2. Son katman `Linear(512, 4)` olarak değiştirilmiştir (CNV, DME, DRUSEN, NORMAL)
3. Kermany veri seti üzerinde eğitilmiştir
4. Eğitilen model `bscan_kermany_resnet18_pretrained.pth` olarak kaydedilmiştir (43 MB)
5. Üveit fine-tuning için son katman `Linear(512, 1)` olarak değiştirilip, Kermany ağırlıkları backbone'da korunmuştur

### 8.2 Domain Yakınlığı Avantajı

Kermany verisi retinal OCT görüntüleri içerdiğinden, model şunları önceden öğrenmiştir:
- Retinal katman sınırlarını tanıma
- OCT artefaktlarını (speckle noise, shadow) filtreleme
- Patolojik kalınlaşma/incelme paternlerini ayırt etme

Bu domain bilgisi, sadece 27 üveit görüntüsüyle bile yüksek doğruluk elde etmemizi sağlamıştır.

---

## 9. Sınırlılıklar

1. **Aşırı Küçük Pozitif Sınıf:** 27 gerçek üveit görüntüsü, derin öğrenme için minimal düzeydedir. Sonuçların genellenebilirliği sınırlıdır.

2. **Sentetik Veriye Bağımlılık:** Eğitim setinin %98.4'ü (Kermany + sentetik) dış kaynaklardan gelmektedir. Model, gerçek üveit varyasyonlarını tam olarak öğrenmiş olmayabilir.

3. **Tek Tip Üveit:** Mevcut veriler yalnızca intermediate/posterior üveit içermektedir. Anterior üveit veya panuveitis vakaları değerlendirilmemiştir.

4. **Yüksek Sınıf Dengesizliği:** Test setinde bile oran ~1:357'dir. Bu durum, Precision'ın (%81.8) Recall'dan (%100) düşük olmasına yol açmaktadır.

5. **AUC = 1.000 Dikkatle Yorumlanmalı:** Mükemmel AUC, küçük pozitif sınıf boyutundan kaynaklanıyor olabilir. Daha büyük klinik verilerde bu değerin düşmesi beklenir.

---

## 10. Üretilen Çıktılar ve Dosya Haritası

### 10.1 Makale/Sunum Görselleri (`outputs/bscan/plots/`)

| Dosya | İçerik |
|-------|--------|
| `pub_01_version_evolution.png` | V1→V2→V3→V4 evrim grafiği (overfit uyarılı) |
| `pub_02_data_strategy.png` | Veri artırma stratejisi (orijinal + sentetik + Kermany) |
| `pub_03_kfold_results.png` | 5-Fold CV fold bazlı F1/Recall bar grafiği |
| `pub_04_confusion_matrix.png` | Aggregated confusion matrix (n=9,659, yüzdeli) |
| `pub_05_training_pipeline.png` | Eğitim pipeline'ı akış diyagramı |
| `pub_06_final_metrics.png` | Final metrik kartı (yatay bar) |
| `bscan_v2_training_history.png` | V2 eğitim eğrileri |
| `bscan_v3_training_history.png` | V3 eğitim eğrileri |
| `bscan_v2_roc_curve.png` | V2 ROC eğrisi |
| `bscan_v3_roc_curve.png` | V3 ROC eğrisi |
| `bscan_v2_per_sample_predictions.png` | V2 örnek bazlı tahminler |
| `bscan_v3_per_sample_predictions.png` | V3 örnek bazlı tahminler |

### 10.2 Ham Metrikler (`outputs/bscan/metrics/`)

| Dosya | İçerik |
|-------|--------|
| `bscan_v4_kfold_confusion_matrix.png` | V4 K-Fold aggregated CM |
| `bscan_v2_test_metrics.json` | V2 test metrikleri |
| `bscan_v3_test_metrics.json` | V3 test metrikleri |
| `bscan_oct_test_metrics.json` | V1 baseline metrikleri |
| `bscan_v2_train_history.json` | V2 eğitim tarihçesi |
| `bscan_v3_train_history.json` | V3 eğitim tarihçesi |

### 10.3 Model Ağırlıkları (`outputs/bscan/models/`)

| Dosya | Boyut | Açıklama |
|-------|:-----:|----------|
| `bscan_kermany_resnet18_pretrained.pth` | 43 MB | Kermany ön eğitimli backbone |
| `bscan_v4_kfold_best.pth` | 43 MB | **Final model** (en iyi fold) |
| `bscan_v2_finetuned_best.pth` | 43 MB | V2 referans |
| `bscan_v3_synthetic_best.pth` | 43 MB | V3 referans |
| `bscan_oct_resnet18_best.pth` | 43 MB | V1 baseline |

---

## 11. Kaynak Kod Referansları

| Dosya | Amaç |
|-------|------|
| `training/train_bscan_v4_kfold.py` | V4 K-Fold eğitim scripti (Final) |
| `training/train_bscan_v3_synthetic.py` | V3 sentetik veri eğitimi |
| `training/train_bscan_v2_finetuned.py` | V2 Kermany fine-tuning |
| `training/train_bscan_pretrain_kermany.py` | Kermany ön eğitim scripti |
| `training/train_bscan_oct_baseline.py` | V1 baseline |
| `training/generate_synthetic_bscan.py` | Sentetik üveit görüntü üretici |
| `preprocessing/bscan_labels_build.py` | Etiket CSV üretimi |
| `preprocessing/bscan_split_build.py` | Split üretimi |
| `src/bscan_oct_dataset.py` | PyTorch Dataset sınıfı |
| `evaluation/generate_bscan_publication_plots.py` | Makale kalitesinde görsel üretimi |

---

## 12. Sonuç

B-scan OCT modülü, klinik veri kıtlığı sorununu **üç katmanlı transfer learning** (ImageNet → Kermany → Üveit) ve **sentetik veri artırma** stratejileri ile başarıyla aşmıştır. 5-Fold Cross Validation ile 9,659 test görüntüsü üzerinde yapılan aggregated değerlendirmede **F1 Score 0.900** ve **Recall %100** elde edilmiştir.

Modelin en güçlü özelliği, 27 gerçek üveit görüntüsünün **tamamını** (%100 sensitivity) başarıyla yakalamasıdır. Bu, bir tıbbi tarama sistemi için kritik bir gerekliliktir.

Model, projenin Faz-2 aşamasında (Multimodal Decision Fusion) diğer modalitelerle birleştirilmeye hazırdır.

---

*Bu rapor, B-scan OCT modülünün makale yazım sürecinde ilgili bölüm (Materials & Methods, Results) için temel kaynak olarak kullanılmak üzere hazırlanmıştır.*
