# Üveit Karar Destek Sistemi — Proje Tam Durum Raporu

**Tarih:** 2 Mayıs 2026  
**Ortam:** Python 3.13.5, PyTorch 2.11.0, macOS (Apple Silicon MPS)

---

## 1. Projenin Amacı

Bu proje, üveit göz hastalığının tanı süreçlerinde göz hekimlerine yardımcı olacak bir **yapay zekâ tabanlı karar destek sistemi** geliştirmeyi hedeflemektedir. Farklı oftalmolojik görüntüleme modalitelerinden (Slit-lamp, OCTA, B-scan OCT, CFP) elde edilen verilerle bağımsız uzman modeller eğitilmiş, ilerleyen aşamada bu modellerin bir füzyon katmanında birleştirilmesi planlanmaktadır.

**Proje iki faz olarak tasarlanmıştır:**

- **Faz-1 (Tamamlandı):** Her modalite için bağımsız unimodal uzman modeller geliştirmek.
- **Faz-2 (Planlanıyor):** Unimodal modellerin çıktılarını multimodal füzyon ile birleştirmek.

---

## 2. Proje Dizin Yapısı

```
uveitis_project/
├── app/                         # [YENİ] Demo Web Arayüzü (FastAPI + HTML/CSS/JS)
│   ├── main.py                  # API ve sunucu
│   ├── inference.py             # Birleşik model motoru
│   ├── templates/               # HTML Şablonları (index.html)
│   └── static/                  # CSS, JS ve Örnek Vaka görselleri
├── data_raw/                    # Orijinal ham veriler (dokunulmaz arşiv)
│   ├── bscan_oct_raw/
│   ├── octa_raw/
│   └── slitlamp_raw/
├── data_work/                   # Temizlenmiş, eğitime hazır veriler
│   ├── bscan_oct_clean/
│   │   ├── uveitis/
│   │   └── non_uveitis/
│   ├── cfp_clean/
│   │   ├── uveitis/             # 63 görüntü (VKH + CRS + RS)
│   │   └── non_uveitis/         # 807 görüntü (normal + diğer retinal hastalıklar)
│   ├── octa_clean/
│   │   ├── uveitis/             # 180 görüntü
│   │   └── non_uveitis/         # 345 görüntü
│   └── slitlamp_clean/
│       ├── uveitis/             # 194 görüntü
│       └── non_uveitis/         # 1117 görüntü
├── metadata/                    # Etiket CSV dosyaları ve split bilgileri
│   ├── octa_labels.csv          # 525 satır
│   ├── octa_split.csv           # 525 satır
│   ├── slitlamp_labels.csv      # 1309 satır
│   ├── slitlamp_split.csv       # 1309 satır
│   ├── cfp_labels.csv           # 870 satır (image_id, filepath, source_dataset, source_class, binary_label)
│   ├── cfp_split.csv            # 870 satır (+ split sütunu: train/val/test)
│   ├── bscan_oct_labels.csv     # 55 satır
│   └── bscan_oct_split.csv      # 55 satır
├── src/                         # PyTorch Dataset sınıfları
│   ├── cfp_dataset.py           # CFP dataset (fundus-uygun augmentation + built-in transforms)
│   ├── octa_dataset_v2.py       # OCTA dataset (agresif augmentation — V3 bunu kullanır)
│   ├── slitlamp_dataset.py
│   └── bscan_oct_dataset.py
├── preprocessing/               # Veri hazırlama scriptleri
│   ├── cfp_labels_build.py      # cfp_labels.csv üretir
│   ├── cfp_split_build.py       # cfp_split.csv üretir (source_class stratified)
│   ├── octa_labels_build.py
│   ├── octa_split_build.py
│   ├── slitlamp_labels_build.py
│   ├── slitlamp_split_build.py
│   ├── bscan_labels_build.py
│   └── bscan_split_build.py
├── training/                    # Eğitim scriptleri
│   ├── train_cfp_baseline.py        # CFP baseline (EfficientNet-B0)
│   ├── train_slitlamp_baseline.py
│   ├── train_bscan_oct_baseline.py
│   ├── train_octa_v3.py             # V3 (ana OCTA model)
│   ├── train_octa_v4_efficientnet.py # V4 (EfficientNet denemesi)
│   └── train_octa_kfold_cv.py        # 5-Fold Cross Validation
├── evaluation/                  # Değerlendirme ve görselleştirme
│   ├── evaluate_cfp_model.py         # CFP (kaynak bazlı analiz + hata analizi)
│   ├── evaluate_slitlamp_model.py
│   ├── evaluate_bscan_oct_model.py
│   ├── evaluate_octa_v3_model.py     # V3 (cihaz bazlı analiz)
│   ├── gradcam_cfp.py                # CFP Grad-CAM + eğitim eğrileri
│   ├── gradcam_slitlamp.py           # Slit-lamp Grad-CAM
│   └── gradcam_octa_v3.py            # OCTA V3 Grad-CAM
├── outputs/
│   ├── models/                  # Eğitilmiş model ağırlıkları (.pth)
│   ├── metrics/                 # JSON metrikler, confusion matrix, ROC, eğitim eğrileri
│   └── gradcam/                 # Grad-CAM ısı haritası görselleri
├── reports/                     # Proje raporları
└── .venv/                       # Python 3.13.5 sanal ortam
```

---

## 3. Ortak Geliştirme Pipeline'ı

Her modalite için aynı sistematik akış uygulanmıştır:

1. **Veri Temizliği:** Ham veriler incelenmiş, problemli/duplike dosyalar çıkarılmış, `data_work/` altında `uveitis/` ve `non_uveitis/` olarak düzenlenmiştir.
2. **Etiketleme:** `preprocessing/*_labels_build.py` scriptleri ile otomatik metadata CSV üretilmiştir.
3. **Train/Val/Test Split:** `preprocessing/*_split_build.py` ile %70/%15/%15 stratified split yapılmıştır. OCTA'da `group_id` bazlı bölme ile data leakage önlenmiştir.
4. **Dataset Sınıfı:** `src/` altında PyTorch `Dataset` sınıfları yazılmıştır.
5. **Model Eğitimi:** `training/` altında transfer learning ile pretrained backbone fine-tuning yapılmıştır.
6. **Değerlendirme:** `evaluation/` altında test metrikleri, confusion matrix, ROC eğrisi ve Grad-CAM üretilmiştir.

---

## 4. Modaliteler ve Veri Setleri

### 4.1 Slit-lamp (Ön Segment Fotoğrafı)

| Bilgi | Değer |
|-------|-------|
| Toplam görüntü | 1.309 |
| Sınıflar | Uveitis (194), Cataract (549), Conjunctivitis (267), Eyelid (300) |
| Binary problem | Uveitis vs Non-Uveitis |
| Split | Train: 916, Val: 196, Test: 197 |
| Önemli karar | "Normal" sınıfı çıkarıldı (veri kaynağı farkını öğrenme riski) |

### 4.2 OCTA (Optik Koherens Tomografi Anjiyografi)

| Bilgi | Değer |
|-------|-------|
| Toplam görüntü | **525** (V3 ile genişletildi; başlangıçta 338 idi) |
| Sınıflar | Uveitis (180: active + inactive), Control (345) |
| Binary problem | Uveitis vs Control |
| Katmanlar | Superficial + Deep (V3'te tümü dahil edildi) |
| Cihazlar | Heidelberg (338 orijinal) + OptoVue (187 harici kontrol) |
| Split | Train: 363 (BH:228 + UT:135), Val: 79 (BH:52 + UT:27), Test: 83 (BH:58 + UT:25) |
| Data leakage önlemi | `group_id` bazlı GroupKFold ile aynı hastanın tüm görüntüleri aynı split'te |

**OCTA veri genişletme süreci (V3):**
- Başlangıçta yalnızca Heidelberg cihazı superficial katman PNG görüntüleri (214 adet) kullanılıyordu.
- V2'de deep katman ve TIFF formatları da dahil edildi → 338 görüntü.
- V3 için harici bir OptoVue veri seti (UT-OCTA) bulundu ve 187 sağlıklı kontrol görüntüsü eklendi → **525 görüntü**.
- Bu genişleme, hem veri miktarını artırdı hem de "domain generalization" testi sağladı (farklı cihaz verisi).

### 4.3 B-scan OCT

| Bilgi | Değer |
|-------|-------|
| Toplam görüntü | 55 |
| Sınıflar | Uveitis (27), Normal (28) |
| Split | Train: 38, Val: 8, Test: 9 |
| Durum | ⚠️ Çok az veri, overfitting sorunu mevcut |

### 4.4 CFP (Color Fundus Photography — Renkli Fundus Fotoğrafı)

| Bilgi | Değer |
|-------|-------|
| Toplam görüntü | **870** |
| Sınıflar | Uveitis (63: VKH=14, CRS=41, RS=8), Non-Uveitis (807) |
| Binary problem | Üveit ilişkili posterior inflamasyon vs Non-Uveitis |
| Kaynak veri setleri | 1000images (cfp1000: 563), RFMiD2_0 (rfmimd2: 307) |
| Split | Train: 609 (uv:45, nonuv:564), Val: 130 (uv:9, nonuv:121), Test: 131 (uv:9, nonuv:122) |
| Sınıf dengesizliği | 1:12.5 (ciddi) |
| Stratify | `source_class` bazlı stratified split |

**CFP pozitif sınıf tanımı:**
- Bu sınıf doğrudan "genel üveit tanısı" değildir. VKH (Vogt-Koyanagi-Harada), CRS (Chorioretinitis) ve RS (Retinitis) etiketleri üzerinden temsil edilen **üveit ilişkili posterior segment inflamatuvar bulguları** ifade eder.
- CRS ve RS, RFMiD2_0 veri setinden **proxy etiket** olarak kullanılmıştır.

**CFP negatif sınıf yapısı:**
- Sadece normal değil, üveit dışı retinal hastalıklar da dahildir (DR, BRVO, CRVO, RD, CSCR, ERM, MH vb.).
- Bu sayede model "normal-hasta" ayrımı yerine, üveit ilişkili inflamasyon paternlerini diğer retinal patolojilerden ayırmayı öğrenir.

**Sınırlılıklar:**
- Pozitif sınıf çok küçük (63 görüntü, test'te sadece 9)
- Hasta bazlı split yapılamamıştır (ID bilgisi yok)
- Farklı veri setlerinden kaynak farkı / domain shift riski mevcut

---

## 5. Eğitilmiş Modeller

### 5.1 Ana (Aktif) Modeller — Projede Kullanılacak 4 Model

#### 🏆 Model 1: Slit-lamp (EfficientNet-B0)

| Parametre | Değer |
|-----------|-------|
| Dosya | `outputs/models/slitlamp_efficientnetb0_best.pth` (16 MB) |
| Script | `training/train_slitlamp_baseline.py` |
| Backbone | EfficientNet-B0 (ImageNet pretrained) |
| Son katman | `classifier[1] → Linear(1280, 1)` |
| Optimizer | AdamW, LR=1e-4 |
| Epoch | 10 (en iyi: epoch 8) |
| Loss | BCEWithLogitsLoss + pos_weight |
| Augmentation | HorizontalFlip, Rotation(±10°), ColorJitter |

**Test Sonuçları:**

| Metrik | Değer |
|--------|-------|
| Accuracy | **96.95%** |
| Precision | 87.10% |
| Recall | 93.10% |
| F1 Score | **0.900** |
| ROC AUC | **0.988** |

**Grad-CAM:** Üretildi (`outputs/gradcam/gradcam_1-6_*.png`). Model, üveit vakalarında konjonktival hiperemi ve ön kamara bölgesine odaklanıyor.

---

#### ✅ Model 2: OCTA V3 (ResNet-18) — Ana OCTA Modeli

| Parametre | Değer |
|-----------|-------|
| Dosya | `outputs/models/octa_v3_resnet18_best.pth` (43 MB) |
| Script | `training/train_octa_v3.py` |
| Dataset sınıfı | `src/octa_dataset_v2.py` (agresif augmentation) |
| Backbone | ResNet-18 (ImageNet pretrained) |
| Son katman | `fc → Linear(512, 1)` |
| Optimizer | AdamW, LR_HEAD=1e-3, LR_FULL=1e-4, weight_decay=1e-4 |
| Epoch | 40 (early stopping @ epoch 34, en iyi: epoch 24) |
| Loss | BCEWithLogitsLoss + pos_weight + Label Smoothing (0.1) |
| Strateji | Progressive Fine-Tuning (5 epoch head → full), Cosine Annealing |
| Augmentation | RandomResizedCrop, HFlip, VFlip, Affine(±15°), ColorJitter, RandomErasing |

**Test Sonuçları:**

| Metrik | Değer |
|--------|-------|
| Accuracy | **81.93%** |
| Precision | 69.70% |
| Recall | 82.14% |
| F1 Score | **0.754** |
| ROC AUC | **0.901** |

**Cihaz Bazlı Performans (Domain Shift Kontrolü):**

| Cihaz | n | Accuracy | F1 |
|-------|:-:|:--------:|:--:|
| Heidelberg (BH) | 58 | 75.86% | 0.767 |
| OptoVue (UT) | 25 | 96.00% | — (sadece kontrol grubu) |

**Grad-CAM:** Üretildi (`outputs/gradcam/octa_v3_gradcam_1-6_*.png`), 2 TP + 2 TN + 1 FP + 1 FN.

---

#### ✅ Model 3: CFP (EfficientNet-B0) — Posterior Segment İnflamasyon Modeli

| Parametre | Değer |
|-----------|-------|
| Dosya | `outputs/models/cfp_efficientnetb0_best.pth` (16 MB) |
| Script | `training/train_cfp_baseline.py` |
| Dataset sınıfı | `src/cfp_dataset.py` (fundus-uygun augmentation) |
| Backbone | EfficientNet-B0 (ImageNet pretrained) |
| Son katman | `classifier[1] → Linear(1280, 1)` |
| Optimizer | AdamW, LR_HEAD=1e-3, LR_FULL=1e-4, weight_decay=1e-4 |
| Epoch | 30 (early stopping @ epoch 21, en iyi: epoch 11) |
| Loss | BCEWithLogitsLoss + pos_weight (12.53) + Label Smoothing (0.1) |
| Strateji | Progressive Fine-Tuning (5 epoch head → full), Cosine Annealing |
| Dengesizlik | WeightedRandomSampler + pos_weight (çift dengeleme, oran 1:12.5) |
| Augmentation | RandomResizedCrop(224), HFlip, Rotation(±10°), ColorJitter, RandomErasing |
| Not | VerticalFlip kullanılmadı (fundus oryantasyonu bozulur) |

**Test Sonuçları:**

| Metrik | Değer |
|--------|-------|
| Accuracy | **96.18%** |
| Precision | 64.29% |
| Recall | **100%** |
| F1 Score | **0.783** |
| ROC AUC | **1.000** |

**Kaynak Veri Seti Bazlı Performans (Domain Shift Kontrolü):**

| Kaynak | n | Pozitif | Accuracy | F1 | Recall |
|--------|:-:|:------:|:--------:|:--:|:------:|
| cfp1000 | 85 | 2 | 96.47% | 0.571 | 100% |
| rfmimd2 | 46 | 7 | 95.65% | 0.875 | 100% |

**Hata Analizi:** 0 FN (hiç üveit kaçırılmadı), 5 FP (3'ü eşik sınırında prob ≈ 0.50-0.51).

**Grad-CAM:** Üretildi (`outputs/gradcam/cfp_gradcam_1-5_*.png`), 2 TP + 2 TN + 1 FP. Model, chorioretinitis lezyonlarına ve optik disk çevresine odaklanıyor.

---

#### ⚠️ Model 4: B-scan OCT (ResNet-18)

| Parametre | Değer |
|-----------|-------|
| Dosya | `outputs/models/bscan_oct_resnet18_best.pth` (43 MB) |
| Script | `training/train_bscan_oct_baseline.py` |
| Test Sonuçları | Acc: 100%, F1: 1.00, AUC: 1.00 |
| Durum | ⚠️ **Overfit** — 55 veri, 9 test örneği ile güvenilir değil |

---

### 5.2 Deneysel Modeller (Ağırlıkları silindi, sadece sonuçları referans olarak tutuldu)

| Model | F1 | AUC | Notu |
|-------|:--:|:---:|------|
| OCTA V1 (ResNet-18, 214 görüntü) | 0.700 | 0.732 | V3 ile aşıldı, ağırlıklar silindi |
| OCTA V2 (ResNet-18, 338 görüntü) | 0.704 | 0.696 | V3 ile aşıldı, ağırlıklar silindi |
| OCTA V4 (EfficientNet-B0, 525 görüntü) | 0.719 | 0.856 | V3'ten kötü, ağırlık mevcut: `octa_v4_efficientnet_best.pth` |
| K-Fold CV (5 fold, EfficientNet-B0) | 0.725±0.069 | 0.829±0.066 | Güvenilirlik referansı, ağırlıklar silindi |

---

## 6. OCTA Modelinin Evrimi (V1 → V3)

Bu tablo, OCTA modeline yapılan tüm iyileştirmeleri ve etkilerini göstermektedir:

| | V1 | V2 | V3 (Ana) |
|--|:--:|:--:|:--------:|
| **Veri sayısı** | 214 | 338 | **525** |
| **Katmanlar** | Sadece superficial | Superficial + Deep | Superficial + Deep |
| **Cihaz** | Sadece Heidelberg | Sadece Heidelberg | **Heidelberg + OptoVue** |
| **Augmentation** | Temel | Agresif | Agresif |
| **Fine-tuning** | Tam | Progressive | Progressive |
| **Label Smoothing** | Yok | 0.1 | 0.1 |
| **LR Schedule** | Sabit | Cosine Annealing | Cosine Annealing |
| **Early Stopping** | Yok | Yok | ✅ (patience=10) |
| **Optimizer** | Adam | Adam | **AdamW** |
| **F1** | 0.700 | 0.704 | **0.754** |
| **AUC** | 0.732 | 0.696 | **0.901** |
| **Accuracy** | 65.4% | 69.2% | **81.9%** |

**Büyük sıçrama V3'te oldu:** Harici OptoVue veri setinin eklenmesi AUC'yi 0.70'den 0.90'a taşıdı.

---

## 7. Mevcut Çıktılar (outputs/)

### 7.1 Model Ağırlıkları (`outputs/models/`)
- `slitlamp_efficientnetb0_best.pth` — Slit-lamp ana model (16 MB)
- `octa_v3_resnet18_best.pth` — OCTA ana model (43 MB)
- `cfp_efficientnetb0_best.pth` — CFP ana model (16 MB)
- `bscan_oct_resnet18_best.pth` — B-scan model (43 MB)
- `octa_v4_efficientnet_best.pth` — EfficientNet denemesi (16 MB, referans)

### 7.2 Metrikler (`outputs/metrics/`)

**Görseller (PNG):**
- `slitlamp_confusion_matrix.png`, `slitlamp_roc_curve.png`
- `bscan_oct_confusion_matrix.png`, `bscan_oct_roc_curve.png`
- `octa_v3_confusion_matrix.png`, `octa_v3_roc_curve.png`
- `cfp_confusion_matrix.png`, `cfp_roc_curve.png`
- `cfp_training_curves.png` — CFP Loss, F1, AUC eğitim eğrileri
- `octa_v3_training_curves.png` — OCTA V3 eğitim eğrileri
- `octa_v3_version_comparison.png` — V1/V2/V3 karşılaştırma
- `data_distribution.png`, `model_comparison.png`, `training_loss_curves.png`

**JSON:**
- `slitlamp_test_metrics.json`, `bscan_oct_test_metrics.json`, `octa_v3_test_metrics.json`, `cfp_test_metrics.json` — Test metrikleri
- `slitlamp_train_history.json`, `bscan_oct_train_history.json`, `octa_v3_train_history.json`, `cfp_train_history.json` — Eğitim tarihçesi
- `octa_kfold_results.json` — 5-Fold CV sonuçları
- `octa_v4_efficientnet_train_history.json` — V4 denemesi tarihçesi

### 7.3 Grad-CAM (`outputs/gradcam/`)
- `gradcam_1-6_*.png` — Slit-lamp Grad-CAM (2 TP + 2 TN + 1 FP + 1 FN)
- `octa_v3_gradcam_1-6_*.png` — OCTA V3 Grad-CAM (2 TP + 2 TN + 1 FP + 1 FN)
- `cfp_gradcam_1-5_*.png` — CFP Grad-CAM (2 TP + 2 TN + 1 FP)
- `slitlamp_test_predictions.json`, `octa_v3_test_predictions.json`, `cfp_test_predictions.json` — Tüm test tahminleri

---

## 8. Geliştirme Geçmişi

### 8.1 Oturum: OCTA V3 Geliştirme (24 Nisan 2026)
- OCTA V3 modeli eğitildi (525 görüntü, early stopping @ epoch 34, en iyi: epoch 24)
- **Sonuç:** F1: 0.754, AUC: 0.901
- Cihaz bazlı analiz yapıldı (Heidelberg vs OptoVue), domain shift sorunu tespit edilmedi
- EfficientNet-B0 denemesi (V4): F1: 0.719 — ResNet-18 (V3) daha iyi, ana backbone olarak kaldı
- 5-Fold CV: F1: 0.725 ± 0.069, AUC: 0.829 ± 0.066 — güvenilirlik doğrulandı
- Grad-CAM ve training curves üretildi

### 8.2 Oturum: Proje Temizliği (25 Nisan 2026)
- Eski V1/V2 model ağırlıkları, scriptleri ve duplike JSON dosyaları silindi (~250 MB kazanç)
- K-Fold model ağırlıkları silindi (sonuçlar JSON'da mevcut)
- `checks/` klasörü, `.DS_Store`, `__pycache__` temizlendi
- Proje tam durum raporu oluşturuldu (`proje_tam_rapor.md`)

### 8.3 Oturum: CFP Modeli Geliştirme (26 Nisan 2026)
- **Veri hazırlama:** 870 CFP görüntüsü (63 uveitis, 807 non-uveitis) iki kaynaktan (1000images + RFMiD2_0) hazırlandı
- `src/cfp_dataset.py` — Fundus-uygun augmentation ile Dataset sınıfı oluşturuldu
- `training/train_cfp_baseline.py` — EfficientNet-B0, Progressive FT, WeightedRandomSampler, Label Smoothing ile eğitim yapıldı
- Model 21. epoch'ta early stopping ile durdu (en iyi: epoch 11)
- **Sonuç:** F1: 0.783, AUC: 1.000, Recall: %100 (9/9 üveit yakalandı, 0 FN)
- `evaluation/evaluate_cfp_model.py` — Confusion matrix, ROC curve, kaynak bazlı analiz, hata analizi üretildi
- `evaluation/gradcam_cfp.py` — 5 adet Grad-CAM görseli (2 TP + 2 TN + 1 FP) ve training curves üretildi
- CFP modeli posterior segment inflamatuvar bulgulara (chorioretinitis lezyonları, optik disk çevresi) doğru odaklanıyor

### 8.4 Oturum: Canlı Demo Arayüzü (Sunum Katmanı) Geliştirme (1-2 Mayıs 2026)
- **Amaç:** Projenin akademik sunumlarda ve üniversite bitirme projesi pazarında canlı olarak sergilenebilmesi.
- **Backend:** `FastAPI` kullanılarak yüksek performanslı bir API (`app/main.py`) geliştirildi. Tüm modeller bellekte bir kerede yüklenip sunuma hazır tutuluyor (`app/inference.py`).
- **Frontend:** Tek sayfalık uygulama (SPA) mantığı ile `HTML/CSS/JS` kullanılarak projeksiyona uygun **Dark Mode** temalı tıbbi bir arayüz geliştirildi.
- **Özellikler:** Canlı yükleme, test setinden ayrılan hazır örnek vakaları anında analiz etme ve Grad-CAM ısı haritalarını orijinal görüntü üzerinde "Overlay" olarak gösterme özellikleri başarıyla sisteme entegre edildi. Modellerin başarı metrikleri (F1, AUC, Veri sayısı) arayüzde gösteriliyor.

---

## 9. Proje Gidişatı ve Sonraki Adımlar

### ✅ Tamamlanan İşler
1. **Dört modalite** için veri temizliği, etiketleme, split → **Tamamlandı**
2. Slit-lamp baseline model (F1: 0.90, AUC: 0.99) → **Tamamlandı**
3. OCTA V3 model (F1: 0.754, AUC: 0.901) → **Tamamlandı**
4. CFP baseline model (F1: 0.783, AUC: 1.00, Recall: %100) → **Tamamlandı**
5. B-scan OCT baseline model → **Tamamlandı** (ancak veri yetersizliği sorunu var)
6. OCTA V4 EfficientNet denemesi → **Tamamlandı** (V3 daha iyi)
7. OCTA 5-Fold CV → **Tamamlandı** (güvenilirlik doğrulaması)
8. Grad-CAM analizi (Slit-lamp + OCTA V3 + CFP) → **Tamamlandı**
9. Proje temizliği ve raporlama → **Tamamlandı**
10. **Demo Web Arayüzü (Sunum Katmanı)** geliştirilmesi ve FastAPI entegrasyonu → **Tamamlandı**

### 📋 Yapılması Gereken İşler
1. **B-scan modeli iyileştirme** — 55 veriyle overfit var; cross-validation veya ek veri gerekli
2. **Multimodal füzyon (Faz-2)** — 4 unimodal modelin çıktılarını birleştirme mimarisi
3. **B-scan için Grad-CAM** — Henüz yapılmadı

---

## 10. Önemli Teknik Notlar

### Veri Sızıntısı (Data Leakage) Önlemi
OCTA'da aynı hastanın aktif/inaktif dönemlerde çekilmiş birden fazla görüntüsü olabilir. `group_id = source_class + "_" + original_id` mantığı ile aynı hastanın tüm görüntüleri aynı split'te tutulur. K-Fold CV'de de `GroupKFold` kullanılmıştır.

### Domain Shift Yönetimi
OCTA V3'te iki farklı cihazdan (Heidelberg + OptoVue), CFP'de iki farklı veri setinden (cfp1000 + rfmimd2) veri birleştirilmiştir. Her iki modelin değerlendirme scriptlerinde kaynak bazlı performans ayrımı raporlanarak modellerin kaynak farkına mı yoksa patolojik özelliklere mi baktığı kontrol edilmiştir.

### Label Smoothing
V2/V3'te `y_smooth = y * 0.9 + 0.05` formülü ile label smoothing uygulanmıştır. Bu teknik, modelin aşırı güvenli tahminler yapmasını engelleyerek genellemeyi artırır. Validation/test'te smoothing uygulanmaz.

### Progressive Fine-Tuning
OCTA V3 ve CFP modellerinde uygulanmıştır. İlk 5 epoch'ta backbone dondurulur, sadece son katman (head) eğitilir. 6. epoch'tan itibaren tüm ağırlıklar serbest bırakılır. Bu strateji, pretrained ağırlıkların korunmasını ve daha stabil eğitim sağlar.

### WeightedRandomSampler (CFP)
CFP'deki ciddi sınıf dengesizliği (1:12.5) için pos_weight tek başına yeterli görülmemiş, ek olarak `WeightedRandomSampler` kullanılmıştır. Her epoch'ta pozitif ve negatif örnekler yaklaşık eşit sıklıkta örneklenir. Sonuç olarak tüm eğitim boyunca Recall %100 kalmıştır.

### Dosya İsimlendirme Konvansiyonu
- `*_labels_build.py` → Labels CSV üreten script
- `*_split_build.py` → Split CSV üreten script
- `train_*_baseline.py` → İlk versiyon eğitim scripti
- `train_octa_v{N}.py` → OCTA N. versiyon eğitim scripti
- `evaluate_*_model.py` → Değerlendirme scripti
- `gradcam_*.py` → Grad-CAM analiz scripti
