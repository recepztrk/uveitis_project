# Üveit Karar Destek Sistemi

**Üniversite Bitirme Projesi** | Python 3.11+ · PyTorch 2.x · FastAPI · Gemini AI  
**Durum:** Faz-1 Tamamlandı ✅ — 5 Uzman Model + Otonom Router + Gemini AI + PDF Raporlama

---

## İçindekiler

1. [Proje Hakkında](#1-proje-hakkında)
2. [Sistem Mimarisi](#2-sistem-mimarisi)
3. [Dizin Yapısı](#3-dizin-yapısı)
4. [Modaliteler ve Veri Setleri](#4-modaliteler-ve-veri-setleri)
5. [Model Performansları](#5-model-performansları)
6. [Eğitim Stratejileri](#6-eğitim-stratejileri)
7. [Web Demo Uygulaması](#7-web-demo-uygulaması)
8. [Kurulum ve Çalıştırma](#8-kurulum-ve-çalıştırma)
9. [Model Dosyaları](#9-model-dosyaları)
10. [Teknik Notlar](#10-teknik-notlar)
11. [Proje Belgeleri](#11-proje-belgeleri)
12. [Detaylı Modalite Raporları](#12-detaylı-modalite-raporları)

---

## 1. Proje Hakkında

Bu proje, **üveit** göz hastalığının tanı süreçlerinde göz hekimlerine yardımcı olacak bir **yapay zekâ tabanlı multimodal karar destek sistemi** geliştirmeyi hedeflemektedir.

Üveit, gözün orta tabakasının (uvea) iltihaplanması ile seyreden, erken teşhis edilmezse kalıcı görme kaybına yol açabilen ciddi bir hastalıktır. Mevcut tanı süreci uzman bağımlı, zaman alıcı ve subjektif değerlendirmelere dayanmaktadır.

### Ne Yapıyor?

5 farklı oftalmolojik görüntüleme cihazından elde edilen görüntüler için bağımsız derin öğrenme modelleri eğitilmiştir. Sisteme yüklenen bir görüntü **otomatik olarak cihaz türü tespit edilip** ilgili uzman modele yönlendirilir, ardından klinik kalitede AI raporu üretilir.

### Proje Fazları

| Faz | Durum | İçerik |
|-----|-------|--------|
| **Faz-1** | ✅ Tamamlandı | 5 unimodal uzman model + MobileNetV3 Router + Web Demo + Gemini AI + PDF |
| **Faz-2** | 🔜 Planlanıyor | Multimodal füzyon (Early/Late Fusion) |

---

## 2. Sistem Mimarisi

```
Kullanıcı Görüntüsü
        │
        ▼
┌─────────────────────────────────────┐
│         Modality Router             │  MobileNetV3-Small
│   (Otomatik Cihaz Tespiti)          │  5 Sınıf · Acc: %100
└──────────────┬──────────────────────┘
               │  Tespit edilen cihaz türü
    ┌──────────┼──────────────────────────────┐
    ▼          ▼          ▼       ▼           ▼
 Slit-lamp  B-scan OCT  OCTA    CFP        AS-OCT
 Efficientnet ResNet-18 ResNet  Efficientnet (EfficientNet
 B0          (Kermany)  -18     B0          B0 + U-Net)
    └──────────┴──────────────────────────────┘
               │
               ▼
       ┌───────────────┐
       │   Grad-CAM    │   Isı haritası (tüm modeller)
       │   U-Net Mask  │   Anatomik segmentasyon (AS-OCT)
       └───────┬───────┘
               │
               ▼
       ┌─────────────────────────────────────┐
       │         Gemini 2.5 Flash            │
       │    EHR-kalitesinde klinik yorum     │
       └───────┬─────────────────────────────┘
               │
               ▼
       ┌─────────────────────────────────────┐
       │      PDF Klinik Rapor İndirme       │
       │  (html2pdf.js · istemci taraflı)    │
       └─────────────────────────────────────┘
```

---

## 3. Dizin Yapısı

```
uveitis_project/
├── app/                                # Web Demo Uygulaması (FastAPI)
│   ├── main.py                         # API route'ları + Gemini endpoint
│   ├── inference.py                    # 7 modeli yöneten Inference Engine
│   ├── templates/
│   │   └── index.html                  # Glassmorphism Dark Mode UI
│   └── static/
│       ├── js/app.js                   # Frontend mantığı (slider, modal, PDF, Gemini)
│       ├── css/style.css               # Responsive tasarım
│       └── img/                        # Karşılaştırma görselleri
│
├── src/                                # PyTorch Dataset Sınıfları
│   ├── slitlamp_dataset.py
│   ├── octa_dataset_v2.py
│   ├── cfp_dataset.py
│   ├── bscan_oct_dataset.py
│   ├── mcoa_dataset.py
│   ├── asoct_seg_dataset.py
│   └── kermany_dataset.py
│
├── training/                           # Model Eğitim Scriptleri
│   ├── train_slitlamp_baseline.py      # Slit-lamp EfficientNet-B0
│   ├── train_octa_v3.py                # OCTA ResNet-18 (Final)
│   ├── train_cfp_baseline.py           # CFP EfficientNet-B0
│   ├── train_bscan_v4_kfold.py         # B-scan K-Fold (Final)
│   ├── train_bscan_pretrain_kermany.py # Kermany domain ön eğitimi
│   ├── generate_synthetic_bscan.py     # B-scan sentetik veri üretici
│   ├── train_mcoa_classifier.py        # AS-OCT EfficientNet
│   ├── train_asoct_seg.py              # AS-OCT U-Net segmentasyon
│   ├── train_router.py                 # MobileNetV3 Router
│   ├── train_octa_v4_efficientnet.py   # OCTA EfficientNet denemesi (referans)
│   └── train_octa_kfold_cv.py          # OCTA K-Fold CV doğrulaması
│
├── preprocessing/                      # Veri Hazırlama Scriptleri
│   ├── octa_labels_build.py
│   ├── octa_split_build.py
│   ├── cfp_labels_build.py
│   ├── cfp_split_build.py
│   ├── slitlamp_labels_build.py
│   ├── slitlamp_split_build.py
│   ├── mcoa_split_build.py
│   ├── bscan_labels_v4_build.py
│   ├── bscan_kfold_split_v4.py
│   └── asoct_mask_builder.py           # U-Net eğitimi için maske üretici
│
├── evaluation/                         # Değerlendirme ve Görsel Üretim
│   ├── evaluate_octa_v5_tta.py         # OCTA TTA değerlendirmesi
│   ├── evaluate_cfp_tta.py             # CFP TTA + threshold değerlendirmesi
│   ├── evaluate_slitlamp_model.py
│   ├── evaluate_mcoa_model.py
│   ├── gradcam_octa_v3.py
│   ├── gradcam_cfp.py
│   ├── gradcam_slitlamp.py
│   ├── gradcam_mcoa.py
│   ├── visualize_asoct_seg.py
│   ├── generate_octa_publication_plots.py
│   ├── generate_cfp_publication_plots.py
│   ├── generate_slitlamp_publication_plots.py
│   ├── generate_mcoa_publication_plots.py
│   ├── generate_bscan_publication_plots.py
│   └── generate_all_models_comparison.py
│
├── metadata/                           # CSV Etiket ve Split Dosyaları
│   ├── slitlamp_labels.csv
│   ├── slitlamp_split.csv
│   ├── octa_labels.csv
│   ├── octa_split.csv
│   ├── cfp_labels.csv
│   ├── cfp_split.csv
│   ├── mcoa_split.csv
│   ├── bscan_oct_labels.csv
│   ├── bscan_oct_labels_v4.csv
│   └── bscan_oct_split_v4_kfold.csv
│
├── outputs/                            # Model Çıktıları
│   ├── models/                         # Paylaşılan .pth ağırlık dosyaları
│   ├── bscan/
│   │   ├── metrics/                    # JSON metrikler + confusion matrix
│   │   └── plots/                      # pub_01…pub_06 makale görselleri
│   ├── octa/
│   │   ├── gradcam/
│   │   ├── metrics/
│   │   ├── models/
│   │   └── plots/
│   ├── cfp/
│   │   ├── gradcam/
│   │   ├── metrics/
│   │   ├── models/
│   │   └── plots/
│   ├── slitlamp/
│   │   ├── gradcam/
│   │   ├── metrics/
│   │   ├── models/
│   │   └── plots/
│   └── mcoa/
│       ├── gradcam/
│       ├── metrics/
│       ├── models/
│       └── plots/
│
├── reports/                            # Teknik Raporlar (Markdown)
│   ├── BSCAN_OCT_FINAL_RAPOR.md
│   ├── OCTA_FINAL_RAPOR.md
│   ├── CFP_FINAL_RAPOR.md
│   ├── SLITLAMP_FINAL_RAPOR.md
│   ├── ASOCT_FINAL_RAPOR.md
│   └── datasets_used.md
│
├── docs/                               # Proje Belgeleri (PDF)
│   ├── YMH-Bitirme-Projesi-Öneri-Formu-2025-Recep-Ötürk.pdf
│   ├── YMH-Bitirme-Projesi-Ara-Tasarım-Raporu-2025-Recep-Öztürk.pdf
│   ├── recep_ozturk_arastirmaprojesi.pdf
│   ├── Recep_Ozturk_Uveitis_AI_Paper_Draft.pdf
│   └── uveitis_project_poster.pdf
│
├── scripts/
│   └── prepare_router_dataset.py       # Router eğitim verisi hazırlama
│
├── data_raw/                           # Ham veriler (git'e dahil değil)
├── data_work/                          # İşlenmiş veriler (git'e dahil değil)
├── .env                                # GEMINI_API_KEY (git'e dahil değil)
├── requirements.txt
└── README.md
```

---

## 4. Modaliteler ve Veri Setleri

### 4.1 Slit-lamp (Ön Segment Fotoğrafı)

Gözün ön segmentini (konjonktiva, kornea, ön kamara) aydınlatarak görüntüleyen yarık lamba fotoğrafıdır. Üveitte konjonktival hiperemi, silier enjeksiyon ve keratik presipitalar bu görüntüde izlenir.

| Parametre | Değer |
|-----------|-------|
| Toplam görüntü | **1,309** |
| Sınıflar | Uveitis (193), Cataract (357), Conjunctivitis (318), Eyelid (441) |
| Binary problem | Uveitis vs Non-Uveitis (1:5.8 dengesizlik) |
| Train / Val / Test | 916 / 196 / 197 |
| Backbone | EfficientNet-B0 (ImageNet pretrained) |
| Veri Kaynağı | Eye Diseases Classification Dataset (Mendeley Data) |

---

### 4.2 OCTA (Optik Koherens Tomografi Anjiyografi)

Retina ve koroiddeki damar ağını kontrast madde kullanmadan görüntüleyen non-invaziv modalitedir. Üveitte retinal damar yoğunluğunda azalma, FAZ genişlemesi ve kapiller perfüzyon bozuklukları OCTA'da gözlemlenir.

| Parametre | Değer |
|-----------|-------|
| Toplam görüntü | **525** (Heidelberg 338 + OptoVue 187) |
| Sınıflar | Üveit: 180 (aktif + inaktif), Kontrol: 345 |
| Katmanlar | Superficial + Deep (V3'ten itibaren her ikisi) |
| Data leakage önlemi | `group_id` bazlı GroupKFold |
| Backbone | ResNet-18 (ImageNet pretrained) |
| Veri Kaynağı | OCTOPUS / Behçet OCTA (MPG Edataverse) + UT-OCTA |

**Model Evrim Özeti:**

| Versiyon | Veri | F1 | AUC | Not |
|----------|------|----|-----|-----|
| V1 | 214 (sadece superficial) | 0.700 | 0.732 | — |
| V2 | 338 (superficial + deep) | 0.704 | 0.696 | — |
| V3 | 525 (+ OptoVue harici veri) | 0.754 | 0.901 | En büyük sıçrama |
| **V5-TTA** | 525 | **0.780** | **0.910** | TTA ile +0.026 F1 |

---

### 4.3 CFP (Renkli Fundus Fotoğrafı)

Retina yüzeyini geniş açıyla görüntüleyen fundus fotoğrafıdır. Posterior üveitte koroiditin chorioretinitis lezyonları, optik disk ödemi ve retinal vaskülit bu görüntüde tespit edilir.

| Parametre | Değer |
|-----------|-------|
| Toplam görüntü | **870** |
| Sınıflar | Üveit: 63 (VKH + CRS + RS), Non-Üveit: 807 |
| Sınıf dengesizliği | **1:12.8** (projedeki en yüksek oran) |
| Train / Val / Test | 609 / 130 / 131 |
| Backbone | EfficientNet-B0 (ImageNet pretrained) |
| Veri Kaynağı | 1000 Fundus Images (OIA-ODIR) + RFMiD 2.0 (IEEE Dataport) |

**Kritik Teknik Kararlar:**
- `WeightedRandomSampler` + `pos_weight` çift katmanlı dengeleme (1:12.8 için)
- Optimal threshold: **t=0.68** (Youden Index ile validation setinden hesaplandı)
- Bu eşik `inference.py`'de sabit olarak tanımlıdır — değiştirme

---

### 4.4 B-scan OCT

Retina katmanlarının kesitsel görüntülerini üreten modalitedir. Üveitte vitreus hücreleri, maküler ödem ve koroidal inflamasyon B-scan OCT ile değerlendirilir.

| Parametre | Değer |
|-----------|-------|
| Orijinal klinik veri | **55** (27 üveit, 28 normal) |
| Sentetik üveit | **1,080** (40× augmentation) |
| Kermany ön eğitim | ~108,312 görüntü (CNV, DME, Drusen, Normal) |
| K-Fold aggregated test | **9,659** görüntü (5 fold) |
| Backbone | ResNet-18 (**Kermany pretrained**) |
| Veri Kaynağı | Djatikusumo & Hanna OCT Dataset + Kermany 2018 (Mendeley) |

**Üç Aşamalı Transfer Learning Stratejisi:**
```
Katman 1: ImageNet → ResNet-18 (genel görsel özellikler)
    ↓
Katman 2: Kermany fine-tuning (retinal OCT domain bilgisi: ~108K görüntü)
    ↓
Katman 3: Üveit binary fine-tuning (hedef görev: 27 gerçek + 1080 sentetik)
```

**Neden bu strateji?**  
Gerçek üveit B-scan OCT verisi yalnızca 27 görüntüdür — derin öğrenme için yetersiz. Kermany veri setindeki 108K retinal OCT görüntüsü ile backbone, retinal katman yapısını, OCT artefaktlarını ve patolojik değişimleri öğrenir. Bu domain bilgisi aktarıldıktan sonra 27 gerçek + 1080 sentetik görüntü yeterli hale gelir.

**K-Fold Aggregated Sonuçlar:**

| Fold | Test n | F1 | Recall | Precision |
|:----:|:------:|:--:|:------:|:---------:|
| 1 | 1,932 | 1.000 | 1.000 | 1.000 |
| 2 | 1,932 | 0.909 | 1.000 | 0.833 |
| 3 | 1,931 | 0.857 | 1.000 | 0.750 |
| 4 | 1,932 | 0.800 | 1.000 | 0.667 |
| 5 | 1,932 | 1.000 | 1.000 | 1.000 |
| **Ortalama** | — | **0.913** | **1.000** | **0.850** |

---

### 4.5 AS-OCT (Ön Segment OCT) — Çift Görevli Mimari

Gözün ön segmentini (kornea, iris, ön kamara açısı) yüksek çözünürlüklü kesitsel görüntüleme yöntemiyle değerlendiren modalitedir. Üveit patogenezinde anterior segment tutulumu sık görülür.

| Parametre | Değer |
|-----------|-------|
| Sınıflandırma verisi | **6,272** (MCOA — Normal Cornea vs Opaque Cornea) |
| Segmentasyon verisi | **1,168** (AIDK — piksel düzeyli annotation) |
| Sınıflandırma modeli | EfficientNet-B0 (**Noisy Student** — `timm`) |
| Segmentasyon modeli | U-Net (encoder: ResNet-34) |
| XAI çıktısı | Grad-CAM ısı haritası + anatomik U-Net segmentasyon maskesi |
| Veri Kaynağı | MCOA Dataset + AIDK Dataset |

**Neden Noisy Student?**  
`timm` kütüphanesindeki `tf_efficientnet_b0.ns_jft_in1k` modeli, JFT-300M ile ön eğitim + self-training (noisy student) yöntemi ile eğitilmiştir. OCT görüntülerindeki speckle gürültüsüne karşı standart ImageNet ön eğitimine kıyasla belirgin şekilde daha dayanıklıdır.

> ⚠️ **Önemli:** `timm` model adı `tf_efficientnet_b0.ns_jft_in1k`'dir (eski deprecated ad: `tf_efficientnet_b0_ns`). `inference.py`'de bu şekilde tanımlıdır.

**iCloud Kilitlenmesi Çözümü:** `signal.alarm` ile OS düzeyinde 2 saniyelik I/O timeout uygulanmıştır. Kilitlenmede dosya siyah piksel matrisiyle doldurulur, sistem çökmez.

---

### 4.6 Modality Router (Otonom Cihaz Tespiti)

Kullanıcının hangi cihazdan görüntü yüklediğini otomatik tespit ederek ilgili hastalık modeline yönlendirir.

| Parametre | Değer |
|-----------|-------|
| Eğitim verisi | **500** (5 cihaz × 100 dengeli görüntü) |
| Backbone | MobileNetV3-Small (ImageNet pretrained) |
| Sınıflar | `slitlamp`, `bscan_oct`, `octa`, `cfp`, `as_oct` |
| Sonuç | Acc: **%100**, Val Loss: 0.0014 |

**Checkpoint Formatı:**  
```python
{'state_dict': ..., 'classes': ['as_oct', 'bscan_oct', 'cfp', 'octa', 'slitlamp']}
```
Sınıf listesi checkpoint içinde saklanır — harici bir dosyaya gerek yoktur.

---

## 5. Model Performansları

| Model | Backbone | Veri (n) | Accuracy | F1 | AUC | Recall |
|-------|----------|:--------:|:--------:|:--:|:---:|:------:|
| **Slit-lamp** | EfficientNet-B0 | 1,309 | 96.95% | 0.900 | 0.988 | 93.1% |
| **OCTA V5-TTA** | ResNet-18 | 525 | 84.34% | 0.780 | 0.910 | 82.1% |
| **CFP + TTA + t=0.68** | EfficientNet-B0 | 870 | 99.24% | **0.947** | **0.998** | **100%** |
| **B-scan V4 K-Fold** | ResNet-18 (Kermany) | 10,739 | 99.94% | 0.900 | **1.000** | **100%** |
| **AS-OCT (MCOA)** | EfficientNet-B0 (NS) | 6,272 | 88.54% | 0.868 | 0.970 | 78.7% |
| **Router** | MobileNetV3-Small | 500 | **100%** | — | — | — |

> **Not:** B-scan sonuçları 5-Fold aggregated değerlendirmeden gelmektedir (n=9,659 test görüntüsü). CFP metrikleri TTA + optimal threshold (t=0.68) sonrasına aittir.

### Karar Eşikleri (Thresholds)

| Model | Threshold | Açıklama |
|-------|:---------:|----------|
| Slit-lamp | 0.50 | Varsayılan |
| OCTA | 0.50 | Varsayılan |
| CFP | **0.68** | Youden Index optimizasyonu |
| B-scan | 0.50 | Varsayılan |
| AS-OCT | 0.50 | Varsayılan |

---

## 6. Eğitim Stratejileri

| Teknik | Kullanıldığı Modeller | Amaç |
|--------|----------------------|------|
| `BCEWithLogitsLoss + pos_weight` | Tümü | Sınıf dengesizliği yönetimi |
| `WeightedRandomSampler` | CFP | Çift katmanlı dengeleme (1:12.8) |
| Progressive Fine-Tuning | OCTA, CFP, B-scan | İlk 5 epoch sadece head, sonra full fine-tuning |
| Label Smoothing (ε=0.1) | OCTA, CFP, B-scan | Overconfidence önleme |
| Cosine Annealing LR | OCTA, B-scan | Stabilize öğrenme oranı |
| Test Time Augmentation (5×) | OCTA, CFP | Yeniden eğitim gerektirmeden performans artışı |
| Optimal Threshold (t=0.68) | CFP | Validation üzerinden F1-maksimize eden eşik |
| 5-Fold Cross Validation | B-scan | 27 gerçek üveit örneğinde güvenilir değerlendirme |
| Noisy Student Pre-training | AS-OCT | OCT speckle gürültüsüne dayanıklılık |
| GroupKFold (group_id bazlı) | OCTA | Data leakage engelleme |
| Kermany Domain Transfer | B-scan | Retinal OCT domain bilgisinin aktarımı |

### Ortak Pipeline

Her modalite için aynı sistematik akış uygulanmıştır:

1. **Veri Temizliği:** Ham veriler incelenip `data_work/` altında ikili sınıf yapısına dönüştürüldü
2. **Etiketleme:** `preprocessing/*_labels_build.py` ile metadata CSV üretildi
3. **Split:** %70/%15/%15 stratified split; OCTA'da `group_id` bazlı GroupKFold
4. **Dataset Sınıfı:** `src/` altında PyTorch `Dataset` sınıfları yazıldı
5. **Eğitim:** Transfer learning + progressive fine-tuning
6. **Değerlendirme:** Test metrikleri, confusion matrix, ROC eğrisi, Grad-CAM üretildi
7. **TTA Optimizasyonu (OCTA, CFP):** Eğitim sonrası ek performans kazanımı

---

## 7. Web Demo Uygulaması

### Backend (FastAPI)

**`app/inference.py`** — Başlatmada tüm modelleri RAM'e yükler:
- 5 hastalık modeli + 1 Router (MobileNetV3) + 1 U-Net segmentasyon
- `modality="auto"` ile Router önce cihaz tespiti yapar, ardından ilgili model devreye girer
- Her tahminle birlikte Grad-CAM ısı haritası otomatik üretilir
- AS-OCT'de U-Net maskesi ek olarak üretilir

**`/api/generate_comment`** — Gemini 2.5 Flash entegrasyonu:
- Tahmin sonuçları + model metrikleri Gemini'ye gönderilir
- 5 modalite için ayrı klinik terminoloji ve prompt stratejisi
- AS-OCT için "U-Net Anatomik Segmentasyon Haritası" ifadesi kullanılır
- Çoklu API key rotation ile kota yönetimi
- "Sorumluluk reddi cümlesi" engeli — arayüzde ayrıca gösterildiğinden yinelemesi önlenir

**Cihaz Seçimi:**  
Apple Silicon MPS otomatik kullanılır. Sıra: `CUDA → MPS → CPU`

### Frontend Özellikleri

| Özellik | Açıklama |
|---------|----------|
| **Glassmorphism Dark Mode** | Tek sayfalık uygulama, Vanilla CSS + JS |
| **Modality Kartları** | 5 cihaz + 1 "🤖 Otomatik Tespit" kartı (dinamik yüklenme) |
| **Grad-CAM Slider** | `clipPath` CSS ile orijinal/ısı haritası fare karşılaştırması |
| **AS-OCT Toggle** | Grad-CAM ↔ U-Net segmentasyon maskesi geçişi |
| **Gemini AI Raporu** | Daktilo (typewriter) efektli akademik klinik yorum |
| **PDF Klinik Rapor** | `html2canvas` + `jsPDF` ile A4 formatında istemci taraflı rapor |
| **Akademik Modallar** | Her cihaz kartında detaylı teknik bilgi pop-up'ı |
| **Session History** | Art arda analizler thumbnail galeri olarak kaydedilir |
| **Görüntü Kalite Skoru** | Laplacian varyansı + çözünürlük + kontrast; Yeşil/Sarı/Kırmızı badge |
| **Belirsizlik Bölgesi** | %40–%60 olasılıkta animasyonlu uyarı paneli |
| **Hastalık Bilgi Bölümü** | "Üveit Nedir?" — 6 kart: Anatomi, Belirtiler, Tanı, Tedavi, AI'nin Rolü, Önemi |

### TTA Notu

OCTA ve CFP modellerinde TTA uygulanmıştır ancak **demo arayüzünde gerçek zamanlı TTA çalışmaz** — yalnızca evaluation scriptlerinde. Demo'daki metrikler TTA sonuçlarını gösterir, ancak inference tek kopya üzerinden yapılır.

---

## 8. Kurulum ve Çalıştırma

### Gereksinimler

```bash
pip install -r requirements.txt
```

> **Not:** Model ağırlıkları (`.pth` dosyaları) git'e dahil değildir. Eğitim scriptlerini çalıştırarak üretmeniz gerekir.

### Ortam Değişkenleri

Proje kök dizininde `.env` dosyası oluşturun:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### Sunucuyu Başlatma

```bash
uvicorn app.main:app --reload
```

Tarayıcıda açın: **http://localhost:8000**

### Demo Kullanımı

1. Bir modalite kartı seçin (veya "🤖 Otomatik Tespit" seçin)
2. Görüntü yükleyin
3. Analiz sonuçlarını, Grad-CAM ısı haritasını ve AI yorumunu görün
4. Klinik raporu PDF olarak indirin

---

## 9. Model Dosyaları

Model ağırlıkları `.gitignore`'da tanımlıdır ve GitHub'a dahil edilmez. Aşağıdaki tabloda her modelin eğitim scripti ve çıktı yolu belirtilmiştir.

| Model | Ağırlık Dosyası | Boyut | Eğitim Scripti |
|-------|----------------|:-----:|----------------|
| Slit-lamp | `outputs/models/slitlamp_efficientnetb0_best.pth` | 16 MB | `train_slitlamp_baseline.py` |
| OCTA (V5-TTA) | `outputs/models/octa_v3_resnet18_best.pth` | 43 MB | `train_octa_v3.py` |
| CFP | `outputs/models/cfp_efficientnetb0_best.pth` | 16 MB | `train_cfp_baseline.py` |
| B-scan (K-Fold) | `outputs/models/bscan_v4_kfold_best.pth` | 43 MB | `train_bscan_v4_kfold.py` |
| B-scan Kermany Backbone | `outputs/models/bscan_kermany_resnet18_pretrained.pth` | 43 MB | `train_bscan_pretrain_kermany.py` |
| AS-OCT Sınıflandırıcı | `outputs/mcoa/models/mcoa_efficientnet_best.pth` | 16 MB | `train_mcoa_classifier.py` |
| AS-OCT U-Net | `outputs/mcoa/models/asoct_unet_best.pth` | ~59 MB | `train_asoct_seg.py` |
| Router | `outputs/models/router_mobilenet_best.pth` | ~6 MB | `train_router.py` |

---

## 10. Teknik Notlar

### Data Leakage Önlemleri

- **OCTA:** `group_id` bazlı GroupKFold — aynı hastanın tüm görüntüleri (superficial + deep, her iki göz) kesinlikle aynı split'te kalır
- **B-scan:** CSV'deki `is_synthetic` sütunu ile sentetik veriler split oluşturma aşamasında test setinden dışlanır

### Inference Engine Model Kayıt Yapısı

Tüm modeller `app/inference.py` içindeki `MODELS` sözlüğünde kayıtlıdır. Her kayıt şunları içerir:
- Backbone türü ve ağırlık dosyası yolu
- Görüntüleme adı ve klinik not
- F1, AUC, Recall gibi performans metrikleri
- Karar eşiği (threshold)
- Modalite-spesifik uyarı mesajı

**Yeni model eklemek için:** `MODELS` dict'ine entry ekle + `_build_model()` fonksiyonunda backbone desteği sağla.

### Cache Busting

`index.html`'de CSS ve JS dosyaları `?v=X.X` sorgu parametresiyle yüklenir. Değişiklik yapıldığında versiyon numarasını artırın, aksi hâlde tarayıcı eski cache'i kullanır.

### Gemini API Kota Yönetimi

`.env` dosyasına birden fazla API key eklenebilir:
```
GEMINI_API_KEY=key1
GEMINI_API_KEY_2=key2
GEMINI_API_KEY_3=key3
```
`inference.py` otomatik rotation ile kota aşımını önler.

---

## 11. Proje Belgeleri

Tüm resmi belgeler `docs/` klasöründe bulunmaktadır:

| Belge | Açıklama |
|-------|----------|
| [Proje Öneri Formu](docs/YMH-Bitirme-Projesi-Öneri-Formu-2025-Recep-Ötürk.pdf) | YMH resmi öneri formu |
| [Ara Tasarım Raporu](docs/YMH-Bitirme-Projesi-Ara-Tasarım-Raporu-2025-Recep-Öztürk.pdf) | Dönem ara tasarım raporu |
| [TÜBİTAK 2209-A Başvurusu](docs/recep_ozturk_arastirmaprojesi.pdf) | TÜBİTAK araştırma projesi |
| [Makale Taslağı](docs/Recep_Ozturk_Uveitis_AI_Paper_Draft.pdf) | Akademik makale taslağı |
| [Proje Posteri](docs/uveitis_project_poster.pdf) | Sunum posteri |

---

## 12. Detaylı Modalite Raporları

Her modalite için makale kalitesinde, kapsamlı teknik raporlar `reports/` klasöründedir:

| Rapor | İçerik |
|-------|--------|
| [SLITLAMP_FINAL_RAPOR.md](reports/SLITLAMP_FINAL_RAPOR.md) | 1,309 görüntü · EfficientNet-B0 · F1: 0.900 · AUC: 0.988 |
| [OCTA_FINAL_RAPOR.md](reports/OCTA_FINAL_RAPOR.md) | 525 görüntü · ResNet-18 + TTA · F1: 0.780 · AUC: 0.910 |
| [CFP_FINAL_RAPOR.md](reports/CFP_FINAL_RAPOR.md) | 870 görüntü · TTA + t=0.68 · F1: 0.947 · AUC: 0.998 |
| [BSCAN_OCT_FINAL_RAPOR.md](reports/BSCAN_OCT_FINAL_RAPOR.md) | 10,739 görüntü · Kermany + K-Fold · F1: 0.900 · AUC: 1.000 |
| [ASOCT_FINAL_RAPOR.md](reports/ASOCT_FINAL_RAPOR.md) | 6,272 görüntü · EfficientNet-B0 (NS) + U-Net · F1: 0.868 |
| [datasets_used.md](reports/datasets_used.md) | Tüm veri setleri, kaynaklar ve atıflar |

---

*Recep Öztürk · 2026 · Üveit Karar Destek Sistemi Bitirme Projesi*
