# Üveit Karar Destek Sistemi — Proje Tam Teknik Raporu

**Tarih:** 14 Mayıs 2026  
**Ortam:** Python 3.13, PyTorch 2.x, macOS (Apple Silicon MPS), FastAPI  
**Durum:** Faz-1 Tamamlandı ✅ — 5 Uzman Model + Otonom Router + Gemini AI + PDF Raporlama

---

## 1. Projenin Amacı

Bu proje, üveit göz hastalığının tanı süreçlerinde göz hekimlerine yardımcı olacak bir **yapay zekâ tabanlı multimodal karar destek sistemi** geliştirmeyi hedeflemektedir. 5 farklı oftalmolojik görüntüleme modalitesinden (Slit-lamp, OCTA, CFP, B-scan OCT, AS-OCT) elde edilen verilerle bağımsız uzman modeller eğitilmiştir. Sisteme ayrıca kullanıcının manuel cihaz seçmesini gerektirmeyen **Otonom Modality Router** ve analiz sonuçlarını klinik kalitede raporlayan **Gemini 2.5 Flash AI** entegre edilmiştir.

**Proje fazları:**
- **Faz-1 (Tamamlandı ✅):** 5 unimodal uzman model + MobileNetV3 router + interaktif web demo + Gemini AI + PDF raporlama
- **Faz-2 (Planlanıyor):** Multimodal füzyon (Early/Late Fusion)

---

## 2. Proje Dizin Yapısı

```
uveitis_project/
├── app/                              # Demo Web Arayüzü (FastAPI)
│   ├── main.py                       # API sunucu + Gemini /api/generate_comment endpoint
│   ├── inference.py                  # 7 model yöneten Inference Motoru
│   ├── templates/index.html          # Glassmorphism Dark Mode UI
│   └── static/
│       ├── js/app.js                 # Slider, Modallar, Session History, PDF, Gemini
│       └── css/style.css             # Responsive tasarım + AI comment + PDF buton stilleri
├── data_raw/                         # Orijinal ham veriler (dokunulmaz)
├── data_work/                        # Eğitime hazır temiz veriler
│   ├── slitlamp_clean/               # Uveitis (193) + Non-Uveitis (1,116)
│   ├── octa_clean/                   # Uveitis (180) + Control (345)
│   ├── cfp_clean/                    # Uveitis (63) + Non-Uveitis (807)
│   ├── bscan_oct_clean/              # Orijinal 55 + Sentetik 1,080 + Kermany Normal 9,632
│   ├── mcoa_clean/                   # AS-OCT MCOA (6,272 görüntü)
│   ├── asoct_seg/                    # AIDK segmentasyon verisi (U-Net eğitimi)
│   ├── router_data/                  # 5 sınıf × 100 = 500 dengeli router verisi
│   └── demo_images/                  # Web demo örnek vakalar (5 cihaz × ~100 resim)
├── metadata/                         # Etiket ve split CSV dosyaları
├── src/                              # PyTorch Dataset sınıfları
├── preprocessing/                    # Veri temizleme ve CSV üretme scriptleri
├── training/                         # Eğitim scriptleri
│   ├── train_slitlamp_baseline.py
│   ├── train_octa_v3.py
│   ├── train_cfp_baseline.py
│   ├── train_bscan_v4_kfold.py       # Kermany + Sentetik + K-Fold
│   ├── train_bscan_pretrain_kermany.py
│   ├── generate_synthetic_bscan.py   # Sentetik veri üretici
│   ├── train_mcoa_classifier.py      # AS-OCT EfficientNet eğitimi
│   ├── train_asoct_segmentation.py   # AS-OCT U-Net segmentasyon eğitimi
│   └── train_router.py               # MobileNetV3 router eğitimi
├── evaluation/                       # Değerlendirme, Grad-CAM, makale görseli üretimi
├── outputs/
│   ├── models/                       # .pth ağırlık dosyaları (tüm modeller)
│   ├── slitlamp/                     # Metrik + Grad-CAM + Pub Plots
│   ├── octa/                         # Metrik + Grad-CAM + Pub Plots
│   ├── cfp/                          # Metrik + Grad-CAM + Pub Plots
│   ├── bscan/                        # Metrik + Grad-CAM + Pub Plots
│   └── mcoa/                         # AS-OCT çıktıları (sınıflandırma + segmentasyon)
├── reports/
│   ├── phase1_unimodal/
│   │   ├── SLITLAMP_FINAL_RAPOR.md
│   │   ├── OCTA_FINAL_RAPOR.md
│   │   ├── CFP_FINAL_RAPOR.md
│   │   ├── BSCAN_OCT_FINAL_RAPOR.md
│   │   ├── ASOCT_FINAL_RAPOR.md      # AS-OCT çift görevli mimari raporu
│   │   └── datasets_used.md          # Tüm veri setleri, kaynaklar ve makale atıfları
│   └── proje_tam_rapor.md            # → README.md'ye yönlendirme
├── .env                              # GEMINI_API_KEY (git'e dahil edilmez)
├── requirements.txt
└── README.md                         # Bu dosya
```

---

## 3. Sistem Mimarisi

```
Kullanıcı Görüntüsü
        │
        ▼
┌─────────────────────────────┐
│   Modality Router           │  MobileNetV3-Small
│   (Otomatik Tespit)         │  Acc: %100 | 5 Sınıf
└────────────┬────────────────┘
             │  Tespit edilen modalite
     ┌───────┼───────────────────────────┐
     ▼       ▼       ▼       ▼          ▼
 Slit-  B-scan  OCTA    CFP        AS-OCT
 lamp   OCT                    (EfficientNet
                               + U-Net)
     └───────┴───────────────────────────┘
             │
             ▼
     ┌───────────────┐
     │  Grad-CAM /   │  Isı haritası veya
     │  U-Net Mask   │  anatomik segmentasyon
     └───────┬───────┘
             │
             ▼
     ┌───────────────────────────────────┐
     │  Gemini 2.5 Flash                 │
     │  Klinik AI Raporlama              │
     │  (EHR-kalitesinde ikinci görüş)   │
     └───────┬───────────────────────────┘
             │
             ▼
     ┌───────────────────────────────────┐
     │  PDF Klinik Rapor İndirme         │
     │  (html2pdf.js, istemci tarafı)    │
     └───────────────────────────────────┘
```

---

## 4. Ortak Geliştirme Pipeline'ı

Her modalite için aynı sistematik akış uygulanmıştır:

1. **Veri Temizliği:** Ham veriler incelenmiş, `data_work/` altında ikili sınıf yapısına dönüştürülmüştür.
2. **Etiketleme:** `preprocessing/*_labels_build.py` scriptleri ile metadata CSV üretilmiştir.
3. **Train/Val/Test Split:** %70/%15/%15 stratified split. OCTA'da `group_id` bazlı GroupKFold ile data leakage önlenmiştir.
4. **Dataset Sınıfı:** `src/` altında PyTorch `Dataset` sınıfları yazılmıştır.
5. **Model Eğitimi:** Transfer learning + fine-tuning. B-scan için üç katmanlı domain transfer (ImageNet → Kermany → Üveit).
6. **Değerlendirme:** Test metrikleri, confusion matrix, ROC eğrisi ve Grad-CAM/U-Net görüntüleri üretilmiştir.
7. **TTA Optimizasyonu (OCTA, CFP):** Eğitim sonrası Test Time Augmentation ile ek performans kazanımı sağlanmıştır.

---

## 5. Modaliteler ve Veri Setleri

### 5.1 Slit-lamp (Ön Segment Fotoğrafı)

| Bilgi | Değer |
|-------|-------|
| Toplam görüntü | **1,309** |
| Sınıflar | Uveitis (193), Cataract (357), Conjunctivitis (318), Eyelid (441) |
| Binary problem | Uveitis vs Non-Uveitis (1:5.8 dengesizlik) |
| Split | Train: 916, Val: 196, Test: 197 |
| Veri Kaynağı | Eye Diseases Classification Dataset (Mendeley Data) |

### 5.2 OCTA (OCT Anjiyografi)

| Bilgi | Değer |
|-------|-------|
| Toplam görüntü | **525** (Heidelberg 338 + OptoVue 187) |
| Sınıflar | Uveitis (180: aktif + inaktif), Control (345) |
| Katmanlar | Superficial + Deep (V3'ten itibaren her ikisi) |
| Data leakage önlemi | `group_id` bazlı GroupKFold |
| Veri Kaynağı | OCTOPUS / Behçet OCTA (MPG Edataverse) + UT-OCTA |

### 5.3 CFP (Renkli Fundus Fotoğrafı)

| Bilgi | Değer |
|-------|-------|
| Toplam görüntü | **870** |
| Sınıflar | Uveitis (63: VKH+CRS+RS), Non-Uveitis (807) |
| Sınıf dengesizliği | **1:12.8** (projedeki en yüksek oran) |
| Split | Train: 609, Val: 130, Test: 131 |
| Veri Kaynağı | 1000 Fundus Images (OIA-ODIR) + RFMiD 2.0 (IEEE Dataport) |

### 5.4 B-scan OCT

| Bilgi | Değer |
|-------|-------|
| Orijinal klinik veri | **55** (27 üveit, 28 normal) |
| Sentetik üveit | **1,080** (40× augmentation) |
| Kermany domain pre-training | **~108,312** görüntü (CNV, DME, Drusen, Normal) |
| K-Fold aggregated test | **9,659** görüntü (5 fold) |
| Veri Kaynağı | Djatikusumo & Hanna OCT Dataset + Kermany 2018 (Mendeley) |

**Üç Aşamalı Transfer Learning:**
```
ImageNet → ResNet-18 (genel görsel)
    → Kermany fine-tuning (retinal OCT domain)
        → Üveit binary (hedef görev)
```

### 5.5 AS-OCT (Ön Segment OCT) — Çift Görevli Mimari

| Bilgi | Değer |
|-------|-------|
| Sınıflandırma verisi | **6,272** (MCOA — Macaque Corneal Opacity) |
| Segmentasyon verisi | **1,168** (AIDK — piksel düzeyli segmentasyon maskesi) |
| Mimari | **EfficientNet-B0** (Noisy Student) + **U-Net** (segmentasyon) |
| XAI çıktısı | Grad-CAM ısı haritası + anatomik segmentasyon maskesi (toggle) |
| Veri Kaynağı | MCOA Dataset + AIDK Dataset |

### 5.6 Modality Router

| Bilgi | Değer |
|-------|-------|
| Eğitim verisi | **500** (5 cihaz × 100 dengeli orijinal klinik görüntü) |
| Mimari | MobileNetV3-Small (ImageNet pretrained) |
| Görev | Yüklenen görüntünün cihaz türünü tespit edip ilgili modele yönlendirme |
| Sonuç | **Acc: %100** |

---

## 6. Final Model Performansı

| Model | Backbone | Veri (n) | Accuracy | F1 | AUC | Recall |
|-------|----------|:--------:|:--------:|:--:|:---:|:------:|
| Slit-lamp | EfficientNet-B0 (ImageNet) | 1,309 | 96.95% | 0.900 | 0.988 | 93.1% |
| OCTA V5-TTA | ResNet-18 (ImageNet) | 525 | 84.34% | 0.780 | 0.910 | 82.1% |
| CFP + TTA + t=0.68 | EfficientNet-B0 (ImageNet) | 870 | 99.24% | **0.947** | **0.998** | **100%** |
| B-scan V4 K-Fold | ResNet-18 (Kermany PT) | 10,739 | 99.94% | 0.900 | **1.000** | **100%** |
| AS-OCT (MCOA) | EfficientNet-B0 (Noisy Student) | 6,272 | 88.54% | 0.868 | **0.970** | 78.7% |
| Router | MobileNetV3-Small | 500 | **100%** | — | — | — |

---

## 7. Eğitim Stratejileri Özeti

| Teknik | Modaliteler | Amaç |
|--------|-------------|------|
| `BCEWithLogitsLoss + pos_weight` | Tümü | Sınıf dengesizliği yönetimi |
| `WeightedRandomSampler` | CFP | Çift katmanlı dengeleme (1:12.8 için) |
| Progressive Fine-Tuning | OCTA, CFP | Backbone dondurup önce sadece head eğitme |
| Label Smoothing (ε=0.1) | OCTA, CFP, B-scan | Overconfidence önleme |
| Cosine Annealing LR | OCTA, B-scan | Stabilize öğrenme oranı |
| Test Time Augmentation (5×) | OCTA, CFP | Yeniden eğitim gerektirmeden performans artışı |
| Optimal Threshold (t=0.68) | CFP | Validation üzerinden F1-maksimize eden eşik |
| 5-Fold Cross Validation | B-scan | 27 gerçek üveit örneğinde güvenilir değerlendirme |
| Noisy Student Pre-training | AS-OCT | OCT speckle gürültüsüne karşı dayanıklılık |
| GroupKFold (group_id bazlı) | OCTA | Data leakage engelleme |

---

## 8. Web Demo Uygulaması

### 8.1 Backend (FastAPI + Gemini)

- **`app/inference.py`:** Başlatmada 7 modeli (5 hastalık + 1 router + 1 U-Net segmentasyon) RAM'e yükler. `modality="auto"` geldiğinde MobileNetV3 ile cihaz tespiti yapar → ilgili modele yönlendirir.
- **Grad-CAM:** Her tahminle birlikte otomatik üretilir. AS-OCT'de U-Net maskesi de üretilir; arayüzde toggle ile geçiş yapılır.
- **`/api/generate_comment`:** Tahmin sonuçlarını ve model metriklerini Gemini 2.5 Flash'a göndererek EHR-kalitesinde akademik klinik yorum alır. AS-OCT için özel prompt (U-Net segmentasyonu, MCOA terminolojisi).

### 8.2 Frontend

- **Glassmorphism Dark Mode** temalı tek sayfalık uygulama (Vanilla CSS + JS)
- **İnteraktif Grad-CAM Slider:** Orijinal ve ısı haritası görüntüleri fare ile karşılaştırılır
- **Gemini AI Rapor Kutusu:** Daktilo efektiyle yazılan akademik klinik yorum
- **PDF Klinik Rapor İndirme:** `html2pdf.js` ile istemci tarafında — orijinal görüntü, Grad-CAM/segmentasyon, Gemini yorumu ve model metrikleri içeren tam A4 PDF çıktısı
- **Akademik Teknik Modallar:** Her modalite kartında detaylı eğitim bilgisi pop-up'ı
- **Session History:** Art arda analizler thumbnail galeri olarak kaydedilir
- **Otomatik Tespit:** "🤖 Otomatik Tespit" seçeneği ile kullanıcı cihaz seçmez

### 8.3 Çalıştırma

```bash
# Gereksinimler
pip install -r requirements.txt

# .env dosyası oluştur (Gemini için)
echo "GEMINI_API_KEY=your_key_here" > .env

# Sunucuyu başlat
uvicorn app.main:app --reload
# → http://localhost:8000
```

---

## 9. Önemli Teknik Notlar

### Data Leakage Önlemi
- **OCTA:** `group_id` bazlı GroupKFold — aynı hastanın tüm görüntüleri aynı split'te kalır
- **B-scan:** `is_synthetic` flag ile sentetik veriler kesinlikle test setine dahil edilmez

### AS-OCT iCloud Kilitlenmesi Çözümü
`signal.alarm` ile OS düzeyinde 2 saniyelik I/O timeout uygulanmıştır. Kilitlenmede dosya siyah piksel matrisiyle doldurulur, sistem çökmez.

### Noisy Student vs ImageNet
AS-OCT'de `timm` kütüphanesinin JFT-300M pre-trained EfficientNet-B0'ı tercih edilmiştir. OCT speckle gürültüsüne karşı standart ImageNet ön eğitimine kıyasla daha iyi dayanıklılık sağlar.

### Gemini Prompt Stratejisi
- 5 modalite için ayrı terminoloji ve XAI açıklaması
- AS-OCT için "Grad-CAM" yerine "U-Net Anatomik Segmentasyon Haritası" ifadesi
- Model metrikleri (F1, AUC, eğitim verisi) prompta dahil edilir
- "Sorumluluk reddi cümlesi kurma" kuralı — arayüzde ayrıca gösterildiği için Gemini'nin yinelemesi önlenir

---

## 10. Model Dosyaları

| Dosya | Boyut | Model |
|-------|:-----:|-------|
| `slitlamp_efficientnetb0_best.pth` | 16 MB | Slit-lamp final |
| `octa_v3_resnet18_best.pth` | 43 MB | OCTA final (TTA bu ağırlıkları kullanır) |
| `cfp_efficientnetb0_best.pth` | 16 MB | CFP final (TTA + t=0.68) |
| `bscan_kermany_resnet18_pretrained.pth` | 43 MB | Kermany domain backbone |
| `bscan_v4_kfold_best.pth` | 43 MB | B-scan K-Fold final |
| `mcoa_efficientnet_best.pth` | 16 MB | AS-OCT sınıflandırıcı |
| `asoct_unet_best.pth` | ~59 MB | AS-OCT U-Net segmentasyon |
| `router_mobilenet_best.pth` | ~9 MB | Modality Router |

---

## 11. Geliştirme Geçmişi

| Tarih | Oturum | Yapılan İşler |
|-------|--------|---------------|
| 24 Nisan | OCTA V3 | 525 görüntü, F1:0.754, AUC:0.901 |
| 25 Nisan | Temizlik | Eski V1/V2 silindi (~250 MB kazanç) |
| 26 Nisan | CFP | 870 görüntü, F1:0.783, AUC:1.000 (baseline) |
| 1–2 Mayıs | Demo v1 | FastAPI + Dark Mode UI (temel sürüm) |
| 10 Mayıs | B-scan V4 | Kermany PT + Sentetik + K-Fold, F1:0.900 |
| 10 Mayıs | OCTA V5-TTA | Test Time Augmentation, F1:0.780 |
| 10 Mayıs | CFP TTA | Optimal threshold t=0.68, F1:0.947 |
| 10 Mayıs | Slit-lamp | Final rapor + pub plots tamamlandı |
| 11 Mayıs | AS-OCT | MCOA 6,272 görüntü, F1:0.920, U-Net segmentasyon |
| 11 Mayıs | Router | MobileNetV3, Acc:%100 |
| 11 Mayıs | Demo v2 | Slider, Modallar, Auto tespit, Session History |
| 12 Mayıs | Router v2 | 5 × 100 homojen veri, yeniden eğitim, Acc:%100 |
| 12 Mayıs | AS-OCT XAI | Classification + Segmentation toggle arayüzde |
| 12 Mayıs | Gemini AI | Gemini 2.5 Flash, /api/generate_comment endpoint |
| 14 Mayıs | PDF Rapor | html2pdf.js ile istemci tarafı klinik PDF indirme |
| 14 Mayıs | Dokümantasyon | Tüm 5 modalite raporu revize + datasets_used.md güncellendi |

---

## 12. Yapılacaklar (Faz-2)

1. **Multimodal Füzyon:** 5 modelin logit/olasılık çıktılarını birleştiren Late Fusion modeli tasarımı ve eğitimi

---

## 13. Detaylı Modalite Raporları

Her modalite için kapsamlı, makale kalitesinde final raporları mevcuttur:

| Rapor | İçerik |
|-------|--------|
| [`SLITLAMP_FINAL_RAPOR.md`](reports/phase1_unimodal/SLITLAMP_FINAL_RAPOR.md) | 1,309 görüntü, EfficientNet-B0, F1:0.900 |
| [`OCTA_FINAL_RAPOR.md`](reports/phase1_unimodal/OCTA_FINAL_RAPOR.md) | 525 görüntü, ResNet-18 + TTA, F1:0.780 |
| [`CFP_FINAL_RAPOR.md`](reports/phase1_unimodal/CFP_FINAL_RAPOR.md) | 870 görüntü, TTA + t=0.68, F1:0.947 |
| [`BSCAN_OCT_FINAL_RAPOR.md`](reports/phase1_unimodal/BSCAN_OCT_FINAL_RAPOR.md) | 10,739 görüntü, Kermany + K-Fold, F1:0.900 |
| [`ASOCT_FINAL_RAPOR.md`](reports/phase1_unimodal/ASOCT_FINAL_RAPOR.md) | 6,272 görüntü, Çift görevli EfficientNet+U-Net, F1:0.920 |
| [`datasets_used.md`](reports/phase1_unimodal/datasets_used.md) | Tüm veri setleri, kaynaklar ve makale atıfları |

---

*Bu dosya projenin canlı teknik referans belgesidir. Her oturum sonunda güncellenmektedir.*
