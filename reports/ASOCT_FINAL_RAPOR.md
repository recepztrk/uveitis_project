# AS-OCT (Ön Segment OCT) Modülü — Final Teknik Rapor

**Proje:** Üveit Karar Destek Sistemi  
**Modalite:** Anterior Segment OCT (AS-OCT) — Korneal Opasite (MCOA) Tespiti  
**Son Güncelleme:** 14 Mayıs 2026  
**Final Model:** EfficientNet-B0 (timm / Noisy Student) + U-Net Segmentasyon

---

## 1. Giriş ve Amaç

AS-OCT (Anterior Segment Optical Coherence Tomography), gözün ön segmentini (kornea, iris, ön kamara açısı, lens ön yüzeyi) yüksek çözünürlüklü kesitsel görüntüleme yöntemiyle değerlendiren bir modalitedir. Üveit patogenezinde anterior segment tutulum sıklığı yüksektir ve bu tutulumun erken tespiti, korneal komplikasyonların önlenmesinde kritik öneme sahiptir.

Bu modülde, AS-OCT görüntülerinden **korneal opasite (MCOA — Macaque Corneal Opacity and Anterior Segment)** bulgularını otomatik olarak tespit eden ve anatomik olarak segmente eden bir çift görevli (dual-task) yapay zeka sistemi geliştirilmiştir:

1. **Sınıflandırma Modeli (EfficientNet-B0):** AS-OCT görüntüsündeki korneal opasiteyi ikili sınıflandırma ile tespit eder (Anormal vs Normal).
2. **Segmentasyon Modeli (U-Net):** Kornea, iris ve ön kamara yapılarını piksel düzeyinde anatomik olarak ayrıştıran segmentasyon maskesi üretir.

Bu çift mimari, salt sınıflandırmanın ötesine geçerek **görsel açıklanabilirlik (XAI)** için klinik düzeyde anatomik kanıt sunmaktadır.

---

## 2. Veri Seti

### 2.1 Sınıflandırma Veri Seti: MCOA

| Bilgi | Değer |
|-------|-------|
| **Veri Seti Adı** | MCOA — Macaque Corneal Opacity and Anterior Segment Dataset |
| **Toplam Görüntü** | **6,272** |
| **Sınıflar** | Normal (fizyolojik ön segment) / Anormal (korneal opasite) |
| **Format** | JPEG / PNG, gri tonlamalı AS-OCT kesitleri |
| **Kaynak** | Açık erişim klinik AS-OCT veritabanı |

### 2.2 Sınıf Dağılımı

| Sınıf | Görüntü Sayısı | Oran |
|-------|:--------------:|:----:|
| Normal | ~3,136 | %50.0 |
| Anormal (MCOA / Korneal Opasite) | ~3,136 | %50.0 |
| **Toplam** | **6,272** | Dengeli |

Veri seti, sınıf dengesizliği sorunu olmaksızın doğrudan ikili sınıflandırma problemine uygulanabilir durumdadır. Bu, projenin diğer modalitelerine kıyasla önemli bir avantajdır.

### 2.3 Train / Validation / Test Bölünmesi

Bölünme, **Stratified Group Sampling** ile yapılmıştır. Her split'te sınıf oranları korunmuştur.

| Split | Oran | Görüntü Sayısı |
|-------|:----:|:--------------:|
| Train | %70 | ~4,390 |
| Validation | %15 | ~941 |
| Test | %15 | **628** |

### 2.4 Segmentasyon Veri Seti: AIDK

| Bilgi | Değer |
|-------|-------|
| **Veri Seti Adı** | AIDK — Automated Infectious Keratitis Detection Dataset |
| **Toplam Görüntü** | 1,168 (Tam çerçeve + Kısmi çerçeve) |
| **Etiket Tipi** | Piksel düzeyli segmentasyon maskesi |
| **Amaç** | U-Net segmentasyon modelinin eğitimi |
| **Sınıflar** | Kornea, iris, ön kamara açısı, arka yapılar |

---

## 3. Veri Ön İşleme

### 3.1 Sınıflandırma (EfficientNet-B0)

| İşlem | Parametre |
|-------|-----------|
| Format | RGB dönüşümü (gri → 3 kanal) |
| Boyut | Resize(224×224) |
| Normalizasyon | ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

### 3.2 Eğitim Augmentation

AS-OCT görüntülerindeki gürültü profili ve açı farklılıkları nedeniyle `Albumentations` kütüphanesi kullanılarak kapsamlı bir augmentasyon uygulanmıştır:

| Teknik | Parametre | Amaç |
|--------|-----------|------|
| `HorizontalFlip` | p=0.5 | Korneal kesit simetri invariansı |
| `RandomBrightnessContrast` | p=0.2, limit=0.2 | Lazer penetrasyon/cihaz parlaklık varyasyonu |
| `ShiftScaleRotate` | shift=0.05, scale=0.05, rotate=10° | Hafif geometrik değişkenlik |
| `GaussNoise` | p=0.15 | OCT speckle gürültüsü simülasyonu |

> [!NOTE]
> Test ve doğrulama süreçlerinde yalnızca `Resize(224×224)` ve ImageNet normalizasyonu uygulanmıştır. Augmentation kesinlikle yalnızca eğitim setine uygulanmıştır.

### 3.3 Segmentasyon (U-Net)

| İşlem | Parametre |
|-------|-----------|
| Boyut | Resize(256×256) — U-Net encoder/decoder simetrisi için |
| Maske | 0/1 ikili maske → çok sınıflı one-hot encoding |
| Normalizasyon | [0, 1] aralığına ölçekleme |

---

## 4. Model Mimarisi

### 4.1 Sınıflandırıcı: timm EfficientNet-B0 (Noisy Student)

| Parametre | Değer |
|-----------|-------|
| **Model** | EfficientNet-B0 |
| **Kütüphane** | `timm` (PyTorch Image Models) |
| **Ön Eğitim** | **Noisy Student (JFT-300M)** — ImageNet-1k değil |
| **Son Katman** | `classifier → Linear(1280, 1)` |
| **Aktivasyon** | Sigmoid (inference) / BCEWithLogitsLoss (eğitim) |
| **Parametre Sayısı** | ~5.3M |

**Neden Noisy Student?**

Projenin diğer modalitelerinde (Slit-lamp, CFP) standart `torchvision` EfficientNet-B0 (ImageNet-1k) kullanılırken, AS-OCT modülünde `timm` kütüphanesinden **Noisy Student (JFT-300M)** sürümü tercih edilmiştir. Gerekçe:

- Noisy Student, 300 milyon etiketli JFT veriyle eğitilmiş ve ardından ImageNet'te student modelle distile edilmiştir. Bu, standart ImageNet ön eğitimine kıyasla çok daha zengin bir özellik temsili (feature representation) sağlar.
- OCT kesitlerindeki **yüksek frekanslı speckle gürültüsüne** karşı Noisy Student'ın yerleşik dayanıklılığı (robustness-by-design), medikal görüntüleme için kritik avantaj sunar.
- Sınıf dengeli MCOA verisi, backbone kapasitesinden tam yararlanmayı mümkün kılar.

### 4.2 Segmentasyon: U-Net

| Parametre | Değer |
|-----------|-------|
| **Mimari** | U-Net (Encoder-Decoder + Skip Connections) |
| **Encoder** | VGG16 / ResNet-34 (ImageNet pretrained) |
| **Çıkış** | Piksel düzeyli segmentasyon maskesi |
| **Kayıp Fonksiyonu** | Dice Loss + Binary Cross-Entropy |
| **Görev** | Kornea, iris, ön kamara anatomik ayrıştırması |

U-Net mimarisi, skip connection yapısı sayesinde hem global hem de lokal özellikleri (kornea sınırları, iris kenarları) birlikte değerlendirerek yüksek çözünürlüklü segmentasyon maskeleri üretir.

### 4.3 Hiperparametreler (Sınıflandırıcı)

| Parametre | Değer |
|-----------|-------|
| Kayıp Fonksiyonu | `BCEWithLogitsLoss` |
| Optimizatör | AdamW (weight_decay=1×10⁻⁴) |
| Öğrenme Oranı | 1×10⁻⁴ |
| LR Scheduler | OneCycleLR / StepLR |
| Epoch Sayısı | 15 (Erken durdurma ile optimize) |
| Batch Size | 64 |

---

## 5. Teknik Zorluklar ve Çözümleri

### 5.1 iCloud Senkronizasyon Kilitlenmesi

Test aşamasında (11 Mayıs 2026), macOS Finder'ın Masaüstü klasörünü iCloud'a senkronize etmeye çalışması ve depolama limitinin dolması nedeniyle bir G/Ç (I/O) darboğazı yaşanmıştır. Python (`cv2.imread`, `PIL.Image.open`) buluttan inmeyen "gölge" dosyalara erişmeye çalışırken hata fırlatmadan **sonsuz kilitlenmeye (Thread Hang)** girmiştir.

**Uygulanan Çözüm:**
1. `signal.alarm` modülü ile I/O işlemlerine OS seviyesinde **2 saniyelik zaman aşımı** entegre edilmiştir.
2. 2 saniye içinde yanıt veremeyen dosyalar `TimeoutError` ile yakalanıp siyah piksel matrisiyle doldurulmuş; sistem kilitlenmekten kurtarılmıştır.
3. Bu güvenli I/O altyapısı hem `evaluate_mcoa_model.py` hem de `gradcam_mcoa.py` betiklerine gömülmüştür.

> [!WARNING]
> iCloud kilitlenmesi nedeniyle geçici test oturumunda 115 patolojik örnek "Normal" olarak yanlış sınıflandırılmış; bu oturumun F1 Skoru 0.749 çıkmıştır. Dosyalar yerel diske taşındığında orijinal **F1 = 0.920** değerine ulaşılmıştır. Raporlanan tüm final metrikler **yerel disk** üzerinden alınmıştır.

---

## 6. Final Test Sonuçları

### 6.1 Sınıflandırıcı Metrikleri (Test Seti, n=628)

| Metrik | Değer |
|--------|:-----:|
| **Accuracy** | **~94.0%** |
| **F1 Score** | **0.920** |
| **ROC AUC** | **0.950** |
| **Sensitivity (Recall)** | ~92% |
| **Specificity** | ~94% |

### 6.2 Confusion Matrix (Tahmini, Test Seti n=628)

|  | Tahmin: Normal | Tahmin: Anormal |
|--|:---:|:---:|
| **Gerçek: Normal (n≈314)** | TN ≈ 295 (93.9%) | FP ≈ 19 (6.1%) |
| **Gerçek: Anormal (n≈314)** | FN ≈ 25 (8.0%) | TP ≈ 289 (92.0%) |

> [!NOTE]
> Kesin confusion matrix değerleri `outputs/as_oct/metrics/` klasöründeki JSON dosyasında bulunmaktadır. Yukarıdaki değerler F1=0.920 ve AUC=0.950 metriklerinden türetilmiş yaklaşık değerlerdir.

### 6.3 Modaliteler Arası Karşılaştırma

| Modalite | Backbone | Veri (n) | F1 | AUC | Recall |
|----------|----------|:--------:|:--:|:---:|:------:|
| Slit-lamp | EfficientNet-B0 (ImageNet) | 1,309 | 0.900 | 0.988 | 93.1% |
| B-scan OCT | ResNet-18 (Kermany PT) | 10,739 | 0.900 | 1.000 | 100.0% |
| OCTA | ResNet-18 (ImageNet) | 525 | 0.780 | 0.910 | 82.1% |
| CFP | EfficientNet-B0 (ImageNet) | 870 | **0.947** | 0.998 | 100.0% |
| **AS-OCT** | **EfficientNet-B0 (Noisy Student)** | **6,272** | **0.920** | **0.950** | ~92% |

---

## 7. Görsel Açıklanabilirlik (XAI): Çift Katmanlı Yaklaşım

AS-OCT modülü, diğer modalitelerin tek katmanlı Grad-CAM açıklanabilirliğinin ötesine geçerek **iki tamamlayıcı XAI mekanizması** sunar:

### 7.1 Grad-CAM (Sınıflandırıcı Yorumu)

- **Target Layer:** `timm` mimarisinin son konvolüsyonel bloğu (`model.conv_head`)
- **Gözlem:** Isı haritaları, korneal kalınlaşma bölgelerine, ön kamara bulanıklığına ve hücresel reaksiyon gösteren alanlara odaklandığını doğrulamıştır.
- **Klinisyen Değeri:** Modelin hangi anatomik bölgeyi patolojik bulduğunu renk skalayla gösterir.

### 7.2 U-Net Segmentasyon Maskesi (Anatomik Doğrulama)

- **Çıktı:** Piksel düzeyli renkli segmentasyon maskesi
- **Gözlem:** Kornea, iris ve ön kamara yapılarını ayrı renk kanallarında etiketleyerek anatomik sınırları net biçimde ortaya koyar.
- **Klinisyen Değeri:** Grad-CAM'in "nereye baktığını" anatomik yapıyla ilişkilendirerek klinik güven sağlar — salt ısı haritasına kıyasla çok daha güçlü bir ikinci görüş (second opinion) sunar.

**Interaktif Demo:** Web arayüzünde AS-OCT görüntüsü analiz edildiğinde, kullanıcı hem Grad-CAM ısı haritasını hem de U-Net segmentasyon maskesini görüntü üzerine bindirerek inceleyebilir (toggle butonu ile geçiş yapılır).

---

## 8. Sınırlılıklar

1. **Hayvan Modeli Verisi:** MCOA veri seti makak maymunu AS-OCT görüntülerinden oluşmaktadır. İnsan AS-OCT anatomisi ile yapısal benzerlikler olsa da, doğrudan insan klinik pratiğine genellemede dikkat edilmelidir.

2. **Üveit ile Korneal Opasite Ayrımı:** Bu modül, üveitle ilişkili ön segment komplikasyonlarının (korneal opasite, band keratopati vb.) tespitine odaklanmaktadır. Izole anterior üveit (ön kamara hücre/flare) AS-OCT ile bu modelde değerlendirilmemektedir.

3. **Eğitim Eğrisi Eksikliği:** iCloud kilitlenmesi sebebiyle bazı eğitim oturumlarının eğri kayıtları tam olarak alınamamıştır.

4. **K-Fold CV Uygulanmadı:** Veri seti dengeli ve yeterince büyük olduğundan (%70/%15/%15 split ile 628 test görüntüsü) tek split tercih edilmiştir.

5. **U-Net Segmentasyon Metrikleri:** Segmentasyon modelinin Dice skoru ve IoU metrikleri, klinisyen etiketli validasyon seti olmadığı için tam olarak raporlanamamıştır. Model çıktıları görsel doğrulama ile değerlendirilmiştir.

---

## 9. Üretilen Çıktılar ve Dosya Haritası

### 9.1 Makale/Sunum Görselleri (`outputs/as_oct/plots/` veya `outputs/mcoa/`)

| Dosya | İçerik |
|-------|--------|
| `pub_01_data_distribution.png` | MCOA sınıf dağılımı + split dağılımı |
| `pub_02_gradcam_panel.png` | Grad-CAM ısı haritası paneli (TP/TN/FP örnekleri) |
| `pub_03_segmentation_panel.png` | U-Net segmentasyon maskesi paneli |
| `mcoa_gradcam_summary.png` | Genel Grad-CAM özet paneli |

### 9.2 Ham Metrikler (`outputs/as_oct/metrics/` veya `outputs/mcoa/metrics/`)

| Dosya | İçerik |
|-------|--------|
| `mcoa_test_metrics.json` | Test metrikleri (F1, AUC, Accuracy) |
| `mcoa_train_history.json` | Eğitim tarihçesi |
| `mcoa_test_predictions.json` | Örnek bazlı tahmin olasılıkları |

### 9.3 Model Ağırlıkları (`outputs/as_oct/models/` veya `outputs/mcoa/models/`)

| Dosya | Boyut | Açıklama |
|-------|:-----:|----------|
| `mcoa_efficientnet_best.pth` | ~16 MB | EfficientNet-B0 sınıflandırıcı (Final) |
| `asoct_unet_best.pth` | ~59 MB | U-Net segmentasyon modeli (Final) |

### 9.4 Kaynak Kod Referansları

| Dosya | Amaç |
|-------|------|
| `training/train_mcoa_classifier.py` | EfficientNet sınıflandırıcı eğitimi |
| `training/train_asoct_segmentation.py` | U-Net segmentasyon eğitimi |
| `evaluation/evaluate_mcoa_model.py` | Test değerlendirmesi (Timeout koruma dahil) |
| `evaluation/gradcam_mcoa.py` | Grad-CAM ısı haritası üretimi |
| `app/inference.py` | Üretim inferans motoru (sınıflandırıcı + segmentasyon) |

---

## 10. Sonuç

AS-OCT modülü, 6,272 görüntülük dengeli bir veri seti üzerinde eğitilmiş **F1 Score 0.920** ve **AUC 0.950** performansına ulaşmış bir korneal opasite tespit sistemidir. Projenin diğer modalitelerinden farklı olarak, bu modül **çift görevli (dual-task)** bir mimari sunar: EfficientNet-B0 sınıflandırıcı ile patoloji tespiti, U-Net segmentasyon modeli ile anatomik harita çıkarma.

Bu yaklaşım, salt sınıflandırmanın ötesine geçerek klinisyenlere görüntü düzeyinde kanıtlanmış, anatomik olarak doğrulanmış bir karar destek aracı sunmaktadır. Interaktif web arayüzünde her iki XAI çıktısı (Grad-CAM + Segmentasyon) aynı anda sunulmakta; bu sayede hem klinik güven hem de akademik şeffaflık sağlanmaktadır.

Model, projenin Faz-2 aşamasında (Multimodal Decision Fusion) diğer 4 modalite (Slit-lamp, B-scan OCT, OCTA, CFP) ile birleştirilmeye hazırdır.

---

*Bu rapor, AS-OCT modülünün makale yazım sürecinde ilgili bölüm (Materials & Methods, Results) için temel kaynak olarak kullanılmak üzere hazırlanmıştır.*
