# Kullanılan Veri Setleri — Üveit Karar Destek Sistemi

Bu dosya, projenin her modalitesi için kullanılan veri setlerini, kaynaklarını, sınıf dağılımlarını ve kullanım amaçlarını belgeler. Makale atıfları ve metodoloji bölümü için referans dosyasıdır.

---

## 1. Slit-lamp (Biyomikroskopi) Modeli

**Final Model:** EfficientNet-B0 (ImageNet) | **F1: 0.900** | **AUC: 0.988**

| Veri Seti | Kaynak | Sınıflar | Görüntü (n) | Kullanım |
|-----------|--------|----------|:-----------:|---------|
| **Eye Diseases Classification Dataset** (Uveitis, Conjunctivitis, Cataract, Eyelid) | [Mendeley Data](https://data.mendeley.com/datasets/s2bydygjrd/1) | Üveit (193), Konjonktivit (318), Katarakt (357), Göz Kapağı (441) | **1,309** | ✅ Eğitim + Test |

**Veri İşleme Notu:**
- 4 sınıflı kaynak veri, ikili probleme dönüştürülmüştür: Üveit (193) vs Non-Üveit (1,116).
- Stratified sampling ile Train/Val/Test bölünmüştür (%70/%15/%15).
- Sınıf dengesizliği (1:5.8) `BCEWithLogitsLoss pos_weight` (≈5.79) ile yönetilmiştir.

**Önerilen Atıf:**
> "Image Dataset on Eye Diseases Classification with Symptoms and SMOTE Validation." Mendeley Data, 2022.

---

## 2. B-scan OCT (Retina OCT) Modeli

**Final Model:** ResNet-18 (Kermany PT) + 5-Fold K-Fold | **F1: 0.900** | **AUC: 1.000** | **Recall: %100**

| Veri Seti | Kaynak | Sınıflar | Görüntü (n) | Kullanım |
|-----------|--------|----------|:-----------:|---------|
| **OCT Images of PVR, Uveitis and Normal Eyes** (Djatikusumo & Hanna) | [Mendeley Data](https://data.mendeley.com/datasets/hbr5pwk5w5) (`hbr5pwk5w5`) | Üveit-Ara (12), Üveit-Arka (16-ish), Panüveit, Normal (28), RRD-PVR (19) | **55** (gerçek klinik) | ✅ Eğitim + Test (hedef görev) |
| **Kermany Retinal OCT Dataset** (2018) | [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/3) (`rscbjbr9sj`) / Kaggle | CNV, DME, Drusen, Normal | **~108,312** | ✅ Backbone ön eğitimi (domain pre-training) |

**Veri İşleme Notu:**
- 27 gerçek üveit görüntüsünden **1,080 sentetik görüntü** üretilmiştir (40× çoğaltma, Albumentations).
- Sentetik görüntüler yalnızca eğitim setinde kullanılmış, test setinde **kesinlikle yer almamıştır**.
- Kermany Normal sınıfından 9,632 görüntü seçilmiştir. Toplam eğitim havuzu: 10,739 görüntü.
- 5-Fold CV ile aggregated test: 9,659 görüntü (27 gerçek üveit + 9,632 normal).

**Önerilen Atıf:**
> Kermany, D. S., et al. "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." *Cell*, 172(5), 1122–1131, 2018. https://doi.org/10.1016/j.cell.2018.02.010

---

## 3. OCTA (OCT Anjiyografi) Modeli

**Final Model:** ResNet-18 (ImageNet) + TTA (V5) | **F1: 0.780** | **AUC: 0.910**

| Veri Seti | Kaynak | Cihaz | Sınıflar | Görüntü (n) | Kullanım |
|-----------|--------|-------|----------|:-----------:|---------|
| **OCTOPUS — Behçet Hastalığı OCTA Veri Seti** | [Edataverse MPG](https://edataverse.mpg.de) (`doi:10.17617/3.06mnjd`) | Heidelberg Spectralis | Aktif Üveit, İnaktif Üveit, Kontrol | **338** (Superficial + Deep) | ✅ Üveit sınıfı |
| **UT-OCTA Kontrol Veri Seti** | Harici klinik | OptoVue RTVue | Sağlıklı Kontrol | **187** | ✅ Kontrol sınıfı (domain diversification) |

**Veri İşleme Notu:**
- Her hasta için Superficial ve Deep vasküler katman görüntülerinin ikisi de kullanılmıştır.
- `GroupKFold` ile hasta bazlı bölünme: aynı hastanın görüntüleri kesinlikle aynı split içinde tutulmuş, data leakage önlenmiştir.
- İkili etiketleme: Aktif + İnaktif Üveit → "Üveit" (180), Sağlıklı → "Kontrol" (345). Toplam: **525**.
- V3'te UT-OCTA kontrollerinin eklenmesi, AUC'yi 0.696'dan 0.901'e taşımıştır (domain diversification etkisi).

**Önerilen Atıf:**
> "OCTOPUS: A Behçet's Disease OCTA Dataset." Max-Planck-Gesellschaft Edataverse, doi:10.17617/3.06mnjd.

---

## 4. CFP (Renkli Fundus Fotoğrafı) Modeli

**Final Model:** EfficientNet-B0 (ImageNet) + TTA (5×) + Optimal Threshold (t=0.68) | **F1: 0.947** | **AUC: 0.998** | **Recall: %100**

| Veri Seti | Kaynak | Sınıflar | Görüntü (n) | Kullanım |
|-----------|--------|----------|:-----------:|---------|
| **1000 Fundus Images (OIA-ODIR)** | Peking Üniversitesi / Kaggle | Normal, DR(1-3), BRVO, CRVO, Makulopati, ERM, RP, Myopia, CSCR, Lazer izleri, VKH, Bietti vb. (~40 sınıf) | **563** (seçilen alt küme) | ✅ Non-üveit sınıfı + 22 üveit (VKH, Bietti) |
| **RFMiD 2.0** (Retinal Fundus Multi-disease Image Dataset) | IEEE Dataport | Normal (WNL), CRS, çeşitli retinal hastalıklar | **307** (seçilen alt küme) | ✅ Non-üveit sınıfı + 41 üveit (CRS) |

**Veri İşleme Notu:**
- İki kaynaktan birleştirilen toplam **870 görüntü**: Üveit (63) vs Non-Üveit (807). Dengesizlik: **1:12.8**.
- Çift dengeleme stratejisi: `WeightedRandomSampler` + `BCEWithLogitsLoss pos_weight` (≈12.5).
- `ReduceLROnPlateau` ile 21 epoch'ta erken durdurma (hedef 30 epoch).
- TTA (5 augmentation) baseline F1'i 0.783'ten **0.947'ye** taşımıştır. Optimal threshold kalibrasyonu (t=0.68) validation üzerinden belirlenmiştir.

**Önerilen Atıflar:**
> Li, T., et al. "Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening." *Information Sciences*, 501, 511–522, 2019. (OIA-ODIR için)

> Panchal, B., et al. "RFMiD 2.0: A dataset of frequently and rarely encountered retinal diseases." *Data*, 8(1), 10, 2023. https://doi.org/10.3390/data8010010

---

## 5. AS-OCT (Ön Segment OCT) Modeli

**Final Model:** EfficientNet-B0 (Noisy Student / timm) + U-Net Segmentasyon | **F1: 0.920** | **AUC: 0.950**

| Veri Seti | Kaynak | Sınıflar | Görüntü (n) | Kullanım |
|-----------|--------|----------|:-----------:|---------|
| **MCOA — Macaque Corneal Opacity and Anterior Segment Dataset** | Açık erişim klinik veritabanı | Normal (fizyolojik ön segment), Anormal (Korneal Opasite / MCOA) | **6,272** | ✅ Sınıflandırıcı eğitimi + test |
| **AIDK — Automated Infectious Keratitis Detection Dataset** | Açık erişim | Enfeksiyöz Keratit AS-OCT görüntüleri + piksel düzeyli segmentasyon maskesi | **1,168** | ✅ U-Net segmentasyon modeli eğitimi |

**Veri İşleme Notu:**
- MCOA verisi dengeli bir ikili probleme karşılık gelmektedir (~%50 / %50); `pos_weight` gerekmemiştir.
- `timm` kütüphanesinin Noisy Student (JFT-300M) ön eğitimli EfficientNet-B0'ı, standart ImageNet ön eğitimine kıyasla OCT gürültüsüne karşı daha iyi dayanıklılık sağlamaktadır.
- U-Net encoder: VGG16 / ResNet-34 (ImageNet pretrained). Kayıp: Dice Loss + BCE.
- Segmentasyon görsel doğrulama ile değerlendirilmiştir; klinik etiketli ground-truth maskesi yoktur.
- iCloud kilitlenmesi çözümü: `signal.alarm` ile 2 saniyelik OS düzeyinde I/O timeout.

---

## 6. Modalite Yönlendirici (Router) Modeli

**Final Model:** MobileNetV3-Small (ImageNet) | **Accuracy: %100** (500 dengeli örnek, 5 modalite)

| Veri Seti | Kaynak | Sınıflar | Görüntü (n) | Kullanım |
|-----------|--------|----------|:-----------:|---------|
| **Dahili Demo Veri Seti** (5 cihaz × 100 görüntü) | Projenin kendi demo_images klasörü | slitlamp, bscan_oct, octa, cfp, as_oct | **500** (dengeli) | ✅ Router eğitimi |

**Veri İşleme Notu:**
- Her modaliteden 100 orijinal klinik görüntü seçilerek dengeli 5 sınıflı bir eğitim seti oluşturulmuştur.
- Train/Val/Test: %70/%15/%15. Stratified bölünme.
- 10 epoch eğitim, %100 test accuracy ile sonuçlanmıştır.
- Router modeli, web arayüzünde "Otomatik Tespit" modu etkinleştirildiğinde kullanılır.

---

## Veri Seti Durumu Açıklaması

| Simge | Anlam |
|:-----:|-------|
| ✅ | Model eğitimi veya değerlendirmesinde aktif olarak kullanılmıştır. |
| 📋 | İndirilmiş ve hazır; gelecek iterasyonlarda (Faz-2 veya veri genişletme) kullanılabilir. |

---

## Önemli Notlar

- **Kermany ve CellData:** Aynı veri setinin (Kermany 2018) farklı kaynaklardan indirilmiş versiyonlarıdır.
- **OCTA Verisi:** Üveit görüntüleri yalnızca Heidelberg Spectralis cihazından gelmektedir; farklı cihazlardaki üveit vakalarında performans doğrulanmamıştır.
- **Veri Gizliliği:** Projedeki tüm veri setleri açık erişimli (open-access) akademik veri tabanlarından temin edilmiştir; hasta mahremiyeti koruması kapsamındadır.
- **Sentetik Veri (B-scan OCT):** Sentetik görüntüler yalnızca eğitim setinde kullanılmıştır. Test/validasyon setleri **yalnızca gerçek klinik görüntüler** içermektedir.
