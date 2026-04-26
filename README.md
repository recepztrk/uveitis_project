# Üveit Karar Destek Sistemi

Multimodal oftalmolojik görüntülerden üveit hastalığının otomatik tespiti için derin öğrenme tabanlı karar destek sistemi.

## Proje Hakkında

Bu proje, farklı oftalmolojik görüntüleme modalitelerinden elde edilen verilerle üveit hastalığının tanısına yardımcı olacak bağımsız uzman modeller geliştirmektedir.

### Desteklenen Modaliteler

| Modalite | Backbone | Veri | F1 | AUC | Durum |
|----------|----------|:----:|:--:|:---:|:-----:|
| **Slit-lamp** | EfficientNet-B0 | 1.309 | 0.900 | 0.988 | ✅ |
| **OCTA** | ResNet-18 | 525 | 0.754 | 0.901 | ✅ |
| **CFP** | EfficientNet-B0 | 870 | 0.783 | 1.000 | ✅ |
| **B-scan OCT** | ResNet-18 | 55 | 1.000 | 1.000 | ⚠️ Overfit |

## Proje Yapısı

```
uveitis_project/
├── src/                    # PyTorch Dataset sınıfları
├── preprocessing/          # Veri hazırlama scriptleri
├── training/               # Model eğitim scriptleri
├── evaluation/             # Değerlendirme ve Grad-CAM
├── metadata/               # Etiket ve split CSV dosyaları
├── outputs/
│   ├── metrics/            # Confusion matrix, ROC, eğitim eğrileri
│   └── gradcam/            # Grad-CAM ısı haritaları
├── reports/                # Teknik raporlar
├── data_work/              # Eğitim verileri (gitignore)
└── data_raw/               # Ham veriler (gitignore)
```

## Kurulum

```bash
# Sanal ortam oluştur
python3 -m venv .venv
source .venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

## Kullanım

### Eğitim

```bash
# Slit-lamp modeli
python training/train_slitlamp_baseline.py

# OCTA V3 modeli
python training/train_octa_v3.py

# CFP modeli
python training/train_cfp_baseline.py

# B-scan OCT modeli
python training/train_bscan_oct_baseline.py
```

### Değerlendirme

```bash
python evaluation/evaluate_slitlamp_model.py
python evaluation/evaluate_octa_v3_model.py
python evaluation/evaluate_cfp_model.py
python evaluation/evaluate_bscan_oct_model.py
```

### Grad-CAM Analizi

```bash
python evaluation/gradcam_slitlamp.py
python evaluation/gradcam_octa_v3.py
python evaluation/gradcam_cfp.py
```

## Teknik Detaylar

- **Transfer Learning:** ImageNet pretrained backboneler üzerinde fine-tuning
- **Progressive Fine-Tuning:** İlk 5 epoch head-only, sonra full fine-tuning
- **Sınıf Dengesizliği:** WeightedRandomSampler, pos_weight, Label Smoothing
- **Açıklanabilirlik:** Grad-CAM ile model karar bölgelerinin görselleştirilmesi

## Dokümantasyon

Projenin tüm teknik detayları, model sonuçları, eğitim stratejileri ve geliştirme geçmişi için:

📄 **[Proje Tam Durum Raporu](reports/proje_tam_rapor.md)**

## Ortam

- Python 3.13
- PyTorch 2.11
- macOS (Apple Silicon MPS desteği)

## Lisans

Bu proje akademik bir bitirme projesi kapsamında geliştirilmiştir.
