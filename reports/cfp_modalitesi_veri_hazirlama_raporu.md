# CFP Modalitesi Veri Hazırlama Raporu  
## Multimodal Görüntü Analizi ile Üveit Tanısı için Yapay Zekâ Tabanlı Karar Destek Sistemi

Bu rapor, bitirme projesi kapsamında geliştirilecek **CFP (Color Fundus Photography / renkli fundus fotoğrafı) unimodal modeline** ait veri hazırlama sürecini açıklamak için hazırlanmıştır. Amaç, Claude ile model geliştirme aşamasına geçmeden önce CFP verilerinin nereden geldiğini, nasıl seçildiğini, nasıl sınıflandırıldığını, nasıl isimlendirildiğini ve metadata/split dosyalarının nasıl oluşturulduğunu net şekilde belgelemektir.

---

## 1. Projenin genel bağlamı

Ana proje, **multimodal görüntü analizi ile üveit tanısı için yapay zekâ tabanlı açıklanabilir karar destek sistemi** geliştirmeyi hedeflemektedir.

Projedeki temel yaklaşım, tek bir görüntü modalitesiyle doğrudan klinik üveit tanısı koymak değildir. Bunun yerine farklı görüntü modaliteleri için ayrı **unimodal expert modeller** geliştirilecek, bu modellerin ürettiği skorlar ileride karar destek sistemi içinde birlikte değerlendirilecektir.

Mevcut modaliteler:

```text
data_work/
├── bscan_oct_clean/
│   ├── non_uveitis/
│   └── uveitis/
├── octa_clean/
│   ├── non_uveitis/
│   └── uveitis/
├── slitlamp_clean/
│   ├── non_uveitis/
│   └── uveitis/
└── cfp_clean/
    ├── non_uveitis/
    └── uveitis/
```

Bu rapor yalnızca **CFP modalitesi** için yapılan veri hazırlama işlemlerini açıklamaktadır.

---

## 2. CFP unimodal modelinin proje içindeki rolü

CFP modeli, ana multimodal karar destek sisteminde **posterior segment/fundus görüntüsü üzerinden üveit ilişkili inflamatuvar bulgu skoru üreten yardımcı model** olarak konumlandırılmıştır.

Bu modelin görevi:

```text
Girdi:
- Renkli fundus fotoğrafı / CFP

Çıktı:
- uveitis veya non_uveitis sınıf tahmini
- pratikte: posterior segmentte üveit ilişkili inflamatuvar bulgu olasılığı
```

Önemli sınırlama:

> CFP modalitesindeki `uveitis` sınıfı, genel klinik üveit tanısının tamamını temsil etmemektedir. Bu sınıf, VKH, chorioretinitis ve retinitis etiketleri üzerinden temsil edilen **üveit ilişkili posterior segment inflamatuvar bulguları** ifade etmektedir.

Yani bu model **genel üveit tanı modeli** olarak değil, **CFP tabanlı posterior inflamasyon/üveit ilişkili bulgu tespit modeli** olarak değerlendirilmelidir.

---

## 3. Kullanılan orijinal veri setleri

CFP modalitesi için iki ayrı renkli fundus veri seti kullanılmıştır.

### 3.1. `1000images` veri seti

Bu veri seti klasör tabanlıdır. Her hastalık/etiket sınıfı ayrı klasör altında bulunmaktadır.

Örnek yapı:

```text
1000images/
├── 0.0.Normal/
├── 0.3.DR1/
├── 1.0.DR2/
├── 1.1.DR3/
├── 2.0.BRVO/
├── 2.1.CRVO/
├── 3.RAO/
├── 4.Rhegmatogenous RD/
├── 5.0.CSCR/
├── 5.1.VKH disease/
├── 6.Maculopathy/
├── 7.ERM/
├── 8.MH/
├── 9.Pathological myopia/
├── 11.Severe hypertensive retinopathy/
├── 15.0.Retinitis pigmentosa/
├── 15.1.Bietti crystalline dystrophy/
├── 16.Peripheral retinal degeneration and break/
└── 27.Laser Spots/
```

Bu veri setinden hem pozitif hem negatif sınıf için görüntüler seçilmiştir.

---

### 3.2. `RFMiD2_0` veri seti

Bu veri seti split tabanlıdır ve her split klasörü içinde görüntüler ile ilgili CSV etiket dosyaları yer almaktadır.

Yapı:

```text
RFMiD2_0/
├── Training_set/
├── Validation_set/
└── Test_set/
```

Her split klasörü içinde ilgili görüntüler ve etiket CSV dosyaları bulunmaktadır. RFMiD2_0 tarafında görüntülerin sınıf bilgisi klasör adından değil, CSV dosyalarındaki etiket kolonlarından okunmuştur.

Bu çalışma için RFMiD2_0 veri setinden yalnızca şu etiketler kullanılmıştır:

```text
WNL → non_uveitis
CRS → uveitis
RS  → uveitis
```

Burada:

- `WNL`: Within Normal Limits / normal fundus
- `CRS`: Chorioretinitis
- `RS`: Retinitis

CRS ve RS doğrudan genel klinik üveit etiketi değildir; ancak posterior segment inflamasyonu ile ilişkili oldukları için CFP modalitesindeki pozitif sınıf için **proxy üveit ilişkili posterior inflamasyon etiketi** olarak kullanılmıştır.

---

## 4. Sınıf tanımı

CFP modeli için binary sınıflandırma yapısı korunmuştur:

```text
cfp_clean/
├── non_uveitis/
└── uveitis/
```

### 4.1. `uveitis` sınıfı

Bu sınıf, CFP görüntüsünde üveit ilişkili posterior segment inflamatuvar bulgu bulunan görüntüleri temsil eder.

Bu sınıfa dahil edilen veriler:

| Kaynak veri seti | Orijinal etiket / klasör | Standart kaynak sınıf adı | Hedef sınıf |
|---|---|---|---|
| 1000images | `5.1.VKH disease` | `vkh` | `uveitis` |
| RFMiD2_0 | `CRS = 1` | `crs` | `uveitis` |
| RFMiD2_0 | `RS = 1` | `rs` | `uveitis` |

Final pozitif sınıf dağılımı:

| source_class | Sayı |
|---|---:|
| `vkh` | 14 |
| `crs` | 41 |
| `rs` | 8 |
| **Toplam** | **63** |

---

### 4.2. `non_uveitis` sınıfı

Bu sınıf iki tip veri içermektedir:

1. Normal fundus görüntüleri
2. Üveit dışı retinal hastalık görüntüleri

Bunun nedeni, modelin sadece “normal mi hastalıklı mı?” ayrımı öğrenmesini engellemek ve üveit ilişkili posterior inflamatuvar bulguları diğer retinal patolojilerden ayırmasını sağlamaktır.

Bu sınıfa dahil edilen veriler:

#### 1000images kaynaklı negatif sınıflar

| Orijinal klasör | Standart source_class | Hedef sınıf |
|---|---|---|
| `0.0.Normal` | `normal` | `non_uveitis` |
| `0.3.DR1` | `dr1` | `non_uveitis` |
| `1.0.DR2` | `dr2` | `non_uveitis` |
| `1.1.DR3` | `dr3` | `non_uveitis` |
| `2.0.BRVO` | `brvo` | `non_uveitis` |
| `2.1.CRVO` | `crvo` | `non_uveitis` |
| `3.RAO` | `rao` | `non_uveitis` |
| `4.Rhegmatogenous RD` | `rd` | `non_uveitis` |
| `5.0.CSCR` | `cscr` | `non_uveitis` |
| `6.Maculopathy` | `maculopathy` | `non_uveitis` |
| `7.ERM` | `erm` | `non_uveitis` |
| `8.MH` | `mh` | `non_uveitis` |
| `9.Pathological myopia` | `pathological_myopia` | `non_uveitis` |
| `11.Severe hypertensive retinopathy` | `hypertensive_retinopathy` | `non_uveitis` |
| `15.0.Retinitis pigmentosa` | `retinitis_pigmentosa` | `non_uveitis` |
| `15.1.Bietti crystalline dystrophy` | `bietti` | `non_uveitis` |
| `16.Peripheral retinal degeneration and break` | `peripheral_degeneration_break` | `non_uveitis` |
| `27.Laser Spots` | `laser_spots` | `non_uveitis` |

#### RFMiD2_0 kaynaklı negatif sınıf

| CSV etiketi | Standart source_class | Hedef sınıf |
|---|---|---|
| `WNL = 1` | `wnl` | `non_uveitis` |

Final negatif sınıf dağılımı:

| source_class | Sayı |
|---|---:|
| `wnl` | 258 |
| `maculopathy` | 74 |
| `rd` | 57 |
| `pathological_myopia` | 54 |
| `dr2` | 48 |
| `brvo` | 44 |
| `dr3` | 39 |
| `normal` | 37 |
| `erm` | 26 |
| `mh` | 23 |
| `retinitis_pigmentosa` | 22 |
| `crvo` | 22 |
| `laser_spots` | 20 |
| `rao` | 16 |
| `dr1` | 16 |
| `hypertensive_retinopathy` | 15 |
| `peripheral_degeneration_break` | 14 |
| `cscr` | 14 |
| `bietti` | 8 |
| **Toplam** | **807** |

---

## 5. Dışarıda bırakılan/ambiguous sınıflar

Bazı sınıflar ilk CFP modeline dahil edilmemiştir. Bunun nedeni, bu bulguların üveitte de görülebilecek inflamatuvar ya da non-spesifik retinal bulgularla karışabilmesidir.

İlk model için negatif sınıfa dahil edilmemesi tercih edilen örnekler:

```text
Disc swelling and elevation
Vitreous particles
Yellow-white spots-flecks
Cotton-wool spots
Vessel tortuosity
CME
ODE
ON
VS
ME
TV
IIH
```

Bu sınıflar doğrudan çöpe atılmamıştır; ancak ilk binary CFP modelinde veri kirliliği oluşturmaması için eğitim setine dahil edilmemiştir.

---

## 6. Veri hazırlama stratejisi

Başlangıçta kullanılacak görüntüler manuel olarak taşınmıştı. Ancak kaynak sınıf bilgisinin kaybolması ve RFMiD2_0 tarafındaki CSV tabanlı etiket yapısı nedeniyle daha temiz bir yöntem tercih edildi.

Daha sonra ayrı bir hazırlık scripti yazıldı. Bu script ana proje içinde değil, orijinal CFP veri setlerinin bulunduğu klasörde çalıştırıldı.

Scriptin çalıştığı klasör yapısı:

```text
Renkli fundus fotoğrafı (CFP : fundus)/
├── 1000images/
├── RFMiD2_0/
└── prepare_cfp_clean.py
```

Scriptin ürettiği çıktı:

```text
cfp_clean_prepared/
├── non_uveitis/
└── uveitis/
```

Daha sonra `cfp_clean_prepared` klasörü ana projeye taşınarak şu yapıya yerleştirildi:

```text
uveitis_project/
└── data_work/
    └── cfp_clean/
        ├── non_uveitis/
        └── uveitis/
```

---

## 7. Görüntü isimlendirme standardı

CFP görüntüleri proje içinde okunabilir ve izlenebilir olacak şekilde yeniden isimlendirilmiştir.

Kullanılan format:

```text
cfp_<source_dataset>_<target_short>_<source_class>_<sequence>.jpg
```

Alanların anlamı:

| Alan | Açıklama |
|---|---|
| `cfp` | Görüntü modalitesi |
| `source_dataset` | Kaynak veri seti: `cfp1000` veya `rfmimd2` |
| `target_short` | Hedef sınıf: `uv` veya `nonuv` |
| `source_class` | Orijinal sınıf/etiket bilgisi |
| `sequence` | Aynı kaynak sınıf içindeki sıra numarası |

Örnek pozitif dosyalar:

```text
cfp_cfp1000_uv_vkh_0001.jpg
cfp_cfp1000_uv_vkh_0002.jpg
cfp_rfmimd2_uv_crs_0001.jpg
cfp_rfmimd2_uv_crs_0002.jpg
cfp_rfmimd2_uv_rs_0001.jpg
```

Örnek negatif dosyalar:

```text
cfp_cfp1000_nonuv_normal_0001.jpg
cfp_cfp1000_nonuv_dr2_0001.jpg
cfp_cfp1000_nonuv_brvo_0001.jpg
cfp_cfp1000_nonuv_maculopathy_0001.jpg
cfp_rfmimd2_nonuv_wnl_0001.jpg
```

Bu isimlendirme sayesinde:

- Görüntünün CFP modalitesine ait olduğu anlaşılır.
- Hangi veri setinden geldiği anlaşılır.
- Hedef sınıfı anlaşılır.
- Kaynak/orijinal sınıfı anlaşılır.
- Metadata üretimi kolaylaşır.
- Hata analizi sırasında dosya kaynağı hızlıca izlenebilir.

---

## 8. Duplicate temizliği

Hazırlama sürecinde exact duplicate kontrolü yapılmıştır. MD5 hash tabanlı kontrol ile aynı içeriğe sahip görüntülerin tekrar kopyalanması engellenmiştir.

Sonuçta bazı duplicate görüntüler atlanmış ve final veri sayısı 870 olmuştur.

Önemli not:

Terminalde `find data_work/cfp_clean -type f | wc -l` komutu bir ara 871 göstermiştir. Ancak `cfp_labels.csv` 870 satır içermektedir. Bu farkın muhtemel nedeni macOS tarafından oluşturulan `.DS_Store` gibi görüntü olmayan ekstra sistem dosyasıdır. Görüntü uzantısı olmayan dosyalar metadata scriptine dahil edilmemiştir.

Temizlik için önerilen komut:

```bash
find data_work/cfp_clean -name ".DS_Store" -delete
```

---

## 9. Final CFP klasör yapısı

Ana proje içinde final CFP klasörü:

```text
uveitis_project/
└── data_work/
    └── cfp_clean/
        ├── non_uveitis/
        └── uveitis/
```

Final görüntü sayıları:

| Klasör | Sayı |
|---|---:|
| `data_work/cfp_clean/uveitis` | 63 |
| `data_work/cfp_clean/non_uveitis` | 807 |
| **Toplam** | **870** |

---

## 10. Metadata oluşturma

CFP verileri ana projeye taşındıktan sonra diğer modalitelerdeki script yapısına benzer şekilde iki script hazırlandı:

```text
preprocessing/
├── cfp_labels_build.py
└── cfp_split_build.py
```

Bu scriptler şu çıktıları üretmiştir:

```text
metadata/
├── cfp_labels.csv
└── cfp_split.csv
```

---

## 11. `cfp_labels.csv` yapısı

`cfp_labels_build.py` scripti `data_work/cfp_clean/uveitis` ve `data_work/cfp_clean/non_uveitis` klasörlerini taramaktadır. Dosya adından kaynak veri seti ve kaynak sınıf bilgisi çıkarılmaktadır.

CSV kolonları:

```text
image_id
filepath
source_dataset
source_class
binary_label
```

Örnek satır mantığı:

```text
image_id: cfp_cfp1000_uv_vkh_0001
filepath: data_work/cfp_clean/uveitis/cfp_cfp1000_uv_vkh_0001.jpg
source_dataset: cfp1000
source_class: vkh
binary_label: 1
```

`binary_label` anlamı:

| binary_label | Anlam |
|---|---|
| 0 | non_uveitis |
| 1 | uveitis |

Final `cfp_labels.csv` dağılımı:

| Alan | Değer |
|---|---:|
| Toplam satır | 870 |
| `binary_label = 0` | 807 |
| `binary_label = 1` | 63 |

Kaynak veri seti dağılımı:

| source_dataset | Sayı |
|---|---:|
| `cfp1000` | 563 |
| `rfmimd2` | 307 |

---

## 12. `cfp_split.csv` yapısı

`cfp_split_build.py` scripti `metadata/cfp_labels.csv` dosyasını okuyarak train/validation/test bölmesi yapmaktadır.

Split oranı:

```text
train: %70
validation: %15
test: %15
```

Öncelikli strateji:

```text
stratify = source_class
```

Bu sayede `vkh`, `crs`, `rs`, `wnl`, `dr2`, `brvo`, vb. alt sınıfların oranı train/val/test içinde korunmaya çalışılmıştır.

Eğer `source_class` üzerinden stratified split hata verseydi, script `binary_label` üzerinden stratified split yapacak şekilde tasarlanmıştır. Ancak final çalışmada `source_class` stratification başarılı şekilde çalışmıştır.

CSV kolonları:

```text
image_id
filepath
source_dataset
source_class
binary_label
split
```

Final split dağılımı:

| Split | Toplam |
|---|---:|
| train | 609 |
| validation | 130 |
| test | 131 |
| **Toplam** | **870** |

Binary label dağılımı:

| Split | non_uveitis | uveitis |
|---|---:|---:|
| train | 564 | 45 |
| validation | 121 | 9 |
| test | 122 | 9 |

Pozitif sınıf tüm splitlere düşmüştür:

| source_class | train | validation | test |
|---|---:|---:|---:|
| `vkh` | 10 | 2 | 2 |
| `crs` | 29 | 6 | 6 |
| `rs` | 6 | 1 | 1 |

Bu dağılım veri boyutu açısından kabul edilebilir düzeydedir. Ancak test setindeki pozitif sayısı düşük olduğu için değerlendirme metrikleri yorumlanırken dikkatli olunmalıdır.

---

## 13. Final source_class dağılımı

`cfp_labels.csv` içinde elde edilen final `source_class` dağılımı:

| source_class | Sayı |
|---|---:|
| `wnl` | 258 |
| `maculopathy` | 74 |
| `rd` | 57 |
| `pathological_myopia` | 54 |
| `dr2` | 48 |
| `brvo` | 44 |
| `crs` | 41 |
| `dr3` | 39 |
| `normal` | 37 |
| `erm` | 26 |
| `mh` | 23 |
| `retinitis_pigmentosa` | 22 |
| `crvo` | 22 |
| `laser_spots` | 20 |
| `rao` | 16 |
| `dr1` | 16 |
| `hypertensive_retinopathy` | 15 |
| `peripheral_degeneration_break` | 14 |
| `cscr` | 14 |
| `vkh` | 14 |
| `bietti` | 8 |
| `rs` | 8 |

---

## 14. CFP modeli için veri setinin güçlü yönleri

Bu veri hazırlama yaklaşımının güçlü tarafları:

1. **Proje yapısıyla uyumlu binary klasörleme yapısı korunmuştur.**

   ```text
   uveitis/
   non_uveitis/
   ```

2. **Pozitif sınıf yalnızca doğrudan veya proxy olarak posterior inflamasyonla ilişkili sınıflardan oluşturulmuştur.**

   ```text
   VKH + CRS + RS
   ```

3. **Negatif sınıf sadece normal görüntülerden oluşmamaktadır.**

   Negatif sınıfa üveit dışı retinal hastalıklar da eklenmiştir. Böylece modelin sadece “normal-hasta” ayrımı öğrenmesi yerine, üveit ilişkili posterior inflamasyon paternlerini diğer retinal patolojilerden ayırması hedeflenmiştir.

4. **Dosya isimlendirme sistemi izlenebilirdir.**

   Her dosyada kaynak veri seti, hedef sınıf ve kaynak sınıf bilgisi bulunmaktadır.

5. **Metadata dosyaları diğer modalitelerle uyumlu şekilde üretilmiştir.**

   ```text
   cfp_labels.csv
   cfp_split.csv
   ```

6. **Split stratified yapılmıştır.**

   Source class dağılımı train/val/test içinde korunmuştur.

---

## 15. CFP veri setinin sınırlılıkları

Bu veri setinin ciddi sınırlılıkları vardır ve model geliştirme/raporlama sırasında açık şekilde belirtilmelidir.

### 15.1. Pozitif sınıf küçüktür

Pozitif sınıf yalnızca 63 görüntüden oluşmaktadır:

```text
vkh = 14
crs = 41
rs = 8
```

Bu nedenle modelin performansı veri bölünmesine ve augmentasyon stratejisine duyarlı olabilir.

### 15.2. Genel üveit spektrumu temsil edilmemektedir

CFP pozitif sınıfı yalnızca posterior segmentte görülebilen bazı inflamatuvar paternleri temsil eder. Anterior üveit, intermediate üveit, Behçet, sarkoidoz, oküler TB, idiopatik üveit gibi geniş klinik spektrum bu CFP veri setinde yeterince temsil edilmemektedir.

### 15.3. CRS/RS proxy etikettir

RFMiD2_0 içindeki CRS ve RS etiketleri doğrudan “genel üveit tanısı” değildir. Bu nedenle model çıktısı klinik üveit tanısı olarak değil, posterior inflamatuvar bulgu skoru olarak yorumlanmalıdır.

### 15.4. Veri seti kaynak farkı olabilir

Pozitif ve negatif görüntüler farklı veri setlerinden geldiği için modelin bazı durumlarda hastalık yerine veri seti/camera/domain farklarını öğrenme riski vardır. Bu nedenle ileride Grad-CAM, external validation ve mümkünse domain-aware evaluation yapılmalıdır.

### 15.5. Hasta bazlı split yoktur

Veri setlerinde hasta ID bilgisi bulunmadığı için hasta bazlı train/test ayrımı yapılamamıştır. Exact duplicate temizliği yapılmıştır, ancak aynı hastaya ait benzer görüntüler teorik olarak farklı splitlere düşmüş olabilir.

---

## 16. Model geliştirme için öneriler

CFP modeli geliştirirken şu strateji önerilir:

### 16.1. Model tipi

Küçük veri nedeniyle sıfırdan CNN eğitmek uygun değildir. Transfer learning kullanılmalıdır.

Önerilen backbone modeller:

```text
EfficientNet-B0
DenseNet121
ResNet50
```

Başlangıç için en uygun seçenekler:

```text
EfficientNet-B0 veya DenseNet121
```

### 16.2. Loss ve sampling

Sınıf dengesizliği belirgindir:

```text
non_uveitis = 807
uveitis = 63
```

Bu nedenle şunlardan biri kullanılmalıdır:

```text
class_weight
WeightedRandomSampler
focal loss
```

Sadece accuracy ile değerlendirme yapılmamalıdır.

### 16.3. Değerlendirme metrikleri

Özellikle pozitif sınıf küçük olduğu için şu metrikler raporlanmalıdır:

```text
AUC
F1-score
Recall / sensitivity
Specificity
Precision
Confusion matrix
ROC curve
PR curve
```

Accuracy tek başına yanıltıcı olabilir.

### 16.4. Explainability

CFP modeli için Grad-CAM önerilir. Modelin karar verirken retinal lezyonlara mı, görüntü kenarlarına mı, cihaz/domain artefaktlarına mı baktığını anlamak için önemlidir.

### 16.5. Augmentasyon

Klinik olarak makul augmentasyonlar kullanılmalıdır:

```text
small rotation ±10°
horizontal flip
mild brightness/contrast
mild Gaussian noise
resize/crop dikkatli şekilde
```

Kaçınılması gerekenler:

```text
aşırı renk bozulması
vertical flip
yüksek dereceli rotation
lezyonu kesen agresif random crop
```

---

## 17. Claude için pratik görev özeti

Claude ile model geliştirmeye geçerken CFP tarafında kullanılacak ana dosyalar:

```text
data_work/cfp_clean/
├── non_uveitis/
└── uveitis/

metadata/cfp_labels.csv
metadata/cfp_split.csv
```

Model eğitiminde kullanılacak ana CSV:

```text
metadata/cfp_split.csv
```

Temel kolonlar:

```text
filepath
binary_label
split
```

Ek analiz için kullanılabilecek kolonlar:

```text
source_dataset
source_class
```

Binary label anlamı:

```text
0 = non_uveitis
1 = uveitis
```

Ama kavramsal açıklama:

```text
1 = CFP görüntüsünde üveit ilişkili posterior segment inflamatuvar bulgu var
0 = normal veya üveit dışı retinal hastalık
```

---

## 18. Sonuç

CFP modalitesi için veri hazırlama süreci tamamlanmıştır. Final veri seti, ana multimodal proje yapısına uygun şekilde binary klasör yapısında düzenlenmiştir:

```text
data_work/cfp_clean/
├── non_uveitis → 807 görüntü
└── uveitis     → 63 görüntü
```

Metadata ve split dosyaları oluşturulmuştur:

```text
metadata/cfp_labels.csv
metadata/cfp_split.csv
```

Bu veri setiyle geliştirilecek CFP modeli, ana sistem içinde **posterior segment üveit ilişkili inflamasyon skoru üreten unimodal expert model** olarak değerlendirilmelidir. Genel klinik üveit tanısı için tek başına yeterli değildir; ancak multimodal karar destek sistemi içinde anlamlı bir yardımcı bileşen olarak kullanılabilir.
