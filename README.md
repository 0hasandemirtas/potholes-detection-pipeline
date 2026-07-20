# Pothole Detection Pipeline

Video görüntülerindeki yol çukurlarını YOLO segmentasyon modeliyle tespit eder, Ultralytics tracking backend’i ile kareler arasında takip eder ve işaretlenmiş video ile CSV özeti üretir.

> `track_id`, tracker tarafından verilen kimliktir. Henüz kalıcı veya global bir `pothole_id` değildir. ID switch yaşanırsa aynı fiziksel çukur birden fazla satır oluşturabilir.

## Özellikler

- YOLO tabanlı segmentasyon ve tracking
- Configurable ByteTrack/Ultralytics tracker
- ROI içinde ardışık gözlem ile track confirmation
- Kayıp track’ler için configurable persistence
- EMA veya kapalı bbox smoothing
- CPU, otomatik cihaz veya belirli CUDA GPU seçimi
- Bounding-box ve segmentasyon maskesi alan metrikleri
- Track başına gözlem sayısı ve ortalama confidence
- Başarısız koşularda mevcut nihai çıktıları koruyan `.partial` dosya akışı
- Benchmark CSV ve dosya logları

## Gereksinimler

- Python 3.10 veya üzeri
- YOLO model ağırlığı
- Girdi videosu

Model ve videolar Git repository’sine dahil edilmez. Varsayılan yollar:

```text
models/best.pt
input/video.mp4
```

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Çalıştırma

Varsayılan config ile:

```bash
python main.py
```

Farklı bir config ile:

```bash
python main.py --config config/experiment.yaml
```

Programın çıkış kodları:

```text
0 → başarılı
1 → çalışma/config/model/video hatası
2 → geçersiz CLI argümanı
```

## Config

Ana ayarlar `config/config.yaml` dosyasındadır.

```yaml
model:
  path: models/best.pt

video:
  input: input/video.mp4
  output: output/videos/output.mp4

tracking:
  backend: ultralytics
  tracker: bytetrack.yaml
  conf: 0.15
  imgsz: 640
  n_confirm: 3
  m_persist: 5
  device: null

smoothing:
  type: none
  alpha: 0.5
```

Cihaz seçenekleri:

```yaml
device: null  # Ultralytics otomatik seçer
device: cpu   # CPU kullanır
device: 0     # İlk CUDA GPU'yu kullanır
```

Config doğrulama kuralları:

- `conf`: `0–1` arasında olmalı.
- `imgsz`: pozitif olmalı.
- `n_confirm`: en az `1` olmalı.
- `m_persist`: negatif olmamalı.
- Girdi ve çıktı videosu aynı dosya olamaz.

ByteTrack ayarları `bytetrack.yaml` dosyasındadır.

## İş Akışı

```text
Video frame
    ↓
YOLO segmentation + tracking
    ↓
TrackedDetection
(box, mask, confidence, class_id)
    ↓
ROI confirmation ve track lifecycle
    ↓
Smoothing + metrik agregasyonu
    ↓
Video ve CSV çıktıları
```

ROI dışındaki tespitler `n_confirm` sayacına dahil edilmez. Confirmed bir track ROI dışına çıktığında state ve smoother belleğinden kaldırılır.

## Çıktılar

Nihai yollar config üzerinden belirlenir:

```yaml
video.output: output/videos/output.mp4
output.csv: output/csv/output.csv
output.log: output/logs/pipeline.log
output.benchmark_csv: output/benchmarks/results.csv
output.auto_name: true
output.root_dir: output
```

`output.auto_name: true` olduğunda video, CSV ve log dosya adları çalışma
başında otomatik değiştirilir. Üç çıktı da tarih-saat, girdi videosu, tracker
ve smoothing bilgisinden üretilen aynı deney kimliğini kullanır:

```text
output/videos/20260720_143522_123_video_bytetrack_none.mp4
output/csv/20260720_143522_123_video_bytetrack_none.csv
output/logs/20260720_143522_123_video_bytetrack_none.log
```

Manuel dosya yollarını kullanmak için `output.auto_name: false` yapılabilir.

### CSV şeması

```text
track_id
ilk_kare
son_kare
gozlem_sayisi
max_bbox_alani_px
max_maske_alani_px
ortalama_confidence
```

Alanların anlamları:

- `track_id`: Tracker tarafından verilen kimlik.
- `ilk_kare`: Track’in confirmed olduktan sonra loglandığı ilk kare.
- `son_kare`: Gerçek detection ile loglandığı son kare.
- `gozlem_sayisi`: Confirmed ve ROI içinde işlenen detection sayısı.
- `max_bbox_alani_px`: En büyük smoothed bounding-box alanı.
- `max_maske_alani_px`: En büyük segmentasyon poligon alanı.
- `ortalama_confidence`: Confidence bulunan gözlemlerin ortalaması.

Piksel alanları fiziksel çukur büyüklüğü değildir. Fiziksel ölçüm için kamera kalibrasyonu, perspektif dönüşümü veya derinlik bilgisi gerekir.

## Güvenli Çıktı Akışı

Video ve CSV önce geçici dosyalara yazılır:

```text
output.partial.mp4
output.partial.csv
```

İşlem başarılı olursa nihai isimlere taşınırlar. İşlem sırasında hata oluşursa mevcut nihai çıktılar korunur; inceleme amacıyla `.partial` dosya kalabilir.

## Testler

Tüm testleri çalıştırmak için:

```bash
python -m pytest -q
```

Mevcut suite şu alanları kapsar:

- Track confirmation ve persistence
- ROI confirmation/çıkış davranışı
- EMA ve NoOp smoothing
- Config sınır doğrulamaları
- Backend/factory parametre aktarımı
- Ultralytics sonucunun `TrackedDetection` modeline dönüşümü
- Pipeline cleanup ve başarısız koşuda export davranışı
- Mask alanı ve confidence agregasyonu
- CLI config yolu ve hata kodu
- Pending CSV davranışı

## Bilinen Sınırlamalar

- `track_id`, gerçek dünyada tekil çukur kimliği değildir.
- ID switch aynı çukurun birden fazla sayılmasına neden olabilir.
- ROI kontrolü şu anda bbox bottom-center noktasıyla yapılır.
- ByteTrack ve uygulama tarafında ayrı persistence mekanizmaları vardır.
- Piksel alanları kamera perspektifine bağlıdır.
- OpenCV video çıktısı kaynak videonun sesini korumaz.

## Proje Yapısı

```text
main.py                    CLI ve uygulama giriş noktası
config/config.yaml         Pipeline ayarları
bytetrack.yaml             ByteTrack parametreleri
src/config.py              Config modelleri ve doğrulama
src/models.py              Track, detection ve metrik modelleri
src/protocols.py           Backend/smoother sözleşmeleri
src/factories.py           Config'den implementasyon oluşturma
src/tracking_backends.py   Ultralytics adapter'ı
src/tracking.py            Track yaşam döngüsü
src/smoothing.py           EMA ve NoOp smoothing
src/roi.py                 ROI geometrisi
src/pipeline.py            Ana video işleme akışı
src/benchmark.py           Benchmark CSV kaydı
tests/                     Unit ve component testleri
```
