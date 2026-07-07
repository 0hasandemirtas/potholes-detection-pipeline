# pothole-detection-pipeline

Videodaki yol çukurlarını YOLO segmentasyon modeliyle tespit edip ByteTrack ile takip eder. Her çukura bir ID atar, çıktı olarak işaretlenmiş video ve çukur başına özet CSV üretir.

## Nasıl çalışıyor

- Her karede YOLO (`models/best.pt`) çukurları buluyor, ByteTrack ID'leri kareler arasında eşliyor.
- Bir track üst üste `n_confirm` karede görünmeden çizilmiyor (tek karelik yanlış tespitleri eler).
- Tespit kaybolursa track `m_persist` kare boyunca son bilinen kutusuyla yaşamaya devam ediyor; kutu ekranın altına veya kenarlarına ulaşınca düşürülüyor.
- Kutular EMA ile yumuşatılıyor (`alpha`), titremeyi azaltmak için.

## Çalıştırma

```bash
pip install ultralytics opencv-python pyyaml
python main.py
```

Girdi videosu, model yolu ve tüm eşikler `config/config.yaml` içinde. ByteTrack parametreleri `bytetrack.yaml`'da.

## Çıktılar

- `output/videos/output.mp4` — maske + kutu + ID çizilmiş video
- `output/videos/output.csv` — çukur başına ilk/son kare ve maksimum piksel alanı

## Dosya yapısı

```
main.py            # giriş noktası
src/pipeline.py    # ana döngü: oku -> tespit -> takip -> çiz -> yaz
src/tracking.py    # track onaylama / düşürme mantığı
src/smoothing.py   # kutu EMA yumuşatma
src/config.py      # yaml -> dataclass config
src/models.py      # Track ve TrackLog veri sınıfları
```
