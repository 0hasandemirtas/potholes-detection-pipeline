import cv2

from src.config import Config
from src.roi import build_roi_polygon


cfg = Config.from_yaml("config/config.yaml")

cap = cv2.VideoCapture(cfg.video.input)
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Video frame okunamadi.")

height, width = frame.shape[:2]
roi_polygon = build_roi_polygon(cfg.roi.points, width, height)

cv2.polylines(
    frame,
    [roi_polygon],
    isClosed=True,
    color=(0, 255, 255),
    thickness=2,
)

cv2.imwrite("output/roi_preview9.jpg", frame)
print("ROI preview kaydedildi: output/roi_preview.jpg")
