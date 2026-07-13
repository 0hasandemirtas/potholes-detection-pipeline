import cv2
import numpy as np


def build_roi_polygon(points: list[list[float]], width: int, height: int) -> np.ndarray:
    """Verilen normalized koordinatları kullanarak ROI poligonunu oluşturur ve döndürür."""
    return np.array(
        [[int(x * width), int(y * height)] for x, y in points],
        dtype=np.int32,
    )


def is_box_in_roi(box, roi_polygon: np.ndarray | None) -> bool:
    """Verilen kutunun ROI içinde olup olmadığını kontrol eder. Eğer ROI yoksa True döndürür."""
    if roi_polygon is None:
        return True

    x1, _, x2, y2 = np.asarray(box, dtype=np.float32)
    bottom_center = (float((x1 + x2) / 2), float(y2))

    return cv2.pointPolygonTest(roi_polygon, bottom_center, False) >= 0
