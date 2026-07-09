import cv2
import numpy as np


def build_roi_polygon(points: list[list[float]], width: int, height: int) -> np.ndarray:
    return np.array(
        [[int(x * width), int(y * height)] for x, y in points],
        dtype=np.int32,
    )


def is_box_in_roi(box, roi_polygon: np.ndarray | None) -> bool:
    if roi_polygon is None:
        return True

    x1, _, x2, y2 = np.asarray(box, dtype=np.float32)
    bottom_center = (float((x1 + x2) / 2), float(y2))

    return cv2.pointPolygonTest(roi_polygon, bottom_center, False) >= 0
