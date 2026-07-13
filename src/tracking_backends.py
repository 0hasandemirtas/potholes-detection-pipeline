from pathlib import Path
from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np


class UltralyticsTrackingBackend:
    def __init__(
        self,
        model_path: str,
        conf: float,
        imgsz: int,
        tracker: str,
    ) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

        self.conf = conf
        self.imgsz = imgsz
        self.tracker = tracker
        self.model = YOLO(model_path)

    def track(
        self,
        frame: np.ndarray,
    ) -> dict[int, tuple[np.ndarray, np.ndarray | None]]:
        """Verilen karedeki nesneleri takip eder ve her bir track_id için kutu ve maske döndürür."""
        results = self.model.track(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            tracker=self.tracker,
            persist=True,
        )

        return self._parse_results(results)

    @staticmethod
    def _parse_results(
        results: list[Results],
    ) -> dict[int, tuple[np.ndarray, np.ndarray | None]]:
        if not results:
            return {}

        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return {}

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()

        masks = (
            result.masks.xy
            if result.masks is not None
            else None
        )

        detections = {}

        for index, track_id in enumerate(track_ids):
            mask = (
                masks[index]
                if masks is not None and index < len(masks)
                else None
            )

            detections[track_id] = (
                boxes[index],
                mask,
            )

        return detections