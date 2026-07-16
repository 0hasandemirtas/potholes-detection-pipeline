from pathlib import Path
from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np
from src.models import TrackedDetection


class UltralyticsTrackingBackend:
    def __init__(
        self,
        model_path: str,
        conf: float,
        imgsz: int,
        tracker: str,
        device: str | int | None = None,
    ) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

        self.conf = conf
        self.imgsz = imgsz
        self.tracker = tracker
        self.device = device
        self.model = YOLO(model_path)

    def track(
        self,
        frame: np.ndarray,
    ) -> dict[int, TrackedDetection]:
        """Verilen karedeki nesneleri takip eder ve her bir track_id için kutu ve maske döndürür."""
        track_options = {
            "conf": self.conf,
            "imgsz": self.imgsz,
            "tracker": self.tracker,
            "persist": True,
        }

        if self.device is not None:
            track_options["device"] = self.device

        results = self.model.track(
            frame,
            **track_options,
        )

        return self._parse_results(results)

    @staticmethod
    def _parse_results(
        results: list[Results],
    ) -> dict[int, TrackedDetection]:
        if not results:
            return {}

        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return {}

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()

        confidences = (
            result.boxes.conf.cpu().tolist()
            if result.boxes.conf is not None
            else []
        )
        class_ids = (
            result.boxes.cls.int().cpu().tolist()
            if result.boxes.cls is not None
            else []
        )

        masks = result.masks.xy if result.masks is not None else None

        detections = {}

        for index, track_id in enumerate(track_ids):
            mask = masks[index] if masks is not None and index < len(masks) else None

            confidence = (
                float(confidences[index])
                if index < len(confidences)
                else None
            )
            class_id = (
                int(class_ids[index])
                if index < len(class_ids)
                else None
            )

            detections[track_id] = TrackedDetection(
                box=boxes[index],
                mask=mask,
                confidence=confidence,
                class_id=class_id
            )

        return detections
