import csv
from pathlib import Path

import cv2
import time
import logging
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.config import Config
from src.models import TrackLog
from src.tracking import TrackState
from src.smoothing import BoxSmoother
from src.roi import build_roi_polygon, is_box_in_roi

logger = logging.getLogger(__name__)

class PotholePipeline:
    def __init__(self, cfg: Config):
        self.model_path = cfg.model.path
        self.video_path = cfg.video.input
        self.output_path = cfg.video.output
        self.csv_path = cfg.output.csv

        self.conf = cfg.tracking.conf
        self.imgsz = cfg.tracking.imgsz
        self.alpha = cfg.tracking.alpha
        self.n_confirm = cfg.tracking.n_confirm
        self.m_persist = cfg.tracking.m_persist
        self.draw_mask = cfg.visualization.draw_mask
        self.draw_roi = cfg.visualization.draw_roi

        self.prev_frame_time = None
        self.current_fps = 0.0

        self.model = None
        self.cap = None
        self.out = None

        self.width = None
        self.height = None
        self.fps = None

        self.roi_points = cfg.roi.points
        self.roi_polygon = None

        self.state = TrackState(
            n_confirm=self.n_confirm,
            m_persist=self.m_persist,
        )

        self.smoother = BoxSmoother(self.alpha)

        self.track_log: dict[int, TrackLog] = {}
        self.frame_count = 0

    def load_model(self) -> None:
        if not Path(self.model_path).is_file():
            logger.error("Model dosyasi bulunamadi: %s", self.model_path)
            raise FileNotFoundError(f"Model dosyasi bulunamadi: {self.model_path}")
        self.model = YOLO(self.model_path)

    def initialize_video(self) -> None:
        if not Path(self.video_path).is_file():
            logger.error("Girdi videosu bulunamadi: %s", self.video_path)
            raise FileNotFoundError(f"Video dosyasi bulunamadi: {self.video_path}")
        self.cap = cv2.VideoCapture(self.video_path)
        logger.info("Video aciliyor: %s", self.video_path)    

        if not self.cap.isOpened():
            logger.error("Video acilmadi.")
            raise RuntimeError("Video acilmadi.")
        

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.roi_polygon = build_roi_polygon(self.roi_points, self.width, self.height)

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )
        if not self.out.isOpened():
            raise RuntimeError("Cikti videosu yazmak icin acilamadi.")

    def track_objects(self, frame) -> list[Results]:

        results = self.model.track(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            persist=True,
            tracker="bytetrack.yaml",
        )

        return results

    def parse_results(self, results) -> dict[int, tuple[np.ndarray, np.ndarray | None]]:
        result = results[0]
        detections = {}
        boxes = (
            result.boxes.xyxy.cpu().numpy()
            if result.boxes is not None
            else np.array([])
        )
        masks = result.masks.xy if result.masks is not None else None
        track_ids = (
            result.boxes.id.int().tolist()
            if result.boxes is not None and result.boxes.id is not None
            else []
        )

        for i, track_id in enumerate(track_ids):
            mask = masks[i] if masks is not None else None
            detections[track_id] = (boxes[i], mask)

        return detections

    def update_tracks(self, detections) -> None:
        for track_id in self.state.update(detections.keys()):
            self.smoother.drop(track_id)

    def process_tracks(self, frame, detections) -> None:
        drawable_tracks = []

        for track_id in list(self.state.tracks.keys()):
            if not self.state.is_visible(track_id):
                continue
            
            drawable = self.handle_track(track_id, detections)
            if drawable is not None:
                drawable_tracks.append(drawable)
        self.draw_tracks(frame, drawable_tracks)

    def handle_track(self, track_id, detections) -> tuple[int, np.ndarray, np.ndarray | None] | None:
        detection = detections.get(track_id)

        if detection is None:
            box = self.smoother.prev_box.get(track_id)
            if box is not None and not is_box_in_roi(box, self.roi_polygon):
                self.state.remove_track(track_id)

            self.smoother.drop(track_id)
            return

        raw_box, mask = detection
        if not is_box_in_roi(raw_box, self.roi_polygon):
            self.state.remove_track(track_id)
            self.smoother.drop(track_id)
            return

        box = self.smoother.smooth(raw_box, track_id)

        x1, y1, x2, y2 = box.astype(int)
        self.update_log(track_id, (x2 - x1) * (y2 - y1))

        return track_id, box, mask

    def draw_tracks(self, frame, drawable_tracks) -> None:
        if self.draw_mask:
            masks = [
                mask.astype(np.int32, copy=False)
                for _, _, mask in drawable_tracks
                if mask is not None and len(mask) > 0
            ]

        if masks:
            overlay = frame.copy()
            cv2.fillPoly(overlay, masks, (0, 200, 0))
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        for track_id, box, _ in drawable_tracks:
            x1, y1, x2, y2 = box.astype(int)


            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    def update_log(self, track_id, area) -> None:
        if track_id not in self.track_log:
            self.track_log[track_id] = TrackLog(
                first_frame=self.frame_count, last_frame=self.frame_count, max_area=area
            )
            return

        log = self.track_log[track_id]
        log.last_frame = self.frame_count
        log.max_area = max(log.max_area, area)

    def draw_track(self, frame, track_id, box, mask) -> None:
        x1, y1, x2, y2 = box

        if self.draw_mask and mask is not None and len(mask) > 0:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [mask.astype(np.int32)], (0, 200, 0))
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    def draw_roi_polygon(self, frame) -> None:
        if not self.draw_roi or self.roi_polygon is None:
            return

        cv2.polylines(
            frame,
            [self.roi_polygon],
            isClosed=True,
            color=(0, 255, 255),
            thickness=2,
        )

    def export_csv(self) -> None:
        csv_path = self.csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "ilk_kare", "son_kare", "max_alan_px"])

            for tid in sorted(self.track_log):
                info = self.track_log[tid]
                writer.writerow(
                    [
                        tid,
                        info.first_frame,
                        info.last_frame,
                        info.max_area,
                    ]
                )

        logger.info("Toplam %s cukur -> %s", len(self.track_log), csv_path)

    def cleanup(self) -> None:
        if self.cap is not None:
            self.cap.release()
        if self.out is not None:
            self.out.release()

    def process_frame(self) -> bool:
        success, frame = self.cap.read()
        if not success:
            return False
        self.frame_count += 1
        results = self.track_objects(frame)
        detections = self.parse_results(results)
        self.update_tracks(detections)
        self.process_tracks(frame, detections)
        self.draw_roi_polygon(frame)
        self.update_fps()
        self.draw_fps(frame)

        if self.out is not None:
            self.out.write(frame)
        return True
    
    def update_fps(self) -> None:
        now = time.perf_counter()

        if self.prev_frame_time is not None:
            elapsed = now - self.prev_frame_time
            if elapsed > 0:
                self.current_fps = 1.0 / elapsed

        self.prev_frame_time = now
    
    def draw_fps(self, frame) -> None:
        cv2.putText(
            frame,
            f"FPS: {self.current_fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )

 

    def run(self):
        self.load_model()
        self.initialize_video()
        try:
            while self.process_frame():
                pass
        finally:
            self.cleanup()
            self.export_csv()
