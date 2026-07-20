import csv
from pathlib import Path

import cv2
import time
import logging
import numpy as np

from src.config import Config
from src.models import (
    TrackLog,
    PipelineMetrics,
)
from src.tracking import TrackState
from src.protocols import (
    BoxSmootherProtocol,
    TrackingBackendProtocol,
)
from src.roi import build_roi_polygon, is_box_in_roi

logger = logging.getLogger(__name__)


class PotholePipeline:
    def __init__(
        self,
        cfg: Config,
        smoother: BoxSmootherProtocol,
        tracking_backend: TrackingBackendProtocol,
    ):
        self.video_path = cfg.video.input
        self.output_path = cfg.video.output
        final_video_path = Path(self.output_path)
        self.pending_video_path = final_video_path.with_name(
            f"{final_video_path.stem}.partial{final_video_path.suffix}"
        )
        self.csv_path = cfg.output.csv
        final_csv_path = Path(self.csv_path)
        self.pending_csv_path = final_csv_path.with_name(
            f"{final_csv_path.stem}.partial{final_csv_path.suffix}"
        )

        self.smoother = smoother
        self.tracking_backend = tracking_backend
        self.n_confirm = cfg.tracking.n_confirm
        self.m_persist = cfg.tracking.m_persist
        self.draw_mask = cfg.visualization.draw_mask
        self.draw_roi = cfg.visualization.draw_roi
        self.max_stale_frames = cfg.visualization.max_stale_frames

        self.prev_frame_time = None
        self.current_fps = 0.0

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
            confirmed_window=cfg.tracking.confirmed_window,
        )

        self.track_log: dict[int, TrackLog] = {}
        self.frame_count = 0

    def initialize_video(self) -> None:
        """Video varsa videoyu açıyor, yoksa hata veriyor. Çıktı videosu için gerekli ayarlamaları yapıyor."""
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
            str(self.pending_video_path), fourcc, self.fps, (self.width, self.height),
        )
        if not self.out.isOpened():
            raise RuntimeError("Cikti videosu yazmak icin acilamadi.")

    def update_tracks(self, detections) -> None:
        """TrackState ve BoxSmoother'ı günceller. Görünür olmayan trackleri temizler."""
        for track_id in self.state.update(detections.keys()):
            self.smoother.drop(track_id)

    def process_tracks(self, frame, detections) -> None:
        """Trackleri işler, ROI kontrolü yapar ve çizim için drawable_tracks listesi oluşturur."""
        drawable_tracks = []

        for track_id in list(self.state.tracks.keys()):
            if not self.state.is_visible(track_id):
                continue

            track = self.state.tracks[track_id]
            if (track.missed_count > 0 and track.missed_count > self.max_stale_frames):
                continue

            drawable = self.handle_track(track_id, detections)
            if drawable is not None:
                drawable_tracks.append(drawable)
        self.draw_tracks(frame, drawable_tracks)

    def handle_track(
        self, track_id, detections
    ) -> tuple[int, np.ndarray, np.ndarray | None] | None:
        """Track için ROI kontrolü yapar ve drawable_tracks listesine ekler."""
        detection = detections.get(track_id)

        if detection is None:
            box = self.smoother.get_last_box(track_id)

            if box is None:
                return None

            if not is_box_in_roi(box, self.roi_polygon):
                self.state.remove_track(track_id)
                self.smoother.drop(track_id)
                return None

            return track_id, box, None

        raw_box = detection.box
        mask = detection.mask

        if not is_box_in_roi(raw_box, self.roi_polygon):
            self.state.remove_track(track_id)
            self.smoother.drop(track_id)
            return None

        box = self.smoother.smooth(raw_box, track_id)

        x1, y1, x2, y2 = box.astype(int)
        box_area = (x2 - x1) * (y2 - y1)
        mask_area = self.calculate_mask_area(mask)

        self.update_log(
            track_id=track_id,
            box_area=box_area,
            mask_area=mask_area,
            confidence=detection.confidence,)

        return track_id, box, mask

    def draw_tracks(self, frame, drawable_tracks) -> None:
        """Drawable_tracks listesindeki trackleri çizer. Eğer draw_mask True ise maskeleri de çizer."""
        masks = []
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

    @staticmethod
    def calculate_mask_area(
        mask: np.ndarray | None,
    ) -> float:
        if mask is None or len(mask) < 3:
            return 0.0

        contour = np.asarray(
            mask,
            dtype=np.float32,
        )
        return float(cv2.contourArea(contour))

    def update_log(
        self,
        track_id: int,
        box_area: int,
        mask_area: float,
        confidence: float | None,
    ) -> None:
        """Track gözlem metriklerini günceller."""
        if track_id not in self.track_log:
            has_confidence = confidence is not None

            self.track_log[track_id] = TrackLog(
                first_frame=self.frame_count,
                last_frame=self.frame_count,
                max_box_area_px=box_area,
                max_mask_area_px=mask_area,
                observation_count=1,
                confidence_sum=(
                    confidence if has_confidence else 0.0
                ),
                confidence_count=(
                    1 if has_confidence else 0
                ),
            )
            return

        log = self.track_log[track_id]
        log.last_frame = self.frame_count
        log.observation_count += 1
        log.max_box_area_px = max(
            log.max_box_area_px,
            box_area,
        )
        log.max_mask_area_px = max(
            log.max_mask_area_px,
            mask_area,
        )

        if confidence is not None:
            log.confidence_sum += confidence
            log.confidence_count += 1

    def draw_roi_polygon(self, frame) -> None:
        """ROI alanını çizer."""
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
        """Track logunu CSV dosyasına yazar."""
        csv_path = self.pending_csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "track_id",
                    "ilk_kare",
                    "son_kare",
                    "gozlem_sayisi",
                    "max_bbox_alani_px",
                    "max_maske_alani_px",
                    "ortalama_confidence",
                    ]
            )

            for tid in sorted(self.track_log):
                info = self.track_log[tid]
                writer.writerow(
                    [
                        tid,
                        info.first_frame,
                        info.last_frame,
                        info.observation_count,
                        info.max_box_area_px,
                        info.max_mask_area_px,
                        info.average_confidence,
                    ]
                )

        logger.info("Toplam %s cukur -> %s", len(self.track_log), csv_path)

    def cleanup(self) -> None:
        """Video kaynaklarını serbest bırakır."""
        if self.cap is not None:
            self.cap.release()
        if self.out is not None:
            self.out.release()

    def publish_outputs(self) -> None:
        """Geçici video ve CSV çıktılarını yayımlar."""
        self.pending_video_path.replace(
            Path(self.output_path)
        )
        self.pending_csv_path.replace(
            Path(self.csv_path)
        )
    def update_fps(self) -> None:
        """FPS değerini günceller."""
        now = time.perf_counter()

        if self.prev_frame_time is not None:
            elapsed = now - self.prev_frame_time
            if elapsed > 0:
                self.current_fps = 1.0 / elapsed

        self.prev_frame_time = now

    def draw_fps(self, frame) -> None:
        """FPS değerini ekrana çizer."""
        cv2.putText(
            frame,
            f"FPS: {self.current_fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )

    def process_frame(self) -> bool:
        """Frame'i işler, takip ve çizim işlemlerini gerçekleştirir. Eğer video bitti ise False döndürür."""
        success, frame = self.cap.read()
        if not success:
            return False
        self.frame_count += 1
        detections = self.tracking_backend.track(frame)
        confirmable_detections = self.filter_detections_in_roi(detections)

        self.update_tracks(confirmable_detections)
        self.process_tracks(frame, detections)
        self.draw_roi_polygon(frame)
        self.update_fps()
        self.draw_fps(frame)

        if self.out is not None:
            self.out.write(frame)
        return True

    def filter_detections_in_roi(self, detections):
        """Yalnız ROI içindeki tespitleri confirmation için döndürür."""
        return {
            track_id: detection
            for track_id, detection in detections.items()
            if is_box_in_roi(detection.box, self.roi_polygon)
        }

    def run(self) -> PipelineMetrics:

        try:
            self.initialize_video()
            start_time = time.perf_counter()

            while self.process_frame():
                pass

            elapsed = time.perf_counter() - start_time
            average_fps = self.frame_count / elapsed if elapsed > 0 else 0.0
            metrics = PipelineMetrics(
                frame_count=self.frame_count,
                elapsed_seconds=elapsed,
                average_fps=average_fps,
                track_count=len(self.track_log),
            )
        finally:
            self.cleanup()

        self.export_csv()
        self.publish_outputs()


        logger.info(
            "Performans: %s frame | %.2f saniye | %.2f ortalama FPS",
            metrics.frame_count,
            metrics.elapsed_seconds,
            metrics.average_fps,
        )

        return metrics
