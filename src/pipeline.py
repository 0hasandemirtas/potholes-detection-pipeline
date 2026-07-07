import csv
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


from src.config import Config
from src.models import TrackLog
from src.tracking import TrackState
from src.smoothing import BoxSmoother

class PotholePipeline:
        
    def __init__(self, cfg: Config):
        self.model_path = cfg.model.path
        self.video_path = cfg.video.input
        self.output_path = cfg.video.output

        self.conf = cfg.tracking.conf
        self.imgsz = cfg.tracking.imgsz
        self.alpha = cfg.tracking.alpha
        self.n_confirm = cfg.tracking.n_confirm
        self.m_persist = cfg.tracking.m_persist
        self.draw_mask = cfg.visualization.draw_mask

        self.model = None
        self.cap = None
        self.out = None

        self.width = None
        self.height = None
        self.fps = None

        self.bottom_limit = cfg.limits.bottom
        self.left_limit = cfg.limits.left
        self.right_limit = cfg.limits.right

        self.state = TrackState(
            n_confirm=self.n_confirm,
            m_persist=self.m_persist,
        )

        self.smoother = BoxSmoother(self.alpha)

        self.track_log: dict[int, TrackLog] = {}
        self.prev_mask: dict[int, np.ndarray] = {}
        self.frame_count = 0


    def load_model(self) -> None:
        self.model = YOLO(self.model_path)

    def initialize_video(self) -> None:
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise RuntimeError("Video acilmadi.")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))

    def track_objects(self, frame)  -> list[Results]:

        results = self.model.track(
            frame, 
            conf=self.conf, 
            imgsz=self.imgsz, 
            persist=True, 
            tracker="bytetrack.yaml"
        )
        
        return results
    
    def parse_results(
            self, 
            results
        ) -> dict[int, tuple[np.ndarray, np.ndarray | None]]:
        result = results[0]
        detections = {}
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
        masks = result.masks.xy if result.masks is not None else None
        track_ids = result.boxes.id.int().tolist() if result.boxes is not None and result.boxes.id is not None else []

        for i,track_id in enumerate(track_ids):
            mask = masks[i] if masks is not None else None
            detections[track_id] = (boxes[i], mask)

        return detections
    
    def update_tracks(self, detections) -> None:
        for track_id in self.state.update(detections.keys()):
            self.smoother.drop(track_id)
            self.prev_mask.pop(track_id, None)

    def process_tracks(self, frame, detections) -> None:
        for track_id in list(self.state.tracks.keys()):

            if not self.state.is_visible(track_id):
                continue

            self.handle_track(
                frame, 
                track_id, 
                detections
            )
            

    def handle_track(self, frame, track_id, detections) -> None:
        detection = detections.get(track_id)
        if detection is None:

            if track_id in self.smoother.prev_box:
                box = self.smoother.prev_box.get(track_id)
                
                if box[3] >= self.height * self.bottom_limit or box[0] <= self.width * self.left_limit or box[2] >= self.width * self.right_limit:
                    self.state.remove_track(track_id)
                    self.smoother.drop(track_id)
                    self.prev_mask.pop(track_id, None)
                    return
                mask = self.prev_mask.get(track_id)
            else:
                return
        else:
            raw_box, mask = detection
            box = self.smoother.smooth(raw_box,track_id)
            if mask is not None:
                self.prev_mask[track_id] = mask

        x1, y1, x2, y2 = box.astype(int)

        self.update_log(track_id, (x2 - x1) * (y2 - y1))
        self.draw_track(frame, track_id, (x1, y1, x2, y2), mask)

    def update_log(self, track_id, area) -> None:
        if track_id not in self.track_log:
            self.track_log[track_id] = TrackLog(
                first_frame=self.frame_count,
                last_frame=self.frame_count,
                max_area=area
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
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    def export_csv(self) -> None:
        csv_path = self.output_path.rsplit(".", 1)[0] + ".csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "ilk_kare", "son_kare", "max_alan_px"])

            for tid in sorted(self.track_log):
                info = self.track_log[tid]
                writer.writerow([
                    tid,
                    info.first_frame,
                    info.last_frame,
                    info.max_area,
                ])

        print(f"Toplam {len(self.track_log)} cukur -> {csv_path}")

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
        if self.out is not None:
            self.out.write(frame)
        return True

    def run(self):
        self.load_model()
        self.initialize_video()
        try:
            while self.process_frame():
                pass
        finally:
            self.cleanup()
            self.export_csv()
