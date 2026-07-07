import numpy as np

class BoxSmoother:
    def __init__(self, alpha=0.5) -> None:
        self.alpha = alpha
        self.prev_box = {}

    def smooth(self, box, track_id) -> np.ndarray[float]:
        box = np.asarray(box, dtype=np.float32)
        if track_id not in self.prev_box:
            self.prev_box[track_id] = box
            return box
        else:
            self.prev_box[track_id] = self.alpha * box + (1 - self.alpha) * self.prev_box[track_id]
            return self.prev_box[track_id]
    def drop(self, track_id) -> None:
        self.prev_box.pop(track_id, None)