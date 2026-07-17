import numpy as np


class EmaBoxSmoother:
    def __init__(self, alpha=0.5) -> None:
        self.alpha = alpha
        self.prev_box = {}

    def smooth(self, box, track_id) -> np.ndarray[float]:
        """Verilen track_id için kutuyu yumuşatır. Eğer track_id yoksa kutuyu kaydeder ve döndürür, varsa önceki kutu ile yumuşatır ve döndürür."""
        box = np.asarray(box, dtype=np.float32)
        if track_id not in self.prev_box:
            self.prev_box[track_id] = box
            return box
        else:
            self.prev_box[track_id] = (
                self.alpha * box + (1 - self.alpha) * self.prev_box[track_id]
            )
            return self.prev_box[track_id]

    def get_last_box(
        self,
        track_id: int,
    ) -> np.ndarray | None:
        return self.prev_box.get(track_id)

    def drop(self, track_id) -> None:
        """Verilen track_id için kaydedilen kutuyu siler."""
        self.prev_box.pop(track_id, None)


class NoOpBoxSmoother:
    """Kutuları değiştirmeden döndüren smoothing stratejisi."""

    def __init__(self) -> None:
        self.prev_box = {}

    def smooth(
        self,
        box,
        track_id: int,
    ) -> np.ndarray:
        box = np.asarray(box, dtype=np.float32)
        self.prev_box[track_id] = box
        return box

    def get_last_box(
        self,
        track_id: int,
    ) -> np.ndarray | None:
        return self.prev_box.get(track_id)

    def drop(self, track_id: int) -> None:
        self.prev_box.pop(track_id, None)
