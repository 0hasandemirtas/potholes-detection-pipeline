from typing import Protocol
import numpy as np
from src.models import TrackedDetection


class BoxSmootherProtocol(Protocol):
    def smooth(self, box: np.ndarray, track_id: int) -> np.ndarray: ...

    def get_last_box(
        self,
        track_id: int,
    ) -> np.ndarray | None: ...

    def drop(self, track_id: int) -> None: ...


class TrackingBackendProtocol(Protocol):
    def track(
        self,
        frame: np.ndarray,
    ) -> dict[int, TrackedDetection]: ...
