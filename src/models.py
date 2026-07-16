from dataclasses import dataclass
import numpy as np


@dataclass
class Track:
    seen_count: int = 0
    missed_count: int = 0
    confirmed: bool = False


@dataclass
class TrackLog:
    first_frame: int
    last_frame: int
    max_box_area_px: int
    max_mask_area_px: float
    observation_count: int = 1
    confidence_sum: float = 0.0
    confidence_count: int = 0

    @property
    def average_confidence(self) -> float | None:
        if self.confidence_count == 0:
            return None

        return self.confidence_sum / self.confidence_count


@dataclass(frozen=True)
class TrackedDetection:
    box: np.ndarray
    mask: np.ndarray | None
    confidence: float | None = None
    class_id: int | None = None

@dataclass(frozen=True)
class PipelineMetrics:
    frame_count: int
    elapsed_seconds: float
    average_fps: float
    track_count: int
