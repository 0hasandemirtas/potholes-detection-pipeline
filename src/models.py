from dataclasses import dataclass


@dataclass
class Track:
    seen_count: int = 0
    missed_count: int = 0
    confirmed: bool = False


@dataclass
class TrackLog:
    first_frame: int
    last_frame: int
    max_area: int

@dataclass(frozen=True)
class PipelineMetrics:
    frame_count: int
    elapsed_seconds: float
    average_fps: float
    track_count: int