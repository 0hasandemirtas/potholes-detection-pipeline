from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    path: str


@dataclass
class VideoConfig:
    input: str
    output: str

    def __post_init__(self) -> None:
        input_path = Path(self.input).resolve()
        output_path = Path(self.output).resolve()

        if input_path == output_path:
            raise ValueError("video.input ve video.output aynı dosya olamaz")


@dataclass
class TrackingConfig:
    backend: str
    tracker: str
    conf: float
    imgsz: int
    n_confirm: int
    m_persist: int
    confirmed_window: int = 5
    device: str | int | None = None

    def __post_init__(self) ->None:
        if not 0 <= self.conf <=1:
            raise ValueError(
                "tracking.conf 0 ile 1 arasında olmalıdır"
            )
        if self.imgsz <= 0:
            raise ValueError(
                "tracking.imgsz pozitif olmalıdır"
            )
        if self.n_confirm <= 0:
            raise ValueError(
                "tracking.n_confirm en az 1 olmalıdır"
            )
        if self.m_persist < 0:
            raise ValueError(
                "tracking.m_persist negatif olamaz"
            )
        if self.confirmed_window < self.n_confirm:
            raise ValueError(
                "tracking.confirmed_window, n_confirm değerinden küçük olamaz"
        )



@dataclass
class VisualizationConfig:
    draw_mask: bool
    draw_roi: bool = False
    max_stale_frames: int = 1

    def __post_init__(self) -> None:
        if self.max_stale_frames < 0:
            raise ValueError(
                "visualization.max_stale_frames negatif olamaz"
            )


@dataclass
class OutputConfig:
    csv: str
    log: str | None = None
    benchmark_csv: str | None = None
    auto_name: bool = False
    root_dir: str = "output"


@dataclass
class RoiConfig:
    points: list[list[float]]


@dataclass
class SmoothingConfig:
    type: str
    alpha: float = 0.5


@dataclass
class Config:
    roi: RoiConfig
    model: ModelConfig
    video: VideoConfig
    output: OutputConfig
    tracking: TrackingConfig
    smoothing: SmoothingConfig
    visualization: VisualizationConfig

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        return cls(
            roi=RoiConfig(**cfg["roi"]),
            model=ModelConfig(**cfg["model"]),
            video=VideoConfig(**cfg["video"]),
            output=OutputConfig(**cfg["output"]),
            tracking=TrackingConfig(**cfg["tracking"]),
            smoothing=SmoothingConfig(**cfg["smoothing"]),
            visualization=VisualizationConfig(**cfg["visualization"]),
        )
