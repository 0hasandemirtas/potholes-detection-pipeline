from dataclasses import dataclass
import yaml


@dataclass
class ModelConfig:
    path: str


@dataclass
class VideoConfig:
    input: str
    output: str


@dataclass
class TrackingConfig:
    conf: float
    imgsz: int
    alpha: float
    n_confirm: int
    m_persist: int


@dataclass
class VisualizationConfig:
    draw_mask: bool
    draw_roi: bool = False


@dataclass
class OutputConfig:
    csv: str
    log: str | None = None

@dataclass
class RoiConfig:
    points: list[list[float]]


@dataclass
class Config:
    roi: RoiConfig
    model: ModelConfig
    video: VideoConfig
    output: OutputConfig
    tracking: TrackingConfig
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
            visualization=VisualizationConfig(**cfg["visualization"]),
        )
