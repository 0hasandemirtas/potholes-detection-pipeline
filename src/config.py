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


@dataclass
class OutputConfig:
    csv: str
    #log: str

@dataclass
class LimitsConfig:
    bottom: float
    left: float
    right: float


@dataclass
class Config:
    model: ModelConfig
    video: VideoConfig
    tracking: TrackingConfig
    visualization: VisualizationConfig
    output: OutputConfig
    limits: LimitsConfig

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**cfg["model"]),
            video=VideoConfig(**cfg["video"]),
            tracking=TrackingConfig(**cfg["tracking"]),
            visualization=VisualizationConfig(**cfg["visualization"]),
            output=OutputConfig(**cfg["output"]),
            limits=LimitsConfig(**cfg["limits"]),
        )