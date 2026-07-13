from src.config import (
    ModelConfig,
    TrackingConfig,
    SmoothingConfig,
)
from src.protocols import (
    BoxSmootherProtocol,
    TrackingBackendProtocol,
)
from src.smoothing import EmaBoxSmoother, NoOpBoxSmoother
from src.tracking_backends import UltralyticsTrackingBackend

def create_box_smoother(config: SmoothingConfig) -> BoxSmootherProtocol:
    if config.type == "ema":
        return EmaBoxSmoother(alpha=config.alpha)
    elif config.type == "none":
        return NoOpBoxSmoother()
    else:
        raise ValueError(f"Geçersiz smoothing türü: {config.type}")

def create_tracking_backend(
    model_config: ModelConfig,
    tracking_config: TrackingConfig,
) -> TrackingBackendProtocol:
    if tracking_config.backend == "ultralytics":
        return UltralyticsTrackingBackend(
            model_path=model_config.path,
            conf=tracking_config.conf,
            imgsz=tracking_config.imgsz,
            tracker=tracking_config.tracker,
        )

    raise ValueError(
        f"Geçersiz tracking backend: "
        f"{tracking_config.backend}"
    )