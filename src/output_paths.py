import re
from datetime import datetime
from pathlib import Path

from src.config import Config


def _slug(value: str) -> str:
    """Dosya adında güvenle kullanılabilecek kısa bir parça üretir."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")
    return cleaned or "run"


def configure_output_paths(
    config: Config,
    *,
    now: datetime | None = None,
) -> str | None:
    """Etkinse tek bir run_id ile video, CSV ve log yollarını oluşturur."""
    if not getattr(config.output, "auto_name", False):
        return None

    timestamp = (now or datetime.now().astimezone()).strftime(
        "%Y%m%d_%H%M%S_%f"
    )[:-3]
    video_name = _slug(Path(config.video.input).stem)
    tracker_name = _slug(Path(config.tracking.tracker).stem)
    smoothing_name = _slug(config.smoothing.type)
    run_id = "_".join(
        [timestamp, video_name, tracker_name, smoothing_name]
    )

    root_dir = Path(getattr(config.output, "root_dir", "output"))
    config.video.output = str(root_dir / "videos" / f"{run_id}.mp4")
    config.output.csv = str(root_dir / "csv" / f"{run_id}.csv")
    config.output.log = str(root_dir / "logs" / f"{run_id}.log")

    return run_id
