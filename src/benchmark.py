import csv
from datetime import datetime
from pathlib import Path

from src.config import Config
from src.models import PipelineMetrics


FIELDNAMES = [
    "timestamp",
    "backend",
    "tracker",
    "smoothing",
    "alpha",
    "model_path",
    "video_path",
    "frame_count",
    "elapsed_seconds",
    "average_fps",
    "track_count",
]


def append_benchmark_result(
    path: str,
    config: Config,
    metrics: PipelineMetrics,
) -> None:
    benchmark_path = Path(path)
    benchmark_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_header = (
        not benchmark_path.exists()
        or benchmark_path.stat().st_size == 0
    )

    with open(benchmark_path, "a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
                "backend": config.tracking.backend,
                "tracker": config.tracking.tracker,
                "smoothing": config.smoothing.type,
                "alpha": config.smoothing.alpha,
                "model_path": config.model.path,
                "video_path": config.video.input,
                "frame_count": metrics.frame_count,
                "elapsed_seconds": metrics.elapsed_seconds,
                "average_fps": metrics.average_fps,
                "track_count": metrics.track_count,
            }
        )