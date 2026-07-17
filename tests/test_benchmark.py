import csv
from types import SimpleNamespace

from src.benchmark import append_benchmark_result
from src.models import PipelineMetrics


def test_append_benchmark_result_writes_csv(tmp_path):
    config = SimpleNamespace(
        tracking=SimpleNamespace(
            backend="ultralytics",
            tracker="bytetrack.yaml",
        ),
        smoothing=SimpleNamespace(
            type="ema",
            alpha=0.5,
        ),
        model=SimpleNamespace(
            path="models/test.pt",
        ),
        video=SimpleNamespace(
            input="input/test.mp4",
        ),
    )

    metrics = PipelineMetrics(
        frame_count=100,
        elapsed_seconds=4.0,
        average_fps=25.0,
        track_count=7,
    )

    output_path = tmp_path / "benchmarks" / "results.csv"

    append_benchmark_result(
        path=str(output_path),
        config=config,
        metrics=metrics,
    )

    with open(output_path, newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["tracker"] == "bytetrack.yaml"
    assert rows[0]["average_fps"] == "25.0"
    assert rows[0]["track_count"] == "7"
