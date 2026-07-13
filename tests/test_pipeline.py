from unittest.mock import Mock

from src.models import PipelineMetrics
from src.pipeline import PotholePipeline


def test_run_returns_pipeline_metrics(monkeypatch):
    pipeline = object.__new__(PotholePipeline)

    pipeline.initialize_video = Mock()
    pipeline.process_frame = Mock(
        side_effect=[True, False]
    )
    pipeline.cleanup = Mock()
    pipeline.export_csv = Mock()

    pipeline.frame_count = 10
    pipeline.track_log = {
        1: Mock(),
        2: Mock(),
    }

    times = iter([100.0, 102.0])

    monkeypatch.setattr(
        "src.pipeline.time.perf_counter",
        lambda: next(times),
    )

    metrics = pipeline.run()

    assert metrics == PipelineMetrics(
        frame_count=10,
        elapsed_seconds=2.0,
        average_fps=5.0,
        track_count=2,
    )

    pipeline.cleanup.assert_called_once()
    pipeline.export_csv.assert_called_once()