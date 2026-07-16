from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.pipeline import PotholePipeline
from src.smoothing import NoOpBoxSmoother
from src.models import PipelineMetrics, TrackedDetection


def test_run_returns_pipeline_metrics(monkeypatch):
    pipeline = object.__new__(PotholePipeline)

    pipeline.initialize_video = Mock()
    pipeline.process_frame = Mock(side_effect=[True, False])
    pipeline.cleanup = Mock()
    pipeline.export_csv = Mock()
    pipeline.publish_outputs = Mock()

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
    pipeline.publish_outputs.assert_called_once()
    pipeline.cleanup.assert_called_once()
    pipeline.export_csv.assert_called_once()

def test_roi_outside_detections_do_not_contribute_to_confirmation():
    cfg = SimpleNamespace(
        video=SimpleNamespace(
            input="unused.mp4",
            output="unused-output.mp4",
        ),
        output=SimpleNamespace(csv="unused.csv"),
        tracking=SimpleNamespace(
            n_confirm=3,
            m_persist=5,
        ),
        visualization=SimpleNamespace(
            draw_mask=False,
            draw_roi=False,
        ),
        roi=SimpleNamespace(points=[]),
    )

    backend = Mock()
    smoother = NoOpBoxSmoother()
    pipeline = PotholePipeline(
        cfg,
        smoother=smoother,
        tracking_backend=backend,
    )

    outside_box = np.array(
        [110, 10, 130, 30],
        dtype=np.float32,
    )
    inside_box = np.array(
        [40, 40, 60, 60],
        dtype=np.float32,
    )

    backend.track.side_effect = [
        {1: TrackedDetection(box=outside_box, mask=None)},
        {1: TrackedDetection(box=outside_box, mask=None)},
        {1: TrackedDetection(box=inside_box, mask=None)},
    ]

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pipeline.cap = Mock()
    pipeline.cap.read.side_effect = [
        (True, frame.copy()),
        (True, frame.copy()),
        (True, frame.copy()),
    ]
    pipeline.out = None

    pipeline.roi_polygon = np.array(
        [
            [0, 0],
            [99, 0],
            [99, 99],
            [0, 99],
        ],
        dtype=np.int32,
    )

    for _ in range(3):
        assert pipeline.process_frame() is True

    assert pipeline.state.is_visible(1) is False

def _make_roi_pipeline(*, n_confirm, detections):
    cfg = SimpleNamespace(
        video=SimpleNamespace(
            input="unused.mp4",
            output="unused-output.mp4",
        ),
        output=SimpleNamespace(csv="unused.csv"),
        tracking=SimpleNamespace(
            n_confirm=n_confirm,
            m_persist=5,
        ),
        visualization=SimpleNamespace(
            draw_mask=False,
            draw_roi=False,
        ),
        roi=SimpleNamespace(points=[]),
    )

    backend = Mock()
    backend.track.side_effect = detections

    pipeline = PotholePipeline(
        cfg,
        smoother=NoOpBoxSmoother(),
        tracking_backend=backend,
    )

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pipeline.cap = Mock()
    pipeline.cap.read.side_effect = [
        (True, frame.copy())
        for _ in detections
    ]
    pipeline.out = None

    pipeline.roi_polygon = np.array(
        [
            [0, 0],
            [99, 0],
            [99, 99],
            [0, 99],
        ],
        dtype=np.int32,
    )

    return pipeline

def test_confirmed_track_is_removed_after_leaving_roi():
    inside_box = np.array(
        [40, 40, 60, 60],
        dtype=np.float32,
    )
    outside_box = np.array(
        [110, 10, 130, 30],
        dtype=np.float32,
    )

    pipeline = _make_roi_pipeline(
        n_confirm=1,
        detections=[
            {1: TrackedDetection(box=inside_box, mask=None)},
            {1: TrackedDetection(box=outside_box, mask=None)},
        ],
    )

    assert pipeline.process_frame() is True
    assert pipeline.state.is_visible(1) is True

    assert pipeline.process_frame() is True

    assert 1 not in pipeline.state.tracks
    assert pipeline.smoother.get_last_box(1) is None

def test_run_does_not_export_csv_when_processing_fails(monkeypatch):
    pipeline = object.__new__(PotholePipeline)

    pipeline.initialize_video = Mock()
    pipeline.process_frame = Mock(
        side_effect=RuntimeError("inference failed")
    )
    pipeline.cleanup = Mock()
    pipeline.export_csv = Mock()

    pipeline.frame_count = 1
    pipeline.track_log = {}

    times = iter([100.0, 101.0])
    monkeypatch.setattr(
        "src.pipeline.time.perf_counter",
        lambda: next(times),
    )

    with pytest.raises(
        RuntimeError,
        match="inference failed",
    ):
        pipeline.run()

    pipeline.cleanup.assert_called_once()
    pipeline.export_csv.assert_not_called()

def test_run_cleans_up_when_video_initialization_fails():
    pipeline = object.__new__(PotholePipeline)

    pipeline.initialize_video = Mock(
        side_effect=RuntimeError("video writer could not be opened")
    )
    pipeline.process_frame = Mock()
    pipeline.cleanup = Mock()
    pipeline.export_csv = Mock()

    with pytest.raises(
        RuntimeError,
        match="video writer could not be opened",
    ):
        pipeline.run()

    pipeline.process_frame.assert_not_called()
    pipeline.cleanup.assert_called_once()
    pipeline.export_csv.assert_not_called()

def test_track_log_aggregates_mask_area_and_confidence():
    first_box = np.array(
        [10, 10, 30, 30],
        dtype=np.float32,
    )
    second_box = np.array(
        [10, 10, 40, 40],
        dtype=np.float32,
    )

    first_mask = np.array(
        [
            [10, 10],
            [20, 10],
            [20, 20],
            [10, 20],
        ],
        dtype=np.float32,
    )
    second_mask = np.array(
        [
            [10, 10],
            [18, 10],
            [18, 18],
            [10, 18],
        ],
        dtype=np.float32,
    )

    pipeline = _make_roi_pipeline(
        n_confirm=1,
        detections=[
            {
                1: TrackedDetection(
                    box=first_box,
                    mask=first_mask,
                    confidence=0.8,
                )
            },
            {
                1: TrackedDetection(
                    box=second_box,
                    mask=second_mask,
                    confidence=0.6,
                )
            },
        ],
    )

    assert pipeline.process_frame() is True
    assert pipeline.process_frame() is True

    log = pipeline.track_log[1]

    assert log.first_frame == 1
    assert log.last_frame == 2
    assert log.observation_count == 2
    assert log.max_box_area_px == 900
    assert log.max_mask_area_px == 100.0
    assert log.average_confidence == pytest.approx(0.7)

def test_export_csv_writes_to_pending_file(
    tmp_path,
):
    final_csv = tmp_path / "output.csv"
    pending_csv = tmp_path / "output.partial.csv"

    final_csv.write_text(
        "previous valid result",
        encoding="utf-8",
    )

    pipeline = object.__new__(PotholePipeline)
    pipeline.csv_path = str(final_csv)
    pipeline.pending_csv_path = pending_csv
    pipeline.track_log = {}

    pipeline.export_csv()

    assert final_csv.read_text(
        encoding="utf-8",
    ) == "previous valid result"

    assert pending_csv.is_file()
    assert "track_id" in pending_csv.read_text(
        encoding="utf-8",
    )