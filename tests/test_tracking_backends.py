from unittest.mock import Mock

import numpy as np

import src.tracking_backends as backend_module
from src.tracking_backends import UltralyticsTrackingBackend


def test_parse_results_returns_empty_dict_for_empty_results():
    detections = UltralyticsTrackingBackend._parse_results([])

    assert detections == {}

def test_backend_calls_model_track(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pt"
    model_path.touch()

    fake_model = Mock()
    fake_model.track.return_value = []

    fake_yolo = Mock(return_value=fake_model)
    monkeypatch.setattr(
        backend_module,
        "YOLO",
        fake_yolo,
    )

    backend = UltralyticsTrackingBackend(
        model_path=str(model_path),
        conf=0.25,
        imgsz=640,
        tracker="bytetrack.yaml",
    )

    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    detections = backend.track(frame)

    assert detections == {}

    fake_yolo.assert_called_once_with(str(model_path))
    fake_model.track.assert_called_once()

    args, kwargs = fake_model.track.call_args

    assert args[0] is frame
    assert kwargs == {
        "conf": 0.25,
        "imgsz": 640,
        "tracker": "bytetrack.yaml",
        "persist": True,
    }