from unittest.mock import Mock

import numpy as np

import src.tracking_backends as backend_module
from src.tracking_backends import UltralyticsTrackingBackend
from types import SimpleNamespace
from src.models import TrackedDetection


def test_parse_results_returns_empty_dict_for_empty_results():
    detections = UltralyticsTrackingBackend._parse_results([])

    assert detections == {}

def test_parse_results_returns_structured_detection():
    boxes = Mock()

    expected_box = np.array(
        [[10, 20, 100, 200]],
        dtype=np.float32,
    )
    boxes.xyxy.cpu.return_value.numpy.return_value = expected_box

    boxes.id.int.return_value.cpu.return_value.tolist.return_value = [7]
    boxes.conf.cpu.return_value.tolist.return_value = [0.9]
    boxes.cls.int.return_value.cpu.return_value.tolist.return_value = [2]

    expected_mask = np.array(
        [[10, 20], [100, 20], [100, 200]],
        dtype=np.float32,
    )

    result = SimpleNamespace(
        boxes=boxes,
        masks=SimpleNamespace(xy=[expected_mask]),
    )

    detections = UltralyticsTrackingBackend._parse_results(
        [result]
    )

    detection = detections[7]

    assert isinstance(detection, TrackedDetection)
    np.testing.assert_array_equal(
        detection.box,
        expected_box[0],
    )
    np.testing.assert_array_equal(
        detection.mask,
        expected_mask,
    )
    assert detection.confidence == 0.9
    assert detection.class_id == 2

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
        device="cpu",
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
        "device": "cpu",
    }
