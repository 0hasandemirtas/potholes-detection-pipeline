from unittest.mock import Mock

import pytest

import src.factories as factories_module
from src.config import (
    ModelConfig,
    SmoothingConfig,
    TrackingConfig,
)
from src.factories import (
    create_box_smoother,
    create_tracking_backend,
)
from src.smoothing import EmaBoxSmoother

def test_factory_creates_ema_smoother():
    config = SmoothingConfig(
        type="ema",
        alpha=0.7,
    )

    smoother = create_box_smoother(config)

    assert isinstance(smoother, EmaBoxSmoother)
    assert smoother.alpha == 0.7

def test_factory_rejects_unknown_smoother_type():
    config = SmoothingConfig(
        type="unknown",
        alpha=0.5,
    )

    with pytest.raises(
        ValueError,
        match="Geçersiz smoothing türü",
    ):
        create_box_smoother(config)

def test_factory_creates_ultralytics_tracking_backend(
    monkeypatch,
):
    fake_backend = object()
    backend_constructor = Mock(
        return_value=fake_backend
    )

    monkeypatch.setattr(
        factories_module,
        "UltralyticsTrackingBackend",
        backend_constructor,
    )

    model_config = ModelConfig(
        path="models/test.pt"
    )
    tracking_config = TrackingConfig(
        backend="ultralytics",
        tracker="bytetrack.yaml",
        conf=0.25,
        imgsz=640,
        n_confirm=3,
        m_persist=5,
    )

    result = create_tracking_backend(
        model_config=model_config,
        tracking_config=tracking_config,
    )

    assert result is fake_backend

    backend_constructor.assert_called_once_with(
        model_path="models/test.pt",
        conf=0.25,
        imgsz=640,
        tracker="bytetrack.yaml",
    )