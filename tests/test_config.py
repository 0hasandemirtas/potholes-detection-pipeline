import pytest

from src.config import TrackingConfig, VideoConfig


def _make_tracking_config(**overrides):
    values = {
        "backend": "ultralytics",
        "tracker": "bytetrack.yaml",
        "conf": 0.25,
        "imgsz": 640,
        "n_confirm": 3,
        "m_persist": 5,
    }
    values.update(overrides)

    return TrackingConfig(**values)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("conf", -0.1),
        ("conf", 1.1),
        ("imgsz", 0),
        ("n_confirm", 0),
        ("m_persist", -1),
    ],
)

def test_tracking_config_rejects_invalid_values(
    field,
    invalid_value,
):
    with pytest.raises(
        ValueError,
        match=field,
    ):
        _make_tracking_config(
            **{field: invalid_value}
        )

def test_video_config_rejects_same_input_and_output_path():
    with pytest.raises(
        ValueError,
        match="video.output",
    ):
        VideoConfig(
            input="video.mp4",
            output="./video.mp4",
        )