from datetime import datetime, timezone
from types import SimpleNamespace

from src.output_paths import configure_output_paths


def _make_config(*, auto_name: bool):
    return SimpleNamespace(
        video=SimpleNamespace(
            input="input/fast road.mp4",
            output="output/videos/manual.mp4",
        ),
        output=SimpleNamespace(
            csv="output/csv/manual.csv",
            log="output/logs/manual.log",
            auto_name=auto_name,
            root_dir="artifacts",
        ),
        tracking=SimpleNamespace(tracker="trackers/byte track.yaml"),
        smoothing=SimpleNamespace(type="none"),
    )


def test_configure_output_paths_uses_same_run_id_for_all_outputs():
    config = _make_config(auto_name=True)
    fixed_time = datetime(
        2026,
        7,
        20,
        14,
        35,
        22,
        123000,
        tzinfo=timezone.utc,
    )

    run_id = configure_output_paths(config, now=fixed_time)

    assert run_id == "20260720_143522_123_fast-road_byte-track_none"
    assert config.video.output == f"artifacts/videos/{run_id}.mp4"
    assert config.output.csv == f"artifacts/csv/{run_id}.csv"
    assert config.output.log == f"artifacts/logs/{run_id}.log"


def test_configure_output_paths_preserves_manual_paths_when_disabled():
    config = _make_config(auto_name=False)

    run_id = configure_output_paths(config)

    assert run_id is None
    assert config.video.output == "output/videos/manual.mp4"
    assert config.output.csv == "output/csv/manual.csv"
    assert config.output.log == "output/logs/manual.log"
