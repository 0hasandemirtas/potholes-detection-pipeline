from types import SimpleNamespace
from unittest.mock import Mock

import main as main_module


def test_main_returns_failure_code_when_pipeline_fails(
    monkeypatch,
    tmp_path,
):
    cfg = SimpleNamespace(
        output=SimpleNamespace(
            log=str(tmp_path / "pipeline.log"),
            benchmark_csv=None,
        ),
        model=SimpleNamespace(path="unused.pt"),
        tracking=SimpleNamespace(
            backend="ultralytics",
            tracker="bytetrack.yaml",
        ),
        smoothing=SimpleNamespace(type="none"),
    )

    config_loader = Mock(return_value=cfg)

    monkeypatch.setattr(
        main_module.Config,
        "from_yaml",
        config_loader,
    )
    monkeypatch.setattr(
        main_module,
        "create_box_smoother",
        Mock(return_value=object()),
    )
    monkeypatch.setattr(
        main_module,
        "create_tracking_backend",
        Mock(return_value=object()),
    )

    pipeline = Mock()
    pipeline.run.side_effect = RuntimeError("inference failed")

    monkeypatch.setattr(
        main_module,
        "PotholePipeline",
        Mock(return_value=pipeline),
    )

    result = main_module.main(
        ["--config", "custom-config.yaml"]
    )

    assert result == 1

    config_loader.assert_called_once_with(
        "custom-config.yaml"
    )