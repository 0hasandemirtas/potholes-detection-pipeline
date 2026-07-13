from src.config import Config
from src.pipeline import PotholePipeline
from src.factories import (
    create_box_smoother,
    create_tracking_backend,
)
from src.benchmark import append_benchmark_result
import logging
from pathlib import Path


def main() -> None:

    logger = logging.getLogger(__name__)
    cfg = Config.from_yaml("config/config.yaml")

    Path(cfg.output.log).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(cfg.output.log),
        ],
    )

    try:
        smoother = create_box_smoother(cfg.smoothing)
        tracking_backend = create_tracking_backend(
            model_config=cfg.model,
            tracking_config=cfg.tracking,
        )
        pipeline = PotholePipeline(
            cfg,
            smoother=smoother,
            tracking_backend=tracking_backend,
        )
        metrics = pipeline.run()

        if cfg.output.benchmark_csv:
            append_benchmark_result(
                path=cfg.output.benchmark_csv,
                config=cfg,
                metrics=metrics,
            )

        logger.info(
            (
                "Deney özeti | backend=%s | tracker=%s | "
                "smoothing=%s | track=%s | ortalama_fps=%.2f"
            ),
            cfg.tracking.backend,
            cfg.tracking.tracker,
            cfg.smoothing.type,
            metrics.track_count,
            metrics.average_fps,
        )
    except FileNotFoundError as exc:
        logger.error("Dosya Hatasi: %s", exc)
    except RuntimeError as exc:
        logger.error("Calisma Hatasi: %s", exc)
    except Exception:
        logger.exception("Beklenmeyen Hata olustu")

if __name__ == "__main__":
    main()