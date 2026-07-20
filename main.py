import argparse
import logging
from pathlib import Path

from src.benchmark import append_benchmark_result
from src.config import Config
from src.factories import (
    create_box_smoother,
    create_tracking_backend,
)
from src.pipeline import PotholePipeline
from src.output_paths import configure_output_paths

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pothole detection and tracking pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="YAML configuration file path",
    )
    return parser.parse_args(argv)


def main(
        argv:list[str] | None = None
) -> int:

    args = parse_args(argv)
    logger = logging.getLogger(__name__)
    cfg = Config.from_yaml(args.config)
    run_id = configure_output_paths(cfg)

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
        if run_id is not None:
            logger.info("Deney kimligi: %s", run_id)
            logger.info("Video ciktisi: %s", cfg.video.output)
            logger.info("CSV ciktisi: %s", cfg.output.csv)

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
        return 1
    except RuntimeError as exc:
        logger.error("Calisma Hatasi: %s", exc)
        return 1
    except Exception:
        logger.exception("Beklenmeyen Hata olustu")
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
