from src.config import Config
from src.pipeline import PotholePipeline
import logging
from pathlib import Path

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
    pipeline = PotholePipeline(cfg)
    pipeline.run()
except FileNotFoundError as exc:
    logger.error("Dosya Hatasi: %s", exc)
except RuntimeError as exc:
    logger.error("Calisma Hatasi: %s", exc)
except Exception:
    logger.error("Bilinmeyen Hata olustu")
