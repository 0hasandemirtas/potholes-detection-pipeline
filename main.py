from src.config import Config
from src.pipeline import PotholePipeline


cfg = Config.from_yaml("config/config.yaml")

pipeline = PotholePipeline(cfg)
pipeline.run()
