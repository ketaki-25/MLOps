import yaml
from pathlib import Path


class ConfigResource:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)

    def load(self):
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)