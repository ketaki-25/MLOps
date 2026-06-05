import json
import yaml


VALID_MODELS = {"linear_regression", "random_forest", "xgboost"}


class ExperimentConfigResource:

    def __init__(self, yaml_path, active_json_path):
        self.yaml_path = yaml_path
        self.active_json_path = active_json_path

    def _validate_experiment(self, name, cfg):

        required = {"dataset", "target", "model", "features", "preprocessing"}

        missing = required - set(cfg.keys())
        if missing:
            raise ValueError(f"Missing keys: {missing}")

        if cfg["model"] not in VALID_MODELS:
            raise ValueError(f"Invalid model: {cfg['model']}")

        # 🔥 ensure all feature groups exist
        for group in ["numeric", "categorical", "boolean", "cyclic", "id_columns"]:
            if group not in cfg["features"]:
                raise ValueError(f"Missing feature group: {group}")

    def get_active_experiment(self):

        with open(self.active_json_path) as f:
            active = json.load(f)

        experiment_name = active["experiment"]

        with open(self.yaml_path) as f:
            experiments = yaml.safe_load(f)["experiments"]

        if experiment_name not in experiments:
            raise ValueError(f"Unknown experiment: {experiment_name}")

        cfg = experiments[experiment_name]

        self._validate_experiment(experiment_name, cfg)

        return cfg