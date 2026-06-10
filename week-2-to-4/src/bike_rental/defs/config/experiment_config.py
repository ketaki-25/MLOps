import yaml


VALID_MODELS = {"linear_regression", "random_forest", "xgboost"}


class ExperimentConfigResource:

    def __init__(self, yaml_path):
        self.yaml_path = yaml_path

    def _validate_experiment(self, name, cfg):

        required = {"dataset", "target", "models"}

        missing = required - set(cfg.keys())
        if missing:
            raise ValueError(
                f"Experiment '{name}' missing keys: {missing}"
            )

        if not cfg["dataset"]:
            raise ValueError(
                f"Experiment '{name}' has empty dataset"
            )

        if not cfg["target"]:
            raise ValueError(
                f"Experiment '{name}' has empty target"
            )

        for model_name, model_cfg in cfg["models"].items():

            if model_name not in VALID_MODELS:
                raise ValueError(
                    f"Experiment '{name}' uses invalid model '{model_name}'"
                )

            model_required = {"features", "preprocessing"}

            missing = model_required - set(model_cfg.keys())
            if missing:
                raise ValueError(
                    f"Model '{model_name}' in experiment '{name}' "
                    f"missing keys: {missing}"
                )

            for group in [
                "numeric",
                "categorical",
                "boolean",
                "cyclic",
                "id_columns",
            ]:
                if group not in model_cfg["features"]:
                    raise ValueError(
                        f"Model '{model_name}' missing feature group '{group}'"
                    )

    def get_all_experiments(self):

        with open(self.yaml_path) as f:
            experiments = yaml.safe_load(f)["experiments"]

        for name, cfg in experiments.items():
            self._validate_experiment(name, cfg)

        return experiments

    def get_experiment(self, experiment_name):

        experiments = self.get_all_experiments()

        if experiment_name not in experiments:
            raise ValueError(
                f"Unknown experiment: {experiment_name}"
            )

        return experiments[experiment_name]