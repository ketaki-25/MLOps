from bike_rental.defs.config.experiment_config import ExperimentConfigResource
from bike_rental.defs.assets.training_dataset_generation.ml_ready_data import create_ml_ready_assets
from bike_rental.defs.assets.models.linear_regression import create_linear_regression_asset
from bike_rental.defs.assets.models.xgboost import create_xgboost_asset
from bike_rental.defs.assets.models.random_forest import create_random_forest_asset

MODEL_REGISTRY = {
    "linear_regression": create_linear_regression_asset,
    "xgboost": create_xgboost_asset,
    "random_forest": create_random_forest_asset,
}


def generate_experiment_assets():

    config = ExperimentConfigResource(
        yaml_path="src/bike_rental/defs/config/experiments.yml",
    )

    assets = []

    experiments = config.get_all_experiments()

    for experiment_name, experiment_cfg in experiments.items():

        models_cfg = experiment_cfg.get("models", {})

        # =========================================================
        # 1. ML-ready assets PER MODEL (IMPORTANT CHANGE)
        # =========================================================
        for model_name, model_cfg in models_cfg.items():

            assets.extend(
                create_ml_ready_assets(
                    experiment_name=experiment_name,
                    model_name=model_name,
                    experiment_cfg=experiment_cfg,
                    model_cfg=model_cfg,
                )
            )

        # =========================================================
        # 2. MODEL TRAINING PIPELINES (unchanged structurally)
        # =========================================================
        for model_name, model_cfg in models_cfg.items():

            model_factory = MODEL_REGISTRY[model_name]

            assets.append(
                model_factory(
                    experiment_name=experiment_name,
                    model_name=model_name,
                    cfg=model_cfg,
                )
            )

    return assets


EXPERIMENT_ASSETS = generate_experiment_assets()