from bike_rental.defs.config.experiment_config import ExperimentConfigResource
from bike_rental.defs.assets.training_dataset_generation.ml_ready_data import create_ml_ready_assets
from bike_rental.defs.assets.models.create_model_asset import create_model_asset
def generate_experiment_assets():

    config = ExperimentConfigResource(
        yaml_path="src/bike_rental/defs/config/experiments.yml",
    )

    assets = []

    experiments = config.get_all_experiments()

    for experiment_name, experiment_cfg in experiments.items():

        models_cfg = experiment_cfg.get("models", {})

        # =========================================================
        # PER MODEL PIPELINE (STRICT ORDER)
        # =========================================================
        for model_name, model_cfg in models_cfg.items():

            # ---------------------------------------
            # 1. ML-ready dataset (single asset)
            # ---------------------------------------
            assets.extend(
                create_ml_ready_assets(
                    experiment_name=experiment_name,
                    model_name=model_name,
                    experiment_cfg=experiment_cfg,
                    model_cfg=model_cfg,
                )
            )

            # ---------------------------------------
            # 2. Model asset (depends on ml_ready_dataset)
            # ---------------------------------------
            estimator_name = model_name  # matches registry keys

            assets.append(
                create_model_asset(
                    experiment_name=experiment_name,
                    model_name=model_name,
                    estimator_name=estimator_name,
                )
            )

    return assets


EXPERIMENT_ASSETS = generate_experiment_assets()