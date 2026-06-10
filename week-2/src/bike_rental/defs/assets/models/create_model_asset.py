from dagster import asset, AssetIn, AssetKey
import joblib
import os

from bike_rental.defs.assets.models.estimator_registry import MODEL_REGISTRY
from bike_rental.defs.assets.evaluation.eval_metrics import evaluate_model
from bike_rental.utils.mlflow_utils import log_model_run, auto_promote_model, get_latest_version


def create_model_asset(
    experiment_name: str,
    model_name: str,
    estimator_name: str,
):
    """
    Generic model training + evaluation asset.

    Adds MLflow logging and optional model registration in addition to the
    existing local joblib persistence.
    """

    asset_name = f"{experiment_name}_{model_name}_model"

    ml_ready_key = AssetKey(
        [f"{experiment_name}_{model_name}_ml_ready_dataset"]
    )

    @asset(
        name=asset_name,
        group_name="model_training",
        ins={
            "ml_ready": AssetIn(key=ml_ready_key),
        },
    )
    def model_asset(context, ml_ready):

        # -------------------------
        # Get estimator
        # -------------------------
        estimator_fn = MODEL_REGISTRY[estimator_name]
        model = estimator_fn()

        # -------------------------
        # Data unpacking
        # -------------------------
        X_train = ml_ready["X_train"]
        y_train = ml_ready["y_train"]
        X_test = ml_ready["X_test"]
        y_test = ml_ready["y_test"]

        # -------------------------
        # Train
        # -------------------------
        model.fit(X_train, y_train)

        # -------------------------
        # Evaluate
        # -------------------------
        metrics = evaluate_model(model, X_test, y_test)

        # -------------------------
        # Persist model locally
        # -------------------------
        path = (
            f"data/output_data/models/"
            f"{experiment_name}_{model_name}.pkl"
        )

        os.makedirs(os.path.dirname(path), exist_ok=True)

        joblib.dump(
            {
                "model": model,
                "features": list(X_train.columns),
                "metrics": metrics,
            },
            path,
        )

        context.add_output_metadata({
            "model_type": estimator_name,
            "experiment": experiment_name,
            "model": model_name,
            "model_path": path,
            **metrics,
        })

        # -------------------------
        # MLflow log metrics and params
        # -------------------------
        log_model_run(
            experiment_name=experiment_name,
            run_name=f"{model_name}_{estimator_name}",
            model=model,
            params={
                "model": estimator_name,
                "n_features": len(X_train.columns),
            },
            metrics=metrics,
            model_name=model_name,
            input_example=X_test.head(5),
        )

        latest_version = get_latest_version(experiment_name, model_name)

        auto_promote_model(
            model_name=f"{experiment_name}_{model_name}",
            version=str(latest_version),
            new_rmse=metrics["rmse"],
        )

        return {
            "model": model,
            "metrics": metrics,
        }

    return model_asset