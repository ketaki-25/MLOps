import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from mlflow import start_run, log_param, log_metric, MlflowClient


MLFLOW_TRACKING_URI = "http://localhost:5000"

def init_mlflow(experiment_name: str):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

def get_latest_version(experiment_name, model_name: str):
    client = MlflowClient()

    latest_version = max(
        int(v.version)
        for v in client.search_model_versions(
            f"name='{experiment_name}_{model_name}'"
        )
    )
    return latest_version

def get_champion_version(model_name):
    client = MlflowClient()

    try:
        mv = client.get_model_version_by_alias(
            model_name,
            "champion"
        )
        return mv
    except Exception:
        return None

def is_better_model(
    new_rmse: float,
    current_rmse: float,
):
    return new_rmse < current_rmse

def promote_to_production(
    model_name: str,
    version: str,
):
    client = MlflowClient()

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=version,
    )

def auto_promote_model(
    model_name: str,
    version: str,
    new_rmse: float,
):
    client = MlflowClient()

    try:
        champion = client.get_model_version_by_alias(
            model_name,
            "champion",
        )

        run = client.get_run(champion.run_id)

        champion_rmse = float(
            run.data.metrics["rmse"]
        )

        if new_rmse < champion_rmse:

            print(
                f"Promoting version {version}. "
                f"RMSE improved "
                f"{champion_rmse:.3f} -> {new_rmse:.3f}"
            )

            client.set_registered_model_alias(
                model_name,
                "champion",
                version,
            )

            return True

        else:
            print(
                f"Keeping champion. "
                f"Current RMSE={champion_rmse:.3f}, "
                f"Candidate RMSE={new_rmse:.3f}"
            )

            return False

    except Exception:
        # No champion exists yet
        client.set_registered_model_alias(
            model_name,
            "champion",
            version,
        )

        return True

def log_model_run(
    experiment_name: str,
    run_name: str,
    model,
    params: dict,
    metrics: dict,
    input_example,
    model_name: str = "model",
    tags : dict = None,
):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

    with start_run(run_name=run_name) as run:

        # log params
        for k, v in params.items():
            log_param(k, v)

        # log metrics
        for k, v in metrics.items():
            log_metric(k, v)

        # log tags
        if tags:
            mlflow.set_tags(tags)

        predictions = model.predict(input_example)

        signature = infer_signature(
            input_example,
            predictions,
        )

        # log model as MLflow artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            registered_model_name=f"{experiment_name}_{model_name}",
            signature=signature,
            input_example=input_example,
        )

        return run.info.run_id