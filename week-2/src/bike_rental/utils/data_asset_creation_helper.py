from dagster import AssetKey

def _get_selected_features(model_cfg: dict) -> list[str]:
    selected_features = []
    for feature_group in model_cfg["features"].values():
        if feature_group:
            selected_features.extend(feature_group)
    return selected_features


def _base_metadata(experiment_name: str, model_name: str, experiment_cfg: dict) -> dict:
    return {
        "experiment": experiment_name,
        "model": model_name,
        "target": experiment_cfg["target"],
        "dataset": experiment_cfg["dataset"],
    }


def _prefix(experiment_name: str, model_name: str) -> str:
    return f"{experiment_name}_{model_name}"


def _ak(name: str) -> AssetKey:
    return AssetKey([name])
