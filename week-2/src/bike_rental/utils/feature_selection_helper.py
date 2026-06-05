

def select_input_features(context):
    """Return consistent feature list across pipeline."""

    cfg = context.resources.experiment_config.get_active_experiment()
    feature_cfg = cfg["features"]

    def safe_get(key):
        value = feature_cfg.get(key, [])
        return value if value else []

    selected_features = (
        safe_get("numeric")
        + safe_get("categorical")
        + safe_get("boolean")
        + safe_get("cyclic")
        + safe_get("id_columns")
    )

    selected_features = list(dict.fromkeys(selected_features))

    return selected_features, cfg

def select_target_features(context):
    """Return target column configured for the active experiment."""

    cfg = (
        context.resources
        .experiment_config
        .get_active_experiment()
    )

    return cfg["target"], cfg