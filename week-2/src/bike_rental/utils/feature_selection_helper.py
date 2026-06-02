

def select_input_features(context):
    """Return feature columns configured for the active experiment."""

    cfg = (
        context.resources
        .experiment_config
        .get_active_experiment()
    )

    feature_cfg = cfg["features"]

    numeric = feature_cfg["numeric"]
    categorical = feature_cfg["categorical"]
    boolean = feature_cfg["boolean"]
    cyclic = feature_cfg["cyclic"]

    selected_features = (
        numeric
        + categorical
        + boolean
        + cyclic
    )

    return selected_features, cfg

def select_target_features(context):
    """Return target column configured for the active experiment."""

    cfg = (
        context.resources
        .experiment_config
        .get_active_experiment()
    )

    return cfg["target"], cfg