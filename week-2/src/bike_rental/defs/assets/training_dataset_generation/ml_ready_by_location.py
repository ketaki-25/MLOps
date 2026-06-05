import pandas as pd
from dagster import AssetExecutionContext, asset
from bike_rental.utils.feature_selection_helper import select_input_features

@asset(
    group_name="input_target_feature_split_by_location",
    required_resource_keys={"experiment_config"},
)
def X_train_hourly_by_location(
    context: AssetExecutionContext,
    train_dataset_hourly_by_location: pd.DataFrame,
):
    """Create training_dataset_generation feature matrix for hourly-by-location dataset
    using experiment configuration."""

    selected_features, cfg = select_input_features(context)

    if "location_id" in train_dataset_hourly_by_location.columns:
        selected_features = ["location_id"] + selected_features

    X = train_dataset_hourly_by_location[selected_features].copy()

    context.add_output_metadata({
        "rows": len(X),
        "columns": X.shape[1],
        "locations": (
            train_dataset_hourly_by_location["location_id"].nunique()
            if "location_id" in train_dataset_hourly_by_location.columns
            else None
        ),
        "model": cfg["model"],
        "target": cfg["target"],
        "feature_count": len(selected_features),
    })

    return X

@asset(
    group_name="input_target_feature_split_by_location",
    required_resource_keys={"experiment_config"},
)
def y_train_hourly_by_location(
    context: AssetExecutionContext,
    train_dataset_hourly_by_location: pd.DataFrame,
):
    """Create training_dataset_generation target vector for hourly-by-location dataset."""

    cfg = (
        context.resources
        .experiment_config
        .get_active_experiment()
    )

    target_col = cfg["target"]

    y = train_dataset_hourly_by_location[target_col].copy()

    context.add_output_metadata({
        "target": target_col,
        "rows": len(y),
        "mean": float(y.mean()),
        "min": float(y.min()),
        "max": float(y.max()),
    })

    return y

@asset(
    group_name="input_target_feature_split_by_location",
    required_resource_keys={"experiment_config"},
)
def X_test_hourly_by_location(
    context: AssetExecutionContext,
    test_dataset_hourly_by_location: pd.DataFrame,
):
    """Create test feature matrix for hourly-by-location dataset."""

    cfg = (
        context.resources
        .experiment_config
        .get_active_experiment()
    )

    feature_cfg = cfg["features"]

    selected_features = (
        feature_cfg["numeric"]
        + feature_cfg["categorical"]
        + feature_cfg["boolean"]
        + feature_cfg["cyclic"]
    )

    if "location_id" in test_dataset_hourly_by_location.columns:
        selected_features = ["location_id"] + selected_features

    X = test_dataset_hourly_by_location[selected_features].copy()

    context.add_output_metadata({
        "rows": len(X),
        "columns": X.shape[1],
        "model": cfg["model"],
        "target": cfg["target"],
    })

    return X

@asset(
    group_name="input_target_feature_split_by_location",
    required_resource_keys={"experiment_config"},
)
def y_test_hourly_by_location(
    context: AssetExecutionContext,
    test_dataset_hourly_by_location: pd.DataFrame,
):
    """Create test target vector for hourly-by-location dataset."""

    cfg = (
        context.resources
        .experiment_config
        .get_active_experiment()
    )

    target_col = cfg["target"]

    y = test_dataset_hourly_by_location[target_col].copy()

    context.add_output_metadata({
        "target": target_col,
        "rows": len(y),
        "mean": float(y.mean()),
        "min": float(y.min()),
        "max": float(y.max()),
    })

    return y


