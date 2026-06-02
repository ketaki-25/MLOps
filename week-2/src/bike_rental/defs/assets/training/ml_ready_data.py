from dagster import AssetExecutionContext
from dagster import asset
from bike_rental.utils.feature_selection_helper import select_input_features, select_target_features


@asset(group_name="ml_ready_data", required_resource_keys={"experiment_config"}, io_manager_key="csv_io_manager")
def X_train_hourly(context: AssetExecutionContext, train_dataset_hourly):
    """Create training feature matrix using the active experiment configuration."""

    selected_features, cfg = select_input_features(context)

    X = train_dataset_hourly[
        selected_features
    ].copy()

    context.add_output_metadata(
        {
            "rows": len(X),
            "columns": X.shape[1],
            "selected_features": selected_features,
            "model": cfg["model"],
            "target": cfg["target"],
        }
    )

    return X


@asset(group_name="ml_ready_data", required_resource_keys={"experiment_config"}, io_manager_key="csv_io_manager")
def y_train_hourly(context: AssetExecutionContext, train_dataset_hourly):
    """Create training target vector using the active experiment configuration."""

    target_col, cfg = (select_target_features(context))

    y = train_dataset_hourly[target_col].copy()

    context.add_output_metadata(
        {
            "target_column": target_col,
            "rows": len(y),
            "mean": float(y.mean()),
            "min": float(y.min()),
            "max": float(y.max()),
            "null_values": int(y.isna().sum()),
        }
    )

    return y

@asset(group_name="ml_ready_data", required_resource_keys={"experiment_config"}, io_manager_key="csv_io_manager")
def X_test_hourly(
    context: AssetExecutionContext,
    test_dataset_hourly,
):
    """
    Create test feature matrix using the active
    experiment configuration.
    """
    selected_features, cfg = select_input_features(context)

    X = test_dataset_hourly[
        selected_features
    ].copy()

    context.add_output_metadata(
        {
            "rows": len(X),
            "columns": X.shape[1],
            "selected_features": selected_features,
            "model": cfg["model"],
            "target": cfg["target"],
        }
    )

    return X


@asset(group_name="ml_ready_data", required_resource_keys={"experiment_config"}, io_manager_key="csv_io_manager")
def y_test_hourly(
    context: AssetExecutionContext,
    test_dataset_hourly,
):
    """Create test target vector for model evaluation."""

    target_col, cfg = (select_target_features(context))

    y = test_dataset_hourly[target_col].copy()

    context.add_output_metadata(
        {
            "target_column": target_col,
            "rows": len(y),
            "mean": float(y.mean()),
            "min": float(y.min()),
            "max": float(y.max()),
            "null_values": int(y.isna().sum()),
        }
    )

    return y