from dagster import asset
from bike_rental.defs.assets.evaluation.eval_metrics import calculate_mae, calculate_rmse, calculate_r2, calculate_mape

@asset(group_name="model_evaluation")
def linear_regression_evaluation(
    context,
    linear_regression_model,
    X_test_processed_hourly,
    y_test_processed_hourly,
):
    """Evaluate Linear Regression model on test data."""

    preds = linear_regression_model.predict(X_test_processed_hourly)

    metrics = {
        "model": "linear_regression",
        "mae": float(calculate_mae(y_test_processed_hourly, preds)),
        "rmse": float(calculate_rmse(y_test_processed_hourly, preds)),
        "r2": float(calculate_r2(y_test_processed_hourly, preds)),
        "mape": float(calculate_mape(y_test_processed_hourly, preds)),
    }

    context.add_output_metadata(metrics)

    return metrics