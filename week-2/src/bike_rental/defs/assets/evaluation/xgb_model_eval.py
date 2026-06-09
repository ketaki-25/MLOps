from dagster import asset
from bike_rental.defs.assets.evaluation.eval_metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_r2,
    calculate_mape,
)

@asset(group_name="model_evaluation")
def total_hourly_xgboost_evaluation(
    context,
    total_hourly_xgboost_model,
    total_hourly_xgboost_X_test_processed,
    total_hourly_xgboost_y_test_processed,
):
    """Evaluate XGBoost model on test data."""

    preds = total_hourly_xgboost_model.predict(
        total_hourly_xgboost_X_test_processed
    )

    metrics = {
        "model": "xgboost",
        "mae": float(calculate_mae(total_hourly_xgboost_y_test_processed, preds)),
        "rmse": float(calculate_rmse(total_hourly_xgboost_y_test_processed, preds)),
        "r2": float(calculate_r2(total_hourly_xgboost_y_test_processed, preds)),
        "mape": float(calculate_mape(total_hourly_xgboost_y_test_processed, preds)),
    }

    context.add_output_metadata(metrics)

    return metrics