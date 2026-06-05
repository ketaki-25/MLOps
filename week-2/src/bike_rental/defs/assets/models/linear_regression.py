from sklearn.linear_model import LinearRegression

import joblib
import os
from dagster import asset, AssetExecutionContext


def train_linear_model(X_train, y_train):
    """Pure training_dataset_generation function"""

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

@asset(group_name="model_training")
def linear_regression_model(
    context: AssetExecutionContext,
    X_train_processed_hourly,
    y_train_processed_hourly,
):
    """Train and persist Linear Regression model."""

    model = train_linear_model(X_train_processed_hourly, y_train_processed_hourly)

    path = "data/output_data/models/linear_regression.pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": list(X_train_processed_hourly.columns),
        },
        path
    )

    context.add_output_metadata({
        "model_type": "LinearRegression",
        "model_path": path,
    })

    return model



