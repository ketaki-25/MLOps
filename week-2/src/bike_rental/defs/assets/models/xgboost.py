import os
import joblib
import numpy as np
import pandas as pd
from dagster import asset, AssetExecutionContext
from xgboost import XGBRegressor


@asset(group_name="models")
def xgboost_model_hourly(
    context: AssetExecutionContext,
    X_train_processed_hourly: pd.DataFrame,
    y_train_processed_hourly: pd.Series,
):
    """
    Train an XGBoost regression model for hourly rental prediction
    and persist it as a .pkl file.
    """

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(X_train_processed_hourly, y_train_processed_hourly)

    path = "data/output_data/xgboost_model_hourly.pkl"
    joblib.dump(model, path)

    context.add_output_metadata({
        "model_path": path,
        "n_features": X_train_processed_hourly.shape[1],
        "n_rows": X_train_processed_hourly.shape[0],
    })

    return model

def train_xgb(X_train, y_train):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)