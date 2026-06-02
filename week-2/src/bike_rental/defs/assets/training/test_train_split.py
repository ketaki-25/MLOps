import pandas as pd
from dagster import asset
from bike_rental.utils.splitting import time_based_split

# -----------------------------
# HOURLY DATASET
# -----------------------------

@asset(group_name="training_data", io_manager_key="pandas_parquet_io_manager")
def train_dataset_hourly(
    context,
    base_dataset_hourly: pd.DataFrame,
):
    """
    Creates the training split of the hourly dataset using a time-based split.

    This asset:
    - Receives the full engineered hourly dataset
    - Splits it chronologically (no shuffling)
    - Returns the training portion

    Metadata includes:
    - Dataset shape
    - Target rental statistics (mean, min, max)
    - Time range covered
    - Missing value count
    """

    train, _ = time_based_split(base_dataset_hourly)

    context.add_output_metadata(
        {
            "rows": len(train),
            "columns": train.shape[1],
            "split_ratio": 0.8,
            "target_total_rentals_mean": float(train["rental_count"].mean()),
            "target_total_rentals_min": int(train["rental_count"].min()),
            "target_total_rentals_max": int(train["rental_count"].max()),
            "start_datetime": str(train["datetime_hour"].min()),
            "end_datetime": str(train["datetime_hour"].max()),
            "missing_values": int(train.isna().sum().sum()),
        }
    )

    return train


@asset(group_name="training_data", io_manager_key="pandas_parquet_io_manager")
def test_dataset_hourly(
    context,
    base_dataset_hourly: pd.DataFrame,
):
    """
    Creates the test split of the hourly dataset using a time-based split.

    This asset:
    - Receives the full engineered hourly dataset
    - Splits it chronologically (no shuffling)
    - Returns the test portion

    Metadata includes:
    - Dataset shape
    - Target rental statistics (mean, min, max)
    - Time range covered
    - Missing value count
    """

    _, test = time_based_split(base_dataset_hourly)

    context.add_output_metadata(
        {
            "rows": len(test),
            "columns": test.shape[1],
            "split_ratio": 0.2,
            "target_total_rentals_mean": float(test["rental_count"].mean()),
            "target_total_rentals_min": int(test["rental_count"].min()),
            "target_total_rentals_max": int(test["rental_count"].max()),
            "start_datetime": str(test["datetime_hour"].min()),
            "end_datetime": str(test["datetime_hour"].max()),
            "missing_values": int(test.isna().sum().sum()),
        }
    )

    return test


# -----------------------------
# BY LOCATION DATASET
# -----------------------------

@asset(group_name="training_data", io_manager_key="pandas_parquet_io_manager")
def train_dataset_hourly_by_location(
    context,
    base_dataset_hourly_by_location: pd.DataFrame,
):
    """ Creates the training split of the hourly-by-location dataset using a time-based split.

    This asset:
    - Receives engineered hourly dataset grouped by location
    - Splits it chronologically (no shuffling)
    - Returns the training portion

    Metadata includes:
    - Dataset shape
    - Number of unique locations
    - Target rental statistics (mean, min, max)
    - Time range covered
    - Missing value count
    """

    train, _ = time_based_split(base_dataset_hourly_by_location)

    context.add_output_metadata(
        {
            "rows": len(train),
            "columns": train.shape[1],
            "locations": train["location_id"].nunique(),
            "target_total_rentals_mean": float(train["rental_count"].mean()),
            "target_total_rentals_min": int(train["rental_count"].min()),
            "target_total_rentals_max": int(train["rental_count"].max()),
            "start_datetime": str(train["datetime_hour"].min()),
            "end_datetime": str(train["datetime_hour"].max()),
            "missing_values": int(train.isna().sum().sum()),
        }
    )

    return train


@asset(group_name="training_data", io_manager_key="pandas_parquet_io_manager")
def test_dataset_hourly_by_location(
    context,
    base_dataset_hourly_by_location: pd.DataFrame,
):
    """ Creates the test split of the hourly-by-location dataset using a time-based split.

    This asset:
    - Receives engineered hourly dataset grouped by location
    - Splits it chronologically (no shuffling)
    - Returns the test portion

    Metadata includes:
    - Dataset shape
    - Number of unique locations
    - Target rental statistics (mean, min, max)
    - Time range covered
    - Missing value count
    """

    _, test = time_based_split(base_dataset_hourly_by_location)

    context.add_output_metadata(
        {
            "rows": len(test),
            "columns": test.shape[1],
            "locations": test["location_id"].nunique(),
            "target_total_rentals_mean": float(test["rental_count"].mean()),
            "target_total_rentals_min": int(test["rental_count"].min()),
            "target_total_rentals_max": int(test["rental_count"].max()),
            "start_datetime": str(test["datetime_hour"].min()),
            "end_datetime": str(test["datetime_hour"].max()),
            "missing_values": int(test.isna().sum().sum()),
        }
    )

    return test