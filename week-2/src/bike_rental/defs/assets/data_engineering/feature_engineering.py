from bike_rental.defs.assets.data_engineering.math_functions_for_feature_engineering import cyclic_time_features, season_features, weekend_features, lag_features, handle_lag_nans
from dagster import asset

from bike_rental.defs.resources import lakefs_resource
from bike_rental.defs.resources.lakefs_resource import LakeFSResource
import mlflow
from bike_rental.defs.resources.lakefs_hook import commit_and_merge_lakefs_hook

def generate_ml_metadata(df):
    return {
        "row_count": df.shape[0],
        "column_count": df.shape[1],
        "min_rental_count": int(df["rental_count"].min()),
        "max_rental_count": int(df["rental_count"].max()),
        "zero_rental_hours": int((df["rental_count"] == 0).sum()),
        "min_datetime_hour": str(df["datetime_hour"].min()),
        "max_datetime_hour": str(df["datetime_hour"].max()),
    }

@asset(io_manager_key="pandas_parquet_io_manager", group_name="final_preprocessed_datasets", tags={"domain": "preprocessing"})
def base_dataset_hourly(context, joined_feature_table):
    """Create an ML-ready dataset with engineered time-based features."""
    cyclic_time_df = cyclic_time_features(joined_feature_table)
    lag_df = lag_features(cyclic_time_df)
    cleaned_lag_df = handle_lag_nans(lag_df)
    season_df = season_features(cleaned_lag_df)
    weekend_df = weekend_features(season_df)

    df = weekend_df.collect() if hasattr(weekend_df, "collect") else weekend_df
    print(type(df))

    df = df.to_pandas() if hasattr(df, "to_pandas") else df
    print(type(df))

    metadata = generate_ml_metadata(df)

    context.add_output_metadata(metadata)

    return df

@asset(io_manager_key="pandas_parquet_io_manager", group_name="final_preprocessed_datasets", required_resource_keys={"lakefs_res"})
def base_dataset_hourly_by_location(context, joined_feature_table_by_location):
    """Create an ML-ready dataset with engineered time-based features."""
    cyclic_time_df = cyclic_time_features(joined_feature_table_by_location)
    lag_df = lag_features(cyclic_time_df)
    cleaned_lag_df = handle_lag_nans(lag_df)
    season_df = season_features(cleaned_lag_df)
    weekend_df = weekend_features(season_df)

    df = weekend_df.collect() if hasattr(weekend_df, "collect") else weekend_df

    df = df.to_pandas() if hasattr(df, "to_pandas") else df

    metadata = generate_ml_metadata(df)


    context.add_output_metadata(metadata)

    return df

#TODO:
# convert the strings like "conditions" column to meaningful data