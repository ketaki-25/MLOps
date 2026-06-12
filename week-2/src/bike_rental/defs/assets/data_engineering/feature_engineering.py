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
def base_dataset_hourly(context, joined_feature_table, lakefs_res : LakeFSResource):
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

    # --- lakeFS Branching Setup ---
    # 2. Ensure isolated lakeFS workspace branch exists before I/O Manager writes to it
    run_id = context.run_id
    branch_name = f"dagster-run-{run_id}"
    #lakefs_res = context.resources.lakefs_res
    lakefs_res.ensure_branch(branch=branch_name, source="dev")

    metadata = generate_ml_metadata(df)

    # --- Commit changes to lakeFS ---

    # Note: If your I/O manager writes *after* the asset returns,
    # look into moving this commit logic into a Dagster @success_hook!
    commit_id = lakefs_res.commit(
        branch=branch_name,
        message=f"Auto committing: Feature engineering dataset created for run {run_id}",
        metadata=metadata
    )
    if commit_id:
        context.log.info(f"Changes detected. Commering to lakeFS. Commit ID: {commit_id}")

        # 2. Merge changes atomically back to dev
        context.log.info(f"Merging {branch_name} into dev branch...")
        lakefs_res.merge_branches(source=branch_name, destination="dev")
        context.log.info("Merge to dev successful!")
    else:
        context.log.info("No data modifications detected. Skipping commit and merge steps.")

    # 3. Always clean up your short-lived run branch to keep the repo uncluttered
    lakefs_res.delete_branch(branch_name)
    context.log.info(f"Cleaned up ephemeral branch: {branch_name}")

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

    # --- lakeFS Branching Setup ---
    run_id = context.run_id
    branch_name = f"dagster-run-{run_id}"

    lakefs_res = context.resources.lakefs_res

    # Ensure our isolated experiment/run branch exists
    lakefs_res.ensure_branch(branch=branch_name, source="main")

    metadata = generate_ml_metadata(df)
    # --- Commit changes to lakeFS ---
    # Note: If your I/O manager writes *after* the asset returns,
    # look into moving this commit logic into a Dagster @success_hook!
    commit_id = lakefs_res.commit(
        branch=branch_name,
        message=f"Feature engineering dataset created for run {run_id}",
        metadata=metadata
    )
    context.log.info(f"Committed data to lakeFS branch {branch_name}. Commit ID: {commit_id}")

    context.add_output_metadata(metadata)

    return df

#TODO:
# convert the strings like "conditions" column to meaningful data