from bike_rental.defs.assets.data_engineering.math_functions_for_feature_engineering import cyclic_time_features, season_features, weekend_features
from dagster import asset


@asset(io_manager_key="parquet_io_manager", group_name="final_preprocessed_datasets", tags={"domain": "preprocessing"})
def base_dataset_hourly(context, joined_feature_table):
    """Create an ML-ready dataset with engineered time-based features."""
    cyclic_time_df = cyclic_time_features(joined_feature_table)
    season_df = season_features(cyclic_time_df)
    weekend_df = weekend_features(season_df)

    preview = weekend_df.collect()

    context.add_output_metadata({
        "row_count": preview.height,
            "column_count": preview.width,
            "min_rental_count": int(preview["rental_count"].min()),
            "max_rental_count": int(preview["rental_count"].max()),
            "zero_rental_hours": int(
                (preview["rental_count"] == 0).sum()
            ),
            "min_datetime_hour": str(
                preview["datetime_hour"].min()
            ),
            "max_datetime_hour": str(
                preview["datetime_hour"].max()
            ),
    })

    return weekend_df

@asset(io_manager_key="parquet_io_manager", group_name="final_preprocessed_datasets")
def base_dataset_hourly_by_location(context, joined_feature_table_by_location):
    """Create an ML-ready dataset with engineered time-based features."""
    cyclic_time_df = cyclic_time_features(joined_feature_table_by_location)
    season_df = season_features(cyclic_time_df)
    weekend_df = weekend_features(season_df)

    preview = weekend_df.collect()

    context.add_output_metadata({
        "row_count": preview.height,
        "column_count": preview.width,
        "min_rental_count": int(preview["rental_count"].min()),
        "max_rental_count": int(preview["rental_count"].max()),
        "zero_rental_hours": int(
            (preview["rental_count"] == 0).sum()
        ),
        "min_datetime_hour": str(
            preview["datetime_hour"].min()
        ),
        "max_datetime_hour": str(
            preview["datetime_hour"].max()
        ),
    })

    return weekend_df