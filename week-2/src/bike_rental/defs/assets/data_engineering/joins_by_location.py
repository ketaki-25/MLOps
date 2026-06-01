import polars as pl
from dagster import asset
from polars import LazyFrame

@asset(group_name="joins_by_location")
def hourly_rentals_by_location_full_grid(
        context,
    full_grid_hourly_by_location,
    hourly_rentals_by_location,
):
    """ Generate full hourly dataset including zero-rental hours."""

    result = (
        full_grid_hourly_by_location.with_columns()
        .join(
            hourly_rentals_by_location,
            on=["datetime_hour", "location_id"],
            how="left",
        )
        .with_columns(
            [
                pl.col("rental_count").fill_null(0),
                pl.col("direct_rentals").fill_null(0),
                pl.col("registered_rentals").fill_null(0),
                pl.col("unique_users").fill_null(0),
            ]
        )
    )

    preview = result.collect()

    context.add_output_metadata({
        "hours_in_grid": preview.height,
        "total_rentals": int(preview["rental_count"].sum()),
        "total_direct_rentals": int(
            preview["direct_rentals"].sum()
        ),
        "total_registered_rentals": int(
            preview["registered_rentals"].sum()
        ),
        "hours_with_zero_rentals": int(
            (preview["rental_count"] == 0).sum()
        ),
        "start_hour": str(
            preview["datetime_hour"].min()
        ),
        "end_hour": str(
            preview["datetime_hour"].max()
        ),
        "no_of_columns": preview.width,
    })

    return result

@asset(group_name="joins_by_location")
def joined_feature_table_by_location(
        context,
    hourly_rentals_by_location_full_grid, holiday_features, hourly_weather
) -> LazyFrame:
    """Join rental, weather, and holiday features into a single dataset."""

    joined_weather_rental_full_grid = (
        hourly_rentals_by_location_full_grid.with_columns(
            pl.col("datetime_hour")
        )
        .join(hourly_weather, on="datetime_hour", how="left")
    )

    final_df = (
        joined_weather_rental_full_grid.with_columns(
            pl.col("datetime_hour").dt.date().alias("date")
        )
        .join(holiday_features, on="date", how="left")
        .with_columns(pl.col("is_holiday").fill_null(0))
    )
    preview = final_df.collect()

    context.add_output_metadata({
        "hours_in_grid": preview.height,
        "total_rentals": int(preview["rental_count"].sum()),
        "total_direct_rentals": int(preview["direct_rentals"].sum()),
        "total_registered_rentals": int(preview["registered_rentals"].sum()),
        "hours_with_zero_rentals": int((preview["rental_count"] == 0).sum()),
        "no_of_columns": preview.width,
    })
    return final_df

