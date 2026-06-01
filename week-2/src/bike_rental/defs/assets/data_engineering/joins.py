import polars as pl
from dagster import asset
from polars import LazyFrame


@asset(group_name="hourly_joins")
def hourly_rentals_full_grid(
        context,
    hourly_time_grid,
    hourly_rentals,
):
    """ Generate full hourly dataset including zero-rental hours."""

    result = (
        hourly_time_grid.with_columns()
        .join(
            hourly_rentals,
            on="datetime_hour",
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


@asset(group_name="hourly_joins")
def joined_feature_table(
        context,
    hourly_rentals_full_grid, holiday_features, hourly_weather
) -> LazyFrame:
    """Join rental, weather, and holiday features into a single dataset."""

    weather_added_df = (
        hourly_rentals_full_grid.with_columns(
            pl.col("datetime_hour")
        )
        .join(hourly_weather, on="datetime_hour", how="left")
    )

    final_df = (
        weather_added_df.with_columns(
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
