import polars as pl
from dagster import asset


@asset
def joined_feature_table(
    hourly_rentals_by_location, hourly_weather, holiday_features
):
    """Join rental, weather, and holiday features into a single dataset."""
    rentals_weather = hourly_rentals_by_location.join(
        hourly_weather, on="datetime_hour", how="left"
    )

    final_df = (
        rentals_weather.with_columns(
            pl.col("datetime_hour").dt.date().alias("date")
        )
        .join(holiday_features, on="date", how="left")
        .with_columns(pl.col("is_holiday").fill_null(0))
    )

    return final_df
