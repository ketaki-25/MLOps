from dagster import asset
import polars as pl


@asset
def raw_direct_rentals():

    return (
        pl.scan_csv(
            "data/direct_pickup_bike_rentals.csv"
        )
        .with_columns(
            pl.col("datetime")
            .str.to_datetime()
        )
    )

@asset
def raw_registered_rentals():

    return (
        pl.scan_csv(
            "data/registered_bike_rentals.csv"
        )
        .with_columns(
            pl.col("datetime")
            .str.to_datetime()
        )
    )

@asset
def raw_weather():

    return (
        pl.scan_csv(
            "data/weather.csv"
        )
        .with_columns(
            pl.col("datetime")
            .str.to_datetime()
        )
    )

@asset
def raw_holidays():

    return (
        pl.scan_csv(
            "data/holidays.csv"
        )
        .with_columns(
            pl.col("date")
            .str.to_date()
        )
    )