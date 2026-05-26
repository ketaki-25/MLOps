from dagster import asset
import polars as pl


@asset
def processed_direct_rentals(direct_rentals):

    return (
        direct_rentals
        .with_columns(
            pl.col("datetime")
            .str.to_datetime()
            .dt.date()
            .alias("date")
        )
    )


@asset
def processed_registered_rentals(registered_rentals):

    return (
        registered_rentals
        .with_columns(
            pl.col("datetime")
            .str.to_datetime()
            .dt.date()
            .alias("date")
        )
    )


@asset
def processed_weather(weather):

    return (
        weather
        .with_columns(
            pl.col("datetime")
            .str.to_datetime()
            .dt.date()
            .alias("date")
        )
    )


@asset
def processed_holidays(holidays):

    return (
        holidays
        .with_columns(
            pl.col("date")
            .str.to_date()
        )
    )