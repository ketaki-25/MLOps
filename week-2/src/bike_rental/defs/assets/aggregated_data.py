from dagster import asset
import polars as pl


@asset
def aggregated_direct_rentals(
    processed_direct_rentals
):

    return (
        processed_direct_rentals
        .group_by("date")
        .agg([
            pl.col("rentals")
            .sum()
            .alias("total_direct_rentals")
        ])
    )

@asset
def aggregated_registered_rentals(
    processed_registered_rentals
):

    return (
        processed_registered_rentals
        .group_by("date")
        .agg([
            pl.col("rentals")
            .sum()
            .alias("total_registered_rentals")
        ])
    )

@asset
def aggregated_weather(
    processed_weather
):

    return (
        processed_weather
        .group_by("date")
        .agg([
            pl.col("temperature").mean(),
            pl.col("humidity").mean(),
            pl.col("windspeed").mean()
        ])
    )