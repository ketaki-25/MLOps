from dagster import asset
import polars as pl


@asset
def hourly_weather(
    raw_weather
):

    return (

        raw_weather

        .with_columns(
            pl.col("datetime")
            .dt.truncate("1h")
            .alias("datetime_hour")
        )

        .group_by("datetime_hour")

        .agg([

            pl.col("temperature_c")
            .mean(),

            pl.col("perceived_temperature_c")
            .mean(),

            pl.col("humidity")
            .mean(),

            pl.col("windspeed_kmh")
            .mean(),

            pl.col("conditions")
            .first()
        ])
    )