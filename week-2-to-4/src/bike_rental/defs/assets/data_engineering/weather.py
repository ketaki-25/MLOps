import polars as pl
from dagster import asset
import os
from polars import LazyFrame

#TODO:
# Setup data validation and testing

@asset(required_resource_keys={"loader"},group_name="hourly_weather")
def hourly_weather(context) -> LazyFrame:
    """Aggregate hourly weather metrics and conditions."""

    raw_weather = context.resources.loader.load_csv(
        os.getenv("WEATHER_PATH"),
        engine="polars",
    )

    #TODO:
    # handle missing and incorrect data. Throw away those rows
    df = raw_weather
    df = df.filter(
        pl.col("humidity").is_not_null()
        & (pl.col("humidity") > 0)
    )

    df = df.filter(
        pl.col("temperature_c").is_not_null()
        & pl.col("perceived_temperature_c").is_not_null()
    )

    # temperature consistency rule:
    # abs(temp - perceived_temp) <= 10
    df = df.filter(
        (pl.col("temperature_c") - pl.col("perceived_temperature_c"))
        .abs()
        <= 10
    )

    df = df.filter(pl.col("datetime").is_not_null())

    result = (
        df.with_columns(
            pl.col("datetime").dt.truncate("1h").alias("datetime_hour")
        )
        .group_by("datetime_hour")
        .agg(
            [
                pl.col("temperature_c").mean(),
                pl.col("perceived_temperature_c").mean(),
                pl.col("humidity").mean(),
                pl.col("windspeed_kmh").mean(),
                pl.col("conditions").first(),
            ]
        )
    )

    preview = result.collect()
    context.add_output_metadata({
        "rows": preview.height,
        "start_hour": str(preview["datetime_hour"].min()),
        "end_hour": str(preview["datetime_hour"].max()),
        "avg_temperature": round(
            preview["temperature_c"].mean(), 2
        ),
        "avg_humidity": round(
            preview["humidity"].mean(), 2
        ),
        "weather_conditions": preview["conditions"].n_unique(),
    })

    return result
