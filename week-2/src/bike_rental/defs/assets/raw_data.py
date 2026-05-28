from dagster import asset
import polars as pl
from polars import LazyFrame

from bike_rental.defs.resources.paths import *

def load_csv(path: Path) -> LazyFrame:
    return (
        pl.scan_csv(path)
    )

@asset
def raw_direct_rentals():

    df = load_csv(DIRECT_RENTALS_PATH)
    return (
        df
        .with_columns(
            pl.col("datetime")
            .str.to_datetime()
        )
    )

@asset
def raw_registered_rentals():

    df = load_csv(REGISTERED_RENTALS_PATH)
    return (
        df
        .with_columns(
            pl.col("datetime")
            .str.to_datetime()
        )
    )

@asset
def raw_weather():

    df = load_csv(WEATHER_PATH)
    return (
        df
        .with_columns(
            pl.col("datetime")
            .str.to_datetime()
        )
    )

@asset
def raw_holidays():

    df = load_csv(HOLIDAYS_PATH)
    return (
        df
        .with_columns(
            pl.col("date")
            .str.to_date()
        )
    )