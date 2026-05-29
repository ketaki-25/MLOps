from pathlib import Path

import polars as pl
from dagster import asset
from polars import LazyFrame

from bike_rental.defs.resources.paths import (
    DIRECT_RENTALS_PATH,
    HOLIDAYS_PATH,
    REGISTERED_RENTALS_PATH,
    WEATHER_PATH,
)


def load_csv(path: Path) -> LazyFrame:
    """Load a CSV file as a Polars LazyFrame."""
    return pl.scan_csv(path)


@asset
def raw_direct_rentals():
    """Load raw direct rentals data."""
    df = load_csv(DIRECT_RENTALS_PATH)
    return df.with_columns(pl.col("datetime").str.to_datetime())


@asset
def raw_registered_rentals():
    """Load raw registered rentals data."""
    df = load_csv(REGISTERED_RENTALS_PATH)
    return df.with_columns(pl.col("datetime").str.to_datetime())


@asset
def raw_weather():
    """Load raw weather data."""
    df = load_csv(WEATHER_PATH)
    return df.with_columns(pl.col("datetime").str.to_datetime())


@asset
def raw_holidays():
    """Load raw holiday data."""
    df = load_csv(HOLIDAYS_PATH)
    return df.with_columns(pl.col("date").str.to_date())
