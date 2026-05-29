import polars as pl
from dagster import asset


@asset
def holiday_features(raw_holidays):
    """Create holiday indicator features for each date."""
    return raw_holidays.with_columns(pl.lit(1).alias("is_holiday")).select(
        ["date", "holiday", "is_holiday"]
    )
