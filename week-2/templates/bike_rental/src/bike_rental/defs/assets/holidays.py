from dagster import asset
import polars as pl


@asset
def holiday_features(
    raw_holidays
):

    return (

        raw_holidays

        .with_columns(
            pl.lit(1)
            .alias("is_holiday")
        )

        .select([
            "date",
            "holiday",
            "is_holiday"
        ])
    )