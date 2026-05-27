import math
from dagster import asset
import polars as pl


@asset
def ml_ready_dataset(
    joined_feature_table
):

    df = (

        joined_feature_table

        .with_columns([

            # ---------------------------------------------
            # HOUR CYCLICAL
            # ---------------------------------------------

            (
                    (
                            2 * math.pi *
                            pl.col("datetime_hour").dt.hour()
                    ) / 24
            )
            .sin()
            .alias("hour_sin"),

            (
                    (
                            2 * math.pi *
                            pl.col("datetime_hour").dt.hour()
                    ) / 24
            )
            .cos()
            .alias("hour_cos"),

            # ---------------------------------------------
            # WEEKDAY CYCLICAL
            # ---------------------------------------------

            (
                    (
                            2 * math.pi *
                            pl.col("datetime_hour").dt.weekday()
                    ) / 7
            )
            .sin()
            .alias("weekday_sin"),

            (
                    (
                            2 * math.pi *
                            pl.col("datetime_hour").dt.weekday()
                    ) / 7
            )
            .cos()
            .alias("weekday_cos"),

            # ---------------------------------------------
            # MONTH CYCLICAL
            # ---------------------------------------------

            (
                    (
                            2 * math.pi *
                            pl.col("datetime_hour").dt.month()
                    ) / 12
            )
            .sin()
            .alias("month_sin"),

            (
                    (
                            2 * math.pi *
                            pl.col("datetime_hour").dt.month()
                    ) / 12
            )
            .cos()
            .alias("month_cos")
        ])

        .with_columns([

            (
                    pl.col("datetime_hour")
                    .dt.weekday() >= 5
            )
            .cast(pl.Int8)
            .alias("is_weekend")
        ])

        .with_columns([

            pl.when(
                pl.col("datetime_hour").dt.month().is_in([12,1,2])
            )
            .then(pl.lit("Winter"))

            .when(
                pl.col("datetime_hour").dt.month().is_in([3,4,5])
            )
            .then(pl.lit("Spring"))

            .when(
                pl.col("datetime_hour").dt.month().is_in([6,7,8])
            )
            .then(pl.lit("Summer"))

            .otherwise(pl.lit("Fall"))
            .alias("season")
        ])

        .drop([
            "date"
        ])
    )

    return df