import math

import polars as pl


def cyclic_time_features(df):

    result = df.with_columns(
        [
            # ---------------------------------------------
            # HOUR CYCLICAL
            # ---------------------------------------------
            ((2 * math.pi * pl.col("datetime_hour").dt.hour()) / 24)
            .sin()
            .alias("hour_sin"),
            ((2 * math.pi * pl.col("datetime_hour").dt.hour()) / 24)
            .cos()
            .alias("hour_cos"),

            # ---------------------------------------------
            # WEEKDAY CYCLICAL
            # ---------------------------------------------
            ((2 * math.pi * pl.col("datetime_hour").dt.weekday()) / 7)
            .sin()
            .alias("weekday_sin"),
            ((2 * math.pi * pl.col("datetime_hour").dt.weekday()) / 7)
            .cos()
            .alias("weekday_cos"),

            # ---------------------------------------------
            # MONTH CYCLICAL
            # ---------------------------------------------
            ((2 * math.pi * pl.col("datetime_hour").dt.month()) / 12)
            .sin()
            .alias("month_sin"),
            ((2 * math.pi * pl.col("datetime_hour").dt.month()) / 12)
            .cos()
            .alias("month_cos"),
        ]
    ).drop(["date"])

    return result

def season_features(df):

    result = df.with_columns(
        [
            pl.when(pl.col("datetime_hour").dt.month().is_in([12, 1, 2]))
            .then(pl.lit("Winter"))
            .when(pl.col("datetime_hour").dt.month().is_in([3, 4, 5]))
            .then(pl.lit("Spring"))
            .when(pl.col("datetime_hour").dt.month().is_in([6, 7, 8]))
            .then(pl.lit("Summer"))
            .otherwise(pl.lit("Fall"))
            .alias("season"),
        ]
    )

    return result

def weekend_features(df):

    result = df.with_columns(
        [
            (pl.col("datetime_hour").dt.weekday() >= 5)
            .cast(pl.Int8)
            .alias("is_weekend"),
        ]
    )

    return result

def lag_features(df):
    result = df.sort("datetime_hour").with_columns(
        [
            # same hour previous day
            pl.col("rental_count").shift(24).alias("total_lag_24"),

            # same hour previous week
            pl.col("rental_count").shift(24 * 7).alias("total_lag_168"),

            # same hour previous day only direct rentals
            pl.col("direct_rentals").shift(24).alias("direct_lag_24"),

            # same hour previous week only direct rentals
            pl.col("direct_rentals").shift(24 * 7).alias("direct_lag_168"),

            # same hour previous week only registered rentals
            pl.col("registered_rentals").shift(24).alias("registered_lag_24"),

            # same hour previous week only registered rentals
            pl.col("registered_rentals").shift(24 * 7).alias("registered_lag_168"),
        ]
    )

    return result

def handle_lag_nans(df):
    return df.drop_nulls(subset=["total_lag_24", "total_lag_168", "direct_lag_24", "direct_lag_168", "registered_lag_24", "registered_lag_168"])