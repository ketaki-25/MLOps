from dagster import asset
import polars as pl

@asset
def direct_rentals_prepared(
    raw_direct_rentals
):

    return (
        raw_direct_rentals

        .with_columns([
            pl.col("datetime")
            .dt.truncate("1h")
            .alias("datetime_hour"),

            pl.lit(1)
            .alias("rental_type_direct")
        ])

        .select([
            "datetime_hour",
            "location_id",
            "user_id",
            "rental_type_direct"
        ])
    )

@asset
def registered_rentals_prepared(
    raw_registered_rentals
):

    return (
        raw_registered_rentals

        .with_columns([
            pl.col("datetime")
            .dt.truncate("1h")
            .alias("datetime_hour"),

            pl.lit(0)
            .alias("rental_type_direct")
        ])

        .select([
            "datetime_hour",
            "location_id",
            "user_id",
            "rental_type_direct"
        ])
    )


@asset
def unified_rentals(
    direct_rentals_prepared,
    registered_rentals_prepared
):

    return pl.concat([
        direct_rentals_prepared,
        registered_rentals_prepared
    ])

@asset
def hourly_rentals_by_location(
    unified_rentals
):

    return (

        unified_rentals

        .group_by([
            "datetime_hour",
            "location_id"
        ])

        .agg([

            pl.len()
            .alias("rental_count"),

            (
                pl.col("rental_type_direct") == 1
            )
            .sum()
            .alias("direct_rentals"),

            (
                pl.col("rental_type_direct") == 0
            )
            .sum()
            .alias("registered_rentals"),

            pl.col("user_id")
            .n_unique()
            .alias("unique_users")
        ])
    )