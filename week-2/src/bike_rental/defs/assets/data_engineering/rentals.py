import polars as pl
from dagster import asset, MaterializeResult
import os
from polars import LazyFrame

"""Registered and direct rentals are concatenated then aggregated"""

@asset(required_resource_keys={"loader"}, group_name="preprocessing_rentals_to_hourly_rentals")
def direct_rentals_prepared(context) -> LazyFrame:
    """Prepare direct rentals data for hourly aggregation."""

    raw_direct_rentals = (
        context.resources.loader.load_csv(
            os.getenv("DIRECT_RENTALS_PATH"),
            engine="polars",
        )
    )

    result = raw_direct_rentals.with_columns(
        [
            pl.col("datetime").dt.truncate("1h").alias("datetime_hour"),
            pl.lit(1).alias("rental_type_direct"),
        ]
    ).select(["datetime_hour", "location_id", "user_id", "rental_type_direct"])

    preview = result.collect()


    context.add_output_metadata({
            "rows": preview.height,
            "unique_users": preview["user_id"].n_unique(),
            "start_hour": str(preview["datetime_hour"].min()),
            "end_hour": str(preview["datetime_hour"].max()),
    })

    return result


@asset(required_resource_keys={"loader"}, group_name="preprocessing_rentals_to_hourly_rentals")
def registered_rentals_prepared(context)-> LazyFrame:
    """Prepare registered rentals data for hourly aggregation."""

    raw_registered_rentals = (
        context.resources.loader.load_csv(
            os.getenv("REGISTERED_RENTALS_PATH"),
            engine="polars",
        )
    )

    result = raw_registered_rentals.with_columns(
        [
            pl.col("datetime").dt.truncate("1h").alias("datetime_hour"),
            pl.lit(0).alias("rental_type_direct"),
        ]
    ).select(["datetime_hour", "location_id", "user_id", "rental_type_direct"])

    preview = result.collect()

    context.add_output_metadata({
            "rows": preview.height,
            "unique_users": preview["user_id"].n_unique(),
            "start_hour": str(preview["datetime_hour"].min()),
            "end_hour": str(preview["datetime_hour"].max()),
    })

    return result

@asset(group_name="preprocessing_rentals_to_hourly_rentals")
def unified_rentals(context,direct_rentals_prepared, registered_rentals_prepared)-> LazyFrame:
    """Combine direct and registered rental datasets."""
    result = pl.concat([direct_rentals_prepared, registered_rentals_prepared])

    preview = result.collect()

    context.add_output_metadata({
            "rows": preview.height,
            "unique_users": preview["user_id"].n_unique(),
            "direct_rentals": (
                    preview["rental_type_direct"] == 1
            ).sum(),
            "registered_rentals": (
                    preview["rental_type_direct"] == 0
            ).sum(),
            "start_hour": str(preview["datetime_hour"].min()),
            "end_hour": str(preview["datetime_hour"].max()),
    })
    return result

@asset(group_name="preprocessing_rentals_to_hourly_rentals")
def hourly_rentals(context, unified_rentals)-> LazyFrame:
    """Aggregate hourly rental metrics for concatenated direct and registered rental datasets."""

    result = unified_rentals.group_by(["datetime_hour"]).agg(
        [
            pl.len().alias("rental_count"),
            (pl.col("rental_type_direct") == 1).sum().alias("direct_rentals"),
            (pl.col("rental_type_direct") == 0)
            .sum()
            .alias("registered_rentals"),
            pl.col("user_id").n_unique().alias("unique_users"),
        ]
    )

    preview = result.collect()

    context.add_output_metadata({
        "hourly_records": preview.height,
        "total_rentals": preview["rental_count"].sum(),
        "total_direct_rentals": preview["direct_rentals"].sum(),
        "total_registered_rentals": preview["registered_rentals"].sum(),
        "max_hourly_rentals": preview["rental_count"].max(),
        "start_hour": str(preview["datetime_hour"].min()),
    })

    return result


@asset(group_name="preprocessing_rentals_to_hourly_rentals")
def hourly_rentals_by_location(context, unified_rentals)-> LazyFrame:
    """Aggregate hourly rental metrics by location."""

    result = unified_rentals.group_by(["datetime_hour", "location_id"]).agg(
        [
            pl.len().alias("rental_count"),
            (pl.col("rental_type_direct") == 1).sum().alias("direct_rentals"),
            (pl.col("rental_type_direct") == 0)
            .sum()
            .alias("registered_rentals"),
            pl.col("user_id").n_unique().alias("unique_users"),
        ]
    )

    preview = result.collect()

    context.add_output_metadata({
        "hourly_records": preview.height,
        "total_rentals": preview["rental_count"].sum(),
        "total_direct_rentals": preview["direct_rentals"].sum(),
        "total_registered_rentals": preview["registered_rentals"].sum(),
        "max_hourly_rentals": preview["rental_count"].max(),
        "start_hour": str(preview["datetime_hour"].min()),
    })

    return result

