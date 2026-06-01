import polars as pl
from dagster import asset
import os

@asset(required_resource_keys={"loader"}, group_name="holiday")
def holiday_features(context):
    """Create holiday indicator features for each date."""
    raw_holidays = (
        context.resources.loader.load_csv(
            os.getenv("HOLIDAYS_PATH"),
            engine="polars",
        )
    )

    result= raw_holidays.with_columns(pl.lit(1).alias("is_holiday")).select(
        ["date", "holiday", "is_holiday"]
    )

    preview = result.collect()

    context.add_output_metadata({
        "rows": preview.height,
        "unique_holidays": preview["holiday"].n_unique(),
        "holiday_dates": preview["is_holiday"].sum(),
        "start_date": str(preview["date"].min()),
        "end_date": str(preview["date"].max()),
    })

    return result
