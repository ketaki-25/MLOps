from dagster import asset
import polars as pl
from polars import LazyFrame

@asset(group_name="full_time_grids")
def hourly_time_grid(context) -> LazyFrame:
    """Create complete hourly grid from weather timestamps."""

    result = pl.LazyFrame(
        {
            "datetime_hour": pl.datetime_range(
                start=pl.datetime(2011, 1, 1, 0, 0),
                end=pl.datetime(2012, 12, 31, 23, 0),
                interval="1h",
                eager=True,
            )
        }
    )
    preview = result.collect()
    context.add_output_metadata({
        "hours_in_grid": preview.height,
        "start_hour": str(preview["datetime_hour"].min()),
        "end_hour": str(preview["datetime_hour"].max()),
        "no_of_columns": preview.width,
    })

    return result

@asset(group_name="full_time_grids")
def full_grid_hourly_by_location(context)-> LazyFrame:
    """Generate a full calendar grid by all 21 locations hourly."""

    time_grid = pl.datetime_range(
        start=pl.datetime(2011, 1, 1, 0, 0),
        end=pl.datetime(2012, 12, 31, 23, 0),
        interval="1h",
        eager=True,
    )

    locations = pl.Series("location_id", list(range(21)))

    result = (
        pl.DataFrame({"datetime_hour": time_grid})
        .join(pl.DataFrame({"location_id": locations}), how="cross")
        .lazy()
    )

    preview = result.collect()

    context.add_output_metadata(
        {
            "rows": preview.height,  # should be 368424
            "num_locations": 21,
            "hours_per_location": int(len(time_grid)),
            "start_hour": str(preview["datetime_hour"].min()),
            "end_hour": str(preview["datetime_hour"].max()),
        }
    )

    return result

