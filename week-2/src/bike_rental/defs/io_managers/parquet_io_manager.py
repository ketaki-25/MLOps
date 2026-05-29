import polars as pl
from dagster import IOManager


class ParquetIOManager(IOManager):
    """Custom IO manager for reading and writing Parquet assets."""

    def handle_output(self, context, obj):
        """Write asset output to a Parquet file."""
        path = f"data/output_data/{context.asset_key.path[-1]}.parquet"

        obj.collect(streaming=True).write_parquet(path)

    def load_input(self, context):
        """Load asset input from a Parquet file."""
        path = f"data/input_data/{context.asset_key.path[-1]}.parquet"

        return pl.scan_parquet(path)
