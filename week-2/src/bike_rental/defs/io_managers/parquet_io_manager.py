import polars as pl
from dagster import IOManager
from pathlib import Path


class ParquetIOManager(IOManager):
    """Custom IO manager for reading and writing Parquet assets."""

    def handle_output(self, context, obj):
        """Write asset output to a Parquet file."""
        path = f"data/output_data/{context.asset_key.path[-1]}.parquet"

        #obj.collect(streaming=True).write_parquet(path)

        df = obj.collect(streaming=True)
        df.write_parquet(path)

        context.add_output_metadata(
            {
                "rows": len(df),
                "columns": len(df.columns),
                "file_path": path,
                "file_size_mb": round(
                    Path(path).stat().st_size / 1024 ** 2,
                    2,
                ),
            }
        )

    def load_input(self, context):
        """Load asset input from a Parquet file."""
        path = f"data/input_data/{context.asset_key.path[-1]}.parquet"

        return pl.scan_parquet(path)
