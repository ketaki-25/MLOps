import polars as pl
from dagster import IOManager
from pathlib import Path
import pandas as pd

'''class ParquetIOManager(IOManager):
    """Custom IO manager for reading and writing Parquet assets."""

    def handle_output(self, context, obj):
        """Write asset output to a Parquet file."""
        path = f"data/output_data/{context.asset_key.path[-1]}.parquet"

        #obj.collect(streaming=True).write_parquet(path)
        obj.to_parquet(path, index=False)

        #df = obj.collect(streaming=True)
        #df.write_parquet(path)

        context.add_output_metadata(
            {
                "rows": len(obj),
                "columns": len(obj.columns),
                "file_path": path,
                "file_size_mb": round(
                    Path(path).stat().st_size / 1024 ** 2,
                    2,
                ),
            }
        )

    def load_input(self, context):
        """Load asset input from a Parquet file."""
        path = f"data/output_data/{context.asset_key.path[-1]}.parquet"

        return pl.read_parquet(path)'''

class PolarsParquetIOManager(IOManager):

    def handle_output(self, context, obj):
        path = f"data/output_data/{context.asset_key.path[-1]}.parquet"
        obj.to_parquet(path, index=False)

        context.add_output_metadata(
            {
                "rows": len(obj),
                "columns": len(obj.columns),
                "file_path": path,
                "file_size_mb": round(
                    Path(path).stat().st_size / 1024 ** 2,
                    2,
                ),
            }
        )

    def load_input(self, context):
        path = f"data/input_data/{context.asset_key.path[-1]}.parquet"
        return pl.scan_parquet(path)


class PandasParquetIOManager(IOManager):

    def handle_output(self, context, obj):
        path = f"data/output_data/{context.asset_key.path[-1]}.parquet"
        obj.to_parquet(path, index=False)

    def load_input(self, context):
        path = f"data/output_data/{context.asset_key.path[-1]}.parquet"
        return pd.read_parquet(path)