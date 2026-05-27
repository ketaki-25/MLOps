from dagster import IOManager
import polars as pl


class ParquetIOManager(IOManager):

    def handle_output(self, context, obj):

        path = f"data/{context.asset_key.path[-1]}.parquet"

        obj.collect().write_parquet(path)

    def load_input(self, context):

        path = f"data/{context.asset_key.path[-1]}.parquet"

        return pl.scan_parquet(path)