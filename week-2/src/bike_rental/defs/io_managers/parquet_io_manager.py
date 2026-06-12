import polars as pl
from pathlib import Path
import pandas as pd
from dagster import IOManager, InputContext, OutputContext

from bike_rental.defs.resources.lakefs_resource import LakeFSResource

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

        return pl.read_parquet(path)

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
        return pd.read_parquet(path)'''


class PandasParquetIOManager(IOManager):
    def _get_lakefs_path_and_options(self, context, lakefs_res: LakeFSResource) -> tuple[str, dict]:
        """Helper to construct the lakeFS path and storage credentials."""

        run_id = context.run_id
        branch_name = f"dagster-run-{run_id}"

        asset_path = "/".join(context.asset_key.path)

        lakefs_res.ensure_branch(branch_name)

        s3_uri = f"lakefs://{lakefs_res.repository}/{branch_name}/{asset_path}.parquet"

        return s3_uri, lakefs_res.storage_options()

    def handle_output(self, context: OutputContext, obj: pd.DataFrame):
        path, storage_options = self._get_lakefs_path_and_options(context, lakefs_res=LakeFSResource())
        context.log.info(f"Writing asset output to lakeFS: {path}")

        # Stream directly via s3fs down to your local container instance
        obj.to_parquet(path, index=False, storage_options=storage_options)

    def load_input(self, context: InputContext):
        path, storage_options = self._get_lakefs_path_and_options(context, lakefs_res=LakeFSResource())
        context.log.info(f"Reading asset input from lakeFS: {path}")

        return pd.read_parquet(path, storage_options=storage_options)


class PolarsParquetIOManager(IOManager):

    def _get_lakefs_path_and_options(self, context, lakefs_res: LakeFSResource) -> tuple[str, dict]:

        run_id = context.run_id
        branch_name = f"dagster-run-{run_id}"

        asset_path = "/".join(context.asset_key.path)
        s3_uri = f"lakefs://{lakefs_res.repository}/{branch_name}/{asset_path}.parquet"

        return s3_uri, lakefs_res.storage_options()

    def handle_output(self, context: OutputContext, obj: pl.DataFrame):
        path, storage_options = self._get_lakefs_path_and_options(context, lakefs_res=LakeFSResource())
        context.log.info(f"Writing Polars asset output to lakeFS: {path}")

        obj.write_parquet(path, storage_options=storage_options)

    def load_input(self, context: InputContext):
        path, storage_options = self._get_lakefs_path_and_options(context, lakefs_res=LakeFSResource())
        context.log.info(f"Reading Polars asset input from lakeFS: {path}")

        return pl.read_parquet(path, storage_options=storage_options)