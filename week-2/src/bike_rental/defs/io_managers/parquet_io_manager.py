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
    def _get_write_path_and_options(self, context: OutputContext, lakefs_res: LakeFSResource) -> tuple[str, str, dict]:
        """Construct ephemeral branch write path for outputs."""
        run_id = context.run_id
        branch_name = f"dagster-run-{run_id}"
        asset_path = "/".join(context.asset_key.path)

        lakefs_res.ensure_branch(branch=branch_name, source="dev")

        lakefs_uri = f"lakefs://{lakefs_res.repository}/{branch_name}/{asset_path}.parquet"
        return lakefs_uri, branch_name, lakefs_res.storage_options()

    def _get_read_path_and_options(self, context: InputContext, lakefs_res: LakeFSResource) -> tuple[str, dict]:
        """Construct stable dev branch read path for inputs."""

        asset_path = "/".join(context.asset_key.path)

        lakefs_uri = f"lakefs://{lakefs_res.repository}/dev/{asset_path}.parquet"
        return lakefs_uri, lakefs_res.storage_options()

    def handle_output(self, context: OutputContext, obj: pd.DataFrame):
        lakefs_res = LakeFSResource()
        path, branch_name, storage_options = self._get_write_path_and_options(context, lakefs_res)

        context.log.info(f"Writing asset output to lakeFS: {path}")
        obj.to_parquet(path, index=False, storage_options=storage_options)

        # Data is now written — safe to commit, merge, and clean up
        try:
            metadata = context.definition_metadata or {}

            commit_id = lakefs_res.commit(
                branch=branch_name,
                message=f"Auto committing: {'/'.join(context.asset_key.path)} written for run {context.run_id}",
                metadata=metadata,
            )

            if commit_id:
                context.log.info(f"Committed to lakeFS. Commit ID: {commit_id}")
                context.log.info(f"Merging {branch_name} into dev...")
                lakefs_res.merge_branches(source=branch_name, destination="dev")
                context.log.info("Merge to dev successful!")
            else:
                context.log.info("No changes detected. Skipping commit and merge.")

        finally:
            try:
                lakefs_res.delete_branch(branch_name)
                context.log.info(f"Cleaned up ephemeral branch: {branch_name}")
            except Exception as e:
                # Non-fatal — log and move on
                context.log.warning(f"Could not delete branch {branch_name}: {e}")

    def load_input(self, context: InputContext):
        lakefs_res = LakeFSResource()

        path, storage_options = self._get_read_path_and_options(context, lakefs_res)

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