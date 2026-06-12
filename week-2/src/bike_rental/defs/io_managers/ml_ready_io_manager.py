import polars as pl
from pathlib import Path
import pandas as pd
from dagster import IOManager, InputContext, OutputContext

from bike_rental.defs.resources.lakefs_resource import LakeFSResource

class MLReadyDatasetIOManager(IOManager):

    def _get_write_paths_and_options(
        self, context: OutputContext, lakefs_res: LakeFSResource
    ) -> tuple[dict[str, str], str, dict]:
        """Construct one write path per dict key under the same asset namespace."""
        run_id = context.run_id
        branch_name = f"dagster-run-{run_id}"
        asset_path = "/".join(context.asset_key.path)

        lakefs_res.ensure_branch(branch=branch_name, source="dev")

        # e.g. lakefs://repo/branch/hourly_total_linear_regression_ml_ready_dataset/X_train.parquet
        paths = {
            key: f"lakefs://{lakefs_res.repository}/{branch_name}/{asset_path}/{key}.parquet"
            for key in ["X_train", "y_train", "X_test", "y_test", "feature_names"]
        }

        return paths, branch_name, lakefs_res.storage_options()

    def _get_read_paths_and_options(
        self, context: InputContext, lakefs_res: LakeFSResource
    ) -> tuple[dict[str, str], dict]:
        """Construct read paths from stable dev branch."""
        asset_path = "/".join(context.asset_key.path)

        paths = {
            key: f"lakefs://{lakefs_res.repository}/dev/{asset_path}/{key}.parquet"
            for key in ["X_train", "y_train", "X_test", "y_test", "feature_names"]
        }

        return paths, lakefs_res.storage_options()

    def handle_output(self, context: OutputContext, obj: dict):
        lakefs_res = LakeFSResource()
        paths, branch_name, storage_options = self._get_write_paths_and_options(context, lakefs_res)

        # -----------------------------------
        # Write each key as its own parquet
        # -----------------------------------
        for key, df in obj.items():
            path = paths[key]
            context.log.info(f"Writing {key} to lakeFS: {path}")

            # feature_names is a list — wrap it as a single-column DataFrame
            if isinstance(df, list):
                df = pd.DataFrame({key: df})
            elif isinstance(df, pd.Series):  # ← add this
                df = df.to_frame(name=key)

            df.to_parquet(path, index=False, storage_options=storage_options)

        # -----------------------------------
        # Commit → Merge → Cleanup
        # -----------------------------------
        commit_id = None

        try:
            metadata = context.definition_metadata or {}
            commit_id = lakefs_res.commit(
                branch=branch_name,
                message=f"Auto committing: {'/'.join(context.asset_key.path)} written for run {context.run_id}",
                metadata=metadata,
            )
            context.log.info(f"Committed to lakeFS. Commit ID: {commit_id}")

        except Exception as e:
            context.log.warning(f"Commit step failed or had no changes: {e}")

        if commit_id:
            try:
                context.log.info(f"Merging {branch_name} into dev...")
                lakefs_res.merge_branches(source=branch_name, destination="dev")
                context.log.info("Merge to dev successful!")
            except Exception as e:
                context.log.error(f"Merge failed — branch {branch_name} preserved for inspection: {e}")
                raise

            context.add_output_metadata({
                "lakefs_commit_id": commit_id,
                "lakefs_branch": "dev",
                "lakefs_repository": lakefs_res.repository,
                "lakefs_dataset_uri": f"lakefs://{lakefs_res.repository}/dev/{asset_path}",
            })
            
        else:
            context.log.info("No commit created. Skipping merge.")

        try:
            lakefs_res.delete_branch(branch_name)
            context.log.info(f"Cleaned up ephemeral branch: {branch_name}")
        except Exception as e:
            context.log.warning(f"Could not delete branch {branch_name}: {e}")

    def load_input(self, context: InputContext) -> dict:
        lakefs_res = LakeFSResource()
        paths, storage_options = self._get_read_paths_and_options(context, lakefs_res)

        context.log.info(f"Reading ML ready dataset from lakeFS dev branch...")

        result = {}
        for key, path in paths.items():
            df = pd.read_parquet(path, storage_options=storage_options)

            # Unwrap feature_names back to a plain list
            if key == "feature_names":
                result[key] = df[key].tolist()
            elif key in ("y_train", "y_test"):  # ← add this
                result[key] = df[key]  # returns a Series
            else:
                result[key] = df

        return result