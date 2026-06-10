import pandas as pd

from dagster import (
    IOManager,
    io_manager,
)


class CsvIOManager(IOManager):

    def handle_output(
        self,
        context,
        obj,
    ):

        path = (
            f"data/output_data/{context.asset_key.path[-1]}.csv"
        )

        obj.to_csv(
            path,
            index=False,
        )

        context.log.info(
            f"Saved asset to {path}"
        )

    def load_input(
        self,
        context,
    ):

        upstream_asset = (
            context.upstream_output.asset_key.path[-1]
        )

        path = f"data/output_data/{upstream_asset}.csv"

        context.log.info(
            f"Loading asset from {path}"
        )

        return pd.read_csv(path)
