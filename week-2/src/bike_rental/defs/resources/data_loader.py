import polars as pl
import pandas as pd


class DataLoader:

    def load_csv(self, path, engine="polars", **kwargs):

        print(f"load_csv called with path={path!r}")

        if engine == "polars":
            lf = pl.scan_csv(path, **kwargs)

            schema = lf.collect_schema()

            transformations = []

            if "date" in schema:
                transformations.append(
                    pl.col("date").str.to_date(strict=False)
                )

            if "datetime" in schema:
                transformations.append(
                    pl.col("datetime").str.to_datetime(strict=False)
                )

            if transformations:
                lf = lf.with_columns(transformations)

            return lf

        elif engine == "pandas":

            df = pd.read_csv(path, **kwargs)

            if "date" in df.columns:
                df["date"] = pd.to_datetime(
                    df["date"],
                    errors="coerce",
                ).dt.date

            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(
                    df["datetime"],
                    errors="coerce",
                )

            return df

        else:
                raise ValueError(
                    "engine must be 'polars' or 'pandas'"
                )

    def load_parquet(self, path, engine="polars", **kwargs):

        if engine == "polars":
            return pl.read_parquet(path, **kwargs)

        elif engine == "pandas":
            return pd.read_parquet(path, **kwargs)

        else:
            raise ValueError(
                "engine must be 'polars' or 'pandas'"
            )