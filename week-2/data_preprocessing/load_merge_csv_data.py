import pandas as pd
from functools import reduce


# =========================================================
# LOAD CSV FILES
# =========================================================

def load_csv_files(file_paths):

    dataframes = {}

    for name, path in file_paths.items():

        df = pd.read_csv(
            path,
            low_memory=False
        )

        dataframes[name] = df

    return dataframes


# =========================================================
# CONVERT DATETIME COLUMNS
# =========================================================

def convert_datetime_to_date(
        df,
        datetime_column="datetime",
        new_column="date"
):
    """
    Convert datetime column into normalized datetime date.
    """

    # If date column already exists
    if new_column in df.columns:

        df[new_column] = pd.to_datetime(
            df[new_column],
            errors="coerce"
        ).dt.normalize()

    # If datetime column exists
    elif datetime_column in df.columns:

        df[datetime_column] = pd.to_datetime(
            df[datetime_column],
            errors="coerce"
        )

        # Keep efficient datetime64 dtype
        df[new_column] = df[datetime_column].dt.normalize()

    return df


# =========================================================
# OPTIMIZED MERGE
# =========================================================

def merge_dataframes_on_date(dataframes):
    """
    Merge multiple DataFrames efficiently on date.
    """

    # Optional:
    # sort by date before merging
    for i in range(len(dataframes)):
        dataframes[i] = dataframes[i].sort_values("date")

    # Sequential merge
    merged_df = reduce(
        lambda left, right: pd.merge(
            left,
            right,
            on="date",
            how="left",
            sort=False
        ),
        dataframes
    )

    return merged_df


def main():

    file_paths = {
        "direct_rentals": "../data/direct_pickup_bike_rentals.csv",
        "registered_rentals": "../data/registered_bike_rentals.csv",
        "weather": "../data/weather.csv",
        "holidays": "../data/holidays.csv"
    }

    dfs = load_csv_files(file_paths)

    # Explicit datetime columns
    datetime_columns = {
        "direct_rentals": "datetime",
        "registered_rentals": "datetime",
        "weather": "datetime"
    }

    # Process datetime columns
    for name, datetime_col in datetime_columns.items():

        dfs[name] = convert_datetime_to_date(
            dfs[name],
            datetime_column=datetime_col
        )

    # Process holidays date column
    dfs["holidays"] = convert_datetime_to_date(
        dfs["holidays"],
        new_column="date"
    )

    # Merge
    final_df = merge_dataframes_on_date([
        dfs["direct_rentals"],
        dfs["registered_rentals"],
        dfs["weather"],
        dfs["holidays"]
    ])

    # Save
    final_df.to_csv(
        "merged_output.csv",
        index=False
    )

    print(final_df.head())

    print("\nMerge completed successfully!")


if __name__ == "__main__":
    main()
