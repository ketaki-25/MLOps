import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    split_ratio: float = 0.8,
):
    df = df.sort_values(
        "datetime_hour"
    ).reset_index(drop=True)

    split_idx = int(
        len(df) * split_ratio
    )

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    return train, test

