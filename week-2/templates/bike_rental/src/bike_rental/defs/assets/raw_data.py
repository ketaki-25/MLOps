from dagster import asset
import polars as pl


@asset
def direct_rentals():

    return pl.read_csv(
        "data/direct_pickup_bike_rentals.csv"
    )


@asset
def registered_rentals():

    return pl.read_csv(
        "data/registered_bike_rentals.csv"
    )


@asset
def weather():

    return pl.read_csv(
        "data/weather.csv"
    )


@asset
def holidays():

    return pl.read_csv(
        "data/holidays.csv"
    )