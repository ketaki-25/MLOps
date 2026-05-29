from dagster import Definitions

from bike_rental.defs.assets.feature_engineering import ml_ready_dataset
from bike_rental.defs.assets.holidays import holiday_features
from bike_rental.defs.assets.joins import joined_feature_table
from bike_rental.defs.assets.raw_data import (
    raw_direct_rentals,
    raw_holidays,
    raw_registered_rentals,
    raw_weather,
)
from bike_rental.defs.assets.rentals import (
    direct_rentals_prepared,
    hourly_rentals_by_location,
    registered_rentals_prepared,
    unified_rentals,
)
from bike_rental.defs.assets.weather import hourly_weather
from bike_rental.defs.io_managers.parquet_io_manager import ParquetIOManager

""" dagster asset definitions and IO configuration
for the bike rental pipeline """

defs = Definitions(
    assets=[
        raw_direct_rentals,
        raw_registered_rentals,
        raw_weather,
        raw_holidays,
        direct_rentals_prepared,
        registered_rentals_prepared,
        unified_rentals,
        hourly_rentals_by_location,
        hourly_weather,
        holiday_features,
        joined_feature_table,
        ml_ready_dataset,
    ],
    resources={"io_manager": ParquetIOManager()},
)
