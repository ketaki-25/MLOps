from dagster import Definitions

from bike_rental.defs.assets.raw_data import *
from bike_rental.defs.assets.rentals import *
from bike_rental.defs.assets.weather import *
from bike_rental.defs.assets.holidays import *
from bike_rental.defs.assets.joins import *
from bike_rental.defs.assets.feature_engineering import *

from bike_rental.defs.io_managers.parquet_io_manager import (
    ParquetIOManager
)


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
        ml_ready_dataset
    ],

    resources={
        "io_manager": ParquetIOManager()
    }
)