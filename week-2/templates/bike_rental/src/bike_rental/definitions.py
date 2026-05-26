from dagster import Definitions, definitions

from bike_rental.defs.assets.hello import hello
from bike_rental.defs.assets.raw_data import direct_rentals, registered_rentals, weather, holidays
from bike_rental.defs.assets.merged_data import merged_bike_data
from bike_rental.defs.assets.processed_data import processed_direct_rentals, processed_registered_rentals, processed_weather, processed_holidays
from bike_rental.defs.io_managers.parquet_io_manager import ParquetIOManager
from bike_rental.defs.assets.aggregated_data import *


@definitions
def defs() -> Definitions:
    return Definitions(
        assets=[
            direct_rentals,
            registered_rentals,
            weather,
            holidays,
            processed_direct_rentals,
            processed_registered_rentals,
            processed_weather,
            processed_holidays,
            merged_bike_data,
            aggregated_weather,
            aggregated_direct_rentals,
            aggregated_registered_rentals,
        ],
        resources={
            "io_manager": ParquetIOManager()
        }
    )
