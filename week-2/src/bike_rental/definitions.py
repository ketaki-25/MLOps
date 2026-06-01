from dagster import Definitions
from bike_rental.defs.resources.data_loader import DataLoader
from bike_rental.defs.assets.data_engineering.feature_engineering import base_dataset_hourly, base_dataset_hourly_by_location
from bike_rental.defs.assets.data_engineering.holidays import holiday_features
from bike_rental.defs.assets.data_engineering.joins import joined_feature_table, hourly_rentals_full_grid
from bike_rental.defs.assets.data_engineering.rentals import (
    direct_rentals_prepared,
    hourly_rentals,
    registered_rentals_prepared,
    unified_rentals,
    hourly_rentals_by_location,
)
from bike_rental.defs.assets.data_engineering.hourly_grids import hourly_time_grid, full_grid_hourly_by_location
from bike_rental.defs.assets.data_engineering.weather import hourly_weather
from bike_rental.defs.assets.data_engineering.joins_by_location import hourly_rentals_by_location_full_grid
from bike_rental.defs.io_managers.parquet_io_manager import ParquetIOManager
from bike_rental.defs.assets.data_engineering.joins_by_location import joined_feature_table_by_location

""" dagster asset definitions and IO configuration
for the bike rental pipeline """

defs = Definitions(
    assets=[
        full_grid_hourly_by_location,
        hourly_time_grid,
        hourly_weather,
        holiday_features,
        direct_rentals_prepared,
        registered_rentals_prepared,
        unified_rentals,
        hourly_rentals,
        hourly_rentals_full_grid,
        joined_feature_table,
        joined_feature_table_by_location,
        base_dataset_hourly,
        hourly_rentals_by_location,
        hourly_rentals_by_location_full_grid,
        base_dataset_hourly_by_location
    ],
    resources={
            "parquet_io_manager": ParquetIOManager(),
            "loader": DataLoader(),
        },
)
