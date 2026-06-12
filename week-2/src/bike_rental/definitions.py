from dagster import Definitions
from bike_rental.defs.config.experiment_config import ExperimentConfigResource
from bike_rental.defs.resources.data_loader import DataLoader
from bike_rental.defs.assets.data_engineering.feature_engineering import base_dataset_hourly, base_dataset_hourly_by_location
from bike_rental.defs.assets.data_engineering.holidays import holiday_features
from bike_rental.defs.assets.data_engineering.joins import joined_feature_table, hourly_rentals_full_grid
from bike_rental.defs.assets.data_engineering.rentals import direct_rentals_prepared, hourly_rentals,registered_rentals_prepared,unified_rentals, hourly_rentals_by_location
from bike_rental.defs.assets.data_engineering.hourly_grids import hourly_time_grid, full_grid_hourly_by_location
from bike_rental.defs.assets.data_engineering.weather import hourly_weather
from bike_rental.defs.assets.data_engineering.joins_by_location import hourly_rentals_by_location_full_grid
from bike_rental.defs.io_managers.parquet_io_manager import PolarsParquetIOManager, PandasParquetIOManager
from bike_rental.defs.io_managers.csv_io_manager import CsvIOManager
from bike_rental.defs.assets.data_engineering.joins_by_location import joined_feature_table_by_location
from bike_rental.defs.assets.training_dataset_generation.test_train_split import train_dataset_hourly, train_dataset_hourly_by_location, test_dataset_hourly, test_dataset_hourly_by_location
from bike_rental.defs.assets.training_dataset_generation.generated_experiments import EXPERIMENT_ASSETS
from bike_rental.defs.resources.lakefs_resource import LakeFSResource

""" dagster asset definitions and IO configuration
for the bike rental pipeline """


lakefs = LakeFSResource()


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
        base_dataset_hourly_by_location,
        train_dataset_hourly,
        train_dataset_hourly_by_location,
        test_dataset_hourly,
        test_dataset_hourly_by_location,
        *EXPERIMENT_ASSETS,
    ],
    resources={
            "lakefs_res": LakeFSResource(),
            "polars_parquet_io_manager": PolarsParquetIOManager(),
            "pandas_parquet_io_manager": PandasParquetIOManager(),
            "csv_io_manager": CsvIOManager(),
            "loader": DataLoader(),
            "experiment_config":
                    ExperimentConfigResource(
                        yaml_path=
                        "src/bike_rental/defs/config/experiments.yml"
                    ),
        },
    jobs=[],
)


#TODO:
# can do asset checks for validation and testing