from pathlib import Path

"""File paths for all raw bike rental project datasets."""

INPUT_DATA_DIR = Path("data/input_data")

REGISTERED_RENTALS_PATH = INPUT_DATA_DIR / "registered_bike_rentals.csv"
DIRECT_RENTALS_PATH = INPUT_DATA_DIR / "direct_pickup_bike_rentals.csv"
WEATHER_PATH = INPUT_DATA_DIR / "weather.csv"
HOLIDAYS_PATH = INPUT_DATA_DIR / "holidays.csv"
