from pathlib import Path

"""File paths for all raw bike rental project datasets."""

DATA_DIR = Path("data")

REGISTERED_RENTALS_PATH = DATA_DIR / "registered_bike_rentals.csv"
DIRECT_RENTALS_PATH = DATA_DIR / "direct_pickup_bike_rentals.csv"
WEATHER_PATH = DATA_DIR / "weather.csv"
HOLIDAYS_PATH = DATA_DIR / "holidays.csv"
