from dagster import asset


@asset
def merged_bike_data(
    aggregated_direct_rentals,
    aggregated_registered_rentals,
    aggregated_weather,
    processed_holidays
):

    merged = (
        aggregated_direct_rentals

        .join(
            aggregated_registered_rentals,
            on="date",
            how="left"
        )

        .join(
            aggregated_weather,
            on="date",
            how="left"
        )

        .join(
            processed_holidays,
            on="date",
            how="left"
        )
    )

    return merged