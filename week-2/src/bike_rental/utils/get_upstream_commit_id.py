# bike_rental/utils/dagster_utils.py

from dagster import DagsterInstance, AssetKey, EventRecordsFilter, DagsterEventType

def get_upstream_commit_id(context, asset_key: AssetKey):
    """
    Fetch the lakeFS commit ID stored in the most recent
    materialization metadata of an upstream asset.
    """
    instance: DagsterInstance = context.instance

    records = instance.get_event_records(
        EventRecordsFilter(
            event_type=DagsterEventType.ASSET_MATERIALIZATION,
            asset_key=asset_key,
        ),
        limit=1,
    )

    if not records:
        return None

    materialization = records[0].event_log_entry.dagster_event.step_materialization_data.materialization
    metadata = materialization.metadata

    entry = metadata.get("lakefs_commit_id")
    return entry.value if entry else None