# bike_rental/defs/hooks/lakefs_hooks.py
from dagster import success_hook, HookContext


@success_hook(required_resource_keys={"lakefs_res"})
def commit_and_merge_lakefs_hook(context: HookContext):
    """Production hook to automate the lakeFS branch lifecycle.

    Triggers automatically only when an asset successfully materializes
    AND its I/O manager finishes writing to storage.
    """
    lakefs_res = context.resources.lakefs_res
    run_id = context.run_id
    branch_name = f"dagster-run-{run_id}"
    asset_key = context.asset_key.path[-1]

    context.log.info(f"Hook processing asset '{asset_key}' on branch '{branch_name}'")

    try:
        # 1. Check for uncommitted files that the I/O Manager just wrote
        commit_id = lakefs_res.commit_if_changed(
            branch_name=branch_name,
            message=f"Auto-commit: Asset '{asset_key}' materialized in run {run_id}."
        )

        if commit_id:
            context.log.info(f"Physical changes detected for '{asset_key}'. Commit ID: {commit_id}")

            # 2. Merge changes atomically back to dev branch
            context.log.info(f"Merging isolated branch '{branch_name}' into 'dev'...")
            lakefs_res.merge_branches(source=branch_name, destination="dev")
            context.log.info(f"Successfully merged '{asset_key}' updates into 'dev' branch!")
        else:
            context.log.info(f"No changes detected in storage for '{asset_key}'. Skipping commit/merge.")

    except Exception as e:
        context.log.error(f"lakeFS hook automation failed for asset '{asset_key}': {str(e)}")
        raise e

    finally:
        # 3. Always delete the short-lived branch to avoid repo cluttering
        context.log.info(f"Cleaning up ephemeral run branch: {branch_name}")
        lakefs_res.delete_branch(branch_name)