import os
import lakefs
import lakefs.exceptions
from lakefs.client import Client
from dagster import ConfigurableResource

class LakeFSResource(ConfigurableResource):
    """Connection and versioning operations for a LakeFS repository.

    LakeFS is git-for-data: nothing is versioned until an explicit commit. This
    resource owns the connection and exposes the operations the pipeline needs
    to write data to a run-scoped branch, commit it, and merge to ``main``.
    """

    # Using default factory patterns to safely fallback to env variables if not explicitly provided in defs
    host: str = os.getenv("LAKEFS_ENDPOINT")
    access_key: str = os.getenv("LAKEFS_ACCESS_KEY_ID", "")
    secret_key: str = os.getenv("LAKEFS_SECRET_ACCESS_KEY", "")
    repository: str = os.getenv("LAKEFS_REPO_NAME", "bike-rental")

    def get_client(self) -> Client:
        return Client(host=self.host, username=self.access_key, password=self.secret_key)

    def get_repo(self) -> lakefs.Repository:
        return lakefs.Repository(self.repository, client=self.get_client())

    def ensure_branch(self, branch: str, source: str = "main") -> None:
        """Create ``branch`` from ``source`` if it does not already exist."""
        self.get_repo().branch(branch).create(source_reference=source, exist_ok=True)

    def commit(self, branch: str, message: str, metadata = None) -> str:
        """Commit the branch and return the resulting commit id.

        If there are no uncommitted changes, returns the current branch head instead.
        """
        target = self.get_repo().branch(branch)
        string_metadata = {key: str(value) for key, value in (metadata or {}).items()}
        try:
            # Performs atomic commit operation on lakeFS
            commit_obj = target.commit(message=message, metadata=string_metadata)
            return commit_obj.id
        except lakefs.exceptions.LakeFSException:
            # Fall back to the current head commit if there are no modifications
            return target.get_commit().id

    def commit_if_changed(self, branch_name: str, message: str, metadata=None):
        """Commit changes on the branch ONLY if uncommitted changes exist.

        Returns the commit ID if a commit happened, or None if no changes were found.
        """
        branch = self.get_repo().branch(branch_name)

        # Check if there are uncommitted files on this isolated run branch
        uncommitted_changes = list(branch.uncommitted())

        if not uncommitted_changes:
            return None  # No changes, skip tracking

        string_metadata = {key: str(value) for key, value in (metadata or {}).items()}
        commit_obj = branch.commit(message=message, metadata=string_metadata)
        return commit_obj.id

    def merge_branches(self, source: str, destination: str) -> None:
        """Merge source branch into destination branch."""
        try:
            # Source branch merges ITSELF into the destination branch object
            source_branch = self.get_repo().branch(source)
            dest_branch = self.get_repo().branch(destination)
            source_branch.merge_into(dest_branch)
        except lakefs.exceptions.BadRequestException as exc:
            if "no changes" in str(exc).lower():
                return
            raise

    def delete_branch(self, branch_name: str) -> None:
        """Safely delete an ephemeral run branch."""
        try:
            self.get_repo().branch(branch_name).delete()
        except lakefs.exceptions.LakeFSException:
            pass

    def object_uri(self, branch: str, path: str) -> str:
        """Build a ``lakefs://`` URI for an object on a branch."""
        return f"local://{self.repository}/{branch}/{path}"

    def storage_options(self) -> dict[str, str]:
        """Return fsspec storage options for lakefs-spec (pandas read/write)."""
        return {
            "host": self.host,
            "username": self.access_key,
            "password": self.secret_key,
        }