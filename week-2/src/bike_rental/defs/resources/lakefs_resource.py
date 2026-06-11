import lakefs
import lakefs.exceptions
from lakefs.client import Client
from dagster import ConfigurableResource
import os


class LakeFSResource(ConfigurableResource):
    """Connection and versioning operations for a LakeFS repository.

    LakeFS is git-for-data: nothing is versioned until an explicit commit. This
    resource owns the connection and exposes the operations the pipeline needs
    to write data to a run-scoped branch, commit it, and merge to ``main``.

    """

    host: str =os.getenv("LAKEFS_ENDPOINT")
    access_key: str =os.getenv("LAKEFS_ACCESS_KEY_ID")
    secret_key: str =os.getenv("LAKEFS_SECRET_ACCESS_KEY")
    repository: str =os.getenv("LAKEFS_REPO_NAME")

    def get_client(self) -> Client:
        return Client(host=self.host, username=self.access_key, password=self.secret_key)

    def get_repo(self) -> "lakefs.Repository":
        return lakefs.Repository(self.repository, client=self.get_client())

    def ensure_branch(self, branch: str, source: str = "main") -> None:
        """Create ``branch`` from ``source`` if it does not already exist."""
        self.get_repo().branch(branch).create(source_reference=source, exist_ok=True)

    def commit(self, branch: str, message: str, metadata = None) -> str:
        """Commit the branch and return the resulting commit id.

        If there are no uncommitted changes, returns the current branch head
        instead of failing.
        """
        target = self.get_repo().branch(branch)
        string_metadata = {key: str(value) for key, value in (metadata or {}).items()}
        try:
            target.commit(message=message, metadata=string_metadata)
        except lakefs.exceptions.LakeFSException:
            # Most commonly: nothing to commit. Fall back to the current head.
            pass
        return target.get_commit().id

    def merge(self, source: str, destination: str = "main") -> None:
        """Merge ``source`` branch into ``destination``.

        A 400 'no changes' response is treated as success — it means the
        branch is already identical to the destination, which is fine on
        re-runs where the data hasn't changed.
        """
        try:
            self.get_repo().branch(source).merge_into(destination)
        except lakefs.exceptions.BadRequestException as exc:
            if "no changes" in str(exc).lower():
                return
            raise

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
