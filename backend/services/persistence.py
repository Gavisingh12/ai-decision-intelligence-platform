"""Optional free-tier persistence via Hugging Face dataset repositories."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from backend.core.config import get_settings


logger = logging.getLogger(__name__)


class HubPersistenceService:
    """Mirror runtime data to a Hugging Face dataset repo when configured."""

    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def enabled(self) -> bool:
        """Return whether remote persistence has been configured."""

        return bool(self.settings.hf_persist_repo_id and self.settings.hf_token)

    def restore_runtime_data(self) -> bool:
        """Restore runtime files from the configured Hugging Face dataset repo."""

        if not self.enabled:
            return False

        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import RepositoryNotFoundError
        except ImportError:
            logger.warning("huggingface_hub is not installed; skipping remote restore.")
            return False

        try:
            snapshot_path = Path(
                snapshot_download(
                    repo_id=self.settings.hf_persist_repo_id,
                    repo_type="dataset",
                    token=self.settings.hf_token,
                    allow_patterns="runtime_data/*",
                )
            )
        except RepositoryNotFoundError:
            logger.info("HF dataset repo %s not found yet; starting with empty local runtime.", self.settings.hf_persist_repo_id)
            return False
        except Exception as exc:
            logger.warning("Unable to restore runtime data from Hugging Face: %s", exc)
            return False

        source_root = snapshot_path / "runtime_data"
        if not source_root.exists():
            return False

        for relative_path in self._runtime_relative_paths():
            source = source_root / relative_path
            destination = self.settings.data_dir / relative_path
            if not source.exists():
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)

        logger.info("Restored runtime data from Hugging Face dataset repo.")
        return True

    def sync_runtime_data(self, *, reason: str) -> bool:
        """Upload runtime files to the configured Hugging Face dataset repo."""

        if not self.enabled:
            return False

        try:
            from huggingface_hub import HfApi
        except ImportError:
            logger.warning("huggingface_hub is not installed; skipping remote sync.")
            return False

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_root = Path(temp_dir) / "runtime_data"
            runtime_root.mkdir(parents=True, exist_ok=True)

            for relative_path in self._runtime_relative_paths():
                source = self.settings.data_dir / relative_path
                destination = runtime_root / relative_path
                if not source.exists():
                    continue

                destination.parent.mkdir(parents=True, exist_ok=True)
                if source.is_dir():
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, destination)

            try:
                api = HfApi(token=self.settings.hf_token)
                api.create_repo(
                    repo_id=self.settings.hf_persist_repo_id,
                    repo_type="dataset",
                    private=self.settings.hf_persist_repo_private,
                    exist_ok=True,
                )
                api.upload_folder(
                    repo_id=self.settings.hf_persist_repo_id,
                    repo_type="dataset",
                    folder_path=str(runtime_root),
                    path_in_repo="runtime_data",
                    commit_message=f"Sync runtime data: {reason}",
                )
                logger.info("Synced runtime data to Hugging Face dataset repo.")
                return True
            except Exception as exc:
                logger.warning("Unable to sync runtime data to Hugging Face: %s", exc)
                return False

    def _runtime_relative_paths(self) -> list[Path]:
        """Return runtime-relative paths that should be backed up."""

        paths = [
            Path("uploads"),
            Path("models"),
            Path("artifacts"),
            Path("vector_store"),
        ]
        sqlite_path = self.settings.sqlite_file_path
        if sqlite_path:
            try:
                paths.append(sqlite_path.relative_to(self.settings.data_dir))
            except ValueError:
                logger.warning("SQLite file %s is outside the data directory; skipping hub sync for the database file.", sqlite_path)
        return paths


hub_persistence_service = HubPersistenceService()
