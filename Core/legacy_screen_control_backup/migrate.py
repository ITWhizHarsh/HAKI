"""
migrate.py — Legacy screen control archival script.

Moves Core/core/automation/mac_controller.py and
Core/core/automation/screen_agent.py into this backup directory
and writes a README.md recording the archival metadata.

Run from the project root (HAKI/):
    python -m Core.legacy_screen_control_backup.migrate

Or import and call archive_legacy() programmatically.
"""

import logging
import shutil
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths are relative to the project root (HAKI/).
BACKUP_DIR = Path("Core/legacy_screen_control_backup")
SOURCES = [
    Path("Core/core/automation/mac_controller.py"),
    Path("Core/core/automation/screen_agent.py"),
]


def archive_legacy() -> None:
    """Move legacy automation modules to the backup directory.

    Creates the backup directory if it does not exist.
    Raises FileExistsError (and logs an error) if any destination file
    already exists, rather than silently overwriting an existing backup.
    """
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    for src in SOURCES:
        dest = BACKUP_DIR / src.name
        if dest.exists():
            logger.error(
                "Backup already exists at %s; aborting to prevent overwrite.", dest
            )
            raise FileExistsError(
                f"Backup already exists at {dest}; aborting to prevent overwrite."
            )

    for src in SOURCES:
        dest = BACKUP_DIR / src.name
        logger.info("Moving %s → %s", src, dest)
        shutil.move(str(src), str(dest))

    _write_readme(BACKUP_DIR)
    logger.info("Legacy archival complete. README written to %s", BACKUP_DIR / "README.md")


def _write_readme(directory: Path) -> None:
    """Write archival metadata to README.md inside *directory*."""
    readme = directory / "README.md"
    readme.write_text(
        f"# Legacy Screen Control Backup\n\n"
        f"Archived: {date.today().isoformat()}\n\n"
        f"## Original paths\n\n"
        f"- `Core/core/automation/mac_controller.py`\n"
        f"- `Core/core/automation/screen_agent.py`\n\n"
        f"## Reason\n\n"
        f"Replaced by Gemini-Sidecar Architecture (SidecarAgentLoop). "
        f"Files preserved for recovery without a git revert.\n"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    archive_legacy()
