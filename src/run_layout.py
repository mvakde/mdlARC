"""Where launch_run.py keeps a run: <repo>/runs/<run_name>.

Set MDLARC_RUNS_DIR to keep runs somewhere else (for example a larger disk).
"""

import os
from pathlib import Path


def runs_root(root: Path) -> Path:
    override = os.environ.get("MDLARC_RUNS_DIR")
    return Path(override).expanduser() if override else Path(root) / "runs"
