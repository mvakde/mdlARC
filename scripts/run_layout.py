"""Branch-scoped, Git-ignored experiment storage shared by local launchers."""

import os
from pathlib import Path
import subprocess
from urllib.parse import quote


def branch_name(root: Path) -> str:
    name = os.environ.get("MDLARC_RUN_BRANCH")
    if name is None:
        result = subprocess.run(["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
                                cwd=root, capture_output=True, text=True, check=False)
        name = result.stdout.strip() if result.returncode == 0 else ""
    if not name or name in {".", ".."} or any(ord(char) < 32 for char in name):
        raise ValueError("Set MDLARC_RUN_BRANCH to the run's branch when Git HEAD is detached or unavailable")
    return name


def branch_folder(name: str) -> str:
    # Keep one directory per branch; encode slashes and percent signs without collisions.
    if not name or name in {".", ".."} or any(ord(char) < 32 for char in name):
        raise ValueError("Invalid run branch")
    return quote(name, safe="-_.")


def results_root(root: Path) -> Path:
    return root / "results" / branch_folder(branch_name(root))


def runs_root(root: Path) -> Path:
    # Colab's isolated source tree retains its historical archive member paths.
    # Only the Mac's persistent cache needs the branch prefix.
    if os.environ.get("MDLARC_REMOTE") == "1":
        return root / "runs"
    return results_root(root) / "runs"
