"""Resolve one pinned Codex runtime for SDK, exec, sandbox and tier wrappers."""
from pathlib import Path
import os
import shutil
import sys


def bundled_codex_binary():
    try:
        from codex_cli_bin import bundled_codex_path
        return str(bundled_codex_path())
    except (ImportError, FileNotFoundError) as exc:
        raise RuntimeError(
            "The pinned Codex runtime is missing. Run: python3 tools/codex.py sync"
        ) from exc


def codex_binary():
    override = os.environ.get("CODEX_BIN")
    if override:
        # The SDK expects a path, whereas callers can supply a command on PATH.
        binary = shutil.which(override)
        if not binary:
            raise RuntimeError("CODEX_BIN is not executable: " + override)
        return str(Path(binary).absolute())
    return bundled_codex_binary()


if __name__ == "__main__":
    # Tier wrappers call this entry point without recursively using CODEX_BIN.
    binary = os.environ.get("ARC_CODEX_REAL_BIN") or bundled_codex_binary()
    os.execvp(binary, [binary, *sys.argv[1:]])
