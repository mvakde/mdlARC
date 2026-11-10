"""Reconstruct only the saved source SVG preview from its preserved task JSON.

Hash callers still compare the reconstructed bytes with the historical pin.
This never replaces or reformats source evidence and never persists a preview.
"""
import json
from pathlib import Path


def rebuild_source_svg(path):
    path = Path(path)
    if path.name != "source_task.svg" or path.parent.name != "workspace" or "llm-inputs" not in path.parts:
        raise ValueError("Not a supported rebuildable source preview: " + str(path))
    source = path.with_name("task.json")
    if source.is_symlink():
        raise ValueError("Preview source must not be a symlink")
    from task_variations.improve.workspace_render import source_svg
    return source_svg(json.loads(source.read_bytes())).encode("utf-8")


def snapshot_bytes(path):
    path = Path(path)
    try:
        return path.read_bytes()
    except FileNotFoundError:
        if path.name != "source_task.svg":
            raise
        return rebuild_source_svg(path)
