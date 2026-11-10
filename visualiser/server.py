#!/usr/bin/env python3
"""Zero-dependency, lazy-loading web server for the mdlARC task visualiser."""

from __future__ import annotations

import argparse
import gzip
import json
import mimetypes
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


APP_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = APP_DIR.parent / "assets_tmp"
STATIC_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.js": "app.js",
    "/styles.css": "styles.css",
}
PAIR_KEYS = ("train", "test")


@dataclass(frozen=True)
class TaskRef:
    dataset_id: str
    key: str
    name: str
    group: str
    path: Path

    def public(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "group": self.group,
        }


class Catalog:
    """Filename-only index plus a small LRU of parsed task files."""

    def __init__(self, source: Path, cache_size: int = 4) -> None:
        self.source = source.resolve()
        self.cache_size = max(1, cache_size)
        self.datasets: list[dict[str, Any]] = []
        self.lookup: dict[tuple[str, str], TaskRef] = {}
        self._cache: OrderedDict[Path, list[dict[str, Any]]] = OrderedDict()
        self._cache_lock = threading.RLock()
        self._scan()

    def _scan(self) -> None:
        if not self.source.is_dir():
            raise FileNotFoundError(f"Dataset source does not exist: {self.source}")

        for dataset_dir in sorted(
            (path for path in self.source.iterdir() if path.is_dir()),
            key=lambda path: path.name.casefold(),
        ):
            scan_root = self._select_scan_root(dataset_dir)
            paths = sorted(scan_root.rglob("*.json"), key=lambda path: path.as_posix())
            tasks: list[TaskRef] = []

            for path in paths:
                relative = path.relative_to(scan_root)
                group_path = relative.parent
                group = "" if group_path == Path(".") else group_path.as_posix()
                key = relative.with_suffix("").as_posix()
                ref = TaskRef(
                    dataset_id=dataset_dir.name,
                    key=key,
                    name=path.stem,
                    group=group,
                    path=path,
                )
                tasks.append(ref)
                self.lookup[(ref.dataset_id, ref.key)] = ref

            if tasks:
                self.datasets.append(
                    {
                        "id": dataset_dir.name,
                        "name": dataset_dir.name,
                        "taskCount": len(tasks),
                        "tasks": [task.public() for task in tasks],
                    }
                )

    @staticmethod
    def _select_scan_root(dataset_dir: Path) -> Path:
        # The raw/tasks preference avoids indexing aggregate copies such as
        # arc1_train_both.json while still supporting direct NVARC directories.
        for preferred in ("raw", "tasks"):
            candidate = dataset_dir / preferred
            if candidate.is_dir():
                return candidate
        return dataset_dir

    def index_payload(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "datasets": self.datasets,
        }

    def page(self, dataset_id: str, task_key: str, page: int, page_size: int = 12) -> dict[str, Any]:
        ref = self.lookup.get((dataset_id, task_key))
        if ref is None:
            raise KeyError(f"Unknown task: {dataset_id}/{task_key}")

        pairs = self._load_pairs(ref.path)
        pair_count = len(pairs)
        page_count = max(1, (pair_count + page_size - 1) // page_size)
        safe_page = min(max(page, 0), page_count - 1)
        start = safe_page * page_size
        end = min(start + page_size, pair_count)

        return {
            "dataset": dataset_id,
            "task": ref.public(),
            "pairCount": pair_count,
            "page": safe_page,
            "pageCount": page_count,
            "pageSize": page_size,
            "start": start,
            "end": end,
            "pairs": pairs[start:end],
        }

    def _load_pairs(self, path: Path) -> list[dict[str, Any]]:
        with self._cache_lock:
            cached = self._cache.pop(path, None)
            if cached is not None:
                self._cache[path] = cached
                return cached

        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        pairs = self._normalise_pairs(raw, path)

        with self._cache_lock:
            # Another request may have populated this while this thread parsed.
            existing = self._cache.pop(path, None)
            self._cache[path] = existing if existing is not None else pairs
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return self._cache[path]

    @staticmethod
    def _normalise_pairs(raw: Any, path: Path) -> list[dict[str, Any]]:
        tagged: list[tuple[str, Any]] = []
        if isinstance(raw, list):
            tagged.extend(("generated", pair) for pair in raw)
        elif isinstance(raw, dict):
            for split in PAIR_KEYS:
                values = raw.get(split, [])
                if values is None:
                    continue
                if not isinstance(values, list):
                    raise TypeError(f"{path}: '{split}' must be a list")
                tagged.extend((split, pair) for pair in values)
        else:
            raise TypeError(f"{path}: expected an object or list")

        pairs: list[dict[str, Any]] = []
        for split, pair in tagged:
            if not isinstance(pair, dict) or "input" not in pair:
                raise TypeError(f"{path}: every pair must contain an input grid")
            pairs.append(
                {
                    "split": split,
                    "input": pair["input"],
                    "output": pair.get("output"),
                }
            )
        return pairs


class VisualiserHandler(BaseHTTPRequestHandler):
    catalog: Catalog
    server_version = "mdlARCVisualiser/1.0"

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/index":
                self._send_json(self.catalog.index_payload())
                return
            if parsed.path == "/api/task":
                self._serve_task(parse_qs(parsed.query))
                return
            if parsed.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            if parsed.path in STATIC_FILES:
                self._serve_static(STATIC_FILES[parsed.path])
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
        except KeyError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc).strip("'"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
        except OSError as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _serve_task(self, query: dict[str, list[str]]) -> None:
        dataset_id = query.get("dataset", [""])[0]
        task_key = query.get("task", [""])[0]
        try:
            page = int(query.get("page", ["0"])[0])
        except ValueError as exc:
            raise ValueError("page must be an integer") from exc
        self._send_json(self.catalog.page(dataset_id, task_key, page))

    def _serve_static(self, filename: str) -> None:
        path = APP_DIR / filename
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        use_gzip = "gzip" in self.headers.get("Accept-Encoding", "") and len(body) > 1024
        if use_gzip:
            body = gzip.compress(body, compresslevel=5)

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the terminal useful: only surface errors.
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(format, *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"dataset directory (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cache-size", type=int, default=4, help="parsed-task LRU size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    catalog = Catalog(args.source, cache_size=args.cache_size)
    VisualiserHandler.catalog = catalog
    server = ThreadingHTTPServer((args.host, args.port), VisualiserHandler)
    task_total = sum(dataset["taskCount"] for dataset in catalog.datasets)
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(
        f"Indexed {task_total:,} tasks across {len(catalog.datasets)} datasets "
        f"in {elapsed_ms:.0f} ms."
    )
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
