"""Execute validated generators and bridge their samples to the Task Atlas."""

from __future__ import annotations

import hashlib
import json
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, Iterator, Optional

import numpy as np

from .artifacts import atomic_write_json
from .config import PROJECT_ROOT, PUZZLE_ID_RE, RunnerConfig


DEFAULT_BULK_SAMPLE_COUNT = 42
DEFAULT_GENERATED_TASK_ROOT = PROJECT_ROOT / "generated_tasks"


@dataclass(frozen=True)
class GeneratorRun:
    puzzle_id: str
    source_split: str
    run_dir: Path
    generator_path: Path
    code_sha256: str
    completed_at: str


@dataclass(frozen=True)
class SampleBatch:
    generator: GeneratorRun
    source_root: Path
    task_dir: Path
    manifest_path: Path
    seeds: tuple[int, ...]
    generated_seeds: tuple[int, ...]
    reused_seeds: tuple[int, ...]
    self_test_mode: str


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _code_sha256(path: Path, expected_hash: Optional[str] = None) -> Optional[str]:
    try:
        code = path.read_text(encoding="utf-8")
        exact_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        normalized_hash = hashlib.sha256(code.rstrip().encode("utf-8")).hexdigest()
        if expected_hash is not None:
            return expected_hash if expected_hash in (exact_hash, normalized_hash) else None
        # Legacy callers use the original response-artifact convention.
        return normalized_hash
    except (OSError, UnicodeError):
        return None


def _candidate_runs(config, puzzle_id, source_split):
    from task_variations.common.data import all_task_refs
    from task_variations.common.storage import current, sha_bytes
    stage = config.stage
    name = 'generator_v2.py' if stage == 'generate-from-reviews' else 'generator_v1.py'
    for ref in all_task_refs(config, source_split):
        if puzzle_id is not None and ref.puzzle_id != puzzle_id:
            continue
        task = config.task_root / ref.source_split / ref.puzzle_id
        path = task / name
        if path.is_file() and current(task, stage):
            yield GeneratorRun(ref.puzzle_id, ref.source_split, task, path,
                sha_bytes(path.read_text().rstrip().encode()), '')


def _candidate_runs_in_directory(
    generator_dir: Path,
    source_split: str,
) -> Iterator[GeneratorRun]:
    root = generator_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Generator directory does not exist: %s" % root)

    for generator_path in sorted(root.rglob("generator.py")):
        run_dir = generator_path.parent
        status = _load_json(run_dir / "status.json")
        validation = _load_json(run_dir / "validation.json")
        metadata = _load_json(run_dir / "metadata.json")
        if status is None or validation is None or metadata is None:
            continue
        target = metadata.get("target")
        metadata_split = target.get("source_split") if isinstance(target, dict) else None
        metadata_id = target.get("id") if isinstance(target, dict) else None
        path_split = run_dir.parent.parent.name
        path_id = run_dir.parent.name
        selected_split = metadata_split if metadata_split in ("training", "evaluation") else path_split
        selected_id = metadata_id if isinstance(metadata_id, str) else path_id
        if (
            selected_split not in ("training", "evaluation")
            or not PUZZLE_ID_RE.fullmatch(selected_id)
            or (source_split != "auto" and selected_split != source_split)
            or status.get("state") != "succeeded"
            or validation.get("valid") is not True
            or metadata.get("state") != "succeeded"
        ):
            continue
        expected_hash = validation.get("code_sha256")
        code_sha256 = (
            _code_sha256(generator_path, expected_hash)
            if isinstance(expected_hash, str)
            else None
        )
        if code_sha256 is None:
            continue
        completed_at = status.get("completed_at") or metadata.get("completed_at") or ""
        yield GeneratorRun(
            puzzle_id=selected_id,
            source_split=selected_split,
            run_dir=run_dir,
            generator_path=generator_path,
            code_sha256=code_sha256,
            completed_at=str(completed_at),
        )


def find_latest_generator(
    config: RunnerConfig,
    puzzle_id: Optional[str] = None,
    source_split: str = "auto",
) -> GeneratorRun:
    if source_split not in ("auto", "training", "evaluation"):
        raise ValueError("Generator split must be auto, training, or evaluation")
    candidates = list(_candidate_runs(config, puzzle_id, source_split))
    if not candidates:
        target = puzzle_id or "any puzzle"
        raise ValueError(
            "No successful, validated generator was found for %s in split %s"
            % (target, source_split)
        )
    return max(
        candidates,
        key=lambda item: (item.completed_at, item.run_dir.name, item.source_split, item.puzzle_id),
    )


def find_latest_generators(
    config: RunnerConfig,
    puzzle_ids: Iterable[str] = (),
    source_split: str = "auto",
    *,
    all_generators: bool = False,
    generator_dir: Optional[Path] = None,
) -> tuple[GeneratorRun, ...]:
    normalized_ids = tuple(sorted(set(puzzle_ids)))
    if source_split not in ("auto", "training", "evaluation"):
        raise ValueError("Generator split must be auto, training, or evaluation")
    if generator_dir is not None and normalized_ids:
        raise ValueError("Use --generator-dir or --puzzle-id selectors, not both")
    if all_generators and normalized_ids:
        raise ValueError("Use --all or --puzzle-id selectors, not both")
    if not all_generators and generator_dir is None:
        if normalized_ids:
            return tuple(
                find_latest_generator(config, puzzle_id, source_split)
                for puzzle_id in normalized_ids
            )
        return (find_latest_generator(config, None, source_split),)

    latest: Dict[tuple[str, str], GeneratorRun] = {}
    candidates = (
        _candidate_runs_in_directory(generator_dir, source_split)
        if generator_dir is not None
        else _candidate_runs(config, None, source_split)
    )
    for candidate in candidates:
        key = (candidate.source_split, candidate.puzzle_id)
        current = latest.get(key)
        if current is None or (candidate.completed_at, candidate.run_dir.name) > (
            current.completed_at,
            current.run_dir.name,
        ):
            latest[key] = candidate
    if not latest:
        location = " under %s" % generator_dir if generator_dir is not None else ""
        raise ValueError(
            "No successful, validated generators were found%s in split %s"
            % (location, source_split)
        )
    return tuple(latest[key] for key in sorted(latest))


def _load_generator(path: Path) -> Dict[str, Any]:
    # runpy would also work, but compiling into a private module gives generated
    # functions a normal module namespace without adding its directory to PATH.
    module = ModuleType("_task_variations_generated")
    module.__file__ = str(path)
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    exec(code, module.__dict__)
    required = ("generate_puzzle", "validate_generated_puzzle", "self_test")
    missing = [name for name in required if not callable(module.__dict__.get(name))]
    if missing:
        raise ValueError("Generator is missing callable functions: %s" % ", ".join(missing))
    return module.__dict__


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Generated puzzle dictionaries must use string keys")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("Generated puzzle contains unsupported value %r" % type(value).__name__)


def _validate_grid(grid: Any, label: str) -> None:
    if not isinstance(grid, list) or not grid:
        raise ValueError("%s must be a non-empty list of rows" % label)
    if any(not isinstance(row, list) or not row for row in grid):
        raise ValueError("%s rows must be non-empty lists" % label)
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("%s must be rectangular" % label)
    for row in grid:
        for color in row:
            if isinstance(color, bool) or not isinstance(color, int) or not 0 <= color <= 9:
                raise ValueError("%s must contain integer ARC colors from 0 through 9" % label)


def _validate_arc_json(puzzle: Any) -> Dict[str, Any]:
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("Generated puzzle must contain exactly train and test")
    for split_name in ("train", "test"):
        pairs = puzzle[split_name]
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("Generated puzzle %s must be a non-empty list" % split_name)
        for index, pair in enumerate(pairs):
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError(
                    "Generated %s pair %d must contain exactly input and output"
                    % (split_name, index)
                )
            _validate_grid(pair["input"], "%s[%d].input" % (split_name, index))
            _validate_grid(pair["output"], "%s[%d].output" % (split_name, index))
    return puzzle


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _self_test_cache_path(config: RunnerConfig, generator_run: GeneratorRun) -> Path:
    return (
        config.artifact_root
        / "samples"
        / generator_run.source_split
        / generator_run.puzzle_id
        / generator_run.run_dir.name
        / "self_test.json"
    )


def _has_cached_self_test(config: RunnerConfig, generator_run: GeneratorRun) -> bool:
    cache_path = _self_test_cache_path(config, generator_run)
    cache = _load_json(cache_path)
    if cache is not None and (
        cache.get("state") == "passed"
        and cache.get("generator_sha256") == generator_run.code_sha256
    ):
        return True

    # Sample batches made by an earlier bridge version are also evidence of a
    # completed self-test. Promote that evidence into the compact cache file.
    run_sample_root = cache_path.parent
    for manifest_path in run_sample_root.glob(
        "*/Generated-%s/manifest.json" % generator_run.puzzle_id
    ):
        manifest = _load_json(manifest_path)
        if manifest is None:
            continue
        generator = manifest.get("generator")
        sampling = manifest.get("sampling")
        if (
            isinstance(generator, dict)
            and isinstance(sampling, dict)
            and generator.get("sha256") == generator_run.code_sha256
            and sampling.get("self_test_passed") is True
        ):
            atomic_write_json(
                cache_path,
                {
                    "schema_version": 1,
                    "state": "passed",
                    "generator_sha256": generator_run.code_sha256,
                    "generator_path": str(generator_run.generator_path),
                    "passed_at": manifest.get("generated_at"),
                },
            )
            return True
    return False


def generate_samples(
    config: RunnerConfig,
    generator_run: GeneratorRun,
    *,
    start_seed: int = 0,
    count: int = 10,
    num_train: Optional[int] = None,
    num_test: Optional[int] = None,
    run_self_test: bool = True,
    source_root: Optional[Path] = None,
    dataset_name: Optional[str] = None,
    tasks_direct: bool = False,
    resume: bool = False,
) -> SampleBatch:
    if start_seed < 0:
        raise ValueError("start seed must be non-negative")
    if count < 1:
        raise ValueError("sample count must be at least 1")
    if _code_sha256(generator_run.generator_path, generator_run.code_sha256) is None:
        raise ValueError("Generator changed after validation; refusing to execute it")

    namespace = _load_generator(generator_run.generator_path)
    self_test_mode = "not-requested"
    if run_self_test:
        if _has_cached_self_test(config, generator_run):
            self_test_mode = "cached"
        else:
            try:
                namespace["self_test"]()
            except Exception as exc:
                raise ValueError("Generator self-test failed: %s" % exc) from exc
            self_test_mode = "executed"
            atomic_write_json(
                _self_test_cache_path(config, generator_run),
                {
                    "schema_version": 1,
                    "state": "passed",
                    "generator_sha256": generator_run.code_sha256,
                    "generator_path": str(generator_run.generator_path),
                    "passed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                },
            )

    batch_root = source_root or (
        config.artifact_root
        / "samples"
        / generator_run.source_split
        / generator_run.puzzle_id
        / generator_run.run_dir.name
        / _timestamp()
    )
    dataset_root = batch_root / (dataset_name or ("Generated-" + generator_run.puzzle_id))
    task_dir = dataset_root if tasks_direct else dataset_root / "tasks"
    seeds = tuple(range(start_seed, start_seed + count))
    task_paths = [task_dir / ("seed-%010d.json" % seed) for seed in seeds]
    manifest_path = dataset_root / ("manifest" if tasks_direct else "manifest.json")

    existing_manifest = _load_json(manifest_path) if resume else None
    existing_generator = (
        existing_manifest.get("generator") if existing_manifest is not None else None
    )
    existing_sampling = (
        existing_manifest.get("sampling") if existing_manifest is not None else None
    )
    resume_matches = bool(
        isinstance(existing_generator, dict)
        and isinstance(existing_sampling, dict)
        and existing_generator.get("sha256") == generator_run.code_sha256
        and existing_sampling.get("start_seed") == start_seed
        and existing_sampling.get("count") == count
        and existing_sampling.get("num_train") == num_train
        and existing_sampling.get("num_test") == num_test
    )
    completed_values = existing_manifest.get("completed_seeds", []) if resume_matches else []
    if not isinstance(completed_values, list):
        completed_values = []
    completed_seeds = {
        seed
        for seed in completed_values
        if isinstance(seed, int)
        and seed in seeds
        and (task_dir / ("seed-%010d.json" % seed)).is_file()
    }
    reset_direct_tasks = (not resume and tasks_direct) or (resume and not resume_matches)
    if reset_direct_tasks and task_dir.is_dir():
        for stale_path in task_dir.glob("seed-*.json"):
            if stale_path.is_file():
                stale_path.unlink()
    elif resume and task_dir.is_dir():
        expected_paths = set(task_paths)
        for stale_path in task_dir.glob("seed-*.json"):
            if stale_path.is_file() and stale_path not in expected_paths:
                stale_path.unlink()

    generated_seeds = []
    reused_seeds = sorted(completed_seeds)

    def write_manifest(state: str, error: Optional[Dict[str, str]] = None) -> None:
        payload: Dict[str, Any] = {
            "schema_version": 2,
            "state": state,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "generator": {
                "puzzle_id": generator_run.puzzle_id,
                "source_split": generator_run.source_split,
                "run_dir": str(generator_run.run_dir),
                "path": str(generator_run.generator_path),
                "sha256": generator_run.code_sha256,
            },
            "sampling": {
                "start_seed": start_seed,
                "count": count,
                "num_train": num_train,
                "num_test": num_test,
                "self_test_passed": run_self_test,
                "self_test_mode": self_test_mode,
            },
            "completed_seeds": sorted(completed_seeds),
            "task_paths": [
                str(task_dir / ("seed-%010d.json" % seed))
                for seed in sorted(completed_seeds)
            ],
        }
        if error is not None:
            payload["error"] = error
        atomic_write_json(manifest_path, payload)

    if resume:
        write_manifest("running")

    for seed in seeds:
        if seed in completed_seeds:
            continue
        try:
            puzzle = namespace["generate_puzzle"](
                seed,
                num_train=num_train,
                num_test=num_test,
            )
            namespace["validate_generated_puzzle"](puzzle)
            serialized = _validate_arc_json(_json_value(puzzle))
        except Exception as exc:
            if resume:
                write_manifest(
                    "failed",
                    {"type": type(exc).__name__, "message": str(exc), "seed": str(seed)},
                )
            raise ValueError("Generator failed for seed %d: %s" % (seed, exc)) from exc
        task_path = task_dir / ("seed-%010d.json" % seed)
        atomic_write_json(task_path, serialized)
        completed_seeds.add(seed)
        generated_seeds.append(seed)
        if resume:
            write_manifest("running")

    write_manifest("complete")
    return SampleBatch(
        generator_run,
        batch_root,
        task_dir,
        manifest_path,
        seeds,
        tuple(generated_seeds),
        tuple(reused_seeds),
        self_test_mode,
    )


def visualizer_command(source_root: Path, host: str, port: int) -> list[str]:
    script = PROJECT_ROOT / "tools" / "visualiser" / "server.py"
    if not script.is_file():
        raise ValueError("Task Atlas server was not found at %s" % script)
    if not 1 <= port <= 65535:
        raise ValueError("port must be from 1 through 65535")
    return [
        sys.executable,
        str(script),
        "--source",
        str(source_root),
        "--host",
        host,
        "--port",
        str(port),
    ]


def serve_samples(
    source_root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    browser_opener: Callable[[str], Any] = webbrowser.open,
) -> int:
    command = visualizer_command(source_root, host, port)
    url_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    if ":" in url_host and not url_host.startswith("["):
        url_host = "[%s]" % url_host
    url = "http://%s:%d" % (url_host, port)
    print("Task Atlas: %s" % url, flush=True)
    print("Press Ctrl-C to stop it.", flush=True)

    process = subprocess.Popen(command)
    stop_browser_wait = threading.Event()
    browser_thread: Optional[threading.Thread] = None
    if open_browser:
        def open_when_ready() -> None:
            deadline = time.monotonic() + 30
            while not stop_browser_wait.is_set() and time.monotonic() < deadline:
                if process.poll() is not None:
                    return
                try:
                    with urllib.request.urlopen(url + "/api/index", timeout=0.5):
                        browser_opener(url)
                        return
                except (OSError, urllib.error.URLError):
                    stop_browser_wait.wait(0.1)

        browser_thread = threading.Thread(target=open_when_ready, daemon=True)
        browser_thread.start()
    try:
        return process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            try:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                process.terminate()
        return 0
    finally:
        stop_browser_wait.set()
        if browser_thread is not None:
            browser_thread.join(timeout=1)
