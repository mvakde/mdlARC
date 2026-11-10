"""Adjacent band contraction followed by stable whole-column deduplication.

No parent modules, files, generation state, or examples are needed by solve.
The generator and self-test perform no disk writes.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _grid(value) -> np.ndarray:
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a


def _run_starts(a: np.ndarray, axis: int) -> list[int]:
    vectors = a if axis == 0 else a.T
    return [0] + [i for i in range(1, len(vectors))
                  if not np.array_equal(vectors[i], vectors[i - 1])]


def _collapse(a: np.ndarray) -> np.ndarray:
    rows = a[_run_starts(a, 0), :]
    return rows[:, _run_starts(rows, 1)].copy()


def _first_columns(a: np.ndarray) -> list[int]:
    seen = set()
    indices = []
    for j in range(a.shape[1]):
        key = tuple(int(c) for c in a[:, j])
        if key not in seen:
            seen.add(key)
            indices.append(j)
    return indices


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Apply the required two stages, with no further row deduplication."""
    compressed = _collapse(_grid(input_grid))
    return compressed[:, _first_columns(compressed)].copy()


def _integer(value, name: str, low: int, high: int | None = None) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _counts(num_train, num_test) -> tuple[int, int]:
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _key(a: np.ndarray) -> tuple:
    return a.shape, a.astype(np.int8).tobytes()


def _line_table(rng, height: int, width: int) -> np.ndarray:
    """A woven line arrangement with one copy of each column type."""
    assert height >= 3 and height % 2 == 1
    palette = rng.permutation(10).astype(np.int8)
    background = palette[0]
    vertical = palette[1:width]
    horizontal = rng.choice(palette[1:], size=height // 2)
    baseline = np.concatenate((np.array([background], dtype=np.int8), vertical))
    table = np.tile(baseline, (height, 1))
    crossing_mode = int(rng.integers(5))
    bits = rng.integers(0, 2, size=(height // 2, width - 1))
    if crossing_mode < 2:
        bits[:] = crossing_mode
    elif crossing_mode == 2:
        bits[:] = bits[:, :1]
    elif crossing_mode == 3:
        bits[:] = bits[:1, :]
    for i, color in enumerate(horizontal):
        table[2 * i + 1, 0] = color
        table[2 * i + 1, 1:] = np.where(bits[i], vertical, color)
    return table


def _tile_table(rng, height: int, width: int) -> np.ndarray:
    """Distinct column signatures, recurring row types, and a near match."""
    types = int(rng.integers(2, min(7, height)))
    minimum = 2
    while minimum ** types < width or minimum ** width < types:
        minimum += 1
    palette = rng.choice(10, size=int(rng.integers(minimum, 9)), replace=False)
    style = int(rng.integers(3))  # dense, sparse, cyclic
    for attempt in range(128):
        if style == 2 or attempt == 127:
            # Constructive fallback: column 0 also distinguishes every row.
            k = max(types, width)
            colors = rng.permutation(10)[:k]
            core = colors[(np.arange(types)[:, None] + np.arange(width)) % k]
        else:
            core = rng.choice(palette, size=(types, width))
            if style == 1:
                core[rng.random(core.shape) < 0.55] = palette[0]
        core[:, 1] = core[:, 0]
        choices = [c for c in (range(10) if attempt == 127 or style == 2 else palette)
                   if c != core[1, 0]]
        core[1, 1] = rng.choice(choices)
        row_keys = {tuple(int(c) for c in row) for row in core}
        column_keys = {tuple(int(c) for c in col) for col in core.T}
        if len(row_keys) == types and len(column_keys) == width:
            break
    else:
        raise RuntimeError("unreachable distinct-signature construction failure")
    ids = [0, 1, 0] + list(range(2, types))
    while len(ids) < height:
        options = [i for i in range(types) if i != ids[-1]]
        ids.append(int(rng.choice(options)))
    return core[ids].astype(np.int8)


def _column_events(rng, width: int, layout: str) -> list[int]:
    if layout == "blocks":
        events = list(range(width))
        tail = list(range(width))
        if rng.integers(2):
            tail.reverse()
        events.extend(tail)
    elif layout == "reflected":
        events = [0, 1, 0] + list(range(2, width)) + list(range(width - 1, -1, -1))
    elif layout == "woven":
        events = [0, 1, 0]
        for new in range(2, width):
            events.extend((new, int(rng.integers(new))))
    else:
        events = [0, 1, 0] + list(range(2, width))
    # Individual events represent maximal column runs. Unequal run lengths
    # are added separately, so adjacency and separated recurrence are distinct.
    events = [v for i, v in enumerate(events) if i == 0 or events[i - 1] != v]
    length = int(rng.integers(len(events), min(23, 2 * width + 4) + 1))
    while len(events) < length:
        events.append(int(rng.choice([v for v in range(width) if v != events[-1]])))
    if layout != "blocks":
        if events[-1] != 1:
            events.append(1)
        events.append(0)
    return events


def _stretch(rng, count: int, *, force: int = 2, maximum: int = 4) -> np.ndarray:
    lengths = np.ones(count, dtype=int)
    chosen = int(rng.integers(count))
    lengths[chosen] = force
    assert lengths.sum() <= 30
    capacity = np.maximum(maximum - lengths, 0)
    extra = int(rng.integers(0, min(30 - int(lengths.sum()), int(capacity.sum())) + 1))
    for _ in range(extra):
        available = np.flatnonzero(capacity)
        index = int(rng.choice(available))
        lengths[index] += 1
        capacity[index] -= 1
    return lengths


def _make_pair(rng, height: int, width: int, kind: str, layout: str,
               long_run: int = 2) -> dict:
    canonical = (_line_table(rng, height, width) if kind == "line"
                 else _tile_table(rng, height, width))
    # The first-seen order varies independently of color values and row order.
    canonical = canonical[:, rng.permutation(width)].copy()
    events = _column_events(rng, width, layout)
    rows = _stretch(rng, height, force=long_run)
    # A no-op adjacent-column stage is allowed; global repeats still exist.
    columns = (_stretch(rng, len(events)) if layout != "blocks" or rng.random() < 0.85
               else np.ones(len(events), dtype=int))
    source = np.repeat(np.repeat(canonical[:, events], rows, axis=0), columns, axis=1)
    # Ground truth is the explicitly constructed canonical table, rather
    # than a call to solve. Validation checks its visible reconstruction.
    return {"input": source.astype(np.int8), "output": canonical.astype(np.int8)}


def _max_run(a: np.ndarray) -> int:
    return max(int(v) for axis in (0, 1)
               for v in np.diff(_run_starts(a, axis) + [a.shape[axis]]))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(3))  # more column types, more row bands, longer runs
    kinds = ["line", "tile"]
    rng.shuffle(kinds)
    while len(kinds) < num_train:
        kinds.append(str(rng.choice(("line", "tile"))))
    puzzle = {"train": [], "test": []}
    seen = set()

    def append(split, height, width, kind, layout, long_run=2):
        # At most 14 retained inputs, at most 32 discarded candidates per pair.
        for _ in range(32):
            pair = _make_pair(rng, height, width, kind, layout, long_run)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                return
        raise RuntimeError("bounded uniqueness retries exhausted")

    for i, kind in enumerate(kinds):
        height = (2 * int(rng.integers(1, 4)) + 1 if kind == "line"
                  else int(rng.integers(3, 10)))
        width = int(rng.integers(3, 7))
        layouts = ("interrupted", "woven", "reflected") if i < 2 else (
            "interrupted", "woven", "reflected", "blocks")
        append("train", height, width, kind, str(rng.choice(layouts)))

    max_height = max(p["output"].shape[0] for p in puzzle["train"])
    max_width = max(p["output"].shape[1] for p in puzzle["train"])
    max_run = max(_max_run(p["input"]) for p in puzzle["train"])
    for i in range(num_test):
        kind = str(rng.choice(("line", "tile")))
        width = int(rng.integers(max_width + 1, 10) if shift == 0 else rng.integers(3, 7))
        if shift == 1:
            options = [v for v in range(max_height + 1, 18) if v % 2 == 1]
            height = int(rng.choice(options))
        else:
            height = (2 * int(rng.integers(1, 4)) + 1 if kind == "line"
                      else int(rng.integers(3, 10)))
        long_run = int(rng.integers(max_run + 1, 10)) if shift == 2 else 2
        layout = str(rng.choice(("interrupted", "woven", "reflected", "blocks")))
        append("test", height, width, kind, layout, long_run)
    validate_generated_puzzle(puzzle)
    return puzzle


def _require(condition, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split types")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    statistics = {"train": [], "test": []}
    witnesses = {"both": False, "recurring_rows": False, "crop": False,
                 "near": False, "last_order": False, "column_runs": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            source, target = _grid(pair["input"]), _grid(pair["output"])
            key = _key(source)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            rows = _run_starts(source, 0)
            reduced = source[rows, :]
            keys = [tuple(int(c) for c in col) for col in reduced.T]
            unique = list(dict.fromkeys(keys))
            first = [keys.index(v) for v in unique]
            expected = source[np.ix_(rows, first)]
            _require(np.array_equal(expected, target), "first-occurrence cells/order differ")
            _require(np.array_equal(solve(source), target), "solver disagreement")
            _require(source.shape[0] > target.shape[0] and source.shape[1] > target.shape[1],
                     "sample must expose both kinds of redundancy")
            _require(len(_run_starts(target, 0)) == target.shape[0], "adjacent output rows equal")
            _require(len(_first_columns(target)) == target.shape[1], "duplicate output columns")
            _require(set(int(v) for v in source.flat) == set(int(v) for v in target.flat),
                     "palette changed")
            # Exact reconstruction checks that no original information except
            # repetition counts and column placement was removed or repainted.
            row_map = np.repeat(np.arange(len(rows)), np.diff(rows + [source.shape[0]]))
            lookup = {v: j for j, v in enumerate(unique)}
            column_map = [lookup[v] for v in keys]
            reconstructed = target[np.ix_(row_map, column_map)]
            _require(np.array_equal(reconstructed, source), "lossless reconstruction failed")
            local = _collapse(source)
            _require(local.shape[1] > target.shape[1],
                     "global deduplication must remove a separated column repeat")
            row_keys = [tuple(int(c) for c in row) for row in target]
            statistics[split].append((target.shape[0], target.shape[1], _max_run(source)))
            if split == "train":
                witnesses["both"] |= (len(rows) < source.shape[0]
                                       and local.shape[1] > target.shape[1])
                witnesses["recurring_rows"] |= len(set(row_keys)) < len(row_keys)
                witnesses["crop"] |= not np.array_equal(local[:, :target.shape[1]], target)
                witnesses["near"] |= any(
                    unique[i][0] == unique[j][0] and unique[i] != unique[j]
                    for i in range(len(unique)) for j in range(i))
                last = sorted(unique, key=lambda v: len(keys) - 1 - keys[::-1].index(v))
                witnesses["last_order"] |= last != unique
                witnesses["column_runs"] |= local.shape[1] < source.shape[1]
    _require(all(witnesses.values()), f"training witnesses missing: {witnesses}")
    maxima = [max(s[i] for s in statistics["train"]) for i in range(3)]
    _require(any(all(s[i] > maxima[i] for s in statistics["test"]) for i in range(3)),
             "no consistent visible train/test shift")


def _fingerprint(puzzle) -> str:
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test() -> dict:
    """42 streaming seeds, every legal count, and bounded semantic probes."""
    shapes, output_shapes, palettes, structures, fingerprints = set(), set(), set(), set(), set()
    shift_axes, runs, widths, heights = set(), set(), set(), set()
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterminism")
        train_stats = [(p["output"].shape[0], p["output"].shape[1], _max_run(p["input"]))
                       for p in puzzle["train"]]
        maxima = [max(s[i] for s in train_stats) for i in range(3)]
        for i in range(3):
            if all((p["output"].shape[0], p["output"].shape[1], _max_run(p["input"]))[i]
                   > maxima[i] for p in puzzle["test"]):
                shift_axes.add(i)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                shapes.add(source.shape)
                output_shapes.add(target.shape)
                palettes.add(tuple(int(c) for c in np.unique(target)))
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in source.flat)
                structures.add((source.shape, hashlib.sha256(normalized).hexdigest()))
                widths.add(target.shape[1])
                heights.add(target.shape[0])
                runs.add(_max_run(source))
                pairs += 1
                # Equality is preserved by deleting duplicate coordinates:
                # reordering the two stages is observationally equivalent.
                reversed_stages = _collapse(source[:, _first_columns(source)])
                _require(np.array_equal(reversed_stages, target), "stage commutation failed")
    _require(len(fingerprints) == 42, "duplicate seeded puzzles")
    _require(len(shapes) >= 60 and len(output_shapes) >= 25, "narrow geometry")
    _require(len(structures) >= 190 and len(palettes) >= 35, "narrow structural/palette variation")
    _require(shift_axes == {0, 1, 2} and len(runs) >= 5, "missing shift/run regimes")
    _require(max(widths) == 9 and max(heights) >= 15, "missing large canonical tables")
    legal = 0
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(104729 + train * 10 + test, train, test)
            validate_generated_puzzle(p)
            legal += 1
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (4.0, 1), (4, True), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for a, expected in (
        ([[7]], [[7]]),
        ([[2, 2, 3, 2, 4]], [[2, 3, 4]]),
        ([[2], [2], [3], [2], [2]], [[2], [3], [2]]),
        ([[1, 1, 2, 1, 3], [1, 1, 2, 1, 3], [4, 4, 5, 4, 6],
          [1, 1, 2, 1, 3]], [[1, 2, 3], [4, 5, 6], [1, 2, 3]]),
        ([[1, 1, 2], [2, 3, 1]], [[1, 1, 2], [2, 3, 1]]),
    ):
        _require(np.array_equal(solve(np.array(a)), np.array(expected)), "semantic fixture")
    rng = np.random.default_rng(7351)
    # Palette permutations commute with the rule: colors have no fixed roles.
    for _ in range(24):
        a = rng.integers(0, 4, size=(int(rng.integers(1, 16)), int(rng.integers(1, 16))))
        palette = rng.permutation(10)
        _require(np.array_equal(solve(palette[a]), palette[solve(a)]), "color dependence")
        _require(np.array_equal(solve(solve(a)), solve(a)), "non-idempotent output")
        rows = rng.integers(1, 3, size=a.shape[0])
        cols = rng.integers(1, 3, size=a.shape[1])
        expanded = np.repeat(np.repeat(a, rows, axis=0), cols, axis=1)
        _require(np.array_equal(solve(expanded), solve(a)), "run expansion changed result")
    return {"seeds": 42, "default_pairs": pairs, "legal_count_combinations": legal,
            "distinct_puzzles": len(fingerprints), "input_shapes": len(shapes),
            "output_shapes": len(output_shapes), "palettes": len(palettes),
            "color_normalized_inputs": len(structures), "shift_axes": sorted(shift_axes),
            "max_run_lengths": sorted(runs), "output_heights": sorted(heights),
            "output_widths": sorted(widths), "independent_solver": False}
