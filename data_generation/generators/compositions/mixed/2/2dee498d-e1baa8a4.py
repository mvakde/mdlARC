"""Adjacent whole-line compaction followed by stable global column deduplication.

Only NumPy and the standard library are used. Generation keeps no files or state.
The required B -> A order is explicit, although these exact operations commute.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or any(not 1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or np.any((array < 0) | (array > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid in colors 0..9")
    return array


def _starts(grid, axis):
    lines = grid if axis == 0 else grid.T
    return [0] + [i for i in range(1, len(lines))
                  if not np.array_equal(lines[i], lines[i - 1])]


def _collapse_runs(grid):
    # B: row runs first, then column runs in the row-compacted grid.
    rows = grid[_starts(grid, 0), :]
    return rows[:, _starts(rows, 1)].copy()


def _stable_columns(grid):
    seen, keep = set(), []
    for j in range(grid.shape[1]):
        key = tuple(int(v) for v in grid[:, j])
        if key not in seen:
            seen.add(key)
            keep.append(j)
    return grid[:, keep].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible full-vector identities alone; never mutate the input."""
    return _stable_columns(_collapse_runs(_grid(input_grid)))


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"illegal {name}")
    return int(value)


def _key(grid):
    return (grid.shape, grid.tobytes())


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _motif(rng, height, width, style, recurrent=False):
    # A recurrent row uses the same underlying coordinate as row zero.
    row_ids = list(range(height))
    if recurrent:
        row_ids[2] = 0
    labels = list(dict.fromkeys(row_ids))
    mapping = [labels.index(i) for i in row_ids]
    effective_height = len(labels)
    minimum = 2
    while minimum ** effective_height < width:
        minimum += 1
    palette_size = int(rng.integers(minimum, 10))
    if style == "sparse":
        palette_size = minimum
    colors = rng.choice(10, size=palette_size, replace=False).astype(np.int8)
    capacity = palette_size ** effective_height
    for attempt in range(160):
        # Sampling integer codes directly makes narrow, many-column motifs
        # safe even when all available distinct column vectors are needed.
        codes = rng.choice(capacity, size=width, replace=False)
        base = np.empty((effective_height, width), dtype=np.int8)
        for r in range(effective_height):
            base[r] = colors[(codes // (palette_size ** r)) % palette_size]
        if style == "sparse" and attempt < 80:
            base = np.full_like(base, colors[0])
            mask = rng.random(base.shape) < .28
            base[mask] = rng.choice(colors[1:], size=int(mask.sum()))
        elif style == "patch" and attempt < 80 and effective_height >= 2:
            r = int(rng.integers(effective_height - 1))
            c = int(rng.integers(width - 1))
            base[r:r + 2, c:c + 2] = colors[0]
        if style == "near":
            # Row one is not aliased by the recurrent-row construction.
            base[:, 1] = base[:, 0]
            old = int(np.flatnonzero(colors == base[1, 0])[0])
            base[1, 1] = colors[(old + int(rng.integers(1, palette_size))) % palette_size]
        motif = base[mapping, :]
        if (len({tuple(col) for col in motif.T}) == width
                and len(_starts(motif, 0)) == height):
            return motif.copy()
    raise RuntimeError("bounded motif sampling exhausted")


def _word(rng, width, style):
    # Entries name whole column vectors. Never create adjacent equal entries;
    # physical adjacent repeats will be introduced by the independent bands.
    if style == "early":
        result = [0, 1, 0] + list(range(2, width))
    else:
        result = list(range(width))

    def append(value):
        value = int(value)
        if result[-1] != value:
            result.append(value)

    if style == "waves":
        for _ in range(int(rng.integers(1, 3))):
            order = list(range(width))
            if rng.integers(2):
                order.reverse()
            for value in order:
                append(value)
    else:
        for value in rng.integers(0, width, size=int(rng.integers(0, 5))):
            append(value)
        # Distinguishes retaining first occurrence from retaining last.
        for value in reversed(range(width)):
            append(value)
    # At least one separated recurrence, including for a two-vector word.
    if len(result) == width:
        append(0)
    return result


def _runs(rng, length, expand=True, unequal=False):
    if not expand:
        return np.ones(length, dtype=int)
    mode = int(rng.integers(5))
    if mode == 0 and not unequal and length <= 15:
        return np.full(length, int(rng.integers(2, min(5, 30 // length) + 1)))
    total = int(rng.integers(length + 1, min(30, 4 * length + 6) + 1))
    result = np.ones(length, dtype=int)
    extra = total - length
    if mode == 1:
        result[int(rng.integers(length))] += extra
    else:
        np.add.at(result, rng.integers(length, size=extra), 1)
        if mode == 2:
            result.sort()
        elif mode == 3:
            result = np.sort(result)[::-1].copy()
    if unequal and len(set(result.tolist())) == 1:
        result[0] += 1
        result[-1] -= 1
    return result


def _make_pair(rng, shape, witness=False):
    height, width = shape
    recurrent = witness or (height >= 3 and rng.random() < .4)
    style = "near" if witness else str(rng.choice(("random", "near", "sparse", "patch")))
    motif = _motif(rng, height, width, style, recurrent)
    word_style = "early" if witness else str(rng.choice(("early", "scatter", "waves")))
    # An early duplicate requires three distinct vectors to defeat a crop.
    if width == 2 and word_style == "early":
        word_style = "scatter"
    word = _word(rng, width, word_style)
    rows = _runs(rng, height, unequal=witness)
    columns = _runs(rng, len(word), expand=witness or rng.random() < .75, unequal=witness)
    compacted = motif[:, word]
    source = np.repeat(np.repeat(compacted, rows, axis=0), columns, axis=1)
    # First occurrences in every word are 0,1,...; label is constructed from
    # the motif, not by calling solve. Validation checks the visible rule.
    return {"input": source.astype(np.int8), "output": motif.copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    first = (int(rng.integers(3, 6)), int(rng.integers(3, 6)))
    train_shapes = [first]
    while len(train_shapes) < num_train:
        shape = (int(rng.integers(2, 7)), int(rng.integers(2, 7)))
        if len(train_shapes) == 1 and shape == first:
            continue
        train_shapes.append(shape)
    shift_axis = int(rng.integers(2))
    largest = max(shape[shift_axis] for shape in train_shapes)
    test_shapes = []
    for _ in range(num_test):
        shape = [int(rng.integers(2, 7)), int(rng.integers(2, 7))]
        shape[shift_axis] = int(rng.integers(largest + 1, 10))
        test_shapes.append(tuple(shape))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, shapes in (("train", train_shapes), ("test", test_shapes)):
        for index, shape in enumerate(shapes):
            for _ in range(32):
                pair = _make_pair(rng, shape, witness=(split == "train" and index == 0))
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _observations(source, target):
    compacted = _collapse_runs(source)
    rows = _starts(source, 0)
    cols = _starts(source, 1)
    row_runs = tuple(np.diff(rows + [source.shape[0]]).tolist())
    col_runs = tuple(np.diff(cols + [source.shape[1]]).tolist())
    row_keys = [tuple(row) for row in target]
    vectors = [tuple(col) for col in target.T]
    near = any(sum(a != b for a, b in zip(x, y)) == 1
               for i, x in enumerate(vectors) for y in vectors[i + 1:])
    return {"row_runs": row_runs, "column_runs": col_runs,
            "recurrent_row": len(set(row_keys)) < len(row_keys),
            "near_columns": near,
            "crop_fails": not np.array_equal(compacted[:, :target.shape[1]], target),
            "a_needed": not np.array_equal(compacted, target),
            "b_needed": not np.array_equal(_stable_columns(source), target)}


def validate_generated_puzzle(puzzle):
    """Check schema, reconstruction, exact order, and both-parent witnesses."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal counts")
    seen, evidence, shapes = set(), [], {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            source, target = _grid(pair["input"]), _grid(pair["output"])
            key = (source.shape, tuple(int(v) for v in source.flat))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            compacted = _collapse_runs(source)
            # Reconstruct input from observed maximal runs; this checks B's
            # whole-line bands independently of the latent construction.
            obs = _observations(source, target)
            rebuilt = np.repeat(np.repeat(compacted, obs["row_runs"], axis=0),
                                obs["column_runs"], axis=1)
            _require(np.array_equal(rebuilt, source), "inconsistent whole-line bands")
            # Direct relation to original lines: retain adjacent row starts
            # and the first ORIGINAL full-column representative of each type.
            original_cols = [tuple(int(v) for v in col) for col in source.T]
            first_cols = [original_cols.index(col) for col in dict.fromkeys(original_cols)]
            direct = source[np.ix_(_starts(source, 0), first_cols)]
            _require(np.array_equal(direct, target), "wrong values or first-occurrence order")
            _require(np.array_equal(solve(source), target), "solve disagrees")
            _require(len({tuple(c) for c in target.T}) == target.shape[1], "duplicate output column")
            _require(len(_starts(target, 0)) == target.shape[0], "adjacent output rows equal")
            _require(np.array_equal(np.unique(source), np.unique(target)), "palette changed")
            _require(obs["a_needed"] and obs["b_needed"], "each pair must exercise A and B")
            shapes[split].append(target.shape)
            if split == "train":
                evidence.append(obs)
    _require(len(set(shapes["train"])) >= 2, "training lacks shape variation")
    _require(any(o["recurrent_row"] and o["crop_fails"] and o["near_columns"]
                 and len(set(o["row_runs"])) > 1 and max(o["column_runs"]) > 1
                 for o in evidence), "training lacks the joint distinguishing witness")
    _require(any(all(shape[axis] > max(s[axis] for s in shapes["train"])
                         for shape in shapes["test"]) for axis in (0, 1)),
             "test must extrapolate a logical axis")


def _normalized_digest(grid):
    roles = {}
    values = bytes(roles.setdefault(int(c), len(roles)) for c in grid.flat)
    return hashlib.sha256(bytes(grid.shape) + values).hexdigest()


def self_test():
    """Bounded 42-seed diversity and semantic checks; no output files or traces."""
    stats = {"seeds": 42, "pairs": 0, "recurrent_rows": 0, "near_columns": 0,
             "crop_failures": 0, "column_expansion": 0, "wide_thin": 0,
             "row_shift_tests": 0, "column_shift_tests": 0}
    puzzles, inputs, outputs, structures, palettes, schedules, palette_sizes = (set() for _ in range(7))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(_fingerprint(puzzle))
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterministic")
        train_shapes = [p["output"].shape for p in puzzle["train"]]
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                before = source.copy()
                _require(np.array_equal(solve(target), target), "not idempotent")
                # A bijective color relabeling must commute with the rule.
                palette_map = np.roll(np.arange(10, dtype=np.int8), seed % 10)
                _require(np.array_equal(solve(palette_map[source]), palette_map[target]), "color dependence")
                _require(np.array_equal(source, before), "input mutated")
                # Extra copies of a physical row or column cannot change output.
                if source.shape[0] < 30:
                    stretched = np.insert(source, 0, source[0], axis=0)
                    _require(np.array_equal(solve(stretched), target), "row stretch changes output")
                if source.shape[1] < 30:
                    stretched = np.insert(source, 0, source[:, 0], axis=1)
                    _require(np.array_equal(solve(stretched), target), "column stretch changes output")
                obs = _observations(source, target)
                stats["pairs"] += 1
                stats["recurrent_rows"] += int(obs["recurrent_row"])
                stats["near_columns"] += int(obs["near_columns"])
                stats["crop_failures"] += int(obs["crop_fails"])
                stats["column_expansion"] += int(max(obs["column_runs"]) > 1)
                stats["wide_thin"] += int(any(min(x, y) == 1 and max(x, y) >= 9
                                                  for runs in (obs["row_runs"], obs["column_runs"])
                                                  for x, y in zip(runs, runs[1:])))
                inputs.add(source.shape)
                outputs.add(target.shape)
                structures.add(_normalized_digest(source))
                palette = tuple(int(v) for v in np.unique(source))
                palettes.add(palette)
                palette_sizes.add(len(palette))
                schedules.add((obs["row_runs"], obs["column_runs"]))
                if split == "test":
                    stats["row_shift_tests"] += int(target.shape[0] > max(h for h, _ in train_shapes))
                    stats["column_shift_tests"] += int(target.shape[1] > max(w for _, w in train_shapes))
    stats.update(distinct_puzzles=len(puzzles), input_shapes=len(inputs), output_shapes=len(outputs),
                 normalized_inputs=len(structures), palettes=len(palettes), band_schedules=len(schedules),
                 palette_sizes=sorted(palette_sizes))
    for name, minimum in (("distinct_puzzles", 42), ("input_shapes", 30), ("output_shapes", 20),
                          ("normalized_inputs", 150), ("palettes", 30), ("band_schedules", 80),
                          ("recurrent_rows", 42), ("crop_failures", 42), ("near_columns", 42),
                          ("wide_thin", 1), ("row_shift_tests", 1), ("column_shift_tests", 1)):
        _require(stats[name] >= minimum, f"insufficient variation: {name}={stats[name]}")
    _require(min(palette_sizes) == 2 and max(palette_sizes) >= 6, "palette-size coverage")
    legal = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(4242 + 10 * a + b, a, b)
            legal += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Boundary members exercise the solver's complete grid domain, including
    # single lines and no-op stages outside the generator's witness schedule.
    for shape in ((1, 1), (1, 30), (30, 1), (30, 30)):
        source = (np.arange(np.prod(shape)).reshape(shape) % 10).astype(np.int8)
        result = solve(source)
        _require(np.array_equal(solve(result), result), "boundary idempotence")
    stats["legal_count_combinations"] = legal
    return stats
