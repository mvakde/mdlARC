"""Collapse repeated pixel rows/columns, then crop by compact-square parity.

Only standard-library modules and NumPy are used. Generation and validation do
not supply any state to solve(). No execution traces or puzzle corpora are saved.
"""
from __future__ import annotations

import hashlib

import numpy as np


_LEGAL_COUNTS = tuple((a, b) for a in range(2, 11)
                      for b in range(1, min(4, a - 1) + 1))

# SHA256 of shapes + int8 cells, input then output, for the original and
# selected-variation pairs supplied as evidence. Used only to exclude copying.
_PARENT_PAIRS = frozenset((
    "02cf1a1c5b4f0841a3d2e4d3966734a1fa1d9875f4dab1f8966444778da9562b",
    "13c97c7ba06034967daae6fff97ac0818a886d39cac012d07aa0ed39f21be093",
    "249ef2bd27295d365606ddc7e8e94f8a80cd4e67b09b72d266b9ebf967f45e12",
    "30c37afa5447c9ef71a8056db2446cf48ae5d35330c3c47339fa7211c16af868",
    "33e5c1eef15e1241c2ccd6116efce49e36b0889dfffa743f764753046515e5f5",
    "36420eca07af91ccb85701c0dd358a5080c6cd3eb90d6198c3b4b6cd2ae2c196",
    "3682461799bf70d86b7b4aea023eafdaf5a47b43eef7fe591082d3d86b016549",
    "48d4b7267fbbd7d96c9f50d8ed8c749b0b7dd0b2071db4d385f7c44f894490e1",
    "4dd8187e059cfa01dc496f54e0de7a5c33182c0e66e83c0e8aa0aba83a4f8049",
    "632cbff5576fc4008e02bd0350b89635e3bf84ca758c2ad3fea6e03c834edb81",
    "63c5b8b7a3a24df0917757aaaef261143ea9233e25b6408203013f1a21dfe570",
    "6acf254736a43249dbdd67c1aec795f7b4656788e04f53e01ab97458f385f428",
    "75ad96019a9aa6a95284d7eca1ac997f5869f1d82af639205678108beddf82cc",
    "9f33142234a0c409b33b4bd0b1c33120382a0a0c2512d877019a42d5ce4e1ba9",
    "bc85d08601f2018b39302ca456828e78cb8887019867b303dede28f67d0a5a0f",
    "bd088196bfabef92168a22a1190d7dc699879c7ea1215175d3e7dc16a6af2a56",
    "bd29ff6b385a103806727af730184754bc957a9827c257a788fc9aae75a631c1",
    "c622ba9bdf30b4aa9eb8374043d8cb563e2933b9d0897b93f16bb92237cf1a69",
    "d47a81ecf3cdd142ae41ad61e52257afea11b509e5d25add32204f79d515a0aa",
    "fee985bbc7a0747c9edc4b760e3ad2602b1bcb35cbe473c466f185d61f8e2a94",
))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    try:
        grid = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("grid must be a rectangular integer array") from error
    if (grid.ndim != 2 or not all(1 <= v <= 30 for v in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not bool(np.all((0 <= grid) & (grid <= 9)))):
        raise ValueError("grid must have integer colors 0..9 and dimensions 1..30")
    return grid


def _collapse(grid):
    # Compare WHOLE pixel vectors. Different colors within one vector do not
    # define boundaries; separated equal vectors are deliberately retained.
    row_starts = np.r_[True, np.any(grid[1:] != grid[:-1], axis=1)]
    col_starts = np.r_[True, np.any(grid[:, 1:] != grid[:, :-1], axis=0)]
    return grid[np.ix_(row_starts, col_starts)].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution for every valid family member, expanded or unit scale.

    A valid input has a square run quotient of side 2..30. A nonsquare raw
    rectangle is allowed; a nonsquare quotient or a 1x1 quotient is not.
    """
    compact = _collapse(_grid(input_grid))
    n = compact.shape[0]
    if n < 2 or compact.shape[1] != n:
        raise ValueError("collapsing runs must leave a square of side at least two")
    return (compact[:2, :2] if n % 2 else compact[:2, -2:]).copy()


def _crop(compact, parity=None):
    p = compact.shape[0] % 2 if parity is None else parity
    return (compact[:2, :2] if p else compact[:2, -2:]).copy()


def _signature(grid):
    canonical = np.asarray(grid, dtype=np.int8)
    return canonical.shape, canonical.tobytes()


def _pair_digest(pair):
    digest = hashlib.sha256()
    for role in ("input", "output"):
        g = np.asarray(pair[role], dtype=np.int8)
        digest.update(bytes(g.shape))
        digest.update(g.tobytes())
    return digest.hexdigest()


def _irreducible(compact):
    return (bool(np.all(np.any(compact[1:] != compact[:-1], axis=1)))
            and bool(np.all(np.any(compact[:, 1:] != compact[:, :-1], axis=0))))


def _stable_unique(compact):
    # Deliberately incorrect alternative: remove nonadjacent copies too.
    rows, cols, seen = [], [], set()
    for i, row in enumerate(compact):
        key = tuple(row)
        if key not in seen:
            rows.append(i)
            seen.add(key)
    seen = set()
    for j, col in enumerate(compact.T):
        key = tuple(col)
        if key not in seen:
            cols.append(j)
            seen.add(key)
    return compact[np.ix_(rows, cols)]


def _corner_witness(compact):
    out = _crop(compact)
    return (not np.array_equal(compact[:2, :2], compact[:2, -2:])
            and not np.array_equal(out[0], out[1])
            and not np.array_equal(out[:, 0], out[:, 1]))


def _recurrence_witness(compact):
    unique = _stable_unique(compact)
    n = compact.shape[0]
    return (unique.shape == (n - 1, n - 1)
            and not np.array_equal(_crop(unique), _crop(compact)))


def _mosaic(rng, n, witness=False, recurring=False):
    """Construct a square with no adjacent identical whole rows or columns."""
    for _ in range(96):
        k = int(rng.integers(3 if witness else 2, 11))
        palette = rng.choice(10, size=k, replace=False).astype(np.int8)
        rows, cols = np.indices((n, n))
        mode = int(rng.integers(0, 7))
        if mode == 0:  # Dense independent cells, including local equal colors.
            labels = rng.integers(k, size=(n, n))
        elif mode == 1:  # A weighted field with a dominant color.
            weights = rng.dirichlet(np.full(k, .7))
            labels = rng.choice(k, size=(n, n), p=weights)
        elif mode == 2:  # Clean periodic tile: distant repeated vectors survive.
            h = int(rng.integers(2, min(4, n) + 1))
            w = int(rng.integers(2, min(4, n) + 1))
            tile = rng.integers(k, size=(h, w))
            labels = tile[rows % h, cols % w]
        elif mode == 3:  # Diagonal arithmetic weave.
            a, b = rng.integers(1, k, size=2)
            labels = (a * rows + b * cols + int(rng.integers(k))) % k
        elif mode == 4:  # Sparse permutation motif on an arbitrary ground color.
            labels = np.zeros((n, n), dtype=int)
            labels[np.arange(n), rng.permutation(n)] = rng.integers(1, k, size=n)
        elif mode == 5:  # Distance from a diagonal; broad areas of equal cells.
            labels = (np.abs(rows - cols) + int(rng.integers(k))) % k
        else:  # Intersecting recurrent sequences, with nonadjacent returns.
            def sequence():
                values = [int(rng.integers(k))]
                for _ in range(n - 1):
                    values.append((values[-1] + int(rng.integers(1, k))) % k)
                return np.array(values)
            labels = (sequence()[:, None] + sequence()[None, :]) % k
        compact = palette[labels]
        compact = np.rot90(compact, int(rng.integers(4))).copy()
        if recurring:
            compact[2] = compact[0]
            compact[:, 2] = compact[:, 0]
        if not _irreducible(compact):
            continue
        if witness and not _corner_witness(compact):
            continue
        if recurring and not _recurrence_witness(compact):
            continue
        return compact
    # Constructive fallback has distinct adjacent vectors for every side. For
    # the odd witness (5,7,9), copying index 0 to index 2 removes exactly one
    # global duplicate on each axis and changes the wrong global-dedup crop.
    rows, cols = np.indices((n, n))
    compact = rng.permutation(10).astype(np.int8)[(3 * rows + cols) % 10]
    if recurring:
        compact[2] = compact[0]
        compact[:, 2] = compact[:, 0]
    _require(_irreducible(compact), "fallback mosaic lost a boundary")
    _require(not witness or _corner_witness(compact), "fallback corner witness failed")
    _require(not recurring or _recurrence_witness(compact), "fallback recurrence failed")
    return compact


def _partition(rng, n, total, thick_indices=()):
    minimum = np.ones(n, dtype=int)
    for index in thick_indices:
        minimum[index] = 2
    remaining = total - int(minimum.sum())
    if remaining < 0:
        raise ValueError("not enough space for required bands")
    lengths = minimum + rng.multinomial(remaining, rng.dirichlet(np.ones(n)))
    if np.all(lengths == lengths[0]) and remaining:
        movable = np.flatnonzero(lengths > minimum)
        if movable.size:
            i = int(rng.choice(movable))
            j = (i + int(rng.integers(1, n))) % n
            lengths[i] -= 1
            lengths[j] += 1
    return lengths


def _lengths(rng, n, force_expansion=False):
    if n == 30:
        return np.ones(n, dtype=int)
    mode = int(rng.integers(5))
    if mode == 0:
        lengths = np.ones(n, dtype=int)
    elif mode == 1:
        t = int(rng.integers(1, min(6, 30 // n) + 1))
        lengths = np.full(n, t, dtype=int)
    elif mode == 2:
        lengths = np.ones(n, dtype=int)
        lengths[int(rng.integers(n))] += int(rng.integers(1, 31 - n))
    elif mode == 3:
        total = int(rng.integers(n, 31))
        lengths = _partition(rng, n, total)
    else:
        lengths = rng.integers(1, 4, size=n)
        while lengths.sum() > 30:
            lengths[int(rng.choice(np.flatnonzero(lengths > 1)))] -= 1
    if force_expansion and n < 30 and lengths.sum() == n:
        lengths[int(rng.integers(n))] = 2
    return lengths


def _make_pair(rng, n, used, shape=None, witness=False, recurring=False,
               force_expansion=False):
    for _ in range(64):
        compact = _mosaic(rng, n, witness, recurring)
        if shape is not None:
            heights = _partition(rng, n, shape[0], (0,))
            widths = _partition(rng, n, shape[1], (0, n - 1))
        else:
            heights = _lengths(rng, n, force_expansion)
            widths = _lengths(rng, n, force_expansion)
        grid = np.repeat(np.repeat(compact, heights, axis=0), widths, axis=1)
        pair = {"input": grid, "output": _crop(compact)}
        key = _signature(grid)
        if key not in used and _pair_digest(pair) not in _PARENT_PAIRS:
            used.add(key)
            return pair
    raise RuntimeError("bounded unique-pair sampling exhausted")


def _check_count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one reproducible puzzle with fixed family conventions."""
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    a = _check_count(num_train, 2, 10, "num_train")
    b = _check_count(num_test, 1, 4, "num_test")
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    # Four consecutive sizes interleave parity and scale. The upper odd side
    # is at least five, permitting a genuinely informative returning vector.
    start = int(rng.integers(3, 7))
    sides = list(range(start, start + 4))
    odd = max(n for n in sides if n % 2)
    even = int(rng.choice([n for n in sides if n % 2 == 0]))
    rest = [n for n in sides if n not in (odd, even)]
    rng.shuffle(rest)
    train_sides = [odd, even] + rest[:a - 2]
    while len(train_sides) < a:
        train_sides.append(int(rng.integers(2, 11)))
    largest = max(odd, even)
    height = int(rng.integers(largest + 4, min(30, largest + 14) + 1))
    width = height if rng.random() < .35 else int(
        rng.integers(largest + 4, min(30, largest + 14) + 1))
    used = set()
    train = [
        _make_pair(rng, n, used, (height, width) if i < 2 else None,
                   witness=i < 4, recurring=i == 0)
        for i, n in enumerate(train_sides)
    ]
    rng.shuffle(train)
    # At least one test extends logical resolution, not merely pixel scale.
    band = int(rng.choice(3, p=(.70, .20, .10)))
    lo, hi = ((11, 18), (19, 24), (25, 29))[band]
    test_sides = [int(rng.integers(lo, hi + 1))]
    if b >= 2:
        test_sides.append(int(rng.choice([n for n in range(2, 31)
                                         if n % 2 != test_sides[0] % 2])))
    while len(test_sides) < b:
        test_sides.append(int(rng.integers(2, 31)))
    test = [_make_pair(rng, n, used, force_expansion=i == 0)
            for i, n in enumerate(test_sides)]
    rng.shuffle(test)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _decode(grid):
    """Reconstruct groups by full-vector scanning, for invariant checks."""
    r = [0]
    c = [0]
    for i in range(1, grid.shape[0]):
        if not np.array_equal(grid[i], grid[i - 1]):
            r.append(i)
    for j in range(1, grid.shape[1]):
        if not np.array_equal(grid[:, j], grid[:, j - 1]):
            c.append(j)
    heights = np.diff(r + [grid.shape[0]])
    widths = np.diff(c + [grid.shape[1]])
    compact = grid[np.ix_(r, c)]
    return compact, heights, widths


def _features(grid):
    compact, heights, widths = _decode(grid)
    out = _crop(compact)
    return {
        "side": compact.shape[0], "shape": grid.shape,
        "heights": tuple(int(v) for v in heights),
        "widths": tuple(int(v) for v in widths),
        "corner_contrast": _corner_witness(compact),
        "witness": (_corner_witness(compact) and heights[0] >= 2
                    and widths[0] >= 2 and widths[-1] >= 2),
        "recurrence": _recurrence_witness(compact),
        "height_shortcut_fails": not np.array_equal(out, _crop(compact, grid.shape[0] % 2)),
        "width_shortcut_fails": not np.array_equal(out, _crop(compact, grid.shape[1] % 2)),
    }


def validate_generated_puzzle(puzzle):
    """Check family geometry, exact values, uniqueness and teaching witnesses.

    This checks the authored rule; it is not an independent semantic review.
    Arrays and ordinary nested integer lists are both accepted.
    """
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "expected exactly train/test keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require((a, b) in _LEGAL_COUNTS, "illegal pair counts")
    seen, info = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "expected exactly input/output keys")
            grid, out = _grid(pair["input"]), _grid(pair["output"])
            compact, heights, widths = _decode(grid)
            n = compact.shape[0]
            _require(compact.shape == (n, n) and n >= 2, "quotient must be square, side >=2")
            _require(_irreducible(compact), "adjacent quotient vectors must differ")
            reconstructed = np.repeat(np.repeat(compact, heights, axis=0), widths, axis=1)
            _require(np.array_equal(reconstructed, grid), "expansion did not reconstruct input")
            # Explicit indexing checks both copying and orientation, independently
            # of the vectorized solve implementation.
            j = 0 if n % 2 else n - 2
            expected = np.array([[compact[0, j], compact[0, j + 1]],
                                 [compact[1, j], compact[1, j + 1]]])
            _require(out.shape == (2, 2) and np.array_equal(out, expected), "wrong corner crop")
            key = _signature(grid)
            _require(key not in seen, "duplicate input")
            _require(_pair_digest(pair) not in _PARENT_PAIRS, "copied parent pair")
            seen.add(key)
            info[split].append(_features(grid))
    train, test = info["train"], info["test"]
    witnesses = [f for f in train if f["witness"]]
    _require({f["side"] % 2 for f in witnesses} == {0, 1}, "need a clear witness of each parity")
    _require(any(x["shape"] == y["shape"] and x["side"] % 2 != y["side"] % 2
                 for x in witnesses for y in witnesses), "need opposite parities at the same raw dimensions")
    _require(any(f["recurrence"] for f in train), "need a run-versus-global-unique witness")
    _require(any(f["height_shortcut_fails"] for f in train), "raw height parity remains a shortcut")
    _require(any(f["width_shortcut_fails"] for f in train), "raw width parity remains a shortcut")
    _require(any(len(set(f["heights"])) > 1 or len(set(f["widths"])) > 1 for f in train),
             "need unequal stretching in training")
    _require(any(f["side"] > max(x["side"] for x in train) for f in test),
             "need at least one test at a new logical resolution")
    if b >= 2:
        _require({f["side"] % 2 for f in test} == {0, 1}, "multiple tests should include both parities")
    if a >= 4:
        for parity in (0, 1):
            _require(len({f["side"] for f in train
                          if f["side"] % 2 == parity and f["corner_contrast"]}) >= 2,
                     "four or more demonstrations need two informative sizes of each parity")


def _puzzle_digest(puzzle):
    return tuple(_pair_digest(pair) for s in ("train", "test") for pair in puzzle[s])


def _normalized_key(grid):
    roles = {}
    return grid.shape, bytes(roles.setdefault(int(c), len(roles)) for c in grid.flat)


def self_test():
    """Bounded streaming checks over 42 seeds; returns descriptive measurements.

    At most two puzzles are retained at once. No files are written. No trace,
    worker, retry journal, index, backup or multi-seed grid archive is created.
    """
    puzzles, shapes, compact_shapes, patterns, palettes, thicknesses = (set() for _ in range(6))
    test_parities, counts = set(), set()
    metrics = {"seeds": 42, "pairs": 0, "uniform_outputs": 0, "equal_corners": 0,
               "unit_scale": 0, "one_axis_only_expansion": 0, "sparse_compacts": 0,
               "repeated_vectors": 0, "height_parity_counterexamples": 0,
               "width_parity_counterexamples": 0}
    for seed in range(42):
        a, b = _LEGAL_COUNTS[seed % len(_LEGAL_COUNTS)]
        puzzle = generate_puzzle(seed, a, b)
        validate_generated_puzzle(puzzle)
        _require(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed, a, b)),
                 "seed is not reproducible")
        puzzles.add(_puzzle_digest(puzzle))
        counts.add((a, b))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, out = pair["input"], pair["output"]
                original = grid.copy()
                _require(np.array_equal(solve(grid), out), "solve disagreed")
                _require(np.array_equal(grid, original), "solve mutated its input")
                compact, heights, widths = _decode(grid)
                def rows_only(value):
                    keep = np.r_[True, np.any(value[1:] != value[:-1], axis=1)]
                    return value[keep].copy()
                forward = rows_only(rows_only(grid).T).T
                reverse = rows_only(rows_only(grid.T).T)
                _require(np.array_equal(forward, compact)
                         and np.array_equal(reverse, compact), "axis-order mismatch")
                f = _features(grid)
                if f["witness"]:
                    _require(not np.array_equal(out, grid[:2, :2])
                             and not np.array_equal(out, grid[:2, -2:]),
                             "physical crop shortcut survived")
                    _require(not np.array_equal(out, _crop(rows_only(grid))),
                             "row-only shortcut survived")
                    _require(not np.array_equal(out, _crop(rows_only(grid.T).T)),
                             "column-only shortcut survived")
                shapes.add(grid.shape)
                compact_shapes.add(compact.shape[0])
                patterns.add(_normalized_key(grid))
                palettes.add(tuple(int(v) for v in np.unique(grid)))
                thicknesses.add((tuple(heights), tuple(widths)))
                if split == "test":
                    test_parities.add(compact.shape[0] % 2)
                metrics["pairs"] += 1
                metrics["uniform_outputs"] += int(np.unique(out).size == 1)
                metrics["equal_corners"] += int(np.array_equal(compact[:2, :2], compact[:2, -2:]))
                metrics["unit_scale"] += int(max(heights) == max(widths) == 1)
                metrics["one_axis_only_expansion"] += int((max(heights) == 1) != (max(widths) == 1))
                metrics["sparse_compacts"] += int(np.bincount(compact.flat, minlength=10).max() / compact.size >= .75)
                metrics["repeated_vectors"] += int(_stable_unique(compact).shape != compact.shape)
                metrics["height_parity_counterexamples"] += int(f["height_shortcut_fails"])
                metrics["width_parity_counterexamples"] += int(f["width_shortcut_fails"])
    _require(len(puzzles) == 42, "different seeds collided")
    _require(len(shapes) >= 50 and len(patterns) >= 120, "insufficient geometric variation")
    _require(len(compact_shapes) >= 20 and len(palettes) >= 30, "insufficient side/palette variation")
    _require(len(thicknesses) >= 100 and test_parities == {0, 1}, "stretch/parity variation missing")
    _require(all(metrics[k] > 0 for k in ("uniform_outputs", "equal_corners", "unit_scale",
                                         "one_axis_only_expansion", "sparse_compacts", "repeated_vectors")),
             "a supported structural regime was never exercised")
    # Exercise all count combinations, including minimum and maximum boundaries.
    for a, b in _LEGAL_COUNTS:
        validate_generated_puzzle(generate_puzzle(9000 + 10 * a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Algebraic coverage of every compact side, including 2 and 30, without
    # storing 29 additional puzzles. These are also valid solve() inputs.
    for n in range(2, 31):
        r, c = np.indices((n, n))
        compact = ((3 * r + c) % 10).astype(np.int8)
        h = np.ones(n, dtype=int)
        w = np.ones(n, dtype=int)
        if n < 30:
            h[0] = 2
            w[-1] = 2
        inp = np.repeat(np.repeat(compact, h, axis=0), w, axis=1)
        _require(np.array_equal(solve(inp), _crop(compact)), "boundary-side solve failed")
    # Concrete same-canvas dependency examples documented in task_spec.md.
    c5 = np.array([[0, 1, 0, 3, 4], [3, 4, 3, 6, 7], [0, 1, 0, 3, 4],
                   [9, 0, 9, 2, 3], [2, 3, 2, 5, 6]], dtype=np.int8)
    c4 = ((3 * np.arange(4)[:, None] + np.arange(4)[None, :]) % 10).astype(np.int8)
    x5 = np.repeat(np.repeat(c5, [2, 1, 3, 2, 2], axis=0), [2] * 5, axis=1)
    x4 = np.repeat(np.repeat(c4, [2, 3, 2, 3], axis=0), [2, 3, 3, 2], axis=1)
    _require(x5.shape == x4.shape == (10, 10), "illustrative canvases changed")
    _require(np.array_equal(solve(x5), [[0, 1], [3, 4]])
             and np.array_equal(solve(x4), [[2, 3], [5, 6]]), "documented outputs changed")
    _require(np.array_equal(_crop(_stable_unique(c5)), [[3, 4], [6, 7]]),
             "global-unique counterexample changed")
    for bad in (np.zeros((1, 1), dtype=int), np.arange(6).reshape(2, 3),
                np.full((2, 2), 10), np.array([[0., 1.], [2., 3.]])):
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
    # A corrupted answer must not pass merely because the schema is valid.
    corrupt = generate_puzzle(781, 2, 1)
    corrupt["train"][0]["output"][0, 0] = (int(corrupt["train"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupt crop")
    metrics.update(distinct_puzzles=len(puzzles), input_shapes=len(shapes),
                   compact_sides=sorted(compact_shapes), palettes=len(palettes),
                   normalized_inputs=len(patterns), stretch_profiles=len(thicknesses),
                   legal_count_combinations=len(counts))
    return metrics


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
