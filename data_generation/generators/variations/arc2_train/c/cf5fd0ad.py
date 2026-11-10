"""Synthetic whole-tile rotation and repetition puzzles."""

import numpy as np


_SOURCE_INPUTS = (
    ((8, 7, 8), (7, 8, 8), (8, 5, 1)),
    ((6, 8, 8), (8, 6, 8), (8, 8, 8)),
    ((1, 8, 8), (8, 8, 8), (8, 8, 8)),
    ((8, 8, 8), (8, 8, 2), (8, 6, 4)),
)
_SOURCE_KEYS = frozenset((3, tuple(v for row in tile for v in row)) for tile in _SOURCE_INPUTS)
_CANONICAL = (2, 3, 1, 0)  # UL half-turn, UR clockwise, LL counterclockwise, LR original.
_SIZES = np.arange(2, 8)
_SIZE_WEIGHTS = np.array([0.12, 0.35, 0.20, 0.14, 0.10, 0.09])
_MODES = ("isolated", "corner", "line", "band", "cross", "ring",
          "scattered", "dense", "balanced", "multicolor", "symmetric")
_TRAIN_WEIGHTS = np.array([10, 11, 11, 8, 8, 6, 16, 16, 7, 4, 3], dtype=float)
_TEST_WEIGHTS = np.array([9, 10, 11, 8, 8, 7, 14, 14, 7, 5, 7], dtype=float)


def _check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _key(grid):
    return (int(grid.shape[0]), tuple(int(x) for x in grid.flat))


def _structure(grid):
    labels = {}
    result = []
    for value in grid.flat:
        value = int(value)
        if value not in labels:
            labels[value] = len(labels)
        result.append(labels[value])
    return tuple(result)


def _full_orbit(grid):
    return len({_key(np.rot90(grid, k)) for k in range(4)}) == 4


def _resolve_counts(seed, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is not None:
        num_train = int(num_train)
    if num_test is not None:
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def sample_puzzle_parameters(rng):
    """Only quadrant order is a learned convention; colors are copied literally."""
    if rng.random() < 0.65:
        rotations = _CANONICAL
    else:
        rotations = tuple(int(x) for x in rng.permutation(4))
        if rotations == _CANONICAL:
            rotations = (rotations[1], rotations[0], rotations[2], rotations[3])
    shared_background = int(rng.integers(10)) if rng.random() < 0.20 else None
    return {"rotations": rotations, "shared_background": shared_background}


def _paint(rng, size, positions, background, min_colors=1, max_colors=5):
    positions = np.asarray(sorted(set(int(x) for x in positions)), dtype=int)
    tile = np.full((size, size), background, dtype=np.int8)
    if len(positions) == 0:
        return tile
    color_count = int(rng.integers(min_colors, min(max_colors, len(positions), 9) + 1))
    palette = np.asarray(rng.choice([c for c in range(10) if c != background],
                                    size=color_count, replace=False), dtype=np.int8)
    values = np.concatenate((palette, rng.choice(palette, size=len(positions) - color_count)))
    rng.shuffle(values)
    tile.flat[positions] = values
    return tile


def _sample_tile(rng, size, split, shared_background, source_profile=None):
    n = size
    area = n * n
    background = int(rng.integers(10)) if shared_background is None else shared_background
    if source_profile is not None:
        if source_profile == "dense_anchor":
            positions = rng.choice(9, size=4, replace=False)
            return _paint(rng, 3, positions, background, min_colors=3, max_colors=3)
        if source_profile == "isolated":
            position = int(rng.choice([0, 1, 2, 3, 5, 6, 7, 8]))
            return _paint(rng, 3, [position], background)
        r0, c0 = int(rng.integers(2)), int(rng.integers(2))
        block = [r * 3 + c for r in (r0, r0 + 1) for c in (c0, c0 + 1)]
        outer = (0 if r0 == 0 else 2) * 3 + (0 if c0 == 0 else 2)
        positions = [outer] + list(rng.choice([p for p in block if p != outer],
                                                size=2, replace=False))
        return _paint(rng, 3, positions, background, min_colors=3, max_colors=3)
    weights = (_TRAIN_WEIGHTS if split == "train" else _TEST_WEIGHTS).copy()
    if n == 2:
        weights[_MODES.index("cross")] = 0
        weights[_MODES.index("ring")] = 0
    if shared_background is not None:
        for mode in ("band", "cross", "ring", "dense", "balanced", "multicolor"):
            weights[_MODES.index(mode)] = 0
        if n == 2:
            weights[:] = 0
            weights[0] = 1
    weights /= weights.sum()
    mode = str(rng.choice(_MODES, p=weights))

    if mode == "isolated":
        choices = [p for p in range(area) if p != area // 2 or n % 2 == 0]
        positions = [int(rng.choice(choices))]
    elif mode == "corner":
        r0 = int(rng.choice((0, n - 2)))
        c0 = int(rng.choice((0, n - 2)))
        block = [r * n + c for r in (r0, r0 + 1) for c in (c0, c0 + 1)]
        outer = (0 if r0 == 0 else n - 1) * n + (0 if c0 == 0 else n - 1)
        positions = [outer] + [int(x) for x in rng.choice(
            [p for p in block if p != outer], size=int(rng.integers(2, 4)), replace=False)]
    elif mode == "line":
        kind = int(rng.integers(4))
        index = int(rng.integers(n))
        if kind == 0:
            line = [index * n + c for c in range(n)]
        elif kind == 1:
            line = [r * n + index for r in range(n)]
        elif kind == 2:
            line = [r * n + r for r in range(n)]
        else:
            line = [r * n + n - 1 - r for r in range(n)]
        length = int(rng.integers(2, n + 1))
        start = int(rng.integers(0, n - length + 1))
        positions = line[start:start + length]
    elif mode == "band":
        horizontal = bool(rng.integers(2))
        width = int(rng.integers(1, min(2, n - 1) + 1))
        start = int(rng.integers(0, n - width + 1))
        positions = [r * n + c for r in range(n) for c in range(n)
                     if start <= (r if horizontal else c) < start + width]
    elif mode == "cross":
        row = int(rng.choice((n // 2, (n - 1) // 2)))
        col = int(rng.choice((n // 2, (n - 1) // 2)))
        positions = [r * n + c for r in range(n) for c in range(n) if r == row or c == col]
        if rng.random() < 0.4:
            positions.remove(int(rng.choice([p for p in positions if p != row * n + col])))
    elif mode == "ring":
        positions = [r * n + c for r in range(n) for c in range(n)
                     if r in (0, n - 1) or c in (0, n - 1)]
        if rng.random() < 0.55:
            remove = rng.choice(positions, size=int(rng.integers(1, 3)), replace=False)
            positions = [p for p in positions if p not in remove]
    elif mode == "scattered":
        maximum = area - 1 if shared_background is None else max(1, (area - 1) // 2)
        count = int(rng.integers(1, maximum + 1))
        positions = list(rng.choice(area, size=count, replace=False))
    elif mode == "dense":
        count = int(rng.integers(min(5, area), area + 1))
        positions = list(rng.choice(area, size=count, replace=False))
    elif mode == "balanced":
        count = int(rng.integers(2, min(5, area) + 1))
        colors = rng.choice(10, size=count, replace=False).astype(np.int8)
        values = np.resize(colors, area)
        rng.shuffle(values)
        return values.reshape((n, n)).copy()
    elif mode == "multicolor":
        count = min(10, area)
        colors = rng.choice(10, size=count, replace=False).astype(np.int8)
        values = np.resize(colors, area)
        rng.shuffle(values)
        return values.reshape((n, n)).copy()
    else:  # exact half-turn symmetry; some samples also have quarter-turn symmetry.
        pairs = [(p, area - 1 - p) for p in range(area // 2)]
        chosen = rng.choice(len(pairs), size=int(rng.integers(1, min(len(pairs), max(2, area // 3)) + 1)), replace=False)
        tile = np.full((n, n), background, dtype=np.int8)
        palette = rng.choice([c for c in range(10) if c != background],
                             size=int(rng.integers(1, min(4, len(chosen)) + 1)), replace=False)
        for i in chosen:
            a, b = pairs[int(i)]
            tile.flat[[a, b]] = int(rng.choice(palette))
        if n % 2 and rng.random() < 0.25:
            tile.flat[area // 2] = int(rng.choice(palette))
        if shared_background is None or int(np.sum(tile != background)) < area // 2:
            return tile
        return _paint(rng, n, [int(rng.integers(area))], background)

    if shared_background is not None and len(positions) >= area / 2:
        positions = list(rng.choice(positions, size=max(1, (area - 1) // 2), replace=False))
    minimum_colors = 2 if mode in ("dense", "ring", "cross") and len(positions) >= 3 else 1
    return _paint(rng, n, positions, background, min_colors=minimum_colors)


def _construct_output(tile, rotations):
    quadrants = [np.tile(np.rot90(tile, k), (2, 2)) for k in rotations]
    return np.block([[quadrants[0], quadrants[1]],
                     [quadrants[2], quadrants[3]]]).astype(np.int8, copy=False)


def _infer_rotations(pair):
    tile = pair["input"]
    output = pair["output"]
    n = tile.shape[0]
    result = []
    for qr, qc in ((0, 0), (0, 1), (1, 0), (1, 1)):
        quadrant = output[qr * 2 * n:(qr + 1) * 2 * n,
                          qc * 2 * n:(qc + 1) * 2 * n]
        matches = [k for k in range(4)
                   if np.array_equal(quadrant, np.tile(np.rot90(tile, k), (2, 2)))]
        if len(matches) != 1:
            return None
        result.append(matches[0])
    return tuple(result) if set(result) == {0, 1, 2, 3} else None


def _not_near_copy(tile, previous):
    structure = _structure(tile)
    for other in previous:
        if tile.shape != other.shape:
            continue
        if sum(a != b for a, b in zip(structure, _structure(other))) <= 1:
            return False
    return True


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    parameters = sample_puzzle_parameters(rng)
    source_like = train_count >= 3 and rng.random() < 0.18
    train_sizes = [int(rng.choice(_SIZES, p=_SIZE_WEIGHTS))]
    options = _SIZE_WEIGHTS.copy()
    options[train_sizes[0] - 2] = 0
    options /= options.sum()
    train_sizes.append(int(rng.choice(_SIZES, p=options)))
    train_sizes.extend(int(rng.choice(_SIZES, p=_SIZE_WEIGHTS)) for _ in range(train_count - 2))
    test_sizes = [int(rng.choice(_SIZES, p=_SIZE_WEIGHTS)) for _ in range(test_count)]
    train_profiles = {}
    test_profiles = {}
    if source_like:
        dense_index, sparse_index = (int(x) for x in rng.choice(train_count, size=2, replace=False))
        train_profiles[dense_index] = "dense_anchor"
        train_profiles[sparse_index] = "isolated"
        train_sizes[dense_index] = train_sizes[sparse_index] = 3
        other_index = next(i for i in range(train_count) if i not in train_profiles)
        train_sizes[other_index] = int(rng.choice([2, 4, 5, 6, 7]))
        cluster_index = int(rng.integers(test_count))
        test_profiles[cluster_index] = "corner_cluster"
        test_sizes[cluster_index] = 3
    anchor_index = int(rng.integers(train_count))
    puzzle = {"train": [], "test": []}
    prior = []
    for split, sizes in (("train", train_sizes), ("test", test_sizes)):
        for index, size in enumerate(sizes):
            needs_orbit = split == "train" and index == anchor_index
            profile = (train_profiles if split == "train" else test_profiles).get(index)
            candidate = None
            for attempt in range(160):
                selected_size = size if attempt < 120 or profile is not None else 7
                tile = _sample_tile(rng, selected_size, split,
                                    parameters["shared_background"], profile)
                if needs_orbit and not _full_orbit(tile):
                    continue
                if len(np.unique(tile)) < 2:
                    continue
                if _key(tile) in _SOURCE_KEYS or not _not_near_copy(tile, prior):
                    continue
                candidate = tile
                break
            if candidate is None:
                raise RuntimeError("Could not construct a distinct tile")
            prior.append(candidate)
            puzzle[split].append({"input": candidate,
                                  "output": _construct_output(candidate, parameters["rotations"])})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle schema")
    train, test = puzzle["train"], puzzle["test"]
    _check(isinstance(train, list) and isinstance(test, list), "Splits must be lists")
    _check(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "Invalid counts")
    all_pairs = train + test
    prior = []
    for pair in all_pairs:
        _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair schema")
        tile, output = pair["input"], pair["output"]
        for grid in (tile, output):
            _check(isinstance(grid, np.ndarray) and grid.ndim == 2, "Grid must be 2D NumPy")
            _check(np.issubdtype(grid.dtype, np.integer), "Grid must have integer dtype")
            _check(all(1 <= d <= 30 for d in grid.shape), "Grid outside ARC size limits")
            _check(0 <= int(grid.min()) and int(grid.max()) <= 9, "Invalid ARC color")
        n = tile.shape[0]
        _check(2 <= n <= 7 and tile.shape == (n, n), "Input must be a supported square tile")
        _check(output.shape == (4 * n, 4 * n), "Output must be four times wider and taller")
        _check(len(np.unique(tile)) >= 2, "Input must contain a visible motif")
        _check(_key(tile) not in _SOURCE_KEYS, "Source input copied")
        _check(all(_key(tile) != _key(old) for old in prior), "Duplicate input")
        _check(_not_near_copy(tile, prior), "Near-copy structure within puzzle")
        _check(set(int(v) for v in np.unique(tile)) == set(int(v) for v in np.unique(output)),
               "Colors were not preserved")
        prior.append(tile)
    _check(len({p["input"].shape[0] for p in train}) >= 2, "Training must show at least two sizes")
    mappings = [_infer_rotations(pair) for pair in train if _full_orbit(pair["input"])]
    _check(len(mappings) >= 1 and all(m is not None and m == mappings[0] for m in mappings),
           "Training does not expose one consistent four-quadrant orientation rule")
    rotations = mappings[0]
    for pair in all_pairs:
        tile, output = pair["input"], pair["output"]
        n = tile.shape[0]
        for qr, qc, k in ((0, 0, rotations[0]), (0, 1, rotations[1]),
                          (1, 0, rotations[2]), (1, 1, rotations[3])):
            quadrant = output[qr * 2 * n:(qr + 1) * 2 * n,
                              qc * 2 * n:(qc + 1) * 2 * n]
            rotated = np.rot90(tile, k)
            for row in range(2):
                for col in range(2):
                    _check(np.array_equal(quadrant[row * n:(row + 1) * n,
                                                    col * n:(col + 1) * n], rotated),
                           "Quadrant does not repeat the prescribed whole-tile rotation")


def _puzzles_equal(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x[part], y[part])
        for x, y in zip(a[split], b[split]) for part in ("input", "output"))
        for split in ("train", "test"))


def _observed_motif_features(grid):
    n = len(grid)
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    positions = np.argwhere(grid != background)
    features = set()
    if len(positions) == 1:
        features.add("isolated")
    if len(positions) >= 2 and (len(set(positions[:, 0])) == 1
                                or len(set(positions[:, 1])) == 1
                                or len(set(positions[:, 0] - positions[:, 1])) == 1
                                or len(set(positions[:, 0] + positions[:, 1])) == 1):
        features.add("line")
    if n == 3 and len(positions) == 3:
        cells = set(map(tuple, positions))
        if any(cells <= {(r, c) for r in (r0, r0 + 1) for c in (c0, c0 + 1)}
               for r0 in (0, 1) for c0 in (0, 1)):
            features.add("corner_cluster")
    if n == 3 and int(counts.max()) <= 4:
        features.add("dense")
    if n == 3 and int(counts.max()) <= 3:
        features.add("balanced")
    if n == 3 and len(values) == 9:
        features.add("multicolor")
    if len(positions) and int(np.unique(grid[grid != background], return_counts=True)[1].max()) >= 2:
        features.add("repeated_accent")
    if not _full_orbit(grid):
        features.add("symmetric")
    if n >= 3:
        for color in values:
            mask = grid != color
            border = np.zeros((n, n), dtype=bool)
            border[0, :] = border[-1, :] = True
            border[:, 0] = border[:, -1] = True
            if np.array_equal(mask, border):
                features.add("ring")
            for index in range(n):
                row = np.zeros((n, n), dtype=bool)
                row[index, :] = True
                column = np.zeros((n, n), dtype=bool)
                column[:, index] = True
                if np.array_equal(mask, row) or np.array_equal(mask, column):
                    features.add("band")
                for other in range(n):
                    cross = row | (np.arange(n)[None, :] == other)
                    if np.array_equal(mask, cross):
                        features.add("cross")
    return features


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _check(len({len(p["train"]) for p in puzzles}) >= 4, "Train counts collapsed")
    _check(len({len(p["test"]) for p in puzzles}) >= 3, "Test counts collapsed")
    _check(_puzzles_equal(generate_puzzle(17), generate_puzzle(17)), "Nondeterministic automatic call")
    _check(_puzzles_equal(generate_puzzle(55, 8, 3), generate_puzzle(55, 8, 3)),
           "Nondeterministic explicit call")
    for seed, nt, ns in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                         (104, None, 1), (105, None, 4), (106, 2, None), (107, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        _check(nt is None or len(puzzle["train"]) == nt, "Training request ignored")
        _check(ns is None or len(puzzle["test"]) == ns, "Test request ignored")
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 2, "num_test": 3}, {"num_train": True},
                    {"num_train": 2.0}):
        try:
            generate_puzzle(88, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Illegal counts accepted: {request}")
    pairs = [pair for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split]]
    sizes = {pair["input"].shape[0] for pair in pairs}
    _check(sizes == set(range(2, 8)), "Square size range collapsed")
    mappings = {_infer_rotations(next(pair for pair in puzzle["train"] if _full_orbit(pair["input"])))
                for puzzle in puzzles}
    _check(_CANONICAL in mappings and len(mappings) >= 3, "Quadrant assignments collapsed")
    _check(any(not _full_orbit(pair["input"]) for puzzle in puzzles for pair in puzzle["test"]),
           "Symmetric test motifs absent")
    _check(any(pair["input"].shape == (3, 3) and len(np.unique(pair["input"])) >= 8
               for pair in pairs), "Fully multicolor 3x3 motif absent")
    _check(any(len(np.unique(pair["input"])) >= 5 for pair in pairs),
           "High-color motifs absent")
    _check(any(pair["input"].shape == (3, 3)
               and int(np.unique(pair["input"], return_counts=True)[1].max()) <= 3
               for pair in pairs), "Balanced 3x3 motif absent")
    train_profiles = {(pair["input"].shape[0], len(np.unique(pair["input"])))
                      for puzzle in puzzles for pair in puzzle["train"]}
    test_profiles = {(pair["input"].shape[0], len(np.unique(pair["input"])))
                     for puzzle in puzzles for pair in puzzle["test"]}
    _check(len(train_profiles & test_profiles) >= 6,
           "Train and test profiles do not overlap")
    dominant_sets = [
        {int(np.unique(pair["input"], return_counts=True)[0][
            np.argmax(np.unique(pair["input"], return_counts=True)[1])])
         for split in ("train", "test") for pair in puzzle[split]}
        for puzzle in puzzles
    ]
    _check(sum(len(colors) > 1 for colors in dominant_sets) >= 15,
           "Pair-specific color roles absent")
    _check(sum(len(colors) == 1 for colors in dominant_sets) >= 2,
           "Constant-background examples absent")
    def profile(grid):
        return (len(grid), int(np.unique(grid, return_counts=True)[1].max()),
                len(np.unique(grid)))
    _check(any((3, 5, 4) in [profile(pair["input"]) for pair in puzzle["train"]]
               and (3, 8, 2) in [profile(pair["input"]) for pair in puzzle["train"]]
               and (3, 6, 4) in [profile(pair["input"]) for pair in puzzle["test"]]
               for puzzle in puzzles), "Source-like progression absent")
    train_features = set().union(*(_observed_motif_features(pair["input"])
                                   for puzzle in puzzles for pair in puzzle["train"]))
    test_features = set().union(*(_observed_motif_features(pair["input"])
                                  for puzzle in puzzles for pair in puzzle["test"]))
    _check({"isolated", "line", "corner_cluster", "dense", "balanced",
            "repeated_accent", "symmetric"} <= train_features & test_features,
           "Train and test motif families do not overlap")
    _check({"band", "cross", "ring", "multicolor"} <= train_features | test_features,
           "Structured or fully multicolor motifs absent")
