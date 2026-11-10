"""Generate exact, unambiguous two-copy ARC puzzles."""

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _pair_key(pair: dict) -> tuple:
    return (_key(pair["input"]), _key(pair["output"]))


def _source_pair(input_rows: tuple[str, ...], output_rows: tuple[str, ...]) -> tuple:
    return (_key(np.array([[int(c) for c in row] for row in input_rows])),
            _key(np.array([[int(c) for c in row] for row in output_rows])))


_SOURCE_PAIRS = frozenset({
    _source_pair(("444444", "648648", "668668"), ("444", "648", "668")),
    _source_pair(("11321132", "11331133", "33113311", "23112311"),
                 ("1132", "1133", "3311", "2311")),
    _source_pair(("23", "32", "44", "23", "32", "44"),
                 ("23", "32", "44")),
    _source_pair(("545", "454", "664", "262", "545", "454", "664", "262"),
                 ("545", "454", "664", "262")),
})


def _matching_halves(grid: np.ndarray) -> list[tuple[str, np.ndarray]]:
    h, w = grid.shape
    matches = []
    if w % 2 == 0 and np.array_equal(grid[:, :w // 2], grid[:, w // 2:]):
        matches.append(("horizontal", grid[:, :w // 2]))
    if h % 2 == 0 and np.array_equal(grid[:h // 2], grid[h // 2:]):
        matches.append(("vertical", grid[:h // 2]))
    return matches


def _count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _resolve_counts(seed: int, train: int | None, test: int | None) -> tuple[int, int]:
    train = _count(train, "num_train", 2, 10)
    test = _count(test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = max(2, test + 1) + (seed // 4) % (11 - max(2, test + 1))
    elif train is None:
        train = max(2, test + 1) + seed % (11 - max(2, test + 1))
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _sample_length(rng: np.random.Generator, maximum: int) -> int:
    roll = rng.random()
    if roll < 0.14:
        return int(rng.integers(1, 4))
    if roll < 0.72:
        return int(rng.integers(2, 13))
    if maximum > 15 and roll > 0.84:
        return int(rng.integers(16, maximum + 1))
    return int(rng.integers(10, 16))


def _sample_shape(rng: np.random.Generator, orientation: str) -> tuple[int, int]:
    h = _sample_length(rng, 30 if orientation == "horizontal" else 15)
    w = _sample_length(rng, 15 if orientation == "horizontal" else 30)
    return h, w


def _motif(rng: np.random.Generator, h: int, w: int,
           palette_size: int, seamless: bool) -> np.ndarray:
    palette = rng.choice(10, size=palette_size, replace=False).astype(np.int8)
    if palette_size == 1:
        return np.full((h, w), palette[0], dtype=np.int8)
    rr, cc = np.indices((h, w))
    style = int(rng.integers(0, 7))
    if style == 0:  # dense irregular cells
        indices = rng.integers(0, palette_size, (h, w))
    elif style == 1:  # diagonal modular lattice
        a, b = rng.integers(1, palette_size + 1, size=2)
        indices = (a * rr + b * cc + int(rng.integers(palette_size))) % palette_size
    elif style == 2:  # overlapping filled rectangles
        indices = np.full((h, w), int(rng.integers(palette_size)))
        for k in rng.permutation(palette_size):
            r0, c0 = int(rng.integers(h)), int(rng.integers(w))
            r1, c1 = int(rng.integers(r0 + 1, h + 1)), int(rng.integers(c0 + 1, w + 1))
            indices[r0:r1, c0:c1] = int(k)
    elif style == 3:  # wide alternating bands
        rh, cw = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        indices = (rr // rh + cc // cw) % palette_size
    elif style == 4:  # sparse points on a field
        indices = np.full((h, w), int(rng.integers(palette_size)))
        n = max(palette_size, (h * w) // int(rng.integers(5, 12)))
        for pos in rng.choice(h * w, size=min(n, h * w), replace=False):
            indices.flat[int(pos)] = int(rng.integers(palette_size))
    elif style == 5:  # connected paths and strokes
        indices = np.zeros((h, w), dtype=np.int64)
        for color in range(1, palette_size):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            for _ in range(max(2, (h + w) // 2)):
                indices[r, c] = color
                r = int(np.clip(r + int(rng.integers(-1, 2)), 0, h - 1))
                c = int(np.clip(c + int(rng.integers(-1, 2)), 0, w - 1))
    else:  # checker, stripes, and larger regular tiles
        rh, cw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        indices = (rr // rh + cc // cw + int(rng.integers(palette_size))) % palette_size

    # The chosen number of colors is the number actually visible, not merely sampled.
    missing = set(range(palette_size)) - set(int(x) for x in np.unique(indices))
    if missing:
        for position, color in zip(rng.permutation(h * w), sorted(missing)):
            indices.flat[int(position)] = color
    motif = np.asarray(palette[indices], dtype=np.int8)
    if seamless:
        if rng.random() < 0.5:
            motif[:, -1] = motif[:, 0]
        else:
            motif[-1] = motif[0]
    return motif


def _make_pair(rng: np.random.Generator, orientation: str,
               used: set[tuple], shape: tuple[int, int] | None = None,
               palette_size: int | None = None, motif: np.ndarray | None = None) -> dict:
    for _ in range(256):
        h, w = shape if shape is not None else _sample_shape(rng, orientation)
        if (orientation == "horizontal" and w > 15) or (orientation == "vertical" and h > 15):
            shape = None
            continue
        colors = palette_size if palette_size is not None else int(rng.choice(
            [1, 2, 3, 4, 5, 6], p=[0.09, 0.20, 0.25, 0.20, 0.16, 0.10]))
        if h * w < colors:
            if shape is not None:
                palette_size = None
            continue
        block = motif.copy() if motif is not None else _motif(rng, h, w, colors, rng.random() < 0.28)
        if len(np.unique(block)) != colors:
            continue
        inp = np.concatenate((block, block), axis=1 if orientation == "horizontal" else 0)
        pair = {"input": inp, "output": block.copy()}
        key = _key(inp)
        if key in used or _pair_key(pair) in _SOURCE_PAIRS:
            continue
        matches = _matching_halves(inp)
        if len(matches) == 1 and matches[0][0] == orientation:
            used.add(key)
            return pair
    # Fixed tiny/even shapes can make a requested palette or axis impossible.
    if shape is not None or motif is not None or palette_size is not None:
        return _make_pair(rng, orientation, used)
    raise RuntimeError("Unable to construct a unique split")


def _orientations(rng: np.random.Generator, train_count: int,
                  test_count: int) -> tuple[list[str], list[str]]:
    mode = int(rng.integers(5))
    if mode < 2:
        axis = ("horizontal", "vertical")[mode]
        return [axis] * train_count, [axis] * test_count
    if mode == 2:
        vertical_count = int(rng.integers(1, train_count))
    elif mode == 3:
        vertical_count = int(rng.integers(1, max(2, train_count // 2 + 1)))
    else:
        vertical_count = train_count - int(rng.integers(1, max(2, train_count // 2 + 1)))
    vertical_count = min(train_count - 1, max(1, vertical_count))
    train = ["vertical"] * vertical_count + ["horizontal"] * (train_count - vertical_count)
    rng.shuffle(train)
    test = ["horizontal" if rng.random() < 0.5 else "vertical" for _ in range(test_count)]
    if test_count >= 2 and rng.random() < 0.5:
        test[0], test[1] = "horizontal", "vertical"
        rng.shuffle(test)
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(int(seed), num_train, num_test)
    train_axes, test_axes = _orientations(rng, train_count, test_count)
    used: set[tuple] = set()
    train: list[dict] = []
    test: list[dict] = []

    # A comparison of the identical motif across axes makes the copying rule
    # especially clear in some mixed puzzles. The motif is sampled once for it.
    paired = (len(set(train_axes)) == 2 and rng.random() < 0.30)
    if paired:
        h, w = int(rng.integers(2, 13)), int(rng.integers(2, 13))
        colors = int(rng.integers(2, min(6, h * w) + 1))
        for _ in range(128):
            common = _motif(rng, h, w, colors, rng.random() < 0.35)
            if len(np.unique(common)) == colors and all(
                len(_matching_halves(np.concatenate((common, common), axis=axis))) == 1
                for axis in (0, 1)
            ):
                break
        else:
            paired = False
    first_axis_index: dict[str, int] = {}
    if paired:
        for index, axis in enumerate(train_axes):
            if axis not in first_axis_index:
                first_axis_index[axis] = index
    for index, axis in enumerate(train_axes):
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        if paired and index in first_axis_index.values():
            pair = _make_pair(pair_rng, axis, used, common.shape, colors, common)
        else:
            shared_shape = None
            if train and rng.random() < 0.23:
                candidate = train[int(rng.integers(len(train)))]["output"].shape
                if (axis == "horizontal" and candidate[1] <= 15) or (axis == "vertical" and candidate[0] <= 15):
                    shared_shape = candidate
            pair = _make_pair(pair_rng, axis, used, shape=shared_shape)
        train.append(pair)
    for axis in test_axes:
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        shared_shape = None
        if rng.random() < 0.32:
            candidate = train[int(rng.integers(len(train)))]["output"].shape
            if (axis == "horizontal" and candidate[1] <= 15) or (axis == "vertical" and candidate[0] <= 15):
                shared_shape = candidate
        test.append(_make_pair(pair_rng, axis, used, shape=shared_shape))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "invalid splits")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid counts")
    keys: set[tuple] = set()
    pair_keys: set[tuple] = set()
    axes: dict[str, set[str]] = {"train": set(), "test": set()}
    for split in ("train", "test"):
        for i, pair in enumerate(puzzle[split]):
            place = f"{split}[{i}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, f"{place} keys")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, f"{place}.{name} type")
                h, w = grid.shape
                _require(1 <= h <= 30 and 1 <= w <= 30, f"{place}.{name} shape")
                _require(np.issubdtype(grid.dtype, np.integer), f"{place}.{name} dtype")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), f"{place}.{name} colors")
            matches = _matching_halves(pair["input"])
            _require(len(matches) == 1, f"{place} ambiguous split")
            axis, expected = matches[0]
            axes[split].add(axis)
            _require(np.array_equal(pair["output"], expected), f"{place} incorrect output")
            _require(np.array_equal(np.unique(pair["input"]), np.unique(pair["output"])), f"{place} palette changed")
            _require(_pair_key(pair) not in _SOURCE_PAIRS, f"{place} copies source")
            input_key, pair_key = _key(pair["input"]), _pair_key(pair)
            _require(input_key not in keys and pair_key not in pair_keys, f"{place} duplicate")
            keys.add(input_key)
            pair_keys.add(pair_key)
    _require(axes["test"] <= axes["train"], "training does not demonstrate a test axis")


def _same_puzzle(a: dict, b: dict) -> bool:
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[n], y[n]) for x, y in zip(a[s], b[s]) for n in ("input", "output")
    ) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        demonstrated = set()
        for pair in puzzle["train"]:
            grid = pair["input"].copy()
            answers = {}
            if grid.shape[1] % 2 == 0:
                left, right = np.hsplit(grid, 2)
                if np.array_equal(left, right):
                    answers["horizontal"] = left
            if grid.shape[0] % 2 == 0:
                top, bottom = np.vsplit(grid, 2)
                if np.array_equal(top, bottom):
                    answers["vertical"] = top
            _require(len(answers) == 1, "independent training split")
            demonstrated.update(answers)
            _require(np.array_equal(pair["output"], next(iter(answers.values()))),
                     "independent training answer")
        for pair in puzzle["test"]:
            grid = pair["input"].copy()
            answers = []
            if "horizontal" in demonstrated and grid.shape[1] % 2 == 0:
                left, right = np.hsplit(grid, 2)
                if np.array_equal(left, right):
                    answers.append(left)
            if "vertical" in demonstrated and grid.shape[0] % 2 == 0:
                top, bottom = np.vsplit(grid, 2)
                if np.array_equal(top, bottom):
                    answers.append(top)
            _require(len(answers) == 1 and np.array_equal(pair["output"], answers[0]),
                     "training-only test prediction")
    for train, test in ((2, 1), (10, 1), (10, 4), (7, 3)):
        puzzle = generate_puzzle(20000 + train + test, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test), "explicit counts")
    for args in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(31000, *args)
        _require((args[0] is None or len(puzzle["train"]) == args[0]) and
                 (args[1] is None or len(puzzle["test"]) == args[1]), "partial counts")
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3)):
        try:
            generate_puzzle(41000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(_same_puzzle(generate_puzzle(8675309, 8, 3), generate_puzzle(8675309, 8, 3)), "determinism")
    _require(_same_puzzle(generate_puzzle(314159), generate_puzzle(314159)), "automatic determinism")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    all_pairs = [pair for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split]]
    _require(len({_pair_key(pair) for pair in all_pairs}) >= len(all_pairs) * 9 // 10, "cross-seed grid diversity")
    _require(len({pair["output"].shape for pair in all_pairs}) >= 35, "shape diversity")
    _require(len({(_matching_halves(pair["input"])[0][0], pair["output"].shape,
                  len(np.unique(pair["output"]))) for pair in all_pairs}) >= 70, "layout diversity")
    _require({"horizontal", "vertical"} == {a for pair in all_pairs for a, _ in _matching_halves(pair["input"])}, "axis diversity")
    _require(len({len(np.unique(pair["output"])) for pair in all_pairs}) >= 5, "palette-size diversity")
    # Inspect returned grids, including cross-pair relations rather than generation metadata.
    test_axes = [{_matching_halves(q["input"])[0][0] for q in p["test"]} for p in puzzles]
    _require(any(x == {"horizontal"} for x in test_axes) and any(x == {"vertical"} for x in test_axes), "test-axis diversity")
    _require(any(x == {"horizontal", "vertical"} for x in test_axes), "mixed tests")
    _require(any(len({_matching_halves(q["input"])[0][0] for q in p["train"]}) == 1 for p in puzzles), "simple puzzles")
    _require(any(len({len(np.unique(q["output"])) for q in p["train"]}) > 1 for p in puzzles), "within-train palette diversity")
    _require(any(q["output"].shape in {r["output"].shape for r in p["train"]}
                 for p in puzzles for q in p["test"]), "shape overlap")
    _require(any(q["output"].shape[0] <= max(r["output"].shape[0] for r in p["train"])
                 for p in puzzles for q in p["test"]), "non-increasing test height")
    _require(any(_matching_halves(q["input"])[0][0] == "vertical" and q["input"].shape[1] > q["input"].shape[0]
                 for q in all_pairs), "wide vertical input")
    _require(any(_matching_halves(q["input"])[0][0] == "horizontal" and q["input"].shape[0] > q["input"].shape[1]
                 for q in all_pairs), "tall horizontal input")
    train_sets = [p["train"] for p in puzzles]
    test_pairs = [(p, q) for p in puzzles for q in p["test"]]
    axis = lambda q: _matching_halves(q["input"])[0][0]
    _require(any(all(axis(q) == "horizontal" for q in t) for t in train_sets)
             and any(all(axis(q) == "vertical" for q in t) for t in train_sets)
             and any({axis(q) for q in t} == {"horizontal", "vertical"} for t in train_sets),
             "training regime diversity")
    _require(any(axis(q) != axis(r) and np.array_equal(q["output"], r["output"])
                 for t in train_sets for i, q in enumerate(t) for r in t[i + 1:]),
             "same block across axes")
    _require(any(q["output"].shape == r["output"].shape and
                 not np.array_equal(q["output"], r["output"])
                 for t in train_sets for i, q in enumerate(t) for r in t[i + 1:]),
             "same shape with new motif")
    heights = [q["output"].shape[0] - max(r["output"].shape[0] for r in p["train"])
               for p, q in test_pairs]
    _require(any(h < 0 for h in heights) and any(h == 0 for h in heights)
             and any(h > 0 for h in heights), "test height relationships")
    color_deltas = [len(np.unique(q["output"])) - np.median(
        [len(np.unique(r["output"])) for r in p["train"]]) for p, q in test_pairs]
    _require(any(d < 0 for d in color_deltas) and any(d == 0 for d in color_deltas)
             and any(d > 0 for d in color_deltas), "test palette relationships")
    seams = [np.array_equal(q["output"][:, 0], q["output"][:, -1]) if axis(q) == "horizontal"
             else np.array_equal(q["output"][0], q["output"][-1]) for q in all_pairs]
    _require(any(seams) and not all(seams), "join appearance diversity")
    _require(any(len(np.unique(q["output"])) == 1 for q in all_pairs)
             and any(max(q["output"].shape) <= 3 for q in all_pairs), "simple motif diversity")
    _require(any(axis(q) == "vertical" and q["input"].shape[1] > q["input"].shape[0]
                 for _, q in test_pairs)
             and any(axis(q) == "horizontal" and q["input"].shape[0] > q["input"].shape[1]
                     for _, q in test_pairs), "test aspect versus axis diversity")
