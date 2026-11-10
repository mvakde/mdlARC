import numpy as np


_AUTO_COUNTS = tuple(
    (a, b) for a in range(2, 11) for b in range(1, min(4, a - 1) + 1)
)

_SOURCE = (
    ((7, 5, 8, 2, 8, 0, 4, 7, 1, 6, 4, 7, 8, 9, 6, 9), 4),
    ((0, 7, 9, 5, 6, 5, 3, 7, 9), 3),
    ((5, 8, 8, 9, 2, 8, 0, 5, 6, 5, 7, 7, 2, 2, 9,
      5, 5, 1, 7, 4, 3, 3, 8, 7, 7), 5),
    ((3, 6, 0, 2, 8, 7, 9, 2, 6, 7, 5, 6, 3, 7, 3, 4,
      8, 0, 8, 6, 3, 0, 8, 3, 8, 8, 5, 9, 0, 1, 6, 7,
      7, 6, 9, 7, 8, 7, 4, 3, 7, 3, 8, 8, 3, 7, 6, 1,
      3, 7, 0, 7, 7, 0, 5, 1, 8, 7, 5, 2, 7, 7, 6, 6), 8),
)
_SOURCE_INPUTS = {(side, cells) for cells, side in _SOURCE}


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _counts(seed, rng, num_train, num_test):
    for name, value, lo, hi in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or not lo <= value <= hi
        ):
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    if num_train is None and num_test is None:
        num_train, num_test = _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _crop(grid):
    return grid[:2, :2].copy() if grid.shape[0] % 2 else grid[:2, -2:].copy()


def _signature(grid):
    return grid.shape, tuple(int(v) for v in grid.flat)


def _train_sides(rng, count):
    odd = [3, 5, 7, 9, 11]
    even = [4, 6, 8, 10, 12]
    if rng.random() < 0.25:
        sides = [int(rng.choice([3, 5])), 4]
    else:
        sides = [int(rng.choice(odd)), int(rng.choice(even))]
    if count >= 4:
        sides += [
            int(rng.choice([v for v in odd if v != sides[0]])),
            int(rng.choice([v for v in even if v != sides[1]])),
        ]
    while len(sides) < count:
        sides.append(int(rng.integers(2, 15)))
    rng.shuffle(sides)
    return sides


def _test_side(rng, parity=None):
    # Small and overlapping sizes remain valid; large inputs carry the source shift.
    band = int(rng.choice(3, p=[0.35, 0.25, 0.40]))
    lo, hi = ((2, 12), (13, 20), (21, 30))[band]
    choices = [v for v in range(lo, hi + 1) if parity is None or v % 2 == parity]
    return int(rng.choice(choices))


def _test_sides(rng, count):
    if count >= 2:
        sides = [_test_side(rng, 1), _test_side(rng, 0)]
    else:
        sides = [_test_side(rng)]
    while len(sides) < count:
        sides.append(_test_side(rng))
    rng.shuffle(sides)
    return sides


def _palette(rng, side, witness):
    weights = np.array([.06, .12, .12, .11, .10, .10, .10, .10, .09, .10])
    if witness:
        weights[0] = 0
    weights[min(10, side * side):] = 0
    weights /= weights.sum()
    size = int(rng.choice(np.arange(1, 11), p=weights))
    return rng.choice(np.arange(10, dtype=np.int8), size=size, replace=False)


def _texture(rng, side, palette):
    k = len(palette)
    rows, cols = np.indices((side, side))
    mode = int(rng.integers(0, 8))
    if mode == 0:  # dense independent cells
        labels = rng.integers(0, k, size=(side, side))
    elif mode == 1:  # independent cells with a dominant color
        weights = rng.dirichlet(np.full(k, float(rng.choice([.25, .5, 1.0]))))
        labels = rng.choice(k, size=(side, side), p=weights)
    elif mode == 2:  # clean periodic tile
        h = int(rng.integers(1, min(4, side) + 1))
        w = int(rng.integers(1, min(4, side) + 1))
        tile = rng.integers(0, k, size=(h, w))
        labels = tile[rows % h, cols % w]
    elif mode == 3:  # bands of independently sampled widths
        rh = int(rng.integers(1, min(5, side) + 1))
        cw = int(rng.integers(1, min(5, side) + 1))
        labels = (int(rng.integers(k)) + rows // rh + int(rng.integers(1, k + 1)) * (cols // cw)) % k
    elif mode == 4:  # solid field with a small rectangular motif
        labels = np.full((side, side), int(rng.integers(k)))
        for _ in range(int(rng.integers(1, 4))):
            h = int(rng.integers(1, min(4, side) + 1))
            w = int(rng.integers(1, min(4, side) + 1))
            r = int(rng.integers(0, side - h + 1))
            c = int(rng.integers(0, side - w + 1))
            labels[r:r + h, c:c + w] = int(rng.integers(k))
    elif mode == 5:  # sparse colored cells
        labels = np.full((side, side), int(rng.integers(k)))
        n = int(rng.integers(1, max(2, side * side // 9 + 1)))
        positions = rng.choice(side * side, size=n, replace=False)
        labels.flat[positions] = rng.integers(0, k, size=n)
    elif mode == 6:  # arithmetic pattern
        labels = (int(rng.integers(1, k + 1)) * rows +
                  int(rng.integers(1, k + 1)) * cols + int(rng.integers(k))) % k
    else:  # reflected field
        labels = rng.integers(0, k, size=(side, side))
        if rng.random() < .5:
            labels[:, side // 2:] = np.flip(labels[:, :side - side // 2], axis=1)
        else:
            labels[side // 2:, :] = np.flip(labels[:side - side // 2, :], axis=0)
    grid = palette[labels].astype(np.int8, copy=True)
    if k > 1 and mode in (2, 3, 6, 7) and rng.random() < .32:
        n = int(rng.integers(1, max(2, side * side // 12 + 1)))
        pos = rng.choice(side * side, size=n, replace=False)
        grid.flat[pos] = rng.choice(palette, size=n)
    return grid


def _paint_corner(rng, grid, left, palette, richness):
    side = grid.shape[0]
    start = 0 if left else side - 2
    number = min(richness, len(palette))
    colors = rng.choice(palette, size=number, replace=False)
    if number == 1:
        patch = np.full((2, 2), colors[0], dtype=np.int8)
    else:
        positions = rng.permutation(4)
        patch = np.full(4, colors[0], dtype=np.int8)
        patch[positions[:number]] = colors
        patch = patch.reshape(2, 2)
    grid[:2, start:start + 2] = patch


def _input(rng, side, witness):
    palette = _palette(rng, side, witness)
    grid = _texture(rng, side, palette)
    selected_left = bool(side % 2)

    # The appearance of either candidate is independent of which one is copied.
    if len(palette) > 1 and side > 2 and rng.random() < .58:
        a = int(rng.choice([1, 2, 3, 4]))
        b = int(rng.choice([1, 2, 3, 4]))
        _paint_corner(rng, grid, not selected_left, palette, b)
        _paint_corner(rng, grid, selected_left, palette, a)

    # Sometimes make either the selected or decoy color common elsewhere.
    if len(palette) > 1 and side >= 4 and rng.random() < .16:
        counts = np.bincount(grid[2:].flat, minlength=10)
        common = int(palette[np.argmax(counts[palette])])
        rare = int(palette[np.argmin(counts[palette])])
        if counts[common] > counts[rare]:
            favored_left = selected_left if rng.random() < .5 else not selected_left
            grid[:2, :2] = common if favored_left else rare
            grid[:2, -2:] = rare if favored_left else common

    if not witness and side >= 4 and rng.random() < .17:
        if selected_left:
            grid[:2, -2:] = grid[:2, :2]
        else:
            grid[:2, :2] = grid[:2, -2:]
    elif witness:
        left, right = grid[:2, :2], grid[:2, -2:]
        if np.array_equal(left, right):
            decoy_col = side - 1 if selected_left else 0
            current = int(grid[0, decoy_col])
            grid[0, decoy_col] = next(int(v) for v in palette if v != current)

    if side >= 4 and rng.random() < .19:
        col = int(rng.integers(0, side - 1))
        grid[-2:, col:col + 2] = _crop(grid)

    # All ten colors occur in this deliberately dense palette case.
    if len(palette) == 10 and side >= 6:
        missing = [int(v) for v in palette if v not in grid]
        if missing:
            positions = rng.choice(np.arange(2 * side, side * side),
                                   size=len(missing), replace=False)
            grid.flat[positions] = missing
    return grid


def _unique_pair(rng, side, witness, used):
    for _ in range(96):
        grid = _input(rng, side, witness)
        key = _signature(grid)
        if key in used or key in _SOURCE_INPUTS:
            continue
        used.add(key)
        return {"input": grid, "output": _crop(grid)}
    raise RuntimeError("could not construct a unique input")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if (not isinstance(seed, (int, np.integer))
            or isinstance(seed, (bool, np.bool_)) or seed < 0):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    # The source has one fixed convention: odd left, even right, 2x2 output.
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    train_sides = _train_sides(rng, train_count)
    test_sides = _test_sides(rng, test_count)
    used = set()
    train = []
    seen_witness_parities = set()
    for side in train_sides:
        witness = side > 2 and side % 2 not in seen_witness_parities
        if witness:
            seen_witness_parities.add(side % 2)
        child = np.random.default_rng(int(rng.integers(0, 2**63, dtype=np.int64)))
        train.append(_unique_pair(child, side, witness, used))
    test = []
    for side in test_sides:
        child = np.random.default_rng(int(rng.integers(0, 2**63, dtype=np.int64)))
        test.append(_unique_pair(child, side, False, used))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _learn_corners(train):
    learned = {0: set(), 1: set()}
    for pair in train:
        grid, out = pair["input"], pair["output"]
        left = np.array_equal(out, grid[:2, :2])
        right = np.array_equal(out, grid[:2, -2:])
        if left != right:
            learned[grid.shape[0] % 2].add("left" if left else "right")
    return learned


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "invalid pair counts")
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs input and output")
            grid, out = pair["input"], pair["output"]
            for value in (grid, out):
                _require(isinstance(value, np.ndarray) and value.ndim == 2,
                         "grid must be a 2D ndarray")
                _require(np.issubdtype(value.dtype, np.integer), "grid dtype must be integer")
                _require(all(1 <= v <= 30 for v in value.shape), "grid dimensions out of bounds")
                _require(bool(np.all((0 <= value) & (value <= 9))), "illegal color")
            side = grid.shape[0]
            _require(grid.shape == (side, side) and side >= 2, "input must be square")
            _require(out.shape == (2, 2), "output must be 2x2")
            _require(np.array_equal(out, _crop(grid)), "incorrect parity crop")
            key = _signature(grid)
            _require(key not in seen, "duplicate input or pair")
            _require(key not in _SOURCE_INPUTS, "source pair copied")
            seen.add(key)
    learned = _learn_corners(puzzle["train"])
    _require(learned == {0: {"right"}, 1: {"left"}},
             "training must distinguish both parity branches")
    # Infer every test answer from training examples and its input alone.
    for pair in puzzle["test"]:
        grid = pair["input"]
        direction = next(iter(learned[grid.shape[0] % 2]))
        prediction = grid[:2, :2] if direction == "left" else grid[:2, -2:]
        _require(np.array_equal(pair["output"], prediction),
                 "test cannot be predicted from training convention")
    if n >= 4:
        sizes = {p: {x["input"].shape[0] for x in puzzle["train"]
                     if x["input"].shape[0] % 2 == p} for p in (0, 1)}
        _require(all(len(v) >= 2 for v in sizes.values()),
                 "larger training sets need multiple sizes of each parity")


def _puzzle_key(puzzle):
    return tuple(tuple((_signature(p["input"]), _signature(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _require(len({_puzzle_key(p) for p in puzzles}) >= 40, "puzzle diversity")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    records = [(split, pair["input"], pair["output"])
               for puzzle in puzzles for split in ("train", "test")
               for pair in puzzle[split]]
    sizes = {g.shape[0] for _, g, _ in records}
    test_sizes = {g.shape[0] for split, g, _ in records if split == "test"}
    palettes = {tuple(np.unique(g)) for _, g, _ in records}
    _require(len(sizes) >= 20 and len(test_sizes) >= 15, "size diversity")
    _require(len(palettes) >= 30, "palette diversity")
    _require(any(g.shape[0] == 6 for split, g, _ in records if split == "train"),
             "size-six training absent")
    _require(any(g.shape[0] % 2 for split, g, _ in records if split == "test") and
             any(g.shape[0] % 2 == 0 for split, g, _ in records if split == "test"),
             "test parity diversity")
    _require(any(len(np.unique(out)) == 1 for _, _, out in records),
             "uniform outputs absent")
    _require(any(g.shape[0] >= 4 and np.array_equal(g[:2, :2], g[:2, -2:])
                 for _, g, _ in records), "equal candidate corners absent")
    _require(any(len(np.unique(g)) <= 3 for _, g, _ in records) and
             any(len(np.unique(g)) == 10 for _, g, _ in records),
             "palette extremes absent")
    _require(any(g.shape[0] == 2 for _, g, _ in records),
             "whole-grid boundary absent")
    _require(any(0 not in g for _, g, _ in records),
             "zero-free palettes absent")
    _require(all(len({tuple(np.unique(x["input"])) for split in ("train", "test")
                      for x in p[split]}) >= 2 for p in puzzles),
             "pair palettes do not change within puzzles")
    _require(sum({x["input"].shape[0] % 2 for x in p["test"]} == {0, 1}
                 for p in puzzles) >= 10, "mixed-parity tests too rare")
    _require(sum(any(x["input"].shape[0] <= max(y["input"].shape[0]
                                                   for y in p["train"])
                     for x in p["test"]) for p in puzzles) >= 10,
             "test sizes never overlap training scale")
    _require(any(split == "test" and g.shape[0] in (7, 11)
                 for split, g, _ in records), "odd medium tests absent")
    selected_richer = decoy_richer = repeated = periodic = sparse = dense = 0
    selected_common = decoy_common = 0
    combined = 0
    for split, g, out in records:
        n = g.shape[0]
        decoy = g[:2, -2:] if n % 2 else g[:2, :2]
        selected_richer += len(np.unique(out)) > len(np.unique(decoy))
        decoy_richer += len(np.unique(out)) < len(np.unique(decoy))
        if n >= 4:
            lower_counts = np.bincount(g[2:].flat, minlength=10)
            selected_frequency = np.mean(lower_counts[np.unique(out)])
            decoy_frequency = np.mean(lower_counts[np.unique(decoy)])
            selected_common += selected_frequency > decoy_frequency
            decoy_common += selected_frequency < decoy_frequency
        repeat = (n >= 4 and len(np.unique(out)) > 1 and
                  any(np.array_equal(g[r:r + 2, c:c + 2], out)
                      for r in range(2, n - 1) for c in range(n - 1)))
        repeated += repeat
        counts = np.bincount(g.flat, minlength=10)
        peak = counts.max() / g.size
        sparse += .90 <= peak < 1
        dense += len(np.unique(g)) >= 7 and peak <= .40
        clean_periodic = (n >= 4 and len(np.unique(g)) > 1 and
                          any(np.array_equal(
                              g, g[np.arange(n)[:, None] % h,
                                   np.arange(n)[None, :] % w])
                              for h in range(1, min(4, n) + 1)
                              for w in range(1, min(4, n) + 1)
                              if h < n or w < n))
        periodic += clean_periodic
        combined += split == "test" and n % 2 == 1 and len(np.unique(g)) <= 3
    _require(selected_richer >= 10 and decoy_richer >= 10,
             "corner appearance became a selection cue")
    _require(selected_common >= 10 and decoy_common >= 10,
             "corner color frequency became a selection cue")
    _require(repeated >= 5 and periodic >= 3 and sparse >= 5 and dense >= 5,
             "texture and repetition diversity collapsed")
    _require(combined >= 2, "odd sparse tests absent")
    for a, b in ((2, 1), (10, 1), (10, 4), (7, 3)):
        p = generate_puzzle(3000 + a + b, a, b)
        _require((len(p["train"]), len(p["test"])) == (a, b), "explicit counts")
        _require(_puzzle_key(p) == _puzzle_key(generate_puzzle(3000 + a + b, a, b)),
                 "explicit determinism")
    for args in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(4100 + (args[0] or 0) + (args[1] or 0), *args))
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(100, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    _require(_puzzle_key(generate_puzzle(9876)) == _puzzle_key(generate_puzzle(9876)),
             "automatic determinism")
    _require(_puzzle_key(generate_puzzle(9876)) != _puzzle_key(generate_puzzle(9877)),
             "cross-seed diversity")
