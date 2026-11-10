"""Generate strip-copy ARC puzzles guided by a single monochromatic line."""

from itertools import combinations

import numpy as np


_SOURCE = {
    ((3, 3), (1, 1, 1, 6, 2, 2, 2, 2, 6)),
    ((3, 3), (2, 4, 3, 2, 3, 4, 2, 3, 4)),
    ((3, 3), (3, 1, 6, 3, 6, 1, 3, 1, 6)),
    ((3, 3), (4, 4, 6, 3, 3, 3, 6, 6, 4)),
    ((3, 3), (6, 6, 3, 4, 4, 3, 4, 4, 3)),
}
_SHAPES = tuple((h, w) for h in range(2, 6) for w in range(2, 6))
_SQUARES = ((2, 2), (3, 3), (4, 4), (5, 5))
_RECTANGLES = tuple(s for s in _SHAPES if s[0] != s[1])
_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 1), (8, 2), (9, 3), (10, 4))


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _key(grid: np.ndarray) -> tuple:
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _guides(grid: np.ndarray) -> list[tuple[str, int]]:
    found = []
    for row in range(grid.shape[0]):
        if bool(np.all(grid[row] == grid[row, 0])):
            found.append(("row", row))
    for col in range(grid.shape[1]):
        if bool(np.all(grid[:, col] == grid[0, col])):
            found.append(("col", col))
    return found


def _make_output(grid: np.ndarray, background: int) -> np.ndarray:
    guides = _guides(grid)
    if len(guides) != 1:
        raise ValueError("the seed needs one spanning guide")
    h, w = grid.shape
    kind, index = guides[0]
    result = np.full((h * h, w * w), background, dtype=np.int8)
    if kind == "row":
        result[index * h:(index + 1) * h, :] = np.tile(grid, (1, w))
    else:
        result[:, index * w:(index + 1) * w] = np.tile(grid, (h, 1))
    return result


def _count(name: str, value: int | None, minimum: int, maximum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is out of range")
    return value


def _counts(rng: np.random.Generator, train: int | None,
            test: int | None) -> tuple[int, int]:
    train = _count("num_train", train, 2, 10)
    test = _count("num_test", test, 1, 4)
    if train is None and test is None:
        return _AUTO_COUNTS[int(rng.integers(len(_AUTO_COUNTS)))]
    if train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _pick(rng: np.random.Generator, choices: tuple):
    return choices[int(rng.integers(len(choices)))]


def _shapes(rng: np.random.Generator, ntrain: int,
            ntest: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    mode = int(rng.choice(3, p=[0.42, 0.30, 0.28]))
    if mode == 0:
        shape = (3, 3) if rng.random() < 0.42 else _pick(rng, _SHAPES)
        return [shape] * ntrain, [shape] * ntest
    if mode == 1:
        first = _pick(rng, _SQUARES)
        second = _pick(rng, tuple(s for s in _SQUARES if s != first))
        pool = _SQUARES
    else:
        first, second = _pick(rng, _RECTANGLES), _pick(rng, _SQUARES)
        pool = _SHAPES
    train = [first, second] + [_pick(rng, pool) for _ in range(ntrain - 2)]
    rng.shuffle(train)
    test = []
    for _ in range(ntest):
        unseen = tuple(s for s in pool if s not in train)
        choices = unseen if unseen and rng.random() < 0.42 else tuple(train)
        test.append(_pick(rng, choices))
    return train, test


def _length(shape: tuple[int, int], kind: str) -> int:
    return shape[0] if kind == "row" else shape[1]


def _guide_plan(rng: np.random.Generator, train_shapes: list[tuple[int, int]],
                test_shapes: list[tuple[int, int]]) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    mode = int(rng.integers(3))
    source_like = mode == 2 and rng.random() < 0.35
    ntrain = len(train_shapes)
    if mode == 0:
        kinds = ["row"] * ntrain
    elif mode == 1:
        kinds = ["col"] * ntrain
    else:
        kinds = ["row", "col"] + [_pick(rng, ("row", "col")) for _ in range(ntrain - 2)]
        rng.shuffle(kinds)
    if source_like:
        kinds[:2] = ["row", "col"]
    train = []
    for i, (kind, shape) in enumerate(zip(kinds, train_shapes)):
        length = _length(shape, kind)
        if source_like and i == 0:
            index = 1
        elif source_like and i == 1:
            index = 0
        elif i == 1:
            index = _pick(rng, tuple(j for j in range(length) if j != train[0][1]))
        else:
            index = int(rng.integers(length))
        train.append((kind, index))
    if source_like and ntrain >= 3:
        train[2] = ("row", 0)
    available = tuple(sorted(set(kinds)))
    test = []
    for shape in test_shapes:
        kind = "col" if source_like else _pick(rng, available)
        length = _length(shape, kind)
        prior = tuple(i for k, i in train if k == kind and i < length)
        if source_like:
            index = length - 1
        elif prior and rng.random() < 0.43:
            index = _pick(rng, prior)
        else:
            held_out = tuple(i for i in range(length) if i not in prior)
            index = _pick(rng, held_out if held_out else tuple(range(length)))
        test.append((kind, index))
    return train, test


def _palette(rng: np.random.Generator, size: int, background: int) -> tuple[int, ...]:
    forced = set()
    if rng.random() < 0.34:
        forced.add(background)
    if background != 0 and rng.random() < 0.35:
        forced.add(0)
    available = tuple(c for c in range(10) if c not in forced)
    extra = rng.choice(available, size=size - len(forced), replace=False)
    return tuple(sorted(forced | {int(c) for c in extra}))


def _seed(rng: np.random.Generator, shape: tuple[int, int],
          guide: tuple[str, int], palette: tuple[int, ...],
          forbidden: set[tuple]) -> np.ndarray:
    h, w = shape
    kind, index = guide
    free = [(r, c) for r in range(h) for c in range(w)
            if (r != index if kind == "row" else c != index)]
    if shape == (2, 2):
        candidates = []
        for line_color in palette:
            alternatives = [c for c in palette if c != line_color]
            for first, second in ((alternatives[0], alternatives[1]),
                                  (alternatives[1], alternatives[0])):
                grid = np.empty(shape, dtype=np.int8)
                if kind == "row":
                    grid[index, :] = line_color
                else:
                    grid[:, index] = line_color
                grid[free[0]], grid[free[1]] = first, second
                key = _key(grid)
                if key not in forbidden and key not in _SOURCE:
                    candidates.append(grid)
        if candidates:
            return candidates[int(rng.integers(len(candidates)))]
        raise RuntimeError("the 2-by-2 palette is exhausted for this guide")
    guide_color = _pick(rng, palette)
    others = [c for c in palette if c != guide_color]
    guide_length = w if kind == "row" else h
    non_guide_lines = h - 1 if kind == "row" else w - 1
    maximum_major = len(free) - max(len(others) - 1, non_guide_lines)
    frequency_mode = "normal"
    if shape == (3, 3) and 3 <= len(palette) <= 4:
        frequency_mode = str(rng.choice(("low", "tie", "normal"), p=(0.50, 0.30, 0.20)))
    else:
        if (len(palette) >= 3 and
                maximum_major >= guide_length + 1 and
                rng.random() < 0.45):
            frequency_mode = "low"
    for attempt in range(1000):
        if attempt == 500:
            frequency_mode = "normal"
        if frequency_mode == "low":
            major = _pick(rng, tuple(others))
            minor = [c for c in others if c != major]
            major_count = min(maximum_major,
                              max(guide_length + 1, (len(free) + 1) // 2))
            values = [major] * major_count + minor[:]
            values.extend(_pick(rng, tuple(minor)) for _ in range(len(free) - len(values)))
        elif frequency_mode == "tie":
            major = _pick(rng, tuple(others))
            minor = [c for c in others if c != major]
            values = [major] * 3 + minor[:]
            values.extend(_pick(rng, tuple(minor)) for _ in range(6 - len(values)))
        else:
            values = others[:]
            values.extend(_pick(rng, palette) for _ in range(len(free) - len(values)))
        rng.shuffle(values)
        grid = np.empty(shape, dtype=np.int8)
        if kind == "row":
            grid[index, :] = guide_color
        else:
            grid[:, index] = guide_color
        for (r, c), value in zip(free, values):
            grid[r, c] = value
        key = _key(grid)
        if _guides(grid) == [guide] and key not in forbidden and key not in _SOURCE:
            return grid
    raise RuntimeError("could not construct a distinct, unambiguous seed")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(rng, num_train, num_test)
    background = 0 if rng.random() < 0.48 else int(rng.integers(1, 10))
    train_shapes, test_shapes = _shapes(rng, ntrain, ntest)
    train_guides, test_guides = _guide_plan(rng, train_shapes, test_shapes)
    specs = list(zip(train_shapes + test_shapes, train_guides + test_guides))
    shared_palette = rng.random() < 0.19
    if shared_palette:
        capacity = min(1 + h * w - (w if guide[0] == "row" else h)
                       for (h, w), guide in specs)
        common = _palette(rng, int(rng.integers(3, min(5, capacity) + 1)), background)
    puzzle = {"train": [], "test": []}
    forbidden: set[tuple] = set()
    first_palette = None
    for i, (shape, guide) in enumerate(specs):
        if shared_palette:
            palette = common
        else:
            h, w = shape
            capacity = min(10, 1 + h * w - (w if guide[0] == "row" else h))
            minimum = 3 if min(shape) == 2 else 2
            if shape == (3, 3):
                size = int(rng.choice(np.arange(2, 8),
                                      p=(0.12, 0.28, 0.22, 0.14, 0.12, 0.12)))
            else:
                size = int(rng.integers(minimum, capacity + 1))
            palette = _palette(rng, size, background)
            if i == 1 and palette == first_palette:
                for _ in range(12):
                    palette = _palette(rng, int(rng.integers(minimum, capacity + 1)), background)
                    if palette != first_palette:
                        break
        if i == 0:
            first_palette = palette
        try:
            grid = _seed(rng, shape, guide, palette, forbidden)
        except RuntimeError:
            # A 2-by-2 tile has only six distinct arrangements per palette
            # and guide. Visit the finite palette space if one is exhausted.
            for alternative in combinations(range(10), len(palette)):
                if alternative == palette:
                    continue
                try:
                    grid = _seed(rng, shape, guide, alternative, forbidden)
                    break
                except RuntimeError:
                    continue
            else:
                raise RuntimeError("no distinct seed remains for this guide")
        forbidden.add(_key(grid))
        split = "train" if i < ntrain else "test"
        puzzle[split].append({"input": grid, "output": _make_output(grid, background)})
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair: dict, label: str) -> tuple[int, tuple[str, int], tuple[int, int]]:
    _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
           f"{label}: invalid pair schema")
    for name in ("input", "output"):
        grid = pair[name]
        _check(isinstance(grid, np.ndarray) and grid.ndim == 2,
               f"{label}.{name}: expected a 2D NumPy array")
        _check(all(1 <= x <= 30 for x in grid.shape), f"{label}.{name}: bad dimensions")
        _check(np.issubdtype(grid.dtype, np.integer), f"{label}.{name}: noninteger data")
        _check(bool(np.all((0 <= grid) & (grid <= 9))), f"{label}.{name}: invalid color")
    grid, out = pair["input"], pair["output"]
    h, w = grid.shape
    _check(2 <= h <= 5 and 2 <= w <= 5, f"{label}: invalid tile shape")
    _check(out.shape == (h * h, w * w), f"{label}: incorrect output size")
    guides = _guides(grid)
    _check(len(guides) == 1, f"{label}: ambiguous guide")
    kind, index = guides[0]
    blanks = []
    for r in range(h):
        for c in range(w):
            block = out[r * h:(r + 1) * h, c * w:(c + 1) * w]
            if (kind == "row" and r == index) or (kind == "col" and c == index):
                _check(np.array_equal(block, grid), f"{label}: tile was altered")
            else:
                _check(bool(np.all(block == block[0, 0])),
                       f"{label}: blank tile is not uniform")
                blanks.append(int(block[0, 0]))
    _check(len(set(blanks)) == 1, f"{label}: inconsistent blank color")
    return blanks[0], guides[0], (h, w)


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "puzzle must have exactly train and test")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
           "train and test must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    backgrounds, train_guides, test_guides, train_shapes, test_shapes = [], [], [], [], []
    for split in ("train", "test"):
        for i, pair in enumerate(puzzle[split]):
            bg, guide, shape = _inspect_pair(pair, f"{split}[{i}]")
            backgrounds.append(bg)
            (train_guides if split == "train" else test_guides).append(guide)
            (train_shapes if split == "train" else test_shapes).append(shape)
            input_key = _key(pair["input"])
            pair_key = (input_key, _key(pair["output"]))
            _check(input_key not in seen_inputs, "duplicate input")
            _check(pair_key not in seen_pairs, "duplicate pair")
            _check(input_key not in _SOURCE, "copied source pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
    _check(len(set(backgrounds)) == 1, "background changes within a puzzle")
    _check(len({i for _, i in train_guides}) >= 2,
           "training does not demonstrate positional correspondence")
    _check({k for k, _ in test_guides} <= {k for k, _ in train_guides},
           "test orientation is absent from training")
    if any(shape not in train_shapes for shape in test_shapes):
        _check(len(set(train_shapes)) >= 2,
               "a new test size needs multiple demonstrated sizes")
    if any(h != w for h, w in test_shapes):
        _check(any(h != w for h, w in train_shapes),
               "a rectangular test needs a rectangular demonstration")


def _signature(puzzle: dict) -> tuple:
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[split])
                 for split in ("train", "test"))


def self_test() -> None:
    for seed, train, test in ((1701, None, None), (1702, 8, 3),
                              (2001, 2, 1), (2002, 10, 1), (2003, 10, 4),
                              (2101, None, 1), (2102, None, 4),
                              (2103, 2, None), (2104, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        _check(_signature(puzzle) == _signature(generate_puzzle(seed, train, test)),
               "generation is not deterministic")
        if train is not None:
            _check(len(puzzle["train"]) == train, "explicit train count ignored")
        if test is not None:
            _check(len(puzzle["test"]) == test, "explicit test count ignored")
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(2200, **request)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {request}")
    counts, tests, signatures, train_guides, test_guides = set(), set(), set(), set(), set()
    shapes, backgrounds, palettes, normalized_patterns = set(), set(), set(), set()
    two = seven = low = ties = internal_bg = zero_copy = off_guide = rectangles = 0
    varied_palettes = varied_shapes = mixed = common = horizontal = vertical = 0
    overlap = held_out = new_shape = new_color = left_to_right = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts.add(len(puzzle["train"]))
        tests.add(len(puzzle["test"]))
        signatures.add(_signature(puzzle))
        local_palettes, local_shapes, local_kinds = set(), set(), set()
        background = _inspect_pair(puzzle["train"][0], "self-test")[0]
        backgrounds.add(background)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                bg, guide, shape = _inspect_pair(pair, "self-test")
                _check(bg == background, "background drift")
                (train_guides if split == "train" else test_guides).add(guide)
                shapes.add(shape)
                local_shapes.add(shape)
                local_kinds.add(guide[0])
                colors = tuple(int(x) for x in np.unique(grid))
                palettes.add(colors)
                local_palettes.add(colors)
                renaming = {}
                pattern = []
                for raw in grid.flat:
                    color = int(raw)
                    if color not in renaming:
                        renaming[color] = len(renaming)
                    pattern.append(renaming[color])
                normalized_patterns.add((shape, tuple(pattern)))
                two += len(colors) == 2 and shape == (3, 3)
                seven += len(colors) == 7 and shape == (3, 3)
                rectangles += shape[0] != shape[1]
                internal_bg += bool(np.any(grid == background))
                zero_copy += background != 0 and bool(np.any(grid == 0))
                guide_color = int(grid[guide[1], 0] if guide[0] == "row" else grid[0, guide[1]])
                away = np.delete(grid, guide[1], axis=0 if guide[0] == "row" else 1)
                off_guide += bool(np.any(away == guide_color))
                frequencies = np.bincount(grid.reshape(-1), minlength=10)
                low += bool(frequencies[guide_color] < frequencies.max())
                ties += bool(np.count_nonzero(frequencies == frequencies.max()) > 1)
        varied_palettes += len(local_palettes) > 1
        varied_shapes += len(local_shapes) > 1
        mixed += len(local_kinds) > 1
        common += len(local_palettes) == 1
        horizontal += local_kinds == {"row"}
        vertical += local_kinds == {"col"}
        trained = {_guides(pair["input"])[0] for pair in puzzle["train"]}
        tested = [_guides(pair["input"])[0] for pair in puzzle["test"]]
        overlap += any(guide in trained for guide in tested)
        held_out += any(guide not in trained for guide in tested)
        training_shapes = {pair["input"].shape for pair in puzzle["train"]}
        new_shape += any(pair["input"].shape not in training_shapes
                         for pair in puzzle["test"])
        training_colors = {int(value) for pair in puzzle["train"]
                           for value in pair["input"].flat}
        new_color += any(int(value) not in training_colors
                         for pair in puzzle["test"] for value in pair["input"].flat)
        left_to_right += (("col", 0) in trained and any(
            guide == ("col", pair["input"].shape[1] - 1) and guide not in trained
            for guide, pair in zip(tested, puzzle["test"])))
    _check(len(counts) >= 4 and len(tests) >= 3, "automatic counts lack diversity")
    _check(len(signatures) == 42 and len(palettes) >= 30, "grids or palettes lack diversity")
    _check(len(shapes) >= 12 and len(backgrounds) >= 4, "shape or background diversity low")
    _check(len(normalized_patterns) >= 100, "normalized structural diversity low")
    _check({("row", i) for i in range(3)} <= train_guides & test_guides,
           "row positions do not appear in both splits")
    _check({("col", i) for i in range(3)} <= train_guides & test_guides,
           "column positions do not appear in both splits")
    _check(min(two, seven, low, ties, internal_bg, zero_copy) > 0,
           "a reviewed palette or frequency case is missing")
    _check(min(off_guide, rectangles, common, horizontal, vertical, overlap,
               held_out, new_shape, new_color, left_to_right) > 0,
           "a reviewed regime or combined feature is missing")
    _check(min(varied_palettes, varied_shapes, mixed) >= 5,
           "within-puzzle diversity is too narrow")
