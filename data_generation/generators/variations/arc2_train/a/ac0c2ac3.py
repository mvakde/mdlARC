"""Generate sparse-radius ARC puzzles with concentric square outputs."""

import numpy as np


_ODD_SIZES = tuple(range(5, 30, 2))
_AUTO_COUNT_OPTIONS = (
    (2, 1), (3, 1), (4, 2), (5, 1), (5, 2), (6, 3),
    (7, 1), (7, 2), (8, 3), (9, 4), (10, 1), (10, 4),
)
_LAYOUTS = ("scatter", "ray", "diagonal", "corners", "cluster", "sides")
_SOURCE_INPUTS = {
    ("777777777", "777477777", "777777377", "777777777", "777787777",
     "777717777", "777777777", "777777777", "777777707"),
    ("5777777", "7777777", "7777777", "7872977", "7777777", "7777777", "7777777"),
    ("77777777777", "77777707777", "77777717777", "77777777777",
     "77777777777", "77777577777", "77777977777", "77777677777",
     "77777777777", "77777777777", "78777777777"),
    ("7777777777770", "7777777777777", "7777777777777", "7777777717777",
     "7777777787777", "7777757777777", "7777772777777", "7777777777777",
     "7777777777777", "7777777777777", "7777737777777", "7477777777777",
     "7777777777777"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _normalize_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _choose_counts(seed: int, rng: np.random.Generator,
                   num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _normalize_count(num_train, "num_train", 2, 10)
    test = _normalize_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        return _AUTO_COUNT_OPTIONS[seed % len(_AUTO_COUNT_OPTIONS)]
    if train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _radius_grid(size: int) -> np.ndarray:
    center = size // 2
    offsets = np.abs(np.arange(size) - center)
    return np.maximum(offsets[:, None], offsets[None, :])


def _size_plan(rng: np.random.Generator, train_count: int, test_count: int,
               repeat_family: bool) -> tuple[list[int], list[int]]:
    cap = 29 if repeat_family else 17
    anchor_options = [size for size in _ODD_SIZES if 9 <= size <= min(23, cap - 2)]
    anchor = int(rng.choice(anchor_options))
    smaller = [size for size in _ODD_SIZES if size < anchor]
    train = [anchor, int(rng.choice(smaller))]
    train.extend(int(rng.choice([size for size in _ODD_SIZES if size <= anchor]))
                 for _ in range(train_count - 2))
    rng.shuffle(train)

    # These regimes are puzzle-level plans, not hidden rules used to decode a grid.
    regime = int(rng.integers(0, 4))  # larger, same, smaller, or a mixture
    test = []
    for index in range(test_count):
        mode = regime if regime < 3 else index % 3
        if mode == 0:
            candidates = [size for size in _ODD_SIZES if anchor < size <= cap]
        elif mode == 1:
            candidates = sorted(set(train))
        else:
            candidates = smaller
        test.append(int(rng.choice(candidates)))
    rng.shuffle(test)
    return train, test


def _sample_shell_colors(rng: np.random.Generator, size: int, background: int,
                         repeat: bool, required_color: int | None) -> np.ndarray:
    shell_count = size // 2 + 1
    available = np.array([color for color in range(10) if color != background], dtype=np.int8)
    if not repeat:
        colors = rng.permutation(available)[:shell_count].copy()
    else:
        # Begin with distinct neighboring radii. Most reuse then joins separated
        # shells; an explicit minority of examples joins neighboring shells.
        colors = np.empty(shell_count, dtype=np.int8)
        colors[0] = (required_color if required_color is not None
                     else int(rng.choice(available)))
        for radius in range(1, shell_count):
            colors[radius] = rng.choice(available[available != colors[radius - 1]])
        if rng.random() < 0.2:
            first = int(rng.integers(0, shell_count - 1))
            colors[first + 1] = colors[first]
        else:
            candidates = [
                (first, second)
                for first in range(shell_count)
                for second in range(first + 2, shell_count)
                if (colors[first] != colors[second - 1]
                    and (second + 1 == shell_count or colors[first] != colors[second + 1]))
            ]
            first, second = candidates[int(rng.integers(0, len(candidates)))]
            colors[second] = colors[first]
    if not repeat and required_color is not None and required_color not in colors:
        # A cross-pair background is made visible as a clue.
        colors[int(rng.integers(0, shell_count))] = required_color
    return colors


def _positions(rng: np.random.Generator, size: int, layout: str) -> list[tuple[int, int]]:
    center = size // 2
    positions = [(center, center)]
    ray_side = int(rng.integers(0, 4))
    row_sign = int(rng.choice(np.array((-1, 1))))
    col_sign = int(rng.choice(np.array((-1, 1))))
    corner_radii = set()
    if layout == "corners":
        radii = np.arange(1, center + 1)
        number = max(2, int(np.ceil(0.65 * center)))
        corner_radii = set(int(x) for x in rng.choice(radii, size=number, replace=False))
    for radius in range(1, center + 1):
        if layout == "ray":
            row, col = ((center - radius, center), (center, center + radius),
                        (center + radius, center), (center, center - radius))[ray_side]
        elif layout == "diagonal" or radius in corner_radii:
            if layout == "diagonal":
                row, col = center + row_sign * radius, center + col_sign * radius
            else:
                row = center + int(rng.choice(np.array((-1, 1)))) * radius
                col = center + int(rng.choice(np.array((-1, 1)))) * radius
        elif layout == "cluster" and radius <= min(3, center):
            offset = int(rng.integers(0, radius + 1))
            if rng.random() < 0.5:
                row, col = center + row_sign * radius, center + col_sign * offset
            else:
                row, col = center + row_sign * offset, center + col_sign * radius
        elif layout == "sides":
            side = (ray_side + radius) % 4
            offset = int(rng.integers(-radius + 1, radius))
            row, col = ((center - radius, center + offset),
                        (center + offset, center + radius),
                        (center + radius, center + offset),
                        (center + offset, center - radius))[side]
        else:
            shell = np.argwhere(_radius_grid(size) == radius)
            row, col = (int(x) for x in shell[int(rng.integers(0, len(shell)))])
        positions.append((row, col))
    return positions


def _make_pair(rng: np.random.Generator, size: int, background: int,
               repeat: bool, layout: str, required_color: int | None) -> dict:
    colors = _sample_shell_colors(rng, size, background, repeat, required_color)
    inp = np.full((size, size), background, dtype=np.int8)
    for radius, (row, col) in enumerate(_positions(rng, size, layout)):
        inp[row, col] = colors[radius]
    out = colors[_radius_grid(size)].astype(np.int8)
    return {"input": inp, "output": out}


def _rows(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def _signature(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _analyze_input(grid: np.ndarray) -> tuple[int, tuple[int, ...], tuple[tuple[int, int], ...]]:
    size = grid.shape[0]
    shell_count = size // 2 + 1
    counts = np.bincount(grid.reshape(-1).astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    _require(int(counts[background]) == size * size - shell_count,
             "background must occupy every cell except one marker per radius")
    colors = []
    positions = []
    for radius in range(shell_count):
        cells = np.argwhere((_radius_grid(size) == radius) & (grid != background))
        _require(len(cells) == 1, f"radius {radius} must contain one visible marker")
        row, col = (int(x) for x in cells[0])
        colors.append(int(grid[row, col]))
        positions.append((row, col))
    return background, tuple(colors), tuple(positions)


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    seen = set()
    train_backgrounds = set()
    test_backgrounds = set()
    train_repeats = False
    test_repeats = False
    train_sizes = set()
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain input and output")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grids must be 2D NumPy arrays")
                _require(np.issubdtype(grid.dtype, np.integer), "grid values must be integers")
                _require(all(1 <= extent <= 30 for extent in grid.shape), "invalid grid dimensions")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), "invalid ARC color")
            size = inp.shape[0]
            _require(inp.shape == out.shape and inp.shape == (size, size)
                     and size in _ODD_SIZES, "pairs must use odd square grids of size 5..29")
            _require(_rows(inp) not in _SOURCE_INPUTS, "source input was copied")
            signature = _signature(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            background, colors, _ = _analyze_input(inp)
            expected = np.empty_like(out)
            center = size // 2
            for row in range(size):
                for col in range(size):
                    expected[row, col] = colors[max(abs(row - center), abs(col - center))]
            _require(np.array_equal(out, expected), "incorrect square-shell expansion")
            _require(background not in colors, "invisible marker or surviving background")
            _require(not np.array_equal(inp, out), "unchanged pair")
            repeat = len(set(colors)) < len(colors)
            if split == "train":
                train_sizes.add(size)
                train_backgrounds.add(background)
                train_repeats |= repeat
            else:
                test_backgrounds.add(background)
                test_repeats |= repeat
    _require(len(train_sizes) >= 2 and max(train_sizes) >= 9,
             "training must show multiple scales and a nontrivial example")
    _require(test_backgrounds <= train_backgrounds,
             "training must demonstrate each test background")
    _require(not test_repeats or train_repeats,
             "a repeated-color test needs a repeated-color training example")
    if len(train_backgrounds) > 1:
        train_colors = [set(_analyze_input(pair["input"])[1]) for pair in train]
        _require(all(any(background in palette for palette in train_colors)
                     for background in train_backgrounds),
                 "mixed backgrounds must also occur as marker colors")
    for test_pair in test:
        _, test_colors, test_positions = _analyze_input(test_pair["input"])
        for train_pair in train:
            if train_pair["input"].shape == test_pair["input"].shape:
                _, train_colors, train_positions = _analyze_input(train_pair["input"])
                _require(test_colors != train_colors and test_positions != train_positions,
                         "same-size tests need new color orders and clue placements")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(seed, rng, num_train, num_test)

    # Puzzle-level generation plans are all revealed by the training examples.
    mixed_background = bool(rng.integers(0, 2))
    repeat_family = bool(rng.integers(0, 2))
    first_background = int(rng.integers(0, 10))
    other_background = int(rng.choice([c for c in range(10) if c != first_background]))
    train_sizes, test_sizes = _size_plan(rng, train_count, test_count, repeat_family)
    train_backgrounds = [first_background] * train_count
    test_backgrounds = [first_background] * test_count
    if mixed_background:
        train_backgrounds[1] = other_background
        for index in range(2, train_count):
            train_backgrounds[index] = int(rng.choice((first_background, other_background)))
        test_backgrounds = [int(rng.choice((first_background, other_background)))
                            for _ in range(test_count)]

    train_layouts = ["scatter", str(rng.choice(_LAYOUTS[1:]))]
    train_layouts.extend(str(rng.choice(_LAYOUTS)) for _ in range(train_count - 2))
    test_layouts = [str(rng.choice(_LAYOUTS)) for _ in range(test_count)]
    train_repeats = [repeat_family] + [bool(repeat_family and rng.random() < 0.6)
                                       for _ in range(train_count - 1)]
    test_repeats = [bool(repeat_family and rng.random() < 0.8)
                    for _ in range(test_count)]
    for index, size in enumerate(train_sizes):
        if size > 17:
            train_repeats[index] = True
    for index, size in enumerate(test_sizes):
        if size > 17:
            test_repeats[index] = True

    puzzle = {"train": [], "test": []}
    seen = set()
    for split, sizes, backgrounds, layouts, repeats in (
        ("train", train_sizes, train_backgrounds, train_layouts, train_repeats),
        ("test", test_sizes, test_backgrounds, test_layouts, test_repeats),
    ):
        for index, (size, background, layout, repeat) in enumerate(
            zip(sizes, backgrounds, layouts, repeats)
        ):
            required = None
            if mixed_background and split == "train" and index < 2:
                required = other_background if index == 0 else first_background
            for _ in range(100):
                pair = _make_pair(rng, size, background, repeat, layout, required)
                signature = _signature(pair["input"])
                new_same_size = True
                if split == "test":
                    _, colors, positions = _analyze_input(pair["input"])
                    for example in puzzle["train"]:
                        if example["input"].shape == pair["input"].shape:
                            _, old_colors, old_positions = _analyze_input(example["input"])
                            if colors == old_colors or positions == old_positions:
                                new_same_size = False
                                break
                if (new_same_size and signature not in seen
                        and _rows(pair["input"]) not in _SOURCE_INPUTS):
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    def puzzle_signature(puzzle: dict) -> tuple:
        return tuple(tuple((_signature(pair["input"]), _signature(pair["output"]))
                           for pair in puzzle[split]) for split in ("train", "test"))
    _require(puzzle_signature(generate_puzzle(37)) == puzzle_signature(generate_puzzle(37)),
             "automatic counts are not deterministic")
    _require(puzzle_signature(generate_puzzle(9001, 7, 3)) ==
             puzzle_signature(generate_puzzle(9001, 7, 3)),
             "explicit counts are not deterministic")
    _require(puzzle_signature(generate_puzzle(12001, 6, 2)) !=
             puzzle_signature(generate_puzzle(12002, 6, 2)),
             "different seeds returned the same puzzle")
    for index, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(20000 + index, train_count, test_count)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count),
                 "explicit count ignored")
    for index, (train_count, test_count) in enumerate(
        ((None, 1), (None, 4), (2, None), (10, None))
    ):
        puzzle = generate_puzzle(21000 + index, train_count, test_count)
        if train_count is not None:
            _require(len(puzzle["train"]) == train_count, "partial train count ignored")
        if test_count is not None:
            _require(len(puzzle["test"]) == test_count, "partial test count ignored")
    for index, (train_count, test_count) in enumerate(
        ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4))
    ):
        try:
            generate_puzzle(22000 + index, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count combination accepted")

    train_counts = set()
    test_counts = set()
    sizes = set()
    scales = set()
    mixed_count = 0
    fixed_count = 0
    repeat_count = 0
    distinct_count = 0
    adjacent_count = 0
    separated_count = 0
    corner_count = 0
    outer_corner_count = 0
    ray_count = 0
    diagonal_count = 0
    cluster_count = 0
    multi_side_count = 0
    scattered_count = 0
    combined_count = 0
    below_max_count = 0
    background_colors = set()
    layouts = set()
    signatures = set()
    for puzzle in puzzles:
        signatures.add(puzzle_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_sizes = [pair["input"].shape[0] for pair in puzzle["train"]]
        backgrounds = [_analyze_input(pair["input"])[0] for pair in puzzle["train"]]
        mixed = len(set(backgrounds)) > 1
        background_colors.update(backgrounds)
        mixed_count += mixed
        fixed_count += not mixed
        has_repeat = any(len(set(_analyze_input(pair["input"])[1])) <
                         len(_analyze_input(pair["input"])[1])
                         for pair in puzzle["train"])
        combined_count += mixed and has_repeat
        for pair in puzzle["test"]:
            size = pair["input"].shape[0]
            below_max_count += size < max(train_sizes)
            scales.add("larger" if size > max(train_sizes) else
                       "same" if size in train_sizes else "smaller")
        for pair in puzzle["train"] + puzzle["test"]:
            inp, out = pair["input"], pair["output"]
            size = inp.shape[0]
            sizes.add(size)
            background, colors, positions = _analyze_input(inp)
            # A separate loop decodes the visible test input without generation metadata.
            predicted = np.array([[colors[max(abs(row - size // 2), abs(col - size // 2))]
                                   for col in range(size)] for row in range(size)], dtype=np.int8)
            _require(np.array_equal(predicted, out), "independent radial decoding failed")
            _require(background not in colors and _rows(inp) not in _SOURCE_INPUTS,
                     "source copy or invisible marker")
            repeated = len(set(colors)) < len(colors)
            repeat_count += repeated
            distinct_count += not repeated
            adjacent_count += any(colors[r] == colors[r + 1] for r in range(len(colors) - 1))
            separated_count += any(colors[r] == colors[s] for r in range(len(colors))
                                   for s in range(r + 2, len(colors)))
            center = size // 2
            corners = sum(abs(row - center) == radius and abs(col - center) == radius
                          for radius, (row, col) in enumerate(positions) if radius)
            corner_count += corners >= 2
            outer_corner_count += (abs(positions[-1][0] - center) == center
                                   and abs(positions[-1][1] - center) == center)
            offsets = [(row - center, col - center) for row, col in positions[1:]]
            ray_count += any(all((dr, dc) == (sign * radius, 0)
                                 for radius, (dr, dc) in enumerate(offsets, 1))
                             for sign in (-1, 1)) or any(
                all((dr, dc) == (0, sign * radius)
                    for radius, (dr, dc) in enumerate(offsets, 1)) for sign in (-1, 1))
            diagonal_count += any(all((dr, dc) == (row_sign * radius, col_sign * radius)
                                      for radius, (dr, dc) in enumerate(offsets, 1))
                                  for row_sign in (-1, 1) for col_sign in (-1, 1))
            inner = offsets[:min(3, center)]
            cluster_count += any(all(dr * row_sign >= 0 and dc * col_sign >= 0
                                     for dr, dc in inner)
                                 for row_sign in (-1, 1) for col_sign in (-1, 1))
            sides = set()
            for radius, (dr, dc) in enumerate(offsets, 1):
                if abs(dr) != abs(dc):
                    sides.add("N" if dr == -radius else "S" if dr == radius
                              else "E" if dc == radius else "W")
            multi_side_count += len(sides) >= 3
            scattered_count += len({(np.sign(dr), np.sign(dc)) for dr, dc in offsets}) >= 3
            layouts.add((size, tuple(positions)))
    _require(len(signatures) == 42 and len(layouts) >= 100, "insufficient structural diversity")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "counts lack diversity")
    _require(set(_ODD_SIZES) <= sizes,
             "size coverage is too narrow")
    _require(scales == {"larger", "same", "smaller"} and below_max_count >= 5,
             "scale regimes missing")
    _require(mixed_count >= 5 and fixed_count >= 5 and combined_count >= 2,
             "background regimes or combined variation missing")
    _require(len(background_colors) == 10, "background color assignments are too narrow")
    _require(repeat_count >= 20 and distinct_count >= 20 and adjacent_count >= 5
             and separated_count > adjacent_count * 2,
             "palette reuse lacks variety or overuses adjacent rings")
    _require(corner_count >= 10 and outer_corner_count >= 10 and ray_count >= 5
             and diagonal_count >= 5 and cluster_count >= 15
             and multi_side_count >= 10 and scattered_count >= 20,
             "clue layouts lack variety")
