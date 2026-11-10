"""Horizontal bar lengths projected as literal square digits."""

import numpy as np


_SOURCE_INPUTS = frozenset({
    ("005", "000", "110", "000"),
    ("0000000500", "0000000000", "0222000000", "0000000000",
     "0330000000", "0000000000", "0111100000", "0000000000",
     "0999990000", "0000000000"),
    ("00000000050", "00440000000", "00000000000", "22000000000",
     "00000000000", "07777700000", "00000000000", "08800000000"),
    ("000005", "220000", "000000", "222000", "000000", "222200"),
    ("000050000000", "000000000000", "220000000000", "000000000000",
     "222000000000", "000000000000", "055000000000", "000000000000",
     "999900000000", "000000000000"),
})
_LENGTHS = (2, 3, 4)
_BACKGROUNDS = (0, 2, 3, 5, 6, 7, 8)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(rng, num_train, num_test):
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _signature(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _key(grid):
    return (tuple(grid.shape), grid.astype(np.uint8, copy=False).tobytes())


def _make_pair(rng, background, marker_color, specs):
    """Each spec is (length, color, side); side is -1 or +1."""
    n = len(specs)
    marker_slot = int(rng.integers(0, n + 1))
    top_padding = int(rng.integers(0, 5))
    bottom_padding = int(rng.integers(0, 5))
    active_rows = [top_padding]
    for index in range(n):
        adjacent_to_marker = index == marker_slot or index + 1 == marker_slot
        distance = int(rng.integers(1 if adjacent_to_marker else 2, 4))
        active_rows.append(active_rows[-1] + distance)
    height = active_rows[-1] + 1 + bottom_padding
    marker_row = active_rows[marker_slot]
    bar_rows = active_rows[:marker_slot] + active_rows[marker_slot + 1:]

    left_lengths = [length for length, _, side in specs if side == -1]
    right_lengths = [length for length, _, side in specs if side == 1]
    left_capacity = (max(left_lengths) + 1 + int(rng.integers(0, 9))
                     if left_lengths else int(rng.integers(0, 12)))
    right_capacity = (max(right_lengths) + 1 + int(rng.integers(0, 9))
                      if right_lengths else int(rng.integers(0, 12)))
    width = left_capacity + 1 + right_capacity
    if 2 * (1 + sum(length for length, _, _ in specs)) >= height * width:
        height += 1
    _check(height <= 30 and width <= 30, "construction exceeded ARC dimensions")
    column = left_capacity
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    inp[marker_row, column] = marker_color
    out[marker_row, column] = marker_color

    for row, (length, color, side) in zip(bar_rows, specs):
        capacity = left_capacity if side == -1 else right_capacity
        minimum_gap = 1 if abs(row - marker_row) == 1 else 0
        gap = int(rng.integers(minimum_gap, capacity - length + 1))
        start = column - gap - length if side == -1 else column + gap + 1
        inp[row, start:start + length] = color
        out[row, start:start + length] = color
        out[row, column] = (length - 1) ** 2 % 10
    return {"input": inp, "output": out}


def _inspect_pair(inp, out):
    """Read roles from cells, independently of the generation specification."""
    _check(inp.shape == out.shape, "shape changed")
    values, counts = np.unique(inp, return_counts=True)
    background = int(values[np.argmax(counts)])
    _check(background in _BACKGROUNDS, "background hides a length code")
    foreground_count = np.count_nonzero(inp != background, axis=1)
    singleton_rows = np.flatnonzero(foreground_count == 1)
    _check(len(singleton_rows) == 1, "exactly one isolated marker row is needed")
    marker_row = int(singleton_rows[0])
    column = int(np.flatnonzero(inp[marker_row] != background)[0])
    marker_color = int(inp[marker_row, column])
    _check(np.count_nonzero(inp[:, column] != background) == 1,
           "marker column must be empty in bar rows")
    _check(int(out[marker_row, column]) == marker_color, "marker changed")

    bars = []
    expected = inp.copy()
    for row in range(inp.shape[0]):
        if row == marker_row:
            continue
        occupied = np.flatnonzero(inp[row] != background)
        if len(occupied) == 0:
            continue
        length = len(occupied)
        _check(length in _LENGTHS, "unsupported bar length")
        start, end = int(occupied[0]), int(occupied[-1])
        _check(np.array_equal(occupied, np.arange(start, end + 1)),
               "bar is not contiguous")
        color = int(inp[row, start])
        _check(np.all(inp[row, start:end + 1] == color), "bar is not monochrome")
        _check(end < column or start > column, "bar crosses marker column")
        side = -1 if end < column else 1
        gap = column - end - 1 if side == -1 else start - column - 1
        if abs(row - marker_row) == 1:
            _check(gap >= 1, "marker touches a bar diagonally")
        bars.append((row, start, length, color, side))
        expected[row, column] = (length - 1) ** 2 % 10
    _check(bars, "no bars")
    _check(all(b[0] - a[0] >= 2 for a, b in zip(bars, bars[1:])),
           "bars touch across rows")
    _check(np.array_equal(out, expected), "length code or preservation failed")
    _check(np.count_nonzero(inp != out) == len(bars), "one cell per bar required")
    return {"background": background, "marker_color": marker_color,
            "marker_row": marker_row, "column": column, "bars": tuple(bars),
            "left_capacity": column, "right_capacity": inp.shape[1] - column - 1}


def _prediction_from_training(training, test_input):
    """Infer a visible length lookup using only demonstrations and an input."""
    lookup = {}
    for pair in training:
        meta = _inspect_pair(pair["input"], pair["output"])
        for row, _, length, _, _ in meta["bars"]:
            digit = int(pair["output"][row, meta["column"]])
            if length in lookup:
                _check(lookup[length] == digit, "training contradicts a length code")
            lookup[length] = digit
    blank_output = test_input.copy()
    # Locate the marker from the singleton row; the background is the modal color.
    values, counts = np.unique(test_input, return_counts=True)
    background = int(values[np.argmax(counts)])
    singles = np.flatnonzero(np.count_nonzero(test_input != background, axis=1) == 1)
    _check(len(singles) == 1, "test marker is not identifiable")
    column = int(np.flatnonzero(test_input[int(singles[0])] != background)[0])
    for row in range(test_input.shape[0]):
        if row == int(singles[0]):
            continue
        length = int(np.count_nonzero(test_input[row] != background))
        if length:
            _check(length in lookup, "test length is not taught")
            blank_output[row, column] = lookup[length]
    return blank_output


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "puzzle must have exactly train and test")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
           "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal pair counts")
    seen = set()
    metas = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
                   "pair must contain exactly input and output")
            for grid in pair.values():
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2,
                       "grid must be a 2D NumPy array")
                _check(all(1 <= d <= 30 for d in grid.shape), "invalid grid size")
                _check(np.issubdtype(grid.dtype, np.integer), "noninteger grid")
                _check(bool(np.all((0 <= grid) & (grid <= 9))), "invalid ARC color")
            key = _key(pair["input"])
            _check(key not in seen, "duplicate input")
            seen.add(key)
            _check(_signature(pair["input"]) not in _SOURCE_INPUTS, "source copy")
            metas[split].append(_inspect_pair(pair["input"], pair["output"]))

    backgrounds = {m["background"] for group in metas.values() for m in group}
    _check(len(backgrounds) == 1, "background is inconsistent within a puzzle")
    learned = {length for m in metas["train"] for _, _, length, _, _ in m["bars"]}
    _check(learned == set(_LENGTHS), "training must teach all three reliable codes")
    length_colors = {length: set() for length in _LENGTHS}
    color_lengths = {}
    for meta in metas["train"]:
        for _, _, length, color, _ in meta["bars"]:
            length_colors[length].add(color)
            color_lengths.setdefault(color, set()).add(length)
    _check(any(len(colors) >= 2 for colors in length_colors.values()),
           "train must vary color at fixed length")
    _check(any(len(lengths) >= 2 for lengths in color_lengths.values()),
           "train must vary length at fixed color")
    _check(any(any(color == m["marker_color"] for _, _, _, color, _ in m["bars"])
               for m in metas["train"]), "training must reuse a marker color on a bar")
    train_right = any(side == 1 for m in metas["train"] for *_, side in m["bars"])
    test_right = any(side == 1 for m in metas["test"] for *_, side in m["bars"])
    _check(not test_right or train_right, "right-hand bars need training evidence")
    train_markers = {m["marker_color"] for m in metas["train"]}
    test_markers = {m["marker_color"] for m in metas["test"]}
    _check(test_markers.issubset(train_markers) or len(train_markers) >= 2,
           "marker color variation must be visible in training")
    for pair, meta in zip(puzzle["test"], metas["test"]):
        _check({bar[2] for bar in meta["bars"]}.issubset(learned),
               "test contains untaught length")
        _check(np.array_equal(_prediction_from_training(puzzle["train"], pair["input"]),
                              pair["output"]), "test is not recoverable from training")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    background = 0 if rng.random() < 0.55 else int(rng.choice(_BACKGROUNDS[1:]))
    available = [color for color in range(10) if color != background]
    source_like = bool(rng.random() < 0.32)
    fixed_marker = bool(rng.random() < 0.55)
    if fixed_marker:
        color = int(rng.choice(available))
        markers = [color] * (num_train + num_test)
    else:
        first_two = [int(x) for x in rng.choice(available, size=2, replace=False)]
        markers = first_two + [int(rng.choice(available))
                               for _ in range(num_train + num_test - 2)]

    train_counts = [3, 2] + [int(rng.integers(1, 6)) for _ in range(num_train - 2)]
    total = sum(train_counts)
    a, b = [int(x) for x in rng.choice(available, size=2, replace=False)]
    palette_size = int(rng.integers(2, 5))
    remaining = [c for c in available if c not in (a, b)]
    palette = [a, b] + [int(x) for x in rng.choice(remaining,
                                                  size=palette_size - 2,
                                                  replace=False)]
    permutation = [int(x) for x in rng.permutation(np.array(_LENGTHS))]
    instructions = [(permutation[0], a, False),
                    (permutation[1], a, False),
                    (permutation[2], b, False),
                    (permutation[0], b, False),
                    (int(rng.choice(_LENGTHS)), 0, True)]
    instructions += [(int(rng.choice(_LENGTHS)), int(rng.choice(palette)), False)
                     for _ in range(total - 5)]
    instructions = [instructions[int(i)] for i in rng.permutation(total)]
    train_specs = []
    offset = 0
    for index, count in enumerate(train_counts):
        entries = instructions[offset:offset + count]
        offset += count
        colors = [(length, markers[index] if use_marker else color)
                  for length, color, use_marker in entries]
        if source_like:
            sides = [-1] * count
        elif index == 0:
            sides = [-1, 1] + [int(rng.choice((-1, 1))) for _ in range(count - 2)]
            sides = [sides[int(i)] for i in rng.permutation(count)]
        else:
            sides = [int(rng.choice((-1, 1))) for _ in range(count)]
        train_specs.append([(length, color, side)
                            for (length, color), side in zip(colors, sides)])

    seen_train_colors = {color for specs in train_specs for _, color, _ in specs}
    all_pairs = {"train": [], "test": []}
    used = set()
    for split in ("train", "test"):
        count = num_train if split == "train" else num_test
        for index in range(count):
            global_index = index if split == "train" else num_train + index
            if split == "train":
                specs = train_specs[index]
            else:
                n_bars = int(rng.integers(1, 7))
                lengths = [int(rng.choice(_LENGTHS)) for _ in range(n_bars)]
                sides = ([-1] * n_bars if source_like else
                         [int(rng.choice((-1, 1))) for _ in range(n_bars)])
                marker_count = int(rng.integers(0, min(3, n_bars) + 1))
                marker_positions = set(int(x) for x in rng.choice(
                    n_bars, size=marker_count, replace=False))
                familiar = [c for c in seen_train_colors
                            if c != markers[global_index]]
                novel = [c for c in available if c not in seen_train_colors
                         and c != markers[global_index]]
                ordinary = [i for i in range(n_bars) if i not in marker_positions]
                new_position = (int(rng.choice(ordinary)) if ordinary and novel
                                and rng.random() < 0.5 else -1)
                specs = []
                for j, (length, side) in enumerate(zip(lengths, sides)):
                    if j in marker_positions:
                        color = markers[global_index]
                    elif j == new_position:
                        color = int(rng.choice(novel))
                    else:
                        color = int(rng.choice(familiar))
                    specs.append((length, color, side))
            for _ in range(32):
                pair = _make_pair(rng, background, markers[global_index], specs)
                key = _key(pair["input"])
                if key not in used and _signature(pair["input"]) not in _SOURCE_INPUTS:
                    used.add(key)
                    all_pairs[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(all_pairs)
    return all_pairs


def _puzzle_key(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _check(_puzzle_key(puzzles[17]) == _puzzle_key(generate_puzzle(17)),
           "automatic call is nondeterministic")
    _check(_puzzle_key(generate_puzzle(314159, 7, 3)) ==
           _puzzle_key(generate_puzzle(314159, 7, 3)),
           "explicit call is nondeterministic")
    _check(len({_puzzle_key(p) for p in puzzles}) == 42,
           "different seeds produced duplicate puzzles")
    for nt, ns in ((2, 1), (10, 1), (10, 4), (7, 3)):
        puzzle = generate_puzzle(1000 + 10 * nt + ns, nt, ns)
        _check((len(puzzle["train"]), len(puzzle["test"])) == (nt, ns),
               "explicit counts were ignored")
    for index, (nt, ns) in enumerate(((None, 1), (None, 4),
                                       (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + index, nt, ns)
        _check((nt is None or len(puzzle["train"]) == nt) and
               (ns is None or len(puzzle["test"]) == ns),
               "partial counts were ignored")
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts were accepted")
    _check(len({len(p["train"]) for p in puzzles}) >= 4, "few train counts")
    _check(len({len(p["test"]) for p in puzzles}) >= 3, "few test counts")
    metas = [(split, _inspect_pair(pair["input"], pair["output"]), pair)
             for p in puzzles for split in ("train", "test") for pair in p[split]]
    _check(len({pair["input"].shape for _, _, pair in metas}) >= 35,
           "few canvas shapes")
    _check(len({tuple((b[0], b[1], b[2], b[4]) for b in meta["bars"])
                for _, meta, _ in metas}) >= 90, "few bar layouts")
    _check({meta["background"] for _, meta, _ in metas} >= {0, 2, 3, 5, 6, 7, 8},
           "background palette is too narrow")
    _check(any(any(b[4] == 1 for b in meta["bars"]) for split, meta, _ in metas
               if split == "train"), "training never shows right bars")
    _check(any(len({b[4] for b in meta["bars"]}) == 2 for _, meta, _ in metas),
           "no mixed-side pair")
    _check(any(meta["marker_row"] > 0 for _, meta, _ in metas),
           "marker is always on top")
    _check(any(meta["background"] != 0 and b[3] == 0
               for _, meta, _ in metas for b in meta["bars"]),
           "zero is never a visible bar")
    _check({len(meta["bars"]) for split, meta, _ in metas if split == "test"}
           >= {1, 2, 3, 4}, "test bar counts are too narrow")
    _check(any(len({m["marker_color"] for m in
                     [_inspect_pair(pair["input"], pair["output"])
                      for pair in p["train"]]}) > 1 for p in puzzles),
           "marker color never varies within training")
    _check(any(all(all(b[4] == -1 for b in _inspect_pair(pair["input"], pair["output"])["bars"])
                   for split in ("train", "test") for pair in p[split])
               for p in puzzles), "no source-like left-only puzzle")
    test_metas = [meta for split, meta, _ in metas if split == "test"]
    marker_bar_counts = {sum(b[3] == meta["marker_color"] for b in meta["bars"])
                         for meta in test_metas}
    _check(0 in marker_bar_counts and 1 in marker_bar_counts and
           any(x >= 2 for x in marker_bar_counts),
           "test marker-color bar counts are too narrow")
    _check(any(len({b[2] for b in m["bars"]}) == len(m["bars"]) and
               len(m["bars"]) >= 2 for m in test_metas),
           "tests never use distinct lengths")
    _check(any(len({b[2] for b in m["bars"]}) < len(m["bars"])
               for m in test_metas), "tests never repeat lengths")
    _check(any(abs(m["marker_row"] - b[0]) == 1 for _, m, _ in metas
               for b in m["bars"]), "marker is never next to a bar row")
    _check(any(max(pair["input"].shape) < max(train["input"].shape)
               for p in puzzles for pair in p["test"] for train in p["train"]),
           "no test is smaller than a training example")
    train_padding = {meta["right_capacity"] for split, meta, _ in metas
                     if split == "train"}
    test_padding = {meta["right_capacity"] for split, meta, _ in metas
                    if split == "test"}
    _check(len(train_padding & test_padding) >= 6,
           "train and test padding have little overlap")
    novel_flags = []
    for puzzle in puzzles:
        train_colors = {bar[3] for pair in puzzle["train"]
                        for bar in _inspect_pair(pair["input"], pair["output"])["bars"]}
        for pair in puzzle["test"]:
            meta = _inspect_pair(pair["input"], pair["output"])
            novel_flags.append(any(bar[3] not in train_colors and
                                   bar[3] != meta["marker_color"]
                                   for bar in meta["bars"]))
    _check(any(novel_flags) and not all(novel_flags),
           "tests must include both new and familiar bar-color cases")
    _check(any(meta["background"] != 0 and
               len({bar[4] for bar in meta["bars"]}) == 2 and
               meta["marker_row"] > 0 and
               any(bar[3] == meta["marker_color"] for bar in meta["bars"])
               for _, meta, _ in metas),
           "compatible geometry and color axes never combine")
