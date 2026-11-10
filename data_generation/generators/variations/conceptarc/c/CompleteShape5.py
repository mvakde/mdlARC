import numpy as np


_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
_SOURCE_INPUTS = {
    ("400330", "440030", "000000", "040030", "044033", "000000"),
    ("000400", "330440", "030000", "000000", "440300", "040330"),
    ("300000", "330300", "000330", "440000", "040033", "000003"),
    ("0300000000", "3303000300", "0003303300", "0000000000",
     "0300000000", "3303000000", "0003300030", "0000000033",
     "0003000000", "0033000000"),
    ("4000000004", "4400000044", "0000400400", "4004404400",
     "4400000000", "0000400400", "4004400440", "4400000000",
     "0004000000", "0044000000"),
}
def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _resolve_counts(rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid_key(grid):
    return tuple(grid.shape), tuple(int(x) for x in grid.flat)


def _rows(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _compatible(a, b):
    """Boxes may meet diagonally but cannot share a cell or an edge."""
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return dr >= 3 or dc >= 3 or (dr == 2 and dc == 2)


def _anchors(rng, height, width, count, seek_contact):
    candidates = [(r, c) for r in range(height - 1)
                  for c in range(width - 1)]
    for attempt in range(64):
        chosen = []
        if seek_contact and attempt < 32:
            for index in rng.permutation(len(candidates)):
                first = candidates[int(index)]
                diagonal = [(first[0] + dr, first[1] + dc)
                            for dr in (-2, 2) for dc in (-2, 2)]
                diagonal = [p for p in diagonal
                            if 0 <= p[0] < height - 1
                            and 0 <= p[1] < width - 1]
                if diagonal:
                    chosen = [first, diagonal[int(rng.integers(len(diagonal)))]]
                    break
            if len(chosen) == count:
                return chosen
        for index in rng.permutation(len(candidates)):
            point = candidates[int(index)]
            if all(_compatible(point, old) for old in chosen):
                chosen.append(point)
                if len(chosen) == count:
                    return chosen
    # This lattice always has at least the sampled capacity.
    lattice = [(r, c) for r in range(0, height - 1, 3)
               for c in range(0, width - 1, 3)]
    return [lattice[int(i)] for i in rng.permutation(len(lattice))[:count]]


def _dimensions(rng, regime, train_dimensions):
    if regime == "target_test" and train_dimensions and rng.random() < 0.22:
        return train_dimensions[int(rng.integers(len(train_dimensions)))]
    if rng.random() < 0.12:
        return (6, 6) if regime == "source_train" else (10, 10)
    if regime == "source_train":
        return int(rng.integers(5, 14)), int(rng.integers(5, 14))
    return int(rng.integers(5, 17)), int(rng.integers(5, 17))


def _generate_pair(rng, regime, background, colors, train_dimensions,
                   force_mixed=False, force_mono=False,
                   dimensions=None, object_count=None):
    if dimensions is None:
        dimensions = _dimensions(rng, regime, train_dimensions)
    height, width = dimensions
    capacity = ((height + 1) // 3) * ((width + 1) // 3)
    if object_count is None:
        limit = min(9 if regime == "source_train" else 12, capacity)
        object_count = int(rng.integers(2, limit + 1))
    seek_contact = bool(rng.random() < 0.32 and object_count < capacity)
    anchors = _anchors(rng, height, width, object_count, seek_contact)

    mono = force_mono or (not force_mixed and rng.random() <
                          (0.35 if regime == "source_train" else 0.48))
    if mono:
        object_colors = [colors[int(rng.integers(2))]] * object_count
    else:
        object_colors = [colors[int(x)] for x in
                         rng.integers(0, 2, size=object_count)]
        if len(set(object_colors)) == 1:
            index = int(rng.integers(object_count))
            object_colors[index] = (colors[1] if object_colors[0] == colors[0]
                                    else colors[0])

    missing_ids = [int(x) for x in rng.integers(0, 4, size=object_count)]
    forbidden = [set() for _ in anchors]
    for i, first in enumerate(anchors):
        for j in range(i + 1, len(anchors)):
            second = anchors[j]
            if abs(first[0] - second[0]) == 2 and abs(first[1] - second[1]) == 2:
                facing = (int(second[0] > first[0]),
                          int(second[1] > first[1]))
                forbidden[i].add(_CORNERS.index(facing))
                forbidden[j].add(_CORNERS.index((1 - facing[0], 1 - facing[1])))
    for i, disallowed in enumerate(forbidden):
        options = [corner for corner in range(4) if corner not in disallowed]
        if options and missing_ids[i] in disallowed:
            missing_ids[i] = options[int(rng.integers(len(options)))]

    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    for (row, col), color, missing_id in zip(anchors, object_colors, missing_ids):
        for corner_id, (dr, dc) in enumerate(_CORNERS):
            if corner_id != missing_id:
                inp[row + dr, col + dc] = color
                out[row + dr, col + dc] = color
        dr, dc = _CORNERS[missing_id]
        out[row + dr, col + dc] = colors[1] if color == colors[0] else colors[0]
    return {"input": inp, "output": out}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    background = 0 if rng.random() < 0.50 else int(rng.integers(1, 10))
    colors = tuple(int(x) for x in rng.choice(
        [x for x in range(10) if x != background], size=2, replace=False))
    source_like = bool(rng.random() < 0.16)
    puzzle = {"train": [], "test": []}
    seen = set()

    def add_pair(split, index, force_mixed=False, force_mono=False,
                 dimensions=None, object_count=None):
        for _ in range(40):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            pair = _generate_pair(
                pair_rng, "source_train" if split == "train" else "target_test",
                background, colors, [p["input"].shape for p in puzzle["train"]],
                force_mixed, force_mono, dimensions, object_count)
            key = _grid_key(pair["input"])
            if key not in seen and _rows(pair["input"]) not in _SOURCE_INPUTS:
                seen.add(key)
                puzzle[split].append(pair)
                return
        raise RuntimeError(f"could not make unique {split} pair {index}")

    for index in range(train_count):
        special = source_like and index == 0
        add_pair("train", index, force_mixed=(index == 0),
                 force_mono=(index == 1 and rng.random() < 0.65),
                 dimensions=(6, 6) if special else None,
                 object_count=4 if special else None)
    mono_test_index = int(rng.integers(test_count)) if test_count > 1 else -1
    large_test_index = int(rng.integers(test_count)) if source_like else -1
    for index in range(test_count):
        special = index == large_test_index
        add_pair("test", index, force_mono=(index == mono_test_index),
                 dimensions=(10, 10) if special else None,
                 object_count=int(rng.integers(7, 10)) if special else None)
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid):
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
             "grid must be a two-dimensional ndarray")
    _require(all(1 <= size <= 30 for size in grid.shape), "illegal grid size")
    _require(grid.dtype.kind in ("i", "u"), "grid must contain integers")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "illegal grid color")


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _shapes(grid, background):
    pending = {tuple(map(int, point)) for point in np.argwhere(grid != background)}
    shapes = []
    while pending:
        start = pending.pop()
        cells = {start}
        stack = [start]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in pending:
                    pending.remove(neighbor)
                    cells.add(neighbor)
                    stack.append(neighbor)
        _require(len(cells) == 3, "foreground component is not three cells")
        top = min(row for row, _ in cells)
        left = min(col for _, col in cells)
        box = {(top + dr, left + dc) for dr, dc in _CORNERS}
        _require(cells <= box and len(box - cells) == 1,
                 "foreground component is not an L")
        object_colors = {int(grid[r, c]) for r, c in cells}
        _require(len(object_colors) == 1, "L is not monochromatic")
        shapes.append((top, left, object_colors.pop(), next(iter(box - cells))))
    shapes.sort()
    for i, first in enumerate(shapes):
        for second in shapes[i + 1:]:
            _require(_compatible(first, second), "L boxes merge or compete")
    return shapes


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list),
             "train and test must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4
             and len(train) > len(test), "illegal pair counts")
    pairs = train + test
    for pair in pairs:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                 "pair needs input and output")
        _validate_grid(pair["input"])
        _validate_grid(pair["output"])
    background = _background(train[0]["input"])
    colors = set()
    for pair in train:
        _require(_background(pair["input"]) == background,
                 "training background changed")
        colors.update(int(x) for x in np.unique(pair["input"]) if x != background)
    _require(len(colors) == 2, "training does not establish two colors")
    mapping = {color: set() for color in colors}
    seen_inputs = set()
    seen_pairs = set()
    for split, group in (("train", train), ("test", test)):
        for pair in group:
            inp, out = pair["input"], pair["output"]
            input_key, output_key = _grid_key(inp), _grid_key(out)
            _require(input_key not in seen_inputs, "duplicate input")
            _require((input_key, output_key) not in seen_pairs, "duplicate pair")
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            _require(_rows(inp) not in _SOURCE_INPUTS, "source input copied")
            _require(inp.shape == out.shape, "output changed grid size")
            _require(_background(inp) == background, "pair background changed")
            _require(set(int(x) for x in np.unique(inp)) <= colors | {background},
                     "foreign input color")
            _require(set(int(x) for x in np.unique(out)) <= colors | {background},
                     "foreign output color")
            shapes = _shapes(inp, background)
            _require(len(shapes) >= 2, "too few Ls")
            expected = inp.copy()
            for _, _, color, missing in shapes:
                complement = next(iter(colors - {color}))
                expected[missing] = complement
                if split == "train":
                    mapping[color].add(int(out[missing]))
            _require(np.array_equal(out, expected), "incorrect L completion")
            _require(np.count_nonzero(inp != out) == len(shapes),
                     "wrong number of changed cells")
    for color in colors:
        _require(mapping[color] == colors - {color},
                 "training does not show both complement directions")


def _same_puzzle(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(pa["input"], pb["input"]) and
        np.array_equal(pa["output"], pb["output"])
        for pa, pb in zip(a[split], b[split])) for split in ("train", "test"))


def _visible_diagonal_contact(shapes):
    for i, first in enumerate(shapes):
        for second in shapes[i + 1:]:
            if abs(first[0] - second[0]) != 2 or abs(first[1] - second[1]) != 2:
                continue
            facing_first = (first[0] + int(second[0] > first[0]),
                            first[1] + int(second[1] > first[1]))
            facing_second = (second[0] + int(first[0] > second[0]),
                             second[1] + int(first[1] > second[1]))
            if first[3] != facing_first and second[3] != facing_second:
                return True
    return False


def self_test() -> None:
    train_counts, test_counts = set(), set()
    backgrounds, palettes, dimensions, object_counts = set(), set(), set(), set()
    signatures = set()
    diagonal_pairs = mono_train = large_mixed = non_cartesian = 0
    repeated_corners = imbalanced = absent_corners = no_blank_row = 0
    rectangular = foreground_zero = same_size_test = 0
    larger_test = smaller_test = 0
    within_size_variety = within_count_variety = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_same_puzzle(puzzle, generate_puzzle(seed)), "nondeterministic puzzle")
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        bg = _background(puzzle["train"][0]["input"])
        backgrounds.add(bg)
        palettes.add(frozenset(int(x) for p in puzzle["train"]
                               for x in np.unique(p["input"]) if x != bg))
        signatures.add(tuple(_grid_key(p["input"]) for split in ("train", "test")
                             for p in puzzle[split]))
        puzzle_sizes = set()
        puzzle_counts = set()
        train_sizes = {pair["input"].shape for pair in puzzle["train"]}
        train_areas = [pair["input"].size for pair in puzzle["train"]]
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                shapes = _shapes(inp, bg)
                dimensions.add(inp.shape)
                object_counts.add(len(shapes))
                puzzle_sizes.add(inp.shape)
                puzzle_counts.add(len(shapes))
                used_colors = {s[2] for s in shapes}
                mono_train += int(split == "train" and len(used_colors) == 1)
                large_mixed += int(split == "test" and inp.size >= 80
                                   and len(used_colors) == 2)
                corners = [(s[3][0] - s[0], s[3][1] - s[1]) for s in shapes]
                repeated_corners += int(len(set(corners)) < len(corners))
                absent_corners += int(len(set(corners)) < 4)
                no_blank_row += int(not any(np.all(row == bg) for row in inp))
                rectangular += int(inp.shape[0] != inp.shape[1])
                foreground_zero += int(0 in used_colors)
                if split == "test":
                    same_size_test += int(inp.shape in train_sizes)
                    larger_test += int(inp.size > min(train_areas))
                    smaller_test += int(inp.size < max(train_areas))
                counts = [sum(s[2] == color for s in shapes) for color in used_colors]
                imbalanced += int(len(counts) == 2 and counts[0] != counts[1])
                anchors = {(s[0], s[1]) for s in shapes}
                rows = {s[0] for s in shapes}
                cols = {s[1] for s in shapes}
                non_cartesian += int(anchors != {(r, c) for r in rows for c in cols})
                diagonal_pairs += int(_visible_diagonal_contact(shapes))
                _require({tuple(map(int, p)) for p in np.argwhere(inp != out)}
                         == {s[3] for s in shapes}, "changed wrong cells")
        within_size_variety += int(len(puzzle_sizes) > 1)
        within_count_variety += int(len(puzzle_counts) > 1)
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "pair count diversity too narrow")
    _require(len(backgrounds) >= 5 and len(palettes) >= 12,
             "palette diversity too narrow")
    _require(len(dimensions) >= 30 and len(object_counts) >= 7,
             "geometry diversity too narrow")
    _require(len(signatures) >= 40, "seed diversity too narrow")
    _require(diagonal_pairs >= 5 and mono_train >= 5 and large_mixed >= 5,
             "contact or color regime missing")
    _require(non_cartesian >= 25 and repeated_corners >= 30 and imbalanced >= 15,
             "layout or local motif diversity too narrow")
    _require(absent_corners >= 20 and no_blank_row >= 5 and rectangular >= 30,
             "corner, row, or aspect-ratio diversity too narrow")
    _require(foreground_zero >= 3 and same_size_test >= 3
             and larger_test >= 10 and smaller_test >= 10,
             "palette or size regimes too narrow")
    _require(within_size_variety >= 30 and within_count_variety >= 30,
             "within-puzzle pair variation too narrow")
    legal = ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
             (2, None), (10, None))
    for offset, (nt, ns) in enumerate(legal):
        puzzle = generate_puzzle(1000 + offset, nt, ns)
        validate_generated_puzzle(puzzle)
        _require(_same_puzzle(puzzle, generate_puzzle(1000 + offset, nt, ns)),
                 "requested counts are nondeterministic")
        _require((nt is None or len(puzzle["train"]) == nt)
                 and (ns is None or len(puzzle["test"]) == ns),
                 "requested counts ignored")
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 3), (3, 4), (2, 4)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
