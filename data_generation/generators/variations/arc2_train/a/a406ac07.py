import numpy as np


_SOURCE_PAIR_ROWS = (
    (
        (
            "0000000005",
            "0000000006",
            "0000000006",
            "0000000007",
            "0000000007",
            "0000000007",
            "0000000008",
            "0000000009",
            "0000000009",
            "5667778999",
        ),
        (
            "5000000005",
            "0660000006",
            "0660000006",
            "0007770007",
            "0007770007",
            "0007770007",
            "0000008008",
            "0000000999",
            "0000000999",
            "5667778999",
        ),
    ),
    (
        (
            "0000000009",
            "0000000009",
            "0000000008",
            "0000000008",
            "0000000007",
            "0000000007",
            "0000000006",
            "0000000006",
            "0000000005",
            "9988776655",
        ),
        (
            "9900000009",
            "9900000009",
            "0088000008",
            "0088000008",
            "0000770007",
            "0000770007",
            "0000006606",
            "0000006606",
            "0000000055",
            "9988776655",
        ),
    ),
    (
        (
            "0000000008",
            "0000000008",
            "0000000008",
            "0000000004",
            "0000000004",
            "0000000005",
            "0000000005",
            "0000000003",
            "0000000003",
            "8844455333",
        ),
        (
            "8800000008",
            "8800000008",
            "8800000008",
            "0044400004",
            "0044400004",
            "0000055005",
            "0000055005",
            "0000000333",
            "0000000333",
            "8844455333",
        ),
    ),
    (
        (
            "0000000003",
            "0000000003",
            "0000000003",
            "0000000004",
            "0000000004",
            "0000000006",
            "0000000006",
            "0000000009",
            "0000000007",
            "3346669977",
        ),
        (
            "3300000003",
            "3300000003",
            "3300000003",
            "0040000004",
            "0040000004",
            "0006660006",
            "0006660006",
            "0000009909",
            "0000000077",
            "3346669977",
        ),
    ),
)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rows_to_grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(value) for value in grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_key(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return (_grid_key(input_grid), _grid_key(output_grid))


_SOURCE_PAIR_KEYS = frozenset(
    _pair_key(_rows_to_grid(input_rows), _rows_to_grid(output_rows))
    for input_rows, output_rows in _SOURCE_PAIR_ROWS
)


def _normalize_optional_count(value, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _run_length_encode(line: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    labels, lengths = [], []
    for cell in line:
        color = int(cell)
        if labels and color == labels[-1]:
            lengths[-1] += 1
        else:
            labels.append(color)
            lengths.append(1)
    return tuple(labels), tuple(lengths)


def _guide_data(grid: np.ndarray, corner: tuple[int, int]) -> dict:
    row_end, column_end = corner
    horizontal = grid[row_end, :]
    vertical = grid[:, column_end]
    if column_end == 0:
        horizontal = horizontal[::-1]
    if row_end == 0:
        vertical = vertical[::-1]
    horizontal_labels, widths = _run_length_encode(horizontal)
    vertical_labels, heights = _run_length_encode(vertical)
    return {"corner": corner, "horizontal": horizontal_labels,
            "vertical": vertical_labels, "widths": widths, "heights": heights}


def _analyze_input(grid: np.ndarray) -> dict:
    height, width = grid.shape
    _check(height >= 4 and width >= 4, "Guided canvas is too small")
    candidates = []
    for row_end in (0, height - 1):
        for column_end in (0, width - 1):
            # Remove the two candidate guide edges, then inspect the untouched field.
            rows = slice(1, height) if row_end == 0 else slice(0, height - 1)
            columns = slice(1, width) if column_end == 0 else slice(0, width - 1)
            field = grid[rows, columns]
            background = int(field[0, 0])
            if not bool(np.all(field == background)):
                continue
            data = _guide_data(grid, (row_end, column_end))
            horizontal, vertical = data["horizontal"], data["vertical"]
            if (len(horizontal) < 4 or len(horizontal) > 7
                    or len(horizontal) != len(vertical)
                    or len(set(horizontal)) != len(horizontal)
                    or len(set(vertical)) != len(vertical)
                    or set(horizontal) != set(vertical)
                    or horizontal[-1] != vertical[-1]
                    or background in horizontal
                    or data["widths"][-1] < 2
                    or data["heights"][-1] < 2):
                continue
            data["background"] = background
            candidates.append(data)
    _check(len(candidates) == 1, "Input must have one unambiguous guide corner")
    return candidates[0]


def _intervals(labels: tuple[int, ...], lengths: tuple[int, ...],
               limit: int, at_start: bool) -> dict[int, tuple[int, int]]:
    result = {}
    offset = 0
    for color, length in zip(labels, lengths):
        if at_start:
            result[color] = (limit - offset - length, limit - offset)
        else:
            result[color] = (offset, offset + length)
        offset += length
    _check(offset == limit, "Guide does not span its edge")
    return result


def _expected_output(grid: np.ndarray, data: dict) -> np.ndarray:
    height, width = grid.shape
    row_end, column_end = data["corner"]
    row_intervals = _intervals(data["vertical"], data["heights"],
                               height, row_end == 0)
    column_intervals = _intervals(data["horizontal"], data["widths"],
                                  width, column_end == 0)
    answer = grid.copy()
    for color in data["horizontal"]:
        top, bottom = row_intervals[color]
        left, right = column_intervals[color]
        answer[top:bottom, left:right] = color
    return answer


def _validate_grid(grid: np.ndarray, label: str) -> None:
    _check(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _check(grid.ndim == 2, f"{label} must have two dimensions")
    _check(all(1 <= side <= 30 for side in grid.shape), f"{label} size is illegal")
    _check(np.issubdtype(grid.dtype, np.integer), f"{label} must be integer")
    _check(bool(np.all((0 <= grid) & (grid <= 9))), f"{label} has an illegal color")


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "Puzzle must have exactly train and test")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
           "Splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
           "Illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
                   f"{label} has incorrect fields")
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp, label + " input")
            _validate_grid(out, label + " output")
            _check(inp.shape == out.shape, f"{label} changed size")
            input_key, pair_key = _grid_key(inp), _pair_key(inp, out)
            _check(input_key not in seen_inputs and pair_key not in seen_pairs,
                   "Duplicate pair or input")
            _check(pair_key not in _SOURCE_PAIR_KEYS, "Exact source pair copied")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            data = _analyze_input(inp)
            expected = _expected_output(inp, data)
            _check(np.array_equal(out, expected), f"{label} has incorrect color match")
            changed = set(int(v) for v in np.unique(out[inp != out]))
            _check(changed == set(data["horizontal"]),
                   f"{label} contains a guide color with no visible fill")
            _check(not np.array_equal(inp, out), f"{label} is unchanged")
            infos[split].append(data)

    train, test = infos["train"], infos["test"]
    backgrounds = {item["background"] for item in train + test}
    _check(len(backgrounds) == 1, "Background convention changes within a puzzle")
    _check({4, 5}.issubset({len(item["horizontal"]) for item in train}),
           "Training must show four and five labels")
    _check(any(item["horizontal"] != item["vertical"] for item in train),
           "Training must demonstrate literal color matching")
    _check(any(_lengths_equal(item) for item in train),
           "Training must show equal corresponding lengths")
    _check(any(not _lengths_equal(item) for item in train),
           "Training must show independent corresponding lengths")
    train_corners = {tuple(coordinate == 0 for coordinate in item["corner"])
                     for item in train}
    train_sizes = {(sum(item["heights"]), sum(item["widths"])) for item in train}
    for item in test:
        _check(tuple(coordinate == 0 for coordinate in item["corner"])
               in train_corners,
               "Test corner was not demonstrated in training")
        _check(len(item["horizontal"]) <= max(len(t["horizontal"]) for t in train),
               "Test uses an unseen guide count")
        if len(train_sizes) == 1:
            _check((sum(item["heights"]), sum(item["widths"])) in train_sizes,
                   "Test size changes without training size variation")


def _lengths_equal(data: dict) -> bool:
    widths = dict(zip(data["horizontal"], data["widths"]))
    heights = dict(zip(data["vertical"], data["heights"]))
    return all(widths[color] == heights[color] for color in widths)


def _composition(rng: np.random.Generator, total: int,
                 minima: tuple[int, ...], frozen: tuple[int, ...] = ()) -> tuple[int, ...]:
    extra = total - sum(minima)
    _check(extra >= 0, "Canvas too small for guide runs")
    if extra == 0:
        return minima
    active = [index for index in range(len(minima)) if index not in frozen]
    _check(bool(active), "No run can receive the remaining cells")
    weights = rng.dirichlet(np.full(len(active), 1.2))
    additions = np.zeros(len(minima), dtype=np.int64)
    additions[active] = rng.multinomial(extra, weights)
    return tuple(int(base + additional) for base, additional in zip(minima, additions))


def _ordered(rng: np.random.Generator, palette: tuple[int, ...],
             corner_color: int) -> tuple[int, ...]:
    other = [color for color in palette if color != corner_color]
    return tuple(int(v) for v in rng.permutation(other)) + (corner_color,)


def _make_pair(rng: np.random.Generator, palette: tuple[int, ...],
               background: int, corner: tuple[int, int], shape: tuple[int, int],
               mode: str, off_diagonal: bool,
               template: dict | None = None) -> dict:
    height, width = shape
    color = int(rng.choice(np.asarray(palette, dtype=np.int8)))
    horizontal = _ordered(rng, palette, color)
    if off_diagonal:
        other = list(horizontal[:-1])
        swap = rng.choice(len(other), size=2, replace=False)
        other[int(swap[0])], other[int(swap[1])] = other[int(swap[1])], other[int(swap[0])]
        vertical = tuple(other) + (color,)
    else:
        vertical = horizontal
    n = len(palette)
    if template is not None:
        widths = template["widths"]
        heights = template["heights"]
        _check((sum(heights), sum(widths)) == shape and len(widths) == n,
               "Invalid contrast template")
    else:
        basic = (1,) * (n - 1) + (2,)
        solid = (2,) * n
        if mode == "equal":
            _check(height == width, "Equal bands need a square canvas")
            widths = _composition(rng, width, basic)
            by_color = dict(zip(horizontal, widths))
            heights = tuple(by_color[c] for c in vertical)
        elif mode in ("vertical", "horizontal", "mixed"):
            a = int(rng.integers(0, n - 1))
            b = int(rng.integers(0, n - 1))
            if mode == "mixed":
                # The narrow vertical and horizontal bands belong to different colors.
                while vertical[b] == horizontal[a]:
                    b = (b + 1) % (n - 1)
            width_min = list(solid)
            height_min = list(solid)
            if mode in ("vertical", "mixed"):
                width_min[a] = 1
            if mode in ("horizontal", "mixed"):
                height_min[b] = 1
            widths = _composition(rng, width, tuple(width_min),
                                  (a,) if mode in ("vertical", "mixed") else ())
            heights = _composition(rng, height, tuple(height_min),
                                   (b,) if mode in ("horizontal", "mixed") else ())
        elif mode == "general":
            widths = _composition(rng, width, basic)
            heights = _composition(rng, height, basic)
            if dict(zip(horizontal, widths)) == dict(zip(vertical, heights)):
                heights = heights[1:-1] + heights[:1] + heights[-1:]
        else:
            raise ValueError(f"Unknown mode {mode}")
    input_grid = np.full(shape, background, dtype=np.int8)
    row_end, column_end = corner
    horizontal_line = np.repeat(np.asarray(horizontal, dtype=np.int8), widths)
    vertical_line = np.repeat(np.asarray(vertical, dtype=np.int8), heights)
    if column_end == 0:
        horizontal_line = horizontal_line[::-1]
    if row_end == 0:
        vertical_line = vertical_line[::-1]
    input_grid[row_end, :] = horizontal_line
    input_grid[:, column_end] = vertical_line
    data = {"corner": corner, "horizontal": horizontal, "vertical": vertical,
            "widths": widths, "heights": heights}
    return {"input": input_grid, "output": _expected_output(input_grid, data)}


def _count_pair(seed: int, rng: np.random.Generator,
                num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _normalize_optional_count(num_train, "num_train")
    test = _normalize_optional_count(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = 1 + seed % 4
        train = max(2, test + 1) + (seed // 4) % (11 - max(2, test + 1))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _shape(rng: np.random.Generator, count: int, mode: str,
           fixed: int | None) -> tuple[int, int]:
    if fixed is not None:
        return fixed, fixed
    minimum = max(10, 2 * count)
    if mode == "equal":
        side = int(rng.integers(minimum, 23))
        return side, side
    return (int(rng.integers(minimum, 23)),
            int(rng.integers(minimum, 23)))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _count_pair(seed, rng, num_train, num_test)
    background = int(rng.integers(0, 10))
    available = tuple(color for color in range(10) if color != background)
    bank_size = int(rng.integers(6, 9))
    bank = tuple(int(v) for v in rng.choice(available, size=bank_size, replace=False))
    # Background and guide-color roles are the only sampled puzzle-level semantics.
    first = tuple(int(v) for v in rng.choice(bank, size=4, replace=False))
    missing = [color for color in bank if color not in first]
    second = tuple(missing + [int(v) for v in rng.choice(first, size=5-len(missing), replace=False)])
    palettes = [first, second]
    for index in range(2, train_count):
        if index == 2 and train_count >= 4:
            count = 4
        elif index == 3 and train_count >= 4:
            palettes.append(second)
            continue
        elif index == 2 or index == 4:
            count = 6
        elif index == 5 and bank_size >= 7:
            count = 7
        else:
            count = int(rng.integers(4, min(7, bank_size) + 1))
        if index == 2 and train_count >= 4:
            retained = tuple(int(v) for v in rng.choice(first, size=3, replace=False))
            replacement = int(rng.choice([v for v in bank if v not in first]))
            palette = retained + (replacement,)
        else:
            palette = tuple(int(v) for v in rng.choice(bank, size=count, replace=False))
        palettes.append(palette)

    fixed_size = int(rng.integers(14, 23)) if rng.random() < 0.36 else None
    all_corners = ((0, 0), (0, -1), (-1, 0), (-1, -1))
    corner_pool = [all_corners[int(rng.integers(0, 4))]]
    if train_count >= 3 and rng.random() < 0.62:
        remaining = [corner for corner in all_corners if corner != corner_pool[0]]
        count = min(train_count - (2 if train_count >= 4 else 0),
                    int(rng.integers(2, 4)))
        corner_pool += [tuple(v) for v in rng.choice(np.asarray(remaining), size=count-1, replace=False)]
    train_corners = [corner_pool[0]]
    train_corners += [corner_pool[1] if len(corner_pool) > 1 else corner_pool[0]]
    for index in range(2, train_count):
        if index == 2 and train_count >= 4:
            train_corners.append(train_corners[0])
        elif index == 3 and train_count >= 4:
            train_corners.append(train_corners[1])
        elif len(corner_pool) > 2 and index == 4:
            train_corners.append(corner_pool[2])
        else:
            train_corners.append(corner_pool[int(rng.integers(0, len(corner_pool)))])

    train = []
    modes = ("equal", "mixed", "vertical", "horizontal", "general")
    for index, palette in enumerate(palettes):
        if index == 0:
            mode = "equal"
        elif index == 1:
            mode = "mixed"
        elif index == 2 and train_count >= 4:
            mode = "equal"
        elif index == 3 and train_count >= 4:
            mode = "horizontal"
        else:
            mode = modes[int(rng.integers(0, len(modes)))]
        shape = _shape(rng, len(palette), mode, fixed_size)
        template = None
        if index == 1 and fixed_size is None:
            if shape[0] == shape[1]:
                shape = (shape[0], 10 if shape[1] == 22 else shape[1] + 1)
        if index == 2 and train_count >= 4:
            source = _analyze_input(train[0]["input"])
            template = source
            shape = train[0]["input"].shape
        if index == 3 and train_count >= 4:
            shape = train[1]["input"].shape
        corner = train_corners[index]
        actual_corner = (0 if corner[0] == 0 else shape[0]-1,
                         0 if corner[1] == 0 else shape[1]-1)
        pair = _make_pair(rng, palette, background, actual_corner,
                          shape, mode, index != 0, template)
        train.append(pair)

    test = []
    train_palette_sets = {frozenset(palette) for palette in palettes}
    for index in range(test_count):
        count = 4 + ((seed + index) % 2)
        palette_mode = (seed + index) % 3
        if palette_mode == 0:
            options = [p for p in palettes if len(p) == count]
            palette = options[int(rng.integers(0, len(options)))]
        elif palette_mode == 1:
            palette = tuple(int(v) for v in rng.choice(bank, size=count, replace=False))
            for _ in range(15):
                if frozenset(palette) not in train_palette_sets:
                    break
                palette = tuple(int(v) for v in rng.choice(bank, size=count, replace=False))
        else:
            unseen = [color for color in available if color not in bank]
            palette = tuple(int(v) for v in rng.choice(bank, size=count-1, replace=False)) + (
                int(rng.choice(unseen)),)
        mode = modes[(seed + index) % len(modes)]
        shape = _shape(rng, count, mode, fixed_size)
        corner = train_corners[int(rng.integers(0, len(train_corners)))]
        actual_corner = (0 if corner[0] == 0 else shape[0]-1,
                         0 if corner[1] == 0 else shape[1]-1)
        test.append(_make_pair(rng, palette, background, actual_corner,
                               shape, mode, bool(rng.integers(0, 2))))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(np.array_equal(a[role], b[role])
               for split in ("train", "test")
               for a, b in zip(first[split], second[split])
               for role in ("input", "output"))


def _direct_rule(input_grid: np.ndarray, corner: tuple[int, int]) -> np.ndarray:
    """Cellwise reconstruction, separate from the interval-based constructor."""
    guide_row, guide_column = corner
    answer = input_grid.copy()
    for row in range(input_grid.shape[0]):
        for column in range(input_grid.shape[1]):
            if row == guide_row or column == guide_column:
                continue
            row_label = input_grid[row, guide_column]
            column_label = input_grid[guide_row, column]
            if row_label == column_label:
                answer[row, column] = row_label
    return answer


def _narrow_class(data: dict) -> str:
    widths = dict(zip(data["horizontal"], data["widths"]))
    heights = dict(zip(data["vertical"], data["heights"]))
    vertical = any(widths[c] == 1 and heights[c] > 1 for c in widths)
    horizontal = any(heights[c] == 1 and widths[c] > 1 for c in widths)
    return ("mixed" if vertical and horizontal else "vertical" if vertical
            else "horizontal" if horizontal else "neither")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    _check(len(train_counts) >= 4 and len(test_counts) >= 3,
           "Automatic pair counts lack diversity")
    _check(len({_grid_key(p["test"][0]["input"]) for p in puzzles}) >= 40,
           "Different seeds generated too few distinct puzzles")
    records = [(split, pair, _analyze_input(pair["input"]))
               for p in puzzles for split in ("train", "test")
               for pair in p[split]]
    for split, pair, data in records:
        _check(np.array_equal(_direct_rule(pair["input"], data["corner"]),
                              pair["output"]),
               f"Independent cellwise rule failed in {split}")
        _check(data["widths"][-1] > 1 and data["heights"][-1] > 1,
               "A terminal guide run did no visible work")
    for source_input, source_output in _SOURCE_PAIR_ROWS:
        source_grid = _rows_to_grid(source_input)
        data = _analyze_input(source_grid)
        _check(np.array_equal(_direct_rule(source_grid, data["corner"]),
                              _rows_to_grid(source_output)),
               "Cellwise rule does not explain the source task")
    _check(len({r[2]["background"] for r in records}) >= 7,
           "Background variation is too narrow")
    _check(len({tuple(v == 0 for v in r[2]["corner"]) for r in records}) == 4,
           "All four corners were not exercised")
    _check(len({r[1]["input"].shape for r in records}) >= 40,
           "Canvas size variation is too narrow")
    _check(len({_grid_key(r[1]["input"]) for r in records}) >= 200,
           "Too few distinct input grids")
    _check({6, 7, 8}.issubset({len(set().union(*(
               set(_analyze_input(pair["input"])["horizontal"])
               for pair in p["train"]))) for p in puzzles}),
           "Training color-bank sizes lack coverage")
    _check(any(len({p["input"].shape for p in puzzle["train"]}) == 1
               for puzzle in puzzles)
           and any(len({p["input"].shape for p in puzzle["train"]}) > 1
                   for puzzle in puzzles),
           "Fixed and varied canvas regimes were not both observed")
    _check(any(len({tuple(v == 0 for v in _analyze_input(p["input"])["corner"])
                    for p in puzzle["train"]}) > 1 for puzzle in puzzles),
           "No puzzle demonstrates corner changes in training")
    _check(any(r[2]["horizontal"] != r[2]["vertical"] for r in records if r[0] == "test"),
           "No off-diagonal test")
    _check(any(r[2]["horizontal"] == r[2]["vertical"] for r in records if r[0] == "test"),
           "No aligned-order test")
    _check({4, 5}.issubset({len(r[2]["horizontal"]) for r in records if r[0] == "test"}),
           "Tests do not cover four and five labels")
    _check(any(_lengths_equal(r[2]) for r in records if r[0] == "test"),
           "Tests do not cover equal lengths")
    _check(any(not _lengths_equal(r[2]) for r in records if r[0] == "test"),
           "Tests do not cover independent lengths")
    _check({"mixed", "vertical", "horizontal", "neither"}.issubset(
               {_narrow_class(r[2]) for r in records if r[0] == "test"}),
           "Test rectangle regimes lack coverage")
    _check({6, 7}.issubset({len(r[2]["horizontal"])
                           for r in records if r[0] == "train"}),
           "Larger training guides lack coverage")
    palette_regimes = set()
    contrast_count = 0
    combined_count = 0
    for puzzle in puzzles:
        training = [_analyze_input(pair["input"]) for pair in puzzle["train"]]
        bank = set().union(*(set(item["horizontal"]) for item in training))
        palettes = {frozenset(item["horizontal"]) for item in training}
        for pair in puzzle["test"]:
            data = _analyze_input(pair["input"])
            colors = set(data["horizontal"])
            regime = ("new" if colors - bank else "reuse"
                      if frozenset(colors) in palettes else "fresh")
            palette_regimes.add(regime)
            combined_count += (data["horizontal"] != data["vertical"]
                               and pair["input"].shape[0] != pair["input"].shape[1]
                               and bool(colors - bank)
                               and data["background"] != 0)
        if len(training) >= 4:
            first, second, third, fourth = training[:4]
            contrast_count += (first["widths"] == third["widths"]
                               and first["heights"] == third["heights"]
                               and set(first["horizontal"]) != set(third["horizontal"]))
            _check(set(second["horizontal"]) == set(fourth["horizontal"])
                   and (second["widths"], second["heights"])
                   != (fourth["widths"], fourth["heights"]),
                   "Same-palette, different-length contrast is missing")
    _check(palette_regimes == {"new", "reuse", "fresh"},
           "Test palette regimes lack coverage")
    _check(contrast_count >= 10, "Same-length, different-color contrasts are rare")
    _check(combined_count >= 1,
           "No test combines new color, off-diagonal matching, rectangle, and colored background")
    for seed, kwargs in ((100, {"num_train": 2, "num_test": 1}),
                         (101, {"num_train": 10, "num_test": 1}),
                         (102, {"num_train": 10, "num_test": 4}),
                         (103, {"num_train": None, "num_test": 1}),
                         (104, {"num_train": None, "num_test": 4}),
                         (105, {"num_train": 2, "num_test": None}),
                         (106, {"num_train": 10, "num_test": None})):
        p = generate_puzzle(seed, **kwargs)
        validate_generated_puzzle(p)
        if kwargs["num_train"] is not None:
            _check(len(p["train"]) == kwargs["num_train"],
                   "Explicit training count was ignored")
        if kwargs["num_test"] is not None:
            _check(len(p["test"]) == kwargs["num_test"],
                   "Explicit test count was ignored")
        _check(_puzzles_equal(p, generate_puzzle(seed, **kwargs)),
               "Generation is nondeterministic")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(900, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Invalid count accepted: {kwargs}")
    for bad_seed in (-1, 1.5, True):
        try:
            generate_puzzle(bad_seed)
        except ValueError:
            continue
        raise AssertionError(f"Invalid seed accepted: {bad_seed}")
