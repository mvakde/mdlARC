import numpy as np


_SOURCE_ROWS = (
    (
        (
            "0000000000",
            "0444440000",
            "0444440000",
            "0444440000",
            "0444440000",
            "0444440000",
            "0000000000",
            "0000044440",
            "0000044440",
            "0000044440",
        ),
        (
            "0000000000",
            "0444440000",
            "0422240000",
            "0422240000",
            "0422240000",
            "0444440000",
            "0000000000",
            "0000044440",
            "0000041140",
            "0000044440",
        ),
    ),
    (
        (
            "0000000000",
            "0444400000",
            "0444400000",
            "0444400000",
            "0444400000",
            "0000000000",
            "0004444440",
            "0004444440",
            "0004444440",
            "0004444440",
        ),
        (
            "0000000000",
            "0444400000",
            "0411400000",
            "0411400000",
            "0444400000",
            "0000000000",
            "0004444440",
            "0004222240",
            "0004222240",
            "0004444440",
        ),
    ),
    (
        (
            "4444440000",
            "4444440000",
            "4444440000",
            "4444440000",
            "4444440000",
            "4444440000",
            "0000000000",
            "0000444444",
            "0000444444",
            "0000444444",
        ),
        (
            "4444440000",
            "4222240000",
            "4222240000",
            "4222240000",
            "4222240000",
            "4444440000",
            "0000000000",
            "0000444444",
            "0000411114",
            "0000444444",
        ),
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rows_key(rows: tuple[str, ...]) -> tuple:
    return (
        len(rows),
        len(rows[0]),
        tuple(int(ch) for row in rows for ch in row),
    )


_SOURCE_INPUT_KEYS = frozenset(
    _rows_key(source_input) for source_input, _ in _SOURCE_ROWS
)
_SOURCE_PAIR_KEYS = frozenset(
    (_rows_key(source_input), _rows_key(source_output))
    for source_input, source_output in _SOURCE_ROWS
)


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        int(grid.shape[0]),
        int(grid.shape[1]),
        tuple(int(value) for value in grid.ravel()),
    )


def _coerce_count(
    value: int | None,
    name: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, inclusive"
        )
    return result


def _resolve_counts(
    rng: np.random.Generator,
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _coerce_count(num_train, "num_train", 2, 10)
    test_count = _coerce_count(num_test, "num_test", 1, 4)

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        # This schedule guarantees count diversity across consecutive seeds.
        test_count = 1 + seed % 4
        minimum_train = max(2, test_count + 1)
        number_of_choices = 11 - minimum_train
        train_count = minimum_train + (seed // 4) % number_of_choices
    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        if minimum_train > 10:
            raise ValueError("no legal num_train exists for this num_test")
        train_count = int(rng.integers(minimum_train, 11))
    else:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal num_test exists for this num_train")
        test_count = int(rng.integers(1, maximum_test + 1))

    if not (
        2 <= train_count <= 10
        and 1 <= test_count <= 4
        and train_count > test_count
    ):
        raise ValueError("the resolved pair counts are illegal")

    return train_count, test_count


def _sample_puzzle_parameters(rng: np.random.Generator, seed: int) -> dict:
    accents = [int(c) for c in rng.choice(10, size=2, replace=False)]
    available = tuple(c for c in range(10) if c not in accents)
    fixed_shapes = ((10, 10), (12, 12), (14, 10), (10, 14), (16, 16))
    fixed_shape = fixed_shapes[(seed // 3) % len(fixed_shapes)] if seed % 3 == 0 else None
    return {
        "small_accent": accents[0],
        "large_accent": accents[1],
        "available": available,
        "fixed_shape": fixed_shape,
        "classic": seed % 5 == 0,
    }


def _comparison_kind(rectangles: list[tuple[int, int, int, int, int]]) -> str:
    small, large = rectangles
    if small[3] == large[3]:
        return "height"
    if small[2] == large[2]:
        return "width"
    return "both"


def _dimension_choices(
    shape: tuple[int, int], kind: str, mode: str, maximum: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    grid_h, grid_w = shape
    choices = []
    for small_h in range(3, min(maximum, grid_h) + 1):
        for small_w in range(3, min(maximum, grid_w) + 1):
            height_steps = (1, 2, 3, 4) if kind != "width" else (0,)
            width_steps = (1, 2, 3, 4) if kind != "height" else (0,)
            for dh in height_steps:
                for dw in width_steps:
                    if kind == "both" and (dh == 0 or dw == 0):
                        continue
                    large_h, large_w = small_h + dh, small_w + dw
                    if large_h > maximum or large_w > maximum:
                        continue
                    sep_padding = 3 if mode == "inset" else 1
                    orth_padding = 2 if mode == "inset" else 0
                    vertical = (small_h + large_h + sep_padding <= grid_h
                                and max(small_w, large_w) + orth_padding <= grid_w)
                    horizontal = (small_w + large_w + sep_padding <= grid_w
                                  and max(small_h, large_h) + orth_padding <= grid_h)
                    if vertical or horizontal:
                        choices.append(((small_h, small_w), (large_h, large_w)))
    return choices


def _place_rectangles(
    rng: np.random.Generator,
    shape: tuple[int, int],
    small_dims: tuple[int, int],
    large_dims: tuple[int, int],
    mode: str,
    first_is_small: bool,
    edge_role: str,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    grid_h, grid_w = shape
    sep_padding = 3 if mode == "inset" else 1
    orth_padding = 2 if mode == "inset" else 0
    possible = []
    if (small_dims[0] + large_dims[0] + sep_padding <= grid_h
            and max(small_dims[1], large_dims[1]) + orth_padding <= grid_w):
        possible.append("vertical")
    if (small_dims[1] + large_dims[1] + sep_padding <= grid_w
            and max(small_dims[0], large_dims[0]) + orth_padding <= grid_h):
        possible.append("horizontal")
    axis = possible[int(rng.integers(len(possible)))]
    first = "small" if first_is_small else "large"
    second = "large" if first_is_small else "small"
    dimensions = {"small": small_dims, "large": large_dims}
    sep_span = grid_h if axis == "vertical" else grid_w
    first_len = dimensions[first][0 if axis == "vertical" else 1]
    second_len = dimensions[second][0 if axis == "vertical" else 1]
    max_gap = min(4, sep_span - first_len - second_len - (2 if mode == "inset" else 0))
    gap = int(rng.integers(1, max_gap + 1))
    slack = sep_span - first_len - second_len - gap
    if mode == "inset":
        first_offset = int(rng.integers(1, slack))
    elif mode in {"edge", "corner"}:
        first_offset = 0 if edge_role == first else slack
    else:
        first_offset = int(rng.integers(0, slack + 1))
    second_offset = first_offset + first_len + gap
    positions = {}
    orth_span = grid_w if axis == "vertical" else grid_h
    limits = {}
    for role in (first, second):
        rect_h, rect_w = dimensions[role]
        orth_len = rect_w if axis == "vertical" else rect_h
        limits[role] = ((1, orth_span - orth_len - 1) if mode == "inset"
                        else (0, orth_span - orth_len))
    if axis == "horizontal":
        # Horizontal separation alone does not fix row-major reading order.
        # Keep the left rectangle at or above the right rectangle.
        if mode == "corner" and edge_role == first:
            first_orth = 0
            second_orth = int(rng.integers(max(first_orth, limits[second][0]),
                                            limits[second][1] + 1))
        elif mode == "corner" and edge_role == second:
            second_orth = limits[second][1]
            first_orth = int(rng.integers(limits[first][0],
                                           min(second_orth, limits[first][1]) + 1))
        else:
            first_orth = int(rng.integers(limits[first][0],
                                          min(limits[first][1], limits[second][1]) + 1))
            second_orth = int(rng.integers(max(first_orth, limits[second][0]),
                                           limits[second][1] + 1))
        orth_offsets = {first: first_orth, second: second_orth}
    else:
        orth_offsets = {}
        for role in (first, second):
            low, high = limits[role]
            if mode == "corner" and role == edge_role:
                orth_offsets[role] = 0 if bool(rng.integers(0, 2)) else high
            else:
                orth_offsets[role] = int(rng.integers(low, high + 1))
    for role, offset in ((first, first_offset), (second, second_offset)):
        rect_h, rect_w = dimensions[role]
        orth_offset = orth_offsets[role]
        row, col = ((offset, orth_offset) if axis == "vertical"
                    else (orth_offset, offset))
        positions[role] = (row, col, rect_h, rect_w)
    return positions["small"], positions["large"]


def _pair_colors(rng: np.random.Generator, parameters: dict, previous: tuple[int, int] | None) -> tuple[int, int]:
    available = parameters["available"]
    choices = [(a, b) for a in available for b in available
               if a != b and (previous is None or (a != previous[0] and b != previous[1]))]
    return choices[int(rng.integers(len(choices)))]


def _render_pair(
    shape: tuple[int, int],
    small_rect: tuple[int, int, int, int],
    large_rect: tuple[int, int, int, int],
    background: int,
    frame: int,
    parameters: dict,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, background, dtype=np.int8)
    out = inp.copy()
    for rect, accent in ((small_rect, parameters["small_accent"]),
                         (large_rect, parameters["large_accent"])):
        row, col, height, width = rect
        inp[row:row + height, col:col + width] = frame
        out[row:row + height, col:col + width] = frame
        out[row + 1:row + height - 1, col + 1:col + width - 1] = accent
    return inp, out


def _choose_mode(rng: np.random.Generator, split: str, classic: bool) -> str:
    if classic:
        return "inset" if split == "train" else "corner"
    if split == "train":
        return ("inset", "inset", "edge", "edge", "free", "free", "corner")[int(rng.integers(7))]
    return ("inset", "inset", "edge", "edge", "corner", "corner", "free")[int(rng.integers(7))]


def _make_pair(
    rng: np.random.Generator,
    split: str,
    parameters: dict,
    kind: str,
    first_is_small: bool,
    previous_colors: tuple[int, int] | None,
) -> tuple[dict, tuple[int, int]]:
    mode = _choose_mode(rng, split, parameters["classic"])
    edge_role = "small" if bool(rng.integers(0, 2)) else "large"
    fixed_shape = parameters["fixed_shape"]
    for _ in range(50):
        if fixed_shape is None:
            if split == "train":
                shape = (int(rng.integers(9, 23)), int(rng.integers(9, 23)))
            else:
                shape = (int(rng.integers(9, 27)), int(rng.integers(9, 27)))
        else:
            shape = fixed_shape
        maximum = 9 if split == "train" else 11
        choices = _dimension_choices(shape, kind, mode, maximum)
        if choices:
            break
    else:
        raise ValueError("no legal rectangle dimensions for canvas")
    if kind == "both" and rng.random() < 0.3:
        square_choices = [(small, large) for small, large in choices
                          if small[0] == small[1] and large[0] == large[1]]
        if square_choices:
            choices = square_choices
    small_dims, large_dims = choices[int(rng.integers(len(choices)))]
    small_rect, large_rect = _place_rectangles(
        rng, shape, small_dims, large_dims, mode, first_is_small, edge_role
    )
    background, frame = _pair_colors(rng, parameters, previous_colors)
    inp, out = _render_pair(shape, small_rect, large_rect, background, frame, parameters)
    return {"input": inp, "output": out}, (background, frame)


def _pair_rng(rng: np.random.Generator) -> np.random.Generator:
    return np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng, seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_colors = None
    height_first = bool(rng.integers(0, 2))
    first_small = bool(rng.integers(0, 2))
    training_plan = [
        ("height" if height_first else "width", first_small),
        ("width" if height_first else "height", not first_small),
    ]
    if train_count >= 4:
        training_plan.extend([
            (training_plan[0][0], not first_small),
            (training_plan[1][0], first_small),
        ])
    elif train_count == 3:
        training_plan.append(("both", bool(rng.integers(0, 2))))
    if train_count >= 5:
        training_plan.append(("both", bool(rng.integers(0, 2))))
    while len(training_plan) < train_count:
        training_plan.append((
            ("height", "width", "both")[int(rng.integers(3))],
            bool(rng.integers(0, 2)),
        ))
    for split in ("train", "test"):
        count = train_count if split == "train" else test_count
        for index in range(count):
            pair_rng = _pair_rng(rng)
            if split == "train":
                kind, order = training_plan[index]
            else:
                kinds = ("height", "width", "both") if any(k == "both" for k, _ in training_plan) else ("height", "width")
                kind = kinds[int(pair_rng.integers(len(kinds)))]
                order = bool(pair_rng.integers(0, 2))
            for _ in range(100):
                pair, colors = _make_pair(pair_rng, split, parameters, kind, order, previous_colors)
                key = _grid_key(pair["input"])
                try:
                    _identify_input(pair["input"])
                except AssertionError:
                    continue
                if key not in seen and key not in _SOURCE_INPUT_KEYS:
                    break
            else:
                raise ValueError("could not construct a unique, identifiable pair")
            puzzle[split].append(pair)
            seen.add(key)
            previous_colors = colors
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid: object, description: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{description} must be a NumPy array")
    _require(grid.ndim == 2, f"{description} must have two dimensions")
    _require(all(1 <= side <= 30 for side in grid.shape), f"{description} has illegal dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{description} must use integer cells")
    _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, f"{description} has illegal colors")


def _axis_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return -1


def _inspect_rectangles(grid: np.ndarray, background: int, frame: int) -> list[tuple[int, int, int, int, int]]:
    _require({int(v) for v in np.unique(grid)} == {background, frame}, "input must use two colors")
    height, width = grid.shape
    visited = np.zeros(grid.shape, dtype=bool)
    rectangles = []
    for start_row in range(height):
        for start_col in range(width):
            if int(grid[start_row, start_col]) != frame or visited[start_row, start_col]:
                continue
            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True
            cells = []
            while stack:
                row, col = stack.pop()
                cells.append((row, col))
                for rr, cc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                    if 0 <= rr < height and 0 <= cc < width and not visited[rr, cc] and int(grid[rr, cc]) == frame:
                        visited[rr, cc] = True
                        stack.append((rr, cc))
            rows, cols = zip(*cells)
            top, left = min(rows), min(cols)
            rect_h, rect_w = max(rows) - top + 1, max(cols) - left + 1
            area = rect_h * rect_w
            _require(rect_h >= 3 and rect_w >= 3 and len(cells) == area, "input has a malformed rectangle")
            rectangles.append((top, left, rect_h, rect_w, area))
    _require(len(rectangles) == 2, "input needs two separated rectangles")
    rectangles.sort(key=lambda rect: rect[4])
    small, large = rectangles
    _require(small[4] < large[4], "rectangle areas must differ")
    _require(small[2] <= large[2] and small[3] <= large[3], "size ranking must be componentwise clear")
    row_gap = _axis_gap(small[0], small[0] + small[2] - 1, large[0], large[0] + large[2] - 1)
    col_gap = _axis_gap(small[1], small[1] + small[3] - 1, large[1], large[1] + large[3] - 1)
    _require(row_gap >= 1 or col_gap >= 1, "rectangles must have a visible gap")
    return rectangles


def _identify_input(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int, int, int, int]]]:
    colors = [int(c) for c in np.unique(grid)]
    _require(len(colors) == 2, "input must use exactly two colors")
    candidates = []
    for frame in colors:
        background = next(c for c in colors if c != frame)
        try:
            rectangles = _inspect_rectangles(grid, background, frame)
        except AssertionError:
            continue
        candidates.append((background, frame, rectangles))
    _require(len(candidates) == 1, "input color roles are ambiguous")
    return candidates[0]


def _expected_output(inp: np.ndarray, rectangles: list[tuple[int, int, int, int, int]], accents: tuple[int, int]) -> np.ndarray:
    out = inp.copy()
    for rect, accent in zip(rectangles, accents):
        row, col, height, width, _ = rect
        out[row + 1:row + height - 1, col + 1:col + width - 1] = accent
    return out


def _edge_names(rect: tuple[int, int, int, int, int], shape: tuple[int, int]) -> set[str]:
    row, col, height, width, _ = rect
    result = set()
    if row == 0:
        result.add("top")
    if col == 0:
        result.add("left")
    if row + height == shape[0]:
        result.add("bottom")
    if col + width == shape[1]:
        result.add("right")
    return result


def _reading_order_role(rectangles: list[tuple[int, int, int, int, int]]) -> str:
    return "small" if (rectangles[0][0], rectangles[0][1]) < (rectangles[1][0], rectangles[1][1]) else "large"


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle needs train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    accents = None
    training_kinds, training_orders = set(), set()
    previous_colors = None
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair needs input and output")
            inp, out = pair["input"], pair["output"]
            _check_grid(inp, f"{split}[{index}].input")
            _check_grid(out, f"{split}[{index}].output")
            _require(inp.shape == out.shape, "grid dimensions changed")
            input_key, output_key = _grid_key(inp), _grid_key(out)
            _require(input_key not in seen_inputs and (input_key, output_key) not in seen_pairs, "duplicate generated pair")
            _require(input_key not in _SOURCE_INPUT_KEYS and (input_key, output_key) not in _SOURCE_PAIR_KEYS, "source pair copied")
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            background, frame, rectangles = _identify_input(inp)
            small, large = rectangles
            colors = (background, frame)
            if split == "train":
                training_kinds.add(_comparison_kind(rectangles))
                training_orders.add(_reading_order_role(rectangles))
                if index > 0:
                    _require(colors[0] != previous_colors[0] and colors[1] != previous_colors[1],
                             "adjacent training pairs must vary background and frame colors")
            previous_colors = colors
            if accents is None:
                accents = (int(out[small[0] + 1, small[1] + 1]),
                           int(out[large[0] + 1, large[1] + 1]))
            _require(len({background, frame, *accents}) == 4, "semantic color roles collide")
            _require(np.array_equal(out, _expected_output(inp, rectangles, accents)), "incorrect size-based interior colors")
            _require(not np.array_equal(inp, out), "pair has no transformation")
            _require({int(c) for c in np.unique(out)} == {background, frame, *accents}, "output roles missing")
            if split == "test" and _comparison_kind(rectangles) == "both":
                _require("both" in training_kinds, "two-dimension comparison was not demonstrated")
    _require({"height", "width"} <= training_kinds, "training must show both comparison axes")
    _require(training_orders == {"small", "large"}, "training must show both reading orders")


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[split])
                 for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "different seeds repeated puzzles")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "training counts lack diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack diversity")
    shapes, layouts, size_pairs, palettes = set(), set(), set(), set()
    arrangements, kinds, edge_roles = set(), set(), set()
    fixed_puzzles = 0
    changed_pair_palettes = 0
    all_four = 0
    compact_tests = portrait_tests = no_jump_tests = 0
    vertical_overlap_tests = horizontal_overlap_tests = staggered_tests = 0
    square_pairs = 0
    small_corner_tests = large_corner_tests = 0
    classic_puzzles = 0
    for puzzle in puzzles:
        all_shapes = set()
        all_palettes = set()
        combo = set()
        train_edges = []
        test_edges = []
        train_max_side = max(max(rect[2], rect[3])
                             for pair in puzzle["train"]
                             for rect in _identify_input(pair["input"])[2])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                bg, frame, rects = _identify_input(inp)
                all_shapes.add(inp.shape)
                all_palettes.add((bg, frame))
                shapes.add(inp.shape)
                palettes.add((bg, frame))
                size_pairs.add(tuple((r[2], r[3]) for r in rects))
                layouts.add((inp.shape, tuple(r[:4] for r in rects)))
                kind = _comparison_kind(rects)
                kinds.add(kind)
                if split == "train":
                    combo.add((kind, _reading_order_role(rects)))
                small_edges = _edge_names(rects[0], inp.shape)
                large_edges = _edge_names(rects[1], inp.shape)
                edge_roles.add((split, bool(small_edges), bool(large_edges)))
                (train_edges if split == "train" else test_edges).append(bool(small_edges or large_edges))
                square_pairs += rects[0][2] == rects[0][3] and rects[1][2] == rects[1][3]
                row_gap = _axis_gap(rects[0][0], rects[0][0] + rects[0][2] - 1,
                                    rects[1][0], rects[1][0] + rects[1][2] - 1)
                col_gap = _axis_gap(rects[0][1], rects[0][1] + rects[0][3] - 1,
                                    rects[1][1], rects[1][1] + rects[1][3] - 1)
                arrangements.add((split, row_gap >= 1, col_gap >= 1))
                if split == "test":
                    compact_tests += inp.shape == (10, 10)
                    portrait_tests += inp.shape[0] > inp.shape[1]
                    no_jump_tests += max(max(rect[2], rect[3]) for rect in rects) <= train_max_side
                    vertical_overlap_tests += row_gap >= 1 and col_gap < 0
                    horizontal_overlap_tests += col_gap >= 1 and row_gap < 0
                    staggered_tests += ((row_gap >= 1 and col_gap < 0 and rects[0][1] != rects[1][1])
                                        or (col_gap >= 1 and row_gap < 0 and rects[0][0] != rects[1][0]))
                    small_corner_tests += len(small_edges) >= 2
                    large_corner_tests += len(large_edges) >= 2
        fixed_puzzles += len(all_shapes) == 1
        changed_pair_palettes += len(all_palettes) > 1
        classic_puzzles += not any(train_edges) and all(test_edges)
        all_four += {("height", "small"), ("height", "large"),
                     ("width", "small"), ("width", "large")} <= combo
    _require(len(shapes) >= 35 and len(layouts) >= 120 and len(size_pairs) >= 35, "structural diversity is too low")
    _require(len(palettes) >= 35 and changed_pair_palettes >= 35, "pair palette variation is too low")
    _require(fixed_puzzles >= 5 and all_four >= 5, "fixed-canvas or axis/order coverage is too low")
    _require(kinds == {"height", "width", "both"}, "comparison kinds lack diversity")
    _require(compact_tests >= 2 and portrait_tests >= 5 and no_jump_tests >= 5,
             "compact, portrait, or modest-scale tests are too rare")
    _require(vertical_overlap_tests >= 5 and horizontal_overlap_tests >= 5
             and staggered_tests >= 5, "test layout types lack diversity")
    _require(square_pairs >= 3 and small_corner_tests >= 3 and large_corner_tests >= 3,
             "square comparisons or corner roles are too rare")
    _require(classic_puzzles >= 3, "source-style inset-train/edge-test puzzles are absent")
    _require(("test", True, False) in edge_roles and ("test", False, True) in edge_roles,
             "test edge role is fixed")
    _require(("test", False, False) in edge_roles and ("train", True, False) in edge_roles,
             "inset tests or edge training are absent")
    _require(any(s == "test" and row and not col for s, row, col in arrangements)
             and any(s == "test" and col and not row for s, row, col in arrangements),
             "test layouts lack vertical or horizontal separation")
    a = generate_puzzle(424242, 7, 3)
    b = generate_puzzle(424242, 7, 3)
    _require(_puzzle_key(a) == _puzzle_key(b), "explicit-count generation is nondeterministic")
    _require(_puzzle_key(generate_puzzle(424243)) == _puzzle_key(generate_puzzle(424243)),
             "automatic generation is nondeterministic")
    for i, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(500000 + i, train_count, test_count)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count),
                 "explicit boundary count ignored")
    for i, (train_count, test_count) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(600000 + i, train_count, test_count)
        _require((train_count is None or len(puzzle["train"]) == train_count)
                 and (test_count is None or len(puzzle["test"]) == test_count),
                 "partial count ignored")
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5),
                                    (2, 2), (2, 4), (3, 3), (3, 4), (4, 4),
                                    (2.0, 1), (True, 1), (3, False)):
        try:
            generate_puzzle(700000, train_count, test_count)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
