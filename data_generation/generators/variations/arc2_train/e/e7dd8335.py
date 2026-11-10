import numpy as np


_SOURCE_PAIRS = (
    (
        (
            "000000000",
            "000000000",
            "011110000",
            "010010000",
            "010010000",
            "010010000",
            "010010000",
            "011110000",
        ),
        (
            "000000000",
            "000000000",
            "011110000",
            "010010000",
            "010010000",
            "020020000",
            "020020000",
            "022220000",
        ),
    ),
    (
        (
            "0111110",
            "0101010",
            "0101010",
            "0101010",
            "0101010",
            "0111110",
            "0000000",
        ),
        (
            "0111110",
            "0101010",
            "0101010",
            "0202020",
            "0202020",
            "0222220",
            "0000000",
        ),
    ),
    (
        (
            "000100000",
            "011111000",
            "001010000",
            "001010000",
            "001010000",
            "001010000",
            "011111000",
            "000100000",
            "000000000",
        ),
        (
            "000100000",
            "011111000",
            "001010000",
            "001010000",
            "002020000",
            "002020000",
            "022222000",
            "000200000",
            "000000000",
        ),
    ),
    (
        (
            "00000000000",
            "00000000000",
            "00001111000",
            "00010000100",
            "11100000010",
            "11100000010",
            "00010000100",
            "00001111000",
            "00000000000",
        ),
        (
            "00000000000",
            "00000000000",
            "00001111000",
            "00010000100",
            "11100000010",
            "22200000020",
            "00020000200",
            "00002222000",
            "00000000000",
        ),
    ),
)


def _check(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _validate_optional_count(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _grid_key(grid: np.ndarray) -> tuple:
    normalized = grid.astype(np.int8, copy=False)
    return tuple(grid.shape), normalized.tobytes()


def _grid_rows(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _is_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    input_rows = _grid_rows(input_grid)
    output_rows = _grid_rows(output_grid)
    return any(
        input_rows == source_input and output_rows == source_output
        for source_input, source_output in _SOURCE_PAIRS
    )


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coordinates = np.argwhere(mask)
    _check(len(coordinates) > 0, "foreground must be nonempty")

    r0, c0 = coordinates.min(axis=0)
    r1, c1 = coordinates.max(axis=0)
    return int(r0), int(r1), int(c0), int(c1)


def _is_eight_connected(mask: np.ndarray) -> bool:
    coordinates = np.argwhere(mask)
    if len(coordinates) == 0:
        return False

    seen = np.zeros(mask.shape, dtype=bool)
    start_r, start_c = (int(x) for x in coordinates[0])
    stack = [(start_r, start_c)]
    seen[start_r, start_c] = True
    visited = 0

    while stack:
        r, c = stack.pop()
        visited += 1

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue

                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < mask.shape[0]
                    and 0 <= nc < mask.shape[1]
                    and mask[nr, nc]
                    and not seen[nr, nc]
                ):
                    seen[nr, nc] = True
                    stack.append((nr, nc))

    return visited == int(mask.sum())


def _make_mask(
    rng: np.random.Generator,
    width: int,
    half_height: int,
    style: str,
) -> np.ndarray:
    """Compose an upper half, then reflect its entire occupancy mask."""
    for _attempt in range(80):
        half = np.zeros((half_height, width), dtype=bool)
        if style in ("frame", "lattice", "dense", "filled"):
            half[0, :] = True
            half[:, 0] = True
            half[:, -1] = True
            if style == "filled":
                half[:, :] = True
            else:
                if style == "lattice":
                    rail_count = min(width - 2, int(rng.integers(1, 5)))
                else:
                    rail_count = min(width - 3, int(rng.integers(0, 4)))
                if rail_count:
                    rails = rng.choice(
                        np.arange(1, width - 1),
                        size=rail_count,
                        replace=False,
                    )
                    half[:, rails] = True
                if style == "lattice":
                    count = int(rng.integers(1, half_height))
                else:
                    count = int(rng.integers(0, half_height))
                if count:
                    rows = rng.choice(
                        np.arange(1, half_height),
                        size=count,
                        replace=False,
                    )
                    for r in rows:
                        a = int(rng.integers(0, width - 2))
                        b = int(rng.integers(a + 2, width + 1))
                        half[r, a:b] = True
                if style == "dense":
                    for r in range(1, half_height):
                        for c in range(1, width - 1):
                            if rng.random() < 0.42:
                                half[r, c] = True
        elif style == "contour":
            max_inset = min(half_height + 1, width - 2, 5)
            left = int(rng.integers(1, max_inset + 1))
            right_inset = int(rng.integers(1, max_inset + 1))
            right = max(left + 1, width - 1 - right_inset)
            cap = int(rng.integers(0, 5))
            if cap < 3:
                half[0, left : right + 1] = True
            elif cap == 3:
                half[0, left] = True
                half[0, right] = True
            else:
                half[0, left : min(right + 1, left + 2)] = True
            for r in range(1, half_height):
                left = max(0, left - int(rng.integers(0, 2)))
                right = min(width - 1, right + int(rng.integers(0, 2)))
                half[r, left] = True
                half[r, right] = True
                if rng.random() < 0.22:
                    half[r, left + 1 : min(right, left + 3)] = True
                if rng.random() < 0.22:
                    half[r, max(left + 1, right - 2) : right] = True
            half[-1, : left + 1] = True
            half[-1, right:] = True
            if cap == 3:
                bridge_row = int(rng.integers(1, half_height))
                half[bridge_row, left : right + 1] = True
            if cap < 3 and rng.random() < 0.35 and width >= 7:
                inner = int(rng.integers(2, width - 2))
                half[:, inner] = True
        elif style == "branch":
            center = int(rng.integers(1, width - 1))
            half[0, center] = True
            for r in range(1, half_height):
                center = int(np.clip(center + rng.integers(-1, 2), 1, width - 2))
                half[r, center] = True
                if rng.random() < 0.75:
                    reach = int(rng.integers(0, center + 1))
                    half[r, reach : center + 1] = True
                if rng.random() < 0.75:
                    reach = int(rng.integers(center, width))
                    half[r, center : reach + 1] = True
            half[-1, :] = True
        else:
            raise ValueError("unknown mask style")

        mask = np.concatenate((half, np.flipud(half)), axis=0)
        _, _, c0, c1 = _bbox(mask)
        if c0 == 0 and c1 == width - 1 and _is_eight_connected(mask):
            return mask

    # Every requested dimension has a deterministic connected fallback.
    half = np.zeros((half_height, width), dtype=bool)
    half[0, :] = True
    half[:, 0] = True
    half[:, -1] = True
    return np.concatenate((half, np.flipud(half)), axis=0)


def _render_pair(
    mask: np.ndarray,
    top: int,
    left: int,
    bottom_padding: int,
    right_padding: int,
    colors: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height = top + mask.shape[0] + bottom_padding
    width = left + mask.shape[1] + right_padding

    background, foreground, accent = colors

    occupied = np.zeros((height, width), dtype=bool)
    occupied[
        top : top + mask.shape[0],
        left : left + mask.shape[1],
    ] = mask

    input_grid = np.full(
        (height, width),
        background,
        dtype=np.int8,
    )
    input_grid[occupied] = foreground

    output_grid = input_grid.copy()
    lower_start = top + mask.shape[0] // 2
    row_numbers = np.arange(height)[:, None]
    lower_foreground = occupied & (row_numbers >= lower_start)
    output_grid[lower_foreground] = accent

    return input_grid, output_grid


def _padding(
    rng: np.random.Generator,
    figure_size: int,
    relation: str,
) -> tuple[int, int]:
    limit = min(10, 30 - figure_size)
    preferred = min(limit, int(rng.choice(
        (1, 2, 3, 4, 6, 8, 10),
        p=(0.08, 0.15, 0.20, 0.20, 0.15, 0.12, 0.10),
    )))
    candidates = [
        (a, b)
        for a in range(preferred + 1)
        for b in range(preferred + 1)
        if a + b <= 30 - figure_size
        and (
            relation == "any"
            or (relation == "before" and a < b)
            or (relation == "after" and a > b)
            or (relation == "first_edge" and a == 0)
            or (relation == "last_edge" and b == 0)
            or (relation == "interior" and a > 0 and b > 0)
        )
    ]
    return candidates[int(rng.integers(0, len(candidates)))]


def _analyze_pair(pair: dict) -> dict:
    input_grid = pair["input"]
    output_grid = pair["output"]

    _check(
        input_grid.shape == output_grid.shape,
        "input/output dimensions must be preserved",
    )

    input_colors = {int(x) for x in np.unique(input_grid)}
    output_colors = {int(x) for x in np.unique(output_grid)}

    _check(
        len(input_colors) == 2,
        "input must contain background and foreground",
    )

    new_colors = output_colors - input_colors
    _check(
        len(output_colors) == 3 and len(new_colors) == 1,
        "output must add exactly one accent color",
    )
    accent = next(iter(new_colors))

    changed = input_grid != output_grid
    _check(changed.any(), "the lower half must change")

    changed_input_colors = {
        int(x) for x in np.unique(input_grid[changed])
    }
    _check(
        len(changed_input_colors) == 1,
        "all changed cells must have the foreground role",
    )
    foreground = next(iter(changed_input_colors))

    background_colors = input_colors - {foreground}
    _check(
        len(background_colors) == 1,
        "background role must be unique",
    )
    background = next(iter(background_colors))

    _check(
        np.all(output_grid[changed] == accent),
        "changed cells must all receive the accent",
    )

    mask = input_grid == foreground
    _check(
        np.all(input_grid[~mask] == background),
        "background must be uniform",
    )
    _check(int(mask.sum()) * 2 < input_grid.size,
           "background must be the unique majority color")

    r0, r1, c0, c1 = _bbox(mask)
    box_height = r1 - r0 + 1
    box_width = c1 - c0 + 1

    _check(
        box_height % 2 == 0,
        "foreground bounding-box height must be even",
    )

    boxed_mask = mask[r0 : r1 + 1, c0 : c1 + 1]
    _check(
        np.array_equal(boxed_mask, np.flipud(boxed_mask)),
        "foreground must have horizontal reflection symmetry",
    )
    _check(
        _is_eight_connected(mask),
        "there must be one 8-connected figure",
    )

    lower_start = r0 + box_height // 2
    row_numbers = np.arange(input_grid.shape[0])[:, None]
    lower_foreground = mask & (row_numbers >= lower_start)
    upper_foreground = mask & ~lower_foreground

    _check(
        lower_foreground.any() and upper_foreground.any(),
        "both reflected halves must be nonempty",
    )

    expected = input_grid.copy()
    expected[lower_foreground] = accent

    _check(
        np.array_equal(output_grid, expected),
        "output must recolor exactly the lower foreground half",
    )
    _check(
        np.array_equal(output_grid != background, mask),
        "foreground occupancy must be preserved",
    )

    return {
        "background": background,
        "foreground": foreground,
        "accent": accent,
        "mask": mask,
        "boxed_mask": boxed_mask,
        "r0": r0,
        "r1": r1,
        "c0": c0,
        "c1": c1,
        "height": box_height,
        "width": box_width,
    }


def _count_components(mask: np.ndarray, diagonals: bool) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    neighbors = (
        ((-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1))
        if diagonals else
        ((-1, 0), (1, 0), (0, -1), (0, 1))
    )
    for row, col in np.argwhere(mask):
        r, c = int(row), int(col)
        if seen[r, c]:
            continue
        count += 1
        seen[r, c] = True
        stack = [(r, c)]
        while stack:
            rr, cc = stack.pop()
            for dr, dc in neighbors:
                nr, nc = rr + dr, cc + dc
                if (0 <= nr < mask.shape[0]
                    and 0 <= nc < mask.shape[1]
                    and mask[nr, nc] and not seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def _enclosed_regions(mask: np.ndarray) -> int:
    background = ~mask
    seen = np.zeros(mask.shape, dtype=bool)
    enclosed = 0
    height, width = mask.shape
    for row, col in np.argwhere(background):
        r, c = int(row), int(col)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        touches_edge = False
        while stack:
            rr, cc = stack.pop()
            if rr in (0, height - 1) or cc in (0, width - 1):
                touches_edge = True
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = rr + dr, cc + dc
                if (0 <= nr < height and 0 <= nc < width
                    and background[nr, nc] and not seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        if not touches_edge:
            enclosed += 1
    return enclosed


def _validate_regime(analysis: dict, regime: str) -> None:
    _check(regime in ("source_train", "target_test"),
           "unknown validation regime")
    _check(4 <= analysis["height"] <= 14
           and analysis["height"] % 2 == 0,
           "figure height is outside the sampled family")
    _check(4 <= analysis["width"] <= 18,
           "figure width is outside the sampled family")


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "puzzle must have exactly train and test")
    _check(isinstance(puzzle["train"], list)
           and isinstance(puzzle["test"], list),
           "train and test must be lists")
    num_train, num_test = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= num_train <= 10 and 1 <= num_test <= 4
           and num_train > num_test, "illegal pair counts")
    inputs, pairs, shapes = set(), set(), set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        regime = "source_train" if section == "train" else "target_test"
        for pair in puzzle[section]:
            _check(isinstance(pair, dict)
                   and set(pair) == {"input", "output"},
                   "each pair needs input and output")
            for grid in (pair["input"], pair["output"]):
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2,
                       "grid must be a 2D ndarray")
                _check(np.issubdtype(grid.dtype, np.integer)
                       and 1 <= grid.shape[0] <= 30
                       and 1 <= grid.shape[1] <= 30
                       and np.all((0 <= grid) & (grid <= 9)),
                       "illegal grid size, dtype, or color")
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            _check(input_key not in inputs and pair_key not in pairs,
                   "duplicate input or pair")
            inputs.add(input_key)
            pairs.add(pair_key)
            _check(not _is_source_pair(pair["input"], pair["output"]),
                   "source pair copied exactly")
            item = _analyze_pair(pair)
            _validate_regime(item, regime)
            shape_key = (item["boxed_mask"].shape,
                         item["boxed_mask"].tobytes())
            _check(shape_key not in shapes,
                   "figures within one puzzle must differ")
            shapes.add(shape_key)
            analyses[section].append(item)

    all_items = analyses["train"] + analyses["test"]
    accents = {item["accent"] for item in all_items}
    _check(len(accents) == 1, "accent must be learned once per puzzle")
    _check(all(item["foreground"] != item["background"]
               and item["accent"] not in
               (item["foreground"], item["background"])
               for item in all_items), "ambiguous color roles")
    _check(any(_enclosed_regions(item["boxed_mask"]) >= 1
               for item in analyses["train"]),
           "training needs a hollow structural demonstration")
    offsets = [
        item["r0"] + item["r1"] - (pair["input"].shape[0] - 1)
        for item, pair in zip(analyses["train"], puzzle["train"])
    ]
    _check(min(offsets) < 0 < max(offsets),
           "training must put the figure axis on both sides of canvas center")


def _choose_colors(
    rng: np.random.Generator,
    accent: int,
    stable: tuple[int, int] | None,
    previous: tuple[int, int] | None,
) -> tuple[int, int, int]:
    if stable is not None:
        return stable[0], stable[1], accent
    choices = [color for color in range(10) if color != accent]
    for _ in range(20):
        selected = rng.choice(choices, size=2, replace=False)
        roles = int(selected[0]), int(selected[1])
        if roles != previous:
            return roles[0], roles[1], accent
    raise RuntimeError("color selection failed")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train = _validate_optional_count(num_train, "num_train", 2, 10)
    num_test = _validate_optional_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        lower = num_test + 1
        num_train = lower + ((seed // 4 + 3 * num_test) % (11 - lower))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    accent = int(rng.integers(0, 10))
    stable_colors = None
    if rng.random() < 0.42:
        available = [color for color in range(10) if color != accent]
        chosen = rng.choice(available, size=2, replace=False)
        stable_colors = (int(chosen[0]), int(chosen[1]))
    wide_shift = rng.random() < 0.48
    puzzle = {"train": [], "test": []}
    used_inputs, used_shapes = set(), set()

    # The positions and rich demonstrations are shuffled across training pairs.
    vertical_relations = ["before", "after"] + ["any"] * (num_train - 2)
    rng.shuffle(vertical_relations)
    training_styles = [
        str(rng.choice(("frame", "lattice"))),
        str(rng.choice(("contour", "contour", "branch", "branch", "dense"))),
    ] + [
        str(rng.choice(("frame", "lattice", "contour", "contour",
                        "branch", "branch", "dense", "dense", "filled")))
        for _ in range(num_train - 2)
    ]
    rng.shuffle(training_styles)
    rich_index = next(i for i, style in enumerate(training_styles)
                      if style in ("frame", "lattice"))
    previous_colors = None

    for section, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _attempt in range(400):
                pair_rng = np.random.default_rng(
                    int(rng.integers(0, 2**63, dtype=np.uint64))
                )
                if section == "train":
                    width = int(pair_rng.integers(
                        5 if training_styles[index] in ("frame", "lattice") else 4,
                        8 if wide_shift else 19))
                    half_height = int(pair_rng.integers(2, 6 if wide_shift else 8))
                    style = training_styles[index]
                    vertical = vertical_relations[index]
                else:
                    width = int(pair_rng.integers(
                        9 if wide_shift else 4, 19))
                    half_height = int(pair_rng.integers(
                        3 if wide_shift else 2, 7 if wide_shift else 8))
                    style = str(pair_rng.choice(
                        ("contour", "contour", "contour", "frame", "lattice",
                         "branch", "branch", "dense", "dense", "filled")))
                    vertical = str(pair_rng.choice(
                        ("any", "interior", "first_edge", "last_edge")))
                mask = _make_mask(pair_rng, width, half_height, style)
                top, bottom = _padding(pair_rng, mask.shape[0], vertical)
                horizontal = str(pair_rng.choice(
                    ("any", "interior", "interior",
                     "first_edge", "last_edge")))
                left, right = _padding(pair_rng, mask.shape[1], horizontal)
                colors = _choose_colors(
                    pair_rng, accent, stable_colors, previous_colors
                )
                pair = {}
                pair["input"], pair["output"] = _render_pair(
                    mask, top, left, bottom, right, colors
                )
                key = _grid_key(pair["input"])
                shape_key = (mask.shape, mask.tobytes())
                if (key in used_inputs or shape_key in used_shapes
                    or _is_source_pair(pair["input"], pair["output"])):
                    continue
                try:
                    item = _analyze_pair(pair)
                except AssertionError:
                    continue
                if section == "train" and index == rich_index and (
                    _enclosed_regions(item["boxed_mask"]) < 1
                ):
                    continue
                puzzle[section].append(pair)
                used_inputs.add(key)
                used_shapes.add(shape_key)
                previous_colors = colors[:2]
                break
            else:
                raise RuntimeError("failed to construct a unique valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(
        len(first[section]) == len(second[section])
        and all(
            np.array_equal(a[name], b[name])
            for a, b in zip(first[section], second[section])
            for name in ("input", "output")
        )
        for section in ("train", "test")
    )


def self_test() -> None:
    automatic = generate_puzzle(20260827)
    validate_generated_puzzle(automatic)
    explicit_a = generate_puzzle(314159, 6, 3)
    explicit_b = generate_puzzle(314159, 6, 3)
    _check(_puzzles_equal(explicit_a, explicit_b),
           "fixed seed and counts must be deterministic")
    _check(not _puzzles_equal(explicit_a, generate_puzzle(314160, 6, 3)),
           "different seeds should differ")
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(5000 + train_count * 10 + test_count,
                                 train_count, test_count)
        _check(len(puzzle["train"]) == train_count
               and len(puzzle["test"]) == test_count,
               "explicit count ignored")
        validate_generated_puzzle(puzzle)
    for index, (train_count, test_count) in enumerate(
        ((None, 1), (None, 4), (2, None), (10, None))
    ):
        puzzle = generate_puzzle(6101 + index, train_count, test_count)
        _check((train_count is None or len(puzzle["train"]) == train_count)
               and (test_count is None or len(puzzle["test"]) == test_count),
               "partial count ignored")
        validate_generated_puzzle(puzzle)
    for kwargs in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4}
    ):
        try:
            generate_puzzle(7000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")

    train_counts, test_counts = set(), set()
    shapes, widths, heights, color_modes = set(), set(), set(), set()
    holes, four_parts, density = set(), set(), set()
    train_test_width = set()
    edge_contacts = set()
    axis_patterns = set()
    pair_color_patterns = set()
    for seed_value in range(1000, 1042):
        puzzle = generate_puzzle(seed_value)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info = [_analyze_pair(pair) for pair in puzzle["train"]]
        test_info = [_analyze_pair(pair) for pair in puzzle["test"]]
        train_test_width.add(
            "wider" if max(i["width"] for i in test_info)
            > max(i["width"] for i in train_info) else "equal_or_smaller"
        )
        roles = {(i["background"], i["foreground"]) for i in train_info}
        color_modes.add("changing" if len(roles) > 1 else "stable")
        pair_color_patterns.add(tuple(sorted(roles)))
        offsets = tuple(
            "above" if i["r0"] + i["r1"] < pair["input"].shape[0] - 1
            else "below" if i["r0"] + i["r1"] > pair["input"].shape[0] - 1
            else "center"
            for i, pair in zip(train_info, puzzle["train"])
        )
        axis_patterns.add(offsets)
        for section in ("train", "test"):
            for pair in puzzle[section]:
                item = _analyze_pair(pair)
                boxed = item["boxed_mask"]
                shapes.add((boxed.shape, boxed.tobytes()))
                widths.add(item["width"])
                heights.add(item["height"])
                holes.add(_enclosed_regions(boxed))
                four_parts.add(_count_components(boxed, False))
                density.add(round(float(boxed.mean()), 1))
                h, w = pair["input"].shape
                contact = (
                    item["r0"] == 0, item["r1"] == h - 1,
                    item["c0"] == 0, item["c1"] == w - 1
                )
                if section == "test":
                    edge_contacts.add(contact)
                mask = pair["input"] == item["foreground"]
                expected = pair["input"].copy()
                expected[mask & (np.arange(h)[:, None]
                                 >= item["r0"] + item["height"] // 2)] = item["accent"]
                _check(np.array_equal(expected, pair["output"]),
                       "independent lower-half rule check failed")
    _check(len(train_counts) >= 4 and len(test_counts) >= 3,
           "automatic pair count diversity failed")
    _check(len(shapes) >= 100 and len(widths) >= 10
           and len(heights) >= 5,
           "structural dimensions lack diversity")
    _check(color_modes == {"stable", "changing"}
           and len(pair_color_patterns) >= 12,
           "color regimes lack diversity")
    _check(max(holes) >= 2 and 0 in holes
           and max(four_parts) >= 5 and len(density) >= 6,
           "figure vocabulary lacks holes, diagonal joints, or density")
    _check(train_test_width == {"wider", "equal_or_smaller"},
           "train/test scale relation lacks diversity")
    _check(len(edge_contacts) >= 8
           and any(x[2] for x in edge_contacts)
           and any(x[3] for x in edge_contacts)
           and any(x[0] for x in edge_contacts)
           and any(x[1] for x in edge_contacts)
           and any(not any(x) for x in edge_contacts),
           "test boundary placement lacks diversity")
    _check(len(axis_patterns) >= 8,
           "training axis ordering lacks diversity")
