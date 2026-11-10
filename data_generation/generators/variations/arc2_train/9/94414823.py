import numpy as np


_SIDE_CORNERS = {
    "top": ("NW", "NE"),
    "bottom": ("SW", "SE"),
    "left": ("NW", "SW"),
    "right": ("NE", "SE"),
}

_OPPOSITE_CORNER = {
    "NW": "SE",
    "NE": "SW",
    "SW": "NE",
    "SE": "NW",
}


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _pair_signature(pair: dict) -> tuple:
    return (
        _grid_signature(pair["input"]),
        _grid_signature(pair["output"]),
    )


def _text_grid_signature(text: str) -> tuple:
    lines = [
        line.strip()
        for line in text.strip().splitlines()
        if line.strip()
    ]
    rows = tuple(tuple(int(ch) for ch in line) for line in lines)
    assert rows
    assert all(len(row) == len(rows[0]) for row in rows)
    return (
        (len(rows), len(rows[0])),
        tuple(value for row in rows for value in row),
    )


_SOURCE_PAIR_TEXTS = (
    (
        """
        0000000000
        0000000060
        0055555500
        0050000500
        0050000500
        0050000500
        0050000500
        0055555500
        0000000070
        0000000000
        """,
        """
        0000000000
        0000000060
        0055555500
        0057766500
        0057766500
        0056677500
        0056677500
        0055555500
        0000000070
        0000000000
        """,
    ),
    (
        """
        0000000000
        0400000080
        0055555500
        0050000500
        0050000500
        0050000500
        0050000500
        0055555500
        0000000000
        0000000000
        """,
        """
        0000000000
        0400000080
        0055555500
        0054488500
        0054488500
        0058844500
        0058844500
        0055555500
        0000000000
        0000000000
        """,
    ),
    (
        """
        0000000000
        0300000000
        0055555500
        0050000500
        0050000500
        0050000500
        0050000500
        0055555500
        0200000000
        0000000000
        """,
        """
        0000000000
        0300000000
        0055555500
        0053322500
        0053322500
        0052233500
        0052233500
        0055555500
        0200000000
        0000000000
        """,
    ),
    (
        """
        0000000000
        0000000000
        0055555500
        0050000500
        0050000500
        0050000500
        0050000500
        0055555500
        0900000010
        0000000000
        """,
        """
        0000000000
        0000000000
        0055555500
        0051199500
        0051199500
        0059911500
        0059911500
        0055555500
        0900000010
        0000000000
        """,
    ),
)

_SOURCE_PAIR_SIGNATURES = frozenset(
    (
        _text_grid_signature(input_text),
        _text_grid_signature(output_text),
    )
    for input_text, output_text in _SOURCE_PAIR_TEXTS
)


def _normalize_requested_count(
    value: int | None,
    name: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    normalized = int(value)
    if not minimum <= normalized <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return normalized


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _normalize_requested_count(
        num_train, "num_train", 2, 10
    )
    test_count = _normalize_requested_count(
        num_test, "num_test", 1, 4
    )

    if train_count is None and test_count is None:
        seed_value = int(seed)
        test_count = 1 + seed_value % 4
        span = 10 - test_count
        train_count = (
            test_count
            + 1
            + (seed_value // 4) % span
        )
    elif train_count is None:
        train_count = int(rng.integers(test_count + 1, 11))
    elif test_count is None:
        largest_test = min(4, train_count - 1)
        if largest_test < 1:
            raise ValueError("no legal test count for num_train")
        test_count = int(rng.integers(1, largest_test + 1))

    if not (2 <= train_count <= 10):
        raise ValueError("resolved num_train is illegal")
    if not (1 <= test_count <= 4):
        raise ValueError("resolved num_test is illegal")
    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    colors = [int(value) for value in rng.permutation(10)]
    return {
        "base_roles": (colors[0], colors[1]),
        "stable_roles": bool(rng.integers(0, 2)),
        "side_mode": int(rng.integers(0, 4)),
    }


def _select_marker_colors(
    rng: np.random.Generator,
    available: list[int],
    anchor: tuple[int, int] | None,
) -> tuple[int, int]:
    reusable = [] if anchor is None else [
        color for color in anchor if color in available
    ]
    overlap = int(rng.choice(3, p=[0.28, 0.32, 0.40]))
    overlap = min(overlap, len(reusable))
    fresh = [color for color in available if color not in reusable]
    overlap = max(overlap, 2 - len(fresh))
    chosen = []
    if overlap:
        chosen.extend(int(value) for value in rng.choice(
            reusable, size=overlap, replace=False
        ))
    chosen.extend(int(value) for value in rng.choice(
        fresh, size=2 - overlap, replace=False
    ))
    rng.shuffle(chosen)
    return chosen[0], chosen[1]


def _corner_positions(
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> dict[str, tuple[int, int]]:
    return {
        "NW": (top - 1, left - 1),
        "NE": (top - 1, right + 1),
        "SW": (bottom + 1, left - 1),
        "SE": (bottom + 1, right + 1),
    }


def _quadrant_slices(info: dict) -> dict[str, tuple[slice, slice]]:
    top = int(info["top"])
    bottom = int(info["bottom"])
    left = int(info["left"])
    right = int(info["right"])

    row_start = top + 1
    col_start = left + 1
    row_mid = row_start + int(info["interior_height"]) // 2
    col_mid = col_start + int(info["interior_width"]) // 2

    return {
        "NW": (
            slice(row_start, row_mid),
            slice(col_start, col_mid),
        ),
        "NE": (
            slice(row_start, row_mid),
            slice(col_mid, right),
        ),
        "SW": (
            slice(row_mid, bottom),
            slice(col_start, col_mid),
        ),
        "SE": (
            slice(row_mid, bottom),
            slice(col_mid, right),
        ),
    }


def _expected_output(
    input_grid: np.ndarray,
    info: dict,
) -> np.ndarray:
    output = input_grid.copy()
    quadrants = _quadrant_slices(info)

    for corner, color in info["markers"].items():
        output[quadrants[corner]] = int(color)
        output[quadrants[_OPPOSITE_CORNER[corner]]] = int(color)

    return output


def _sample_sides(
    rng: np.random.Generator,
    train_count: int,
    test_count: int,
    mode: int,
) -> tuple[list[str], list[str]]:
    sides = ["top", "bottom", "left", "right"]
    if mode == 0:
        withheld = "bottom"
        train_pool = [side for side in sides if side != withheld]
    elif mode == 1:
        withheld = str(rng.choice(["top", "left", "right"]))
        train_pool = [side for side in sides if side != withheld]
    else:
        withheld = None
        train_pool = sides
    first = [str(side) for side in rng.choice(train_pool, 2, replace=False)]
    train_sides = first + [
        str(rng.choice(train_pool)) for _ in range(train_count - 2)
    ]
    if withheld is not None:
        test_sides = [withheld] * test_count
    elif mode == 2:
        test_sides = [str(rng.choice(train_sides)) for _ in range(test_count)]
    else:
        test_sides = [str(rng.choice(sides)) for _ in range(test_count)]
        if test_count >= 2:
            test_sides[1] = str(rng.choice([
                side for side in sides if side != test_sides[0]
            ]))
    return train_sides, test_sides


def _sample_roles(
    rng: np.random.Generator,
    parameters: dict,
    pair_index: int,
) -> tuple[int, int]:
    base = parameters["base_roles"]
    if parameters["stable_roles"] or pair_index == 0:
        return base
    if pair_index == 1 or rng.random() < 0.25:
        if rng.random() < 0.5:
            return base[1], base[0]
        candidates = [
            (a, b) for a in range(10) for b in range(10)
            if a != b and (a, b) != base
        ]
        return candidates[int(rng.integers(len(candidates)))]
    if rng.random() < 0.45:
        return base
    values = rng.choice(10, size=2, replace=False)
    return int(values[0]), int(values[1])


def _sample_cavity(
    rng: np.random.Generator,
    pair_index: int,
) -> tuple[int, int]:
    if pair_index < 2:
        return (
            2 * int(rng.integers(2, 9)),
            2 * int(rng.integers(2, 9)),
        )
    mode = rng.random()
    if mode < 0.14:
        choices = ((2, 2), (2, 4), (4, 2))
        return choices[int(rng.integers(len(choices)))]
    if mode < 0.30:
        large = 2 * int(rng.integers(10, 14))
        other = 2 * int(rng.integers(1, 14))
        return (large, other) if rng.random() < 0.5 else (other, large)
    return (
        2 * int(rng.integers(1, 14)),
        2 * int(rng.integers(1, 14)),
    )


def _sample_margins(
    rng: np.random.Generator,
    cavity: int,
    minimum_before: int,
    minimum_after: int,
    edge: str | None,
    intro: bool,
) -> tuple[int, int]:
    spare = 30 - cavity - 2 - minimum_before - minimum_after
    if spare < 0:
        raise AssertionError("cavity cannot fit markers")
    choice = rng.random()
    if intro:
        extra = int(rng.integers(0, min(spare, 6) + 1))
    elif choice < 0.13:
        extra = spare
    elif choice < 0.30:
        extra = 0
    else:
        extra = int(rng.integers(0, min(spare, 9) + 1))
    if edge == "after":
        return minimum_before + extra, minimum_after
    if edge == "before":
        return minimum_before, minimum_after + extra
    before_extra = int(rng.integers(0, extra + 1))
    return minimum_before + before_extra, minimum_after + extra - before_extra


def _construct_pair(
    rng: np.random.Generator,
    side: str,
    background: int,
    frame: int,
    markers: tuple[int, int],
    pair_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    interior_height, interior_width = _sample_cavity(rng, pair_index)
    intro = pair_index < 2
    edge = not intro and rng.random() < 0.24
    min_top = 1 if intro or side != "bottom" else 0
    min_bottom = 1 if intro or side != "top" else 0
    min_left = 1 if intro or side != "right" else 0
    min_right = 1 if intro or side != "left" else 0

    vertical_edge = ("after" if side == "top" else "before") if edge and side in ("top", "bottom") else None
    horizontal_edge = ("after" if side == "left" else "before") if edge and side in ("left", "right") else None
    top_margin, bottom_margin = _sample_margins(
        rng, interior_height, min_top, min_bottom, vertical_edge, intro
    )
    left_margin, right_margin = _sample_margins(
        rng, interior_width, min_left, min_right, horizontal_edge, intro
    )

    height = top_margin + interior_height + 2 + bottom_margin
    width = left_margin + interior_width + 2 + right_margin
    grid = np.full((height, width), background, dtype=np.int8)
    top, left = top_margin, left_margin
    bottom = top + interior_height + 1
    right = left + interior_width + 1
    grid[top, left:right + 1] = frame
    grid[bottom, left:right + 1] = frame
    grid[top:bottom + 1, left] = frame
    grid[top:bottom + 1, right] = frame
    corner_a, corner_b = _SIDE_CORNERS[side]
    positions = _corner_positions(top, bottom, left, right)
    grid[positions[corner_a]] = markers[0]
    grid[positions[corner_b]] = markers[1]
    info = {
        "background": background, "frame": frame,
        "top": top, "bottom": bottom, "left": left, "right": right,
        "interior_height": interior_height,
        "interior_width": interior_width,
        "side": side,
        "markers": {corner_a: markers[0], corner_b: markers[1]},
    }
    return grid, _expected_output(grid, info)


def _analyze_input(grid: np.ndarray) -> dict:
    height, width = grid.shape
    candidates = []
    for value in np.unique(grid):
        frame = int(value)
        coords = np.argwhere(grid == frame)
        if len(coords) < 12:
            continue
        top, left = (int(x) for x in coords.min(axis=0))
        bottom, right = (int(x) for x in coords.max(axis=0))
        ih, iw = bottom - top - 1, right - left - 1
        if ih < 2 or iw < 2 or ih % 2 or iw % 2:
            continue
        perimeter = 2 * (ih + 2) + 2 * (iw + 2) - 4
        if len(coords) != perimeter:
            continue
        if not (
            np.all(grid[top, left:right + 1] == frame)
            and np.all(grid[bottom, left:right + 1] == frame)
            and np.all(grid[top:bottom + 1, left] == frame)
            and np.all(grid[top:bottom + 1, right] == frame)
        ):
            continue
        interior = grid[top + 1:bottom, left + 1:right]
        background = int(interior[0, 0])
        if background == frame or not np.all(interior == background):
            continue
        marker_coords = np.argwhere((grid != background) & (grid != frame))
        if len(marker_coords) != 2:
            continue
        actual_positions = {tuple(int(x) for x in row) for row in marker_coords}
        positions = _corner_positions(top, bottom, left, right)
        for side, corners in _SIDE_CORNERS.items():
            expected_positions = {positions[corner] for corner in corners}
            if actual_positions != expected_positions:
                continue
            colors = [int(grid[positions[corner]]) for corner in corners]
            if colors[0] == colors[1]:
                continue
            template = np.full(grid.shape, background, dtype=np.int8)
            template[top, left:right + 1] = frame
            template[bottom, left:right + 1] = frame
            template[top:bottom + 1, left] = frame
            template[top:bottom + 1, right] = frame
            for corner, color in zip(corners, colors):
                template[positions[corner]] = color
            if not np.array_equal(grid, template):
                continue
            candidates.append({
                "background": background, "frame": frame,
                "top": top, "bottom": bottom, "left": left, "right": right,
                "interior_height": ih, "interior_width": iw,
                "side": side,
                "markers": dict(zip(corners, colors)),
            })
    if len(candidates) != 1:
        raise AssertionError("input must have one complete frame and two corner markers")
    return candidates[0]


def _draw_unique_pair(
    rng: np.random.Generator,
    side: str,
    parameters: dict,
    pair_index: int,
    history: list[tuple[str, tuple[int, int]]],
    seen_inputs: set,
) -> dict:
    for _ in range(80):
        background, frame = _sample_roles(rng, parameters, pair_index)
        available = [c for c in range(10) if c not in (background, frame)]
        anchor = history[0][1] if history else None
        earlier_same_side = [
            colors for earlier_side, colors in history
            if earlier_side == side and all(c in available for c in colors)
        ]
        if earlier_same_side and rng.random() < 0.42:
            prior = earlier_same_side[int(rng.integers(len(earlier_same_side)))]
            markers = prior[1], prior[0]
        else:
            markers = _select_marker_colors(rng, available, anchor)
        child_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        input_grid, output_grid = _construct_pair(
            child_rng, side, background, frame, markers, pair_index
        )
        pair = {"input": input_grid, "output": output_grid}
        signature = _grid_signature(input_grid)
        if signature in seen_inputs or _pair_signature(pair) in _SOURCE_PAIR_SIGNATURES:
            continue
        seen_inputs.add(signature)
        history.append((side, markers))
        return pair
    raise RuntimeError("could not draw a unique pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    train_sides, test_sides = _sample_sides(
        rng, train_count, test_count, parameters["side_mode"]
    )
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    history = []
    for pair_index, side in enumerate(train_sides + test_sides):
        split = "train" if pair_index < train_count else "test"
        puzzle[split].append(_draw_unique_pair(
            rng, side, parameters, pair_index, history, seen_inputs
        ))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise AssertionError("puzzle must contain exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in ("train", "test")):
        raise AssertionError("splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    if not (2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne):
        raise AssertionError("illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise AssertionError("pair must contain input and output")
            for name in ("input", "output"):
                grid = pair[name]
                if not isinstance(grid, np.ndarray) or grid.ndim != 2:
                    raise AssertionError("grid must be a 2D NumPy array")
                if not np.issubdtype(grid.dtype, np.integer):
                    raise AssertionError("grid must have integer dtype")
                if not all(1 <= d <= 30 for d in grid.shape):
                    raise AssertionError("grid dimensions out of range")
                if np.any(grid < 0) or np.any(grid > 9):
                    raise AssertionError("grid colors out of range")
            inp, out = pair["input"], pair["output"]
            if inp.shape != out.shape or np.array_equal(inp, out):
                raise AssertionError("dimensions or changed cells are wrong")
            signature = _grid_signature(inp)
            pair_signature = _pair_signature(pair)
            if signature in seen_inputs or pair_signature in seen_pairs:
                raise AssertionError("duplicate pair or input")
            if pair_signature in _SOURCE_PAIR_SIGNATURES:
                raise AssertionError("source pair copied")
            seen_inputs.add(signature)
            seen_pairs.add(pair_signature)
            info = _analyze_input(inp)
            if len(np.unique(inp)) != 4 or len(np.unique(out)) != 4:
                raise AssertionError("expected four color roles")
            if not np.array_equal(out, _expected_output(inp, info)):
                raise AssertionError("incorrect diagonal quadrant fill")
    first_two = [_analyze_input(pair["input"])["side"] for pair in puzzle["train"][:2]]
    if first_two[0] == first_two[1]:
        raise AssertionError("opening training pairs must vary marked side")
    for pair in puzzle["train"][:2]:
        info = _analyze_input(pair["input"])
        if min(info["interior_height"], info["interior_width"]) < 4:
            raise AssertionError("opening demonstration is too small")


def _independent_prediction(input_grid: np.ndarray) -> np.ndarray:
    # Reconstruct the visible rule with cellwise parity, independent of
    # the quadrant-slice constructor used to create ground truth.
    info = _analyze_input(input_grid)
    result = input_grid.copy()
    for row in range(info["top"] + 1, info["bottom"]):
        for col in range(info["left"] + 1, info["right"]):
            vertical = row - info["top"] - 1 >= info["interior_height"] // 2
            horizontal = col - info["left"] - 1 >= info["interior_width"] // 2
            corner = ("S" if vertical else "N") + ("E" if horizontal else "W")
            label = corner if corner in info["markers"] else _OPPOSITE_CORNER[corner]
            result[row, col] = info["markers"][label]
    return result


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    shapes, cavities, layouts, palettes = set(), set(), set(), set()
    sides_train, sides_test = set(), set()
    withheld_bottom = withheld_other = bottom_trained = shared_side = mixed_tests = 0
    overlap_counts, train_overlaps = set(), set()
    tiny, large, edges, thirty = 0, 0, 0, 0
    role_changes = role_exchanges = swaps = moved_colors = 0
    varied_cavities = varied_marker_pairs = 0
    total_pairs = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        infos = {s: [_analyze_input(pair["input"]) for pair in puzzle[s]]
                 for s in ("train", "test")}
        train_sides = {i["side"] for i in infos["train"]}
        test_sides = {i["side"] for i in infos["test"]}
        sides_train.update(train_sides)
        sides_test.update(test_sides)
        bottom_trained += "bottom" in train_sides
        shared_side += bool(train_sides & test_sides)
        mixed_tests += len(test_sides) > 1
        withheld_bottom += test_sides == {"bottom"} and "bottom" not in train_sides
        withheld_other += bool(test_sides - train_sides - {"bottom"})
        role_pairs = [(i["background"], i["frame"]) for i in infos["train"] + infos["test"]]
        role_changes += len(set(role_pairs)) > 1
        role_exchanges += any((b, a) in role_pairs for a, b in role_pairs if a != b)
        palettes.add(tuple(role_pairs))
        training_colors = {c for i in infos["train"] for c in i["markers"].values()}
        for info in infos["test"]:
            overlap_counts.add(len(set(info["markers"].values()) & training_colors))
        first_markers = set(infos["train"][0]["markers"].values())
        for info in infos["train"][1:]:
            train_overlaps.add(len(set(info["markers"].values()) & first_markers))
        varied_cavities += len({(i["interior_height"], i["interior_width"])
                                for i in infos["train"] + infos["test"]}) > 1
        varied_marker_pairs += len({frozenset(i["markers"].values())
                                    for i in infos["train"] + infos["test"]}) > 1
        observed = []
        occupied_corners = {}
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], infos[split]):
                total_pairs += 1
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(out, _independent_prediction(inp))
                assert _pair_signature(pair) not in _SOURCE_PAIR_SIGNATURES
                shapes.add(inp.shape)
                cavities.add((info["interior_height"], info["interior_width"]))
                layouts.add((inp.shape, info["top"], info["left"],
                             info["interior_height"], info["interior_width"], info["side"]))
                (sides_train if split == "train" else sides_test).add(info["side"])
                tiny += (info["interior_height"], info["interior_width"]) in ((2, 2), (2, 4), (4, 2))
                large += max(info["interior_height"], info["interior_width"]) >= 20
                thirty += 30 in inp.shape
                opposite_margin = {
                    "top": inp.shape[0] - 1 - info["bottom"],
                    "bottom": info["top"], "left": inp.shape[1] - 1 - info["right"],
                    "right": info["left"],
                }[info["side"]]
                edges += opposite_margin == 0
                colors = tuple(info["markers"][c] for c in _SIDE_CORNERS[info["side"]])
                swaps += any(s == info["side"] and cs == colors[::-1] for s, cs in observed)
                observed.append((info["side"], colors))
                for corner, color in info["markers"].items():
                    moved_colors += color in occupied_corners and corner not in occupied_corners[color]
                    occupied_corners.setdefault(color, set()).add(corner)
    assert sides_train == sides_test == set(_SIDE_CORNERS)
    assert min(withheld_bottom, withheld_other, bottom_trained, shared_side, mixed_tests) >= 1
    assert overlap_counts == train_overlaps == {0, 1, 2}
    assert min(role_changes, role_exchanges, swaps, moved_colors,
               tiny, large, edges, thirty) >= 1
    assert varied_cavities == len(puzzles) and varied_marker_pairs >= 30
    assert len(shapes) >= 35 and len(cavities) >= 35 and len(layouts) >= total_pairs // 2
    assert len(palettes) >= 20
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1),
                        (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(6000 + (train or 0) * 10 + (test or 0), train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(7000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    for args in ((20260827, None, None), (314159, 7, 3)):
        first = generate_puzzle(*args)
        second = generate_puzzle(*args)
        for split in ("train", "test"):
            for a, b in zip(first[split], second[split]):
                assert all(np.array_equal(a[k], b[k]) for k in ("input", "output"))
    assert all(not np.array_equal(puzzles[0]["train"][0]["input"],
                                      other["train"][0]["input"])
               for other in puzzles[1:])
    valid = generate_puzzle(8080, 5, 2)
    broken = {split: [{k: v.copy() for k, v in pair.items()} for pair in valid[split]]
              for split in ("train", "test")}
    info = _analyze_input(broken["test"][0]["input"])
    broken["test"][0]["output"][info["top"] + 1, info["left"] + 1] = info["background"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
