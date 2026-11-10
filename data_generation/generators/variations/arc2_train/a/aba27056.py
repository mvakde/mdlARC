import numpy as np


_SIDES = ("left", "up", "right", "down")

_AUTOMATIC_COUNT_SCHEDULE = (
    (2, 1),
    (3, 1),
    (4, 2),
    (5, 1),
    (5, 3),
    (6, 2),
    (7, 3),
    (8, 4),
    (9, 2),
    (10, 4),
    (10, 1),
    (6, 4),
)

_SOURCE_PAIR_ROWS = frozenset(
    {
        (
            (
                "000000000",
                "000000000",
                "000077777",
                "000070007",
                "000000007",
                "000000007",
                "000000007",
                "000070007",
                "000077777",
            ),
            (
                "400000000",
                "040000000",
                "004077777",
                "000474447",
                "444444447",
                "444444447",
                "444444447",
                "000474447",
                "004077777",
            ),
        ),
        (
            (
                "0000000",
                "0000000",
                "0000000",
                "0000000",
                "0660660",
                "0600060",
                "0666660",
            ),
            (
                "0004000",
                "4004004",
                "0404040",
                "0044400",
                "0664660",
                "0644460",
                "0666660",
            ),
        ),
        (
            (
                "333333",
                "300003",
                "300003",
                "330033",
                "000000",
                "000000",
            ),
            (
                "333333",
                "344443",
                "344443",
                "334433",
                "044440",
                "404404",
            ),
        ),
        (
            (
                "0222200000",
                "0200200000",
                "0200000000",
                "0200000000",
                "0200000000",
                "0200000000",
                "0200000000",
                "0200200000",
                "0222200000",
                "0000000000",
            ),
            (
                "0222204000",
                "0244240000",
                "0244444444",
                "0244444444",
                "0244444444",
                "0244444444",
                "0244444444",
                "0244240000",
                "0222204000",
                "0000000400",
            ),
        ),
    }
)


def _validate_requested_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    rng: np.random.Generator,
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_requested_count("num_train", num_train, 2, 10)
    _validate_requested_count("num_test", num_test, 1, 4)

    train = None if num_train is None else int(num_train)
    test = None if num_test is None else int(num_test)

    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must be greater than num_test")
        return train, test

    if train is None and test is None:
        index = int(seed) % len(_AUTOMATIC_COUNT_SCHEDULE)
        return _AUTOMATIC_COUNT_SCHEDULE[index]

    if train is None:
        assert test is not None
        low = max(2, test + 1)
        train = int(rng.integers(low, 11))
        return train, test

    assert test is None
    highest_test = min(4, train - 1)
    if highest_test < 1:
        raise ValueError("no legal test count exists for num_train")
    test = int(rng.integers(1, highest_test + 1))
    return train, test


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    return {"added": int(rng.integers(0, 10))}


def _sample_covering_sequence(
    rng: np.random.Generator,
    values: tuple,
    length: int,
) -> list:
    if length <= len(values):
        indices = rng.permutation(len(values))[:length]
        return [values[int(i)] for i in indices]

    result = list(values)
    while len(result) < length:
        result.append(values[int(rng.integers(0, len(values)))])

    order = rng.permutation(len(result))
    return [result[int(i)] for i in order]


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        int(grid.shape[0]),
        int(grid.shape[1]),
        tuple(int(value) for value in grid.ravel()),
    )


def _rows_key(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid.tolist()
    )


def _pair_key(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return _grid_key(input_grid), _grid_key(output_grid)


def _is_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    return (_rows_key(input_grid), _rows_key(output_grid)) in _SOURCE_PAIR_ROWS


def _candidate_geometry(
    grid: np.ndarray,
    background: int,
    frame: int,
) -> dict | None:
    frame_mask = grid == frame
    rows, cols = np.nonzero(frame_mask)
    if len(rows) == 0:
        return None

    top = int(rows.min())
    bottom = int(rows.max())
    left = int(cols.min())
    right = int(cols.max())

    if bottom - top < 2 or right - left < 2:
        return None

    boundary = np.zeros(grid.shape, dtype=bool)
    boundary[top, left : right + 1] = True
    boundary[bottom, left : right + 1] = True
    boundary[top : bottom + 1, left] = True
    boundary[top : bottom + 1, right] = True

    if np.any(frame_mask & ~boundary):
        return None

    missing = np.argwhere(boundary & ~frame_mask)
    if len(missing) == 0:
        return None

    matches = []
    if np.all(missing[:, 1] == left):
        matches.append("left")
    if np.all(missing[:, 1] == right):
        matches.append("right")
    if np.all(missing[:, 0] == top):
        matches.append("up")
    if np.all(missing[:, 0] == bottom):
        matches.append("down")

    if len(matches) != 1:
        return None

    side = matches[0]
    if side in ("left", "right"):
        ordered = sorted(int(value) for value in missing[:, 0])
        low, high = top, bottom
    else:
        ordered = sorted(int(value) for value in missing[:, 1])
        low, high = left, right

    if ordered != list(range(ordered[0], ordered[-1] + 1)):
        return None

    before_margin = ordered[0] - low - 1
    after_margin = high - ordered[-1] - 1

    # The opening must be central and must have an intact frame cell
    # immediately on both sides.
    if before_margin < 1 or before_margin != after_margin:
        return None

    return {
        "background": int(background),
        "frame": int(frame),
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "side": side,
        "gap_start": ordered[0],
        "gap_end": ordered[-1],
        "gap_width": len(ordered),
        "margin": before_margin,
    }


def _analyze_input(grid: np.ndarray) -> dict:
    values = [int(value) for value in np.unique(grid)]
    assert len(values) == 2

    candidates = []
    for frame in values:
        background = values[1] if frame == values[0] else values[0]
        candidate = _candidate_geometry(grid, background, frame)
        if candidate is not None:
            candidates.append(candidate)

    assert len(candidates) == 1
    return candidates[0]


def _inside_grid(
    row: int,
    col: int,
    height: int,
    width: int,
) -> bool:
    return 0 <= row < height and 0 <= col < width


def _expected_output(
    input_grid: np.ndarray,
    geometry: dict,
    added: int,
) -> np.ndarray:
    output = input_grid.copy()
    height, width = output.shape

    background = geometry["background"]
    top = geometry["top"]
    bottom = geometry["bottom"]
    left = geometry["left"]
    right = geometry["right"]
    side = geometry["side"]
    gap_start = geometry["gap_start"]
    gap_end = geometry["gap_end"]

    # Fill the strict rectangular interior.
    output[top + 1 : bottom, left + 1 : right] = added

    if side == "left":
        outward = (0, -1)
        opening = [(row, left) for row in range(gap_start, gap_end + 1)]
        flanks = (
            ((gap_start - 1, left), (-1, 0)),
            ((gap_end + 1, left), (1, 0)),
        )
    elif side == "right":
        outward = (0, 1)
        opening = [(row, right) for row in range(gap_start, gap_end + 1)]
        flanks = (
            ((gap_start - 1, right), (-1, 0)),
            ((gap_end + 1, right), (1, 0)),
        )
    elif side == "up":
        outward = (-1, 0)
        opening = [(top, col) for col in range(gap_start, gap_end + 1)]
        flanks = (
            ((top, gap_start - 1), (0, -1)),
            ((top, gap_end + 1), (0, 1)),
        )
    else:
        assert side == "down"
        outward = (1, 0)
        opening = [(bottom, col) for col in range(gap_start, gap_end + 1)]
        flanks = (
            ((bottom, gap_start - 1), (0, -1)),
            ((bottom, gap_end + 1), (0, 1)),
        )

    out_row, out_col = outward

    # Each opening cell emits a perpendicular ray, including the gap cell.
    for start_row, start_col in opening:
        row, col = start_row, start_col
        while _inside_grid(row, col, height, width):
            if output[row, col] == background:
                output[row, col] = added
            row += out_row
            col += out_col

    # A flanker first steps perpendicularly outside the frame. Subsequent
    # layers step both outward and away from the opening.
    for (flank_row, flank_col), (away_row, away_col) in flanks:
        layer = 1
        while True:
            row = flank_row + layer * out_row + (layer - 1) * away_row
            col = flank_col + layer * out_col + (layer - 1) * away_col
            if not _inside_grid(row, col, height, width):
                break
            if output[row, col] == background:
                output[row, col] = added
            layer += 1

    return output


def _build_input(
    height: int,
    width: int,
    background: int,
    frame: int,
    top: int,
    bottom: int,
    left: int,
    right: int,
    side: str,
    gap_start: int,
    gap_end: int,
) -> np.ndarray:
    grid = np.full((height, width), background, dtype=np.int8)

    grid[top, left : right + 1] = frame
    grid[bottom, left : right + 1] = frame
    grid[top : bottom + 1, left] = frame
    grid[top : bottom + 1, right] = frame

    if side == "left":
        grid[gap_start : gap_end + 1, left] = background
    elif side == "right":
        grid[gap_start : gap_end + 1, right] = background
    elif side == "up":
        grid[top, gap_start : gap_end + 1] = background
    else:
        assert side == "down"
        grid[bottom, gap_start : gap_end + 1] = background

    return grid


def _clip_lengths(geometry: dict, shape: tuple[int, int]) -> tuple[int, int, int]:
    """Outward clearance and the two possible diagonal lengths."""
    height, width = shape
    side = geometry["side"]
    if side == "left":
        clearance = geometry["left"]
        first = geometry["gap_start"]
        second = height - geometry["gap_end"] - 1
    elif side == "right":
        clearance = width - geometry["right"] - 1
        first = geometry["gap_start"]
        second = height - geometry["gap_end"] - 1
    elif side == "up":
        clearance = geometry["top"]
        first = geometry["gap_start"]
        second = width - geometry["gap_end"] - 1
    else:
        clearance = height - geometry["bottom"] - 1
        first = geometry["gap_start"]
        second = width - geometry["gap_end"] - 1
    return clearance, min(clearance, first), min(clearance, second)


def _sample_geometry(
    rng: np.random.Generator,
    side: str,
    gap_width: int,
    style: str,
    large: bool,
    compact: bool,
) -> tuple[int, int, int, int, int, int, int, int]:
    """Sample local rectangle coordinates, then rotate them into the grid."""
    margin = int(rng.integers(1, 3) if compact else
                 rng.integers(2, 6) if large else rng.integers(1, 6))
    side_length = gap_width + 2 * margin + 2
    depth = int(rng.integers(3, 7) if compact else
                rng.integers(9, 17) if large else rng.integers(3, 17))
    along_budget = 30 - side_length
    outward_budget = 30 - depth

    if style == "zero":
        clearance = 0
    elif style == "short":
        clearance = 1
    elif style in ("first", "second", "both"):
        clearance = int(rng.integers(margin + 2, min(margin + 7, outward_budget + 1)))
    elif style == "none":
        maximum = min(4 if compact else 10, outward_budget,
                      margin + 1 + along_budget // 2)
        clearance = int(rng.integers(2, maximum + 1))
    elif style == "showcase":
        clearance = int(rng.integers(3, min(5 if compact else 7,
                                             outward_budget + 1)))
    else:
        assert style == "long"
        clearance = int(rng.integers(7, min(13, outward_budget + 1)))

    pad_cap = min(2 if compact else 9, along_budget)
    if style == "none":
        minimum = max(0, clearance - margin - 1)
        before = int(rng.integers(minimum, min(pad_cap, along_budget - minimum) + 1))
        after = int(rng.integers(minimum, min(pad_cap, along_budget - before) + 1))
    elif style in ("first", "second"):
        required = clearance - margin - 1
        clipped = int(rng.integers(0, min(pad_cap, required - 1) + 1))
        intact = int(rng.integers(required, min(pad_cap, along_budget - clipped) + 1))
        before, after = (clipped, intact) if style == "first" else (intact, clipped)
    elif style == "both":
        maximum = clearance - margin - 2
        before = int(rng.integers(0, min(pad_cap, maximum) + 1))
        after = int(rng.integers(0, min(pad_cap, maximum, along_budget - before) + 1))
    else:
        before = int(rng.integers(0, pad_cap + 1))
        after = int(rng.integers(0, min(pad_cap, along_budget - before) + 1))
    if large:
        after = max(after, 14 - side_length - before)

    behind = int(rng.integers(0, min(2 if compact else 9,
                                     30 - depth - clearance) + 1))
    gap_start_local = margin + 1
    gap_end_local = gap_start_local + gap_width - 1

    if side in ("left", "right"):
        height = side_length + before + after
        width = depth + clearance + behind
        top = before
        bottom = top + side_length - 1
        left = clearance if side == "left" else behind
        right = left + depth - 1
        gap_start = top + gap_start_local
        gap_end = top + gap_end_local
    else:
        width = side_length + before + after
        height = depth + clearance + behind
        left = before
        right = left + side_length - 1
        top = clearance if side == "up" else behind
        bottom = top + depth - 1
        gap_start = left + gap_start_local
        gap_end = left + gap_end_local

    return height, width, top, bottom, left, right, gap_start, gap_end


def _orientation_plan(
    rng: np.random.Generator,
    seed: int,
    train_count: int,
    test_count: int,
) -> tuple[list[str], list[str]]:
    mode = int(seed) % 5
    rotation = int(rng.integers(0, 4))
    sides = list(_SIDES[rotation:] + _SIDES[:rotation])
    if mode == 0:
        sides = list(_SIDES)
        test = [sides[2]] * test_count
        train = [sides[i % 3] for i in range(train_count) if sides[i % 3] != sides[2]]
        while len(train) < train_count:
            train.append(sides[int(rng.choice((0, 1, 3)))])
    elif mode == 1:
        focus = sides[0]
        train = [focus, sides[1]]
        train += [sides[int(rng.integers(0, 4))] for _ in range(train_count - 2)]
        test = [focus] * test_count
    elif mode == 2:
        heldout = sides[3]
        train = [sides[i % 3] for i in range(train_count)]
        test = [heldout] * test_count
    elif mode == 3:
        train = [sides[0], sides[1]]
        train += [sides[int(rng.integers(0, 4))] for _ in range(train_count - 2)]
        test = [sides[(i + 1) % 4] for i in range(test_count)]
    else:
        train = [sides[0], sides[2]]
        train += [sides[int(rng.integers(0, 4))] for _ in range(train_count - 2)]
        test = [sides[int(rng.integers(0, 4))] for _ in range(test_count)]
    return train, test


def _choose_colors(
    rng: np.random.Generator,
    added: int,
    trained_frames: list[int],
    trained_backgrounds: list[int],
    reserved_frame: int,
    test: bool,
    reuse: bool | None,
) -> tuple[int, int]:
    available = [color for color in range(10) if color != added]
    if test and (reuse if reuse is not None else int(rng.integers(0, 2)) == 0):
        frame = int(rng.choice(trained_frames))
    elif test:
        unseen = [color for color in available if color not in trained_frames]
        frame = int(rng.choice(unseen))
    else:
        frames = [color for color in available if color != reserved_frame]
        if trained_frames:
            frames = [color for color in frames if color != trained_frames[-1]]
        frame = int(rng.choice(frames))
    backgrounds = [color for color in available if color != frame]
    if trained_backgrounds and not test:
        backgrounds = [color for color in backgrounds
                       if color != trained_backgrounds[-1]]
    background = int(rng.choice(backgrounds))
    return background, frame


def _make_pair(
    rng: np.random.Generator,
    side: str,
    gap_width: int,
    style: str,
    added: int,
    trained_frames: list[int],
    trained_backgrounds: list[int],
    reserved_frame: int,
    test: bool,
    reuse: bool | None,
    large: bool,
    compact: bool,
) -> tuple[np.ndarray, np.ndarray]:
    background, frame = _choose_colors(rng, added, trained_frames,
                                       trained_backgrounds, reserved_frame,
                                       test, reuse)
    geometry = _sample_geometry(rng, side, gap_width, style, large, compact)
    input_grid = _build_input(*geometry[:2], background, frame, *geometry[2:6],
                              side, *geometry[6:])
    parsed = _analyze_input(input_grid)
    output_grid = _expected_output(input_grid, parsed, added)
    return input_grid, output_grid


def _spawn_rng(rng: np.random.Generator) -> np.random.Generator:
    return np.random.default_rng(int(rng.bit_generator.random_raw()))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    added = sample_puzzle_parameters(rng)["added"]
    train_sides, test_sides = _orientation_plan(rng, seed, train_count, test_count)
    reserved_frame = int(rng.choice([color for color in range(10)
                                     if color != added]))
    styles = ("none", "first", "second", "both", "zero", "short", "long")
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    trained_frames = []
    trained_backgrounds = []

    for section, count, sides in (("train", train_count, train_sides),
                                  ("test", test_count, test_sides)):
        for index in range(count):
            test = section == "test"
            if not test and index == 0:
                style = "showcase"
            elif not test and int(seed) % 5 == 0:
                style = ("none", "short")[int(rng.integers(0, 2))]
            elif test and int(seed) % 5 == 0:
                style = "long"
            else:
                style = styles[int(rng.integers(0, len(styles)))]
            if int(seed) % 5 == 0:
                gap_width = int(rng.integers(4, 9) if test else rng.integers(1, 4))
            else:
                gap_width = int(rng.integers(1, 9) if test else rng.integers(1, 7))
            large = bool(test and int(seed) % 5 == 0)
            compact = bool(not test and int(seed) % 5 == 0)
            reuse = (True if index == 0 else False if index == 1 else None) if (
                test and test_count >= 2) else (False if test and int(seed) % 5 == 0
                                                   else None)
            for _ in range(24):
                pair_rng = _spawn_rng(rng)
                inp, out = _make_pair(pair_rng, sides[index], gap_width, style,
                                      added, trained_frames, trained_backgrounds,
                                      reserved_frame, test, reuse, large, compact)
                if _grid_key(inp) in used_inputs or _is_source_pair(inp, out):
                    continue
                puzzle[section].append({"input": inp, "output": out})
                used_inputs.add(_grid_key(inp))
                if not test:
                    parsed = _analyze_input(inp)
                    trained_frames.append(parsed["frame"])
                    trained_backgrounds.append(parsed["background"])
                break
            else:
                raise ValueError("unable to construct enough unique valid pairs")

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[section], list) for section in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    input_keys = set()
    pair_keys = set()
    train_frames = set()
    train_geometries = []
    added = None
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            input_key, pair_key = _grid_key(inp), _pair_key(inp, out)
            assert input_key not in input_keys and pair_key not in pair_keys
            input_keys.add(input_key)
            pair_keys.add(pair_key)
            assert not _is_source_pair(inp, out)

            geometry = _analyze_input(inp)
            input_colors = {int(value) for value in np.unique(inp)}
            output_colors = {int(value) for value in np.unique(out)}
            new_colors = output_colors - input_colors
            assert len(new_colors) == 1
            pair_added = next(iter(new_colors))
            if added is None:
                assert section == "train"
                added = pair_added
            assert pair_added == added
            assert input_colors == {geometry["background"], geometry["frame"]}
            assert output_colors <= input_colors | {added}
            assert {geometry["frame"], added} <= output_colors
            assert np.array_equal(out, _expected_output(inp, geometry, added))
            assert np.all(out[inp == geometry["frame"]] == geometry["frame"])
            clearance, first, second = _clip_lengths(geometry, inp.shape)
            assert 0 <= first <= clearance and 0 <= second <= clearance
            if section == "train":
                train_geometries.append((geometry, inp.shape))
                train_frames.add(geometry["frame"])
    assert added is not None
    assert len({geo["side"] for geo, _ in train_geometries}) >= 2
    assert any(
        (lambda lengths: lengths[0] >= 3 and lengths[1] >= 2 and lengths[2] >= 2)(
            _clip_lengths(geo, shape)
        ) for geo, shape in train_geometries
    )
    # Every test's only new color occurs in training outputs. The frame and
    # background are identifiable locally, including for previously unseen colors.
    assert all(
        added not in {int(v) for v in pair["input"].flat}
        for section in ("train", "test") for pair in puzzle[section]
    )


def _predict_from_examples(train: list[dict], input_grid: np.ndarray) -> np.ndarray:
    """Infer the missing color from demonstrations and apply the stencil afresh."""
    new_colors = [
        {int(v) for v in np.unique(pair["output"])} -
        {int(v) for v in np.unique(pair["input"])}
        for pair in train
    ]
    assert all(len(colors) == 1 for colors in new_colors)
    added = next(iter(new_colors[0]))
    assert all(next(iter(colors)) == added for colors in new_colors)
    geometry = _analyze_input(input_grid)
    result = input_grid.copy()
    top, bottom = geometry["top"], geometry["bottom"]
    left, right = geometry["left"], geometry["right"]
    start, end = geometry["gap_start"], geometry["gap_end"]
    result[top + 1:bottom, left + 1:right] = added

    if geometry["side"] in ("left", "right"):
        outward = (0, -1 if geometry["side"] == "left" else 1)
        edge = left if geometry["side"] == "left" else right
        opening = [(position, edge) for position in range(start, end + 1)]
        flanks = ((start - 1, edge, -1, 0),
                  (end + 1, edge, 1, 0))
    else:
        outward = (-1 if geometry["side"] == "up" else 1, 0)
        edge = top if geometry["side"] == "up" else bottom
        opening = [(edge, position) for position in range(start, end + 1)]
        flanks = ((edge, start - 1, 0, -1),
                  (edge, end + 1, 0, 1))

    for row, col in opening:
        while _inside_grid(row, col, *result.shape):
            result[row, col] = added
            row += outward[0]
            col += outward[1]
    for row, col, away_row, away_col in flanks:
        layer = 1
        while True:
            y = row + layer * outward[0] + (layer - 1) * away_row
            x = col + layer * outward[1] + (layer - 1) * away_col
            if not _inside_grid(y, x, *result.shape):
                break
            result[y, x] = added
            layer += 1
    return result


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple(_pair_key(pair["input"], pair["output"])
                       for pair in puzzle[section]) for section in ("train", "test"))


def _assert_value_error(**kwargs) -> None:
    try:
        generate_puzzle(12345, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for arguments {kwargs}")


def self_test() -> None:
    source_pairs = [
        {"input": np.array([[int(value) for value in row] for row in input_rows],
                           dtype=np.int8),
         "output": np.array([[int(value) for value in row] for row in output_rows],
                            dtype=np.int8)}
        for input_rows, output_rows in _SOURCE_PAIR_ROWS
    ]
    for pair in source_pairs:
        assert np.array_equal(
            _predict_from_examples(source_pairs, pair["input"]), pair["output"]
        )

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) >= 40

    records = []
    layouts = set()
    relationship_counts = {name: 0 for name in
                           ("right_train", "mixed_test", "same_orientation",
                            "heldout_nonright", "scale_up", "scale_down")}
    for puzzle in puzzles:
        train_frames = {_analyze_input(p["input"])["frame"] for p in puzzle["train"]}
        train_backgrounds = {_analyze_input(p["input"])["background"]
                             for p in puzzle["train"]}
        assert len(train_frames) >= 2 and len(train_backgrounds) >= 2
        if len(puzzle["test"]) >= 2:
            test_frames = [_analyze_input(p["input"])["frame"]
                           for p in puzzle["test"]]
            assert any(frame in train_frames for frame in test_frames)
            assert any(frame not in train_frames for frame in test_frames)
        train_sides = {_analyze_input(p["input"])["side"] for p in puzzle["train"]}
        test_sides = {_analyze_input(p["input"])["side"] for p in puzzle["test"]}
        relationship_counts["right_train"] += "right" in train_sides
        relationship_counts["mixed_test"] += len(test_sides) > 1
        relationship_counts["same_orientation"] += test_sides <= train_sides
        relationship_counts["heldout_nonright"] += bool(
            test_sides - train_sides - {"right"}
        )
        max_height = max(p["input"].shape[0] for p in puzzle["train"])
        max_width = max(p["input"].shape[1] for p in puzzle["train"])
        relationship_counts["scale_up"] += any(
            p["input"].shape[0] > max_height and
            p["input"].shape[1] > max_width for p in puzzle["test"]
        )
        relationship_counts["scale_down"] += any(
            p["input"].shape[0] < max_height and
            p["input"].shape[1] < max_width for p in puzzle["test"]
        )
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                geo = _analyze_input(inp)
                added = next(iter({int(v) for v in np.unique(out)} -
                                  {int(v) for v in np.unique(inp)}))
                assert np.array_equal(out, _expected_output(inp, geo, added))
                assert np.array_equal(
                    out, _predict_from_examples(puzzle["train"], inp)
                )
                clearance, first, second = _clip_lengths(geo, inp.shape)
                category = ("zero" if clearance == 0 else
                            "none" if first == clearance and second == clearance else
                            "both" if first < clearance and second < clearance else
                            "first" if first < clearance else "second")
                records.append((section, geo, inp.shape, added, category,
                                geo["frame"] in train_frames, clearance))
                layouts.add((inp.shape, geo["side"], geo["top"], geo["left"],
                             geo["bottom"], geo["right"], geo["gap_width"]))
    assert len(layouts) >= 180
    assert len({r[2] for r in records}) >= 60
    assert len({r[3] for r in records}) >= 7
    assert {r[1]["side"] for r in records if r[0] == "train"} == set(_SIDES)
    assert {r[1]["side"] for r in records if r[0] == "test"} == set(_SIDES)
    assert {r[4] for r in records if r[0] == "test"} == {
        "zero", "none", "both", "first", "second"
    }
    assert {r[1]["margin"] for r in records} == set(range(1, 6))
    assert max(r[1]["gap_width"] for r in records) >= 8
    assert any(r[0] == "test" and r[5] for r in records)
    assert any(r[0] == "test" and not r[5] for r in records)
    assert any(r[0] == "test" and r[6] == 1 for r in records)
    assert any(r[0] == "test" and r[6] >= 7 for r in records)
    assert len({(r[1]["side"], r[4]) for r in records if r[0] == "test"}) >= 16
    assert relationship_counts["right_train"] >= 10
    assert relationship_counts["mixed_test"] >= 5
    assert relationship_counts["same_orientation"] >= 10
    assert relationship_counts["heldout_nonright"] >= 5
    assert relationship_counts["scale_up"] >= 8
    assert relationship_counts["scale_down"] >= 15

    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (7, 3)):
        puzzle = generate_puzzle(20000 + train_count * 10 + test_count,
                                 train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        validate_generated_puzzle(puzzle)
    for index, request in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(30000 + index, *request)
        validate_generated_puzzle(puzzle)
        assert request[0] is None or len(puzzle["train"]) == request[0]
        assert request[1] is None or len(puzzle["test"]) == request[1]
    assert _puzzle_key(generate_puzzle(9001, 7, 3)) == _puzzle_key(
        generate_puzzle(9001, 7, 3))
    assert _puzzle_key(generate_puzzle(1001, 6, 2)) != _puzzle_key(
        generate_puzzle(1002, 6, 2))
    for request in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4}, {"num_train": True},
        {"num_test": 2.5},
    ):
        _assert_value_error(**request)
