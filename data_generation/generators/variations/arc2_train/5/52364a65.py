import numpy as np


_SOURCE_ROWS = (
    (
        (
            "88880088888",
            "88880088888",
            "88000000888",
            "88880088888",
            "88880088888",
            "33388888888",
            "33338888999",
            "88888888999",
            "66688888888",
            "66666888888",
            "88866888888",
        ),
        (
            "88880088888",
            "88880088888",
            "88880000888",
            "88880088888",
            "88880088888",
            "88388888888",
            "88338888889",
            "88888888889",
            "88688888888",
            "88666888888",
            "88866888888",
        ),
    ),
    (
        (
            "899999888",
            "899999888",
            "888888808",
            "880000008",
            "888888808",
            "822222888",
            "888228888",
            "888288833",
            "888888833",
        ),
        (
            "888999888",
            "888999888",
            "888888808",
            "888800008",
            "888888808",
            "888222888",
            "888228888",
            "888288888",
            "888888888",
        ),
    ),
    (
        (
            "88888866",
            "88888866",
            "48848866",
            "44448888",
            "48848888",
            "88888888",
            "22285555",
            "22285555",
        ),
        (
            "88888888",
            "88888888",
            "88848888",
            "88448888",
            "88848888",
            "88888888",
            "88288855",
            "88288855",
        ),
    ),
)


def _rows_to_array(rows: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = tuple(
    (_rows_to_array(input_rows), _rows_to_array(output_rows))
    for input_rows, output_rows in _SOURCE_ROWS
)


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _validate_requested_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_requested_count(
        "num_train", num_train, 2, 10
    )
    num_test = _validate_requested_count(
        "num_test", num_test, 1, 4
    )

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum_train = num_test + 1
        span = 11 - minimum_train
        num_train = minimum_train + (seed // 4) % span
    elif num_train is None:
        assert num_test is not None
        minimum_train = max(2, num_test + 1)
        span = 11 - minimum_train
        num_train = minimum_train + seed % span
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = 1 + seed % maximum_test

    assert num_train is not None
    assert num_test is not None

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _infer_background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    maximum = int(np.max(counts))
    winner_indices = np.flatnonzero(counts == maximum)
    assert len(winner_indices) == 1, "background must be unambiguous"
    return int(values[int(winner_indices[0])])


def _extract_objects(
    grid: np.ndarray,
    background: int,
) -> list[dict]:
    height, width = grid.shape
    visited = np.zeros(grid.shape, dtype=bool)
    objects: list[dict] = []

    for color_value in np.unique(grid):
        color = int(color_value)
        if color == background:
            continue

        starts = np.argwhere(grid == color)
        for start_row_value, start_col_value in starts:
            start_row = int(start_row_value)
            start_col = int(start_col_value)
            if visited[start_row, start_col]:
                continue

            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True
            coordinates: list[tuple[int, int]] = []

            while stack:
                row, col = stack.pop()
                coordinates.append((row, col))

                for delta_row, delta_col in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ):
                    next_row = row + delta_row
                    next_col = col + delta_col
                    if not (
                        0 <= next_row < height
                        and 0 <= next_col < width
                    ):
                        continue
                    if visited[next_row, next_col]:
                        continue
                    if int(grid[next_row, next_col]) != color:
                        continue

                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))

            coordinate_array = np.asarray(
                coordinates, dtype=np.int16
            )
            minimum_row = int(np.min(coordinate_array[:, 0]))
            maximum_row = int(np.max(coordinate_array[:, 0]))
            minimum_col = int(np.min(coordinate_array[:, 1]))
            maximum_col = int(np.max(coordinate_array[:, 1]))

            objects.append(
                {
                    "color": color,
                    "coords": coordinate_array,
                    "bbox": (
                        minimum_row,
                        maximum_row,
                        minimum_col,
                        maximum_col,
                    ),
                }
            )

    objects.sort(
        key=lambda obj: (
            obj["color"],
            obj["bbox"][0],
            obj["bbox"][2],
        )
    )
    return objects


def _derive_output(input_grid: np.ndarray) -> np.ndarray:
    background = _infer_background(input_grid)
    output_grid = input_grid.copy()

    for obj in _extract_objects(input_grid, background):
        coordinates = obj["coords"]
        left_col = int(obj["bbox"][2])
        erase = coordinates[:, 1] < left_col + 2
        erased_coordinates = coordinates[erase]
        output_grid[
            erased_coordinates[:, 0],
            erased_coordinates[:, 1],
        ] = background

    return output_grid


def _make_mask(rng: np.random.Generator, height: int, width: int,
               mode: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if width == 1:
        mask[:, 0] = True
        return mask
    if mode == "diagnostic":
        spine = width - 1
        mask[:, spine] = True
        bridge = int(rng.integers(0, height))
        sparse = (bridge + int(rng.integers(1, height))) % height
        mask[bridge, :] = True
        for row in range(height):
            if row != bridge and row != sparse and rng.random() < .45:
                left = int(rng.integers(0, spine + 1))
                mask[row, left:] = True
        return mask
    if mode == "ring" and height >= 4 and width >= 4:
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        if height > 4 and width > 4 and rng.random() < .5:
            mask[1, 1] = True
    elif mode == "u" and width >= 3:
        mask[:, 0] = mask[:, -1] = True
        mask[-1, :] = True
    elif mode == "h" and width >= 3:
        mask[:, 0] = mask[:, -1] = True
        mask[int(rng.integers(1, height - 1)) if height > 2 else 0, :] = True
    elif mode == "cross":
        mask[int(rng.integers(0, height)), :] = True
        mask[:, int(rng.integers(0, width))] = True
    elif mode == "c":
        mask[:, 0] = True
        mask[0, :] = mask[-1, :] = True
    elif mode == "stair":
        spine = int(rng.integers(0, width))
        for row in range(height):
            mask[row, spine] = True
            if row + 1 < height:
                nxt = min(width - 1, max(0, spine + int(rng.integers(-1, 2))))
                mask[row, min(spine, nxt):max(spine, nxt) + 1] = True
                spine = nxt
        mask[int(rng.integers(0, height)), :] = True
    else:
        spine = int(rng.integers(0, width))
        mask[:, spine] = True
        for row in range(height):
            if rng.random() < .7:
                left = int(rng.integers(0, spine + 1))
                right = int(rng.integers(spine, width))
                mask[row, left:right + 1] = True
        mask[int(rng.integers(0, height)), :] = True
    if rng.random() < .5:
        mask = np.flip(mask, axis=1)
    if rng.random() < .5:
        mask = np.flip(mask, axis=0)
    return mask


def _object_specs(rng: np.random.Generator, section: str,
                  index: int, simple: bool) -> list[tuple[str, int, int]]:
    modes = ("ring", "u", "h", "cross", "c", "stair", "branch")
    if simple:
        count = 3
        if section == "train" and index == 0:
            specs = [("diagnostic", 4, int(rng.integers(3, 6))),
                     ("branch", 2, int(rng.integers(2, 5)))]
        elif section == "train" and index == 1:
            specs = [("ring", 4, 4)]
        elif section == "test":
            specs = [(str(rng.choice(modes)), 4,
                      int(rng.integers(3, 6)))]
        else:
            specs = []
    elif section == "train" and index == 0:
        count = int(rng.integers(3, 6))
        specs = [("diagnostic", int(rng.integers(4, 8)),
                  int(rng.integers(3, 7))),
                 ("branch", 2, int(rng.integers(2, 6)))]
    elif section == "train" and index == 1:
        count = int(rng.integers(2, 6))
        specs = [("ring", int(rng.integers(4, 7)),
                  int(rng.integers(4, 7)))]
    elif section == "test":
        count = int(rng.integers(2, 6))
        specs = [(str(rng.choice(modes)), int(rng.integers(4, 8)),
                  int(rng.integers(4, 7)))]
    else:
        count = int(rng.integers(2, 6))
        specs = []
    if simple:
        narrow_count = int(rng.integers(0, 3))
    elif section == "test":
        narrow_count = int(rng.choice((0, 1, 2), p=(.38, .37, .25)))
        narrow_count = min(narrow_count, count - 1)
    else:
        narrow_count = int(rng.choice((0, 1, 2), p=(.43, .40, .17)))
    while len(specs) < count:
        if narrow_count:
            width = int(rng.integers(1, 3))
            narrow_count -= 1
        else:
            width = (int(rng.integers(3, 5)) if simple else
                     int(rng.choice((3, 4, 5, 6, 7),
                                    p=(.22, .28, .25, .17, .08))))
        height = int(rng.integers(2 if width <= 2 else 3,
                                  5 if simple else 7))
        mode = str(rng.choice(modes))
        if mode == "ring" and (height < 4 or width < 4):
            mode = "branch"
        specs.append((mode, width, height))
    return specs


def _canvas_shape(rng: np.random.Generator, count: int,
                  section: str, index: int, attempt: int,
                  simple: bool) -> tuple[int, int]:
    if simple and attempt < 12:
        return (int(rng.integers(10, 14)),
                int(rng.integers(10, 14)))
    minimum = {2: 8, 3: 9, 4: 11, 5: 13}[count]
    if section == "train" and index < 2:
        minimum = max(minimum, 11)
    if attempt >= 12:
        minimum = max(minimum, 18)
    upper = 22 if attempt >= 12 else (20 if section == "test" else 19)
    return (int(rng.integers(minimum, upper + 1)),
            int(rng.integers(minimum, upper + 1)))


def _place_masks(rng: np.random.Generator, shape: tuple[int, int],
                 masks: list[np.ndarray]) -> list[tuple[int, int]] | None:
    height, width = shape
    placed: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int] | None] = [None] * len(masks)
    order = sorted(range(len(masks)),
                   key=lambda i: masks[i].size, reverse=True)
    for i in order:
        obj_h, obj_w = masks[i].shape
        if obj_h > height or obj_w > width:
            return None
        for _ in range(130):
            row = int(rng.integers(0, height - obj_h + 1))
            col = int(rng.integers(0, width - obj_w + 1))
            if all(row + obj_h <= r or r + h <= row or
                   col + obj_w <= c or c + w <= col
                   for r, c, h, w in placed):
                placed.append((row, col, obj_h, obj_w))
                positions[i] = (row, col)
                break
        else:
            return None
    return [p for p in positions if p is not None]


def _crop_canvas(grid: np.ndarray, rng: np.random.Generator,
                 dense: bool) -> np.ndarray:
    background = _infer_background(grid)
    rows, cols = np.nonzero(grid != background)
    margin = 0 if dense else int(rng.integers(0, 3))
    r0 = max(0, int(rows.min()) - margin)
    r1 = min(grid.shape[0], int(rows.max()) + 1 + margin)
    c0 = max(0, int(cols.min()) - margin)
    c1 = min(grid.shape[1], int(cols.max()) + 1 + margin)
    while r1 - r0 < 7:
        if r0 > 0:
            r0 -= 1
        elif r1 < grid.shape[0]:
            r1 += 1
        else:
            break
    while c1 - c0 < 7:
        if c0 > 0:
            c0 -= 1
        elif c1 < grid.shape[1]:
            c1 += 1
        else:
            break
    return grid[r0:r1, c0:c1].copy()


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    colors = [int(x) for x in rng.permutation(10)]
    simple = bool(rng.integers(0, 6) == 0)
    rotating = False if simple else bool(rng.integers(0, 2))
    return {"base": colors[0], "alternate": colors[1],
            "rotating": rotating,
            "simple": simple,
            "palette_bank": tuple(colors[2:5]) if simple else
            tuple(colors[2:7]) if not rotating and rng.integers(0, 4) == 0
            else None}


def _pair_background(parameters: dict, section: str, index: int,
                     rng: np.random.Generator) -> int:
    if not parameters["rotating"]:
        return parameters["base"]
    if section == "train" and index < 2:
        return parameters["base"] if index == 0 else parameters["alternate"]
    return int(rng.choice((parameters["base"], parameters["alternate"])))


def _generate_pair(rng: np.random.Generator, section: str, index: int,
                   parameters: dict) -> tuple[np.ndarray, np.ndarray]:
    background = _pair_background(parameters, section, index, rng)
    for attempt in range(25):
        specs = _object_specs(rng, section, index, parameters["simple"])
        shape = _canvas_shape(rng, len(specs), section, index, attempt,
                              parameters["simple"])
        masks = [_make_mask(rng, h, w, mode) for mode, w, h in specs]
        positions = _place_masks(rng, shape, masks)
        if positions is None:
            continue
        available = (parameters["palette_bank"] if
                     parameters["palette_bank"] is not None else
                     tuple(color for color in range(10) if color != background))
        palette = [int(x) for x in rng.permutation(available)[:len(specs)]]
        if parameters["rotating"] and section == "train" and index == 0:
            if parameters["alternate"] in palette:
                other = palette.index(parameters["alternate"])
                palette[0], palette[other] = palette[other], palette[0]
            else:
                palette[0] = parameters["alternate"]
        if not parameters["simple"] and len(specs) >= 3 and rng.random() < .4:
            eligible = []
            for i in range(len(specs)):
                ri, ci = positions[i]
                hi, wi = masks[i].shape
                for j in range(i + 1, len(specs)):
                    rj, cj = positions[j]
                    hj, wj = masks[j].shape
                    if ri + hi < rj or rj + hj < ri or \
                       ci + wi < cj or cj + wj < ci:
                        eligible.append((i, j))
            if eligible:
                i, j = eligible[int(rng.integers(0, len(eligible)))]
                palette[j] = palette[i]
        grid = np.full(shape, background, dtype=np.int8)
        for mask, (row, col), color in zip(masks, positions, palette):
            view = grid[row:row + mask.shape[0], col:col + mask.shape[1]]
            view[mask] = color
        if len(_extract_objects(grid, background)) != len(specs):
            continue
        if section == "train" and index == 0:
            dense = bool(rng.integers(0, 2))
        else:
            dense = bool(rng.integers(0, 4) == 0)
        grid = _crop_canvas(grid, rng, dense)
        try:
            if _infer_background(grid) != background:
                continue
        except AssertionError:
            continue
        output = _derive_output(grid)
        if np.array_equal(grid, output) or _matches_source_pair(grid, output):
            continue
        return grid, output
    raise ValueError("bounded object placement failed")


def _array_key(array: np.ndarray) -> tuple:
    return (tuple(int(x) for x in array.shape), array.dtype.str,
            array.tobytes(order="C"))


def _matches_source_pair(input_grid: np.ndarray,
                         output_grid: np.ndarray) -> bool:
    return any(np.array_equal(input_grid, src_in) and
               np.array_equal(output_grid, src_out)
               for src_in, src_out in _SOURCE_PAIRS)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _normalize_seed(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    parameters = _sample_puzzle_parameters(rng)
    child_seeds = rng.integers(0, np.iinfo(np.int64).max,
                               size=num_train + num_test, dtype=np.int64)
    puzzle: dict = {"train": [], "test": []}
    seen: set[tuple] = set()
    offset = 0
    for section, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(child_seeds[offset]))
            offset += 1
            for _ in range(12):
                input_grid, output_grid = _generate_pair(
                    pair_rng, section, index, parameters)
                key = _array_key(input_grid)
                if key not in seen:
                    break
            else:
                raise ValueError("unable to produce a unique pair")
            seen.add(key)
            puzzle[section].append({"input": input_grid,
                                    "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= side <= 30 for side in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert bool(np.all((0 <= grid) & (grid <= 9)))


def _has_hole(mask: np.ndarray) -> bool:
    height, width = mask.shape
    empty = {(r, c) for r in range(height) for c in range(width)
             if not mask[r, c]}
    boundary = [(r, c) for r, c in empty
                if r in (0, height - 1) or c in (0, width - 1)]
    reachable = set(boundary)
    while boundary:
        r, c = boundary.pop()
        for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if point in empty and point not in reachable:
                reachable.add(point)
                boundary.append(point)
    return bool(empty - reachable)


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    assert input_grid.shape == output_grid.shape
    background = _infer_background(input_grid)
    objects = _extract_objects(input_grid, background)
    assert 2 <= len(objects) <= 5
    expected = input_grid.copy()
    widths = []
    rowwise = split = hole = second_column = third_column = False
    repeated = len({obj["color"] for obj in objects}) < len(objects)
    touches_edge = False
    for obj in objects:
        r0, r1, c0, c1 = obj["bbox"]
        coords = obj["coords"]
        width = c1 - c0 + 1
        widths.append(width)
        second_column |= bool(np.any(coords[:, 1] == c0 + 1))
        third_column |= bool(np.any(coords[:, 1] == c0 + 2))
        assert 1 <= width <= 7
        assert 1 <= r1 - r0 + 1 <= 7
        assert len(coords) >= 2
        touches_edge |= r0 == 0 or c0 == 0 or \
                        r1 == input_grid.shape[0] - 1 or \
                        c1 == input_grid.shape[1] - 1
        mask = np.zeros((r1 - r0 + 1, width), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        hole |= _has_hole(mask)
        for local_row in range(mask.shape[0]):
            columns = np.flatnonzero(mask[local_row])
            if len(columns):
                rowwise |= int(columns[0]) >= 2
                split |= bool(np.any(np.diff(columns) > 1))
        erased = coords[:, 1] < c0 + 2
        assert bool(np.any(erased))
        expected[coords[erased, 0], coords[erased, 1]] = background
        if width >= 3:
            assert bool(np.any(~erased))
    assert np.array_equal(expected, output_grid)
    assert not np.array_equal(input_grid, output_grid)
    assert bool(np.all(output_grid[input_grid != output_grid] == background))
    blank_rows = int(np.sum(np.all(input_grid == background, axis=1)))
    blank_cols = int(np.sum(np.all(input_grid == background, axis=0)))
    return {"background": background, "count": len(objects),
            "widths": tuple(widths), "rowwise": rowwise,
            "split": split, "hole": hole, "repeated": repeated,
            "second_column": second_column, "third_column": third_column,
            "edge": touches_edge, "blank_rows": blank_rows,
            "blank_cols": blank_cols}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp)
            _validate_grid(out)
            assert inp.shape == out.shape
            key = _array_key(inp)
            pair_key = (key, _array_key(out))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert not _matches_source_pair(inp, out)
            analyses[section].append(_analyze_pair(inp, out))
    training = analyses["train"]
    assert any(2 in a["widths"] for a in training)
    assert any(any(w >= 4 for w in a["widths"]) for a in training)
    assert any(a["rowwise"] for a in training)
    assert any(a["second_column"] and a["third_column"] for a in training)
    assert any(a["split"] and a["hole"] for a in training)
    assert all(any(w >= 4 for w in a["widths"]) for a in analyses["test"])
    backgrounds = {a["background"] for a in training}
    test_backgrounds = {a["background"] for a in analyses["test"]}
    assert test_backgrounds <= backgrounds


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[k], b[k])
                   for a, b in zip(first[s], second[s])
                   for k in ("input", "output"))
               for s in ("train", "test"))


def _layout_signature(grid: np.ndarray) -> tuple:
    background = _infer_background(grid)
    records = []
    for obj in _extract_objects(grid, background):
        r0, r1, c0, c1 = obj["bbox"]
        records.append((r0, c0, r1 - r0 + 1, c1 - c0 + 1,
                        tuple(sorted((int(r - r0), int(c - c0))
                                     for r, c in obj["coords"]))))
    return (grid.shape, tuple(sorted(records)))


def self_test() -> None:
    for seed, tr, te in ((137, None, None), (991, 7, 3),
                         (201, 2, 1), (202, 10, 1), (203, 10, 4),
                         (301, None, 1), (302, None, 4),
                         (303, 2, None), (304, 10, None)):
        first = generate_puzzle(seed, tr, te)
        second = generate_puzzle(seed, tr, te)
        assert _puzzles_equal(first, second)
        validate_generated_puzzle(first)
        if tr is not None:
            assert len(first["train"]) == tr
        if te is not None:
            assert len(first["test"]) == te
    for arguments in ({"num_train": 1}, {"num_train": 11},
                      {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4},
                      {"num_train": 2.0, "num_test": 1}):
        try:
            generate_puzzle(400, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    train_counts, test_counts, shapes, layouts = set(), set(), set(), set()
    backgrounds, counts, widths, palettes = set(), set(), set(), set()
    changed_bg_puzzles = repeated_puzzles = gutterless_puzzles = 0
    varied_count_puzzles = fixed_palette_puzzles = novel_test_puzzles = 0
    larger_test_puzzles = all_wide_test_puzzles = multiple_deletion_puzzles = 0
    split_pairs = hole_pairs = large_test_pairs = 0
    aspect_ratios = set()
    puzzle_keys = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_key = []
        pair_backgrounds = set()
        visible_palettes = set()
        training_colors: set[int] = set()
        test_colors: set[int] = set()
        pair_counts = set()
        has_repeat = has_gutterless = False
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                analysis = _analyze_pair(inp, out)
                assert np.array_equal(_derive_output(inp), out)
                assert not _matches_source_pair(inp, out)
                pair_backgrounds.add(analysis["background"])
                pair_counts.add(analysis["count"])
                foreground = {int(c) for c in np.unique(inp)
                              if int(c) != analysis["background"]}
                (training_colors if section == "train" else test_colors).update(
                    foreground)
                visible_palettes.add((analysis["background"],
                                      tuple(sorted(foreground))))
                backgrounds.add(analysis["background"])
                counts.add(analysis["count"])
                widths.update(analysis["widths"])
                shapes.add(inp.shape)
                aspect_ratios.add((inp.shape[0] > inp.shape[1]) -
                                  (inp.shape[0] < inp.shape[1]))
                layouts.add(_layout_signature(inp))
                palettes.add(tuple(int(c) for c in np.unique(inp)))
                split_pairs += analysis["split"]
                hole_pairs += analysis["hole"]
                has_repeat |= analysis["repeated"]
                has_gutterless |= (analysis["blank_rows"] == 0 and
                                   analysis["blank_cols"] == 0)
                if section == "test":
                    large_test_pairs += any(w >= 4 for w in analysis["widths"])
                puzzle_key.append((_array_key(inp), _array_key(out)))
        puzzle_keys.add(tuple(puzzle_key))
        changed_bg_puzzles += len(pair_backgrounds) > 1
        repeated_puzzles += has_repeat
        gutterless_puzzles += has_gutterless
        varied_count_puzzles += len(pair_counts) > 1
        fixed_palette_puzzles += len(visible_palettes) == 1
        novel_test_puzzles += bool(test_colors - training_colors)
        larger_test_puzzles += max(p["input"].size for p in puzzle["test"]) > \
                               max(p["input"].size for p in puzzle["train"])
        test_analyses = [_analyze_pair(p["input"], p["output"])
                         for p in puzzle["test"]]
        all_wide_test_puzzles += any(all(w > 2 for w in a["widths"])
                                     for a in test_analyses)
        multiple_deletion_puzzles += any(sum(w <= 2 for w in a["widths"]) >= 2
                                         for a in test_analyses)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_keys) >= 40 and len(layouts) >= 200
    assert len(shapes) >= 35 and len(backgrounds) >= 7
    assert aspect_ratios == {-1, 0, 1}
    assert counts == {2, 3, 4, 5}
    assert {1, 2, 3, 4, 5, 6} <= widths
    assert len(palettes) >= 100
    assert changed_bg_puzzles >= 10
    assert varied_count_puzzles >= 25
    assert fixed_palette_puzzles >= 2
    assert novel_test_puzzles >= 5
    assert larger_test_puzzles >= 5
    assert all_wide_test_puzzles >= 10
    assert multiple_deletion_puzzles >= 5
    assert repeated_puzzles >= 8
    assert gutterless_puzzles >= 4
    assert split_pairs >= 40 and hole_pairs >= 35
    assert large_test_pairs >= 40
