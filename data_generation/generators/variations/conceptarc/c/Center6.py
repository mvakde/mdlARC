import hashlib

import numpy as np


_SOURCE_PAIR_HASHES = frozenset({
    "73919a797649cca621dd5207352be54206febe9c72bdaf8b70f3bdf8a907779c",
    "f47796ea9e162e8af036ffc99e3b57536c486b226a75af965e656f0a0d214b61",
    "16bea447ef1b22ba4893cf2f564c53b2e9e626cf9c4c97c10c14e162465766e2",
    "4adcdbe733b62e0de1fd0acdd767ad9f43c79c324f0d5b3a8c4805c2a8b839e9",
    "da83f8993623f61b6fd6b58536e0e72e15d94375b78d4c0d6327a14ad5b4e82f",
    "d1915a703be8dc0fcfa44bfe8bc9b17fd23609d72a01f0304bd8680d37b3fba2",
})


def _resolve_counts(rng, num_train, num_test):
    for value, low, high, name in (
        (num_train, 2, 10, "num_train"),
        (num_test, 1, 4, "num_test"),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
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


def _choices(low, high, parity):
    return [n for n in range(low, high + 1) if n % 2 == parity]


def _size(rng, values):
    assert values
    if len(values) > 2 and rng.random() < 0.32:
        values = values[len(values) * 2 // 3:]
    return int(rng.choice(values))


def _canvas_dimension(rng):
    band = int(rng.choice(3, p=(0.40, 0.34, 0.26)))
    low, high = ((10, 15), (16, 22), (23, 30))[band]
    return int(rng.integers(low, high + 1))


def _shape_mask(kind, height, width):
    if kind == "filled_rect":
        return np.ones((height, width), dtype=bool)
    if kind == "hollow_rect":
        mask = np.zeros((height, width), dtype=bool)
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        return mask
    rows, cols = np.indices((height, width))
    if kind == "cross":
        return (rows == height // 2) | (cols == width // 2)
    distance = abs(rows - height // 2) + abs(cols - width // 2)
    if kind == "filled_diamond":
        return distance <= height // 2
    if kind == "outline_diamond":
        return distance == height // 2
    raise ValueError("unknown shape")


def _shape_dimensions(rng, kind, marker_size, canvas_height, canvas_width):
    if marker_size == 1:
        if "diamond" in kind:
            side = _size(rng, _choices(5, min(canvas_height, canvas_width), 1))
            return side, side
        return (
            _size(rng, _choices(5, canvas_height, 1)),
            _size(rng, _choices(5, canvas_width, 1)),
        )
    if kind == "hollow_rect":
        return (
            _size(rng, _choices(4, canvas_height, 0)),
            _size(rng, _choices(4, canvas_width, 0)),
        )
    orientation = int(rng.choice(3, p=(0.30, 0.30, 0.40)))
    if orientation == 0:
        return 2, _size(rng, _choices(6, canvas_width, 0))
    if orientation == 1:
        return _size(rng, _choices(6, canvas_height, 0)), 2
    return (
        _size(rng, _choices(4, canvas_height, 0)),
        _size(rng, _choices(4, canvas_width, 0)),
    )


def _origin(rng, canvas, extent):
    last = canvas - extent
    if last == 0:
        return 0
    if rng.random() < 0.35:
        return int(rng.choice((0, last)))
    return int(rng.integers(last + 1))


def _marker_locations(occupied, kind, box, marker_size):
    height, width = occupied.shape
    shape_row, shape_col, shape_height, shape_width = box
    close, far = [], []
    for row in range(height - marker_size + 1):
        for col in range(width - marker_size + 1):
            if kind == "outline_diamond":
                # Its hollow interior is part of the large object. Only the
                # exterior corner gaps of its box are detached locations.
                inside = False
                for rr in range(row, row + marker_size):
                    for cc in range(col, col + marker_size):
                        if (
                            shape_row <= rr < shape_row + shape_height
                            and shape_col <= cc < shape_col + shape_width
                            and abs(rr - (shape_row + shape_height // 2))
                            + abs(cc - (shape_col + shape_width // 2))
                            < shape_height // 2
                        ):
                            inside = True
                if inside:
                    continue
            if kind not in ("outline_diamond", "filled_diamond") and (
                row < shape_row + shape_height
                and row + marker_size > shape_row
                and col < shape_col + shape_width
                and col + marker_size > shape_col
            ):
                continue
            if np.any(occupied[row:row + marker_size, col:col + marker_size]):
                continue
            # Different colors alone do not make touching cells separate objects.
            if (
                (row > 0 and np.any(occupied[row - 1, col:col + marker_size]))
                or (row + marker_size < height and np.any(
                    occupied[row + marker_size, col:col + marker_size]
                ))
                or (col > 0 and np.any(occupied[row:row + marker_size, col - 1]))
                or (col + marker_size < width and np.any(
                    occupied[row:row + marker_size, col + marker_size]
                ))
            ):
                continue
            row_gap = max(
                shape_row - (row + marker_size),
                row - (shape_row + shape_height),
                0,
            )
            col_gap = max(
                shape_col - (col + marker_size),
                col - (shape_col + shape_width),
                0,
            )
            (close if max(row_gap, col_gap) <= 1 else far).append((row, col))
    return close, far


def _sample_pair(rng, kind, marker_size, background, shape_color, marker_color):
    for attempt in range(48):
        canvas_height = _canvas_dimension(rng)
        canvas_width = _canvas_dimension(rng)
        if attempt == 47:
            canvas_height = canvas_width = 13
        shape_height, shape_width = _shape_dimensions(
            rng, kind, marker_size, canvas_height, canvas_width
        )
        if attempt == 47:
            if marker_size == 1:
                shape_height = shape_width = 5
            elif kind == "hollow_rect":
                shape_height = shape_width = 4
            else:
                shape_height = 2
                shape_width = 6
        shape_row = _origin(rng, canvas_height, shape_height)
        shape_col = _origin(rng, canvas_width, shape_width)
        if attempt == 47:
            shape_row = shape_col = 3
        mask = _shape_mask(kind, shape_height, shape_width)
        occupied = np.zeros((canvas_height, canvas_width), dtype=bool)
        occupied[
            shape_row:shape_row + shape_height,
            shape_col:shape_col + shape_width,
        ] = mask
        background_count = canvas_height * canvas_width - int(mask.sum()) - marker_size ** 2
        if background_count <= 4:
            continue
        box = (shape_row, shape_col, shape_height, shape_width)
        close, far = _marker_locations(occupied, kind, box, marker_size)
        if not close and not far:
            continue
        if close and far:
            locations = close if rng.random() < 0.48 else far
        else:
            locations = close or far
        marker_row, marker_col = locations[int(rng.integers(len(locations)))]
        input_grid = np.full(
            (canvas_height, canvas_width), background, dtype=np.int8
        )
        input_grid[occupied] = shape_color
        input_grid[
            marker_row:marker_row + marker_size,
            marker_col:marker_col + marker_size,
        ] = marker_color
        output_grid = input_grid.copy()
        output_grid[
            marker_row:marker_row + marker_size,
            marker_col:marker_col + marker_size,
        ] = background
        target_row = shape_row + (shape_height - marker_size) // 2
        target_col = shape_col + (shape_width - marker_size) // 2
        output_grid[
            target_row:target_row + marker_size,
            target_col:target_col + marker_size,
        ] = marker_color
        return {"input": input_grid, "output": output_grid}
    raise RuntimeError("no legal shape and detached marker placement")


def _grid_key(grid):
    return tuple(grid.shape), grid.tobytes()


def _pair_hash(pair):
    digest = hashlib.sha256()
    for field in ("input", "output"):
        grid = np.asarray(pair[field], dtype=np.uint8)
        digest.update(bytes(grid.shape))
        digest.update(grid.tobytes())
    return digest.hexdigest()


def _recognize_shape(mask, marker_size):
    height, width = mask.shape
    if (height - marker_size) % 2 or (width - marker_size) % 2:
        return None
    if marker_size == 1:
        if height < 5 or width < 5:
            return None
        kinds = (
            "filled_rect", "hollow_rect", "outline_diamond",
            "filled_diamond", "cross",
        )
    else:
        if min(height, width) < 2 or max(height, width) < 6:
            if min(height, width) < 4:
                return None
        kinds = ("filled_rect", "hollow_rect")
    for kind in kinds:
        if "diamond" in kind and height != width:
            continue
        if kind == "hollow_rect" and min(height, width) < 4:
            continue
        if np.array_equal(mask, _shape_mask(kind, height, width)):
            return kind
    return None


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    input_grid, output_grid = pair["input"], pair["output"]
    for grid in (input_grid, output_grid):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert all(1 <= n <= 30 for n in grid.shape)
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((grid >= 0) & (grid <= 9))
    assert input_grid.shape == output_grid.shape
    colors, counts = np.unique(input_grid, return_counts=True)
    assert len(colors) == 3
    small = [int(color) for color, count in zip(colors, counts) if count in (1, 4)]
    assert len(small) == 1
    marker_color = small[0]
    marker_coords = np.argwhere(input_grid == marker_color)
    marker_size = 1 if len(marker_coords) == 1 else 2
    marker_row, marker_col = (int(v) for v in marker_coords.min(axis=0))
    assert np.array_equal(
        marker_coords.max(axis=0) - marker_coords.min(axis=0),
        [marker_size - 1, marker_size - 1],
    )
    assert np.all(
        input_grid[
            marker_row:marker_row + marker_size,
            marker_col:marker_col + marker_size,
        ] == marker_color
    )
    candidates = []
    for color in colors:
        color = int(color)
        if color == marker_color:
            continue
        coords = np.argwhere(input_grid == color)
        row, col = (int(v) for v in coords.min(axis=0))
        last_row, last_col = (int(v) for v in coords.max(axis=0))
        height, width = last_row - row + 1, last_col - col + 1
        kind = _recognize_shape(
            input_grid[row:last_row + 1, col:last_col + 1] == color,
            marker_size,
        )
        if kind is not None and len(coords) > 4:
            candidates.append((color, row, col, height, width, kind))
    assert len(candidates) == 1
    shape_color, row, col, height, width, kind = candidates[0]
    background = next(
        int(c) for c in colors if c not in (shape_color, marker_color)
    )
    assert np.count_nonzero(input_grid == background) > 4
    occupied = input_grid == shape_color
    close, far = _marker_locations(
        occupied, kind, (row, col, height, width), marker_size
    )
    assert (marker_row, marker_col) in close or (marker_row, marker_col) in far
    target_row = row + (height - marker_size) // 2
    target_col = col + (width - marker_size) // 2
    center_before = input_grid[
        target_row:target_row + marker_size,
        target_col:target_col + marker_size,
    ]
    assert np.all(center_before == center_before[0, 0])
    assert center_before[0, 0] in (background, shape_color)
    expected = input_grid.copy()
    expected[
        marker_row:marker_row + marker_size,
        marker_col:marker_col + marker_size,
    ] = background
    expected[
        target_row:target_row + marker_size,
        target_col:target_col + marker_size,
    ] = marker_color
    assert np.array_equal(output_grid, expected)
    assert not np.array_equal(input_grid, output_grid)
    assert _pair_hash(pair) not in _SOURCE_PAIR_HASHES
    return {
        "kind": kind,
        "marker_size": marker_size,
        "background": background,
        "shape_color": shape_color,
        "marker_color": marker_color,
        "shape_box": (row, col, height, width),
        "marker_pos": (marker_row, marker_col),
        "center_under": int(center_before[0, 0]),
        "close": (marker_row, marker_col) in close,
        "edge": row == 0 or col == 0
        or row + height == input_grid.shape[0]
        or col + width == input_grid.shape[1],
        "grid_shape": input_grid.shape,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    num_train, num_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= num_train <= 10 and 1 <= num_test <= 4
    assert num_train > num_test
    seen_inputs, seen_pairs = set(), set()
    metadata = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            meta = _inspect_pair(pair)
            metadata[split].append(meta)
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
    train = metadata["train"]
    assert all(
        m["marker_size"] == 1
        or (
            m["marker_size"] == 2
            and m["kind"] in ("filled_rect", "hollow_rect")
        )
        for m in train + metadata["test"]
    )
    assert {m["center_under"] == m["background"] for m in train} == {
        True, False
    }
    assert train[0]["shape_color"] == train[1]["marker_color"]
    assert train[0]["marker_color"] == train[1]["shape_color"]
    # All color assignments are visible within their own input. The shared
    # convention, established by training, is selection by object geometry.


def _kind(rng, center_background, marker_size):
    if center_background:
        if marker_size == 2:
            return "hollow_rect"
        return str(rng.choice(("outline_diamond", "hollow_rect")))
    if marker_size == 2:
        return "filled_rect"
    return str(rng.choice(("filled_rect", "filled_diamond", "cross")))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    # The transformation is always the same. These puzzle-level choices set
    # only the distribution of visible examples, never a hidden output rule.
    footprint_mode = str(rng.choice(("source_like", "taught", "single")))
    palette_mode = str(rng.choice(("fixed", "variable"), p=(0.36, 0.64)))
    background_mode = str(rng.choice(
        ("zero", "constant_nonzero", "variable"), p=(0.46, 0.22, 0.32)
    ))
    fixed_background = (
        0 if background_mode == "zero"
        else int(rng.integers(1, 10)) if background_mode == "constant_nonzero"
        else None
    )
    available = [
        color for color in range(1, 10)
        if color != fixed_background
    ]
    first_colors = tuple(int(v) for v in rng.choice(available, size=2, replace=False))
    center_modes = [True, False]
    rng.shuffle(center_modes)
    taught_index = int(rng.integers(2))
    source_test_index = int(rng.integers(num_test))
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train":
                center_background = (
                    center_modes[index] if index < 2 else bool(rng.integers(2))
                )
                marker_size = (
                    2 if footprint_mode == "taught"
                    and (index == taught_index or (index >= 2 and rng.random() < 0.35))
                    else 1
                )
            else:
                center_background = bool(rng.integers(2))
                marker_size = (
                    1 if footprint_mode == "single"
                    else 2 if footprint_mode == "source_like"
                    and index == source_test_index
                    else 2 if rng.random() < 0.45 else 1
                )
            kind = _kind(rng, center_background, marker_size)
            if split == "train" and index < 2:
                shape_color, marker_color = (
                    first_colors if index == 0 else first_colors[::-1]
                )
            elif palette_mode == "fixed":
                shape_color, marker_color = (
                    first_colors if rng.random() < 0.5 else first_colors[::-1]
                )
            else:
                available_colors = [
                    color for color in range(10)
                    if color != fixed_background
                ]
                shape_color, marker_color = (
                    int(v) for v in rng.choice(
                        available_colors, size=2, replace=False
                    )
                )
            if background_mode == "variable":
                background = int(rng.choice([
                    color for color in range(10)
                    if color not in (shape_color, marker_color)
                ]))
            else:
                background = fixed_background
            for _ in range(8):
                child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair = _sample_pair(
                    np.random.default_rng(child_seed),
                    kind, marker_size, background, shape_color, marker_color,
                )
                key = _grid_key(pair["input"])
                if key not in seen_inputs and _pair_hash(pair) not in _SOURCE_PAIR_HASHES:
                    break
            else:
                raise RuntimeError("could not construct a distinct pair")
            seen_inputs.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _same_puzzles(first, second):
    return all(
        len(first[split]) == len(second[split])
        and all(
            np.array_equal(p[field], q[field])
            for p, q in zip(first[split], second[split])
            for field in ("input", "output")
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts = set(), set()
    palettes, dimensions, layouts, signatures = set(), set(), set(), set()
    kinds, marker_sizes, backgrounds, shape_sizes = set(), set(), set(), set()
    close_count = edge_count = train_large = test_large = 0
    single_only = variable_background = variable_palette = 0
    even_geometries = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzles(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train = [_inspect_pair(pair) for pair in puzzle["train"]]
        test = [_inspect_pair(pair) for pair in puzzle["test"]]
        all_meta = train + test
        if all(m["marker_size"] == 1 for m in all_meta):
            single_only += 1
        if len({m["background"] for m in all_meta}) > 1:
            variable_background += 1
        if len({
            (m["shape_color"], m["marker_color"]) for m in all_meta
        }) > 2:
            variable_palette += 1
        train_large += sum(m["marker_size"] == 2 for m in train)
        test_large += sum(m["marker_size"] == 2 for m in test)
        for meta in all_meta:
            kinds.add(meta["kind"])
            marker_sizes.add(meta["marker_size"])
            backgrounds.add(meta["background"])
            palettes.add((
                meta["background"], meta["shape_color"], meta["marker_color"]
            ))
            dimensions.add(meta["grid_shape"])
            shape_sizes.add(meta["shape_box"][2:])
            layouts.add((
                meta["grid_shape"], meta["shape_box"], meta["marker_pos"]
            ))
            close_count += meta["close"]
            edge_count += meta["edge"]
            if meta["marker_size"] == 2:
                h, w = meta["shape_box"][2:]
                even_geometries.add(
                    "horizontal" if h == 2 else
                    "vertical" if w == 2 else
                    "hollow" if meta["kind"] == "hollow_rect" else "thick"
                )
        signatures.add(tuple(
            _grid_key(pair["input"])
            for split in ("train", "test") for pair in puzzle[split]
        ))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palettes) >= 30 and len(backgrounds) >= 4
    assert len(dimensions) >= 40 and len(layouts) >= 150
    assert len(shape_sizes) >= 35 and len(signatures) == 42
    assert len(kinds) == 5 and marker_sizes == {1, 2}
    assert close_count >= 30 and edge_count >= 30
    assert train_large >= 5 and test_large >= 15
    assert single_only >= 3 and variable_background >= 4
    assert variable_palette >= 4
    assert even_geometries == {"horizontal", "vertical", "thick", "hollow"}
    for index, (num_train, num_test) in enumerate((
        (2, 1), (10, 1), (10, 4),
        (None, 1), (None, 4), (2, None), (10, None),
    )):
        seed = 100 + index
        puzzle = generate_puzzle(seed, num_train, num_test)
        validate_generated_puzzle(puzzle)
        assert _same_puzzles(
            puzzle, generate_puzzle(seed, num_train, num_test)
        )
        assert num_train is None or len(puzzle["train"]) == num_train
        assert num_test is None or len(puzzle["test"]) == num_test
    for num_train, num_test in (
        (1, None), (11, None), (None, 0), (None, 5),
        (2, 2), (3, 4), (True, 1), (2, 1.5),
    ):
        try:
            generate_puzzle(5, num_train, num_test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
