import hashlib
from itertools import combinations

import numpy as np


_SOURCE_HASHES = {
    "dcd3d1f70f31fd663bd60cee36c6376612569fcfdd622a5785939c7115900b93",
    "7b51bcc82c34876f695882c2b2cae9f928b13b9fe481d1d3813d5f395dc8daa2",
    "d2fcaa8b6bf505804a505661804b447d96aa210b0343a92423d2da5bd911f877",
    "7fa853de68000f3725aebf36c4deee9c9ae98f9b67acc07befbcc50578be027a",
}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _resolve_counts(seed, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return int(num_train), int(num_test)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        minimum = max(2, num_test + 1)
        num_train = minimum + seed % (11 - minimum)
    else:
        num_test = 1 + seed % min(4, num_train - 1)
    return int(num_train), int(num_test)


def _axis_length(tile_size, gutter, count):
    return count * tile_size + (count - 1) * gutter


def _axis_sample(rng, tile_size, gutter, count):
    core = _axis_length(tile_size, gutter, count)
    options = []
    for leading in range(gutter + 1):
        for trailing in range(min(gutter + tile_size - 1, 30 - core - leading) + 1):
            options.append((leading, trailing))
    for prefix in range(1, tile_size):
        origin = prefix + gutter
        for trailing in range(min(gutter + tile_size - 1, 30 - core - origin) + 1):
            options.append((origin, trailing))
    if not options:
        raise ValueError("axis does not fit")
    origin, trailing = options[int(rng.integers(len(options)))]
    return core + origin + trailing, origin


def _axis_labels(length, origin, tile_size, gutter):
    relative = np.arange(length) - origin
    local = relative % (tile_size + gutter)
    active = local < tile_size
    edge = (local == 0) | (local == tile_size - 1)
    return active, edge


def _render_input(shape, origins, tile_size, gutters, colors, rectangles):
    height, width = shape
    row_origin, col_origin = origins
    row_gutter, col_gutter = gutters
    background, frame, default = colors[:3]
    row_active, row_edge = _axis_labels(height, row_origin, tile_size, row_gutter)
    col_active, col_edge = _axis_labels(width, col_origin, tile_size, col_gutter)
    tiles = row_active[:, None] & col_active[None, :]
    frames = tiles & (row_edge[:, None] | col_edge[None, :])
    grid = np.full(shape, background, dtype=np.int8)
    grid[tiles] = default
    grid[frames] = frame
    for color, (rows, cols) in rectangles.items():
        for tile_row in rows:
            for tile_col in cols:
                row_start = row_origin + tile_row * (tile_size + row_gutter)
                col_start = col_origin + tile_col * (tile_size + col_gutter)
                grid[
                    row_start + 1 : row_start + tile_size - 1,
                    col_start + 1 : col_start + tile_size - 1,
                ] = color
    return grid


def _paint_connectors(input_grid, origins, tile_size, gutters, background, rectangles):
    output = input_grid.copy()
    row_origin, col_origin = origins
    row_gutter, col_gutter = gutters
    row_pitch = tile_size + row_gutter
    col_pitch = tile_size + col_gutter
    for color, (rows, cols) in rectangles.items():
        for tile_row in rows:
            row_start = row_origin + tile_row * row_pitch
            for gap_col in range(cols[0], cols[1]):
                col_start = col_origin + gap_col * col_pitch + tile_size
                block = output[
                    row_start + 1 : row_start + tile_size - 1,
                    col_start : col_start + col_gutter,
                ]
                if np.any((block != background) & (block != color)):
                    raise ValueError("incompatible rectangle sides")
                block[:] = color
        for tile_col in cols:
            col_start = col_origin + tile_col * col_pitch
            for gap_row in range(rows[0], rows[1]):
                row_start = row_origin + gap_row * row_pitch + tile_size
                block = output[
                    row_start : row_start + row_gutter,
                    col_start + 1 : col_start + tile_size - 1,
                ]
                if np.any((block != background) & (block != color)):
                    raise ValueError("incompatible rectangle sides")
                block[:] = color
    return output


def _rectangle_parts(rectangle):
    rows, cols = rectangle
    corners = {(row, col) for row in rows for col in cols}
    sides = {
        ("column", row, col)
        for row in rows
        for col in range(cols[0], cols[1])
    }
    sides.update(
        ("row", row, col)
        for col in cols
        for row in range(rows[0], rows[1])
    )
    return corners, sides


def _sample_rectangles(rng, nrows, ncols, colors):
    candidates = [
        (rows, cols)
        for rows in combinations(range(nrows), 2)
        for cols in combinations(range(ncols), 2)
    ]
    parts = [_rectangle_parts(rectangle) for rectangle in candidates]
    order = rng.permutation(len(candidates))
    for _ in range(32):
        chosen = []
        used_corners = set()
        used_sides = set()
        for index in order:
            corners, sides = parts[int(index)]
            if corners & used_corners or sides & used_sides:
                continue
            chosen.append(candidates[int(index)])
            used_corners.update(corners)
            used_sides.update(sides)
            if len(chosen) == len(colors):
                return dict(zip(colors, chosen))
        order = rng.permutation(len(candidates))
    raise ValueError("could not place compatible rectangles")


def _sample_geometry(rng, regime, pair_index, first_tile_size, accent_count, scale_shift):
    if regime == "source_train":
        sizes = [3, 4] if scale_shift else [3, 4, 5]
        if pair_index == 1:
            sizes.remove(first_tile_size)
        tile_size = int(rng.choice(sizes))
    elif scale_shift and pair_index == 0:
        tile_size = 5
    else:
        tile_size = int(rng.choice([3, 4, 5, 5]))
    options = []
    for row_gutter in (1, 2, 3):
        for col_gutter in (1, 2, 3):
            maximum_rows = (30 + row_gutter) // (tile_size + row_gutter)
            maximum_cols = (30 + col_gutter) // (tile_size + col_gutter)
            for nrows in range(3, maximum_rows + 1):
                for ncols in range(3, maximum_cols + 1):
                    if nrows * ncols >= 4 * accent_count + 5:
                        options.append((row_gutter, col_gutter, nrows, ncols))
    if not options:
        raise ValueError("no fitting geometry")
    row_gutter, col_gutter, nrows, ncols = options[int(rng.integers(len(options)))]
    height, row_origin = _axis_sample(rng, tile_size, row_gutter, nrows)
    width, col_origin = _axis_sample(rng, tile_size, col_gutter, ncols)
    return (height, width), (row_origin, col_origin), tile_size, (row_gutter, col_gutter), nrows, ncols


def _pair_hash(input_grid, output_grid):
    content = str(input_grid.shape).encode() + input_grid.astype(np.int8).tobytes()
    content += output_grid.astype(np.int8).tobytes()
    return hashlib.sha256(content).hexdigest()


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(int(seed), num_train, num_test)
    first_palette = [int(color) for color in rng.permutation(10)[:6]]
    scale_shift = bool(rng.integers(4) == 0)
    puzzle = {"train": [], "test": []}
    seen = set()
    train_accents = set()
    first_tile_size = None
    for split, count in (("train", train_count), ("test", test_count)):
        for pair_index in range(count):
            if split == "train" and pair_index == 0:
                accent_count = 2
            elif split == "train" and pair_index == 1:
                accent_count = 1
            else:
                accent_count = int(rng.choice([1, 2, 2, 3]))
            for attempt in range(120):
                pair_rng = np.random.default_rng(
                    int(rng.integers(0, np.iinfo(np.int64).max))
                )
                if split == "train" and pair_index == 0:
                    palette = first_palette[:]
                elif split == "train" and pair_index == 1:
                    palette = first_palette[:]
                    palette[0], palette[2] = palette[2], palette[0]
                    palette[1], palette[3] = palette[3], palette[1]
                else:
                    palette = [int(color) for color in pair_rng.permutation(10)[:6]]
                if split == "test" and pair_index == 0:
                    novel = [color for color in range(10) if color not in train_accents and color not in palette[:3]]
                    if novel:
                        novel_color = int(pair_rng.choice(novel))
                        if novel_color in palette[4:]:
                            novel_index = palette.index(novel_color)
                            palette[3], palette[novel_index] = palette[novel_index], palette[3]
                        else:
                            palette[3] = novel_color
                if len(set(palette[:3 + accent_count])) != 3 + accent_count:
                    continue
                try:
                    regime = "source_train" if split == "train" else "target_test"
                    geometry = _sample_geometry(pair_rng, regime, pair_index, first_tile_size, accent_count, scale_shift)
                    shape, origins, tile_size, gutters, nrows, ncols = geometry
                    rectangles = _sample_rectangles(pair_rng, nrows, ncols, palette[3 : 3 + accent_count])
                except ValueError:
                    continue
                input_grid = _render_input(shape, origins, tile_size, gutters, palette, rectangles)
                output_grid = _paint_connectors(input_grid, origins, tile_size, gutters, palette[0], rectangles)
                if _grid_key(input_grid) in seen or _pair_hash(input_grid, output_grid) in _SOURCE_HASHES:
                    continue
                seen.add(_grid_key(input_grid))
                puzzle[split].append({"input": input_grid, "output": output_grid})
                if split == "train":
                    train_accents.update(rectangles)
                    if pair_index == 0:
                        first_tile_size = tile_size
                break
            else:
                raise ValueError("could not construct unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _explain_input(input_grid, background=None):
    height, width = input_grid.shape
    explanations = []
    backgrounds = [background] if background is not None else [int(color) for color in np.unique(input_grid)]
    for background_color in backgrounds:
        for tile_size in (3, 4, 5):
            for row_gutter in (1, 2, 3):
                for col_gutter in (1, 2, 3):
                    for row_origin in range(tile_size + row_gutter):
                        row_starts = list(range(row_origin, height - tile_size + 1, tile_size + row_gutter))
                        if len(row_starts) < 3:
                            continue
                        row_active, row_edge = _axis_labels(height, row_origin, tile_size, row_gutter)
                        for col_origin in range(tile_size + col_gutter):
                            col_starts = list(range(col_origin, width - tile_size + 1, tile_size + col_gutter))
                            if len(col_starts) < 3:
                                continue
                            col_active, col_edge = _axis_labels(width, col_origin, tile_size, col_gutter)
                            tiles = row_active[:, None] & col_active[None, :]
                            if not np.all(input_grid[~tiles] == background_color):
                                continue
                            frames = tiles & (row_edge[:, None] | col_edge[None, :])
                            frame_values = np.unique(input_grid[frames])
                            if len(frame_values) != 1:
                                continue
                            frame = int(frame_values[0])
                            if frame == background_color:
                                continue
                            tile_colors = {}
                            valid = True
                            for tile_row, row_start in enumerate(row_starts):
                                for tile_col, col_start in enumerate(col_starts):
                                    block = input_grid[
                                        row_start + 1 : row_start + tile_size - 1,
                                        col_start + 1 : col_start + tile_size - 1,
                                    ]
                                    if not np.all(block == block[0, 0]):
                                        valid = False
                                        break
                                    tile_colors[tile_row, tile_col] = int(block[0, 0])
                                if not valid:
                                    break
                            if not valid:
                                continue
                            counts = {color: list(tile_colors.values()).count(color) for color in set(tile_colors.values())}
                            defaults = [color for color, count in counts.items() if count >= 5]
                            if len(defaults) != 1:
                                continue
                            default = defaults[0]
                            if default in (background_color, frame):
                                continue
                            accents = [color for color in counts if color != default]
                            if not 1 <= len(accents) <= 3 or any(counts[color] != 4 for color in accents):
                                continue
                            if len({background_color, frame, default, *accents}) != 3 + len(accents):
                                continue
                            rectangles = {}
                            used_sides = set()
                            for color in accents:
                                positions = {position for position, value in tile_colors.items() if value == color}
                                rows = tuple(sorted({row for row, _ in positions}))
                                cols = tuple(sorted({col for _, col in positions}))
                                if len(rows) != 2 or len(cols) != 2 or positions != {(row, col) for row in rows for col in cols}:
                                    valid = False
                                    break
                                _, sides = _rectangle_parts((rows, cols))
                                if sides & used_sides:
                                    valid = False
                                    break
                                used_sides.update(sides)
                                rectangles[color] = (rows, cols)
                            if not valid:
                                continue
                            shape = (height, width)
                            origins = (row_origin, col_origin)
                            gutters = (row_gutter, col_gutter)
                            expected_input = _render_input(shape, origins, tile_size, gutters, (background_color, frame, default), rectangles)
                            if not np.array_equal(expected_input, input_grid):
                                continue
                            output_grid = _paint_connectors(input_grid, origins, tile_size, gutters, background_color, rectangles)
                            explanations.append({
                                "output": output_grid,
                                "tile_size": tile_size,
                                "gutters": gutters,
                                "nrows": len(row_starts),
                                "ncols": len(col_starts),
                                "origins": origins,
                                "row_tail": height - row_starts[-1] - tile_size,
                                "col_tail": width - col_starts[-1] - tile_size,
                                "background": background_color,
                                "frame": frame,
                                "default": default,
                                "rectangles": rectangles,
                            })
    return explanations


def _validate_grid(grid, label):
    _require(isinstance(grid, np.ndarray), f"{label} is not an ndarray")
    _require(grid.ndim == 2, f"{label} is not two-dimensional")
    _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, f"{label} has invalid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} has noninteger dtype")
    _require(np.all((grid >= 0) & (grid <= 9)), f"{label} has illegal colors")


def _validate_and_describe(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "invalid pair counts")
    descriptions = {"train": [], "test": []}
    seen_inputs = set()
    for split in ("train", "test"):
        for pair_index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            input_grid, output_grid = pair["input"], pair["output"]
            _validate_grid(input_grid, f"{split}[{pair_index}].input")
            _validate_grid(output_grid, f"{split}[{pair_index}].output")
            _require(input_grid.shape == output_grid.shape, "dimensions changed")
            _require(not np.array_equal(input_grid, output_grid), "unchanged pair")
            key = _grid_key(input_grid)
            _require(key not in seen_inputs, "duplicate input")
            seen_inputs.add(key)
            _require(_pair_hash(input_grid, output_grid) not in _SOURCE_HASHES, "source pair copied")
            changed = input_grid != output_grid
            changed_colors = np.unique(input_grid[changed])
            _require(len(changed_colors) == 1, "changed cells have multiple backgrounds")
            candidates = _explain_input(input_grid, int(changed_colors[0]))
            _require(candidates, f"{split}[{pair_index}] has no valid lattice")
            _require(all(np.array_equal(candidate["output"], output_grid) for candidate in candidates), "ambiguous or incorrect output")
            descriptions[split].append(candidates[0])
    train_accents = [len(descriptor["rectangles"]) for descriptor in descriptions["train"]]
    _require(1 in train_accents and 2 in train_accents, "training does not demonstrate one and two rectangles")
    _require(descriptions["train"][0]["tile_size"] != descriptions["train"][1]["tile_size"], "training lacks scale variation")
    _require(descriptions["train"][1]["frame"] in descriptions["train"][0]["rectangles"], "training lacks a color-role swap")
    _require(all(descriptions["train"][0][role] != descriptions["train"][1][role] for role in ("background", "frame", "default")), "training color roles did not vary")
    return descriptions


def validate_generated_puzzle(puzzle: dict) -> None:
    _validate_and_describe(puzzle)


def _puzzles_equal(first, second):
    return all(
        len(first[split]) == len(second[split])
        and all(
            np.array_equal(pair_a["input"], pair_b["input"])
            and np.array_equal(pair_a["output"], pair_b["output"])
            for pair_a, pair_b in zip(first[split], second[split])
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    train_counts = set()
    test_counts = set()
    shapes = set()
    sizes = {"train": set(), "test": set()}
    gutters = {"train": set(), "test": set()}
    accent_counts = {"train": set(), "test": set()}
    layouts = {"train": set(), "test": set()}
    backgrounds = set()
    partial_leading = 0
    partial_trailing = 0
    uncropped_tests = 0
    unequal_gutters = 0
    adjacent = 0
    long_span = 0
    shared_axis = 0
    novel_test_accents = 0
    all_inputs = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        descriptions = _validate_and_describe(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        training_accents = set().union(*(set(item["rectangles"]) for item in descriptions["train"]))
        for split in ("train", "test"):
            for pair, descriptor in zip(puzzle[split], descriptions[split]):
                prediction = _explain_input(pair["input"])
                _require(prediction and all(np.array_equal(item["output"], pair["output"]) for item in prediction), "input-only prediction failed")
                all_inputs.add(_grid_key(pair["input"]))
                shapes.add(pair["input"].shape)
                sizes[split].add(descriptor["tile_size"])
                gutters[split].add(descriptor["gutters"])
                accent_counts[split].add(len(descriptor["rectangles"]))
                backgrounds.add(descriptor["background"])
                layout = tuple(sorted((tuple(rows), tuple(cols)) for rows, cols in descriptor["rectangles"].values()))
                layouts[split].add((descriptor["nrows"], descriptor["ncols"], layout))
                row_origin, col_origin = descriptor["origins"]
                row_gutter, col_gutter = descriptor["gutters"]
                partial_leading += row_origin > row_gutter or col_origin > col_gutter
                partial_trailing += descriptor["row_tail"] > row_gutter or descriptor["col_tail"] > col_gutter
                unequal_gutters += row_gutter != col_gutter
                if split == "test":
                    uncropped_tests += row_origin <= row_gutter and col_origin <= col_gutter and descriptor["row_tail"] <= row_gutter and descriptor["col_tail"] <= col_gutter
                    novel_test_accents += bool(set(descriptor["rectangles"]) - training_accents)
                rectangles = list(descriptor["rectangles"].values())
                for rows, cols in rectangles:
                    adjacent += rows[1] - rows[0] == 1 or cols[1] - cols[0] == 1
                    long_span += rows[1] - rows[0] >= 3 or cols[1] - cols[0] >= 3
                for first, second in combinations(rectangles, 2):
                    shared_axis += bool(set(first[0]) & set(second[0]) or set(first[1]) & set(second[1]))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack diversity")
    _require(len(shapes) >= 35 and len(all_inputs) >= 150, "grid diversity too narrow")
    _require(sizes["train"] == sizes["test"] == {3, 4, 5}, "tile sizes lack split coverage")
    _require(len(gutters["train"]) >= 7 and len(gutters["test"]) >= 7, "gutter combinations too narrow")
    _require(accent_counts["test"] == {1, 2, 3}, "test accent counts lack coverage")
    _require(len(layouts["test"]) >= 30, "test rectangle layouts too narrow")
    _require(0 in backgrounds and len(backgrounds) >= 8, "background colors lack diversity")
    _require(partial_leading > 10 and partial_trailing > 10 and uncropped_tests > 0, "boundary exposure lacks diversity")
    _require(unequal_gutters > 10 and adjacent > 10 and long_span > 10 and shared_axis > 0, "geometry combinations lack diversity")
    _require(novel_test_accents > 0, "no novel test accent colors")
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        _require(len(generate_puzzle(9000 + 10 * train_count + test_count, train_count, test_count)["train"]) == train_count, "boundary count failed")
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(111, train_count, test_count)
        _require(train_count is None or len(puzzle["train"]) == train_count, "partial train count failed")
        _require(test_count is None or len(puzzle["test"]) == test_count, "partial test count failed")
    for options in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}):
        try:
            generate_puzzle(111, **options)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    _require(_puzzles_equal(generate_puzzle(333, 7, 3), generate_puzzle(333, 7, 3)), "generation is nondeterministic")
    _require(not _puzzles_equal(generate_puzzle(333, 7, 3), generate_puzzle(334, 7, 3)), "different seeds duplicated")
