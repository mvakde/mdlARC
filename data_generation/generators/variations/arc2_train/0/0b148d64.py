import numpy as np


_SOURCE_INPUT_CODES = {
    (
        "2022220000202222002/"
        "2222022000022222000/"
        "0022020000222022222/"
        "2020220000022222200/"
        "0202222000000220222/"
        "2220202000202222020/"
        "0000000000000000000/"
        "0000000000000000000/"
        "0000000000000000000/"
        "0000000000000000000/"
        "0000000000000000000/"
        "2020002000033333033/"
        "0220022000333000330/"
        "0220020000333030300/"
        "2220022000330003333/"
        "2002220000300030303/"
        "2020002000033033303/"
        "0220220000033003030"
    ),
    (
        "888880888800008888088/"
        "800808088800008880008/"
        "888000888800008808888/"
        "880888808800008800088/"
        "888808808800008880888/"
        "000880800800008000800/"
        "888800808000008880888/"
        "800800880800008088888/"
        "888888080000008888800/"
        "000000000000000000000/"
        "000000000000000000000/"
        "022200222200008808808/"
        "202220022200008888080/"
        "022222202000008880008/"
        "222202222200008808880/"
        "222222020000008888800/"
        "222220202200008080888/"
        "220220000000008808008/"
        "022002200200008000880/"
        "222222222200000880088/"
        "202202222200008880888"
    ),
    (
        "01011100110100000/"
        "10100000111111011/"
        "11011000111111011/"
        "11001100110111111/"
        "01110000110001110/"
        "10010000110011111/"
        "00011000111001001/"
        "00000000000000000/"
        "00000000000000000/"
        "00000000000000000/"
        "40040400100111111/"
        "44440400101111110/"
        "40400400010011111/"
        "04444000110010101/"
        "44404400111111110/"
        "04444000010000111/"
        "04440400010101110/"
        "04000000101110101/"
        "44040400111001110"
    ),
    (
        "11110100303333330/"
        "10101100030333000/"
        "11011000000333300/"
        "00011100330330300/"
        "11111100030333033/"
        "11111100330003003/"
        "00000000000000000/"
        "00000000000000000/"
        "00000000000000000/"
        "00000000000000000/"
        "30000300333030303/"
        "03300300030333000/"
        "33333000300030003/"
        "30303000033333303/"
        "03300000003330330"
    ),
}


def _validated_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _optional_count(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _optional_count("num_train", num_train)
    num_test = _optional_count("num_test", num_test)

    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        choices = 10 - num_test
        num_train = num_test + 1 + (seed // 4) % choices
    elif num_train is None:
        choices = 10 - num_test
        num_train = num_test + 1 + seed % choices
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = 1 + seed % maximum_test

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def sample_puzzle_parameters(rng: np.random.Generator, num_train: int) -> dict:
    palette = [int(value) for value in rng.permutation(10)]
    return {
        "palette": palette,
        "palette_mode": str(rng.choice(("stable", "rotating", "broad"))),
        "position_mode": str(rng.choice(("mixed", "consistent_train", "heldout", "source_like"))),
        "corners": [int(value) for value in rng.permutation(4)],
        "blank_demo": int(rng.integers(0, num_train)),
        "train_count": num_train,
        "common_style": bool(rng.integers(0, 4) == 0),
        "style": str(rng.choice(("speckle", "stripes", "patches", "holes", "motif", "dense", "sparse", "solid"))),
    }


def _sample_axis(rng: np.random.Generator, bands: int, full: bool) -> tuple[list[int], list[int]]:
    gutters = [int(rng.integers(1, 5)) for _ in range(bands - 1)]
    minimum = 2 * bands + sum(gutters)
    mode = str(rng.choice(("equal", "small", "large", "extreme", "free")))
    if full:
        total = 30
        mode = "free"
    elif mode == "equal":
        upper = (30 - sum(gutters)) // bands
        length = int(rng.integers(2, upper + 1))
        return [length] * bands, gutters
    elif mode == "small":
        total = int(rng.integers(minimum, min(30, minimum + 6) + 1))
    elif mode == "large":
        total = int(rng.integers(max(minimum, 24), 31))
    else:
        total = int(rng.integers(minimum, 31))
    extra = total - minimum
    if mode == "extreme":
        weights = np.full(bands, 0.07)
        weights[int(rng.integers(0, bands))] = 1.0
        weights /= weights.sum()
    else:
        weights = rng.dirichlet(np.ones(bands))
    allocation = rng.multinomial(extra, weights)
    return [2 + int(value) for value in allocation], gutters


def _initial_mask(rng: np.random.Generator, shape: tuple[int, int], style: str) -> np.ndarray:
    height, width = shape
    if style in ("speckle", "sparse", "dense"):
        bounds = {"speckle": (0.25, 0.72), "sparse": (0.06, 0.24), "dense": (0.76, 0.96)}
        low, high = bounds[style]
        return rng.random(shape) < rng.uniform(low, high)
    if style == "solid":
        return np.ones(shape, dtype=bool)
    if style == "stripes":
        period = int(rng.integers(2, 5))
        thickness = int(rng.integers(1, period))
        offset = int(rng.integers(0, period))
        if rng.integers(0, 2):
            mask = np.broadcast_to(((np.arange(height) + offset) % period < thickness)[:, None], shape).copy()
        else:
            mask = np.broadcast_to(((np.arange(width) + offset) % period < thickness)[None, :], shape).copy()
        return np.where(rng.random(shape) < 0.12, ~mask, mask)
    if style == "patches":
        mask = np.zeros(shape, dtype=bool)
        for _ in range(int(rng.integers(2, 6))):
            row = int(rng.integers(0, height))
            column = int(rng.integers(0, width))
            mask[row:row + int(rng.integers(1, max(2, height // 2 + 2))), column:column + int(rng.integers(1, max(2, width // 2 + 2)))] = True
        return mask
    if style == "holes":
        mask = np.ones(shape, dtype=bool)
        for _ in range(int(rng.integers(1, 5))):
            row = int(rng.integers(0, height))
            column = int(rng.integers(0, width))
            mask[row:row + int(rng.integers(1, 4)), column:column + int(rng.integers(1, 4))] = False
        return mask & (rng.random(shape) > 0.08)
    tile_height = int(rng.integers(2, 4))
    tile_width = int(rng.integers(2, 4))
    tile = rng.random((tile_height, tile_width)) < rng.uniform(0.3, 0.75)
    return np.tile(tile, (height // tile_height + 1, width // tile_width + 1))[:height, :width].copy()


def _blank_edges(rng: np.random.Generator, mask: np.ndarray, locked: np.ndarray) -> None:
    height, width = mask.shape
    sides = [int(value) for value in rng.permutation(4)]
    for side in sides[:int(rng.integers(1, 3))]:
        thickness = int(rng.integers(1, min(3, (height if side < 2 else width))))
        proposed = locked.copy()
        if side == 0:
            proposed[:thickness, :] = True
        elif side == 1:
            proposed[-thickness:, :] = True
        elif side == 2:
            proposed[:, :thickness] = True
        else:
            proposed[:, -thickness:] = True
        if np.any(~proposed):
            locked[:] = proposed
    mask[locked] = False


def _make_panel_masks(rng: np.random.Generator, row_heights: list[int], column_widths: list[int],
                      odd_index: int, target_edge: str, common_style: str | None,
                      force_ordinary_edge: bool) -> list[np.ndarray]:
    columns = len(column_widths)
    shapes = [(height, width) for height in row_heights for width in column_widths]
    styles = ("speckle", "stripes", "patches", "holes", "motif", "dense", "sparse", "solid")
    masks = [_initial_mask(rng, shape, common_style or str(rng.choice(styles))) for shape in shapes]
    locked = [np.zeros(shape, dtype=bool) for shape in shapes]
    for index, mask in enumerate(masks):
        if index == odd_index:
            if target_edge == "blank":
                _blank_edges(rng, mask, locked[index])
        elif rng.random() < 0.25 or (force_ordinary_edge and index == (odd_index + 1) % len(masks)):
            _blank_edges(rng, mask, locked[index])

    def activate(candidates: list[tuple[int, int, int]]) -> None:
        allowed = [(index, row, column) for index, row, column in candidates if not locked[index][row, column]]
        if not allowed:
            ordinary = [item for item in candidates if item[0] != odd_index]
            allowed = ordinary or candidates
            index, row, column = allowed[int(rng.integers(0, len(allowed)))]
            locked[index][row, column] = False
        else:
            index, row, column = allowed[int(rng.integers(0, len(allowed)))]
        masks[index][row, column] = True

    for index, mask in enumerate(masks):
        if not np.any(mask):
            positions = np.argwhere(~locked[index])
            if not len(positions):
                locked[index][0, 0] = False
                positions = np.array([[0, 0]])
            row, column = positions[int(rng.integers(0, len(positions)))]
            mask[row, column] = True

    if target_edge == "clean":
        mask = masks[odd_index]
        edges = ([(0, column) for column in range(mask.shape[1])],
                 [(mask.shape[0] - 1, column) for column in range(mask.shape[1])],
                 [(row, 0) for row in range(mask.shape[0])],
                 [(row, mask.shape[1] - 1) for row in range(mask.shape[0])])
        for edge in edges:
            if not any(mask[row, column] for row, column in edge):
                activate([(odd_index, row, column) for row, column in edge])

    for band, height in enumerate(row_heights):
        for row in range(height):
            candidates = [(band * columns + column_band, row, column)
                          for column_band, width in enumerate(column_widths) for column in range(width)]
            if not any(masks[index][local_row, local_column] for index, local_row, local_column in candidates):
                activate(candidates)
    for band, width in enumerate(column_widths):
        for column in range(width):
            candidates = [(row_band * columns + band, row, column)
                          for row_band, height in enumerate(row_heights) for row in range(height)]
            if not any(masks[index][local_row, local_column] for index, local_row, local_column in candidates):
                activate(candidates)
    return masks


def _pair_colors(rng: np.random.Generator, pair_number: int, parameters: dict) -> tuple[int, int, int]:
    palette = parameters["palette"]
    if pair_number == 0:
        shared, exceptional = palette[:2]
        background = palette[2]
    elif pair_number == 1:
        exceptional, shared = palette[:2]
        background = palette[3]
    elif parameters["palette_mode"] == "stable":
        shared, exceptional = (palette[:2] if rng.integers(0, 2) else palette[1::-1])
        background = int(rng.choice(palette[2:]))
    elif parameters["palette_mode"] == "rotating":
        shared, exceptional = [int(value) for value in rng.choice(palette[:5], 2, replace=False)]
        background = int(rng.choice([value for value in palette if value not in (shared, exceptional)]))
    else:
        background, shared, exceptional = [int(value) for value in rng.choice(10, 3, replace=False)]
    return background, shared, exceptional


def _target_index(rng: np.random.Generator, regime: str, pair_index: int,
                  rows: int, columns: int, parameters: dict) -> int:
    mode = parameters["position_mode"]
    corners = parameters["corners"]
    if mode == "source_like":
        corner = (2, 3)[(pair_index + corners[0]) % 2] if regime == "source_train" else (0, 1)[int(rng.integers(0, 2))]
    elif mode == "consistent_train":
        corner = corners[0] if regime == "source_train" else corners[1]
    elif mode == "heldout":
        corner = corners[(pair_index % 3) + 1] if regime == "source_train" else corners[0]
    else:
        corner = corners[pair_index % 4] if regime == "source_train" and pair_index < 2 else int(rng.integers(0, 4))
    if rows == 3 and columns == 3 and mode == "mixed" and rng.random() < 0.55:
        return int(rng.choice((1, 3, 4, 5, 7)))
    row = rows - 1 if corner >= 2 else 0
    column = columns - 1 if corner % 2 else 0
    return row * columns + column


def _slices(lengths: list[int], gutters: list[int]) -> list[slice]:
    position = 0
    result = []
    for index, length in enumerate(lengths):
        result.append(slice(position, position + length))
        position += length + (gutters[index] if index < len(gutters) else 0)
    return result


def _generate_pair(rng: np.random.Generator, regime: str, pair_index: int,
                   pair_number: int, parameters: dict,
                   layout_override: tuple | None = None) -> tuple[np.ndarray, np.ndarray]:
    if layout_override is not None:
        row_heights, column_widths, row_gutters, column_gutters = [list(values) for values in layout_override]
        array_shape = (len(row_heights), len(column_widths))
    elif regime == "source_train":
        array_shape = (2, 2) if pair_index == 0 else (2, 3) if pair_index == 1 else ((2, 2), (2, 3), (3, 2), (3, 3))[int(rng.integers(0, 4))]
    else:
        array_shape = ((2, 2), (2, 3), (3, 2), (3, 3))[int(rng.integers(0, 4))]
    rows, columns = array_shape
    if layout_override is None:
        if array_shape == (2, 2) and rng.random() < 0.24:
            row_heights = [int(rng.integers(5, 11)) for _ in range(2)]
            column_widths = [int(rng.integers(5, 11)) for _ in range(2)]
            row_gutters = [int(rng.integers(2, 5))]
            column_gutters = [int(rng.integers(2, 5))]
        else:
            full = bool(rng.random() < 0.045)
            row_heights, row_gutters = _sample_axis(rng, rows, full)
            column_widths, column_gutters = _sample_axis(rng, columns, full)
    odd_index = _target_index(rng, regime, pair_index, rows, columns, parameters)
    background, shared, exceptional = _pair_colors(rng, pair_number, parameters)
    if regime == "source_train" and pair_index == parameters["blank_demo"]:
        target_edge = "blank"
    elif regime == "source_train" and pair_index == (parameters["blank_demo"] + 1) % parameters["train_count"]:
        target_edge = "clean"
    else:
        target_edge = ("blank", "clean", "random")[int(rng.integers(0, 3))]
    common_style = parameters["style"] if parameters["common_style"] else None
    force_ordinary_edge = (regime == "source_train" and pair_index ==
                           (parameters["blank_demo"] + 1) % parameters["train_count"])
    masks = _make_panel_masks(rng, row_heights, column_widths, odd_index, target_edge,
                              common_style, force_ordinary_edge)
    height = sum(row_heights) + sum(row_gutters)
    width = sum(column_widths) + sum(column_gutters)
    grid = np.full((height, width), background, dtype=np.int8)
    row_slices = _slices(row_heights, row_gutters)
    column_slices = _slices(column_widths, column_gutters)
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            panel = grid[row_slices[row], column_slices[column]]
            panel[masks[index]] = exceptional if index == odd_index else shared
    odd_row, odd_column = divmod(odd_index, columns)
    return grid, grid[row_slices[odd_row], column_slices[odd_column]].copy()


def _runs(mask: np.ndarray) -> list[slice]:
    result = []
    start = None
    for index, value in enumerate(list(mask) + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append(slice(start, index))
            start = None
    return result


def _analyze_input(grid: np.ndarray) -> dict:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    candidates = []
    for raw_background in np.unique(grid):
        background = int(raw_background)
        row_bands = _runs(~np.all(grid == background, axis=1))
        column_bands = _runs(~np.all(grid == background, axis=0))
        if not (2 <= len(row_bands) <= 3 and 2 <= len(column_bands) <= 3):
            continue
        if row_bands[0].start != 0 or row_bands[-1].stop != grid.shape[0]:
            continue
        if column_bands[0].start != 0 or column_bands[-1].stop != grid.shape[1]:
            continue
        panels = [grid[row_slice, column_slice] for row_slice in row_bands for column_slice in column_bands]
        foregrounds = []
        for panel in panels:
            colors = [int(value) for value in np.unique(panel) if int(value) != background]
            if len(colors) != 1:
                break
            foregrounds.append(colors[0])
        if len(foregrounds) != len(panels):
            continue
        counts = {color: foregrounds.count(color) for color in set(foregrounds)}
        if sorted(counts.values()) != [1, len(panels) - 1]:
            continue
        exceptional = next(color for color, count in counts.items() if count == 1)
        shared = next(color for color, count in counts.items() if count == len(panels) - 1)
        candidates.append({"background": background, "row_bands": row_bands, "column_bands": column_bands,
                           "panels": panels, "foreground_colors": tuple(foregrounds),
                           "shared_color": shared, "exceptional_color": exceptional,
                           "odd_index": foregrounds.index(exceptional),
                           "layout": (tuple(item.stop - item.start for item in row_bands),
                                      tuple(item.stop - item.start for item in column_bands),
                                      tuple(row_bands[index + 1].start - item.stop for index, item in enumerate(row_bands[:-1])),
                                      tuple(column_bands[index + 1].start - item.stop for index, item in enumerate(column_bands[:-1])))})
    assert len(candidates) == 1, "input must have one uniquely readable panel array"
    return candidates[0]
def _array_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(value) for value in grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _grid_code(grid: np.ndarray) -> str:
    return "/".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _has_blank_outer_edge(
    grid: np.ndarray,
    background: int,
) -> bool:
    return bool(
        np.all(grid[0, :] == background)
        or np.all(grid[-1, :] == background)
        or np.all(grid[:, 0] == background)
        or np.all(grid[:, -1] == background)
    )


def _validate_generated_puzzle_impl(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs = set()
    seen_pairs = set()
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= extent <= 30 for extent in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _array_key(pair["input"])
            pair_key = (input_key, _array_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _grid_code(pair["input"]) not in _SOURCE_INPUT_CODES
            assert not np.array_equal(pair["input"], pair["output"])
            analysis = _analyze_input(pair["input"])
            assert np.array_equal(pair["output"], analysis["panels"][analysis["odd_index"]])
            assert {int(value) for value in np.unique(pair["output"])} <= {
                analysis["background"], analysis["exceptional_color"]}
            records[split].append(analysis)
    train = records["train"]
    assert any(len(item["panels"]) == 4 for item in train)
    assert any(len(item["panels"]) > 4 for item in train)
    assert len({item["background"] for item in train}) >= 2
    assert any(_has_blank_outer_edge(pair["output"], record["background"])
               for pair, record in zip(puzzle["train"], train))
    assert any(not _has_blank_outer_edge(pair["output"], record["background"])
               and any(_has_blank_outer_edge(panel, record["background"])
                       for index, panel in enumerate(record["panels"]) if index != record["odd_index"])
               for pair, record in zip(puzzle["train"], train))
    assert any(left["shared_color"] == right["exceptional_color"]
               and left["exceptional_color"] == right["shared_color"]
               for left in train for right in train if left is not right)
    if any(_has_blank_outer_edge(pair["output"], record["background"])
           for pair, record in zip(puzzle["test"], records["test"])):
        assert any(_has_blank_outer_edge(pair["output"], record["background"])
                   for pair, record in zip(puzzle["train"], train))
    if any(len(item["panels"]) > 4 for item in records["test"]):
        assert any(len(item["panels"]) > 4 for item in train)


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_generated_puzzle_impl(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("invalid generated puzzle") from exc


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _validated_seed(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    master_rng = np.random.default_rng(seed)
    parameters = sample_puzzle_parameters(master_rng, num_train)
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    layouts = []
    for regime, split, count in (("source_train", "train", num_train), ("target_test", "test", num_test)):
        for pair_index in range(count):
            pair_number = pair_index if split == "train" else num_train + pair_index
            for _ in range(24):
                pair_rng = np.random.default_rng(int(master_rng.integers(0, np.iinfo(np.int64).max)))
                reuse = pair_number >= 2 and layouts and pair_rng.random() < 0.18
                layout_override = layouts[int(pair_rng.integers(0, len(layouts)))] if reuse else None
                input_grid, output_grid = _generate_pair(pair_rng, regime, pair_index, pair_number, parameters, layout_override)
                input_key = _array_key(input_grid)
                if input_key in used_inputs or _grid_code(input_grid) in _SOURCE_INPUT_CODES:
                    continue
                analysis = _analyze_input(input_grid)
                if not np.array_equal(output_grid, analysis["panels"][analysis["odd_index"]]):
                    continue
                puzzle[split].append({"input": input_grid, "output": output_grid})
                used_inputs.add(input_key)
                layouts.append(analysis["layout"])
                break
            else:
                raise RuntimeError("could not generate a unique valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[split]) == len(second[split])
               and all(np.array_equal(left[role], right[role])
                       for left, right in zip(first[split], second[split]) for role in ("input", "output"))
               for split in ("train", "test"))


def _require_value_error(seed: int, num_train: int | None = None, num_test: int | None = None) -> None:
    try:
        generate_puzzle(seed, num_train, num_test)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(17), generate_puzzle(17))
    assert _puzzles_equal(generate_puzzle(913, 7, 3), generate_puzzle(913, 7, 3))
    assert not _puzzles_equal(generate_puzzle(913, 7, 3), generate_puzzle(914, 7, 3))
    for seed, train_count, test_count in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                                          (201, None, 1), (202, None, 4), (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        _require_value_error(0, train_count, test_count)
    _require_value_error(-1)
    train_counts, test_counts = set(), set()
    inputs, layouts, shapes, backgrounds, foregrounds = set(), set(), set(), set(), set()
    train_corners, test_corners, arrays, gutter_widths = set(), set(), set(), set()
    interior_positions = 0
    equal_row = equal_column = full_axis = small_axis = sparse = dense = solid = 0
    blank_target = clean_target_blank_distractor = pixel_counterexample = 0
    mixed_positions = consistent_train_positions = 0
    source_like = reused_layout_puzzles = 0
    stable_palettes = varied_palettes = heldout_position_puzzles = 0
    mixed_density_pairs = largest_target = smallest_target = 0
    both_blank = target_only_blank = thick_blank = extreme_panels = 0
    blank_sides = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        positions = set()
        test_positions = set()
        puzzle_layouts = []
        puzzle_foregrounds = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                input_grid, output_grid = pair["input"], pair["output"]
                analysis = _analyze_input(input_grid)
                assert np.array_equal(output_grid, analysis["panels"][analysis["odd_index"]])
                assert _grid_code(input_grid) not in _SOURCE_INPUT_CODES
                inputs.add(_array_key(input_grid))
                layouts.add(analysis["layout"])
                puzzle_layouts.append(analysis["layout"])
                shapes.add(output_grid.shape)
                backgrounds.add(analysis["background"])
                foregrounds.update(analysis["foreground_colors"])
                puzzle_foregrounds.update(analysis["foreground_colors"])
                row_lengths, column_lengths, row_gutters, column_gutters = analysis["layout"]
                source_like += (len(row_lengths) == len(column_lengths) == 2
                                and all(5 <= extent <= 10 for extent in row_lengths + column_lengths)
                                and all(2 <= extent <= 4 for extent in row_gutters + column_gutters))
                arrays.add((len(row_lengths), len(column_lengths)))
                gutter_widths.update(row_gutters + column_gutters)
                equal_row += len(set(row_lengths)) == 1
                equal_column += len(set(column_lengths)) == 1
                full_axis += 30 in input_grid.shape
                small_axis += min(input_grid.shape) <= 10
                odd_row, odd_column = divmod(analysis["odd_index"], len(column_lengths))
                corner = (2 if odd_row == len(row_lengths) - 1 else 0) + (1 if odd_column == len(column_lengths) - 1 else 0)
                if odd_row not in (0, len(row_lengths) - 1) or odd_column not in (0, len(column_lengths) - 1):
                    corner = 4
                    interior_positions += 1
                if split == "train":
                    if corner < 4:
                        train_corners.add(corner)
                    positions.add(corner)
                else:
                    if corner < 4:
                        test_corners.add(corner)
                    test_positions.add(corner)
                area = output_grid.size
                count = int(np.count_nonzero(output_grid == analysis["exceptional_color"]))
                density = count / area
                sparse += density <= 0.20
                dense += density >= 0.78
                solid += density == 1
                ordinary_count = sum(int(np.count_nonzero(panel == analysis["shared_color"]))
                                     for index, panel in enumerate(analysis["panels"]) if index != analysis["odd_index"])
                pixel_counterexample += count >= ordinary_count
                panel_densities = [np.count_nonzero(panel != analysis["background"]) / panel.size
                                   for panel in analysis["panels"]]
                mixed_density_pairs += max(panel_densities) - min(panel_densities) >= 0.4
                largest_target += panel_densities[analysis["odd_index"]] == max(panel_densities)
                smallest_target += panel_densities[analysis["odd_index"]] == min(panel_densities)
                panel_areas = [panel.size for panel in analysis["panels"]]
                extreme_panels += max(panel_areas) >= 4 * min(panel_areas)
                blank = _has_blank_outer_edge(output_grid, analysis["background"])
                blank_target += blank
                ordinary_blank = any(
                    _has_blank_outer_edge(panel, analysis["background"])
                    for index, panel in enumerate(analysis["panels"]) if index != analysis["odd_index"])
                clean_target_blank_distractor += not blank and ordinary_blank
                both_blank += blank and ordinary_blank
                target_only_blank += blank and not ordinary_blank
                for side, edge in enumerate((output_grid[0], output_grid[-1], output_grid[:, 0], output_grid[:, -1])):
                    if np.all(edge == analysis["background"]):
                        blank_sides.add(side)
                thick_blank += (output_grid.shape[0] >= 3 and
                                (np.all(output_grid[:2] == analysis["background"]) or
                                 np.all(output_grid[-2:] == analysis["background"]))) or (
                                    output_grid.shape[1] >= 3 and
                                    (np.all(output_grid[:, :2] == analysis["background"]) or
                                     np.all(output_grid[:, -2:] == analysis["background"])))
        mixed_positions += len(positions) > 1
        consistent_train_positions += len(positions) == 1
        heldout_position_puzzles += bool(test_positions - positions)
        stable_palettes += len(puzzle_foregrounds) == 2
        varied_palettes += len(puzzle_foregrounds) > 2
        reused_layout_puzzles += len(puzzle_layouts) != len(set(puzzle_layouts))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(inputs) >= 200 and len(layouts) >= 120 and len(shapes) >= 20
    assert train_corners == test_corners == {0, 1, 2, 3}
    assert interior_positions > 0
    assert arrays == {(2, 2), (2, 3), (3, 2), (3, 3)}
    assert len(backgrounds) >= 8 and 0 in foregrounds
    assert gutter_widths == {1, 2, 3, 4}
    assert equal_row > 0 and equal_column > 0 and full_axis > 0 and small_axis > 0
    assert sparse > 0 and dense > 0 and solid > 0
    assert blank_target > 0 and clean_target_blank_distractor > 0 and pixel_counterexample > 0
    assert mixed_positions > 0 and consistent_train_positions > 0
    assert source_like > 0 and reused_layout_puzzles > 0
    assert stable_palettes > 0 and varied_palettes > 0 and heldout_position_puzzles > 0
    assert mixed_density_pairs > 0 and largest_target > 0 and smallest_target > 0
    assert both_blank > 0 and target_only_blank < blank_target and thick_blank > 0
    assert blank_sides == {0, 1, 2, 3} and extreme_panels > 0
