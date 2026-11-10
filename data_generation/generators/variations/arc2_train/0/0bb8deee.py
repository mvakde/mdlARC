import numpy as np


_SOURCE_INPUTS = (
    "0000100000000/0000100000000/0000100300000/0000100033000/0000100030000/0220100000000/2220100000000/0200100000000/0000100000000/1111111111111/0000100000000/0050100000000/5550100808000/0500100080000/0000100800000",
    "0011000200033/0001000200330/0001100200030/2222222222222/0000000200000/0000000200000/0000000200000/0040000200000/0404000205500/0040000200500/0000000200050/0000000200000",
    "000000300110/002200301000/020200300100/002000300000/333333333333/000000300000/000000300050/000000300505/000000300555/000000300000/000000300000/000000300000/400000300000/444000300000/040000300000/000000300000",
    "0000001000000/0000001000000/0000001003330/0022001000330/0020001003000/0202001000000/0000001000000/1111111111111/0000001000000/0000001000000/0000001000400/0066001004040/0606001000400/0660001000000",
)
_TILE_SIZES = ((2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (3, 5), (4, 2), (4, 3), (4, 4), (5, 3), (5, 5))
_EXTENDED_LAYOUTS = ((2, 3), (3, 2), (1, 3), (3, 1), (1, 2), (2, 1))


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(seed)


def _normalize_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    num_train = _normalize_count(num_train, "num_train")
    num_test = _normalize_count(num_test, "num_test")
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = max(2, num_test + 1)
        return minimum + (seed // 4) % (11 - minimum), num_test
    if num_train is None:
        minimum = max(2, num_test + 1)
        return minimum + seed % (11 - minimum), num_test
    return num_train, 1 + seed % min(4, num_train - 1)


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _source_copy(input_grid: np.ndarray) -> bool:
    rows = "/".join("".join(str(int(value)) for value in row) for row in input_grid)
    return rows in _SOURCE_INPUTS


def _orbit_size(mask: np.ndarray) -> int:
    forms = set()
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        for variant in (rotated, np.fliplr(rotated)):
            forms.add((variant.shape, tuple(bool(value) for value in variant.flat)))
    return len(forms)


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if not mask[row, col] or seen[row, col]:
                continue
            count += 1
            stack = [(row, col)]
            seen[row, col] = True
            while stack:
                current_row, current_col = stack.pop()
                for row_step, col_step in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_row, next_col = current_row + row_step, current_col + col_step
                    if 0 <= next_row < mask.shape[0] and 0 <= next_col < mask.shape[1] and mask[next_row, next_col] and not seen[next_row, next_col]:
                        seen[next_row, next_col] = True
                        stack.append((next_row, next_col))
    return count


def _derive_output(input_grid: np.ndarray) -> tuple[np.ndarray, dict]:
    height, width = input_grid.shape
    candidates = []
    for color in np.unique(input_grid):
        row_lines = np.flatnonzero(np.all(input_grid == color, axis=1))
        col_lines = np.flatnonzero(np.all(input_grid == color, axis=0))
        if not row_lines.size and not col_lines.size:
            continue
        line_mask = np.zeros((height, width), dtype=bool)
        line_mask[row_lines, :] = True
        line_mask[:, col_lines] = True
        if np.array_equal(input_grid == color, line_mask):
            candidates.append((int(color), row_lines, col_lines))
    _check(len(candidates) == 1, "separator color is not uniquely visible")
    divider, row_lines, col_lines = candidates[0]
    row_edges = (0,) + tuple(int(value) + 1 for value in row_lines) + (height + 1,)
    col_edges = (0,) + tuple(int(value) + 1 for value in col_lines) + (width + 1,)
    row_ranges = [(row_edges[index], row_edges[index + 1] - 1) for index in range(len(row_edges) - 1)]
    col_ranges = [(col_edges[index], col_edges[index + 1] - 1) for index in range(len(col_edges) - 1)]
    _check(len(row_ranges) * len(col_ranges) >= 2, "at least two compartments are required")
    _check(all(stop - start >= 2 for start, stop in row_ranges + col_ranges), "compartments are too small")
    regions = [input_grid[row_start:row_stop, col_start:col_stop] for row_start, row_stop in row_ranges for col_start, col_stop in col_ranges]
    color_sets = [set(int(value) for value in np.unique(region)) for region in regions]
    _check(all(len(colors) == 2 for colors in color_sets), "each compartment needs background and one foreground")
    common = set.intersection(*color_sets)
    _check(len(common) == 1, "background must be identifiable even with shared foreground colors")
    background = common.pop()
    _check(background != divider, "background and separator must differ")
    tiles, payload_colors, offsets, masks = [], [], [], []
    for region, colors in zip(regions, color_sets):
        payload = (colors - {background}).pop()
        payload_rows, payload_cols = np.where(region == payload)
        _check(payload_rows.size >= 2, "motif must have recoverable extent")
        row_start, row_stop = int(payload_rows.min()), int(payload_rows.max()) + 1
        col_start, col_stop = int(payload_cols.min()), int(payload_cols.max()) + 1
        tile = region[row_start:row_stop, col_start:col_stop].copy()
        mask = tile == payload
        _check(tile.shape[0] >= 2 and tile.shape[1] >= 2, "motif needs two-dimensional bounds")
        _check(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any(), "motif bounds are not tight")
        tiles.append(tile)
        payload_colors.append(payload)
        offsets.append((row_start, col_start, region.shape[0] - row_stop, region.shape[1] - col_stop))
        masks.append(mask)
    _check(len({tile.shape for tile in tiles}) == 1, "all tiles in a pair must share dimensions")
    output = np.concatenate([np.concatenate(tiles[index * len(col_ranges):(index + 1) * len(col_ranges)], axis=1) for index in range(len(row_ranges))], axis=0)
    _check(output.shape[0] <= 30 and output.shape[1] <= 30, "assembled output exceeds ARC bounds")
    metadata = {
        "layout": (len(row_ranges), len(col_ranges)), "tile_size": tiles[0].shape,
        "input_size": (height, width),
        "row_spans": tuple(stop - start for start, stop in row_ranges),
        "col_spans": tuple(stop - start for start, stop in col_ranges),
        "background": background, "divider": divider,
        "payload_colors": tuple(payload_colors), "offsets": tuple(offsets), "masks": tuple(masks),
    }
    return output, metadata


def _motif(rng: np.random.Generator, size: tuple[int, int], asymmetric: bool = False) -> np.ndarray:
    height, width = size
    for _ in range(24):
        mode = int(rng.integers(0, 8))
        if mode == 0:
            mask = np.ones(size, dtype=bool)
        elif mode in (1, 2):
            mask = np.zeros(size, dtype=bool)
            if int(rng.integers(0, 2)):
                mask[0, 0] = mask[-1, -1] = True
            else:
                mask[0, -1] = mask[-1, 0] = True
            if mode == 2:
                mask[int(rng.integers(0, height)), int(rng.integers(0, width))] = True
        else:
            mask = rng.random(size) < float(rng.uniform(0.25, 0.85))
            mask[0, int(rng.integers(0, width))] = True
            mask[-1, int(rng.integers(0, width))] = True
            mask[int(rng.integers(0, height)), 0] = True
            mask[int(rng.integers(0, height)), -1] = True
        if not asymmetric or _orbit_size(mask) >= 4:
            return mask
    mask = np.zeros(size, dtype=bool)
    mask[0, 0] = mask[-1, -1] = mask[0, -1] = True
    return mask


def _sample_tile_size(rng: np.random.Generator, excluded: tuple[int, int] | None = None) -> tuple[int, int]:
    choices = [size for size in _TILE_SIZES if size != excluded]
    weights = np.array([3 if size == (3, 3) else 2 if size == (2, 2) else 1 for size in choices], dtype=float)
    weights /= weights.sum()
    return choices[int(rng.choice(len(choices), p=weights))]


def _sample_spans(rng: np.random.Generator, count: int, tile_extent: int, balance: str, interior: bool, compact: bool) -> tuple[int, ...]:
    minimum = tile_extent + (2 if interior else 1)
    capacity = 30 - (count * minimum + count - 1)
    _check(capacity >= 0, "layout cannot fit in 30 cells")
    scale = int(rng.integers(0, 20))
    if compact:
        extras = int(rng.integers(0, min(1, capacity) + 1))
    elif scale < 10:
        extras = int(rng.integers(0, min(3, capacity) + 1))
    elif scale < 16:
        extras = int(rng.integers(0, min(10, capacity) + 1))
    elif scale < 19:
        extras = int(rng.integers(capacity // 2, capacity + 1))
    else:
        extras = capacity
    if count == 1:
        return (minimum + extras,)
    if balance == "off" and capacity >= 2:
        extras = max(2, extras)
    if balance == "center":
        spans = [minimum + extras // count] * count
        for index in rng.permutation(count)[:extras % count]:
            spans[int(index)] += 1
    elif balance == "off":
        spans = [minimum] * count
        spans[int(rng.integers(0, count))] += extras
    else:
        distribution = rng.multinomial(extras, np.full(count, 1 / count))
        spans = [minimum + int(value) for value in distribution]
    return tuple(spans)


def _choose_colors(rng: np.random.Generator, count: int, familiar_pool: tuple[int, ...], seen_payloads: set[int], mode: str) -> tuple[int, int, tuple[int, ...]]:
    if mode == "familiar" and len(seen_payloads) >= 2:
        structural = [color for color in range(10) if color not in seen_payloads]
        background, divider = (int(value) for value in rng.choice(structural, size=2, replace=False))
        available = sorted(seen_payloads)
    elif mode == "one_novel" and 2 <= len(seen_payloads) <= 7:
        novel = int(rng.choice([color for color in range(10) if color not in seen_payloads]))
        structural = [color for color in range(10) if color not in seen_payloads and color != novel]
        background, divider = (int(value) for value in rng.choice(structural, size=2, replace=False))
        available = sorted(seen_payloads) + [novel]
    else:
        background, divider = (int(value) for value in rng.choice(10, size=2, replace=False))
        available = [color for color in (familiar_pool if mode == "train" else tuple(range(10))) if color not in (background, divider)]
    if len(available) < 2:
        available = [color for color in range(10) if color not in (background, divider)]
    if mode == "one_novel" and 2 <= len(seen_payloads) <= 7:
        chosen = [novel, int(rng.choice(sorted(seen_payloads)))]
    else:
        distinct_count = int(rng.integers(2, min(5, count, len(available)) + 1))
        chosen = [int(value) for value in rng.choice(available, size=distinct_count, replace=False)]
    colors = [int(rng.choice(chosen)) for _ in range(count)]
    for index, color in enumerate(chosen):
        colors[index] = color
    rng.shuffle(colors)
    return background, divider, tuple(colors)


def _generate_pair(rng: np.random.Generator, layout: tuple[int, int], size: tuple[int, int], balance: str, interior: bool, familiar_pool: tuple[int, ...], seen_payloads: set[int], palette_mode: str, reusable_masks: dict) -> dict:
    row_count, col_count = layout
    tile_height, tile_width = size
    compact = layout == (2, 2) and tile_height <= 3 and tile_width <= 3 and rng.random() < 0.5
    row_spans = _sample_spans(rng, row_count, tile_height, balance, interior, compact)
    col_spans = _sample_spans(rng, col_count, tile_width, balance, interior, compact)
    height, width = sum(row_spans) + row_count - 1, sum(col_spans) + col_count - 1
    background, divider, payloads = _choose_colors(rng, row_count * col_count, familiar_pool, seen_payloads, palette_mode)
    input_grid = np.full((height, width), background, dtype=np.int8)
    row_starts, col_starts = [], []
    position = 0
    for span in row_spans:
        row_starts.append(position)
        position += span
        if position < height:
            input_grid[position, :] = divider
            position += 1
    position = 0
    for span in col_spans:
        col_starts.append(position)
        position += span
        if position < width:
            input_grid[:, position] = divider
            position += 1
    masks = []
    for row_index in range(row_count):
        for col_index in range(col_count):
            motif_index = row_index * col_count + col_index
            if masks and rng.random() < 0.15:
                mask = masks[int(rng.integers(0, len(masks)))].copy()
            elif size in reusable_masks and rng.random() < 0.22:
                choices = reusable_masks[size]
                mask = choices[int(rng.integers(0, len(choices)))].copy()
            else:
                mask = _motif(rng, size, asymmetric=motif_index == 0 and palette_mode == "train")
            masks.append(mask)
            row_room = row_spans[row_index] - tile_height
            col_room = col_spans[col_index] - tile_width
            row_offset = int(rng.integers(1, row_room)) if interior else int(rng.integers(0, row_room + 1))
            col_offset = int(rng.integers(1, col_room)) if interior else int(rng.integers(0, col_room + 1))
            row_start = row_starts[row_index] + row_offset
            col_start = col_starts[col_index] + col_offset
            target = input_grid[row_start:row_start + tile_height, col_start:col_start + tile_width]
            target[mask] = payloads[motif_index]
    output_grid, _ = _derive_output(input_grid)
    reusable_masks.setdefault(size, []).extend(mask.copy() for mask in masks)
    return {"input": input_grid, "output": output_grid}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _normalize_seed(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    familiar_pool = tuple(int(value) for value in rng.choice(10, size=6, replace=False))
    extended = _EXTENDED_LAYOUTS[int(rng.integers(0, len(_EXTENDED_LAYOUTS)))]
    first_size = _sample_tile_size(rng)
    second_size = _sample_tile_size(rng, excluded=first_size)
    train_layouts, train_sizes = [(2, 2), extended], [first_size, second_size]
    for _ in range(2, num_train):
        train_layouts.append((2, 2) if rng.random() < 0.55 else _EXTENDED_LAYOUTS[int(rng.integers(0, len(_EXTENDED_LAYOUTS)))])
        train_sizes.append(_sample_tile_size(rng))
    training_balance = ("center", "off") if int(rng.integers(0, 2)) else ("off", "center")
    puzzle = {"train": [], "test": []}
    reusable_masks, seen_payloads = {}, set()
    for index in range(num_train):
        balance = training_balance[index] if index < 2 else ("center", "off", "free")[int(rng.integers(0, 3))]
        layout, size = train_layouts[index], train_sizes[index]
        interior = layout == (2, 2) and max(size) <= 4 and rng.random() < 0.22
        pair = _generate_pair(rng, layout, size, balance, interior, familiar_pool, seen_payloads, "train", reusable_masks)
        puzzle["train"].append(pair)
        _, metadata = _derive_output(pair["input"])
        seen_payloads.update(metadata["payload_colors"])
    for _ in range(num_test):
        layout = (2, 2) if rng.random() < 0.5 else train_layouts[int(rng.integers(0, len(train_layouts)))]
        size = _sample_tile_size(rng)
        balance = ("center", "off", "free")[int(rng.integers(0, 3))]
        interior = layout == (2, 2) and max(size) <= 4 and rng.random() < 0.25
        palette_mode = ("familiar", "one_novel", "fresh")[int(rng.integers(0, 3))]
        puzzle["test"].append(_generate_pair(rng, layout, size, balance, interior, familiar_pool, seen_payloads, palette_mode, reusable_masks))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle needs exactly train and test")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "pair collections must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "illegal pair counts")
    seen_inputs, train_metadata, test_metadata = set(), [], []
    for section, metadata_list in (("train", train_metadata), ("test", test_metadata)):
        for pair in puzzle[section]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair needs exactly input and output")
            for grid in pair.values():
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2, "grids must be 2D arrays")
                _check(all(1 <= extent <= 30 for extent in grid.shape), "grid exceeds ARC bounds")
                _check(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
                _check(int(grid.min()) >= 0 and int(grid.max()) <= 9, "grid uses non-ARC colors")
            input_key = _grid_key(pair["input"])
            _check(input_key not in seen_inputs, "duplicate input")
            seen_inputs.add(input_key)
            _check(not _source_copy(pair["input"]), "source input copied")
            expected, metadata = _derive_output(pair["input"])
            _check(np.array_equal(pair["output"], expected), "output must be exact spatial assembly")
            _check(not np.array_equal(pair["input"], pair["output"]), "unchanged input is not meaningful")
            _check(metadata["divider"] not in set(int(value) for value in np.unique(pair["output"])), "divider survives output")
            metadata_list.append(metadata)
    _check(train_metadata[0]["layout"] == (2, 2), "training must establish the source cross")
    _check(any(item["layout"] != (2, 2) for item in train_metadata), "training must demonstrate layout extension")
    _check(len({item["tile_size"] for item in train_metadata}) >= 2, "training must demonstrate changing tile sizes")
    _check(any(_orbit_size(mask) >= 4 for item in train_metadata for mask in item["masks"]), "training must show orientation-sensitive geometry")
    train_layouts = {item["layout"] for item in train_metadata}
    _check(all(item["layout"] in train_layouts for item in test_metadata), "test layout not demonstrated in training")


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple((section, tuple((_grid_key(pair["input"]), _grid_key(pair["output"])) for pair in puzzle[section])) for section in ("train", "test"))


def self_test() -> None:
    first = generate_puzzle(12345, 7, 3)
    _check(_puzzle_key(first) == _puzzle_key(generate_puzzle(12345, 7, 3)), "explicit generation is not deterministic")
    _check(_puzzle_key(generate_puzzle(54321)) == _puzzle_key(generate_puzzle(54321)), "automatic generation is not deterministic")
    train_counts, test_counts, puzzle_keys = set(), set(), set()
    layouts, tile_sizes, input_sizes, palettes = set(), set(), set(), set()
    symmetric = repeated = disconnected = solid = sparse = interior = shared = background_zero = 0
    centered_train = off_train = centered_test = off_test = 0
    compact = long_narrow = maximum_extent = internal_blank = same_offsets = 0
    familiar_tests = one_novel_tests = multiple_novel_tests = 0
    varying_backgrounds = varying_dividers = role_switches = 0
    varying_palettes = reused_test_motifs = new_test_motifs = 0
    one_dimensional = multiple_separators = combined_extended = mixed_axes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(_puzzle_key(puzzle))
        train_masks = set()
        train_payloads = set()
        puzzle_metadata = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                expected, metadata = _derive_output(pair["input"])
                _check(np.array_equal(expected, pair["output"]), "crop reconstruction failed")
                puzzle_metadata.append(metadata)
                layouts.add(metadata["layout"])
                tile_sizes.add(metadata["tile_size"])
                input_sizes.add(metadata["input_size"])
                palettes.add((metadata["background"], metadata["divider"], metadata["payload_colors"]))
                height, width = metadata["input_size"]
                compact += height <= 10 and width <= 10
                long_narrow += max(height, width) >= 2 * min(height, width)
                maximum_extent += 30 in (height, width)
                same_offsets += len(set(metadata["offsets"])) < len(metadata["offsets"])
                background_zero += metadata["background"] == 0
                shared += len(set(metadata["payload_colors"])) < len(metadata["payload_colors"])
                one_dimensional += min(metadata["layout"]) == 1
                multiple_separators += metadata["layout"] in ((2, 3), (3, 2))
                combined_extended += metadata["layout"] != (2, 2) and metadata["tile_size"] != (3, 3) and len(set(metadata["payload_colors"])) < len(metadata["payload_colors"])
                interior += all(min(offset) > 0 for offset in metadata["offsets"])
                spans = (metadata["row_spans"], metadata["col_spans"])
                centered = all(max(axis) - min(axis) <= 1 for axis in spans)
                mixed_axes += len(spans[0]) > 1 and len(spans[1]) > 1 and ((max(spans[0]) - min(spans[0]) <= 1) != (max(spans[1]) - min(spans[1]) <= 1))
                if section == "train":
                    centered_train += centered
                    off_train += not centered
                else:
                    centered_test += centered
                    off_test += not centered
                signatures = [(mask.shape, tuple(bool(value) for value in mask.flat)) for mask in metadata["masks"]]
                repeated += len(signatures) != len(set(signatures)) or (section == "test" and any(signature in train_masks for signature in signatures))
                if section == "train":
                    train_masks.update(signatures)
                    train_payloads.update(metadata["payload_colors"])
                else:
                    reused_test_motifs += any(signature in train_masks for signature in signatures)
                    new_test_motifs += any(signature not in train_masks for signature in signatures)
                    novel_count = len(set(metadata["payload_colors"]) - train_payloads)
                    familiar_tests += novel_count == 0
                    one_novel_tests += novel_count == 1
                    multiple_novel_tests += novel_count >= 2
                for mask in metadata["masks"]:
                    symmetric += _orbit_size(mask) < 8
                    disconnected += _component_count(mask) > 1
                    solid += bool(mask.all())
                    sparse += int(mask.sum()) <= max(mask.shape)
                    internal_blank += any(not mask[row].any() for row in range(1, mask.shape[0] - 1)) or any(not mask[:, col].any() for col in range(1, mask.shape[1] - 1))
        varying_backgrounds += len({item["background"] for item in puzzle_metadata}) > 1
        varying_dividers += len({item["divider"] for item in puzzle_metadata}) > 1
        varying_palettes += len({item["payload_colors"] for item in puzzle_metadata}) > 1
        role_switches += any(
            first["background"] == second["divider"]
            or first["divider"] == second["background"]
            or first["background"] in second["payload_colors"]
            or first["divider"] in second["payload_colors"]
            for first in puzzle_metadata for second in puzzle_metadata if first is not second
        )
    _check(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts are too narrow")
    _check(len(puzzle_keys) >= 40 and len(input_sizes) >= 40 and len(palettes) >= 40, "puzzles lack structural or color diversity")
    _check(len(layouts) >= 5 and len(tile_sizes) >= 8, "layout or tile diversity is too narrow")
    _check(min(symmetric, repeated, disconnected, solid, sparse, interior, shared, background_zero, centered_train, off_train, centered_test, off_test) > 0, "a review variation is absent")
    _check(min(compact, long_narrow, maximum_extent, internal_blank, same_offsets, familiar_tests, one_novel_tests, multiple_novel_tests) > 0, "a requested structural or palette regime is absent")
    _check(min(one_dimensional, multiple_separators, combined_extended, mixed_axes, reused_test_motifs, new_test_motifs) > 0, "a layout or motif combination is absent")
    _check(min(varying_backgrounds, varying_dividers, varying_palettes, role_switches) >= 20, "color roles do not vary within enough puzzles")
    for seed, train_count, test_count in ((9001, 2, 1), (9002, 10, 1), (9003, 10, 4), (9101, None, 1), (9102, None, 4), (9103, 2, None), (9104, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            _check(len(puzzle["train"]) == train_count, "train count not honored")
        if test_count is not None:
            _check(len(puzzle["test"]) == test_count, "test count not honored")
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(9200, train_count, test_count)
        except ValueError:
            continue
        raise AssertionError("illegal count request was accepted")
    corrupted = generate_puzzle(9300, 3, 1)
    altered = corrupted["test"][0]["output"].copy()
    altered[0, 0] = (int(altered[0, 0]) + 1) % 10
    corrupted["test"][0]["output"] = altered
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        return
    raise AssertionError("validator accepted an incorrect output")
