"""Scattered tiles compacted into a separator-spaced spatial montage."""

from itertools import combinations
import hashlib

import numpy as np


_SOURCE_PAIR_DIGESTS = {
    "edda96aa4d6a09cb6c2e1b03beb02b310881e8e167c61a620adf143f8c741760",
    "994e571d1890c6d04e5a2624b162ce883bd604cb89edc4c741576257185bbffb",
    "b3c5a585d80f2a3f1cdc263ea5c3d35c77cccb4dc1bbc213ef7abf022206cdbc",
    "0d9ec81c7e5a660ba6834558033b729ab921c938d1418885f0d4b2c23de37bcc",
}


def _validate_count(name, value, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _validate_count("num_train", num_train, 2, 10)
    num_test = _validate_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = seed % 4 + 1
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        minimum = max(2, num_test + 1)
        num_train = minimum + seed % (11 - minimum)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("Counts must satisfy 2 <= train <= 10, 1 <= test <= 4, train > test")
    return num_train, num_test


def _source_digest(pair):
    chunks = []
    for role in ("input", "output"):
        grid = pair[role]
        chunks.append(f"{grid.shape[0]}x{grid.shape[1]}:")
        chunks.append("".join(str(int(x)) for x in grid.flat))
        if role == "input":
            chunks.append("|")
    return hashlib.sha256("".join(chunks).encode()).hexdigest()


def _signature(grid):
    return grid.shape, grid.dtype.str, grid.tobytes()


def _puzzle_signature(puzzle):
    return tuple(_signature(pair[role]) for split in ("train", "test")
                 for pair in puzzle[split] for role in ("input", "output"))


def _random_mask(rng, height, width):
    # A full bounding box is essential even when its colored cells disconnect.
    if height == width == 3 and rng.random() < 0.045:
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = mask[0, 2] = mask[2, 0] = mask[2, 2] = True
        return mask
    mask = rng.random((height, width)) < float(rng.uniform(0.24, 0.86))
    for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]):
        if not np.any(edge):
            edge[int(rng.integers(len(edge)))] = True
    return mask


def _make_tiles(rng, rows, columns, height, width, background, inks):
    masks, tiles = [], []
    for _ in range(rows * columns):
        if masks and rng.random() < 0.22:
            mask = masks[int(rng.integers(len(masks)))].copy()
        else:
            mask = _random_mask(rng, height, width)
        masks.append(mask)
        tile = np.full((height, width), background, dtype=np.int8)
        count = int(np.count_nonzero(mask))
        tile[mask] = inks[0] if len(inks) == 1 else rng.choice(inks, size=count)
        tiles.append(tile)
    return tiles


def _assemble(tiles, rows, columns, height, width, background):
    output = np.full((rows * (height + 1) - 1,
                      columns * (width + 1) - 1), background, dtype=np.int8)
    for row in range(rows):
        for column in range(columns):
            output[row * (height + 1):row * (height + 1) + height,
                   column * (width + 1):column * (width + 1) + width] = tiles[row * columns + column]
    return output


def _dimension_layout(rng, bands, across, tile_length, challenge, compact=False):
    # Band spans include all staggered tiles; neighboring bands never overlap.
    if compact:
        offsets = rng.integers(0, 3, size=(bands, across), dtype=np.int16)
        offsets -= offsets.min(axis=1, keepdims=True)
        before, after = int(rng.integers(0, 2)), int(rng.integers(0, 2))
        starts = [before]
        for band in range(bands - 1):
            starts.append(starts[-1] + tile_length + int(offsets[band].max())
                          + int(rng.integers(1, 3)))
        size = starts[-1] + tile_length + int(offsets[-1].max()) + after
        return offsets + np.asarray(starts, dtype=np.int16)[:, None], size
    for _ in range(24):
        offsets = rng.integers(0, 7, size=(bands, across), dtype=np.int16)
        if rng.random() < 0.17:
            offsets[:] = 0
        elif rng.random() < 0.23:
            offsets[:, int(rng.integers(across))] = 0
        offsets -= offsets.min(axis=1, keepdims=True)
        gaps = [int(rng.integers(1, 7 if challenge else 8)) for _ in range(bands - 1)]
        before = int(rng.integers(0, 5 if challenge else 7))
        after = int(rng.integers(0, 5 if challenge else 7))
        starts = [before]
        for band in range(bands - 1):
            starts.append(starts[-1] + tile_length + int(offsets[band].max()) + gaps[band])
        size = starts[-1] + tile_length + int(offsets[-1].max()) + after
        if size <= 30:
            return offsets + np.asarray(starts, dtype=np.int16)[:, None], size
    # Deterministic fallback for the largest arrays.
    gap, before = (1, 0) if challenge else (3, 2)
    starts = [before + band * (tile_length + gap) for band in range(bands)]
    return np.asarray(starts, dtype=np.int16)[:, None] + np.zeros((bands, across), dtype=np.int16), starts[-1] + tile_length + before


def _pair(rng, configuration, regime, background=None):
    rows, columns, height, width = configuration
    challenge = (rng.random() < (0.68 if regime == "target_test" else 0.39))
    compact = configuration == (2, 2, 3, 3) and rng.random() < 0.28
    if background is None:
        background = int(rng.integers(10))
    ink_pool = [color for color in range(10) if color != background]
    color_count = int(rng.choice((1, 1, 1, 1, 2, 2, 3)))
    inks = tuple(int(x) for x in rng.choice(ink_pool, size=color_count, replace=False))
    tiles = _make_tiles(rng, rows, columns, height, width, background, inks)
    row_anchors, input_height = _dimension_layout(rng, rows, columns, height, challenge, compact)
    column_anchors, input_width = _dimension_layout(rng, columns, rows, width, challenge, compact)
    column_anchors = column_anchors.T
    input_grid = np.full((input_height, input_width), background, dtype=np.int8)
    for row in range(rows):
        for column in range(columns):
            y, x = int(row_anchors[row, column]), int(column_anchors[row, column])
            input_grid[y:y + height, x:x + width] = tiles[row * columns + column]
    return {"input": input_grid,
            "output": _assemble(tiles, rows, columns, height, width, background)}


def _partitions(occupied, count):
    possible = [index + 1 for index, delta in enumerate(np.diff(occupied)) if delta > 1]
    for chosen in combinations(possible, count - 1):
        cuts = (0,) + chosen + (len(occupied),)
        yield tuple((int(occupied[cuts[i]]), int(occupied[cuts[i + 1] - 1]))
                    for i in range(count))


def _decompositions(grid, background, limit=2):
    points = np.argwhere(grid != background)
    if not len(points):
        return []
    row_values, column_values = np.unique(points[:, 0]), np.unique(points[:, 1])
    results = []
    for rows in (2, 3):
        for row_bands in _partitions(row_values, rows):
            row_groups = [points[(points[:, 0] >= lo) & (points[:, 0] <= hi)]
                          for lo, hi in row_bands]
            for columns in (2, 3):
                for column_bands in _partitions(column_values, columns):
                    anchors, shape, valid = [], None, True
                    for r in range(rows):
                        for c in range(columns):
                            lo, hi = column_bands[c]
                            group = row_groups[r]
                            slot = group[(group[:, 1] >= lo) & (group[:, 1] <= hi)]
                            if not len(slot):
                                valid = False
                                break
                            y, x = int(slot[:, 0].min()), int(slot[:, 1].min())
                            current = (int(slot[:, 0].max()) - y + 1,
                                       int(slot[:, 1].max()) - x + 1)
                            if not (2 <= current[0] <= 5 and 2 <= current[1] <= 5):
                                valid = False
                                break
                            if shape is None:
                                shape = current
                            elif current != shape:
                                valid = False
                                break
                            anchors.append((y, x))
                        if not valid:
                            break
                    if valid:
                        height, width = shape
                        tiles = [grid[y:y + height, x:x + width].copy() for y, x in anchors]
                        results.append((rows, columns, height, width, tuple(anchors), tiles))
                        if len(results) >= limit:
                            return results
    return results


def _analyze_pair(pair):
    input_grid, output_grid = pair["input"], pair["output"]
    values, counts = np.unique(input_grid, return_counts=True)
    assert len(values) >= 2, "Input needs background and visible tiles"
    largest = int(np.argmax(counts))
    background = int(values[largest])
    assert counts[largest] > input_grid.size // 2, "Background must be dominant"
    candidates = _decompositions(input_grid, background)
    assert len(candidates) == 1, "Input has no unique complete-tile arrangement"
    rows, columns, height, width, anchors, tiles = candidates[0]
    assert np.array_equal(output_grid,
                          _assemble(tiles, rows, columns, height, width, background)), (
                              "Output must copy each tile unchanged into its spatial slot")
    return {"rows": rows, "columns": columns, "height": height, "width": width,
            "anchors": anchors, "background": background,
            "colors": tuple(int(x) for x in values),
            "tiles": tiles, "input_shape": input_grid.shape}


def _new_configuration(rng):
    options = ((2, 2), (2, 3), (3, 2), (3, 3))
    rows, columns = options[int(rng.integers(len(options)))]
    height, width = int(rng.integers(2, 6)), int(rng.integers(2, 6))
    if (rows, columns, height, width) == (2, 2, 3, 3):
        width = 4
    return rows, columns, height, width


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)

    # Only spatial order and one-cell separators are puzzle-wide conventions.
    configurations = [(2, 2, 3, 3), _new_configuration(rng)]
    for _ in range(num_train - 2):
        configurations.append((2, 2, 3, 3) if rng.random() < 0.28
                              else _new_configuration(rng))
    rng.shuffle(configurations)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_background = int(rng.integers(10))
    second_background = (first_background + int(rng.integers(1, 10))) % 10
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train":
                configuration = configurations[index]
                forced_background = (first_background if index == 0 else
                                     second_background if index == 1 else None)
            else:
                configuration = configurations[int(rng.integers(len(configurations)))]
                forced_background = None
            for _ in range(160):
                pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair = _pair(np.random.default_rng(pair_seed), configuration,
                             "source_train" if split == "train" else "target_test",
                             forced_background)
                signature = _signature(pair["input"])
                if signature in seen or _source_digest(pair) in _SOURCE_PAIR_DIGESTS:
                    continue
                try:
                    _analyze_pair(pair)
                except AssertionError:
                    continue
                seen.add(signature)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("Could not construct a uniquely recoverable pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for role in ("input", "output"):
                grid = pair[role]
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert pair["input"].shape != pair["output"].shape or not np.array_equal(
                pair["input"], pair["output"])
            input_key = _signature(pair["input"])
            pair_key = (input_key, _signature(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert _source_digest(pair) not in _SOURCE_PAIR_DIGESTS
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analyses[split].append(_analyze_pair(pair))
    train_configurations = {(a["rows"], a["columns"], a["height"], a["width"])
                            for a in analyses["train"]}
    assert (2, 2, 3, 3) in train_configurations
    assert len(train_configurations) >= 2, "Training must show size/array variation"
    assert len({a["background"] for a in analyses["train"]}) >= 2
    for a in analyses["test"]:
        assert (a["rows"], a["columns"], a["height"], a["width"]) in train_configurations, (
            "Test tile size and array shape must be demonstrated in training")


def self_test() -> None:
    for seed, train, test in ((17, None, None), (73, 6, 3), (100, 2, 1),
                              (101, 10, 1), (102, 10, 4), (103, None, 1),
                              (104, None, 4), (105, 2, None), (106, 10, None)):
        first, second = generate_puzzle(seed, train, test), generate_puzzle(seed, train, test)
        assert _puzzle_signature(first) == _puzzle_signature(second)
        validate_generated_puzzle(first)
        if train is not None:
            assert len(first["train"]) == train
        if test is not None:
            assert len(first["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(200, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Illegal counts accepted: {kwargs}")

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_signature(puzzle) for puzzle in puzzles}) == 42
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    configurations, input_shapes, output_shapes, backgrounds = set(), set(), set(), set()
    alignments, gaps, margins, palettes, masks, layouts = set(), set(), set(), set(), set(), set()
    compact = repeated = multicolor = larger_stagger = 0
    sparse = dense = asymmetric = corners = rectangular = 0
    edge = full_alignment = square_test = rectangular_test = 0
    source_palette = multi_extended = repeat_extended = 0
    top_aligned_test = top_staggered_test = 0
    bottom_aligned_test = bottom_staggered_test = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze_pair(pair)
                rows, cols, height, width = (a[k] for k in ("rows", "columns", "height", "width"))
                config, anchors = (rows, cols, height, width), a["anchors"]
                configurations.add(config)
                input_shapes.add(a["input_shape"])
                output_shapes.add(pair["output"].shape)
                backgrounds.add(a["background"])
                palettes.add(a["colors"])
                tile_masks = [(tile != a["background"]) for tile in a["tiles"]]
                masks.update((mask.shape, mask.tobytes()) for mask in tile_masks)
                is_repeated = len({tile.tobytes() for tile in a["tiles"]}) < len(a["tiles"])
                repeated += is_repeated
                multicolor += len(a["colors"]) > 2
                compact += max(a["input_shape"]) <= 12
                rectangular += height != width
                source_palette += a["colors"] == (0, 2)
                edge += any(value == 0 for value in (
                    min(y for y, _ in anchors), min(x for _, x in anchors),
                    a["input_shape"][0] - max(y + height for y, _ in anchors),
                    a["input_shape"][1] - max(x + width for _, x in anchors)))
                multi_extended += len(a["colors"]) > 2 and config != (2, 2, 3, 3)
                repeat_extended += is_repeated and config != (2, 2, 3, 3)
                if split == "test":
                    square_test += a["input_shape"][0] == a["input_shape"][1]
                    rectangular_test += a["input_shape"][0] != a["input_shape"][1]
                    if rows == cols == 2:
                        top_aligned_test += anchors[0][0] == anchors[1][0]
                        top_staggered_test += anchors[0][0] != anchors[1][0]
                        bottom_aligned_test += anchors[2][0] == anchors[3][0]
                        bottom_staggered_test += anchors[2][0] != anchors[3][0]
                for mask in tile_masks:
                    density = float(np.mean(mask))
                    sparse += density <= 0.35
                    dense += density >= 0.75
                    asymmetric += not np.array_equal(mask, mask[::-1]) and not np.array_equal(mask, mask[:, ::-1])
                    corners += mask.shape == (3, 3) and np.array_equal(
                        mask, ((1, 0, 1), (0, 0, 0), (1, 0, 1)))
                row_offsets = tuple(anchors[r * cols + c][0] - min(
                    anchors[r * cols + k][0] for k in range(cols))
                    for r in range(rows) for c in range(cols))
                col_offsets = tuple(anchors[r * cols + c][1] - min(
                    anchors[k * cols + c][1] for k in range(rows))
                    for r in range(rows) for c in range(cols))
                alignments.add((max(row_offsets), max(col_offsets)))
                larger_stagger += max(row_offsets + col_offsets) >= 5
                full_alignment += max(row_offsets + col_offsets) == 0
                for r in range(rows - 1):
                    upper = max(anchors[r * cols + c][0] + height - 1 for c in range(cols))
                    lower = min(anchors[(r + 1) * cols + c][0] for c in range(cols))
                    gaps.add(lower - upper - 1)
                for c in range(cols - 1):
                    left = max(anchors[r * cols + c][1] + width - 1 for r in range(rows))
                    right = min(anchors[r * cols + c + 1][1] for r in range(rows))
                    gaps.add(right - left - 1)
                top, left = min(y for y, _ in anchors), min(x for _, x in anchors)
                bottom = a["input_shape"][0] - max(y + height for y, _ in anchors)
                right = a["input_shape"][1] - max(x + width for _, x in anchors)
                margins.add((top, left, bottom, right))
                layouts.add((config, anchors, a["input_shape"]))
    assert len(configurations) >= 18 and len(input_shapes) >= 25
    assert len(output_shapes) >= 12 and len(backgrounds) == 10
    assert len(palettes) >= 35 and len(masks) >= 120 and len(layouts) >= 150
    assert len(alignments) >= 10 and 1 in gaps and max(gaps) >= 5
    assert any(0 in item for item in margins) and len(margins) >= 25
    assert compact >= 3 and repeated >= 8 and multicolor >= 20 and larger_stagger >= 15
    assert {(2, 2), (2, 3), (3, 2), (3, 3)} <= {
        (rows, columns) for rows, columns, _, _ in configurations}
    assert rectangular >= 30 and source_palette >= 1
    assert sparse >= 30 and dense >= 100 and asymmetric >= 200 and corners >= 1
    assert edge >= 50 and full_alignment >= 5
    assert square_test >= 1 and rectangular_test >= 10
    assert top_aligned_test and top_staggered_test
    assert bottom_aligned_test and bottom_staggered_test
    assert multi_extended >= 10 and repeat_extended >= 10
