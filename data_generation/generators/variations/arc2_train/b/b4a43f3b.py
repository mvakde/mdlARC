"""Generate puzzles that decode a block-encoded motif and place it at mask cells."""

import numpy as np


_SOURCE_INPUT_ROWS = {
    ("110011", "110011", "002200", "002200", "332200", "332200",
     "555555", "000000", "002000", "022220", "002000", "000000", "000000"),
    ("113366", "113366", "001100", "001100", "222211", "222211",
     "555555", "000000", "220000", "220000", "002000", "000200", "000000"),
    ("110011", "110011", "001100", "001100", "333333", "333333",
     "555555", "000000", "000000", "002000", "002000", "022200", "000000"),
    ("332200", "332200", "002200", "002200", "006611", "006611",
     "555555", "000000", "020000", "002000", "022220", "000000", "000000"),
    ("001100", "001100", "330033", "330033", "002200", "002200",
     "555555", "202000", "002000", "222222", "002000", "002000", "002000"),
}


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return train, test


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _dimensions(rng, previous=None, change_height=False, change_geometry=False,
                large_mask=False):
    choices = []
    for scale in (2, 3):
        for height in range(2, 6):
            for width in (2, 3):
                mask_width = width * scale
                if width * mask_width > 30:
                    continue
                for mask_height in range(4, 10):
                    if height * mask_height > 30 or height * scale + 1 + mask_height > 30:
                        continue
                    if large_mask and (mask_height < 6 or mask_width < 6):
                        continue
                    shape = (scale, height, width, mask_height)
                    if previous is not None:
                        if change_height and mask_height == previous[3]:
                            continue
                        if change_geometry and shape[:3] == previous[:3]:
                            continue
                    choices.append(shape)
    return choices[int(rng.integers(len(choices)))]


def _motif_palette(rng, roles, cell_count, marker_policy, background_policy):
    background, separator, marker = roles
    include_marker = (marker_policy == "present" or
                      (marker_policy == "random" and bool(rng.integers(2))))
    include_background = (background_policy == "present" or
                          (background_policy == "random" and rng.random() < 0.7))
    palette = []
    if include_background:
        palette.append(background)
    if include_marker:
        palette.append(marker)
    extras = [int(v) for v in rng.permutation(10)
              if int(v) not in (background, separator, marker)]
    maximum = min(5, cell_count)
    minimum = max(2, len(palette))
    target = int(rng.integers(minimum, maximum + 1))
    palette.extend(extras[:target - len(palette)])
    return np.array(palette, dtype=np.int8), include_marker


def _sample_motif(rng, roles, dimensions, marker_policy, background_policy, seen):
    _, height, width, _ = dimensions
    palette, require_marker = _motif_palette(
        rng, roles, height * width, marker_policy, background_policy
    )
    modes = ("scramble", "bottom_band", "right_band", "checker", "mirror", "blocks")
    for attempt in range(240):
        mode = modes[int(rng.integers(len(modes)))]
        motif = rng.choice(palette, size=(height, width)).astype(np.int8)
        if mode == "bottom_band":
            motif[-1, :] = palette[int(rng.integers(len(palette)))]
        elif mode == "right_band":
            motif[:, -1] = palette[int(rng.integers(len(palette)))]
        elif mode == "checker":
            a, b = rng.choice(palette, size=2, replace=False)
            if require_marker and roles[2] not in (a, b):
                a = roles[2]
            motif = np.fromfunction(
                lambda row, col: np.where((row + col) % 2 == 0, a, b),
                (height, width), dtype=int
            ).astype(np.int8)
        elif mode == "mirror" and width == 3:
            motif[:, -1] = motif[:, 0]
        elif mode == "blocks":
            motif[1:, 1:] = motif[1:, :1]
        # The first horizontal and vertical boundaries expose the local scale
        # even when a later row or column is a solid band.
        if motif[0, 0] == motif[0, 1] or motif[0, 0] == motif[1, 0]:
            continue
        colors = set(int(v) for v in np.unique(motif))
        if require_marker and roles[2] not in colors:
            continue
        if len(colors) < 2:
            continue
        key = _grid_key(motif)
        if key not in seen:
            seen.add(key)
            return motif
    # A two-color checker is a deterministic construction if random draws
    # happen to repeat earlier small tiles.
    a = int(roles[2]) if require_marker else int(palette[0])
    b = next(int(v) for v in palette if int(v) != a)
    for offset in range(2):
        motif = np.fromfunction(
            lambda row, col: np.where((row + col + offset) % 2 == 0, a, b),
            (height, width), dtype=int
        ).astype(np.int8)
        key = _grid_key(motif)
        if key not in seen:
            seen.add(key)
            return motif
    raise AssertionError("unable to construct a distinct motif")


def _pick_positions(rng, positions, count):
    if count <= 0:
        return []
    indexes = rng.choice(len(positions), size=count, replace=False)
    return [positions[int(i)] for i in np.atleast_1d(indexes)]


def _mask_scaffold(rng, allowed, mode):
    rows, cols = np.nonzero(allowed)
    row_values = sorted(set(int(v) for v in rows))
    col_values = sorted(set(int(v) for v in cols))
    if mode == "random":
        return set()
    if mode == "horizontal" or mode == "full_row":
        row = row_values[int(rng.integers(len(row_values)))]
        length = len(col_values) if mode == "full_row" else min(
            len(col_values), int(rng.integers(2, min(5, len(col_values)) + 1))
        )
        start = int(rng.integers(len(col_values) - length + 1))
        return {(row, col) for col in col_values[start:start + length]}
    if mode == "vertical":
        col = col_values[int(rng.integers(len(col_values)))]
        length = min(len(row_values), int(rng.integers(2, min(5, len(row_values)) + 1)))
        start = int(rng.integers(len(row_values) - length + 1))
        return {(row, col) for row in row_values[start:start + length]}
    if mode == "cross":
        row = row_values[int(rng.integers(len(row_values)))]
        col = col_values[int(rng.integers(len(col_values)))]
        return {(row, c) for c in col_values} | {(r, col) for r in row_values}
    if mode == "alternating":
        parity = int(rng.integers(2))
        return {(int(r), int(c)) for r, c in zip(rows, cols) if (r + c) % 2 == parity}
    if mode == "hollow" and len(row_values) >= 3 and len(col_values) >= 3:
        top = row_values[int(rng.integers(len(row_values) - 2))]
        bottom_choices = [r for r in row_values if r >= top + 2]
        bottom = bottom_choices[int(rng.integers(len(bottom_choices)))]
        left = col_values[int(rng.integers(len(col_values) - 2))]
        right_choices = [c for c in col_values if c >= left + 2]
        right = right_choices[int(rng.integers(len(right_choices)))]
        return {(r, c) for r in range(top, bottom + 1)
                for c in range(left, right + 1)
                if r in (top, bottom) or c in (left, right)}
    if mode == "block":
        rh = min(len(row_values), int(rng.integers(2, min(4, len(row_values)) + 1)))
        cw = min(len(col_values), int(rng.integers(2, min(4, len(col_values)) + 1)))
        r0 = int(rng.integers(len(row_values) - rh + 1))
        c0 = int(rng.integers(len(col_values) - cw + 1))
        return {(r, c) for r in row_values[r0:r0 + rh]
                for c in col_values[c0:c0 + cw]}
    positions = list(zip((int(v) for v in rows), (int(v) for v in cols)))
    return set(_pick_positions(rng, positions, min(3, len(positions))))


def _sample_mask(rng, roles, dimensions, style, seen):
    background, _, marker = roles
    scale, _, width, height = dimensions
    mask_width = width * scale
    modes = ("random", "horizontal", "vertical", "cross", "alternating",
             "hollow", "block", "full_row")
    for attempt in range(240):
        allowed = np.ones((height, mask_width), dtype=bool)
        interior = style in ("anchor", "anchor_dense", "source_sparse",
                             "reverse_sparse") or (
            style != "boundary" and rng.random() < (0.28 if style == "sparse" else 0.12)
        )
        if interior:
            allowed[0, :] = False
            allowed[-1, :] = False
            allowed[:, 0] = False
            allowed[:, -1] = False
        positions = [tuple(map(int, point)) for point in np.argwhere(allowed)]
        capacity = len(positions)
        if style in ("anchor", "anchor_dense"):
            mode = ("random", "horizontal", "vertical", "block")[int(rng.integers(4))]
            if style == "anchor_dense":
                total = height * mask_width
                low = min(capacity, int(np.ceil(0.34 * total)))
                high = min(capacity, int(np.ceil(0.46 * total)))
            else:
                low, high = 2, min(capacity, 4)
        else:
            mode = modes[int(rng.integers(len(modes)))]
            if interior and mode == "full_row":
                mode = "horizontal"
            total = height * mask_width
            if style in ("sparse", "source_sparse", "reverse_sparse"):
                low, high = 2, max(2, int(np.floor(0.28 * total)))
                if style == "reverse_sparse":
                    high = max(2, int(np.floor(0.20 * total)))
            elif style == "medium":
                low, high = max(2, int(np.ceil(0.22 * total))), int(np.ceil(0.52 * total))
            else:
                low, high = int(np.ceil(0.48 * total)), int(np.floor(0.88 * total))
            high = min(high, capacity, total - 1)
            low = min(low, high)
        scaffold = _mask_scaffold(rng, allowed, mode)
        target = int(rng.integers(low, high + 1))
        if len(scaffold) > target:
            if mode in ("horizontal", "vertical", "cross", "hollow", "block",
                        "full_row"):
                extra = 1 if mode == "full_row" else 0
                target = min(len(scaffold) + extra, capacity, height * mask_width - 1)
        if len(scaffold) > target:
            scaffold = set(_pick_positions(rng, sorted(scaffold), target))
        active = np.zeros_like(allowed)
        for row, col in scaffold:
            active[row, col] = True
        hole = set()
        if mode == "hollow" and scaffold:
            top = min(row for row, _ in scaffold)
            bottom = max(row for row, _ in scaffold)
            left = min(col for _, col in scaffold)
            right = max(col for _, col in scaffold)
            hole = {(row, col) for row in range(top + 1, bottom)
                    for col in range(left + 1, right)}
        available = [point for point in positions if not active[point]
                     and point not in hole]
        target = min(target, int(active.sum()) + len(available))
        for row, col in _pick_positions(rng, available, target - int(active.sum())):
            active[row, col] = True
        if active.sum() < 2 or active.all():
            continue
        key = _grid_key(active)
        if key in seen:
            continue
        seen.add(key)
        return np.where(active, marker, background).astype(np.int8)
    raise AssertionError("unable to construct a distinct mask")


def _construct_output(mask, motif, roles):
    background, _, marker = roles
    tile_height, tile_width = motif.shape
    output = np.full((mask.shape[0] * tile_height, mask.shape[1] * tile_width),
                     background, dtype=np.int8)
    for row, col in np.argwhere(mask == marker):
        row, col = int(row), int(col)
        output[row * tile_height:(row + 1) * tile_height,
               col * tile_width:(col + 1) * tile_width] = motif
    return output


def _assemble_pair(motif, mask, roles, scale):
    upper = np.repeat(np.repeat(motif, scale, axis=0), scale, axis=1)
    separator = np.full((1, mask.shape[1]), roles[1], dtype=np.int8)
    return {"input": np.concatenate((upper, separator, mask), axis=0),
            "output": _construct_output(mask, motif, roles)}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    roles = tuple(int(v) for v in rng.permutation(10)[:3])
    regime = int(rng.integers(3))
    # Masks, motifs, and sizes are local to pairs. Color roles remain shared.
    pair_seeds = rng.integers(0, 2**63 - 1, size=train_count + test_count, dtype=np.int64)
    train, test = [], []
    seen_inputs, seen_motifs, seen_masks = set(), set(), set()
    first_dimensions = None
    for index, pair_seed in enumerate(pair_seeds):
        pair_rng = np.random.default_rng(int(pair_seed))
        is_train = index < train_count
        if index == 0:
            dimensions = _dimensions(pair_rng, large_mask=regime == 1)
            first_dimensions = dimensions
        elif index == 1:
            dimensions = _dimensions(
                pair_rng, first_dimensions, change_height=True,
                change_geometry=bool(pair_rng.integers(2))
            )
        else:
            dimensions = _dimensions(pair_rng)
        scale = dimensions[0]
        if index == 0:
            marker_policy, background_policy = "present", "present"
            style = "anchor_dense" if regime == 1 else "anchor"
        elif index == 1:
            marker_policy, background_policy = "absent", "random"
            if regime == 0:
                style = "source_sparse"
            elif regime == 1:
                style = ("dense", "boundary")[int(pair_rng.integers(2))]
            else:
                style = ("sparse", "medium", "dense", "boundary")[int(pair_rng.integers(4))]
        else:
            marker_policy, background_policy = "random", "random"
            if regime == 0:
                choices = ("source_sparse",) if is_train else ("dense", "boundary")
            elif regime == 1:
                choices = ("dense", "boundary") if is_train else ("reverse_sparse",)
            else:
                choices = ("sparse", "medium", "dense", "boundary")
            style = choices[int(pair_rng.integers(len(choices)))]
        for _ in range(200):
            motif = _sample_motif(pair_rng, roles, dimensions,
                                  marker_policy, background_policy, seen_motifs)
            mask = _sample_mask(pair_rng, roles, dimensions, style, seen_masks)
            pair = _assemble_pair(motif, mask, roles, scale)
            key = _grid_key(pair["input"])
            if key in seen_inputs or _is_source_input(pair["input"]):
                continue
            try:
                _split_input(pair["input"])
            except AssertionError:
                continue
            seen_inputs.add(key)
            (train if is_train else test).append(pair)
            break
        else:
            raise AssertionError("unable to construct a distinct unambiguous pair")
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _is_source_input(grid):
    return grid.shape == (13, 6) and tuple(
        "".join(str(int(v)) for v in row) for row in grid
    ) in _SOURCE_INPUT_ROWS


def _split_input(input_grid):
    height, width = input_grid.shape
    candidates = []
    for row in range(1, height - 1):
        color = int(input_grid[row, 0])
        if (np.all(input_grid[row] == color) and
                np.count_nonzero(input_grid == color) == width):
            candidates.append((row, color))
    assert len(candidates) == 1
    row, separator = candidates[0]
    return input_grid[:row], separator, input_grid[row + 1:]


def _decode_upper(upper):
    horizontal = np.flatnonzero(upper[0, 1:] != upper[0, :-1])
    vertical = np.flatnonzero(upper[1:, 0] != upper[:-1, 0])
    assert horizontal.size and vertical.size
    scale = int(horizontal[0]) + 1
    assert scale == int(vertical[0]) + 1
    assert scale in (2, 3)
    assert upper.shape[0] % scale == 0 and upper.shape[1] % scale == 0
    motif = upper[::scale, ::scale].copy()
    assert 2 <= motif.shape[0] <= 5 and 2 <= motif.shape[1] <= 3
    assert np.array_equal(upper, np.repeat(np.repeat(motif, scale, axis=0), scale, axis=1))
    return motif, scale


def _check_pair(pair, roles):
    inp, output = pair["input"], pair["output"]
    upper, separator, mask = _split_input(inp)
    background, expected_separator, marker = roles
    assert separator == expected_separator
    assert set(int(v) for v in np.unique(mask)) == {background, marker}
    assert mask.shape[0] >= 4
    motif, scale = _decode_upper(upper)
    assert mask.shape[1] == motif.shape[1] * scale
    assert separator not in motif
    expected = _construct_output(mask, motif, roles)
    assert np.array_equal(output, expected)
    assert inp.shape != output.shape or not np.array_equal(inp, output)
    return motif, mask, scale


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    first = puzzle["train"][0]
    upper, separator, mask = _split_input(first["input"])
    assert np.all(mask[0] == mask[0, 0])
    background = int(mask[0, 0])
    colors = set(int(v) for v in np.unique(mask))
    assert len(colors) == 2
    marker = next(v for v in colors if v != background)
    roles = (background, separator, marker)
    seen_inputs, seen_pairs = set(), set()
    motifs, masks = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _grid_key(pair["input"])
            pair_key = input_key, _grid_key(pair["output"])
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert not _is_source_input(pair["input"])
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            motif, current_mask, _ = _check_pair(pair, roles)
            if split == "train":
                motifs.append(motif)
                masks.append(current_mask)
    assert roles[2] in motifs[0] and roles[2] not in motifs[1]
    assert masks[0].shape[0] != masks[1].shape[0]
    assert np.any(masks[0] == background) and np.any(masks[0] == marker)


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[r], y[r]) for x, y in zip(a[s], b[s])
        for r in ("input", "output")
    ) for s in ("train", "test"))


def self_test() -> None:
    role_triples, scales, tile_shapes, mask_heights = set(), set(), set(), set()
    train_edge = train_full = test_sparse = test_interior = test_no_full = 0
    uniform_rows = uniform_cols = palette_changes = 0
    train_full_with_band = test_interior_sparse_with_band = 0
    regimes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        first_upper, separator, first_mask = _split_input(puzzle["train"][0]["input"])
        background = int(first_mask[0, 0])
        marker = next(int(v) for v in np.unique(first_mask) if int(v) != background)
        roles = (background, separator, marker)
        role_triples.add(roles)
        train_densities, test_densities = [], []
        palettes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                motif, mask, scale = _check_pair(pair, roles)
                active = mask == marker
                scales.add(scale)
                tile_shapes.add(motif.shape)
                mask_heights.add(mask.shape[0])
                palettes.add(tuple(int(v) for v in np.unique(motif)))
                uniform_rows += any(np.all(row == row[0]) for row in motif)
                uniform_cols += any(np.all(motif[:, c] == motif[0, c]) for c in range(motif.shape[1]))
                if split == "train":
                    train_densities.append(float(active.mean()))
                    train_edge += bool(np.any(active[0]) or np.any(active[-1]) or
                                       np.any(active[:, 0]) or np.any(active[:, -1]))
                    train_full += bool(np.any(np.all(active, axis=1)))
                    train_full_with_band += bool(
                        np.any(np.all(active, axis=1)) and
                        any(np.all(row == row[0]) for row in motif)
                    )
                else:
                    test_densities.append(float(active.mean()))
                    test_sparse += active.mean() < 0.3
                    interior = not (np.any(active[0]) or np.any(active[-1]) or
                                    np.any(active[:, 0]) or np.any(active[:, -1]))
                    test_interior += interior
                    test_no_full += not np.any(np.all(active, axis=1))
                    test_interior_sparse_with_band += bool(
                        interior and active.mean() < 0.3 and
                        any(np.all(row == row[0]) for row in motif)
                    )
        palette_changes += len(palettes) > 1
        if max(train_densities) < min(test_densities):
            regimes.add("source")
        elif min(train_densities) > max(test_densities):
            regimes.add("reverse")
        else:
            regimes.add("overlap")
    assert len(role_triples) >= 20
    assert scales == {2, 3} and len(tile_shapes) >= 5 and len(mask_heights) >= 4
    assert train_edge >= 10 and train_full >= 3
    assert test_sparse >= 4 and test_interior >= 3 and test_no_full >= 10
    assert uniform_rows >= 10 and uniform_cols >= 10
    assert palette_changes == 42 and regimes == {"source", "reverse", "overlap"}
    assert train_full_with_band >= 3 and test_interior_sparse_with_band >= 2
    a = generate_puzzle(314159, 6, 3)
    b = generate_puzzle(314159, 6, 3)
    assert _puzzles_equal(a, b)
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (1004, 7, 3), (2001, None, 4), (2002, 2, None)):
        sample = generate_puzzle(seed, train, test)
        if train is not None:
            assert len(sample["train"]) == train
        if test is not None:
            assert len(sample["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": True}, {"num_test": 1.5}):
        try:
            generate_puzzle(3001, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count was accepted")
