"""Generate variable-resolution lattice compression puzzles."""

import hashlib

import numpy as np


# Complete input/output pairs from the four source demonstrations, encoded as
# digests so square canvases and other source-like geometry remain available.
_SOURCE_PAIR_DIGESTS = frozenset({
    "a34aae9a2026f77ff6e795b55f25416c522aef92aa469e116b679260d38bd8be",
    "9df90e7bbbe5cff0e88acf57e6bc941c5d7883aa3e59097f41361c1d37e52846",
    "a38a7c6adcde547083b22cddffe8c61e8bedb7c9f6cecd90fc16aaf49118d4f7",
    "66df21e88f68b15aca9e5e410f5a2b2e21e8b7c7240addf8796540f8015f5371",
})


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _checked_count(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    num_train = _checked_count(num_train, "num_train")
    num_test = _checked_count(num_test, "num_test")
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = low + seed % (11 - low)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    return num_train, num_test


def _runs(mask):
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return ()
    result = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        index = int(raw)
        if index != previous + 1:
            result.append((start, previous))
            start = index
        previous = index
    result.append((start, previous))
    return tuple(result)


def _connected(mask):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    first = tuple(int(x) for x in coords[0])
    seen = {first}
    stack = [first]
    height, width = mask.shape
    while stack:
        row, col = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if (0 <= nr < height and 0 <= nc < width
                    and mask[nr, nc] and (nr, nc) not in seen):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _decode_lattice(grid):
    """Parse geometry and read each cell using only its visible input."""
    candidates = []
    for raw_separator in np.unique(grid):
        separator = int(raw_separator)
        rows, cols = np.where(grid == separator)
        top, bottom = int(rows.min()), int(rows.max())
        left, right = int(cols.min()), int(cols.max())
        sub = grid[top:bottom + 1, left:right + 1]
        full_rows = np.all(sub == separator, axis=1)
        full_cols = np.all(sub == separator, axis=0)
        row_bands = _runs(full_rows)
        col_bands = _runs(full_cols)
        if len(row_bands) < 2 or len(col_bands) < 2:
            continue
        if (row_bands[0][0] != 0 or row_bands[-1][1] != len(sub) - 1
                or col_bands[0][0] != 0 or col_bands[-1][1] != sub.shape[1] - 1):
            continue
        structure = full_rows[:, None] | full_cols[None, :]
        if not np.array_equal(sub == separator, structure):
            continue
        row_gaps = tuple((row_bands[i][1] + 1, row_bands[i + 1][0])
                         for i in range(len(row_bands) - 1))
        col_gaps = tuple((col_bands[i][1] + 1, col_bands[i + 1][0])
                         for i in range(len(col_bands) - 1))
        if any(end - start < 3 for start, end in row_gaps + col_gaps):
            continue
        canvas = int(sub[row_gaps[0][0], col_gaps[0][0]])
        if canvas == separator:
            continue
        outside = np.ones(grid.shape, dtype=bool)
        outside[top:bottom + 1, left:right + 1] = False
        if np.any(grid[outside] != canvas):
            continue
        output = np.full((len(row_gaps), len(col_gaps)), canvas, dtype=np.int8)
        patches = []
        valid = True
        for ri, (rs, re) in enumerate(row_gaps):
            for ci, (cs, ce) in enumerate(col_gaps):
                block = sub[rs:re, cs:ce]
                border = np.zeros(block.shape, dtype=bool)
                border[0, :] = border[-1, :] = True
                border[:, 0] = border[:, -1] = True
                if np.any(block[border] != canvas):
                    valid = False
                    break
                colors = set(int(x) for x in np.unique(block)) - {canvas}
                if len(colors) > 1 or separator in colors:
                    valid = False
                    break
                if colors:
                    color = next(iter(colors))
                    mask = block == color
                    if not _connected(mask):
                        valid = False
                        break
                    coords = np.argwhere(mask)
                    min_r, min_c = coords.min(axis=0)
                    max_r, max_c = coords.max(axis=0)
                    area = len(coords)
                    rectangular = area == (max_r - min_r + 1) * (max_c - min_c + 1)
                    patches.append((area, bool(rectangular), int(max_r - min_r + 1),
                                    int(max_c - min_c + 1)))
                    output[ri, ci] = color
            if not valid:
                break
        if valid:
            candidates.append((output, {
                "canvas": canvas,
                "separator": separator,
                "row_bands": row_bands,
                "col_bands": col_bands,
                "row_gaps": row_gaps,
                "col_gaps": col_gaps,
                "horizontal_thicknesses": tuple(e - s + 1 for s, e in row_bands),
                "vertical_thicknesses": tuple(e - s + 1 for s, e in col_bands),
                "row_gap_sizes": tuple(e - s for s, e in row_gaps),
                "col_gap_sizes": tuple(e - s for s, e in col_gaps),
                "margins": (top, left, grid.shape[0] - bottom - 1,
                            grid.shape[1] - right - 1),
                "patches": tuple(patches),
            }))
    _require(len(candidates) == 1, "input must contain exactly one unambiguous lattice")
    return candidates[0]


def _sample_axis(rng, count):
    """Return independent outer/inner bands, compartments, and margins."""
    for _ in range(300):
        if rng.random() < 0.22:
            bands = [int(rng.integers(1, 4))] * (count + 1)
        else:
            bands = [int(x) for x in rng.integers(1, 4, size=count + 1)]
        before, after = (int(x) for x in rng.integers(0, 4, size=2))
        budget = 30 - before - after - sum(bands)
        if budget < 3 * count:
            continue
        sizes = [3] * count
        extra = int(rng.integers(0, min(5 * count, budget - 3 * count) + 1))
        for _ in range(extra):
            eligible = [i for i, size in enumerate(sizes) if size < 8]
            sizes[int(rng.choice(eligible))] += 1
        band_ranges, gap_ranges = [], []
        cursor = before
        for i, band in enumerate(bands):
            band_ranges.append((cursor, cursor + band))
            cursor += band
            if i < count:
                gap_ranges.append((cursor, cursor + sizes[i]))
                cursor += sizes[i]
        return cursor + after, tuple(band_ranges), tuple(gap_ranges)
    # A six-compartment axis can have few feasible thick-band combinations.
    # Keep construction total even if an unusually long rejection streak occurs.
    bands = [(i * 4, i * 4 + 1) for i in range(count + 1)]
    gaps = [(i * 4 + 1, i * 4 + 4) for i in range(count)]
    return 4 * count + 1, tuple(bands), tuple(gaps)


def _logical_shape(rng, regime, index):
    if regime == "source_train" and index == 0:
        choices = np.arange(2, 6)
        weights = np.array([5, 5, 3, 1], dtype=float)
    else:
        choices = np.arange(1, 7)
        weights = np.array([2, 5, 5, 4, 3, 1] if regime == "source_train"
                           else [2, 3, 4, 4, 4, 3], dtype=float)
    weights /= weights.sum()
    for _ in range(20):
        rows = int(rng.choice(choices, p=weights))
        cols = int(rng.choice(choices, p=weights))
        if rows * cols > 1:
            return rows, cols
    return 2, 2


def _positions(rng, rows, cols, occupancy):
    all_positions = [(r, c) for r in range(rows) for c in range(cols)]
    if occupancy < rows * cols and rng.random() < 0.34:
        exclusions = []
        if rows > 1:
            exclusions.append(("row", int(rng.integers(rows))))
        if cols > 1:
            exclusions.append(("col", int(rng.integers(cols))))
        if exclusions:
            axis, index = exclusions[int(rng.integers(len(exclusions)))]
            eligible = [(r, c) for r, c in all_positions
                        if (r if axis == "row" else c) != index]
            if occupancy <= len(eligible):
                all_positions = eligible
    if rng.random() < 0.37:
        center_r, center_c = all_positions[int(rng.integers(len(all_positions)))]
        all_positions.sort(key=lambda rc: (
            abs(rc[0] - center_r) + abs(rc[1] - center_c), rng.random()))
        return all_positions[:occupancy]
    chosen = rng.choice(len(all_positions), size=occupancy, replace=False)
    return [all_positions[int(i)] for i in chosen]


def _paint_patch(block, color, rng):
    height, width = block.shape
    if height >= 4 and width >= 4 and rng.random() < 0.13:
        top = int(rng.integers(1, height - 2))
        left = int(rng.integers(1, width - 2))
        shape = ((0, 0), (1, 0), (1, 1), (0, 1))
        omitted = int(rng.integers(4))
        for i, (dr, dc) in enumerate(shape):
            if i != omitted:
                block[top + dr, left + dc] = color
        return
    patch_height = int(rng.integers(1, height - 1))
    patch_width = int(rng.integers(1, width - 1))
    top = int(rng.integers(1, height - patch_height))
    left = int(rng.integers(1, width - patch_width))
    block[top:top + patch_height, left:left + patch_width] = color


def _choose_roles(rng, history, color_pool):
    if history and rng.random() < 0.48:
        old_canvas, old_separator = history[int(rng.integers(len(history)))]
        if rng.random() < 0.48:
            return old_canvas, old_separator
        return old_separator, old_canvas
    selected = rng.choice(color_pool, size=2, replace=False)
    return int(selected[0]), int(selected[1])


def _occupancy(rng, cells, regime, index, contrast):
    if regime == "source_train" and index < 2:
        if (index == 0) == contrast:
            maximum = max(1, cells // 3)
            return int(rng.integers(1, maximum + 1))
        minimum = max(1, (2 * cells) // 3)
        return int(rng.integers(minimum, cells))
    if regime == "target_test" and rng.random() < 0.38:
        minimum = max(1, (2 * cells) // 3)
        return int(rng.integers(minimum, cells + 1))
    return int(rng.integers(1, cells + 1))


def _payload_colors(rng, available, occupancy):
    maximum = min(occupancy, len(available))
    distinct = int(rng.integers(1, maximum + 1))
    base = [int(x) for x in rng.choice(available, size=distinct, replace=False)]
    colors = base + [base[int(rng.integers(distinct))]
                     for _ in range(occupancy - distinct)]
    rng.shuffle(colors)
    return colors


def generate_pair(rng, regime, index, roles, color_pool, contrast):
    rows, cols = _logical_shape(rng, regime, index)
    height, row_bands, row_gaps = _sample_axis(rng, rows)
    width, col_bands, col_gaps = _sample_axis(rng, cols)
    canvas, separator = roles
    cells = rows * cols
    occupancy = _occupancy(rng, cells, regime, index, contrast)
    positions = _positions(rng, rows, cols, occupancy)
    available = [x for x in color_pool if x not in roles]
    colors = _payload_colors(rng, available, occupancy)
    output = np.full((rows, cols), canvas, dtype=np.int8)
    grid = np.full((height, width), canvas, dtype=np.int8)
    lattice_top, lattice_bottom = row_bands[0][0], row_bands[-1][1]
    lattice_left, lattice_right = col_bands[0][0], col_bands[-1][1]
    for start, end in row_bands:
        grid[start:end, lattice_left:lattice_right] = separator
    for start, end in col_bands:
        grid[lattice_top:lattice_bottom, start:end] = separator
    for (ri, ci), color in zip(positions, colors):
        output[ri, ci] = color
        rs, re = row_gaps[ri]
        cs, ce = col_gaps[ci]
        _paint_patch(grid[rs:re, cs:ce], color, rng)
    return grid, output


def _grid_key(grid):
    return tuple(grid.shape), grid.tobytes()


def _pair_digest(input_grid, output_grid):
    data = (bytes(input_grid.shape) + input_grid.astype(np.uint8).tobytes()
            + bytes(output_grid.shape) + output_grid.astype(np.uint8).tobytes())
    return hashlib.sha256(data).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    inputs = set()
    pairs = set()
    training_metadata = []
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain exactly input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         f"{name} must be a 2D NumPy array")
                _require(np.issubdtype(grid.dtype, np.integer),
                         f"{name} must have integer dtype")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                         f"{name} has illegal dimensions")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         f"{name} has illegal colors")
            input_grid, output_grid = pair["input"], pair["output"]
            key = _grid_key(input_grid)
            pair_key = key, _grid_key(output_grid)
            _require(key not in inputs and pair_key not in pairs,
                     "duplicate input or pair")
            inputs.add(key)
            pairs.add(pair_key)
            _require(_pair_digest(input_grid, output_grid) not in _SOURCE_PAIR_DIGESTS,
                     "generated pair exactly copies source")
            decoded, metadata = _decode_lattice(input_grid)
            _require(np.array_equal(decoded, output_grid),
                     "output differs from visible lattice compression")
            _require(input_grid.shape != output_grid.shape,
                     "input must be physically larger than output")
            _require(len(metadata["patches"]) > 0, "pair needs a visible payload")
            if split == "train":
                training_metadata.append(metadata)
                if index == 0:
                    _require(output_grid.shape[0] >= 2 and output_grid.shape[1] >= 2,
                             "first training pair must teach both lattice axes")
                if index < 2:
                    _require(np.any(output_grid == metadata["canvas"]),
                             "first two training pairs must expose empty cells")
    # No hidden color map or geometry convention is shared between pairs. Each
    # input visibly supplies its own canvas, separator bands, and payload colors.
    _require(len(training_metadata) >= 2, "insufficient training demonstrations")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = tuple(int(x) for x in rng.permutation(10))
    train_pool_size = int(rng.integers(5, 10))
    train_pool = palette[:train_pool_size]
    contrast = bool(rng.integers(2))
    child_seeds = rng.integers(0, np.iinfo(np.int64).max,
                               size=train_count + test_count, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    history = []
    used = set()
    cursor = 0
    for regime, split, count in (("source_train", "train", train_count),
                                 ("target_test", "test", test_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(child_seeds[cursor]))
            cursor += 1
            color_pool = train_pool if split == "train" else palette
            for _ in range(80):
                roles = _choose_roles(pair_rng, history, color_pool)
                input_grid, output_grid = generate_pair(
                    pair_rng, regime, index, roles, color_pool, contrast)
                if _grid_key(input_grid) in used:
                    continue
                if _pair_digest(input_grid, output_grid) in _SOURCE_PAIR_DIGESTS:
                    continue
                try:
                    decoded, _ = _decode_lattice(input_grid)
                except AssertionError:
                    continue
                if not np.array_equal(decoded, output_grid):
                    continue
                puzzle[split].append({"input": input_grid, "output": output_grid})
                used.add(_grid_key(input_grid))
                history.append(roles)
                break
            else:
                raise RuntimeError("could not construct a unique valid lattice")
    validate_generated_puzzle(puzzle)
    return puzzle


def _same_puzzle(first, second):
    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for a, b in zip(first[split], second[split]):
            if any(not np.array_equal(a[name], b[name]) for name in ("input", "output")):
                return False
    return True


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4,
             "train counts lack diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3,
             "test counts lack diversity")
    shapes, input_shapes, layouts, roles = set(), set(), set(), set()
    properties = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_roles = []
        train_colors = set()
        for pair in puzzle["train"]:
            _, meta = _decode_lattice(pair["input"])
            train_roles.append((meta["canvas"], meta["separator"]))
            train_colors.update(int(x) for x in np.unique(pair["input"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                decoded, meta = _decode_lattice(grid)
                _require(np.array_equal(decoded, output),
                         "independent input decoding failed")
                shapes.add(tuple(output.shape))
                input_shapes.add(tuple(grid.shape))
                roles.add((meta["canvas"], meta["separator"]))
                layouts.add((tuple(output.shape), meta["horizontal_thicknesses"],
                             meta["vertical_thicknesses"], meta["row_gap_sizes"],
                             meta["col_gap_sizes"], meta["margins"]))
                if grid.shape[0] == grid.shape[1]:
                    properties.add("square")
                if 3 in meta["horizontal_thicknesses"] + meta["vertical_thicknesses"]:
                    properties.add("width3")
                if (len(set(meta["horizontal_thicknesses"])) == 1
                        and len(set(meta["vertical_thicknesses"])) == 1):
                    properties.add("uniform_bands")
                if len(set(meta["row_gap_sizes"])) == 1 or len(set(meta["col_gap_sizes"])) == 1:
                    properties.add("equal_gaps")
                if 0 in meta["margins"] and any(x > 0 for x in meta["margins"]):
                    properties.add("mixed_margins")
                if any(a >= 6 and rectangle for a, rectangle, _, _ in meta["patches"]):
                    properties.add("large_rectangle")
                if any(not rectangle for _, rectangle, _, _ in meta["patches"]):
                    properties.add("irregular")
                payload = output[output != meta["canvas"]]
                if len(payload) > len(set(int(x) for x in payload)):
                    properties.add("repeat")
                if any(np.all(output[r, :] == meta["canvas"]) for r in range(output.shape[0])):
                    properties.add("empty_row")
                if any(np.all(output[:, c] == meta["canvas"]) for c in range(output.shape[1])):
                    properties.add("empty_col")
                if split == "test" and any(int(x) not in train_colors for x in payload):
                    properties.add("unseen_test_payload")
                if split == "test" and output.shape in [p["output"].shape for p in puzzle["train"]]:
                    properties.add("matching_test_shape")
        if len(set(train_roles)) < len(train_roles):
            properties.add("stable_roles")
        if any((b, a) in train_roles for a, b in train_roles):
            properties.add("role_swap")
    _require(len(shapes) >= 12 and len(input_shapes) >= 35 and len(layouts) >= 100,
             "insufficient geometric diversity")
    _require(len(roles) >= 35, "insufficient color-role diversity")
    required = {"square", "width3", "uniform_bands", "equal_gaps",
                "mixed_margins", "large_rectangle", "irregular", "repeat",
                "empty_row", "empty_col", "unseen_test_payload",
                "matching_test_shape", "stable_roles", "role_swap"}
    _require(required <= properties, f"missing review modes: {required - properties}")
    for seed, train_count, test_count in ((1001, 2, 1), (1002, 10, 1),
                                          (1003, 10, 4)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        _require((len(puzzle["train"]), len(puzzle["test"])) ==
                 (train_count, test_count), "explicit counts ignored")
    for seed, nt, ne in ((2001, None, 1), (2002, None, 4),
                         (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, nt, ne)
        _require((nt is None or len(puzzle["train"]) == nt)
                 and (ne is None or len(puzzle["test"]) == ne),
                 "partial count ignored")
    a = generate_puzzle(314159, 7, 3)
    _require(_same_puzzle(a, generate_puzzle(314159, 7, 3)),
             "generation is not deterministic")
    _require(not _same_puzzle(a, generate_puzzle(314160, 7, 3)),
             "adjacent seeds produce identical puzzles")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 2.5},
                   {"num_test": True}):
        try:
            generate_puzzle(9999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {kwargs}")
