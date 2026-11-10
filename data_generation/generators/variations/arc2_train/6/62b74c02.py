"""Generate full-height edge-copy and row-connector ARC puzzles."""

import numpy as np


def _rows(rows):
    return tuple(tuple(int(c) for c in row) for row in rows)


_SOURCE_PAIR_KEYS = frozenset(
    (
        (_rows(("323000000000", "121000000000", "323000000000", "131000000000")),
         _rows(("323333333323", "121111111121", "323333333323", "131111111131"))),
        (_rows(("8280000000", "1810000000", "1810000000")),
         _rows(("8288888828", "1811111181", "1811111181"))),
        (_rows(("23820000000000", "28820000000000", "28320000000000")),
         _rows(("23822222222382", "28822222222882", "28322222222832"))),
        (_rows(("38830000000000", "21120000000000", "13310000000000", "21120000000000")),
         _rows(("38833333333883", "21122222222112", "13311111111331", "21122222222112"))),
    )
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid_key(grid):
    return tuple(tuple(int(value) for value in row) for row in grid)


def _checked_count(value, name, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        minimum = max(2, test + 1)
        train = minimum + seed % (11 - minimum)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _layout(rng, informative=False):
    height = int(rng.integers(3 if informative else 2, 9))
    seed_width = int(rng.integers(4 if informative else 3, 9))
    gap = int(rng.integers(1, 15))
    return height, seed_width, gap


def _different(rng, colors, excluded):
    options = [color for color in colors if color != excluded]
    return int(options[int(rng.integers(len(options)))])


def _sample_tile(rng, height, seed_width, colors, uniform, symmetric, informative):
    colors = tuple(int(c) for c in colors)
    tile = np.empty((height, seed_width), dtype=np.int8)
    if uniform:
        boundaries = np.full(height, int(rng.choice(colors)), dtype=np.int8)
    else:
        boundaries = rng.choice(colors, size=height).astype(np.int8)
        if informative and np.unique(boundaries).size == 1:
            boundaries[1] = _different(rng, colors, int(boundaries[0]))
    tile[:, 0] = boundaries
    tile[:, -1] = boundaries

    interior_width = seed_width - 2
    for row in range(height):
        boundary = int(boundaries[row])
        mode = int(rng.integers(10))
        if row and not informative and mode < 2:
            # An exact repeated row remains a legitimate seed motif.
            tile[row, :] = tile[row - 1, :]
            boundaries[row] = boundaries[row - 1]
            continue
        if mode == 2:
            interior = np.full(interior_width, boundary, dtype=np.int8)
        elif mode == 3:
            motif = rng.choice(colors, size=min(2, interior_width))
            interior = np.resize(motif, interior_width).astype(np.int8)
        else:
            interior = rng.choice(colors, size=interior_width).astype(np.int8)
        if symmetric or mode == 4:
            interior[interior_width // 2 + interior_width % 2:] = interior[:interior_width // 2][::-1]
        tile[row, 1:-1] = interior

    if informative:
        # One training pair jointly demonstrates the row rule and exact,
        # unreflected translation, even when only two trains are requested.
        tile[0, 1] = _different(rng, colors, int(tile[0, -2]))
        if np.unique(tile[:, 0]).size == 1:
            tile[1, 0] = tile[1, -1] = _different(rng, colors, int(tile[0, 0]))

    if not np.any(tile[:, 1:-1] != tile[:, :1]):
        tile[0, 1] = _different(rng, colors, int(tile[0, 0]))
        if symmetric:
            tile[0, -2] = tile[0, 1]

    if uniform and height > 1 and np.array_equal(tile[0], tile[1]):
        tile[1, 1] = _different(rng, colors, int(tile[0, 1]))
        if symmetric:
            tile[1, -2] = tile[1, 1]

    # Solid rows are valid, but keep them occasional and surrounded by
    # rows whose copied interiors remain visible.
    flat_row = int(rng.integers(1 if informative else 0, height)) if rng.random() < 0.30 else None
    for row in range(height):
        boundary = int(tile[row, 0])
        if row == flat_row:
            tile[row, 1:-1] = boundary
        elif np.all(tile[row, 1:-1] == boundary):
            tile[row, 1] = _different(rng, colors, boundary)
            if symmetric:
                tile[row, -2] = tile[row, 1]
    return tile


def _build_pair(rng, layout, background, colors, *, informative=False, test=False):
    height, seed_width, gap = layout
    uniform = not informative and rng.random() < (0.32 if test else 0.21)
    symmetric = not informative and rng.random() < (0.34 if test else 0.23)
    tile = _sample_tile(rng, height, seed_width, colors, uniform, symmetric, informative)
    width = 2 * seed_width + gap
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[:, :seed_width] = tile
    output_grid = input_grid.copy()
    output_grid[:, seed_width:width - seed_width] = tile[:, :1]
    output_grid[:, width - seed_width:] = tile
    return {"input": input_grid, "output": output_grid}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)

    # The transformation has no hidden palette convention. This construction
    # choice only controls whether literal colors happen to recur across pairs.
    shared_palette = bool(rng.random() < 0.22)
    if shared_palette:
        common_background = int(rng.integers(10))
        available = [c for c in range(10) if c != common_background]
        common_colors = tuple(int(c) for c in rng.choice(available, size=int(rng.integers(4, 7)), replace=False))
    else:
        background_order = [int(c) for c in rng.permutation(10)]
    # Some puzzles echo the source's richer test palette, while most allow
    # train and test color counts to overlap freely.
    train_color_cap = 3 if rng.random() < 0.25 else 5

    puzzle = {"train": [], "test": []}
    used_inputs = set()
    previous_train_layouts = []
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            informative = section == "train" and index == 0
            pair_number = len(puzzle["train"]) + len(puzzle["test"])
            background = common_background if shared_palette else background_order[pair_number % 10]
            available = common_colors if shared_palette else tuple(c for c in range(10) if c != background)
            for attempt in range(128):
                if section == "train" and index > 1 and attempt == 0 and rng.random() < 0.40:
                    layout = previous_train_layouts[int(rng.integers(len(previous_train_layouts)))]
                else:
                    layout = _layout(rng, informative)
                height, seed_width, _ = layout
                capacity = height * (seed_width - 1)
                maximum_colors = train_color_cap if section == "train" else 5
                color_count = int(rng.integers(2, min(maximum_colors, len(available), capacity) + 1))
                colors = tuple(int(c) for c in rng.choice(available, size=color_count, replace=False))
                pair = _build_pair(rng, layout, background, colors,
                                   informative=informative, test=section == "test")
                input_key = _grid_key(pair["input"])
                pair_key = (input_key, _grid_key(pair["output"]))
                if input_key not in used_inputs and pair_key not in _SOURCE_PAIR_KEYS:
                    break
            else:
                raise RuntimeError("Unable to construct a unique pair")
            used_inputs.add(input_key)
            puzzle[section].append(pair)
            if section == "train":
                previous_train_layouts.append(layout)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid, label):
    _require(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _require(grid.ndim == 2, f"{label} must be two-dimensional")
    height, width = grid.shape
    _require(1 <= height <= 30 and 1 <= width <= 30, f"{label} dimensions out of range")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} must have integer dtype")
    _require(0 <= int(np.min(grid)) and int(np.max(grid)) <= 9, f"{label} colors out of range")


def _inspect_pair(pair, label):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, f"{label} schema")
    inp, out = pair["input"], pair["output"]
    _check_grid(inp, f"{label} input")
    _check_grid(out, f"{label} output")
    _require(inp.shape == out.shape, f"{label} dimensions changed")
    height, width = inp.shape
    background = int(inp[0, -1])
    _require(np.all(inp[:, -1] == background), f"{label} suffix is not uniform")
    occupied = np.any(inp != background, axis=0)
    _require(bool(occupied[0]), f"{label} has no left seed")
    seed_width = int(np.argmax(~occupied)) if np.any(~occupied) else width
    _require(3 <= seed_width <= 8, f"{label} seed width out of regime")
    _require(2 <= height <= 8, f"{label} height out of regime")
    _require(np.all(occupied[:seed_width]) and not np.any(occupied[seed_width:]), f"{label} seed prefix invalid")
    _require(width > 2 * seed_width, f"{label} missing connector gap")
    gap = width - 2 * seed_width
    _require(1 <= gap <= 14, f"{label} connector length out of regime")
    _require(np.all(inp[:, :seed_width] != background), f"{label} seed contains background")
    _require(np.all(inp[:, seed_width:] == background), f"{label} suffix contains an object")
    tile = inp[:, :seed_width]
    _require(np.array_equal(tile[:, 0], tile[:, -1]), f"{label} boundary mismatch")
    _require(np.any(tile[:, 1:-1] != tile[:, :1]), f"{label} seed has no visible interior")
    expected = inp.copy()
    expected[:, seed_width:width - seed_width] = tile[:, :1]
    expected[:, width - seed_width:] = tile
    _require(np.array_equal(out, expected), f"{label} violates copy-and-fill rule")
    _require(not np.array_equal(inp, out), f"{label} output unchanged")
    _require(not np.any(out == background), f"{label} output retains background")
    pair_key = (_grid_key(inp), _grid_key(out))
    _require(pair_key not in _SOURCE_PAIR_KEYS, f"{label} copies source pair")
    return {"height": height, "seed_width": seed_width, "gap": gap,
            "background": background, "tile": tile,
            "colors": frozenset(int(c) for c in np.unique(tile)),
            "row_specific": np.unique(tile[:, 0]).size > 1,
            "asymmetric": np.any(tile[:, 1:-1] != tile[:, 1:-1][:, ::-1])}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "Pair lists required")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "Pair counts out of bounds")
    seen_inputs = set()
    train_metadata = []
    for section in ("train", "test"):
        for index, pair in enumerate(puzzle[section]):
            metadata = _inspect_pair(pair, f"{section}[{index}]")
            key = _grid_key(pair["input"])
            _require(key not in seen_inputs, "Duplicate input")
            seen_inputs.add(key)
            if section == "train":
                train_metadata.append(metadata)
    _require(any(m["row_specific"] and m["asymmetric"] for m in train_metadata),
             "Training must jointly show per-row filling and exact translation")


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[section]) for section in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "Puzzles repeat across seeds")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "Train counts lack variety")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "Test counts lack variety")
    heights, widths, gaps, backgrounds, color_counts = set(), set(), set(), set(), set()
    uniform_count = symmetric_count = flat_count = repeated_count = 0
    same_connector_interiors = uniform_symmetric = source_like = 0
    test_smaller = test_between = test_larger = 0
    test_richer = test_not_richer = test_new_color = test_familiar = 0
    structures = set()
    background_variation = gap_variation = repeated_layouts = 0
    height_variation = width_variation = palette_variation = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        metadata = []
        train_metadata = []
        test_metadata = []
        for section in ("train", "test"):
            for index, pair in enumerate(puzzle[section]):
                m = _inspect_pair(pair, f"self-test {section}[{index}]")
                metadata.append(m)
                (train_metadata if section == "train" else test_metadata).append(m)
                heights.add(m["height"])
                widths.add(m["seed_width"])
                gaps.add(m["gap"])
                backgrounds.add(m["background"])
                color_counts.add(len(m["colors"]))
                tile = m["tile"]
                uniform_count += not m["row_specific"]
                symmetric_count += m["seed_width"] >= 4 and not m["asymmetric"]
                flat_count += int(np.sum(np.all(tile == tile[:, :1], axis=1)))
                repeated_count += len({_grid_key(tile[i:i + 1]) for i in range(len(tile))}) < len(tile)
                same_connector_interiors += any(
                    tile[i, 0] == tile[j, 0] and not np.array_equal(tile[i], tile[j])
                    for i in range(len(tile)) for j in range(i + 1, len(tile))
                )
                uniform_symmetric += not m["row_specific"] and m["seed_width"] >= 4 and not m["asymmetric"]
                source_like += (m["height"] in (3, 4) and m["seed_width"] in (3, 4)
                                and 10 <= 2 * m["seed_width"] + m["gap"] <= 14)
                # Shape and normalized pattern, rather than literal colors.
                mapping = {}
                pattern = tuple(mapping.setdefault(int(v), len(mapping)) for v in tile.flat)
                structures.add((m["height"], m["seed_width"], m["gap"], pattern))
        background_variation += len({m["background"] for m in metadata}) > 1
        gap_variation += len({m["gap"] for m in metadata}) > 1
        height_variation += len({m["height"] for m in metadata}) > 1
        width_variation += len({m["seed_width"] for m in metadata}) > 1
        palette_variation += len({len(m["colors"]) for m in metadata}) > 1
        repeated_layouts += len({(m["height"], m["seed_width"], m["gap"]) for m in metadata}) < len(metadata)
        training_colors = set().union(*(m["colors"] for m in train_metadata))
        training_areas = [m["height"] * (2 * m["seed_width"] + m["gap"]) for m in train_metadata]
        largest_training_palette = max(len(m["colors"]) for m in train_metadata)
        for m in test_metadata:
            area = m["height"] * (2 * m["seed_width"] + m["gap"])
            test_smaller += area < min(training_areas)
            test_between += min(training_areas) <= area <= max(training_areas)
            test_larger += area > max(training_areas)
            test_richer += len(m["colors"]) > largest_training_palette
            test_not_richer += len(m["colors"]) <= largest_training_palette
            test_new_color += bool(m["colors"] - training_colors)
            test_familiar += not bool(m["colors"] - training_colors)
    _require(len(heights) >= 6 and len(widths) >= 5 and len(gaps) >= 12,
             "Dimensions or gaps lack diversity")
    _require(len(backgrounds) >= 8 and len(color_counts) >= 3,
             "Color roles lack diversity")
    _require(uniform_count >= 8 and symmetric_count >= 8 and flat_count >= 8 and repeated_count >= 8,
             "Seed motifs lack diversity")
    _require(background_variation >= 20 and gap_variation >= 35,
             "Within-puzzle pair variation is too narrow")
    _require(height_variation >= 35 and width_variation >= 35 and palette_variation >= 30,
             "Within-puzzle geometry or palette variety is too narrow")
    _require(repeated_layouts >= 15 and source_like >= 3,
             "Repeated and source-like layouts are missing")
    _require(same_connector_interiors >= 50 and uniform_symmetric >= 5,
             "Compatible seed motifs are missing")
    _require(min(test_smaller, test_between, test_larger) >= 5,
             "Test dimensions do not span training sizes")
    _require(test_richer >= 2 and test_not_richer >= 40 and
             test_new_color >= 5 and test_familiar >= 40,
             "Test palette regimes lack variety")
    _require(len(structures) >= 150, "Normalized seed structures lack diversity")

    for seed, train_count, test_count in ((777, None, None), (778, 7, 3),
                                          (1000, 2, 1), (1001, 10, 1), (1002, 10, 4),
                                          (2000, None, 1), (2001, None, 4),
                                          (2002, 2, None), (2003, 10, None)):
        first = generate_puzzle(seed, train_count, test_count)
        second = generate_puzzle(seed, train_count, test_count)
        _require(_puzzle_key(first) == _puzzle_key(second), "Generation is nondeterministic")
        validate_generated_puzzle(first)
        if train_count is not None:
            _require(len(first["train"]) == train_count, "Requested train count ignored")
        if test_count is not None:
            _require(len(first["test"]) == test_count, "Requested test count ignored")
    implicit = generate_puzzle(779)
    explicit = generate_puzzle(779, len(implicit["train"]), len(implicit["test"]))
    _require(_puzzle_key(implicit) == _puzzle_key(explicit), "Implicit counts change RNG state")
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 3, "num_test": 4},
                    {"num_train": True, "num_test": 1}):
        try:
            generate_puzzle(3000, **request)
        except ValueError:
            continue
        raise AssertionError(f"Illegal count request accepted: {request}")
