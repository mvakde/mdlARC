import numpy as np


_SOURCE_ROWS = (
    ("7444447", "4777774", "4777774", "4777774", "4777774", "4779774", "4777774", "4777714", "7444447"),
    ("73333337", "37777773", "37777773", "37773773", "37777773", "37777773", "73333337"),
    ("799999997", "977777779", "977977779", "979797779", "977977779", "977777779", "977777779", "977777779", "799999997"),
    ("7666667", "6778776", "6727276", "6728276", "6777776", "7666667"),
    ("78887", "87778", "88888", "88478", "88748", "78887"),
)


def _key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


_SOURCE_KEYS = frozenset(
    ((len(rows), len(rows[0])), tuple(int(x) for row in rows for x in row))
    for rows in _SOURCE_ROWS
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _counts(seed, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + ((seed // 4 + 3 * seed) % (10 - num_test))
    elif num_train is None:
        num_train = int(num_test) + 1 + seed % (10 - int(num_test))
    elif num_test is None:
        num_test = 1 + seed % min(4, int(num_train) - 1)
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _make_input(height, width, background, frame, core, row, col):
    grid = np.full((height, width), background, dtype=np.int8)
    grid[0, 1:-1] = frame
    grid[-1, 1:-1] = frame
    grid[1:-1, 0] = frame
    grid[1:-1, -1] = frame
    grid[row + 1:row + 4, col + 1:col + 4] = core
    return grid


def _make_output(core, background, frame):
    tile = np.full((5, 5), frame, dtype=np.int8)
    tile[0, 0] = tile[0, -1] = tile[-1, 0] = tile[-1, -1] = background
    tile[1:4, 1:4] = core
    output = np.full((15, 15), background, dtype=np.int8)
    output[2:7, 2:7] = tile
    output[8:13, 8:13] = tile
    return output


def _mask(bits):
    return np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)


_FULL_MASKS = {
    count: tuple(
        bits for bits in range(1, 512)
        if bits.bit_count() == count
        and np.any(_mask(bits)[0, :]) and np.any(_mask(bits)[-1, :])
        and np.any(_mask(bits)[:, 0]) and np.any(_mask(bits)[:, -1])
    ) for count in range(3, 10)
}


def _sample_core(rng, kind, background, frame, palette, flavor="random"):
    core = np.full((3, 3), background, dtype=np.int8)
    accents = [color for color in palette if color not in (background, frame)]
    if kind == "singleton":
        color = frame if not accents or rng.random() < 0.62 else int(rng.choice(accents))
        core[1, 1] = color
        return core
    if kind == "horizontal":
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, :] = True
    elif kind == "vertical":
        mask = np.zeros((3, 3), dtype=bool)
        mask[:, 1] = True
    elif kind == "two_row":
        mask = np.zeros((3, 3), dtype=bool)
        start = int(rng.integers(0, 2))
        mask[start:start + 2, :] = True
    elif kind == "two_col":
        mask = np.zeros((3, 3), dtype=bool)
        start = int(rng.integers(0, 2))
        mask[:, start:start + 2] = True
    elif kind == "two_square":
        mask = np.zeros((3, 3), dtype=bool)
        row, col = int(rng.integers(0, 2)), int(rng.integers(0, 2))
        mask[row:row + 2, col:col + 2] = True
    elif kind == "full":
        if flavor == "dense":
            mask = np.ones((3, 3), dtype=bool)
        elif flavor in ("top", "lower", "column"):
            mask = np.zeros((3, 3), dtype=bool)
            if flavor == "column":
                mask[:, int(rng.integers(0, 3))] = True
                mask[int(rng.integers(0, 3)), 0] = True
                mask[int(rng.integers(0, 3)), 2] = True
            else:
                stripe_row = 0 if flavor == "top" else 2
                mask[stripe_row, :] = True
                mask[2 - stripe_row, int(rng.integers(0, 3))] = True
            free = np.argwhere(~mask)
            if len(free):
                for index in rng.choice(len(free), size=int(rng.integers(0, min(4, len(free)) + 1)), replace=False):
                    row, col = free[int(index)]
                    mask[row, col] = True
        else:
            occupied = int(rng.integers(3, 9 if flavor == "training_anchor" else 10))
            choices = _FULL_MASKS[occupied]
            if flavor == "training_anchor":
                choices = tuple(bits for bits in choices if not np.array_equal(_mask(bits), np.rot90(_mask(bits))) and not np.array_equal(_mask(bits), np.fliplr(_mask(bits))))
            mask = _mask(int(rng.choice(choices)))
    else:
        raise ValueError("unknown core kind")
    positions = np.argwhere(mask)
    if flavor == "frame_only" or not accents:
        colors = [frame]
    elif flavor == "no_frame":
        colors = list(rng.choice(accents, size=int(rng.integers(1, min(3, len(accents)) + 1)), replace=False))
    elif flavor == "training_anchor":
        colors = [frame] + list(rng.choice(accents, size=min(2, len(accents)), replace=False))
    else:
        use_frame = rng.random() < 0.62
        accent_count = int(rng.integers(0 if use_frame else 1, min(3, len(accents)) + 1))
        colors = ([frame] if use_frame else []) + list(rng.choice(accents, size=accent_count, replace=False))
        if not colors:
            colors = [frame]
    rng.shuffle(positions)
    for index, (row, col) in enumerate(positions):
        color = colors[index] if index < len(colors) else colors[int(rng.integers(0, len(colors)))]
        core[int(row), int(col)] = int(color)
    return core


def _carrier(rng, regime, index, kind):
    if regime == "source_train" and index < 2:
        compaction = "both"
    else:
        weights = (0.18, 0.39, 0.43) if regime == "source_train" else (0.19, 0.47, 0.34)
        compaction = str(rng.choice(("none", "one", "both"), p=weights))
    cap = 10 if regime == "source_train" else 13
    if compaction == "none":
        extra_row = extra_col = 0
    elif compaction == "one":
        extra = int(rng.integers(1, cap + 1))
        if rng.random() < 0.5:
            extra_row, extra_col = extra, 0
        else:
            extra_row, extra_col = 0, extra
    else:
        extra_row = int(rng.integers(1, cap + 1))
        extra_col = int(rng.integers(1, cap + 1))
    if kind in ("two_row", "two_col", "two_square"):
        if kind in ("two_row", "two_square"):
            extra_row = 0
        if kind in ("two_col", "two_square"):
            extra_col = 0
    def offset(extra):
        if extra == 0:
            return 0
        if extra == 1:
            return int(rng.integers(0, 2))
        position = int(rng.choice((0, 1, 2), p=(0.25, 0.50, 0.25)))
        return 0 if position == 0 else extra if position == 2 else int(rng.integers(1, extra))
    row = offset(extra_row)
    col = offset(extra_col)
    return 5 + extra_row, 5 + extra_col, row, col


def _generate_pair(rng, regime, index, palette, first_roles):
    if regime == "source_train" and index == 0:
        background, frame = first_roles
        kind, flavor = "full", "training_anchor"
    else:
        roles = [(a, b) for a in palette for b in palette if a != b]
        if regime == "source_train" and index == 1:
            roles = [pair for pair in roles if pair != first_roles]
        background, frame = roles[int(rng.integers(0, len(roles)))]
        if regime == "source_train" and index == 1:
            kind, flavor = "singleton", "random"
        else:
            kinds = ("full", "singleton", "horizontal", "vertical", "two_row", "two_col", "two_square")
            weights = (0.40, 0.10, 0.15, 0.15, 0.08, 0.08, 0.04) if regime == "source_train" else (0.36, 0.15, 0.145, 0.145, 0.08, 0.08, 0.04)
            kind = str(rng.choice(kinds, p=weights))
            if kind == "full":
                flavor = str(rng.choice(("random", "dense", "top", "lower", "column", "frame_only", "no_frame"), p=(0.33, 0.15, 0.10, 0.10, 0.10, 0.11, 0.11)))
                if flavor == "no_frame" and len(palette) < 3:
                    flavor = "random"
            else:
                flavor = "random"
    height, width, row, col = _carrier(rng, regime, index, kind)
    core = _sample_core(rng, kind, background, frame, palette, flavor)
    return _make_input(height, width, background, frame, core, row, col), _make_output(core, background, frame)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    palette_size = int(rng.choice((2, 3, 4, 5, 6), p=(0.17, 0.20, 0.25, 0.23, 0.15)))
    palette = tuple(int(x) for x in rng.permutation(10)[:palette_size])
    first_roles = (palette[0], palette[1])
    puzzle = {"train": [], "test": []}
    used = set()
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=train_count + test_count, dtype=np.int64)
    number = 0
    for section, regime, count in (("train", "source_train", train_count), ("test", "target_test", test_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(pair_seeds[number]))
            number += 1
            for attempt in range(128):
                input_grid, output_grid = _generate_pair(pair_rng, regime, index, palette, first_roles)
                key = _key(input_grid)
                if key in used or key in _SOURCE_KEYS:
                    continue
                if section == "train" and index == 1 and input_grid.shape == puzzle["train"][0]["input"].shape:
                    continue
                break
            else:
                raise RuntimeError("could not construct a distinct framed example")
            used.add(key)
            puzzle[section].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid, label):
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2, f"{label} must be a two-dimensional NumPy array")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} needs an integer dtype")
    _require(all(1 <= size <= 30 for size in grid.shape), f"{label} has illegal dimensions")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), f"{label} has illegal colors")


def _analyze_input(grid):
    height, width = grid.shape
    _require(height >= 5 and width >= 5, "framed input is smaller than 5 by 5")
    background = int(grid[0, 0])
    _require(all(int(grid[row, col]) == background for row, col in ((0, -1), (-1, 0), (-1, -1))), "frame corners disagree")
    border = np.concatenate((grid[0, 1:-1], grid[-1, 1:-1], grid[1:-1, 0], grid[1:-1, -1]))
    frame = int(border[0])
    _require(frame != background and bool(np.all(border == frame)), "frame color is inconsistent")
    interior = grid[1:-1, 1:-1]
    points = np.argwhere(interior != background)
    _require(len(points) > 0, "empty motif")
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = tuple(int(x) for x in maximum - minimum + 1)
    _require(all(length in (1, 2, 3) for length in span), "invalid motif extent")
    _require(span[0] != 2 or height == 5, "two-row extent needs a five-row carrier")
    _require(span[1] != 2 or width == 5, "two-column extent needs a five-column carrier")
    row = 0 if span[0] == 2 else int(minimum[0]) - (span[0] == 1)
    col = 0 if span[1] == 2 else int(minimum[1]) - (span[1] == 1)
    _require(0 <= row <= height - 5 and 0 <= col <= width - 5, "centered core does not fit")
    outside = interior.copy()
    outside[row:row + 3, col:col + 3] = background
    _require(bool(np.all(outside == background)), "non-background content outside core")
    core = interior[row:row + 3, col:col + 3].copy()
    kind = {(1, 1): "singleton", (1, 3): "horizontal", (3, 1): "vertical", (3, 3): "full", (2, 3): "two_row", (3, 2): "two_col", (2, 2): "two_square"}.get(span)
    _require(kind is not None, "unsupported motif extent")
    return {"background": background, "frame": frame, "core": core, "kind": kind, "shape": grid.shape, "row": row, "col": col, "span": span}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys must be train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "sections must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "illegal pair counts")
    seen_inputs = set()
    seen_pairs = set()
    info = {"train": [], "test": []}
    for section in ("train", "test"):
        for index, pair in enumerate(puzzle[section]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair must have input and output")
            source, result = pair["input"], pair["output"]
            _validate_grid(source, f"{section}[{index}].input")
            _validate_grid(result, f"{section}[{index}].output")
            input_key = _key(source)
            pair_key = (input_key, _key(result))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate pair or input")
            _require(input_key not in _SOURCE_KEYS, "copied source input")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            item = _analyze_input(source)
            expected = _make_output(item["core"], item["background"], item["frame"])
            _require(np.array_equal(result, expected), "incorrect normalized double stamp")
            info[section].append(item)
    training = info["train"]
    _require(training[0]["kind"] == "full" and training[1]["kind"] == "singleton", "training must demonstrate full and haloed motifs")
    _require(training[0]["shape"][0] > 5 and training[0]["shape"][1] > 5, "training needs two-axis compaction")
    _require(training[1]["shape"][0] > 5 and training[1]["shape"][1] > 5, "halo example needs padding")
    _require(len({item["shape"] for item in training}) >= 2, "training carrier shapes do not vary")
    anchor = training[0]
    _require(anchor["background"] in anchor["core"] and anchor["frame"] in anchor["core"], "training anchor must show holes and frame-color reuse")
    _require(not np.array_equal(anchor["core"], np.rot90(anchor["core"])), "training anchor must demonstrate orientation")


def _puzzle_key(puzzle):
    return tuple((section, tuple((_key(pair["input"]), _key(pair["output"])) for pair in puzzle[section])) for section in ("train", "test"))


def self_test() -> None:
    for seed, train, test in ((17, None, None), (18, 7, 3), (19, 2, 1), (20, 10, 1), (21, 10, 4), (22, None, 1), (23, None, 4), (24, 2, None), (25, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "requested train count was ignored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "requested test count was ignored")
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed, train, test)), "generation is nondeterministic")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {kwargs}")
    train_counts, test_counts, puzzles = set(), set(), set()
    shapes = {"train": set(), "test": set()}
    kinds = {"train": set(), "test": set()}
    compactions = {"train": set(), "test": set()}
    placements = set()
    backgrounds, frames, palette_sizes, hole_counts = set(), set(), set(), set()
    frame_only = no_frame = dense = halo_own_frame = 0
    across_color_roles = 0
    review_counts = {name: 0 for name in (
        "test_repeat_frame", "test_new_frame", "test_dense", "test_no_frame",
        "test_frame_only", "test_isolated_frame", "test_lower_stripe",
        "test_vertical_stripe", "train_dense", "train_no_frame",
    )}
    combinations = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzles.add(_puzzle_key(puzzle))
        roles = set()
        colors = set()
        info_by_section = {"train": [], "test": []}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                source, result = pair["input"], pair["output"]
                item = _analyze_input(source)
                info_by_section[section].append(item)
                tile_a, tile_b = result[2:7, 2:7], result[8:13, 8:13]
                _require(np.array_equal(tile_a, tile_b), "double stamp differs")
                _require(np.array_equal(tile_a[1:4, 1:4], item["core"]), "core changed during stamping")
                blank = result.copy()
                blank[2:7, 2:7] = item["background"]
                blank[8:13, 8:13] = item["background"]
                _require(bool(np.all(blank == item["background"])), "output has extra marks")
                shapes[section].add(item["shape"])
                kinds[section].add(item["kind"])
                extra_row, extra_col = item["shape"][0] - 5, item["shape"][1] - 5
                compactions[section].add("none" if extra_row == extra_col == 0 else "both" if extra_row and extra_col else "one")
                if extra_row >= 2 and extra_col >= 2:
                    placements.add(("top" if item["row"] == 0 else "bottom" if item["row"] == extra_row else "middle", "left" if item["col"] == 0 else "right" if item["col"] == extra_col else "middle"))
                backgrounds.add(item["background"])
                frames.add(item["frame"])
                roles.add((item["background"], item["frame"]))
                colors.update(int(x) for x in source.flat)
                core_colors = set(int(x) for x in item["core"].flat)
                occupied = int(np.count_nonzero(item["core"] != item["background"]))
                frame_cells = int(np.count_nonzero(item["core"] == item["frame"]))
                combinations.add((item["kind"], "none" if extra_row == extra_col == 0 else "both" if extra_row and extra_col else "one", "frame_only" if core_colors <= {item["background"], item["frame"]} else "no_frame" if frame_cells == 0 else "mixed"))
                if item["kind"] == "full":
                    hole_counts.add(9 - occupied)
                    dense += occupied == 9
                    frame_only += core_colors <= {item["background"], item["frame"]}
                    no_frame += item["frame"] not in core_colors
                    review_counts[section + "_dense"] += occupied == 9
                if item["kind"] == "singleton":
                    halo_own_frame += int(item["core"][1, 1]) == item["frame"]
                if section == "test":
                    review_counts["test_no_frame"] += frame_cells == 0
                    review_counts["test_frame_only"] += core_colors <= {item["background"], item["frame"]}
                    review_counts["test_isolated_frame"] += frame_cells == 1
                    review_counts["test_lower_stripe"] += bool(np.all(item["core"][2, :] == item["frame"]))
                    review_counts["test_vertical_stripe"] += any(bool(np.all(item["core"][:, col] == item["frame"])) for col in range(3))
                else:
                    review_counts["train_no_frame"] += frame_cells == 0
        palette_sizes.add(len(colors))
        across_color_roles += len(roles) >= 2
        train_frames = {item["frame"] for item in info_by_section["train"]}
        review_counts["test_repeat_frame"] += any(item["frame"] in train_frames for item in info_by_section["test"])
        review_counts["test_new_frame"] += any(item["frame"] not in train_frames for item in info_by_section["test"])
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack range")
    _require(len(puzzles) == 42, "puzzles repeat across seeds")
    _require(len(shapes["train"]) >= 30 and len(shapes["test"]) >= 20, "carrier shapes lack range")
    expected_kinds = {"full", "singleton", "horizontal", "vertical", "two_row", "two_col", "two_square"}
    _require(kinds["train"] == kinds["test"] == expected_kinds, "motif shapes lack range")
    _require(compactions["train"] == compactions["test"] == {"none", "one", "both"}, "compaction types lack range")
    _require(placements == {(row, col) for row in ("top", "middle", "bottom") for col in ("left", "middle", "right")}, "core placement lacks corners, edges, or middle")
    _require(len(backgrounds) >= 7 and len(frames) >= 7 and len(palette_sizes) >= 4, "colors lack range")
    _require(len(hole_counts) >= 5 and dense >= 3 and frame_only >= 10 and no_frame >= 3, "full-core fill lacks range")
    _require(halo_own_frame >= 12 and across_color_roles >= 35, "local color roles lack range")
    _require(len(combinations) >= 28, "motif, compaction, and color variation do not combine")
    _require(review_counts["test_repeat_frame"] >= 20 and review_counts["test_new_frame"] >= 4, "test frame roles lack variation")
    _require(all(review_counts[name] >= 3 for name in ("test_dense", "test_no_frame", "test_frame_only", "test_isolated_frame", "test_lower_stripe", "test_vertical_stripe", "train_dense", "train_no_frame")), "reviewed core variants lack coverage")
