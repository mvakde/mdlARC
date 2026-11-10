"""Generate examples of a lattice-masked row-and-column cross."""

import numpy as np


# Exact source pairs, used only to prevent copies of the original task.
_SOURCE_SPECS = (
    (20, 20, 1, 1, 14, 11, 0, 1, 3),
    (20, 20, 2, 2, 6, 6, 0, 1, 2),
    (20, 20, 1, 2, 5, 15, 0, 1, 8),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _coerce_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed: int, rng: np.random.Generator,
                    num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train_count = _coerce_count(num_train, "num_train")
    test_count = _coerce_count(num_test, "num_test")
    if train_count is not None and not 2 <= train_count <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test_count is not None and not 1 <= test_count <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must exceed num_test")
        return train_count, test_count
    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        minimum = max(2, test_count + 1)
        train_count = minimum + (seed // 4) % (11 - minimum)
    elif train_count is None:
        train_count = int(rng.integers(max(2, test_count + 1), 11))
    else:
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    return train_count, test_count


def _complete_band_starts(length: int, phase: int) -> list[int]:
    return [i for i in range(length - 1)
            if i % 3 != phase and (i + 1) % 3 != phase]


def _construct_instance(height: int, width: int, row_phase: int,
                        column_phase: int, seed_row: int, seed_column: int,
                        background: int, fill: int, accent: int
                        ) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.indices((height, width))
    gutters = (rows % 3 == row_phase) | (columns % 3 == column_phase)
    input_grid = np.full((height, width), fill, dtype=np.int8)
    input_grid[gutters] = background
    input_grid[seed_row:seed_row + 2, seed_column:seed_column + 2] = accent
    selected = ~gutters & (((rows == seed_row) | (rows == seed_row + 1)) |
                           ((columns == seed_column) | (columns == seed_column + 1)))
    output_grid = input_grid.copy()
    output_grid[selected] = accent
    return input_grid, output_grid


def _grid_key(grid: np.ndarray) -> tuple:
    return (tuple(int(v) for v in grid.shape),
            grid.astype(np.int16, copy=False).tobytes(order="C"))


def _pair_key(pair: dict) -> tuple:
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _matches_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    if input_grid.shape != (20, 20) or output_grid.shape != (20, 20):
        return False
    for spec in _SOURCE_SPECS:
        source_input, source_output = _construct_instance(*spec)
        if np.array_equal(input_grid, source_input) and np.array_equal(output_grid, source_output):
            return True
    return False


def _infer_input(input_grid: np.ndarray) -> dict:
    height, width = input_grid.shape
    values = [int(v) for v in np.unique(input_grid)]
    _require(len(values) == 3, "input needs exactly three semantic colors")
    rows, columns = np.indices(input_grid.shape)
    candidates = []
    for background in values:
        for row_phase in range(3):
            for column_phase in range(3):
                gutters = (rows % 3 == row_phase) | (columns % 3 == column_phase)
                if not np.array_equal(input_grid == background, gutters):
                    continue
                remaining = [v for v in values if v != background]
                accents = [v for v in remaining if np.count_nonzero(input_grid == v) == 4]
                if len(accents) != 1:
                    continue
                accent = accents[0]
                fill = next(v for v in remaining if v != accent)
                if np.count_nonzero(input_grid == fill) <= 4:
                    continue
                coordinates = np.argwhere(input_grid == accent)
                seed_row, seed_column = (int(v) for v in coordinates.min(axis=0))
                expected_seed = np.zeros(input_grid.shape, dtype=bool)
                expected_seed[seed_row:seed_row + 2, seed_column:seed_column + 2] = True
                if not np.array_equal(input_grid == accent, expected_seed):
                    continue
                if seed_row not in _complete_band_starts(height, row_phase):
                    continue
                if seed_column not in _complete_band_starts(width, column_phase):
                    continue
                candidates.append(dict(height=height, width=width,
                                       background=background, fill=fill, accent=accent,
                                       row_phase=row_phase, column_phase=column_phase,
                                       seed_row=seed_row, seed_column=seed_column))
    _require(len(candidates) == 1, "input needs one unambiguous lattice and seed")
    return candidates[0]


def _expected_output(input_grid: np.ndarray, info: dict) -> np.ndarray:
    rows, columns = np.indices(input_grid.shape)
    gutters = ((rows % 3 == info["row_phase"]) |
               (columns % 3 == info["column_phase"]))
    cross = (((rows == info["seed_row"]) | (rows == info["seed_row"] + 1)) |
             ((columns == info["seed_column"]) | (columns == info["seed_column"] + 1)))
    expected = input_grid.copy()
    expected[cross & ~gutters] = info["accent"]
    return expected


def _inspect_transformation(pair: dict) -> dict:
    input_grid, output_grid = pair["input"], pair["output"]
    info = _infer_input(input_grid)
    _require(np.array_equal(output_grid, _expected_output(input_grid, info)),
             "output is not the lattice-masked cross")
    _require(not np.array_equal(input_grid, output_grid), "cross must grow")
    changed = output_grid != input_grid
    row_band = changed[info["seed_row"]:info["seed_row"] + 2, :]
    column_band = changed[:, info["seed_column"]:info["seed_column"] + 2]
    _require(bool(row_band.any() and column_band.any()),
             "both cross directions must be demonstrated")
    return info


def _seed_region(info: dict) -> tuple[int, int]:
    return (int(2 * info["seed_row"] + 1 >= info["height"]),
            int(2 * info["seed_column"] + 1 >= info["width"]))


def _choose_start(rng: np.random.Generator, starts: list[int], length: int,
                  side: int | None) -> int:
    if side is not None:
        subset = [v for v in starts if int(2 * v + 1 >= length) == side]
        if subset:
            starts = subset
    mode = int(rng.integers(0, 5))
    if mode == 0:
        return starts[0]
    if mode == 1:
        return starts[-1]
    if mode == 2:
        distance = min(abs(2 * v + 1 - length) for v in starts)
        starts = [v for v in starts if abs(2 * v + 1 - length) == distance]
    return int(starts[int(rng.integers(0, len(starts)))])


def _sample_roles(rng: np.random.Generator, train_count: int, test_count: int
                  ) -> list[tuple[int, int, int]]:
    """Color policy is sampled once; actual role colors are local and visible."""
    palette = [int(v) for v in rng.permutation(10)]
    mode = int(rng.integers(0, 4))  # shared, varied, swapped, varied
    base_background, base_fill = palette[:2]
    roles = []
    train_accents = []
    for index in range(train_count + test_count):
        if mode == 0 or index == 0:
            background, fill = base_background, base_fill
        elif mode == 2 and index == 1:
            background, fill = base_fill, base_background
        else:
            choices = [(a, b) for a in range(10) for b in range(10)
                       if a != b and (index != 1 or (a, b) != (base_background, base_fill))]
            background, fill = choices[int(rng.integers(0, len(choices)))]
        available = [v for v in range(10) if v not in (background, fill)]
        if index < train_count:
            if index == 0:
                accent = palette[2]
            elif index == 1:
                options = [v for v in available if v != train_accents[0]]
                accent = options[int(rng.integers(0, len(options)))]
            else:
                accent = available[int(rng.integers(0, len(available)))]
            train_accents.append(accent)
        else:
            reuse = [v for v in train_accents if v in available]
            novel = [v for v in available if v not in train_accents]
            options = reuse if reuse and (not novel or rng.random() < 0.5) else novel
            accent = options[int(rng.integers(0, len(options)))]
        roles.append((background, fill, accent))
    return roles


def _sample_phases(rng: np.random.Generator, train_count: int, test_count: int
                   ) -> list[tuple[int, int]]:
    # Occasionally reproduce the source's factor recombination, with marker
    # positions and palettes still sampled separately.
    source_like = rng.random() < 0.16
    if source_like:
        first = (1, 1) if rng.random() < 0.5 else (2, 2)
        second = (2, 2) if first == (1, 1) else (1, 1)
    else:
        first = (int(rng.integers(0, 3)), int(rng.integers(0, 3)))
        second = (int(rng.choice([v for v in range(3) if v != first[0]])),
                  int(rng.choice([v for v in range(3) if v != first[1]])))
    phases = [first, second]
    for _ in range(2, train_count):
        phases.append((int(rng.integers(0, 3)), int(rng.integers(0, 3))))
    for test_index in range(test_count):
        if source_like and test_index == 0:
            phase = (1, 2) if rng.random() < 0.5 else (2, 1)
        elif test_index == 0 and rng.random() < 0.65:
            unused = [(r, c) for r in range(3) for c in range(3)
                      if (r, c) not in phases[:train_count]]
            phase = unused[int(rng.integers(0, len(unused)))] if unused else (
                int(rng.integers(0, 3)), int(rng.integers(0, 3)))
        else:
            phase = (int(rng.integers(0, 3)), int(rng.integers(0, 3)))
        phases.append(phase)
    return phases


def _sample_shapes(rng: np.random.Generator, train_count: int, test_count: int
                   ) -> list[tuple[int, int]]:
    total = train_count + test_count
    mode = int(rng.choice(3, p=[0.14, 0.32, 0.54]))
    if mode == 0:
        return [(20, 20)] * total
    common = (int(rng.integers(12, 31)), int(rng.integers(12, 31)))
    shapes = []
    for index in range(total):
        if mode == 1 and (index < 2 or (index == train_count and rng.random() < 0.5)):
            shape = common
        else:
            shape = (int(rng.integers(12, 31)), int(rng.integers(12, 31)))
        shapes.append(shape)
    return shapes


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain exactly input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                height, width = grid.shape
                _require(1 <= height <= 30 and 1 <= width <= 30,
                         "grid exceeds ARC bounds")
                _require(np.issubdtype(grid.dtype, np.integer), "grid needs integer dtype")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         "grid has an illegal color")
            input_grid, output_grid = pair["input"], pair["output"]
            _require(input_grid.shape == output_grid.shape, "cross must preserve size")
            _require(min(input_grid.shape) >= 12, "too few periods to infer lattice")
            input_key, pair_key = _grid_key(input_grid), _pair_key(pair)
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate grid or pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _require(not _matches_source_pair(input_grid, output_grid),
                     "pair exactly copies the source")
            infos[split].append(_inspect_transformation(pair))
    # Both axes and the color-independent accent role are demonstrated twice.
    first, second = infos["train"][:2]
    _require(first["row_phase"] != second["row_phase"] and
             first["column_phase"] != second["column_phase"],
             "training needs independent gutter-phase evidence")
    _require(first["accent"] != second["accent"],
             "training needs two accent literals")
    _require(_seed_region(first) != _seed_region(second),
             "training needs distinct marker placements")
    train_palettes = {(v["background"], v["fill"]) for v in infos["train"]}
    test_palettes = {(v["background"], v["fill"]) for v in infos["test"]}
    if test_palettes - train_palettes:
        _require(len(train_palettes) > 1,
                 "training must demonstrate local background/fill changes")
    train_layouts = {(v["height"], v["width"], v["row_phase"],
                      v["column_phase"], v["seed_row"], v["seed_column"])
                     for v in infos["train"]}
    for info in infos["test"]:
        layout = (info["height"], info["width"], info["row_phase"],
                  info["column_phase"], info["seed_row"], info["seed_column"])
        _require(layout not in train_layouts,
                 "test must have a new geometric configuration")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    roles = _sample_roles(rng, train_count, test_count)
    phases = _sample_phases(rng, train_count, test_count)
    shapes = _sample_shapes(rng, train_count, test_count)
    first_region = (int(rng.integers(0, 2)), int(rng.integers(0, 2)))
    puzzle = {"train": [], "test": []}
    seen = set()
    train_layouts = set()
    for index in range(train_count + test_count):
        split = "train" if index < train_count else "test"
        height, width = shapes[index]
        row_phase, column_phase = phases[index]
        background, fill, accent = roles[index]
        if index == 0:
            region = first_region
        elif index == 1:
            region = (1 - first_region[0], 1 - first_region[1])
        else:
            region = None
        row_starts = _complete_band_starts(height, row_phase)
        column_starts = _complete_band_starts(width, column_phase)
        # A deterministic bounded fallback advances the seed block if a grid
        # happens to duplicate a previous pair with the same colors and phase.
        for attempt in range(64):
            seed_row = _choose_start(rng, row_starts, height,
                                     None if region is None else region[0])
            seed_column = _choose_start(rng, column_starts, width,
                                        None if region is None else region[1])
            if attempt >= 32:
                seed_row = row_starts[(attempt - 32) % len(row_starts)]
                seed_column = column_starts[((attempt - 32) // len(row_starts)) % len(column_starts)]
            input_grid, output_grid = _construct_instance(
                height, width, row_phase, column_phase, seed_row, seed_column,
                background, fill, accent)
            key = _grid_key(input_grid)
            layout = (height, width, row_phase, column_phase, seed_row, seed_column)
            if (key not in seen and (split == "train" or layout not in train_layouts)
                    and not _matches_source_pair(input_grid, output_grid)):
                break
        else:
            raise AssertionError("could not construct a distinct pair")
        seen.add(key)
        if split == "train":
            train_layouts.add(layout)
        puzzle[split].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[name], b[name]) for a, b in zip(first[split], second[split])
                   for name in ("input", "output"))
               for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "few train counts")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "few test counts")
    signatures = set()
    layouts, shapes, roles, inputs = set(), set(), set(), set()
    split_phases = {"train": set(), "test": set()}
    split_regions = {"train": set(), "test": set()}
    shared_palette = varied_palette = swaps = reused_accent = novel_accent = 0
    same_shape = smaller_test = all_twenty = edge_seed = central_seed = 0
    clipped_edge = balanced_arms = unequal_arms = phase0_edge = 0
    source_recombination = combined_variation = 0
    aligned = {"train": 0, "test": 0}
    mixed = {"train": 0, "test": 0}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        infos = {split: [_infer_input(p["input"]) for p in puzzle[split]]
                 for split in ("train", "test")}
        signatures.add(tuple(_pair_key(pair) for split in ("train", "test")
                             for pair in puzzle[split]))
        all_infos = infos["train"] + infos["test"]
        train_colors = {(v["background"], v["fill"]) for v in infos["train"]}
        shared_palette += len({(v["background"], v["fill"]) for v in all_infos}) == 1
        varied_palette += len(train_colors) > 1
        swaps += any(a["background"] == b["fill"] and a["fill"] == b["background"]
                     for a in infos["train"] for b in infos["train"] if a is not b)
        train_accents = {v["accent"] for v in infos["train"]}
        reused_accent += any(v["accent"] in train_accents for v in infos["test"])
        novel_accent += any(v["accent"] not in train_accents for v in infos["test"])
        same_shape += len({(v["height"], v["width"]) for v in all_infos}) < len(all_infos)
        smaller_test += any(t["height"] <= r["height"] and t["width"] <= r["width"]
                            for t in infos["test"] for r in infos["train"])
        all_twenty += all(v["height"] == v["width"] == 20 for v in all_infos)
        source_recombination += (
            {(v["row_phase"], v["column_phase"]) for v in infos["train"][:2]}
            == {(1, 1), (2, 2)}
            and (infos["test"][0]["row_phase"], infos["test"][0]["column_phase"])
            in {(1, 2), (2, 1)})
        combined_variation += (
            len(train_colors) > 1 and
            any(v["row_phase"] != v["column_phase"] for v in infos["train"]))
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], infos[split]):
                _require(np.array_equal(pair["output"], _expected_output(pair["input"], info)),
                         "self-test transformation mismatch")
                _require(not _matches_source_pair(pair["input"], pair["output"]),
                         "self-test source copy")
                shapes.add((info["height"], info["width"]))
                roles.add((info["background"], info["fill"], info["accent"]))
                inputs.add(_grid_key(pair["input"]))
                layouts.add((info["height"], info["width"], info["row_phase"],
                             info["column_phase"], info["seed_row"], info["seed_column"]))
                split_phases[split].add((info["row_phase"], info["column_phase"]))
                split_regions[split].add(_seed_region(info))
                aligned[split] += info["row_phase"] == info["column_phase"]
                mixed[split] += info["row_phase"] != info["column_phase"]
                row_starts = _complete_band_starts(info["height"], info["row_phase"])
                col_starts = _complete_band_starts(info["width"], info["column_phase"])
                on_edge = (info["seed_row"] in (row_starts[0], row_starts[-1]) or
                           info["seed_column"] in (col_starts[0], col_starts[-1]))
                edge_seed += on_edge
                central_seed += (abs(2 * info["seed_row"] + 1 - info["height"]) <= 3 and
                                 abs(2 * info["seed_column"] + 1 - info["width"]) <= 3)
                phase0_edge += on_edge and (info["row_phase"] == 0 or
                                            info["column_phase"] == 0)
                clipped_edge += (
                    (info["height"] - 1) % 3 != info["row_phase"] and
                    (info["height"] - 2) % 3 == info["row_phase"] or
                    (info["width"] - 1) % 3 != info["column_phase"] and
                    (info["width"] - 2) % 3 == info["column_phase"])
                active_rows = [r for r in range(info["height"])
                               if r % 3 != info["row_phase"]]
                active_columns = [c for c in range(info["width"])
                                  if c % 3 != info["column_phase"]]
                up = sum(r < info["seed_row"] for r in active_rows)
                down = sum(r > info["seed_row"] + 1 for r in active_rows)
                left = sum(c < info["seed_column"] for c in active_columns)
                right = sum(c > info["seed_column"] + 1 for c in active_columns)
                balanced_arms += abs(up - down) <= 2 and abs(left - right) <= 2
                unequal_arms += max(abs(up - down), abs(left - right)) >= 8
    _require(len(signatures) == 42 and len(inputs) >= 180, "weak grid diversity")
    _require(len(shapes) >= 80 and len(layouts) >= 180 and len(roles) >= 100,
             "weak structural or color diversity")
    _require(all(len(split_phases[s]) == 9 and len(split_regions[s]) == 4
                 for s in ("train", "test")), "phase or quadrant coverage is incomplete")
    _require(min(shared_palette, varied_palette, swaps, reused_accent, novel_accent,
                 same_shape, smaller_test, edge_seed, central_seed) > 0,
             "a review variation was not observed")
    _require(all_twenty > 0, "source-like comparable canvases were absent")
    _require(min(clipped_edge, balanced_arms, unequal_arms, phase0_edge,
                 source_recombination, combined_variation) > 0,
             "a geometric or combined review variation was absent")
    _require(all(aligned[s] > 0 and mixed[s] > 0 for s in ("train", "test")),
             "both splits need aligned and mixed lattices")
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4)):
        puzzle = generate_puzzle(seed, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                 "explicit count not honored")
    for seed, train, test in ((2001, None, 1), (2002, None, 4),
                              (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        _require(train is None or len(puzzle["train"]) == train, "partial train count")
        _require(test is None or len(puzzle["test"]) == test, "partial test count")
    _require(_puzzles_equal(generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3)),
             "explicit generation is not deterministic")
    _require(_puzzles_equal(generate_puzzle(271828), generate_puzzle(271828)),
             "automatic generation is not deterministic")
    _require(not _puzzles_equal(generate_puzzle(9001), generate_puzzle(9002)),
             "different seeds made the same puzzle")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4), (2.5, 1), (3, 1.5)):
        try:
            generate_puzzle(12345, train, test)
        except ValueError:
            continue
        raise AssertionError("illegal count did not raise ValueError")
