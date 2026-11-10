import numpy as np


# Exact source inputs are retained only so validation can reject accidental copies.
_SOURCE_INPUT_STRINGS = frozenset(
    (
        (
            "10001000000010101"
            "11010101111100110"
            "00010001001100001"
            "00001010101100010"
            "00100110000011011"
            "00000111011011000"
            "00010011000001000"
            "18188888810010110"
            "08181118801100000"
            "01181181800110000"
            "01001000100000010"
            "01000010101000011"
            "10001001000101001"
            "00101010101010100"
            "00110111111000001"
            "01101100000101000"
            "00100000010100011"
        ),
        (
            "00000000000000000"
            "11011101110111011"
            "00000888880000000"
            "11101118111011101"
            "00000888880000000"
            "11011181110111011"
            "00000888880000000"
            "11101110111011101"
            "00000000000000000"
            "11011101110111011"
            "00000000000000000"
            "11101110111011101"
            "00000000000000000"
            "11011101110111011"
            "00000000000000000"
            "11101110111011101"
            "00000000000000000"
        ),
        (
            "01001001001001001"
            "10110110110110110"
            "01001001001001001"
            "10118118110110110"
            "01001881801001001"
            "10118118110110110"
            "01001881801001001"
            "10118118110110110"
            "01001001001001001"
            "10110110110110110"
            "01001001001001001"
            "10110110110110110"
            "01001001001001001"
            "10110110110110110"
            "01001001001001001"
            "10110110110110110"
            "01001001001001001"
        ),
        (
            "10010010010010010"
            "01001001001001001"
            "00100100100100100"
            "10010010010010010"
            "01001001001001001"
            "00100100100100100"
            "10010010010010010"
            "01001001001001001"
            "00100100100100100"
            "10010010010010010"
            "01001001001001001"
            "00100108188180100"
            "10010018818810010"
            "01001001881881001"
            "00100108188180100"
            "10010010010010010"
            "01001001001001001"
        ),
    )
)


def _is_integer_argument(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _validate_optional_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> None:
    if value is None:
        return
    if not _is_integer_argument(value):
        raise ValueError(f"{name} must be an integer or None")
    if not minimum <= int(value) <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, inclusive"
        )


def _resolve_counts(
    rng: np.random.Generator,
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_optional_count("num_train", num_train, 2, 10)
    _validate_optional_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        # This deterministic schedule guarantees genuine count diversity across
        # sequential seeds while remaining fully seed-dependent.
        test_count = 1 + (int(seed) % 4)
        span = 10 - test_count
        train_count = (
            test_count
            + 1
            + ((int(seed) // 4) % span)
        )
    elif num_train is None:
        test_count = int(num_test)
        train_count = int(rng.integers(test_count + 1, 11))
    elif num_test is None:
        train_count = int(num_train)
        largest_test = min(4, train_count - 1)
        test_count = int(rng.integers(1, largest_test + 1))
    else:
        train_count = int(num_train)
        test_count = int(num_test)

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    colors = [
        int(value)
        for value in rng.choice(10, size=4, replace=False)
    ]
    field_a, field_b, marker, completion = colors

    return {
        "marked_base": field_a,
        "complementary_base": field_b,
        "marker": marker,
        "completion": completion,
        "mix_palettes": bool(rng.random() < 0.78),
    }


def _array_key(grid: np.ndarray) -> tuple:
    normalized = np.asarray(grid, dtype=np.int8)
    return tuple(normalized.shape), normalized.tobytes()


def _pair_key(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return _array_key(input_grid), _array_key(output_grid)


def _puzzle_key(puzzle: dict) -> tuple:
    result = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            result.append(
                (
                    split,
                    _array_key(pair["input"]),
                    _array_key(pair["output"]),
                )
            )
    return tuple(result)


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first.keys()) != {"train", "test"}:
        return False
    if set(second.keys()) != {"train", "test"}:
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for pair_a, pair_b in zip(first[split], second[split]):
            if not np.array_equal(pair_a["input"], pair_b["input"]):
                return False
            if not np.array_equal(pair_a["output"], pair_b["output"]):
                return False
    return True


def _matches_source_input(grid: np.ndarray) -> bool:
    if grid.shape != (17, 17):
        return False
    flattened = "".join(str(int(value)) for value in grid.ravel())
    return flattened in _SOURCE_INPUT_STRINGS


_TEXTURES = (
    "irregular", "clustered", "horizontal", "vertical",
    "checker", "tile", "diagonal_down", "diagonal_up",
)


def _marker_patch_is_valid(mask: np.ndarray) -> bool:
    return bool(
        mask.ndim == 2
        and mask.shape[0] >= 2
        and mask.shape[1] >= 2
        and np.any(mask)
        and np.any(~mask)
        and np.any(mask[0])
        and np.any(mask[-1])
        and np.any(mask[:, 0])
        and np.any(mask[:, -1])
    )


def _sample_grid_dimensions(rng: np.random.Generator) -> tuple[int, int]:
    def dimension() -> int:
        tier = int(rng.choice(3, p=(0.24, 0.49, 0.27)))
        bounds = ((8, 13), (13, 23), (23, 31))[tier]
        return int(rng.integers(*bounds))

    return dimension(), dimension()


def _target_shape(
    rng: np.random.Generator, height: int, width: int, regime: str,
    requested: str | None = None,
) -> tuple[int, int]:
    if requested is None:
        kinds = ("compact", "medium", "wide", "tall", "large", "narrow")
        weights = ((0.18, 0.25, 0.17, 0.17, 0.15, 0.08)
                   if regime == "source_train" else
                   (0.09, 0.17, 0.18, 0.18, 0.23, 0.15))
        requested = str(rng.choice(kinds, p=weights))

    max_h, max_w = height - 2, width - 2

    def pick(low: int, high: int, cap: int) -> int:
        high = min(high, cap)
        low = min(low, high)
        return int(rng.integers(low, high + 1))

    if requested == "compact":
        return pick(2, 6, max_h), pick(2, 6, max_w)
    if requested == "medium":
        return pick(3, 12, max_h), pick(3, 12, max_w)
    if requested == "wide":
        return pick(2, 6, max_h), pick(7, max_w, max_w)
    if requested == "tall":
        return pick(7, max_h, max_h), pick(2, 6, max_w)
    if requested == "large":
        return (pick(max(3, int(0.58 * height)), max_h, max_h),
                pick(max(3, int(0.58 * width)), max_w, max_w))
    if requested == "narrow":
        if rng.random() < 0.5:
            return pick(2, 3, max_h), pick(7, max_w, max_w)
        return pick(7, max_h, max_h), pick(2, 3, max_w)
    raise ValueError("unknown target shape category")


def _field_mask(
    rng: np.random.Generator, height: int, width: int, kind: str,
) -> np.ndarray:
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    if kind in ("irregular", "clustered"):
        band = int(rng.integers(0, 4))
        density_bounds = ((0.12, 0.28), (0.32, 0.68),
                          (0.75, 0.93), (0.22, 0.83))[band]
        density = float(rng.uniform(*density_bounds))
        if kind == "irregular":
            return rng.random((height, width)) < density
        block_h = int(rng.integers(2, 6))
        block_w = int(rng.integers(2, 6))
        blocks = rng.random(
            ((height + block_h - 1) // block_h,
             (width + block_w - 1) // block_w)
        ) < density
        mask = np.repeat(np.repeat(blocks, block_h, 0), block_w, 1)
        mask = mask[:height, :width]
        return np.logical_xor(mask, rng.random((height, width)) < 0.06)
    if kind in ("horizontal", "vertical"):
        period = int(rng.integers(2, 8))
        run = int(rng.integers(1, period))
        phase = int(rng.integers(0, period))
        if kind == "horizontal":
            return np.broadcast_to(((rows + phase) % period) < run,
                                   (height, width)).copy()
        return np.broadcast_to(((cols + phase) % period) < run,
                               (height, width)).copy()
    if kind == "checker":
        block_h = int(rng.integers(1, 4))
        block_w = int(rng.integers(1, 4))
        phase = int(rng.integers(0, 2))
        return ((rows // block_h + cols // block_w + phase) % 2) == 0
    if kind == "tile":
        tile_h = int(rng.integers(2, 7))
        tile_w = int(rng.integers(2, 7))
        density = float(rng.uniform(0.18, 0.82))
        tile = rng.random((tile_h, tile_w)) < density
        phase_h = int(rng.integers(0, tile_h))
        phase_w = int(rng.integers(0, tile_w))
        return tile[(rows + phase_h) % tile_h, (cols + phase_w) % tile_w]
    if kind in ("diagonal_down", "diagonal_up"):
        period = int(rng.integers(3, 7))
        run = int(rng.integers(1, period))
        phase = int(rng.integers(0, period))
        slope = -1 if kind == "diagonal_down" else 1
        return ((rows + slope * cols + phase) % period) < run
    raise ValueError("unknown texture")


def _choose_position(
    rng: np.random.Generator, height: int, width: int,
    target_height: int, target_width: int, zone: str | None,
) -> tuple[int, int] | None:
    minimum_top, maximum_top = 0, height - target_height
    if zone == "upper":
        maximum_top = min(maximum_top, height // 2 - 1)
    elif zone == "lower":
        minimum_top = height // 2
    if minimum_top > maximum_top:
        return None
    top = int(rng.integers(minimum_top, maximum_top + 1))
    left = int(rng.integers(0, width - target_width + 1))
    if rng.random() < 0.31:
        possible_edges = [2, 3]
        if minimum_top == 0:
            possible_edges.append(0)
        if maximum_top == height - target_height:
            possible_edges.append(1)
        edge = int(rng.choice(possible_edges))
        if edge == 0:
            top = 0
        elif edge == 1:
            top = height - target_height
        elif edge == 2:
            left = 0
        else:
            left = width - target_width
    return top, left


def _pair_palette(
    rng: np.random.Generator, parameters: dict, previous: set,
) -> tuple[int, int]:
    if not parameters["mix_palettes"]:
        return parameters["marked_base"], parameters["complementary_base"]
    available = [c for c in range(10)
                 if c not in (parameters["marker"], parameters["completion"])]
    for _ in range(20):
        choice = rng.choice(available, size=2, replace=False)
        palette = int(choice[0]), int(choice[1])
        if palette not in previous:
            return palette
    return palette


def _make_pair(
    rng: np.random.Generator, parameters: dict, regime: str,
    texture: str, requested_shape: str | None, palette: tuple[int, int],
    seen_inputs: set, forbidden_shape: tuple[int, int] | None = None,
    zone: str | None = None,
) -> dict:
    marked_base, complementary_base = palette
    for _ in range(600):
        height, width = _sample_grid_dimensions(rng)
        th, tw = _target_shape(rng, height, width, regime, requested_shape)
        if (th, tw) == forbidden_shape:
            continue
        position = _choose_position(rng, height, width, th, tw, zone)
        if position is None:
            continue
        top, left = position
        mask = _field_mask(rng, height, width, texture)
        patch = mask[top:top + th, left:left + tw]
        if texture == "irregular":
            # The visible extremities are part of the field class; no
            # particular interior location is reserved as a forced hole.
            patch[0, int(rng.integers(0, tw))] = True
            patch[-1, int(rng.integers(0, tw))] = True
            patch[int(rng.integers(0, th)), 0] = True
            patch[int(rng.integers(0, th)), -1] = True
            if np.all(patch):
                patch[int(rng.integers(0, th)),
                      int(rng.integers(0, tw))] = False
        if not _marker_patch_is_valid(patch):
            continue
        outside_region = np.ones((height, width), dtype=bool)
        outside_region[top:top + th, left:left + tw] = False
        outside_values = mask[outside_region]
        if int(np.count_nonzero(outside_values)) < 3:
            continue
        if int(np.count_nonzero(~outside_values)) < 3:
            continue
        field = np.where(mask, marked_base, complementary_base).astype(np.int8)
        inp = field.copy()
        target = inp[top:top + th, left:left + tw]
        target[patch] = parameters["marker"]
        out = inp.copy()
        target_out = out[top:top + th, left:left + tw]
        target_out[~patch] = parameters["completion"]
        if _matches_source_input(inp) or _array_key(inp) in seen_inputs:
            continue
        try:
            info = _analyze_pair(inp, out)
        except AssertionError:
            continue
        if (info["marker"], info["completion"]) != (
            parameters["marker"], parameters["completion"]
        ):
            continue
        seen_inputs.add(_array_key(inp))
        return {"input": inp, "output": out}
    raise RuntimeError("failed to construct an identifiable rectangle")


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    assert input_grid.shape == output_grid.shape
    difference = input_grid != output_grid
    assert np.any(difference)
    from_colors = np.unique(input_grid[difference])
    to_colors = np.unique(output_grid[difference])
    assert len(from_colors) == len(to_colors) == 1
    changed_from = int(from_colors[0])
    completion = int(to_colors[0])
    input_colors = {int(c) for c in np.unique(input_grid)}
    assert len(input_colors) == 3
    assert completion not in input_colors
    assert {int(c) for c in np.unique(output_grid)} == input_colors | {completion}
    candidates = []
    for marker in input_colors - {changed_from}:
        points = np.argwhere(input_grid == marker)
        top, left = np.min(points, axis=0)
        bottom, right = np.max(points, axis=0)
        top, bottom, left, right = map(int, (top, bottom, left, right))
        patch = input_grid[top:bottom + 1, left:right + 1]
        if {int(c) for c in np.unique(patch)} != {marker, changed_from}:
            continue
        if not _marker_patch_is_valid(patch == marker):
            continue
        expected = np.zeros(input_grid.shape, dtype=bool)
        expected[top:bottom + 1, left:right + 1] = patch == changed_from
        if not np.array_equal(expected, difference):
            continue
        marked_base = next(iter(input_colors - {marker, changed_from}))
        outside = np.ones(input_grid.shape, dtype=bool)
        outside[top:bottom + 1, left:right + 1] = False
        if np.count_nonzero(input_grid[outside] == marked_base) < 3:
            continue
        if np.count_nonzero(input_grid[outside] == changed_from) < 3:
            continue
        field = input_grid.copy()
        field[field == marker] = marked_base
        candidates.append({
            "marker": marker, "completion": completion,
            "changed_from": changed_from, "marked_base": marked_base,
            "bbox": (top, bottom, left, right),
            "target_height": bottom - top + 1,
            "target_width": right - left + 1,
            "marker_count": int(np.count_nonzero(patch == marker)),
            "completion_count": int(np.count_nonzero(expected)),
            "field": field,
        })
    assert len(candidates) == 1
    return candidates[0]


def _infer_test_output(train_pairs: list, test_input: np.ndarray) -> np.ndarray:
    # The only learned values are the marker and the output-only color.
    learned = set()
    for pair in train_pairs:
        info = _analyze_pair(pair["input"], pair["output"])
        learned.add((info["marker"], info["completion"]))
    assert len(learned) == 1
    marker, completion = next(iter(learned))
    assert completion not in test_input
    points = np.argwhere(test_input == marker)
    assert len(points) > 0
    top, left = np.min(points, axis=0)
    bottom, right = np.max(points, axis=0)
    result = test_input.copy()
    patch = result[top:bottom + 1, left:right + 1]
    remaining = {int(c) for c in np.unique(patch)} - {marker}
    assert len(remaining) == 1
    patch[patch == next(iter(remaining))] = completion
    return result


def _texture_plan(rng: np.random.Generator, train_count: int, test_count: int) -> tuple[list, list]:
    if rng.random() < 0.24:
        common = str(rng.choice(_TEXTURES))
        train = [common] * train_count
    else:
        train = [str(rng.choice(("irregular", "clustered"))),
                 str(rng.choice(_TEXTURES[2:]))]
        train.extend(str(rng.choice(_TEXTURES)) for _ in range(train_count - 2))
        rng.shuffle(train)
    test = [str(rng.choice(_TEXTURES)) for _ in range(test_count)]
    return train, test


def generate_puzzle(
    seed: int, num_train: int | None = None, num_test: int | None = None,
) -> dict:
    if not _is_integer_argument(seed) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    parameters = sample_puzzle_parameters(rng)
    train_count, test_count = _resolve_counts(rng, int(seed), num_train, num_test)
    train_textures, test_textures = _texture_plan(rng, train_count, test_count)
    placement = str(rng.choice(
        ("free", "upper_to_lower", "lower_to_upper", "mixed"),
        p=(0.37, 0.19, 0.19, 0.25),
    ))
    if placement == "upper_to_lower":
        train_zones = ["upper"] * train_count
        test_zones = ["lower"] * test_count
    elif placement == "lower_to_upper":
        train_zones = ["lower"] * train_count
        test_zones = ["upper"] * test_count
    elif placement == "mixed":
        train_zones = [str(rng.choice(("upper", "lower", "free")))
                       for _ in range(train_count)]
        test_zones = [str(rng.choice(("upper", "lower", "free")))
                      for _ in range(test_count)]
    else:
        train_zones = ["free"] * train_count
        test_zones = ["free"] * test_count
    if rng.random() < 0.24:
        train_shapes = [None] * train_count
    else:
        train_shapes = ["compact", str(rng.choice(("wide", "tall", "large")))]
        train_shapes.extend([None] * (train_count - 2))
        rng.shuffle(train_shapes)
    if placement == "lower_to_upper":
        train_shapes = ["compact", "wide"]
        train_shapes.extend(str(rng.choice(("compact", "medium", "wide")))
                            for _ in range(train_count - 2))
        rng.shuffle(train_shapes)
    pair_seeds = rng.integers(0, 2**63 - 1, size=train_count + test_count,
                              dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen_inputs: set = set()
    used_palettes: set = set()
    first_shape = None
    for index, texture in enumerate(train_textures):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        palette = _pair_palette(pair_rng, parameters, used_palettes)
        used_palettes.add(palette)
        pair = _make_pair(pair_rng, parameters, "source_train", texture,
                          train_shapes[index], palette, seen_inputs,
                          first_shape if index == train_count - 1 else None,
                          None if train_zones[index] == "free" else train_zones[index])
        puzzle["train"].append(pair)
        if index == 0:
            info = _analyze_pair(pair["input"], pair["output"])
            first_shape = (info["target_height"], info["target_width"])
    for offset, texture in enumerate(test_textures):
        pair_rng = np.random.default_rng(int(pair_seeds[train_count + offset]))
        palette = _pair_palette(pair_rng, parameters, used_palettes)
        used_palettes.add(palette)
        test_shape = (str(pair_rng.choice(("compact", "medium", "wide")))
                      if test_zones[offset] == "lower" else None)
        pair = _make_pair(pair_rng, parameters, "target_test", texture,
                          test_shape, palette, seen_inputs,
                          zone=None if test_zones[offset] == "free" else test_zones[offset])
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test
    seen_inputs, seen_pairs = set(), set()
    conventions = set()
    train_shapes = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape
            assert not _matches_source_input(inp)
            input_key = _array_key(inp)
            pair_key = _pair_key(inp, out)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            info = _analyze_pair(inp, out)
            conventions.add((info["marker"], info["completion"]))
            if split == "train":
                train_shapes.add((info["target_height"], info["target_width"]))
    assert len(conventions) == 1
    assert len(train_shapes) >= 2
    for pair in puzzle["test"]:
        assert np.array_equal(
            _infer_test_output(puzzle["train"], pair["input"]), pair["output"]
        )


def _field_class(field: np.ndarray) -> str:
    # Reconstruct texture from the returned grid, independent of sampling data.
    height, width = field.shape
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    colors = [int(c) for c in np.unique(field)]
    for slope, label in ((-1, "diagonal_down"), (1, "diagonal_up")):
        for period in range(3, 7):
            for run in range(1, period):
                for phase in range(period):
                    mask = ((rows + slope * cols + phase) % period) < run
                    for marked in colors:
                        other = colors[1] if marked == colors[0] else colors[0]
                        if np.array_equal(field, np.where(mask, marked, other)):
                            return label
    if np.all(field == field[:, :1]):
        return "horizontal"
    if np.all(field == field[:1, :]):
        return "vertical"
    for bh in range(1, 4):
        for bw in range(1, 4):
            for phase in range(2):
                mask = ((rows // bh + cols // bw + phase) % 2) == 0
                for marked in colors:
                    other = colors[1] if marked == colors[0] else colors[0]
                    if np.array_equal(field, np.where(mask, marked, other)):
                        return "checker"
    for rh in range(1, 7):
        for cw in range(1, 7):
            expected = field[:rh, :cw][rows % rh, cols % cw]
            if np.array_equal(field, expected):
                return "tile"
    return "irregular"


def self_test() -> None:
    train_counts, test_counts = set(), set()
    puzzle_keys, grid_shapes, target_shapes, palettes, textures = (
        set(), set(), set(), set(), set()
    )
    positions = {"train": set(), "test": set()}
    edges = {"train": set(), "test": set()}
    aspect_texture = set()
    densities, completion_counts = [], []
    target_fractions = []
    empty_internal_rows = 0
    empty_internal_columns = 0
    unmarked_corners = 0
    clustered_patches = 0
    irregular_center_marked = 0
    irregular_center_open = 0
    mixed_palettes = 0
    shared_palettes = 0
    placement_patterns = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(_puzzle_key(puzzle))
        local_palettes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _analyze_pair(inp, out)
                top, bottom, left, right = info["bbox"]
                height, width = inp.shape
                # This checks the recoloring from returned pixels, including
                # every unchanged location outside the marker rectangle.
                expected = inp.copy()
                patch = expected[top:bottom + 1, left:right + 1]
                patch[patch == info["changed_from"]] = info["completion"]
                assert np.array_equal(expected, out)
                grid_shapes.add((height, width))
                shape = (info["target_height"], info["target_width"])
                target_shapes.add(shape)
                palettes.add((info["changed_from"], info["marked_base"]))
                local_palettes.add((info["changed_from"], info["marked_base"]))
                texture = _field_class(info["field"])
                textures.add((split, texture))
                aspect = ("wide" if shape[1] >= 2 * shape[0]
                          else "tall" if shape[0] >= 2 * shape[1]
                          else "balanced")
                aspect_texture.add((texture, aspect))
                positions[split].add("upper" if top < height // 2 else "lower")
                if top == 0:
                    edges[split].add("top")
                if bottom == height - 1:
                    edges[split].add("bottom")
                if left == 0:
                    edges[split].add("left")
                if right == width - 1:
                    edges[split].add("right")
                densities.append(info["marker_count"] / (shape[0] * shape[1]))
                completion_counts.append(info["completion_count"])
                target_fractions.append(shape[0] * shape[1] / (height * width))
                marker_patch = inp[top:bottom + 1, left:right + 1] == info["marker"]
                empty_internal_rows += (
                    shape[0] > 2 and not np.all(np.any(marker_patch[1:-1], axis=1))
                )
                empty_internal_columns += (
                    shape[1] > 2 and not np.all(np.any(marker_patch[:, 1:-1], axis=0))
                )
                unmarked_corners += not any((
                    marker_patch[0, 0], marker_patch[0, -1],
                    marker_patch[-1, 0], marker_patch[-1, -1],
                ))
                if texture == "irregular":
                    center_marked = bool(marker_patch[shape[0] // 2, shape[1] // 2])
                    irregular_center_marked += center_marked
                    irregular_center_open += not center_marked
                    if min(shape) >= 4:
                        agreement = np.mean(np.concatenate((
                            (marker_patch[1:] == marker_patch[:-1]).ravel(),
                            (marker_patch[:, 1:] == marker_patch[:, :-1]).ravel(),
                        )))
                        density = np.mean(marker_patch)
                        clustered_patches += (
                            0.25 <= density <= 0.75 and agreement > 0.70
                        )
                if split == "test":
                    assert np.array_equal(_infer_test_output(
                        puzzle["train"], inp.copy()), out)
        mixed_palettes += len(local_palettes) > 1
        shared_palettes += len(local_palettes) == 1
        train_rows = {
            "upper" if _analyze_pair(p["input"], p["output"])["bbox"][0]
            < p["input"].shape[0] // 2 else "lower"
            for p in puzzle["train"]
        }
        test_rows = {
            "upper" if _analyze_pair(p["input"], p["output"])["bbox"][0]
            < p["input"].shape[0] // 2 else "lower"
            for p in puzzle["test"]
        }
        if train_rows == {"upper"} and test_rows == {"lower"}:
            placement_patterns.add("upper_to_lower")
        if train_rows == {"lower"} and test_rows == {"upper"}:
            placement_patterns.add("lower_to_upper")
        if train_rows & test_rows:
            placement_patterns.add("overlap")
        if len(train_rows | test_rows) == 2 and (len(train_rows) > 1 or len(test_rows) > 1):
            placement_patterns.add("mixed")
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_keys) >= 40
    assert len(grid_shapes) >= 50 and len(target_shapes) >= 40
    assert len(palettes) >= 20
    assert all(positions[split] == {"upper", "lower"}
               for split in ("train", "test"))
    assert all(edges[split] == {"top", "bottom", "left", "right"}
               for split in ("train", "test"))
    assert len(textures) >= 12 and len(aspect_texture) >= 12
    assert all(
        (split, family) in textures
        for split in ("train", "test")
        for family in ("irregular", "horizontal", "vertical", "checker",
                       "tile", "diagonal_down", "diagonal_up")
    )
    assert placement_patterns == {
        "upper_to_lower", "lower_to_upper", "overlap", "mixed"
    }
    assert min(densities) < 0.30 and max(densities) > 0.80
    assert min(completion_counts) <= 4 and max(completion_counts) >= 50
    assert max(target_fractions) > 0.60
    assert empty_internal_rows > 0 and empty_internal_columns > 0
    assert unmarked_corners > 0
    assert clustered_patches >= 3
    assert irregular_center_marked > 0 and irregular_center_open > 0
    assert mixed_palettes >= 20 and shared_palettes >= 5
    for seed, n_train, n_test in ((1001, 7, 3), (2001, 2, 1),
                                  (2002, 10, 1), (2003, 10, 4)):
        puzzle = generate_puzzle(seed, n_train, n_test)
        validate_generated_puzzle(puzzle)
        assert len(puzzle["train"]) == n_train
        assert len(puzzle["test"]) == n_test
        assert _puzzles_equal(puzzle, generate_puzzle(seed, n_train, n_test))
    for seed, n_train, n_test in ((3001, None, 1), (3002, None, 4),
                                  (3003, 2, None), (3004, 10, None)):
        puzzle = generate_puzzle(seed, n_train, n_test)
        validate_generated_puzzle(puzzle)
        if n_train is not None:
            assert len(puzzle["train"]) == n_train
        if n_test is not None:
            assert len(puzzle["test"]) == n_test
    for request in ({"num_train": 1}, {"num_train": 11},
                    {"num_test": 0}, {"num_test": 5},
                    {"num_train": 2, "num_test": 2},
                    {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(4001, **request)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
