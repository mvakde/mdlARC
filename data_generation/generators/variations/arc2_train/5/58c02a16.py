import numpy as np


_SOURCE_INPUT_SIGNATURES = frozenset(
    (
        ("3747777", "7347777", "4447777") + ("7777777",) * 4,
        ("227877777", "722877777", "277877777", "888877777")
        + ("777777777",) * 5,
        ("97077777", "77077777", "79077777", "00077777")
        + ("77777777",) * 8,
        ("4745777777777777", "4475777777777777",
         "4745777777777777", "5555777777777777")
        + ("7777777777777777",) * 12,
    )
)
_SMALL_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3))
_LARGE_SHAPES = ((3, 4), (4, 3), (4, 4))
_CORE_SHAPES = _SMALL_SHAPES + _LARGE_SHAPES


def _coerce_seed(seed: int) -> int:
    if (not isinstance(seed, (int, np.integer))
            or isinstance(seed, (bool, np.bool_))):
        raise ValueError("seed must be an integer")
    return int(seed) % (1 << 64)


def _validate_count(name: str, value: int, lower: int, upper: int) -> int:
    if (not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(seed_value: int, num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    train = None if num_train is None else _validate_count("num_train", num_train, 2, 10)
    test = None if num_test is None else _validate_count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must be greater than num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed_value % 4
        minimum = max(2, test + 1)
        return minimum + (seed_value // 4) % (11 - minimum), test
    if train is None:
        minimum = max(2, test + 1)
        return minimum + seed_value % (11 - minimum), test
    return train, 1 + seed_value % min(4, train - 1)


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    colors = tuple(int(c) for c in rng.choice(10, size=3, replace=False))
    return {
        "base_palette": colors,  # background, primary, secondary
        "shared_background": bool(rng.integers(0, 2)),
        "expansive_test": bool(rng.integers(0, 3) == 0),
    }


def _random_palette(rng: np.random.Generator, background: int | None) -> tuple:
    if background is None:
        return tuple(int(c) for c in rng.choice(10, size=3, replace=False))
    remaining = [c for c in range(10) if c != background]
    primary, secondary = rng.choice(remaining, size=2, replace=False)
    return background, int(primary), int(secondary)


def _palette_for_pair(rng: np.random.Generator, parameters: dict,
                      history: list[tuple], index: int) -> tuple:
    base = parameters["base_palette"]
    shared = parameters["shared_background"]
    if index == 0:
        return base
    if index == 1:
        if not shared:
            choices = [c for c in range(10) if c != base[0]]
            return _random_palette(rng, int(rng.choice(choices)))
        return base if rng.random() < 0.5 else (base[0], base[2], base[1])
    if rng.random() < 0.55:
        palette = history[int(rng.integers(len(history)))]
        if rng.random() < 0.35:
            return palette[0], palette[2], palette[1]
        return palette
    background = base[0] if shared else None
    return _random_palette(rng, background)


def _two_dimensional(mask: np.ndarray) -> bool:
    return (np.unique(mask, axis=0).shape[0] > 1
            and np.unique(mask, axis=1).shape[1] > 1)


def _sample_core_mask(rng: np.random.Generator, height: int, width: int,
                      require_two_dimensional: bool = False) -> np.ndarray:
    total = height * width
    if require_two_dimensional and (height, width) == (2, 2):
        return np.indices((2, 2)).sum(axis=0) % 2 == int(rng.integers(2))
    for _ in range(80):
        mode = int(rng.integers(14))
        mask = np.zeros((height, width), dtype=bool)
        if mode <= 3 or mode >= 12:
            if mode == 0:
                count = 1
            elif mode == 1:
                count = total - 1
            else:
                count = int(rng.integers(2, total - 1))
            mask.ravel()[rng.choice(total, size=count, replace=False)] = True
        elif mode in (4, 5):
            offset = int(rng.integers(2))
            mask = (np.indices(mask.shape).sum(axis=0) + offset) % 2 == 0
        elif mode in (6, 7):
            axis = int(rng.integers(2))
            length = height if axis == 0 else width
            chosen = rng.choice(length, size=int(rng.integers(1, length)),
                                replace=False)
            if axis == 0:
                mask[chosen, :] = True
            else:
                mask[:, chosen] = True
        elif mode in (8, 9):
            top = int(rng.integers(height))
            left = int(rng.integers(width))
            bottom = int(rng.integers(top + 1, height + 1))
            right = int(rng.integers(left + 1, width + 1))
            mask[top:bottom, left:right] = True
        else:
            half = (width + 1) // 2
            fragment = rng.integers(0, 2, size=(height, half)).astype(bool)
            mask[:, :half] = fragment
            mask[:, width // 2 + width % 2:] = fragment[:, :width // 2][:, ::-1]
        count = int(mask.sum())
        if mode not in (0, 1) and not (2 <= count <= total - 2):
            continue
        if 0 < count < total and (not require_two_dimensional
                                  or _two_dimensional(mask)):
            return mask
    # A deterministic, two-dimensional fallback for the smallest motifs too.
    mask = np.indices((height, width)).sum(axis=0) % 2 == 0
    return mask


def _make_input(core_mask: np.ndarray, background: int, primary: int,
                secondary: int, canvas_height: int, canvas_width: int) -> np.ndarray:
    height, width = core_mask.shape
    grid = np.full((canvas_height, canvas_width), background, dtype=np.int8)
    grid[:height, :width] = np.where(core_mask, primary, background)
    grid[height, :width + 1] = secondary
    grid[:height + 1, width] = secondary
    return grid


def _render_output(core_mask: np.ndarray, background: int, primary: int,
                   secondary: int, canvas_height: int,
                   canvas_width: int) -> np.ndarray:
    height, width = core_mask.shape
    rows = np.arange(canvas_height)
    cols = np.arange(canvas_width)
    inner = core_mask[rows[:, None] % height, cols[None, :] % width]
    outer = core_mask[(rows[:, None] // height) % height,
                      (cols[None, :] // width) % width]
    output = np.full((canvas_height, canvas_width), background, dtype=np.int8)
    output[inner & outer] = primary
    output[inner & ~outer] = secondary
    return output


def _sample_shapes(rng: np.random.Generator, train_count: int,
                   test_count: int) -> tuple[list[tuple], list[tuple]]:
    first = _SMALL_SHAPES[int(rng.integers(len(_SMALL_SHAPES)))]
    second_choices = [shape for shape in _SMALL_SHAPES if shape != first]
    second = second_choices[int(rng.integers(len(second_choices)))]
    train = [first, second]
    for _ in range(2, train_count):
        pool = _CORE_SHAPES if rng.random() < 0.28 else _SMALL_SHAPES
        train.append(pool[int(rng.integers(len(pool)))])
    # Larger test cores have already appeared in a training demonstration.
    trained_large = [shape for shape in _LARGE_SHAPES if shape in train]
    test = []
    for _ in range(test_count):
        pool = trained_large if trained_large and rng.random() < 0.35 else _SMALL_SHAPES
        test.append(pool[int(rng.integers(len(pool)))])
    return train, test


def _sample_axis(rng: np.random.Generator, tile: int, upper: int,
                 lower: int, exact: bool) -> int:
    candidates = [size for size in range(max(tile, lower), upper + 1)
                  if (size % tile == 0) == exact]
    if not candidates:
        candidates = [size for size in range(tile, upper + 1)
                      if (size % tile == 0) == exact]
    return int(rng.choice(candidates))


def _sample_canvas(rng: np.random.Generator, shape: tuple, regime: str,
                   parameters: dict, teach_repeat: bool) -> tuple[int, int]:
    tile_h, tile_w = shape[0] ** 2, shape[1] ** 2
    if regime == "source_train":
        upper_h = max(tile_h + 1, min(30, tile_h + 12))
        upper_w = max(tile_w + 1, min(30, tile_w + 12))
        if parameters["expansive_test"]:
            upper_h = max(tile_h + 1, min(upper_h, 17))
            upper_w = max(tile_w + 1, min(upper_w, 17))
        lower_h, lower_w = tile_h, tile_w
    else:
        upper_h = upper_w = 30
        floor = 15 if parameters["expansive_test"] else 0
        lower_h, lower_w = max(tile_h, floor), max(tile_w, floor)
        if shape[0] == shape[1] and rng.random() < 0.3:
            exact = bool(rng.integers(2))
            side = _sample_axis(rng, tile_h, 30, lower_h, exact)
            return side, side
    if teach_repeat:
        mode = 3  # a visible partial continuation in both directions
    else:
        mode = int(rng.integers(4))
    height = _sample_axis(rng, tile_h, upper_h, lower_h, mode in (0, 1))
    width = _sample_axis(rng, tile_w, upper_w, lower_w, mode in (0, 2))
    return height, width


def generate_pair(rng: np.random.Generator, regime: str,
                  puzzle_parameters: dict, shape: tuple, palette: tuple,
                  teach_two_dimensional: bool = False,
                  teach_repeat: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError(f"unknown pair regime: {regime}")
    canvas_h, canvas_w = _sample_canvas(
        rng, shape, regime, puzzle_parameters, teach_repeat)
    mask = _sample_core_mask(rng, *shape, teach_two_dimensional)
    background, primary, secondary = palette
    return (
        _make_input(mask, background, primary, secondary, canvas_h, canvas_w),
        _render_output(mask, background, primary, secondary, canvas_h, canvas_w),
    )


def _rows_signature(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _array_value_key(array: np.ndarray) -> tuple:
    return tuple(array.shape), tuple(int(v) for v in array.ravel())


def _decode_input(grid: np.ndarray) -> dict:
    height, width = grid.shape
    background = int(grid[-1, -1])
    assert np.all(grid[-1] == background) and np.all(grid[:, -1] == background)
    colors = {int(v) for v in np.unique(grid)}
    assert len(colors) == 3
    solutions = []
    for secondary in colors - {background}:
        positions = np.argwhere(grid == secondary)
        core_h = int(positions[:, 0].max())
        core_w = int(positions[:, 1].max())
        if (core_h, core_w) not in _CORE_SHAPES:
            continue
        if height < core_h ** 2 or width < core_w ** 2:
            continue
        expected_border = np.zeros(grid.shape, dtype=bool)
        expected_border[core_h, :core_w + 1] = True
        expected_border[:core_h + 1, core_w] = True
        if not np.array_equal(grid == secondary, expected_border):
            continue
        primary = next(iter(colors - {background, secondary}))
        core = grid[:core_h, :core_w]
        if not np.all((core == background) | (core == primary)):
            continue
        mask = core == primary
        if not (0 < int(mask.sum()) < core_h * core_w):
            continue
        if not np.array_equal(
                grid, _make_input(mask, background, primary, secondary, height, width)):
            continue
        solutions.append({
            "core_height": core_h, "core_width": core_w, "core_mask": mask,
            "background": background, "primary": primary, "secondary": secondary,
            "canvas_height": height, "canvas_width": width,
        })
    assert len(solutions) == 1
    return solutions[0]


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed_value = _coerce_seed(seed)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    rng = np.random.default_rng(seed_value)
    parameters = sample_puzzle_parameters(rng)
    train_shapes, test_shapes = _sample_shapes(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    palettes = []
    for split, shapes, regime in (
            ("train", train_shapes, "source_train"),
            ("test", test_shapes, "target_test")):
        for pair_index, shape in enumerate(shapes):
            global_index = len(palettes)
            palette = _palette_for_pair(rng, parameters, palettes, global_index)
            palettes.append(palette)
            for attempt in range(100):
                child_seed = int(rng.integers(0, 1 << 63, dtype=np.int64))
                pair_rng = np.random.default_rng(child_seed)
                pair = generate_pair(
                    pair_rng, regime, parameters, shape, palette,
                    teach_two_dimensional=(split == "train" and pair_index == 0),
                    teach_repeat=(split == "train" and pair_index == 0),
                )
                inp, out = pair
                key = _array_value_key(inp)
                if (key in used_inputs or _rows_signature(inp) in _SOURCE_INPUT_SIGNATURES
                        or np.array_equal(inp, out)):
                    continue
                used_inputs.add(key)
                puzzle[split].append({"input": inp, "output": out})
                break
            else:
                raise ValueError("unable to construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    information = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            input_key = _array_value_key(inp)
            pair_key = input_key, _array_value_key(out)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _rows_signature(inp) not in _SOURCE_INPUT_SIGNATURES
            info = _decode_input(inp)
            expected = _render_output(
                info["core_mask"], info["background"], info["primary"],
                info["secondary"], *inp.shape)
            assert np.array_equal(out, expected)
            roles = {info["background"], info["primary"], info["secondary"]}
            assert {int(v) for v in np.unique(inp)} == roles
            assert {int(v) for v in np.unique(out)} == roles
            information[split].append(info)
    train = information["train"]
    test = information["test"]
    assert len({(x["core_height"], x["core_width"]) for x in train}) >= 2
    assert any(_two_dimensional(x["core_mask"]) for x in train)
    assert any(
        x["canvas_height"] > x["core_height"] ** 2
        and x["canvas_width"] > x["core_width"] ** 2
        and x["canvas_height"] % (x["core_height"] ** 2) != 0
        and x["canvas_width"] % (x["core_width"] ** 2) != 0
        for x in train)
    trained_large = {(x["core_height"], x["core_width"]) for x in train
                     if (x["core_height"], x["core_width"]) in _LARGE_SHAPES}
    for x in test:
        shape = x["core_height"], x["core_width"]
        assert shape not in _LARGE_SHAPES or shape in trained_large


def _puzzles_equal(a: dict, b: dict) -> bool:
    return all(
        len(a[split]) == len(b[split])
        and all(np.array_equal(p["input"], q["input"])
                and np.array_equal(p["output"], q["output"])
                for p, q in zip(a[split], b[split]))
        for split in ("train", "test"))


def _predict_from_visible_input(grid: np.ndarray) -> np.ndarray:
    """A direct block substitution, separate from the vectorized label builder."""
    info = _decode_input(grid)
    core = info["core_mask"]
    core_h, core_w = core.shape
    result = np.full(grid.shape, info["background"], dtype=np.int8)
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            if core[row % core_h, column % core_w]:
                macro_row = (row // core_h) % core_h
                macro_column = (column // core_w) % core_w
                result[row, column] = (
                    info["primary"] if core[macro_row, macro_column]
                    else info["secondary"])
    return result


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({tuple(_array_value_key(pair["input"])
                      for split in ("train", "test") for pair in p[split])
                for p in puzzles}) == 42
    layouts, motifs, palettes, shapes = set(), set(), set(), set()
    crop_types, motif_counts = set(), set()
    one_background = False
    background_changes = palette_reuse = palette_swaps = 0
    rectangular_tests = large_train = large_test = repeated_3x3_train = overlap = 0
    larger_test = striped = source_like_square_test = first_square_test = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        infos = {}
        for split in ("train", "test"):
            infos[split] = []
            for pair in puzzle[split]:
                assert np.array_equal(
                    _predict_from_visible_input(pair["input"]), pair["output"])
                x = _decode_input(pair["input"])
                infos[split].append(x)
                shape = x["core_height"], x["core_width"]
                canvas = x["canvas_height"], x["canvas_width"]
                shapes.add(shape)
                layouts.add((split, shape, canvas))
                motifs.add((shape, tuple(bool(v) for v in x["core_mask"].ravel())))
                palettes.add((x["background"], x["primary"], x["secondary"]))
                motif_counts.add(int(x["core_mask"].sum()))
                one_background |= int(x["core_mask"].sum()) == shape[0] * shape[1] - 1
                if not _two_dimensional(x["core_mask"]):
                    striped += 1
                if split == "test":
                    crop_types.add((canvas[0] % (shape[0] ** 2) == 0,
                                    canvas[1] % (shape[1] ** 2) == 0))
                    rectangular_tests += shape[0] != shape[1]
                    large_test += shape in _LARGE_SHAPES
                    source_like_square_test += (
                        shape == (3, 3) and canvas[0] == canvas[1]
                        and canvas[0] >= 14)
                else:
                    large_train += shape in _LARGE_SHAPES
                    repeated_3x3_train += (
                        shape == (3, 3) and canvas[0] > 9 and canvas[1] > 9)
        all_info = infos["train"] + infos["test"]
        first_square_test += (
            infos["test"][0]["canvas_height"] == infos["test"][0]["canvas_width"])
        role_palettes = [(x["background"], x["primary"], x["secondary"])
                         for x in all_info]
        background_changes += len({x["background"] for x in all_info}) > 1
        palette_reuse += len(set(role_palettes)) < len(role_palettes)
        palette_swaps += any((b, s, p) in role_palettes
                             for b, p, s in role_palettes)
        overlap += any(
            t["canvas_height"] <= max(x["canvas_height"] for x in infos["train"])
            or t["canvas_width"] <= max(x["canvas_width"] for x in infos["train"])
            for t in infos["test"])
        larger_test += any(
            t["canvas_height"] > max(x["canvas_height"] for x in infos["train"])
            and t["canvas_width"] > max(x["canvas_width"] for x in infos["train"])
            for t in infos["test"])
    assert shapes == set(_CORE_SHAPES)
    assert len(layouts) >= 100 and len(motifs) >= 80 and len(palettes) >= 75
    assert crop_types == {(True, True), (True, False), (False, True), (False, False)}
    assert 1 in motif_counts and one_background
    assert background_changes >= 10 and palette_reuse >= 10 and palette_swaps >= 5
    assert rectangular_tests >= 12 and large_train >= 5
    assert large_test >= 5 and source_like_square_test >= 2
    assert 0 < first_square_test < 42
    assert repeated_3x3_train >= 5 and overlap >= 10 and larger_test >= 8
    assert striped >= 15
    for seed, requested_train, requested_test in (
            (123456, 7, 3), (2001, 2, 1), (2002, 10, 1),
            (2003, 10, 4), (3001, None, 1), (3002, None, 4),
            (3003, 2, None), (3004, 10, None)):
        p = generate_puzzle(seed, requested_train, requested_test)
        validate_generated_puzzle(p)
        assert _puzzles_equal(p, generate_puzzle(seed, requested_train, requested_test))
        if requested_train is not None:
            assert len(p["train"]) == requested_train
        if requested_test is not None:
            assert len(p["test"]) == requested_test
    for request in (
            {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
            {"num_test": 5}, {"num_train": 2, "num_test": 2},
            {"num_train": 3, "num_test": 4}, {"num_train": 2.0, "num_test": 1}):
        try:
            generate_puzzle(4000, **request)
        except ValueError:
            continue
        raise AssertionError(f"invalid count request accepted: {request}")
