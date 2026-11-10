import numpy as np


# These are the four complete source inputs, including their original colors.
_SOURCE_INPUTS = {
    (16, 16, 7, 7, 1, 3, 3, 0),
    (23, 23, 5, 6, 4, 2, 6, 0),
    (26, 26, 5, 11, 5, 2, 3, 0),
    (16, 16, 7, 5, 2, 1, 6, 0),
}
_SHAPES = tuple(
    (h, w) for h in range(1, 6) for w in range(1, 6)
    if h != w and h * w <= 20
)
_SMALL_SHAPES = tuple(shape for shape in _SHAPES
                      if 2 * sum(shape) + max(shape) <= 16)
_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _checked_count(value, name, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _seed_info(grid):
    colors, counts = np.unique(grid, return_counts=True)
    _require(len(colors) == 2, "input must have a background and one seed color")
    background = int(colors[int(np.argmax(counts))])
    seed_color = int(colors[int(np.argmin(counts))])
    cells = np.argwhere(grid == seed_color)
    r0, c0 = map(int, cells.min(axis=0))
    r1, c1 = map(int, cells.max(axis=0))
    h, w = r1 - r0 + 1, c1 - c0 + 1
    _require(h != w and (h, w) in _SHAPES, "seed must be a supported non-square rectangle")
    _require(len(cells) == h * w, "seed must be solid")
    expected = np.full(grid.shape, background, dtype=grid.dtype)
    expected[r0:r1 + 1, c0:c1 + 1] = seed_color
    _require(np.array_equal(grid, expected), "input must contain one solid seed rectangle")
    return r0, c0, h, w, seed_color, background


def _descriptor(grid):
    return (*map(int, grid.shape), *_seed_info(grid))


def _copies(shape, seed):
    rows, cols = shape
    sr, sc, h, w = seed[:4]
    for direction, (dr, dc) in enumerate(_DIRECTIONS):
        top, bottom, left, right = sr, sr + h - 1, sc, sc + w - 1
        for step in range(1, 64):
            ch, cw = (w, h) if step % 2 else (h, w)
            r0, r1 = (top - ch, top - 1) if dr < 0 else (bottom + 1, bottom + ch)
            c0, c1 = (left - cw, left - 1) if dc < 0 else (right + 1, right + cw)
            if r1 < 0 or r0 >= rows or c1 < 0 or c0 >= cols:
                break
            full = r0 >= 0 and r1 < rows and c0 >= 0 and c1 < cols
            yield direction, step, r0, r1, c0, c1, full
            top, bottom, left, right = r0, r1, c0, c1
        else:
            raise AssertionError("ray did not leave the canvas")


def _stats(shape, seed):
    result = {"visible": 0, "partial": 0, "full_step1": 0,
              "full_step2": 0, "partial_step1": 0,
              "rays": set(), "lengths": [0, 0, 0, 0]}
    for direction, step, _, _, _, _, full in _copies(shape, seed):
        result["visible"] += 1
        result["rays"].add(direction)
        result["lengths"][direction] += 1
        result["partial"] += not full
        result["partial_step1"] += step == 1 and not full
        result["full_step1"] += step == 1 and full
        result["full_step2"] += step == 2 and full
    return result


def _paint(input_grid, alternate):
    seed = _seed_info(input_grid)
    rows, cols = input_grid.shape
    output = input_grid.copy()
    for _, step, r0, r1, c0, c1, _ in _copies(input_grid.shape, seed):
        color = alternate if step % 2 else seed[4]
        rr0, rr1 = max(0, r0), min(rows - 1, r1)
        cc0, cc1 = max(0, c0), min(cols - 1, c1)
        view = output[rr0:rr1 + 1, cc0:cc1 + 1]
        _require(bool(np.all((view == seed[5]) | (view == color))), "rays overlap inconsistently")
        view[:] = color
    return output


def _predict_independently(input_grid, alternate):
    """Cumulative-offset reconstruction, separate from the stamping iterator."""
    sr, sc, h, w, seed_color, _ = _seed_info(input_grid)
    rows, cols = input_grid.shape
    output = input_grid.copy()
    for dr, dc in _DIRECTIONS:
        used_rows = used_cols = 0
        for step in range(1, 64):
            ch, cw = (w, h) if step % 2 else (h, w)
            r0 = sr - used_rows - ch if dr < 0 else sr + h + used_rows
            c0 = sc - used_cols - cw if dc < 0 else sc + w + used_cols
            r1, c1 = r0 + ch - 1, c0 + cw - 1
            if r1 < 0 or r0 >= rows or c1 < 0 or c0 >= cols:
                break
            color = alternate if step % 2 else seed_color
            output[max(0, r0):min(rows, r1 + 1),
                   max(0, c0):min(cols, c1 + 1)] = color
            used_rows += ch
            used_cols += cw
        else:
            raise AssertionError("independent ray did not leave the canvas")
    return output


def _aspect(shape):
    rows, cols = shape
    if rows == cols:
        return "square"
    if abs(rows - cols) == 1:
        return "near_square"
    return "wide" if cols > rows else "tall"


def _shape_choice(rng, orientation=None, exclude=()):
    choices = [shape for shape in _SHAPES
               if (orientation is None or (shape[0] < shape[1]) == (orientation == "horizontal"))
               and shape not in exclude]
    return choices[int(rng.integers(len(choices)))]


def _canvas(rng, shape, mode, preferred_aspect=None):
    h, w = shape
    minimum = max(12, 2 * (h + w) + max(h, w)) if mode == "rich" else 12
    minimum = min(minimum, 30)
    aspect = preferred_aspect or ("square", "near_square", "wide", "tall")[int(rng.integers(4))]
    if aspect == "square":
        if mode == "rich" and minimum <= 16 and rng.random() < 0.3:
            return 16, 16
        side = int(rng.integers(minimum, 31))
        return side, side
    if aspect == "near_square":
        short = int(rng.integers(minimum, 30))
        long = short + 1
        return (short, long) if bool(rng.integers(2)) else (long, short)
    else:
        short = int(rng.integers(minimum, 29))
        long = int(rng.integers(short + 2, 31))
    return (short, long) if aspect == "wide" else (long, short)


def _position(rng, canvas, shape, mode):
    rows, cols = canvas
    h, w = shape
    if mode == "rich":
        margin = h + w
        lo_r, hi_r = margin, rows - h - margin
        lo_c, hi_c = margin, cols - w - margin
        if lo_r > hi_r or lo_c > hi_c:
            return None
        return int(rng.integers(lo_r, hi_r + 1)), int(rng.integers(lo_c, hi_c + 1))
    if mode == "sparse":
        # Side and corner placements allow zero, one, or two outward branches.
        side = int(rng.integers(4))
        r = 0 if side == 0 else rows - h if side == 1 else int(rng.integers(rows - h + 1))
        c = 0 if side == 2 else cols - w if side == 3 else int(rng.integers(cols - w + 1))
        if rng.random() < 0.35:
            r = 0 if rng.random() < 0.5 else rows - h
            c = 0 if rng.random() < 0.5 else cols - w
        return r, c
    if mode == "near_edge":
        if w > 1 and (h == 1 or bool(rng.integers(2))):
            distance = int(rng.integers(1, w))
            r = distance if bool(rng.integers(2)) else rows - h - distance
            c = int(rng.integers(cols - w + 1))
        else:
            distance = int(rng.integers(1, h))
            r = int(rng.integers(rows - h + 1))
            c = distance if bool(rng.integers(2)) else cols - w - distance
        return r, c
    return int(rng.integers(rows - h + 1)), int(rng.integers(cols - w + 1))


def _make_pair(rng, shape, seed_color, background, mode,
               preferred_aspect=None, fixed_canvas=None):
    for _ in range(700):
        canvas = fixed_canvas if fixed_canvas is not None else _canvas(rng, shape, mode, preferred_aspect)
        position = _position(rng, canvas, shape, mode)
        if position is None:
            continue
        r, c = position
        stats = _stats(canvas, (r, c, *shape))
        if stats["visible"] < 1:
            continue
        if mode == "rich" and (stats["full_step1"] != 4 or stats["full_step2"] != 4):
            continue
        if mode == "sparse" and stats["full_step1"] == 4:
            continue
        if mode == "near_edge" and stats["partial_step1"] == 0:
            continue
        if mode == "no_clip" and stats["partial"]:
            continue
        grid = np.full(canvas, background, dtype=np.int8)
        grid[r:r + shape[0], c:c + shape[1]] = seed_color
        if _descriptor(grid) in _SOURCE_INPUTS:
            continue
        return grid
    raise ValueError(f"unable to construct {mode} pair")


def _grid_key(grid):
    return (tuple(map(int, grid.shape)), np.asarray(grid, dtype=np.int8).tobytes())


def _fingerprint(puzzle):
    return tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                       for pair in puzzle[section]) for section in ("train", "test"))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _resolve_counts(rng, num_train, num_test)

    # The reciprocal stamp colors are learned from demonstrations. The seed role
    # and background are properties of an individual input, not hidden settings.
    palette = rng.choice(np.arange(1, 10), size=2, replace=False)
    color_a, color_b = map(int, palette)
    first_horizontal = bool(rng.integers(2))
    first_orientation = "horizontal" if first_horizontal else "vertical"
    second_orientation = "vertical" if first_horizontal else "horizontal"
    orientations = (first_orientation, second_orientation)
    square_index = int(rng.integers(2))
    train_shapes = []
    for i, orientation in enumerate(orientations):
        pool = _SMALL_SHAPES if i == square_index else _SHAPES
        choices = [shape for shape in pool
                   if (shape[0] < shape[1]) == (orientation == "horizontal")]
        train_shapes.append(choices[int(rng.integers(len(choices)))])
    while len(train_shapes) < num_train:
        train_shapes.append(_shape_choice(rng))

    puzzle = {"train": [], "test": []}
    used = set()
    aspect_names = ("square", "near_square", "wide", "tall")
    rich_aspects = [None, None]
    rich_aspects[square_index] = 0
    rich_aspects[1 - square_index] = int(rng.integers(1, 4))
    for index in range(num_train + num_test):
        training = index < num_train
        pair_index = index if training else index - num_train
        if training:
            shape = train_shapes[pair_index]
            color = color_a if pair_index == 0 else color_b if pair_index == 1 else (
                color_a if bool(rng.integers(2)) else color_b)
            mode = "rich" if pair_index < 2 else (
                "rich", "general", "sparse", "near_edge", "no_clip")[int(rng.integers(5))]
            preferred = aspect_names[int(rich_aspects[pair_index])] if pair_index < 2 else None
        else:
            mode = ("rich", "general", "sparse", "near_edge", "no_clip")[int(rng.integers(5))]
            if rng.random() < 0.32:
                shape = train_shapes[int(rng.integers(num_train))]
            else:
                shape = _shape_choice(rng, exclude=set(train_shapes))
            if shape in train_shapes and mode in ("rich", "general"):
                mode = ("sparse", "near_edge", "no_clip")[int(rng.integers(3))]
            color = color_a if bool(rng.integers(2)) else color_b
            preferred = None
        background_choices = [c for c in range(10) if c not in (color_a, color_b)]
        background = background_choices[int(rng.integers(len(background_choices)))]
        fixed_canvas = None
        if not training and mode in ("sparse", "near_edge") and rng.random() < 0.5:
            fixed_canvas = puzzle["train"][int(rng.integers(num_train))]["input"].shape
        for _ in range(20):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            input_grid = _make_pair(pair_rng, shape, color, background, mode,
                                    preferred, fixed_canvas)
            key = _grid_key(input_grid)
            if key not in used:
                break
        else:
            raise ValueError("unable to create distinct inputs")
        used.add(key)
        alternate = color_b if color == color_a else color_a
        pair = {"input": input_grid, "output": _paint(input_grid, alternate)}
        puzzle["train" if training else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle structure")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "sections must be lists")
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test, "invalid pair counts")
    inputs, pairs = set(), set()
    palette = None
    training_colors, training_orientations, rich_colors, rich_orientations = set(), set(), set(), set()
    train_shapes = set()
    records = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair structure")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D array")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must have integer dtype")
                _require(all(1 <= dim <= 30 for dim in grid.shape), "grid dimensions out of bounds")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "invalid ARC color")
            _require(inp.shape == out.shape, "dimensions changed")
            seed = _seed_info(inp)
            r, c, h, w, color, background = seed
            _require(_descriptor(inp) not in _SOURCE_INPUTS, "source pair copied")
            _require(inp.shape[0] >= 12 and inp.shape[1] >= 12, "canvas too small for this family")
            input_key, output_key = _grid_key(inp), _grid_key(out)
            _require(input_key not in inputs and (input_key, output_key) not in pairs, "duplicate pair")
            inputs.add(input_key)
            pairs.add((input_key, output_key))
            output_colors = set(map(int, np.unique(out)))
            _require(background in output_colors and color in output_colors and len(output_colors) == 3,
                     "output must show both stamp colors and background")
            if section == "train":
                alternate = next(iter(output_colors - {background, color}))
            else:
                _require(palette is not None and color in palette,
                         "test seed color is not demonstrated in training")
                alternate = next(iter(palette - {color}))
                _require(output_colors == {background, color, alternate},
                         "test output uses an undemonstrated color")
            pair_palette = frozenset((color, alternate))
            _require(0 not in pair_palette, "stamp colors must be nonzero")
            if palette is None:
                palette = pair_palette
            _require(pair_palette == palette, "stamp palette changed within puzzle")
            _require(np.array_equal(out, _predict_independently(inp, alternate)), "incorrect corner stamps")
            _require(not np.array_equal(inp, out), "unchanged output")
            stats = _stats(inp.shape, seed)
            _require(stats["visible"] >= 1, "no visible ray")
            orientation = "horizontal" if h < w else "vertical"
            if section == "train":
                train_shapes.add((h, w))
                training_colors.add(color)
                training_orientations.add(orientation)
                if stats["full_step1"] == 4 and stats["full_step2"] == 4:
                    rich_colors.add(color)
                    rich_orientations.add(orientation)
            records.append((section, (h, w), stats))
    _require(training_colors == palette, "training must demonstrate both color directions")
    _require(training_orientations == {"horizontal", "vertical"}, "training must show both orientations")
    _require(rich_colors == palette and rich_orientations == {"horizontal", "vertical"},
             "training must show two full steps in both directions, colors, and orientations")
    # The same demonstrated rule applies even when a test has fewer visible arms,
    # no clipped stamp, or a rectangle size absent from training.
    for section, shape, stats in records:
        if section == "test":
            _require(stats["visible"] >= 1 and len(train_shapes) >= 2, "test lacks demonstrated rule")
            _require(shape not in train_shapes or stats["full_step1"] < 4 or stats["partial"] == 0,
                     "test repeats a training shape without a boundary shift")


def self_test() -> None:
    train_counts, test_counts, fingerprints = set(), set(), set()
    input_keys, descriptors, canvas_shapes = set(), set(), set()
    seed_shapes, aspects, palettes, backgrounds = set(), set(), set(), set()
    clipped, unclipped, sparse, four_arm, shared_test_shape, novel_test_shape = 0, 0, 0, 0, 0, 0
    first_partial = one_ray = two_rays = train_16 = large_test = same_canvas = 0
    test_first_color = test_other_color = asymmetric = 0
    combinations = set()
    mixed_aspect_puzzles = mixed_background_puzzles = 0
    min_canvas = 30
    max_canvas = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(_fingerprint(puzzle))
        puzzle_aspects, puzzle_backgrounds = set(), set()
        shapes_in_train = {_seed_info(pair["input"])[2:4] for pair in puzzle["train"]}
        first_two_colors = {_seed_info(pair["input"])[4] for pair in puzzle["train"][:2]}
        _require(len(first_two_colors) == 2, "two-pair color demonstration failed")
        first_color = _seed_info(puzzle["train"][0]["input"])[4]
        first_bg = _seed_info(puzzle["train"][0]["input"])[5]
        palette = set(map(int, np.unique(puzzle["train"][0]["output"]))) - {first_bg}
        train_canvases = {pair["input"].shape for pair in puzzle["train"]}
        test_canvases = {pair["input"].shape for pair in puzzle["test"]}
        same_canvas += bool(train_canvases & test_canvases)
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                info = _seed_info(inp)
                stats = _stats(inp.shape, info)
                alternate = next(iter(palette - {info[4]}))
                _require(np.array_equal(out, _predict_independently(inp, alternate)), "self-test solver failed")
                input_keys.add(_grid_key(inp))
                descriptors.add(_descriptor(inp))
                canvas_shapes.add(inp.shape)
                seed_shapes.add(info[2:4])
                backgrounds.add(info[5])
                puzzle_backgrounds.add(info[5])
                puzzle_aspects.add(_aspect(inp.shape))
                aspects.add(_aspect(inp.shape))
                palettes.add(frozenset((info[4], alternate)))
                min_canvas = min(min_canvas, *inp.shape)
                max_canvas = max(max_canvas, *inp.shape)
                clipped += stats["partial"] > 0
                unclipped += stats["partial"] == 0
                sparse += stats["full_step1"] < 4
                four_arm += stats["full_step1"] == 4
                first_partial += stats["partial_step1"] > 0
                one_ray += len(stats["rays"]) == 1
                two_rays += len(stats["rays"]) == 2
                asymmetric += len(set(stats["lengths"])) > 1
                combinations.add((stats["partial"] > 0, stats["full_step1"] < 4,
                                  min(info[2:4]) >= 3))
                train_16 += section == "train" and inp.shape == (16, 16)
                large_test += section == "test" and max(inp.shape) >= 24
                if section == "test":
                    shared_test_shape += info[2:4] in shapes_in_train
                    novel_test_shape += info[2:4] not in shapes_in_train
                    test_first_color += info[4] == first_color
                    test_other_color += info[4] != first_color
        mixed_aspect_puzzles += len(puzzle_aspects) > 1
        mixed_background_puzzles += len(puzzle_backgrounds) > 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack diversity")
    _require(len(fingerprints) >= 40 and len(input_keys) >= 200 and len(descriptors) >= 200,
             "puzzles or layouts lack diversity")
    _require(len(canvas_shapes) >= 40 and len(seed_shapes) >= 12, "size or seed geometry lacks diversity")
    _require(aspects == {"square", "near_square", "wide", "tall"} and mixed_aspect_puzzles >= 25,
             "canvas aspects lack within-puzzle diversity")
    _require(len(palettes) >= 12 and len(backgrounds) >= 7 and mixed_background_puzzles >= 25,
             "palette or background diversity too low")
    _require(clipped >= 30 and unclipped >= 10 and sparse >= 20 and four_arm >= 80,
             "ray and clipping regimes lack diversity")
    _require(shared_test_shape >= 10 and novel_test_shape >= 20, "test shape shift is fixed")
    _require(first_partial >= 20 and one_ray >= 10 and two_rays >= 20,
             "near-edge and corner rays are underrepresented")
    _require(train_16 >= 5 and large_test >= 10 and same_canvas >= 10,
             "train and test canvas scales do not overlap enough")
    _require(test_first_color >= 5 and test_other_color >= 5 and asymmetric >= 100,
             "color direction or ray-length variation is weak")
    _require(len(combinations) == 8, "clipping, sparsity, and thickness do not combine")
    _require(min_canvas <= 16 and max_canvas >= 28, "canvas scale is narrow")
    _require(_fingerprint(generate_puzzle(10101, 7, 3)) == _fingerprint(generate_puzzle(10101, 7, 3)),
             "explicit generation is nondeterministic")
    _require(_fingerprint(generate_puzzle(20202)) == _fingerprint(generate_puzzle(20202)),
             "automatic generation is nondeterministic")
    for index, (n_train, n_test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(30000 + index, n_train, n_test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (n_train, n_test), "explicit counts ignored")
    for index, (n_train, n_test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(31000 + index, n_train, n_test)
        _require(n_train is None or len(puzzle["train"]) == n_train, "partial train count ignored")
        _require(n_test is None or len(puzzle["test"]) == n_test, "partial test count ignored")
    for index, (n_train, n_test) in enumerate(((1, None), (11, None), (None, 0), (None, 5),
                                                (2, 2), (3, 4), (4, 4), (True, 1))):
        try:
            generate_puzzle(32000 + index, n_train, n_test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
