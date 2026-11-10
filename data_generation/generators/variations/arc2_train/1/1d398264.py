import numpy as np


_DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),             (0, 1),
    (1, -1),  (1, 0),   (1, 1),
)
_ADJACENT = ((0, 1), (1, 2), (2, 4), (4, 7),
             (7, 6), (6, 5), (5, 3), (3, 0))
_OPPOSITE = ((0, 7), (1, 6), (2, 5), (3, 4))
_SOURCE_DESCRIPTORS = {
    (20, 20, 0, 6, 15, (6, 2, 2, 4, 5, 4, 6, 8, 8)),
    (10, 10, 0, 2, 3, (3, 1, 2, 2, 6, 2, 2, 7, 7)),
    (15, 15, 0, 1, 2, (2, 5, 7, 2, 8, 7, 3, 3, 3)),
    (16, 16, 0, 2, 3, (4, 2, 5, 2, 9, 5, 4, 1, 1)),
    (12, 12, 0, 2, 3, (6, 1, 1, 6, 7, 1, 3, 3, 1)),
}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _checked_optional_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    train = _checked_optional_count(num_train, "num_train")
    test = _checked_optional_count(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        low = max(2, test + 1)
        train = low + seed % (11 - low)
    else:
        test = 1 + seed % min(4, train - 1)
    return train, test


def _analyze_input(grid):
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
             "input must be a two-dimensional NumPy array")
    height, width = grid.shape
    _require(5 <= height <= 30 and 5 <= width <= 30,
             "a visible 3 by 3 source needs a 5 to 30 canvas")
    _require(np.issubdtype(grid.dtype, np.integer), "input dtype must be integral")
    _require(np.all((grid >= 0) & (grid <= 9)), "input colors are out of range")
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    _require(int(np.max(counts)) == height * width - 9,
             "input must have exactly nine non-background cells")
    cells = np.argwhere(grid != background)
    _require(cells.shape == (9, 2), "input must have nine source cells")
    top, left = (int(x) for x in cells.min(axis=0))
    _require(top + 2 < height and left + 2 < width,
             "source block extends beyond the canvas")
    _require(np.all(grid[top:top + 3, left:left + 3] != background),
             "source cells must form a full 3 by 3 block")
    center = (top + 1, left + 1)
    emitters = tuple(int(grid[center[0] + dr, center[1] + dc])
                     for dr, dc in _DIRECTIONS)
    return {"height": height, "width": width, "background": background,
            "top": top, "left": left, "center": center,
            "core": int(grid[center]), "emitters": emitters}


def _construct_output(input_grid):
    info = _analyze_input(input_grid)
    result = input_grid.copy()
    row0, col0 = info["center"]
    for (dr, dc), color in zip(_DIRECTIONS, info["emitters"]):
        row, col = row0 + dr, col0 + dc
        while 0 <= row < info["height"] and 0 <= col < info["width"]:
            result[row, col] = color
            row += dr
            col += dc
    return result


def _grid_key(grid):
    normalized = np.ascontiguousarray(grid, dtype=np.int8)
    return tuple(normalized.shape), normalized.tobytes()


def _matches_source_input(grid):
    info = _analyze_input(grid)
    block = tuple(int(x) for x in grid[info["top"]:info["top"] + 3,
                                         info["left"]:info["left"] + 3].flat)
    return (info["height"], info["width"], info["background"],
            info["top"], info["left"], block) in _SOURCE_DESCRIPTORS


def _sample_shape(rng, demonstrator=False, square_only=False):
    category = 0 if square_only else int(rng.integers(0, 6))
    if category == 0:
        side = int(rng.integers(8 if demonstrator else 5, 31))
        return side, side
    if category == 1:
        low = 8 if demonstrator else 5
        return int(rng.integers(low, 13)), int(rng.integers(low, 13))
    if category == 2:
        return int(rng.integers(10, 22)), int(rng.integers(10, 22))
    if category == 3:
        return int(rng.integers(19, 31)), int(rng.integers(19, 31))
    if category == 4:
        return (int(rng.integers(8 if demonstrator else 5, 13)),
                int(rng.integers(18, 31)))
    return (int(rng.integers(18, 31)),
            int(rng.integers(8 if demonstrator else 5, 13)))


def _sample_center(rng, shape, demonstrator=False, target=False):
    height, width = shape
    if demonstrator:
        return (int(rng.integers(2, height - 2)),
                int(rng.integers(2, width - 2)))
    category = int(rng.choice(6, p=(.23, .20, .13, .17, .15, .12)
                              if target else (.15, .18, .14, .18, .20, .15)))
    if category == 0:
        return (1 if int(rng.integers(2)) == 0 else height - 2,
                1 if int(rng.integers(2)) == 0 else width - 2)
    if category == 1:
        if int(rng.integers(2)):
            return (1 if int(rng.integers(2)) == 0 else height - 2,
                    int(rng.integers(1, width - 1)))
        return (int(rng.integers(1, height - 1)),
                1 if int(rng.integers(2)) == 0 else width - 2)
    if category == 2:
        return (height // 2, width // 2)
    if category == 3:
        return (2 if int(rng.integers(2)) == 0 else height - 3,
                2 if int(rng.integers(2)) == 0 else width - 3)
    if category == 4:
        return (int(rng.integers(2, height - 2)),
                int(rng.integers(2, width - 2)))
    return (int(rng.integers(1, height - 1)),
            int(rng.integers(1, width - 1)))


def _sample_colors(rng, background, distinct_count, distinct_core=False,
                   fixed_palette=None, fixed_core=None):
    available = np.array([c for c in range(10) if c != background], dtype=np.int8)
    palette = ([int(x) for x in rng.choice(available, size=distinct_count, replace=False)]
               if fixed_palette is None else fixed_palette)
    assignments = [-1] * 8
    positions = [int(x) for x in rng.permutation(8)]
    if 2 <= distinct_count <= 7 and int(rng.integers(3)) != 0:
        group = _ADJACENT if int(rng.integers(2)) == 0 else _OPPOSITE
        a, b = group[int(rng.integers(len(group)))]
        assignments[a] = assignments[b] = 0
        positions = [p for p in positions if p not in (a, b)]
        for index, position in enumerate(positions[:distinct_count - 1], 1):
            assignments[position] = index
    else:
        for index, position in enumerate(positions[:distinct_count]):
            assignments[position] = index
    for position in range(8):
        if assignments[position] < 0:
            assignments[position] = int(rng.integers(distinct_count))
    emitters = tuple(palette[index] for index in assignments)
    if fixed_core is not None:
        core = fixed_core
    elif distinct_count == 8 or (not distinct_core and float(rng.random()) < .43):
        core = palette[int(rng.integers(distinct_count))]
    else:
        unused = [c for c in available if int(c) not in palette]
        core = int(rng.choice(unused))
    return core, emitters


def _make_pair(rng, shape, center, background, distinct_count, distinct_core=False,
               fixed_palette=None, fixed_core=None):
    core, emitters = _sample_colors(rng, background, distinct_count, distinct_core,
                                   fixed_palette, fixed_core)
    grid = np.full(shape, background, dtype=np.int8)
    row, col = center
    grid[row, col] = core
    for (dr, dc), color in zip(_DIRECTIONS, emitters):
        grid[row + dr, col + dc] = color
    return {"input": grid, "output": _construct_output(grid)}


def _sample_layout(rng, split, first, size_mode, train_shapes, used_layouts,
                   train_centers, square_only=False):
    for _ in range(96):
        if split == "test" and size_mode == 0 and float(rng.random()) < .68:
            shape = train_shapes[int(rng.integers(len(train_shapes)))]
        else:
            shape = _sample_shape(rng, demonstrator=first, square_only=square_only)
        if split == "test" and size_mode == 1 and shape in train_shapes:
            continue
        center = _sample_center(rng, shape, demonstrator=first,
                                target=(split == "test"))
        layout = shape + center
        if layout in used_layouts:
            continue
        if split == "train" and len(train_shapes) == 1:
            if shape == train_shapes[0] or center == train_centers[0]:
                continue
        return shape, center
    for height in range(5, 31):
        for width in range(5, 31):
            if square_only and height != width:
                continue
            shape = (height, width)
            if split == "test" and size_mode == 1 and shape in train_shapes:
                continue
            if split == "train" and len(train_shapes) == 1 and shape == train_shapes[0]:
                continue
            for row in range(1, height - 1):
                for col in range(1, width - 1):
                    center = (row, col)
                    if shape + center not in used_layouts and (
                        split != "train" or len(train_shapes) != 1
                        or center != train_centers[0]
                    ):
                        return shape, center
    raise RuntimeError("no unused source-block layout exists")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)

    # These are distribution modes, not hidden transformation conventions.
    # Color meanings and source position are always local to each input.
    simple_mode = float(rng.random()) < .10
    stable_background = simple_mode or float(rng.random()) < .25
    base_background = int(rng.integers(10))
    size_mode = int(rng.integers(3))  # overlap, held-out dimensions, mixed
    if simple_mode:
        simple_count = int(rng.integers(4, 8))
        available = np.array([c for c in range(10) if c != base_background],
                             dtype=np.int8)
        simple_colors = [int(x) for x in rng.choice(available,
                                                  size=simple_count + 1,
                                                  replace=False)]
        fixed_core, fixed_palette = simple_colors[0], simple_colors[1:]
    else:
        simple_count, fixed_core, fixed_palette = None, None, None
    puzzle = {"train": [], "test": []}
    train_shapes = []
    train_centers = []
    used_layouts = set()
    used_inputs = set()

    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            first = split == "train" and index == 0
            shape, center = _sample_layout(pair_rng, split, first, size_mode,
                                           train_shapes, used_layouts, train_centers,
                                           square_only=simple_mode)
            used_layouts.add(shape + center)
            if split == "train":
                train_shapes.append(shape)
                train_centers.append(center)
            if stable_background or first:
                background = base_background
            else:
                background = int(pair_rng.integers(10))
                if split == "train" and index == 1 and background == base_background:
                    background = (background + 1 + int(pair_rng.integers(9))) % 10
            distinct_count = (simple_count if simple_mode else
                              int(pair_rng.integers(4, 8) if first
                                  else pair_rng.integers(1, 9)))
            for _ in range(64):
                pair = _make_pair(pair_rng, shape, center, background, distinct_count,
                                  distinct_core=first, fixed_palette=fixed_palette,
                                  fixed_core=fixed_core)
                key = _grid_key(pair["input"])
                if key not in used_inputs and not _matches_source_input(pair["input"]):
                    break
            else:
                raise RuntimeError("could not construct a unique source block")
            used_inputs.add(key)
            puzzle[split].append(pair)

    validate_generated_puzzle(puzzle)
    return puzzle


def _assert_ray_semantics(input_grid, output_grid):
    info = _analyze_input(input_grid)
    covered = np.zeros(input_grid.shape, dtype=bool)
    row0, col0 = info["center"]
    for (dr, dc), color in zip(_DIRECTIONS, info["emitters"]):
        row, col = row0 + dr, col0 + dc
        while 0 <= row < info["height"] and 0 <= col < info["width"]:
            _require(int(output_grid[row, col]) == color, "ray color is wrong")
            covered[row, col] = True
            row += dr
            col += dc
    _require(np.array_equal(output_grid[~covered], input_grid[~covered]),
             "core or off-ray cells changed")
    _require(np.any(output_grid != input_grid), "at least one ray must extend")


def _cellwise_expected(input_grid):
    info = _analyze_input(input_grid)
    result = input_grid.copy()
    row0, col0 = info["center"]
    for row in range(info["height"]):
        for col in range(info["width"]):
            dr, dc = row - row0, col - col0
            if max(abs(dr), abs(dc)) <= 1:
                continue
            if dr == 0 or dc == 0 or abs(dr) == abs(dc):
                sr = (dr > 0) - (dr < 0)
                sc = (dc > 0) - (dc < 0)
                result[row, col] = input_grid[row0 + sr, col0 + sc]
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    records = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "each pair needs exactly input and output")
            source, target = pair["input"], pair["output"]
            for grid in (source, target):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integral")
                _require(all(1 <= dimension <= 30 for dimension in grid.shape),
                         "grid dimensions are out of range")
                _require(np.all((grid >= 0) & (grid <= 9)), "grid colors are out of range")
            _require(source.shape == target.shape, "input and output shapes must match")
            info = _analyze_input(source)
            _assert_ray_semantics(source, target)
            _require(np.array_equal(_cellwise_expected(source), target),
                     "output violates cellwise radial rule")
            _require(not _matches_source_input(source), "source task was copied")
            key = _grid_key(source)
            pair_key = (key, _grid_key(target))
            _require(key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            records[split].append(info)

    train = records["train"]
    tests = records["test"]
    train_layouts = {(i["height"], i["width"], *i["center"]) for i in train}
    _require(len({(i["height"], i["width"]) for i in train}) >= 2,
             "training needs distinct canvas shapes")
    _require(len({i["center"] for i in train}) >= 2,
             "training needs distinct source positions")
    _require(all((i["height"], i["width"], *i["center"]) not in train_layouts
                 for i in tests), "test must use new structural layouts")
    _require(len(set(train[0]["emitters"])) >= 4
             and train[0]["core"] not in train[0]["emitters"],
             "first training pair must identify the inert core clearly")
    demonstrated = set()
    for info, pair in zip(train, puzzle["train"]):
        row0, col0 = info["center"]
        for index, ((dr, dc), color) in enumerate(zip(_DIRECTIONS, info["emitters"])):
            row, col = row0 + 2 * dr, col0 + 2 * dc
            if 0 <= row < info["height"] and 0 <= col < info["width"]:
                _require(int(pair["input"][row, col]) == info["background"]
                         and int(pair["output"][row, col]) == color,
                         "training direction is not demonstrated")
                demonstrated.add(index)
    _require(demonstrated == set(range(8)),
             "training must demonstrate all eight directions")


def _puzzles_equal(first, second):
    if set(first) != set(second):
        return False
    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for a, b in zip(first[split], second[split]):
            if not np.array_equal(a["input"], b["input"]):
                return False
            if not np.array_equal(a["output"], b["output"]):
                return False
    return True


def _predict_from_demonstrations(train_pairs, test_input):
    demonstrated = set()
    for pair in train_pairs:
        source, target = pair["input"], pair["output"]
        _require(np.array_equal(_cellwise_expected(source), target),
                 "training examples do not establish the radial rule")
        info = _analyze_input(source)
        row0, col0 = info["center"]
        for index, (dr, dc) in enumerate(_DIRECTIONS):
            row, col = row0 + 2 * dr, col0 + 2 * dc
            if 0 <= row < info["height"] and 0 <= col < info["width"]:
                demonstrated.add(index)
    _require(demonstrated == set(range(8)),
             "training examples do not reveal every ray direction")
    return _cellwise_expected(test_input)


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, layouts, inputs, puzzle_keys = set(), set(), set(), set()
    color_counts = {"train": set(), "test": set()}
    backgrounds, touching, shared_cores = set(), 0, 0
    rectangular_tests, overlap_tests, small_tests, large_tests = 0, 0, 0, 0
    varied_background_puzzles, distinct_test_centers = 0, 0
    corner_quadrants = set()
    combined = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        training_copies = [{"input": pair["input"].copy(),
                            "output": pair["output"].copy()}
                           for pair in puzzle["train"]]
        for pair in puzzle["test"]:
            prediction = _predict_from_demonstrations(training_copies,
                                                      pair["input"].copy())
            _require(np.array_equal(prediction, pair["output"]),
                     "training-based test prediction is wrong")
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signature = tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                                for p in puzzle[split]) for split in ("train", "test"))
        puzzle_keys.add(signature)
        train_info = [_analyze_input(p["input"]) for p in puzzle["train"]]
        test_info = [_analyze_input(p["input"]) for p in puzzle["test"]]
        if len({i["background"] for i in train_info + test_info}) > 1:
            varied_background_puzzles += 1
        if any(i["center"] not in {t["center"] for t in train_info}
               for i in test_info):
            distinct_test_centers += 1
        train_shapes = {(i["height"], i["width"]) for i in train_info}
        for split, infos in (("train", train_info), ("test", test_info)):
            for info, pair in zip(infos, puzzle[split]):
                source, target = pair["input"], pair["output"]
                _require(np.array_equal(_cellwise_expected(source), target),
                         "cellwise prediction differs from the returned output")
                _require(not _matches_source_input(source), "source pair copied")
                shapes.add(source.shape)
                layouts.add(source.shape + info["center"])
                inputs.add(_grid_key(source))
                backgrounds.add(info["background"])
                distinct = len(set(info["emitters"]))
                color_counts[split].add(distinct)
                edge = (info["center"][0] in (1, info["height"] - 2)
                        or info["center"][1] in (1, info["width"] - 2))
                touching += int(edge)
                shared_cores += int(info["core"] in info["emitters"])
                if split == "test":
                    rectangular_tests += int(info["height"] != info["width"])
                    overlap_tests += int(source.shape in train_shapes)
                    small_tests += int(min(source.shape) <= 7)
                    large_tests += int(max(source.shape) >= 21)
                    if (info["center"][0] in (1, info["height"] - 2)
                            and info["center"][1] in (1, info["width"] - 2)):
                        corner_quadrants.add((info["center"][0] == 1,
                                              info["center"][1] == 1))
                    combined += int(info["height"] != info["width"] and edge
                                    and info["core"] in info["emitters"]
                                    and distinct < 8)

    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack variety")
    _require(len(puzzle_keys) == 42 and len(inputs) >= 180,
             "whole-puzzle or input diversity is too low")
    _require(len(shapes) >= 60 and len(layouts) >= 140,
             "geometry diversity is too low")
    _require(len(backgrounds) == 10 and varied_background_puzzles >= 20,
             "background roles lack local variation")
    _require(color_counts["train"] == set(range(1, 9))
             and color_counts["test"] == set(range(1, 9)),
             "emitter color counts do not span one through eight in both splits")
    _require(touching >= 25 and shared_cores >= 25,
             "edge sources or shared core colors are too scarce")
    _require(rectangular_tests >= 30 and overlap_tests >= 8
             and small_tests >= 8 and large_tests >= 15,
             "test canvas regimes are too narrow")
    _require(distinct_test_centers >= 30 and len(corner_quadrants) == 4
             and combined >= 3,
             "test positions or combined regimes are too narrow")

    _require(_puzzles_equal(generate_puzzle(314159), generate_puzzle(314159)),
             "automatic generation is not deterministic")
    _require(_puzzles_equal(generate_puzzle(271828, 7, 3),
                            generate_puzzle(271828, 7, 3)),
             "explicit generation is not deterministic")
    _require(not _puzzles_equal(generate_puzzle(123456, 7, 3),
                                generate_puzzle(123457, 7, 3)),
             "neighboring seeds give the same puzzle")
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (2001, None, 1), (2002, None, 4),
                              (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "train count ignored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "test count ignored")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(3001, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts did not raise ValueError")
