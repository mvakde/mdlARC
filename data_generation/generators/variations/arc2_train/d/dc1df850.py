import numpy as np


_SOURCE_PAIR_ROWS = (
    (
        (
            "00000002",
            "00000000",
            "00030000",
            "00000000",
            "00000080",
            "00000000",
            "00200000",
            "00000000",
        ),
        (
            "00000012",
            "00000011",
            "00030000",
            "00000000",
            "00000080",
            "01110000",
            "01210000",
            "01110000",
        ),
    ),
    (
        (
            "20000",
            "00020",
            "00000",
            "06000",
            "00000",
        ),
        (
            "21111",
            "11121",
            "00111",
            "06000",
            "00000",
        ),
    ),
    (
        (
            "00000",
            "02000",
            "00000",
            "00000",
        ),
        (
            "11100",
            "12100",
            "11100",
            "00000",
        ),
    ),
    (
        (
            "0000000070",
            "0020000000",
            "0000000000",
            "0000000200",
            "0000000000",
            "0700000000",
            "0000000000",
            "0000020000",
            "0000000000",
            "0000000005",
        ),
        (
            "0111000070",
            "0121000000",
            "0111001110",
            "0000001210",
            "0000001110",
            "0700000000",
            "0000111000",
            "0000121000",
            "0000111000",
            "0000000005",
        ),
    ),
)


def _array_from_rows(rows) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = tuple(
    (
        _array_from_rows(input_rows),
        _array_from_rows(output_rows),
    )
    for input_rows, output_rows in _SOURCE_PAIR_ROWS
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _checked_count(
    name: str,
    value: int | None,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(
            f"{name} must be between {lower} and {upper}"
        )
    return value


def _choose_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))
    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, maximum_test + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _apply_halo(
    input_grid: np.ndarray,
    background: int,
    seed_color: int,
    halo_color: int,
) -> np.ndarray:
    output_grid = input_grid.copy()
    height, width = input_grid.shape

    for row, col in np.argwhere(input_grid == seed_color):
        row = int(row)
        col = int(col)

        for neighbor_row in range(
            max(0, row - 1),
            min(height, row + 2),
        ):
            for neighbor_col in range(
                max(0, col - 1),
                min(width, col + 2),
            ):
                if input_grid[neighbor_row, neighbor_col] == background:
                    output_grid[neighbor_row, neighbor_col] = halo_color

    return output_grid


def _background(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    color = int(np.argmax(counts))
    _require(counts[color] > grid.size // 2, "background must occupy a majority")
    return color


def _chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _neighbors(point, height, width):
    row, col = point
    return [(r, c) for r in range(max(0, row - 1), min(height, row + 2))
            for c in range(max(0, col - 1), min(width, col + 2))
            if (r, c) != point]


def _cells(height, width):
    return [(r, c) for r in range(height) for c in range(width)]


def _boundary(point, height, width):
    return point[0] in (0, height - 1) or point[1] in (0, width - 1)


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _place_seeds(rng, height, width, count, style):
    cells = _cells(height, width)
    interior = [p for p in cells if not _boundary(p, height, width)]
    boundary = [p for p in cells if _boundary(p, height, width)]
    if style in ("overlap", "combined"):
        first_pool = boundary if style == "combined" else interior
        pairs = [(a, b) for a in first_pool for b in cells
                 if _chebyshev(a, b) == 2 and (style == "combined" or b in interior)]
        if not pairs:
            raise RuntimeError("no overlapping seed pair")
        initial = list(_pick(rng, pairs))
    elif style == "boundary":
        initial = [_pick(rng, boundary)]
    elif style in ("interior", "clean"):
        initial = [_pick(rng, interior)]
    else:
        initial = [_pick(rng, cells)]
    if len(initial) == count:
        return initial
    pool = interior if style in ("interior", "clean") else cells
    distance = 3 if style == "clean" else 2
    for _ in range(64):
        chosen = initial.copy()
        for index in rng.permutation(len(pool)):
            point = pool[int(index)]
            if all(_chebyshev(point, other) >= distance for other in chosen):
                chosen.append(point)
                if len(chosen) == count:
                    return chosen
    raise RuntimeError("could not place separated seeds")


def _make_pair(rng, seed_color, halo_color, background, regime, slot,
               seed_count, shape_override=None):
    if regime == "train":
        height = int(rng.integers(5, 11)) if slot < 2 else int(rng.integers(4, 11))
        width = int(rng.integers(5, 11)) if slot < 2 else int(rng.integers(4, 11))
        if slot == 0:
            style, seed_count = "boundary", 1
        elif slot == 1:
            style, seed_count = "overlap", 2
        else:
            style = _pick(rng, ("any", "boundary", "interior", "overlap"))
            if style == "overlap":
                seed_count = 2
    else:
        height, width = (shape_override if shape_override is not None else
                         (int(rng.integers(5, 19)), int(rng.integers(5, 19))))
        if seed_count >= 5:
            height, width = max(height, 8), max(width, 8)
        if seed_count == 1:
            style = _pick(rng, ("any", "boundary", "interior"))
        elif seed_count == 3 and rng.random() < 0.5:
            style = "clean"
        else:
            style = _pick(rng, ("any", "boundary", "interior", "overlap", "combined", "clean"))
            if style == "clean" and seed_count > 4:
                height, width = max(height, 10), max(width, 10)
        if style == "clean" and seed_count == 3:
            height, width = max(height, 7), max(width, 7)
    seeds = _place_seeds(rng, height, width, seed_count, style)
    grid = np.full((height, width), background, dtype=np.int8)
    for point in seeds:
        grid[point] = seed_color
    keep_clear = (set().union(*(set(_neighbors(seed, height, width)) for seed in seeds))
                  if regime == "train" and slot == 1 else set())
    max_inert = min((height * width - 1) // 2 - seed_count,
                    6 if regime == "train" else 8)
    if regime == "train" and slot == 0:
        count = int(rng.integers(2, 6))
    else:
        count = int(rng.integers(0, max_inert + 1))
    if style == "combined":
        count = max(1, count)
    count = min(count, max_inert)
    distractors = []
    force_block = (style != "clean" and not (regime == "train" and slot == 1) and
                   ((regime == "train" and slot == 0) or style == "combined"
                    or (count > 0 and rng.random() < 0.35)))
    if force_block and count:
        available = [p for seed in seeds for p in _neighbors(seed, height, width)
                     if grid[p] == background and p not in keep_clear]
        if available:
            point = _pick(rng, available)
            distractors.append(point)
            grid[point] = -1
    if regime == "train" and slot == 0:
        far = [p for p in _cells(height, width) if grid[p] == background
               and all(_chebyshev(p, seed) >= 2 for seed in seeds)]
        point = _pick(rng, far)
        distractors.append(point)
        grid[point] = -1
    for index in rng.permutation(height * width):
        if len(distractors) >= count:
            break
        point = (int(index) // width, int(index) % width)
        if grid[point] != background or point in keep_clear:
            continue
        if style == "clean" and any(_chebyshev(point, seed) <= 1 for seed in seeds):
            continue
        grid[point] = -1
        if all(sum(grid[p] == background for p in _neighbors(seed, height, width)) >= 2
               for seed in seeds):
            distractors.append(point)
        else:
            grid[point] = background
    if len(distractors) != count:
        raise RuntimeError("could not place distractors")
    inert_colors = [x for x in range(10) if x not in (background, seed_color, halo_color)]
    if count:
        unique_count = (2 if regime == "train" and slot == 0 else
                        int(rng.integers(1, min(4, count) + 1)))
        palette = [int(x) for x in rng.choice(inert_colors, size=unique_count, replace=False)]
        assignments = palette + [int(x) for x in rng.choice(palette, size=count - unique_count)]
        for index in rng.permutation(count):
            grid[distractors[int(index)]] = assignments[int(index)]
    return grid, _apply_halo(grid, background, seed_color, halo_color)


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _is_source_pair(input_grid, output_grid):
    return any(np.array_equal(input_grid, a) and np.array_equal(output_grid, b)
               for a, b in _SOURCE_PAIRS)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    seed_color, halo_color = [int(x) for x in rng.choice(10, size=2, replace=False)]
    backgrounds = [x for x in range(10) if x not in (seed_color, halo_color)]
    first_background = _pick(rng, backgrounds)
    second_background = _pick(rng, [x for x in backgrounds if x != first_background])
    test_seed_counts = [int(x) for x in rng.permutation(np.arange(1, 7))[:test_count]]
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, count in (("train", train_count), ("test", test_count)):
        for slot in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            shape_override = None
            if regime == "test" and test_seed_counts[slot] <= 3 and pair_rng.random() < 0.45:
                minimum = 7 if test_seed_counts[slot] == 3 else 5
                compatible = [pair["input"].shape for pair in puzzle["train"]
                              if min(pair["input"].shape) >= minimum]
                if compatible:
                    shape_override = _pick(pair_rng, compatible)
            for _ in range(96):
                background = (first_background if regime == "train" and slot == 0 else
                              second_background if regime == "train" and slot == 1 else
                              _pick(pair_rng, backgrounds))
                seed_count = (int(pair_rng.integers(1, 3)) if regime == "train" else
                              test_seed_counts[slot])
                try:
                    inp, out = _make_pair(pair_rng, seed_color, halo_color,
                                          background, regime, slot, seed_count,
                                          shape_override)
                except RuntimeError:
                    continue
                key = _grid_key(inp)
                if key in seen or _is_source_pair(inp, out):
                    continue
                seen.add(key)
                puzzle[regime].append({"input": inp, "output": out})
                break
            else:
                raise ValueError("could not generate a unique valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _infer_roles_from_training(train_pairs):
    backgrounds, halos = [], set()
    for pair in train_pairs:
        inp, out = pair["input"], pair["output"]
        background = _background(inp)
        changed = inp != out
        _require(bool(np.any(changed)), "each training pair must change")
        _require(np.all(inp[changed] == background), "only background may change")
        backgrounds.append(background)
        halos.update(int(x) for x in np.unique(out[changed]))
    _require(len(halos) == 1, "one halo color must be learned")
    halo = next(iter(halos))
    candidates = []
    for color in range(10):
        if color == halo or any(color == bg for bg in backgrounds):
            continue
        if all(np.any(pair["input"] == color) and
               np.array_equal(_apply_halo(pair["input"], bg, color, halo), pair["output"])
               for pair, bg in zip(train_pairs, backgrounds)):
            candidates.append(color)
    _require(len(candidates) == 1, "training must identify exactly one seed color")
    return candidates[0], halo


def _features(inp, out, seed_color):
    height, width = inp.shape
    background = _background(inp)
    seeds = [tuple(map(int, p)) for p in np.argwhere(inp == seed_color)]
    inert = [tuple(map(int, p)) for p in np.argwhere((inp != background) & (inp != seed_color))]
    clipped = any(_boundary(p, height, width) for p in seeds)
    interior = any(not _boundary(p, height, width) for p in seeds)
    overlap = any(_chebyshev(a, b) == 2 for i, a in enumerate(seeds) for b in seeds[i + 1:])
    blocked = any(_chebyshev(a, b) == 1 for a in seeds for b in inert)
    return {"seeds": seeds, "inert": inert, "clipped": clipped,
            "interior": interior, "overlap": overlap, "blocked": blocked,
            "inert_colors": {int(inp[p]) for p in inert}, "background": background,
            "shape": inp.shape, "changes": int(np.count_nonzero(inp != out))}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(all(isinstance(puzzle[x], list) for x in ("train", "test")),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input and output")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2 and
                         np.issubdtype(grid.dtype, np.integer), "grid must be a 2D integer array")
                _require(all(1 <= x <= 30 for x in grid.shape) and
                         np.all((grid >= 0) & (grid <= 9)), "illegal dimensions or colors")
            _require(inp.shape == out.shape and not np.array_equal(inp, out),
                     "dimensions or changed cells invalid")
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _require(not _is_source_pair(inp, out), "source pair copied")
    seed_color, halo_color = _infer_roles_from_training(puzzle["train"])
    features = {"train": [], "test": []}
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            inp, out = pair["input"], pair["output"]
            background = _background(inp)
            _require(background not in (seed_color, halo_color) and not np.any(inp == halo_color),
                     "color roles conflict")
            f = _features(inp, out, seed_color)
            seeds = f["seeds"]
            _require(seeds and all(_chebyshev(a, b) >= 2 for i, a in enumerate(seeds)
                                   for b in seeds[i + 1:]), "active seeds must be singletons")
            _require(all(sum(inp[p] == background for p in _neighbors(seed, *inp.shape)) >= 2
                         for seed in seeds), "each stamp must remain visible")
            _require(np.array_equal(_apply_halo(inp, background, seed_color, halo_color), out),
                     "background-only 3x3 halo rule violated")
            height, width = inp.shape
            if regime == "train":
                _require(4 <= height <= 10 and 4 <= width <= 10 and 1 <= len(seeds) <= 2,
                         "source_train regime violated")
            else:
                _require(5 <= height <= 18 and 5 <= width <= 18 and 1 <= len(seeds) <= 6,
                         "target_test regime violated")
            features[regime].append(f)
    training = features["train"]
    _require(len({f["background"] for f in training}) >= 2,
             "training must demonstrate variable backgrounds")
    _require(any(f["clipped"] for f in training), "training must demonstrate clipping")
    _require(any(f["interior"] for f in training), "training must demonstrate a full stamp")
    _require(any(
        any(not _boundary(seed, *inp.shape) and
            all(inp[p] == f["background"] for p in _neighbors(seed, *inp.shape))
            for seed in f["seeds"])
        for pair, f in zip(puzzle["train"], training)
        for inp in (pair["input"],)),
        "training must show an unobstructed interior stamp")
    _require(any(f["overlap"] for f in training), "training must demonstrate overlap")
    _require(any(
        any(inp[p] == f["background"] for p in
            set(_neighbors(f["seeds"][0], *inp.shape)) &
            set(_neighbors(f["seeds"][1], *inp.shape)))
        for pair, f in zip(puzzle["train"], training)
        for inp in (pair["input"],) if len(f["seeds"]) == 2 and f["overlap"]),
        "training must leave a shared halo cell visible")
    _require(any(f["blocked"] for f in training), "training must demonstrate inert blocking")
    _require(len(set().union(*(f["inert_colors"] for f in training))) >= 2,
             "training must distinguish active and inert colors")
    _require(len({len(f["seeds"]) for f in training}) == 2,
             "training must show one and two seeds")
    if len(features["test"]) > 1:
        _require(len({len(f["seeds"]) for f in features["test"]}) > 1,
                 "test seed counts must vary within a puzzle")


def _puzzles_equal(a, b):
    return all(len(a[regime]) == len(b[regime]) and
               all(np.array_equal(x[key], y[key]) for x, y in zip(a[regime], b[regime])
                   for key in ("input", "output")) for regime in ("train", "test"))


def _independent_stamp(inp, background, seed_color, halo_color):
    seeds = inp == seed_color
    covered = np.zeros(inp.shape, dtype=bool)
    height, width = inp.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), min(height, height + dr)
            c0, c1 = max(0, dc), min(width, width + dc)
            covered[r0:r1, c0:c1] |= seeds[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    result = inp.copy()
    result[covered & (inp == background)] = halo_color
    return result


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        seed_color, halo_color = _infer_roles_from_training(puzzle["train"])
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                inp = pair["input"]
                predicted = _independent_stamp(inp, _background(inp), seed_color, halo_color)
                _require(np.array_equal(predicted, pair["output"]),
                         "training-only prediction disagrees with ground truth")
    _require(len({len(p["train"]) for p in puzzles}) >= 4 and
             len({len(p["test"]) for p in puzzles}) >= 3, "automatic counts lack range")
    roles = {_infer_roles_from_training(p["train"]) for p in puzzles}
    _require(len(roles) >= 20, "puzzle color roles lack diversity")
    shapes, layouts = set(), set()
    test_seed_counts, test_distractor_counts, test_palette_sizes, test_changes = set(), set(), set(), set()
    clipped_tests = overlap_tests = blocked_tests = combined_tests = 0
    empty_tests = rich_tests = elongated_tests = 0
    for puzzle in puzzles:
        seed_color, _ = _infer_roles_from_training(puzzle["train"])
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                f = _features(pair["input"], pair["output"], seed_color)
                shapes.add(f["shape"])
                layouts.add((regime, f["shape"], tuple(f["seeds"]), tuple(f["inert"])))
                if regime == "test":
                    test_seed_counts.add(len(f["seeds"]))
                    test_distractor_counts.add(len(f["inert"]))
                    test_palette_sizes.add(len(f["inert_colors"]))
                    test_changes.add(f["changes"])
                    clipped_tests += f["clipped"]
                    overlap_tests += f["overlap"]
                    blocked_tests += f["blocked"]
                    combined_tests += f["clipped"] and f["overlap"] and f["blocked"]
                    empty_tests += not f["inert"]
                    rich_tests += len(f["inert_colors"]) >= 3
                    elongated_tests += max(f["shape"]) >= 2 * min(f["shape"])
    _require(len(shapes) >= 60 and len(layouts) >= 180, "geometric diversity collapsed")
    _require(test_seed_counts == set(range(1, 7)), "test seed counts lack range")
    _require(len(test_distractor_counts) >= 7 and len(test_palette_sizes) >= 4,
             "distractor schedules lack diversity")
    _require(len(test_changes) >= 20, "halo sizes lack diversity")
    _require(min(clipped_tests, overlap_tests, blocked_tests, empty_tests,
                 rich_tests, elongated_tests) >= 4 and combined_tests >= 2,
             "reviewed test variations or combinations are missing")
    for seed, train_count, test_count in ((1001, 2, 1), (1002, 10, 1),
                                          (1003, 10, 4), (1004, 7, 3)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        _require((len(puzzle["train"]), len(puzzle["test"])) ==
                 (train_count, test_count), "explicit counts ignored")
    for seed, train_count, test_count in ((1101, None, 1), (1102, None, 4),
                                          (1103, 2, None), (1104, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        _require((train_count is None or len(puzzle["train"]) == train_count) and
                 (test_count is None or len(puzzle["test"]) == test_count),
                 "partial counts ignored")
    _require(_puzzles_equal(generate_puzzle(1201, 8, 3), generate_puzzle(1201, 8, 3)),
             "generation is nondeterministic")
    _require(not _puzzles_equal(generate_puzzle(1201, 8, 3), generate_puzzle(1202, 8, 3)),
             "different seeds gave identical puzzles")
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5),
                                    (2, 2), (3, 4), (4, 4), (2.0, 1), (True, 1)):
        try:
            generate_puzzle(1301, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
