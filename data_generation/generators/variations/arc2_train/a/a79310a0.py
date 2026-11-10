import numpy as np


_AUTO_COUNTS = (
    (2, 1), (3, 1), (4, 1), (4, 2), (2, 1), (3, 2),
    (5, 1), (6, 2), (7, 3), (8, 2), (9, 3), (10, 4),
)
_SIMPLE_CLEAR = ("singleton", "hline")
_OVERLAP = ("vline", "rectangle", "ell", "tee", "fork", "step", "branch", "ring", "blob")
_ALL_KINDS = _SIMPLE_CLEAR + _OVERLAP
_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _grid_key(grid: np.ndarray) -> tuple:
    return (tuple(int(x) for x in grid.shape), tuple(int(x) for x in grid.ravel()))


def _pair_key(pair: dict) -> tuple:
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _source_pair_keys() -> frozenset:
    rows = (
        (("080", "000", "000"), ("000", "020", "000")),
        (("88000", "88000", "00000", "00000", "00000"),
         ("00000", "22000", "22000", "00000", "00000")),
        (("00000", "08880", "00000", "00000", "00000"),
         ("00000", "00000", "02220", "00000", "00000")),
        (("00800", "08800", "00800", "00000", "00000"),
         ("00000", "00200", "02200", "00200", "00000")),
    )
    def array(strings):
        return np.array([[int(c) for c in row] for row in strings], dtype=np.int8)
    return frozenset((_grid_key(array(a)), _grid_key(array(b))) for a, b in rows)


_SOURCE_PAIRS = _source_pair_keys()


def _count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        train, test = _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    elif train is None:
        minimum = max(2, test + 1)
        train = minimum + seed % (11 - minimum)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _crop(mask: np.ndarray) -> np.ndarray:
    cells = np.argwhere(mask)
    _check(len(cells) > 0, "empty object")
    lo = cells.min(axis=0)
    hi = cells.max(axis=0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]]


def _from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    coordinates = np.array(sorted(cells), dtype=int)
    coordinates -= coordinates.min(axis=0)
    height, width = coordinates.max(axis=0) + 1
    mask = np.zeros((height, width), dtype=bool)
    mask[coordinates[:, 0], coordinates[:, 1]] = True
    return mask


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def _make_mask(rng: np.random.Generator, kind: str, large: bool) -> np.ndarray:
    if kind == "singleton":
        return np.ones((1, 1), dtype=bool)
    if kind == "hline":
        return np.ones((1, _randint(rng, 8 if large else 2, 23 if large else 10)), dtype=bool)
    if kind == "vline":
        return np.ones((_randint(rng, 8 if large else 2, 23 if large else 10), 1), dtype=bool)
    if kind == "rectangle":
        return np.ones((_randint(rng, 5 if large else 2, 16 if large else 7),
                        _randint(rng, 5 if large else 2, 17 if large else 8)), dtype=bool)
    if kind == "ring":
        height = _randint(rng, 5 if large else 3, 16 if large else 9)
        width = _randint(rng, 5 if large else 3, 17 if large else 9)
        mask = np.ones((height, width), dtype=bool)
        mask[1:-1, 1:-1] = False
        return mask
    if kind == "ell":
        height = _randint(rng, 5 if large else 2, 17 if large else 9)
        width = _randint(rng, 5 if large else 2, 18 if large else 9)
        mask = np.zeros((height, width), dtype=bool)
        mask[:, 0] = True
        mask[-1, :] = True
        return np.rot90(mask, _randint(rng, 0, 3))
    if kind == "tee":
        small = not large and rng.random() < 0.75
        bar = 3 if small else _randint(rng, 5 if large else 3, 17 if large else 9)
        stem = 1 if small else _randint(rng, 4 if large else 1, 14 if large else 8)
        mask = np.zeros((stem + 1, bar), dtype=bool)
        mask[0, :] = True
        mask[:, _randint(rng, 1, bar - 2)] = True
        return np.rot90(mask, _randint(rng, 0, 3))
    if kind == "fork":
        stem = _randint(rng, 6 if large else 3, 16 if large else 10)
        left = _randint(rng, 2 if large else 1, 8 if large else 4)
        right = _randint(rng, 2 if large else 1, 9 if large else 5)
        if left == right:
            right += 1
        mask = np.zeros((stem, left + right + 1), dtype=bool)
        mask[:, left] = True
        mask[_randint(rng, 1, stem - 2), :] = True
        return np.rot90(mask, _randint(rng, 0, 3))
    if kind == "step":
        horizontal = _randint(rng, 5 if large else 2, 11 if large else 6)
        vertical = _randint(rng, 5 if large else 2, 11 if large else 6)
        tail = _randint(rng, 4 if large else 1, 10 if large else 6)
        cells = {(0, c) for c in range(horizontal)}
        cells.update((r, horizontal - 1) for r in range(vertical))
        cells.update((vertical - 1, c) for c in range(horizontal - 1, horizontal + tail))
        return np.rot90(_from_cells(cells), _randint(rng, 0, 3))
    if kind == "blob":
        height = _randint(rng, 5 if large else 2, 13 if large else 7)
        width = _randint(rng, 5 if large else 2, 13 if large else 8)
        mask = np.ones((height, width), dtype=bool)
        side = _randint(rng, 0, 3)
        if side < 2:
            extension = np.zeros((height, _randint(rng, 1, 4)), dtype=bool)
            extension[_randint(rng, 0, height - 1), :] = True
            mask = np.concatenate((extension, mask), axis=1)
        else:
            extension = np.zeros((_randint(rng, 1, 4), width), dtype=bool)
            extension[:, _randint(rng, 0, width - 1)] = True
            mask = np.concatenate((extension, mask), axis=0)
        return np.rot90(mask, _randint(rng, 0, 3))
    if kind == "branch":
        target = _randint(rng, 18 if large else 5, 45 if large else 17)
        cells = {(0, 0), (1, 0), (0, 1), (0, -1)}
        while len(cells) < target:
            candidates = sorted({(r + dr, c + dc) for r, c in cells
                                 for dr, dc in _DIRECTIONS} - cells)
            chosen = candidates[_randint(rng, 0, len(candidates) - 1)]
            cells.add(chosen)
        return _from_cells(cells)
    raise ValueError(f"unknown shape kind: {kind}")


def _place_and_transform(rng: np.random.Generator, mask: np.ndarray,
                         background: int, source: int, result: int) -> tuple[np.ndarray, np.ndarray]:
    mask = _crop(mask)
    object_height, object_width = mask.shape
    # Bound even a long random branch without changing its connected silhouette.
    if object_height > 27 or object_width > 28:
        raise ValueError("object exceeds canvas construction limit")
    canvas_mode = int(rng.choice(5, p=(0.23, 0.38, 0.21, 0.09, 0.09)))
    if canvas_mode == 0:
        extra_height, extra_width = _randint(rng, 1, 3), _randint(rng, 0, 2)
    elif canvas_mode == 1:
        extra_height, extra_width = _randint(rng, 2, 7), _randint(rng, 1, 7)
    elif canvas_mode == 2:
        extra_height, extra_width = _randint(rng, 7, 20), _randint(rng, 7, 20)
    elif canvas_mode == 3:
        extra_height, extra_width = _randint(rng, 12, 27), _randint(rng, 0, 6)
    else:
        extra_height, extra_width = _randint(rng, 1, 8), _randint(rng, 12, 27)
    height = min(30, object_height + extra_height)
    width = min(30, object_width + extra_width)
    top = _randint(rng, 0, height - object_height - 1)
    left = _randint(rng, 0, width - object_width)
    coordinates = np.argwhere(mask)
    rows, columns = coordinates[:, 0] + top, coordinates[:, 1] + left
    input_grid = np.full((height, width), background, dtype=np.int8)
    output_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[rows, columns] = source
    output_grid[rows + 1, columns] = result
    return input_grid, output_grid


def _shape_key(mask: np.ndarray) -> tuple:
    box = _crop(mask)
    return (box.shape, box.tobytes())


def _connected(mask: np.ndarray) -> bool:
    coordinates = np.argwhere(mask)
    if len(coordinates) == 0:
        return False
    first = tuple(int(v) for v in coordinates[0])
    seen = {first}
    stack = [first]
    while stack:
        row, col = stack.pop()
        for dr, dc in _DIRECTIONS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _analyze_pair(pair: dict) -> dict:
    source_grid, result_grid = pair["input"], pair["output"]
    _check(source_grid.shape == result_grid.shape, "grid dimensions change")
    before = {int(x) for x in np.unique(source_grid)}
    after = {int(x) for x in np.unique(result_grid)}
    common = before & after
    _check(len(before) == 2 and len(after) == 2 and len(common) == 1,
           "each pair needs distinct background, source, and result colors")
    background = next(iter(common))
    source = next(iter(before - common))
    result = next(iter(after - common))
    input_mask = source_grid == source
    output_mask = result_grid == result
    _check(_connected(input_mask), "source object must be connected")
    _check(_connected(output_mask), "result object must be connected")
    _check(np.any(source_grid == background) and np.any(result_grid == background),
           "background must remain visible")
    _check(not input_mask[-1].any(), "object cannot move down without clipping")
    expected = np.zeros_like(input_mask)
    expected[1:] = input_mask[:-1]
    _check(np.array_equal(expected, output_mask), "output mask must move one row down")
    _check(np.array_equal(result_grid, np.where(expected, result, background)),
           "vacated cells and holes must use the background")
    return {"background": background, "source": source, "result": result,
            "mask": input_mask, "overlap": bool(np.any(input_mask & expected))}


def _learn(train: list[dict]) -> tuple[int, int]:
    analyses = [_analyze_pair(pair) for pair in train]
    mappings = {(a["source"], a["result"]) for a in analyses}
    _check(len(mappings) == 1, "training color mapping changes")
    return next(iter(mappings))


def _predict(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    source, result = _learn(train)
    colors = {int(x) for x in np.unique(test_input)}
    _check(len(colors) == 2 and source in colors, "test source color was not learned")
    background = next(iter(colors - {source}))
    _check(background != result, "test background conflicts with learned result color")
    mask = test_input == source
    _check(not mask[-1].any(), "test translation would clip")
    expected = np.full(test_input.shape, background, dtype=test_input.dtype)
    expected[1:][mask[:-1]] = result
    return expected


def _choose_plan(rng: np.random.Generator, train_count: int,
                 test_count: int) -> tuple[list[str], list[str], int]:
    clear = str(rng.choice(_SIMPLE_CLEAR))
    overlap = str(rng.choice(_OVERLAP))
    train_kinds = [clear, overlap]
    while len(train_kinds) < train_count:
        options = [kind for kind in _ALL_KINDS if kind != train_kinds[-1]
                   and (kind != "singleton" or kind not in train_kinds)]
        train_kinds.append(str(rng.choice(options)))
    rng.shuffle(train_kinds)
    mode = _randint(rng, 0, 2)
    if mode == 0:  # Complex shape that need not have appeared in training.
        choices = ("tee", "tee", "tee", "tee", "fork", "step", "branch", "ring", "ell", "blob")
    elif mode == 1:  # Revisit a trained family with a fresh geometry and position.
        choices = tuple(k for k in train_kinds if k != "singleton") or tuple(train_kinds)
    else:  # A simple test after training on both clear and overlapping footprints.
        choices = ("singleton", "hline", "vline", "rectangle", "ell")
    test_kinds = [str(rng.choice(choices)) for _ in range(test_count)]
    return train_kinds, test_kinds, mode


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    source, result = (int(x) for x in rng.choice(10, size=2, replace=False))
    available_backgrounds = [x for x in range(10) if x not in (source, result)]
    background_count = int(rng.choice((1, 2, 3, 4), p=(0.30, 0.35, 0.25, 0.10)))
    backgrounds = [int(x) for x in rng.choice(available_backgrounds,
                                               size=background_count, replace=False)]
    train_kinds, test_kinds, mode = _choose_plan(rng, train_count, test_count)
    pair_root = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    train_shapes = set()
    for split_code, (split, kinds) in enumerate((("train", train_kinds), ("test", test_kinds))):
        for index, kind in enumerate(kinds):
            # Cycling guarantees that a variable-background puzzle demonstrates
            # that variation in training, while colors remain distinct.
            background = backgrounds[index % min(len(backgrounds), train_count)]
            if split == "test" and len(backgrounds) > 1:
                background = backgrounds[(index + 1) % len(backgrounds)]
            for attempt in range(256):
                pair_rng = np.random.default_rng(np.random.SeedSequence(
                    [pair_root, split_code, index, attempt]))
                large = bool(pair_rng.random() < 0.16)
                try:
                    mask = _make_mask(pair_rng, kind, large)
                    input_grid, output_grid = _place_and_transform(
                        pair_rng, mask, background, source, result)
                except ValueError:
                    continue
                pair = {"input": input_grid, "output": output_grid}
                key = _grid_key(input_grid)
                shape = _shape_key(input_grid == source)
                if key in used_inputs or _pair_key(pair) in _SOURCE_PAIRS:
                    continue
                if split == "test" and mode == 0 and shape in train_shapes:
                    continue
                if split == "train" and index >= 2 and shape in train_shapes:
                    continue
                puzzle[split].append(pair)
                used_inputs.add(key)
                if split == "train":
                    train_shapes.add(shape)
                break
            else:
                raise ValueError("could not construct enough unique pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "puzzle must have exactly train and test")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
           "splits must be lists")
    _check(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
           and len(puzzle["train"]) > len(puzzle["test"]), "invalid pair counts")
    inputs = set()
    pairs = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
                   "pair must have input and output")
            for grid in pair.values():
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2,
                       "grids must be two-dimensional arrays")
                _check(np.issubdtype(grid.dtype, np.integer), "grid values must be integers")
                _check(all(1 <= d <= 30 for d in grid.shape), "grid dimensions out of range")
                _check(int(grid.min()) >= 0 and int(grid.max()) <= 9, "invalid ARC color")
            input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            _check(input_key not in inputs and pair_key not in pairs, "duplicate pair or input")
            _check(pair_key not in _SOURCE_PAIRS, "source pair copied")
            inputs.add(input_key)
            pairs.add(pair_key)
            analyses[split].append(_analyze_pair(pair))
    source, result = _learn(puzzle["train"])
    _check(any(not a["overlap"] for a in analyses["train"]),
           "training needs a non-overlap clearing example")
    _check(any(a["overlap"] for a in analyses["train"]),
           "training needs an overlapping-footprint example")
    _check(len({_shape_key(a["mask"]) for a in analyses["train"]}) >= 2,
           "training shapes must differ")
    for pair, analysis in zip(puzzle["test"], analyses["test"]):
        _check((analysis["source"], analysis["result"]) == (source, result),
               "test changes learned color mapping")
        _check(np.array_equal(_predict(puzzle["train"], pair["input"]), pair["output"]),
               "test output is not determined from training")


def _is_four_t(mask: np.ndarray) -> bool:
    box = _crop(mask)
    if int(box.sum()) != 4 or box.shape not in ((2, 3), (3, 2)):
        return False
    for r, c in np.argwhere(box):
        if sum(0 <= r + dr < box.shape[0] and 0 <= c + dc < box.shape[1]
               and box[r + dr, c + dc] for dr, dc in _DIRECTIONS) == 3:
            return True
    return False


def _max_degree(mask: np.ndarray) -> int:
    box = _crop(mask)
    return max(sum(0 <= r + dr < box.shape[0] and 0 <= c + dc < box.shape[1]
                   and box[r + dr, c + dc] for dr, dc in _DIRECTIONS)
               for r, c in np.argwhere(box))


def _turn_count(mask: np.ndarray) -> int:
    box = _crop(mask)
    if _max_degree(box) > 2:
        return -1
    ends = []
    for r, c in np.argwhere(box):
        neighbors = [(int(r + dr), int(c + dc)) for dr, dc in _DIRECTIONS
                     if 0 <= r + dr < box.shape[0] and 0 <= c + dc < box.shape[1]
                     and box[r + dr, c + dc]]
        if len(neighbors) == 1:
            ends.append((int(r), int(c)))
    if len(ends) != 2:
        return -1
    previous = None
    current = ends[0]
    previous_step = None
    turns = 0
    while current != ends[1]:
        r, c = current
        neighbors = [(r + dr, c + dc) for dr, dc in _DIRECTIONS
                     if 0 <= r + dr < box.shape[0] and 0 <= c + dc < box.shape[1]
                     and box[r + dr, c + dc] and (r + dr, c + dc) != previous]
        if len(neighbors) != 1:
            return -1
        next_cell = neighbors[0]
        step = (next_cell[0] - r, next_cell[1] - c)
        turns += previous_step is not None and step != previous_step
        previous_step = step
        previous, current = current, next_cell
    return turns


def _asymmetric_branch(mask: np.ndarray) -> bool:
    box = _crop(mask)
    for r, c in np.argwhere(box):
        lengths = []
        for dr, dc in _DIRECTIONS:
            length = 0
            nr, nc = int(r + dr), int(c + dc)
            while 0 <= nr < box.shape[0] and 0 <= nc < box.shape[1] and box[nr, nc]:
                length += 1
                nr += dr
                nc += dc
            lengths.append(length)
        if sum(length > 0 for length in lengths) >= 3:
            if (lengths[0] and lengths[1] and lengths[0] != lengths[1]) or (
                    lengths[2] and lengths[3] and lengths[2] != lengths[3]):
                return True
    return False


def _family(mask: np.ndarray) -> str:
    box = _crop(mask)
    if box.size == 1:
        return "singleton"
    if box.shape[0] == 1:
        return "hline"
    if box.shape[1] == 1:
        return "vline"
    if box.all():
        return "rectangle"
    if _has_hole(box):
        return "hollow"
    if _max_degree(box) >= 3:
        return "branch"
    return "bent"


def _has_hole(mask: np.ndarray) -> bool:
    box = _crop(mask)
    outside = np.pad(~box, 1, constant_values=True)
    seen = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        r, c = stack.pop()
        for dr, dc in _DIRECTIONS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < outside.shape[0] and 0 <= nc < outside.shape[1]
                    and outside[nr, nc] and (nr, nc) not in seen):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return int(outside.sum()) > len(seen)


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple(_pair_key(pair) for pair in puzzle[split])
                 for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, mappings, dimensions, shapes, sizes = (set() for _ in range(6))
    clear_and_overlap = varied_background = simple_test = novel_test = 0
    four_t = holes = full_width = tight_down = large_grid = 0
    four_t_orientations = set()
    families = {"train": set(), "test": set()}
    first_clear = first_overlap = 0
    similar_family = novel_family = simple_test_seed = combined_seed = 0
    short_seed = long_seed = asymmetric_test = bent_train = step_train = 0
    large_object = above_bottom = source_majority = 0
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        mappings.add(_learn(puzzle["train"]))
        train_analyses = [_analyze_pair(pair) for pair in puzzle["train"]]
        train_shapes = {_shape_key(a["mask"]) for a in train_analyses}
        train_families = {_family(a["mask"]) for a in train_analyses}
        test_analyses = [_analyze_pair(pair) for pair in puzzle["test"]]
        test_families = {_family(a["mask"]) for a in test_analyses}
        clear_and_overlap += any(not a["overlap"] for a in train_analyses) and any(
            a["overlap"] for a in train_analyses)
        varied_background += len({a["background"] for a in train_analyses}) > 1
        first_clear += not train_analyses[0]["overlap"]
        first_overlap += train_analyses[0]["overlap"]
        similar_family += bool(train_families & test_families)
        novel_family += bool(test_families - train_families)
        simple_test_seed += any(int(a["mask"].sum()) <= 4 for a in test_analyses)
        short_seed += len(puzzle["train"]) <= 4
        long_seed += len(puzzle["train"]) >= 7
        combined_seed += (len({a["background"] for a in train_analyses}) > 1
                          and bool(test_families & {"branch", "hollow"})
                          and any(max(pair["input"].shape) >= 20
                                  for pair in puzzle["train"] + puzzle["test"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze_pair(pair)
                grid = pair["input"]
                box = _crop(a["mask"])
                families[split].add(_family(a["mask"]))
                dimensions.add(grid.shape)
                shapes.add(_shape_key(a["mask"]))
                sizes.add(int(a["mask"].sum()))
                full_width += box.shape[1] == grid.shape[1]
                tight_down += bool(a["mask"][-2].any())
                above_bottom += not bool(a["mask"][-2].any())
                large_grid += max(grid.shape) >= 20
                large_object += int(a["mask"].sum()) >= 30
                source_majority += int(a["mask"].sum()) > grid.size // 2
                if split == "train":
                    bent_train += _turn_count(a["mask"]) == 1
                    step_train += _turn_count(a["mask"]) >= 2
                if split == "test":
                    four_t += _is_four_t(a["mask"])
                    if _is_four_t(a["mask"]):
                        four_t_orientations.add(_shape_key(a["mask"]))
                    holes += _has_hole(a["mask"])
                    asymmetric_test += _asymmetric_branch(a["mask"])
                    simple_test += int(a["mask"].sum()) <= 4
                    novel_test += _shape_key(a["mask"]) not in train_shapes
                    expected_mask = np.zeros_like(a["mask"])
                    expected_mask[1:, :] = a["mask"][:-1, :]
                    _check(np.array_equal(pair["output"] == a["result"], expected_mask),
                           "self-test found an incorrect displacement")
                    _check(np.array_equal(_predict(puzzle["train"], grid), pair["output"]),
                           "test cannot be inferred from training")
    _check(len(train_counts) >= 4 and len(test_counts) >= 3, "pair-count diversity too low")
    _check(len(fingerprints) == 42 and len(mappings) >= 25, "puzzle diversity too low")
    _check(len(dimensions) >= 40 and len(shapes) >= 70 and len(sizes) >= 20,
           "structural diversity too low")
    _check(clear_and_overlap == 42 and varied_background >= 15,
           "training evidence or background diversity too low")
    _check(four_t >= 4 and len(four_t_orientations) >= 3 and holes >= 2
           and full_width >= 2 and tight_down >= 20 and above_bottom >= 20
           and large_grid >= 20 and large_object >= 15 and source_majority >= 1,
           "thin, hollow, tight, or large cases too rare")
    _check(simple_test >= 5 and novel_test >= 15, "test-regime diversity too low")
    _check(first_clear >= 5 and first_overlap >= 5 and similar_family >= 10
           and novel_family >= 4 and simple_test_seed >= 5
           and short_seed >= 15 and long_seed >= 5 and combined_seed >= 2,
           "train/test regime combinations too rare")
    _check({"singleton", "hline", "vline", "rectangle", "bent", "branch", "hollow"}
           <= families["train"] and {"hline", "vline", "rectangle", "bent", "branch", "hollow"}
           <= families["test"] and asymmetric_test >= 5 and bent_train >= 3
           and step_train >= 3, "shape-family diversity too low")
    for seed, train, test in ((8001, 2, 1), (8002, 10, 1), (8003, 10, 4),
                              (8004, 7, 3), (8101, None, 1), (8102, None, 4),
                              (8103, 2, None), (8104, 10, None)):
        a = generate_puzzle(seed, train, test)
        b = generate_puzzle(seed, train, test)
        _check(_puzzle_key(a) == _puzzle_key(b), "generator is not deterministic")
        if train is not None:
            _check(len(a["train"]) == train, "num_train not honored")
        if test is not None:
            _check(len(a["test"]) == test, "num_test not honored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (4, 4), (3, 4)):
        try:
            generate_puzzle(9001, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
