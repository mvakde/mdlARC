"""Generate two-panel ARC puzzles with a learned separator color."""

import numpy as np


_SOURCE_INPUTS = (
    ((7, 1), (1, 7), (6, 5), (5, 6)),
    ((4,), (0,)),
    ((2, 2, 2), (2, 8, 8), (2, 2, 2), (8, 7, 7), (8, 8, 7), (8, 7, 7)),
    ((1, 1, 1, 6, 1), (1, 1, 1, 1, 1), (6, 6, 6, 6, 6), (6, 1, 6, 6, 6)),
)


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _optional_count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    num_train = _optional_count("num_train", num_train, 2, 10)
    num_test = _optional_count("num_test", num_test, 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    else:
        num_test = 1 + seed % min(4, num_train - 1)
    return int(num_train), int(num_test)


def _transform(input_grid: np.ndarray, separator: int) -> np.ndarray:
    rows, cols = input_grid.shape
    height = rows // 2
    output = np.full((rows + 1, cols + 1), separator, dtype=np.int8)
    output[:height, :cols] = input_grid[:height]
    output[height + 1:, 1:] = input_grid[height:]
    return output


def _sample_shape(rng: np.random.Generator, regime: str) -> tuple[int, int]:
    # Both coordinates are drawn separately; the test range reaches farther.
    if regime == "source_train":
        heights = (1, 1, 2, 2, 3, 3, 4, 5, 6, 7)
        widths = (1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9)
    else:
        heights = (1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9)
        widths = (1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12)
    return (
        int(heights[int(rng.integers(len(heights)))]),
        int(widths[int(rng.integers(len(widths)))]),
    )


def _palette_pair(
    rng: np.random.Generator, separator: int, area: int, relation: str
) -> tuple[np.ndarray, np.ndarray]:
    available = np.array([c for c in range(10) if c != separator], dtype=np.int8)
    available = rng.permutation(available)
    if relation == "partial" and area < 2:
        relation = "full"
    if relation == "disjoint":
        upper_count = int(rng.integers(1, min(4, area) + 1))
        lower_count = int(rng.integers(1, min(4, area, 9 - upper_count) + 1))
        return available[:upper_count], available[upper_count:upper_count + lower_count]
    if relation == "partial":
        upper_count = int(rng.integers(2, min(4, area) + 1))
        lower_count = int(rng.integers(2, min(4, area) + 1))
        shared_count = int(rng.integers(1, min(upper_count, lower_count)))
        shared = available[:shared_count]
        upper_only = available[shared_count:upper_count]
        lower_only = available[upper_count:upper_count + lower_count - shared_count]
        return np.concatenate((shared, upper_only)), np.concatenate((shared, lower_only))
    if relation == "full":
        # An occasional one-color input is a valid, useful limit case.
        count = 1 if rng.random() < 0.13 else int(rng.integers(1, min(4, area) + 1))
        palette = available[:count]
        return palette, palette
    raise ValueError("unknown palette relation")


def _fill_panel(
    rng: np.random.Generator, height: int, width: int, palette: np.ndarray
) -> np.ndarray:
    count = len(palette)
    if count == 1:
        return np.full((height, width), palette[0], dtype=np.int8)
    mode = int(rng.integers(5))
    rr, cc = np.indices((height, width))
    phase = int(rng.integers(count))
    if mode == 0:  # irregular
        weights = rng.dirichlet(np.ones(count))
        indices = rng.choice(count, size=(height, width), p=weights)
    elif mode == 1:  # row motif
        indices = (rr + phase) % count
    elif mode == 2:  # column motif
        indices = (cc + phase) % count
    elif mode == 3:  # diagonal/check motif
        indices = (rr + cc + phase) % count
    else:  # patches of irregular size and location
        indices = np.full((height, width), phase, dtype=np.int64)
        for _ in range(int(rng.integers(1, count + 3))):
            r0 = int(rng.integers(height))
            c0 = int(rng.integers(width))
            r1 = int(rng.integers(r0 + 1, height + 1))
            c1 = int(rng.integers(c0 + 1, width + 1))
            indices[r0:r1, c0:c1] = int(rng.integers(count))
    if rng.random() < 0.5:
        edits = int(rng.integers(1, max(2, height * width // 5 + 1)))
        positions = rng.choice(height * width, size=edits, replace=False)
        indices.flat[positions] = rng.integers(count, size=edits)
    # Realize the palette exactly, including in small panels.
    missing = [i for i in range(count) if not np.any(indices == i)]
    if missing:
        positions = rng.choice(height * width, size=count, replace=False)
        indices.flat[positions] = np.arange(count)
    return np.asarray(palette[indices], dtype=np.int8)


def _make_input(
    rng: np.random.Generator, shape: tuple[int, int], separator: int, relation: str
) -> np.ndarray:
    height, width = shape
    upper_palette, lower_palette = _palette_pair(rng, separator, height * width, relation)
    upper = _fill_panel(rng, height, width, upper_palette)
    if relation == "full" and rng.random() < 0.16:
        lower = upper.copy()
    else:
        lower = _fill_panel(rng, height, width, lower_palette)
    return np.vstack((upper, lower))


def _grid_key(grid: np.ndarray) -> tuple:
    return (tuple(grid.shape), tuple(int(v) for v in grid.flat))


def _source_input(input_grid: np.ndarray) -> bool:
    return any(input_grid.tolist() == [list(row) for row in source] for source in _SOURCE_INPUTS)


def _draw_pair(
    rng: np.random.Generator,
    shape: tuple[int, int],
    separator: int,
    relation: str,
    seen: set,
) -> dict:
    for _ in range(300):
        input_grid = _make_input(rng, shape, separator, relation)
        key = _grid_key(input_grid)
        if key not in seen and not _source_input(input_grid):
            seen.add(key)
            return {"input": input_grid, "output": _transform(input_grid, separator)}
    raise RuntimeError("could not draw a unique pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _normalize_seed(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # The separator is the sole puzzle-level convention.
    separator = int(rng.integers(10))
    source_like = rng.random() < 0.23

    if source_like:
        sizes = [int(v) for v in rng.permutation(np.array([1, 2, 3, 4]))]
        train_shapes = [(sizes[i % len(sizes)], sizes[i % len(sizes)]) for i in range(num_train)]
    else:
        anchor = int(rng.integers(2, 6))
        train_shapes = [(anchor, anchor)]
        while len(train_shapes) < num_train:
            shape = _sample_shape(rng, "source_train")
            if shape not in train_shapes or len(train_shapes) >= 2:
                train_shapes.append(shape)

    relations = ["disjoint", "full"]
    rng.shuffle(relations)
    while len(relations) < num_train:
        relations.append(("disjoint", "partial", "full")[int(rng.integers(3))])
    # Shuffle the remaining axes separately, so shape, palette, and texture combine.
    rng.shuffle(train_shapes)
    rng.shuffle(relations)

    puzzle = {"train": [], "test": []}
    seen: set[tuple] = set()
    for shape, relation in zip(train_shapes, relations):
        puzzle["train"].append(_draw_pair(rng, shape, separator, relation, seen))

    for _ in range(num_test):
        if source_like:
            shape = (2, 5) if rng.random() < 0.75 else (
                int(rng.integers(1, 4)), int(rng.integers(4, 10))
            )
        elif rng.random() < 0.27:
            shape = train_shapes[int(rng.integers(num_train))]
        else:
            shape = _sample_shape(rng, "target_test")
        relation = ("disjoint", "partial", "full")[int(rng.integers(3))]
        puzzle["test"].append(_draw_pair(rng, shape, separator, relation, seen))

    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid: np.ndarray, name: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{name} must be an ndarray")
    _require(grid.ndim == 2, f"{name} must be two-dimensional")
    _require(all(1 <= n <= 30 for n in grid.shape), f"{name} dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), f"{name} must be integer")
    _require(np.all((0 <= grid) & (grid <= 9)), f"{name} colors must be 0..9")


def _validate_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> int:
    rows, cols = input_grid.shape
    _require(rows % 2 == 0, "input height must be even")
    height = rows // 2
    _require(output_grid.shape == (rows + 1, cols + 1), "wrong output shape")
    cross = np.concatenate((
        output_grid[height, :],
        output_grid[:height, cols],
        output_grid[height + 1:, 0],
    ))
    separator = int(cross[0])
    _require(np.all(cross == separator), "inserted cross must have one color")
    _require(not np.any(input_grid == separator), "separator must be absent from input")
    _require(np.array_equal(output_grid, _transform(input_grid, separator)),
             "panel contents or offset changed")
    return separator


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    seen: set[tuple] = set()
    separators: set[int] = set()
    train_shapes: set[tuple[int, int]] = set()
    train_relations: set[str] = set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            name = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{name} must contain input and output")
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp, name + " input")
            _validate_grid(out, name + " output")
            key = _grid_key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(not _source_input(inp), "copied source input")
            separators.add(_validate_pair(inp, out))
            if split == "train":
                height = inp.shape[0] // 2
                train_shapes.add((height, inp.shape[1]))
                upper = set(int(v) for v in np.unique(inp[:height]))
                lower = set(int(v) for v in np.unique(inp[height:]))
                train_relations.add("disjoint" if not upper & lower else "shared")
    _require(len(separators) == 1, "one separator must be learned across the puzzle")
    _require(len(train_shapes) >= 2, "training needs distinct panel dimensions")
    _require(train_relations == {"disjoint", "shared"},
             "training needs both disjoint and shared panel palettes")


def _equal_puzzles(first: dict, second: dict) -> bool:
    return all(
        len(first[split]) == len(second[split])
        and all(
            np.array_equal(a[role], b[role])
            for a, b in zip(first[split], second[split])
            for role in ("input", "output")
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack variation")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack variation")
    _require(len({_validate_pair(p["train"][0]["input"], p["train"][0]["output"])
                  for p in puzzles}) >= 8, "separator colors lack variation")

    shapes = {split: set() for split in ("train", "test")}
    relations = {split: set() for split in ("train", "test")}
    content = set()
    shared_ratio_puzzles = 0
    source_like_puzzles = 0
    nontrivial_disjoint = {"train": 0, "test": 0}
    large_tests = 0
    nonmonochrome_identical = 0
    for puzzle in puzzles:
        train_shapes = {(p["input"].shape[0] // 2, p["input"].shape[1]) for p in puzzle["train"]}
        if all(h == w for h, w in train_shapes) and any(
            p["input"].shape[0] // 2 < p["input"].shape[1] for p in puzzle["test"]
        ):
            source_like_puzzles += 1
        if any((p["input"].shape[0] // 2, p["input"].shape[1]) in train_shapes
               for p in puzzle["test"]):
            shared_ratio_puzzles += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                height, width = inp.shape[0] // 2, inp.shape[1]
                shapes[split].add((height, width))
                upper, lower = inp[:height], inp[height:]
                a, b = set(int(v) for v in np.unique(upper)), set(int(v) for v in np.unique(lower))
                relations[split].add(
                    "disjoint" if not a & b else "full" if a == b else "partial"
                )
                if np.array_equal(upper, lower):
                    content.add("identical")
                    if len(a | b) > 1:
                        nonmonochrome_identical += 1
                if len(a | b) == 1:
                    content.add("monochrome")
                if not a & b and height * width > 1:
                    nontrivial_disjoint[split] += 1
                if split == "test" and height >= 7 and width >= 10:
                    large_tests += 1
    _require(all(len(shapes[s]) >= 20 for s in shapes), "panel shapes lack variation")
    _require(all(relations[s] == {"disjoint", "partial", "full"} for s in relations),
             "palette relations lack variation")
    _require(content == {"identical", "monochrome"}, "limit cases absent")
    _require(shared_ratio_puzzles >= 3, "train/test shape sharing absent")
    _require(source_like_puzzles >= 1, "source-like square/wide shift absent")
    _require(all(nontrivial_disjoint[s] >= 1 for s in nontrivial_disjoint),
             "nontrivial disjoint panels absent")
    _require(large_tests >= 1, "large rectangular tests absent")
    _require(nonmonochrome_identical >= 1, "identical patterned panels absent")
    _require(any(h > w for h, w in shapes["test"]) and
             any(w > h for h, w in shapes["test"]) and
             any(h == w for h, w in shapes["test"]), "test aspect ratios lack variation")

    for seed, train, test in ((100, 2, 1), (101, 10, 4), (102, 10, 1)):
        a = generate_puzzle(seed, train, test)
        b = generate_puzzle(seed, train, test)
        _require(_equal_puzzles(a, b), "specified-count generation is nondeterministic")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2}):
        try:
            generate_puzzle(500, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
