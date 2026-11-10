"""Compress a chain of nested rectangles into canonical colored rings."""

import hashlib

import numpy as np


# (input shape, output shape, SHA-256 of flattened input then output)
_SOURCE_PAIRS = {
    ((22, 25), (3, 3), "0d494429bc963dfffc237f1823905f1c5025fba324252ea636a30a3da6bacbac"),
    ((23, 27), (5, 5), "6a7b99adf5941791b4eeecec9ce9149c57c709734b0f1de881cef9e147806e0b"),
    ((21, 22), (7, 7), "66c854a10f44b4527e2dee7cc619a477dce714082e57d5e3653d26287b082b35"),
    ((26, 27), (9, 9), "67dfbb7667563ed5de19222b3a100057ab5a9ed8bf3fe3d510ee0befd0ce88a1"),
}


def _integer_argument(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    if num_train is not None:
        if not _integer_argument(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _integer_argument(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    # The source family has no hidden color or direction convention.
    return {}


def _train_depths(rng: np.random.Generator, count: int) -> list[int]:
    if count >= 3 and rng.random() < 0.25:
        start = int(rng.integers(2, 5))
        return [start, start + 1, start + 2] + [
            int(rng.integers(2, 8)) for _ in range(count - 3)
        ]
    low = int(rng.integers(2, 5))
    high = int(rng.integers(max(4, low + 1), 8))
    result = [low, high] + [int(rng.integers(2, 8)) for _ in range(count - 2)]
    rng.shuffle(result)
    return result


def _test_depth(rng: np.random.Generator, training: list[int]) -> int:
    low, high = min(training), max(training)
    mode = int(rng.integers(0, 3))
    if mode == 0:
        return int(rng.choice(training))
    if mode == 1:
        return int(rng.integers(2, low if low > 2 else high))
    return int(rng.integers(high + 1, 9))


def _axis_length(rng: np.random.Generator, depth: int) -> int:
    minimum = max(12, 2 * depth - 1)
    mode = rng.random()
    if mode < 0.28:
        low, high = minimum, min(30, minimum + 4)
    elif mode < 0.56:
        low, high = max(minimum, 24), 30
    else:
        low, high = minimum, 30
    return int(rng.integers(low, high + 1))


def _shrink(rng: np.random.Generator, maximum: int) -> int:
    assert maximum >= 2
    mode = rng.random()
    if mode < 0.34:
        low, high = 2, min(maximum, 4)
    elif mode < 0.67:
        low, high = 2, maximum
    else:
        low, high = max(2, maximum - 5), maximum
    return int(rng.integers(low, high + 1))


def _bounds(rng: np.random.Generator, height: int, width: int, depth: int) -> list[tuple[int, int, int, int]]:
    result = [(0, height, 0, width)]
    point_center = rng.random() < 0.12
    for level in range(1, depth):
        top, bottom, left, right = result[-1]
        minimum_child = 1 + 2 * (depth - 1 - level)
        vmax = bottom - top - minimum_child
        hmax = right - left - minimum_child
        if point_center and level == depth - 1:
            vs, hs = vmax, hmax
        else:
            vs, hs = _shrink(rng, vmax), _shrink(rng, hmax)
        tm = int(rng.integers(1, vs))
        lm = int(rng.integers(1, hs))
        result.append((top + tm, bottom - (vs - tm), left + lm, right - (hs - lm)))
    return result


def _colors(rng: np.random.Generator, depth: int, background: int, recurrence: str) -> list[int]:
    other = np.array([c for c in range(10) if c != background], dtype=np.int8)
    rng.shuffle(other)
    result = [background] + [int(c) for c in other[:depth - 1]]
    if recurrence == "outer_center":
        assert depth >= 3
        result[-1] = result[0]
    elif recurrence == "interior":
        assert depth >= 5
        result[3] = result[1]
    elif recurrence == "multiple":
        assert depth >= 5
        result[2] = result[0]
        result[4] = result[0]
    else:
        assert recurrence == "distinct"
    assert all(a != b for a, b in zip(result, result[1:]))
    return result


def _canonical_output(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    out = np.empty((side, side), dtype=np.int8)
    for index, color in enumerate(colors):
        out[index:side - index, index:side - index] = color
    return out


def _make_pair(
    rng: np.random.Generator, depth: int, background: int, recurrence: str,
    dimensions: tuple[int, int] | None = None,
) -> dict:
    height, width = dimensions if dimensions is not None else (
        _axis_length(rng, depth), _axis_length(rng, depth)
    )
    bounds = _bounds(rng, height, width, depth)
    colors = _colors(rng, depth, background, recurrence)
    inp = np.full((height, width), background, dtype=np.int8)
    for color, (top, bottom, left, right) in zip(colors[1:], bounds[1:]):
        inp[top:bottom, left:right] = color
    return {"input": inp, "output": _canonical_output(colors)}


def _grid_key(grid: np.ndarray) -> tuple:
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _source_signature(pair: dict) -> tuple:
    inp, out = pair["input"], pair["output"]
    digest = hashlib.sha256(inp.astype(np.int8).tobytes() + out.astype(np.int8).tobytes()).hexdigest()
    return tuple(inp.shape), tuple(out.shape), digest


def _analyze_nested_grid(grid: np.ndarray) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    height, width = grid.shape
    color = int(grid[0, 0])
    assert np.all(grid[0, :] == color) and np.all(grid[-1, :] == color)
    assert np.all(grid[:, 0] == color) and np.all(grid[:, -1] == color)
    top, bottom, left, right = 0, height, 0, width
    colors: list[int] = []
    bounds: list[tuple[int, int, int, int]] = []
    while True:
        colors.append(color)
        bounds.append((top, bottom, left, right))
        current = grid[top:bottom, left:right]
        ys, xs = np.nonzero(current != color)
        if not len(ys):
            break
        rt, rb = int(ys.min()), int(ys.max()) + 1
        rl, rr = int(xs.min()), int(xs.max()) + 1
        assert 0 < rt < rb < current.shape[0]
        assert 0 < rl < rr < current.shape[1]
        outside = np.ones(current.shape, dtype=bool)
        outside[rt:rb, rl:rr] = False
        assert np.all(current[outside] == color)
        top, bottom = top + rt, top + rb
        left, right = left + rl, left + rr
        child = grid[top:bottom, left:right]
        color = int(child[0, 0])
        assert color != colors[-1]
        assert np.all(child[0, :] == color) and np.all(child[-1, :] == color)
        assert np.all(child[:, 0] == color) and np.all(child[:, -1] == color)
        assert len(colors) < 8
    return colors, bounds


def _has_recurrence(colors: list[int]) -> bool:
    return len(set(colors)) < len(colors)


def generate_puzzle(
    seed: int, num_train: int | None = None, num_test: int | None = None
) -> dict:
    if not _integer_argument(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    sample_puzzle_parameters(rng)
    training = _train_depths(rng, num_train)
    deepest_index = max(range(num_train), key=lambda i: training[i])
    if training[deepest_index] >= 5:
        recurrence_style = ("outer_center", "interior", "multiple")[int(rng.integers(0, 3))]
    else:
        recurrence_style = "outer_center"
    testing = [_test_depth(rng, training) for _ in range(num_test)]

    puzzle = {"train": [], "test": []}
    seen_inputs: set[tuple] = set()
    first_background: int | None = None
    for split, depths in (("train", training), ("test", testing)):
        for index, depth in enumerate(depths):
            if split == "train" and index == deepest_index:
                recurrence = recurrence_style
            elif split == "test":
                recurrence = recurrence_style if rng.random() < 0.55 else "distinct"
            elif rng.random() < 0.30:
                recurrence = recurrence_style
            else:
                recurrence = "distinct"
            if depth < 5 and recurrence in ("interior", "multiple"):
                recurrence = "outer_center" if depth >= 3 and rng.random() < 0.4 else "distinct"
            if depth < 3:
                recurrence = "distinct"
            for _ in range(32):
                child_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
                pair_rng = np.random.default_rng(child_seed)
                if split == "train" and index == 1:
                    background = int(pair_rng.choice([c for c in range(10) if c != first_background]))
                else:
                    background = int(pair_rng.integers(0, 10))
                dimensions = None
                if split == "test" and pair_rng.random() < 0.40:
                    compatible = [tuple(p["input"].shape) for p in puzzle["train"]
                                  if min(p["input"].shape) >= 2 * depth - 1]
                    if compatible:
                        dimensions = compatible[int(pair_rng.integers(0, len(compatible)))]
                pair = _make_pair(pair_rng, depth, background, recurrence, dimensions)
                key = _grid_key(pair["input"])
                if (key in seen_inputs or _source_signature(pair) in _SOURCE_PAIRS
                        or (pair["input"].shape == pair["output"].shape
                            and np.array_equal(pair["input"], pair["output"]))):
                    continue
                seen_inputs.add(key)
                puzzle[split].append(pair)
                if split == "train" and index == 0:
                    first_background = background
                break
            else:
                raise ValueError("could not construct a novel nested rectangle")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    training_depths = []
    training_recurrence = False
    test_recurrence = False
    backgrounds = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for role in ("input", "output"):
                grid = pair[role]
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            input_key, output_key = _grid_key(inp), _grid_key(out)
            assert input_key not in seen_inputs
            assert (input_key, output_key) not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            assert _source_signature(pair) not in _SOURCE_PAIRS
            assert inp.shape != out.shape or not np.array_equal(inp, out)
            colors, bounds = _analyze_nested_grid(inp)
            assert 2 <= len(colors) <= 8
            assert np.array_equal(out, _canonical_output(colors))
            assert out.shape == (2 * len(colors) - 1, 2 * len(colors) - 1)
            for parent, child in zip(bounds, bounds[1:]):
                assert parent[0] < child[0] < child[1] < parent[1]
                assert parent[2] < child[2] < child[3] < parent[3]
            backgrounds.append(colors[0])
            if split == "train":
                training_depths.append(len(colors))
                training_recurrence |= _has_recurrence(colors)
            else:
                test_recurrence |= _has_recurrence(colors)
    assert len(set(training_depths)) >= 2
    assert training_recurrence
    assert not test_recurrence or training_recurrence
    assert backgrounds[0] != backgrounds[1]


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def _raises_value_error(train: int | None, test: int | None) -> bool:
    try:
        generate_puzzle(17, train, test)
    except ValueError:
        return True
    return False


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                colors, _ = _analyze_nested_grid(pair["input"])
                out = pair["output"]
                for level, color in enumerate(colors):
                    end = out.shape[0] - 1 - level
                    assert np.all(out[level, level:end + 1] == color)
                    assert np.all(out[end, level:end + 1] == color)
                    assert np.all(out[level:end + 1, level] == color)
                    assert np.all(out[level:end + 1, end] == color)
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3))):
        puzzle = generate_puzzle(100 + index, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
        validate_generated_puzzle(puzzle)
    for index, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(200 + index, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    first = generate_puzzle(987654, 7, 3)
    assert _puzzle_key(first) == _puzzle_key(generate_puzzle(987654, 7, 3))
    assert _puzzle_key(first) != _puzzle_key(generate_puzzle(987655, 7, 3))
    assert all(_raises_value_error(a, b) for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)))
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) >= 40

    shapes, layouts, train_depths, test_depths = set(), set(), set(), set()
    distinct_tests = recurring_tests = interior_recurring = triple_recurring = 0
    point_centers = mixed_bands = wide = tall = overlapping_sizes = source_shapes = 0
    relations = set()
    for puzzle in puzzles:
        training_depths = [len(_analyze_nested_grid(p["input"])[0]) for p in puzzle["train"]]
        train_shapes = {p["input"].shape for p in puzzle["train"]}
        test_shapes = {p["input"].shape for p in puzzle["test"]}
        overlapping_sizes += bool(train_shapes & test_shapes)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                colors, bounds = _analyze_nested_grid(inp)
                depth = len(colors)
                shapes.add(inp.shape)
                layouts.add((inp.shape, tuple(bounds)))
                (train_depths if split == "train" else test_depths).add(depth)
                source_shapes += inp.shape in {(22, 25), (23, 27), (21, 22), (26, 27)}
                tall += inp.shape[0] >= inp.shape[1] + 6
                wide += inp.shape[1] >= inp.shape[0] + 6
                point_centers += bounds[-1][1] - bounds[-1][0] == 1 and bounds[-1][3] - bounds[-1][2] == 1
                margins = [m for a, b in zip(bounds, bounds[1:]) for m in (b[0]-a[0], a[1]-b[1], b[2]-a[2], a[3]-b[3])]
                mixed_bands += min(margins) == 1 and max(margins) >= 4
                if split == "test":
                    distinct_tests += len(set(colors)) == depth
                    recurring_tests += _has_recurrence(colors)
                    if depth < min(training_depths):
                        relations.add("below_all")
                    if depth in training_depths:
                        relations.add("match")
                    if depth > max(training_depths):
                        relations.add("extend")
                if any(colors[i] == colors[j] for i in range(1, depth - 1) for j in range(i + 2, depth - 1)):
                    interior_recurring += 1
                if any(colors.count(c) >= 3 for c in set(colors)):
                    triple_recurring += 1
    assert len(shapes) >= 80 and len(layouts) >= 180
    assert train_depths >= {2, 3, 4, 5, 6, 7}
    assert test_depths >= {2, 3, 4, 5, 6, 7, 8}
    assert relations == {"below_all", "match", "extend"}
    assert distinct_tests >= 8 and recurring_tests >= 8
    assert interior_recurring >= 4 and triple_recurring >= 4
    assert point_centers >= 4 and mixed_bands >= 25
    assert wide >= 10 and tall >= 10
    assert overlapping_sizes >= 1 and source_shapes >= 1
