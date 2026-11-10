import numpy as np


_SOURCE_PAIR_ROWS = {
    (
        (
            "9000000300",
            "0000000000",
            "0000008003",
            "0009000000",
            "0000000000",
            "0000070000",
            "0000000000",
            "0800000000",
            "0000000000",
            "0000000007",
        ),
        (
            "9000000300",
            "0900000030",
            "0090008003",
            "0009080000",
            "0000800000",
            "0008070000",
            "0080007000",
            "0800000700",
            "0000000070",
            "0000000007",
        ),
    ),
    (
        (
            "0020060000",
            "0000000000",
            "2000000000",
            "0000000000",
            "0000000006",
            "0004000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000400",
        ),
        (
            "0020060000",
            "0200006000",
            "2000000600",
            "0000000060",
            "0000000006",
            "0004000000",
            "0000400000",
            "0000040000",
            "0000004000",
            "0000000400",
        ),
    ),
    (
        (
            "0006080000",
            "0000000000",
            "0040000000",
            "6000000000",
            "0000000008",
            "0000900000",
            "0000004000",
            "0000000000",
            "0000000000",
            "9000000000",
        ),
        (
            "0006080000",
            "0060008000",
            "0640000800",
            "6004000080",
            "0000400008",
            "0000940000",
            "0009004000",
            "0090000000",
            "0900000000",
            "9000000000",
        ),
    ),
    (
        (
            "0000003009",
            "7000000000",
            "0000000000",
            "0003009000",
            "6000000000",
            "0000000004",
            "0000000000",
            "0006007000",
            "0000000000",
            "0000040000",
        ),
        (
            "0000003009",
            "7000030090",
            "0700300900",
            "0073009000",
            "6007000000",
            "0600700004",
            "0060070040",
            "0006007400",
            "0000004000",
            "0000040000",
        ),
    ),
}


def _validate_seed(seed: int) -> int:
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
    ):
        raise ValueError("seed must be a non-negative integer")

    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _validate_optional_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_optional_count(
        "num_train", num_train, 2, 10
    )
    num_test = _validate_optional_count(
        "num_test", num_test, 1, 4
    )

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum_train = max(2, num_test + 1)
        span = 11 - minimum_train
        num_train = minimum_train + (seed // 4) % span
    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        span = 11 - minimum_train
        num_train = minimum_train + seed % span
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = 1 + seed % maximum_test

    if not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return int(num_train), int(num_test)


def _grid_key(grid: np.ndarray) -> tuple:
    normalized = np.ascontiguousarray(grid, dtype=np.int16)
    return tuple(grid.shape), normalized.tobytes()


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        tuple(
            (
                _grid_key(pair["input"]),
                _grid_key(pair["output"]),
            )
            for pair in puzzle[split]
        )
        for split in ("train", "test")
    )


def _grid_rows(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _is_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    return (
        _grid_rows(input_grid),
        _grid_rows(output_grid),
    ) in _SOURCE_PAIR_ROWS


def _infer_background(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    maximum = int(counts.max())
    assert int(np.count_nonzero(counts == maximum)) == 1, (
        "background must be uniquely dominant"
    )
    return int(np.argmax(counts))


def _segments(input_grid: np.ndarray) -> tuple[int, list[dict]]:
    """Recover every complete, nonintersecting diagonal from endpoint colors."""
    background = _infer_background(input_grid)
    segments = []
    occupied = set()
    for color in range(10):
        if color == background or not np.any(input_grid == color):
            continue
        endpoints = np.argwhere(input_grid == color)
        assert endpoints.shape == (2, 2), "each color needs two endpoints"
        upper, lower = sorted((tuple(map(int, p)) for p in endpoints))
        gap = lower[0] - upper[0]
        assert gap >= 2, "a segment needs an interior pixel"
        assert abs(lower[1] - upper[1]) == gap, "endpoints must align at 45 degrees"
        direction = 1 if lower[1] > upper[1] else -1
        cells = tuple((upper[0] + step, upper[1] + direction * step)
                      for step in range(gap + 1))
        assert all(0 <= r < input_grid.shape[0] and 0 <= c < input_grid.shape[1]
                   for r, c in cells)
        assert occupied.isdisjoint(cells), "different segments overlap"
        occupied.update(cells)
        assert all(int(input_grid[r, c]) == background for r, c in cells[1:-1]), (
            "segment interior is not blank"
        )
        segments.append({"color": color, "upper": upper, "lower": lower,
                         "gap": gap, "direction": direction, "cells": cells})
    assert segments, "at least one endpoint pair is needed"
    return background, segments


def _construct_output(input_grid: np.ndarray) -> np.ndarray:
    output = input_grid.copy()
    _, segments = _segments(input_grid)
    for segment in segments:
        for row, col in segment["cells"][1:-1]:
            output[row, col] = segment["color"]
    return output


def _metadata(input_grid: np.ndarray) -> dict:
    background, segments = _segments(input_grid)
    return {"background": background,
            "colors": tuple(s["color"] for s in segments),
            "count": len(segments), "shape": tuple(input_grid.shape),
            "gaps": tuple(s["gap"] for s in segments),
            "directions": tuple(s["direction"] for s in segments),
            "segments": segments}


def _sample_shape(rng: np.random.Generator) -> tuple[int, int]:
    choice = float(rng.random())
    if choice < 0.16:
        return (10, 10)
    if choice < 0.26:
        return (30, 30)
    if choice < 0.43:
        return (int(rng.integers(5, 10)), int(rng.integers(5, 11)))
    if choice < 0.65:
        shape = (int(rng.integers(5, 12)), int(rng.integers(18, 31)))
        return shape if rng.random() < 0.5 else shape[::-1]
    if choice < 0.84:
        return (int(rng.integers(10, 21)), int(rng.integers(10, 21)))
    return (int(rng.integers(19, 31)), int(rng.integers(19, 31)))


def _capacity(shape: tuple[int, int]) -> int:
    # Sparse enough that the background is readily identifiable even on tiny grids.
    return min(9, max(2, shape[0] * shape[1] // 9))


def _sample_parameters(rng: np.random.Generator) -> dict:
    shape_mode = "fixed" if rng.random() < 0.19 else "varied"
    color_draw = float(rng.random())
    color_mode = ("stable" if color_draw < 0.18 else
                  "fixed_background" if color_draw < 0.43 else "varied")
    background = int(rng.integers(0, 10))
    fixed_shape = _sample_shape(rng) if shape_mode == "fixed" else None
    if color_mode == "stable":
        palette = [c for c in range(10) if c != background]
        rng.shuffle(palette)
        maximum = _capacity(fixed_shape) if fixed_shape is not None else 7
        stable_count = int(rng.integers(2, maximum + 1))
        stable_colors = tuple(palette[:stable_count])
    else:
        stable_colors = ()
    # A reserved test color is a diversity device, not a learned color map.
    reserve = None
    if color_mode != "stable" and rng.random() < 0.57:
        reserve = int(rng.choice([c for c in range(10) if c != background]))
    return {"shape_mode": shape_mode, "fixed_shape": fixed_shape,
            "color_mode": color_mode, "background": background,
            "stable_colors": stable_colors, "reserve": reserve}


def _pair_shape(rng: np.random.Generator, params: dict) -> tuple[int, int]:
    if params["shape_mode"] == "fixed":
        return params["fixed_shape"]
    shape = _sample_shape(rng)
    if params["color_mode"] == "stable":
        for _ in range(16):
            if _capacity(shape) >= len(params["stable_colors"]):
                break
            shape = _sample_shape(rng)
        else:
            shape = (10, 10)
    return shape


def _sample_count(rng: np.random.Generator, shape: tuple[int, int],
                  params: dict, split: str, train_counts: list[int]) -> int:
    capacity = _capacity(shape)
    if params["color_mode"] == "stable":
        return min(capacity, len(params["stable_colors"]))
    if split == "train":
        return int(rng.integers(1, capacity + 1))
    style = int(rng.integers(0, 4))
    if style == 0 and max(train_counts) < capacity:
        return int(rng.integers(max(train_counts) + 1, capacity + 1))
    if style == 1:
        choices = [n for n in train_counts if n <= capacity]
        if choices:
            return int(rng.choice(choices))
    if style == 2:
        return int(rng.integers(1, min(capacity, max(2, min(train_counts))) + 1))
    return int(rng.integers(1, capacity + 1))


def _sample_background(rng: np.random.Generator, params: dict,
                       split: str, index: int) -> int:
    if params["color_mode"] != "varied":
        return params["background"]
    if split == "train" and index == 0:
        return params["background"]
    if split == "train" and index == 1:
        return int(rng.choice([c for c in range(10)
                               if c != params["background"]]))
    if split == "test" and index == 0 and params["reserve"] is not None:
        return int(rng.choice([c for c in range(10)
                               if c != params["reserve"]]))
    return int(rng.integers(0, 10))


def _sample_colors(rng: np.random.Generator, params: dict,
                   background: int, count: int, split: str,
                   index: int) -> tuple[int, ...]:
    if params["color_mode"] == "stable":
        return params["stable_colors"][:count]
    required = []
    if params["color_mode"] == "varied" and split == "train" and index == 1:
        required.append(params["background"])
    if split == "test" and index == 0 and params["reserve"] is not None:
        required.append(params["reserve"])
    excluded = {background}
    if split == "train" and params["reserve"] is not None:
        excluded.add(params["reserve"])
    candidates = [c for c in range(10) if c not in excluded and c not in required]
    rng.shuffle(candidates)
    assert all(c not in excluded for c in required)
    return tuple(required + candidates[:count - len(required)])


def _sample_directions(rng: np.random.Generator, count: int,
                       first_train: bool) -> list[int]:
    mode = int(rng.integers(0, 5))
    if mode == 0:
        directions = [-1] * count
    elif mode == 1:
        directions = [1] * count
    elif mode == 2:
        directions = [(-1 if i % 2 else 1) for i in range(count)]
    elif mode == 3:
        minority = -1 if rng.random() < 0.5 else 1
        directions = [-minority] * count
        directions[0] = minority
    else:
        directions = [int(rng.choice((-1, 1))) for _ in range(count)]
    if first_train:
        directions[0], directions[1] = -1, 1
    rng.shuffle(directions)
    return directions


def _sample_gaps(rng: np.random.Generator, shape: tuple[int, int],
                 count: int, first_train: bool) -> list[int]:
    maximum = min(shape) - 1
    mode = int(rng.integers(0, 5))
    if mode == 0:  # short and similar
        ceiling = max(2, min(maximum, 4))
        gaps = [int(rng.integers(2, ceiling + 1)) for _ in range(count)]
    elif mode == 1:  # long
        floor = max(2, int(np.ceil(maximum * 0.68)))
        gaps = [int(rng.integers(floor, maximum + 1)) for _ in range(count)]
    elif mode == 2:  # nearly equal
        center = int(rng.integers(2, maximum + 1))
        gaps = [max(2, min(maximum, center + int(rng.integers(-1, 2))))
                for _ in range(count)]
    else:
        gaps = [int(rng.integers(2, maximum + 1)) for _ in range(count)]
        if mode == 3 and count >= 2:  # a deliberately wide length spread
            gaps[0] = 2
            gaps[1] = maximum
    if first_train:
        gaps[0] = max(4, gaps[0])
    return gaps


def _candidate_cells(row: int, col: int, gap: int,
                     direction: int) -> tuple[tuple[int, int], ...]:
    upper_col = col if direction == 1 else col + gap
    return tuple((row + step, upper_col + direction * step)
                 for step in range(gap + 1))


def _place_one(rng: np.random.Generator, shape: tuple[int, int],
               desired_gap: int, direction: int, used: set,
               endpoints: list[tuple[int, int]], mode: str,
               minimum_gap: int = 2) -> tuple | None:
    height, width = shape
    for gap in range(desired_gap, minimum_gap - 1, -1):
        possible = []
        for row in range(height - gap):
            for col in range(width - gap):
                cells = _candidate_cells(row, col, gap, direction)
                if used.isdisjoint(cells):
                    possible.append(cells)
        if not possible:
            continue
        if mode == "random" or (mode in ("close", "spread") and not endpoints):
            return possible[int(rng.integers(0, len(possible)))]
        scored = []
        for cells in possible:
            if mode in ("close", "spread"):
                score = min(max(abs(p[0] - q[0]), abs(p[1] - q[1]))
                            for p in (cells[0], cells[-1]) for q in endpoints)
            else:
                score = min(min(p[0], p[1], height - 1 - p[0], width - 1 - p[1])
                            for p in (cells[0], cells[-1]))
            scored.append((score, cells))
        scored.sort(key=lambda x: x[0],
                    reverse=mode in ("spread", "interior"))
        pool = max(1, len(scored) // 7)
        return scored[int(rng.integers(0, pool))][1]
    return None


def _make_pair(rng: np.random.Generator, shape: tuple[int, int],
               background: int, colors: tuple[int, ...],
               first_train: bool) -> tuple[np.ndarray, np.ndarray]:
    count = len(colors)
    for attempt in range(28):
        directions = _sample_directions(rng, count, first_train)
        gaps = _sample_gaps(rng, shape, count, first_train)
        if attempt >= 18:
            gaps = [4 if first_train and i == 0 else 2 for i in range(count)]
        order = sorted(range(count), key=lambda i: gaps[i], reverse=True)
        modes = ("random", "close", "spread", "edge", "interior")
        placement_mode = modes[int(rng.integers(0, len(modes)))]
        used = set()
        endpoints = []
        objects = [None] * count
        success = True
        for idx in order:
            cells = _place_one(rng, shape, gaps[idx], directions[idx], used,
                               endpoints, placement_mode,
                               minimum_gap=4 if first_train and idx == 0 else 2)
            if cells is None:
                success = False
                break
            objects[idx] = cells
            used.update(cells)
            endpoints.extend((cells[0], cells[-1]))
        if not success:
            continue
        input_grid = np.full(shape, background, dtype=np.int8)
        for color, cells in zip(colors, objects):
            input_grid[cells[0]] = color
            input_grid[cells[-1]] = color
        output_grid = _construct_output(input_grid)
        return input_grid, output_grid
    raise RuntimeError("could not place nonintersecting diagonals")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _validate_seed(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    params = _sample_parameters(rng)
    pair_rngs = [np.random.default_rng(child) for child in
                 np.random.SeedSequence(int(rng.integers(0, 2**63))).spawn(
                     num_train + num_test)]
    puzzle = {"train": [], "test": []}
    seen = set()
    train_counts = []
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            pair_rng = pair_rngs.pop(0)
            for attempt in range(25):
                shape = _pair_shape(pair_rng, params)
                count = _sample_count(pair_rng, shape, params, split, train_counts)
                if split == "train" and index == 0:
                    count = max(2, count)
                background = _sample_background(pair_rng, params, split, index)
                colors = _sample_colors(pair_rng, params, background, count,
                                        split, index)
                if len(colors) != count:
                    continue
                try:
                    input_grid, output_grid = _make_pair(
                        pair_rng, shape, background, colors,
                        first_train=split == "train" and index == 0)
                except RuntimeError:
                    continue
                key = _grid_key(input_grid)
                if key in seen or _is_source_pair(input_grid, output_grid):
                    continue
                seen.add(key)
                puzzle[split].append({"input": input_grid, "output": output_grid})
                if split == "train":
                    train_counts.append(count)
                break
            else:
                raise RuntimeError("could not build a unique puzzle pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test
    seen_inputs, seen_pairs = set(), set()
    train_metadata = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert input_grid.shape == output_grid.shape
            input_key = _grid_key(input_grid)
            pair_key = (input_key, _grid_key(output_grid))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _is_source_pair(input_grid, output_grid)
            background, segments = _segments(input_grid)
            assert np.count_nonzero(input_grid == background) * 5 >= input_grid.size * 3
            expected = input_grid.copy()
            for segment in segments:
                color = segment["color"]
                assert input_grid[segment["upper"]] == color
                assert input_grid[segment["lower"]] == color
                for row, col in segment["cells"][1:-1]:
                    assert input_grid[row, col] == background
                    assert expected[row, col] == background
                    expected[row, col] = color
            assert np.array_equal(output_grid, expected), (
                "output must contain exactly the inclusive diagonal segments")
            assert np.count_nonzero(input_grid != output_grid) > 0
            if split == "train":
                train_metadata.append(_metadata(input_grid))
    # The first demonstration has two identities, both slopes, and a line
    # with several inserted cells. The rule is then color-generic at test time.
    first = train_metadata[0]
    assert first["count"] >= 2
    assert set(first["directions"]) == {-1, 1}
    assert max(first["gaps"]) >= 4


def self_test() -> None:
    automatic = generate_puzzle(12345)
    validate_generated_puzzle(automatic)
    a = generate_puzzle(54321, 7, 3)
    b = generate_puzzle(54321, 7, 3)
    assert _puzzle_key(a) == _puzzle_key(b)
    assert len({_puzzle_key(generate_puzzle(s, 5, 2)) for s in range(200, 206)}) >= 5
    for offset, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(7000 + offset, *counts)
        assert (len(p["train"]), len(p["test"])) == counts
    for offset, request in enumerate(((None, 1), (None, 4), (2, None),
                                      (10, None))):
        p = generate_puzzle(8000 + offset, *request)
        if request[0] is not None:
            assert len(p["train"]) == request[0]
        if request[1] is not None:
            assert len(p["test"]) == request[1]
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (2, 3), (4, 4)):
        try:
            generate_puzzle(9000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Regression on the original solved task: all four 10x10 joins are straight.
    for input_rows, output_rows in _SOURCE_PAIR_ROWS:
        inp = np.array([[int(c) for c in row] for row in input_rows], dtype=np.int8)
        out = np.array([[int(c) for c in row] for row in output_rows], dtype=np.int8)
        assert np.array_equal(_construct_output(inp), out)
    # Mutations that used to pass the lower-anchored interpretation must fail.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in automatic[s]]
           for s in ("train", "test")}
    color = _metadata(bad["train"][0]["input"])["colors"][0]
    positions = np.argwhere(bad["train"][0]["input"] == color)
    upper = tuple(map(int, positions[np.argmin(positions[:, 0])]))
    bg = _infer_background(bad["train"][0]["input"])
    bad["train"][0]["input"][upper] = bg
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing endpoint accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in automatic[s]]
           for s in ("train", "test")}
    inp = bad["train"][0]["input"]
    segment = _metadata(inp)["segments"][0]
    upper = segment["upper"]
    displacement = next((upper[0], upper[1] + delta)
                        for delta in (-1, 1)
                        if 0 <= upper[1] + delta < inp.shape[1]
                        and inp[upper[0], upper[1] + delta] == bg)
    inp[upper] = bg
    inp[displacement] = segment["color"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("displaced endpoint accepted")
    for mutation in ("missing", "extra"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in automatic[s]]
               for s in ("train", "test")}
        inp, out = bad["train"][0]["input"], bad["train"][0]["output"]
        if mutation == "missing":
            changed = np.argwhere(inp != out)
            row, col = map(int, changed[0])
            out[row, col] = inp[row, col]
        else:
            blank = np.argwhere(inp == out)
            row, col = map(int, blank[0])
            out[row, col] = (int(out[row, col]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("missing or extra line pixel accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts, test_counts, counts, shapes, gaps = set(), set(), set(), set(), set()
    backgrounds, layouts, profiles = set(), set(), set()
    varied_backgrounds = new_test_colors = stable_palettes = 0
    role_switches = new_test_sizes = fixed_sizes = 0
    denser_tests = simpler_tests = equal_count_tests = 0
    parallel = mixed = long_lines = close_pairs = interleaved_pairs = 0
    edge_pairs = interior_pairs = combined_pairs = 0
    source_size_counts = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_fg = set()
        train_bg = set()
        train_shapes = set()
        pair_meta = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                meta = _metadata(inp)
                pair_meta.append(meta)
                counts.add(meta["count"])
                shapes.add(meta["shape"])
                gaps.update(meta["gaps"])
                backgrounds.add(meta["background"])
                if meta["shape"] == (10, 10):
                    source_size_counts.add(meta["count"])
                if split == "train":
                    train_fg.update(meta["colors"])
                    train_bg.add(meta["background"])
                    train_shapes.add(meta["shape"])
                elif set(meta["colors"]) - train_fg:
                    new_test_colors += 1
                directions = set(meta["directions"])
                parallel += len(directions) == 1
                mixed += len(directions) == 2
                long_lines += max(meta["gaps"]) >= min(meta["shape"]) * 0.7
                segments = meta["segments"]
                close = any(max(abs(a[0]-b[0]), abs(a[1]-b[1])) <= 2
                            for i, s in enumerate(segments) for t in segments[i+1:]
                            for a in s["cells"] for b in t["cells"])
                interleaved = any(
                    s["upper"][0] < t["upper"][0] < s["lower"][0] < t["lower"][0]
                    or t["upper"][0] < s["upper"][0] < t["lower"][0] < s["lower"][0]
                    for i, s in enumerate(segments) for t in segments[i+1:])
                close_pairs += close
                interleaved_pairs += interleaved
                combined_pairs += close and interleaved and len(directions) == 2
                h, w = meta["shape"]
                margins = [min(p[0], p[1], h - 1 - p[0], w - 1 - p[1])
                           for s in segments for p in (s["upper"], s["lower"])]
                edge_pairs += min(margins) == 0
                interior_pairs += min(margins) >= 2
                mask = np.ascontiguousarray(inp != meta["background"], dtype=np.uint8)
                layouts.add((meta["shape"], mask.tobytes()))
                profiles.add((meta["shape"], meta["count"],
                              tuple(sorted(meta["gaps"])),
                              sum(d == 1 for d in meta["directions"])))
        varied_backgrounds += len({m["background"] for m in pair_meta}) > 1
        stable_palettes += len({(m["background"], tuple(m["colors"]))
                                for m in pair_meta}) == 1
        fixed_sizes += len({m["shape"] for m in pair_meta}) == 1
        role_switches += any(m["background"] in {
            c for n in pair_meta if n is not m for c in n["colors"]
        } for m in pair_meta)
        new_test_sizes += any(m["shape"] not in train_shapes
                              for m in pair_meta[len(puzzle["train"]):])
        train_counts_here = [m["count"] for m in pair_meta[:len(puzzle["train"])]]
        test_counts_here = [m["count"] for m in pair_meta[len(puzzle["train"]):]]
        denser_tests += any(n > max(train_counts_here) for n in test_counts_here)
        simpler_tests += any(n < min(train_counts_here) for n in test_counts_here)
        equal_count_tests += any(n in train_counts_here for n in test_counts_here)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert {1, 2, 8, 9}.issubset(counts)
    assert (10, 10) in shapes and (30, 30) in shapes
    assert min(min(s) for s in shapes) <= 7 and max(max(s) for s in shapes) == 30
    assert max(gaps) >= 20 and min(gaps) == 2
    assert len(backgrounds) >= 8 and varied_backgrounds >= 8
    assert new_test_colors >= 5 and stable_palettes >= 2
    assert parallel >= 20 and mixed >= 20 and long_lines >= 15
    assert close_pairs >= 20 and interleaved_pairs >= 20 and combined_pairs >= 10
    assert edge_pairs >= 20 and interior_pairs >= 5
    assert fixed_sizes >= 2 and role_switches >= 8 and new_test_sizes >= 8
    assert denser_tests >= 3 and simpler_tests >= 3 and equal_count_tests >= 8
    assert len(source_size_counts) >= 5
    assert len(shapes) >= 50 and len(layouts) >= 200 and len(profiles) >= 150
