"""Recover a square mask from a two-axis symmetric mosaic."""

import hashlib
import numpy as np


_SOURCE_DIGESTS = frozenset({
    "21483be627d28a7638f4db4ef82945788212cb297bd9520c6a060e98938de868",
    "685d69b39b2cb6caa557857bab7189422bd7a7219e0a42040c6622e0d7c9e9bd",
    "c72514eb30163abf67c4ff5ce6d59a2e50c2d8bab0a0db24eb115e9757d5d30d",
    "6d8a5cfd4437b710aaf1bb34fcd42a77cb8990e64e24c74fefa58df7267f0caa",
})
_POSITION_CACHE = {}


def _counts(rng, num_train, num_test):
    for name, value, lo, hi in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi
        ):
            raise ValueError(f"{name} must be an integer from {lo} through {hi}")
    if num_test is None:
        num_test = int(rng.integers(
            1, min(4, (num_train if num_train is not None else 10) - 1) + 1
        ))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _axis_type(height, width, size, row, col):
    other_row, other_col = height - row - size, width - col - size
    h = max(row, other_row) < min(row + size, other_row + size)
    v = max(col, other_col) < min(col + size, other_col + size)
    if h and v:
        return "blocked"
    if h:
        return "horizontal"
    if v:
        return "vertical"
    return "neither"


def _positions(height, width, size, axis):
    key = (height, width, size, axis)
    if key not in _POSITION_CACHE:
        _POSITION_CACHE[key] = tuple(
            (r, c)
            for r in range(height - size + 1)
            for c in range(width - size + 1)
            if _axis_type(height, width, size, r, c) == axis
        )
    return _POSITION_CACHE[key]


def _quadrant(height, width, size, row, col):
    return ("upper" if 2 * row + size < height else "lower") + "_" + (
        "left" if 2 * col + size < width else "right"
    )


def _choose_position(rng, height, width, size, axis, quadrant, preference,
                     used_locations):
    pool = list(_positions(height, width, size, axis))
    pool = [p for p in pool
            if (height, width, p[0], p[1]) not in used_locations]
    assert pool
    if quadrant is not None:
        subset = [
            p for p in pool
            if _quadrant(height, width, size, *p) == quadrant
        ]
        if subset:
            pool = subset
    if preference == "edge":
        subset = [
            p for p in pool if p[0] == 0 or p[1] == 0
            or p[0] + size == height or p[1] + size == width
        ]
        if subset:
            pool = subset
    elif preference == "center":
        distances = [
            min(abs(2 * p[0] - height), abs(2 * (p[0] + size) - height))
            + min(abs(2 * p[1] - width), abs(2 * (p[1] + size) - width))
            for p in pool
        ]
        smallest = min(distances)
        pool = [p for p, distance in zip(pool, distances)
                if distance <= smallest + 2]
    return pool[int(rng.integers(len(pool)))]


def _seed_pattern(rng, height, width, colors):
    rows, cols = (height + 1) // 2, (width + 1) // 2
    style = int(rng.integers(5))
    if style == 0:
        seed = rng.choice(colors, size=(rows, cols))
    elif style == 1:
        coarse = rng.choice(colors, size=((rows + 1) // 2, (cols + 1) // 2))
        seed = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)[:rows, :cols]
    elif style == 2:
        by_row = rng.choice(colors, size=rows)
        by_col = rng.choice(colors, size=cols)
        seed = np.where(rng.random((rows, cols)) < .5,
                        by_row[:, None], by_col[None, :])
    elif style == 3:
        rr, cc = np.indices((rows, cols))
        seed = colors[(int(rng.integers(1, 5)) * rr
                       + int(rng.integers(1, 5)) * cc
                       + int(rng.integers(len(colors)))) % len(colors)]
    else:
        rr, cc = np.indices((rows, cols))
        seed = colors[(rr // int(rng.integers(2, 5))
                       + cc // int(rng.integers(2, 5))) % len(colors)]
    seed = np.asarray(seed, dtype=np.int64).copy()
    for _ in range(max(3, rows * cols // 6)):
        seed[int(rng.integers(rows)), int(rng.integers(cols))] = int(
            rng.choice(colors)
        )
    for color in colors:
        seed[int(rng.integers(rows)), int(rng.integers(cols))] = int(color)
    return seed


def _mirror(seed, height, width):
    ys = np.minimum(np.arange(height), height - 1 - np.arange(height))
    xs = np.minimum(np.arange(width), width - 1 - np.arange(width))
    return seed[np.ix_(ys, xs)].copy()


def _marker_candidates(grid):
    candidates = []
    for color, count in zip(*np.unique(grid, return_counts=True)):
        size = int(np.sqrt(int(count)))
        if not 2 <= size <= 6 or size * size != count:
            continue
        cells = np.argwhere(grid == color)
        top, left = cells.min(axis=0)
        bottom, right = cells.max(axis=0)
        if bottom - top + 1 == size and right - left + 1 == size:
            candidates.append((int(top), int(left), size, int(color)))
    return candidates


def _recover(grid, row, col, size, marker):
    height, width = grid.shape
    hidden = np.zeros((height, width), dtype=bool)
    hidden[row:row + size, col:col + size] = True
    out = np.empty((size, size), dtype=np.int64)
    for dy in range(size):
        for dx in range(size):
            y, x = row + dy, col + dx
            orbit = {(y, x), (height - 1 - y, x),
                     (y, width - 1 - x),
                     (height - 1 - y, width - 1 - x)}
            visible = [int(grid[a, b]) for a, b in orbit if not hidden[a, b]]
            assert visible and len(set(visible)) == 1
            assert visible[0] != marker
            out[dy, dx] = visible[0]
    restored = grid.copy()
    restored[row:row + size, col:col + size] = out
    assert np.array_equal(restored, restored[::-1, :])
    assert np.array_equal(restored, restored[:, ::-1])
    return out


def _digest(inp, out):
    if inp.shape != (16, 16) or out.shape != (4, 4):
        return None
    return hashlib.sha256(
        inp.astype(np.uint8).tobytes() + out.astype(np.uint8).tobytes()
    ).hexdigest()


def _make_pair(rng, height, width, size, axis, quadrant, preference,
               marker, previous_colors, used_locations):
    for _ in range(60):
        row, col = _choose_position(
            rng, height, width, size, axis, quadrant, preference,
            used_locations
        )
        available = np.array([c for c in range(10) if c != marker])
        if (previous_colors is not None and marker not in previous_colors
                and rng.random() < .32):
            colors = np.asarray(previous_colors, dtype=np.int64)
        else:
            count = int(rng.integers(2, 5) if rng.random() < .35
                        else rng.integers(5, 9))
            colors = rng.choice(available, size=count, replace=False)
        seed = _seed_pattern(rng, height, width, colors)
        detail = float(rng.random())
        target = (np.array([int(rng.choice(colors))]) if detail < .15 else
                  np.asarray(rng.choice(colors, size=2, replace=False))
                  if detail < .31 else None)
        if target is not None:
            orbit_keys = sorted({
                (min(y, height - 1 - y), min(x, width - 1 - x))
                for y in range(row, row + size)
                for x in range(col, col + size)
            })
            for i, (y, x) in enumerate(orbit_keys):
                seed[y, x] = target[i % len(target)]
        full = _mirror(seed, height, width)
        out = full[row:row + size, col:col + size].copy()
        inp = full.copy()
        inp[row:row + size, col:col + size] = marker
        if _marker_candidates(inp) != [(row, col, size, marker)]:
            continue
        if _digest(inp, out) in _SOURCE_DIGESTS:
            continue
        assert np.array_equal(_recover(inp, row, col, size, marker), out)
        return {"input": inp, "output": out}, tuple(map(int, colors))
    # The two-color reflected checkerboard has many cells of each color,
    # so neither visible color can masquerade as a 2..6 square marker.
    row, col = _choose_position(
        rng, height, width, size, axis, quadrant, preference, used_locations
    )
    fallback_colors = tuple(c for c in range(10) if c != marker)[:2]
    rr, cc = np.indices(((height + 1) // 2, (width + 1) // 2))
    seed = np.where((rr + cc) % 2 == 0,
                    fallback_colors[0], fallback_colors[1])
    full = _mirror(seed, height, width)
    out = full[row:row + size, col:col + size].copy()
    inp = full.copy()
    inp[row:row + size, col:col + size] = marker
    assert _marker_candidates(inp) == [(row, col, size, marker)]
    assert _digest(inp, out) not in _SOURCE_DIGESTS
    return {"input": inp, "output": out}, fallback_colors


def _dimensions(rng, classic, previous_size=None):
    if classic:
        return 16, 16, 4
    size = int(rng.integers(2, 7))
    if size == previous_size:
        size = 2 + (size - 1) % 5
    minimum = max(10, 2 * size + 2)
    height = int(rng.integers(minimum, 31))
    width = height if rng.random() < .26 else int(rng.integers(minimum, 31))
    return height, width, size


def _axis_plan(rng, train_count, test_count, classic):
    mode = int(rng.choice(4, p=[.25, .26, .26, .23]))
    if classic:
        mode = 2 if rng.random() < .65 else 0
    if mode == 0:
        return ["neither"] * train_count, ["neither"] * test_count
    if mode == 1:
        train = ["horizontal", "neither"] + [
            "horizontal" if rng.random() < .45 else "neither"
            for _ in range(train_count - 2)
        ]
        test = ["horizontal"] + [
            "horizontal" if rng.random() < .55 else "neither"
            for _ in range(test_count - 1)
        ]
    elif mode == 2:
        train = ["vertical", "neither"] + [
            "vertical" if rng.random() < .45 else "neither"
            for _ in range(train_count - 2)
        ]
        test = ["neither" if classic else "vertical"] + [
            "vertical" if rng.random() < .55 else "neither"
            for _ in range(test_count - 1)
        ]
    else:
        choices = ("horizontal", "vertical", "neither")
        train = ["horizontal", "vertical"] + [
            choices[int(rng.integers(3))] for _ in range(train_count - 2)
        ]
        test = [choices[int(rng.integers(3))] for _ in range(test_count)]
    return train, test


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    classic = bool(rng.random() < .2)
    variable_marker = bool(rng.random() < .34)
    fixed_marker = int(rng.integers(10))
    train_axes, test_axes = _axis_plan(rng, num_train, num_test, classic)
    quadrants = ["upper_left", "upper_right", "lower_left", "lower_right"]
    rng.shuffle(quadrants)
    puzzle = {"train": [], "test": []}
    previous_colors, previous_size = None, None
    used_locations = set()
    train_markers = []
    for split, axes in (("train", train_axes), ("test", test_axes)):
        for index, axis in enumerate(axes):
            pair_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
            pair_rng = np.random.default_rng(pair_seed)
            height, width, size = _dimensions(
                pair_rng, classic,
                previous_size if split == "train" and index == 1 else None
            )
            if split == "train" and index == 0:
                previous_size = size
            if variable_marker:
                choices = [
                    c for c in range(10)
                    if split != "train" or index != 1 or c != train_markers[0]
                ]
                marker = int(pair_rng.choice(choices))
            else:
                marker = fixed_marker
            if split == "train":
                train_markers.append(marker)
            quadrant = (
                quadrants[index % 4] if axis == "neither" and split == "train"
                else "lower_left" if classic and split == "test" else None
            )
            preference = (
                "center" if classic and split == "test" else
                ("edge", "center", None)[int(pair_rng.integers(3))]
                if split == "test" else None
            )
            pair, previous_colors = _make_pair(
                pair_rng, height, width, size, axis, quadrant,
                preference, marker, previous_colors, used_locations
            )
            puzzle[split].append(pair)
            row, col, _, _ = _marker_candidates(pair["input"])[0]
            used_locations.add((height, width, row, col))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    axes = {"train": set(), "test": set()}
    markers = {"train": set(), "test": set()}
    train_locations = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert 10 <= inp.shape[0] <= 30 and 10 <= inp.shape[1] <= 30
            assert inp.shape != out.shape
            candidates = _marker_candidates(inp)
            assert len(candidates) == 1
            row, col, size, marker = candidates[0]
            assert out.shape == (size, size) and marker not in out
            assert np.array_equal(_recover(inp, row, col, size, marker), out)
            axis = _axis_type(*inp.shape, size, row, col)
            assert axis != "blocked"
            axes[split].add(axis)
            markers[split].add(marker)
            assert _digest(inp, out) not in _SOURCE_DIGESTS
            key = (inp.shape, inp.astype(np.uint8).tobytes())
            both = (key, out.shape, out.astype(np.uint8).tobytes())
            assert key not in inputs and both not in pairs
            inputs.add(key)
            pairs.add(both)
            if split == "train":
                train_locations.add((inp.shape, row, col))
            else:
                assert (inp.shape, row, col) not in train_locations
    assert len(train_locations) >= 2
    assert axes["test"] - {"neither"} <= axes["train"]
    if len(markers["train"]) == 1:
        assert markers["test"] <= markers["train"]


def _same_puzzle(left, right):
    return all(
        len(left[split]) == len(right[split])
        and all(np.array_equal(a[name], b[name])
                for a, b in zip(left[split], right[split])
                for name in ("input", "output"))
        for split in ("train", "test")
    )


def _predict_from_demonstrations(train, test_input):
    train_axes, train_markers = set(), set()
    for pair in train:
        (row, col, size, marker), = _marker_candidates(pair["input"])
        assert np.array_equal(
            _recover(pair["input"], row, col, size, marker), pair["output"]
        )
        train_axes.add(_axis_type(*pair["input"].shape, size, row, col))
        train_markers.add(marker)
    (row, col, size, marker), = _marker_candidates(test_input)
    axis = _axis_type(*test_input.shape, size, row, col)
    assert axis == "neither" or axis in train_axes
    assert len(train_markers) >= 2 or marker in train_markers
    return _recover(test_input, row, col, size, marker)


def self_test() -> None:
    train_counts, test_counts = set(), set()
    sizes, shapes, markers, locations, palettes = set(), set(), set(), set(), set()
    axes, output_counts, first_inputs = set(), set(), set()
    within_scale = variable_markers = repeated_palettes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        first_inputs.add(puzzle["train"][0]["input"].tobytes())
        local_shapes, local_markers, local_palettes = set(), set(), []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                (row, col, size, marker), = _marker_candidates(inp)
                assert np.array_equal(_recover(inp, row, col, size, marker), out)
                if split == "test":
                    assert np.array_equal(
                        _predict_from_demonstrations(puzzle["train"], inp), out
                    )
                sizes.add(size)
                shapes.add(inp.shape)
                markers.add(marker)
                locations.add((inp.shape, row, col))
                axes.add(_axis_type(*inp.shape, size, row, col))
                output_counts.add(len(np.unique(out)))
                colors = frozenset(int(c) for c in np.unique(inp) if c != marker)
                palettes.add(colors)
                local_palettes.append(colors)
                local_shapes.add((inp.shape, size))
                local_markers.add(marker)
        within_scale += len(local_shapes) > 1
        variable_markers += len(local_markers) > 1
        repeated_palettes += len(set(local_palettes)) < len(local_palettes)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(first_inputs) >= 40
    assert sizes == {2, 3, 4, 5, 6} and len(shapes) >= 35
    assert markers == set(range(10))
    assert len(locations) >= 100 and len(palettes) >= 40
    assert axes == {"neither", "horizontal", "vertical"}
    assert {1, 2} <= output_counts
    assert within_scale >= 20 and variable_markers >= 8
    assert repeated_palettes >= 2
    for tr, te in ((2, 1), (10, 1), (10, 4)):
        seed = 700 + tr * 10 + te
        puzzle = generate_puzzle(seed, tr, te)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (tr, te)
        assert _same_puzzle(puzzle, generate_puzzle(seed, tr, te))
        validate_generated_puzzle(puzzle)
    for tr, te in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(920 + (tr or 0) + (te or 0), tr, te)
        assert tr is None or len(puzzle["train"]) == tr
        assert te is None or len(puzzle["test"]) == te
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted illegal counts: {kwargs}")
