import math

import numpy as np


_SOURCE_PAIRS = (
    (
        "0000020040 0555020040 0555020040 0555020040 0000020040 "
        "0010020040 0010020040 0010020040 0010020040 0010020040 "
        "0010020040 0010020040",
        "124 124 124",
    ),
    (
        "000000000000000 000000000000000 222222222222222 "
        "000000000000000 000000000000000 111111111111111 "
        "000000000000000 000000000000000 333333333333333 "
        "000000000000000 055555500000000 055555504444444 "
        "055555500000000 055555500000000 055555508888888 "
        "055555500000000 000000000000000 666666666666666",
        "222222 111111 333333 444444 888888 666666",
    ),
    (
        "000000000000 222220555502 000000555500 000000555500 "
        "888880555508 000000000000 000000000000 444444444444 "
        "000000000000 000000000000 111111111111 000000000000",
        "2222 8888 4444 1111",
    ),
    (
        "2222222000000000222 0000000055555550000 0000000055555550000 "
        "3333333055555550333 0000000055555550000 0000000055555550000 "
        "8888888055555550888 0000000055555550000 0000000000000000000 "
        "4444444444444444444 0000000000000000000 0000000000000000000 "
        "6666666666666666666 0000000000000000000 0000000000000000000 "
        "1111111111111111111 0000000000000000000 0000000000000000000 "
        "7777777777777777777",
        "2222222 3333333 8888888 4444444 6666666 1111111 7777777",
    ),
)


def _signature(a, b):
    return (a.shape, a.tobytes(), b.shape, b.tobytes())


_SOURCE_SIGNATURES = frozenset(
    _signature(
        np.array([[int(c) for c in row] for row in inp.split()], dtype=np.int8),
        np.array([[int(c) for c in row] for row in out.split()], dtype=np.int8),
    )
    for inp, out in _SOURCE_PAIRS
)


def _normalize_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    return int(seed) % (1 << 64)


def _validate_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _validate_count("num_train", num_train, 2, 10)
    num_test = _validate_count("num_test", num_test, 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
    elif num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _geometry_options(n):
    """All legal transverse layouts, indexed by number of masked lines."""
    base = 3 * (n - 1) + 1
    options = {}
    for extent in range(max(base, n + 2), 31):
        for first in range(extent - base + 1):
            positions = tuple(first + 3 * i for i in range(n))
            for square_at in range(1, extent - n):
                overlap = sum(square_at - 1 <= p <= square_at + n for p in positions)
                options.setdefault(overlap, []).append((extent, first, square_at))
    return options


_GEOMETRY = {n: _geometry_options(n) for n in range(3, 9)}


def _choose_layout(rng, n, regime, require_overlap=False):
    options = _GEOMETRY[n]
    counts = [k for k in sorted(options) if not require_overlap or k > 0]
    # Both splits span the overlap range. Tests favor its tails and larger canvases.
    weights = np.array([2.0 if k in (0, 1, max(counts)) else 1.0 for k in counts])
    if regime == "source_train":
        weights = np.ones(len(counts), dtype=float)
    weights /= weights.sum()
    overlap = int(rng.choice(counts, p=weights))
    choices = options[overlap]
    extent, first, square_at = choices[int(rng.integers(len(choices)))]
    return extent, tuple(first + 3 * i for i in range(n)), square_at


def _choose_cross_axis(rng, n, require_both=False):
    high = min(30, n + int(rng.integers(6, 21)))
    extent = int(rng.integers(n + (4 if require_both else 3), high + 1))
    far = extent - n - 1
    if require_both:
        return extent, int(rng.integers(2, far))
    sidedness = int(rng.integers(3))
    if sidedness == 0 or far == 1:
        square_at = 1
    elif sidedness == 1 or far == 2:
        square_at = far
    else:
        square_at = int(rng.integers(2, far))
    return extent, square_at


def _sample_colors(rng, background, target, n, repeated):
    available = [c for c in range(10) if c not in (background, target)]
    if repeated:
        hue_count = int(rng.integers(2, min(n - 1, len(available)) + 1))
        hues = rng.choice(available, size=hue_count, replace=False).tolist()
        colors = hues + [int(rng.choice(hues)) for _ in range(n - hue_count)]
        rng.shuffle(colors)
    else:
        colors = rng.choice(available, size=n, replace=False).tolist()
    return tuple(int(c) for c in colors)


def _make_pair(rng, n, orientation, regime, roles, repeated, demonstration=False):
    background, target = roles
    colors = _sample_colors(rng, background, target, n, repeated)
    transverse_extent, positions, square_axis = _choose_layout(rng, n, regime, demonstration)
    parallel_extent, square_cross = _choose_cross_axis(rng, n, demonstration)
    if orientation == "H":
        height, width = transverse_extent, parallel_extent
        row, col = square_axis, square_cross
    else:
        height, width = parallel_extent, transverse_extent
        row, col = square_cross, square_axis
    grid = np.full((height, width), background, dtype=np.int8)
    for position, color in zip(positions, colors):
        if orientation == "H":
            grid[position, :] = color
        else:
            grid[:, position] = color
    grid[row - 1 : row + n + 1, col - 1 : col + n + 1] = background
    grid[row : row + n, col : col + n] = target
    palette = np.asarray(colors, dtype=np.int8)
    output = (np.repeat(palette[:, None], n, axis=1) if orientation == "H"
              else np.repeat(palette[None, :], n, axis=0))
    return {"input": grid, "output": output}


def _read_input(grid):
    """Recover the complete rule from visible cells, without generation metadata."""
    height, width = grid.shape
    candidates = []
    for target in np.unique(grid):
        cells = np.argwhere(grid == target)
        n = math.isqrt(len(cells))
        if n < 3 or n > 8 or n * n != len(cells):
            continue
        row, col = cells.min(axis=0)
        if (cells.max(axis=0) != (row + n - 1, col + n - 1)).any():
            continue
        if row < 1 or col < 1 or row + n >= height or col + n >= width:
            continue
        if not np.all(grid[row : row + n, col : col + n] == target):
            continue
        patch = grid[row - 1 : row + n + 1, col - 1 : col + n + 1]
        border = np.concatenate((patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]))
        if len(np.unique(border)) != 1:
            continue
        background = int(border[0])
        if background == target:
            continue
        remainder = (grid != background) & (grid != target)
        for orientation in ("H", "V"):
            axis = 0 if orientation == "H" else 1
            coords = np.argwhere(remainder)
            positions = tuple(int(p) for p in np.unique(coords[:, axis]))
            if len(positions) != n or any(b - a != 3 for a, b in zip(positions, positions[1:])):
                continue
            colors = []
            for p in positions:
                stripe = grid[p, :] if orientation == "H" else grid[:, p]
                values = np.unique(stripe[(stripe != background) & (stripe != target)])
                if len(values) != 1:
                    break
                colors.append(int(values[0]))
            if len(colors) != n or len(set(colors)) < 2:
                continue
            rebuilt = np.full(grid.shape, background, dtype=np.int8)
            for p, color in zip(positions, colors):
                if orientation == "H":
                    rebuilt[p, :] = color
                else:
                    rebuilt[:, p] = color
            rebuilt[row - 1 : row + n + 1, col - 1 : col + n + 1] = background
            rebuilt[row : row + n, col : col + n] = target
            if not np.array_equal(rebuilt, grid):
                continue
            square_axis = int(row if orientation == "H" else col)
            cross_axis = int(col if orientation == "H" else row)
            parallel_extent = width if orientation == "H" else height
            overlap = sum(square_axis - 1 <= p <= square_axis + n for p in positions)
            candidates.append({
                "n": n, "orientation": orientation, "colors": tuple(colors),
                "background": background, "target": int(target), "positions": positions,
                "square_row": int(row), "square_col": int(col),
                "masked_count": overlap,
                "fragment_sides": (int(cross_axis > 1) + int(cross_axis + n < parallel_extent - 1)) if overlap else 0,
                "shape": (height, width),
            })
    assert len(candidates) == 1, f"expected one interpretation, got {len(candidates)}"
    return candidates[0]


def _expected_output(info):
    n = info["n"]
    colors = np.asarray(info["colors"], dtype=np.int8)
    return (np.repeat(colors[:, None], n, axis=1) if info["orientation"] == "H"
            else np.repeat(colors[None, :], n, axis=0))


def _roles_for_pair(rng, base_roles, mode, index):
    background, target = base_roles
    if index == 0 or mode == "stable":
        return background, target
    if mode in ("background", "both"):
        background = int(rng.choice([c for c in range(10) if c not in (target, base_roles[0])]))
    if mode in ("target", "both"):
        target = int(rng.choice([c for c in range(10) if c not in (background, base_roles[1])]))
    return background, target


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(_normalize_seed(seed))
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    base_roles = tuple(int(c) for c in rng.choice(10, size=2, replace=False))
    role_mode = str(rng.choice(("stable", "background", "target", "both")))
    orientation_mode = str(rng.choice(("H", "V", "mixed")))
    repeat_mode = bool(rng.integers(2))

    if orientation_mode == "mixed":
        train_orientations = ["H", "V"] + [str(rng.choice(("H", "V"))) for _ in range(num_train - 2)]
        rng.shuffle(train_orientations)
        test_orientations = [str(rng.choice(("H", "V"))) for _ in range(num_test)]
    else:
        train_orientations = [orientation_mode] * num_train
        test_orientations = [orientation_mode] * num_test

    # A puzzle may teach a compact, middle, or full size range. This leaves
    # room for both extrapolating and already demonstrated test sizes.
    train_cap = int(rng.choice((5, 6, 7, 8)))
    train_sizes = [train_cap, int(rng.integers(3, train_cap))]
    train_sizes += [int(rng.integers(3, train_cap + 1)) for _ in range(num_train - 2)]
    rng.shuffle(train_sizes)
    test_sizes = []
    for _ in range(num_test):
        category = int(rng.integers(3))
        if category == 0 and max(train_sizes) < 8:
            choices = list(range(max(train_sizes) + 1, 9))
        elif category == 1:
            choices = sorted(set(train_sizes))
        else:
            choices = list(range(3, max(train_sizes)))
        test_sizes.append(int(rng.choice(choices)))

    train_repeats = [bool(rng.integers(2)) for _ in range(num_train)]
    test_repeats = [bool(rng.integers(2)) for _ in range(num_test)] if repeat_mode else [False] * num_test
    if any(test_repeats):
        train_repeats[0] = True
    if repeat_mode:
        train_repeats[0] = True

    puzzle = {"train": [], "test": []}
    seen = set()
    demonstration_index = int(rng.integers(num_train))
    for split, sizes, orientations, repeats in (
        ("train", train_sizes, train_orientations, train_repeats),
        ("test", test_sizes, test_orientations, test_repeats),
    ):
        regime = "source_train" if split == "train" else "target_test"
        for i, (n, orientation, repeated) in enumerate(zip(sizes, orientations, repeats)):
            role_index = i if split == "train" else i + num_train
            roles = _roles_for_pair(rng, base_roles, role_mode, role_index)
            for _ in range(200):
                pair = _make_pair(rng, n, orientation, regime, roles, repeated,
                                  split == "train" and i == demonstration_index)
                sig = _signature(pair["input"], pair["output"])
                key = (pair["input"].shape, pair["input"].tobytes())
                if key in seen or sig in _SOURCE_SIGNATURES:
                    continue
                try:
                    info = _read_input(pair["input"])
                except AssertionError:
                    continue
                if not np.array_equal(pair["output"], _expected_output(info)):
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise ValueError("could not construct a unique recoverable pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            key = (pair["input"].shape, pair["input"].tobytes())
            assert key not in seen
            seen.add(key)
            assert _signature(pair["input"], pair["output"]) not in _SOURCE_SIGNATURES
            assert not np.array_equal(pair["input"], pair["output"])
            metadata = _read_input(pair["input"])
            assert np.array_equal(pair["output"], _expected_output(metadata))
            assert pair["output"].shape == (metadata["n"], metadata["n"])
            info[split].append(metadata)
    train, test = info["train"], info["test"]
    assert len({item["n"] for item in train}) >= 2
    assert any(item["masked_count"] > 0 and item["fragment_sides"] == 2 for item in train)
    train_orientations = {item["orientation"] for item in train}
    assert {item["orientation"] for item in test} <= train_orientations
    if any(len(set(item["colors"])) < item["n"] for item in test):
        assert any(len(set(item["colors"])) < item["n"] for item in train)
    for role in ("background", "target"):
        trained = {item[role] for item in train}
        if any(item[role] not in trained for item in test):
            assert len(trained) >= 2


def _puzzle_equal(a, b):
    return all(
        len(a[split]) == len(b[split]) and all(
            np.array_equal(x[name], y[name])
            for x, y in zip(a[split], b[split]) for name in ("input", "output")
        ) for split in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts = set(), set()
    orientations, sizes, overlaps, fragments, shapes, roles = set(), set(), set(), set(), set(), set()
    layouts = set()
    puzzle_modes = set()
    within = {"overlap": 0, "shape": 0, "background": 0, "target": 0}
    test_shifts = set()
    repeated_train = repeated_test = heavy_train_6 = light_test = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local = {split: [_read_input(p["input"]) for p in puzzle[split]] for split in ("train", "test")}
        all_info = local["train"] + local["test"]
        puzzle_modes.add(tuple(sorted({m["orientation"] for m in local["train"] + local["test"]})))
        for name, field in (("overlap", "masked_count"), ("shape", "shape"),
                            ("background", "background"), ("target", "target")):
            within[name] += len({m[field] for m in all_info}) > 1
        largest_train = max(m["n"] for m in local["train"])
        for split in ("train", "test"):
            for m in local[split]:
                orientations.add((split, m["orientation"]))
                sizes.add((split, m["n"]))
                overlaps.add((split, m["masked_count"]))
                fragments.add((split, m["fragment_sides"]))
                shapes.add(m["shape"])
                roles.add((m["background"], m["target"]))
                layouts.add((m["orientation"], m["n"], m["shape"], m["positions"], m["square_row"], m["square_col"]))
                repeated = len(set(m["colors"])) < m["n"]
                if split == "train":
                    repeated_train += repeated
                    heavy_train_6 += m["n"] == 6 and m["masked_count"] >= 3
                else:
                    repeated_test += repeated
                    light_test += m["masked_count"] <= 1
                    test_shifts.add("larger" if m["n"] > largest_train else
                                    "equal" if m["n"] == largest_train else "smaller")
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert puzzle_modes == {("H",), ("V",), ("H", "V")}
    assert orientations == {("train", "H"), ("train", "V"), ("test", "H"), ("test", "V")}
    assert {n for split, n in sizes if split == "train"} == set(range(3, 9))
    assert {n for split, n in sizes if split == "test"} == set(range(3, 9))
    assert heavy_train_6 > 0 and light_test > 0
    assert repeated_train > 0 and repeated_test > 0
    assert {k for split, k in overlaps if split == "train"} == set(range(5))
    assert {k for split, k in overlaps if split == "test"} == set(range(5))
    assert {k for split, k in fragments if split == "train"} == {0, 1, 2}
    assert {k for split, k in fragments if split == "test"} == {0, 1, 2}
    assert test_shifts == {"larger", "equal", "smaller"}
    assert within["overlap"] >= 35 and within["shape"] >= 35
    assert within["background"] >= 8 and within["target"] >= 8
    assert len(fragments) >= 4 and len(shapes) >= 30 and len(roles) >= 15 and len(layouts) >= 100
    assert _puzzle_equal(generate_puzzle(812345, 7, 3), generate_puzzle(812345, 7, 3))
    assert not _puzzle_equal(generate_puzzle(812345, 7, 3), generate_puzzle(812346, 7, 3))
    for i, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4), (6, 2), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(7100 + i, nt, ne)
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ne is not None:
            assert len(puzzle["test"]) == ne
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(1, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
