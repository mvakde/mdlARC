"""One-period motion of a square along a dashed straight guide."""

import numpy as np


_SOURCE_PAIRS = {
    (
        ("0000300", "0000000", "0000300", "0002220", "0002320",
         "0002220", "0000300", "0000000", "0000300", "0000000",
         "0000300", "0000000", "0000300"),
        ("0000300", "0000000", "0000300", "0000000", "0000300",
         "0002220", "0002320", "0002220", "0000300", "0000000",
         "0000300", "0000000", "0000300"),
    ),
    (
        ("0000000000000", "0000000000000", "2220000000000",
         "2323030303030", "2220000000000", "0000000000000",
         "0000000000000"),
        ("0000000000000", "0000000000000", "0022200000000",
         "0323230303030", "0022200000000", "0000000000000",
         "0000000000000"),
    ),
    (
        ("0030000", "0222000", "0232000", "0222000", "0030000",
         "0000000", "0030000"),
        ("0030000", "0000000", "0030000", "0222000", "0232000",
         "0222000", "0030000"),
    ),
    (
        ("00000000000000000", "00000000000000000",
         "00000000000000000", "00000002220000000",
         "30303032323030303", "00000002220000000",
         "00000000000000000"),
        ("00000000000000000", "00000000000000000",
         "00000000000000000", "00000000022200000",
         "30303030323230303", "00000000022200000",
         "00000000000000000"),
    ),
}


def _rows(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
    elif num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _place_square(grid, row, col, marker, payload):
    grid[row - 1:row + 2, col - 1:col + 2] = payload
    grid[row, col] = marker


def _render(height, width, orientation, line, start, count, center,
            background, marker, payload):
    latent = np.full((height, width), background, dtype=np.int8)
    positions = start + 2 * np.arange(count)
    if orientation == "horizontal":
        latent[line, positions] = marker
        row, col = line, center
        next_row, next_col = line, center + 2
    else:
        latent[positions, line] = marker
        row, col = center, line
        next_row, next_col = center + 2, line
    before = latent.copy()
    after = latent.copy()
    _place_square(before, row, col, marker, payload)
    _place_square(after, next_row, next_col, marker, payload)
    return before, after


def _axis_length(rng, split):
    # All size bands occur in both splits; difficult test cases are somewhat
    # more likely to use a longer guide.
    band = int(rng.choice(3, p=(.35, .40, .25) if split == "train"
                          else (.20, .30, .50)))
    low, high = ((8, 12), (13, 19), (20, 30))[band]
    return int(rng.integers(low, high + 1))


def _geometry(rng, split, orientation):
    axis = _axis_length(rng, split)
    transverse = int(rng.integers(5, 25))
    height, width = ((transverse, axis) if orientation == "horizontal"
                     else (axis, transverse))
    line = int(rng.integers(1, transverse - 1))

    # Full-span and bounded segments use the same phase and motion rule.
    # The edge case chooses a guide ending at axis-2, so the translated
    # square ends exactly at the canvas boundary.
    kind = int(rng.choice(4, p=(.30, .36, .19, .15)))
    if axis == 8 and kind == 3:
        kind = 0
    if kind == 0:
        start = int(rng.integers(0, 2))
        count = (axis - 1 - start) // 2 + 1
    elif kind == 3:
        end = axis - 2
        count = int(rng.integers(4, (end - 1) // 2 + 2))
        start = end - 2 * (count - 1)
    else:
        maximum = (axis + 1) // 2
        count = int(rng.integers(4, maximum + 1))
        start = int(rng.integers(0, axis - 2 * (count - 1)))

    position_kind = int(rng.choice(3, p=(.30, .35, .35) if split == "train"
                                   else (.20, .55, .25)))
    if kind != 3 and position_kind == 0 and start == 0:
        # An endpoint square needs one in-bounds cell before its center.
        # Shifting the segment phase makes the first marker available.
        start = 1
        if start + 2 * (count - 1) >= axis:
            count -= 1

    positions = list(range(start, start + 2 * count, 2))
    eligible = [i for i, pos in enumerate(positions[:-1])
                if pos >= 1 and pos + 3 < axis]
    if not eligible:
        # This can occur only in the shortest edge-segment configuration.
        start, count = 1, 4
        positions = [1, 3, 5, 7]
        eligible = [0, 1]

    if kind == 3 and positions[-1] == axis - 2:
        center_index = eligible[-1]
    else:
        if position_kind == 0:
            center_index = eligible[0]
        elif position_kind == 2:
            center_index = eligible[-1]
        else:
            interior = [i for i in eligible if i > 0 and i < count - 2]
            candidates = interior if interior else eligible
            midpoint = (count - 1) // 2
            if count % 2 and midpoint in candidates and rng.random() < .45:
                center_index = midpoint
            else:
                center_index = int(rng.choice(candidates))
    return height, width, line, start, count, positions[center_index]


def _sample_pair(rng, split, orientation, palette):
    height, width, line, start, count, center = _geometry(rng, split, orientation)
    return _render(height, width, orientation, line, start, count, center, *palette)


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= n <= 30 for n in grid.shape)
    assert bool(np.all((0 <= grid) & (grid <= 9)))


def _decode_input(grid):
    """Recover geometry and roles from visible cells, without generation state."""
    height, width = grid.shape
    colors = [int(x) for x in np.unique(grid)]
    assert len(colors) == 3
    candidates = []
    for payload in colors:
        ys, xs = np.where(grid == payload)
        if len(ys) != 8 or len(xs) != 8:
            continue
        top, left = int(ys.min()), int(xs.min())
        if int(ys.max()) != top + 2 or int(xs.max()) != left + 2:
            continue
        center_row, center_col = top + 1, left + 1
        ring = {(r, c) for r in range(top, top + 3)
                for c in range(left, left + 3) if (r, c) != (center_row, center_col)}
        if set(zip(ys.tolist(), xs.tolist())) != ring:
            continue
        marker = int(grid[center_row, center_col])
        if marker == payload:
            continue
        background = next(x for x in colors if x not in (payload, marker))
        marker_y, marker_x = np.where(grid == marker)
        if len(marker_y) < 4:
            continue
        if all(y == center_row for y in marker_y):
            orientation, line, center = "horizontal", center_row, center_col
            positions = sorted(int(x) for x in marker_x)
            length = width
        elif all(x == center_col for x in marker_x):
            orientation, line, center = "vertical", center_col, center_row
            positions = sorted(int(y) for y in marker_y)
            length = height
        else:
            continue
        if positions != list(range(positions[0], positions[-1] + 1, 2)):
            continue
        start, count = positions[0], len(positions)
        if center not in positions or center + 2 not in positions:
            continue
        if center < 1 or center + 3 >= length:
            continue
        expected, output = _render(height, width, orientation, line, start,
                                   count, center, background, marker, payload)
        if np.array_equal(grid, expected):
            candidates.append({"orientation": orientation, "line": line,
                               "start": start, "count": count, "center": center,
                               "axis": length, "transverse": height if orientation == "horizontal" else width,
                               "palette": (background, marker, payload),
                               "output": output})
    assert len(candidates) == 1, f"input has {len(candidates)} task interpretations"
    return candidates[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    descriptors = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _check_grid(inp)
            _check_grid(out)
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            input_key, pair_key = _key(inp), (_rows(inp), _rows(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert pair_key not in _SOURCE_PAIRS
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            descriptor = _decode_input(inp)
            assert np.array_equal(out, descriptor.pop("output"))
            descriptors[split].append(descriptor)

    # The direction is learned for each tested orientation. Color roles are
    # local to each input, and no unseen literal color convention is needed.
    trained_axes = {d["orientation"] for d in descriptors["train"]}
    assert {d["orientation"] for d in descriptors["test"]} <= trained_axes
    # The requested palette variation is a puzzle convention. If palettes
    # vary, each pair permutes the same three literal colors and training
    # demonstrates at least one such permutation.
    train_palettes = [d["palette"] for d in descriptors["train"]]
    all_palettes = train_palettes + [d["palette"] for d in descriptors["test"]]
    assert len({frozenset(p) for p in all_palettes}) == 1
    if len(set(all_palettes)) > 1:
        assert len(set(train_palettes)) > 1


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)

    # Puzzle-level choices are the available orientations and whether the
    # three visible colors keep their literal identities between pairs.
    mode = int(rng.integers(0, 3))
    palette_mode = int(rng.integers(0, 2))
    base = tuple(int(x) for x in rng.choice(10, 3, replace=False))
    if mode == 0:
        train_axes = ["horizontal"] * num_train
    elif mode == 1:
        train_axes = ["vertical"] * num_train
    else:
        first = "horizontal" if int(rng.integers(0, 2)) else "vertical"
        other = "vertical" if first == "horizontal" else "horizontal"
        train_axes = [first, other] + [first if int(rng.integers(0, 2)) else other
                                       for _ in range(num_train - 2)]
    test_axes = [train_axes[int(rng.integers(0, len(train_axes)))]
                 for _ in range(num_test)]

    puzzle = {"train": [], "test": []}
    seen = set()
    for split, axes in (("train", train_axes), ("test", test_axes)):
        for index, orientation in enumerate(axes):
            if not palette_mode or (split == "train" and index == 0):
                palette = base
            elif split == "train" and index == 1:
                palette = (base[0], base[2], base[1])
            else:
                order = rng.permutation(3)
                palette = tuple(base[int(j)] for j in order)
            for _ in range(120):
                child = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                inp, out = _sample_pair(child, split, orientation, palette)
                pair_key = (_rows(inp), _rows(out))
                if _key(inp) in seen or pair_key in _SOURCE_PAIRS:
                    continue
                seen.add(_key(inp))
                puzzle[split].append({"input": inp, "output": out})
                break
            else:
                raise ValueError("unable to construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test() -> None:
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1),
                        (None, 4), (2, None), (10, None), (None, None)):
        puzzle = generate_puzzle(800 + (train or 0) * 10 + (test or 0), train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 3), (4, 4)):
        try:
            generate_puzzle(500, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")

    counts_train, counts_test, puzzles = set(), set(), set()
    orientations = {"train": set(), "test": set()}
    phases = {"horizontal": set(), "vertical": set()}
    split_phases = {"train": {"horizontal": set(), "vertical": set()},
                    "test": {"horizontal": set(), "vertical": set()}}
    split_positions = {"train": set(), "test": set()}
    lengths = {"horizontal": set(), "vertical": set()}
    split_lengths = {"train": set(), "test": set()}
    positions = set()
    shapes, structures = set(), set()
    stable, varying, partial, full, edge, compact_test, long_vertical = 0, 0, 0, 0, 0, 0, 0
    unequal, equal, within_positions = 0, 0, 0
    same_axis, mixed_axis = 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        puzzles.add(_puzzle_key(puzzle))
        counts_train.add(len(puzzle["train"]))
        counts_test.add(len(puzzle["test"]))
        train_axes = {_decode_input(p["input"])["orientation"] for p in puzzle["train"]}
        same_axis += len(train_axes) == 1
        mixed_axis += len(train_axes) == 2
        palettes = set()
        puzzle_positions = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                d = _decode_input(inp)
                assert np.array_equal(out, d["output"])
                palettes.add(d["palette"])
                orientations[split].add(d["orientation"])
                phases[d["orientation"]].add(d["start"] % 2)
                split_phases[split][d["orientation"]].add(d["start"] % 2)
                lengths[d["orientation"]].add(d["axis"])
                split_lengths[split].add(d["axis"])
                shapes.add(inp.shape)
                structures.add((d["orientation"], d["axis"], d["transverse"],
                                d["line"], d["start"], d["count"], d["center"]))
                before = d["center"] - d["start"]
                after = d["start"] + 2 * (d["count"] - 1) - d["center"]
                position = "start" if before == 0 else "late" if after == 2 else "interior"
                positions.add(position)
                split_positions[split].add(position)
                puzzle_positions.add(position)
                unequal += before != after
                equal += before == after
                full += d["start"] <= 1 and d["start"] + 2 * (d["count"] - 1) >= d["axis"] - 2
                partial += d["start"] > 1 or d["start"] + 2 * (d["count"] - 1) < d["axis"] - 2
                edge += d["center"] + 2 == d["axis"] - 2
                compact_test += split == "test" and d["axis"] <= 12
                long_vertical += d["orientation"] == "vertical" and d["axis"] >= 20
        stable += len(palettes) == 1
        varying += len(palettes) > 1
        within_positions += len(puzzle_positions) > 1
    assert len(counts_train) >= 4 and len(counts_test) >= 3
    assert len(puzzles) >= 40 and len(shapes) >= 25 and len(structures) >= 150
    assert all(orientations[s] == {"horizontal", "vertical"} for s in orientations)
    assert all(phases[o] == {0, 1} for o in phases)
    assert all(split_phases[s][o] == {0, 1} for s in split_phases for o in split_phases[s])
    assert all(len(lengths[o]) >= 12 for o in lengths)
    assert positions == {"start", "interior", "late"}
    assert all(split_positions[s] == positions for s in split_positions)
    assert bool(split_lengths["train"] & split_lengths["test"])
    assert min(unequal, equal) > 0 and within_positions >= 30
    assert min(stable, varying, same_axis, mixed_axis, partial, full, edge,
               compact_test, long_vertical) > 0
