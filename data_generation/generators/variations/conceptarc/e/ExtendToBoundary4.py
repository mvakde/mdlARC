"""Generate two-stroke ARC puzzles with a geometric stopping boundary."""

import numpy as np


_SOURCE_INPUTS = {
    ("000000", "111110", "000000", "000200", "000200", "000200"),
    ("060000", "060000", "000000", "000000", "333330", "000000"),
    ("000040", "000040", "000040", "000000", "000077", "000000"),
    ("080000", "080999", "080000", "000000", "000000", "000000"),
    ("000000", "000000", "006000", "006000", "006077", "006000"),
    ("000800", "000000", "033333", "000000", "000000", "000000"),
    ("200008", "000080", "000800", "008000", "080000", "800000"),
}

# Boundary direction, perpendicular growth direction.
_DIRECTIONS = {
    "H": ((0, 1), (1, 0)),
    "V": ((1, 0), (0, 1)),
    "plus": ((1, 1), (1, -1)),
    "anti": ((1, -1), (1, 1)),
}


def _counts(rng, num_train, num_test):
    for value, low, high, label in (
        (num_train, 2, 10, "num_train"),
        (num_test, 1, 4, "num_test"),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"{label} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _metric(kind, point):
    row, col = point
    if kind == "H":
        return row
    if kind == "V":
        return col
    if kind == "plus":
        return row - col
    return row + col


def _line_kind(cells):
    cells = set(cells)
    if len(cells) == 1:
        return "point"
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) == 1 and cols == set(range(min(cols), max(cols) + 1)):
        return "H"
    if len(cols) == 1 and rows == set(range(min(rows), max(rows) + 1)):
        return "V"
    if (
        len({r - c for r, c in cells}) == 1
        and rows == set(range(min(rows), max(rows) + 1))
    ):
        return "plus"
    if (
        len({r + c for r, c in cells}) == 1
        and rows == set(range(min(rows), max(rows) + 1))
    ):
        return "anti"
    return None


def _full_lines(height, width, kind):
    groups = {}
    for row in range(height):
        for col in range(width):
            key = _metric(kind, (row, col)) if kind in ("plus", "anti") else (
                row if kind == "H" else col
            )
            groups.setdefault(key, []).append((row, col))
    return [
        sorted(cells, key=lambda p: p[1] if kind == "H" else p[0])
        for cells in groups.values() if len(cells) >= 2
    ]


def _inside(point, height, width):
    return 0 <= point[0] < height and 0 <= point[1] < width


def _configurations(height, width, kind, boundary):
    """Enumerate visible seeds that stop at a cell or between two cells."""
    _, growth = _DIRECTIONS[kind]
    crosses = [(2 * r, 2 * c, "cell", i) for i, (r, c) in enumerate(boundary)]
    if kind in ("plus", "anti"):
        crosses.extend(
            (boundary[i][0] + boundary[i + 1][0],
             boundary[i][1] + boundary[i + 1][1], "between", i)
            for i in range(len(boundary) - 1)
        )
    options = []
    area = height * width
    for row2, col2, parity, index in crosses:
        for side in (-1, 1):
            for gap in range(1, max(height, width)):
                offset2 = 2 * (gap + 1) if parity == "cell" else 2 * gap + 1
                near2 = (
                    row2 + side * offset2 * growth[0],
                    col2 + side * offset2 * growth[1],
                )
                if near2[0] % 2 or near2[1] % 2:
                    continue
                near = (near2[0] // 2, near2[1] // 2)
                if not _inside(near, height, width):
                    continue
                for length in range(1, max(height, width) + 1):
                    far = (
                        near[0] + side * (length - 1) * growth[0],
                        near[1] + side * (length - 1) * growth[1],
                    )
                    if not _inside(far, height, width):
                        break
                    if len(boundary) + length + gap >= area // 2:
                        continue
                    seed = tuple(
                        (near[0] + side * step * growth[0],
                         near[1] + side * step * growth[1])
                        for step in range(length)
                    )
                    added = tuple(
                        (near[0] - side * step * growth[0],
                         near[1] - side * step * growth[1])
                        for step in range(1, gap + 1)
                    )
                    if any(not _inside(p, height, width) for p in added):
                        continue
                    if set(boundary) & (set(seed) | set(added)):
                        continue
                    options.append((seed, added, parity, gap, length, index, side))
    return options


def _grid_key(grid):
    return grid.shape, tuple(int(value) for value in grid.flat)


def _grid_rows(grid):
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def _construct_pair(rng, kind, point=None, gap_band=None, parity=None):
    for _ in range(180):
        height = int(rng.integers(5, 17))
        width = int(rng.integers(5, 17))
        lines = _full_lines(height, width, kind)
        line = lines[int(rng.integers(len(lines)))]
        boundary_length = int(rng.integers(2, len(line) + 1))
        start = int(rng.integers(len(line) - boundary_length + 1))
        boundary = tuple(line[start:start + boundary_length])
        options = _configurations(height, width, kind, boundary)
        if point is not None:
            options = [x for x in options if (x[4] == 1) == point]
        if gap_band is not None:
            options = [x for x in options if (x[3] == 1) == (gap_band == "short")]
        if parity is not None:
            options = [x for x in options if x[2] == parity]
        if not options:
            continue
        # Choose the crossing type and gap before the exact seed length so
        # easy, dense layouts do not drown out long continuations.
        parity_choice = options[int(rng.integers(len(options)))][2]
        options = [x for x in options if x[2] == parity_choice]
        gaps = sorted({x[3] for x in options})
        gap = gaps[int(rng.integers(len(gaps)))]
        options = [x for x in options if x[3] == gap]
        lengths = sorted({x[4] for x in options})
        length = lengths[int(rng.integers(len(lengths)))]
        options = [x for x in options if x[4] == length]
        seed, added, _, _, _, _, _ = options[int(rng.integers(len(options)))]
        background, boundary_color, seed_color = (
            int(x) for x in rng.choice(10, size=3, replace=False)
        )
        input_grid = np.full((height, width), background, dtype=np.int8)
        for row, col in boundary:
            input_grid[row, col] = boundary_color
        for row, col in seed:
            input_grid[row, col] = seed_color
        output_grid = input_grid.copy()
        for row, col in added:
            output_grid[row, col] = seed_color
        try:
            predicted, _ = _predict(input_grid)
        except AssertionError:
            continue
        if np.array_equal(predicted, output_grid):
            return {"input": input_grid, "output": output_grid}
    raise ValueError("could not construct a clear two-stroke pair")


def _candidate(input_grid, background, boundary_color, seed_color):
    height, width = input_grid.shape
    boundary = {
        (int(r), int(c)) for r, c in np.argwhere(input_grid == boundary_color)
    }
    seed = {(int(r), int(c)) for r, c in np.argwhere(input_grid == seed_color)}
    kind = _line_kind(boundary)
    if kind not in _DIRECTIONS or len(boundary) < 2:
        return None
    seed_kind = _line_kind(seed)
    expected_kind = {"H": "V", "V": "H", "plus": "anti", "anti": "plus"}[kind]
    if seed_kind not in ("point", expected_kind):
        return None
    intercept = _metric(kind, next(iter(boundary)))
    differences = {_metric(kind, p) - intercept for p in seed}
    if not differences or 0 in differences:
        return None
    sides = {int(np.sign(value)) for value in differences}
    if len(sides) != 1:
        return None
    side = sides.pop()
    near = min(seed, key=lambda p: abs(_metric(kind, p) - intercept))
    _, growth = _DIRECTIONS[kind]
    step = (-side * growth[0], -side * growth[1])
    result = set(seed)
    cursor = near
    added = []
    for _ in range(max(height, width) + 1):
        next_cell = (cursor[0] + step[0], cursor[1] + step[1])
        if not _inside(next_cell, height, width):
            return None
        next_difference = _metric(kind, next_cell) - intercept
        if side * next_difference <= 0:
            if next_difference == 0:
                if next_cell not in boundary:
                    return None
                parity = "cell"
                crossing = next_cell
            else:
                if kind in ("H", "V"):
                    return None
                brackets = {
                    (cursor[0] + step[0], cursor[1]),
                    (cursor[0], cursor[1] + step[1]),
                }
                if not brackets <= boundary:
                    return None
                parity = "between"
                crossing = tuple((cursor[k] + next_cell[k]) / 2 for k in (0, 1))
            break
        if next_cell in seed or next_cell in boundary:
            return None
        result.add(next_cell)
        added.append(next_cell)
        cursor = next_cell
    else:
        return None
    if not added or _line_kind(result) != expected_kind:
        return None
    output = input_grid.copy()
    for row, col in added:
        output[row, col] = seed_color
    details = {
        "kind": kind, "parity": parity, "gap": len(added),
        "seed_length": len(seed), "boundary_length": len(boundary),
        "background": background, "seed_color": seed_color,
        "boundary_color": boundary_color, "side": side,
        "crossing": crossing,
    }
    return output, details


def _predict(input_grid):
    colors, counts = np.unique(input_grid, return_counts=True)
    assert len(colors) == 3
    largest = int(np.argmax(counts))
    assert counts[largest] > input_grid.size // 2
    background = int(colors[largest])
    foreground = [int(color) for color in colors if color != background]
    choices = []
    for boundary_color, seed_color in (foreground, foreground[::-1]):
        candidate = _candidate(input_grid, background, boundary_color, seed_color)
        if candidate is not None:
            choices.append(candidate)
    assert len(choices) == 1
    return choices[0]


def _schedule(rng, train_count, test_count):
    # These modes vary the curriculum, not the rule. No color or geometric
    # convention is hidden across pairs.
    mode = ("axis_only", "mixed", "held_diagonal")[int(rng.integers(3))]
    if mode == "axis_only":
        train = ["H", "V"] + [
            ("H", "V")[int(rng.integers(2))] for _ in range(train_count - 2)
        ]
        test = [("H", "V")[int(rng.integers(2))] for _ in range(test_count)]
    elif mode == "held_diagonal":
        train = ["H", "V"] + [
            ("H", "V")[int(rng.integers(2))] for _ in range(train_count - 2)
        ]
        test = [("plus", "anti")[int(rng.integers(2))]] + [
            ("H", "V", "plus", "anti")[int(rng.integers(4))]
            for _ in range(test_count - 1)
        ]
    else:
        train = [
            ("H", "V")[int(rng.integers(2))],
            ("plus", "anti")[int(rng.integers(2))],
        ] + [
            ("H", "V", "plus", "anti")[int(rng.integers(4))]
            for _ in range(train_count - 2)
        ]
        test = [("plus", "anti")[int(rng.integers(2))]] + [
            ("H", "V", "plus", "anti")[int(rng.integers(4))]
            for _ in range(test_count - 1)
        ]
    rng.shuffle(train)
    rng.shuffle(test)
    return mode, train + test


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    mode, kinds = _schedule(rng, train_count, test_count)
    total = train_count + test_count
    point_flags = [True, False] + [bool(rng.integers(2)) for _ in range(total - 2)]
    gap_bands = ["short", "long"] + [
        ("short", "long")[int(rng.integers(2))] for _ in range(total - 2)
    ]
    rng.shuffle(point_flags)
    rng.shuffle(gap_bands)
    if mode == "held_diagonal":
        # Keep a source-like held-out diagonal point as one recurring regime.
        diagonal_test = next(
            i for i in range(train_count, total) if kinds[i] in ("plus", "anti")
        )
        point_flags[diagonal_test] = True
        point_flags[0] = False
    pair_seeds = rng.integers(0, 2**63 - 1, size=total, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen = set()
    for index, kind in enumerate(kinds):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        for _ in range(30):
            parity = None
            if (
                mode == "held_diagonal" and index >= train_count
                and point_flags[index] and kind in ("plus", "anti")
            ):
                parity = "between"
            pair = _construct_pair(
                pair_rng, kind, point=point_flags[index],
                gap_band=gap_bands[index], parity=parity,
            )
            key = _grid_key(pair["input"])
            if key in seen or _grid_rows(pair["input"]) in _SOURCE_INPUTS:
                continue
            if index == total - 1:
                previous = puzzle["train"] + puzzle["test"]
                if all(p["input"].shape == pair["input"].shape for p in previous):
                    continue
                backgrounds = [_predict(p["input"])[1]["background"] for p in previous]
                if all(bg == _predict(pair["input"])[1]["background"] for bg in backgrounds):
                    continue
            seen.add(key)
            split = "train" if index < train_count else "test"
            puzzle[split].append(pair)
            break
        else:
            raise ValueError("could not make unique diverse pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    infos = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            assert _grid_rows(inp) not in _SOURCE_INPUTS
            key = _grid_key(inp)
            pair_key = (key, _grid_key(out))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            predicted, info = _predict(inp)
            assert np.array_equal(predicted, out)
            assert info["gap"] >= 1
            infos.append(info)
    assert len({p["input"].shape for s in puzzle for p in puzzle[s]}) >= 2
    assert len({info["background"] for info in infos}) >= 2
    assert {info["seed_length"] == 1 for info in infos} == {True, False}
    assert {info["gap"] == 1 for info in infos} == {True, False}
    train_kinds = {info["kind"] for info in infos[:len(puzzle["train"])]}
    test_kinds = {info["kind"] for info in infos[len(puzzle["train"]):]}
    assert len(train_kinds) >= 2
    if test_kinds & {"plus", "anti"} and not train_kinds & {"plus", "anti"}:
        assert train_kinds == {"H", "V"}


def _same_puzzle(first, second):
    return all(
        len(first[split]) == len(second[split])
        and all(
            np.array_equal(a[field], b[field])
            for a, b in zip(first[split], second[split])
            for field in ("input", "output")
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts = set(), set()
    kinds, parities, shapes, backgrounds = set(), set(), set(), set()
    gaps, seed_lengths, boundary_lengths, sides = set(), set(), set(), set()
    structures, fingerprints = set(), set()
    modes_seen = set()
    within_shapes = within_backgrounds = within_gaps = within_seeds = 0
    diagonal_multi = diagonal_finite = diagonal_cell = diagonal_between = 0
    combined_diagonal = train_point_puzzles = 0
    crossing_positions = set()
    length_ratios = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pair_infos = []
        fingerprint = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                predicted, info = _predict(pair["input"].copy())
                assert np.array_equal(predicted, pair["output"])
                pair_infos.append((pair["input"].shape, info))
                kinds.add(info["kind"])
                parities.add(info["parity"])
                shapes.add(pair["input"].shape)
                backgrounds.add(info["background"])
                gaps.add(info["gap"])
                seed_lengths.add(info["seed_length"])
                boundary_lengths.add(info["boundary_length"])
                sides.add((info["kind"], info["side"]))
                length_ratios.append(info["seed_length"] / info["boundary_length"])
                boundary = {
                    (int(r), int(c))
                    for r, c in np.argwhere(
                        pair["input"] == info["boundary_color"]
                    )
                }
                full_line = next(
                    set(line)
                    for line in _full_lines(*pair["input"].shape, info["kind"])
                    if boundary <= set(line)
                )
                finite = boundary != full_line
                ordered = sorted(
                    boundary,
                    key=lambda p: p[1] if info["kind"] == "H" else p[0],
                )
                if info["parity"] == "cell":
                    is_endpoint = info["crossing"] in (ordered[0], ordered[-1])
                else:
                    midpoints = [
                        tuple((a[k] + b[k]) / 2 for k in (0, 1))
                        for a, b in zip(ordered, ordered[1:])
                    ]
                    assert info["crossing"] in midpoints
                    crossing_index = midpoints.index(info["crossing"])
                    is_endpoint = crossing_index in (0, len(midpoints) - 1)
                crossing_positions.add(
                    "endpoint" if is_endpoint else "interior"
                )
                if info["kind"] in ("plus", "anti"):
                    diagonal_multi += info["seed_length"] > 1
                    diagonal_finite += finite
                    diagonal_cell += info["parity"] == "cell"
                    diagonal_between += info["parity"] == "between"
                    combined_diagonal += (
                        info["parity"] == "between"
                        and info["seed_length"] > 1
                        and finite and info["background"] != 0
                    )
                # Color-independent occupancy plus orientation/gap/length.
                structures.add((
                    pair["input"].shape, info["kind"], info["parity"],
                    info["gap"], info["seed_length"], info["boundary_length"],
                ))
                fingerprint.append((_grid_key(pair["input"]), _grid_key(pair["output"])))
        fingerprints.add(tuple(fingerprint))
        within_shapes += len({x[0] for x in pair_infos}) > 1
        within_backgrounds += len({x[1]["background"] for x in pair_infos}) > 1
        within_gaps += len({x[1]["gap"] for x in pair_infos}) > 1
        within_seeds += len({x[1]["seed_length"] == 1 for x in pair_infos}) > 1
        train_point_puzzles += any(
            x[1]["seed_length"] == 1
            for x in pair_infos[:len(puzzle["train"])]
        )
        train_kinds = {x[1]["kind"] for x in pair_infos[:len(puzzle["train"])]}
        test_kinds = {x[1]["kind"] for x in pair_infos[len(puzzle["train"]):]}
        if not (train_kinds | test_kinds) & {"plus", "anti"}:
            modes_seen.add("axis_only")
        elif not train_kinds & {"plus", "anti"}:
            modes_seen.add("held_diagonal")
        else:
            modes_seen.add("mixed")
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert kinds == set(_DIRECTIONS) and parities == {"cell", "between"}
    assert modes_seen == {"axis_only", "mixed", "held_diagonal"}
    assert len(shapes) >= 30 and min(min(s) for s in shapes) == 5
    assert max(max(s) for s in shapes) >= 13
    assert len(backgrounds) >= 8 and 0 in backgrounds
    assert max(gaps) >= 5 and max(seed_lengths) >= 4
    assert max(boundary_lengths) >= 7
    assert len(sides) == 8
    assert len(structures) >= 120 and len(fingerprints) == 42
    assert (within_shapes, within_backgrounds, within_gaps, within_seeds) == (42, 42, 42, 42)
    assert train_point_puzzles >= 25
    assert diagonal_multi >= 20 and diagonal_finite >= 20
    assert diagonal_cell >= 15 and diagonal_between >= 15
    assert combined_diagonal >= 2
    assert crossing_positions == {"endpoint", "interior"}
    assert min(length_ratios) <= 0.2 and max(length_ratios) >= 3
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (4, 2)):
        puzzle = generate_puzzle(1234, train_count, test_count)
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count
        assert _same_puzzle(puzzle, generate_puzzle(1234, train_count, test_count))
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(9876, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for train_count, test_count in (
        (1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4),
    ):
        try:
            generate_puzzle(0, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
