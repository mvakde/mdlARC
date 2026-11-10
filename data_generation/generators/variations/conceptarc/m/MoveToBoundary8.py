"""Solved variants of the evenly spaced markers sliding to a straight barrier."""

import hashlib

import numpy as np


_DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)
_CARDINAL = tuple(direction for direction in _DIRECTIONS if 0 in direction)
_DIAGONAL = tuple(direction for direction in _DIRECTIONS if 0 not in direction)
_SOURCE_PAIR_DIGESTS = frozenset({
    "fd00982eae321b2f2c8d7b52d487b99d903d4d7d1a4b3f17d421ffc50ef12d7c",
    "2693852f6566c5aa6690812c5a53601966262063ec8248227cc0da42b6b592cf",
    "b141a4171eb4bf09cb9e2d9f0af92875ef7d35d2e5de5bf2e4cd1b0fc6abda2c",
    "42c8ac4883388d06d5d3166561c78dff215e2e414f43bb090fd62262b0e3ef95",
    "feb6b8452a23bb0313a09ff119cce13913494ae46bb00e9e7b499466dadd1c5b",
})


def _valid_count(value, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"count must be an integer from {low} to {high}")
    return int(value)


def _grid_key(grid):
    return grid.shape, tuple(int(value) for value in grid.flat)


def _source_digest(pair):
    digest = hashlib.sha256()
    for role in ("input", "output"):
        grid = pair[role]
        digest.update(bytes(grid.shape))
        digest.update(np.asarray(grid, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _pick_travel(rng, regime, index):
    # The first two demonstrations both move; at least the first moves
    # farther than one cell. Other pairs span all four distance regimes.
    if regime == "train" and index == 0:
        return int(rng.integers(2, 9))
    if regime == "train" and index == 1:
        return int(rng.integers(1, 9))
    category = int(rng.choice(4, p=(0.14, 0.17, 0.42, 0.27)))
    if category == 0:
        return 0
    if category == 1:
        return 1
    if category == 2:
        return int(rng.integers(2, 6))
    return int(rng.integers(6, 13))


def _margins(rng, extent):
    room = 30 - extent
    extra = int(rng.integers(0, room + 1))
    before = int(rng.integers(0, extra + 1))
    return before, extra - before


def _make_pair(rng, direction, orientation, travel, background, colors):
    dr, dc = direction
    marker_color, barrier_color = colors
    assert len({marker_color, barrier_color, background}) == 3
    for attempt in range(361):
        if attempt == 360:
            marker_count, spacing, barrier_length, crossing_index = 3, 2, 4, 1
        else:
            marker_count = int(rng.choice((3, 4, 5, 6), p=(0.43, 0.32, 0.18, 0.07)))
            spacing = int(rng.choice((2, 3, 4, 5), p=(0.43, 0.30, 0.18, 0.09)))
            barrier_length = int(rng.integers(4, 15))
            crossing_index = int(rng.integers(0, barrier_length))
        if orientation == "vertical":
            barrier = [(i - crossing_index, 0) for i in range(barrier_length)]
        else:
            barrier = [(0, i - crossing_index) for i in range(barrier_length)]
        markers = [
            (-(travel + 1 + i * spacing) * dr,
             -(travel + 1 + i * spacing) * dc)
            for i in range(marker_count)
        ]
        points = barrier + markers
        min_row = min(row for row, _ in points)
        max_row = max(row for row, _ in points)
        min_col = min(col for _, col in points)
        max_col = max(col for _, col in points)
        core_height = max_row - min_row + 1
        core_width = max_col - min_col + 1
        if core_height > 30 or core_width > 30:
            continue
        top, bottom = _margins(rng, core_height)
        left, right = _margins(rng, core_width)
        height = core_height + top + bottom
        width = core_width + left + right
        row_shift, col_shift = top - min_row, left - min_col
        inp = np.full((height, width), background, dtype=np.int8)
        for row, col in barrier:
            inp[row + row_shift, col + col_shift] = barrier_color
        for row, col in markers:
            rr, cc = row + row_shift, col + col_shift
            assert inp[rr, cc] == background
            inp[rr, cc] = marker_color
        out = inp.copy()
        for row, col in markers:
            out[row + row_shift, col + col_shift] = background
        for row, col in markers:
            rr = row + row_shift + travel * dr
            cc = col + col_shift + travel * dc
            assert out[rr, cc] == background
            out[rr, cc] = marker_color
        return {"input": inp, "output": out}
    raise AssertionError("could not fit sampled geometry")


def _analyze_input(inp):
    colors, counts = np.unique(inp, return_counts=True)
    assert len(colors) == 3
    background = int(colors[np.argmax(counts)])
    foreground = [int(color) for color in colors if color != background]
    candidates = []
    for color in foreground:
        points = [tuple(map(int, point)) for point in np.argwhere(inp == color)]
        rows = [row for row, _ in points]
        cols = [col for _, col in points]
        if 4 <= len(points) <= 14:
            if (len(set(rows)) == 1
                    and sorted(cols) == list(range(min(cols), max(cols) + 1))):
                candidates.append((color, "horizontal", points))
            if (len(set(cols)) == 1
                    and sorted(rows) == list(range(min(rows), max(rows) + 1))):
                candidates.append((color, "vertical", points))
    assert len(candidates) == 1
    barrier_color, orientation, barrier = candidates[0]
    marker_color = next(color for color in foreground if color != barrier_color)
    markers = [tuple(map(int, point)) for point in np.argwhere(inp == marker_color)]
    assert 3 <= len(markers) <= 6
    height, width = inp.shape
    hits = []
    barrier_points = set(barrier)
    for dr, dc in _DIRECTIONS:
        if ((dr == 0 and orientation != "vertical")
                or (dc == 0 and orientation != "horizontal")):
            continue
        ordered = sorted(markers, key=lambda p: p[0] * dr + p[1] * dc)
        for spacing in range(2, 6):
            if any((ordered[i + 1][0] - ordered[i][0],
                    ordered[i + 1][1] - ordered[i][1]) != (spacing * dr, spacing * dc)
                   for i in range(len(ordered) - 1)):
                continue
            lead_row, lead_col = ordered[-1]
            distance = 1
            while 0 <= lead_row + distance * dr < height and 0 <= lead_col + distance * dc < width:
                point = (lead_row + distance * dr, lead_col + distance * dc)
                cell = int(inp[point])
                if point in barrier_points:
                    hits.append(((dr, dc), spacing, distance - 1, point))
                    break
                if cell != background:
                    break
                distance += 1
    assert len(hits) == 1
    direction, spacing, travel, crossing = hits[0]
    dr, dc = direction
    expected = inp.copy()
    for row, col in markers:
        expected[row, col] = background
    for row, col in markers:
        rr, cc = row + travel * dr, col + travel * dc
        assert 0 <= rr < height and 0 <= cc < width
        assert expected[rr, cc] == background
        expected[rr, cc] = marker_color
    rows, cols = np.nonzero(inp != background)
    margin = (int(rows.min()), height - 1 - int(rows.max()),
              int(cols.min()), width - 1 - int(cols.max()))
    return {
        "background": background, "marker_color": marker_color,
        "barrier_color": barrier_color, "direction": direction,
        "orientation": orientation, "travel": travel,
        "spacing": spacing, "marker_count": len(markers),
        "barrier_length": len(barrier),
        "crossing_index": (crossing[0] - min(row for row, _ in barrier)
                           if orientation == "vertical" else
                           crossing[1] - min(col for _, col in barrier)),
        "margins": margin, "expected_output": expected,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4
    assert train_count > test_count
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            key = (_grid_key(inp), _grid_key(out))
            assert key[0] not in seen_inputs and key not in seen_pairs
            assert _source_digest(pair) not in _SOURCE_PAIR_DIGESTS
            seen_inputs.add(key[0])
            seen_pairs.add(key)
            info = _analyze_input(inp)
            assert np.array_equal(out, info["expected_output"])
            infos[split].append(info)
    train = infos["train"]
    assert train[0]["travel"] >= 2 and train[1]["travel"] >= 1
    assert any(0 in info["direction"] for info in train)
    assert any(0 not in info["direction"] for info in train)
    assert len({info["direction"] for info in train}) >= 2
    assert len({info["marker_color"] for info in train}) >= 2
    assert len({info["barrier_color"] for info in train}) >= 2


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    if num_train is not None:
        num_train = _valid_count(num_train, 2, 10)
    if num_test is not None:
        num_test = _valid_count(num_test, 1, 4)
    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    # The geometric roles and stopping rule are the puzzle-level convention.
    # This palette policy varies by puzzle; colors themselves carry no role.
    variable_background = bool(rng.integers(0, 2))
    base_background = int(rng.integers(0, 10))
    first_cardinal = _CARDINAL[int(rng.integers(0, len(_CARDINAL)))]
    first_diagonal = _DIAGONAL[int(rng.integers(0, len(_DIAGONAL)))]
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    train_directions = []
    first_colors = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train" and index == 0:
                direction = first_cardinal
            elif split == "train" and index == 1:
                direction = first_diagonal
            elif split == "test" and index == 0 and int(rng.integers(0, 3)) == 0:
                direction = train_directions[0]
            else:
                direction = _DIRECTIONS[int(rng.integers(0, len(_DIRECTIONS)))]
            orientation = (
                "horizontal" if direction[1] == 0 else
                "vertical" if direction[0] == 0 else
                ("horizontal" if bool(rng.integers(0, 2)) else "vertical")
            )
            travel = _pick_travel(rng, split, index)
            if split == "test" and index == 0 and direction == train_directions[0]:
                if int(rng.integers(0, 3)) == 0:
                    travel = 0
            if not variable_background or (split == "train" and index == 0):
                background = base_background
            elif split == "train" and index == 1:
                choices = [color for color in range(10) if color != base_background]
                background = int(rng.choice(choices))
            else:
                background = int(rng.integers(0, 10))
            available = [color for color in range(10) if color != background]
            if split == "train" and index == 1:
                marker_choices = [color for color in available if color != first_colors[0]]
                marker_color = int(rng.choice(marker_choices))
                barrier_choices = [color for color in available
                                   if color != marker_color and color != first_colors[1]]
                barrier_color = int(rng.choice(barrier_choices))
            else:
                selected = rng.choice(available, size=2, replace=False)
                marker_color, barrier_color = map(int, selected)
            colors = (marker_color, barrier_color)
            if split == "train" and index == 0:
                first_colors = colors
            for _ in range(40):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _make_pair(pair_rng, direction, orientation, travel,
                                  background, colors)
                key = _grid_key(pair["input"])
                if key not in seen_inputs and _source_digest(pair) not in _SOURCE_PAIR_DIGESTS:
                    seen_inputs.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("could not make a distinct pair")
            if split == "train":
                train_directions.append(direction)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, signatures, shapes = set(), set(), set(), set()
    structural = set()
    directions = {"train": set(), "test": set()}
    travel_classes = {"train": set(), "test": set()}
    lengths, spacings, marker_counts, crossings = set(), set(), set(), set()
    variable_backgrounds = 0
    same_direction_stationary_moving = 0
    large_margin = 0
    edge_flush = 0
    diagonal_orientations = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        signatures.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        per_puzzle = []
        backgrounds = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_input(pair["input"])
                assert np.array_equal(pair["output"], info["expected_output"])
                assert _source_digest(pair) not in _SOURCE_PAIR_DIGESTS
                directions[split].add(info["direction"])
                travel = info["travel"]
                travel_classes[split].add(0 if travel == 0 else 1 if travel == 1
                                          else 2 if travel <= 5 else 3)
                lengths.add(info["barrier_length"])
                spacings.add(info["spacing"])
                marker_counts.add(info["marker_count"])
                crossings.add(info["crossing_index"] in
                              (0, info["barrier_length"] - 1))
                shapes.add(pair["input"].shape)
                structural.add((info["direction"], info["orientation"],
                                info["marker_count"], info["spacing"],
                                info["barrier_length"], info["crossing_index"], travel))
                backgrounds.add(info["background"])
                large_margin += max(info["margins"]) >= 7
                edge_flush += 0 in info["margins"]
                if 0 not in info["direction"]:
                    diagonal_orientations.add(info["orientation"])
                per_puzzle.append(info)
        variable_backgrounds += len(backgrounds) > 1
        same_direction_stationary_moving += any(
            a["direction"] == b["direction"] and
            ((a["travel"] == 0) != (b["travel"] == 0))
            for a in per_puzzle for b in per_puzzle)
    assert len(signatures) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert directions["train"] == set(_DIRECTIONS)
    assert directions["test"] == set(_DIRECTIONS)
    assert travel_classes["train"] == travel_classes["test"] == {0, 1, 2, 3}
    assert {4, 5, 8, 10, 12}.issubset(lengths)
    assert spacings == {2, 3, 4, 5}
    assert marker_counts == {3, 4, 5, 6}
    assert crossings == {False, True}
    assert diagonal_orientations == {"horizontal", "vertical"}
    assert len(shapes) >= 70 and len(structural) >= 150
    assert variable_backgrounds >= 8
    assert same_direction_stationary_moving >= 1
    assert large_margin >= 20 and edge_flush >= 20
    for seed, train_count, test_count in ((101, 2, 1), (102, 10, 1),
                                          (103, 10, 4), (104, 5, 3)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed, train_count, test_count))
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": None, "num_test": 1},
                   {"num_train": None, "num_test": 4},
                   {"num_train": 2, "num_test": None},
                   {"num_train": 10, "num_test": None}):
        puzzle = generate_puzzle(211, **kwargs)
        validate_generated_puzzle(puzzle)
        if kwargs["num_train"] is not None:
            assert len(puzzle["train"]) == kwargs["num_train"]
        if kwargs["num_test"] is not None:
            assert len(puzzle["test"]) == kwargs["num_test"]
    for kwargs in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 3},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts were accepted")
