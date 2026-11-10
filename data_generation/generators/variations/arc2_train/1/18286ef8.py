"""Generator for a marker-addressed 3-by-3 register puzzle."""

import hashlib

import numpy as np


DIRECTIONS = tuple(
    (row, col)
    for row in range(3)
    for col in range(3)
    if (row, col) != (1, 1)
)
CORNERS = ((0, 0), (0, 2), (2, 0), (2, 2))
CARDINALS = ((0, 1), (1, 0), (1, 2), (2, 1))
SOURCE_PAIR_HASHES = {
    "93292aa03a4c2d2120dd84a6b7608f159521b4dc63c3d2af2f816dbd7b3bfbf2",
    "87dd6e421b757dcebfa2b9cb7246cd484bc2a61e7f5918eb84521d61dc0676ee",
    "e4d70a79504099f21988f385ffc6d7a2b5814a4799de81fcb5de0436c6dd62cb",
    "a846e18227d2a112f51d67fc2965d62ced2480620a818920d3c4da02b721487e",
}


def _valid_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _counts(rng, num_train, num_test):
    if num_train is not None:
        if not _valid_int(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer in 2..10")
        num_train = int(num_train)
    if num_test is not None:
        if not _valid_int(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer in 1..4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _directions(rng, train_count, test_count):
    # A corner plus a cardinal address fixes the orientation of the register.
    first = CORNERS[int(rng.integers(0, 4))]
    second = CARDINALS[int(rng.integers(0, 4))]
    train = [first, second]
    if train_count >= 3:
        train.append(first if rng.random() < 0.55 else DIRECTIONS[int(rng.integers(0, 8))])
    while len(train) < train_count:
        candidates = [d for d in DIRECTIONS if d not in train]
        pool = candidates if candidates and rng.random() < 0.65 else DIRECTIONS
        train.append(pool[int(rng.integers(0, len(pool)))])
    rng.shuffle(train)

    familiar = list(dict.fromkeys(train))
    novel = [d for d in DIRECTIONS if d not in familiar]
    test = []
    if test_count == 1:
        pool = familiar if not novel or rng.random() < 0.35 else novel
        test.append(pool[int(rng.integers(0, len(pool)))])
    else:
        test.append(familiar[int(rng.integers(0, len(familiar)))])
        pool = novel if novel else [d for d in DIRECTIONS if d != test[0]]
        test.append(pool[int(rng.integers(0, len(pool)))])
        while len(test) < test_count:
            pool = [d for d in DIRECTIONS if d not in test]
            test.append(pool[int(rng.integers(0, len(pool)))])
        rng.shuffle(test)
    return train, test


def _dimension(rng, regime):
    draw = rng.random()
    if regime == "source_train":
        bounds = (7, 20) if draw < 0.60 else ((21, 30) if draw < 0.80 else (7, 30))
    else:
        bounds = (16, 30) if draw < 0.55 else ((7, 15) if draw < 0.80 else (7, 30))
    return int(rng.integers(bounds[0], bounds[1] + 1))


def _layout(rng, regime):
    height = _dimension(rng, regime)
    width = _dimension(rng, regime)
    top = int(rng.integers(1, height - 5))
    left = int(rng.integers(1, width - 5))
    return height, width, top, left


def _pair_colors(rng, instruction, first_roles=None):
    available = [color for color in range(10) if color != instruction]
    roles = tuple(int(x) for x in rng.permutation(available)[:4])
    if first_roles is not None:
        for _ in range(24):
            if all(a != b for a, b in zip(roles, first_roles)):
                break
            roles = tuple(int(x) for x in rng.permutation(available)[:4])
        else:
            roles = tuple(available[(available.index(color) + 1) % 9] for color in first_roles)
    background, divider, fill, result = roles
    palette = tuple(color for color in available if color not in roles)
    return background, divider, fill, result, palette


def _macro(position, first, second):
    if position < first:
        return 0
    if first < position < second:
        return 1
    if position > second:
        return 2
    raise AssertionError("divider cell has no compartment")


def _outer_cells(height, width, row0, col0):
    row1, col1 = row0 + 4, col0 + 4
    return [
        (row, col, _macro(row, row0, row1), _macro(col, col0, col1))
        for row in range(height)
        if row not in (row0, row1)
        for col in range(width)
        if col not in (col0, col1)
        if not (row0 < row < row1 and col0 < col < col1)
    ]


def _clutter_count(rng, capacity):
    mode = int(rng.choice(4, p=(0.18, 0.29, 0.28, 0.25)))
    if mode == 0:
        return int(rng.integers(0, min(2, capacity) + 1))
    if mode == 1:
        low, high = 2, min(12, max(2, capacity // 10))
    elif mode == 2:
        low, high = 5, min(45, max(5, capacity // 7))
    else:
        low, high = 8, min(220, max(8, capacity * 3 // 10))
    high = min(high, capacity * 3 // 10)
    low = min(low, high)
    return int(rng.integers(low, high + 1))


def _motif(rng, cells, count, mode):
    if count < 3 or mode not in ("run", "cluster", "repeat"):
        return []
    allowed = set(cells)
    patterns = []
    if mode == "run":
        for row, col in cells:
            for dr, dc in ((1, 0), (0, 1)):
                group = [(row + k * dr, col + k * dc) for k in range(3)]
                if all(cell in allowed for cell in group):
                    patterns.append(group)
    elif mode == "cluster":
        if count >= 4:
            for row, col in cells:
                group = [(row, col), (row + 1, col), (row, col + 1), (row + 1, col + 1)]
                if all(cell in allowed for cell in group):
                    patterns.append(group)
    else:
        if count >= 4:
            for row, col in cells:
                for dr, dc in ((1, 0), (0, 1)):
                    group = [(row, col), (row + dr, col + dc)]
                    if all(cell in allowed for cell in group):
                        patterns.append(group)
            rng.shuffle(patterns)
            for first in patterns:
                for second in patterns:
                    if not set(first) & set(second):
                        return first + second
    if patterns:
        return patterns[int(rng.integers(0, len(patterns)))]
    return []


def _clutter(rng, grid, outer, marker, direction, palette):
    cells = [(r, c) for r, c, _, _ in outer if (r, c) != marker]
    count = _clutter_count(rng, len(cells))
    if count == 0:
        return
    mode = ("scatter", "run", "cluster", "repeat", "concentrated")[
        int(rng.integers(0, 5))
    ]
    motif = _motif(rng, cells, count, mode)
    selected = list(motif)
    used = set(selected)
    pool = [cell for cell in cells if cell not in used]
    if mode == "concentrated":
        target = [
            (r, c) for r, c, mr, mc in outer
            if (mr, mc) == direction and (r, c) != marker and (r, c) not in used
        ]
        rng.shuffle(target)
        take = min(len(target), max(0, (count * 2 // 3) - len(selected)))
        selected.extend(target[:take])
        used.update(target[:take])
        pool = [cell for cell in pool if cell not in used]
    remaining = count - len(selected)
    if remaining:
        indices = rng.choice(len(pool), size=remaining, replace=False)
        selected.extend(pool[int(index)] for index in indices)
    motif_size = len(motif)
    active_limit = min(len(palette), 1 + count - motif_size)
    active_count = int(rng.integers(1, active_limit + 1))
    colors = [int(x) for x in rng.choice(palette, size=active_count, replace=False)]
    assignments = [colors[0]] * motif_size
    remainder = count - motif_size
    if remainder:
        tail = colors[1:] + [colors[0]] * max(0, remainder - active_count + 1)
        rng.shuffle(tail)
        assignments.extend(tail)
    for (row, col), color in zip(selected, assignments):
        grid[row, col] = color


def _make_pair(rng, instruction, direction, roles, layout, away_from=None):
    height, width, row0, col0 = layout
    background, divider, fill, result, palette = roles
    row1, col1 = row0 + 4, col0 + 4
    inp = np.full((height, width), background, dtype=np.uint8)
    inp[[row0, row1], :] = divider
    inp[:, [col0, col1]] = divider
    inp[row0 + 1:row1, col0 + 1:col1] = fill
    center = (row0 + 2, col0 + 2)
    inp[center] = result
    outer = _outer_cells(height, width, row0, col0)
    options = [(r, c) for r, c, mr, mc in outer if (mr, mc) == direction]
    if away_from is not None and len(options) > 1:
        distances = [abs(r - away_from[0]) + abs(c - away_from[1]) for r, c in options]
        far = [cell for cell, distance in zip(options, distances) if distance >= max(distances) - 1]
        marker = far[int(rng.integers(0, len(far)))]
    else:
        marker = options[int(rng.integers(0, len(options)))]
    inp[marker] = instruction
    _clutter(rng, inp, outer, marker, direction, palette)
    out = inp.copy()
    out[marker] = result
    out[center] = fill
    out[row0 + 1 + direction[0], col0 + 1 + direction[1]] = result
    return {"input": inp, "output": out}


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _pair_hash(pair):
    digest = hashlib.sha256()
    for role in ("input", "output"):
        grid = pair[role]
        digest.update(bytes(grid.shape))
        digest.update(np.asarray(grid, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _inspect_pair(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    height, width = inp.shape
    rows = [r for r in range(height) if np.all(inp[r] == inp[r, 0])]
    cols = [c for c in range(width) if np.all(inp[:, c] == inp[0, c])]
    assert len(rows) == len(cols) == 2
    row0, row1 = rows
    col0, col1 = cols
    assert row1 == row0 + 4 and col1 == col0 + 4
    assert row0 >= 1 and col0 >= 1 and row1 < height - 1 and col1 < width - 1
    divider = int(inp[row0, 0])
    mask = np.zeros(inp.shape, dtype=bool)
    mask[rows, :] = True
    mask[:, cols] = True
    assert np.all(inp[mask] == divider) and not np.any(inp[~mask] == divider)
    register = inp[row0 + 1:row1, col0 + 1:col1]
    fill = int(register[0, 0])
    result = int(register[1, 1])
    assert fill != result
    assert np.all(register[np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)] == fill)
    changed = [tuple(map(int, point)) for point in np.argwhere(inp != out)]
    assert len(changed) == 3
    external = [(r, c) for r, c in changed if not (row0 < r < row1 and col0 < c < col1)]
    assert len(external) == 1
    marker = external[0]
    instruction = int(inp[marker])
    direction = (_macro(marker[0], row0, row1), _macro(marker[1], col0, col1))
    assert direction != (1, 1)
    assert np.count_nonzero(inp == instruction) == 1
    assert np.count_nonzero(inp == result) == 1
    assert np.count_nonzero(inp == fill) == 8
    outer_mask = ~mask
    outer_mask[row0 + 1:row1, col0 + 1:col1] = False
    outer = inp[outer_mask]
    colors, counts = np.unique(outer, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    assert int(counts.max()) * 2 > outer.size
    roles = (background, divider, fill, instruction, result)
    assert len(set(roles)) == 5
    assert not np.any(np.isin(outer, (divider, fill, result)))
    expected = inp.copy()
    expected[marker] = result
    expected[row0 + 2, col0 + 2] = fill
    expected[row0 + 1 + direction[0], col0 + 1 + direction[1]] = result
    assert np.array_equal(out, expected)
    return {
        "roles": roles,
        "direction": direction,
        "layout": (height, width, row0, col0),
        "marker": marker,
        "clutter": int(np.count_nonzero((outer != background) & (outer != instruction))),
    }


def _predict(train, test_input):
    instruction_colors = []
    for pair in train:
        info = _inspect_pair(pair)
        instruction_colors.append(info["roles"][3])
    assert len(set(instruction_colors)) == 1
    instruction = instruction_colors[0]
    inp = test_input
    rows = [r for r in range(inp.shape[0]) if np.all(inp[r] == inp[r, 0])]
    cols = [c for c in range(inp.shape[1]) if np.all(inp[:, c] == inp[0, c])]
    assert len(rows) == len(cols) == 2
    points = np.argwhere(inp == instruction)
    assert len(points) == 1
    row, col = map(int, points[0])
    direction = (_macro(row, *rows), _macro(col, *cols))
    fill = int(inp[rows[0] + 1, cols[0] + 1])
    result = int(inp[rows[0] + 2, cols[0] + 2])
    out = inp.copy()
    out[row, col] = result
    out[rows[0] + 2, cols[0] + 2] = fill
    out[rows[0] + 1 + direction[0], cols[0] + 1 + direction[1]] = result
    return out


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    inputs, pairs = set(), set()
    metadata = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            key = (_grid_key(pair["input"]), _grid_key(pair["output"]))
            assert key[0] not in inputs and key not in pairs
            assert _pair_hash(pair) not in SOURCE_PAIR_HASHES
            inputs.add(key[0])
            pairs.add(key)
            metadata[split].append(_inspect_pair(pair))
    all_info = metadata["train"] + metadata["test"]
    assert len({item["roles"][3] for item in all_info}) == 1
    train_directions = {item["direction"] for item in metadata["train"]}
    assert any(direction in CORNERS for direction in train_directions)
    assert any(direction in CARDINALS for direction in train_directions)
    for role_index in (0, 1, 2, 4):
        assert len({item["roles"][role_index] for item in metadata["train"]}) >= 2
    for pair in puzzle["test"]:
        assert np.array_equal(_predict(puzzle["train"], pair["input"]), pair["output"])


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    instruction = int(rng.integers(0, 10))
    train_directions, test_directions = _directions(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    first_roles = None
    layouts = []
    prior_by_direction = {}
    seen = set()
    for split, regime, directions in (
        ("train", "source_train", train_directions),
        ("test", "target_test", test_directions),
    ):
        for index, direction in enumerate(directions):
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                roles = _pair_colors(pair_rng, instruction, first_roles if split == "train" and index == 1 else None)
                prior = prior_by_direction.get(direction)
                if prior is not None and pair_rng.random() < 0.55:
                    layout = prior[0]
                    away_from = prior[1]
                else:
                    layout = layouts[int(pair_rng.integers(0, len(layouts)))] if layouts and pair_rng.random() < 0.24 else _layout(pair_rng, regime)
                    away_from = None
                pair = _make_pair(pair_rng, instruction, direction, roles, layout, away_from)
                signature = _grid_key(pair["input"])
                if signature not in seen and _pair_hash(pair) not in SOURCE_PAIR_HASHES:
                    seen.add(signature)
                    layouts.append(layout)
                    puzzle[split].append(pair)
                    if split == "train" and index == 0:
                        first_roles = roles[:4]
                    prior_by_direction[direction] = (layout, _inspect_pair(pair)["marker"])
                    break
            else:
                raise RuntimeError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple(
        tuple((_grid_key(pair["input"]), _grid_key(pair["output"])) for pair in puzzle[split])
        for split in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(puzzle["train"]) for puzzle in puzzles}
    test_counts = {len(puzzle["test"]) for puzzle in puzzles}
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len({_signature(puzzle) for puzzle in puzzles}) == 42
    infos = [_inspect_pair(pair) for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split]]
    assert set(item["direction"] for item in infos) == set(DIRECTIONS)
    assert len(set(item["layout"] for item in infos)) >= 100
    assert len(set((item["layout"][0], item["layout"][1]) for item in infos)) >= 75
    assert min(item["clutter"] for item in infos) == 0
    assert max(item["clutter"] for item in infos) >= 30
    assert len({item["roles"] for item in infos}) >= 100
    assert any(item["layout"][0] == 7 or item["layout"][1] == 7 for item in infos)
    assert any(item["layout"][0] >= 25 or item["layout"][1] >= 25 for item in infos)
    assert any(item["layout"][0] >= 2 * item["layout"][1] or item["layout"][1] >= 2 * item["layout"][0] for item in infos)
    assert any(item["layout"] in {x["layout"] for x in [_inspect_pair(p) for p in puzzle["train"]]} for puzzle in puzzles for item in [_inspect_pair(p) for p in puzzle["test"]])
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        assert len({_inspect_pair(pair)["roles"][4] for pair in puzzle["train"]}) >= 2
        for pair in puzzle["test"]:
            assert np.array_equal(_predict(puzzle["train"], pair["input"]), pair["output"])
    for seed, train, test in (
        (101, 2, 1), (102, 10, 1), (103, 10, 4),
        (104, None, 1), (105, None, 4), (106, 2, None), (107, 10, None),
    ):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert (train is None or len(puzzle["train"]) == train)
        assert (test is None or len(puzzle["test"]) == test)
    assert _signature(generate_puzzle(999, 7, 3)) == _signature(generate_puzzle(999, 7, 3))
    assert _signature(generate_puzzle(998)) != _signature(generate_puzzle(999))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (3, False)):
        try:
            generate_puzzle(200, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count request accepted")
