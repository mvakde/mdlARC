import numpy as np

VEC = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
PERP = {"U": ("L", "R"), "D": ("L", "R"), "L": ("U", "D"), "R": ("U", "D")}
SOURCE_INPUTS = {
"0000000000000000000000000000000050000000005000000000555550000050000000005000000000000000000000000000",
"0000000000000007777700000007000000000700000000070000000007000000000700000000000000000000000000000000",
"0000000000000000000000040000000004000000000400000000040000000044400000000000000000000000000000000000",
"0000000000000000000000000002000000000200000000020000222222000000000200000000020000000002000000000000",
"0000000000008000000000888800000080000000000000000000000008000000888800000000080000000000000000000000",
}


def _cells(direction, center, arm, stem):
    r, c = center
    result = {(r, c)}
    for side in (*PERP[direction], direction):
        dr, dc = VEC[side]
        result.update((r + k * dr, c + k * dc)
                      for k in range(1, (stem if side == direction else arm) + 1))
    return result


def _distance(direction, center, stem, height, width):
    r, c = center
    return {"U": r - stem, "D": height - 1 - r - stem,
            "L": c - stem, "R": width - 1 - c - stem}[direction]


def _shift(cells, direction, distance):
    dr, dc = VEC[direction]
    return {(r + distance * dr, c + distance * dc) for r, c in cells}


def _sweep(cells, direction, distance):
    return set().union(*(_shift(cells, direction, k) for k in range(distance + 1)))


def _separate(cells, occupied):
    return all((r, c) not in occupied and
               all((r + dr, c + dc) not in occupied for dr, dc in VEC.values())
               for r, c in cells)


def _candidate(rng, direction, height, width):
    perpendicular = height if direction in "LR" else width
    longitudinal = width if direction in "LR" else height
    arm = int(rng.integers(1, min(5, (perpendicular - 2) // 2) + 1))
    stem = int(rng.integers(1, min(9, longitudinal - 3) + 1))
    if direction in "LR":
        r = int(rng.integers(arm, height - arm))
        c = (int(rng.integers(stem + 1, width)) if direction == "L"
             else int(rng.integers(0, width - stem - 1)))
    else:
        c = int(rng.integers(arm, width - arm))
        r = (int(rng.integers(stem + 1, height)) if direction == "U"
             else int(rng.integers(0, height - stem - 1)))
    center = (r, c)
    distance = _distance(direction, center, stem, height, width)
    start = _cells(direction, center, arm, stem)
    return start, _shift(start, direction, distance), _sweep(start, direction, distance)


def _make_pair(rng, directions, height, width, background, colors):
    for _ in range(16):
        objects, swept = [], set()
        for direction, color in zip(directions, colors):
            for _ in range(300):
                start, end, path = _candidate(rng, direction, height, width)
                if _separate(path, swept):
                    objects.append((start, end, color))
                    swept.update(path)
                    break
            else:
                break
        if len(objects) == len(directions):
            inp = np.full((height, width), background, dtype=np.int8)
            out = np.full_like(inp, background)
            for start, end, color in objects:
                for r, c in start:
                    inp[r, c] = color
                for r, c in end:
                    out[r, c] = color
            return {"input": inp, "output": out}
    raise RuntimeError("Could not place separated T paths")


def _background(grid):
    return int(np.argmax(np.bincount(grid.astype(np.int64).ravel(), minlength=10)))


def _components(grid, background):
    remaining = {tuple(pos) for pos in np.argwhere(grid != background)}
    result = []
    while remaining:
        start = remaining.pop()
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in VEC.values():
                neighbor = (r + dr, c + dc)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        result.append(component)
    return result


def _parse_t_shapes(grid):
    result = []
    for component in _components(grid, _background(grid)):
        colors = {int(grid[r, c]) for r, c in component}
        assert len(colors) == 1
        junctions = []
        for r, c in component:
            neighbors = {d for d, (dr, dc) in VEC.items()
                         if (r + dr, c + dc) in component}
            if len(neighbors) == 3:
                junctions.append(((r, c), neighbors))
        assert len(junctions) == 1
        center, neighbors = junctions[0]
        directions = [d for d in VEC if neighbors == {d, *PERP[d]}]
        assert len(directions) == 1
        direction = directions[0]
        lengths = []
        for side in (*PERP[direction], direction):
            dr, dc = VEC[side]
            k = 0
            while (center[0] + (k + 1) * dr,
                   center[1] + (k + 1) * dc) in component:
                k += 1
            lengths.append(k)
        assert lengths[0] == lengths[1] >= 1 and lengths[2] >= 1
        arm, stem = lengths[0], lengths[2]
        assert component == _cells(direction, center, arm, stem)
        result.append((direction, center, arm, stem, colors.pop()))
    return result


def _expected_output(inp):
    height, width = inp.shape
    background = _background(inp)
    objects = _parse_t_shapes(inp)
    assert 1 <= len(objects) <= 3
    out = np.full_like(inp, background)
    swept = set()
    for direction, center, arm, stem, color in objects:
        distance = _distance(direction, center, stem, height, width)
        assert distance >= 1
        start = _cells(direction, center, arm, stem)
        path = _sweep(start, direction, distance)
        assert _separate(path, swept)
        swept.update(path)
        for r, c in _shift(start, direction, distance):
            assert 0 <= r < height and 0 <= c < width
            out[r, c] = color
    return out, objects


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _is_source_input(grid):
    return grid.shape == (10, 10) and "".join(map(str, grid.ravel())) in SOURCE_INPUTS


def _counts(rng, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= value <= high):
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if num_train is None:
        num_train = int(rng.integers(2 if num_test is None else max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _directions(rng, count, allowed):
    first = str(rng.choice(list(allowed)))
    if count == 1:
        return (first,)
    if count == 3:
        return (first, first, first)
    relation = int(rng.integers(0, 3))
    second = (first if relation == 0 else
              {"U": "D", "D": "U", "L": "R", "R": "L"}[first] if relation == 1 else
              str(rng.choice(list("LR" if first in "UD" else "UD"))))
    return first, second


def _dimension(rng, count):
    if count == 1:
        lo, hi = ((8, 13), (14, 21), (22, 30))[int(rng.integers(0, 3))]
    elif count == 2:
        lo, hi = ((15, 21), (22, 30))[int(rng.integers(0, 2))]
    else:
        lo, hi = 24, 30
    return int(rng.integers(lo, hi + 1))


def _plan(rng, num_train, num_test):
    mode = int(rng.integers(0, 4))
    if mode == 0:
        train_pool, test_pool, train = "RD", "UL", [("R",), ("D",)]
    elif mode == 1:
        train_pool, test_pool, train = "UL", "RD", [("U",), ("L",)]
    elif mode == 2:
        train_pool, test_pool, train = "UDLR", "UDLR", [("U", "R"), ("D", "L")]
    else:
        train_pool = test_pool = "UDLR"
        shuffled = list(rng.permutation(list("UDLR")))
        train = [(shuffled[0],), (shuffled[1],)]
    for _ in range(num_train - 2):
        train.append(_directions(rng, 2 if rng.random() < 0.48 else 1, train_pool))
    test = [_directions(rng, int(rng.choice((1, 2, 3), p=(0.36, 0.48, 0.16))),
                        test_pool) for _ in range(num_test)]
    if mode in (0, 1) and num_test == 1:
        test[0] = tuple(test_pool)
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    train_plan, test_plan = _plan(rng, num_train, num_test)
    base_background = 0 if rng.random() < 0.67 else int(rng.integers(1, 10))
    backgrounds = [base_background]
    if rng.random() < 0.35:
        backgrounds.append(int(rng.choice([x for x in range(10)
                                           if x != base_background])))
    palette = list(map(int, rng.permutation(10)))
    train_colors = set()
    puzzle, used = {"train": [], "test": []}, set()
    for regime, plan in (("train", train_plan), ("test", test_plan)):
        for pair_index, directions in enumerate(plan):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            background = (backgrounds[pair_index] if regime == "train" and
                          pair_index < len(backgrounds) else
                          int(pair_rng.choice(backgrounds)))
            options = [color for color in palette if color != background]
            if regime == "test" and train_colors and pair_rng.random() < 0.52:
                shared = [color for color in options if color in train_colors]
                if shared:
                    options = shared
            colors = [int(pair_rng.choice(options))]
            for _ in directions[1:]:
                other = [color for color in options if color != colors[0]]
                colors.append(int(pair_rng.choice(other))
                              if other and pair_rng.random() < 0.52 else colors[0])
            if regime == "train":
                train_colors.update(colors)
            for _ in range(12):
                if len(directions) == 1 and pair_rng.random() < 0.09:
                    height, width = 10, 10
                else:
                    height, width = (_dimension(pair_rng, len(directions)),
                                     _dimension(pair_rng, len(directions)))
                try:
                    pair = _make_pair(pair_rng, directions, height, width,
                                      background, colors)
                except RuntimeError:
                    continue
                key = _grid_key(pair["input"])
                if key not in used and not _is_source_input(pair["input"]):
                    used.add(key)
                    puzzle[regime].append(pair)
                    break
            else:
                raise RuntimeError("Could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    inputs, pairs, train_dirs, train_backgrounds, test_backgrounds = set(), set(), set(), set(), set()
    for regime, examples in (("train", train), ("test", test)):
        for pair in examples:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert not _is_source_input(inp)
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            assert input_key not in inputs and pair_key not in pairs
            inputs.add(input_key)
            pairs.add(pair_key)
            expected, objects = _expected_output(inp)
            assert np.array_equal(out, expected)
            assert _background(inp) == _background(out)
            if regime == "train":
                train_dirs.update(obj[0] for obj in objects)
                train_backgrounds.add(_background(inp))
            else:
                test_backgrounds.add(_background(inp))
    assert len(train_dirs) >= 2
    assert test_backgrounds <= train_backgrounds


def _same_puzzle(a, b):
    return all(len(a[regime]) == len(b[regime]) and
               all(np.array_equal(x[key], y[key])
                   for x, y in zip(a[regime], b[regime])
                   for key in ("input", "output"))
               for regime in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts, test_counts, shapes, motifs, relations = set(), set(), set(), set(), set()
    train_multi = test_multi = one_stem = nonzero_bg = varied_sizes = 0
    mixed_colors = shared_colors = 0
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_shapes, train_palette = set(), set()
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                inp = pair["input"]
                expected, objects = _expected_output(inp)
                assert np.array_equal(expected, pair["output"])
                puzzle_shapes.add(inp.shape)
                shapes.add(inp.shape)
                motifs.update((d, arm, stem) for d, _, arm, stem, _ in objects)
                one_stem += sum(stem == 1 for _, _, _, stem, _ in objects)
                nonzero_bg += _background(inp) != 0
                if len(objects) > 1:
                    train_multi += regime == "train"
                    test_multi += regime == "test"
                    mixed_colors += len({obj[4] for obj in objects}) > 1
                    if len(objects) == 2:
                        d1, d2 = objects[0][0], objects[1][0]
                        relations.add("same" if d1 == d2 else
                                      "opposite" if {d1, d2} in
                                      ({"U", "D"}, {"L", "R"}) else "perpendicular")
                if regime == "train":
                    train_palette.update(obj[4] for obj in objects)
                else:
                    shared_colors += any(obj[4] in train_palette for obj in objects)
        varied_sizes += len(puzzle_shapes) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 45 and len(motifs) >= 65
    assert train_multi >= 8 and test_multi >= 12
    assert one_stem >= 10 and nonzero_bg >= 10
    assert varied_sizes >= 35 and mixed_colors >= 8 and shared_colors >= 5
    assert relations == {"same", "opposite", "perpendicular"}
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                      (None, 1), (None, 4),
                                      (2, None), (10, None))):
        seed = 1000 + i
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        assert _same_puzzle(puzzle, generate_puzzle(seed, train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal pair counts accepted")
