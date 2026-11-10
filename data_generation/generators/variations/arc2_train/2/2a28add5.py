import numpy as np

_SOURCE_INPUTS = {
    ("7747670777","7513777791","7237277777","7897870728","5971756878","7270777176","7777777774","7787740775","6777127704","7777721767"),
    ("7777777577","7796772777","7777777297","2799767577","7777777777","7777727777","7596777779","7777777777","7277797777","7227777675"),
    ("0574751377","7774465277","4628778703","1775376377","0977737777","7930705537","7171755876","7773327092","7707607717","1747127717"),
}

def _checked(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} outside {low}..{high}")
    return value

def _counts(seed, train, test):
    train = None if train is None else _checked(train, "num_train", 2, 10)
    test = None if test is None else _checked(test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test

def _mode(grid):
    return int(np.bincount(grid.ravel(), minlength=10).argmax())

def _solve(grid, background, anchor, paint, vertical=False):
    source = grid.T if vertical else grid
    output = np.full(source.shape, background, dtype=int)
    for r, line in enumerate(source):
        marks = np.flatnonzero(line == anchor)
        if len(marks) > 1:
            raise ValueError("multiple anchors on a line")
        if len(marks):
            c = int(marks[0])
            left = int(np.count_nonzero(line[:c] != background))
            right = int(np.count_nonzero(line[c+1:] != background))
            output[r, c-left:c+right+1] = paint
    return output.T.copy() if vertical else output

def _dimensions(rng):
    family = int(rng.integers(5))
    if family == 0:
        return int(rng.integers(5, 10)), int(rng.integers(5, 11))
    if family == 1:
        return int(rng.integers(5, 11)), int(rng.integers(14, 31))
    if family == 2:
        return int(rng.integers(14, 31)), int(rng.integers(5, 11))
    if family == 3:
        return int(rng.integers(10, 21)), int(rng.integers(10, 21))
    return int(rng.integers(20, 31)), int(rng.integers(18, 31))

def _line_counts(rng, width):
    kind = int(rng.choice(6, p=[.08,.10,.44,.20,.12,.06]))
    if kind == 0:
        length = 1
    elif kind == 1:
        length = 2
    elif kind == 2:
        length = int(rng.integers(3, min(width, 7)+1))
    elif kind == 3:
        length = int(rng.integers(3, width+1))
    elif kind == 4:
        length = int(rng.integers(max(2, width//2), width+1))
    else:
        length = width
    placement = int(rng.integers(5))
    if placement == 0:
        column = int(rng.choice([0, width-1]))
    elif placement == 1 and length < width-1 and width >= 5:
        column = int(rng.integers(1, width-1))
        length = min(length, max(column, width-1-column)+1)
    else:
        column = int(rng.integers(length-1, width))
    lo = max(0, length-1-(width-1-column))
    hi = min(length-1, column)
    if placement == 1 and lo == 0:
        left = 0
    elif placement == 1 and hi == length-1:
        left = hi
    else:
        left = int(rng.integers(lo, hi+1))
    return column, left, length-1-left

def _positions(rng, capacity, count, style):
    if not count:
        return np.empty(0, dtype=int)
    if style == 3:
        return np.arange(1, count+1)
    if style == 1:
        start = int(rng.integers(1, capacity-count+2))
        return np.arange(start, start+count)
    if style == 2 and count > 1:
        near = int(rng.integers(1, count))
        far = rng.choice(np.arange(near+1, capacity+1), size=count-near, replace=False)
        return np.concatenate((np.arange(1, near+1), far))
    return rng.choice(np.arange(1, capacity+1), size=count, replace=False)

def _pair(rng, anchor, paint, vertical, background, first_train, test):
    height, width = _dimensions(rng)
    available = [c for c in range(10) if c not in (background, anchor)]
    include_paint = bool(rng.integers(2))
    size = int(rng.integers(3 if first_train else 1, 8 + int(include_paint)))
    other = [c for c in available if c != paint]
    palette = rng.choice(other, size=size-int(include_paint), replace=False).tolist()
    if include_paint:
        palette.append(paint)
    if first_train:
        low = min(3, height)
    elif test:
        low = 0
    else:
        low = 1
    count = int(rng.integers(low, height+1))
    active = set(int(x) for x in rng.choice(height, size=count, replace=False))
    grid = np.full((height, width), background, dtype=int)
    for row in active:
        col, left, right = _line_counts(rng, width)
        style = int(rng.integers(4))
        positions = [col-int(d) for d in _positions(rng, col, left, style)]
        positions += [col+int(d) for d in _positions(rng, width-col-1, right, style)]
        if positions:
            grid[row, positions] = rng.choice(palette, size=len(positions), replace=True)
        grid[row, col] = anchor
    for row in range(height):
        if row in active:
            continue
        density = int(rng.choice(4, p=[.22,.38,.25,.15]))
        if density == 0:
            tokens = 0
        elif density == 1:
            tokens = int(rng.integers(1, min(4, width)+1))
        elif density == 2:
            tokens = int(rng.integers(1, max(2, width//2+1)))
        else:
            tokens = int(rng.integers(max(1, width//2), width))
        if tokens:
            cols = rng.choice(width, size=tokens, replace=False)
            grid[row, cols] = rng.choice(palette, size=tokens, replace=True)
    totals = np.bincount(grid.ravel(), minlength=10)
    if totals[background] <= max(int(totals[c]) for c in range(10) if c != background):
        return None
    if vertical:
        grid = grid.T.copy()
    output = _solve(grid, background, anchor, paint, vertical)
    if np.array_equal(grid, output):
        return None
    return {"input": grid, "output": output}

def _signature(grid):
    return grid.shape, grid.tobytes()

def _roles(train):
    backgrounds = [_mode(p["input"]) for p in train]
    candidates = []
    for anchor in range(10):
        if not any(np.any(p["input"] == anchor) for p in train):
            continue
        for paint in range(10):
            if anchor == paint or any(b in (anchor, paint) for b in backgrounds):
                continue
            for vertical in (False, True):
                try:
                    fits = all(np.array_equal(_solve(p["input"], b, anchor, paint, vertical), p["output"])
                               for p, b in zip(train, backgrounds))
                except ValueError:
                    fits = False
                if fits:
                    candidates.append((anchor, paint, vertical))
    return candidates

def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = [int(c) for c in rng.permutation(10)]
    anchor, paint = colors[:2]
    vertical = bool(rng.integers(2))
    varied_background = bool(rng.random() < .35)
    backgrounds = colors[2:2+min(3, train_count)] if varied_background else colors[2:3]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, n in (("train", train_count), ("test", test_count)):
        for index in range(n):
            for _ in range(120):
                if split == "train" and index < len(backgrounds):
                    background = backgrounds[index]
                else:
                    background = int(rng.choice(backgrounds))
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63-1)))
                pair = _pair(pair_rng, anchor, paint, vertical, background,
                             split == "train" and index == 0, split == "test")
                if pair is None:
                    continue
                signature = _signature(pair["input"])
                strings = tuple("".join(map(str, row)) for row in pair["input"])
                if signature in seen or strings in _SOURCE_INPUTS:
                    continue
                if split == "train" and index == 0 and len(_roles([pair])) != 1:
                    continue
                if split == "train" and index == 1 and pair["input"].shape == puzzle["train"][0]["input"].shape:
                    continue
                seen.add(signature)
                puzzle[split].append(pair)
                break
            else:
                raise ValueError("could not generate a distinct learnable pair")
    validate_generated_puzzle(puzzle)
    return puzzle

def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in (pair["input"], pair["output"]):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert all(1 <= d <= 30 for d in grid.shape)
            assert np.all((0 <= grid) & (grid <= 9))
        source, target = pair["input"], pair["output"]
        assert source.shape == target.shape and not np.array_equal(source, target)
        signature = _signature(source)
        assert signature not in seen
        seen.add(signature)
        assert tuple("".join(map(str, row)) for row in source) not in _SOURCE_INPUTS
    assert len({p["input"].shape for p in puzzle["train"]}) > 1
    roles = _roles(puzzle["train"])
    assert len(roles) == 1, f"ambiguous roles: {roles}"
    anchor, paint, vertical = roles[0]
    for pair in puzzle["train"] + puzzle["test"]:
        grid = pair["input"]
        background = _mode(grid)
        assert background not in (anchor, paint)
        totals = np.bincount(grid.ravel(), minlength=10)
        assert totals[background] > max(int(totals[c]) for c in range(10) if c != background)
        lines = grid.T if vertical else grid
        assert np.all(np.count_nonzero(lines == anchor, axis=1) <= 1)
        expected = _solve(grid, background, anchor, paint, vertical)
        assert np.array_equal(expected, pair["output"])
        assert set(int(x) for x in np.unique(pair["output"])) <= {background, paint}

def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))

def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, structures, roles, axes, lengths = set(), set(), set(), set(), set()
    blank = dense = one_sided = full = 0
    paint_absent = 0
    varied_backgrounds = 0
    active_counts = {"train": set(), "test": set()}
    palette_sizes = {"train": set(), "test": set()}
    ordinal_counts = {0: set(), 1: set(), 2: set()}
    balanced = left_heavy = right_heavy = 0
    no_anchor_tests = all_active = triples = isolated = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        anchor, paint, vertical = _roles(puzzle["train"])[0]
        roles.add((anchor, paint))
        axes.add(vertical)
        varied_backgrounds += len({_mode(p["input"]) for p in puzzle["train"] + puzzle["test"]}) > 1
        for split, pair in [(s, p) for s in ("train", "test") for p in puzzle[s]]:
            grid = pair["input"].T if vertical else pair["input"]
            output = pair["output"].T if vertical else pair["output"]
            background = _mode(grid)
            shapes.add(pair["input"].shape)
            active = tuple(int(i) for i in np.flatnonzero(np.any(grid == anchor, axis=1)))
            structures.add((grid.shape, active))
            active_counts[split].add(len(active))
            palette_sizes[split].add(len(np.unique(grid)))
            paint_absent += not np.any(grid == paint)
            no_anchor_tests += split == "test" and not active
            all_active += len(active) == len(grid)
            triples += any(active[i+2] == active[i] + 2 for i in range(len(active)-2))
            isolated += bool(active) and all(abs(x-y) > 1 for x, y in zip(active, active[1:]))
            ordinal = 0
            for line, out in zip(grid, output):
                blank += int(np.all(line == background))
                dense += int(np.count_nonzero(line != background) >= len(line)//2)
                mark = np.flatnonzero(line == anchor)
                if len(mark):
                    col = int(mark[0])
                    left = int(np.count_nonzero(line[:col] != background))
                    right = int(np.count_nonzero(line[col+1:] != background))
                    if split == "train" and ordinal < 3:
                        ordinal_counts[ordinal].add((left, right))
                    ordinal += 1
                    balanced += left == right
                    left_heavy += left > right
                    right_heavy += right > left
                    one_sided += int((left == 0) != (right == 0))
                    size = int(np.count_nonzero(out == paint))
                    lengths.add(size)
                    full += int(size == len(line))
                    assert size == left + right + 1
    assert len(shapes) >= 25 and len(structures) >= 100
    assert len(roles) >= 20 and axes == {False, True}
    assert {1, 2}.issubset(lengths) and max(lengths) >= 20
    assert blank and dense and one_sided and full
    assert paint_absent and varied_backgrounds and no_anchor_tests and all_active
    assert triples and isolated and balanced and left_heavy and right_heavy
    assert all(len(values) >= 8 for values in ordinal_counts.values())
    assert len(active_counts["train"]) >= 10 and len(active_counts["test"]) >= 8
    assert len(palette_sizes["train"]) >= 5 and len(palette_sizes["test"]) >= 5
    for train, test in ((2,1),(10,1),(10,4),(6,3),(None,1),(None,4),(2,None),(10,None)):
        seed = 314159 + (train or 0) + (test or 0)
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert _equal(puzzle, generate_puzzle(seed, train, test))
    assert not _equal(generate_puzzle(314159), generate_puzzle(314160))
    for train, test in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,3),(2,4),(4,4)):
        try:
            generate_puzzle(123, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
