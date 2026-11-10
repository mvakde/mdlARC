import numpy as np


_SOURCE_ROWS = (
    (("333301110", "300301010", "300301010", "333301110"), ("333", "300", "110")),
    (("777000000", "707708888", "700708008", "777708888"), ("777", "880", "000")),
    (("999905555", "900905005", "990905055", "099905550"), ("999", "555", "000")),
    (("888066600", "808060600", "888066600", "000000000"), ("800", "600", "000")),
    (("044466666", "040460006", "044460006", "000066666"), ("400", "666", "666")),
    (("777088800", "707080880", "707080080", "777088880"), ("770", "888", "000")),
    (("444000777", "404440707", "400040777", "444440000"), ("444", "400", "700")),
)


def _key(grid):
    return tuple(tuple(int(x) for x in row) for row in grid)


_SOURCE_KEYS = frozenset(
    (tuple(tuple(int(x) for x in row) for row in inp),
     tuple(tuple(int(x) for x in row) for row in out))
    for inp, out in _SOURCE_ROWS
)


def _neighbors(r, c, h, w):
    if r:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            cells = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in _neighbors(y, x, h, w):
                    if mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            result.append(cells)
    return result


def _holes(mask):
    h, w = mask.shape
    return [
        cells for cells in _components(~mask)
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)
    ]


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _make_shape_bank():
    bank = {1: (((0, 0),),)}
    for n in range(2, 7):
        shapes = set()
        for shape in bank[n - 1]:
            occupied = set(shape)
            for r, c in shape:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    new = (r + dr, c + dc)
                    if new not in occupied:
                        shapes.add(_normalize(occupied | {new}))
        bank[n] = tuple(sorted(shapes))
    return bank


_SHAPES = _make_shape_bank()
_VALID_AREAS = tuple(
    (left, right)
    for left in range(1, 7)
    for right in range(1, 7)
    if (left + 2) // 3 + (right + 2) // 3 <= 3
)


def _cavity_shape(rng, area, mode="any", compact=False):
    shapes = _SHAPES[area]
    if compact:
        small = tuple(s for s in shapes
                      if max(r for r, _ in s) < 2 and max(c for _, c in s) < 3)
        if small:
            shapes = small
    regular = [s for s in shapes if len({r for r, _ in s}) * len({c for _, c in s}) == area]
    irregular = [s for s in shapes if s not in regular]
    if mode == "regular" and regular:
        choices = regular
    elif mode == "irregular" and irregular:
        choices = irregular
    elif mode == "any" and regular and irregular and rng.random() < 0.25:
        choices = regular
    else:
        choices = irregular or regular
    return choices[int(rng.integers(len(choices)))]


def _contour(rng, area, mode="any", compact=False):
    shape = _cavity_shape(rng, area, mode, compact)
    cavity = {(r + 3, c + 3) for r, c in shape}
    h = max(r for r, _ in cavity) + 4
    w = max(c for _, c in cavity) + 4
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cavity:
        for y in range(r - 1, r + 2):
            for x in range(c - 1, c + 2):
                if (y, x) not in cavity:
                    mask[y, x] = True
    assert len(_components(mask)) == 1
    assert len(_holes(mask)) == 1 and set(_holes(mask)[0]) == cavity

    # Grow the exterior in several directions; the cavity is checked after
    # each addition, so a protrusion cannot make another closed pocket.
    for _ in range(0 if compact else int(rng.integers(0, 9))):
        candidates = set()
        for r, c in np.argwhere(mask):
            for y, x in _neighbors(int(r), int(c), h, w):
                if not mask[y, x] and (y, x) not in cavity:
                    candidates.add((y, x))
        if not candidates:
            break
        order = sorted(candidates)
        for index in rng.permutation(len(order)):
            y, x = order[int(index)]
            trial = mask.copy()
            trial[y, x] = True
            holes = _holes(trial)
            if len(holes) == 1 and set(holes[0]) == cavity:
                mask = trial
                break

    # The single-cell shell has expendable corners. Removing a few produces
    # open-looking outer silhouettes without opening the enclosed region.
    for _ in range(int(rng.integers(0, 2 if compact else 3))):
        edge = sorted((int(r), int(c)) for r, c in np.argwhere(mask))
        for index in rng.permutation(len(edge)):
            y, x = edge[int(index)]
            trial = mask.copy()
            trial[y, x] = False
            holes = _holes(trial)
            if len(_components(trial)) == 1 and len(holes) == 1 and set(holes[0]) == cavity:
                mask = trial
                break
    ys, xs = np.where(mask)
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    assert len(_components(mask)) == 1
    assert len(_holes(mask)) == 1 and len(_holes(mask)[0]) == area
    return mask


def _inspect(grid):
    values = [int(v) for v in np.unique(grid)]
    assert len(values) == 3
    valid = []
    h, w = grid.shape
    for bg in values:
        colors = [v for v in values if v != bg]
        if any(len(_components(grid == color)) != 1 for color in colors):
            continue
        bounds = []
        for color in colors:
            ys, xs = np.where(grid == color)
            bounds.append((int(xs.min()), int(xs.max()), color))
        bounds.sort()
        if bounds[0][1] >= bounds[1][0]:
            continue
        ordered = (bounds[0][2], bounds[1][2])
        cavities = {color: [] for color in ordered}
        exterior = 0
        for cells in _components(grid == bg):
            if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
                exterior += 1
                continue
            adjacent = set()
            for r, c in cells:
                for y, x in _neighbors(r, c, h, w):
                    if grid[y, x] != bg:
                        adjacent.add(int(grid[y, x]))
            if len(adjacent) != 1 or next(iter(adjacent)) not in cavities:
                break
            cavities[next(iter(adjacent))].append(cells)
        else:
            if exterior >= 1 and all(len(cavities[color]) == 1 for color in ordered):
                areas = tuple(len(cavities[color][0]) for color in ordered)
                if areas in _VALID_AREAS:
                    valid.append((bg, ordered, areas, cavities))
    assert len(valid) == 1
    return valid[0]


def _output(bg, colors, areas):
    result = np.full((3, 3), bg, dtype=np.int8)
    row = 0
    for color, area in zip(colors, areas):
        for i in range(area):
            result[row + i // 3, i % 3] = color
        row += (area + 2) // 3
    assert row <= 3
    return result


def _pair(rng, areas, bg, colors, modes, compact=False):
    for _ in range(32):
        left = _contour(rng, areas[0], modes[0], compact)
        right = _contour(rng, areas[1], modes[1], compact)
        lh, lw = left.shape
        rh, rw = right.shape
        gaps = [g for g in (0, 1, 2, 3, 4) if lw + rw + g <= 28]
        if not gaps:
            continue
        if compact:
            gap = 1 if 1 in gaps else 0
        else:
            weights = np.array([3 if g == 0 else 2 if g == 1 else 1 for g in gaps])
            gap = int(rng.choice(gaps, p=weights / weights.sum()))
        shifts = list(range(-rh + 1, lh))
        if gap == 0:
            left_edge = {r for r in range(lh) if left[r, lw - 1]}
            right_edge = {r for r in range(rh) if right[r, 0]}
            shifts = [d for d in shifts if any(r - d in right_edge for r in left_edge)]
        if not shifts:
            continue
        shift = min(shifts, key=lambda d: abs(d)) if compact else shifts[int(rng.integers(len(shifts)))]
        left_y = max(0, -shift)
        right_y = left_y + shift
        core_h = max(left_y + lh, right_y + rh)
        free_w = 30 - lw - rw - gap
        lm = 0 if compact else int(rng.integers(0, min(3, free_w) + 1))
        rm = 0 if compact else int(rng.integers(0, min(3, free_w - lm) + 1))
        tm = 0 if compact else int(rng.integers(0, 4))
        bm = 0 if compact else int(rng.integers(0, 4))
        if lm + rm + tm + bm == 0:
            if gap == 0:
                tm = 1
        grid = np.full((core_h + tm + bm, lw + rw + gap + lm + rm), bg, dtype=np.int8)
        lregion = grid[tm + left_y:tm + left_y + lh, lm:lm + lw]
        lregion[left] = colors[0]
        rcol = lm + lw + gap
        rregion = grid[tm + right_y:tm + right_y + rh, rcol:rcol + rw]
        rregion[right] = colors[1]
        try:
            found_bg, found_colors, found_areas, _ = _inspect(grid)
        except AssertionError:
            continue
        if (found_bg, found_colors, found_areas) == (bg, colors, areas):
            return grid, _output(bg, colors, areas)
    raise AssertionError("contour placement failed")


def _check_arg(name, value, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _counts(rng, num_train, num_test):
    _check_arg("num_train", num_train, 2, 10)
    _check_arg("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(rng, num_train, num_test)
    # The shared convention is always left-to-right, three-column unary
    # encoding. Colors are payload identities and can change in every pair.
    first = (4, int(rng.integers(1, 4)))
    second = (int(rng.integers(1, 4)), int(rng.integers(5, 7)))
    train_areas = [first, second]
    other = [a for a in _VALID_AREAS if a not in train_areas]
    rng.shuffle(other)
    train_areas += other[:ntrain - 2]
    if ntrain > 2:
        rng.shuffle(train_areas)
    test_choices = [a for a in _VALID_AREAS if a not in train_areas]
    rng.shuffle(test_choices)
    test_areas = test_choices[:ntest]
    jobs = [("train", a) for a in train_areas] + [("test", a) for a in test_areas]
    backgrounds = list(int(x) for x in rng.choice(10, size=min(3, len(jobs)), replace=False))
    backgrounds += [int(rng.integers(0, 10)) for _ in range(len(jobs) - len(backgrounds))]
    rng.shuffle(backgrounds)
    puzzle = {"train": [], "test": []}
    used = set()
    for index, (section, areas) in enumerate(jobs):
        bg = backgrounds[index]
        colors = tuple(int(x) for x in rng.choice([x for x in range(10) if x != bg], size=2, replace=False))
        modes = ["any", "any"]
        if section == "train" and areas == first:
            modes[0] = "regular" if rng.random() < 0.35 else "irregular"
        if section == "train" and areas == second:
            modes[1] = "irregular"
        compact = bool(rng.random() < 0.32)
        if section == "train" and areas == second:
            compact = False
        for _ in range(48):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            inp, out = _pair(pair_rng, areas, bg, colors, modes, compact)
            key = (_key(inp), _key(out))
            if key not in used and key not in _SOURCE_KEYS:
                puzzle[section].append({"input": inp, "output": out})
                used.add(key)
                break
        else:
            raise AssertionError("unique pair construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def _irregular(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1) != len(cells)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen = set()
    metadata = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert out.shape == (3, 3) and not np.array_equal(inp, out)
            key = (_key(inp), _key(out))
            assert key not in seen and key not in _SOURCE_KEYS
            seen.add(key)
            bg, colors, areas, cavities = _inspect(inp)
            assert np.array_equal(out, _output(bg, colors, areas))
            metadata[section].append((bg, colors, areas, cavities))
    all_inputs = [_key(p["input"]) for sec in ("train", "test") for p in puzzle[sec]]
    assert len(set(all_inputs)) == len(all_inputs)
    train_areas = [m[2] for m in metadata["train"]]
    test_areas = [m[2] for m in metadata["test"]]
    assert any(a[0] == 4 and 1 <= a[1] <= 3 for a in train_areas)
    assert any(1 <= a[0] <= 3 and 5 <= a[1] <= 6 for a in train_areas)
    assert any(_irregular(m[3][color][0]) for m in metadata["train"] for color in m[1])
    assert len(set(train_areas)) == nt
    assert len(set(test_areas)) == nq
    assert not (set(train_areas) & set(test_areas))
    assert len({m[0] for sec in metadata.values() for m in sec}) >= 2


def _puzzle_key(puzzle):
    return tuple((section, tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[section]))
                 for section in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    inputs = [p["input"] for task in puzzles for sec in ("train", "test") for p in task[sec]]
    observations = [_inspect(grid) for grid in inputs]
    assert len({grid.shape for grid in inputs}) >= 12
    assert len({o[2] for o in observations}) >= 20
    assert len({o[0] for o in observations}) >= 7
    assert any(o[0] != 0 for o in observations)
    assert any(o[0] == 0 for o in observations)
    assert any(o[2][0] == 6 for o in observations)
    assert any(o[2][1] in (5, 6) for o in observations)
    assert any(o[2][0] == 5 or o[2][1] == 5 for o in observations)
    assert any(not _irregular(o[3][color][0]) and o[2][i] == 4
               for o in observations for i, color in enumerate(o[1]))
    assert any(_irregular(o[3][color][0]) and o[2][i] == 4
               for o in observations for i, color in enumerate(o[1]))
    assert any(grid.shape != (4, 9) for grid in inputs)
    assert any(any(grid[r, c] != o[0] and grid[r, c + 1] != o[0] and
                   grid[r, c] != grid[r, c + 1]
                   for r in range(grid.shape[0]) for c in range(grid.shape[1] - 1))
               for grid, o in zip(inputs, observations))
    gaps = set()
    cavity_shapes = {area: set() for area in range(1, 7)}
    foreground_sizes = {area: set() for area in range(1, 7)}
    roles = {color: set() for color in range(10)}
    source_size = combined = 0
    for grid, (bg, colors, areas, cavities) in zip(inputs, observations):
        roles[bg].add("background")
        roles[colors[0]].add("left")
        roles[colors[1]].add("right")
        gap = int(np.where(grid == colors[1])[1].min()
                  - np.where(grid == colors[0])[1].max() - 1)
        gaps.add(gap)
        source_size += grid.shape == (4, 9)
        for color, area in zip(colors, areas):
            cells = cavities[color][0]
            y0 = min(r for r, _ in cells)
            x0 = min(c for _, c in cells)
            cavity_shapes[area].add(tuple(sorted((r - y0, c - x0) for r, c in cells)))
            foreground_sizes[area].add(int(np.count_nonzero(grid == color)))
        if (gap == 0 and grid.shape != (4, 9) and bg != 0
                and any(_irregular(cavities[color][0]) for color in colors)):
            combined += 1
    assert gaps == {0, 1, 2, 3, 4}
    assert source_size >= 10 and combined >= 10
    assert len(cavity_shapes[4]) >= 10
    assert len(cavity_shapes[5]) >= 20 and len(cavity_shapes[6]) >= 20
    assert len(foreground_sizes[4]) >= 6
    assert all(len(roles[color]) == 3 for color in roles)
    for nt, nq in ((2, 1), (10, 1), (10, 4), (6, 2)):
        task = generate_puzzle(1000 + nt * 10 + nq, nt, nq)
        assert (len(task["train"]), len(task["test"])) == (nt, nq)
        validate_generated_puzzle(task)
    for nt, nq in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(2000 + (nt or 0) * 10 + (nq or 0), nt, nq))
    assert _puzzle_key(generate_puzzle(314159, 8, 3)) == _puzzle_key(generate_puzzle(314159, 8, 3))
    assert len({_puzzle_key(generate_puzzle(s, 6, 2)) for s in range(3000, 3010)}) == 10
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(4000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
