import numpy as np


# Original inputs, with literal colors. Output is fixed by the mosaic rule.
_SOURCE_INPUTS = (
    ((8, 0, 8), (8, 0, 0), (8, 0, 0)),
    ((3, 0, 3), (0, 3, 3), (3, 3, 3)),
    ((3, 3, 3), (0, 0, 3), (3, 0, 0)),
    ((0, 7, 7), (0, 0, 0), (7, 7, 0)),
    ((9, 9, 9), (0, 0, 0), (9, 9, 0)),
    ((6, 6, 0), (6, 6, 0), (0, 0, 6)),
)
_COUNT_PAIRS = tuple(
    (a, b) for a in range(2, 11) for b in range(1, 5) if a > b
)
_STYLES = ("sparse", "dense", "noise", "clusters", "frame", "bars", "diagonal", "checker")
_SYMMETRIES = ("chiral", "mirror", "half", "quarter", "free")


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _key(grid):
    return (tuple(grid.shape), tuple(int(v) for v in grid.flat))


def _input_tuple(grid):
    return tuple(tuple(int(v) for v in row) for row in grid)


def _dihedral_views(grid):
    reflected = np.fliplr(grid)
    return tuple(_key(np.rot90(grid, k)) for k in range(4)) + tuple(
        _key(np.rot90(reflected, k)) for k in range(4)
    )


def _symmetry_class(grid):
    rotations = {_key(np.rot90(grid, k)) for k in range(4)}
    if len(rotations) == 1:
        return "quarter"
    if len(rotations) == 2:
        return "half"
    if len(set(_dihedral_views(grid))) < 8:
        return "mirror"
    return "chiral"


def _pinwheel(grid):
    return np.block([
        [grid, np.rot90(grid, 1)],
        [np.rot90(grid, 2), np.rot90(grid, -1)],
    ]).astype(np.int8, copy=False)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    a = _count("num_train", num_train, 2, 10)
    b = _count("num_test", num_test, 1, 4)
    if a is None and b is None:
        a, b = _COUNT_PAIRS[(seed * 17 + 11) % len(_COUNT_PAIRS)]
    elif a is None:
        a = int(rng.integers(max(2, b + 1), 11))
    elif b is None:
        b = int(rng.integers(1, min(4, a - 1) + 1))
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _raw_mask(rng, n, style):
    if style in ("sparse", "dense", "noise"):
        low, high = {"sparse": (.09, .29), "dense": (.70, .91), "noise": (.30, .68)}[style]
        return rng.random((n, n)) < rng.uniform(low, high)
    mask = np.zeros((n, n), dtype=bool)
    if style == "clusters":
        for _ in range(int(rng.integers(1, 4))):
            r0, r1 = sorted(int(v) for v in rng.integers(0, n + 1, size=2))
            c0, c1 = sorted(int(v) for v in rng.integers(0, n + 1, size=2))
            mask[r0:min(n, max(r0 + 1, r1)), c0:min(n, max(c0 + 1, c1))] = True
        mask &= rng.random((n, n)) < rng.uniform(.72, .98)
    elif style == "frame":
        top = int(rng.integers(0, max(1, n // 3 + 1)))
        left = int(rng.integers(0, max(1, n // 3 + 1)))
        bottom = int(rng.integers(max(top + 2, n - n // 3), n + 1))
        right = int(rng.integers(max(left + 2, n - n // 3), n + 1))
        mask[top:bottom, left] = True
        mask[top:bottom, right - 1] = True
        mask[top, left:right] = True
        mask[bottom - 1, left:right] = True
        mask |= (rng.random((n, n)) < .06)
    elif style == "bars":
        for _ in range(int(rng.integers(1, 4))):
            row = int(rng.integers(0, n))
            col = int(rng.integers(0, n))
            length = int(rng.integers(2, n + 1))
            if bool(rng.integers(0, 2)):
                mask[row, col:min(n, col + length)] = True
            else:
                mask[row:min(n, row + length), col] = True
        mask |= rng.random((n, n)) < .04
    elif style == "diagonal":
        slope = 1 if bool(rng.integers(0, 2)) else -1
        offset = int(rng.integers(0, n))
        for r in range(n):
            mask[r, (slope * r + offset) % n] = True
        mask |= rng.random((n, n)) < .10
    else:
        phase = int(rng.integers(0, 2))
        rows, cols = np.indices((n, n))
        mask = ((rows + cols + phase) % 2 == 0)
        mask &= rng.random((n, n)) < .92
    return mask


def _orbit(r, c, n, symmetry):
    if symmetry == "mirror":
        return {(r, c), (r, n - 1 - c)}
    if symmetry == "half":
        return {(r, c), (n - 1 - r, n - 1 - c)}
    if symmetry == "quarter":
        return {(r, c), (n - 1 - c, r), (n - 1 - r, n - 1 - c), (c, n - 1 - r)}
    return {(r, c)}


def _orbits(n, symmetry):
    seen = set()
    result = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit = _orbit(r, c, n, symmetry)
            seen.update(orbit)
            result.append(sorted(orbit))
    return result


def _grid(rng, n, palette, symmetry, style):
    base = palette[0]
    accents = palette[1:]
    orbits = _orbits(n, symmetry)
    for _ in range(96):
        raw = _raw_mask(rng, n, style)
        occupied = [i for i, orbit in enumerate(orbits) if raw[orbit[0]]]
        if len(occupied) < len(accents) or len(occupied) == len(orbits):
            continue
        grid = np.full((n, n), base, dtype=np.int8)
        rng.shuffle(occupied)
        choices = list(accents)
        rng.shuffle(choices)
        assignments = {index: color for index, color in zip(occupied, choices)}
        for index in occupied[len(choices):]:
            assignments[index] = int(rng.choice(accents))
        for index, color in assignments.items():
            for r, c in orbits[index]:
                grid[r, c] = color
        actual = _symmetry_class(grid)
        if symmetry == "free" or actual == symmetry:
            return grid
    # A deterministic, nonempty fallback keeps legal seeds bounded.
    for _ in range(256):
        raw = rng.random((n, n)) < .43
        occupied = [i for i, orbit in enumerate(orbits) if raw[orbit[0]]]
        if len(occupied) < len(accents) or len(occupied) == len(orbits):
            continue
        grid = np.full((n, n), base, dtype=np.int8)
        rng.shuffle(occupied)
        for j, index in enumerate(occupied):
            color = accents[j % len(accents)]
            for r, c in orbits[index]:
                grid[r, c] = color
        if symmetry == "free" or _symmetry_class(grid) == symmetry:
            return grid
    raise RuntimeError("unable to construct requested motif")


def _palette(rng, pool, count, force_nonzero_base=False, novel=None):
    choices = list(pool)
    if force_nonzero_base:
        bases = [x for x in choices if x != 0]
    elif 0 in choices and rng.random() < .55:
        bases = [0]
    else:
        bases = choices
    base = int(rng.choice(bases))
    remaining = [x for x in choices if x != base and x != novel]
    rng.shuffle(remaining)
    accents = remaining[:count - 1 - int(novel is not None and novel != base)]
    if novel is not None and novel != base:
        accents.append(int(novel))
    return (base, *accents)


def _draw_pair(rng, size, palette, symmetry, used):
    for attempt in range(96):
        style = _STYLES[int(rng.integers(0, len(_STYLES)))]
        grid = _grid(rng, size, palette, symmetry, style)
        key = _key(grid)
        if key not in used and _input_tuple(grid) not in _SOURCE_INPUTS:
            used.add(key)
            return {"input": grid, "output": _pinwheel(grid)}
    raise RuntimeError("unable to draw a distinct input")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)

    # The only learned convention is the invariant rotation order. This pool
    # controls sampling breadth; its membership never changes the rule.
    color_pool = (0, *[int(v) for v in rng.choice(np.arange(1, 10), size=6, replace=False)])
    fresh_colors = [v for v in range(1, 10) if v not in color_pool]
    used = set()

    small = 3 if rng.random() < .46 else int(rng.integers(3, 7))
    large = int(rng.integers(6, 16))
    if large == small:
        large = 15 if small != 15 else 6
    train_sizes = [small, large]
    train_sizes += [int(rng.integers(3, 16)) for _ in range(num_train - 2)]
    rng.shuffle(train_sizes)
    revealing_index = int(rng.integers(0, num_train))
    rich_index = (revealing_index + int(rng.integers(1, num_train))) % num_train
    train = []
    for i, size in enumerate(train_sizes):
        if i == revealing_index:
            symmetry = "chiral"
            color_count = 2
        else:
            symmetry = str(rng.choice(_SYMMETRIES, p=(.30, .20, .15, .12, .23)))
            color_count = int(rng.integers(2, min(5, len(_orbits(size, symmetry))) + 1))
        if i == rich_index:
            color_count = int(rng.integers(3, min(5, len(_orbits(size, symmetry))) + 1))
        palette = _palette(rng, color_pool, color_count, force_nonzero_base=(i == rich_index))
        train.append(_draw_pair(rng, size, palette, symmetry, used))

    test = []
    repeat_geometry = seed % 4 == 0
    for i in range(num_test):
        if repeat_geometry and i == 0:
            source = train[int(rng.integers(0, num_train))]["input"]
            values = [int(v) for v in np.unique(source)]
            novel = int(rng.choice(fresh_colors)) if rng.random() < .5 else None
            palette = _palette(rng, color_pool, len(values), novel=novel)
            mapping = dict(zip(values, palette))
            grid = np.rot90(source, int(rng.integers(0, 4))).copy()
            for old, new in mapping.items():
                grid[grid == old] = -old - 1
            for old, new in mapping.items():
                grid[grid == -old - 1] = new
            if _key(grid) not in used and _input_tuple(grid) not in _SOURCE_INPUTS:
                used.add(_key(grid))
                test.append({"input": grid, "output": _pinwheel(grid)})
                continue
        size = int(rng.choice(train_sizes)) if rng.random() < .48 else int(rng.integers(3, 16))
        symmetry = str(rng.choice(_SYMMETRIES, p=(.29, .21, .16, .13, .21)))
        reuse = rng.random() < .28
        if reuse:
            source = train[int(rng.integers(0, num_train))]["input"]
            values, counts = np.unique(source, return_counts=True)
            base = int(values[int(np.argmax(counts))])
            palette = (base, *[int(v) for v in values if int(v) != base])
        else:
            color_count = int(rng.integers(2, min(5, len(_orbits(size, symmetry))) + 1))
            novel = int(rng.choice(fresh_colors)) if rng.random() < .48 else None
            palette = _palette(rng, color_pool, color_count, novel=novel)
        if len(palette) > len(_orbits(size, symmetry)):
            size = max(5, size)
        test.append(_draw_pair(rng, size, palette, symmetry, used))

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "pair counts")
    used = set()
    train_sizes = set()
    train_has_chiral = False
    train_has_multicolor = False
    train_has_binary = False
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, f"{label} keys")
            for role in ("input", "output"):
                grid = pair[role]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, f"{label} {role} array")
                _require(np.issubdtype(grid.dtype, np.integer), f"{label} {role} dtype")
                _require(all(1 <= d <= 30 for d in grid.shape), f"{label} {role} shape")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), f"{label} {role} color")
            inp, out = pair["input"], pair["output"]
            n, m = inp.shape
            _require(n == m and 3 <= n <= 15, f"{label} square input")
            _require(out.shape == (2 * n, 2 * n), f"{label} doubled output")
            _require(len(np.unique(inp)) >= 2, f"{label} needs a pattern")
            _require(np.array_equal(out[:n, :n], inp), f"{label} original quadrant")
            _require(np.array_equal(out[:n, n:], np.rot90(inp, 1)), f"{label} CCW quadrant")
            _require(np.array_equal(out[n:, :n], np.rot90(inp, 2)), f"{label} half-turn quadrant")
            _require(np.array_equal(out[n:, n:], np.rot90(inp, -1)), f"{label} CW quadrant")
            key = _key(inp)
            _require(key not in used, f"{label} duplicate input")
            _require(_input_tuple(inp) not in _SOURCE_INPUTS, f"{label} source copy")
            used.add(key)
            if split == "train":
                train_sizes.add(n)
                train_has_chiral |= _symmetry_class(inp) == "chiral"
                train_has_multicolor |= len(np.unique(inp)) >= 3
                train_has_binary |= len(np.unique(inp)) == 2
    _require(len(train_sizes) >= 2, "training needs two sizes")
    _require(train_has_chiral, "training needs orientation evidence")
    _require(train_has_multicolor, "training needs multicolor evidence")
    _require(train_has_binary, "training needs a simple binary example")


def _equal_puzzles(first, second):
    return all(
        len(first[split]) == len(second[split]) and all(
            np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"])
            for x, y in zip(first[split], second[split])
        ) for split in ("train", "test")
    )


def self_test() -> None:
    first = generate_puzzle(123456)
    _require(_equal_puzzles(first, generate_puzzle(123456)), "automatic determinism")
    explicit = generate_puzzle(654321, 7, 3)
    _require(_equal_puzzles(explicit, generate_puzzle(654321, 7, 3)), "explicit determinism")
    for index, (a, b) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(7000 + index, a, b)
        validate_generated_puzzle(puzzle)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "explicit counts")
    for index, (a, b) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(8000 + index, a, b)
        validate_generated_puzzle(puzzle)
        _require(a is None or len(puzzle["train"]) == a, "partial train count")
        _require(b is None or len(puzzle["test"]) == b, "partial test count")
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 3), (3, 4)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _require(not _equal_puzzles(puzzles[0], puzzles[1]), "cross-seed difference")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    inputs = [pair["input"] for p in puzzles for split in ("train", "test") for pair in p[split]]
    tests = [(p, pair) for p in puzzles for pair in p["test"]]
    _require(len({_key(g) for g in inputs}) >= 250, "colored-grid diversity")
    _require(len({_key((g != int(np.unique(g, return_counts=True)[0][np.argmax(np.unique(g, return_counts=True)[1])])).astype(np.int8)) for g in inputs}) >= 170, "geometry diversity")
    _require(len({g.shape[0] for g in inputs}) >= 10, "size diversity")
    _require({2, 3, 4, 5}.issubset({len(np.unique(g)) for g in inputs}), "color-count diversity")
    _require(all(any(len(np.unique(t["input"])) < 3 for t in p["train"]) and any(len(np.unique(t["input"])) >= 3 for t in p["train"]) for p in puzzles), "training palette coverage")
    _require({"chiral", "mirror", "half", "quarter"}.issubset({_symmetry_class(g) for g in inputs}), "symmetry diversity")
    _require(any(len(np.unique(pair["input"])) >= 3 for _, pair in tests), "multicolor tests")
    _require(any(np.array_equal(pair["input"], np.rot90(pair["input"], 1)) for _, pair in tests), "symmetric tests")
    _require(any(0 not in np.unique(g) for g in inputs), "nonzero fields")
    _require(any(len({tuple(np.unique(t["input"])) for t in p["test"]}) > 1 for p in puzzles if len(p["test"]) > 1), "test palette diversity")
    _require(any(set(np.unique(t["input"])) <= set(int(v) for q in p["train"] for v in np.unique(q["input"])) for p, t in tests), "reused test colors")
    _require(any(not set(np.unique(t["input"])) <= set(int(v) for q in p["train"] for v in np.unique(q["input"])) for p, t in tests), "novel test colors")

    def normalized_orbit(grid):
        def normalize(view):
            labels = {}
            values = []
            for value in view.flat:
                value = int(value)
                if value not in labels:
                    labels[value] = len(labels)
                values.append(labels[value])
            return (view.shape, tuple(values))
        return min(normalize(np.rot90(grid, k)) for k in range(4))

    def components(mask):
        remaining = set(zip(*np.nonzero(mask)))
        count = 0
        while remaining:
            todo = [remaining.pop()]
            count += 1
            while todo:
                r, c = todo.pop()
                for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        todo.append(neighbor)
        return count

    size_reuse = size_change = palette_reuse = palette_novel = 0
    simpler_test = symmetric_test = repeated_geometry = 0
    multi_test_palettes = 0
    for puzzle in puzzles:
        train = puzzle["train"]
        test = puzzle["test"]
        train_sizes = {len(pair["input"]) for pair in train}
        train_colors = {int(v) for pair in train for v in np.unique(pair["input"])}
        all_grids = [pair["input"] for pair in train + test]
        size_reuse += any(len(pair["input"]) in train_sizes for pair in test)
        size_change += any(len(pair["input"]) not in train_sizes for pair in test)
        palette_reuse += any(set(np.unique(pair["input"])) <= train_colors for pair in test)
        palette_novel += any(not set(np.unique(pair["input"])) <= train_colors for pair in test)
        simpler_test += any(len(np.unique(pair["input"])) < max(len(np.unique(q["input"])) for q in train) for pair in test)
        symmetric_test += any(_symmetry_class(pair["input"]) != "chiral" for pair in test)
        multi_test_palettes += len({tuple(np.unique(pair["input"])) for pair in test}) > 1
        repeated_geometry += any(
            normalized_orbit(grid) == normalized_orbit(prior)
            for i, grid in enumerate(all_grids) for prior in all_grids[:i]
        )

        # Infer four placements from demonstrations alone, then use only the
        # test input to predict each held-out output.
        possible = [set(range(8)) for _ in range(4)]
        for pair in train:
            inp, out = pair["input"], pair["output"]
            n = len(inp)
            views = [np.rot90(inp, k) for k in range(4)] + [
                np.rot90(np.fliplr(inp), k) for k in range(4)
            ]
            quadrants = (out[:n, :n], out[:n, n:], out[n:, :n], out[n:, n:])
            for q, quadrant in enumerate(quadrants):
                possible[q] &= {k for k, view in enumerate(views) if np.array_equal(view, quadrant)}
        _require(all(len(options) == 1 for options in possible), "ambiguous train convention")
        for pair in test:
            inp = pair["input"].copy()
            views = [np.rot90(inp, k) for k in range(4)] + [
                np.rot90(np.fliplr(inp), k) for k in range(4)
            ]
            predicted = [views[next(iter(options))] for options in possible]
            prediction = np.block([[predicted[0], predicted[1]], [predicted[2], predicted[3]]])
            _require(np.array_equal(prediction, pair["output"]), "training-only prediction")

    _require(size_reuse >= 10 and size_change >= 10, "test-size regimes")
    _require(palette_reuse >= 10 and palette_novel >= 10, "test-palette regimes")
    _require(simpler_test >= 10 and symmetric_test >= 10, "simpler symmetric tests")
    _require(multi_test_palettes >= 8, "multiple test palettes")
    _require(repeated_geometry >= 5, "rotated or recolored geometry reuse")

    connected = disconnected = sparse = dense = corner_yes = corner_no = 0
    nonzero_field = many_nonzero = large_multicolor_symmetric = 0
    for grid in inputs:
        colors, frequencies = np.unique(grid, return_counts=True)
        background = int(colors[int(np.argmax(frequencies))])
        mask = grid != background
        component_count = components(mask)
        connected += component_count == 1
        disconnected += component_count >= 2
        sparse += .02 < float(mask.mean()) < .20
        dense += float(mask.mean()) > .60
        corner_yes += bool(mask[0, 0] or mask[0, -1] or mask[-1, 0] or mask[-1, -1])
        corner_no += not bool(mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1])
        nonzero_field += background != 0
        many_nonzero += sum(int(v != 0) for v in colors) >= 3
        large_multicolor_symmetric += len(grid) >= 9 and len(colors) >= 3 and _symmetry_class(grid) != "chiral"
    _require(connected >= 15 and disconnected >= 30, "component geometry diversity")
    _require(sparse >= 15 and dense >= 10, "density diversity")
    _require(corner_yes >= 15 and corner_no >= 15, "boundary geometry diversity")
    _require(nonzero_field >= 30 and many_nonzero >= 20, "multicolor field diversity")
    _require(large_multicolor_symmetric >= 5, "combined size-color-symmetry diversity")
