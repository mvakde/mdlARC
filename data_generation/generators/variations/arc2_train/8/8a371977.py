"""Complete lattices whose outer compartment ring receives a distinct fill."""

import numpy as np


_SOURCE = frozenset({
    ((1,) * 11, (1,) * 11, (1,) * 12, (1,) * 12),
    ((4,) * 3, (4,) * 3, (1, 2, 2, 2), (1, 2, 2, 2)),
    ((3,) * 6, (3,) * 6, (1,) * 7, (1,) * 7),
    ((4,) * 5, (4,) * 5, (1, 2, 2, 2, 2, 1), (1, 2, 2, 2, 2, 1)),
})


def _check_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, num_train, num_test):
    num_train = _check_count(num_train, "num_train", 2, 10)
    num_test = _check_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + int(seed) % 4
        low = max(2, num_test + 1)
        num_train = low + (int(seed) // 4) % (11 - low)
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = low + int(seed) % (11 - low)
    elif num_test is None:
        num_test = 1 + int(seed) % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _axis(rng, n, style, irregular):
    settings = {
        "fine": (1, 2, 2), "macro": (3, 6, 3),
        "long": (2, 7, 2), "compact": (1, 3, 2),
        "medium": (2, 4, 2),
    }
    minimum, maximum, max_band = settings[style]
    if not irregular:
        choices = [(length, band, outer_a, outer_b)
                   for length in range(minimum, maximum + 1)
                   for band in range(1, max_band + 1)
                   for outer_a in (1, 2) for outer_b in (1, 2)
                   if n * length + (n - 1) * band + outer_a + outer_b <= 30]
        length, band, outer_a, outer_b = choices[int(rng.integers(len(choices)))]
        return (length,) * n, (outer_a,) + (band,) * (n - 1) + (outer_b,)
    lengths = [minimum] * n
    bands = [1] * (n + 1)
    budget = 30 - (n * minimum + n + 1)
    assert budget >= 1
    extras = int(rng.integers(1, min(budget, 10) + 1))
    for _ in range(extras):
        options = [(0, i) for i, value in enumerate(lengths) if value < maximum]
        options += [(1, i) for i, value in enumerate(bands)
                    if value < (2 if i in (0, n) else max_band)]
        if not options:
            break
        kind, index = options[int(rng.integers(len(options)))]
        if kind == 0:
            lengths[index] += 1
        else:
            bands[index] += 1
    if len(set(lengths)) == 1 and len(set(bands[1:-1])) == 1:
        if lengths[0] > minimum:
            lengths[0] -= 1
        elif sum(lengths) + sum(bands) < 30:
            lengths[0] += 1
        else:
            # Transfer one existing band cell without increasing the canvas.
            donor = next(i for i, value in enumerate(bands) if value > 1)
            bands[donor] -= 1
            lengths[0] += 1
    assert sum(lengths) + sum(bands) <= 30
    return tuple(lengths), tuple(bands)


def _spec(rng, category, irregular):
    if category == "fine":
        nr, nc = int(rng.integers(7, 12)), int(rng.integers(7, 12))
        style = "fine"
    elif category == "fine_train":
        nr, nc = int(rng.integers(6, 10)), int(rng.integers(6, 10))
        style = "fine"
    elif category == "macro_train":
        nr, nc = int(rng.integers(3, 5)), int(rng.integers(3, 5))
        style = "macro"
    elif category == "familiar":
        nr, nc = int(rng.integers(4, 6)), int(rng.integers(4, 6))
        style = "macro"
    elif category == "elongated":
        nr, nc = (3, int(rng.integers(4, 8))) if int(rng.integers(2)) else (
            int(rng.integers(4, 8)), 3)
        style = "long"
    elif category == "compact":
        nr, nc = int(rng.integers(3, 5)), int(rng.integers(3, 6))
        style = "compact"
    else:
        nr, nc = int(rng.integers(5, 8)), int(rng.integers(5, 8))
        style = "medium"
    rows, rb = _axis(rng, nr, style, irregular)
    cols, cb = _axis(rng, nc, style, irregular)
    return rows, cols, rb, cb


def _core(spec):
    rows, cols, rb, cb = spec
    return rows, cols, rb[1:-1], cb[1:-1]


def _build(spec, roles):
    rows, cols, rb, cb = spec
    empty, structural, perimeter, interior = roles
    shape = (sum(rows) + sum(rb), sum(cols) + sum(cb))
    source = np.full(shape, structural, dtype=np.int8)
    target = source.copy()
    r = rb[0]
    for i, height in enumerate(rows):
        c = cb[0]
        for j, width in enumerate(cols):
            region = np.s_[r:r + height, c:c + width]
            source[region] = empty
            target[region] = perimeter if i in (0, len(rows) - 1) or j in (
                0, len(cols) - 1) else interior
            c += width + cb[j + 1]
        r += height + rb[i + 1]
    return {"input": source, "output": target}


def _source_copy(spec, roles):
    return spec in _SOURCE and roles == (0, 1, 2, 3)


def _grid_key(grid):
    data = np.asarray(grid, dtype=np.int16)
    return data.shape, data.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(seed, num_train, num_test)
    perimeter, interior = (int(x) for x in rng.choice(10, size=2, replace=False))
    available = [x for x in range(10) if x not in (perimeter, interior)]
    fixed_palette = bool(rng.integers(2))
    irregular_family = bool(rng.random() < 0.8)
    fixed_visible = tuple(int(x) for x in rng.choice(available, size=2, replace=False))
    used_inputs, used_cores = set(), set()
    previous_visible = None

    def make_pair(category, irregular, distinct_core):
        nonlocal previous_visible
        for _ in range(40):
            spec = _spec(rng, category, irregular)
            if distinct_core and _core(spec) in used_cores:
                continue
            if fixed_palette:
                empty, structural = fixed_visible
            else:
                choices = [(a, b) for a in available for b in available
                           if a != b and (previous_visible is None or
                                          (a != previous_visible[0] and
                                           b != previous_visible[1]))]
                empty, structural = choices[int(rng.integers(len(choices)))]
            roles = (empty, structural, perimeter, interior)
            if _source_copy(spec, roles):
                continue
            pair = _build(spec, roles)
            key = _grid_key(pair["input"])
            if key in used_inputs:
                continue
            used_inputs.add(key)
            used_cores.add(_core(spec))
            previous_visible = (empty, structural)
            return pair
        raise AssertionError("could not construct a distinct lattice")

    train = [make_pair("fine_train", irregular_family, False),
             make_pair("macro_train", irregular_family, False)]
    extras = ("medium", "compact", "fine_train", "macro_train", "elongated")
    for _ in range(2, num_train):
        category = extras[int(rng.integers(len(extras)))]
        train.append(make_pair(category, irregular_family and bool(rng.integers(4)), False))
    categories = ("familiar", "fine", "elongated", "compact", "medium")
    test = []
    for i in range(num_test):
        category = categories[(int(seed) + i) % len(categories)]
        test.append(make_pair(category, irregular_family and bool(rng.integers(4)), True))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _runs(mask):
    changes = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    edges = np.concatenate(([0], changes, [mask.size]))
    return [(int(a), int(b), bool(mask[a])) for a, b in zip(edges[:-1], edges[1:])]


def _analyze(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    input_colors = set(int(x) for x in np.unique(input_grid))
    assert len(input_colors) == 2
    structural = int(input_grid[0, 0])
    empty = next(iter(input_colors - {structural}))
    rr = _runs(np.all(input_grid == structural, axis=1))
    cr = _runs(np.all(input_grid == structural, axis=0))
    assert rr[0][2] and rr[-1][2] and cr[0][2] and cr[-1][2]
    row_cells = [(a, b) for a, b, full in rr if not full]
    col_cells = [(a, b) for a, b, full in cr if not full]
    rb = tuple(b - a for a, b, full in rr if full)
    cb = tuple(b - a for a, b, full in cr if full)
    rows = tuple(b - a for a, b in row_cells)
    cols = tuple(b - a for a, b in col_cells)
    assert len(rows) >= 3 and len(cols) >= 3
    assert len(rb) == len(rows) + 1 and len(cb) == len(cols) + 1
    assert all(x in (1, 2) for x in (rb[0], rb[-1], cb[0], cb[-1]))
    assert all(1 <= x <= 3 for x in rb[1:-1] + cb[1:-1])
    reconstructed = np.full(input_grid.shape, structural, dtype=np.int8)
    for a, b in row_cells:
        for c, d in col_cells:
            reconstructed[a:b, c:d] = empty
    assert np.array_equal(input_grid, reconstructed)
    perimeter = int(output_grid[row_cells[0][0], col_cells[0][0]])
    interior = int(output_grid[row_cells[1][0], col_cells[1][0]])
    assert len({empty, structural, perimeter, interior}) == 4
    reconstructed = np.full(output_grid.shape, structural, dtype=np.int8)
    for i, (a, b) in enumerate(row_cells):
        for j, (c, d) in enumerate(col_cells):
            reconstructed[a:b, c:d] = perimeter if i in (0, len(rows) - 1) or j in (
                0, len(cols) - 1) else interior
    assert np.array_equal(output_grid, reconstructed)
    return (rows, cols, rb, cb), (empty, structural, perimeter, interior)


def _irregular(spec):
    rows, cols, rb, cb = spec
    return (len(set(rows)) > 1 or len(set(cols)) > 1 or
            len(set(rb[1:-1])) > 1 or len(set(cb[1:-1])) > 1)


def _predict(input_grid, perimeter, interior):
    structural = int(input_grid[0, 0])
    rr = _runs(np.all(input_grid == structural, axis=1))
    cr = _runs(np.all(input_grid == structural, axis=0))
    rows = [(a, b) for a, b, full in rr if not full]
    cols = [(a, b) for a, b, full in cr if not full]
    result = np.full(input_grid.shape, structural, dtype=np.int8)
    for i, (a, b) in enumerate(rows):
        for j, (c, d) in enumerate(cols):
            result[a:b, c:d] = perimeter if i in (0, len(rows) - 1) or j in (
                0, len(cols) - 1) else interior
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            ikey = _grid_key(pair["input"])
            pkey = (ikey, _grid_key(pair["output"]))
            assert ikey not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            spec, roles = _analyze(pair["input"], pair["output"])
            assert not _source_copy(spec, roles)
            analyses[split].append((spec, roles))
    all_roles = [roles for split in analyses.values() for _, roles in split]
    fills = {(roles[2], roles[3]) for _, roles in analyses["train"]}
    assert len(fills) == 1
    perimeter, interior = next(iter(fills))
    train_visible = {roles[:2] for _, roles in analyses["train"]}
    all_visible = {roles[:2] for roles in all_roles}
    assert len(all_visible) == 1 or len(train_visible) > 1
    assert any(_irregular(spec) for spec, _ in analyses["train"]) or not any(
        _irregular(spec) for spec, _ in analyses["test"])
    assert len(analyses["train"][0][0][0]) >= 6
    assert len(analyses["train"][0][0][1]) >= 6
    assert len(analyses["train"][1][0][0]) in (3, 4)
    assert len(analyses["train"][1][0][1]) in (3, 4)
    test_cores = [_core(spec) for spec, _ in analyses["test"]]
    assert len(test_cores) == len(set(test_cores))
    assert not set(test_cores) & {_core(spec) for spec, _ in analyses["train"]}
    for pair in puzzle["test"]:
        assert np.array_equal(_predict(pair["input"], perimeter, interior), pair["output"])


def _equal(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a[name], b[name])
        for a, b in zip(first[split], second[split])
        for name in ("input", "output"))
        for split in ("train", "test"))


def self_test() -> None:
    for nt, ns in ((2, 1), (10, 1), (10, 4),
                   (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(901 + (nt or 0) + (ns or 0), nt, ns)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ns is not None:
            assert len(puzzle["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 3), (4, 4)):
        try:
            generate_puzzle(77, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _equal(generate_puzzle(12345, 8, 3), generate_puzzle(12345, 8, 3))
    assert not _equal(generate_puzzle(12345, 8, 3), generate_puzzle(12346, 8, 3))
    train_counts, test_counts, fills, cores, shapes = set(), set(), set(), set(), set()
    irregular_train = irregular_test = varied_palette = 0
    smaller_tests = mixed_sizes = mixed_bands = combined_irregular_palette = 0
    elongated_tests = fine_tests = 0
    categories = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        tr = [_analyze(p["input"], p["output"]) for p in puzzle["train"]]
        te = [_analyze(p["input"], p["output"]) for p in puzzle["test"]]
        fills.add(tr[0][1][2:])
        irregular_train += any(_irregular(s) for s, _ in tr)
        irregular_test += any(_irregular(s) for s, _ in te)
        palette_changes = len({r[:2] for _, r in tr}) > 1
        varied_palette += palette_changes
        smaller_tests += any(
            p["input"].size < max(q["input"].size for q in puzzle["train"])
            for p in puzzle["test"])
        for spec, _ in tr + te:
            cores.add(_core(spec))
            shapes.add((sum(spec[0]) + sum(spec[2]), sum(spec[1]) + sum(spec[3])))
            mixed_sizes += len(set(spec[0])) > 1 or len(set(spec[1])) > 1
            mixed_bands += (len(set(spec[2][1:-1])) > 1 or
                            len(set(spec[3][1:-1])) > 1)
        for spec, _ in te:
            nr, nc = len(spec[0]), len(spec[1])
            fine_tests += nr >= 7 and nc >= 7
            elongated_tests += 3 in (nr, nc) and max(spec[0] + spec[1]) >= 4
            combined_irregular_palette += _irregular(spec) and palette_changes
            if nr >= 7 and nc >= 7:
                categories.add("fine")
            if 3 in (nr, nc) and max(spec[0] + spec[1]) >= 4:
                categories.add("elongated")
            if nr <= 4 and nc <= 5:
                categories.add("compact")
            if nr == 5 and nc == 5:
                categories.add("familiar")
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fills) >= 20 and len(cores) >= 100 and len(shapes) >= 40
    assert irregular_train >= 15 and irregular_test >= 12 and varied_palette >= 10
    assert smaller_tests >= 10 and mixed_sizes >= 50 and mixed_bands >= 40
    assert fine_tests >= 5 and elongated_tests >= 5
    assert combined_irregular_palette >= 5
    assert {"fine", "elongated", "compact", "familiar"} <= categories
