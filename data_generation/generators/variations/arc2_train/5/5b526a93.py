import numpy as np


_SOURCE_LAYOUTS = (
    ((1, 7, 11, 17), (1, 6, 12), 22, 16, 1, 1),
    ((3, 8, 15), (2, 6, 12), 19, 17, 15, 2),
    ((1, 6, 12, 16), (2, 7, 11, 17, 21), 21, 30, 16, 2),
)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _max_count(size):
    return min(6, 1 + (30 - size) // (size + 1))


def _sample_axis(rng, size, count):
    gap_min = size + 1
    slack = 30 - size - (count - 1) * gap_min
    assert slack >= 0
    extra_cap = min(slack, 14 * (count - 1))
    if rng.random() < 0.7:
        extra_cap = min(extra_cap, 7)
    extra = int(rng.integers(0, extra_cap + 1))
    additions = [0] * (count - 1)
    for _ in range(extra):
        eligible = [i for i, n in enumerate(additions) if n < 14]
        additions[int(rng.choice(eligible))] += 1
    gaps = [gap_min + n for n in additions]
    remaining = slack - extra
    lead_cap = min(8, remaining)
    if rng.random() < 0.65:
        lead_cap = min(lead_cap, 3)
    lead = int(rng.integers(0, lead_cap + 1))
    trail_cap = min(8, remaining - lead)
    if rng.random() < 0.65:
        trail_cap = min(trail_cap, 3)
    trail = int(rng.integers(0, trail_cap + 1))
    positions = [lead]
    for gap in gaps:
        positions.append(positions[-1] + gap)
    return tuple(positions), positions[-1] + size + trail


def _stamp(grid, row, col, height, width, color):
    grid[row, col:col + width] = color
    grid[row + height - 1, col:col + width] = color
    grid[row:row + height, col] = color
    grid[row:row + height, col + width - 1] = color


def _build_pair(rows, columns, height, width, anchor_row, anchor_col,
                motif_height, motif_width, background, seed_color, completion):
    inp = np.full((height, width), background, dtype=np.int8)
    seeds = {(anchor_row, c) for c in columns} | {(r, anchor_col) for r in rows}
    for r, c in seeds:
        _stamp(inp, r, c, motif_height, motif_width, seed_color)
    out = inp.copy()
    for r in rows:
        for c in columns:
            if (r, c) not in seeds:
                _stamp(out, r, c, motif_height, motif_width, completion)
    return inp, out


def _components(grid, color):
    mask = grid == color
    seen = np.zeros(grid.shape, dtype=bool)
    found = []
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        stack = [(row, col)]
        seen[row, col] = True
        cells = set()
        while stack:
            r, c = stack.pop()
            cells.add((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]
                        and mask[rr, cc] and not seen[rr, cc]):
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        found.append(cells)
    return found


def _outlines(grid, color):
    positions, shapes = set(), set()
    for cells in _components(grid, color):
        top = min(r for r, _ in cells)
        left = min(c for _, c in cells)
        height = max(r for r, _ in cells) - top + 1
        width = max(c for _, c in cells) - left + 1
        assert 3 <= height <= 5 and 3 <= width <= 5
        expected = {(top + dr, left + dc)
                    for dr in range(height) for dc in range(width)
                    if dr in (0, height - 1) or dc in (0, width - 1)}
        assert cells == expected, "component is not a hollow outline"
        positions.add((top, left))
        shapes.add((height, width))
    assert len(shapes) == 1
    return positions, next(iter(shapes))


def _key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _inspect_pair(inp, out):
    assert inp.shape == out.shape
    changed = inp != out
    assert np.any(changed)
    before, after = np.unique(inp[changed]), np.unique(out[changed])
    assert len(before) == len(after) == 1
    background, completion = int(before[0]), int(after[0])
    input_colors = {int(x) for x in np.unique(inp)}
    assert len(input_colors) == 2 and background in input_colors
    seed_color = next(iter(input_colors - {background}))
    assert len({background, seed_color, completion}) == 3
    assert {int(x) for x in np.unique(out)} == {background, seed_color, completion}
    seeds, shape = _outlines(inp, seed_color)
    rows = tuple(sorted({r for r, _ in seeds}))
    cols = tuple(sorted({c for _, c in seeds}))
    assert 2 <= len(rows) <= 6 and 2 <= len(cols) <= 6
    full_rows = [r for r in rows if sum(rr == r for rr, _ in seeds) > 1]
    full_cols = [c for c in cols if sum(cc == c for _, cc in seeds) > 1]
    assert len(full_rows) == len(full_cols) == 1
    ar, ac = full_rows[0], full_cols[0]
    assert seeds == {(ar, c) for c in cols} | {(r, ac) for r in rows}
    row_gaps = tuple(b - a for a, b in zip(rows, rows[1:]))
    col_gaps = tuple(b - a for a, b in zip(cols, cols[1:]))
    assert min(row_gaps) >= shape[0] + 1
    assert min(col_gaps) >= shape[1] + 1
    missing = {(r, c) for r in rows for c in cols} - seeds
    assert missing
    added, added_shape = _outlines(out, completion)
    retained, retained_shape = _outlines(out, seed_color)
    assert added == missing and added_shape == shape
    assert retained == seeds and retained_shape == shape
    rebuilt = inp.copy()
    for r, c in missing:
        _stamp(rebuilt, r, c, *shape, completion)
    assert np.array_equal(rebuilt, out), "incorrect Cartesian completion"
    row_inside = ar not in (rows[0], rows[-1])
    col_inside = ac not in (cols[0], cols[-1])
    category = "plus" if row_inside and col_inside else (
        "tee" if row_inside or col_inside else "corner")
    return {"shape": shape, "rows": rows, "columns": cols,
            "row_gaps": row_gaps, "column_gaps": col_gaps,
            "anchor": (ar, ac), "category": category,
            "background": background, "seed": seed_color, "completion": completion,
            "margins": (rows[0], inp.shape[0] - rows[-1] - shape[0],
                        cols[0], inp.shape[1] - cols[-1] - shape[1])}


def _is_source_pair(inp, out):
    if inp.shape != out.shape:
        return False
    for rows, cols, height, width, ar, ac in _SOURCE_LAYOUTS:
        if inp.shape == (height, width):
            source_in, source_out = _build_pair(rows, cols, height, width,
                                                 ar, ac, 3, 3, 0, 1, 8)
            if np.array_equal(inp, source_in) and np.array_equal(out, source_out):
                return True
    return False


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            input_key = _key(pair["input"])
            pair_key = (input_key, _key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _is_source_pair(pair["input"], pair["output"])
            infos[section].append(_inspect_pair(pair["input"], pair["output"]))
    trains, tests = infos["train"], infos["test"]
    completion = trains[0]["completion"]
    assert all(info["completion"] == completion for info in trains + tests)
    assert len({(i["background"], i["seed"]) for i in trains}) >= 2
    assert trains[0]["shape"] == (3, 3) and trains[0]["category"] == "corner"
    assert len(trains[0]["rows"]) >= 3 and len(trains[0]["columns"]) >= 3
    train_shapes = {i["shape"] for i in trains}
    train_categories = {i["category"] for i in trains}
    assert all(i["shape"] in train_shapes for i in tests)
    assert all(i["category"] == "corner" or i["category"] in train_categories
               for i in tests)


def _sample_shape(rng):
    if rng.random() < 0.58:
        return (3, 3)
    shapes = ((4, 4), (5, 5), (3, 4), (4, 3), (3, 5), (5, 3), (4, 5), (5, 4))
    return shapes[int(rng.integers(len(shapes)))]


def _sample_category(rng):
    return ("corner", "tee", "plus")[int(rng.choice(3, p=(0.5, 0.3, 0.2)))]


def _sample_counts(rng, shape, category, limit, relation=None):
    options = [(r, c)
               for r in range(2, min(limit, _max_count(shape[0])) + 1)
               for c in range(2, min(limit, _max_count(shape[1])) + 1)
               if (category != "plus" or (r >= 3 and c >= 3))
               and (category != "tee" or r >= 3 or c >= 3)]
    if relation is not None:
        bound, larger = relation
        filtered = [(r, c) for r, c in options if (r * c > bound) == larger]
        if filtered:
            options = filtered
    return options[int(rng.integers(len(options)))]


def _choose_anchor(rng, rows, cols, category):
    if category == "corner":
        return rows[int(rng.choice((0, -1)))], cols[int(rng.choice((0, -1)))]
    if category == "plus":
        return (rows[int(rng.integers(1, len(rows) - 1))],
                cols[int(rng.integers(1, len(cols) - 1))])
    if (rng.random() < 0.5 and len(rows) >= 3) or len(cols) == 2:
        return (rows[int(rng.integers(1, len(rows) - 1))],
                cols[int(rng.choice((0, -1)))])
    return (rows[int(rng.choice((0, -1)))],
            cols[int(rng.integers(1, len(cols) - 1))])


def _sample_pair(rng, shape, category, counts, colors):
    rows, height = _sample_axis(rng, shape[0], counts[0])
    cols, width = _sample_axis(rng, shape[1], counts[1])
    ar, ac = _choose_anchor(rng, rows, cols, category)
    inp, out = _build_pair(rows, cols, height, width, ar, ac,
                           shape[0], shape[1], *colors)
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    completion = int(rng.integers(10))
    visible = [c for c in range(10) if c != completion]
    first_colors = tuple(int(x) for x in rng.choice(visible, size=2, replace=False))
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    train_specs = []

    def add_pair(section, shape, category, counts, colors):
        for _ in range(64):
            pair = _sample_pair(rng, shape, category, counts, colors)
            key = _key(pair["input"])
            if key not in used_inputs and not _is_source_pair(pair["input"], pair["output"]):
                used_inputs.add(key)
                puzzle[section].append(pair)
                return
        for background in visible:
            for seed_color in visible:
                if background == seed_color or (section == "train" and
                        len(puzzle["train"]) == 1 and
                        (background, seed_color) == first_colors):
                    continue
                pair = _sample_pair(rng, shape, category, counts,
                                    (background, seed_color, completion))
                key = _key(pair["input"])
                if key not in used_inputs and not _is_source_pair(pair["input"], pair["output"]):
                    used_inputs.add(key)
                    puzzle[section].append(pair)
                    return
        raise RuntimeError("unable to produce a distinct pair")

    for index in range(num_train):
        shape = (3, 3) if index == 0 else _sample_shape(rng)
        category = "corner" if index == 0 else _sample_category(rng)
        counts = ((int(rng.integers(3, 5)), int(rng.integers(3, 5)))
                  if index == 0 else _sample_counts(rng, shape, category, 5))
        if index == 0:
            colors = (*first_colors, completion)
        elif index == 1 and rng.random() < 0.45:
            colors = (first_colors[1], first_colors[0], completion)
        else:
            choices = [(b, s) for b in visible for s in visible
                       if b != s and (index != 1 or (b, s) != first_colors)]
            colors = (*choices[int(rng.integers(len(choices)))], completion)
        add_pair("train", shape, category, counts, colors)
        train_specs.append((shape, category, counts))

    max_train_area = max(r * c for _, _, (r, c) in train_specs)
    categories = {category for _, category, _ in train_specs}
    for _ in range(num_test):
        shape = train_specs[int(rng.integers(len(train_specs)))][0]
        if rng.random() < 0.35:
            shape = (3, 3)
        category = _sample_category(rng)
        if category not in categories:
            category = "corner"
        relation = (max_train_area, bool(rng.integers(2)))
        counts = _sample_counts(rng, shape, category, 6, relation)
        colors = (*[int(x) for x in rng.choice(visible, size=2, replace=False)],
                  completion)
        add_pair("test", shape, category, counts, colors)

    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"]))
                       for p in puzzle[section]) for section in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    corners, categories, shapes, counts = set(), set(), set(), set()
    aspects, gaps, margins, roles = set(), set(), set(), set()
    regular = irregular = right_train = recolored = smaller = larger = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        trains = [_inspect_pair(p["input"], p["output"]) for p in puzzle["train"]]
        tests = [_inspect_pair(p["input"], p["output"]) for p in puzzle["test"]]
        recolored += len({(i["background"], i["seed"]) for i in trains}) > 1
        right_train += any(i["anchor"][1] == i["columns"][-1] for i in trains)
        maximum = max(len(i["rows"]) * len(i["columns"]) for i in trains)
        smaller += any(len(i["rows"]) * len(i["columns"]) <= maximum for i in tests)
        larger += any(len(i["rows"]) * len(i["columns"]) > maximum for i in tests)
        for info in trains + tests:
            categories.add(info["category"])
            shapes.add(info["shape"])
            counts.add((len(info["rows"]), len(info["columns"])))
            aspects.add(np.sign(len(info["rows"]) - len(info["columns"])))
            roles.add((info["background"], info["seed"], info["completion"]))
            margins.update(info["margins"])
            for axis in (info["row_gaps"], info["column_gaps"]):
                gaps.update(axis)
                regular += len(set(axis)) == 1
                irregular += len(set(axis)) > 1
            if info["category"] == "corner":
                corners.add((info["anchor"][0] == info["rows"][0],
                             info["anchor"][1] == info["columns"][0]))
    assert recolored == 42 and right_train >= 20
    assert smaller >= 8 and larger >= 8
    assert len(corners) == 4 and categories == {"corner", "tee", "plus"}
    assert len(shapes) >= 6 and len(counts) >= 12 and len(aspects) == 3
    assert min(gaps) == 4 and max(gaps) >= 9
    assert 0 in margins and max(margins) >= 6
    assert regular >= 20 and irregular >= 20 and len(roles) >= 30
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(7000 + train * 10 + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for index, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(8000 + index, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(9000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    assert _puzzle_key(generate_puzzle(314159, 6, 3)) == _puzzle_key(generate_puzzle(314159, 6, 3))
    assert _puzzle_key(generate_puzzle(1001)) != _puzzle_key(generate_puzzle(1002))
