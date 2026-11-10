"""Solved ARC puzzles that turn payload area into a centered unary bar."""
import numpy as np

VERTICAL = "vertical"
HORIZONTAL = "horizontal"

# An input match is sufficient to exclude an exact source pair. Source sizes,
# colors, and guide layouts remain available for new configurations.
SOURCE_INPUTS = {
    tuple("""77707770777
77707770777
77707770777
77707770777
77705550777
00005550000
77705550777
77707770777
77707770777
77707770777
77707770777""".splitlines()),
    tuple("""777707777
777707777
000000000
777757777
777757777
777555777
000000000
777707777
777707777""".splitlines()),
    tuple("""7770777770777
7770777770777
7770777770777
7770777770777
7770577770777
7770555770777
0000555550000
7770555770777
7770577770777
7770777770777
7770777770777
7770777770777
7770777770777""".splitlines()),
    tuple("""7777777770777777777
7777777770777777777
7777777770777777777
7777777770777777777
7777777770777777777
0000000000000000000
7777777757577777777
7777777757577777777
7777777775777777777
7777777775777777777
7777777775777777777
7777777775777777777
7777777755577777777
0000000000000000000
7777777770777777777
7777777770777777777
7777777770777777777
7777777770777777777
7777777770777777777""".splitlines()),
}


def _count_arg(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, num_train, num_test):
    train = _count_arg("num_train", num_train, 2, 10)
    test = _count_arg("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        low = max(2, test + 1)
        train = low + seed % (11 - low)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _to_rc(cells, orientation):
    # u crosses the paired guides; v runs parallel to them.
    return {(v, u) for u, v in cells} if orientation == VERTICAL else set(cells)


def _to_uv(cells, orientation):
    return {(c, r) for r, c in cells} if orientation == VERTICAL else set(cells)


def _components(cells):
    pending = set(cells)
    total = 0
    while pending:
        total += 1
        stack = [pending.pop()]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in pending:
                    pending.remove(neighbor)
                    stack.append(neighbor)
    return total


def _input(shape, colors, orientation, first, second, cross, payload_cells):
    background, guide, payload = colors
    grid = np.full(shape, background, dtype=np.int8)
    if orientation == VERTICAL:
        grid[:, first] = guide
        grid[:, second] = guide
        grid[cross, :first + 1] = guide
        grid[cross, second:] = guide
    else:
        grid[first, :] = guide
        grid[second, :] = guide
        grid[:first + 1, cross] = guide
        grid[second:, cross] = guide
    for row, col in payload_cells:
        grid[row, col] = payload
    return grid


def _output(shape, colors, orientation, middle, cross, count):
    background, guide, payload = colors
    height, width = shape
    half = count // 2
    length = height if orientation == VERTICAL else width
    if count < 3 or count % 2 != 1 or cross - half < 0 or cross + half >= length:
        return None
    grid = np.full(shape, background, dtype=np.int8)
    if orientation == VERTICAL:
        grid[:, middle - 1] = guide
        grid[:, middle + 1] = guide
        grid[cross, :] = guide
        grid[cross - half:cross + half + 1, middle] = payload
    else:
        grid[middle - 1, :] = guide
        grid[middle + 1, :] = guide
        grid[:, cross] = guide
        grid[middle, cross - half:cross + half + 1] = payload
    return grid


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _rows(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _infer_input(grid):
    """Recover geometry and color roles from the visible input alone."""
    height, width = grid.shape
    values = [int(x) for x in np.unique(grid)]
    if len(values) != 3:
        return []
    matches = []
    for guide in values:
        for orientation in (VERTICAL, HORIZONTAL):
            if orientation == VERTICAL:
                full = [u for u in range(width) if np.all(grid[:, u] == guide)]
                ulen, vlen = width, height
            else:
                full = [u for u in range(height) if np.all(grid[u, :] == guide)]
                ulen, vlen = height, width
            if len(full) != 2:
                continue
            first, second = full
            strip = second - first - 1
            if strip < 3 or strip % 2 == 0 or first < 1 or second > ulen - 2:
                continue
            middle = (first + second) // 2
            for background in values:
                if background == guide:
                    continue
                payload = next(x for x in values if x not in (guide, background))
                cells = {(int(r), int(c)) for r, c in np.argwhere(grid == payload)}
                count = len(cells)
                if count < 3 or count % 2 == 0:
                    continue
                uv = _to_uv(cells, orientation)
                if any(not first < u < second for u, _ in uv):
                    continue
                colors = (background, guide, payload)
                half = count // 2
                for cross in range(half, vlen - half):
                    if np.array_equal(grid, _input(grid.shape, colors, orientation,
                                                   first, second, cross, cells)):
                        matches.append(dict(colors=colors, orientation=orientation,
                                            first=first, second=second, strip=strip,
                                            middle=middle, cross=cross, count=count,
                                            cells=cells, shape=grid.shape))
    return matches


def _sample_dimensions(rng, regime, source_like, min_v):
    if source_like:
        lo, hi = (9, 14) if regime == "train" else (19, 29)
    else:
        group = int(rng.choice(3, p=(.43, .34, .23) if regime == "train"
                               else (.29, .29, .42)))
        lo, hi = ((9, 14), (15, 22), (23, 30))[group]
    # Axes are drawn separately, allowing both rectangles and even dimensions.
    height = int(rng.integers(lo, hi + 1))
    width = int(rng.integers(lo, hi + 1))
    if min_v is not None:
        height = max(height, min_v)
        width = max(width, min_v)
    return height, width


def _grow(rng, pool, count):
    cells = {pool[int(rng.integers(len(pool)))]}
    pool_set = set(pool)
    while len(cells) < count:
        frontier = set()
        for u, v in cells:
            frontier.update(((u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)))
        choices = sorted((frontier & pool_set) - cells)
        if not choices:
            return None
        cells.add(choices[int(rng.integers(len(choices)))])
    return cells


def _payload(rng, first, second, vlen, count, style, orientation):
    strip = second - first - 1
    for _ in range(100):
        if style == "narrow":
            choices = [x for x in (1, 2, 3) if x <= strip]
        elif style == "dispersed":
            choices = list(range(1, strip + 1))
        else:
            choices = list(range(2, min(strip, 6) + 1))
        lanes = int(rng.choice(choices))
        u0 = int(rng.integers(first + 1, second - lanes + 1))
        min_span = (count + lanes - 1) // lanes
        if style == "clumps":
            min_span = max(3, min_span + 1)
        if min_span > vlen:
            continue
        if style == "dispersed":
            span_lo, span_hi = max(min_span, (vlen + 1) // 2), vlen
        else:
            extension = 3 if style == "compact" else 7
            span_lo, span_hi = min_span, min(vlen, min_span + extension)
        span = int(rng.integers(span_lo, span_hi + 1))
        v0 = int(rng.integers(0, vlen - span + 1))
        pool = [(u, v) for u in range(u0, u0 + lanes)
                for v in range(v0, v0 + span)]
        if len(pool) < count:
            continue
        if style == "clumps":
            separator = v0 + int(rng.integers(1, span - 1))
            left = [p for p in pool if p[1] < separator]
            right = [p for p in pool if p[1] > separator]
            if not left or not right or len(left) + len(right) < count:
                continue
            left_low = max(1, count - len(right))
            left_high = min(len(left), count - 1)
            if left_low > left_high:
                continue
            left_n = int(rng.integers(left_low, left_high + 1))
            right_n = count - left_n
            chosen = {left[int(i)] for i in rng.choice(len(left), left_n, replace=False)}
            chosen.update(right[int(i)] for i in rng.choice(len(right), right_n, replace=False))
        elif style in ("compact", "branched"):
            chosen = _grow(rng, pool, count)
            if chosen is None:
                continue
        else:
            chosen = {pool[int(i)] for i in rng.choice(len(pool), count, replace=False)}
        cells = _to_rc(chosen, orientation)
        if style == "clumps" and _components(cells) < 2:
            continue
        return cells
    raise ValueError("could not construct payload")


def _pair(rng, regime, orientation, colors, source_like=False,
          requested_count=None, different_from=None, style=None,
          force_free=False, requested_strip=None):
    for _ in range(100):
        shape = _sample_dimensions(rng, regime, source_like, requested_count)
        height, width = shape
        ulen, vlen = (width, height) if orientation == VERTICAL else (height, width)
        widths = [s for s in (3, 5, 7, 9, 11) if s <= ulen - 4]
        if source_like:
            widths = [s for s in widths if s in ((3, 5) if regime == "train" else (7, 9, 11))]
        if requested_strip is not None:
            widths = [s for s in widths if s == requested_strip]
        if not widths:
            continue
        strip = int(rng.choice(widths))
        first = int(rng.integers(1, ulen - strip - 2))
        second = first + strip + 1
        if requested_count is None:
            cap = min(21, vlen if vlen % 2 else vlen - 1)
            counts = [n for n in range(3, cap + 1, 2) if n != different_from]
            if force_free:
                counts = [n for n in counts if n <= vlen - 2]
            if not counts:
                continue
            count = int(rng.choice(counts))
        else:
            count = requested_count
            if count > vlen or (force_free and count > vlen - 2):
                continue
        half = count // 2
        free_crosses = list(range(half + 1, vlen - half - 1))
        all_crosses = list(range(half, vlen - half))
        if force_free or (free_crosses and rng.random() < .78):
            crosses = free_crosses
        else:
            crosses = all_crosses
        if not crosses:
            continue
        cross = int(rng.choice(crosses))
        chosen_style = style or str(rng.choice(
            ("compact", "branched", "narrow", "clumps", "dispersed"),
            p=(.22, .19, .20, .21, .18) if regime == "train"
            else (.14, .17, .20, .24, .25)))
        try:
            cells = _payload(rng, first, second, vlen, count, chosen_style, orientation)
        except ValueError:
            continue
        inp = _input(shape, colors, orientation, first, second, cross, cells)
        out = _output(shape, colors, orientation, (first + second) // 2, cross, count)
        if out is None or _rows(inp) in SOURCE_INPUTS:
            continue
        if len(_infer_input(inp)) != 1:
            continue
        return {"input": inp, "output": out}
    raise ValueError("could not construct a distinct, identifiable pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_n, test_n = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)

    # Presentation choices are sampled once. Geometry and count remain local
    # and visible; recolored seeds demonstrate local rather than fixed colors.
    source_like = bool(rng.random() < .23)
    recolor = bool(rng.random() < .56)
    single_orientation = not source_like and bool(rng.random() < .25)
    single_width = bool(rng.random() < .25)
    base_colors = tuple(int(x) for x in rng.choice(10, 3, replace=False))
    first_orientation = VERTICAL if rng.random() < .5 else HORIZONTAL
    train_orientations = [first_orientation]
    train_orientations.append(first_orientation if single_orientation else
        (HORIZONTAL if first_orientation == VERTICAL else VERTICAL)
        if source_like or rng.random() < .63 else first_orientation)
    for _ in range(train_n - 2):
        train_orientations.append(first_orientation if single_orientation else
                                  str(rng.choice((VERTICAL, HORIZONTAL))))
    if source_like and HORIZONTAL not in train_orientations:
        train_orientations[-1] = HORIZONTAL
    demonstrated = tuple(sorted(set(train_orientations)))
    test_orientations = [HORIZONTAL if source_like else str(rng.choice(demonstrated))
                         for _ in range(test_n)]

    palette = [base_colors]
    for index in range(1, train_n + test_n):
        if recolor:
            choices = tuple(int(x) for x in rng.choice(10, 3, replace=False))
            if index == 1 and choices == base_colors:
                choices = (choices[1], choices[2], choices[0])
            palette.append(choices)
        else:
            palette.append(base_colors)
    seeds = rng.integers(0, 2**63 - 1, size=train_n + test_n)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_count = None
    first_strip = None
    for index in range(train_n + test_n):
        split = "train" if index < train_n else "test"
        local = index if split == "train" else index - train_n
        orientation = (train_orientations if split == "train" else test_orientations)[local]
        pair_rng = np.random.default_rng(int(seeds[index]))
        requested = first_count if split == "train" and local == 2 else None
        different = first_count if split == "train" and local == 1 else None
        style = "compact" if split == "train" and local == 0 else (
            "clumps" if split == "train" and local == 1 else None)
        for _ in range(20):
            pair = _pair(pair_rng, split, orientation, palette[index], source_like,
                         requested, different, style,
                         force_free=(split == "train" and local == 0),
                         requested_strip=(first_strip if split == "train" and
                                          local > 0 and single_width else None))
            key = _grid_key(pair["input"])
            if key not in seen:
                break
        else:
            raise ValueError("could not find a unique input")
        seen.add(key)
        puzzle[split].append(pair)
        if split == "train" and local == 0:
            first_count = int(np.count_nonzero(pair["input"] == palette[index][2]))
            first_strip = _infer_input(pair["input"])[0]["strip"]
    validate_generated_puzzle(puzzle)
    return puzzle


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
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            ikey, okey = _grid_key(inp), _grid_key(out)
            assert ikey not in seen_inputs and (ikey, okey) not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add((ikey, okey))
            assert _rows(inp) not in SOURCE_INPUTS
            matches = _infer_input(inp)
            assert len(matches) == 1
            info = matches[0]
            expected = _output(info["shape"], info["colors"], info["orientation"],
                               info["middle"], info["cross"], info["count"])
            assert expected is not None and np.array_equal(out, expected)
            analyses[split].append(info)
    demonstrated = {a["orientation"] for a in analyses["train"]}
    assert {a["orientation"] for a in analyses["test"]} <= demonstrated
    train_colors = {a["colors"] for a in analyses["train"]}
    if any(a["colors"] not in train_colors for a in analyses["test"]):
        assert len(train_colors) > 1  # training demonstrates local recoloring
    # Paired evidence shows both counting and centering with visible free ends.
    assert analyses["train"][0]["count"] != analyses["train"][1]["count"]
    first = analyses["train"][0]
    length = first["shape"][0 if first["orientation"] == VERTICAL else 1]
    assert first["cross"] - first["count"] // 2 > 0
    assert first["cross"] + first["count"] // 2 < length - 1
    assert _components(analyses["train"][1]["cells"]) >= 2
    if len(analyses["train"]) >= 3:
        assert analyses["train"][2]["count"] == first["count"]


def _equal_puzzles(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x[name], y[name])
        for x, y in zip(a[split], b[split]) for name in ("input", "output"))
        for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for p in puzzles:
        validate_generated_puzzle(p)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    for offset, counts in enumerate(((2, 1), (10, 1), (10, 4), (6, 2))):
        p = generate_puzzle(1000 + offset, *counts)
        assert (len(p["train"]), len(p["test"])) == counts
        validate_generated_puzzle(p)
    for seed, train, test in ((2000, None, 1), (2001, None, 4),
                              (2002, 2, None), (2003, 10, None)):
        p = generate_puzzle(seed, train, test)
        assert train is None or len(p["train"]) == train
        assert test is None or len(p["test"]) == test
    assert _equal_puzzles(generate_puzzle(314159, 7, 3),
                          generate_puzzle(314159, 7, 3))
    assert not _equal_puzzles(generate_puzzle(2718), generate_puzzle(2719))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3.0, "num_test": 1}):
        try:
            generate_puzzle(0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid counts: {kwargs}")
    all_info = []
    shapes, layouts, payload_shapes, mappings = set(), set(), set(), set()
    alignments, components = set(), set()
    for p in puzzles:
        for split in ("train", "test"):
            for pair in p[split]:
                a = _infer_input(pair["input"])[0]
                all_info.append((split, a))
                shapes.add(a["shape"])
                mappings.add(a["colors"])
                layouts.add((a["shape"], a["orientation"], a["first"],
                             a["cross"], a["strip"], a["count"]))
                cells = a["cells"]
                r0 = min(r for r, _ in cells)
                c0 = min(c for _, c in cells)
                payload_shapes.add(tuple(sorted((r - r0, c - c0) for r, c in cells)))
                alignments.add(min(2, sum(v == a["cross"] for _, v in
                                      _to_uv(cells, a["orientation"]))))
                components.add(min(2, _components(cells)))
    assert len(shapes) >= 35 and len(layouts) >= 150
    assert len(payload_shapes) >= 120 and len(mappings) >= 25
    assert alignments == {0, 1, 2} and components == {1, 2}
    assert {a["orientation"] for _, a in all_info} == {VERTICAL, HORIZONTAL}
    assert {a["strip"] for _, a in all_info} >= {3, 5, 7, 9}
    assert any(h != w for h, w in shapes)
    assert any(h % 2 == 0 or w % 2 == 0 for h, w in shapes)
    assert any(a["count"] == 3 for _, a in all_info)
    assert any(split == "train" and _components(a["cells"]) >= 2
               for split, a in all_info)
    assert any(split == "test" and a["orientation"] == VERTICAL
               for split, a in all_info)
    assert any(split == "train" and a["strip"] >= 7 and
               len({u for u, _ in _to_uv(a["cells"], a["orientation"])}) <= 3
               for split, a in all_info)
    assert any(split == "test" and a["strip"] >= 7 and
               len({u for u, _ in _to_uv(a["cells"], a["orientation"])}) <= 3
               for split, a in all_info)
    assert any(split == "test" and a["orientation"] == VERTICAL and
               max(a["shape"]) >= 19 for split, a in all_info)
    assert any(split == "test" and max(a["shape"]) <= 14
               for split, a in all_info)
    assert any(split == "train" and max(a["shape"]) >= 19
               for split, a in all_info)
    assert any(a["count"] == (a["shape"][0] if a["orientation"] == VERTICAL
                                else a["shape"][1]) for _, a in all_info)
    assert any(len({p["colors"] for p in
                    (_infer_input(x["input"])[0] for split in ("train", "test")
                     for x in puzzle[split])}) == 1 for puzzle in puzzles)
    assert any(len({_infer_input(x["input"])[0]["colors"]
                    for x in puzzle["train"]}) > 1 for puzzle in puzzles)
    assert any(len({_infer_input(x["input"])[0]["orientation"]
                    for x in puzzle["train"]}) == 1 for puzzle in puzzles)
