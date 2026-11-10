"""Sparse quarter-turn motif completion for ARC grids."""

import numpy as np


# These are the four solved source pairs, used only to exclude exact copies.
_SOURCE = (
    (("0000000000","0000000000","0020300000","0004040000","0030403000",
      "0004040000","0000300000","0000000000","0000000000","0000000000"),
     ("0000000000","0000000000","0020302000","0004040000","0030403000",
      "0004040000","0020302000","0000000000","0000000000","0000000000")),
    (("0000000000","0030800000","0002020000","0080308000","0002020000",
      "0000800000","0000000000","0000000000","0000000000","0000000000"),
     ("0000000000","0030803000","0002020000","0080308000","0002020000",
      "0030803000","0000000000","0000000000","0000000000","0000000000")),
    (("0000000000","0008080800","0000400000","0008010800","0000000000",
      "0008080800","0000000000","0000000000","0000000000","0000000000"),
     ("0000000000","0008080800","0000404000","0008010800","0000404000",
      "0008080800","0000000000","0000000000","0000000000","0000000000")),
    (("0000000000","0000000000","0104010000","0020200000","0001000000",
      "0020200000","0100010000","0000000000","0000000000","0000000000"),
     ("0000000000","0000000000","0104010000","0020200000","0401040000",
      "0020200000","0104010000","0000000000","0000000000","0000000000")),
)


def _need(ok, message):
    if not ok:
        raise AssertionError(message)


def _counts(rng, seed, num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lo <= int(value) <= hi):
            raise ValueError(f"{name} outside {lo}..{hi}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = max(2, num_test + 1) + (seed // 4) % (11 - max(2, num_test + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, int(num_test) + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _orbit(dr, dc):
    result = []
    for _ in range(4):
        result.append((dr, dc))
        dr, dc = -dc, dr
    return tuple(result)


def _orbit_key(dr, dc):
    return min(_orbit(dr, dc))


def _available(radius):
    result = [("axis", radius, (radius, 0)),
              ("diagonal", max(1, radius - 1), (max(1, radius - 1), max(1, radius - 1)))]
    for r in range(2, radius + 1):
        if r != radius:
            result.append(("axis", r, (r, 0)))
        result.append(("diagonal", r, (r, r)))
        for b in range(1, r):
            result.append(("offaxis", r, (r, b)))
    return result


def _spec(rng, forced=None, shared=None, radius_cap=4):
    if shared is not None:
        return shared
    radius = int(rng.integers(2, radius_cap + 1))
    options = _available(radius)
    chosen = [options[0], options[1]]
    pool = [option for option in options if option not in chosen]
    if forced is not None and forced not in {x[0] for x in chosen}:
        matching = [x for x in pool if x[0] == forced]
        item = matching[int(rng.integers(len(matching)))]
        chosen.append(item)
        pool.remove(item)
    target = int(rng.integers(max(3, len(chosen)), min(6, len(options)) + 1))
    rng.shuffle(pool)
    chosen.extend(pool[:target - len(chosen)])
    return tuple(chosen)


def _colors(rng, bg, n_orbits):
    n = int(rng.integers(1, min(4 if n_orbits <= 3 else 6, n_orbits + 1) + 1))
    pool = [color for color in range(10) if color != bg]
    chosen = [int(x) for x in rng.choice(pool, size=n, replace=False)]
    # Reuse is intentional: orbit identity is geometric, not a color code.
    rng.shuffle(chosen)
    return [chosen[int(rng.integers(n))] for _ in range(n_orbits + 1)]


def _motif(rng, spec, bg, damage_hint, allow_complete=False):
    radius = max(item[1] for item in spec)
    side = 2 * radius + 1
    colors = _colors(rng, bg, len(spec))
    out = np.full((side, side), bg, dtype=np.int8)
    inp = out.copy()
    out[radius, radius] = colors[0]
    inp[radius, radius] = colors[0]
    names = [item[0] for item in spec]
    candidates = [i for i, name in enumerate(names) if damage_hint is None or name == damage_hint]
    if not candidates:
        candidates = list(range(len(spec)))
    if allow_complete:
        damaged = []
    else:
        first = candidates[int(rng.integers(len(candidates)))]
        damaged = [first]
        if len(spec) >= 4 and int(rng.integers(4)) == 0:
            other = [i for i in range(len(spec)) if i != first]
            damaged.append(other[int(rng.integers(len(other)))])
    for i, (_, _, offset) in enumerate(spec):
        offsets = _orbit(*offset)
        survivors = set(range(4))
        if i in damaged:
            keep = int(rng.integers(1, 4))
            survivors = set(int(x) for x in rng.choice(4, size=keep, replace=False))
        for index, (dr, dc) in enumerate(offsets):
            out[radius + dr, radius + dc] = colors[i + 1]
            if index in survivors:
                inp[radius + dr, radius + dc] = colors[i + 1]
    return inp, out


def _layout(rng, motifs, close):
    sides = [item[0].shape[0] for item in motifs]
    k = len(motifs)
    horizontal = bool(int(rng.integers(2)))
    gap = [int(rng.integers(5, 8)) for _ in range(k - 1)]
    span = sum(sides) + sum(gap)
    breadth = max(sides)
    if horizontal:
        minimum_h, minimum_w = breadth, span
    else:
        minimum_h, minimum_w = span, breadth
    occupied = sum(int(np.count_nonzero(item[1] != item[2])) for item in motifs)
    if close:
        h, w = minimum_h, minimum_w
        while h * w <= 2 * occupied:
            if h <= w and h < 30:
                h += 1
            else:
                w += 1
    else:
        h = min(30, minimum_h + int(rng.integers(2, 15)))
        w = min(30, minimum_w + int(rng.integers(2, 15)))
        if int(rng.integers(5)) == 0:
            if int(rng.integers(2)):
                h = 30
            else:
                w = 30
    if h > 30 or w > 30 or h * w <= 2 * occupied:
        raise ValueError("motifs do not fit identifiable canvas")
    row0 = int(rng.integers(h - minimum_h + 1))
    col0 = int(rng.integers(w - minimum_w + 1))
    bg = motifs[0][2]
    inp = np.full((h, w), bg, dtype=np.int8)
    out = inp.copy()
    cursor = 0
    for index, (local_in, local_out, _) in enumerate(motifs):
        side = sides[index]
        cross = int(rng.integers(breadth - side + 1))
        if horizontal:
            row, col = row0 + cross, col0 + cursor
        else:
            row, col = row0 + cursor, col0 + cross
        inp[row:row + side, col:col + side] = local_in
        out[row:row + side, col:col + side] = local_out
        cursor += side + (gap[index] if index < k - 1 else 0)
    return {"input": inp, "output": out}


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    _need(int(counts[index]) > grid.size // 2, "background must be a majority")
    return int(colors[index])


def _clusters(grid, bg):
    positions = {tuple(int(v) for v in point) for point in np.argwhere(grid != bg)}
    clusters = []
    while positions:
        cluster = {positions.pop()}
        fringe = list(cluster)
        while fringe:
            point = fringe.pop()
            adjacent = {q for q in positions if max(abs(q[0] - point[0]),
                                                     abs(q[1] - point[1])) <= 4}
            positions.difference_update(adjacent)
            cluster.update(adjacent)
            fringe.extend(adjacent)
        clusters.append(cluster)
    return clusters


def _interpret_cluster(grid, cluster, bg):
    h, w = grid.shape
    interpretations = []
    for center in cluster:
        groups = {}
        valid = True
        for row, col in cluster:
            if (row, col) == center:
                continue
            dr, dc = row - center[0], col - center[1]
            key = _orbit_key(dr, dc)
            groups.setdefault(key, []).append((row, col))
        complete = 0
        for key, points in groups.items():
            locations = [(center[0] + dr, center[1] + dc) for dr, dc in _orbit(*key)]
            if any(not (0 <= row < h and 0 <= col < w) for row, col in locations):
                valid = False
                break
            if len({int(grid[row, col]) for row, col in points}) != 1:
                valid = False
                break
            complete += len(points) == 4
        if valid and complete >= 2:
            interpretations.append((center, groups))
    _need(len(interpretations) == 1, "motif center is ambiguous")
    return interpretations[0]


def _solve_input(grid):
    bg = _background(grid)
    clusters = _clusters(grid, bg)
    _need(1 <= len(clusters) <= 3, "expected one to three separate motifs")
    result = grid.copy()
    for cluster in clusters:
        center, groups = _interpret_cluster(grid, cluster, bg)
        for key, points in groups.items():
            color = int(grid[points[0]])
            for dr, dc in _orbit(*key):
                row, col = center[0] + dr, center[1] + dc
                if result[row, col] == bg:
                    result[row, col] = color
                else:
                    _need(int(result[row, col]) == color, "conflicting orbit colors")
    return result


def _signature(grid):
    return grid.shape, grid.tobytes()


def _source_copy(pair):
    rows = lambda grid: tuple("".join(str(int(x)) for x in row) for row in grid)
    return (rows(pair["input"]), rows(pair["output"])) in _SOURCE


def _make_pair(rng, regime, index, shared, multi, seen):
    for _ in range(100):
        bg = int(rng.integers(10))
        count = 1
        if multi and index > 0 and int(rng.integers(3)) != 0:
            count = 2 + int(rng.integers(5) == 0)
        motifs = []
        for obj in range(count):
            hint = None
            if obj == 0:
                if regime == "source_train":
                    hint = ("diagonal", "offaxis", "axis")[index % 3]
                elif int(rng.integers(3)) == 0:
                    hint = "axis"
            spec = _spec(rng, hint, shared, radius_cap=2 if count == 3 else 4)
            local_in, local_out = _motif(
                rng, spec, bg, hint,
                allow_complete=obj > 0 and int(rng.integers(4)) == 0)
            motifs.append((local_in, local_out, bg))
        try:
            pair = _layout(rng, motifs, close=(index == 0 or int(rng.integers(4)) == 0))
            expected = _solve_input(pair["input"])
            if not np.array_equal(expected, pair["output"]):
                continue
            if _source_copy(pair):
                continue
            signature = _signature(pair["input"])
            if signature in seen:
                continue
            seen.add(signature)
            return pair
        except (AssertionError, ValueError):
            continue
    raise RuntimeError("unable to construct unambiguous pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(rng, seed, num_train, num_test)
    multi = bool(int(rng.integers(2)))
    shared = None
    if int(rng.integers(4)) == 0:
        shared = _spec(rng, forced="offaxis")
    puzzle = {"train": [], "test": []}
    seen = set()
    for index in range(ntrain):
        puzzle["train"].append(_make_pair(rng, "source_train", index, shared, multi, seen))
    for index in range(ntest):
        puzzle["test"].append(_make_pair(rng, "target_test", index + 1, shared, multi, seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
          "puzzle must contain train and test")
    _need(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
          "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal pair counts")
    inputs, pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"},
                  "pair must contain input and output")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _need(isinstance(grid, np.ndarray) and grid.ndim == 2,
                      "grids must be two-dimensional NumPy arrays")
                _need(all(1 <= dimension <= 30 for dimension in grid.shape)
                      and np.issubdtype(grid.dtype, np.integer)
                      and bool(np.all((0 <= grid) & (grid <= 9))),
                      "invalid ARC grid")
            _need(inp.shape == out.shape, "dimensions changed")
            key = _signature(inp)
            pair_key = (key, _signature(out))
            _need(key not in inputs and pair_key not in pairs, "duplicate pair")
            inputs.add(key)
            pairs.add(pair_key)
            _need(not _source_copy(pair), "exact source copy")
            _need(np.array_equal(_solve_input(inp), out), "incorrect quarter-turn completion")
            bg = _background(inp)
            changed = inp != out
            _need(bool(np.any(changed)) and bool(np.all(inp[changed] == bg)),
                  "only missing background cells may change")
            _need(set(int(x) for x in np.unique(inp)) ==
                  set(int(x) for x in np.unique(out)), "output introduced a color")


def self_test() -> None:
    shapes, backgrounds, survivor_counts, radii, damage_classes = (
        set(), set(), set(), set(), set())
    multiple_motifs = 0
    within_shape = 0
    within_background = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        local_shapes, local_backgrounds = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                bg = _background(inp)
                shape = tuple(inp.shape)
                shapes.add(shape)
                backgrounds.add(bg)
                local_shapes.add(shape)
                local_backgrounds.add(bg)
                clusters = _clusters(inp, bg)
                multiple_motifs += len(clusters) > 1
                for cluster in clusters:
                    _, groups = _interpret_cluster(inp, cluster, bg)
                    for key, points in groups.items():
                        radii.add(max(abs(key[0]), abs(key[1])))
                        if len(points) < 4:
                            survivor_counts.add(len(points))
                            if key[0] == 0 or key[1] == 0:
                                damage_classes.add("axis")
                            elif abs(key[0]) == abs(key[1]):
                                damage_classes.add("diagonal")
                            else:
                                damage_classes.add("offaxis")
        within_shape += len(local_shapes) > 1
        within_background += len(local_backgrounds) > 1
    _need(len(shapes) >= 50 and len(backgrounds) == 10,
          "insufficient shape or background diversity")
    _need(within_shape == 42 and within_background == 42,
          "pair-level dimensions or backgrounds collapsed")
    _need(multiple_motifs >= 40 and survivor_counts == {1, 2, 3}
          and radii == {1, 2, 3, 4}
          and damage_classes == {"axis", "diagonal", "offaxis"},
          "motif or damage diversity collapsed")
    for seed, train, test in ((80, 2, 1), (81, 10, 1), (82, 10, 4),
                              (90, None, 4), (91, 2, None)):
        puzzle = generate_puzzle(seed, train, test)
        _need(train is None or len(puzzle["train"]) == train, "train count ignored")
        _need(test is None or len(puzzle["test"]) == test, "test count ignored")
    for kwargs in ({"num_train": 1}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}):
        try:
            generate_puzzle(100, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    a, b = generate_puzzle(123456, 8, 3), generate_puzzle(123456, 8, 3)
    for split in ("train", "test"):
        for x, y in zip(a[split], b[split]):
            _need(np.array_equal(x["input"], y["input"]) and
                  np.array_equal(x["output"], y["output"]), "nondeterministic")
