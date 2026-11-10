"""Parallel bars encode a boundary-seeded route by unary clearances.

Construction uses a sampled geometric plan. Validation and prediction decode
only visible grids; only the disappearing marker/permanent color convention
is learned across pairs. Direction, background and route are visible per pair.
"""
import numpy as np


_DIRECTIONS = ("down", "right", "up", "left")


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _choose_counts(rng, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _from_forward(array, direction):
    if direction == "down":
        return array.copy()
    if direction == "up":
        return array[::-1].copy()
    if direction == "right":
        return array.T.copy()
    assert direction == "left"
    return array[::-1].T.copy()


def _to_forward(array, direction):
    if direction in ("down", "up"):
        return _from_forward(array, direction)
    if direction == "right":
        return array.T.copy()
    assert direction == "left"
    return array.T[::-1].copy()


def _grid_key(array):
    return array.shape, tuple(map(int, array.flat))


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[s]) for s in ("train", "test"))


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    return int(colors[counts.argmax()])


def _learn_roles(train_pairs):
    """Learn replacement from demonstrations, without assuming literal colors."""
    conventions = set()
    for pair in train_pairs:
        a, b = pair["input"], pair["output"]
        assert a.shape == b.shape
        background = _background(a)
        replacements = set(zip(map(int, a[(a != b) & (a != background)]),
                               map(int, b[(a != b) & (a != background)])))
        assert len(replacements) == 1
        marker, bar = replacements.pop()
        assert marker != bar and background not in (marker, bar)
        assert np.all(b[a == marker] == bar)
        assert np.all(b[a == bar] == bar)
        conventions.add((bar, marker))
    assert len(conventions) == 1, "training disagrees about marker replacement"
    return conventions.pop()


def _decode(grid, convention):
    """Read bars and trace their commands; no generation plan or labels used."""
    bar, marker = convention
    background = _background(grid)
    colors = set(map(int, np.unique(grid)))
    assert len(colors) == 4 and {background, bar, marker} <= colors
    route = (colors - {background, bar, marker}).pop()
    seeds = np.argwhere(grid == route)
    assert seeds.shape == (1, 2)
    row, col = map(int, seeds[0])
    h, w = grid.shape
    directions = [d for d, match in zip(_DIRECTIONS,
                  (row == 0, col == 0, row == h - 1, col == w - 1)) if match]
    assert len(directions) == 1, "seed must identify one inward direction"
    direction = directions[0]
    a = _to_forward(grid, direction)
    height, width = a.shape
    initial = int(np.argwhere(a == route)[0, 1])
    structure = (a == bar) | (a == marker)
    rows = np.flatnonzero(structure.any(axis=1))
    assert 1 <= len(rows) <= 6
    result = a.copy()
    result[result == marker] = bar
    path = np.zeros(a.shape, dtype=bool)
    current, start, previous = initial, 0, -1
    ns, signs, spans, turns, symmetric = [], [], [], [], []
    for raw_f in rows:
        f = int(raw_f)
        assert 0 < f < height - 1
        occupied = np.flatnonzero(structure[f])
        lo, hi = int(occupied[0]), int(occupied[-1])
        assert len(occupied) == hi - lo + 1
        assert np.any(a[f, lo:hi + 1] == bar), "permanent role must remain visible"
        markers = tuple(map(int, np.flatnonzero(a[f] == marker)))
        n = len(markers)
        assert 1 <= n <= 4
        assert hi - lo + 1 >= max(2, 2 * n - 1)
        low = markers == tuple(lo + 2 * j for j in range(n))
        high = markers == tuple(hi - 2 * j for j in reversed(range(n)))
        d = n + 1
        candidates = []
        if low and hi + d < width:
            candidates.append((1, hi + d))
        if high and lo - d >= 0:
            candidates.append((-1, lo - d))
        assert len(candidates) == 1, "bar has no unique boundary-feasible bypass"
        sign, target = candidates[0]
        turn = f - d
        assert turn >= 1 and turn > previous and turn >= start
        assert lo <= current <= hi, "each bar must intercept the incoming ray"
        # Stepwise tracing is separate from the constructor's slice painting.
        for r in range(start, turn + 1):
            path[r, current] = True
        for c in range(current, target + sign, sign):
            path[turn, c] = True
        current, start, previous = target, turn, f
        ns.append(n)
        signs.append(sign)
        spans.append((lo, hi))
        turns.append(turn)
        symmetric.append(bool(low and high))
    for r in range(start, height):
        path[r, current] = True
    assert not np.any(path & structure)
    result[path] = route
    metadata = {
        "palette": (background, route, bar, marker), "direction": direction,
        "shape": tuple(grid.shape), "forward_size": height, "lateral_size": width,
        "bar_count": len(rows), "marker_counts": tuple(ns),
        "bypass_sides": tuple(signs), "symmetric": tuple(symmetric),
        "bar_rows": tuple(map(int, rows)), "spans": tuple(spans),
        "lengths": tuple(hi - lo + 1 for lo, hi in spans),
        "turns": tuple(turns), "initial": initial,
        "approach": turns[0], "exit": height - 1 - int(rows[-1]),
        "gaps": tuple(turns[i] - int(rows[i - 1]) for i in range(1, len(rows))),
        "repeated": any(a == b for a, b in zip(signs, signs[1:])),
    }
    return _from_forward(result, direction), metadata


def solve(train_pairs, test_input):
    """Predict using only training copies and the queried input."""
    return _decode(test_input, _learn_roles(train_pairs))[0]


def _analyze_pair(input_grid, output_grid):
    convention = _learn_roles([{"input": input_grid, "output": output_grid}])
    expected, features = _decode(input_grid, convention)
    assert np.array_equal(expected, output_grid)
    return features


def _span_options(width, current, n, side, symmetry):
    """Enumerate geometry, retaining symmetric spans only for unique exits."""
    d = n + 1
    result = []
    for lo in range(current + 1):
        for hi in range(max(current, lo + max(2, 2 * n - 1) - 1), width):
            symmetric = hi - lo + 1 == 2 * n - 1
            if symmetry is True and not symmetric:
                continue
            if symmetry is False and symmetric:
                continue
            target = hi + d if side == 1 else lo - d
            if not 0 <= target < width:
                continue
            other = lo - d if side == 1 else hi + d
            if symmetric and 0 <= other < width:
                continue
            result.append((lo, hi, target, side, symmetric))
    return result


def _sample_spans(rng, width, counts, alternating, repeat_at, symmetric_at,
                  allow_symmetric, initial=None):
    # Failed states are memoized: at most 6 * 30 * 3, no exponential retries.
    failed, options = set(), {}
    repeat_side = int(rng.choice((-1, 1)))

    def search(index, current, previous):
        if index == len(counts):
            return []
        state = (index, current, previous)
        if state in failed:
            return None
        sides = (-previous,) if alternating and previous else (-1, 1)
        if repeat_at is not None and index in (repeat_at, repeat_at + 1):
            sides = (repeat_side,)
        symmetry = True if index == symmetric_at else (None if allow_symmetric else False)
        key = (index, current, sides, symmetry)
        if key not in options:
            candidates = [option for side in sides for option in
                          _span_options(width, current, counts[index], side, symmetry)]
            if candidates:
                candidates = [candidates[int(j)] for j in rng.permutation(len(candidates))]
                if allow_symmetric and rng.random() < 0.3:
                    candidates.sort(key=lambda x: not x[4])
            options[key] = candidates
        for candidate in options[key]:
            rest = search(index + 1, candidate[2], candidate[3])
            if rest is not None:
                return [candidate] + rest
        failed.add(state)
        return None

    initials = [initial] if initial is not None else map(int, rng.permutation(np.arange(1, width - 1)))
    for start in initials:
        spans = search(0, start, 0)
        if spans is not None:
            return start, spans
    return None


def _forward_positions(rng, counts, size):
    minimum = sum(n + 1 for n in counts) + len(counts) + 2
    assert minimum <= size <= 30
    slack = size - minimum
    allocation = np.zeros(len(counts) + 1, dtype=int)
    if rng.random() < 0.3:
        allocation[int(rng.integers(len(allocation)))] = slack
    else:
        allocation = rng.multinomial(slack, np.full(len(allocation), 1 / len(allocation)))
    positions, previous = [], -1
    for index, n in enumerate(counts):
        turn = (1 if index == 0 else previous + 1) + int(allocation[index])
        previous = turn + n + 1
        positions.append(previous)
    assert positions[-1] + 1 + int(allocation[-1]) == size - 1
    return positions


def _draw_plan(shape, initial, positions, counts, spans, colors):
    background, route, bar, marker = colors
    a = np.full(shape, background, dtype=np.int8)
    b = a.copy()
    a[0, initial] = route
    current, start = initial, 0
    for f, n, (lo, hi, target, side, _) in zip(positions, counts, spans):
        a[f, lo:hi + 1] = bar
        for j in range(n):
            a[f, lo + 2 * j if side == 1 else hi - 2 * j] = marker
        b[f, lo:hi + 1] = bar
        turn = f - n - 1
        b[start:turn + 1, current] = route
        b[turn, min(current, target):max(current, target) + 1] = route
        current, start = target, turn
    b[start:, current] = route
    return a, b


def _make_pair(rng, colors, direction, counts, regime, alternating=False,
               repeat_at=None, symmetric_at=None, allow_symmetric=False,
               size_band=(5, 30), shape=None):
    assert regime in ("source_train", "target_test")
    minimum = sum(n + 1 for n in counts) + len(counts) + 2
    assert minimum <= 30
    if shape is None:
        lower, upper = max(minimum, size_band[0]), max(minimum, size_band[1])
        height = upper if rng.random() < 0.15 else int(rng.integers(lower, upper + 1))
        # Try compact widths too; feasibility, not a blanket palette/count bound,
        # determines which dimensions can support the entire sequence.
        width = int(rng.integers(5, 31))
        if rng.random() < 0.2:
            width = height
        if rng.random() < 0.15:
            width = 30
        widths = list(dict.fromkeys([width, min(30, width + 4), min(30, width + 9), 30]))
    else:
        height, width = shape
        assert minimum <= height <= 30 and 5 <= width <= 30
        widths = [width]
    for width in widths:
        plan = _sample_spans(rng, width, counts, alternating, repeat_at,
                             symmetric_at, allow_symmetric)
        if plan is not None:
            initial, spans = plan
            positions = _forward_positions(rng, counts, height)
            a, b = _draw_plan((height, width), initial, positions, counts, spans, colors)
            return {"input": _from_forward(a, direction),
                    "output": _from_forward(b, direction)}
    raise AssertionError("no feasible route for requested geometry")


def _marker_sequence(rng, count, maximum, required):
    ns = list(map(int, rng.integers(1, maximum + 1, size=count)))
    for index, value in required.items():
        ns[index] = value
    # Six bars fit when the sum of unary counts is at most sixteen.
    while sum(ns) + 2 * count + 2 > 30:
        choices = [i for i, n in enumerate(ns) if n > 1 and i not in required]
        assert choices
        i = int(rng.choice(choices))
        ns[i] -= 1
    return ns


def _has_repeat(features):
    return any(x["repeated"] for x in features)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _choose_counts(rng, num_train, num_test)
    colors = tuple(map(int, rng.choice(10, 4, replace=False)))
    maximum = int(rng.choice((2, 3, 4), p=(0.2, 0.25, 0.55)))
    repeated_mode = bool(rng.random() < 0.65)
    symmetric_mode = bool(rng.random() < 0.55)
    recolor = bool(rng.random() < 0.65)
    # These regimes change distributions, never the transformation.
    orientation_mode = int(rng.integers(3))  # shared, mixed, held-out direction
    directions = list(rng.permutation(_DIRECTIONS))
    train_dirs = directions[:1] if orientation_mode == 0 else directions[:3] if orientation_mode == 2 else directions
    test_dirs = directions[:1] if orientation_mode == 0 else directions[3:] if orientation_mode == 2 else directions
    scale_shift = bool(rng.random() < 0.3)
    count_shift = bool(rng.random() < 0.25)
    train_max = int(rng.integers(3, 6)) if count_shift else 6
    counts = list(map(int, rng.integers(1, train_max + 1, size=num_train)))
    while sum(counts) < maximum:
        i = int(rng.integers(num_train))
        if counts[i] < train_max:
            counts[i] += 1
    repeat_pair = int(rng.integers(num_train)) if repeated_mode else None
    if repeat_pair is not None:
        counts[repeat_pair] = max(2, counts[repeat_pair])
    requirements = [{} for _ in counts]
    slots = [(i, j) for i, count in enumerate(counts) for j in range(count)]
    for value, slot in zip(rng.permutation(np.arange(1, maximum + 1)), rng.permutation(len(slots))):
        i, j = slots[int(slot)]
        requirements[i][j] = int(value)
    sequences = [_marker_sequence(rng, count, maximum, required)
                 for count, required in zip(counts, requirements)]
    sym_pair, sym_index = None, None
    if symmetric_mode:
        candidates = [(i, j) for i, ns in enumerate(sequences) if i != repeat_pair
                      for j, n in enumerate(ns) if n >= 2]
        if not candidates:
            i = next(i for i in range(num_train) if i != repeat_pair)
            # Exchange bars to preserve counting coverage and feasibility budgets.
            donor = next((k, j) for k, ns in enumerate(sequences)
                         for j, n in enumerate(ns) if n >= 2)
            k, j = donor
            sequences[i][0], sequences[k][j] = sequences[k][j], sequences[i][0]
            candidates = [(i, 0)]
        sym_pair, sym_index = candidates[int(rng.integers(len(candidates)))]
    puzzle, seen = {"train": [], "test": []}, set()
    source_keys = {(_grid_key(p["input"]), _grid_key(p["output"])) for p in _source_pairs()}

    def add(split, ns, direction, repeat_at=None, symmetric_at=None):
        regime = "source_train" if split == "train" else "target_test"
        band = (5, 30)
        if scale_shift:
            band = (5, 20) if split == "train" else (21, 30)
        for _ in range(64):
            pair_colors = colors
            if recolor:
                available = [c for c in range(10) if c not in colors[2:]]
                bg, route = map(int, rng.choice(available, 2, replace=False))
                pair_colors = (bg, route, *colors[2:])
            pair = _make_pair(rng, pair_colors, direction, ns, regime,
                              alternating=not repeated_mode, repeat_at=repeat_at,
                              symmetric_at=symmetric_at, allow_symmetric=symmetric_mode,
                              size_band=band)
            key = _grid_key(pair["input"])
            if key not in seen and (key, _grid_key(pair["output"])) not in source_keys:
                seen.add(key)
                puzzle[split].append(pair)
                return
        raise AssertionError("exhausted unique geometric configurations")

    # Coverage obligations occupy randomly selected pairs and bar positions.
    for i, ns in enumerate(sequences):
        repeat_at = int(rng.integers(len(ns) - 1)) if i == repeat_pair else None
        add("train", ns, str(rng.choice(train_dirs)), repeat_at,
            sym_index if i == sym_pair else None)
    features = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
    # Both endpoint meanings are demonstrated even in a two-pair puzzle.
    if len({s for item in features for s in item["bypass_sides"]}) == 1:
        pair = puzzle["train"][-1]
        direction = features[-1]["direction"]
        for role in pair:
            pair[role] = _from_forward(_to_forward(pair[role], direction)[:, ::-1], direction)
        seen = {_grid_key(p["input"]) for p in puzzle["train"]}
    for _ in range(num_test):
        count = (int(rng.integers(train_max + 1, 7)) if count_shift else
                 int(rng.choice(np.arange(1, 7), p=(0.23, 0.2, 0.16, 0.15, 0.13, 0.13))))
        ns = _marker_sequence(rng, count, maximum, {})
        add("test", ns, str(rng.choice(test_dirs)))
    # Random presentation order prevents opening examples becoming templates.
    for split in puzzle:
        puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


# Compact transcription of the six staged source grids and labeled route bends.
# Used only for exact-copy rejection and regression, never for sampling layouts.
_SOURCE_CASES = (
    ("down", (19, 13), 8, ((7, 6, 12, (6, 8, 10, 12)),), ((2, 1, 8),)),
    ("right", (19, 17), 3, ((6, 0, 6, (0,)), (16, 7, 16, (12, 14, 16))), ((4, 3, 8), (12, 3, 8))),
    ("down", (19, 19), 4, ((4, 0, 6, (0,)), (10, 4, 18, (16, 18)), (16, 0, 5, (0, 2))), ((2, 4, 8), (7, 1, 8), (13, 1, 8))),
    ("right", (19, 12), 4, ((5, 0, 5, (0, 2)), (14, 6, 11, (11,))), ((2, 4, 8), (12, 4, 8))),
    ("left", (26, 14), 6, ((8, 0, 9, (0, 2, 4)), (18, 5, 13, (11, 13)), (24, 0, 5, (0,))), ((4, 6, 13), (15, 2, 13), (22, 2, 7))),
    ("down", (29, 16), 8, ((3, 4, 15, (15,)), (9, 0, 6, (0, 2)), (17, 6, 15, (11, 13, 15)), (27, 0, 8, (0, 2, 4, 6))), ((1, 2, 8), (6, 2, 9), (13, 2, 9), (22, 2, 13))),
)


def _source_pairs():
    pairs = []
    for direction, shape, initial, bars, bends in _SOURCE_CASES:
        a = np.full(shape, 8, dtype=np.int8)
        a[0, initial] = 3
        for f, lo, hi, markers in bars:
            a[f, lo:hi + 1] = 2
            a[f, list(markers)] = 1
        b = a.copy()
        b[b == 1] = 2
        previous, current = 0, initial
        for f, lo, hi in bends:
            assert current in (lo, hi)
            b[previous:f + 1, current] = 3
            b[f, lo:hi + 1] = 3
            previous, current = f, hi if current == lo else lo
        b[previous:, current] = 3
        pairs.append({"input": _from_forward(a, direction), "output": _from_forward(b, direction)})
    return pairs


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, pair_keys = set(), set()
    source_keys = {(_grid_key(p["input"]), _grid_key(p["output"])) for p in _source_pairs()}
    analyses = {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for a in pair.values():
                assert isinstance(a, np.ndarray) and a.ndim == 2
                assert all(1 <= size <= 30 for size in a.shape)
                assert np.issubdtype(a.dtype, np.integer)
                assert np.all((a >= 0) & (a <= 9))
            a, b = pair["input"], pair["output"]
            assert a.shape == b.shape and not np.array_equal(a, b)
            key = _grid_key(a)
            pair_key = key, _grid_key(b)
            assert key not in seen and pair_key not in pair_keys and pair_key not in source_keys
            seen.add(key)
            pair_keys.add(pair_key)
            analyses[split].append(_analyze_pair(a, b))
    convention = _learn_roles(puzzle["train"])
    assert all(x["palette"][2:] == convention for items in analyses.values() for x in items)
    train_counts = {n for x in analyses["train"] for n in x["marker_counts"]}
    assert len(train_counts) >= 2
    assert {s for x in analyses["train"] for s in x["bypass_sides"]} == {-1, 1}
    assert {n for x in analyses["test"] for n in x["marker_counts"]} <= train_counts
    if _has_repeat(analyses["test"]):
        assert _has_repeat(analyses["train"])
    if any(any(x["symmetric"]) for x in analyses["test"]):
        assert any(any(x["symmetric"]) for x in analyses["train"])
    training = [{k: v.copy() for k, v in p.items()} for p in puzzle["train"]]
    for pair in puzzle["test"]:
        prediction = solve(training, pair["input"].copy())
        assert np.array_equal(prediction, pair["output"])


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert _puzzle_key(automatic[7]) == _puzzle_key(generate_puzzle(7))
    assert _puzzle_key(generate_puzzle(1234, 6, 3)) == _puzzle_key(generate_puzzle(1234, 6, 3))
    assert len({_puzzle_key(p) for p in automatic}) == 42
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(178, *counts)
        validate_generated_puzzle(puzzle)
        for split, count in zip(("train", "test"), counts):
            assert count is None or len(puzzle[split]) == count
    for counts in ((1, 1), (11, 1), (3, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    source = _source_pairs()
    for i, pair in enumerate(source):
        # Each query sees the four source demonstrations, excluding its own label.
        training = [p for j, p in enumerate(source[:4]) if j != i]
        assert np.array_equal(solve(training, pair["input"].copy()), pair["output"])
    # Symmetric one-exit source; translating it to the middle creates two exits.
    assert _analyze_pair(source[0]["input"], source[0]["output"])["symmetric"] == (True,)
    ambiguous = np.full((19, 23), 8, dtype=np.int8)
    ambiguous[0, 10] = 3
    ambiguous[7, 8:15] = (1, 2, 1, 2, 1, 2, 1)
    try:
        solve(source[:4], ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError("two feasible symmetric exits were accepted")
    # Repeated-side geometry has literal expected turns, not solver-derived labels.
    spans = [(1, 4, 6, 1, False), (5, 8, 10, 1, False), (7, 12, 5, -1, False)]
    a, b = _draw_plan((18, 18), 3, [4, 9, 14], [1, 1, 1], spans, (8, 3, 2, 1))
    expected = np.full((18, 18), 8, dtype=np.int8)
    expected[4, 1:5] = expected[9, 5:9] = expected[14, 7:13] = 2
    expected[:3, 3] = expected[2, 3:7] = expected[2:8, 6] = 3
    expected[7, 6:11] = expected[7:13, 10] = expected[12, 5:11] = expected[12:, 5] = 3
    assert np.array_equal(b, expected)
    for direction in _DIRECTIONS:
        permutation = np.array([4, 9, 0, 6, 3, 5, 2, 8, 1, 7])
        training = [{k: permutation[_from_forward(v, direction)] for k, v in p.items()} for p in source[:4]]
        assert np.array_equal(solve(training, permutation[_from_forward(a, direction)]),
                              permutation[_from_forward(expected, direction)])
    rng = np.random.default_rng(91)
    for direction in _DIRECTIONS:
        for shape, ns in (((6, 5), [1]), ((30, 30), [1, 2, 3, 4, 2, 1])):
            pair = _make_pair(rng, (8, 3, 2, 1), direction, ns, "source_train", shape=shape)
            _analyze_pair(pair["input"], pair["output"])
    # Original dimensions remain usable with new contents.
    for direction, shape, *_ in _SOURCE_CASES:
        pair = _make_pair(rng, (0, 1, 2, 3), direction, [1], "target_test", shape=shape)
        _analyze_pair(pair["input"], pair["output"])
    # A legal-looking grid with one missing route cell must fail validation.
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in automatic[7][s]]
               for s in automatic[7]}
    pair = damaged["test"][0]
    background, route, _, _ = _analyze_pair(pair["input"], pair["output"])["palette"]
    row, col = np.argwhere((pair["input"] == background) & (pair["output"] == route))[0]
    pair["output"][row, col] = background
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incomplete route was accepted")
    features, joint, shapes, layouts, roles = [], set(), set(), set(), set()
    palette_varied = orientation_shared = orientation_mixed = heldout = repeated_puzzles = 0
    same_count_range = longer_tests = larger_tests = shared_scale = 0
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        by_split = {s: [_analyze_pair(p["input"], p["output"]) for p in puzzle[s]] for s in puzzle}
        local = by_split["train"] + by_split["test"]
        features.extend(local)
        palette_varied += len({x["palette"][:2] for x in local}) > 1
        dirs = {x["direction"] for x in local}
        orientation_shared += len(dirs) == 1
        orientation_mixed += len(dirs) > 1
        heldout += bool({x["direction"] for x in by_split["test"]} - {x["direction"] for x in by_split["train"]})
        repeated_puzzles += _has_repeat(local)
        train_max = max(x["bar_count"] for x in by_split["train"])
        test_max = max(x["bar_count"] for x in by_split["test"])
        longer_tests += test_max > train_max
        same_count_range += test_max <= train_max
        large = max(x["forward_size"] for x in by_split["test"]) > max(x["forward_size"] for x in by_split["train"])
        larger_tests += large
        shared_scale += not large
        for split in puzzle:
            for pair, x in zip(puzzle[split], by_split[split]):
                joint.add((x["direction"], x["bar_count"], split))
                shapes.add(x["shape"])
                roles.add(x["palette"][2:])
                normalized = np.empty_like(pair["input"])
                for i, color in enumerate(x["palette"]):
                    normalized[pair["input"] == color] = i
                layouts.add(_grid_key(normalized))
    assert len(shapes) >= 100 and len(layouts) >= 250 and len(roles) >= 20
    assert len(joint) >= 42
    assert all((direction, count, split) in joint for direction in _DIRECTIONS
               for count in (1, 2, 4) for split in ("train", "test"))
    assert {n for x in features for n in x["marker_counts"]} == {1, 2, 3, 4}
    assert sum(any(x["symmetric"]) for x in features) >= 20
    transitions = {(a, b) for x in features for a, b in zip(x["bypass_sides"], x["bypass_sides"][1:])}
    assert transitions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    assert min(x["lateral_size"] for x in features) <= 6
    assert max(x["lateral_size"] for x in features) == 30
    assert (30, 30) in shapes
    assert 1 in {gap for x in features for gap in x["gaps"]}
    assert max(x["approach"] for x in features) >= 8 and max(x["exit"] for x in features) >= 8
    assert min(palette_varied, orientation_shared, orientation_mixed, heldout,
               repeated_puzzles, same_count_range, longer_tests, larger_tests, shared_scale) >= 4
    assert any(x["repeated"] and 4 in x["marker_counts"] and x["lateral_size"] > 19 for x in features)
