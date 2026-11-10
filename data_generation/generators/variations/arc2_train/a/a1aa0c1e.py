import numpy as np


_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 1), (8, 2), (9, 3), (10, 4), (5, 1), (7, 3), (10, 1))
# Shape, crossbar counts, and full-width platform rows of the four source pairs.
_SOURCE_FINGERPRINTS = {
    ((21, 21), (0, 2, 3), (1, 5, 11, 19)),
    ((21, 21), (1, 2, 3), (2, 6, 12, 20)),
    ((21, 21), (2, 2, 2), (1, 7, 14, 20)),
    ((21, 21), (2, 1, 3), (1, 8, 12, 20)),
}


def _counts(seed, num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                 ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(name + " is outside its legal range")
    if num_train is None and num_test is None:
        return _AUTO_COUNTS[int(seed) % len(_AUTO_COUNTS)]
    if num_train is None:
        choices = list(range(max(2, int(num_test) + 1), 11))
        num_train = choices[int(seed) % len(choices)]
    if num_test is None:
        choices = list(range(1, min(4, int(num_train) - 1) + 1))
        num_test = choices[int(seed) % len(choices)]
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _signature(grid):
    return (tuple(grid.shape), grid.astype(np.int8, copy=False).tobytes())


def _roles(rng, stable):
    if stable is not None:
        return stable
    return tuple(int(x) for x in rng.choice(10, 3, replace=False))


def _fresh_counts(rng, n, absent=None):
    for _ in range(32):
        values = [int(x) for x in rng.integers(1, 4, size=n)]
        if absent is not None:
            values[absent] = 0
        if sum(values) <= (8 if n == 4 else 9):
            return tuple(values)
    values = [1 + i % 3 for i in range(n)]
    if absent is not None:
        values[absent] = 0
    return tuple(values)


def _ladder_pattern(rng, count, extra):
    assert count >= 1
    assert extra == 0 or count >= 2
    between = [0] * (count - 1)
    for _ in range(extra):
        between[int(rng.integers(0, count - 1))] += 1
    pattern = [False]
    for rung in range(count):
        pattern.extend((True, False))
        if rung < count - 1:
            pattern.extend([False] * between[rung])
    return tuple(pattern)


def _build_pair(rng, n, kind, counts, break_at, roles, force_extra=False):
    background, terminal, token = roles
    assert len(counts) == n and counts[-1] > 0
    ladders = []
    gaps = []
    distances = []
    forced_index = int(rng.choice([i for i, c in enumerate(counts) if c >= 2])) if force_extra else -1
    for i, count in enumerate(counts):
        if count == 0:
            pattern = ()
            gap = int(rng.integers(1, 5))
            distance = gap + 1
        else:
            if force_extra:
                extra = 1 if i == forced_index else 0
            else:
                extra = int(rng.integers(1, 3)) if count >= 2 and rng.random() < 0.42 else 0
            pattern = _ladder_pattern(rng, count, extra)
            gap = int(rng.integers(1, 4)) if kind == "gap" and i == break_at else 0
            distance = len(pattern) + gap + 1
        ladders.append(pattern)
        gaps.append(gap)
        distances.append(distance)
    base = 1 + sum(distances)
    # Remove optional rail rows if they would squeeze out the scene.
    if base > 28:
        for i in range(n):
            if counts[i] and len(ladders[i]) > 2 * counts[i] + 1:
                ladders[i] = tuple([False] + [x for _ in range(counts[i]) for x in (True, False)])
                distances[i] = len(ladders[i]) + gaps[i] + 1
        base = 1 + sum(distances)
    assert base <= 30
    free = 30 - base
    top = int(rng.integers(0, min(4, free) + 1))
    free -= top
    bottom = int(rng.integers(0, min(4, free) + 1))
    height = base + top + bottom
    width = int(rng.integers(8, 31))
    rows = [top]
    for distance in distances:
        rows.append(rows[-1] + distance)
    assert rows[-1] + bottom == height - 1
    palette = [c for c in range(10) if c not in roles]
    colors = tuple(int(x) for x in rng.choice(palette, n, replace=False))
    grid = np.full((height, width), background, dtype=np.int8)
    for row, color in zip(rows, colors + (terminal,)):
        grid[row, :] = color
    lefts = []
    for i, pattern in enumerate(ladders):
        if not pattern:
            lefts.append(-1)
            continue
        left = int(rng.integers(1, width - 3))
        lefts.append(left)
        for offset, crossbar in enumerate(pattern, start=1):
            row = rows[i] + offset
            grid[row, left] = colors[i]
            grid[row, left + 2] = colors[i]
            if crossbar:
                grid[row, left + 1] = colors[i]
    last_left = lefts[-1]
    candidates = [c for c in range(1, width - 1)
                  if not last_left - 1 <= c <= last_left + 3]
    assert candidates
    column = int(rng.choice(candidates))
    assert grid[rows[-1] - 1, column] == background
    grid[rows[-1] - 1, column] = token
    output, metadata = _solve(grid)
    assert metadata["counts"] == counts
    return {"input": grid, "output": output}


def _inspect(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    h, w = grid.shape
    assert 1 <= h <= 30 and 1 <= w <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert grid.min() >= 0 and grid.max() <= 9
    hist = np.bincount(grid.astype(np.int64).ravel(), minlength=10)
    background = int(np.argmax(hist))
    assert np.sum(hist == hist[background]) == 1
    rows = []
    colors = []
    for r in range(h):
        color = int(grid[r, 0])
        if color != background and np.all(grid[r] == color):
            rows.append(r)
            colors.append(color)
    n = len(rows) - 1
    assert 2 <= n <= 4 and len(set(colors)) == len(colors)
    assert np.all(grid[:rows[0]] == background)
    assert np.all(grid[rows[-1] + 1:] == background)
    other = np.unique(grid[~np.isin(grid, np.array([background] + colors))])
    assert len(other) == 1
    token = int(other[0])
    positions = np.argwhere(grid == token)
    assert positions.shape == (1, 2)
    token_row, token_col = map(int, positions[0])
    assert token_row == rows[-1] - 1
    assert grid[rows[-1], token_col] == colors[-1]
    for dr, dc in ((-1, 0), (0, -1), (0, 1)):
        rr, cc = token_row + dr, token_col + dc
        if 0 <= rr < h and 0 <= cc < w:
            assert grid[rr, cc] == background
    counts, touching, gaps, lefts, extras = [], [], [], [], []
    for i in range(n):
        segment = grid[rows[i] + 1:rows[i + 1]]
        allowed = [background, colors[i]] + ([token] if i == n - 1 else [])
        assert np.all(np.isin(segment, allowed))
        occupied = np.flatnonzero(np.any(segment == colors[i], axis=1))
        if len(occupied) == 0:
            counts.append(0)
            touching.append(False)
            gaps.append(len(segment))
            lefts.append(-1)
            extras.append(0)
            continue
        assert occupied[0] == 0
        assert np.array_equal(occupied, np.arange(occupied[-1] + 1))
        length = len(occupied)
        first = np.flatnonzero(segment[0] == colors[i])
        assert len(first) == 2 and first[1] - first[0] == 2
        left = int(first[0])
        assert 1 <= left <= w - 4
        full = []
        for j in range(length):
            cells = np.flatnonzero(segment[j] == colors[i])
            assert np.array_equal(cells, [left, left + 2]) or np.array_equal(cells, [left, left + 1, left + 2])
            full.append(len(cells) == 3)
        assert not full[0] and not full[-1] and full[1] and full[-2]
        assert not any(full[j] and full[j + 1] for j in range(length - 1))
        count = sum(full)
        assert 1 <= count <= 3
        gap = len(segment) - length
        counts.append(count)
        touching.append(gap == 0)
        gaps.append(gap)
        lefts.append(left)
        extra = length - (2 * count + 1)
        assert 0 <= extra <= 2
        extras.append(extra)
    assert counts[-1] > 0 and touching[-1]
    assert not lefts[-1] - 1 <= token_col <= lefts[-1] + 3
    reachable = n - 1
    while reachable > 0 and touching[reachable - 1]:
        reachable -= 1
    return {"background": background, "terminal": colors[-1], "token": token,
            "colors": tuple(colors[:-1]), "rows": tuple(rows), "counts": tuple(counts),
            "touching": tuple(touching), "gaps": tuple(gaps), "lefts": tuple(lefts),
            "extras": tuple(extras), "reachable": reachable, "shape": (h, w)}


def _solve(grid):
    m = _inspect(grid)
    n = len(m["counts"])
    output = np.full((n, 5), m["background"], dtype=np.int8)
    for i, (color, count) in enumerate(zip(m["colors"], m["counts"])):
        output[i, :count] = color
    output[:, 3] = m["terminal"]
    output[m["reachable"], 4] = m["token"]
    return output, m


def _spec(rng, n, kind, break_at, forced_counts=None):
    counts = forced_counts if forced_counts is not None else _fresh_counts(rng, n, break_at if kind == "absent" else None)
    return kind, tuple(counts), break_at


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(int(seed))
    n = (2, 3, 4)[int(seed) % 3]
    stable = _roles(rng, None) if int(seed) % 4 == 0 else None
    positive = [1, 2, 3]
    rng.shuffle(positive)
    if n == 2:
        first = tuple(positive[:2])
        second = (0, positive[2])
    else:
        first = tuple(positive + ([int(rng.integers(1, 3))] if n == 4 else []))
        first = tuple(int(x) for x in rng.permutation(first))
        second = _fresh_counts(rng, n, 0)
    train_specs = [_spec(rng, n, "all", None, first),
                   _spec(rng, n, "absent", 0, second)]
    if train_count >= 3:
        gap_at = int(rng.integers(1, n - 1)) if n >= 3 else 0
        train_specs.append(_spec(rng, n, "gap", gap_at))
    if train_count >= 4 and n >= 3:
        train_specs.append(_spec(rng, n, "absent", 1))
    for _ in range(len(train_specs), train_count):
        kind = ("all", "absent", "gap")[int(rng.integers(0, 3))]
        at = None if kind == "all" else (int(rng.integers(0, n - 1)) if kind == "absent" or n == 2 else int(rng.integers(1, n - 1)))
        train_specs.append(_spec(rng, n, kind, at))
    puzzle = {"train": [], "test": []}
    seen = set()
    def add(split, spec):
        kind, counts, at = spec
        for _ in range(80):
            child = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            roles = _roles(child, stable)
            if stable is None and split == "train" and len(puzzle["train"]) == 1:
                first_roles = _inspect(puzzle["train"][0]["input"])
                if roles == (first_roles["background"], first_roles["terminal"], first_roles["token"]):
                    roles = (roles[1], roles[2], roles[0])
            pair = _build_pair(child, n, kind, counts, at, roles,
                               force_extra=(split == "train" and not puzzle["train"]))
            m = _inspect(pair["input"])
            fp = (m["shape"], m["counts"], m["rows"])
            sig = _signature(pair["input"])
            if sig not in seen and fp not in _SOURCE_FINGERPRINTS:
                seen.add(sig)
                puzzle[split].append(pair)
                return
        raise AssertionError("distinct pair construction exhausted")
    for spec in train_specs:
        add("train", spec)
    train_meta = [_inspect(p["input"]) for p in puzzle["train"]]
    reached = {m["reachable"] for m in train_meta}
    gap_shown = any(any(c > 0 and not t for c, t in zip(m["counts"], m["touching"])) for m in train_meta)
    test_modes = [("all", None), ("absent", 0)]
    if gap_shown and 1 in reached:
        test_modes.append(("gap", 0))
    for m in train_meta[2:]:
        for i, (c, t) in enumerate(zip(m["counts"][:-1], m["touching"][:-1])):
            if not t and i + 1 in reached:
                mode = ("absent" if c == 0 else "gap", i)
                if mode not in test_modes:
                    test_modes.append(mode)
    for i in range(test_count):
        # Rotate through demonstrated reach levels; the new top-gap combination
        # is used when training has separately shown a gap and a row-1 stop.
        mode = test_modes[(int(seed) + i + 2) % len(test_modes)]
        add("test", _spec(rng, n, mode[0], mode[1]))
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    metadata = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= x <= 30 for x in grid.shape)
                assert grid.min() >= 0 and grid.max() <= 9
            sig = _signature(pair["input"])
            assert sig not in seen
            seen.add(sig)
            expected, m = _solve(pair["input"])
            assert np.array_equal(expected, pair["output"])
            assert (m["shape"], m["counts"], m["rows"]) not in _SOURCE_FINGERPRINTS
            metadata[split].append(m)
    all_meta = metadata["train"] + metadata["test"]
    n = len(all_meta[0]["counts"])
    assert all(len(m["counts"]) == n for m in all_meta)
    train = metadata["train"]
    assert {v for m in train for v in m["counts"]} == {0, 1, 2, 3}
    assert any(m["reachable"] == 0 for m in train)
    assert any(m["counts"][0] == 0 and m["reachable"] == 1 for m in train)
    if len(train) >= 3:
        assert any(any(c > 0 and not t for c, t in zip(m["counts"], m["touching"])) for m in train)
    assert any(any(extra > 0 for extra in m["extras"]) for m in train)
    if n >= 3:
        for m in train:
            assert not (m["counts"][0] > 0 and not m["touching"][0])
    if len(train) == 2:
        assert all(not any(c > 0 and not t for c, t in zip(m["counts"], m["touching"])) for m in train)
    train_reached = {m["reachable"] for m in train}
    train_gap = any(any(c > 0 and not t for c, t in zip(m["counts"], m["touching"])) for m in train)
    for m in metadata["test"]:
        assert m["reachable"] in train_reached
        if any(c > 0 and not t for c, t in zip(m["counts"], m["touching"])):
            assert train_gap
    roles = {(m["background"], m["terminal"], m["token"]) for m in all_meta}
    assert all(len(set(x)) == 3 for x in roles)
    if len(roles) > 1:
        assert len({(m["background"], m["terminal"], m["token"]) for m in train}) > 1


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("malformed puzzle") from exc


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        assert any(x > 0 for x in _inspect(puzzle["train"][0]["input"])["extras"])
        assert len({tuple(pair["input"].shape) for pair in puzzle["train"] + puzzle["test"]}) > 1
        if len(puzzle["test"]) > 1:
            assert len({(_inspect(pair["input"])["counts"], _inspect(pair["input"])["reachable"])
                        for pair in puzzle["test"]}) > 1
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    records = [_inspect(pair["input"]) for p in puzzles for split in ("train", "test") for pair in p[split]]
    assert {len(m["counts"]) for m in records} == {2, 3, 4}
    assert len({m["shape"] for m in records}) >= 25
    assert len({m["counts"] for m in records}) >= 20
    assert len({m["reachable"] for m in records}) >= 3
    assert any(any(x > 0 for x in m["extras"]) for m in records)
    assert any(m["counts"][1] == 0 and m["counts"][0] > 0 for m in records if len(m["counts"]) >= 3)
    assert any(m["counts"][0] > 0 and not m["touching"][0] for m in records)
    assert any(len({(m["background"], m["terminal"], m["token"]) for m in [_inspect(q["input"]) for q in p["train"] + p["test"]]}) > 1 for p in puzzles)
    assert any(len({(m["background"], m["terminal"], m["token"]) for m in [_inspect(q["input"]) for q in p["train"] + p["test"]]}) == 1 for p in puzzles)
    tests = [_inspect(pair["input"]) for p in puzzles for pair in p["test"]]
    assert len({(m["counts"], m["reachable"]) for m in tests}) >= 50
    assert len({m["shape"][1] for m in tests}) >= 20
    assert {m["rows"][0] for m in tests} == {0, 1, 2, 3, 4}
    assert {m["reachable"] for m in tests} == {0, 1, 2, 3}
    assert any(any(x > 0 for x in m["extras"]) and any(c > 0 and not t for c, t in zip(m["counts"], m["touching"])) for m in records)
    by_counts = {}
    for m in records:
        by_counts.setdefault(m["counts"], set()).add(m["reachable"])
    assert sum(len(reaches) > 1 for reaches in by_counts.values()) >= 20
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(1000 + a * 10 + b, a, b)
        assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(2027, a, b))
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(42, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    p = generate_puzzle(345, 6, 3)
    q = generate_puzzle(345, 6, 3)
    r = generate_puzzle(346, 6, 3)
    assert all(np.array_equal(x[k], y[k]) for split in ("train", "test") for x, y in zip(p[split], q[split]) for k in ("input", "output"))
    assert any(not np.array_equal(x["input"], y["input"]) for x, y in zip(p["train"], r["train"]))
