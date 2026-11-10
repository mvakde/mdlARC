"""Keyed color echoes: encounter rank determines radial half-turn distance.

Only standard Python and NumPy are used.  solve() reads a single input; generation
constructs its labels from planned tiles, and validation checks scalar coordinates.
"""

from collections import Counter
from itertools import combinations

import numpy as np


DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0))
CONFIGURATIONS = tuple((a, b, r) for a in (0, 1) for b in (False, True) for r in (False, True))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _grid_key(a):
    return a.shape, a.tobytes()


def _vectors(configuration):
    axis, bottom, right = configuration
    sy, sx = (-1 if bottom else 1), (-1 if right else 1)
    return ((0, sx), (sy, 0)) if axis == 0 else ((sy, 0), (0, sx))


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _ordered(directions, configuration, alternative="correct"):
    fast, slow = _vectors(configuration)
    if alternative == "physical":
        return sorted(directions)
    if alternative == "fast_major":
        return sorted(directions, key=lambda d: (_dot(d, fast), _dot(d, slow)))
    sf = -1 if alternative == "reverse_fast" else 1
    ss = -1 if alternative == "reverse_slow" else 1
    return sorted(directions, key=lambda d: (ss * _dot(d, slow), sf * _dot(d, fast)))


def _ranks(samples, configuration, alternative="correct"):
    order = _ordered(samples, configuration, alternative)
    frequency, seen, answer = Counter(samples.values()), Counter(), {}
    for i, d in enumerate(order):
        seen[samples[d]] += 1
        if alternative == "one_shell":
            answer[d] = 1
        elif alternative == "total_frequency":
            answer[d] = frequency[samples[d]]
        elif alternative == "global_index":
            answer[d] = i + 1
        else:
            answer[d] = seen[samples[d]]
    return answer


def _read_key(a):
    candidates = []
    if min(a.shape) < 8:
        raise ValueError("canvas too small for this family")
    for bottom in (False, True):
        for right in (False, True):
            aligned = a[::-1] if bottom else a
            aligned = aligned[:, ::-1] if right else aligned
            origin = int(aligned[0, 0])
            for axis in (0, 1):
                solid = aligned[0, 1:] if axis == 0 else aligned[1:, 0]
                dashed = aligned[1:, 0] if axis == 0 else aligned[0, 1:]
                guide, bg = int(solid[0]), int(dashed[1])
                if len({origin, guide, bg}) != 3:
                    continue
                if not (np.all(solid == guide) and np.all(dashed[::2] == guide)
                        and np.all(dashed[1::2] == bg)):
                    continue
                if not (np.all(aligned[-1, 1:] == bg) and np.all(aligned[1:, -1] == bg)):
                    continue
                candidates.append(dict(configuration=(axis, bottom, right), background=bg,
                                       origin=origin, guide=guide))
    _require(len(candidates) == 1, "key must determine exactly one ordered coordinate system")
    return candidates[0]


def _components(mask):
    points = set(map(tuple, np.argwhere(mask)))
    result = []
    while points:
        p = min(points)
        points.remove(p)
        todo, component = [p], []
        while todo:
            y, x = todo.pop()
            component.append((int(y), int(x)))
            for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if q in points:
                    points.remove(q)
                    todo.append(q)
        result.append(component)
    return result


def _parse(value):
    a = _grid(value)
    info = _read_key(a)
    axis, bottom, right = info["configuration"]
    bg = info["background"]
    payload = a != bg
    key_row, key_col = (-1 if bottom else 0), (-1 if right else 0)
    payload[key_row, :] = False
    payload[:, key_col] = False
    singletons = [c[0] for c in _components(payload) if len(c) == 1]
    _require(len(singletons) == 1, "expected one isolated hub beacon")
    y, x = singletons[0]
    carrier = int(a[y, x])
    _require(carrier not in (bg, info["origin"], info["guide"]), "carrier role collision")
    _require(y >= 5 and x >= 5 and y + 5 <= a.shape[0] and x + 5 <= a.shape[1],
             "the full source-slot square and its margin must fit")
    expected = np.zeros(a.shape, dtype=bool)
    expected[y, x] = True
    tiles, samples = {}, {}
    for dr, dc in DIRECTIONS:
        r, c = y + 3 * dr - 1, x + 3 * dc - 1
        tile = a[r:r + 3, c:c + 3]
        if np.all(tile == bg):
            continue
        _require(bool(np.all(tile != bg)), "source slots must be empty or full 3x3 tiles")
        _require(bool(np.any(tile == carrier)), "every tile must contain the visible carrier")
        sample = int(tile[1 - dr, 1 - dc])
        _require(sample not in (carrier, bg), "inward sample must differ from carrier and background")
        expected[r:r + 3, c:c + 3] = True
        tiles[(dr, dc)], samples[(dr, dc)] = tile.copy(), sample
    _require(3 <= len(tiles) <= 8, "expected three to eight tiles")
    _require(np.array_equal(payload, expected), "noise or malformed central hole")
    _require(max(Counter(samples.values()).values()) <= 3, "sample color occurs more than three times")
    ranks = _ranks(samples, info["configuration"])
    # Include the complete source-slot square in the protected bounding footprint.
    lo_r, hi_r, lo_c, hi_c = y - 4, y + 4, x - 4, x + 4
    for (dr, dc), rank in ranks.items():
        r, c = y + 3 * (rank + 1) * dr, x + 3 * (rank + 1) * dc
        lo_r, hi_r = min(lo_r, r - 1), max(hi_r, r + 1)
        lo_c, hi_c = min(lo_c, c - 1), max(hi_c, c + 1)
    interior_lo_r, interior_lo_c = int(not bottom), int(not right)
    interior_hi_r = a.shape[0] - 1 - int(bottom)
    interior_hi_c = a.shape[1] - 1 - int(right)
    _require(lo_r >= interior_lo_r + 1 and hi_r <= interior_hi_r - 1
             and lo_c >= interior_lo_c + 1 and hi_c <= interior_hi_c - 1,
             "echo footprint needs one background cell of interior margin")
    info.update(grid=a, hub=(y, x), carrier=carrier, tiles=tiles, samples=samples,
                ranks=ranks, footprint=(lo_r, hi_r, lo_c, hi_c))
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible key, beacon, source tiles, and sample-color equality."""
    d = _parse(input_grid)
    a = d["grid"]
    out = a.copy()
    y, x = d["hub"]
    for direction in _ordered(d["tiles"], d["configuration"]):
        dr, dc = direction
        out[y + dr, x + dc] = d["samples"][direction]
        radius = 3 * (d["ranks"][direction] + 1)
        r, c = y + radius * dr - 1, x + radius * dc - 1
        out[r:r + 3, c:c + 3] = d["tiles"][direction][::-1, ::-1]
    return out


def _views(tile):
    return tuple(np.rot90(tile, k) for k in range(4)) + tuple(
        np.rot90(tile[:, ::-1], k) for k in range(4))


def _witness_tile(rng, carrier, sample, pool, direction):
    """Random three-mark motif with a unique sample and no D4 stabilizer."""
    inward = (1 - direction[0], 1 - direction[1])
    auxiliary = rng.choice([c for c in pool if c != sample], size=2, replace=False)
    positions = [(r, c) for r in range(3) for c in range(3) if (r, c) != inward]
    choices = []
    for p, q in combinations(positions, 2):
        tile = np.full((3, 3), carrier, dtype=np.int8)
        tile[inward], tile[p], tile[q] = sample, auxiliary[0], auxiliary[1]
        if len({t.tobytes() for t in _views(tile)}) == 8:
            choices.append(tile)
    _require(bool(choices), "no asymmetric motif")
    return choices[int(rng.integers(len(choices)))].copy()


def _ordinary_tile(rng, carrier, sample, pool, direction):
    inward = (1 - direction[0], 1 - direction[1])
    tile = np.full((3, 3), carrier, dtype=np.int8)
    tile[inward] = sample
    mode = int(rng.integers(5))
    if mode == 0:
        return tile
    if mode == 4:
        tile[2 - inward[0], 2 - inward[1]] = sample
        pairs = [((0, 0), (2, 2)), ((0, 1), (2, 1)),
                 ((0, 2), (2, 0)), ((1, 0), (1, 2))]
        rng.shuffle(pairs)
        for p, q in pairs[:int(rng.integers(3))]:
            if inward not in (p, q):
                tile[p] = tile[q] = rng.choice(pool)
        return tile
    positions = [(r, c) for r in range(3) for c in range(3) if (r, c) != inward]
    rng.shuffle(positions)
    count = int(rng.integers(1, 7))
    if mode == 1:  # line segments, with optional extra marks
        axis, fixed = int(rng.integers(2)), int(rng.integers(3))
        positions.sort(key=lambda p: p[axis] != fixed)
    elif mode == 2:  # a small connected patch from the sample
        cluster = {inward}
        selected = []
        for _ in range(count):
            frontier = sorted({q for r, c in cluster for q in
                               ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                               if 0 <= q[0] < 3 and 0 <= q[1] < 3} - cluster)
            p = frontier[int(rng.integers(len(frontier)))]
            selected.append(p)
            cluster.add(p)
        positions = selected
    local = [sample] + list(rng.choice(pool, size=int(rng.integers(1, len(pool) + 1)), replace=False))
    common = int(rng.choice(local))
    for p in positions[:count]:
        tile[p] = rng.choice(local) if mode == 3 else common
    return tile


def _labels(rng, n, mode):
    if mode == "full":
        labels = [0, 0, 0, 1, 1, 1, 2, 2]
    else:
        low = (n + 2) // 3
        high = n - 1 if mode in ("sparse", "query") else n
        groups = int(rng.integers(low, high + 1))
        labels = list(range(groups))
        while len(labels) < n:
            eligible = [g for g in range(groups) if labels.count(g) < 3]
            labels.append(int(rng.choice(eligible)))
    rng.shuffle(labels)
    return labels


def _make_pair(rng, configuration, mode="ordinary"):
    n = 8 if mode == "full" else int(rng.integers(5, 8) if mode == "sparse" else rng.integers(3, 9))
    directions = list(DIRECTIONS) if n == 8 else [DIRECTIONS[int(i)] for i in
                                                               sorted(rng.choice(8, size=n, replace=False))]
    bg, origin, guide, carrier = map(int, rng.choice(10, size=4, replace=False))
    pool = [c for c in range(10) if c not in (bg, carrier)]
    labels = _labels(rng, n, mode)
    colors = rng.choice(pool, size=max(labels) + 1, replace=False)
    samples = {d: int(colors[label]) for d, label in zip(directions, labels)}
    tiles = {d: (_witness_tile(rng, carrier, samples[d], pool, d) if mode == "full" else
                 _ordinary_tile(rng, carrier, samples[d], pool, d)) for d in directions}
    if mode == "sparse":
        # Repeated color inside a tile is deliberately unrelated to occurrence rank.
        chosen = directions[int(rng.integers(n))]
        inward = (1 - chosen[0], 1 - chosen[1])
        other = [(r, c) for r in range(3) for c in range(3)
                 if (r, c) != inward and tiles[chosen][r, c] != carrier]
        if not other:
            other = [(r, c) for r in range(3) for c in range(3) if (r, c) != inward]
        tiles[chosen][other[int(rng.integers(len(other)))]] = samples[chosen]
    # Plan the ground truth independently of parsing or solve().
    fast, slow = _vectors(configuration)
    planned_order = sorted(directions, key=lambda d: (d[0]*slow[0] + d[1]*slow[1],
                                                      d[0]*fast[0] + d[1]*fast[1]))
    seen, echo_centers = {}, {}
    for d in planned_order:
        color = samples[d]
        seen[color] = seen.get(color, 0) + 1
        echo_centers[d] = (3 * (seen[color] + 1) * d[0], 3 * (seen[color] + 1) * d[1])
    low = [min([-4] + [p[axis] - 1 for p in echo_centers.values()]) for axis in (0, 1)]
    high = [max([4] + [p[axis] + 1 for p in echo_centers.values()]) for axis in (0, 1)]
    interior_shape, hub = [], []
    for axis in (0, 1):
        extent = high[axis] - low[axis] + 1
        extra = int(rng.integers(28 - extent))  # extent+2+extra <= 29
        near = int(rng.integers(extra + 1))
        interior_shape.append(extent + 2 + extra)
        hub.append(1 + near - low[axis])
    axis, bottom, right = configuration
    hub[0] += int(not bottom)
    hub[1] += int(not right)
    y, x = hub
    a = np.full((interior_shape[0] + 1, interior_shape[1] + 1), bg, dtype=np.int8)
    origin_row = a.shape[0] - 1 if bottom else 0
    origin_col = a.shape[1] - 1 if right else 0
    a[origin_row, origin_col] = origin
    sy, sx = (-1 if bottom else 1), (-1 if right else 1)
    for t in range(1, a.shape[1]):
        a[origin_row, origin_col + sx*t] = guide if axis == 0 or t % 2 else bg
    for t in range(1, a.shape[0]):
        a[origin_row + sy*t, origin_col] = guide if axis == 1 or t % 2 else bg
    a[y, x] = carrier
    for (dr, dc), tile in tiles.items():
        r, c = y + 3*dr - 1, x + 3*dc - 1
        a[r:r+3, c:c+3] = tile
    b = a.copy()
    for (dr, dc), tile in tiles.items():
        b[y+dr, x+dc] = samples[(dr, dc)]
        ey, ex = echo_centers[(dr, dc)]
        b[y+ey-1:y+ey+2, x+ex-1:x+ex+2] = np.rot90(tile, 2)
    return {"input": a, "output": b}


_ALTERNATIVES = ("physical", "fast_major", "reverse_fast", "reverse_slow",
                 "one_shell", "total_frequency", "global_index")


def _training_evidence(train):
    infos = [_parse(p["input"]) for p in train]
    for alternative in _ALTERNATIVES:
        _require(any(d["ranks"] != _ranks(d["samples"], d["configuration"], alternative)
                     for d in infos), "unresolved training shortcut: " + alternative)
    positions = {d: set(range(9)) for d in DIRECTIONS}
    operations = {d: set(range(8)) for d in DIRECTIONS}
    for pair, info in zip(train, infos):
        y, x = info["hub"]
        out = _grid(pair["output"])
        for (dr, dc), tile in info["tiles"].items():
            d = (dr, dc)
            positions[d].intersection_update(j for j in range(9)
                                             if tile.flat[j] == out[y+dr, x+dc])
            radius = 3 * (info["ranks"][d] + 1)
            target = out[y+radius*dr-1:y+radius*dr+2, x+radius*dc-1:x+radius*dc+2]
            operations[d].intersection_update(k for k, view in enumerate(_views(tile))
                                              if np.array_equal(view, target))
    for dr, dc in DIRECTIONS:
        _require(positions[(dr, dc)] == {(1-dr)*3 + 1-dc}, "inward sample coordinate is ambiguous")
        _require(operations[(dr, dc)] == {2}, "half-turn is ambiguous with another tile transform")
    _require(any(len(d["tiles"]) < 8 for d in infos), "sparse-ring demonstration missing")
    _require(any(np.count_nonzero(t == d["samples"][direction]) > 1
                 for d in infos for direction, t in d["tiles"].items()), "repeated sample mark missing")
    return infos


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    a, b = _grid(pair["input"]), _grid(pair["output"])
    _require(a.shape == b.shape, "output shape changed")
    d = _parse(a)
    y, x = d["hub"]
    assigned = np.zeros(a.shape, dtype=bool)
    for (dr, dc), rank in d["ranks"].items():
        assigned[y+dr, x+dc] = True
        _require(b[y+dr, x+dc] == a[y+2*dr, x+2*dc], "incorrect inward summary")
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                source = (y + 3*dr + dy, x + 3*dc + dx)
                target = (y + 3*(rank+1)*dr - dy, x + 3*(rank+1)*dc - dx)
                _require(not assigned[target], "destinations overlap")
                _require(a[target] == d["background"], "echo overwrites original data")
                _require(b[target] == a[source], "incorrect echo distance or half-turn")
                assigned[target] = True
    _require(np.all(a[assigned] == d["background"]), "destinations were not blank")
    _require(np.array_equal(a[~assigned], b[~assigned]), "source, key, or unused background changed")
    _require(int(np.count_nonzero(a != b)) == 10 * len(d["tiles"]), "wrong change count")
    _require(np.array_equal(np.unique(a), np.unique(b)), "palette changed")
    return d


def validate_generated_puzzle(puzzle):
    """Check visible geometry, scalar labels, evidence, and puzzle invariants."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "invalid pair counts")
    used, configurations = set(), {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _grid_key(info["grid"])
            _require(key not in used, "duplicate input")
            used.add(key)
            configurations[split].add(info["configuration"])
            if split == "test":
                _require(max(info["ranks"].values()) >= 2, "query must exercise per-color counting")
    _training_evidence(puzzle["train"])
    _require(bool(configurations["test"] - configurations["train"]), "held-out key configuration missing")


def _count(value, low, high, name):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
             and low <= value <= high, "invalid " + name)
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    n, m = _count(num_train, 2, 10, "num_train"), _count(num_test, 1, 4, "num_test")
    _require(n > m, "num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    first = CONFIGURATIONS[int(rng.integers(8))]
    second = (1-first[0], not first[1], not first[2])
    held_options = [c for c in CONFIGURATIONS if c not in (first, second)]
    held = held_options[int(rng.integers(len(held_options)))]
    training_options = [c for c in CONFIGURATIONS if c != held]
    train_configs = [first, second] + [training_options[int(rng.integers(7))] for _ in range(n-2)]
    test_configs = [held] + [CONFIGURATIONS[int(rng.integers(8))] for _ in range(m-1)]
    # Repairs are driven by observable ambiguities, not by the test labels.
    for _ in range(64):
        training = [_make_pair(rng, first, "full"), _make_pair(rng, second, "sparse")]
        try:
            _training_evidence(training)
        except ValueError:
            continue
        break
    else:
        raise RuntimeError("bounded training witness construction exhausted")
    used = {_grid_key(p["input"]) for p in training}
    puzzle = {"train": training, "test": []}
    for split, configs in (("train", train_configs[2:]), ("test", test_configs)):
        for config in configs:
            for _ in range(64):
                pair = _make_pair(rng, config, "query" if split == "test" else "ordinary")
                key = _grid_key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def puzzle_features(puzzle):
    """Small, observable measurements for review; no sampler metadata."""
    rows = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            d = _parse(pair["input"])
            colors = {}
            pattern = tuple(colors.setdefault(d["samples"].get(v), len(colors))
                            if v in d["samples"] else -1 for v in DIRECTIONS)
            rows.append(dict(split=split, shape=d["grid"].shape, hub=d["hub"],
                             configuration=d["configuration"], background=d["background"],
                             origin=d["origin"], guide=d["guide"], carrier=d["carrier"],
                             count=len(d["tiles"]), occupancy=tuple(d["tiles"]),
                             sample_pattern=pattern, ranks=tuple(d["ranks"][v] for v in d["tiles"]),
                             palette_size=len(np.unique(d["grid"])),
                             repeated_marks=sum(np.count_nonzero(t == d["samples"][v]) > 1
                                                for v, t in d["tiles"].items()),
                             symmetric_tiles=sum(np.array_equal(t, t[::-1, ::-1]) for t in d["tiles"].values())))
    return rows


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(_grid(p["input"])), _grid_key(_grid(p["output"]))) for p in puzzle[s])
                 for s in ("train", "test"))


def _expect_invalid(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("invalid case accepted")


def self_test():
    """Bounded 42-seed structural/semantic checks and adversarial regressions."""
    features, keys, structures = [], set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        keys.add(_puzzle_key(p))
        features.extend(puzzle_features(p))
        for split in ("train", "test"):
            for pair in p[split]:
                original = pair["input"].copy()
                assert np.array_equal(solve(original), pair["output"])
                assert np.array_equal(original, pair["input"]), "solver mutated input"
                roles = {}
                structures.add((original.shape, tuple(roles.setdefault(int(v), len(roles)) for v in original.flat)))
        if seed < 3:
            assert _puzzle_key(p) == _puzzle_key(generate_puzzle(seed))
    assert len(keys) == 42 and len(structures) >= 160
    assert len({r["shape"] for r in features}) >= 35
    assert len({(r["shape"], r["hub"]) for r in features}) >= 80
    assert {r["configuration"] for r in features} == set(CONFIGURATIONS)
    assert {r["count"] for r in features} == set(range(3, 9))
    assert len({r["occupancy"] for r in features}) >= 35
    assert len({r["sample_pattern"] for r in features}) >= 80
    assert {k for r in features for k in r["ranks"]} == {1, 2, 3}
    for role in ("background", "origin", "guide", "carrier"):
        assert {r[role] for r in features} == set(range(10))
    assert sum(r["symmetric_tiles"] > 0 for r in features) >= 25
    assert sum(r["repeated_marks"] > 0 for r in features) >= 80
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(10000 + 10*n + m, n, m)
            assert len(p["train"]) == n and len(p["test"]) == m
    for n, m in ((1, 1), (11, 1), (3, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.0, 1), (4, 1.5), (None, 1), (4, None)):
        _expect_invalid(generate_puzzle, 0, n, m)
    for seed in (-1, True, 1.5, None):
        _expect_invalid(generate_puzzle, seed)
    # Literal color roles are inferred locally: any permutation commutes with solve.
    rng = np.random.default_rng(314159)
    p = generate_puzzle(424242, 2, 1)
    for pair in p["train"] + p["test"]:
        mapping = rng.permutation(10)
        assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
    # Input-only geometric equivariance includes bottom origins and reflected keys.
    pair = p["train"][0]
    for k in range(4):
        assert np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k))
        assert np.array_equal(solve(np.rot90(pair["input"][:, ::-1], k)),
                              np.rot90(pair["output"][:, ::-1], k))
    # A border-only intervention changes ranks but not inward sample identities.
    info = _parse(pair["input"])
    assert any(info["ranks"] != _ranks(info["samples"], c) for c in CONFIGURATIONS)
    # A copied color away from the inward coordinate cannot alter any echo radius.
    a = pair["input"].copy()
    info = _parse(a)
    d = next(iter(info["tiles"]))
    dr, dc = d
    # Preserve a carrier at another cell and edit the tile center only.
    a[info["hub"][0]+3*dr, info["hub"][1]+3*dc] = info["samples"][d]
    changed = _parse(a)
    assert changed["ranks"] == info["ranks"] and changed["samples"] == info["samples"]
    _check_pair({"input": a, "output": solve(a)})
    # Corrupt a center sample, an echo, a source, and untouched background.
    for where in ("center", "echo", "source", "unused"):
        broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        target = broken["test"][0]
        d = _parse(target["input"])
        y, x = d["hub"]
        dr, dc = next(iter(d["tiles"]))
        radius = 3 * (d["ranks"][(dr, dc)] + 1)
        point = {"center": (y+dr, x+dc), "echo": (y+radius*dr, x+radius*dc),
                 "source": (y+3*dr, x+3*dc), "unused": (y, x)}[where]
        target["output"][point] = (int(target["output"][point]) + 1) % 10
        _expect_invalid(validate_generated_puzzle, broken)
    damaged = p["test"][0]["input"].copy()
    info = _parse(damaged)
    dr, dc = next(iter(info["tiles"]))
    damaged[info["hub"][0]+3*dr, info["hub"][1]+3*dc] = info["background"]
    _expect_invalid(solve, damaged)
    return {"seeds": 42, "distinct_puzzles": len(keys), "normalized_structures": len(structures),
            "shapes": len({r["shape"] for r in features}),
            "occupancy_patterns": len({r["occupancy"] for r in features}),
            "sample_patterns": len({r["sample_pattern"] for r in features})}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
