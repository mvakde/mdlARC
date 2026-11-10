"""Pulse loom: periodic scan lines define local corners that gather payload.

All transformation conventions are fixed or supplied by the three visible ruler
ticks. No parent generator, example lookup, or generation state is used by solve.
"""

import numpy as np


def _integer(value, name, lo, hi=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a


def _analyze(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 4 or counts.max() * 2 <= a.size:
        raise ValueError("four distinct visible roles and a majority background required")
    bg = int(colors[np.argmax(counts)])
    candidates = []
    for turn in range(4):
        g = np.rot90(a, turn)
        h, w = g.shape
        if h < 5 or w < 5:
            continue
        positions = np.flatnonzero(g[0] != bg)
        if not len(positions):
            continue
        start, end = int(positions[0]), int(positions[-1]) + 1
        if start < 1 or end >= w or not np.array_equal(positions, np.arange(start, end)):
            continue
        marker = int(g[0, start])
        accent = int(g[4, start])
        if marker == bg or accent in (marker, bg):
            continue
        expected_marker = np.zeros(g.shape, dtype=bool)
        expected_accent = np.zeros(g.shape, dtype=bool)
        expected_marker[0, start:end] = True
        expected_marker[2, start:end] = True
        expected_accent[4, start:end] = True
        if not (np.array_equal(g == marker, expected_marker)
                and np.array_equal(g == accent, expected_accent)):
            continue
        rest = set(map(int, colors)) - {bg, marker, accent}
        payload = rest.pop()
        mask = g == payload
        if np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
            continue
        candidates.append(dict(grid=g, turn=turn, background=bg, marker=marker,
                               accent=accent, payload=payload, start=start, end=end,
                               pulses=tuple(range(4, h, 6))))
    if len(candidates) != 1:
        raise ValueError("input must contain one unambiguous three-tick edge ruler")
    return candidates[0]


def _components(cells):
    todo = set(cells)
    result = []
    while todo:
        root = min(todo)
        todo.remove(root)
        stack, found = [root], []
        while stack:
            r, c = stack.pop()
            found.append((r, c))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
        result.append(found)
    return result


def _groups(info, *, tie_far=False, single_pulse=False):
    """Partition original off-strip, off-pulse payload, including its fragments."""
    groups = {}
    pulses = info["pulses"]
    for rr, cc in np.argwhere(info["grid"] == info["payload"]):
        r, c = int(rr), int(cc)
        if info["start"] <= c < info["end"] or r in pulses:
            continue
        p = pulses[0] if single_pulse else min(
            pulses, key=lambda v: (abs(r - v), -v if tie_far else v))
        key = (p, r < p, c < info["start"])
        groups.setdefault(key, []).append((r, c))
    return groups


def _shift(info, key, cells):
    p, above, left = key
    rs, cs = zip(*cells)
    dr = p - 1 - max(rs) if above else p + 1 - min(rs)
    dc = info["start"] - 1 - max(cs) if left else info["end"] - min(cs)
    return dr, dc


def _render(info, mode="correct"):
    """Correct painter plus explicitly named incomplete rules for witness checks."""
    g = info["grid"]
    bg, mark, high, obj = (info[k] for k in ("background", "marker", "accent", "payload"))
    start, end = info["start"], info["end"]
    out = np.full(g.shape, bg, dtype=np.int8)
    for d in range(len(g)):
        out[d, start:end] = bg if d % 2 else high if d % 6 == 4 else mark
        if d in info["pulses"]:
            out[d, (g[d] == obj) & ((np.arange(g.shape[1]) < start)
                                   | (np.arange(g.shape[1]) >= end))] = high
    groups = _groups(info, tie_far=mode == "tie_far", single_pulse=mode == "single_pulse")
    for key, cells in groups.items():
        pieces = _components(cells) if mode == "per_component" else [cells]
        for piece in pieces:
            dr, dc = (0, 0) if mode == "no_gather" else _shift(info, key, piece)
            for r, c in piece:
                out[r + dr, c + dc] = obj
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique family output using only visible input pixels."""
    info = _analyze(input_grid)
    return np.rot90(_render(info), -info["turn"]).copy()


def _features(value):
    info = _analyze(value)
    g, start, end = info["grid"], info["start"], info["end"]
    mask = g == info["payload"]
    groups = _groups(info)
    correct = _render(info)
    moved_quadrants = set()
    fragmented = 0
    moves = []
    for key, cells in groups.items():
        delta = _shift(info, key, cells)
        if delta != (0, 0):
            moved_quadrants.add(key[1:])
        fragmented += len(_components(cells)) > 1
        moves.append(delta)
    strip = np.zeros_like(mask)
    strip[:, start:end] = True
    d = np.arange(len(g))[:, None]
    high = np.broadcast_to(d % 6 == 4, g.shape)
    labels = {cell: key for key, cells in groups.items() for cell in cells}
    split_object = any(len({labels.get(p, "fixed_or_erased") for p in comp}) > 1
                       for comp in _components(map(tuple, np.argwhere(mask))))
    return dict(
        shape=tuple(np.asarray(value).shape), depth=len(g), width=g.shape[1],
        turn=info["turn"], bar_length=end-start,
        roles=tuple(info[k] for k in ("background", "marker", "accent", "payload")),
        pulses=len(info["pulses"]), payload_cells=int(mask.sum()),
        groups=len(groups), fragmented_groups=fragmented, moved_quadrants=moved_quadrants,
        split_object=split_object, group_sizes=tuple(len(v) for v in groups.values()),
        strip_odd=bool(np.any(mask & strip & (d % 2 == 1))),
        strip_ordinary=bool(np.any(mask & strip & (d > 4) & ((d % 6 == 0) | (d % 6 == 2)))),
        strip_special=bool(np.any(mask & strip & high)),
        remote_highlight=bool(np.any(mask & ~strip & high)),
        highlight_gap=bool(np.any((g == info["background"]) & ~strip & high)),
        mutants={mode: not np.array_equal(correct, _render(info, mode))
                 for mode in ("no_gather", "single_pulse", "per_component", "tie_far")})


def _teaches(records):
    return (all(any(f[k] for f in records) for k in
                ("strip_odd", "strip_ordinary", "strip_special", "remote_highlight",
                 "highlight_gap", "split_object"))
            and set().union(*(f["moved_quadrants"] for f in records))
            == {(a, b) for a in (False, True) for b in (False, True)}
            and all(any(f["mutants"][m] for f in records) for m in
                    ("no_gather", "single_pulse", "per_component", "tie_far")))


def _grow(rng, allowed, count, thin=False):
    points = np.argwhere(allowed)
    if not len(points):
        return set()
    cells = {tuple(map(int, points[int(rng.integers(len(points)))]))}
    for _ in range(count - 1):
        frontier = sorted({q for r, c in cells
                           for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                           if 0 <= q[0] < len(allowed) and 0 <= q[1] < allowed.shape[1]
                           and allowed[q] and q not in cells})
        if not frontier:
            break
        if thin:
            degrees = [sum(p in cells for p in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)))
                       for r,c in frontier]
            best = min(degrees)
            frontier = [q for q, degree in zip(frontier, degrees) if degree == best]
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return cells


def _scene(rng, depth, width, start, end, kind=None):
    allowed = np.ones((depth, width), dtype=bool)
    allowed[0] = allowed[-1] = False
    allowed[:, 0] = allowed[:, -1] = False
    allowed[2, start:end] = allowed[4, start:end] = False
    mask = np.zeros_like(allowed)
    kind = int(rng.integers(5)) if kind is None else kind
    for _ in range(int(rng.integers(2, 11))):
        if kind < 2:
            count = int(rng.integers(3, max(4, min(65, depth * width // 5))))
            for p in _grow(rng, allowed, count, thin=kind == 1):
                mask[p] = True
        else:
            h = int(rng.integers(1, min(10, depth - 2) + 1))
            w = int(rng.integers(1, min(10, width - 2) + 1))
            r, c = int(rng.integers(1, depth-h)), int(rng.integers(1, width-w))
            local = np.ones((h, w), dtype=bool)
            if kind == 2:
                # A changing row profile rather than a bank of fixed shapes.
                for row in range(h):
                    a = int(rng.integers(w))
                    b = int(rng.integers(a + 1, w + 1))
                    local[row, :a] = local[row, b:] = False
            elif kind == 3 and h >= 3 and w >= 3:
                local[1:-1, 1:-1] = rng.random((h-2, w-2)) < .25
                if rng.random() < .5:
                    local[-1, w//2] = False
            elif kind == 4:
                local &= rng.random((h, w)) < float(rng.uniform(.25, .7))
            mask[r:r+h, c:c+w] |= local
    mask &= allowed
    return mask


def _fallback_mask(rng, depth):
    """A finite emergency witness, used only after natural proposals fail.

    Two variable-width ribbons and a joining arm, with one gap between fragments.
    This is not inserted into ordinary scenes; all normal shapes precede checks.
    """
    width = int(rng.integers(22, 31))
    start = int(rng.integers(7, 11))
    end = start + int(rng.integers(1, 6))
    mask = np.zeros((depth, width), dtype=bool)
    for a, b in ((1, int(rng.integers(3, 5))),
                 (width-int(rng.integers(3, 5)), width-1)):
        mask[1:13, a:b] = True
        mask[6, a:b] = False
    mask[3, 1:width-1] = True
    mask[5:7, start:end] = True
    mask[10, start:end] = True
    # An unequal lateral extent at a midpoint makes the tie convention observable.
    mask[7, start-3:start-1] = True
    return width, start, end, mask


def _draw(rng, depth, roles, turn, fallback=False):
    if fallback:
        width, start, end, mask = _fallback_mask(rng, depth)
    else:
        width = int(rng.integers(9, 31))
        length = int(rng.integers(1, min(9, width - 4) + 1))
        start = int(rng.integers(2, width - length - 1))
        end = start + length
        mask = _scene(rng, depth, width, start, end)
    bg, mark, high, obj = roles
    g = np.full(mask.shape, bg, dtype=np.int8)
    g[mask] = obj
    g[0, start:end] = g[2, start:end] = mark
    g[4, start:end] = high
    g = np.rot90(g, -turn).copy()
    try:
        info = _analyze(g)
    except ValueError:
        return None
    return {"input": g, "output": np.rot90(_render(info), -turn).copy()}


def _key(g):
    a = np.asarray(g)
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    ns = _integer(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    varies = rng.random() < .65
    base = int(rng.integers(4))
    regime = int(rng.integers(4))  # familiar, reverse, quarter-turn, mixed
    puzzle, seen, teaching = {"train": [], "test": []}, set(), []
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            if varies:
                roles = tuple(map(int, rng.choice(10, 4, replace=False)))
            if split == "train":
                turn = (base + (index % 2 if regime == 3 else 0)) % 4
                depth = int(rng.integers(14, 17))  # exactly two special pulses
            else:
                turn = (base + (0, 2, 1, 2 + index % 2)[regime]) % 4
                depth = int(rng.integers(20, 31))  # three to five pulses
            last_train = split == "train" and index == count - 1
            for attempt in range(96):
                pair = _draw(rng, depth, roles, turn, fallback=attempt >= 80)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                f = _features(pair["input"]) if split == "train" else None
                if last_train and not _teaches(teaching + [f]):
                    continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                if f is not None:
                    teaching.append(f)
                break
            else:
                raise RuntimeError("finite distinct-scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, transport geometry, conservation, and demonstration witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and not np.array_equal(a, b)
            assert _key(a) not in seen
            seen.add(_key(a))
            info = _analyze(a)
            g = info["grid"]
            out = np.rot90(b, info["turn"])
            assert np.array_equal(solve(a.copy()), b)
            groups = _groups(info)
            occupied = set()
            for key, cells in groups.items():
                dr, dc = _shift(info, key, cells)
                targets = {(r+dr, c+dc) for r, c in cells}
                assert len(targets) == len(cells) and not (targets & occupied)
                occupied |= targets
                p, above, left = key
                rs, cs = zip(*targets)
                assert (max(rs) == p-1 if above else min(rs) == p+1)
                assert (max(cs) == info["start"]-1 if left else min(cs) == info["end"])
                for r, c in targets:
                    assert 0 <= r < len(g) and 0 <= c < g.shape[1]
                    assert r not in info["pulses"]
                    assert c < info["start"] or c >= info["end"]
                    assert min(info["pulses"], key=lambda v: (abs(r-v), v)) == p
                    assert out[r, c] == info["payload"]
            assert int(np.sum(out == info["payload"])) == sum(map(len, groups.values()))
            for p in info["pulses"]:
                for c in list(range(info["start"])) + list(range(info["end"], g.shape[1])):
                    assert out[p, c] == (info["accent"] if g[p, c] == info["payload"]
                                         else info["background"])
            features[split].append(_features(a))
    assert _teaches(features["train"]), "demonstrations fail to expose a necessary interaction"
    assert all(f["pulses"] > max(t["pulses"] for t in features["train"])
               for f in features["test"]), "missing pulse-count extrapolation"


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in a)


def self_test():
    """Bounded checks across 42 seeds, every legal count, and malformed requests."""
    records, structures = [], set()
    for seed in range(42):
        nt = 2 + seed % 9
        ns = 1 + (seed // 3) % min(4, nt - 1)
        p = generate_puzzle(seed, nt, ns)
        for split in p:
            for pair in p[split]:
                f = _features(pair["input"])
                records.append((split, f))
                g = _analyze(pair["input"])["grid"]
                roles = {}
                structures.add((g.shape, tuple(roles.setdefault(int(c), len(roles)) for c in g.flat)))
    assert _equal(generate_puzzle(173, 7, 3), generate_puzzle(173, 7, 3))
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1) + 1):
            p = generate_puzzle(7300 + nt*7 + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(2.5,1),(4,None)):
        try:
            generate_puzzle(3, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    for split in ("train", "test"):
        fs = [f for s, f in records if s == split]
        assert {f["turn"] for f in fs} == set(range(4))
        assert sum(f["fragmented_groups"] > 0 for f in fs) >= 5
        assert {np.sign(f["shape"][0]-f["shape"][1]) for f in fs} >= {-1, 1}
    fs = [f for _, f in records]
    assert len(structures) >= len(records) * .95
    assert len({f["shape"] for f in fs}) >= 60
    assert {f["bar_length"] for f in fs} >= {1, 2, 3, 4, 5, 6}
    assert {f["pulses"] for f in fs} == {2, 3, 4, 5}
    assert all({f["roles"][i] for f in fs} == set(range(10)) for i in range(4))
    assert len({f["payload_cells"] for f in fs}) > 60
    assert len({f["groups"] for f in fs}) >= 8
    # Rotation equivariance, including rectangular grids, is checked from pixels.
    a = generate_puzzle(0)["test"][0]["input"]
    for k in range(4):
        assert np.array_equal(solve(np.rot90(a, k)), np.rot90(solve(a), k))
    # The input-only rule also handles one pulse and a truncated final basin.
    for depth in (5, 6, 10, 11, 17, 30):
        g = np.zeros((depth, 11), dtype=np.int8)
        g[0, 5] = g[2, 5] = 1
        g[4, 5] = 2
        g[1, 2] = g[depth-2, 8] = 3
        for k in range(4):
            out = solve(np.rot90(g, k))
            assert out.shape == np.rot90(g, k).shape
    # Collective extrema can align to a corner that remains unoccupied.
    g = np.zeros((10, 11), dtype=np.int8)
    g[0, 5] = g[2, 5] = 1
    g[4, 5] = 2
    g[1, 2] = g[2, 1] = g[4, 1] = 3
    out = solve(g)
    assert set(map(tuple, np.argwhere(out == 3))) == {(2, 4), (3, 3)}
    assert out[3, 4] == 0 and out[4, 1] == 2
    # A payload entirely inside the strip is overwritten, with no packet left.
    g[g == 3] = 0
    g[5, 5] = 3
    out = solve(g)
    assert not np.any(out == 3) and out[5, 5] == 0
    for depth in (14, 15, 16):
        for turn in range(4):
            for variant in range(2):
                pair = _draw(np.random.default_rng(variant), depth, (0, 1, 2, 3),
                             turn, fallback=True)
                assert pair is not None and _teaches([_features(pair["input"])])
    p = generate_puzzle(21, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("pulse loom self_test passed (42 seeds and all legal counts)")
