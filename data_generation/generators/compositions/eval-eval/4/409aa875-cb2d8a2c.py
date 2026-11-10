"""Twelve-step pulses: unary detours consume travel, then endpoints interact.

Only NumPy and the Python standard library are used. Geometry and color roles
are decoded afresh from the queried input. No parent code or fixtures are used.
"""
from collections import defaultdict
from itertools import combinations

import numpy as np


BUDGET = 12


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, pending = {start}, [start]
        while pending:
            r, c = pending.pop()
            neighbors = {(r + dr, c + dc) for dr in (-1, 0, 1)
                         for dc in (-1, 0, 1)} & remaining
            remaining -= neighbors
            group |= neighbors
            pending.extend(neighbors)
        result.append(frozenset(group))
    return result


def _arrow_cells(tip, vector):
    r, c = tip
    dr, dc = vector
    offsets = ((0, 0), (-dr, -1), (-dr, 1)) if dc == 0 else (
        (0, 0), (-1, -dc), (1, -dc))
    return frozenset((r + a, c + b) for a, b in offsets)


def _separated(a, b):
    return all(max(abs(r - s), abs(c - t)) > 1
               for r, c in a for s, t in b)


def _decode(input_grid):
    """Return canonical horizontal bars and vertical arrows, from pixels only."""
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > grid.size / 2, "background must be a strict majority"
    groups = _components(map(tuple, np.argwhere(grid != bg).tolist()))
    arrow_groups, bar_groups = [], []
    vectors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for group in groups:
        colors = {int(grid[p]) for p in group}
        if len(group) == 3 and len(colors) == 1:
            matches = [(p, v) for p in sorted(group) for v in vectors
                       if _arrow_cells(p, v) == group]
            assert len(matches) == 1, "a three-cell component must be a cardinal arrow"
            arrow_groups.append((matches[0], group, next(iter(colors))))
        else:
            assert len(group) >= 3 and len(colors) == 2
            rows, cols = {r for r, _ in group}, {c for _, c in group}
            assert (len(rows) == 1) != (len(cols) == 1), "bars must be straight"
            assert len(group) == max(len(rows), len(cols)), "bars must be contiguous"
            bar_groups.append(group)
    assert arrow_groups and bar_groups, "need both arrows and marked bars"
    horizontal = len({r for r, _ in bar_groups[0]}) == 1
    assert all((len({r for r, _ in b}) == 1) == horizontal for b in bar_groups)
    a = grid if horizontal else grid.T.copy()
    bars, permanent_colors, marker_colors = [], set(), set()
    for group in bar_groups:
        cells = group if horizontal else frozenset((c, r) for r, c in group)
        f = next(iter(cells))[0]
        lo, hi = min(c for _, c in cells), max(c for _, c in cells)
        line = a[f, lo:hi + 1]
        colors, nums = np.unique(line, return_counts=True)
        assert nums[0] != nums[1], "permanent body has a strict local majority"
        body, marker = map(int, colors[np.argsort(nums)[::-1]])
        positions = tuple(map(int, np.flatnonzero(line == marker) + lo))
        n = len(positions)
        assert 1 <= n <= 4 and hi - lo + 1 >= 2 * n + 1
        left = positions == tuple(lo + 2 * j for j in range(n))
        right = positions == tuple(hi - 2 * j for j in reversed(range(n)))
        assert left != right, "one marked end, with a permanent tail"
        side = 1 if left else -1
        exit_c = hi + n + 1 if left else lo - n - 1
        assert 0 <= exit_c < a.shape[1], "commanded bypass must fit"
        assert 0 < f < a.shape[0] - 1
        bars.append((f, lo, hi, n, side, exit_c))
        permanent_colors.add(body)
        marker_colors.add(marker)
    bars.sort()
    assert len(permanent_colors) == len(marker_colors) == 1
    body, marker = permanent_colors.pop(), marker_colors.pop()
    arrow_colors = {color for _, _, color in arrow_groups}
    assert len(arrow_colors) == 1
    arrow_color = arrow_colors.pop()
    assert len({bg, body, marker, arrow_color}) == 4
    assert 1 <= len(bars) <= 4
    for first, second in zip(bars, bars[1:]):
        assert second[0] - first[0] >= max(first[3], second[3]) + 2
    arrows = []
    for (tip, vector), cells, _ in arrow_groups:
        if not horizontal:
            tip, vector = tip[::-1], vector[::-1]
            cells = frozenset((c, r) for r, c in cells)
        assert vector[1] == 0, "all launches are transverse to the bars"
        arrows.append((tip, vector[0], cells))
    arrows.sort(key=lambda obj: (obj[0], obj[1]))
    return a, horizontal, (bg, arrow_color, body, marker), bars, arrows


def _trace(tip, sign, bars, shape):
    """Spend exactly twelve orthogonal steps; no arrow can block a pulse."""
    r, c = tip
    path, commands = [(r, c)], []
    occupied = {(f, j) for f, lo, hi, *_ in bars for j in range(lo, hi + 1)}

    def move(target):
        nonlocal r, c
        tr, tc = target
        assert r == tr or c == tc
        dr, dc = int(np.sign(tr - r)), int(np.sign(tc - c))
        while (r, c) != target and len(path) <= BUDGET:
            r, c = r + dr, c + dc
            if not (0 <= r < shape[0] and 0 <= c < shape[1]):
                return False
            if (r, c) in occupied:
                return False
            path.append((r, c))
        return True

    while len(path) <= BUDGET:
        ahead = [(sign * (b[0] - r), i, b) for i, b in enumerate(bars)
                 if sign * (b[0] - r) > 0 and b[1] <= c <= b[2]]
        if not ahead:
            if not move((r + sign * (BUDGET + 1 - len(path)), c)):
                return None
            break
        distance, index, bar = min(ahead)
        f, _, _, n, side, exit_c = bar
        d = n + 1
        if distance < d:
            return None  # Launching after the required turn is outside the family.
        turn = f - sign * d
        if not move((turn, c)):
            return None
        if len(path) > BUDGET:
            break
        start = (r, c)
        old_len = len(path)
        if not move((r, exit_c)):
            return None
        commands.append((index, start, (r, exit_c), len(path) - old_len,
                         c == exit_c))
    assert len(path) == BUDGET + 1
    return tuple(path), tuple(commands)


def _events(arrows, endpoints):
    owner = {p: i for i, arrow in enumerate(arrows) for p in arrow[2]}
    assert len(owner) == 3 * len(arrows)
    counts = defaultdict(int)
    for endpoint in endpoints:
        counts[endpoint] += 1
    assignments = defaultdict(set)
    for p, count in counts.items():
        if p in owner:
            assignments[owner[p]].add(count > 1)
    assert all(len(roles) == 1 for roles in assignments.values()), \
        "different endpoints cannot assign incompatible colors to one arrow"
    return owner, dict(counts), dict(assignments)


def _paint(a, palette, bars, arrows, endpoints, *, collisions=True, whole=True):
    _, _, body, marker = palette
    owner, counts, _ = _events(arrows, endpoints)
    out = a.copy()
    out[out == marker] = body
    for point, count in counts.items():
        color = marker if collisions and count > 1 else body
        cells = arrows[owner[point]][2] if whole and point in owner else (point,)
        for cell in cells:
            out[cell] = color
    return out


def _analysis(input_grid):
    a, horizontal, palette, bars, arrows = _decode(input_grid)
    paths, commands, endpoints = [], [], []
    for tip, sign, _ in arrows:
        flight = _trace(tip, sign, bars, a.shape)
        assert flight is not None, "pulse leaves the grid or cannot obey a command"
        path, used = flight
        paths.append(path)
        commands.append(used)
        endpoints.append(path[-1])
    owner, counts, assignments = _events(arrows, endpoints)
    return {"grid": a, "horizontal": horizontal, "palette": palette, "bars": bars,
            "arrows": arrows, "paths": paths, "commands": commands,
            "endpoints": endpoints, "owner": owner, "counts": counts,
            "assignments": assignments}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver. Twelve is a fixed family convention, not latent state."""
    f = _analysis(input_grid)
    out = _paint(f["grid"], f["palette"], f["bars"], f["arrows"], f["endpoints"])
    return out.copy() if f["horizontal"] else out.T.copy()


def _candidates(bars, shape):
    h, w = shape
    blocked = {(f + dr, c + dc) for f, lo, hi, *_ in bars
               for c in range(lo, hi + 1) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    candidates = []
    for r in range(h):
        for c in range(1, w - 1):
            for sign in (-1, 1):
                tip = (r, c)
                cells = _arrow_cells(tip, (sign, 0))
                if any(not (0 <= x < h and 0 <= y < w) or (x, y) in blocked
                       for x, y in cells):
                    continue
                flight = _trace(tip, sign, bars, shape)
                if flight is None:
                    continue
                path, commands = flight
                candidates.append({"arrow": (tip, sign, cells), "end": path[-1],
                                   "path": path, "commands": commands,
                                   "naive": (r + sign * BUDGET, c)})
    return candidates


def _safe_add(selected, candidate):
    if not all(_separated(candidate["arrow"][2], obj["arrow"][2]) for obj in selected):
        return False
    try:
        _events([obj["arrow"] for obj in selected + [candidate]],
                [obj["end"] for obj in selected + [candidate]])
    except AssertionError:
        return False
    return True


def _choose_order(rng, values):
    return [values[int(i)] for i in rng.permutation(len(values))]


def _sample_bars(rng, ns, both_sides=False):
    ns = [int(n) for n in rng.permutation(ns)]
    gaps = [max(a, b) + 2 for a, b in zip(ns, ns[1:])]
    h = int(rng.integers(max(17, sum(gaps) + 6), 31))
    w = int(rng.integers(max(11, 3 * max(ns) + 3), 31))
    if rng.random() < .18:
        h = w = max(h, w)
    slack = h - sum(gaps) - 5
    allocation = rng.multinomial(slack, np.full(len(ns) + 1, 1 / (len(ns) + 1)))
    f = 2 + int(allocation[0])
    sides = [int(rng.choice((-1, 1))) for _ in ns]
    if both_sides and len(ns) >= 2:
        sides[0], sides[1] = -1, 1
    bars = []
    for i, (n, side) in enumerate(zip(ns, sides)):
        d = n + 1
        maximum = w - d
        if rng.random() < .45:
            maximum = min(maximum, 2 * n + 5)
        length = int(rng.integers(2 * n + 1, maximum + 1))
        if side == 1:
            lo = int(rng.integers(0, w - d - length + 1))
        else:
            lo = int(rng.integers(d, w - length + 1))
        hi = lo + length - 1
        exit_c = hi + d if side == 1 else lo - d
        bars.append((f, lo, hi, n, side, exit_c))
        if i < len(gaps):
            f += gaps[i] + int(allocation[i + 1])
    return bars, (h, w)


def _witness(rng, candidates, mode, shape):
    h, w = shape
    if mode == "collision":
        grouped = defaultdict(list)
        for obj in candidates:
            # This collision is a funnel of PARALLEL arrows with valid, different
            # straight-shot endpoints. It cannot be explained by parent A alone.
            if obj["commands"] and 0 <= obj["naive"][0] < h:
                grouped[(obj["end"], obj["arrow"][1])].append(obj)
        options = [(a, b) for group in grouped.values() for a, b in combinations(group, 2)
                   if _safe_add([a], b)]
        if not options:
            return None
        return list(options[int(rng.integers(len(options)))])
    if mode == "strike":
        at_cell = defaultdict(list)
        for target in candidates:
            for cell in target["arrow"][2]:
                at_cell[cell].append(target)
        options = []
        for shooter in candidates:
            if not shooter["commands"]:
                continue
            for target in at_cell.get(shooter["end"], []):
                if not _safe_add([shooter], target):
                    continue
                if target["end"] in shooter["arrow"][2] or target["end"] == shooter["end"]:
                    continue
                options.append((shooter, target))
        if not options:
            return None
        return list(options[int(rng.integers(len(options)))])
    gated = [obj for obj in candidates if obj["commands"]]
    if not gated:
        return None
    return [gated[int(rng.integers(len(gated)))]]


def _make_pair(rng, palette, ns, mode, teach=False):
    for _ in range(100):
        bars, shape = _sample_bars(rng, ns, both_sides=teach)
        candidates = _candidates(bars, shape)
        selected = _witness(rng, candidates, mode, shape)
        if selected is None:
            continue
        ordered = _choose_order(rng, candidates)
        strike_endpoint = selected[0]["end"] if mode == "strike" else None
        direct = None

        def preserve_witnesses(obj):
            if strike_endpoint is not None and obj["end"] == strike_endpoint:
                return False
            if direct is not None:
                if (obj["end"] == direct["end"] or obj["end"] in direct["arrow"][2]
                        or direct["end"] in obj["arrow"][2]):
                    return False
            return True

        if teach:
            # Effective commands, not merely visible unused bars, teach every
            # unary count and both endpoint meanings in a minimal-training task.
            obligations = [("n", n) for n in sorted(set(ns))] + [("side", -1), ("side", 1)]
            failed = False
            for kind, value in obligations:
                def covers(obj):
                    return any(bars[cmd[0]][3 if kind == "n" else 4] == value
                               for cmd in obj["commands"])
                if any(covers(obj) for obj in selected):
                    continue
                extra = next((obj for obj in ordered if covers(obj) and
                              preserve_witnesses(obj) and _safe_add(selected, obj)), None)
                if extra is None:
                    failed = True
                    break
                selected.append(extra)
            if failed:
                continue
            # An unobstructed endpoint twelve cells from its tip teaches the
            # flight budget independently of routing and collision effects.
            direct = next((obj for obj in ordered if not obj["commands"] and
                           obj["end"] == obj["naive"] and _safe_add(selected, obj) and
                           obj["end"] not in {s["end"] for s in selected} and
                           all(obj["end"] not in s["arrow"][2] for s in selected) and
                           all(s["end"] not in obj["arrow"][2] for s in selected)), None)
            if direct is None:
                continue
            selected.append(direct)
        desired = int(rng.integers(max(1, len(selected)), max(2, len(selected) + 5)))
        # Occasionally add another converging shot or a target at a collision.
        if mode == "collision" and rng.random() < .45:
            endpoint = selected[0]["end"]
            specials = [obj for obj in ordered if obj["end"] == endpoint or
                        endpoint in obj["arrow"][2]]
            for obj in specials:
                if preserve_witnesses(obj) and _safe_add(selected, obj):
                    selected.append(obj)
                    break
        for obj in ordered:
            if len(selected) >= desired:
                break
            if preserve_witnesses(obj) and _safe_add(selected, obj):
                selected.append(obj)
        bg, arrow, body, marker = palette
        a = np.full(shape, bg, dtype=np.int8)
        for f, lo, hi, n, side, _ in bars:
            a[f, lo:hi + 1] = body
            for j in range(n):
                a[f, lo + 2 * j if side == 1 else hi - 2 * j] = marker
        for obj in selected:
            for cell in obj["arrow"][2]:
                a[cell] = arrow
        if np.count_nonzero(a == bg) <= a.size / 2:
            continue
        b = _paint(a, palette, bars, [s["arrow"] for s in selected], [s["end"] for s in selected])
        return {"input": a, "output": b}
    raise RuntimeError("bounded search could not construct the requested witnesses")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must be in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    maximum = int(rng.integers(2, 5))
    train_axis = int(rng.integers(2))
    specs = [("collision", list(range(1, maximum + 1)), True),
             ("strike", list(map(int, rng.integers(1, maximum + 1, size=int(rng.integers(1, 5))))), True)]
    # The first teaching scene already supplies every count and both ends;
    # the second still supplies clear budget evidence but need not repeat all ends.
    if len(specs[1][1]) == 1:
        specs[1] = ("strike", specs[1][1], False)
    for _ in range(num_train - 2):
        ns = list(map(int, rng.integers(1, maximum + 1, size=int(rng.integers(1, 5)))))
        specs.append((str(rng.choice(("ordinary", "collision", "strike"))), ns, False))
    puzzle, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        local_specs = specs if split == "train" else [
            (str(rng.choice(("ordinary", "collision", "strike"))),
             list(map(int, rng.integers(1, maximum + 1, size=int(rng.integers(1, 5))))), False)
            for _ in range(num_test)]
        for mode, ns, teach in local_specs:
            for _ in range(12):
                pair = _make_pair(rng, palette, ns, mode, teach)
                # All training bars share an axis, both headings are available.
                # Every query uses the perpendicular axis: a spatial, not color,
                # shift independent of obstacle and interaction counts.
                turns = train_axis + (1 if split == "test" else 0) + 2 * int(rng.integers(2))
                reflect = bool(rng.integers(2))
                for role in pair:
                    a = np.rot90(pair[role], turns)
                    pair[role] = (a[:, ::-1] if reflect else a).copy()
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not generate distinct inputs")
        puzzle[split] = _choose_order(rng, puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(f):
    bars, arrows = f["bars"], f["arrows"]
    used = {cmd[0] for commands in f["commands"] for cmd in commands}
    direct = any(not commands and endpoint == (tip[0] + sign * BUDGET, tip[1])
                 and endpoint not in f["owner"] and f["counts"][endpoint] == 1
                 for (tip, sign, _), commands, endpoint in
                 zip(arrows, f["commands"], f["endpoints"]))
    return {"shape": f["grid"].shape, "axis": f["horizontal"],
            "arrows": len(arrows), "bars": len(bars),
            "counts": {b[3] for b in bars}, "used_counts": {bars[i][3] for i in used},
            "used_sides": {bars[i][4] for i in used}, "direct": direct,
            "collision": any(n > 1 for n in f["counts"].values()),
            "single_strike": any(p in f["owner"] and n == 1 for p, n in f["counts"].items()),
            "collision_strike": any(p in f["owner"] and n > 1 for p, n in f["counts"].items()),
            "max_convergence": max(f["counts"].values()),
            "partial": any(not cmd[4] for commands in f["commands"] for cmd in commands),
            "multi_gate": any(len(commands) >= 2 for commands in f["commands"]),
            "repeated_side": any(bars[a[0]][4] == bars[b[0]][4]
                                 for commands in f["commands"] for a, b in zip(commands, commands[1:])),
            "opposing": len({s for _, s, _ in arrows}) == 2,
            "crossing": any(set(a[1:-1]) & set(b[1:-1]) and ea != eb
                            for i, (a, ea) in enumerate(zip(f["paths"], f["endpoints"]))
                            for b, eb in zip(f["paths"][:i], f["endpoints"][:i]))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, palettes = set(), set()
    analyses, features = {"train": [], "test": []}, {"train": [], "test": []}
    funnel_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape
            assert _key(a) not in seen
            seen.add(_key(a))
            f = _analysis(a)
            palettes.add(f["palette"])
            analyses[split].append(f)
            features[split].append(_features(f))
            assert np.array_equal(solve(a), b), "incorrect output"
            # Budget, monotonic heading, route avoidance and snapshot semantics.
            occupied = {(r, c) for r, lo, hi, *_ in f["bars"] for c in range(lo, hi + 1)}
            for (tip, sign, _), path in zip(f["arrows"], f["paths"]):
                assert path[0] == tip and len(path) == BUDGET + 1
                assert not occupied.intersection(path)
                assert all(abs(r - s) + abs(c - t) == 1 and sign * (s - r) >= 0
                           for (r, c), (s, t) in zip(path, path[1:]))
                lateral = sum(c != t for (_, c), (_, t) in zip(path, path[1:]))
                assert sign * (path[-1][0] - tip[0]) + lateral == BUDGET
            if split == "train":
                for i, endpoint in enumerate(f["endpoints"]):
                    for j in range(i):
                        ti, si, _ = f["arrows"][i]
                        tj, sj, _ = f["arrows"][j]
                        if endpoint == f["endpoints"][j] and si == sj and f["commands"][i] and f["commands"][j]:
                            if all(0 <= t[0] + s * BUDGET < f["grid"].shape[0]
                                   for t, s in ((ti, si), (tj, sj))):
                                funnel_witness = True
    assert len(palettes) == 1, "one shared structurally identified palette per puzzle"
    train = features["train"]
    assert any(f["direct"] for f in train), "need a direct distance witness"
    assert funnel_witness, "need a detour-induced parallel collision"
    assert any(f["single_strike"] for f in train), "teach whole-arrow recoloring"
    assert {s for f in train for s in f["used_sides"]} == {-1, 1}
    taught_counts = {n for f in train for n in f["used_counts"]}
    assert len(taught_counts) >= 2
    assert {n for f in features["test"] for n in f["counts"]} <= taught_counts
    assert len({f["axis"] for f in train}) == 1
    assert {f["axis"] for f in features["test"]}.isdisjoint({f["axis"] for f in train})
    # Parent A's collision and whole-object effects each have a visible witness.
    assert any(not np.array_equal(_paint(f["grid"], f["palette"], f["bars"], f["arrows"],
                                       f["endpoints"], collisions=False),
                                 _paint(f["grid"], f["palette"], f["bars"], f["arrows"], f["endpoints"]))
               for f in analyses["train"])
    assert any(not np.array_equal(_paint(f["grid"], f["palette"], f["bars"], f["arrows"],
                                       f["endpoints"], whole=False),
                                 _paint(f["grid"], f["palette"], f["bars"], f["arrows"], f["endpoints"]))
               for f in analyses["train"])


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed variation audit plus literal semantics and mutations."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    records, palettes, normalized = [], set(), set()
    for puzzle in puzzles:
        for split in puzzle:
            for pair in puzzle[split]:
                f = _analysis(pair["input"])
                record = _features(f)
                record["split"] = split
                records.append(record)
                palettes.add(f["palette"])
                semantic = np.empty_like(f["grid"])
                for i, color in enumerate(f["palette"]):
                    semantic[f["grid"] == color] = i
                normalized.add(_key(semantic))
    assert len(palettes) >= 30 and len(normalized) >= 180
    assert len({f["shape"] for f in records}) >= 60
    assert {f["bars"] for f in records} == {1, 2, 3, 4}
    assert {n for f in records for n in f["used_counts"]} == {1, 2, 3, 4}
    assert max(f["arrows"] for f in records) >= 8
    for name in ("partial", "multi_gate", "repeated_side", "opposing", "crossing", "collision_strike"):
        assert any(f[name] for f in records), name
    for split in ("train", "test"):
        subset = [f for f in records if f["split"] == split]
        assert {f["axis"] for f in subset} == {False, True}
        assert any(f["collision"] for f in subset) and any(not f["collision"] for f in subset)
        assert any(f["single_strike"] for f in subset)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(142 + a * 10 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Literal one-gate flight: start (2,5), bar row 6 span 4..8, one
    # left marker -> turn (4,5), exit column 10; 2+5+5 steps -> (9,10).
    bars = [(6, 4, 8, 1, 1, 10)]
    path, commands = _trace((2, 5), 1, bars, (20, 18))
    assert path[-1] == (9, 10) and commands[0][1] == (4, 5)
    assert _trace((3, 4), 1, bars, (20, 18))[0][-1] == (9, 10)
    # Four markers exhaust travel partway along a lateral detour.
    path, commands = _trace((1, 4), 1, [(8, 4, 12, 4, 1, 17)], (20, 20))
    assert path[-1] == (3, 14) and not commands[0][4]
    # Valid, separated literal motifs: parallel gated launches converge on a
    # third arrow, which still fires upward. All naive endpoints are in bounds.
    a = np.zeros((24, 20), dtype=np.int8)
    a[10, 4:13] = 5
    a[10, 4] = 2
    arrows = [((5, 10), 1, _arrow_cells((5, 10), (1, 0))),
              ((8, 7), 1, _arrow_cells((8, 7), (1, 0))),
              ((13, 14), -1, _arrow_cells((13, 14), (-1, 0)))]
    for _, _, cells in arrows:
        for cell in cells:
            a[cell] = 1
    expected = a.copy()
    expected[10, 4] = 5
    for cell in arrows[2][2]:
        expected[cell] = 2
    expected[1, 14] = 5
    assert np.array_equal(solve(a), expected)
    f = _analysis(a)
    assert f["endpoints"] == [(13, 14), (13, 14), (1, 14)]
    without_b = _paint(a, f["palette"], f["bars"], f["arrows"],
                       [(r + sign * BUDGET, c) for (r, c), sign, _ in f["arrows"]])
    assert not np.array_equal(without_b, expected)
    without_a = a.copy()
    without_a[without_a == 2] = 5
    assert not np.array_equal(without_a, expected)
    # Removing the second launcher changes a collision strike to a single
    # strike. The complete receiver still changes and its own endpoint survives.
    single = a.copy()
    for cell in arrows[1][2]:
        single[cell] = 0
    single_expected = single.copy()
    single_expected[10, 4] = single_expected[1, 14] = 5
    for cell in arrows[2][2]:
        single_expected[cell] = 5
    assert np.array_equal(solve(single), single_expected)
    # Reject contradictory whole-object assignments, rather than inventing an
    # order-dependent priority between single and collision colors.
    try:
        _events(arrows, [(13, 14), (13, 14), (14, 13)])
    except AssertionError:
        pass
    else:
        raise AssertionError("incompatible multiple-hit roles accepted")
    # Pure spatial and color equivariance, tested without generation metadata.
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        permutation = np.array([8, 4, 0, 9, 3, 1, 7, 2, 6, 5], dtype=np.int8)
        for turns in range(4):
            a = permutation[np.rot90(pair["input"], turns)[:, ::-1]]
            b = permutation[np.rot90(pair["output"], turns)[:, ::-1]]
            assert np.array_equal(solve(a), b)
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("bad output accepted")
    return {"seeds": 42, "pairs": len(records), "palettes": len(palettes),
            "shapes": len({f["shape"] for f in records}),
            "normalized_inputs": len(normalized)}


if __name__ == "__main__":
    print(self_test())
