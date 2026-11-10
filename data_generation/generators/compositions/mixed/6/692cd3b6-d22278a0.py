"""Gated shortest-route union, colored through corner territories and contours.

Only standard-library modules and NumPy are used. Nothing is read from disk.
Generation and all checks are bounded; self_test streams puzzles rather than
retaining grids or path lists. No execution traces or JSON artifacts are saved.
"""

from collections import deque
import hashlib
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a


def _corners(n):
    return ((0, 0), (0, n - 1), (n - 1, n - 1), (n - 1, 0))


def _parse(a):
    """Infer every role from this input, then check the two chamber shells."""
    n, m = a.shape
    if n != m or n < 11:
        raise ValueError("family inputs are squares of side 11..30")
    colors, counts = np.unique(a, return_counts=True)
    background = int(colors[np.argmax(counts)])
    corner_positions = _corners(n)
    sources = [(p, int(a[p])) for p in corner_positions if a[p] != background]
    source_colors = {color for _, color in sources}
    if not 2 <= len(sources) <= 4 or len(source_colors) != len(sources):
        raise ValueError("need 2..4 differently colored corner sources")
    other = [(int(c), int(k)) for c, k in zip(colors, counts)
             if c != background and c not in source_colors]
    wall_options = [c for c, k in other if k == 14]
    terminal_options = [c for c, k in other if k == 2]
    if len(other) != 2 or len(wall_options) != 1 or len(terminal_options) != 1:
        raise ValueError("need fourteen wall cells and two matching terminals")
    wall, terminal = wall_options[0], terminal_options[0]
    centers = tuple(tuple(map(int, p)) for p in np.argwhere(a == terminal))
    expected = np.full(a.shape, background, dtype=np.int8)
    for p, color in sources:
        expected[p] = color
    areas, openings = [], []
    for r, c in centers:
        if not (1 <= r < n - 1 and 1 <= c < n - 1):
            raise ValueError("terminal must be inside a complete 3x3 shell")
        area = {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        if area.intersection(corner_positions):
            raise ValueError("chambers must not cover canvas corners")
        areas.append(area)
        gates = [i for i, (dr, dc) in enumerate(_DIRS)
                 if a[r + dr, c + dc] == background]
        if len(gates) != 1:
            raise ValueError("each terminal must have exactly one gate")
        gate = gates[0]
        dr, dc = _DIRS[gate]
        if not (0 <= r + 2 * dr < n and 0 <= c + 2 * dc < n):
            raise ValueError("gate has no exit inside the canvas")
        openings.append(gate)
        expected[r - 1:r + 2, c - 1:c + 2] = wall
        expected[r, c] = terminal
        expected[r + dr, c + dc] = background
    if (areas[0].intersection(areas[1]) or
            max(abs(centers[0][0] - centers[1][0]),
                abs(centers[0][1] - centers[1][1])) < 4):
        raise ValueError("chambers require a separating background gap")
    if not np.array_equal(expected, a):
        raise ValueError("unexpected marks or malformed shells")
    return background, wall, terminal, centers, tuple(openings), sources


def _distances(walkable, start):
    n, m = walkable.shape
    d = np.full((n, m), -1, dtype=np.int16)
    d[start] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        next_d = int(d[r, c]) + 1
        for dr, dc in _DIRS:
            rr, cc = r + dr, c + dc
            if (0 <= rr < n and 0 <= cc < m and walkable[rr, cc]
                    and d[rr, cc] < 0):
                d[rr, cc] = next_d
                q.append((rr, cc))
    return d


def _analyze(a):
    background, wall, terminal, centers, openings, sources = _parse(a)
    walkable = (a == background) | (a == terminal)
    ds = _distances(walkable, centers[0])
    dt = _distances(walkable, centers[1])
    length = int(ds[centers[1]])
    if length < 0:
        raise ValueError("terminals are disconnected")
    route = ((a == background) & (ds >= 0) & (dt >= 0)
             & (ds + dt == length))
    rows, cols = np.indices(a.shape)
    offsets = [(abs(rows - r), abs(cols - c)) for (r, c), _ in sources]
    l1 = np.stack([dr + dc for dr, dc in offsets])
    linf = np.stack([np.maximum(dr, dc) for dr, dc in offsets])
    winner = l1.argmin(axis=0)
    nearest = l1 == l1.min(axis=0)
    unique = nearest.sum(axis=0) == 1
    radius = np.take_along_axis(linf, winner[None], axis=0)[0]
    even = radius % 2 == 0
    paint = route & unique & even
    return dict(background=background, wall=wall, terminal=terminal,
                centers=centers, openings=openings, sources=sources,
                ds=ds, dt=dt, length=length, route=route, winner=winner,
                nearest=nearest, unique=unique, even=even, paint=paint,
                l1=l1, linf=linf)


def _render(a, info, mask):
    result = a.copy()
    for index, (_, color) in enumerate(info["sources"]):
        result[mask & (info["winner"] == index)] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only shortest-path selection followed by territory/contour clipping."""
    a = _grid(input_grid)
    info = _analyze(a)
    return _render(a, info, info["paint"])


def _features(a, info):
    """Counterexamples to shortcuts, computed without storing any route list."""
    route, unique, even, paint = (info[k] for k in ("route", "unique", "even", "paint"))
    layers = np.bincount(info["ds"][paint], minlength=info["length"] + 1)
    # Two painted cells at the same distance cannot belong to any ONE shortest
    # path. This is stronger than only checking one arbitrary path choice.
    union_needed = bool(np.any(layers > 1))
    tie_even = route & ~unique & np.any(info["nearest"] & (info["linf"] % 2 == 0), axis=0)
    l1_even = info["l1"].min(axis=0) % 2 == 0
    cheb_owner = info["linf"].argmin(axis=0)
    cheb_unique = (info["linf"] == info["linf"].min(axis=0)).sum(axis=0) == 1
    cheb_paint = route & cheb_unique & (info["linf"].min(axis=0) % 2 == 0)
    owner_error = ((paint != cheb_paint) |
                   (paint & cheb_paint & (info["winner"] != cheb_owner)))
    # Removing walls still preserves visible marks, but makes the geometric
    # shortest union ignore the forced gate detours.
    free = np.ones(a.shape, dtype=bool)
    for p, _ in info["sources"]:
        free[p] = False
    f = _distances(free, info["centers"][0])
    t = _distances(free, info["centers"][1])
    wall_free = ((a == info["background"]) & (f >= 0) & (t >= 0)
                 & (f + t == f[info["centers"][1]]))
    boundary = np.zeros(a.shape, dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    return {
        "union_needed": union_needed,
        "tie_needed": bool(tie_even.any()),
        "parity_needed": bool((route & unique & ~even).any()),
        "route_needed": bool(((a == info["background"]) & ~route & unique & even).any()),
        "walls_needed": bool(((wall_free ^ route) & unique & even).any()),
        "metric_needed": bool((route & unique & (even != l1_even)).any()),
        "ownership_needed": bool(owner_error.any()),
        "changed": int(paint.sum()),
        "colors_used": len(set(map(int, info["winner"][paint]))),
        "route_cells": int(route.sum()),
        "edge_paint": bool((paint & boundary).any()),
        "source_count": len(info["sources"]),
        "single_route": int(route.sum()) == info["length"] - 1,
    }


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return int(value)


def _counts(num_train, num_test):
    # None is a convenience for the supplied inspection helper. The public
    # defaults remain exactly 4/1, and explicit illegal counts are rejected.
    if num_train is None:
        num_train = max(4, (1 if num_test is None else _count(num_test, "num_test", 1, 4)) + 1)
    if num_test is None:
        num_test = 1
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _place(a, center, gate):
    r, c = center
    a[r - 1:r + 2, c - 1:c + 2] = 1
    a[r, c] = 2
    dr, dc = _DIRS[gate]
    a[r + dr, c + dc] = 0


def _candidate(rng, mode, sizes, count):
    n = int(rng.choice(sizes))
    a = np.zeros((n, n), dtype=np.int8)
    if mode in ("broad", "straight"):
        r = int(rng.integers(2, n - 2))
        c0 = int(rng.integers(2, n - 6))
        c1 = int(rng.integers(c0 + 4, n - 2))
        centers = ((r, c0), (r, c1))
        gates = (3, 1) if mode == "broad" else (1, 3)
    elif mode == "edge":
        c0 = int(rng.integers(2, n - 6))
        c1 = int(rng.integers(c0 + 4, n - 2))
        centers, gates = ((2, c0), (n - 2, c1)), (0, 1)
    else:
        centers = tuple(tuple(map(int, p)) for p in rng.integers(1, n - 1, size=(2, 2)))
        if max(abs(centers[0][0] - centers[1][0]), abs(centers[0][1] - centers[1][1])) < 4:
            return None
        gates = tuple(map(int, rng.integers(4, size=2)))
    for center, gate in zip(centers, gates):
        r, c = center
        dr, dc = _DIRS[gate]
        if not (0 <= r + 2 * dr < n and 0 <= c + 2 * dc < n):
            return None
        if set(_corners(n)).intersection((r + x, c + y) for x in (-1, 0, 1) for y in (-1, 0, 1)):
            return None
        _place(a, center, gate)
    if mode == "broad":
        # An odd adjacent-corner comparison exposes a real tie line. Its
        # placement rotates/reflects with the whole input below.
        mask = (0, 1)
    else:
        mask = tuple(sorted(map(int, rng.choice(4, size=count, replace=False))))
    for i, corner in enumerate(mask):
        a[_corners(n)[corner]] = 3 + i
    a = np.rot90(a, int(rng.integers(4))).copy()
    if rng.integers(2):
        a = np.fliplr(a).copy()
    try:
        info = _analyze(a)
    except ValueError:
        return None
    return a, info


def _geometry_key(a):
    # Canonical roles in sampled candidates, independent of literal colors.
    normalized = a.copy()
    for i, p in enumerate(p for p in _corners(a.shape[0]) if a[p] >= 3):
        normalized[p] = 3 + i
    return a.shape, normalized.tobytes()


def _sample_pair(rng, mode, sizes, count, roles, used, essential=False):
    for _ in range(800):
        candidate = _candidate(rng, mode, sizes, count)
        if candidate is None:
            continue
        a, info = candidate
        key = _geometry_key(a)
        if key in used:
            continue
        f = _features(a, info)
        if f["changed"] < 4 or not f["parity_needed"] or not f["route_needed"]:
            continue
        if mode == "broad" and not (f["union_needed"] and f["tie_needed"]
                                     and f["walls_needed"] and f["ownership_needed"]
                                     and f["colors_used"] == 2):
            continue
        if mode == "edge" and not f["edge_paint"]:
            continue
        if essential and not (f["metric_needed"] and f["colors_used"] >= 2):
            continue
        palette = [c for c in range(10) if c not in roles]
        rng.shuffle(palette)
        mapping = np.array(list(roles) + palette, dtype=np.int8)
        colored = mapping[a]
        used.add(key)
        return {"input": colored, "output": solve(colored)}
    raise RuntimeError(f"bounded sampler exhausted for {mode}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return a reproducible puzzle; every output-driving fact is in its input."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_count, test_count = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, size=3, replace=False)))
    used = set()
    puzzle = {"train": [], "test": []}
    # Both required evidence cases survive even when only two demos are asked.
    puzzle["train"].append(_sample_pair(rng, "broad", list(range(11, 22, 2)), 2, roles, used))
    puzzle["train"].append(_sample_pair(rng, "random", list(range(12, 23, 2)), 3, roles, used, essential=True))
    for _ in range(train_count - 2):
        mode = ("random", "straight", "edge")[int(rng.choice(3, p=(0.55, 0.25, 0.20)))]
        puzzle["train"].append(_sample_pair(rng, mode, list(range(11, 24)), int(rng.integers(2, 5)), roles, used))
    rng.shuffle(puzzle["train"])
    max_train = max(p["input"].shape[0] for p in puzzle["train"])
    # A guaranteed scale shift is visible and does not require a hidden gate
    # script or a new convention. All tests are new geometry and larger squares.
    for _ in range(test_count):
        mode = ("random", "straight", "edge")[int(rng.choice(3, p=(0.65, 0.15, 0.20)))]
        puzzle["test"].append(_sample_pair(rng, mode, list(range(max_train + 1, 31)), int(rng.integers(2, 5)), roles, used))
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check shape, roles, routes, clipped colors, and necessary demo evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, geometry, records = set(), set(), {"train": [], "test": []}
    common_roles = None
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "canvas dimensions changed")
            key = (a.shape, a.astype(np.int8).tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze(a)
            roles = tuple(info[k] for k in ("background", "wall", "terminal"))
            if common_roles is None:
                common_roles = roles
            _require(roles == common_roles, "puzzle-level chamber roles changed")
            norm = np.zeros(a.shape, dtype=np.int8)
            norm[a == info["wall"]] = 1
            norm[a == info["terminal"]] = 2
            for i, (p, _) in enumerate(info["sources"]):
                norm[p] = i + 3
            gkey = _geometry_key(norm)
            _require(gkey not in geometry, "recolored geometry repeated")
            geometry.add(gkey)
            _require(np.array_equal(b, _render(a, info, info["paint"])), "wrong output")
            changed = a != b
            _require(np.array_equal(changed, info["paint"]), "changed cells differ from clipped route mask")
            _require(np.array_equal(a[a != info["background"]], b[a != info["background"]]), "a visible mark changed")
            for p, gate in zip(info["centers"], info["openings"]):
                dr, dc = _DIRS[gate]
                _require(info["route"][p[0] + dr, p[1] + dc], "route union missed a mandatory opening")
            f = _features(a, info)
            _require(f["changed"] >= 4 and f["route_needed"] and f["parity_needed"], "insufficient interaction evidence")
            records[split].append((a.shape[0], f))
    train_features = [f for _, f in records["train"]]
    for name in ("union_needed", "tie_needed", "walls_needed", "metric_needed", "ownership_needed"):
        _require(any(f[name] for f in train_features), f"training lacks {name} witness")
    _require(any(f["source_count"] == 2 and f["colors_used"] == 2 for f in train_features), "need a two-source demo")
    _require(any(f["source_count"] == 3 and f["colors_used"] >= 2 for f in train_features), "need a three-source demo")
    max_train = max(n for n, _ in records["train"])
    _require(all(n > max_train for n, _ in records["test"]), "test must apply the rule at a larger scale")


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        h.update(split.encode())
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 streamed seeds, meaningful invariants, equivariance, and count edges."""
    fingerprints, sizes, masks, openings, route_sizes, backgrounds, source_counts = (set() for _ in range(7))
    edge_seen = single_seen = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                info = _analyze(a)
                f = _features(a, info)
                sizes.add(a.shape[0])
                backgrounds.add(info["background"])
                masks.add(tuple(index for index, p in enumerate(_corners(a.shape[0]))
                                if a[p] != info["background"]))
                openings.add(info["openings"])
                route_sizes.add(f["route_cells"])
                source_counts.add(f["source_count"])
                edge_seen += f["edge_paint"]
                single_seen += f["single_route"]
                _require(np.array_equal(solve(a), b), "solve mismatch")
        # Symmetries and color relabeling change no geometric decisions.
        a, b = puzzle["test"][0]["input"], puzzle["test"][0]["output"]
        _require(np.array_equal(solve(np.rot90(a)), np.rot90(b)), "rotation failed")
        _require(np.array_equal(solve(np.fliplr(a)), np.fliplr(b)), "reflection failed")
        permutation = np.array([7, 3, 9, 0, 6, 2, 8, 1, 5, 4], dtype=np.int8)
        _require(np.array_equal(solve(permutation[a]), permutation[b]), "palette equivariance failed")
    _require(len(fingerprints) == 42, "seed diversity failed")
    _require(len(sizes) >= 16 and len(route_sizes) >= 45, "size/route diversity too small")
    _require(source_counts == {2, 3, 4} and len(openings) >= 12, "source/gate diversity too small")
    _require(len(backgrounds) >= 8 and len(masks) >= 10, "palette/source geometry diversity too small")
    _require(edge_seen >= 5 and single_seen >= 5, "edge or single-route coverage missing")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(900 + 10 * train + test, train, test)
            validate_generated_puzzle(p)
            _require(len(p["train"]) == train and len(p["test"]) == test, "counts ignored")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(_fingerprint(generate_puzzle(1729)) == _fingerprint(generate_puzzle(1729)), "nondeterminism")
    return {"seeds": 42, "puzzles": len(fingerprints), "sizes": len(sizes),
            "route_sizes": len(route_sizes), "opening_pairs": len(openings),
            "corner_masks": len(masks),
            "backgrounds": len(backgrounds), "source_counts": sorted(source_counts),
            "edge_pairs": edge_seen, "single_route_pairs": single_seen}
