"""Parity-selected clamp, unary slack, and a taut digital obstacle contour.

The two L-shaped ends belong to ONE support. Parity releases one end and
fixes the other. No parent generator, external file, or generation state is
used by solve. Generation and self_test do not write artifacts.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


_DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SEEDS = (
    0, 553226517, 402094453, 373160373, 873392093, 100991525,
    280267203, 858457139, 883495826, 88554841, 148193221, 379725923,
    130001818, 337161906, 689648810, 737386478, 67049010, 541591569,
    870090522, 945160322, 27144846, 883899868, 612904092, 801072780,
    354915368, 944832517, 644247088, 353865736, 98915119, 195752370,
    307608990, 653774884, 555096492, 698410954, 878454759, 422419924,
    295545199, 580508615, 795784753, 618387138, 91482613, 177418640,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, low, high=None):
    _require(isinstance(value, (int, np.integer))
             and not isinstance(value, (bool, np.bool_)), f"{name} must be an integer")
    value = int(value)
    _require(value >= low and (high is None or value <= high), f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "illegal grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "illegal grid colors")
    return a


def _component(cells, start=None, directions=_DIR8):
    if not cells:
        return set()
    start = min(cells) if start is None else start
    if start not in cells:
        return set()
    reached, todo = {start}, [start]
    while todo:
        r, c = todo.pop()
        for dr, dc in directions:
            q = r + dr, c + dc
            if q in cells and q not in reached:
                reached.add(q)
                todo.append(q)
    return reached


def _key(grid):
    return (grid.shape, grid.astype(np.int8).tobytes())


def _hash(grid):
    return hashlib.sha256(bytes(grid.shape) + grid.astype(np.int8).tobytes()).hexdigest()


def _local_obstacle(obstacle, n, left):
    # Endpoint (1,0) or (1,n-1), forward down, lateral toward the interior.
    return [(r - 1, c if left else n - 1 - c) for r, c in sorted(obstacle)]


def _route(obstacle, n, left, length):
    """Analytic clearance envelope, evaluated on the actual silhouette.

    Each obstacle cell (t,v) contributes v+1-|u-t|. The maximum is the
    tightest one-cell-clear contour with parallel/45-degree segments.
    The visibly encoded support picks the inward lateral axis.
    """
    local = _local_obstacle(obstacle, n, left)
    profile = [max(v + 1 - abs(u - t) for t, v in local) for u in range(length)]
    return [(u + 1, v if left else n - 1 - v) for u, v in enumerate(profile)]


def _sweep_route(obstacle, n, left, length):
    """Construction cross-check: propagate clearance by two linear sweeps."""
    local = _local_obstacle(obstacle, n, left)
    count = max(length, max(t for t, v in local) + 2)
    profile = [-1000] * count
    for t, v in local:
        profile[t] = max(profile[t], v + 1)
    for u in range(1, count):
        profile[u] = max(profile[u], profile[u - 1] - 1)
    for u in range(count - 2, -1, -1):
        profile[u] = max(profile[u], profile[u + 1] - 1)
    return [(u + 1, v if left else n - 1 - v)
            for u, v in enumerate(profile[:length])]


def _inside(cell, n):
    return 0 <= cell[0] < n and 0 <= cell[1] < n


def _parse(input_grid):
    g = _grid(input_grid)
    n = g.shape[0]
    _require(g.shape == (n, n) and n >= 9, "family inputs are squares of side 9..30")
    background, support_color, strand_color = map(int, (g[-1, 0], g[0, 0], g[1, 1]))
    _require(len({background, support_color, strand_color}) == 3, "corner roles must differ")
    colors = set(map(int, np.unique(g)))
    other = colors - {background, support_color, strand_color}
    _require(len(colors) == 4 and len(other) == 1, "exactly four distinct role colors required")
    obstacle_color = other.pop()
    support = {(0, c) for c in range(n)} | {(1, 0), (1, n - 1)}
    actual_support = {tuple(map(int, p)) for p in np.argwhere(g == support_color)}
    _require(actual_support == support, "support must be a single top bracket")
    _require(int(g[1, n - 2]) == strand_color, "both crops need inward strand cells")
    slack = {tuple(map(int, p)) for p in np.argwhere(g == strand_color)}
    obstacle = {tuple(map(int, p)) for p in np.argwhere(g == obstacle_color)}
    _require(_component(slack) == slack, "slack must be one eight-connected object")
    _require(_component(obstacle) == obstacle, "obstacle must be one eight-connected object")
    near = min(r for r, c in obstacle)
    _require(near >= 4, "obstacle too close to support")
    _require(all(1 <= c <= n - 2 and 1 <= r <= near - 2 for r, c in slack),
             "slack must lie above the separated obstacle, inside the rails")
    _require(all(1 <= c <= n - 2 for r, c in obstacle), "obstacle must be interior laterally")
    # BOTH corners must be genuine possible clamps, even before parity selects.
    for left in (True, False):
        local = _local_obstacle(obstacle, n, left)
        _require(max(v - u for u, v in local) == 0, "diagonal tangent must meet its crop seed")
        path = _route(obstacle, n, left, max(len(slack), n + 2))
        profile = [c if left else n - 1 - c for r, c in path]
        slopes = [b - a for a, b in zip(profile, profile[1:])]
        _require(all(s in (-1, 0, 1) for s in slopes)
                 and slopes == sorted(slopes, reverse=True), "nonconvex exposed contour")
        _require(path[0] == (1, 1 if left else n - 2), "wrong fixed attachment")
        _require(not set(path) & (obstacle | support), "route crosses preserved material")
    return {"grid": g, "n": n, "colors": (background, support_color, strand_color, obstacle_color),
            "support": support, "slack": slack, "obstacle": obstacle,
            "length": len(slack), "left": bool(n % 2)}


def _paint(scene, route):
    out = scene["grid"].copy()
    background, _, strand_color, _ = scene["colors"]
    out[out == strand_color] = background
    for p in route:
        if _inside(p, scene["n"]):
            _require(int(out[p]) == background, "route collision")
            out[p] = strand_color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the answer from the query alone; never mutate the query."""
    scene = _parse(input_grid)
    # This is B's literal crop. Its lower inward cell is A's fixed seed.
    col0 = 0 if scene["left"] else scene["n"] - 2
    crop = scene["grid"][:2, col0:col0 + 2].copy()
    seed_cells = np.argwhere(crop == scene["colors"][2])
    _require(seed_cells.shape == (1, 2), "crop must have one inward strand seed")
    sr, sc = map(int, seed_cells[0])
    seed = sr, sc + col0
    # Read the L's orientation: its bottom support neighbor is on the outside.
    inward_right = int(crop[1, 0]) == scene["colors"][1]
    route = _route(scene["obstacle"], scene["n"], inward_right, scene["length"])
    _require(route[0] == seed, "contour must begin at the cropped attachment")
    return _paint(scene, route)


def _obstacle(n, width, height, shape):
    """Convex lattice silhouettes: independent left and right ramp schedules.

    shape = (left expansion, left plateau, right expansion, right plateau).
    Filling each row interval includes points, bars, rectangles, diamonds,
    tapered wedges and asymmetric stepped hexagons. No bounding-box proxy.
    """
    lu, lf, ru, rf = shape
    r0, c0 = (n + width) // 2, (n - width) // 2
    cells = set()
    for u in range(height):
        lo = -min(u, lu) + max(0, u - lu - lf)
        hi = width - 1 + min(u, ru) - max(0, u - ru - rf)
        _require(lo <= hi, "empty obstacle slice")
        cells.update((r0 + u, c0 + c) for c in range(lo, hi + 1))
    _require(cells and all(4 <= r < n and 1 <= c <= n - 2 for r, c in cells),
             "obstacle does not fit")
    return cells


def _sample_obstacle(rng, n, lesson=False):
    if lesson:
        width = 1 if n % 2 else 2
        height, shape = lesson
        return _obstacle(n, width, height, shape)
    widths = [w for w in range(1, min(8, n - 4)) if w % 2 == n % 2]
    width = int(rng.choice(widths))
    height = int(rng.integers(1, min(8, n - (n + width) // 2) + 1))
    mode = int(rng.integers(0, 5))
    for _ in range(48):
        if mode == 0:  # rectangle, including points and bars
            shape = (0, height - 1, 0, height - 1)
        elif mode == 1:  # expanding wedge
            shape = (height - 1, 0, height - 1, 0)
        elif mode == 2:  # a symmetric cap or a cap with a flat middle
            up = (height - 1) // 2
            flat = height - 1 - 2 * up
            shape = (up, flat, up, flat)
        else:  # independently stepped convex flanks
            lu, ru = map(int, rng.integers(0, height, size=2))
            lf, rf = int(rng.integers(0, height - lu)), int(rng.integers(0, height - ru))
            shape = (lu, lf, ru, rf)
        try:
            return _obstacle(n, width, height, shape)
        except ValueError:
            continue
    # Guaranteed rectangle with the same sampled size, rather than a whole-pair bank.
    return _obstacle(n, width, height, (0, height - 1, 0, height - 1))


def _slack(rng, n, length, depth, style):
    """A bridge between the two candidate pins, with independently grown slack."""
    _require(n - 2 <= length <= depth * (n - 2), "slack cannot fit")
    cells, row = {(1, 1)}, 1
    for col in range(2, n - 1):
        if col == n - 2:
            row = 1
        elif style == 0:
            # Independent wavy chains; column 2 ensures a diagonal-only link.
            row = 2 if col == 2 else int(rng.integers(1, min(2, depth) + 1))
        else:
            options = [r for r in (row - 1, row, row + 1)
                       if 1 <= r <= min(depth, n - 1 - col)]
            row = int(rng.choice(options))
        cells.add((row, col))
    while len(cells) < length:
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _DIR8
                           if 1 <= r + dr <= depth and 1 <= c + dc <= n - 2} - cells)
        _require(bool(frontier), "slack growth exhausted")
        neighbors = np.array([sum((r + dr, c + dc) in cells for dr, dc in _DIR8)
                              for r, c in frontier], dtype=float)
        weights = (neighbors + 1) ** (2 if style == 1 else -1 if style == 2 else 0)
        index = int(rng.choice(len(frontier), p=weights / weights.sum()))
        cells.add(frontier[index])
    return cells


def _make_pair(rng, n, length, palette=None, lesson=False):
    colors = tuple(map(int, rng.choice(10, size=4, replace=False))) if palette is None else palette
    obstacle = _sample_obstacle(rng, n, lesson)
    depth_limit = min(5, min(r for r, c in obstacle) - 2)
    min_depth = max(2, (length + n - 3) // (n - 2))
    _require(min_depth <= depth_limit, "insufficient slack space")
    depth = int(rng.integers(min_depth, depth_limit + 1))
    slack = _slack(rng, n, length, depth, 0 if lesson and length == n - 2 else int(rng.integers(0, 4)))
    bg, sc, tc, oc = colors
    inp = np.full((n, n), bg, dtype=np.int8)
    inp[0, :] = sc
    inp[1, 0] = inp[1, -1] = sc
    for p in obstacle:
        inp[p] = oc
    for p in slack:
        inp[p] = tc
    path = _sweep_route(obstacle, n, bool(n % 2), length)
    out = inp.copy()
    out[out == tc] = bg
    for p in path:
        if _inside(p, n):
            _require(int(out[p]) == bg, "construction collision")
            out[p] = tc
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    _require(nt > nq, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base = int(rng.integers(13, 23))
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    width = 1 if base % 2 else 2
    height = int(rng.integers(3, min(6, base - 2 - (base + width) // 2) + 1))
    lu, ru = map(int, rng.integers(1, height, size=2))
    lf = int(rng.integers(max(0, height - 1 - 2 * lu), height - lu))
    rf = int(rng.integers(max(0, height - 1 - 2 * ru), height - ru))
    lesson = (height, (lu, lf, ru, rf))
    # The first two lessons suffice even at the minimum requested count:
    # both parities, an exact eight-connected length, wrapping and clipping.
    plans = [(base, base - 2, palette, lesson),
             (base + 1, base + int(rng.integers(3, 9)), palette, lesson)]
    if nt >= 3:
        plans.append((base - 1, base - 2, palette, lesson))
    if nt >= 4:
        plans.append((base + 2, base + 1, None, False))
    while len(plans) < nt:
        n = int(rng.integers(9, 29))
        length = int(rng.integers(n - 2, n + 10))
        plans.append((n, length, None, False))
    puzzle, seen = {"train": [], "test": []}, set()
    for n, length, colors, lesson in plans:
        for _ in range(32):
            pair = _make_pair(rng, n, length, colors, lesson)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle["train"].append(pair)
                break
        else:
            raise RuntimeError("unique training input construction exhausted")
    train_sizes = {p["input"].shape[0] for p in puzzle["train"]}
    parities = [0, 1] if nq >= 2 else [int(rng.integers(0, 2))]
    while len(parities) < nq:
        parities.append(int(rng.integers(0, 2)))
    rng.shuffle(parities)
    for parity in parities:
        candidates = [n for n in range(9, 31) if n not in train_sizes and n % 2 == parity]
        for _ in range(32):
            n = int(rng.choice(candidates))
            length = int(rng.integers(n - 2, n + 10))
            pair = _make_pair(rng, n, length)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle["test"].append(pair)
                break
        else:
            raise RuntimeError("unique test input construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _straight(scene):
    n = scene["n"]
    return [(r, r if scene["left"] else n - 1 - r) for r in range(1, scene["length"] + 1)]


def _measure(scene):
    n, length, left = scene["n"], scene["length"], scene["left"]
    obstacle, slack = scene["obstacle"], scene["slack"]
    route = _route(obstacle, n, left, length)
    rows, cols = zip(*obstacle)
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    bbox = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    visible = {p for p in route if _inside(p, n)}
    return {"side": n, "left": left, "length": length, "visible": len(visible),
            "clipped": len(visible) < length, "obstacle_height": r1 - r0 + 1,
            "obstacle_width": c1 - c0 + 1, "irregular": obstacle != bbox,
            "silhouette": tuple(sorted((r - r0, c - c0) for r, c in obstacle)),
            "needs_diagonal_count": len(_component(slack, (1, 1), _DIR4)) != length,
            "slack_depth": max(r for r, c in slack),
            "outgoing_visible": any(r > r1 for r, c in visible),
            "bbox_changes_route": _route(bbox, n, left, length) != route}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, lessons = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            scene = _parse(inp)
            n, length = scene["n"], scene["length"]
            _require(inp.shape == out.shape, "output shape changed")
            _require(n - 2 <= length <= n + 9, "generated length outside sampling range")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(np.array_equal(solve(inp), out), "input-only answer mismatch")
            bg, sc, tc, oc = scene["colors"]
            _require(np.all(out[inp == sc] == sc) and np.all(out[inp == oc] == oc),
                     "support or obstacle changed")
            route = _route(scene["obstacle"], n, scene["left"], length)
            _require(route == _sweep_route(scene["obstacle"], n, scene["left"], length),
                     "analytic and sweep envelopes disagree")
            visible = {p for p in route if _inside(p, n)}
            _require(len(visible) == min(length, n - 1), "unexpected clipping/reentry")
            _require(int(out[route[0]]) == tc, "fixed attachment lost")
            _require(int(np.sum(out == tc)) == len(visible), "unary length/clipping mismatch")
            wrong = _paint(scene, _route(scene["obstacle"], n, not scene["left"], length))
            _require(not np.array_equal(wrong, out), "crop selection has no causal effect")
            if split == "train":
                lessons.append((scene, _measure(scene), out))
    _require({s["n"] % 2 for s, m, o in lessons} == {0, 1}, "missing parity lesson")
    _require(any(not m["clipped"] and m["needs_diagonal_count"] and m["outgoing_visible"]
                 and m["bbox_changes_route"] for s, m, o in lessons),
             "need a full-length, diagonal-connected, silhouette-sensitive wrapping lesson")
    _require(any(m["clipped"] for s, m, o in lessons), "missing clipping lesson")
    _require(len({m["length"] for s, m, o in lessons}) >= 2, "missing unary length variation")
    _require(any(not np.array_equal(_paint(s, _straight(s)), o) for s, m, o in lessons),
             "ignoring obstacle contour fits all demonstrations")
    if nt >= 4:
        for parity in (0, 1):
            _require(len({s["n"] for s, m, o in lessons if s["n"] % 2 == parity}) >= 2,
                     "four lessons need multiple sizes per parity")
    train_sizes = {s["n"] for s, m, o in lessons}
    _require(all(np.asarray(p["input"]).shape[0] not in train_sizes for p in puzzle["test"]),
             "test must transfer parity and routing to an unseen side length")
    if nq >= 2:
        _require({np.asarray(p["input"]).shape[0] % 2 for p in puzzle["test"]} == {0, 1},
                 "multiple tests must include both parities")


def self_test():
    """Bounded, streaming 42-seed audit; returns compact-serializable metrics.

    Keeps hashes/features only, one puzzle at a time. No execution traces or
    payload copies are written. This is generator-owned testing, not an
    independent semantic certification.
    """
    puzzles, structures, sizes, backgrounds, palettes = set(), set(), set(), set(), set()
    silhouettes, lengths, depths, widths, heights = set(), set(), set(), set(), set()
    counts = Counter()
    for seed in _SEEDS:
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        signature = []
        for split in ("train", "test"):
            for pair, other in zip(puzzle[split], again[split]):
                assert all(np.array_equal(pair[k], other[k]) for k in ("input", "output"))
                original = pair["input"].copy()
                assert np.array_equal(solve(original), pair["output"])
                assert np.array_equal(original, pair["input"]), "solve mutated its input"
                scene = _parse(original)
                m = _measure(scene)
                signature.append((_hash(original), _hash(pair["output"])))
                sizes.add(m["side"])
                lengths.add(m["length"])
                depths.add(m["slack_depth"])
                widths.add(m["obstacle_width"])
                heights.add(m["obstacle_height"])
                silhouettes.add(m["silhouette"])
                backgrounds.add(scene["colors"][0])
                palettes.add(scene["colors"])
                normalized = np.zeros_like(original)
                for role, color in enumerate(scene["colors"]):
                    normalized[original == color] = role
                structures.add(_hash(normalized))
                counts[split + "_pairs"] += 1
                counts["odd" if m["left"] else "even"] += 1
                for feature in ("clipped", "irregular", "needs_diagonal_count", "bbox_changes_route", "outgoing_visible"):
                    counts[feature] += int(m[feature])
                counts["unclipped"] += int(not m["clipped"])
        puzzles.add(tuple(signature))
    assert len(puzzles) == 42 and len(backgrounds) == 10
    assert len(structures) >= 180 and len(silhouettes) >= 25
    assert len(sizes) >= 18 and len(palettes) >= 70 and len(lengths) >= 20
    assert len(depths) >= 4 and len(widths) >= 6 and len(heights) >= 5
    for feature in ("odd", "even", "clipped", "unclipped", "irregular", "needs_diagonal_count", "bbox_changes_route"):
        assert counts[feature] >= 20, f"insufficient {feature} coverage"
    explicit_calls = 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(991 + 17 * nt + nq, nt, nq)
            assert (len(p["train"]), len(p["test"])) == (nt, nq)
            explicit_calls += 1
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None), ("4", 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A small hand-calculated cap checks early departure rather than far-face travel.
    obstacle = {(5, 4), (6, 3), (6, 4), (6, 5), (7, 4)}
    expected = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 5), (8, 4)]
    assert _route(obstacle, 9, True, 8) == expected
    assert _sweep_route(obstacle, 9, True, 8) == expected
    # Counterfactuals: same count/different slack, palette permutation, and crop.
    p = generate_puzzle(177, 4, 1)
    for pair in p["train"]:
        s = _parse(pair["input"])
        perm = np.array([4, 8, 1, 9, 6, 0, 7, 2, 5, 3], dtype=np.int8)
        assert np.array_equal(solve(perm[s["grid"]]), perm[pair["output"]])
        rng = np.random.default_rng(s["n"])
        depth = min(5, min(r for r, c in s["obstacle"]) - 2)
        slack = _slack(rng, s["n"], s["length"], depth, 2)
        altered = s["grid"].copy()
        altered[altered == s["colors"][2]] = s["colors"][0]
        for q in slack:
            altered[q] = s["colors"][2]
        assert np.array_equal(solve(altered), pair["output"])
        if s["length"] == s["n"] - 2:
            # One extra slack cell appends exactly one conceptual route step.
            frontier = sorted({(r + dr, c + dc) for r, c in s["slack"] for dr, dc in _DIR8
                               if 1 <= r + dr <= depth and 1 <= c + dc <= s["n"] - 2}
                              - s["slack"])
            longer = s["grid"].copy()
            longer[frontier[0]] = s["colors"][2]
            answer = solve(longer)
            assert int(np.sum(answer == s["colors"][2])) == s["length"] + 1
            assert int(np.sum(answer != pair["output"])) == 1
            # Same seed and same unary count, but a different actual obstacle.
            simpler = s["grid"].copy()
            simpler[simpler == s["colors"][3]] = s["colors"][0]
            width = 1 if s["n"] % 2 else 2
            for q in _obstacle(s["n"], width, 1, (0, 0, 0, 0)):
                simpler[q] = s["colors"][3]
            assert not np.array_equal(solve(simpler), pair["output"])
    damaged = generate_puzzle(177, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("validator accepted damaged preserved support")
    return {"seeds": len(_SEEDS), "distinct_puzzles": len(puzzles), "counts": dict(counts),
            "side_lengths": sorted(sizes), "lengths": sorted(lengths),
            "backgrounds": sorted(backgrounds), "role_palettes": len(palettes),
            "normalized_inputs": len(structures), "obstacle_silhouettes": len(silhouettes),
            "obstacle_heights": sorted(heights), "obstacle_widths": sorted(widths),
            "slack_depths": sorted(depths), "legal_count_calls": explicit_calls,
            "semantic_verification": "generator-owned checks only"}
