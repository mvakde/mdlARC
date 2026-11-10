"""Route-activated bend diagonals. Only stdlib and NumPy; no disk I/O.

0: floor, 1: routing wall, 4: new route ink. Two boundary dots are
terminals; the other nonstandard color marks transparent floor overlays.
See task_spec.md for the corner-chord convention and drawing priority.
"""
from collections import deque
import hashlib
import json

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DOT_COLORS = (2, 3, 5, 6, 7, 8, 9)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(n < 1 or n > 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _neighbors(p, shape):
    h, w = shape
    r, c = p
    return [(r + dr, c + dc) for dr, dc in _DIRS
            if 0 <= r + dr < h and 0 <= c + dc < w]


def _route(walls, start, goal, forbidden_edge=None):
    """BFS with path multiplicity capped at two (ties are never broken)."""
    distance = {start: 0}
    ways = {start: 1}
    previous = {}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for q in _neighbors(p, walls.shape):
            if walls[q] or forbidden_edge == (p, q) or forbidden_edge == (q, p):
                continue
            if q not in distance:
                distance[q] = distance[p] + 1
                ways[q] = ways[p]
                previous[q] = p
                queue.append(q)
            elif distance[q] == distance[p] + 1:
                ways[q] = min(2, ways[q] + ways[p])
    if goal not in distance:
        return [], 0
    p, path = goal, [goal]
    while p != start:
        p = previous[p]
        path.append(p)
    return path[::-1], ways[goal]


def _bend_slope(before, after):
    """Slope of the chord across a right-angle bend, independent of travel."""
    dr, dc = after[0] - before[0], after[1] - before[1]
    if abs(dr) == abs(dc) == 1:
        return 1 if dr * dc > 0 else -1
    return None


def _diagonal(point, slope, shape):
    r, c = point
    h, w = shape
    return {(rr, c + slope * (rr - r)) for rr in range(h)
            if 0 <= c + slope * (rr - r) < w}


def _roles(a):
    if np.any(a == 4):
        raise ValueError("Route ink 4 must be absent from inputs")
    boundary = np.zeros(a.shape, dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    endpoint_colors = set(map(int, np.unique(a[boundary]))) - {0, 1}
    if len(endpoint_colors) != 1:
        raise ValueError("Exactly one terminal color must occur on the boundary")
    terminal = endpoint_colors.pop()
    endpoints = [tuple(map(int, p)) for p in np.argwhere(a == terminal)]
    if len(endpoints) != 2 or not all(boundary[p] for p in endpoints):
        raise ValueError("Exactly two terminal dots, both on the boundary, required")
    other = set(map(int, np.unique(a))) - {0, 1, terminal}
    if len(other) > 1:
        raise ValueError("At most one secondary marker color is allowed")
    secondary = next(iter(other)) if other else None
    return terminal, secondary, endpoints


def _analyze(a):
    terminal, secondary, endpoints = _roles(a)
    path, count = _route(a == 1, *endpoints)
    if count != 1 or len(path) < 3:
        raise ValueError("The terminal route must have exactly one shortest path")
    path_set = set(path)
    active, inert, lines = [], set(), []
    indices = {p: i for i, p in enumerate(path)}
    if secondary is not None:
        for p_array in np.argwhere(a == secondary):
            p = tuple(map(int, p_array))
            if p not in path_set:
                inert.add(p)
                continue
            i = indices[p]
            slope = _bend_slope(path[i - 1], path[i + 1])
            if slope is None:
                raise ValueError("On-route secondary markers must be at bends")
            line = _diagonal(p, -slope, a.shape)
            if line.intersection(endpoints):
                raise ValueError("Secondary diagonals must avoid the terminals")
            active.append((p, -slope))
            lines.append(line)
    return {"terminal": terminal, "secondary": secondary, "endpoints": endpoints,
            "path": path, "path_set": path_set, "active": active,
            "inert": inert, "lines": lines}


def _render(a, info):
    out = a.copy()
    for p in info["path"][1:-1]:
        if a[p] == 0:
            out[p] = 4
    for line in info["lines"]:
        for p in line:
            out[p] = info["secondary"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer both dot roles and the entire answer from the given input only."""
    a = _grid(input_grid)
    return _render(a, _analyze(a))


def _has_alternative(walls, path):
    # Any distinct simple terminal route omits an edge of the optimum. Blocking
    # edges also detects detours that keep every original route vertex.
    for edge in zip(path, path[1:]):
        other, _ = _route(walls, path[0], path[-1], forbidden_edge=edge)
        if other:
            assert len(other) > len(path)
            return True
    return False


def _orient(walls, start, goal, rng):
    tags = np.zeros(walls.shape, dtype=np.int8)
    tags[start], tags[goal] = 1, 2
    k = int(rng.integers(4))
    walls, tags = np.rot90(walls, k), np.rot90(tags, k)
    if rng.random() < .5:
        walls, tags = walls[:, ::-1], tags[:, ::-1]
    points = [tuple(map(int, np.argwhere(tags == n)[0])) for n in (1, 2)]
    return walls.copy(), *points


def _maze(rng):
    """Carve a fresh lattice maze, then add loops and occasional wider rooms."""
    nr, nc = map(int, rng.integers(4, 14, size=2))
    walls = np.ones((2 * nr + 1, 2 * nc + 1), dtype=bool)
    origin = (int(rng.integers(nr)), int(rng.integers(nc)))
    visited, stack = {origin}, [origin]
    walls[2 * origin[0] + 1, 2 * origin[1] + 1] = False
    while stack:
        r, c = stack[-1]
        options = [(rr, cc) for rr, cc in _neighbors((r, c), (nr, nc))
                   if (rr, cc) not in visited]
        if not options:
            stack.pop()
            continue
        rr, cc = options[int(rng.integers(len(options)))]
        walls[r + rr + 1, c + cc + 1] = False
        walls[2 * rr + 1, 2 * cc + 1] = False
        visited.add((rr, cc))
        stack.append((rr, cc))
    extra = []
    for r in range(nr):
        for c in range(nc):
            if r + 1 < nr and walls[2 * r + 2, 2 * c + 1]:
                extra.append((2 * r + 2, 2 * c + 1))
            if c + 1 < nc and walls[2 * r + 1, 2 * c + 2]:
                extra.append((2 * r + 1, 2 * c + 2))
    if extra:
        for i in rng.permutation(len(extra))[:int(rng.integers(1, min(9, len(extra)) + 1))]:
            walls[extra[int(i)]] = False
    if rng.random() < .25:
        r, c = int(rng.integers(1, 2 * nr - 1)), int(rng.integers(1, 2 * nc - 1))
        walls[r:r + 2, c:c + 2] = False
    h, w = walls.shape
    ports = ([(0, 2 * c + 1) for c in range(nc)]
             + [(h - 1, 2 * c + 1) for c in range(nc)]
             + [(2 * r + 1, 0) for r in range(nr)]
             + [(2 * r + 1, w - 1) for r in range(nr)])
    ix = rng.choice(len(ports), 2, replace=False)
    start, goal = (ports[int(i)] for i in ix)
    walls[start] = walls[goal] = False
    return _orient(walls, start, goal, rng)


def _barriers(rng):
    """Partial slabs, split barriers and compound wall stubs in open space."""
    h = int(rng.integers(8, 29))
    n = int(rng.integers(1, 6))
    thickness = [int(rng.choice((1, 1, 2, 3))) for _ in range(n)]
    gaps = [int(rng.choice((1, 1, 1, 2))) for _ in range(n - 1)]
    w = 2 + sum(thickness) + sum(gaps)
    walls = np.zeros((h, w), dtype=bool)
    c = 1
    for i, thick in enumerate(thickness):
        tip = int(rng.integers(2, h - 2))
        if i % 2:
            walls[tip:, c:c + thick] = True
        else:
            walls[:tip, c:c + thick] = True
        if thick == 1 and rng.random() < .3:
            walls[int(rng.integers(1, h - 1)), c] = False
        if rng.random() < .25 and c + thick < w - 1:
            rr = min(h - 2, tip + 1) if i % 2 else max(1, tip - 2)
            walls[rr, c + thick] = True
        c += thick + (gaps[i] if i < n - 1 else 0)
    start = (int(rng.choice((0, h - 1))), 0)
    goal = (int(rng.choice((0, h - 1))), w - 1)
    return _orient(walls, start, goal, rng)


def _decorate(walls, start, goal, rng, colors, slopes, need_cross, need_alternative):
    path, count = _route(walls, start, goal)
    if count != 1 or len(path) < 9:
        return None
    if len(path) - 1 <= abs(start[0] - goal[0]) + abs(start[1] - goal[1]):
        return None
    h, w = walls.shape
    path_set = set(path)
    choices = {-1: [], 1: []}
    all_bends = []
    for i, p in enumerate(path[1:-1], 1):
        slope = _bend_slope(path[i - 1], path[i + 1])
        if slope is None:
            continue
        all_bends.append(p)
        if p[0] in (0, h - 1) or p[1] in (0, w - 1):
            continue
        line = _diagonal(p, -slope, walls.shape)
        if line.intersection((start, goal)) or len(line) < 4:
            continue
        if not any(walls[q] for q in line):
            continue
        choices[-slope].append((p, line))
    if any(not choices[s] for s in slopes) or len(all_bends) < 2:
        return None
    chosen = []
    for s in slopes:
        pool = choices[s]
        if need_cross and not chosen:
            pool = [(p, line) for p, line in pool if (line & path_set) - {p}]
        if not pool:
            return None
        p, line = pool[int(rng.integers(len(pool)))]
        chosen.append((p, s, line))
    extra_pool = [(p, s, line) for s in set(slopes) for p, line in choices[s]
                  if p not in {item[0] for item in chosen}]
    extra_max = min(len(extra_pool), 3)
    extra_n = int(rng.integers(extra_max + 1)) if slopes else 0
    for i in rng.permutation(len(extra_pool)):
        if extra_n == 0:
            break
        p, s, line = extra_pool[int(i)]
        if any(line == item[2] for item in chosen):
            continue
        chosen.append((p, s, line))
        extra_n -= 1
    lines = set().union(*(item[2] for item in chosen))
    if len(path_set - lines - {start, goal}) < 3:
        return None
    if not (set(all_bends) - {item[0] for item in chosen}):
        # Unmarked bends are useful evidence; some small fallback layouts have
        # only two bends, so mixed tests may legitimately mark both of them.
        if len(slopes) < 2:
            return None
    free = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
            if not walls[r, c] and (r, c) not in path_set]
    inert_pool = [p for p in free if p not in lines]
    if not inert_pool:
        return None
    if need_alternative and not _has_alternative(walls, path):
        return None
    # Prefer a decoy at a genuine off-route elbow when one is available.
    false_bends = []
    for p in inert_pool:
        adjacent = [q for q in _neighbors(p, walls.shape) if not walls[q]]
        if len(adjacent) == 2 and _bend_slope(*adjacent) is not None:
            false_bends.append(p)
    if need_cross and not false_bends:
        return None
    pool = false_bends or inert_pool
    inert = [pool[int(rng.integers(len(pool)))]]
    desired = int(rng.integers(1, min(5, len(free)) + 1))
    for i in rng.permutation(len(free)):
        if len(inert) >= desired:
            break
        if free[int(i)] not in inert:
            inert.append(free[int(i)])
    terminal, secondary = colors
    a = walls.astype(np.int8)
    a[start] = a[goal] = terminal
    for p, _, _ in chosen:
        a[p] = secondary
    for p in inert:
        a[p] = secondary
    output = solve(a)
    if need_cross:
        p = inert[0]
        adjacent = [q for q in _neighbors(p, a.shape) if not walls[q]]
        slope = _bend_slope(*adjacent)
        if not any(output[q] != secondary for q in _diagonal(p, -slope, a.shape)):
            return None
    return {"input": a, "output": output}


def _fallback(rng, colors, slopes, need_cross, used):
    """Finite varied U detours, each with an exactly longer lower bypass.

    120 distinct shapes exceed the maximum 14 pairs, even before symmetries.
    This is only a termination fallback, never a stored puzzle bank.
    """
    shapes = [(h, w) for h in range(18, 30) for w in range(7, 17)]
    for i in rng.permutation(len(shapes)):
        h, w = shapes[int(i)]
        walls = np.ones((h, w), dtype=bool)
        walls[:h - 1, 1] = walls[:h - 1, w - 2] = False
        walls[h - 3:h - 1, 1:w - 1] = False
        # Without symmetry both emission slopes have a crossing witness.
        pair = _decorate(walls, (0, 1), (0, w - 2), rng, colors,
                         slopes, need_cross, True)
        if pair is not None and _key(pair["input"]) not in used:
            return pair
    raise RuntimeError("The finite fallback capacity was unexpectedly exhausted")


def _make_pair(rng, colors, slopes, need_cross, need_alternative, used):
    for _ in range(96):
        walls, start, goal = _maze(rng) if rng.random() < .60 else _barriers(rng)
        pair = _decorate(walls, start, goal, rng, colors, slopes,
                         need_cross, need_alternative)
        if pair is not None and _key(pair["input"]) not in used:
            return pair
    return _fallback(rng, colors, slopes, need_cross, used)


def _counts(num_train, num_test):
    for n, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                or not low <= int(n) <= high):
            raise ValueError("Counts must be integers in the documented ranges")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = rng.permutation(_DOT_COLORS)
    colors = tuple(map(int, palette[:2]))
    first_sign = int(rng.choice((-1, 1)))
    plan = [((first_sign,), True, True), ((-first_sign,), False, False)]
    while len(plan) < num_train:
        slopes = () if rng.random() < .30 else (int(rng.choice((-1, 1))),)
        plan.append((slopes, False, False))
    rng.shuffle(plan)
    puzzle, used = {"train": [], "test": []}, set()
    for split, specifications in (("train", plan),
                                  ("test", [((-1, 1), False, True)] * num_test)):
        for slopes, need_cross, alternative in specifications:
            pair = _make_pair(rng, colors, slopes, need_cross, alternative, used)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    used, roles, training_slopes = set(), set(), set()
    alternative_witness = crossing_witness = unmarked_witness = decoy_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in used
            used.add(_key(a))
            info = _analyze(a)
            roles.add((info["terminal"], info["secondary"]))
            assert info["secondary"] is not None and info["inert"]
            expected = _render(a, info)
            assert np.array_equal(expected, b), "Incorrect route/diagonal output"
            line_union = set().union(*info["lines"])
            assert info["inert"] - line_union, "An unobscured inert marker is required"
            assert np.all(b[a == info["terminal"]] == info["terminal"])
            assert np.all(b[a == info["secondary"]] == info["secondary"])
            assert int(np.sum(b == 4)) >= 3
            assert len(info["path"]) - 1 > sum(abs(x - y) for x, y in zip(*info["endpoints"]))
            active_slopes = {s for _, s in info["active"]}
            for line in info["lines"]:
                assert any(a[p] == 1 for p in line), "Show diagonal passage through walls"
                assert all(b[p] == info["secondary"] for p in line)
            if split == "train":
                assert len(active_slopes) <= 1
                training_slopes.update(active_slopes)
                if not alternative_witness:
                    alternative_witness = _has_alternative(a == 1, info["path"])
                crossing_witness |= any((line & info["path_set"]) - {p}
                                        for (p, _), line in zip(info["active"], info["lines"]))
                unmarked_witness |= any(_bend_slope(info["path"][i - 1], info["path"][i + 1])
                                        is not None and a[p] == 0
                                        for i, p in enumerate(info["path"][1:-1], 1))
                for p in info["inert"] - line_union:
                    adjacent = [q for q in _neighbors(p, a.shape) if a[q] != 1]
                    slope = _bend_slope(*adjacent) if len(adjacent) == 2 else None
                    decoy_witness |= slope is not None and any(
                        b[q] != info["secondary"] for q in _diagonal(p, -slope, a.shape))
            else:
                assert active_slopes == {-1, 1}
                assert _has_alternative(a == 1, info["path"])
            # Omitting B leaves a route-only picture, observably different.
            route_only = a.copy()
            for p in info["path"][1:-1]:
                if a[p] == 0:
                    route_only[p] = 4
            if info["active"]:
                assert not np.array_equal(route_only, b)
                # If overlays were impassable, the actual answer's path is lost.
                blocked_path, _ = _route((a == 1) | (a == info["secondary"]), *info["endpoints"])
                assert blocked_path != info["path"]
            # Omitting A supplies no route to activate B and yields the input.
            assert not np.array_equal(a, b)
    assert len(roles) == 1 and training_slopes == {-1, 1}
    assert alternative_witness and crossing_witness and unmarked_witness and decoy_witness


def self_test():
    """42 puzzles streamed in memory; no files, caches or unbounded retries."""
    shapes, structures, palettes, lengths, counts, fingerprints = (set() for _ in range(6))
    zero, multiple, route_crossings, false_bends = 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                info = _analyze(a)
                shapes.add(a.shape)
                lengths.add(len(info["path"]))
                counts.add(len(info["active"]))
                palettes.add((info["terminal"], info["secondary"]))
                normalized = a.copy()
                normalized[a == info["terminal"]] = 2
                normalized[a == info["secondary"]] = 3
                structures.add(_key(normalized))
                digest.update(str(a.shape).encode())
                digest.update(a.tobytes())
                digest.update(pair["output"].tobytes())
                zero += not info["active"]
                multiple += len(info["active"]) >= 3
                route_crossings += any((line & info["path_set"]) - {p}
                                       for (p, _), line in zip(info["active"], info["lines"]))
                for p in info["inert"]:
                    neighbors = [q for q in _neighbors(p, a.shape) if a[q] != 1]
                    false_bends += len(neighbors) == 2 and _bend_slope(*neighbors) is not None
                assert np.array_equal(solve(a.tolist()), pair["output"])
        fingerprints.add(digest.digest())
    assert len(fingerprints) == 42
    assert len(shapes) >= 35 and len(structures) >= 170
    assert len(palettes) >= 20 and len(lengths) >= 40
    assert {0, 1, 2, 3} <= counts
    assert zero >= 10 and multiple >= 15 and route_crossings >= 42 and false_bends >= 10
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(203 + nt * 7 + nq, nt, nq)
            assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    p, q = generate_puzzle(1234, 2, 1), generate_puzzle(1234, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
               for k in ("input", "output"))
    # Explicit semantic witnesses independent of random construction.
    walls = np.ones((20, 11), dtype=bool)
    walls[:19, 1] = walls[:19, 9] = False
    walls[17:19, 1:10] = False
    a = walls.astype(np.int8)
    a[0, 1] = a[0, 9] = 2
    a[17, 1] = a[17, 9] = a[18, 1] = 3
    b = solve(a)
    # Both active bends emit opposite chords; an off-route elbow stays inert.
    assert b[16, 2] == b[16, 8] == 3
    assert b[18, 2] == 0 and b[19, 2] == 1
    assert b[17, 2] == 4 and b[19, 0] == 1  # no line from off-route elbow
    assert b[9, 9] == b[9, 1] == 3  # secondary beats route ink
    assert b[0, 1] == b[0, 9] == 2
    assert np.array_equal(solve(np.rot90(a)), np.rot90(b))
    assert np.array_equal(solve(a[:, ::-1]), b[:, ::-1])
    no_secondary = a.copy()
    no_secondary[a == 3] = 0
    assert set(map(int, np.unique(solve(no_secondary)))) == {0, 1, 2, 4}
    # Add a dead-end elbow lying on an emitted line. It stays inert; its own
    # perpendicular continuation would incorrectly paint the wall at (14, 2).
    nonrecursive = a.copy()
    nonrecursive[15, 3] = 3
    nonrecursive[14, 3] = nonrecursive[15, 2] = 0
    assert solve(nonrecursive)[14, 2] == 1
    assert (15, 3) in _analyze(nonrecursive)["inert"]
    assert (15, 3) in set().union(*_analyze(nonrecursive)["lines"])
    # Removing the right-hand active dot keeps the route but removes its line.
    removed = a.copy()
    removed[17, 9] = 0
    assert solve(removed)[16, 8] == 1
    # A longer bypass can keep all optimum vertices: edge deletion detects it.
    open_square = np.zeros((2, 2), dtype=bool)
    direct, nways = _route(open_square, (0, 0), (0, 1))
    assert nways == 1 and _has_alternative(open_square, direct)
    used = set()
    for index in range(14):
        slopes = (-1, 1) if index % 3 == 0 else ((1,) if index % 3 == 1 else ())
        pair = _fallback(np.random.default_rng(index), (2, 3), slopes,
                         len(slopes) == 1, used)
        assert _key(pair["input"]) not in used
        used.add(_key(pair["input"]))
        assert np.array_equal(solve(pair["input"]), pair["output"])
    # A marker on a straight part has no diagonal reference, so is out of family.
    invalid = a.copy()
    invalid[5, 1] = 3
    try:
        solve(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Straight-route marker accepted")
    tie = np.zeros((5, 5), dtype=np.int8)
    tie[0, 0] = tie[-1, -1] = 2
    try:
        solve(tie)
    except ValueError:
        pass
    else:
        raise AssertionError("Tied route accepted")
    p["test"][0]["output"][0, 0] = 4
    try:
        validate_generated_puzzle(p)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Corrupted target accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "normalized_inputs": len(structures),
            "role_palettes": len(palettes), "route_lengths": len(lengths),
            "active_counts": sorted(counts), "zero_active_pairs": zero,
            "three_or_more_active_pairs": multiple, "route_crossing_pairs": route_crossings,
            "inert_elbow_markers": false_bends}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
