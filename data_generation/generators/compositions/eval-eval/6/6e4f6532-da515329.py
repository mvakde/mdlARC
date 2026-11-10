"""Pinwheel transport: registration docking in a compass carried by a winding.

No parent module, saved pair, seed, or training example is used by solve().
All generation is bounded; diagnostics are aggregates, never execution traces.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


# Clockwise order, in screen coordinates.
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _check_grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid of size 1..30 and colors 0..9")
    return a.astype(np.int8, copy=False)


def _integer(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"illegal {name}")
    return int(value)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _transform(p, t):
    r, c = p
    if t >= 4:
        c = -c
    for _ in range(t % 4):
        r, c = c, -r
    return r, c


def _turn(p, quarter_turns):
    r, c = p
    for _ in range(quarter_turns % 4):
        r, c = c, -r
    return r, c


def _shift(cells, origin):
    return {(r + origin[0], c + origin[1]): v for (r, c), v in cells.items()}


def _halo(points):
    return {(r + dr, c + dc) for r, c in points for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)}


def _components(points):
    left = set(points)
    result = []
    while left:
        p = min(left)
        left.remove(p)
        stack, component = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in NEIGHBORS:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    stack.append(q)
                    component.add(q)
        result.append(component)
    return result


def _cross(length, center):
    return {(center[0] + n * dr, center[1] + n * dc)
            for dr, dc in DIRS for n in range(1, length + 1)}


def _tracks(shape, center, length, thickness):
    """Return visible cells with (radial phase, outgoing direction, stage, part).

    The walk continues outside the window and can re-enter. Only stored cells
    are clipped. Part 0 is the long forward run, 1 the inward jog, 2 the tail.
    Corners receive their outgoing, rather than incoming, direction.
    """
    h, w = shape
    cy, cx = center
    radius = max(2, length)
    extent = max(cy, cx, h - 1 - cy, w - 1 - cx)
    stages = 4 * (extent // 8 + 2)
    cells = {}

    def put(y, x, phase, direction, stage, part):
        p = y + cy, x + cx
        if thickness <= p[0] < h - thickness and thickness <= p[1] < w - thickness:
            if p in cells and cells[p][0] != phase:
                raise ValueError("intersecting winding phases")
            cells[p] = phase, direction, stage, part

    for phase, (dy, dx) in enumerate(DIRS):
        for n in range(1, radius):
            put(n * dy, n * dx, phase, phase, -1, -1)
        y, x = radius * dy, radius * dx
        direction = (phase + 1) % 4
        for k in range(stages):
            for part, (d, steps) in enumerate(((direction, 4 * k + 2),
                                              ((direction + 1) % 4, radius - 2),
                                              (direction, radius))):
                vy, vx = DIRS[d]
                for _ in range(steps):
                    put(y, x, phase, d, k, part)
                    y, x = y + vy, x + vx
            direction = (direction + 1) % 4
    return cells


def _analytic_mask(shape, center, length, thickness):
    """Interval cross-check of the winding geometry, without path walking."""
    y, x = np.indices(shape)
    y, x = y - center[0], x - center[1]
    radius = max(2, length)
    result = np.zeros(shape, dtype=bool)
    bound = max(int(np.abs(y).max()), int(np.abs(x).max()))
    for _ in range(4):
        result |= (x == 0) & (-radius <= y) & (y <= -1)
        for cycle in range(bound // 8 + 3):
            starts = ((-radius - 8 * cycle, -8 * cycle),
                      (-2 - 8 * cycle, radius + 8 * cycle + 2),
                      (radius + 8 * cycle + 4, 8 * cycle + 4),
                      (8 * cycle + 6, -radius - 8 * cycle - 6))
            for stage, ((sy, sx), (dy, dx)) in enumerate(zip(starts, DIRS[1:] + DIRS[:1])):
                along = (y - sy) * dy + (x - sx) * dx
                across = (y - sy) * dx - (x - sx) * dy
                run = 16 * cycle + 4 * stage + 2
                result |= (across == 0) & (0 <= along) & (along <= run)
                result |= (along == run) & (0 <= across) & (across <= radius - 2)
                result |= (across == radius - 2) & (run <= along) & (along <= run + radius)
        y, x = x, -y
    result[:thickness] = result[-thickness:] = False
    result[:, :thickness] = result[:, -thickness:] = False
    return result


def _frame(shape, palette, thickness):
    bg, _, _, _, _, north, east, south, west = palette
    a = np.full(shape, bg, dtype=np.int8)
    # Background corner squares make the compass convention rotation-equivariant.
    a[thickness:-thickness, :thickness] = west
    a[thickness:-thickness, -thickness:] = east
    a[:thickness, thickness:-thickness] = north
    a[-thickness:, thickness:-thickness] = south
    return a


def _inside(points, shape, thickness):
    return all(thickness <= r < shape[0] - thickness
               and thickness <= c < shape[1] - thickness for r, c in points)


def _read_cross(grid, color):
    points = set(map(tuple, np.argwhere(grid == color)))
    if len(points) not in (4, 8, 12, 16):
        return None
    length = len(points) // 4
    rows = Counter(r for r, _ in points)
    cols = Counter(c for _, c in points)
    ys = [r for r, n in rows.items() if n == 2 * length]
    xs = [c for c, n in cols.items() if n == 2 * length]
    if len(ys) != 1 or len(xs) != 1:
        return None
    center = ys[0], xs[0]
    return (length, center) if points == _cross(length, center) else None


def _parse(value):
    a = _check_grid(value)
    h, w = a.shape
    if min(h, w) < 9:
        raise ValueError("canvas too small for a framed winding and pieces")
    north, south = int(a[0, w // 2]), int(a[-1, w // 2])
    east, west = int(a[h // 2, -1]), int(a[h // 2, 0])
    bands = (north, east, south, west)
    assert len(set(bands)) == 4
    t = 0
    while t < h and np.all(a[t, 2:-2] == north):
        t += 1
    assert t in (1, 2)
    interior = a[t:-t, t:-t]
    values, counts = np.unique(interior, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    assert bg not in bands and sum(counts == counts.max()) == 1
    expected_frame = _frame(a.shape, (bg, 0, 0, 0, 0, *bands), t)
    structural = np.ones(a.shape, dtype=bool)
    structural[t:-t, t:-t] = False
    assert np.array_equal(a[structural], expected_frame[structural])
    candidates = []
    for color in sorted(set(int(v) for v in values) - {bg} - set(bands)):
        cross = _read_cross(a, color)
        if cross is not None:
            candidates.append((color, *cross))
    assert len(candidates) == 1, "one uniquely colored hollow cross is required"
    track, length, center = candidates[0]
    assert a[center] == bg
    mask = (a != bg) & (a != track) & ~structural
    objects = [{p: int(a[p]) for p in component}
               for component in _components(map(tuple, np.argwhere(mask)))]
    targets = [obj for obj in objects if len(set(obj.values())) == 2]
    sources = [obj for obj in objects if len(set(obj.values())) > 2]
    assert targets and len(sources) == len(targets) and len(objects) == 2 * len(targets)
    role_pairs = []
    for target in targets:
        hist = Counter(target.values())
        singleton = [v for v, n in hist.items() if n == 1]
        assert len(singleton) == 1
        pivot = singleton[0]
        reg = next(v for v in hist if v != pivot)
        assert 2 <= hist[reg] <= 4
        role_pairs.append((pivot, reg))
    assert len(set(role_pairs)) == 1
    pivot, reg = role_pairs[0]
    body_roles = set().union(*(set(s.values()) for s in sources)) - set(bands) - {pivot, reg}
    assert len(body_roles) == 1
    body = next(iter(body_roles))
    assert len({bg, track, pivot, reg, body, *bands}) == 9
    routes = _tracks(a.shape, center, length, t)
    for target in targets:
        p = next(p for p, v in target.items() if v == pivot)
        assert p in routes and routes[p][2] >= 0
        phase, direction, _, part = routes[p]
        dr, dc = DIRS[direction]
        before, after = routes.get((p[0] - dr, p[1] - dc)), routes.get((p[0] + dr, p[1] + dc))
        assert part in (0, 2) and before is not None and after is not None
        assert before[:2] == (phase, direction) == after[:2], "station must be on a straight run"
    for source in sources:
        assert list(source.values()).count(pivot) == 1
        assert list(source.values()).count(reg) in (2, 3, 4)
        assert body in source.values()
        p = next(p for p, v in source.items() if v == pivot)
        directions = []
        for q, color in source.items():
            if color in bands:
                dr, dc = q[0] - p[0], q[1] - p[1]
                assert (dr == 0) != (dc == 0)
                d = DIRS.index((int(np.sign(dr)), int(np.sign(dc))))
                directions.append(d)
        assert len(directions) >= 2 and len(set(directions)) == len(directions)
        assert any((d + 1) % 4 in directions for d in directions)
    return {"grid": a, "bg": bg, "track": track, "body": body, "pivot": pivot,
            "reg": reg, "bands": bands, "t": t, "length": length, "center": center,
            "structural": structural, "targets": targets, "sources": sources, "routes": routes}


def _placements(info, use_labels=True, transported=True, limit=2):
    """Enumerate resulting colored maps; no hidden tie-breaking convention."""
    pivot, reg, bands = info["pivot"], info["reg"], info["bands"]
    choices = []
    protected = _cross(info["length"], info["center"]) | {info["center"]}
    for source in info["sources"]:
        root = next(p for p, v in source.items() if v == pivot)
        options, seen = [], set()
        relative = {(p[0] - root[0], p[1] - root[1]): v for p, v in source.items()}
        for target_index, target in enumerate(info["targets"]):
            anchor = next(p for p, v in target.items() if v == pivot)
            phase, direction, _, _ = info["routes"][anchor]
            rotation = (direction - phase) % 4 if transported else 0
            for t in range(8):
                cells = _shift({_transform(p, t): v for p, v in relative.items()}, anchor)
                marks = {p: v for p, v in cells.items() if v in (pivot, reg)}
                if marks != target or not _inside(cells, info["grid"].shape, info["t"]):
                    continue
                if cells.keys() & protected:
                    continue
                if use_labels and any(
                    (p[0] - anchor[0], p[1] - anchor[1]) !=
                    tuple(n * x for x in _turn(DIRS[bands.index(v)], rotation))
                    for p, v in cells.items() if v in bands
                    for n in (abs(p[0] - anchor[0]) + abs(p[1] - anchor[1]),)
                ):
                    continue
                signature = tuple(sorted(cells.items()))
                if signature not in seen:
                    options.append((target_index, cells))
                    seen.add(signature)
        choices.append(options)
    order = sorted(range(len(choices)), key=lambda i: len(choices[i]))
    answers = {}

    def search(depth, used, occupied, selected):
        if len(answers) >= limit:
            return
        if depth == len(order):
            answers[tuple(sorted(occupied.items()))] = list(selected)
            return
        for target_index, cells in choices[order[depth]]:
            if target_index in used or cells.keys() & occupied.keys():
                continue
            search(depth + 1, used | {target_index}, {**occupied, **cells}, selected + [cells])

    search(0, set(), {}, [])
    return list(answers.values())


def _render(info, placed, analytic=False):
    out = np.full_like(info["grid"], info["bg"])
    out[info["structural"]] = info["grid"][info["structural"]]
    if analytic:
        out[_analytic_mask(out.shape, info["center"], info["length"], info["t"])] = info["track"]
    else:
        for p in info["routes"]:
            out[p] = info["track"]
    for piece in placed:
        for p, v in piece.items():
            out[p] = v
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this grid alone, including local route frames and D4 docking."""
    info = _parse(input_grid)
    answers = _placements(info)
    if len(answers) != 1:
        raise ValueError("input must determine one transported docking output")
    return _render(info, answers[0])


def _pattern_key(points):
    return min(tuple(sorted(_transform(p, t) for p in points)) for t in range(8))


def _pattern(rng, index, used):
    if index == 0:
        points = {(-1, 0), (1, 0)}
    elif index == 1:
        points = {(0, 1), (1, 0)}
    else:
        for _ in range(40):
            # A procedural rooted, 8-connected cluster; the root is the pivot.
            cluster = {(0, 0)}
            for _ in range(int(rng.integers(2, 5))):
                frontier = sorted({(r + dr, c + dc) for r, c in cluster
                                   for dr, dc in NEIGHBORS
                                   if abs(r + dr) <= 2 and abs(c + dc) <= 2} - cluster)
                cluster.add(frontier[int(rng.integers(len(frontier)))])
            points = cluster - {(0, 0)}
            if _pattern_key(points) not in used:
                break
        else:
            points = {(-1, 0), (0, 1), (1, 0), (0, -1)}
    used.add(_pattern_key(points))
    t = int(rng.integers(8))
    return {_transform(p, t) for p in points}


def _piece(rng, pattern, palette, rotation):
    _, _, body, pivot, reg, *bands = palette
    # Extents and branches are independent of the registration vocabulary.
    extents = [int(rng.choice([1, 2, 3], p=[.25, .65, .10])) for _ in DIRS]
    for r, c in pattern:
        for d, amount in ((0, -r), (1, c), (2, r), (3, -c)):
            extents[d] = max(extents[d], amount)
    core = {(0, 0)} | set(pattern)
    for r, c in list(core):
        # Manhattan connectors preserve diagonal marks without disconnected bodies.
        core.update((y, 0) for y in range(min(0, r), max(0, r) + 1))
        core.update((r, x) for x in range(min(0, c), max(0, c) + 1))
    for d, (dr, dc) in enumerate(DIRS):
        core.update((n * dr, n * dc) for n in range(1, extents[d] + 1))
    area = (extents[0] + extents[2] + 1) * (extents[1] + extents[3] + 1)
    desired = int(area * rng.uniform(.36, .72))
    for _ in range(area):
        if len(core) >= desired:
            break
        frontier = sorted({(r + dr, c + dc) for r, c in core for dr, dc in DIRS
                           if -extents[0] <= r + dr <= extents[2]
                           and -extents[3] <= c + dc <= extents[1]} - core)
        if not frontier:
            break
        core.add(frontier[int(rng.integers(len(frontier)))])
    cells = {p: body for p in core}
    cells.update({p: reg for p in pattern})
    cells[(0, 0)] = pivot
    start = int(rng.integers(4))
    labels = {(start + i) % 4 for i in range(int(rng.integers(2, 5)))}
    for label in sorted(labels):
        direction = (label + rotation) % 4
        dr, dc = DIRS[direction]
        n = extents[direction] + 1
        cells[(n * dr, n * dc)] = bands[label]
    return cells


def _construct(rng, length, count, allowed_stages, first_stages, demonstration):
    palette = tuple(int(v) for v in rng.permutation(10)[:9])
    bg, track, _, pivot, reg, *_ = palette
    thickness = int(rng.choice([1, 2], p=[.75, .25]))
    low = 21 if count < 3 else 25
    shape = tuple(int(v) for v in rng.integers(low, 31, size=2))
    if rng.random() < .15:
        shape = (30, 30)
    margin = thickness + max(2, length) + 2
    center = tuple(int(rng.integers(margin, side - margin)) for side in shape)
    base = _frame(shape, palette, thickness)
    seed = _cross(length, center)
    protected = seed | {center}
    routes = _tracks(shape, center, length, thickness)
    stations = []
    for p, data in sorted(routes.items()):
        phase, direction, stage, part = data
        dr, dc = DIRS[direction]
        previous = routes.get((p[0] - dr, p[1] - dc))
        following = routes.get((p[0] + dr, p[1] + dc))
        if (part in (0, 2) and previous is not None and following is not None
                and previous[:2] == data[:2] == following[:2]):
            stations.append((p, data))
    finals, occupied, patterns = [], set(), set()
    for index in range(count):
        pattern = _pattern(rng, index, patterns)
        options = [s for s in stations if s[1][2] in (first_stages if index == 0 else allowed_stages)]
        rng.shuffle(options)
        found = False
        for anchor, (phase, direction, _, _) in options:
            cells = _piece(rng, pattern, palette, (direction - phase) % 4)
            placed = _shift(cells, anchor)
            if (not _inside(placed, shape, thickness) or placed.keys() & occupied
                    or placed.keys() & protected):
                continue
            finals.append(placed)
            occupied.update(_halo(placed))
            found = True
            break
        if not found:
            return None
    targets = {p: v for piece in finals for p, v in piece.items() if v in (pivot, reg)}
    blocked = _halo(targets) | _halo(seed)
    sources = []
    for index, final in enumerate(finals):
        anchor = next(p for p, v in final.items() if v == pivot)
        relative = {(p[0] - anchor[0], p[1] - anchor[1]): v for p, v in final.items()}
        if index == 0 and demonstration == "rotation":
            t = int(rng.integers(1, 4))
        elif index == 0 and demonstration == "reflection":
            t = int(rng.integers(4, 8))
        else:
            t = int(rng.integers(8))
        cells = {_transform(p, t): v for p, v in relative.items()}
        r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
        c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
        origins = [(r, c) for r in range(thickness - r0, shape[0] - thickness - r1)
                   for c in range(thickness - c0, shape[1] - thickness - c1)]
        rng.shuffle(origins)
        for origin in origins:
            placed = _shift(cells, origin)
            if not placed.keys() & blocked and origin != anchor:
                sources.append(placed)
                blocked.update(_halo(placed))
                break
        else:
            return None
    inp = base.copy()
    for p in seed:
        inp[p] = track
    for cells in [targets] + sources:
        for p, v in cells.items():
            inp[p] = v
    info = _parse(inp)
    answers = _placements(info)
    if len(answers) != 1:
        return None
    out = _render(info, finals)
    if not np.array_equal(out, _render(info, answers[0])):
        return None
    # The symmetric first stencil must actually require compass evidence.
    if len(_placements(info, use_labels=False)) < 2:
        return None
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ntrain = _integer(num_train, 2, 10, "num_train")
    ntest = _integer(num_test, 1, 4, "num_test")
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    regime = str(rng.choice(["later", "load", "scale"]))
    lengths = [2, 3]
    lengths += [int(rng.choice([1, 2, 3] if regime == "scale" else [1, 2, 3, 4]))
                for _ in range(ntrain - 2)]
    test_lengths = [4 if regime == "scale" else int(rng.choice(sorted(set(lengths))))
                    for _ in range(ntest)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, split_lengths in (("train", lengths), ("test", test_lengths)):
        for i, length in enumerate(split_lengths):
            if split == "train":
                count = i + 1 if i < 2 else int(rng.integers(1, 3 if regime == "load" else 4))
                stages = {0, 1} if regime == "later" else {0, 1, 2, 3}
                # One explicit half-turn witness, then no shape-to-frame codebook.
                first = ({1} if i == 0 else (stages - {1}) if i == 1 else stages)
                demonstration = "rotation" if i == 0 else "reflection" if i == 1 else None
            else:
                count = 3 if regime == "load" else int(rng.integers(1, 4))
                stages = {1, 2, 3, 4} if regime == "later" else {0, 1, 2, 3}
                first = {2, 3} if regime == "later" else stages
                demonstration = None
            for _ in range(100):
                pair = _construct(rng, length, count, stages, first, demonstration)
                if pair is not None and _key(pair["input"]) not in seen:
                    puzzle[split].append(pair)
                    seen.add(_key(pair["input"]))
                    break
            else:
                raise RuntimeError("bounded construction budget exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(pair):
    a, b = _check_grid(pair["input"]), _check_grid(pair["output"])
    assert a.shape == b.shape
    info = _parse(a)
    answers = _placements(info)
    assert len(answers) == 1
    placed = answers[0]
    assert np.array_equal(b, _render(info, placed, analytic=True)), "wrong winding or docking"
    assert not np.array_equal(a, b)
    assert set(np.unique(a)) == set(np.unique(b)), "palette must be preserved"
    assert b[info["center"]] == info["bg"]
    assert all(b[p] == info["track"] for p in _cross(info["length"], info["center"]))
    assert np.array_equal(a[info["structural"]], b[info["structural"]])
    assert all(b[p] == v for target in info["targets"] for p, v in target.items())
    assert len(_placements(info, use_labels=False)) >= 2, "compass became dispensable"
    global_answers = _placements(info, transported=False)
    global_differs = len(global_answers) != 1 or not np.array_equal(b, _render(info, global_answers[0]))
    # Reconstruct actual metrics rather than reporting sampled preferences.
    target_pivots = [next(p for p, v in target.items() if v == info["pivot"])
                     for target in info["targets"]]
    old = set().union(*(set(s) for s in info["sources"]))
    new = set().union(*(set(s) for s in placed))
    hole_track = 0
    for piece in placed:
        for r in range(min(r for r, _ in piece), max(r for r, _ in piece) + 1):
            for c in range(min(c for _, c in piece), max(c for _, c in piece) + 1):
                hole_track += int((r, c) not in piece and b[r, c] == info["track"])
    poses = []
    for source in info["sources"]:
        root = next(p for p, v in source.items() if v == info["pivot"])
        rel = {(p[0] - root[0], p[1] - root[1]): v for p, v in source.items()}
        matches = []
        for final in placed:
            dest = next(p for p, v in final.items() if v == info["pivot"])
            matches += [t for t in range(8)
                        if _shift({_transform(p, t): v for p, v in rel.items()}, dest) == final]
        assert matches
        poses.append("translation" if 0 in matches else "rotation" if min(matches) < 4 else "reflection")
    return {"shape": a.shape, "length": info["length"], "thickness": info["t"],
            "count": len(placed), "bg": info["bg"], "track": info["track"],
            "stages": [info["routes"][p][2] for p in target_pivots],
            "phases": [info["routes"][p][0] for p in target_pivots],
            "rotations": [(info["routes"][p][1] - info["routes"][p][0]) % 4 for p in target_pivots],
            "global_differs": global_differs, "overlap": len(old & new),
            "track_in_gaps": hole_track, "moves": poses,
            "patterns": [_pattern_key({(p[0] - root[0], p[1] - root[1])
                                       for p, v in target.items() if v == info["reg"]})
                         for target, root in zip(info["targets"], target_pivots)],
            "piece_shapes": [(max(r for r, _ in p) - min(r for r, _ in p) + 1,
                              max(c for _, c in p) - min(c for _, c in p) + 1) for p in placed]}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a = _check_grid(pair["input"])
            assert _key(a) not in seen
            seen.add(_key(a))
            record = _inspect(pair)
            if split == "train":
                training.append(record)
            else:
                testing.append(record)
    assert {2, 3} <= {r["length"] for r in training}
    assert any(r["global_differs"] for r in training)
    assert any("rotation" in r["moves"] for r in training)
    assert any("reflection" in r["moves"] for r in training)
    assert len({q for r in training for q in r["rotations"]}) >= 2, "constant port rotation shortcut"
    line = _pattern_key({(-1, 0), (1, 0)})
    assert len({q for r in training for p, q in zip(r["patterns"], r["rotations"])
                if p == line}) >= 2, "stencil-to-frame shortcut"
    # Equal-count, noncongruent rooted stencils defeat registration counting.
    assert any(any(len(p) == len(q) and p != q for i, p in enumerate(r["patterns"])
                   for q in r["patterns"][i + 1:]) for r in training)
    later = (all(max(r["stages"]) <= 1 for r in training)
             and all(max(r["stages"]) >= 2 for r in testing))
    more_load = max(r["count"] for r in training) <= 2 and all(r["count"] == 3 for r in testing)
    new_scale = max(r["length"] for r in training) == 3 and all(r["length"] == 4 for r in testing)
    assert later or more_load or new_scale, "missing a supported, visible train/test shift"


def self_test():
    """Stream 42 seeds, boundary counts and semantic/metamorphic checks; no I/O."""
    shapes, patterns, piece_shapes, signatures = set(), set(), set(), set()
    lengths, phases, rotations, thicknesses, moves = set(), set(), set(), set(), set()
    backgrounds, track_colors = set(), set()
    totals = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                record = _inspect(pair)
                shapes.add(record["shape"])
                patterns.update(record["patterns"])
                piece_shapes.update(record["piece_shapes"])
                lengths.add(record["length"])
                phases.update(record["phases"])
                rotations.update(record["rotations"])
                thicknesses.add(record["thickness"])
                moves.update(record["moves"])
                backgrounds.add(record["bg"])
                track_colors.add(record["track"])
                totals["pairs"] += 1
                totals["pieces"] += record["count"]
                totals["overlap_pairs"] += int(record["overlap"] > 0)
                totals["gap_track_pairs"] += int(record["track_in_gaps"] > 0)
                totals["global_compass_failures"] += int(record["global_differs"])
        # Whole-scene rotation transports the compass and preserves clockwise winding.
        a, b = puzzle["test"][0]["input"], puzzle["test"][0]["output"]
        assert np.array_equal(solve(np.rot90(a)), np.rot90(b))
        perm = np.array([7, 4, 9, 0, 3, 8, 2, 5, 1, 6], dtype=np.int8)
        assert np.array_equal(solve(perm[a]), perm[b])
    assert len(signatures) == 42 and len(shapes) >= 30
    assert len(patterns) >= 15 and len(piece_shapes) >= 15
    assert lengths == {1, 2, 3, 4} and phases == rotations == {0, 1, 2, 3}
    assert thicknesses == {1, 2} and len(moves) == 3
    assert backgrounds == track_colors == set(range(10))
    assert totals["overlap_pairs"] > 0 and totals["gap_track_pairs"] > 0
    assert totals["global_compass_failures"] >= 84
    for length in range(1, 5):
        for shape, center in (((30, 30), (length + 2, length + 2)),
                              ((21, 29), (15, 20)), ((30, 20), (15, 10))):
            for t in (1, 2):
                mask = np.zeros(shape, dtype=bool)
                for p in _tracks(shape, center, length, t):
                    mask[p] = True
                assert np.array_equal(mask, _analytic_mask(shape, center, length, t))
    a = generate_puzzle(777, 6, 3)
    assert _digest(a) == _digest(generate_puzzle(777, 6, 3))
    assert _digest(a) != _digest(generate_puzzle(778, 6, 3))
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(9000 + ntrain + ntest, ntrain, ntest))
    for ntrain, ntest in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                          (True, 1), (4, False), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Mutating a target pivot is detected while grid schema remains legal.
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    info = _parse(broken["test"][0]["input"])
    root = next(p for p, v in info["targets"][0].items() if v == info["pivot"])
    broken["test"][0]["output"][root] = info["bg"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong docking output accepted")
    return {"seeds": 42, **dict(totals), "shapes": len(shapes),
            "rooted_patterns": len(patterns), "piece_shapes": len(piece_shapes),
            "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
