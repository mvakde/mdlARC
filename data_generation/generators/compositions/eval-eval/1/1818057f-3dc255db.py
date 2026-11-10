"""Competing siphons: split the simultaneous plus union by nearest intakes.

Only NumPy and the standard library are used. Generation metadata is never
consulted by solve(). Diagnostics are streamed; this module writes no files.
"""
from __future__ import annotations

import hashlib
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # U, R, D, L
HOST_SEEDS = (0,361842110,943014165,72904005,953611017,467525769,436272835,285104483,198182007,384858663,491477137,254673665,146737863,950443129,56686186,30176446,770655606,522626907,782617102,217931325,953708343,805767586,838371280,562117153,703225259,746256972,861505918,606908503,842964426,231377937,274676318,775320440,29699329,550695153,776707666,888824499,752305488,799387510,511071077,909596967,750776288,531040811)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def _components(points):
    remaining, result = set(points), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, comp = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (r + dr, c + dc)
                    if p in remaining:
                        remaining.remove(p)
                        comp.add(p)
                        stack.append(p)
        result.append(comp)
    return result


def _local(p, direction):
    r, c = p
    return ((r, c), (-c, r), (-r, -c), (c, -r))[direction]


def _world(p, direction):
    y, x = p
    return ((y, x), (x, -y), (-y, -x), (-x, y))[direction]


def _envelope(frame, direction):
    """Read a thin, monotone pointed frame; no sampled profile is needed."""
    local = {_local(p, direction) for p in frame}
    y0, y1 = min(y for y, x in local), max(y for y, x in local)
    if y1 - y0 < 3:
        return None
    rows = [sorted(x for y, x in local if y == level)
            for level in range(y0, y1 + 1)]
    if len(rows[0]) != 1:
        return None
    x0 = rows[0][0]
    lo = hi = x0
    split, cavity = False, set()
    for offset, row in enumerate(rows[1:], 1):
        if len(row) == 1 and not split and row[0] == x0:
            continue
        if len(row) != 2 or row[1] - row[0] < 2:
            return None
        left, right = row
        if not (0 <= lo - left <= 1 and 0 <= right - hi <= 1):
            return None
        lo, hi, split = left, right, True
        cavity.update(_world((y0 + offset, x), direction)
                      for x in range(left + 1, right))
    if not split or hi - lo < 4 or not lo < x0 < hi:
        return None
    # Intake lies at the open end, on the tip's axis, not at a hidden centroid.
    intake = _world((y1, x0), direction)
    if intake not in cavity:
        return None
    tip = _world((y0, x0), direction)
    return tip, intake, cavity


def _ray(shape, tip, direction, quantity=None):
    dr, dc = DIRS[direction]
    r, c = tip
    cells = []
    if quantity == 0:
        return cells
    while True:
        r, c = r + dr, c + dc
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            break
        cells.append((r, c))
        if quantity is not None and len(cells) >= quantity:
            break
    return cells


def _read(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if int(counts.max()) <= a.size // 2:
        raise ValueError("background must occupy more than half the canvas")
    possibilities = []
    for color in map(int, values):
        if color == bg:
            continue
        ports, valid = [], True
        for frame in _components(map(tuple, np.argwhere(a == color))):
            matches = []
            if len(frame) >= 7:
                for direction in range(4):
                    geometry = _envelope(frame, direction)
                    if geometry is None:
                        continue
                    tip, intake, cavity = geometry
                    if any(not (0 <= r < a.shape[0] and 0 <= c < a.shape[1])
                           for r, c in cavity):
                        continue
                    ink = int(a[intake])
                    if ink in (bg, color):
                        continue
                    if any(a[p] != (ink if p == intake else bg) for p in cavity):
                        continue
                    rs, cs = zip(*frame)
                    box = {(r, c) for r in range(min(rs), max(rs) + 1)
                           for c in range(min(cs), max(cs) + 1)}
                    if any(a[p] != bg for p in box - frame - cavity):
                        continue
                    matches.append(dict(frame=frame, tip=tip, intake=intake,
                                        cavity=cavity, box=box, direction=direction,
                                        ink=ink))
            if len(matches) != 1:
                valid = False
                break
            ports.append(matches[0])
        if valid and ports:
            possibilities.append((color, ports))
    if len(possibilities) != 1:
        raise ValueError("input must have one unambiguous common frame color")
    frame_color, ports = possibilities[0]
    ports.sort(key=lambda p: p["intake"])
    covered, corridors = set(), set()
    for port in ports:
        if not covered.isdisjoint(port["box"]):
            raise ValueError("frame boxes overlap")
        covered.update(port["box"])
        ray = set(_ray(a.shape, port["tip"], port["direction"]))
        if not corridors.isdisjoint(ray) or any(a[p] != bg for p in ray):
            raise ValueError("export corridors must be clear and disjoint")
        corridors.update(ray)
    source_colors = {int(a[p]) for p in map(tuple, np.argwhere(a != bg))
                     if p not in covered}
    if len(source_colors) != 1:
        raise ValueError("one common exterior source color is required")
    source_color = source_colors.pop()
    if source_color in {bg, frame_color} | {p["ink"] for p in ports}:
        raise ValueError("source color must differ from all other local roles")
    source = a == source_color
    return bg, frame_color, source_color, ports, source


def _plus_union(mask):
    centers = np.zeros(mask.shape, dtype=bool)
    if min(mask.shape) >= 3:
        centers[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
            & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:])
    union = centers.copy()
    union[:-1] |= centers[1:]
    union[1:] |= centers[:-1]
    union[:, :-1] |= centers[:, 1:]
    union[:, 1:] |= centers[:, :-1]
    return centers, union


def _owners(mask, ports, metric="manhattan"):
    points = np.argwhere(mask)
    if not len(points):
        return points, np.empty(0, dtype=int)
    sites = np.array([p["intake"] for p in ports], dtype=int)
    delta = np.abs(points[:, None, :] - sites[None, :, :])
    distances = ((delta * delta).sum(2) if metric == "euclidean"
                 else delta.sum(2))
    best = distances.min(1)
    owners = distances.argmin(1)
    owners[(distances == best[:, None]).sum(1) > 1] = -1
    return points, owners


def _emit(a, bg, ports, mask, metric="manhattan"):
    out = a.copy()
    points, owners = _owners(mask, ports, metric)
    for point in points[owners >= 0]:
        out[tuple(point)] = bg
    quantities = np.bincount(owners[owners >= 0], minlength=len(ports))
    for port, quantity in zip(ports, quantities):
        for cell in _ray(a.shape, port["tip"], port["direction"], int(quantity)):
            out[cell] = port["ink"]
    return out, quantities


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input alone; ties retain source ink, and zero means no ray."""
    a = _grid(input_grid)
    bg, _, _, ports, source = _read(a)
    _, eligible = _plus_union(source)
    return _emit(a, bg, ports, eligible)[0]


def _scan_source(mask):
    """Set-based audit/construction detector, distinct from mask dilation."""
    source = set(map(tuple, np.argwhere(mask)))
    centers, cells = set(), set()
    for r, c in sorted(source):
        cross = {(r, c)} | {(r + dr, c + dc) for dr, dc in DIRS}
        if cross <= source:
            centers.add((r, c))
            cells.update(cross)
    return centers, cells


def _authored_output(inp, bg, ports, source):
    """Paint labels from sampled cells and visible sites using scalar arithmetic."""
    _, cells = _scan_source(source)
    out, quantities = inp.copy(), [0] * len(ports)
    for cell in sorted(cells):
        distances = [abs(cell[0] - p["intake"][0]) + abs(cell[1] - p["intake"][1])
                     for p in ports]
        best = min(distances)
        if distances.count(best) == 1:
            quantities[distances.index(best)] += 1
            out[cell] = bg
    for port, q in zip(ports, quantities):
        dr, dc = DIRS[port["direction"]]
        for k in range(1, q + 1):
            r, c = port["tip"][0] + k * dr, port["tip"][1] + k * dc
            if 0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]:
                out[r, c] = port["ink"]
    return out


def _whole_motif_output(inp, bg, ports, centers):
    """A plausible rejected rule: each plus follows its center as one object."""
    destinations = [set() for _ in ports]
    for r, c in map(tuple, np.argwhere(centers)):
        distances = [abs(r-p["intake"][0])+abs(c-p["intake"][1]) for p in ports]
        if distances.count(min(distances)) == 1:
            index = distances.index(min(distances))
            destinations[index].update({(r,c)} | {(r+dr,c+dc) for dr,dc in DIRS})
    out = inp.copy()
    for port, cells in zip(ports, destinations):
        for p in cells:
            out[p] = bg
        for p in _ray(inp.shape, port["tip"], port["direction"], len(cells)):
            out[p] = port["ink"]
    return out


def _profile(rng, direction):
    a, b = map(int, rng.integers(2, 4, size=2))
    if rng.random() < .45:
        b = a
    rows = [(0, 0)] * (1 + int(rng.random() < .2))
    left = right = 0
    # The first slope step opens both sides so there is one visible axis.
    left, right = 1, 1
    rows.append((-left, right))
    while left < a or right < b:
        dl = left < a and (right == b or rng.random() < .8)
        dr = right < b and (left == a or rng.random() < .8)
        if not dl and not dr:
            dl = left < a
            dr = not dl
        left, right = left + int(dl), right + int(dr)
        rows.append((-left, right))
    rows.extend([(-a, b)] * int(rng.integers(1, 4)))
    frame, cavity = set(), set()
    for y, (lo, hi) in enumerate(rows):
        frame.update({(y, lo), (y, hi)})
        cavity.update((y, x) for x in range(lo + 1, hi))
    intake = (len(rows) - 1, 0)
    def relative(point):
        p = _world(point, direction)
        s = _world(intake, direction)
        return p[0] - s[0], p[1] - s[1]
    return dict(frame={relative(p) for p in frame}, cavity={relative(p) for p in cavity},
                tip=relative((0, 0)), direction=direction)


def _layout(rng, count, large=False):
    """Place outward-facing intakes around one shared, invisible source region."""
    for _ in range(160):
        h, w = map(int, rng.integers(22 if large else 19, 31, size=2))
        sh, sw = map(int, rng.integers(5, 10 if not large else 12, size=2))
        r0 = (h - sh) // 2 + int(rng.integers(-2, 3))
        c0 = (w - sw) // 2 + int(rng.integers(-2, 3))
        directions = list(map(int, rng.choice(4, count, replace=False)))
        ports, occupied = [], set()
        good = True
        for direction in directions:
            profile = _profile(rng, direction)
            gap = int(rng.integers(0, 3))
            if direction in (0, 2):
                site = (r0 - 1 - gap if direction == 0 else r0 + sh + gap,
                        c0 + int(rng.integers(1, sw - 1)))
            else:
                site = (r0 + int(rng.integers(1, sh - 1)),
                        c0 + sw + gap if direction == 1 else c0 - 1 - gap)
            def translate(p):
                return p[0] + site[0], p[1] + site[1]
            frame = {translate(p) for p in profile["frame"]}
            cavity = {translate(p) for p in profile["cavity"]}
            rs, cs = zip(*frame)
            box = {(r, c) for r in range(min(rs), max(rs) + 1)
                   for c in range(min(cs), max(cs) + 1)}
            guard = {(r + dr, c + dc) for r, c in box
                     for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            if (any(not (0 <= r < h and 0 <= c < w) for r, c in box)
                    or not occupied.isdisjoint(guard)):
                good = False
                break
            occupied.update(box)
            ports.append(dict(frame=frame, cavity=cavity, box=box, intake=site,
                              tip=translate(profile["tip"]), direction=direction))
        if not good:
            continue
        region = {(r, c) for r in range(r0, r0 + sh) for c in range(c0, c0 + sw)}
        corridors = set()
        for port in ports:
            ray = set(_ray((h, w), port["tip"], port["direction"]))
            if not ray.isdisjoint(occupied | region | corridors):
                good = False
                break
            corridors.update(ray)
        if good:
            return (h, w), (r0, c0, sh, sw), ports
    raise RuntimeError("bounded layout search exhausted")


def _source_mask(rng, shape, region, rich=False):
    r0, c0, h, w = region
    local = np.zeros((h, w), dtype=bool)
    kind = int(rng.integers(5))
    def add(points):
        for r, c in points:
            if 0 <= r < h and 0 <= c < w:
                local[r, c] = True
    def cross(r, c):
        return [(r, c)] + [(r + dr, c + dc) for dr, dc in DIRS]
    if kind <= 1:
        # Two adjacent centers in a solid 3x4, with optional extra width.
        bh, bw = (3, int(rng.integers(4, min(6, w) + 1)))
        if h >= bw and rng.random() < .5:
            bh, bw = bw, bh
        y, x = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
        local[y:y + bh, x:x + bw] = True
    else:
        n = int(rng.integers(1, 4 if rich else 3))
        r, c = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
        add(cross(r, c))
        for _ in range(n - 1):
            if kind == 2:
                dr, dc = DIRS[int(rng.integers(4))]
                r, c = min(h-2, max(1, r+2*dr)), min(w-2, max(1, c+2*dc))
            else:
                r, c = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
            add(cross(r, c))
    # Extensions, bent strokes, T near-misses and scattered marks can coexist.
    for _ in range(int(rng.integers(1, 5 if rich else 4))):
        r, c = int(rng.integers(h)), int(rng.integers(w))
        mode = int(rng.integers(4))
        if mode == 0:
            add([(r, c)])
        elif mode == 1:
            add([(r, c), (r, c+1), (r+1, c+1)])
        elif mode == 2:
            add([(r, c-1), (r, c), (r, c+1), (r+1, c)])
        else:
            dr, dc = DIRS[int(rng.integers(4))]
            add([(r+k*dr, c+k*dc) for k in range(int(rng.integers(2, 5)))])
    mask = np.zeros(shape, dtype=bool)
    mask[r0:r0+h, c0:c0+w] = local
    return mask


def _palette(rng, ports):
    colors = list(map(int, rng.choice(10, 3 + len(ports), replace=False)))
    bg, frame_color, source_color = colors[:3]
    inks = colors[3:]
    if rng.random() < .35:
        inks = [inks[0]] * len(ports)
    return bg, frame_color, source_color, inks


def _paint(shape, ports, source, palette):
    bg, frame_color, source_color, inks = palette
    inp = np.full(shape, bg, dtype=np.int8)
    painted = []
    for port, ink in zip(ports, inks):
        p = dict(port, ink=ink)
        for point in port["frame"]:
            inp[point] = frame_color
        inp[port["intake"]] = ink
        painted.append(p)
    inp[source] = source_color
    out = _authored_output(inp, bg, painted, source)
    assert np.array_equal(solve(inp), out), "construction/input-reader disagreement"
    return {"input": inp, "output": out}


def _facts(pair):
    inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
    bg, frame_color, source_color, ports, source = _read(inp)
    centers, eligible = _plus_union(source)
    points, owners = _owners(eligible, ports)
    _, q = _emit(inp, bg, ports, eligible)
    assigned = {tuple(p): int(i) for p, i in zip(points, owners)}
    full = {int(n) for p, n in zip(ports, q)
            if 0 < n < len(_ray(inp.shape, p["tip"], p["direction"]))}
    split = any(len({assigned.get((int(r)+dr, int(c)+dc), -1)
                     for dr, dc in ((0, 0),) + DIRS} - {-1}) >= 2
                for r, c in np.argwhere(centers))
    sequential = source.copy()
    destructive = np.zeros(source.shape, dtype=bool)
    for r in range(1, source.shape[0]-1):
        for c in range(1, source.shape[1]-1):
            cross = [(r,c)] + [(r+dr,c+dc) for dr,dc in DIRS]
            if all(sequential[p] for p in cross):
                for point in cross:
                    sequential[point] = False
                    destructive[point] = True
    attached = any(source[r+dr,c+dc] and not eligible[r+dr,c+dc]
                   for r,c in map(tuple, np.argwhere(eligible)) for dr,dc in DIRS
                   if 0 <= r+dr < inp.shape[0] and 0 <= c+dc < inp.shape[1])
    return dict(ports=ports, source=source, eligible=eligible, centers=centers,
                bg=bg, frame_color=frame_color, source_color=source_color,
                quantities=tuple(map(int, q)), full=full, tie=bool(np.any(owners < 0)),
                split=split, attached=attached, overlap=int(5*centers.sum()) > int(eligible.sum()),
                simultaneous=not np.array_equal(_emit(inp,bg,ports,destructive)[0],out),
                metric=not np.array_equal(_emit(inp,bg,ports,eligible,"euclidean")[0],out),
                whole_motif=not np.array_equal(_whole_motif_output(inp,bg,ports,centers),out),
                all_source=not np.array_equal(_emit(inp,bg,ports,source)[0],out),
                free=bool(full), active=sum(n > 0 for n in q))


def _evidence(facts):
    flags = ("tie", "split", "attached", "overlap", "simultaneous", "metric", "whole_motif", "all_source")
    return (all(any(f[k] for f in facts) for k in flags)
            and len(set().union(*(f["full"] for f in facts))) >= 2)


def _visible_count_change(first, second, shape):
    return any(0 < a < len(_ray(shape,p["tip"],p["direction"]))
               and 0 < b < len(_ray(shape,p["tip"],p["direction"])) and a != b
               for p,a,b in zip(first["ports"],first["quantities"],second["quantities"]))


def _witnesses(rng, count):
    # Retain at most two candidate pairs, never a bank of reconstructible grids.
    for _ in range(45):
        shape, region, ports = _layout(rng, count)
        palette = _palette(rng, ports)
        first = first_facts = None
        best = -1
        for _ in range(55):
            pair = _paint(shape, ports, _source_mask(rng, shape, region, True), palette)
            facts = _facts(pair)
            if facts["active"] < 2:
                continue
            if first is not None and _key(first["input"]) != _key(pair["input"]):
                if (facts["quantities"] != first_facts["quantities"]
                        and _evidence([first_facts, facts])
                        and _visible_count_change(first_facts, facts, shape)):
                    return [first, pair]
            score = sum(facts[k] for k in ("tie", "split", "attached", "overlap", "simultaneous", "metric", "whole_motif", "all_source")) + min(2, len(facts["full"]))
            if score > best:
                first, first_facts, best = pair, facts, score
    raise RuntimeError("bounded training-witness search exhausted")


def _ordinary(rng, count, seen, large=False, require_free=False):
    for _ in range(60):
        shape, region, ports = _layout(rng, count, large)
        pair = _paint(shape, ports, _source_mask(rng, shape, region, large), _palette(rng, ports))
        f = _facts(pair)
        if ((f["free"] or not require_free) and f["active"] >= min(2, count)
                and _key(pair["input"]) not in seen):
            return pair
    raise RuntimeError("bounded ordinary-scene search exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, "num_train", 2, 10)
    m = _count(num_test, "num_test", 1, 4)
    if n <= m:
        raise ValueError("num_train must be greater than num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base_count = int(rng.choice((2, 3)))
    train = _witnesses(rng, base_count)
    seen = {_key(p["input"]) for p in train}
    for _ in range(n - 2):
        pair = _ordinary(rng, int(rng.integers(1, base_count + 1)), seen)
        train.append(pair)
        seen.add(_key(pair["input"]))
    test = []
    for _ in range(m):
        pair = _ordinary(rng, base_count + 1, seen, large=bool(rng.random() < .5), require_free=True)
        test.append(pair)
        seen.add(_key(pair["input"]))
    puzzle = {"train": [train[int(i)] for i in rng.permutation(n)], "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    n, m = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen, facts = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            f = _facts(pair)
            assert 1 <= len(f["ports"]) <= 4
            assert f["centers"].any() and f["active"] >= 1
            centers, cells = _scan_source(f["source"])
            assert centers == set(map(tuple, np.argwhere(f["centers"])))
            assert cells == set(map(tuple, np.argwhere(f["eligible"])))
            expected = _authored_output(inp, f["bg"], f["ports"], f["source"])
            assert np.array_equal(out, expected), "wrong transport, tie handling, or preservation"
            assert np.array_equal(out, solve(inp.copy())), "input-only solver disagreement"
            assert not np.array_equal(inp, out)
            # The sum of assigned cells and retained ties equals the union size.
            points, owners = _owners(f["eligible"], f["ports"])
            assert sum(f["quantities"]) + int((owners < 0).sum()) == len(cells)
            for p, owner in zip(points, owners):
                assert out[tuple(p)] == (f["source_color"] if owner < 0 else f["bg"])
            assert np.array_equal(inp[f["source"] & ~f["eligible"]], out[f["source"] & ~f["eligible"]])
            for port, q in zip(f["ports"], f["quantities"]):
                assert all(out[p] == f["frame_color"] for p in port["frame"])
                assert out[port["intake"]] == port["ink"]
                ray = _ray(inp.shape, port["tip"], port["direction"])
                assert all(out[p] == (port["ink"] if i < q else f["bg"])
                           for i, p in enumerate(ray))
            facts[split].append(f)
    assert _evidence(facts["train"]), "missing explanatory training witnesses"
    assert all(f["free"] for f in facts["test"]), "test hides all count-dependent stopping"
    assert min(len(f["ports"]) for f in facts["test"]) > max(len(f["ports"]) for f in facts["train"])
    # Matched layout witnesses vary the allocation while leaving all port roles fixed.
    def signature(pair, f):
        a = np.asarray(pair["input"]).copy()
        a[f["source"]] = f["bg"]
        return _key(a)
    groups = {}
    for pair, f in zip(puzzle["train"], facts["train"]):
        groups.setdefault(signature(pair, f), []).append(f)
    assert any(_visible_count_change(a,b,shape)
               for (shape,_),fs in groups.items() for a in fs for b in fs), "count is confounded with port geometry or hidden by clipping"


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, data = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(data)
    return digest.hexdigest()


def self_test():
    """Bounded, streaming tests over the 42 supplied seeds; returns measurements."""
    shapes, profiles, fingerprints, directions, quantities = set(), set(), set(), set(), set()
    backgrounds, source_colors, frame_colors, ink_colors, counts = set(), set(), set(), set(), set()
    totals = {k: 0 for k in ("tie", "split", "overlap", "simultaneous", "metric", "whole_motif", "attached", "clipped", "zero", "idle", "shared_ink")}
    pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _facts(pair)
                pairs += 1
                shapes.add(pair["input"].shape)
                backgrounds.add(f["bg"])
                source_colors.add(f["source_color"])
                frame_colors.add(f["frame_color"])
                counts.add(len(f["ports"]))
                for k in ("tie", "split", "overlap", "simultaneous", "metric", "whole_motif", "attached"):
                    totals[k] += int(f[k])
                totals["shared_ink"] += len({p["ink"] for p in f["ports"]}) < len(f["ports"])
                for port, q in zip(f["ports"], f["quantities"]):
                    directions.add(port["direction"])
                    ink_colors.add(port["ink"])
                    quantities.add(q)
                    room = len(_ray(pair["input"].shape, port["tip"], port["direction"]))
                    totals["clipped"] += int(0 < room < q)
                    totals["zero"] += int(room == 0 and q > 0)
                    totals["idle"] += int(q == 0)
                    local = [_local(p, port["direction"]) for p in port["frame"]]
                    y0, x0 = min(y for y,x in local), min(x for y,x in local)
                    profiles.add(tuple(sorted((y-y0,x-x0) for y,x in local)))
    assert len(fingerprints) == 42 and len(shapes) >= 30 and len(profiles) >= 25
    assert directions == {0,1,2,3} and counts == {1,2,3,4}
    assert all(len(s) >= 8 for s in (backgrounds, source_colors, frame_colors, ink_colors))
    assert all(totals[k] > 0 for k in totals) and len(quantities) >= 8
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0))
    # Exhaustive local recognition, including absent center / missing arm cases.
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3,3)
        c, u = _plus_union(mask)
        cs, us = _scan_source(mask)
        assert cs == set(map(tuple, np.argwhere(c)))
        assert us == set(map(tuple, np.argwhere(u)))
    block = np.ones((3,4), dtype=bool)
    c, u = _plus_union(block)
    assert c.sum() == 2 and u.sum() == 8
    # Hand-counted shared union: 8 eligible cells become 4 left, 1 right, 3 ties.
    inp = np.zeros((11,21),dtype=np.int8)
    inp[4:7,8:12] = 6
    ports = [dict(intake=(5,5),tip=(5,4),direction=3,ink=2),
             dict(intake=(5,15),tip=(5,16),direction=1,ink=3)]
    _, eligible = _plus_union(inp == 6)
    expected = inp.copy()
    expected[5,8] = expected[4,9] = expected[5,9] = expected[6,9] = expected[5,11] = 0
    expected[5,:4] = 2
    expected[5,17] = 3
    emitted, q = _emit(inp,0,ports,eligible)
    assert tuple(q) == (4,1) and np.array_equal(emitted,expected)
    # Translation, all rotations and reflection preserve distance/transport rules.
    p = generate_puzzle(0)
    for pair in p["train"] + p["test"]:
        for k in range(4):
            for reflect in (False, True):
                inp, out = np.rot90(pair["input"], k), np.rot90(pair["output"], k)
                if reflect:
                    inp, out = np.fliplr(inp), np.fliplr(out)
                assert np.array_equal(solve(inp), out)
        permutation = np.random.default_rng(813).permutation(10)
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    for n in range(2,11):
        for m in range(1,min(4,n-1)+1):
            q = generate_puzzle(0,n,m)
            assert (len(q["train"]),len(q["test"])) == (n,m)
    for n,m in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(4.0,1),(None,1),(4,None)):
        try:
            generate_puzzle(0,n,m)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    damaged = {s: [{k:v.copy() for k,v in pair.items()} for pair in p[s]] for s in p}
    damaged["test"][0]["output"][0,0] = (int(damaged["test"][0]["output"][0,0])+1)%10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a changed unrelated cell")
    return {"seeds":len(HOST_SEEDS),"pairs":pairs,"distinct_puzzles":len(fingerprints),
            "shapes":len(shapes),"frame_profiles":len(profiles),"port_counts":sorted(counts),
            "quantities":sorted(quantities),"features":totals,
            "scope":"authored checks, not independent semantic certification"}
