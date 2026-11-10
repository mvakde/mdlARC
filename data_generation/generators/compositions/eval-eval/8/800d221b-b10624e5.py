"""Port-elected mirror stamping. All inference uses visible input cells only."""
from __future__ import annotations

import heapq
from collections import Counter

import numpy as np

_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("Require 2..10 train, 1..4 test, and train > test")
    return a, b


def _array(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), min(cs), max(rs) + 1, max(cs) + 1


def _rectangle(r, c, h, w):
    return {(y, x) for y in range(r, r + h) for x in range(c, c + w)}


def _neighbors(cells):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in _DIRS}


def _square(cells):
    r, c, bottom, right = _box(cells)
    assert bottom - r == right - c and len(cells) == (bottom - r) ** 2
    return r, c, bottom - r


def _port(side, pos, anchor):
    r, c, n = anchor
    return ((r - 1, c + pos), (r + pos, c + n),
            (r + n, c + pos), (r + pos, c - 1))[side]


def _interpret(g, bg, same, flip, wire, parts):
    anchors, panels = [], []
    for cells in parts:
        colors = {int(g[p]) for p in cells}
        if colors == {same}:
            anchors.append((_square(cells), cells))
        else:
            assert colors == {same, flip}
            r, c, bottom, right = _box(cells)
            assert len(cells) == (bottom - r) * (right - c) and len(cells) >= 3
            counts = Counter(int(g[p]) for p in cells)
            assert counts[same] != counts[flip]
            panels.append((cells, counts, max(counts, key=counts.get)))
    assert len(anchors) == 2 and 2 <= len(panels) <= 8
    anchor_ids = {p: i for i, (_, cells) in enumerate(anchors) for p in cells}
    panel_ids = {p: i for i, (cells, _, _) in enumerate(panels) for p in cells}
    routes = _components(g == wire)
    assert routes
    records, used_panels, used_anchors = [], set(), set()
    for cells in routes:
        around = _neighbors(cells)
        owners = {panel_ids[p] for p in around if p in panel_ids}
        targets = {anchor_ids[p] for p in around if p in anchor_ids}
        assert len(owners) == len(targets) == 1
        owner, target = owners.pop(), targets.pop()
        used_panels.add(owner)
        used_anchors.add(target)
        records.append((cells, owner, target))
    assert used_panels == set(range(len(panels))) and len(used_anchors) == 1
    destination = used_anchors.pop()
    reference = 1 - destination
    ar, ac, source_side = anchors[reference][0]
    # The source is the sole non-circuit object. It is connected to its anchor.
    omitted = set(panel_ids) | set().union(*routes) | anchors[destination][1]
    source = set(map(tuple, np.argwhere(g != bg).tolist())) - omitted
    assert anchors[reference][1] < source
    assert len(_components(np.array([[(r, c) in source for c in range(g.shape[1])]
                                    for r in range(g.shape[0])], dtype=bool))) == 1
    assert all(int(g[p]) not in (wire, flip) for p in source)
    assert not (_neighbors(source) & omitted)
    votes = [[0, 0], [0, 0]]  # horizontal, vertical; retain, reflect
    panel_weights = [[0, 0] for _ in panels]
    dest = anchors[destination][0]
    ports = {(_port(side, pos, dest)): int(side % 2 == 0)
             for side in range(4) for pos in range(dest[2])}
    for cells, owner, target in records:
        assert target == destination
        touched = {p for p in cells if _neighbors({p}) & anchors[destination][1]}
        assert touched and touched <= ports.keys()
        for p in touched:
            axis = ports[p]
            bit = int(panels[owner][2] == flip)
            votes[axis][bit] += 1
            panel_weights[owner][axis] += 1
    assert all(min(v) > 0 and v[0] != v[1] for v in votes)
    flags = tuple(int(v[1] > v[0]) for v in votes)
    motif = {(r - ar, c - ac): int(g[r, c]) for r, c in source}
    return {"background": bg, "same": same, "flip": flip, "wire": wire,
            "source": source, "source_anchor": (ar, ac, source_side),
            "destination_anchor": dest, "motif": motif, "panels": panels,
            "routes": records, "votes": votes, "flags": flags,
            "weights": panel_weights}


def _decode(input_grid):
    g = _array(input_grid)
    bg = int(g[0, 0])
    assert np.all(g[0] == bg) and np.all(g[-1] == bg)
    assert np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)
    values = [int(x) for x in np.unique(g) if x != bg]
    interpretations = []
    for same in values:
        for flip in values:
            if same == flip:
                continue
            parts = _components((g == same) | (g == flip))
            # Cheap geometric filter before trying the remaining wire roles.
            pure = [p for p in parts if all(g[q] == same for q in p)]
            if len(pure) != 2 or len(parts) < 4:
                continue
            for wire in values:
                if wire in (same, flip):
                    continue
                try:
                    interpretations.append(_interpret(g, bg, same, flip, wire, parts))
                except AssertionError:
                    continue
    assert len(interpretations) == 1, "Input must have one geometric role interpretation"
    return interpretations[0]


def _scaled(data, horizontal, vertical, destination_side=None):
    """Exact pixel resampling, rejecting any boundary that would need rounding."""
    source_side = data["source_anchor"][2]
    n = data["destination_anchor"][2] if destination_side is None else destination_side
    motif, bg = data["motif"], data["background"]
    r0, c0, r1, c1 = _box(motif)
    r0 = r0 // source_side * source_side
    c0 = c0 // source_side * source_side
    r1 = -(-r1 // source_side) * source_side
    c1 = -(-c1 // source_side) * source_side
    raw = np.full((r1 - r0, c1 - c0), bg, dtype=np.int8)
    for (r, c), color in motif.items():
        raw[r - r0, c - c0] = color
    fine = np.repeat(np.repeat(raw, n, axis=0), n, axis=1)
    h, w = fine.shape[0] // source_side, fine.shape[1] // source_side
    blocks = fine.reshape(h, source_side, w, source_side)
    low, high = blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))
    assert np.array_equal(low, high), "Ambiguous nonintegral color boundary"
    offset = [-r0 * n // source_side, -c0 * n // source_side]
    if vertical:
        low = low[::-1]
        offset[0] = h - offset[0] - n
    if horizontal:
        low = low[:, ::-1]
        offset[1] = w - offset[1] - n
    return low, offset


def _prediction(g, data, flags=None):
    flags = data["flags"] if flags is None else flags
    raster, offset = _scaled(data, *flags)
    out = np.full(g.shape, data["background"], dtype=np.int8)
    for p in data["source"]:
        out[p] = g[p]
    r, c, _ = data["destination_anchor"]
    top, left = r - offset[0], c - offset[1]
    painted, lost = set(), 0
    for y, x in np.argwhere(raster != data["background"]):
        p = top + int(y), left + int(x)
        if 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]:
            assert p not in data["source"], "Stamp would overwrite the source"
            out[p] = raster[y, x]
            painted.add(p)
        else:
            lost += 1
    assert not (_neighbors(painted) & data["source"]), "Source and stamp must stay separate"
    return out, {"lost": lost, "painted": painted}


def solve(input_grid: np.ndarray) -> np.ndarray:
    g = _array(input_grid)
    return _prediction(g, _decode(g))[0]


def _grow_motif(rng, a, long_arm):
    anchor = _rectangle(0, 0, a, a)
    motif = {p: 0 for p in anchor}
    # Unequal arms start the shape; subsequent growth changes its topology.
    l1 = int(rng.integers(3, 6)) if long_arm else int(rng.integers(1, 4))
    l2 = int(rng.integers(1, 4))
    for p in _rectangle(-l1, 0, l1, int(rng.integers(1, a + 1))):
        motif[p] = 1
    for p in _rectangle(0, a, int(rng.integers(1, a + 1)), l2):
        motif[p] = 2
    for _ in range(int(rng.integers(0, 6))):
        cells = sorted(motif)
        r, c = cells[int(rng.integers(len(cells)))]
        dr, dc = _DIRS[int(rng.integers(4))]
        region = {(r + dr * k, c + dc * k) for k in range(1, int(rng.integers(2, 5)))}
        combined = set(motif) | region
        r0, c0, r1, c1 = _box(combined)
        if r1 - r0 <= 8 and c1 - c0 <= 8:
            color = int(rng.integers(1, 4))
            for p in region - anchor:
                motif[p] = color
    for _ in range(int(rng.integers(4))):
        motif = {(c, a - 1 - r): v for (r, c), v in motif.items()}
    if rng.random() < .5:
        motif = {(r, a - 1 - c): v for (r, c), v in motif.items()}
    return motif


def _axis_groups(rng, n, axis, flag, witness, rich):
    sides = [1, 3] if axis == 0 else [0, 2]
    if rng.random() < .5:
        sides.reverse()
    if witness:
        weight = int(rng.integers(3, n + 1))
        start = int(rng.integers(n - weight + 1))
        groups = [(sides[0], list(range(start, start + weight))),
                  (sides[1], [0]), (sides[1], [n - 1])]
        labels = [flag, 1 - flag, 1 - flag]
    else:
        count = int(rng.choice([2, 3, 4] if rich and n >= 4 else [2, 3]))
        groups = []
        for i, side in enumerate(sides):
            split = count == 4 or (count == 3 and i == 1)
            if split:
                width = int(rng.integers(1, min(2, n - 2) + 1))
                groups.extend([(side, list(range(width))), (side, [n - 1])])
            else:
                width = int(rng.integers(1, n + 1))
                start = int(rng.integers(n - width + 1))
                groups.append((side, list(range(start, start + width))))
        weights = [len(p) for _, p in groups]
        options = [bits for bits in range(1, (1 << len(groups)) - 1)
                   if sum(w for i, w in enumerate(weights) if bits >> i & 1) > sum(weights) / 2]
        if not options:
            return _axis_groups(rng, n, axis, flag, True, rich)
        bits = options[int(rng.integers(len(options)))]
        labels = [flag if bits >> i & 1 else 1 - flag for i in range(len(groups))]
    return [{"side": s, "positions": ps, "label": label, "axis": axis,
             "witness_winner": witness and i == 0}
            for i, ((s, ps), label) in enumerate(zip(groups, labels))]


def _path(start, targets, allowed, costs):
    h, w = allowed.shape
    heap, distance, parent = [(0., *start)], {start: 0.}, {}
    while heap:
        d, r, c = heapq.heappop(heap)
        p = r, c
        if d != distance[p]:
            continue
        if p in targets:
            result = {p}
            while p != start:
                p = parent[p]
                result.add(p)
            return result
        for dr, dc in _DIRS:
            q = r + dr, c + dc
            if 0 <= q[0] < h and 0 <= q[1] < w and allowed[q]:
                nd = d + float(costs[q])
                if nd < distance.get(q, float("inf")):
                    distance[q], parent[q] = nd, p
                    heapq.heappush(heap, (nd, *q))
    return None


def _place_panels(rng, shape, anchor, groups, occupied):
    h, w = shape
    ar, ac, n = anchor
    reserved = occupied | _neighbors(occupied) | _rectangle(ar - 2, ac - 2, n + 4, n + 4)
    panels = []
    for group in groups:
        side = group["side"]
        peers = [g for g in groups if g["side"] == side]
        rank = sorted(peers, key=lambda g: min(g["positions"])).index(group)
        for _ in range(50):
            # Area is independent of port weight, with an extra raw-count
            # counterexample bias on the mandatory voting witness.
            if group["witness_winner"]:
                ph, pw = 1, int(rng.integers(3, 6))
            else:
                ph, pw = int(rng.integers(2, 5)), int(rng.integers(2, 5))
            tangent = ((-2 if rank == 0 else n + 1) if len(peers) == 2
                       else float(np.mean(group["positions"])))
            tangent += int(rng.integers(-2, 3))
            if side in (0, 2):
                c = int(round(ac + tangent - (pw - 1) / 2))
                low, high = (1, ar - ph - 2) if side == 0 else (ar + n + 2, h - ph - 1)
                if low > high:
                    continue
                r = int(rng.integers(low, high + 1))
            else:
                r = int(round(ar + tangent - (ph - 1) / 2))
                low, high = (1, ac - pw - 2) if side == 3 else (ac + n + 2, w - pw - 1)
                if low > high:
                    continue
                c = int(rng.integers(low, high + 1))
            cells = _rectangle(r, c, ph, pw)
            if min(r, c) < 1 or r + ph >= h or c + pw >= w or cells & reserved:
                continue
            panels.append(cells)
            reserved |= cells | _neighbors(cells)
            break
        else:
            return None
    return panels


def _routes(rng, shape, anchor, groups, panels, source):
    h, w = shape
    ar, ac, n = anchor
    core = _rectangle(ar, ac, n, n)
    collar = _rectangle(ar - 1, ac - 1, n + 2, n + 2) - core
    all_panels = set().union(*panels)
    ports = [{_port(g["side"], p, anchor) for p in g["positions"]} for g in groups]
    networks = [set(p) for p in ports]
    order = sorted(range(len(groups)), key=lambda i: (-len(ports[i]), i))
    for i in order:
        other = set().union(*(net for j, net in enumerate(networks) if j != i))
        blocked = core | collar | all_panels | source | _neighbors(source)
        blocked |= other | _neighbors(other) | _neighbors(all_panels - panels[i])
        blocked -= ports[i]
        allowed = np.ones(shape, dtype=bool)
        allowed[0] = allowed[-1] = False
        allowed[:, 0] = allowed[:, -1] = False
        for r, c in blocked:
            if 0 <= r < h and 0 <= c < w:
                allowed[r, c] = False
        target = {p for p in _neighbors(panels[i]) - panels[i]
                  if 0 <= p[0] < h and 0 <= p[1] < w and allowed[p]}
        if not target:
            return None
        costs = 1 + rng.random(shape) * float(rng.uniform(.1, 5.))
        # Spatial cost bands produce genuine bends and varying branch lengths.
        if rng.random() < .5:
            costs += np.repeat(np.repeat(rng.random(((h + 3) // 4, (w + 3) // 4)) * 5,
                                         4, axis=0), 4, axis=1)[:h, :w]
        net = set()
        for port in sorted(ports[i]):
            path = _path(port, target | net, allowed, costs)
            if path is None:
                return None
            net |= path
        if net & collar != ports[i]:
            return None
        networks[i] = net
    return networks


def _draw_motif(shape, motif, a, unit, anchor, palette, flags=(0, 0)):
    """Construction renderer: forward base-lattice rectangles, no input parser."""
    cells = {}
    ar, ac, _ = anchor
    for (r, c), role in motif.items():
        rr = a - 1 - r if flags[1] else r
        cc = a - 1 - c if flags[0] else c
        for p in _rectangle(ar + rr * unit, ac + cc * unit, unit, unit):
            cells[p] = palette[role]
    visible = {p: v for p, v in cells.items() if 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]}
    return visible, len(cells) - len(visible)


def _make_pair(rng, roles, flags, witness_axis, clipped, rich, source_corner=None):
    bg, wire, same, flip, *details = roles
    for _ in range(180):
        h, w = int(rng.integers(24, 31)), int(rng.integers(24, 31))
        a = int(rng.choice([1, 1, 2]))
        du = int(rng.choice([3, 4, 5] if a == 1 else [2, 3]))
        if not rich and a == 1:
            du = int(rng.choice([3, 4]))
        su = int(rng.integers(1, 6 if a == 1 else 4))
        n = a * du
        if witness_axis is not None and a * su == n:
            continue
        ar = int(rng.integers(7, h - n - 5))
        ac = int(rng.integers(7, w - n - 5))
        destination = ar, ac, n
        motif = _grow_motif(rng, a, clipped is True)
        # Every mandatory example visibly distinguishes all reflections.
        if any({(a - 1 - r if v else r, a - 1 - c if q else c): k
                for (r, c), k in motif.items()} == motif
               for q, v in ((1, 0), (0, 1), (1, 1))):
            continue
        r0, c0, r1, c1 = _box(motif)
        mh, mw = (r1 - r0) * su, (c1 - c0) * su
        if max(mh, mw) > 12:
            continue
        corner = int(rng.integers(4)) if source_corner is None else source_corner
        top = 1 if corner // 2 == 0 else h - mh - 1
        left = 1 if corner % 2 == 0 else w - mw - 1
        reference = top - r0 * su, left - c0 * su, a * su
        source_center = (reference[0] + reference[2] / 2, reference[1] + reference[2] / 2)
        dest_center = ar + n / 2, ac + n / 2
        if any(source_center[i] == dest_center[i] for i in (0, 1)):
            continue
        if (int(source_center[0] > dest_center[0]), int(source_center[1] > dest_center[1])) != (corner // 2, corner % 2):
            continue
        palette = [same, *details]
        source, _ = _draw_motif((h, w), motif, a, su, reference, palette)
        stamp, lost = _draw_motif((h, w), motif, a, du, destination, palette, flags)
        if clipped is not None and (lost > 0) != clipped:
            continue
        if len(stamp) <= n * n or (set(stamp) | _neighbors(stamp)) & source.keys():
            continue
        if (set(source) | _neighbors(source)) & _rectangle(ar - 2, ac - 2, n + 4, n + 4):
            continue
        groups = []
        for axis in (0, 1):
            groups += _axis_groups(rng, n, axis, flags[axis], axis == witness_axis, rich)
        panels = _place_panels(rng, (h, w), destination, groups, set(source))
        if panels is None:
            continue
        networks = _routes(rng, (h, w), destination, groups, panels, set(source))
        if networks is None:
            continue
        inp = np.full((h, w), bg, dtype=np.int8)
        for p, color in source.items():
            inp[p] = color
        inp[ar:ar + n, ac:ac + n] = same
        for group, cells, route in zip(groups, panels, networks):
            major = (same, flip)[group["label"]]
            minor = flip if major == same else same
            points = sorted(cells)
            # Local margin is unrelated to port multiplicity.
            nminor = int(rng.integers(1, (len(points) - 1) // 2 + 1))
            if witness_axis == group["axis"] and not group["witness_winner"]:
                nminor = 1
            order = rng.permutation(len(points))
            minority = {points[int(j)] for j in order[:nminor]}
            for p in points:
                inp[p] = minor if p in minority else major
            for p in route:
                inp[p] = wire
        out = np.full((h, w), bg, dtype=np.int8)
        for p, color in source.items():
            out[p] = color
        for p, color in stamp.items():
            out[p] = color
        try:
            data = _decode(inp)
            predicted, info = _prediction(inp, data)
            assert data["flags"] == flags and np.array_equal(out, predicted)
            if witness_axis is not None:
                m = _measure(inp, data)
                assert m["unweighted_wrong"][witness_axis]
                assert m["raw_wrong"][witness_axis]
                assert m["orientation_observable"][witness_axis]
            return {"input": inp, "output": out}
        except AssertionError:
            continue
    raise RuntimeError("Bounded construction exhausted")


def _key(g):
    return g.shape, g.astype(np.uint8).tobytes()


def _measure(g, data=None):
    d = _decode(g) if data is None else data
    expected, info = _prediction(g, d)
    unweighted, raw = [], []
    for axis in (0, 1):
        counts, pixels = [0, 0], [0, 0]
        for (cells, local, label), weights in zip(d["panels"], d["weights"]):
            if weights[axis]:
                counts[int(label == d["flip"])] += 1
                pixels[0] += local[d["same"]]
                pixels[1] += local[d["flip"]]
        correct = d["flags"][axis]
        unweighted.append(counts[1 - correct] > counts[correct])
        raw.append(pixels[1 - correct] > pixels[correct])
    raster, _ = _scaled(d, *d["flags"])
    observable = []
    for axis in (0, 1):
        wrong_flags = list(d["flags"])
        wrong_flags[axis] ^= 1
        try:
            alternative = _prediction(g, d, wrong_flags)[0]
            observable.append(not np.array_equal(expected, alternative))
        except AssertionError:
            # A different orientation that collides with the preserved source
            # is also observably different from the valid, separated stamp.
            observable.append(True)
    ar, ac, a = d["source_anchor"]
    dr, dc, n = d["destination_anchor"]
    quadrant = (int(ar + a / 2 > dr + n / 2), int(ac + a / 2 > dc + n / 2))
    return {"shape": list(g.shape), "source_side": d["source_anchor"][2],
            "destination_side": d["destination_anchor"][2], "flags": d["flags"],
            "votes": d["votes"], "weights": d["weights"],
            "panel_count": len(d["panels"]), "panel_areas": [len(p[0]) for p in d["panels"]],
            "route_cells": sum(len(r[0]) for r in d["routes"]),
            "source_cells": len(d["source"]), "stamp_shape": list(raster.shape),
            "lost": info["lost"], "unweighted_wrong": unweighted, "raw_wrong": raw,
            "orientation_observable": observable,
            "source_quadrant": quadrant,
            "roles": [d[k] for k in ("background", "wire", "same", "flip")]}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = [int(x) for x in rng.permutation(10)[:7]]
    varying_palette = bool(rng.integers(2))
    first_flag = int(rng.integers(2))
    witness_corner = int(rng.integers(4))
    train_flags = [(first_flag, 1 - first_flag), (1 - first_flag, first_flag)]
    train_flags += [tuple(map(int, rng.integers(2, size=2))) for _ in range(num_train - 2)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            flags = train_flags[i] if split == "train" else (lambda v: (v, v))(int(rng.integers(2)))
            witness_axis = i if split == "train" and i < 2 else None
            clipped = (False if i == 0 else True) if split == "train" and i < 2 else None
            roles = [int(x) for x in rng.permutation(10)[:7]] if varying_palette else palette
            for _ in range(5):
                pair = _make_pair(rng, roles, flags, witness_axis, clipped, split == "test",
                                  witness_corner if witness_axis is not None else None)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("Could not construct a distinct input")
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            data = _decode(inp)
            predicted, info = _prediction(inp, data)
            assert np.array_equal(predicted, out) and not np.array_equal(inp, out)
            assert len(info["painted"]) > data["destination_anchor"][2] ** 2
            assert all(out[p] == inp[p] for p in data["source"])
            # Anchor is preserved; circuit colors disappear outside the new motif.
            r, c, n = data["destination_anchor"]
            assert np.all(out[r:r + n, c:c + n] == data["same"])
            assert not np.any(out == data["wire"]) and not np.any(out == data["flip"])
            if split == "train":
                train.append(_measure(inp, data))
    for axis in (0, 1):
        assert any(t["unweighted_wrong"][axis] and t["raw_wrong"][axis]
                   and t["orientation_observable"][axis] for t in train)
        assert {t["flags"][axis] for t in train} == {0, 1}
    assert any(t["lost"] == 0 for t in train) and any(t["lost"] > 0 for t in train)
    assert any(t["source_side"] != t["destination_side"] for t in train)
    assert any(a["flags"] == (0, 1) and b["flags"] == (1, 0)
               and a["source_quadrant"] == b["source_quadrant"] for a in train for b in train)


def self_test():
    """42 seeds, sequential pairs, summaries only; never writes files."""
    shapes, palettes, motifs, flags, sides, ratios, panel_counts = set(), set(), set(), set(), set(), set(), set()
    clipped, full, raw_witnesses = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed < 2:
            repeated = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], repeated[s]) for k in p)
        for split in puzzle:
            for pair in puzzle[split]:
                inp = pair["input"]
                d = _decode(inp)
                m = _measure(inp, d)
                if seed == 0:
                    # The rule is independent of absolute orientation and color.
                    assert np.array_equal(solve(inp[:, ::-1]), pair["output"][:, ::-1])
                    assert np.array_equal(solve((9 - inp).T), (9 - pair["output"]).T)
                    normal, _ = _scaled(d, 0, 0)
                    for horizontal, vertical in ((1, 0), (0, 1), (1, 1)):
                        transformed, _ = _scaled(d, horizontal, vertical)
                        assert not np.array_equal(normal, transformed)
                    # Ballot arrangement has no meaning beyond local counts.
                    changed = inp.copy()
                    points = sorted(d["panels"][0][0])
                    values = [int(inp[p]) for p in points]
                    for p, value in zip(points, values[1:] + values[:1]):
                        changed[p] = value
                    assert np.array_equal(solve(changed), pair["output"])
                shapes.add(inp.shape)
                palettes.add(tuple(m["roles"]))
                flags.add(m["flags"])
                sides.add(m["destination_side"])
                divisor = int(np.gcd(m["destination_side"], m["source_side"]))
                ratios.add((m["destination_side"] // divisor, m["source_side"] // divisor))
                panel_counts.add(m["panel_count"])
                roles = {d["same"]: 0}
                signature = tuple((r, c, roles.setdefault(v, len(roles)))
                                  for (r, c), v in sorted(d["motif"].items()))
                motifs.add(signature)
                clipped += m["lost"] > 0
                full += m["lost"] == 0
                raw_witnesses += any(m["raw_wrong"])
    assert len(shapes) >= 25 and len(palettes) >= 30 and len(motifs) >= 100
    assert flags == {(0, 0), (0, 1), (1, 0), (1, 1)} and sides >= {3, 4, 5, 6}
    assert len(ratios) >= 6 and len(panel_counts) >= 3 and clipped >= 42 and full >= 42
    assert raw_witnesses >= 84
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4., 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    puzzle = generate_puzzle(75, 2, 1)
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    # A sub-anchor detail cannot be rounded at a 3/2 scale.
    fine = {(r, c): 1 for r in range(2) for c in range(2)}
    fine[-1, 0] = 2
    data = {"source_anchor": (0, 0, 2), "destination_anchor": (0, 0, 3),
            "motif": fine, "background": 0}
    try:
        _scaled(data, 0, 0)
    except AssertionError:
        pass
    else:
        raise AssertionError("Ambiguous scaling boundary accepted")
    # Hand-constructed rule regression outside the side-local sampling layout:
    # each of two panels sends the same label into BOTH axis elections.
    shared = np.zeros((20, 20), dtype=np.int8)
    shared[2, 2], shared[1, 2], shared[2, 3] = 2, 4, 6
    shared[9:12, 9:12] = 2
    shared[5:7, 4:6] = [[2, 2], [3, 2]]
    shared[14:16, 14:17] = [[3, 3, 3], [2, 3, 3]]
    winning = {(8, 9), (8, 10), (7, 9), (7, 8), (7, 7), (6, 7), (5, 7),
               (5, 6), (9, 8), (10, 8), (9, 7), (10, 7), (8, 7)}
    losing = {(12, 11), (13, 11), (13, 12), (13, 13), (13, 14),
              (11, 12), (11, 13), (12, 13)}
    for p in winning | losing:
        shared[p] = 5
    expected = np.zeros_like(shared)
    expected[2, 2], expected[1, 2], expected[2, 3] = 2, 4, 6
    expected[9:12, 9:12], expected[6:9, 9:12], expected[9:12, 12:15] = 2, 4, 6
    assert np.array_equal(solve(shared), expected)
    assert sorted(map(tuple, _decode(shared)["weights"])) == [(1, 1), (2, 2)]
    return {"seeds": 42, "shapes": len(shapes), "palettes": len(palettes),
            "motifs": len(motifs), "ratios": len(ratios), "panel_counts": sorted(panel_counts),
            "clipped_pairs": clipped, "full_pairs": full}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
