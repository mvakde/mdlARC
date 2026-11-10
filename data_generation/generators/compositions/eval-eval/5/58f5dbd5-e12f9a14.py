"""Crossfire stencil: interacting light selects color-matched negative imprints.

Only standard-library modules and NumPy are used. The parser and tracer receive
only the input grid. No parent generator, examples, seed or sampling state is
used by solve.
"""
from __future__ import annotations

import json
import math
from collections import deque

import numpy as np


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(np.all((0 <= a) & (a <= 9)), "invalid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _ring(h, w):
    return [(r, c) for r in range(h + 2) for c in range(w + 2)
            if r in (0, h + 1) or c in (0, w + 1)]


def _direction(p, h, w):
    r, c = p
    return (1 if r == 0 else -1 if r == h + 1 else 0,
            1 if c == 0 else -1 if c == w + 1 else 0)


def _parse(a):
    """A unique interrupted frame supplies a common, explicitly bounded register."""
    colors, counts = np.unique(a, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "background tie")
    bg = int(colors[np.argmax(counts)])
    candidates = []
    for marker in colors:
        marker = int(marker)
        if marker == bg:
            continue
        cells = np.argwhere(a == marker)
        y, x = map(int, cells.min(axis=0))
        end_y, end_x = map(int, cells.max(axis=0) + 1)
        h, w = end_y - y - 2, end_x - x - 2
        if min(h, w) < 3:
            continue
        perimeter = _ring(h, w)
        if any(int(a[y+r, x+c]) not in (bg, marker) for r, c in perimeter):
            continue
        if np.any(a[y+1:end_y-1, x+1:end_x-1] == marker):
            continue
        gaps = [(r, c) for r, c in perimeter if a[y+r, x+c] == bg]
        if not 1 <= len(gaps) <= 6 or len(cells) < len(perimeter) // 2:
            continue
        candidates.append((marker, y, x, h, w, gaps))
    _require(len(candidates) == 1, "frame must be uniquely identifiable")
    marker, y, x, h, w, ports = candidates[0]
    outside = np.ones(a.shape, dtype=bool)
    outside[y:y+h+2, x:x+w+2] = False
    visited = np.zeros(a.shape, dtype=bool)
    plates = {}
    for rr, cc in np.argwhere(outside & (a != bg)):
        rr, cc = int(rr), int(cc)
        if visited[rr, cc]:
            continue
        color = int(a[rr, cc])
        queue, component = deque([(rr, cc)]), []
        visited[rr, cc] = True
        while queue:
            r, c = queue.popleft()
            component.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r+dr, c+dc
                if (0 <= nr < a.shape[0] and 0 <= nc < a.shape[1]
                        and outside[nr, nc] and not visited[nr, nc] and a[nr, nc] == color):
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        coords = np.array(component)
        top, left = map(int, coords.min(axis=0))
        bottom, right = map(int, coords.max(axis=0) + 1)
        _require((bottom-top, right-left) == (h+2, w+2)
                 and len(component) == (h+2)*(w+2), "outside object is not a solid plate")
        _require(color not in plates and color != marker, "duplicate plate color")
        _require(outside[top:bottom, left:right].all(), "plate touches the arena")
        plates[color] = (top, left)
    _require(1 <= len(plates) <= 6, "plate count")
    boxes = [(y, x)] + list(plates.values())
    for i, (r, c) in enumerate(boxes):
        for s, t in boxes[i+1:]:
            _require(r+h+2 < s or s+h+2 < r or c+w+2 < t or t+w+2 < c,
                     "objects need one background cell of clearance")
    ink = a[y+1:y+h+1, x+1:x+w+1].copy()
    _require(set(plates) <= set(map(int, np.unique(ink))), "plate lacks its stencil ink")
    return {"bg": bg, "marker": marker, "origin": (y, x), "shape": (h, w),
            "ports": sorted(ports), "plates": plates, "ink": ink}


def _walk(start, direction, h, w, ceil_raster=False):
    """Floor rasterization: rational slopes start with the major-axis step."""
    dr, dc = direction
    scale = max(abs(dr), abs(dc))
    points = [start]
    for tick in range(1, 3*(h+w)+1):
        bias = scale-1 if ceil_raster else 0
        p = (start[0] + (1 if dr >= 0 else -1)*((tick*abs(dr)+bias)//scale),
             start[1] + (1 if dc >= 0 else -1)*((tick*abs(dc)+bias)//scale))
        if not (1 <= p[0] <= h and 1 <= p[1] <= w):
            break
        points.append(p)
    return points


def _contact(paths_a, paths_b):
    hits = []
    for ia, aa in enumerate(paths_a):
        positions = {p: i for i, p in enumerate(aa)}
        diagonal = {(p[0]+q[0], p[1]+q[1]): i
                    for i, (p, q) in enumerate(zip(aa, aa[1:]))
                    if p[0] != q[0] and p[1] != q[1]}
        for ib, bb in enumerate(paths_b):
            for j, p in enumerate(bb):
                if p in positions:
                    hits.append((positions[p]-1, j-1, ia, ib, "cell"))
            for j, (p, q) in enumerate(zip(bb, bb[1:])):
                if p[0] == q[0] or p[1] == q[1]:
                    continue
                midpoint = (p[0]+q[0], p[1]+q[1])
                if midpoint in diagonal:
                    i = diagonal[midpoint]
                    if {aa[i], aa[i+1]} != {p, q}:
                        hits.append((i, j, ia, ib, "between"))
    if not hits:
        return None
    rank = min((t[0]+t[1], max(t[:2])) for t in hits)
    first = [t for t in hits if (t[0]+t[1], max(t[:2])) == rank]
    _require(len({t[:2] for t in first}) == 1, "ambiguous equal-rank contacts")
    return min(first, key=lambda t: (t[2], t[3]))


def _kind(u, v):
    _require(max(map(abs, u)) == max(map(abs, v)) == 1,
             "staircase bundles are terminal")
    dot = u[0]*v[0] + u[1]*v[1]
    nu, nv = sum(map(abs, u)), sum(map(abs, v))
    if nu == nv == 1:
        _require(dot == 0, "parallel or head-on beams")
        return "perpendicular"
    if nu == nv == 2:
        _require(dot == 0, "parallel or head-on diagonal beams")
        return "diagonals"
    _require(dot > 0, "obtuse corner/side encounter")
    return "staircase"


def _trace(h, w, ports, *, straight=False, sign_sum=False, weighted_sum=False, ceil_raster=False):
    """Trace spatial mutual-first encounters, preserving strand offsets.

    The optional flags are only counterfactual checks; solve always uses
    the full heading sum and collisions.
    """
    groups = [{"v": _direction(p, h, w), "members": [(i, p)], "depth": 0}
              for i, p in enumerate(sorted(ports))]
    if straight:
        return set(p for g in groups for _, start in g["members"]
                   for p in _walk(start, g["v"], h, w)), []
    painted, last, segments, events = {}, {}, set(), []

    def paint(strand, path):
        for index, p in enumerate(path):
            _require(p not in painted or (index == 0 and painted[p] == strand and last[strand] == p),
                     "overlap or self-intersection")
            painted[p] = strand
            last[strand] = p
        for p, q in zip(path, path[1:]):
            if p[0] != q[0] and p[1] != q[1]:
                mid = (p[0]+q[0], p[1]+q[1])
                slope = 1 if (q[0]-p[0])*(q[1]-p[1]) > 0 else -1
                _require((mid, -slope) not in segments, "unresolved crossing")
                segments.add((mid, slope))

    for _ in range(len(ports)+1):
        forecasts = [[_walk(start, g["v"], h, w, ceil_raster) for _, start in g["members"]] for g in groups]
        contacts = [(i, j, hit) for i in range(len(groups)) for j in range(i+1, len(groups))
                    if (hit := _contact(forecasts[i], forecasts[j])) is not None]
        if not contacts:
            for g, paths in zip(groups, forecasts):
                for (strand, _), path in zip(g["members"], paths):
                    paint(strand, path)
            return set(painted), events
        nearest = {}
        for i, j, hit in contacts:
            nearest[i] = min(nearest.get(i, 999), hit[0])
            nearest[j] = min(nearest.get(j, 999), hit[1])
        ready = [(i, j, t) for i, j, t in contacts
                 if t[0] == nearest[i] and t[1] == nearest[j]]
        _require(bool(ready), "cyclic encounter order")
        i, j, hit = min(ready, key=lambda t: (max(t[2][:2]), sum(t[2][:2]), t[0], t[1]))
        for k, l, other in contacts:
            if (k, l) != (i, j):
                _require(not ((k in (i, j) and other[0] == nearest[k]) or
                              (l in (i, j) and other[1] == nearest[l])), "multiway encounter")
        a, b = groups[i], groups[j]
        kind = _kind(a["v"], b["v"])
        wa, wb = (len(a["members"]), len(b["members"])) if weighted_sum else (1, 1)
        velocity = tuple(wa*a["v"][k]+wb*b["v"][k] for k in (0, 1))
        divisor = math.gcd(*map(abs, velocity))
        _require(divisor > 0 and min(hit[:2]) >= 0, "invalid encounter")
        velocity = tuple(v//divisor for v in velocity)
        if sign_sum:
            velocity = tuple(int(np.sign(v)) for v in velocity)
        members = []
        for index, cut in ((i, hit[0]), (j, hit[1])):
            for (strand, _), path in zip(groups[index]["members"], forecasts[index]):
                _require(cut < len(path)-1, "bundle partly leaves before encounter")
                paint(strand, path[:cut+1])
                members.append((strand, path[cut]))
        depth = 1+max(a["depth"], b["depth"])
        events.append({"kind": kind, "intersection": hit[4], "depth": depth,
                       "incoming": (a["v"], b["v"]), "outgoing": velocity,
                       "cuts": hit[:2], "strands": len(members)})
        groups = [g for k, g in enumerate(groups) if k not in (i, j)]
        groups.append({"v": velocity, "members": members, "depth": depth})
    raise AssertionError("too many beam merges")


def _illumination(info, **flags):
    h, w = info["shape"]
    points, events = _trace(h, w, info["ports"], **flags)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in points:
        if 1 <= r <= h and 1 <= c <= w:
            mask[r-1, c-1] = True
    return mask, points, events


def _render(a, info, illuminated, points, *, all_ink=False, all_beam=False):
    out = a.copy()
    y, x = info["origin"]
    for r, c in points:
        out[y+r, x+c] = info["marker"]
    h, w = info["shape"]
    for color, (top, left) in info["plates"].items():
        selected = (info["ink"] == color) & illuminated
        if all_ink:
            selected = info["ink"] == color
        if all_beam:
            selected = illuminated
        out[top+1:top+h+1, left+1:left+w+1][selected] = info["bg"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    info = _parse(a)
    light, points, _ = _illumination(info)
    return _render(a, info, light, points)


def _orientation_candidates(info, light):
    """Which D4 transformations leave all transferred hole masks unchanged?"""
    candidates = set(range(8))
    for color in info["plates"]:
        holes = (info["ink"] == color) & light
        transforms = [np.rot90(holes, k) for k in range(4)] + [
            np.rot90(np.fliplr(holes), k) for k in range(4)]
        candidates &= {k for k, mask in enumerate(transforms)
                       if mask.shape == holes.shape and np.array_equal(mask, holes)}
    return candidates


_ALTERNATIVES = ({"sign_sum": True}, {"weighted_sum": True}, {"ceil_raster": True})


def _alternative_agrees(pair, flags):
    try:
        a = _grid(pair["input"])
        info = _parse(a)
        light, points, _ = _illumination(info, **flags)
        return np.array_equal(_render(a, info, light, points), pair["output"])
    except AssertionError:
        return False


def _surviving_alternatives(train):
    return [flags for flags in _ALTERNATIVES if all(_alternative_agrees(p, flags) for p in train)]


def _port_scene(rng, h, w, mode, allowed, extra):
    if mode == "isolated":
        options = [p for p in _ring(h, w) if sum(map(abs, _direction(p, h, w))) == 1]
        ports = [_pick(rng, options)]
    elif mode == "perpendicular":
        row, col = int(rng.integers(1, h)), int(rng.integers(1, w))
        if row == col:
            col = col % (w-1) + 1
        ports = [(row, 0), (0, col)]
    elif mode == "diagonals":
        ports = [(0, 0), (0, w+1)]
    else:
        ports = [(0, 0), (0, int(rng.integers(2, min(w, h-2)+1)))]
    points, events = _trace(h, w, ports)
    _require(mode == "isolated" or any(e["kind"] == mode for e in events), "missing intended encounter")
    for _ in range(extra*12):
        if len(ports) >= min(6, (1 if mode == "isolated" else 2)+extra):
            break
        p = _pick(rng, [p for p in _ring(h, w) if p not in ports])
        try:
            pp, ee = _trace(h, w, ports+[p])
            _require({e["kind"] for e in ee} <= allowed, "untaught collision")
            _require(mode != "isolated" or not ee, "control must be straight")
            _require(mode == "isolated" or any(e["kind"] == mode for e in ee), "primary mechanism erased")
            ports.append(p)
            points, events = pp, ee
        except AssertionError:
            continue
    return ports, points, events


def _positions(rng, h, w, count):
    bh, bw = h+2, w+2
    choices = [(r, c, gap) for r in range(1, 5) for c in range(1, 5) for gap in (1, 2, 3)
               if count <= r*c <= count+3 and r*bh+(r-1)*gap <= 28 and c*bw+(c-1)*gap <= 28]
    _require(bool(choices), "objects do not fit")
    if rng.random() < .55:
        for _ in range(12):
            positions = []
            for _ in range(count):
                available = [(r, c) for r in range(1, 30-bh) for c in range(1, 30-bw)
                             if all(r+bh < s or s+bh < r or c+bw < t or t+bw < c
                                    for s, t in positions)]
                if not available:
                    break
                positions.append(_pick(rng, available))
            if len(positions) == count:
                return positions
    rows, cols, gap = _pick(rng, choices)
    slots = [(1+r*(bh+gap), 1+c*(bw+gap)) for r in range(rows) for c in range(cols)]
    rng.shuffle(slots)
    return slots[:count]


def _make_pair(rng, mode, allowed, regime, is_test, palette, allow_chain=True, contacts=None):
    for _ in range(160):
        if regime == "reach":
            low, high = (6, 9) if is_test else (4, 6)
        else:
            low, high = 4, (7 if regime == "shape" else 6)
        h, w = int(rng.integers(low, high+1)), int(rng.integers(low, high+1))
        if regime == "shape":
            if not is_test:
                w = h
            elif h == w:
                w = 4 if h != 4 else 7
        if mode == "diagonals":
            h = max(h, (w+1)//2+2)
        if mode == "staircase":
            h = max(h, 6)
        if regime == "shape":
            if not is_test:
                w = h
            elif h == w:
                w = h-1 if h > 4 else h+1
        if regime == "reach" and is_test and max(h, w) == 6:
            h = 7
        nplates = int(rng.integers(3, 6)) if is_test and regime == "fanout" else int(rng.integers(1, 4))
        if not is_test and regime == "fanout":
            nplates = int(rng.integers(1, 3))
        # Every straight control carries two plates, making color transfer visible.
        if mode == "isolated" and not is_test:
            nplates = 2
        try:
            positions = _positions(rng, h, w, nplates+1)
            ports, points, events = _port_scene(rng, h, w, mode, allowed,
                                               int(rng.integers(0, 4 if is_test else 3)))
            _require(allow_chain or not any(e["depth"] > 1 for e in events), "untaught bundle encounter")
            light = np.zeros((h, w), dtype=bool)
            for r, c in points:
                if 1 <= r <= h and 1 <= c <= w:
                    light[r-1, c-1] = True
            lit, dark = list(map(tuple, np.argwhere(light))), list(map(tuple, np.argwhere(~light)))
            _require(len(lit) > nplates and len(dark) >= nplates, "insufficient selective stencil space")
            if palette is None:
                role_palette = [int(c) for c in rng.permutation(10)]
            else:
                role_palette = list(palette)
            bg, marker = role_palette[:2]
            payloads = role_palette[2:2+nplates]
            nd = int(rng.integers(0, 3))
            distractors = role_palette[2+nplates:2+nplates+nd]
            ink = np.full((h, w), bg, dtype=np.int8)
            rng.shuffle(lit); rng.shuffle(dark)
            reserved = set()
            for color, p, q in zip(payloads, lit, dark):
                ink[p] = ink[q] = color
                reserved.update((p, q))
            # A blank illuminated cell rules out copying the entire beam mask.
            reserved.add(lit[nplates])
            for p in np.ndindex(h, w):
                if p not in reserved and rng.random() < float(rng.uniform(.3, .8)):
                    ink[p] = _pick(rng, payloads+distractors)
            for d in distractors:
                free = [p for p in np.ndindex(h, w) if p not in reserved]
                _require(bool(free), "no distractor space")
                p = _pick(rng, free)
                ink[p] = d
                reserved.add(p)
            if events:
                straight, _ = _trace(h, w, ports, straight=True)
                straight_mask = np.zeros((h, w), dtype=bool)
                for r, c in straight:
                    if 1 <= r <= h and 1 <= c <= w:
                        straight_mask[r-1, c-1] = True
                witness = list(map(tuple, np.argwhere(light != straight_mask)))
                _require(bool(witness), "collision has no visible consequence")
                # Preserve the guaranteed blank and each color's lit/unlit witnesses.
                free = [p for p in witness if p not in reserved]
                if free:
                    ink[_pick(rng, free)] = _pick(rng, payloads)
                _require(any(np.any((ink == c) & (light != straight_mask)) for c in payloads),
                         "collisions must change a plate, too")
            a = np.full((30, 30), bg, dtype=np.int8)
            y, x = positions[0]
            for r, c in _ring(h, w):
                if (r, c) not in ports:
                    a[y+r, x+c] = marker
            a[y+1:y+h+1, x+1:x+w+1] = ink
            for color, (r, c) in zip(payloads, positions[1:]):
                a[r:r+h+2, c:c+w+2] = color
            yy, xx = np.nonzero(a != bg)
            a = a[yy.min()-1:yy.max()+2, xx.min()-1:xx.max()+2]
            padding = []
            for dim in a.shape:
                extra = int(rng.integers(min(4, 30-dim)+1))
                before = int(rng.integers(extra+1))
                padding.append((before, extra-before))
            a = np.pad(a, padding, constant_values=bg)
            while np.count_nonzero(a == bg) <= max(np.count_nonzero(a == c) for c in role_palette[1:]):
                axis = 0 if a.shape[0] < 30 else 1
                _require(a.shape[axis] < 30, "background not dominant")
                pad = [(0, 0), (0, 0)]; pad[axis] = (0, 1)
                a = np.pad(a, pad, constant_values=bg)
            a = np.rot90(a, int(rng.integers(4)))
            if rng.random() < .5:
                a = np.fliplr(a)
            a = np.ascontiguousarray(a)
            info = _parse(a)
            light, pp, ee = _illumination(info)
            _require({e["kind"] for e in ee} <= allowed, "transformed scene changed mechanisms")
            _require(mode == "isolated" or any(e["kind"] == mode for e in ee), "missing final primary mechanism")
            if mode == "perpendicular":
                _require(any(e["kind"] == mode and e["depth"] == 1 and e["cuts"][0] != e["cuts"][1]
                             for e in ee), "perpendicular demonstration must reveal spatial timing")
            if contacts is not None:
                _require({e["kind"]+":"+e["intersection"] for e in ee} <= contacts,
                         "untaught intersection geometry")
            if mode == "isolated" and not is_test:
                _require(_orientation_candidates(info, light) == {0}, "control must reveal registration")
            out = _render(a, info, light, pp)
            return {"input": a, "output": out}
        except AssertionError:
            continue
    raise RuntimeError("bounded scene construction exhausted")


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    vocabulary = ["perpendicular", "diagonals", "staircase"]
    rng.shuffle(vocabulary)
    kinds = vocabulary[:int(rng.integers(1, min(3, nt-1)+1))]
    regime = _pick(rng, ["reach", "fanout", "shape"])
    fixed_palette = tuple(int(c) for c in rng.permutation(10)) if rng.random() < .25 else None
    train_modes = ["isolated"] + kinds
    while len(train_modes) < nt:
        train_modes.append(_pick(rng, ["isolated"]+kinds))
    rng.shuffle(train_modes)
    puzzle = {"train": [], "test": []}
    seen = set()
    taught = set()
    taught_chain = False
    for split, count in (("train", nt), ("test", ne)):
        variants = _surviving_alternatives(puzzle["train"]) if split == "test" else []
        for index in range(count):
            mode = train_modes[index] if split == "train" else _pick(rng, kinds)
            allowed = set(kinds) if split == "train" and mode != "isolated" else (
                set() if split == "train" else taught)
            for _ in range(16):
                pair = _make_pair(rng, mode, allowed, regime, split == "test", fixed_palette,
                                  allow_chain=split == "train" or taught_chain,
                                  contacts=taught if split == "test" else None)
                if split == "test" and not all(_alternative_agrees(pair, v) for v in variants):
                    continue
                if _key(pair["input"]) not in seen:
                    info = _parse(pair["input"])
                    _, _, events = _illumination(info)
                    if split == "train":
                        taught.update(e["kind"] for e in events)
                        taught.update(e["kind"]+":"+e["intersection"] for e in events)
                        taught_chain |= any(e["depth"] > 1 for e in events)
                    puzzle[split].append(pair)
                    seen.add(_key(pair["input"]))
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split type")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, taught, needed = set(), set(), set()
    has_control = has_color = has_collision_plate = False
    orientation = set(range(8))
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == out.shape and _key(a) not in seen, "shape change or duplicate input")
            seen.add(_key(a))
            info = _parse(a)
            light, points, events = _illumination(info)
            expected = _render(a, info, light, points)
            _require(np.array_equal(out, expected), "incorrect beam or imprint")
            _require(not np.array_equal(a, out), "unchanged pair")
            h, w = info["shape"]
            for color, (y, x) in info["plates"].items():
                tile = out[y:y+h+2, x:x+w+2]
                border = np.ones(tile.shape, bool); border[1:-1, 1:-1] = False
                _require(np.all(tile[border] == color), "plate border changed")
                _require(np.any((info["ink"] == color) & light), "plate lacks illuminated ink")
                _require(np.any((info["ink"] == color) & ~light), "plate lacks unlit ink")
                _require(np.count_nonzero(tile == info["bg"]) ==
                         np.count_nonzero((info["ink"] == color) & light), "hole/ink conservation")
            _require(not np.array_equal(out, _render(a, info, light, points, all_ink=True)),
                     "carve-all shortcut survives")
            _require(not np.array_equal(out, _render(a, info, light, points, all_beam=True)),
                     "beam-only shortcut survives")
            if events:
                sl, sp, _ = _illumination(info, straight=True)
                changed_plate = any(np.any((info["ink"] == c) & (sl != light)) for c in info["plates"])
                _require(changed_plate, "omitting collisions must alter subtraction")
                if split == "train":
                    has_collision_plate = True
            if split == "train":
                taught.update(e["kind"] for e in events)
                taught.update(e["kind"]+":"+e["intersection"] for e in events)
                if any(e["depth"] > 1 for e in events):
                    taught.add("bundle")
                orientation &= _orientation_candidates(info, light)
                has_control |= not events
                has_color |= len(info["plates"]) >= 2
            else:
                needed.update(e["kind"] for e in events)
                needed.update(e["kind"]+":"+e["intersection"] for e in events)
                if any(e["depth"] > 1 for e in events):
                    needed.add("bundle")
    _require(needed <= taught, "untaught test collision type")
    _require(has_control and has_color and has_collision_plate, "inadequate demonstrations")
    _require(orientation == {0}, "ambiguous imprint orientation")
    _require(all(_alternative_agrees(p, v) for v in _surviving_alternatives(puzzle["train"])
                 for p in puzzle["test"]), "training leaves a competing test prediction")


def self_test() -> dict:
    """Bounded audit of 42 seeds, structural variation and counterfactuals."""
    # Unequal-distance perpendicular encounter; explicit geometric fixture.
    points, events = _trace(5, 7, [(2, 0), (0, 5)])
    fixture = {(2, c) for c in range(5)} | {(0, 5), (1, 5),
               (3, 5), (4, 6), (5, 7), (2, 6), (3, 7)}
    _require(points == fixture and events[0]["cuts"] == (1, 4), "spatial encounter fixture")
    # Major-axis-first staircase; the outgoing strand can visit the would-be
    # shared cell after the incoming paths turn.
    points, _ = _trace(8, 6, [(0, 0), (0, 3)])
    fixture = {(0, 0), (1, 1), (2, 2), (0, 3), (1, 3), (2, 3),
               (3, 2), (4, 3), (5, 3), (6, 4), (7, 4), (8, 5),
               (3, 3), (4, 4), (5, 4), (6, 5), (7, 5), (8, 6)}
    _require(points == fixture, "major-axis-first staircase fixture")
    # Smaller register and six plates outside the automatic sampler's range.
    a = np.zeros((22, 23), dtype=np.int8)
    for r, c in _ring(3, 4):
        a[1+r, 1+c] = 1
    a[1, 3] = 0
    for color, p in zip(range(2, 8), [(0, 1), (1, 1), (2, 1), (0, 3), (1, 0), (2, 2)]):
        a[2+p[0], 2+p[1]] = color
    for color, (r, c) in zip(range(2, 8), [(1, 8), (1, 15), (8, 1), (8, 8), (8, 15), (15, 1)]):
        a[r:r+5, c:c+6] = color
    expected = a.copy(); expected[1:5, 3] = 1
    for p in ((2, 10), (3, 17), (11, 3)):
        expected[p] = 0
    _require(np.array_equal(solve(a), expected), "registered six-plate fixture")
    stats = {k: set() for k in ("puzzles", "shapes", "registers", "plate_counts", "ports", "kinds", "intersections", "roles", "structures", "depths", "distractors")}
    event_count = dark_ink = staircase_witness = unequal = 0
    shifts = {"reach": 0, "fanout": 0, "shape": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        infos = {s: [_parse(p["input"]) for p in puzzle[s]] for s in puzzle}
        visible_shifts = {
            "reach": min(max(i["shape"]) for i in infos["test"]) > max(max(i["shape"]) for i in infos["train"]),
            "fanout": min(len(i["plates"]) for i in infos["test"]) > max(len(i["plates"]) for i in infos["train"]),
            "shape": all(i["shape"][0] == i["shape"][1] for i in infos["train"]) and
                     all(i["shape"][0] != i["shape"][1] for i in infos["test"])}
        _require(any(visible_shifts.values()), "no visible train/test shift")
        for name, present in visible_shifts.items():
            shifts[name] += int(present)
        stats["puzzles"].add(tuple(_key(p["input"]) for s in puzzle.values() for p in s))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                info = _parse(a)
                light, _, events = _illumination(info)
                _require(np.array_equal(solve(a.copy()), pair["output"]), "solve mismatch")
                stats["shapes"].add(a.shape)
                stats["registers"].add(info["shape"])
                stats["plate_counts"].add(len(info["plates"]))
                stats["ports"].add(len(info["ports"]))
                stats["roles"].add((info["bg"], info["marker"]))
                stats["distractors"].add(len(set(map(int, np.unique(info["ink"]))) - {info["bg"]} - set(info["plates"])))
                roles = {}
                stats["structures"].add((a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)))
                for event in events:
                    event_count += 1
                    stats["kinds"].add(event["kind"])
                    stats["intersections"].add(event["intersection"])
                    stats["depths"].add(event["depth"])
                    unequal += event["cuts"][0] != event["cuts"][1]
                dark_ink += np.count_nonzero((info["ink"] != info["bg"]) & ~light)
                if any(e["kind"] == "staircase" for e in events):
                    try:
                        alt, _, _ = _illumination(info, sign_sum=True)
                        staircase_witness += not np.array_equal(light, alt)
                    except AssertionError:
                        staircase_witness += 1
    _require(len(stats["puzzles"]) == 42, "seed diversity")
    _require(len(stats["shapes"]) >= 35 and len(stats["registers"]) >= 16, "geometry diversity")
    _require({1, 2, 3, 4, 5} <= stats["plate_counts"], "plate count diversity")
    _require({1, 2, 3, 4} <= stats["ports"], "port count diversity")
    _require(stats["kinds"] == {"perpendicular", "diagonals", "staircase"}, "collision diversity")
    _require(stats["intersections"] == {"cell", "between"}, "contact diversity")
    _require(len(stats["roles"]) >= 20 and len(stats["structures"]) >= 180, "palette/structure diversity")
    _require(staircase_witness > 10 and unequal > 10 and dark_ink > 100, "missing mechanism evidence")
    _require(min(shifts.values()) >= 3, "curriculum diversity")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            generate_puzzle(1701+nt*10+ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first, again = generate_puzzle(91, 2, 1), generate_puzzle(91, 2, 1)
    _require(all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], again[s])
                 for k in ("input", "output")), "nondeterminism")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in pairs] for s, pairs in first.items()}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": 210, "events": event_count, "unequal_encounters": int(unequal),
            "staircase_counterexamples": int(staircase_witness),
            "visible_shift_counts": shifts,
            **{k: len(v) for k, v in stats.items()}}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
