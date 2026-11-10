"""Beam-registered jigsaws: boundary clues constrain the assembled positions.

All solution information is visible in a single input. No parent code, labels,
seed, training pairs, or generated-example cache is used by solve().
"""
from __future__ import annotations

import numpy as np


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(d < 1 or d > 30 for d in a.shape)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        cells = [p]
        for r, c in cells:
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in unseen:
                    unseen.remove(q)
                    cells.append(q)
        result.append(cells)
    return result


def _marker(point, corner):
    r, c = point
    dy = -1 if corner[0] == "T" else 1
    dx = -1 if corner[1] == "L" else 1
    return ((r+dy, c), (r, c+dx), (r+dy, c+dx))


def _ports(labels, offset):
    """(local row, local column, outward dy, outward dx) per side/slot."""
    length = labels.size // 2
    result = []
    for side in range(2):
        row = []
        for slot in range(length):
            p = np.argwhere(labels == side*length+slot)[0]
            q = np.argwhere(labels == (1-side)*length+slot)[0]
            row.append((int(p[0])+offset[0], int(p[1])+offset[1],
                        int(p[0]-q[0]), int(p[1]-q[1])))
        result.append(tuple(row))
    return tuple(result)


def _extract(g):
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("background must be uniquely most frequent")
    candidates = []
    for color in range(10):
        q = np.argwhere(g == color)
        if color == bg or len(q) != 3:
            continue
        lo, hi = q.min(0), q.max(0)
        if tuple(hi-lo) != (1, 1):
            continue
        square = {(int(r), int(c)) for r in (lo[0], hi[0])
                  for c in (lo[1], hi[1])}
        missing = (square - set(map(tuple, q.tolist()))).pop()
        if g[missing] in (bg, color):
            continue
        corner = {(1, 1): "TL", (1, 0): "TR", (0, 1): "BL",
                  (0, 0): "BR"}[tuple(np.array(missing)-lo)]
        candidates.append((color, missing, corner))
    if len(candidates) != 1:
        raise ValueError("expected one external L marker")
    marker, point, corner = candidates[0]
    border = np.zeros(g.shape, bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    signals = [(int(r), int(c), int(g[r, c]))
               for r, c in np.argwhere(border & (g != bg))]
    if not signals or any(r in (0, g.shape[0]-1) and c in (0, g.shape[1]-1)
                          for r, c, _ in signals):
        raise ValueError("non-corner boundary beacons required")
    mask = (g != bg) & (g != marker) & ~border
    pieces, anchor = [], None
    carrier = None
    for cells in _components(mask):
        q = np.array(cells)
        lo, hi = q.min(0), q.max(0)+1
        r, c = map(int, lo)
        a = g[r:int(hi[0]), c:int(hi[1])].copy()
        if a.size != len(cells) or min(a.shape) < 4:
            raise ValueError("tiles must be isolated filled rectangles")
        rim = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
        if len(np.unique(rim)) != 1:
            raise ValueError("uniform tile rims required")
        value = int(rim[0])
        if carrier is None:
            carrier = value
        if value != carrier:
            raise ValueError("one common carrier required")
        bq = np.argwhere(a != carrier)
        if not len(bq):
            raise ValueError("tile has no emitter")
        blo, bhi = bq.min(0), bq.max(0)+1
        br, bc = map(int, blo)
        bar = a[br:int(bhi[0]), bc:int(bhi[1])].copy()
        if (len(bq) != bar.size or min(bar.shape) != 2
                or not 4 <= max(bar.shape) <= 7):
            raise ValueError("one solid 2 by L emitter inside every tile")
        if br < 1 or bc < 1 or bhi[0] >= a.shape[0] or bhi[1] >= a.shape[1]:
            raise ValueError("emitter must have a carrier margin")
        if point in cells:
            anchor = len(pieces)
        pieces.append(dict(arr=a, pos=(r, c), bar=bar, offset=(br, bc)))
    if not 3 <= len(pieces) <= 6 or anchor is None:
        raise ValueError("expected 3..6 tiles and a marked anchor")
    canonical = pieces[0]["bar"]
    if canonical.shape[0] != 2:
        canonical = np.rot90(canonical).copy()
    forms = [_orient(canonical, t) for t in range(8)]
    if len({_key(a) for a in forms}) != 8:
        raise ValueError("emitter correspondence is ambiguous")
    structural = set(map(int, canonical.flat))
    if not 2 <= len(structural) <= 3:
        raise ValueError("expected 2..3 emitter colors")
    roles = structural | {bg, carrier, marker}
    if len(roles) != len(structural)+3 or roles & {c for _, _, c in signals}:
        raise ValueError("local color roles must be disjoint")
    labels = np.arange(canonical.size).reshape(canonical.shape)
    for p in pieces:
        ts = [t for t, a in enumerate(forms) if np.array_equal(a, p["bar"])]
        if len(ts) != 1:
            raise ValueError("tiles do not contain copies of one emitter")
        p["transform"] = ts[0]
        p["ports"] = _ports(_orient(labels, ts[0]), p["offset"])
    ar, ac = pieces[anchor]["pos"]
    ah, aw = pieces[anchor]["arr"].shape
    if point != (ar+(ah-1 if corner[0] == "B" else 0),
                 ac+(aw-1 if corner[1] == "R" else 0)):
        raise ValueError("marker must touch the corresponding tile corner")
    return dict(bg=bg, marker=marker, carrier=carrier, structural=structural,
                point=point, corner=corner, anchor=anchor, pieces=pieces,
                signals=signals, shape=g.shape, length=canonical.shape[1])


def _packings(info):
    """Complete translation-only rectangular packing search; no edge heuristic."""
    pieces, anchor = info["pieces"], info["anchor"]
    shapes = [p["arr"].shape for p in pieces]
    area = sum(h*w for h, w in shapes)
    canvas = info["shape"]
    point, corner = info["point"], info["corner"]
    nodes = 0
    for h in range(1, canvas[0]-1):
        if area % h:
            continue
        w = area // h
        oy = point[0]-(h-1 if corner[0] == "B" else 0)
        ox = point[1]-(w-1 if corner[1] == "R" else 0)
        # Assembled carrier stays strictly inside the boundary beacons.
        if (oy < 1 or ox < 1 or oy+h >= canvas[0] or ox+w >= canvas[1]
                or any(ph > h or pw > w for ph, pw in shapes)):
            continue
        bits = [sum(((1 << pw)-1) << (y*w) for y in range(ph))
                for ph, pw in shapes]
        ar, ac = pieces[anchor]["pos"]
        y, x = ar-oy, ac-ox
        ah, aw = shapes[anchor]
        if y < 0 or x < 0 or y+ah > h or x+aw > w:
            continue
        positions = [None]*len(pieces)
        positions[anchor] = (y, x)
        full = (1 << area)-1

        def visit(occupied, remaining):
            nonlocal nodes
            nodes += 1
            if nodes > 120000:
                raise ValueError("packing search exceeded its complete-search budget")
            if not remaining:
                yield (h, w), (oy, ox), tuple((r+oy, c+ox) for r, c in positions)
                return
            free = full ^ occupied
            r, c = divmod((free & -free).bit_length()-1, w)
            for i in remaining:
                ph, pw = shapes[i]
                if r+ph > h or c+pw > w:
                    continue
                b = bits[i] << (r*w+c)
                if not b & occupied:
                    positions[i] = (r, c)
                    yield from visit(occupied | b, tuple(j for j in remaining if j != i))
        yield from visit(bits[anchor] << (y*w+x),
                         tuple(i for i in range(len(pieces)) if i != anchor))


def _ray(port, position, shape):
    r, c, dy, dx = port
    r += position[0]+dy
    c += position[1]+dx
    cells = []
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        cells.append((r, c))
        r += dy
        c += dx
    return tuple(cells)


def _registered(info, packing):
    """Test a tiling using beacons, then complete its pooled ray code."""
    target, origin, positions = packing
    pieces, signals, shape = info["pieces"], info["signals"], info["shape"]
    bars = set()
    for p, (r, c) in zip(pieces, positions):
        br, bc = p["offset"]
        for y in range(p["bar"].shape[0]):
            for x in range(p["bar"].shape[1]):
                bars.add((r+br+y, c+bc+x))
    code, sides, observations = {}, set(), []
    allocations = [set() for _ in pieces]
    for sr, sc, color in signals:
        matches = []
        for i, (p, pos) in enumerate(zip(pieces, positions)):
            for side, ports in enumerate(p["ports"]):
                for slot, port in enumerate(ports):
                    r, c, dy, dx = port
                    r += pos[0]
                    c += pos[1]
                    aligned = ((dy and sc == c and (sr-r)*dy > 0)
                               or (dx and sr == r and (sc-c)*dx > 0))
                    if not aligned:
                        continue
                    ray = _ray(port, pos, shape)
                    if ray[-1] == (sr, sc) and bars.isdisjoint(ray):
                        matches.append((i, side, slot))
        if len(matches) != 1:
            return None
        i, side, slot = matches[0]
        if slot in code and code[slot] != color:
            return None
        code[slot] = color
        sides.add(side)
        allocations[i].add(slot)
        observations.append((i, side, slot, color))
    if len(sides) != 1 or not 1 <= len(code) <= 4:
        return None
    side = sides.pop()
    painted, directions, lengths, along = {}, [], [], []
    for p, pos in zip(pieces, positions):
        directions.append(p["ports"][side][0][2:])
        first, last = p["ports"][side][0], p["ports"][side][-1]
        along.append((last[0]-first[0], last[1]-first[1]))
        for slot, color in code.items():
            ray = _ray(p["ports"][side][slot], pos, shape)
            if len(ray) < 2 or not bars.isdisjoint(ray):
                return None
            for q in ray:
                if q in painted:
                    return None
                painted[q] = color
            lengths.append(len(ray))
    if any(painted.get((r, c)) != color for r, c, color in signals):
        return None
    scene = np.full(shape, info["bg"], np.int8)
    for p, (r, c) in zip(pieces, positions):
        h, w = p["arr"].shape
        scene[r:r+h, c:c+w] = p["arr"]
    out = scene.copy()
    for q, color in painted.items():
        out[q] = color
    reflected = any(directions[i] == directions[j]
                    and along[i] == tuple(-v for v in along[j])
                    for i in range(len(pieces)) for j in range(i))
    asymmetric_code = code != {info["length"]-1-s: c for s, c in code.items()}
    features = dict(target=target, origin=origin, positions=positions,
                    code=tuple(sorted(code.items())), side=side,
                    allocations=tuple(len(s) for s in allocations),
                    directions=tuple(directions), ray_lengths=tuple(lengths),
                    pooled=all(len(s) < len(code) for s in allocations),
                    unhinted=any(not s for s in allocations),
                    redundant=len(signals) > len(code), scene=scene,
                    reflected_transfer=reflected, asymmetric_code=asymmetric_code,
                    painted=painted, observations=tuple(observations))
    return out, features


def _infer(g):
    info = _extract(g)
    answer, winner = None, None
    raw_keys, valid_packings = set(), 0
    for packing in _packings(info):
        target, origin, positions = packing
        # Distinct pre-ray pictures certify actual assembly ambiguity.
        signatures = tuple(sorted((r+p["offset"][0], c+p["offset"][1],
                                   _key(p["bar"]))
                                  for p, (r, c) in zip(info["pieces"], positions)))
        raw_keys.add((target, origin, signatures))
        result = _registered(info, packing)
        if result is None:
            continue
        out, features = result
        valid_packings += 1
        if answer is not None and not np.array_equal(out, answer):
            raise ValueError("beacons leave different valid answers")
        answer, winner = out, features
    if answer is None:
        raise ValueError("no tiling explains the boundary ray code")
    winner["raw_assemblies"] = len(raw_keys)
    winner["valid_packings"] = valid_packings
    winner["corner"] = info["corner"]
    winner["piece_count"] = len(info["pieces"])
    winner["length"] = info["length"]
    winner["shape"] = g.shape
    winner["background"] = info["bg"]
    winner["carrier"] = info["carrier"]
    winner["marker"] = info["marker"]
    winner["transforms"] = tuple(p["transform"] for p in info["pieces"])
    rects = [(*pos, *p["arr"].shape)
             for p, pos in zip(info["pieces"], winner["positions"])]
    oy, ox = winner["origin"]
    th, tw = winner["target"]
    xs = {x for r, c, h, w in rects for x in (c, c+w) if ox < x < ox+tw}
    ys = {y for r, c, h, w in rects for y in (r, r+h) if oy < y < oy+th}
    winner["partial_seam"] = (any(any(c < x < c+w for r, c, h, w in rects) for x in xs)
                              or any(any(r < y < r+h for r, c, h, w in rects) for y in ys))
    # A counterfactual: leave tiles scattered and try the same ray decoder.
    stationary = (g.shape, (0, 0), tuple(p["pos"] for p in info["pieces"]))
    fixed_result = _registered(info, stationary)
    winner["needs_assembly"] = fixed_result is None or not np.array_equal(fixed_result[0], answer)
    return answer, winner, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the unique assembly and code from the input alone."""
    return _infer(_grid(input_grid))[0]


def _template(rng, length, colors):
    for _ in range(50):
        a = rng.choice(colors, (2, length)).astype(np.int8)
        if rng.random() < .65:
            a[0] = colors[0]
            a[1] = colors[1]
            a[1, :int(rng.integers(1, length))] = colors[0]
            for j, color in enumerate(colors[2:]):
                a[1, j+1] = color
        if set(map(int, a.flat)) == set(colors) and len({_key(_orient(a, t)) for t in range(8)}) == 8:
            return a
    a = np.full((2, length), colors[0], np.int8)
    a[1, 1:] = colors[1]
    for j, color in enumerate(colors[2:]):
        a[1, j+2] = color
    return a


def _partition(rng, mode, length):
    """Construct compatible perimeter partitions, not completed parent puzzles."""
    if mode in ("hstrip", "vstrip"):
        count = int(rng.integers(3, 5))
        height = int(rng.integers(4, 9))
        widths = [length+2+int(rng.integers(2)) for _ in range(count)]
        if sum(widths) > 26:
            return None
        rects, x = [], 0
        for w in widths:
            rects.append((0, x, height, w))
            x += w
        if mode == "vstrip":
            rects = [(c, r, w, h) for r, c, h, w in rects]
        return rects
    if mode == "outer":
        count = int(rng.integers(2, 4))
        heights = [length+2+int(rng.integers(2)) for _ in range(count)]
        if sum(heights) > 25:
            return None
        left, right = int(rng.integers(4, 8)), int(rng.integers(4, 9))
        rects, y = [(0, 0, sum(heights), left)], 0
        for h in heights:
            rects.append((y, left, h, right))
            y += h
        if rng.integers(2):
            rects = [(c, r, w, h) for r, c, h, w in rects]
        return rects
    cols = 3 if mode == "six" else 2
    widths = [length+2+int(rng.integers(2)) for _ in range(cols)]
    heights = [length+2+int(rng.integers(2)) for _ in range(2)]
    if sum(widths) > 26:
        return None
    if mode == "staggered":
        widths[0] *= 2
        if sum(widths) > 26:
            return None
    rects, y = [], 0
    for h in heights:
        x = 0
        for w in widths:
            rects.append((y, x, h, w))
            x += w
        y += h
    if mode == "staggered":
        r, c, h, w = rects.pop(0)
        cut = w//2
        rects.extend(((r, c, h, cut), (r, c+cut, h, w-cut)))
    return rects


def _gap(a, b):
    r, c, h, w = a
    y, x, ph, pw = b
    return r+h < y or y+ph < r or c+w < x or x+pw < c


def _scatter(rng, arrays, final, anchor, marks, shape):
    remaining = sorted((i for i in range(len(arrays)) if i != anchor),
                       key=lambda i: -arrays[i].size)
    for _ in range(16):
        pos = [None]*len(arrays)
        pos[anchor] = final[anchor]
        placed = [(*final[anchor], *arrays[anchor].shape)]
        for i in remaining:
            h, w = arrays[i].shape
            for _ in range(180):
                r = int(rng.integers(1, shape[0]-h))
                c = int(rng.integers(1, shape[1]-w))
                rect = (r, c, h, w)
                if any(r-1 <= y <= r+h and c-1 <= x <= c+w for y, x in marks):
                    continue
                if all(_gap(rect, p) for p in placed):
                    pos[i] = (r, c)
                    placed.append(rect)
                    break
            else:
                break
        if all(p is not None for p in pos):
            return pos
    return None


def _make_pair(rng, mode, corner, demonstration=None):
    for attempt in range(100):
        length = int(rng.integers(4, 7 if mode in ("six", "staggered") else 8))
        rects = _partition(rng, mode, length)
        if rects is None:
            continue
        th = max(r+h for r, c, h, w in rects)
        tw = max(c+w for r, c, h, w in rects)
        desired_area = th*tw*2.4
        if rng.random() < .35:
            bh = bw = max(int(np.ceil(np.sqrt(desired_area))), th+4, tw+4, 19)
        else:
            # Independent canvas axes: narrow assemblies need not sit on squares.
            aspect = float(rng.uniform(.6, 1.65))
            bh = max(th+4, min(30, int(np.ceil(np.sqrt(desired_area*aspect)))))
            bw = max(tw+4, int(np.ceil(desired_area/bh)))
        if bh > 30 or bw > 30:
            continue
        shape = (min(30, bh+int(rng.integers(3))),
                 min(30, bw+int(rng.integers(3))))
        origin = (int(rng.integers(2, shape[0]-th-1)),
                  int(rng.integers(2, shape[1]-tw-1)))
        point = (origin[0]+(th-1 if corner[0] == "B" else 0),
                 origin[1]+(tw-1 if corner[1] == "R" else 0))
        marks = _marker(point, corner)
        colors = list(map(int, rng.permutation(10)))
        bg, marker, carrier = colors[:3]
        structural_count = int(rng.integers(2, 4))
        structural = colors[3:3+structural_count]
        count = int(rng.integers(2, min(4, 7-structural_count)+1))
        slots = tuple(sorted(map(int, rng.choice(length, count, replace=False))))
        values = colors[3+structural_count:3+structural_count+count]
        if demonstration is None and count >= 3 and rng.random() < .3:
            values[-1] = values[0]
        code = dict(zip(slots, values))
        template = _template(rng, length, structural)
        labels = np.arange(template.size).reshape(template.shape)
        arrays, ports, transforms = [], [], []
        for i, (r, c, h, w) in enumerate(rects):
            directions = []
            if r == 0 and w >= length+2:
                directions.append((-1, 0))
            if r+h == th and w >= length+2:
                directions.append((1, 0))
            if c == 0 and h >= length+2:
                directions.append((0, -1))
            if c+w == tw and h >= length+2:
                directions.append((0, 1))
            if not directions:
                break
            options = [t for t in range(8)
                       if _ports(_orient(labels, t), (0, 0))[0][0][2:] in directions]
            t = int(rng.choice(options))
            if demonstration == "pool" and mode == "hstrip" and i < 2:
                t = (0, 4)[i]
            bar = _orient(template, t)
            br = int(rng.integers(1, h-bar.shape[0]))
            bc = int(rng.integers(1, w-bar.shape[1]))
            a = np.full((h, w), carrier, np.int8)
            a[br:br+bar.shape[0], bc:bc+bar.shape[1]] = bar
            arrays.append(a)
            ports.append(_ports(_orient(labels, t), (br, bc)))
            transforms.append(t)
        if len(arrays) != len(rects):
            continue
        if demonstration == "turn" and len({t % 2 for t in transforms}) != 2:
            continue
        final = [(origin[0]+r, origin[1]+c) for r, c, h, w in rects]
        anchor = next(i for i, (r, c, h, w) in enumerate(rects)
                      if r <= point[0]-origin[0] < r+h
                      and c <= point[1]-origin[1] < c+w)
        positions = _scatter(rng, arrays, final, anchor, marks, shape)
        if positions is None:
            continue
        endpoints = {(i, s): _ray(ports[i][0][s], final[i], shape)[-1]
                     for i in range(len(arrays)) for s in slots}
        allocation = [set() for _ in arrays]
        # In the pooling demonstration, every tile individually lacks some code.
        skip = int(rng.integers(len(arrays))) if rng.random() < .3 else -1
        for i in range(len(arrays)):
            if i != skip:
                allocation[i].add(int(rng.choice(slots)))
        for s in slots:
            if not any(s in a for a in allocation):
                choices = [i for i, a in enumerate(allocation)
                           if demonstration != "pool" or len(a) < count-1]
                if not choices:
                    break
                allocation[int(rng.choice(choices))].add(s)
        if set().union(*allocation) != set(slots):
            continue
        if demonstration != "pool":
            for _ in range(int(rng.integers(len(arrays)+1))):
                allocation[int(rng.integers(len(arrays)))].add(int(rng.choice(slots)))
        inp = np.full(shape, bg, np.int8)
        for a, (r, c) in zip(arrays, positions):
            inp[r:r+a.shape[0], c:c+a.shape[1]] = a
        for q in marks:
            inp[q] = marker
        for i, observed in enumerate(allocation):
            for s in observed:
                inp[endpoints[i, s]] = code[s]
        expected = np.full(shape, bg, np.int8)
        for a, (r, c) in zip(arrays, final):
            expected[r:r+a.shape[0], c:c+a.shape[1]] = a
        for i, pos in enumerate(final):
            for s, color in code.items():
                for q in _ray(ports[i][0][s], pos, shape):
                    expected[q] = color
        try:
            answer, f, _ = _infer(inp)
        except ValueError:
            continue
        if (not np.array_equal(answer, expected) or f["raw_assemblies"] < 2
                or not f["needs_assembly"]):
            continue
        if demonstration == "pool" and not f["pooled"]:
            continue
        return {"input": inp, "output": expected}
    # Resample the same semantics in a roomier simple partition. No special rule.
    if mode not in ("hstrip", "grid"):
        return _make_pair(rng, "grid" if demonstration == "turn" else "hstrip",
                          corner, demonstration)
    raise RuntimeError("could not find a uniquely beam-registered jigsaw")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+" must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name+" out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(3))
    corners = ("TL", "TR", "BR", "BL")
    start = int(rng.integers(4))
    training_corners = (corners[start], corners[(start+2) % 4])
    test_corners = (corners[(start+1) % 4], corners[(start+3) % 4]) if shift == 1 else corners
    puzzle, seen = {"train": [], "test": []}, set()
    first_corner = None
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            demonstration = "pool" if split == "train" and i == 0 else "turn" if split == "train" and i == 1 else None
            if demonstration:
                mode = "hstrip" if demonstration == "pool" else "grid"
            elif split == "test":
                mode = str(rng.choice(("six", "staggered") if shift == 0 else
                                      ("outer", "staggered") if shift == 2 else
                                      ("hstrip", "vstrip", "grid", "outer", "six", "staggered")))
            else:
                mode = str(rng.choice(("hstrip", "vstrip", "grid") if shift == 2 else
                                      ("hstrip", "vstrip", "grid", "outer", "six")))
            corner = str(rng.choice(training_corners if split == "train" else test_corners))
            if demonstration == "pool":
                first_corner = corner
            elif demonstration == "turn":
                # Two marker directions establish the geometric convention even
                # at the minimum legal training count. Consume the same RNG draw.
                corner = corners[(corners.index(first_corner)+2) % 4]
            for _ in range(10):
                pair = _make_pair(rng, mode, corner, demonstration)
                if split == "test" and shift in (0, 2):
                    f = _infer(pair["input"])[1]
                    if shift == 0 and f["piece_count"] < 5:
                        continue
                    if shift == 2 and not f["partial_seam"]:
                        continue
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise AssertionError("exactly train/test required")
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _count(len(puzzle["train"]), "train", 2, 10)
    _count(len(puzzle["test"]), "test", 1, 4)
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            expected, f, info = _infer(inp)
            assert np.array_equal(out, expected), "incorrect assembly or ray completion"
            assert f["raw_assemblies"] >= 2, "beams must resolve real packing ambiguity"
            assert f["needs_assembly"], "scattered-position ray shortcut must fail"
            # Both parent ablations change returned pixels, on every example.
            assert not np.array_equal(out, f["scene"]), "omitting the shared code must change the answer"
            assert not np.any(out == info["marker"])
            assert all(out[q] == info["bg"] for q in _marker(info["point"], info["corner"]))
            # Exact emitter preservation at all solved translations, fixed anchor.
            assert f["positions"][info["anchor"]] == info["pieces"][info["anchor"]]["pos"]
            for p, (r, c) in zip(info["pieces"], f["positions"]):
                br, bc = p["offset"]
                h, w = p["bar"].shape
                assert np.array_equal(out[r+br:r+br+h, c+bc:c+bc+w], p["bar"])
            # Painting can change only carrier/background; every other cell stays.
            for q, color in f["painted"].items():
                assert f["scene"][q] in (info["bg"], info["carrier"])
                assert out[q] == color
            if split == "train":
                if f["pooled"]:
                    partial = f["scene"].copy()
                    for i, side, slot, color in f["observations"]:
                        for q in _ray(info["pieces"][i]["ports"][side][slot],
                                      f["positions"][i], info["shape"]):
                            partial[q] = color
                    assert not np.array_equal(partial, out), "pooling must add missing rays"
                train.append(f)
    assert any(f["pooled"] and len(f["code"]) >= 2 and f["reflected_transfer"]
               and f["asymmetric_code"] for f in train)
    assert any(len({d[0] == 0 for d in f["directions"]}) == 2 for f in train)
    assert any(len(set(f["transforms"])) > 1 for f in train)


def self_test():
    """Bounded semantic/diversity checks on 42 seeds, plus equivariance tests."""
    puzzles = [generate_puzzle(s) for s in range(42)]
    features = []
    signatures = set()
    normalized = set()
    for p in puzzles:
        signatures.add(tuple(_key(x["input"]) for x in p["train"]+p["test"]))
        fs = [_infer(x["input"])[1] for x in p["train"]+p["test"]]
        features.extend(fs)
        assert len({f["corner"] for f in fs[:len(p["train"])]}) == 2
        assert len({f["target"] for f in fs}) > 1
        assert len({f["background"] for f in fs}) > 1
        for pair in p["train"]+p["test"]:
            names = {}
            normalized.add((pair["input"].shape, tuple(names.setdefault(int(c), len(names))
                                                     for c in pair["input"].flat)))
    assert len(signatures) == 42
    assert len(normalized) >= 180
    assert len({f["shape"] for f in features}) >= 15
    assert any(f["shape"][0] >= 1.5*f["shape"][1] for f in features)
    assert any(f["shape"][1] >= 1.5*f["shape"][0] for f in features)
    assert len({f["target"] for f in features}) >= 30
    assert {f["corner"] for f in features} == {"TL", "TR", "BL", "BR"}
    assert {f["piece_count"] for f in features} >= {3, 4, 5, 6}
    assert {f["length"] for f in features} == {4, 5, 6, 7}
    assert any(f["unhinted"] for f in features)
    assert any(len(set(f["directions"])) >= 3 for f in features)
    assert any(len(set(f["directions"])) < f["piece_count"] for f in features)
    assert any(len({c for s, c in f["code"]}) < len(f["code"]) for f in features)
    assert any(any(b-a == 1 for (a, _), (b, _) in zip(f["code"], f["code"][1:]))
               for f in features)
    assert all(f["raw_assemblies"] > 1 and f["needs_assembly"] for f in features)
    p = generate_puzzle(9081, 2, 1)
    repeat = generate_puzzle(9081, 2, 1)
    assert tuple(_key(x["input"]) for x in p["train"]+p["test"]) == tuple(
        _key(x["input"]) for x in repeat["train"]+repeat["test"])
    # Whole-grid rotations/reflections and arbitrary palette permutations must
    # commute with the input-only solution, including background-role swaps.
    pair = puzzles[0]["train"][1]
    for t in range(8):
        palette = np.roll(np.arange(10), t+1)
        a, b = (_orient(palette[pair[k]], t) for k in ("input", "output"))
        assert np.array_equal(solve(a), b)
    # Duplicate inputs, missing rays, and retained marker rejected.
    for mutation in ("ray", "marker", "duplicate"):
        damaged = {s: [{k: a.copy() for k, a in x.items()} for x in p[s]] for s in p}
        pair = damaged["test"][0]
        _, f, info = _infer(pair["input"])
        if mutation == "ray":
            q = next(iter(f["painted"]))
            pair["output"][q] = info["bg"]
        elif mutation == "marker":
            pair["output"][_marker(info["point"], info["corner"])[0]] = info["marker"]
        else:
            damaged["train"][1] = {k: a.copy() for k, a in damaged["train"][0].items()}
        try:
            validate_generated_puzzle(damaged)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("corruption accepted")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            q = generate_puzzle(1000+nt*10+ne, nt, ne)
            assert (len(q["train"]), len(q["test"])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
