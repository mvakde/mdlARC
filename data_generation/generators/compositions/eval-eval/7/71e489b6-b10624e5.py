"""Shared fault atlas: register, fuse, and broadcast logical-cell defects.

Only NumPy and the standard library are used. solve() reads one grid, including
its clean master and seal. Construction labels and input decoding are separate
paths; no parent generators, training labels, or random state enter solve().
"""
from itertools import permutations
import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 training, 1..4 test, and training > test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or not np.all((0 <= a) & (a <= 9)):
        raise ValueError("grid must contain integer colors 0..9")
    return a.astype(np.int64, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, points = [start], [start]
        while stack:
            y, x = stack.pop()
            for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    points.append(q)
        result.append(points)
    return result


def _signs(corner):
    return (1 if corner < 2 else -1, 1 if corner % 2 == 0 else -1)


def _corners(y, x, h, w):
    return ((y, x), (y, x+w-1), (y+h-1, x), (y+h-1, x+w-1))


def _frames(grid, bg):
    candidates = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        parts = _components(grid == color)
        if not 3 <= len(parts) <= 5:
            continue
        cards = []
        for points in parts:
            ys, xs = zip(*points)
            y, x, h, w = min(ys), min(xs), max(ys)-min(ys)+1, max(xs)-min(xs)+1
            if min(h, w) < 5:
                break
            perimeter = {(r, c) for r in range(y, y+h) for c in range(x, x+w)
                         if r in (y, y+h-1) or c in (x, x+w-1)}
            absent = perimeter - set(points)
            cs = _corners(y, x, h, w)
            if len(absent) != 1 or not absent <= set(cs) or set(points) != perimeter-absent:
                break
            corner = cs.index(next(iter(absent)))
            cards.append({"y": y+1, "x": x+1, "h": h-2, "w": w-2,
                          "corner": corner, "mark": int(grid[cs[corner]]),
                          "box": (y, x, h, w)})
        else:
            sealed = [i for i, c in enumerate(cards) if c["mark"] != bg]
            if len(sealed) != 1:
                continue
            halo = cards[sealed[0]]["mark"]
            if halo == color or np.count_nonzero(grid == halo) != 1:
                continue
            covered = np.zeros(grid.shape, dtype=bool)
            valid = True
            for c in cards:
                y, x, h, w = c["box"]
                if y < 1 or x < 1 or y+h >= grid.shape[0] or x+w >= grid.shape[1]:
                    valid = False
                    break
                if covered[max(0, y-1):y+h+1, max(0, x-1):x+w+1].any():
                    # Frames may have one field cell between them, but never touch.
                    valid = False
                    break
                covered[y:y+h, x:x+w] = True
            if valid and np.all(grid[~covered] == bg):
                candidates.append((color, halo, sealed[0], cards))
    if len(candidates) != 1:
        raise ValueError("frame geometry must identify one sealed master")
    return candidates[0]


def _square(panel, color):
    pts = np.argwhere(panel == color)
    if not len(pts):
        return None
    y, x = pts.min(0).tolist()
    yy, xx = (pts.max(0)+1).tolist()
    u = yy-y
    if u != xx-x or len(pts) != u*u or not 1 <= u <= 3:
        return None
    if y % u or x % u or panel.shape[0] % u or panel.shape[1] % u:
        return None
    return y//u, x//u, u


def _decode(value, *, reflect=True):
    grid = _grid(value)
    bg = int(grid[0, 0])
    frame, halo, ref, cards = _frames(grid, bg)
    panels = [grid[c["y"]:c["y"]+c["h"], c["x"]:c["x"]+c["w"]] for c in cards]
    common = set.intersection(*(set(map(int, np.unique(p))) for p in panels)) - {bg, frame, halo}
    anchors = [(color, [_square(p, color) for p in panels]) for color in sorted(common)]
    anchors = [(color, squares) for color, squares in anchors if all(s is not None for s in squares)]
    if len(anchors) != 1:
        raise ValueError("anchor must be the unique common square color")
    anchor, squares = anchors[0]
    for c, p, (ay, ax, u) in zip(cards, panels, squares):
        lh, lw = c["h"]//u, c["w"]//u
        if min(lh, lw) < 3:
            raise ValueError("interior must have at least three logical rows/columns")
        blocks = p.reshape(lh, u, lw, u)
        if not np.array_equal(blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))):
            raise ValueError("all observations must be whole logical cells")
        tiles = blocks[:, 0, :, 0].copy()
        sy, sx = _signs(c["corner"] if reflect else cards[ref]["corner"])
        c.update(ay=ay, ax=ax, unit=u, lh=lh, lw=lw, sy=sy, sx=sx, tiles=tiles)
        c["observed"] = {(sy*(r-ay), sx*(s-ax)): int(tiles[r, s])
                         for r in range(lh) for s in range(lw)}
        if set(map(int, np.unique(tiles))) & {frame, halo}:
            raise ValueError("frame and seal roles cannot occur inside specimens")
    master = {q: color for q, color in cards[ref]["observed"].items() if color != bg}
    if master.get((0, 0)) != anchor:
        raise ValueError("master lacks its anchor")
    materials = set(master.values()) - {anchor}
    if len(materials) < 2:
        raise ValueError("master must have at least two material colors")
    missing, extras = [], []
    for i, c in enumerate(cards):
        values = set(c["observed"].values())
        if i != ref:
            if bg not in values or len(values-{bg, anchor}) != 1 or not values-{bg, anchor} <= materials:
                raise ValueError("replicas must contain one visible binary material color")
        missing.append(set() if i == ref else {q for q, col in c["observed"].items()
                                               if q in master and q != (0, 0) and col == bg})
        extras.append(set() if i == ref else {q for q, col in c["observed"].items()
                                              if q not in master and col != bg})
    faults = set().union(*missing)
    return {"grid": grid, "bg": bg, "frame": frame, "halo": halo, "anchor": anchor,
            "ref": ref, "cards": cards, "master": master, "materials": materials,
            "missing": missing, "extras": extras, "faults": faults}


def _neighborhood(points):
    return {(r+dr, c+dc) for r, c in points for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if dr or dc} - set(points)


def _predict(data, *, local=False, intact=False, keep_extras=False, binary=False):
    out = data["grid"].copy()
    for i, c in enumerate(data["cards"]):
        faults = set() if intact else (data["missing"][i] if local else data["faults"])
        ring = _neighborhood(faults)
        u = c["unit"]
        binary_color = next(iter(set(c["observed"].values())-{data["bg"], data["anchor"]})) if i != data["ref"] else None
        for r in range(c["lh"]):
            for s in range(c["lw"]):
                q = (c["sy"]*(r-c["ay"]), c["sx"]*(s-c["ax"]))
                color = data["master"].get(q, data["bg"])
                if binary and color in data["materials"] and binary_color is not None:
                    color = binary_color
                if keep_extras and q in data["extras"][i]:
                    color = c["observed"][q]
                if q in ring:
                    color = data["halo"]
                if q in faults:
                    color = data["bg"]
                if q == (0, 0):
                    color = data["anchor"]
                y, x = c["y"]+r*u, c["x"]+s*u
                out[y:y+u, x:x+u] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode all conventions and shared faults from this input alone."""
    return _predict(_decode(input_grid))


def _pixel_ring_ablation(data):
    """The plausible but wrong radius-one-pixel interpretation."""
    out = _predict(data, intact=True)
    for c in data["cards"]:
        u = c["unit"]
        missing = np.zeros((c["h"], c["w"]), dtype=bool)
        for r in range(c["lh"]):
            for s in range(c["lw"]):
                q = (c["sy"]*(r-c["ay"]), c["sx"]*(s-c["ax"]))
                if q in data["faults"]:
                    missing[r*u:(r+1)*u, s*u:(s+1)*u] = True
        padded = np.pad(missing, 1)
        ring = np.logical_or.reduce([padded[dy:dy+c["h"], dx:dx+c["w"]]
                                     for dy in range(3) for dx in range(3)]) & ~missing
        part = out[c["y"]:c["y"]+c["h"], c["x"]:c["x"]+c["w"]]
        part[ring] = data["halo"]
        part[missing] = data["bg"]
        ay, ax = c["ay"]*u, c["ax"]*u
        part[ay:ay+u, ax:ax+u] = data["anchor"]
    return out


def _features(data):
    faults, master, cards = data["faults"], data["master"], data["cards"]
    ring = _neighborhood(faults)
    crop = broadcast = clipped = near = far = overlap = False
    crop_count = private_count = 0
    surviving_colors = set(master[q] for q in set(master)-faults-ring-{(0, 0)})
    transferred_colors = 0
    for q in faults:
        owners = [i for i in range(len(cards)) if q in data["missing"][i]]
        if len(owners) == 1 and any(i != data["ref"] and i != owners[0] and q in c["observed"]
                                  for i, c in enumerate(cards)):
            private_count += 1
    coverage = {}
    for q in faults:
        for n in _neighborhood({q}):
            coverage[n] = coverage.get(n, 0)+1
    for i, c in enumerate(cards):
        visible = set(c["observed"])
        crop_count += bool(set(master)-visible)
        crop |= bool(set(master)-visible)
        if i != data["ref"]:
            broadcast |= any(q in visible and q not in data["missing"][i] for q in faults)
            transferred_colors = max(transferred_colors,
                                     len({master[q] for q in visible & set(master)-faults-ring-{(0, 0)}}))
        clipped |= bool(ring-visible)
        near |= bool(data["extras"][i] & ring)
        far |= bool(data["extras"][i] - ring)
        overlap |= any(coverage.get(q, 0) > 1 for q in ring & visible - {(0, 0)})
    orthogonal = any((y+1, x) in faults or (y, x+1) in faults for y, x in faults)
    diagonal = any((y+1, x+1) in faults or (y+1, x-1) in faults for y, x in faults)
    coords = sorted(master)
    ys, xs = zip(*coords)
    normalized = tuple((y-min(ys), x-min(xs), q == (0, 0)) for q in coords for y, x in [q])
    return {"cards": len(cards), "units": tuple(sorted(c["unit"] for c in cards)),
            "corners": tuple(c["corner"] for c in cards), "master_corner": cards[data["ref"]]["corner"],
            "fault_count": len(faults), "extra_count": sum(map(len, data["extras"])),
            "crop": crop, "broadcast": broadcast, "clipped": clipped, "near": near, "far": far,
            "crop_count": crop_count, "private_count": private_count,
            "surviving_colors": len(surviving_colors), "transferred_colors": transferred_colors,
            "orthogonal": orthogonal, "diagonal": diagonal, "overlap": overlap,
            "scaled": len({c["unit"] for c in cards}) > 1,
            "motif": normalized, "span": (max(ys)-min(ys)+1, max(xs)-min(xs)+1),
            "palette": (data["bg"], data["frame"], data["anchor"], data["halo"])}


def _motif(rng, style, anchor, materials):
    h, w = int(rng.integers(4, 7)), int(rng.integers(4, 8))
    occupied = set()
    if style == 0:
        # Unequal L arms, with independently chosen thicknesses.
        a, b = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        occupied = {(r, c) for r in range(h) for c in range(w) if r < a or c < b}
    elif style == 1:
        trunk = int(rng.integers(1, w-1))
        occupied = {(r, trunk) for r in range(h)}
        levels = sorted(map(int, rng.choice(h, size=int(rng.integers(2, h+1)), replace=False)))
        for level in levels:
            lo, hi = int(rng.integers(0, trunk+1)), int(rng.integers(trunk+1, w+1))
            occupied.update((level, c) for c in range(lo, hi))
    elif style == 2:
        occupied = {(r, c) for r in range(h) for c in range(w)
                    if r in (0, h-1) or c in (0, w-1)}
        if rng.random() < .5:
            row = int(rng.integers(1, h-1))
            occupied.update((row, c) for c in range(int(rng.integers(2, w))))
    elif style == 3:
        occupied = {(r, w//2) for r in range(h)}
        for _ in range(int(rng.integers(2, 5))):
            r0, r1 = sorted(map(int, rng.choice(h+1, size=2, replace=False)))
            c0 = int(rng.integers(0, w//2+1))
            c1 = int(rng.integers(w//2+1, w+1))
            occupied.update((r, c) for r in range(r0, r1) for c in range(c0, c1))
    else:
        occupied = {(h//2, w//2)}
        wanted = int(rng.integers(9, min(h*w, 23)+1))
        while len(occupied) < wanted:
            frontier = sorted({(r+dr, c+dc) for r, c in occupied
                               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                               if 0 <= r+dr < h and 0 <= c+dc < w} - occupied)
            occupied.add(frontier[int(rng.integers(len(frontier)))])
    if len(occupied) < 9:
        return None
    # Remove unused extent, then optionally retain one logical field margin.
    yy, xx = zip(*occupied)
    occupied = {(r-min(yy), c-min(xx)) for r, c in occupied}
    h, w = max(yy)-min(yy)+1, max(xx)-min(xx)+1
    py, px = int(rng.integers(2)), int(rng.integers(2))
    if style in (1, 2):
        py = px = 0  # a fault on a long side can exhibit frame clipping
    occupied = {(r+py, c+px) for r, c in occupied}
    h += py + int(rng.integers(2))
    w += px + int(rng.integers(2))
    ay, ax = sorted(occupied)[int(rng.integers(len(occupied)))]
    motif = {(0, 0): anchor}
    points = sorted(occupied - {(ay, ax)})
    # Colored stripes or bands give region structure, plus occasional recolored branches.
    axis = int(rng.integers(2))
    phase = int(rng.integers(len(materials)))
    for r, c in points:
        color = materials[((r if axis == 0 else c)+phase) % len(materials)]
        motif[r-ay, c-ax] = color
    # Guarantee every requested material color occurs without a hidden palette role.
    for q, color in zip(rng.permutation(len(points)), materials):
        r, c = points[int(q)]
        motif[r-ay, c-ax] = color
    # A source symmetric about its anchor would weaken the orientation evidence.
    if any({(sy*r, sx*c): col for (r, c), col in motif.items()} == motif
           for sy, sx in ((-1, 1), (1, -1), (-1, -1))):
        return None
    return motif, h, w, ay, ax


def _pack(rng, cards):
    sizes = [(c["lh"]*c["unit"]+2, c["lw"]*c["unit"]+2) for c in cards]
    n, options = len(cards), []
    # Search all small ragged-shelf arrangements; no fixed panel/cross layout.
    orders = list(permutations(range(n)))
    rng.shuffle(orders)
    for order in orders:
        for cuts in range(1 << (n-1)):
            rows, current = [], []
            for j, i in enumerate(order):
                current.append(i)
                if j == n-1 or cuts & (1 << j):
                    rows.append(current)
                    current = []
            height = sum(max(sizes[i][0] for i in row) for row in rows)+len(rows)-1
            width = max(sum(sizes[i][1] for i in row)+len(row)-1 for row in rows)
            if height <= 28 and width <= 28:
                options.append((rows, height, width))
        if len(options) > 30:
            break
    if not options:
        return None
    rows, minimum_h, minimum_w = options[int(rng.integers(len(options)))]
    height = int(rng.integers(minimum_h+2, min(30, minimum_h+6)+1))
    width = int(rng.integers(minimum_w+2, min(30, minimum_w+6)+1))
    vertical_slack = height-2-minimum_h
    # Distribute spare cells among margins and inter-shelf gaps.
    gaps = [0]*(len(rows)+1)
    for _ in range(vertical_slack):
        gaps[int(rng.integers(len(gaps)))] += 1
    top = 1+gaps[0]
    for j, row in enumerate(rows):
        row_h = max(sizes[i][0] for i in row)
        row_w = sum(sizes[i][1] for i in row)+len(row)-1
        spare = width-2-row_w
        hgap = [0]*(len(row)+1)
        for _ in range(spare):
            hgap[int(rng.integers(len(hgap)))] += 1
        left = 1+hgap[0]
        for k, i in enumerate(row):
            hh, ww = sizes[i]
            y = top+int(rng.integers(row_h-hh+1))
            cards[i].update(y=y+1, x=left+1, h=hh-2, w=ww-2)
            left += ww+1+hgap[k+1]
        top += row_h+1+gaps[j+1]
    return height, width


def _visible(c):
    sy, sx = _signs(c["corner"])
    return {(sy*(r-c["ay"]), sx*(s-c["ax"])) for r in range(c["lh"]) for s in range(c["lw"])}


def _paint_cell(array, c, q, color):
    sy, sx = _signs(c["corner"])
    r, s = c["ay"]+sy*q[0], c["ax"]+sx*q[1]
    if 0 <= r < c["lh"] and 0 <= s < c["lw"]:
        u = c["unit"]
        y, x = c["y"]+r*u, c["x"]+s*u
        array[y:y+u, x:x+u] = color


def _build(rng, style, ref_corner, witness, advanced):
    palette = list(map(int, rng.permutation(10)))
    bg, frame, anchor, halo = palette[:4]
    materials = palette[4:4+int(rng.integers(2, 5))]
    source = _motif(rng, style, anchor, materials)
    if source is None:
        return None
    motif, lh, lw, ay, ax = source
    n = 4 if witness == 1 else int(rng.integers(4, 6)) if witness == 2 else int(rng.integers(3, 6))
    units = list(map(int, rng.choice([1, 1, 1, 2, 2, 3], size=n)))
    if witness:
        units[0] = int(rng.choice([1, 2]))
        units[1] = 2 if units[0] == 1 else 1
    corners = [ref_corner]
    if witness == 1:
        corners += [c for c in range(4) if c != ref_corner]
    else:
        corners += list(map(int, rng.integers(4, size=n-1)))
    cards = []
    for i in range(n):
        corner = corners[i]
        sy, sx = _signs(corner)
        hy, wx = lh, lw
        yy = ay if sy == 1 else lh-1-ay
        xx = ax if sx == 1 else lw-1-ax
        if i and advanced and (not witness or (witness == 2 and i == 3)):
            hy = max(3, lh+int(rng.integers(-2, 2)))
            wx = max(3, lw+int(rng.integers(-2, 2)))
            yy = min(hy-1, max(0, yy+int(rng.integers(-1, 2))))
            xx = min(wx-1, max(0, xx+int(rng.integers(-1, 2))))
        c = {"lh": hy, "lw": wx, "ay": yy, "ax": xx, "unit": units[i], "corner": corner}
        c["visible"] = _visible(c)
        if len((set(motif)-{(0, 0)}) & c["visible"]) < 4:
            return None
        cards.append(c)
    shape = _pack(rng, cards)
    if shape is None:
        return None
    common = sorted((set(motif)-{(0, 0)}) & cards[1]["visible"] & cards[2]["visible"])
    if len(common) < 3:
        return None
    selected = set()
    if witness == 2:
        triples = [((y, x), (y+1, x), (y+1, x+1)) for y, x in common
                   if (y+1, x) in common and (y+1, x+1) in common]
        triples += [((y, x), (y, x+1), (y+1, x+1)) for y, x in common
                    if (y, x+1) in common and (y+1, x+1) in common]
        if not triples:
            return None
        selected.update(triples[int(rng.integers(len(triples)))])
        edge = [q for q in common if ay+q[0] in (0, lh-1) or ax+q[1] in (0, lw-1)]
        if edge:
            selected.add(edge[int(rng.integers(len(edge)))])
    budget = (int(rng.integers(2, 4)) if witness == 1 else
              int(rng.integers(3, min(6, max(3, (len(motif)-1)//4))+1)))
    pool = common if witness else sorted({q for c in cards[1:] for q in c["visible"] if q in motif and q != (0, 0)})
    rng.shuffle(pool)
    selected.update(pool[:max(0, min(budget, len(pool)-2)-len(selected))])
    if len(selected) < 2:
        return None
    faults = [set() for _ in cards]
    ordered = sorted(selected)
    rng.shuffle(ordered)
    # Two common coordinates have exclusive owners, visibly teaching broadcast.
    private = [q for q in ordered if q in common][:2]
    if len(private) < 2:
        return None
    faults[1].add(private[0])
    faults[2].add(private[1])
    for q in ordered:
        if q in private:
            continue
        owners = [i for i in range(1, n) if q in cards[i]["visible"]]
        if not owners:
            return None
        rng.shuffle(owners)
        for i in owners[:2 if rng.random() < .25 else 1]:
            faults[i].add(q)
    if any(len((set(motif)-{(0, 0)}) & c["visible"] - faults[i]) < 2 for i, c in enumerate(cards[1:], 1)):
        return None
    all_faults = set().union(*faults)
    ring = _neighborhood(all_faults)
    extra = [set() for _ in cards]
    for i, c in enumerate(cards[1:], 1):
        empty = sorted(c["visible"] - set(motif))
        rng.shuffle(empty)
        extra[i].update(empty[:int(rng.integers(0, min(4, len(empty))+1))])
    requirement = (lambda q: q not in ring) if witness == 1 else (lambda q: q in ring)
    if witness:
        choices = [(i, q) for i, c in enumerate(cards[1:], 1)
                   for q in sorted(c["visible"]-set(motif)) if requirement(q)]
        if not choices:
            return None
        i, q = choices[int(rng.integers(len(choices)))]
        extra[i].add(q)
    inp = np.full(shape, bg, dtype=np.int64)
    for i, c in enumerate(cards):
        y, x, h, w = c["y"]-1, c["x"]-1, c["h"]+2, c["w"]+2
        inp[y:y+h, x] = inp[y:y+h, x+w-1] = frame
        inp[y, x:x+w] = inp[y+h-1, x:x+w] = frame
        inp[_corners(y, x, h, w)[c["corner"]]] = halo if i == 0 else bg
        material = int(rng.choice(materials))
        for q, color in motif.items():
            _paint_cell(inp, c, q, color if i == 0 or q == (0, 0) else material)
        for q in faults[i]:
            _paint_cell(inp, c, q, bg)
        for q in extra[i]:
            _paint_cell(inp, c, q, material)
    # Construction path paints from explicit clean geometry, then diagnostics.
    # It does not decode the input, invoke solve(), or use _predict().
    out = inp.copy()
    for c in cards:
        out[c["y"]:c["y"]+c["h"], c["x"]:c["x"]+c["w"]] = bg
        for q, color in motif.items():
            _paint_cell(out, c, q, color)
        for y, x in all_faults:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy or dx:
                        _paint_cell(out, c, (y+dy, x+dx), halo)
        for q in all_faults:
            _paint_cell(out, c, q, bg)
        _paint_cell(out, c, (0, 0), anchor)
    return {"input": inp, "output": out}


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.uint8).tobytes()


def _pair_analysis(pair):
    data = _decode(pair["input"])
    expected = _predict(data)
    if not np.array_equal(expected, pair["output"]):
        raise AssertionError("incorrect registered shared-fault output")
    if not data["faults"]:
        raise AssertionError("at least one visible missing cell is required")
    remaining = set(data["master"])
    stack = [remaining.pop()]
    while stack:
        y, x = stack.pop()
        for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if q in remaining:
                remaining.remove(q)
                stack.append(q)
    assert not remaining, "master material must be connected to its anchor"
    return data, _features(data)


def _unobserved_ablation(data):
    altered = dict(data)
    altered["faults"] = data["faults"] | set().union(*(set(data["master"])-set(c["observed"])
                                                        for c in data["cards"]))
    return _predict(altered)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    first_corner = int(rng.integers(4))
    second_corner = (first_corner+int(rng.integers(1, 4))) % 4
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = index+1 if split == "train" and index < 2 else 0
            corner = first_corner if witness == 1 else second_corner if witness == 2 else int(rng.integers(4))
            for attempt in range(512):
                style = 0 if witness == 1 else int(rng.choice([1, 2, 3])) if witness == 2 else int(rng.integers(1, 5))
                pair = _build(rng, style, corner, witness,
                              split == "test" or witness == 2 or (not witness and rng.random() < .6))
                if pair is None or _key(pair["input"]) in seen:
                    continue
                try:
                    data, f = _pair_analysis(pair)
                except (ValueError, AssertionError):
                    continue
                if f["surviving_colors"] < 2 or f["transferred_colors"] < 2:
                    continue
                if witness:
                    if not (f["private_count"] >= 2 and f["scaled"] and f["extra_count"]):
                        continue
                    if np.array_equal(_pixel_ring_ablation(data), pair["output"]):
                        continue
                    if np.array_equal(_predict(_decode(pair["input"], reflect=False)), pair["output"]):
                        continue
                    if witness == 1 and not f["far"]:
                        continue
                    if witness == 2 and not all(f[k] for k in ("near", "clipped", "orthogonal", "diagonal", "overlap")):
                        continue
                    if witness == 2 and (not f["crop"] or np.array_equal(_unobserved_ablation(data), pair["output"])):
                        continue
                if split == "test" and (f["crop_count"] < 2 or np.array_equal(_unobserved_ablation(data), pair["output"])):
                    continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded sampling could not produce a readable teaching case")
    # The required examples are not always first in the delivered puzzle.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records, data_train = set(), {"train": [], "test": []}, []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            data, f = _pair_analysis({"input": inp, "output": out})
            assert f["surviving_colors"] >= 2 and f["transferred_colors"] >= 2, "colored material must survive the halos"
            records[split].append(f)
            if split == "train":
                data_train.append((data, out))
            # Frames/exterior/seal and all anchor pixels must be unchanged.
            protected = np.ones(inp.shape, dtype=bool)
            for c in data["cards"]:
                protected[c["y"]:c["y"]+c["h"], c["x"]:c["x"]+c["w"]] = False
                u = c["unit"]
                y, x = c["y"]+c["ay"]*u, c["x"]+c["ax"]*u
                assert np.all(out[y:y+u, x:x+u] == data["anchor"])
            assert np.array_equal(inp[protected], out[protected])
    training = records["train"]
    assert sum(f["private_count"] >= 2 for f in training) >= 2, "two demonstrations must show private faults travelling"
    assert any(f["far"] for f in training), "teach erasure away from halos"
    assert any(f["scaled"] for f in training), "teach scale correspondence"
    assert len({f["master_corner"] for f in training}) >= 2, "teach relative orientation"
    assert set().union(*(set(f["corners"]) for f in training)) == set(range(4))
    for feature in ("near", "clipped", "orthogonal", "diagonal", "overlap"):
        assert any(f[feature] for f in training), f"training lacks {feature}"
    # Directly challenge useful, simpler alternative rules against demonstrations.
    for mode in ("intact", "local", "keep_extras", "binary"):
        assert any(not np.array_equal(_predict(d, **{mode: True}), out) for d, out in data_train), mode
    assert any(not np.array_equal(_pixel_ring_ablation(d), out) for d, out in data_train)
    assert any(not np.array_equal(_unobserved_ablation(d), out) for d, out in data_train), "teach unobserved versus missing"
    assert any(not np.array_equal(_predict(_decode(d["grid"], reflect=False)), out)
               for d, out in data_train), "reflection omission must change an answer"
    assert all(f["crop_count"] >= 2 for f in records["test"]), "test requires multi-replica crops"


def _puzzle_key(p):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"])) for pair in p[s]) for s in ("train", "test"))


def self_test() -> dict:
    """Bounded authored checks and measured diversity over 42 seeds."""
    signatures, shapes, motifs, palettes, counts, corners, units, fault_counts, extra_counts = (set() for _ in range(9))
    crop = overlap = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_puzzle_key(p))
        if seed < 3:
            assert _puzzle_key(p) == _puzzle_key(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in p[split]:
                d, f = _pair_analysis(pair)
                shapes.add(d["grid"].shape)
                motifs.add(f["motif"])
                palettes.add(f["palette"])
                counts.add(f["cards"])
                corners.update(f["corners"])
                units.update(f["units"])
                fault_counts.add(f["fault_count"])
                extra_counts.add(f["extra_count"])
                crop += f["crop"]
                overlap += f["overlap"]
        # Coordinate conventions must work after rotations and reflections too.
        if seed < 4:
            pair = p["test"][0]
            for k in range(4):
                a, b = np.rot90(pair["input"], k), np.rot90(pair["output"], k)
                assert np.array_equal(solve(a), b)
                assert np.array_equal(solve(a.T), b.T)
    assert len(signatures) == 42 and len(shapes) >= 35 and len(motifs) >= 100
    assert len(palettes) >= 100 and counts == {3, 4, 5} and corners == {0, 1, 2, 3}
    assert units == {1, 2, 3} and len(fault_counts) >= 4 and len(extra_counts) >= 7
    assert crop >= 42 and overlap >= 42
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(101, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(777, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a damaged output")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "motifs": len(motifs), "palettes": len(palettes), "specimen_counts": sorted(counts),
            "units": sorted(units), "fault_counts": sorted(fault_counts),
            "extra_counts": sorted(extra_counts), "cropped_pairs": crop, "overlap_pairs": overlap}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
