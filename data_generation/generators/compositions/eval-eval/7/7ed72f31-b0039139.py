"""Counted symmetry relays: a local hinge turns unary tokens into a ribbon.

Only NumPy and the Python standard library are used. No files are read or written
by this module. Generation labels use tile placement; solve uses successive
coordinate reflections. Background 0 and hinge 5 are fixed family conventions.
"""
from __future__ import annotations

import hashlib
import numpy as np


BACKGROUND = 0
HINGE = 5
INKS = (1, 2, 3, 4, 6, 7, 8, 9)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid bounds")
    _require(np.issubdtype(g.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "illegal color")
    return g.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    steps = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if diagonal:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    groups = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        stack, group = [p], [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
                    group.append(q)
        groups.append(sorted(group))
    return groups


def _bounds(cells):
    a = np.asarray(cells, dtype=int)
    return tuple(map(int, a.min(axis=0))), tuple(map(int, a.max(axis=0)))


def _trim(mask):
    lo, hi = _bounds(np.argwhere(mask))
    return mask[lo[0]:hi[0]+1, lo[1]:hi[1]+1].copy()


def _distance(first, second):
    return min(max(abs(r-a), abs(c-b)) for r, c in first for a, b in second)


def _key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def _reflect(cells, kind, anchor):
    ar, ac = anchor
    if kind == "H":
        return [(2*ar-r, c) for r, c in cells]
    if kind == "V":
        return [(r, 2*ac-c) for r, c in cells]
    return [(2*ar-r, 2*ac-c) for r, c in cells]


def _marker(kind, direction, shape, span):
    h, w = shape
    dr, dc = direction
    if kind == "H":
        return [(h if dr > 0 else -1, c) for c in range(span[0], span[1]+1)]
    if kind == "V":
        return [(r, w if dc > 0 else -1) for r in range(span[0], span[1]+1)]
    return [(h if dr > 0 else -1, w if dc > 0 else -1)]


def _pitch(kind, direction, shape):
    h, w = shape
    dr, dc = direction
    return dr*(h+1), dc*(w+1)


def _decode(input_grid):
    """One geometric parse, with no training pairs or generation metadata."""
    g = _grid(input_grid)
    gray = _components(g == HINGE, diagonal=True)
    _require(len(gray) == 1, "exactly one gray hinge is required")
    marker = gray[0]
    low, high = _bounds(marker)
    if len(marker) == 1:
        kind = "P"
    elif low[0] == high[0] and len(marker) == high[1]-low[1]+1:
        kind = "H"
    elif low[1] == high[1] and len(marker) == high[0]-low[0]+1:
        kind = "V"
    else:
        raise AssertionError("hinge must be a point or a straight segment")
    colors = set(map(int, np.unique(g))) - {BACKGROUND, HINGE}
    _require(len(colors) == 2, "one stencil ink and one token ink are required")
    contacts = []
    for color in sorted(colors):
        pieces = _components(g == color, diagonal=True)
        for piece in pieces:
            if _distance(piece, marker) == 1:
                contacts.append((color, piece, len(pieces)))
    _require(len(contacts) == 1, "hinge ownership is ambiguous or missing")
    ink, payload, pieces = contacts[0]
    _require(pieces == 1, "all stencil ink must form one eight-connected shape")
    fill = next(iter(colors - {ink}))
    tokens4 = _components(g == fill)
    tokens8 = _components(g == fill, diagonal=True)
    _require(len(tokens4) == len(tokens8) and tokens4, "ambiguous token connectivity")
    for token in tokens8:
        _require(_distance(token, marker+payload) >= 2, "token touches stencil or hinge")
    (r0, c0), (r1, c1) = _bounds(payload)
    h, w = r1-r0+1, c1-c0+1
    ar, ac = marker[0]
    if kind == "H":
        _require(ar in (r0-1, r1+1), "horizontal hinge must border the stencil box")
        direction = (1 if ar == r1+1 else -1, 0)
        span = (low[1]-c0, high[1]-c0)
    elif kind == "V":
        _require(ac in (c0-1, c1+1), "vertical hinge must border the stencil box")
        direction = (0, 1 if ac == c1+1 else -1)
        span = (low[0]-r0, high[0]-r0)
    else:
        _require(ar in (r0-1, r1+1) and ac in (c0-1, c1+1),
                 "point hinge must be one cell diagonally beyond a box corner")
        direction = (1 if ar == r1+1 else -1, 1 if ac == c1+1 else -1)
        span = (0, 0)
    tile = (g == ink)[r0:r1+1, c0:c1+1].copy()
    n = len(tokens4)
    # The relay may grow beyond the input canvas, but its cropped output fits ARC.
    if kind in ("H", "P"):
        _require(n*(h+1)-1 <= 30, "vertical relay exceeds output budget")
    if kind in ("V", "P"):
        _require(n*(w+1)-1 <= 30, "horizontal relay exceeds output budget")
    if n > 1 and kind in ("H", "V"):
        cross = w if kind == "H" else h
        _require(max(cross-1, span[1])-min(0, span[0])+1 <= 30,
                 "marker overhang exceeds output budget")
    return {"kind": kind, "direction": direction, "span": span, "tile": tile,
            "count": n, "ink": ink, "fill": fill, "tokens": tokens4,
            "origin": (r0, c0), "marker": marker, "payload": payload,
            "input_shape": g.shape}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reflect the previous rectangle across each advancing hinge in turn."""
    d = _decode(input_grid)
    kind, n, tile = d["kind"], d["count"], d["tile"]
    h, w = tile.shape
    current = [tuple(map(int, p)) for p in np.argwhere(tile)]
    # Carry all four rectangle corners: blank stencil cells are part of the tile.
    corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
    colored = list(current)
    frame = list(corners)
    gray = []
    marker = _marker(kind, d["direction"], tile.shape, d["span"])
    pr, pc = _pitch(kind, d["direction"], tile.shape)
    for step in range(n-1):
        hinge = [(r+step*pr, c+step*pc) for r, c in marker]
        current = _reflect(current, kind, hinge[0])
        corners = _reflect(corners, kind, hinge[0])
        colored.extend(current)
        frame.extend(corners)
        gray.extend(hinge)
    (r0, c0), (r1, c1) = _bounds(frame+gray)
    out = np.full((r1-r0+1, c1-c0+1), d["fill"], dtype=np.int8)
    for r, c in colored:
        out[r-r0, c-c0] = d["ink"]
    for r, c in gray:
        out[r-r0, c-c0] = HINGE
    return out


def _place_tiles(tile, kind, direction, span, n, ink, fill, alternate=True):
    """Label construction by parity and translated tile boxes, not by solve."""
    h, w = tile.shape
    pr, pc = _pitch(kind, direction, tile.shape)
    marker = _marker(kind, direction, tile.shape, span)
    origins = [(i*pr, i*pc) for i in range(n)]
    hinges = [(r+i*pr, c+i*pc) for i in range(n-1) for r, c in marker]
    box = [(r+dr, c+dc) for r, c in origins for dr, dc in ((0, 0), (h-1, w-1))]
    (r0, c0), (r1, c1) = _bounds(box+hinges)
    out = np.full((r1-r0+1, c1-c0+1), fill, dtype=np.int8)
    for i, (r, c) in enumerate(origins):
        mask = tile
        if alternate and i % 2:
            mask = tile[::-1] if kind == "H" else tile[:, ::-1] if kind == "V" else tile[::-1, ::-1]
        out[r-r0:r-r0+h, c-c0:c-c0+w] = np.where(mask, ink, fill)
    for r, c in hinges:
        out[r-r0, c-c0] = HINGE
    return out


def _asymmetric(mask):
    return all(not np.array_equal(mask, other) for other in
               (mask[::-1], mask[:, ::-1], mask[::-1, ::-1]))


def _silhouette(rng, h, w, witness=False):
    """Tight eight-connected shapes: erosion, walks, branches, rings and solids."""
    for _ in range(24):
        mode = int(rng.integers(6))
        a = np.zeros((h, w), bool)
        if mode == 0:
            a[:] = True
            # Erode random boundary/interior cells while keeping the bbox and
            # eight-connectivity. No prescribed opposite-corner path.
            target = int(rng.integers(max(1, (h*w)//3), h*w+1))
            for index in rng.permutation(h*w):
                if int(a.sum()) <= target:
                    break
                r, c = divmod(int(index), w)
                a[r, c] = False
                if not a.any() or _trim(a).shape != (h, w) or len(_components(a, True)) != 1:
                    a[r, c] = True
        elif mode == 1:
            # King-move walks retain diagonal bridges and sparse bends.
            r, c = int(rng.integers(h)), int(rng.integers(w))
            for _ in range(int(rng.integers(max(h, w), 3*h*w+1))):
                a[r, c] = True
                r = min(h-1, max(0, r+int(rng.integers(-1, 2))))
                c = min(w-1, max(0, c+int(rng.integers(-1, 2))))
        elif mode == 2:
            a[0, :] = a[-1, :] = True
            a[:, 0] = a[:, -1] = True
            if h > 2 and w > 2 and rng.random() < .65:
                a[1:int(rng.integers(2, h)), int(rng.integers(1, w-1))] = True
        elif mode == 3:
            # Thin diagonal zigzag, with optional short edge-connected twigs.
            if h < w:
                a = a.T
            hh, ww = a.shape
            c = int(rng.integers(ww))
            for r in range(hh):
                a[r, c] = True
                if ww > 1:
                    c = min(ww-1, max(0, c+int(rng.choice([-1, 1]))))
            if h < w:
                a = a.T
        elif mode == 4:
            a[:] = True
            if h > 1 and w > 1:
                a[:int(rng.integers(1, h)), :int(rng.integers(1, w))] = False
        else:
            a[:] = True
        if rng.random() < .5:
            a = a[::-1]
        if rng.random() < .5:
            a = a[:, ::-1]
        a = _trim(a)
        if not witness or (min(a.shape) >= 2 and _asymmetric(a)):
            return a.copy()
    # Guaranteed discriminating witness; ordinary symmetric objects remain valid.
    a = np.ones((max(2, h), max(2, w)), bool)
    a[int(rng.choice([0, a.shape[0]-1])), int(rng.choice([0, a.shape[1]-1]))] = False
    return a


def _sample_span(rng, tile, kind, direction):
    if kind == "P":
        return (0, 0)
    cross = tile.shape[1] if kind == "H" else tile.shape[0]
    edge = (tile[-1] if direction[0] > 0 else tile[0]) if kind == "H" else (
        tile[:, -1] if direction[1] > 0 else tile[:, 0])
    contact = int(rng.choice(np.flatnonzero(edge)))
    if rng.random() < .35:
        lo, hi = -int(rng.integers(3)), cross-1+int(rng.integers(3))
    else:
        lo = int(rng.integers(-2, contact+1))
        hi = int(rng.integers(max(lo+1, contact), max(lo+2, cross+2)))
    if hi == lo:
        hi += 1
    return lo, hi


def _token_mask(rng, area=None, compact=False):
    if area is None:
        if compact:
            return np.ones((1, 1), bool)
        mode = int(rng.integers(5))
        h, w = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        if mode == 0:
            return np.ones((1, int(rng.integers(1, 9))), bool)
        if mode == 1:
            return np.ones((int(rng.integers(1, 9)), 1), bool)
        if mode == 2:
            return np.ones((h, w), bool)
        if mode == 3:
            h, w = max(3, h), max(3, w)
            a = np.ones((h, w), bool)
            a[1:-1, 1:-1] = False
            return a
        area = int(rng.integers(1, h*w+1))
    if compact:
        return np.ones((1, area), bool)
    h = int(rng.integers(1, min(6, area)+1))
    w = max(1, (area+h-1)//h) + int(rng.integers(3))
    a = np.zeros((h, w), bool)
    frontier = {(int(rng.integers(h)), int(rng.integers(w)))}
    for _ in range(area):
        options = sorted(frontier)
        r, c = options[int(rng.integers(len(options)))]
        frontier.remove((r, c))
        a[r, c] = True
        for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= q[0] < h and 0 <= q[1] < w and not a[q]:
                frontier.add(q)
    return _trim(a)


def _areas(rng, n, total):
    if total is None:
        return [None]*n
    sizes = [1]*n
    for _ in range(total-n):
        sizes[int(rng.integers(n))] += 1
    return sizes


def _input_scene(rng, tile, kind, direction, span, n, ink, fill, total_area=None):
    """Free placement with one-cell halos; bounded compact fallback at 30x30."""
    marker = _marker(kind, direction, tile.shape, span)
    payload = list(map(tuple, np.argwhere(tile).tolist()))
    (r0, c0), (r1, c1) = _bounds(payload+marker)
    sh, sw = r1-r0+1, c1-c0+1
    payload = [(r-r0, c-c0) for r, c in payload]
    marker = [(r-r0, c-c0) for r, c in marker]
    sizes = _areas(rng, n, total_area)
    for attempt in range(10):
        compact = attempt == 9
        if compact:
            height = width = 30
            top = left = 0
        else:
            height = int(rng.integers(max(14, sh+2), 31))
            width = int(rng.integers(max(14, sw+2), 31))
            top, left = int(rng.integers(height-sh+1)), int(rng.integers(width-sw+1))
        g = np.zeros((height, width), np.int8)
        blocked = np.zeros(g.shape, bool)

        def paint(cells, color):
            for r, c in cells:
                g[r, c] = color
                blocked[max(0, r-1):min(height, r+2), max(0, c-1):min(width, c+2)] = True

        paint([(r+top, c+left) for r, c in payload], ink)
        paint([(r+top, c+left) for r, c in marker], HINGE)
        # Reserve the source box, including its cavities, against inserted tokens.
        blocked[max(0, top-1):min(height, top+sh+1), max(0, left-1):min(width, left+sw+1)] = True
        objects = [_token_mask(rng, area, compact) for area in sizes]
        objects.sort(key=lambda a: -a.size)
        okay = True
        for obj in objects:
            oh, ow = obj.shape
            sites = [(r, c) for r in range(height-oh+1) for c in range(width-ow+1)]
            if not compact:
                rng.shuffle(sites)
            accepted = False
            points = list(map(tuple, np.argwhere(obj).tolist()))
            for r, c in sites:
                moved = [(r+a, c+b) for a, b in points]
                if all(not blocked[q] for q in moved):
                    paint(moved, fill)
                    accepted = True
                    break
            if not accepted:
                okay = False
                break
        if okay:
            return g
    raise AssertionError("bounded scene packing exhausted")


def _direction(rng, kind):
    sign = lambda: int(rng.choice([-1, 1]))
    return (sign(), 0) if kind == "H" else (0, sign()) if kind == "V" else (sign(), sign())


def _pair(rng, kind, count, size, witness=False, total_area=None):
    h, w = size
    tile = _silhouette(rng, h, w, witness)
    direction = _direction(rng, kind)
    if kind == "P":
        # A diagonal point touches a foreground corner. Orient the mask so this
        # corner is occupied; retain irregularity elsewhere.
        rr = -1 if direction[0] > 0 else 0
        cc = -1 if direction[1] > 0 else 0
        if not tile[rr, cc]:
            # Connect the corner by a short king-move path to existing ink.
            r = tile.shape[0]-1 if rr == -1 else 0
            c = tile.shape[1]-1 if cc == -1 else 0
            q = min(map(tuple, np.argwhere(tile).tolist()), key=lambda p: max(abs(p[0]-r), abs(p[1]-c)))
            tile[r, c] = True
            while (r, c) != q:
                r += int(np.sign(q[0]-r))
                c += int(np.sign(q[1]-c))
                tile[r, c] = True
        if witness and not _asymmetric(tile):
            tile = np.ones((max(2, h), max(2, w)), bool)
            # Remove a corner other than the hinge-facing one.
            tile[0 if rr == -1 else -1, 0 if cc == -1 else -1] = False
    span = _sample_span(rng, tile, kind, direction)
    ink, fill = map(int, rng.choice(INKS, size=2, replace=False))
    inp = _input_scene(rng, tile, kind, direction, span, count, ink, fill, total_area)
    out = _place_tiles(tile, kind, direction, span, count, ink, fill)
    return {"input": inp, "output": out}


def _legal_integer(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Sample one rule with training witnesses and count/extent transfer tests."""
    nt = _legal_integer(num_train, "num_train", 2, 10)
    nq = _legal_integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    pools = [("H",), ("V",), ("P",), ("H", "V"), ("H", "P"), ("V", "P")]
    if nt >= 3:
        pools.append(("H", "V", "P"))
    kinds = pools[int(rng.integers(len(pools)))]
    plans = list(map(str, rng.permutation(kinds)))
    while len(plans) < nt:
        plans.append(str(rng.choice(kinds)))
    witness_area = int(rng.integers(12, 21))
    counts = list(map(int, rng.permutation([3, 4])))
    puzzle, seen = {"train": [], "test": []}, set()
    decoded_train = []
    for i, kind in enumerate(plans):
        witness = i < max(2, len(kinds))
        n = counts[i] if i < 2 else int(rng.integers(3, 5) if witness else rng.integers(1, 7))
        axial_max = min(7, 31//n-1)
        for _ in range(24):
            h = int(rng.integers(2 if witness else 1, (min(5, axial_max) if kind in ("H", "P") else 9)+1))
            w = int(rng.integers(2 if witness else 1, (min(5, axial_max) if kind in ("V", "P") else 9)+1))
            pair = _pair(rng, kind, n, (h, w), witness, witness_area if i < 2 else None)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise AssertionError("distinct training construction exhausted")
        puzzle["train"].append(pair)
        seen.add(_key(pair["input"]))
        decoded_train.append(_decode(pair["input"]))
    for _ in range(nq):
        kind = str(rng.choice(kinds))
        # Every test extrapolates either count or a tile dimension; size transfer
        # uses fewer tiles so large motifs need not coincide with long relays.
        count_shift = kind == "P" or rng.random() < .55
        if count_shift:
            n = int(rng.integers(max(d["count"] for d in decoded_train)+1,
                                 min(9, max(d["count"] for d in decoded_train)+3)+1))
            cap = 31//n-1
            h = int(rng.integers(2, cap+1)) if kind in ("H", "P") else int(rng.integers(2, 13))
            w = int(rng.integers(2, cap+1)) if kind in ("V", "P") else int(rng.integers(2, 13))
        else:
            n = int(rng.integers(2, 5))
            largest = max(d["tile"].shape[1 if kind == "H" else 0] for d in decoded_train)
            big = int(rng.integers(largest+1, min(15, largest+5)+1))
            axial = int(rng.integers(2, min(6, 31//n-1)+1))
            h, w = (axial, big) if kind == "H" else (big, axial)
        for _attempt in range(24):
            # For extent transfer, retain requested bbox through an asymmetric
            # eroded/full silhouette; ordinary walks can crop to a smaller box.
            pair = _pair(rng, kind, n, (h, w), witness=not count_shift)
            d = _decode(pair["input"])
            extent_okay = count_shift or d["tile"].shape[1 if kind == "H" else 0] > largest
            if _key(pair["input"]) not in seen and extent_okay:
                break
        else:
            raise AssertionError("distinct shifted test construction exhausted")
        puzzle["test"].append(pair)
        seen.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Reconstruct visible roles; check topology, recurrence and teaching evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, train, tests = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            predicted = solve(inp)
            _require(np.array_equal(predicted, out), "incorrect relay")
            n, tile = d["count"], d["tile"]
            _require(int(np.count_nonzero(out == d["ink"])) == n*int(tile.sum()), "lost/duplicated tile ink")
            _require(int(np.count_nonzero(out == HINGE)) == (n-1)*len(d["marker"]), "wrong number of hinges")
            _require(set(map(int, np.unique(out))) <= {d["ink"], d["fill"], HINGE}, "unexplained output colors")
            # Independently compose arrays to catch sign, rotation and crop errors.
            placed = _place_tiles(tile, d["kind"], d["direction"], d["span"], n, d["ink"], d["fill"])
            _require(np.array_equal(out, placed), "tile placement disagrees with coordinate recurrence")
            (train if split == "train" else tests).append(d)
    _require(len({d["count"] for d in train}) >= 2, "training must vary counts")
    _require(any(sum(map(len, a["tokens"])) == sum(map(len, b["tokens"])) and a["count"] != b["count"]
                 for a in train for b in train), "missing equal-area/different-count witness")
    active = {d["kind"] for d in train}
    _require({d["kind"] for d in tests} <= active, "untrained hinge kind")
    for kind in active:
        _require(any(d["kind"] == kind and d["count"] >= 3 and _asymmetric(d["tile"]) for d in train),
                 "missing asymmetric multi-step witness for " + kind)
        witness = next(d for d in train if d["kind"] == kind and d["count"] >= 3 and _asymmetric(d["tile"]))
        args = (witness["tile"], kind, witness["direction"], witness["span"],
                witness["count"], witness["ink"], witness["fill"])
        _require(not np.array_equal(_place_tiles(*args), _place_tiles(*args, alternate=False)),
                 "ignoring symmetry does not change witness")
    _require(any(sum(map(len, d["tokens"])) != d["count"] for d in train), "pixel-count shortcut survives")
    _require(any(d["input_shape"] != d["tile"].shape and not d["tile"].all() for d in train), "missing crop/recolor witness")
    max_count = max(d["count"] for d in train)
    for d in tests:
        grows = d["count"] > max_count
        axis = 1 if d["kind"] == "H" else 0
        grows |= d["tile"].shape[axis] > max(t["tile"].shape[axis] for t in train)
        _require(grows, "test lacks count or extent transfer")


def _digest(puzzle):
    result = hashlib.sha256()
    for split in ("train", "test"):
        result.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], np.int8)
                result.update(bytes(g.shape))
                result.update(g.tobytes())
    return result.hexdigest()


def self_test():
    """42 seeds, streaming metrics, metamorphic checks and corruption rejection.

    At most three puzzles plus a few 30x30 arrays are retained. No disk output.
    The returned small dictionary is descriptive evidence, not independent review.
    """
    shapes, structures, masks, signatures, directions, palettes, counts = set(), set(), set(), set(), set(), set(), set()
    totals = {"seeds": 42, "pairs": 0, "line_overhangs": 0, "point_pairs": 0,
              "stencil_holes": 0, "diagonal_stencils": 0, "single_tile_pairs": 0,
              "counter_size_max": 0, "input_border_pairs": 0}
    sample = None
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_digest(p))
        if seed in (0, 41):
            _require(_digest(p) == _digest(generate_puzzle(seed)), "nondeterminism")
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                d = _decode(inp)
                tile, kind = d["tile"], d["kind"]
                totals["pairs"] += 1
                shapes.add((inp.shape, out.shape))
                masks.add(hashlib.sha256(bytes(tile.shape)+tile.tobytes()).hexdigest())
                roles = np.zeros(inp.shape, np.int8)
                roles[inp == HINGE] = 1
                roles[inp == d["ink"]] = 2
                roles[inp == d["fill"]] = 3
                structures.add(hashlib.sha256(bytes(inp.shape)+roles.tobytes()).hexdigest())
                directions.add((kind, d["direction"]))
                palettes.add((d["ink"], d["fill"]))
                counts.add(d["count"])
                totals["point_pairs"] += kind == "P"
                totals["single_tile_pairs"] += d["count"] == 1
                totals["diagonal_stencils"] += len(_components(tile)) > 1
                totals["stencil_holes"] += len(_components(~np.pad(tile, 1))) > 1
                totals["counter_size_max"] = max(totals["counter_size_max"], max(map(len, d["tokens"])))
                totals["input_border_pairs"] += any(np.any(edge != 0) for edge in (inp[0], inp[-1], inp[:, 0], inp[:, -1]))
                if kind != "P":
                    cross = tile.shape[1] if kind == "H" else tile.shape[0]
                    totals["line_overhangs"] += d["span"][0] > 0 or d["span"][1] < cross-1
                # Token shapes/areas and positions have no geometrical output role.
                # Replace each token by one of its cells, preserving count/color.
                thinner = inp.copy()
                thinner[thinner == d["fill"]] = 0
                for token in d["tokens"]:
                    thinner[token[0]] = d["fill"]
                _require(np.array_equal(solve(thinner), out), "token shape affected output")
                # Reassign source/token colors, retaining geometry and fixed roles.
                mapping = np.arange(10, dtype=np.int8)
                mapping[d["ink"]], mapping[d["fill"]] = d["fill"], d["ink"]
                _require(np.array_equal(solve(mapping[inp]), mapping[out]), "palette-role failure")
                if sample is None and d["count"] >= 3:
                    sample = {s: [{k: v.copy() for k, v in q.items()} for q in p[s]] for s in p}
    _require(len(signatures) == 42 and len(structures) == 210, "seed/structure diversity collapsed")
    _require(len(masks) >= 70 and len(shapes) >= 60 and len(palettes) >= 30, "geometry/palette diversity collapsed")
    _require(len(directions) == 8 and set(range(1, 8)) <= counts, "direction/count coverage")
    for key in ("line_overhangs", "point_pairs", "stencil_holes", "diagonal_stencils", "single_tile_pairs", "input_border_pairs"):
        _require(totals[key] > 0, "missing feature: " + key)
    _require(totals["counter_size_max"] >= 16, "only tiny counters generated")
    del p
    # All explicit legal count requests; illegal types/ranges must raise ValueError.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            q = generate_puzzle(1000+nt*7+nq, nt, nq)
            _require(len(q["train"]) == nt and len(q["test"]) == nq, "explicit counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Unusually large family members exercise the input-only solver beyond the
    # ordinary sampling envelope: 15 one-cell tiles, and 14x14 point tiles.
    boundary_rng = np.random.default_rng(73519)
    for kind, shape, n in (("H", (1, 1), 15), ("V", (1, 1), 15),
                           ("P", (1, 1), 15), ("P", (14, 14), 2)):
        pair = _pair(boundary_rng, kind, n, shape)
        _require(np.array_equal(solve(pair["input"]), pair["output"]), "boundary relay failed")
        _require(all(1 <= a <= 30 for a in pair["output"].shape), "boundary output overflow")
    sample["test"][0]["output"].flat[0] = (int(sample["test"][0]["output"].flat[0])+1) % 10
    try:
        validate_generated_puzzle(sample)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    totals.update({"distinct_puzzles": len(signatures), "shape_pairs": len(shapes),
                   "normalized_inputs": len(structures), "stencil_masks": len(masks),
                   "directions": len(directions), "palettes": len(palettes), "counts": sorted(counts)})
    return totals
