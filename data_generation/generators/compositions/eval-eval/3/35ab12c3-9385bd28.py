"""Motion fringes: sparse contours, visible displacement pins, ordered paint.

The contour encloses a solid lattice mask S.  A neighbouring pin gives d.
Only S XOR (S+d) is a layer; common area is transparent.  Earlier table rows
cover later rows, even with literal background paint.  Construction guides
disappear.  The framed table is preserved.  No file I/O or parent imports.
"""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np


OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_ALTERNATIVES = (
    "original", "translated", "union", "intersection", "gained", "lost",
    "boundary", "boxes", "hull", "cue_paint", "reverse", "transparent_bg",
    "global_cancel", "keep_guides", "fixed_direction",
)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not 2 <= int(a) <= 10:
        raise ValueError("num_train must be an integer from 2 through 10")
    if not _integer(b) or not 1 <= int(b) <= 4:
        raise ValueError("num_test must be an integer from 1 through 4")
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer color grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def _line(a, b):
    dr, dc = b[0]-a[0], b[1]-a[1]
    n = max(abs(dr), abs(dc))
    if n == 0 or not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        raise ValueError("invalid contour edge")
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return [(a[0]+i*sr, a[1]+i*sc) for i in range(n+1)]


def _intersects(a, b, c, d):
    if any(max(a[k], b[k]) < min(c[k], d[k])
           or max(c[k], d[k]) < min(a[k], b[k]) for k in (0, 1)):
        return False
    return _cross(a, b, c)*_cross(a, b, d) <= 0 and _cross(c, d, a)*_cross(c, d, b) <= 0


def _cycle(points):
    """Unique noncrossing Hamiltonian cycle on visible eligible alignments.

    Intermediate landmarks block an edge.  Unit edges are eligible.  Reversal
    and a different start are the same cycle.  Half-cell crossings are rejected.
    The search stops after two solutions and has a finite exploration budget.
    """
    points = tuple(sorted(points))
    n = len(points)
    if not 4 <= n <= 10 or len(set(points)) != n:
        raise ValueError("a contour needs 4..10 distinct landmarks")
    edges, adjacency = [], [[] for _ in points]
    for i, j in combinations(range(n), 2):
        a, b = points[i], points[j]
        dr, dc = abs(a[0]-b[0]), abs(a[1]-b[1])
        if (dr == 0 or dc == 0 or dr == dc) and not set(_line(a, b)[1:-1]).intersection(points):
            k = len(edges)
            edges.append((i, j))
            adjacency[i].append((j, k))
            adjacency[j].append((i, k))
    if any(len(x) < 2 for x in adjacency):
        raise ValueError("landmarks do not close")
    conflicts = [set() for _ in edges]
    for k, (a, b) in enumerate(edges):
        for l, (c, d) in enumerate(edges[:k]):
            if not {a, b}.intersection((c, d)) and _intersects(points[a], points[b], points[c], points[d]):
                conflicts[k].add(l)
                conflicts[l].add(k)
    solutions, visited = {}, 0

    def visit(order, used, chosen):
        nonlocal visited
        visited += 1
        if visited > 12000:
            raise ValueError("contour search budget exceeded")
        if len(solutions) > 1:
            return
        if len(order) == n:
            if order[1] > order[-1]:
                return
            for neighbor, edge in adjacency[order[-1]]:
                if neighbor == 0 and not conflicts[edge].intersection(chosen):
                    signature = tuple(sorted(chosen + [edge]))
                    solutions[signature] = tuple(points[i] for i in order)
            return
        for neighbor, edge in adjacency[order[-1]]:
            if neighbor not in used and not conflicts[edge].intersection(chosen):
                visit(order+[neighbor], used | {neighbor}, chosen+[edge])

    visit([0], {0}, [])
    if len(solutions) != 1:
        raise ValueError("contour does not have one unique simple completion")
    polygon = next(iter(solutions.values()))
    # No straight-through vertex masquerades as a missing turn.
    if any(_cross(polygon[i-1], polygon[i], polygon[(i+1) % n]) == 0 for i in range(n)):
        raise ValueError("all landmarks must be turns")
    return polygon, len(edges)-n


def _boundary(points):
    return set().union(*(set(_line(a, b)) for a, b in zip(points, points[1:]+points[:1])))


def _mask(points, shape):
    """Point-centre polygon membership, including its one-cell boundary."""
    rr, cc = np.indices(shape)
    inside = np.zeros(shape, dtype=bool)
    for a, b in zip(points, points[1:]+points[:1]):
        if a[0] != b[0]:
            crossing = a[1] + (rr-a[0])*(b[1]-a[1])/(b[0]-a[0])
            inside ^= ((a[0] > rr) != (b[0] > rr)) & (cc < crossing)
    for r, c in _boundary(points):
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            inside[r, c] = True
    return inside


def _shift(mask, delta):
    dr, dc = delta
    result = np.zeros_like(mask)
    h, w = mask.shape
    r0, r1 = max(0, dr), min(h, h+dr)
    c0, c1 = max(0, dc), min(w, w+dc)
    if r0 < r1 and c0 < c1:
        result[r0:r1, c0:c1] = mask[r0-dr:r1-dr, c0-dc:c1-dc]
    return result


def _hull(points):
    ps = sorted(set(points))
    sides = []
    for seq in (ps, ps[::-1]):
        side = []
        for p in seq:
            while len(side) >= 2 and _cross(side[-2], side[-1], p) <= 0:
                side.pop()
            side.append(p)
        sides.append(side[:-1])
    return tuple(sides[0]+sides[1])


def _parse_candidate(grid, bg, r, c, n, frame):
    h, w = grid.shape
    th = n+2
    if r+th > h:
        raise ValueError("table does not fit")
    table_mask = np.zeros(grid.shape, dtype=bool)
    table_mask[r:r+th, c:c+4] = True
    tile = grid[r:r+th, c:c+4]
    if not (np.all(tile[0] == frame) and np.all(tile[-1] == frame)
            and np.all(tile[:, 0] == frame) and np.all(tile[:, -1] == frame)):
        raise ValueError("incomplete table frame")
    cues = list(map(int, tile[1:-1, 1]))
    fills = list(map(int, tile[1:-1, 2]))
    if len(set(cues)) != n or any(x in (bg, frame) for x in cues) or frame in fills:
        raise ValueError("invalid table roles")
    moat = grid[max(0, r-1):min(h, r+th+1), max(0, c-1):min(w, c+5)].copy()
    local = table_mask[max(0, r-1):min(h, r+th+1), max(0, c-1):min(w, c+5)]
    if np.any(moat[~local] != bg):
        raise ValueError("table requires a local empty moat")
    scene = grid.copy()
    scene[table_mask] = bg
    if set(map(int, np.unique(scene))) - {bg, frame, *cues}:
        raise ValueError("unlisted scene color")
    objects, all_vertices = [], []
    for cue, fill in zip(cues, fills):
        vertices = [tuple(map(int, p)) for p in np.argwhere(scene == cue)]
        polygon, chords = _cycle(vertices)
        s = _mask(polygon, grid.shape)
        if not np.any(s & ~(np.isin(grid, (cue, frame)))):
            raise ValueError("already complete shape")
        objects.append(dict(cue=cue, fill=fill, points=polygon, source=s, chords=chords))
        all_vertices.extend((len(objects)-1, p) for p in vertices)
    pins = [tuple(map(int, p)) for p in np.argwhere(scene == frame)]
    if len(pins) != n:
        raise ValueError("one pin is required per contour")
    assigned = set()
    for pin in pins:
        near = [(i, p) for i, p in all_vertices if max(abs(p[0]-pin[0]), abs(p[1]-pin[1])) == 1]
        if len(near) != 1 or near[0][0] in assigned:
            raise ValueError("ambiguous displacement pin")
        i, anchor = near[0]
        obj = objects[i]
        delta = (pin[0]-anchor[0], pin[1]-anchor[1])
        if pin in _boundary(obj["points"]):
            raise ValueError("pin must not lie on its contour")
        shifted_vertices = tuple((a+delta[0], b+delta[1]) for a, b in obj["points"])
        if any(not (0 <= a < h and 0 <= b < w) for a, b in shifted_vertices):
            raise ValueError("shifted mask is clipped")
        moved = _shift(obj["source"], delta)
        band = obj["source"] ^ moved
        if (int(moved.sum()) != int(obj["source"].sum()) or not band.any()
                or int(np.count_nonzero(moved & ~obj["source"]))
                != int(np.count_nonzero(obj["source"] & ~moved))):
            raise ValueError("movement must preserve area and have equal gain/loss")
        if np.any((obj["source"] | moved) & table_mask):
            raise ValueError("table intersects a shape")
        obj.update(delta=delta, pin=pin, moved=moved, band=band)
        assigned.add(i)
    return dict(background=bg, frame=frame, table=(r, c, n), table_mask=table_mask,
                objects=objects)


def _parse(value):
    grid = _grid(value)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg]*2 <= grid.size:
        raise ValueError("background must occupy more than half the input")
    # Only complete, four-cell monochrome horizontal runs can head a frame.
    headers = np.argwhere((grid[:, :-3] != bg)
                         & (grid[:, :-3] == grid[:, 1:-2])
                         & (grid[:, :-3] == grid[:, 2:-1])
                         & (grid[:, :-3] == grid[:, 3:]))
    found = []
    for rr, cc in headers:
        r, c = int(rr), int(cc)
        for n in range(1, 5):
            try:
                found.append(_parse_candidate(grid, bg, r, c, n, int(grid[r, c])))
            except ValueError:
                continue
    if len(found) != 1:
        raise ValueError("input needs one unambiguous framed table and scene")
    return grid, found[0]


def _base(grid, parsed, keep_guides=False):
    if keep_guides:
        return grid.copy()
    result = np.full_like(grid, parsed["background"])
    result[parsed["table_mask"]] = grid[parsed["table_mask"]]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver: recover geometry, then take the first covering row."""
    grid, parsed = _parse(input_grid)
    result = _base(grid, parsed)
    claimed = np.zeros(grid.shape, dtype=bool)
    for obj in parsed["objects"]:
        visible = obj["band"] & ~claimed
        result[visible] = obj["fill"]
        claimed |= obj["band"]  # Includes opaque background-colored layers.
    return result


def _render(grid, parsed, alternative=None):
    """Bottom-up compositor, also used for explicit shortcut counterfactuals."""
    result = _base(grid, parsed, alternative == "keep_guides")
    objects = parsed["objects"]
    order = objects if alternative == "reverse" else objects[::-1]
    coverage = np.zeros(grid.shape, dtype=np.int8)
    for obj in order:
        s, t = obj["source"], obj["moved"]
        if alternative == "fixed_direction":
            t = _shift(s, objects[0]["delta"])
        if alternative == "original":
            band = s
        elif alternative == "translated":
            band = t
        elif alternative == "union":
            band = s | t
        elif alternative == "intersection":
            band = s & t
        elif alternative == "gained":
            band = t & ~s
        elif alternative == "lost":
            band = s & ~t
        elif alternative in ("boxes", "hull", "boundary"):
            if alternative == "boxes":
                band_source = np.zeros(grid.shape, dtype=bool)
                coords = np.argwhere(s)
                lo, hi = coords.min(0), coords.max(0)
                band_source[lo[0]:hi[0]+1, lo[1]:hi[1]+1] = True
            elif alternative == "hull":
                # Convex-hull edges may have arbitrary slopes; membership is
                # evaluated by half-planes rather than the contour rasterizer.
                vertices = _hull(obj["points"])
                rr, cc = np.indices(grid.shape)
                band_source = np.ones(grid.shape, dtype=bool)
                for a, b in zip(vertices, vertices[1:]+vertices[:1]):
                    band_source &= (b[0]-a[0])*(cc-a[1])-(b[1]-a[1])*(rr-a[0]) >= 0
            else:
                band_source = np.zeros(grid.shape, dtype=bool)
                for p in _boundary(obj["points"]):
                    band_source[p] = True
            band = band_source ^ _shift(band_source, obj["delta"])
        else:
            band = s ^ t
        if alternative == "transparent_bg" and obj["fill"] == parsed["background"]:
            continue
        result[band] = obj["cue"] if alternative == "cue_paint" else obj["fill"]
        coverage += band
    if alternative == "global_cancel":
        result[(coverage > 0) & (coverage % 2 == 0)] = parsed["background"]
    return result


def _concave(points):
    signs = {int(np.sign(_cross(points[i-1], p, points[(i+1) % len(points)])))
             for i, p in enumerate(points)}
    return -1 in signs and 1 in signs


def _features(grid, parsed, output):
    objects = parsed["objects"]
    bg, claimed = parsed["background"], np.zeros(grid.shape, dtype=bool)
    ink_priority = clearing = gained = lost = common = 0
    for i, obj in enumerate(objects):
        visible = obj["band"] & ~claimed
        below = np.zeros_like(visible)
        below_ink = np.zeros_like(visible)
        # Topmost lower layer decides whether opaque clearing is observable.
        lower_claimed = np.zeros_like(visible)
        for other in objects[i+1:]:
            part = other["band"] & ~lower_claimed
            if other["fill"] != obj["fill"] and other["fill"] != bg:
                below_ink |= part
            below |= part
            lower_claimed |= other["band"]
        if obj["fill"] == bg:
            clearing += int(np.count_nonzero(visible & below_ink))
        else:
            ink_priority += int(np.count_nonzero(visible & below_ink))
            gained += int(np.count_nonzero(visible & obj["moved"] & ~obj["source"]))
            lost += int(np.count_nonzero(visible & obj["source"] & ~obj["moved"]))
        claimed |= obj["band"]
    for obj in objects:
        common += int(np.count_nonzero(obj["source"] & obj["moved"] & ~claimed))
    differences = {mode: int(np.count_nonzero(output != _render(grid, parsed, mode)))
                   for mode in _ALTERNATIVES}
    return dict(shape=tuple(grid.shape), background=bg, frame=parsed["frame"],
                layers=len(objects), vertices=tuple(len(o["points"]) for o in objects),
                offsets=tuple(o["delta"] for o in objects),
                concave=sum(_concave(o["points"]) for o in objects),
                chords=sum(o["chords"] for o in objects),
                diagonal=sum(any(a[0] != b[0] and a[1] != b[1]
                                 for a, b in zip(o["points"], o["points"][1:]+o["points"][:1])) for o in objects),
                ink_priority=ink_priority, clearing=clearing, gained=gained, lost=lost,
                common=common, differences=differences,
                boundary=any(np.any(o["source"][[0, -1]]) or np.any(o["source"][:, [0, -1]])
                             or np.any(o["moved"][[0, -1]]) or np.any(o["moved"][:, [0, -1]]) for o in objects))


def _transform(points, code):
    out = []
    for r, c in points:
        if code >= 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        out.append((r, c))
    r0, c0 = min(p[0] for p in out), min(p[1] for p in out)
    return tuple((r-r0, c-c0) for r, c in out)


def _shape(rng, force_concave=False):
    """Parametric polygons; dimensions and cut positions are independently drawn."""
    families = ("elbow", "notch") if force_concave else ("rectangle", "diamond", "chamfer", "elbow", "notch")
    family = str(rng.choice(families))
    for _ in range(60):
        h, w = map(int, rng.integers(4, 12, size=2))
        if family == "rectangle":
            p = ((0, 0), (0, w), (h, w), (h, 0))
        elif family == "diamond":
            d = int(rng.integers(2, 7))
            p = ((0, d), (d, 2*d), (2*d, d), (d, 0))
        elif family == "chamfer":
            cuts = list(map(int, rng.integers(0, min(h, w)//2, size=4)))
            if not any(cuts):
                cuts[0] = 1
            a, b, c, d = cuts
            raw = ((0, a), (0, w-b), (b, w), (h-c, w), (h, w-c), (h, d), (h-d, 0), (a, 0))
            p = tuple(q for i, q in enumerate(raw) if q != raw[i-1])
        elif family == "elbow":
            a, b = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
            p = ((0, 0), (0, w), (a, w), (a, b), (h, b), (h, 0))
        else:
            a, b = sorted(map(int, rng.choice(np.arange(1, w), 2, replace=False)))
            d = int(rng.integers(1, h-1))
            p = ((0, 0), (0, a), (d, a), (d, b), (0, b), (0, w), (h, w), (h, 0))
        p = _transform(p, int(rng.integers(8)))
        try:
            inferred, _ = _cycle(p)
        except ValueError:
            continue
        intended = {frozenset((a, b)) for a, b in zip(p, p[1:]+p[:1])}
        actual = {frozenset((a, b)) for a, b in zip(inferred, inferred[1:]+inferred[:1])}
        if intended == actual and (not force_concave or _concave(inferred)):
            return p
    # Finite known-unique shape, still rotated/reflected and later freely placed.
    p = ((0, 0), (0, 9), (3, 9), (3, 4), (8, 4), (8, 0)) if force_concave else ((0, 0), (0, 7), (5, 7), (5, 0))
    return _transform(p, int(rng.integers(8)))


def _local_object(rng, offsets, force_concave, forced_delta=None):
    p = _shape(rng, force_concave)
    deltas = (forced_delta,) if forced_delta is not None else offsets
    options = []
    boundary = _boundary(p)
    for delta in deltas:
        for anchor in p:
            pin = (anchor[0]+delta[0], anchor[1]+delta[1])
            if pin not in boundary and sum(max(abs(pin[0]-q[0]), abs(pin[1]-q[1])) == 1 for q in p) == 1:
                options.append((delta, pin))
    if not options:
        return None
    delta, pin = options[int(rng.integers(len(options)))]
    r0, c0 = min(0, delta[0]), min(0, delta[1])
    p = tuple((r-r0, c-c0) for r, c in p)
    pin = (pin[0]-r0, pin[1]-c0)
    shape = (max(r for r, c in p)+max(0, delta[0])+1,
             max(c for r, c in p)+max(0, delta[1])+1)
    s = _mask(p, shape)
    return dict(points=p, delta=delta, pin=pin, source=s,
                moved=_shift(s, delta), band=s ^ _shift(s, delta))


def _placed(local, r, c, shape):
    obj = dict(local)
    obj["points"] = tuple((a+r, b+c) for a, b in local["points"])
    obj["pin"] = (local["pin"][0]+r, local["pin"][1]+c)
    for key in ("source", "moved", "band"):
        obj[key] = np.zeros(shape, dtype=bool)
        hh, ww = local[key].shape
        obj[key][r:r+hh, c:c+ww] = local[key]
    return obj


def _layout(rng, n, offsets, force_concave, forced_delta, overlap):
    shape = tuple(map(int, rng.integers(17 if n > 1 else 12, 31, size=2)))
    objects = []
    for i in range(n):
        local = _local_object(rng, offsets, force_concave and i == 0, forced_delta if i == 0 else None)
        if local is None:
            return None
        hh, ww = local["source"].shape
        if hh > shape[0] or ww > shape[1]:
            return None
        candidates = []
        if i > 0 and (overlap or rng.random() < .6):
            previous = objects[int(rng.integers(i))]
            a, b = np.argwhere(previous["band"]), np.argwhere(local["band"])
            for _ in range(100):
                q, p = a[int(rng.integers(len(a)))], b[int(rng.integers(len(b)))]
                candidates.append((int(q[0]-p[0]), int(q[1]-p[1])))
        else:
            candidates = [(int(rng.integers(shape[0]-hh+1)), int(rng.integers(shape[1]-ww+1))) for _ in range(70)]
        for r, c in candidates:
            if r < 0 or c < 0 or r+hh > shape[0] or c+ww > shape[1]:
                continue
            obj = _placed(local, r, c, shape)
            vertices = [p for o in objects for p in o["points"]] + list(obj["points"])
            pins = [o["pin"] for o in objects] + [obj["pin"]]
            if len(set(vertices+pins)) != len(vertices)+len(pins):
                continue
            if any(sum(max(abs(pin[0]-p[0]), abs(pin[1]-p[1])) == 1 for p in vertices) != 1 for pin in pins):
                continue
            objects.append(obj)
            break
        else:
            return None
    return shape, objects


def _candidate(rng, n, offsets, kind, forced_delta):
    layout = _layout(rng, n, offsets, kind == "geometry", forced_delta,
                     kind in ("geometry", "clearing"))
    if layout is None:
        return None
    shape, objects = layout
    roles = list(map(int, rng.permutation(10)))
    bg, frame = roles[:2]
    for cue, obj in zip(roles[2:], objects):
        obj["cue"] = cue
    objects = [objects[int(i)] for i in rng.permutation(n)]
    paint_colors = [x for x in range(10) if x not in (bg, frame)]
    if kind in ("geometry", "clearing"):
        paints = list(map(int, rng.choice(paint_colors, size=n, replace=False)))
    else:
        paints = list(map(int, rng.choice([x for x in range(10) if x != frame], size=n)))
    for obj, fill in zip(objects, paints):
        obj["fill"] = fill
    if kind == "clearing":
        options = [i for i in range(n-1) if any(np.any(objects[i]["band"] & o["band"])
                                                for o in objects[i+1:])]
        if not options:
            return None
        objects[int(rng.choice(options))]["fill"] = bg
    blocked = np.zeros(shape, dtype=bool)
    for obj in objects:
        blocked |= obj["source"] | obj["moved"]
    positions = []
    th = n+2
    for r in range(shape[0]-th+1):
        for c in range(shape[1]-3):
            if not np.any(blocked[max(0, r-1):min(shape[0], r+th+1), max(0, c-1):min(shape[1], c+5)]):
                positions.append((r, c))
    if not positions:
        return None
    tr, tc = positions[int(rng.integers(len(positions)))]
    grid = np.full(shape, bg, dtype=np.int8)
    for obj in objects:
        for p in obj["points"]:
            grid[p] = obj["cue"]
        grid[obj["pin"]] = frame
    grid[tr:tr+th, tc:tc+4] = frame
    for i, obj in enumerate(objects):
        grid[tr+i+1, tc+1:tc+3] = (obj["cue"], obj["fill"])
    # Ground truth uses the sampled polygons and reverse compositing.  The
    # solver reconstructs polygons from public arrays and paints top-first.
    table_mask = np.zeros(shape, dtype=bool)
    table_mask[tr:tr+th, tc:tc+4] = True
    private = dict(background=bg, objects=objects, table_mask=table_mask)
    output = _render(grid, private)
    try:
        _, parsed = _parse(grid)
    except ValueError:
        return None
    if not np.array_equal(output, solve(grid)) or np.array_equal(grid, output):
        return None
    features = _features(grid, parsed, output)
    if (not features["gained"] or not features["lost"]
            or not features["differences"]["cue_paint"]):
        return None
    if kind == "geometry" and (not features["ink_priority"] or not features["gained"]
                               or not features["lost"] or not features["common"]
                               or not features["differences"]["hull"]
                               or not features["differences"]["boundary"]
                               or not features["differences"]["keep_guides"]
                               or not features["differences"]["fixed_direction"]):
        return None
    if kind == "clearing" and not features["clearing"]:
        return None
    return dict(input=grid, output=output)


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _build_pair(rng, n, offsets, kind, forced_delta, seen):
    for _ in range(180):
        pair = _candidate(rng, n, offsets, kind, forced_delta)
        if pair is not None and _fingerprint(pair["input"]) not in seen:
            return pair
    raise RuntimeError("bounded geometric sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return one reproducible puzzle; all semantic conventions are input-visible."""
    nt, nq = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(2))  # held-out displacement, or increased layer count
    held = OFFSETS[int(rng.integers(8))]
    offsets = tuple(d for d in OFFSETS if regime != 0 or d != held)
    first = offsets[int(rng.integers(len(offsets)))]
    second = offsets[int(rng.choice([i for i, d in enumerate(offsets) if first[0]*d[1]-first[1]*d[0] != 0]))]
    plans = [("geometry", int(rng.integers(2, 4)) if regime == 0 else 2, first),
             ("clearing", int(rng.integers(2, 5)) if regime == 0 else 2, second)]
    for _ in range(nt-2):
        plans.append(("free", int(rng.integers(1, 5 if regime == 0 else 3)), None))
    rng.shuffle(plans)
    puzzle, seen = {"train": [], "test": []}, set()
    for kind, n, forced in plans:
        pair = _build_pair(rng, n, offsets, kind, forced, seen)
        seen.add(_fingerprint(pair["input"]))
        puzzle["train"].append(pair)
    familiar = tuple(sorted({d for p in puzzle["train"] for d in _features(p["input"], _parse(p["input"])[1], p["output"])["offsets"]}))
    for _ in range(nq):
        test_offsets = OFFSETS if regime == 0 else familiar
        n = int(rng.integers(1, 5)) if regime == 0 else int(rng.integers(3, 5))
        pair = _build_pair(rng, n, test_offsets, "free", held if regime == 0 else None, seen)
        seen.add(_fingerprint(pair["input"]))
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check public-array semantics, teaching witnesses, and the actual shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in records:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            assert grid.shape == output.shape
            key = _fingerprint(grid)
            assert key not in seen, "duplicate input"
            seen.add(key)
            _, parsed = _parse(grid)
            assert np.array_equal(solve(grid), output), "incorrect output"
            assert np.array_equal(_render(grid, parsed), output), "compositor disagreement"
            assert np.any(grid != output), "identity example"
            assert np.array_equal(grid[parsed["table_mask"]], output[parsed["table_mask"]])
            info = _features(grid, parsed, output)
            assert info["gained"] and info["lost"], "both motion fringes must be visible"
            assert info["differences"]["cue_paint"], "paint lookup has no effect"
            records[split].append(info)
    train = records["train"]
    assert any(x["ink_priority"] for x in train), "no contrasting painted crossing"
    assert any(x["clearing"] for x in train), "no opaque background crossing"
    assert any(x["concave"] for x in train), "no concavity teaching"
    assert all(any(x["differences"][mode] for x in train) for mode in _ALTERNATIVES), "a simpler alternative survives all demonstrations"
    assert all(any(x[key] for x in train) for key in ("gained", "lost", "common"))
    training_offsets = {d for x in train for d in x["offsets"]}
    assert any(a[0]*b[1] != a[1]*b[0] for a, b in combinations(training_offsets, 2)), "offsets fail to establish a two-dimensional displacement rule"
    test = records["test"]
    assert (any(set(x["offsets"])-training_offsets for x in test)
            or min(x["layers"] for x in test) > max(x["layers"] for x in train)), "no meaningful train/test shift"


def self_test():
    """Bounded 42-seed sweep; only summaries/digests retained in memory."""
    records, signatures, normalized = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
                digest.update(pair["output"].tobytes())
                _, parsed = _parse(grid)
                info = _features(grid, parsed, pair["output"])
                info["split"] = split
                records.append(info)
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in grid.flat)
                normalized.add((grid.shape, hashlib.sha256(norm).digest()))
        signatures.add(digest.digest())
    assert len(signatures) == 42
    assert len(normalized) >= 180
    assert len({r["shape"] for r in records}) >= 50
    assert {r["background"] for r in records} == set(range(10))
    assert {r["frame"] for r in records} == set(range(10))
    assert {d for r in records for d in r["offsets"]} == set(OFFSETS)
    assert {r["layers"] for r in records} == {1, 2, 3, 4}
    assert len({r["vertices"] for r in records}) >= 15
    for split in ("train", "test"):
        rows = [r for r in records if r["split"] == split]
        for key in ("concave", "diagonal", "chords", "ink_priority", "clearing", "boundary"):
            assert any(r[key] for r in rows), (split, key)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(700+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(123456, 5, 2), generate_puzzle(123456, 5, 2)
    assert all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in ("input", "output"))
    broken = generate_puzzle(0, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted output")
    # Deliberate hand fixture: a 3x4 solid rectangle moving right gives only
    # the vacated left column and newly occupied right column, never its middle.
    s = _mask(((1, 1), (1, 4), (3, 4), (3, 1)), (6, 7))
    expected = np.zeros((6, 7), dtype=bool)
    expected[1:4, [1, 5]] = True
    assert np.array_equal(s ^ _shift(s, (0, 1)), expected)
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(signatures),
                color_normalized_inputs=len(normalized), shapes=len({r["shape"] for r in records}),
                offsets=len({d for r in records for d in r["offsets"]}))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
