"""Distributed mirror drawings: pool anchor-normalized clues, then draw everywhere.

Only the standard library and NumPy are used. No parent generator is imported.
"""
from itertools import combinations
import numpy as np


_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_SMALL_LAYOUTS = ((1, 3), (3, 1), (2, 2))
_LARGE_LAYOUTS = ((2, 3), (3, 2))


def _integer(value, name, lo, hi=None):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or value < lo or
            (hi is not None and value > hi)):
        raise ValueError(f"illegal {name}: {value!r}")
    return int(value)


def _segment(a, b):
    dr, dc = b[0]-a[0], b[1]-a[1]
    length = max(abs(dr), abs(dc))
    assert length and (dr == 0 or dc == 0 or abs(dr) == abs(dc))
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return {(a[0]+i*sr, a[1]+i*sc) for i in range(length+1)}


def _edges(points):
    occupied = set(points)
    result = []
    for i, j in combinations(range(len(points)), 2):
        a, b = points[i], points[j]
        dr, dc = abs(a[0]-b[0]), abs(a[1]-b[1])
        if dr == 0 or dc == 0 or dr == dc:
            if not ((_segment(a, b)-{a, b}) & occupied):
                result.append((i, j))
    return result


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _intersect(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or
            max(c[0], d[0]) < min(a[0], b[0]) or
            max(a[1], b[1]) < min(c[1], d[1]) or
            max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return (_cross(a, b, c)*_cross(a, b, d) <= 0 and
            _cross(c, d, a)*_cross(c, d, b) <= 0)


def _topology(points):
    """Unique simple Hamiltonian cycle preferred to a unique simple full path.

    Paths differing only by reversal and cycles differing by traversal/start
    represent the same edge set. Continuous intersections include half pixels.
    """
    points = sorted(set(points))
    n = len(points)
    assert 2 <= n <= 10, "expected 2..10 distinct pooled landmarks"
    edges = _edges(points)
    adjacent = [[] for _ in points]
    for k, (i, j) in enumerate(edges):
        adjacent[i].append((j, k))
        adjacent[j].append((i, k))
    assert all(adjacent), "unlinked landmark"
    conflicts = [set() for _ in edges]
    for k, (a, b) in enumerate(edges):
        for m, (c, d) in enumerate(edges[:k]):
            if not ({a, b} & {c, d}) and _intersect(points[a], points[b], points[c], points[d]):
                conflicts[k].add(m)
                conflicts[m].add(k)

    def enumerate_orders(closed):
        answers = {}

        def extend(order, used, chosen):
            if len(answers) >= 2:
                return
            if len(order) == n:
                final = chosen
                if closed:
                    closing = [k for j, k in adjacent[order[-1]] if j == order[0]]
                    if not closing or conflicts[closing[0]] & chosen:
                        return
                    final = chosen | {closing[0]}
                answers[tuple(sorted(final))] = tuple(order)
                return
            for neighbor, edge in adjacent[order[-1]]:
                if neighbor not in used and not (conflicts[edge] & chosen):
                    extend(order+[neighbor], used | {neighbor}, chosen | {edge})

        for start in (range(1) if closed else range(n)):
            extend([start], {start}, set())
            if len(answers) >= 2:
                break
        return answers

    cycles = enumerate_orders(True) if n >= 3 else {}
    assert len(cycles) <= 1, "competing full cycles"
    answers = cycles if cycles else enumerate_orders(False)
    assert len(answers) == 1, "no unique full drawing"
    indices, order = next(iter(answers.items()))
    selected = [edges[k] for k in indices]
    pixels = set().union(*(_segment(points[i], points[j]) for i, j in selected))
    assert pixels-set(points), "drawing must have missing cells"
    all_pixels = set().union(*(_segment(points[i], points[j]) for i, j in edges))
    turns = ([_cross(points[order[i-1]], points[order[i]], points[order[(i+1) % n]])
              for i in range(n)] if cycles else [])
    return {"points": points, "edges": selected, "pixels": pixels,
            "closed": bool(cycles), "chord_cells": len(all_pixels-pixels),
            "concave": any(t < 0 for t in turns) and any(t > 0 for t in turns)}


def _rooms(shape, rows, cols):
    rr, cc = [-1]+list(rows)+[shape[0]], [-1]+list(cols)+[shape[1]]
    return [{"bounds": (rr[i]+1, rr[i+1], cc[j]+1, cc[j+1]),
             "sign": (1 if i % 2 == 0 else -1, 1 if j % 2 == 0 else -1)}
            for i in range(len(rr)-1) for j in range(len(cc)-1)]


def _square(panel, color):
    coords = np.argwhere(panel == color)
    assert len(coords), "missing square anchor"
    top, left = map(int, coords.min(axis=0))
    bottom, right = map(int, coords.max(axis=0)+1)
    side = bottom-top
    assert side == right-left and len(coords) == side*side
    return top, left, side


def _decode(input_grid):
    grid = np.asarray(input_grid)
    assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= d <= 30 for d in grid.shape) and np.all((grid >= 0) & (grid <= 9))
    grid = grid.astype(np.int64, copy=False)
    bg = int(np.bincount(grid.ravel(), minlength=10).argmax())
    assert np.count_nonzero(grid == bg) > grid.size/2, "background must be a strict majority"
    rows = [r for r in range(grid.shape[0]) if grid[r, 0] != bg and np.all(grid[r] == grid[r, 0])]
    cols = [c for c in range(grid.shape[1]) if grid[0, c] != bg and np.all(grid[:, c] == grid[0, c])]
    assert rows or cols, "missing dividers"
    divider_colors = {int(grid[r, 0]) for r in rows} | {int(grid[0, c]) for c in cols}
    assert len(divider_colors) == 1
    divider = divider_colors.pop()
    assert all(0 < r < grid.shape[0]-1 for r in rows)
    assert all(0 < c < grid.shape[1]-1 for c in cols)
    rooms = _rooms(grid.shape, rows, cols)
    assert 3 <= len(rooms) <= 6 and len(rows) <= 2 and len(cols) <= 2
    palettes = []
    for room in rooms:
        a, b, c, d = room["bounds"]
        assert a < b and c < d
        room["panel"] = grid[a:b, c:d]
        palettes.append(set(map(int, np.unique(room["panel"]))))
    common = set.intersection(*palettes)-{bg}
    assert len(common) == 1, "anchor is the unique color present in every room"
    anchor = common.pop()
    assert anchor != divider
    cells_by_color = {}
    occurrences = {}
    for room in rooms:
        panel = room["panel"]
        r, c, side = _square(panel, anchor)
        assert 1 <= side <= 3
        room["anchor"] = (r, c, side)
        room["cells"] = {}
        sr, sc = room["sign"]
        for y, x in np.argwhere((panel != bg) & (panel != anchor)):
            u, v = (int(y)-r)//side, (int(x)-c)//side
            t, l = r+side*u, c+side*v
            assert 0 <= t <= panel.shape[0]-side and 0 <= l <= panel.shape[1]-side
            color = int(panel[y, x])
            assert color not in (divider, anchor, bg) and np.all(panel[t:t+side, l:l+side] == color)
            point = (sr*u, sc*v)
            room["cells"][point] = color
        for point, color in room["cells"].items():
            cells_by_color.setdefault(color, set()).add(point)
            occurrences.setdefault(color, []).append((room, point))
    sources = [color for color, points in cells_by_color.items() if len(points) >= 2]
    assert len(sources) == 1, "expected one pooled landmark color"
    source = sources.pop()
    vertices = cells_by_color[source]
    assert (0, 0) not in vertices
    topo = _topology(vertices)
    assert (0, 0) not in topo["pixels"], "anchor cannot lie on the drawing"
    assert sum(any(c == source for c in room["cells"].values()) for room in rooms) >= 2
    assert any(not room["cells"] for room in rooms), "need an anchor-only room"
    assert all({p for p, c in room["cells"].items() if c == source} != vertices for room in rooms), "no room may supply the entire drawing"
    companions = set(cells_by_color)-{source}
    assert len(companions) <= 1
    companion, offset, marker_point = None, None, None
    if companions:
        companion = companions.pop()
        assert len(occurrences[companion]) == 1, "one global companion bead"
        marker_room, marker_point = occurrences[companion][0]
        candidates = [v for v in vertices if max(abs(v[0]-marker_point[0]), abs(v[1]-marker_point[1])) == 1]
        assert len(candidates) == 1
        v = candidates[0]
        assert marker_room["cells"].get(v) == source, "marker's source bead must be locally visible"
        offset = (marker_point[0]-v[0], marker_point[1]-v[1])
        assert marker_point not in topo["pixels"]
        assert (0, 0) not in {(r+offset[0], c+offset[1]) for r, c in topo["pixels"]}
    return {"grid": grid, "background": bg, "divider": divider, "anchor_color": anchor,
            "rooms": rooms, "source": source, "companion": companion, "offset": offset,
            "marker": marker_point, "topology": topo, "rows": rows, "cols": cols}


def _paint(out, room, points, color):
    a, b, c, d = room["bounds"]
    y, x, side = room["anchor"]
    sr, sc = room["sign"]
    for r, col in points:
        top, left = a+y+side*sr*r, c+x+side*sc*col
        r0, r1, c0, c1 = max(a, top), min(b, top+side), max(c, left), min(d, left+side)
        if r0 < r1 and c0 < c1:
            out[r0:r1, c0:c1] = color


def _render(data, source_pixels=None, offset=None, reflect=True):
    out = data["grid"].copy()
    pixels = data["topology"]["pixels"] if source_pixels is None else source_pixels
    delta = data["offset"] if offset is None else offset
    for room in data["rooms"]:
        draw_room = room if reflect else dict(room, sign=(1, 1))
        if data["companion"] is not None:
            copied = {(r+delta[0], c+delta[1]) for r, c in pixels}
            _paint(out, draw_room, copied, data["companion"])
        _paint(out, draw_room, pixels, data["source"])
    # All supplied cells, including the one companion bead, remain in front.
    mask = data["grid"] != data["background"]
    out[mask] = data["grid"][mask]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode and solve entirely from the visible grid; does not mutate it."""
    return _render(_decode(input_grid))


def _orient(points, code):
    result = []
    for r, c in points:
        if code & 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        result.append((r, c))
    return result


def _shape(rng, kind):
    if kind == "cycle":
        h, w = map(int, rng.integers(2, 6, size=2))
        style = int(rng.integers(4))
        if style == 0:
            return [(0, 0), (0, w), (h, w), (h, 0)], True
        if style == 1:
            k = int(rng.integers(1, 4))
            return [(0, k), (k, 2*k), (2*k, k), (k, 0)], True
        if style == 2:
            k = int(rng.integers(1, min(h, w)))
            return [(0, k), (0, w), (h, w), (h, 0), (k, 0)], True
        w = max(3, w)
        u, v = sorted(map(int, rng.choice(range(1, w), size=2, replace=False)))
        depth = int(rng.integers(1, h))
        return [(0, 0), (0, w), (h, w), (h, v), (h-depth, v),
                (h-depth, u), (h, u), (h, 0)], True
    points = [(0, 0)]
    count = 2 if kind == "segment" else int(rng.integers(3, 8))
    for _ in range(count-1):
        dr, dc = _NEIGHBORS[int(rng.integers(8))]
        length = int(rng.integers(2 if kind == "segment" else 1, 5))
        points.append((points[-1][0]+dr*length, points[-1][1]+dc*length))
    return points, False


def _make_drawing(rng, kind, witness, fallback=False):
    for _ in range(100):
        if fallback:
            points, closed = (([(1, 1), (1, 3), (2, 3)], False) if kind != "cycle" else
                              ([(1, 1), (1, 3), (3, 3), (3, 1)], True))
        else:
            points, closed = _shape(rng, kind)
        if len(set(points)) != len(points):
            continue
        points = _orient(points, int(rng.integers(8)))
        try:
            topo = _topology(points)
        except AssertionError:
            continue
        proposed = {frozenset((points[i], points[(i+1) % len(points)]))
                    for i in range(len(points) if closed else len(points)-1)}
        inferred = {frozenset((topo["points"][i], topo["points"][j])) for i, j in topo["edges"]}
        if proposed != inferred or (witness and not topo["chord_cells"]):
            continue
        pixels = topo["pixels"]
        if max(max(r for r, c in pixels)-min(r for r, c in pixels),
               max(c for r, c in pixels)-min(c for r, c in pixels)) > 7:
            continue
        markers = []
        for vertex in points:
            for dr, dc in _NEIGHBORS:
                marker = (vertex[0]+dr, vertex[1]+dc)
                copied = {(r+dr, c+dc) for r, c in pixels}
                if (marker in pixels or (fallback and (0, 0) in copied) or
                        sum(max(abs(marker[0]-v[0]), abs(marker[1]-v[1])) == 1 for v in points) != 1):
                    continue
                if witness and (not ((copied & pixels)-set(points)) or len(copied-pixels-{marker}) < 2):
                    continue
                markers.append((vertex, marker, (dr, dc), copied))
        if not markers:
            continue
        vertex, marker, offset, copied = markers[int(rng.integers(len(markers)))]
        occupied = pixels | copied
        origins = [(r, c) for r in range(min(p[0] for p in occupied)-1, max(p[0] for p in occupied)+2)
                   for c in range(min(p[1] for p in occupied)-1, max(p[1] for p in occupied)+2)
                   if (r, c) not in occupied]
        # Interior and exterior anchors both occur; they are coordinate origins.
        origin = (0, 0) if fallback else origins[int(rng.integers(len(origins)))]
        move = lambda p: (p[0]-origin[0], p[1]-origin[1])
        return {"vertices": [move(p) for p in points], "pixels": {move(p) for p in pixels},
                "vertex": move(vertex), "marker": move(marker), "offset": offset,
                "closed": closed}
    return None


def _bbox(points):
    return (min(r for r, c in points), max(r for r, c in points),
            min(c for r, c in points), max(c for r, c in points))


def _placements(room_shape, side, sign, required, whole, clipped):
    sr, sc = sign
    required = {(sr*r, sc*c) for r, c in required} | {(0, 0)}
    whole = {(sr*r, sc*c) for r, c in whole} | {(0, 0)}
    u, v, x, z = _bbox(required)
    wu, wv, wx, wz = _bbox(whole)
    h, w = room_shape
    candidates = []
    for r in range(max(0, -u*side), min(h-side, h-(v+1)*side)+1):
        for c in range(max(0, -x*side), min(w-side, w-(z+1)*side)+1):
            cut = r+wu*side < 0 or r+(wv+1)*side > h or c+wx*side < 0 or c+(wz+1)*side > w
            if cut != clipped:
                continue
            visible = sum(0 <= r+a*side < h and 0 <= c+b*side < w
                          for a, b in whole-{(0, 0)})
            if visible >= 2:
                candidates.append((r, c, side))
    return candidates


def _axis_sizes(rng, minima):
    budget = 30-(len(minima)-1)-sum(minima)
    if budget < 0:
        return None
    extras = [0]*len(minima)
    for _ in range(int(rng.integers(budget+1))):
        extras[int(rng.integers(len(minima)))] += 1
    return [a+b for a, b in zip(minima, extras)]


def _build_pair(rng, layout, kind, witness=False, clip_demo=False,
                repeated=True, large_scale=False, fallback=False, marked=True):
    drawing = _make_drawing(rng, kind, witness, fallback)
    if drawing is None:
        return None
    nrows, ncols = layout
    count = nrows*ncols
    vertices = drawing["vertices"]
    blank = int(rng.integers(count))
    available = [i for i in range(count) if i != blank]
    active = [int(i) for i in rng.choice(available, size=int(rng.integers(2, min(len(vertices), len(available))+1)), replace=False)]
    subsets = [set() for _ in range(count)]
    order = [vertices[int(i)] for i in rng.permutation(len(vertices))]
    for index, v in enumerate(order):
        subsets[active[index % len(active)]].add(v)
    if repeated and len(vertices) > 2:
        for v in order:
            if rng.random() < .45:
                candidates = [i for i in active if v not in subsets[i] and len(subsets[i]) < len(vertices)-1]
                if candidates:
                    subsets[int(rng.choice(candidates))].add(v)
        if sum(map(len, subsets)) == len(vertices):
            # Make the redundancy-to-disjoint-clues curriculum an actual shift,
            # even when only two training examples are requested.
            choices = [(i, v) for i in active for v in vertices
                       if v not in subsets[i] and len(subsets[i]) < len(vertices)-1]
            i, v = choices[int(rng.integers(len(choices)))]
            subsets[i].add(v)
    if any(len(s) == len(vertices) for s in subsets):
        return None
    marker_owner = int(rng.choice([i for i in active if drawing["vertex"] in subsets[i]]))
    copied = {(r+drawing["offset"][0], c+drawing["offset"][1]) for r, c in drawing["pixels"]} if marked else set()
    whole = drawing["pixels"] | copied | {(0, 0)}
    u, v, x, z = _bbox(whole)
    logical_h, logical_w = v-u+1, z-x+1
    sides = [1]*count if fallback else [int(rng.choice([1, 1, 1, 2])) for _ in range(count)]
    if witness:
        sides = [1]*count
        sides[int(rng.integers(count))] = 2
    elif large_scale:
        sides = [1]*count
        sides[int(rng.integers(count))] = 3
    row_min = [max(logical_h*sides[i*ncols+j] for j in range(ncols)) for i in range(nrows)]
    col_min = [max(logical_w*sides[i*ncols+j] for i in range(nrows)) for j in range(ncols)]
    heights, widths = _axis_sizes(rng, row_min), _axis_sizes(rng, col_min)
    if heights is None or widths is None:
        return None
    rows = np.cumsum([h+1 for h in heights[:-1]], dtype=int)-1
    cols = np.cumsum([w+1 for w in widths[:-1]], dtype=int)-1
    shape = (sum(heights)+nrows-1, sum(widths)+ncols-1)
    colors = [int(c) for c in rng.permutation(10)]
    bg, divider, anchor, source, companion = colors[:5]
    inp = np.full(shape, bg, dtype=np.int8)
    inp[rows, :] = divider
    inp[:, cols] = divider
    rooms = _rooms(shape, rows, cols)
    for i, room in enumerate(rooms):
        a, b, c, d = room["bounds"]
        required = subsets[i] | ({drawing["marker"]} if marked and i == marker_owner else set())
        clipped = (i == blank if clip_demo else not witness and not fallback and i != active[0] and rng.random() < .35)
        placements = _placements((b-a, d-c), sides[i], room["sign"], required, whole, clipped)
        if not placements and clipped and not (clip_demo and i == blank):
            placements = _placements((b-a, d-c), sides[i], room["sign"], required, whole, False)
        if not placements:
            return None
        room["anchor"] = placements[int(rng.integers(len(placements)))]
        y, xx, side = room["anchor"]
        inp[a+y:a+y+side, c+xx:c+xx+side] = anchor
        _paint(inp, room, subsets[i], source)
        if marked and i == marker_owner:
            _paint(inp, room, {drawing["marker"]}, companion)
    if np.count_nonzero(inp == bg) <= inp.size/2:
        return None
    out = inp.copy()
    for room in rooms:
        if marked:
            _paint(out, room, copied, companion)
        _paint(out, room, drawing["pixels"], source)
    mask = inp != bg
    out[mask] = inp[mask]
    pair = {"input": inp, "output": out}
    try:
        data = _decode(inp)
        assert np.array_equal(_render(data), out)
        if witness:
            f = _features(pair, data)
            assert f["priority_cells"] > 0 and f["visible_chord_difference"] > 0
        if clip_demo:
            assert _features(pair, data)["clipped_rooms"] > 0
    except AssertionError:
        return None
    return pair


def _key(grid):
    arr = np.asarray(grid, dtype=np.int8)
    return arr.shape, arr.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # more rooms / fewer repeated clues / a new scale
    strip = _SMALL_LAYOUTS[int(rng.integers(2))]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            witness = split == "train" and index == 0
            clip_demo = split == "train" and index == 1
            if witness:
                layout, kind = (2, 2), "cycle"
            elif clip_demo:
                layout, kind = strip, "walk"
            else:
                pool = _LARGE_LAYOUTS if split == "test" and regime == 0 else _SMALL_LAYOUTS if regime == 0 else _SMALL_LAYOUTS+_LARGE_LAYOUTS
                layout = pool[int(rng.integers(len(pool)))]
                kind = str(rng.choice(["cycle", "walk", "segment"], p=[.48, .36, .16]))
            repeated = split == "train" or regime != 1
            large_scale = split == "test" and regime == 2
            marked = witness or clip_demo or rng.random() < .85
            pair = None
            for attempt in range(240):
                candidate = _build_pair(rng, layout, kind, witness, clip_demo,
                                        repeated, large_scale, marked=marked)
                if candidate is not None and _key(candidate["input"]) not in seen:
                    pair = candidate
                    break
            if pair is None:
                # A small known contour is a finite capacity fallback. A requested
                # open demonstration instead uses a compact one-bend walk.
                candidate = _build_pair(rng, layout, kind, witness, clip_demo,
                                        repeated, large_scale, fallback=True, marked=marked)
                assert candidate is not None, "compact capacity fallback failed"
                if _key(candidate["input"]) not in seen:
                    pair = candidate
                else:
                    # Ninety distinct background/anchor bijections suffice
                    # against at most thirteen previously accepted inputs.
                    d = _decode(candidate["input"])
                    old = [d["background"], d["anchor_color"]]
                    old += [c for c in range(10) if c not in old]
                    for background in range(10):
                        for anchor in range(10):
                            if background == anchor:
                                continue
                            new = [background, anchor]
                            new += [c for c in range(10) if c not in new]
                            mapping = np.empty(10, dtype=np.int8)
                            mapping[old] = new
                            recolored = {k: mapping[v] for k, v in candidate.items()}
                            if _key(recolored["input"]) not in seen:
                                pair = recolored
                                break
                        if pair is not None:
                            break
            assert pair is not None, "bounded construction exhausted"
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair, data=None):
    data = _decode(pair["input"]) if data is None else data
    topo, bg = data["topology"], data["background"]
    pixels = topo["pixels"]
    copied = ({(r+data["offset"][0], c+data["offset"][1]) for r, c in pixels}
              if data["companion"] is not None else set())
    out = np.asarray(pair["output"])
    clipped = 0
    priority = 0
    full = 0
    for room in data["rooms"]:
        a, b, c, d = room["bounds"]
        y, x, side = room["anchor"]
        sr, sc = room["sign"]
        outside = any(a+y+side*sr*r < a or a+y+side*sr*r+side > b or
                      c+x+side*sc*col < c or c+x+side*sc*col+side > d for r, col in pixels | copied)
        clipped += outside
        full += not outside
        for r, col in (pixels & copied)-set(topo["points"]):
            top, left = a+y+side*sr*r, c+x+side*sc*col
            if a <= top and top+side <= b and c <= left and left+side <= d:
                priority += int(np.count_nonzero((data["grid"][top:top+side, left:left+side] == bg) &
                                                (out[top:top+side, left:left+side] == data["source"])))
    visits = sum(sum(color == data["source"] for color in r["cells"].values()) for r in data["rooms"])
    source_missing = int(np.count_nonzero((out == data["source"]) & (data["grid"] == bg)))
    companion_added = (int(np.count_nonzero((out == data["companion"]) & (data["grid"] == bg)))
                       if data["companion"] is not None else 0)
    all_pixels = set().union(*(_segment(topo["points"][i], topo["points"][j])
                              for i, j in _edges(topo["points"])))
    chord_difference = int(np.count_nonzero(_render(data, source_pixels=all_pixels) != out))
    return {"shape": list(out.shape), "layout": [len(data["rows"])+1, len(data["cols"])+1],
            "vertices": len(topo["points"]), "closed": topo["closed"], "concave": topo["concave"],
            "chord_cells": topo["chord_cells"], "visible_chord_difference": chord_difference,
            "clipped_rooms": clipped, "full_rooms": full,
            "sides": [r["anchor"][2] for r in data["rooms"]], "offset": data["offset"],
            "priority_cells": priority, "source_added": source_missing, "companion_added": companion_added,
            "repeated_clues": visits-len(topo["points"]), "background": bg,
            "geometry": repr((tuple(topo["points"]), tuple(topo["edges"]), data["offset"])),
            "marker_room": next((i for i, r in enumerate(data["rooms"]) if data["companion"] in r["cells"].values()), -1)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            for grid in (inp, out):
                assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape) and np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            data = _decode(inp)
            assert np.array_equal(_render(data), out), "incorrect distributed drawing"
            mask = inp != data["background"]
            assert np.array_equal(inp[mask], out[mask]), "given cells were overwritten"
            f = _features(pair, data)
            assert f["source_added"] > 0 and f["full_rooms"] > 0
            assert not np.array_equal(_render(data, source_pixels=set(data["topology"]["points"])), out), "completion must matter"
            # The anchor-only room changes. Independent local completions cannot
            # produce it, even if their local topology happens to be correct.
            assert any(not r["cells"] and np.any(out[r["bounds"][0]:r["bounds"][1], r["bounds"][2]:r["bounds"][3]] != r["panel"])
                       for r in data["rooms"])
            if split == "train":
                training.append((f, data, out))
    assert any(f["closed"] and f["visible_chord_difference"] > 0 and f["priority_cells"] > 0 and
               f["companion_added"] > 0 and len(set(f["sides"])) >= 2 and f["clipped_rooms"] == 0
               for f, data, out in training), "need a complete topology/scale/companion demonstration"
    assert any(not f["closed"] and f["clipped_rooms"] > 0 for f, data, out in training), "need open-path and clipping evidence"
    assert any(not np.array_equal(_render(data, reflect=False), out) for f, data, out in training), "reflections must be observable"
    assert any(not np.array_equal(_render(data, offset=(0, 0)), out) for f, data, out in training), "companion translation must be observable"
    layouts = {tuple(f["layout"]) for f, data, out in training}
    assert (2, 2) in layouts and bool(layouts & {(1, 3), (3, 1)})


def self_test():
    """Bounded 42-seed structural, semantic, metamorphic and diversity checks."""
    features, keys = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        keys.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"].copy()
                assert np.array_equal(solve(inp), pair["output"]) and np.array_equal(inp, pair["input"])
                features.append(_features(pair))
        if seed < 3:
            other = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in ("train", "test")
                       for p, q in zip(puzzle[s], other[s]) for k in ("input", "output"))
    assert len(keys) == 42
    assert {tuple(f["layout"]) for f in features} == set(_SMALL_LAYOUTS+_LARGE_LAYOUTS)
    assert len({tuple(f["shape"]) for f in features}) >= 40
    assert len({f["geometry"] for f in features}) >= 80
    assert {s for f in features for s in f["sides"]} == {1, 2, 3}
    assert {f["background"] for f in features} == set(range(10))
    assert {f["offset"] for f in features if f["offset"] is not None} == set(_NEIGHBORS)
    assert any(f["concave"] for f in features)
    assert any(f["vertices"] == 2 for f in features) and any(f["vertices"] >= 7 for f in features)
    assert any(f["repeated_clues"] > 0 for f in features) and any(f["repeated_clues"] == 0 for f in features)
    assert any(f["companion_added"] == 0 for f in features)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            generate_puzzle(1000+a*10+b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    puzzle = generate_puzzle(0, 2, 1)
    pair = puzzle["test"][0]
    # Recoloring is equivariant; roles are inferred from geometry.
    permutation = np.array([9, 6, 3, 0, 7, 4, 1, 8, 5, 2], dtype=np.int8)
    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    # Whole-grid reflections and rotations also preserve the relative-mirror rule.
    for turns in range(4):
        for mirror in (False, True):
            a, b = np.rot90(pair["input"], turns), np.rot90(pair["output"], turns)
            if mirror:
                a, b = np.fliplr(a), np.fliplr(b)
            assert np.array_equal(solve(a), b)
    # Exercise the capacity fallback, rather than leaving it untested.
    for layout in _SMALL_LAYOUTS+_LARGE_LAYOUTS:
        for kind in ("cycle", "walk"):
            for marked in (False, True):
                fallback = _build_pair(np.random.default_rng(7), layout, kind,
                                       clip_demo=kind == "walk", fallback=True,
                                       large_scale=True, marked=marked)
                assert fallback is not None and np.array_equal(solve(fallback["input"]), fallback["output"])
    location = tuple(np.argwhere(pair["input"] != pair["output"])[0])
    pair["output"][location] = pair["input"][location]
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
