"""Ordinal docking: ranked hollow/solid edge ports assemble anchored panels.

Only the standard library and NumPy are used. No files, generation metadata,
examples, or seed are consulted by solve(). Diagnostic routines store no grids.
"""
from itertools import product
import hashlib

import numpy as np


_OPPOSITE = {"T": "B", "B": "T", "L": "R", "R": "L"}
_CORNERS = ("TL", "TR", "BL", "BR")


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= d <= 30 for d in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((0 <= g) & (g <= 9))
    return g.astype(np.int8, copy=False)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _marker_cells(point, corner):
    y, x = point
    dy = -1 if corner[0] == "T" else 1
    dx = -1 if corner[1] == "L" else 1
    return [(y+dy, x), (y, x+dx), (y+dy, x+dx)]


def _parse(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    assert all(np.all(edge == bg) for edge in (g[0], g[-1], g[:, 0], g[:, -1])), "background perimeter"
    markers = []
    for color in np.unique(g):
        q = np.argwhere(g == color)
        if color == bg or len(q) != 3:
            continue
        y, x = q.min(0)
        h, w = q.max(0)-q.min(0)+1
        if (h, w) != (2, 2):
            continue
        missing = {(int(y), int(x)), (int(y+1), int(x)),
                   (int(y), int(x+1)), (int(y+1), int(x+1))} - set(map(tuple, q.tolist()))
        point = missing.pop()
        if g[point] in (bg, color):
            continue
        corner = {(1, 1): "TL", (1, 0): "TR", (0, 1): "BL", (0, 0): "BR"}[(point[0]-y, point[1]-x)]
        markers.append((int(color), point, corner))
    assert len(markers) == 1, "one external three-cell L marker"
    marker, point, corner = markers[0]
    panels, ports = [], []
    carrier, size, anchor = None, None, None
    for cells in _components((g != bg) & (g != marker)):
        q = np.asarray(cells)
        y, x = map(int, q.min(0))
        h, w = map(int, q.max(0)-q.min(0)+1)
        assert len(cells) == h*w, "filled rectangular panel"
        tile = g[y:y+h, x:x+w].copy()
        base = int(tile[0, 0])
        assert all(int(tile[p]) == base for p in ((0, w-1), (h-1, 0), (h-1, w-1)))
        assert base not in (bg, marker)
        assert carrier is None or carrier == base, "common panel carrier"
        carrier = base
        index = len(panels)
        panels.append({"top": y, "left": x, "tile": tile})
        if point in cells:
            anchor = index
            assert point == (y+(h-1 if corner[0] == "B" else 0),
                             x+(w-1 if corner[1] == "R" else 0))
        local_ports = []
        for ink_cells in _components(tile != base):
            z = np.asarray(ink_cells)
            py, px = map(int, z.min(0))
            ph, pw = map(int, z.max(0)-z.min(0)+1)
            assert ph == pw and ph >= 3, "square port"
            assert size is None or size == ph, "common port scale"
            size = ph
            glyph = tile[py:py+ph, px:px+ph]
            color = int(glyph[0, 0])
            assert color not in (bg, marker, base)
            expected = np.full((ph, ph), color, dtype=np.int8)
            if np.all(glyph == color):
                kind = "S"
            else:
                kind = "H"
                expected[1:-1, 1:-1] = base
                assert np.array_equal(glyph, expected), "one-cell hollow rim"
            sides = [side for side, yes in (("T", py == 0), ("B", py+ph == h),
                                            ("L", px == 0), ("R", px+ph == w)) if yes]
            assert len(sides) == 1, "one outward side per port"
            # Corners of panels are plain; symbols are clear of side endpoints.
            if sides[0] in ("T", "B"):
                assert 1 <= px and px+ph < w
            else:
                assert 1 <= py and py+ph < h
            p = {"panel": index, "y": py, "x": px, "side": sides[0],
                 "kind": kind, "color": color, "row": y+py, "col": x+px}
            local_ports.append(p)
            ports.append(p)
        assert local_ports, "each panel participates in docking"
        for i, p in enumerate(local_ports):
            for q in local_ports[:i]:
                assert (p["y"]+size < q["y"] or q["y"]+size < p["y"] or
                        p["x"]+size < q["x"] or q["x"]+size < p["x"]), "port gap"
    assert anchor is not None and len(panels) >= 2
    for i, p in enumerate(panels):
        for q in panels[:i]:
            y, x, h, w = p["top"], p["left"], *p["tile"].shape
            v, u, a, b = q["top"], q["left"], *q["tile"].shape
            assert y+h < v or v+a < y or x+w < u or u+b < x, "panels separated by background"
    hqueue = sorted((p for p in ports if p["kind"] == "H"), key=lambda p: (p["row"], p["col"]))
    squeue = sorted((p for p in ports if p["kind"] == "S"), key=lambda p: (p["row"], p["col"]))
    assert hqueue and len(hqueue) == len(squeue), "balanced port queues"
    return {"grid": g, "background": bg, "marker": marker, "point": point,
            "corner": corner, "carrier": carrier, "size": size, "panels": panels,
            "ports": ports, "anchor": anchor, "H": hqueue, "S": squeue}


def _assemble(a, pairs=None):
    """Propagate visible seam offsets, then verify occupied-cell closure."""
    pairs = list(zip(a["H"], a["S"])) if pairs is None else pairs
    panels, s = a["panels"], a["size"]
    graph = [[] for _ in panels]
    for p, q in pairs:
        i, j = p["panel"], q["panel"]
        assert i != j and _OPPOSITE[p["side"]] == q["side"], "opposed ports on different panels"
        dy, dx = {"T": (-s, 0), "B": (s, 0), "L": (0, -s), "R": (0, s)}[p["side"]]
        offset = (p["y"]+dy-q["y"], p["x"]+dx-q["x"])
        graph[i].append((j, offset))
        graph[j].append((i, (-offset[0], -offset[1])))
    anchor = a["anchor"]
    positions = [None]*len(panels)
    positions[anchor] = (panels[anchor]["top"], panels[anchor]["left"])
    queue = [anchor]
    for i in queue:
        y, x = positions[i]
        for j, (dy, dx) in graph[i]:
            proposed = (y+dy, x+dx)
            if positions[j] is None:
                positions[j] = proposed
                queue.append(j)
            else:
                assert positions[j] == proposed, "inconsistent docking cycle"
    assert all(p is not None for p in positions), "connected port graph"
    out = np.full(a["grid"].shape, a["background"], dtype=np.int8)
    occupied = np.zeros(out.shape, dtype=bool)
    for panel, (y, x) in zip(panels, positions):
        tile = panel["tile"]
        h, w = tile.shape
        assert 0 <= y and 0 <= x and y+h <= out.shape[0] and x+w <= out.shape[1], "assembly fits canvas"
        assert not occupied[y:y+h, x:x+w].any(), "no panel overlap"
        out[y:y+h, x:x+w] = tile
        occupied[y:y+h, x:x+w] = True
    q = np.argwhere(occupied)
    y, x = map(int, q.min(0))
    v, u = map(int, q.max(0)+1)
    assert occupied[y:v, x:u].all(), "gapless rectangular assembly"
    assert a["point"] == (v-1 if a["corner"][0] == "B" else y,
                          u-1 if a["corner"][1] == "R" else x), "marked outer corner"
    assert all(out[p] == a["background"] for p in _marker_cells(a["point"], a["corner"]))
    return out, positions, (y, x, v-y, u-x)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the input, using the family's fixed visible conventions."""
    return _assemble(_parse(input_grid))[0]


def _partition(rng, n, s):
    minimum = s+2
    factors = [(r, c) for r in range(1, 4) for c in range(1, 5)
               if r*c >= n and r*minimum <= 24 and c*minimum <= 24]
    r, c = factors[int(rng.integers(len(factors)))]
    h = int(rng.integers(r*minimum, min(24, r*minimum+6)+1))
    w = int(rng.integers(c*minimum, min(24, c*minimum+6)+1))
    rects = [(0, 0, h, w)]
    for _ in range(n-1):
        choices = []
        capacity = sum((hh//minimum)*(ww//minimum) for _, _, hh, ww in rects)
        for i, (_, _, hh, ww) in enumerate(rects):
            old_capacity = (hh//minimum)*(ww//minimum)
            for axis, length in enumerate((hh, ww)):
                for cut in range(minimum, length-minimum+1):
                    new_capacity = ((cut//minimum)+((length-cut)//minimum))*((ww if axis == 0 else hh)//minimum)
                    if capacity-old_capacity+new_capacity >= n:
                        choices.append((i, axis, cut))
        if not choices:
            return None
        i, axis, cut = choices[int(rng.integers(len(choices)))]
        y, x, hh, ww = rects.pop(i)
        if axis == 0:
            rects.extend([(y, x, cut, ww), (y+cut, x, hh-cut, ww)])
        else:
            rects.extend([(y, x, hh, cut), (y, x+cut, hh, ww-cut)])
    return rects, (h, w)


def _port_options(rects, s):
    edges = []
    for i, (y, x, h, w) in enumerate(rects):
        for j, (v, u, a, b) in enumerate(rects):
            if i >= j:
                continue
            if x+w == u or u+b == x:
                start, end = max(y, v), min(y+h, v+a)
                p, q = (i, j) if x+w == u else (j, i)
                options = [((p, t-rects[p][0], rects[p][3]-s, "R"),
                            (q, t-rects[q][0], 0, "L")) for t in range(start+1, end-s)]
            elif y+h == v or v+a == y:
                start, end = max(x, u), min(x+w, u+b)
                p, q = (i, j) if y+h == v else (j, i)
                options = [((p, rects[p][2]-s, t-rects[p][1], "B"),
                            (q, 0, t-rects[q][1], "T")) for t in range(start+1, end-s)]
            else:
                continue
            if options:
                edges.append((i, j, options))
    return edges


def _choose_ports(rng, rects, s, cyclic):
    edges, roots, result = _port_options(rects, s), list(range(len(rects))), []

    def root(i):
        while roots[i] != i:
            i = roots[i]
        return i

    def clear(p):
        i, y, x, _ = p
        return all(i != q[0] or y+s < q[1] or q[1]+s < y or x+s < q[2] or q[2]+s < x
                   for pair in result for q in pair)

    for k in rng.permutation(len(edges)):
        i, j, options = edges[int(k)]
        if root(i) == root(j) and not cyclic:
            continue
        for z in rng.permutation(len(options)):
            p, q = options[int(z)]
            if clear(p) and clear(q):
                result.append((p, q))
                roots[root(i)] = root(j)
                break
    if len({root(i) for i in range(len(rects))}) != 1:
        return None
    return result


def _scatter(rng, rects, shape, anchor, anchor_pos, markers):
    order = sorted((i for i in range(len(rects)) if i != anchor),
                   key=lambda i: -rects[i][2]*rects[i][3])
    for _ in range(5):
        positions = [None]*len(rects)
        positions[anchor] = anchor_pos
        for i in order:
            _, _, h, w = rects[i]
            yy, xx = np.indices((shape[0]-h-1, shape[1]-w-1))
            yy, xx = yy+1, xx+1
            legal = np.ones(yy.shape, bool)
            for j, p in enumerate(positions):
                if p is None:
                    continue
                v, u = p
                a, b = rects[j][2:]
                legal &= (yy+h < v) | (v+a < yy) | (xx+w < u) | (u+b < xx)
            for v, u in markers:
                legal &= (yy-1 > v) | (yy+h < v) | (xx-1 > u) | (xx+w < u)
            options = np.argwhere(legal)
            if not len(options):
                break
            y, x = options[int(rng.integers(len(options)))]
            positions[i] = (int(y)+1, int(x)+1)
        if all(p is not None for p in positions):
            return positions
    return None


def _polarities(pairs, positions, rng, allow_ties=True):
    order = sorted((positions[p[0]][0]+p[1], positions[p[0]][1]+p[2], i, k)
                   for i, pair in enumerate(pairs) for k, p in enumerate(pair))
    choices = []
    for tail in product(range(2), repeat=len(pairs)-1):
        bits = (0,)+tail
        first = [i for _, _, i, k in order if k == bits[i]]
        second = [i for _, _, i, k in order if k != bits[i]]
        if first == second:
            if not allow_ties:
                first_rows = [y for y, _, i, k in order if k == bits[i]]
                second_rows = [y for y, _, i, k in order if k != bits[i]]
                if len(set(first_rows)) != len(first_rows) or len(set(second_rows)) != len(second_rows):
                    continue
            choices.append(bits)
    if not choices:
        return None
    bits = choices[int(rng.integers(len(choices)))]
    return tuple(1-b for b in bits) if rng.integers(2) else bits


def _paint(rng, rects, pairs, positions, bits, shape, point, corner, s, witness):
    bg, carrier, marker, *available = map(int, rng.permutation(10))
    palette = available[:int(rng.integers(2, 4)) if witness else int(rng.integers(1, 4))]
    inks = rng.choice(palette, size=(len(pairs), 2)).astype(int)
    if witness:
        # Equality is deliberately false evidence: each of two colors appears
        # in both geometries, but the two first designated mates differ.
        for i in range(2):
            inks[i, bits[i]] = palette[i]
            inks[i, 1-bits[i]] = palette[1-i]
    tiles = [np.full((h, w), carrier, dtype=np.int8) for _, _, h, w in rects]
    for i, pair in enumerate(pairs):
        for k, (panel, y, x, _) in enumerate(pair):
            glyph = np.full((s, s), inks[i, k], dtype=np.int8)
            if k == bits[i]:
                glyph[1:-1, 1:-1] = carrier
            tiles[panel][y:y+s, x:x+s] = glyph
    g = np.full(shape, bg, dtype=np.int8)
    for tile, (y, x) in zip(tiles, positions):
        h, w = tile.shape
        g[y:y+h, x:x+w] = tile
    for p in _marker_cells(point, corner):
        g[p] = marker
    return g


def _matching_key(pairs):
    return frozenset(((p["panel"], p["y"], p["x"]), (q["panel"], q["y"], q["x"])) for p, q in pairs)


def _geometry_alternative(a):
    """A concrete port-blind answer: exchange two unanchored sibling regions.

    This is an ablation, not a second solver. It preserves the anchor, exact
    tiles and rectangular closure, while deliberately discarding ordinal seams.
    """
    actual, positions, _ = _assemble(a)
    for i, panel in enumerate(a["panels"]):
        if i == a["anchor"]:
            continue
        y, x = positions[i]
        h, w = panel["tile"].shape
        for j in range(i):
            if j == a["anchor"]:
                continue
            v, u = positions[j]
            hh, ww = a["panels"][j]["tile"].shape
            alternative = list(positions)
            if x == u and w == ww and (y+h == v or v+hh == y):
                top = min(y, v)
                alternative[i] = (top+hh if y < v else top, x)
                alternative[j] = (top if y < v else top+h, x)
            elif y == v and h == hh and (x+w == u or u+ww == x):
                left = min(x, u)
                alternative[i] = (y, left+ww if x < u else left)
                alternative[j] = (y, left if x < u else left+w)
            else:
                continue
            out = np.full(actual.shape, a["background"], dtype=np.int8)
            for p, (r, c) in zip(a["panels"], alternative):
                ph, pw = p["tile"].shape
                out[r:r+ph, c:c+pw] = p["tile"]
            if not np.array_equal(out, actual):
                return out
    return None


def _witnesses(a):
    actual = _matching_key(zip(a["H"], a["S"]))
    horizontal_pairs = list(zip(sorted(a["H"], key=lambda p: (p["col"], p["row"])),
                                sorted(a["S"], key=lambda p: (p["col"], p["row"]))))
    horizontal = _matching_key(horizontal_pairs) != actual
    # Repeatedly take the closest still-unpaired opposite shapes. It is a
    # concrete proximity alternative, not an exhaustive human-rule analysis.
    hh, ss, nearest = list(a["H"]), list(a["S"]), []
    while hh:
        _, i, j = min((abs(p["row"]-q["row"])+abs(p["col"]-q["col"]), i, j)
                      for i, p in enumerate(hh) for j, q in enumerate(ss))
        nearest.append((hh.pop(i), ss.pop(j)))
    proximity = _matching_key(nearest) != actual
    sequence = [p["kind"] for p in sorted(a["ports"], key=lambda p: (p["row"], p["col"]))]
    independent = any(p == q for p, q in zip(sequence, sequence[1:]))
    color = any(p["color"] != q["color"] for p, q in zip(a["H"], a["S"]))
    # Check that a changed pairing actually invalidates or changes the answer.
    for alternate_pairs, name in ((nearest, "proximity"), (horizontal_pairs, "horizontal")):
        try:
            alternate = _assemble(a, alternate_pairs)[0]
            differs = not np.array_equal(alternate, _assemble(a)[0])
            if name == "proximity":
                proximity = proximity and differs
            else:
                horizontal = horizontal and differs
        except AssertionError:
            pass
    axes = bool({p["side"] for p in a["ports"]} & {"T", "B"}) and bool({p["side"] for p in a["ports"]} & {"L", "R"})
    offset = any((p["y"] != q["y"] if p["side"] in ("L", "R") else p["x"] != q["x"])
                 for p, q in zip(a["H"], a["S"]))
    return {"horizontal": horizontal, "proximity": proximity,
            "independent": independent, "color": color, "axes": axes, "offset": offset,
            "geometry": _geometry_alternative(a) is not None}


def _fallback(rng, corner, witness):
    """Finite guaranteed four-panel construction, reflected as a whole.

    It is a construction of ports and rectangles, never a saved input/output.
    Both queue orders reverse under vertical reflection, preserving mates.
    """
    rects = [(0, 0, 9, 8), (0, 8, 8, 8), (9, 0, 7, 8), (8, 8, 8, 8)]
    pairs = [((0, 2, 5, "R"), (1, 2, 0, "L")),
             ((2, 3, 5, "R"), (3, 4, 0, "L")),
             ((0, 6, 1, "B"), (2, 0, 1, "T"))]
    positions = [(5, 5), (1, 16), (16, 1), (18, 18)]
    g = _paint(rng, rects, pairs, positions, (0, 0, 0), (28, 28), (5, 5), "TL", 3, witness)
    if corner[0] == "B":
        g = g[::-1].copy()
    if corner[1] == "R":
        g = g[:, ::-1].copy()
    return {"input": g, "output": solve(g)}


def _generate_pair(rng, n, s, corner, witness, allow_ties=True):
    for _ in range(144):
        partition = _partition(rng, n, s)
        if partition is None:
            continue
        rects, target = partition
        pairs = _choose_ports(rng, rects, s, cyclic=rng.random() < .3)
        if pairs is None:
            continue
        h, w = target
        spread = min(30, int(np.ceil(np.sqrt(h*w*1.8)))-2)
        lower = (max(h+4, spread), max(w+4, spread))
        tight = rng.random() < .3
        shape = tuple(int(rng.integers(low, min(30, low+3)+1 if tight else 31)) for low in lower)
        origin = (int(rng.integers(2, shape[0]-h-1)), int(rng.integers(2, shape[1]-w-1)))
        local = (h-1 if corner[0] == "B" else 0, w-1 if corner[1] == "R" else 0)
        anchor = next(i for i, (y, x, hh, ww) in enumerate(rects) if y <= local[0] < y+hh and x <= local[1] < x+ww)
        point = (origin[0]+local[0], origin[1]+local[1])
        anchor_pos = (origin[0]+rects[anchor][0], origin[1]+rects[anchor][1])
        for _ in range(16):
            positions = _scatter(rng, rects, shape, anchor, anchor_pos, _marker_cells(point, corner))
            if positions is None:
                continue
            bits = _polarities(pairs, positions, rng, allow_ties)
            if bits is None:
                continue
            g = _paint(rng, rects, pairs, positions, bits, shape, point, corner, s, witness)
            a = _parse(g)
            if witness and not all(_witnesses(a).values()):
                continue
            out, recovered, _ = _assemble(a)
            # Compare with the independently retained geometric construction;
            # the emitted label itself comes entirely from the visible input.
            expected = np.full(shape, a["background"], dtype=np.int8)
            for panel, old in zip(a["panels"], recovered):
                original_index = positions.index((panel["top"], panel["left"]))
                y, x = origin[0]+rects[original_index][0], origin[1]+rects[original_index][1]
                assert old == (y, x)
                hh, ww = panel["tile"].shape
                expected[y:y+hh, x:x+ww] = panel["tile"]
            assert np.array_equal(out, expected)
            return {"input": g, "output": out}
    return _fallback(rng, corner, witness)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _has_tie(a):
    return any(len({p["row"] for p in a[kind]}) < len(a[kind]) for kind in ("H", "S"))


def _tie_witness(a):
    if not _has_tie(a):
        return False
    alternate = list(zip(sorted(a["H"], key=lambda p: (p["row"], -p["col"])),
                         sorted(a["S"], key=lambda p: (p["row"], -p["col"]))))
    if _matching_key(alternate) == _matching_key(zip(a["H"], a["S"])):
        return False
    try:
        return not np.array_equal(_assemble(a, alternate)[0], _assemble(a)[0])
    except AssertionError:
        return True


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    corners = list(rng.permutation(_CORNERS))
    varying_size = rng.random() < .75
    initial_size = int(rng.integers(3, 5))
    puzzle, seen, taught_ties = {"train": [], "test": []}, set(), False
    for i in range(num_train+num_test):
        training = i < num_train
        corner = str(corners[i % 2] if training else corners[2+(i-num_train) % 2])
        s = 7-initial_size if varying_size and i == 1 else (int(rng.integers(3, 5)) if varying_size and i > 1 else initial_size)
        n = int(rng.integers(4, 6)) if i == 0 else int(rng.integers(3, 7))
        for _ in range(12):
            pair = _generate_pair(rng, n, s, corner, witness=i < 2, allow_ties=training or taught_ties)
            key = (pair["input"].shape, pair["input"].tobytes())
            analysis = _parse(pair["input"])
            has_tie = _has_tie(analysis)
            if key not in seen and (training or taught_ties or not has_tie):
                break
        else:
            # The geometric fallback has distinct top rows in both queues.
            pair = _fallback(rng, corner, witness=i < 2)
            for shift in range(10):
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    break
                pair = {k: ((g.astype(np.int16)+1) % 10).astype(np.int8) for k, g in pair.items()}
            assert key not in seen
            has_tie = False
        seen.add(key)
        if training:
            taught_ties |= _tie_witness(_parse(pair["input"]))
        puzzle["train" if training else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = (inp.shape, inp.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            a = _parse(inp)
            expected, positions, bbox = _assemble(a)
            assert np.array_equal(out, expected), "incorrect anchored assembly"
            assert 3 <= len(a["panels"]) <= 6 and a["size"] in (3, 4)
            assert not np.array_equal(inp, out)
            before, after = np.bincount(inp.ravel(), minlength=10), np.bincount(out.ravel(), minlength=10)
            assert before[a["marker"]] == 3 and after[a["marker"]] == 0
            assert after[a["background"]] == before[a["background"]]+3
            assert all(before[c] == after[c] for c in range(10) if c not in (a["background"], a["marker"]))
            assert sum(p["tile"].size for p in a["panels"]) == bbox[2]*bbox[3]
            assert positions[a["anchor"]] == (a["panels"][a["anchor"]]["top"], a["panels"][a["anchor"]]["left"])
            (training if split == "train" else testing).append(a)
    witnesses = [_witnesses(a) for a in training]
    for name in ("horizontal", "proximity", "independent", "color", "axes", "offset", "geometry"):
        assert any(w[name] for w in witnesses), f"missing {name} training witness"
    assert len({a["corner"] for a in training}) == 2
    assert {a["corner"] for a in training}.isdisjoint(a["corner"] for a in testing), "corner holdout"
    assert not any(_has_tie(a) for a in testing) or any(_tie_witness(a) for a in training), "untaught top-row tie convention"


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for key in ("input", "output"):
                g = np.asarray(pair[key], dtype=np.int8)
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def _features(pair):
    a = _parse(pair["input"])
    _, positions, bbox = _assemble(a)
    degrees = [0]*len(a["panels"])
    for p in a["ports"]:
        degrees[p["panel"]] += 1
    normalized = []
    for p, (y, x) in zip(a["panels"], positions):
        normalized.append((y-bbox[0], x-bbox[1], *p["tile"].shape))
    return {"count": len(degrees), "edges": len(a["H"]), "size": a["size"],
            "shape": a["grid"].shape, "target": bbox[2:], "background": a["background"],
            "corner": a["corner"], "branch": max(degrees) >= 3,
            "cycle": len(a["H"]) >= len(degrees),
            "tie": _has_tie(a),
            "unequal": len({p["tile"].shape for p in a["panels"]}) > 1,
            "partition": tuple(sorted(normalized)), "witnesses": _witnesses(a)}


def self_test():
    """42 seeds, compact aggregates only; bounded memory, no disk writes."""
    signatures, features, within = set(), [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        signatures.add(_signature(puzzle))
        local = [_features(p) for split in ("train", "test") for p in puzzle[split]]
        within.append(local)
        features.extend(local)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                copied = pair["input"].copy()
                assert np.array_equal(solve(copied), pair["output"])
                assert np.array_equal(copied, pair["input"]), "solve mutated input"
    assert len(signatures) == 42
    assert {f["size"] for f in features} == {3, 4}
    assert {f["count"] for f in features} == {3, 4, 5, 6}
    assert {f["corner"] for f in features} == set(_CORNERS)
    assert {f["background"] for f in features} == set(range(10))
    assert len({f["shape"] for f in features}) >= 25
    assert len({f["target"] for f in features}) >= 40
    assert len({f["partition"] for f in features}) >= 100
    assert sum(f["branch"] for f in features) >= 10
    assert any(f["cycle"] for f in features)
    assert any(f["tie"] for f in features)
    assert sum(f["unequal"] for f in features) >= 100
    assert sum(len({f["count"] for f in group}) >= 2 for group in within) >= 30
    # All legal explicit count requests and illegal types/bounds are exercised.
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(19, train, test)
            assert len(p["train"]) == train and len(p["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for corner in _CORNERS:
        pair = _fallback(np.random.default_rng(41), corner, True)
        a = _parse(pair["input"])
        assert all(_witnesses(a).values())
        alternative = _geometry_alternative(a)
        assert alternative is not None and not np.array_equal(alternative, pair["output"])
        assert np.array_equal(np.bincount(alternative.ravel(), minlength=10),
                              np.bincount(pair["output"].ravel(), minlength=10))
        assert np.array_equal(alternative != a["background"], pair["output"] != a["background"])
        assert np.array_equal(solve(pair["input"]), pair["output"])
    # A valid input with an inset anchor may assemble right up to the far edge.
    a = _parse(_fallback(np.random.default_rng(71), "TL", True)["input"])
    edge_input = np.full((30, 21), a["background"], dtype=np.int8)
    relocation = {(5, 5): (10, 5), (1, 16): (1, 1), (16, 1): (20, 1), (18, 18): (20, 12)}
    for panel in a["panels"]:
        y, x = relocation[panel["top"], panel["left"]]
        h, w = panel["tile"].shape
        edge_input[y:y+h, x:x+w] = panel["tile"]
    for p in _marker_cells((10, 5), "TL"):
        edge_input[p] = a["marker"]
    edge_output = solve(edge_input)
    assert np.any(edge_output[:, -1] != a["background"])
    assert np.count_nonzero(edge_output != a["background"]) == sum(p["tile"].size for p in a["panels"])
    # Negative controls: validator must reject loss of exact contents, anchor,
    # marker erasure, or distinctness. No tests mirror just a sampling label.
    original = generate_puzzle(0)
    for corruption in ("pixel", "marker", "translation", "rank", "duplicate"):
        damaged = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
        p = damaged["train"][0] if corruption == "rank" else damaged["test"][0]
        a = _parse(p["input"])
        if corruption == "pixel":
            y, x = np.argwhere(p["output"] != a["background"])[0]
            p["output"][y, x] = (int(p["output"][y, x])+1) % 10
        elif corruption == "marker":
            p["output"][_marker_cells(a["point"], a["corner"])[0]] = a["marker"]
        elif corruption == "translation":
            p["output"] = np.roll(p["output"], 1, axis=0)
        elif corruption == "rank":
            p["output"] = _geometry_alternative(a)
            assert p["output"] is not None
        else:
            damaged["train"][1] = damaged["train"][0]
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"accepted {corruption} corruption")
    return {"seeds": 42, "pairs": len(features), "partitions": len({f["partition"] for f in features}),
            "canvas_shapes": len({f["shape"] for f in features}), "target_shapes": len({f["target"] for f in features}),
            "branched": sum(f["branch"] for f in features), "cyclic": sum(f["cycle"] for f in features),
            "tied_rows": sum(f["tie"] for f in features)}
