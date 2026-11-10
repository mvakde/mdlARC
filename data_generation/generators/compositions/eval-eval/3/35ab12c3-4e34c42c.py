"""Registered observations: consensus landmarks reveal a drawing and its echo.

Only the standard library and NumPy are used. The solver reads the cards,
registration glyphs, votes, and marker from pixels; it knows no sampled state.
"""
from itertools import permutations, product
import numpy as np


OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0, 762524043, 215190380, 571148513, 825748497, 567098791,
              455147067, 259221807, 818642173, 212478325, 546234476,
              372416237, 176675378, 239871831, 292327560, 720561726,
              169034981, 936700622, 173788017, 210406638, 277413393,
              427754654, 716235101, 55657388, 451460720, 141390382,
              953244512, 922468194, 556930429, 531682454, 933837835,
              361056934, 182424820, 282182082, 250229230, 274528163,
              845983385, 180364433, 310265235, 149437453, 330287556,
              637688482)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _groups(points):
    remaining, result = set(points), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _bounds(points):
    return (min(r for r, c in points), min(c for r, c in points),
            max(r for r, c in points), max(c for r, c in points))


def _signature(cells):
    r0, c0, _, _ = _bounds(cells)
    return tuple(sorted((r-r0, c-c0, int(v)) for (r, c), v in cells.items()))


def _aligned(a, b):
    dr, dc = abs(a[0]-b[0]), abs(a[1]-b[1])
    return max(dr, dc) > 0 and (dr == 0 or dc == 0 or dr == dc)


def _line(a, b):
    if not _aligned(a, b):
        raise ValueError("unaligned endpoints")
    dr, dc = b[0]-a[0], b[1]-a[1]
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return {(a[0]+i*sr, a[1]+i*sc)
            for i in range(max(abs(dr), abs(dc))+1)}


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def _intersect(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0])
            or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1])
            or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return _cross(a, b, c)*_cross(a, b, d) <= 0 and _cross(c, d, a)*_cross(c, d, b) <= 0


def _topology(points):
    """A unique simple full cycle wins; otherwise require a unique full path.

    Edge sets, not enumeration order, identify solutions. Continuous crossings
    are forbidden, including diagonals crossing between pixel centers.
    """
    points = sorted(points)
    n = len(points)
    if not 2 <= n <= 10:
        raise ValueError("need 2..10 consensus vertices")
    vertices = set(points)
    edges = [(i, j) for i in range(n) for j in range(i+1, n)
             if _aligned(points[i], points[j])
             and not (_line(points[i], points[j])-{points[i], points[j]}) & vertices]
    adjacency = [[] for _ in points]
    for k, (i, j) in enumerate(edges):
        adjacency[i].append((j, k))
        adjacency[j].append((i, k))
    incompatible = [set() for _ in edges]
    for k, (i, j) in enumerate(edges):
        for l, (u, v) in enumerate(edges[:k]):
            if not {i, j} & {u, v} and _intersect(points[i], points[j], points[u], points[v]):
                incompatible[k].add(l)
                incompatible[l].add(k)

    def enumerate_orders(closed):
        solutions, visits = set(), 0

        def walk(order, used, chosen):
            nonlocal visits
            visits += 1
            if visits > 50000:
                raise ValueError("ambiguous topology exceeds search bound")
            if len(solutions) > 1:
                return
            if len(order) == n:
                if closed:
                    closing = [k for j, k in adjacency[order[-1]] if j == order[0]]
                    if not closing or incompatible[closing[0]] & chosen:
                        return
                    chosen = chosen | {closing[0]}
                solutions.add(frozenset(chosen))
                return
            for j, k in adjacency[order[-1]]:
                if j not in used and not incompatible[k] & chosen:
                    walk(order+[j], used | {j}, chosen | {k})

        for start in (range(1) if closed else range(n)):
            walk([start], {start}, set())
            if len(solutions) > 1:
                break
        return solutions

    cycles = enumerate_orders(True) if n >= 3 else set()
    if len(cycles) > 1:
        raise ValueError("competing full cycles")
    solutions = cycles or enumerate_orders(False)
    if len(solutions) != 1:
        raise ValueError("no unique full drawing")
    selected = sorted(next(iter(solutions)))
    segments = [(points[edges[k][0]], points[edges[k][1]]) for k in selected]
    pixels = set().union(*(_line(a, b) for a, b in segments))
    # In this family every landmark is isolated; there are no unit-only stubs.
    if any(max(abs(a[0]-b[0]), abs(a[1]-b[1])) < 2 for a, b in segments):
        raise ValueError("landmarks must be separated")
    all_aligned = set().union(*(_line(points[i], points[j]) for i, j in edges))
    return {"vertices": points, "segments": segments, "pixels": pixels,
            "kind": "cycle" if cycles else "path",
            "chord_pixels": all_aligned-pixels}


def _read_cards(grid):
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if counts.max()*2 <= grid.size:
        raise ValueError("background must occupy over half the input")
    borders = {}
    for color in values:
        if color == bg:
            continue
        rectangles = []
        for group in _groups(map(tuple, np.argwhere(grid == color))):
            r0, c0, r1, c1 = _bounds(group)
            perimeter = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)
                         if r in (r0, r1) or c in (c0, c1)}
            if r1-r0 >= 3 and c1-c0 >= 3 and group == perimeter:
                rectangles.append((r0, c0, r1, c1))
        if len(rectangles) >= 2:
            borders[int(color)] = rectangles
    if len(borders) != 1:
        raise ValueError("need one shared frame color")
    frame, rectangles = next(iter(borders.items()))
    cards, covered = [], set()
    for r0, c0, r1, c1 in sorted(rectangles):
        area = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)}
        if covered & area:
            raise ValueError("overlapping observation cards")
        covered.update(area)
        content = {(r-r0-1, c-c0-1): int(grid[r, c])
                   for r in range(r0+1, r1) for c in range(c0+1, c1)
                   if grid[r, c] != bg}
        if not content or frame in content.values():
            raise ValueError("empty card or interior frame ink")
        glyphs, small = [], {}
        for group in _groups(content):
            cells = {p: content[p] for p in group}
            if len(group) >= 3:
                if len(set(cells.values())) < 2:
                    raise ValueError("glyphs must be multicolor")
                glyphs.append((_signature(cells), _bounds(group)[:2]))
            else:
                small.update(cells)
        if not glyphs or not small:
            raise ValueError("each card needs glyphs and observations")
        cards.append({"glyphs": glyphs, "small": small,
                      "origin": (r0+1, c0+1), "shape": (r1-r0-1, c1-c0-1)})
    if any(tuple(p) not in covered for p in np.argwhere(grid != bg)):
        raise ValueError("ink outside the cards")
    if not 2 <= len(cards) <= 6:
        raise ValueError("need 2..6 cards")
    return bg, frame, cards


def _registered_evidence(grid):
    bg, frame, cards = _read_cards(grid)
    occurrences = {}
    glyph_colors = set()
    for i, card in enumerate(cards):
        local = set()
        for signature, anchor in card["glyphs"]:
            if signature in local:
                raise ValueError("a glyph cannot repeat inside one card")
            local.add(signature)
            occurrences.setdefault(signature, []).append((i, anchor))
            glyph_colors.update(v for r, c, v in signature)
    adjacency = [[] for _ in cards]
    for copies in occurrences.values():
        if len(copies) < 2:
            raise ValueError("every glyph must be shared")
        for (i, a), (j, b) in zip(copies, copies[1:]):
            delta = (a[0]-b[0], a[1]-b[1])
            adjacency[i].append((j, delta))
            adjacency[j].append((i, (-delta[0], -delta[1])))
    offsets, stack = {0: (0, 0)}, [0]
    while stack:
        i = stack.pop()
        for j, (dr, dc) in adjacency[i]:
            candidate = (offsets[i][0]+dr, offsets[i][1]+dc)
            if j in offsets:
                if offsets[j] != candidate:
                    raise ValueError("inconsistent registration cycle")
            else:
                offsets[j] = candidate
                stack.append(j)
    if len(offsets) != len(cards):
        raise ValueError("disconnected glyph graph")
    small_counts = {}
    for card in cards:
        for color in card["small"].values():
            small_counts[color] = small_counts.get(color, 0)+1
    source = max(small_counts, key=small_counts.get)
    if (list(small_counts.values()).count(small_counts[source]) != 1
            or source in glyph_colors or frame in glyph_colors
            or bg in glyph_colors):
        raise ValueError("unclear or overlapping ink roles")
    others = set(small_counts)-{source}
    if len(others) > 1 or any(small_counts[c] != 1 or c in glyph_colors for c in others):
        raise ValueError("at most one differently colored marker is allowed")
    votes, marker, seen_ink = {}, None, {}
    for i, card in enumerate(cards):
        dr, dc = offsets[i]
        for (r, c), color in card["small"].items():
            p = (r+dr, c+dc)
            if p in seen_ink and seen_ink[p] != color:
                raise ValueError("conflicting observations")
            seen_ink[p] = color
            if color == source:
                votes.setdefault(p, set()).add(i)
            else:
                marker = (p, color, i)
    vertices = {p for p, voters in votes.items() if len(voters) >= 2}
    rejected = set(votes)-vertices
    topology = _topology(vertices)
    copy_pixels, displacement = set(), None
    if marker is not None:
        point, copy_color, carrier = marker
        anchors = [p for p in vertices if max(abs(point[0]-p[0]), abs(point[1]-p[1])) == 1]
        if len(anchors) != 1 or point in topology["pixels"]:
            raise ValueError("marker must offset exactly one confirmed vertex")
        anchor = anchors[0]
        displacement = (point[0]-anchor[0], point[1]-anchor[1])
        copy_pixels = {(r+displacement[0], c+displacement[1]) for r, c in topology["pixels"]}
        topology["marker_remote"] = carrier not in votes[anchor]
    else:
        copy_color = None
        topology["marker_remote"] = False
    return {"background": bg, "frame": frame, "cards": cards,
            "offsets": offsets, "glyphs": occurrences, "source": source,
            "votes": votes, "rejected": rejected, "marker": marker,
            "copy_color": copy_color, "copy_pixels": copy_pixels,
            "displacement": displacement, **topology}


def _render_evidence(info, complete=True, echo=True, include_rejected=False):
    source = info["pixels"] if complete else set(info["vertices"])
    cells = {}
    if echo and info["marker"] is not None:
        cells.update({p: info["copy_color"] for p in info["copy_pixels"]})
    cells.update({p: info["source"] for p in source})
    if include_rejected:
        cells.update({p: info["source"] for p in info["rejected"]})
    r0, c0, r1, c1 = _bounds(cells)
    if max(r1-r0+1, c1-c0+1) > 30:
        raise ValueError("completed picture exceeds ARC bounds")
    result = np.full((r1-r0+1, c1-c0+1), info["background"], dtype=np.int8)
    for (r, c), color in cells.items():
        result[r-r0, c-c0] = color
    return result


def _normalized_points(points):
    if not points:
        return frozenset()
    r0, c0, _, _ = _bounds(points)
    return frozenset((r-r0, c-c0) for r, c in points)


def _frame_corner_support(info):
    """A concrete B ablation: align card interior corners instead of glyphs."""
    votes = {}
    for card in info["cards"]:
        for p, color in card["small"].items():
            if color == info["source"]:
                votes[p] = votes.get(p, 0)+1
    return {p for p, count in votes.items() if count >= 2}


def _alternative_answer(info, vertices, marker_point=None):
    """Complete a specified shortcut's vertices, or report an undefined rule."""
    try:
        alternate = {**info, **_topology(vertices)}
        alternate["copy_pixels"] = set()
        if info["marker"] is not None:
            point = info["marker"][0] if marker_point is None else marker_point
            anchors = [p for p in vertices
                       if max(abs(point[0]-p[0]), abs(point[1]-p[1])) == 1]
            if len(anchors) != 1 or point in alternate["pixels"]:
                return None
            dr, dc = point[0]-anchors[0][0], point[1]-anchors[0][1]
            alternate["copy_pixels"] = {(r+dr, c+dc) for r, c in alternate["pixels"]}
        return _render_evidence(alternate)
    except ValueError:
        return None


def _frame_corner_answer(info):
    marker_point = None
    if info["marker"] is not None:
        carrier = info["marker"][2]
        marker_point = next(p for p, color in info["cards"][carrier]["small"].items()
                            if color == info["copy_color"])
    return _alternative_answer(info, _frame_corner_support(info), marker_point)


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _render_evidence(_registered_evidence(_grid(input_grid)))


def _transform(points, code):
    result = []
    for r, c in points:
        if code & 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        result.append((r, c))
    r0, c0, _, _ = _bounds(result)
    return [(r-r0, c-c0) for r, c in result]


def _geometry(rng, requested=None):
    for _ in range(180):
        kind = requested or str(rng.choice(["cycle", "path", "segment"], p=[.5, .4, .1]))
        if kind == "cycle":
            h, w = map(int, rng.integers(5, 10, size=2))
            style = int(rng.integers(4))
            if style == 0:
                points = [(0, 0), (0, w), (h, w), (h, 0)]
            elif style == 1:
                d = int(rng.integers(2, 5))
                points = [(0, d), (d, 2*d), (2*d, d), (d, 0)]
            elif style == 2:
                # Unequal corner clips permit six- and eight-vertex contours.
                cuts = [int(rng.integers(0, 3)) for _ in range(4)]
                a, b, c, d = cuts
                points = [(0, a), (0, w-b), (b, w), (h-c, w),
                          (h, w-c), (h, d), (h-d, 0), (a, 0)]
                points = [p for i, p in enumerate(points) if p != points[i-1]]
            else:
                a, b = 2, int(rng.integers(2, h-1))
                points = [(0, 0), (0, w), (h, w), (h, w-a),
                          (h-b, w-a), (h-b, a), (h, a), (h, 0)]
        else:
            count = 2 if kind == "segment" else int(rng.integers(3, 7))
            points = [(0, 0)]
            for _ in range(count-1):
                dr, dc = OFFSETS[int(rng.integers(8))]
                length = int(rng.integers(2, 7))
                points.append((points[-1][0]+dr*length, points[-1][1]+dc*length))
        if len(set(points)) != len(points):
            continue
        points = _transform(points, int(rng.integers(8)))
        if max(_bounds(points)[2:]) > 9:
            continue
        if any(max(abs(a[0]-b[0]), abs(a[1]-b[1])) < 2
               for i, a in enumerate(points) for b in points[i+1:]):
            continue
        try:
            info = _topology(points)
        except ValueError:
            continue
        expected = {frozenset((points[i], points[i+1])) for i in range(len(points)-1)}
        if kind == "cycle":
            expected.add(frozenset((points[-1], points[0])))
        if {frozenset(e) for e in info["segments"]} != expected:
            continue
        if requested == "cycle" and not info["chord_pixels"]:
            continue
        return info
    points = [(0, 0), (0, 6), (6, 6), (6, 0)] if requested == "cycle" else [(0, 0), (0, 6), (3, 9)]
    return _topology(points)


def _assign_votes(rng, vertices, count, triple):
    for _ in range(150):
        memberships = {p: set(map(int, rng.choice(count, 2, replace=False))) for p in vertices}
        if triple and count >= 3 and len(vertices) >= 4:
            p = vertices[int(rng.integers(len(vertices)))]
            memberships[p] = set(map(int, rng.choice(count, 3, replace=False)))
        sizes = [sum(i in owners for owners in memberships.values()) for i in range(count)]
        if min(sizes) >= 1 and (count == 2 or max(sizes) < len(vertices)):
            return memberships
    # Balanced pairs provide a finite fallback even for a two-vertex segment.
    memberships = {p: {i % count, (i+1) % count} for i, p in enumerate(vertices)}
    if set().union(*memberships.values()) != set(range(count)):
        raise ValueError("not enough vertices for this card count")
    return memberships


def _glyph(rng, colors, used):
    shapes = (((0, 0), (0, 1), (0, 2)),
              ((0, 0), (1, 0), (2, 0)),
              ((0, 0), (0, 1), (1, 0)),
              ((0, 0), (0, 1), (1, 0), (1, 1)),
              ((0, 0), (0, 1), (0, 2), (1, 1)),
              ((0, 0), (1, 0), (2, 0), (2, 1)),
              tuple(product(range(2), range(3))))
    for _ in range(100):
        shape = shapes[int(rng.integers(len(shapes)))]
        cells = {p: int(rng.choice(colors)) for p in shape}
        signature = _signature(cells)
        if len(set(cells.values())) >= 2 and signature not in used:
            used.add(signature)
            return cells
    raise ValueError("glyph identity sampling exhausted")


def _halo(points):
    return {(r+dr, c+dc) for r, c in points for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _pack(cards, bg, rng):
    """Random separated rectangles, with a finite shelf-packing fallback."""
    n = len(cards)
    shapes = [a.shape for a in cards]
    for attempt in range(32):
        h = int(rng.integers(max(ph for ph, pw in shapes), 31))
        w = int(rng.integers(max(pw for ph, pw in shapes), 31))
        if attempt >= 16:
            h = w = 30
        blocked = np.zeros((h, w), dtype=bool)
        order = sorted(range(n), key=lambda i: -cards[i].size)
        placements = {}
        for i in order:
            ph, pw = shapes[i]
            options = [(r, c) for r in range(h-ph+1) for c in range(w-pw+1)
                       if not blocked[r:r+ph, c:c+pw].any()]
            if not options:
                break
            r, c = options[int(rng.integers(len(options)))]
            placements[i] = (r, c)
            blocked[max(0, r-1):min(h, r+ph+1), max(0, c-1):min(w, c+pw+1)] = True
        if len(placements) == n:
            break
    else:
        layouts = []
        for order in permutations(range(n)):
            for mask in range(1 << (n-1)):
                rows, row = [], [order[0]]
                for k in range(n-1):
                    if mask & (1 << k):
                        rows.append(row)
                        row = []
                    row.append(order[k+1])
                rows.append(row)
                heights = [max(shapes[i][0] for i in row) for row in rows]
                widths = [sum(shapes[i][1] for i in row)+len(row)-1 for row in rows]
                if sum(heights)+len(rows)-1 <= 30 and max(widths) <= 30:
                    layouts.append((rows, heights, widths))
        if not layouts:
            raise ValueError("cards do not fit")
        rows, heights, widths = layouts[int(rng.integers(len(layouts)))]
        base_h, base_w = sum(heights)+len(rows)-1, max(widths)
        h = base_h+int(rng.integers(31-base_h))
        w = base_w+int(rng.integers(31-base_w))
        r = int(rng.integers(h-base_h+1))
        placements = {}
        for row, rh, rw in zip(rows, heights, widths):
            c = int(rng.integers(w-rw+1))
            for i in row:
                placements[i] = (r+int(rng.integers(rh-shapes[i][0]+1)), c)
                c += shapes[i][1]+1
            r += rh+1
    result = np.full((h, w), bg, dtype=np.int8)
    for i, (r, c) in placements.items():
        ph, pw = shapes[i]
        result[r:r+ph, c:c+pw] = cards[i]
    return result


def _make_pair(rng, count, kind=None, offset=None, remote=False, triple=False):
    geometry = _geometry(rng, kind)
    vertices = geometry["vertices"]
    if count == 4 and len(vertices) < 4:
        raise ValueError("query needs distributed landmarks")
    memberships = _assign_votes(rng, vertices, count, triple)
    if triple and not any(len(x) == 3 for x in memberships.values()):
        raise ValueError("triple consensus witness missing")
    palette = list(map(int, rng.permutation(10)))
    bg, frame, source, companion = palette[:4]
    glyph_colors = palette[4:4+int(rng.integers(2, 5))]
    observations = [{p: source for p in vertices if i in memberships[p]} for i in range(count)]
    marker, copy_pixels = None, set()
    marked = offset is not None
    if marked:
        choices = []
        for anchor in vertices:
            point = (anchor[0]+offset[0], anchor[1]+offset[1])
            copied = {(r+offset[0], c+offset[1]) for r, c in geometry["pixels"]}
            if (point in geometry["pixels"]
                    or sum(max(abs(point[0]-p[0]), abs(point[1]-p[1])) == 1 for p in vertices) != 1
                    or len(copied-geometry["pixels"]) < 3
                    or (remote and not copied & (geometry["pixels"]-set(vertices)))):
                continue
            carriers = set(range(count))-memberships[anchor] if remote else memberships[anchor]
            for carrier in sorted(carriers):
                choices.append((point, carrier, copied))
        if not choices:
            raise ValueError("no observable copy marker")
        point, carrier, copy_pixels = choices[int(rng.integers(len(choices)))]
        observations[carrier][point] = companion
        marker = point
    # Tree keys link cards without any universal registration badge. A redundant
    # graph edge can introduce a cycle; all copies still encode the same plane.
    order = list(map(int, rng.permutation(count)))
    edges = [(order[i], order[int(rng.integers(i))]) for i in range(1, count)]
    if count >= 3 and rng.random() < .4:
        unused = [(i, j) for i in range(count) for j in range(i+1, count)
                  if {i, j} not in [set(e) for e in edges]]
        if unused:
            edges.append(unused[int(rng.integers(len(unused)))])
    h, w = _bounds(vertices)[2:]
    # Registration objects can lie inside or outside the drawing. The plane is
    # compact enough that four differently cropped observation cards can fit.
    domain = [(r, c) for r in range(-1, max(h+2, 7)) for c in range(-1, max(w+2, 7))]
    blocked = _halo(set(vertices) | ({marker} if marker is not None else set()))
    used = set()
    for i, j in edges:
        glyph = _glyph(rng, glyph_colors, used)
        locations = []
        for dr, dc in domain:
            shifted = {(r+dr, c+dc): v for (r, c), v in glyph.items()}
            if not set(shifted) & blocked and set(shifted) <= set(domain):
                locations.append(shifted)
        if not locations:
            raise ValueError("glyph plane crowded")
        cells = locations[int(rng.integers(len(locations)))]
        observations[i].update(cells)
        observations[j].update(cells)
        blocked.update(_halo(cells))
    forbidden = geometry["pixels"] | copy_pixels
    # Singleton faults never coincide across views and never hide on an output
    # stroke. Thus union instead of consensus is visibly wrong in every pair.
    for i in range(count):
        candidates = [p for p in domain if p not in blocked and p not in forbidden]
        rng.shuffle(candidates)
        for _ in range(int(rng.integers(1, 3))):
            candidates = [p for p in candidates if p not in blocked]
            if not candidates:
                raise ValueError("no isolated fault location")
            p = candidates.pop()
            observations[i][p] = source
            blocked.update(_halo({p}))
    cards = []
    for cells in observations:
        r0, c0, r1, c1 = _bounds(cells)
        ih, iw = r1-r0+1, c1-c0+1
        # Unequal margins change each card's local coordinate origin, so aligning
        # top-left frame corners does not substitute for glyph registration.
        top, left = int(rng.integers(2)), int(rng.integers(2))
        bottom, right = int(rng.integers(2)), int(rng.integers(2))
        if ih+top+bottom > 12 or iw+left+right > 12:
            raise ValueError("observation card too large")
        card = np.full((ih+top+bottom+2, iw+left+right+2), bg, dtype=np.int8)
        card[0, :] = card[-1, :] = frame
        card[:, 0] = card[:, -1] = frame
        for (r, c), color in cells.items():
            card[r-r0+top+1, c-c0+left+1] = color
        cards.append(card)
    inp = _pack(cards, bg, rng)
    info = _registered_evidence(inp)
    # Compare inferred geometric content to the proposed shape without relying
    # on its absolute origin. Labels are rasterized from the proposal separately.
    proposed = {p: source for p in geometry["pixels"]}
    if marked:
        proposed = {**{p: companion for p in copy_pixels}, **proposed}
    r0, c0, r1, c1 = _bounds(proposed)
    out = np.full((r1-r0+1, c1-c0+1), bg, dtype=np.int8)
    for (r, c), color in proposed.items():
        out[r-r0, c-c0] = color
    if not np.array_equal(_render_evidence(info), out):
        raise ValueError("input evidence disagrees with proposed drawing")
    if len(set(info["offsets"].values())) < 2:
        raise ValueError("frame-alignment shortcut not excluded")
    if _normalized_points(_frame_corner_support(info)) == _normalized_points(info["vertices"]):
        raise ValueError("frame corners accidentally reproduce the consensus")
    wrong_registration = _frame_corner_answer(info)
    if wrong_registration is not None and np.array_equal(wrong_registration, out):
        raise ValueError("frame registration accidentally reproduces the answer")
    if remote and not info["marker_remote"]:
        raise ValueError("remote marker witness lost")
    return {"input": inp, "output": out}


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    nq = _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    demo_cycle, demo_path = map(int, rng.choice(nt, 2, replace=False))
    directions = list(map(int, rng.choice(8, 2, replace=False)))
    held = int(rng.choice([i for i in range(8) if i not in directions]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            witness = split == "train" and index == demo_cycle
            path_demo = split == "train" and index == demo_path
            n_cards = 4 if split == "test" else 3 if witness else int(rng.integers(2, 4))
            kind = "cycle" if witness else "path" if path_demo else None
            direction = directions[0] if witness else directions[1] if path_demo else (
                held if split == "test" and index == 0 else int(rng.choice(
                    [i for i in range(8) if split == "test" or i != held])))
            offset = OFFSETS[direction] if witness or path_demo or split == "test" or rng.random() < .65 else None
            triple = witness or (n_cards >= 3 and rng.random() < .4)
            for attempt in range(260):
                try:
                    pair = _make_pair(rng, n_cards, kind, offset, remote=witness, triple=triple)
                    key = (pair["input"].shape, pair["input"].tobytes())
                    if key in seen:
                        continue
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("bounded observation construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = (inp.shape, inp.tobytes())
            assert key not in seen, "duplicate observation input"
            seen.add(key)
            info = _registered_evidence(inp)
            assert np.array_equal(out, _render_evidence(info)), "wrong consensus, completion, copy, or crop"
            assert len(info["rejected"]) >= len(info["cards"]), "fault observations missing"
            assert info["pixels"]-set(info["vertices"]), "completion is vacuous"
            assert info["rejected"]-info["pixels"]-info["copy_pixels"], "union ablation is invisible"
            assert not np.array_equal(out, _render_evidence(info, include_rejected=True))
            assert not np.array_equal(out, _render_evidence(info, complete=False, echo=False)), "A ablation is invisible"
            assert _normalized_points(_frame_corner_support(info)) != _normalized_points(info["vertices"]), "B ablation is invisible"
            without_b = _frame_corner_answer(info)
            assert without_b is None or not np.array_equal(out, without_b), "B ablation answers correctly"
            assert all(len(owners) <= len(info["cards"]) for owners in info["votes"].values())
            assert all(any(i in owners for owners in info["votes"].values()) for i in range(len(info["cards"])))
            # Glyphs and frames are evidence only; the result uses drawing ink.
            allowed = {info["background"], info["source"], info["copy_color"]}
            assert set(map(int, np.unique(out))) <= allowed
            assert np.any(out[0] != info["background"]) and np.any(out[-1] != info["background"])
            assert np.any(out[:, 0] != info["background"]) and np.any(out[:, -1] != info["background"])
            if info["marker"] is not None:
                assert len(info["copy_pixels"]-info["pixels"]) >= 3
                assert not np.array_equal(out, _render_evidence(info, echo=False)), "copy ablation is invisible"
            if split == "train":
                training.append(info)
    assert any(i["chord_pixels"] for i in training), "chord rejection not demonstrated"
    assert any(i["kind"] == "path" for i in training), "open completion not demonstrated"
    assert any(i["marker_remote"] for i in training), "registration of remote marker not demonstrated"
    assert any(any(len(v) >= 3 for v in i["votes"].values()) for i in training), "at-least-two convention not demonstrated"
    assert any((lambda answer: answer is None or not np.array_equal(answer, _render_evidence(i)))(
        _alternative_answer(i, {p for p, votes in i["votes"].items() if len(votes) == 2}))
        for i in training if any(len(v) >= 3 for v in i["votes"].values())), "exactly-two shortcut not distinguished"
    assert len({i["displacement"] for i in training if i["marker"] is not None}) >= 2
    assert any(i["copy_pixels"] & (i["pixels"]-set(i["vertices"])) for i in training), "source priority not demonstrated"


def _puzzle_key(puzzle):
    return tuple((s, tuple((np.asarray(p[k]).shape, np.asarray(p[k], dtype=np.int8).tobytes())
                          for p in puzzle[s] for k in ("input", "output"))) for s in ("train", "test"))


def self_test():
    """Bounded pixel-derived semantic and diversity checks on the 42 host seeds."""
    puzzles = [generate_puzzle(seed) for seed in HOST_SEEDS]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    pairs = [pair for p in puzzles for s in ("train", "test") for pair in p[s]]
    infos = [_registered_evidence(p["input"]) for p in pairs]
    assert {len(i["cards"]) for i in infos} == {2, 3, 4}
    assert {i["background"] for i in infos} == set(range(10))
    assert {i["source"] for i in infos} == set(range(10))
    assert {i["displacement"] for i in infos if i["marker"] is not None} == set(OFFSETS)
    assert len({p["input"].shape for p in pairs}) >= 20
    assert len({p["output"].shape for p in pairs}) >= 15
    assert len({tuple(sorted(i["vertices"])) for i in infos}) >= 60
    assert max(len(i["vertices"]) for i in infos) >= 6
    assert any(i["marker"] is None for i in infos)
    assert any(all(a[0] != b[0] and a[1] != b[1] for a, b in i["segments"]) for i in infos)
    assert any(len(i["glyphs"]) >= len(i["cards"]) for i in infos)
    assert len({len(signature) for i in infos for signature in i["glyphs"]}) >= 3
    for puzzle in puzzles:
        train = [_registered_evidence(p["input"]) for p in puzzle["train"]]
        test = [_registered_evidence(p["input"]) for p in puzzle["test"]]
        assert max(len(i["cards"]) for i in train) < min(len(i["cards"]) for i in test)
        assert test[0]["displacement"] not in {i["displacement"] for i in train}
        for pair in puzzle["train"]+puzzle["test"]:
            assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
    assert _puzzle_key(generate_puzzle(123, 2, 1)) == _puzzle_key(generate_puzzle(123, 2, 1))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(91, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Spatial repacking and arbitrary color permutation preserve the drawing.
    sample = pairs[0]
    bg, frame, cards = _read_cards(sample["input"])
    rectangles = []
    for card in cards:
        r, c = card["origin"]
        h, w = card["shape"]
        rectangles.append(sample["input"][r-1:r+h+1, c-1:c+w+1].copy())
    repacked = _pack(list(reversed(rectangles)), bg, np.random.default_rng(22))
    assert np.array_equal(solve(repacked), sample["output"])
    recolor = np.array([7, 9, 3, 0, 5, 2, 8, 1, 6, 4], dtype=np.int8)
    assert np.array_equal(solve(recolor[sample["input"]]), recolor[sample["output"]])
    for turns in range(4):
        for reflected in (False, True):
            inp = np.rot90(sample["input"], turns)
            out = np.rot90(sample["output"], turns)
            if reflected:
                inp, out = np.fliplr(inp), np.fliplr(out)
            assert np.array_equal(solve(inp), out)
    # There are multiple full noncrossing cycles through this fork/diamond set.
    try:
        _topology({(0, 2), (2, 0), (2, 2), (2, 4), (4, 2)})
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous contour accepted")
    # Missing output material must be rejected by the validator.
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
              for s in ("train", "test")}
    pair = broken["test"][0]
    info = _registered_evidence(pair["input"])
    p = tuple(np.argwhere(pair["output"] != info["background"])[0])
    pair["output"][p] = info["background"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("lost stroke accepted")
    return {"seeds": len(puzzles), "pairs": len(pairs),
            "input_shapes": len({p["input"].shape for p in pairs}),
            "output_shapes": len({p["output"].shape for p in pairs}),
            "vertex_counts": sorted({len(i["vertices"]) for i in infos}),
            "glyph_sizes": sorted({len(s) for i in infos for s in i["glyphs"]}),
            "remote_markers": sum(i["marker_remote"] for i in infos),
            "chord_rejection_pairs": sum(bool(i["chord_pixels"]) for i in infos)}
