"""Stained-glass loom: routed pigments fill complementary chamber parities.

No parent module is imported. Construction paths supply the expected answer;
the solver instead parses enclosure, stencil seeds and a directional cable graph.
"""
from __future__ import annotations

from itertools import permutations
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
CHECKER = ~DIAMOND
HOST_SEEDS = (0, 920767826, 7976620, 209444628, 462409398, 77061240,
              484624760, 851709872, 725819445, 524752391, 974761622,
              488497325, 870348490, 790895720, 906522673, 447167956,
              620728650, 906641419, 152355126, 234099150, 62676231,
              109481022, 355317587, 394043731, 407159612, 797695781,
              169086959, 518153178, 507955837, 261651957, 930805408,
              688861948, 558611359, 854675934, 478484634, 576270583,
              911825604, 6637041, 113660748, 210662145, 122413363,
              465298280)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _components(cells, diagonal=False):
    remaining = set(cells)
    steps = DIRS + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else ())
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        part, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _bbox(cells):
    p = np.asarray(list(cells), dtype=int)
    r, c = p.min(axis=0)
    R, C = p.max(axis=0) + 1
    return int(r), int(c), int(R), int(C)


def _rooms(grid, wall):
    # Padded exterior also handles frames touching a boundary in hand inputs.
    mask = np.pad(grid != wall, 1, constant_values=True)
    parts = _components(map(tuple, np.argwhere(mask)))
    h, w = mask.shape
    return [{(r - 1, c - 1) for r, c in part} for part in parts
            if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in part)]


def _values(grid, cells):
    return {int(grid[p]) for p in cells}


def _motifs(grid, room, tag):
    r, c, R, C = _bbox(room)
    found = []
    for y in range(r + 1, R - 1):
        for x in range(c + 1, C - 1):
            footprint = {(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
            if not footprint <= room:
                continue
            block = grid[y - 1:y + 2, x - 1:x + 2] == tag
            for name, pattern in (("diamond", DIAMOND), ("checker", CHECKER)):
                if np.array_equal(block, pattern):
                    marked = {(y + dy - 1, x + dx - 1) for dy, dx in np.argwhere(pattern)}
                    phases = {(a + b) % 2 for a, b in marked}
                    assert len(phases) == 1
                    found.append(dict(center=(y, x), kind=name,
                                      phase=phases.pop(), marked=marked))
    assert len(found) == 2
    assert {m['kind'] for m in found} == {'diamond', 'checker'}
    assert {m['phase'] for m in found} == {0, 1}
    assert len({m['center'][0] for m in found}) == 1
    assert abs(found[0]['center'][1] - found[1]['center'][1]) >= 4
    assert set.union(*(m['marked'] for m in found)) == {p for p in room if grid[p] == tag}
    return sorted(found, key=lambda m: m['center'])


def _rectangles(grid, color):
    result = []
    for part in _components(map(tuple, np.argwhere(grid == color))):
        r, c, R, C = _bbox(part)
        assert R - r >= 2 and C - c >= 2
        assert len(part) == (R - r) * (C - c)
        result.append((r, c, R, C))
    assert result
    return result


def _mask_links(rect, visible, endpoints):
    r, c, R, C = rect
    boundary = [((r - 1, x), 2) for x in range(c, C)]
    boundary += [((R, x), 0) for x in range(c, C)]
    boundary += [((y, c - 1), 1) for y in range(r, R)]
    boundary += [((y, C), 3) for y in range(r, R)]
    ports = []
    for p, inward in boundary:
        degree = sum((p[0] + dr, p[1] + dc) in visible for dr, dc in DIRS)
        if p in visible and (degree in (1, 3) or endpoints.get(p) == inward):
            ports.append((p, inward))
    assert len(ports) >= 2 and len(ports) % 2 == 0
    if len(ports) == 2:
        return [(ports[0], ports[1])]
    remaining, links = set(ports), []
    while remaining:
        a = min(remaining)
        remaining.remove(a)
        p, d = a
        candidates = [b for b in remaining if b[1] == (d + 2) % 4
                      and b[0][1 if d % 2 == 0 else 0] == p[1 if d % 2 == 0 else 0]]
        assert len(candidates) == 1, 'mask conceals an ambiguous permutation'
        b = candidates[0]
        remaining.remove(b)
        links.append((a, b))
    return links


def _trace(grid, wire, cross, rectangles, starts, ends):
    visible = set(map(tuple, np.argwhere(grid == wire)))
    if cross is not None:
        visible |= set(map(tuple, np.argwhere(grid == cross)))
    endpoints = {p: 2 for p in starts}
    endpoints.update({p: 0 for p in ends})
    graph = {p: {} for p in visible}
    for p in visible:
        for d, (dr, dc) in enumerate(DIRS):
            q = p[0] + dr, p[1] + dc
            if q in visible:
                graph[p][d] = (q, (d + 2) % 4)
    mask_details = []
    for rect in rectangles:
        links = _mask_links(rect, visible, endpoints)
        for (p, dp), (q, dq) in links:
            assert dp not in graph[p] and dq not in graph[q]
            graph[p][dp], graph[q][dq] = (q, dq), (p, dp)
        mask_details.append((rect, links))
    crossings = []
    for p, edges in graph.items():
        if p in endpoints:
            assert len(edges) == 1 and endpoints[p] in edges
        elif len(edges) == 4:
            assert cross is not None and grid[p] == cross, 'unmarked junction'
            crossings.append(p)
        else:
            assert len(edges) == 2 and grid[p] == wire, 'broken strand or malformed crossing'
    mapping, routes, seen = {}, [], set()
    for source in sorted(starts):
        p, incoming = source, 0
        route, states = [], set()
        while True:
            assert (p, incoming) not in states, 'closed cable loop'
            states.add((p, incoming))
            route.append(p)
            seen.add(p)
            if p in ends:
                mapping[source] = p
                break
            assert p == source or p not in starts, 'cable returns to donor bank'
            edges = graph[p]
            if p == source:
                outgoing = 2
            elif len(edges) == 4:
                outgoing = (incoming + 2) % 4
            else:
                assert incoming in edges
                outgoing = next(d for d in edges if d != incoming)
            assert outgoing in edges
            p, incoming = edges[outgoing]
        routes.append(route)
    assert set(mapping.values()) == set(ends) and len(mapping) == len(ends)
    assert seen == visible, 'unattached wire fragments'
    return mapping, routes, crossings, mask_details


def _parse(input_grid):
    original = _grid(input_grid)
    corners = [int(original[r, c]) for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    assert len(set(corners)) == 1
    bg = corners[0]
    candidates = []
    # Enclosure classification is rotation invariant. Screen roles once rather
    # than flood-filling every pigment and routing color in four orientations.
    possible_walls = []
    for color in sorted(set(map(int, np.unique(original))) - {bg}):
        regions = _rooms(original, color)
        value_sets = [_values(original, region) for region in regions]
        solid = sum(len(v) == 1 and bg not in v for v in value_sets)
        seeded = sum(len(v) == 2 and bg in v for v in value_sets)
        if seeded and solid == 2 * seeded and solid + seeded == len(regions):
            possible_walls.append(color)
    # Canonical orientation has the donor wells above the receiving chambers.
    for rotation in range(4):
        g = np.rot90(original, rotation).copy()
        for wall in possible_walls:
            try:
                rooms = _rooms(g, wall)
                sources, targets, colors = [], [], []
                for room in rooms:
                    vals = _values(g, room)
                    if len(vals) == 1 and bg not in vals:
                        sources.append(room)
                        colors.append(next(iter(vals)))
                    elif bg in vals and len(vals) == 2:
                        targets.append(room)
                    else:
                        raise AssertionError('unexpected enclosed region')
                assert len(targets) >= 1 and len(sources) == 2 * len(targets)
                assert all(_bbox(s)[2] < min(_bbox(t)[0] for t in targets) for s in sources)
                assert len(set(colors)) >= 2
                tags = set.union(*(_values(g, t) - {bg} for t in targets))
                assert len(tags) == 1
                tag = tags.pop()
                assert tag not in colors
                motifs = [_motifs(g, t, tag) for t in targets]
                target_top = {_bbox(t)[0] - 1 for t in targets}
                source_bottom = {_bbox(s)[2] for s in sources}
                assert len(target_top) == len(source_bottom) == 1
                top, bottom = target_top.pop(), source_bottom.pop()
                ends = {}
                for j, ms in enumerate(motifs):
                    for m in ms:
                        y, x = m['center']
                        assert y == top + 2 and g[top, x] == wall
                        ends[top - 1, x] = (j, m)
                wire_values = {int(g[p]) for p in ends}
                assert len(wire_values) == 1
                wire = wire_values.pop()
                assert wire not in {bg, wall, tag, *colors}
                starts = {}
                for j, room in enumerate(sources):
                    attached = [(bottom + 1, x) for y, x in room if y == bottom - 1
                                and g[bottom, x] == wall and g[bottom + 1, x] == wire]
                    assert len(attached) == 1
                    starts[attached[0]] = j
                other = set(map(int, np.unique(g))) - {bg, wall, tag, wire, *colors}
                assert len(other) <= 2
                cross, mask, rectangles = None, None, []
                for color in sorted(other):
                    try:
                        rects = _rectangles(g, color)
                    except AssertionError:
                        assert cross is None
                        cross = color
                    else:
                        assert mask is None
                        mask, rectangles = color, rects
                mapping, routes, crossings, mask_details = _trace(g, wire, cross, rectangles, starts, ends)
                # Every wall belongs to exactly one of the two compartment figures.
                wall_parts = _components(map(tuple, np.argwhere(g == wall)), diagonal=True)
                assert len(wall_parts) == 2
                adjacent = {(r + dr, c + dc) for room in targets for r, c in room
                            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                receiver_parts = [p for p in wall_parts if p & adjacent]
                assert len(receiver_parts) == 1
                receiver_wall = receiver_parts[0]
                assert (adjacent & set(map(tuple, np.argwhere(g == wall)))) <= receiver_wall
                r, c, R, C = _bbox(receiver_wall)
                crop = np.full((R - r, C - c), bg, dtype=np.int8)
                for y, x in receiver_wall:
                    crop[y - r, x - c] = wall
                candidates.append(dict(grid=g, bg=bg, wall=wall, wire=wire, cross=cross,
                                       mask=mask, tag=tag, rotation=rotation, sources=sources,
                                       targets=targets, colors=colors, motifs=motifs, starts=starts,
                                       ends=ends, mapping=mapping, routes=routes, crossings=crossings,
                                       masks=mask_details, receiver_wall=receiver_wall,
                                       bbox=(r, c, R, C), crop=crop))
            except (AssertionError, IndexError, StopIteration):
                continue
    assert len(candidates) == 1, f'input has {len(candidates)} valid interpretations'
    return candidates[0]


def _render(info, use_routes=True, use_chambers=True):
    out = info['crop'].copy()
    r, c, R, C = info['bbox']
    phases = [dict() for _ in info['targets']]
    ordered_ends = sorted(info['ends'])
    for k, source in enumerate(sorted(info['starts'])):
        end = info['mapping'][source] if use_routes else ordered_ends[k]
        j, motif = info['ends'][end]
        phases[j][motif['phase']] = info['colors'][info['starts'][source]]
    for j, room in enumerate(info['targets']):
        assert set(phases[j]) == {0, 1}
        if use_chambers:
            cells = room
        else:
            # B-only ablation: complete only the terminal stencil footprints.
            cells = set.union(*(m['marked'] for m in info['motifs'][j]))
        for y, x in cells:
            out[y - r, x - c] = phases[j][(y + x) % 2]
    return np.rot90(out, -info['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the full answer from this grid, without training or RNG state."""
    return _render(_parse(input_grid))


def _framework(room_sets, shape, bg, wall):
    g = np.full(shape, bg, dtype=np.int8)
    interior = set.union(*room_sets)
    boundary = {(r + dr, c + dc) for r, c in interior
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)} - interior
    for p in boundary:
        g[p] = wall
    return g


def _paths(rng, centers, start, heights_budget, target_perm):
    # Separate ingress and egress tracks make a full crossbar possible in 30x30.
    # Each strand has a unique lateral row, not an adjacent-swap layer.
    n = len(centers)
    rows = [start + 1 + 2 * k for k in range(n)]
    extra = int(rng.integers(heights_budget + 1))
    for _ in range(extra):
        cut = int(rng.integers(n + 1))
        if cut == n:
            continue
        for k in range(cut, n):
            rows[k] += 1
    assigned_rows = list(map(int, rng.permutation(rows)))
    jog = max(rows) + 2
    end = jog + 1
    paths = []
    for i, col in enumerate(centers):
        j = target_perm[i]
        exit_col, row = centers[j] + 2, assigned_rows[i]
        path = [(r, col) for r in range(start, row + 1)]
        step = 1 if exit_col > col else -1
        path += [(row, x) for x in range(col + step, exit_col + step, step)]
        path += [(r, exit_col) for r in range(row + 1, jog + 1)]
        path += [(jog, exit_col - 1), (jog, exit_col - 2), (end, centers[j])]
        paths.append(path)
    return paths, end


def _draw_paths(shape, paths, bg, wire, cross):
    g = np.full(shape, bg, dtype=np.int8)
    owners = {}
    for i, path in enumerate(paths):
        assert len(path) == len(set(path))
        for p in path:
            owners.setdefault(p, set()).add(i)
            g[p] = wire
    for p, routes in owners.items():
        if len(routes) == 2:
            g[p] = cross
        else:
            assert len(routes) == 1
    return g


def _add_masks(rng, g, paths, starts, ends, wire, cross, mask, wanted, keep_crossing=False):
    visible = set(map(tuple, np.argwhere((g == wire) | (g == cross))))
    points = sorted(visible)
    used = []
    # Random proposals first; bounded exhaustive 2x2 proposals guarantee a
    # simple hidden-stretch witness when a training lesson requests one.
    fallback = [(r, c, r + 2, c + 2) for r in range(min(p[0] for p in starts) + 1,
                                                    max(p[0] for p in ends) - 1)
                for c in range(1, g.shape[1] - 2)]
    for attempt in range(100 + len(fallback)):
        if len(used) == wanted:
            break
        if attempt < 100:
            y, x = points[int(rng.integers(len(points)))]
            hh, ww = int(rng.integers(2, 5)), int(rng.integers(2, 7))
            r, c = y - int(rng.integers(hh)), x - int(rng.integers(ww))
            R, C = r + hh, c + ww
        else:
            r, c, R, C = fallback[attempt - 100]
        if r <= min(p[0] for p in starts) or R > max(p[0] for p in ends):
            continue
        if c < 1 or C >= g.shape[1]:
            continue
        if any(not (R < a or b < r or C < d or e < c) for a, d, b, e in used):
            continue
        inside = lambda p: r <= p[0] < R and c <= p[1] < C
        actual = set()
        for path in paths:
            entry = None
            for a, b in zip(path, path[1:]):
                if not inside(a) and inside(b):
                    entry = a
                if inside(a) and not inside(b):
                    assert entry is not None
                    actual.add(frozenset((entry, b)))
                    entry = None
        if not actual:
            continue
        candidate = g.copy()
        candidate[r:R, c:C] = mask
        try:
            links = _mask_links((r, c, R, C), {p for p in visible if not inside(p)},
                                {**{p: 2 for p in starts}, **{p: 0 for p in ends}})
            assert {frozenset((a[0], b[0])) for a, b in links} == actual
            traced = _trace(candidate, wire, cross if cross in candidate else None,
                            used + [(r, c, R, C)], starts, ends)
            assert not keep_crossing or traced[2]
        except AssertionError:
            continue
        g = candidate
        used.append((r, c, R, C))
        visible = set(map(tuple, np.argwhere((g == wire) | (g == cross))))
    return g, used


def _make_pair(rng, n, lesson, shift, test, rotation):
    # All six roles can change locally; their grammar is recognized by _parse.
    palette = list(map(int, rng.permutation(10)))
    bg, wall, wire, cross, mask, tag = palette[:6]
    pigment_pool = palette[6:]
    gaps = [4] * (n - 1)
    width_budget = 24 - sum(gaps)
    for k in map(int, rng.permutation(n - 1)):
        if width_budget >= 2 and rng.random() < .4:
            gaps[k] += 2
            width_budget -= 2
    centers = [2]
    for gap in gaps:
        centers.append(centers[-1] + gap)
    w = centers[-1] + 4
    source_heights = list(map(int, rng.integers(1, 4, n)))
    source_bottom = max(source_heights) + 2
    start = source_bottom + 1
    min_target_top = start + 2 * n + 3
    max_height = min(8, 27 - min_target_top)
    assert max_height >= 4
    if shift == 'geometry' and not test:
        room_heights = [4] * (n // 2)
    elif shift == 'geometry' and test:
        room_heights = [max_height] * (n // 2)
    else:
        room_heights = list(map(int, rng.integers(4, max_height + 1, n // 2)))
    height_budget = max(0, 27 - min_target_top - max(room_heights))
    target_perm = list(map(int, rng.permutation(n)))
    if lesson is None and rng.random() < .18:
        target_perm = list(range(n))
    if lesson == 'distinct':
        target_perm = list(reversed(range(n)))
        labels = pigment_pool.copy()
        assert n == 4
    else:
        richness = int(rng.integers(2, 5))
        labels = pigment_pool[:richness] + list(map(int, rng.choice(pigment_pool[:richness], n - richness)))
        rng.shuffle(labels)
    if lesson == 'equal':
        # One chamber receives identical colors; the other still has two pigments.
        labels = [pigment_pool[0]] * n
        labels[target_perm.index(2)], labels[target_perm.index(3)] = pigment_pool[1:3]
    if shift == 'routing' and test:
        target_perm = list(reversed(range(n))) if rng.random() < .5 else list(map(int, rng.permutation(n)))
    if lesson is None and target_perm == list(range(n)) and rng.random() < .75:
        end = start + int(rng.integers(4, 10))
        paths = [[(y, x) for y in range(start, end + 1)] for x in centers]
    else:
        paths, end = _paths(rng, centers, start, height_budget, target_perm)
    top = end + 1
    h = top + max(room_heights) + 3
    assert h <= 30 and w <= 30
    source_rooms = []
    # Midpoints define shared walls, with independent well heights and side cuts.
    bounds = [0] + [(a + b) // 2 for a, b in zip(centers, centers[1:])] + [centers[-1] + 2]
    for i, col in enumerate(centers):
        room = {(y, x) for y in range(source_bottom - source_heights[i], source_bottom)
                for x in range(bounds[i] + 1, bounds[i + 1])}
        if len(room) > 3 and rng.random() < .4:
            r, c, R, C = _bbox(room)
            q = (r, c if rng.random() < .5 else C - 1)
            if q[1] != col:
                room.remove(q)
        source_rooms.append(room)
    target_rooms = []
    for j, hh in enumerate(room_heights):
        left, right = bounds[2 * j], bounds[2 * j + 2]
        room = {(y, x) for y in range(top + 1, top + hh + 1)
                for x in range(left + 1, right)}
        # Lower corner steps cannot disturb the two visible 3x3 seed stencils.
        cut = int(rng.integers(1, 4)) if lesson in ('distinct', 'equal') else int(rng.integers(4))
        if cut:
            side = int(rng.integers(2))
            depth = min(hh - 3, cut)
            for d in range(depth):
                span = 1 + d % 2
                for x in (range(left + 1, left + 1 + span) if side == 0 else range(right - span, right)):
                    room.discard((top + hh - d, x))
        target_rooms.append(room)
    inp = _framework(source_rooms + target_rooms, (h, w), bg, wall)
    # A continuous bottom rail keeps notched, unequal donor wells one figure.
    inp[source_bottom, :centers[-1] + 3] = wall
    for room, color in zip(source_rooms, labels):
        for p in room:
            inp[p] = color
    feed_colors = [None] * n
    for i, j in enumerate(target_perm):
        feed_colors[j] = labels[i]
    expected = _framework(target_rooms, (h, w), bg, wall)
    for j, room in enumerate(target_rooms):
        swap = int(rng.integers(2))
        phases = {}
        for k in range(2):
            col = centers[2 * j + k]
            center = (top + 2, col)
            pattern = DIAMOND if k == swap else CHECKER
            for dy, dx in np.argwhere(pattern):
                p = center[0] + int(dy) - 1, center[1] + int(dx) - 1
                assert p in room
                inp[p] = tag
                phases[(sum(p)) % 2] = feed_colors[2 * j + k]
        assert len(phases) == 2
        for p in room:
            expected[p] = phases[sum(p) % 2]
    route_grid = _draw_paths((h, w), paths, bg, wire, cross)
    starts = {path[0] for path in paths}
    ends = {path[-1] for path in paths}
    wanted = 1 if lesson == 'equal' else int(rng.integers(3))
    if shift == 'routing':
        wanted = 2 if test else min(1, wanted)
    route_grid, masks = _add_masks(rng, route_grid, paths, starts, ends, wire, cross, mask,
                                  wanted, keep_crossing=lesson == 'distinct')
    assert lesson != 'equal' or masks
    assert shift != 'routing' or not test or len(masks) == 2
    drawn = route_grid != bg
    assert np.all(inp[drawn] == bg)
    inp[drawn] = route_grid[drawn]
    receiver_wall = set(map(tuple, np.argwhere(expected == wall)))
    r, c, R, C = _bbox(receiver_wall)
    out = expected[r:R, c:C]
    # Independent padding of the whole scene, budget permitting.
    extra_h, extra_w = int(rng.integers(31 - h)), int(rng.integers(31 - w))
    before_h, before_w = int(rng.integers(extra_h + 1)), int(rng.integers(extra_w + 1))
    inp = np.pad(inp, ((before_h, extra_h - before_h), (before_w, extra_w - before_w)), constant_values=bg)
    if rng.random() < .5:
        inp, out = inp[:, ::-1], out[:, ::-1]
    inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
    return {'input': inp, 'output': out}


def _count(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lower <= value <= upper:
        raise ValueError(f'{name} must be in {lower}..{upper}')
    return int(value)


def _key(g):
    return tuple(g.shape), g.astype(np.int8).tobytes()


def _unique_recolor(pair, seen):
    """At least 90 distinct local background/wall assignments, for <=14 pairs."""
    info = _parse(pair['input'])
    old = [info['bg'], info['wall']] + [v for v in range(10) if v not in (info['bg'], info['wall'])]
    for bg in range(10):
        for wall in range(10):
            if bg == wall:
                continue
            new = [bg, wall] + [v for v in range(10) if v not in (bg, wall)]
            lut = np.empty(10, dtype=np.int8)
            lut[old] = new
            candidate = {k: lut[v] for k, v in pair.items()}
            if _key(candidate['input']) not in seen:
                return candidate
    raise AssertionError('finite recoloring bound violated')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(['count', 'geometry', 'routing', 'combination']))
    orientation = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, total in (('train', nt), ('test', nq)):
        for i in range(total):
            lesson = ('distinct' if i == 0 else 'equal' if i == 1 else None) if split == 'train' else None
            n = 4 if lesson or (shift in ('count', 'combination') and split == 'train') else int(rng.choice([4, 6]))
            if split == 'test' and shift in ('count', 'combination'):
                n = 6
            rotation = (orientation + int(i == 1)) % 4 if lesson else int(rng.integers(4))
            for attempt in range(32):
                pair = _make_pair(rng, n, lesson, shift, split == 'test', rotation)
                if _key(pair['input']) not in seen:
                    break
            else:
                pair = _unique_recolor(pair, seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    return puzzle


def _measure(pair, info):
    colors = [dict() for _ in info['targets']]
    for source, end in info['mapping'].items():
        j, m = info['ends'][end]
        colors[j][m['phase']] = info['colors'][info['starts'][source]]
    r, c, R, C = info['bbox']
    bends = []
    for route in info['routes']:
        ds = [(int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))) for a, b in zip(route, route[1:])]
        bends.append(sum(a != b for a, b in zip(ds, ds[1:])))
    return dict(chambers=len(info['targets']), wells=len(info['sources']),
                permutation=tuple(sorted(info['ends']).index(info['mapping'][p]) for p in sorted(info['starts'])),
                crossings=len(info['crossings']), masks=len(info['masks']),
                mask_shapes=tuple((R - r, C - c) for (r, c, R, C), _ in info['masks']),
                elbows=sum(len(links) == 1 and links[0][0][1] % 2 != links[0][1][1] % 2
                           for _, links in info['masks']),
                multi_masks=sum(len(links) > 1 for _, links in info['masks']),
                room_areas=tuple(len(room) for room in info['targets']),
                room_heights=tuple(_bbox(room)[2] - _bbox(room)[0] for room in info['targets']),
                nonrectangular=sum(len(room) != (_bbox(room)[2] - _bbox(room)[0]) *
                                   (_bbox(room)[3] - _bbox(room)[1]) for room in info['targets']),
                solid_chambers=sum(len(set(p.values())) == 1 for p in colors),
                pigments=len(set(info['colors'])), bends=tuple(bends),
                orientation=info['rotation'], roles=(info['bg'], info['wall'], info['wire'], info['tag']),
                shape=tuple(np.asarray(pair['input']).shape), output_shape=tuple(np.asarray(pair['output']).shape),
                route_ablation=not np.array_equal(_render(info, use_routes=False), pair['output']),
                chamber_ablation=not np.array_equal(_render(info, use_chambers=False), pair['output']))


def inspect_pair(pair):
    """Reconstruct measurements from the input rather than construction state."""
    return _measure(pair, _parse(pair['input']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            predicted = _render(info)
            assert np.array_equal(predicted, out), 'wrong routed chamber fill'
            canonical_out = np.rot90(out, info['rotation'])
            r, c, R, C = info['bbox']
            interior = np.zeros(info['crop'].shape, dtype=bool)
            for room in info['targets']:
                for y, x in room:
                    interior[y - r, x - c] = True
            assert np.array_equal(canonical_out[~interior], info['crop'][~interior]), 'walls or exterior changed'
            assert np.all(canonical_out[interior] != info['bg'])
            for room in info['targets']:
                for phase in (0, 1):
                    vals = {int(canonical_out[y - r, x - c]) for y, x in room if (y + x) % 2 == phase}
                    assert len(vals) == 1 and vals <= set(info['colors'])
            metric = _measure(pair, info)
            assert metric['chamber_ablation']
            if split == 'train':
                training.append(metric)
            else:
                testing.append(metric)
    assert training[0]['pigments'] == 4 and training[0]['route_ablation']
    assert training[0]['crossings'] > 0 and training[0]['nonrectangular'] > 0
    assert training[1]['solid_chambers'] > 0 and training[1]['solid_chambers'] < training[1]['chambers']
    assert training[1]['masks'] > 0
    assert len({m['orientation'] for m in training[:2]}) == 2
    # Every query has a visible structural shift, not just a randomly new grid.
    train_counts = {m['wells'] for m in training}
    train_height = max(max(m['room_heights']) for m in training)
    train_masks = max(m['masks'] for m in training)
    for m in testing:
        assert (m['wells'] not in train_counts or max(m['room_heights']) > train_height
                or m['masks'] > train_masks), 'no structural train/test shift'


def self_test():
    signatures, shapes, normalized, metrics = set(), set(), set(), []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(p[k]) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output'))
        assert signature not in signatures
        signatures.add(signature)
        for split in ('train', 'test'):
            for p in puzzle[split]:
                m = inspect_pair(p)
                metrics.append(m)
                shapes.add(m['shape'])
                info = _parse(p['input'])
                normalized.add((_key(info['grid'] == info['wall']), m['permutation'], m['room_areas']))
    assert {m['wells'] for m in metrics} == {4, 6}
    assert {m['orientation'] for m in metrics} == {0, 1, 2, 3}
    assert len(shapes) >= 20 and len(normalized) >= 100
    assert len({m['roles'] for m in metrics}) >= 80
    assert {m['pigments'] for m in metrics} >= {2, 3, 4}
    assert any(m['masks'] == 0 for m in metrics) and any(m['masks'] == 2 for m in metrics)
    assert any(m['elbows'] for m in metrics) and any(m['multi_masks'] for m in metrics)
    assert any(m['solid_chambers'] for m in metrics)
    assert any(m['bends'] == (0,) * m['wells'] for m in metrics)
    assert all(m['chamber_ablation'] for m in metrics)
    a, b = generate_puzzle(713, 2, 1), generate_puzzle(713, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in ('input', 'output'))
    fallback = _unique_recolor(a['train'][0], {_key(a['train'][0]['input'])})
    assert _key(fallback['input']) != _key(a['train'][0]['input'])
    assert np.array_equal(solve(fallback['input']), fallback['output'])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(714, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # All n! routings are supported by this crossbar, without reading answers.
    rng = np.random.default_rng(715)
    for n in (4, 6):
        centers = list(range(2, 4 * n, 4))
        for perm in permutations(range(n)):
            paths, end = _paths(rng, centers, 4, 0, perm)
            g = _draw_paths((end + 1, 4 * n + 2), paths, 0, 1, 2)
            starts, ends = {p[0] for p in paths}, {p[-1] for p in paths}
            mapping, _, _, _ = _trace(g, 1, 2 if 2 in g else None, [], starts, ends)
            assert all(mapping[paths[i][0]] == paths[i][-1] for i in range(n))
    # Fixed geometry/cables with the two glyph forms exchanged must exchange
    # the pigments on every cell of that chamber, not just the seed footprints.
    witness = generate_puzzle(717, 2, 1)['train'][0]
    info = _parse(witness['input'])
    canonical = info['grid'].copy()
    baseline = np.rot90(witness['output'], info['rotation']).copy()
    r, c, R, C = info['bbox']
    room = info['targets'][0]
    ms = info['motifs'][0]
    for p in set.union(*(m['marked'] for m in ms)):
        canonical[p] = info['bg']
    for m in ms:
        y, x = m['center']
        pattern = CHECKER if m['kind'] == 'diamond' else DIAMOND
        for dy, dx in np.argwhere(pattern):
            canonical[y + int(dy) - 1, x + int(dx) - 1] = info['tag']
    phase_colors = {phase: next(int(baseline[y - r, x - c]) for y, x in room
                               if (y + x) % 2 == phase) for phase in (0, 1)}
    assert phase_colors[0] != phase_colors[1]
    expected = baseline.copy()
    for y, x in room:
        expected[y - r, x - c] = phase_colors[1 - (y + x) % 2]
    assert np.array_equal(solve(canonical), expected)
    # Perturb one well, preserving geometry and all other pigment assignments.
    canonical = info['grid'].copy()
    source = min(info['starts'])
    index = info['starts'][source]
    old = info['colors'][index]
    replacement = next(v for v in info['colors'] if v != old)
    for p in info['sources'][index]:
        canonical[p] = replacement
    j, m = info['ends'][info['mapping'][source]]
    expected = baseline.copy()
    for y, x in info['targets'][j]:
        if (y + x) % 2 == m['phase']:
            expected[y - r, x - c] = replacement
    unchanged = canonical.copy()
    assert np.array_equal(solve(canonical), expected)
    assert np.array_equal(canonical, unchanged), 'solve mutated its input'
    # Reject an output that changes a preserved wall.
    bad = generate_puzzle(716, 2, 1)
    bad['test'][0]['output'] = bad['test'][0]['output'].copy()
    info = _parse(bad['test'][0]['input'])
    y, x = np.argwhere(bad['test'][0]['output'] == info['wall'])[0]
    bad['test'][0]['output'][y, x] = info['bg']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        return {'seeds': 42, 'pairs': len(metrics), 'input_shapes': len(shapes),
                'structures': len(normalized), 'routing_permutations': 744}
    raise AssertionError('corrupted wall accepted')
