"""Compressed stencil ribbons unfolded onto a rewired, gravity-directed loom.

Only NumPy and the standard library are used. All solving conventions are fixed
or visible in the input; no training labels, seed, or generation state are read.
"""
from __future__ import annotations

import json
import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _box(cells):
    a = np.asarray(list(cells), dtype=int)
    assert len(a)
    lo, hi = a.min(axis=0), a.max(axis=0) + 1
    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _DIRS:
                nxt = (r + dr, c + dc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
        result.append(cells)
    return result


def _paths(mask):
    """Every component must be a simple path, with no isolated cell or cycle."""
    result = []
    for cells in _components(mask):
        domain = set(cells)
        adjacent = {p: [(p[0] + dr, p[1] + dc) for dr, dc in _DIRS
                        if (p[0] + dr, p[1] + dc) in domain] for p in cells}
        assert all(len(v) in (1, 2) for v in adjacent.values()), 'branch or isolated track'
        ends = sorted(p for p, v in adjacent.items() if len(v) == 1)
        assert len(ends) == 2, 'closed loop'
        path, previous, current = [ends[0]], None, ends[0]
        while current != ends[1]:
            choices = [p for p in adjacent[current] if p != previous]
            assert len(choices) == 1
            previous, current = current, choices[0]
            path.append(current)
        assert len(path) == len(cells)
        result.append(path)
    return sorted(result, key=lambda p: (p[0], p[-1]))


def _frame(grid):
    candidates = []
    for color in np.unique(grid):
        cells = np.argwhere(grid == color)
        r, e, c, d = _box(cells)
        h, w = e - r, d - c
        if min(h, w) < 7 or len(cells) != 2 * h + 2 * w - 3:
            continue
        if not (np.all(grid[r, c:d] == color) and np.all(grid[e-1, c:d] == color)
                and np.all(grid[r:e, c] == color) and np.all(grid[r:e, d-1] == color)):
            continue
        extras = [(int(a), int(b)) for a, b in cells if r < a < e-1 and c < b < d-1]
        if len(extras) != 1:
            continue
        a, b = extras[0]
        if ((a in (r+1, e-2) and c+2 <= b <= d-3)
                or (b in (c+1, d-2) and r+2 <= a <= e-3)):
            candidates.append(((r, e, c, d), int(color), (a, b)))
    assert len(candidates) == 1, 'need one exclusive frame color with one inward tab'
    return candidates[0]


def _canonical(grid):
    _frame(grid)
    for q in range(4):
        normalized = np.rot90(grid, q).copy()
        box, color, tab = _frame(normalized)
        if tab[0] == box[1] - 2:
            return normalized, q, box, color, tab
    raise AssertionError('gravity tab is not on a frame side')


def _rewire(core, bg, track, divider, sources):
    h, w = core.shape
    assert h >= 3 and h % 2 == 1 and w >= 9
    assert np.all(np.isin(core[::2], [bg, track, *sources]))
    assert np.all(np.isin(core[1::2], [bg, divider]))
    assert np.all(np.isin(core[::2, [0, w-1]], [track, *sources]))
    writes, centers, row_counts = {}, [], []
    overlaps = 0
    for r in range(1, h, 2):
        xs = np.flatnonzero(core[r] == divider).tolist()
        runs = []
        for x in xs:
            if not runs or x != runs[-1][-1] + 1:
                runs.append([])
            runs[-1].append(x)
        row_counts.append(len(runs))
        for run in runs:
            assert len(run) == 3 and 1 <= run[0] and run[-1] <= w-2
            c = run[1]
            assert core[r-1, c] == core[r+1, c] == track
            assert np.all(np.isin(core[r-1:r+2:2, c-1:c+2], [track, *sources]))
            centers.append((r, c))
            for a in range(r-1, r+2):
                for b in range(c-1, c+2):
                    value = divider if b == c else track
                    if (a, b) in writes:
                        assert writes[a, b] == value, 'conflicting switch neighborhoods'
                        overlaps += 1
                    writes[a, b] = value
    assert centers
    structural = core.copy()
    structural[np.isin(structural, list(sources))] = track
    for p, value in writes.items():
        structural[p] = value
    paths = _paths(structural == track)
    ends = set()
    for path in paths:
        steps = {b[0] - a[0] for a, b in zip(path, path[1:]) if a[0] != b[0]}
        assert len(steps) <= 1, 'route has a gravitational reversal'
        for r, c in (path[0], path[-1]):
            assert r % 2 == 0
            assert c in (0, w-1) or core[r, c-1] == bg or core[r, c+1] == bg
            ends.add((r, c))
    markers = {tuple(map(int, p)): int(core[tuple(p)])
               for p in np.argwhere(np.isin(core, list(sources)))}
    assert set(markers) <= ends, 'packet is not an exposed terminal'
    return dict(structural=structural, paths=paths, markers=markers,
                centers=centers, row_counts=row_counts, overlaps=overlaps,
                breaks=[tuple(map(int, p)) for p in np.argwhere((core == bg) &
                        (np.indices(core.shape)[0] % 2 == 0))])


def _analyze(input_grid):
    original = _grid(input_grid)
    grid, q, frame, frame_color, tab = _canonical(original)
    r, e, c, d = frame
    corners = [grid[a, b] for a, b in ((r+1, c+1), (r+1, d-2), (e-2, c+1), (e-2, d-2))]
    assert len(set(map(int, corners))) == 1
    bg = int(corners[0])
    assert bg != frame_color
    exterior = grid != bg
    exterior[r:e, c:d] = False
    key_mask = np.zeros(grid.shape, dtype=bool)
    mapping, key_boxes = {}, {}
    for cells in _components(exterior):
        values, counts = np.unique([grid[p] for p in cells], return_counts=True)
        if len(values) == 1:
            continue
        a, b, x, y = _box(cells)
        assert (b-a, y-x) == (2, 2) and len(cells) == 4
        assert len(values) == 2 and sorted(counts.tolist()) == [1, 3]
        source = int(values[np.argmax(counts)])
        target = int(values[np.argmin(counts)])
        assert grid[a, x] == grid[a, y-1] == grid[b-1, x] == source
        assert grid[b-1, y-1] == target
        assert source not in mapping and frame_color not in (source, target)
        mapping[source] = target
        key_boxes[source] = (a, b, x, y)
        key_mask[a:b, x:y] = True
    assert 1 <= len(mapping) <= 4
    remaining = exterior & ~key_mask
    assert set(map(int, np.unique(grid[remaining]))) == set(mapping)
    stencils, stencil_boxes, sizes = {}, {}, set()
    for source in mapping:
        box = _box(np.argwhere(remaining & (grid == source)))
        a, b, x, y = box
        assert b-a == y-x and 2 <= b-a <= 4
        tile = grid[a:b, x:y]
        assert np.all(np.isin(tile, [bg, source]))
        mask = tile == source
        assert all(edge.any() for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]))
        stencils[source], stencil_boxes[source] = mask.copy(), box
        sizes.add(b-a)
    assert len(sizes) == 1
    boxes = list(key_boxes.values()) + list(stencil_boxes.values())
    for i, (a, b, x, y) in enumerate(boxes):
        assert b <= r or a >= e or y <= c or x >= d
        for aa, bb, xx, yy in boxes[i+1:]:
            assert b < aa or bb < a or y < xx or yy < x, 'clue rectangles need a blank gap'
    interior_mask = (grid != bg) & (grid != frame_color)
    interior_mask[:r+1] = interior_mask[e-1:] = False
    interior_mask[:, :c+1] = interior_mask[:, d-1:] = False
    cr, ce, cc, cd = _box(np.argwhere(interior_mask))
    core = grid[cr:ce, cc:cd].copy()
    # At least one blank layer separates the loom from the tab and frame.
    assert cr >= r+2 and ce <= e-3 and cc >= c+2 and cd <= d-2
    divider_colors = set(map(int, np.unique(core[1::2]))) - {bg}
    assert len(divider_colors) == 1
    divider = next(iter(divider_colors))
    rest = set(map(int, np.unique(core))) - {bg, divider, *mapping}
    assert len(rest) == 1
    track = next(iter(rest))
    assert not ({bg, frame_color, track, divider} & (set(mapping) | set(mapping.values())))
    info = _rewire(core, bg, track, divider, set(mapping))
    assert set(info['markers'].values()) == set(mapping)
    n = next(iter(sizes))
    for path in info['paths']:
        count = sum(p in info['markers'] for p in (path[0], path[-1]))
        assert count * n * n <= len(path), 'complete ribbons must fit the route'
    return dict(grid=grid, q=q, frame_box=frame, frame=frame_color, tab=tab,
                background=bg, track=track, divider=divider,
                core=core, core_box=(cr, ce, cc, cd), n=n,
                mapping=mapping, stencils=stencils,
                stencil_boxes=stencil_boxes, key_boxes=key_boxes, **info)


def _jobs(paths, markers, length, reverse_order=False):
    jobs = []
    for path in paths:
        ends = [p for p in (path[0], path[-1]) if p in markers]
        if not ends:
            continue
        if path[0][0] == path[-1][0]:
            for p in ends:
                inward = path if p == path[0] else path[::-1]
                jobs.append((markers[p], p, inward[:length]))
        else:
            inward = path if path[0][0] > path[-1][0] else path[::-1]
            # Low-terminal packet precedes high-terminal packet, if present.
            ends.sort(key=lambda p: -p[0], reverse=reverse_order)
            for i, p in enumerate(ends):
                jobs.append((markers[p], p, inward[i*length:(i+1)*length]))
    return jobs


def _print_core(info, variant='correct'):
    n, track = info['n'], info['track']
    paths, structural = info['paths'], info['structural']
    if variant == 'straight':
        structural = info['core'].copy()
        structural[np.isin(structural, list(info['mapping']))] = track
        paths = _paths(structural == track)
    answer = structural.copy()
    length = 1 if variant == 'unit' else n*n
    jobs = _jobs(paths, info['markers'], length, reverse_order=variant == 'reverse_order')
    for source, origin, cells in jobs:
        bits = info['stencils'][source].ravel().tolist()
        target = info['mapping'][source]
        if variant == 'unit':
            bits, target = [True], source
        elif variant == 'solid':
            bits = [True] * (n*n)
        elif variant == 'compact':
            bits = [True] * sum(bits)
        elif variant == 'reverse_bits':
            bits = bits[::-1]
        elif variant == 'column_bits':
            bits = info['stencils'][source].T.ravel().tolist()
        elif variant == 'identity_ink':
            target = source
        elif variant == 'recursive':
            seen = {source}
            while target in info['mapping'] and target not in seen:
                seen.add(target)
                target = info['mapping'][target]
        for p, bit in zip(cells, bits):
            if bit:
                answer[p] = target
    return answer


def _answer(info, variant='correct'):
    r, e, c, d = info['frame_box']
    cr, ce, cc, cd = info['core_box']
    result = info['grid'][r:e, c:d].copy()
    result[cr-r:ce-r, cc-c:cd-c] = _print_core(info, variant)
    return np.rot90(result, -info['q']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from this input alone."""
    return _answer(_analyze(input_grid))


def _mask(rng, n, informative=False):
    for _ in range(80):
        y, x = np.indices((n, n))
        kind = int(rng.integers(7))
        if kind == 0:
            mask = (x == 0) | (x == n-1) | (y == 0) | (y == n-1)
        elif kind == 1:
            mask = x == y
        elif kind == 2:
            mask = (x+y) % 2 == 0
        elif kind == 3:
            mask = (y % 2 == 0) | (y == n-1)
        elif kind == 4:
            mask = (x == n//2) | (y == n//2)
        elif kind == 5 and not informative:
            mask = np.ones((n, n), bool)
        else:
            mask = rng.random((n, n)) < rng.uniform(.25, .85)
        for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]):
            if not edge.any():
                edge[int(rng.integers(n))] = True
        if informative:
            mask[0, 0] = mask[-1, -1] = True
            mask[0, 1] = False
            if mask.all() or np.array_equal(mask.ravel(), mask.ravel()[::-1]):
                continue
        return mask
    mask = np.ones((n, n), bool)
    mask[0, 1] = False
    return mask


def _base(lanes, width, centers, cuts, bg, track, divider):
    core = np.full((2*lanes-1, width), bg, dtype=np.int8)
    core[::2] = track
    for r, c in centers:
        core[r, c-1:c+2] = divider
    for p in cuts:
        core[p] = bg
    return core


def _layout(rng, lanes, width, bg, track, divider, level=False):
    rows, centers = [], []
    blank = int(rng.choice([0, lanes-2])) if level else -1
    lattice = rng.random() < .45
    for i in range(lanes-1):
        choices = list(range(2 + 2*(i % 2), width-2, 4)) if lattice else list(range(2, width-2))
        rng.shuffle(choices)
        previous = rows[-1] if rows else []
        wanted = 0 if i == blank else int(rng.choice([0, 1, 1, 1, 2, 2, 3]))
        row = []
        for c in choices:
            if len(row) >= wanted:
                break
            if all(abs(c-p) >= 4 for p in row) and all(abs(c-p) >= 2 for p in previous):
                row.append(c)
        rows.append(sorted(row))
        centers.extend((2*i+1, c) for c in row)
    if not centers:
        eligible = [i for i in range(lanes-1) if i != blank]
        i = int(rng.choice(eligible))
        centers = [(2*i+1, int(rng.integers(2, width-2)))]
    cuts = []
    for r in range(0, 2*lanes-1, 2):
        ports = sorted((c, a) for a, c in centers if abs(a-r) == 1)
        for (a, side_a), (b, side_b) in zip(ports, ports[1:]):
            if side_a == side_b:
                cuts.append((r, int(rng.integers(a+2, b-1))))
    for _ in range(int(rng.choice([0, 0, 1, 2]))):
        pool = [(r, c) for r in range(0, 2*lanes-1, 2) for c in range(2, width-2)
                if all(abs(c-b) >= 2 for a, b in centers if abs(a-r) == 1)
                and all(abs(c-b) >= 3 for a, b in cuts if a == r)]
        if pool:
            cuts.append(pool[int(rng.integers(len(pool)))])
    core = _base(lanes, width, centers, cuts, bg, track, divider)
    return core, _rewire(core, bg, track, divider, set())


def _reserve(occupied, r, c, h, w):
    occupied[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True


def _available(occupied, h, w):
    summed = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    counts = summed[h:, w:] - summed[:-h, w:] - summed[h:, :-w] + summed[:-h, :-w]
    return np.argwhere(counts == 0)


def _place_clues(rng, bh, bw, n, sources):
    for attempt in range(24):
        r = int(rng.choice([0, 30-bh])) if attempt % 2 else int(rng.integers(31-bh))
        c = int(rng.choice([0, 30-bw])) if attempt % 2 else int(rng.integers(31-bw))
        occupied = np.zeros((30, 30), bool)
        _reserve(occupied, r, c, bh, bw)
        positions = {}
        items = [('stencil', int(s), n) for s in rng.permutation(sources)]
        items += [('key', int(s), 2) for s in rng.permutation(sources)]
        for kind, source, size in items:
            spots = _available(occupied, size, size)
            if not len(spots):
                break
            a, b = map(int, spots[int(rng.integers(len(spots)))])
            positions[kind, source] = (a, b)
            _reserve(occupied, a, b, size, size)
        if len(positions) == len(items):
            return (30, 30), (r, c), positions, False
    # Guaranteed upper shelves, with independently shuffled entry orders.
    h, w = bh+n+4, max(bw, len(sources)*(n+1)-1)
    assert h <= 30 and w <= 30
    positions = {}
    for kind, size, top in (('stencil', n, 0), ('key', 2, n+1)):
        span = len(sources)*(size+1)-1
        left = int(rng.integers(w-span+1))
        for i, source in enumerate(rng.permutation(sources)):
            positions[kind, int(source)] = (top, left+i*(size+1))
    return (h, w), (n+4, int(rng.integers(w-bw+1))), positions, True


def _draw_frame(grid, r, c, h, w, color):
    grid[r, c:c+w] = grid[r+h-1, c:c+w] = color
    grid[r:r+h, c] = grid[r:r+h, c+w-1] = color
    grid[r+h-2, c+w//2] = color


def _make_pair(rng, params, split, demo):
    training = split == 'train'
    shift = params['shift']
    if demo == 'single':
        n = params['single_n']
    elif demo == 'collision':
        n = 5-params['single_n']  # the two essential demos use sizes 2 and 3
    elif shift == 'words':
        n = int(rng.choice([2, 3])) if training else 4
    else:
        n = int(rng.choice([2, 3, 4], p=[.4, .45, .15]))
    if shift == 'routes':
        lane_range = (3, 5) if training else (6, 9)
    else:
        lane_range = (3, 9)
    if training and demo and shift != 'routes':
        lane_range = (3, 6)
    roles = list(map(int, rng.permutation(10)[:4])) if params['vary_roles'] else params['roles']
    bg, track, divider, frame = roles
    available = [c for c in range(10) if c not in roles]
    rng.shuffle(available)
    k = int(rng.integers(2 if demo else 1, 5))
    if shift == 'load' and training:
        k = 2
    sources = available[:k]
    for attempt in range(100):
        lanes = int(rng.integers(*lane_range))
        low_width = 19 if demo == 'collision' and n == 3 else (17 if n == 4 else 13)
        width = int(rng.integers(low_width, 26))
        try:
            core, net = _layout(rng, lanes, width, bg, track, divider, level=demo == 'single')
        except AssertionError:
            continue
        paths = net['paths']
        vertical = [i for i, p in enumerate(paths) if p[0][0] != p[-1][0] and len(p) >= n*n]
        level = [i for i, p in enumerate(paths) if p[0][0] == p[-1][0] and len(p) >= n*n]
        double = [i for i in vertical if len(paths[i]) >= 2*n*n]
        bent_double = [i for i in double if any(p[0] % 2 == 1 for p in
                       (paths[i] if paths[i][0][0] > paths[i][-1][0] else paths[i][::-1])[:2*n*n])]
        capacities = [min(2, len(p)//(n*n)) for p in paths]
        needed = k+2 if demo else k
        if shift == 'load' and training:
            needed += 1  # at least one eligible slot must remain empty in training
        if not vertical or sum(capacities) < needed:
            continue
        if demo == 'single' and (not level or len(vertical) < 2):
            continue
        if demo == 'collision' and (len(double) < 2 or not bent_double):
            continue
        break
    else:
        # A readable capacity-safe construction, retaining requested lane range.
        lanes, width = max(lane_range[0], 4 if n == 4 and k == 4 else 3), 25
        centers = [(3 if demo == 'single' else 1, n*n if demo == 'collision' else 12)]
        core = _base(lanes, width, centers, [], bg, track, divider)
        net = _rewire(core, bg, track, divider, set())
        paths = net['paths']
        vertical = [i for i, p in enumerate(paths) if p[0][0] != p[-1][0]]
        level = [i for i, p in enumerate(paths) if p[0][0] == p[-1][0]]
        double = [i for i in vertical if len(paths[i]) >= 2*n*n]
        bent_double = [i for i in double if any(p[0] % 2 == 1 for p in
                       (paths[i] if paths[i][0][0] > paths[i][-1][0] else paths[i][::-1])[:2*n*n])]
        capacities = [min(2, len(p)//(n*n)) for p in paths]
        assert demo != 'collision' or len(double) >= 2
    markers, frozen = {}, set()
    if demo == 'single':
        i, j = map(int, rng.choice(vertical, 2, replace=False))
        p = paths[i]
        markers[min(p[0], p[-1])] = sources[0]
        frozen.add(i)
        z = int(rng.choice(level))
        markers[paths[z][0 if rng.random() < .5 else -1]] = sources[0]
        markers[paths[j][0 if rng.random() < .5 else -1]] = sources[1]
    elif demo == 'collision':
        first = int(rng.choice(bent_double))
        second = int(rng.choice([i for i in double if i != first]))
        for z, i in enumerate((first, second)):
            i = int(i)
            p = paths[i]
            markers[max(p[0], p[-1])] = sources[z]
            markers[min(p[0], p[-1])] = sources[1-z]
            frozen.add(i)
    # Ensure every visible legend source has at least one packet.
    for source in sources:
        if source in markers.values():
            continue
        pool = [(i, p) for i, path in enumerate(paths) if i not in frozen
                for p in (path[0], path[-1]) if p not in markers
                and sum(x in markers for x in (path[0], path[-1])) < capacities[i]]
        assert pool
        i, p = pool[int(rng.integers(len(pool)))]
        markers[p] = source
    probability = (0.0 if demo else .27) if training else (.95 if shift == 'load' else .55)
    if shift == 'load' and training:
        # Freeze a route with an unused slot. Tests fill every eligible slot.
        spare = [i for i, p in enumerate(paths)
                 if sum(x in markers for x in (p[0], p[-1])) < capacities[i]]
        assert spare
        frozen.add(int(rng.choice(spare)))
    for i in rng.permutation(len(paths)):
        i = int(i)
        if i in frozen:
            continue
        path = paths[i]
        for p in (path[0], path[-1]) if rng.random() < .5 else (path[-1], path[0]):
            count = sum(x in markers for x in (path[0], path[-1]))
            if p not in markers and count < capacities[i] and (rng.random() < probability or
                                                              (not training and shift == 'load')):
                markers[p] = int(rng.choice(sources))
    for p, source in markers.items():
        core[p] = source
    mapping = {}
    if demo == 'collision':
        mapping[sources[0]] = sources[1]
        mapping[sources[1]] = available[k]
    for source in sources:
        if source not in mapping:
            choices = available[k:] if demo == 'single' else [c for c in available if c != source]
            mapping[source] = int(rng.choice(choices))
    stencils = {s: _mask(rng, n, informative=bool(demo)) for s in sources}
    if demo == 'collision' and np.array_equal(stencils[sources[0]], stencils[sources[1]]):
        stencils[sources[1]] = stencils[sources[0]].T.copy()
    if demo == 'collision':
        # Explicitly teach spatial unfolding around a bend, including in a
        # minimal two-example puzzle. A random hole must not hide every turn.
        jobs = _jobs(paths, markers, n*n)
        bent = [(s, j) for s, origin, cells in jobs for j, p in enumerate(cells) if p[0] % 2]
        assert bent
        if not any(stencils[s].ravel()[j] for s, j in bent):
            s, j = bent[0]
            mask = stencils[s].copy()
            mask.ravel()[j] = True
            if mask.all() or np.array_equal(mask.ravel(), mask.ravel()[::-1]):
                for zero in range(1, n*n-1):
                    if zero == j:
                        continue
                    trial = mask.copy()
                    trial.ravel()[zero] = False
                    if not np.array_equal(trial.ravel(), trial.ravel()[::-1]):
                        mask = trial
                        break
            stencils[s] = mask
    # The canonical frame includes a blank top/side rim and two bottom layers.
    top, left = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    bottom = 3  # blank separation, then tab layer, then border
    bh, bw = core.shape[0]+top+bottom, core.shape[1]+left+2
    # Upper-shelf fallback must always fit even for the largest word.
    if bh+n+4 > 30:
        top = 2
        bh = core.shape[0]+top+bottom
    shape, (r, c), positions, _ = _place_clues(rng, bh, bw, n, sources)
    grid = np.full(shape, bg, dtype=np.int8)
    _draw_frame(grid, r, c, bh, bw, frame)
    grid[r+top:r+top+core.shape[0], c+left:c+left+core.shape[1]] = core
    for source in sources:
        a, b = positions['stencil', source]
        grid[a:a+n, b:b+n][stencils[source]] = source
        a, b = positions['key', source]
        grid[a:a+2, b:b+2] = [[source, source], [source, mapping[source]]]
    # Construct the label using known routes and assignments, before parsing.
    construction = dict(net, core=core, markers=markers, n=n, track=track,
                        mapping=mapping, stencils=stencils)
    output = grid[r:r+bh, c:c+bw].copy()
    output[top:top+core.shape[0], left:left+core.shape[1]] = _print_core(construction)
    a, b, x, y = _box(np.argwhere(grid != bg))
    a, x = max(0, a-int(rng.integers(3))), max(0, x-int(rng.integers(3)))
    b, y = min(shape[0], b+int(rng.integers(3))), min(shape[1], y+int(rng.integers(3)))
    grid = grid[a:b, x:y].copy()
    rotation = int(rng.integers(4))
    return {'input': np.rot90(grid, rotation).copy(),
            'output': np.rot90(output, rotation).copy()}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(int(seed))
    params = dict(shift=str(rng.choice(['words', 'routes', 'load'])),
                  single_n=int(rng.choice([2, 3])),
                  roles=list(map(int, rng.permutation(10)[:4])),
                  vary_roles=bool(rng.random() < .55))
    demos = list(map(int, rng.permutation(num_train)[:2]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            demo = ('single' if index == demos[0] else 'collision' if index == demos[1] else '') if split == 'train' else ''
            for _ in range(30):
                pair = _make_pair(rng, params, split, demo)
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not generate distinct inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    n, markers = info['n'], info['markers']
    jobs = _jobs(info['paths'], markers, n*n)
    single = level = mixed = 0
    for path in info['paths']:
        ends = [p for p in (path[0], path[-1]) if p in markers]
        if path[0][0] == path[-1][0]:
            level += len(ends)
        else:
            single += len(ends) == 1 and ends[0] == min(path[0], path[-1])
            mixed += len(ends) == 2 and markers[ends[0]] != markers[ends[1]]
    return dict(n=n, lanes=(info['core'].shape[0]+1)//2, width=info['core'].shape[1],
                categories=len(info['mapping']), markers=len(markers),
                dividers=len(info['centers']), breaks=len(info['breaks']),
                row_counts=info['row_counts'], overlaps=info['overlaps'],
                single=single, level=level, mixed=mixed, rotation=info['q'],
                bent_ink=any(bit and p[0] % 2 == 1
                             for s, origin, cells in jobs
                             for p, bit in zip(cells, info['stencils'][s].ravel())),
                disconnected=sum(len(_components(m)) > 1 for m in info['stencils'].values()),
                holes=sum(int((~m).sum()) for m in info['stencils'].values()),
                overlap_map=any(t in info['mapping'] for t in info['mapping'].values()),
                palette=(info['background'], info['track'], info['divider'], info['frame']),
                tape_signatures=[m.tobytes().hex() for m in info['stencils'].values()])


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            assert np.array_equal(out, _answer(info)), 'incorrect ribbon placement'
            # All stencil slots, including holes, occupy disjoint track cells.
            jobs = _jobs(info['paths'], info['markers'], info['n']**2)
            occupied = [p for _, _, cells in jobs for p in cells]
            assert len(occupied) == len(set(occupied)) == len(info['markers'])*info['n']**2
            assert all(info['structural'][p] == info['track'] for p in occupied)
            # Paint colors are disjoint from infrastructure: exact ink mass is measurable.
            expected_ink = sum(int(info['stencils'][s].sum()) for s in info['markers'].values())
            actual_ink = int(np.isin(_print_core(info), list(set(info['mapping'].values()))).sum())
            assert actual_ink == expected_ink
            r, e, c, d = info['frame_box']
            canonical_out = np.rot90(out, info['q'])
            frame_mask = info['grid'][r:e, c:d] == info['frame']
            assert np.array_equal(canonical_out == info['frame'], frame_mask)
            if split == 'train':
                train.append(info)
    features = [_features(a) for a in train]
    assert any(f['single'] for f in features) and any(f['level'] for f in features)
    assert any(f['mixed'] >= 2 for f in features)
    assert any(f['bent_ink'] for f in features), 'training must show ink passing through a bend'
    assert any(f['overlap_map'] for f in features)
    for alternative in ('straight', 'unit', 'solid', 'compact', 'reverse_bits', 'column_bits',
                        'reverse_order', 'identity_ink', 'recursive'):
        assert any(not np.array_equal(_answer(a), _answer(a, alternative)) for a in train), alternative


def _puzzle_key(puzzle):
    return tuple((s, _key(p['input']), _key(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def _regression_cases():
    """Hand-authored examples check coordinates beyond renderer self-agreement."""
    def fixture(core, masks, mapping):
        n = next(iter(masks.values())).shape[0]
        bh, bw, top = core.shape[0]+5, core.shape[1]+4, n+4
        inp = np.zeros((top+bh, bw), dtype=np.int8)
        _draw_frame(inp, top, 0, bh, bw, 5)
        inp[top+2:top+2+core.shape[0], 2:2+core.shape[1]] = core
        for i, s in enumerate(sorted(masks)):
            c = i*(n+1)
            inp[:n, c:c+n][masks[s]] = s
            inp[n+1:n+3, c:c+2] = [[s, s], [s, mapping[s]]]
        blank_board = inp[top:].copy()
        return inp, blank_board

    def expected(board, core):
        out = board.copy()
        out[2:2+core.shape[0], 2:2+core.shape[1]] = core
        return out

    def digits(rows):
        return np.array([[int(c) for c in row] for row in rows], dtype=np.int8)

    masks = {1: np.array([[1, 0], [1, 1]], bool),
             2: np.array([[1, 1], [0, 1]], bool)}
    core = digits(['2888888888888', '0007770000000', '1888888888882'])
    inp, board = fixture(core, masks, {1: 2, 2: 3})
    known = digits(['8383788888888', '0003780000000', '2822788883833'])
    out = expected(board, known)
    assert np.array_equal(solve(inp), out)
    # Reversing endpoint identities reverses whole ribbon order, not their bits.
    swapped = inp.copy()
    swapped[8, 2], swapped[10, 2] = 1, 2
    swap_known = digits(['8228788888888', '0002780000000', '3383788883833'])
    assert np.array_equal(solve(swapped), expected(board, swap_known))
    # Relocating either clue object cannot change the printed board.
    for a, size in ((0, 2), (3, 2)):
        moved = inp.copy()
        tile = moved[a:a+size, :size].copy()
        moved[a:a+size, :size] = 0
        moved[a:a+size, 10:10+size] = tile
        assert np.array_equal(solve(moved), out)
    changed_mask = inp.copy()
    changed_mask[1, 0] = 0  # turn the L into a disconnected diagonal
    changed_known = known.copy()
    changed_known[2, 2] = 8
    assert np.array_equal(solve(changed_mask), expected(board, changed_known))

    # Two words on a level lane grow from opposite original terminals.
    masks3 = {1: np.eye(3, dtype=bool),
              2: np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], bool)}
    level_core = _base(3, 21, [(1, 10)], [], 0, 8, 7)
    level_core[4, 0], level_core[4, 20] = 1, 2
    level_in, level_board = fixture(level_core, masks3, {1: 2, 2: 3})
    level_known = np.zeros((5, 21), dtype=np.int8)
    level_known[::2] = 8
    level_known[:3, 10] = 7
    level_known[1, [9, 11]] = 8
    level_known[4, [0, 4, 8]] = 2
    level_known[4, [20, 19, 16, 12]] = 3
    assert np.array_equal(solve(level_in), expected(level_board, level_known))

    # An n=4 mixed collision, with a hand-enumerated 57-cell three-lane path.
    y, x = np.indices((4, 4))
    masks4 = {1: np.eye(4, dtype=bool),
              2: (x == 0) | (x == 3) | (y == 0) | (y == 3)}
    long_core = _base(3, 25, [(1, 4), (3, 20)], [], 0, 8, 7)
    long_core[4, 0], long_core[0, 24] = 1, 2
    long_in, long_board = fixture(long_core, masks4, {1: 2, 2: 3})
    long_known = np.zeros((5, 25), dtype=np.int8)
    long_known[::2] = 8
    long_known[:3, 4] = long_known[2:, 20] = 7
    long_known[1, [3, 5]] = long_known[3, [19, 21]] = 8
    manual_path = ([(4, c) for c in range(20)] + [(3, 19)] +
                   [(2, c) for c in range(19, 4, -1)] + [(1, 5)] +
                   [(0, c) for c in range(5, 25)])
    for offset, source, target in ((0, 1, 2), (16, 2, 3)):
        for p, bit in zip(manual_path[offset:offset+16], masks4[source].ravel()):
            if bit:
                long_known[p] = target
    assert np.array_equal(solve(long_in), expected(long_board, long_known))

    invalid = [lambda: _rewire(_base(2, 13, [(1, 4), (1, 8)], [], 0, 8, 7), 0, 8, 7, set()),
               lambda: _rewire(_base(3, 13, [(1, 4), (3, 4)], [], 0, 8, 7), 0, 8, 7, set()),
               lambda: _rewire(_base(3, 13, [(1, 4), (3, 5)], [], 0, 8, 7), 0, 8, 7, set())]
    too_long, _ = fixture(core, masks3, {1: 2, 2: 3})
    invalid.append(lambda: solve(too_long))
    interior_packet = inp.copy()
    interior_packet[8, 2] = 8
    interior_packet[8, 11] = 2
    invalid.append(lambda: solve(interior_packet))
    for case in invalid:
        try:
            case()
        except AssertionError:
            pass
        else:
            raise AssertionError('invalid network, capacity or marker placement accepted')


def self_test():
    """Bounded semantic checks and measured structural variation over 42 seeds."""
    _regression_cases()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(puzzles[0])
    records, normalized, shapes, masks = [], set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                unchanged = inp.copy()
                assert np.array_equal(solve(inp), out)
                assert np.array_equal(inp, unchanged), 'solve mutated input'
                info = _analyze(inp)
                record = _features(info)
                record['split'] = split
                records.append(record)
                shapes.add((inp.shape, out.shape))
                # Normalize literal colors, while preserving geometry and associations.
                roles = {}
                norm = np.array([[roles.setdefault(int(c), len(roles)) for c in row] for row in inp], np.int8)
                normalized.add(_key(norm))
                masks.update((info['n'], s) for s in record['tape_signatures'])
    assert {r['n'] for r in records} == {2, 3, 4}
    assert {r['categories'] for r in records} == {1, 2, 3, 4}
    assert {r['rotation'] for r in records} == {0, 1, 2, 3}
    assert {r['lanes'] for r in records} >= {3, 4, 6, 7, 8}
    assert len(normalized) >= 180 and len(shapes) >= 70 and len(masks) >= 60
    assert len({r['dividers'] for r in records}) >= 8
    assert len({r['markers'] for r in records}) >= 10
    assert any(r['overlaps'] for r in records) and any(max(r['row_counts']) >= 2 for r in records)
    assert any(r['breaks'] for r in records) and any(not r['breaks'] for r in records)
    assert any(r['bent_ink'] for r in records) and sum(r['disconnected'] for r in records) >= 30
    assert {r['palette'][i] for r in records for i in range(4)} == set(range(10))
    rng = np.random.default_rng(987654)
    for puzzle in puzzles[:6]:
        for pair in puzzle['train'] + puzzle['test']:
            # Semantic colors and tab orientation, rather than literals, determine the answer.
            permutation = rng.permutation(10)
            assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
            for q in (1, 2, 3):
                assert np.array_equal(solve(np.rot90(pair['input'], q)), np.rot90(pair['output'], q))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(54321, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    corrupt = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    corrupt['test'][0]['output'][0, 0] = (int(corrupt['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted label accepted')
    return dict(seeds=42, pairs=len(records), distinct_inputs=len(normalized),
                shape_pairs=len(shapes), stencil_patterns=len(masks),
                lane_counts=sorted({r['lanes'] for r in records}),
                divider_counts=sorted({r['dividers'] for r in records}),
                packet_counts=sorted({r['markers'] for r in records}),
                disconnected_stencils=sum(r['disconnected'] for r in records),
                compatible_overlap_pairs=sum(bool(r['overlaps']) for r in records),
                bent_ink_pairs=sum(bool(r['bent_ink']) for r in records))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
