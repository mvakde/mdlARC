"""Expandable circuit switches. Standard library + NumPy; no external state.

The key above the 8-colored divider is part of every input. In the work field,
code points mark square sockets. Their trace ports determine rotation and ink.
After expansion and repair, 9 reroutes a terminal branch through its adjacent
pivot. A module is graph geometry, not an indivisible object to keep or erase.
"""
from __future__ import annotations

import copy
import json
from collections import Counter

import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
PAYLOAD = (1, 2, 3, 4, 5, 7)
BACKGROUND, REPAIR, FRAME, DIRECTION = 0, 6, 8, 9


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _neighbors(p, cells):
    return [q for d in DIRS if (q := _add(p, d)) in cells]


def _cells(mask):
    return {tuple(map(int, p)) for p in np.argwhere(mask)}


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            for q in _neighbors(stack.pop(), remaining):
                remaining.remove(q)
                group.add(q)
                stack.append(q)
        result.append(group)
    return result


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    dr, dc = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
    return [(a[0] + i * dr, a[1] + i * dc)
            for i in range(max(abs(a[0] - b[0]), abs(a[1] - b[1])) + 1)]


def _polyline(points):
    result = [points[0]]
    for a, b in zip(points, points[1:]):
        result.extend(_line(a, b)[1:])
    return result


def _ports(mask):
    """Endpoint coordinates and outward normals; corners are not ports."""
    cells, s = _cells(mask), mask.shape[0]
    result = []
    for r, c in sorted(cells):
        if len(_neighbors((r, c), cells)) != 1:
            continue
        sides = [d for d in DIRS if not _inside(_add((r, c), d), mask.shape)]
        if len(sides) == 1:
            result.append(((r, c), sides[0]))
    assert len(result) in (2, 3)
    assert all(0 < r < s - 1 or 0 < c < s - 1 for (r, c), _ in result)
    return result


def _orientations(mask):
    result, seen = [], set()
    for k in range(4):
        m = np.rot90(mask, k).copy()
        signature = m.tobytes()
        if signature not in seen:
            result.append(m)
            seen.add(signature)
    return result


def _check_mask(mask):
    s = mask.shape[0]
    assert mask.shape == (s, s) and 3 <= s <= 5
    cells = _cells(mask)
    assert len(_components(cells)) == 1
    degrees = [len(_neighbors(p, cells)) for p in cells]
    assert sum(degrees) == 2 * (len(cells) - 1), 'motif must be a tree'
    assert max(degrees) <= 3
    ports = _ports(mask)
    assert sum(d == 1 for d in degrees) == len(ports), 'all leaves must be ports'
    assert sum(d == 3 for d in degrees) == len(ports) - 2
    # Port alignment must distinguish every geometrically different rotation.
    signatures = [tuple(sorted(p for p, _ in _ports(m))) for m in _orientations(mask)]
    assert len(set(signatures)) == len(signatures), 'ambiguous socket orientation'


def _read_key(inp):
    separators = np.flatnonzero(np.all(inp == FRAME, axis=1))
    assert len(separators) == 1, 'one full-width divider is required'
    divider = int(separators[0])
    assert 6 <= divider <= 8 and divider + 1 < inp.shape[0]
    header = inp[:divider]
    book = {}
    frames = _components(_cells(header == FRAME))
    assert len(frames) == 2, 'the key has two framed motifs'
    for cells in frames:
        r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
        c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
        assert r0 == 0 and r1 - r0 == c1 - c0
        expected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                    if r in (r0, r1) or c in (c0, c1)}
        assert cells == expected, 'key frames must be complete squares'
        interior = header[r0 + 1:r1, c0 + 1:c1]
        colors = set(map(int, np.unique(interior))) - {BACKGROUND}
        assert len(colors) == 1
        code = colors.pop()
        assert code in PAYLOAD and code not in book
        mask = interior == code
        _check_mask(mask)
        book[code] = mask
    assert len({m.shape for m in book.values()}) == 1
    assert sorted(len(_ports(m)) for m in book.values()) == [2, 3]
    allowed = {BACKGROUND, FRAME} | set(book)
    assert set(map(int, np.unique(header))) <= allowed
    # Key material only occurs inside these frames.
    covered = np.zeros(header.shape, bool)
    for cells in frames:
        r1, c0, c1 = max(r for r, _ in cells), min(c for _, c in cells), max(c for _, c in cells)
        covered[:r1 + 1, c0:c1 + 1] = True
    assert not np.any((header != 0) & ~covered)
    return divider + 1, book


def _expand(inp, book):
    """Read square sockets, orient from all ports, then copy with local ink."""
    shape = inp.shape
    s = next(iter(book.values())).shape[0]
    codes = set(book)
    traces = set(map(int, np.unique(inp))) - codes - {0, 6, 8, 9}
    assert traces and traces <= set(PAYLOAD)
    assert FRAME not in inp
    windows = set()
    records = []
    for r, c in sorted(_cells(np.isin(inp, list(codes)))):
        assert r + s <= shape[0] and c + s <= shape[1], 'socket must fit'
        window = {(r + a, c + b) for a in range(s) for b in range(s)}
        assert not windows & window, 'overlapping sockets'
        windows.update(window)
        code = int(inp[r, c])
        block = inp[r:r + s, c:c + s]
        assert set(map(int, np.unique(block))) <= {0, code, 6, 9}
        assert np.count_nonzero(block == code) == 1
        touches = set()
        inks = set()
        for a in range(s):
            for b, d in (((a, 0), (0, -1)), ((a, s - 1), (0, 1)),
                         ((0, a), (-1, 0)), ((s - 1, a), (1, 0))):
                q = _add((r + b[0], c + b[1]), d)
                if _inside(q, shape) and int(inp[q]) in traces | {REPAIR}:
                    touches.add(b)
                    if int(inp[q]) in traces:
                        inks.add(int(inp[q]))
        assert len(inks) == 1, 'all socket ports must witness one ink'
        fits = [m for m in _orientations(book[code])
                if {p for p, _ in _ports(m)} == touches]
        assert len(fits) == 1, 'ports must specify one rotation'
        mask, ink = fits[0], inks.pop()
        cells = {(r + a, c + b) for a, b in _cells(mask)}
        assert all(int(inp[q]) != DIRECTION for q in cells)
        assert all(q in cells for q in _cells(inp == REPAIR) if q in window)
        records.append({'origin': (r, c), 'code': code, 'ink': ink,
                        'mask': mask, 'cells': cells, 'window': window})
    assert records
    layers = {color: _cells(inp == color) for color in traces}
    for record in records:
        layers[record['ink']].update(record['cells'])
    pending = _cells(inp == REPAIR) - windows
    for _ in range(len(pending) + 1):
        changed = False
        for p in sorted(pending):
            candidates = {c for c, cells in layers.items() if _neighbors(p, cells)}
            assert len(candidates) <= 1, 'ambiguous repair ink'
            if candidates:
                layers[candidates.pop()].add(p)
                pending.remove(p)
                changed = True
        if not changed:
            break
    assert not pending, 'repair marker has no trace witness'
    # Fill isolated straight omissions outside sockets, simultaneously. Explicit
    # template blanks, directional cells and code points are never interpolated.
    additions = {c: set() for c in layers}
    for r, c in np.argwhere(inp == BACKGROUND):
        p = (int(r), int(c))
        if p in windows:
            continue
        candidates = set()
        for color, cells in layers.items():
            for d in ((1, 0), (0, 1)):
                if _add(p, d) in cells and _add(p, (-d[0], -d[1])) in cells:
                    candidates.add(color)
        assert len(candidates) <= 1, 'gap has conflicting witnesses'
        if candidates:
            additions[candidates.pop()].add(p)
    for c in layers:
        layers[c].update(additions[c])
    assert sum(map(len, layers.values())) == len(set().union(*layers.values())), 'initial layers must not cross'
    return layers, records, windows, set().union(*additions.values())


def _walk_branch(cells, pivot, direction, shape=None):
    previous, current = pivot, _add(pivot, direction)
    assert current in cells
    walk = []
    for _ in range(len(cells)):
        assert current not in walk, 'terminal branch cannot loop'
        walk.append(current)
        following = [q for q in _neighbors(current, cells) if q != previous]
        if not following:
            if shape is not None:
                assert current[0] in (0, shape[0] - 1) or current[1] in (0, shape[1] - 1)
            return walk
        assert len(following) == 1, 'old branch must not fork'
        previous, current = current, following[0]
    raise AssertionError('no terminal endpoint')


def _edits(inp, layers):
    edits = []
    for q in sorted(_cells(inp == DIRECTION)):
        candidates = []
        for d in DIRS:
            pivot = _add(q, (-d[0], -d[1]))
            for color, cells in layers.items():
                if pivot not in cells or q in cells or _add(pivot, (-d[0], -d[1])) not in cells:
                    continue
                perp = (-d[1], d[0])
                if not (_add(pivot, perp) in cells or _add(pivot, (-perp[0], -perp[1])) in cells):
                    continue
                try:
                    walk = _walk_branch(cells, pivot, (-d[0], -d[1]), inp.shape)
                except AssertionError:
                    continue
                ray, p = set(), q
                while _inside(p, inp.shape):
                    ray.add(p)
                    p = _add(p, d)
                candidates.append({'pivot': pivot, 'ink': color, 'direction': d,
                                   'old': set(walk), 'walk': walk, 'ray': ray})
        assert len(candidates) == 1, 'direction marker must select one terminal route'
        edits.append(candidates[0])
    assert edits
    for i, a in enumerate(edits):
        for b in edits[i + 1:]:
            assert not (a['ink'] == b['ink'] and a['old'] & b['old'])
            assert a['pivot'] not in b['old'] and b['pivot'] not in a['old']
            assert a['ink'] == b['ink'] or not a['ray'] & b['ray'], 'different-ink rays cannot meet'
    return edits


def _render(shape, layers, edits):
    kept = {c: set(cells) for c, cells in layers.items()}
    for e in edits:
        kept[e['ink']].difference_update(e['old'])
    out = np.zeros(shape, np.int8)
    for color, cells in kept.items():
        for p in cells:
            out[p] = color
    for e in edits:  # New rays are foreground over any retained material.
        for p in e['ray']:
            out[p] = e['ink']
    return out


def _analyse(input_grid):
    inp = _grid(input_grid)
    start, book = _read_key(inp)
    scene = inp[start:]
    layers, records, windows, gaps = _expand(scene, book)
    edits = _edits(scene, layers)
    return inp, start, book, scene, layers, records, windows, gaps, edits


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from this input, including its visible key."""
    inp, start, _, scene, layers, _, _, _, edits = _analyse(input_grid)
    out = inp.copy()
    out[start:] = _render(scene.shape, layers, edits)
    return out


def _sample_masks(rng, s):
    row = s - 2
    cp = int(rng.integers(1, s - 1))
    ct = int(rng.integers(1, s - 1)) if s == 5 else cp
    switch = np.zeros((s, s), bool)
    switch[row, :] = True
    bend = 1 if ct != cp else 0
    switch_cells = _polyline([(0, ct), (bend, ct), (bend, cp), (row, cp)])
    for p in switch_cells:
        switch[p] = True
    _check_mask(switch)
    for _ in range(100):
        a, b = map(int, rng.integers(1, s - 1, size=2))
        v = int(rng.integers(1, s - 1))
        wire = np.zeros((s, s), bool)
        for p in _polyline([(a, 0), (a, v), (b, v), (b, s - 1)]):
            wire[p] = True
        try:
            _check_mask(wire)
        except AssertionError:
            continue
        return switch, wire, (row, cp)
    raise AssertionError('could not sample a port-distinguishable wire')


def _translated(mask, origin):
    return {(origin[0] + a, origin[1] + b) for a, b in _cells(mask)}


def _port(record, direction):
    matches = [(p, d) for p, d in _ports(record['mask']) if d == direction]
    assert len(matches) == 1
    p, d = matches[0]
    return _add(record['origin'], p), d


def _try_route(graph, windows, a, normal_a, b, normal_b, shape, rng):
    """Connect endpoints without incidental contacts or entering sockets."""
    p = _add(a, normal_a)
    q = _add(b, normal_b) if normal_b is not None else b
    if not _inside(p, shape) or not _inside(q, shape):
        return None
    middle = [[p, (p[0], q[1]), q], [p, (q[0], p[1]), q]]
    rows = list(range(shape[0]))
    cols = list(range(shape[1]))
    rng.shuffle(rows)
    rng.shuffle(cols)
    middle += [[p, (r, p[1]), (r, q[1]), q] for r in rows]
    middle += [[p, (p[0], c), (q[0], c), q] for c in cols]
    # Prefer short, but keep stochastic ties and occasional winding routes.
    rng.shuffle(middle)
    middle.sort(key=lambda points: sum(abs(x[0] - y[0]) + abs(x[1] - y[1])
                                      for x, y in zip(points, points[1:])))
    allowed = {a, b} if normal_b is not None else {a}
    for points in middle:
        walk = _polyline([a] + points + ([b] if normal_b is not None else []))
        # Repeated vertices caused only by zero-length segments are harmless.
        walk = [v for i, v in enumerate(walk) if i == 0 or v != walk[i - 1]]
        cells = set(walk)
        if len(cells) != len(walk) or any(not _inside(v, shape) for v in cells):
            continue
        if any(len(_neighbors(v, cells)) > 2 for v in cells):
            continue
        fresh = cells - allowed
        if fresh & (graph | windows):
            continue
        if any(set(_neighbors(v, windows)) - allowed for v in fresh):
            continue  # Every socket contact must be an actual endpoint port.
        if any(set(_neighbors(v, graph)) - allowed for v in fresh):
            continue
        if any(len(_neighbors(v, graph | cells)) != 2 for v in allowed):
            continue
        graph.update(cells)
        return cells
    return None


def _terminal(graph, windows, record, direction, shape, rng, boundary):
    a, normal = _port(record, direction)
    if boundary:
        # Any old route endpoint on the requested edge is legitimate; the
        # module chain and connecting elbows determine how it reaches that edge.
        if direction[0]:
            targets = [(0 if direction[0] < 0 else shape[0] - 1, c)
                       for c in range(1, shape[1] - 1)]
        else:
            targets = [(r, 0 if direction[1] < 0 else shape[1] - 1)
                       for r in range(1, shape[0] - 1)]
    else:
        targets = []
        for distance in range(2, 7):
            for offset in range(-3, 4):
                perp = (-normal[1], normal[0])
                q = (a[0] + distance * normal[0] + offset * perp[0],
                     a[1] + distance * normal[1] + offset * perp[1])
                if _inside(q, shape):
                    targets.append(q)
    rng.shuffle(targets)
    for b in targets:
        route = _try_route(graph, windows, a, normal, b, None, shape, rng)
        if route is not None:
            return route
    return None


def _canonical_scene(rng, switch, wire, pivot_local, n_old, retained_wire):
    s = switch.shape[0]
    limit = 26 - s  # Either rotation still fits below the key in a 30-row grid.
    h = int(rng.integers(max(2 * s + 5, (n_old + 1) * s + n_old + 2), limit + 1))
    w = int(rng.integers(2 * s + 7, limit + 1))
    min_main = n_old * (s + 1) + 1
    main_r = int(rng.integers(min_main, h - s))
    main_c = int(rng.integers(s + 2, w - s))
    records = [{'kind': 'switch', 'origin': (main_r, main_c), 'mask': switch}]
    slack = main_r - min_main
    extra = rng.multinomial(slack, np.full(n_old + 1, 1 / (n_old + 1)))
    row = 1 + int(extra[0])
    old_records = []
    for i in range(n_old):
        rec = {'kind': 'wire', 'origin': (row, 0),
               'mask': np.rot90(wire, int(rng.choice((1, 3)))).copy()}
        old_records.append(rec)
        records.append(rec)
        row += s + 1 + int(extra[i + 1])
    # Align successive external ports. Internal detours in the wire motifs
    # still shift the route sideways, so the chain is not a magnified line.
    target_col = main_c + next(p[1] for p, d in _ports(switch) if d == (-1, 0))
    for rec in reversed(old_records):
        bottom = next(p for p, d in _ports(rec['mask']) if d == (1, 0))
        col = target_col - bottom[1]
        if not 1 <= col < w - s:
            return None
        rec['origin'] = (rec['origin'][0], col)
        top = next(p for p, d in _ports(rec['mask']) if d == (-1, 0))
        target_col = col + top[1]
    stable = None
    if retained_wire:
        wc = int(rng.integers(1, main_c - s))
        stable = {'kind': 'wire', 'origin': (0, wc),
                  'mask': np.rot90(wire, int(rng.choice((0, 2)))).copy()}
        right_row = next(p[0] for p, d in _ports(stable['mask']) if d == (0, 1))
        wr = main_r + pivot_local[0] - right_row
        if not 1 <= wr < h - s:
            return None
        stable['origin'] = (wr, wc)
        records.append(stable)
    windows, graph = set(), set()
    for rec in records:
        r, c = rec['origin']
        rec['window'] = {(r + a, c + b) for a in range(s) for b in range(s)}
        rec['cells'] = _translated(rec['mask'], rec['origin'])
        if windows & rec['window'] or any(_neighbors(p, graph) for p in rec['cells']):
            return None
        windows.update(rec['window'])
        graph.update(rec['cells'])
    main = records[0]
    pivot = _add(main['origin'], pivot_local)
    old = set(_walk_branch(main['cells'], pivot, (-1, 0)))
    previous = main
    for rec in reversed(old_records):
        a, da = _port(previous, (-1, 0))
        b, db = _port(rec, (1, 0))
        route = _try_route(graph, windows, a, da, b, db, (h, w), rng)
        if route is None:
            return None
        old.update(route)
        old.update(rec['cells'])
        previous = rec
    route = _terminal(graph, windows, previous, (-1, 0), (h, w), rng, True)
    if route is None:
        return None
    old.update(route)
    if stable is not None:
        a, da = _port(main, (0, -1))
        b, db = _port(stable, (0, 1))
        if _try_route(graph, windows, a, da, b, db, (h, w), rng) is None:
            return None
        left = stable
    else:
        left = main
    if _terminal(graph, windows, left, (0, -1), (h, w), rng, False) is None:
        return None
    if _terminal(graph, windows, main, (0, 1), (h, w), rng, False) is None:
        return None
    if len(_components(graph)) != 1:
        return None
    actual_old = set(_walk_branch(graph, pivot, (-1, 0), (h, w)))
    if actual_old != old:
        return None
    return (h, w), graph, records, windows, pivot, old


def _transform_mask(mask, k, reflect):
    return np.rot90(np.fliplr(mask) if reflect else mask, k).copy()


def _transform_cells(cells, shape, k, reflect):
    mask = np.zeros(shape, bool)
    for p in cells:
        mask[p] = True
    return _cells(_transform_mask(mask, k, reflect))


def _header(rng, width, book):
    s = next(iter(book.values())).shape[0]
    header = np.zeros((s + 4, width), np.int8)
    cards = list(book.items())
    rng.shuffle(cards)
    space = width - 2 * (s + 2) - 1
    left = int(rng.integers(space + 1))
    gap = int(rng.integers(1, space - left + 2))
    for i, (code, mask) in enumerate(cards):
        c = left + i * (s + 2 + gap)
        header[0, c:c + s + 2] = header[s + 1, c:c + s + 2] = FRAME
        header[:s + 2, c] = header[:s + 2, c + s + 1] = FRAME
        header[1:s + 1, c + 1:c + s + 1][mask] = code
    header[-1] = FRAME
    return header


def _make_pair(rng, params, n_old, retained_wire, k, marked_pivot):
    switch, wire, local_pivot = params['canonical']
    for _ in range(120):
        built = _canonical_scene(rng, switch, wire, local_pivot, n_old, retained_wire)
        if built is None:
            continue
        shape, graph, records, windows, pivot, old = built
        ink = int(rng.choice(params['inks']))
        # Rotate geometry, not the code point. Socket origins are transformed
        # as rectangles and then re-encoded at their new top-left.
        reflect = False
        graph_t = _transform_cells(graph, shape, k, reflect)
        windows_t = _transform_cells(windows, shape, k, reflect)
        old_t = _transform_cells(old, shape, k, reflect)
        pivot_t = next(iter(_transform_cells({pivot}, shape, k, reflect)))
        marker_t = next(iter(_transform_cells({_add(pivot, (1, 0))}, shape, k, reflect)))
        final_shape = shape if k % 2 == 0 else shape[::-1]
        scene = np.zeros(final_shape, np.int8)
        for p in graph_t - windows_t:
            scene[p] = ink
        for rec in records:
            win = _transform_cells(rec['window'], shape, k, reflect)
            origin = (min(r for r, _ in win), min(c for _, c in win))
            scene[origin] = params['codes'][rec['kind']]
        scene[marker_t] = DIRECTION
        if marked_pivot:
            scene[pivot_t] = REPAIR
        # Ground truth comes from the explicitly constructed old-route set.
        truth = np.zeros(final_shape, np.int8)
        for p in graph_t - old_t:
            truth[p] = ink
        d = (marker_t[0] - pivot_t[0], marker_t[1] - pivot_t[1])
        p = marker_t
        while _inside(p, final_shape):
            truth[p] = ink
            p = _add(p, d)
        header = _header(rng, scene.shape[1], params['book'])
        inp = np.vstack((header, scene))
        out = np.vstack((header, truth))
        try:
            details = _analyse(inp)
            if details[4] != {ink: graph_t} or not np.array_equal(solve(inp), out):
                continue
        except AssertionError:
            continue
        # Add ordinary repair annotations both inside and outside modules.
        eligible = sorted(graph_t - {origin for origin in _cells(np.isin(scene, list(params['book'])))}
                          - {marker_t, pivot_t})
        rng.shuffle(eligible)
        wanted = int(rng.integers(1, 5))
        added = 0
        for p in eligible:
            if scene[p] == DIRECTION:
                continue
            original = int(scene[p])
            scene[p] = REPAIR
            candidate = np.vstack((header, scene))
            try:
                a = _analyse(candidate)
                assert a[4] == {ink: graph_t}
                assert np.array_equal(solve(candidate), out)
            except AssertionError:
                scene[p] = original
            else:
                added += 1
                if added >= wanted:
                    break
        # Sparse straight gaps are proposed outside modules. They may lie on
        # deleted or retained branches; equality with latent geometry guards
        # against accidental additional interpolation.
        candidates = [p for p in graph_t - windows_t if scene[p] == ink
                      and any(_add(p, v) in graph_t and _add(p, (-v[0], -v[1])) in graph_t
                              for v in ((1, 0), (0, 1)))]
        rng.shuffle(candidates)
        gaps = 0
        for p in candidates:
            scene[p] = BACKGROUND
            candidate = np.vstack((header, scene))
            try:
                a = _analyse(candidate)
                assert a[4] == {ink: graph_t}
                assert np.array_equal(solve(candidate), out)
            except AssertionError:
                scene[p] = ink
            else:
                gaps += 1
                if gaps >= int(rng.integers(1, 4)):
                    break
        return {'input': np.vstack((header, scene)), 'output': out}
    raise AssertionError('bounded circuit construction exhausted')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    nt = _count(num_train, 'num_train', 2, 10)
    nq = _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    s = int(rng.integers(3, 6))
    switch, wire, pivot = _sample_masks(rng, s)
    palette = list(map(int, rng.permutation(PAYLOAD)))
    codes = {'switch': palette[0], 'wire': palette[1]}
    params = {'canonical': (switch, wire, pivot), 'codes': codes,
              'book': {codes['switch']: np.rot90(switch, int(rng.integers(4))).copy(),
                       codes['wire']: np.rot90(wire, int(rng.integers(4))).copy()},
              'inks': palette[2:]}
    base = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for i in range(nt + nq):
        split = 'train' if i < nt else 'test'
        if i == 0:
            n_old, stable, k, marked = 0, True, base, False
        elif i == 1:
            n_old, stable, k, marked = 1, False, (base + 1) % 4, True
        elif split == 'train':
            n_old, stable = int(rng.integers(2)), bool(rng.integers(2))
            k, marked = int(rng.integers(4)), bool(rng.integers(2))
        else:
            n_old, stable = 2, bool(rng.integers(2))
            k, marked = (base + 2 + i - nt) % 4, True
        for _ in range(12):
            pair = _make_pair(rng, params, n_old, stable, k, marked)
            if _key(pair['input']) not in seen:
                break
        else:
            raise AssertionError('distinct-input construction exhausted')
        seen.add(_key(pair['input']))
        puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def measure_pair(pair):
    inp, start, book, scene, layers, records, windows, gaps, edits = _analyse(pair['input'])
    old = set().union(*(e['old'] for e in edits))
    full = set().union(*layers.values())
    wire_codes = {c for c, m in book.items() if len(_ports(m)) == 2}
    partial = sum(bool(r['cells'] & old) and bool(r['cells'] - old) for r in records)
    return {'shape': list(inp.shape), 'size': next(iter(book.values())).shape[0],
            'modules': len(records), 'deleted_wires': sum(r['code'] in wire_codes and r['cells'] <= old for r in records),
            'retained_wires': sum(r['code'] in wire_codes and not r['cells'] & old for r in records),
            'partial_modules': partial, 'ink': sorted(layers),
            'direction': [list(e['direction']) for e in edits], 'old_length': sum(len(e['old']) for e in edits),
            'old_bends': sum(sum(a != b for a, b in zip(ds, ds[1:])) for e in edits
                             for ds in [[(b[0] - a[0], b[1] - a[1])
                                         for a, b in zip([e['pivot']] + e['walk'], e['walk'])]]),
            'repairs': int(np.count_nonzero(scene == REPAIR)), 'gaps': len(gaps),
            'marked_pivot': any(scene[e['pivot']] == REPAIR for e in edits),
            'macro_cells': len(full - windows),
            'ray_retained_meetings': sum(len(e['ray'] & (full - old)) for e in edits)}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4 and len(puzzle['train']) > len(puzzle['test'])
    seen, key_signature, measurements = set(), None, {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            _, start, book, scene, layers, records, windows, gaps, edits = _analyse(inp)
            signature = tuple((c, _key(m)) for c, m in sorted(book.items()))
            if key_signature is None:
                key_signature = signature
            assert signature == key_signature, 'key meanings changed within puzzle'
            assert np.array_equal(inp[:start], out[:start]), 'key must be preserved'
            assert np.array_equal(solve(inp), out), 'input-only prediction disagrees'
            assert not np.array_equal(inp, out)
            assert len(layers) == 1 and len(edits) == 1, 'generated scene is one circuit with one edit'
            cells = set().union(*layers.values())
            assert len(_components(cells)) == 1
            assert sum(len(_neighbors(p, cells)) for p in cells) == 2 * (len(cells) - 1)
            assert Counter(len(_neighbors(p, cells)) for p in cells)[3] == 1
            assert np.count_nonzero(scene == 0) > scene.size // 2
            assert set(map(int, np.unique(out[start:]))) <= {0} | set(layers)
            m = measure_pair(pair)
            assert m['partial_modules'] == 1, 'rerouting must cut through a motif'
            assert m['repairs'] >= 1
            measurements[split].append(m)
    train = measurements['train']
    assert any(m['retained_wires'] >= 1 for m in train), 'teach full wire expansion'
    assert any(m['deleted_wires'] == 1 and m['marked_pivot'] for m in train), 'teach deletion through a symbolic wire'
    assert all(m['deleted_wires'] <= 1 for m in train)
    assert all(m['deleted_wires'] == 2 and m['marked_pivot'] for m in measurements['test'])


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(_key(x[r]) == _key(y[r])
               for x, y in zip(a[s], b[s]) for r in ('input', 'output')) for s in ('train', 'test'))


def self_test() -> dict:
    """Bounded 42-seed variation, input-only ablations and negative labels."""
    metrics, signatures, dictionaries, structures = [], set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        book = _read_key(puzzle['train'][0]['input'])[1]
        dictionaries.add(tuple((c, _key(m)) for c, m in sorted(book.items())))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                m = measure_pair(pair)
                metrics.append(m)
                roles = {}
                structures.add(tuple(roles.setdefault(int(v), len(roles)) for v in pair['input'].flat))
                _, start, _, scene, layers, records, windows, _, edits = _analyse(pair['input'])
                # Omitting B leaves complete templates and the old branch.
                no_b = _render(scene.shape, layers, [])
                assert not np.array_equal(no_b, pair['output'][start:])
                # Omitting A loses retained glyph pixels, not just erased ones.
                glyph_cells = set().union(*(r['cells'] for r in records))
                no_a = {c: cells - glyph_cells for c, cells in layers.items()}
                ablated = _render(scene.shape, no_a, edits)
                assert not np.array_equal(ablated, pair['output'][start:])
        if seed < 3:
            assert _equal(puzzle, generate_puzzle(seed))
            # Deleting a preserved motif pixel and restoring an erased branch
            # must both be rejected without altering the inputs or training.
            q = puzzle['test'][0]
            _, start, _, _, layers, records, _, _, edits = _analyse(q['input'])
            old, ray = edits[0]['old'], edits[0]['ray']
            retained = set().union(*(r['cells'] for r in records)) - old - ray
            for p in (min(retained), min(old - ray)):
                damaged = copy.deepcopy(puzzle)
                absolute = (p[0] + start, p[1])
                value = int(damaged['test'][0]['output'][absolute])
                damaged['test'][0]['output'][absolute] = 0 if value else edits[0]['ink']
                try:
                    validate_generated_puzzle(damaged)
                except AssertionError:
                    pass
                else:
                    raise AssertionError('corrupt label accepted')
    assert len(signatures) == 42 and len(dictionaries) >= 35
    assert {m['size'] for m in metrics} == {3, 4, 5}
    assert len({tuple(m['shape']) for m in metrics}) >= 30
    assert {m['modules'] for m in metrics} >= {1, 2, 3, 4}
    assert {tuple(d) for m in metrics for d in m['direction']} == set(DIRS)
    assert sum(m['gaps'] > 0 for m in metrics) >= 100
    assert max(m['old_bends'] for m in metrics) >= 5
    assert len(structures) >= 180
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(42 + nt * 5 + nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    return {'seeds': 42, 'pairs': len(metrics), 'dictionaries': len(dictionaries),
            'shapes': len({tuple(m['shape']) for m in metrics}),
            'normalized_inputs': len(structures), 'sizes': sorted({m['size'] for m in metrics}),
            'module_counts': sorted({m['modules'] for m in metrics}),
            'max_old_bends': max(m['old_bends'] for m in metrics),
            'pairs_with_gaps': sum(m['gaps'] > 0 for m in metrics),
            'semantic_verification': 'author-owned checks, no independent solver'}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
