"""Cable-stitched silhouettes: complementary shape halves travel along cables.

Only NumPy and the standard library are used. No evidence files are loaded.
Generation knows its paths; solve reconstructs a directional graph from cells.
Experiments are in-memory and no execution traces or puzzle corpora are saved.
"""
import hashlib
import json
from itertools import permutations

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
AUDIT_SEEDS = (0, 976119149, 440108155, 988281488, 806353288, 490257438,
               912163976, 891555645, 909013394, 629596561, 329851643,
               862952272, 895151888, 823494526, 998154815, 603078614,
               894328426, 227109604, 953111050, 22656029, 759814709,
               512012473, 535052371, 695750323, 750944961, 581623763,
               67836059, 950485516, 618459421, 317521209, 705154992,
               730194039, 533572448, 621775794, 535897130, 836915260,
               727937462, 620978668, 876179344, 731668257, 614445854, 734956877)


def _components(cells):
    todo, result = set(cells), []
    while todo:
        root = min(todo)
        todo.remove(root)
        found, stack = {root}, [root]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = r + dr, c + dc
                if q in todo:
                    todo.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _normal(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _grid_key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def _array(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((0 <= g) & (g <= 9))
    return g.astype(np.int8, copy=False)


def _rectangles(g, color):
    result = []
    for cells in _components(map(tuple, np.argwhere(g == color))):
        r0 = min(r for r, c in cells)
        r1 = max(r for r, c in cells)
        c0 = min(c for r, c in cells)
        c1 = max(c for r, c in cells)
        assert r1 > r0 and c1 > c0
        assert len(cells) == (r1-r0+1)*(c1-c0+1)
        result.append((r0, r1, c0, c1))
    return result


def _mask_links(rect, visible, endpoints):
    r0, r1, c0, c1 = rect
    perimeter = ([((r0-1, c), 2) for c in range(c0, c1+1)]
                 + [((r1+1, c), 0) for c in range(c0, c1+1)]
                 + [((r, c0-1), 1) for r in range(r0, r1+1)]
                 + [((r, c1+1), 3) for r in range(r0, r1+1)])
    ports = []
    for p, d in perimeter:
        degree = sum((p[0]+dr, p[1]+dc) in visible for dr, dc in DIRS)
        if p in visible and (degree % 2 == 1 or endpoints.get(p) == d):
            ports.append((p, d))
    assert len(ports) >= 2 and len(ports) % 2 == 0
    if len(ports) == 2:
        return [(ports[0], ports[1])]
    todo, links = set(ports), []
    while todo:
        a = min(todo)
        todo.remove(a)
        p, d = a
        matches = [b for b in todo if b[1] == (d+2) % 4
                   and p[1 if d % 2 == 0 else 0] == b[0][1 if d % 2 == 0 else 0]]
        assert len(matches) == 1, 'opaque mask has an ambiguous pairing'
        b = matches[0]
        todo.remove(b)
        links.append((a, b))
    return links


def _trace(g, centers, start, end, wire, cross, mask):
    visible = set(map(tuple, np.argwhere(g == wire)))
    if cross is not None:
        visible.update(map(tuple, np.argwhere(g == cross)))
    endpoints = {(start, c): 2 for c in centers}
    endpoints.update({(end, c): 0 for c in centers})
    graph = {p: {} for p in visible}
    for r, c in visible:
        for d, (dr, dc) in enumerate(DIRS):
            q = r+dr, c+dc
            if q in visible:
                graph[r, c][d] = (q, (d+2) % 4)
    masks = []
    for rect in ([] if mask is None else _rectangles(g, mask)):
        links = _mask_links(rect, visible, endpoints)
        for (p, dp), (q, dq) in links:
            assert dp not in graph[p] and dq not in graph[q]
            graph[p][dp], graph[q][dq] = (q, dq), (p, dp)
        masks.append((rect, links))
    crossings = []
    for p, edges in graph.items():
        if p in endpoints:
            assert len(edges) == 1 and endpoints[p] in edges
        elif len(edges) == 4:
            assert cross is not None
            assert any(all((p[0]+DIRS[d][0], p[1]+DIRS[d][1]) in visible
                           and g[p[0]+DIRS[d][0], p[1]+DIRS[d][1]] == cross
                           for d in axis) for axis in ((0, 2), (1, 3)))
            crossings.append(p)
        else:
            assert len(edges) == 2, 'a strand branches or ends inside the field'
    pairing, routes, seen = [], [], set()
    for c in centers:
        p, incoming, states, route = (start, c), 0, set(), []
        while True:
            assert (p, incoming) not in states
            states.add((p, incoming))
            seen.add(p)
            route.append(p)
            if p in endpoints and p[0] == end:
                pairing.append(centers.index(p[1]))
                break
            edges = graph[p]
            if p == (start, c):
                d = 2
            else:
                assert incoming in edges
                d = ((incoming+2) % 4 if len(edges) == 4
                     else next(k for k in edges if k != incoming))
            p, incoming = edges[d]
            assert p[0] != start, 'a route returns to its source bank'
        routes.append(route)
    assert sorted(pairing) == list(range(len(centers)))
    assert seen == visible, 'unattached wire fragment'
    return pairing, routes, masks, crossings


def _banks(g, centers, start, end, payload):
    banks = [[set() for _ in centers] for _ in range(2)]
    for rr, cc in np.argwhere(g == payload):
        r, c = int(rr), int(cc)
        assert 1 <= r <= start-2 or end+2 <= r <= g.shape[0]-2
        distance = [abs(c-v) for v in centers]
        assert distance.count(min(distance)) == 1, 'equidistant payload'
        banks[int(r > end)][distance.index(min(distance))].add((r, c))
    for bank in banks:
        for c, cells in zip(centers, bank):
            assert cells and len(_components(cells)) == 1
            assert any(q[1] < c for q in cells)
            assert any(q[1] > c for q in cells) and any(q[1] == c for q in cells)
    return banks


def _parse(input_grid):
    original = _array(input_grid)
    bg = int(original[0, 0])
    assert all(original[r, c] == bg for r, c in ((0, -1), (-1, 0), (-1, -1)))
    vertical = (np.any(original[0] != bg) and np.any(original[-1] != bg)
                and np.all(original[:, 0] == bg) and np.all(original[:, -1] == bg))
    horizontal = (np.any(original[:, 0] != bg) and np.any(original[:, -1] != bg)
                  and np.all(original[0] == bg) and np.all(original[-1] == bg))
    assert vertical != horizontal
    g = original.T if horizontal else original
    h, w = g.shape
    centers = list(map(int, np.flatnonzero(g[0] != bg)))
    assert 2 <= len(centers) <= 4 and centers == list(np.flatnonzero(g[-1] != bg))
    assert min(np.diff(centers)) >= 4
    marker_colors = set(map(int, g[0, centers])) | set(map(int, g[-1, centers]))
    assert len(marker_colors) == 1
    guide = marker_colors.pop()
    assert np.count_nonzero(g == guide) == 2*len(centers)
    candidates = []
    for wire in sorted(set(map(int, np.unique(g))) - {bg, guide}):
        pos = np.argwhere(g == wire)
        start, end = int(pos[:, 0].min()), int(pos[:, 0].max())
        if start < 5 or end > h-6 or end-start < 4:
            continue
        if list(np.flatnonzero(g[start] == wire)) != centers:
            continue
        if list(np.flatnonzero(g[end] == wire)) != centers:
            continue
        try:
            outside = np.concatenate((g[:start].ravel(), g[end+1:].ravel()))
            extra = set(map(int, np.unique(outside))) - {bg, guide}
            assert len(extra) == 1
            payload = extra.pop()
            assert payload != wire and not np.any(g[start:end+1] == payload)
            field = g[start:end+1]
            others = sorted(set(map(int, np.unique(field))) - {bg, wire})
            assert len(others) <= 2
            cross = mask = None
            for color in others:
                try:
                    _rectangles(g, color)
                except AssertionError:
                    assert cross is None
                    cross = color
                else:
                    assert mask is None
                    mask = color
            banks = _banks(g, centers, start, end, payload)
            pairing, routes, masks, crossings = _trace(g, centers, start, end, wire, cross, mask)
            candidates.append(dict(grid=g, transpose=bool(horizontal), background=bg,
                                   guide=guide, payload=payload, wire=wire, cross=cross,
                                   mask=mask, centers=centers, start=start, end=end,
                                   banks=banks, pairing=pairing, routes=routes,
                                   masks=masks, crossings=crossings))
        except (AssertionError, StopIteration, KeyError):
            continue
    assert len(candidates) == 1, f'expected one interpretation, found {len(candidates)}'
    return candidates[0]


def _half(cells, c, side):
    return {p for p in cells if (p[1] < c if side == 0 else p[1] > c)}


def _place(out, cells, center, bank, side, color, separate=False):
    groups = _components(cells) if separate else [cells]
    for group in groups:
        rs, cs = zip(*group)
        dr = -min(rs) if bank == 0 else out.shape[0]-1-max(rs)
        dc = center-1-max(cs) if side == 0 else center+1-min(cs)
        for r, c in group:
            assert 0 <= r+dr < out.shape[0] and 0 <= c+dc < out.shape[1]
            out[r+dr, c+dc] = color


def _assemble(info, *, ignore_routing=False, separate=False, nearest=False, keep_whole=False):
    g, centers = info['grid'], info['centers']
    bg, payload, guide = (info[k] for k in ('background', 'payload', 'guide'))
    banks = info['banks']
    out = g.copy()
    out[out == payload] = bg
    for bank, stamps in enumerate(banks):
        for center, cells in zip(centers, stamps):
            axial = sorted(r for r, c in cells if c == center)
            for r in axial:
                out[r, center] = payload
            stop = (min(axial) if bank == 0 else max(axial)) if nearest else (
                max(axial) if bank == 0 else min(axial))
            rows = range(stop+1) if bank == 0 else range(stop, g.shape[0])
            for r in rows:
                if out[r, center] == bg:
                    out[r, center] = guide
    mapping = list(range(len(centers))) if ignore_routing else info['pairing']
    for i, j in enumerate(mapping):
        left = _half(banks[0][i], centers[i], 0)
        right = _half(banks[1][j], centers[j], 1)
        for bank, slot in ((0, i), (1, j)):
            for side, cells in enumerate((left, right)):
                if keep_whole:
                    cells = _half(banks[bank][slot], centers[slot], side)
                _place(out, cells, centers[slot], bank, side, payload, separate)
    return out.T.copy() if info['transpose'] else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read markers, cut stamps, trace cables, and register complementary halves."""
    return _assemble(_parse(input_grid))


def _cycle_moves(target, rng):
    # A spare channel allows any permutation without an adjacent-swap scaffold.
    n = len(target)
    state, moves = list(range(n)) + [None], []
    for j in map(int, rng.permutation(n)):
        if state[j] == target[j]:
            continue
        saved = state[j]
        moves.append((j, n))
        state[n], state[j] = state[j], None
        hole = j
        while target[hole] != saved:
            k = state.index(target[hole])
            moves.append((k, hole))
            state[hole], state[k] = state[k], None
            hole = k
        moves.append((n, hole))
        state[hole], state[n] = state[n], None
    assert state[:n] == target
    return moves


def _routing(rng, n, target, wanted, detour):
    moves = _cycle_moves(target, rng)
    if detour and len(moves) <= 5:
        k = int(rng.integers(n))
        moves = [(k, n), (n, k)] + moves
    lead = 4 if wanted else 2
    wire_len = lead + 3 + 2*max(0, len(moves)-1)
    # Identity has no moves; a long straight bundle remains meaningful.
    if not moves:
        wire_len = int(rng.integers(7, 13))
    assert wire_len <= 20
    budget = 30-wire_len-10
    d0 = 5+int(rng.integers(min(4, budget)+1))
    d1 = 5+int(rng.integers(min(4, budget-(d0-5))+1))
    extra = int(rng.integers(min(4, 30-wire_len-d0-d1)+1))
    wire_len += extra
    qmax = {2: 4, 3: 3, 4: 2}[n]
    q = int(rng.integers(1, qmax+1))
    pitch = 2*q+2
    centers = [q+1+i*pitch for i in range(n)]
    spare = centers[-1]+int(rng.integers(3, 6))
    right_margin = max(2, q+2-(spare-centers[-1]))
    base_width = spare+right_margin
    assert base_width <= 30
    left = int(rng.integers(min(3, 30-base_width)+1))
    centers = [c+left for c in centers]
    spare += left
    width = spare+right_margin+int(rng.integers(min(3, 30-spare-right_margin)+1))
    # If room permits, make terminal gaps unequal while keeping all half-cells clear.
    if n > 2 and width < 30 and rng.random() < .5:
        k = int(rng.integers(1, n))
        centers[k:] = [c+1 for c in centers[k:]]
        spare += 1
        width += 1
    columns = centers+[spare]
    start, end = d0, d0+wire_len-1
    h = end+d1+1
    paths = [[(start, c)] for c in centers]
    state, previous = list(range(n))+[None], start
    for m, (a, b) in enumerate(moves):
        r = start+lead+2*m
        for lane, owner in enumerate(state):
            if owner is not None:
                paths[owner].extend((rr, columns[lane]) for rr in range(previous+1, r+1))
        owner = state[a]
        step = 1 if columns[b] > columns[a] else -1
        paths[owner].extend((r, cc) for cc in range(columns[a]+step, columns[b]+step, step))
        state[b], state[a] = state[a], None
        previous = r
    for lane, owner in enumerate(state):
        if owner is not None:
            paths[owner].extend((rr, columns[lane]) for rr in range(previous+1, end+1))
    field, visits = np.zeros((h, width), dtype=np.int8), {}
    for i, path in enumerate(paths):
        for p in path:
            field[p] = 1
            visits.setdefault(p, set()).add(i)
    for p, owners in visits.items():
        if len(owners) == 2:
            axis = (0, 2) if rng.random() < .5 else (1, 3)
            for d in axis:
                dr, dc = DIRS[d]
                field[p[0]+dr, p[1]+dc] = 2
        else:
            assert len(owners) == 1
    return field, centers, start, end, paths, q, min(d0, d1)-2


def _occlude(rng, field, centers, start, end, paths, wanted, keep_crossing):
    original = field.copy()
    points = sorted(set(p for path in paths for p in path))
    used = []
    proposals = []
    # A requested occlusion shift must have at least two actual masks. These
    # separated lead rectangles are guaranteed by the four-cell straight lead;
    # a third mask can still cover a bend or several strands farther inward.
    if wanted >= 2:
        proposals.extend((start+1, start+2, c-1, c) for c in (centers[0], centers[-1]))
    for _ in range(70):
        r, c = points[int(rng.integers(len(points)))]
        hh = int(rng.integers(2, min(5, end-start)+1))
        ww = int(rng.integers(2, min(12, field.shape[1]-2)+1))
        r0 = max(start+1, min(end-hh, r-int(rng.integers(hh))))
        c0 = max(1, min(field.shape[1]-ww-1, c-int(rng.integers(ww))))
        proposals.append((r0, r0+hh-1, c0, c0+ww-1))
    # Four-cell straight leads make these fallback masks input-unambiguous.
    for c in centers:
        proposals.append((start+1, start+2, c-1, c))
    expected = [centers.index(path[-1][1]) for path in paths]
    for rect in proposals:
        if len(used) >= wanted:
            break
        r0, r1, c0, c1 = rect
        if r1 >= end or any(not (r1+1 < a or b+1 < r0 or c1+1 < x or y+1 < c0)
                             for a, b, x, y in used):
            continue
        def inside(p):
            return r0 <= p[0] <= r1 and c0 <= p[1] <= c1
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
        visible = set(map(tuple, np.argwhere((original == 1) | (original == 2))))
        visible = {p for p in visible if not inside(p)}
        endpoints = {(start, c): 2 for c in centers} | {(end, c): 0 for c in centers}
        try:
            links = _mask_links(rect, visible, endpoints)
            assert {frozenset((a[0], b[0])) for a, b in links} == actual
            candidate = field.copy()
            candidate[r0:r1+1, c0:c1+1] = 3
            pairing, _, _, crosses = _trace(candidate, centers, start, end, 1, 2, 3)
            assert pairing == expected and (not keep_crossing or crosses)
        except (AssertionError, KeyError, StopIteration):
            continue
        field = candidate
        used.append(rect)
    return field, len(used)


def _shape(rng, q, cap, retained, witness, used):
    for attempt in range(48):
        hp = int(rng.integers(3, cap+1))
        if witness:
            # A winding bridge on the discarded side connects two retained fragments.
            bridge = q+1 if retained == 0 else q-1
            cells = {(0, q-1), (0, q), (0, q+1), (1, bridge),
                     (2, q-1), (2, q), (2, q+1)}
            forbidden = {(1, c) for c in range(2*q+1)
                         if c <= q} if retained == 0 else {
                             (1, c) for c in range(2*q+1) if c >= q}
        else:
            r = int(rng.integers(hp))
            cells = {(r, q-1), (r, q), (r, q+1)}
            forbidden = set()
        count = int(rng.integers(len(cells), max(len(cells)+1, int(hp*(2*q+1)*.85))))
        for _ in range(count-len(cells)):
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in DIRS
                               if 0 <= r+dr < hp and 0 <= c+dc <= 2*q
                               and (r+dr, c+dc) not in cells | forbidden})
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        signature = _normal(_half(cells, q, retained))
        if signature not in used:
            used.add(signature)
            return cells
    # At cap=3 and q=1 these four signatures suffice for all four terminals.
    assert not witness, 'the first stamp must have no used signatures'
    for rows in ({0}, {0, 1}, {0, 2}, {0, 1, 2}):
        cells = {(r, q) for r in range(3)}
        cells |= {(r, q-1 if retained == 0 else q+1) for r in rows}
        cells.add((1, q+1 if retained == 0 else q-1))
        signature = _normal(_half(cells, q, retained))
        if signature not in used:
            used.add(signature)
            return cells
    raise AssertionError('finite shape fallback bound violated')


def _make_pair(rng, palette, n, target, transpose, wanted, lesson, detour=False):
    field, centers, start, end, paths, q, cap = _routing(rng, n, target, wanted, detour)
    if wanted:
        field, masks = _occlude(rng, field, centers, start, end, paths, wanted, lesson)
        assert masks >= 1
    inp = np.asarray(palette, dtype=np.int8)[field]
    bg, wire, cross, mask, payload, guide = palette[:6]
    banks = [[], []]
    for bank in range(2):
        used = set()
        for i, center in enumerate(centers):
            cells = _shape(rng, q, cap, bank, lesson and i == 0, used)
            ph = max(r for r, c in cells)+1
            low, high = (1, start-1-ph) if bank == 0 else (end+2, inp.shape[0]-1-ph)
            r0 = int(rng.integers(low, high+1))
            located = {(r+r0, c+center-q) for r, c in cells}
            for p in located:
                inp[p] = payload
            banks[bank].append(located)
            inp[0 if bank == 0 else -1, center] = guide
    mapping = [centers.index(path[-1][1]) for path in paths]
    known = dict(grid=inp, centers=centers, background=bg, guide=guide,
                 payload=payload, banks=banks, pairing=mapping, transpose=False)
    out = _assemble(known)
    if transpose:
        inp, out = inp.T.copy(), out.T.copy()
    return {'input': inp, 'output': out}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= value <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    nq = _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))  # held-out count, more masks, richer pairing
    orientation = int(rng.integers(3))  # vertical, horizontal, mixed
    vary_palette = bool(rng.integers(2))
    palette = list(map(int, rng.permutation(10)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, total in (('train', nt), ('test', nq)):
        for i in range(total):
            lesson = split == 'train' and i == 0
            if lesson:
                n = 2
            elif split == 'train' and i == 1:
                n = 3
            elif mode == 0:
                n = int(rng.integers(2, 4)) if split == 'train' else 4
            elif mode == 1:
                n = int(rng.integers(2, 5))
            else:
                n = 3 if split == 'train' else int(rng.integers(3, 5))
            if lesson:
                target = [1, 0]
            elif split == 'train' and i == 1:
                target = list(range(n))
            elif mode == 2 and split == 'train':
                target = list(range(n))
                if rng.random() < .7:
                    target[-2], target[-1] = target[-1], target[-2]
            elif mode == 2 and split == 'test':
                target = list(reversed(range(n)))
            elif rng.random() < .25:
                target = list(range(n))
            else:
                target = list(map(int, rng.permutation(n)))
            if lesson:
                wanted = 1
            elif mode == 1:
                wanted = int(rng.integers(2)) if split == 'train' else min(n, int(rng.integers(2, 4)))
            else:
                wanted = int(rng.integers(3))
            trans = orientation == 1 or (orientation == 2 and bool(i % 2))
            if vary_palette:
                palette = list(map(int, rng.permutation(10)))
            for attempt in range(16):
                pair = _make_pair(rng, palette, n, target, trans, wanted, lesson,
                                  detour=not lesson and mode != 2 and wanted == 0 and rng.random() < .25)
                key = _grid_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('bounded distinct-input attempts exhausted')
    return puzzle


def inspect_pair(pair):
    info = _parse(pair['input'])
    fragments, gaps, sizes = [], [], []
    for bank, stamps in enumerate(info['banks']):
        for c, cells in zip(info['centers'], stamps):
            selected = _half(cells, c, bank)
            fragments.append(len(_components(selected)))
            sizes.append(len(selected))
            axial = [r for r, cc in cells if cc == c]
            gaps.append(max(axial)-min(axial)+1-len(axial))
    return dict(shape=info['grid'].shape, orientation=int(info['transpose']),
                cables=len(info['centers']), pairing=tuple(info['pairing']),
                pitch=tuple(map(int, np.diff(info['centers']))),
                crossings=len(info['crossings']), masks=len(info['masks']),
                mask_shapes=[(b-a+1, d-c+1) for (a, b, c, d), _ in info['masks']],
                elbows=sum(len(links) == 1 and links[0][0][1] % 2 != links[0][1][1] % 2
                           for _, links in info['masks']),
                multi_masks=sum(len(links) > 1 for _, links in info['masks']),
                max_fragments=max(fragments), axial_gaps=sum(gaps),
                selected_sizes=tuple(sizes),
                roles=tuple(info[k] for k in ('background', 'wire', 'payload', 'guide')))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, evidence = set(), set()
    shifts = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _array(pair['input']), _array(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            info = _parse(inp)
            shifts[split].append((len(info['centers']), len(info['masks']),
                                  max(abs(i-j) for i, j in enumerate(info['pairing']))))
            canonical = out.T if info['transpose'] else out
            assert np.array_equal(_assemble(info), out)
            a, b = info['start'], info['end']
            assert np.array_equal(canonical[a:b+1], info['grid'][a:b+1])
            expected_count = 0
            # Geometric correspondence and conservation are checked from input pieces.
            for i, j in enumerate(info['pairing']):
                l = _half(info['banks'][0][i], info['centers'][i], 0)
                r = _half(info['banks'][1][j], info['centers'][j], 1)
                expected_count += 2*(len(l)+len(r))
                for bank, slot in ((0, i), (1, j)):
                    center = info['centers'][slot]
                    cells = set(map(tuple, np.argwhere(canonical == info['payload'])))
                    cells = {p for p in cells if (p[0] < a if bank == 0 else p[0] > b)
                             and abs(p[1]-center) == min(abs(p[1]-v) for v in info['centers'])}
                    assert _normal(_half(cells, center, 0)) == _normal(l)
                    assert _normal(_half(cells, center, 1)) == _normal(r)
            axial_count = sum(sum(cc == c for rr, cc in cells)
                              for bank in info['banks'] for c, cells in zip(info['centers'], bank))
            assert np.count_nonzero(out == info['payload']) == expected_count+axial_count
            if split == 'train':
                if not np.array_equal(_assemble(info, ignore_routing=True), out):
                    evidence.add('routing')
                reverse = dict(info, pairing=list(reversed(range(len(info['centers'])))))
                if not np.array_equal(_assemble(reverse), out):
                    evidence.add('nonreversal')
                if not np.array_equal(_assemble(info, separate=True), out):
                    evidence.add('collective_cut')
                if not np.array_equal(_assemble(info, nearest=True), out):
                    evidence.add('farthest_guide')
                if not np.array_equal(_assemble(info, keep_whole=True), out):
                    evidence.add('complementary_graft')
                if info['masks']:
                    evidence.add('mask')
                if info['crossings']:
                    evidence.add('crossing')
    assert {'routing', 'nonreversal', 'collective_cut', 'farthest_guide',
            'complementary_graft', 'mask', 'crossing'} <= evidence
    assert any(min(f[k] for f in shifts['test']) > max(f[k] for f in shifts['train'])
               for k in range(3)), 'no actual count, mask, or routing-span shift'


def self_test():
    """Bounded streaming checks: 42 seeds, every legal count, and semantic mutants."""
    routing_rng = np.random.default_rng(831)
    for target in permutations(range(4)):
        field, centers, start, end, _, _, _ = _routing(routing_rng, 4, list(target), 0, False)
        mapping, _, _, _ = _trace(field, centers, start, end, 1, 2, None)
        assert mapping == [target.index(i) for i in range(4)]
        assert max(field.shape) <= 30
    shapes, structures, palettes, counts, orientations, mask_counts = (set() for _ in range(6))
    signatures, features = set(), []
    for seed in AUDIT_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        h = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                h.update(bytes(inp.shape))
                h.update(inp.tobytes())
                f = inspect_pair(pair)
                features.append(f)
                shapes.add(inp.shape)
                palettes.add(f['roles'])
                counts.add(f['cables'])
                orientations.add(f['orientation'])
                mask_counts.add(f['masks'])
                info = _parse(inp)
                structure = tuple(_normal(_half(stamp, c, bank)) for bank, stamps in enumerate(info['banks'])
                                  for c, stamp in zip(info['centers'], stamps))
                structures.add(structure)
        signatures.add(h.digest())
    assert len(signatures) == 42 and len(shapes) >= 30 and len(structures) >= 180
    assert len(palettes) >= 20 and counts == {2, 3, 4} and orientations == {0, 1}
    assert mask_counts >= {0, 1, 2, 3}
    assert any(f['elbows'] for f in features) and any(f['multi_masks'] for f in features)
    assert any(len(set(f['pitch'])) > 1 for f in features)
    assert any(f['crossings'] == 0 for f in features)
    assert max(f['max_fragments'] for f in features) >= 2
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(0, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    first, second = generate_puzzle(0, 5, 3), generate_puzzle(0, 5, 3)
    assert all(np.array_equal(p[k], q[k]) for s in first
               for p, q in zip(first[s], second[s]) for k in ('input', 'output'))
    # Tampering with a wire cell or transported fragment must be rejected.
    corrupt = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    corrupt['test'][0]['output'][0, 0] = (int(corrupt['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return {'seeds': 42, 'pairs': len(features), 'shapes': len(shapes),
            'half_assemblies': len(structures), 'role_palettes': len(palettes),
            'cable_counts': sorted(counts), 'orientations': sorted(orientations),
            'mask_counts': sorted(mask_counts),
            'elbow_pairs': sum(bool(f['elbows']) for f in features),
            'multi_mask_pairs': sum(bool(f['multi_masks']) for f in features),
            'max_fragments': max(f['max_fragments'] for f in features)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
