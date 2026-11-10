"""Linked cargo: cable-paired rectangles compact together toward a visible rail.

All semantic information is in the input.  The generator constructs paths and
pairings; solve reconstructs the individual strands from the visible pixels.
No parent code, source examples, files, or random state are used by solve.
"""
from itertools import permutations, product

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _components(cells):
    todo = set(cells)
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        found, stack = {start}, [start]
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


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((0 <= g) & (g <= 9))
    return g.astype(np.int8, copy=True)


def _row_overlap(a, b):
    return max(a['r'], b['r']) < min(a['r'] + a['h'], b['r'] + b['h'])


def _overlap(a, b):
    return (_row_overlap(a, b)
            and max(a['c'], b['c']) < min(a['c'] + a['w'], b['c'] + b['w']))


def _frame_candidates(g, background, rail):
    candidates = []
    for color in sorted(set(map(int, np.unique(g))) - {background, rail}):
        boxes = []
        for comp in _components(map(tuple, np.argwhere(g == color))):
            r0, c0 = min(r for r, c in comp), min(c for r, c in comp)
            r1, c1 = max(r for r, c in comp), max(c for r, c in comp)
            h, w = r1-r0+1, c1-c0+1
            if not (3 <= h <= 6 and 3 <= w <= 6):
                break
            border = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)
                      if r in (r0, r1) or c in (c0, c1)}
            if comp != border:
                break
            patch = g[r0:r1+1, c0:c1+1].copy()
            if np.any(np.isin(patch[1:-1, 1:-1], [background, color, rail])):
                break
            boxes.append(dict(r=r0, c=c0, h=h, w=w, patch=patch))
        else:
            if 4 <= len(boxes) <= 10 and len(boxes) % 2 == 0:
                candidates.append((color, boxes))
    return candidates


def _parse(input_grid):
    raw = _grid(input_grid)
    counts = np.bincount(raw.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] * 2 > raw.size, 'background must occupy over half the input'
    canonical = []
    for k in range(4):
        g = np.rot90(raw, k).copy()
        rail = int(g[0, 0])
        if rail != bg and np.all(g[:, 0] == rail):
            canonical.append((k, g, rail))
    assert len(canonical) == 1, 'one complete boundary rail is required'
    rotation, g, rail = canonical[0]
    candidates = _frame_candidates(g, bg, rail)
    interpretations = []
    for frame, boxes in candidates:
        try:
            interpretations.append(_parse_frames(g, bg, rail, rotation, frame, boxes))
        except (AssertionError, StopIteration):
            continue
    assert len(interpretations) == 1, 'ambiguous frame and cable interpretation'
    return interpretations[0]


def _parse_frames(g, bg, rail, rotation, frame, boxes):
    """Validate a whole frame hypothesis, including its sockets and strands."""
    covered = g == bg
    covered[:, 0] = True
    for o in boxes:
        r, c, h, w = (o[x] for x in ('r', 'c', 'h', 'w'))
        assert c >= 1 and not np.any(covered[r:r+h, c:c+w])
        covered[r:r+h, c:c+w] = True
    wire = set(map(tuple, np.argwhere(~covered)))
    wire_colors = set(int(g[p]) for p in wire)
    assert 1 <= len(wire_colors) <= 2
    assert not wire_colors & {bg, rail, frame}
    for o in boxes:
        assert not set(map(int, np.unique(o['patch'][1:-1, 1:-1]))) & wire_colors
    edges = {p: {} for p in wire}
    for p in wire:
        for d, (dr, dc) in enumerate(DIRS):
            q = p[0]+dr, p[1]+dc
            if q in wire:
                edges[p][d] = q
    ends = {}
    for i, o in enumerate(boxes):
        r, c, h, w = (o[x] for x in ('r', 'c', 'h', 'w'))
        # Socket = left middle column for an even-width frame.
        center = c + (w-1)//2
        ports = [(r-1, center), (r+h, center)]
        contacts = []
        for p in wire:
            if ((p[0] in (r-1, r+h) and c <= p[1] < c+w)
                    or (p[1] in (c-1, c+w) and r <= p[0] < r+h)):
                contacts.append(p)
        assert len(contacts) == 1 and contacts[0] in ports, 'one centered socket per box'
        p = contacts[0]
        assert p not in ends and len(edges[p]) == 1
        ends[p] = i
    assert {p for p in wire if len(edges[p]) == 1} == set(ends)
    crossings = {p for p in wire if len(edges[p]) == 4}
    assert all(len(edges[p]) in (1, 2, 4) for p in wire)
    if crossings:
        cross_colors = {int(g[p]) for p in crossings}
        assert len(cross_colors) == 1 and len(wire_colors) == 2
        cross_color = next(iter(cross_colors))
        assert {p for p in wire if g[p] == cross_color} == crossings
    else:
        assert len(wire_colors) == 1
    assemblies, routes, visited, paired = [], [], set(), set()
    for start in sorted(ends):
        if ends[start] in paired:
            continue
        p, incoming = start, None
        route, states = [], set()
        while True:
            assert (p, incoming) not in states, 'strand cycle'
            states.add((p, incoming))
            route.append(p)
            visited.add(p)
            if p != start and p in ends:
                break
            if incoming is None:
                outgoing = next(iter(edges[p]))
            elif len(edges[p]) == 4:
                outgoing = (incoming+2) % 4
            else:
                outgoing = next(d for d in edges[p] if d != incoming)
            p = edges[p][outgoing]
            incoming = (outgoing+2) % 4
        i, j = ends[start], ends[p]
        assert i != j and i not in paired and j not in paired
        paired.update((i, j))
        assemblies.append((i, j))
        routes.append(route)
    assert len(paired) == len(boxes) and visited == wire
    return dict(grid=g, bg=bg, rail=rail, frame=frame, boxes=boxes,
                assemblies=assemblies, rotation=rotation, routes=routes,
                crossings=crossings, wire=wire)


def _constraints(boxes, assemblies):
    """d[behind] <= d[ahead] + initial gap, plus each rail bound."""
    owner = {member: i for i, group in enumerate(assemblies) for member in group}
    bounds = [min(boxes[j]['c']-1 for j in group) for group in assemblies]
    links = []
    for i, a in enumerate(boxes):
        for j in range(i+1, len(boxes)):
            b = boxes[j]
            if not _row_overlap(a, b) or owner[i] == owner[j]:
                continue
            assert not _overlap(a, b)
            first, second = (i, j) if a['c'] < b['c'] else (j, i)
            gap = boxes[second]['c'] - boxes[first]['c'] - boxes[first]['w']
            assert gap >= 0
            links.append((owner[first], owner[second], gap))
    return bounds, links


def _distances(boxes, assemblies):
    bounds, links = _constraints(boxes, assemblies)
    distances = list(bounds)
    # Multi-source shortest distances from the rail. Nonnegative edge lengths
    # also handle interlocked pairs, including zero-slack contact cycles.
    for _ in range(len(assemblies)):
        changed = False
        for ahead, behind, gap in links:
            candidate = distances[ahead] + gap
            if candidate < distances[behind]:
                distances[behind] = candidate
                changed = True
        if not changed:
            break
    assert all(distances[b] <= distances[a]+gap for a, b, gap in links)
    return distances


def _render(shape, bg, rail, boxes, assemblies, distances):
    output = np.full(shape, bg, dtype=np.int8)
    output[:, 0] = rail
    occupied = np.zeros(shape, dtype=bool)
    occupied[:, 0] = True
    for group, distance in zip(assemblies, distances):
        for member in group:
            o = boxes[member]
            r, c, h, w = o['r'], o['c']-distance, o['h'], o['w']
            assert c >= 1 and c+w <= shape[1]
            assert not np.any(occupied[r:r+h, c:c+w])
            occupied[r:r+h, c:c+w] = True
            output[r:r+h, c:c+w] = o['patch']
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase the wiring and rigidly compact its paired endpoint blocks."""
    info = _parse(input_grid)
    d = _distances(info['boxes'], info['assemblies'])
    output = _render(info['grid'].shape, info['bg'], info['rail'],
                     info['boxes'], info['assemblies'], d)
    return np.rot90(output, -info['rotation']).copy()


def _routing_moves(target, rng, detour=False):
    """An empty exterior track permits arbitrary orthogonal permutations."""
    n = len(target)
    state = list(range(n)) + [None]
    moves = []
    for dest in map(int, rng.permutation(n)):
        if state[dest] == target[dest]:
            continue
        saved = state[dest]
        moves.append((dest, n))
        state[n], state[dest] = saved, None
        hole = dest
        while target[hole] != saved:
            old = state.index(target[hole])
            moves.append((old, hole))
            state[hole], state[old] = state[old], None
            hole = old
        moves.append((n, hole))
        state[hole], state[n] = saved, None
    if detour and len(moves) <= 4:
        lane = int(rng.integers(n))
        moves = [(lane, n), (n, lane)] + moves
    return moves


def _payload(rng, h, w, palette):
    count = int(rng.integers(1, min(4, len(palette))+1))
    colors = np.asarray(rng.choice(palette, count, replace=False), dtype=np.int8)
    rr, cc = np.indices((h-2, w-2))
    mode = int(rng.integers(5))
    index = (np.zeros_like(rr) if mode == 0 else rr if mode == 1 else cc
             if mode == 2 else rr+cc if mode == 3
             else rng.integers(count, size=rr.shape))
    return colors[index % count]


def _make_pair(rng, n, target, style, rotation, palette, compact=False):
    """Ground truth uses generated endpoint ownership, not the input parser."""
    bg, frame, wire, cross, rail = map(int, palette[:5])
    moves = _routing_moves(target, rng, detour=style == 'free' and rng.random() < .25)
    m = len(moves)
    min_field = 2*m+1 if m else 3
    # Widths, gaps, row offsets and cable segment lengths are independently
    # sampled subject only to the visible geometry and the 30-cell limit.
    widths = rng.integers(3, (4 if compact or n == 5 else 5 if n == 4 else 7), size=(2, n))
    heights = rng.integers(3, 4 if compact else 6, size=(2, n))
    if style == 'chain':
        heights[:] = int(rng.integers(3, 5))
    slot_widths = widths.max(axis=0)
    gaps = rng.integers(1, 4, size=n-1)
    first = int(rng.integers(2 if style != 'free' else 1, 6))
    centers, left = [], first
    for i in range(n):
        centers.append(left + (int(slot_widths[i])-1)//2)
        left += int(slot_widths[i]) + (int(gaps[i]) if i < n-1 else 0)
    spare = left + int(rng.integers(1, 4))
    width = spare + int(rng.integers(2, 5))
    if width > 30:
        raise ValueError('width budget')
    if style == 'chain':
        step = int(rng.integers((int(heights[0, 0])+1)//2, int(heights[0, 0])))
        offsets = np.tile(np.arange(n)*step, (2, 1))
        if rng.random() < .5:
            offsets = offsets[:, ::-1].copy()
    elif style == 'interlock':
        offsets = np.zeros((2, n), dtype=int)
    else:
        budget = 30 - min_field - 2 - int(heights[0].max()) - int(heights[1].max())
        if budget < 0:
            raise ValueError('height budget')
        a = int(rng.integers(min(5, budget)+1))
        b = int(rng.integers(min(5, budget-a)+1))
        offsets = np.stack([rng.integers(a+1, size=n), rng.integers(b+1, size=n)])
        # Also sample exact alignments and independently separated row bands.
        for bank in range(2):
            if rng.random() < .3:
                offsets[bank] = 0
    spans = (heights+offsets).max(axis=1)
    extra_budget = 30 - int(spans.sum()) - min_field - 2
    if extra_budget < 0:
        raise ValueError('height budget')
    extra = int(rng.integers(min(5, extra_budget)+1))
    lead, tail = 1, 1
    separations = [2]*max(0, m-1)
    for _ in range(extra):
        item = int(rng.integers(2+len(separations)))
        if item == 0:
            lead += 1
        elif item == 1:
            tail += 1
        else:
            separations[item-2] += 1
    start = 1 + int(spans[0])
    move_rows = []
    row = start + lead
    for i in range(m):
        move_rows.append(row)
        if i < m-1:
            row += separations[i]
    end = (move_rows[-1]+tail) if m else start+lead+tail
    height = end + int(spans[1]) + 2
    assert height <= 30
    boxes = []
    for bank in range(2):
        for i in range(n):
            h, w = int(heights[bank, i]), int(widths[bank, i])
            r = 1 + int(offsets[bank, i]) if bank == 0 else end+1+int(offsets[bank, i])
            c = centers[i]-(w-1)//2
            patch = np.full((h, w), frame, dtype=np.int8)
            patch[1:-1, 1:-1] = _payload(rng, h, w, palette[5:])
            boxes.append(dict(r=r, c=c, h=h, w=w, patch=patch))
    # paths[i] belongs to top box i. target[j] is the source at bottom slot j.
    paths = []
    for i, c in enumerate(centers):
        first_wire = boxes[i]['r']+boxes[i]['h']
        paths.append([(r, c) for r in range(first_wire, start+1)])
    state, previous = list(range(n))+[None], start
    lanes = centers+[spare]
    for row, (old, new) in zip(move_rows, moves):
        for lane, owner in enumerate(state):
            if owner is not None:
                paths[owner].extend((r, lanes[lane]) for r in range(previous+1, row+1))
        owner = state[old]
        step = 1 if lanes[new] > lanes[old] else -1
        paths[owner].extend((row, c) for c in range(lanes[old]+step, lanes[new]+step, step))
        state[new], state[old] = owner, None
        previous = row
    assert state[:n] == list(target)
    groups = []
    for lane in range(n):
        owner = state[lane]
        stop = boxes[n+lane]['r']-1
        paths[owner].extend((r, centers[lane]) for r in range(previous+1, stop+1))
        groups.append((owner, n+lane))
    input_grid = _render((height, width), bg, rail, boxes, groups, [0]*n)
    visits = {}
    for i, path in enumerate(paths):
        for p in path:
            assert input_grid[p] in (bg, wire), 'cable enters an object or rail'
            visits.setdefault(p, set()).add(i)
            input_grid[p] = wire
    for p, owners in visits.items():
        if len(owners) > 1:
            assert len(owners) == 2
            input_grid[p] = cross
    if np.count_nonzero(input_grid == bg)*2 <= input_grid.size:
        raise ValueError('background budget')
    output = _render(input_grid.shape, bg, rail, boxes, groups, _distances(boxes, groups))
    return {'input': np.rot90(input_grid, rotation).copy(),
            'output': np.rot90(output, rotation).copy()}


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, 'num_train')
    nq = _count(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    rotation = int(rng.integers(4))
    palette = list(map(int, rng.permutation(10)))
    vary_palette = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, total in (('train', nt), ('test', nq)):
        for index in range(total):
            for attempt in range(160):
                if split == 'train' and index == 0:
                    n, target, style = 2, [1, 0], 'interlock'
                elif split == 'train' and index == 1:
                    n, target, style = 3, [0, 1, 2], 'chain'
                elif split == 'train':
                    n, style = int(rng.integers(2, 4)), 'free'
                    target = list(range(n))
                    if rng.random() < .7:
                        a, b = map(int, rng.choice(n, 2, replace=False))
                        target[a], target[b] = target[b], target[a]
                else:
                    n, style = int(rng.integers(3, 6)), 'free'
                    # A new cycle of length 3..n, with any remaining slots fixed.
                    cycle = list(map(int, rng.choice(n, int(rng.integers(3, n+1)), replace=False)))
                    target = list(range(n))
                    for a, b in zip(cycle, cycle[1:]+cycle[:1]):
                        target[a] = b
                local_palette = list(map(int, rng.permutation(10))) if vary_palette else palette
                try:
                    pair = _make_pair(rng, n, target, style, rotation, local_palette, compact=attempt >= 80)
                    info = _parse(pair['input'])
                    # These witnesses are measured from input, not assumed from
                    # a requested style or construction permutation.
                    evidence = _measure_info(info)
                    if not any(evidence['travels']):
                        continue
                    if index == 0 and split == 'train' and not (
                            evidence['link_witness'] and evidence['route_witness']
                            and evidence['cycle'] and evidence['different_travel']
                            and evidence['hollow_witness']):
                        continue
                    if index == 1 and split == 'train' and not (evidence['overtaking_witness'] and evidence['different_travel']):
                        continue
                    if split == 'test' and not evidence['link_witness']:
                        continue
                    key = pair['input'].shape, pair['input'].tobytes()
                    if key in seen:
                        continue
                except (ValueError, AssertionError, StopIteration):
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError('bounded cargo construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _edge_first(boxes, groups):
    """A deliberately wrong repacker: permits jumping ahead of predecessors."""
    distances, settled = [0]*len(groups), []
    order = sorted(range(len(groups)), key=lambda i: min(boxes[j]['c'] for j in groups[i]))
    for i in order:
        limit = min(boxes[j]['c']-1 for j in groups[i])
        for d in range(limit, -1, -1):
            shifted = [dict(boxes[j], c=boxes[j]['c']-d) for j in groups[i]]
            if not any(_overlap(a, b) for a in shifted for b in settled):
                distances[i] = d
                settled.extend(shifted)
                break
    return distances


def _measure_info(info):
    boxes, groups = info['boxes'], info['assemblies']
    bounds, links = _constraints(boxes, groups)
    d = _distances(boxes, groups)
    individual = _distances(boxes, [(i,) for i in range(len(boxes))])
    owner = {j: i for i, group in enumerate(groups) for j in group}
    n = len(groups)
    by_row = sorted(range(len(boxes)), key=lambda i: boxes[i]['r'])
    banks = [sorted(bank, key=lambda i: boxes[i]['c']+(boxes[i]['w']-1)//2)
             for bank in (by_row[:n], by_row[n:])]
    aligned = list(zip(*banks))
    aligned_d = _distances(boxes, aligned)
    aligned_owner = {j: i for i, group in enumerate(aligned) for j in group}
    final_bounds = []
    for i, group in enumerate(groups):
        members = [boxes[j] for j in group]
        r = min(o['r'] for o in members)
        c = min(o['c'] for o in members)-d[i]
        final_bounds.append(dict(r=r, c=c,
                                 h=max(o['r']+o['h'] for o in members)-r,
                                 w=max(o['c']+o['w'] for o in members)-d[i]-c))
    adjacency = {(a, b) for a, b, gap in links}
    reach = set(adjacency)
    for _ in groups:
        reach |= {(a, d1) for a, b in list(reach) for c, d1 in list(reach) if b == c}
    return dict(links=len(groups), shape=info['grid'].shape, frame=info['frame'], rotation=info['rotation'],
                roles=(info['bg'], info['frame'], info['rail']),
                footprints=tuple(sorted({(o['h'], o['w']) for o in boxes})),
                crossings=len(info['crossings']), lengths=tuple(map(len, info['routes'])),
                edge_count=len(links), cycle=any(a == b for a, b in reach),
                travels=tuple(d), stationary=d.count(0), different_travel=len(set(d)) > 1,
                link_witness=any(individual[j] != d[owner[j]] for j in range(len(boxes))),
                route_witness=any(aligned_d[aligned_owner[j]] != d[owner[j]] for j in range(len(boxes))),
                packing_witness=any(a < b for a, b in zip(d, bounds)),
                hollow_witness=any(_overlap(a, b) for i, a in enumerate(final_bounds)
                                   for b in final_bounds[i+1:]),
                overtaking_witness=_edge_first(boxes, groups) != d,
                payload_colors=len({int(c) for o in boxes for c in np.unique(o['patch'][1:-1, 1:-1])}),
                patterned=any(len(np.unique(o['patch'][1:-1, 1:-1])) > 1 for o in boxes))


def _check_packing_certificate(info, output):
    """Check preservation, feasibility, order, and maximality without solving.

    A maximal displacement has a tight dependency path to a rail-bound member.
    This also certifies a linked convoy containing cyclic order dependencies.
    """
    boxes, groups = info['boxes'], info['assemblies']
    actual = np.rot90(output, info['rotation'])
    # Frame top-left corners uniquely identify translated patches at fixed rows;
    # try displacement 0..rail bound, then select the full-scene exact cover.
    candidates = []
    for group in groups:
        options = []
        for d in range(min(boxes[j]['c']-1 for j in group)+1):
            if all(np.array_equal(actual[boxes[j]['r']:boxes[j]['r']+boxes[j]['h'],
                                         boxes[j]['c']-d:boxes[j]['c']-d+boxes[j]['w']],
                                  boxes[j]['patch']) for j in group):
                options.append(d)
        assert options, 'linked patches were changed or traveled different distances'
        candidates.append(options)
    bounds, links = _constraints(boxes, groups)
    certificates = []
    for ds in product(*candidates):
        if any(ds[b] > ds[a]+gap for a, b, gap in links):
            continue
        tight_to_rail = {i for i, bound in enumerate(bounds) if ds[i] == bound}
        for _ in groups:
            tight_to_rail |= {b for a, b, gap in links
                              if a in tight_to_rail and ds[b] == ds[a]+gap}
        if len(tight_to_rail) != len(groups):
            continue
        try:
            rebuilt = _render(actual.shape, info['bg'], info['rail'], boxes, groups, ds)
        except AssertionError:
            continue
        if np.array_equal(rebuilt, actual):
            certificates.append(ds)
    assert len(certificates) == 1, 'packing is not the uniquely certified maximal translation'
    return certificates[0]


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            g, out = _grid(pair['input']), _grid(pair['output'])
            assert g.shape == out.shape
            key = g.shape, g.tobytes()
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info = _parse(g)
            ds = _check_packing_certificate(info, out)
            assert any(ds), 'packing must visibly move cargo'
            metrics = _measure_info(info)
            if split == 'train':
                training.append(metrics)
            assert np.array_equal(solve(g), out)
    assert any(m['link_witness'] and m['route_witness'] and m['cycle']
               and m['different_travel'] and m['hollow_witness'] for m in training)
    assert any(m['overtaking_witness'] and m['different_travel'] for m in training)
    assert any(m['packing_witness'] for m in training)
    assert any(m['crossings'] for m in training)


def _fingerprint(puzzle):
    return tuple((s, np.asarray(p['input']).shape, np.asarray(p['input']).tobytes(),
                  np.asarray(p['output']).tobytes())
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    corpus = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in corpus}) == 42
    measurements = [_measure_info(_parse(pair['input'])) for puzzle in corpus
                    for split in ('train', 'test') for pair in puzzle[split]]
    assert {m['links'] for m in measurements} == {2, 3, 4, 5}
    assert len({m['shape'] for m in measurements}) >= 35
    assert len({m['roles'] for m in measurements}) >= 30
    assert len({m['footprints'] for m in measurements}) >= 35
    assert {bool(m['crossings']) for m in measurements} == {False, True}
    assert any(m['patterned'] for m in measurements)
    assert any(m['cycle'] for m in measurements)
    assert any(m['different_travel'] for m in measurements)
    assert {m['rotation'] for m in measurements} == {0, 1, 2, 3}
    assert _fingerprint(corpus[0]) == _fingerprint(generate_puzzle(0))
    # Count boundaries and every explicit legal count combination.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(20261+nt*7+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Exhaust every four-cable permutation with independently parsed topology.
    rng = np.random.default_rng(471)
    for target in permutations(range(4)):
        for attempt in range(100):
            try:
                pair = _make_pair(rng, 4, target, 'free', 0, list(range(10)), compact=attempt > 50)
                info = _parse(pair['input'])
                break
            except (AssertionError, ValueError):
                continue
        else:
            raise AssertionError('could not exercise permutation')
        boxes = info['boxes']
        top = sorted(range(4), key=lambda i: boxes[i]['c'])
        bottom = sorted(range(4, 8), key=lambda i: boxes[i]['c'])
        paired = {a: b for a, b in info['assemblies']}
        assert [top.index(next(a for a, b in paired.items() if b == j)) for j in bottom] == list(target)
        assert np.array_equal(solve(pair['input']), pair['output'])
    # An exhaustive tiny constraint check establishes the componentwise maximum.
    boxes = [dict(r=r, c=c, h=3, w=3) for r, c in ((0, 3), (2, 8), (4, 13),
                                                                 (10, 3), (12, 8), (14, 13))]
    groups = [(0, 3), (1, 4), (2, 5)]
    expected = _distances(boxes, groups)
    assert expected == [2, 4, 6]
    bounds, links = _constraints(boxes, groups)
    feasible = [ds for ds in product(*(range(b+1) for b in bounds))
                if all(ds[j] <= ds[i]+gap for i, j, gap in links)]
    assert all(all(ds[i] <= expected[i] for i in range(3)) for ds in feasible)
    assert _edge_first(boxes, groups) != expected
    # A carried pattern may itself resemble a small frame. Socket continuity
    # must identify the true outer frame rather than banning such payloads.
    nested_boxes = []
    for r, c in ((1, 2), (1, 10), (13, 2), (13, 10)):
        patch = np.ones((5, 5), dtype=np.int8)
        patch[1:-1, 1:-1] = 3
        patch[2, 2] = 5
        nested_boxes.append(dict(r=r, c=c, h=5, w=5, patch=patch))
    nested_groups = [(0, 2), (1, 3)]
    nested = _render((19, 20), 0, 4, nested_boxes, nested_groups, [0, 0])
    nested[6:13, [4, 12]] = 2
    assert len(_frame_candidates(nested, 0, 4)) == 2
    nested_answer = _render(nested.shape, 0, 4, nested_boxes, nested_groups,
                            _distances(nested_boxes, nested_groups))
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(nested, rotation)),
                              np.rot90(nested_answer, rotation))
    # Structured negative examples exercise each inherited mechanism, beyond
    # comparing the generator and solver on their own positive outputs.
    def must_reject_output(index, canonical_output):
        changed = {s: [{k: v.copy() for k, v in pair.items()} for pair in corpus[0][s]]
                   for s in corpus[0]}
        info = _parse(changed['train'][index]['input'])
        changed['train'][index]['output'] = np.rot90(canonical_output, -info['rotation']).copy()
        try:
            validate_generated_puzzle(changed)
        except AssertionError:
            return
        raise AssertionError('a mechanism-omitting answer was accepted')

    linked = _parse(corpus[0]['train'][0]['input'])
    singleton = [(i,) for i in range(len(linked['boxes']))]
    uncoupled = _render(linked['grid'].shape, linked['bg'], linked['rail'], linked['boxes'],
                        singleton, _distances(linked['boxes'], singleton))
    must_reject_output(0, uncoupled)
    merged = [tuple(range(len(linked['boxes'])))]
    flood = _render(linked['grid'].shape, linked['bg'], linked['rail'], linked['boxes'],
                    merged, _distances(linked['boxes'], merged))
    must_reject_output(0, flood)
    chain = _parse(corpus[0]['train'][1]['input'])
    jumped = _render(chain['grid'].shape, chain['bg'], chain['rail'], chain['boxes'],
                      chain['assemblies'], _edge_first(chain['boxes'], chain['assemblies']))
    must_reject_output(1, jumped)
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in corpus[0][s]] for s in corpus[0]}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('output corruption accepted')
    return {'seeds': 42, 'pairs': len(measurements),
            'shapes': len({m['shape'] for m in measurements}),
            'footprint_sets': len({m['footprints'] for m in measurements}),
            'role_assignments': len({m['roles'] for m in measurements}),
            'link_counts': sorted({m['links'] for m in measurements})}


if __name__ == '__main__':
    print(self_test())
