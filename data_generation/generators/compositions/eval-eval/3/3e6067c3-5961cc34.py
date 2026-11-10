"""A colored instruction tape programs a network of multiport relays.

The solver reads only the supplied input. Sampling policies carry no semantics.
"""
from __future__ import annotations

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             "grid cells must be integers in 0..9")
    return a.astype(np.int8, copy=True)


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _parse(inp):
    h, w = inp.shape
    bg = int(inp[0, 0])
    _require(all(int(inp[q]) == bg for q in ((0, w-1), (h-1, 0), (h-1, w-1))),
             "corners must be background")
    components = _components(tuple(map(int, q)) for q in np.argwhere(inp != bg))
    boundary = [o for o in components if any(r in (0, h-1) or c in (0, w-1) for r, c in o)]
    _require(len(boundary) == 1, "one boundary instruction tape required")
    launcher = boundary[0]
    ends = [q for q in launcher if q[0] in (0, h-1) or q[1] in (0, w-1)]
    _require(len(ends) == 1, "tape touches one boundary cell")
    start = ends[0]
    direction = 2 if start[0] == 0 else 0 if start[0] == h-1 else 1 if start[1] == 0 else 3
    dr, dc = _DIRS[direction]
    line = [(start[0]+k*dr, start[1]+k*dc) for k in range(len(launcher))]
    _require(set(line) == launcher and len(line) >= 3, "tape must be a straight unit-width strip")
    tape = [int(inp[q]) for q in line[:-1]]
    cap = line[-1]
    cap_color = int(inp[cap])
    _require(cap_color not in tape, "cap must contrast with tape tokens")
    objects, carrier = [], None
    for cells in components:
        if cells == launcher:
            continue
        values, counts = np.unique([inp[q] for q in sorted(cells)], return_counts=True)
        body_color = int(values[int(np.argmax(counts))])
        _require(sum(counts == counts.max()) == 1 and len(values) >= 3,
                 "relay requires a uniquely most numerous carrier and at least two colored ports")
        _require(carrier is None or body_color == carrier, "one scene-local carrier required")
        carrier = body_color
        body = {q for q in cells if int(inp[q]) == carrier}
        r0, r1, c0, c1 = _bbox(cells)
        _require(0 < r0 <= r1 < h-1 and 0 < c0 <= c1 < w-1, "relays must be interior")
        ports, sides = {}, set()
        for color in values:
            color = int(color)
            if color == carrier:
                continue
            port = {q for q in cells if int(inp[q]) == color}
            choices = [d for d, ok in enumerate((all(r == r0 for r, c in port),
                       all(c == c1 for r, c in port), all(r == r1 for r, c in port),
                       all(c == c0 for r, c in port))) if ok]
            _require(len(choices) == 1, "port must lie on one extreme side, away from corners")
            d = choices[0]
            transverse = sorted(c if d % 2 == 0 else r for r, c in port)
            _require(1 <= len(port) <= 3 and transverse == list(range(transverse[0], transverse[0]+len(port))),
                     "ports are contiguous spans of width 1..3")
            _require(d not in sides, "at most one port per side")
            sides.add(d)
            dr, dc = _DIRS[d]
            _require(all((r-dr, c-dc) in body for r, c in port), "port needs inward carrier neighbors")
            ports[color] = (d, sorted(port))
        objects.append(dict(cells=cells, ports=ports, bbox=(r0, r1, c0, c1)))
    _require(objects and carrier not in (bg, cap_color) and carrier not in tape, "distinct color roles required")
    port_colors = set().union(*(set(o['ports']) for o in objects))
    _require(not port_colors & {bg, carrier, cap_color} and set(tape) <= port_colors,
             "tape colors must denote port colors")
    # Objects and tape have at least one blank cell between them, diagonally too.
    for i, a in enumerate(components):
        for b in components[i+1:]:
            _require(not any((r+dr, c+dc) in b for r, c in a
                             for dr in (-1, 0, 1) for dc in (-1, 0, 1)), "objects touch")
    return dict(bg=bg, carrier=carrier, cap_color=cap_color, tape=tape,
                line=line, cap=cap, initial=direction, objects=objects)


def _project(shape, objects, origins, direction):
    """Find the nearest original footprint in every lane, using distances."""
    h, w = shape
    dr, dc = _DIRS[direction]
    stops = []
    for r, c in origins:
        boundary = r+1 if direction == 0 else w-c if direction == 1 else h-r if direction == 2 else c+1
        best = (boundary, -1)
        for i, obj in enumerate(objects):
            for rr, cc in obj['cells']:
                distance = (rr-r)*dr + (cc-c)*dc
                if distance > 0 and (rr-r)*dc - (cc-c)*dr == 0 and distance < best[0]:
                    best = distance, i
        stops.append(best)
    _require(len(set(stops)) == 1, "partial-width or simultaneous different-object collision")
    distance, target = stops[0]
    segment = {(r+k*dr, c+k*dc) for r, c in origins for k in range(1, distance)}
    return target, segment, distance-1


def _analyze(inp):
    a = _parse(inp)
    out = np.full_like(inp, a['bg'])
    for q in a['line'][:-1]:
        out[q] = inp[q]
    out[a['cap']] = a['tape'][0]
    origins, direction, ink = [a['cap']], a['initial'], a['tape'][0]
    painted = set(a['line'])
    visited, widths, directions, gaps, segments = [], [], [], [], []
    for token in a['tape']:
        i, segment, gap = _project(inp.shape, a['objects'], origins, direction)
        _require(i >= 0 and i not in visited, "tape must reach a new relay for every instruction")
        _require(not segment & painted, "rays must not overlap earlier traces or the tape")
        visited.append(i)
        obj = a['objects'][i]
        _require(token in obj['ports'], "reached relay lacks the current instruction's port")
        d, port = obj['ports'][token]
        _require(d != (direction+2) % 4, "selected port must not reverse into the entry face")
        for q in segment:
            out[q] = ink
        for q in obj['cells']:
            out[q] = token
        painted |= segment | obj['cells']
        segments.append(segment)
        directions.append((direction, d))
        gaps.append(gap)
        widths.append(len(port))
        origins, direction, ink = port, d, token
    i, segment, gap = _project(inp.shape, a['objects'], origins, direction)
    _require(i == -1 and not segment & painted, "final selected port must reach an unobstructed boundary")
    for q in segment:
        out[q] = ink
    segments.append(segment)
    # Alternative ports pointing at unused relays show that the program selects
    # reachability, rather than simply annotating an already determined route.
    branches = []
    for step, i in enumerate(visited):
        for color, (d, port) in a['objects'][i]['ports'].items():
            if color == a['tape'][step]:
                continue
            try:
                j, _, _ = _project(inp.shape, a['objects'], port, d)
            except ValueError:
                continue
            if j >= 0 and j not in visited:
                branches.append((step, color, j))
    a.update(visited=visited, widths=widths, directions=directions, gaps=gaps,
             segments=segments, branches=branches, unused=len(a['objects'])-len(visited))
    return out, a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Run the visible tape; no seed, examples, construction state, or cache."""
    return _analyze(_grid(input_grid))[0]


def _port(obj, side, width):
    r0, r1, c0, c1 = obj['bbox']
    center = (c0+c1)//2 if side % 2 == 0 else (r0+r1)//2
    start = center - width//2
    if side == 0:
        cells = [(r0, start+k) for k in range(width)]
    elif side == 2:
        cells = [(r1, start+k) for k in range(width)]
    elif side == 1:
        cells = [(start+k, c1) for k in range(width)]
    else:
        cells = [(start+k, c0) for k in range(width)]
    dr, dc = _DIRS[side]
    for q in cells:
        interior_span = r0 < q[0] < r1 if side % 2 else c0 < q[1] < c1
        if not (interior_span and q in obj['cells'] and (q[0]-dr, q[1]-dc) in obj['cells']):
            return None
    return cells


def _shape(rng, r, c, h, w, rich):
    cells = {(y, x) for y in range(r-h//2, r+h//2+1) for x in range(c-w//2, c+w//2+1)}
    if rich and min(h, w) >= 5:
        corners = [(r-h//2, c-w//2), (r-h//2, c+w//2),
                   (r+h//2, c-w//2), (r+h//2, c+w//2)]
        for q in corners:
            if rng.random() < .55:
                cells.remove(q)
        # An asymmetric shallow notch avoids the central firing lanes.
        if h >= 7 and w >= 5 and rng.random() < .45:
            cells.discard((r-h//2, c-w//2+1))
    return dict(cells=cells, bbox=_bbox(cells))


def _random_scene(rng, n, decoys, rich):
    total = n+decoys
    options = [(r, c) for r in range(1, 5) for c in range(1, 5)
               if total <= r*c <= total+4 and r*5+r+1 <= 30 and n+2+c*5+c <= 30]
    rows, cols = options[int(rng.integers(len(options)))]
    occupied = sorted(int(i) for i in rng.choice(rows*cols, total, replace=False))
    rh = [int(rng.choice((3, 5, 5, 5, 7))) for _ in range(rows)]
    cw = [int(rng.choice((3, 5, 5, 5, 7))) for _ in range(cols)]
    if sum(rh)+rows+1 > 30 or n+2+sum(cw)+cols > 30:
        return None
    # Independent gaps and margins; row/column bands need not be uniform.
    rg = [1]*max(0, rows-1)
    cg = [1]*max(0, cols-1)
    for gaps, spare in ((rg, 28-sum(rh)-sum(rg)), (cg, 28-n-2-sum(cw)-sum(cg))):
        for j in rng.permutation(len(gaps)):
            extra = int(rng.integers(0, min(5, max(0, spare))+1))
            gaps[int(j)] += extra
            spare -= extra
    min_h, min_w = 2+sum(rh)+sum(rg), n+3+sum(cw)+sum(cg)
    h = int(rng.integers(min_h, 31))
    w = int(rng.integers(min_w, 31))
    top = int(rng.integers(1, h-(sum(rh)+sum(rg))))
    left = int(rng.integers(n+2, w-(sum(cw)+sum(cg))))
    ys, xs = [top+rh[0]//2], [left+cw[0]//2]
    for j in range(1, rows):
        ys.append(ys[-1]+rh[j-1]//2+rg[j-1]+1+rh[j]//2)
    for j in range(1, cols):
        xs.append(xs[-1]+cw[j-1]//2+cg[j-1]+1+cw[j]//2)
    objects = []
    for k in occupied:
        i, j = divmod(k, cols)
        oh = int(rng.choice(list(range(3, rh[i]+1, 2))))
        ow = int(rng.choice(list(range(3, cw[j]+1, 2))))
        objects.append(_shape(rng, ys[i], xs[j], oh, ow, rich))
    return (h, w), objects, ys


def _find_route(rng, shape, objects, rows, n, widths, need_branch, need_straight):
    budget = [2500]
    def visit(path, selected, segments, trace, incoming, cap):
        budget[0] -= 1
        if budget[0] < 0:
            return None
        step, obj = len(path)-1, objects[path[-1]]
        for d in rng.permutation(4):
            d = int(d)
            if d == (incoming+2) % 4:
                continue
            port = _port(obj, d, widths[step])
            if port is None:
                continue
            try:
                hit, segment, _ = _project(shape, objects, port, d)
            except ValueError:
                continue
            if segment & trace:
                continue
            chosen = selected+[(d, port)]
            if len(path) == n:
                if hit != -1:
                    continue
                if need_straight and not any(a == b for a, b in zip([1]+[s[0] for s in chosen[:-1]], [s[0] for s in chosen])):
                    continue
                if need_straight and not any(a != b for a, b in zip([1]+[s[0] for s in chosen[:-1]], [s[0] for s in chosen])):
                    continue
                branches = []
                for j, index in enumerate(path):
                    for alt in range(4):
                        if alt == chosen[j][0]:
                            continue
                        p = _port(objects[index], alt, 1)
                        if p is None:
                            continue
                        try:
                            target, _, _ = _project(shape, objects, p, alt)
                        except ValueError:
                            continue
                        if target >= 0 and target not in path and (need_branch != 'later' or j > 0):
                            branches.append((j, alt, p))
                if need_branch and not branches:
                    continue
                branch = branches[int(rng.integers(len(branches)))] if branches else None
                return path, chosen, segments+[segment], cap, branch
            if hit < 0 or hit in path:
                continue
            found = visit(path+[hit], chosen, segments+[segment], trace|segment|objects[hit]['cells'], d, cap)
            if found is not None:
                return found
        return None
    for row in rng.permutation(rows):
        cap = (int(row), n)
        line = {(int(row), k) for k in range(n+1)}
        try:
            hit, segment, _ = _project(shape, objects, [cap], 1)
        except ValueError:
            continue
        if hit < 0 or segment & line:
            continue
        found = visit([hit], [], [segment], line|segment|objects[hit]['cells'], 1, cap)
        if found is not None:
            return found
    return None


def _fallback_scene(rng, n, decoys):
    paths = {2: [(0, 0), (0, 1)], 3: [(0, 0), (0, 1), (1, 1)],
             4: [(0, 0), (0, 1), (1, 1), (1, 2)],
             5: [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
             6: [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 1)]}
    occupied = list(paths[n])
    # Reserve the outward final ray; other unused sites are freely sampled.
    forbidden = {(1, 2)} if n == 3 else {(2, 2)} if n == 4 else set()
    available = [(1, 0)] + [(i, j) for i in range(3) for j in range(3)
                            if (i, j) not in set(occupied)|forbidden|{(1, 0)}]
    occupied += available[:decoys]
    objects = [_shape(rng, 3+i*7, n+5+j*7, 5, 5, False) for i, j in occupied]
    return (23, n+23), objects, [3, 10, 17]


def _build_pair(rng, n, decoys, wide, colors, rich, branch=None, straight=False, fallback=False):
    # Complexity is fixed before geometry; retries do not redraw n or decoys.
    widths = [int(x) for x in rng.choice([1, wide], n)]
    widths[:2] = [int(x) for x in rng.permutation([1, wide])]
    found = None
    for _ in range(0 if fallback else 75):
        scene = _random_scene(rng, n, decoys, rich)
        if scene is None:
            continue
        shape, objects, rows = scene
        found = _find_route(rng, shape, objects, rows, n, widths, branch, straight)
        if found is not None:
            break
    if found is None:
        shape, objects, rows = _fallback_scene(rng, n, decoys)
        found = _find_route(rng, shape, objects, rows, n, widths, branch, straight)
        _require(found is not None, "fallback route construction failed")
    path, selected, segments, cap, witness = found
    bg, carrier, cap_color = colors[:3]
    # Every relay offers a menu of two to four distinct port colors.
    pool = list(colors[3:])
    key = [int(x) for x in rng.choice(pool, n)]
    key[1] = pool[(pool.index(key[0])+1) % len(pool)]
    if witness and branch == 'later':
        key[witness[0]] = pool[(pool.index(key[0])+1) % len(pool)]
    if branch == 'any' and key == key[::-1]:
        # One coverage tape distinguishes the two possible reading orders.
        key[-1] = pool[(pool.index(key[0])+1) % len(pool)]
    inp = np.full(shape, bg, dtype=np.int8)
    out = np.full(shape, bg, dtype=np.int8)
    for i, obj in enumerate(objects):
        for q in obj['cells']:
            inp[q] = carrier
        step = path.index(i) if i in path else None
        assigned = {}
        if step is not None:
            assigned[key[step]] = selected[step]
        if witness and step == witness[0]:
            alt_color = key[0] if branch == 'later' else next(c for c in pool if c != key[step])
            assigned[alt_color] = witness[1:]
        for color in rng.permutation(pool):
            color = int(color)
            if color in assigned:
                continue
            candidates = []
            used = {d for d, p in assigned.values()}
            for d in range(4):
                if d in used:
                    continue
                for width in (1, 2, 3):
                    port = _port(obj, d, width)
                    if port is not None:
                        candidates.append((d, port))
            if not candidates:
                continue
            assigned[color] = candidates[int(rng.integers(len(candidates)))]
        _require(len(assigned) >= 2, "relay needs a choice of ports")
        for color, (d, port) in assigned.items():
            for q in port:
                inp[q] = color
    for j, color in enumerate(key):
        inp[cap[0], j] = out[cap[0], j] = color
    inp[cap] = cap_color
    out[cap] = key[0]
    for j, segment in enumerate(segments):
        ink = key[max(0, j-1)]
        for q in segment:
            out[q] = ink
    for j, i in enumerate(path):
        for q in objects[i]['cells']:
            out[q] = key[j]
    # Rotation and reflection apply to tape and scene together: reading order
    # is always boundary-to-cap, not a hidden compass convention.
    if rng.random() < .5:
        inp, out = np.flipud(inp), np.flipud(out)
    rotation = int(rng.integers(4))
    return dict(input=np.rot90(inp, rotation).copy(), output=np.rot90(out, rotation).copy())


def _signature(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    nq = _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    wide = int(rng.choice((2, 3)))
    base = [int(x) for x in rng.permutation(10)]
    varied = bool(rng.random() < .7)
    rich = bool(rng.random() < .7)
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            anchor = split == 'train' and index < 2
            n = 3+index if anchor else int(rng.integers(2, 5)) if split == 'train' else int(rng.integers(5, 7))
            decoys = int(rng.integers(1 if anchor else 0, 4))
            pool_size = int(rng.integers(2, 5))
            palette = base if not varied or (anchor and index == 0) else list(np.roll(base, 1)) if anchor else list(map(int, rng.permutation(10)))
            colors = palette[:3+pool_size]
            for attempt in range(20):
                pair = _build_pair(rng, n, decoys, wide, colors, rich,
                                   branch=('any' if index == 0 else 'later') if anchor else None,
                                   straight=anchor, fallback=attempt == 19)
                key = _signature(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('distinct input construction failed')
            seen.add(key)
            puzzle[split].append(pair)
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'illegal pair counts')
    seen, facts = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'shape must be preserved')
            signature = _signature(inp)
            _require(signature not in seen, 'duplicate inputs')
            seen.add(signature)
            expected, a = _analyze(inp)
            _require(np.array_equal(expected, out), 'incorrect programmed trace')
            _require(not np.array_equal(inp, out), 'output must change')
            # Explicit footprint and erasure checks, in addition to prediction.
            for j, obj in enumerate(a['objects']):
                color = a['tape'][a['visited'].index(j)] if j in a['visited'] else a['bg']
                _require(all(int(out[q]) == color for q in obj['cells']), 'whole-footprint activation/erasure failed')
            _require(all(int(out[q]) == int(inp[q]) for q in a['line'][:-1]), 'instruction tape changed')
            facts[split].append(a)
    train, test = facts['train'], facts['test']
    _require(any(a['branches'] for a in train), 'training must show a rejected reachable alternative')
    _require(any(a['tape'] != a['tape'][::-1] for a in train), 'training must distinguish tape reading order')
    _require(any(j > 0 and a['tape'][j] != a['tape'][0] and color == a['tape'][0]
                 for a in train for j, color, target in a['branches']), 'training must show later instruction consumption')
    widths = {w for a in train for w in a['widths']}
    _require(1 in widths and len(widths) >= 2, 'training must demonstrate changing widths')
    _require(all(set(a['widths']) <= widths for a in test), 'test selected widths need demonstration')
    turns = [x == y for a in train for x, y in a['directions']]
    _require(any(turns) and not all(turns), 'training must show straight emission and a turn')
    _require(all(len(a['tape']) > max(len(b['tape']) for b in train) for a in test),
             'tests must extend the demonstrated program length')


def _fixture(first=2, swap=False, width=1):
    """Small valid fork for causal checks, not a stored generated puzzle."""
    inp = np.zeros((24, 24), dtype=np.int8)
    inp[8, 0:3] = [first, 3, 4]
    boxes = [(6, 6), (6, 16), (16, 6)]
    for r, c in boxes:
        inp[r:r+5, c:c+5] = 1
    a, b = (3, 2) if swap else (2, 3)
    start = 8-width//2
    inp[start:start+width, 10] = a
    inp[10, 8] = b
    inp[6, 18] = 3
    inp[8, 20] = 2
    inp[18, 10] = 3
    inp[20, 8] = 2
    return inp


def self_test() -> dict:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, normalized, shapes = [], set(), set()
    for p in puzzles:
        validate_generated_puzzle(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                inp = pair['input']
                expected, a = _analyze(inp)
                assert np.array_equal(solve(inp), expected)
                assert np.array_equal(solve(inp.tolist()), expected)
                shapes.add(inp.shape)
                mapping = {}
                normalized.add((inp.shape, tuple(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)))
                records.append((split, a))
    assert len({_signature(p['train'][0]['input']) for p in puzzles}) == 42
    assert len(shapes) >= 25 and len(normalized) >= 120
    assert {a['initial'] for s, a in records} == set(range(4))
    assert {len(a['tape']) for s, a in records} == {2, 3, 4, 5, 6}
    assert {w for s, a in records for w in a['widths']} == {1, 2, 3}
    assert {a['unused'] for s, a in records} >= {0, 1, 2, 3}
    assert any(a['tape'][0] in a['tape'][1:] for s, a in records)
    assert any(any(x == y for x, y in zip(a['tape'], a['tape'][1:])) for s, a in records)
    assert len({tuple(o['bbox'] for o in a['objects']) for s, a in records}) >= 120
    # Geometry and tape independently cause different answers in a valid fork.
    x = _fixture()
    assert _analyze(x)[1]['visited'] != _analyze(_fixture(first=3))[1]['visited']
    assert _analyze(x)[1]['visited'] != _analyze(_fixture(swap=True))[1]['visited']
    assert not np.array_equal(solve(x), solve(_fixture(width=3)))
    offset = x.copy()
    offset[8, 10], offset[7, 10] = 1, 2
    assert not np.array_equal(solve(x), solve(offset))
    hole = x.copy()
    hole[8, 8] = 0
    assert solve(hole)[8, 8] == 0 and solve(hole)[8, 7] == 2
    # Color renaming changes only literal colors, never routing semantics.
    permutation = np.array([7, 4, 9, 0, 6, 1, 8, 3, 5, 2], dtype=np.int8)
    assert np.array_equal(solve(permutation[x]), permutation[solve(x)])
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(x, rotation)), np.rot90(solve(x), rotation))
    assert np.array_equal(solve(np.flipud(x)), np.flipud(solve(x)))
    # Exercise the bounded fallback directly, preserving counts and witnesses.
    rng = np.random.default_rng(593)
    for n in range(2, 7):
        for wide in (2, 3):
            pair = _build_pair(rng, n, 3, wide, list(range(7)), True,
                               branch='later' if n >= 3 else 'any', straight=True, fallback=True)
            out, a = _analyze(pair['input'])
            assert np.array_equal(out, pair['output']) and len(a['visited']) == n and a['unused'] == 3
    # All explicit count combinations, determinism, and rejected bad counts.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            generate_puzzle(71+nt*10+nq, nt, nq)
    def signature(p):
        return tuple(_signature(pair[k]) for s in ('train', 'test') for pair in p[s] for k in ('input', 'output'))
    assert signature(puzzles[0]) == signature(generate_puzzle(0))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, pairs=len(records), shapes=len(shapes), normalized_inputs=len(normalized))
