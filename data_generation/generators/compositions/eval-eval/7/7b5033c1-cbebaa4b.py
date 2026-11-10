"""Last-visit bridges: trace a fixed guide, then dock rigid two-pin pieces.

Only the input is needed by solve. Construction metadata never reaches it.
Generation and self_test use bounded loops, keep no on-disk cache or traces, and
retain at most one puzzle at a time (14 pairs of <= 30x30 int8 grids).
"""
from collections import deque
import hashlib

import numpy as np


N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "invalid colors")
    return a


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, found = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in N4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _shift(cells, d):
    return {(r + d[0], c + d[1]) for r, c in cells}


def _origin(cells):
    return min(r for r, c in cells), min(c for r, c in cells)


def _mask(cells):
    t, l = _origin(cells)
    return tuple(sorted((r - t, c - l) for r, c in cells))


def _chain(cells):
    neighbors = {q: [(q[0] + dr, q[1] + dc) for dr, dc in N4
                     if (q[0] + dr, q[1] + dc) in cells] for q in cells}
    ends = [q for q in neighbors if len(neighbors[q]) == 1]
    if len(ends) != 2 or any(len(v) not in (1, 2) for v in neighbors.values()):
        return None
    if ends[0][0] == ends[1][0]:
        return None
    start = min(ends, key=lambda q: q[0])
    order, previous, current = [], None, start
    while current is not None:
        order.append(current)
        nxt = [q for q in neighbors[current] if q != previous]
        previous, current = current, (nxt[0] if nxt else None)
        if len(order) > len(cells):
            return None
    return order if len(order) == len(cells) else None


def _square(cells):
    return any({(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} <= cells
               for r, c in cells)


def _analyze(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(counts[bg] > grid.size // 2, "background must occupy a strict majority")
    cc = _components(map(tuple, np.argwhere(grid != bg).tolist()))
    chains = [(i, _chain(c)) for i, c in enumerate(cc)]
    chains = [(i, p) for i, p in chains if p is not None]
    _require(len(chains) == 1, "expected one guide path")
    guide_index, path = chains[0]
    guide = set(path)
    guide_colors = {int(grid[q]) for q in path}
    guide_hist = {c: sum(grid[q] == c for q in path) for c in guide_colors}
    carrier = max(guide_hist, key=guide_hist.get)
    _require(all(guide_hist[carrier] > v for c, v in guide_hist.items() if c != carrier),
             "guide carrier is not distinguishable")
    pieces = []
    for i, cells in enumerate(cc):
        if i == guide_index:
            continue
        colors = {int(grid[q]) for q in cells}
        pin_hues = colors & guide_colors
        _require(len(colors) == 2 and len(pin_hues) == 1, "piece needs a body hue and a guide hue")
        hue = next(iter(pin_hues))
        _require(hue != carrier, "carrier cannot be a pin hue")
        body_hue = next(iter(colors - pin_hues))
        pins = {q for q in cells if grid[q] == hue}
        body = cells - pins
        _require(len(pins) == 2 and len(_components(body)) == 1 and _square(body),
                 "piece must have two pins and a connected body containing a 2x2 block")
        visits = [q for q in path if grid[q] == hue]
        _require(len(visits) >= 4 and len(visits) % 2 == 0, "need at least two complete visits per hue")
        _require(_mask(pins) == _mask(visits[-2:]), "last socket pair does not match the pins")
        pieces.append(dict(cells=cells, pins=pins, body=body, hue=hue,
                           body_hue=body_hue, visits=visits))
    _require(1 <= len(pieces) <= 3, "expected one through three bridges")
    _require(len({p['hue'] for p in pieces}) == len(pieces), "one piece per pin hue")
    _require({p['hue'] for p in pieces} == guide_colors - {carrier}, "unassigned guide hue")
    return dict(grid=grid, bg=bg, guide=guide, path=path, carrier=carrier, pieces=pieces)


def _assemble(info, earlier=False):
    """The earlier option is used only for an explicit order-ablation check."""
    grid, bg = info['grid'], info['bg']
    out = np.full(grid.shape, bg, dtype=np.int8)
    for q in info['guide']:
        out[q] = grid[q]
    positions = []
    for p in info['pieces']:
        target = set(p['visits'][:2] if earlier else p['visits'][-2:])
        a, b = min(p['pins']), min(target)
        d = b[0] - a[0], b[1] - a[1]
        _require(_shift(p['pins'], d) == target, "no translation matches the complete pin mask")
        body = _shift(p['body'], d)
        _require(all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] for r, c in body),
                 "docked body crosses boundary")
        _require(all(out[q] == bg for q in body), "docked body overlaps the guide or another body")
        for q in body:
            out[q] = p['body_hue']
        positions.append(d)
    return out, positions


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Trace from the higher endpoint and rigidly dock to the last two visits."""
    return _assemble(_analyze(input_grid))[0]


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f'{name} outside its integer range')
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _make_guide(rng, long=False, above=False):
    """A variable-pitch lattice walk, expanded to separate nearby folds."""
    for _ in range(160):
        gh, gw = _pick(rng, ((4, 5), (5, 4), (5, 5))) if long else (int(rng.integers(3, 6)), int(rng.integers(3, 6)))
        low, high = (18, 22) if long else (9, 13)
        size = int(rng.integers(min(low, gh * gw - 1), min(high, gh * gw) + 1))
        start = int(rng.integers(gh)), int(rng.integers(gw))
        coarse, used = [start], {start}
        for step in range(size - 1):
            r, c = coarse[-1]
            choices = [(r + dr, c + dc) for dr, dc in N4
                       if 0 <= r + dr < gh and 0 <= c + dc < gw and (r + dr, c + dc) not in used]
            if not choices:
                break
            q = _pick(rng, choices)
            coarse.append(q)
            used.add(q)
        if len(coarse) != size:
            continue
        ys = np.r_[0, np.cumsum(rng.integers(4, 6, size=gh - 1))].tolist()
        xs = np.r_[0, np.cumsum(rng.integers(4, 6, size=gw - 1))].tolist()
        points = [(ys[r], xs[c]) for r, c in coarse]
        path = [points[0]]
        for a, b in zip(points, points[1:]):
            dr, dc = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
            for j in range(abs(b[0] - a[0]) + abs(b[1] - a[1])):
                path.append((path[-1][0] + dr, path[-1][1] + dc))
        if path[0][0] == path[-1][0]:
            continue
        if path[0][0] > path[-1][0]:
            path.reverse()
        if above and min(r for r, c in path) >= path[0][0]:
            continue
        moves = {(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])}
        if len(moves) < 4:
            continue
        t, l = _origin(path)
        path = [(r - t, c - l) for r, c in path]
        bh, bw = max(r for r, c in path) + 1, max(c for r, c in path) + 1
        h = int(rng.integers(max(22, bh + 6), 31))
        w = int(rng.integers(max(22, bw + 6), 31))
        d = int(rng.integers(3, h - bh - 2)), int(rng.integers(3, w - bw - 2))
        return [(r + d[0], c + d[1]) for r, c in path], (h, w)
    raise RuntimeError('bounded guide construction failed')


def _route(rng, allowed, starts, goals):
    starts, goals = sorted(set(starts) & allowed), set(goals) & allowed
    if not starts or not goals:
        return None
    rng.shuffle(starts)
    queue, previous = deque(starts), {q: None for q in starts}
    directions = [N4[int(i)] for i in rng.permutation(4)]
    while queue:
        q = queue.popleft()
        if q in goals:
            path = {q}
            while previous[q] is not None:
                q = previous[q]
                path.add(q)
            return path
        for dr, dc in directions:
            p = q[0] + dr, q[1] + dc
            if p in allowed and p not in previous:
                previous[p] = q
                queue.append(p)
    return None


def _bridge_body(rng, signature, sites, guide, reserved, shape):
    """One rigid mask must fit ALL its earlier and final docking sites."""
    pins = set(signature)
    maxr, maxc = max(r for r, c in pins), max(c for r, c in pins)
    allowed = {(r, c) for r in range(-2, maxr + 3) for c in range(-2, maxc + 3)} - pins
    for origin in sites:
        allowed = {q for q in allowed if 0 <= q[0] + origin[0] < shape[0]
                   and 0 <= q[1] + origin[1] < shape[1]
                   and (q[0] + origin[0], q[1] + origin[1]) not in guide | reserved}
    a, b = signature
    na = [(a[0] + dr, a[1] + dc) for dr, dc in N4]
    nb = [(b[0] + dr, b[1] + dc) for dr, dc in N4]
    body = _route(rng, allowed, na, nb)
    if body is None:
        return None
    squares = [{(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
               for r, c in sorted(allowed)
               if {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} <= allowed]
    rng.shuffle(squares)
    squares.sort(key=lambda s: min(abs(r - a) + abs(c - b) for r, c in s for a, b in body))
    for square in squares[:12]:
        route = _route(rng, allowed, body, square)
        if route is None:
            continue
        body |= route | square
        break
    else:
        return None
    for _ in range(int(rng.integers(0, 7))):
        frontier = sorted({(r + dr, c + dc) for r, c in body for dr, dc in N4} & allowed - body)
        if frontier:
            body.add(_pick(rng, frontier))
    if len(body) > 25:
        return None
    full = body | pins
    moved = [_shift(full, site) for site in sites]
    if any(a & b for i, a in enumerate(moved) for b in moved[:i]):
        return None
    if any(cells & reserved for cells in moved):
        return None
    return body


def _sites(path):
    groups = {}
    # Short windows include straight, L-shaped, and folded socket pairs.
    for i in range(len(path)):
        for j in range(i + 2, min(len(path), i + 8)):
            pins = {path[i], path[j]}
            sig = _mask(pins)
            if max(r for r, c in sig) > 4 or max(c for r, c in sig) > 4:
                continue
            if sum(abs(a - b) for a, b in zip(path[i], path[j])) < 2:
                continue
            groups.setdefault(sig, []).append((i, j, _origin(pins)))
    return groups


def _generate_pair(rng, roles, key_count, visit_count, long, reuse, witness, above=False):
    bg, carrier, pin_pool, body_pool = roles
    for _ in range(72):
        path, shape = _make_guide(rng, long, above)
        guide, groups = set(path), _sites(path)
        signatures = sorted(sig for sig, sites in groups.items() if len(sites) >= visit_count)
        reserved, selected = set(), []
        for key_index in range(key_count):
            for attempt in range(220):
                sig = _pick(rng, signatures)
                sites = groups[sig]
                chosen = sorted([sites[int(i)] for i in rng.choice(len(sites), visit_count, replace=False)])
                if any(a[1] >= b[0] for a, b in zip(chosen, chosen[1:])):
                    continue
                origins = [site[2] for site in chosen]
                pins = set().union(*(_shift(sig, origin) for origin in origins))
                if pins & reserved:
                    continue
                body = _bridge_body(rng, sig, origins, guide, reserved, shape)
                if body is None:
                    continue
                full = body | set(sig)
                all_cells = set().union(*(_shift(full, origin) for origin in origins))
                reserved |= all_cells
                selected.append(dict(body=body, pins=set(sig), sites=chosen))
                break
            else:
                break
        if len(selected) != key_count:
            continue
        if witness and not any(_origin(s['body']) != _origin(s['pins']) for s in selected):
            continue
        # Teach temporal order against both row-major and column-major reading.
        if witness and (not any(set((path[s['sites'][-1][0]], path[s['sites'][-1][1]])) !=
                                set(sorted([path[i] for x in s['sites'] for i in x[:2]])[-2:])
                                for s in selected) or
                        not any(set((path[s['sites'][-1][0]], path[s['sites'][-1][1]])) !=
                                set(sorted([path[i] for x in s['sites'] for i in x[:2]], key=lambda q: (q[1], q[0]))[-2:])
                                for s in selected)):
            continue
        pin_colors = [int(c) for c in rng.choice(pin_pool, key_count, replace=False)]
        if reuse:
            payloads = [int(_pick(rng, body_pool))] * key_count
        else:
            payloads = [int(c) for c in rng.choice(body_pool, key_count, replace=False)]
        inp = np.full(shape, bg, dtype=np.int8)
        for q in path:
            inp[q] = carrier
        for s, hue in zip(selected, pin_colors):
            for i, j, origin in s['sites']:
                inp[path[i]] = inp[path[j]] = hue
        out = inp.copy()
        for s, hue in zip(selected, payloads):
            for q in _shift(s['body'], s['sites'][-1][2]):
                out[q] = hue
        # Scatter in genuinely empty space; location encodes no piece order.
        forbidden = {(r + dr, c + dc) for r, c in guide for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        success = True
        for s, pin_hue, body_hue in zip(selected, pin_colors, payloads):
            full = s['body'] | s['pins']
            t, l = _origin(full)
            bh, bw = max(r for r, c in full) - t + 1, max(c for r, c in full) - l + 1
            offsets = [(r - t, c - l) for r in range(shape[0] - bh + 1) for c in range(shape[1] - bw + 1)]
            rng.shuffle(offsets)
            for d in offsets:
                cells = _shift(full, d)
                if cells & forbidden:
                    continue
                for q in _shift(s['body'], d):
                    inp[q] = body_hue
                for q in _shift(s['pins'], d):
                    inp[q] = pin_hue
                forbidden |= {(r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                break
            else:
                success = False
                break
        if success:
            return dict(input=inp, output=out)
    raise RuntimeError('bounded bridge construction failed')


def _digest(grid):
    return (tuple(grid.shape), hashlib.sha256(np.asarray(grid, dtype=np.uint8).tobytes()).hexdigest())


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(num_train, num_test)
    palette = [int(c) for c in rng.permutation(10)]
    roles = palette[0], palette[1], palette[2:5], palette[5:]
    regime = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            target = split == 'test'
            if regime == 0:
                keys = 3 if target else (2 if i < 2 else int(rng.integers(1, 3)))
            elif regime == 2:
                keys = 2
            else:
                keys = int(rng.integers(2, 4)) if target else 2
            visits = 3 if regime == 2 and target else 2
            long = target and regime in (1, 2)
            reuse = (target and regime == 3) or (regime != 3 and i >= 2 and rng.random() < 0.4)
            for _ in range(12):
                pair = _generate_pair(rng, roles, keys, visits, long, reuse, i < 2 or target,
                                      above=split == 'train' and i == 0)
                key = _digest(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input construction failed')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    grid, path, pieces = info['grid'], info['path'], info['pieces']
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    row_failure = any(set(p['visits'][-2:]) != set(sorted(p['visits'])[-2:]) for p in pieces)
    column_failure = any(set(p['visits'][-2:]) != set(sorted(p['visits'], key=lambda q: (q[1], q[0]))[-2:])
                         for p in pieces)
    return dict(shape=tuple(grid.shape), path_length=len(path), keys=len(pieces),
                turns=sum(a != b for a, b in zip(steps, steps[1:])),
                above=min(r for r, c in path) < path[0][0],
                row_failure=row_failure, column_failure=column_failure,
                corner_failure=any(_origin(p['body']) != _origin(p['pins']) for p in pieces),
                visits=tuple(len(p['visits']) // 2 for p in pieces),
                pin_masks=tuple(_mask(p['pins']) for p in pieces),
                body_masks=tuple(_mask(p['body']) for p in pieces),
                body_areas=tuple(len(p['body']) for p in pieces),
                reuse=len({p['body_hue'] for p in pieces}) < len(pieces),
                background=info['bg'], carrier=info['carrier'])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "wrong puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "wrong pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            info = _analyze(inp)
            expected, offsets = _assemble(info)
            _require(inp.shape == out.shape and np.array_equal(expected, out), "wrong rigid assembly")
            _require(all(d != (0, 0) for d in offsets), "loose pieces must move")
            _require(all(out[q] == inp[q] for q in info['guide']), "fixed guide changed")
            _require(_digest(inp) not in seen, "duplicate input")
            seen.add(_digest(inp))
            # Every generated example explicitly admits an earlier, geometrically
            # valid alternative. Order is essential, even in minimal 2/1 puzzles.
            earlier, _ = _assemble(info, earlier=True)
            _require(not np.array_equal(earlier, expected), "tracing contributes no decision")
            for p in info['pieces']:
                for k in range(0, len(p['visits']), 2):
                    _require(_mask(p['visits'][k:k + 2]) == _mask(p['pins']), "incomplete geometric decoy")
                _require(np.count_nonzero(out == p['hue']) == len(p['visits']), "pin overlap is not exact")
            features[split].append(_features(info))
    _require(any(f['row_failure'] and f['column_failure'] for f in features['train']),
             "training must distinguish spatial sorting")
    _require(any(f['row_failure'] and f['column_failure'] for f in features['test']),
             "test must require tracing")
    _require(any(f['above'] for f in features['train']), "higher endpoint needs an above-endpoint witness")
    _require(any(f['corner_failure'] for f in features['train']), "training must require pin registration")
    return None


def _nonlattice_regression():
    """A hand-built four-site history outside the sampler's 4/5-pitch plans."""
    path = [(4, 2)]
    for dr, dc, n in ((0, 1, 10), (1, 0, 6), (0, -1, 7),
                      (1, 0, 5), (0, 1, 11), (-1, 0, 2)):
        for _ in range(n):
            path.append((path[-1][0] + dr, path[-1][1] + dc))
    inp = np.zeros((25, 24), dtype=np.int8)
    for q in path:
        inp[q] = 1
    for q in ((4, 3), (4, 6), (4, 7), (4, 10), (10, 8), (10, 11), (15, 8), (15, 11)):
        inp[q] = 2
    out = inp.copy()
    inp[20, 1] = inp[20, 4] = 2
    inp[21:23, 1:5] = 3
    out[16:18, 8:12] = 3
    return dict(input=inp, output=out)


def self_test():
    """42 seeds, streamed metrics, count boundaries, and semantic corruptions."""
    shapes, paths, pins, bodies, areas, backgrounds, carriers, puzzles = (set() for _ in range(8))
    key_counts, visit_counts, turns = set(), set(), set()
    diagonal = reuse = above = total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(tuple(_digest(pair['input']) for s in ('train', 'test') for pair in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _analyze(pair['input'])
                f = _features(info)
                _require(np.array_equal(solve(pair['input']), pair['output']), "solve mismatch")
                shapes.add(f['shape'])
                paths.add(_mask(info['path']))
                pins.update(f['pin_masks'])
                bodies.update(f['body_masks'])
                areas.update(f['body_areas'])
                backgrounds.add(f['background'])
                carriers.add(f['carrier'])
                key_counts.add(f['keys'])
                visit_counts.update(f['visits'])
                turns.add(f['turns'])
                diagonal += any(max(r for r, c in p) > 0 and max(c for r, c in p) > 0 for p in f['pin_masks'])
                reuse += f['reuse']
                above += f['above']
                total += 1
    _require(len(puzzles) == 42 and len(shapes) >= 25 and len(paths) >= 150, "insufficient structural variation")
    _require(len(pins) >= 12 and len(bodies) >= 100 and len(areas) >= 8, "insufficient bridge variation")
    _require(len(backgrounds) >= 8 and len(carriers) >= 8, "insufficient role variation")
    _require(key_counts >= {1, 2, 3} and visit_counts >= {2, 3}, "missing count variation")
    _require(len(turns) >= 8 and diagonal >= 20 and reuse >= 10 and above >= 10, "missing mechanisms")
    fixture = _nonlattice_regression()
    _require(np.array_equal(solve(fixture['input']), fixture['output']), "nonlattice/four-site regression")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(910 + a * 10 + b, a, b)
            _require((len(p['train']), len(p['test'])) == (a, b), "wrong counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1),
                 (4, '1'), (None, 1), (4, None), (None, None), (4, False), (4, np.nan)):
        try:
            generate_puzzle(9, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    p, q = generate_puzzle(17, 2, 1), generate_puzzle(17, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                 for k in ('input', 'output')), "nondeterministic puzzle")
    for mutation in ('wrong_site', 'lost_body_pixel', 'moved_guide'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = bad['test'][0]
        info = _analyze(pair['input'])
        if mutation == 'wrong_site':
            pair['output'] = _assemble(info, earlier=True)[0]
        elif mutation == 'lost_body_pixel':
            piece = info['pieces'][0]
            d = _assemble(info)[1][0]
            pair['output'][min(_shift(piece['body'], d))] = info['bg']
        else:
            pair['output'][info['path'][0]] = info['bg']
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            continue
        raise AssertionError(f'{mutation} accepted')
    return dict(seeds=42, pairs=total, input_shapes=len(shapes), guide_shapes=len(paths),
                pin_masks=len(pins), body_masks=len(bodies), body_areas=sorted(areas),
                key_counts=sorted(key_counts), visit_counts=sorted(visit_counts),
                turn_counts=sorted(turns), diagonal_pairs=diagonal, reused_body_pairs=reuse,
                above_endpoint_pairs=above)
