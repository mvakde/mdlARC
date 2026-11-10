"""Chromatic loom: a pointed inventory feeds a color-scheduled rail route.

Only NumPy and the standard library are used. No parent imports or latent state.
Diagnostics are streamed in memory; self_test writes no files.
"""
from collections import Counter
import hashlib
from itertools import combinations
import json
import numpy as np


_DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _need(bool(np.all((a >= 0) & (a <= 9))), "ARC colors required")
    return a


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _components(cells, diagonal=True):
    pending = set(cells)
    result = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while pending:
        start = min(pending)
        pending.remove(start)
        stack, part = [start], {start}
        while stack:
            p = stack.pop()
            for d in steps:
                q = _add(p, d)
                if q in pending:
                    pending.remove(q)
                    stack.append(q)
                    part.add(q)
        result.append(part)
    return result


def _envelope(cells, d):
    """Read thin, monotone pointed arms in axis-relative cross sections."""
    normal = (-d[1], d[0])
    local = {(-p[0]*d[0]-p[1]*d[1], p[0]*normal[0]+p[1]*normal[1]): p for p in cells}
    first, last = min(y for y, x in local), max(y for y, x in local)
    if last - first < 4:
        return None
    rows = [sorted(x for y, x in local if y == level) for level in range(first, last + 1)]
    if len(rows[0]) != 1:
        return None
    center = rows[0][0]
    lo = hi = center
    cavity, split = set(), False
    for offset, row in enumerate(rows[1:], 1):
        if len(row) not in (1, 2):
            return None
        left, right = row[0], row[-1]
        if left > lo or right < hi or lo-left > 1 or right-hi > 1:
            return None
        if len(row) == 1 and (split or left != center):
            return None
        if len(row) == 2:
            split = True
            for x in range(left + 1, right):
                y = first + offset
                cavity.add((-y*d[0]+x*normal[0], -y*d[1]+x*normal[1]))
        lo, hi = left, right
    if not (lo <= center - 2 and hi >= center + 2 and rows[-1] == rows[-2]):
        return None
    return local[first, center], cavity


def _hopper(g, bg):
    candidates = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        cells = set(map(tuple, np.argwhere(g == color)))
        # The reserved outline identity occurs only on this one frame.
        if len(cells) < 9 or len(_components(cells)) != 1:
            continue
        for d in _DIRS:
            geometry = _envelope(cells, d)
            if geometry is None:
                continue
            tip, cavity = geometry
            if not all(_inside(g, p) for p in cavity):
                continue
            inventory = {p for p in cavity if int(g[p]) != bg}
            colors = {int(g[p]) for p in inventory}
            if color in colors or len(colors) < 2:
                continue
            candidates.append(dict(frame=cells, frame_color=color, tip=tip,
                                   direction=d, cavity=cavity, inventory=inventory))
    _need(len(candidates) == 1, "one unambiguous pointed inventory is required")
    return candidates[0]


def _flanks(p, d):
    normal = (-d[1], d[0])
    return _add(p, normal), _add(p, (-normal[0], -normal[1]))


def _request(g, p, d, bg, frame_color):
    a, b = _flanks(p, d)
    if not (_inside(g, a) and _inside(g, b)):
        return None
    color = int(g[a])
    return color if color == int(g[b]) and color not in (bg, frame_color) else None


def _analyse(input_grid):
    g = _grid(input_grid)
    values, counts = np.unique(g, return_counts=True)
    bg = int(values[np.argmax(counts)])
    _need(np.count_nonzero(g == bg) > g.size // 2, "background must occupy a majority")
    hopper = _hopper(g, bg)
    p, d = hopper['tip'], hopper['direction']
    path, directions, requests, mismatches, gaps, turns = [], [], [], [], [], []
    seen = set()
    while True:
        state = p, d
        _need(state not in seen, "cyclic route")
        seen.add(state)
        q = _add(p, d)
        if not _inside(g, q):
            break
        if int(g[q]) == bg:
            _need(q not in path, "route revisits a cell")
            p = q
            path.append(p)
            directions.append(d)
            color = _request(g, p, d, bg, hopper['frame_color'])
            requests.append(color)
            a, b = _flanks(p, d)
            if _inside(g, a) and _inside(g, b):
                x, y = int(g[a]), int(g[b])
                if x not in (bg, hopper['frame_color']) and y not in (bg, hopper['frame_color']) and x != y:
                    mismatches.append(p)
                if x == y == bg:
                    gaps.append(p)
            continue
        choices = []
        for turn in ((-d[1], d[0]), (d[1], -d[0])):
            q = _add(p, turn)
            if (_inside(g, q) and int(g[q]) == bg
                    and _request(g, q, turn, bg, hopper['frame_color']) is not None):
                choices.append(turn)
        _need(len(choices) <= 1, "two competing turns")
        if not choices:
            break
        turns.append(p)
        d = choices[0]
    _need(path, "no route beyond tip")
    hopper.update(grid=g, background=bg, path=path, directions=directions,
                  requests=requests, mismatches=mismatches, gaps=gaps, turns=turns,
                  stock=Counter(int(g[p]) for p in hopper['inventory']),
                  demand=Counter(c for c in requests if c is not None),
                  open=path[-1][0] in (0, g.shape[0]-1) or path[-1][1] in (0, g.shape[1]-1))
    return hopper


def _answer(a, mode='normal'):
    out = a['grid'].copy()
    for p in a['inventory']:
        out[p] = a['background']
    stock = a['stock'].copy()
    pooled = sum(stock.values())
    for p, d, color in zip(a['path'], a['directions'], a['requests']):
        if mode == 'ignore_equality' and p in a['mismatches']:
            color = int(a['grid'][_flanks(p, d)[0]])
        if color is not None and (mode == 'unlimited' or
                                  (pooled > 0 if mode == 'pooled' else stock[color] > 0)):
            out[p] = color
            stock[color] -= 1
            pooled -= 1
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Count the visible inventory, trace visible rails, and satisfy requests."""
    return _answer(_analyse(input_grid))


def _profile(rng):
    upper, lower = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    stem, tail = int(rng.integers(0, 2)), int(rng.integers(3, 6))
    widths = [(0, 0)] * (stem + 1)
    for s in range(1, max(upper, lower) + 1):
        widths.append((-min(s, upper), min(s, lower)))
    widths.extend([(-upper, lower)] * tail)
    frame, cavity = set(), set()
    for s, (lo, hi) in enumerate(widths):
        frame.update(((lo, -s), (hi, -s)))
        cavity.update((r, -s) for r in range(lo+1, hi))
    return frame, cavity


def _polyline(points):
    path, directions = [], []
    for a, b in zip(points, points[1:]):
        d = (int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1])))
        if (d[0] == 0) == (d[1] == 0):
            return None
        p = a
        while p != b:
            p = _add(p, d)
            path.append(p)
            directions.append(d)
    if len(path) != len(set(path)):
        return None
    cells = set(path)
    for i, p in enumerate(path):
        degree = sum(_add(p, d) in cells for d in _DIRS)
        if degree != (1 if i in (0, len(path)-1) else 2):
            return None
    return path, directions


def _random_route(rng, h, w, tip, bends, open_end):
    points, d = [tip], (0, 1)
    for j in range(bends+1):
        p = points[-1]
        choices = []
        for length in range(4, max(h, w)):
            q = p[0]+d[0]*length, p[1]+d[1]*length
            if 1 <= q[0] <= h-2 and 1 <= q[1] <= w-2:
                choices.append(q)
        if j == bends and open_end:
            q = ((h-1 if d[0] > 0 else 0), p[1]) if d[0] else (p[0], w-1 if d[1] > 0 else 0)
        elif choices:
            q = choices[int(rng.integers(len(choices)))]
        else:
            return None
        points.append(q)
        d = (-d[1], d[0]) if rng.integers(2) else (d[1], -d[0])
    return _polyline(points)


def _walls_for(path, tip, frame, cavity, h, w):
    cells, occupied = set(path), frame | cavity
    if cells & occupied:
        return None
    walls = set()
    for p in path:
        for d in _DIRS:
            q = _add(p, d)
            if 0 <= q[0] < h and 0 <= q[1] < w and q not in cells and q != tip:
                if q in occupied:
                    return None
                walls.add(q)
    return walls


def _decorate(rng, path, directions, walls, palette, quotas):
    """Give independent cross-sections a shuffled color schedule.

    Outgoing elbow flanks are constrained equal. Other nearby shared walls
    retain that constraint. Far straight stations can independently mismatch
    or disappear without turning into flood-fill corridors.
    """
    pairs = {}
    usage = Counter()
    for i, (p, d) in enumerate(zip(path, directions)):
        pair = _flanks(p, d)
        if all(q in walls for q in pair):
            pairs[i] = pair
            usage.update(pair)
    independent = [i for i, pair in pairs.items() if all(usage[p] == 1 for p in pair)
                   and i >= 2 and i+2 < len(path)
                   and len(set(directions[i-2:i+3])) == 1]
    # Reserve a real early mismatch, a gap, and a request with no stock.
    required = sum(quotas) + 3
    if len(independent) < required:
        return None
    labels = [c for c, count in zip(palette, quotas) for _ in range(count)]
    extra = len(independent) - len(labels) - 3
    labels += [int(rng.choice(palette)) if rng.random() < .72 else None for _ in range(extra)]
    rng.shuffle(labels)
    schedule = {independent[0]: 'mismatch', independent[1]: 'gap',
                independent[2]: palette[-1]}
    schedule.update(zip(independent[3:], labels))
    colors = {p: int(rng.choice(palette)) for p in sorted(walls)}
    # Disjoint-set equalities handle the shared inside wall near an elbow.
    parent = {p: p for p in sorted(walls)}
    def root(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p
    def union(a, b):
        parent[root(a)] = root(b)
    for i, pair in pairs.items():
        if i not in schedule:
            union(*pair)
    roots = {p: root(p) for p in sorted(walls)}
    root_colors = {p: int(rng.choice(palette)) for p in sorted(set(roots.values()))}
    colors = {p: root_colors[roots[p]] for p in sorted(walls)}
    for i, label in schedule.items():
        a, b = pairs[i]
        if label == 'gap' or label is None:
            colors.pop(a, None)
            colors.pop(b, None)
        elif label == 'mismatch':
            colors[a], colors[b] = palette[0], palette[1]
        else:
            colors[a] = colors[b] = int(label)
    # An unrailed approach makes the reserved mismatch the first request-like
    # site, while its first inventory color is certainly still available.
    for i in (0, 1):
        if i in pairs:
            for p in pairs[i]:
                colors.pop(p, None)
    return colors


def _geometry(rng, palette, kind, quotas=None):
    """Bounded sampling with a roomy one-elbow fallback."""
    if quotas is None:
        quotas = [5, 4] + ([3] if len(palette) == 4 else [])
    for attempt in range(180):
        fallback = attempt >= 160
        low_h = 14 if kind == 'ordinary' else 20 if kind == 'witness' else 23
        low_w = 18 if kind == 'ordinary' else 23 if kind == 'witness' else 25
        h = w = 30 if fallback else int(rng.integers(low_h, 31))
        if not fallback:
            w = int(rng.integers(low_w, 31))
        f, cavity = _profile(rng)
        depth = -min(p[1] for p in f)
        tip = (h-6 if fallback else int(rng.integers(5, h-4)), depth+int(rng.integers(1, 4)))
        frame = {_add(p, tip) for p in f}
        cavity = {_add(p, tip) for p in cavity}
        if fallback:
            points = [tip, (tip[0], w-3), (1, w-3)]
            route = _polyline(points)
        else:
            bends = int(rng.choice([1, 2])) if kind == 'witness' else int(rng.choice([2, 3, 4])) if kind == 'test' else int(rng.integers(0, 5))
            route = _random_route(rng, h, w, tip, bends, bool(rng.integers(2)))
        if route is None:
            continue
        path, directions = route
        walls = _walls_for(path, tip, frame, cavity, h, w)
        if walls is None:
            continue
        colors = _decorate(rng, path, directions, walls, palette, quotas)
        if colors is None:
            continue
        return h, w, frame, cavity, tip, path, directions, colors
    raise AssertionError('bounded geometry construction exhausted')


def _payload(rng, cavity, stock, palette, style):
    remaining = set(cavity)
    assigned = {}
    # The second inventory color is deliberately connected in witness 0,
    # disconnected in witness 1, with the SAME cell count.
    for index in [1, 0] + list(range(2, len(stock))):
        quantity = stock[index]
        chosen = set()
        if style == 'compact' and index == 1:
            for start in sorted(remaining):
                run = {(start[0], start[1]+j) for j in range(quantity)}
                if run <= remaining:
                    chosen = run
                    break
        elif style == 'scattered' and index == 1:
            candidates = list(sorted(remaining))
            rng.shuffle(candidates)
            for p in candidates:
                if all(max(abs(p[0]-q[0]), abs(p[1]-q[1])) > 1 for q in chosen):
                    chosen.add(p)
                    if len(chosen) == quantity:
                        break
            if len(chosen) < quantity:
                for group in combinations(candidates, quantity):
                    if all(max(abs(a[0]-b[0]), abs(a[1]-b[1])) > 1
                           for a, b in combinations(group, 2)):
                        chosen = set(group)
                        break
            _need(len(chosen) == quantity, 'scattered control does not fit')
        if len(chosen) != quantity:
            cells = sorted(remaining)
            chosen = {cells[int(i)] for i in rng.choice(len(cells), quantity, replace=False)}
        for p in chosen:
            assigned[p] = palette[index]
        remaining -= chosen
    return assigned


def _render(rng, geometry, roles, stock, style, rotation):
    bg, frame_color, palette = roles
    h, w, frame, cavity, tip, path, directions, walls = geometry
    inp = np.full((h, w), bg, dtype=np.int8)
    for p in frame:
        inp[p] = frame_color
    for p, color in walls.items():
        inp[p] = color
    payload = _payload(rng, cavity, stock, palette, style)
    for p, color in payload.items():
        inp[p] = color
    out = inp.copy()
    for p in payload:
        out[p] = bg
    left = Counter(payload.values())
    for p, d in zip(path, directions):
        a, b = _flanks(p, d)
        # Direct construction label, using the constructed route rather than
        # asking solve to create its own expected answer.
        if 0 <= a[0] < h and 0 <= a[1] < w and 0 <= b[0] < h and 0 <= b[1] < w:
            color = int(inp[a])
            if color == int(inp[b]) and color not in (bg, frame_color) and left[color] > 0:
                out[p] = color
                left[color] -= 1
    _need(not any(left.values()), 'insufficient scheduled capacity')
    # A different input reader must recover the intended complete route.
    analysis = _analyse(inp)
    _need(analysis['path'] == path, 'painted rails disagree with constructed route')
    _need(np.array_equal(_answer(analysis), out), 'construction/reader disagreement')
    return {"input": np.rot90(inp, rotation).copy(), "output": np.rot90(out, rotation).copy()}


def _check_counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def _key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = list(map(int, rng.permutation(10)))
    k = int(rng.integers(2, 4))
    roles = colors[0], colors[1], colors[2:3+k]
    palette = roles[2]
    rotation = int(rng.integers(4))
    shared = _geometry(rng, palette, 'witness')
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            for attempt in range(24):
                if split == 'train' and index < 2:
                    geometry = shared
                    pair_roles, turn = roles, rotation
                    stock = [2 if index == 0 else 4, 3] + ([2] if k == 3 else [])
                    style = 'compact' if index == 0 else 'scattered'
                else:
                    # The control pair keeps roles fixed; later examples can
                    # permute every literal role without changing the rule.
                    perm = list(map(int, rng.permutation(10)))
                    pair_roles = (perm[0], perm[1], perm[2:3+k]) if rng.random() < .55 else roles
                    stock = [int(rng.integers(5, 7)) if split == 'test' else int(rng.integers(1, 6)),
                             int(rng.integers(1, 5))] + ([int(rng.integers(1, 4))] if k == 3 else [])
                    geometry = _geometry(rng, pair_roles[2], 'test' if split == 'test' else 'ordinary',
                                         quotas=[q+1 for q in stock])
                    turn = int(rng.integers(4))
                    style = 'random'
                try:
                    pair = _render(rng, geometry, pair_roles, stock, style, turn)
                    a = _analyse(pair['input'])
                    if split == 'train' and index == 0 and not _export_after_turn(a, pair['output']):
                        shared = _geometry(rng, palette, 'witness')
                        continue
                    if np.array_equal(_answer(a, 'ignore_equality'), pair['output']):
                        if split == 'train' and index == 0:
                            shared = _geometry(rng, palette, 'witness')
                        continue
                    if np.array_equal(_answer(a, 'pooled'), pair['output']):
                        if split == 'train' and index == 0:
                            shared = _geometry(rng, palette, 'witness')
                        continue
                    key = _key(pair['input'])
                    if key in seen:
                        continue
                except AssertionError:
                    if split == 'train' and index == 0:
                        shared = _geometry(rng, palette, 'witness')
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise AssertionError('bounded pair construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_evidence(pairs):
    analyses = [_analyse(p['input']) for p in pairs]
    _need(len(analyses) >= 2, 'two demonstrations required')
    a, b = analyses[:2]
    _need(a['path'] == b['path'] and a['frame'] == b['frame'], 'missing controlled geometry')
    _need(a['background'] == b['background'] and a['requests'] == b['requests'], 'changed control conventions')
    low, high = sorted(a['stock'].values()), sorted(b['stock'].values())
    _need(low != high, 'count never changes at fixed geometry')
    _need(all(_export_after_turn(a, p['output']) for a, p in zip(analyses[:2], pairs[:2])),
          'controlled demonstrations must visibly route inventory around a bend')
    repeated = set(a['stock']) & set(b['stock'])
    _need(any(a['stock'][c] == b['stock'][c] >= 3 and
              len(_components({p for p in a['inventory'] if a['grid'][p] == c})) == 1 and
              len(_components({p for p in b['inventory'] if b['grid'][p] == c})) == a['stock'][c]
              for c in repeated), 'same count needs connected/scattered evidence')
    return analyses


def _export_after_turn(a, output):
    output = np.asarray(output)
    if not a['turns']:
        return False
    first = a['path'].index(a['turns'][0])
    return any(output[p] != a['background'] for p in a['path'][first+1:])


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _need(all(isinstance(puzzle[k], list) for k in puzzle), 'split lists')
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, 'shape preservation')
            key = _key(inp)
            _need(key not in seen, 'duplicate input')
            seen.add(key)
            a = _analyse(inp)
            _need(np.array_equal(out, _answer(a)), 'incorrect scheduled export')
            _need(a['mismatches'] and a['gaps'], 'mismatch/gap contrast missing')
            _need(all(a['demand'][c] > q for c, q in a['stock'].items()), 'exhaustion must remain visible')
            _need(set(a['demand']) - set(a['stock']), 'unstocked request identity missing')
            _need(all(out[p] == a['background'] for p in a['inventory']), 'inventory not emptied')
            fixed = np.ones(inp.shape, dtype=bool)
            for p in a['inventory'] | set(a['path']):
                fixed[p] = False
            _need(np.array_equal(inp[fixed], out[fixed]), 'frame/rails/exterior changed')
            for color in range(10):
                _need(np.count_nonzero(inp == color) == np.count_nonzero(out == color), 'color inventory conservation')
            for color, quantity in a['stock'].items():
                eligible = [p for p, c in zip(a['path'], a['requests']) if c == color]
                _need(all(out[p] == color for p in eligible[:quantity]), 'earliest requests must be filled')
                _need(all(out[p] == a['background'] for p in eligible[quantity:]), 'requests after exhaustion must be empty')
            _need(not np.array_equal(out, _answer(a, 'unlimited')), 'count ablation has no effect')
            _need(not np.array_equal(out, _answer(a, 'ignore_equality')), 'matching ablation has no effect')
            _need(not np.array_equal(out, _answer(a, 'pooled')), 'per-color stock ablation has no effect')
            if split == 'test':
                _need(max(a['stock'].values()) > 4, 'test must increase the controlled inventory range')
    _training_evidence(puzzle['train'])


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds, equivariance, counts, mutations, and actual geometry."""
    signatures, shapes, profiles, schedules = set(), set(), set(), set()
    turns, directions, quantities, backgrounds, end_types, palette_sizes = set(), set(), set(), set(), set(), set()
    pairs_checked = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_fingerprint(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                g = pair['input']
                a = _analyse(g)
                shapes.add(g.shape)
                profiles.add(tuple(sorted((x[0]-a['tip'][0], x[1]-a['tip'][1]) for x in a['frame'])))
                schedules.add(tuple(a['requests']))
                turns.add(len(a['turns']))
                directions.add(a['direction'])
                quantities.update(a['stock'].values())
                backgrounds.add(a['background'])
                end_types.add(a['open'])
                palette_sizes.add(len(a['demand']))
                pairs_checked += 1
        # Role and rotation changes must follow the grid, including background 0.
        pair = p['test'][0]
        perm = np.random.default_rng(seed+912).permutation(10)
        for k in range(4):
            changed = perm[np.rot90(pair['input'], k)]
            expected = perm[np.rot90(pair['output'], k)]
            _need(np.array_equal(solve(changed), expected), 'rotation/palette equivariance')
        if seed < 3:
            _need(_fingerprint(p) == _fingerprint(generate_puzzle(seed)), 'nondeterminism')
    _need(len(signatures) == 42 and len(shapes) >= 25, 'cross-seed diversity')
    _need(len(profiles) >= 30 and len(schedules) >= 100, 'structural variation')
    _need({1, 2, 3} <= turns and len(directions) == 4, 'routing variation')
    _need(set(range(1, 7)) <= quantities and len(backgrounds) == 10, 'inventory/palette variation')
    _need(end_types == {False, True} and {3, 4} <= palette_sizes, 'terminal/request variation')
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            q = generate_puzzle(100+n*7+m, n, m)
            _need(len(q['train']) == n and len(q['test']) == m, 'explicit counts')
    for n, m in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(4.0,1),(4,None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    bad = generate_puzzle(11, 2, 1)
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('identity output mutation accepted')
    return {'seeds':42,'pairs':pairs_checked,'shapes':len(shapes),'profiles':len(profiles),
            'schedules':len(schedules),'turn_counts':sorted(turns),'directions':len(directions),
            'inventory_quantities':sorted(quantities),'backgrounds':len(backgrounds),
            'request_palette_sizes':sorted(palette_sizes),'equivariance_checks':168}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
