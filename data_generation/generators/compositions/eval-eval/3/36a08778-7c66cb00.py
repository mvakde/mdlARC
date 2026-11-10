"""Broadcast stamps: the color-routed receptacles are also beam splitters.

Only NumPy and the standard library are needed. No files, global RNG, parent
generators, examples, or seeds are consulted by solve(). No traces are saved.
"""
from __future__ import annotations

import hashlib
import numpy as np


FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 289739534, 816215426, 423338332, 228911531, 377014721,
              799958525, 357685874, 686239356, 26574554, 942617962,
              945043229, 733701177, 36672128, 995678386, 155486527,
              834121285, 907118540, 229144087, 420821005, 913093709,
              89511568, 605743424, 378701566, 857397849, 738622797,
              435625318, 338876697, 631714540, 555856493, 354613460,
              449330618, 160845835, 105443393, 280863825, 422217226,
              221779537, 326954155, 633567215, 915297555, 217076198,
              121080178)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(value, name, lo, hi=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if value < lo or hi is not None and value > hi:
        raise ValueError(name + " is outside its legal range")
    return value


def _array(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "grid colors")
    return a.astype(np.int8, copy=False)


def _components(a, stop, bg, diagonal=False):
    unseen = set(map(tuple, np.argwhere(a[:stop] != bg)))
    result = []
    neighbors = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                      if r or c) if diagonal else FOUR
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells, color = [start], [], int(a[start])
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in neighbors:
                p = r + dr, c + dc
                if p in unseen and a[p] == color:
                    unseen.remove(p)
                    stack.append(p)
        result.append(dict(color=color, cells=tuple(sorted(cells)),
                           port=min(c for r, c in cells)))
    return result


def _parse_canonical(a, s, bg, divider):
    """Read the inventory/ports and complete two-railed rectangles from pixels."""
    h, w = a.shape
    _require(1 <= s < h - 4, "separator position")
    _require(a[0, 0] == bg and a[0, -1] == bg, "inventory corners")
    ports = tuple(map(int, np.flatnonzero(a[s + 1] != bg)))
    _require(ports and np.array_equal(a[s + 1], a[s + 2]), "two-cell launch ports")
    guides = set(map(int, a[s + 1, list(ports)]))
    _require(len(guides) == 1, "one signal color")
    guide = guides.pop()
    _require(len({bg, divider, guide}) == 3, "distinct structural colors")
    _require(not np.any(a[:s] == divider) and not np.any(a[:s] == guide), "inventory colors")
    _require(not np.any(a[s + 3:] == divider) and not np.any(a[s + 3:] == guide), "arena colors")
    comps = _components(a, s, bg)
    _require(comps and sorted(c['port'] for c in comps) == list(ports), "one port per component")
    bins, prior = [], {}
    for r in range(s + 3, h):
        current = {}
        c = 0
        while c < w:
            if a[r, c] == bg:
                c += 1
                continue
            left = c
            while c + 1 < w and a[r, c + 1] != bg:
                c += 1
            right = c
            rail, route = int(a[r, left]), int(a[r, left + 1]) if right > left else -1
            _require(right - left >= 4 and a[r, right] == rail and
                     route != rail and np.all(a[r, left + 1:right] == route), "complete tray row")
            key = left, right, route, rail
            if key in prior:
                j = prior[key]
                bins[j]['bottom'] = r
            else:
                j = len(bins)
                bins.append(dict(top=r, bottom=r, left=left, right=right, route=route, rail=rail))
            current[key] = j
            c += 1
        prior = current
    _require(bins, "no trays")
    rails = {}
    for b in bins:
        _require(not {b['route'], b['rail']} & {bg, divider, guide}, "tray roles")
        _require(b['route'] not in rails or rails[b['route']] == b['rail'], "one ink per route")
        rails[b['route']] = b['rail']
    _require(all(c['color'] in rails for c in comps), "component has no route")
    owner = np.full(a.shape, -1, dtype=np.int16)
    for j, b in enumerate(bins):
        owner[b['top']:b['bottom'] + 1, b['left']:b['right'] + 1] = j
    return dict(a=a, separator=s, background=bg, divider=divider, guide=guide,
                components=comps, bins=bins, owner=owner, ports=ports)


def _analyze(input_grid):
    grid = _array(input_grid)
    choices = []
    for turn in range(4):
        a = np.rot90(grid, turn)
        bg = int(a[0, 0])
        for s in range(1, a.shape[0] - 4):
            divider = int(a[s, 0])
            if divider == bg or not np.all(a[s] == divider):
                continue
            try:
                d = _parse_canonical(a, s, bg, divider)
            except AssertionError:
                continue
            d['turn'] = turn
            choices.append(d)
    _require(len(choices) == 1, "one visible orientation and role interpretation required")
    return choices[0]


def _walk(h, w, start, port, bins, owner, straight=False):
    """A's layer simulation; paths are sets and each struck span forks once."""
    active, trace, hits = {port}, set(), []
    seen_straight = set()
    for r in range(start, h):
        groups, following = {}, set()
        for c in sorted(active):
            j = int(owner[r, c])
            if j < 0:
                trace.add((r, c))
                following.add(c)
            else:
                groups.setdefault(j, []).append(c)
        for j, columns in sorted(groups.items()):
            b = bins[j]
            if straight:
                fresh = [c for c in columns if (j, c) not in seen_straight]
                seen_straight.update((j, c) for c in fresh)
                if fresh:
                    hits.append((j, tuple(fresh)))
                following.update(columns)
            else:
                hits.append((j, tuple(columns)))
                trace.update((r - 1, c) for c in range(max(0, b['left'] - 1), min(w, b['right'] + 2)))
                for c in (b['left'] - 1, b['right'] + 1):
                    if 0 <= c < w:
                        trace.add((r, c))
                        following.add(c)
        active = following
    return trace, hits


def _render(d, mutation=None):
    a, s, bg = d['a'], d['separator'], d['background']
    out, ink = a.copy(), {}
    out[:s] = bg
    comps = d['components']
    if mutation == 'group_color':
        grouped = {}
        for c in comps:
            grouped.setdefault(c['color'], []).extend(c['cells'])
        comps = [dict(color=color, cells=tuple(cells), port=min(c for r, c in cells))
                 for color, cells in sorted(grouped.items())]
    deliveries, all_hits, traces = [], [], set()
    for index, comp in enumerate(comps):
        trace, hits = _walk(*a.shape, s + 1, comp['port'], d['bins'], d['owner'],
                            straight=mutation == 'straight')
        traces.update(trace)
        all_hits.append(hits)
        used = False
        for j, columns in hits:
            b = d['bins'][j]
            if comp['color'] != b['route'] and mutation != 'ignore_color':
                continue
            if mutation == 'first_delivery' and used:
                continue
            used = True
            if mutation == 'one_per_tray':
                columns = columns[:1]
            for c in columns:
                dy = b['bottom'] - max(r for r, x in comp['cells'])
                dx = c - comp['port']
                cells = {(r + dy, x + dx) for r, x in comp['cells']}
                fits = all(b['top'] <= r <= b['bottom'] and b['left'] < x < b['right'] for r, x in cells)
                if mutation is None:
                    _require(fits, "complete stamp must fit its matching tray")
                else:
                    # Faulty hypotheses are clipped for diagnostics only.
                    cells = {(r, x) for r, x in cells if b['top'] <= r <= b['bottom'] and b['left'] < x < b['right']}
                for p in cells:
                    ink[p] = b['rail']
                deliveries.append((index, j, c, tuple(sorted(cells))))
    for p in traces:
        if a[p] == bg:
            out[p] = d['guide']
    for p, color in ink.items():
        out[p] = color
    return out, dict(ink=ink, deliveries=deliveries, hits=all_hits, trace=traces)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, component, ray and stamp from the input alone."""
    d = _analyze(input_grid)
    out, _ = _render(d)
    return np.rot90(out, -d['turn']).copy()


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _mask(rng, height, width):
    h, w = int(rng.integers(1, height + 1)), int(rng.integers(1, width + 1))
    kind = int(rng.integers(6))
    if kind == 0:
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 1:
        cells = {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}
    elif kind == 2 and h >= 3 and w >= 3:
        cells = {(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)}
    elif kind == 3:
        cells = {(h // 2, c) for c in range(w)} | {(r, w // 2) for r in range(h)}
    else:
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        goal = int(rng.integers(1, h * w + 1))
        while len(cells) < goal:
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in FOUR
                               if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _network(rng, s, w, ports, stages):
    """Compose tray geometry, without deciding what colors or masks it routes."""
    available = 30 - (s + 1) - 2 - 1
    heights = list(map(int, rng.integers(1, 4, size=stages)))
    gaps = list(map(int, rng.integers(1, 4, size=stages)))
    while sum(heights) + sum(gaps) > available:
        values = gaps if max(gaps) > 1 else heights
        values[values.index(max(values))] -= 1
    extra = int(rng.integers(available - sum(heights) - sum(gaps) + 1))
    for _ in range(extra):
        eligible = [i for i, n in enumerate(heights) if n < 5]
        if eligible:
            heights[int(rng.choice(eligible))] += 1
    rows, cursor = [], s + 3
    for height, gap in zip(heights, gaps):
        rows.append((cursor, height))
        cursor += height + gap
    h = cursor + int(rng.integers(0, min(3, 31 - cursor)))
    owner = np.full((h, w), -1, dtype=np.int16)
    bins = []
    for top, height in rows:
        active = set()
        for port in ports:
            # A prefix walk gives the active cells on this layer.
            trace, _ = _walk(top + 1, w, s + 1, port, bins, owner[:top + 1])
            active.update(c for r, c in trace if r == top)
        if not active:
            return None
        pivots = list(map(int, rng.permutation(sorted(active))))
        quota = 2 if rng.random() < .40 else 1
        intervals = []
        for pivot in pivots:
            if len(intervals) >= quota:
                break
            choices = [(l, r) for l in range(0, max(0, pivot))
                       for r in range(pivot + 1, w) if r - l >= 4 and
                       all(r + 1 < x or y + 1 < l for x, y in intervals)]
            if not choices:
                continue
            # Short and long spans coexist, and hit offsets are not centered.
            weights = np.array([1.0 / (r - l) for l, r in choices])
            left, right = choices[int(rng.choice(len(choices), p=weights / weights.sum()))]
            intervals.append((left, right))
        if not intervals:
            return None
        for left, right in sorted(intervals):
            j = len(bins)
            b = dict(top=top, bottom=top + height - 1, left=left, right=right)
            bins.append(b)
            owner[top:top + height, left:right + 1] = j
        # Optional inert or partially reached tray; it has no special palette.
        if rng.random() < .25:
            choices = [(l, r) for l in range(w - 4) for r in range(l + 4, w)
                       if all(r + 1 < x or y + 1 < l for x, y in intervals)]
            if choices:
                left, right = choices[int(rng.integers(len(choices)))]
                j = len(bins)
                bins.append(dict(top=top, bottom=top + height - 1, left=left, right=right))
                owner[top:top + height, left:right + 1] = j
    return h, bins, owner


def _assign_routes(rng, bins, hits, groups, minimums):
    k = max(groups) + 1
    eligible = []
    for j, b in enumerate(bins):
        options = []
        for g in range(k):
            ok = True
            for i, group in enumerate(groups):
                if group != g:
                    continue
                for bj, columns in hits[i]:
                    if bj == j:
                        mh, mw = minimums[i]
                        ok &= b['bottom'] - b['top'] + 1 >= mh and all(
                            b['left'] < c and c + mw <= b['right'] for c in columns)
            if ok:
                options.append(g)
        if not options:
            return None
        eligible.append(options)
    for _ in range(80):
        assignment = tuple(int(rng.choice(options)) for options in eligible)
        if all(any(assignment[j] == groups[i] for j, columns in incoming)
               for i, incoming in enumerate(hits)):
            return assignment
    return None


def _source(rng, s, w, ports, groups, caps, contact):
    stage = np.full((s, w), -1, dtype=np.int8)
    if contact:
        h = int(rng.integers(3, min(s, caps[0][0]) + 1))
        width = int(rng.integers(3, min(caps[0][1], w - ports[0] - 1) + 1))
        top = int(rng.integers(s - h + 1))
        left = ports[0]
        for r in range(h):
            for c in range(width):
                if r in (0, h - 1) or c in (0, width - 1):
                    stage[top + r, left + c] = groups[0]
        ih = int(rng.integers(1, min(h - 2, caps[1][0]) + 1))
        iw = int(rng.integers(1, min(width - 2, caps[1][1]) + 1))
        stage[top + 1:top + 1 + ih, left + 1:left + 1 + iw] = groups[1]
        return stage
    bottoms = {}
    for i, (port, group) in enumerate(zip(ports, groups)):
        maxw = min(caps[i][1], w - port - 1)
        if i + 1 < len(ports):
            maxw = min(maxw, ports[i + 1] - port - 1)
        if maxw < 1:
            return None
        mh = min(s, caps[i][0])
        for _ in range(20):
            mask = _mask(rng, mh, maxw)
            height = 1 + max(r for r, c in mask)
            top = int(rng.integers(s - height + 1))
            bottom = top + height - 1
            if group in bottoms and bottom in bottoms[group]:
                continue
            for r, c in mask:
                stage[top + r, port + c] = group
            bottoms.setdefault(group, set()).add(bottom)
            break
        else:
            return None
    return stage


def _candidate(rng, lesson, test, palette, turn):
    s, w = int(rng.integers(4, 10)), int(rng.integers(17, 31))
    contact = lesson == 'contact' or lesson is None and rng.random() < .25
    if contact:
        p = int(rng.integers(3, w - 8))
        ports, groups, minimums = [p, p + 1], [0, 1], [(3, 3), (1, 1)]
    else:
        n = 3 if lesson == 'independent' else int(rng.integers(2, 5))
        ports = sorted(map(int, rng.choice(np.arange(1, w - 2), n, replace=False)))
        if any(b - a < 2 for a, b in zip(ports, ports[1:])):
            return None
        groups = [0, 0, 1] if lesson == 'independent' else list(range(min(n, 3))) + [0] * max(0, n - 3)
        rng.shuffle(groups)
        minimums = [(1, 1)] * n
    stages = 4 if test else (3 if lesson else int(rng.integers(2, 4)))
    net = _network(rng, s, w, ports, stages)
    if net is None:
        return None
    h, bins, owner = net
    hits = [_walk(h, w, s + 1, p, bins, owner)[1] for p in ports]
    assignment = _assign_routes(rng, bins, hits, groups, minimums)
    if assignment is None:
        return None
    caps = []
    for i, incoming in enumerate(hits):
        matching = [(bins[j], columns) for j, columns in incoming if assignment[j] == groups[i]]
        caps.append((min(b['bottom'] - b['top'] + 1 for b, cs in matching),
                     min(b['right'] - c for b, cs in matching for c in cs)))
    stage = _source(rng, s, w, ports, groups, caps, contact)
    if stage is None:
        return None
    bg, divider, guide = palette[:3]
    k = max(groups) + 1
    routes = palette[3:3 + k]
    extra = palette[3 + k:]
    rails = [int(rng.choice([c for c in routes if c != route] + extra)) for route in routes]
    a = np.full((h, w), bg, dtype=np.int8)
    a[s] = divider
    for g, route in enumerate(routes):
        a[:s][stage == g] = route
    a[s + 1:s + 3, ports] = guide
    for j, b in enumerate(bins):
        g = assignment[j]
        b.update(route=routes[g], rail=rails[g])
        a[b['top']:b['bottom'] + 1, b['left']:b['right'] + 1] = routes[g]
        a[b['top']:b['bottom'] + 1, [b['left'], b['right']]] = rails[g]
    try:
        d = _parse_canonical(a, s, bg, divider)
        out, stats = _render(d)
        if len(_components(a, s, bg, True)) != len(d['components']):
            return None
        # Both colors and the splitting geometry must change delivered ink,
        # not merely add visually different signal traces.
        if stats['ink'] == _render(d, 'straight')[1]['ink']:
            return None
        if stats['ink'] == _render(d, 'ignore_color')[1]['ink']:
            return None
        if lesson == 'independent' and stats['ink'] == _render(d, 'group_color')[1]['ink']:
            return None
        if lesson == 'independent' and np.array_equal(out, _render(d, 'first_delivery')[0]):
            return None
        _require(all(any(row[0] == i for row in stats['deliveries']) for i in range(len(d['components']))),
                 "each template must have a delivery")
    except AssertionError:
        return None
    pair = dict(input=np.rot90(a, turn).copy(), output=np.rot90(out, turn).copy())
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    fixed_palette = list(map(int, rng.permutation(10)))
    fixed_turn = int(rng.integers(4))
    mixed = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    admitted = None
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            lesson = ('contact' if i == 0 else 'independent' if i == 1 else None) if split == 'train' else None
            for attempt in range(800):
                palette = list(map(int, rng.permutation(10))) if mixed else fixed_palette
                turn = int(rng.integers(4)) if mixed else fixed_turn
                pair = _candidate(rng, lesson, split == 'test', palette, turn)
                if pair is None or _key(pair['input']) in seen:
                    continue
                if split == 'test':
                    d = _analyze(pair['input'])
                    if any(not np.array_equal(_render(d, m)[0], _render(d)[0]) for m in admitted):
                        continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError('bounded scene construction exhausted')
        if split == 'train':
            admitted = [m for m in ('one_per_tray', 'group_color', 'first_delivery')
                        if all(np.array_equal(_render(_analyze(p['input']), m)[0],
                                              np.rot90(p['output'], _analyze(p['input'])['turn']))
                               for p in puzzle['train'])]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    seen, train_data, test_data = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            a, out = _array(pair['input']), _array(pair['output'])
            _require(a.shape == out.shape and _key(a) not in seen, 'shape or duplicate input')
            seen.add(_key(a))
            d = _analyze(a)
            expected, stats = _render(d)
            canonical_out = np.rot90(out, d['turn'])
            _require(np.array_equal(canonical_out, expected), 'incorrect output')
            s, bg = d['separator'], d['background']
            _require(np.all(canonical_out[:s] == bg), 'inventory not cleared')
            _require(np.array_equal(canonical_out[s], d['a'][s]), 'separator changed')
            _require(len(_components(d['a'], s, bg, True)) == len(d['components']), 'diagonal grouping ambiguity')
            _require(all(any(t[0] == i for t in stats['deliveries']) for i in range(len(d['components']))), 'undelivered template')
            for b in d['bins']:
                for c in (b['left'], b['right']):
                    _require(np.all(canonical_out[b['top']:b['bottom'] + 1, c] == b['rail']), 'rail overwritten')
                for r in range(b['top'], b['bottom'] + 1):
                    for c in range(b['left'] + 1, b['right']):
                        _require(canonical_out[r, c] == stats['ink'].get((r, c), b['route']), 'tray interior changed without a stamp')
            _require(stats['ink'] != _render(d, 'straight')[1]['ink'], 'splitting has no delivery consequence')
            _require(stats['ink'] != _render(d, 'ignore_color')[1]['ink'], 'routing has no delivery consequence')
            (train_data if split == 'train' else test_data).append((d, expected, stats))
    contact = False
    for d, out, st in train_data:
        for comp in d['components']:
            for r, c in comp['cells']:
                for dr, dc in FOUR:
                    if 0 <= r + dr < d['separator'] and 0 <= c + dc < d['a'].shape[1]:
                        contact |= int(d['a'][r + dr, c + dc]) not in (d['background'], comp['color'])
    _require(contact, 'no multicolor separation lesson')
    _require(any(st['ink'] != _render(d, 'group_color')[1]['ink'] for d, out, st in train_data), 'no independent component lesson')
    _require(any(x['color'] == y['color'] and
                 max(r for r, c in x['cells']) != max(r for r, c in y['cells'])
                 for d, out, st in train_data for i, x in enumerate(d['components'])
                 for y in d['components'][i + 1:]), 'no unequal-depth same-color witness')
    _require(any(not np.array_equal(out, _render(d, 'first_delivery')[0]) for d, out, st in train_data), 'no repeated broadcast lesson')
    _require(max(len({b['top'] for b in d['bins']}) for d, _, _ in train_data) <= 3 and
             all(len({b['top'] for b in d['bins']}) == 4 for d, _, _ in test_data), 'depth shift')
    for m in ('one_per_tray', 'group_color', 'first_delivery'):
        if all(np.array_equal(out, _render(d, m)[0]) for d, out, st in train_data):
            _require(all(np.array_equal(out, _render(d, m)[0]) for d, out, st in test_data), 'unresolved training shortcut changes test')


def measure_pair(pair):
    d = _analyze(pair['input'])
    out, st = _render(d)
    comps, bins = d['components'], d['bins']
    masks = [tuple((r - min(y for y, x in c['cells']), x - c['port']) for r, x in c['cells']) for c in comps]
    all_reached = {j for incoming in st['hits'] for j, columns in incoming}
    return dict(shape=tuple(np.asarray(pair['input']).shape), turn=d['turn'],
                palette=(d['background'], d['divider'], d['guide']),
                components=len(comps), masks=masks, trays=len(bins),
                stages=len({b['top'] for b in bins}), ports=tuple(d['ports']),
                depths=tuple(b['bottom'] - b['top'] + 1 for b in bins),
                widths=tuple(b['right'] - b['left'] + 1 for b in bins),
                inert=len(bins) - len(all_reached),
                clipped=sum(bins[j]['left'] == 0 or bins[j]['right'] == d['a'].shape[1] - 1
                            for incoming in st['hits'] for j, cols in incoming),
                multi_hit=sum(len(cols) > 1 for incoming in st['hits'] for j, cols in incoming),
                shared_rails=len({b['rail'] for b in bins}) < len({b['route'] for b in bins}),
                reuse=any(b['rail'] in {c['color'] for c in comps} for b in bins),
                deliveries=len(st['deliveries']), ink_cells=len(st['ink']),
                repeated=any(sum(t[0] == i for t in st['deliveries']) > 1 for i in range(len(comps))))


def self_test():
    """Bounded, in-memory 42-seed measurements and explicit negative checks."""
    # A hand-calculated two-level network. L and adjacent singleton are separate
    # templates; another route-1 singleton is independently bottom-aligned.
    # The right-hand second-level tray receives overlapping L/singleton copies.
    a = np.zeros((16, 15), dtype=np.int8)
    a[0, 5] = a[1, 5] = a[1, 6] = 1
    a[2, 6] = 2
    a[3, 8] = 1
    a[4] = 9
    a[5:7, [5, 6, 8]] = 8
    a[7:10, 3:11] = 1
    a[7:10, [3, 10]] = 4
    a[11:14, 0:6] = 2
    a[11:14, [0, 5]] = 6
    a[11:14, 8:15] = 1
    a[11:14, [8, 14]] = 4
    expected = a.copy()
    expected[:4] = 0
    expected[6, 2:12] = 8
    expected[7:10, [2, 11]] = 8
    expected[10] = 8
    expected[11:16, [6, 7]] = 8
    for p in ((8, 5), (9, 5), (9, 6), (9, 8), (12, 11), (13, 11), (13, 12)):
        expected[p] = 4
    expected[13, 2] = 6
    for turn in range(4):
        mapping = (np.arange(10, dtype=np.int8) + 3 * turn) % 10
        inp = np.rot90(mapping[a], turn).copy()
        saved = inp.copy()
        _require(np.array_equal(solve(inp.tolist()), np.rot90(mapping[expected], turn)), 'hand-calculated fork/stamp regression')
        _require(np.array_equal(inp, saved), 'input was mutated')
    records, signatures, masks, shapes = [], set(), set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                _require(np.array_equal(solve(pair['input']), pair['output']), 'solve regression')
                digest.update(bytes(pair['input'].shape) + pair['input'].tobytes())
                m = measure_pair(pair)
                records.append((split, m))
                masks.update(m['masks'])
                shapes.add(m['shape'])
        signatures.add(digest.hexdigest())
    _require(len(signatures) == 42 and len(masks) >= 30 and len(shapes) >= 25, 'variation collapsed')
    for split in ('train', 'test'):
        ms = [m for s, m in records if s == split]
        _require({m['turn'] for m in ms} == set(range(4)), 'orientation variation')
        _require(len({m['components'] for m in ms}) >= 2, 'component count variation')
        _require(len({m['trays'] for m in ms}) >= 3, 'tray count variation')
        _require(len({n for m in ms for n in m['widths']}) >= 8, 'tray span variation')
        _require(any(m['multi_hit'] for m in ms) and any(m['clipped'] for m in ms), 'branch geometry variation')
    a, b = generate_puzzle(91, 2, 1), generate_puzzle(91, 2, 1)
    _require(all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ('input', 'output')), 'nondeterminism')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(17, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'count request')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    p = generate_puzzle(32, 2, 1)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(signatures),
                shapes=len(shapes), component_masks=len(masks),
                independent_solver=False)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
