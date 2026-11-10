"""Sliding color siphons: input-only geometry, original-color transfers.

Only NumPy and the standard library are used. Generation and self_test perform
no disk I/O. No parent code, construction metadata, or labels enter solve().
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

import numpy as np

DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _array(value):
    grid = np.asarray(value)
    _need(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "invalid grid dimensions")
    _need(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((0 <= grid) & (grid <= 9))), "invalid grid color")
    return grid.astype(np.int8, copy=True)


def _inside(point, shape):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def _add(point, direction, amount=1):
    return point[0] + amount * direction[0], point[1] + amount * direction[1]


def _footprint(center, direction):
    side = (-direction[1], direction[0])
    return {center, _add(center, side), _add(center, side, -1), _add(center, direction)}


def _halo(points):
    return points | {_add(p, d) for p in points for d in DIRECTIONS}


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {start}
        while stack:
            p = stack.pop()
            for d in DIRECTIONS:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _ray(labels, center, direction):
    point = _add(center, direction)
    while _inside(point, labels.shape):
        if labels[point] >= 0:
            return int(labels[point]), point
        point = _add(point, direction)
    return None


def _decode(input_grid):
    grid = _array(input_grid)
    h, w = grid.shape
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    _need(counts.max() > grid.size / 2, "background is not a strict majority")
    anchors, regions, raw_probes = defaultdict(set), [], []
    for cells in _components(grid != background):
        point = min(cells)
        if len(cells) == 1:
            r, c = point
            _need((r in (0, h - 1)) != (c in (0, w - 1)), "pin must be on one noncorner edge")
            anchors[int(grid[point])].add(point)
        elif len(cells) == 4:
            centers = [p for p in cells if sum(_add(p, d) in cells for d in DIRECTIONS) == 3]
            _need(len(centers) == 1, "instrument must be an exact four-cell T")
            center = centers[0]
            direction = tuple(sum(p[k] - center[k] for p in cells) for k in (0, 1))
            _need(direction in DIRECTIONS and cells == _footprint(center, direction), "invalid T geometry")
            ink = {int(grid[p]) for p in cells if p != center}
            _need(len(ink) == 1 and int(grid[center]) not in ink, "T needs a contrasting registration junction")
            raw_probes.append(dict(cells=cells, center=center, direction=direction,
                                   marker=int(grid[center]), ink=ink.pop()))
        else:
            _need(len(cells) >= 5, "territories have at least five cells")
            colors = {int(grid[p]) for p in cells}
            _need(len(colors) == 1, "territory is not monochromatic")
            regions.append(dict(cells=cells, color=colors.pop()))
    _need(regions and raw_probes and anchors, "missing territories, instruments, or pins")
    systems = {}
    for color, points in anchors.items():
        horizontal = len({r for r, _ in points}) == 1 and all(c in (0, w - 1) for _, c in points)
        vertical = len({c for _, c in points}) == 1 and all(r in (0, h - 1) for r, _ in points)
        _need(horizontal != vertical, "pins do not define one row or column")
        axis = 0 if horizontal else 1
        systems[color] = dict(axis=axis, guide=next(iter(points))[axis], anchors=points)
    labels = np.full(grid.shape, -1, dtype=np.int16)
    base = np.full_like(grid, background)
    for i, region in enumerate(regions):
        for p in region['cells']:
            labels[p] = i
            base[p] = region['color']
    for color, points in anchors.items():
        for p in points:
            base[p] = color
    probes, assignments, used_virtual = [], {}, set()
    for raw in raw_probes:
        _need(raw['marker'] in systems, "junction has no matching pin")
        system = systems[raw['marker']]
        axis, direction = system['axis'], raw['direction']
        _need(direction[axis] == 0, "T direction must run parallel to its guide")
        projected = list(raw['center'])
        projected[axis] = system['guide']
        projected = tuple(projected)
        footprint = _footprint(projected, direction)
        _need(all(_inside(p, grid.shape) and base[p] == background for p in footprint),
              "projected instrument must fit in empty space")
        _need(not footprint & used_virtual, "projected instruments overlap")
        used_virtual.update(footprint)
        rear = _ray(labels, projected, (-direction[0], -direction[1]))
        front = _ray(labels, projected, direction)
        _need(rear is not None and front is not None and rear[0] != front[0],
              "instrument must see different rear and front territories")
        source, target = rear[0], front[0]
        payload = regions[source]['color']
        _need(target not in assignments or assignments[target] == payload, "conflicting incoming colors")
        assignments[target] = payload
        probes.append(dict(raw, axis=axis, projected=projected, virtual=footprint,
                           delta=system['guide'] - raw['center'][axis],
                           source=source, target=target, rear=rear[1], front=front[1], payload=payload))
    _need(set(systems) == {p['marker'] for p in probes}, "unused pin identity")
    return dict(grid=grid, background=background, systems=systems, regions=regions,
                labels=labels, base=base, probes=probes, assignments=assignments)


def _paint(info, assignments):
    result = info['base'].copy()
    for target, color in assignments.items():
        result[info['labels'] == target] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Project T instruments, ray-match territories, and transfer original colors."""
    info = _decode(input_grid)
    return _paint(info, info['assignments'])


def _alternative(info, mode):
    """Explicit wrong rules used to check witnesses, never to choose an answer."""
    if mode == 'no_alignment':
        assignments = {}
        for p in info['probes']:
            d = p['direction']
            rear = _ray(info['labels'], p['center'], (-d[0], -d[1]))
            front = _ray(info['labels'], p['center'], d)
            if rear is not None and front is not None and rear[0] != front[0]:
                assignments[front[0]] = info['regions'][rear[0]]['color']
        return _paint(info, assignments)
    if mode == 'propagate_again':
        assignments = {}
        for p in info['probes']:
            assignments[p['target']] = info['assignments'].get(p['source'], p['payload'])
        return _paint(info, assignments)
    if mode == 'color_wide':
        assignments = {}
        for target, color in info['assignments'].items():
            original = info['regions'][target]['color']
            for i, region in enumerate(info['regions']):
                if region['color'] == original:
                    assignments[i] = color
        return _paint(info, assignments)
    if mode == 'tip_pixel_only':
        result = info['base'].copy()
        for p in info['probes']:
            result[p['front']] = p['payload']
        return result
    if mode in ('junction_payload', 'ink_payload'):
        key = 'marker' if mode == 'junction_payload' else 'ink'
        return _paint(info, {p['target']: p[key] for p in info['probes']})
    if mode == 'ignore_tip':
        assignments = {}
        for p in info['probes']:
            d = (0, 1) if p['axis'] == 0 else (1, 0)
            rear = _ray(info['labels'], p['projected'], (-d[0], -d[1]))
            front = _ray(info['labels'], p['projected'], d)
            assignments[front[0]] = info['regions'][rear[0]]['color']
        return _paint(info, assignments)
    raise ValueError('unknown counterfactual')


def _features(info):
    answer = _paint(info, info['assignments'])
    regions = info['regions']
    def differs(mode):
        return not np.array_equal(answer, _alternative(info, mode))
    return dict(shape=list(info['grid'].shape), territories=len(regions),
                probes=len(info['probes']), guides=len(info['systems']),
                axes=sorted({s['axis'] for s in info['systems'].values()}),
                directions=sorted({DIRECTIONS.index(p['direction']) for p in info['probes']}),
                offsets=[abs(p['delta']) for p in info['probes']],
                background=info['background'],
                bilateral=sum(len(s['anchors']) == 2 for s in info['systems'].values()),
                areas=[len(r['cells']) for r in regions],
                nonrectangular=sum(len(r['cells']) != (max(p[0] for p in r['cells']) - min(p[0] for p in r['cells']) + 1)
                                   * (max(p[1] for p in r['cells']) - min(p[1] for p in r['cells']) + 1) for r in regions),
                chain=differs('propagate_again'), locality=differs('color_wide'),
                alignment=differs('no_alignment'), whole_fill=differs('tip_pixel_only'),
                direction=differs('ignore_tip'), junction_payload=differs('junction_payload'),
                ink_payload=differs('ink_payload'),
                fanout=any(len({p['target'] for p in info['probes'] if p['source'] == i}) > 1
                           for i in range(len(regions))),
                cycle=any(a['source'] == b['target'] and a['target'] == b['source']
                          for a in info['probes'] for b in info['probes']),
                aligned=sum(p['delta'] == 0 for p in info['probes']))


def _shape(rng):
    h, w = int(rng.integers(2, 7)), int(rng.integers(2, 7))
    kind = int(rng.integers(6))
    if h * w < 5:
        return None
    if kind == 0:
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 1:
        h, w = max(3, h), max(3, w)
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1)}
    elif kind == 2:
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r >= h // 2 or c < max(1, w // 2)}
    elif kind == 3:
        widths = sorted(map(int, rng.integers(1, w + 1, size=h)))
        cells = {(r, c) for r in range(h) for c in range(widths[r])}
    else:
        cells = {(h // 2, w // 2)}
        for _ in range(int(rng.integers(5, min(h * w, 21) + 1)) - 1):
            choices = sorted({_add(p, d) for p in cells for d in DIRECTIONS
                              if _inside(_add(p, d), (h, w))} - cells)
            if not choices:
                break
            cells.add(choices[int(rng.integers(len(choices)))])
    if len(cells) < 5:
        return None
    turns = int(rng.integers(4))
    for _ in range(turns):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _substrate(rng, test):
    h, w = map(int, rng.integers(23 if test else 18, 31 if test else 29, size=2))
    n = int(rng.integers(7, 9) if test else rng.integers(4, 7))
    labels = np.full((h, w), -1, dtype=np.int16)
    occupied, regions = set(), []
    for i in range(n):
        for _ in range(120):
            shape = _shape(rng)
            if shape is None:
                continue
            sh, sw = 1 + max(r for r, _ in shape), 1 + max(c for _, c in shape)
            if sh + 2 > h or sw + 2 > w:
                continue
            r0, c0 = int(rng.integers(1, h - sh)), int(rng.integers(1, w - sw))
            cells = {(r + r0, c + c0) for r, c in shape}
            if cells & _halo(occupied):
                continue
            regions.append(cells)
            occupied.update(cells)
            for p in cells:
                labels[p] = i
            break
        else:
            return None
    if len(occupied) + 32 >= labels.size / 2:
        return None
    return labels, regions


def _anchor_options(labels, axis, guide):
    h, w = labels.shape
    points = ((guide, 0), (guide, w - 1)) if axis == 0 else ((0, guide), (h - 1, guide))
    return [p for p in points if all(not _inside(q, labels.shape) or labels[q] < 0 for q in _halo({p}))]


def _candidates(labels, axes):
    candidates = []
    for axis in axes:
        for guide in range(1, labels.shape[axis] - 1):
            if not _anchor_options(labels, axis, guide):
                continue
            line = labels[guide, :] if axis == 0 else labels[:, guide]
            before, after = [], [None] * len(line)
            last = None
            for i, value in enumerate(line):
                before.append(last)
                if value >= 0:
                    last = int(value)
            last = None
            for i in range(len(line) - 1, -1, -1):
                after[i] = last
                if line[i] >= 0:
                    last = int(line[i])
            for along in range(2, len(line) - 2):
                if line[along] >= 0 or before[along] is None or after[along] is None or before[along] == after[along]:
                    continue
                center = (guide, along) if axis == 0 else (along, guide)
                for sign in (-1, 1):
                    direction = (0, sign) if axis == 0 else (sign, 0)
                    footprint = _footprint(center, direction)
                    if not all(labels[p] < 0 for p in footprint):
                        continue
                    source, target = (before[along], after[along]) if sign == 1 else (after[along], before[along])
                    candidates.append(dict(axis=axis, guide=guide, projected=center,
                                           direction=direction, source=source, target=target, virtual=footprint))
    return candidates


def _plan(rng, candidates, n, desired, chain, mixed, repeat, reverse):
    if not candidates:
        return None
    order = [candidates[int(i)] for i in rng.permutation(len(candidates))]
    selected = []
    if chain:
        by_source = defaultdict(list)
        for c in order:
            by_source[c['source']].append(c)
        for first in order:
            for second in by_source[first['target']]:
                if second['target'] == first['source'] or (mixed and first['axis'] == second['axis']):
                    continue
                if first['virtual'] & second['virtual']:
                    continue
                selected = [first, second]
                break
            if selected:
                break
        if not selected:
            return None
    else:
        initial = [c for c in order if not reverse or sum(c['direction']) < 0]
        if not initial:
            return None
        selected = [initial[0]]
    used_ids = {c[k] for c in selected for k in ('source', 'target')}
    unused = [i for i in range(n) if i not in used_ids]
    protected = int(rng.choice(unused)) if repeat and unused else None
    if repeat and protected is None:
        return None
    occupied = set().union(*(c['virtual'] for c in selected))
    systems = {(c['axis'], c['guide']) for c in selected}
    assignments = {c['target']: c['source'] for c in selected}
    for c in order:
        if len(selected) >= desired:
            break
        key = c['axis'], c['guide']
        if protected in (c['source'], c['target']) or occupied & c['virtual']:
            continue
        if key not in systems and len(systems) >= 4:
            continue
        if c['target'] in assignments:
            if assignments[c['target']] != c['source'] or rng.random() > .12:
                continue
        selected.append(c)
        occupied.update(c['virtual'])
        systems.add(key)
        assignments[c['target']] = c['source']
    if len(selected) != desired:
        return None
    return selected, protected


def _build_pair(rng, axes, palette, test, witness, reverse=False):
    substrate = _substrate(rng, test)
    if substrate is None:
        return None
    labels, regions = substrate
    candidates = _candidates(labels, axes)
    desired = int(rng.integers(4, 7) if test else rng.integers(2 if witness else 1, 4))
    chain = witness or bool(test and rng.random() < .7)
    repeat = witness or bool(rng.random() < .55)
    plan = _plan(rng, candidates, len(regions), desired, chain,
                 witness and len(axes) == 2, repeat, reverse)
    if plan is None:
        return None
    probes, protected = plan
    palette = list(map(int, rng.permutation(10))) if palette is None else list(palette)
    background = palette[0]
    keys = sorted({(c['axis'], c['guide']) for c in probes})
    markers = dict(zip(keys, palette[1:1 + len(keys)]))
    region_colors = list(map(int, rng.choice(palette[1:], size=len(regions), replace=False)))
    if protected is not None:
        region_colors[protected] = region_colors[probes[0]['target']]
    inp = np.full(labels.shape, background, dtype=np.int8)
    out = inp.copy()
    for i, cells in enumerate(regions):
        for p in cells:
            inp[p] = out[p] = region_colors[i]
    occupied = set().union(*regions)
    for key in keys:
        options = _anchor_options(labels, *key)
        rng.shuffle(options)
        options = [p for p in options if p not in _halo(occupied)]
        if not options:
            return None
        chosen = options if len(options) == 2 and rng.random() < .35 else options[:1]
        for p in chosen:
            inp[p] = out[p] = markers[key]
            occupied.add(p)
    # Virtual instrument positions were planned on the original substrate.
    # Display positions can lie elsewhere, including against an input edge.
    for c in [probes[int(i)] for i in rng.permutation(len(probes))]:
        axis, guide = c['axis'], c['guide']
        choices = []
        blocked = _halo(occupied)
        for coordinate in range(1, inp.shape[axis] - 1):
            delta = guide - coordinate
            if test and abs(delta) < 5:
                continue
            if not test and abs(delta) > 7:
                continue
            center = list(c['projected'])
            center[axis] = coordinate
            center = tuple(center)
            cells = _footprint(center, c['direction'])
            if not cells & blocked:
                choices.append((center, cells))
        if not choices:
            return None
        center, cells = choices[int(rng.integers(len(choices)))]
        marker = markers[axis, guide]
        ink = int(rng.choice([v for v in palette if v not in (background, marker)]))
        for p in cells:
            inp[p] = marker if p == center else ink
        occupied.update(cells)
    assignments = {c['target']: region_colors[c['source']] for c in probes}
    for target, color in assignments.items():
        out[labels == target] = color
    try:
        info = _decode(inp)
    except AssertionError:
        return None
    if not np.array_equal(_paint(info, info['assignments']), out):
        return None
    features = _features(info)
    if not features['alignment'] or not features['whole_fill']:
        return None
    if witness and not (features['chain'] and features['locality']):
        return None
    if witness and not (features['junction_payload'] and features['ink_payload']):
        return None
    if reverse and not features['direction']:
        return None
    return dict(input=inp, output=out)


def _signature(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """A seed reproduces the full puzzle; all explicit legal counts are supported."""
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    for value, lo, hi, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if not _integer(value) or not lo <= value <= hi:
            raise ValueError(name + ' is outside its legal integer range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(int(seed))
    mode = int(rng.choice(3, p=[.2, .2, .6]))
    axes = (mode,) if mode < 2 else (0, 1)
    palette = tuple(map(int, rng.permutation(10))) if rng.random() < .25 else None
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            for _ in range(240):
                pair = _build_pair(rng, axes, palette, split == 'test',
                                   split == 'train' and index == 0, split == 'train' and index == 1)
                if pair is not None and _signature(pair['input']) not in seen:
                    seen.add(_signature(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded scene construction exhausted')
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Reconstruct semantic invariants from arrays, without generation records."""
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'incorrect puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'illegal example counts')
    seen, stats = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'incorrect pair keys')
            inp, out = _array(pair['input']), _array(pair['output'])
            key = _signature(inp)
            _need(key not in seen, 'duplicate input')
            seen.add(key)
            _need(inp.shape == out.shape, 'canvas dimensions must be preserved')
            info = _decode(inp)
            _need(np.array_equal(_paint(info, info['assignments']), out), 'wrong directed territory transfer')
            for region_id, region in enumerate(info['regions']):
                expected = info['assignments'].get(region_id, region['color'])
                _need(all(out[p] == expected for p in region['cells']), 'territory was only partly painted')
            _need(all(out[p] == color for color, system in info['systems'].items() for p in system['anchors']),
                  'a boundary pin moved or changed color')
            probe_cells = set().union(*(p['cells'] for p in info['probes']))
            _need(all(out[p] == info['background'] for p in probe_cells), 'an instrument was not erased')
            f = _features(info)
            _need(f['alignment'] and f['whole_fill'], 'parent contribution lacks a counterfactual witness')
            stats[split].append(f)
    _need(any(f['chain'] and f['locality'] for f in stats['train']),
          'training must jointly distinguish simultaneous and component-local transfer')
    for feature in ('direction', 'junction_payload', 'ink_payload'):
        _need(any(f[feature] for f in stats['train']), 'missing training contrast: ' + feature)
    train_axes = {axis for f in stats['train'] for axis in f['axes']}
    test_axes = {axis for f in stats['test'] for axis in f['axes']}
    _need(test_axes <= train_axes, 'test guide axis was not demonstrated')
    if len(test_axes) == 2:
        _need(any(len(f['axes']) == 2 for f in stats['train']), 'mixed guide composition was not demonstrated')


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds; no puzzle corpus, logs, or traces are written."""
    # A separately specified four-territory elbow checks a full expected array.
    # It is deliberately built without _substrate, _plan, _decode, or _paint.
    manual_in = np.zeros((19, 19), dtype=np.int8)
    manual_out = np.zeros_like(manual_in)
    for rows, cols, before, after in ((slice(2, 5), slice(2, 5), 2, 2),
                                     (slice(2, 5), slice(13, 16), 3, 2),
                                     (slice(13, 16), slice(13, 16), 4, 3),
                                     (slice(13, 16), slice(2, 5), 3, 3)):
        manual_in[rows, cols], manual_out[rows, cols] = before, after
    manual_in[3, 0] = manual_out[3, 0] = 8
    manual_in[0, 14] = manual_out[0, 14] = 9
    manual_in[8, 8] = 8
    manual_in[7, 8] = manual_in[9, 8] = manual_in[8, 9] = 6
    manual_in[8, 5] = 9
    manual_in[8, 4] = manual_in[8, 6] = manual_in[9, 5] = 7
    _need(np.array_equal(solve(manual_in), manual_out), 'hand-specified elbow regression failed')
    manual = _decode(manual_in)
    for mode in ('no_alignment', 'propagate_again', 'color_wide', 'tip_pixel_only',
                 'junction_payload', 'ink_payload'):
        _need(not np.array_equal(_alternative(manual, mode), manual_out), 'manual contrast failed: ' + mode)
    dimensions, structures, puzzles, region_shapes = set(), set(), set(), set()
    counter, axes, directions, backgrounds = Counter(), set(), set(), set()
    offsets = {'train': [], 'test': []}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_digest(puzzle))
        if seed in (0, 41):
            _need(_digest(puzzle) == _digest(generate_puzzle(seed)), 'nondeterministic generation')
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                _need(np.array_equal(solve(pair['input']), pair['output']), 'input-only prediction failed')
                f = _features(info)
                dimensions.add(tuple(f['shape']))
                axes.update(f['axes'])
                directions.update(f['directions'])
                backgrounds.add(f['background'])
                offsets[split].extend(f['offsets'])
                for feature in ('chain', 'locality', 'alignment', 'whole_fill', 'direction',
                                'junction_payload', 'ink_payload', 'fanout', 'cycle', 'aligned'):
                    counter[feature] += bool(f[feature])
                counter['nonrectangular'] += f['nonrectangular']
                counter['bilateral'] += f['bilateral']
                counter['pairs'] += 1
                counter['mixed'] += len(f['axes']) == 2
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in pair['input'].ravel())
                structures.add(hashlib.sha256(bytes(pair['input'].shape) + normalized).hexdigest())
                for region in info['regions']:
                    cells = region['cells']
                    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
                    region_shapes.add(tuple(sorted((r - r0, c - c0) for r, c in cells)))
    _need(len(puzzles) == 42 and len(structures) >= 180 and len(dimensions) >= 50, 'collapsed scene variation')
    _need(len(region_shapes) >= 80 and counter['nonrectangular'] >= 100, 'collapsed territory geometry')
    _need(axes == {0, 1} and directions == {0, 1, 2, 3}, 'missing geometric directions')
    _need(backgrounds == set(range(10)), 'palette roles failed to vary')
    _need(counter['chain'] >= 42 and counter['locality'] >= 42 and counter['mixed'] >= 10, 'missing rule witnesses')
    _need(counter['fanout'] >= 3 and counter['cycle'] >= 3 and counter['bilateral'] >= 10, 'collapsed transfer graphs')
    _need(min(offsets['test']) >= 5 and max(offsets['train']) <= 7, 'offset curriculum broken')
    puzzle = generate_puzzle(0, 2, 1)
    # Geometry and semantic roles commute with rotations and arbitrary recoloring.
    rng = np.random.default_rng(20261023)
    for turns in range(4):
        changed = dict(train=[], test=[])
        for split in changed:
            for pair in puzzle[split]:
                colors = rng.permutation(10)
                q = {role: colors[np.rot90(grid, turns)].astype(np.int8) for role, grid in pair.items()}
                _need(np.array_equal(solve(q['input']), q['output']), 'rotation/palette metamorphism failed')
                changed[split].append(q)
        validate_generated_puzzle(changed)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output was accepted')
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            q = generate_puzzle(1000 + 10 * a + b, a, b)
            _need(len(q['train']) == a and len(q['test']) == b, 'explicit count ignored')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, pairs=counter['pairs'], distinct_puzzles=len(puzzles),
                distinct_inputs_up_to_recoloring=len(structures), dimensions=len(dimensions),
                territory_shapes=len(region_shapes), features=dict(counter),
                note='Generator-owned checks, not independent semantic certification.')
