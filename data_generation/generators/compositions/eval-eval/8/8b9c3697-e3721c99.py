"""Topology-authorized docking of T bridges into three-pronged receivers.

Only pixels are needed by solve. Black is the fixed swept-trail color; every
other color role is local to the input. No parent module is imported.
"""
from __future__ import annotations

import json
import numpy as np

DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
TRAIL = 0


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        cells, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _mask(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    h = max(r for r, c in cells) - r0 + 1
    w = max(c for r, c in cells) - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r-r0, c-c0] = True
    return mask, (r0, c0)


def _holes(cells):
    mask, _ = _mask(cells)
    return len(_components(~np.pad(mask, 1))) - 1


def _paint(grid, cells, color):
    for p in cells:
        grid[p] = color


def _local(cells, outward):
    dr, dc = outward
    perp = -dc, dr
    kmin = min(r*dr+c*dc for r, c in cells)
    qmin = min(r*perp[0]+c*perp[1] for r, c in cells)
    local = {(r*dr+c*dc-kmin, r*perp[0]+c*perp[1]-qmin)
             for r, c in cells}
    return local, perp, kmin, qmin


def _runs(values):
    runs, q = [], 0
    while q < len(values):
        if not values[q]:
            q += 1
            continue
        start, value = q, values[q]
        while q < len(values) and values[q] == value:
            q += 1
        runs.append((start, q, value))
    return runs


def _fork(cells):
    parses = []
    for outward in DIRS:
        local, perp, kmin, qmin = _local(cells, outward)
        width = max(q for k, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [sum(k > 0 and qq == q for k, qq in local)
                   for q in range(width)]
        expected = {(0, q) for q in range(width)}
        expected |= {(k, q) for q, length in enumerate(lengths)
                     for k in range(1, length+1)}
        runs = _runs(lengths)
        if local != expected or len(runs) != 3:
            continue
        if runs[0][0] != 0 or runs[-1][1] != width:
            continue
        if any(runs[i][1] >= runs[i+1][0] for i in (0, 1)):
            continue
        parses.append(dict(cells=cells, out=outward, perp=perp, kmin=kmin,
                           qmin=qmin, width=width, runs=tuple(runs),
                           center=runs[1][2], neck=runs[1][:2],
                           channel=(runs[0][1], runs[2][0])))
    return parses[0] if len(parses) == 1 else None


def _bridge(cells):
    """A solid rectangular head with one positive uniform inward stem."""
    parses = []
    for outward in DIRS:
        local, perp, kmin, qmin = _local(cells, outward)
        height = max(k for k, q in local) + 1
        width = max(q for k, q in local) + 1
        rows = [{q for k, q in local if k == rr} for rr in range(height)]
        full = set(range(width))
        b = 0
        while b < height and rows[height-1-b] == full:
            b += 1
        stem = height - b
        if not b or not stem:
            continue
        neck = rows[0]
        if not neck or neck != set(range(min(neck), max(neck)+1)):
            continue
        if min(neck) == 0 or max(neck) == width-1:
            continue
        if any(row != neck for row in rows[:stem]):
            continue
        parses.append(dict(cells=cells, out=outward, perp=perp, kmin=kmin,
                           qmin=qmin, width=width, stem=stem, thick=b,
                           neck=(min(neck), max(neck)+1)))
    return parses[0] if len(parses) == 1 else None


def _candidate(fork, bridge, grid, bg):
    if fork['out'] != bridge['out']:
        return None
    if (bridge['qmin'], bridge['qmin']+bridge['width']) != tuple(
            fork['qmin']+q for q in fork['channel']):
        return None
    if tuple(bridge['qmin']+q for q in bridge['neck']) != tuple(
            fork['qmin']+q for q in fork['neck']):
        return None
    destination = fork['kmin'] + fork['center'] + 1
    distance = bridge['kmin'] - destination
    # Initial stem is separated from the complete receiver, not already docked.
    if distance <= 0 or bridge['kmin'] <= max(
            r*fork['out'][0]+c*fork['out'][1] for r, c in fork['cells'])+1:
        return None
    dr, dc = fork['out']
    route = {(r-step*dr, c-step*dc) for step in range(distance+1)
             for r, c in bridge['cells']}
    if any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1])
           or ((r, c) not in bridge['cells'] and grid[r, c] != bg)
           for r, c in route):
        return None
    final = {(r-distance*dr, c-distance*dc) for r, c in bridge['cells']}
    assembly = fork['cells'] | final
    return dict(route=route, final=final, distance=distance,
                holes=_holes(assembly), assembly=assembly)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError('expected an integer ARC grid with dimensions 1..30')
    return grid.astype(np.int8, copy=True)


def _analyze(input_grid):
    grid = _grid(input_grid)
    colors, frequencies = np.unique(grid, return_counts=True)
    if 0 in colors or sum(frequencies == frequencies.max()) != 1:
        raise ValueError('black must be absent and background uniquely most frequent')
    bg = int(colors[np.argmax(frequencies)])
    candidates = []
    for color in colors:
        color = int(color)
        if color == bg:
            continue
        forks, bridges = [], []
        valid = True
        for cells in _components(grid == color):
            fork, bridge = _fork(cells), _bridge(cells)
            if fork is not None and bridge is None:
                forks.append(fork)
            elif bridge is not None and fork is None:
                bridges.append(bridge)
            else:
                valid = False
                break
        if valid and forks and bridges:
            candidates.append((color, forks, bridges))
    if len(candidates) != 1:
        raise ValueError('need one neutral role consisting of forks and T bridges')
    neutral, forks, bridges = candidates[0]
    mapping, keys = {}, []
    for color in colors:
        color = int(color)
        if color in (bg, neutral):
            continue
        comps = _components(grid == color)
        if len(comps) != 1:
            raise ValueError('each key color must form one glyph')
        cells = comps[0]
        k = _holes(cells)
        if k in mapping or k not in (0, 1, 2):
            raise ValueError('key classes must be unique and between zero and two')
        # Cavities really contain background, not other objects.
        mask, (r0, c0) = _mask(cells)
        for cavity in _components(~np.pad(mask, 1)):
            if all(0 < r < mask.shape[0]+1 and 0 < c < mask.shape[1]+1
                   for r, c in cavity):
                if any(grid[r0+r-1, c0+c-1] != bg for r, c in cavity):
                    raise ValueError('nonbackground key cavity')
        mapping[k] = color
        keys.append(cells)
    if not mapping:
        raise ValueError('at least one topology glyph is required')
    matches, used, swept = [], set(), set()
    for fi, fork in enumerate(forks):
        options = []
        for ti, bridge in enumerate(bridges):
            match = _candidate(fork, bridge, grid, bg)
            if match is not None:
                options.append((ti, match))
        if len(options) > 1:
            raise ValueError('multiple geometrically matching bridges')
        if options:
            ti, match = options[0]
            if ti in used or not swept.isdisjoint(match['route']):
                raise ValueError('shared bridge or interacting swept regions')
            if match['holes'] not in (0, 1, 2):
                raise ValueError('a three-pronged assembly has at most two bays')
            used.add(ti)
            swept.update(match['route'])
            matches.append(dict(match, fi=fi, ti=ti,
                                accepted=match['holes'] in mapping))
    return dict(grid=grid, background=bg, neutral=neutral, mapping=mapping,
                keys=keys, forks=forks, bridges=bridges, matches=matches,
                unaligned=len(bridges)-len(used))


def _answer(a):
    out = a['grid'].copy()
    for bridge in a['bridges']:
        _paint(out, bridge['cells'], a['background'])
    for match in a['matches']:
        if match['accepted']:
            _paint(out, match['route']-match['final'], TRAIL)
            _paint(out, match['final'], a['mapping'][match['holes']])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode geometric docking and hypothetical assembly topology from input."""
    return _answer(_analyze(input_grid))


def _module(rng, holes, training, compact=False, force_stem=False,
            force_open_threshold=False):
    """Construct one mechanism and its full reserved travel envelope."""
    s = 1 if compact else int(rng.integers(1, 3))
    gl, gr = (1, 1) if compact else map(int, rng.integers(1, 4, size=2))
    left, right = (1, 1) if compact else map(int, rng.integers(1, 3, size=2))
    c = 1 if compact else int(rng.integers(1, 4))
    t = (1 if training else 2) if compact else int(rng.integers(1, 3) if training else rng.integers(2, 4))
    if force_stem:
        t = 2
    b = 1 if training else 2
    d = c+t+1  # Proximal head row when docked.
    closed = [True]*holes + [False]*(2-holes)
    rng.shuffle(closed)
    lengths = [(d-1 if shut else 1) if compact else
               int(rng.integers(d-1, d+2) if shut else rng.integers(1, d-1))
               for shut in closed]
    if force_open_threshold:
        # Arms longer than the center still leave both bays open for a long stem.
        lengths = [d-2, d-2]
    w = left+gl+s+gr+right
    q0 = left+gl
    structure = {(0, q) for q in range(w)}
    for qa, qb, length in ((0, left, lengths[0]), (q0, q0+s, c),
                           (w-right, w, lengths[1])):
        structure |= {(k, q) for k in range(1, length+1) for q in range(qa, qb)}
    start = max(lengths+[c]) + (2 if compact else int(rng.integers(2, 5)))
    token = {(k, q) for k in range(start, start+t) for q in range(q0, q0+s)}
    token |= {(k, q) for k in range(start+t, start+t+b) for q in range(left, w-right)}
    final = {(k, q) for k in range(c+1, c+1+t) for q in range(q0, q0+s)}
    final |= {(k, q) for k in range(d, d+b) for q in range(left, w-right)}
    # Construct ground truth by unit steps, before any input parser is called.
    route = set()
    for step in range(start-c):
        route |= {(k-step, q) for k, q in token}
    height = max(k for k, q in structure | route)+1
    masks = []
    for cells in (structure, token, final, route):
        mask = np.zeros((height, w), dtype=bool)
        for k, q in cells:
            mask[k, q] = True
        masks.append(mask)
    rotation = int(rng.integers(4))
    masks = [np.rot90(mask, rotation).copy() for mask in masks]
    return dict(masks=masks, holes=holes)


def _positions(free, h, w):
    if h > free.shape[0] or w > free.shape[1]:
        return np.empty((0, 2), dtype=int)
    prefix = np.pad((~free).astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = prefix[h:, w:]-prefix[:-h, w:]-prefix[h:, :-w]+prefix[:-h, :-w]
    return np.argwhere(totals == 0)


def _reserve(free, r, c, h, w):
    free[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = False


def _glyph(rng, k):
    if k == 0:
        mask = np.ones((int(rng.integers(1, 4)), int(rng.integers(2, 5))), dtype=bool)
    else:
        cavity_widths = list(map(int, rng.integers(1, 3, size=k)))
        h = int(rng.integers(3, 6))
        w = sum(cavity_widths)+k+1
        mask = np.ones((h, w), dtype=bool)
        c = 1
        for cw in cavity_widths:
            mask[1:-1, c:c+cw] = False
            c += cw+1
        if rng.random() < .4:
            # A clipped corner gives a diagonal-only seam, still four-enclosed.
            mask[0, 0] = False
    return np.rot90(mask, int(rng.integers(4))).copy()


def _loose_bridge(rng):
    s, gl, gr, t = map(int, (rng.integers(1, 3), rng.integers(1, 3),
                            rng.integers(1, 3), rng.integers(1, 3)))
    mask = np.zeros((t+1, gl+s+gr), dtype=bool)
    mask[:t, gl:gl+s] = True
    mask[t, :] = True
    return np.rot90(mask, int(rng.integers(4))).copy()


def _generate_pair(rng, split, index, base_rotation, _force_fallback=False):
    training = split == 'train'
    # The first two demonstrations expose every hypothetical cavity class.
    classes = [0, 1, 2] if training and index < 2 else list(map(
        int, rng.integers(0, 3, size=int(rng.integers(2, 5)))))
    if training and index == 0:
        keys = [0, 2]
    elif training and index == 1:
        keys = [1, int(rng.choice([0, 2]))]
    else:
        keys = list(map(int, rng.choice(3, size=int(rng.integers(1, 4)), replace=False)))
        if not any(k in keys for k in classes):
            classes[0] = keys[0]
    rng.shuffle(classes)
    rng.shuffle(keys)
    for attempt in (range(47, 48) if _force_fallback else range(48)):
        compact = attempt >= 32
        fallback = attempt == 47
        h = int(rng.integers(24, 31)) if not compact else 30
        w = int(rng.integers(24, 31)) if not compact else 30
        palette = list(map(int, rng.permutation(np.arange(1, 10))))
        bg, neutral = palette[:2]
        mapping = dict(zip(keys, palette[2:]))
        inp = np.full((h, w), bg, dtype=np.int8)
        out = inp.copy()
        free = np.ones((h, w), dtype=bool)
        modules = [_module(rng, k, training, compact,
                           force_stem=training and index == 0 and k in (0, 2),
                           force_open_threshold=training and index == 0 and k == 0)
                   for k in classes]
        modules.sort(key=lambda m: m['masks'][0].size, reverse=True)
        ok = True
        for mi, module in enumerate(modules):
            masks = module['masks']
            mh, mw = masks[0].shape
            positions = _positions(free, mh, mw)
            if not len(positions):
                ok = False
                break
            r, c = ((1+15*(mi//2), 1+15*(mi%2)) if fallback else
                    map(int, positions[int(rng.integers(len(positions)))]))
            ip, op = inp[r:r+mh, c:c+mw], out[r:r+mh, c:c+mw]
            ip[masks[0] | masks[1]] = neutral
            op[masks[0]] = neutral
            if module['holes'] in mapping:
                op[masks[3] & ~masks[2]] = TRAIL
                op[masks[2]] = mapping[module['holes']]
            _reserve(free, r, c, mh, mw)
        if not ok:
            continue
        # Keys have no panel: their unique colors identify floating prototypes.
        for ki, k in enumerate(keys):
            if fallback:
                mask = np.ones((3, max(3, 2*k+1)), dtype=bool)
                mask[1, 1:2*k:2] = False
            else:
                mask = _glyph(rng, k)
            gh, gw = mask.shape
            positions = _positions(free, gh, gw)
            if not len(positions):
                ok = False
                break
            r, c = ((12, 1+9*ki) if fallback else
                    map(int, positions[int(rng.integers(len(positions)))]))
            inp[r:r+gh, c:c+gw][mask] = mapping[k]
            out[r:r+gh, c:c+gw][mask] = mapping[k]
            _reserve(free, r, c, gh, gw)
        if not ok:
            continue
        loose_count = int(rng.integers(1, 3))
        for li in range(loose_count):
            mask = (np.asarray([[False, True, False], [True, True, True]])
                    if fallback else _loose_bridge(rng))
            gh, gw = mask.shape
            positions = _positions(free, gh, gw)
            if not len(positions):
                ok = False
                break
            r, c = ((26, 1+9*li) if fallback else
                    map(int, positions[int(rng.integers(len(positions)))]))
            inp[r:r+gh, c:c+gw][mask] = neutral
            _reserve(free, r, c, gh, gw)
        if not ok:
            continue
        rotation = (base_rotation + (index % 2 if training else 2)) % 4
        inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
        try:
            analysis = _analyze(inp)
            # Check construction against visible geometry, not hidden class labels.
            if (len(analysis['forks']) != len(classes)
                    or sorted(m['holes'] for m in analysis['matches']) != sorted(classes)
                    or analysis['unaligned'] != loose_count
                    or not np.array_equal(_answer(analysis), out)):
                continue
        except ValueError:
            continue
        return {'input': inp, 'output': out}
    raise RuntimeError('bounded free packing did not find a valid scene')


def _count(value, low, high):
    return (isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_)) and low <= value <= high)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _count(num_train, 2, 10) or not _count(num_test, 1, 4) or num_train <= num_test:
        raise ValueError('require 2..10 train, 1..4 test, and train > test')
    if (not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0):
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    rotation = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            for _ in range(16):
                pair = _generate_pair(rng, split, index, rotation)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not produce distinct inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def _ablation(a, omit):
    out = a['grid'].copy()
    for bridge in a['bridges']:
        _paint(out, bridge['cells'], a['background'])
    if omit == 'B':
        # Accept every geometrically fitting bridge, preserving neutral identity.
        for m in a['matches']:
            _paint(out, m['route']-m['final'], TRAIL)
            _paint(out, m['final'], a['neutral'])
    elif omit == 'A':
        # Classify loose bridges in place; every T has zero holes in isolation.
        for bridge in a['bridges']:
            _paint(out, bridge['cells'], a['mapping'].get(_holes(bridge['cells']), a['background']))
    else:
        raise ValueError('unknown ablation')
    return out


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert _count(len(puzzle['train']), 2, 10) and _count(len(puzzle['test']), 1, 4)
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, demonstrations = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            a = _analyze(inp)
            assert np.array_equal(_answer(a), out), 'incorrect docking/topology output'
            assert not np.array_equal(inp, out)
            assert all(_holes(x['cells']) == 0 for x in a['forks']+a['bridges'])
            assert any(m['accepted'] for m in a['matches']), 'need successful motion'
            assert a['unaligned'] >= 1, 'need a geometric distractor'
            for fork in a['forks']:
                assert all(out[p] == a['neutral'] for p in fork['cells'])
            for cells in a['keys']:
                assert all(out[p] == inp[p] for p in cells)
            for m in a['matches']:
                assert len(_components(_mask(m['assembly'])[0])) == 1
                if m['accepted']:
                    assert all(out[p] == a['mapping'][m['holes']] for p in m['final'])
                    assert all(out[p] == TRAIL for p in m['route']-m['final'])
                else:
                    assert all(out[p] == a['background'] for p in m['route'])
            assert not np.array_equal(_ablation(a, 'A'), out)
            assert not np.array_equal(_ablation(a, 'B'), out)
            if split == 'train':
                demonstrations.append(a)
    # Teach all potential counts and both rejection causes within two examples.
    first = demonstrations[:2]
    assert all({m['holes'] for m in a['matches']} == {0, 1, 2} for a in first)
    assert all(any(not m['accepted'] for m in a['matches']) for a in first)
    assert {m['holes'] for a in first for m in a['matches'] if m['accepted']} == {0, 1, 2}
    assert any(m['accepted'] and a['bridges'][m['ti']]['stem'] >= 2
               for a in demonstrations for m in a['matches']), 'expose swept T geometry'
    assert any(m['accepted'] and m['holes'] == 0 and
               min(a['forks'][m['fi']]['runs'][i][2] for i in (0, 2)) >
               a['forks'][m['fi']]['center'] for a in first for m in a['matches']), \
        'side arms longer than the center need not seal the head'


def self_test() -> dict:
    """Finite semantic fixtures, 42-seed diversity, and all explicit counts."""
    # Four-background connectivity treats a diagonal seam as closed.
    diagonal = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
    assert _holes(diagonal) == 1
    assert _holes(diagonal-{(0, 1)}) == 0
    # A hand-built docking fixture: both guards end diagonally beside the head.
    fixture = np.full((14, 18), 9, dtype=np.int8)
    structure = {(0, q) for q in range(7)}
    structure |= {(r, q) for r in range(1, 4) for q in (0, 6)}
    structure |= {(r, 3) for r in (1, 2)}
    token = {(7, 3)} | {(8, q) for q in range(1, 6)}
    key = {(r, c) for r in range(1, 4) for c in range(11, 16)} - {(2, 12), (2, 14)}
    _paint(fixture, structure | token, 5)
    _paint(fixture, key, 2)
    expected = fixture.copy()
    _paint(expected, token, 9)
    _paint(expected, {(r, q) for r in range(5, 9) for q in range(1, 6)}, 0)
    _paint(expected, {(3, 3)} | {(4, q) for q in range(1, 6)}, 2)
    assert np.array_equal(solve(fixture), expected)
    opened = fixture.copy()
    opened[3, 0] = 9
    erased = opened.copy()
    _paint(erased, token, 9)
    assert np.array_equal(solve(opened), erased)
    perm = np.asarray([0, 7, 3, 6, 8, 2, 5, 1, 9, 4], dtype=np.int8)
    assert np.array_equal(solve(perm[fixture]), perm[expected])
    # Force the guaranteed compact packing rather than infer fallback coverage.
    for seed in range(8):
        for split, index in (('train', 0), ('train', 1), ('test', 0)):
            p = _generate_pair(np.random.default_rng(seed), split, index, seed % 4,
                               _force_fallback=True)
            assert np.array_equal(solve(p['input']), p['output'])
    shapes, palettes, signatures, geometry, orientations, populations, travels = (set() for _ in range(7))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for s in ('train', 'test'):
            for pair in puzzle[s]:
                a = _analyze(pair['input'])
                shapes.add(pair['input'].shape)
                palettes.add((a['background'], a['neutral'], tuple(sorted(a['mapping'].items()))))
                populations.add((len(a['forks']), len(a['bridges']), len(a['mapping'])))
                for m in a['matches']:
                    f, b = a['forks'][m['fi']], a['bridges'][m['ti']]
                    geometry.add((f['width'], f['runs'], b['stem'], b['thick'], m['holes']))
                    orientations.add(f['out'])
                    travels.add(m['distance'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                # Solver commutes with rotations.
                for k in (1, 2, 3):
                    assert np.array_equal(solve(np.rot90(pair['input'], k)), np.rot90(pair['output'], k))
    assert len(signatures) == 42 and len(shapes) >= 20 and len(palettes) >= 100
    assert len(geometry) >= 100 and len(populations) >= 8
    assert len(orientations) == 4 and len(travels) >= 5
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            validate_generated_puzzle(generate_puzzle(5000+n*10+m, n, m))
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    a, b = generate_puzzle(71, 2, 1), generate_puzzle(71, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    # Semantic corruption is rejected even though its array schema is legal.
    corrupt = generate_puzzle(17, 2, 1)
    corrupt['test'][0]['output'][0, 0] = (int(corrupt['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), shapes=len(shapes),
                palettes=len(palettes), geometries=len(geometry),
                population_patterns=len(populations), orientations=len(orientations),
                travel_distances=sorted(map(int, travels)))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
