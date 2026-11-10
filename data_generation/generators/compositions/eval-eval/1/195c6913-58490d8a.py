"""Folded counting tape: object-sized cyclic batches on terrain routes.

The solver reads only its argument. No parent code, files, seed, or training
examples are consulted. Generation labels use the constructed paths and object
counts; validation reconstructs both from the finished visible grids.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
_GLYPHS = (_DIAMOND, _SQUARE, ((0, 0),),
           ((0, 0), (1, 1), (2, 2)),
           ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)),
           ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
           ((0, 0), (0, 1), (1, 1)), ((0, 0), (0, 1), (0, 2)))
_REQUIRED = {'unit', 'dense', 'deduplicate', 'reset', 'marker_clock',
             'shared_clock', 'four_neighbors', 'area', 'include_keys',
             'finite', 'straight'}


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _array(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _orient(grid, code):
    return np.rot90(np.fliplr(grid) if code >= 4 else grid, code % 4).copy()


def _inverse(grid, code):
    a = np.rot90(grid, -(code % 4))
    return (np.fliplr(a) if code >= 4 else a).copy()


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, comp = [p], []
        while stack:
            r, c = stack.pop()
            comp.append((int(r), int(c)))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(comp)))
    return result


def _trace(mask, start):
    """Cell walker: a bend belongs to the arriving leg, exactly once."""
    r, c = start
    vertical = False
    path, collisions, legs = [], [], [0]
    while True:
        assert mask[r, c] and (r, c) not in path
        path.append((r, c))
        legs[-1] += 1
        ahead = (r - 1, c) if vertical else (r, c + 1)
        if not _inside(mask.shape, ahead):
            break
        if mask[ahead]:
            r, c = ahead
            continue
        collisions.append(ahead)
        turn = (r, c + 1) if vertical else (r - 1, c)
        if not _inside(mask.shape, turn) or not mask[turn]:
            break
        vertical = not vertical
        legs.append(0)
        r, c = turn
    return {'path': tuple(path), 'collisions': tuple(collisions), 'legs': tuple(legs)}


def _swatches(panel, excluded):
    result = []
    for color in sorted(set(map(int, np.unique(panel))) - set(excluded)):
        for comp in _components(panel == color):
            if len(comp) != 4:
                continue
            r, c = min(p[0] for p in comp), min(p[1] for p in comp)
            if set(comp) == {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}:
                result.append({'row': r, 'col': c, 'color': color, 'cells': comp})
    return result


def _decode_panel(panel, field):
    """Enumerate visible relative orientations, accepting exactly one."""
    candidates = []
    for route in sorted(set(map(int, np.unique(panel))) - {field}):
        # Terrain must be extensive; this is a family constraint, not a color rank.
        if np.count_nonzero(panel == route) < 25:
            continue
        for code in range(8):
            p = _orient(panel, code)
            swatches = _swatches(p, (field, route))
            if not 3 <= len(swatches) <= 6:
                continue
            cells = {q for s in swatches for q in s['cells']}
            starts = tuple(tuple(map(int, q)) for q in np.argwhere(
                ~np.isin(p, [field, route])) if tuple(q) not in cells)
            if not 1 <= len(starts) <= 3 or len({int(p[q]) for q in starts}) != 1:
                continue
            restored = p.copy()
            for q in cells:
                restored[q] = field
            for q in starts:
                restored[q] = route
            if set(map(int, np.unique(restored))) != {field, route}:
                continue
            mask = restored == route
            if not all(c > 0 and c + 1 < p.shape[1] and not mask[r, c-1]
                       and mask[r, c+1] for r, c in starts):
                continue
            for marker in swatches:
                legend = sorted([s for s in swatches if s is not marker], key=lambda s: s['col'])
                seq = tuple(s['color'] for s in legend)
                if len(set(seq)) < 2 or marker['color'] in seq:
                    continue
                if len({s['row'] for s in legend}) != 1:
                    continue
                if any(b['col'] - a['col'] < 3 for a, b in zip(legend, legend[1:])):
                    continue
                if marker['row'] < legend[0]['row'] + 4:
                    continue
                if not all(r >= legend[0]['row'] + 4 and int(p[r, c]) == seq[0]
                           for r, c in starts):
                    continue
                # All annotations have a complete, inferable field ring.
                if any(not np.all(restored[s['row']-1:s['row']+3,
                                            s['col']-1:s['col']+3] == field)
                       for s in swatches):
                    continue
                records = [_trace(mask, s) for s in starts]
                occupied = set()
                valid = True
                for rec in records:
                    footprint = set(rec['path']) | set(rec['collisions'])
                    if footprint & (occupied | cells):
                        valid = False
                    occupied |= footprint
                if not valid:
                    continue
                candidates.append({'panel': p, 'restored': restored, 'field': field,
                                   'route': route, 'code': code, 'sequence': seq,
                                   'marker': marker['color'], 'starts': starts,
                                   'swatches': swatches, 'records': records})
    assert len(candidates) == 1, 'Program and route orientation must be unique'
    return candidates[0]


def _inspect(input_grid):
    grid = _array(input_grid)
    candidates = []
    for field in map(int, np.unique(grid)):
        rr, cc = np.where(grid == field)
        r0, r1, c0, c1 = int(rr.min()), int(rr.max())+1, int(cc.min()), int(cc.max())+1
        panel = grid[r0:r1, c0:c1]
        if min(panel.shape) < 10:
            continue
        if not all(np.all(e == field) for e in (panel[0], panel[-1], panel[:, 0], panel[:, -1])):
            continue
        outside = np.ones(grid.shape, bool)
        outside[r0:r1, c0:c1] = False
        if not outside.any():
            continue
        bg = int(np.argmax(np.bincount(grid[outside], minlength=10)))
        if bg == field or np.mean(grid[outside] == bg) <= .60:
            continue
        ring = np.zeros(grid.shape, bool)
        ring[max(0, r0-1):min(grid.shape[0], r1+1),
             max(0, c0-1):min(grid.shape[1], c1+1)] = True
        ring[r0:r1, c0:c1] = False
        if not np.all(grid[ring] == bg):
            continue
        try:
            info = _decode_panel(panel, field)
        except AssertionError:
            continue
        if bg in {info['route'], info['marker']} | set(info['sequence']):
            continue
        info.update(bounds=(r0, r1, c0, c1), outside=outside, background=bg)
        candidates.append(info)
    assert len(candidates) == 1, 'Exactly one framed terrain panel is required'
    info = candidates[0]
    objects, counts, four, area = {}, {}, {}, {}
    owner = np.full(grid.shape, -1, int)
    serial = 0
    exterior_colors = sorted(set(map(int, grid[info['outside']])) - {info['background']})
    assert not set(exterior_colors) & {info['field'], info['route'], info['marker']}
    for color in exterior_colors:
        mask = (grid == color) & info['outside']
        comps = _components(mask)
        objects[color] = comps
        counts[color], four[color], area[color] = len(comps), len(_components(mask, False)), int(mask.sum())
        for comp in comps:
            for q in comp:
                owner[q] = serial
            serial += 1
    for r, c in np.argwhere(owner >= 0):
        neighborhood = owner[max(0, r-1):r+2, max(0, c-1):c+2]
        assert np.all((neighborhood < 0) | (neighborhood == owner[r, c])), 'Objects touch'
    assert set(info['sequence']) <= set(counts), 'Each program color needs a positive count'
    info.update(objects=objects, counts=counts, four=four, area=area,
                distractors=tuple(sorted(set(counts) - set(info['sequence']))))
    return info


def _paint(info, variant=None):
    """Render the intended rule, or explicit simple alternatives for witnesses."""
    word = info['sequence']
    counts = info['counts'].copy()
    if variant == 'unit':
        counts = dict.fromkeys(counts, 1)
    elif variant == 'four_neighbors':
        counts = info['four']
    elif variant == 'area':
        counts = info['area']
    elif variant == 'include_keys':
        counts = {c: n + word.count(c) + (len(info['starts']) if c == word[0] else 0)
                  for c, n in counts.items()}
    if variant == 'deduplicate':
        word = tuple(dict.fromkeys(word))
    beads = tuple(c for c in word for _ in range(counts[c]))
    out = info['restored'].copy()
    shared = 0
    for rec in info['records']:
        leg_index = 0
        leg_start = 0
        ends = np.cumsum(rec['legs'])
        for i, q in enumerate(rec['path']):
            while i >= ends[leg_index]:
                leg_start = int(ends[leg_index])
                leg_index += 1
            if variant == 'straight' and leg_index:
                continue
            tick = i
            if variant == 'reset':
                tick -= leg_start
            elif variant == 'marker_clock':
                tick += leg_index
            elif variant == 'shared_clock':
                tick += shared
            stride = 1 if variant == 'dense' else 2
            if tick % stride:
                continue
            bead = tick // stride
            if variant == 'finite' and bead >= len(beads):
                continue
            out[q] = beads[bead % len(beads)]
        for q in rec['collisions']:
            out[q] = info['marker']
        shared += len(rec['path'])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Count exterior objects, execute their batches, and return the panel."""
    info = _inspect(input_grid)
    return _inverse(_paint(info), info['code'])


def _evidence(info):
    correct = _paint(info)
    return {name for name in _REQUIRED if not np.array_equal(correct, _paint(info, name))}


def _normalize(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _shape(rng):
    if rng.random() < .65:
        cells = _GLYPHS[int(rng.integers(len(_GLYPHS)))]
    else:
        cells = {(1, 1)}
        for _ in range(int(rng.integers(1, 8))):
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if 0 <= r+dr < 3 and 0 <= c+dc < 3} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    if rng.integers(2):
        cells = [(r, -c) for r, c in cells]
    for _ in range(int(rng.integers(4))):
        cells = [(-c, r) for r, c in cells]
    return _normalize(cells)


def _partition(rng, total, n):
    if n == 1:
        return [total]
    cuts = sorted(map(int, rng.choice(np.arange(1, total), n-1, replace=False)))
    return np.diff([0] + cuts + [total]).tolist()


def _terrain_panel(rng, h, w, word, counts, colors, routes, leg_count):
    field, route, marker = colors
    header = int(rng.integers(1, 3))
    legend_left = int(rng.integers(1, w - 3*len(word) + 2))
    columns = [legend_left + 3*i for i in range(len(word))]
    # Random extra legend gaps when room permits.
    for i in range(1, len(columns)):
        slack = w - 3 - columns[-1]
        extra = int(rng.integers(min(2, slack)+1))
        columns[i:] = [c + extra for c in columns[i:]]
    starts, paths, collisions, leg_lengths = [], [], [], []
    forbidden = np.zeros((h, w), bool)
    forbidden[:header+4] = True
    forbidden[-1] = forbidden[:, 0] = forbidden[:, -1] = True
    r = h - 2 - 2*(routes-1)
    c = int(rng.integers(1, 4))
    starts.append((r, c))
    horizontal = (leg_count+1)//2
    vertical = leg_count//2
    dx = w-2-c-int(rng.integers(0, 3))
    dy = r-header-4-int(rng.integers(0, 2))
    if vertical and dy < vertical:
        raise ValueError('Too little height for requested legs')
    xs = iter(_partition(rng, dx, horizontal))
    ys = iter(_partition(rng, dy, vertical)) if vertical else iter(())
    path, hits, lengths = [(r, c)], [], []
    for leg in range(leg_count):
        n = next(xs) if leg % 2 == 0 else next(ys)
        dr, dc = ((0, 1) if leg % 2 == 0 else (-1, 0))
        for _ in range(n):
            r, c = r+dr, c+dc
            path.append((r, c))
        lengths.append(n + (1 if leg == 0 else 0))
        hit = (r+dr, c+dc)
        hits.append(hit)
        forbidden[hit] = True
    turn = (r-1, c) if leg_count % 2 else (r, c+1)
    forbidden[turn] = True
    paths.append(tuple(path))
    collisions.append(tuple(hits))
    leg_lengths.append(tuple(lengths))
    # Extra starts have short independent straight routes, not copied subpuzzles.
    for j in range(1, routes):
        rr = starts[0][0] + 2*j
        cc = int(rng.integers(1, 4))
        end = int(rng.integers(max(cc+4, w//3), w-2))
        starts.append((rr, cc))
        paths.append(tuple((rr, x) for x in range(cc, end+1)))
        collisions.append(((rr, end+1),))
        leg_lengths.append((end-cc+1,))
        forbidden[rr, end+1] = forbidden[rr-1, end] = True
    for rr, cc in starts:
        forbidden[rr, cc-1] = True
    footprint = {q for p in paths for q in p} | {q for p in collisions for q in p}
    # Reserve a separate 2x2 marker and its field ring, away from all routes.
    options = [(rr, cc) for rr in range(header+4, h-2) for cc in range(1, w-2)
               if not any((r1, c1) in footprint
                          for r1 in range(rr-1, rr+3) for c1 in range(cc-1, cc+3))]
    if not options:
        raise ValueError('No collision swatch position')
    mr, mc = options[int(rng.integers(len(options)))]
    forbidden[mr-1:mr+3, mc-1:mc+3] = True
    mask = np.zeros((h, w), bool)
    for _ in range(int(rng.integers(8, 24))):
        rr = int(rng.integers(header+4, h-1))
        cc = int(rng.integers(1, w-1))
        dh, dw = int(rng.integers(1, h-rr)), int(rng.integers(1, w-cc))
        mask[rr:rr+dh, cc:cc+dw] = True
    mask[forbidden] = False
    for p in paths:
        for q in p:
            if forbidden[q]:
                raise ValueError('Conflicting skeleton')
            mask[q] = True
    path_cells = {q for p in paths for q in p}
    for comp in _components(mask, False):
        if not set(comp) & path_cells:
            for q in comp:
                mask[q] = False
    panel = np.where(mask, route, field).astype(np.int8)
    for cc, color in zip(columns, word):
        panel[header:header+2, cc:cc+2] = color
    panel[mr:mr+2, mc:mc+2] = marker
    for q in starts:
        panel[q] = word[0]
    # Construction labels deliberately use the planned geometry and inventory.
    output = np.where(mask, route, field).astype(np.int8)
    beads = tuple(c for c in word for _ in range(counts[c]))
    for path, hits in zip(paths, collisions):
        for i, q in enumerate(path):
            if i % 2 == 0:
                output[q] = beads[(i//2) % len(beads)]
        for q in hits:
            output[q] = marker
    return panel, output


def _place(grid, blocked, color, shape, rng):
    sh, sw = max(r for r, c in shape)+1, max(c for r, c in shape)+1
    nh, nw = grid.shape[0]-sh+1, grid.shape[1]-sw+1
    allowed = np.ones((nh, nw), bool)
    for r, c in shape:
        allowed &= ~blocked[r:r+nh, c:c+nw]
    positions = np.argwhere(allowed)
    if not len(positions):
        return False
    r0, c0 = positions[int(rng.integers(len(positions)))]
    for dr, dc in shape:
        r, c = int(r0+dr), int(c0+dc)
        grid[r, c] = color
        blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
    return True


def _surround(rng, panel, objects, bg):
    ph, pw = panel.shape
    for attempt in range(10):
        if attempt >= 3:
            h = w = 30
        else:
            h = int(rng.integers(max(ph+4, 22), 31))
            w = int(rng.integers(max(pw+4, 22), 31))
        # Edges, corners, and interiors all occur; large panels favor an edge.
        r0 = int(rng.choice([0, h-ph])) if rng.random() < .65 else int(rng.integers(h-ph+1))
        c0 = int(rng.choice([0, w-pw])) if rng.random() < .65 else int(rng.integers(w-pw+1))
        grid = np.full((h, w), bg, np.int8)
        grid[r0:r0+ph, c0:c0+pw] = panel
        blocked = np.zeros((h, w), bool)
        blocked[max(0, r0-1):r0+ph+1, max(0, c0-1):c0+pw+1] = True
        ordered = list(objects)
        rng.shuffle(ordered)
        ordered.sort(key=lambda obj: -(max(r for r, c in obj[1])+2)*(max(c for r, c in obj[1])+2))
        if all(_place(grid, blocked, color, shape, rng) for color, shape in ordered):
            return grid
    raise ValueError('Packing budget exhausted')


def _make_pair(rng, split, regime, witness, role_color, orientation):
    for attempt in range(100):
        palette = list(map(int, rng.permutation(10)))
        if witness in (0, 1):
            palette.remove(role_color)
        bg, field, route, marker = palette[:4]
        pool = palette[4:]
        k = 2 if witness in (0, 1) else int(rng.integers(2, 4))
        if witness == 0:
            keys = [role_color, pool[0]]
            rest = pool[1:]
        else:
            keys, rest = pool[:k], pool[k:]
        cap = 4 if split == 'test' and regime == 0 else 3
        counts = {c: int(rng.integers(1, cap+1)) for c in keys}
        if witness == 0:
            counts[keys[0]], counts[keys[1]] = 3, int(rng.integers(1, 3))
        if split == 'test' and regime == 0:
            counts[keys[int(rng.integers(k))]] = 4
        length = 5 if split == 'test' and regime == 1 else int(rng.integers(k, 5))
        if witness == 0:
            length = 3
        word = keys + [keys[int(rng.integers(k))] for _ in range(length-k)]
        rng.shuffle(word)
        # The diagonal/area witness must be visible even if it is not first.
        nd = 1 if witness == 1 else int(rng.integers(0, min(2, len(rest))+1))
        distractors = ([role_color] if witness == 1 else rest[:nd])
        objects = []
        for color in keys + distractors:
            n = counts[color] if color in counts else int(rng.integers(1, 5))
            prototype = _shape(rng)
            for j in range(n):
                shape = prototype if rng.random() < .25 else _shape(rng)
                if witness == 0 and color == keys[0]:
                    shape = (_DIAMOND, _SQUARE, ((0, 0),))[j]
                objects.append((color, shape))
        routes = 1 if witness == 0 else 2 if witness == 1 else int(rng.choice([1, 2, 3], p=[.5, .4, .1]))
        legs = int(rng.integers(5, 7)) if split == 'test' and regime == 2 else int(rng.integers(1, 5))
        if witness == 0:
            legs = int(rng.integers(3, 5))
        h = int(rng.integers(17 if witness is not None else 15, 23))
        w = int(rng.integers(max(16, 3*len(word)+2), 24))
        try:
            panel, output = _terrain_panel(rng, h, w, tuple(word), counts,
                                           (field, route, marker), routes, legs)
            grid = _surround(rng, panel, objects, bg)
            pair = {'input': _orient(grid, orientation), 'output': _orient(output, orientation)}
            info = _inspect(pair['input'])
            assert np.array_equal(_inverse(_paint(info), info['code']), pair['output'])
            # Long routes reveal a complete inventory cycle, not just its prefix.
            if witness == 0:
                if not (_REQUIRED - {'shared_clock'}) <= _evidence(info):
                    continue
            if witness == 1 and 'shared_clock' not in _evidence(info):
                continue
            if split == 'test' and regime == 0:
                smaller = dict(info, counts={c: min(n, 3) for c, n in info['counts'].items()})
                if np.array_equal(_paint(info), _paint(smaller)):
                    continue
            if split == 'test' and regime == 1:
                shorter = dict(info, sequence=info['sequence'][:-1])
                if np.array_equal(_paint(info), _paint(shorter)):
                    continue
            return pair
        except (ValueError, AssertionError):
            continue
    raise RuntimeError('Could not construct an informative scene in 100 bounded attempts')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not _integer(num_train) or not _integer(num_test):
        raise ValueError('counts must be integers')
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('require 2..10 train, 1..4 test, and num_train > num_test')
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))
    fixed_orientation = int(rng.integers(8))
    mixed = bool(rng.integers(2))
    role_color = int(rng.integers(10))
    puzzle, used = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for i in range(count):
            witness = i if split == 'train' and i < 2 else None
            code = int(rng.integers(8)) if mixed else fixed_orientation
            for _ in range(8):
                pair = _make_pair(rng, split, regime, witness, role_color, code)
                key = _key(pair['input'])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Distinct input budget exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    used, evidence, training = set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _array(pair['input']), _array(pair['output'])
            assert _key(inp) not in used
            used.add(_key(inp))
            info = _inspect(inp)
            assert np.array_equal(out, _inverse(_paint(info), info['code']))
            assert out.size < inp.size, 'Output is the extracted panel'
            assert np.count_nonzero(info['restored'] == info['route']) >= 25
            # Explicit preservation: only route beads and collision cells change.
            changed = _paint(info) != info['restored']
            allowed = np.zeros(changed.shape, bool)
            for rec in info['records']:
                for q in rec['path'][::2] + rec['collisions']:
                    allowed[q] = True
            assert np.array_equal(changed, allowed)
            if split == 'train':
                evidence |= _evidence(info)
                training.append(info)
    assert _REQUIRED <= evidence, _REQUIRED - evidence
    # Same color is counted in one example and ignored in another.
    assert any(set(a['sequence']) & set(b['distractors']) for a in training for b in training)
    assert any(any(len({len(comp) for comp in m['objects'][c]}) > 1
                   and m['four'][c] > m['counts'][c] for c in m['sequence']) for m in training)


def self_test():
    """Bounded in-memory checks; no sample/trace/log files are written."""
    metrics = {'shapes': set(), 'panels': set(), 'words': set(), 'inventories': set(),
               'orientations': set(), 'routes': set(), 'legs': set(), 'glyphs': set(),
               'backgrounds': set(), 'fields': set(), 'count_values': set()}
    digests = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                for g in pair.values():
                    digest.update(bytes(g.shape) + g.tobytes())
                m = _inspect(pair['input'])
                roles = {}
                metrics['shapes'].add(pair['input'].shape)
                metrics['panels'].add(pair['output'].shape)
                metrics['words'].add(tuple(roles.setdefault(c, len(roles)) for c in m['sequence']))
                metrics['inventories'].add(tuple(m['counts'][c] for c in dict.fromkeys(m['sequence'])))
                metrics['orientations'].add(m['code'])
                metrics['routes'].add(len(m['starts']))
                metrics['legs'].update(len(r['legs']) for r in m['records'])
                metrics['backgrounds'].add(m['background'])
                metrics['fields'].add(m['field'])
                metrics['count_values'].update(m['counts'][c] for c in m['sequence'])
                metrics['glyphs'].update(_normalize(comp) for comps in m['objects'].values() for comp in comps)
        digests.add(digest.digest())
    assert len(digests) == 42
    assert len(metrics['shapes']) >= 25 and len(metrics['panels']) >= 30
    assert len(metrics['words']) >= 15 and len(metrics['inventories']) >= 15
    assert metrics['orientations'] == set(range(8))
    assert metrics['routes'] == {1, 2, 3} and metrics['legs'] >= {1, 2, 3, 4, 5, 6}
    assert metrics['count_values'] == {1, 2, 3, 4}
    assert metrics['fields'] == metrics['backgrounds'] == set(range(10))
    assert len(metrics['glyphs']) >= 35
    # Exercise every legal explicit pair count, and reject types as well as bounds.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(20261022+nt*10+ne, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    a, b = generate_puzzle(713135099, 2, 1), generate_puzzle(713135099, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    # These symmetries and recolorings preserve the rule, including crop orientation.
    sample = a['test'][0]
    for code in range(8):
        assert np.array_equal(solve(_orient(sample['input'], code)),
                              _orient(sample['output'], code))
    palette = np.random.default_rng(314159).permutation(10).astype(np.int8)
    assert np.array_equal(solve(palette[sample['input']]), palette[sample['output']])
    # The public validator also accepts the nested-list format in the contract.
    listed = {s: [{k: v.tolist() for k, v in p.items()} for p in a[s]] for s in a}
    validate_generated_puzzle(listed)
    b['test'][0]['output'][0, 0] = (int(b['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(b)
    except AssertionError:
        pass
    else:
        raise AssertionError('Incorrect output accepted')
    return {k: len(v) for k, v in metrics.items()} | {'seeds': 42, 'distinct_puzzles': 42}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
