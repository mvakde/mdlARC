"""Quota containers: counted glyphs partition an independently sorted solid queue.

Only NumPy and the Python standard library are used. No parent is imported.
"""
from __future__ import annotations

import json
import numpy as np


DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
SINGLE = ((0, 0),)
GLYPHS = (DIAMOND, SQUARE, SINGLE,
          ((0, 0), (1, 1), (2, 2)),
          ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)),
          ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
          ((0, 0), (0, 1), (0, 2)),
          ((0, 0), (1, 0), (1, 1)))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _key(value):
    a = np.asarray(value)
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _signature(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def _components(mask, diagonal=True):
    todo = set(map(tuple, np.argwhere(mask)))
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in steps:
                point = (r + dr, c + dc)
                if point in todo:
                    todo.remove(point)
                    stack.append(point)
        result.append(cells)
    return result


def _tile(size, color, bg, kind):
    a = np.full((size, size), color, dtype=np.int8)
    if kind == 'H':
        a[1:-1, 1:-1] = bg
    return a


def _inspect(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    assert all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1]))
    frames, solids, glyphs, boxes = [], [], [], []
    for color in sorted(set(map(int, np.unique(a))) - {bg}):
        for cells in _components(a == color):
            r = min(p[0] for p in cells)
            c = min(p[1] for p in cells)
            h = max(p[0] for p in cells) - r + 1
            w = max(p[1] for p in cells) - c + 1
            obj = dict(r=r, c=c, h=h, w=w, color=color, cells=tuple(cells))
            boxes.append(obj)
            if h <= 3 and w <= 3:
                glyphs.append(obj)
                continue
            assert h == w and h >= 4, 'Large objects must be square tiles'
            tile = a[r:r+h, c:c+w].copy()
            kind = 'S' if np.all(tile == color) else 'H'
            assert np.array_equal(tile, _tile(h, color, bg, kind))
            obj.update(size=h, kind=kind, tile=tile)
            (frames if kind == 'H' else solids).append(obj)
    assert frames and solids
    sizes = {o['size'] for o in frames + solids}
    assert len(sizes) == 1
    size = sizes.pop()
    # A complete background cell separates bounding boxes, even at corners.
    for i, x in enumerate(boxes):
        assert 0 < x['r'] and 0 < x['c']
        assert x['r'] + x['h'] < a.shape[0] and x['c'] + x['w'] < a.shape[1]
        for y in boxes[:i]:
            assert _separated(x, y), 'Object bounding boxes touch or overlap'
    frames.sort(key=lambda o: (o['r'], o['c']))
    solids.sort(key=lambda o: (o['r'], o['c']))
    counts = {c: sum(g['color'] == c for g in glyphs)
              for c in {f['color'] for f in frames}}
    assert all(q > 0 for q in counts.values()), 'Every frame key has a positive quota'
    meta = dict(bg=bg, size=size, frames=frames, solids=solids,
                glyphs=glyphs, counts=counts)
    assert _output_shape(meta)[0] <= 30 and _output_shape(meta)[1] <= 30
    return meta


def _separated(x, y):
    return (x['r'] + x['h'] < y['r'] or y['r'] + y['h'] < x['r'] or
            x['c'] + x['w'] < y['c'] or y['c'] + y['w'] < x['c'])


def _output_shape(meta, counts=None):
    counts = meta['counts'] if counts is None else counts
    s = meta['size']
    return (len(meta['frames']) * (s + 3) - 1,
            max(counts[f['color']] for f in meta['frames']) * (s + 1) + 1)


def _render(meta, counts=None, frames=None, solids=None, restart=False, wrap=False):
    counts = meta['counts'] if counts is None else counts
    frames = meta['frames'] if frames is None else frames
    solids = meta['solids'] if solids is None else solids
    s, bg = meta['size'], meta['bg']
    out = np.full(_output_shape(meta, counts), bg, dtype=np.int8)
    cursor = 0
    for rank, frame in enumerate(frames):
        q = counts[frame['color']]
        r, w = rank * (s + 3), q * (s + 1) + 1
        out[r, :w] = out[r+s+1, :w] = frame['color']
        out[r:r+s+2, 0] = out[r:r+s+2, w-1] = frame['color']
        if restart:
            cursor = 0
        for j in range(q):
            index = cursor + j
            if wrap:
                index %= len(solids)
            if index < len(solids):
                c = 1 + j * (s + 1)
                out[r+1:r+s+1, c:c+s] = solids[index]['tile']
        cursor += q
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all colors, quotas, geometry and queue orders from this input only."""
    return _render(_inspect(input_grid))


def _alternative(meta, mode):
    """Finite shortcut probes for training evidence, not an independent oracle."""
    if mode in ('one', 'four', 'area', 'all_objects'):
        counts = {}
        for color in meta['counts']:
            gs = [g for g in meta['glyphs'] if g['color'] == color]
            if mode == 'one':
                counts[color] = 1
            elif mode == 'area':
                counts[color] = sum(len(g['cells']) for g in gs)
            elif mode == 'all_objects':
                counts[color] = len(gs) + sum(o['color'] == color for o in
                                              meta['frames'] + meta['solids'])
            else:
                n = 0
                for g in gs:
                    mask = np.zeros((g['h'], g['w']), bool)
                    for r, c in g['cells']:
                        mask[r-g['r'], c-g['c']] = True
                    n += len(_components(mask, False))
                counts[color] = n
        return _render(meta, counts=counts)
    if mode in ('restart', 'wrap'):
        return _render(meta, **{mode: True})
    which, order = mode.split('_')
    objects = meta['frames'] if which == 'frames' else meta['solids']
    if order == 'reverse':
        objects = list(reversed(objects))
    elif order == 'horizontal':
        objects = sorted(objects, key=lambda o: (o['c'], o['r']))
    elif order == 'horizontalreverse':
        objects = sorted(objects, key=lambda o: (o['c'], o['r']), reverse=True)
    return _render(meta, **{which: objects})


PROBES = ('one', 'four', 'area', 'all_objects', 'restart', 'wrap',
          'frames_reverse', 'solids_reverse', 'frames_horizontal',
          'solids_horizontal', 'frames_horizontalreverse', 'solids_horizontalreverse')


def _normalize(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _glyph(rng):
    if rng.random() < .65:
        cells = GLYPHS[int(rng.integers(len(GLYPHS)))]
    else:
        cells = {(1, 1)}
        for _ in range(int(rng.integers(1, 8))):
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if 0 <= r+dr < 3 and 0 <= c+dc < 3} - cells)
            if frontier:
                cells.add(frontier[int(rng.integers(len(frontier)))])
    cells = tuple(cells)
    if rng.integers(2):
        cells = tuple((r, -c) for r, c in cells)
    for _ in range(int(rng.integers(4))):
        cells = tuple((-c, r) for r, c in cells)
    return _normalize(cells)


def _layout(rng, objects, size, tie=False):
    """Guaranteed shelf initialization followed by free rectangle relocation."""
    big = [o for o in objects if o.get('kind') in ('H', 'S')]
    small = [o for o in objects if 'kind' not in o]
    options = []
    for cols in range(2, min(5, len(big)) + 1):
        rows = (len(big) + cols - 1) // cols
        bh, bw = rows * (size+1) - 1, cols * (size+1) - 1
        if bh > 28 or bw > 28:
            continue
        sites = [(r, c) for r in range(bh+2, 27, 4) for c in range(1, 27, 4)]
        sites += [(r, c) for r in range(1, bh, 4) for c in range(bw+2, 27, 4)
                  if r+2 <= bh]
        if len(sites) >= len(small):
            options.append((cols, sites))
    assert options, 'Packing budget exceeds guaranteed capacity'
    cols, sites = options[int(rng.integers(len(options)))]
    positions = [(1 + (i//cols)*(size+1), 1 + (i%cols)*(size+1))
                 for i in range(len(big))]
    if tie:
        hs = [o for o in big if o['kind'] == 'H']
        rest = [o for o in big if o['kind'] == 'S']
        big = hs + rest
    else:
        rng.shuffle(big)
    for o, (r, c) in zip(big, positions):
        o.update(r=r, c=c)
    if rng.random() < .65:
        sites.sort(key=lambda p: p[0]+p[1]+int(rng.integers(0, 6)))
    else:
        rng.shuffle(sites)
    for o, (r, c) in zip(small, sites):
        o.update(r=r, c=c)
    min_h = max(o['r']+o['h'] for o in objects) + 1
    min_w = max(o['c']+o['w'] for o in objects) + 1
    assert min_h <= 30 and min_w <= 30
    h = int(rng.integers(min_h, 31))
    w = int(rng.integers(min_w, 31))
    dr, dc = int(rng.integers(h-min_h+1)), int(rng.integers(w-min_w+1))
    for o in objects:
        o['r'] += dr
        o['c'] += dc
    locked = {id(o) for o in big[:2]} if tie else set()
    for _ in range(3):
        for i in rng.permutation(len(objects)):
            o = objects[int(i)]
            if id(o) in locked:
                continue
            rr, cc = np.indices((h-o['h']-1, w-o['w']-1))
            rr, cc = rr+1, cc+1
            legal = np.ones(rr.shape, bool)
            for other in objects:
                if other is not o:
                    legal &= ((rr+o['h'] < other['r']) |
                              (other['r']+other['h'] < rr) |
                              (cc+o['w'] < other['c']) |
                              (other['c']+other['w'] < cc))
            candidates = np.argwhere(legal)
            assert len(candidates)
            r, c = candidates[int(rng.integers(len(candidates)))] + 1
            o.update(r=int(r), c=int(c))
    return h, w


def _make_pair(rng, s, nh, ns, bg, main, witness=0, cap=3,
               repeat=False, force_quota=None):
    pool = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
    if witness == 1:
        pool.remove(main)
        pool.insert(0, main)
    if witness == 2:
        pool.remove(main)
    frame_colors = pool[:nh]
    if repeat:
        frame_colors[-1] = frame_colors[0]
    counts = {c: int(rng.integers(1, cap+1)) for c in frame_colors}
    if witness == 1:
        counts = {frame_colors[0]: 3, frame_colors[1]: int(rng.integers(1, 3))}
        ns = sum(counts.values()) + 1
    elif witness == 2:
        counts = {frame_colors[0]: 1, frame_colors[1]: 2}
        ns = 2
    elif force_quota is not None:
        counts[frame_colors[0]] = force_quota
    frames = [dict(kind='H', color=c, size=s, h=s, w=s) for c in frame_colors]
    solids = [dict(kind='S', color=int(rng.choice(pool)), size=s, h=s, w=s)
              for _ in range(ns)]
    if witness == 1:
        for o, c in zip(solids, pool):
            o['color'] = c
    if witness == 2:
        solids[0]['color'] = main
        solids[1]['color'] = pool[-1]
    glyphs = []
    for color, count in counts.items():
        for j in range(count):
            if witness == 1:
                # The two keyed colors have area/count ratios three and one,
                # ruling out a single fixed area divisor in this one example.
                cells = (DIAMOND, SQUARE, SINGLE)[j] if color == main else SINGLE
            else:
                cells = _glyph(rng)
            glyphs.append(dict(color=color, cells=cells,
                               h=max(r for r, c in cells)+1, w=max(c for r, c in cells)+1))
    distractors = [c for c in pool if c not in counts]
    if witness == 2:
        distractors.insert(0, main)
    nd = int(rng.integers(0, 3)) if witness != 2 else int(rng.integers(1, 3))
    for color in distractors[:nd]:
        for _ in range(int(rng.integers(1, 4))):
            cells = _glyph(rng)
            glyphs.append(dict(color=color, cells=cells,
                               h=max(r for r, c in cells)+1, w=max(c for r, c in cells)+1))
    objects = frames + solids + glyphs
    shape = _layout(rng, objects, s, tie=witness == 2)
    # Quotas follow frame colors, not position; put the large quota first in
    # the extension witness and one-slot quota first in the exhaustion witness.
    frames.sort(key=lambda o: (o['r'], o['c']))
    if witness:
        for o, color in zip(frames, frame_colors):
            o['color'] = color
    solids.sort(key=lambda o: (o['r'], o['c']))
    if witness == 1:
        for o, color in zip(solids, pool):
            o['color'] = color
    grid = np.full(shape, bg, dtype=np.int8)
    for o in frames + solids:
        o['tile'] = _tile(s, o['color'], bg, o['kind'])
        grid[o['r']:o['r']+s, o['c']:o['c']+s] = o['tile']
    for o in glyphs:
        for dr, dc in o['cells']:
            grid[o['r']+dr, o['c']+dc] = o['color']
    construction = dict(size=s, bg=bg, frames=frames, solids=solids, counts=counts)
    out = _render(construction)
    meta = _inspect(grid)
    assert meta['counts'] == counts and len(meta['solids']) == ns
    assert np.array_equal(out, _render(meta))
    return dict(input=grid, output=out)


def _integer(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(int(seed))
    main = int(rng.integers(10))
    fixed_bg = int(rng.choice([c for c in range(10) if c != main]))
    consistent = rng.random() < .18
    first_s = int(rng.integers(4, 6))
    regime = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', nt), ('test', nq)):
        for i in range(n):
            witness = i+1 if split == 'train' and i < 2 else 0
            s = first_s if witness == 1 else (9-first_s if witness == 2 else int(rng.integers(4, 6)))
            nh = 2 if witness else int(rng.integers(1, 4))
            repeat, force, cap = False, None, 3
            if split == 'test':
                if regime == 0:
                    s = 6
                    cap = 4
                elif regime == 1:
                    cap = 29 // (s+1)
                    force = cap
                elif regime == 2:
                    nh = int(rng.integers(2, 4))
                    repeat = True
                else:
                    s, nh = 4, 4
            elif not witness and rng.random() < .25 and nh > 1:
                repeat = True
            # Choose the resource budget AFTER the scale-shift regime. Seven
            # side-six solids plus three frames leave insufficient guaranteed
            # space for all quota glyphs; side-four tiles can use seven solids.
            ns = int(rng.integers(1, 8 if s == 4 else 7))
            bg = fixed_bg if consistent else int(rng.choice([c for c in range(10) if c != main]))
            for attempt in range(48):
                pair = _make_pair(rng, s, nh, ns, bg, main, witness, cap, repeat, force)
                key = _key(pair['input'])
                if key in seen:
                    continue
                if witness == 1:
                    m = _inspect(pair['input'])
                    needed = ('frames_horizontal', 'solids_horizontal', 'solids_horizontalreverse')
                    if any(np.array_equal(pair['output'], _alternative(m, mode)) for mode in needed):
                        continue
                break
            else:
                raise RuntimeError('Bounded sampling could not construct training evidence')
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            meta = _inspect(inp)
            assert np.array_equal(out, _render(meta)), 'Incorrect allocation or border geometry'
            assert not np.array_equal(inp, out)
            # A conserved prefix is allocated exactly once; right padding is
            # background, and frame widths encode every quota, even if empty.
            demand = sum(meta['counts'][f['color']] for f in meta['frames'])
            assert demand >= 1
            occupied, exhausted = [], False
            s = meta['size']
            for rank, frame in enumerate(meta['frames']):
                for j in range(meta['counts'][frame['color']]):
                    r, c = rank * (s+3)+1, 1+j*(s+1)
                    slot = out[r:r+s, c:c+s]
                    if np.all(slot == meta['bg']):
                        exhausted = True
                    else:
                        assert not exhausted, 'Filled slot after exhaustion'
                        assert np.all(slot == slot[0, 0]), 'Resource must remain a complete solid'
                        occupied.append(int(slot[0, 0]))
            assert occupied == [o['color'] for o in meta['solids'][:min(demand, len(meta['solids']))]], 'Resource prefix conservation'
            if split == 'train':
                training.append(meta)
    for mode in PROBES:
        assert any(not np.array_equal(_render(m), _alternative(m, mode)) for m in training), mode
    differences = [len(m['solids']) - sum(m['counts'][f['color']] for f in m['frames'])
                   for m in training]
    assert min(differences) < 0 < max(differences), 'Need shortage and surplus demonstrations'
    assert any(len({f['r'] for f in m['frames']}) < len(m['frames']) and
               len({f['color'] for f in m['frames']}) > 1 for m in training), 'Need visible tie convention'
    keyed = {f['color'] for m in training for f in m['frames']}
    ignored = {g['color'] for m in training for g in m['glyphs'] if g['color'] not in m['counts']}
    assert keyed & ignored, 'Need locally keyed versus ignored color evidence'


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    metas = [_inspect(p['input']) for puzzle in puzzles for split in ('train', 'test')
             for p in puzzle[split]]
    assert len({_signature(p) for p in puzzles}) == 42
    assert _signature(puzzles[12]) == _signature(generate_puzzle(12))
    assert {m['size'] for m in metas} == {4, 5, 6}
    assert len({p['input'].shape for puzzle in puzzles for split in ('train', 'test')
                for p in puzzle[split]}) >= 30
    assert {m['bg'] for m in metas} == set(range(10))
    assert {len(m['frames']) for m in metas} >= {1, 2, 3, 4}
    assert {len(m['solids']) for m in metas} >= set(range(1, 8))
    assert {q for m in metas for q in m['counts'].values()} >= set(range(1, 6))
    assert any(len(m['counts']) < len(m['frames']) for m in metas)
    assert any(len({o['color'] for o in m['solids']}) < len(m['solids']) for m in metas)
    vocab = {_normalize(g['cells']) for m in metas for g in m['glyphs']}
    assert len(vocab) >= 30
    normalized = set()
    for puzzle in puzzles:
        for split in ('train', 'test'):
            for p in puzzle[split]:
                roles = {}
                normalized.add(tuple(tuple(roles.setdefault(int(c), len(roles)) for c in row)
                                     for row in p['input']))
                assert np.array_equal(solve(p['input'].copy()), p['output'])
    assert len(normalized) >= 180
    # The rule and parser also work at physically compatible larger scales,
    # beyond the current sampler's ordinary 4..6 scale distribution.
    for side in range(7, 14):
        pair = _make_pair(np.random.default_rng(900+side), side, 1, 1, 0, 1, cap=1)
        assert np.array_equal(solve(pair['input']), pair['output'])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+10*nt+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('Illegal example counts accepted')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Damaged container border accepted')


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': 42}, separators=(',', ':')))
