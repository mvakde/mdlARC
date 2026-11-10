"""Census-cut scaffolds: object cardinality selects equal-length colored rods.

Only the standard library and NumPy are used. No files, traces, or caches are
written by this module. All generation/retry loops and self-tests are bounded.
"""
from __future__ import annotations

import hashlib
import numpy as np


_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
_GLYPHS = (_DIAMOND, _SQUARE, ((0, 0),),
           ((0, 0), (1, 1), (2, 2)),
           ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)),
           ((0, 0), (0, 1)), ((0, 0), (1, 0), (1, 1)),
           ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)))


def _check_grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], []
        while stack:
            r, c = stack.pop()
            component.append((int(r), int(c)))
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(component)))
    return result


def _normalize(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


# A rod is (orientation, top row, left column, length, color), in scene coords.
def _cells(p):
    o, r, c, n, _ = p
    return frozenset((r + (i if o == 'V' else 0),
                      c + (i if o == 'H' else 0)) for i in range(n))


def _shift(p, d):
    return p[0], p[1] + d, p[2], p[3], p[4]


def _settle(pieces, height):
    """Greatest simultaneously movable set, with downward unit translations."""
    result = list(pieces)
    for _ in range(height * (len(result) + 1)):
        cells = [_cells(p) for p in result]
        targets = [frozenset((r+1, c) for r, c in s) for s in cells]
        moving = {i for i, s in enumerate(targets) if all(r < height for r, c in s)}
        while True:
            blocked = {i for i in moving for j in range(len(result))
                       if j not in moving and targets[i] & cells[j]}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return result
        result = [_shift(p, 1) if i in moving else p for i, p in enumerate(result)]
    raise AssertionError('Gravity exceeded its finite displacement bound')


def _settle_direct(pieces, height):
    """Separate lowest-first distance scan, used only as a physics cross-check."""
    result = list(pieces)
    for _ in range(height * (len(result) + 1)):
        changed = False
        order = sorted(range(len(result)),
                       key=lambda i: max(r for r, c in _cells(result[i])), reverse=True)
        for i in order:
            occupied = set().union(*[_cells(p) for j, p in enumerate(result) if j != i])
            distance = 0
            for d in range(1, height + 1):
                target = _cells(_shift(result[i], d))
                if any(r >= height for r, c in target) or target & occupied:
                    break
                distance = d
            if distance:
                result[i] = _shift(result[i], distance)
                changed = True
        if not changed:
            return result
    raise AssertionError('Direct settling failed to converge')


def _parse(input_grid):
    grid = _check_grid(input_grid)
    h, w = grid.shape
    assert h >= 8 and w >= 5
    frame = int(grid[0, 0])
    full = np.flatnonzero(np.all(grid == frame, axis=1))
    assert len(full) == 2 and full[0] == 0
    bottom = int(full[1])
    assert 3 <= bottom < h - 3
    start = bottom + 2
    bg = int(grid[bottom+1, 0])
    assert bg != frame and np.all(grid[bottom+1] == bg)
    assert np.all(grid[:bottom+1, 0] == frame)
    assert np.all(grid[:bottom+1, -1] == frame)
    frame_mask = np.zeros(grid.shape, bool)
    frame_mask[[0, bottom], :] = True
    frame_mask[:bottom+1, [0, w-1]] = True
    assert np.array_equal(grid == frame, frame_mask), 'Frame color is exclusive'
    inventory = grid[1:bottom, 1:-1]
    counts, objects = {}, {}
    owner = np.full(inventory.shape, -1, int)
    serial = 0
    for color in sorted(set(map(int, inventory.flat)) - {bg}):
        comps = _components(inventory == color)
        counts[color] = len(comps)
        objects[color] = tuple(_normalize(s) for s in comps)
        for comp in comps:
            for cell in comp:
                owner[cell] = serial
            serial += 1
    assert counts, 'Inventory is nonempty'
    # Different inventory objects, even differently colored ones, have halos.
    for r, c in np.argwhere(owner >= 0):
        neighbors = owner[max(0, r-1):r+2, max(0, c-1):c+2]
        assert np.all((neighbors < 0) | (neighbors == owner[r, c]))
    scene = grid[start:]
    rods = []
    for color in sorted(set(map(int, scene.flat)) - {bg}):
        comps = _components(scene == color, False)
        assert len(comps) == len(_components(scene == color)), 'Same-color input rods have a halo'
        for comp in comps:
            rows, cols = {r for r, c in comp}, {c for r, c in comp}
            assert len(comp) >= 2 and (len(rows) == 1 or len(cols) == 1)
            r, c = min(rows), min(cols)
            o = 'H' if len(rows) == 1 else 'V'
            rod = (o, r, c, len(comp), color)
            assert _cells(rod) == frozenset(comp), 'Every rod is a complete straight run'
            rods.append(rod)
    assert rods
    return dict(grid=grid, frame=frame, bg=bg, start=start, height=h-start,
                counts=counts, objects=objects, inventory=inventory, rods=rods)


def _render(meta, pieces):
    out = meta['grid'].copy()
    out[meta['start']:] = meta['bg']
    occupied = set()
    for p in pieces:
        cells = _cells(p)
        assert not cells & occupied
        occupied.update(cells)
        for r, c in cells:
            assert 0 <= r < meta['height'] and 0 <= c < out.shape[1]
            out[meta['start']+r, c] = p[4]
    return out


def _survivors(meta, counts=None, comparison='equal'):
    counts = meta['counts'] if counts is None else counts
    def deleted(p):
        if p[4] not in counts:
            return False
        if comparison == 'all_color':
            return True
        if comparison == 'at_most':
            return p[3] <= counts[p[4]]
        if comparison == 'at_least':
            return p[3] >= counts[p[4]]
        return p[3] == counts[p[4]]
    return [p for p in meta['rods'] if not deleted(p)]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """The visible input alone supplies the counts, pieces, and boundary."""
    meta = _parse(input_grid)
    return _render(meta, _settle(_survivors(meta), meta['height']))


def _edges(pieces):
    cells = [_cells(p) for p in pieces]
    return [(i, j) for i, p in enumerate(pieces) for j in range(len(pieces))
            if i != j and _cells(_shift(p, 1)) & cells[j]]


def _features(input_grid):
    m = _parse(input_grid)
    rods = m['rods']
    remaining = _survivors(m)
    settled = _settle(remaining, m['height'])
    distances = [q[1] - p[1] for p, q in zip(remaining, settled)]
    edges = _edges(rods)
    support = {i: [j for a, j in edges if a == i] for i in range(len(rods))}
    def depth(i, path=()):
        assert i not in path
        return 1 + max((depth(j, path+(i,)) for j in support[i]), default=0)
    positive = sorted(set(distances) - {0})
    short = any(distances[i] > 0 and distances[j] == 0 and
                remaining[i][0] == 'H' and remaining[j][0] == 'V' and
                remaining[j][1] > remaining[i][1] + 1
                for i, j in _edges(settled))
    partial = any(any(p[4] == c and p[3] == n for p in rods) and
                  any(p[4] == c and p[3] != n for p in rods)
                  for c, n in m['counts'].items())
    m.update(remaining=remaining, settled=settled, distances=distances,
             deleted=len(rods)-len(remaining), moved=sum(d > 0 for d in distances),
             falls=positive, depth=max(depth(i) for i in range(len(rods))),
             redundant=any(len(s) >= 2 for s in support.values()),
             short_landing=short, partial_color=partial,
             expected=_render(m, settled))
    return m


def _alternative(meta, kind):
    """Deliberately wrong rules, for demonstration-discrimination checks."""
    counts = meta['counts']
    if kind == 'four_connected':
        counts = {c: len(_components(meta['inventory'] == c, False)) for c in counts}
    elif kind == 'area':
        counts = {c: int(np.sum(meta['inventory'] == c)) for c in counts}
    elif kind == 'count_rods':
        counts = {c: sum(p[4] == c for p in meta['rods']) for c in counts}
    if kind == 'length_only':
        remaining = [p for p in meta['rods'] if p[3] not in counts.values()]
    else:
        remaining = _survivors(meta, counts, kind)
    settled = remaining if kind == 'no_gravity' else _settle(remaining, meta['height'])
    return _render(meta, settled)


_ALTERNATIVES = ('four_connected', 'area', 'all_color', 'at_most', 'at_least',
                 'count_rods', 'length_only', 'no_gravity')


def _shape(rng):
    if rng.random() < .65:
        cells = _GLYPHS[int(rng.integers(len(_GLYPHS)))]
    else:
        cells = {(1, 1)}
        for _ in range(int(rng.integers(0, 7))):
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if 0 <= r+dr < 3 and 0 <= c+dc < 3} - cells)
            if frontier:
                cells.add(frontier[int(rng.integers(len(frontier)))])
    if rng.random() < .5:
        cells = tuple((r, -c) for r, c in cells)
    for _ in range(int(rng.integers(4))):
        cells = tuple((-c, r) for r, c in cells)
    return _normalize(cells)


def _pack_inventory(rng, counts, bg, frame, width, witness):
    objects = []
    primary = next(iter(counts))
    for color, n in counts.items():
        for i in range(n):
            shape = (_DIAMOND, _SQUARE, ((0, 0),))[i] if witness and color == primary and i < 3 else _shape(rng)
            objects.append((color, shape))
    cols = (width-1)//4
    rows = (len(objects)+cols-1)//cols
    # Slots guarantee a lossless fallback; ordinary placement has no slot grid.
    height = 4*rows+1 + int(rng.integers(0, 2))
    assert height <= 15
    for attempt in range(5):
        panel = np.full((height, width), bg, np.int8)
        panel[[0, -1], :] = frame
        panel[:, [0, -1]] = frame
        blocked = panel == frame
        order = list(objects)
        rng.shuffle(order)
        order.sort(key=lambda item: -(1+max(r for r, c in item[1]))*(1+max(c for r, c in item[1])))
        if attempt == 4:
            slots = [(1+4*r, 1+4*c) for r in range(rows) for c in range(cols)]
            rng.shuffle(slots)
            for (color, shape), (r, c) in zip(order, slots):
                for dr, dc in shape:
                    panel[r+dr, c+dc] = color
            return panel
        success = True
        for color, shape in order:
            sh, sw = 1+max(r for r, c in shape), 1+max(c for r, c in shape)
            nh, nw = height-sh+1, width-sw+1
            allowed = np.ones((nh, nw), bool)
            for dr, dc in shape:
                allowed &= ~blocked[dr:dr+nh, dc:dc+nw]
            locations = np.argwhere(allowed)
            if not len(locations):
                success = False
                break
            r, c = map(int, locations[int(rng.integers(len(locations)))])
            for dr, dc in shape:
                rr, cc = r+dr, c+dc
                panel[rr, cc] = color
                blocked[max(0, rr-1):rr+2, max(0, cc-1):cc+2] = True
        if success:
            return panel
    raise AssertionError('Unreachable inventory fallback')


def _drop(pieces, orientation, col, length, height):
    p = (orientation, 0, col, length, -1)
    occupied = set().union(*[_cells(q) for q in pieces])
    if any(r >= height for r, c in _cells(p)) or _cells(p) & occupied:
        return False
    for d in range(1, height+1):
        target = _cells(_shift(p, d))
        if any(r >= height for r, c in target) or target & occupied:
            pieces.append(_shift(p, d-1))
            return True
    return False


def _geometry(rng, height, width, n, kind):
    """Seed a mechanical interaction, then grow a network by deposition.

    The first two teaching cores carry specific counterexamples. General cores
    vary spans, post locations, number of landings, and redundant/joint release.
    Colors 0..3 here are role indexes, replaced with local palette values later.
    """
    f = height-1
    if kind == 'witness':
        x = int(rng.integers(0, width-14))
        pieces = [('V', f-2, x+1, 3, 0), ('V', f-1, x+6, 2, 0),
                  ('V', f-3, x+11, 4, 0), ('H', f-3, x, 8, 2),
                  ('V', f-5, x+3, 2, 0), ('H', f-6, x+2, 3, 3),
                  ('H', f-4, x+9, 5, 1)]
    elif kind == 'redundant_witness':
        x = int(rng.integers(0, width-14))
        pieces = [('V', f-n+1, x+1, n, 0), ('V', f-n+1, x+6, n, 1),
                  ('H', f-n, x, 8, 2), ('V', f-n+1, x+11, n, 0),
                  ('H', f-n, x+9, 5, 3)]
    elif kind == 'plinth':
        x = int(rng.integers(0, width-n+1))
        pieces = [('H', f, x, n, 0)]
        # Uprights can overhang the same removable footing through shared caps.
        for i, c in enumerate(sorted(set((x, x+n-1)))):
            length = int(rng.integers(2, min(6, height-4)))
            pieces.append(('V', f-length, c, length, 1 if i == 0 else 3))
        _drop(pieces, 'H', max(0, x-1), min(n+2, width-max(0, x-1)), height)
    else:
        span = int(rng.integers(max(5, n+1), min(16, width)+1))
        left = int(rng.integers(0, width-span+1))
        xs = list(map(int, rng.choice(np.arange(left, left+span), 2, replace=False)))
        xs.sort()
        pieces = [('V', f-n+1, xs[0], n, 0), ('H', f-n, left, span, 2)]
        if kind in ('joint', 'redundant'):
            pieces.append(('V', f-n+1, xs[1], n, 1))
        # Short posts do not support the initial beam; they become landing sites.
        candidates = [c for c in range(left, left+span)
                      if all(abs(c-p[2]) >= 2 for p in pieces if p[0] == 'V')]
        rng.shuffle(candidates)
        for c in candidates[:int(rng.integers(0, min(3, len(candidates))+1))]:
            if n >= 3 and all(abs(c-p[2]) >= 2 for p in pieces if p[0] == 'V'):
                length = int(rng.integers(2, n))
                pieces.append(('V', f-length+1, c, length, 0 if rng.random() < .5 else 3))
        # More floor posts outside the beam create possible cross-stack contacts.
        outside = [c for c in range(width) if c < left-1 or c > left+span]
        rng.shuffle(outside)
        for c in outside[:int(rng.integers(0, min(3, len(outside))+1))]:
            length = int(rng.integers(2, min(11, height-3)))
            p = ('V', f-length+1, c, length, -1)
            if not any(_cells(p) & _cells(q) for q in pieces):
                pieces.append(p)
    # Random straight pieces deposited from above create shared beams, shelves,
    # branches, and towers. Their lengths/positions are independent samples.
    extra = int(rng.integers(0 if kind == 'witness' else 2,
                             5 if kind == 'witness' else 12))
    target = len(pieces)+extra
    for _ in range(4*extra+4):
        if len(pieces) >= target:
            break
        o = 'H' if rng.random() < .6 else 'V'
        length = int(rng.integers(2, min(width, 16)+1)) if o == 'H' else int(rng.integers(2, min(8, height-2)+1))
        col = int(rng.integers(0, width-length+1)) if o == 'H' else int(rng.integers(width))
        _drop(pieces, o, col, length, height)
    return pieces


def _color(rng, geometry, palette, witness):
    cells = [_cells(p) for p in geometry]
    neighbors = [s | {(r+dr, c+dc) for r, c in s
                      for dr in (-1, 0, 1) for dc in (-1, 0, 1)} for s in cells]
    assigned = {i: p[4] for i, p in enumerate(geometry) if p[4] >= 0}
    for i, role in assigned.items():
        if any(i != j and role == other and neighbors[i] & cells[j]
               for j, other in assigned.items()):
            return None
    for i in range(len(geometry)):
        if i in assigned:
            continue
        forbidden = {role for j, role in assigned.items() if neighbors[i] & cells[j]}
        choices = [j for j in range(len(palette)) if j not in forbidden and not (witness and j == 0)]
        if not choices:
            return None
        assigned[i] = int(rng.choice(choices))
    return [(*p[:4], palette[assigned[i]]) for i, p in enumerate(geometry)]


def _make_pair(rng, target, kind):
    for attempt in range(32):
        colors = list(map(int, rng.permutation(10)))
        bg, frame = colors[:2]
        palette = colors[2:2+int(rng.integers(4, 7))]
        counts = {palette[0]: target}
        if kind == 'joint':
            counts[palette[1]] = target
        elif kind in ('witness', 'redundant_witness', 'redundant'):
            counts[palette[1]] = 2 if target != 2 else 3
        else:
            counts[palette[1]] = int(rng.integers(1, 5))
        if kind not in ('witness', 'redundant_witness') and rng.random() < .35:
            # Inventory-only colors are harmless; the frame still names no rod.
            counts[colors[-1]] = int(rng.integers(1, 3))
        # At most three 3-cell-high slot rows, leaving room for the scaffold.
        minimum_width = max(18, 4*((sum(counts.values())+2)//3)+1)
        width = int(rng.integers(minimum_width, 31))
        panel = _pack_inventory(rng, counts, bg, frame, width, kind == 'witness')
        low = max(10, target+5)
        high = min(20, 29-panel.shape[0])
        if low > high:
            continue
        height = int(rng.integers(low, high+1))
        geometry = _geometry(rng, height, width, target, kind)
        rods = _color(rng, geometry, palette, kind == 'witness')
        if rods is None:
            continue
        if _settle(rods, height) != rods:
            continue
        grid = np.full((panel.shape[0]+1+height, width), bg, np.int8)
        grid[:len(panel)] = panel
        meta = dict(grid=grid, bg=bg, start=len(panel)+1, height=height)
        grid = _render(meta, rods)
        # Construction labels use the actual object list and complete rods.
        survivors = [p for p in rods if counts.get(p[4]) != p[3]]
        meta['grid'] = grid
        output = _render(meta, _settle(survivors, height))
        parsed = _features(grid)
        assert parsed['counts'] == counts and set(parsed['rods']) == set(rods)
        assert np.array_equal(parsed['expected'], output)
        if kind in ('witness', 'redundant_witness', 'landing', 'joint', 'plinth') and not parsed['moved']:
            continue
        if kind == 'witness' and not all(
                not np.array_equal(_alternative(parsed, alt), output) for alt in _ALTERNATIVES):
            continue
        return {'input': grid, 'output': output}
    raise RuntimeError('Unable to construct a readable census scaffold in 32 attempts')


def _integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+' must be an integer')
    if not low <= value <= high:
        raise ValueError(name+' is outside the allowed range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt = _integer(num_train, 2, 10, 'num_train')
    nq = _integer(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(int(seed))
    puzzle = {'train': [], 'test': []}
    seen = set()
    specs = [('train', 3, 'witness'),
             ('train', int(rng.choice((2, 4))), 'redundant_witness')]
    kinds = ('landing', 'joint', 'redundant', 'plinth')
    specs += [('train', int(rng.integers(2, 5)), str(rng.choice(kinds))) for _ in range(nt-2)]
    # A quantity extrapolation, with recombined geometry rather than new physics.
    specs += [('test', int(rng.integers(5, 8)), str(rng.choice(('landing', 'joint', 'plinth'))))
              for _ in range(nq)]
    for split, target, kind in specs:
        for _ in range(12):
            try:
                pair = _make_pair(rng, target, kind)
            except RuntimeError:
                continue
            if _key(pair['input']) not in seen:
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError('Bounded scene construction failed')
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check visible segmentation, mechanical invariants, and teaching evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _check_grid(pair['input']), _check_grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            m = _features(inp)
            assert _settle(m['rods'], m['height']) == m['rods'], 'Input scaffold is stable'
            assert _settle(m['settled'], m['height']) == m['settled']
            assert inp.shape == out.shape and np.array_equal(inp[:m['start']], out[:m['start']])
            assert np.array_equal(out, m['expected'])
            assert all(q[0] == p[0] and q[2:] == p[2:] and q[1] >= p[1]
                       for p, q in zip(m['remaining'], m['settled']))
            assert m['remaining'] and m['deleted'] > 0
            records[split].append(m)
    training = records['train']
    assert any(m['partial_color'] and m['short_landing'] and m['moved'] >= 2 for m in training)
    assert any(m['redundant'] for m in training)
    for alt in _ALTERNATIVES:
        assert any(not np.array_equal(_alternative(m, alt), m['expected']) for m in training), alt
    # Sampled shift is independently visible, rather than a hidden solver mode.
    assert max(n for m in training for n in m['counts'].values()) <= 4
    assert all(max(m['counts'].values()) >= 5 and m['moved'] > 0 for m in records['test'])


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 sequential seeds, bounded memory, no stored samples or trace payloads."""
    shapes, normalized, glyphs, depths, rod_counts = set(), set(), set(), set(), set()
    backgrounds, frames, targets, signatures = set(), set(), set(), set()
    flags = dict(unequal_falls=0, short_landing=0, redundant=0, partial_color=0,
                 horizontal_cut=0, vertical_cut=0)
    pairs_checked = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for pair in puzzle['train']+puzzle['test']:
            inp = pair['input']
            m = _features(inp)
            assert _settle_direct(m['remaining'], m['height']) == m['settled']
            query = inp.copy()
            assert np.array_equal(solve(query), pair['output'])
            assert np.array_equal(query, inp)
            shapes.add(inp.shape)
            role_map = {}
            norm = bytes(role_map.setdefault(int(c), len(role_map)) for c in inp.flat)
            normalized.add(hashlib.sha256(bytes(inp.shape)+norm).hexdigest())
            glyphs.update(s for ss in m['objects'].values() for s in ss)
            backgrounds.add(m['bg'])
            frames.add(m['frame'])
            targets.update(m['counts'].values())
            depths.add(m['depth'])
            rod_counts.add(len(m['rods']))
            flags['unequal_falls'] += len(m['falls']) >= 2
            for name in ('short_landing', 'redundant', 'partial_color'):
                flags[name] += bool(m[name])
            deleted = [p for p in m['rods'] if p not in m['remaining']]
            flags['horizontal_cut'] += any(p[0] == 'H' for p in deleted)
            flags['vertical_cut'] += any(p[0] == 'V' for p in deleted)
            pairs_checked += 1
        if seed < 3:
            assert _digest(puzzle) == _digest(generate_puzzle(seed))
    assert len(signatures) == 42 and len(normalized) >= 200
    assert len(shapes) >= 35 and len(glyphs) >= 35
    assert backgrounds == frames == set(range(10))
    assert targets >= set(range(1, 8)) and len(depths) >= 5 and len(rod_counts) >= 10
    assert all(value >= 3 for value in flags.values()), flags
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(42000+10*a+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # A fixed rod unit test explicitly needs simultaneous stack descent.
    tower = [('V', 7, 2, 3, 1), ('H', 6, 0, 5, 2), ('V', 4, 2, 2, 3)]
    assert _settle(tower, 10) == tower
    assert [q[1]-p[1] for p, q in zip(tower[1:], _settle(tower[1:], 10))] == [3, 3]
    bad = generate_puzzle(0, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Validator accepted a corrupt output')
    return dict(seeds=42, pairs=pairs_checked, distinct_puzzles=len(signatures),
                shapes=len(shapes), normalized_structures=len(normalized), glyphs=len(glyphs),
                support_depths=sorted(depths), rod_counts=sorted(rod_counts), **flags)
