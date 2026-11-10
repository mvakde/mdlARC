"""Rhythmic embroidery: cadence along a folded path, terminal-selected tile polarity.

Only visible pixels determine the rule. Generation paints from sampled path
indices; solve decodes the frames and reconstructs pixels by inverse coordinates.
No files, parent modules, random state, or training examples are read by solve.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


HOST_SEEDS = (0,161193292,376010238,679028213,259912573,733584712,215896095,
             899438921,726518184,429745032,772120242,668637419,650748994,
             393543363,246075393,666876714,200138113,740300209,564939853,
             307375072,499073468,441840387,948466054,396908550,795094051,
             158307278,647421510,110708103,910556411,874669484,663279133,
             63659772,98881777,982084392,492261188,737324120,352587344,
             434627379,782621673,879830087,565074604,923264868)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(int, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _neighbors(point):
    r, c = point
    return ((r-1, c), (r+1, c), (r, c-1), (r, c+1))


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        comp, stack = {start}, [start]
        while stack:
            for point in _neighbors(stack.pop()):
                if point in todo:
                    todo.remove(point)
                    comp.add(point)
                    stack.append(point)
        result.append(comp)
    return result


def _frame_boxes(grid):
    """The unique color consisting of exactly two complete rectangular rings."""
    candidates = []
    for color in np.unique(grid):
        comps = _components(grid == color)
        if len(comps) != 2:
            continue
        boxes = []
        for comp in comps:
            rr, cc = zip(*comp)
            r, c, h, w = min(rr), min(cc), max(rr)-min(rr)+1, max(cc)-min(cc)+1
            if min(h, w) < 4:
                break
            ring = {(y, x) for y in range(r, r+h) for x in range(c, c+w)
                    if y in (r, r+h-1) or x in (c, c+w-1)}
            if comp != ring:
                break
            boxes.append((r, c, h, w))
        if len(boxes) == 2:
            candidates.append((int(color), boxes))
    if len(candidates) != 1:
        raise ValueError("need exactly two unambiguous framed objects")
    return candidates[0]


def _ordered_path(mask):
    points = set(map(tuple, np.argwhere(mask)))
    adjacent = {p: [q for q in _neighbors(p) if q in points] for p in points}
    ends = sorted(p for p in points if len(adjacent[p]) == 1)
    if len(ends) != 2 or any(len(v) not in (1, 2) for v in adjacent.values()):
        raise ValueError("the guide must be an open, unbranched, one-cell-wide path")
    path, previous, current = [], None, ends[0]
    while True:
        path.append(current)
        forward = [q for q in adjacent[current] if q != previous]
        if not forward:
            break
        previous, current = current, forward[0]
        if len(path) > len(points):
            raise ValueError("guide contains a cycle")
    if len(path) != len(points):
        raise ValueError("guide is disconnected")
    return path


def _decode(input_grid):
    grid = _grid(input_grid)
    frame, boxes = _frame_boxes(grid)
    outside = np.ones(grid.shape, bool)
    objects = []
    for r, c, h, w in boxes:
        outside[r:r+h, c:c+w] = False
        objects.append(grid[r+1:r+h-1, c+1:c+w-1].copy())
    grounds = np.unique(grid[outside])
    if len(grounds) != 1:
        raise ValueError("the outside ground must be uniform and visible")
    ground = int(grounds[0])
    tile_choices = [i for i, a in enumerate(objects)
                    if a.shape[0] == a.shape[1] and 2 <= len(a) <= 5
                    and len(np.unique(a)) == 2 and ground not in a]
    if len(tile_choices) != 1:
        raise ValueError("one complete binary square specimen is required")
    tile = objects[tile_choices[0]]
    field = objects[1-tile_choices[0]]
    reference = int(tile[0, 0])
    other = int(next(c for c in np.unique(tile) if c != reference))
    guide_choices = set(map(int, np.unique(field))) - {ground, reference, other}
    if len(guide_choices) != 1 or ground not in field:
        raise ValueError("field needs a distinct guide hue and visible ground")
    guide = guide_choices.pop()
    if len({frame, ground, reference, other, guide}) != 5:
        raise ValueError("frame, ground, guide and both tile roles must be distinct")
    n = len(tile)
    if any(s*n > 30 for s in field.shape):
        raise ValueError("expanded field exceeds ARC bounds")
    path = _ordered_path(field != ground)
    refs = [i for i, point in enumerate(path) if field[point] == reference]
    if len(refs) != 2:
        raise ValueError("exactly two specimen-corner-color reference marks are required")
    a, b = refs
    period = b-a
    extras = [i for i, point in enumerate(path) if field[point] == other]
    aligned = [i for i in extras if (i-a) % period == 0]
    if len(aligned) > 1:
        raise ValueError("multiple on-cadence terminals are ambiguous")
    terminal = aligned[0] if aligned else None
    if terminal is not None and a < terminal < b:
        raise ValueError("terminal must be beyond the reference pair")
    decoys = [i for i in extras if i != terminal]
    return dict(tile=tile, field=field, path=path, a=a, b=b, period=period,
                terminal=terminal, decoys=decoys, reference=reference, other=other,
                ground=ground, guide=guide, frame=frame)


def _beat_indices(d):
    a, b, p, t = d['a'], d['b'], d['period'], d['terminal']
    low, high = (0, len(d['path'])-1) if t is None else (min(a, t), max(b, t))
    return [i for i in range(low, high+1) if (i-a) % p == 0]


def _render(d, behavior='normal'):
    """Inverse-coordinate pixel reconstruction; variants only audit shortcuts."""
    tile, field, path = d['tile'], d['field'], d['path']
    n = len(tile)
    beats = _beat_indices(d)
    if behavior == 'observed_only':
        beats = [d['a'], d['b']] + ([] if d['terminal'] is None else [d['terminal']])
    elif behavior == 'solid_path':
        low, high = (0, len(path)-1) if d['terminal'] is None else (
            min(d['a'], d['terminal']), max(d['b'], d['terminal']))
        beats = list(range(low, high+1))
    elif behavior in ('ignore_terminal', 'unbounded_terminal'):
        beats = list(range(d['a'] % d['period'], len(path), d['period']))
    elif behavior == 'manhattan':
        r, c = path[d['a']]
        low, high = (0, len(path)-1) if d['terminal'] is None else (
            min(d['a'], d['terminal']), max(d['b'], d['terminal']))
        beats = [i for i, (y, x) in enumerate(path)
                 if low <= i <= high and (abs(y-r)+abs(x-c)) % d['period'] == 0]
    stamp_index = np.full(field.shape, -1, int)
    for i in beats:
        stamp_index[path[i]] = i
    yy, xx = np.indices((field.shape[0]*n, field.shape[1]*n))
    mr, mc = yy//n, xx//n
    out = field[mr, mc].copy()
    if behavior == 'erase_decoys':
        for i in d['decoys']:
            r, c = path[i]
            out[r*n:(r+1)*n, c*n:(c+1)*n] = d['guide']
    ty, tx = yy % n, xx % n
    if behavior == 'rotate_tile':
        turns = np.zeros(field.shape, int)
        directions = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}
        for i, point in enumerate(path):
            before, after = (point, path[i+1]) if i+1 < len(path) else (path[i-1], point)
            turns[point] = directions[(after[0]-before[0], after[1]-before[1])]
        k = turns[mr, mc]
        for _ in range(3):
            active = k > 0
            ty, tx = np.where(active, n-1-tx, ty), np.where(active, ty, tx)
            k = k-1
    values = tile[ty, tx]
    exchange = d['terminal'] is not None and behavior not in ('no_exchange', 'ignore_terminal')
    if exchange:
        values = d['reference'] + d['other'] - values
    if behavior == 'solid_stamp':
        values = np.full(out.shape, d['other'] if exchange else d['reference'], int)
    active = stamp_index[mr, mc] >= 0
    if behavior == 'partial_stamp':
        active &= values == (d['other'] if exchange else d['reference'])
    out[active] = values[active]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve any well-formed family input without a seed or demonstrations."""
    return _render(_decode(input_grid))


def _turns(path):
    steps = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(steps, steps[1:]))


def _serpent(rng, h, w, regular=False):
    rows = [0]
    while rows[-1]+2 < h:
        gap = 2 if regular or rows[-1]+3 >= h else int(rng.integers(2, 4))
        rows.append(rows[-1]+gap)
    if len(rows) < 3:
        rows = [0, 2, 4]
    col, path = 0, [(0, 0)]
    for j, row in enumerate(rows):
        choices = list(range(max(3, w-3), w)) if j % 2 == 0 else list(range(min(3, w-3)))
        choices = [x for x in choices if abs(x-col) >= 3]
        end = (w-1 if j % 2 == 0 else 0) if regular else int(_pick(rng, choices))
        step = 1 if end > col else -1
        path.extend((row, x) for x in range(col+step, end+step, step))
        col = end
        if j+1 < len(rows):
            path.extend((y, col) for y in range(row+1, rows[j+1]+1))
    return path


def _spiral(h, w):
    path, used, direction = [(0, 0)], {(0, 0)}, 0
    moves = ((0, 1), (1, 0), (0, -1), (-1, 0))
    while True:
        added = False
        for _ in range(2):
            r, c = path[-1]
            dr, dc = moves[direction]
            point = (r+dr, c+dc)
            if (0 <= point[0] < h and 0 <= point[1] < w and point not in used
                    and all(q not in used or q == path[-1] for q in _neighbors(point))):
                path.append(point)
                used.add(point)
                added = True
                break
            direction = (direction+1) % 4
        if not added:
            return path


def _wander(rng, h, w):
    point = (int(rng.integers(h)), int(rng.integers(w)))
    path, used = [point], {point}
    while True:
        choices = [p for p in _neighbors(path[-1])
                   if 0 <= p[0] < h and 0 <= p[1] < w and p not in used
                   and all(q not in used or q == path[-1] for q in _neighbors(p))]
        if not choices:
            return path
        point = _pick(rng, choices)
        path.append(point)
        used.add(point)


def _path(rng, h, w, rescue=False):
    style = 0 if rescue else int(rng.integers(3))
    path = (_serpent(rng, h, w, regular=rescue) if style == 0 else
            _spiral(h, w) if style == 1 else _wander(rng, h, w))
    if len(path) < 14 or _turns(path) < 3:
        path = _serpent(rng, h, w, regular=True)
    # Crop endpoints without changing the guide's connectivity or metric.
    spare = min(5, len(path)-14)
    if spare and not rescue:
        first = int(rng.integers(spare+1))
        last = int(rng.integers(spare-first+1))
        shortened = path[first:len(path)-last]
        if _turns(shortened) >= 3:
            path = shortened
    mask = np.zeros((h, w), bool)
    for point in path:
        mask[point] = True
    mask = np.rot90(mask, int(rng.integers(4)))
    if rng.integers(2):
        mask = mask[:, ::-1]
    return mask, _ordered_path(mask)


def _signal_options(path, period, kind):
    length = len(path)
    result = []
    for a in range(length-period):
        b = a+period
        phase = list(range(a % period, length, period))
        if len(phase) < 3:
            continue
        if kind == 'boundary_witness':
            if a < period or b+period >= length:
                continue
            if path[a][0] == path[b][0] or path[a][1] == path[b][1]:
                continue
            # Parity alone cannot establish path distance: use period three,
            # and require witnesses against Manhattan phase from BOTH marks.
            if any({i for i, point in enumerate(path)
                    if (abs(point[0]-path[j][0])+abs(point[1]-path[j][1])) % period == 0}
                   == set(phase) for j in (a, b)):
                continue
            result.append((a, b, None))
        elif kind == 'boundary':
            result.append((a, b, None))
        else:
            for t in phase:
                if a <= t <= b or min(abs(t-a), abs(t-b)) < 2*period:
                    continue
                lo, hi = min(a, t), max(b, t)
                if kind == 'stop_witness':
                    if not any(i < lo or i > hi for i in phase) or _turns(path[lo:hi+1]) < 1:
                        continue
                result.append((a, b, t))
    return result


def _tile(rng, n, colors, witness):
    for _ in range(32):
        bits = rng.random((n, n)) < float(rng.uniform(.2, .8))
        bits[0, 0] = False
        if not bits.any():
            continue
        if witness and any(np.array_equal(bits, np.rot90(bits, k)) for k in (1, 2, 3)):
            continue
        return np.where(bits, colors[1], colors[0]).astype(int)
    bits = np.zeros((n, n), bool)
    bits[0, -1] = True
    return np.where(bits, colors[1], colors[0]).astype(int)


def _pack(rng, tile, field, ground, frame):
    def framed(a):
        return np.pad(a, 1, constant_values=frame)
    first, second = framed(tile), framed(field)
    if rng.integers(2):
        first, second = second, first
    horizontal = bool(rng.integers(2))
    gap = int(rng.integers(1, 4))
    padding = [int(rng.integers(1, 3)) for _ in range(4)]
    if horizontal:
        h, w = max(len(first), len(second)), first.shape[1]+gap+second.shape[1]
        origins = [(int(rng.integers(h-len(first)+1)), 0),
                   (int(rng.integers(h-len(second)+1)), first.shape[1]+gap)]
    else:
        h, w = len(first)+gap+len(second), max(first.shape[1], second.shape[1])
        origins = [(0, int(rng.integers(w-first.shape[1]+1))),
                   (len(first)+gap, int(rng.integers(w-second.shape[1]+1)))]
    top, bottom, left, right = padding
    grid = np.full((h+top+bottom, w+left+right), ground, int)
    for a, (r, c) in zip((first, second), origins):
        grid[top+r:top+r+len(a), left+c:left+c+a.shape[1]] = a
    assert max(grid.shape) <= 30
    return grid


_SHORTCUTS = ('observed_only', 'solid_path', 'ignore_terminal', 'unbounded_terminal',
              'no_exchange', 'solid_stamp', 'rotate_tile', 'manhattan',
              'erase_decoys', 'partial_stamp')


def _make_pair(rng, role):
    for attempt in range(64):
        rescue = attempt >= 48
        n = int(rng.integers(2, 6))
        maximum = min(14, 30//n)
        h, w = ((5, 5) if rescue else tuple(map(int, rng.integers(5, maximum+1, size=2))))
        mask, path = _path(rng, h, w, rescue)
        if role == 'boundary_witness':
            periods, kind = [3], role
        elif role == 'stop_witness':
            periods, kind = [int(rng.integers(1, 3))], role
        elif role == 'test':
            periods = list(range(4, min(8, (len(path)-1)//2)+1))
            kind = 'terminal' if rng.integers(2) else 'boundary'
        else:
            periods, kind = [int(rng.integers(1, 4))], ('terminal' if rng.integers(2) else 'boundary')
        choices = [(p, opts) for p in periods if (opts := _signal_options(path, p, kind))]
        if not choices:
            continue
        p, options = _pick(rng, choices)
        a, b, terminal = _pick(rng, options)
        ground, frame, guide, reference, other = map(int, rng.choice(10, 5, replace=False))
        tile = _tile(rng, n, (reference, other), role.endswith('witness'))
        field = np.full(mask.shape, ground, int)
        field[mask] = guide
        field[path[a]] = field[path[b]] = reference
        if terminal is not None:
            field[path[terminal]] = other
        eligible = [i for i in range(len(path)) if (i-a) % p]
        count = (1 if role == 'boundary_witness' else
                 0 if not eligible else int(rng.integers(min(2, len(eligible))+1)))
        decoys = list(map(int, rng.choice(eligible, count, replace=False)))
        for i in decoys:
            field[path[i]] = other
        # Forward labels use sampled ordered coordinates, independently of the
        # input parser and inverse-coordinate rendering in solve.
        output = np.repeat(np.repeat(field, n, axis=0), n, axis=1)
        low, high = (0, len(path)-1) if terminal is None else (min(a, terminal), max(b, terminal))
        stamp = tile if terminal is None else reference+other-tile
        for i in range(low, high+1):
            if (i-a) % p == 0:
                r, c = path[i]
                output[r*n:(r+1)*n, c*n:(c+1)*n] = stamp
        inp = _pack(rng, tile, field, ground, frame)
        d = _decode(inp)
        assert np.array_equal(_render(d), output), 'forward/inverse disagreement'
        required = (('ignore_terminal', 'unbounded_terminal', 'no_exchange') if role == 'stop_witness'
                    else ('observed_only', 'solid_path', 'solid_stamp', 'rotate_tile',
                          'manhattan', 'erase_decoys', 'partial_stamp') if role == 'boundary_witness' else ())
        if any(np.array_equal(_render(d, rule), output) for rule in required):
            continue
        return {'input': inp, 'output': output}
    raise RuntimeError('bounded path construction exhausted')


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count('num_train', num_train, 2, 10)
    num_test = _count('num_test', num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    roles = ['stop_witness', 'boundary_witness'] + ['extra']*(num_train-2)
    rng.shuffle(roles)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, requests in (('train', roles), ('test', ['test']*num_test)):
        for role in requests:
            for _ in range(8):
                pair = _make_pair(rng, role)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('bounded uniqueness construction exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            d = _decode(inp)
            n, field = len(d['tile']), d['field']
            assert len(d['path']) >= 14 and _turns(d['path']) >= 3
            assert out.shape == (len(field)*n, field.shape[1]*n)
            assert np.array_equal(out, solve(inp)), 'incorrect output'
            beats = set(_beat_indices(d))
            assert len(beats) >= 3 and beats - {d['a'], d['b'], d['terminal']}, 'no completed beat'
            stamp = d['tile'] if d['terminal'] is None else d['reference']+d['other']-d['tile']
            index = {p: i for i, p in enumerate(d['path'])}
            # Explicit whole-block overwrite, transparency, and decoy preservation.
            for r in range(len(field)):
                for c in range(field.shape[1]):
                    block = out[r*n:(r+1)*n, c*n:(c+1)*n]
                    if index.get((r, c)) in beats:
                        assert np.array_equal(block, stamp)
                    else:
                        assert np.all(block == field[r, c]), 'untouched macrocell changed'
            assert (d['period'] <= 3 if split == 'train' else d['period'] >= 4), 'cadence shift missing'
            analyses[split].append((d, out))
    training = analyses['train']
    assert any(d['terminal'] is not None for d, _ in training)
    assert any(d['terminal'] is None and d['a'] >= d['period']
               and d['b']+d['period'] < len(d['path']) for d, _ in training)
    for rule in _SHORTCUTS:
        assert any(not np.array_equal(_render(d, rule), out) for d, out in training), (
            f'training does not distinguish {rule}')


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed stream; no file I/O, puzzle archive, or execution traces."""
    # Hand-specified semantic fixtures: the expected beat coordinates below
    # are literal evidence, not computed by the progression implementation.
    fixture_path = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,4),(2,4),(2,3),(2,2),
                    (2,1),(2,0),(3,0),(4,0),(4,1),(4,2),(4,3),(4,4)]
    fixture_tile = np.array([[1,2],[2,2]])
    for refs, terminal, decoy, expected_points, exchanged in (
            ((3,5),11,8,[(0,3),(1,4),(2,3),(2,1),(3,0)],True),
            ((5,8),None,6,[(0,2),(1,4),(2,2),(3,0),(4,2)],False)):
        field = np.zeros((5,5), int)
        for point in fixture_path:
            field[point] = 5
        for i in refs:
            field[fixture_path[i]] = 1
        if terminal is not None:
            field[fixture_path[terminal]] = 2
        field[fixture_path[decoy]] = 2
        inp = np.zeros((9,15), int)
        inp[1:5,1:5] = 9
        inp[2:4,2:4] = fixture_tile
        inp[1:8,7:14] = 9
        inp[2:7,8:13] = field
        expected = np.repeat(np.repeat(field, 2, axis=0), 2, axis=1)
        for r, c in expected_points:
            expected[2*r:2*r+2,2*c:2*c+2] = 3-fixture_tile if exchanged else fixture_tile
        assert np.array_equal(solve(inp), expected), 'literal folded-path fixture failed'
        # A second on-phase terminal is invalid, rather than a hidden tie-break.
        ambiguous = inp.copy()
        if terminal is not None:
            r, c = fixture_path[1]
            ambiguous[2+r,8+c] = 2
            try:
                solve(ambiguous)
            except ValueError:
                pass
            else:
                raise AssertionError('ambiguous second terminal accepted')
    shapes, sizes, lengths, periods, turns, palettes, fingerprints = (set() for _ in range(7))
    backgrounds, motifs, modes, directions = set(), set(), set(), set()
    decoys = added = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _decode(pair['input'])
                shapes.add(tuple(pair['output'].shape))
                sizes.add(len(d['tile']))
                lengths.add(len(d['path']))
                periods.add(d['period'])
                turns.add(_turns(d['path']))
                palettes.add(tuple(map(int, np.unique(pair['input']))))
                backgrounds.add(d['ground'])
                motifs.add((d['tile'].shape, (d['tile'] == d['reference']).tobytes()))
                modes.add((split, d['terminal'] is not None))
                if d['terminal'] is not None:
                    directions.add(-1 if d['terminal'] < d['a'] else 1)
                decoys += len(d['decoys'])
                added += len(set(_beat_indices(d)) - {d['a'], d['b'], d['terminal']})
        del puzzle
    assert len(fingerprints) == 42
    assert sizes == {2, 3, 4, 5}
    assert len(shapes) >= 35 and len(lengths) >= 25 and len(turns) >= 5
    assert {1, 2, 3, 4, 5, 6} <= periods and max(periods) >= 7
    assert len(palettes) >= 40 and len(backgrounds) == 10 and len(motifs) >= 80
    assert modes == {(s, t) for s in ('train', 'test') for t in (False, True)}
    assert directions == {-1, 1} and decoys >= 100 and added >= 500
    # All legal explicit counts, plus type/range/order failures.
    legal_calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(900+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
            legal_calls += 1
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),
                 (3.5,1),(4,1.0),(None,1),(4,None)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Corruption and ambiguous-marker rejection.
    p = generate_puzzle(0, 2, 1)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect label accepted')
    # Exhaustive color permutation is unnecessary: several permutations must
    # commute with solve, even when black is a foreground or frame color.
    for pair in p['train']+p['test']:
        for shift in (1, 4, 7):
            assert np.array_equal(solve((pair['input']+shift) % 10), (pair['output']+shift) % 10)
        # Path direction is a parser convenience, not a visible convention.
        d = _decode(pair['input'])
        rev = dict(d, path=d['path'][::-1], a=len(d['path'])-1-d['b'],
                   b=len(d['path'])-1-d['a'],
                   terminal=None if d['terminal'] is None else len(d['path'])-1-d['terminal'],
                   decoys=[len(d['path'])-1-i for i in d['decoys']])
        assert np.array_equal(_render(rev), pair['output'])
    return {'seeds': 42, 'pairs': 210, 'legal_count_combinations': legal_calls,
            'distinct_puzzles': len(fingerprints), 'output_shapes': len(shapes),
            'tile_sizes': sorted(sizes), 'path_lengths': len(lengths),
            'turn_counts': sorted(turns), 'periods': sorted(periods),
            'palettes': len(palettes), 'backgrounds': len(backgrounds),
            'binary_motifs': len(motifs), 'decoys': decoys, 'completed_beats': added,
            'training_shortcuts_rejected': list(_SHORTCUTS),
            'semantic_verification': 'generator-owned checks only'}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
