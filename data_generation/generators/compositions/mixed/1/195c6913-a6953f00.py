"""Cyclic staircase painting followed by a parity-selected top-corner crop.

Only NumPy and the standard library are used. Generation never imports a parent
generator. The solver decodes visible annotations and walks the input terrain.
No execution traces, per-seed payloads, or logs are written by this module.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer)
            or any(not 1 <= n <= 30 for n in grid.shape)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return grid.astype(np.int8, copy=True)


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(grid, color):
    remaining = set(map(tuple, np.argwhere(grid == color)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _decode(input_grid):
    grid = _grid(input_grid)
    n, w = grid.shape
    if n != w or n < 8:
        raise ValueError("the annotated canvas must be square")
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) < 5:
        raise ValueError("missing terrain or program colors")
    order = np.argsort(counts)
    if counts[order[-3]] >= counts[order[-2]]:
        raise ValueError("terrain colors must strictly outnumber symbol colors")
    terrains = set(map(int, colors[order[-2:]]))
    swatches, starts = [], []
    for color in sorted(set(map(int, colors)) - terrains):
        for component in _components(grid, color):
            if len(component) == 1:
                starts.append(component[0])
                continue
            r, c = component[0]
            if set(component) != {(r, c), (r+1, c), (r, c+1), (r+1, c+1)}:
                raise ValueError("program symbols must be isolated 2x2 swatches or starts")
            swatches.append(dict(row=r, col=c, color=color, cells=component))
    if not 1 <= len(starts) <= 2 or not 3 <= len(swatches) <= 7:
        raise ValueError("need one/two starts, two..six entries, and a marker swatch")
    groups = {}
    for s in swatches:
        groups.setdefault(s['row'], []).append(s)
    legends = [v for v in groups.values() if len(v) >= 2]
    if len(legends) != 1 or len(legends[0]) != len(swatches)-1:
        raise ValueError("legend and separate marker are not uniquely grouped")
    legend = sorted(legends[0], key=lambda s: s['col'])
    marker = next(s for s in swatches if s not in legend)
    if any(b['col']-a['col'] < 3 for a, b in zip(legend, legend[1:])):
        raise ValueError("swatches need terrain gaps")
    annotations = {p for s in swatches for p in s['cells']}
    ring = {int(grid[p]) for r, c in annotations
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
            if _inside(grid.shape, p) and p not in annotations}
    if len(ring) != 1 or not ring <= terrains:
        raise ValueError("swatches must have an inferable surrounding terrain")
    bg = ring.pop()
    route = next(iter(terrains-{bg}))
    firsts = {int(grid[p]) for p in starts}
    left = any(c == 0 for r, c in starts)
    right = any(c == n-1 for r, c in starts)
    if len(firsts) != 1 or left == right:
        raise ValueError("an edge start must identify exactly one entry side")
    reflected = right
    word = tuple(s['color'] for s in (legend[::-1] if reflected else legend))
    if firsts != {word[0]} or marker['color'] in word:
        raise ValueError("start/legend/marker roles disagree")
    if any(r < legend[0]['row']+4 for r, c in starts):
        raise ValueError("starts must lie below the legend")
    restored = grid.copy()
    for p in annotations:
        restored[p] = bg
    for p in starts:
        restored[p] = route
    if reflected:
        restored = restored[:, ::-1].copy()
        starts = [(r, n-1-c) for r, c in starts]
        annotations = {(r, n-1-c) for r, c in annotations}
    starts = tuple(sorted(starts))
    mask = restored == route
    for r, c in starts:
        if ((c > 0 and mask[r, c-1]) or c+1 >= n or not mask[r, c+1]):
            raise ValueError("each start must enter terrain from the common entry side")
    return dict(grid=grid, terrain=restored, mask=mask, starts=starts,
                word=word, marker=marker['color'], bg=bg, route=route,
                reflected=reflected, annotations=annotations,
                legend_row=legend[0]['row'], swatches=swatches)


def _trace(mask, start, turn_at_edge=False):
    """Cell-step walk; a bend cell is counted once, an edge exit is terminal."""
    r, c = start
    direction = 0
    path, collisions, legs = [], [], [0]
    while True:
        if not _inside(mask.shape, (r, c)) or not mask[r, c]:
            raise ValueError("route left its terrain")
        path.append((r, c))
        legs[-1] += 1
        dr, dc = ((0, 1), (-1, 0))[direction]
        ahead = (r+dr, c+dc)
        if _inside(mask.shape, ahead) and mask[ahead]:
            r, c = ahead
            continue
        outside = not _inside(mask.shape, ahead)
        if outside and not turn_at_edge:
            end = 'edge'
            break
        if not outside:
            collisions.append(ahead)
        direction = 1-direction
        dr, dc = ((0, 1), (-1, 0))[direction]
        turn = (r+dr, c+dc)
        if not _inside(mask.shape, turn) or not mask[turn]:
            end = 'edge' if outside else 'blocked'
            break
        legs.append(0)
        r, c = turn
        if len(path) > sum(mask.shape):
            raise ValueError("non-monotone route")
    return dict(path=tuple(path), collisions=tuple(collisions), legs=tuple(legs), end=end)


def _records(info):
    records = [_trace(info['mask'], start) for start in info['starts']]
    occupied = set()
    for record in records:
        footprint = set(record['path']) | set(record['collisions'])
        if footprint & (occupied | info['annotations']):
            raise ValueError("routes or annotation footprints conflict")
        occupied |= footprint
    return records


def _render(info, variant=None):
    canvas = info['terrain'].copy()
    records = ([_trace(info['mask'], s, True) for s in info['starts']]
               if variant == 'turn_at_edge' else info['records'])
    word = info['word']
    if variant == 'deduplicate':
        word = tuple(dict.fromkeys(word))
    offset = 0
    for record in records:
        index = 0
        for j, length in enumerate(record['legs']):
            for k in range(length):
                phase = index
                if variant == 'reset_bends':
                    phase = k
                elif variant == 'consume_markers':
                    phase += j
                elif variant == 'global_phase':
                    phase += offset
                if variant != 'one_pass' or index < len(word):
                    canvas[record['path'][index]] = word[phase % len(word)]
                index += 1
        if variant != 'no_markers':
            for p in record['collisions']:
                canvas[p] = info['marker']
        offset += len(record['path'])
    return canvas[:, ::-1].copy() if info['reflected'] else canvas


def _analyze(grid):
    info = _decode(grid)
    info['records'] = _records(info)
    info['painted'] = _render(info)
    return info


def _crop(grid):
    return grid[:2, :2].copy() if grid.shape[0] % 2 else grid[:2, -2:].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Paint the visible cyclic program first, then crop by original side parity."""
    return _crop(_analyze(input_grid)['painted'])


_ALTERNATIVES = ('deduplicate', 'reset_bends', 'consume_markers', 'global_phase',
                 'no_markers', 'one_pass', 'turn_at_edge')


def _coverage(info):
    correct = _crop(info['painted'])
    return {mode for mode in _ALTERNATIVES
            if not np.array_equal(correct, _crop(_render(info, mode)))}


def _positive_partition(rng, total, count):
    if count == 1:
        return [total]
    cuts = sorted(map(int, rng.choice(np.arange(1, total), count-1, replace=False)))
    return list(np.diff([0]+cuts+[total]).astype(int))


def _plan(start, endpoints, n):
    """Construct geometry as maximal legs, independently of the solver's walk."""
    current, path, collisions, legs = start, [], [], []
    for j, end in enumerate(endpoints):
        dr, dc = ((0, 1), (-1, 0))[j % 2]
        distance = (end[0]-current[0])*dr + (end[1]-current[1])*dc
        if distance < 0 or (current[0]+distance*dr, current[1]+distance*dc) != end:
            raise ValueError("invalid planned leg")
        cells = [(current[0]+k*dr, current[1]+k*dc) for k in range(distance+1)]
        path.extend(cells)
        legs.append(len(cells))
        ahead = (end[0]+dr, end[1]+dc)
        if _inside((n, n), ahead):
            collisions.append(ahead)
        elif j+1 != len(endpoints):
            raise ValueError("cannot turn after exiting the canvas")
        tr, tc = ((-1, 0), (0, 1))[j % 2]
        current = (end[0]+tr, end[1]+tc)
    required_bg = set(collisions)
    if _inside((n, n), current) and _inside((n, n), ahead):
        required_bg.add(current)
    return dict(start=start, path=tuple(path), collisions=tuple(collisions),
                legs=tuple(legs), bg=required_bg)


def _plans(rng, n, legend_row, single, witness):
    low, high = legend_row+4, n-5
    r = int(rng.integers(low, high+1))
    if single:
        end_x = n-2 if rng.random() < .55 else n-1
        points = [(r, 1), (1, 1), (1, end_x)]
        if end_x == n-2:
            points.append((0, end_x))
        return [_plan((r, 0), points, n)], set()
    near_top = 1 if witness else int(rng.integers(0, 2))
    near = _plan((r, 0), [(r, 1), (near_top, 1)], n)
    r2 = int(rng.integers(low+1, high+1)) if high > low else high
    max_h = min(4, r2-low+1, max(1, n//2-4))
    hlegs = int(rng.integers(2, max_h+1)) if max_h >= 2 else 1
    finish_horizontal = not witness and rng.random() < .35
    # The final horizontal exit has one new cell, following a vertical collision.
    target_x = n-2
    displacement = int(rng.integers(max(hlegs, 2), max(max(hlegs, 2), n//2-3)+1))
    start_x = target_x-displacement
    dxs = _positive_partition(rng, displacement, hlegs)
    if hlegs > 1:
        upper = int(rng.integers(low, r2-(hlegs-2)))
        dys = _positive_partition(rng, r2-upper, hlegs-1)
    else:
        dys = []
    rr, cc, points = r2, start_x, []
    for j, dx in enumerate(dxs):
        cc += int(dx)
        points.append((rr, cc))
        if j < hlegs-1:
            rr -= int(dys[j])
            points.append((rr, cc))
    top = 1 if finish_horizontal else (0 if witness else int(rng.integers(0, 2)))
    points.append((top, target_x))
    if finish_horizontal:
        points.append((1, n-1))
    far = _plan((r2, start_x), points, n)
    # Available terrain beyond an upward edge exit must remain unpainted.
    extra_route = {(0, n-1)} if top == 0 else set()
    return [near, far], extra_route


def _word(rng, length, repeated=False):
    distinct = int(rng.integers(2, length if repeated else length+1))
    result = list(range(distinct)) + list(map(int, rng.integers(0, distinct, length-distinct)))
    rng.shuffle(result)
    return tuple(result)


def _construct(rng, n, length, *, single=False, witness=False, reflected=None):
    for _ in range(100):
        palette = tuple(map(int, rng.permutation(10)))
        bg, route, marker = palette[:3]
        labels = _word(rng, length, repeated=witness)
        word = tuple(palette[3+x] for x in labels)
        lr = int(rng.integers(3, min(5, n-12)+1))
        width = 3*length-1
        if width > n-6:
            raise ValueError("legend does not fit")
        lc = int(rng.integers(3, n-2-width))
        columns = [lc+3*j for j in range(length)]
        # Distribute spare width into swatch gaps without breaking the group.
        spare = n-3-(columns[-1]+2)
        for j in range(1, length):
            if spare > 0 and rng.random() < .35:
                columns[j:] = [c+1 for c in columns[j:]]
                spare -= 1
        plans, extra = _plans(rng, n, lr, single, witness)
        must_route = {p for plan in plans for p in plan['path']} | extra
        must_bg = set().union(*(plan['bg'] for plan in plans))
        for plan in plans:
            r, c = plan['start']
            if c > 0:
                must_bg.add((r, c-1))
        mr, mc = n-3, int(rng.integers(1, n-2))
        for ra, ca, rb, cb in ((lr-1, columns[0]-1, lr+3, columns[-1]+3),
                              (mr-1, mc-1, mr+3, mc+3)):
            must_bg.update((r, c) for r in range(ra, min(n, rb))
                           for c in range(ca, min(n, cb)))
        if must_route & must_bg:
            continue
        # Independent broad profiles and rectangular bays give unused terrain
        # real structure; mandatory path/collision cells pin the intended walk.
        knots = np.linspace(0, n-1, int(rng.integers(3, 7)))
        lo = np.rint(np.interp(np.arange(n), knots, rng.uniform(-n*.2, n*.5, len(knots)))).astype(int)
        hi = np.rint(np.interp(np.arange(n), knots, rng.uniform(n*.5, n*1.2, len(knots)))).astype(int)
        cc = np.arange(n)[None, :]
        mask = (cc >= lo[:, None]) & (cc <= hi[:, None])
        for _ in range(int(rng.integers(2, 7))):
            r0, r1 = sorted(map(int, rng.choice(n+1, 2, replace=False)))
            c0, c1 = sorted(map(int, rng.choice(n+1, 2, replace=False)))
            mask[r0:r1, c0:c1] = bool(rng.integers(2))
        for p in must_route:
            mask[p] = True
        for p in must_bg:
            mask[p] = False
        terrain = np.where(mask, route, bg).astype(np.int8)
        inp, painted = terrain.copy(), terrain.copy()
        for c, color in zip(columns, word):
            inp[lr:lr+2, c:c+2] = color
        inp[mr:mr+2, mc:mc+2] = marker
        for plan in plans:
            inp[plan['start']] = word[0]
            for i, p in enumerate(plan['path']):
                painted[p] = word[i % len(word)]
            for p in plan['collisions']:
                painted[p] = marker
        # Frequency is only a syntactic terrain-candidate detector, never a
        # choice between traversable and surrounding terrain.
        if min(np.count_nonzero(inp == bg), np.count_nonzero(inp == route)) <= 4*length+2:
            continue
        flip = bool(rng.integers(2)) if reflected is None else reflected
        if flip:
            inp, painted = inp[:, ::-1].copy(), painted[:, ::-1].copy()
        try:
            info = _analyze(inp)
        except ValueError:
            continue
        if not np.array_equal(painted, info['painted']):
            continue
        # Audit repair: both candidate crops contain actual newly painted route
        # cells, and neither raw cropping nor a fixed-corner choice works.
        if any(np.array_equal(inp[:2, sl], painted[:2, sl]) for sl in (slice(0, 2), slice(-2, None))):
            continue
        if np.array_equal(painted[:2, :2], painted[:2, -2:]):
            continue
        return dict(input=inp, output=_crop(painted))
    raise RuntimeError("bounded construction exhausted")


def _expand_sibling(pair, rng):
    inp = pair['input']
    info = _analyze(inp)
    n = inp.shape[0]
    # Insert a column between the routes and never split a swatch. Append one
    # background row below every start. Both painted top corners stay identical.
    candidates = [c for c in range(n//2-2, n//2+3)
                  if not any(s['col']+1 == c for s in info['swatches'])]
    rng.shuffle(candidates)
    for c in candidates:
        expanded = np.insert(inp, c, info['bg'], axis=1)
        expanded = np.vstack((expanded, np.full((1, n+1), info['bg'], dtype=np.int8)))
        try:
            other = _analyze(expanded)
        except ValueError:
            continue
        if all(np.array_equal(info['painted'][:2, sl], other['painted'][:2, sl])
               and np.array_equal(inp[:2, sl], expanded[:2, sl])
               for sl in (slice(0, 2), slice(-2, None))):
            return dict(input=expanded, output=_crop(other['painted']))
    return None


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(int(seed))
    schedule = int(rng.integers(3))  # longer program / one spanning route / larger square
    for _ in range(300):
        base = int(rng.integers(20, 24 if schedule == 2 else 28))
        length = int(rng.integers(3, min(4, (base-5)//3)+1))
        a = _construct(rng, base, length, witness=True)
        b = _expand_sibling(a, rng)
        if b is None:
            continue
        evidence = _coverage(_analyze(a['input'])) | _coverage(_analyze(b['input']))
        if set(_ALTERNATIVES) <= evidence:
            train = [a, b]
            break
    else:
        raise RuntimeError("could not construct informative parity siblings")
    used = {_key(p['input']) for p in train}
    for split, count in (('train', int(num_train)-2), ('test', int(num_test))):
        pairs = []
        parity_offset = int(rng.integers(2))
        for i in range(count):
            for _ in range(32):
                if split == 'test':
                    parity = (parity_offset+i) % 2
                    lengths = [5, 6] if schedule == 0 else list(range(2, 7))
                    length = int(rng.choice(lengths))
                    low = max(16, 3*length+5, 26 if schedule == 2 else 16)
                    n = int(rng.choice([s for s in range(low, 31) if s % 2 == parity]))
                    single = schedule == 1
                else:
                    length = int(rng.integers(2, 5))
                    n = int(rng.integers(max(16, 3*length+5), 25 if schedule == 2 else 29))
                    single = False if schedule == 1 else bool(rng.integers(2))
                pair = _construct(rng, n, length, single=single)
                key = _key(pair['input'])
                if key not in used:
                    used.add(key)
                    pairs.append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
        if split == 'train':
            train.extend(pairs)
        else:
            test = pairs
    rng.shuffle(train)
    puzzle = dict(train=train, test=test)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, evidence, parities = set(), set(), set()
    sibling_views = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert 16 <= inp.shape[0] <= 30 and inp.shape[0] == inp.shape[1]
            assert out.shape == (2, 2)
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            info = _analyze(inp)
            assert np.array_equal(out, _crop(info['painted']))
            for sl in (slice(0, 2), slice(-2, None)):
                assert not np.array_equal(inp[:2, sl], info['painted'][:2, sl])
            assert not np.array_equal(info['painted'][:2, :2], info['painted'][:2, -2:])
            # Check actual route cells, not only erased symbols or markers.
            actual_route = {p for r in info['records'] for p in r['path']}
            if info['reflected']:
                actual_route = {(r, inp.shape[0]-1-c) for r, c in actual_route}
            assert any(r < 2 and c < 2 and inp[r, c] != info['painted'][r, c]
                       for r, c in actual_route)
            assert any(r < 2 and c >= inp.shape[0]-2 and inp[r, c] != info['painted'][r, c]
                       for r, c in actual_route)
            if split == 'train':
                parities.add(inp.shape[0] % 2)
                evidence |= _coverage(info)
                sibling_views.append((inp.shape[0], info['painted'][:2, :2].tobytes(),
                                      info['painted'][:2, -2:].tobytes(),
                                      inp[:2, :2].tobytes(), inp[:2, -2:].tobytes()))
    assert parities == {0, 1}
    assert set(_ALTERNATIVES) <= evidence, set(_ALTERNATIVES)-evidence
    assert any(abs(a[0]-b[0]) == 1 and a[1:] == b[1:]
               for i, a in enumerate(sibling_views) for b in sibling_views[i+1:]), "missing parity siblings"
    if ne >= 2:
        assert {np.asarray(p['input']).shape[0] % 2 for p in puzzle['test']} == {0, 1}


def _signature(puzzle):
    return tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                 for split in ('train', 'test') for p in puzzle[split])


def self_test():
    """Stream 42 seeds; retain small signatures/statistics, never puzzle files."""
    signatures, normalized, sides, words, roles, counts, leg_counts = set(), set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(hashlib.sha256(repr(_signature(puzzle)).encode()).hexdigest())
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _analyze(pair['input'])
                sides.add(pair['input'].shape[0])
                mapping = {}
                norm = bytes(mapping.setdefault(int(c), len(mapping)) for c in pair['input'].flat)
                normalized.add(hashlib.sha256(norm).hexdigest())
                mapping = {}
                words.add(tuple(mapping.setdefault(c, len(mapping)) for c in info['word']))
                roles.add((info['bg'], info['route'], info['marker'], info['reflected']))
                counts.add(len(info['starts']))
                leg_counts.update(len(r['legs']) for r in info['records'])
                assert np.array_equal(solve(pair['input']), pair['output'])
    assert len(signatures) == 42 and len(normalized) >= 150
    assert len(sides) >= 12 and len(words) >= 15 and len(roles) >= 100
    assert counts == {1, 2} and {2, 3, 4, 6} <= leg_counts
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            generate_puzzle(12345+nt*5+ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (2.0, 1), (True, 1), (4, None), (4, 1.0)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Both edge orientations stop despite available terrain around the corner.
    horizontal = np.zeros((4, 8), dtype=bool)
    horizontal[2, :] = True
    horizontal[1:, -1] = True
    record = _trace(horizontal, (2, 0))
    assert record['path'] == tuple((2, c) for c in range(8))
    assert record['end'] == 'edge' and not record['collisions']
    assert (1, 7) not in record['path'] and (1, 7) in _trace(horizontal, (2, 0), True)['path']
    mask = np.zeros((6, 8), dtype=bool)
    mask[4, :4] = True
    mask[:5, 3] = True
    mask[0, 3:] = True
    record = _trace(mask, (4, 0))
    assert record['path'][-1] == (0, 3) and record['end'] == 'edge'
    assert (0, 4) not in record['path'] and (0, 4) in _trace(mask, (4, 0), True)['path']
    # A blocked turn marks only the forward collision, with no extra turn mark.
    blocked = np.zeros((4, 8), dtype=bool)
    blocked[2, :4] = True
    record = _trace(blocked, (2, 0))
    assert record['path'] == tuple((2, c) for c in range(4))
    assert record['end'] == 'blocked' and record['collisions'] == ((2, 4),)
    # A solved array is a fresh result and the input remains untouched.
    puzzle = generate_puzzle(4321)
    pair = puzzle['test'][0]
    original = pair['input'].copy()
    prediction = solve(pair['input'])
    assert np.array_equal(pair['input'], original)
    assert not np.shares_memory(prediction, pair['input'])
    # Wrong answers must be rejected by reconstruction.
    old = int(pair['output'][0, 0])
    pair['output'][0, 0] = (old+1) % 10
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    pair['output'][0, 0] = old


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': 42}, separators=(",", ":")))
