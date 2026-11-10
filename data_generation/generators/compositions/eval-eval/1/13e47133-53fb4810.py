"""Contour looms: a hub's vector tile is indexed by erosion depth and lateral phase.

Only standard-library modules and NumPy are used. Generation retains one puzzle
at a time; neither public solver nor validator reads files or construction state.
"""
from collections import deque
import hashlib
import json

import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _components(mask):
    labels = np.full(mask.shape, -1, dtype=np.int16)
    groups = []
    h, w = mask.shape
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        number = len(groups)
        labels[r, c] = number
        points, queue = [], [(r, c)]
        for a, b in queue:
            points.append((a, b))
            for dr, dc in D4:
                x, y = a + dr, b + dc
                if 0 <= x < h and 0 <= y < w and mask[x, y] and labels[x, y] < 0:
                    labels[x, y] = number
                    queue.append((x, y))
        groups.append(points)
    return labels, groups


def _depths(mask, neighbors=D8):
    """Boundary-seeded distance; canvas exterior is outside the chamber."""
    h, w = mask.shape
    depth = np.full(mask.shape, -1, dtype=np.int16)
    queue = deque()
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if any(not (0 <= r + dr < h and 0 <= c + dc < w)
               or not mask[r + dr, c + dc] for dr, dc in neighbors):
            depth[r, c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in neighbors:
            a, b = r + dr, c + dc
            if 0 <= a < h and 0 <= b < w and mask[a, b] and depth[a, b] < 0:
                depth[a, b] = depth[r, c] + 1
                queue.append((a, b))
    return depth


def _erosion(mask):
    """Separate iterative implementation used in construction and validation."""
    current = mask.copy()
    depth = np.full(mask.shape, -1, dtype=np.int16)
    h, w = mask.shape
    level = 0
    while np.any(current):
        padded = np.pad(current, 1)
        inner = current.copy()
        for dr, dc in D8:
            inner &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        depth[current & ~inner] = level
        current = inner
        level += 1
    return depth


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _slice(origin, direction, width, distance):
    r, c = origin
    dr, dc = direction
    nr, nc = (0, 1) if dr else (1, 0)
    return [(r + distance * dr + j * nr, c + distance * dc + j * nc)
            for j in range(width)]


def _bar(origin, direction, width):
    r, c = origin
    dr, dc = direction
    nr, nc = (0, 1) if dr else (1, 0)
    return [(r - dr + j * nr, c - dc + j * nc) for j in range(-1, width + 1)]


def _period(tile):
    return next(p for p in range(1, len(tile) + 1)
                if len(tile) % p == 0 and np.array_equal(tile, tile[np.arange(len(tile)) % p]))


def _interpret(g, background, wall, hub):
    barrier = (g == wall) | (g == hub)
    assert np.all(barrier[0]) and np.all(barrier[-1])
    assert np.all(barrier[:, 0]) and np.all(barrier[:, -1])
    labels, groups = _components(~barrier)
    assert groups
    regions = [dict(mask=labels == i, depth=_depths(labels == i), source=None)
               for i in range(len(groups))]
    _, bars = _components(g == hub)
    assert bars
    claimed = np.zeros(g.shape, dtype=bool)
    for cells in bars:
        rr, cc = zip(*cells)
        h, w = max(rr) - min(rr) + 1, max(cc) - min(cc) + 1
        assert min(h, w) == 1 and 3 <= max(h, w) <= 6
        assert len(cells) == h * w
        width = max(h, w) - 2
        choices = []
        directions = ((-1, 0), (1, 0)) if h == 1 else ((0, -1), (0, 1))
        for dr, dc in directions:
            origin = (min(rr) + dr + (1 if dc else 0),
                      min(cc) + dc + (1 if dr else 0))
            points = _slice(origin, (dr, dc), width, 0)
            if not all(_inside(g.shape, p) and int(g[p]) not in (background, wall, hub)
                       for p in points):
                continue
            region_id = int(labels[points[0]])
            if not all(labels[p] == region_id for p in points):
                continue
            tile, coords = [], []
            for s in range(30):
                points = _slice(origin, (dr, dc), width, s)
                if not all(_inside(g.shape, p) and labels[p] == region_id for p in points):
                    break
                values = [int(g[p]) for p in points]
                if all(v not in (background, wall, hub) for v in values):
                    assert all(regions[region_id]['depth'][p] == s for p in points)
                    tile.append(values)
                    coords.extend(points)
                else:
                    assert all(v == background for v in values), 'partial source slice'
                    break
            assert tile
            # A blank slice delimits a whole word. It cannot abut another wall.
            tail = _slice(origin, (dr, dc), width, len(tile))
            assert all(_inside(g.shape, p) and labels[p] == region_id
                       and g[p] == background for p in tail)
            choices.append((region_id, dict(origin=origin, direction=(dr, dc), width=width,
                                           tile=np.asarray(tile, dtype=np.int8), cells=coords,
                                           bar=cells)))
        assert len(choices) == 1, 'one inward-facing strip per hub'
        i, source = choices[0]
        assert regions[i]['source'] is None, 'one loom per marked chamber'
        regions[i]['source'] = source
        for p in source['cells']:
            assert not claimed[p]
            claimed[p] = True
    assert np.array_equal(claimed, (g != background) & ~barrier), 'unattached payload'
    return dict(background=background, wall=wall, hub=hub, barrier=barrier, regions=regions)


def _parse(input_grid):
    g = _grid(input_grid)
    wall = int(g[0, 0])
    assert all(int(g[p]) == wall for p in ((0, -1), (-1, 0), (-1, -1)))
    counts = np.bincount(g.ravel(), minlength=10)
    counts[wall] = 0
    background = int(np.argmax(counts))
    assert counts[background] > 0 and np.count_nonzero(counts == counts[background]) == 1
    interpretations = []
    for hub in np.unique(g):
        hub = int(hub)
        if hub in (background, wall):
            continue
        try:
            interpretations.append(_interpret(g, background, wall, hub))
        except AssertionError:
            pass
    assert len(interpretations) == 1, 'color roles must have one geometric interpretation'
    return g, interpretations[0]


def _paint_region(output, region, source, depth=None, tile=None, straight=False):
    rr, cc = np.nonzero(region['mask'])
    r, c = source['origin']
    dr, dc = source['direction']
    tile = source['tile'] if tile is None else tile
    lateral = cc - c if dr else rr - r
    if straight:
        inward = (rr - r) * dr + (cc - c) * dc
    else:
        inward = (region['depth'] if depth is None else depth)[rr, cc]
    output[rr, cc] = tile[inward % len(tile), lateral % tile.shape[1]]


def solve(input_grid: np.ndarray) -> np.ndarray:
    g, analysis = _parse(input_grid)
    output = g.copy()
    for region in analysis['regions']:
        if region['source'] is not None:
            _paint_region(output, region, region['source'])
    return output


def _counterfactuals(g, analysis):
    names = ('straight', 'scalar', 'no_depth_cycle', 'box', 'four_neighbor', 'prefix')
    totals = dict.fromkeys(names, 0)
    for region in analysis['regions']:
        source = region['source']
        if source is None:
            continue
        tile = source['tile']
        mask = region['mask']
        target = mask.copy()
        for p in source['cells']:
            target[p] = False
        correct = g.copy()
        _paint_region(correct, region, source)
        rr, cc = np.nonzero(mask)
        box = np.full(g.shape, -1, dtype=np.int16)
        box[rr, cc] = np.minimum.reduce((rr - rr.min(), rr.max() - rr,
                                        cc - cc.min(), cc.max() - cc))
        short = next(p for p in range(1, len(tile) + 1)
                     if np.array_equal(tile, tile[np.arange(len(tile)) % p]))
        for name in names:
            other = g.copy()
            if name == 'straight':
                _paint_region(other, region, source, straight=True)
            elif name == 'scalar':
                _paint_region(other, region, source, tile=tile[:, :1])
            elif name == 'no_depth_cycle':
                _paint_region(other, region, source, tile=tile[:1])
            elif name == 'prefix':
                _paint_region(other, region, source, tile=tile[:short])
            else:
                depth = box if name == 'box' else _depths(mask, D4)
                _paint_region(other, region, source, depth=depth)
            totals[name] += int(np.count_nonzero(target & (correct != other)))
    return totals


def _architecture(rng, shape, count, fingers, pillar):
    h, w = shape
    wall = np.zeros(shape, dtype=bool)
    wall[0] = wall[-1] = True
    wall[:, 0] = wall[:, -1] = True
    rooms = [(1, h - 1, 1, w - 1)]
    while len(rooms) < count:
        options = [(i, axis) for i, (a, b, c, d) in enumerate(rooms)
                   for axis, size in enumerate((b - a, d - c)) if size >= 11]
        if not options:
            break
        i, axis = options[int(rng.integers(len(options)))]
        a, b, c, d = rooms.pop(i)
        if axis == 0:
            cut = int(rng.integers(a + 5, b - 5))
            wall[cut, c - 1:d + 1] = True
            rooms.extend(((a, cut, c, d), (cut + 1, b, c, d)))
        else:
            cut = int(rng.integers(c + 5, d - 5))
            wall[a - 1:b + 1, cut] = True
            rooms.extend(((a, b, c, cut), (a, b, cut + 1, d)))
    actual = len(rooms)
    for _ in range(fingers):
        for _attempt in range(30):
            points = np.argwhere(wall)
            r, c = map(int, points[int(rng.integers(len(points)))])
            dr, dc = D4[int(rng.integers(4))]
            length = int(rng.integers(1, 5))
            path = [(r + k * dr, c + k * dc) for k in range(1, length + 1)]
            if not all(_inside(shape, p) and not wall[p] for p in path):
                continue
            trial = wall.copy()
            for p in path:
                trial[p] = True
            _, groups = _components(~trial)
            if len(groups) == actual and min(map(len, groups)) >= 20:
                wall = trial
                break
    if pillar:
        depth = _erosion(~wall)
        choices = np.argwhere(depth >= 3)
        if len(choices):
            r, c = map(int, choices[int(rng.integers(len(choices)))])
            ph, pw = int(rng.integers(1, 3)), int(rng.integers(1, 3))
            trial = wall.copy()
            trial[r:r + ph, c:c + pw] = True
            if len(_components(~trial)[1]) == actual:
                wall = trial
    return wall


def _landings(wall, depth, width, length, occupied):
    options = []
    for r, c in zip(*np.nonzero(depth == 0)):
        r, c = int(r), int(c)
        for direction in D4:
            bar = _bar((r, c), direction, width)
            if not all(_inside(wall.shape, p) and wall[p] and p not in occupied for p in bar):
                continue
            good = True
            for s in range(length):
                if not all(_inside(wall.shape, p) and depth[p] == s
                           for p in _slice((r, c), direction, width, s)):
                    good = False
                    break
            if good and all(_inside(wall.shape, p) and depth[p] >= 0
                            for p in _slice((r, c), direction, width, length)):
                options.append(((r, c), direction, bar))
    return options


def _make_tile(rng, palette, width, period, whole=False, rich=False):
    if whole:
        a, b, c, d = [int(v) for v in rng.choice(palette, 4, replace=False)]
        return np.array([[a, b], [c, d], [a, b]], dtype=np.int8)
    if rich:
        return rng.choice(palette, size=(period, width), replace=False).astype(np.int8)
    local = rng.choice(palette, size=int(rng.integers(1, min(len(palette), 5) + 1)), replace=False)
    tile = rng.choice(local, size=(period, width)).astype(np.int8)
    if period >= 2 and _period(tile) == 1:
        tile[-1, -1] = next(int(v) for v in palette if v != int(tile[0, -1]))
    return tile


def _fallback_wall(kind):
    wall = np.zeros((30, 30), dtype=bool)
    wall[0] = wall[-1] = True
    wall[:, 0] = wall[:, -1] = True
    if kind == 0:
        wall[:, 19] = True
        wall[14, 19:] = True
    elif kind == 1:
        wall[:, 16] = True
    wall[12:15, 8] = True
    wall[7, 1:4] = True
    return wall


def _make_pair(rng, kind, background, used, fallback_only=False):
    # kind 0/1 are teaching anchors; 2 is free training; 3 is wider transfer.
    width, period = ((2, 3) if kind == 0 else (3, 2) if kind == 1 else
                     (4, int(rng.integers(2, 4))) if kind == 3 else
                     (int(rng.integers(1, 4)), int(rng.integers(1, 4))))
    colors = [int(c) for c in rng.permutation([c for c in range(10) if c != background])]
    wall_color, hub_color, palette = colors[0], colors[1], colors[2:]
    for attempt in ((63,) if fallback_only else range(64)):
        if attempt == 63:
            wall = _fallback_wall(kind)
        else:
            low = 24 if kind == 0 else 18 if kind in (1, 3) else 11
            shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
            count = 3 if kind == 0 else int(rng.integers(1 if kind != 1 else 2, 5))
            wall = _architecture(rng, shape, count, int(rng.integers(1, 5)),
                                 kind in (0, 1) or rng.random() < .5)
        labels, groups = _components(~wall)
        regions = [dict(mask=labels == i, depth=_erosion(labels == i), source=None)
                   for i in range(len(groups))]
        candidates = []
        for i, region in enumerate(regions):
            if int(region['depth'].max()) >= (4 if kind == 0 else period):
                landings = _landings(wall, region['depth'], width, period, set())
                if landings:
                    candidates.append((i, landings))
        if not candidates or (kind == 0 and len(regions) < 3):
            continue
        # The escape layout reserves its largest chamber for the shaped witness.
        # Choosing a small rectangular chamber could lose the contour evidence.
        if attempt == 63 and kind == 0:
            primary, landings = max(candidates, key=lambda x: int(regions[x[0]]['mask'].sum()))
        else:
            primary, landings = candidates[int(rng.integers(len(candidates)))]
        assignments = [(primary, width, period, landings)]
        remaining = [i for i in range(len(regions)) if i != primary]
        rng.shuffle(remaining)
        occupied = set()
        inp = np.full(wall.shape, background, dtype=np.int8)
        inp[wall] = wall_color
        for j, i in enumerate(remaining):
            # The first anchor always has two different local looms and a blank room.
            if kind == 0 and j >= 1:
                continue
            if kind != 0 and rng.random() < .25:
                continue
            sw = 1 if kind == 0 else int(rng.integers(1, 4))
            sp = 1 if kind == 0 else int(rng.integers(1, 3))
            options = _landings(wall, regions[i]['depth'], sw, sp, set())
            if options:
                assignments.append((i, sw, sp, options))
        failed = False
        for i, sw, sp, options in assignments:
            options = [o for o in options if not any(p in occupied for p in o[2])]
            if not options:
                failed = True
                break
            origin, direction, bar = options[int(rng.integers(len(options)))]
            tile = _make_tile(rng, palette, sw, sp, whole=kind == 0 and i == primary,
                              rich=kind == 1 and i == primary)
            # Optional whole copies preserve exactly the same cyclic word.
            if i != primary and rng.random() < .3:
                longer = _landings(wall, regions[i]['depth'], sw, sp * 2, occupied)
                compatible = [o for o in longer if o[0] == origin and o[1] == direction]
                if compatible:
                    tile = np.concatenate((tile, tile))
            for p in bar:
                inp[p] = hub_color
                occupied.update((p[0] + dr, p[1] + dc) for dr, dc in D4 + ((0, 0),))
            coords = []
            for s, values in enumerate(tile):
                points = _slice(origin, direction, sw, s)
                for p, value in zip(points, values):
                    inp[p] = value
                coords.extend(points)
            source = dict(origin=origin, direction=direction, width=sw, tile=tile,
                          cells=coords, bar=bar)
            regions[i]['source'] = source
        if failed or (kind == 0 and sum(r['source'] is not None for r in regions) < 2):
            continue
        out = inp.copy()
        for region in regions:
            if region['source'] is not None:
                _paint_region(out, region, region['source'])
        try:
            _, analysis = _parse(inp)
            assert np.array_equal(solve(inp), out)
            if kind == 0:
                differences = _counterfactuals(inp, analysis)
                assert all(differences[name] > 0 for name in differences)
            key = (inp.shape, inp.tobytes())
            if key not in used:
                return dict(input=inp, output=out)
        except AssertionError:
            pass
    raise RuntimeError('bounded construction exhausted, including spacious fallback')


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(f'{name} outside {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count('num_train', num_train, 2, 10)
    ne = _count('num_test', num_test, 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    backgrounds = [int(v) for v in rng.permutation(10)]
    puzzle, used = {'train': [], 'test': []}, set()
    for i in range(nt + ne):
        kind = i if i < 2 else 2 if i < nt else 3
        pair = _make_pair(rng, kind, backgrounds[i % 10], used)
        used.add((pair['input'].shape, pair['input'].tobytes()))
        puzzle['train' if i < nt else 'test'].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            _, a = _parse(inp)
            assert np.array_equal(solve(inp), out)
            assert np.array_equal(inp[a['barrier']], out[a['barrier']])
            assert set(np.unique(out)) <= set(np.unique(inp))
            assert np.any(inp != out)
            for region in a['regions']:
                assert np.array_equal(_erosion(region['mask']), region['depth'])
                source = region['source']
                if source is None:
                    assert np.all(out[region['mask']] == a['background'])
                else:
                    assert all(inp[p] == out[p] for p in source['cells'])
                    assert np.all(out[region['mask']] != a['background'])
            records[split].append((inp, a))
    training = [r for _, a in records['train'] for r in a['regions']]
    sources = [r['source'] for r in training if r['source'] is not None]
    assert {1, 2, 3} <= {_period(s['tile']) for s in sources}
    assert {2, 3} <= {s['width'] for s in sources}
    assert any(r['source'] is None for r in training)
    assert any(sum(r['source'] is not None for r in a['regions']) >= 2
               for _, a in records['train'])
    differences = [_counterfactuals(g, a) for g, a in records['train']]
    assert all(any(d[name] > 0 for d in differences) for name in differences[0])
    assert all(any(r['source'] is not None and r['source']['width'] == 4
                   for r in a['regions']) for _, a in records['test'])
    assert all(s['width'] <= 3 for s in sources)
    assert len({a['background'] for split in records for _, a in records[split]}) >= 3


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                g = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def self_test():
    """42 sequential seeds, aggregate measurements only, no disk writes."""
    shapes, roles, directions, widths, periods, counts, geometries, puzzles = (set() for _ in range(8))
    totals = dict.fromkeys(('straight', 'scalar', 'no_depth_cycle', 'box', 'four_neighbor', 'prefix'), 0)
    marked, unmarked, holes, max_depth, palettes, copies = 0, 0, 0, 0, set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                g, a = _parse(pair['input'])
                shapes.add(g.shape)
                roles.add((a['background'], a['wall'], a['hub']))
                counts.add(len(a['regions']))
                palettes.add(len(np.unique(g)))
                geometries.add(hashlib.sha256(a['barrier'].tobytes() + bytes(g.shape)).hexdigest())
                for name, value in _counterfactuals(g, a).items():
                    totals[name] += value
                # Enclosed barrier components beyond the outer frame are pillars.
                holes += max(0, len(_components(a['barrier'])[1]) - 1)
                for r in a['regions']:
                    max_depth = max(max_depth, int(r['depth'].max()))
                    s = r['source']
                    if s is None:
                        unmarked += 1
                    else:
                        marked += 1
                        widths.add(s['width'])
                        periods.add(_period(s['tile']))
                        directions.add(s['direction'])
                        copies += len(s['tile']) > _period(s['tile'])
    assert len(puzzles) == 42 and len(shapes) >= 60 and len(roles) >= 80
    assert len(geometries) >= 120 and counts >= {1, 2, 3, 4}
    assert widths == {1, 2, 3, 4} and periods >= {1, 2, 3}
    assert directions == set(D4) and holes > 20 and unmarked >= 42
    assert max_depth >= 8 and copies > 0 and len(palettes) >= 4
    assert all(v > 0 for v in totals.values())
    # Exercise the bounded escape construction even when ordinary sampling succeeds.
    for kind in range(4):
        pair = _make_pair(np.random.default_rng(700 + kind), kind, 0, set(), fallback_only=True)
        assert np.array_equal(solve(pair['input']), pair['output'])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1000 + nt * 10 + ne, nt, ne)
            assert len(puzzle['train']) == nt and len(puzzle['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(0)
    assert _fingerprint(p) == _fingerprint(generate_puzzle(0))
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged wall accepted')
    return dict(seeds=42, pairs=210, distinct_puzzles=len(puzzles), grid_shapes=len(shapes),
                role_triples=len(roles), barrier_geometries=len(geometries),
                chamber_counts=sorted(counts), directions=sorted(directions),
                widths=sorted(widths), primitive_periods=sorted(periods),
                input_palette_sizes=sorted(palettes), marked_chambers=marked,
                blank_chambers=unmarked, pillar_components=holes, maximum_depth=max_depth,
                multiple_copy_sources=copies, counterfactual_changed_targets=totals)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
