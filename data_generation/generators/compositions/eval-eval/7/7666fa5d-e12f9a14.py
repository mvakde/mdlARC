"""Dye jets captured by a ribbon inferred from finite diagonal rails.

All rules, including the structural-colored tie seam, are fixed by the
family. No learned color, random state, or generation metadata enters solve.
"""
from collections import deque
import heapq
import numpy as np


OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (-1, 2),
           (0, -1), (0, 2), (1, -1), (1, 2),
           (2, -1), (2, 0), (2, 1), (2, 2))
CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
INF = 10000


def _direction(offset):
    return tuple(-1 if x == -1 else 1 if x == 2 else 0 for x in offset)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= x <= 30 for x in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((g >= 0) & (g <= 9))):
        raise ValueError('expected an integer ARC grid')
    return g.astype(np.int8, copy=False)


def _parse(grid):
    h, w = grid.shape
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if counts.max() * 2 <= grid.size:
        raise ValueError('background must occupy a strict majority')
    launchers = []
    structures = set()
    occupied = np.zeros(grid.shape, bool)
    for r in range(1, h - 2):
        for c in range(1, w - 2):
            color = int(grid[r, c])
            if color == bg or not np.all(grid[r:r+2, c:c+2] == color):
                continue
            ring = [int(grid[r+a, c+b]) for a, b in OFFSETS]
            nonbg = set(ring) - {bg}
            if len(nonbg) != 1 or color in nonbg or not 1 <= ring.count(bg) <= 3:
                continue
            structure = nonbg.pop()
            if occupied[r-1:r+3, c-1:c+3].any():
                raise ValueError('overlapping launchers')
            occupied[r-1:r+3, c-1:c+3] = True
            structures.add(structure)
            launchers.append({'core': (r, c), 'color': color,
                              'ports': [o for o, v in zip(OFFSETS, ring) if v == bg]})
    if not launchers or len(structures) != 1:
        raise ValueError('one common ring structure is required')
    structure = structures.pop()
    rail_cells = np.argwhere((grid != bg) & ~occupied)
    if not len(rail_cells) or np.any(grid[~occupied & (grid != bg)] != structure):
        raise ValueError('rails must share the structural color')
    candidates = []
    for slope in (-1, 1):
        lines = {}
        for r, c in rail_cells:
            lines.setdefault(int(c - slope*r), []).append(int(r))
        rails = []
        good = True
        for q, rows in sorted(lines.items()):
            runs = []
            for r in sorted(rows):
                if not runs or r != runs[-1][-1] + 1:
                    runs.append([])
                runs[-1].append(r)
            for run in runs:
                if len(run) < 2:
                    good = False
                    break
                rails.append((q, 2*run[0] + slope*q, 2*run[-1] + slope*q))
        if good and len(rails) >= 2:
            candidates.append((slope, rails))
    if len(candidates) != 1:
        raise ValueError('the rails must have a uniquely readable diagonal slope')
    slope, rails = candidates[0]
    return {'background': bg, 'structure': structure, 'launchers': launchers,
            'footprints': occupied, 'slope': slope, 'rails': rails}


def _mask(grid, info, variant=''):
    rows, cols = np.indices(grid.shape)
    q = cols - info['slope']*rows
    t = rows + info['slope']*cols
    rails = info['rails']
    domain = np.zeros(grid.shape, bool)
    if variant == 'unbounded':
        domain[:] = True
    elif variant == 'rectangle':
        cells = np.argwhere((grid == info['structure']) & ~info['footprints'])
        low, high = cells.min(axis=0), cells.max(axis=0)
        domain[low[0]:high[0]+1, low[1]:high[1]+1] = True
    elif variant in ('pairwise', 'consecutive', 'pixel_centers'):
        pairs = [(a, b) for i, a in enumerate(rails) for b in rails[i+1:]]
        if variant == 'consecutive':
            pairs = list(zip(rails, rails[1:]))
        for a, b in pairs:
            active = ((t >= max(a[1], b[1])) & (t <= min(a[2], b[2]))
                      & (q > min(a[0], b[0])) & (q < max(a[0], b[0])))
            if variant == 'pixel_centers':
                active &= ((t-a[0]) % 2 == 0) & ((t-b[0]) % 2 == 0)
            domain |= active
    else:
        for level in range(int(t.min()), int(t.max())+1):
            active = [a[0] for a in rails if a[1] <= level <= a[2]]
            if len(active) >= 2:
                domain |= (t == level) & (q > min(active)) & (q < max(active))
    return domain & (grid == info['background']) & ~info['footprints']


def _route(grid, info, domain, launcher, offset, virtual=False):
    r, c = launcher['core']
    a, b = offset
    point = (r+a, c+b)
    direction = _direction(offset)
    path = []
    for tick in range(31):
        r, c = point
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            return None
        if tick and (grid[point] != info['background'] or info['footprints'][point]):
            return None
        if not virtual or tick or grid[point] == info['background']:
            path.append(point)
        if domain[point]:
            return path, tick
        point = (r+direction[0], c+direction[1])
    return None


def _sources(grid, info, domain, virtual=False, strict=True):
    seeds = {}
    traces = []
    used = set()
    diagonals = set()
    for launcher in info['launchers']:
        for offset in (OFFSETS if virtual else launcher['ports']):
            route = _route(grid, info, domain, launcher, offset, virtual)
            if route is None:
                if strict:
                    raise ValueError('every jet must reach the ribbon before a blocker or edge')
                continue
            path, tick = route
            if strict and any(p in used for p in path):
                raise ValueError('jets must be disjoint before capture')
            if strict:
                for p, q in zip(path, path[1:]):
                    if p[0] != q[0] and p[1] != q[1]:
                        midpoint = (p[0]+q[0], p[1]+q[1])
                        sign = 1 if (q[0]-p[0])*(q[1]-p[1]) > 0 else -1
                        if (midpoint, -sign) in diagonals:
                            raise ValueError('jets cross between cells before capture')
                        diagonals.add((midpoint, sign))
            used.update(path)
            color = launcher['color']
            seeds.setdefault(color, []).append((tick, path[-1]))
            traces.append((color, path, tick))
    return seeds, traces


def _distances(domain, sources, variant=''):
    fields = {}
    neighbors = CARDINAL
    if variant == 'eight':
        neighbors = tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0))
    rr, cc = np.indices(domain.shape)
    for color, seeds in sources.items():
        distance = np.full(domain.shape, INF, np.int32)
        if variant == 'manhattan':
            for delay, (r, c) in seeds:
                distance = np.minimum(distance, delay + abs(rr-r) + abs(cc-c))
            distance[~domain] = INF
        else:
            heap = []
            for delay, point in seeds:
                if variant == 'zero_delay':
                    delay = 0
                if delay < distance[point]:
                    distance[point] = delay
                    heapq.heappush(heap, (delay, *point))
            while heap:
                d, r, c = heapq.heappop(heap)
                if d != distance[r, c]:
                    continue
                for a, b in neighbors:
                    x, y = r+a, c+b
                    if (0 <= x < domain.shape[0] and 0 <= y < domain.shape[1]
                            and domain[x, y] and d+1 < distance[x, y]):
                        distance[x, y] = d+1
                        heapq.heappush(heap, (d+1, x, y))
        fields[color] = distance
    return fields


def _paint(grid, info, domain, fields, traces, variant=''):
    out = grid.copy()
    # Generated jets do not intersect. Virtual competing-rule jets can;
    # their pre-capture traces are irrelevant to witness comparisons.
    for color, path, _ in traces:
        for p in path:
            if grid[p] == info['background']:
                out[p] = color
    if not fields:
        return out
    colors = sorted(fields)
    stack = np.stack([fields[c] for c in colors])
    best = stack.min(axis=0)
    tied = (stack == best).sum(axis=0) > 1
    reached = domain & (best < INF)
    winners = np.array(colors, np.int8)[stack.argmin(axis=0)]
    if variant != 'tie_low':
        winners[tied] = info['structure']
    out[reached] = winners[reached]
    return out


def _predict(grid, variant=''):
    info = _parse(grid)
    domain = _mask(grid, info, variant)
    virtual = variant == 'all_ports'
    sources, traces = _sources(grid, info, domain, virtual, not variant)
    fields = _distances(domain, sources, variant)
    if variant == 'jets_only':
        fields = {}
    out = _paint(grid, info, domain, fields, traces, variant)
    return out, info, domain, sources, fields, traces


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Parse visible objects, capture jets, and resolve geodesic dye competition."""
    return _predict(_grid(input_grid))[0]


ALTERNATIVES = ('unbounded', 'rectangle', 'consecutive', 'pixel_centers',
                'all_ports', 'jets_only', 'manhattan', 'eight', 'zero_delay', 'tie_low')


def _witnesses(pair):
    _, info, domain, _, fields, traces = _predict(pair['input'])
    evidence = set()
    for variant in ALTERNATIVES:
        alternative = _predict(pair['input'], variant)[0]
        difference = alternative != pair['output']
        # The aperture witness must change the ribbon partition itself,
        # not merely add a short trace outside a closed ring site.
        if variant == 'all_ports':
            difference &= domain
        if difference.any():
            evidence.add(variant)
    if np.any(domain & (pair['output'] == info['background'])):
        evidence.add('@unreached')
    for launcher in info['launchers']:
        for offset in launcher['ports']:
            evidence.add('@cardinal' if sum(map(abs, _direction(offset))) == 1 else '@diagonal')
    if any(len(l['ports']) > 1 for l in info['launchers']):
        evidence.add('@multiport')
    return evidence


def _rail_scene(rng, mode, split):
    count = 2 if mode == 'simple' else int(rng.integers(3, 6 if mode == 'bridge' else (8 if split == 'test' else 7)))
    span = int(rng.integers(12 if mode == 'simple' else max(18, 3*(count-1)), 27))
    options = list(range(3, span-2))
    rng.shuffle(options)
    qs = [0, span]
    for q in options:
        if len(qs) == count:
            break
        if all(abs(q-a) >= 3 for a in qs):
            qs.append(q)
        if len(qs) == count:
            break
    if len(qs) != count:
        return None
    qs.sort()
    rails = []
    if mode in ('bridge', 'simple'):
        end = int(rng.integers(7 if mode == 'simple' else 10, 15))*2
        short = int(rng.integers(1, count-1)) if count > 2 else -1
        for i, q in enumerate(qs):
            if i in (0, count-1):
                lo, hi = q % 2, end-(q % 2)
            elif i == short:
                lo = 2*int(rng.integers(3, max(4, end//2-3))) + q % 2
                hi = lo+2*int(rng.integers(1, 4))
            else:
                lo = 2*int(rng.integers(0, 5)) + q % 2
                hi = min(end-(q % 2), lo+2*int(rng.integers(3, 11)))
            rails.append((q, lo, hi))
    else:
        for q in qs:
            lo = 2*int(rng.integers(-2, 7))+q % 2
            hi = lo+2*int(rng.integers(3, 15))
            rails.append((q, lo, hi))
    points = [(int((t-q)//2), int((t+q)//2)) for q, lo, hi in rails
              for t in range(lo, hi+1, 2)]
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    dims = high-low+1
    if max(dims) > 28:
        return None
    pad = [int(rng.integers(0, min(4, 30-int(n))+1)) for n in dims]
    shape = tuple(int(n+p) for n, p in zip(dims, pad))
    offset = [int(rng.integers(0, p+1)) for p in pad]
    grid = np.zeros(shape, np.int8)
    for r, c in points:
        grid[r-int(low[0])+offset[0], c-int(low[1])+offset[1]] = 8
    # Use a minimal info record to obtain the raw envelope before docking cores.
    shifted = []
    for q, lo, hi in rails:
        dr, dc = -int(low[0])+offset[0], -int(low[1])+offset[1]
        shifted.append((q+dc-dr, lo+dr+dc, hi+dr+dc))
    info = {'slope': 1, 'rails': shifted, 'background': 0,
            'footprints': np.zeros(shape, bool)}
    mask = _mask(grid, info)
    if mask.sum() < (40 if mode == 'simple' else 100):
        return None
    return grid, mask


def _dock(rng, grid, rawmask, nlaunchers, multi, delay_witness=False):
    h, w = grid.shape
    candidates = [(r, c) for r in range(1, h-2) for c in range(1, w-2)
                  if not grid[r-1:r+3, c-1:c+3].any()
                  and rawmask[r-1:r+3, c-1:c+3].sum() >= 4]
    rng.shuffle(candidates)
    if rng.random() < .45:
        # Partial docking exposes short approach jets as well as immediate
        # capture. Shuffle first so equally exposed sites stay randomized.
        candidates.sort(key=lambda p: int(rawmask[p[0]-1:p[0]+3, p[1]-1:p[1]+3].sum()))
    forced = None
    if delay_witness:
        # Find an exposed docking site whose outward jet crosses a short
        # unbracketed approach before entering a later interval envelope.
        # This is a search over the same random rail scene, not a fixture bank.
        for r, c in candidates:
            footprint = np.zeros(grid.shape, bool)
            footprint[r-1:r+3, c-1:c+3] = True
            domain = rawmask & ~footprint
            options = list(OFFSETS)
            rng.shuffle(options)
            for offset in options:
                route = _route(grid, {'background': 0, 'footprints': footprint}, domain,
                               {'core': (r, c)}, offset, True)
                if route is not None and 2 <= route[1] <= 4:
                    forced = ((r, c), offset)
                    break
            if forced is not None:
                break
        if forced is None:
            return None
    selected = [forced[0]] if forced is not None else []
    for r, c in candidates:
        if any(abs(r-a) <= 4 and abs(c-b) <= 4 for a, b in selected):
            continue
        selected.append((r, c))
        if len(selected) == nlaunchers:
            break
    if len(selected) < nlaunchers:
        return None
    if forced is not None:
        selected = selected[1:] + selected[:1]
    occupied = np.zeros(grid.shape, bool)
    for r, c in selected:
        occupied[r-1:r+3, c-1:c+3] = True
    domain = rawmask & ~occupied
    info = {'background': 0, 'footprints': occupied}
    jets = []
    result = grid.copy()
    for index, (r, c) in enumerate(selected):
        launcher = {'core': (r, c), 'color': index+1}
        available = []
        for offset in OFFSETS:
            route = _route(grid, info, domain, launcher, offset, True)
            if route is not None and 1 <= route[1] <= 4:
                available.append((offset, route[0]))
        rng.shuffle(available)
        if rng.random() < .65:
            available.sort(key=lambda item: -len(item[1]))
        target = int(rng.integers(1, 4)) if multi else 1
        if forced is not None and (r, c) == forced[0]:
            available = [item for item in available if item[0] == forced[1]]
            target = 1
        chosen = []
        for offset, path in available:
            if any(set(path) & set(other) for other in jets):
                continue
            chosen.append(offset)
            jets.append(path)
            if len(chosen) == target:
                break
        if not chosen or (multi and index == 0 and len(chosen) < 2):
            return None
        result[r-1:r+3, c-1:c+3] = 8
        result[r:r+2, c:c+2] = index+1
        for a, b in chosen:
            result[r+a, c+b] = 0
    return result


def _make_pair(rng, mode, split, palette, multi=False, delay_witness=False):
    for _ in range(180):
        scene = _rail_scene(rng, mode, split)
        if scene is None:
            continue
        grid, rawmask = scene
        n = int(rng.choice([1, 2], p=[.7, .3])) if mode == 'simple' else int(rng.choice([2, 3], p=[.65, .35]))
        grid = _dock(rng, grid, rawmask, n, multi, delay_witness)
        if grid is None:
            continue
        transform = int(rng.integers(4))
        grid = np.rot90(grid, transform)
        if rng.integers(2):
            grid = np.fliplr(grid)
        lookup = np.arange(10, dtype=np.int8)
        for old, new in zip((0, 8, 1, 2, 3), palette[:5]):
            lookup[old] = new
        grid = np.ascontiguousarray(lookup[grid])
        try:
            output, info, domain, sources, fields, traces = _predict(grid)
        except ValueError:
            continue
        # Every payload must own a nontrivial portion of the inferred ribbon.
        if any(np.sum((output == l['color']) & domain) < 8 for l in info['launchers']):
            continue
        if delay_witness and np.array_equal(_predict(grid, 'zero_delay')[0], output):
            continue
        return {'input': grid, 'output': output}
    raise RuntimeError('bounded scene construction exhausted')


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    nt, ne = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    base_palette = rng.permutation(10)
    puzzle = {'train': [], 'test': []}
    seen = set()
    needed = {'unbounded', 'all_ports', 'jets_only'}
    for _ in range(ne):
        for attempt in range(60):
            palette = base_palette if rng.random() < .5 else rng.permutation(10)
            pair = _make_pair(rng, 'staggered', 'test', palette, multi=True)
            if _key(pair['input']) not in seen:
                seen.add(_key(pair['input']))
                puzzle['test'].append(pair)
                needed |= _witnesses(pair)
                break
        else:
            raise RuntimeError('unique tests exhausted')
    # Select two geometrically different demonstrations that jointly resolve
    # all explicit alternatives which change any supplied test answer.
    pool = [[], []]
    chosen = None
    for attempt in range(100):
        kind = attempt % 2
        palette = base_palette if kind == 0 else rng.permutation(10)
        delay_witness = (kind == 1 and 'zero_delay' in needed
                         and not any('zero_delay' in e for _, e in pool[1]))
        pair = _make_pair(rng, ('bridge', 'staggered')[kind], 'train', palette,
                          multi=kind == 1, delay_witness=delay_witness)
        if _key(pair['input']) in seen:
            continue
        evidence = _witnesses(pair)
        pool[kind].append((pair, evidence))
        for other, other_evidence in pool[1-kind]:
            if needed <= evidence | other_evidence and _key(pair['input']) != _key(other['input']):
                chosen = (pair, other) if kind == 0 else (other, pair)
                break
        if chosen is not None:
            break
    if chosen is None:
        covered = set().union(*(e for group in pool for _, e in group))
        raise RuntimeError(f'bounded teaching selection exhausted: seed={seed}, uncovered={sorted(needed-covered)}')
    puzzle['train'].extend(chosen)
    seen.update(_key(p['input']) for p in chosen)
    while len(puzzle['train']) < nt:
        for _ in range(60):
            palette = base_palette if rng.random() < .5 else rng.permutation(10)
            mode = str(rng.choice(['simple', 'bridge', 'staggered'], p=[.25, .35, .40]))
            pair = _make_pair(rng, mode, 'train', palette, multi=bool(rng.integers(2)))
            if _key(pair['input']) not in seen:
                puzzle['train'].append(pair)
                seen.add(_key(pair['input']))
                break
        else:
            raise RuntimeError('unique training construction exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_field(domain, seeds, field):
    """Check the shortest-distance Bellman equations at every cell.

    Positive step costs plus source bounds certify distances, including
    disconnected components; this does not re-run the heap search.
    """
    source_bound = np.full(domain.shape, INF, np.int32)
    for delay, point in seeds:
        source_bound[point] = min(source_bound[point], delay)
    expected = source_bound.copy()
    expected[1:, :] = np.minimum(expected[1:, :], field[:-1, :]+1)
    expected[:-1, :] = np.minimum(expected[:-1, :], field[1:, :]+1)
    expected[:, 1:] = np.minimum(expected[:, 1:], field[:, :-1]+1)
    expected[:, :-1] = np.minimum(expected[:, :-1], field[:, 1:]+1)
    expected[~domain] = INF
    assert np.all((field >= 0) & (field <= INF)), 'invalid distances'
    assert np.array_equal(expected, field), 'geodesic recurrence violated'


def validate_generated_puzzle(puzzle):
    """Validate returned grids and conditional demonstration sufficiency."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'illegal counts'
    seen = set()
    evidence = {'train': set(), 'test': set()}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            grid, out = _grid(pair['input']), _grid(pair['output'])
            assert grid.shape == out.shape
            assert _key(grid) not in seen, 'duplicate input'
            seen.add(_key(grid))
            expected, info, domain, sources, fields, traces = _predict(grid)
            assert np.array_equal(domain, _mask(grid, info, 'pairwise')), 'sweep/pair-union mismatch'
            assert np.array_equal(out, expected), 'incorrect output'
            assert np.array_equal(grid[grid != info['background']], out[grid != info['background']]), 'foreground changed'
            assert np.any(grid != out), 'an example must show dye propagation'
            assert len({l['color'] for l in info['launchers']}) == len(info['launchers']), 'payloads must be distinct within a grid'
            for color, field in fields.items():
                _check_field(domain, sources[color], field)
            ray_cells = {p for _, path, _ in traces for p in path}
            allowed = domain.copy()
            for p in ray_cells:
                allowed[p] = True
            for color, path, _ in traces:
                assert all(out[p] == color for p in path[:-1]), 'pre-capture jet color changed'
            assert np.all((grid == out) | allowed), 'dye escaped its ribbon or jets'
            reached = np.min(np.stack(list(fields.values())), axis=0) < INF
            assert np.all(out[domain & ~reached] == info['background']), 'unseeded region changed'
            normalized = {'input': grid, 'output': out}
            evidence[split] |= _witnesses(normalized)
    assert {'unbounded', 'all_ports', 'jets_only'} <= evidence['train'], 'missing parent-contribution evidence'
    assert evidence['test'] <= evidence['train'], 'test depends on an unresolved alternative or emission feature'


def _features(pair):
    out, info, domain, sources, fields, traces = _predict(_grid(pair['input']))
    rails = info['rails']
    stack = np.stack(list(fields.values()))
    best = stack.min(axis=0)
    ties = int(np.sum(domain & (best < INF) & ((stack == best).sum(axis=0) > 1)))
    groups, unseen = [], set(range(len(rails)))
    while unseen:
        component = {unseen.pop()}
        todo = list(component)
        while todo:
            i = todo.pop()
            linked = {j for j in unseen if max(rails[i][1], rails[j][1]) <= min(rails[i][2], rails[j][2])}
            unseen -= linked
            component |= linked
            todo.extend(linked)
        groups.append(component)
    r, c = np.where((pair['input'] == info['structure']) & ~info['footprints'])
    return {'shape': pair['input'].shape, 'rails': len(rails),
            'lengths': tuple((hi-lo)//2+1 for _, lo, hi in rails),
            'components': len(groups), 'slope': info['slope'],
            'domain_components': len(_components_of_mask(domain)),
            'launchers': len(info['launchers']),
            'ports': tuple(len(l['ports']) for l in info['launchers']),
            'headings': tuple(sorted({_direction(o) for l in info['launchers'] for o in l['ports']})),
            'delays': tuple(t for _, _, t in traces), 'ties': ties,
            'unreached': int(np.sum(domain & (out == info['background']))),
            'edge': bool(r.min() == 0 or c.min() == 0 or r.max() == out.shape[0]-1 or c.max() == out.shape[1]-1),
            'palette': (info['background'], info['structure']),
            'witnesses': tuple(sorted(_witnesses(pair)))}


def _fingerprint(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def _components_of_mask(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    components = []
    while unseen:
        start = unseen.pop()
        todo = deque([start])
        component = {start}
        while todo:
            r, c = todo.popleft()
            for a, b in CARDINAL:
                p = (r+a, c+b)
                if p in unseen:
                    unseen.remove(p)
                    component.add(p)
                    todo.append(p)
        components.append(component)
    return components


def self_test():
    """Bounded deterministic coverage audit, fixtures, and rejection checks."""
    # A continuous-endpoint fixture, calculated explicitly rather than by
    # sampling: q=2 and 6, t in [-2,2], has these seven bracketed pixels.
    grid = np.zeros((5, 5), np.int8)
    grid[(0, 1, 2, 2, 3, 4), (2, 1, 0, 4, 3, 2)] = 8
    info = {'slope': -1, 'rails': [(2, -2, 2), (6, -2, 2)],
            'background': 0, 'footprints': np.zeros(grid.shape, bool)}
    expected = np.zeros(grid.shape, bool)
    expected[(1, 1, 2, 2, 2, 3, 3), (2, 3, 1, 2, 3, 1, 2)] = True
    assert np.array_equal(_mask(grid, info), expected)
    assert np.array_equal(_mask(grid, info, 'pairwise'), expected)
    assert not np.array_equal(_mask(grid, info, 'pixel_centers'), expected)
    # A manually enumerable obstacle detour and an isolated, unseeded cell.
    domain = np.ones((5, 7), bool)
    domain[:4, 3] = False
    domain[0, 5] = domain[1, 6] = False
    sources = {1: [(1, (1, 1))], 2: [(1, (1, 5))]}
    fields = _distances(domain, sources)
    assert fields[1][1, 5] == 11  # down 3, across 4, up 3, plus launch cost
    assert fields[1][0, 6] == INF and fields[2][0, 6] == INF
    assert fields[1][4, 3] == fields[2][4, 3] == 6
    for color in fields:
        _check_field(domain, sources[color], fields[color])
    dummy = np.zeros(domain.shape, np.int8)
    painted = _paint(dummy, {'background': 0, 'structure': 8}, domain, fields, [])
    assert painted[4, 3] == 8 and painted[0, 6] == 0
    assert _distances(domain, sources, 'manhattan')[1][1, 5] == 5
    # A corner jet counts diagonal travel as one step per cell, then hands
    # off to cardinal spreading. The only admissible entry is two steps out.
    jet_grid = np.zeros((8, 8), np.int8)
    footprint = np.zeros(jet_grid.shape, bool)
    footprint[0:4, 0:4] = True
    domain = np.zeros(jet_grid.shape, bool)
    domain[5, 5] = True
    route = _route(jet_grid, {'background': 0, 'footprints': footprint}, domain,
                   {'core': (1, 1), 'color': 3}, (2, 2))
    assert route == ([(3, 3), (4, 4), (5, 5)], 2)
    records, puzzles, fingerprints, normalized = [], [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        for split in puzzle:
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair['input']), pair['output'])
                records.append(_features(pair))
                mapping = {}
                normalized.add((pair['input'].shape, tuple(mapping.setdefault(int(c), len(mapping))
                                                          for c in pair['input'].ravel())))
    assert len(fingerprints) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    assert len(normalized) >= 150, 'geometry diversity collapsed'
    assert len({r['shape'] for r in records}) >= 50
    assert {2, 3, 4, 5, 6, 7} <= {r['rails'] for r in records}
    assert {1, 2, 3} <= {r['launchers'] for r in records}
    assert {1, 2, 3} <= {n for r in records for n in r['ports']}
    assert {r['slope'] for r in records} == {-1, 1}
    assert len({d for r in records for d in r['headings']}) == 8
    assert {1, 2} <= {d for r in records for d in r['delays']}
    assert any(r['components'] > 1 for r in records)
    assert any(r['domain_components'] > 1 for r in records)
    assert any(r['unreached'] for r in records)
    assert any(r['ties'] for r in records) and any(not r['ties'] for r in records)
    assert any(r['edge'] for r in records) and any(not r['edge'] for r in records)
    assert len({r['palette'] for r in records}) >= 30
    assert set(ALTERNATIVES) <= {v for r in records for v in r['witnesses']}
    # Regression for the first uncovered travel-cost counterexample.
    regression = generate_puzzle(34, 2, 1)
    if 'zero_delay' in _witnesses(regression['test'][0]):
        assert any('zero_delay' in _witnesses(p) for p in regression['train'])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(314159, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Reject a corrupted output and an otherwise correctly labeled but
    # inadequate demonstration set.
    for seed in range(12):
        p = generate_puzzle(seed, 2, 1)
        if 'tie_low' in _witnesses(p['test'][0]):
            break
    else:
        raise AssertionError('tie-rule regression fixture not found')
    corrupt = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    g = corrupt['test'][0]['output']
    g[0, 0] = (int(g[0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect label accepted')
    # Replace both demonstrations with different one-core simple brackets:
    # labels remain valid, but no training can distinguish a tie rule used
    # by this test, or adequately teach competing apertures.
    rng = np.random.default_rng(99)
    simple = []
    for _ in range(80):
        pair = _make_pair(rng, 'simple', 'train', rng.permutation(10))
        if len(_parse(pair['input'])['launchers']) == 1:
            simple.append(pair)
        if len(simple) == 2:
            break
    inadequate = {'train': simple, 'test': [p['test'][0]]}
    try:
        validate_generated_puzzle(inadequate)
    except AssertionError:
        pass
    else:
        raise AssertionError('insufficient demonstrations accepted')
    return {'seeds_checked': 42, 'pairs_checked': len(records),
            'distinct_shapes': len({r['shape'] for r in records}),
            'normalized_inputs': len(normalized),
            'rail_counts': sorted({r['rails'] for r in records}),
            'launcher_counts': sorted({r['launchers'] for r in records}),
            'semantic_verification': 'self-authored geometry, recurrence, and alternative checks'}
