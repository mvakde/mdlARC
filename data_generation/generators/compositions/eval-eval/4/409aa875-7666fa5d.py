"""Impact windows: five-step arrows delimit intervals on diagonal loom rails.

Only NumPy and the standard library are used. No I/O occurs during generation,
validation, or self_test; diagnostic suites keep just hashes and small features.
"""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np


DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
DISTANCE = 5


def _arrow(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        wings = ((-1, -dc), (1, -dc))
    elif dc == 0:
        wings = ((-dr, -1), (-dr, 1))
    else:
        wings = ((-dr, 0), (0, -dc))
    return tuple(sorted(((r, c), *((r+a, c+b) for a, b in wings))))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a


def _parse(value):
    """Read all roles geometrically from input pixels, never generation state."""
    grid = _grid(value)
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if int(counts.max()) * 2 <= grid.size:
        raise ValueError("The field must occupy a strict majority")
    remaining = set(map(tuple, np.argwhere(grid != bg).tolist()))
    arrows, rails = [], []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        color = int(grid[start])
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            nearby = {(r+a, c+b) for a in (-1, 0, 1) for b in (-1, 0, 1)} & remaining
            nearby = {p for p in nearby if int(grid[p]) == color}
            remaining -= nearby
            component |= nearby
            stack.extend(sorted(nearby))
        cells = tuple(sorted(component))
        if len(cells) == 3:
            matches = [(tip, d) for tip in cells for d in DIRECTIONS
                       if _arrow(tip, d) == cells]
            if len(matches) != 1:
                raise ValueError("Unrecognizable arrowhead")
            tip, d = matches[0]
            endpoint = (tip[0]+DISTANCE*d[0], tip[1]+DISTANCE*d[1])
            if not (0 <= endpoint[0] < grid.shape[0] and 0 <= endpoint[1] < grid.shape[1]):
                raise ValueError("Shot endpoint outside canvas")
            arrows.append(dict(tip=tip, direction=d, cells=cells, endpoint=endpoint, color=color))
        else:
            if len(cells) < 4:
                raise ValueError("Rails have at least four cells")
            r0, c0 = cells[0]
            slopes = [s for s in (-1, 1)
                      if cells == tuple((r0+k, c0+s*k) for k in range(len(cells)))]
            if len(slopes) != 1:
                raise ValueError("A rail must be a straight contiguous diagonal")
            s = slopes[0]
            ts = [r+s*c for r, c in cells]
            rails.append(dict(cells=cells, q=c0-s*r0, lo=min(ts), hi=max(ts),
                              slope=s, color=color))
    if not arrows or len(rails) < 2 or len({x['slope'] for x in rails}) != 1:
        raise ValueError("Need arrows and at least two parallel rails")
    inks = {a['color'] for a in arrows}
    if len(inks) != 1 or inks & {r['color'] for r in rails}:
        raise ValueError("All arrows share one ink, distinct from the rail colors")
    arrows.sort(key=lambda a: a['tip'])
    rails.sort(key=lambda a: (a['q'], a['lo']))
    owner = {p: i for i, rail in enumerate(rails) for p in rail['cells']}
    # Arrowheads can touch differently colored rails, but never each other.
    arrow_cells = {p for a in arrows for p in a['cells']}
    if any(a['endpoint'] in arrow_cells for a in arrows):
        raise ValueError("Shots may terminate on rails or field, not on arrows")
    return dict(grid=grid, background=bg, ink=inks.pop(), slope=rails[0]['slope'],
                arrows=arrows, rails=rails, owner=owner)


def _windows(info, mode="windows", distance=DISTANCE):
    hits = [set() for _ in info['rails']]
    endpoint_counts = {}
    for arrow in info['arrows']:
        p = tuple(arrow['tip'][i]+distance*arrow['direction'][i] for i in (0, 1))
        endpoint_counts[p] = endpoint_counts.get(p, 0)+1
    for arrow in info['arrows']:
        point = tuple(arrow['tip'][i]+distance*arrow['direction'][i] for i in (0, 1))
        if mode == 'cancel_collisions' and endpoint_counts[point] > 1:
            continue
        if mode == 'first_obstacle':
            for step in range(1, DISTANCE+1):
                p = tuple(arrow['tip'][i]+step*arrow['direction'][i] for i in (0, 1))
                if p in info['owner']:
                    point = p
                    break
        if point in info['owner']:
            hits[info['owner'][point]].add(point[0]+info['slope']*point[1])
    result = []
    for rail, levels in zip(info['rails'], hits):
        if mode == 'all_rails' or (mode == 'hit_rails' and levels):
            result.append((rail['q'], rail['lo'], rail['hi']))
        elif levels and (mode != 'ignore_singletons' or len(levels) > 1):
            result.append((rail['q'], min(levels), max(levels)))
        else:
            result.append(None)
    return result, hits


def _sweep(info, mode="windows", distance=DISTANCE):
    out = info['grid'].copy()
    if mode == 'dots':
        for arrow in info['arrows']:
            if out[arrow['endpoint']] == info['background']:
                out[arrow['endpoint']] = info['ink']
        return out
    windows, _ = _windows(info, mode, distance)
    rr, cc = np.indices(out.shape)
    q, t = cc-info['slope']*rr, rr+info['slope']*cc
    for level in range(int(t.min()), int(t.max())+1):
        active = [w[0] for w in windows if w is not None and w[1] <= level <= w[2]
                  and (mode != 'pixel_centers' or (level-w[0]) % 2 == 0)]
        if len(active) < 2:
            continue
        field = (info['grid'] == info['background']) & (t == level)
        if mode == 'same_color':
            colors = {r['color'] for r in info['rails']}
            for color in colors:
                qs = [w[0] for w, rail in zip(windows, info['rails'])
                      if rail['color'] == color and w and w[1] <= level <= w[2]]
                if len(qs) >= 2:
                    out[field & (q > min(qs)) & (q < max(qs))] = info['ink']
        elif mode == 'consecutive':
            # Static adjacency includes unhit and temporarily inactive rails.
            for a, b in zip(windows, windows[1:]):
                if a and b and a[1] <= level <= a[2] and b[1] <= level <= b[2]:
                    out[field & (q > a[0]) & (q < b[0])] = info['ink']
        else:
            out[field & (q > min(active)) & (q < max(active))] = info['ink']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver. Ink and rail slope are visibly encoded; distance is five."""
    return _sweep(_parse(input_grid))


def _label(grid, background, ink, slope, rails, shots):
    """Constructor's label: union pairwise window parallelograms, not a sweep.

    Receives the sampled objects, whereas solve reconstructs them from pixels.
    The two implementations are self-authored consistency checks, not an oracle.
    """
    owner = {p: i for i, rail in enumerate(rails) for p in rail}
    levels = [set() for _ in rails]
    for tip, direction in shots:
        p = (tip[0]+DISTANCE*direction[0], tip[1]+DISTANCE*direction[1])
        if p in owner:
            levels[owner[p]].add(p[0]+slope*p[1])
    rr, cc = np.indices(grid.shape)
    q, t = cc-slope*rr, rr+slope*cc
    paint = np.zeros(grid.shape, dtype=bool)
    for i, j in combinations(range(len(rails)), 2):
        if not levels[i] or not levels[j]:
            continue
        low = max(min(levels[i]), min(levels[j]))
        high = min(max(levels[i]), max(levels[j]))
        a, b = sorted((rails[i][0][1]-slope*rails[i][0][0],
                       rails[j][0][1]-slope*rails[j][0][0]))
        paint |= (t >= low) & (t <= high) & (q > a) & (q < b)
    out = grid.copy()
    out[paint & (grid == background)] = ink
    return out


def _bounds(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _fits(cells):
    a, b, c, d = _bounds(cells)
    return c-a < 30 and d-b < 30


def _place_shots(rng, rails, endpoints):
    """Finite constraint search, at most 3000 nodes, no retained search traces."""
    occupied = {p for rail in rails for p in rail}
    options = []
    for endpoint in endpoints:
        candidates = []
        for index in rng.permutation(len(DIRECTIONS)):
            dr, dc = DIRECTIONS[int(index)]
            tip = (endpoint[0]-DISTANCE*dr, endpoint[1]-DISTANCE*dc)
            cells = _arrow(tip, (dr, dc))
            if not (set(cells) & occupied) and _fits(occupied | set(cells)):
                candidates.append((tip, (dr, dc), cells))
        if not candidates:
            return None
        options.append(candidates)
    options.sort(key=len)
    visits = 0

    def search(index, selected, blocked, all_cells):
        nonlocal visits
        visits += 1
        if visits > 3000:
            return None
        if index == len(options):
            return [(tip, d) for tip, d, cells in selected]
        for candidate in options[index]:
            tip, d, cells = candidate
            if set(cells) & blocked or not _fits(all_cells | set(cells)):
                continue
            halo = {(r+a, c+b) for r, c in cells for a in (-1, 0, 1) for b in (-1, 0, 1)}
            result = search(index+1, selected+[candidate], blocked | halo, all_cells | set(cells))
            if result is not None:
                return result
        return None

    return search(0, [], set(), occupied)


def _layout(rng, profile):
    """Build varied rail intervals, then solve shot placement geometrically."""
    n = 3 if profile in ('bypass', 'singleton') else int(rng.integers(2, 6))
    qs = [0]
    for _ in range(n-1):
        qs.append(qs[-1]+int(rng.integers(4, 9 if n < 4 else 7)))
    rails = []
    regime = int(rng.integers(3)) if profile == 'free' else 0
    order = [int(i) for i in rng.permutation(n)]
    for rank, q in enumerate(qs):
        lo = int(rng.integers(-12, -5))
        if regime == 1:
            lo += 4*order[rank]
        elif regime == 2 and n >= 4:
            lo = -14 if order[rank] < n//2 else 2
        lo -= (lo-q) % 2
        lower = 8 if profile != 'free' else 4
        upper = (14 if n < 4 else 11) if regime == 0 else 8
        length = int(rng.integers(lower, upper))
        rails.append(tuple(((t-q)//2, (t+q)//2) for t in range(lo, lo+2*length, 2)))
    if not _fits({p for rail in rails for p in rail}):
        return None
    endpoints, singleton_points = [], []
    if profile == 'bypass':
        counts = [2, 0, 2]
    elif profile == 'singleton':
        counts = [2, 2, 1]
        if rng.random() < .5:
            counts.reverse()
    else:
        counts = [int(rng.choice((0, 1, 2, 3), p=(.15, .20, .55, .10))) for _ in rails]
        # Two broad windows guarantee useful teaching and constrain packing.
        for i in rng.choice(n, size=2, replace=False):
            counts[int(i)] = 2
    for rail, count in zip(rails, counts):
        if count == 0:
            continue
        if count == 1:
            indices = [len(rail)//2+int(rng.integers(-1, 2))]
            singleton_points.append(rail[indices[0]])
        elif profile in ('bypass', 'singleton'):
            # Keep clear unused tails: strikes, not visible rail ends, bound fill.
            indices = [int(rng.integers(1, 3)), len(rail)-int(rng.integers(2, 4))]
        else:
            indices = sorted(int(x) for x in rng.choice(len(rail), size=count, replace=False))
        endpoints.extend(rail[i] for i in indices)
    # Simultaneous convergent shots sometimes share one rail pixel. Extrema use
    # coordinates, never multiplicity.
    if profile == 'singleton':
        endpoints.append(singleton_points[0])
    elif rng.random() < .28 and len(endpoints) < 9:
        endpoints.append(endpoints[int(rng.integers(len(endpoints)))])
    shots = _place_shots(rng, rails, endpoints)
    if shots is None:
        return None
    if profile == 'bypass' or rng.random() < .4:
        miss = _passing_miss(rng, rails, shots)
        if miss is None and profile == 'bypass':
            return None
        if miss is not None:
            shots.append(miss)
    return rails, shots


def _passing_miss(rng, rails, shots):
    """A real near miss: its ray crosses a rail before its empty endpoint."""
    occupied = {p for rail in rails for p in rail}
    arrows = {p for tip, d in shots for p in _arrow(tip, d)}
    halo = {(r+a, c+b) for r, c in arrows for a in (-1, 0, 1) for b in (-1, 0, 1)}
    choices = [(p, d, k) for rail in rails for p in rail for d in DIRECTIONS for k in (2, 3, 4)]
    for index in rng.permutation(len(choices))[:300]:
        p, d, k = choices[int(index)]
        tip = (p[0]-k*d[0], p[1]-k*d[1])
        endpoint = (tip[0]+DISTANCE*d[0], tip[1]+DISTANCE*d[1])
        cells = set(_arrow(tip, d))
        if endpoint in occupied or endpoint in arrows or cells & (occupied | halo):
            continue
        if _fits(occupied | arrows | cells | {endpoint}):
            return tip, d
    return None


def _make_pair(rng, profile, slope, bg, ink, rail_palette, seen, required=()):
    for attempt in range(160):
        layout = _layout(rng, profile)
        if layout is None:
            continue
        rails, shots = layout
        # Reflection changes the rail slope and all arrow directions together.
        reflect_row = bool(rng.integers(2))
        def point(p):
            r, c = p
            if reflect_row:
                r, c = -r, -c
            return (r, slope*c)
        rails = [tuple(sorted(point(p) for p in rail)) for rail in rails]
        shots = [(point(tip), point(d)) for tip, d in shots]
        cells = {p for rail in rails for p in rail}
        for tip, d in shots:
            cells.update(_arrow(tip, d))
            cells.add((tip[0]+DISTANCE*d[0], tip[1]+DISTANCE*d[1]))
        r0, c0, r1, c1 = _bounds(cells)
        spans = (r1-r0+1, c1-c0+1)
        if max(spans) > 30:
            continue
        shape = tuple(span+int(rng.integers(0, min(30-span, 6)+1)) for span in spans)
        offset = (int(rng.integers(shape[0]-spans[0]+1))-r0,
                  int(rng.integers(shape[1]-spans[1]+1))-c0)
        def move(p):
            return p[0]+offset[0], p[1]+offset[1]
        rails = [tuple(move(p) for p in rail) for rail in rails]
        shots = [(move(tip), d) for tip, d in shots]
        grid = np.full(shape, bg, dtype=np.int8)
        rail_colors = [int(rng.choice(rail_palette)) for _ in rails]
        for rail, color in zip(rails, rail_colors):
            for p in rail:
                grid[p] = color
        for tip, d in shots:
            for p in _arrow(tip, d):
                grid[p] = ink
        key = (grid.shape, grid.tobytes())
        if key in seen:
            continue
        try:
            info = _parse(grid)
        except ValueError:
            continue
        output = _label(grid, bg, ink, slope, rails, shots)
        if not np.any(grid != output):
            continue
        if any(np.array_equal(output, _sweep(info, mode)) for mode in required):
            continue
        return dict(input=grid, output=output)
    raise RuntimeError(f"Bounded shot placement exhausted for {profile}")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Deterministic puzzles with a reflected diagonal orientation at test."""
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(10)]
    train_slope = int(rng.choice((-1, 1)))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            profile = ('bypass', 'singleton')[index] if split == 'train' and index < 2 else 'free'
            required = ('all_rails', 'hit_rails', 'dots', 'consecutive', 'pixel_centers',
                        'same_color', 'first_obstacle') if profile == 'bypass' else ()
            if profile == 'singleton':
                required = ('ignore_singletons', 'cancel_collisions')
            # Role colors vary per example, with two distinct training inks.
            # Rails may be mono- or multicolored, always disjoint from ink/bg.
            ink = palette[index % 10] if split == 'train' else int(rng.choice(palette))
            bg = int(rng.choice([c for c in palette if c != ink]))
            candidates = [c for c in palette if c not in (bg, ink)]
            rail_palette = candidates[:int(rng.integers(2 if profile == 'bypass' else 1, 5))]
            pair = _make_pair(rng, profile, train_slope if split == 'train' else -train_slope,
                              bg, ink, rail_palette, seen, required)
            puzzle[split].append(pair)
            seen.add((pair['input'].shape, pair['input'].tobytes()))
    # Teaching order carries no semantics.
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


ALTERNATIVES = ('all_rails', 'hit_rails', 'dots', 'consecutive', 'pixel_centers',
                'ignore_singletons', 'same_color', 'first_obstacle', 'cancel_collisions')


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, differences = set(), {'train': set(), 'test': set()}
    inks = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = (inp.shape, inp.astype(np.int8).tobytes())
            assert key not in seen, 'Duplicate input'
            seen.add(key)
            info = _parse(inp)
            foreground = inp != info['background']
            assert np.array_equal(out[foreground], inp[foreground]), 'Preserve all input objects'
            assert np.any(out != inp), 'Every generated example has a visible stitch'
            assert np.all(out[out != inp] == info['ink']), 'Use the visible arrow ink'
            assert np.array_equal(out, solve(inp)), 'Incorrect window geometry'
            windows, hits = _windows(info)
            assert sum(w is not None for w in windows) >= 2
            for rail, w in zip(info['rails'], windows):
                if w is not None:
                    assert rail['lo'] <= w[1] <= w[2] <= rail['hi']
            for mode in ALTERNATIVES:
                if not np.array_equal(out, _sweep(info, mode)):
                    differences[split].add(mode)
            if split == 'train':
                inks.add(info['ink'])
    assert len(inks) >= 2, 'Teach ink by role, using two visible literals'
    assert set(ALTERNATIVES) <= differences['train'], 'Missing geometric teaching witness'
    # Recoverability check within the intended vector hypothesis family: visible
    # demonstrations must reject every competing forward distance and reversal.
    train_info = [(_parse(p['input']), np.asarray(p['output'])) for p in puzzle['train']]
    distances = [d for d in range(-29, 30) if d and all(
        np.array_equal(out, _sweep(info, distance=d)) for info, out in train_info)]
    assert distances == [DISTANCE], 'Distance or direction is not taught uniquely'


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _regressions():
    # Hand-calculated fixture, not sampled by the constructor: an unhit middle
    # rail is bypassed, and two outer windows are t=16..22 at q=0 and q=8.
    # Of the 24 interior lattice cells, four belong to the preserved middle rail.
    rails = [tuple((r, r) for r in range(6, 14)),
             tuple((r, r+4) for r in range(4, 12)),
             tuple((r, r+8) for r in range(2, 10))]
    shots = [((13, 8), (-1, 0)), ((11, 6), (0, 1)),
             ((4, 7), (0, 1)), ((12, 15), (-1, 0))]
    grid = np.zeros((18, 20), dtype=np.int8)
    for rail in rails:
        for p in rail:
            grid[p] = 3
    for tip, d in shots:
        for p in _arrow(tip, d):
            grid[p] = 2
    expected = grid.copy()
    twenty = ((7, 9), (5, 11), (8, 9), (7, 10), (6, 11), (5, 12),
              (8, 10), (6, 12), (9, 10), (8, 11), (7, 12), (6, 13),
              (9, 11), (7, 13), (10, 11), (9, 12), (8, 13), (7, 14),
              (10, 12), (8, 14))
    for p in twenty:
        expected[p] = 2
    assert np.count_nonzero(expected != grid) == 20
    assert np.array_equal(solve(grid), expected)
    assert np.array_equal(_label(grid, 0, 2, 1, rails, shots), expected)
    assert np.array_equal(_label(grid, 0, 2, 1, rails[::-1], shots[::-1]), expected)
    for mode in ('all_rails', 'hit_rails', 'dots', 'consecutive', 'pixel_centers', 'first_obstacle'):
        assert not np.array_equal(_sweep(_parse(grid), mode), expected), mode
    for k in range(4):
        assert np.array_equal(solve(np.rot90(grid, k)), np.rot90(expected, k))
    # Convergence at an already hit coordinate cannot widen a window.
    duplicate = ((8, 3), (0, 1))
    more = grid.copy()
    for p in _arrow(*duplicate):
        more[p] = 2
    assert _windows(_parse(grid))[0] == _windows(_parse(more))[0]
    more_expected = more.copy()
    for p in twenty:
        more_expected[p] = 2
    assert np.array_equal(solve(more), more_expected)
    # One hit gives an inclusive zero-length interval. At t=18, a new outer
    # q=12 rail adds exactly (4,14), rather than its full length or no interval.
    one = grid.copy()
    for p in tuple((r, r+12) for r in range(8)):
        one[p] = 3
    for p in _arrow((8, 15), (-1, 0)):
        one[p] = 2
    one_expected = one.copy()
    for p in twenty + ((4, 14),):
        one_expected[p] = 2
    assert np.array_equal(solve(one), one_expected)
    assert not np.array_equal(_sweep(_parse(one), 'ignore_singletons'), one_expected)


def self_test():
    """Bounded streaming checks over 42 seeds; no files or full suite retained."""
    _regressions()
    signatures, shapes, layouts, directions, counts, slopes, inks = (set() for _ in range(7))
    rail_lengths, gaps, arrow_counts, inactive, singleton, repeated = set(), set(), set(), 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = _digest(puzzle)
        assert signature == _digest(generate_puzzle(seed)), 'Nondeterminism'
        signatures.add(signature)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                info = _parse(inp)
                windows, hits = _windows(info)
                shapes.add(inp.shape)
                counts.add(len(info['rails']))
                slopes.add(info['slope'])
                inks.add(info['ink'])
                arrow_counts.add(len(info['arrows']))
                directions.update(a['direction'] for a in info['arrows'])
                rail_lengths.update(len(r['cells']) for r in info['rails'])
                qs = [r['q'] for r in info['rails']]
                gaps.update(b-a for a, b in zip(qs, qs[1:]))
                inactive += any(not h for h in hits)
                singleton += any(len(h) == 1 for h in hits)
                repeated += len({a['endpoint'] for a in info['arrows']}) < len(info['arrows'])
                # Geometry fingerprint ignores palette, translation and padding.
                occupied = inp != info['background']
                rr, cc = np.where(occupied)
                crop = occupied[rr.min():rr.max()+1, cc.min():cc.max()+1]
                layouts.add(hashlib.sha256(bytes(crop.shape)+crop.tobytes()).hexdigest())
                assert np.array_equal(solve(inp.tolist()), pair['output'])
    assert len(signatures) == 42 and len(layouts) >= 150 and len(shapes) >= 35
    assert counts == {2, 3, 4, 5} and slopes == {-1, 1} and directions == set(DIRECTIONS)
    assert len(inks) == 10 and len(rail_lengths) >= 6 and len(gaps) >= 4
    assert len(arrow_counts) >= 5 and min(arrow_counts) <= 4 and max(arrow_counts) >= 8
    assert inactive >= 35 and singleton >= 40 and repeated >= 8
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(500+nt*10+nq, nt, nq)
            assert (len(p['train']), len(p['test'])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    p = generate_puzzle(12, 2, 1)
    p['test'][0]['output'] = p['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupt output accepted')


if __name__ == '__main__':
    self_test()
