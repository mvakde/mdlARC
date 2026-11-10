"""Diamond-cap ribbons represented by the contours of their unshadowed lanes.

Only NumPy and the standard library are used. All interpretation is input-only.
See task_spec.md for the thin diagonal lattice and clipping conventions.
"""
from functools import lru_cache
import json

import numpy as np


VECTORS = ((-1, -1), (-1, 1), (1, 1), (1, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer grid with dimensions 1..30 and colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


@lru_cache(None)
def _outline_offsets(radius):
    return tuple((dr, dc) for dr in range(-radius, radius + 1)
                 for dc in range(-radius, radius + 1) if abs(dr) + abs(dc) == radius)


def _outline(center, radius):
    return [(center[0] + dr, center[1] + dc) for dr, dc in _outline_offsets(radius)]


def _cap(center, radius, direction):
    vr, vc = VECTORS[direction]
    return [(center[0] + vr * (k + 1), center[1] + vc * (radius - k + 1))
            for k in range(radius + 1)]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _base(shape, background):
    return np.asarray(background, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _analyze(value):
    """Read parity palette, complete diamonds and uniform caps from pixels."""
    a = _grid(value)
    parity = np.indices(a.shape).sum(axis=0) % 2
    background = []
    for phase in (0, 1):
        cells = a[parity == phase]
        counts = np.bincount(cells, minlength=10)
        color = int(counts.argmax())
        if counts[color] <= len(cells) // 2:
            raise ValueError("checkerboard phase must have a strict background majority")
        background.append(color)
    if background[0] == background[1]:
        raise ValueError("background must have two distinct checkerboard colors")
    background = tuple(background)
    base = _base(a.shape, background)
    foreground = a != base
    candidates = []
    h, w = a.shape
    for radius in range(2, 6):
        if min(h, w) < 2 * radius + 1:
            continue
        # Only active caps must fit. An inactive side may face a touched edge.
        margin = radius
        ny, nx = h - 2 * margin, w - 2 * margin
        dr, dc = _outline_offsets(radius)[0]
        colors = a[margin + dr:margin + dr + ny, margin + dc:margin + dc + nx]
        ok = (colors != background[0]) & (colors != background[1])
        for dr, dc in _outline_offsets(radius)[1:]:
            ok &= a[margin + dr:margin + dr + ny, margin + dc:margin + dc + nx] == colors
        for y, x in np.argwhere(ok):
            center = (int(y) + margin, int(x) + margin)
            structure = int(colors[y, x])
            caps = {}
            for direction in range(4):
                points = _cap(center, radius, direction)
                if not all(_inside(p, a.shape) for p in points):
                    continue
                values = {int(a[p]) for p in points}
                if len(values) == 1:
                    color = values.pop()
                    if color not in background and color != structure:
                        caps[direction] = color
            if caps:
                candidates.append(dict(center=center, radius=radius,
                                       structure=structure, caps=caps))
    if not 1 <= len(candidates) <= 4:
        raise ValueError("expected one to four recognizable capped diamonds")
    candidates.sort(key=lambda e: e['center'])
    source_mask = np.zeros(a.shape, dtype=bool)
    cap_colors = set()
    for e in candidates:
        points = _outline(e['center'], e['radius'])
        for direction, color in e['caps'].items():
            points += _cap(e['center'], e['radius'], direction)
            cap_colors.add(color)
        for p in points:
            if source_mask[p]:
                raise ValueError("source objects overlap or have ambiguous ownership")
            source_mask[p] = True
    if len(cap_colors) != 1:
        raise ValueError("all caps in an input must share one light color")
    light = cap_colors.pop()
    if any(e['structure'] == light for e in candidates):
        raise ValueError("light and structural colors must differ")
    extras = foreground & ~source_mask
    forbidden = set(background) | {light} | {e['structure'] for e in candidates}
    if any(int(c) in forbidden for c in np.unique(a[extras])):
        raise ValueError("extra objects must use separate opaque colors")
    return dict(grid=a, base=base, background=background, light=light,
                emitters=candidates, foreground=foreground, sources=source_mask, extras=extras)


def _runs(alive):
    result = []
    start = None
    for k, yes in enumerate(list(alive) + [False]):
        if yes and start is None:
            start = k
        elif not yes and start is not None:
            result.append((start, k - 1))
            start = None
    return result


def _trace(a, reverse=False):
    """Compute latent illuminated regions, their contours, and diagnostic masks.

    Off-grid lanes remain conceptually alive. This prevents canvas clipping from
    incorrectly moving a boundary inward. Only original foreground casts shadows.
    """
    grid, foreground = a['grid'], a['foreground']
    contour = np.zeros(grid.shape, dtype=bool)
    lit = np.zeros_like(contour)
    ideal = np.zeros_like(contour)
    original_edges = np.zeros_like(contour)
    effective = np.zeros_like(contour)
    first_hits = np.zeros_like(contour)
    shadow_cells = 0
    split_slices = 0
    interior_hits = 0
    endpoint_hits = 0
    new_edge_streak = 0
    clean_directions = set()
    beams = [(e, d) for e in a['emitters'] for d in e['caps']]
    if reverse:
        beams.reverse()
    for e, direction in beams:
        points = _cap(e['center'], e['radius'], direction)
        vr, vc = VECTORS[direction]
        alive = [True] * len(points)
        streaks = [0] * len(points)
        clean_steps = 0
        for t in range(1, max(grid.shape) + 1):
            section = [(y + t * vr, x + t * vc) for y, x in points]
            visible = [_inside(p, grid.shape) for p in section]
            if not any(visible):
                break
            for k, p in enumerate(section):
                if not visible[k]:
                    continue
                ideal[p] = True
                if k in (0, e['radius']) and not foreground[p]:
                    original_edges[p] = True
                if foreground[p]:
                    effective[p] = True
                    if alive[k]:
                        first_hits[p] = True
                        interior_hits += 0 < k < e['radius']
                        endpoint_hits += k in (0, e['radius'])
                    alive[k] = False
                if alive[k]:
                    lit[p] = True
                elif not foreground[p]:
                    shadow_cells += 1
            runs = _runs(alive)
            split_slices += len(runs) >= 2 and any(visible)
            drawn_indices = set()
            for first, last in runs:
                for k in {first, last}:
                    if visible[k]:
                        contour[section[k]] = True
                        drawn_indices.add(k)
            for k in range(len(points)):
                streaks[k] = streaks[k] + 1 if k in drawn_indices else 0
                if 0 < k < e['radius']:
                    new_edge_streak = max(new_edge_streak, streaks[k])
            if all(visible) and all(alive):
                clean_steps += 1
            if clean_steps >= 2:
                clean_directions.add(direction)
    # No foreground can ever be drawn: a foreground lane dies before extraction.
    assert not np.any(contour & foreground)
    return dict(contour=contour, lit=lit, ideal=ideal, original_edges=original_edges,
                effective=effective, first_hits=first_hits, shadow_cells=int(shadow_cells),
                split_slices=int(split_slices), interior_hits=int(interior_hits),
                endpoint_hits=int(endpoint_hits), new_edge_streak=int(new_edge_streak),
                clean_directions=tuple(sorted(clean_directions)))


def _output(a, trace, mode='contour'):
    out = a['grid'].copy()
    mask = trace[{'contour': 'contour', 'omit_A': 'lit', 'omit_B': 'original_edges'}[mode]]
    out[mask & ~a['foreground']] = a['light']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _analyze(input_grid)
    return _output(a, _trace(a))


def _measure(value):
    a = _analyze(value)
    t = _trace(a)
    # Counterfactual: source objects are preserved but do not cast shadows.
    # This is a targeted ablation, not an independent semantic solver.
    transparent_sources = _trace(dict(a, foreground=a['extras']))
    source_impact = np.count_nonzero((t['contour'] ^ transparent_sources['contour']) & ~a['foreground'])
    groups = []
    for color in np.unique(a['grid'][a['extras']]):
        remaining = set(map(tuple, np.argwhere(a['extras'] & (a['grid'] == color))))
        while remaining:
            start = remaining.pop()
            pending, group = [start], [start]
            while pending:
                y, x = pending.pop()
                for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if p in remaining:
                        remaining.remove(p)
                        pending.append(p)
                        group.append(p)
            groups.append(group)
    return dict(shape=a['grid'].shape, background=a['background'], light=a['light'],
                count=len(a['emitters']), radii=tuple(e['radius'] for e in a['emitters']),
                masks=tuple(sum(1 << d for d in e['caps']) for e in a['emitters']),
                directions=tuple(sorted({d for e in a['emitters'] for d in e['caps']})),
                extras=int(a['extras'].sum()),
                effective_colors=len(np.unique(a['grid'][a['extras'] & t['first_hits']])),
                inert_extras=int(np.count_nonzero(a['extras'] & ~t['effective'])),
                source_hits=int(np.count_nonzero(a['sources'] & t['first_hits'])),
                source_impact=int(source_impact),
                grouped_effective=any(len(g) > 1 and any(t['effective'][p] for p in g) for g in groups),
                new_edges=int(np.count_nonzero(t['contour'] & ~t['original_edges'])),
                lost_edges=int(np.count_nonzero(t['original_edges'] & ~t['contour'])),
                inner_unpainted=int(np.count_nonzero(t['lit'] & ~t['contour'] & ~a['foreground'])),
                shadow_cells=t['shadow_cells'], split_slices=t['split_slices'],
                interior_hits=t['interior_hits'], endpoint_hits=t['endpoint_hits'],
                new_edge_streak=t['new_edge_streak'], clean_directions=t['clean_directions'],
                layout=tuple((e['center'], e['radius'], tuple(e['caps'])) for e in a['emitters']))


def _count(value, name, lower, upper):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lower <= int(value) <= upper):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


def _source_scene(rng, maximum, count, teaching, palette):
    """Free placement of physical objects; virtual ribbons need not be disjoint."""
    if teaching == 'clean':
        radii = [maximum]
    else:
        upper = min(maximum, 3) if count == 4 else maximum
        radii = list(map(int, rng.integers(2, upper + 1, size=count)))
        if teaching == 'blocked':
            radii[0] = maximum
    low = (2 * maximum + 9 if teaching == 'clean' else
           2 * maximum + 15 if teaching == 'blocked' else
           max(2 * max(radii) + 5, (11, 20, 24, 27)[count - 1]))
    h, w = (int(rng.integers(low, 31)) for _ in range(2))
    if rng.random() < .1:
        h = w = 30
    grid = _base((h, w), tuple(map(int, palette[:2])))
    occupied = set()
    specs = []
    for index, radius in enumerate(radii):
        mask = 15 if teaching and index == 0 else int(rng.integers(1, 16))
        caps = [d for d in range(4) if mask & (1 << d)]
        structure = int(palette[2 + (index % 2)])
        for _ in range(180):
            margin = radius + (3 if teaching == 'clean' else 1)
            center = (int(rng.integers(margin, h - margin)),
                      int(rng.integers(margin, w - margin)))
            if teaching == 'blocked' and index == 0:
                center = (int(rng.integers(radius + 4, h - radius - 4)),
                          int(rng.integers(radius + 4, w - radius - 4)))
            points = _outline(center, radius)
            for d in caps:
                points += _cap(center, radius, d)
            cells = set(points)
            if cells & occupied:
                continue
            if any(sum(abs(center[j] - e['center'][j]) for j in (0, 1))
                   < radius + e['radius'] + 3 for e in specs):
                continue
            occupied.update(cells)
            specs.append(dict(center=center, radius=radius, structure=structure,
                              caps={d: int(palette[4]) for d in caps}))
            for p in _outline(center, radius):
                grid[p] = structure
            for d in caps:
                for p in _cap(center, radius, d):
                    grid[p] = int(palette[4])
            break
        else:
            return None
    try:
        a = _analyze(grid)
    except ValueError:
        return None
    if a['emitters'] != sorted(specs, key=lambda e: e['center']):
        return None
    return grid, specs


def _place_obstacles(rng, grid, specs, palette, teaching, style):
    base = _base(grid.shape, tuple(map(int, palette[:2])))
    # The obstructed witness places two colors in known lanes. Acceptance below
    # checks the resulting visible evidence, rather than assuming these succeeded.
    if teaching == 'blocked':
        e = specs[0]
        directions = list(e['caps'])
        rng.shuffle(directions)
        for d in directions:
            vr, vc = VECTORS[d]
            q = _cap(e['center'], e['radius'], d)
            points = [(q[1][0] + 2 * vr, q[1][1] + 2 * vc),
                      (q[-1][0] + 4 * vr, q[-1][1] + 4 * vc)]
            if all(_inside(p, grid.shape) and grid[p] == base[p] for p in points):
                for p, color in zip(points, palette[5:7]):
                    grid[p] = int(color)
                break
        else:
            return None
    if teaching != 'clean':
        groups = (int(rng.integers(0, 3)) if teaching else
                  int(rng.integers(1, 4)) if style == 'light' else int(rng.integers(3, 8)))
        offsets_bank = (((0, 0),), ((0, 0), (1, 0)), ((0, 0), (0, 1)),
                        ((0, 0), (1, 0), (0, 1), (1, 1)),
                        ((0, 0), (1, 0), (1, 1)))
        a = _analyze(grid)
        ideal = _trace(a)['ideal']
        for _ in range(groups):
            offsets = offsets_bank[int(rng.integers(len(offsets_bank)))]
            for _ in range(60):
                y, x = int(rng.integers(grid.shape[0])), int(rng.integers(grid.shape[1]))
                points = [(y + dy, x + dx) for dy, dx in offsets]
                if not all(_inside(p, grid.shape) and grid[p] == base[p] for p in points):
                    continue
                if not any(ideal[p] for p in points):
                    continue
                color = int(rng.choice(palette[5:8]))
                for p in points:
                    grid[p] = color
                break
    # An inert cell is useful evidence that color alone does not cause occlusion.
    a = _analyze(grid)
    inert = np.argwhere(~_trace(a)['ideal'] & ~a['foreground'])
    if teaching != 'clean' or rng.random() < .35:
        if len(inert):
            y, x = inert[int(rng.integers(len(inert)))]
            grid[y, x] = int(palette[5])
    return grid


def _sample_pair(rng, maximum, count, teaching, palette, style):
    for _ in range(240):
        scene = _source_scene(rng, maximum, count, teaching, palette)
        if scene is None:
            continue
        grid, specs = scene
        try:
            grid = _place_obstacles(rng, grid, specs, palette, teaching, style)
            if grid is None:
                continue
            a = _analyze(grid)
            if a['emitters'] != sorted(specs, key=lambda e: e['center']):
                continue
            trace = _trace(a)
            output = _output(a, trace)
            f = _measure(grid)
        except ValueError:
            continue
        if np.array_equal(grid, output):
            continue
        if teaching == 'clean':
            if (f['source_hits'] or f['shadow_cells'] or f['inner_unpainted'] < 8
                    or f['clean_directions'] != (0, 1, 2, 3)):
                continue
        elif teaching == 'blocked':
            if (f['source_hits'] == 0 or f['new_edges'] < 3 or f['lost_edges'] < 2
                    or f['effective_colors'] < 2 or f['inert_extras'] == 0
                    or f['split_slices'] < 2 or f['interior_hits'] == 0
                    or f['new_edge_streak'] < 2 or f['source_impact'] == 0):
                continue
        elif style == 'test':
            if (f['source_hits'] == 0 or f['new_edges'] == 0 or f['shadow_cells'] < 2
                    or f['source_impact'] == 0 or not f['grouped_effective']):
                continue
        return {'input': grid, 'output': output}
    raise RuntimeError("bounded geometric construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    maximum = int(rng.integers(3, 6))
    shared = rng.random() < .25
    palette = rng.permutation(10)
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            teaching = ('clean' if index == 0 else 'blocked' if index == 1 else None) if split == 'train' else None
            n = (1 if teaching == 'clean' else 2 if teaching == 'blocked' else
                 int(rng.integers(2, 5)) if split == 'test' else int(rng.integers(1, 5)))
            colors = palette if shared else rng.permutation(10)
            style = 'test' if split == 'test' else str(rng.choice(['light', 'dense']))
            for _ in range(20):
                pair = _sample_pair(rng, maximum, n, teaching, colors, style)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    features = {'train': [], 'test': []}
    omission_A = omission_B = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            grid, out = _grid(pair['input']), _grid(pair['output'])
            assert grid.shape == out.shape and _key(grid) not in seen
            seen.add(_key(grid))
            a = _analyze(grid)
            t = _trace(a)
            assert np.array_equal(_output(a, t), out), "input-only contour prediction differs"
            assert np.array_equal(out[a['foreground']], grid[a['foreground']])
            assert np.array_equal(out[~t['contour']], grid[~t['contour']])
            assert np.any(out != grid) and np.all(out[out != grid] == a['light'])
            features[split].append(_measure(grid))
            if split == 'train':
                omission_A |= not np.array_equal(_output(a, t, 'omit_A'), out)
                omission_B |= not np.array_equal(_output(a, t, 'omit_B'), out)
    train = features['train']
    clean = [f for f in train if f['shadow_cells'] == 0 and f['source_hits'] == 0
             and f['count'] == 1 and f['directions'] == (0, 1, 2, 3) and f['inner_unpainted'] >= 8
             and f['clean_directions'] == (0, 1, 2, 3)]
    assert clean, "missing clean all-direction endpoint teaching example"
    maximum_taught = max(max(f['radii']) for f in clean)
    assert all(max(f['radii']) <= maximum_taught for f in features['train'] + features['test'])
    assert any(f['source_hits'] > 0 and f['new_edges'] >= 3 and f['lost_edges'] >= 2
               and f['effective_colors'] >= 2 and f['inert_extras'] > 0
               and f['split_slices'] >= 2 and f['interior_hits'] > 0
               and f['source_impact'] > 0 and f['new_edge_streak'] >= 2
               for f in train), "missing shadow teaching evidence"
    assert omission_A and omission_B, "training fails to distinguish a parent omission"
    assert all(f['count'] >= 2 and f['source_hits'] > 0 and f['new_edges'] > 0
               and f['shadow_cells'] >= 2 and f['source_impact'] > 0
               and f['grouped_effective'] for f in features['test'])


def _manual_regressions():
    """Hand-specified lanes establish splitting, phase and clipping behavior."""
    grid = _base((25, 25), (0, 1))
    center, radius = (6, 6), 4
    for p in _outline(center, radius):
        grid[p] = 2
    for p in _cap(center, radius, 2):
        grid[p] = 3
    q = _cap(center, radius, 2)
    point = lambda k, t: (q[k][0] + t, q[k][1] + t)
    clean = solve(grid)
    for t in (1, 2, 6):
        assert clean[point(0, t)] == clean[point(4, t)] == 3
        assert all(clean[point(k, t)] == grid[point(k, t)] for k in (1, 2, 3))
    grid[point(1, 2)] = 4
    grid[point(4, 4)] = 5
    grid[point(1, 6)] = 6  # An already dead lane does not create a wider shadow.
    out = solve(grid)
    for t in (2, 3):
        assert all(out[point(k, t)] == 3 for k in (0, 2, 4))
        assert out[point(3, t)] == grid[point(3, t)]
    for t in (4, 5, 6, 7):
        assert all(out[point(k, t)] == 3 for k in (0, 2, 3))
        assert out[point(4, t)] == grid[point(4, t)]
    assert out[point(1, 2)] == 4 and out[point(4, 4)] == 5 and out[point(1, 6)] == 6
    # Opposite checkerboard phase lies between lane sites and is inert.
    inert = (point(2, 8)[0], point(2, 8)[1] + 1)
    grid[inert] = 7
    out2 = solve(grid)
    mask = np.ones(grid.shape, dtype=bool)
    mask[inert] = False
    assert np.array_equal(out[mask], out2[mask])
    assert out2[inert] == 7
    # Shadows can consume an entire remaining run, then extinguish the beam.
    grid[point(2, 8)] = 8
    grid[point(3, 9)] = 9
    grid[point(0, 11)] = 4
    extinguished = solve(grid)
    for t in (9, 10):
        assert extinguished[point(0, t)] == 3
        assert all(extinguished[point(k, t)] == grid[point(k, t)] for k in (1, 2, 3, 4))
    for t in (11, 12):
        assert all(extinguished[point(k, t)] == grid[point(k, t)] for k in range(5))
    # Clipping the original outer rail does not turn an inner lane into an edge.
    cropped = _base((14, 17), (0, 1))
    for p in _outline(center, radius):
        cropped[p] = 2
    for p in q:
        cropped[p] = 3
    clip_out = solve(cropped)
    assert not _inside(point(4, 3), cropped.shape)
    assert _inside(point(3, 3), cropped.shape)
    assert clip_out[point(3, 3)] == cropped[point(3, 3)]
    # Complete outlines may touch an edge when their active cap faces inward.
    edge_source = _base((15, 17), (0, 1))
    edge_center = (4, 4)
    for p in _outline(edge_center, 4):
        edge_source[p] = 2
    edge_cap = _cap(edge_center, 4, 2)
    for p in edge_cap:
        edge_source[p] = 3
    edge_out = solve(edge_source)
    assert edge_out[edge_cap[0][0] + 2, edge_cap[0][1] + 2] == 3
    assert edge_out[edge_cap[2][0] + 2, edge_cap[2][1] + 2] == edge_source[edge_cap[2][0] + 2, edge_cap[2][1] + 2]
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(grid, rotation)), np.rot90(extinguished, rotation))


def self_test():
    _manual_regressions()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    all_features = [_measure(pair['input']) for p in puzzles for s in ('train', 'test') for pair in p[s]]
    signatures = {tuple(_key(pair['input']) for s in ('train', 'test') for pair in p[s]) for p in puzzles}
    assert len(signatures) == 42
    assert len({f['shape'] for f in all_features}) >= 35
    assert len({f['layout'] for f in all_features}) >= 150
    assert {f['count'] for f in all_features} == {1, 2, 3, 4}
    assert {r for f in all_features for r in f['radii']} == {2, 3, 4, 5}
    assert {m for f in all_features for m in f['masks']} == set(range(1, 16))
    assert sum(f['count'] > 1 and len(set(f['radii'])) > 1 for f in all_features) >= 20
    assert sum(f['source_hits'] > 0 and f['new_edges'] > 0 for f in all_features) >= 70
    assert any(f['shape'] == (30, 30) for f in all_features)
    assert min(min(f['shape']) for f in all_features) <= 15
    assert len({f['background'] for f in all_features}) >= 40
    for seed in (0, 13, 41):
        p = generate_puzzle(seed)
        assert all(_key(a[k]) == _key(b[k]) for s in ('train', 'test')
                   for a, b in zip(puzzles[seed][s], p[s]) for k in ('input', 'output'))
    for p in puzzles:
        for s in ('train', 'test'):
            for pair in p[s]:
                a = _analyze(pair['input'])
                assert np.array_equal(_output(a, _trace(a, reverse=True)), pair['output'])
    for train_count in range(2, 11):
        for test_count in range(1, min(4, train_count - 1) + 1):
            p = generate_puzzle(42 + 10 * train_count + test_count, train_count, test_count)
            assert len(p['train']) == train_count and len(p['test']) == test_count
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in ('train', 'test')}
    pair = bad['test'][0]
    p = tuple(np.argwhere(pair['input'] != pair['output'])[0])
    pair['output'][p] = pair['input'][p]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged label accepted")
    return {'seeds': 42, 'distinct_puzzles': len(signatures),
            'distinct_shapes': len({f['shape'] for f in all_features}),
            'distinct_layouts': len({f['layout'] for f in all_features}),
            'pairs': len(all_features), 'independent_semantic_check': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
