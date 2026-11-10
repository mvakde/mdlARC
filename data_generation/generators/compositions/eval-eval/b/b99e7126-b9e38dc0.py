"""Stencil relays: microscopic offspring offsets, macroscopic light and shadows.

Only NumPy and the standard library are used. The decoder reads all geometry and
roles from the input; construction and prediction use different closure scans.
"""
from collections import deque
from functools import lru_cache

import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _runs(indices):
    result = []
    for i in map(int, indices):
        if result and i == result[-1][1] + 1:
            result[-1] = result[-1][0], i
        else:
            result.append((i, i))
    return result


def _lattice(a):
    d = int(a[0, 0])
    rr = _runs(np.flatnonzero(np.all(a == d, axis=1)))
    cc = _runs(np.flatnonzero(np.all(a == d, axis=0)))
    if (len(rr) < 3 or len(cc) < 3 or rr[0][0] != 0 or cc[0][0] != 0
            or rr[-1][1] != a.shape[0]-1 or cc[-1][1] != a.shape[1]-1):
        raise ValueError("missing complete divider lattice")
    thickness = {e-s+1 for s, e in rr+cc}
    heights = {rr[i+1][0]-rr[i][1]-1 for i in range(len(rr)-1)}
    widths = {cc[i+1][0]-cc[i][1]-1 for i in range(len(cc)-1)}
    if len(thickness) != 1 or len(heights) != 1 or len(widths) != 1:
        raise ValueError("irregular lattice")
    t, h, w = thickness.pop(), heights.pop(), widths.pop()
    if min(h, w) < 2 or max(h, w) > 3 or t not in (1, 2):
        raise ValueError("unsupported tile geometry")
    tiles = np.array([[a[y+1:y+1+h, x+1:x+1+w] for _, x in cc[:-1]]
                      for _, y in rr[:-1]], dtype=np.int8)
    if np.any(tiles == d):
        raise ValueError("divider color occurs inside a tile")
    return tiles, d, t


def _flank(values, sign):
    """Three terminal sections uniquely distinguish periods 1 and 2, and phase."""
    matches = []
    values = np.asarray(values, dtype=int)
    for period in (1, 2):
        for phase in range(period):
            expected = values[-1] + sign*((np.arange(-2, 1)+phase)//period)
            if np.array_equal(values[-3:], expected):
                matches.append((period, phase))
    if len(matches) != 1:
        raise ValueError("terminal rail cadence is ambiguous")
    return matches[0]


def _read_funnel(kinds):
    wall = kinds == 2
    pts = np.argwhere(wall)
    if len(pts) < 9:
        raise ValueError("too little funnel evidence")
    rear, mouth = int(pts[:, 1].min()), int(pts[:, 1].max())
    cap = np.flatnonzero(wall[:, rear])
    if (mouth-rear < 3 or len(cap) < 3 or np.any(np.diff(cap) != 1)
            or mouth == kinds.shape[1]-1):
        raise ValueError("not a closed-back, open-mouth funnel")
    lower, upper = [], []
    for x in range(rear+1, mouth+1):
        rows = np.flatnonzero(wall[:, x])
        if len(rows) != 2 or rows[1]-rows[0] < 2:
            raise ValueError("rails must have one cell on each side")
        lower.append(int(rows[0]))
        upper.append(int(rows[1]))
    if cap[0] != lower[0] or cap[-1] != upper[0]:
        raise ValueError("rear cap and first rails do not meet")
    if any(abs(int(v)) > 1 for v in np.diff(lower)) or any(abs(int(v)) > 1 for v in np.diff(upper)):
        raise ValueError("disconnected rail")
    periods = (_flank(lower, -1), _flank(upper, 1))
    footprint = np.zeros(kinds.shape, dtype=bool)
    bounds = []
    for x in range(rear+1, kinds.shape[1]):
        if x <= mouth:
            lo, hi = lower[x-rear-1], upper[x-rear-1]
        else:
            (p, q), (s, t) = periods
            lo = lower[-1] - (x-mouth+q)//p
            hi = upper[-1] + (x-mouth+t)//s
        footprint[:, x] = (np.arange(kinds.shape[0]) > lo) & (np.arange(kinds.shape[0]) < hi)
        bounds.append((lo, hi))
    if np.any(footprint & wall):
        raise ValueError("rail crosses the interior")
    return footprint, dict(rear=rear, mouth=mouth, periods=periods, bounds=tuple(bounds))


def _illumination(kinds, footprint, ignored_colors=(), solid_values=None):
    lit = footprint.copy()
    for y in range(kinds.shape[0]):
        blocked = False
        for x in range(kinds.shape[1]):
            if blocked:
                lit[y, x] = False
            if kinds[y, x] == 3 and footprint[y, x]:
                if solid_values is None or int(solid_values[y, x]) not in ignored_colors:
                    blocked = True
    return lit


def _relay(kinds, steps, lit):
    """Input-only breadth-first closure. Glyphs are transparent, never opaque."""
    reached = set(map(tuple, np.argwhere(kinds == 1)))
    todo = deque(sorted(reached))
    h, w = kinds.shape
    while todo:
        y, x = todo.popleft()
        for dy, dx in steps:
            yy, xx = y+dy, x+dx
            if (0 <= yy < h and 0 <= xx < w and lit[yy, xx]
                    and kinds[yy, xx] in (0, 1) and (yy, xx) not in reached):
                reached.add((yy, xx))
                todo.append((yy, xx))
    return reached


def _decode(input_grid):
    inp = _grid(input_grid)
    raw, divider, thickness = _lattice(inp)
    records = {}
    for y in range(raw.shape[0]):
        for x in range(raw.shape[1]):
            tile = raw[y, x]
            records.setdefault(_key(tile), [tile, []])[1].append((y, x))
    patterned = [item for item in records.values() if len(np.unique(item[0])) > 1]
    if len(patterned) != 2:
        raise ValueError("need exactly an ordinary pattern and a glyph pattern")
    patterned.sort(key=lambda item: len(item[1]), reverse=True)
    ordinary, ordinary_positions = patterned[0]
    glyph, seeds = patterned[1]
    if len(ordinary_positions) <= len(seeds) or len(np.unique(ordinary)) != 2:
        raise ValueError("ordinary pattern must be uniquely modal among patterned tiles")
    novel = set(map(int, np.unique(glyph))) - set(map(int, np.unique(ordinary)))
    if len(novel) != 2:
        raise ValueError("glyph needs a unique anchor and ink")
    anchors = [c for c in novel if np.count_nonzero(glyph == c) == 1]
    if len(anchors) != 1:
        raise ValueError("anchor must be singleton and ink must occur at least twice")
    anchor = anchors[0]
    ink = (novel-{anchor}).pop()
    if len(np.unique(glyph[(glyph != ink) & (glyph != anchor)])) != 1:
        raise ValueError("glyph background must be constant")
    solid_colors = [int(tile[0, 0]) for tile, _ in records.values() if len(np.unique(tile)) == 1]
    if any(c in set(np.unique(ordinary)) | novel for c in solid_colors):
        raise ValueError("solid roles must have distinct colors")
    matches = []
    for rotation in range(4):
        a = np.rot90(inp, -rotation).copy()
        tiles, _, t = _lattice(a)
        g = np.rot90(glyph, -rotation).copy()
        o = np.rot90(ordinary, -rotation).copy()
        ay, ax = map(int, np.argwhere(g == anchor)[0])
        locations = np.argwhere(g == ink)
        if ax != 0 or np.any(locations[:, 1] <= ax):
            continue
        steps = tuple(sorted((int(y)-ay, int(x)-ax) for y, x in locations))
        for rail_color in solid_colors:
            kinds = np.full(tiles.shape[:2], 3, dtype=np.int8)
            values = tiles[:, :, 0, 0].copy()
            for y in range(kinds.shape[0]):
                for x in range(kinds.shape[1]):
                    tile = tiles[y, x]
                    if np.array_equal(tile, o):
                        kinds[y, x] = 0
                    elif np.array_equal(tile, g):
                        kinds[y, x] = 1
                    elif np.all(tile == rail_color):
                        kinds[y, x] = 2
            try:
                footprint, geometry = _read_funnel(kinds)
            except ValueError:
                continue
            lit = _illumination(kinds, footprint)
            source = kinds == 1
            chamber = footprint.copy()
            chamber[:, geometry['mouth']+1:] = False
            if np.any(source & ~chamber) or np.any(source & ~lit):
                continue
            matches.append(dict(canonical=a, tiles=tiles, ordinary=o, glyph=g,
                                kinds=kinds, footprint=footprint, lit=lit,
                                rotation=rotation, divider=divider, thickness=t,
                                anchor=anchor, ink=ink, steps=steps, values=values,
                                rail=rail_color, **geometry))
    if len(matches) != 1:
        raise ValueError("input must identify one glyph direction and one funnel")
    return matches[0]


def _render(tiles, divider, thickness):
    nr, nc, h, w = tiles.shape
    a = np.full((nr*(h+thickness)+thickness, nc*(w+thickness)+thickness), divider, dtype=np.int8)
    for y in range(nr):
        for x in range(nc):
            yy, xx = thickness+y*(h+thickness), thickness+x*(w+thickness)
            a[yy:yy+h, xx:xx+w] = tiles[y, x]
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    f = _decode(input_grid)
    tiles = f['tiles'].copy()
    for y, x in _relay(f['kinds'], f['steps'], f['lit']):
        tiles[y, x] = f['glyph']
    return np.rot90(_render(tiles, f['divider'], f['thickness']), f['rotation']).copy()


@lru_cache(maxsize=8)
def _geometry_options(h, w, thickness):
    max_rows, max_cols = (30-thickness)//(h+thickness), (30-thickness)//(w+thickness)
    result = []
    for rear in (0, 1):
        for reservoir in (0, 1):
            for length in (3, 4):
                mouth = rear+reservoir+length
                if mouth+2 > max_cols:
                    continue
                for up in (1, 2):
                    for down in (1, 2):
                        for pu in range(up):
                            for pd in range(down):
                                for u in (1, 2):
                                    for d in (1, 2):
                                        top = [u+(j+pu)//up for j in range(length)]
                                        bottom = [d+(j+pd)//down for j in range(length)]
                                        if reservoir:
                                            top = [u+1]+top
                                            bottom = [d+1]+bottom
                                        span = max(top)+max(bottom)+1
                                        if span <= max_rows:
                                            result.append((rear, mouth, tuple(top), tuple(bottom)))
    return tuple(result)


def _sample_geometry(rng, shape, thickness, wide=False):
    h, w = shape
    options = _geometry_options(h, w, thickness)
    rear, mouth, top, bottom = options[int(rng.integers(len(options)))]
    max_rows, max_cols = (30-thickness)//(h+thickness), (30-thickness)//(w+thickness)
    nr = int(rng.integers(max(top)+max(bottom)+1, max_rows+1))
    minimum_columns = max(mouth+2, 6 if wide else 0)
    if minimum_columns > max_cols:
        minimum_columns = max_cols
    nc = int(rng.integers(minimum_columns, max_cols+1))
    center = int(rng.integers(max(top), nr-max(bottom)))
    wall = np.zeros((nr, nc), dtype=bool)
    wall[center-top[0]:center+bottom[0]+1, rear] = True
    for j, (u, d) in enumerate(zip(top, bottom), rear+1):
        wall[center-u, j] = wall[center+d, j] = True
    kinds = np.where(wall, 2, 0).astype(np.int8)
    footprint, geometry = _read_funnel(kinds)
    return kinds, footprint, geometry, center


def _construction_closure(kinds, steps, lit):
    """Construction uses a forward column pass; no prediction/solve call."""
    reached = kinds == 1
    for x in range(kinds.shape[1]):
        for y in np.flatnonzero(reached[:, x]):
            for dy, dx in steps:
                yy, xx = int(y)+dy, x+dx
                if (0 <= yy < kinds.shape[0] and xx < kinds.shape[1]
                        and lit[yy, xx] and kinds[yy, xx] in (0, 1)):
                    reached[yy, xx] = True
    return set(map(tuple, np.argwhere(reached)))


def _features(f):
    kinds, steps, lit = f['kinds'], f['steps'], f['lit']
    seeds = set(map(tuple, np.argwhere(kinds == 1)))
    reached = _relay(kinds, steps, lit)
    added = reached-seeds
    no_shadow = _relay(kinds, steps, f['footprint'])
    shadow_mask = np.zeros(kinds.shape, dtype=bool)
    for y, x in np.argwhere((kinds == 3) & f['footprint']):
        shadow_mask[y, x+1:] = True
    no_cone = _relay(kinds, steps, ~shadow_mask)
    one = seeds.copy()
    for y, x in seeds:
        for dy, dx in steps:
            yy, xx = y+dy, x+dx
            if (0 <= yy < kinds.shape[0] and xx < kinds.shape[1]
                    and lit[yy, xx] and kinds[yy, xx] in (0, 1)):
                one.add((yy, xx))
    bg_mask = (f['glyph'] != f['anchor']) & (f['glyph'] != f['ink'])
    blank = set(map(tuple, np.argwhere(lit & (kinds == 0))))
    effective_colors = sorted(set(map(int, f['values'][(kinds == 3) & f['footprint']])))
    color_effects = sum(_relay(kinds, steps, _illumination(kinds, f['footprint'], (c,), f['values'])) != reached
                        for c in effective_colors)
    blockers = (kinds == 3) & f['footprint']
    return dict(added=len(added), seeds=len(seeds), ink=len(steps), steps=steps,
                recursive=reached != one, selective=bool(blank-added),
                shadow_sensitive=reached != no_shadow, cone_sensitive=reached != no_cone,
                off_axis=any(y not in {r for r, _ in seeds} for y, _ in added),
                full_copy=bool(np.any(f['ordinary'][bg_mask] != f['glyph'][bg_mask])),
                blocker_colors=len(effective_colors), effective_colors=color_effects,
                repeated_lane=bool(np.any(blockers.sum(axis=1) >= 2)),
                grouped=bool(np.any(blockers[:-1] & blockers[1:]) or np.any(blockers[:, :-1] & blockers[:, 1:])),
                shape=f['canonical'].shape, tile_shape=f['glyph'].shape,
                coarse_shape=kinds.shape, periods=f['periods'], thickness=f['thickness'],
                rotation=f['rotation'], glyph=_key(f['glyph'] == f['ink']),
                geometry=(kinds.shape, f['rear'], f['mouth'], f['bounds']),
                palette=(f['divider'], f['rail'], f['anchor'], f['ink'], tuple(np.unique(f['ordinary']))))


def _make_pair(rng, palette, mode):
    # Teaching roles constrain evidence, not the solver's rule.
    if mode == 'clean':
        shape = (int(rng.integers(2, 4)), 3)
        thickness = 1
    elif mode == 'opaque':
        shape, thickness = (3, 3), 1
    else:
        shape = ((2, 2), (2, 3), (3, 2), (3, 3))[int(rng.integers(4))]
        thickness = 2 if shape == (2, 2) and rng.random() < .4 else 1
    h, w = shape
    divider, base, accent, rail, anchor, ink, block_a, block_b = map(int, palette[:8])
    ordinary = np.full(shape, base, dtype=np.int8)
    ordinary.flat[rng.choice(h*w, int(rng.integers(1, h*w)), replace=False)] = accent
    ay = int(rng.integers(h))
    candidates = [(y, x) for y in range(h) for x in range(1, w)]
    if mode == 'clean':
        chosen = {(ay, 2), ((ay+1) % h, 2)}
        if h == 3 and rng.random() < .5:
            chosen.add(((ay+2) % h, 2))
    else:
        n = int(rng.integers(2, len(candidates)+1))
        chosen = {candidates[int(i)] for i in rng.choice(len(candidates), n, replace=False)}
    glyph = np.full(shape, base, dtype=np.int8)
    glyph[ay, 0] = anchor
    for y, x in chosen:
        glyph[y, x] = ink
    bg = (glyph != anchor) & (glyph != ink)
    if not np.any(ordinary[bg] != glyph[bg]):
        ordinary[tuple(np.argwhere(bg)[0])] = accent
        if np.all(ordinary == accent):
            ordinary[ay, 0] = base
    steps = tuple(sorted((y-ay, x) for y, x in chosen))
    kinds, footprint, geom, center = _sample_geometry(rng, shape, thickness, mode in ('clean', 'opaque'))
    nr, nc = kinds.shape
    if mode == 'clean':
        seeds = [(center, geom['rear']+1)]
    else:
        chamber = footprint.copy()
        chamber[:, geom['mouth']+1:] = False
        chamber[:, geom['rear']+3:] = False
        possible = list(map(tuple, np.argwhere(chamber)))
        count = 2 if mode == 'opaque' else int(rng.integers(2, 4) if mode == 'test' else rng.integers(1, 3))
        seeds = [possible[int(i)] for i in rng.choice(len(possible), count, replace=False)]
    for y, x in seeds:
        kinds[y, x] = 1
    clean_reached = _construction_closure(kinds, steps, footprint)
    values = np.full(kinds.shape, -1, dtype=int)
    if mode != 'clean':
        # Pick blockers that have potential offspring downstream in their lane.
        possible = [(y, x) for y, x in map(tuple, np.argwhere(footprint & (kinds == 0)))
                    if all(not (sy == y and sx >= x) for sy, sx in seeds)
                    and any(ry == y and rx > x for ry, rx in clean_reached)]
        if len(possible) < 2:
            raise ValueError("not enough effective blocker lanes")
        n = min(len(possible), int(rng.integers(2, 5) if mode == 'test' else rng.integers(2, 4)))
        selected = [possible[int(i)] for i in rng.choice(len(possible), n, replace=False)]
        for i, (y, x) in enumerate(selected):
            kinds[y, x] = 3
            values[y, x] = block_a if i % 2 == 0 else block_b
        if mode == 'test' and rng.random() < .75:
            y, x = selected[0]
            # A group or a second blocker in the same lane recombines familiar opacity.
            extras = [(y+1, x), (y-1, x), (y, x+1), (y, x+2)]
            for yy, xx in extras:
                if (0 <= yy < nr and 0 <= xx < nc and kinds[yy, xx] == 0
                        and all(not (sy == yy and sx >= xx) for sy, sx in seeds)):
                    kinds[yy, xx] = 3
                    values[yy, xx] = block_a
                    break
    # Inert blockers can have the same color as active ones.
    outside = list(map(tuple, np.argwhere(~footprint & (kinds == 0))))
    for i in rng.choice(len(outside), min(len(outside), int(rng.integers(0, 3))), replace=False):
        y, x = outside[int(i)]
        kinds[y, x] = 3
        values[y, x] = block_a if rng.integers(2) else block_b
    lit = _illumination(kinds, footprint)
    if any(not lit[y, x] for y, x in seeds):
        raise ValueError("shadowed source")
    reached = _construction_closure(kinds, steps, lit)
    tiles = np.broadcast_to(ordinary, (nr, nc, h, w)).copy()
    for y in range(nr):
        for x in range(nc):
            if kinds[y, x] == 1:
                tiles[y, x] = glyph
            elif kinds[y, x] == 2:
                tiles[y, x] = rail
            elif kinds[y, x] == 3:
                tiles[y, x] = values[y, x]
    output = tiles.copy()
    for y, x in reached:
        output[y, x] = glyph
    rotation = int(rng.integers(4))
    pair = {k: np.rot90(_render(v, divider, thickness), rotation).copy()
            for k, v in (('input', tiles), ('output', output))}
    f = _decode(pair['input'])
    info = _features(f)
    if not (info['added'] and info['selective'] and info['full_copy']):
        raise ValueError("insufficient stencil evidence")
    if mode == 'clean' and not info['recursive']:
        raise ValueError("clean example must show multiple generations")
    if mode == 'opaque' and not (info['shadow_sensitive'] and info['cone_sensitive']
                                 and info['effective_colors'] >= 2 and info['off_axis']):
        raise ValueError("insufficient two-color shadow and cone evidence")
    if mode == 'test' and not (info['recursive'] and info['shadow_sensitive']):
        raise ValueError("query must recombine recurrence and shadows")
    return pair


def _fallback_pair(rng, palette, mode, code):
    """Finite constructive reserve, with no stored input/output example lookup."""
    divider, base, accent, rail, anchor, ink, block_a, block_b = map(int, palette[:8])
    ordinary = np.array([accent if code & (1 << i) else base for i in range(9)],
                        dtype=np.int8).reshape(3, 3)
    ordinary[0, 0], ordinary[1, 0] = accent, base
    glyph = np.full((3, 3), base, dtype=np.int8)
    glyph[1, 0] = anchor
    kinds = np.zeros((7, 7), dtype=np.int8)
    if mode == 'clean':
        top, bottom = (1, 2, 3), (1, 2, 3)
        glyph[0, 2] = glyph[1, 2] = ink
        seeds, blocks = [(3, 1)], []
    else:
        top, bottom = (3, 2, 3, 3), (3, 2, 2, 3)
        glyph[:, 1:] = ink
        seeds = [(2, 1), (4, 2)]
        if mode == 'test':
            seeds.append((3, 1))
        blocks = [(2, 2, block_a), (1, 5, block_b), (5, 5, block_b)]
    kinds[3-top[0]:4+bottom[0], 0] = 2
    for x, (u, d) in enumerate(zip(top, bottom), 1):
        kinds[3-u, x] = kinds[3+d, x] = 2
    footprint, _ = _read_funnel(kinds)
    for y, x in seeds:
        kinds[y, x] = 1
    tiles = np.broadcast_to(ordinary, (7, 7, 3, 3)).copy()
    tiles[kinds == 2] = rail
    for y, x in seeds:
        tiles[y, x] = glyph
    for y, x, color in blocks:
        kinds[y, x] = 3
        tiles[y, x] = color
    steps = tuple((int(y)-1, int(x)) for y, x in np.argwhere(glyph == ink))
    output = tiles.copy()
    for y, x in _construction_closure(kinds, steps, _illumination(kinds, footprint)):
        output[y, x] = glyph
    rotation = int(rng.integers(4))
    return {k: np.rot90(_render(v, divider, 1), rotation).copy()
            for k, v in (('input', tiles), ('output', output))}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shared_palette = rng.random() < .25
    palette = rng.permutation(10)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            mode = ('test' if split == 'test' else 'clean' if i == 0 else 'opaque' if i == 1 else 'varied')
            for attempt in range(512):
                colors = palette if shared_palette else rng.permutation(10)
                try:
                    pair = _make_pair(rng, colors, mode)
                except ValueError:
                    continue
                if split == 'train' and i == 1:
                    if _decode(pair['input'])['steps'] == _decode(puzzle['train'][0]['input'])['steps']:
                        continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                # The reserve has 2**7 ordinary motifs, more than the maximum
                # 14 used inputs; both teaching constructions have fixed witnesses.
                for code in range(1, 512):
                    pair = _fallback_pair(rng, palette, mode, code)
                    key = _key(pair['input'])
                    if key not in seen:
                        seen.add(key)
                        puzzle[split].append(pair)
                        break
                else:
                    raise AssertionError('constructive reserve exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            f = _decode(inp)
            features[split].append(_features(f))
            expected = solve(inp)
            assert np.array_equal(expected, out), 'input-derived relay closure disagrees'
            reached = _construction_closure(f['kinds'], f['steps'], f['lit'])
            assert reached == _relay(f['kinds'], f['steps'], f['lit'])
            output_tiles, _, _ = _lattice(np.rot90(out, -f['rotation']))
            for y in range(f['kinds'].shape[0]):
                for x in range(f['kinds'].shape[1]):
                    target = f['glyph'] if (y, x) in reached else f['tiles'][y, x]
                    assert np.array_equal(output_tiles[y, x], target), 'must copy whole glyph or preserve whole tile'
            info = features[split][-1]
            assert info['added'] and info['selective'] and info['full_copy']
            assert 1 <= info['seeds'] <= 3
    train = features['train']
    assert any(f['seeds'] == 1 and f['blocker_colors'] == 0 and f['recursive'] for f in train)
    assert any(f['seeds'] >= 2 and f['effective_colors'] >= 2 and f['cone_sensitive']
               and f['shadow_sensitive'] and f['off_axis'] for f in train)
    assert len({f['steps'] for f in train}) >= 2
    assert all(f['seeds'] >= 2 and f['recursive'] and f['shadow_sensitive'] for f in features['test'])


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, records = set(), []
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            records.extend((split, _features(_decode(p['input']))) for p in puzzle[split])
        if seed in (0, 13, 41):
            again = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in ('train', 'test')
                       for a, b in zip(puzzle[s], again[s]) for k in ('input', 'output'))
    assert len(signatures) == 42
    fs = [f for _, f in records]
    assert len({f['geometry'] for f in fs}) >= 60
    assert len({f['steps'] for f in fs}) >= 30
    assert len({f['shape'] for f in fs}) >= 15
    assert len({f['palette'] for f in fs}) >= 70
    assert {f['tile_shape'] for f in fs} == {(2, 2), (2, 3), (3, 2), (3, 3)}
    assert {f['thickness'] for f in fs} == {1, 2}
    for split in ('train', 'test'):
        assert {f['rotation'] for s, f in records if s == split} == {0, 1, 2, 3}
    assert any(f['grouped'] for s, f in records if s == 'test')
    assert any(f['repeated_lane'] for s, f in records if s == 'test')
    assert any(f['coarse_shape'][0] != f['coarse_shape'][1] for f in fs)
    assert any(f['periods'][0] != f['periods'][1] for f in fs)
    # Rotation, reflection and palette changes preserve the inferred relationship.
    example = puzzles[0]['train'][0]
    palette = np.random.default_rng(910).permutation(10)
    for rotation in range(4):
        inp, out = (np.rot90(example[k], rotation) for k in ('input', 'output'))
        assert np.array_equal(solve(inp), out)
        assert np.array_equal(solve(np.flipud(inp)), np.flipud(out))
        assert np.array_equal(solve(palette[inp]), palette[out])
    reserve_rng = np.random.default_rng(811)
    for mode in ('clean', 'opaque', 'test', 'varied'):
        pair = _fallback_pair(reserve_rng, np.arange(10), mode, 7)
        assert np.array_equal(solve(pair['input']), pair['output'])
        f = _features(_decode(pair['input']))
        assert f['added'] and f['recursive'] and f['selective'] and f['full_copy']
        if mode != 'clean':
            assert f['shadow_sensitive'] and f['cone_sensitive'] and f['effective_colors'] == 2
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(17000+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # A hand-built macro regression: parity, a straight shadow, and an inert
    # exterior obstacle. These expected locations are written without a solver.
    kinds = np.zeros((7, 8), dtype=np.int8)
    kinds[3, 1] = 1
    kinds[3, 4] = 3
    kinds[0, 2] = 3
    footprint = np.zeros(kinds.shape, dtype=bool)
    for x in range(1, 8):
        footprint[max(0, 4-x):min(7, 3+x), x] = True
    lit = _illumination(kinds, footprint)
    assert _relay(kinds, ((0, 2),), lit) == {(3, 1), (3, 3)}
    assert _relay(kinds, ((0, 2),), footprint) == {(3, 1), (3, 3), (3, 5), (3, 7)}
    assert not lit[3, 7] and lit[1, 7]
    wall = np.zeros((7, 7), dtype=np.int8)
    wall[2:5, 0] = 2
    for x, lo in enumerate((2, 2, 1), 1):
        wall[lo, x] = wall[6-lo, x] = 2
    beam, geometry = _read_funnel(wall)
    assert geometry['periods'] == ((2, 0), (2, 0))
    assert not beam[1, 4] and beam[2, 4]
    assert not beam[0, 5] and beam[1, 5]
    # A handwritten whole-grid case outside the sampler's thickness regime:
    # 2x3 interiors and two-cell dividers, with an off-center fine anchor.
    ordinary = np.array([[0, 1, 1], [0, 1, 0]], dtype=np.int8)
    glyph = np.array([[3, 0, 4], [0, 4, 0]], dtype=np.int8)
    tiles = np.broadcast_to(ordinary, (7, 5, 2, 3)).copy()
    tiles[(wall[:, :5] == 2)] = 2
    tiles[3, 1] = glyph
    tiles[6, 2] = 5  # Outside the footprint: its lane stays inert.
    expected = tiles.copy()
    expected[3, 3] = expected[4, 4] = glyph
    assert np.array_equal(solve(_render(tiles, 9, 2)), _render(expected, 9, 2))
    tiles[4, 3] = 5  # Inside the footprint: target (4,4) is now shadowed.
    expected = tiles.copy()
    expected[3, 3] = glyph
    assert np.array_equal(solve(_render(tiles, 9, 2)), _render(expected, 9, 2))
    # Reject a plausible ink-only-copy label by changing a copied background pixel.
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    pair = damaged['test'][0]
    f = _decode(pair['input'])
    y, x = sorted(_relay(f['kinds'], f['steps'], f['lit'])
                  - set(map(tuple, np.argwhere(f['kinds'] == 1))))[0]
    bg = (f['glyph'] != f['ink']) & (f['glyph'] != f['anchor'])
    dy, dx = map(int, np.argwhere(bg & (f['ordinary'] != f['glyph']))[0])
    canonical_out = np.rot90(pair['output'], -f['rotation']).copy()
    h, w = f['glyph'].shape
    t = f['thickness']
    canonical_out[t+y*(h+t)+dy, t+x*(w+t)+dx] = f['ordinary'][dy, dx]
    pair['output'] = np.rot90(canonical_out, f['rotation']).copy()
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
