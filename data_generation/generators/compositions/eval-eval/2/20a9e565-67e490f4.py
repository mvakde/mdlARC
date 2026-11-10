"""The hidden electorate: extrapolate trails, select whole voters, fill a stencil.

Only standard-library modules and NumPy are used. Labels are constructed from
explicit hypothetical objects; solve fits returned pixels without sampling state.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import numpy as np


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        origin = min(remaining)
        remaining.remove(origin)
        stack, cells = [origin], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _NEIGHBORS:
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
        result.append(tuple(sorted(cells)))
    return result


def _normal(cells):
    lo_r = min(r for r, c in cells)
    lo_c = min(c for r, c in cells)
    return tuple(sorted((int(r - lo_r), int(c - lo_c)) for r, c in cells))


def _transform(cells, operation):
    """One fixed D4 operation, about an object's bounding-box center."""
    turns, mirror = operation
    result = []
    for r, c in cells:
        if mirror:
            c = -c
        for _ in range(turns):
            r, c = c, -r
        result.append((r, c))
    return tuple(sorted(result))


_OPERATIONS = tuple((k, mirror) for mirror in (False, True) for k in range(4))


@lru_cache(maxsize=2048)
def _canonical(cells):
    return min(_normal(_transform(cells, op)) for op in _OPERATIONS)


def _centered(cells):
    rs, cs = zip(*cells)
    assert (min(rs) + max(rs)) % 2 == 0
    assert (min(cs) + max(cs)) % 2 == 0
    r = (min(rs) + max(rs)) // 2
    c = (min(cs) + max(cs)) // 2
    return (r, c), tuple(sorted((y - r, x - c) for y, x in cells))


def _rot_key(cells):
    return min(_normal(_transform(cells, (k, False))) for k in range(4))


@lru_cache(maxsize=1)
def _shape_pool():
    """Enumerate geometry, not stored scenes: odd-box, hole-free small shapes."""
    pool = set()
    for bits in range(1, 512):
        cells = tuple((i // 3, i % 3) for i in range(9) if bits & (1 << i))
        raw = _normal(cells)
        h, w = max(r for r, c in raw) + 1, max(c for r, c in raw) + 1
        if h % 2 == 0 or w % 2 == 0 or len(raw) > 7:
            continue
        mask = np.zeros((h + 2, w + 2), dtype=bool)
        for r, c in raw:
            mask[r + 1, c + 1] = True
        if len(_components(mask)) != 1 or len(_components(~mask)) != 1:
            continue
        pool.add(_canonical(raw))
    return tuple(sorted(pool, key=lambda s: (len(s), s)))


def _marker_cells(bounds):
    r0, r1, c0, c1 = bounds
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _find_markers(grid):
    candidates = []
    for color in np.unique(grid):
        cells = set(map(tuple, np.argwhere(grid == color)))
        if len(cells) != 6:
            continue
        rs, cs = zip(*cells)
        bounds = (min(rs), max(rs), min(cs), max(cs))
        r0, r1, c0, c1 = bounds
        if r1 <= r0 or c1 <= c0 or r1 - r0 + c1 - c0 < 4:
            continue
        if cells == _marker_cells(bounds) and len(_components(grid == color)) == 2:
            candidates.append((int(color), tuple(map(int, bounds))))
    assert len(candidates) == 1, 'ambiguous or malformed L aperture'
    return candidates[0]


def _grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))
    return grid.astype(int, copy=False)


def _parse(grid):
    marker, bounds = _find_markers(grid)
    values, counts = np.unique(grid, return_counts=True)
    assert sum(counts == max(counts)) == 1, 'background must be uniquely most frequent'
    bg = int(values[np.argmax(counts)])
    assert bg != marker
    r0, r1, c0, c1 = bounds
    assert np.all(np.isin(grid[r0:r1 + 1, c0:c1 + 1], (bg, marker)))
    rectangles = []
    for color in values:
        if int(color) in (bg, marker):
            continue
        cells = tuple(map(tuple, np.argwhere(grid == color)))
        if len(cells) < 8:
            continue
        rs, cs = zip(*cells)
        y0, y1, x0, x1 = min(rs), max(rs), min(cs), max(cs)
        box = grid[y0:y1 + 1, x0:x1 + 1]
        if min(box.shape) < 3 or set(map(int, np.unique(box))) != {bg, int(color)}:
            continue
        if all((np.all(box[0] == color), np.all(box[-1] == color),
                np.all(box[:, 0] == color), np.all(box[:, -1] == color))):
            if len(_components(box == color)) == 1:
                rectangles.append((int(color), (y0, y1, x0, x1)))
    assert len(rectangles) == 1, 'exactly one connected rectangular carrier is required'
    carrier, stencil = rectangles[0]
    sy0, sy1, sx0, sx1 = stencil
    box = grid[sy0:sy1 + 1, sx0:sx1 + 1].copy()
    cavities = _components(box == bg)
    assert cavities
    for cells in cavities:
        _, raw = _centered(cells)
        assert max(max(abs(r), abs(c)) for r, c in raw) <= 1
    clean = grid.copy()
    clean[sy0:sy1 + 1, sx0:sx1 + 1] = bg
    clean[clean == marker] = bg
    objects = []
    for cells in _components(clean != bg):
        colors = {int(clean[r, c]) for r, c in cells}
        assert len(colors) == 1, 'voters must be isolated monochromatic objects'
        center, raw = _centered(cells)
        assert max(max(abs(r), abs(c)) for r, c in raw) <= 1
        objects.append((colors.pop(), center, raw))
    assert objects
    return dict(background=bg, marker=marker, bounds=bounds, carrier=carrier,
                stencil=stencil, box=box, cavities=cavities, objects=objects)


def _fit_axis(data, axis):
    """Recover shared rank coordinates and noncrossing parallel trails."""
    # Canonical axis 1 means increasing columns. A transpose supplies axis 0.
    swap = axis == 0
    groups = defaultdict(list)
    for color, (r, c), raw in data['objects']:
        transverse, axial = (c, r) if swap else (r, c)
        shape = tuple(sorted((x, y) for y, x in raw)) if swap else raw
        groups[axial].append((transverse, color, shape))
    starts = sorted(groups)
    if not 3 <= len(starts) <= 5:
        return None
    k = len(groups[starts[0]])
    if not 1 <= k <= 4 or any(len(groups[x]) != k for x in starts):
        return None
    ranks = [sorted(groups[x]) for x in starts]
    differences = np.diff(starts)
    acceleration = int(differences[1] - differences[0])
    if not np.all(np.diff(differences) == acceleration):
        return None
    if not -1 <= acceleration <= 1 or min(differences) < 4:
        return None
    b = data['bounds']
    bounds = (b[2], b[3], b[0], b[1]) if swap else b
    all_cells = [(center[axis] + point[axis])
                 for _, center, raw in data['objects'] for point in raw]
    forward = min(all_cells) > bounds[3]
    backward = max(all_cells) < bounds[2]
    # Names refer to the visible observations' side; prediction may use i<0.
    if not (forward or backward):
        return None
    tracks = []
    for j in range(k):
        observations = [rank[j] for rank in ranks]
        transverse = [o[0] for o in observations]
        drift = transverse[1] - transverse[0]
        if abs(drift) > 1 or any(y != transverse[0] + i * drift
                                for i, y in enumerate(transverse)):
            return None
        shapes = [o[2] for o in observations]
        if len({_canonical(raw) for raw in shapes}) != 1:
            return None
        operations = []
        for op in _OPERATIONS:
            current = shapes[0]
            for expected in shapes[1:]:
                current = _transform(current, op)
                if current != expected:
                    break
            else:
                operations.append(op)
        if not operations:
            return None
        colors = [o[1] for o in observations]
        periods = [p for p in range(1, min(3, len(colors) - 1) + 1)
                   if len(set(colors[:p])) == p
                   and all(color == colors[i % p] for i, color in enumerate(colors))]
        if not periods:
            return None
        tracks.append(dict(y=transverse[0], drift=drift, shape=shapes[0],
                           operations=operations, palette=tuple(colors[:min(periods)]),
                           key=_canonical(shapes[0])))
    return dict(axis=axis, bounds=bounds, x=starts[0], step=int(differences[0]),
                acceleration=acceleration, visible=len(starts), tracks=tracks,
                direction=('up' if swap else 'left') if forward else
                          ('down' if swap else 'right'))


def _extrapolated(model):
    """Finite relevant indices, including skipped ranks; no latent canvas."""
    r0, r1, c0, c1 = model['bounds']
    n = model['visible']
    indices = []
    if c0 > model['x']:
        candidates = range(n, 35)
    else:
        candidates = range(-1, -35, -1)
    for i in candidates:
        x = model['x'] + i * model['step'] + i * (i - 1) // 2 * model['acceleration']
        gap_index = i - 1 if i >= n else i
        gap = model['step'] + gap_index * model['acceleration']
        if gap < 4:
            break
        if (i >= n and x - 1 > c1) or (i < 0 and x + 1 < c0):
            break
        if x + 1 >= c0 and x - 1 <= c1:
            indices.append((i, x))
    voters = []
    for j, track in enumerate(model['tracks']):
        for i, x in indices:
            possible = []
            for op in track['operations']:
                shape = track['shape']
                # D4 operations have order at most four. This also handles i<0.
                for _ in range(i % 4):
                    shape = _transform(shape, op)
                possible.append(shape)
            assert len(set(possible)) == 1, 'observed orientations leave hidden geometry ambiguous'
            shape = possible[0]
            y = track['y'] + i * track['drift']
            cells = tuple((y + r, x + c) for r, c in shape)
            inside = all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cells)
            center_inside = r0 <= y <= r1 and c0 <= x <= c1
            intersects = any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cells)
            voters.append(dict(key=track['key'], color=track['palette'][i % len(track['palette'])],
                               cells=cells, inside=inside, center_inside=center_inside,
                               intersects=intersects, track=j, index=i))
    return voters


def _ballots(voters, membership='inside', equivalence='shape'):
    votes = defaultdict(Counter)
    for voter in voters:
        if voter[membership]:
            key = len(voter['key']) if equivalence == 'area' else voter['key']
            votes[key][voter['color']] += 1
    return dict(votes)


def _fill(data, votes, equivalence='shape'):
    output = data['box'].copy()
    winners = {}
    for cells in data['cavities']:
        key = len(cells) if equivalence == 'area' else _canonical(_normal(cells))
        assert key in votes and votes[key], 'a cavity has no eligible vote'
        top = max(votes[key].values())
        colors = [color for color, count in votes[key].items() if count == top]
        assert len(colors) == 1, 'eligible shape ballot is tied'
        winners[key] = colors[0]
        for r, c in cells:
            output[r, c] = colors[0]
    return output, winners


def _analyze(input_grid):
    grid = _grid(input_grid)
    data = _parse(grid)
    models = [model for axis in (1, 0) if (model := _fit_axis(data, axis)) is not None]
    assert models, 'observations do not establish a supported trail construction'
    answers = []
    cavity_keys = {_canonical(_normal(cells)) for cells in data['cavities']}
    for model in models:
        assert {t['key'] for t in model['tracks']} == cavity_keys
        voters = _extrapolated(model)
        votes = _ballots(voters)
        output, winners = _fill(data, votes)
        answers.append((output, model, voters, votes, winners))
    assert all(np.array_equal(a[0], answers[0][0]) for a in answers), 'ambiguous progression axis'
    output, model, voters, votes, winners = answers[0]
    data.update(model=model, voters=voters, votes=votes, winners=winners)
    return output, data


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the completed stencil entirely from the given input."""
    return _analyze(input_grid)[0]


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _cavity_layout(rng, classes, available_height, teaching=False, fallback=False):
    if fallback:
        # The conservative pool is constructible for every margin variant.
        placements = []
        for j, key in enumerate(classes):
            raw = _transform(_centered(key)[1], _pick(rng, _OPERATIONS))
            placements.append((key, tuple((2 + r, 2 + 4 * j + c) for r, c in raw)))
        return 5, 4 * len(classes) + 1, placements
    height = int(rng.integers(5, min(7 if teaching else 9, available_height) + 1))
    limit = 7 if height >= 9 else 4
    n = len(classes) if teaching else int(rng.integers(len(classes), limit + 1))
    chosen = list(classes) + [_pick(rng, classes) for _ in range(n - len(classes))]
    rng.shuffle(chosen)
    minimum_width = 4 * ((n + 1) // 2 if height >= 9 else n) + 1
    width = minimum_width + int(rng.integers(0, min(3, 19 - minimum_width) + 1))
    available = np.ones((height, width), dtype=bool)
    available[[0, -1], :] = False
    available[:, [0, -1]] = False
    placements = []
    for key in chosen:
        raw = _transform(_centered(key)[1], _pick(rng, _OPERATIONS))
        positions = [(y, x) for y in range(1, height - 1) for x in range(1, width - 1)
                     if all(0 <= y + r < height and 0 <= x + c < width
                            and available[y + r, x + c] for r, c in raw)]
        if not positions:
            return None
        y, x = _pick(rng, positions)
        cells = tuple((y + r, x + c) for r, c in raw)
        placements.append((key, cells))
        for r, c in cells:
            available[max(0, r - 1):min(height, r + 2),
                      max(0, c - 1):min(width, c + 2)] = False
    return height, width, placements


def _build_pair(rng, config, mode=None, fallback_variant=None):
    pool = _shape_pool()
    teaching = mode in ('pool', 'clip', 'far')
    if teaching:
        classes = config['anchors']
        first, second = classes
        if mode in ('pool', 'far'):
            trail_classes = [first, second, first, first]
            visible, hidden, skip = 3, 2, int(mode == 'far')
            cycles = [(0, 1), (3,), (2, 1), (0, 1)]
        else:
            trail_classes = [first, second, first]
            visible, hidden, skip = 4, 1, 1
            cycles = [(0, 1, 2), (3,), (1,)]
        step = 4
        acceleration = int(mode == 'pool')
        drift = [0] * len(trail_classes)
        if mode == 'clip':
            drift[-1] = 1 if fallback_variant is not None else int(_pick(rng, (-1, 1)))
    else:
        visible = int(rng.integers(3, 6))
        hidden = int(rng.integers(1, 4))
        skip = int(rng.integers(0, 2))
        k = int(rng.integers(1, 5))
        class_count = int(rng.integers(1, min(3, k) + 1))
        classes = tuple(pool[int(i)] for i in rng.choice(len(pool), class_count, replace=False))
        trail_classes = list(classes) + [_pick(rng, classes) for _ in range(k - class_count)]
        rng.shuffle(trail_classes)
        cycles = []
        for _ in trail_classes:
            p = int(rng.integers(1, min(3, visible - 1) + 1))
            cycles.append(tuple(map(int, rng.choice(7, p, replace=False))))
        acceleration = int(_pick(rng, (-1, 0, 0, 0, 1)))
        step = int(rng.integers(4, 7))
        drift = [0 if k == 4 else int(_pick(rng, (-1, 0, 0, 0, 1))) for _ in trail_classes]
    total = visible + skip + hidden
    xs = [1 + i * step + i * (i - 1) // 2 * acceleration for i in range(total)]
    if min(np.diff(xs)) < 4 or xs[-1] + 2 > 30:
        return None
    colors = list(map(int, rng.permutation(10)))
    bg, marker, carrier = colors[:3]
    vote_colors = colors[3:]
    if mode == 'clip' and vote_colors[2] == config.get('first_winner'):
        vote_colors[2], vote_colors[4] = vote_colors[4], vote_colors[2]
    tracks, y_cursor = [], 1
    for j, (key, cycle, dy) in enumerate(zip(trail_classes, cycles, drift)):
        raw = _centered(key)[1]
        # All trails of the teaching chiral class have one handedness; the
        # cavity will be explicitly drawn in the other handedness below.
        base_op = (int(rng.integers(4)), False) if teaching else _pick(rng, _OPERATIONS)
        raw = _transform(raw, base_op)
        op = (int(rng.integers(4)), False) if fallback_variant is None else (0, False)
        band = 3 + abs(dy) * (total - 1)
        y = y_cursor + (total - 1 if dy < 0 else 0)
        tracks.append(dict(key=key, shape=raw, operation=op, y=y, drift=dy,
                           palette=tuple(vote_colors[i] for i in cycle)))
        y_cursor += band + (1 if fallback_variant is not None else int(rng.integers(1, 3)))
    # y_cursor is the next band's first center; +1 keeps a blank separating row.
    stencil_y = y_cursor + 1
    if stencil_y + 5 > 30:
        return None
    layout = _cavity_layout(rng, classes, 30 - stencil_y, teaching,
                            fallback_variant is not None)
    if layout is None:
        return None
    sh, sw, cavities = layout
    if teaching:
        changed = []
        for key, cells in cavities:
            center, _ = _centered(cells)
            if key == classes[0]:
                raw = _transform(_centered(key)[1], (int(rng.integers(4)), True))
                cells = tuple((center[0] + r, center[1] + c) for r, c in raw)
            changed.append((key, cells))
        cavities = changed
    near = set()
    for _, cells in cavities:
        if any(cell in near for cell in cells):
            return None
        for r, c in cells:
            near.update((r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    if stencil_y + sh > 30:
        return None
    hypothetical = []
    for j, track in enumerate(tracks):
        raw = track['shape']
        for i, x in enumerate(xs):
            y = track['y'] + i * track['drift']
            cells = tuple((y + r, x + c) for r, c in raw)
            hypothetical.append(dict(key=track['key'], color=track['palette'][i % len(track['palette'])],
                                     cells=cells, track=j, index=i))
            raw = _transform(raw, track['operation'])
    targets = [v for v in hypothetical if v['index'] >= visible + skip]
    c0, c1 = xs[visible + skip] - 1, xs[-1] + 1
    r0 = min(r for v in targets for r, c in v['cells'])
    r1 = max(r for v in targets for r, c in v['cells'])
    if mode == 'clip':
        # Include the excluded trail's center but cut its lower boundary.
        last_center = tracks[-1]['y'] + (total - 1) * tracks[-1]['drift']
        r1 = last_center
    elif not teaching:
        c0 += int(_pick(rng, (0, 0, 0, 1, 2)))
        c1 -= int(_pick(rng, (0, 0, 0, 1, 2)))
        r0 += int(_pick(rng, (0, 0, 0, 1)))
        r1 -= int(_pick(rng, (0, 0, 0, 1)))
    bounds = (r0, r1, c0, c1)
    if r1 <= r0 or c1 <= c0 or r1 - r0 + c1 - c0 < 4:
        return None
    ballot = defaultdict(Counter)
    for voter in targets:
        if all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in voter['cells']):
            ballot[voter['key']][voter['color']] += 1
    winners = {}
    for key in classes:
        if key not in ballot:
            return None
        highest = max(ballot[key].values())
        best = [color for color, count in ballot[key].items() if count == highest]
        if len(best) != 1:
            return None
        winners[key] = best[0]
    box = np.full((sh, sw), carrier, dtype=int)
    output = box.copy()
    for key, cells in cavities:
        for r, c in cells:
            box[r, c] = bg
            output[r, c] = winners[key]
    if len(_components(box == carrier)) != 1:
        return None
    top = int(rng.integers(0, min(2, 30 - stencil_y - sh) + 1))
    left = int(rng.integers(0, min(2, 30 - xs[-1] - 2) + 1))
    if fallback_variant is not None:
        divisor = 4 if mode == 'pool' else 8
        top, left = fallback_variant // divisor, fallback_variant % divisor
    base_height = stencil_y + sh
    base_width = max(xs[-1] + 2, sw)
    if base_height + top > 30 or base_width + left > 30:
        return None
    bottom = 0 if fallback_variant is not None else int(rng.integers(0, min(2, 30 - base_height - top) + 1))
    right = 0 if fallback_variant is not None else int(rng.integers(0, min(2, 30 - base_width - left) + 1))
    grid = np.full((base_height + top + bottom, base_width + left + right), bg, dtype=int)
    for voter in hypothetical:
        if voter['index'] < visible:
            for r, c in voter['cells']:
                grid[r + top, c + left] = voter['color']
    stencil_x = int(rng.integers(0, grid.shape[1] - sw + 1))
    grid[stencil_y + top:stencil_y + top + sh, stencil_x:stencil_x + sw] = box
    bounds = (r0 + top, r1 + top, c0 + left, c1 + left)
    # Transform the scene and answer; redraw the Ls in their fixed convention.
    if config['flip_y']:
        grid, output = grid[::-1].copy(), output[::-1].copy()
        bounds = (len(grid) - 1 - bounds[1], len(grid) - 1 - bounds[0], bounds[2], bounds[3])
    if config['flip_x']:
        grid, output = grid[:, ::-1].copy(), output[:, ::-1].copy()
        bounds = (bounds[0], bounds[1], grid.shape[1] - 1 - bounds[3], grid.shape[1] - 1 - bounds[2])
    if config['axis'] == 0:
        grid, output = grid.T.copy(), output.T.copy()
        bounds = (bounds[2], bounds[3], bounds[0], bounds[1])
    for r, c in _marker_cells(bounds):
        grid[r, c] = marker
    pair = {'input': grid, 'output': output}
    try:
        predicted, data = _analyze(grid)
    except AssertionError:
        return None
    if not np.array_equal(predicted, output):
        return None
    if mode == 'clip':
        visible_votes = defaultdict(Counter)
        for color, _, raw in data['objects']:
            visible_votes[_canonical(raw)][color] += 1
        # The teaching flip is semantic, not a sampled flag with no effect.
        counts = visible_votes[classes[0]]
        high = max(counts.values())
        old = [c for c, n in counts.items() if n == high]
        if len(old) != 1 or old[0] == winners[classes[0]]:
            return None
        if not any(v['center_inside'] and not v['inside'] for v in data['voters']):
            return None
    return pair


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + ' must be an integer')
        if not lo <= value <= hi:
            raise ValueError(name + ' is out of range')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    pool = _shape_pool()
    chiral = [s for s in pool if _rot_key(s) != _rot_key(_transform(s, (0, True)))
              and any(t != s and len(t) == len(s) and
                      max(r for r, c in t) == max(r for r, c in s) and
                      max(c for r, c in t) == max(c for r, c in s) for t in pool)]
    anchor = _pick(rng, chiral)
    companions = [s for s in pool if s != anchor and len(s) == len(anchor)
                  and max(r for r, c in s) == max(r for r, c in anchor)
                  and max(c for r, c in s) == max(c for r, c in anchor)]
    config = dict(anchors=(anchor, _pick(rng, companions)),
                  flip_x=bool(rng.integers(2)), flip_y=bool(rng.integers(2)),
                  axis=int(rng.integers(2)))
    regime = int(rng.integers(3))  # same axis, switched axis, mixed axis
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            local = dict(config)
            if split == 'test' and regime == 1:
                local['axis'] = 1 - config['axis']
            elif regime == 2 and index >= 2:
                local['axis'] = int(rng.integers(2))
            if index >= 2 or split == 'test':
                local['flip_x'], local['flip_y'] = bool(rng.integers(2)), bool(rng.integers(2))
            mode = ('pool' if index == 0 else 'clip') if split == 'train' and index < 2 else None
            pair = None
            for _ in range(120):
                candidate = _build_pair(rng, local, mode)
                if candidate is not None and _key(candidate['input']) not in seen:
                    _, d = _analyze(candidate['input'])
                    distances = _request_distances(d)
                    # Every demonstration reaches at most the second
                    # unobserved rank; every test reaches at least the third.
                    if not distances or (split == 'test' and max(distances) < 3):
                        continue
                    if split == 'train' and max(distances) > 2:
                        continue
                    pair = candidate
                    break
            if pair is None:
                # Guaranteed teaching ballot, no packing search, 32 potential
                # margin variants. This fallback retains both parent concepts.
                fallback_mode = mode or ('far' if split == 'test' else 'pool')
                for variant in range(32):
                    candidate = _build_pair(rng, local, fallback_mode, variant)
                    if candidate is not None and _key(candidate['input']) not in seen:
                        pair = candidate
                        break
            assert pair is not None, 'bounded construction could not find a valid unique pair'
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train' and index == 0:
                _, d = _analyze(pair['input'])
                config['first_winner'] = d['winners'][anchor]
    return puzzle


def _alternative(data, membership='inside', equivalence='shape', visible=False):
    if visible:
        votes = defaultdict(Counter)
        for color, _, raw in data['objects']:
            key = len(raw) if equivalence == 'area' else _canonical(raw)
            votes[key][color] += 1
    else:
        votes = _ballots(data['voters'], membership, equivalence)
    return _fill(data, votes, equivalence)[0]


def _fails(data, expected, **kwargs):
    try:
        return not np.array_equal(_alternative(data, **kwargs), expected)
    except AssertionError:
        return True


def _request_distances(data):
    n = data['model']['visible']
    return [v['index'] - n + 1 if v['index'] >= n else -v['index']
            for v in data['voters'] if v['inside']]


def _repeat_nearest_gap(data):
    """Counterfactual: extrapolate the last local gap without its change."""
    model = dict(data['model'])
    n = model['visible']
    if model['bounds'][2] > model['x']:
        last_x = (model['x'] + (n - 1) * model['step'] +
                  (n - 1) * (n - 2) // 2 * model['acceleration'])
        gap = model['step'] + (n - 2) * model['acceleration']
        model['x'] = last_x - (n - 1) * gap
        model['step'] = gap
    model['acceleration'] = 0
    return _fill(data, _ballots(_extrapolated(model)))[0]


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            predicted, d = _analyze(inp)
            assert np.array_equal(predicted, out), 'wrong election result or stencil crop'
            assert d['marker'] not in out
            assert np.all(out[d['box'] == d['carrier']] == d['carrier'])
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            assert set(d['votes']) == {_canonical(_normal(c)) for c in d['cavities']}
            assert all(v['color'] not in (d['background'], d['marker'], d['carrier']) for v in d['voters'])
            assert any(v['inside'] for v in d['voters'])
            reach = max(_request_distances(d))
            assert reach <= 2 if split == 'train' else reach >= 3
            analyses[split].append((out, d))
    train = analyses['train']
    # Actual answer differences, not latent generator flags.
    assert any(_fails(d, out, equivalence='area') for out, d in train)
    assert any(_fails(d, out, visible=True) for out, d in train)
    assert any(_fails(d, out, membership='center_inside') for out, d in train)
    assert any(len(counts) >= 3 and max(counts.values()) * 2 <= sum(counts.values())
               for _, d in train for counts in d['votes'].values())
    assert any(any(v['intersects'] and not v['inside'] for v in d['voters']) for _, d in train)
    history = defaultdict(set)
    for _, d in train:
        for key, color in d['winners'].items():
            history[key].add(color)
    assert any(len(colors) > 1 for colors in history.values())
    # In the first example, a same-area merger actually predicts a different
    # array and reflected observed shapes do not rotate into their cavity.
    out, d = train[0]
    assert abs(d['model']['acceleration']) == 1
    assert not np.array_equal(_repeat_nearest_gap(d), out)
    assert not np.array_equal(_alternative(d, equivalence='area'), out)
    assert any(all(_rot_key(cells) != _rot_key(raw) for _, _, raw in d['objects']
                   if _canonical(raw) == _canonical(_normal(cells)))
               for cells in d['cavities'])
    out, d = train[1]
    assert any(abs(t['drift']) == 1 for t in d['model']['tracks'])
    assert not np.array_equal(_alternative(d, visible=True), out)


def self_test():
    """Bounded 42-seed semantic, diversity, determinism and corruption checks."""
    # A smallest compatible carrier is a 3x3 border with a singleton cavity.
    # This input is a rule boundary fixture, not a sampled teaching puzzle.
    small = np.zeros((9, 18), dtype=int)
    small[1, 1], small[1, 5], small[1, 9] = 2, 3, 2
    small[5:8, 1:4] = 8
    small[6, 2] = 0
    for r, c in _marker_cells((0, 2, 12, 15)):
        small[r, c] = 7
    assert np.array_equal(solve(small), np.array([[8, 8, 8], [8, 3, 8], [8, 8, 8]]))
    keys, shapes, directions, classes, roles, ballots = set(), set(), set(), set(), set(), set()
    deltas, drifts, tracks, counts, cavity_counts, geometries = set(), set(), set(), set(), set(), set()
    partial = reflected = unseen = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])
        keys.add(signature)
        known = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                output, d = _analyze(pair['input'])
                assert np.array_equal(solve(pair['input'].tolist()), output)
                m = d['model']
                shapes.add((pair['input'].shape, output.shape))
                directions.add(m['direction'])
                classes.update(d['votes'])
                roles.add((d['background'], d['marker'], d['carrier']))
                ballots.update(tuple(sorted(v.values(), reverse=True)) for v in d['votes'].values())
                deltas.add(m['acceleration'])
                drifts.update(t['drift'] for t in m['tracks'])
                tracks.add(len(m['tracks']))
                counts.add(m['visible'])
                cavity_counts.add(len(d['cavities']))
                geometries.add(tuple(t['shape'] for t in m['tracks']))
                partial += any(v['intersects'] and not v['inside'] for v in d['voters'])
                reflected += any(all(_rot_key(c) != _rot_key(raw) for _, _, raw in d['objects']
                                     if _canonical(raw) == _canonical(_normal(c))) for c in d['cavities'])
                if split == 'train':
                    known.update(d['votes'])
                else:
                    unseen += bool(set(d['votes']) - known)
    assert len(keys) == 42 and len(shapes) >= 35
    assert directions == {'left', 'right', 'up', 'down'}
    assert len(classes) >= 10 and len(roles) >= 80 and len(ballots) >= 8
    assert deltas == {-1, 0, 1} and drifts == {-1, 0, 1}
    assert tracks == {1, 2, 3, 4} and counts == {3, 4, 5}
    assert len(cavity_counts) >= 4 and len(geometries) >= 80
    assert partial >= 42 and reflected >= 42 and unseen >= 10
    # Every fallback variant is constructible, not merely another random
    # attempt. Its dimensions distinguish 32 inputs, exceeding the maximum
    # 13 preceding examples in any legal request.
    anchor = _canonical(((0, 1), (0, 2), (1, 0), (1, 1), (2, 1)))
    companion = _canonical(((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)))
    fallback_config = dict(anchors=(anchor, companion), axis=1,
                           flip_x=False, flip_y=False)
    for mode in ('pool', 'clip', 'far'):
        fallback_keys = set()
        for variant in range(32):
            pair = _build_pair(np.random.default_rng(44), fallback_config, mode, variant)
            assert pair is not None
            assert np.array_equal(solve(pair['input']), pair['output'])
            fallback_keys.add(_key(pair['input']))
        assert len(fallback_keys) == 32
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(130 + a * 7 + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    first, second = generate_puzzle(17, 2, 1), generate_puzzle(17, 2, 1)
    for split in first:
        for a, b in zip(first[split], second[split]):
            assert all(np.array_equal(a[k], b[k]) for k in a)
    corrupt = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]] for s in first}
    corrupt['test'][0]['output'][0, 0] = (int(corrupt['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
