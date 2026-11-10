"""Thread a hub's cyclic ribbon around its compartment's exterior shore.

Only standard-library and NumPy dependencies. All decisions in solve() come
from the query's pixels. Generation never supplies metadata to the solver.
No files, examples, random state, or palette dictionary are read by solve().
"""
from collections import deque
import hashlib
import itertools
import numpy as np


N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 513407886, 243570336, 54906322, 235851876, 445807722,
              664160971, 279714071, 334078803, 295835898, 314980916,
              127486046, 900317527, 388972661, 478573911, 894740442,
              926749717, 647879223, 326712650, 58726327, 838806669,
              945137690, 103284065, 164761366, 764351547, 451257216,
              254985731, 242921145, 497416957, 382955112, 869774445,
              960796560, 955234244, 570239888, 125105275, 704140973,
              80057914, 503249789, 267353831, 884186858, 735704244, 93890243)


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= d <= 30 for d in g.shape)
    assert np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9))
    return g


def _flood(allowed, start):
    h, w = allowed.shape
    seen = np.zeros((h, w), bool)
    if not allowed[start]:
        return seen
    seen[start] = True
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in N4:
            x, y = r + dr, c + dc
            if 0 <= x < h and 0 <= y < w and allowed[x, y] and not seen[x, y]:
                seen[x, y] = True
                queue.append((x, y))
    return seen


def _components(mask):
    remaining = mask.copy()
    groups = []
    while remaining.any():
        p = tuple(int(x) for x in np.argwhere(remaining)[0])
        component = _flood(remaining, p)
        remaining[component] = False
        groups.append(component)
    return groups


def _dilate(mask, diagonal=True):
    h, w = mask.shape
    p = np.pad(mask, 1)
    result = np.zeros_like(mask)
    shifts = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)] if diagonal else N4
    for dr, dc in shifts:
        result |= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


def _shore(region):
    # Padding makes the outside explicit even for fully enclosed rooms.
    outside = _flood(np.pad(~region, 1, constant_values=True), (0, 0))
    rim = _dilate(outside)[1:-1, 1:-1] & region
    orth = _dilate(outside, False)[1:-1, 1:-1] & region
    return rim, outside[1:-1, 1:-1], rim & ~orth


def _cycle(rim):
    """A valid shore is exactly one simple four-neighbor cycle."""
    cells = {tuple(int(x) for x in p) for p in np.argwhere(rim)}
    assert len(cells) >= 12
    neighbors = {p: sorted((p[0] + dr, p[1] + dc) for dr, dc in N4
                           if (p[0] + dr, p[1] + dc) in cells) for p in cells}
    assert all(len(ns) == 2 for ns in neighbors.values()), "shore branches or pinches"
    start = min(cells)
    previous, current = start, neighbors[start][0]
    order = [start]
    while current != start:
        assert current not in order, "shore is not simple"
        order.append(current)
        following = next(p for p in neighbors[current] if p != previous)
        previous, current = current, following
    assert len(order) == len(cells), "multiple exterior cycles"
    return order


def _winding(order):
    # Signed area in screen coordinates. Only its sign is used for sampling.
    area = sum(c * y - d * r for (r, c), (y, d) in zip(order, order[1:] + order[:1]))
    return 1 if area > 0 else -1


def _period(word):
    return next(p for p in range(1, len(word) + 1)
                if len(word) % p == 0 and all(v == word[i % p] for i, v in enumerate(word)))


def _straight(path):
    return len(path) < 3 or len({(b[0] - a[0], b[1] - a[1])
                               for a, b in zip(path, path[1:])}) == 1


def _hub_paths(region, rim, length, desired_winding=None):
    """Return compatible 2x2 corner hubs and both possible directed shores."""
    cycle = _cycle(rim)
    index = {p: i for i, p in enumerate(cycle)}
    h, w = region.shape
    options = []
    for r in range(h - 1):
        for c in range(w - 1):
            if not region[r:r + 2, c:c + 2].all() or rim[r:r + 2, c:c + 2].sum() != 3:
                continue
            hub = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
            shore_hub = set(index).intersection(hub)
            for p in sorted(shore_hub):
                for step in (-1, 1):
                    i = index[p]
                    if cycle[(i + step) % len(cycle)] in hub:
                        continue
                    path = []
                    j = (i + step) % len(cycle)
                    while cycle[j] not in hub:
                        path.append(cycle[j])
                        j = (j + step) % len(cycle)
                    if len(path) != len(cycle) - 3 or len(path) < length + 8:
                        continue
                    if not _straight(path[:length]):
                        continue
                    winding = _winding(cycle) * step
                    if desired_winding is None or winding == desired_winding:
                        options.append((hub, path, winding))
    return options


def _retained_walls(walls, region):
    kept = np.zeros_like(walls)
    touching = _dilate(region, False)
    for component in _components(walls):
        if np.any(component & touching):
            kept |= component
    return kept


def _analyze(input_grid):
    g = _grid(input_grid)
    h, w = g.shape
    wall = int(g[0, 0])
    assert np.all(g[0] == wall) and np.all(g[-1] == wall)
    assert np.all(g[:, 0] == wall) and np.all(g[:, -1] == wall), "solid outer frame required"
    values, counts = np.unique(g, return_counts=True)
    bg = int(values[np.argmax(counts)])
    assert bg != wall and np.count_nonzero(counts == counts.max()) == 1
    hub_colors = []
    for color in values:
        if int(color) in (bg, wall):
            continue
        cells = np.argwhere(g == color)
        if len(cells) == 4 and np.ptp(cells[:, 0]) == np.ptp(cells[:, 1]) == 1:
            hub_colors.append(int(color))
    assert len(hub_colors) == 1, "exactly one solid 2x2 colored hub required"
    hub_color = hub_colors[0]
    hub = {tuple(int(x) for x in p) for p in np.argwhere(g == hub_color)}
    region = _flood(g != wall, min(hub))
    assert all(region[p] for p in hub)
    rim, outside, diagonal = _shore(region)
    cycle = _cycle(rim)
    assert sum(rim[p] for p in hub) == 3
    chain = [p for p in cycle if p not in hub]
    # Removing three consecutive hub cells must leave one path, not two arcs.
    path_cells = set(chain)
    endpoints = sorted(p for p in chain if sum((p[0] + dr, p[1] + dc) in path_cells
                                              for dr, dc in N4) == 1)
    assert len(endpoints) == 2
    pigment = set(int(c) for c in values) - {bg, wall, hub_color}
    starts = [p for p in endpoints if int(g[p]) in pigment]
    assert len(starts) == 1, "one painted departure and one blank return port required"
    path, previous, current = [], None, starts[0]
    while True:
        path.append(current)
        following = [(current[0] + dr, current[1] + dc) for dr, dc in N4
                     if (current[0] + dr, current[1] + dc) in path_cells
                     and (current[0] + dr, current[1] + dc) != previous]
        if not following:
            break
        assert len(following) == 1 and following[0] not in path
        previous, current = current, following[0]
    assert len(path) == len(cycle) - 3
    length = next((i for i, p in enumerate(path) if int(g[p]) == bg), len(path))
    assert 2 <= length <= len(path) - 1 and _straight(path[:length])
    word = [int(g[p]) for p in path[:length]]
    assert all(v in pigment for v in word)
    period = _period(word)
    assert length >= 2 * period, "at least two whole copies of the motif are visible"
    assert all(int(g[p]) == bg for p in path[length:])
    claimed = np.zeros(g.shape, bool)
    for p in hub | set(path[:length]):
        claimed[p] = True
    assert np.all(claimed[(g != wall) & (g != bg)]), "stray colored material"
    kept = _retained_walls(g == wall, region)
    output = g.copy()
    output[(g == wall) & ~kept] = bg
    for i, p in enumerate(path[length:], length):
        output[p] = word[i % period]
    return dict(grid=g, background=bg, wall=wall, hub_color=hub_color, hub=hub,
                region=region, rim=rim, outside=outside, diagonal=diagonal,
                path=path, word=word, period=period, length=length,
                kept=kept, output=output)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover roles, region, direction, motif, and phase entirely from input."""
    return _analyze(input_grid)["output"].copy()


def _frame(h, w):
    walls = np.zeros((h, w), bool)
    walls[[0, -1], :] = True
    walls[:, [0, -1]] = True
    return walls


def _partition(rng, h, w, count):
    walls = _frame(h, w)
    rooms = [(1, h - 1, 1, w - 1)]
    while len(rooms) < count:
        choices = [(i, axis) for i, (r0, r1, c0, c1) in enumerate(rooms)
                   for axis, extent in ((0, r1 - r0), (1, c1 - c0)) if extent >= 17]
        if not choices:
            break
        i, axis = choices[int(rng.integers(len(choices)))]
        r0, r1, c0, c1 = rooms.pop(i)
        if axis == 0:
            cut = int(rng.integers(r0 + 7, r1 - 7))
            walls[cut, c0:c1] = True
            rooms.extend([(r0, cut, c0, c1), (cut + 1, r1, c0, c1)])
        else:
            cut = int(rng.integers(c0 + 7, c1 - 7))
            walls[r0:r1, cut] = True
            rooms.extend([(r0, r1, c0, cut), (r0, r1, cut + 1, c1)])
    return walls, rooms


def _notch(rng, walls, rect, region, number):
    r0, r1, c0, c1 = rect
    accepted = 0
    for _ in range(24):
        if accepted >= number:
            break
        side = int(rng.integers(4))
        longitudinal = c1 - c0 if side < 2 else r1 - r0
        transverse = r1 - r0 if side < 2 else c1 - c0
        if longitudinal < 6 or transverse < 7:
            continue
        width = int(rng.integers(1, min(6, longitudinal - 4) + 1))
        offset = int(rng.integers(2, longitudinal - width - 1))
        depth = int(rng.integers(1, min(4, transverse - 5) + 1))
        trial = walls.copy()
        if side == 0:
            trial[r0:r0 + depth, c0 + offset:c0 + offset + width] = True
        elif side == 1:
            trial[r1 - depth:r1, c0 + offset:c0 + offset + width] = True
        elif side == 2:
            trial[r0 + offset:r0 + offset + width, c0:c0 + depth] = True
        else:
            trial[r0 + offset:r0 + offset + width, c1 - depth:c1] = True
        left = region & ~trial
        if not left.any():
            continue
        start = tuple(int(v) for v in np.argwhere(left)[0])
        if not np.array_equal(_flood(~trial, start), left):
            continue
        try:
            _cycle(_shore(left)[0])
        except AssertionError:
            continue
        if np.array_equal(trial, walls):
            continue
        walls, region = trial, left
        accepted += 1
    return walls, region, accepted


def _island(rng, walls, region):
    # Two blank cells surround each island, protecting the outside shore.
    candidates = region & ~_dilate(_dilate(walls))
    points = np.argwhere(candidates)
    rng.shuffle(points)
    for r, c in points:
        sh, sw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        mask = np.ones((sh, sw), bool)
        kind = int(rng.integers(4))
        if kind == 1:
            mask[1:, 1:] = False
        elif kind == 2:
            mask = mask[:1]
        elif kind == 3:
            for row in range(sh):
                mask[row, int(rng.integers(1, sw + 1)):] = False
        sh, sw = mask.shape
        r, c = int(r), int(c)
        if r + sh > walls.shape[0] or c + sw > walls.shape[1]:
            continue
        if not np.all(candidates[r:r + sh, c:c + sw][mask]):
            continue
        trial = walls.copy()
        trial[r:r + sh, c:c + sw] |= mask
        return trial, True
    return walls, False


def _tile(rng, period, forced=None):
    if forced is not None:
        return list(forced)
    size = int(rng.integers(1 if period == 1 else 2, min(period, 5) + 1))
    word = [int(v) for v in rng.integers(size, size=period)]
    if _period(word) != period:
        word = [0] * (period - 1) + [1]
    return word


def _make_scene(rng, witness, period):
    if witness:
        h, w = int(rng.integers(23, 31)), int(rng.integers(25, 31))
        walls = _frame(h, w)
        cut = int(rng.integers(9, 12))
        walls[:, cut] = True
        rooms = [(1, h - 1, 1, cut), (1, h - 1, cut + 1, w - 1)]
    else:
        h, w = int(rng.integers(16, 31)), int(rng.integers(16, 31))
        if rng.random() < .2:
            h = int(rng.integers(12, 17))
        walls, rooms = _partition(rng, h, w, int(rng.integers(1, 5)))
    masks = []
    for r0, r1, c0, c1 in rooms:
        mask = np.zeros_like(walls)
        mask[r0:r1, c0:c1] = True
        masks.append(mask)
    sizes = [int(m.sum()) for m in masks]
    selected = int(np.argmin(sizes)) if witness == 1 else int(np.argmax(sizes)) if witness == 2 else int(rng.integers(len(rooms)))
    target = masks[selected]
    walls, target, notches = _notch(rng, walls, rooms[selected], target, int(rng.integers(1 if witness else 0, 5)))
    if witness and not notches:
        return None
    for i, rect in enumerate(rooms):
        if i != selected and rng.random() < .5:
            walls, masks[i], _ = _notch(rng, walls, rect, masks[i], int(rng.integers(1, 3)))
    masks[selected] = target
    # Recompute room masks after intrusions, before adding detached objects.
    for i, m in enumerate(masks):
        masks[i] = m & ~walls
    island_count = int(rng.integers(1 if witness else 0, 4))
    placed = 0
    for _ in range(island_count):
        walls, success = _island(rng, walls, masks[selected] & ~walls)
        placed += success
    if witness and not placed:
        return None
    others = [m for i, m in enumerate(masks) if i != selected]
    erased = 0
    for _ in range(int(rng.integers(1 if witness else 0, 5))):
        if others:
            walls, success = _island(rng, walls, others[int(rng.integers(len(others)))] & ~walls)
            erased += success
    if witness and not erased:
        return None
    region = masks[selected] & ~walls
    if int((~walls).sum()) <= int(walls.sum()) + 20:
        return None
    return walls, region


def _fallback_scene(witness):
    walls = _frame(28, 30)
    walls[:, 11] = True
    walls[10:14, 1:3] = True
    walls[1:4, 19:22] = True
    walls[18:20, 6:8] = True
    walls[16:18, 21:23] = True
    start = (5, 5) if witness != 2 else (5, 16)
    return walls, _flood(~walls, start)


def _key(g):
    return g.shape, np.ascontiguousarray(g, dtype=np.int8).tobytes()


def _fresh_input(inp, out, used):
    if _key(inp) not in used:
        return inp, out
    # At most thirteen previous inputs exist. Injective recolorings of the
    # actually used colors give distinct inputs, so len(used)+1 is sufficient.
    # Color roles are geometric, and output introduces no new color.
    colors = [int(v) for v in np.unique(inp)]
    for values in itertools.islice(itertools.permutations(range(10), len(colors)), len(used) + 1):
        mapping = np.arange(10, dtype=np.int8)
        mapping[colors] = values
        candidate = mapping[inp]
        if _key(candidate) not in used:
            return candidate, mapping[out]
    raise AssertionError('finite injective recoloring space exhausted')


def _materialize(rng, walls, region, word, copies, winding, used, forbidden_side=None):
    length = len(word) * copies
    try:
        options = _hub_paths(region, _shore(region)[0], length, winding)
    except AssertionError:
        return None
    rng.shuffle(options)
    for hub, path, _ in options:
        for _ in range(2):
            palette = [int(v) for v in rng.permutation(10)]
            bg, wall, marker = palette[:3]
            colors = [palette[3 + v] for v in word]
            g = np.full(walls.shape, bg, np.int8)
            g[walls] = wall
            for p in hub:
                g[p] = marker
            for i, p in enumerate(path[:length]):
                g[p] = colors[i % len(colors)]
            # Labels use the constructed path and sampled word. The input-only
            # checker separately rediscovers both after rotation/reflection.
            out = g.copy()
            out[walls & ~_retained_walls(walls, region)] = bg
            for i, p in enumerate(path[length:], length):
                out[p] = colors[i % len(colors)]
            rotation = int(rng.integers(4))
            g, out = np.rot90(g, rotation).copy(), np.rot90(out, rotation).copy()
            # Reflection would reverse the requested winding, so orientation
            # diversity is obtained with independently sampled signed winding.
            g, out = _fresh_input(g, out, used)
            try:
                info = _analyze(g)
                assert np.array_equal(info['output'], out)
            except AssertionError:
                continue
            f = _features(info)
            if forbidden_side is not None and f['side'] == forbidden_side:
                continue
            return {'input': g, 'output': out}, f
    return None


def _make_pair(rng, witness, period, winding, used, forbidden_side=None):
    word = _tile(rng, period, [0, 1, 0] if witness == 1 else [0, 1] if witness == 2 else None)
    copies = 2 if witness or period >= 4 else int(rng.integers(2, 4))
    for _ in range(24):
        scene = _make_scene(rng, witness, period)
        if scene is None:
            continue
        result = _materialize(rng, *scene, word, copies, winding, used, forbidden_side)
        if result is None:
            continue
        pair, features = result
        if witness and not (features['islands'] and features['erased'] and features['diagonal'] and features['reset_errors']):
            continue
        if witness == 1 and features['selection'] != 'small':
            continue
        if witness == 2 and features['selection'] != 'large':
            continue
        return pair, features
    result = _materialize(rng, *_fallback_scene(witness), word, copies, winding, used, forbidden_side)
    assert result is not None, 'constructive fallback exhausted'
    return result


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count('num_train', num_train, 2, 10)
    num_test = _count('num_test', num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    puzzle, used = {'train': [], 'test': []}, set()
    first_winding = 1 if rng.integers(2) else -1
    first_side = None
    for i in range(num_train):
        witness = i + 1 if i < 2 else 0
        period = 3 if witness == 1 else 2 if witness == 2 else int(rng.integers(1, 4))
        winding = first_winding if i == 0 else -first_winding if i == 1 else None
        pair, f = _make_pair(rng, witness, period, winding, used, first_side if i == 1 else None)
        if i == 0:
            first_side = f['side']
        puzzle['train'].append(pair)
        used.add(_key(pair['input']))
    rng.shuffle(puzzle['train'])
    # Longer visible complete motifs, with familiar geometry re-combined.
    for _ in range(num_test):
        pair, _ = _make_pair(rng, 0, int(rng.integers(4, 6)), None, used)
        puzzle['test'].append(pair)
        used.add(_key(pair['input']))
    return puzzle


def _features(a):
    g, bg, wall = a['grid'], a['background'], a['wall']
    regions = _components(g != wall)
    sizes = [int(m.sum()) for m in regions]
    area = int(a['region'].sum())
    selection = 'only' if len(sizes) == 1 else 'small' if area < max(sizes) else 'large' if area > min(sizes) else 'equal'
    center = np.argwhere(a['region']).mean(axis=0)
    dy, dx = (center + .5) / np.array(g.shape) - .5
    side = ('top' if dy < 0 else 'bottom') if abs(dy) > abs(dx) else ('left' if dx < 0 else 'right')
    islands = sum(not np.any(m & a['outside']) for m in _components(g == wall))
    erased = sum(not np.any(m & a['kept']) for m in _components(g == wall))
    dirs = [(y - r, d - c) for (r, c), (y, d) in zip(a['path'], a['path'][1:])]
    turns = sum(x != y for x, y in zip(dirs, dirs[1:]))
    phase, wrong = 0, 0
    for i, p in enumerate(a['path']):
        if i >= 2 and dirs[i - 1] != dirs[i - 2]:
            phase = 0
        if i >= a['length'] and a['word'][phase % a['period']] != int(a['output'][p]):
            wrong += 1
        phase += 1
    # Complete the directed cycle through the three hub cells to measure sign.
    path = a['path']
    closure = [p for p in a['hub'] if a['rim'][p]]
    current, end = path[-1], path[0]
    closed = list(path)
    while closure:
        following = next(p for p in closure if abs(p[0] - current[0]) + abs(p[1] - current[1]) == 1)
        closed.append(following)
        closure.remove(following)
        current = following
    assert abs(current[0] - end[0]) + abs(current[1] - end[1]) == 1
    return dict(shape=g.shape, rooms=len(regions), selection=selection, side=side,
                islands=islands, erased=erased, diagonal=int(a['diagonal'].sum()),
                period=a['period'], copies=a['length'] // a['period'], turns=turns,
                winding=_winding(closed), path_length=len(path),
                clipped=len(path) % a['period'] != 0, reset_errors=wrong,
                initial_direction=dirs[0], hub_position=min(a['hub']),
                background=bg, wall=wall, hub_color=a['hub_color'],
                palette_size=len(np.unique(g)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    used, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            g, out = _grid(pair['input']), _grid(pair['output'])
            assert g.shape == out.shape and _key(g) not in used
            used.add(_key(g))
            a = _analyze(g)
            assert np.array_equal(out, a['output']), 'wrong threading, phase, or component erasure'
            assert not np.array_equal(g, out)
            protected = (g != a['background']) & (g != a['wall'])
            assert np.array_equal(g[protected], out[protected]), 'hub or original ribbon changed'
            assert np.array_equal(g[a['region'] & ~a['rim']], out[a['region'] & ~a['rim']])
            assert np.all(out[(g == a['wall']) & a['kept']] == a['wall'])
            assert set(np.unique(out)) <= set(np.unique(g))
            features[split].append(_features(a))
    train = features['train']
    assert {2, 3} <= {f['period'] for f in train}, 'teach a cyclic word, including repeated colors'
    assert {f['winding'] for f in train} == {-1, 1}, 'departure side must determine direction'
    assert {'small', 'large'} <= {f['selection'] for f in train}
    assert len({f['side'] for f in train}) >= 2
    for name in ('islands', 'erased', 'diagonal', 'reset_errors', 'clipped'):
        assert any(f[name] for f in train), f'missing {name} witness'
    assert max(f['period'] for f in train) < min(f['period'] for f in features['test']), 'missing longer-word test shift'


def self_test():
    """42 seeds, semantic mutations, equivariance, bounds, and finite fallbacks.

    Streams one puzzle at a time; stores only tiny scalar diversity sets and
    32-byte digests. Does not write files or save traces.
    """
    diversity = {k: set() for k in ('shape', 'rooms', 'islands', 'erased', 'period',
                                    'copies', 'turns', 'winding', 'initial_direction',
                                    'background', 'wall', 'hub_color', 'path_length')}
    signatures = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                g = pair['input']
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
                a = _analyze(g)
                f = _features(a)
                for name in diversity:
                    diversity[name].add(f[name])
                # Omitting A: a flat rim disagrees with a multicolor cycle.
                if f['period'] > 1:
                    assert any(int(a['output'][p]) != a['word'][0] for p in a['path'][a['length']:])
                assert np.array_equal(solve(g), pair['output'])
        signatures.add(digest.digest())
    assert len(signatures) == 42
    for name, lower in (('shape', 50), ('rooms', 3), ('islands', 3), ('erased', 3),
                        ('period', 5), ('copies', 2), ('turns', 5), ('winding', 2),
                        ('initial_direction', 4), ('background', 8), ('wall', 8),
                        ('hub_color', 8), ('path_length', 30)):
        assert len(diversity[name]) >= lower, (name, len(diversity[name]))
    base = generate_puzzle(0, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in base for a, b in
               zip(base[s], generate_puzzle(0, 2, 1)[s]) for k in ('input', 'output'))
    seen = set()
    for _ in range(14):
        pair = base['train'][0]
        inp, out = _fresh_input(pair['input'], pair['output'], seen)
        assert _key(inp) not in seen and np.array_equal(solve(inp), out)
        seen.add(_key(inp))
    for pair in base['train']:
        for rotation in range(4):
            for reflect in (False, True):
                g, out = np.rot90(pair['input'], rotation), np.rot90(pair['output'], rotation)
                if reflect:
                    g, out = np.fliplr(g), np.fliplr(out)
                before = g.copy()
                assert np.array_equal(solve(g), out) and np.array_equal(g, before)
        mapping = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], np.int8)
        assert np.array_equal(solve(mapping[pair['input']]), mapping[pair['output']])
        a = _analyze(pair['input'])
        # Omitting B's hole exclusion would color neighbors of this island.
        island = (pair['input'] == a['wall']) & ~a['outside']
        neighbors = _dilate(island) & a['region'] & ~a['rim'] & (pair['input'] == a['background'])
        assert neighbors.any() and np.all(pair['output'][neighbors] == a['background'])
        # Remove an enclosed island: the route and every ribbon phase stay fixed.
        cleared = pair['input'].copy()
        cleared[island] = a['background']
        expected = pair['output'].copy()
        expected[island] = a['background']
        assert np.array_equal(solve(cleared), expected)
    # Hand-specified routes: attaching an island to the wall creates a detour.
    # It adds fourteen cells to an ABA thread, changing the downstream phase
    # by two. These expected coordinates are not obtained from the rim parser.
    grid = np.zeros((12, 16), np.int8)
    grid[_frame(12, 16)] = 5
    grid[1:3, 1:3] = 9
    grid[5, 7] = 5
    shared = ([(1, c) for c in range(3, 15)]
              + [(r, 14) for r in range(2, 11)]
              + [(10, c) for c in range(13, 0, -1)])
    old_path = shared + [(r, 1) for r in range(9, 2, -1)]
    word = [2, 3, 2]
    for i, p in enumerate(old_path[:6]):
        grid[p] = word[i % 3]
    expected = grid.copy()
    for i, p in enumerate(old_path):
        expected[p] = word[i % 3]
    assert np.array_equal(solve(grid), expected)
    attached = grid.copy()
    attached[5, 1:8] = 5
    new_path = (shared + [(r, 1) for r in range(9, 6, -1)]
                + [(6, c) for c in range(1, 9)] + [(5, 8)]
                + [(4, c) for c in range(8, 0, -1)] + [(3, 1)])
    expected_attached = attached.copy()
    for i, p in enumerate(new_path):
        expected_attached[p] = word[i % 3]
    assert len(new_path) - len(old_path) == 14
    assert expected[3, 1] == 3 and expected_attached[3, 1] == 2
    assert np.array_equal(solve(attached), expected_attached)
    # Bounded constructive escape path works for all generated word lengths.
    for witness, period in ((1, 3), (2, 2), (0, 4), (0, 5)):
        word = [0, 1, 0] if witness == 1 else [0] * (period - 1) + [1]
        result = _materialize(np.random.default_rng(period), *_fallback_scene(witness), word, 2, None, set())
        assert result is not None
        pair, f = result
        assert f['islands'] and f['erased'] and f['diagonal'] and f['reset_errors']
        assert np.array_equal(solve(pair['input']), pair['output'])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(419, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in base[s]] for s in base}
    a = _analyze(damaged['test'][0]['input'])
    p = a['path'][a['length']]
    damaged['test'][0]['output'][p] = a['background']
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged output accepted')
