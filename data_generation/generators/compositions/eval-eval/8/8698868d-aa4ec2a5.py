"""Keyed halo circuits: cavity counts address a morphological contact network.

Only standard-library modules and NumPy are used. Generation writes no files.
The public solver uses the visible card, original components and original holes.
"""
from collections import deque
import hashlib
import itertools

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid size")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(((a >= 0) & (a <= 9)).all()), "ARC colors required")
    return a.astype(np.int8, copy=False)


def _regions(mask, neighbors=N4):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    cells.append(q)
                    stack.append(q)
        result.append(cells)
    return result


def _mask(cells, shape):
    result = np.zeros(shape, bool)
    rr, cc = np.array(cells, int).T
    result[rr, cc] = True
    return result


def _box(mask):
    rr, cc = np.nonzero(mask)
    return int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1


def _crop(mask):
    r, c, s, t = _box(mask)
    return mask[r:s, c:t].copy()


def _holes(mask, neighbors=N4):
    mask = np.asarray(mask, bool)
    result = np.zeros_like(mask)
    if not mask.any():
        return result
    # Nothing outside the occupied box can be enclosed. Cropping avoids
    # repeatedly flooding the entire canvas for each small object.
    r, c, s, t = _box(mask)
    padded = np.pad(mask[r:s, c:t], 1)
    exterior = np.zeros_like(padded)
    exterior[0] = exterior[-1] = True
    exterior[:, 0] = exterior[:, -1] = True
    h, w = padded.shape
    shifts = [((slice(max(0, dr), h + min(0, dr)), slice(max(0, dc), w + min(0, dc))),
               (slice(max(0, -dr), h + min(0, -dr)), slice(max(0, -dc), w + min(0, -dc))))
              for dr, dc in neighbors]
    background = ~padded
    # Monotone array propagation is the same outside flood, without a Python
    # visit for every exterior cell. At most padded.size cells can be added.
    while True:
        grown = exterior.copy()
        for destination, source in shifts:
            grown[destination] |= exterior[source]
        grown &= background
        if np.array_equal(grown, exterior):
            break
        exterior = grown
    result[r:s, c:t] = (background & ~exterior)[1:-1, 1:-1]
    return result


def _dilate(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w] for dr, dc in N8])


def _clear_topology(mask):
    a, b, c, d = mask[:-1, :-1], mask[:-1, 1:], mask[1:, :-1], mask[1:, 1:]
    return not bool(np.any((a == d) & (b == c) & (a != b)))


def _analysis(input_grid):
    """Decode all roles and the graph from input pixels, with no generation data."""
    grid = _grid(input_grid)
    colors, sizes = np.unique(grid, return_counts=True)
    _require(len(colors) == 4 and sizes.max() > grid.size // 2, "four colors and majority background")
    bg = int(colors[np.argmax(sizes)])
    components = _regions(grid != bg)
    cards, objects = [], []
    for cells in components:
        body = _mask(cells, grid.shape)
        values = np.unique(grid[body])
        if len(values) > 1:
            cards.append(body)
        else:
            objects.append(body)
    _require(len(cards) == 1 and 4 <= len(objects) <= 8, "one card and 4..8 objects required")
    card = cards[0]
    r, c, s, t = _box(card)
    block = grid[r:s, c:t]
    _require(bool(card[r:s, c:t].all()) and min(block.shape) >= 3, "filled rectangular card")
    rim = np.concatenate((block[0], block[-1], block[:, 0], block[:, -1]))
    ink = int(rim[0])
    _require(bool((rim == ink).all()), "uniform card border")
    pip_colors = [int(v) for v in np.unique(block) if v != ink]
    _require(len(pip_colors) == 1 and pip_colors[0] != bg, "one card pip color")
    fill = pip_colors[0]
    pips = block == fill
    k = int(pips.sum())
    _require(1 <= k <= 3 and all(len(x) == 1 for x in _regions(pips)), "1..3 isolated pips")
    fg_values = {int(grid[m][0]) for m in objects}
    _require(len(fg_values) == 1, "one common object color")
    fg = fg_values.pop()
    _require(len({bg, ink, fill, fg}) == 4, "distinct visible roles")
    body = grid == fg
    _require(len(_regions(body, N8)) == len(objects), "no diagonal body contacts")
    holes, counts, halos = [], [], []
    union_holes = np.zeros_like(body)
    for obj in objects:
        _require(_clear_topology(obj), "diagonal topology ambiguity")
        hole = _holes(obj)
        _require(np.array_equal(hole, _holes(obj, N8)), "diagonal cavity leak")
        regions = _regions(hole)
        _require(len(regions) == len(_regions(hole, N8)) <= 3, "0..3 separate cavities")
        _require(not bool(np.any(hole & (body | card | union_holes))), "nested or shared cavity")
        holes.append(hole)
        counts.append(len(regions))
        union_holes |= hole
    _require(np.array_equal(union_holes, _holes(body)), "collective enclosure")
    exterior = (grid == bg) & ~union_holes
    for obj in objects:
        halos.append(_dilate(obj) & exterior)
    roots = [i for i, n in enumerate(counts) if n == k]
    _require(len(roots) == 1, "card count must address exactly one object")
    graph = [set() for _ in objects]
    for i, j in itertools.combinations(range(len(objects)), 2):
        if np.any(halos[i] & halos[j]):
            graph[i].add(j)
            graph[j].add(i)
    root = roots[0]
    distance = {root: 0}
    queue = deque([root])
    while queue:
        i = queue.popleft()
        for j in sorted(graph[i]):
            if j not in distance:
                distance[j] = distance[i] + 1
                queue.append(j)
    return dict(grid=grid, background=bg, foreground=fg, ink=ink, fill=fill,
                card=card, card_box=(r, c, s, t), key=k, objects=objects, holes=holes,
                counts=counts, halos=halos, graph=graph, root=root, distance=distance)


def _paint(a, active):
    out = a['grid'].copy()
    for i in sorted(active):
        out[a['halos'][i]] = a['ink']
        if a['counts'][i]:
            out[a['objects'][i]] = a['ink']
            out[a['holes'][i]] = a['fill']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Count the card pips, find the source, then follow shared exterior halos."""
    a = _analysis(input_grid)
    return _paint(a, a['distance'])


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _orient(rng, mask):
    m = np.rot90(mask, int(rng.integers(4)))
    return _crop(np.fliplr(m) if rng.integers(2) else m)


def _shape(rng, count, style='ordinary', small=False):
    """Connected sculpted bodies, independently shaped sealed pockets/open bays."""
    if count == 0 and style != 'open' and rng.random() < .35:
        tiny = _choice(rng, [np.ones((1, 1), bool), np.ones((1, int(rng.integers(2, 6))), bool),
                             np.array([[1, 0], [1, 1]], bool),
                             np.array([[0, 1, 0], [1, 1, 1]], bool)])
        return _orient(rng, tiny)
    lo, hi = (5, 6) if small else (5, 9)
    h, w = int(rng.integers(lo, hi)), int(rng.integers(lo, hi))
    m = np.ones((h, w), bool)
    if count == 0:
        if style == 'open' or rng.random() < .65:
            # A sealed-looking chamber with an actual orthogonal door.
            wall = 1 if min(h, w) < 7 else int(rng.integers(1, 3))
            m[wall:-wall, wall:-wall] = False
            door = int(rng.integers(wall, w - wall))
            m[:wall, door] = False
        else:
            m[:] = False
            m[h // 2, :] = True
            m[:, int(rng.integers(w))] = True
            for _ in range(int(rng.integers(1, 5))):
                rr, cc = np.argwhere(m)[int(rng.integers(m.sum()))]
                mh, mw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
                trial = m.copy()
                trial[rr:min(h, rr + mh), cc:min(w, cc + mw)] = True
                if _clear_topology(trial):
                    m = trial
            m |= _holes(m)
    else:
        # Separated interior corner seeds guarantee capacity even at 5x5.
        positions = [(1, 1), (1, w - 2), (h - 2, 1), (h - 2, w - 2)]
        hole = np.zeros_like(m)
        if style == 'large' and count == 1:
            hole[1:-1, 1:-1] = True
            if h >= 6 and w >= 6 and rng.random() < .6:
                hole[-2, -2] = False
        else:
            for p in rng.permutation(4)[:count]:
                hole[positions[int(p)]] = True
            if style != 'tiny':
                budget = int(rng.integers(0, max(2, (h - 2) * (w - 2) // 2)))
                for _ in range(budget):
                    candidates = np.argwhere(_dilate(hole) & ~hole)
                    rng.shuffle(candidates)
                    for rr, cc in candidates:
                        if not (1 <= rr < h - 1 and 1 <= cc < w - 1):
                            continue
                        trial = hole.copy()
                        trial[rr, cc] = True
                        if (_clear_topology(trial) and len(_regions(trial)) == count
                                and len(_regions(~trial)) == 1):
                            hole = trial
                            break
                    else:
                        break
        m &= ~hole
        # Uneven outside cuts preserve every original cavity and its walls.
        protected = hole | _dilate(hole)
        for _ in range(int(rng.integers(0, 6))):
            trial = m.copy()
            corner = int(rng.integers(4))
            rh, cw = int(rng.integers(1, 3)), int(rng.integers(1, 3))
            rs = slice(0, rh) if corner < 2 else slice(h - rh, h)
            cs = slice(0, cw) if corner % 2 == 0 else slice(w - cw, w)
            trial[rs, cs] &= protected[rs, cs]
            if (_clear_topology(trial) and len(_regions(trial)) == 1
                    and np.array_equal(_holes(trial), hole)):
                m = trial
    result = _orient(rng, m)
    _require(len(_regions(result)) == 1 and len(_regions(_holes(result))) == count, "shape topology")
    return result


def _card(rng, count, ink, fill):
    h, w = _choice(rng, [(3, 7), (4, 7), (5, 5), (5, 6), (6, 5)])
    pips = np.zeros((h, w), bool)
    # A finite greedy independent set, with a capacity-guaranteed interior row.
    for index in rng.permutation((h - 2) * (w - 2)):
        r, c = divmod(int(index), w - 2)
        r, c = r + 1, c + 1
        if all(not pips[r + dr, c + dc] for dr, dc in N4):
            pips[r, c] = True
            if pips.sum() == count:
                break
    if pips.sum() != count:
        pips = np.zeros((5, 5), bool)
        for r, c in [(1, 1), (1, 3), (3, 1)][:count]:
            pips[r, c] = True
    return np.where(pips, fill, ink).astype(np.int8)


def _put(mask, r, c):
    result = np.zeros((30, 30), bool)
    h, w = mask.shape
    result[r:r + h, c:c + w] = mask
    return result


def _positions(h, w, boxes, gap=1):
    rr, cc = np.indices((31 - h, 31 - w))
    valid = np.ones(rr.shape, bool)
    for r, c, bh, bw in boxes:
        valid &= ((rr + h + gap <= r) | (r + bh + gap <= rr)
                  | (cc + w + gap <= c) | (c + bw + gap <= cc))
    return np.argwhere(valid)


def _layout(rng, masks, path_length, active_count, card, address_foil=False):
    """Free origins; required path edges and one nonconducting near miss."""
    boxes, bodies, halos = [], [], []
    for index, mask in enumerate(masks):
        h, w = mask.shape
        padded = np.pad(mask, 1)
        local_halo = _dilate(padded) & ~padded & ~np.pad(_holes(mask), 1)
        origins = _positions(h, w, boxes)
        rng.shuffle(origins)
        found = None
        # All candidates are bounded by the 30x30 canvas; stop at first fit.
        active_union = np.logical_or.reduce(halos[:active_count]) if halos else np.zeros((30, 30), bool)
        active_touch = _dilate(active_union)
        for rr, cc in origins:
            r, c = int(rr), int(cc)
            if index == 0 and not (3 <= r <= 21 - h // 2 and 3 <= c <= 21 - w // 2):
                continue
            halo = np.zeros((30, 30), bool)
            r0, c0, r1, c1 = max(0, r - 1), max(0, c - 1), min(30, r + h + 1), min(30, c + w + 1)
            halo[r0:r1, c0:c1] = local_halo[r0 - r + 1:r1 - r + 1, c0 - c + 1:c1 - c + 1]
            contacts = {i for i, old in enumerate(halos) if np.any(old & halo)}
            if index and index <= path_length and contacts != {index - 1}:
                continue
            if path_length < index < active_count and not contacts:
                continue
            if path_length == 2 and path_length < index < active_count and not (contacts & {0, 1}):
                continue
            if path_length == 3 and path_length < index < active_count and (not (contacts & {0, 1}) or contacts & {2, 3}):
                continue
            if index >= active_count:
                if any(i < active_count for i in contacts):
                    continue
                if index == active_count and not np.any(halo & active_touch):
                    continue
            found = (r, c, _put(mask, r, c), halo)
            break
        if found is None:
            return None
        r, c, body, halo = found
        boxes.append((r, c, h, w))
        bodies.append(body)
        halos.append(halo)
    origins = _positions(*card.shape, boxes, gap=2)
    if not len(origins):
        return None
    if address_foil:
        # The first card is closer to an inactive body than the source under
        # common cell-distance metrics; source addressing cannot be proximity.
        def distances(origin, body):
            rr, cc = np.nonzero(body)
            r, c = origin
            dr = np.maximum.reduce([r - rr, rr - (r + card.shape[0] - 1), np.zeros_like(rr)])
            dc = np.maximum.reduce([c - cc, cc - (c + card.shape[1] - 1), np.zeros_like(cc)])
            return np.array([(dr + dc).min(), np.maximum(dr, dc).min(), (dr * dr + dc * dc).min()])
        origins = [p for p in origins if np.all(distances(p, bodies[active_count]) < distances(p, bodies[0]))]
        if not len(origins):
            return None
    r, c = map(int, origins[int(rng.integers(len(origins)))])
    return bodies, (r, c)


def _fallback(rng, key, test, colors):
    """Small proven packing, used only after bounded free-placement attempts."""
    other = 3 if key == 1 else 1
    inactive = next(v for v in (2, 3, 1) if v not in (key, other))
    counts = [key, 0, other] + ([0] if test else []) + [inactive]
    masks = []
    for index, k in enumerate(counts):
        m = np.ones((5, 5), bool)
        if k == 0:
            m[1:4, 1:4] = False
            m[0, 2] = False
        elif k == 1:
            m[1:4, 1:4] = False
        else:
            for p in [(1, 1), (1, 3), (3, 1)][:k]:
                m[p] = False
        masks.append(_orient(rng, m))
    origins = [(2, 2), (2, 8), (2, 14)] + ([(8, 16)] if test else []) + [(9, 8)]
    bodies = [_put(m, *p) for m, p in zip(masks, origins)]
    card = _card(rng, key, colors[2], colors[3])
    return bodies, card, (21, 2)


def _make_pair(rng, key, path_length, first=False):
    colors = list(map(int, rng.permutation(10)[:4]))
    bg, fg, ink, fill = colors
    other = _choice(rng, [v for v in (1, 2, 3) if v != key])
    if first:
        other = 3
    inactive = _choice(rng, [v for v in (1, 2, 3) if v not in (key, other)])
    # Every pair contains a hollow endpoint, a cavity-free relay, and an inactive hollow.
    counts = [key, 0, other] + ([0] if path_length == 3 else [])
    branch_count = int(rng.integers(0, 2))
    counts += [_choice(rng, [0, other]) for _ in range(branch_count)]
    active_count = len(counts)
    counts += [inactive]
    if rng.random() < .35:
        counts += [_choice(rng, [0, inactive])]
    card = _card(rng, key, ink, fill)
    packed = None
    for attempt in range(12):
        masks = [_shape(rng, k, 'large' if first and i == 0 else
                        'tiny' if first and i == active_count else
                        'open' if i == 1 else 'ordinary', small=attempt >= 6)
                 for i, k in enumerate(counts)]
        packed = _layout(rng, masks, path_length, active_count, card, address_foil=first)
        if packed is not None:
            bodies, origin = packed
            break
    else:
        bodies, card, origin = _fallback(rng, key, path_length == 3, colors)
    inp = np.full((30, 30), bg, np.int8)
    for b in bodies:
        inp[b] = fg
    r, c = origin
    inp[r:r + card.shape[0], c:c + card.shape[1]] = card
    # Crop only empty margins; source geometry and visible orientation stay intact.
    occupied = inp != bg
    r, c, s, t = _box(occupied)
    margins = rng.integers(0, 4, size=4)
    r, c, s, t = max(0, r - int(margins[0])), max(0, c - int(margins[1])), min(30, s + int(margins[2])), min(30, t + int(margins[3]))
    candidate = inp[r:s, c:t].copy()
    if np.count_nonzero(candidate == bg) > candidate.size // 2:
        inp = candidate
    a = _analysis(inp)
    _require(max(a['distance'].values()) >= path_length, "packing lost path depth")
    return {'input': inp, 'output': _paint(a, a['distance'])}


def _key(grid):
    a = np.asarray(grid, np.int8)
    return a.shape, a.tobytes()


def _fresh_colors(pair, seen):
    present = list(map(int, np.unique(pair['input'])))
    # Ninety distinct images of the first two visible colors exceed 13 prior inputs.
    for a, b in itertools.permutations(range(10), 2):
        target = [a, b] + [c for c in range(10) if c not in (a, b)]
        source = present + [c for c in range(10) if c not in present]
        lookup = np.empty(10, np.int8)
        lookup[source] = target
        fresh = {k: lookup[v] for k, v in pair.items()}
        if _key(fresh['input']) not in seen:
            return fresh
    raise AssertionError('uniqueness bound')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError('illegal train/test count')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    keys = [1, int(rng.integers(2, 4))]
    keys += [int(rng.integers(1, 4)) for _ in range(num_train - 2)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', num_train), ('test', num_test)):
        for index in range(n):
            key = keys[index] if split == 'train' else _choice(rng, sorted(set(keys)))
            pair = _make_pair(rng, key, 2 if split == 'train' else 3, first=split == 'train' and index == 0)
            if _key(pair['input']) in seen:
                pair = _fresh_colors(pair, seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    active = set(a['distance'])
    inactive = set(range(len(a['objects']))) - active
    touching = any(np.any(_dilate(a['halos'][i]) & a['halos'][j])
                   for i in active for j in inactive)
    inversion = any(a['counts'][i] < a['counts'][j] and a['holes'][i].sum() > a['holes'][j].sum()
                    for i in range(len(a['objects'])) for j in range(len(a['objects']))
                    if a['counts'][i] and a['counts'][j])
    root_inversion = any(a['counts'][j] > a['key'] and a['holes'][j].sum() < a['holes'][a['root']].sum()
                         for j in range(len(a['objects'])))
    open_relay = any(a['counts'][i] == 0 and _holes(a['objects'][i] | a['halos'][i]).any()
                     for i in active)
    return dict(objects=len(a['objects']), active=len(active), depth=max(a['distance'].values()),
                key=a['key'], touching=touching, inversion=inversion, root_inversion=root_inversion, open_relay=open_relay,
                active_solid=any(a['counts'][i] == 0 for i in active),
                active_other_hollow=any(a['counts'][i] and i != a['root'] for i in active),
                inactive_hollow=any(a['counts'][i] for i in inactive))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "pair counts")
    seen, features, keys = set(), {'train': [], 'test': []}, {'train': set(), 'test': set()}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _analysis(inp)
            f = _features(a)
            active = set(a['distance'])
            _require(out.shape == inp.shape and np.array_equal(out, _paint(a, active)), "incorrect output")
            _require(all(f[v] for v in ('touching', 'active_solid', 'active_other_hollow', 'inactive_hollow')), "missing circuit contrast")
            _require(f['depth'] == 2 if split == 'train' else f['depth'] >= 3, "missing relay depth shift")
            # Actual counterfactual answers, rather than sampler flags, prove both mechanisms matter.
            _require(not np.array_equal(out, _paint(a, range(len(a['objects'])))), "ignoring address has no effect")
            _require(not np.array_equal(out, _paint(a, [a['root']])), "ignoring halo propagation has no effect")
            features[split].append(f)
            keys[split].add(a['key'])
    _require(len(keys['train']) >= 2 and keys['test'] <= keys['train'], "address curriculum")
    _require(any(f['root_inversion'] and f['key'] == 1 for f in features['train']), "area-count shortcut unrefuted")
    _require(any(f['open_relay'] for f in features['train']), "original open bay must remain unfilled")


def _regressions():
    # A hand-positioned chain with fixed expected membership and pixel colors.
    inp = np.zeros((24, 29), np.int8)
    fixtures = []
    for count in (1, 0, 2, 3):
        m = np.ones((5, 5), bool)
        if count <= 1:
            m[1:4, 1:4] = False
            if not count:
                m[0, 2] = False
        else:
            for p in [(1, 1), (1, 3), (3, 1)][:count]:
                m[p] = False
        fixtures.append(m)
    origins = [(2, 2), (2, 8), (2, 14), (2, 21)]
    for m, (r, c) in zip(fixtures, origins):
        inp[r:r + 5, c:c + 5][m] = 1
    inp[15:20, 2:7] = 2
    inp[16, 3] = 3
    a = _analysis(inp)
    _require(a['graph'] == [{1}, {0, 2}, {1}, set()] and a['distance'] == {0: 0, 1: 1, 2: 2}, "fixed graph")
    out = solve(inp)
    expected = inp.copy()
    # Explicit neighbor loops, with fixed fixture membership, construct the label.
    original_holes = np.logical_or.reduce(a['holes'])
    for i in (0, 1, 2):
        for r, c in np.argwhere(a['objects'][i]):
            for dr, dc in N8:
                rr, cc = int(r) + dr, int(c) + dc
                if 0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1] and inp[rr, cc] == 0 and not original_holes[rr, cc]:
                    expected[rr, cc] = 2
        if i != 1:
            expected[a['objects'][i]] = 2
            expected[a['holes'][i]] = 3
    _require(np.array_equal(out, expected), "fixed circuit label")
    _require(out[4, 10] == 0 and out[2, 10] == 2 and out[3, 8] == 1, "original open-bay topology")
    _require(np.all(out[a['objects'][3]] == 1) and np.all(out[a['holes'][3]] == 0), "touching is not sharing")
    # Diagonal halo overlap conducts; diagonal halo contact does not.
    m, n = np.zeros((10, 10), bool), np.zeros((10, 10), bool)
    m[2, 2], n[4, 4] = True, True
    _require(np.array_equal(_dilate(m) & _dilate(n), _mask([(3, 3)], m.shape)), "diagonal shared cell")
    n[:] = False
    n[5, 5] = True
    _require(not np.any(_dilate(m) & _dilate(n)) and np.any(_dilate(_dilate(m)) & _dilate(n)), "diagonal touching only")
    # Bounding boxes approach diagonally but their empty corners do not conduct.
    m[:], n[:] = False, False
    m[2, 2:5] = m[2:5, 2] = True
    n[8, 6:9] = n[6:9, 8] = True
    _require(not np.any(_dilate(m) & _dilate(n)), "box proximity is not mask contact")


def self_test():
    """Forty-two seeds, streamed; no retained puzzles, files, traces or RNG state."""
    _regressions()
    dimensions, masks, graphs, palettes, fingerprints = set(), set(), set(), set(), set()
    totals = dict(pairs=0, open_relays=0, area_inversions=0, branching=0, clipped=0, edge_outlines=0, irregular_cavities=0)
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                inp, out = pair['input'], pair['output']
                _require(np.array_equal(solve(inp.copy()), out), "input-only solving")
                a, f = _analysis(inp), None
                f = _features(a)
                dimensions.add(inp.shape)
                palettes.add((a['background'], a['foreground'], a['ink'], a['fill']))
                graphs.add((tuple(sorted(a['counts'])), tuple(sorted(map(len, a['graph']))), f['depth']))
                digest.update(inp.tobytes())
                totals['pairs'] += 1
                totals['open_relays'] += int(f['open_relay'])
                totals['area_inversions'] += int(f['inversion'])
                totals['branching'] += any(len(v) >= 3 for v in a['graph'])
                halo = np.logical_or.reduce([a['halos'][i] for i in a['distance']])
                totals['edge_outlines'] += bool(halo[0].any() or halo[-1].any() or halo[:, 0].any() or halo[:, -1].any())
                active_body = np.logical_or.reduce([a['objects'][i] for i in a['distance']])
                totals['clipped'] += bool(active_body[0].any() or active_body[-1].any() or active_body[:, 0].any() or active_body[:, -1].any())
                for body, holes in zip(a['objects'], a['holes']):
                    m = _crop(body)
                    canonical = min(np.rot90(v, k).tobytes() for v in (m, m.T) for k in range(4))
                    masks.add((min(m.shape), max(m.shape), canonical))
                    for cells in _regions(holes):
                        h = _crop(_mask(cells, holes.shape))
                        totals['irregular_cavities'] += not bool(h.all())
        fingerprints.add(digest.digest())
    _require(len(fingerprints) == 42 and len(dimensions) >= 25, "seed and canvas diversity")
    _require(len(masks) >= 100 and len(graphs) >= 20 and len(palettes) >= 100, "structural and role diversity")
    _require(all(totals[k] >= 3 for k in totals if k != 'pairs'), "missing geometric variation")
    # Every legal explicit request, plus type and boundary errors.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(733 + nt * 10 + ne, nt, ne)
            _require(len(p['train']) == nt and len(p['test']) == ne, "explicit counts")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p, q = generate_puzzle(0, 2, 1), generate_puzzle(0, 2, 1)
    _require(all(np.array_equal(p[s][i][v], q[s][i][v]) for s in p for i in range(len(p[s])) for v in ('input', 'output')), "determinism")
    for role in ('card', 'body', 'cavity', 'outline', 'background'):
        broken = {s: [{v: a.copy() for v, a in pair.items()} for pair in p[s]] for s in p}
        pair = broken['test'][0]
        a = _analysis(pair['input'])
        active = list(a['distance'])
        hollow = next(i for i in active if a['counts'][i])
        zones = {'card': a['card'], 'body': a['objects'][hollow], 'cavity': a['holes'][hollow],
                 'outline': a['halos'][a['root']], 'background': (pair['output'] == a['background'])}
        r, c = np.argwhere(zones[role])[0]
        pair['output'][r, c] = (int(pair['output'][r, c]) + 1) % 10
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError('corruption accepted: ' + role)
    return dict(seeds=42, canvas_shapes=len(dimensions), object_shapes=len(masks), graph_signatures=len(graphs),
                palettes=len(palettes), **totals)
