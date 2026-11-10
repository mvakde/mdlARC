"""Keyed multi-pin stencils: overlap registration plus conjunctive stamping.

Only standard Python and NumPy are used. Generation, validation and self-test
write no files, retain no traces and keep at most one puzzle under construction.
"""
from collections import deque
from hashlib import sha256
import numpy as np

NEUTRAL = 8
KEY = 9
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _groups(points, diagonal=True):
    left = set(points)
    groups = []
    steps = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or not (r and c)))
    while left:
        p = min(left)
        left.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _origin(points):
    return min(r for r, c in points), min(c for r, c in points)


def _normal(points):
    r0, c0 = _origin(points)
    return tuple(sorted((r-r0, c-c0) for r, c in points))


def _shift(points, delta):
    return {(r+delta[0], c+delta[1]) for r, c in points}


def _halo(points, radius=1):
    return {(r+dr, c+dc) for r, c in points
            for dr in range(-radius, radius+1) for dc in range(-radius, radius+1)}


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'integer cells required')
    _require(bool(np.all((a >= 0) & (a <= 9))), 'ARC colors required')
    return a.astype(np.int8, copy=False)


def _background(a):
    counts = np.bincount(a.ravel(), minlength=10)
    color = int(counts.argmax())
    _require(counts[color]*2 > a.size and color not in (NEUTRAL, KEY), 'dominant background')
    return color


def _parse(a):
    """Read pieces, local label rays and exact whole-key equivalence classes."""
    bg = _background(a)
    support = {tuple(map(int, p)) for p in np.argwhere((a == NEUTRAL) | (a == KEY))}
    components = _groups(support)
    _require(2 <= len(components) <= 4, 'need two to four stencil pieces')
    labels = {tuple(map(int, p)) for p in np.argwhere((a != bg) & (a < NEUTRAL))}
    pieces = []
    all_labels = set()
    for cells in components:
        mask = {p for p in cells if a[p] == NEUTRAL}
        key_cells = cells-mask
        _require(mask and key_cells, 'each piece needs a mask and keys')
        near = _halo(cells)
        candidates = []
        for sr, sc in sorted(labels):
            source = int(a[sr, sc])
            for dr, dc in DIRECTIONS:
                t, q = (sr+dr, sc+dc), (sr+2*dr, sc+2*dc)
                if q not in near or not (0 <= q[0] < a.shape[0] and 0 <= q[1] < a.shape[1]):
                    continue
                target = int(a[t])
                if target in (bg, NEUTRAL, KEY, source) or int(a[q]) not in (bg, NEUTRAL):
                    continue
                candidates.append((source, target, q, (sr, sc), t, (dr, dc)))
        _require(len(candidates) == 1, 'ambiguous or absent registration ray')
        source, target, q, sl, tl, direction = candidates[0]
        _require(not ({sl, tl} & all_labels), 'shared label')
        all_labels.update((sl, tl))
        keys = []
        for group in _groups(key_cells):
            _require(3 <= len(group) <= 6, 'whole key size')
            keys.append((_normal(group), _origin(group)))
        _require(len({k for k, p in keys}) == len(keys), 'key repeated within a piece')
        pieces.append({'cells': cells, 'mask': mask, 'keys': keys, 'q': q,
                       'source': source, 'target': target, 'labels': {sl, tl},
                       'direction': direction})
    _require(len({p['source'] for p in pieces}) == len(pieces), 'distinct pin colors required')
    targets = {p['target'] for p in pieces}
    _require(len(targets) == 1 and targets.isdisjoint({p['source'] for p in pieces}), 'one inactive target color')
    occurrences = {}
    for i, piece in enumerate(pieces):
        for key, origin in piece['keys']:
            occurrences.setdefault(key, []).append((i, origin))
    _require(all(len(v) == 2 for v in occurrences.values()), 'each whole key occurs in two pieces')
    adjacency = [[] for _ in pieces]
    for copies in occurrences.values():
        (i, p), (j, q) = copies
        dr, dc = p[0]-q[0], p[1]-q[1]
        adjacency[i].append((j, (dr, dc)))
        adjacency[j].append((i, (-dr, -dc)))
    offsets = {0: (0, 0)}
    stack = [0]
    while stack:
        i = stack.pop()
        for j, delta in adjacency[i]:
            proposed = offsets[i][0]+delta[0], offsets[i][1]+delta[1]
            if j in offsets:
                _require(offsets[j] == proposed, 'inconsistent key cycle')
            else:
                offsets[j] = proposed
                stack.append(j)
    _require(len(offsets) == len(pieces), 'disconnected key graph')
    assembled_mask, assembled_keys, pins, placed_masks = set(), set(), [], []
    for i, piece in enumerate(pieces):
        delta = offsets[i]
        mask = _shift(piece['mask'], delta)
        placed_masks.append(mask)
        assembled_mask |= mask
        assembled_keys |= _shift(piece['cells']-piece['mask'], delta)
        pins.append((piece['q'][0]+delta[0], piece['q'][1]+delta[1], piece['source']))
    _require(not (assembled_mask & assembled_keys), 'mask/key collision after registration')
    _require(len({(r, c) for r, c, s in pins}) == len(pins), 'coincident pins')
    qr, qc, _ = pins[0]
    normalized_mask = {(r-qr, c-qc) for r, c in assembled_mask}
    normalized_keys = {(r-qr, c-qc) for r, c in assembled_keys}
    normalized_pins = [(r-qr, c-qc, s) for r, c, s in pins]
    for piece in pieces:
        piece['local_mask'] = {(r-piece['q'][0], c-piece['q'][1]) for r, c in piece['mask']}
    return {'background': bg, 'pieces': pieces, 'labels': all_labels,
            'mask': normalized_mask, 'keys': normalized_keys, 'pins': normalized_pins,
            'target': next(iter(targets)), 'placed_masks': placed_masks,
            'key_count': len(occurrences), 'offsets': offsets}


def _matches(a, info):
    pins, labels = info['pins'], info['labels']
    pivot = pins[0][2]
    result = []
    for r, c in np.argwhere(a == pivot):
        r, c = int(r), int(c)
        anchors = {(r+dr, c+dc) for dr, dc, s in pins}
        if anchors & labels:
            continue
        if all(0 <= r+dr < a.shape[0] and 0 <= c+dc < a.shape[1]
               and int(a[r+dr, c+dc]) == s for dr, dc, s in pins):
            result.append((r, c))
    return result


def _evaluate(a, info, matches=None, mask=None):
    matches = _matches(a, info) if matches is None else matches
    mask = info['mask'] if mask is None else mask
    out = a.copy()
    out[(a == NEUTRAL) | (a == KEY)] = info['background']
    for p in info['labels']:
        out[p] = info['background']
    for r, c in matches:
        for dr, dc, s in info['pins']:
            out[r+dr, c+dc] = info['background']
    for r, c in matches:
        for dr, dc in mask:
            if 0 <= r+dr < a.shape[0] and 0 <= c+dc < a.shape[1]:
                out[r+dr, c+dc] = info['target']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode only this grid; no examples, seed or generation records are used."""
    a = _grid(input_grid)
    return _evaluate(a, _parse(a))


def _independent_prints(a, info):
    """A deliberately wrong B-only rule used to check discriminating examples."""
    out = a.copy()
    out[(a == NEUTRAL) | (a == KEY)] = info['background']
    for p in info['labels']:
        out[p] = info['background']
    prints = set()
    for piece in info['pieces']:
        for r, c in np.argwhere(a == piece['source']):
            if (int(r), int(c)) in info['labels']:
                continue
            out[r, c] = info['background']
            prints |= _shift(piece['local_mask'], (int(r), int(c)))
    for r, c in prints:
        if 0 <= r < a.shape[0] and 0 <= c < a.shape[1]:
            out[r, c] = info['target']
    return out


def _features(a, info):
    matches = _matches(a, info)
    masks = info['placed_masks']
    exclusive = [len(mask-set().union(*(m for j, m in enumerate(masks) if j != i)))
                 for i, mask in enumerate(masks)]
    anchors = {(r+dr, c+dc) for r, c in matches for dr, dc, s in info['pins']}
    source_colors = {s for dr, dc, s in info['pins']}
    external = {tuple(map(int, p)) for p in np.argwhere(np.isin(a, list(source_colors)))}-info['labels']
    complete = sum(all(0 <= r+dr < a.shape[0] and 0 <= c+dc < a.shape[1]
                       for dr, dc in info['mask']) for r, c in matches)
    clipped = len(matches)-complete
    holes = sum(a[p['q']] == info['background'] for p in info['pieces'])
    return {'pieces': len(masks), 'keys': info['key_count'], 'matches': len(matches),
            'decoys': len(external-anchors), 'complete': complete, 'clipped': clipped,
            'hole_pins': int(holes), 'exclusive': exclusive,
            'mask_size': len(info['mask']), 'input_shape': tuple(a.shape),
            'mask_shape': (max(r for r, c in info['mask'])-min(r for r, c in info['mask'])+1,
                           max(c for r, c in info['mask'])-min(c for r, c in info['mask'])+1),
            'directions': tuple(sorted({p['direction'] for p in info['pieces']}))}


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _key_shape(rng):
    cells = {(1, 1)}
    desired = int(rng.integers(3, 7))
    for _ in range(80):
        if len(cells) == desired:
            break
        r, c = _pick(rng, sorted(cells))
        dr, dc = _pick(rng, DIRECTIONS)
        if 0 <= r+dr < 3 and 0 <= c+dc < 3:
            cells.add((r+dr, c+dc))
    return set(_normal(cells))


def _path(start, goals, forbidden, bounds):
    queue, previous = deque([start]), {start: None}
    end = None
    while queue:
        p = queue.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in DIRECTIONS:
            q = p[0]+dr, p[1]+dc
            if (bounds[0] <= q[0] <= bounds[2] and bounds[1] <= q[1] <= bounds[3]
                    and q not in forbidden and q not in previous):
                previous[q] = p
                queue.append(q)
    _require(end is not None, 'path blocked')
    cells = set()
    while end is not None:
        cells.add(end)
        end = previous[end]
    return cells


def _make_kit(rng, n, sources, target, hole, compact=False):
    if n == 2:
        edges = [(0, 1)]
    elif n == 3:
        edges = [(0, 1), (1, 2)]
        if rng.random() < .3:
            edges.append((0, 2))
    else:
        edges = _pick(rng, [[(0, 1), (1, 2), (2, 3)],
                            [(0, 1), (0, 2), (0, 3)],
                            [(0, 1), (1, 2), (2, 3), (3, 0)]])
    spacing = 4 if compact else int(rng.integers(4, 7))
    origins = [(0, 0), (int(rng.integers(0, 2)), spacing),
               (spacing, int(rng.integers(0, 2))), (spacing, spacing)]
    if len(edges) == 2 and rng.random() < .5:
        origins[1] = (spacing, int(rng.integers(2, spacing+1)))
    shapes, keys = set(), []
    for i in range(len(edges)):
        for _ in range(80):
            shape = _key_shape(rng)
            signature = _normal(shape)
            if signature not in shapes:
                shapes.add(signature)
                keys.append(_shift(shape, origins[i]))
                break
        else:
            raise AssertionError('key diversity exhausted')
    all_keys = set().union(*keys)
    bounds = (-7, -7, spacing+9, spacing+9)
    supports = []
    for i in range(n):
        own = [keys[e] for e, (a, b) in enumerate(edges) if i in (a, b)]
        part = set().union(*own)
        connected = set(own[0])
        for more in own[1:]:
            path = _path(min(more), connected, all_keys-part, bounds)
            connected |= path | more
        mask = connected-all_keys
        protected = set()
        forced_q = None
        if hole and i == 0:
            # A C-shaped tab supplies an interior, visibly empty registration.
            qr, qc = min(r for r, c in part)+1, min(c for r, c in part)-3
            forced_q = (qr, qc)
            protected = {(qr, qc), (qr, qc-1), (qr, qc-2)}
            tab = {(qr+dr, qc+dc) for dr in (-1, 1) for dc in (-1, 0, 1)} | {(qr, qc+1)}
            _require(not (tab & all_keys), 'tab hits key')
            path = _path((qr, qc+1), part | mask, (all_keys-part) | protected, bounds)
            mask |= (path | tab)-all_keys
        occupied = part | mask
        desired = int(rng.integers(4, 9 if compact else 16))
        for _ in range(desired*8):
            if len(mask) >= desired+len(connected-all_keys):
                break
            r, c = _pick(rng, sorted(occupied))
            dr, dc = _pick(rng, DIRECTIONS)
            q = r+dr, c+dc
            if (q not in all_keys and q not in protected
                    and bounds[0]+2 <= q[0] <= bounds[2]-2 and bounds[1]+2 <= q[1] <= bounds[3]-2):
                mask.add(q)
                occupied.add(q)
        _require(mask and len(_groups(occupied)) == 1, 'disconnected piece')
        near = _halo(occupied)
        candidates = []
        for q in sorted(mask if forced_q is None else {forced_q}):
            for dr, dc in DIRECTIONS:
                sl, tl = (q[0]-2*dr, q[1]-2*dc), (q[0]-dr, q[1]-dc)
                reverse_q = q[0]-3*dr, q[1]-3*dc
                if sl in occupied or tl in occupied or reverse_q in near:
                    continue
                candidates.append((q, sl, tl, (dr, dc)))
        _require(candidates, 'no unambiguous pin ray')
        q, sl, tl, direction = _pick(rng, candidates)
        cells = {p: NEUTRAL for p in mask}
        cells.update({p: KEY for p in part})
        cells[sl], cells[tl] = sources[i], target
        supports.append({'cells': cells, 'mask': mask, 'keys': part, 'q': q,
                         'source': sources[i], 'labels': {sl, tl}, 'direction': direction})
    union = set().union(*(p['mask'] for p in supports))
    _require(not union & all_keys, 'assembled color conflict')
    _require(len({p['q'] for p in supports}) == n, 'coincident pins')
    _require(all(p['mask']-set().union(*(q['mask'] for j, q in enumerate(supports) if j != i))
                 for i, p in enumerate(supports)), 'each piece must contribute')
    # One common rigid symmetry changes the kit, not any piece during solving.
    turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
    def transform(p):
        r, c = p
        if reflect:
            r = -r
        for _ in range(turns):
            r, c = c, -r
        return r, c
    for p in supports:
        p['cells'] = {transform(q): v for q, v in p['cells'].items()}
        for name in ('mask', 'keys', 'labels'):
            p[name] = {transform(q) for q in p[name]}
        p['q'] = transform(p['q'])
    return supports


def _inside(points, shape):
    return {(r, c) for r, c in points if 0 <= r < shape[0] and 0 <= c < shape[1]}


def _build_pair(rng, n, bg, colors, index, testing, attempt):
    sources, target = colors[:n], colors[n]
    hole = index == 1 or (index > 1 and rng.random() < .45) or (testing and rng.random() < .5)
    kit = _make_kit(rng, n, sources, target, hole, compact=attempt >= 8)
    mask = set().union(*(p['mask'] for p in kit))
    pins = [(p['q'][0], p['q'][1], p['source']) for p in kit]
    shape = ((30, 30) if attempt >= 5 else
             (int(rng.integers(21 if n == 2 else 27, 31)),
              int(rng.integers(21 if n == 2 else 27, 31))))
    a = np.full(shape, bg, dtype=np.int8)
    footprint = mask | {(r, c) for r, c, s in pins}
    r0, c0 = _origin(footprint)
    r1, c1 = max(r for r, c in footprint), max(c for r, c in footprint)
    locations = [(r, c) for r in range(-r0, shape[0]-r1)
                 for c in range(-c0, shape[1]-c1)]
    _require(locations, 'complete stencil does not fit')
    delta = _pick(rng, locations)
    placements = [delta]
    painted = _shift(mask, delta)
    used_anchors = _shift({(r, c) for r, c, s in pins}, delta)
    for r, c, s in pins:
        a[r+delta[0], c+delta[1]] = s
    desired = 1 if index == 0 and not testing else int(rng.integers(2, 4))
    # Additional occurrences may overlap and may clip; pins themselves stay visible.
    pmin = _origin({(r, c) for r, c, s in pins})
    pmax = max(r for r, c, s in pins), max(c for r, c, s in pins)
    possible = [(r, c) for r in range(-pmin[0], shape[0]-pmax[0])
                for c in range(-pmin[1], shape[1]-pmax[1])]
    rng.shuffle(possible)
    if index == 1:
        possible.sort(key=lambda d: len(_inside(_shift(mask, d), shape)) == len(mask))
    for d in possible:
        if len(placements) >= desired:
            break
        anchors = _shift({(r, c) for r, c, s in pins}, d)
        paint = _inside(_shift(mask, d), shape)
        if anchors & used_anchors or not paint:
            continue
        if index < 2 and paint & painted:
            continue
        for r, c, s in pins:
            a[r+d[0], c+d[1]] = s
        used_anchors |= anchors
        painted |= paint
        placements.append(d)
    _require(len(placements) >= min(desired, 2), 'not enough placements')
    reserved = _halo(painted | used_anchors)
    instruction_cells, piece_halo = set(), set()
    for p in sorted(kit, key=lambda p: -len(p['cells'])):
        cells = set(p['cells']) | {p['q']}
        r0, c0 = _origin(cells)
        r1, c1 = max(r for r, c in cells), max(c for r, c in cells)
        positions = [(r, c) for r in range(-r0, shape[0]-r1)
                     for c in range(-c0, shape[1]-c1)]
        rng.shuffle(positions)
        for d in positions:
            shifted = _shift(cells, d)
            if shifted & (reserved | piece_halo):
                continue
            for (r, c), color in p['cells'].items():
                a[r+d[0], c+d[1]] = color
            instruction_cells |= _shift(set(p['cells']), d)
            piece_halo |= _halo(shifted)
            break
        else:
            raise AssertionError('pieces do not fit')
    info = _parse(a)
    found = _matches(a, info)
    _require(len(found) == len(placements), 'unintended constellation')
    forbidden = _halo(instruction_cells, 2) | painted
    # A nearly complete false constellation teaches the conjunctive lock.
    planted = False
    rng.shuffle(possible)
    missing = n-1
    for d in possible:
        partial = [(r+d[0], c+d[1], s) for j, (r, c, s) in enumerate(pins) if j != missing]
        if any((r, c) in forbidden or a[r, c] != bg for r, c, s in partial):
            continue
        for r, c, s in partial:
            a[r, c] = s
        if len(_matches(a, info)) == len(found):
            planted = True
            break
        for r, c, s in partial:
            a[r, c] = bg
    _require(planted, 'no incomplete constellation space')
    # Extra unmatched source pixels and inactive pixels vary independently.
    all_positions = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    rng.shuffle(all_positions)
    extras = int(rng.integers(0, 4))
    for r, c in all_positions:
        if extras == 0:
            break
        if (r, c) in forbidden or a[r, c] != bg:
            continue
        a[r, c] = _pick(rng, sources)
        if len(_matches(a, info)) != len(found):
            a[r, c] = bg
        else:
            extras -= 1
    inactive = [target] + colors[n+1:]
    for _ in range(int(rng.integers(0, 5))):
        for r, c in all_positions:
            if a[r, c] == bg and (r, c) not in forbidden:
                a[r, c] = _pick(rng, inactive)
                break
    # Construction uses the sampled kit, independently of input decoding.
    out = a.copy()
    for p in instruction_cells | used_anchors:
        out[p] = bg
    for p in painted:
        out[p] = target
    # Keep prospective paint when reducing only unused outer margins.
    visible = {tuple(map(int, p)) for p in np.argwhere(a != bg)} | painted
    r0, c0 = _origin(visible)
    r1, c1 = max(r for r, c in visible), max(c for r, c in visible)
    r0, c0 = max(0, r0-int(rng.integers(3))), max(0, c0-int(rng.integers(3)))
    r1, c1 = min(shape[0], r1+1+int(rng.integers(3))), min(shape[1], c1+1+int(rng.integers(3)))
    a, out = a[r0:r1, c0:c1].copy(), out[r0:r1, c0:c1].copy()
    decoded = _parse(a)
    features = _features(a, decoded)
    _require(np.array_equal(_evaluate(a, decoded), out), 'sampled and decoded outputs disagree')
    _require(features['complete'] >= 1 and features['decoys'] >= n-1, 'teaching witnesses absent')
    _require(all(features['exclusive']), 'absorbed paint piece')
    _require(not np.array_equal(_independent_prints(a, decoded), out), 'independent-print shortcut survives')
    _require(not np.array_equal(_evaluate(a, decoded, mask=decoded['pieces'][0]['local_mask']), out),
             'single-piece shortcut survives')
    return {'input': a, 'output': out}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name+' out of range')
    return int(value)


def _registered_geometry(info):
    """Geometry only: neither colors nor source-piece presentation order."""
    mask = info['mask']
    r0, c0 = _origin(mask)
    return (_normal(mask), tuple(sorted((r-r0, c-c0) for r, c, s in info['pins'])))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(8)]
    # Neutral/key roles are fixed; every other literal color has a local role.
    common_bg = palette[0]
    puzzle, seen, train_geometry = {'train': [], 'test': []}, set(), set()
    teaching_source = teaching_target = None
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            testing = split == 'test'
            n = (int(rng.integers(3, 5)) if testing else 2 if index == 0 else 3 if index == 1
                 else int(rng.integers(2, 5)))
            bg = common_bg if rng.random() < .65 else int(rng.integers(8))
            colors = [int(c) for c in rng.permutation([c for c in range(8) if c != bg])]
            if not testing and index == 1:
                # Reuse a source but change its target: the labels are local
                # instructions, not a dictionary learned from demonstration 1.
                if bg == teaching_source:
                    bg = int(rng.choice([c for c in range(8) if c != teaching_source]))
                pool = [int(c) for c in rng.permutation([c for c in range(8) if c != bg])]
                target = _pick(rng, [c for c in pool if c not in (teaching_source, teaching_target)])
                other = [c for c in pool if c not in (teaching_source, target)]
                colors = [teaching_source] + other[:n-1] + [target] + other[n-1:]
            for attempt in range(48):
                try:
                    pair = _build_pair(rng, n, bg, colors, index, testing, attempt)
                    key = (pair['input'].shape, pair['input'].tobytes())
                    geometry = _registered_geometry(_parse(pair['input']))
                    if key not in seen and (not testing or geometry not in train_geometry):
                        seen.add(key)
                        if not testing:
                            train_geometry.add(geometry)
                            if index == 0:
                                teaching_source, teaching_target = colors[0], colors[n]
                        puzzle[split].append(pair)
                        break
                except AssertionError:
                    continue
            else:
                raise RuntimeError('bounded multi-pin construction exhausted')
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'split lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'counts')
    seen, train_features, training_maps = set(), [], {}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            a, out = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == out.shape, 'canvas dimensions changed')
            key = (a.shape, a.tobytes())
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            info = _parse(a)
            _require(np.array_equal(_evaluate(a, info), out), 'wrong output')
            f = _features(a, info)
            _require(f['matches'] >= 1 and f['complete'] >= 1 and f['decoys'] >= 1, 'missing witnesses')
            _require(all(f['exclusive']), 'each piece must add paint')
            _require(not np.array_equal(_independent_prints(a, info), out), 'B-only shortcut survives')
            _require(not np.array_equal(_evaluate(a, info, mask=info['pieces'][0]['local_mask']), out),
                     'assembly omission survives')
            if split == 'train':
                train_features.append(f)
                for piece in info['pieces']:
                    training_maps.setdefault(piece['source'], set()).add(piece['target'])
    _require(any(f['pieces'] == 2 for f in train_features), 'simple overlap demonstration missing')
    _require(any(f['pieces'] >= 3 and f['hole_pins'] for f in train_features), 'linked and hole demonstration missing')
    _require(any(len(targets) > 1 for targets in training_maps.values()), 'local color labels not demonstrated')


def _puzzle_digest(puzzle):
    h = sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _manual_regression():
    """Two hand-drawn pieces, a hole pin, complete/clipped prints and decoys."""
    a = np.zeros((14, 20), dtype=np.int8)
    # Both pieces show this same L key; their original canvas locations differ.
    key = {(0, 0), (1, 0), (1, 1)}
    first = {(-1, 2), (0, 1), (0, 2), (0, 3)}
    second = {(2, -1), (2, 0), (2, 1), (3, 1),
              (4, -1), (4, 0), (4, 1)}
    for points, delta, color in ((key, (1, 1), 9), (key, (8, 3), 9),
                                  (first, (1, 1), 8), (second, (8, 3), 8)):
        for p in _shift(points, delta):
            a[p] = color
    a[1, 6], a[1, 5] = 1, 3      # Pin at local (0,3), pointing left.
    a[11, 1], a[11, 2] = 2, 3    # Empty pin at local (3,0), pointing right.
    a[5, 13], a[8, 10] = 1, 2    # Complete occurrence, common offset (5,10).
    a[0, 19], a[3, 16] = 1, 2    # Top-clipped occurrence, offset (0,16).
    a[7, 18], a[13, 18] = 1, 2   # Unmatched pins must survive.
    a[13, 0], a[6, 7] = 3, 4     # Inactive target and unrelated color.
    expected = np.zeros_like(a)
    for p in ((4, 12), (5, 11), (5, 12), (5, 13),
              (7, 9), (7, 10), (7, 11), (8, 11),
              (9, 9), (9, 10), (9, 11),
              (0, 17), (0, 18), (0, 19),
              (2, 15), (2, 16), (2, 17), (3, 17),
              (4, 15), (4, 16), (4, 17), (13, 0)):
        expected[p] = 3
    expected[7, 18], expected[13, 18], expected[6, 7] = 1, 2, 4
    _require(np.array_equal(solve(a), expected), 'hand-specified print regression')
    _require(expected[8, 10] == 0, 'hole pin must be erased without filling the hole')
    _require(expected[5, 10] == 0, 'registration key must not be printed')
    info = _parse(a)
    _require(len(_matches(a, info)) == 2, 'hand-specified pin matching')
    # Breaking one key copy makes registration unsupported, rather than inviting
    # a largest-overlap or nearest-piece guess.
    broken = a.copy()
    broken[8, 3] = 0
    try:
        solve(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('unpaired key accepted')


def self_test():
    """A streaming 42-seed audit plus hand-specified semantic regressions."""
    _manual_regression()
    signatures, shapes, masks, key_shapes = set(), set(), set(), set()
    counts, backgrounds, directions = set(), set(), set()
    holes = clips = overlaps = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                info = _parse(a)
                f = _features(a, info)
                shapes.add(a.shape)
                masks.add(_normal(info['mask']))
                key_shapes.update(k for p in info['pieces'] for k, origin in p['keys'])
                counts.add(f['pieces'])
                backgrounds.add(info['background'])
                directions.update(f['directions'])
                holes += f['hole_pins'] > 0
                clips += f['clipped'] > 0
                footprints = [_inside(_shift(info['mask'], q), a.shape) for q in _matches(a, info)]
                overlaps += sum(map(len, footprints)) > len(set().union(*footprints))
                # Label colors are read locally; whole-palette permutations of
                # colors 0..7 preserve the answer and the two structural roles.
                if seed < 2:
                    mapping = np.array(list(reversed(range(8)))+[8, 9], dtype=np.int8)
                    _require(np.array_equal(solve(mapping[a]), mapping[pair['output']]), 'palette equivariance')
                    _require(np.array_equal(solve(a.T), pair['output'].T), 'transpose equivariance')
        if seed < 2:
            _require(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed)), 'determinism')
    _require(len(signatures) == 42 and len(shapes) >= 20 and len(masks) >= 150, 'structural diversity')
    _require(counts == {2, 3, 4} and len(backgrounds) >= 7 and len(key_shapes) >= 20, 'role/key variation')
    _require(directions == set(DIRECTIONS) and holes >= 42 and clips >= 10 and overlaps >= 1, 'geometry coverage')
    for nt, nq in ((2, 1), (10, 4), (7, 3)):
        validate_generated_puzzle(generate_puzzle(1234, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = generate_puzzle(876, 2, 1)
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong output accepted')
