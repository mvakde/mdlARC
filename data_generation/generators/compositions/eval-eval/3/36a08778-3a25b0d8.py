"""Color-gated chamber probes. Standard library and NumPy only.

The solver reads geometry, colors, source direction and correspondence from the
query itself. Construction labels are discarded before a pair is accepted.
"""
from itertools import product
import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    parts = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, part = [start], [start]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    part.append(q)
        parts.append(np.array(part, dtype=int))
    return parts


def _bbox(points):
    return tuple(map(int, points.min(0))) + tuple(map(int, points.max(0)+1))


def _dilate(mask):
    a = np.pad(mask, 1)
    return np.logical_or.reduce([a[r:r+mask.shape[0], c:c+mask.shape[1]]
                                 for r, c in product(range(3), repeat=2)])


def _rooms(grid, wall):
    a = np.pad(grid != wall, 1, constant_values=True)
    return [p-1 for p in _components(a)
            if np.all(p > 0) and np.all(p < np.array(a.shape)-1)]


def _relations(rooms):
    boxes = [_bbox(p) for p in rooms]
    return [[tuple(-1 if a[k+2] <= b[k] else 1 if b[k+2] <= a[k] else 0
                   for k in (0, 1)) for b in boxes] for a in boxes]


def _match(donor_rooms, target_rooms, colors):
    """Exact oriented projection graph matching; no area or scan-order key."""
    a, b = _relations(donor_rooms), _relations(target_rooms)
    n = len(a)
    answers = []

    def search(mapping, used):
        if len(mapping) == n:
            payload = [None]*n
            for i, j in mapping.items():
                payload[j] = colors[i]
            answers.append(tuple(payload))
            return
        choices = {i: [j for j in range(n) if j not in used and all(
            a[i][k] == b[j][v] for k, v in mapping.items())]
            for i in range(n) if i not in mapping}
        i = min(choices, key=lambda k: len(choices[k]))
        for j in choices[i]:
            mapping[i] = j
            search(mapping, used | {j})
            del mapping[i]

    search({}, set())
    _require(answers and len(set(answers)) == 1, 'ambiguous spatial color correspondence')
    return list(answers[0])


def _panel_boxes(grid, wall):
    candidates = []
    for p in _components(grid == wall):
        r, c, R, C = _bbox(p)
        if R-r < 5 or C-c < 5:
            continue
        crop = grid[r:R, c:C]
        if all(np.all(edge == wall) for edge in
               (crop[0], crop[-1], crop[:, 0], crop[:, -1])):
            candidates.append((r, c, R, C))
    return [a for a in candidates if not any(a != b and
            b[0] <= a[0] and b[1] <= a[1] and a[2] <= b[2] and a[3] <= b[3]
            for b in candidates)]


def _footprints(field, wall, rooms):
    owner = np.full(field.shape, -1, dtype=int)
    footprints = []
    for i, p in enumerate(rooms):
        interior = np.zeros(field.shape, dtype=bool)
        interior[tuple(p.T)] = True
        shell = _dilate(interior)
        _require(np.all(field[shell & ~interior] == wall), 'incomplete chamber shell')
        _require(np.all(owner[_dilate(shell)] == -1), 'chambers must have a gap')
        owner[shell] = i
        footprints.append(shell)
    _require(np.all(owner[field == wall] >= 0), 'unassigned interior wall')
    return owner, footprints


def _source(field, bg, wall, owner):
    candidates = []
    for rotation in range(4):
        a = np.rot90(field, rotation)
        o = np.rot90(owner, rotation)
        payload = (a != bg) & (a != wall)
        values = np.unique(a[payload])
        if len(values) != 1 or np.any(o[payload] >= 0):
            continue
        key = int(values[0])
        entries = list(map(int, np.flatnonzero(a[0] == key)))
        if not entries or any(b-a < 2 for a, b in zip(entries, entries[1:])):
            continue
        mask = np.zeros(a.shape, dtype=bool)
        lengths = []
        for c in entries:
            rows = np.flatnonzero(a[:, c] == key)
            if not 2 <= len(rows) < a.shape[0] or not np.array_equal(rows, np.arange(len(rows))):
                break
            mask[:len(rows), c] = True
            lengths.append(len(rows))
        else:
            if np.array_equal(mask, payload):
                candidates.append((rotation, key, entries, lengths))
    _require(len(candidates) == 1, 'ambiguous source direction')
    return candidates[0]


def _parse(input_grid):
    grid = np.asarray(input_grid)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), 'bad input dimensions')
    _require(np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9)), 'bad input colors')
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in
                 (grid[0], grid[-1], grid[:, 0], grid[:, -1])), 'expected background margin')
    possibilities = []
    for wall in map(int, np.unique(grid)):
        if wall == bg:
            continue
        boxes = _panel_boxes(grid, wall)
        if len(boxes) != 2:
            continue
        for donor_box, target_box in (boxes, boxes[::-1]):
            dr, dc, dR, dC = donor_box
            tr, tc, tR, tC = target_box
            if not (dR < tr or tR < dr or dC < tc or tC < dc):
                continue
            donor = grid[dr:dR, dc:dC]
            target = grid[tr:tR, tc:tC]
            field = target[1:-1, 1:-1]
            try:
                drooms, trooms = _rooms(donor, wall), _rooms(field, wall)
                _require(3 <= len(drooms) == len(trooms) <= 8, 'room count mismatch')
                colors = []
                for room in drooms:
                    values = np.unique(donor[tuple(room.T)])
                    _require(len(values) == 1 and values[0] not in (bg, wall), 'nonuniform donor room')
                    colors.append(int(values[0]))
                _require(len(set(colors)) >= 2, 'donor needs multiple colors')
                _require(sum(len(p) for p in drooms) == np.count_nonzero((donor != wall) & (donor != bg)),
                         'unassigned donor payload')
                _require(all(np.all(field[tuple(p.T)] == bg) for p in trooms), 'target rooms must be blank')
                owner, footprints = _footprints(field, wall, trooms)
                rotation, key, entries, lengths = _source(field, bg, wall, owner)
                mapped = _match(drooms, trooms, colors)
                _require(key in mapped and any(c != key for c in mapped), 'both permeability classes required')
                covered = np.zeros(grid.shape, dtype=bool)
                covered[dr:dR, dc:dC] = True
                covered[tr:tR, tc:tC] = True
                _require(np.all(grid[~covered] == bg), 'extraneous objects')
                possibilities.append(dict(bg=bg, wall=wall, key=key, donor=donor,
                    target=target, field=field, rooms=trooms, donor_rooms=drooms,
                    colors=mapped, owner=owner, footprints=footprints,
                    rotation=rotation, entries=entries, lengths=lengths,
                    donor_box=donor_box, target_box=target_box))
            except AssertionError:
                pass
    _require(len(possibilities) == 1, 'expected one visible donor/target interpretation')
    return possibilities[0]


def _simulate(info, shortcut=None):
    """Immutable chamber permeability; only forward contact activates rooms."""
    rot = info['rotation']
    a = np.rot90(info['field'], rot)
    original_owner = np.rot90(info['owner'], rot)
    owner = original_owner.copy()
    if shortcut == 'bounding_boxes':
        owner[:] = -1
        for i, mask in enumerate(info['footprints']):
            points = np.argwhere(np.rot90(mask, rot))
            r, c, R, C = _bbox(points)
            owner[r:R, c:C] = i
    bg, wall, key = info['bg'], info['wall'], info['key']
    colors = info['colors']
    if shortcut == 'scan_order':
        colors = [int(info['donor'][tuple(p.T)][0]) for p in info['donor_rooms']]
    out = a.copy()
    active = set(info['entries'])
    activated, events, passed, contact_layers = set(), [], set(), []
    h, w = a.shape
    for r in range(h):
        hits, following, contacts = {}, set(), set()
        for c in sorted(active):
            i = int(owner[r, c])
            if i >= 0:
                activated.add(i)
                contacts.add(i)
            transparent = i < 0 or (colors[i] == key and shortcut != 'all_opaque')
            if shortcut == 'straight':
                transparent = True
            if transparent:
                following.add(c)
                if i >= 0:
                    passed.add(i)
                if owner[r, c] < 0 and a[r, c] == bg:
                    out[r, c] = key
            else:
                left = right = c
                while left > 0 and owner[r, left-1] == i:
                    left -= 1
                while right+1 < w and owner[r, right+1] == i:
                    right += 1
                hits[(i, left, right)] = True
        for i, left, right in hits:
            events.append((r, i, left, right))
            if r:
                for c in range(max(0, left-1), min(w, right+2)):
                    # Crosspieces cannot activate or overwrite a chamber.
                    if original_owner[r-1, c] < 0 and a[r-1, c] == bg:
                        out[r-1, c] = key
            for c in (left-1, right+1):
                if 0 <= c < w:
                    following.add(c)
                    if owner[r, c] < 0 and a[r, c] == bg:
                        out[r, c] = key
        active = following
        contact_layers.append(contacts)
    out = np.rot90(out, -rot).copy()
    if shortcut == 'fill_all':
        activated = set(range(len(colors)))
    for i in activated:
        out[tuple(info['rooms'][i].T)] = colors[i]
    result = info['target'].copy()
    result[1:-1, 1:-1] = out
    stats = dict(activated=activated, passed=passed, events=events,
                 inactive=set(range(len(colors)))-activated,
                 key_activated=sum(colors[i] == key for i in activated),
                 opaque_activated=sum(colors[i] != key for i in activated),
                 clipped=sum(left == 0 or right == w-1 for _, _, left, right in events),
                 simultaneous=sum(len(contacts) > 1 for contacts in contact_layers))
    return result, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _simulate(_parse(input_grid))[0]


def _plan(rng, core=False, deep=False):
    if core:
        return np.array([[-1, 0, -1], [1, 2, 3]], dtype=int)
    rows, cols = ((4, 2) if deep else ((2, 3), (3, 2), (2, 4), (4, 2))[int(rng.integers(4))])
    p = np.arange(rows*cols).reshape(rows, cols)
    for _ in range(int(rng.integers(3))):
        if len(set(p.flat)-{-1}) <= 4:
            break
        r, c = int(rng.integers(rows)), int(rng.integers(cols))
        if np.count_nonzero(p == p[r, c]) == 1:
            p[r, c] = -1
    for _ in range(int(rng.integers(3))):
        edges = [(p[r, c], p[R, C]) for r, c in product(range(rows), range(cols))
                 for R, C in ((r+1, c), (r, c+1))
                 if R < rows and C < cols and min(p[r, c], p[R, C]) >= 0 and p[r, c] != p[R, C]]
        if not edges or len(set(p.flat)-{-1}) <= 4:
            break
        a, b = edges[int(rng.integers(len(edges)))]
        points = np.argwhere((p == a) | (p == b))
        r, c, R, C = _bbox(points)
        if len(points) == (R-r)*(C-c):
            p[p == b] = a
    for i, old in enumerate(sorted(set(p.flat)-{-1})):
        p[p == old] = i
    return p


def _notch(rng, mask, force=False):
    p = np.argwhere(mask)
    r, c, R, C = _bbox(p)
    if R-r < 2 or C-c < 2 or (not force and rng.random() < .55):
        return mask
    height = min(2, R-r-1)
    width = min(2, C-c-1)
    if force:
        mask[r:r+height, C-width:C] = False
    else:
        if rng.random() < .5:
            r = R-height
        if rng.random() < .5:
            c = C-width
        mask[r:r+height, c:c+width] = False
    return mask


def _draw_donor(rng, plan, bg, wall, colors):
    heights = rng.integers(1, 3, size=plan.shape[0])
    widths = rng.integers(1, 4, size=plan.shape[1])
    ys = np.r_[0, np.cumsum(heights+1)]
    xs = np.r_[0, np.cumsum(widths+1)]
    grid = np.full((ys[-1]+1, xs[-1]+1), wall, dtype=np.int8)
    for i, color in enumerate(colors):
        r, c, R, C = _bbox(np.argwhere(plan == i))
        mask = np.zeros(grid.shape, dtype=bool)
        mask[ys[r]+1:ys[R], xs[c]+1:xs[C]] = True
        mask = _notch(rng, mask)
        grid[mask] = color
    return grid


def _draw_target(rng, plan, bg, wall, key, witness):
    rows, cols = plan.shape
    heights = rng.integers(1, 5, size=rows)
    widths = rng.integers(1, 5, size=cols)
    if witness == 'pass':
        widths[1] = int(rng.integers(4, 7))
        heights[1] = int(rng.integers(4, 6))
        heights[0] = int(rng.integers(1, 3))
    ys = np.r_[4, 4+np.cumsum(heights+3)]
    xs = np.r_[1, 1+np.cumsum(widths+3)]
    h, w = int(ys[-2]+heights[-1]+1), int(xs[-2]+widths[-1]+1)
    masks, owner = [], np.full((h, w), -1, dtype=int)
    field = np.full((h, w), bg, dtype=np.int8)
    for i in range(len(set(plan.flat)-{-1})):
        r, c, R, C = _bbox(np.argwhere(plan == i))
        top, left = int(ys[r]), int(xs[c])
        bottom, right = int(ys[R-1]+heights[R-1]), int(xs[C-1]+widths[C-1])
        if witness and i == 0:
            left -= 1
            right += 1
            if witness == 'pass':
                right -= 2
        elif witness == 'pass' and i in (1, 3):
            top += 1
            bottom -= 1
        elif not witness:
            dx = int(rng.integers(-1, 2))
            left, right = max(1, left+dx), min(w-1, right+dx)
            dy = int(rng.integers(-1, 2))
            top, bottom = max(3, top+dy), min(h-1, bottom+dy)
            if right-left >= 3 and rng.random() < .4:
                left += int(rng.integers(2))
                right -= int(rng.integers(2))
        mask = np.zeros(field.shape, dtype=bool)
        _require(top < bottom and left < right, 'empty target chamber')
        mask[top:bottom, left:right] = True
        if witness == 'pass' and i == 2:
            mask = _notch(rng, mask, force=True)
        elif not witness:
            mask = _notch(rng, mask)
        shell = _dilate(mask)
        _require(not np.any(owner[_dilate(shell)] >= 0), 'target footprints touch')
        owner[shell] = i
        field[shell & ~mask] = wall
        masks.append(mask)
    if witness:
        c = int(xs[1]+widths[1]-2) if witness == 'pass' else int(xs[1]+widths[1]//2)
        entries = [c]
    else:
        candidates = []
        for c in range(w):
            hit = np.flatnonzero(owner[:, c] >= 0)
            if len(hit):
                candidates.append(c)
        rng.shuffle(candidates)
        entries = []
        desired = int(rng.integers(1, 4))
        for c in candidates:
            if all(abs(c-v) >= 2 for v in entries):
                entries.append(c)
            if len(entries) == desired:
                break
    for c in entries:
        field[:int(rng.integers(2, 4)), c] = key
    panel = np.full((h+2, w+2), wall, dtype=np.int8)
    panel[1:-1, 1:-1] = field
    return panel, masks


def _embed(rng, donor, target, bg):
    dh, dw = donor.shape
    th, tw = target.shape
    choices = []
    if dw+tw+3 <= 30 and max(dh, th)+2 <= 30:
        choices.append('side')
    if dh+th+3 <= 30 and max(dw, tw)+2 <= 30:
        choices.append('stack')
    _require(choices, 'panels do not fit')
    mode = choices[int(rng.integers(len(choices)))]
    reverse = rng.random() < .5
    first, second = (target, donor) if reverse else (donor, target)
    if mode == 'side':
        h = max(dh, th)+2
        w = dw+tw+3
        positions = [(int(rng.integers(1, h-first.shape[0])), 1),
                     (int(rng.integers(1, h-second.shape[0])), first.shape[1]+2)]
    else:
        h = dh+th+3
        w = max(dw, tw)+2
        positions = [(1, int(rng.integers(1, w-first.shape[1]))),
                     (first.shape[0]+2, int(rng.integers(1, w-second.shape[1])))]
    grid = np.full((h, w), bg, dtype=np.int8)
    for panel, (r, c) in zip((first, second), positions):
        grid[r:r+panel.shape[0], c:c+panel.shape[1]] = panel
    return grid


def _fallback_pair(rng, witness, rotation, seen):
    """Finite, fitting constructions, with 720 distinct role assignments.

    These obey the same input rule. A fallback is not a new solving convention.
    """
    is_pass = witness == 'pass' or (witness is None and rng.random() < .5)
    bg, wall, key = 0, 1, 2
    colors = [key if is_pass else 3, key, 4, 3]
    donor = np.full((6, 9), wall, dtype=np.int8)
    donor[1, 4] = colors[0]
    donor[3:5, 1:3] = colors[1]
    donor[3:5, 4] = colors[2]
    donor[3:5, 6:8] = colors[3]
    if is_pass:
        field = np.full((14, 14), bg, dtype=np.int8)
        boxes = [(4, 4, 6, 8), (10, 1, 12, 2),
                 (9, 5, 13, 9), (10, 12, 12, 13)]
        source = 7
    else:
        field = np.full((11, 11), bg, dtype=np.int8)
        boxes = [(4, 4, 6, 7), (9, 1, 10, 2),
                 (9, 5, 10, 6), (9, 9, 10, 10)]
        source = 5
    for i, (r, c, R, C) in enumerate(boxes):
        mask = np.zeros(field.shape, dtype=bool)
        mask[r:R, c:C] = True
        if is_pass and i == 2:
            mask[r:r+2, C-2:C] = False
        field[_dilate(mask) & ~mask] = wall
    field[:2, source] = key
    target = np.full(tuple(n+2 for n in field.shape), wall, dtype=np.int8)
    target[1:-1, 1:-1] = field
    inp = np.rot90(_embed(rng, donor, target, bg), rotation).copy()
    ordering = list(map(int, rng.permutation(10)))
    for a, b, c in product(ordering, repeat=3):
        if len({a, b, c}) != 3:
            continue
        palette = np.array([a, b, c]+[v for v in ordering if v not in (a, b, c)], dtype=np.int8)
        candidate = palette[inp]
        if _key(candidate) not in seen:
            info = _parse(candidate)
            out, stats = _simulate(info)
            _require(stats['key_activated'] and stats['opaque_activated'], 'fallback lacks interaction')
            if is_pass:
                _require(not np.array_equal(_simulate(info, 'scan_order')[0], out), 'fallback scan witness')
                _require(not np.array_equal(_simulate(info, 'bounding_boxes')[0], out), 'fallback slice witness')
            return {'input': candidate, 'output': out}
    raise AssertionError('more unique inputs excluded than legal puzzle size')


def _make_pair(rng, witness, rotation, deep, seen, attempts=180):
    # Finite rejection checks visible correspondence and teaching evidence.
    for attempt in range(attempts):
        plan = _plan(rng, core=bool(witness), deep=deep)
        n = len(set(plan.flat)-{-1})
        palette = list(map(int, rng.permutation(10)))
        bg, wall, key = palette[:3]
        richness = int(rng.integers(2, min(n, 8)+1))
        available = palette[2:2+richness]
        colors = available + [int(rng.choice(available)) for _ in range(n-richness)]
        rng.shuffle(colors)
        if witness:
            colors[0] = key if witness == 'pass' else palette[3]
            colors[2] = palette[4]
            colors[1], colors[3] = key, palette[3]
        try:
            donor = _draw_donor(rng, plan, bg, wall, colors)
            target, masks = _draw_target(rng, plan, bg, wall, key, witness)
            inp = np.ascontiguousarray(np.rot90(_embed(rng, donor, target, bg), rotation))
            if _key(inp) in seen:
                continue
            info = _parse(inp)
            out, stats = _simulate(info)
            # A generation-label answer checks that spatial matching really
            # assigns the intended colors, rather than just trusting itself.
            canonical = np.rot90(out, -rotation)
            canonical_owner = np.rot90(info['owner'], -rotation)
            for i, mask in enumerate(masks):
                identities = np.unique(canonical_owner[mask])
                _require(len(identities) == 1 and identities[0] >= 0 and
                         info['colors'][int(identities[0])] == colors[i], 'prospective payload mismatch')
                values = np.unique(canonical[1:-1, 1:-1][mask])
                _require(len(values) == 1 and values[0] in (bg, colors[i]), 'wrong matched payload')
            _require(stats['key_activated'] and stats['opaque_activated'], 'missing interacting contacts')
            _require(not np.array_equal(_simulate(info, 'all_opaque')[0], out), 'permeability redundant')
            _require(not np.array_equal(_simulate(info, 'straight')[0], out), 'branching redundant')
            if witness == 'split':
                _require(len(stats['inactive']) >= 1 and stats['simultaneous'], 'split witness missing side hits')
            elif witness == 'pass':
                _require(len(stats['inactive']) >= 2, 'pass witness must leave sides blank')
                _require(not np.array_equal(_simulate(info, 'bounding_boxes')[0], out), 'local slice witness redundant')
                _require(not np.array_equal(_simulate(info, 'scan_order')[0], out), 'scan-order witness redundant')
            return {'input': inp, 'output': out}
        except AssertionError:
            continue
    return _fallback_pair(rng, witness, rotation, seen)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    base, curriculum = int(rng.integers(4)), int(rng.integers(3))
    held = (base+1) % 4
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            witness = ('split', 'pass')[i] if split == 'train' and i < 2 else None
            rotation = (base+2*i) % 4 if split == 'train' else held
            if curriculum != 0 and i >= 2:
                rotation = int(rng.integers(4))
            pair = _make_pair(rng, witness, rotation,
                              deep=split == 'test' and curriculum == 1, seen=seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'bad puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'illegal counts')
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'bad pair schema')
            inp, out = (np.asarray(pair[k]) for k in ('input', 'output'))
            for a in (inp, out):
                _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'bad grid dimensions')
                _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), 'bad grid colors')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            info = _parse(inp)
            expected, stats = _simulate(info)
            _require(np.array_equal(out, expected), 'incorrect output')
            target = info['target']
            _require(out.shape == target.shape, 'target crop dimensions')
            _require(np.array_equal(out[target == info['wall']], target[target == info['wall']]), 'walls changed')
            field = out[1:-1, 1:-1]
            for i, room in enumerate(info['rooms']):
                color = info['colors'][i] if i in stats['activated'] else info['bg']
                _require(np.all(field[tuple(room.T)] == color), 'incorrect uniform activation fill')
            outside = info['owner'] < 0
            old = info['field']
            _require(np.all((field[outside] == old[outside]) | (field[outside] == info['key'])), 'invalid probe trace')
            _require(np.all(field[old == info['key']] == info['key']), 'stub changed')
            _require(stats['key_activated'] and stats['opaque_activated'], 'one mechanism unused')
            for shortcut in ('straight', 'all_opaque'):
                _require(not np.array_equal(_simulate(info, shortcut)[0], out), f'{shortcut} has no effect')
            if split == 'train':
                training.append((info, stats, out))
    _require(any(s['inactive'] for _, s, _ in training), 'no selective transfer witness')
    _require(any(s['simultaneous'] for _, s, _ in training), 'no simultaneous branching witness')
    _require(any(not np.array_equal(_simulate(info, 'bounding_boxes')[0], out)
                 for info, _, out in training), 'no local slice witness')
    _require(any(not np.array_equal(_simulate(info, 'fill_all')[0], out)
                 for info, _, out in training), 'fill-all shortcut undistinguished')
    _require(any(not np.array_equal(_simulate(info, 'scan_order')[0], out)
                 for info, _, out in training), 'scan-order shortcut undistinguished')


def self_test():
    structures, shapes, room_counts, role_sets = set(), set(), set(), [set() for _ in range(3)]
    directions, entry_counts, areas, layouts, signatures = set(), set(), set(), set(), set()
    nonrectangular = clipped = simultaneous = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(tuple(_key(pair['input']) for s in p for pair in p[s]))
        for split in ('train', 'test'):
            for pair in p[split]:
                info = _parse(pair['input'])
                out, stats = _simulate(info)
                _require(np.array_equal(solve(pair['input']), out), 'input-only solve mismatch')
                roles = {}
                structures.add((pair['input'].shape, tuple(roles.setdefault(int(v), len(roles))
                                                          for v in pair['input'].flat)))
                shapes.add((pair['input'].shape, out.shape))
                room_counts.add(len(info['rooms']))
                directions.add(info['rotation'])
                entry_counts.add(len(info['entries']))
                for j, role in enumerate(('bg', 'wall', 'key')):
                    role_sets[j].add(info[role])
                for room in info['rooms']:
                    r, c, R, C = _bbox(room)
                    nonrectangular += len(room) < (R-r)*(C-c)
                    areas.add(len(room))
                dr, dc, dR, dC = info['donor_box']
                tr, tc, tR, tC = info['target_box']
                layouts.add('stack' if dR < tr or tR < dr else 'side')
                clipped += stats['clipped'] > 0
                simultaneous += stats['simultaneous'] > 0
        again = generate_puzzle(seed)
        _require(all(_key(a[k]) == _key(b[k]) for s in p for a, b in zip(p[s], again[s])
                     for k in ('input', 'output')), 'nondeterminism')
    _require(len(signatures) == 42 and len(structures) >= 170 and len(shapes) >= 80, 'geometry diversity')
    _require(len(room_counts) >= 4 and len(areas) >= 15, 'chamber diversity')
    _require(directions == set(range(4)) and len(entry_counts) >= 2, 'probe variation')
    _require(all(len(s) == 10 for s in role_sets), 'palette roles collapsed')
    _require(layouts == {'stack', 'side'} and nonrectangular >= 42 and clipped and simultaneous, 'family feature variation')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(7201, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'explicit counts ignored')
    for rotation in range(4):
        rng = np.random.default_rng(3900+rotation)
        seen = set()
        p = {'train': [], 'test': []}
        for split, witness in (('train', 'split'), ('train', 'pass'), ('test', None)):
            pair = _make_pair(rng, witness, rotation, False, seen, attempts=0)
            seen.add(_key(pair['input']))
            p[split].append(pair)
        validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    p = generate_puzzle(21, 2, 1)
    p['test'][0]['output'] = p['test'][0]['output'].copy()
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corruption was not rejected')
    return dict(seeds=42, structures=len(structures), shapes=len(shapes), room_counts=sorted(room_counts),
                nonrectangular_rooms=nonrectangular, clipped_pairs=clipped,
                simultaneous_pairs=simultaneous, entry_counts=sorted(entry_counts))
