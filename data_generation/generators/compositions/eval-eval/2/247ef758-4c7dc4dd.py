"""Elastic wiring atlas: local graph analogies embedded in colored coordinates.

Only NumPy and the Python standard library are used. Generation constructs labels
from logical records; solve parses visible pixels, with no generation state.
"""
from __future__ import annotations

import json
from numbers import Integral

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _dot_mask(n, m):
    a = np.zeros((2*n-1, 2*m-1), dtype=bool)
    a[::2, ::2] = True
    return a


def _check_network(mask):
    h, w = mask.shape
    _require(h in (3, 5, 7) and w in (3, 5, 7), "invalid prototype scale")
    _require(bool(mask[::2, ::2].all()), "a completed network must preserve every node")
    _require(bool(np.any(mask & ~_dot_mask((h+1)//2, (w+1)//2))), "network needs an edge")
    for r in range(1, h, 2):
        for c in range(1, w, 2):
            if mask[r, c]:
                _require(bool(mask[r-1, c] and mask[r+1, c]
                              and mask[r, c-1] and mask[r, c+1]), "a filled face needs four boundary edges")


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    groups = []
    while todo:
        stack = [todo.pop()]
        found = set(stack)
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in todo:
                    todo.remove(p)
                    found.add(p)
                    stack.append(p)
        groups.append(found)
    return groups


def _frame_components(a, color):
    return _components(a == color)


def _parse_canonical(g, transposed):
    """Return candidates for an atlas-above-field view of the displayed grid."""
    h, w = g.shape
    if h < 11 or w < 7:
        return []
    scenes = []
    for d in range(5, h-5):
        divider = int(g[d, 0])
        if not np.all(g[d] == divider):
            continue
        interior = g[d+2:-1, 1:-1]
        if not interior.size or not np.all(interior == interior[0, 0]):
            continue
        bg, frame = int(interior[0, 0]), int(g[d+1, 0])
        if len({bg, frame, divider}) != 3:
            continue
        if any(int(g[r, c]) != frame for r, c in ((d+1, w-1), (h-1, 0), (h-1, w-1))):
            continue
        if not np.array_equal(g[d+1, 1:-1], g[-1, 1:-1]):
            continue
        if not np.array_equal(g[d+2:-1, 0], g[d+2:-1, -1]):
            continue
        # Divider color is structural and occurs on this line only.
        if np.count_nonzero(g == divider) != w:
            continue
        try:
            band = g[:d]
            parts = _frame_components(band, frame)
            _require(1 <= len(parts) <= 2, "need one or two disjoint cards")
            cards, occupied = [], np.zeros(band.shape, dtype=bool)
            for cells in parts:
                rr, cc = zip(*cells)
                r, c = int(min(rr)), int(min(cc))
                bh, bw = int(max(rr)-r+1), int(max(cc)-c+1)
                _require(bh-2 in (3, 5, 7) and bw in (9, 13, 17), "invalid card dimensions")
                mid = c+bw//2
                expected_frame = {(y, x) for y in range(r, r+bh) for x in range(c, c+bw)
                                  if y in (r, r+bh-1) or x in (c, mid, c+bw-1)}
                _require(cells == expected_frame, "card frame or shared wall is incomplete")
                left = band[r+1:r+bh-1, c+1:mid]
                right = band[r+1:r+bh-1, mid+1:c+bw-1]
                colors = set(map(int, np.unique(np.stack((left, right))))) - {bg}
                _require(len(colors) == 1 and frame not in colors and divider not in colors,
                         "each card uses one network color")
                color = next(iter(colors))
                n, m = (bh-1)//2, (bw-1)//4
                dots = _dot_mask(n, m)
                states = [left == color, right == color]
                source_slots = [i for i, mask in enumerate(states) if np.array_equal(mask, dots)]
                _require(len(source_slots) == 1, "card must have one exact sparse lattice")
                source_slot = source_slots[0]
                network = states[1-source_slot].copy()
                _check_network(network)
                _require(not occupied[r:r+bh, c:c+bw].any(), "overlapping cards")
                occupied[r:r+bh, c:c+bw] = True
                # Priority is ordinary reading order in the DISPLAYED image.
                priority = (c, r) if transposed else (r, c)
                cards.append(dict(color=color, n=n, m=m, network=network,
                                  box=(r, c, bh, bw), source_slot=source_slot, priority=priority))
            _require(bool(np.all(band[~occupied] == bg)), "atlas exterior must be background")
            _require(len({x['color'] for x in cards}) == len(cards), "card colors must be distinct")
            if len(cards) == 2:
                r, c, bh, bw = cards[0]['box']
                y, x, ah, aw = cards[1]['box']
                _require(r+bh < y or y+ah < r or c+bw < x or x+aw < c,
                         "cards need a background gap")
            rows, cols = {}, {}
            for i, value in enumerate(g[d+2:-1, 0]):
                color = int(value)
                if color != frame:
                    rows.setdefault(color, []).append(i)
            for i, value in enumerate(g[d+1, 1:-1]):
                color = int(value)
                if color != frame:
                    cols.setdefault(color, []).append(i)
            _require(set(rows) | set(cols) <= {x['color'] for x in cards}, "unknown coordinate color")
            active = 0
            for card in cards:
                rs, cs = rows.get(card['color'], []), cols.get(card['color'], [])
                _require(len(rs) in (0, card['n']) and len(cs) in (0, card['m']),
                         "coordinate cardinality must match the sparse lattice")
                _require(all(y-x >= 2 for x, y in zip(rs, rs[1:]))
                         and all(y-x >= 2 for x, y in zip(cs, cs[1:])), "same-color gaps must be at least two")
                card['rows'], card['cols'] = rs, cs
                active += bool(rs and cs)
            _require(active > 0, "at least one card must have both axes")
            cards.sort(key=lambda x: x['priority'])
            scenes.append(dict(grid=g, d=d, bg=bg, frame=frame, divider=divider,
                               cards=cards, shape=interior.shape, transposed=transposed))
        except ValueError:
            continue
    return scenes


def _parse(input_grid):
    original = _grid(input_grid)
    candidates = _parse_canonical(original, False) + _parse_canonical(original.T, True)
    _require(len(candidates) == 1, "input must have a unique atlas and field")
    return candidates[0]


def _index_images(coordinates):
    images = []
    for i, v in enumerate(coordinates):
        if i:
            images.append(range(coordinates[i-1]+1, v))
        images.append((v,))
    return images


def _footprint(card, shape, nodes_only=False, uniform=False):
    mask = np.zeros(shape, dtype=bool)
    rs, cs = card['rows'], card['cols']
    if not rs or not cs:
        return mask
    if uniform:
        rs = [rs[0]+2*i for i in range(len(rs))]
        cs = [cs[0]+2*i for i in range(len(cs))]
    rows, cols = _index_images(rs), _index_images(cs)
    network = _dot_mask(card['n'], card['m']) if nodes_only else card['network']
    for r, c in np.argwhere(network):
        for y in rows[int(r)]:
            for x in cols[int(c)]:
                mask[y, x] = True
    return mask


def _render_scene(scene, nodes_only=False, uniform=False, order='source'):
    cards = list(scene['cards'])
    if order == 'destination':
        cards.sort(key=lambda x: (min(x['cols'], default=99), min(x['rows'], default=99))
                   if scene['transposed'] else (min(x['rows'], default=99), min(x['cols'], default=99)))
    elif order == 'reverse':
        cards.reverse()
    elif order == 'size':
        cards.sort(key=lambda x: x['network'].size)
    out = np.full(scene['shape'], scene['bg'], dtype=np.int8)
    claimed = np.zeros(scene['shape'], dtype=bool)
    for card in cards:
        mask = _footprint(card, scene['shape'], nodes_only, uniform)
        out[mask & ~claimed] = card['color']
        claimed |= mask
    return out.T.copy() if scene['transposed'] else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct only the visible input's missing coordinate field interior."""
    return _render_scene(_parse(input_grid))


def _network(rng, n, m):
    a = _dot_mask(n, m)
    mode = int(rng.integers(5))
    h, w = a.shape
    if mode == 0:
        a[::2, 1::2] = True  # horizontal tracks, optionally linked
        a[1::2, ::2] = rng.random((n-1, m)) < .25
    elif mode == 1:
        a[1::2, ::2] = True
        a[::2, 1::2] = rng.random((n, m-1)) < .25
    elif mode == 2:
        a[::2, 1::2] = True
        a[1::2, ::2] = True
    elif mode == 3:
        # A walk through node adjacency can leave isolated nodes elsewhere.
        r, c = int(rng.integers(n)), int(rng.integers(m))
        for _ in range(int(rng.integers(2, 2*n*m))):
            neighbors = [(r+dr, c+dc) for dr, dc in ((1,0), (-1,0), (0,1), (0,-1))
                         if 0 <= r+dr < n and 0 <= c+dc < m]
            y, x = neighbors[int(rng.integers(len(neighbors)))]
            a[r+y, c+x] = True
            r, c = y, x
    else:
        p = float(rng.uniform(.15, .85))
        a[::2, 1::2] = rng.random((n, m-1)) < p
        a[1::2, ::2] = rng.random((n-1, m)) < p
    if np.array_equal(a, _dot_mask(n, m)):
        a[0, 1] = True
    for r in range(1, h, 2):
        for c in range(1, w, 2):
            if all(a[y, x] for y, x in ((r-1,c), (r+1,c), (r,c-1), (r,c+1))):
                a[r, c] = rng.random() < float(rng.choice([0., .4, 1.]))
    return a


def _outline(g, r, c, h, w, color):
    g[r, c:c+w] = g[r+h-1, c:c+w] = color
    g[r:r+h, c] = g[r:r+h, c+w-1] = color


def _construction_output(cards, shape, bg, transpose):
    """Ground truth from construction records: repeats, then reverse painting.

    This does not call the visible-input parser, footprint or solver.
    """
    out = np.full(shape, bg, dtype=np.int8)
    ordered = sorted(cards, key=lambda x: (x['box'][1], x['box'][0]) if transpose else x['box'][:2])
    for card in reversed(ordered):
        rs, cs = card['rows'], card['cols']
        if not rs or not cs:
            continue
        row_widths = [1]
        col_widths = [1]
        for lo, hi in zip(rs, rs[1:]):
            row_widths.extend((hi-lo-1, 1))
        for lo, hi in zip(cs, cs[1:]):
            col_widths.extend((hi-lo-1, 1))
        patch = np.repeat(np.repeat(card['network'], row_widths, axis=0), col_widths, axis=1)
        view = out[rs[0]:rs[-1]+1, cs[0]:cs[-1]+1]
        view[patch] = card['color']
    return out.T.copy() if transpose else out


def _assemble(cards, band_h, width, field_h, palette, transpose):
    bg, frame, divider = map(int, palette[:3])
    g = np.full((band_h+field_h+3, width), bg, dtype=np.int8)
    g[band_h] = divider
    _outline(g, band_h+1, 0, field_h+2, width, frame)
    for card in cards:
        r, c, bh, bw = card['box']
        color = card['color']
        _outline(g, r, c, bh, bw, frame)
        mid = c+bw//2
        g[r:r+bh, mid] = frame
        n, m = card['n'], card['m']
        before = np.where(_dot_mask(n, m), color, bg)
        after = np.where(card['network'], color, bg)
        for index, panel in enumerate((before, after) if card['source_slot'] == 0 else (after, before)):
            start = c+1 if index == 0 else mid+1
            g[r+1:r+bh-1, start:start+panel.shape[1]] = panel
        for y in card['rows']:
            g[band_h+2+y, 0] = g[band_h+2+y, -1] = color
        for x in card['cols']:
            g[band_h+1, 1+x] = g[-1, 1+x] = color
    out = _construction_output(cards, (field_h, width-2), bg, transpose)
    return {'input': g.T.copy() if transpose else g, 'output': out}


def _diagnostic_twins(rng, preference):
    palette = rng.permutation(10)
    # Full horizontal/vertical incidence, two hollow faces in the first card.
    a = np.ones((3, 5), dtype=bool)
    a[1, 1::2] = False
    b = np.ones((3, 3), dtype=bool)
    margin = int(rng.integers(2))
    gap = int(rng.integers(1, 3))
    tops = [int(rng.integers(2))]*2
    lefts = [margin, margin+13+gap]
    cards = [dict(n=2, m=3, network=a, color=int(palette[3]),
                  box=(tops[0], lefts[0], 5, 13), source_slot=int(rng.integers(2)),
                  rows=[5, 11], cols=[5, 10, 17]),
             dict(n=2, m=2, network=b, color=int(palette[4]),
                  box=(tops[1], lefts[1], 5, 9), source_slot=0,
                  rows=[2, 8], cols=[2, 8])]
    cards[1]['source_slot'] = 1-cards[0]['source_slot']
    band_h = tops[0]+5+int(rng.integers(2))
    width = max(lefts[1]+9+int(rng.integers(2)), 22)
    field_h = int(rng.integers(13, min(19, 28-band_h)))
    transpose = preference
    first = _assemble(cards, band_h, width, field_h, palette, transpose)
    twins = [{**x, 'network':x['network'].copy()} for x in cards]
    # Fill one or both of the first card's chambers, hollow the second card.
    twins[0]['network'][1, 1] = True
    if rng.integers(2):
        twins[0]['network'][1, 3] = True
    twins[1]['network'][1, 1] = False
    second = _assemble(twins, band_h, width, field_h, palette, transpose)
    return first, second


def _axis_coordinates(rng, cards, axis, extent):
    """Assign distinct axis positions, fitting arbitrary color multiplicities."""
    counts = [x['n' if axis == 0 else 'm'] for x in cards]
    size = sum(counts)
    # Each color's consecutive coordinates have >=2 spacing. We use global
    # gaps >=2 here; the solver also accepts adjacent marks of different colors.
    available = extent-1-2*(size-1)
    _require(available >= 0, "axis too short")
    slack = int(rng.integers(available+1))
    increments = rng.multinomial(slack, [1/(size+1)]*(size+1))
    positions = [int(increments[0])]
    for i in range(size-1):
        positions.append(positions[-1]+2+int(increments[i+1]))
    owners = [i for i, count in enumerate(counts) for _ in range(count)]
    rng.shuffle(owners)
    result = [[] for _ in cards]
    for position, owner in zip(positions, owners):
        result[owner].append(position)
    return result


def _random_pair(rng, preference, large=False, inactive=False, allow_inactive=False):
    count = 2 if inactive else int(rng.choice([1, 2], p=[.45, .55]))
    if large:
        count = 1
    dims = []
    for _ in range(count):
        if large:
            dims.append((4, int(rng.choice([3, 4]))))
        else:
            limit = 5 if count == 1 else 4
            dims.append((int(rng.integers(2, limit)), int(rng.integers(2, limit))))
    palette = rng.permutation(10)
    cards = []
    widths = [4*m+1 for _, m in dims]
    gap = int(rng.integers(1, 4)) if count == 2 else 0
    min_width = sum(widths)+gap
    # Canvas dimensions obey the 30-cell budget; all fitting cards are kept.
    width = int(rng.integers(max(min_width, 2*sum(m for _, m in dims)+3, 14), 31))
    left = int(rng.integers(width-min_width+1))
    for i, (n, m) in enumerate(dims):
        bh, bw = 2*n+1, 4*m+1
        top = int(rng.integers(3))
        cards.append(dict(n=n, m=m, network=_network(rng, n, m), color=int(palette[3+i]),
                          box=(top, left, bh, bw), source_slot=int(rng.integers(2))))
        left += bw+gap
    min_band = max(x['box'][0]+x['box'][2] for x in cards)
    # Reserve room for each color's entire row axis even if it becomes inactive.
    minimum_field = max(2*sum(n for n, _ in dims)-1, 8)
    max_band = 27-minimum_field
    band_h = min_band+int(rng.integers(min(3, max_band-min_band)+1))
    field_h = int(rng.integers(minimum_field, 28-band_h))
    rs = _axis_coordinates(rng, cards, 0, field_h)
    cs = _axis_coordinates(rng, cards, 1, width-2)
    for i, card in enumerate(cards):
        card['rows'], card['cols'] = rs[i], cs[i]
    if inactive or (allow_inactive and count == 2 and rng.random() < .28):
        i = int(rng.integers(2))
        state = int(rng.integers(3))
        if state != 1:
            cards[i]['rows'] = []
        if state != 0:
            cards[i]['cols'] = []
    transpose = preference if rng.random() < .55 else bool(rng.integers(2))
    return _assemble(cards, band_h, width, field_h, palette, transpose)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a = _count(num_train, 'num_train', 2, 10)
    b = _count(num_test, 'num_test', 1, 4)
    if a <= b:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    preference, extrapolate = bool(rng.integers(2)), bool(rng.integers(2))
    first, twin = _diagnostic_twins(rng, preference)
    puzzle = {'train': [first, twin], 'test': []}
    seen = {_key(first['input']), _key(twin['input'])}
    for split, count in (('train', a-2), ('test', b)):
        for i in range(count):
            for _ in range(50):
                pair = _random_pair(rng, preference,
                                    large=split == 'test' and extrapolate,
                                    inactive=split == 'train' and i == 0,
                                    allow_inactive=a >= 3)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded input uniqueness search exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(scene):
    masks = [_footprint(x, scene['shape']) for x in scene['cards']]
    overlap = int(np.count_nonzero(masks[0] & masks[1])) if len(masks) == 2 else 0
    correct = _render_scene(scene)
    graphs, gaps = [], []
    for card in scene['cards']:
        mask = card['network']
        degrees = []
        for r in range(0, mask.shape[0], 2):
            for c in range(0, mask.shape[1], 2):
                degrees.append(sum(bool(0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x])
                                   for y, x in ((r-1,c), (r+1,c), (r,c-1), (r,c+1))))
        graphs.append(dict(nodes=(card['n'], card['m']), mask=_key(mask.astype(np.int8)),
                           edges=int(mask[::2,1::2].sum()+mask[1::2,::2].sum()),
                           faces=int(mask[1::2,1::2].sum()),
                           hollow=sum(not mask[r,c] and all(mask[y,x] for y,x in
                                      ((r-1,c), (r+1,c), (r,c-1), (r,c+1)))
                                      for r in range(1,mask.shape[0],2) for c in range(1,mask.shape[1],2)),
                           isolated=degrees.count(0), branches=sum(d >= 3 for d in degrees),
                           components=len(_components(mask)), source_slot=card['source_slot']))
        gaps.extend(int(y-x) for coordinates in (card['rows'], card['cols'])
                    for x, y in zip(coordinates, coordinates[1:]))
    return dict(cards=len(scene['cards']), active=sum(bool(x['rows'] and x['cols']) for x in scene['cards']),
                inactive=any(not x['rows'] or not x['cols'] for x in scene['cards']),
                shape=scene['shape'], transpose=scene['transposed'],
                roles=(scene['bg'], scene['frame'], scene['divider']), graphs=graphs, gaps=gaps,
                overlap=overlap, omit_a=not np.array_equal(correct, _render_scene(scene, uniform=True)),
                omit_b=not np.array_equal(correct, _render_scene(scene, nodes_only=True)),
                destination_diff=not np.array_equal(correct, _render_scene(scene, order='destination')),
                size_diff=not np.array_equal(correct, _render_scene(scene, order='size')),
                reverse_diff=not np.array_equal(correct, _render_scene(scene, order='reverse')))


def validate_generated_puzzle(puzzle) -> None:
    """Reject malformed geometry, wrong reconstruction and weak demonstrations."""
    try:
        _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
        _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'pairs must be lists')
        a, b = len(puzzle['train']), len(puzzle['test'])
        _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'invalid puzzle counts')
        seen, features = set(), {'train': [], 'test': []}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
                inp, out = _grid(pair['input']), _grid(pair['output'])
                _require(_key(inp) not in seen, 'duplicate inputs')
                seen.add(_key(inp))
                scene = _parse(inp)
                shape = scene['shape'][::-1] if scene['transposed'] else scene['shape']
                _require(out.shape == shape, 'return the entire border-free field interior')
                _require(np.array_equal(out, _render_scene(scene)), 'incorrect elastic embedding or priority')
                # Audit ownership directly: every nonbackground output is the
                # earliest card whose stretched footprint contains this cell.
                canonical_out = out.T if scene['transposed'] else out
                masks = [_footprint(card, scene['shape']) for card in scene['cards']]
                union = np.logical_or.reduce(masks)
                _require(np.array_equal(canonical_out != scene['bg'], union), 'lost/added footprint pixels')
                for i, card in enumerate(scene['cards']):
                    earlier = np.logical_or.reduce(masks[:i]) if i else np.zeros(scene['shape'], bool)
                    _require(bool(np.all(canonical_out[masks[i] & ~earlier] == card['color'])), 'wrong collision owner')
                features[split].append(_features(scene))
        train, test = features['train'], features['test']
        _require(any(f['omit_a'] for f in train), 'no training witness for actual coordinates')
        _require(any(f['omit_b'] for f in train), 'no training witness for completed reference')
        _require(any(g['faces'] for f in train for g in f['graphs'])
                 and any(g['hollow'] for f in train for g in f['graphs']), 'teach both hollow and filled chambers')
        if any(f['overlap'] for f in test):
            _require(all(any(f[k] for f in train) for k in ('destination_diff', 'size_diff', 'reverse_diff')),
                     'test collisions need distinguishing source-priority evidence')
        if any(f['inactive'] for f in test):
            _require(any(f['inactive'] for f in train), 'inactive test needs training evidence')
    except (TypeError, IndexError, KeyError, ValueError) as exc:
        raise AssertionError(str(exc)) from exc


def _signature(puzzle):
    return tuple(tuple((_key(x['input']), _key(x['output'])) for x in puzzle[s]) for s in ('train', 'test'))


def _assert_rejected(puzzle):
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        return
    raise AssertionError('validator accepted damaged puzzle')


def self_test():
    """Bounded, task-specific diagnostics and diversity over 42 seeds."""
    shapes, output_shapes, graphs, palettes, axes, gaps, signatures = set(), set(), set(), set(), set(), set(), set()
    orientations, counts, flags = set(), set(), set()
    total, local_varied = 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        signatures.add(_signature(puzzle))
        local = set()
        first, twin = puzzle['train'][:2]
        # Target field and sparse reference members are unchanged in the twins.
        one, two = _parse(first['input']), _parse(twin['input'])
        assert one['d'] == two['d'] and one['transposed'] == two['transposed']
        assert np.array_equal(one['grid'][one['d']:], two['grid'][two['d']:])
        assert [(c['rows'], c['cols']) for c in one['cards']] == [(c['rows'], c['cols']) for c in two['cards']]
        assert not np.array_equal(first['output'], twin['output'])
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                scene = _parse(pair['input'])
                f = _features(scene)
                assert np.array_equal(solve(pair['input']), pair['output'])
                # A global palette permutation preserves equality roles, even
                # when black becomes frame or network rather than background.
                recolor = (np.arange(10, dtype=np.int8)+3) % 10
                assert np.array_equal(solve(recolor[pair['input']]), recolor[pair['output']])
                # Transposition may reorder cards. Use displayed reading order
                # and fresh parsed footprints, not an assumed commuting priority.
                rotated = _parse(pair['input'].T)
                expected = _construction_output(rotated['cards'], rotated['shape'], rotated['bg'], rotated['transposed'])
                assert np.array_equal(solve(pair['input'].T), expected)
                total += 1
                shapes.add(pair['input'].shape)
                output_shapes.add(pair['output'].shape)
                palettes.add(f['roles'])
                local.add((pair['input'].shape, tuple(g['mask'] for g in f['graphs']), f['active']))
                orientations.add(f['transpose'])
                counts.add((f['cards'], f['active']))
                gaps.update(f['gaps'])
                for g in f['graphs']:
                    graphs.add(g['mask'])
                    axes.add(g['nodes'])
                    flags.update(k for k in ('faces', 'hollow', 'isolated', 'branches') if g[k])
                    if g['components'] > 1:
                        flags.add('disconnected')
                flags.update(k for k in ('overlap', 'omit_a', 'omit_b', 'inactive', 'destination_diff', 'size_diff', 'reverse_diff') if f[k])
        local_varied += len(local) >= 3
    assert len(signatures) == 42 and total == 210
    assert len(shapes) >= 30 and len(output_shapes) >= 25 and len(graphs) >= 65 and len(palettes) >= 50
    assert orientations == {False, True} and counts >= {(1,1), (2,1), (2,2)}
    assert len(axes) >= 6 and max(gaps) >= 6 and len(gaps) >= 5 and local_varied == 42
    assert flags >= {'faces', 'hollow', 'isolated', 'branches', 'disconnected', 'overlap',
                     'omit_a', 'omit_b', 'inactive', 'destination_diff', 'size_diff', 'reverse_diff'}
    # All legal count combinations, including the minimum evidence budget.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(918+a*10+b, a, b)
            assert (len(p['train']), len(p['test'])) == (a, b)
    for a, b in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (4,1.5), (None,1), (4,None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(0)
    bad = {s:[{k:np.array(v, copy=True) for k,v in x.items()} for x in p[s]] for s in p}
    bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0])+1) % 10
    _assert_rejected(bad)
    for ablation in ('nodes_only', 'uniform'):
        bad = {s:[{k:np.array(v, copy=True) for k,v in x.items()} for x in p[s]] for s in p}
        bad['train'][0]['output'] = _render_scene(_parse(bad['train'][0]['input']), **{ablation:True})
        _assert_rejected(bad)
    # Move one paired coordinate while preserving cardinality and gap legality.
    scene = _parse(p['train'][0]['input'])
    g = scene['grid'].copy()
    card = next(x for x in scene['cards'] if x['m'] == 3)
    old, new = card['cols'][1], card['cols'][1]+1
    g[scene['d']+1, old+1] = g[-1, old+1] = scene['frame']
    g[scene['d']+1, new+1] = g[-1, new+1] = card['color']
    displayed = g.T if scene['transposed'] else g
    assert not np.array_equal(solve(displayed), p['train'][0]['output'])
    # Valid family members outside the random sampler's packing preferences:
    # two 4x4-node cards stacked, adjacent DIFFERENT-color axis marks, 30 rows,
    # and the smallest fitting one-card scene with a 3x7 output.
    full = np.ones((7, 7), bool)
    hollow = full.copy()
    hollow[1::2, 1::2] = False
    cards = [dict(n=4, m=4, network=full, color=3, box=(0,0,9,17),
                  source_slot=1, rows=[0,2,4,6], cols=[0,2,4,6]),
             dict(n=4, m=4, network=hollow, color=4, box=(10,0,9,17),
                  source_slot=0, rows=[1,3,5,7], cols=[1,3,5,7])]
    for transpose in (False, True):
        edge = _assemble(cards, 19, 17, 8, np.arange(10), transpose)
        assert np.array_equal(solve(edge['input']), edge['output'])
    small_card = dict(n=2, m=2, network=np.ones((3,3), bool), color=3,
                      box=(0,0,5,9), source_slot=0, rows=[0,2], cols=[0,2])
    small = _assemble([small_card], 5, 9, 3, np.arange(10), False)
    assert small['input'].shape == (11,9) and small['output'].shape == (3,7)
    assert np.array_equal(solve(small['input']), small['output'])
    return {'seeds':42, 'pairs':total, 'input_shapes':len(shapes), 'output_shapes':len(output_shapes),
            'network_masks':len(graphs), 'role_palettes':len(palettes), 'node_counts':sorted(axes),
            'gaps':sorted(gaps), 'within_puzzle_structure_variation':local_varied,
            'features':sorted(flags), 'semantic_verification':'generator-owned only'}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
