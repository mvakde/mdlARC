"""Complete a self-indexing lattice inside an identity-oriented, exact crop.

The input is the whole specification: no generation state is used by solve.
Only standard-library modules and NumPy are required. No files are written.
"""
from __future__ import annotations

import hashlib
from itertools import product
import numpy as np

SIDES = ('T', 'R', 'B', 'L')
NORMALS = {'T': (-1, 0), 'R': (0, 1), 'B': (1, 0), 'L': (0, -1)}
SIDE_MAPS = {
    'identity': ('T', 'R', 'B', 'L'),
    'r90': ('L', 'T', 'R', 'B'),
    'r180': ('B', 'L', 'T', 'R'),
    'r270': ('R', 'B', 'L', 'T'),
    'flip_lr': ('T', 'L', 'B', 'R'),
    'flip_ud': ('B', 'R', 'T', 'L'),
    'transpose': ('L', 'B', 'R', 'T'),
    'anti_transpose': ('R', 'T', 'L', 'B'),
}
HOST_SEEDS = (0,541087346,570745391,471839837,428200697,62142045,945516131,
    384723844,262772267,636275768,296522417,690565793,775617026,418538946,
    807841814,380306951,147892046,120208538,445462574,968479052,889551112,
    861803191,696407537,103156584,885431655,81711408,282059705,529405507,
    994708141,903296102,484090267,346778169,679130801,972703798,857897764,
    256901199,57886169,745796643,382471223,45463416,340104461,906795947)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape),
             'grid dimensions must be 1..30')
    _require(np.issubdtype(grid.dtype, np.integer) and
             np.all((0 <= grid) & (grid <= 9)), 'grid cells must be integers 0..9')
    return grid


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    _require(num_train > num_test, 'num_train must exceed num_test')


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _key(grid):
    grid = np.asarray(grid, dtype=np.uint8)
    return grid.shape, grid.tobytes()


def _side(grid, side):
    return {'T': grid[0, :], 'R': grid[:, -1],
            'B': grid[-1, :], 'L': grid[:, 0]}[side]


def _adjacent(index):
    return (SIDES[index], SIDES[(index + 1) % 4])


def _transform(grid, name):
    if name == 'identity':
        result = grid
    elif name.startswith('r'):
        result = np.rot90(grid, {'r90': 1, 'r180': 2, 'r270': 3}[name])
    elif name == 'flip_lr':
        result = np.fliplr(grid)
    elif name == 'flip_ud':
        result = np.flipud(grid)
    elif name == 'transpose':
        result = grid.T
    elif name == 'anti_transpose':
        result = np.rot90(grid.T, 2)
    else:
        raise ValueError('unknown symmetry')
    return result.copy()


def _components(grid, color):
    remaining = set(zip(*np.where(grid == color)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = [], [start]
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in NORMALS.values():
                pos = r + dr, c + dc
                if pos in remaining:
                    remaining.remove(pos)
                    stack.append(pos)
        result.append(cells)
    return result


def _interior_bar(grid, color):
    H, W = grid.shape
    candidates = []
    for cells in _components(grid, color):
        if any(r in (0, H - 1) or c in (0, W - 1) for r, c in cells):
            continue
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        if len(cells) >= 5 and (r0 == r1 or c0 == c1):
            _require(len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1), 'broken guide')
            candidates.append((r0, c0, r1, c1))
    _require(len(candidates) == 1, 'each marker needs exactly one interior guide')
    return candidates[0]


def _lattice(payload, background):
    """A clue's hole gives its macro index, hence the unique lattice origin."""
    colors = set(map(int, np.unique(payload))) - {background}
    _require(len(colors) == 1, 'payload needs exactly one foreground role')
    foreground = next(iter(colors))
    clues, origins = [], set()
    for cells in _components(payload, foreground):
        rr, cc = zip(*cells)
        r, c = min(rr), min(cc)
        n, width = max(rr) - r + 1, max(cc) - c + 1
        _require(n in (2, 3, 4) and n == width and len(cells) == n*n - 1,
                 'each visible tile must be an n-by-n square with one hole')
        tile = payload[r:r+n, c:c+n]
        holes = np.argwhere(tile == background)
        _require(len(holes) == 1, 'visible tile needs exactly one hole')
        i, j = map(int, holes[0])
        origins.add((n, r - i*(n+1), c - j*(n+1)))
        clues.append((i, j))
    _require(len(origins) == 1, 'clues must identify one lattice and coordinate law')
    n, r0, c0 = origins.pop()
    size = n*n + n - 1
    _require(r0 >= 0 and c0 >= 0 and r0+size <= payload.shape[0]
             and c0+size <= payload.shape[1], 'lattice extends beyond guide crop')
    outside = np.ones(payload.shape, dtype=bool)
    outside[r0:r0+size, c0:c0+size] = False
    _require(np.all(payload[outside] == background), 'only background may surround lattice')
    board = payload[r0:r0+size, c0:c0+size]
    _require(np.all(board[n::n+1, :] == background) and
             np.all(board[:, n::n+1] == background), 'separator damage')
    mask = np.zeros((n, n), dtype=bool)
    for i, j in product(range(n), repeat=2):
        tile = board[i*(n+1):i*(n+1)+n, j*(n+1):j*(n+1)+n]
        if np.all(tile == background):
            continue
        expected = np.full((n, n), foreground, dtype=np.int8)
        expected[i, j] = background
        _require(np.array_equal(tile, expected), 'incorrect completed tile')
        mask[i, j] = True
    _require(1 <= mask.sum() < n*n, 'need a visible tile and a missing tile')
    return dict(n=n, origin=(r0, c0), mask=mask, foreground=foreground,
                padding=(r0, payload.shape[0]-r0-size,
                         c0, payload.shape[1]-c0-size))


def _decode(input_grid):
    grid = _grid(input_grid)
    H, W = grid.shape
    _require(min(H, W) >= 10, 'canvas too small for separated guides and controls')
    # The corners and the clear ring around the payload carry the background.
    background = int(grid[0, 0])
    _require(all(int(grid[r, c]) == background for r, c in
                 ((0, 0), (0, W-1), (H-1, 0), (H-1, W-1))), 'corner convention')
    controls = {}
    for side in SIDES:
        core = _side(grid, side)[1:-1]
        if np.all(core == core[0]) and int(core[0]) != background:
            controls[side] = int(core[0])
    _require(len(controls) == 2 and len(set(controls.values())) == 2
             and any(set(controls) == set(_adjacent(k)) for k in range(4)),
             'need two distinct adjacent outer controls')
    bars = [(color, _interior_bar(grid, color)) for color in controls.values()]
    horizontal = [item for item in bars if item[1][0] == item[1][2]]
    vertical = [item for item in bars if item[1][1] == item[1][3]]
    _require(len(horizontal) == len(vertical) == 1, 'need perpendicular guides')
    hc, (hr, cs, _, ce) = horizontal[0]
    vc, (rs, vcol, re, _) = vertical[0]
    h, w = re-rs+1, ce-cs+1
    _require(hr in (rs-1, re+1) and vcol in (cs-1, ce+1), 'guide endpoint alignment')
    guides = {('T' if hr == rs-1 else 'B'): hc,
              ('L' if vcol == cs-1 else 'R'): vc}
    r, c = rs, cs
    # One clear cell between the outer frame and the payload/guide box.
    rb, cb = min(hr, r), min(vcol, c)
    rend, cend = max(hr+1, r+h), max(vcol+1, c+w)
    _require(rb >= 2 and cb >= 2 and rend <= H-2 and cend <= W-2,
             'payload guides need a clear separation from frame')
    ring = np.concatenate((grid[rb-1, cb-1:cend+1], grid[rend, cb-1:cend+1],
                           grid[rb:rend, cb-1], grid[rb:rend, cend]))
    _require(np.all(ring == background), 'payload moat is not clear')
    payload = grid[r:r+h, c:c+w].copy()
    _require(not (set(map(int, np.unique(payload))) & set(controls.values())),
             'marker identities cannot be payload colors')
    expected_markers = np.full(grid.shape, background, dtype=np.int8)
    for side, color in controls.items():
        _side(expected_markers, side)[1:-1] = color
    for color, (br, bc, er, ec) in bars:
        expected_markers[br:er+1, bc:ec+1] = color
    for color in controls.values():
        _require(np.array_equal(grid == color, expected_markers == color),
                 'marker cells must belong to their control or guide')
    reserved = expected_markers != background
    reserved[r:r+h, c:c+w] = True
    decoration = (grid != background) & ~reserved
    dr, dc = np.where(decoration)
    _require(np.all((dr == 0) | (dr == H-1) | (dc == 0) | (dc == W-1))
             and len(np.unique(grid[decoration])) <= 1,
             'decoration must use one distinct color on the outer border')
    info = _lattice(payload, background)
    # All foreground belongs to the selected array; no second payload exists.
    outside = np.ones(grid.shape, dtype=bool)
    outside[r:r+h, c:c+w] = False
    _require(not np.any(grid[outside] == info['foreground']), 'foreground outside payload')
    matches = [name for name, targets in SIDE_MAPS.items()
               if all(controls.get(targets[SIDES.index(side)]) == color
                      for side, color in guides.items())]
    _require(len(matches) == 1, 'guide identities must select a unique D4 map')
    info.update(background=background, controls=controls, guides=guides,
                bounds=(r, c, h, w), payload=payload, transform=matches[0])
    return info


def _complete_payload(info):
    result = info['payload'].copy()
    n, (r0, c0) = info['n'], info['origin']
    for i, j in product(range(n), repeat=2):
        if info['mask'][i, j]:
            continue
        r, c = r0 + i*(n+1), c0 + j*(n+1)
        result[r:r+n, c:c+n] = info['foreground']
        result[r+i, c+j] = info['background']
    return result


def _coordinate_map(info):
    matrix = np.zeros((2, 2), dtype=int)
    for side, color in info['guides'].items():
        target = next(s for s, col in info['controls'].items() if col == color)
        matrix += np.outer(NORMALS[target], NORMALS[side])
    positions = np.indices(info['payload'].shape).reshape(2, -1)
    mapped = matrix @ positions
    mapped -= mapped.min(axis=1, keepdims=True)
    return mapped, tuple(map(int, mapped.max(axis=1)+1))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover crop, lattice coordinates and orientation from this input alone."""
    info = _decode(input_grid)
    completed = _complete_payload(info)
    coordinates, shape = _coordinate_map(info)
    output = np.empty(shape, dtype=np.int8)
    output[coordinates[0], coordinates[1]] = completed.ravel()
    return output


def _board(n, background, foreground):
    size = n*n+n-1
    result = np.full((size, size), background, dtype=np.int8)
    for i, j in product(range(n), repeat=2):
        r, c = i*(n+1), j*(n+1)
        result[r:r+n, c:c+n] = foreground
        result[r+i, c+j] = background
    return result


def _mask(rng, n, mode, anchor=False):
    result = np.zeros((n, n), dtype=bool)
    if anchor:
        # Three axis values, an off-diagonal hole, and >=2 blank tiles.
        rr = rng.choice(n, 3, replace=False)
        cc = rng.choice(n, 3, replace=False)
        result[rr, cc] = True
        if not any(i != j for i, j in zip(rr, cc)):
            choices = [(i, j) for i, j in product(range(n), repeat=2) if i != j]
            result[_pick(rng, choices)] = True
        free = np.argwhere(~result)
        extras = int(rng.integers(0, min(n, len(free)-2)+1))
        for index in rng.permutation(len(free))[:extras]:
            result[tuple(free[index])] = True
        return result
    if mode == 'band':
        axis = int(rng.integers(2))
        band = int(rng.integers(n))
        if axis == 0:
            result[band, :] = True
        else:
            result[:, band] = True
        if rng.random() < .35:
            positions = np.argwhere(result)
            result[tuple(positions[int(rng.integers(len(positions)))])] = False
        return result
    count = int(_pick(rng, (1, n, max(1, n*n//2), n*n-1, max(1, n*n-2))))
    style = int(rng.integers(4))
    noise = rng.random((n, n))
    cr, cc = int(rng.integers(n)), int(rng.integers(n))
    coords = list(product(range(n), repeat=2))
    if style == 0:
        coords.sort(key=lambda p: noise[p])
    elif style == 1:
        coords.sort(key=lambda p: abs(p[0]-cr)+abs(p[1]-cc)+noise[p]/2)
    elif style == 2:
        coords.sort(key=lambda p: (p[0] if cr % 2 else p[1])+noise[p]/2)
    else:
        coords.sort(key=lambda p: max(abs(p[0]-cr), abs(p[1]-cc))+noise[p]/2)
    for pos in coords[:count]:
        result[pos] = True
    return result


def _palette(rng):
    return tuple(map(int, rng.choice(10, 5, replace=False)))


def _make_pair(rng, n, transform, outer_index, palette, mode='mixed', anchor=False):
    bg, fg, marker_a, marker_b, noise = palette
    mask = _mask(rng, n, mode, anchor)
    board = _board(n, bg, fg)
    size = board.shape[0]
    # Both opposite margin pairs differ; unequal totals make all eight full
    # completed crops distinct. Later examples may also have symmetric crops.
    if anchor or rng.random() < .75:
        margins = [(t, b, l, r) for t, b, l, r in product(range(4), repeat=4)
                   if t != b and l != r and t+b != l+r]
        t, b, l, rr = _pick(rng, margins)
    else:
        t, b, l, rr = map(int, rng.integers(0, 4, size=4))
    h, w = size+t+b, size+l+rr
    full = np.full((h, w), bg, dtype=np.int8)
    full[t:t+size, l:l+size] = board
    partial = full.copy()
    for i, j in product(range(n), repeat=2):
        if not mask[i, j]:
            row, col = t+i*(n+1), l+j*(n+1)
            partial[row:row+n, col:col+n] = bg
    H, W = int(rng.integers(h+5, 31)), int(rng.integers(w+5, 31))
    if rng.random() < .12:
        H = W = 30
    outer_sides = _adjacent(outer_index)
    targets = SIDE_MAPS[transform]
    inner_sides = [side for side in SIDES if targets[SIDES.index(side)] in outer_sides]
    guides = dict(zip(inner_sides, (marker_a, marker_b)))
    controls = {targets[SIDES.index(side)]: color for side, color in guides.items()}
    row = int(rng.integers(2+('T' in guides), H-h-1-('B' in guides)))
    col = int(rng.integers(2+('L' in guides), W-w-1-('R' in guides)))
    grid = np.full((H, W), bg, dtype=np.int8)
    grid[row:row+h, col:col+w] = partial
    for side, color in guides.items():
        if side == 'T':
            grid[row-1, col:col+w] = color
        elif side == 'B':
            grid[row+h, col:col+w] = color
        elif side == 'L':
            grid[row:row+h, col-1] = color
        else:
            grid[row:row+h, col+w] = color
    for side, color in controls.items():
        _side(grid, side)[1:-1] = color
    # Fragmented decoration is geometrically weaker than a full control.
    if rng.random() < .65:
        for side in SIDES:
            if side in controls:
                continue
            edge = _side(grid, side)
            period = int(rng.integers(2, 6))
            offset = int(rng.integers(period))
            for pos in range(1, len(edge)-1):
                if pos % period == offset and rng.random() < .8:
                    edge[pos] = noise
    return {'input': grid, 'output': _transform(full, transform)}


def _swap_outer(pair):
    copied = {name: grid.copy() for name, grid in pair.items()}
    info = _decode(copied['input'])
    colors = list(info['controls'].values())
    for side, color in info['controls'].items():
        _side(copied['input'], side)[1:-1] = colors[1-colors.index(color)]
    # Construction uses a named transform, distinct from solve's coordinate map.
    changed = _decode(copied['input'])
    copied['output'] = _transform(_complete_payload(info), changed['transform'])
    return copied


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    held_outer = int(rng.integers(4))
    allowed = [k for k in range(4) if k != held_outer]
    regime = int(rng.integers(3))
    anchor_n = 3 if regime == 0 else int(_pick(rng, (3, 4)))
    stable = bool(rng.random() < .25)
    palette = _palette(rng)
    first = _make_pair(rng, anchor_n, _pick(rng, ('r90', 'r270')),
                       _pick(rng, allowed), palette, anchor=True)
    puzzle = {'train': [first, _swap_outer(first)], 'test': []}
    seen = {_key(pair['input']) for pair in puzzle['train']}
    for index in range(2, int(num_train)+int(num_test)):
        train = index < num_train
        # Regime 0 extrapolates scale; regime 1 tests an erased macro band;
        # regime 2 mixes sizes. Every regime holds out an outer corner.
        n = int(_pick(rng, (2, 3) if train and regime == 0 else (2, 3, 4)))
        if not train and regime == 0:
            n = 4
        if not train and regime == 1:
            n = anchor_n
        mode = 'band' if not train and regime == 1 else 'mixed'
        for attempt in range(32):
            pair = _make_pair(rng, n, _pick(rng, tuple(SIDE_MAPS)),
                              _pick(rng, allowed) if train else held_outer,
                              palette if stable else _palette(rng), mode)
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key)
                break
        else:
            raise RuntimeError('bounded distinct-input sampling exhausted')
        puzzle['train' if train else 'test'].append(pair)
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
    inp, output = _grid(pair['input']), _grid(pair['output'])
    info = _decode(inp)
    coords, shape = _coordinate_map(info)
    _require(output.shape == shape, 'output extent must equal the transformed full crop')
    # Pull the answer back to source coordinates. Check the mathematical cell
    # law directly rather than using generation's board or completion helper.
    restored = output[coords[0], coords[1]].reshape(info['payload'].shape)
    n, (r, c), bg, fg = info['n'], info['origin'], info['background'], info['foreground']
    size = n*n+n-1
    expected = np.full(restored.shape, bg, dtype=np.int8)
    u, v = np.indices((size, size))
    paint = (u % (n+1) != n) & (v % (n+1) != n)
    paint &= ~((u % (n+2) == 0) & (v % (n+2) == 0))
    expected[r:r+size, c:c+size][paint] = fg
    _require(np.array_equal(restored, expected), 'output violates coordinate-hole law')
    original = info['payload']
    _require(np.all(restored[original == fg] == fg), 'given tile cells were not preserved')
    additions = int(np.count_nonzero(restored == fg)-np.count_nonzero(original == fg))
    _require(additions == (n*n-int(info['mask'].sum()))*(n*n-1), 'completion area invariant')
    _require(np.array_equal(solve(inp), output), 'input-only solver mismatch')
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    infos, seen = {}, set()
    for split in ('train', 'test'):
        infos[split] = []
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            key = _key(pair['input'])
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            infos[split].append(info)
    train = infos['train']
    _require(any(len({i for i, j in np.argwhere(a['mask'])}) >= 3
                 and len({j for i, j in np.argwhere(a['mask'])}) >= 3
                 and any(i != j for i, j in np.argwhere(a['mask']))
                 and a['n']*a['n']-a['mask'].sum() >= 2 for a in train),
             'training must expose both coordinate axes and distinct missing holes')
    contrasts = [(a, b) for i, a in enumerate(train) for b in train[i+1:]
                 if a['bounds'] == b['bounds'] and a['guides'] == b['guides']
                 and np.array_equal(a['payload'], b['payload'])
                 and a['controls'] != b['controls']
                 and a['transform'] != b['transform']]
    _require(contrasts, 'need a copied-payload marker correspondence contrast')
    a, b = contrasts[0]
    full = _complete_payload(a)
    _require(len({_key(_transform(full, name)) for name in SIDE_MAPS}) == 8,
             'training full crop must distinguish all eight symmetries')
    _require(any(a['transform'] in ('r90', 'r270') for a in train),
             'training needs a directed quarter turn')
    _require(any(a['transform'] in ('flip_lr', 'flip_ud', 'transpose', 'anti_transpose')
                 for a in train), 'training needs a reflection')
    _require(any(any(a['padding']) for a in train), 'training needs full background crop witness')
    train_corners = {frozenset(a['controls']) for a in train}
    _require(all(frozenset(a['controls']) not in train_corners for a in infos['test']),
             'tests must hold out an outer corner')


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for name in ('input', 'output'):
                shape, content = _key(pair[name])
                digest.update(bytes(shape)); digest.update(content)
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed checks. Retain only small metrics and digests, no files."""
    transforms = {s: set() for s in ('train', 'test')}
    orders, shapes, masks, paddings, palettes, signatures = set(), set(), set(), set(), set(), set()
    sparse = near_full = gaps = symmetric = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = _signature(puzzle)
        _require(digest == _signature(generate_puzzle(seed)), 'nondeterminism')
        signatures.add(digest)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _decode(pair['input'])
                n, mask = a['n'], a['mask']
                transforms[split].add(a['transform'])
                orders.add(n); shapes.add(pair['input'].shape)
                masks.add((n, mask.tobytes())); paddings.add(a['padding'])
                palettes.add((a['background'], a['foreground'], *a['guides'].values()))
                sparse += int(mask.sum()*3 <= n*n)
                near_full += int(mask.sum() >= n*n-2)
                gaps += int(not mask.any(axis=0).all() or not mask.any(axis=1).all())
                full = _complete_payload(a)
                symmetric += int(len({_key(_transform(full, t)) for t in SIDE_MAPS}) < 8)
                # Omitting A (raw crop) or its distinct holes must change it.
                raw = _transform(a['payload'], a['transform'])
                _require(not np.array_equal(raw, pair['output']), 'A is vacuous')
                uniform = full.copy()
                r0, c0 = a['origin']
                for i, j in np.argwhere(~mask):
                    uniform[r0+i*(n+1)+i, c0+j*(n+1)+j] = a['foreground']
                _require(not np.array_equal(_transform(uniform, a['transform']), pair['output']),
                         'coordinate holes are vacuous')
    _require(len(signatures) == 42 and orders == {2, 3, 4}, 'seed/order coverage')
    _require(all(values == set(SIDE_MAPS) for values in transforms.values()), 'D4 coverage')
    _require(len(shapes) >= 45 and len(masks) >= 65 and len(paddings) >= 60
             and len(palettes) >= 80, 'structural variation coverage')
    _require(sparse > 0 and near_full > 0 and gaps > 0 and symmetric > 0,
             'erasure and symmetry variation')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(97+a*11+b, a, b)
            validate_generated_puzzle(puzzle)
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),
                 (4.0,1),(4,1.0),(None,1),(4,None),(-1,1),(4,-1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Corruption checks exercise separators, clue coordinates, guides, crop,
    # and output cells. Each starts from a fresh bounded 5-pair puzzle.
    for kind in ('separator', 'clue', 'guide', 'crop', 'output'):
        bad = generate_puzzle(0)
        pair = bad['train'][0]
        info = _decode(pair['input'])
        r, c, h, w = info['bounds']
        n, (r0, c0) = info['n'], info['origin']
        if kind == 'separator':
            pair['input'][r+r0+n, c+c0] = info['foreground']
        elif kind == 'clue':
            i, j = np.argwhere(info['mask'])[0]
            pair['input'][r+r0+i*(n+1)+i, c+c0+j*(n+1)+j] = info['foreground']
        elif kind == 'guide':
            side, color = next(iter(info['guides'].items()))
            cells = next(cells for cells in _components(pair['input'], color)
                         if not any(rr in (0, pair['input'].shape[0]-1) or
                                    cc in (0, pair['input'].shape[1]-1) for rr, cc in cells))
            pair['input'][cells[len(cells)//2]] = info['background']
        elif kind == 'crop':
            pair['output'] = pair['output'][1:, :].copy()
        else:
            pair['output'][0, 0] = (int(pair['output'][0, 0])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            continue
        raise AssertionError('corrupted puzzle accepted: '+kind)
    return {'seeds': 42, 'pairs': 210, 'orders': sorted(orders),
            'symmetries': {s: sorted(t) for s, t in transforms.items()},
            'canvas_shapes': len(shapes), 'clue_masks': len(masks),
            'padding_tuples': len(paddings), 'role_palettes': len(palettes),
            'sparse_pairs': sparse, 'near_complete_pairs': near_full,
            'missing_axis_pairs': gaps, 'symmetric_completed_crops': symmetric}
