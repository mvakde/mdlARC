"""Topological mail sorting: exact exceptions route and overprint by hole count.

Only NumPy and the Python standard library are used. Generation performs no IO;
solve parses only its input. See task_spec.md for the fixed family conventions.
"""
from collections import Counter
import hashlib

import numpy as np


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    groups = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            a, b = stack.pop()
            cells.append((a, b))
            for x, y in ((a-1, b), (a+1, b), (a, b-1), (a, b+1)):
                if 0 <= x < h and 0 <= y < w and mask[x, y] and not seen[x, y]:
                    seen[x, y] = True
                    stack.append((x, y))
        groups.append(cells)
    return groups


def _holes(mask):
    """The padded complement contains exactly one exterior component."""
    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    return [cells for cells in _components(~padded)
            if all(0 < r < padded.shape[0]-1 and 0 < c < padded.shape[1]-1
                   for r, c in cells)]


def _count(mask):
    return len(_holes(mask))


def _connected(mask):
    return len(_components(mask)) == 1


def _signature(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _check_grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((0 <= g) & (g <= 9))
    return g


def _assemble(headers, tiles, lattice):
    bands, copies = len(headers), len(tiles[0])
    ph, kw = headers[0].shape
    pw = tiles[0][0].shape[1]
    grid = np.full((bands*(ph+1)+1, kw+2+copies*(pw+1)), lattice, dtype=np.int8)
    for b in range(bands):
        r = b*(ph+1)+1
        grid[r:r+ph, 1:kw+1] = headers[b]
        for j, tile in enumerate(tiles[b]):
            c = kw+2+j*(pw+1)
            grid[r:r+ph, c:c+pw] = tile
    return grid


def _parse(input_grid):
    g = _check_grid(input_grid)
    lattice = int(g[0, 0])
    rs = np.flatnonzero(np.all(g == lattice, axis=1))
    cs = np.flatnonzero(np.all(g == lattice, axis=0))
    assert len(rs) >= 3 and len(cs) >= 5, 'need shelves, address, and three copies'
    assert rs[0] == cs[0] == 0 and rs[-1] == g.shape[0]-1 and cs[-1] == g.shape[1]-1
    assert np.all(np.diff(rs) == rs[1])
    assert np.all(np.diff(cs[1:]) == cs[2]-cs[1])
    ph, kw, pw = int(rs[1]-1), int(cs[1]-1), int(cs[2]-cs[1]-1)
    assert min(ph, kw, pw) >= 1
    headers, rows = [], []
    for r, s in zip(rs, rs[1:]):
        cells = [g[r+1:s, c+1:d].copy() for c, d in zip(cs, cs[1:])]
        assert all(np.all(p != lattice) for p in cells)
        headers.append(cells[0])
        rows.append(cells[1:])
    bg = int(headers[0][0, 0])
    assert bg != lattice
    mapping, colors, keys, glyphs = {}, set(), [], []
    for b, header in enumerate(headers):
        assert np.all(header[[0, -1], :] == bg) and np.all(header[:, [0, -1]] == bg)
        ink_colors = set(map(int, np.unique(header))) - {bg}
        assert len(ink_colors) == 1
        color = ink_colors.pop()
        mask = header == color
        assert _connected(mask) and color not in colors and color != lattice
        k = _count(mask)
        assert k not in mapping, 'duplicate topological address'
        mapping[k] = (b, color)
        keys.append(k)
        glyphs.append(mask)
        colors.add(color)
    specimen_colors = set(int(x) for row in rows for p in row for x in np.unique(p))
    assert len(specimen_colors - {bg}) == 1
    query = (specimen_colors - {bg}).pop()
    assert query not in colors | {bg, lattice}
    masters, payloads, columns = [], [], []
    for row in rows:
        counts = Counter(_signature(p) for p in row)
        repeated = [key for key, n in counts.items() if n == len(row)-1]
        assert len(counts) == 2 and len(repeated) == 1, 'one exact bitmap exception required'
        master_key = repeated[0]
        j = next(i for i, p in enumerate(row) if _signature(p) != master_key)
        master = next(p == query for p in row if _signature(p) == master_key)
        payload = row[j] == query
        assert _connected(master) and _connected(payload)
        masters.append(master)
        payloads.append(payload)
        columns.append(j)
    return dict(lattice=lattice, bg=bg, query=query, headers=headers,
                rows=rows, ph=ph, pw=pw, kw=kw, bands=len(rows), copies=len(rows[0]),
                mapping=mapping, keys=keys, glyphs=glyphs, masters=masters,
                payloads=payloads, columns=columns)


def _routed_masks(masks, mapping):
    result = [np.zeros(masks[0].shape, dtype=bool) for _ in range(len(mapping))]
    for mask in masks:
        address = mapping.get(_count(mask))
        if address is not None:
            result[address[0]] |= mask
    return result


def _output(parsed, masks=None, local=False, use_master_address=False):
    payloads = parsed['payloads'] if masks is None else masks
    if local:
        merged = payloads
    elif use_master_address:
        merged = [np.zeros(payloads[0].shape, dtype=bool) for _ in payloads]
        for payload, master in zip(payloads, parsed['masters']):
            address = parsed['mapping'].get(_count(master))
            if address is not None:
                merged[address[0]] |= payload
    else:
        merged = _routed_masks(payloads, parsed['mapping'])
    tiles = []
    for b, mask in enumerate(merged):
        tile = np.full(mask.shape, parsed['bg'], dtype=np.int8)
        color = parsed['mapping'][parsed['keys'][b]][1]
        tile[mask] = color
        tiles.append([tile])
    return _assemble(parsed['headers'], tiles, parsed['lattice'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive bitmap exceptions, their topology, routing and union from pixels."""
    return _output(_parse(input_grid))


def _glyph_options(k, h, w):
    if k == 0:
        return [(2, 2)] if min(h, w) >= 2 else []
    return [(a, b) for a in range(3, h+1, 2) for b in range(3, w+1, 2)
            if ((a-1)//2)*((b-1)//2) >= k]


def _key_pool(ph, kw):
    return [k for k in range(5) if _glyph_options(k, ph-2, kw-2)]


def _capacity(ph, pw):
    return ((ph-1)//2)*((pw-1)//2)


# Enumerated once, in RAM. No seed-specific solution state is retained.
_GEOMETRIES = tuple(
    (ph, pw, kw, bands, copies)
    for ph in range(5, 10) for pw in range(3, 8) for kw in (5, 7, 9)
    for bands in range(2, 5) for copies in range(3, 6)
    if bands*(ph+1)+1 <= 30 and kw+2+copies*(pw+1) <= 30
    and len(_key_pool(ph, kw)) >= bands and _capacity(ph, pw) >= bands-1
)


def _glyph(k, ph, kw, rng):
    opts = _glyph_options(k, ph-2, kw-2)
    h, w = opts[int(rng.integers(len(opts)))]
    glyph = np.ones((h, w), dtype=bool)
    if k:
        slots = [(r, c) for r in range(1, h-1, 2) for c in range(1, w-1, 2)]
        for i in rng.choice(len(slots), k, replace=False):
            glyph[slots[int(i)]] = False
    canvas = np.zeros((ph, kw), dtype=bool)
    r = int(rng.integers(1, ph-h))
    c = int(rng.integers(1, kw-w))
    canvas[r:r+h, c:c+w] = glyph
    assert _connected(canvas) and _count(canvas) == k
    return canvas


def _object(k, ph, pw, rng):
    """Finite connected shape construction; hole/contour edits check topology."""
    if k == 0:
        h = int(rng.integers(2, ph+1))
        w = int(rng.integers(2, pw+1))
        style = int(rng.integers(3))
        mask = np.ones((h, w), dtype=bool)
        if style == 1:
            mask[:] = False
            mask[-1, :] = mask[:, 0] = mask[:, -1] = True
        elif style == 2:
            mask[:] = False
            mask[h//2, w//2] = True
            for _ in range(h*w):
                neighbors = np.zeros(mask.shape, dtype=np.int8)
                neighbors[1:] += mask[:-1]
                neighbors[:-1] += mask[1:]
                neighbors[:, 1:] += mask[:, :-1]
                neighbors[:, :-1] += mask[:, 1:]
                choices = np.argwhere(~mask & (neighbors == 1))
                if not len(choices) or mask.sum() >= max(3, h*w*.6):
                    break
                r, c = choices[int(rng.integers(len(choices)))]
                candidate = mask.copy()
                candidate[r, c] = True
                if _count(candidate) == 0:
                    mask = candidate
    else:
        choices = [(h, w) for h in range(3, ph+1) for w in range(3, pw+1)
                   if _capacity(h, w) >= k]
        h, w = choices[int(rng.integers(len(choices)))]
        mask = np.ones((h, w), dtype=bool)
        slots = [(r, c) for r in range(1, h-1, 2) for c in range(1, w-1, 2)]
        for i in rng.choice(len(slots), k, replace=False):
            mask[slots[int(i)]] = False
        # Grow unequal, potentially bent cavities without joining or opening them.
        for _ in range(int(rng.integers(0, 12))):
            candidates = [(r, c) for r in range(1, h-1) for c in range(1, w-1)
                          if mask[r, c] and any(not mask[a, b] for a, b in
                          ((r-1, c), (r+1, c), (r, c-1), (r, c+1)))]
            if not candidates:
                break
            p = candidates[int(rng.integers(len(candidates)))]
            candidate = mask.copy()
            candidate[p] = False
            if _connected(candidate) and _count(candidate) == k:
                mask = candidate
    # Sculpt exterior contours too. Small shapes remain legitimate family members.
    for _ in range(int(rng.integers(0, 10))):
        exterior = np.zeros(mask.shape, dtype=bool)
        for cells in _components(~np.pad(mask, 1)):
            if any(r in (0, h+1) or c in (0, w+1) for r, c in cells):
                for r, c in cells:
                    if 1 <= r <= h and 1 <= c <= w:
                        exterior[r-1, c-1] = True
        edge = [(r, c) for r, c in np.argwhere(mask)
                if r in (0, h-1) or c in (0, w-1) or any(exterior[a, b] for a, b in
                ((r-1, c), (r+1, c), (r, c-1), (r, c+1)))]
        if not edge:
            break
        candidate = mask.copy()
        candidate[edge[int(rng.integers(len(edge)))]] = False
        if _connected(candidate) and _count(candidate) == k:
            mask = candidate
    if rng.random() < .5:
        mask = mask[::-1]
    if rng.random() < .5:
        mask = mask[:, ::-1]
    result = np.zeros((ph, pw), dtype=bool)
    r, c = int(rng.integers(ph-h+1)), int(rng.integers(pw-w+1))
    result[r:r+h, c:c+w] = mask
    assert _connected(result) and _count(result) == k
    return result


def _balanced_template(mask):
    """Find a different bitmap with the same area and topology, or return None."""
    k = _count(mask)
    for candidate in (mask[::-1], mask[:, ::-1], mask[::-1, ::-1]):
        if not np.array_equal(candidate, mask):
            return candidate.copy()
    # Translate when the symmetric object leaves room in its tile.
    for axis in (0, 1):
        for step in (-1, 1):
            if not np.take(mask, 0 if step < 0 else mask.shape[axis]-1, axis=axis).any():
                candidate = np.roll(mask, step, axis=axis)
                if not np.array_equal(candidate, mask):
                    return candidate
    # Bounded local exchange for a shape with no usable reflection or translation.
    for r, c in np.argwhere(mask):
        for a, b in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= a < mask.shape[0] and 0 <= b < mask.shape[1] and not mask[a, b]:
                candidate = mask.copy()
                candidate[r, c], candidate[a, b] = False, True
                if _connected(candidate) and _count(candidate) == k:
                    return candidate
    return None


def _topology_template(mask, rng):
    k = _count(mask)
    if k:
        # Fill one complete cavity, never a partial region.
        candidate = mask.copy()
        cells = _holes(mask)[int(rng.integers(k))]
        for r, c in cells:
            candidate[r-1, c-1] = True
        assert _connected(candidate) and _count(candidate) == k-1
        return candidate
    return _object(1, *mask.shape, rng)


def _make_pair(rng, geometry, palette, teaching=None):
    ph, pw, kw, bands, copies = geometry
    bg, query, lattice = palette[:3]
    pool = [k for k in _key_pool(ph, kw) if k <= _capacity(ph, pw)]
    if teaching == 'merge':
        keys = [int(x) for x in rng.choice([0, 1, 2, 3], 3, replace=False)]
        retained = int(rng.choice([k for k in keys if k > 0]))
        absent = next(k for k in range(4) if k not in keys)
        counts = [retained, retained, absent]
    elif teaching == 'cross':
        keys = [0, 1]
        rng.shuffle(keys)
        counts = keys[::-1]
    else:
        keys = [int(k) for k in rng.choice(pool, bands, replace=False)]
        counts = [int(rng.choice(keys)) if rng.random() < .75 else
                  int(rng.integers(min(6, _capacity(ph, pw))+1)) for _ in range(bands)]
    colors = [int(x) for x in palette[3:3+bands]]
    headers = []
    for k, color in zip(keys, colors):
        header = np.full((ph, kw), bg, dtype=np.int8)
        header[_glyph(k, ph, kw, rng)] = color
        headers.append(header)
    payloads = [_object(k, ph, pw, rng) for k in counts]
    if teaching == 'merge':
        # A guaranteed nontrivial same-class collision. Equal-area vertical
        # translations each supply unique pixels, and leave room for a mirror.
        base = _object(counts[0], ph-1, pw, rng)
        payloads[0] = np.pad(base, ((0, 1), (0, 0)))
        payloads[1] = np.pad(base, ((1, 0), (0, 0)))
        # Randomly assign the three roles to source shelves; destinations remain
        # determined only by the address glyphs, never by this assignment.
        order = list(map(int, rng.permutation(bands)))
        payloads = [payloads[i] for i in order]
    masters = []
    for i, payload in enumerate(payloads):
        master = _balanced_template(payload) if teaching == 'merge' or rng.random() < .45 else None
        if master is None:
            master = _topology_template(payload, rng)
        if teaching == 'cross' and i == 0:
            master = _topology_template(payload, rng)
        assert not np.array_equal(master, payload) and _connected(master)
        masters.append(master)
    columns = list(map(int, rng.integers(copies, size=bands)))
    if teaching == 'merge' and len(set(columns)) == 1:
        columns[1] = (columns[0]+1) % copies
    rows = []
    for i in range(bands):
        row = []
        for j in range(copies):
            tile = np.full((ph, pw), bg, dtype=np.int8)
            tile[payloads[i] if j == columns[i] else masters[i]] = query
            row.append(tile)
        rows.append(row)
    # Construct truth from generated masks, without invoking parsing or solve.
    mapping = {k: (b, colors[b]) for b, k in enumerate(keys)}
    output_masks = [np.zeros((ph, pw), dtype=bool) for _ in range(bands)]
    for payload in payloads:
        k = _count(payload)
        if k in mapping:
            output_masks[mapping[k][0]] |= payload
    output_rows = []
    for b, mask in enumerate(output_masks):
        tile = np.full((ph, pw), bg, dtype=np.int8)
        tile[mask] = colors[b]
        output_rows.append([tile])
    return {'input': _assemble(headers, rows, lattice),
            'output': _assemble(headers, output_rows, lattice)}


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f'illegal {name}')
    return value


def _choose(rng, choices):
    assert choices, 'geometry constraints must be feasible'
    return choices[int(rng.integers(len(choices)))]


def _unique(pair, seen):
    if _signature(pair['input']) not in seen:
        return pair
    # Up to 14 pairs; 90 role permutations here give a finite collision fallback.
    a, b = sorted(map(int, np.unique(pair['input'])))[:2]
    for x in range(10):
        for y in range(10):
            if x == y:
                continue
            rest = [c for c in range(10) if c not in (x, y)]
            old = [c for c in range(10) if c not in (a, b)]
            lut = np.zeros(10, dtype=np.int8)
            lut[a], lut[b], lut[old] = x, y, rest
            result = {key: lut[grid] for key, grid in pair.items()}
            if _signature(result['input']) not in seen:
                return result
    raise AssertionError('finite distinct-palette fallback exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    ne = _integer(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(['shelves', 'copies', 'height']))
    stable_palette = bool(rng.random() < .25)
    palette = tuple(map(int, rng.permutation(10)))
    training = [g for g in _GEOMETRIES if g[3] <= 3 and g[4] == 3
                and (regime != 'height' or g[0] <= 7)]
    merge_choices = [g for g in training if g[0] == 7 and g[1] >= 5
                     and g[2] >= 7 and g[3] == 3]
    cross_choices = [g for g in training if g[3] == 2]
    if regime == 'shelves':
        testing = [g for g in _GEOMETRIES if g[3] == 4]
    elif regime == 'copies':
        testing = [g for g in _GEOMETRIES if g[4] > 3]
    else:
        testing = [g for g in _GEOMETRIES if g[0] > 7]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            teaching = ('merge' if i == 0 else 'cross' if i == 1 else None) if split == 'train' else None
            choices = (merge_choices if teaching == 'merge' else cross_choices
                       if teaching == 'cross' else training if split == 'train' else testing)
            geometry = _choose(rng, choices)
            if not stable_palette:
                palette = tuple(map(int, rng.permutation(10)))
            pair = _unique(_make_pair(rng, geometry, palette, teaching), seen)
            seen.add(_signature(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_features(pair):
    p = _parse(pair['input'])
    out = np.asarray(pair['output'])
    assert np.array_equal(_output(p), out), 'incorrect routing, union, recoloring or framing'
    counts = [_count(mask) for mask in p['payloads']]
    masters = [_count(mask) for mask in p['masters']]
    groups = [[mask for mask, k in zip(p['payloads'], counts) if k == address]
              for address in p['keys']]
    # Each of at least two messages adds pixels absent from the other messages.
    collision = False
    for group in groups:
        if len(group) >= 2:
            independent = 0
            for i, mask in enumerate(group):
                others = np.logical_or.reduce([m for j, m in enumerate(group) if j != i])
                independent += int(np.any(mask & ~others))
            collision |= independent >= 2
    return dict(shape=np.asarray(pair['input']).shape,
                geometry=(p['ph'], p['pw'], p['kw'], p['bands'], p['copies']),
                keys=tuple(p['keys']), counts=tuple(counts), columns=tuple(p['columns']),
                palette=(p['bg'], p['query'], p['lattice']),
                erasures=sum(k not in p['mapping'] for k in counts),
                retained=len(set(counts) & set(p['keys'])), collision=bool(collision),
                unused=sum(k not in counts for k in p['keys']),
                cross=sum(k in p['mapping'] and p['mapping'][k][0] != i
                          for i, k in enumerate(counts)),
                balanced=any(np.sum(a) == np.sum(b) and x == y
                             for a, b, x, y in zip(p['masters'], p['payloads'], masters, counts)),
                topology_change=masters != counts,
                without_a=not np.array_equal(_output(p, p['masters']), out),
                without_b=not np.array_equal(_output(p, local=True), out),
                wrong_address=not np.array_equal(_output(p, use_master_address=True), out),
                hole_areas=tuple(len(hole) for mask in p['payloads'] for hole in _holes(mask)),
                irregular_holes=sum(len(hole) != (max(r for r, c in hole)-min(r for r, c in hole)+1)*
                                    (max(c for r, c in hole)-min(c for r, c in hole)+1)
                                    for mask in p['payloads'] for hole in _holes(mask)),
                edge=sum(bool(mask[0].any() or mask[-1].any() or
                              mask[:, 0].any() or mask[:, -1].any()) for mask in p['payloads']),
                masks=tuple(_signature(mask) for mask in p['payloads']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                _check_grid(grid)
            key = _signature(pair['input'])
            assert key not in seen, 'duplicate input'
            seen.add(key)
            f = _pair_features(pair)
            if split == 'train':
                training.append(f)
    assert any(f['balanced'] for f in training), 'expose exact bitmap comparison'
    assert any(f['collision'] for f in training), 'expose aligned union'
    assert any(f['erasures'] for f in training), 'expose unlisted-class erasure'
    assert any(f['retained'] >= 2 for f in training), 'expose at least two class colors'
    assert any(f['cross'] >= 2 for f in training), 'expose cross-shelf routing'
    assert any(len(set(f['columns'])) > 1 for f in training), 'exclude fixed-column selection'
    for field in ('without_a', 'without_b', 'topology_change', 'wrong_address'):
        assert any(f[field] for f in training), 'missing teaching witness: '+field


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for key in ('input', 'output'):
                g = np.asarray(pair[key], dtype=np.int8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def self_test():
    """Stream 42 puzzles; bounded semantic fixtures, mutations and count checks."""
    diagonal = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    assert _count(diagonal) == 1
    diagonal[0, 1] = False
    assert _count(diagonal) == 0
    assert _count(np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)) == 0
    assert _count(np.ones((2, 6), dtype=bool)) == 0
    # Hand-computed routing fixture: address 1 is the second shelf; two distinct
    # one-hole 3x3 rings arrive, their union is a 3x4 rectangle border with a seam.
    x = np.zeros((5, 5), dtype=bool)
    x[0:3, 0:3] = True
    x[1, 1] = False
    y = np.zeros((5, 5), dtype=bool)
    y[0:3, 1:4] = True
    y[1, 2] = False
    z = np.zeros((5, 5), dtype=bool)
    z[2:, 2:] = True
    z[3, 3] = False
    z[2, 3] = False  # Open notch: class zero, absent below.
    routed = _routed_masks([x, y, z], {2: (0, 4), 1: (1, 7), 3: (2, 9)})
    expected = np.zeros((5, 5), dtype=bool)
    expected[:3, :4] = True
    assert not routed[0].any() and not routed[2].any()
    assert np.array_equal(routed[1], expected) and _count(expected) == 0
    # This fixture explicitly disproves reclassifying the union (address 1 -> 0).
    records, signatures = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_digest(p))
        records.extend(_pair_features(pair) for split in p for pair in p[split])
    assert len(signatures) == 42
    assert len({f['shape'] for f in records}) >= 20
    assert len({f['geometry'] for f in records}) >= 25
    assert {f['geometry'][3] for f in records} == {2, 3, 4}
    assert {f['geometry'][4] for f in records} >= {3, 4}
    assert len({f['keys'] for f in records}) >= 12
    assert len({f['masks'] for f in records}) >= 100
    assert set(c for f in records for c in f['palette']) == set(range(10))
    assert any(f['irregular_holes'] for f in records)
    assert max(a for f in records for a in f['hole_areas']) >= 3
    assert any(f['edge'] for f in records)
    assert any(f['erasures'] == 0 for f in records)
    assert any(f['erasures'] > 0 for f in records)
    assert any(f['unused'] == 0 for f in records)
    assert any(f['unused'] > 0 for f in records)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(800+nt*10+ne, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    assert _digest(generate_puzzle(85952142, 2, 1)) == _digest(generate_puzzle(85952142, 2, 1))
    # Verify each structural expansion, regardless of seed's chosen regime.
    for seed in range(8):
        p = generate_puzzle(seed)
        train = [_parse(q['input']) for q in p['train']]
        test = [_parse(q['input']) for q in p['test']]
        assert all(t['bands'] > max(x['bands'] for x in train) or
                   t['copies'] > max(x['copies'] for x in train) or
                   t['ph'] > max(x['ph'] for x in train) for t in test)
    p = generate_puzzle(0, 2, 1)
    for defect in ('output', 'master', 'duplicate_key'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = bad['train'][0]
        parsed = _parse(pair['input'])
        if defect == 'output':
            pair['output'][0, 0] = (int(pair['output'][0, 0])+1) % 10
        elif defect == 'master':
            # Remove the exceptional tile entirely from one shelf.
            j = parsed['columns'][0]
            c = parsed['kw']+2+j*(parsed['pw']+1)
            tile = np.full((parsed['ph'], parsed['pw']), parsed['bg'], dtype=np.int8)
            tile[parsed['masters'][0]] = parsed['query']
            pair['input'][1:1+parsed['ph'], c:c+parsed['pw']] = tile
        else:
            # Duplicate one address topology with the other shelf's own color.
            r = parsed['ph']+2
            color = parsed['mapping'][parsed['keys'][1]][1]
            header = np.full((parsed['ph'], parsed['kw']), parsed['bg'], dtype=np.int8)
            header[parsed['glyphs'][0]] = color
            pair['input'][r:r+parsed['ph'], 1:1+parsed['kw']] = header
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('mutation accepted: '+defect)
