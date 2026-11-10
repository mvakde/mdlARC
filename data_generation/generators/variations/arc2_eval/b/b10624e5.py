"""Exact, anchor-relative prototype transfer across a four-panel cross.

The rule is shared by a puzzle. Colors and geometry are observed locally;
there is deliberately no hidden palette, reference quadrant, or scale map.
"""
import hashlib
import numpy as np


_SOURCE_INPUT_DIGESTS = {
    '100d1368169136a1a663ec38fe7767ed035bdfa163f3d9ee42d0aa76dce32d4e',
    '2955e73938427785c2a9664b8bc29ef455aaf80b483d976759454d5231d1387a',
    'ef8c50fe327a68a6bf3451def3ff50ba09c0235489648d3276302427be65a2fb',
}


def _as_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer or None')
    return int(value)


def _resolve_counts(seed, rng, num_train, num_test):
    if num_train is not None:
        num_train = _as_int(num_train, 'num_train')
        if not 2 <= num_train <= 10:
            raise ValueError('num_train must be between 2 and 10')
    if num_test is not None:
        num_test = _as_int(num_test, 'num_test')
        if not 1 <= num_test <= 4:
            raise ValueError('num_test must be between 1 and 4')
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError('num_train must exceed num_test')
    elif num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _panel_bounds(height, width, row, col):
    return ((0, row, 0, col), (0, row, col + 1, width),
            (row + 1, height, 0, col), (row + 1, height, col + 1, width))


def _signs(reference, destination):
    return (-1 if reference // 2 != destination // 2 else 1,
            -1 if reference % 2 != destination % 2 else 1)


def _grid_key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _is_source(grid):
    return grid.shape == (27, 27) and hashlib.sha256(
        grid.astype(np.uint8).tobytes()).hexdigest() in _SOURCE_INPUT_DIGESTS


def _connected(points):
    remaining = set(points)
    if not remaining:
        return False
    stack = [remaining.pop()]
    while stack:
        r, c = stack.pop()
        for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if p in remaining:
                remaining.remove(p)
                stack.append(p)
    return not remaining


def _asymmetric(motif, side):
    return all({(r if v == 1 else side-1-r,
                 c if h == 1 else side-1-c): color
                for (r, c), color in motif.items()} != motif
               for v, h in ((1, -1), (-1, 1), (-1, -1)))


def _palette(rng):
    colors = [int(x) for x in rng.permutation(10)]
    bg, divider, anchor = colors[:3]
    count = int(rng.integers(1, 7))
    pool = colors[3:]
    rng.shuffle(pool)
    details = pool[:count]
    if rng.random() < .48:
        details[int(rng.integers(count))] = divider
    return bg, divider, anchor, details


def _motif(rng, anchor, details, simple=False):
    """Grow connected colored rectangles and narrow paths on a pixel lattice."""
    a = 1 if simple else int(rng.choice([1, 1, 1, 2, 2, 3]))
    motif = {(r, c): anchor for r in range(a) for c in range(a)}
    style = 0 if simple else int(rng.integers(0, 6))
    if style == 0:
        # Simple Ls remain part of the distribution, with independently sized arms.
        length1 = a if simple else int(rng.integers(1, 2*a + 1))
        length2 = a if simple else int(rng.integers(1, 2*a + 1))
        for r in range(-length1, 0):
            for c in range(a):
                motif[r, c] = details[0]
        for r in range(a):
            for c in range(a, a + length2):
                motif[r, c] = details[-1]
    elif style == 2:
        # A variable tree: a thin trunk with spaced branches on either side.
        # Its gaps survive scaling, unlike a dense neighborhood of square tiles.
        length = int(rng.integers(4, 10))
        trunk_col = int(rng.integers(a))
        for r in range(-length, 0):
            motif[r, trunk_col] = details[0]
        levels = list(range(1, length+1, 2))
        rng.shuffle(levels)
        levels = levels[:int(rng.integers(2, len(levels)+1))]
        preferred = int(rng.choice([-1, 1]))
        for index, level in enumerate(levels):
            direction = preferred if rng.random() < .75 else -preferred
            extent = int(rng.integers(1, 6))
            for distance in range(1, extent+1):
                motif[-level, trunk_col+direction*distance] = details[(index+1) % len(details)]
    else:
        steps = int(rng.integers(2, 11))
        preferred = int(rng.integers(4))
        for step in range(steps):
            for _ in range(24):
                cells = sorted(motif)
                r, c = cells[int(rng.integers(len(cells)))]
                direction = preferred if style == 1 and rng.random() < .8 else int(rng.integers(4))
                dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[direction]
                length = int(rng.integers(1, (6 if style in (1, 4) else a+3)))
                thickness = 1 if style == 1 else int(rng.integers(1, a+2))
                offset = int(rng.integers(thickness))
                region = {(r + dr*t + (j-offset if dc else 0),
                           c + dc*t + (j-offset if dr else 0))
                          for t in range(1, length+1) for j in range(thickness)}
                new = region - set(motif)
                if not new:
                    continue
                combined = set(motif) | region
                rs, cs = zip(*combined)
                limit = 2*a+3 if style == 3 else 12
                if max(rs)-min(rs)+1 > limit or max(cs)-min(cs)+1 > limit:
                    continue
                color = details[step % len(details)] if step < len(details) else int(rng.choice(details))
                for point in region:
                    if motif.get(point) != anchor:
                        motif[point] = color
                break
    # Transform the whole source geometry, including the interior of its anchor.
    for _ in range(int(rng.integers(4))):
        motif = {(c, a-1-r): color for (r, c), color in motif.items()}
    if rng.random() < .5:
        motif = {(r, a-1-c): color for (r, c), color in motif.items()}
    return motif, a


def _raster(motif, a, unit, bg, vertical=1, horizontal=1):
    transformed = {(r if vertical == 1 else a-1-r,
                    c if horizontal == 1 else a-1-c): color
                   for (r, c), color in motif.items()}
    rs, cs = zip(*transformed)
    top, left = min(rs), min(cs)
    raster = np.full(((max(rs)-top+1)*unit, (max(cs)-left+1)*unit), bg, dtype=np.int8)
    for (r, c), color in transformed.items():
        raster[(r-top)*unit:(r-top+1)*unit, (c-left)*unit:(c-left+1)*unit] = color
    return raster, (-top*unit, -left*unit)


def _paste(panel, raster, top, left, bg):
    r0, c0 = max(0, top), max(0, left)
    r1, c1 = min(panel.shape[0], top+raster.shape[0]), min(panel.shape[1], left+raster.shape[1])
    if r0 < r1 and c0 < c1:
        tile = raster[r0-top:r1-top, c0-left:c1-left]
        target = panel[r0:r1, c0:c1]
        mask = tile != bg
        target[mask] = tile[mask]


def _placement(rng, shape, raster, offset, side, bg, clipped):
    mask = (raster != bg).astype(int)
    prefix = np.pad(mask.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    total = int(mask.sum())
    candidates = []
    for r in range(shape[0]-side+1):
        for c in range(shape[1]-side+1):
            top, left = r-offset[0], c-offset[1]
            r0, c0 = max(0, -top), max(0, -left)
            r1 = min(raster.shape[0], shape[0]-top)
            c1 = min(raster.shape[1], shape[1]-left)
            visible = int(prefix[r1, c1]-prefix[r0, c1]-prefix[r1, c0]+prefix[r0, c0])
            if (visible < total) == clipped and visible-side*side >= max(1, (total-side*side)//10):
                candidates.append((r, c))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _axis_lengths(rng, minimum, full_size):
    low = sum(minimum) + 1
    if low > 30:
        return None
    if full_size and low <= full_size:
        total = full_size
    else:
        total = int(rng.integers(low, 31))
    extra = total-low
    first = int(rng.integers(extra+1))
    return minimum[0]+first, minimum[1]+extra-first


def _build_pair(rng, reference, clip_flags, witness, ratios=None, fallback=False):
    bg, divider, anchor, details = _palette(rng)
    motif, a = _motif(rng, anchor, details, simple=fallback)
    if len(set(motif.values())) < 2 or not _connected(motif):
        return None
    if witness and not _asymmetric(motif, a):
        return None
    reference_unit = 1 if fallback else int(rng.integers(1, min(5, 6//a)+1))
    choices = [k for k in range(1, (2 if fallback else min(6//a, 5))+1)
               if ratios is None or _ratio(k, reference_unit) in ratios]
    if not choices:
        return None
    relation = int(rng.integers(5))
    selected = [k for k in choices if (k < reference_unit if relation == 0 else
                                      k > reference_unit if relation == 1 else
                                      k == reference_unit if relation == 2 else True)]
    selected = selected or choices
    destinations = [p for p in range(4) if p != reference]
    units = {p: int(rng.choice(selected)) for p in destinations}
    if witness:
        other = [k for k in choices if k != reference_unit]
        if not other or reference_unit not in choices:
            return None
        units[destinations[0]] = reference_unit
        units[destinations[1]] = int(rng.choice(other))
    units[reference] = reference_unit
    rasters = {p: _raster(motif, a, units[p], bg, *_signs(reference, p)) for p in range(4)}
    minimum_rows, minimum_cols = [3, 3], [3, 3]
    for p in range(4):
        raster, _ = rasters[p]
        side = a*units[p]
        ph, pw = (side+2, side+2) if clip_flags[p] else (max(side+2, raster.shape[0]), max(side+2, raster.shape[1]))
        minimum_rows[p//2] = max(minimum_rows[p//2], ph)
        minimum_cols[p%2] = max(minimum_cols[p%2], pw)
    roll = rng.random()
    full_size = 30 if roll < .10 else 27 if roll < .16 else 0
    heights = _axis_lengths(rng, minimum_rows, full_size)
    widths = _axis_lengths(rng, minimum_cols, full_size)
    if heights is None or widths is None:
        return None
    height, width = sum(heights)+1, sum(widths)+1
    bounds = _panel_bounds(height, width, heights[0], widths[0])
    inp = np.full((height, width), bg, dtype=np.int8)
    inp[heights[0], :] = divider
    inp[:, widths[0]] = divider
    placements = {}
    for p, (r0, r1, c0, c1) in enumerate(bounds):
        raster, offset = rasters[p]
        side = a*units[p]
        shape = (r1-r0, c1-c0)
        if p != reference and shape[0]*shape[1] <= 2*side*side:
            return None
        placement = _placement(rng, shape, raster, offset, side, bg, clip_flags[p])
        if placement is None:
            return None
        placements[p] = placement
        r, c = placement
        panel = inp[r0:r1, c0:c1]
        panel[r:r+side, c:c+side] = anchor
        if p == reference:
            _paste(panel, raster, r-offset[0], c-offset[1], bg)
    out = inp.copy()
    for p in destinations:
        r0, r1, c0, c1 = bounds[p]
        raster, offset = rasters[p]
        r, c = placements[p]
        _paste(out[r0:r1, c0:c1], raster, r-offset[0], c-offset[1], bg)
    return {'input': inp, 'output': out}


def _fallback_pair(rng, reference, flags, witness, allowed, seen):
    # A 2x2 L with units <=2 fits all constructed panels. Even a clipped
    # destination can retain one arm. Enumerating 90 role assignments guarantees
    # a fresh input against at most 13 previous pairs and three source inputs.
    pair = _build_pair(rng, reference, flags, witness, allowed, fallback=True)
    assert pair is not None
    data = _decode(pair['input'])
    old_roles = [data['background'], data['anchor']]
    old_rest = [c for c in range(10) if c not in old_roles]
    for background in range(10):
        for anchor in range(10):
            if background == anchor:
                continue
            new_roles = [background, anchor]
            new_rest = [c for c in range(10) if c not in new_roles]
            palette = np.zeros(10, dtype=np.int8)
            palette[old_roles+old_rest] = new_roles+new_rest
            candidate = {key: palette[grid] for key, grid in pair.items()}
            if _grid_key(candidate['input']) not in seen and not _is_source(candidate['input']):
                return candidate
    raise AssertionError('finite fallback palette space exhausted')


def _find_divider(grid):
    rows = [r for r in range(1, grid.shape[0]-1) if np.all(grid[r] == grid[r, 0])]
    cols = [c for c in range(1, grid.shape[1]-1) if np.all(grid[:, c] == grid[0, c])]
    candidates = [(r, c, int(grid[r, c])) for r in rows for c in cols if grid[r, 0] == grid[0, c]]
    assert len(candidates) == 1, 'expected one geometric cross'
    return candidates[0]


def _locate_square(panel, color):
    coordinates = np.argwhere(panel == color)
    assert len(coordinates), 'missing anchor'
    top, left = coordinates.min(0).tolist()
    bottom, right = (coordinates.max(0)+1).tolist()
    side = bottom-top
    assert side == right-left and len(coordinates) == side*side, 'anchor must be a single solid square'
    return top, left, side


def _decode(grid):
    row, col, divider = _find_divider(grid)
    bounds = _panel_bounds(*grid.shape, row, col)
    panels = [grid[r0:r1, c0:c1] for r0, r1, c0, c1 in bounds]
    values = np.concatenate([panel.ravel() for panel in panels])
    bg = int(np.bincount(values.astype(int), minlength=10).argmax())
    assert bg != divider
    sets = [set(map(int, np.unique(panel))) for panel in panels]
    common = set.intersection(*sets)-{bg}
    assert len(common) == 1, 'anchor identity must be common to all panels'
    anchor = next(iter(common))
    assert anchor != divider
    references = [p for p in range(4) if sets[p]-{bg, anchor}]
    assert len(references) == 1, 'expected one complete multicolored prototype'
    reference = references[0]
    for p in range(4):
        if p != reference:
            assert sets[p] == {bg, anchor}, 'destination contains unrelated detail'
    anchors = [_locate_square(panel, anchor) for panel in panels]
    r, c, side = anchors[reference]
    motif = {(int(y)-r, int(x)-c): int(panels[reference][y, x])
             for y, x in np.argwhere(panels[reference] != bg)}
    assert _connected(motif), 'reference must be connected'
    return {'background': bg, 'divider': divider, 'anchor': anchor,
            'reference': reference, 'bounds': bounds, 'panels': panels,
            'anchors': anchors, 'motif': motif, 'reference_side': side}


def _exact_scaled(data, destination_side, vertical, horizontal):
    """Independent pixel renderer: refine, then require uniform rational cells.

    Padding to anchor-side multiples lets fractional scales merge identical
    pixels, but never round a boundary or interpolate two different colors.
    """
    source_side = data['reference_side']
    motif, bg = data['motif'], data['background']
    rs, cs = zip(*motif)
    top = min(rs)//source_side*source_side
    left = min(cs)//source_side*source_side
    bottom = ((max(rs)+1+source_side-1)//source_side)*source_side
    right = ((max(cs)+1+source_side-1)//source_side)*source_side
    raw = np.full((bottom-top, right-left), bg, dtype=np.int8)
    for (r, c), color in motif.items():
        raw[r-top, c-left] = color
    expanded = np.repeat(np.repeat(raw, destination_side, axis=0), destination_side, axis=1)
    h, w = expanded.shape[0]//source_side, expanded.shape[1]//source_side
    blocks = expanded.reshape(h, source_side, w, source_side)
    low, high = blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))
    assert np.array_equal(low, high), 'nonintegral color boundary at requested scale'
    offset = [-top*destination_side//source_side, -left*destination_side//source_side]
    if vertical == -1:
        low = low[::-1, :]
        offset[0] = h-offset[0]-destination_side
    if horizontal == -1:
        low = low[:, ::-1]
        offset[1] = w-offset[1]-destination_side
    return low, tuple(offset)


def _prediction(grid, data=None):
    data = _decode(grid) if data is None else data
    expected = grid.copy()
    records = []
    for p, (r0, r1, c0, c1) in enumerate(data['bounds']):
        if p == data['reference']:
            continue
        r, c, side = data['anchors'][p]
        raster, offset = _exact_scaled(data, side, *_signs(data['reference'], p))
        top, left = r-offset[0], c-offset[1]
        # Pull destination pixels from the independent scaled raster.
        rr = np.arange(r1-r0)-top
        cc = np.arange(c1-c0)-left
        inside = ((rr[:, None] >= 0) & (rr[:, None] < raster.shape[0]) &
                  (cc[None, :] >= 0) & (cc[None, :] < raster.shape[1]))
        mapped = raster[np.clip(rr, 0, raster.shape[0]-1)[:, None],
                        np.clip(cc, 0, raster.shape[1]-1)[None, :]]
        region = expected[r0:r1, c0:c1]
        region[inside] = mapped[inside]
        total = int(np.count_nonzero(raster != data['background']))
        visible = int(np.count_nonzero(region != data['background']))
        records.append({'panel': p, 'side': side, 'total': total, 'visible': visible,
                        'lost': total-visible})
    return expected, records


def _ratio(numerator, denominator):
    divisor = int(np.gcd(numerator, denominator))
    return numerator//divisor, denominator//divisor


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    data = _decode(inp)
    expected, records = _prediction(inp, data)
    assert np.array_equal(expected, out), 'incorrect scaled/reflected/clipped completion'
    assert not np.array_equal(inp, out)
    for record in records:
        assert record['visible'] > record['side']**2, 'completion has no visible detail'
    data['copies'] = records
    data['clipped_count'] = sum(record['lost'] > 0 for record in records)
    data['ratios'] = {_ratio(record['side'], data['reference_side']) for record in records}
    data['asymmetric'] = _asymmetric(data['motif'], data['reference_side'])
    return data


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _as_int(seed, 'seed')
    if seed < 0:
        raise ValueError('seed must be non-negative')
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    # These select teaching regimes, not output-affecting hidden conventions.
    curriculum = int(rng.choice(3, p=[.25, .60, .15]))  # held-out clipping / mixed / full
    mixed_reference = rng.random() < .80
    reuse_ratios = rng.random() < .40
    references = [int(x) for x in rng.permutation(4)]
    puzzle = {'train': [], 'test': []}
    seen, training_ratios = set(), set()
    for split, count, regime in (('train', num_train, 'source_train'),
                                 ('test', num_test, 'target_test')):
        for index in range(count):
            witness = regime == 'source_train' and (index == 0 or (index == 1 and mixed_reference))
            reference = (references[index % 4] if mixed_reference and split == 'train' and index < 2 else
                         int(rng.integers(4)) if mixed_reference else references[0])
            flags = [False]*4
            if not witness:
                if curriculum == 0 and regime == 'target_test':
                    destinations = [p for p in range(4) if p != reference]
                    for p in rng.choice(destinations, size=2, replace=False):
                        flags[int(p)] = True
                elif curriculum == 1:
                    probability = float(rng.uniform(.15, .90))
                    flags = [bool(rng.random() < probability) and p != reference for p in range(4)]
            allowed = training_ratios if regime == 'target_test' and reuse_ratios else None
            pair = None
            for _ in range(128):
                candidate = _build_pair(rng, reference, flags, witness, allowed)
                if candidate is None or _is_source(candidate['input']):
                    continue
                key = _grid_key(candidate['input'])
                if key in seen:
                    continue
                # The majority background remains directly recoverable even for dense motifs.
                try:
                    analysis = _analyze_pair(candidate['input'], candidate['output'])
                except AssertionError:
                    continue
                pair = candidate
                break
            if pair is None:
                pair = _fallback_pair(rng, reference, flags, witness, allowed, seen)
                key = _grid_key(pair['input'])
                analysis = _analyze_pair(pair['input'], pair['output'])
            seen.add(key)
            puzzle[split].append(pair)
            if split == 'train':
                training_ratios.update(analysis['ratios'])
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs = set(), set()
    analyses = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            ikey, okey = _grid_key(inp), _grid_key(out)
            assert ikey not in seen_inputs and (ikey, okey) not in seen_pairs
            assert not _is_source(inp), 'exact source content'
            seen_inputs.add(ikey)
            seen_pairs.add((ikey, okey))
            analyses[split].append(_analyze_pair(inp, out))
    # A complete asymmetric scaled demonstration exposes every reflection,
    # anchor-relative translation, scale, and exact color transfer. All other
    # pairs use those same equations; pair-local roles need no learned lookup.
    witnesses = [a for a in analyses['train'] if a['asymmetric'] and a['clipped_count'] == 0
                 and any(n != d for n, d in a['ratios'])]
    assert witnesses, 'training must visibly establish reflections and scaling'
    refs = {a['reference'] for a in analyses['train']+analyses['test']}
    if len(refs) > 1:
        assert len({a['reference'] for a in witnesses}) > 1, 'moving reference requires relative-orientation demonstrations'


def review_features(pair):
    """Measurements decoded from actual returned grids, not sampled metadata."""
    a = _analyze_pair(pair['input'], pair['output'])
    motif, side = a['motif'], a['reference_side']
    rs, cs = zip(*motif)
    h, w = max(rs)-min(rs)+1, max(cs)-min(cs)+1
    boundary = []
    for (r, c), color in motif.items():
        for dr, dc, offset in ((-1, 0, r), (1, 0, r+1), (0, -1, c), (0, 1, c+1)):
            if motif.get((r+dr, c+dc), a['background']) != color:
                boundary.append(offset)
    fine = any(offset % side for offset in boundary)
    anchor_edges = [name for name, touches in (('top', min(rs) == 0), ('left', min(cs) == 0),
                                               ('bottom', max(rs) == side-1), ('right', max(cs) == side-1))
                    if touches]
    at_corner = bool(set(anchor_edges) & {'top', 'bottom'}) and bool(set(anchor_edges) & {'left', 'right'})
    relation = ('smaller' if all(n < d for n, d in a['ratios']) else
                'larger' if all(n > d for n, d in a['ratios']) else
                'equal' if a['ratios'] == {(1, 1)} else 'mixed')
    colors = set(motif.values())-{a['anchor']}
    parts = []
    for color in colors:
        remaining = {p for p, value in motif.items() if value == color}
        count = 0
        while remaining:
            count += 1
            stack = [remaining.pop()]
            while stack:
                r, c = stack.pop()
                for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if p in remaining:
                        remaining.remove(p)
                        stack.append(p)
        parts.append(count)
    quantum = int(np.gcd.reduce([side] + boundary))
    normalized = {(r//quantum, c//quantum): color for (r, c), color in motif.items()}
    palette = {a['anchor']: 1}
    signature = []
    for (r, c), color in sorted(normalized.items()):
        if color not in palette:
            palette[color] = len(palette)+1
        signature.append((r, c, palette[color]))
    occupied = np.zeros((h+2, w+2), dtype=bool)
    for r, c in motif:
        occupied[r-min(rs)+1, c-min(cs)+1] = True
    corners = (occupied[:-1, :-1].astype(int) + occupied[1:, :-1] +
               occupied[:-1, 1:] + occupied[1:, 1:])
    max_runs = max(int(np.count_nonzero(row[1:] & ~row[:-1]))
                   for array in (occupied, occupied.T) for row in array)
    lost_sides, boundary_types, corner_clips = set(), set(), 0
    for record in a['copies']:
        p = record['panel']
        raster, offset = _exact_scaled(a, record['side'], *_signs(a['reference'], p))
        coords = np.argwhere(raster != a['background'])
        y, x, _ = a['anchors'][p]
        top, left = (coords.min(0) + [y-offset[0], x-offset[1]]).tolist()
        bottom, right = (coords.max(0)+1 + [y-offset[0], x-offset[1]]).tolist()
        ph, pw = a['panels'][p].shape
        sides = {name for name, clipped in (('top', top < 0), ('bottom', bottom > ph),
                                            ('left', left < 0), ('right', right > pw)) if clipped}
        lost_sides.update(sides)
        for name in sides:
            outer = ((name == 'top' and p//2 == 0) or (name == 'bottom' and p//2 == 1) or
                     (name == 'left' and p%2 == 0) or (name == 'right' and p%2 == 1))
            boundary_types.add('outer' if outer else 'divider')
        corner_clips += bool(sides & {'top', 'bottom'}) and bool(sides & {'left', 'right'})
    return {'shape': list(pair['input'].shape), 'reference': a['reference'],
            'roles': [a['background'], a['divider'], a['anchor']],
            'detail_count': len(colors), 'divider_content': a['divider'] in colors,
            'reference_side': side, 'destination_sides': [r['side'] for r in a['copies']],
            'ratios': [list(r) for r in sorted(a['ratios'])], 'scale_regime': relation,
            'fine_geometry': bool(fine), 'motif_span': [h, w], 'motif_cells': len(motif),
            'density': len(motif)/(h*w), 'elongated': max(h, w) >= 2*min(h, w),
            'anchor_location': 'corner' if at_corner else 'edge' if anchor_edges else 'interior',
            'anchor_edges': anchor_edges,
            'repeated_color_regions': max(parts) > 1,
            'asymmetric': a['asymmetric'], 'clipped_count': a['clipped_count'],
            'lost_fractions': [r['lost']/r['total'] for r in a['copies']],
            'clipped_sides': sorted(lost_sides), 'clipped_boundaries': sorted(boundary_types),
            'corner_clips': corner_clips, 'concave_corners': int(np.count_nonzero(corners == 3)),
            'max_parallel_runs': max_runs, 'motif_structure': signature,
            'wider_than_three_anchors': max(h, w) > 3*side,
            'panel_shapes': [[r1-r0, c1-c0] for r0, r1, c0, c1 in a['bounds']]}


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p['input']), _grid_key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test() -> None:
    features, counts, puzzles = [], set(), set()
    mixed_refs = varied_sides = palettes = clipped_training = full_tests = 0
    ratios, regimes, refs, locations, shapes = set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        key = _puzzle_key(puzzle)
        puzzles.add(key)
        counts.add((len(puzzle['train']), len(puzzle['test'])))
        if seed < 3:
            assert key == _puzzle_key(generate_puzzle(seed))
        local = [review_features(p) for s in ('train', 'test') for p in puzzle[s]]
        features.extend(local)
        mixed_refs += len({f['reference'] for f in local}) > 1
        varied_sides += len({f['reference_side'] for f in local}) > 1
        palettes += len({tuple(f['roles']) for f in local}) > 1
        clipped_training += any(review_features(p)['clipped_count'] for p in puzzle['train'])
        full_tests += any(review_features(p)['clipped_count'] == 0 for p in puzzle['test'])
        for f in local:
            shapes.add(tuple(f['shape']))
            refs.add(f['reference'])
            regimes.add(f['scale_regime'])
            locations.add(f['anchor_location'])
            ratios.update(tuple(r) for r in f['ratios'])
    assert len(puzzles) == 42
    assert len({a for a, b in counts}) >= 4 and len({b for a, b in counts}) >= 3
    assert len(shapes) >= 80 and (27, 27) in shapes and (30, 30) in shapes
    assert min(min(f['shape']) for f in features) <= 12
    assert refs == {0, 1, 2, 3} and mixed_refs >= 25 and palettes == 42 and varied_sides >= 30
    assert len(ratios) >= 12 and regimes == {'smaller', 'larger', 'equal', 'mixed'}
    assert locations == {'interior', 'edge', 'corner'}
    assert {f['clipped_count'] for f in features} == {0, 1, 2, 3}
    assert clipped_training >= 8 and full_tests >= 8
    assert sum(f['fine_geometry'] for f in features) >= 60
    assert sum(f['divider_content'] for f in features) >= 70
    assert sum(f['elongated'] for f in features) >= 35
    assert sum(f['density'] < .60 for f in features) >= 25
    assert sum(f['repeated_color_regions'] for f in features) >= 30
    assert max(f['detail_count'] for f in features) >= 5
    assert max(max(f['motif_span']) for f in features) >= 15
    assert any(max(f['lost_fractions']) > .5 for f in features)
    assert any(f['fine_geometry'] and f['divider_content'] and f['clipped_count'] for f in features)
    assert {s for f in features for s in f['clipped_sides']} == {'top', 'bottom', 'left', 'right'}
    assert {s for f in features for s in f['clipped_boundaries']} == {'outer', 'divider'}
    assert sum(f['corner_clips'] for f in features) >= 20
    assert sum(f['wider_than_three_anchors'] for f in features) >= 80
    assert sum(f['max_parallel_runs'] >= 3 for f in features) >= 10
    assert len({tuple(f['motif_structure']) for f in features}) >= 180
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                          (None, 4), (2, None), (10, None), (6, 3))):
        puzzle = generate_puzzle(200+index, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle['train']) == train
        assert test is None or len(puzzle['test']) == test
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(200+index, train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(300, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    fallback_seen = set()
    for reference in range(4):
        for clipped in (False, True):
            for reused in (None, {(1, 1)}):
                pair = _fallback_pair(np.random.default_rng(51), reference,
                                      [clipped and p != reference for p in range(4)],
                                      not clipped and reused is None, reused, fallback_seen)
                fallback_seen.add(_grid_key(pair['input']))
                analysis = _analyze_pair(pair['input'], pair['output'])
                assert analysis['clipped_count'] == (3 if clipped else 0)
    # Source train 2's sub-anchor details: factors 1 and 2 are exact; 3/2 is not.
    motif = {(r, c): 2 for r in (0, 1) for c in (0, 1)}
    motif.update({(-2, 1): 3, (-1, 1): 3, (-1, 0): 8,
                  (0, -1): 9, (1, -1): 9, (0, 2): 1, (0, 3): 1, (1, 2): 1, (1, 3): 1})
    data = {'motif': motif, 'background': 4, 'reference_side': 2}
    for side in (2, 4):
        for vertical, horizontal in ((1, -1), (-1, 1), (-1, -1)):
            _exact_scaled(data, side, vertical, horizontal)
    try:
        _exact_scaled(data, 3, 1, -1)
    except AssertionError:
        pass
    else:
        raise AssertionError('ambiguous fractional scale was accepted')
    # Corruption checks establish that validation is inspecting returned pixels.
    puzzle = generate_puzzle(401, 2, 1)
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output was accepted')
