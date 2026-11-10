"""A swatch renders conditionally activated diagonals as scaled motif ribbons.

Only standard-library facilities and NumPy are used. No parent generator,
stored puzzle, RNG state, or puzzle-specific convention is used by solve.
"""
from collections import Counter
import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _array(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _squares(grid, color, odd=True):
    squares = []
    for part in _components(tuple(map(int, p)) for p in np.argwhere(grid == color)):
        rows, cols = zip(*part)
        y, x = min(rows), min(cols)
        side = max(rows)-y+1
        assert max(cols)-x+1 == side and len(part) == side*side, "not a solid square"
        assert not odd or side % 2 == 1, "shuttle center must be an integer cell"
        squares.append((y, x, side))
    return squares


def _segments(points):
    points = sorted(points)
    degree = Counter()
    segments = []
    for i, a in enumerate(points):
        for b in points[i+1:]:
            if a[0] != b[0] and abs(a[0]-b[0]) == abs(a[1]-b[1]):
                degree[a] += 1
                degree[b] += 1
                m = 1 if b[1] > a[1] else -1
                cells = tuple((a[0]+t, a[1]+m*t) for t in range(b[0]-a[0]+1))
                assert len(cells) >= 3, "degenerate bridge"
                segments.append((cells, m))
    assert segments and all(n == 1 for n in degree.values()), "ambiguous endpoint pairing"
    return segments


def _decode(input_grid):
    grid = _array(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > grid.size//2, "background must dominate the input"
    separators = []
    for r in range(grid.shape[0]):
        if grid[r, 0] != bg and np.all(grid[r] == grid[r, 0]):
            separators.append((0, r, int(grid[r, 0])))
    for c in range(grid.shape[1]):
        if grid[0, c] != bg and np.all(grid[:, c] == grid[0, c]):
            separators.append((1, c, int(grid[0, c])))
    assert len(separators) == 1, "expected one geometric separator"
    axis, pos, primary = separators[0]
    assert 0 < pos < grid.shape[axis]-1, "separator must have two nonempty regions"
    h, w = grid.shape
    bounds = ((0, pos, 0, w), (pos+1, h, 0, w)) if axis == 0 else (
        (0, h, 0, pos), (0, h, pos+1, w))
    regions = [grid[y:z, x:t] for y, z, x, t in bounds]
    colors = [set(map(int, np.unique(a))) for a in regions]
    shared = colors[0] & colors[1] - {bg, primary}
    assert len(shared) == 1, "anchor role must be visible locally"
    anchor = next(iter(shared))
    refs = [i for i, cs in enumerate(colors) if cs - {bg, primary, anchor}]
    assert len(refs) == 1, "expected one swatch region"
    reference = refs[0]
    work_index = 1-reference
    source, work = regions[reference], regions[work_index]
    assert colors[work_index] <= {bg, primary, anchor}
    source_anchors = _squares(source, anchor, odd=False)
    assert len(source_anchors) == 1
    sy, sx, unit = source_anchors[0]
    coords = np.argwhere(source != bg)
    assert tuple(coords.min(0)) == (sy, sx), "swatch anchor must be its top-left tile"
    offsets = {(int(r)-sy)//unit for r, _ in coords}
    col_offsets = {(int(c)-sx)//unit for _, c in coords}
    mh, mw = max(offsets)+1, max(col_offsets)+1
    assert min(mh, mw) >= 2
    motif = {}
    rebuilt = np.full(source.shape, bg, dtype=np.int8)
    for r in range(mh):
        for c in range(mw):
            tile = source[sy+r*unit:sy+(r+1)*unit, sx+c*unit:sx+(c+1)*unit]
            assert tile.shape == (unit, unit) and np.all(tile == tile[0, 0]), "nonintegral swatch tile"
            color = int(tile[0, 0])
            if color != bg:
                motif[r, c] = color
                rebuilt[sy+r*unit:sy+(r+1)*unit, sx+c*unit:sx+(c+1)*unit] = color
    assert np.array_equal(rebuilt, source)
    assert motif.get((0, 0)) == anchor and sum(q == anchor for q in motif.values()) == 1
    assert len(_components(motif)) == 1
    squares = _squares(work, anchor)
    assert squares, "at least one original square identifies the anchor color"
    points = [tuple(map(int, p)) for p in np.argwhere(work == primary)]
    segments = _segments(points)
    active, inactive = {}, []
    for i, (y, x, side) in enumerate(squares):
        center = (y+side//2, x+side//2)
        hits = [m for cells, m in segments if center in cells[1:-1]]
        assert len(hits) <= 1, "two launch slopes at one center"
        assert not any(center in (cells[0], cells[-1]) for cells, _ in segments)
        if hits:
            active[i] = hits[0]
        else:
            inactive.append(i)
    return dict(grid=grid, bg=bg, primary=primary, anchor=anchor,
                separator=(axis, pos), bounds=bounds, reference=reference,
                work_index=work_index, work=work, source=source, unit=unit,
                motif=motif, motif_shape=(mh, mw), period=max(mh, mw)+1,
                squares=squares, points=points, segments=segments,
                active=active, inactive=inactive)


def _bridge_base(data):
    base = data['work'].copy()
    for cells, _ in data['segments']:
        for p in cells:
            if base[p] == data['bg']:
                base[p] = data['primary']
            else:
                assert base[p] in (data['primary'], data['anchor'])
    return base


def _stamp_ink(data, emitters, *, reflections=True, perpendicular=True,
               repetitions=True, scaled=True, strict=True):
    """Union rectangular tile ink; no input pixels are mutated here."""
    h, w = data['work'].shape
    ink, records = {}, []
    for index, m in sorted(emitters.items()):
        y, x, side = data['squares'][index]
        s = side if scaled else 1
        if not scaled:
            y += side//2
            x += side//2
        direction_col = -m if perpendicular else m
        for v, u in ((1, direction_col), (-1, -direction_col)):
            max_k = max(h, w)//(data['period']*s)+2 if repetitions else 0
            for k in range(max_k+1):
                ay, ax = y+v*k*data['period']*s, x+u*k*data['period']*s
                visible = 0
                clipped = False
                for (r, c), color in data['motif'].items():
                    top = ay+(v if reflections else 1)*r*s
                    left = ax+(u if reflections else 1)*c*s
                    y0, y1 = max(0, top), min(h, top+s)
                    x0, x1 = max(0, left), min(w, left+s)
                    if y0 >= y1 or x0 >= x1:
                        continue
                    visible += (y1-y0)*(x1-x0)
                    clipped |= y1-y0 != s or x1-x0 != s
                    for yy in range(y0, y1):
                        for xx in range(x0, x1):
                            p = (yy, xx)
                            assert not strict or p not in ink or ink[p] == color, "different-color ribbon collision"
                            ink[p] = color
                if visible:
                    records.append(dict(emitter=index, k=k, direction=(v, u),
                                        visible=visible, clipped=clipped,
                                        full_anchor=0 <= ay <= h-s and 0 <= ax <= w-s))
    return ink, records


def _render(data, emitters=None, *, strict=True, **options):
    emitters = data['active'] if emitters is None else emitters
    base = _bridge_base(data)
    ink, records = _stamp_ink(data, emitters, strict=strict, **options)
    for p, color in ink.items():
        assert not strict or base[p] in (data['bg'], color), "stamp conflicts with an original marker or bridge"
        # Relaxed renderings are counterfactual rules used by validation only.
        if strict or data['work'][p] == data['bg']:
            base[p] = color
    output = data['grid'].copy()
    y, z, x, t = data['bounds'][data['work_index']]
    output[y:z, x:t] = base
    return output, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer solely from the supplied input grid."""
    return _render(_decode(input_grid))[0]


def _no_a(data):
    base = _bridge_base(data)
    h, w = base.shape
    for i, m in data['active'].items():
        y, x, s = data['squares'][i]
        cy, cx = y+s//2, x+s//2
        for r in range(h):
            c = cx-m*(r-cy)
            if 0 <= c < w and base[r, c] == data['bg']:
                base[r, c] = data['anchor']
    output = data['grid'].copy()
    y, z, x, t = data['bounds'][data['work_index']]
    output[y:z, x:t] = base
    return output


def _beyond(data):
    result = []
    for i in data['inactive']:
        y, x, s = data['squares'][i]
        cy, cx = y+s//2, x+s//2
        for cells, m in data['segments']:
            a, b = cells[0], cells[-1]
            if cx-a[1] == m*(cy-a[0]) and not a[0] <= cy <= b[0]:
                result.append((i, m))
    return result


def _analyze(inp, out):
    d = _decode(inp)
    expected, records = _render(d)
    out = _array(out)
    assert np.array_equal(out, expected), "wrong patterned diagonal completion"
    assert out.shape == d['grid'].shape and not np.array_equal(out, d['grid'])
    original = d['grid'] != d['bg']
    assert np.array_equal(out[original], d['grid'][original]), "original content changed"
    d['records'] = records
    d['beyond'] = _beyond(d)
    d['nonunit'] = any(d['squares'][i][2] != d['unit'] for i in d['active'])
    d['later_anchor'] = any(r['k'] > 0 and r['full_anchor'] for r in records)
    d['bridge_fill'] = int(np.count_nonzero((_bridge_base(d) == d['primary']) & (d['work'] == d['bg'])))
    return d


def _motif(rng, extent, anchor, details, style=None):
    """Connected swatches with independent arms, paths, branches, and colors."""
    style = int(rng.integers(4)) if style is None else style
    cells = {(0, 0)}
    if style == 0:
        short = int(rng.integers(2, extent+1))
        cells.update((0, c) for c in range(extent))
        cells.update((r, 0) for r in range(short))
        if rng.random() < .5:
            cells = {(c, r) for r, c in cells}
    elif style == 1:
        for r in range(extent):
            cells.add((r, r))
            if r:
                cells.add((r-1, r))
    elif style == 2:
        cells.update((r, 0) for r in range(extent))
        levels = rng.choice(extent, size=int(rng.integers(1, extent+1)), replace=False)
        for j, r in enumerate(levels):
            length = extent if j == 0 else int(rng.integers(2, extent+1))
            cells.update((int(r), c) for c in range(length))
    else:
        cells.update((0, c) for c in range(extent))
        cells.add((1, 0))
        target = int(rng.integers(len(cells), extent*extent+1))
        while len(cells) < target:
            frontier = sorted({p for r, c in cells for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                               if 0 <= p[0] < extent and 0 <= p[1] < extent and p not in cells})
            cells.add(frontier[int(rng.integers(len(frontier)))])
    order = sorted(cells-{(0, 0)})
    rng.shuffle(order)
    motif = {(0, 0): anchor}
    for i, p in enumerate(order):
        motif[p] = int(details[i % len(details)] if i < len(details) else rng.choice(details))
    return motif


def _assemble(rng, motif, unit, work, bg, primary, edge=None):
    mh = max(r for r, c in motif)+1
    mw = max(c for r, c in motif)+1
    edge = int(rng.integers(4)) if edge is None else edge
    horizontal = edge < 2
    thickness = (mh if horizontal else mw)*unit+2
    thickness += int(rng.integers(min(3, 30-1-thickness-(work.shape[0] if horizontal else work.shape[1]))+1))
    h = thickness if horizontal else work.shape[0]
    w = work.shape[1] if horizontal else thickness
    assert h >= mh*unit+2 and w >= mw*unit+2
    band = np.full((h, w), bg, dtype=np.int8)
    y = int(rng.integers(1, h-mh*unit))
    x = int(rng.integers(1, w-mw*unit))
    for (r, c), q in motif.items():
        band[y+r*unit:y+(r+1)*unit, x+c*unit:x+(c+1)*unit] = q
    separator = np.full((1, w) if horizontal else (h, 1), primary, dtype=np.int8)
    parts = (band, separator, work) if edge in (0, 2) else (work, separator, band)
    return np.concatenate(parts, axis=0 if horizontal else 1)


def _place_square(work, center, side, anchor, bg):
    y, x = center[0]-side//2, center[1]-side//2
    h, w = work.shape
    if not (0 <= y <= h-side and 0 <= x <= w-side):
        return False
    if not np.all(work[y:y+side, x:x+side] == bg):
        return False
    borders = (work[max(0, y-1):y, x:x+side], work[y+side:min(h, y+side+1), x:x+side],
               work[y:y+side, max(0, x-1):x], work[y:y+side, x+side:min(w, x+side+1)])
    if any(np.any(border == anchor) for border in borders):
        return False
    work[y:y+side, x:x+side] = anchor
    return True


def _build(rng, stage, shift, slope, seen, fallback=False):
    """Generate geometry directly; no composition of parent puzzle generators."""
    colors = [int(q) for q in rng.permutation(10)]
    bg, primary, anchor = colors[:3]
    detail_count = int(rng.integers(1, 5))
    details = colors[3:3+detail_count]
    if len(details) > 1 and rng.random() < .30:
        details[-1] = primary
    if stage == 0:
        extent, unit, side = 2, int(rng.choice([2, 3])), 1
        active_count = 2 if shift == 0 else 1
    elif stage == 1:
        extent, unit, side, active_count = 3, 1, 3, 1
    elif stage == 2:
        extent = 4 if shift == 1 else int(rng.choice([2, 2, 3]))
        unit = int(rng.choice([1, 1, 2]))
        side = 5 if shift == 2 else 1
        active_count = 3 if shift == 0 else 1
    else:
        extent = int(rng.choice([2, 3]))
        unit = int(rng.choice([1, 1, 2, 3]))
        side = int(rng.choice([1, 1, 1, 3]))
        active_count = int(rng.choice([0, 1, 1, 1, 2]))
        if active_count == 2:
            side = 1
    if fallback:
        unit = 2 if stage == 0 else 1
        extent = 4 if stage == 2 and shift == 1 else 3 if stage == 1 else 2
        if stage not in (0, 1, 2):
            side, active_count = 1, 1
    motif = _motif(rng, extent, anchor, details, style=0 if fallback else None)
    mh, mw = max(r for r, c in motif)+1, max(c for r, c in motif)+1
    edge = 0 if fallback else int(rng.integers(4))
    cap_h = 29-mh*unit-2 if edge < 2 else 30
    cap_w = 30 if edge < 2 else 29-mw*unit-2
    if min(cap_h, cap_w) < max(13, mw*unit+2, mh*unit+2):
        return None
    if fallback:
        h, w = min(23, cap_h), min(24, cap_w)
    else:
        low = 19 if active_count >= 2 or side >= 3 else 14
        h = int(rng.integers(min(low, cap_h), cap_h+1))
        w = int(rng.integers(min(low, cap_w), cap_w+1))
    if (edge < 2 and w < mw*unit+2) or (edge >= 2 and h < mh*unit+2):
        return None
    work = np.full((h, w), bg, dtype=np.int8)
    max_steps = min(h, w)-3
    min_steps = max(5, active_count*(extent+1)+1, side+3)
    if max_steps < min_steps:
        return None
    steps = min(15, max_steps) if fallback else int(rng.integers(min_steps, max_steps+1))
    r = 3 if fallback and h-steps >= 5 else int(rng.integers(1, h-steps-1))
    c = 3 if fallback and w-steps >= 5 else int(rng.integers(1, w-steps-1))
    if fallback and stage == 1:
        # A skewed short bracket leaves room for a later full 3-cell anchor
        # at the extent-three swatch's twelve-cell physical repeat interval.
        steps, r, c = 9, 2, 10
    if slope == -1:
        c = w-1-c
    endpoints = [(r, c), (r+steps, c+slope*steps)]
    for p in endpoints:
        work[p] = primary
    segments_count = 1
    # Two freely placed parallel bridges occasionally share the working field.
    if not fallback and stage == 3 and rng.random() < .20:
        for _ in range(15):
            length = int(rng.integers(3, min(h, w, 10)))
            yy = int(rng.integers(h-length+1))
            xx = int(rng.integers(w-length+1))
            if slope == -1:
                xx += length-1
            a, b = (yy, xx), (yy+length-1, xx+slope*(length-1))
            if any(abs(p[0]-q[0]) == abs(p[1]-q[1]) for p in (a, b) for q in endpoints):
                continue
            for p in (a, b):
                work[p] = primary
            endpoints += [a, b]
            segments_count = 2
            break
    radius = side//2
    eligible = list(range(radius+2, steps-radius-1))
    if len(eligible) < active_count:
        return None
    if fallback:
        chosen = [3+4*i for i in range(active_count)] if active_count > 1 else [steps//2]
    else:
        chosen = list(map(int, rng.choice(eligible, size=active_count, replace=False)))
    for t in chosen:
        if not _place_square(work, (r+t, c+slope*t), side, anchor, bg):
            return None
    # Collinear but beyond the bracket is a guaranteed finite-activation witness.
    idle_count = 1 if stage in (0, 1) or fallback else int(rng.integers(0, 3))
    if active_count == 0:
        idle_count = max(1, idle_count)
    if idle_count:
        outside = [(r-1, c-slope), (r+steps+1, c+slope*(steps+1))]
        rng.shuffle(outside)
        if not any(_place_square(work, p, 1, anchor, bg) for p in outside):
            return None
        for _ in range(idle_count-1):
            for p_index in rng.permutation(h*w):
                p = int(p_index)//w, int(p_index)%w
                if any(p in cells for cells, _ in _segments(endpoints)):
                    continue
                if _place_square(work, p, int(rng.choice([1, 1, 3])), anchor, bg):
                    break
            else:
                return None
    try:
        inp = _assemble(rng, motif, unit, work, bg, primary, edge)
        d = _decode(inp)
        out, _ = _render(d)
        if not fallback and stage == 3:
            # Unpaired dots are optional and must avoid every constructed pixel.
            target = int(rng.choice([0, 0, 1, 2]))
            for flat in rng.permutation(h*w):
                if target == 0:
                    break
                p = int(flat)//w, int(flat)%w
                wy, wz, wx, wt = d['bounds'][d['work_index']]
                if work[p] != bg or out[wy+p[0], wx+p[1]] != bg:
                    continue
                if any(abs(p[0]-q[0]) == abs(p[1]-q[1]) for q in d['points']):
                    continue
                inp[wy+p[0], wx+p[1]] = primary
                d = _decode(inp)
                target -= 1
            out, _ = _render(d)
        a = _analyze(inp, out)
        if stage in (0, 1):
            if len(a['active']) != active_count or not a['beyond'] or not a['bridge_fill']:
                return None
            if not any(q not in (bg, anchor, primary) for p, q in _stamp_ink(a, a['active'])[0].items()
                       if a['work'][p] == bg):
                return None
            if not a['later_anchor']:
                return None
        if stage == 2 and (len(a['active']) != active_count or not a['bridge_fill']):
            return None
        if _key(inp) in seen:
            return None
        return {'input': inp, 'output': out}
    except AssertionError:
        return None


def _make_pair(rng, stage, shift, slope, seen):
    for _ in range(256):
        pair = _build(rng, stage, shift, slope, seen)
        if pair is not None:
            return pair
    # The emergency geometry is valid independently of its colors. Enumerate
    # ninety bijections, so at most thirteen existing inputs cannot exhaust it.
    pair = _build(rng, stage, shift, slope, set(), fallback=True)
    assert pair is not None, 'emergency geometry must be valid'
    a = _decode(pair['input'])
    old_roles = [a['bg'], a['anchor']]
    old_rest = [q for q in range(10) if q not in old_roles]
    for bg in range(10):
        for anchor in range(10):
            if bg == anchor:
                continue
            new_roles = [bg, anchor]
            new_rest = [q for q in range(10) if q not in new_roles]
            mapping = np.zeros(10, dtype=np.int8)
            mapping[old_roles+old_rest] = new_roles+new_rest
            candidate = {key: mapping[grid] for key, grid in pair.items()}
            if _key(candidate['input']) not in seen:
                return candidate
    raise AssertionError('finite emergency palette space exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    num_train = _integer(num_train, 'num_train')
    num_test = _integer(num_test, 'num_test')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('require 2..10 training, 1..4 test, and num_train > num_test')
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(3))
    first_slope = int(rng.choice([-1, 1]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            stage = i if split == 'train' and i < 2 else 3 if split == 'train' else 2
            slope = first_slope if stage == 0 else -first_slope if stage == 1 else int(rng.choice([-1, 1]))
            pair = _make_pair(rng, stage, shift, slope, seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _array(pair['input']), _array(pair['output'])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            analyses[split].append(_analyze(inp, out))
    train = analyses['train']
    assert {m for a in train for m in a['active'].values()} == {-1, 1}
    assert {3, 4} <= {a['period'] for a in train}
    assert any(a['nonunit'] for a in train)
    assert any(a['later_anchor'] for a in train)
    assert any(a['bridge_fill'] for a in train)
    assert any(a['beyond'] for a in train)
    def different(a, **kw):
        return not np.array_equal(_render(a, strict=False, **kw)[0], _render(a)[0])
    assert any(not np.array_equal(_no_a(a), _render(a)[0]) for a in train), "A contributes no observed difference"
    no_b_witness = False
    for a in train:
        for i, m in a['beyond']:
            no_b_witness |= not np.array_equal(_render(a, {**a['active'], i: m}, strict=False)[0], _render(a)[0])
    assert no_b_witness, "finite activation has no observed difference"
    for option in ('reflections', 'perpendicular', 'repetitions', 'scaled'):
        assert any(different(a, **{option: False}) for a in train), f"no evidence for {option}"
    for fixed_period in (3, 4):
        assert any(not np.array_equal(_render({**a, 'period': fixed_period}, strict=False)[0],
                                     _render(a)[0]) for a in train), "fixed-period shortcut fits training"
    # Every query exhibits at least one declared shift, using actual pixels.
    max_train_extent = max(max(a['motif_shape']) for a in train)
    max_train_side = max(s for a in train for _, _, s in a['squares'])
    max_train_emitters = max(len(a['active']) for a in train)
    for a in analyses['test']:
        assert a['active'], "queries must use the motif mechanism"
        assert (max(a['motif_shape']) > max_train_extent or
                max(a['squares'][i][2] for i in a['active']) > max_train_side or
                len(a['active']) > max_train_emitters), "missing structural train/test shift"


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output']))) for p in puzzle[s])
                 for s in ('train', 'test'))


def features(pair):
    """Compact observable measurements for bounded self-tests and inspection."""
    a = _analyze(pair['input'], pair['output'])
    palette, pattern = {a['anchor']: 0}, []
    for (r, c), color in sorted(a['motif'].items()):
        pattern.append((r, c, palette.setdefault(color, len(palette))))
    used = {p for cells, _ in a['segments'] for p in (cells[0], cells[-1])}
    return dict(shape=a['grid'].shape, work_shape=a['work'].shape,
                reference=(a['separator'][0], a['reference']),
                roles=(a['bg'], a['primary'], a['anchor']), unit=a['unit'],
                motif=tuple(pattern), extent=max(a['motif_shape']), period=a['period'],
                segments=len(a['segments']), lengths=tuple(len(s) for s, _ in a['segments']),
                slopes=tuple(m for s, m in a['segments']), active=len(a['active']), idle=len(a['inactive']),
                sides=tuple(s for _, _, s in a['squares']),
                unpaired=len(set(a['points'])-used), later_anchor=a['later_anchor'],
                partial=any(r['clipped'] for r in a['records']),
                beyond=bool(a['beyond']), primary_detail=a['primary'] in a['motif'].values())


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    rows = [features(pair) for p in puzzles for split in ('train', 'test') for pair in p[split]]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({r['shape'] for r in rows}) >= 40
    assert len({r['motif'] for r in rows}) >= 25
    assert len({tuple((r, c) for r, c, q in f['motif']) for f in rows}) >= 15
    assert len({(r['work_shape'], r['lengths'], r['active'], r['sides']) for r in rows}) >= 130
    assert len({r['roles'] for r in rows}) >= 50
    assert {r['reference'] for r in rows} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert {r['period'] for r in rows} >= {3, 4, 5}
    assert {s for r in rows for s in r['sides']} >= {1, 3, 5}
    assert {r['active'] for r in rows} >= {0, 1, 2, 3}
    assert any(r['segments'] > 1 for r in rows)
    assert any(r['unpaired'] for r in rows)
    assert any(r['primary_detail'] for r in rows)
    assert any(r['partial'] for r in rows)
    assert sum(r['later_anchor'] for r in rows) >= 70
    for seed in (0, 21, 41):
        assert _puzzle_key(puzzles[seed]) == _puzzle_key(generate_puzzle(seed))
    for a, b in ((2, 1), (3, 2), (5, 4), (10, 1), (10, 4)):
        p = generate_puzzle(100+a*10+b, a, b)
        assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Solver and validator must accept the documented nested-list schema too.
    listed = {s: [{k: v.tolist() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    validate_generated_puzzle(listed)
    # Input-only geometry, not generator state, supports a side-seven shuttle.
    rng = np.random.default_rng(501)
    work = np.zeros((23, 24), dtype=np.int8)
    work[2, 2] = work[20, 20] = 1
    work[8:15, 8:15] = 2
    inp = _assemble(rng, {(0, 0): 2, (0, 1): 3, (1, 0): 4}, 1, work, 0, 1, 0)
    d = _decode(inp)
    assert d['squares'][0][2] == 7 and len(d['active']) == 1
    _analyze(inp, solve(inp))
    # A square's corner touches the bridge, but its center does not activate.
    edge_work = np.zeros((23, 24), dtype=np.int8)
    edge_work[2, 2] = edge_work[19, 19] = 1
    edge_work[9:12, 11:14] = 2
    edge_inp = _assemble(rng, {(0, 0): 2, (0, 1): 3, (1, 0): 4}, 1, edge_work, 0, 1, 0)
    edge_data = _decode(edge_inp)
    assert not edge_data['active'] and edge_data['work'][11, 11] == 2
    edge_out = solve(edge_inp)
    _analyze(edge_inp, edge_out)
    y, z, x, t = edge_data['bounds'][edge_data['work_index']]
    assert not np.any(np.isin(edge_out[y:z, x:t], [3, 4]))
    # An original idle square can receive same-color ink from a ribbon; it does
    # not become a second pivot with its own oppositely facing pair of motifs.
    inert_work = np.zeros((23, 24), dtype=np.int8)
    inert_work[2, 2] = inert_work[20, 20] = 1
    inert_work[10, 10] = inert_work[13, 7] = 2
    inert_inp = _assemble(rng, {(0, 0): 2, (0, 1): 3, (1, 0): 4}, 1, inert_work, 0, 1, 0)
    inert = _decode(inert_inp)
    assert len(inert['active']) == 1 and len(inert['inactive']) == 1
    idle = inert['inactive'][0]
    ink, _ = _stamp_ink(inert, inert['active'])
    assert ink[13, 7] == 2
    assert not np.array_equal(solve(inert_inp), _render(inert, {**inert['active'], idle: 1}, strict=False)[0])
    _analyze(inert_inp, solve(inert_inp))
    # The input-only solver accepts a compatible two-slope primary crossing.
    crossing_work = np.zeros((12, 12), dtype=np.int8)
    for p in ((1, 1), (8, 8), (2, 10), (9, 3)):
        crossing_work[p] = 1
    crossing_work[0, 6] = 2
    crossing = _assemble(rng, {(0, 0): 2, (0, 1): 3, (1, 0): 4}, 1, crossing_work, 0, 1, 0)
    assert {m for cells, m in _decode(crossing)['segments']} == {-1, 1}
    _analyze(crossing, solve(crossing))
    def rejects(grid):
        try:
            solve(grid)
        except AssertionError:
            return
        raise AssertionError('invalid family geometry was accepted')
    ambiguous_work = crossing_work.copy()
    ambiguous_work[6, 6] = 2
    ambiguous = _assemble(rng, {(0, 0): 2, (0, 1): 3, (1, 0): 4}, 1, ambiguous_work, 0, 1, 0)
    rejects(ambiguous)  # the center lies on two primary slopes
    conflict_work = np.zeros((12, 12), dtype=np.int8)
    for p in ((1, 1), (8, 8), (1, 3), (6, 8)):
        conflict_work[p] = 1
    conflict_work[4, 4] = 2
    conflict = _assemble(rng, {(0, 0): 2, (0, 1): 3, (1, 0): 4, (1, 1): 4},
                         1, conflict_work, 0, 1, 0)
    rejects(conflict)  # reflected detail at (3,5) conflicts with a bridge
    # Force the emergency geometry in every teaching and query regime.
    seen = set()
    for slope in (-1, 1):
        for shift in range(3):
            for stage in (0, 1, 2, 3):
                pair = _build(np.random.default_rng(900+shift*4+stage), stage, shift, slope, seen, fallback=True)
                assert pair is not None
                seen.add(_key(pair['input']))
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('42-seed self-test passed')
