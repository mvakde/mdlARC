"""Anchor-normalized, reversible edit analogies. Standard library + NumPy only.

No parent code, stored puzzles, seed lookup, or generation metadata is used by
solve(). Ground truth is painted from logical masks; solve pulls input pixels
through exact anchor coordinates and computes a symmetric difference.
"""
from __future__ import annotations

from math import lcm
import numpy as np


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid dimensions"
    assert np.issubdtype(g.dtype, np.integer), "noninteger cells"
    assert np.all((0 <= g) & (g <= 9)), "invalid colors"
    return g.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _bounds(h, w, r, c):
    return ((0, r, 0, c), (0, r, c+1, w),
            (r+1, h, 0, c), (r+1, h, c+1, w))


def _square(panel, color):
    points = np.argwhere(panel == color)
    assert len(points), "absent anchor"
    r, c = points.min(0).tolist()
    y, x = (points.max(0)+1).tolist()
    assert y-r == x-c and len(points) == (y-r)**2, "anchor is not a solid square"
    assert 1 <= y-r <= 6, "anchor side outside family"
    return r, c, y-r


def _decode(grid):
    """Read roles from geometry, never from a literal color dictionary."""
    g = _grid(grid)
    bg = int(g[0, 0])
    assert all(int(g[r, c]) == bg for r, c in ((0, -1), (-1, 0), (-1, -1))), "corner background"
    rows = [r for r in range(1, g.shape[0]-1)
            if int(g[r, 0]) != bg and np.all(g[r] == g[r, 0])]
    cols = [c for c in range(1, g.shape[1]-1)
            if int(g[0, c]) != bg and np.all(g[:, c] == g[0, c])]
    crosses = [(r, c) for r in rows for c in cols if g[r, 0] == g[0, c]]
    assert len(crosses) == 1, "expected one cross divider"
    r, c = crosses[0]
    divider = int(g[r, c])
    bounds = _bounds(*g.shape, r, c)
    panels = [g[y:z, x:t] for y, z, x, t in bounds]
    sets = [set(map(int, np.unique(p)))-{bg} for p in panels]
    blanks = [p for p in range(4) if len(sets[p]) == 1]
    assert len(blanks) == 1, "expected one anchor-only panel"
    missing = blanks[0]
    qa = next(iter(sets[missing]))
    known = [p for p in range(4) if p != missing and qa in sets[p]]
    assert len(known) == 1, "query anchors must pair uniquely"
    query = known[0]
    refs = [p for p in range(4) if p not in (missing, query)]
    assert all(len(sets[p]) == 2 for p in refs+[query]), "each known panel needs one ink"
    candidates = []
    for color in sorted(sets[refs[0]] & sets[refs[1]]):
        try:
            for p in refs:
                _square(panels[p], color)
        except AssertionError:
            continue
        candidates.append(color)
    assert len(candidates) == 1, "ambiguous reference anchor identity"
    ra = candidates[0]
    ri = next(iter(sets[refs[0]]-{ra}))
    assert sets[refs[1]] == {ra, ri}, "reference ink identity"
    qi = next(iter(sets[query]-{qa}))
    assert len({bg, divider, ra, qa, ri, qi}) == 6, "six distinct local roles required"
    anchors = [_square(panel, qa if p in (missing, query) else ra)
               for p, panel in enumerate(panels)]
    assert all(0 < y and 0 < x and y+s < panels[p].shape[0]
               and x+s < panels[p].shape[1] for p, (y, x, s) in enumerate(anchors)), "anchors need a margin"
    inks = {refs[0]: ri, refs[1]: ri, query: qi}
    return dict(grid=g, bg=bg, divider=divider, bounds=bounds, panels=panels,
                anchors=anchors, refs=refs, query=query, missing=missing,
                ra=ra, qa=qa, ri=ri, qi=qi, inks=inks)


def _sample_fields(data, *, scaled=True, reflected=True):
    """Pull all three visible patterns onto the missing panel's exact microgrid.

    Anchor side L is a common integer resolution, not a rounding convention.
    A bottom/right panel is reflected about its anchor's center. Sampling at
    microcells retains sub-anchor geometry and allows cancellation before the
    final exact raster check.
    """
    dest = data['missing']
    ar, ac, side = data['anchors'][dest]
    height, width = data['panels'][dest].shape
    resolution = lcm(*(s for _, _, s in data['anchors'])) if scaled else 1
    factor = resolution//side if scaled else 1
    flip_width = resolution if scaled else side
    yy = np.arange(height*factor)-ar*factor
    xx = np.arange(width*factor)-ac*factor
    if reflected and dest//2:
        yy = flip_width-1-yy
    if reflected and dest%2:
        xx = flip_width-1-xx
    fields = {}
    for p in data['refs']+[data['query']]:
        r, c, s = data['anchors'][p]
        step = resolution//s if scaled else 1
        span = resolution if scaled else s
        yr = span-1-yy if reflected and p//2 else yy
        xc = span-1-xx if reflected and p%2 else xx
        iy, ix = r+yr//step, c+xc//step
        panel = data['panels'][p]
        valid = ((iy[:, None] >= 0) & (iy[:, None] < panel.shape[0])
                 & (ix[None, :] >= 0) & (ix[None, :] < panel.shape[1]))
        sampled = panel[np.clip(iy, 0, panel.shape[0]-1)[:, None],
                        np.clip(ix, 0, panel.shape[1]-1)[None, :]]
        fields[p] = valid & (sampled == data['inks'][p])
    return fields, factor


def _prediction(data, *, scaled=True, reflected=True, mode='xor'):
    fields, factor = _sample_fields(data, scaled=scaled, reflected=reflected)
    a, b = (fields[p] for p in data['refs'])
    t = fields[data['query']]
    if mode == 'xor':
        mask = t ^ a ^ b
    elif mode == 'copy_query':
        mask = t
    elif mode == 'union':
        mask = t | a | b
    elif mode == 'add_only':
        mask = t | (a ^ b)
    elif mode == 'copy_reference':
        mask = b
    else:
        raise ValueError('unknown diagnostic mode')
    shape = data['panels'][data['missing']].shape
    blocks = mask.reshape(shape[0], factor, shape[1], factor)
    low, high = blocks.all(axis=(1, 3)), blocks.any(axis=(1, 3))
    assert np.array_equal(low, high), "answer has a nonintegral color boundary"
    result = np.where(low, data['qi'], data['bg']).astype(np.int8)
    r, c, s = data['anchors'][data['missing']]
    if scaled and reflected:
        assert not low[r:r+s, c:c+s].any(), "edit overlaps protected anchor"
    result[r:r+s, c:c+s] = data['qa']
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return only the reconstructed missing panel, excluding the cross."""
    return _prediction(_decode(input_grid))


def _motif(rng, shape, protected):
    """Procedural silhouettes: growth, strokes, regions, rings and symmetry."""
    h, w = shape
    rr, cc = np.indices(shape)
    mode = int(rng.integers(6))
    mask = np.zeros(shape, dtype=bool)
    free = list(map(tuple, np.argwhere(~protected)))
    if mode in (0, 5):
        r, c = free[int(rng.integers(len(free)))]
        mask[r, c] = True
        count = int(rng.integers(3, max(4, len(free)*3//4)))
        for _ in range(count-1):
            frontier = set()
            for y, x in np.argwhere(mask):
                for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if 0 <= p[0] < h and 0 <= p[1] < w and not protected[p] and not mask[p]:
                        frontier.add(p)
            if not frontier:
                break
            choices = sorted(frontier)
            mask[choices[int(rng.integers(len(choices)))]] = True
        if mode == 5:
            mask |= mask[::-1] if rng.integers(2) else mask[:, ::-1]
    elif mode == 1:
        for _ in range(int(rng.integers(1, 4))):
            y, z = sorted(map(int, rng.choice(h+1, 2, replace=False)))
            x, t = sorted(map(int, rng.choice(w+1, 2, replace=False)))
            mask[y:z, x:t] = True
    elif mode == 2:
        y, x = int(rng.integers(h)), int(rng.integers(w))
        for _ in range(int(rng.integers(3, 8))):
            dy, dx = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
            for _ in range(int(rng.integers(1, max(h, w)))):
                mask[y, x] = True
                y, x = int(np.clip(y+dy, 0, h-1)), int(np.clip(x+dx, 0, w-1))
    elif mode == 3:
        d = abs(rr-int(rng.integers(h)))+abs(cc-int(rng.integers(w)))
        radius = int(rng.integers(1, max(h, w)))
        mask = d <= radius if rng.integers(2) else (d == radius) | (d == radius-1)
    else:
        y, x = int(rng.integers(h)), int(rng.integers(w))
        mask[y, :int(rng.integers(1, w+1))] = True
        mask[:int(rng.integers(1, h+1)), x] = True
        if rng.integers(2):
            mask[-1-y] |= mask[y]
    mask[protected] = False
    return mask


def _logical(rng):
    a = int(rng.choice([1, 1, 2]))
    h, w = map(int, rng.integers(a+2, 8, size=2))
    ar, ac = int(rng.integers(h-a+1)), int(rng.integers(w-a+1))
    protected = np.zeros((h, w), dtype=bool)
    protected[ar:ar+a, ac:ac+a] = True
    for _ in range(48):
        delta = _motif(rng, (h, w), protected)
        changed = list(map(tuple, np.argwhere(delta)))
        unchanged = list(map(tuple, np.argwhere(~delta & ~protected)))
        if len(changed) < 2 or len(unchanged) < 2:
            continue
        base = _motif(rng, (h, w), protected)
        query = _motif(rng, (h, w), protected)
        d0, d1 = [changed[int(i)] for i in rng.choice(len(changed), 2, replace=False)]
        e0, e1 = [unchanged[int(i)] for i in rng.choice(len(unchanged), 2, replace=False)]
        # Explicit diagnostic evidence: an erasure, an addition, a reference
        # decoy to exclude, and unrelated query content to carry through.
        base[d0], base[d1], base[e0], base[e1] = True, False, True, False
        query[d0], query[d1], query[e0], query[e1] = True, False, False, True
        return base, base ^ delta, query, query ^ delta, (ar, ac, a)
    return None


def _orient(array, quadrant, unit):
    out = array[::-1] if quadrant//2 else array
    out = out[:, ::-1] if quadrant%2 else out
    return np.repeat(np.repeat(out, unit, axis=0), unit, axis=1)


def _anchor_offset(shape, anchor, quadrant, unit):
    r, c, a = anchor
    if quadrant//2:
        r = shape[0]-r-a
    if quadrant%2:
        c = shape[1]-c-a
    return r*unit, c*unit, a*unit


def _paste_mask(panel, mask, top, left, color):
    """Construction-only painting, including geometric clipping."""
    r0, r1 = max(0, top), min(panel.shape[0], top+mask.shape[0])
    c0, c1 = max(0, left), min(panel.shape[1], left+mask.shape[1])
    if r0 < r1 and c0 < c1:
        part = mask[r0-top:r1-top, c0-left:c1-left]
        region = panel[r0:r1, c0:c1]
        region[part] = color


def _visible_count(points, top, left, shape):
    return int(np.count_nonzero((points[:, 0]+top >= 0) & (points[:, 0]+top < shape[0])
                               & (points[:, 1]+left >= 0) & (points[:, 1]+left < shape[1])))


def _missing_position(rng, shape, offset, masks, clipped):
    a, b, t, result = masks
    delta = a ^ b
    witnesses = [delta & t, delta & ~t, t & ~a & ~b, a & b & ~t]
    point_sets = [np.argwhere(m) for m in [result, t, delta]+witnesses]
    y, x, side = offset
    candidates = []
    for r in range(1, shape[0]-side):
        for c in range(1, shape[1]-side):
            top, left = r-y, c-x
            counts = [_visible_count(p, top, left, shape) for p in point_sets]
            lost = counts[0] < len(point_sets[0])
            if lost != clipped or min(counts[3:]) == 0:
                continue
            if not clipped and any(counts[i] != len(point_sets[i]) for i in range(3)):
                continue
            candidates.append((r, c))
    return None if not candidates else candidates[int(rng.integers(len(candidates)))]


_PARTITIONS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))


def _build_pair(rng, partition, clipped, fractional_reference=False):
    logical = _logical(rng)
    if logical is None:
        return None
    a, b, t, result, anchor = logical
    h, w = a.shape
    groups = _PARTITIONS[partition]
    group = int(rng.integers(2))
    refs, target = list(groups[group]), list(groups[1-group])
    rng.shuffle(refs)
    rng.shuffle(target)
    query, missing = target
    units = [int(x) for x in rng.integers(1, 4, size=4)]
    ref_units = [2, 3] if fractional_reference else [1, int(rng.choice([2, 3]))]
    rng.shuffle(ref_units)
    units[refs[0]], units[refs[1]] = ref_units
    minimum_y, minimum_x = [3, 3], [3, 3]
    for p in range(4):
        mh = h*units[p]+2 if p != missing else anchor[2]*units[p]+4
        mw = w*units[p]+2 if p != missing else anchor[2]*units[p]+4
        minimum_y[p//2] = max(minimum_y[p//2], mh)
        minimum_x[p%2] = max(minimum_x[p%2], mw)
    if sum(minimum_y)+1 > 30 or sum(minimum_x)+1 > 30:
        return None
    lengths = []
    for minima in (minimum_y, minimum_x):
        extra = int(rng.integers(0, 31-(sum(minima)+1)))
        first = int(rng.integers(extra+1))
        lengths.append((minima[0]+first, minima[1]+extra-first))
    heights, widths = lengths
    shape = sum(heights)+1, sum(widths)+1
    bounds = _bounds(*shape, heights[0], widths[0])
    bg, divider, ra, qa, ri, qi = map(int, rng.permutation(10)[:6])
    inp = np.full(shape, bg, dtype=np.int8)
    inp[heights[0]] = divider
    inp[:, widths[0]] = divider
    assignment = {refs[0]: a, refs[1]: b, query: t}
    placements = {}
    for p, (r0, r1, c0, c1) in enumerate(bounds):
        panel = inp[r0:r1, c0:c1]
        offset = _anchor_offset((h, w), anchor, p, units[p])
        ay, ax, side = offset
        if p == missing:
            masks = [_orient(m, p, units[p]) for m in (a, b, t, result)]
            position = _missing_position(rng, panel.shape, offset, masks, clipped)
            if position is None:
                return None
            ar, ac = position
        else:
            mask = _orient(assignment[p], p, units[p])
            top = int(rng.integers(1, panel.shape[0]-mask.shape[0]))
            left = int(rng.integers(1, panel.shape[1]-mask.shape[1]))
            _paste_mask(panel, mask, top, left, ri if p in refs else qi)
            ar, ac = top+ay, left+ax
        panel[ar:ar+side, ac:ac+side] = ra if p in refs else qa
        placements[p] = (ar, ac)
    r0, r1, c0, c1 = bounds[missing]
    out = inp[r0:r1, c0:c1].copy()
    ay, ax, _ = _anchor_offset((h, w), anchor, missing, units[missing])
    ar, ac = placements[missing]
    _paste_mask(out, _orient(result, missing, units[missing]), ar-ay, ac-ax, qi)
    return {'input': inp, 'output': out}


def _analyze(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    data = _decode(inp)
    expected = _prediction(data)
    assert out.shape == data['panels'][data['missing']].shape, "wrong crop"
    assert np.array_equal(out, expected), "incorrect edit transfer"
    fields, factor = _sample_fields(data)
    a, b = (fields[p] for p in data['refs'])
    t = fields[data['query']]
    delta = a ^ b
    assert np.any(a & ~b) and np.any(b & ~a), "reference needs reversible changes"
    assert np.any(delta & t) and np.any(delta & ~t), "both erasure and addition must be visible"
    assert np.any(t & ~a & ~b), "unrelated query-only content must survive"
    assert np.any(a & b & ~t), "unchanged reference content must be excluded"
    assert not np.array_equal(out, _prediction(data, mode='copy_query')), "reference unused"
    assert not np.array_equal(out, _prediction(data, mode='add_only')), "union shortcut"
    assert not np.array_equal(out, _prediction(data, mode='union')), "reference decoy copied"
    assert not np.array_equal(out, _prediction(data, mode='copy_reference')), "query content ignored"
    # Every returned pair, including clipped cases, witnesses scale and reflection.
    assert not np.array_equal(out, _prediction(data, reflected=False)), "reflection unused"
    assert not np.array_equal(out, _prediction(data, scaled=False)), "scale unused"
    return data


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            partition = (index if index < 2 else int(rng.integers(2))) if split == 'train' else 2
            # Two demonstrations suffice: one complete horizontal analogy and
            # one vertical analogy with clipping. Tests recombine diagonally.
            clipped = (index == 1 or (index > 1 and bool(rng.integers(2)))) if split == 'train' else bool(rng.integers(2))
            for _ in range(640):
                pair = _build_pair(rng, partition, clipped, fractional_reference=(split == 'test'))
                if pair is None or _key(pair['input']) in seen:
                    continue
                try:
                    _analyze(pair)
                except AssertionError:
                    continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError('bounded construction exhausted')
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    train_partitions = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            data = _analyze(pair)
            key = _key(data['grid'])
            assert key not in seen, "duplicate input"
            seen.add(key)
            partition = next(i for i, groups in enumerate(_PARTITIONS)
                             if tuple(sorted(data['refs'])) in groups)
            if split == 'train':
                train_partitions.add(partition)
    assert {0, 1} <= train_partitions, "training needs horizontal and vertical evidence"


def _signature(puzzle):
    return tuple(tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


def _canonical_signature(data):
    """Scale-free exact stencil signature, for diversity diagnostics only."""
    # Use exact occupied-area samples on a fixed half-anchor lattice. The
    # generator's finest canonical cell is one-half anchor wide, so this is
    # lossless for the generated stencils, including uniform enlargements.
    cells = []
    for p in data['refs']:
        r, c, s = data['anchors'][p]
        panel = data['panels'][p]
        points = set()
        for y in range(-16, 18):
            for x in range(-16, 18):
                # half-unit cell centers, mapped with integer arithmetic
                yy, xx = ((2*y+1)*s)//4, ((2*x+1)*s)//4
                yy = s-1-yy if p//2 else yy
                xx = s-1-xx if p%2 else xx
                yy, xx = r+yy, c+xx
                if 0 <= yy < panel.shape[0] and 0 <= xx < panel.shape[1] and panel[yy, xx] == data['ri']:
                    points.add((y, x))
        cells.append(points)
    return tuple(sorted(cells[0] ^ cells[1]))


def _has_fine_detail(data):
    for p in data['refs']+[data['query']]:
        r, c, side = data['anchors'][p]
        mask = data['panels'][p] == data['inks'][p]
        for y, x in np.argwhere(mask):
            for dy, dx, boundary in ((-1, 0, y-r), (1, 0, y+1-r),
                                     (0, -1, x-c), (0, 1, x+1-c)):
                yy, xx = y+dy, x+dx
                neighbor = 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx]
                if not neighbor and boundary % side:
                    return True
    return False


def self_test():
    """Bounded, in-memory 42-seed structural and semantic diagnostics."""
    shapes, outputs, sides, stencils, positions, palettes = set(), set(), set(), set(), set(), set()
    signatures, fine, clipped = set(), 0, 0
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        if seed < 3:
            assert _signature(puzzle) == _signature(generate_puzzle(seed)), "nondeterministic seed"
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                data = _decode(pair['input'])
                shapes.add(data['grid'].shape)
                outputs.add(pair['output'].shape)
                sides.update(s for _, _, s in data['anchors'])
                positions.add((data['missing'], data['query'], tuple(data['refs'])))
                palettes.add((data['bg'], data['divider'], data['ra'], data['qa'], data['ri'], data['qi']))
                signature = _canonical_signature(data)
                stencils.add(signature)
                fine += _has_fine_detail(data)
                # The full canonical answer's area can be compared with the
                # cropped output by counting normalized source areas.
                clipped += _is_clipped(data, pair['output'])
                pairs += 1
    assert len(signatures) == 42 and len(shapes) >= 30 and len(outputs) >= 20
    assert sides == {1, 2, 3, 4, 6} and len(stencils) >= 100
    assert len(positions) == 12 and len(palettes) >= 150 and fine > 0 and clipped >= 42
    for n in range(2, 11):
        for k in range(1, min(4, n-1)+1):
            validate_generated_puzzle(generate_puzzle(1000+n*10+k, n, k))
    for n, k in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, k)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(8, 2, 1)
    broken = {s: [{role: g.copy() for role, g in pair.items()} for pair in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    _counterfactual_test()
    return dict(seeds=42, pairs=pairs, input_shapes=len(shapes), output_shapes=len(outputs),
                stencils=len(stencils), role_assignments=len(palettes),
                anchor_sides=sorted(sides), clipped_pairs=clipped, fine_detail_pairs=fine)


def _full_fields(data):
    """Canonical masks on a bounded common lattice; used only by diagnostics."""
    resolution = lcm(*(s for _, _, s in data['anchors']))
    rectangles = {}
    for p in data['refs']+[data['query']]:
        r, c, s = data['anchors'][p]
        step = resolution//s
        rects = []
        for y, x in np.argwhere(data['panels'][p] == data['inks'][p]):
            yy, xx = (int(y)-r)*step, (int(x)-c)*step
            if p//2:
                yy = resolution-yy-step
            if p%2:
                xx = resolution-xx-step
            rects.append((yy, xx, step))
        rectangles[p] = rects
    all_rects = [v for values in rectangles.values() for v in values]
    top, left = min(v[0] for v in all_rects), min(v[1] for v in all_rects)
    bottom = max(y+k for y, _, k in all_rects)
    right = max(x+k for _, x, k in all_rects)
    fields = {}
    for p, rects in rectangles.items():
        field = np.zeros((bottom-top, right-left), dtype=bool)
        for y, x, k in rects:
            field[y-top:y-top+k, x-left:x-left+k] = True
        fields[p] = field
    return fields, resolution


def _is_clipped(data, output):
    fields, resolution = _full_fields(data)
    a, b = (fields[p] for p in data['refs'])
    answer = a ^ b ^ fields[data['query']]
    side = data['anchors'][data['missing']][2]
    return int(answer.sum())*side*side > int(np.count_nonzero(output == data['qi']))*resolution*resolution


def _counterfactual_test():
    """Same query, different reference evidence; slot order has no semantics."""
    pair = generate_puzzle(23, 2, 1)['train'][0]
    data = _decode(pair['input'])
    base = solve(pair['input'])
    # Swap complete reference states while retaining each panel's scale and
    # reflection. This is a content swap, not a raw-pixel rectangle swap.
    swapped = pair['input'].copy()
    for p, other in zip(data['refs'], data['refs'][::-1]):
        fields, factor = _sample_fields(dict(data, missing=p))
        shape = data['panels'][p].shape
        blocks = fields[other].reshape(shape[0], factor, shape[1], factor)
        mask = blocks.any(axis=(1, 3))
        assert np.array_equal(mask, blocks.all(axis=(1, 3)))
        r0, r1, c0, c1 = data['bounds'][p]
        panel = swapped[r0:r1, c0:c1]
        panel[panel == data['ri']] = data['bg']
        panel[mask] = data['ri']
    assert np.array_equal(solve(swapped), base), "reference order changed the answer"
    # A full scene twin changes only one informative reference block.
    changed = False
    for p in data['refs']:
        panel = data['panels'][p]
        r0, r1, c0, c1 = data['bounds'][p]
        ar, ac, s = data['anchors'][p]
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)):
            y, x = ar+dy*s, ac+dx*s
            if not (0 < y and y+s < panel.shape[0] and 0 < x and x+s < panel.shape[1]):
                continue
            block = panel[y:y+s, x:x+s]
            if not (np.all(block == data['bg']) or np.all(block == data['ri'])):
                continue
            twin = pair['input'].copy()
            twin[r0+y:r0+y+s, c0+x:c0+x+s] = data['ri'] if np.all(block == data['bg']) else data['bg']
            try:
                other = solve(twin)
            except AssertionError:
                continue
            if not np.array_equal(base, other):
                assert np.array_equal(twin[data['bounds'][data['query']][0]:data['bounds'][data['query']][1],
                                           data['bounds'][data['query']][2]:data['bounds'][data['query']][3]],
                                      data['panels'][data['query']])
                changed = True
                break
        if changed:
            break
    assert changed, "reference-dependence twin unavailable"


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
