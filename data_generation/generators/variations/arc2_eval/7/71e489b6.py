"""Rectilinear error cleanup with asymmetric, merging eight-neighbor halos.

Construction uses rectangle compositions. Reconstruction sees only the returned
pixels: long signed edges on both axes vote for rectangular partitions. Color
conventions are inferred from training, never from a region's size or location.
"""
import numpy as np
from functools import lru_cache

def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None:
        if (
            isinstance(num_train, (bool, np.bool_))
            or not isinstance(num_train, (int, np.integer))
        ):
            raise ValueError("num_train must be an integer or None")
        num_train = int(num_train)
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")

    if num_test is not None:
        if (
            isinstance(num_test, (bool, np.bool_))
            or not isinstance(num_test, (int, np.integer))
        ):
            raise ValueError("num_test must be an integer or None")
        num_test = int(num_test)
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        maximum = min(4, num_train - 1)
        if maximum < 1:
            raise ValueError("no legal num_test exists")
        num_test = int(rng.integers(1, maximum + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _signature(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _dilate(a):
    h, w = a.shape
    p = np.pad(a, 1)
    return np.logical_or.reduce([p[y:y+h, x:x+w]
                                 for y in range(3) for x in range(3)])


def _components(a, diagonal=False):
    remaining = set(map(tuple, np.argwhere(a)))
    result = []
    offsets = [(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)
               if (y or x) and (diagonal or abs(y)+abs(x) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, points = [start], [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in offsets:
                q = (y+dy, x+dx)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    points.append(q)
        mask = np.zeros_like(a)
        yy, xx = zip(*points)
        mask[yy, xx] = True
        result.append(mask)
    return result


def _turns(a):
    # Symmetric in the two input colors, including at the canvas boundary.
    total = 0
    for b in (a, ~a):
        p = np.pad(b.astype(np.int16), 1)
        total += np.abs(p[1:, 1:] - p[1:, :-1]
                        - p[:-1, 1:] + p[:-1, :-1]).sum()
    return int(total)//2


def _edge_cuts(a, length):
    delta = np.diff(a.astype(np.int8), axis=0)
    support = np.zeros(a.shape[0]-1, dtype=bool)
    for x in range(a.shape[1]-length+1):
        window = delta[:, x:x+length]
        support |= (window == 1).all(axis=1) | (window == -1).all(axis=1)
    return [0] + (np.flatnonzero(support)+1).tolist() + [a.shape[0]]


def _partition_vote(a, yy, xx):
    sums = np.add.reduceat(np.add.reduceat(a.astype(np.int16), yy[:-1], axis=0),
                          xx[:-1], axis=1)
    sizes = np.outer(np.diff(yy), np.diff(xx))
    if np.any(2*sums == sizes):
        return None
    tiles = 2*sums > sizes
    return np.repeat(np.repeat(tiles, np.diff(yy), axis=0), np.diff(xx), axis=1)


def _canonical_cuts(a):
    return ([0] + (np.flatnonzero(np.any(a[1:] != a[:-1], axis=1))+1).tolist()
            + [a.shape[0]])


@lru_cache(maxsize=4096)
def _recover_cached(shape, pixels):
    observed = np.frombuffer(pixels, dtype=np.uint8).reshape(shape).astype(bool)
    candidates = {}
    row_cuts = [_edge_cuts(observed, k) for k in (2, 3, 4, 5)]
    col_cuts = [_edge_cuts(observed.T, k) for k in (2, 3, 4, 5)]
    for yy in row_cuts:
        for xx in col_cuts:
            b = _partition_vote(observed, yy, xx)
            if b is not None:
                candidates[b.tobytes()] = b
    if not candidates:
        raise ValueError('tied rectangular partition')

    def score(b, penalty=1.5):
        return int(np.count_nonzero(b != observed)) + penalty*_turns(b)

    ordered = sorted(candidates.values(), key=score)
    best = ordered[0]
    if not best.any() or best.all():
        raise ValueError('scaffold must retain both colors')
    # Stability under a range of simplicity penalties prevents tiny cavities
    # from being silently preferred over sparse clusters, or vice versa.
    for penalty in (1.25, 1.5, 1.75):
        top = score(best, penalty)
        if any(not np.array_equal(b, best) and score(b, penalty) <= top+.2
               for b in ordered[1:]):
            raise ValueError('competing rectilinear completions')

    # Also challenge the inferred coordinates, not just the edge thresholds.
    yy, xx = _canonical_cuts(best), _canonical_cuts(best.T)
    for axis, cuts in enumerate((yy, xx)):
        for i in range(1, len(cuts)-1):
            for step in (-1, 1):
                moved = cuts.copy()
                moved[i] += step
                if not moved[i-1] < moved[i] < moved[i+1]:
                    continue
                b = _partition_vote(observed, moved if axis == 0 else yy,
                                     moved if axis == 1 else xx)
                if b is None:
                    continue
                if not np.array_equal(b, best) and score(b) <= score(best)+1:
                    raise ValueError('boundary position is ambiguous')
    if np.count_nonzero(best != observed) > max(5, observed.size//9):
        raise ValueError('insufficient intact geometric evidence')
    return best.tobytes()


def _recover_base(inp, solid):
    a = inp == solid
    return np.frombuffer(_recover_cached(a.shape, a.astype(np.uint8).tobytes()),
                         dtype=bool).reshape(a.shape).copy()


def _paint_constructed(base, holes, extras, field, solid, highlight):
    """Construction renderer: explicit masks, no decoder or predictor calls."""
    inp = np.where(base, solid, field).astype(np.int64)
    inp[holes], inp[extras] = field, solid
    out = np.where(base, solid, field).astype(np.int64)
    for y, x in np.argwhere(holes):
        out[max(0, y-1):y+2, max(0, x-1):x+2] = highlight
    out[holes] = field
    return {'input': inp, 'output': out}


def _prediction(inp, field, solid, highlight, base=None):
    """Input-based rule application, separate from construction painting."""
    if base is None:
        base = _recover_base(inp, solid)
    missing = base & (inp == field)
    result = inp.copy()
    result[(~base) & (inp == solid)] = field
    result[_dilate(missing) & ~missing] = highlight
    return result


def _learn(training):
    input_colors = set().union(*(set(map(int, np.unique(p['input']))) for p in training))
    output_colors = set().union(*(set(map(int, np.unique(p['output']))) for p in training))
    new = output_colors-input_colors
    assert len(new) == 1, 'training must show one output-only highlight'
    highlight = next(iter(new))
    roles = []
    for p in training:
        colors = sorted(map(int, np.unique(p['input'])))
        assert len(colors) == 2
        fits = []
        for field, solid in (colors, colors[::-1]):
            try:
                prediction = _prediction(p['input'], field, solid, highlight)
            except ValueError:
                continue
            if np.array_equal(prediction, p['output']):
                fits.append((field, solid))
        assert len(fits) == 1, 'training does not distinguish the asymmetric roles'
        roles.append(fits[0])
    fields, solids = set(f for f, s in roles), set(s for f, s in roles)
    assert len(fields) == 1 or len(solids) == 1, 'both roles cannot vary without a cue'
    return fields, solids, highlight


def _roles_for(inp, learned):
    fields, solids, highlight = learned
    colors = set(map(int, np.unique(inp)))
    assert len(colors) == 2 and highlight not in colors
    if len(fields) == len(solids) == 1:
        field, solid = next(iter(fields)), next(iter(solids))
        assert colors == {field, solid}
    elif len(solids) == 1:
        solid = next(iter(solids))
        assert solid in colors
        field = next(iter(colors-{solid}))
    else:
        field = next(iter(fields))
        assert field in colors
        solid = next(iter(colors-{field}))
    return field, solid, highlight


def solve(training, test_input):
    """Predict with training and the input only; no generation state or labels."""
    return _prediction(test_input, *_roles_for(test_input, _learn(training)))


def _allocate(total, minimums, rng):
    values = list(minimums)
    for _ in range(total-sum(values)):
        values[int(rng.integers(len(values)))] += 1
    return values


def _span(n, minimum, rng, touch=False):
    extent = int(rng.integers(minimum, n+1))
    spare = n-extent
    starts = [0, spare] + list(range(2, spare-1))
    start = int(rng.choice(starts if not touch else [0, spare]))
    return start, start+extent


def _local_shape(h, w, kind, rng):
    a = np.zeros((h, w), dtype=bool)
    if kind == 'rectangle':
        y, v = _span(h, min(4, h), rng)
        x, u = _span(w, min(4, w), rng)
        a[y:v, x:u] = True
    elif kind in ('L', 'tee', 'cross', 'arms'):
        y, v = _span(h, min(8, h), rng)
        x, u = _span(w, min(8, w), rng)
        thick = int(rng.integers(2, max(3, min(7, (u-x)//2)+1)))
        left = x if kind in ('L', 'tee') else int(rng.integers(x, u-thick+1))
        a[y:v, left:left+thick] = True
        arms = int(rng.integers(2, 5)) if kind == 'arms' else 1
        for j in range(arms):
            height = int(rng.integers(2, max(3, min(6, (v-y)//2)+1)))
            top = y if kind == 'L' else int(rng.integers(y, v-height+1))
            lo = int(rng.integers(x, left+1))
            hi = int(rng.integers(left+thick, u+1))
            if kind in ('L', 'tee', 'cross'):
                lo, hi = x, u
            a[top:top+height, lo:hi] = True
    else:
        raise ValueError('unknown local shape')
    return a


def _nested(h, w, rng, island=True):
    a = np.zeros((h, w), dtype=bool)
    margins = [int(rng.integers(0, 3)) for _ in range(4)]
    if rng.random() < .35:
        margins = [0]*4
    t, b, l, r = margins
    if h-t-b < 11: t = b = 0
    if w-l-r < 11: l = r = 0
    a[t:h-b, l:w-r] = True
    rt = int(rng.integers(2, max(3, (h-t-b-5)//2+1)))
    rb = int(rng.integers(2, max(3, h-t-b-rt-4)))
    rl = int(rng.integers(2, max(3, (w-l-r-5)//2+1)))
    rr = int(rng.integers(2, max(3, w-l-r-rl-4)))
    y, v, x, u = t+rt, h-b-rb, l+rl, w-r-rr
    a[y:v, x:u] = False
    if island and v-y >= 5 and u-x >= 5:
        ih = int(rng.integers(3, v-y-1))
        iw = int(rng.integers(3, u-x-1))
        iy = int(rng.integers(y+1, v-ih))
        ix = int(rng.integers(x+1, u-iw))
        a[iy:iy+ih, ix:ix+iw] = True
    return a


def _slots(box, count, rng, minimum=5):
    slots = [box]
    for _ in range(count-1):
        options = [(i, axis) for i, (y, v, x, u) in enumerate(slots)
                   for axis, n in enumerate((v-y, u-x)) if n >= 2*minimum+2]
        if not options:
            break
        i, axis = options[int(rng.integers(len(options)))]
        y, v, x, u = slots.pop(i)
        n = (v-y, u-x)[axis]
        gap = int(rng.integers(2, min(5, n-2*minimum)+1))
        cut = int(rng.integers(minimum, n-gap-minimum+1))
        if axis == 0:
            slots.extend([(y, y+cut, x, u), (y+cut+gap, v, x, u)])
        else:
            slots.extend([(y, v, x, x+cut), (y, v, x+cut+gap, u)])
    return slots


_SIMPLE = ('rectangle', 'bands', 'nested')
_COMPOSITE = ('L', 'tee', 'cross', 'arms', 'separate', 'cavities', 'mixed_nested')


def _source_kind(a):
    """Recognize simple scaffolds geometrically, in either orientation."""
    if not a.any() or a.all():
        return None
    if np.all(a == a[:, :1]) or np.all(a == a[:1, :]):
        return 'bands'
    def box(b):
        points = np.argwhere(b)
        y, x = points.min(axis=0)
        v, u = points.max(axis=0)+1
        rectangle = np.zeros_like(b)
        rectangle[y:v, x:u] = True
        return rectangle, (y, v, x, u)
    outer, (y, v, x, u) = box(a)
    if np.array_equal(outer, a):
        return 'rectangle'
    cavity, (cy, cv, cx, cu) = box(outer & ~a)
    if not (y < cy < cv < v and x < cx < cu < u):
        return None
    inner = a & cavity
    if inner.any() and not np.array_equal(box(inner)[0], inner):
        return None
    if np.array_equal((outer & ~cavity) | inner, a):
        return 'nested'
    return None


def _make_base(rng, kind, orientation=None, require_edge=False):
    size = int(rng.choice(3, p=[.28, .43, .29]))
    low, high = ((10, 17), (16, 25), (24, 31))[size]
    h, w = int(rng.integers(low, high)), int(rng.integers(10, 31))
    if kind == 'nested':
        h, w = max(14, h), max(14, w)
    if kind in ('cavities', 'mixed_nested'):
        h, w = max(18, h), max(24, w)
    if kind in ('rectangle', 'L', 'tee', 'cross', 'arms'):
        a = _local_shape(h, w, kind, rng)
    elif kind == 'bands':
        count = int(rng.integers(3, min(9, h//2+1)))
        lengths = _allocate(h, [2]*count, rng)
        a = np.zeros((h, w), dtype=bool)
        y = 0
        phase = int(rng.integers(2))
        for i, length in enumerate(lengths):
            a[y:y+length] = (i+phase) % 2 == 0
            y += length
    elif kind == 'nested':
        a = _nested(h, w, rng)
    elif kind == 'separate':
        a = np.zeros((h, w), dtype=bool)
        for y, v, x, u in _slots((0, h, 0, w), int(rng.integers(2, 6)), rng):
            local = 'rectangle'
            if min(v-y, u-x) >= 8 and rng.random() < .55:
                local = str(rng.choice(('L', 'tee', 'arms')))
            a[y:v, x:u] |= _local_shape(v-y, u-x, local, rng)
    elif kind == 'cavities':
        a = np.ones((h, w), dtype=bool)
        for y, v, x, u in _slots((2, h-2, 2, w-2), int(rng.integers(2, 5)), rng):
            a[y:v, x:u] = False
            if min(v-y, u-x) >= 7 and rng.random() < .5:
                a[y+2:v-2, x+2:u-2] = True
    elif kind == 'mixed_nested':
        a = np.zeros((h, w), dtype=bool)
        cut = int(rng.integers(13, w-7))
        a[:, :cut] = _nested(h, cut, rng)
        y, v = _span(h, 5, rng)
        a[y:v, cut+3:w] = True
    else:
        raise ValueError('unknown scaffold family')
    if require_edge and not (a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any()):
        # Translate the whole scaffold to a feasible edge, preserving geometry.
        points = np.argwhere(a)
        shift = int(points[:, 0].min())
        a = np.roll(a, -shift, axis=0)
    k = int(rng.integers(4)) if orientation is None else orientation
    a = np.rot90(a, k)
    if rng.random() < .5:
        a = np.fliplr(a)
    return a.copy()


_MOTIFS = (
    ((0, 0),), ((0, 0), (0, 1)), ((0, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 1), (2, 2)),
    ((0, 0), (0, 2)), ((0, 0), (1, 1), (2, 0), (3, 1)),
    ((0, 0), (0, 2), (1, 3), (2, 2), (3, 3)),
)


def _place(rng, available, offsets, selected, anchor=None):
    pts = np.asarray(offsets, dtype=int)
    if rng.random() < .5: pts = pts[:, ::-1]
    pts *= rng.choice((-1, 1), size=2)
    pts -= pts.min(axis=0)
    hh, ww = pts.max(axis=0)+1
    h, w = available.shape
    if hh > h or ww > w:
        return False
    good = np.ones((h-hh+1, w-ww+1), dtype=bool)
    for dy, dx in pts:
        good &= available[dy:dy+good.shape[0], dx:dx+good.shape[1]]
        good &= ~selected[dy:dy+good.shape[0], dx:dx+good.shape[1]]
    if anchor is not None:
        touches = np.zeros_like(good)
        for dy, dx in pts:
            touches |= anchor[dy:dy+good.shape[0], dx:dx+good.shape[1]]
        good &= touches
    starts = np.argwhere(good)
    if not len(starts):
        return False
    start = starts[int(rng.integers(len(starts)))]
    pts += start
    selected[pts[:, 0], pts[:, 1]] = True
    return True


def _features(base, holes, extras):
    h, w = base.shape
    halo = _dilate(holes) & ~holes
    groups = _components(holes, diagonal=True)
    counts = np.zeros_like(base, dtype=np.int16)
    for group in groups:
        counts += _dilate(group)
    edge = np.zeros_like(base)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    boundary = base & _dilate(~base)
    up = np.pad(base, 1, constant_values=True)
    vertical = ~up[:-2, 1:-1] | ~up[2:, 1:-1]
    horizontal = ~up[1:-1, :-2] | ~up[1:-1, 2:]
    ortho = bool((holes[1:] & holes[:-1]).any() or (holes[:, 1:] & holes[:, :-1]).any())
    diagonal = bool((holes[1:, 1:] & holes[:-1, :-1]).any()
                    or (holes[1:, :-1] & holes[:-1, 1:]).any())
    changes = holes | extras
    comps = _components(base)
    cavity_components = [b for b in _components(~base) if not (b & edge).any()]
    def rectangle(b):
        points = np.argwhere(b)
        extent = points.max(axis=0)-points.min(axis=0)+1
        return int(b.sum()) == int(np.prod(extent))
    def component_corners(b):
        p = np.pad(b.astype(np.int16), 1)
        return int(np.abs(p[1:, 1:]-p[:-1, 1:]-p[1:, :-1]+p[:-1, :-1]).sum())
    group_sizes = [int(g.sum()) for g in groups]
    triple_l = triple_diagonal = False
    for group in groups:
        points = np.argwhere(group)
        extent = points.max(axis=0)-points.min(axis=0)+1
        if len(points) == 3:
            triple_l |= bool(np.array_equal(extent, [2, 2]))
            triple_diagonal |= bool(np.array_equal(extent, [3, 3])
                                    and len(set(points[:, 0])) == len(set(points[:, 1])) == 3)
    nested = False
    for cavity in cavity_components:
        points = np.argwhere(cavity)
        y, x = points.min(axis=0)
        v, u = points.max(axis=0)+1
        nested |= bool(base[y:v, x:u].any())
    return {
        'height': h, 'width': w, 'holes': int(holes.sum()), 'extras': int(extras.sum()),
        'hole_groups': len(groups), 'largest_hole_group': max([int(g.sum()) for g in groups], default=0),
        'orthogonal': ortho, 'diagonal': diagonal,
        'overlap': bool(((counts >= 2) & halo).any()),
        'edge': bool((holes & edge).any()), 'boundary': bool((holes & boundary).any()),
        'corner': bool((holes & vertical & horizontal).any()),
        'interior': bool((holes & ~boundary & ~edge).any()),
        'adjacent_extras': int((extras & halo).sum()),
        'far_extras': int((extras & ~_dilate(holes)).sum()),
        'extra_cluster': any(g.sum() > 1 for g in _components(extras, diagonal=True)),
        'max_row_flips': int(changes.sum(axis=1).max()),
        'max_col_flips': int(changes.sum(axis=0).max()),
        'row_runs': len(_canonical_cuts(base))-1,
        'min_row_repeat': min(np.unique(base, axis=0, return_counts=True)[1]).item(),
        'min_col_repeat': min(np.unique(base.T, axis=0, return_counts=True)[1]).item(),
        'components': len(comps), 'cavities': len(cavity_components),
        'rectangles': sum(rectangle(b) for b in comps),
        'L_components': sum(component_corners(b) == 6 for b in comps),
        'complex_components': sum(component_corners(b) >= 10 for b in comps),
        'nested': nested, 'triple_L': triple_l, 'triple_diagonal': triple_diagonal,
        'dimers': group_sizes.count(2), 'isolated_holes': group_sizes.count(1),
        'turns': _turns(base),
        'horizontal_bands': bool(np.all(base == base[:, :1])),
        'vertical_bands': bool(np.all(base == base[:1, :])),
        'edge_contacts': int(base[0].any())+int(base[-1].any())+int(base[:, 0].any())+int(base[:, -1].any()),
        'solid_cells': int(base.sum()),
    }


def _covers(features, required):
    return all(features[k] for k in required)


def _corrupt(rng, base, colors, required, richness):
    field, solid, highlight = colors
    solid_cells, field_cells = int(base.sum()), int((~base).sum())
    if min(solid_cells, field_cells) < 6:
        return None
    max_holes = min(18, max(1, solid_cells//14), max(2, base.size//24))
    max_extras = min(10, max(1, field_cells//14))
    boundary = base & _dilate(~base)
    edge = np.zeros_like(base)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    for _ in range(36):
        holes, extras = np.zeros_like(base), np.zeros_like(base)
        simpler = rng.random() < .14 and not required
        wanted = int(rng.integers(1, max_holes+1))
        if richness == 0: wanted = min(wanted, int(rng.integers(1, 5)))
        if simpler: wanted = int(rng.choice((0, 1)))
        if 'overlap' in required: wanted = max(3, wanted)
        if 'edge' in required:
            if not _place(rng, base, _MOTIFS[0], holes, edge): continue
        if 'boundary' in required or 'adjacent_extras' in required:
            if not _place(rng, base, _MOTIFS[0], holes, boundary): continue
        if 'overlap' in required:
            if not _place(rng, base, _MOTIFS[5], holes): continue
        if 'orthogonal' in required:
            if not _place(rng, base, _MOTIFS[1], holes): continue
        if 'diagonal' in required:
            if not _place(rng, base, _MOTIFS[2], holes): continue
        for _ in range(24):
            left = wanted-int(holes.sum())
            if left <= 0: break
            options = [m for m in _MOTIFS if len(m) <= left]
            motif = options[int(rng.integers(len(options)))]
            _place(rng, base, motif, holes)
        if 'holes' in required and not holes.any():
            _place(rng, base, _MOTIFS[0], holes)
        extra_count = int(rng.integers(0, max_extras+1))
        if not holes.any(): extra_count = max(1, extra_count)
        if 'far_extras' in required: extra_count = max(1, extra_count)
        if 'adjacent_extras' in required: extra_count = max(2, extra_count)
        proximity = int(rng.choice(3, p=[.45, .30, .25]))
        if 'adjacent_extras' in required: proximity = 2
        near = ~base & _dilate(holes)
        if proximity == 2:
            for _ in range(int(rng.integers(1, 4))):
                _place(rng, near, _MOTIFS[0], extras)
        available = ~base & (~_dilate(holes) if proximity == 0 else True)
        for _ in range(18):
            if extras.sum() >= extra_count: break
            motif = _MOTIFS[int(rng.choice((0, 0, 0, 1, 2)))]
            if len(motif) > extra_count-extras.sum(): motif = _MOTIFS[0]
            _place(rng, available, motif, extras)
        if 'far_extras' in required and not (extras & ~_dilate(holes)).any():
            _place(rng, ~base & ~_dilate(holes), _MOTIFS[0], extras)
        features = _features(base, holes, extras)
        if not _covers(features, required): continue
        pair = _paint_constructed(base, holes, extras, *colors)
        try:
            recovered = _recover_base(pair['input'], solid)
        except ValueError:
            continue
        if np.array_equal(recovered, base):
            return pair
    return None


def _make_pair(rng, kinds, colors, required, richness, orientation=None):
    for attempt in range(100):
        kind = str(rng.choice(kinds))
        base = _make_base(rng, kind, orientation, 'edge' in required)
        if kinds == _COMPOSITE and _source_kind(base) is not None:
            continue
        # First ensure the clean composition itself has readable long sides.
        try:
            if not np.array_equal(_recover_base(base.astype(int), 1), base):
                continue
        except ValueError:
            continue
        pair = _corrupt(rng, base, colors, required, richness)
        if pair is not None:
            return pair
    # A bounded fallback uses spacious representatives of the requested family.
    # It preserves both the split regime and the teaching effects. Ordinary
    # samples above use freely composed dimensions, positions, and rectangles.
    for attempt in range(32):
        kind = kinds[attempt % len(kinds)]
        base = np.zeros((30, 30), dtype=bool)
        if kind == 'rectangle':
            base[:22, 2:26] = True
        elif kind == 'bands':
            base[:8] = True
            base[12:20] = True
            base[24:] = True
        elif kind == 'nested':
            base[:] = True
            base[5:25, 5:25] = False
            base[10:20, 10:20] = True
        elif kind == 'L':
            base[:24, :8] = True
            base[:7, :27] = True
        elif kind == 'tee':
            base[:, :7] = True
            base[10:18, :26] = True
        elif kind == 'cross':
            base[:, 9:17] = True
            base[9:17] = True
        elif kind == 'arms':
            base[:, 8:15] = True
            base[4:9, :26] = True
            base[15:21, 4:] = True
        elif kind == 'separate':
            base[:17, :8] = True
            base[:6, :17] = True
            base[20:, 17:] = True
        elif kind == 'cavities':
            base[:] = True
            base[4:14, 4:12] = False
            base[17:26, 16:26] = False
        else:
            base[:, :18] = True
            base[5:25, 4:14] = False
            base[10:20, 7:11] = True
            base[9:, 22:] = True
        base = np.rot90(base, attempt%4 if orientation is None else orientation)
        pair = _corrupt(rng, base, colors, required, 1)
        if pair is not None:
            return pair
    raise RuntimeError('could not construct the required visible evidence')


def _inspect(pair, learned):
    field, solid, highlight = _roles_for(pair['input'], learned)
    base = _recover_base(pair['input'], solid)
    holes = base & (pair['input'] == field)
    extras = ~base & (pair['input'] == solid)
    return base, _features(base, holes, extras)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    colors = list(map(int, rng.permutation(10)))
    field, solid, highlight = colors[:3]
    palette_mode = int(rng.choice(3, p=[.44, .28, .28]))
    varying = [colors[0 if palette_mode == 1 else 1]] + colors[3:]
    rng.shuffle(varying)
    regime = int(rng.integers(4))
    richness = int(rng.integers(3))
    familiar = tuple(map(str, rng.choice(_SIMPLE+_COMPOSITE, size=2, replace=False)))
    if regime == 0:
        train_kinds, test_kinds = _SIMPLE, _COMPOSITE
    elif regime == 1:
        train_kinds = test_kinds = _SIMPLE+_COMPOSITE
    else:
        train_kinds = test_kinds = familiar
    # Familiar-family tests must actually have demonstrations of their families.
    # Orientation shifts keep the family fixed while geometry and noise vary.
    if regime == 3:
        train_kinds = test_kinds = (familiar[0],)
    demonstrated = tuple(map(str, rng.permutation(train_kinds)))
    training_choices = [(demonstrated[i],) if i < min(2, len(demonstrated))
                        else train_kinds for i in range(num_train)]
    palette_indices = rng.integers(len(varying), size=num_train+num_test)
    palette_indices[:2] = [0, 1]

    def pair_colors(index):
        if palette_mode == 1: return varying[palette_indices[index]], solid, highlight
        if palette_mode == 2: return field, varying[palette_indices[index]], highlight
        return field, solid, highlight

    puzzle = {'train': [], 'test': []}
    test_features = []
    for i in range(num_test):
        pair = _make_pair(rng, test_kinds, pair_colors(num_train+i), set(), richness,
                          int(rng.choice((1, 3))) if regime == 3 else None)
        puzzle['test'].append(pair)
        f, s, _ = pair_colors(num_train+i)
        b = _recover_base(pair['input'], s)
        test_features.append(_features(b, b & (pair['input'] == f), ~b & (pair['input'] == s)))
    needs = {k for k in ('edge', 'boundary', 'overlap', 'adjacent_extras',
                         'orthogonal', 'diagonal')
             if any(f[k] for f in test_features)}
    # Distribute teaching cases across two anchors, with independent remaining
    # training examples. Neither every pair nor every puzzle must show clipping.
    requirements = [{'holes', 'far_extras'}, {'holes'}] + [set() for _ in range(num_train-2)]
    for k in needs:
        target = 0 if k in ('edge', 'orthogonal') else 1
        requirements[target].add(k)
    for i in range(num_train):
        pair = _make_pair(rng, training_choices[i], pair_colors(i), requirements[i], richness,
                          int(rng.choice((0, 2))) if regime == 3 else None)
        puzzle['train'].append(pair)
    # Resolve rare equal-count or input collisions with fresh deterministic draws.
    learned = _learn(puzzle['train'])
    hole_counts = [_inspect(p, learned)[1]['holes'] for p in puzzle['train']+puzzle['test']]
    if len(set(hole_counts)) == 1:
        for _ in range(20):
            p = _make_pair(rng, training_choices[1], pair_colors(1), requirements[1], richness,
                           int(rng.choice((0, 2))) if regime == 3 else None)
            if _inspect(p, learned)[1]['holes'] != hole_counts[0]:
                puzzle['train'][1] = p
                break
    seen = set()
    for split, kinds in (('train', train_kinds), ('test', test_kinds)):
        for i, pair in enumerate(puzzle[split]):
            index = i if split == 'train' else num_train+i
            req = requirements[i] if split == 'train' else set()
            for attempt in range(20):
                key = _signature(pair['input'])
                if key not in seen and not _is_source(pair): break
                pair = _make_pair(rng, training_choices[i] if split == 'train' else kinds,
                                  pair_colors(index), req, richness,
                                  int(rng.choice((0, 2) if split == 'train' else (1, 3)))
                                  if regime == 3 else None)
            seen.add(_signature(pair['input']))
            puzzle[split][i] = pair
    validate_generated_puzzle(puzzle)
    return puzzle


def _is_source(pair):
    encoded = tuple('/'.join(''.join(map(str, row)) for row in pair[k].tolist())
                    for k in ('input', 'output'))
    return encoded in _SOURCE_PAIRS


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate(puzzle)
    except (ValueError, IndexError, KeyError, TypeError) as error:
        raise AssertionError(str(error)) from error


def _validate(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for a in pair.values():
                assert isinstance(a, np.ndarray) and a.ndim == 2
                assert np.issubdtype(a.dtype, np.integer)
                assert all(1 <= n <= 30 for n in a.shape)
                assert ((a >= 0) & (a <= 9)).all()
            assert pair['input'].shape == pair['output'].shape
            assert not np.array_equal(pair['input'], pair['output'])
            key = _signature(pair['input'])
            assert key not in seen and not _is_source(pair)
            seen.add(key)
    learned = _learn(puzzle['train'])
    training_features, testing_features = [], []
    for split, records in (('train', training_features), ('test', testing_features)):
        for pair in puzzle[split]:
            inp = pair['input']
            roles = _roles_for(inp, learned)
            base, features = _inspect(pair, learned)
            assert np.array_equal(_prediction(inp, *roles, base), pair['output'])
            assert features['holes'] or features['extras']
            assert features['solid_cells'] >= 6 and inp.size-features['solid_cells'] >= 6
            records.append(features)
    assert any(f['holes'] for f in training_features)
    assert any(f['far_extras'] for f in training_features), 'teach erasure outside halos'
    for effect in ('edge', 'boundary', 'overlap', 'adjacent_extras',
                   'orthogonal', 'diagonal'):
        assert not any(f[effect] for f in testing_features) or any(f[effect] for f in training_features), effect
    assert len({f['holes'] for f in training_features+testing_features}) > 1
    # This prediction receives no output from the pair being tested.
    for pair in puzzle['test']:
        predicted = solve(puzzle['train'], pair['input'].copy())
        assert np.array_equal(predicted, pair['output'])


def _source_puzzle():
    pairs = [{'input': np.array([list(map(int, r)) for r in inp.split('/')]),
              'output': np.array([list(map(int, r)) for r in out.split('/')])}
             for inp, out in _SOURCE_PAIRS]
    return {'train': pairs[:3], 'test': pairs[3:]}


def _source_regressions():
    source = _source_puzzle()
    # Clean scaffolds inferred from the source inputs, independent of painter.
    b0 = np.zeros((19, 17), dtype=bool)
    b0[:5] = True; b0[9:14] = True; b0[17:] = True
    b1 = np.ones((16, 16), dtype=bool)
    b1[3:11, 2:13] = False; b1[5:10, 6:12] = True
    assert _source_kind(b1) == 'nested'
    b2 = np.zeros((16, 16), dtype=bool); b2[4:13, 4:14] = True
    b3 = np.zeros((16, 16), dtype=bool); b3[:, :5] = True; b3[3:12, :14] = True
    b4 = np.zeros((18, 19), dtype=bool)
    b4[:3, 2:] = True; b4[:13, 2:9] = True; b4[6:, 11:] = True
    for k in range(4):
        for reflection in (False, True):
            def transform(a):
                return np.rot90(np.fliplr(a) if reflection else a, k).copy()
            transformed_train = [{key: transform(a) for key, a in p.items()} for p in source['train']]
            for pair, base in zip(source['train']+source['test'], (b0, b1, b2, b3, b4)):
                inp = transform(pair['input'])
                assert np.array_equal(_recover_base(inp, 1), transform(base))
                prediction = solve(transformed_train, inp)
                assert np.array_equal(prediction, transform(pair['output']))


def self_test() -> None:
    _source_regressions()
    for inp_text, out_text, roles in _REVIEWED_REGRESSIONS:
        inp = np.array([list(map(int, row)) for row in inp_text.split('/')])
        out = np.array([list(map(int, row)) for row in out_text.split('/')])
        for k in range(4):
            assert np.array_equal(_prediction(np.rot90(inp, k).T, *roles),
                                  np.rot90(out, k).T)
    compact = np.zeros((12, 15), dtype=bool)
    compact[:, 2:5] = True
    compact[:, 9:11] = True
    holes = np.zeros_like(compact)
    holes[4, 3] = holes[5, 2] = holes[5, 3] = True
    extras = np.zeros_like(compact)
    extras[1, 6] = extras[2, 7] = True
    pair = _paint_constructed(compact, holes, extras, 0, 1, 7)
    for k in range(4):
        assert np.array_equal(_recover_base(np.rot90(pair['input'], k), 1), np.rot90(compact, k))
        assert np.array_equal(_prediction(np.rot90(pair['input'], k), 0, 1, 7),
                              np.rot90(pair['output'], k))
    sizes, train_counts, test_counts, palettes, structures = set(), set(), set(), set(), set()
    features, modes = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        learned = _learn(p['train'])
        fields, solids, highlight = learned
        modes.add((len(fields) > 1, len(solids) > 1))
        train_counts.add(len(p['train'])); test_counts.add(len(p['test']))
        palettes.add((tuple(sorted(fields)), tuple(sorted(solids)), highlight))
        local = []
        for pair in p['train']+p['test']:
            base, f = _inspect(pair, learned)
            features.append(f); local.append(f)
            sizes.add(base.shape); structures.add(_signature(base))
        assert len({f['holes'] for f in local}) >= 2
        if seed < 3:
            q = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    assert len(sizes) >= 65 and len(structures) >= 200
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palettes) >= 25 and len(modes) == 3
    assert min(f['height'] for f in features) <= 12 and max(f['height'] for f in features) == 30
    assert len({f['holes'] for f in features}) >= 9
    assert len({f['extras'] for f in features}) >= 7
    assert max(f['components'] for f in features) >= 4
    assert max(f['cavities'] for f in features) >= 2
    assert max(f['L_components'] for f in features) >= 1
    assert max(f['complex_components'] for f in features) >= 1
    assert max(f['dimers'] for f in features) >= 2
    assert max(f['isolated_holes'] for f in features) >= 3
    assert any(f['triple_L'] for f in features)
    assert any(f['triple_diagonal'] for f in features)
    assert any(f['nested'] and f['components'] >= 3 for f in features)
    assert max(f['largest_hole_group'] for f in features) >= 4
    assert max(f['max_row_flips'] for f in features) >= 4
    assert max(f['max_col_flips'] for f in features) >= 4
    assert min(f['min_row_repeat'] for f in features) < 4
    assert min(f['min_col_repeat'] for f in features) < 4
    for flag in ('horizontal_bands', 'vertical_bands', 'edge', 'boundary', 'corner',
                 'interior', 'overlap', 'orthogonal', 'diagonal', 'extra_cluster'):
        assert any(f[flag] for f in features), flag
    assert any(f['adjacent_extras'] >= 2 for f in features)
    assert any(f['extras'] and f['adjacent_extras'] == 0 for f in features)
    assert any(f['holes'] == 0 for f in features)
    assert any(f['extras'] == 0 for f in features)
    assert any(max(f['height'], f['width']) <= 17 and f['hole_groups'] >= 2 for f in features)
    assert any(f['cavities'] and f['components'] > 1 for f in features)
    for tr, te in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(9001, tr, te)
        if tr is not None: assert len(p['train']) == tr
        if te is not None: assert len(p['test']) == te
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (3, 1.5)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    # A fully erased tiny candidate object has multiple plausible explanations.
    ambiguous = np.zeros((10, 10), dtype=int)
    ambiguous[4:6, 4:6] = 1
    try:
        _recover_base(ambiguous, 1)
    except ValueError:
        pass
    else:
        raise AssertionError('tiny unsupported region accepted')


# Exact immutable source evidence for regressions and content-based exclusion.
# These are never sampled as generated examples or consulted by solve().
_SOURCE_PAIRS = (('11101111111111111/11111111111111111/11111111111111110/11111111111111111/11111111011111111/00000000000000000/00000000000000000/00000000000001000/00000000000000000/11111111111111111/11111101111111111/11111001111111111/11111111111111111/11111111111111111/00000000100000000/00010000000000000/00000000000000000/11111111111111011/11111111111111111', '11707111111111111/11777111111111177/11111111111111170/11111117771111177/11111117071111111/00000007770000000/00000000000000000/00000000000000000/00000000000000000/11111777111111111/11117707111111111/11117007111111111/11117777111111111/11111111111111111/00000000000000000/00000000000000000/00000000000007770/11111111111117071/11111111111117771'), ('1111111111111110/1111111111111111/1111111111111111/1100000000000111/1100000000000111/1110001111110111/1100001111110101/1100001101110111/1100101111110110/1100001111110111/1100000000000111/1111111111111111/1111111111111111/1111110011111111/1111111111111111/1111111111111111', '1111111111111170/1111111111111177/1111111111111111/1100000000000111/1100000000000111/1100001111110777/1100001777110707/1100001707110777/1100001777110170/1100001111110177/1100000000000111/1111111111111111/1111177771111111/1111170071111111/1111177771111111/1111111111111111'), ('0000000000000000/0000000000000000/0000000000000000/0000001000000000/0000111111111101/0000111111111100/0000111111011100/0000111111101100/0000111111111100/0000111111111100/0000111111111100/0000111111111100/0000111011111100/0000000000000000/0100000000000000/0000000000000000', '0000000000000000/0000000000000000/0000000000000000/0000000000000000/0000111111111100/0000111117771100/0000111117077100/0000111117707100/0000111111777100/0000111111111100/0000111111111100/0000117771111100/0000117071111100/0000007770000000/0000000000000000/0000000000000000'), ('1111100000000000/1111100000000000/1111100000000000/1111111111101100/1111111111111101/1111101111111100/1111011111111101/1110111111111100/1111111111101100/1111111111101110/1111111111111100/1111110111111100/1111100000000000/1111100000000000/1111100000001000/1111100000000001', '1111100000000000/1111100000000000/1111100000777000/1111111111707100/1111777111777100/1117707111111100/1177077111111100/1170771111777100/1177711111707100/1111111111707100/1111177711777100/1111170711111100/1111177700000000/1111100000000000/1111100000000000/1111100000000000'), ('0011111111111111111/0111111111111111111/0011011111111111011/0011111110000000000/0011110110000100000/0011111110000000000/1011111110011111111/0011111110011111111/0011101110011111111/0011010110011101111/0011111110011101111/0011111110011111111/0011111110011111111/0001000000011111111/0000000000011111111/0000000010011111110/0000000000010111111/0000000000011011111', '0011111111111111111/0017771111111117771/0017071111111117071/0017777710000007770/0011170710000000000/0011177710000000000/0011111110011111111/0011777110011111111/0017707710011777111/0017070710011707111/0017777710011707111/0011111110011777111/0011111110011111111/0000000000011111111/0000000000011111177/0000000000077711170/0000000000070771177/0000000000077071111'))

_REVIEWED_REGRESSIONS = (('2222232222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233332332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2223333333333333333333333333/2223333333333333333333333322/2223333333333333333333333332/2223333333333333333333333332/2223333333333333333333333332/2222222222222233333332222222/2222222222222233323332222222/2222222222222233332332222222/2222222222222222222222222222/2222222222222222222222222222/2222222222222222222222222222/2222222222222222222222222222', '2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233344432222222/2222222222222233342432222222/2222222222222233344432222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2222222222222233333332222222/2223333333333333333333333444/2223333333333333333333333424/2223333333333333333333333444/2223333333333333333333333332/2223333333333333333333333332/2222222222222233444332222222/2222222222222233424432222222/2222222222222233442432222222/2222222222222222244422222222/2222222222222222222222222222/2222222222222222222222222222/2222222222222222222222222222', (2, 3, 4)), ('533333335555555553535555555555/533333335555555555555555555555/533333335555555555555555555555/533333335555553555555555555555/533333333333333333335555555555/533333333333333333335555555555/533333333333333333535555555555/533333333333333333335555555555/533333333333333333335555555555/533333335555555555555555555555/533333335555555555555555555555/533333335555555555555555555555/533333335555555555555555555555/533333335555555555555555555555/533333335555555555555555555555/555555555555555555555555555555/555555555555555555555555555555/555555555555555555555555555555/555555555555555555555555555555/555555555555555555555533333335/555555555555555555555533333335/555555555555555555555533333335/555555555555555555555533333335/555555555555555555555533333335/555555555555555555555533333335/555555555555555555555533333335/555555555555555555555533333335/555555555555555555555535533335/555555555555555555555533333353', '533333335555555555555555555555/533333335555555555555555555555/533333335555555555555555555555/533333335555555555555555555555/533333333333333333335555555555/533333333333333334445555555555/533333333333333334545555555555/533333333333333334445555555555/533333333333333333335555555555/533333335555555555555555555555/533333335555555555555555555555/533333335555555555555555555555/533333335555555555555555555555/533333335555555555555555555555/533333335555555555555555555555/555555555555555555555555555555/555555555555555555555555555555/555555555555555555555555555555/555555555555555555555555555555/555555555555555555555533333335/555555555555555555555533333335/555555555555555555555533333335/555555555555555555555533333335/555555555555555555555533333335/555555555555555555555533333335/555555555555555555555533333335/555555555555555555555544443335/555555555555555555555545543444/555555555555555555555544443454', (5, 3, 4)))
