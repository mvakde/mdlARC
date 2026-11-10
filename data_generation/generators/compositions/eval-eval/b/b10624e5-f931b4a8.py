"""Sealed chamber printing: anchored geometry, topology, and native texture.

All conventions are visible in one input. The header contains a framed texture;
the field contains one hollow specimen and several matching square reservoirs.
"""
from collections import deque
import json
import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    return int(value)


def _counts(rng, train, test):
    # None is supported as a convenience; the public defaults are exactly 4/1.
    if train is not None:
        train = _integer(train, 'num_train')
        if not 2 <= train <= 10:
            raise ValueError('num_train must be 2..10')
    if test is not None:
        test = _integer(test, 'num_test')
        if not 1 <= test <= 4:
            raise ValueError('num_test must be 1..4')
    if train is None and test is None:
        test = int(rng.integers(1, 5))
    if train is None:
        train = int(rng.integers(test + 1, 11))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError('num_train must exceed num_test')
    return train, test


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.uint8).tobytes()


def _normal(grid):
    mapping = {}
    return grid.shape, bytes(mapping.setdefault(int(x), len(mapping)) for x in grid.flat)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        todo, cells = [p], [p]
        while todo:
            r, c = todo.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in unseen:
                    unseen.remove(q)
                    cells.append(q)
                    todo.append(q)
        result.append(cells)
    return result


def _enclosed(wall):
    """Four-connected background that cannot reach any field boundary."""
    h, w = wall.shape
    exterior = np.zeros((h, w), dtype=bool)
    todo = deque()
    for r, c in ([(0, c) for c in range(w)] + [(h-1, c) for c in range(w)]
                 + [(r, 0) for r in range(h)] + [(r, w-1) for r in range(h)]):
        if not wall[r, c] and not exterior[r, c]:
            exterior[r, c] = True
            todo.append((r, c))
    while todo:
        r, c = todo.popleft()
        for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= y < h and 0 <= x < w and not wall[y, x] and not exterior[y, x]:
                exterior[y, x] = True
                todo.append((y, x))
    return ~wall & ~exterior


def _local_holes(wall):
    # The padding supplies the infinite-plane exterior of a complete specimen.
    return _enclosed(np.pad(wall, 1))[1:-1, 1:-1]


def _place(array, shape, top, left, fill):
    result = np.full(shape, fill, dtype=array.dtype)
    r0, c0 = max(0, top), max(0, left)
    r1, c1 = min(shape[0], top+array.shape[0]), min(shape[1], left+array.shape[1])
    if r0 < r1 and c0 < c1:
        result[r0:r1, c0:c1] = array[r0-top:r1-top, c0-left:c1-left]
    return result


def _square(grid, cells, color):
    assert cells
    rr, cc = np.array(cells).T
    r, c, h, w = int(rr.min()), int(cc.min()), int(np.ptp(rr)+1), int(np.ptp(cc)+1)
    assert h == w and len(cells) == h*h
    assert np.all(grid[r:r+h, c:c+h] == color)
    return r, c, h


def _decode(input_grid):
    g = np.asarray(input_grid)
    assert g.ndim == 2 and np.issubdtype(g.dtype, np.integer)
    assert all(1 <= n <= 30 for n in g.shape) and np.all((0 <= g) & (g <= 9))
    frame, bg = int(g[0, 0]), int(g[0, -1])
    assert frame != bg
    separators = [r for r in range(g.shape[0]) if np.all(g[r] == frame)]
    assert len(separators) == 1
    sep = separators[0]
    nonframe = np.flatnonzero(g[0] != frame)
    assert len(nonframe)
    boxwidth = int(nonframe[0])
    assert sep >= 3 and boxwidth >= 3 and boxwidth < g.shape[1]
    assert np.all(g[:sep, boxwidth:] == bg)
    assert np.all(g[0, :boxwidth] == frame) and np.all(g[sep-1, :boxwidth] == frame)
    assert np.all(g[:sep, 0] == frame) and np.all(g[:sep, boxwidth-1] == frame)
    texture = g[1:sep-1, 1:boxwidth-1].copy()
    assert np.any(texture == bg) and np.any(texture != bg)
    field = g[sep+1:].copy()
    assert min(field.shape) >= 3
    comps = _components(field != bg)
    sources = [cells for cells in comps if len({int(field[p]) for p in cells}) > 1]
    assert len(sources) == 1, 'one connected multicolored specimen required'
    source = sources[0]
    target_cells = [cells for cells in comps if cells is not source]
    assert len(target_cells) >= 2
    target_colors = {int(field[cells[0]]) for cells in target_cells}
    assert len(target_colors) == 1
    anchor = target_colors.pop()
    assert anchor not in (bg, frame) and not np.any(texture == anchor)
    targets = sorted(_square(field, cells, anchor) for cells in target_cells)
    source_anchor = [p for p in source if field[p] == anchor]
    origin = _square(field, source_anchor, anchor)
    rr, cc = np.array(source).T
    r0, c0, r1, c1 = int(rr.min()), int(cc.min()), int(rr.max()+1), int(cc.max()+1)
    assert 0 < r0 < r1 < field.shape[0] and 0 < c0 < c1 < field.shape[1]
    raw = field[r0:r1, c0:c1].copy()
    assert _local_holes(raw != bg).any(), 'specimen must contain a closed chamber'
    return dict(grid=g, background=bg, frame=frame, separator=sep, texture=texture,
                field=field, anchor=anchor, origin=origin, targets=targets,
                raw=raw, offset=(origin[0]-r0, origin[1]-c0),
                source_bounds=(r0, r1, c0, c1))


def _flips(data, target):
    r, c, side = target
    ar, ac, a = data['origin']
    # Ties have the unreflected convention, without a seed-dependent choice.
    return 2*r+side < 2*ar+a, 2*c+side < 2*ac+a


def _scaled(data, side, vr, hc):
    """Exact rational pixel scaling; refuse a cell straddling two colors."""
    raw, a = data['raw'], data['origin'][2]
    assert raw.shape[0] % a == 0 and raw.shape[1] % a == 0
    expanded = np.repeat(np.repeat(raw, side, axis=0), side, axis=1)
    h, w = expanded.shape[0]//a, expanded.shape[1]//a
    blocks = expanded.reshape(h, a, w, a)
    low, high = blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))
    assert np.array_equal(low, high), 'ambiguous scale boundary'
    offsets = [v*side//a for v in data['offset']]
    assert all(v*side % a == 0 for v in data['offset'])
    if vr:
        low = low[::-1]
        offsets[0] = h-offsets[0]-side
    if hc:
        low = low[:, ::-1]
        offsets[1] = w-offsets[1]-side
    return low, offsets


def _texture_values(data, target, vr, hc, mode='correct'):
    r, c, side = target
    rr, cc = np.indices(data['field'].shape)
    yy, xx = rr-r, cc-c
    if vr:
        yy = side-1-yy
    if hc:
        xx = side-1-xx
    th, tw = data['texture'].shape
    if mode == 'scaled_texture':
        yy = yy*data['origin'][2]//side
        xx = xx*data['origin'][2]//side
    elif mode == 'global_phase':
        yy = -1-rr if vr else rr
        xx = -1-cc if hc else cc
    elif mode == 'phase_rows':
        yy = yy+1
    elif mode == 'phase_columns':
        xx = xx+1
    tiled = data['texture'][yy % th, xx % tw]
    if mode == 'copy_once':
        tiled = np.where((yy >= 0) & (yy < th) & (xx >= 0) & (xx < tw),
                         tiled, data['background'])
    return tiled


def _render(data, mode='correct', records=False):
    bg, anchor = data['background'], data['anchor']
    out = data['grid'].copy()
    field = out[data['separator']+1:]
    info = []
    for target in data['targets']:
        r, c, side = target
        vr, hc = _flips(data, target)
        if mode == 'horizontal_reflection':
            hc = False
        if mode == 'vertical_reflection':
            vr = False
        if mode == 'always_horizontal':
            hc = True
        if mode == 'always_vertical':
            vr = True
        if mode == 'field_center_reflection':
            vr = 2*r+side < field.shape[0]
            hc = 2*c+side < field.shape[1]
        use_side = data['origin'][2] if mode == 'anchor_scale' else side
        raster, off = _scaled(data, use_side, vr, hc)
        top, left = r-off[0], c-off[1]
        wall = _place(raster, field.shape, top, left, bg)
        wall_mask = wall != bg
        closed = _enclosed(wall_mask)
        holes = closed
        potential = None
        if records or mode == 'enclosure_after_clip':
            potential = _place(_local_holes(raster != bg), field.shape, top, left, False)
        if mode == 'enclosure_after_clip':
            holes = potential
        elif mode == 'cavity_gate':
            holes = _place(np.ones(raster.shape, bool), field.shape, top, left, False) & ~wall_mask
        tiled = _texture_values(data, target, vr, hc, mode)
        fill = np.where(tiled == bg, anchor, tiled)
        if mode == 'ignore_texture':
            fill = np.full(field.shape, anchor)
        elif mode == 'undercoat' or mode == 'transparent_overlay':
            fill = tiled  # Losing the carrier undercoat makes transparent cells blank.
        field[wall_mask] = wall[wall_mask]
        field[holes] = fill[holes]
        # Even the deliberately wrong scale alternative must retain given anchors.
        field[r:r+side, c:c+side] = anchor
        if records:
            info.append(dict(target=target, flips=(vr, hc),
                             total_wall=int(np.count_nonzero(raster != bg)),
                             visible_wall=int(wall_mask.sum()), holes=int(holes.sum()),
                             potential=int(potential.sum()),
                             leaked=int(np.count_nonzero(potential & ~closed)),
                             transparent=int(np.count_nonzero(holes & (tiled == bg))),
                             opaque=int(np.count_nonzero(holes & (tiled != bg))),
                             bounds=(max(0, top), min(field.shape[0], top+raster.shape[0]),
                                     max(0, left), min(field.shape[1], left+raster.shape[1]))))
    return (out, info) if records else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the specimen, anchors, and swatch entirely from this input."""
    return _render(_decode(input_grid))


_EVIDENCE = frozenset(('horizontal_reflection', 'vertical_reflection', 'anchor_scale',
                       'always_horizontal', 'always_vertical', 'field_center_reflection',
                       'scaled_texture', 'global_phase', 'phase_rows', 'phase_columns',
                       'copy_once', 'enclosure_after_clip', 'cavity_gate',
                       'ignore_texture', 'transparent_overlay'))


def _evidence(pair, data=None):
    data = _decode(pair['input']) if data is None else data
    out = np.asarray(pair['output'])
    return {name for name in _EVIDENCE if not np.array_equal(_render(data, name), out)}


def _grow(rng, shape, count):
    h, w = shape
    start = (int(rng.integers(h)), int(rng.integers(w)))
    points, frontier = set(), {start}
    while len(points) < count:
        choices = sorted(frontier)
        p = choices[int(rng.integers(len(choices)))]
        frontier.remove(p)
        points.add(p)
        r, c = p
        for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= y < h and 0 <= x < w and (y, x) not in points:
                frontier.add((y, x))
    mask = np.zeros(shape, bool)
    for p in points:
        mask[p] = True
    return mask


def _specimen(rng, bg, anchor, inks):
    h, w = int(rng.integers(4, 8)), int(rng.integers(4, 9))
    kind = int(rng.integers(4))
    cavity = np.zeros((h, w), bool)
    if kind in (0, 1):
        cavity[1:-1, 1:-1] = True
        wall = ~cavity
        if kind == 1 and w >= 6:
            wall[:, int(rng.integers(2, w-2))] = True
        if rng.random() < .4:
            # A wall tooth changes chamber geometry without changing the rule.
            wall[1:int(rng.integers(2, h-1)), int(rng.integers(1, w-1))] = True
    else:
        cavity[1:-1, 1:-1] = _grow(rng, (h-2, w-2), int(rng.integers(2, (h-2)*(w-2)+1)))
        dilation = np.zeros_like(cavity)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                dilation[1+dr:h-1+dr, 1+dc:w-1+dc] |= cavity[1:-1, 1:-1]
        wall = dilation & ~cavity
    # Connected wall appendages give asymmetric silhouettes and clipping that
    # can remove an arm while leaving a room closed.
    wall = np.pad(wall, 2)
    for _ in range(int(rng.integers(0, 5))):
        rr, cc = np.nonzero(wall)
        i = int(rng.integers(len(rr)))
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
        y, x = int(rr[i]+dr), int(cc[i]+dc)
        if 0 <= y < wall.shape[0] and 0 <= x < wall.shape[1]:
            if not _local_holes(wall)[y, x]:
                wall[y, x] = True
    if len(_components(wall)) != 1:
        return None
    holes = _local_holes(wall)
    if not holes.any():
        return None
    rr, cc = np.nonzero(wall)
    wall = wall[rr.min():rr.max()+1, cc.min():cc.max()+1]
    if max(wall.shape) > 9:
        return None
    yy, xx = np.indices(wall.shape)
    k = len(inks)
    color_mode = int(rng.integers(4))
    if color_mode == 0:
        labels = (xx*max(1, k)//wall.shape[1]) % k
    elif color_mode == 1:
        labels = (yy+2*xx)//int(rng.integers(1, 4)) % k
    elif color_mode == 2:
        centers = rng.choice(wall.size, k, replace=False)
        dist = abs(yy[..., None]-centers//wall.shape[1]) + abs(xx[..., None]-centers % wall.shape[1])
        labels = dist.argmin(axis=2)
    else:
        labels = np.zeros(wall.shape, int)
        for _ in range(k+1):
            y, x = int(rng.integers(wall.shape[0])), int(rng.integers(wall.shape[1]))
            labels[y:, x:] = int(rng.integers(k))
    raw = np.full(wall.shape, bg, dtype=np.int8)
    raw[wall] = np.asarray(inks)[labels[wall]]
    exterior = ~wall & ~_local_holes(wall)
    border = []
    for r, c in np.argwhere(wall):
        if any(y < 0 or y >= wall.shape[0] or x < 0 or x >= wall.shape[1]
               or exterior[y, x] for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))):
            border.append((int(r), int(c)))
    ar, ac = border[int(rng.integers(len(border)))]
    raw[ar, ac] = anchor
    turns = int(rng.integers(4))
    raw = np.rot90(raw, turns).copy()
    if rng.random() < .5:
        raw = raw[:, ::-1].copy()
    ar, ac = np.argwhere(raw == anchor)[0].tolist()
    return raw, (ar, ac)


def _swatch(rng, shape, bg, inks):
    h, w = shape
    yy, xx = np.indices(shape)
    kind = int(rng.integers(5))
    if kind == 0:
        mask = rng.random(shape) < float(rng.uniform(.2, .8))
    elif kind == 1:
        mask = (yy+xx) % 2 == int(rng.integers(2))
    elif kind == 2:
        mask = _grow(rng, shape, int(rng.integers(1, h*w)))
    elif kind == 3:
        mask = (xx % int(rng.integers(2, w+1))) == int(rng.integers(2))
    else:
        mask = (yy == int(rng.integers(h))) | (xx == int(rng.integers(w)))
    if not mask.any():
        mask.flat[int(rng.integers(mask.size))] = True
    if mask.all():
        mask.flat[int(rng.integers(mask.size))] = False
    swatch = np.full(shape, bg, dtype=np.int8)
    k = min(len(inks), int(mask.sum()), int(rng.integers(1, 4)))
    colors = rng.choice(inks, k, replace=False)
    values = rng.choice(colors, int(mask.sum()))
    values[:k] = colors
    rng.shuffle(values)
    swatch[mask] = values
    return swatch


def _disjoint(a, b, gap=1):
    return a[1]+gap <= b[0] or b[1]+gap <= a[0] or a[3]+gap <= b[2] or b[3]+gap <= a[2]


def _candidate_targets(rng, data, reserved, desired, expanded=False):
    """Enumerate finite legal placements, then evaluate topology on a shortlist."""
    h, w = data['field'].shape
    candidates = []
    for side in ((2, 3) if expanded else (1, 2, 3)):
        rasters = {(v, c): _scaled(data, side, v, c) for v in (False, True) for c in (False, True)}
        for r in range(h-side+1):
            for c in range(w-side+1):
                target = (r, c, side)
                vr, hc = _flips(data, target)
                raster, off = rasters[vr, hc]
                top, left = r-off[0], c-off[1]
                bounds = (max(0, top), min(h, top+raster.shape[0]),
                          max(0, left), min(w, left+raster.shape[1]))
                if not all(_disjoint(bounds, b) for b in reserved):
                    continue
                clipped_box = top < 0 or left < 0 or top+raster.shape[0] > h or left+raster.shape[1] > w
                if desired == 'full' and clipped_box:
                    continue
                if desired == 'leak' and not clipped_box:
                    continue
                if bounds[1]-bounds[0] < side+1 or bounds[3]-bounds[2] < side+1:
                    continue
                candidates.append((target, raster, top, left, bounds))
    for i in rng.permutation(len(candidates))[:100]:
        target, raster, top, left, bounds = candidates[int(i)]
        wall = _place(raster, (h, w), top, left, data['background'])
        wm = wall != data['background']
        if int(wm.sum()) <= target[2]**2:
            continue
        if desired in ('leak', 'closed'):
            holes = _enclosed(wm)
            if desired == 'leak':
                potential = _place(_local_holes(raster != data['background']), (h, w), top, left, False)
                if not np.any(potential & ~holes):
                    continue
            elif not holes.any():
                continue
        return target, wall, bounds
    return None


def _sample_pair(rng, period, regime):
    colors = [int(x) for x in rng.permutation(10)]
    bg, frame, anchor = colors[:3]
    wallinks = colors[3:3+int(rng.integers(1, 4))]
    if rng.random() < .3:
        wallinks[-1] = frame
    # A shared literal ink remains a wall ink or a texture ink by geometry.
    textureinks = colors[3:3+int(rng.integers(2, 7))]
    swatch = _swatch(rng, period, bg, textureinks)
    specimen = _specimen(rng, bg, anchor, wallinks)
    if specimen is None:
        return None
    raw, offset = specimen
    refside = int(rng.choice([1, 1, 1, 2]))
    source = np.repeat(np.repeat(raw, refside, 0), refside, 1)
    sep = period[0]+2
    h = int(rng.integers(max(16, source.shape[0]+3), 30-sep)) if source.shape[0]+3 < 30-sep else 0
    w = int(rng.integers(max(19, source.shape[1]+3), 31)) if source.shape[1]+3 <= 30 else 0
    if h == 0 or w == 0 or source.shape[0]+2 > h:
        return None
    sr, sc = int(rng.integers(1, h-source.shape[0])), int(rng.integers(1, w-source.shape[1]))
    field = np.full((h, w), bg, dtype=np.int8)
    field[sr:sr+source.shape[0], sc:sc+source.shape[1]] = source
    origin = (sr+offset[0]*refside, sc+offset[1]*refside, refside)
    data = dict(raw=source, offset=(offset[0]*refside, offset[1]*refside),
                origin=origin, field=field, background=bg, anchor=anchor,
                texture=swatch)
    reserved = [(sr, sr+source.shape[0], sc, sc+source.shape[1])]
    targets, rendered_walls = [], []
    n = int(rng.choice([2, 2, 3, 3, 4]))
    for index in range(n):
        if index == 0:
            desired = 'full'  # An interpretable complete example in every scene.
        elif index == 1 and regime == 'leak':
            desired = 'leak'
        else:
            desired = str(rng.choice(['full', 'closed', 'any'], p=[.35, .2, .45]))
        candidate = _candidate_targets(rng, data, reserved, desired, expanded=index == 0)
        if candidate is None:
            if index < 2:
                return None
            break
        target, wall, bounds = candidate
        targets.append(target)
        rendered_walls.append(wall)
        reserved.append(bounds)
    for r, c, side in targets:
        field[r:r+side, c:c+side] = anchor
    g = np.full((sep+1+h, w), bg, dtype=np.int8)
    g[:sep, :period[1]+2] = frame
    g[1:sep-1, 1:period[1]+1] = swatch
    g[sep] = frame
    g[sep+1:] = field
    # Construction paints generated wall rasters, then fills actual closed
    # regions. It does not call solve or its rational reconstruction renderer.
    out = g.copy()
    destfield = out[sep+1:]
    for target, wall in zip(targets, rendered_walls):
        mask = wall != bg
        destfield[mask] = wall[mask]
        holes = _enclosed(mask)
        r, c, side = target
        vr, hc = _flips(data, target)
        for y, x in np.argwhere(holes):
            yy, xx = int(y)-r, int(x)-c
            if vr:
                yy = side-1-yy
            if hc:
                xx = side-1-xx
            ink = int(swatch[yy % period[0], xx % period[1]])
            destfield[y, x] = anchor if ink == bg else ink
    pair = dict(input=g, output=out)
    # Validate observable geometry, including the noninteracting footprints.
    try:
        decoded = _decode(g)
        predicted, records = _render(decoded, records=True)
        if not np.array_equal(predicted, out):
            return None
        if regime == 'leak' and not any(rec['leaked'] for rec in records):
            return None
        if not any(rec['opaque'] and rec['transparent'] for rec in records):
            return None
    except AssertionError:
        return None
    return pair


def _fallback_pair(rng, period, scene, seen=(), avoid_specimen=None):
    """Finite construction with a closed room, an asymmetric arm, and leakage.

    The two scenes use opposite spatial relations. A six-cell-wide cavity in
    the 3x copy exposes every supported swatch period. Ninety background/anchor
    role assignments exceed the maximum fourteen existing inputs.
    """
    th, tw = period
    sep = th+2
    base = None
    for variant in range(3):
        raw = np.array([[2, 3, 3, 3, 4, 4], [3, 0, 0, 3, 0, 0],
                        [3, 0, 0, 3, 0, 0], [3, 3, 3, 3, 0, 0]], dtype=np.int8)
        if scene == 1:
            raw[3, variant+1] = 4
        swatch = np.zeros(period, dtype=np.int8)
        swatch[0, 0], swatch[1, 1] = 5, 6
        candidate = np.zeros((sep+24, 30), dtype=np.int8)
        candidate[:sep, :tw+2] = 9
        candidate[1:th+1, 1:tw+1] = swatch
        candidate[sep] = 9
        field = candidate[sep+1:]
        if scene == 0:
            field[18:22, 23:29] = raw
            field[11:14, 16:19] = 2  # full double reflection at 3x
            field[19, 3] = 2          # an arm clips, but the room stays sealed
        else:
            field[5:9, 1:7] = raw
            field[11:14, 11:14] = 2  # full unreflected 3x copy
            field[2, 4] = 2          # the reflected room opens at the north edge
        data = _decode(candidate)
        if avoid_specimen is None or _normal(data['raw']) != avoid_specimen:
            base = candidate
            break
    assert base is not None
    order = [int(x) for x in rng.permutation(10)]
    role_choices = [(order[0], order[2])]
    role_choices.extend((b, a) for b in range(10) for a in range(10) if b != a)
    for bg, anchor in role_choices:
        palette = np.empty(10, dtype=np.int8)
        palette[0], palette[2] = bg, anchor
        rest = [x for x in order if x not in (bg, anchor)]
        palette[[x for x in range(10) if x not in (0, 2)]] = rest
        inp = palette[base]
        if _key(inp) not in seen:
            return {'input': inp, 'output': solve(inp)}
    raise AssertionError('finite fallback role space exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    periods = [(h, w) for h in range(2, 5) for w in range(2, 6)]
    ids = rng.permutation(len(periods))
    teaching_periods = [periods[int(i)] for i in ids[:2]]
    test_periods = [periods[int(i)] for i in ids[2:]]
    # Select two varied demonstrations that jointly defeat every named shortcut.
    training, covers = [], []
    for period, regime in zip(teaching_periods, ('ordinary', 'leak')):
        found = None
        for _ in range(160):
            candidate = _sample_pair(rng, period, regime)
            if candidate is None:
                continue
            cover = _evidence(candidate)
            if not training:
                needed = _EVIDENCE-{'enclosure_after_clip', 'cavity_gate',
                                    'always_horizontal', 'always_vertical', 'field_center_reflection'}
                if not needed <= cover:
                    continue
            elif not _EVIDENCE <= covers[0] | cover:
                continue
            found = candidate
            covers.append(cover)
            break
        if found is None:
            avoid = _normal(_decode(training[0]['input'])['raw']) if training else None
            found = _fallback_pair(rng, period, int(bool(training)),
                                   {_key(p['input']) for p in training}, avoid)
            cover = _evidence(found)
            assert (_EVIDENCE-{'enclosure_after_clip', 'cavity_gate', 'always_horizontal',
                               'always_vertical', 'field_center_reflection'} <= cover
                    if not training else _EVIDENCE <= covers[0] | cover)
            covers.append(cover)
        training.append(found)
    puzzle = dict(train=training, test=[])
    seen = {_key(p['input']) for p in training}
    for split, count in (('train', train_count), ('test', test_count)):
        while len(puzzle[split]) < count:
            choices = teaching_periods if split == 'train' else test_periods
            period = choices[int(rng.integers(len(choices)))]
            regime = str(rng.choice(['ordinary', 'leak'], p=[.55, .45]))
            for _ in range(160):
                pair = _sample_pair(rng, period, regime)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                pair = _fallback_pair(rng, period, int(rng.integers(2)), seen)
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, evidence, periods = set(), set(), {'train': set(), 'test': set()}
    specimens = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                a = np.asarray(grid)
                assert a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
                assert all(1 <= n <= 30 for n in a.shape) and np.all((0 <= a) & (a <= 9))
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            data = _decode(inp)
            expected, records = _render(data, records=True)
            assert np.array_equal(out, expected), 'incorrect chamber completion or texture'
            sep = data['separator']
            assert np.array_equal(inp[:sep+1], out[:sep+1]), 'header changed'
            r0, r1, c0, c1 = data['source_bounds']
            assert np.array_equal(inp[sep+1+r0:sep+1+r1, c0:c1], out[sep+1+r0:sep+1+r1, c0:c1])
            bounds = [data['source_bounds']] + [rec['bounds'] for rec in records]
            assert all(_disjoint(a, b) for i, a in enumerate(bounds) for b in bounds[i+1:])
            assert all(rec['visible_wall'] > rec['target'][2]**2 for rec in records)
            assert any(rec['holes'] and rec['opaque'] and rec['transparent'] for rec in records)
            periods[split].add(data['texture'].shape)
            if split == 'train':
                evidence |= _evidence(pair, data)
                specimens.add(_normal(data['raw']))
    assert _EVIDENCE <= evidence, 'missing teaching evidence: ' + str(_EVIDENCE-evidence)
    assert len(periods['train']) == 2 and periods['train'].isdisjoint(periods['test'])
    assert len(specimens) >= 2


def _puzzle_key(puzzle):
    return tuple((_key(pair['input']), _key(pair['output']))
                 for split in ('train', 'test') for pair in puzzle[split])


def self_test():
    global _sample_pair
    stats = dict(seeds=42, pairs=0)
    shapes, roles, periods, scales, signatures, motifs, textures = (set() for _ in range(7))
    directions, counts, cavity_counts = set(), set(), set()
    leaked = retained_clipped = reductions = rectangular = shared = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        if seed < 3:
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                stats['pairs'] += 1
                data = _decode(pair['input'])
                out, records = _render(data, records=True)
                assert np.array_equal(solve(pair['input']), out)
                shapes.add(pair['input'].shape)
                roles.add((data['background'], data['frame'], data['anchor']))
                periods.add(data['texture'].shape)
                motifs.add(_normal(data['raw']))
                textures.add(_normal(data['texture']))
                counts.add(len(records))
                cavity_counts.add(len(_components(_local_holes(data['raw'] != data['background']))))
                rectangular += data['texture'].shape[0] != data['texture'].shape[1]
                shared += bool((set(np.unique(data['texture']))-{data['background']}) &
                               (set(np.unique(data['raw']))-{data['background'], data['anchor']}))
                for rec in records:
                    side, a = rec['target'][2], data['origin'][2]
                    divisor = int(np.gcd(side, a))
                    scales.add((side//divisor, a//divisor))
                    directions.add(rec['flips'])
                    reductions += side < a
                    leaked += rec['leaked'] > 0
                    retained_clipped += rec['visible_wall'] < rec['total_wall'] and rec['holes'] > 0
    assert len(signatures) == 42 and len(shapes) >= 25
    assert len(roles) >= 100 and len(motifs) >= 100 and len(textures) >= 70
    assert len(periods) == 12 and len(scales) >= 5 and reductions > 0
    assert directions == {(False, False), (False, True), (True, False), (True, True)}
    assert {2, 3} <= counts and {1, 2} <= cavity_counts
    assert leaked >= 42 and retained_clipped > 0 and rectangular > 0 and shared > 0
    # All legal explicit counts, and the optional partially specified interface.
    for a in range(2, 11):
        for b in range(1, min(a-1, 4)+1):
            p = generate_puzzle(11000+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for i, (a, b) in enumerate(((None, None), (None, 4), (2, None), (10, None))):
        generate_puzzle(22000+i, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.0, 1), (4, '1')):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # The finite fallback must establish the same rule for every header period.
    fallback_rng = np.random.default_rng(44556)
    all_periods = [(h, w) for h in range(2, 5) for w in range(2, 6)]
    for i, period in enumerate(all_periods):
        other = all_periods[(i+1) % len(all_periods)]
        first = _fallback_pair(fallback_rng, period, 0)
        second = _fallback_pair(fallback_rng, other, 1)
        test = _fallback_pair(fallback_rng, all_periods[(i+2) % len(all_periods)], i % 2)
        validate_generated_puzzle({'train': [first, second], 'test': [test]})
    seen_fallback = set()
    for _ in range(14):
        pair = _fallback_pair(np.random.default_rng(66), (4, 5), 1, seen_fallback)
        seen_fallback.add(_key(pair['input']))
    assert len(seen_fallback) == 14
    # Force budget exhaustion to exercise the public generator's fallback path,
    # including the maximum fourteen examples. Restore sampling even on failure.
    sampler = _sample_pair
    try:
        _sample_pair = lambda *args: None
        for a, b in ((2, 1), (10, 4)):
            forced = generate_puzzle(778899+a, a, b)
            assert len(forced['train']) == a and len(forced['test']) == b
    finally:
        _sample_pair = sampler
    # A small, hand-calculated regression outside the random sampler's layout.
    # A 2x source reduces to 1x north-facing copies. One loses its top wall;
    # its surviving three cavity pixels touch the boundary and remain blank.
    fixture = np.zeros((23, 22), dtype=np.int8)
    fixture[:4, :5] = 9
    fixture[1:3, 1:4] = np.array([[0, 3, 0], [4, 0, 4]])
    fixture[4] = 9
    field = fixture[5:]
    wall = np.array([[2, 7, 7, 7, 7], [7, 0, 0, 0, 7],
                     [7, 0, 0, 0, 7], [7, 7, 7, 7, 7]], dtype=np.int8)
    field[8:16, 1:11] = np.repeat(np.repeat(wall, 2, 0), 2, 1)
    field[1, 4] = field[5, 15] = 2
    predicted = solve(fixture)[5:]
    assert np.array_equal(predicted[0:2, 4:9], [[7, 0, 0, 0, 7], [2, 7, 7, 7, 7]])
    assert np.array_equal(predicted[2:6, 15:20],
                          [[7, 7, 7, 7, 7], [7, 3, 2, 2, 7],
                           [7, 2, 4, 4, 7], [2, 7, 7, 7, 7]])
    assert np.array_equal(predicted[8:16, 1:11], field[8:16, 1:11])
    # Literal color equivariance: role recovery must not depend on black or 9.
    p = generate_puzzle(33000, 2, 1)
    permutation = np.random.default_rng(99).permutation(10)
    for pair in p['train']+p['test']:
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    broken = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged output accepted')
    stats.update(grid_shapes=len(shapes), role_assignments=len(roles), specimen_structures=len(motifs),
                 texture_structures=len(textures), periods=len(periods), scale_ratios=sorted(scales),
                 target_counts=sorted(counts), cavity_counts=sorted(cavity_counts),
                 leaking_copies=leaked, clipped_copies_with_closed_rooms=retained_clipped,
                 reduced_copies=reductions, shared_wall_texture_inks=shared)
    return stats


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
