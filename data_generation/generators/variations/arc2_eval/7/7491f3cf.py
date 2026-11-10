"""Marker-selected regional compositing in four aligned panels."""

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)

def _validate_count(value, name, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")

def _resolve_counts(rng, num_train, num_test):
    _validate_count(num_train, "num_train", 2, 10)
    _validate_count(num_test, "num_test", 1, 4)

    if num_train is not None:
        num_train = int(num_train)
    if num_test is not None:
        num_test = int(num_test)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        legal = [
            (train_count, test_count)
            for test_count in range(1, 5)
            for train_count in range(max(2, test_count + 1), 11)
        ]
        return legal[int(rng.integers(0, len(legal)))]

    if num_train is None:
        low = max(2, num_test + 1)
        return int(rng.integers(low, 11)), num_test

    high = min(4, num_train - 1)
    return num_train, int(rng.integers(1, high + 1))

def _components(mask, diagonal):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)

    if diagonal:
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    result = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            seen[r, c] = True
            stack = [(r, c)]
            cells = []

            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))

                for dr, dc in directions:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            result.append(cells)

    return result

def _cells_to_mask(shape, cells):
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask

def _region_masks(cut):
    components = _components(~cut, diagonal=False)
    _require(
        len(components) == 2,
        "the cut must induce exactly two regions",
    )
    return [
        _cells_to_mask(cut.shape, cells)
        for cells in components
    ]

def _marked_and_other(cut, marker):
    regions = _region_masks(cut)

    if regions[0][marker]:
        return regions[0], regions[1]

    _require(
        regions[1][marker],
        "marker is not inside a region",
    )
    return regions[1], regions[0]

def _compose(
    panel_a,
    panel_b,
    background,
    cut,
    marked,
    other,
):
    result = np.full(
        panel_a.shape,
        background,
        dtype=np.int8,
    )

    result[marked] = panel_a[marked]
    result[other] = panel_b[other]

    b_cut_foreground = cut & (panel_b != background)
    a_cut_foreground = cut & (panel_a != background)

    result[b_cut_foreground] = panel_b[b_cut_foreground]
    result[a_cut_foreground] = panel_a[a_cut_foreground]

    return result

def _pack_panels(panels, frame_color):
    h, w = panels[0].shape

    grid = np.full(
        (h + 2, 4 * w + 5),
        frame_color,
        dtype=np.int8,
    )

    for index, panel in enumerate(panels):
        start = 1 + index * (w + 1)
        grid[1:h + 1, start:start + w] = panel

    return grid

def _grid_key(grid):
    normalized = np.ascontiguousarray(
        grid,
        dtype=np.int64,
    )
    return tuple(grid.shape), normalized.tobytes()

def _puzzles_equal(first, second):
    if set(first.keys()) != {
        "train",
        "test",
    }:
        return False

    if set(second.keys()) != {
        "train",
        "test",
    }:
        return False

    for section in ("train", "test"):
        if (
            len(first[section])
            != len(second[section])
        ):
            return False

        for pair_a, pair_b in zip(
            first[section],
            second[section],
        ):
            for name in ("input", "output"):
                if not np.array_equal(
                    pair_a[name],
                    pair_b[name],
                ):
                    return False

    return True

def _puzzle_key(puzzle):
    return tuple(
        (
            section,
            tuple(
                (
                    _grid_key(pair["input"]),
                    _grid_key(pair["output"]),
                )
                for pair in puzzle[section]
            ),
        )
        for section in ("train", "test")
    )


def _path_graph(cut):
    """Digital path edges, without diagonal shortcuts across an orthogonal bend."""
    cells = {tuple(p) for p in np.argwhere(cut)}
    graph = {p: [] for p in cells}
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = (r + dr, c + dc)
                if (dr or dc) and q in cells:
                    if dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells):
                        continue
                    graph[r, c].append(q)
    return graph


def _marker_sites(cut, region):
    near = cut.copy()
    h, w = cut.shape
    for r, c in np.argwhere(cut):
        near[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = True
    return [tuple(map(int, p)) for p in np.argwhere(region & ~near)]


def _cut_geometry(cut):
    graph = _path_graph(cut)
    _require(len(graph) >= 3, 'cut is too short')
    ends = [p for p, neighbors in graph.items() if len(neighbors) == 1]
    _require(len(ends) == 2 and all(1 <= len(n) <= 2 for n in graph.values()),
             'cut must be a path without branches or loops')
    h, w = cut.shape
    border = lambda p: p[0] in (0, h-1) or p[1] in (0, w-1)
    _require(all(border(p) for p in ends), 'cut endpoints must meet the boundary')
    _require(all(not border(p) for p in graph if p not in ends),
             'only cut endpoints may meet the boundary')
    order, previous, current = [], None, min(ends)
    while current is not None and current not in order:
        order.append(current)
        following = [p for p in graph[current] if p != previous]
        previous, current = current, following[0] if following else None
    _require(len(order) == len(graph), 'disconnected path or extra loop')
    regions = _region_masks(cut)
    sites = [_marker_sites(cut, region) for region in regions]
    _require(all(sites), 'both regions need an isolated marker position')
    return regions, sites, order


def _line(p, q):
    # Bresenham's digital segment, including corner-to-corner diagonals.
    r, c = p
    r1, c1 = q
    dr, dc = abs(r1-r), abs(c1-c)
    sr, sc = (1 if r < r1 else -1), (1 if c < c1 else -1)
    error, points = dc-dr, []
    while True:
        points.append((r, c))
        if (r, c) == (r1, c1):
            return points
        twice = 2*error
        if twice > -dr:
            error -= dr
            c += sc
        if twice < dc:
            error += dc
            r += sr


def _sample_cut(rng, h, w, family, flip=None):
    for attempt in range(160):
        cut = np.zeros((h, w), dtype=bool)
        if family == 'horizontal':
            cut[int(rng.integers(2, h-2)), :] = True
        elif family == 'vertical':
            cut[:, int(rng.integers(2, w-2))] = True
        elif family == 'diagonal':
            ports = ([(0, c) for c in range(w)] + [(h-1, c) for c in range(w)]
                     + [(r, 0) for r in range(1, h-1)] + [(r, w-1) for r in range(1, h-1)])
            if h == w and rng.random() < .4:
                p, q = (0, 0), (h-1, w-1)
            else:
                p, q = [ports[int(i)] for i in rng.choice(len(ports), 2, replace=False)]
            if p[0] == q[0] or p[1] == q[1]:
                continue
            for r, c in _line(p, q):
                cut[r, c] = True
        elif family == 'elbow':
            r = int(rng.integers(2, h-2))
            c = int(rng.integers(1, w-2))
            cut[:r+1, c] = True
            cut[r, c:] = True
        elif family == 'stepped':
            # Alternating vertical runs and horizontal steps; two to ten bends.
            across = rng.random() < .3
            ch,cw = (w,h) if across else (h,w)
            cut = np.zeros((ch,cw),dtype=bool)
            rows = list(range(2, ch-2, 2))
            count = int(rng.integers(1, min(5, len(rows))+1))
            rows = sorted(int(x) for x in rng.choice(rows, count, replace=False))
            col, last = int(rng.integers(1, cw-1)), 0
            for row in rows:
                cut[last:row+1, col] = True
                choices = [c for c in range(1, cw-1) if c != col]
                next_col = choices[int(rng.integers(len(choices)))]
                cut[row, min(col, next_col):max(col, next_col)+1] = True
                col, last = next_col, row
            cut[last:, col] = True
            if across:
                cut = cut.T.copy()
        else:
            raise ValueError('unknown cut family')
        transform = int(rng.integers(4)) if flip is None else flip
        if transform & 1:
            cut = cut[:, ::-1].copy()
        if transform & 2:
            cut = cut[::-1, :].copy()
        try:
            _cut_geometry(cut)
            return cut
        except AssertionError:
            pass
    # Valid constructive fallbacks retain the requested family and dimensions.
    cut = np.zeros((h, w), dtype=bool)
    if family == 'horizontal':
        cut[h//2, :] = True
    elif family == 'vertical':
        cut[:, w//2] = True
    elif family == 'diagonal':
        for p in _line((0, 1), (h-1, w-2)):
            cut[p] = True
    elif family == 'elbow':
        cut[:h//2+1, 1] = True
        cut[h//2, 1:] = True
    else:
        cut[:h//2+1, 1] = True
        cut[h//2, 1:w-1] = True
        cut[h//2:, w-2] = True
    transform = int(rng.integers(4)) if flip is None else flip
    if transform & 1:
        cut = cut[:, ::-1].copy()
    if transform & 2:
        cut = cut[::-1, :].copy()
    _cut_geometry(cut)
    return cut


def _sample_size(rng, family, richness):
    if family == 'diagonal' and rng.random() < .35:
        side = int(rng.integers(5, 7))
        return side, side
    w = int(rng.integers(3, 7))
    if family in ('vertical', 'diagonal', 'stepped'):
        w = max(5, w)
    elif family == 'elbow':
        w = max(4, w)
    if family == 'vertical' and rng.random() < .3:
        return int(rng.integers(3, 6)), w
    low = 7 if family == 'stepped' else 5
    if rng.random() < (.12, .25, .42)[richness]:
        return int(rng.integers(max(11, low), 29)), w
    return int(rng.integers(low, max(low+1, 11))), w


def _motif(rng, h, w, style):
    """Clean procedural shapes stay clean. Irregular masks are a separate style."""
    rr, cc = np.indices((h, w))
    r, c = int(rng.integers(h)), int(rng.integers(w))
    mask = np.zeros((h, w), dtype=bool)
    if style == 'cross':
        if rng.random() < .6:
            r, c = h//2, w//2
        ar, ac = int(rng.integers(1, h+1)), int(rng.integers(1, w+1))
        mask[max(0,r-ar):r+ar+1, c] = True
        mask[r, max(0,c-ac):c+ac+1] = True
    elif style in ('solid', 'hollow'):
        r0, r1 = sorted(int(x) for x in rng.choice(h, 2, replace=False))
        c0, c1 = sorted(int(x) for x in rng.choice(w, 2, replace=False))
        mask[r0:r1+1, c0:c1+1] = True
        if style == 'hollow':
            mask[r0+1:r1, c0+1:c1] = False
    elif style == 'diamond':
        r, c = h//2, w//2
        radius = int(rng.integers(1, min(h,w)//2+1))
        distance = abs(rr-r)+abs(cc-c)
        mask = distance == radius if rng.random() < .5 else distance <= radius
    elif style == 'x':
        # A centered digital X, with no added noise.
        cols = np.rint(np.linspace(0, w-1, h)).astype(int)
        mask[np.arange(h), cols] = True
        mask[np.arange(h), w-1-cols] = True
    elif style == 'checker':
        period = int(rng.integers(2, 5))
        mask = (rr+cc+int(rng.integers(period))) % period == 0
    elif style == 'stripes':
        period = int(rng.integers(2, 5))
        axis = rr if rng.random() < .5 else cc
        mask = (axis+int(rng.integers(period))) % period == 0
    elif style == 'segments':
        for _ in range(int(rng.integers(1, 5))):
            if rng.random() < .5:
                r = int(rng.integers(h))
                a, b = sorted(int(x) for x in rng.choice(w, 2, replace=False))
                mask[r, a:b+1] = True
            else:
                c = int(rng.integers(w))
                a, b = sorted(int(x) for x in rng.choice(h, 2, replace=False))
                mask[a:b+1, c] = True
    elif style == 'wrapping':
        period, slope = int(rng.integers(2, w+2)), int(rng.choice([-2,-1,1,2]))
        mask = (cc-slope*rr-c) % period == 0
    elif style == 'brackets':
        dr, dc = int(rng.integers(2,h+1)), int(rng.integers(2,w+1))
        mask[0, :dc] = True
        mask[:dr, 0] = True
        if rng.random() < .65:
            mask |= mask[::-1, ::-1]
        if rng.random() < .5:
            mask = mask[:, ::-1].copy()
    elif style == 'symmetric':
        mask = rng.random((h,w)) < float(rng.uniform(.08,.3))
        if rng.random() < .5:
            mask |= mask[:, ::-1]
        else:
            mask |= mask[::-1, :]
        if rng.random() < .35:
            mask |= mask[::-1, ::-1]
    elif style == 'components':
        for _ in range(int(rng.integers(2,6))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            mask[r:min(h,r+int(rng.integers(1,4))), c:min(w,c+int(rng.integers(1,3)))] = True
    else:
        mask = rng.random((h,w)) < float(rng.uniform(.07,.8))
    return mask


_STYLES = ('cross', 'solid', 'hollow', 'diamond', 'x', 'checker', 'stripes',
           'segments', 'wrapping', 'brackets', 'symmetric', 'components', 'irregular')
_FAMILIES = ('horizontal', 'vertical', 'diagonal', 'elbow', 'stepped')
_EVIDENCE = ('marked_uses_a', 'marked_ignores_b', 'other_uses_b', 'other_ignores_a',
             'cut_a_precedence', 'cut_keeps_b', 'cut_keeps_a', 'cut_can_be_blank')


def _contained_shape(rng, region):
    # Grow a rectangle wholly inside a region instead of erasing seam pixels.
    r0, c0 = [int(x) for x in np.argwhere(region)[int(rng.integers(region.sum()))]]
    r1, c1 = r0+1, c0+1
    h, w = region.shape
    for _ in range(h+w):
        options = [(max(0,r0-1),r1,c0,c1), (r0,min(h,r1+1),c0,c1),
                   (r0,r1,max(0,c0-1),c1), (r0,r1,c0,min(w,c1+1))]
        options = [q for q in options if q != (r0,r1,c0,c1) and np.all(region[q[0]:q[1],q[2]:q[3]])]
        if not options:
            break
        r0,r1,c0,c1 = options[int(rng.integers(len(options)))]
        if rng.random() < .2:
            break
    mask = np.zeros(region.shape, dtype=bool)
    mask[r0:r1,c0:c1] = True
    if rng.random() < .5:
        mask[r0+1:r1-1,c0+1:c1-1] = False
    return mask


def _palette(rng, parameters):
    base = list(parameters['palette'])
    mode = parameters['palette_mode']
    if mode == 'fixed' or (mode == 'mixed' and rng.random() < .25):
        return tuple(base)
    if mode == 'partial' or rng.random() < .5:
        count = int(rng.integers(1, 5))
        roles = [int(x) for x in rng.choice(5, count, replace=False)]
        available = [v for v in range(10) if v not in [base[i] for i in range(5) if i not in roles]]
        values = rng.choice(available, count, replace=False)
        for i, value in zip(roles, values):
            base[i] = int(value)
        return tuple(base)
    # Full reassignment changes each role, without requiring disjoint color sets.
    for _ in range(30):
        colors = tuple(int(x) for x in rng.choice(10, 5, replace=False))
        if all(a != b for a,b in zip(colors,base)):
            return colors
    return tuple((x+1) % 10 for x in base)


def _paint(rng, mask, bg, colors):
    panel = np.full(mask.shape, bg, dtype=np.int8)
    panel[mask] = colors[0]
    if len(colors) > 1:
        rr, cc = np.indices(mask.shape)
        stripes = (rr+cc+int(rng.integers(2))) % 2 == 0
        panel[mask & stripes] = colors[1]
    return panel


def _make_pair(cut, marker, a, b, frame, bg, control):
    ctrl = np.full(cut.shape, bg, dtype=np.int8)
    ctrl[cut] = control
    ctrl[marker] = control
    blank = np.full(cut.shape, bg, dtype=np.int8)
    inp = _pack_panels([ctrl, a, b, blank], frame)
    marked, other = _marked_and_other(cut, marker)
    result = _compose(a, b, bg, cut, marked, other)
    out = inp.copy()
    out[1:-1, 3*(cut.shape[1]+1)+1:-1] = result
    return {'input': inp, 'output': out}


def _new_pair(rng, parameters, family, side, flip=None, multi=False):
    h, w = _sample_size(rng, family, parameters['richness'])
    cut = _sample_cut(rng, h, w, family, flip)
    regions, sites, _ = _cut_geometry(cut)
    marker = sites[side][int(rng.integers(len(sites[side])))]
    frame,bg,control,ca,cb = _palette(rng, parameters)
    mode = float(rng.random())
    for _ in range(100):
        if mode < .16:
            mask_a = _contained_shape(rng, regions[int(rng.integers(2))])
            mask_b = _contained_shape(rng, regions[int(rng.integers(2))])
        elif mode < .27:
            rr,cc = np.indices((h,w))
            axis = rr+cc if rng.random() < .5 else (rr if rng.random() < .5 else cc)
            mask_a = (axis % 2 == 0)
            mask_b = ~mask_a
        else:
            # Every style is available in either split and at every richness.
            styles = list(_STYLES)
            if parameters['richness'] == 0:
                styles += list(_STYLES[:7])*2
            mask_a = _motif(rng,h,w,styles[int(rng.integers(len(styles)))])
            mask_b = _motif(rng,h,w,styles[int(rng.integers(len(styles)))])
        if (0 < mask_a.sum() < h*w and 0 < mask_b.sum() < h*w
                and (np.any(mask_a & (regions[side] | cut))
                     or np.any(mask_b & (regions[1-side] | cut)))):
            break
    else:
        rr,cc = np.indices((h,w))
        mask_a,mask_b = (rr+cc)%2 == 0, rr%3 == 0
    colors_a,colors_b = [ca],[cb]
    if multi:
        extras = [int(x) for x in rng.permutation(10) if x not in (frame,bg,control,ca,cb)]
        colors_a.append(extras[0])
        colors_b.append(extras[1])
    a,b = _paint(rng,mask_a,bg,colors_a),_paint(rng,mask_b,bg,colors_b)
    return _make_pair(cut,marker,a,b,frame,bg,control)


def _decode(inp):
    h, width = inp.shape
    _require((width-5)%4 == 0 and 3 <= h <= 30, 'invalid four-panel dimensions')
    w = (width-5)//4
    _require(1 <= w <= 6, 'invalid panel width')
    frame = int(inp[0,0])
    _require(np.all(inp[[0,-1],:] == frame), 'broken horizontal frame')
    _require(np.all(inp[:,::w+1] == frame), 'broken frame or dividers')
    panels = [inp[1:-1,1+i*(w+1):1+i*(w+1)+w] for i in range(4)]
    bg = int(panels[3][0,0])
    _require(np.all(panels[3] == bg), 'destination must start blank')
    values = set(map(int,np.unique(panels[0]))) - {bg}
    _require(len(values) == 1, 'control needs one foreground color')
    control = next(iter(values))
    _require(len({frame,bg,control}) == 3, 'frame, background and control must differ')
    ac = set(map(int,np.unique(panels[1]))) - {bg}
    bc = set(map(int,np.unique(panels[2]))) - {bg}
    _require(ac and bc and not (ac | bc) & {frame,control}, 'unreadable source roles')
    _require(bg in panels[1] and bg in panels[2], 'shared source background is missing')
    components = sorted(_components(panels[0] != bg, True), key=len)
    _require(len(components) == 2 and len(components[0]) == 1 and len(components[1]) > 1,
             'control must have one isolated marker and one cut')
    marker = components[0][0]
    cut = _cells_to_mask(panels[0].shape,components[1])
    regions,sites,order = _cut_geometry(cut)
    side = 0 if regions[0][marker] else 1
    _require(marker in sites[side], 'marker touches the cut')
    return {'panels':panels, 'panel_a':panels[1], 'panel_b':panels[2],
            'background':bg, 'frame':frame, 'control':control, 'colors_a':ac, 'colors_b':bc,
            'cut':cut, 'marker':marker, 'marked':regions[side], 'other':regions[1-side],
            'side':side, 'sites':sites, 'path':order, 'shape':inp.shape}


def _predict_from_input(inp):
    """Cellwise validation path, deliberately independent of the vector compositor."""
    info = _decode(inp)
    out = inp.copy()
    bg, a, b, cut = info['background'],info['panel_a'],info['panel_b'],info['cut']
    start = 3*(cut.shape[1]+1)+1
    for r in range(cut.shape[0]):
        for c in range(cut.shape[1]):
            if cut[r,c]:
                value = a[r,c] if a[r,c] != bg else b[r,c]
            elif info['marked'][r,c]:
                value = a[r,c]
            else:
                value = b[r,c]
            out[r+1,start+c] = value
    return out


def _inspect_pair(pair, expected_regime=None):
    _require(pair['input'].shape == pair['output'].shape, 'input/output shapes differ')
    info = _decode(pair['input'])
    _require(np.array_equal(_predict_from_input(pair['input']),pair['output']),
             'incorrect composite or changed frame/control/source')
    w = info['cut'].shape[1]
    info['result'] = pair['output'][1:-1,3*(w+1)+1:-1]
    return info


def _evidence_flags(info):
    a,b,bg = info['panel_a'],info['panel_b'],info['background']
    am,bm = a != bg,b != bg
    cut,m,o = info['cut'],info['marked'],info['other']
    return dict(zip(_EVIDENCE, map(bool, (
        np.any(m & am), np.any(m & ~am & bm), np.any(o & bm), np.any(o & am & ~bm),
        np.any(cut & am & bm & (a != b)), np.any(cut & ~am & bm),
        np.any(cut & am & ~bm), np.any(cut & ~am & ~bm)))))


def _flags(pair):
    return {name for name,present in _evidence_flags(_decode(pair['input'])).items() if present}


def _multicolor_evidence(info):
    bg = info['background']
    a,b = info['panel_a'],info['panel_b']
    av = a[info['marked'] | info['cut']]
    bv = b[info['other'] | (info['cut'] & (a == bg))]
    return len(set(map(int,av))-{bg}) >= 2 and len(set(map(int,bv))-{bg}) >= 2


def _opposite(pair, rng):
    info = _decode(pair['input'])
    sites = info['sites'][1-info['side']]
    marker = sites[int(rng.integers(len(sites)))]
    return _make_pair(info['cut'],marker,info['panel_a'],info['panel_b'],
                      info['frame'],info['background'],info['control'])


def _evidence_fallback(pair):
    """Bounded last resort for a missing training convention, never regular noise."""
    d = _decode(pair['input'])
    h,w = d['cut'].shape
    bg = d['background']
    a,b = np.full((h,w),bg,dtype=np.int8),np.full((h,w),bg,dtype=np.int8)
    ca,cb = min(d['colors_a']),min(d['colors_b'])
    # Four cut states and complementary region evidence in one sparse diagram.
    seam = d['path']
    _require(len(seam) >= 4, 'evidence cut must have at least four cells')
    a[seam[0]],b[seam[0]],b[seam[1]],a[seam[2]] = ca,cb,cb,ca
    for region in (d['marked'],d['other']):
        cells = [tuple(p) for p in np.argwhere(region)]
        a[cells[0]],b[cells[-1]] = ca,cb
    if len(d['colors_a']) > 1 and len(d['colors_b']) > 1:
        cells = [tuple(p) for p in np.argwhere(d['marked'])]
        a[cells[1]] = max(d['colors_a'])
        cells = [tuple(p) for p in np.argwhere(d['other'])]
        b[cells[-2]] = max(d['colors_b'])
    return _make_pair(d['cut'],d['marker'],a,b,d['frame'],bg,d['control'])


def _plan(rng, train_count, test_count):
    richness = int(rng.integers(3))
    regime = str(rng.choice(['mixed','same_family','straight_to_bent','more_bends','held_orientation']))
    params = {'richness':richness, 'regime':regime,
              'palette':tuple(int(x) for x in rng.choice(10,5,replace=False)),
              'palette_mode':str(rng.choice(['fixed','partial','mixed'],p=[.2,.25,.55])),
              'multi':bool(rng.random() < .25)}
    if regime == 'straight_to_bent':
        train = [str(rng.choice(_FAMILIES[:3])) for _ in range(train_count)]
        test = ['elbow']*test_count
    elif regime == 'more_bends':
        train = [str(rng.choice(_FAMILIES[:4])) for _ in range(train_count)]
        train[0] = 'elbow'
        test = ['stepped']*test_count
    elif regime == 'held_orientation':
        train,test = ['elbow']*train_count,['elbow']*test_count
    elif regime == 'same_family':
        family = str(rng.choice(_FAMILIES))
        train,test = [family]*train_count,[family]*test_count
    else:
        train = [str(rng.choice(_FAMILIES)) for _ in range(train_count)]
        train[0],train[-1] = str(rng.choice(_FAMILIES[:3])),str(rng.choice(_FAMILIES[3:]))
        test = [str(rng.choice(_FAMILIES)) for _ in range(test_count)]
    return params,train,test


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt,nq = _resolve_counts(rng,num_train,num_test)
    params,train_families,test_families = _plan(rng,nt,nq)
    puzzle = {'train':[], 'test':[]}
    seen,covered = set(),set()
    first_side = int(rng.integers(2))
    twin = nt >= 3 and rng.random() < .4
    for index,family in enumerate(train_families):
        side = first_side if index == 0 else (1-first_side if index == 1 else int(rng.integers(2)))
        flip = int(rng.integers(2)) if params['regime'] == 'held_orientation' else None
        if index == 1 and twin:
            pair = _opposite(puzzle['train'][0],rng)
        else:
            missing = set(_EVIDENCE)-covered if index == nt-1 else set()
            # Palette and multicolor copying are visible, not hidden conventions.
            multi = params['multi'] and (index in (0,nt-1) or rng.random() < .45)
            for attempt in range(256):
                pair = _new_pair(rng,params,family,side,flip,multi)
                if _grid_key(pair['input']) in seen or _is_source_pair(pair):
                    continue
                if index < 2:
                    d = _decode(pair['input'])
                    if not np.any((d['panel_a'] != d['panel_b']) & ~d['cut']):
                        continue
                if multi and index == 0 and not _multicolor_evidence(_decode(pair['input'])):
                    continue
                if twin and index == 0:
                    opposite = _opposite(pair,rng)
                    if np.array_equal(opposite['input'],opposite['output']):
                        continue
                if missing <= _flags(pair):
                    break
            else:
                # A known roomy size makes this fallback bounded for every family.
                d = _decode(pair['input'])
                cut = _sample_cut(rng,9,6,family,flip)
                _,sites,_ = _cut_geometry(cut)
                a = np.full((9,6),d['background'],dtype=np.int8)
                b = a.copy()
                for i,color in enumerate(sorted(d['colors_a'])):
                    a[0,i] = color
                for i,color in enumerate(sorted(d['colors_b'])):
                    b[0,i] = color
                pair = _make_pair(cut,sites[side][0],a,b,d['frame'],d['background'],d['control'])
                pair = _evidence_fallback(pair)
        puzzle['train'].append(pair)
        seen.add(_grid_key(pair['input']))
        covered |= _flags(pair)
    for index,family in enumerate(test_families):
        flip = 2+int(rng.integers(2)) if params['regime'] == 'held_orientation' else None
        for attempt in range(128):
            side = int(rng.integers(2))
            pair = _new_pair(rng,params,family,side,flip,params['multi'] and rng.random() < .6)
            # Occasional exact source reuse with a new visible control.
            if params['regime'] in ('mixed','same_family') and rng.random() < .18:
                template = puzzle['train'][int(rng.integers(nt))]
                d = _decode(template['input'])
                h,w = d['cut'].shape
                cut = np.flip(d['cut'],axis=int(rng.integers(2))).copy()
                _,sites,_ = _cut_geometry(cut)
                marker = sites[side][int(rng.integers(len(sites[side])))]
                pair = _make_pair(cut,marker,d['panel_a'],d['panel_b'],d['frame'],d['background'],d['control'])
            if (_grid_key(pair['input']) not in seen and not _is_source_pair(pair)
                    and not np.array_equal(pair['input'],pair['output'])):
                break
        else:
            # A new frame/background assignment separates even tiny repeated layouts.
            pair = _new_pair(rng,params,family,side,flip,False)
            for shift in range(1,10):
                candidate = {key:(value+shift).astype(np.int8)%10 for key,value in pair.items()}
                if _grid_key(candidate['input']) not in seen and not _is_source_pair(candidate):
                    pair = candidate
                    break
            else:
                raise AssertionError('unable to separate test examples')
        seen.add(_grid_key(pair['input']))
        puzzle['test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


# Exact staged source fixtures: regression evidence, never sampling templates.
_SOURCE_ROWS = (
    (('5555555555555555555555555', '5644465244425443445444445', '5444645424245443445444445', '5446445442445333335444445', '5464445424245443445444445', '5644445244425443445444445', '5555555555555555555555555'), ('5555555555555555555555555', '5644465244425443445244425', '5444645424245443445424245', '5446445442445333335442335', '5464445424245443445423445', '5644445244425443445243445', '5555555555555555555555555')),
    (('1111111111111111111111111', '1884881288821883881888881', '1884881828281883881888881', '1484881882881333331888881', '1884881828281883881888881', '1884881288821883881888881', '1111111111111111111111111'), ('1111111111111111111111111', '1884881288821883881283881', '1884881828281883881823881', '1484881882881333331882331', '1884881828281883881823881', '1884881288821883881283881', '1111111111111111111111111')),
    (('6666666666666666666666666', '6799996999446919196999996', '6979996999446191916999996', '6997996999996919196999996', '6999796449996191916999996', '6799976449996919196999996', '6666666666666666666666666'), ('6666666666666666666666666', '6799996999446919196919196', '6979996999446191916991916', '6997996999996919196999196', '6999796449996191916449916', '6799976449996919196449996', '6666666666666666666666666')),
    (('4444444444444444444444444', '4001004200024003004000004', '4000004020204003004000004', '4111114002004333334000004', '4000004020204003004000004', '4000004200024003004000004', '4444444444444444444444444'), ('4444444444444444444444444', '4001004200024003004200024', '4000004020204003004020204', '4111114002004333334332334', '4000004020204003004003004', '4000004200024003004003004', '4444444444444444444444444')),
    (('2222222222222222222222222', '2558582535352775772555552', '2558552335332755572555552', '2558882555552555552555552', '2555552335332755572555552', '2555552535352775772555552', '2222222222222222222222222'), ('2222222222222222222222222', '2558582535352775772775352', '2558552335332755572755332', '2558882555552555552555552', '2555552335332755572755572', '2555552535352775772775772', '2222222222222222222222222')),
)


def _source_pairs():
    return [{key:np.array([[int(c) for c in row] for row in rows],dtype=np.int8)
             for key,rows in zip(('input','output'),pair)} for pair in _SOURCE_ROWS]


def _is_source_pair(pair):
    if pair['input'].shape != (7,25):
        return False
    return any(np.array_equal(pair['input'],s['input']) and np.array_equal(pair['output'],s['output'])
               for s in _source_pairs())


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle,dict) and set(puzzle) == {'train','test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s],list) for s in puzzle), 'splits must be lists')
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), 'invalid pair counts')
    seen,seen_pairs,coverage,meaningful_sides = set(),set(),set(),set()
    multicolor_taught = False
    needs_multicolor = False
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {'input','output'}, 'invalid pair keys')
            for grid in pair.values():
                _require(isinstance(grid,np.ndarray) and grid.ndim == 2, 'grid must be 2D NumPy array')
                _require(np.issubdtype(grid.dtype,np.integer), 'grid must be integer')
                _require(all(1 <= n <= 30 for n in grid.shape), 'invalid grid dimensions')
                _require(np.all((0 <= grid) & (grid <= 9)), 'invalid ARC colors')
            key = _grid_key(pair['input'])
            pair_key = (key,_grid_key(pair['output']))
            _require(key not in seen and pair_key not in seen_pairs, 'duplicate example')
            _require(not _is_source_pair(pair), 'exact copy of a source pair')
            _require(not np.array_equal(pair['input'],pair['output']), 'empty generated result')
            seen.add(key)
            seen_pairs.add(pair_key)
            d = _inspect_pair(pair, 'source_train' if split == 'train' else 'target_test')
            if split == 'train':
                coverage |= {k for k,v in _evidence_flags(d).items() if v}
                multicolor_taught |= _multicolor_evidence(d)
                if np.any((d['panel_a'] != d['panel_b']) & ~d['cut']):
                    meaningful_sides.add(d['side'])
            else:
                needs_multicolor |= len(d['colors_a']) > 1 or len(d['colors_b']) > 1
    _require(coverage == set(_EVIDENCE), 'training does not collectively demonstrate compositing')
    _require(meaningful_sides == {0,1}, 'training must show effective selections of both regions')
    _require(not needs_multicolor or multicolor_taught, 'multicolor copying was not demonstrated')
    # Regimes change geometry and appearance, never the ordered-source or seam rule.
    # No orientation, selected side, palette or source-size restriction is imposed.


def _cut_family(d):
    path = d['path']
    rows,cols = zip(*path)
    if len(set(rows)) == 1:
        return 'horizontal'
    if len(set(cols)) == 1:
        return 'vertical'
    if path == _line(path[0],path[-1]) or path[::-1] == _line(path[-1],path[0]):
        return 'diagonal'
    steps = [(b[0]-a[0],b[1]-a[1]) for a,b in zip(path,path[1:])]
    changes = sum(a != b for a,b in zip(steps,steps[1:]))
    return 'elbow' if changes == 1 else 'stepped'


def _endpoint_sides(d):
    h,w = d['cut'].shape
    def label(p):
        r,c = p
        return ''.join(name for yes,name in ((r==0,'T'),(r==h-1,'B'),(c==0,'L'),(c==w-1,'R')) if yes)
    return '_'.join(sorted(label(p) for p in (d['path'][0],d['path'][-1])))


def _mask_key(panel,bg):
    return panel.shape, (panel != bg).tobytes()


def _features(pair):
    """Measurements reconstructed from returned grids, with no sampling metadata."""
    d = _decode(pair['input'])
    a,b,bg = d['panel_a'],d['panel_b'],d['background']
    am,bm,cut = a != bg,b != bg,d['cut']
    family = _cut_family(d)
    result = pair['output'][1:-1,3*(cut.shape[1]+1)+1:-1]
    visible_a = np.any(am & (d['marked'] | cut))
    visible_b = np.any(bm & (d['other'] | (cut & ~am)))
    steps = [(q[0]-p[0],q[1]-p[1]) for p,q in zip(d['path'],d['path'][1:])]
    return {'family':family, 'side':d['side'], 'endpoint_sides':_endpoint_sides(d),
            'panel_shape':cut.shape, 'canvas_shape':pair['input'].shape,
            'bends':sum(x != y for x,y in zip(steps,steps[1:])),
            'marker_edge':d['marker'][0] in (0,cut.shape[0]-1) or d['marker'][1] in (0,cut.shape[1]-1),
            'region_fraction':round(float(d['marked'].sum()/(~cut).sum()),3),
            'roles':(d['frame'],bg,d['control'],tuple(sorted(d['colors_a'])),tuple(sorted(d['colors_b']))),
            'palette':tuple(map(int,np.unique(pair['input']))),
            'lr_symmetric':int(np.array_equal(am,am[:,::-1]))+int(np.array_equal(bm,bm[:,::-1])),
            'tb_symmetric':int(np.array_equal(am,am[::-1,:]))+int(np.array_equal(bm,bm[::-1,:])),
            'empty_seam':not np.any(cut & (am | bm)),
            'no_collision':not np.any(cut & am & bm),
            'full_cut_truth_table':all(_evidence_flags(d)[k] for k in _EVIDENCE[4:]),
            'single_contributor':bool(visible_a) != bool(visible_b),
            'multicolor':len(d['colors_a']) > 1 or len(d['colors_b']) > 1,
            'density_a':round(float(am.mean()),3), 'density_b':round(float(bm.mean()),3),
            'area_ratio':round(float(max(am.sum(),bm.sum())/min(am.sum(),bm.sum())),3),
            'components_a':len(_components(am,False)), 'components_b':len(_components(bm,False)),
            'output_colors':len(set(map(int,result.flat))-{bg})}


def _reflect_pair(pair, axis):
    """Reflect each aligned interior, keeping the ordered panel positions."""
    d = _decode(pair['input'])
    out_panels = [pair['output'][1:-1,1+i*(d['cut'].shape[1]+1):1+i*(d['cut'].shape[1]+1)+d['cut'].shape[1]]
                  for i in range(4)]
    return {'input':_pack_panels([np.flip(p,axis) for p in d['panels']],d['frame']),
            'output':_pack_panels([np.flip(p,axis) for p in out_panels],d['frame'])}


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) >= 40, 'insufficient puzzle diversity')
    _require(len({len(p['train']) for p in puzzles}) >= 4, 'automatic train counts collapsed')
    _require(len({len(p['test']) for p in puzzles}) >= 3, 'automatic test counts collapsed')
    _require(_puzzles_equal(puzzles[0],generate_puzzle(0)), 'automatic determinism failed')
    _require(_puzzles_equal(generate_puzzle(12345,7,3),generate_puzzle(12345,7,3)), 'explicit determinism failed')
    for i,(nt,nq) in enumerate(((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None))):
        p = generate_puzzle(7000+i,nt,nq)
        _require(nt is None or len(p['train']) == nt, 'explicit train count ignored')
        _require(nq is None or len(p['test']) == nq, 'explicit test count ignored')
        validate_generated_puzzle(p)
    for nt,nq in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(4,4),(True,1),(3,1.5)):
        try:
            generate_puzzle(9000,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')

    # All five original outputs, including the bent test, use the rule-level path.
    for pair in _source_pairs():
        _inspect_pair(pair)
        _require(np.array_equal(_predict_from_input(pair['input']),pair['output']), 'source regression')

    features,layouts,motifs,side_families = [],set(),set(),set()
    split_families = {'train':set(),'test':set()}
    varied_roles = varied_sides = twins = source_reuses = same_family = mixed_splits = straight_shift = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local_roles,local_sides,local_motifs,controls = set(),set(),[],[]
        kinds = {'train':set(),'test':set()}
        for split in ('train','test'):
            for pair in puzzle[split]:
                _require(np.array_equal(_predict_from_input(pair['input']),pair['output']), 'cellwise check failed')
                d = _decode(pair['input'])
                f = _features(pair)
                features.append(f)
                family = f['family']
                split_families[split].add(family)
                kinds[split].add(family)
                side_families.add((family,f['side']))
                local_roles.add(f['roles'])
                local_sides.add(f['side'])
                layouts.add((f['panel_shape'],d['cut'].tobytes()))
                mk = (_mask_key(d['panel_a'],d['background']),_mask_key(d['panel_b'],d['background']))
                motifs.add(mk)
                local_motifs.append(mk)
                controls.append((mk,d['cut'].tobytes(),d['side']))
        varied_roles += len(local_roles) > 1
        varied_sides += len(local_sides) > 1
        source_reuses += len(set(local_motifs)) < len(local_motifs)
        twins += any((mk,cut,1-side) in controls for mk,cut,side in controls)
        same_family += len(kinds['train'] | kinds['test']) == 1
        mixed_splits += all(bool(kinds[s] & {'elbow','stepped'}) and bool(kinds[s] & set(_FAMILIES[:3])) for s in kinds)
        straight_shift += kinds['train'] <= set(_FAMILIES[:3]) and kinds['test'] == {'elbow'}

        pair = puzzle['train'][0]
        info = _inspect_pair(pair)
        flipped = _opposite(pair,np.random.default_rng(71))
        other = _inspect_pair(flipped)
        _require(np.array_equal(info['result'][info['cut']],other['result'][info['cut']]), 'marker changed seam priority')
        _require(np.array_equal(other['result'][info['marked']],info['panel_b'][info['marked']]), 'marker did not swap A region')
        _require(np.array_equal(other['result'][info['other']],info['panel_a'][info['other']]), 'marker did not swap B region')
        for axis in (0,1):
            _inspect_pair(_reflect_pair(pair,axis))

    _require(all(v == set(_FAMILIES) for v in split_families.values()), 'cut family split collapsed')
    _require(len(side_families) == 10, 'a cut family has a fixed marker side')
    _require(varied_sides == 42 and varied_roles >= 24, 'within-puzzle side/palette diversity collapsed')
    _require(twins >= 5 and source_reuses >= 8, 'controlled motif reuse is missing')
    _require(same_family >= 2 and mixed_splits >= 2 and straight_shift >= 2, 'train/test regimes collapsed')
    _require(len(layouts) >= 100 and len(motifs) >= 120, 'structural diversity collapsed')
    _require(len({f['panel_shape'] for f in features}) >= 35, 'too few panel sizes')
    _require(any(f['panel_shape'] == (5,5) for f in features), 'source-sized panels absent')
    _require(min(f['panel_shape'][0] for f in features) <= 4 and max(f['panel_shape'][0] for f in features) == 28,
             'small/tall panels absent')
    _require(sum(f['lr_symmetric'] for f in features) >= 50 and sum(f['tb_symmetric'] for f in features) >= 40,
             'clean symmetric motifs too rare')
    _require(sum(f['empty_seam'] for f in features) >= 20 and sum(f['single_contributor'] for f in features) >= 10,
             'simple overlap cases too rare')
    _require(sum(f['no_collision'] for f in features) >= 60, 'collision-free cases absent')
    _require(sum(f['full_cut_truth_table'] for f in features) < len(features)*.6, 'seam truth table is overforced')
    _require(sum(f['multicolor'] for f in features) >= 10, 'multicolor copying absent')
    _require(any(f['area_ratio'] >= 4 for f in features), 'unequal source areas absent')
    _require(any(f['components_a'] >= 4 for f in features), 'disconnected sources absent')
    _require(min(f['density_a'] for f in features) < .1 and max(f['density_a'] for f in features) > .65,
             'source densities collapsed')
    _require(len({f['endpoint_sides'] for f in features if f['family'] == 'elbow'}) == 4,
             'elbow endpoint orientations collapsed')
    _require(any(f['bends'] >= 4 for f in features if f['family'] == 'stepped'), 'multiple-bend cuts absent')
    _require({f['endpoint_sides'] for f in features if f['family'] == 'stepped'} == {'B_T','L_R'},
             'stepped cuts do not vary their endpoint sides')
    _require(sum(f['marker_edge'] for f in features) >= 30, 'edge markers excluded')
    _require(min(f['region_fraction'] for f in features) < .2 and max(f['region_fraction'] for f in features) > .8,
             'region area diversity collapsed')
    # Corruption, branches, loops and contacting markers are actively rejected.
    bad = {s:[{k:v.copy() for k,v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['train'][0]['output'][0,0] = (int(bad['train'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a changed frame')
    bad_pair = {k:v.copy() for k,v in puzzles[0]['train'][0].items()}
    d = _decode(bad_pair['input'])
    w = d['cut'].shape[1]
    bad_pair['output'][1,3*(w+1)+1] = (int(bad_pair['output'][1,3*(w+1)+1])+1)%10
    try:
        _inspect_pair(bad_pair)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted an incorrect destination cell')
    control = d['panels'][0].copy()
    control[d['marker']] = d['background']
    touching = next((r,c) for r,c in np.argwhere(~d['cut'])
                    if np.any(d['cut'][max(0,r-1):r+2,max(0,c-1):c+2]))
    control[touching] = d['control']
    malformed = _pack_panels([control,*d['panels'][1:]],d['frame'])
    try:
        _decode(malformed)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a marker touching the cut')
    for style in ('branch','loop','extra_region'):
        cut = np.zeros((9,7),dtype=bool)
        if style == 'branch':
            cut[:,3] = True
            cut[4,3:6] = True
        elif style == 'loop':
            cut[2:7,2] = cut[2:7,4] = True
            cut[2,2:5] = cut[6,2:5] = True
        else:
            cut[:,2] = cut[:,4] = True
        try:
            _cut_geometry(cut)
        except AssertionError:
            pass
        else:
            raise AssertionError('invalid cut accepted')
