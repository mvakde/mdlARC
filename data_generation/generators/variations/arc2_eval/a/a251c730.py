"""Anchor-keyed transparent copying between two framed panels."""

import numpy as np


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    def checked(value, name, low, high):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer or None")
        value = int(value)
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
        return value

    num_train = checked(num_train, "num_train", 2, 10)
    num_test = checked(num_test, "num_test", 1, 4)
    token = abs(int(seed))

    if num_train is None and num_test is None:
        num_test = 1 + token % 4
        num_train = num_test + 1 + (token // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + token % (10 - num_test)
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = 1 + token % maximum_test

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return int(num_train), int(num_test)



# Exact source content: regression fixtures and content-based exclusion only.
_SOURCE_ROWS = ((('135791357913579135791357913579', '135791357913579135791357913579', '135791357913579135791357913579', '135791357913579135791357913579', '135791357913579135791357913579', '136666666666666666791357913579', '136111111111111116791357913579', '136111111111111116791357913579', '136112121111111116791357913579', '136112221111111116791357913579', '136111811111111116791357913579', '136111111111111116791357913579', '136111111111111116791357913579', '136111111111111116791357913579', '136111111112121116791357913579', '136111111112221116791357913579', '136111111111811116791357913579', '136111111111111116791357913579', '136666666666666666791357913579', '135791357913579135791357913579', '135793333333333333333333313579', '135793111111111111111111313579', '135793111111111111111111313579', '135793111111111111111111313579', '135793111111111181118111313579', '135793111811111111111111313579', '135793111111111111111111313579', '135793111111111111111111313579', '135793333333333333333333313579', '135791357913579135791357913579'), ('33333333333333333333', '31111111111111111113', '31111111112121212113', '31121211112221222113', '31122211111811181113', '31118111111111111113', '31111111111111111113', '31111111111111111113', '33333333333333333333')), (('000000000000000000000000000000', '222222222222222222222222222222', '444444444444444444444444444444', '666111111111111163333333366666', '888122222222222183444444388888', '000122222222222103444444300000', '222122282222222123444444322222', '444122818222222143444444344444', '666122282222222163444444366666', '888122222222222183444444388888', '000122222222222103444414300000', '222122222282222123444444322222', '444122222818222143444444344444', '666122222282222163444444366666', '888122222222222183444444388888', '000122222222222103444444300000', '222122222222222123444444322222', '444122282222222143444444344444', '666122818222222163441444366666', '888122282222222183444444388888', '000122222222222103444444300000', '222111111111111123333333322222', '444444444444444444444444444444', '666666666666666666666666666666', '888888888888888888888888888888', '000000000000000000000000000000', '222222222222222222222222222222', '444444444444444444444444444444', '666666666666666666666666666666', '888888888888888888888888888888'), ('33333333', '34444443', '34444443', '34444443', '34444443', '34444443', '34444843', '34448183', '34444843', '34444443', '34444443', '34444443', '34444443', '34444443', '34484443', '34818443', '34484443', '34444443', '33333333')), (('012301230123012301230123012301', '014444444444444444440123012301', '014333333333333333340123012301', '014333333333333333340123012301', '014333331333355533340123012301', '014333316133357533340123012301', '014333311133333333340123012301', '014333333333333333340123012301', '014333333333331333340123012301', '014333555333316133340123012301', '014333575333311133340123012301', '014333333333333333340123012301', '014444444444444444440123012301', '012301230123012301230123012301', '012301230123012301230123012301', '022222222222222301230123012301', '028888888888882301230123012301', '028888888888882301230123012301', '028888888888882301230123012301', '028888868888882301230123012301', '028888888887882301230123012301', '028688888888882301230123012301', '028888888888882301230123012301', '022222222222222301230123012301', '012301230123012301230123012301', '012301230123012301230123012301', '012301230123012301230123012301', '012301230123012301230123012301', '012301230123012301230123012301', '012301230123012301230123012301'), ('22222222222222', '28888888888882', '28888888888882', '28888818888882', '28888161855582', '28188111857582', '21618888888882', '21118888888882', '22222222222222')))

_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _D4:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _crop(grid, box):
    y, x, h, w = box
    return grid[y:y+h, x:x+w]


def _find_rectangular_frames(grid):
    """Find borders by their four straight sides, not color connectivity.

    A uniform inner rim makes a panel recognizable even if its frame is
    connected to stripes outside, or shares colors with interior objects.
    """
    h, w = grid.shape
    down = np.ones((h, w), dtype=int)
    right = np.ones((h, w), dtype=int)
    for y in range(h-2, -1, -1):
        down[y] += (grid[y] == grid[y+1]) * down[y+1]
    for x in range(w-2, -1, -1):
        right[:, x] += (grid[:, x] == grid[:, x+1]) * right[:, x+1]
    frames = []
    for y in range(h-4):
        for bottom in range(y+4, h):
            cols = np.flatnonzero(down[y] >= bottom-y+1)
            for i, x in enumerate(cols):
                for end in cols[i+1:]:
                    width = int(end-x+1)
                    if width < 5 or right[y, x] < width or right[bottom, x] < width:
                        continue
                    box = (y, int(x), bottom-y+1, width)
                    inner = _crop(grid, box)[1:-1, 1:-1]
                    fill = int(inner[0, 0])
                    if fill == int(grid[y, x]):
                        continue
                    if not (np.all(inner[0] == fill) and np.all(inner[-1] == fill)
                            and np.all(inner[:, 0] == fill) and np.all(inner[:, -1] == fill)):
                        continue
                    if np.count_nonzero(inner == fill) <= inner.size/2:
                        continue
                    components = _components(inner != fill)
                    if components:
                        frames.append({'box': box, 'fill': fill,
                                       'frame': int(grid[y, x]), 'components': components})
    return frames


def _disjoint(a, b):
    y, x, h, w = a
    v, u, k, l = b
    return y+h <= v or v+k <= y or x+w <= u or u+l <= x


def _dictionary(grid, panel):
    inner = _crop(grid, panel['box'])[1:-1, 1:-1]
    templates, repetitions = {}, {}
    for component in panel['components']:
        if len(component) < 3:
            raise AssertionError('reference occurrences must be connected complete motifs')
        colors, counts = np.unique([inner[r, c] for r, c in component], return_counts=True)
        singletons = colors[counts == 1]
        assert len(singletons) == 1, 'one distinctive key per complete motif'
        key = int(singletons[0])
        ar, ac = next((r, c) for r, c in component if inner[r, c] == key)
        signature = tuple(sorted((r-ar, c-ac, int(inner[r, c])) for r, c in component))
        assert key not in templates or templates[key] == signature, 'conflicting template definitions'
        templates[key] = signature
        repetitions[key] = repetitions.get(key, 0) + 1
    keys = set(templates)
    assert all(color not in keys for key, sig in templates.items()
               for r, c, color in sig if (r, c) != (0, 0)), 'decoration cannot masquerade as a key'
    return templates, repetitions


def _reconstruct(grid):
    """Interpret only the input, then translate the dictionary into the crop."""
    panels = _find_rectangular_frames(grid)
    interpretations = []
    for target in panels:
        if not all(len(c) == 1 for c in target['components']):
            continue
        for reference in panels:
            if not _disjoint(target['box'], reference['box']):
                continue
            try:
                templates, repetitions = _dictionary(grid, reference)
                ti = _crop(grid, target['box'])[1:-1, 1:-1]
                anchors = [(c[0][0], c[0][1], int(ti[c[0]])) for c in target['components']]
                assert all(k in templates for _, _, k in anchors)
                expected = _crop(grid, target['box']).copy()
                occupied = set()
                for ar, ac, key in anchors:
                    for dr, dc, color in templates[key]:
                        r, c = ar+dr, ac+dc
                        assert 0 <= r < ti.shape[0] and 0 <= c < ti.shape[1], 'clipped stamp'
                        assert (r, c) not in occupied, 'overlapping stamps'
                        assert int(ti[r, c]) == (key if (dr, dc) == (0, 0) else target['fill'])
                        assert color != target['fill'], 'required cells must remain visible'
                        occupied.add((r, c))
                        expected[r+1, c+1] = color
                assert len(_components(expected[1:-1, 1:-1] != target['fill'])) == len(anchors)
                interpretations.append((expected, {'reference': reference, 'target': target,
                    'templates': templates, 'repetitions': repetitions, 'anchors': anchors}))
            except AssertionError:
                continue
    assert len(interpretations) == 1, 'input must have one reference/target interpretation'
    return interpretations[0]


def solve_from_training(train_pairs, test_input):
    """Prediction interface never receives a test label or generation state."""
    assert train_pairs
    for pair in train_pairs:
        predicted, _ = _reconstruct(pair['input'])
        assert np.array_equal(predicted, pair['output']), 'training contradicts transparent copying'
    return _reconstruct(test_input)[0]


def _shape(rng, small=False):
    n = int(rng.integers(3, 8 if small else 16))
    height = int(rng.integers(2, 5 if small else 7))
    width = int(rng.integers(2, 5 if small else 7))
    if rng.random() < .22:
        height, width = (2, int(rng.integers(3, 7))) if rng.random() < .5 else (int(rng.integers(3, 7)), 2)
    n = min(n, height*width)
    occupied = {(int(rng.integers(height)), int(rng.integers(width)))}
    style = int(rng.integers(3))
    while len(occupied) < n:
        frontier = sorted({(r+dr, c+dc) for r, c in occupied for dr, dc in _D4
                           if 0 <= r+dr < height and 0 <= c+dc < width} - occupied)
        neighbors = np.array([sum((r+dr, c+dc) in occupied for dr, dc in _D4)
                              for r, c in frontier])
        weights = np.ones(len(frontier)) if style == 0 else (
            1.0/(neighbors**2) if style == 1 else neighbors.astype(float)**3)
        chosen = int(rng.choice(len(frontier), p=weights/weights.sum()))
        occupied.add(frontier[chosen])
    cells = sorted(occupied)
    if rng.random() < .55:
        boundary = [p for p in cells if any((p[0]+dr, p[1]+dc) not in occupied for dr, dc in _D4)]
        ar, ac = _pick(rng, boundary)
    else:
        ar, ac = _pick(rng, cells)
    return tuple(sorted((r-ar, c-ac) for r, c in cells))


def _asymmetric(shape):
    s = set(shape)
    return s != {(-c, r) for r, c in s} and s != {(r, -c) for r, c in s}


def _templates(rng, count, palette, small, require_asym):
    available = list(palette)
    keys = available[:count]
    decorations = available[count:]
    result = []
    for i, key in enumerate(keys):
        shape = _shape(rng, small)
        if i and rng.random() < .18:
            shape = tuple((r, c) for r, c, _ in result[0])
        if require_asym and i == 0 and not _asymmetric(shape):
            # Re-anchoring at an extremal occupied cell makes direction visible.
            ar, ac = min(shape)
            shape = tuple((r-ar, c-ac) for r, c in shape)
            if not _asymmetric(shape):
                shape = ((0, 0), (1, 0), (1, 1), (2, 0))
        if require_asym and i == 0:
            rows, cols = zip(*shape)
            if (max(rows)-min(rows)+1)*(max(cols)-min(cols)+1) == len(shape):
                # A visible hole distinguishes transparency from copying a
                # rectangular patch filled with the reference background.
                corners = [(r, c) for r in (min(rows), max(rows))
                           for c in (min(cols), max(cols)) if (r, c) != (0, 0)]
                shape = tuple(p for p in shape if p != corners[-1])
                if len(shape) < 3 or not _asymmetric(shape):
                    shape = ((0, 0), (1, 0), (1, 1), (2, 0))
        offsets = [p for p in shape if p != (0, 0)]
        num_colors = 1
        if len(offsets) >= 4 and rng.random() < .5:
            num_colors = int(rng.integers(2, min(3, len(decorations), len(offsets)//2)+1)) if len(decorations) >= 2 else 1
        colors = list(rng.choice(decorations, size=num_colors, replace=False))
        # At least two copies of each decoration color: only the key is a singleton.
        assignment = colors*2 + [_pick(rng, colors) for _ in range(len(offsets)-2*num_colors)]
        if num_colors == 1:
            assignment = colors*len(offsets)
        rng.shuffle(assignment)
        result.append(tuple(sorted([(0, 0, int(key))] +
                                  [(r, c, int(color)) for (r, c), color in zip(offsets, assignment)])))
    return result


def _bounds(template):
    rows = [r for r, _, _ in template]
    cols = [c for _, c, _ in template]
    return min(rows), min(cols), max(rows)-min(rows)+1, max(cols)-min(cols)+1


def _shelf(templates, ids, width):
    """Guaranteed separated bounding boxes with a one-cell interior margin."""
    y, x, shelf_h = 2, 2, 0
    places = []
    for key in ids:
        low_r, low_c, h, w = _bounds(templates[key])
        if w+4 > width:
            return None
        if x+w > width-2:
            y += shelf_h+1
            x, shelf_h = 2, 0
        places.append((key, y-low_r, x-low_c))
        x += w+1
        shelf_h = max(shelf_h, h)
    return y+shelf_h+2, places


def _panel_options(templates, ids):
    options = []
    for w in range(5, 29):
        packed = _shelf(templates, ids, w)
        if packed is not None:
            h, places = packed
            if h <= 28 and (h-2)*(w-2) > 2*sum(len(templates[k]) for k in ids):
                options.append((h, w, places))
    return options


def _place_in_panel(rng, templates, ids, shape, fallback):
    h, w = shape
    order = sorted(ids, key=lambda k: -np.prod(_bounds(templates[k])[2:]))
    occupied = np.zeros(shape, dtype=bool)
    places = []
    for key in order:
        lr, lc, bh, bw = _bounds(templates[key])
        candidates = [(r, c) for r in range(2, h-bh-1) for c in range(2, w-bw-1)]
        rng.shuffle(candidates)
        for r, c in candidates:
            if not np.any(occupied[r-1:r+bh+1, c-1:c+bw+1]):
                occupied[r:r+bh, c:c+bw] = True
                places.append((key, r-lr, c-lc))
                break
        else:
            # The precomputed shelf remains a guaranteed packing at this size.
            maxr = max(ar+max(r for r, _, _ in templates[k]) for k, ar, _ in fallback)
            maxc = max(ac+max(c for _, c, _ in templates[k]) for k, _, ac in fallback)
            dy = int(rng.integers(0, h-maxr-2))
            dx = int(rng.integers(0, w-maxc-2))
            return [(k, ar+dy, ac+dx) for k, ar, ac in fallback]
    return places


def _geometry(rng, templates, ref_ids, target_ids, layout):
    refs = _panel_options(templates, ref_ids)
    targets = _panel_options(templates, target_ids)
    axis = 0 if layout == 'vertical' else 1
    feasible = [(r, t) for r in refs for t in targets if r[axis]+t[axis]+3 <= 30]
    if not feasible:
        return None
    size_mode = _pick(rng, ('compact', 'balanced', 'balanced', 'spacious'))
    if size_mode == 'compact':
        feasible.sort(key=lambda rt: (rt[0][axis]+rt[1][axis]+3)
                      * (max(rt[0][1-axis], rt[1][1-axis])+2))
        feasible = feasible[:max(1, len(feasible)//6)]
    ref, target = _pick(rng, feasible)
    shapes = [list(ref[:2]), list(target[:2])]
    if rng.random() < .18:
        equal = [max(shapes[0][i], shapes[1][i]) for i in (0, 1)]
        if 2*equal[axis]+3 <= 30:
            shapes = [equal.copy(), equal.copy()]
    else:
        for a in (0, 1):
            for dim in (0, 1):
                slack = (27-shapes[0][dim]-shapes[1][dim] if dim == axis else 28-shapes[a][dim])
                shapes[a][dim] += int(rng.integers(0, min(2 if size_mode == 'compact' else 5, slack)+1))
    gap = int(rng.integers(1, min(4, 28-shapes[0][axis]-shapes[1][axis])+1))
    required = [max(s[0] for s in shapes)+2, max(s[1] for s in shapes)+2]
    required[axis] = shapes[0][axis]+shapes[1][axis]+gap+2
    canvas = [int(rng.integers(n, 31 if size_mode == 'spacious' else min(31, n+4))) for n in required]
    if size_mode != 'compact' and rng.random() < .15:
        canvas = [30, 30]
    order = [0, 1] if rng.random() < .5 else [1, 0]
    positions = [[0, 0], [0, 0]]
    start = int(rng.integers(1, canvas[axis]-shapes[0][axis]-shapes[1][axis]-gap))
    positions[order[0]][axis] = start
    positions[order[1]][axis] = start+shapes[order[0]][axis]+gap
    other = 1-axis
    for i in (0, 1):
        positions[i][other] = int(rng.integers(1, canvas[other]-shapes[i][other]))
    # Ensure the intended stacked/side-by-side relation has overlapping projections.
    a, b = positions[0][other], positions[1][other]
    if a+shapes[0][other] <= b or b+shapes[1][other] <= a:
        low = max(1, a-shapes[1][other]+1)
        high = min(canvas[other]-shapes[1][other]-1, a+shapes[0][other]-1)
        positions[1][other] = int(rng.integers(low, high+1))
    boxes = [tuple(positions[i]+shapes[i]) for i in (0, 1)]
    rp = _place_in_panel(rng, templates, ref_ids, shapes[0], ref[2])
    tp = _place_in_panel(rng, templates, target_ids, shapes[1], target[2])
    return tuple(canvas), boxes, rp, tp


def _make_canvas(rng, shape, palette):
    dr, dc = _pick(rng, ((1, 0), (0, 1), (1, 1), (1, -1)))
    extent = shape[0] if dc == 0 else shape[1] if dr == 0 else sum(shape)-1
    max_period = min(18, extent//2)
    count = int(rng.integers(2, min(6, len(palette), max_period)+1))
    band = int(rng.integers(1, min(3, max_period//count)+1))
    length = int(rng.integers(count, min(count+3, max_period//band)+1))
    colors = list(rng.choice(palette, size=count, replace=False))
    word = colors + [_pick(rng, colors) for _ in range(length-count)]
    rng.shuffle(word)
    phase = int(rng.integers(length*band))
    rows, cols = np.indices(shape)
    return np.asarray(word, dtype=np.int8)[((dr*rows+dc*cols+phase)//band) % length]


def _palette(rng, config, ntypes, distinct):
    base = config['palette'].copy()
    mode = config['palette_mode']
    if mode in ('independent', 'simple'):
        rng.shuffle(base)
    elif mode == 'partial':
        tail = base[2:].copy()
        rng.shuffle(tail)
        base[2:] = tail
    budget = int(rng.integers(max(4, ntypes+3), 11)) if mode != 'simple' else max(4, ntypes+3)
    active = [int(c) for c in base[:budget]]
    rf = active[0]
    tf = active[1] if distinct or rng.random() < .65 else rf
    motif = [c for c in active if c not in (rf, tf)]
    rframe = _pick(rng, [c for c in active if c != rf])
    tframe = rframe if rframe != tf and rng.random() < .32 else _pick(rng, [c for c in active if c != tf])
    return rf, tf, rframe, tframe, motif, active


def _draw_panel(grid, box, frame, fill):
    panel = _crop(grid, box)
    panel[:] = frame
    panel[1:-1, 1:-1] = fill


def _stamp(grid, templates, places):
    for key, ar, ac in places:
        for dr, dc, color in templates[key]:
            grid[ar+dr, ac+dc] = color


def _generate_pair(rng, config, regime, layout, demonstrate=False):
    mode = config['type_mode']
    if mode == 'single':
        ntypes = 1
    elif mode == 'source_shift':
        ntypes = 1 if regime == 'source_train' else int(rng.integers(2, 5))
    elif mode == 'combine':
        ntypes = int(rng.integers(1, 3)) if regime == 'source_train' else int(rng.integers(3, 5))
    else:
        ntypes = int(rng.integers(2, 5))
    for attempt in range(40):
        small = rng.random() < .4 or attempt >= 20
        rf, tf, rframe, tframe, motif, active = _palette(rng, config, ntypes, demonstrate)
        templates = _templates(rng, ntypes, motif, small, demonstrate)
        ref_ids = [i for i in range(ntypes) for _ in range(int(rng.choice([1, 2, 3, 4], p=[.13, .48, .28, .11])))]
        if demonstrate:
            ref_ids = [i for i in range(ntypes) for _ in range(max(2, ref_ids.count(i)))]
        target_keys = list(range(ntypes))
        if not demonstrate and ntypes > 1 and mode not in ('source_shift', 'combine') and rng.random() < .28:
            target_keys = list(rng.choice(ntypes, size=int(rng.integers(1, ntypes)), replace=False))
        total = int(rng.integers(len(target_keys), max(len(target_keys)+1, 11 if small else 8)))
        target_ids = target_keys + [_pick(rng, target_keys) for _ in range(total-len(target_keys))]
        rng.shuffle(ref_ids)
        rng.shuffle(target_ids)
        geometry = _geometry(rng, templates, ref_ids, target_ids, layout)
        if geometry is None:
            continue
        shape, boxes, rp, tp = geometry
        grid = _make_canvas(rng, shape, active)
        _draw_panel(grid, boxes[0], rframe, rf)
        _draw_panel(grid, boxes[1], tframe, tf)
        _stamp(_crop(grid, boxes[0]), templates, rp)
        target = _crop(grid, boxes[1])
        for key, ar, ac in tp:
            target[ar, ac] = next(color for r, c, color in templates[key] if (r, c) == (0, 0))
        out = target.copy()
        _stamp(out, templates, tp)
        try:
            predicted, _ = _reconstruct(grid)
            assert np.array_equal(predicted, out)
            return {'input': grid, 'output': out}
        except AssertionError:
            continue
    # A bounded constructive fallback retains all requested types. Three-cell
    # L motifs and two examples per key fit either layout even with four keys.
    rf, tf, rframe, tframe, motif, active = _palette(rng, config, ntypes, True)
    templates = [((0, 0, motif[k]), (1, 0, motif[ntypes]), (1, 1, motif[ntypes])) for k in range(ntypes)]
    ids = list(range(ntypes))*2
    geometry = _geometry(rng, templates, ids, list(range(ntypes)), layout)
    assert geometry is not None
    shape, boxes, rp, tp = geometry
    grid = _make_canvas(rng, shape, active)
    _draw_panel(grid, boxes[0], rframe, rf)
    _draw_panel(grid, boxes[1], tframe, tf)
    _stamp(_crop(grid, boxes[0]), templates, rp)
    target = _crop(grid, boxes[1])
    for k, ar, ac in tp:
        target[ar, ac] = motif[k]
    out = target.copy()
    _stamp(out, templates, tp)
    assert np.array_equal(_reconstruct(grid)[0], out)
    return {'input': grid, 'output': out}


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p['input']), _grid_key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def _source_pairs():
    return [{'input': np.array([[int(c) for c in row] for row in inp], dtype=np.int8),
             'output': np.array([[int(c) for c in row] for row in out], dtype=np.int8)}
            for inp, out in _SOURCE_ROWS]


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # These are distribution choices, not hidden codes. Every pair visibly
    # supplies its own dictionary; only copying, transparency and crop persist.
    config = {'palette': rng.permutation(10),
              'palette_mode': _pick(rng, ('independent', 'partial', 'stable', 'simple')),
              'type_mode': _pick(rng, ('single', 'source_shift', 'mixed', 'combine')),
              'layout_mode': _pick(rng, ('vertical', 'horizontal', 'mixed', 'v_to_h', 'h_to_v'))}
    result = {'train': [], 'test': []}
    seen = set()
    for split, count, regime in (('train', train_count, 'source_train'), ('test', test_count, 'target_test')):
        for index in range(count):
            layout = config['layout_mode']
            if layout == 'mixed':
                layout = _pick(rng, ('vertical', 'horizontal'))
            elif layout == 'v_to_h':
                layout = 'vertical' if regime == 'source_train' else 'horizontal'
            elif layout == 'h_to_v':
                layout = 'horizontal' if regime == 'source_train' else 'vertical'
            pair_rng = np.random.default_rng(int(rng.integers(2**63)))
            for _ in range(8):
                pair = _generate_pair(pair_rng, config, regime, layout, split == 'train' and index == 0)
                key = _grid_key(pair['input'])
                if key not in seen:
                    break
            assert key not in seen
            seen.add(key)
            result[split].append(pair)
    validate_generated_puzzle(result)
    return result


def _canvas_features(grid, info):
    mask = np.ones(grid.shape, dtype=bool)
    for role in ('reference', 'target'):
        y, x, h, w = info[role]['box']
        mask[y:y+h, x:x+w] = False
    rows, cols = np.nonzero(mask)
    values = grid[mask]
    observed = tuple(int(c) for c in np.unique(values))
    assert len(observed) >= 2
    candidates = []
    for name, dr, dc in (('rows', 1, 0), ('columns', 0, 1), ('diagonal', 1, 1), ('antidiagonal', 1, -1)):
        coordinate = dr*rows+dc*cols
        for period in range(2, 19):
            counts = np.zeros((period, 10), dtype=int)
            np.add.at(counts, (coordinate % period, values), 1)
            if np.all(np.count_nonzero(counts, axis=1) == 1):
                word = tuple(int(c) for c in np.argmax(counts, axis=1))
                transitions = sum(word[i] != word[(i+1) % period] for i in range(period))
                candidates.append((period, name, period/transitions, word))
                break
    assert candidates, 'exterior must be a regular repeating color word'
    period, orientation, mean_band, word = min(candidates)
    return {'canvas_colors': observed, 'canvas_period': period,
            'canvas_orientation': orientation, 'canvas_mean_band': mean_band}


def pair_features(pair):
    """Measurements reconstructed from visible cells, never sampled metadata."""
    predicted, info = _reconstruct(pair['input'])
    assert np.array_equal(predicted, pair['output'])
    ref, target = info['reference'], info['target']
    ry, rx, rh, rw = ref['box']
    ty, tx, th, tw = target['box']
    templates = info['templates']
    shapes = [tuple((r, c) for r, c, _ in sig) for sig in templates.values()]
    bounds = [_bounds(sig) for sig in templates.values()]
    decorations = [set(color for r, c, color in sig if (r, c) != (0, 0)) for sig in templates.values()]
    counts = {k: sum(key == k for _, _, key in info['anchors']) for k in templates}
    canvas = _canvas_features(pair['input'], info)
    exterior = set(canvas['canvas_colors'])
    layout = 'vertical' if ry+rh <= ty or ty+th <= ry else 'horizontal'
    shared_decor = any(a & b for i, a in enumerate(decorations) for b in decorations[i+1:])
    nonfill = set(templates).union(*decorations)
    touching = False
    grid = pair['input']
    for panel in (ref, target):
        y, x, h, w = panel['box']
        frame = panel['frame']
        strips = (grid[y-1, x:x+w] if y else [], grid[y+h, x:x+w] if y+h < grid.shape[0] else [],
                  grid[y:y+h, x-1] if x else [], grid[y:y+h, x+w] if x+w < grid.shape[1] else [])
        touching |= any(np.any(np.asarray(strip) == frame) for strip in strips)
    geometries = tuple(sorted(shapes))
    result = {'layout': layout, 'reference_shape': (rh, rw), 'target_shape': (th, tw),
              'input_shape': grid.shape, 'input_colors': len(np.unique(grid)),
              'reference_types': len(templates), 'target_types': sum(n > 0 for n in counts.values()),
              'reference_repetitions': tuple(sorted(info['repetitions'].values())),
              'target_multiplicities': tuple(sorted(counts.values())), 'target_total': sum(counts.values()),
              'unused_type': any(n == 0 for n in counts.values()),
              'equal_fills': ref['fill'] == target['fill'], 'equal_frames': ref['frame'] == target['frame'],
              'motif_frame_reuse': bool(nonfill & {ref['frame'], target['frame']}),
              'motif_exterior_reuse': bool(nonfill & exterior),
              'fill_exterior_reuse': bool({ref['fill'], target['fill']} & exterior),
              'frame_exterior_reuse': bool({ref['frame'], target['frame']} & exterior),
              'frame_touches_exterior': bool(touching), 'shared_decoration': shared_decor,
              'multicolor_motif': any(len(d) > 1 for d in decorations),
              'motif_sizes': tuple(sorted(len(s) for s in shapes)),
              'motif_bounds': tuple(sorted((h, w) for _, _, h, w in bounds)),
              'max_motif_extent': max(max(h, w) for _, _, h, w in bounds),
              'off_center_anchor': any(2*lr+h-1 != 0 or 2*lc+w-1 != 0 for lr, lc, h, w in bounds),
              'extremal_anchor': any(lr in (0, 1-h) or lc in (0, 1-w) for lr, lc, h, w in bounds),
              'asymmetric': any(_asymmetric(s) for s in shapes),
              'equal_panel_sizes': (rh, rw) == (th, tw),
              'horizontal_unequal_heights': layout == 'horizontal' and rh != th,
              'horizontal_offset': layout == 'horizontal' and ry != ty,
              'reference_first': (ry < ty if layout == 'vertical' else rx < tx),
              'gap': (max(ry, ty)-min(ry+rh, ty+th) if layout == 'vertical'
                      else max(rx, tx)-min(rx+rw, tx+tw)),
              'minimum_margin': min(ry, rx, ty, tx, grid.shape[0]-ry-rh, grid.shape[1]-rx-rw,
                                    grid.shape[0]-ty-th, grid.shape[1]-tx-tw),
              'same_silhouette': len(set(shapes)) < len(shapes), 'geometries': geometries,
              'role_profile': (ref['fill'], target['fill'], ref['frame'], target['frame'],
                               tuple(templates), tuple(tuple(sorted(d)) for d in decorations))}
    result.update(canvas)
    result['rich_combination'] = (result['reference_types'] >= 3 and shared_decor
        and result['max_motif_extent'] >= 4 and result['multicolor_motif']
        and len(exterior) >= 4 and result['off_center_anchor'])
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
        assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
        assert 2 <= len(puzzle['train']) <= 10
        assert 1 <= len(puzzle['test']) <= 4 and len(puzzle['train']) > len(puzzle['test'])
        seen_inputs, seen_pairs = set(), set()
        source_keys = {(_grid_key(p['input']), _grid_key(p['output'])) for p in _source_pairs()}
        evidence = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
                for grid in pair.values():
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert all(1 <= n <= 30 for n in grid.shape)
                    assert np.all((grid >= 0) & (grid <= 9))
                key = (_grid_key(pair['input']), _grid_key(pair['output']))
                assert key[0] not in seen_inputs and key not in seen_pairs and key not in source_keys
                assert not np.array_equal(pair['input'], pair['output'])
                seen_inputs.add(key[0])
                seen_pairs.add(key)
                predicted, info = _reconstruct(pair['input'])
                assert np.array_equal(predicted, pair['output']), 'incorrect transparent stamp or crop'
                _canvas_features(pair['input'], info)
                if split == 'train':
                    templates = list(info['templates'].values())
                    evidence.append((info['reference']['fill'] != info['target']['fill'],
                                     any(_asymmetric([(r, c) for r, c, _ in sig])
                                         and _bounds(sig)[2]*_bounds(sig)[3] > len(sig)
                                         for sig in templates)))
        assert any(a and b for a, b in evidence), 'training must show transparency and orientation'
        # Templates are visible per input. Counts, palettes and layouts impose no
        # extra hidden regime convention; all train/test pairs use this parser.
    except AssertionError:
        raise
    except Exception as error:
        raise AssertionError(f'invalid puzzle: {error}') from error


def self_test() -> None:
    source = _source_pairs()
    for pair in source:
        prediction, _ = _reconstruct(pair['input'].copy())
        assert np.array_equal(prediction, pair['output']), 'source regression'
    assert np.array_equal(solve_from_training(source[:2], source[2]['input'].copy()), source[2]['output'])
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    records = []
    layout_regimes, type_regimes = set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = [{k: v.copy() for k, v in pair.items()} for pair in puzzle['train']]
        features = {s: [pair_features(pair) for pair in puzzle[s]] for s in ('train', 'test')}
        for pair in puzzle['test']:
            prediction = solve_from_training(train, pair['input'].copy())
            assert np.array_equal(prediction, pair['output'])
        records.extend(features['train']+features['test'])
        layout_regimes.add(tuple(tuple(sorted({f['layout'] for f in features[s]})) for s in ('train', 'test')))
        type_regimes.add(tuple(tuple(sorted({f['target_types'] for f in features[s]})) for s in ('train', 'test')))
        assert len({f['input_shape'] for fs in features.values() for f in fs}) >= 2
        assert len({f['geometries'] for fs in features.values() for f in fs}) >= 2
    assert len(layout_regimes) >= 5 and len(type_regimes) >= 5
    assert {r['reference_types'] for r in records} == {1, 2, 3, 4}
    assert {r['target_types'] for r in records} == {1, 2, 3, 4}
    sizes = {n for r in records for n in r['motif_sizes']}
    assert set(range(3, 12)) <= sizes
    assert len({shape for r in records for shape in r['geometries']}) >= 150
    assert len({r['input_shape'] for r in records}) >= 50
    assert len({r['target_shape'] for r in records}) >= 60
    assert any(r['input_shape'] == (30, 30) for r in records)
    assert {n for r in records for n in r['reference_repetitions']} >= {1, 2, 3, 4}
    assert {r['target_total'] for r in records} >= {1, 2, 3, 4, 5, 6, 7, 8}
    assert {len(r['canvas_colors']) for r in records} >= {2, 3, 4, 5, 6}
    assert len({r['canvas_period'] for r in records}) >= 10
    assert len({r['canvas_orientation'] for r in records}) == 4
    assert any(r['canvas_mean_band'] >= 2 for r in records)
    assert len({r['role_profile'] for r in records}) >= 100
    for name in ('equal_fills', 'equal_frames', 'motif_frame_reuse', 'motif_exterior_reuse',
                 'fill_exterior_reuse', 'frame_exterior_reuse', 'shared_decoration',
                 'unused_type', 'multicolor_motif', 'equal_panel_sizes', 'horizontal_unequal_heights',
                 'horizontal_offset', 'same_silhouette', 'rich_combination', 'frame_touches_exterior'):
        assert sum(bool(r[name]) for r in records) >= 3, name
    assert {r['reference_first'] for r in records} == {True, False}
    assert len({r['gap'] for r in records}) >= 3
    assert len({r['minimum_margin'] for r in records}) >= 2
    assert sum(r['max_motif_extent'] > 3 for r in records) > 100
    first = generate_puzzle(12345, 6, 3)
    assert _puzzle_key(first) == _puzzle_key(generate_puzzle(12345, 6, 3))
    assert _puzzle_key(first) != _puzzle_key(generate_puzzle(12346, 6, 3))
    for index, (a, b) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(201+index, a, b)
        validate_generated_puzzle(puzzle)
        assert a is None or len(puzzle['train']) == a
        assert b is None or len(puzzle['test']) == b
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(17, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal count request accepted')
    # Equal sizes and colors must not affect input-only target selection.
    demo = np.tile(np.array([0, 1], dtype=np.int8), (20, 12))
    for box in ((2, 2, 8, 8), (11, 12, 8, 8)):
        _draw_panel(demo, box, 2, 3)
    _stamp(_crop(demo, (2, 2, 8, 8)), [((0, 0, 4), (1, 0, 5), (1, 1, 5))], [(0, 3, 3)])
    demo[14, 15] = 4
    predicted, info = _reconstruct(demo)
    assert info['target']['box'] == (11, 12, 8, 8)
    assert predicted[4, 3] == predicted[4, 4] == 5
    # Corrupted labels and conflicting dictionaries must fail validation.
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('bad frame label accepted')
    conflicting = source[0]['input'].copy()
    conflicting[8, 5] = 1
    ambiguous = demo.copy()
    _draw_panel(ambiguous, (2, 12, 8, 8), 2, 3)
    _stamp(_crop(ambiguous, (2, 12, 8, 8)),
           [((0, 0, 4), (1, 0, 5), (1, 1, 5))], [(0, 3, 3)])
    for malformed in (conflicting, ambiguous):
        try:
            _reconstruct(malformed)
        except AssertionError:
            continue
        raise AssertionError('conflicting or ambiguous input accepted')
