import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Rejecting any exact source input is enough to reject an exact source pair.
_SOURCE_INPUTS = {
    "000000000300111100000300000100000300233320000300000130000300"
    "000131100300000030100300000030100300000023330200000000100000"
    "000000110110000000000011",
    "000000000070777700000070333330333470000700003070000000000070"
    "000777703000000000703070007777703070007000003070330333333070"
    "007777777770000000000000",
    "007030600200007000000100007030600100337320600100007000606166"
    "007000000100707000000000000000000100000021111200000010000000"
    "000000000000000010000000",
    "010000100060010000100060616666060660010000100040000000100000"
    "011101100000000000000000000000000000000000000000000000000000"
    "000000000000000000000000",
    "000070000070000070000070000070000070427720027720070000070040"
    "070000070000070000070000666666606666070000070000027777720000"
    "000000000000000000000000",
}


def _as_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, rng, num_train, num_test):
    if num_train is not None:
        num_train = _as_int(num_train, "num_train")
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")
    if num_test is not None:
        num_test = _as_int(num_test, "num_test")
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")

    if num_train is None and num_test is None:
        choices = (
            (2, 1),
            (3, 1),
            (4, 2),
            (5, 2),
            (5, 3),
            (6, 1),
            (6, 4),
            (7, 2),
            (8, 3),
            (9, 4),
            (10, 4),
        )
        return choices[int(seed) % len(choices)]

    if num_train is None:
        legal = np.arange(max(2, num_test + 1), 11)
        num_train = int(legal[int(rng.integers(len(legal)))])
    elif num_test is None:
        legal = np.arange(1, min(4, num_train - 1) + 1)
        num_test = int(legal[int(rng.integers(len(legal)))])

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


# A crossing order is a learned convention, never an invisible drawing order.
# The first demonstration witnesses every pair of payload colors. Backgrounds
# can change per example; both annotation colors remain reserved throughout.


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _boundary(p, shape):
    return p[0] in (0, shape[0] - 1) or p[1] in (0, shape[1] - 1)


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    dr, dc = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
    return [(a[0] + i * dr, a[1] + i * dc)
            for i in range(max(abs(a[0] - b[0]), abs(a[1] - b[1])) + 1)]


def _polyline(points):
    cells = [points[0]]
    for a, b in zip(points, points[1:]):
        cells.extend(_line(a, b)[1:])
    return cells


def _ray(p, d, shape):
    cells = []
    p = _add(p, d)
    while _inside(p, shape):
        cells.append(p)
        p = _add(p, d)
    return cells


def _neighbors(p, cells):
    return [q for d in _DIRS if (q := _add(p, d)) in cells]


def _straight(p, cells):
    ns = _neighbors(p, cells)
    return len(ns) == 2 and ns[0][0] + ns[1][0] == 2*p[0] and ns[0][1] + ns[1][1] == 2*p[1]


def _background(inp):
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > inp.size // 2, 'background must be visually identifiable'
    return bg


def _reconstruct(inp, repair, direction):
    """Recover separate color layers using only the observed grid.

    Marked corners/endpoints are waypoints. Unmarked omissions are short,
    straight gaps. Crossing pixels are retained in both geometrical layers;
    their visible owner is decided later from training evidence.
    """
    bg = _background(inp)
    assert bg not in (repair, direction)
    colors = sorted(set(map(int, np.unique(inp))) - {bg, repair, direction})
    assert colors
    work = inp.copy()
    pending = {tuple(map(int, p)) for p in np.argwhere(inp == repair)}
    for _ in range(len(pending) + 1):
        changed = False
        for p in sorted(pending):
            immediate = {int(work[q]) for d in _DIRS
                         if _inside(q := _add(p, d), inp.shape) and int(work[q]) in colors}
            candidates = immediate
            if len(candidates) != 1:
                aligned = set()
                for d in ((1, 0), (0, 1)):
                    a, b = _add(p, d), _add(p, (-d[0], -d[1]))
                    if _inside(a, inp.shape) and _inside(b, inp.shape):
                        if int(work[a]) == int(work[b]) and int(work[a]) in colors:
                            aligned.add(int(work[a]))
                candidates = aligned
            if len(candidates) == 1:
                work[p] = next(iter(candidates))
                pending.remove(p)
                changed = True
        if not changed:
            break
    assert not pending, 'repair marker has no unique trace witness'
    layers = {color: {tuple(map(int, p)) for p in np.argwhere(work == color)} for color in colors}
    # Bridge at most two missing cells, simultaneously. No arbitrary extension
    # of endpoints, diagonal interpolation, or connection of distant fragments.
    for color, cells in layers.items():
        original = set(cells)
        for p in sorted(original):
            for d in ((1, 0), (0, 1)):
                for distance in (2, 3):
                    q = p[0]+distance*d[0], p[1]+distance*d[1]
                    if q not in original:
                        continue
                    perpendicular = (-d[1], d[0])
                    # Two already turning fragments separated by empty space
                    # are not evidence for a connecting stroke. This matters
                    # at the parallel zigzags and adjacent corners in source.
                    def turns_at(x):
                        return any(_add(x, t) in original for t in
                                   (perpendicular, (-perpendicular[0], -perpendicular[1])))
                    if turns_at(p) and turns_at(q):
                        continue
                    between = _line(p, q)[1:-1]
                    if any(int(inp[x]) == direction for x in between):
                        continue
                    cells.update(between)
    return bg, layers


def _edits(inp, layers, direction):
    """Trace an old branch to a boundary before changing any layer."""
    edits = []
    for q0 in np.argwhere(inp == direction):
        q = tuple(map(int, q0))
        candidates = []
        for d in _DIRS:
            pivot = q[0]-d[0], q[1]-d[1]
            old_start = pivot[0]-d[0], pivot[1]-d[1]
            for color, cells in layers.items():
                if pivot not in cells or old_start not in cells or q in cells:
                    continue
                perpendicular = (-d[1], d[0])
                if not any(_add(pivot, t) in cells for t in (perpendicular, (-perpendicular[0], -perpendicular[1]))):
                    continue
                previous, current, old = pivot, old_start, []
                for _ in range(inp.size):
                    if current in old:
                        break
                    old.append(current)
                    following = [x for x in _neighbors(current, cells) if x != previous]
                    if not following:
                        if _boundary(current, inp.shape):
                            candidates.append({'pivot': pivot, 'color': color, 'direction': d,
                                               'old': set(old), 'walk': tuple([pivot]+old),
                                               'new': set(_ray(pivot, d, inp.shape))})
                        break
                    if len(following) != 1:
                        break
                    previous, current = current, following[0]
        assert len(candidates) == 1, 'direction marker must identify one boundary-reaching terminal branch'
        edits.append(candidates[0])
    for i, a in enumerate(edits):
        for b in edits[i+1:]:
            assert a['color'] != b['color'] or not a['old'] & b['old']
            assert a['pivot'] not in b['old'] or a['color'] != b['color']
            assert b['pivot'] not in a['old'] or a['color'] != b['color']
    return edits


def _retained(layers, edits):
    kept = {c: set(cells) for c, cells in layers.items()}
    rays = {c: set() for c in layers}
    # All deletion precedes any drawing, including intersections of new rays
    # with branches removed by a different marker.
    for edit in edits:
        kept[edit['color']].difference_update(edit['old'])
        rays[edit['color']].update(edit['new'])
    return kept, rays


def _render(shape, bg, layers, order, rays=None):
    out = np.full(shape, bg, dtype=np.int8)
    for collection in (layers, rays or {}):
        for c in order:
            for p in collection.get(c, ()):
                out[p] = c
    return out


def _predict(inp, repair, direction, order):
    bg, layers = _reconstruct(inp, repair, direction)
    assert set(layers) <= set(order), 'payload convention absent from training'
    edits = _edits(inp, layers, direction)
    kept, rays = _retained(layers, edits)
    return _render(inp.shape, bg, kept, order, rays)


def _learn(train):
    """Infer roles and precedence solely from copies of training pairs."""
    disappeared = set()
    output_colors = set()
    for p in train:
        disappeared.update(set(map(int, np.unique(p['input']))) - set(map(int, np.unique(p['output']))))
        output_colors.update(map(int, np.unique(p['output'])))
    annotations = disappeared - output_colors
    assert len(annotations) == 2, 'both reserved annotation roles must be demonstrated'
    answers = []
    for repair, direction in (tuple(sorted(annotations)), tuple(sorted(annotations, reverse=True))):
        try:
            records, colors, edges = [], set(), set()
            for pair in train:
                inp, out = pair['input'], pair['output']
                bg, layers = _reconstruct(inp, repair, direction)
                kept, rays = _retained(layers, _edits(inp, layers, direction))
                colors.update(layers)
                records.append((inp.shape, bg, kept, rays, out))
                for p in set().union(*kept.values(), *rays.values()):
                    owners = [c for c in rays if p in rays[c]] or [c for c in kept if p in kept[c]]
                    owner = int(out[p])
                    assert owner in owners
                    edges.update((c, owner) for c in owners if c != owner)
            # Full coverage: a unique topological order for colors that can
            # meet in queries. The calibration demonstration supplies it.
            remaining, order = set(colors), []
            while remaining:
                roots = sorted(c for c in remaining if not any(b == c and a in remaining for a, b in edges))
                assert len(roots) == 1, 'crossing precedence is not uniquely taught'
                order.extend(roots)
                remaining.remove(roots[0])
            for shape, bg, kept, rays, out in records:
                assert np.array_equal(_render(shape, bg, kept, order, rays), out)
            answers.append((repair, direction, tuple(order)))
        except AssertionError:
            pass
    assert len(answers) == 1, 'training must distinguish both marker meanings and crossing ownership'
    return answers[0]


def solve(train_pairs, test_input):
    """Prediction API: no labels, RNG, seed, or generation metadata for query."""
    repair, direction, order = _learn(train_pairs)
    return _predict(test_input.copy(), repair, direction, order)


def _valid_path(cells):
    s = set(cells)
    return len(s) == len(cells) and all(len(_neighbors(p, s)) <= 2 for p in s)


def _random_path(rng, shape, terminal=False):
    h, w = shape
    # Nonuniform spacings, segment lengths and numbers of bends. Endpoints
    # and bends are kept at least three cells apart to distinguish continuations.
    if terminal:
        side = int(rng.integers(4))
        if side < 2:
            start = (0 if side == 0 else h-1, int(rng.integers(1, w-1)))
            axis = 0
        else:
            start = (int(rng.integers(1, h-1)), 0 if side == 2 else w-1)
            axis = 1
    else:
        start = (int(rng.integers(h)), int(rng.integers(w)))
        axis = int(rng.integers(2))
    points = [start]
    desired = int(rng.integers(2 if terminal else 1, 8))
    for _ in range(desired):
        current = points[-1]
        choices = [v for v in range(1, shape[axis]-1) if abs(v-current[axis]) >= 3]
        rng.shuffle(choices)
        for v in choices:
            q = list(current)
            q[axis] = v
            candidate = points + [tuple(q)]
            if _valid_path(_polyline(candidate)):
                points = candidate
                break
        else:
            break
        axis = 1-axis
    return points


def _compatible(layers, color, cells):
    s = set(cells)
    if not _valid_path(cells):
        return False
    for c, other in layers.items():
        if c == color:
            if s & other or any(_neighbors(p, other) for p in s):
                return False
        else:
            for p in s & other:
                if not _straight(p, s) or not _straight(p, other):
                    return False
                # They must be transverse strokes, never coincident segments.
                if set(_neighbors(p, s)) & set(_neighbors(p, other)):
                    return False
            # Side-by-side strokes obscure component identity and pivots.
            for p in s-other:
                ns = _neighbors(p, other)
                if ns and not any(q in s for q in ns):
                    return False
    return True


def _random_scene(rng, shape, colors, redirect_count, directions):
    layers = {c: set() for c in colors}
    edits = []
    for index in range(redirect_count):
        color = colors[index % len(colors)]
        options = []
        for _ in range(80):
            points = _random_path(rng, shape, terminal=True)
            if len(points) < 3:
                continue
            path = _polyline(points)
            if not _compatible(layers, color, path):
                continue
            for j in range(1, len(points)-1):
                pivot = points[j]
                prev = points[j-1]
                d = int(np.sign(pivot[0]-prev[0])), int(np.sign(pivot[1]-prev[1]))
                if d not in directions:
                    continue
                q = _add(pivot, d)
                if q in path or not _inside(q, shape) or any(q in s for s in layers.values()):
                    continue
                at = path.index(pivot)
                options.append((path, {'pivot': pivot, 'direction': d, 'color': color,
                                      'old': set(path[:at]), 'walk': tuple(reversed(path[:at+1])),
                                      'new': set(_ray(pivot, d, shape))}))
            if options:
                break
        if not options:
            return None
        path, edit = options[int(rng.integers(len(options)))]
        layers[color].update(path)
        edits.append(edit)
        # Occasional unequal-arm junction at the pivot; old branch stays unique.
        if rng.random() < .22:
            pivot, d = edit['pivot'], edit['direction']
            for side in ((-d[1], d[0]), (d[1], -d[0])):
                arm = _ray(pivot, side, shape)[:int(rng.integers(2, 6))]
                s = layers[color]
                if len(arm) >= 2 and not any(p in s for p in arm) and all(
                    not (_neighbors(p, s) and p != arm[0]) for p in arm):
                    s.update(arm)
                    break
    target_count = int(rng.integers(max(1, len(colors)), len(colors)+4))
    for i in range(target_count):
        color = colors[i % len(colors)]
        if i < redirect_count and layers[color] and rng.random() < .65:
            continue
        for _ in range(30):
            path = _polyline(_random_path(rng, shape))
            if len(path) >= 4 and _compatible(layers, color, path):
                layers[color].update(path)
                break
    layers = {c: s for c, s in layers.items() if s}
    if not layers:
        return None
    return layers, edits


def _calibration(rng, shape, colors):
    # All pairs of colors meet in this demonstration. Independent spacings,
    # order, extra components and symmetries vary the crossing witnesses.
    n = len(colors)
    h, w = shape
    rows = [int(v)+2+2*i for i,v in enumerate(sorted(rng.choice(h-4-2*(n-1),n,replace=False)))]
    cols = [int(v)+2+2*i for i,v in enumerate(sorted(rng.choice(w-4-2*(n-1),n,replace=False)))]
    shuffled = list(colors)
    rng.shuffle(shuffled)
    layers = {}
    for i, color in enumerate(shuffled):
        layers[color] = set(_polyline([(0, cols[i]), (rows[i], cols[i]), (rows[i], w-1)]))
    for _ in range(int(rng.integers(3))):
        color = int(rng.choice(colors))
        for _ in range(20):
            path = _polyline(_random_path(rng,shape))
            if len(path) >= 4 and _compatible(layers,color,path):
                layers[color].update(path)
                break
    return layers, []


def _damage(rng, shape, bg, repair, direction, order, layers, edits, mode, dense=False):
    complete = _render(shape, bg, layers, order)
    kept, rays = _retained(layers, edits)
    out = _render(shape, bg, kept, order, rays)
    # A straight new ray must not create a broad same-color ribbon alongside
    # a retained segment. Reject merged 2-by-2 blocks rather than hiding them.
    for color in layers:
        mask = out == color
        if np.any(mask[:-1,:-1] & mask[1:,:-1] & mask[:-1,1:] & mask[1:,1:]):
            return None
    inp = complete.copy()
    qs = [_add(e['pivot'], e['direction']) for e in edits]
    if len(set(qs)) != len(qs) or any(inp[p] != bg for p in qs):
        return None
    for p in qs:
        inp[p] = direction
    try:
        inferred_bg, recovered = _reconstruct(inp, repair, direction)
        if recovered != layers or inferred_bg != bg:
            return None
        inferred = _edits(inp, recovered, direction)
        if {(e['pivot'], e['color'], frozenset(e['old'])) for e in inferred} != {
                (e['pivot'], e['color'], frozenset(e['old'])) for e in edits}:
            return None
    except AssertionError:
        return None
    candidates = [p for c,s in layers.items() for p in s if complete[p] == c]
    rng.shuffle(candidates)
    if mode != 'direction':
        # Prefer a diverse set of semantic sites, including annotated pivots,
        # terminal endpoints, bends, and ordinary straight waypoints.
        special = [e['pivot'] for e in edits]
        special += [p for c,s in layers.items() for p in s if not _straight(p,s) and complete[p] == c]
        rng.shuffle(special)
        if rng.random() < .7:
            candidates = list(dict.fromkeys(special+candidates))
        desired = min(int(rng.integers(1, 11 if dense else 8)), max(1,len(candidates)//3))
        successes = 0
        for p in candidates:
            previous = int(inp[p])
            inp[p] = repair
            try:
                _, reconstructed = _reconstruct(inp, repair, direction)
                assert reconstructed == layers
                assert np.array_equal(_predict(inp, repair, direction, order), out)
            except AssertionError:
                inp[p] = previous
            else:
                successes += 1
                if successes == desired:
                    break
        if successes == 0:
            return None
    # Sparse omissions can neighbor annotations, including along deleted paths.
    candidates = [p for p in candidates if inp[p] not in (repair, direction)]
    rng.shuffle(candidates)
    desired_gaps = min(int(rng.integers(0, 7 if dense else 4)),max(1,len(candidates)//6))
    successes = 0
    if desired_gaps:
        for p in candidates:
            previous = int(inp[p])
            inp[p] = bg
            try:
                _, reconstructed = _reconstruct(inp, repair, direction)
                assert reconstructed == layers
                assert np.array_equal(_predict(inp, repair, direction, order), out)
            except AssertionError:
                inp[p] = previous
            else:
                successes += 1
                if successes == desired_gaps:
                    break
    # Overwrite crossing owners with an underlying payload pixel. The learned
    # precedence repairs this corruption; geometry independently reveals layers.
    overlaps = [p for p in candidates if sum(p in s for s in layers.values()) > 1]
    rng.shuffle(overlaps)
    for p in overlaps[:int(rng.integers(0, 4))]:
        wrong = [c for c,s in layers.items() if p in s and c != complete[p]]
        if not wrong:
            continue
        previous = int(inp[p])
        inp[p] = wrong[int(rng.integers(len(wrong)))]
        try:
            _, reconstructed = _reconstruct(inp, repair, direction)
            assert reconstructed == layers
        except AssertionError:
            inp[p] = previous
    try:
        assert np.array_equal(_predict(inp, repair, direction, order), out)
        assert _background(out) == bg
    except AssertionError:
        return None
    if np.array_equal(inp, out):
        return None
    return {'input': inp, 'output': out}


def _transform(pair, k, reflect=False):
    return {role: np.rot90(np.fliplr(grid) if reflect else grid, k).copy()
            for role, grid in pair.items()}


def _make_pair(rng, params, mode, size_range, directions, calibration=False, teach_ray=False):
    repair, direction, order = params['repair'], params['direction'], params['order']
    for attempt in range(100):
        bg = int(rng.choice(params['backgrounds']))
        low, high = size_range
        shape = (int(rng.integers(low, high+1)), int(rng.integers(low, high+1)))
        if calibration:
            shape = tuple(max(n, 3*len(order)+3) for n in shape)
            layers, edits = _calibration(rng, shape, order)
        else:
            # Payload subsets vary independently from marker modes. Different
            # redirected colors can coexist, and all colors can play either role.
            count = int(rng.integers(1, len(order)+1))
            colors = list(map(int, rng.choice(order, count, replace=False)))
            n_redirect = 0 if mode == 'repair' else int(rng.integers(1, min(3, count+1)+1))
            if teach_ray:
                colors = list(order)
                rng.shuffle(colors)
                n_redirect = 1
            scene = _random_scene(rng, shape, colors, n_redirect, directions)
            if scene is None:
                continue
            layers, edits = scene
            if teach_ray:
                # Demonstrate that a freshly drawn ray can cover a retained
                # higher-precedence trace. Later tests may reuse this convention.
                good = False
                for e in edits:
                    for c,s in layers.items():
                        if order.index(c) > order.index(e['color']) and (s-e['old']) & e['new']:
                            good = True
                if not good:
                    continue
        pair = _damage(rng, shape, bg, repair, direction, order, layers, edits, mode,
                       dense=bool(rng.integers(2)))
        if pair is None:
            continue
        if not params.get('allow_ray_meetings',True) and any(
                a['color'] != b['color'] and a['new'] & b['new']
                for i,a in enumerate(edits) for b in edits[i+1:]):
            continue
        if calibration:
            pair = _transform(pair, int(rng.integers(4)), bool(rng.integers(2)))
        return pair
    # Deterministic bounded fallback retains requested marker semantics and
    # provides a witness for ray precedence, with randomized lengths/palette.
    bg = int(rng.choice(params['backgrounds']))
    h, w = max(18, size_range[0]), max(19, size_range[0])
    if calibration:
        layers, edits = _calibration(rng, (h,w), order)
    else:
        c, other = order[0], order[-1]
        r, col = 6+int(rng.integers(3)), 7+int(rng.integers(3))
        path = _polyline([(0,col+4), (r,col+4), (r,col), (h-3,col)])
        layers = {c:set(path), other:set(_line((r-3,col-3),(r+3,col-3)))}
        pivot = (r,col)
        edits = [] if mode == 'repair' else [{'pivot':pivot,'color':c,'direction':(0,-1),
                   'old':set(path[:path.index(pivot)]),'new':set(_ray(pivot,(0,-1),(h,w)))}]
    pair = _damage(rng, (h,w), bg, repair, direction, order, layers, edits, mode)
    assert pair is not None, 'constructive fallback failed'
    if not calibration:
        desired = directions[int(rng.integers(len(directions)))]
        k = ((0,-1), (1,0), (0,1), (-1,0)).index(desired)
        pair = _transform(pair, k)
    return pair


def _pair_key(pair):
    return tuple((a.shape, a.astype(np.int8).tobytes()) for a in (pair['input'], pair['output']))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _resolve_counts(seed, rng, num_train, num_test)
    palette = list(map(int, rng.permutation(10)))
    count = int(rng.integers(2, 5))
    params = {'repair':palette[0], 'direction':palette[1], 'order':tuple(palette[2:2+count]),
              'backgrounds':tuple(palette[2+count:])}
    regime = int(rng.integers(4))
    orientation = int(rng.integers(4))
    homogeneous = bool(rng.random() < .22)
    held_out = bool(rng.random() < .4) and not homogeneous
    train_dirs = [_DIRS[orientation]] if homogeneous else [d for i,d in enumerate(_DIRS) if not held_out or i != orientation]
    test_dirs = [_DIRS[orientation]] if homogeneous or held_out else list(_DIRS)
    size_mode = int(rng.integers(4))
    train_size, test_size = ((9,30),(9,30))
    if size_mode == 1:
        train_size, test_size = (10,21),(17,30)
    elif size_mode == 2:
        train_size, test_size = (17,30),(9,20)
    elif size_mode == 3:
        train_size, test_size = (9,17),(9,17)
    # A separate-demonstrations-to-combination regime remains available.
    # Other regimes mix operations in demonstrations or ask single-role queries.
    modes = ['repair', 'direction' if regime == 0 else 'mixed']
    modes += [str(rng.choice(['repair','direction'] if regime == 0 else ['repair','direction','mixed']))
              for _ in range(n_train-2)]
    train = [_make_pair(rng, params, mode, train_size, train_dirs,
                        calibration=(i == 0), teach_ray=(i == 1)) for i,mode in enumerate(modes)]
    params['allow_ray_meetings'] = False
    for pair in train:
        _,layers = _reconstruct(pair['input'],params['repair'],params['direction'])
        edits = _edits(pair['input'],layers,params['direction'])
        params['allow_ray_meetings'] |= any(a['color'] != b['color'] and a['new'] & b['new']
                                           for i,a in enumerate(edits) for b in edits[i+1:])
    test_modes = ['mixed' if regime == 0 else str(rng.choice(['repair','direction','mixed'])) for _ in range(n_test)]
    test = [_make_pair(rng, params, mode, test_size, test_dirs) for mode in test_modes]
    # The crossing demonstration need not be first in the returned puzzle.
    rng.shuffle(train)
    puzzle = {'train':train,'test':test}
    # Duplicates are extremely unlikely, but resolve them constructively using
    # unused recolorings of background rather than an unbounded retry loop.
    seen = set()
    for pair in train+test:
        key = _pair_key(pair)[0]
        if key in seen:
            old_bg = _background(pair['input'])
            for new_bg in params['backgrounds']:
                candidate = {role:np.where(a == old_bg, new_bg, a).astype(np.int8) for role,a in pair.items()}
                if _pair_key(candidate)[0] not in seen:
                    pair.update(candidate)
                    break
        seen.add(_pair_key(pair)[0])
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    for pair in puzzle['train']+puzzle['test']:
        assert isinstance(pair,dict) and set(pair) == {'input','output'}
        for a in pair.values():
            assert isinstance(a,np.ndarray) and a.ndim == 2
            assert all(1 <= n <= 30 for n in a.shape)
            assert np.issubdtype(a.dtype,np.integer) and np.all((0 <= a)&(a <= 9))
        inp, out = pair['input'], pair['output']
        assert inp.shape == out.shape and not np.array_equal(inp,out)
        bg = _background(inp)
        for color in set(map(int,np.unique(out))) - {bg}:
            mask = out == color
            assert not np.any(mask[:-1,:-1]&mask[1:,:-1]&mask[:-1,1:]&mask[1:,1:]), 'trace became a thick block'
        key = _pair_key(pair)[0]
        assert key not in seen
        seen.add(key)
        if inp.shape == (12,12):
            assert ''.join(map(str,map(int,inp.flat))) not in _SOURCE_INPUTS
    repair, direction, order = _learn(puzzle['train'])
    for pair in puzzle['train']+puzzle['test']:
        # Compare EVERY pixel after prediction, including unaffected content.
        predicted = _predict(pair['input'].copy(),repair,direction,order)
        assert np.array_equal(predicted,pair['output']), 'output violates repair/rerouting semantics'
    # Ray-over-retained precedence is an extension explicitly witnessed in train.
    witnessed = False
    needed = False
    ray_meeting_witness, ray_meeting_needed = False, False
    for split in ('train','test'):
        for pair in puzzle[split]:
            _, layers = _reconstruct(pair['input'],repair,direction)
            kept,rays = _retained(layers,_edits(pair['input'],layers,direction))
            meeting = any(rays[a]&rays[b] for a in rays for b in rays if a != b)
            hit = any(s & kept.get(c,set()) for a,s in rays.items() for c in kept
                      if order.index(c) > order.index(a))
            if split == 'train':
                witnessed |= hit
                ray_meeting_witness |= meeting
            else:
                needed |= hit
                ray_meeting_needed |= meeting
    assert not needed or witnessed, 'new crossing convention was not demonstrated'
    assert not ray_meeting_needed or ray_meeting_witness, 'multiple-ray crossing was not demonstrated'


def _component_sets(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            p = stack.pop()
            for q in _neighbors(p, remaining):
                remaining.remove(q)
                component.add(q)
                stack.append(q)
        result.append(component)
    return result


def _turn_count(walk):
    ds = [(b[0]-a[0],b[1]-a[1]) for a,b in zip(walk,walk[1:])]
    return sum(a != b for a,b in zip(ds,ds[1:]))


def _segment_lengths(layers):
    lengths = []
    for cells in layers.values():
        for p in cells:
            for d in ((1,0),(0,1)):
                if _add(p,(-d[0],-d[1])) in cells or _add(p,d) not in cells:
                    continue
                length,q = 1,p
                while (q := _add(q,d)) in cells:
                    length += 1
                lengths.append(length)
    return sorted(lengths)


def measure_pair(pair, convention):
    """Review features reconstructed from public grids, never sampled metadata."""
    repair,direction,order = convention
    inp,out = pair['input'],pair['output']
    bg,layers = _reconstruct(inp,repair,direction)
    edits = _edits(inp,layers,direction)
    complete = _render(inp.shape,bg,layers,order)
    kept,rays = _retained(layers,edits)
    annotated = {tuple(map(int,p)) for p in np.argwhere(inp == repair)}
    removed = set().union(*(e['old'] for e in edits)) if edits else set()
    markers = {'straight':0,'bend':0,'endpoint':0,'junction':0}
    for p in annotated:
        cells = layers[int(complete[p])]
        degree = len(_neighbors(p,cells))
        name = 'endpoint' if degree <= 1 else 'junction' if degree > 2 else 'straight' if _straight(p,cells) else 'bend'
        markers[name] += 1
    components = [component for s in layers.values() for component in _component_sets(s)]
    all_new = set().union(*rays.values())
    erased_crossings = sum(int(inp[p]) == e['color'] and any(p in s for c,s in kept.items() if c != e['color'])
                           and p not in all_new for e in edits for p in e['old'])
    damaged_crossings = int(np.count_nonzero((inp != complete) & np.isin(inp,list(layers))))
    adjacent_damage = sum(any(_inside(q := _add(p,d),inp.shape) and inp[q] in (bg,repair,direction)
                               and complete[q] != bg for d in _DIRS) for p in annotated)
    return {
        'shape':list(inp.shape), 'background':bg, 'payload_count':len(layers),
        'payloads':sorted(layers), 'component_count':len(components),
        'same_color_components':max(len(_component_sets(s)) for s in layers.values()),
        'component_lengths':sorted(map(len,components)),
        'segment_lengths':_segment_lengths(layers),
        'turns':sum(not _straight(p,s) and len(_neighbors(p,s)) == 2 for s in layers.values() for p in s),
        'junctions':sum(len(_neighbors(p,s)) > 2 for s in layers.values() for p in s),
        'repair_count':len(annotated), 'direction_count':len(edits),
        'mode':'mixed' if annotated and edits else 'direction' if edits else 'repair',
        'repair_sites':markers, 'marked_pivots':sum(e['pivot'] in annotated for e in edits),
        'corner_pivots':sum(len(_neighbors(e['pivot'],layers[e['color']])) == 2 for e in edits),
        'junction_pivots':sum(len(_neighbors(e['pivot'],layers[e['color']])) > 2 for e in edits),
        'deleted_markers':len(annotated & removed), 'adjacent_damage':adjacent_damage,
        'boundary_markers':sum(_boundary(p,inp.shape) for p in annotated),
        'old_marker_counts':[len(annotated & e['old']) for e in edits],
        'old_gap_counts':[int(sum(inp[p] == bg for p in e['old'])) for e in edits],
        'retained_turns':sum(not _straight(p,s) and len(_neighbors(p,s)) == 2 for s in kept.values() for p in s),
        'gap_count':int(np.count_nonzero((inp == bg)&(complete != bg))),
        'crossings':sum(sum(p in s for s in layers.values()) > 1 for p in set().union(*layers.values())),
        'damaged_crossings':damaged_crossings, 'restored_crossings':erased_crossings,
        'trace_recolors':int(np.count_nonzero((inp != out)&np.isin(inp,list(layers))&np.isin(out,list(layers)))),
        'new_ray_crossings':sum(len(e['new'] & s) for e in edits for c,s in kept.items() if c != e['color']),
        'new_old_interactions':sum(len(a['new'] & b['old']) for i,a in enumerate(edits) for j,b in enumerate(edits) if i != j),
        'new_new_interactions':sum(len(a['new'] & b['new']) for i,a in enumerate(edits) for b in edits[i+1:] if a['color'] != b['color']),
        'different_color_redirects':len({e['color'] for e in edits}) > 1,
        'redirect_colors':[e['color'] for e in edits],
        'directions':[list(e['direction']) for e in edits],
        'old_bends':[_turn_count(e['walk']) for e in edits],
        'old_lengths':[len(e['old']) for e in edits], 'ray_lengths':[len(e['new']) for e in edits],
        'repair_quadrants':sorted({2*int(p[0] >= inp.shape[0]/2)+int(p[1] >= inp.shape[1]/2) for p in annotated}),
    }


def _source_regressions():
    # Exact source fixtures. Orders are scene-specific crossing interpretations,
    # not a claimed globally learnable literal-color hierarchy for the source.
    # These test reconstruction and routing against evidence; they are NOT an
    # independent solver of the unresolved original precedence convention.
    fixtures = (
        ("000000000300 111100000300 000100000300 233320000300 000130000300 000131100300 000030100300 000030100300 000023330200 000000100000 000000110110 000000000011",
         "000000000300 111100000300 000100000300 333130000300 000130000300 000111100300 000030100300 000030100300 000033133300 000000100000 000000111110 000000000011", (3,1)),
        ("000000000070 777700000070 333330333470 000700003070 000000000070 000777703000 000000703070 007777703070 007000003070 330333333070 007777777770 000000000000",
         "000000000070 777700000070 000700003333 000700003070 000700003070 000777703070 000000703070 007777703070 007000003070 333333333070 007777777770 000000000000", (7,3)),
        ("007030600200 007000000100 007030600100 337320600100 007000606166 007000000100 707000000000 000000000100 000021111200 000010000000 000000000000 000010000000",
         "007030600100 007030600100 007030600100 333330600100 007000666666 007000000100 777000000100 000000000100 000011111100 000010000000 000010000000 000010000000", (7,1,3,6)),
        ("010000100060 010000100060 616666060660 010000100040 000000100000 011101100000 000000000000 000000000000 000000000000 000000000000 000000000000 000000000000",
         "010000100000 010000100000 616666166660 010000100060 010000100060 011111100060 000000000060 000000000060 000000000060 000000000060 000000000060 000000000060", (6,1)),
        ("000070000070 000070000070 000070000070 427720027720 070000070040 070000070000 070000070000 666666606666 070000070000 027777720000 000000000000 000000000000",
         "000000000000 000000000000 000000000000 770000077770 070000070070 070000070070 070000070070 676666676676 070000070070 077777770070 000000000070 000000000070", (6,7)),
    )
    for left,right,order in fixtures:
        inp = np.array([list(map(int,row)) for row in left.split()],dtype=np.int8)
        out = np.array([list(map(int,row)) for row in right.split()],dtype=np.int8)
        for k in range(4):
            pair = _transform({'input':inp,'output':out},k)
            assert np.array_equal(_predict(pair['input'],2,4,order),pair['output'])


def _puzzles_equal(a,b):
    return all(len(a[s]) == len(b[s]) and all(_pair_key(x) == _pair_key(y) for x,y in zip(a[s],b[s])) for s in ('train','test'))


def self_test() -> None:
    _source_regressions()
    automatic = [generate_puzzle(seed) for seed in range(42)]
    features = []
    held_out, homogeneous, mixed_demonstrations = set(), 0, 0
    within_payload, within_components, size_shifts = 0, 0, set()
    split_modes = {'train':set(),'test':set()}
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        convention = _learn(puzzle['train'])
        pair_features = {split:[measure_pair(pair,convention) for pair in puzzle[split]]
                         for split in ('train','test')}
        train_dirs = {tuple(d) for f in pair_features['train'] for d in f['directions']}
        test_dirs = {tuple(d) for f in pair_features['test'] for d in f['directions']}
        held_out.update(test_dirs-train_dirs)
        homogeneous += len(train_dirs|test_dirs) == 1
        mixed_demonstrations += any(f['mode'] == 'mixed' for f in pair_features['train'])
        group = pair_features['train']+pair_features['test']
        within_payload += len({f['payload_count'] for f in group}) > 1
        within_components += len({f['component_count'] for f in group}) > 1
        mean_areas = [sum(np.prod(f['shape']) for f in pair_features[s])/len(pair_features[s])
                      for s in ('train','test')]
        size_shifts.add(int(np.sign(mean_areas[1]-mean_areas[0])))
        for split in split_modes:
            split_modes[split].update(f['mode'] for f in pair_features[split])
        for split in ('train','test'):
            for pair in puzzle[split]:
                features.append(measure_pair(pair,convention))
                bg,layers = _reconstruct(pair['input'],*convention[:2])
                edits = _edits(pair['input'],layers,convention[1])
                kept,rays = _retained(layers,list(reversed(edits)))
                assert np.array_equal(_render(pair['input'].shape,bg,kept,convention[2],rays),pair['output'])
                transformed = _transform(pair,1,True)
                assert np.array_equal(_predict(transformed['input'],*convention),transformed['output'])
        for query in puzzle['test']:
            # The prediction function's interface cannot access the test label.
            assert np.array_equal(solve(puzzle['train'],query['input'].copy()),query['output'])
    assert len({len(p['train']) for p in automatic}) >= 4
    assert len({len(p['test']) for p in automatic}) >= 3
    assert len({tuple(_pair_key(q) for q in p['train']+p['test']) for p in automatic}) == 42
    assert held_out == set(_DIRS) and homogeneous >= 3
    assert 5 <= mixed_demonstrations <= 37
    assert within_payload >= 35 and within_components >= 35
    assert {-1,1} <= size_shifts
    assert all(modes == {'repair','direction','mixed'} for modes in split_modes.values())
    assert _puzzles_equal(generate_puzzle(17),generate_puzzle(17))
    for i,(nt,nq) in enumerate(((2,1),(10,1),(10,4),(7,3),(None,1),(None,4),(2,None),(10,None))):
        p = generate_puzzle(800+i,nt,nq)
        validate_generated_puzzle(p)
        assert nt is None or len(p['train']) == nt
        assert nq is None or len(p['test']) == nq
        assert _puzzles_equal(p,generate_puzzle(800+i,nt,nq))
    for nt,nq in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(4,4),(True,1),(3,1.5)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Orientation invariance, full-pixel preservation, and marker/ray negatives.
    for p in automatic[:4]:
        for k in range(1,4):
            validate_generated_puzzle({s:[_transform(q,k,True) for q in p[s]] for s in p})
        q = p['test'][0]
        convention = _learn(p['train'])
        bg,layers = _reconstruct(q['input'],*convention[:2])
        edits = _edits(q['input'],layers,convention[1])
        sites = [tuple(map(int,x)) for x in np.argwhere(q['input'] == q['output'])[:3]]
        sites += [e['pivot'] for e in edits]
        sites += [min(e['new']) for e in edits]
        for point in sites:
            mutated = {'train':p['train'], 'test':[dict(pair) for pair in p['test']]}
            mutated['test'][0]['output'] = q['output'].copy()
            mutated['test'][0]['output'][point] = (int(q['output'][point])+1)%10
            try:
                validate_generated_puzzle(mutated)
            except AssertionError:
                pass
            else:
                raise AssertionError('corrupt output accepted')
    targeted = set()
    for puzzle in automatic:
        convention = _learn(puzzle['train'])
        for index,q in enumerate(puzzle['test']):
            inp,out = q['input'],q['output']
            bg,layers = _reconstruct(inp,*convention[:2])
            edits = _edits(inp,layers,convention[1])
            kept,_ = _retained(layers,edits)
            sites = {}
            for e in edits:
                if _turn_count(e['walk']):
                    sites['bent deletion'] = min(e['old'])
                for point in e['old']:
                    if any(point in s for c,s in kept.items() if c != e['color']):
                        sites['restored crossing'] = point
            for point in map(tuple,np.argwhere((inp != out)&np.isin(inp,list(layers))&np.isin(out,list(layers)))):
                sites['trace recolor'] = point
            for name,point in sites.items():
                if name in targeted:
                    continue
                targeted.add(name)
                mutated = {'train':puzzle['train'],'test':[dict(pair) for pair in puzzle['test']]}
                mutated['test'][index]['output'] = out.copy()
                mutated['test'][index]['output'][point] = (int(out[point])+1)%10
                try:
                    validate_generated_puzzle(mutated)
                except AssertionError:
                    pass
                else:
                    raise AssertionError(name+' mutation accepted')
    assert targeted == {'bent deletion','restored crossing','trace recolor'}
    # Meaningful distributions, reconstructed from actual returned grids.
    assert len({tuple(f['shape']) for f in features}) >= 60
    assert {f['mode'] for f in features} == {'repair','direction','mixed'}
    assert {tuple(d) for f in features for d in f['directions']} == set(_DIRS)
    assert {f['payload_count'] for f in features} >= {1,2,3,4}
    assert {f['direction_count'] for f in features} >= {0,1,2,3}
    assert {b for f in features for b in f['old_bends']} >= {0,1,2}
    assert min(n for f in features for n in f['ray_lengths']) == 1
    assert max(n for f in features for n in f['ray_lengths']) >= 20
    assert min(min(f['shape']) for f in features) <= 10
    assert max(max(f['shape']) for f in features) == 30
    for name in ('marked_pivots','junction_pivots','deleted_markers','adjacent_damage',
                 'damaged_crossings','restored_crossings','new_ray_crossings','new_old_interactions'):
        assert sum(f[name] for f in features) > 0, name
    for name in ('straight','bend','endpoint'):
        assert sum(f['repair_sites'][name] for f in features) >= 10, name
    assert sum(f['different_color_redirects'] for f in features) >= 3
    assert max(f['same_color_components'] for f in features) >= 2
    assert max(n for f in features for n in f['old_marker_counts']) >= 2
    assert max(n for f in features for n in f['old_gap_counts']) >= 2
    assert sum(f['boundary_markers'] for f in features) >= 10
    assert len({n for f in features for n in f['segment_lengths']}) >= 20
    assert len({f['background'] for f in features}) >= 8
    assert {q for f in features for q in f['repair_quadrants']} == {0,1,2,3}
    assert sum(len({_background(q['input']) for q in p['train']+p['test']}) > 1 for p in automatic) >= 35
