"""A perforated rail carrier whose transported defect halos affect clearance.

Only standard-library modules and NumPy are used. All semantic inference in
solve() is input-only; construction metadata is used only by the label renderer.
"""
from __future__ import annotations

import json
import numpy as np


WALL, FLOOR, RAIL, HALO, GOAL = 0, 1, 2, 3, 8
BODY_COLORS = (4, 5, 6, 7, 9)
HOST_SEEDS = (0, 137447902, 930271885, 89249627, 650569222, 960257451,
              772754959, 662387185, 823250992, 816395416, 710848444,
              220265133, 416012046, 779538214, 705243438, 633352363,
              238475597, 797288180, 96258088, 927255129, 955154322,
              964977693, 203801469, 117072591, 24072220, 961150617,
              519400112, 4400438, 165335402, 371052030, 149353555,
              896626057, 263829821, 595620341, 7042468, 809819486,
              682222162, 200842092, 630022955, 84662458, 531489855,
              483010144)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and
             bool(((0 <= a) & (a <= 9)).all()), "grid needs integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _neighbors(p, shape):
    y, x = p
    for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
        if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
            yield q


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        queue = [p]
        for p in queue:
            for q in _neighbors(p, mask.shape):
                if q in remaining:
                    remaining.remove(q)
                    queue.append(q)
        result.append(queue)
    return sorted(result, key=len, reverse=True)


def _dilate(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[y:y+h, x:x+w]
                                 for y in range(3) for x in range(3)])


def _decode(input_grid):
    """Complete the square and its occluded initial rail, from pixels only."""
    a = _grid(input_grid)
    colors = set(map(int, np.unique(a)))
    body_colors = colors & set(BODY_COLORS)
    _require(len(body_colors) == 1 and colors <= {0, 1, 2, 8} | body_colors,
             "one carrier color and the fixed terrain/rail/beacon roles required")
    body = body_colors.pop()
    groups = _components(a == body)
    _require(groups and len(groups[0]) >= 6, "carrier must retain an intact scaffold")
    _require(len(groups) == 1 or len(groups[0]) > len(groups[1]),
             "carrier component is ambiguous")
    points = np.asarray(groups[0])
    lo, hi = points.min(axis=0), points.max(axis=0)+1
    k = int(hi[0]-lo[0])
    _require(k >= 3 and k % 2 == 1 and hi[1]-lo[1] == k,
             "carrier bounding box must be an odd square")
    y, x = map(int, lo)
    tile = a[y:y+k, x:x+k]
    _require(bool(((tile == body) | (tile == FLOOR)).all()),
             "carrier holes must show floor")
    holes = tile == FLOOR
    _require(int(holes.sum()) <= k*k//3, "too little square evidence")
    _require(int((tile == body).sum()) == len(groups[0]),
             "carrier body must remain connected")
    extras = a == body
    extras[y:y+k, x:x+k] = False
    _require(all(len(g) <= 2 for g in _components(extras)),
             "external specks may have at most two orthogonally joined cells")
    for q in np.argwhere(extras):
        _require(not (y-1 <= q[0] <= y+k and x-1 <= q[1] <= x+k),
                 "specks must be separated from the carrier square")
    h = k//2
    start = (y+h, x+h)
    targets = np.argwhere(a == GOAL)
    _require(len(targets) == 1, "one docking beacon required")
    target = tuple(map(int, targets[0]))
    track = (a == RAIL) | (a == GOAL)
    exits = []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        q = start[0]+(h+1)*dy, start[1]+(h+1)*dx
        if 0 <= q[0] < a.shape[0] and 0 <= q[1] < a.shape[1] and track[q]:
            exits.append((dy, dx))
    _require(len(exits) == 1, "square must have one straight central rail exit")
    dy, dx = exits[0]
    for j in range(h+1):
        track[start[0]+j*dy, start[1]+j*dx] = True
    track_points = list(map(tuple, np.argwhere(track)))
    for p in track_points:
        degree = sum(bool(track[q]) for q in _neighbors(p, a.shape))
        _require(degree == (1 if p in (start, target) else 2),
                 "rail must be a single unbranched path")
    route, previous, p = [], None, start
    while True:
        _require(p not in route, "rail loop")
        route.append(p)
        if p == target:
            break
        options = [q for q in _neighbors(p, a.shape) if track[q] and q != previous]
        _require(len(options) == 1, "rail order is ambiguous")
        previous, p = p, options[0]
    _require(len(route) == len(track_points), "unrelated rail component")
    # Coordinates relative to the carrier's center. The safety envelope includes
    # the entire square, even its holes, plus the exact one-cell defect halo.
    padded_holes = np.pad(holes, 1)
    halo = _dilate(padded_holes) & ~padded_holes
    envelope = halo.copy()
    envelope[1:k+1, 1:k+1] = True
    offsets = np.argwhere(envelope) - (h+1)
    return dict(a=a, body=body, k=k, h=h, lo=(y, x), start=start,
                target=target, route=route, track=track, holes=holes,
                halo=halo, offsets=offsets, extras=extras)


def _last_safe(info, square_only=False, point_only=False):
    a, h = info['a'], info['h']
    if point_only:
        offsets = np.array([[0, 0]])
    elif square_only:
        offsets = np.array([(y, x) for y in range(-h, h+1)
                            for x in range(-h, h+1)])
    else:
        offsets = info['offsets']
    last = -1
    for i, p in enumerate(info['route']):
        pts = offsets + p
        if ((pts < 0).any() or (pts[:, 0] >= a.shape[0]).any()
                or (pts[:, 1] >= a.shape[1]).any()):
            break
        if (a[pts[:, 0], pts[:, 1]] == WALL).any():
            break
        last = i
    _require(last >= 0, "the starting safety envelope must fit")
    return last


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Translate along the completed rail until the first envelope collision."""
    d = _decode(input_grid)
    index = _last_safe(d)
    out = d['a'].copy()
    out[d['a'] == d['body']] = FLOOR
    # Only the occluded rail under the old carrier is restored; visible terrain
    # and the docking beacon retain their original values until the final stamp.
    y, x = d['lo']
    k, h = d['k'], d['h']
    old_box = np.zeros(out.shape, dtype=bool)
    old_box[y:y+k, x:x+k] = True
    out[old_box] = FLOOR
    out[d['track'] & old_box] = RAIL
    cy, cx = d['route'][index]
    y, x = cy-h, cx-h
    out[y:y+k, x:x+k] = d['body']
    view = out[y:y+k, x:x+k]
    view[d['holes']] = FLOOR
    halo_points = np.argwhere(d['halo']) + (cy-h-1, cx-h-1)
    if len(halo_points):
        out[halo_points[:, 0], halo_points[:, 1]] = HALO
    return out


def _counts(num_train, num_test):
    for v, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                or not low <= v <= high):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _polyline(rng, k, longer):
    """Grow a random coarse walk, then expand its independently spaced legs."""
    margin, gap = k//2+1, k+2
    limit = min(4, (29-2*margin)//gap+1)
    nr = int(rng.integers(1, limit+1))
    nc = int(rng.integers(1, limit+1))
    if nr*nc < 2:
        nc = 2
    if longer:
        nr, nc = max(2, nr), max(2, nc)
    def coords(n):
        gaps = np.full(n-1, gap, dtype=int)
        spare = 29-2*margin - int(gaps.sum())
        for j in rng.permutation(n-1):
            extra = int(rng.integers(0, min(4, spare)+1))
            gaps[j] += extra
            spare -= extra
        return np.r_[0, np.cumsum(gaps)]
    yy, xx = coords(nr), coords(nc)
    want = int(rng.integers(4 if longer else 2,
                            min(nr*nc, 11 if longer else 4)+1))
    best = []
    for _ in range(16):
        p = int(rng.integers(nr)), int(rng.integers(nc))
        nodes = [p]
        while len(nodes) < want:
            options = [q for q in _neighbors(nodes[-1], (nr, nc)) if q not in nodes]
            if not options:
                break
            nodes.append(options[int(rng.integers(len(options)))])
        if len(nodes) > len(best):
            best = nodes
        if len(best) == want:
            break
    centers = [(int(yy[y]), int(xx[x])) for y, x in best]
    route = [centers[0]]
    for a, b in zip(centers, centers[1:]):
        dy, dx = np.sign(np.subtract(b, a))
        p = a
        while p != b:
            p = p[0]+int(dy), p[1]+int(dx)
            route.append(p)
    arr = np.asarray(route)
    arr -= arr.min(axis=0)
    span = arr.max(axis=0)+1
    pads = []
    for extent in span:
        total = int(rng.integers(2*margin, min(30-int(extent), 2*margin+6)+1))
        before = int(rng.integers(margin, total-margin+1))
        pads.append((before, total-before))
    arr += [p[0] for p in pads]
    shape = tuple(int(span[j]+sum(pads[j])) for j in range(2))
    return list(map(tuple, arr.tolist())), shape


def _choose_holes(rng, k, anchor, rich):
    """Sparse holes that preserve a connected square with four supported sides."""
    for _ in range(80):
        holes = np.zeros((k, k), dtype=bool)
        # Anchors deliberately carry an outward halo; its side varies.
        if anchor:
            side = int(rng.integers(4))
            j = int(rng.integers(1, k-1))
            p = ((0, j), (k-1, j), (j, 0), (j, k-1))[side]
            holes[p] = True
        count = int(rng.integers(1, min(k*k//3, 7 if rich else 3)+1))
        if rich:
            count = max(min(4, k*k//3), count)
        while int(holes.sum()) < count:
            if holes.any() and rng.random() < .48:
                p = np.argwhere(holes)[int(rng.integers(int(holes.sum())))]
                q = p + rng.integers(-1, 2, size=2)
                if ((q < 0) | (q >= k)).any():
                    continue
                holes[tuple(q)] = True
            else:
                holes[int(rng.integers(k)), int(rng.integers(k))] = True
        body = ~holes
        groups = _components(body)
        if len(groups) == 1 and all((body[0].any(), body[-1].any(),
                                    body[:, 0].any(), body[:, -1].any())):
            return holes
    raise RuntimeError("bounded hole construction exhausted")


def _box(mask, p, radius, value=True):
    y, x = p
    mask[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1] = value


def _construct_pair(rng, k, body, outcome, longer=False, rich=False):
    """Labels use explicit route, scaffold and holes, never the input decoder."""
    for _ in range(180):
        route, shape = _polyline(rng, k, longer)
        holes = _choose_holes(rng, k, outcome in ('success', 'halo'), rich)
        h = k//2
        local_halo = set()
        for y, x in np.argwhere(holes):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = int(y+dy), int(x+dx)
                    if not (0 <= q[0] < k and 0 <= q[1] < k and holes[q]):
                        local_halo.add((q[0]-h, q[1]-h))
        square = {(y, x) for y in range(-h, h+1) for x in range(-h, h+1)}
        safety = square | local_halo
        floor = np.zeros(shape, dtype=bool)
        # Independent piecewise passage widths create widenings, passable
        # narrowings, short throats, and wide regions beyond earlier obstacles.
        breakpoints = sorted(set([0, len(route)] + list(map(int, rng.integers(
            1, len(route), size=int(rng.integers(2, 7)))))))
        for lo, hi in zip(breakpoints, breakpoints[1:]):
            radius = h + int(rng.choice((0, 0, 1, 2)))
            if outcome == 'mixed' and rng.random() < .3:
                radius = max(0, h-1)
            for p in route[lo:hi]:
                _box(floor, p, radius)
        for p in (route[0], route[-1]):
            _box(floor, p, h+1)
        if outcome == 'success':
            for cy, cx in route:
                for dy, dx in safety:
                    floor[cy+dy, cx+dx] = True
        elif outcome == 'halo':
            # Expose actual motion before the halo encounters the first narrow
            # section, while preserving all square placements everywhere.
            for p in route:
                _box(floor, p, h)
            prefix = int(rng.integers(h+2, max(h+3, len(route)//2)))
            for cy, cx in route[:prefix]:
                for dy, dx in safety:
                    floor[cy+dy, cx+dx] = True
        for p in route:
            floor[p] = True
        # Construction-time collision calculation is a scalar loop over the
        # explicit stencil, separate from solve's recovered vectorized masks.
        final = -1
        for i, (cy, cx) in enumerate(route):
            if not all(0 <= cy+dy < shape[0] and 0 <= cx+dx < shape[1]
                       and floor[cy+dy, cx+dx] for dy, dx in safety):
                break
            final = i
        if final < h+1 or (outcome == 'success' and final != len(route)-1):
            continue
        if outcome == 'halo' and final == len(route)-1:
            continue
        inp = np.where(floor, FLOOR, WALL).astype(np.int8)
        for p in route:
            inp[p] = RAIL
        inp[route[-1]] = GOAL
        sy, sx = route[0]
        inp[sy-h:sy+h+1, sx-h:sx+h+1] = body
        for y, x in np.argwhere(holes):
            inp[sy-h+y, sx-h+x] = FLOOR
        # Extras are genuinely floor specks, never occluded rail or wall cells.
        available = inp == FLOOR
        available[max(0, sy-h-1):sy+h+2, max(0, sx-h-1):sx+h+2] = False
        candidates = np.argwhere(available)
        extra_count = int(rng.integers(1, 7))
        if len(candidates) < extra_count:
            continue
        selected = []
        for j in rng.permutation(len(candidates)):
            p = tuple(map(int, candidates[int(j)]))
            if all(max(abs(p[0]-q[0]), abs(p[1]-q[1])) > 1 for q in selected):
                selected.append(p)
            if len(selected) == extra_count:
                break
        for p in selected:
            inp[p] = body
        out = np.where(floor, FLOOR, WALL).astype(np.int8)
        for p in route:
            out[p] = RAIL
        out[route[-1]] = GOAL
        cy, cx = route[final]
        out[cy-h:cy+h+1, cx-h:cx+h+1] = body
        for y, x in np.argwhere(holes):
            out[cy-h+y, cx-h+x] = FLOOR
        for dy, dx in local_halo:
            out[cy+dy, cx+dx] = HALO
        if not any(out[p] == FLOOR for p in selected):
            # Keep at least one speck deletion visible outside the final stamp.
            continue
        turns = int(rng.integers(4))
        reflect = bool(rng.integers(2))
        def transform(a):
            return np.rot90(np.fliplr(a) if reflect else a, turns).copy()
        pair = {'input': transform(inp), 'output': transform(out)}
        try:
            d = _decode(pair['input'])
            if _last_safe(d) != final or not np.array_equal(solve(pair['input']), pair['output']):
                continue
            if outcome == 'halo' and _last_safe(d, square_only=True) != len(route)-1:
                continue
        except ValueError:
            continue
        return pair
    raise RuntimeError("bounded carrier construction exhausted")


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    anchor_k = 3 if regime == 0 else (5 if regime == 2 else int(rng.choice((3, 5))))
    palette = list(map(int, rng.permutation(BODY_COLORS)))
    variable_color = bool(rng.integers(2))
    train_specs = [(anchor_k, 'success', False, False),
                   (anchor_k, 'halo', False, False)]
    for _ in range(num_train-2):
        train_specs.append((anchor_k if regime == 0 else int(rng.choice((3, 5))),
                            str(rng.choice(('success', 'halo', 'mixed'))), False, False))
    train_specs = [train_specs[int(i)] for i in rng.permutation(num_train)]
    test_specs = []
    for _ in range(num_test):
        k = int(rng.choice((5, 7))) if regime == 0 else anchor_k
        test_specs.append((k, str(rng.choice(('success', 'halo', 'mixed'))),
                           regime == 1, regime == 2))
    puzzle, seen = {'train': [], 'test': []}, set()
    index = 0
    for split, specs in (('train', train_specs), ('test', test_specs)):
        for k, outcome, longer, rich in specs:
            color = palette[index % len(palette)] if variable_color else palette[0]
            for _ in range(30):
                pair = _construct_pair(rng, k, color, outcome, longer, rich)
                signature = _signature(pair['input'])
                # The long-rail regime requires an observable extrapolation,
                # rather than merely requesting a more complex latent mode.
                length_shift = (split != 'test' or regime != 1 or
                                len(_decode(pair['input'])['route']) > max(
                                    len(_decode(q['input'])['route']) for q in puzzle['train']))
                if signature not in seen and length_shift:
                    break
            else:
                raise RuntimeError("could not obtain distinct inputs")
            seen.add(signature)
            puzzle[split].append(pair)
            index += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(d):
    final = _last_safe(d)
    headings = [tuple(np.subtract(b, a)) for a, b in zip(d['route'], d['route'][1:])]
    return dict(k=d['k'], holes=int(d['holes'].sum()), extras=int(d['extras'].sum()),
                turns=sum(a != b for a, b in zip(headings, headings[1:])),
                length=len(d['route']), travel=final,
                success=final == len(d['route'])-1,
                halo_changes_stop=final != _last_safe(d, square_only=True),
                point_changes_stop=final != _last_safe(d, point_only=True),
                protrusions=int(len(d['offsets'])-d['k']**2))


def validate_generated_puzzle(puzzle):
    """Validate supplied pixels, outputs, and the two teaching witnesses."""
    try:
        _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'},
                 "exact train/test keys required")
        _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
        _counts(len(puzzle['train']), len(puzzle['test']))
        seen, train_features, test_features = set(), [], []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                _require(isinstance(pair, dict) and set(pair) == {'input', 'output'},
                         "exact input/output keys required")
                inp, out = _grid(pair['input']), _grid(pair['output'])
                _require(inp.shape == out.shape, "shape must be preserved")
                key = _signature(inp)
                _require(key not in seen, "duplicate input")
                seen.add(key)
                d = _decode(inp)
                f = _features(d)
                _require(np.array_equal(solve(inp), out), "wrong transported stencil or stop")
                _require(bool((out[inp == WALL] == WALL).all()), "wall overwritten")
                _require(f['travel'] > 0 and f['holes'] > 0 and f['extras'] > 0,
                         "each generated pair must move and expose both corruptions")
                _require(bool((d['extras'] & (out == FLOOR)).any()),
                         "at least one exterior speck erasure must remain visible")
                cy, cx = d['route'][f['travel']]
                for y, x in np.argwhere(d['holes']):
                    _require(out[cy-d['h']+y, cx-d['h']+x] == FLOOR,
                             "missing cells must remain unfilled")
                if not f['success']:
                    _require(f['point_changes_stop'], "point mover would ignore collision")
                if split == 'train':
                    train_features.append(f)
                else:
                    test_features.append(f)
        _require(any(f['success'] for f in train_features), "teach successful docking")
        _require(any(not f['success'] and f['halo_changes_stop'] for f in train_features),
                 "teach a halo-induced moving collision")
        _require(any(all(t[feature] > max(f[feature] for f in train_features)
                             for t in test_features)
                         for feature in ('k', 'length', 'holes')),
                 "tests must extrapolate size, rail length, or defect count")
    except (ValueError, KeyError, TypeError, IndexError) as error:
        raise AssertionError(str(error)) from error


def self_test():
    """Bounded semantic regressions, symmetries, mutations and 42-seed diversity."""
    # A 3-square with one missing top-middle cell has a three-cell top bump.
    # The square alone fits the whole horizontal tube. Its bump stops at x=6.
    inp = np.zeros((9, 20), dtype=np.int8)
    inp[3:6, 2:18] = FLOOR
    inp[2:7, 2:8] = FLOOR
    inp[2:7, 15:19] = FLOOR
    inp[4, 3:18] = RAIL
    inp[4, 17] = GOAL
    inp[3:6, 2:5] = 4
    inp[3, 3] = FLOOR
    inp[6, 6] = 4
    d = _decode(inp)
    assert d['route'][_last_safe(d)] == (4, 6)
    assert d['route'][_last_safe(d, square_only=True)] == (4, 17)
    expected = inp.copy()
    expected[3:6, 2:5] = FLOOR
    expected[4, 3:5] = RAIL
    expected[6, 6] = FLOOR
    expected[3:6, 5:8] = 4
    expected[3, 6] = FLOOR
    expected[2:5, 5:8] = HALO
    expected[3, 6] = FLOOR
    assert np.array_equal(solve(inp), expected)
    for turns in range(4):
        for reflection in (False, True):
            def tf(a):
                return np.rot90(np.fliplr(a) if reflection else a, turns).copy()
            assert np.array_equal(solve(tf(inp)), tf(expected))
    # On ONE asymmetric tunnel, changing only the defect side changes the stop.
    # Both inputs have the same bottom lane, square size, rail and beacon.
    lane = inp.copy()
    lane[6, 2:19] = FLOOR
    assert _decode(lane)['route'][_last_safe(_decode(lane))] == (4, 6)
    changed = lane.copy()
    changed[3, 3], changed[5, 3] = 4, FLOOR
    dc = _decode(changed)
    assert _last_safe(dc) == len(dc['route'])-1
    assert not np.array_equal(solve(lane), solve(changed))
    # A frozen 5-square docking case: overlapping diagonal/orthogonal holes.
    # Explicit final coordinates test merged halos and exclusion of ALL holes,
    # including the hole that covers the docking beacon itself.
    cluster = np.zeros((13, 24), dtype=np.int8)
    cluster[3:10, 1:23] = FLOOR
    cluster[6, 4:20] = RAIL
    cluster[6, 19] = GOAL
    cluster[4:9, 2:7] = 9
    cluster[4, 3] = cluster[5, 4] = cluster[6, 4] = FLOOR
    cluster[3, 10] = 9
    wanted = np.zeros_like(cluster)
    wanted[3:10, 1:23] = FLOOR
    wanted[6, 4:20] = RAIL
    wanted[4:9, 17:22] = 9
    for point in ((4, 18), (5, 19), (6, 19)):
        wanted[point] = FLOOR
    for point in ((3, 17), (3, 18), (3, 19), (4, 17), (4, 19), (4, 20),
                  (5, 17), (5, 18), (5, 20), (6, 18), (6, 20),
                  (7, 18), (7, 19), (7, 20)):
        wanted[point] = HALO
    assert np.array_equal(solve(cluster), wanted)
    shapes, structures, palettes, ks, holes, turns, lengths = set(), set(), set(), set(), set(), set(), set()
    test_outcomes, test_shifts, fingerprints = set(), set(), set()
    halo_collisions = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprints.add(tuple(_signature(q['input']) for s in p for q in p[s]))
        training = [_features(_decode(q['input'])) for q in p['train']]
        for split in ('train', 'test'):
            for pair in p[split]:
                d = _decode(pair['input'])
                f = _features(d)
                shapes.add(d['a'].shape)
                normalized = d['a'].copy()
                normalized[normalized == d['body']] = 4
                structures.add(_signature(normalized))
                palettes.add(d['body'])
                ks.add(f['k']); holes.add(f['holes']); turns.add(f['turns']); lengths.add(f['length'])
                halo_collisions += f['halo_changes_stop']
                # Quarter turns preserve the nonrotating translation rule.
                assert np.array_equal(solve(np.rot90(pair['input'])), np.rot90(pair['output']))
                if split == 'test':
                    test_outcomes.add(f['success'])
                    if f['k'] > max(t['k'] for t in training): test_shifts.add('size')
                    if f['turns'] > max(t['turns'] for t in training): test_shifts.add('turns')
                    if f['holes'] > max(t['holes'] for t in training): test_shifts.add('holes')
        if seed == 0:
            q = generate_puzzle(seed)
            assert all(np.array_equal(a[key], b[key]) for split in p
                       for a, b in zip(p[split], q[split]) for key in a)
    assert len(fingerprints) == 42 and len(structures) >= 190
    assert len(shapes) >= 40 and palettes == set(BODY_COLORS)
    assert ks == {3, 5, 7} and len(holes) >= 5 and len(lengths) >= 30
    assert max(turns) >= 4 and test_outcomes == {False, True}
    assert test_shifts == {'size', 'turns', 'holes'} and halo_collisions >= 42
    # Validate every legal explicit count and reject nonintegral/bool counts.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(31+a*10+b, a, b)
            assert (len(p['train']), len(p['test'])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    for kind in ('wall', 'hole', 'halo'):
        bad = {s: [{key: value.copy() for key, value in pair.items()} for pair in p[s]] for s in p}
        pair = bad['test'][0]
        d = _decode(pair['input'])
        if kind == 'wall':
            point = tuple(np.argwhere(pair['input'] == WALL)[0])
            pair['output'][point] = FLOOR
        elif kind == 'halo':
            point = tuple(np.argwhere(pair['output'] == HALO)[0])
            pair['output'][point] = d['body']
        else:
            cy, cx = d['route'][_last_safe(d)]
            y, x = np.argwhere(d['holes'])[0]
            pair['output'][cy-d['h']+y, cx-d['h']+x] = d['body']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted " + kind + " mutation")


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': 42}, separators=(',', ':')))
