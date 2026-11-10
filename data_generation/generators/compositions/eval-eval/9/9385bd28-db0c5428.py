"""Swatch Carousel: inward-keyed, half-turned, periodically tiled opaque panels.

Only Python's standard library and NumPy are used. solve reads the visible grid;
generation preferences and training coverage never supply hidden solver state.
"""
from itertools import combinations

import numpy as np


# Fixed spatial reading order, also the front-to-back layer order.
DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
              (1, 0), (1, -1), (0, -1), (-1, -1))
CARDINALS = (0, 2, 4, 6)
CORNERS = (1, 3, 5, 7)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.flat))


def _corners(bounds, diagonal):
    r, c, b, e = bounds
    if diagonal == 0:
        return {(r, c), (r, c+1), (r+1, c),
                (b, e), (b, e-1), (b-1, e)}
    return {(r, e), (r, e-1), (r+1, e),
            (b, c), (b, c+1), (b-1, c)}


def _inside(r, c, box):
    return box[0] <= r <= box[2] and box[1] <= c <= box[3]


def _overlap(a, b):
    return max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3])


def _recognize(points):
    _require(len(points) == 6, "a panel needs exactly two L triplets")
    r = min(p[0] for p in points)
    c = min(p[1] for p in points)
    b = max(p[0] for p in points)
    e = max(p[1] for p in points)
    h, w = b-r+1, e-c+1
    _require(min(h, w) >= 2 and (min(h, w) > 2 or max(h, w) >= 4),
             "one-cell panels and solid 2x3/2x2 blocks are excluded")
    # Local recognition is expressed separately from the marker constructor.
    local = {(y-r, x-c) for y, x in points}
    patterns = ({(0, 0), (0, 1), (1, 0), (h-1, w-1), (h-1, w-2), (h-2, w-1)},
                {(0, w-1), (0, w-2), (1, w-1), (h-1, 0), (h-1, 1), (h-2, 0)})
    _require(local in patterns, "malformed opposite corners")
    return (r, c, b, e), patterns.index(local)


def _analyze(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(np.count_nonzero(counts == counts[bg]) == 1, "ambiguous background")
    rest = counts.copy()
    rest[bg] = 0
    carrier = int(rest.argmax())
    _require(rest[carrier] >= 20 and np.count_nonzero(rest == rest[carrier]) == 1,
             "ambiguous common carrier")
    positions = np.argwhere(a == carrier)
    r, c = map(int, positions.min(axis=0))
    b, e = map(int, positions.max(axis=0))
    _require((b-r, e-c) == (8, 8), "carrier must bound a 9x9 carousel")
    _require(bool(np.all(a[r+3:r+6, c+3:c+6] == bg)), "central index must be empty")
    tiles, keys = {}, {}
    for i, (dy, dx) in enumerate(DIRECTIONS):
        t = a[r+3*(dy+1):r+3*(dy+2), c+3*(dx+1):c+3*(dx+2)].copy()
        if np.all(t == bg):
            continue
        _require(np.count_nonzero(t == carrier) >= 5, "swatch carrier is unclear")
        k = int(t[1-dy, 1-dx])
        _require(k not in (bg, carrier) and k not in keys.values(), "inward keys must be distinct")
        if i in CARDINALS:
            _require(t[1+dy, 1+dx] == carrier, "outward cardinal anchor must be carrier")
        tiles[i], keys[i] = t, k
    _require(all(i in tiles for i in CARDINALS), "all cardinal swatches must be present")
    scene = a.copy()
    scene[r:r+9, c:c+9] = bg
    lookup = {k: i for i, k in keys.items()}
    specs = []
    moat = (r-1, c-1, r+9, c+9)
    for k in map(int, np.unique(scene)):
        if k == bg:
            continue
        _require(k in lookup, "scene color has no inward key")
        markers = {tuple(map(int, p)) for p in np.argwhere(scene == k)}
        bounds, diagonal = _recognize(markers)
        _require(not _overlap(bounds, moat), "panel intersects carousel moat")
        specs.append(dict(direction=lookup[k], key=k, bounds=bounds,
                          markers=markers, diagonal=diagonal))
    specs.sort(key=lambda o: o['direction'])
    _require(2 <= len(specs) <= 8, "expected 2..8 panels")
    return dict(background=bg, carrier=carrier, origin=(r, c), tiles=tiles,
                views={i: _views(t) for i, t in tiles.items()}, keys=keys, specs=specs)


def _views(tile):
    # Half-turn is index 2.
    return tuple(np.rot90(tile, k) for k in range(4)) + tuple(
        np.rot90(tile[:, ::-1], k) for k in range(4))


def _render(a, info, *, operation=2, reverse=False, transparent=False,
            flat=False, stamp=False, retain=True, global_phase=False,
            protect_all_markers=False):
    """Layer renderer; keyword alternatives are used only for shortcut audits."""
    out = a.copy()
    specs = list(reversed(info['specs'])) if reverse else info['specs']
    bg = info['background']
    for obj in reversed(specs):
        r, c, b, e = obj['bounds']
        if stamp:
            b, e = min(b, r+2), min(e, c+2)
        pattern = info['views'][obj['direction']][operation]
        yy, xx = np.arange(b-r+1), np.arange(e-c+1)
        if global_phase:
            yy = yy + r-info['origin'][0]
            xx = xx + c-info['origin'][1]
        paint = pattern[yy[:, None] % 3, xx[None, :] % 3]
        if flat:
            paint = np.full(paint.shape, obj['key'], dtype=a.dtype)
        target = out[r:b+1, c:e+1]
        if transparent:
            target[paint != bg] = paint[paint != bg]
        else:
            target[:] = paint
        if retain:
            for y, x in obj['markers']:
                if y <= b and x <= e:
                    out[y, x] = obj['key']
    if protect_all_markers:
        for obj in specs:
            for y, x in obj['markers']:
                out[y, x] = obj['key']
    r, c = info['origin']
    out[r+4, c+4] = info['carrier']
    for i, (dy, dx) in enumerate(DIRECTIONS):
        out[r+4+dy, c+4+dx] = info['keys'].get(i, bg)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from visible keys, patterns, bounds and order."""
    a = _grid(input_grid)
    return _render(a, _analyze(a))


def _value(info, obj, y, x, operation=2):
    if (y, x) in obj['markers']:
        return obj['key']
    tile = info['views'][obj['direction']][operation]
    return int(tile[(y-obj['bounds'][0]) % 3, (x-obj['bounds'][1]) % 3])


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
    a, out = _grid(pair['input']), _grid(pair['output'])
    _require(a.shape == out.shape, "output shape changed")
    info = _analyze(a)
    r, c = info['origin']
    # Scalar, first-covering-layer check independent of reverse-order slice painting.
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            if r+3 <= y <= r+5 and c+3 <= x <= c+5:
                d = (y-r-4, x-c-4)
                expected = info['carrier'] if d == (0, 0) else info['keys'].get(DIRECTIONS.index(d), info['background'])
            else:
                covering = next((o for o in info['specs'] if _inside(y, x, o['bounds'])), None)
                expected = int(a[y, x]) if covering is None else _value(info, covering, y, x)
            _require(int(out[y, x]) == expected, "incorrect pattern, priority, corner or preserved cell")
    _require(not np.array_equal(a, out), "identity example")
    return info


def _coverage(train):
    """Visible witnesses for sampling, D4 choice, and angular precedence."""
    sample_candidates = [set(range(9)) for _ in DIRECTIONS]
    operation_candidates = [set(range(8)) for _ in DIRECTIONS]
    edges, seen = set(), set()
    for pair in train:
        a, out = np.asarray(pair['input']), np.asarray(pair['output'])
        info = _analyze(a)
        r, c = info['origin']
        for i, t in info['tiles'].items():
            seen.add(i)
            dy, dx = DIRECTIONS[i]
            observed = int(out[r+4+dy, c+4+dx])
            sample_candidates[i].intersection_update(j for j in range(9) if t.flat[j] == observed)
        exposed = {o['direction']: [] for o in info['specs']}
        for y in range(a.shape[0]):
            for x in range(a.shape[1]):
                stack = [o for o in info['specs'] if _inside(y, x, o['bounds'])]
                if not stack:
                    continue
                winner = stack[0]
                if (y, x) not in winner['markers']:
                    exposed[winner['direction']].append((y, x, int(out[y, x])))
                if len(stack) == 2:
                    u, v = stack
                    vu, vv = _value(info, u, y, x), _value(info, v, y, x)
                    if vu != vv and out[y, x] == vu:
                        edges.add((u['direction'], v['direction']))
        for obj in info['specs']:
            i = obj['direction']
            operation_candidates[i].intersection_update(k for k in range(8) if all(
                _value(info, obj, y, x, k) == val for y, x, val in exposed[i]))
    closure = set(edges)
    for k in range(8):
        for i in range(8):
            for j in range(8):
                if (i, k) in closure and (k, j) in closure:
                    closure.add((i, j))
    _require(all(sample_candidates[i] and operation_candidates[i] for i in seen), "contradictory demonstrations")
    return sample_candidates, operation_candidates, closure, seen


def _check_teaching(train, tests):
    samples, operations, precedence, seen = _coverage(train)
    for pair in tests:
        info = _analyze(pair['input'])
        for i, tile in info['tiles'].items():
            _require(i in seen, "query swatch direction was not demonstrated")
            _require({int(tile.flat[j]) for j in samples[i]} == {info['keys'][i]}, "query sampling is ambiguous")
        for obj in info['specs']:
            for y in range(obj['bounds'][0], obj['bounds'][2]+1):
                for x in range(obj['bounds'][1], obj['bounds'][3]+1):
                    winner = next(o for o in info['specs'] if _inside(y, x, o['bounds']))
                    if winner is obj and (y, x) not in obj['markers']:
                        values = {_value(info, obj, y, x, k) for k in operations[obj['direction']]}
                        _require(len(values) == 1, "query needs an untaught tile orientation")
        for u, v in combinations(info['specs'], 2):
            if _overlap(u['bounds'], v['bounds']):
                _require((u['direction'], v['direction']) in precedence, "query overlap priority was not taught")


def _features(pair):
    a, out = np.asarray(pair['input']), np.asarray(pair['output'])
    info = _analyze(a)
    shortcuts = {name: not np.array_equal(out, _render(a, info, **kw)) for name, kw in (
        ('priority', {'reverse': True}), ('rotation', {'operation': 0}),
        ('opaque', {'transparent': True}), ('pattern', {'flat': True}),
        ('extent', {'stamp': True}), ('corners', {'retain': False}),
        ('occlusion', {'protect_all_markers': True}),
        ('phase', {'global_phase': True}))}
    sizes = [(o['bounds'][2]-o['bounds'][0]+1, o['bounds'][3]-o['bounds'][1]+1) for o in info['specs']]
    triple = any(sum(_inside(y, x, o['bounds']) for o in info['specs']) >= 3
                 for y in range(a.shape[0]) for x in range(a.shape[1]))
    repeat = any(np.count_nonzero(t == info['keys'][i]) > 1 for i, t in info['tiles'].items())
    return dict(shortcuts, panels=len(info['specs']), swatches=len(info['tiles']),
                background=info['background'], carrier=info['carrier'], shape=a.shape,
                origin=info['origin'], sizes=tuple(sizes),
                diagonals=tuple(sorted({o['diagonal'] for o in info['specs']})),
                thin=any(min(s) == 2 for s in sizes), triple=triple, repeated=repeat,
                partial=any(h % 3 or w % 3 for h, w in sizes),
                edge=any(0 in o['bounds'][:2] or o['bounds'][2] == a.shape[0]-1 or o['bounds'][3] == a.shape[1]-1 for o in info['specs']))


def _tiles(rng, roster, palette, teaching=False, repeated=False):
    bg, carrier = palette[:2]
    keys = {i: palette[2+j] for j, i in enumerate(roster)}
    tiles = {}
    for i in roster:
        dy, dx = DIRECTIONS[i]
        inward = (1-dy, 1-dx)
        reserve = {(1+dy, 1+dx)} if i in CARDINALS else set()
        available = [p for p in np.ndindex(3, 3) if p != inward and p not in reserve]
        for _ in range(80):
            t = np.full((3, 3), carrier, dtype=np.int8)
            t[inward] = keys[i]
            positions = list(available)
            rng.shuffle(positions)
            if teaching:
                t[positions[0]] = bg
                choices = [v for v in palette if v not in (carrier, keys[i])]
                for p in positions[1:1+int(rng.integers(0, 3))]:
                    t[p] = int(rng.choice(choices))
                if len({v.tobytes() for v in _views(t)}) != 8:
                    continue
            else:
                count = int(rng.choice(4, p=[.15, .25, .35, .25]))
                motif = int(rng.integers(3))
                if motif == 1:
                    axis, line = int(rng.integers(2)), int(rng.integers(3))
                    positions.sort(key=lambda p: p[axis] != line)
                elif motif == 2:
                    positions.sort(key=lambda p: abs(p[0]-inward[0])+abs(p[1]-inward[1]))
                choices = [v for v in palette if v != carrier]
                for p in positions[:count]:
                    t[p] = int(rng.choice(choices))
                if repeated:
                    # Position remains the selector even when its color repeats.
                    t[positions[0]] = keys[i]
            tiles[i] = t
            break
        else:
            raise AssertionError("failed to make an asymmetric teaching swatch")
    return tiles, keys


def _build(rng, shape, origin, bounds, active, roster, palette, teaching=False,
           repeated=False, same_diagonal=False):
    tiles, keys = _tiles(rng, roster, palette, teaching, repeated)
    a = np.full(shape, palette[0], dtype=np.int8)
    r, c = origin
    for i, t in tiles.items():
        dy, dx = DIRECTIONS[i]
        a[r+3*(dy+1):r+3*(dy+2), c+3*(dx+1):c+3*(dx+2)] = t
    used = set()
    diagonal = int(rng.integers(2))
    for box, i in zip(bounds, active):
        choices = [diagonal] if same_diagonal else list(map(int, rng.permutation(2)))
        for d in choices:
            points = _corners(box, d)
            if not points & used:
                break
        else:
            return None
        used.update(points)
        for y, x in points:
            a[y, x] = keys[i]
    try:
        info = _analyze(a)
    except AssertionError:
        return None
    return dict(input=a, output=_render(a, info))


def _layout_transform(shape, origin, boxes, operation):
    h, w = shape
    mirror, turns = operation >= 4, operation % 4
    def point(y, x):
        hh, ww = h, w
        if mirror:
            x = ww-1-x
        for _ in range(turns):
            y, x, hh, ww = ww-1-x, y, ww, hh
        return y, x
    def box(b):
        points = [point(y, x) for y in (b[0], b[2]) for x in (b[1], b[3])]
        return min(y for y, x in points), min(x for y, x in points), max(y for y, x in points), max(x for y, x in points)
    hub = box((origin[0], origin[1], origin[0]+8, origin[1]+8))
    return ((h, w) if turns % 2 == 0 else (w, h)), hub[:2], [box(b) for b in boxes]


def _palette(rng, fixed_bg):
    values = list(map(int, rng.permutation(10)))
    if fixed_bg is not None:
        j = values.index(fixed_bg)
        values[0], values[j] = values[j], values[0]
    return values


def _teaching_pair(rng, roster, fixed_bg):
    n = len(roster)
    for _ in range(160):
        pw = int(rng.integers(7, 11))
        footprint = (3*n+1, 10+n-1+pw)
        h = footprint[0]+int(rng.integers(min(5, 30-footprint[0])+1))
        w = footprint[1]+int(rng.integers(min(5, 30-footprint[1])+1))
        r = int(rng.integers(h-footprint[0]+1))
        c = int(rng.integers(w-footprint[1]+1))
        boxes = [(r+3*j, c+10+j, r+3*j+3, c+10+j+pw-1) for j in range(n)]
        shape, origin, boxes = _layout_transform((h, w), (r, c), boxes, int(rng.integers(8)))
        active = roster if rng.random() < .5 else list(reversed(roster))
        pair = _build(rng, shape, origin, boxes, active, roster, _palette(rng, fixed_bg),
                      teaching=True, same_diagonal=True)
        if pair is None:
            continue
        f = _features(pair)
        s, ops, edges, seen = _coverage([pair])
        if all(ops[i] == {2} and len(s[i]) == 1 for i in roster) and all(
                (i, j) in edges for i, j in combinations(roster, 2)) and all(
                f[k] for k in ('priority', 'rotation', 'opaque', 'pattern', 'extent', 'corners', 'occlusion')):
            return pair
    raise AssertionError("teaching construction exhausted")


def _random_box(rng, domain, thin=False, long=False):
    r, c, b, e = domain
    h, w = b-r+1, e-c+1
    if min(h, w) < 2 or max(h, w) < 4:
        return None
    mode = int(rng.integers(5))
    if thin:
        mode = int(rng.choice([0, 1]))
    if mode == 0 and w >= 4:
        hh, ww = 2, int(rng.integers(4, w+1))
    elif mode == 1 and h >= 4:
        hh, ww = int(rng.integers(4, h+1)), 2
    elif mode == 2 and min(h, w) >= 3:
        hh = ww = int(rng.integers(3, min(h, w, 10)+1))
    else:
        hh = int(rng.integers(2, h+1 if long else min(h, 11)+1))
        ww = int(rng.integers(2, w+1 if long else min(w, 11)+1))
    if min(hh, ww) == 2 and max(hh, ww) < 4:
        return None
    y = int(rng.integers(r, b-hh+2))
    x = int(rng.integers(c, e-ww+2))
    return y, x, y+hh-1, x+ww-1


def _free_pair(rng, roster, fixed_bg, shift, test=False, demonstration=False,
               allow_thin=True, allow_triple=True):
    for _ in range(500):
        lo, hi = (24, 31) if test and shift == 'extent' else ((17, 25) if shift == 'extent' else (17, 31))
        h, w = map(int, rng.integers(lo, hi, size=2))
        r = int(rng.choice([0, h-9, int(rng.integers(h-8))]))
        c = int(rng.choice([0, w-9, int(rng.integers(w-8))]))
        domains = [(0, 0, r-2, w-1), (r+10, 0, h-1, w-1),
                   (0, 0, h-1, c-2), (0, c+10, h-1, w-1)]
        domains = [d for d in domains if min(d[2]-d[0]+1, d[3]-d[1]+1) >= 2 and max(d[2]-d[0]+1, d[3]-d[1]+1) >= 4]
        if not domains:
            continue
        present = sorted(set(CARDINALS) | {i for i in roster if i in CORNERS and rng.random() < .8})
        count = (2 if test and not allow_triple else
                 int(rng.integers(3 if demonstration or shift == 'overlap' else 2, len(present)+1)))
        active = list(map(int, rng.choice(present, count, replace=False)))
        boxes, used = [], set()
        for j in range(count):
            for _ in range(100):
                domain = domains[int(rng.integers(len(domains)))]
                box = _random_box(rng, domain, thin=(j == 0 and shift == 'thin'), long=test and shift == 'extent')
                if box is None or box in boxes:
                    continue
                if not allow_thin and min(box[2]-box[0]+1, box[3]-box[1]+1) == 2:
                    continue
                choices = [_corners(box, d) for d in (0, 1)]
                if any(not points & used for points in choices):
                    points = next(p for p in choices if not p & used)
                    boxes.append(box)
                    used.update(points)
                    break
            else:
                break
        if len(boxes) != count:
            continue
        pair = _build(rng, (h, w), (r, c), boxes, active, present,
                      _palette(rng, fixed_bg), repeated=demonstration)
        if pair is None:
            continue
        f = _features(pair)
        if demonstration and (not f['phase'] or not f['partial'] or len(f['diagonals']) < 2):
            continue
        if shift == 'overlap' and (demonstration or test) and not f['triple']:
            continue
        if shift == 'thin' and (demonstration or test) and not f['thin']:
            continue
        if test and not f['partial']:
            continue
        return pair
    raise AssertionError("free layout construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, m = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if n <= m:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    extra = int(rng.choice(5, p=[.15, .25, .30, .20, .10]))
    roster = sorted(set(CARDINALS) | set(map(int, rng.choice(CORNERS, extra, replace=False))))
    shift = str(rng.choice(['extent', 'overlap', 'thin']))
    fixed_bg = int(rng.integers(10)) if rng.random() < .25 else None
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', n), ('test', m)):
        taught = [_features(p) for p in puzzle['train']] if split == 'test' else []
        for index in range(count):
            for _ in range(30):
                pair = (_teaching_pair(rng, roster, fixed_bg) if split == 'train' and index == 0 else
                        _free_pair(rng, roster, fixed_bg, shift, test=split == 'test',
                                   demonstration=split == 'train' and index == 1,
                                   allow_thin=split == 'train' or any(f['thin'] for f in taught),
                                   allow_triple=split == 'train' or any(f['triple'] for f in taught)))
                if split == 'test':
                    f = _features(pair)
                    if any(f[k] and not any(g[k] for g in taught) for k in ('thin', 'triple')):
                        continue
                signature = _key(pair['input'])
                if signature not in seen:
                    break
            else:
                raise AssertionError("distinct-input retries exhausted")
            seen.add(signature)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "invalid counts")
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check_pair(pair)
            key = _key(pair['input'])
            _require(key not in seen, "duplicate input")
            seen.add(key)
    features = [_features(p) for p in puzzle['train']]
    for name in ('priority', 'rotation', 'opaque', 'pattern', 'extent', 'corners', 'occlusion', 'phase'):
        _require(any(f[name] for f in features), f"no training witness for {name}")
    _require(any(f['repeated'] for f in features), "position-based repeated-key sampling is not demonstrated")
    _require(any(len(f['diagonals']) == 2 for f in features), "both corner diagonals need a demonstration")
    for pair in puzzle['test']:
        f = _features(pair)
        for name in ('thin', 'triple'):
            if f[name]:
                _require(any(g[name] for g in features), f"untaught {name} query")
    _check_teaching(puzzle['train'], puzzle['test'])


def self_test():
    """Bounded semantic audits and measured structural variation on 42 seeds."""
    records, signatures, structures = [], set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(tuple(_key(v) for s in ('train', 'test') for pair in p[s] for v in pair.values()))
        for split in ('train', 'test'):
            for pair in p[split]:
                a = pair['input']
                before = a.copy()
                _require(np.array_equal(solve(a), pair['output']), "solve disagrees")
                _require(np.array_equal(before, a), "solve mutated its input")
                f = _features(pair)
                f['split'] = split
                records.append(f)
                roles = {}
                structures.add((a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)))
    _require(len(signatures) == 42 and len(structures) >= 150, "seed or structural diversity collapsed")
    _require({f['background'] for f in records} == set(range(10)), "background roles omitted")
    _require({f['carrier'] for f in records} == set(range(10)), "carrier roles omitted")
    _require({f['swatches'] for f in records} == set(range(4, 9)), "swatch counts omitted")
    _require(len({f['shape'] for f in records}) >= 70, "canvas variation too narrow")
    _require(len({f['sizes'] for f in records}) >= 140, "panel geometry variation too narrow")
    for name in ('thin', 'triple', 'repeated', 'partial', 'edge', 'opaque'):
        _require(sum(bool(f[name]) for f in records) >= 12, f"variation missing: {name}")
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(1000+10*n+m, n, m)
            _require(len(p['train']) == n and len(p['test']) == m, "counts ignored")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(314159, 2, 1), generate_puzzle(314159, 2, 1)
    _require(all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x), "nondeterminism")
    info = _analyze(a['test'][0]['input'])
    r, c = info['origin']
    panel_point = info['specs'][0]['bounds'][:2]
    unassigned = next((y, x) for y, x in np.ndindex(a['test'][0]['input'].shape)
                      if not _inside(y, x, (r, c, r+8, c+8)) and
                      not any(_inside(y, x, o['bounds']) for o in info['specs']))
    for point in ((r+4, c+4), (r, c), panel_point, unassigned):
        broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
        out = broken['test'][0]['output']
        out[point] = (int(out[point])+1) % 10
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted a corrupted output cell")
    return dict(seeds=42, pairs=len(records), structures=len(structures),
                shapes=len({f['shape'] for f in records}),
                panel_counts=sorted({f['panels'] for f in records}),
                swatch_counts=sorted({f['swatches'] for f in records}))
