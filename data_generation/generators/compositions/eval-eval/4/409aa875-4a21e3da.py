"""Five-step impulses move guide-defined payload partitions as rigid cohorts.

Only Python's standard library and NumPy are used.  All solving conventions are
fixed or visible in the input; construction metadata is never used by solve.
"""
from collections import Counter
import json

import numpy as np


DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
              (1, 0), (1, -1), (0, -1), (-1, -1))
DISTANCE = 5


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape),
             "grids must be two-dimensional with dimensions 1..30")
    _require(np.issubdtype(g.dtype, np.integer) and
             bool(np.all((g >= 0) & (g <= 9))), "colors must be integers 0..9")
    return g.astype(np.int8, copy=False)


def _components(cells, diagonal=False):
    remaining, result = set(cells), []
    offsets = DIRECTIONS if diagonal else ((-1, 0), (0, 1), (1, 0), (0, -1))
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in offsets:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _arrow(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return (tip, direction, frozenset((r + a, c + b) for a, b in offsets))


def _decode_arrow(cells):
    matches = [_arrow(p, d) for p in sorted(cells) for d in DIRECTIONS
               if _arrow(p, d)[2] == cells]
    _require(len(matches) == 1, "each arrow must have unique V or L geometry")
    return matches[0]


def _endpoint(arrow):
    (r, c), (dr, dc), _ = arrow
    return (r + DISTANCE * dr, c + DISTANCE * dc)


def _partition(cells, row_axis, col_axis):
    fixed, groups = set(), {}
    for r, c in sorted(cells):
        if r == row_axis or c == col_axis:
            fixed.add((r, c))
        else:
            key = ((-1 if r < row_axis else 1) if row_axis is not None else 0,
                   (-1 if c < col_axis else 1) if col_axis is not None else 0)
            groups.setdefault(key, set()).add((r, c))
    return fixed, groups


def _read(input_grid):
    g = _grid(input_grid)
    h, w = g.shape
    _require(h >= 7 and w >= 7, "this family needs room for markers and arrows")
    corners = [int(g[p]) for p in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))]
    _require(len(set(corners)) == 1, "input corners must be background")
    bg = corners[0]
    border = {(r, c) for r in range(h) for c in range(w)
              if r in (0, h-1) or c in (0, w-1)}
    marks = sorted(p for p in border if g[p] != bg)
    _require(len(marks) in (1, 2), "one or two border markers required")
    marker_colors = {int(g[p]) for p in marks}
    _require(len(marker_colors) == 1, "border markers must share a color")
    marker = marker_colors.pop()
    _require(int(np.count_nonzero(g == marker)) == len(marks),
             "marker color occurs only on the border in the input")
    markers, ar, ac = {}, None, None
    for r, c in marks:
        side = 'top' if r == 0 else 'bottom' if r == h-1 else 'left' if c == 0 else 'right'
        _require(side not in markers, "at most one marker per side")
        markers[side] = (r, c)
        if side in ('top', 'bottom'):
            ac = c
        else:
            ar = r
    _require(len(marks) == 1 or (ar is not None and ac is not None),
             "two markers must occupy adjacent sides")
    colors = sorted(set(map(int, np.unique(g))) - {bg, marker})
    _require(len(colors) == 2, "exactly four visible roles required")
    decoded = []
    for color in colors:
        cells = set(map(tuple, np.argwhere(g == color).tolist()))
        try:
            pieces = _components(cells, diagonal=True)
            _require(all(len(p) == 3 for p in pieces), "not the arrow color")
            arrows = [_decode_arrow(p) for p in pieces]
            decoded.append((color, arrows))
        except ValueError:
            pass
    _require(len(decoded) == 1, "unique arrow and payload role assignment required")
    arrow_color, arrows = decoded[0]
    payload = next(c for c in colors if c != arrow_color)
    cells = set(map(tuple, np.argwhere(g == payload).tolist()))
    _require(len(cells) >= 7 and len(_components(cells)) == 1,
             "the original payload must be one connected object of at least seven cells")
    _require(not (cells & border), "payload must be interior")
    fixed, groups = _partition(cells, ar, ac)
    _require(len(groups) == (4 if len(marks) == 2 else 2), "every cut region must be nonempty")
    for side, (r, c) in markers.items():
        aligned = [p for p in cells if p[1] == c] if side in ('top', 'bottom') else [p for p in cells if p[0] == r]
        _require(bool(aligned), "every guide must intersect the payload")
    for arrow in arrows:
        _require(not (arrow[2] & border), "arrowheads must be interior")
        p = _endpoint(arrow)
        _require(0 <= p[0] < h and 0 <= p[1] < w, "shot endpoint outside grid")
        _require(int(g[p]) in (bg, payload) and p[0] != ar and p[1] != ac,
                 "endpoints must be background or payload away from the guide axes")
    return {'grid': g, 'background': bg, 'marker': marker,
            'arrow_color': arrow_color, 'payload': payload, 'markers': markers,
            'row_axis': ar, 'col_axis': ac, 'arrows': arrows,
            'cells': cells, 'fixed': fixed, 'groups': groups}


def _guide_paths(info):
    cells, paths = info['cells'], []
    h, w = info['grid'].shape
    for side, (r, c) in info['markers'].items():
        if side in ('top', 'bottom'):
            aligned = [a for a, b in cells if b == c]
            rows = range(max(aligned)+1) if side == 'top' else range(min(aligned), h)
            paths.append([(a, c) for a in rows])
        else:
            aligned = [b for a, b in cells if a == r]
            cols = range(max(aligned)+1) if side == 'left' else range(min(aligned), w)
            paths.append([(r, b) for b in cols])
    return paths


def _transform(info):
    g, bg = info['grid'], info['background']
    out = g.copy()
    for cells in info['groups'].values():
        for p in cells:
            out[p] = bg
    for path in _guide_paths(info):
        for p in path:
            if g[p] == bg:
                out[p] = info['marker']
    endpoints = [_endpoint(a) for a in info['arrows']]
    counts = Counter(endpoints)
    # Misses are stationary marks in the original coordinate frame.
    for p, n in counts.items():
        if g[p] == bg:
            out[p] = info['marker'] if n > 1 else info['arrow_color']
    records = []
    for key, cells in sorted(info['groups'].items()):
        shooters = [a for a, p in zip(info['arrows'], endpoints) if p in cells]
        vr = sum(a[1][0] for a in shooters)
        vc = sum(a[1][1] for a in shooters)
        rows, cols = zip(*cells)
        dr = -min(rows) if vr < 0 else g.shape[0]-1-max(rows) if vr > 0 else 0
        dc = -min(cols) if vc < 0 else g.shape[1]-1-max(cols) if vc > 0 else 0
        collision = any(counts[_endpoint(a)] > 1 for a in shooters)
        color = (info['marker'] if collision else info['arrow_color']) if shooters else info['payload']
        targets = {(r+dr, c+dc) for r, c in cells}
        _require(all(0 <= r < g.shape[0] and 0 <= c < g.shape[1] for r, c in targets),
                 "translated partition outside grid")
        _require(all(out[p] == bg for p in targets),
                 "translated regions must not overlap guides, arrows, marks, or each other")
        for p in targets:
            out[p] = color
        records.append({'key': key, 'cells': cells, 'targets': targets,
                        'impulse': (vr, vc), 'translation': (dr, dc),
                        'color': color, 'shooters': shooters, 'collision': collision})
    return out, records, counts


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read all roles and geometry from this input; return a fresh output grid."""
    return _transform(_read(input_grid))[0]


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _shape(rng, double, witness):
    """Connected paths with branches; cuts, rather than components, define groups."""
    h, w = map(int, rng.integers(7 if witness else 5, 14, size=2))
    ac = int(rng.integers(2, w-2))
    ar = int(rng.integers(2, h-2)) if double else None
    cells = set()
    if witness or rng.random() < .45:
        # A ladder uses a detour beside the vertical axis.  The axis intersects
        # separated rungs, and cutting it disconnects the right-hand branches.
        # The off-axis backbone extends below the last rung: a guide stops at
        # its own farthest payload hit, short of the object's overall box.
        rung_rows = sorted({0, h-2, int(rng.integers(2, h-2))})
        cells |= {(r, ac-1) for r in range(h)}
        for r in rung_rows:
            left = int(rng.integers(0, ac))
            right = int(rng.integers(ac+2, w))
            cells |= {(r, c) for c in range(left, right+1)}
        if double:
            cells |= {(ar, c) for c in range(w)}
        # Independent branch lengths and shapes keep the ladder from being a
        # repeated fixed silhouette.  Protect a cut gap for essential witnesses.
        forbidden = {(r, ac) for r in range(h) if r not in rung_rows}
        if double:
            forbidden.discard((ar, ac))
        for _ in range(int(rng.integers(0, h*w//3+1))):
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                               if 0 <= r+dr < h and 0 <= c+dc < w} - cells - forbidden)
            if frontier:
                cells.add(_pick(rng, frontier))
    else:
        # Several random Manhattan paths create sparse winding and dense forms.
        cells = {(ar if double else h//2, ac)}
        waypoints = [(0, 0), (h-1, w-1), (0, w-1), (h-1, 0)]
        rng.shuffle(waypoints)
        waypoints += [tuple(map(int, rng.integers((0, 0), (h, w)))) for _ in range(3)]
        for tr, tc in waypoints:
            r, c = _pick(rng, sorted(cells))
            while (r, c) != (tr, tc):
                if r != tr and (c == tc or rng.random() < .5):
                    r += 1 if tr > r else -1
                else:
                    c += 1 if tc > c else -1
                cells.add((r, c))
    if rng.random() < .5:
        cells = {(r, w-1-c) for r, c in cells}
        ac = w-1-ac
    fixed, groups = _partition(cells, ar, ac)
    if len(groups) != (4 if double else 2):
        return None
    if not any(c == ac for r, c in cells) or (double and not any(r == ar for r, c in cells)):
        return None
    if witness and not any(len(_components(v)) > 1 and len(v) >= 2 for v in groups.values()):
        return None
    return cells, ar, ac, h, w


def _compatible(arrow, selected):
    occupied = {p for a in selected for p in a[2]}
    return not any((r+dr, c+dc) in occupied for r, c in arrow[2]
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _valid_arrow(arrow, payload, shape):
    h, w = shape
    return (all(1 <= r < h-1 and 1 <= c < w-1 for r, c in arrow[2])
            and not (arrow[2] & payload))


def _profile(key, kind):
    """Directions and endpoint-equality blocks, independent of payload layout."""
    vr, vc = key
    vr = vr or -1
    vc = vc or 1
    diagonal = (vr, vc)
    if kind == 'solo':
        return [diagonal], [0]
    if kind == 'edge':
        return ([(vr, -1), (vr, 1)] if key[0] else [(-1, vc), (1, vc)]), [0, 1]
    if kind == 'balanced':
        return [diagonal, (-vr, -vc)], [0, 1]
    if kind == 'collision':
        return [(vr, 0), (0, vc)], [0, 0]
    if kind == 'mixed':
        # The collision contributes two arrows, not one normalized endpoint
        # vote.  Its diagonal/sideward pair plus an opposite diagonal sums to
        # a cardinal vector; normalizing per endpoint would cancel it entirely.
        return [(0, vc), diagonal, (-vr, -vc)], [0, 0, 1]
    if kind == 'skew':
        # Both components cancel once; the third arrow determines their signs.
        return [(vr, -vc), (-vr, vc), diagonal], [0, 1, 2]
    raise ValueError('unknown sampling profile')


def _place_hits(rng, group, key, kind, payload, shape, selected):
    directions, blocks = _profile(key, kind)
    by_block = {}
    for d, b in zip(directions, blocks):
        by_block.setdefault(b, []).append(d)
    options = {}
    for b, dirs in by_block.items():
        opts = []
        for p in sorted(group):
            arrows = [_arrow((p[0]-5*d[0], p[1]-5*d[1]), d) for d in dirs]
            if all(_valid_arrow(a, payload, shape) for a in arrows):
                if all(_compatible(a, arrows[:i]) for i, a in enumerate(arrows)):
                    opts.append((p, arrows))
        if not opts:
            return None
        options[b] = opts
    # Distinct equality blocks explicitly mean distinct endpoints, not simply
    # different shooters.  This is what distinguishes collision from multi-hit.
    order = sorted(options, key=lambda b: len(options[b]))
    for _ in range(90):
        added, used = [], set()
        for b in order:
            possibilities = [(p, arr) for p, arr in options[b] if p not in used
                             and all(_compatible(a, selected+added) for a in arr)]
            if not possibilities:
                break
            p, arr = _pick(rng, possibilities)
            used.add(p)
            added.extend(arr)
        else:
            return added
    return None


def _add_misses(rng, selected, payload, axes, shape, count):
    h, w = shape
    occupied = {p for a in selected for p in a[2]}
    for _ in range(count):
        for _ in range(350):
            d = _pick(rng, DIRECTIONS)
            p = (int(rng.integers(h)), int(rng.integers(w)))
            if p in payload or p in occupied or p[0] == axes[0] or p[1] == axes[1]:
                continue
            a = _arrow((p[0]-5*d[0], p[1]-5*d[1]), d)
            if _valid_arrow(a, payload, shape) and _compatible(a, selected):
                if any(_endpoint(old) in a[2] for old in selected):
                    continue
                selected.append(a)
                occupied.update(a[2])
                break
        else:
            return False
    return True


def _draw(rng, mode, roles, rotation, reflect):
    double = mode != 'distinct' and (mode == 'collision' or rng.random() < .58)
    witness = mode == 'distinct'
    for _ in range(240):
        shaped = _shape(rng, double, witness)
        if shaped is None:
            continue
        cells, ar, ac, ph, pw = shaped
        # Teaching margins permit opposing shooters. Other examples can be
        # smaller or off-center when their actual shot geometry fits.
        essential = mode in ('distinct', 'collision')
        margin = 6 if essential else 2
        padding = 12 if essential else 5
        h = int(rng.integers(max(13, ph+padding), 31))
        w = int(rng.integers(max(13, pw+padding), 31))
        r0 = int(rng.integers(margin, h-ph-(4 if essential else 1)))
        c0 = int(rng.integers(margin, w-pw-(4 if essential else 1)))
        cells = {(r+r0, c+c0) for r, c in cells}
        ar = ar+r0 if ar is not None else None
        ac += c0
        fixed, groups = _partition(cells, ar, ac)
        keys = sorted(groups)
        plans = []
        if mode == 'distinct':
            fragment_keys = [k for k in keys if len(_components(groups[k])) > 1]
            k = _pick(rng, fragment_keys)
            plans = [(k, 'edge'), (next(q for q in keys if q != k), 'balanced')]
        elif mode == 'collision':
            k = _pick(rng, [q for q in keys if len(groups[q]) >= 2])
            plans = [(k, 'mixed'), ((-k[0], -k[1]), 'solo')]
        elif mode == 'combine':
            rng.shuffle(keys)
            plans = [(keys[0], 'collision'), (keys[1], 'balanced')]
            if double and rng.random() < .6:
                plans.append((keys[2], _pick(rng, ('solo', 'edge', 'skew'))))
        else:
            rng.shuffle(keys)
            active = int(rng.integers(1, len(keys)+1))
            for k in keys[:active]:
                kind = str(rng.choice(('solo', 'edge', 'balanced', 'collision', 'mixed', 'skew'),
                                      p=(.20, .20, .15, .15, .15, .15)))
                plans.append((k, kind))
        selected = []
        for key, kind in plans:
            added = _place_hits(rng, groups[key], key, kind, cells, (h, w), selected)
            if added is None:
                break
            selected.extend(added)
        else:
            misses = 1 if mode in ('distinct', 'collision') else int(rng.integers(0, 3))
            if not _add_misses(rng, selected, cells, (ar, ac), (h, w), misses):
                continue
            bg, payload, marker, arrow_color = roles
            g = np.full((h, w), bg, dtype=np.int8)
            for p in cells:
                g[p] = payload
            g[0, ac] = marker
            if double:
                g[ar, -1] = marker
            for a in selected:
                for p in a[2]:
                    g[p] = arrow_color
            if reflect:
                g = np.fliplr(g)
            g = np.rot90(g, rotation).copy()
            try:
                out = solve(g)
                f = pair_features(g)
                if mode == 'distinct' and not (f['unhit_fragment'] and f['distinct_multi']
                                               and f['balanced'] and f['guide_gap'] and f['short_guide']):
                    continue
                if mode == 'collision' and not (f['mixed_collision'] and f['diagonal_move'] and f['unhit']):
                    continue
                if mode == 'separate' and f['collision'] and f['balanced']:
                    continue
                return {'input': g, 'output': out}
            except ValueError:
                continue
    raise RuntimeError('bounded geometry proposals exhausted')


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
                 and lo <= int(value) <= hi, f'{name} must be an integer in {lo}..{hi}')
    _require(num_train > num_test, 'num_train must exceed num_test')
    return int(num_train), int(num_test)


def _grid_key(g):
    return g.shape, g.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, 'seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    varied_palette = bool(rng.random() < .65)
    base_rotation, reflect = int(rng.integers(4)), bool(rng.integers(2))
    held_out = bool(rng.random() < .7)
    puzzle, seen = {'train': [], 'test': []}, set()
    modes = ['distinct', 'collision'] + ['general' if held_out else 'separate']*(ntrain-2)
    rng.shuffle(modes)
    test_modes = ['general']*ntest if held_out else ['combine'] + ['general']*(ntest-1)
    for split, schedule in (('train', modes), ('test', test_modes)):
        for mode in schedule:
            if varied_palette:
                roles = tuple(map(int, rng.choice(10, 4, replace=False)))
            rotation = ((base_rotation+int(rng.integers(1, 4))) % 4
                        if split == 'test' and held_out else base_rotation)
            for _ in range(12):
                pair = _draw(rng, mode, roles, rotation, reflect)
                key = _grid_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not obtain distinct inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(input_grid):
    info = _read(input_grid)
    _, records, counts = _transform(info)
    endpoints = [_endpoint(a) for a in info['arrows']]
    gap = False
    for axis, coordinate in ((0, info['col_axis']), (1, info['row_axis'])):
        if coordinate is not None:
            values = sorted(p[axis] for p in info['cells'] if p[1-axis] == coordinate)
            gap |= any(b-a > 1 for a, b in zip(values, values[1:]))
    hit_records = [r for r in records if r['shooters']]
    short_guide = False
    for side, (r, c) in info['markers'].items():
        axis = 0 if side in ('top', 'bottom') else 1
        coordinate = c if axis == 0 else r
        aligned = [p[axis] for p in info['cells'] if p[1-axis] == coordinate]
        far = max if side in ('top', 'left') else min
        short_guide |= far(aligned) != far(p[axis] for p in info['cells'])
    # A first-obstacle mutant uses the first payload encountered before step
    # five. Compare its actual response (signs and endpoint collision role),
    # rather than merely comparing vector magnitudes that might be irrelevant.
    first_hits = []
    for a in info['arrows']:
        (r, c), (dr, dc), _ = a
        path = [(r+k*dr, c+k*dc) for k in range(1, DISTANCE+1)]
        first_hits.append(next((p for p in path if p in info['cells']), path[-1]))
    first_counts = Counter(first_hits)
    first_obstacle_changes_response = False
    endpoint_normalized_changes_response = False
    for rec in records:
        shooters = [a for a, p in zip(info['arrows'], first_hits) if p in rec['cells']]
        impulse = tuple(sum(a[1][axis] for a in shooters) for axis in (0, 1))
        sign = lambda v: tuple(int(x > 0)-int(x < 0) for x in v)
        collision = any(first_counts[p] > 1 for p in first_hits if p in rec['cells'])
        first_obstacle_changes_response |= (sign(impulse) != sign(rec['impulse'])
                                           or bool(shooters) != bool(rec['shooters'])
                                           or collision != rec['collision'])
        endpoint_vectors = {}
        for a in rec['shooters']:
            p = _endpoint(a)
            v = endpoint_vectors.setdefault(p, [0, 0])
            v[0] += a[1][0]
            v[1] += a[1][1]
        normalized_impulse = tuple(sum(sign(v)[axis] for v in endpoint_vectors.values())
                                   for axis in (0, 1))
        endpoint_normalized_changes_response |= sign(normalized_impulse) != sign(rec['impulse'])
    return {'shape': info['grid'].shape, 'roles': tuple(info[k] for k in
            ('background', 'payload', 'marker', 'arrow_color')),
            'markers': tuple(sorted(info['markers'])), 'marker_count': len(info['markers']),
            'arrows': len(info['arrows']), 'directions': {a[1] for a in info['arrows']},
            'payload_size': len(info['cells']), 'partition_sizes': tuple(len(r['cells']) for r in records),
            'guide_gap': bool(gap),
            'short_guide': bool(short_guide),
            'first_obstacle_changes_response': bool(first_obstacle_changes_response),
            'endpoint_normalized_changes_response': bool(endpoint_normalized_changes_response),
            'hit_fragment': any(len(_components(r['cells'])) > 1 for r in hit_records),
            'unhit_fragment': any(any(r['translation']) and
                                  any(not (piece & set(endpoints)) for piece in _components(r['cells']))
                                  for r in hit_records),
            'distinct_multi': any(len(r['shooters']) > 1 and not r['collision'] for r in hit_records),
            'collision': any(n > 1 for n in counts.values()),
            'mixed_collision': any(r['collision'] and any(counts[_endpoint(a)] == 1 for a in r['shooters'])
                                   for r in hit_records),
            'balanced': any(r['impulse'] == (0, 0) for r in hit_records),
            'cardinal_move': any((r['translation'][0] == 0) != (r['translation'][1] == 0) for r in hit_records),
            'diagonal_move': any(all(r['translation']) for r in hit_records),
            'unhit': any(not r['shooters'] for r in records),
            'misses': sum(info['grid'][p] == info['background'] for p in endpoints),
            'impulses': tuple(r['impulse'] for r in hit_records)}


def validate_generated_puzzle(puzzle):
    _require(type(puzzle) is dict and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(type(pair) is dict and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'output must retain dimensions')
            _require(_grid_key(inp) not in seen, 'duplicate input')
            seen.add(_grid_key(inp))
            info = _read(inp)
            expected, records, _ = _transform(info)
            _require(np.array_equal(expected, out), 'incorrect output')
            _require(not np.array_equal(inp, out), 'identity example')
            _require(all(out[p] == info['payload'] for p in info['fixed']), 'axial payload must stay fixed')
            _require(all(out[p] == info['arrow_color'] for a in info['arrows'] for p in a[2]),
                     'shooters must remain unchanged')
            # The partition's complete relative geometry, including disconnected
            # fragments, is conserved under precisely one translation.
            total = 0
            for rec in records:
                dr, dc = rec['translation']
                _require(rec['targets'] == {(r+dr, c+dc) for r, c in rec['cells']}, 'non-rigid motion')
                _require(all(out[p] == rec['color'] for p in rec['targets']), 'wrong region color')
                total += len(rec['targets'])
            _require(total + len(info['fixed']) == len(info['cells']), 'payload geometry lost or duplicated')
            features[split].append(pair_features(inp))
    train = features['train']
    for feature in ('unhit_fragment', 'distinct_multi', 'mixed_collision', 'balanced',
                    'cardinal_move', 'diagonal_move', 'unhit', 'guide_gap', 'short_guide',
                    'first_obstacle_changes_response', 'endpoint_normalized_changes_response'):
        _require(any(f[feature] for f in train), f'missing training witness: {feature}')
    _require({f['marker_count'] for f in train} == {1, 2}, 'teach one and two cuts')
    _require(any(f['misses'] for f in train), 'need a visible five-step distance witness')


def _puzzle_key(puzzle):
    return tuple((_grid_key(_grid(p['input'])), _grid_key(_grid(p['output'])))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded checks on 42 seeds, metamorphic invariants, and count boundaries."""
    # Hand-constructed family members outside the sampler's outward profiles:
    # an inward transfer can cross a cut, and a three-shot collision acts on a
    # thin cohort without clipping or being mistaken for three painted cells.
    inward = np.zeros((24, 24), dtype=np.int8)
    inward[10:15, 9:16] = 2
    inward[0, 12] = 8
    for p in _arrow((10, 4), (0, 1))[2]:
        inward[p] = 4
    answer = solve(inward)
    assert np.all(answer[10:15, 21:24] == 4)
    assert np.all(answer[10:15, 9:12] == 0)
    assert np.all(answer[10:15, 13:16] == 2)
    triple = np.zeros((24, 24), dtype=np.int8)
    triple[10:15, 9:13] = 2
    triple[12, 13:16] = 2
    triple[0, 12] = 8
    for tip, direction in (((17, 10), (-1, 1)), ((7, 10), (1, 1)), ((12, 20), (0, -1))):
        for p in _arrow(tip, direction)[2]:
            triple[p] = 4
    answer = solve(triple)
    assert np.all(answer[12, 21:24] == 8)
    assert np.all(answer[12, 13:16] == 0)
    assert np.all(answer[10:15, 9:13] == 2)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, structures = [], set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = _grid(pair['input'])
                before = inp.copy()
                assert np.array_equal(solve(inp), pair['output'])
                assert np.array_equal(inp, before), 'solve mutated its input'
                records.append(dict(pair_features(inp), split=split))
                roles = pair_features(inp)['roles']
                normalized = np.zeros_like(inp)
                for k, color in enumerate(roles):
                    normalized[inp == color] = k
                structures.add(_grid_key(normalized))
        # Global rotation/reflection preserves the vector and bbox interpretation.
        for operation in (lambda g: np.rot90(g), lambda g: np.fliplr(g)):
            p = puzzle['train'][0]
            assert np.array_equal(solve(operation(p['input'])), operation(p['output']))
        remap = np.array([7, 4, 9, 0, 2, 6, 1, 8, 5, 3], dtype=np.int8)
        p = puzzle['train'][0]
        assert np.array_equal(solve(remap[p['input']]), remap[p['output']])
        # Omitting A: leaving the payload fixed with only guide completion gives
        # a different answer. Omitting B: moving only connected hit components
        # misses a fragment that shares the cut region but receives no endpoint.
        fragment_witness = False
        for p in puzzle['train']:
            info = _read(p['input'])
            out, regions, counts = _transform(info)
            hits = set(counts)
            for r in regions:
                if any(r['translation']) and any(not (piece & hits) for piece in _components(r['cells'])):
                    assert r['color'] != info['payload']
                    assert r['targets'] != r['cells']
                    fragment_witness = True
                if r['shooters'] and r['impulse'] == (0, 0):
                    assert r['translation'] == (0, 0)
                    assert all(out[q] == info['arrow_color'] for q in r['cells']) or r['collision']
                if len(r['shooters']) > 1 and not r['collision']:
                    assert r['color'] == info['arrow_color'], 'distinct hits are not a collision'
                if r['collision']:
                    assert r['color'] == info['marker'], 'collision color must dominate ordinary hits'
        assert fragment_witness
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len(structures) >= 180
    assert len({r['shape'] for r in records}) >= 45
    assert len({r['payload_size'] for r in records}) >= 30
    assert len({r['arrows'] for r in records}) >= 7
    assert {d for r in records for d in r['directions']} == set(DIRECTIONS)
    assert {r['roles'][0] for r in records} == set(range(10))
    assert len({r['roles'] for r in records}) >= 80
    for split in ('train', 'test'):
        subset = [r for r in records if r['split'] == split]
        assert {r['marker_count'] for r in subset} == {1, 2}
        for name in ('hit_fragment', 'distinct_multi', 'collision', 'balanced', 'cardinal_move', 'diagonal_move'):
            assert any(r[name] for r in subset), name
    assert _puzzle_key(generate_puzzle(19, 2, 1)) == _puzzle_key(generate_puzzle(19, 2, 1))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(100+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return {'seeds': 42, 'pairs': len(records), 'structures': len(structures),
            'shapes': len({r['shape'] for r in records}),
            'arrow_counts': sorted({r['arrows'] for r in records}),
            'payload_sizes': sorted({r['payload_size'] for r in records})}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
