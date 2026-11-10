"""Rotating weld census: rotate inner layers, then count their welded objects.

Only standard library and NumPy are used. No files, traces, or caches are written.
"""
from collections import Counter
import hashlib

import numpy as np


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _check(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _check(bool(np.all((a >= 0) & (a <= 9))), "color bounds")
    return a


def _groups(cells):
    left = set(cells)
    groups = []
    while left:
        start = min(left)
        left.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in left:
                    left.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _parts(grid, bg):
    result = []
    for color in sorted(int(x) for x in np.unique(grid) if x != bg):
        for cells in _groups(map(tuple, np.argwhere(grid == color).tolist())):
            rr, cc = zip(*cells)
            result.append(dict(color=color, cells=cells,
                               box=(min(rr), min(cc), max(rr)+1, max(cc)+1)))
    return result


def _perimeter(r, c, h, w):
    return {(y, x) for y in range(r, r+h) for x in range(c, c+w)
            if y in (r, r+h-1) or x in (c, c+w-1)}


def _rectangle(part):
    r, c, e, f = part['box']
    return part['cells'] == _perimeter(r, c, e-r, f-c)


def _turn(p, n, k):
    r, c = p
    for _ in range(k % 4):
        r, c = c, n-1-r
    return r, c


def _coordinate_census(core, bg, colors, turns, check_collisions=True):
    """Coordinate/set reconstruction, separate from array-based solve."""
    n = len(core)-4
    working = core[1:-1, 1:-1]
    source = core[2:-2, 2:-2]
    used, counts, features = set(), {}, {}
    for color in colors:
        src = set(map(tuple, np.argwhere(source == color).tolist()))
        fixed = {(r, c) for r, c in np.argwhere(working == color).tolist()
                 if r in (0, n+1) or c in (0, n+1)}
        moving = {tuple(x+1 for x in _turn(p, n, turns.get(color, 0))) for p in src}
        if check_collisions:
            _check(not used.intersection(moving), "rotated layers collide")
        used.update(moving)
        counts[color] = len(_groups(fixed | moving))
        features[color] = dict(area=len(src)+len(fixed), moving=len(_groups(src)),
                               tracks=len(_groups(fixed)), count=counts[color])
    return counts, features


def _inspect(input_grid):
    grid = _grid(input_grid)
    vals, frequencies = np.unique(grid, return_counts=True)
    _check(np.sum(frequencies == frequencies.max()) == 1, "ambiguous background")
    bg = int(vals[frequencies.argmax()])
    parts = _parts(grid, bg)
    squares = [p for p in parts if _rectangle(p)
               and p['box'][2]-p['box'][0] == p['box'][3]-p['box'][1]
               and p['box'][2]-p['box'][0] >= 9]
    _check(len(squares) == 1, "need one square workshop outline")
    workshop = squares[0]
    r, c, e, f = workshop['box']
    core = grid[r:e, c:f].copy()
    n = len(core)-4
    _check(n >= 5 and n % 2 == 1, "odd inner square of side at least five")
    outside = grid.copy()
    outside[r:e, c:f] = bg
    external = _parts(outside, bg)
    frames = [p for p in external if _rectangle(p)
              and min(p['box'][2]-p['box'][0], p['box'][3]-p['box'][1]) in (5, 7)
              and max(p['box'][2]-p['box'][0], p['box'][3]-p['box'][1]) >
              min(p['box'][2]-p['box'][0], p['box'][3]-p['box'][1])]
    _check(1 <= len(frames) <= 3, "need one through three tally frames")
    colors = tuple(sorted(p['color'] for p in frames))
    _check(len(set(colors)) == len(colors), "distinct frame colors")
    _check(bg not in colors and workshop['color'] not in colors, "role colors collide")
    _check(set(map(int, np.unique(core[1:-1, 1:-1]))) <= set(colors) | {bg},
           "unknown workshop color")
    _check(all(np.any(core[2:-2, 2:-2] == color) for color in colors),
           "every category needs movable material")
    for frame in frames:
        y, x, a, b = frame['box']
        _check(np.all(grid[y+1:a-1, x+1:b-1] == bg), "nonempty tally frame")
        # The workshop-facing end is the one nearer its center along the long axis.
        horizontal = b-x > a-y
        delta = c+f-x-b if horizontal else r+e-y-a
        _check(delta != 0, "equidistant frame ends")
        frame['direction'] = ('right' if delta > 0 else 'left') if horizontal else (
            'down' if delta > 0 else 'up')
    frame_ids = {id(p) for p in frames}
    markers = [p for p in external if id(p) not in frame_ids]
    turns = {}
    for marker in markers:
        color = marker['color']
        y, x, a, b = marker['box']
        length = len(marker['cells'])
        _check(color in colors and color not in turns, "one matched bar per color")
        _check((a-y == 1 or b-x == 1) and length == (a-y)*(b-x), "counter is not a bar")
        _check(1 <= length <= 9, "bar length must be 1..9")
        turns[color] = length
    # External objects are detached from every reserved rectangle, even diagonally.
    objects = [workshop] + external
    for i, p in enumerate(objects):
        for q in objects[i+1:]:
            _check(not any((y+dy, x+dx) in q['cells'] for y, x in p['cells']
                           for dy in (-1, 0, 1) for dx in (-1, 0, 1)),
                   "external structures touch")
    counts, features = _coordinate_census(core, bg, colors, turns)
    for frame in frames:
        y, x, a, b = frame['box']
        _check(counts[frame['color']] <= (max(a-y, b-x)-3)//2, "tally overflow")
    return dict(background=bg, workshop=workshop, core=core, colors=colors,
                frames=frames, markers=markers, turns=turns, counts=counts,
                features=features, n=n)


def _draw_tallies(grid, info, counts):
    out = np.full(grid.shape, info['background'], dtype=np.int8)
    for frame in info['frames']:
        color = frame['color']
        for r, c in frame['cells']:
            out[r, c] = color
        r, c, e, f = frame['box']
        direction = frame['direction']
        for j in range(1, counts[color]+1):
            y = e-1-2*j if direction == 'down' else r+2*j if direction == 'up' else (r+e-1)//2
            x = f-1-2*j if direction == 'right' else c+2*j if direction == 'left' else (c+f-1)//2
            out[y, x] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """All semantic roles, rotations, contacts, and tally ends come from input."""
    grid = _grid(input_grid)
    info = _inspect(grid)
    core, bg = info['core'], info['background']
    source, body = core[2:-2, 2:-2], core[1:-1, 1:-1]
    counts = {}
    for color in info['colors']:
        layer = body == color
        layer[1:-1, 1:-1] = np.rot90(source == color, -info['turns'].get(color, 0))
        counts[color] = len(_groups(map(tuple, np.argwhere(layer).tolist())))
    return _draw_tallies(grid, info, counts)


def _shape(rng, area):
    cells = {(0, 0)}
    while len(cells) < area:
        boundary = sorted({(r+dr, c+dc) for r, c in cells
                           for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))} - cells)
        cells.add(boundary[int(rng.integers(len(boundary)))])
    y, x = min(r for r, c in cells), min(c for r, c in cells)
    return {(r-y, c-x) for r, c in cells}


def _sample_core(rng, n, bg, neutral, colors, turn_rows, teaching, require_long_effect=False):
    """Bounded rejection over small geometric scenes; no payloads are retained."""
    ring = ([(0, c) for c in range(n+2)] + [(r, n+1) for r in range(1, n+2)] +
            [(n+1, c) for c in range(n, -1, -1)] + [(r, 0) for r in range(n, 0, -1)])
    for attempt in range(1600):
        core = np.full((n+4, n+4), bg, dtype=np.int8)
        for p in _perimeter(0, 0, n+4, n+4):
            core[p] = neutral
        source = core[2:-2, 2:-2]
        masks, okay = {}, True
        for color in colors:
            placed = set()
            wanted = int(rng.integers(2, 4)) if teaching else int(rng.integers(1, 4))
            for _ in range(wanted):
                shape = _shape(rng, int(rng.integers(1, 6)))
                h, w = 1+max(r for r, c in shape), 1+max(c for r, c in shape)
                candidates = []
                for y in range(n-h+1):
                    for x in range(n-w+1):
                        points = {(y+r, x+c) for r, c in shape}
                        if any(source[p] != bg for p in points):
                            continue
                        if any((r+dr, c+dc) in placed for r, c in points
                               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0))):
                            continue
                        candidates.append((points, any(r in (0, n-1) or c in (0, n-1) for r, c in points)))
                if not candidates:
                    okay = False
                    break
                edge = [v for v in candidates if v[1]]
                pool = edge if edge and rng.random() < .90 else candidates
                points = pool[int(rng.integers(len(pool)))][0]
                placed.update(points)
                for p in points:
                    source[p] = color
            if not okay:
                break
            masks[color] = placed
        if not okay:
            continue
        # The two matched teaching scenes must both have disjoint rotated layers.
        for turns in turn_rows:
            used = set()
            for color, k in zip(colors, turns):
                rotated = {_turn(p, n, k) for p in masks[color]}
                if used & rotated:
                    okay = False
                    break
                used.update(rotated)
            if not okay:
                break
        if not okay:
            continue
        # Rim tracks may bend around a corner; actual contacts, not placement
        # metadata, decide which pieces become a single object.
        occupied = set()
        for color in colors:
            for _ in range(int(rng.integers(1, 3))):
                length = int(rng.integers(2, min(8, n+3)))
                starts = [j for j in range(len(ring)) if not any(
                    (j+t) % len(ring) in occupied for t in range(-1, length+1))]
                if not starts:
                    continue
                j = int(rng.choice(starts))
                for t in range(length):
                    index = (j+t) % len(ring)
                    occupied.add(index)
                    r, c = ring[index]
                    core[1+r, 1+c] = color
        signatures = {}
        for color in colors:
            signatures[color] = [_coordinate_census(core, bg, [color], {color:k})[0][color]
                                 for k in range(4)]
        if max(v for values in signatures.values() for v in values) > 6:
            continue
        if require_long_effect and not any(
                turns >= 5 and signatures[color][turns % 4] != signatures[color][0]
                for color, turns in zip(colors, turn_rows[0])):
            continue
        if teaching:
            a, b = (signatures[color] for color in colors)
            if not (all(a[1] != a[k] for k in (0, 2, 3)) and b[2] != b[0] and b[2] != b[1]):
                continue
            # Include an actual weld, and distinguish ignoring the fixed material.
            c, feats = _coordinate_census(core, bg, colors, dict(zip(colors, turn_rows[0])))
            if not any(feats[x]['count'] < feats[x]['moving']+feats[x]['tracks'] for x in colors):
                continue
            if not any(feats[x]['count'] != feats[x]['moving'] for x in colors):
                continue
        return core, signatures
    raise RuntimeError("bounded workshop sampling exhausted")


def _free_starts(blocked, h, w):
    H, W = blocked.shape
    if h > H or w > W:
        return np.empty((0, 2), dtype=int)
    prefix = np.pad(blocked.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    mass = prefix[h:, w:] - prefix[:-h, w:] - prefix[h:, :-w] + prefix[:-h, :-w]
    return np.argwhere(mass == 0)


def _reserve(blocked, r, c, h, w):
    blocked[max(0,r-1):min(len(blocked),r+h+1), max(0,c-1):min(blocked.shape[1],c+w+1)] = True


def _scene_layout(rng, core, bg, colors, signatures, turn_rows):
    """Pack freely on 30x30, then crop with random margins; at most 32 attempts."""
    side = len(core)
    for attempt in range(32):
        base = np.full((30, 30), bg, dtype=np.int8)
        blocked = np.zeros((30, 30), dtype=bool)
        r, c = int(rng.integers(0, 31-side)), int(rng.integers(0, 31-side))
        base[r:r+side, c:c+side] = core
        _reserve(blocked, r, c, side, side)
        okay = True
        first_axis = first_sign = None
        for color in rng.permutation(colors):
            cap = max(signatures[int(color)])
            short = 7 if len(colors) == 1 and rng.random() < .3 else 5
            long = max(short+2, 2*cap+3) + int(rng.integers(0, 3))
            horizontal = not first_axis if len(turn_rows) > 1 and first_axis is not None else bool(rng.integers(2))
            h, w = (short, long) if horizontal else (long, short)
            starts = _free_starts(blocked, h, w)
            if len(starts):
                delta = (2*c+side-1)-(2*starts[:,1]+w-1) if horizontal else (
                    (2*r+side-1)-(2*starts[:,0]+h-1))
                valid = delta != 0
                if len(turn_rows) > 1 and first_sign is not None:
                    valid &= (delta > 0) != first_sign
                starts = starts[valid]
            if not len(starts):
                okay = False
                break
            y, x = map(int, starts[int(rng.integers(len(starts)))])
            delta = (2*c+side-1)-(2*x+w-1) if horizontal else (2*r+side-1)-(2*y+h-1)
            if first_axis is None:
                first_axis, first_sign = horizontal, delta > 0
            for p in _perimeter(y, x, h, w):
                base[p] = color
            _reserve(blocked, y, x, h, w)
        if not okay:
            continue
        grids = [base.copy() for row in turn_rows]
        # Reserve the maximum length used by the matched pair so only counters
        # change between its two demonstrations, including their canvas bounds.
        for color_index in rng.permutation(len(colors)):
            color = colors[int(color_index)]
            length = max(row[int(color_index)] for row in turn_rows)
            if not length:
                continue
            horizontal = bool(rng.integers(2))
            h, w = (1, length) if horizontal else (length, 1)
            starts = _free_starts(blocked, h, w)
            if not len(starts):
                okay = False
                break
            y, x = map(int, starts[int(rng.integers(len(starts)))])
            for grid, row in zip(grids, turn_rows):
                for j in range(row[int(color_index)]):
                    grid[y+(0 if horizontal else j), x+(j if horizontal else 0)] = color
            _reserve(blocked, y, x, h, w)
        if not okay:
            continue
        union = np.logical_or.reduce([grid != bg for grid in grids])
        rr, cc = np.nonzero(union)
        top = max(0, int(rr.min())-int(rng.integers(0, 4)))
        left = max(0, int(cc.min())-int(rng.integers(0, 4)))
        bottom = min(30, int(rr.max())+1+int(rng.integers(0, 4)))
        right = min(30, int(cc.max())+1+int(rng.integers(0, 4)))
        grids = [grid[top:bottom, left:right].copy() for grid in grids]
        if any(_inspect(grid)['background'] != bg for grid in grids):
            continue
        return [dict(input=grid, output=solve(grid)) for grid in grids]
    raise RuntimeError("bounded scene packing exhausted")


def _palette(rng, count, previous_bg=None):
    bg = int(rng.choice([c for c in range(10) if c != previous_bg]))
    chosen = [int(c) for c in rng.choice([c for c in range(10) if c != bg], count+1, replace=False)]
    return bg, chosen[0], chosen[1:]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    bg, neutral, colors = _palette(rng, 2)
    n = int(rng.choice([5, 7, 9]))
    rows = [[1, 2], [3, 0]]
    core, signatures = _sample_core(rng, n, bg, neutral, colors, rows, True)
    teaching = _scene_layout(rng, core, bg, colors, signatures, rows)
    puzzle = dict(train=teaching, test=[])
    seen = {(p['input'].shape, p['input'].tobytes()) for p in teaching}
    # Every puzzle has a visible shift: new long counters, or expansion from
    # at most two categories to three. No hidden output convention changes.
    category_shift = bool(rng.integers(2))
    previous_bg = bg
    for split, count in (('train', nt-2), ('test', nq)):
        for index in range(count):
            for retry in range(8):
                k = 3 if split == 'test' and category_shift else int(
                    rng.integers(1, 3 if category_shift else 4))
                bg, neutral, colors = _palette(rng, k, previous_bg)
                n = int(rng.choice([7, 9, 11] if k == 3 else [5, 7, 9, 11]))
                maximum = 7 if split == 'test' or category_shift else 3
                turns = [int(x) for x in rng.integers(0, maximum+1, size=k)]
                if split == 'test':
                    turns[int(rng.integers(k))] = int(rng.integers(5, 8))
                core, signatures = _sample_core(rng, n, bg, neutral, colors, [turns], False,
                                                require_long_effect=split == 'test')
                pair = _scene_layout(rng, core, bg, colors, signatures, [turns])[0]
                key = pair['input'].shape, pair['input'].tobytes()
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct input sampling exhausted")
            seen.add(key)
            previous_bg = bg
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _expected(grid, info):
    # Validation draws tally cells directly rather than invoking solve.
    result = np.full(grid.shape, info['background'], dtype=np.int8)
    for f in info['frames']:
        r, c, e, b = f['box']
        color, direction = f['color'], f['direction']
        for p in f['cells']:
            result[p] = color
        count = info['counts'][color]
        if direction in ('right', 'left'):
            xs = [b-1-2*j if direction == 'right' else c+2*j for j in range(1, count+1)]
            for x in xs:
                result[(r+e-1)//2, x] = color
        else:
            ys = [e-1-2*j if direction == 'down' else r+2*j for j in range(1, count+1)]
            for y in ys:
                result[y, (c+b-1)//2] = color
    return result


def _training_evidence(infos):
    entries = [(color, f['area'], f['count']) for info in infos
               for color, f in info['features'].items()]
    collision = any(c == d and a == b and n != m for c, a, n in entries for d, b, m in entries)
    residues = {info['turns'].get(c, 0) % 4 for info in infos for c in info['colors']}
    no_rotation = clockwise = independent = welding = fixed = False
    for info in infos:
        core, bg, colors = info['core'], info['background'], info['colors']
        actual = info['counts']
        zero = _coordinate_census(core, bg, colors, {})[0]
        reverse = _coordinate_census(core, bg, colors, {c:-v for c,v in info['turns'].items()}, False)[0]
        no_rotation |= zero != actual
        clockwise |= reverse != actual
        for k in range(4):
            uniform = _coordinate_census(core, bg, colors, {c:k for c in colors})[0]
            if uniform == actual:
                break
        else:
            independent = True
        welding |= any(f['count'] < f['moving']+f['tracks'] for f in info['features'].values())
        fixed |= any(f['count'] != f['moving'] for f in info['features'].values())
    directions = {f['direction'] for info in infos for f in info['frames']}
    layout = (bool(directions & {'left', 'right'}) and bool(directions & {'up', 'down'})
              and bool(directions & {'right', 'down'}) and bool(directions & {'left', 'up'}))
    return dict(area_collision=collision, residues=residues == set(range(4)),
                rotation=no_rotation, clockwise=clockwise, independent=independent,
                welding=welding, fixed_material=fixed, frame_directions=layout)


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _check(inp.shape == out.shape, "output dimensions")
            key = inp.shape, inp.astype(np.int8).tobytes()
            _check(key not in seen, "duplicate input")
            seen.add(key)
            info = _inspect(inp)
            _check(np.array_equal(out, _expected(inp, info)), "wrong welded census or tally")
            _check(np.array_equal(out, solve(inp)), "solve disagreement")
            if split == 'train':
                infos.append(info)
            else:
                long_effect = any(k >= 5 and _coordinate_census(
                    info['core'],info['background'],[c],{c:k})[0][c] !=
                    _coordinate_census(info['core'],info['background'],[c],{})[0][c]
                    for c,k in info['turns'].items())
                _check(long_effect, "test must expose the long counter's census effect")
    evidence = _training_evidence(infos)
    _check(all(evidence.values()), "insufficient demonstrations: " + str(evidence))


def self_test():
    """42 seeds, streaming metrics, bounded memory; no saved sample collection."""
    structures, dimensions, backgrounds, sides, sizes, categories, turns = (set() for _ in range(7))
    signatures, shapes = set(), set()
    totals = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                info = _inspect(inp)
                dimensions.add(inp.shape)
                backgrounds.add(info['background'])
                sizes.add(info['n'])
                categories.add(len(info['colors']))
                turns.update(info['turns'].get(c, 0) for c in info['colors'])
                sides.update(f['direction'] for f in info['frames'])
                roles = {}
                canonical = bytes(roles.setdefault(int(v), len(roles)) for v in info['core'].flat)
                structures.add((info['core'].shape, canonical))
                for color in info['colors']:
                    for group in _groups(map(tuple, np.argwhere(info['core'][2:-2,2:-2] == color).tolist())):
                        r, c = min(y for y,x in group), min(x for y,x in group)
                        shapes.add(tuple(sorted((y-r, x-c) for y,x in group)))
                totals['pairs'] += 1
                totals['welded_categories'] += sum(f['count'] < f['moving']+f['tracks']
                                                   for f in info['features'].values())
                totals['rotation_changes_census'] += (
                    _coordinate_census(info['core'], info['background'], info['colors'], {})[0] != info['counts'])
        signatures.add(digest.hexdigest())
        # Palette equivariance exercises every semantic color role, including 0.
        pair = puzzle['test'][0]
        permutation = np.random.default_rng(seed+42000).permutation(10)
        _check(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]), "palette equivariance")
    _check(len(signatures) == 42, "repeated puzzles")
    _check(backgrounds == set(range(10)), "background coverage")
    _check(sides == {'left', 'right', 'up', 'down'}, "tally direction coverage")
    _check(sizes == {5, 7, 9, 11} and categories == {1, 2, 3}, "structural coverage")
    _check(turns >= set(range(8)), "counter coverage")
    _check(len(structures) >= 100 and len(dimensions) >= 30 and len(shapes) >= 25, "geometry diversity")
    _check(totals['welded_categories'] >= 80 and totals['rotation_changes_census'] >= 60, "weak interaction coverage")
    a, b = generate_puzzle(1901, 2, 1), generate_puzzle(1901, 2, 1)
    _check(all(np.array_equal(p[k], q[k]) for s in a for p,q in zip(a[s],b[s]) for k in p), "determinism")
    for nt, nq in ((2,1), (10,4), (5,4)):
        p = generate_puzzle(800+nt, nt, nq)
        _check(len(p['train']) == nt and len(p['test']) == nq, "explicit counts")
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Edge adjacency joins; diagonal contact does not. A rim bridge can merge
    # two separate movable pieces into one object.
    _check(len(_groups({(0,0),(1,1)})) == 2, "diagonal welding")
    _check(len(_groups({(0,0),(0,1),(0,2),(1,0),(1,2)})) == 1, "rim bridge")
    bad = {s:[{k:v.copy() for k,v in pair.items()} for pair in a[s]] for s in a}
    bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt census accepted")
    return dict(seeds=42, pairs=totals['pairs'], workshop_structures=len(structures),
                canvas_shapes=len(dimensions), movable_shapes=len(shapes),
                backgrounds=sorted(backgrounds), tally_directions=sorted(sides),
                inner_sizes=sorted(sizes), categories=sorted(categories),
                control_counts=sorted(turns), welded_categories=totals['welded_categories'],
                rotation_changes_census=totals['rotation_changes_census'])
