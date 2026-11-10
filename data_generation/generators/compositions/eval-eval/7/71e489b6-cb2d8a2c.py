"""Warning-island detours: recovered damage is a spatial routing program.

Only standard-library Python and NumPy are used.  solve() reads one input;
the constructor's rectangle/defect plans are never available to it.
"""
from __future__ import annotations

import json
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(int, copy=True)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    steps = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
             if (dy or dx) and (diagonal or abs(dy) + abs(dx) == 1)]
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, points = [first], [first]
        while stack:
            y, x = stack.pop()
            for dy, dx in steps:
                q = y + dy, x + dx
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    points.append(q)
        groups.append(points)
    return groups


def _dilate(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[y:y+h, x:x+w]
                                 for y in range(3) for x in range(3)])


def _box(points):
    a = np.asarray(points)
    y, x = a.min(axis=0)
    v, u = a.max(axis=0)
    return int(y), int(v), int(x), int(u)


def _forward(a, direction):
    if direction == 0:
        return a.copy()
    if direction == 1:
        return a.T.copy()
    if direction == 2:
        return a[::-1].copy()
    return a.T[::-1].copy()


def _physical(a, direction):
    if direction == 0:
        return a.copy()
    if direction == 1:
        return a.T.copy()
    if direction == 2:
        return a[::-1].copy()
    return a[::-1].T.copy()


def _roles(a):
    corners = (a[0, 0], a[0, -1], a[-1, 0], a[-1, -1])
    if len(set(map(int, corners))) != 1:
        raise ValueError("the four corners must show the field")
    field = int(corners[0])
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 3:
        raise ValueError("input must have field, solid, and seed colors")
    singleton = [int(c) for c, n in zip(colors, counts) if c != field and n == 1]
    if len(singleton) != 1:
        raise ValueError("one unique seed color is required")
    ink = singleton[0]
    solid = next(int(c) for c in colors if c not in (field, ink))
    y, x = map(int, np.argwhere(a == ink)[0])
    candidates = [i for i, yes in enumerate(
        (y == 0, x == 0, y == a.shape[0]-1, x == a.shape[1]-1)) if yes]
    if len(candidates) != 1:
        raise ValueError("seed must lie on exactly one boundary")
    return field, solid, ink, candidates[0]


def _warning_counts(halo, lo, hi):
    counts = [0, 0]
    midpoint = (lo + hi) / 2
    for component in _components(halo, diagonal=True):
        xs = [x for _, x in component]
        if max(xs) < midpoint:
            counts[0] += 1
        elif min(xs) > midpoint:
            counts[1] += 1
        else:
            raise ValueError("a warning island crosses the obstacle midline")
    return tuple(counts)


def _decode(input_grid):
    original = _grid(input_grid)
    field, solid, ink, direction = _roles(original)
    a = _forward(original, direction)
    h, w = a.shape
    initial = int(np.argwhere(a == ink)[0, 1])
    base = np.zeros_like(a, dtype=bool)
    rectangles = []
    for component in _components(a == solid):
        if len(component) <= 2:
            continue
        if len(component) < 8:
            raise ValueError("an unsupported intermediate-size solid component")
        top, bottom, lo, hi = _box(component)
        if not (3 <= bottom-top+1 <= 5 and hi-lo+1 >= 7
                and hi-lo+1 >= bottom-top+3):
            raise ValueError("obstacles must be transverse thick rectangles")
        if not all(a[y, x] == solid for y in (top, bottom) for x in (lo, hi)):
            raise ValueError("each rectangle must retain its four corners")
        region = np.zeros_like(base)
        region[top:bottom+1, lo:hi+1] = True
        if np.count_nonzero(region & (a == solid)) != len(component):
            raise ValueError("the rectangle's entire retained body must be four-connected")
        if (base & region).any() or (region & (a == ink)).any():
            raise ValueError("overlapping or seed-containing obstacles")
        if np.count_nonzero(region & (a == field)) > region.sum() // 3:
            raise ValueError("too little intact rectangle evidence")
        base |= region
        rectangles.append((top, bottom, lo, hi))
    rectangles.sort()
    if not 1 <= len(rectangles) <= 3:
        raise ValueError("one to three obstacles are required")
    holes = base & (a == field)
    extras = ~base & (a == solid)
    halos = _dilate(holes) & ~holes
    path = np.zeros_like(base)
    current, start, previous_bottom = initial, 0, -1
    records = []
    for top, bottom, lo, hi in rectangles:
        local_holes = np.zeros_like(base)
        local_holes[top:bottom+1, lo:hi+1] = holes[top:bottom+1, lo:hi+1]
        local_halo = _dilate(local_holes) & ~holes
        left, right = _warning_counts(local_halo, lo, hi)
        if left == right or not (1 <= max(left, right) <= 3):
            raise ValueError("warning counts must be unequal, with maximum 1..3")
        d = max(left, right) + 1
        side = 1 if left > right else -1
        envelope = local_halo.copy()
        envelope[top:bottom+1, lo:hi+1] = True
        near, far, low, high = _box(np.argwhere(envelope))
        turn = near - d
        target = high + d if side == 1 else low - d
        if not (turn >= 1 and turn > previous_bottom and start <= turn
                and 0 <= target < w and lo <= current <= hi):
            raise ValueError("the encoded detour is not feasible")
        for y in range(start, turn+1):
            path[y, current] = True
        for x in range(min(current, target), max(current, target)+1):
            path[turn, x] = True
        records.append({"rectangle": (top, bottom, lo, hi),
                        "envelope": (near, far, low, high),
                        "counts": (left, right), "clearance": d,
                        "side": side, "turn": turn, "target": target,
                        "hole_count": int(local_holes.sum()),
                        "hole_groups": len(_components(local_holes, True)),
                        "halo_groups4": len(_components(local_halo)),
                        "leading_spill": near < top,
                        "lateral_spill": low < lo or high > hi,
                        "raw_side_wrong": (
                            (local_holes[:, :int(np.ceil((lo+hi)/2))].sum()
                             < local_holes[:, int(np.floor((lo+hi)/2))+1:].sum())
                            if side == 1 else
                            (local_holes[:, :int(np.ceil((lo+hi)/2))].sum()
                             > local_holes[:, int(np.floor((lo+hi)/2))+1:].sum()))})
        current, start, previous_bottom = target, turn, far
    path[start:, current] = True
    if (path & (base | _dilate(holes))).any():
        raise ValueError("route intersects a rectangle or its warning zone")
    output = np.where(base, solid, field).astype(int)
    output[holes] = field
    output[halos | path] = ink
    edge = np.zeros_like(base)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    info = {"palette": (field, solid, ink), "direction": direction,
            "base": base, "holes": holes, "extras": extras,
            "halos": halos, "path": path, "records": records,
            "shape": original.shape, "forward_shape": a.shape,
            "near_extra": bool((extras & halos).any()),
            "far_erasure": bool((extras & ~halos & ~path).any()),
            "route_extra": bool((extras & path).any()),
            "clipped": bool((holes & edge).any()),
            "repeated": any(r['side'] == s['side']
                            for r, s in zip(records, records[1:]))}
    return _physical(output, direction), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, recovered obstacle, instruction, and route from input."""
    return _decode(input_grid)[0]


_MOTIFS = {
    "single": ((0, 0),),
    "vertical": ((0, 0), (1, 0)),
    "triple": ((0, 0), (1, 0), (2, 0)),
    "diagonal": ((0, 0), (1, 1)),
    "L": ((0, 0), (1, 0), (1, 1)),
    "dense": ((0, 0), (1, 0), (1, 1), (2, 1)),
    "merge": ((0, 0), (0, 2)),
    "touch": ((0, 0), (0, 3)),
    "corner_touch": ((0, 0), (3, 3)),
}


def _pattern(rng, count, height, style="random"):
    """Coordinates measured inward from one end; separate packets have gaps >=4."""
    if not count:
        return []
    points, offset = [], 0
    for i in range(count):
        if style == "singles":
            kind = "single"
        elif style == "merge":
            kind = "corner_touch"
        elif style == "rival":
            kind = "dense" if count == 1 else "single"
        else:
            choices = ["single", "vertical", "diagonal", "L", "merge", "touch"]
            if count == 3:
                choices = ["single", "vertical"]
            elif count == 2:
                choices = ["single", "vertical", "diagonal", "L"]
            choices = [k for k in choices
                       if max(y for y, _ in _MOTIFS[k]) + 3 <= height]
            kind = str(rng.choice(choices))
        motif = _MOTIFS[kind]
        if kind == 'corner_touch' and rng.random() < .5:
            # Grow the inward packet without filling the diagonal-only bridge.
            motif += ((3, int(rng.choice((4, 5)))),)
        extent_y = max(y for y, _ in motif)
        positions = [y for y in range(height-extent_y)
                     if not any((y+dy in (0, height-1)) and offset+dx == 0
                                for dy, dx in motif)]
        if not positions:
            raise ValueError("motif cannot retain the rectangle corners")
        y0 = int(rng.choice(positions))
        # Anchored singles after the first can reach a leading/trailing side.
        if style == "singles" and i == 1:
            y0 = 0
        points.extend((y0+dy, offset+dx) for dy, dx in motif)
        offset += max(x for _, x in motif) + 4
        if style == "random" and count < 3 and rng.random() < .3:
            offset += 1
    if style == 'merge' and rng.random() < .5:
        points = [(height-1-y, x) for y, x in points]
    return points


def _obstacle_style(rng, n, lower, sign, style, compact=False):
    height = 5 if style in ("competition", "merge") else 3 if compact else int(rng.integers(3, 6))
    winning_style = "singles" if style == "competition" else "merge" if style == "merge" else "random"
    winning = _pattern(rng, n, height, winning_style)
    losing = _pattern(rng, lower, height, "rival" if style == "competition" else "random")
    left, right = (winning, losing) if sign == 1 else (losing, winning)
    extent = max([x for _, x in left+right], default=0)
    minimum_length = max(7, 2*(extent+1)+2)
    lead = int(any(y == 0 for y, _ in left+right))
    trail = int(any(y == height-1 for y, _ in left+right))
    return {"n": n, "lower": lower, "sign": sign, "height": height,
            "left": left, "right": right, "minimum": minimum_length,
            "lead": lead, "trail": trail}


def _lateral_plan(rng, width, styles):
    options, failed = {}, set()

    def search(index, current):
        if index == len(styles):
            return []
        key = index, current
        if key in failed:
            return None
        s = styles[index]
        if key not in options:
            candidates = []
            for lo in range(current+1):
                for hi in range(max(current, lo+s['minimum']-1), width):
                    low = lo-int(lo > 0 and any(x == 0 for _, x in s['left']))
                    high = hi+int(hi < width-1 and any(x == 0 for _, x in s['right']))
                    target = high+s['n']+1 if s['sign'] == 1 else low-s['n']-1
                    if 0 <= target < width:
                        candidates.append((lo, hi, target))
            if candidates:
                candidates = [candidates[int(j)] for j in rng.permutation(len(candidates))]
                if rng.random() < .45:
                    candidates.sort(key=lambda z: not (z[0] == 0 or z[1] == width-1))
            options[key] = candidates
        for candidate in options[key]:
            rest = search(index+1, candidate[2])
            if rest is not None:
                return [candidate]+rest
        failed.add(key)
        return None

    for initial in map(int, rng.permutation(np.arange(1, width-1))):
        result = search(0, initial)
        if result is not None:
            return initial, result
    return None


def _paint_plan(rng, styles, spans, initial, shape, palette, required):
    """Construct labels from explicit geometry; no call to solve or _decode."""
    field, solid, ink = palette
    h, w = shape
    minimum = 2+sum(s['n']+1+s['height']+s['lead']+s['trail'] for s in styles)
    slack = h-minimum
    allocation = rng.multinomial(slack, np.full(len(styles)+1, 1/(len(styles)+1)))
    base, holes, path = [np.zeros(shape, dtype=bool) for _ in range(3)]
    current, start, prior_bottom = initial, 0, -1
    expected_records = []
    for i, (s, (lo, hi, target)) in enumerate(zip(styles, spans)):
        turn = (1 if i == 0 else prior_bottom+1)+int(allocation[i])
        near = turn+s['n']+1
        top = near+s['lead']
        bottom = top+s['height']-1
        prior_bottom = bottom+s['trail']
        base[top:bottom+1, lo:hi+1] = True
        for y, x in s['left']:
            holes[top+y, lo+x] = True
        for y, x in s['right']:
            holes[top+y, hi-x] = True
        path[start:turn+1, current] = True
        path[turn, min(current, target):max(current, target)+1] = True
        current, start = target, turn
        expected_records.append((top, bottom, lo, hi))
    path[start:, current] = True
    # The constructor uses literal neighborhoods; solve uses mask dilation.
    halo = np.zeros(shape, dtype=bool)
    for y, x in np.argwhere(holes):
        halo[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)] = True
    halo[holes] = False
    observed_solid = base & ~holes
    adjacent_solid = np.zeros(shape, dtype=bool)
    adjacent_solid[1:] |= observed_solid[:-1]
    adjacent_solid[:-1] |= observed_solid[1:]
    adjacent_solid[:, 1:] |= observed_solid[:, :-1]
    adjacent_solid[:, :-1] |= observed_solid[:, 1:]
    allowed = ~base & ~adjacent_solid
    allowed[0, initial] = False
    allowed[[0, -1], 0] = False
    allowed[[0, -1], -1] = False
    extras = np.zeros(shape, dtype=bool)

    def place(available, compulsory=False):
        cells = np.argwhere(allowed & available & ~extras)
        if not len(cells):
            if compulsory:
                raise ValueError("required extra-speck teaching cell is unavailable")
            return
        for raw in rng.permutation(len(cells)):
            y, x = cells[int(raw)]
            extras[y, x] = True
            if max(map(len, _components(extras)), default=0) <= 2:
                return
            extras[y, x] = False
        if compulsory:
            raise ValueError("extra clusters cannot be kept small")

    if 'far' in required:
        place(~halo & ~path, True)
    if 'near' in required:
        place(halo, True)
    if 'route' in required:
        place(path & ~halo, True)
    for _ in range(int(rng.integers(0, 5))):
        place(np.ones(shape, dtype=bool))
    inp = np.where(base, solid, field).astype(int)
    inp[holes], inp[extras] = field, solid
    inp[0, initial] = ink
    out = np.where(base, solid, field).astype(int)
    out[holes] = field
    out[halo | path] = ink
    return {'input': inp, 'output': out}, base, holes, expected_records


def _make_pair(rng, palette, direction, commands, anchor=None):
    # Every retry changes motifs and geometry, retaining the requested teaching role.
    for _ in range(96):
        styles = []
        compact = len(commands) == 3
        for j, (n, lower, sign) in enumerate(commands):
            style = 'merge' if anchor == 'merge' and j == 0 else (
                'competition' if anchor == 'competition' and j == 0 else 'random')
            styles.append(_obstacle_style(rng, n, lower, sign, style, compact))
        minimum = 2+sum(s['n']+1+s['height']+s['lead']+s['trail'] for s in styles)
        if minimum > 30:
            continue
        width = int(rng.integers(12, 31))
        plan = _lateral_plan(rng, width, styles)
        if plan is None and width != 30:
            width, plan = 30, _lateral_plan(rng, 30, styles)
        if plan is None:
            continue
        initial, spans = plan
        height = int(rng.integers(max(8, minimum), 31))
        if rng.random() < .15 and width >= minimum:
            height = width
        required = {'far', 'route'} if anchor == 'merge' else {'near', 'far'} if anchor == 'competition' else set()
        try:
            pair, base, holes, boxes = _paint_plan(
                rng, styles, spans, initial, (height, width), palette, required)
            physical_pair = {k: _physical(v, direction) for k, v in pair.items()}
            predicted, f = _decode(physical_pair['input'])
            if not (np.array_equal(f['base'], base) and np.array_equal(f['holes'], holes)
                    and [r['rectangle'] for r in f['records']] == boxes
                    and np.array_equal(predicted, physical_pair['output'])):
                continue
            if anchor == 'competition' and not any(r['raw_side_wrong'] for r in f['records']):
                continue
            return physical_pair
        except ValueError:
            continue
    raise RuntimeError("failed to find feasible warning-island geometry")


def _check_counts(num_train, num_test):
    for name, n, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                or not low <= int(n) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def _puzzle_key(p):
    return tuple(tuple((_key(pair['input']), _key(pair['output'])) for pair in p[s])
                 for s in ('train', 'test'))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = int(num_train), int(num_test)
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    varying_palette = bool(rng.integers(2))
    maximum = int(rng.choice((2, 3)))
    shift = int(rng.integers(2))  # new direction, or longer sequence
    directions = list(map(int, rng.permutation(4)))
    first_side = int(rng.choice((-1, 1)))
    train_dirs = directions[:3] if shift == 0 else directions
    test_dirs = directions[3:] if shift == 0 else directions
    result, seen = {'train': [], 'test': []}, set()

    def add(split, commands, anchor=None):
        for _ in range(16):
            colors = tuple(map(int, rng.choice(10, 3, replace=False))) if varying_palette else palette
            direction = int(rng.choice(train_dirs if split == 'train' else test_dirs))
            pair = _make_pair(rng, colors, direction, commands, anchor)
            signature = _key(pair['input'])
            if signature not in seen:
                result[split].append(pair)
                seen.add(signature)
                return
        raise RuntimeError("could not construct distinct inputs")

    add('train', [(1, 0, first_side)], 'merge')
    add('train', [(maximum, 1, -first_side),
                  (2 if maximum == 3 else 1, 0, -first_side)], 'competition')

    def random_commands(count, cap):
        ns = list(map(int, rng.integers(1, cap+1, size=count)))
        # Three obstacles use compact packets; this is geometric feasibility,
        # not a different routing rule or a split-specific color convention.
        return [(n, int(rng.integers(n)), int(rng.choice((-1, 1)))) for n in ns]

    for _ in range(num_train-2):
        count = int(rng.integers(1, 3 if shift == 1 else 4))
        add('train', random_commands(count, min(maximum, 2) if count == 3 else maximum))
    for _ in range(num_test):
        count = 3 if shift == 1 else int(rng.integers(1, 4))
        add('test', random_commands(count, min(maximum, 2) if count == 3 else maximum))
    for split in result:
        result[split] = [result[split][int(i)] for i in rng.permutation(len(result[split]))]
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    """Check pixel-derived family invariants and demonstration coverage."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            predicted, f = _decode(inp)
            assert np.array_equal(predicted, out), 'output violates warning-island routing'
            canonical = _forward(out, f['direction'])
            field, solid, ink = f['palette']
            assert np.all(canonical[f['holes']] == field), 'missing centers must remain empty'
            assert np.all(canonical[f['halos']] == ink), 'all eight-neighbor warnings must be painted'
            assert not (f['path'] & (f['base'] | _dilate(f['holes']))).any()
            assert np.all(canonical[f['extras'] & ~f['halos'] & ~f['path']] == field)
            assert np.all(canonical[f['base'] & ~f['holes'] & ~f['halos']] == solid)
            assert len(_components(f['path'])) == 1
            degree = np.zeros_like(f['path'], dtype=int)
            degree[1:] += f['path'][:-1]
            degree[:-1] += f['path'][1:]
            degree[:, 1:] += f['path'][:, :-1]
            degree[:, :-1] += f['path'][:, 1:]
            assert np.count_nonzero(f['path'] & (degree == 1)) == 2
            assert np.all((degree[f['path']] == 1) | (degree[f['path']] == 2))
            records[split].append(f)
    train, test = records['train'], records['test']
    tr = [r for f in train for r in f['records']]
    te = [r for f in test for r in f['records']]
    assert {r['side'] for r in tr} == {-1, 1}, 'demonstrate both bypass ends'
    assert {r['clearance'] for r in te} <= {r['clearance'] for r in tr}
    assert len({r['clearance'] for r in tr}) >= 2
    assert any(r['raw_side_wrong'] for r in tr), 'raw hole totals must choose a wrong side'
    assert any(r['hole_groups'] > sum(r['counts']) for r in tr), 'teach fusion of disconnected holes'
    assert any(r['halo_groups4'] > sum(r['counts']) for r in tr), 'teach diagonal warning connectivity'
    assert any(r['leading_spill'] for r in tr), 'teach the halo-expanded envelope'
    assert any(f['near_extra'] for f in train) and any(f['far_erasure'] for f in train)
    assert any(f['route_extra'] for f in train), 'specks must not become route obstacles'
    assert any(f['repeated'] for f in train), 'avoid an automatic zigzag shortcut'
    # One deliberate, visible shift occurs in every generated puzzle.
    new_direction = {f['direction'] for f in test} - {f['direction'] for f in train}
    longer = min(len(f['records']) for f in test) > max(len(f['records']) for f in train)
    assert new_direction or longer, 'expected a direction or obstacle-sequence shift'


def _literal_regression():
    """A hand-specified competition: two singles beat four holes in one island."""
    a = np.zeros((17, 22), dtype=int)
    a[0, 7] = 8
    a[5:10, 3:15] = 2
    missing = [(6, 3), (5, 7), (6, 14), (7, 14), (7, 13), (8, 13)]
    for p in missing:
        a[p] = 0
    a[2, 18] = 2
    a[7, 15] = 2  # adjacent extra is erased, then included in a warning halo
    b = np.zeros_like(a)
    b[5:10, 3:15] = 2
    for y, x in missing:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                b[y+dy, x+dx] = 8
    for p in missing:
        b[p] = 0
    # Leading halo row is 4, winning count 2 -> turn row 1.
    # Rival halo extends to column 15 -> outgoing column 18.
    b[0:2, 7] = 8
    b[1, 7:19] = 8
    b[1:, 18] = 8
    assert np.array_equal(solve(a), b)
    for k in range(4):
        for mirror in (False, True):
            x = np.rot90(np.fliplr(a) if mirror else a, k).copy()
            y = np.rot90(np.fliplr(b) if mirror else b, k).copy()
            assert np.array_equal(solve(x), y)
            recolor = np.array([6, 0, 9, 2, 3, 4, 5, 7, 1, 8])
            assert np.array_equal(solve(recolor[x]), recolor[y])
    # Two warning squares touch only at a diagonal, so eight-connectivity matters.
    a = np.zeros((15, 20), dtype=int)
    a[0, 8] = 6
    a[4:9, 4:16] = 3
    a[5, 4] = a[8, 7] = 0
    b = np.zeros_like(a)
    b[4:9, 4:16] = 3
    b[4:7, 3:6] = b[7:10, 6:9] = 6
    b[5, 4] = b[8, 7] = 0
    b[:3, 8] = b[2, 8:18] = b[2:, 17] = 6
    prediction, f = _decode(a)
    assert f['records'][0]['counts'] == (1, 0)
    assert f['records'][0]['halo_groups4'] == 2
    assert np.array_equal(prediction, b)
    # Equal end counts and unsupported attached noise are explicitly rejected.
    tied = np.zeros((15, 20), dtype=int)
    tied[0, 8] = 6
    tied[4:9, 4:16] = 3
    tied[6, 4] = tied[6, 15] = 0
    attached = a.copy()
    attached[6, 16] = 3
    no_exit = np.zeros((15, 20), dtype=int)
    no_exit[0, 8] = 6
    no_exit[4:9, 4:] = 3
    no_exit[6, 4] = 0  # encoded right exit is outside; left exit is not substituted
    stranded = np.zeros((15, 20), dtype=int)
    stranded[0, 8] = 6
    stranded[4:9, 4:16] = 3
    for p in ((5, 6), (6, 5), (6, 7), (7, 6)):
        stranded[p] = 0  # leaves a detached solid pixel at (6, 6)
    for invalid in (tied, attached, no_exit, stranded):
        try:
            solve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('ambiguous or unsupported corruption accepted')


def self_test():
    """Bounded 42-seed suite, exact regressions, count checks, and mutations."""
    _literal_regression()
    signatures, shapes, structures, palettes = set(), set(), set(), set()
    directions, counts, packets, transitions, flags = set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        if seed < 2:
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        for split in puzzle:
            for pair in puzzle[split]:
                _, f = _decode(pair['input'])
                shapes.add(f['shape'])
                palettes.add(f['palette'])
                normalized = np.empty_like(pair['input'])
                for i, c in enumerate(f['palette']):
                    normalized[pair['input'] == c] = i
                structures.add(_key(normalized))
                directions.add(f['direction'])
                counts.add(len(f['records']))
                packets.update(max(r['counts']) for r in f['records'])
                signs = [r['side'] for r in f['records']]
                transitions.update(zip(signs, signs[1:]))
                flags.update(k for k in ('clipped', 'near_extra', 'far_erasure', 'route_extra') if f[k])
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
        if seed < 3:
            for pair in puzzle['test']:
                assert np.array_equal(solve(np.rot90(pair['input'])), np.rot90(pair['output']))
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 45 and len(palettes) >= 35
    assert directions == {0, 1, 2, 3} and counts == {1, 2, 3}
    assert packets == {1, 2, 3}
    assert transitions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    assert flags == {'clipped', 'near_extra', 'far_erasure', 'route_extra'}
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(91, tr, te)
            assert len(p['train']) == tr and len(p['test']) == te
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    bad = generate_puzzle(3, 2, 1)
    p = bad['test'][0]
    _, f = _decode(p['input'])
    canonical = _forward(p['output'], f['direction'])
    y, x = np.argwhere(f['path'])[1]
    canonical[y, x] = f['palette'][0]
    p['output'] = _physical(canonical, f['direction'])
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('a missing route cell escaped validation')
    return {'seeds': 42, 'distinct_puzzles': len(signatures),
            'shapes': len(shapes), 'color_normalized_inputs': len(structures),
            'palettes': len(palettes)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
