"""Harvest tokens reachable through a funnel's unpainted, shadowed light field.

The center of the eight-slot collector is a permanent token-color legend.
Only input geometry and colors are used by solve. No parent code is imported.
All experiments are in memory; this module writes no files or execution traces.
"""
from collections import deque
import hashlib
import numpy as np


ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test) and
            2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("Require integer counts 2..10 / 1..4, with num_train > num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Noninteger grid")
    _require(bool(np.all((a >= 0) & (a <= 9))), "Invalid color")
    return a


def _signature(a):
    return (a.shape, a.astype(np.int8).tobytes())


def _collector(a):
    """The slot color occurs exactly eight times, in one marked 3x3 square."""
    candidates = []
    for color in np.unique(a):
        points = np.argwhere(a == color)
        if len(points) != 8:
            continue
        r, c = map(int, points.min(axis=0))
        if tuple(points.max(axis=0)) != (r + 2, c + 2):
            continue
        patch = a[r:r+3, c:c+3]
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 1] = False
        if np.all(patch[mask] == color) and patch[1, 1] != color:
            candidates.append((r, c, int(color), int(patch[1, 1])))
    _require(len(candidates) == 1, "Need one marked eight-slot collector")
    return candidates[0]


def _components(mask, diagonal=False):
    cells = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c) if diagonal else ORTHOGONAL
    while cells:
        start = min(cells)
        cells.remove(start)
        group = [start]
        for r, c in group:
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in cells:
                    cells.remove(q)
                    group.append(q)
        result.append(group)
    return result


def _fit_flank(values, sign):
    """Fit the longest terminal stair; each period has at least two rises."""
    values = np.asarray(values, dtype=int)
    candidates = []
    for period in (1, 2):
        for phase in range(period):
            t = np.arange(len(values)) - len(values) + 1
            predicted = values[-1] + sign * ((t + phase) // period)
            length = 0
            for same in (values == predicted)[::-1]:
                if not same:
                    break
                length += 1
            if length >= 2 * period + 1 and abs(int(values[-1] - values[-length])) >= 2:
                candidates.append((length, -period, phase))
    _require(bool(candidates), "Unreadable terminal stair")
    length, negative_period, phase = max(candidates)
    return -negative_period, phase, length


def _analyze(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel().astype(int), minlength=10)
    background = int(np.argmax(counts))
    _require(counts[background] > a.size // 2, "Background must be a strict majority")
    cr, cc, slots, token = _collector(a)
    roles = set(map(int, np.unique(a))) - {background, slots, token}
    _require(len(roles) == 1 and len({background, slots, token}) == 3, "Need four distinct color roles")
    wall = roles.pop()
    matches = []
    for rotation in range(4):
        g = np.rot90(a, -rotation)
        r, c, _, _ = _collector(g)
        center, rear = r + 1, c - 1
        if rear < 0:
            continue
        for group in _components(g == wall, diagonal=True):
            points = np.asarray(group, dtype=int)
            if int(points[:, 1].min()) != rear:
                continue
            mouth = int(points[:, 1].max())
            if mouth - rear < 6:
                continue
            cap = sorted(y for y, x in group if x == rear)
            if not cap or cap != list(range(cap[0], cap[-1] + 1)) or not (cap[0] <= r-1 and cap[-1] >= r+3):
                continue
            upper, lower = [], []
            for x in range(rear + 1, mouth + 1):
                rows = sorted(y for y, xx in group if xx == x)
                if len(rows) != 2 or not (rows[0] < r and rows[1] > r+2):
                    # Beyond the collector, rails may also narrow to width three.
                    if len(rows) != 2 or x <= c+2 or not (rows[0] < center < rows[1]):
                        break
                upper.append(rows[0])
                lower.append(rows[1])
            if len(upper) != mouth - rear:
                continue
            if abs(upper[0] - cap[0]) > 1 or abs(lower[0] - cap[-1]) > 1:
                continue
            try:
                fits = (_fit_flank(upper, -1), _fit_flank(lower, 1))
            except AssertionError:
                continue
            footprint = np.zeros(g.shape, dtype=bool)
            bounds = []
            for x in range(rear + 1, g.shape[1]):
                limits = []
                for ys, fit, sign in zip((upper, lower), fits, (-1, 1)):
                    period, phase, _ = fit
                    y = ys[x-rear-1] if x <= mouth else ys[-1] + sign*((x-mouth+phase)//period)
                    limits.append(int(y))
                bounds.append(tuple(limits))
                footprint[:, x] = (np.arange(g.shape[0]) > limits[0]) & (np.arange(g.shape[0]) < limits[1])
            emitter = np.zeros(g.shape, dtype=bool)
            emitter[points[:, 0], points[:, 1]] = True
            if np.any(emitter & footprint):
                continue
            collector = np.zeros(g.shape, dtype=bool)
            collector[r:r+3, c:c+3] = True
            if not np.all(footprint[collector]):
                continue
            obstacles = (g == wall) & ~emitter
            light = footprint & ~(g == wall)
            # A blocker affects its lane only if its own cell is in the cone.
            for y in range(g.shape[0]):
                blocked = False
                for x in range(rear + 1, g.shape[1]):
                    if not footprint[y, x]:
                        continue
                    if obstacles[y, x]:
                        blocked = True
                    if blocked:
                        light[y, x] = False
            matches.append(dict(grid=g, rotation=rotation, background=background, wall=wall,
                                slots=slots, token=token, pos=(r, c), center=center,
                                rear=rear, mouth=mouth, upper=tuple(upper), lower=tuple(lower),
                                fits=fits, bounds=tuple(bounds), emitter=emitter,
                                obstacles=obstacles, footprint=footprint, light=light,
                                collector=collector))
    _require(len(matches) == 1, "Need one unambiguous directional emitter")
    return matches[0]


def _reachable(allowed, starts):
    reached = np.zeros(allowed.shape, dtype=bool)
    queue = deque()
    for y, x in np.argwhere(starts & allowed):
        reached[y, x] = True
        queue.append((int(y), int(x)))
    h, w = allowed.shape
    while queue:
        y, x = queue.popleft()
        for dy, dx in ORTHOGONAL:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and allowed[yy, xx] and not reached[yy, xx]:
                reached[yy, xx] = True
                queue.append((yy, xx))
    return reached


def _wave_reachable(allowed, starts):
    """Array wavefront, separate from solve's queue implementation."""
    seen = starts & allowed
    for _ in range(allowed.size):
        spread = seen.copy()
        spread[1:] |= seen[:-1]
        spread[:-1] |= seen[1:]
        spread[:, 1:] |= seen[:, :-1]
        spread[:, :-1] |= seen[:, 1:]
        new = spread & allowed
        if np.array_equal(seen, new):
            return seen
        seen = new
    raise AssertionError("Wavefront did not converge")


def _write_result(g, selected, pos, background, token):
    n = int(selected.sum())
    _require(n <= 8, "This family has no excess reachable supply")
    result = g.copy()
    result[selected] = background
    r, c = pos
    slots = [(r+i, c+j) for i in range(3) for j in range(3) if (i, j) != (1, 1)]
    for y, x in slots[:n]:
        result[y, x] = token
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only: extrapolate light, remove shadows, trace access, count/pack."""
    a = _analyze(input_grid)
    reached = _reachable(a['light'], a['collector'])
    selected = reached & (a['grid'] == a['token']) & ~a['collector']
    # Rotate the selected MASK back, then serialize once in displayed order.
    displayed_selection = np.rot90(selected, a['rotation'])
    r, c, _, _ = _collector(np.asarray(input_grid))
    return _write_result(np.asarray(input_grid), displayed_selection, (r, c), a['background'], a['token'])


def _geometry(rng, test):
    """Compose a rear socket, optional bulge/neck, and independent thin flanks."""
    rear = int(rng.integers(0, 4))
    length = int(rng.integers(3, 7))
    periods = [int(rng.integers(1, 3)) for _ in range(2)]
    phases = [int(rng.integers(p)) for p in periods]
    flare = 2 * max(periods) + 1
    profiles = []
    for p, phase in zip(periods, phases):
        initial = int(rng.integers(2, 4))
        ys = [initial] * 3
        while len(ys) < length:
            ys.append(max(2, min(4, ys[-1] + int(rng.integers(-1, 2)))))
        profiles.append(ys + [ys[-1] + (t+phase)//p for t in range(1, flare+1)])
    mouth = rear + length + flare
    maximum = min(14 if test else 10, 29 - mouth)
    minimum = 11 if test else 6
    if maximum < minimum:
        return None
    width = mouth + 1 + int(rng.integers(minimum, maximum+1))
    span = max(profiles[0]) + max(profiles[1]) + 1
    if span + 6 > 30:
        return None
    height = int(rng.integers(span + 6, 31))
    margin = int(rng.integers(3, height-span-2))
    center = max(profiles[0]) + margin
    upper = [center - n for n in profiles[0]]
    lower = [center + n for n in profiles[1]]
    wall = np.zeros((height, width), dtype=bool)
    wall[upper[0]:lower[0]+1, rear] = True
    footprint = np.zeros_like(wall)
    for x in range(rear + 1, width):
        if x <= mouth:
            u, l = upper[x-rear-1], lower[x-rear-1]
            wall[u, x] = wall[l, x] = True
        else:
            dt = x - (rear + length)
            u = center - profiles[0][length-1] - (dt+phases[0])//periods[0]
            l = center + profiles[1][length-1] + (dt+phases[1])//periods[1]
        footprint[max(0, u+1):min(height, l), x] = True
    pos = (center-1, rear+1)
    collector = np.zeros_like(wall)
    r, c = pos
    collector[r:r+3, c:c+3] = True
    return wall, footprint, collector, pos, mouth, (upper[-1], lower[-1])


def _optics(footprint, obstacles, wall):
    shadow = np.zeros(footprint.shape, dtype=bool)
    for y, x in np.argwhere(footprint & obstacles):
        shadow[y, x:] = True
    light = footprint & ~wall & ~shadow
    return light, shadow & footprint


def _choose(rng, mask, count):
    points = np.argwhere(mask)
    _require(len(points) >= count, "Insufficient candidate cells")
    return [tuple(map(int, p)) for p in points[rng.choice(len(points), size=count, replace=False)]]


def _build_pair(rng, colors, rotation, kind, load, test):
    """Bounded rejection; labels use constructed masks and an array wavefront."""
    background, wall_color, slot_color, token = map(int, colors)
    for _ in range(100):
        geometry = _geometry(rng, test)
        if geometry is None:
            continue
        emitter, footprint, collector, pos, mouth, mouth_bounds = geometry
        h, w = emitter.shape
        r, c = pos
        center = r + 1
        obstacles = np.zeros_like(emitter)
        if kind in ('island', 'combined'):
            options = []
            for y in range(h):
                xs = np.flatnonzero(footprint[y])
                if len(xs) < 3:
                    continue
                x = int(xs[0])
                outward = -1 if y < center else 1
                yy = y + outward
                if (mouth+2 <= x <= w-3 and 0 <= yy < h and
                        np.any(footprint[yy, x+1:])):
                    options.append((y, x, outward))
            if not options:
                continue
            first = options[int(rng.integers(len(options)))]
            gates = [first]
            opposite = [q for q in options if q[2] != first[2]]
            if opposite and rng.random() < .45:
                gates.append(opposite[int(rng.integers(len(opposite)))])
            for y, x, outward in gates:
                # The outermost blocker starts where this lane first lights.
                # Axial bars repeat one shadow; inward bars widen it by cells.
                axial = int(rng.integers(1, 4))
                transverse = int(rng.integers(1, 3))
                for dx in range(axial):
                    for dy in range(transverse):
                        yy = y - outward*dy
                        if 0 <= yy < h and x+dx < w:
                            obstacles[yy, x+dx] = True
        if kind in ('shadow', 'combined', 'island'):
            y = center + int(rng.integers(-1, 2))
            x = int(rng.integers(mouth+2, w-2))
            transverse = int(rng.integers(1, 3))
            axial = int(rng.integers(1, 3))
            obstacles[y:y+transverse, x:x+axial] = True
        # Optional inert blockers outside the prospective beam. These do not
        # acquire a shadow when their lane eventually enters the widening cone.
        if rng.random() < .5:
            outside = ~footprint & ~emitter
            outside[:, :mouth+2] = False
            if outside.any():
                y, x = _choose(rng, outside, 1)[0]
                obstacles[y, x] = True
        wall = emitter | obstacles
        light, shadow = _optics(footprint, obstacles, wall)
        reachable = _wave_reachable(light, collector)
        active = reachable & ~collector
        island = light & ~reachable
        dark = shadow & ~wall
        outside = ~footprint & ~wall & ~collector
        if not outside.any() or (kind != 'clean' and not dark.any()):
            continue
        if kind in ('island', 'combined') and not island.any():
            continue
        # Witness extrapolation: a harvested token outside the mouth's fixed strip.
        wide = active.copy()
        wide[mouth_bounds[0]+1:mouth_bounds[1], :] = False
        wide[:, :mouth+1] = False
        if load and not wide.any():
            continue
        chosen = _choose(rng, wide, 1) if load else []
        # Teach straight shadows with an adjacent unshadowed downstream token.
        if kind == 'shadow' and load >= 2:
            adjacent = np.zeros_like(active)
            adjacent[1:] |= dark[:-1]
            adjacent[:-1] |= dark[1:]
            adjacent &= active
            adjacent[:, :mouth+4] = False
            for point in chosen:
                adjacent[point] = False
            if not adjacent.any():
                continue
            chosen += _choose(rng, adjacent, 1)
        # A connected pair demonstrates cell counting and transparent payload.
        if kind in ('island', 'combined') and load >= 5:
            starts = active[:, :-1] & active[:, 1:]
            starts[:, :mouth+1] = False
            for y, x in chosen:
                starts[y, max(0, x-1):min(starts.shape[1], x+1)] = False
            if not starts.any():
                continue
            y, x = _choose(rng, starts, 1)[0]
            chosen += [(y, x), (y, x+1)]
        remaining = active.copy()
        for point in chosen:
            remaining[point] = False
        if int(remaining.sum()) < load-len(chosen):
            continue
        chosen += _choose(rng, remaining, load-len(chosen))
        inert = _choose(rng, outside, int(rng.integers(1, min(4, int(outside.sum()))+1)))
        if kind != 'clean':
            inert += _choose(rng, dark, int(rng.integers(1, min(4, int(dark.sum()))+1)))
        if kind in ('island', 'combined'):
            inert += _choose(rng, island, int(rng.integers(1, min(4, int(island.sum()))+1)))
        g = np.full((h, w), background, dtype=np.int8)
        g[wall] = wall_color
        for point in chosen + inert:
            g[point] = token
        g[r:r+3, c:c+3] = slot_color
        g[r+1, c+1] = token
        selected = np.zeros_like(wall)
        for point in chosen:
            selected[point] = True
        # Rotate input and selected cells FIRST; pack in displayed row-major order.
        inp = np.rot90(g, rotation).copy()
        picked = np.rot90(selected, rotation)
        rr, cc, _, _ = _collector(inp)
        out = _write_result(inp, picked, (rr, cc), background, token)
        try:
            _analyze(inp)
        except AssertionError:
            continue
        if np.array_equal(solve(inp), out):
            return {'input': inp, 'output': out}
    raise RuntimeError("Bounded pair construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shared_palette = bool(rng.random() < .3)
    colors = rng.choice(10, 4, replace=False)
    preferred_rotation = int(rng.integers(4))
    shared_direction = bool(rng.random() < .2)
    requests = [('shadow', int(rng.choice([2, 4]))),
                ('island', int(rng.integers(5, 8)))]
    def extra():
        kind = str(rng.choice(['clean', 'shadow', 'island', 'combined']))
        load = 0 if rng.random() < .06 else int(rng.integers(1, 9))
        return kind, load
    requests.extend(extra() for _ in range(num_train-2))
    requests = [requests[int(i)] for i in rng.permutation(num_train)]
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, specs in (('train', requests), ('test', [extra() for _ in range(num_test)])):
        for kind, load in specs:
            for _ in range(8):
                palette = colors if shared_palette else rng.choice(10, 4, replace=False)
                rotation = preferred_rotation if shared_direction else int(rng.integers(4))
                pair = _build_pair(rng, palette, rotation, kind, load, split == 'test')
                key = _signature(pair['input'])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("Bounded distinct-input construction exhausted")
    return puzzle


def pair_features(input_grid):
    a = _analyze(input_grid)
    reached = _reachable(a['light'], a['collector'])
    field_tokens = (a['grid'] == a['token']) & ~a['collector']
    selected = field_tokens & reached
    no_optics = _reachable(a['grid'] != a['wall'], a['collector']) & field_tokens
    no_shadows = _reachable(a['footprint'] & (a['grid'] != a['wall']), a['collector']) & field_tokens
    illumination_only = field_tokens & a['light']
    u, l = a['upper'][-1], a['lower'][-1]
    strip = np.zeros(a['grid'].shape, dtype=bool)
    strip[u+1:l, a['mouth']+1:] = True
    wide = selected & ~strip
    wide[:, :a['mouth']+1] = False
    outline = a['emitter'][:, a['rear']:a['mouth']+1]
    ys = np.flatnonzero(outline.any(axis=1))
    outline = outline[ys[0]:ys[-1]+1]
    return dict(shape=tuple(np.asarray(input_grid).shape), rotation=a['rotation'],
                palette=(a['background'], a['wall'], a['slots'], a['token']),
                periods=tuple(f[0] for f in a['fits']),
                outline=(outline.shape, outline.tobytes()),
                projection=a['grid'].shape[1]-a['mouth']-1,
                selected=int(selected.sum()), shadowed=int((field_tokens & a['footprint'] & ~a['light']).sum()),
                stranded=int((field_tokens & a['light'] & ~reached).sum()),
                exterior=int((field_tokens & ~a['footprint']).sum()),
                wide_selected=int(wide.sum()), blockers=int(a['obstacles'].sum()),
                light_components=len(_components(a['light'])),
                no_A_changes=not np.array_equal(selected, no_optics),
                no_shadow_changes=not np.array_equal(selected, no_shadows),
                no_B_access_changes=not np.array_equal(selected, illumination_only))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Invalid splits")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "Size changed")
            sig = _signature(inp)
            _require(sig not in seen, "Duplicate input")
            seen.add(sig)
            a = _analyze(inp)
            reached = _reachable(a['light'], a['collector'])
            _require(np.array_equal(reached, _wave_reachable(a['light'], a['collector'])), "Access mismatch")
            tokens = (a['grid'] == a['token']) & ~a['collector']
            selected = tokens & reached
            n = int(selected.sum())
            _require(0 <= n <= 8, "Counter overflow")
            _require(np.array_equal(solve(inp), out), "Wrong label")
            changed = inp != out
            r, c, slot, token = _collector(inp)
            cm = np.zeros(inp.shape, dtype=bool)
            cm[r:r+3, c:c+3] = True
            _require(out[r+1, c+1] == token, "Legend changed")
            _require(int((changed & ~cm).sum()) == n, "Incorrect source-cell count")
            _require(int((changed & cm).sum()) == n, "Incorrect unary prefix size")
            _require(bool(np.all(inp[changed & ~cm] == token)), "Erased non-token")
            _require(bool(np.all(out[changed & ~cm] == a['background'])), "Wrong erasure color")
            _require(np.array_equal(out[inp == a['wall']], inp[inp == a['wall']]), "Wall changed")
            _require(int((out == token).sum()) == int((inp == token).sum()), "Token conservation failed")
            f = pair_features(inp)
            _require(f['exterior'] > 0, "Missing cone-boundary witness")
            _require(n == 0 or f['wide_selected'] > 0, "Missing extrapolation witness")
            features[split].append(f)
    train = features['train']
    _require(any(f['selected'] in (2, 4) and f['shadowed'] and f['no_shadow_changes'] for f in train),
             "Need partial-prefix and shadow evidence")
    _require(any(5 <= f['selected'] <= 7 and f['stranded'] and f['no_B_access_changes'] for f in train),
             "Need illuminated-but-inaccessible evidence")
    _require(all(f['no_A_changes'] for f in train), "Training does not expose A")
    _require(min(f['projection'] for f in features['test']) > max(f['projection'] for f in train),
             "Missing extrapolation-length shift")
    return None


def self_test():
    """42 streaming seeds; no saved grids/logs, and bounded temporary arrays."""
    # Hand-constructed optics/access regression, independent of the sampler.
    # Period-one rails end at x=6. At x=7 the virtual upper boundary is y=1;
    # lane y=1 first lights at x=8, where a blocker seals off the y=0 wing.
    fixture = np.zeros((15, 17), dtype=np.int8)
    fixture[5:10, 0] = 1
    for x, upper in enumerate((5, 5, 5, 4, 3, 2), 1):
        fixture[upper, x] = fixture[14-upper, x] = 1
    fixture[6:9, 1:4] = 2
    fixture[7, 2] = 3
    for point in ((1, 8), (7, 9), (0, 7)):
        fixture[point] = 1
    picked = ((7, 8), (6, 12), (6, 13))
    for point in picked + ((0, 11), (7, 12), (1, 7), (14, 0)):
        fixture[point] = 3
    for rotation in range(4):
        inp = np.rot90(fixture, rotation).copy()
        expected_field = fixture.copy()
        for point in picked:
            expected_field[point] = 0
        expected = np.rot90(expected_field, rotation).copy()
        r, c, _, _ = _collector(inp)
        expected[r, c:c+3] = 3  # Exactly three displayed top-row slots.
        _require(np.array_equal(solve(inp), expected), "Shadow/island/boundary/order fixture failed")
    opened = fixture.copy()
    opened[1, 8] = 0
    expected = opened.copy()
    for point in picked + ((0, 11),):
        expected[point] = 0
    expected[6, 1:4] = 3
    expected[7, 1] = 3
    _require(np.array_equal(solve(opened), expected), "Outside blocker incorrectly shadows a future lane")
    signatures, shapes, outlines, palettes, periods = set(), set(), set(), set(), set()
    rotations, loads, components = set(), set(), set()
    flags = dict(shadow=0, island=0, clean=0, zero=0, full=0)
    pairs = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                f = pair_features(pair['input'])
                pairs += 1
                digest.update(repr(pair['input'].shape).encode())
                digest.update(pair['input'].tobytes())
                shapes.add(f['shape'])
                outlines.add(f['outline'])
                palettes.add(f['palette'])
                periods.add(f['periods'])
                rotations.add(f['rotation'])
                loads.add(f['selected'])
                components.add(f['light_components'])
                flags['shadow'] += f['shadowed'] > 0
                flags['island'] += f['stranded'] > 0
                flags['clean'] += f['blockers'] == 0
                flags['zero'] += f['selected'] == 0
                flags['full'] += f['selected'] == 8
                before = pair['input'].copy()
                solve(pair['input'])
                _require(np.array_equal(pair['input'], before), "solve mutates input")
        signatures.add(digest.digest())
    _require(len(signatures) == 42, "Seed diversity collapsed")
    _require(len(shapes) >= 50 and len(outlines) >= 100, "Structural diversity collapsed")
    _require(len(palettes) >= 80 and len(periods) == 4 and rotations == set(range(4)), "Role/geometry coverage collapsed")
    _require(loads == set(range(9)), "Missing load including zero/full")
    _require(len(components) >= 3 and min(flags.values()) >= 1, "Missing topology/regime")
    p = generate_puzzle(0)
    q = generate_puzzle(0)
    _require(all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a), "Nondeterminism")
    # Recoloring every role must commute with the input-only transformation.
    permutation = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
    for pair in p['train'] + p['test']:
        _require(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]), "Palette-equivariance failed")
    # Returned-cell corruption must be detected.
    # The two required demonstrations are positive; a test may validly be zero.
    p['train'][0]['output'] = p['train'][0]['input'].copy()
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupt output accepted")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid counts accepted")
    return dict(seeds=42, pairs=pairs, shapes=len(shapes), outlines=len(outlines),
                palettes=len(palettes), period_pairs=len(periods),
                rotations=sorted(rotations), loads=sorted(loads),
                light_component_counts=sorted(components), regimes=flags)
