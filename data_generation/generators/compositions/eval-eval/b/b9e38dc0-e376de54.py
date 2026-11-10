"""Spotlight-selected orchestra: light selects an ordered length-reference group.

The sampler draws labels from explicit geometry. The public solver reconstructs
geometry, membership and lengths from the input alone. No parent code is imported.
"""

from itertools import combinations
import numpy as np


CAMPAIGN_SEEDS = (0, 963217233, 7638119, 981180752, 909080058, 185087629,
                 265149187, 613549431, 460034477, 309165895, 154414292,
                 922531068, 423937291, 65252652, 908984302, 454243451,
                 751970817, 694918742, 24196373, 484074837, 99064573,
                 508360906, 170363666, 470499147, 155583031, 632030142,
                 467200679, 723586322, 351687319, 203196083, 332025263,
                 985714854, 790507664, 497689134, 507516917, 323431891,
                 915065911, 270015443, 534298398, 387098480, 169269066,
                 534210160)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        queue, group = [first], []
        while queue:
            y, x = queue.pop()
            group.append((int(y), int(x)))
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if p in unseen:
                    unseen.remove(p)
                    queue.append(p)
        result.append(group)
    return result


def _terminal_fit(values, sign):
    """Fit the longest terminal thin stair with two complete outward steps."""
    values = np.array(values, dtype=int)
    candidates = []
    distance = np.arange(len(values)) - len(values) + 1
    for period in (1, 2, 3):
        for phase in range(period):
            predicted = values[-1] + sign * ((distance + phase) // period)
            mismatches = np.flatnonzero(values != predicted)
            start = int(mismatches[-1]) + 1 if len(mismatches) else 0
            length = len(values) - start
            if length >= 2 * period + 1:
                candidates.append((length, -period, phase))
    if not candidates:
        raise ValueError("terminal rails do not expose a periodic continuation")
    length, negative_period, phase = max(candidates)
    return -negative_period, phase, int(values[-1]), length


def _parse_canonical(a, bg, wall, rotation):
    coords = np.argwhere(a == wall)
    if len(coords) < 9:
        raise ValueError("not a chamber wall")
    rear, mouth = int(coords[:, 1].min()), int(coords[:, 1].max())
    cap = np.flatnonzero(a[:, rear] == wall)
    if len(cap) < 3 or np.any(np.diff(cap) != 1) or mouth-rear < 4:
        raise ValueError("no closed back")
    upper, lower = [], []
    for x in range(rear+1, mouth+1):
        ys = np.flatnonzero(a[:, x] == wall)
        if len(ys) != 2 or ys[1]-ys[0] < 2:
            raise ValueError("need two thin rails per section")
        upper.append(int(ys[0]))
        lower.append(int(ys[1]))
    if cap[0] != upper[0] or cap[-1] != lower[0]:
        raise ValueError("cap and rails disagree")
    if any(abs(v) > 1 for v in np.diff(upper)) or any(abs(v) > 1 for v in np.diff(lower)):
        raise ValueError("disconnected wall")
    fits = (_terminal_fit(upper, -1), _terminal_fit(lower, 1))
    footprint = np.zeros(a.shape, dtype=bool)
    ys = np.arange(a.shape[0])
    for x in range(rear+1, a.shape[1]):
        if x <= mouth:
            lo, hi = upper[x-rear-1], lower[x-rear-1]
        else:
            lo = fits[0][2] - (x-mouth+fits[0][1]) // fits[0][0]
            hi = fits[1][2] + (x-mouth+fits[1][1]) // fits[1][0]
        footprint[:, x] = (ys > lo) & (ys < hi)
    chamber = footprint.copy()
    chamber[:, mouth+1:] = False
    colors = np.unique(a[chamber & (a != bg)])
    if len(colors) != 1 or colors[0] == wall:
        raise ValueError("chamber must contain one payload color")
    payload = int(colors[0])
    patch = a == payload
    if np.any(patch & ~chamber) or len(_components(patch)) != 1:
        raise ValueError("payload must be one connected interior patch")
    material = (a != bg) & (a != wall) & (a != payload)
    groups = _components(material)
    if not groups:
        raise ValueError("no receiver array")
    baseline = max(min(x for y, x in group) for group in groups)
    if baseline < mouth+3:
        raise ValueError("receiver baseline too close to mouth")
    rods, obstacles = [], np.zeros(a.shape, dtype=bool)
    for group in groups:
        start = min(x for y, x in group)
        if start == baseline:
            rows = {y for y, x in group}
            cols = sorted(x for y, x in group)
            shades = {int(a[y, x]) for y, x in group}
            if (len(rows) != 1 or len(shades) != 1
                    or cols != list(range(baseline, baseline+len(group)))):
                raise ValueError("receiver must be a monochrome straight stroke")
            rods.append((rows.pop(), len(group), shades.pop()))
        else:
            if any(x <= mouth or x > baseline-2 for y, x in group):
                raise ValueError("opaque objects belong strictly between mouth and receivers")
            for y, x in group:
                obstacles[y, x] = True
    rods.sort()
    if not 3 <= len(rods) <= 13 or min(np.diff([r[0] for r in rods])) < 2:
        raise ValueError("need a separated ordered receiver array")
    active = []
    for i, (y, length, color) in enumerate(rods):
        if footprint[y, baseline] and not np.any(obstacles[y, :baseline] & footprint[y, :baseline]):
            active.append(i)
    if len(active) < 3 or len(active) % 2 == 0:
        raise ValueError("illuminated anchors must form an odd group of at least three")
    reference = active[len(active)//2]
    return dict(a=a, rotation=rotation, bg=bg, wall=wall, payload=payload,
                rear=rear, mouth=mouth, upper=upper, lower=lower, fits=fits,
                footprint=footprint, obstacles=obstacles, baseline=baseline,
                rods=rods, active=active, reference=reference)


def _inspect(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= a.size//2:
        raise ValueError("background must occupy more than half the input")
    matches = []
    for rotation in range(4):
        canonical = np.rot90(a, -rotation)
        for wall in np.unique(a):
            if wall == bg:
                continue
            try:
                matches.append(_parse_canonical(canonical, bg, int(wall), rotation))
            except ValueError:
                pass
    if len(matches) != 1:
        raise ValueError("input must identify exactly one emitter and receiver interpretation")
    return matches[0]


def _prediction(info, active=None, normalize=True):
    """Repair receivers first; then independently scan opaque light lanes."""
    a, bg = info['a'], info['bg']
    out = a.copy()
    if active is None:
        active = info['active']
    if normalize:
        target = info['rods'][active[len(active)//2]][1]
        if info['baseline']+target > a.shape[1]:
            raise ValueError("normalized receivers would exceed the canvas")
        for i in active:
            y, length, color = info['rods'][i]
            out[y, info['baseline']:info['baseline']+length] = bg
            out[y, info['baseline']:info['baseline']+target] = color
    # Use repaired rods for the physical shadows. Light never overwrites them.
    opaque = (out != bg) & (out != info['wall']) & (out != info['payload'])
    for y in range(a.shape[0]):
        blocked = False
        for x in range(info['rear']+1, a.shape[1]):
            if not info['footprint'][y, x]:
                continue
            if opaque[y, x]:
                blocked = True
            elif out[y, x] == bg and not blocked:
                out[y, x] = info['payload']
    return np.rot90(out, info['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from visible input geometry and colors."""
    return _prediction(_inspect(input_grid))


def _geometry(rng):
    height = int(rng.integers(16, 31))
    periods = [int(rng.integers(1, 4)) for _ in range(2)]
    phases = [int(rng.integers(p)) for p in periods]
    back_length = int(rng.integers(1, 5))
    profiles = []
    for p, phase in zip(periods, phases):
        v = int(rng.integers(1, 4))
        back = [v]
        for _ in range(back_length-1):
            v = max(1, min(4, v+int(rng.integers(-1, 2))))
            back.append(v)
        flare = [v+(t+phase)//p for t in range(1, 2*max(periods)+2)]
        profiles.append(back+flare)
    span = max(profiles[0])+max(profiles[1])+1
    if span > height:
        return None
    center = int(rng.integers(max(profiles[0]), height-max(profiles[1])))
    rear = int(rng.integers(0, 4))
    mouth = rear+len(profiles[0])
    baseline = mouth+int(rng.integers(3, 10))
    if baseline > 25:
        return None
    capacity = int(rng.integers(4, min(12, 30-baseline)+1))
    width = baseline+capacity
    wall = np.zeros((height, width), dtype=bool)
    footprint = np.zeros_like(wall)
    wall[center-profiles[0][0]:center+profiles[1][0]+1, rear] = True
    ys = np.arange(height)
    for x in range(rear+1, width):
        if x <= mouth:
            j = x-rear-1
            lo, hi = center-profiles[0][j], center+profiles[1][j]
            wall[lo, x] = wall[hi, x] = True
        else:
            t = x-(rear+back_length)
            lo = center-profiles[0][back_length-1]-(t+phases[0])//periods[0]
            hi = center+profiles[1][back_length-1]+(t+phases[1])//periods[1]
        footprint[:, x] = (ys > lo) & (ys < hi)
    return dict(wall=wall, footprint=footprint, rear=rear, mouth=mouth,
                baseline=baseline, capacity=capacity, back_length=back_length)


def _geometric_index(rows):
    distances = [abs(2*y-rows[0]-rows[-1]) for y in rows]
    best = min(distances)
    ids = [i for i, v in enumerate(distances) if v == best]
    return ids[0] if len(ids) == 1 else None


def _choose_layout(rng, geo, kind, avoid_count):
    h, w = geo['wall'].shape
    counts = [n for n in (7, 9, 11) if n != avoid_count] if kind == 'witness' else [n for n in range(5, 12) if n != avoid_count]
    counts = [n for n in counts if 2*(n-1) < h]
    if not counts:
        return None
    n = int(rng.choice(counts))
    gaps = [2]*(n-1)
    for _ in range(int(rng.integers(0, h-2*(n-1)))):
        gaps[int(rng.integers(n-1))] += 1
    span = sum(gaps)
    offset = int(rng.integers(h-span))
    rows = [offset]+[offset+int(x) for x in np.cumsum(gaps)]
    eligible = [i for i, y in enumerate(rows) if geo['footprint'][y, geo['baseline']]]
    if len(eligible) < 3 or len(eligible) % 2 == 0:
        return None
    if kind == 'clean':
        # Teach that membership is decided at the anchor, not at any later
        # intersection of a long receiver with the widening beam.
        tail_only = [i for i, y in enumerate(rows) if i not in eligible
                     and np.any(geo['footprint'][y, geo['baseline']+1:])]
        if not tail_only:
            return None
        return rows, eligible, []
    blockable = {i for i in eligible if np.any(geo['footprint'][rows[i], geo['mouth']+1:geo['baseline']-2])}
    sizes = [5] if kind == 'witness' else [k for k in (3, 5, 7) if k < len(eligible)]
    options = []
    for k in sizes:
        if k >= len(eligible):
            continue
        for group in combinations(eligible, k):
            removed = set(eligible)-set(group)
            if not removed <= blockable:
                continue
            ref = group[k//2]
            if kind == 'witness':
                selected_rows = [rows[i] for i in group]
                gi = _geometric_index(selected_rows)
                if ref in (eligible[len(eligible)//2], n//2) or gi is None or gi == k//2:
                    continue
            options.append((list(group), sorted(removed)))
    if not options:
        return None
    active, removed = options[int(rng.integers(len(options)))]
    return rows, active, removed


def _build(rng, palette, rotation, kind, avoid_count=None, avoid_refs=(), rich=False):
    geo = _geometry(rng)
    if geo is None:
        return None
    layout = _choose_layout(rng, geo, kind, avoid_count)
    if layout is None:
        return None
    rows, active, removed = layout
    bg, wall, light, obstacle_a, obstacle_b = map(int, palette[:5])
    a = np.full(geo['wall'].shape, bg, dtype=np.int8)
    a[geo['wall']] = wall
    # One to four connected payload pixels, at a variable chamber location.
    chamber = geo['footprint'].copy()
    chamber[:, geo['mouth']+1:] = False
    available = set(map(tuple, np.argwhere(chamber)))
    first = sorted(available)[int(rng.integers(len(available)))]
    patch = {first}
    for _ in range(int(rng.integers(0, 4))):
        frontier = sorted({q for y, x in patch for q in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)) if q in available-patch})
        if not frontier:
            break
        patch.add(frontier[int(rng.integers(len(frontier)))])
    for y, x in patch:
        a[y, x] = light
    obstacles = np.zeros(a.shape, dtype=bool)
    for j, i in enumerate(removed):
        y = rows[i]
        positions = [x for x in range(geo['mouth']+1, geo['baseline']-2) if geo['footprint'][y, x]]
        x = int(rng.choice(positions))
        shape = ((0, 0),)
        if rich or (kind != 'witness' and rng.random() < .4):
            shape = ((0,0),(1,0),(0,1),(1,1)) if rng.random() < .5 else ((0,0),(1,0),(1,1))
        for dy, dx in shape:
            yy, xx = y+dy, x+dx
            if yy < a.shape[0] and xx <= geo['baseline']-2 and yy not in [rows[z] for z in active]:
                a[yy, xx] = obstacle_a if j % 2 == 0 else obstacle_b
                obstacles[yy, xx] = True
        if rich and x+2 <= geo['baseline']-2:
            a[y, x+2] = obstacle_b
            obstacles[y, x+2] = True
    # Optional inert colored material: it starts outside the virtual beam.
    if rng.random() < .7:
        candidates = [(int(y), int(x)) for y, x in np.argwhere(~geo['footprint'] & (a == bg))
                      if geo['mouth'] < x <= geo['baseline']-2]
        if candidates:
            y, x = candidates[int(rng.integers(len(candidates)))]
            a[y, x] = obstacle_b
            obstacles[y, x] = True
    eligible_refs = [r for r in range(2, geo['capacity']) if r not in avoid_refs]
    if not eligible_refs:
        eligible_refs = list(range(2, geo['capacity']))
    target = int(rng.choice(eligible_refs))
    reference = active[len(active)//2]
    lengths = [int(rng.integers(1, geo['capacity']+1)) for _ in rows]
    def short():
        return int(rng.integers(1, target))
    def long():
        return int(rng.integers(target+1, geo['capacity']+1))
    others = [i for i in active if i != reference]
    rng.shuffle(others)
    if kind == 'witness':
        mostly_short = bool(rng.integers(2))
        for i in others:
            lengths[i] = short() if mostly_short else long()
        lengths[others[0]] = long() if mostly_short else short()
        # All tempting but wrong reference members have a different span.
        for i in range(len(rows)):
            if i not in active and lengths[i] == target:
                lengths[i] = short()
    else:
        mode = 'mixed' if kind == 'clean' else str(rng.choice(['mixed','extend','trim','one']))
        for i in others:
            lengths[i] = target if mode == 'one' or rng.random() < .2 else (short() if mode == 'extend' or (mode == 'mixed' and rng.random() < .5) else long())
        lengths[others[0]] = short() if mode in ('mixed','extend','one') else long()
        if mode == 'mixed':
            lengths[others[1]] = long()
    lengths[reference] = target
    if kind == 'clean':
        tail_only = [i for i, y in enumerate(rows) if i not in active
                     and np.any(geo['footprint'][y, geo['baseline']+1:])]
        lengths[int(rng.choice(tail_only))] = geo['capacity']
    pool = list(map(int, palette[5:]))
    size = 1 if kind == 'clean' else int(rng.integers(2, 6)) if kind == 'witness' else int(rng.integers(1, 6))
    colors = [int(rng.choice(pool[:size])) for _ in rows]
    if kind == 'witness':
        colors[others[0]], colors[others[1]] = pool[0], pool[1]
    out = a.copy()
    for i, (y, length, color) in enumerate(zip(rows, lengths, colors)):
        a[y, geo['baseline']:geo['baseline']+length] = color
        span = target if i in active else length
        out[y, geo['baseline']:geo['baseline']+span] = color
    # Construction uses a cumulative shadow mask; solver uses a lane scan.
    opaque = (out != bg) & (out != wall) & (out != light) & geo['footprint']
    shadow = np.maximum.accumulate(opaque, axis=1)
    out[geo['footprint'] & (out == bg) & ~shadow] = light
    return {'input': np.rot90(a, rotation).copy(), 'output': np.rot90(out, rotation).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    orientations = list(map(int, rng.permutation(4)))
    shared_palette = rng.random() < .25
    palette = rng.permutation(10)
    puzzle, seen = {'train': [], 'test': []}, set()
    first_count, references = None, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            kind = ('clean' if index == 0 else 'witness' if index == 1 else str(rng.choice(['clean','blocked']))) if split == 'train' else 'blocked'
            rotation = orientations[index % 2] if split == 'train' else orientations[2+index % 2]
            colors = palette if shared_palette else rng.permutation(10)
            for _ in range(2500):
                pair = _build(rng, colors, rotation, kind,
                              first_count if split == 'train' and index == 1 else None,
                              references if (split == 'train' and index == 1) else (),
                              rich=split == 'test' or (index >= 2 and rng.random() < .5))
                if pair is None or _key(pair['input']) in seen:
                    continue
                try:
                    info = _inspect(pair['input'])
                except ValueError:
                    continue
                if not np.array_equal(_prediction(info), pair['output']):
                    raise AssertionError("construction and input-only solution disagree")
                if kind == 'witness' and not _features(info)['selection_witness']:
                    continue
                puzzle[split].append(pair)
                seen.add(_key(pair['input']))
                if split == 'train':
                    first_count = len(info['rods']) if index == 0 else first_count
                    references.add(info['rods'][info['reference']][1])
                break
            else:
                raise RuntimeError("bounded geometry search exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    a, rods, active = info['a'], info['rods'], info['active']
    eligible = [i for i, (y, length, color) in enumerate(rods) if info['footprint'][y, info['baseline']]]
    rows = [rods[i][0] for i in active]
    lengths = [rods[i][1] for i in active]
    target = rods[info['reference']][1]
    geo_index = _geometric_index(rows)
    shadow_reference = eligible[len(eligible)//2]
    selection_witness = (len(active) == 5 and len(rods) % 2 == 1
                         and info['reference'] not in (shadow_reference, len(rods)//2)
                         and geo_index is not None and geo_index != len(active)//2
                         and lengths[geo_index] != target
                         and sorted(lengths)[len(active)//2] != target
                         and rods[shadow_reference][1] != target
                         and rods[len(rods)//2][1] != target)
    # Role-normalized structure preserves geometry while removing the palette.
    normalized = np.zeros(a.shape, dtype=np.int8)
    normalized[a == info['wall']] = 1
    normalized[a == info['payload']] = 2
    normalized[info['obstacles']] = 3
    for y, length, color in rods:
        normalized[y, info['baseline']:info['baseline']+length] = 4
    wall_pts = np.argwhere(a == info['wall'])
    lo, hi = wall_pts.min(axis=0), wall_pts.max(axis=0)
    outline = (a[lo[0]:hi[0]+1, lo[1]:hi[1]+1] == info['wall']).astype(np.int8)
    tail_only = any(i not in eligible and length != target
                    and np.any(info['footprint'][y, info['baseline']:info['baseline']+length])
                    for i, (y, length, color) in enumerate(rods))
    return dict(rotation=info['rotation'], count=len(rods), active_count=len(active),
                eligible_count=len(eligible), reference=target, rows=tuple(y for y,l,c in rods),
                active_rows=tuple(rows), lengths=tuple(lengths),
                blocked_anchors=len(eligible)-len(active), selection_witness=selection_witness,
                extension=any(v < target for v in lengths), trim=any(v > target for v in lengths),
                periods=tuple(f[0] for f in info['fits']), shape=a.shape,
                palette=(info['bg'],info['wall'],info['payload']),
                receiver_colors=len({c for y,l,c in rods}), outline=_key(outline),
                structure=_key(normalized), baseline=info['baseline'],
                projection=a.shape[1]-info['mouth']-1,
                asymmetry=any(x+y != info['upper'][0]+info['lower'][0] for x,y in zip(info['upper'],info['lower'])),
                nonmonotone=any(d < 0 for d in np.diff(np.array(info['lower'])-np.array(info['upper']))),
                grouped=any(len(g)>1 for g in _components(info['obstacles'])),
                repeated_lane=bool(np.any((info['obstacles'] & info['footprint']).sum(axis=1)>1)),
                inert=bool(np.any(info['obstacles'] & ~info['footprint'])),
                tail_only=tail_only)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train','test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, records, directions = set(), {'train': [], 'test': []}, {}
    for split in ('train', 'test'):
        directions[split] = set()
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _key(inp.astype(np.int64))
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _inspect(inp)
            assert np.array_equal(_prediction(info), out), "wrong solution"
            assert np.any(inp != out), "identity pair"
            canonical_out = np.rot90(out, -info['rotation'])
            fixed = (info['a'] == info['wall']) | (info['a'] == info['payload']) | info['obstacles']
            assert np.array_equal(info['a'][fixed], canonical_out[fixed])
            target = info['rods'][info['reference']][1]
            for i, (y, length, color) in enumerate(info['rods']):
                span = target if i in info['active'] else length
                assert np.all(canonical_out[y, info['baseline']:info['baseline']+span] == color)
                if i in info['active'] and length > target:
                    assert np.all(canonical_out[y, info['baseline']+target:info['baseline']+length] == info['bg'])
            features = _features(info)
            records[split].append(features)
            directions[split].add(info['rotation'])
    train = records['train']
    assert any(f['blocked_anchors'] == 0 for f in train)
    assert any(f['blocked_anchors'] == 0 and f['tail_only'] for f in train)
    assert any(f['selection_witness'] for f in train)
    assert len({f['reference'] for f in train}) >= 2
    assert len({f['count'] for f in train}) >= 2
    assert any(f['receiver_colors'] == 1 for f in train)
    assert any(f['receiver_colors'] > 1 for f in train)
    assert any(f['extension'] for f in train) and any(f['trim'] for f in train)
    assert directions['train'].isdisjoint(directions['test']), "lost rotation transfer"
    assert all(f['blocked_anchors'] > 0 for f in records['test'])


def self_test():
    """Bounded 42-seed campaign, semantic ablations and equivariance checks."""
    rng = np.random.default_rng(71831)
    # Hand-specified large-group boundary case, outside the sampler's narrower
    # count range. Emitter geometry supplies the axis of a one-cell reference.
    fixture = np.zeros((27,18), dtype=np.int8)
    fixture[9:16,0] = 1
    for x in range(1,6):
        fixture[10-x,x] = fixture[14+x,x] = 1
    fixture[12,2] = 2
    expected = fixture.copy()
    for i,y in enumerate(range(0,25,2)):
        length, color = (1 if i == 6 else 2+i % 4), 3+i % 3
        fixture[y,11:11+length] = color
        expected[y,11] = color
    for y in range(27):
        for x in range(1,18):
            # No construction or parser masks determine this expected grid.
            inside = 10-x < y < 14+x
            dark = y in range(0,25,2) and x > 11
            if inside and not dark and expected[y,x] == 0:
                expected[y,x] = 2
    assert len(_inspect(fixture)['active']) == 13
    assert np.array_equal(solve(fixture), expected)
    features, structures, puzzles = [], set(), set()
    for j, seed in enumerate(CAMPAIGN_SEEDS):
        nt = 2+j % 9
        nq = 1+(j//9) % min(4, nt-1)
        puzzle = generate_puzzle(seed, nt, nq)
        validate_generated_puzzle(puzzle)
        fingerprint = tuple(_key(p['input']) for s in ('train','test') for p in puzzle[s])
        puzzles.add(fingerprint)
        if j < 3:
            repeated = generate_puzzle(seed, nt, nq)
            assert all(np.array_equal(p[k], q[k]) for s in ('train','test') for p,q in zip(puzzle[s],repeated[s]) for k in ('input','output'))
        for split in ('train','test'):
            for i, pair in enumerate(puzzle[split]):
                info = _inspect(pair['input'])
                f = _features(info)
                features.append(f)
                structures.add(f['structure'])
                if f['selection_witness']:
                    eligible = [z for z,(y,l,c) in enumerate(info['rods']) if info['footprint'][y,info['baseline']]]
                    assert not np.array_equal(_prediction(info, active=eligible), pair['output']), "ignoring A did not change answer"
                    assert not np.array_equal(_prediction(info, normalize=False), pair['output']), "ignoring B did not change answer"
                    without_obstacles = info['a'].copy()
                    without_obstacles[info['obstacles']] = info['bg']
                    clear = np.rot90(without_obstacles, info['rotation']).copy()
                    clear_info = _inspect(clear)
                    assert clear_info['reference'] != info['reference']
                    assert not np.array_equal(solve(clear), pair['output'])
                    new_reference = info['a'].copy()
                    y, length, color = info['rods'][info['reference']]
                    new_reference[y, info['baseline']+length] = color
                    altered = np.rot90(new_reference, info['rotation']).copy()
                    assert not np.array_equal(solve(altered), pair['output'])
                if i == 0:
                    palette = rng.permutation(10)
                    assert np.array_equal(solve(palette[pair['input']]), palette[pair['output']])
                    for turns in range(4):
                        for mirror in (False, True):
                            inp, out = np.rot90(pair['input'],turns), np.rot90(pair['output'],turns)
                            if mirror:
                                inp, out = np.fliplr(inp), np.fliplr(out)
                            assert np.array_equal(solve(inp), out)
    assert len(puzzles) == 42
    assert len(structures) >= 180
    assert len({f['shape'] for f in features}) >= 50
    assert len({f['outline'] for f in features}) >= 80
    assert {f['rotation'] for f in features} == set(range(4))
    assert {p for f in features for p in f['periods']} == {1,2,3}
    assert len({f['reference'] for f in features}) >= 7
    assert len({f['rows'] for f in features}) >= 100
    assert {f['count'] for f in features} >= {5,6,7,8,9,10,11}
    for flag in ('asymmetry','nonmonotone','grouped','repeated_lane','inert'):
        assert sum(bool(f[flag]) for f in features) >= 5, flag
    for role in range(3):
        assert {f['palette'][role] for f in features} == set(range(10))
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            p = generate_puzzle(103+a*10+b,a,b)
            assert len(p['train']) == a and len(p['test']) == b
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    damaged = generate_puzzle(39,2,1)
    pair = damaged['test'][0]
    y, x = np.argwhere(pair['input'] != pair['output'])[0]
    pair['output'][y,x] = pair['input'][y,x]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return
    raise AssertionError("corrupted label accepted")
