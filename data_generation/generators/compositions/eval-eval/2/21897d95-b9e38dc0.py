"""Pigment spotlight: first optical contact transfers color to a whole territory.

Only NumPy and the standard library are used. The decoder reconstructs all roles
and original components from the input. Construction keeps its own component IDs
and uses a union of lane shadows; prediction uses a first-contact lane scan.
"""
from __future__ import annotations

import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grids must be integer 2D arrays of size 1..30 with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(a, mask):
    """Four-connected, equal-color ORIGINAL objects, excluding the source."""
    ids = np.full(a.shape, -1, dtype=np.int16)
    objects, colors = [], []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if ids[r, c] >= 0:
            continue
        label, color = len(objects), int(a[r, c])
        ids[r, c] = label
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in _D4:
                yy, xx = y + dy, x + dx
                if (0 <= yy < a.shape[0] and 0 <= xx < a.shape[1]
                        and mask[yy, xx] and ids[yy, xx] < 0 and a[yy, xx] == color):
                    ids[yy, xx] = label
                    stack.append((yy, xx))
        objects.append(tuple(cells))
        colors.append(color)
    return ids, objects, colors


def _fit_flank(values, sign):
    """Fit a terminal thin stair with at least two complete outward steps.

    Prefer the longest matching suffix. Requiring 2*p+1 samples and three
    different levels makes the forward phase visible, even at a run's end.
    """
    values = np.asarray(values, dtype=int)
    offset = np.arange(len(values)) - len(values) + 1
    candidates = []
    for period in (1, 2, 3):
        for phase in range(period):
            predicted = values[-1] + sign * ((offset + phase) // period)
            length = 0
            for equal in (predicted == values)[::-1]:
                if not equal:
                    break
                length += 1
            if length >= 2 * period + 1 and len(set(values[-length:])) >= 3:
                candidates.append((length, -period, phase))
    if not candidates:
        raise ValueError("unreadable terminal stair")
    length, negative_period, phase = max(candidates)
    return -negative_period, phase, int(values[-1]), length


def _read_input(input_grid):
    """Decode without the seed, output, puzzle demonstrations or generation data."""
    inp = _grid(input_grid)
    frequencies = np.bincount(inp.ravel(), minlength=10)
    bg = int(frequencies.argmax())
    if frequencies[bg] <= inp.size // 2:
        raise ValueError("the uniform background must occupy more than half the input")
    matches = []
    for rotation in range(4):
        # rotation=0/1/2/3 denotes a right/up/left/down emitting input.
        a = np.rot90(inp, -rotation).copy()
        h, w = a.shape
        for wall_value in np.unique(a):
            wall = int(wall_value)
            if wall == bg:
                continue
            positions = np.argwhere(a == wall)
            if len(positions) < 15:
                continue
            rear, mouth = int(positions[:, 1].min()), int(positions[:, 1].max())
            cap = np.flatnonzero(a[:, rear] == wall)
            if (len(cap) < 5 or np.any(np.diff(cap) != 1)
                    or mouth - rear < 6 or mouth >= w - 1):
                continue
            upper, lower = [], []
            valid = True
            for x in range(rear + 1, mouth + 1):
                rows = np.flatnonzero(a[:, x] == wall)
                if len(rows) != 2 or rows[1] - rows[0] < 4:
                    valid = False
                    break
                upper.append(int(rows[0]))
                lower.append(int(rows[1]))
            if (not valid or cap[0] != upper[0] or cap[-1] != lower[0]
                    or np.any(np.abs(np.diff(upper)) > 1)
                    or np.any(np.abs(np.diff(lower)) > 1)):
                continue
            try:
                fits = (_fit_flank(upper, -1), _fit_flank(lower, 1))
            except ValueError:
                continue
            chamber = np.zeros(a.shape, dtype=bool)
            for x, lo, hi in zip(range(rear + 1, mouth + 1), upper, lower):
                chamber[lo + 1:hi, x] = True
            annotations = []
            for y, x in np.argwhere(chamber):
                y, x = int(y), int(x)
                cells = ((y - 1, x), (y, x), (y + 1, x), (y, x + 1))
                if not all(0 <= yy < h and 0 <= xx < w and chamber[yy, xx]
                           for yy, xx in cells):
                    continue
                ink = int(a[y - 1, x])
                if ink in (bg, wall) or a[y + 1, x] != ink or a[y, x + 1] != ink:
                    continue
                remainder = chamber.copy()
                for p in cells:
                    remainder[p] = False
                donors = np.unique(a[remainder])
                if len(donors) != 1:
                    continue
                donor = int(donors[0])
                if donor in (bg, wall, ink):
                    continue
                explicit = int(a[y, x]) != ink
                pigment = int(a[y, x]) if explicit else donor
                if pigment in (bg, wall, ink):
                    continue
                expected_ink = np.zeros(a.shape, dtype=bool)
                for p in cells:
                    expected_ink[p] = a[p] == ink
                if not np.array_equal(a == ink, expected_ink):
                    continue
                annotations.append((cells, donor, ink, pigment, explicit, (y, x)))
            if len(annotations) != 1:
                continue
            cells, donor, ink, pigment, explicit, center = annotations[0]
            receiver_mask = (a != bg) & (a != wall) & ~chamber
            if np.any(receiver_mask[:, :mouth + 1]):
                continue
            ids, objects, colors = _components(a, receiver_mask)
            if not objects:
                continue
            beam = np.zeros(a.shape, dtype=bool)
            boundaries = []
            for x in range(mouth + 1, w):
                p, phase, end, _ = fits[0]
                lo = end - (x - mouth + phase) // p
                p, phase, end, _ = fits[1]
                hi = end + (x - mouth + phase) // p
                beam[:, x] = (np.arange(h) > lo) & (np.arange(h) < hi)
                boundaries.append((lo, hi))
            clean = a.copy()
            for point in cells:
                clean[point] = donor
            matches.append(dict(canonical=a, clean=clean, rotation=rotation,
                                bg=bg, wall=wall, ink=ink, donor=donor, pigment=pigment,
                                explicit=explicit, arrow=cells, center=center, rear=rear,
                                mouth=mouth, chamber=chamber, fits=fits, beam=beam,
                                boundaries=boundaries, ids=ids, objects=objects, colors=colors))
    if len(matches) != 1:
        raise ValueError("input must specify exactly one reservoir, T and open funnel")
    return matches[0]


def _scan(info):
    """Scan the original material, keeping opacity separate from output color."""
    a, ids, beam = info['canonical'], info['ids'], info['beam']
    out = info['clean'].copy()
    hit = set()
    first_contacts = np.zeros(a.shape, dtype=bool)
    shadow = np.zeros(a.shape, dtype=bool)
    for y in range(a.shape[0]):
        blocked = False
        for x in range(info['mouth'] + 1, a.shape[1]):
            if not beam[y, x]:
                continue
            shadow[y, x] = blocked
            if ids[y, x] >= 0:
                if not blocked:
                    hit.add(int(ids[y, x]))
                    first_contacts[y, x] = True
                blocked = True
            elif not blocked:
                out[y, x] = info['pigment']
    for label in hit:
        out[ids == label] = info['pigment']
    return out, hit, first_contacts, shadow


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Paint first-contact territories and pre-contact empty lanes from input only."""
    info = _read_input(input_grid)
    return np.rot90(_scan(info)[0], info['rotation']).copy()


def _sample_geometry(rng, projection_range):
    periods = [int(rng.integers(1, 4)), int(rng.integers(1, 4))]
    if rng.random() < .55:
        periods[1] = periods[0]
    phases = [int(rng.integers(p)) for p in periods]
    reservoir_length = int(rng.integers(3, 6))
    profiles = []
    for _ in range(2):
        values = [int(rng.integers(2, 4))]
        for _ in range(reservoir_length - 1):
            values.append(int(np.clip(values[-1] + rng.integers(-1, 2), 2, 4)))
        profiles.append(values)
    if rng.random() < .25:
        profiles[1] = profiles[0].copy()
        periods[1], phases[1] = periods[0], phases[0]
    flare_length = 2 * max(periods) + 1
    extended = []
    for profile, p, phase in zip(profiles, periods, phases):
        extended.append(profile + [profile[-1] + (t + phase) // p
                                   for t in range(1, flare_length + 1)])
    length = reservoir_length + flare_length
    # Reserve at least 13 columns for held-out longer projections.
    rear = int(rng.integers(0, min(3, 16 - length) + 1))
    mouth = rear + length
    low_projection, high_projection = projection_range
    projection = int(rng.integers(low_projection, min(high_projection, 29 - mouth) + 1))
    w = mouth + projection + 1
    top_width, bottom_width = max(extended[0]), max(extended[1])
    h = int(rng.integers(top_width + bottom_width + 7, 31))
    center = int(rng.integers(top_width + 3, h - bottom_width - 3))
    wall = np.zeros((h, w), dtype=bool)
    chamber = np.zeros_like(wall)
    wall[center - profiles[0][0]:center + profiles[1][0] + 1, rear] = True
    for i, (up, down) in enumerate(zip(*extended), 1):
        x = rear + i
        wall[center - up, x] = True
        wall[center + down, x] = True
        chamber[center - up + 1:center + down, x] = True
    beam = np.zeros_like(wall)
    for x in range(mouth + 1, w):
        t = x - rear - reservoir_length
        up = center - profiles[0][-1] - (t + phases[0]) // periods[0]
        down = center + profiles[1][-1] + (t + phases[1]) // periods[1]
        beam[:, x] = (np.arange(h) > up) & (np.arange(h) < down)
    return dict(wall=wall, chamber=chamber, beam=beam, rear=rear, mouth=mouth, center=center)


def _shadow_of(ids, beam):
    shadow = np.zeros(ids.shape, dtype=bool)
    for y, x in np.argwhere((ids >= 0) & beam):
        shadow[y, x + 1:] = True
    return shadow & beam


def _register(a, ids, points, color, bg):
    points = sorted(set(points))
    if not points or any(not (0 <= y < a.shape[0] and 0 <= x < a.shape[1])
                         or a[y, x] != bg for y, x in points):
        return False
    for y, x in points:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < a.shape[0] and 0 <= xx < a.shape[1] and ids[yy, xx] >= 0:
                    return False
    label = int(ids.max()) + 1
    for point in points:
        a[point] = color
        ids[point] = label
    return True


def _shape(rng):
    h, w = int(rng.integers(1, 5)), int(rng.integers(1, 5))
    style = int(rng.integers(5))
    if style == 0 or min(h, w) == 1:
        points = {(y, x) for y in range(h) for x in range(w)}
    elif style == 1:
        points = {(y, 0) for y in range(h)} | {(h - 1, x) for x in range(w)}
    elif style == 2:
        points = {(y, x) for y in range(h) for x in range(min(w, y + 1))}
    elif style == 3:
        points = {(y, 0) for y in range(h)} | {(0, x) for x in range(w)} | {(h - 1, x) for x in range(w)}
    else:
        points = {(y, w // 2) for y in range(h)} | {(h // 2, x) for x in range(w)}
    turns = int(rng.integers(4))
    for _ in range(turns):
        points = {(x, -y) for y, x in points}
    y0, x0 = min(y for y, x in points), min(x for y, x in points)
    return sorted((y - y0, x - x0) for y, x in points)


def _place_shape(rng, a, ids, offsets, color, geom, bg, kind):
    shadow = _shadow_of(ids, geom['beam'])
    height = max(y for y, x in offsets) + 1
    width = max(x for y, x in offsets) + 1
    choices = []
    for x in range(geom['mouth'] + 1, a.shape[1] - width + 1):
        for y in range(a.shape[0] - height + 1):
            points = [(y + dy, x + dx) for dy, dx in offsets]
            if any(a[p] != bg for p in points):
                continue
            if any(ids[max(0, yy - 1):yy + 2, max(0, xx - 1):xx + 2].max() >= 0
                   for yy, xx in points):
                continue
            in_beam = [bool(geom['beam'][p]) for p in points]
            reached = [bool(geom['beam'][p] and not shadow[p]) for p in points]
            if kind == 'hit' and not any(reached):
                continue
            if kind == 'inert' and any(reached):
                continue
            if kind == 'straddle' and (not any(reached) or all(in_beam)):
                continue
            if kind == 'opacity' and (not any(reached) or
                    not any(reached[i] and xx <= a.shape[1] - 3
                            and np.any(geom['beam'][yy, xx + 1:] & (a[yy, xx + 1:] == bg))
                            for i, (yy, xx) in enumerate(points))):
                continue
            choices.append(points)
    if not choices:
        return False
    return _register(a, ids, choices[int(rng.integers(len(choices)))], color, bg)


def _construct_output(a, ids, geom, donor, pigment, arrow):
    """Construction uses explicit original IDs and unioned downstream shadows."""
    shadow = _shadow_of(ids, geom['beam'])
    out = a.copy()
    for p in arrow:
        out[p] = donor
    out[geom['beam'] & (a == geom['bg']) & ~shadow] = pigment
    hits = set(map(int, ids[geom['beam'] & (ids >= 0) & ~shadow]))
    for label in hits:
        out[ids == label] = pigment
    return out


def _build_pair(rng, roles, kind, rotation):
    bg, wall, ink, donor, pigment = map(int, roles[:5])
    explicit = kind != 'contact' and (kind == 'shadow' or rng.random() < .55)
    if not explicit:
        pigment = donor
    projection_range = (8, 12) if kind == 'contact' else ((11, 16) if kind == 'shadow' else
                                                       ((13, 21) if kind == 'test' else (8, 17)))
    geom = _sample_geometry(rng, projection_range)
    geom['bg'] = bg
    a = np.full(geom['wall'].shape, bg, dtype=np.int8)
    a[geom['wall']] = wall
    a[geom['chamber']] = donor
    # The arrow may be off-center and may sit in different parts of the reservoir.
    eligible = []
    for y, x in np.argwhere(geom['chamber']):
        y, x = int(y), int(x)
        cells = ((y - 1, x), (y, x), (y + 1, x), (y, x + 1))
        if x <= geom['rear'] + 4 and all(0 <= yy < a.shape[0] and 0 <= xx < a.shape[1]
                                       and geom['chamber'][yy, xx] for yy, xx in cells):
            eligible.append((cells, (y, x)))
    arrow, center = eligible[int(rng.integers(len(eligible)))]
    for p in arrow:
        a[p] = ink
    if explicit:
        a[center] = pigment
    ids = np.full(a.shape, -1, dtype=np.int16)
    receiver_colors = [int(c) for c in roles if c not in (bg, wall, ink, pigment, donor)]
    if kind in ('shadow', 'test'):
        y, m = geom['center'], geom['mouth']
        front_x = m + int(rng.integers(1, 3))
        front_width, front_height = int(rng.integers(1, 3)), int(rng.integers(2, 5))
        front = [(yy, xx) for yy in range(y - 1, y - 1 + front_height)
                 for xx in range(front_x, front_x + front_width)]
        _need(_register(a, ids, front, receiver_colors[0], bg), 'foreground placement')
        hidden_x = front_x + front_width + 1
        hidden_width = int(rng.integers(1, 3))
        hidden_y = int(rng.integers(y - 1, y - 1 + front_height))
        hidden = [(hidden_y, xx) for xx in range(hidden_x, hidden_x + hidden_width)]
        _need(_register(a, ids, hidden, receiver_colors[0], bg), 'hidden placement')
        partial_x = hidden_x + hidden_width + 1
        partial_y = y + front_height - 2
        partial_height = int(rng.integers(2, 5))
        partial = [(yy, partial_x) for yy in range(partial_y, partial_y + partial_height)]
        if kind == 'test' and rng.random() < .6 and partial_x + 1 < a.shape[1]:
            partial.append((partial_y + partial_height - 1, partial_x + 1))
        _need(_register(a, ids, partial, receiver_colors[1], bg), 'partial placement')
        # This is optional in tests. The implicit contact demonstration teaches
        # pigment opacity, so explicit transfers do not rely on an exterior token.
        if kind == 'test' and rng.random() < .45:
            _place_shape(rng, a, ids, [(0, 0)], pigment, geom, bg, 'opacity')
        desired = int(rng.integers(5, 9)) if kind == 'test' else int(rng.integers(4, 6))
    elif kind == 'contact':
        bar = [(yy, 0) for yy in range(int(rng.integers(3, 6)))]
        _need(_place_shape(rng, a, ids, bar, receiver_colors[0], geom, bg, 'straddle'),
              'no boundary contact')
        _need(_place_shape(rng, a, ids, _shape(rng), receiver_colors[1], geom, bg, 'hit'),
              'no second contact')
        # Same-color matter still blocks light. Give this witness to the implicit
        # example; the explicit example then has a pigment unique to its junction.
        _need(_place_shape(rng, a, ids, [(0, 0)], pigment, geom, bg, 'opacity'),
              'no pigment-opacity contact')
        desired = int(rng.integers(3, 5))
    else:
        desired = int(rng.integers(1, 8))
        _need(_place_shape(rng, a, ids, _shape(rng), receiver_colors[0], geom, bg, 'hit'),
              'no initial contact')
    for _ in range(30):
        if ids.max() + 1 >= desired:
            break
        color = int(rng.choice(receiver_colors))
        _place_shape(rng, a, ids, _shape(rng), color, geom, bg,
                     str(rng.choice(['hit', 'hit', 'inert', 'any'])))
    _need(ids.max() + 1 == desired, 'not enough receivers')
    out = _construct_output(a, ids, geom, donor, pigment, arrow)
    pair = dict(input=np.rot90(a, rotation).copy(), output=np.rot90(out, rotation).copy())
    info = _read_input(pair['input'])
    _need(np.array_equal(_scan(info)[0], out), 'construction/decoder discrepancy')
    features = _features(info)
    if kind == 'contact':
        _need(features['outside_transfer'] > 0 and features['pigment_shadow'] and not explicit,
              'missing closure/opacity teaching')
    if kind in ('shadow', 'test'):
        _need(features['locality'] and features['partial_transfer'] > 0,
              'missing interacting witnesses')
    if kind == 'shadow':
        _need(features['junction_only'], 'explicit teaching pigment must be unique to its junction')
    return pair


def _features(info):
    a, ids, beam = info['canonical'], info['ids'], info['beam']
    out, hit, contacts, shadow = _scan(info)
    receivers = ids >= 0
    active = np.isin(ids, list(hit)) & receivers
    partial = np.zeros(a.shape, dtype=bool)
    for y, x in np.argwhere(active & shadow):
        upstream = ids[y, :x][beam[y, :x] & (ids[y, :x] >= 0)]
        partial[y, x] = bool(len(upstream) and upstream[0] != ids[y, x])
    colors = info['colors']
    inert = set(range(len(colors))) - hit
    locality = any(colors[i] == colors[j] != info['pigment'] for i in hit for j in inert)
    pigment_shadow = False
    for y, x in np.argwhere(contacts & (a == info['pigment'])):
        pigment_shadow |= bool(np.any(beam[y, x + 1:] & (a[y, x + 1:] == info['bg'])
                                      & (out[y, x + 1:] == info['bg'])))
    return dict(shape=a.shape, rotation=info['rotation'], explicit=info['explicit'],
                junction_only=bool(info['explicit'] and np.count_nonzero(a == info['pigment']) == 1),
                periods=tuple(f[0] for f in info['fits']),
                phases=tuple(f[1] for f in info['fits']), receivers=len(colors), hits=len(hit),
                outside_transfer=int((active & ~beam & (a != info['pigment'])).sum()),
                partial_transfer=int((partial & (a != info['pigment'])).sum()),
                shadow_blank=int((shadow & (a == info['bg'])).sum()),
                hidden=sum(any(beam[p] for p in info['objects'][i]) for i in inert),
                locality=locality, pigment_shadow=pigment_shadow,
                palette=(info['bg'], info['wall'], info['ink'], info['donor'], info['pigment']),
                center=info['center'], projection=a.shape[1] - info['mouth'] - 1)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (not _integer(num_train) or not _integer(num_test) or not 2 <= num_train <= 10
            or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError('require 2..10 train, 1..4 test, and num_train > num_test')
    rng = np.random.default_rng(int(seed))
    shared_palette = bool(rng.random() < .25)
    roles = rng.permutation(10)
    shared_direction = bool(rng.random() < .25)
    direction = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            kind = ('contact' if index == 0 else 'shadow' if index == 1 else 'mixed') if split == 'train' else 'test'
            rotation = direction if shared_direction else int(rng.integers(4))
            colors = roles if shared_palette else rng.permutation(10)
            for _ in range(90):
                try:
                    pair = _build_pair(rng, colors, kind, rotation)
                except (AssertionError, ValueError):
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('bounded construction exhausted')
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    n, m = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= n <= 10 and 1 <= m <= 4 and n > m, 'illegal counts')
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, 'spotlight preserves dimensions')
            key = _key(inp)
            _need(key not in seen, 'duplicate input')
            seen.add(key)
            info = _read_input(inp)
            expected, hit, _, shadow = _scan(info)
            actual = np.rot90(out, -info['rotation'])
            _need(np.array_equal(actual, expected), 'wrong contact, closure or shadow')
            a = info['canonical']
            _need(np.array_equal(actual[a == info['wall']], a[a == info['wall']]), 'wall changed')
            _need(np.all(actual[info['chamber']] == info['donor']), 'reservoir or T restoration changed')
            for label, color in enumerate(info['colors']):
                target = info['pigment'] if label in hit else color
                _need(np.all(actual[info['ids'] == label] == target), 'nonuniform original territory transfer')
            _need(np.all(actual[shadow & (a == info['bg'])] == info['bg']), 'shadowed air painted')
            exterior_air = ~info['beam'] & ~info['chamber'] & (a == info['bg'])
            _need(np.all(actual[exterior_air] == info['bg']), 'virtual boundary or exterior painted')
            f = _features(info)
            _need(f['hits'] > 0 and np.any(inp != out), 'degenerate pair')
            features[split].append(f)
    train = features['train']
    _need(any(not f['explicit'] and f['outside_transfer'] > 0 and f['pigment_shadow'] for f in train),
          'training lacks implicit whole-territory transfer and original-pigment opacity')
    _need(any(f['explicit'] and f['junction_only'] and f['locality'] and f['partial_transfer'] > 0
              and f['shadow_blank'] > 0 for f in train),
          'training lacks explicit shadow/locality/opacity contrast')
    # These ablations model omissions of parent A and parent B respectively.
    _need(any(_ablation_diff(p['input'], 'A') > 0 for p in puzzle['train']), 'A is dispensable')
    _need(any(_ablation_diff(p['input'], 'B') > 0 for p in puzzle['train']), 'B is dispensable')


def _ablation_diff(inp, omitted):
    info = _read_input(inp)
    correct, _, _, shadow = _scan(info)
    alternative = info['clean'].copy()
    a, ids, beam = info['canonical'], info['ids'], info['beam']
    if omitted == 'A':
        # B's ordinary opaque blockers are preserved, without territory transfer.
        alternative[beam & (a == info['bg']) & ~shadow] = info['pigment']
    elif omitted == 'B':
        # Every geometric cone intersection gets the color; no optical occlusion.
        alternative[beam & (a == info['bg'])] = info['pigment']
        for label in set(map(int, ids[beam & (ids >= 0)])):
            alternative[ids == label] = info['pigment']
    else:
        raise ValueError('omitted must be A or B')
    return int(np.count_nonzero(correct != alternative))


def _manual_fixture():
    """Hand-defined phase/contact regression, without any sampling helpers."""
    a = np.zeros((15, 23), dtype=np.int8)
    a[5:10, 0] = 5
    for x, top in enumerate((5, 5, 5, 4, 4, 3, 3, 2, 2), 1):
        a[top, x] = a[14 - top, x] = 5
        a[top + 1:14 - top, x] = 3
    a[6, 2] = a[8, 2] = a[7, 3] = 8
    a[7, 2] = 4
    a[6:9, 12] = 2
    a[7, 16] = 2
    a[8:11, 19] = 9
    a[0:4, 11] = 6
    a[4, 13] = 4
    return a


def self_test():
    """Bounded semantic regressions, all count modes, and 42-seed diversity."""
    a = _manual_fixture()
    o = solve(a)
    _need(o[7, 2] == 3 and np.all(o[6:9, 12] == 4), 'token or first contact')
    _need(o[7, 16] == 2 and o[7, 14] == 0, 'hidden same-color object or straight shadow')
    _need(np.all(o[8:11, 19] == 4) and np.all(o[0:4, 11] == 4), 'component closure')
    _need(o[4, 14] == 0 and o[11, 14] == 4, 'pigment opacity or widening')
    _need(o[1, 10] == 0 and o[2, 10] == 4, 'thin virtual boundary phase')
    exposed = a.copy()
    exposed[6:9, 12] = 0
    _need(solve(exposed)[7, 16] == 4, 'removing occluder must expose receiver')
    for rotation in range(4):
        _need(np.array_equal(solve(np.rot90(a, rotation)), np.rot90(o, rotation)), 'direction equivariance')
    permutation = np.array([7, 6, 8, 1, 0, 9, 4, 2, 3, 5], dtype=np.int8)
    _need(np.array_equal(solve(permutation[a]), permutation[o]), 'role recoloring')
    shapes, outlines, object_shapes, palettes, centers = set(), set(), set(), set(), set()
    fingerprints, periods, counts = set(), set(), set()
    directions = {'train': set(), 'test': set()}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(p['input']) for s in puzzle for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _read_input(pair['input'])
                f = _features(info)
                _need(np.array_equal(solve(pair['input']), pair['output']), 'input-only solve')
                shapes.add(pair['input'].shape)
                palettes.add(f['palette'])
                periods.add(f['periods'])
                counts.add(f['receivers'])
                directions[split].add(f['rotation'])
                centers.add((f['center'][0], f['center'][1] - info['rear']))
                wall = info['canonical'] == info['wall']
                rr, cc = np.nonzero(wall)
                outlines.add(_key(wall[rr.min():rr.max() + 1, cc.min():cc.max() + 1].astype(np.int8)))
                for cells in info['objects']:
                    y0, x0 = min(y for y, x in cells), min(x for y, x in cells)
                    object_shapes.add(tuple(sorted((y - y0, x - x0) for y, x in cells)))
        if seed in (0, 13, 41):
            repeated = generate_puzzle(seed)
            _need(all(np.array_equal(p[k], q[k]) for s in puzzle
                      for p, q in zip(puzzle[s], repeated[s]) for k in ('input', 'output')), 'determinism')
    _need(len(fingerprints) == 42 and len(shapes) >= 40, 'collapsed puzzles or dimensions')
    _need(len(outlines) >= 60 and len(object_shapes) >= 25, 'collapsed source or receiver geometry')
    _need(len(palettes) >= 60 and len(centers) >= 20, 'collapsed palettes or marker positions')
    _need(len(periods) >= 6 and {1, 2, 3} == {p for pair in periods for p in pair}, 'collapsed cadences')
    _need(all(ds == set(range(4)) for ds in directions.values()), 'missing direction in a split')
    _need({1, 2, 4, 5, 6} <= counts, 'collapsed receiver counts')
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(100 + 10 * n + m, n, m)
            _need(len(p['train']) == n and len(p['test']) == m, 'explicit counts ignored')
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    puzzle = generate_puzzle(83, 2, 1)
    pair = puzzle['test'][0]
    y, x = np.argwhere(pair['input'] != pair['output'])[0]
    pair['output'][y, x] = pair['input'][y, x]
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, semantic fixtures, and every legal count combination)')
