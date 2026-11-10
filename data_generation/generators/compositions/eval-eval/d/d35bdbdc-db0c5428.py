"""Radial relays: endpoint requests send remote donor motifs outward.

All semantic data used by solve are recovered from the input. Generation uses
no parent imports, saved examples, or external resources.
"""
import numpy as np


DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'invalid grid shape')
    _require(np.issubdtype(a.dtype, np.integer), 'noninteger grid')
    _require(bool(np.all((a >= 0) & (a <= 9))), 'invalid color')
    return a


def _components(points):
    remaining, result = set(points), []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, group = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRECTIONS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _d4(tile):
    return tuple(np.rot90(tile, k) for k in range(4)) + tuple(
        np.rot90(tile[:, ::-1], k) for k in range(4))


def _parse_with_roles(a, bg, neutral):
    groups = _components(zip(*np.where((a != bg) & (a != neutral))))
    _require(3 <= len(groups) <= 8, 'expected 3..8 tiles')
    tiles = []
    for group in groups:
        r, c = min(p[0] for p in group), min(p[1] for p in group)
        _require(group == {(r+y, c+x) for y in range(3) for x in range(3)},
                 'tiles must be full separated 3x3 squares')
        tile = a[r:r+3, c:c+3].copy()
        colors, counts = np.unique(tile, return_counts=True)
        body = int(colors[np.argmax(counts)])
        _require(int(counts.max()) >= 6, 'tile identity needs at least six cells')
        tiles.append(dict(center=(r+1, c+1), body=body, tile=tile))
    _require(len({t['body'] for t in tiles}) == len(tiles), 'duplicate body identity')
    neutral_cells = set(zip(*np.where(a == neutral)))
    parts = _components(neutral_cells)
    hubs = [next(iter(p)) for p in parts if len(p) == 1]
    _require(len(hubs) == 1, 'hub must be the sole isolated neutral cell')
    hub = hubs[0]
    _require(all(0 <= hub[0]+dr < a.shape[0] and 0 <= hub[1]+dc < a.shape[1]
                 and a[hub[0]+dr, hub[1]+dc] == bg for dr, dc in DIRECTIONS),
             'hub neighbors must initially be empty')
    used, directions = {hub}, set()
    for t in tiles:
        dy, dx = t['center'][0]-hub[0], t['center'][1]-hub[1]
        radius = max(abs(dy), abs(dx))
        _require(4 <= radius <= 7, 'source radius must be 4..7')
        dr, dc = int(np.sign(dy)), int(np.sign(dx))
        _require((dy, dx) == (radius*dr, radius*dc), 'tile is not on a compass ray')
        _require((dr, dc) not in directions, 'multiple tiles on one ray')
        directions.add((dr, dc))
        spoke = {(hub[0]+k*dr, hub[1]+k*dc) for k in range(2, radius-1)}
        present = spoke & neutral_cells
        _require(not present or (radius >= 5 and present == spoke), 'partial or short spoke')
        _require(present or radius <= 6, 'unselected radius must be 4..6')
        t.update(direction=(dr, dc), radius=radius, active=bool(present),
                 value=int(t['tile'][1-dr, 1-dc]))
        used.update(present)
    _require(used == neutral_cells, 'unassigned neutral pixels')
    _require(any(t['active'] for t in tiles), 'no requests')
    return dict(background=bg, neutral=neutral, hub=hub, tiles=tiles)


def _parse(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    _require(int(counts.max()) > a.size // 2, 'background must be a strict majority')
    bg = int(colors[np.argmax(counts)])
    _require(all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])),
             'original objects need a background margin')
    interpretations = []
    for neutral in colors:
        if neutral == bg:
            continue
        try:
            interpretations.append(_parse_with_roles(a, bg, int(neutral)))
        except AssertionError:
            pass
    _require(len(interpretations) == 1, 'geometry must give unique color roles')
    info = interpretations[0]
    lookup = {t['body']: t for t in info['tiles']}
    occupied = set(zip(*np.where(a != bg)))
    destinations = set()
    hr, hc = info['hub']
    for t in info['tiles']:
        donor = lookup.get(t['value'])
        t['donor'] = donor if donor is not t else None
        if t['active'] and t['donor'] is not None:
            dr, dc = t['direction']
            r, c = hr + 2*t['radius']*dr, hc + 2*t['radius']*dc
            cells = {(r+y, c+x) for y in (-1, 0, 1) for x in (-1, 0, 1)}
            _require(all(0 <= y < a.shape[0] and 0 <= x < a.shape[1] for y, x in cells),
                     'output packet would be clipped')
            _require(not (cells & (occupied | destinations)), 'packet collision')
            destinations.update(cells)
    return a, info


def _render(a, info, alternative=None):
    """Normal rule, with explicit ablations used only for coverage checks."""
    out = a.copy()
    hr, hc = info['hub']
    tiles = info['tiles']
    sample_mode = alternative if alternative in ('center', 'outward') else 'inward'
    def value(t):
        dr, dc = t['direction']
        if sample_mode == 'center':
            return int(t['tile'][1, 1])
        if sample_mode == 'outward':
            return int(t['tile'][1+dr, 1+dc])
        return t['value']
    lookup = {t['body']: t for t in tiles}
    for t in tiles:
        if not t['active'] and alternative != 'all_tiles':
            continue
        donors = ([u for u in tiles if u is not t and value(u) == t['body']]
                  if alternative == 'reverse_lookup' else
                  [u for u in tiles if u is not t and u['body'] == value(t)])
        if not donors:
            continue
        donor = donors[0]
        dr, dc = t['direction']
        z = value(donor)
        if alternative == 'request_hub':
            z = value(t)
        if alternative == 'recursive':
            visited = {t['body'], donor['body']}
            while z in lookup and z not in visited:
                visited.add(z)
                z = value(lookup[z])
        out[hr+dr, hc+dc] = z
        tile = (t if alternative == 'requester_payload' else donor)['tile']
        if alternative == 'identity':
            packet = tile
        elif alternative == 'radial_reflection':
            packet = tile[::-1, :] if dc == 0 else tile[:, ::-1] if dr == 0 else tile[::-1, ::-1]
        else:
            packet = tile[::-1, ::-1]
        distance = 10 if alternative == 'fixed_distance' else 2*t['radius']
        r, c = hr + distance*dr, hc + distance*dc
        if 1 <= r < a.shape[0]-1 and 1 <= c < a.shape[1]-1:
            out[r-1:r+2, c-1:c+2] = packet
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Resolve and transport packets using only visible input geometry/colors."""
    a, info = _parse(input_grid)
    return _render(a, info)


def _counts(rng, num_train, num_test):
    # Explicit defaults are 4/1. None additionally supports local inspection tools.
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi):
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train)-1)+1)) if num_train is not None else int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, int(num_test)+1), 11))
    if int(num_train) <= int(num_test):
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _motif(rng, body, value, direction, identities, witness=False):
    dr, dc = direction
    sample = (1-dr, 1-dc)
    other_positions = [p for p in range(9) if p != sample[0]*3+sample[1]]
    for attempt in range(64):
        tile = np.full((3, 3), body, dtype=np.int8)
        tile[sample] = value
        extras = 2 if witness else int(rng.integers(0, 3))
        chosen = rng.choice(other_positions, extras, replace=False)
        details = [v for v in identities if v != body]
        if witness:
            details = [v for v in details if v != value]
            colors = rng.choice(details, 2, replace=False)
        else:
            colors = [value if value != body and rng.random() < .45 else int(rng.choice(details))
                      for _ in range(extras)]
        for p, color in zip(chosen, colors):
            tile.flat[int(p)] = int(color)
        if not witness or len({t.tobytes() for t in _d4(tile)}) == 8:
            return tile
    # Finite exhaustive fallback over two colored marks; no rejection loop can
    # make a seed hang. At least one such asymmetric support always exists.
    details = [v for v in identities if v not in (body, value)]
    for p in other_positions:
        for q in other_positions:
            if p == q:
                continue
            tile = np.full((3, 3), body, dtype=np.int8)
            tile[sample] = value
            tile.flat[p], tile.flat[q] = details[:2]
            if len({t.tobytes() for t in _d4(tile)}) == 8:
                return tile
    raise AssertionError('no asymmetric tile motif exists')


def _make_pair(rng, parameters, split, success_directions=None):
    order = [int(v) for v in rng.permutation(10)]
    if parameters['stable_roles']:
        bg, neutral = parameters['roles']
        identities = [int(v) for v in rng.permutation([v for v in range(10) if v not in (bg, neutral)])]
    else:
        bg, neutral, identities = order[0], order[1], order[2:]
    witness = success_directions is not None
    if witness:
        # a->b->c->d->missing; e->f->f; g->b. c is a remote,
        # unselected but color-compatible donor. f has no OTHER self donor.
        successes = list(success_directions)
        rng.shuffle(successes)
        remaining = [d for d in DIRECTIONS if d not in successes]
        rng.shuffle(remaining)
        dirs = [successes[0], successes[1], remaining[2], remaining[0],
                successes[2], remaining[1], successes[3]]
        body_indices = list(range(7))
        values = [1, 2, 3, 7, 5, 5, 1]
        active = [True, True, False, True, True, True, True]
    else:
        n = int(rng.integers(3, 5)) if split == 'test' and parameters['shift'] == 'sparse' else int(rng.integers(5 if split == 'train' and parameters['shift'] == 'sparse' else 3, 9))
        dirs = [DIRECTIONS[int(i)] for i in rng.choice(8, n, replace=False)]
        body_indices = list(range(n))
        values = [int(i) for i in rng.integers(0, 8, size=n)]
        count = int(rng.integers(1, min(n, 6)+1))
        selected = set(int(v) for v in rng.choice(n, count, replace=False))
        active = [i in selected for i in range(n)]
        source = min(selected)
        donor_choices = [i for i in range(n) if i != source]
        values[source] = int(rng.choice(donor_choices))  # At least one valid request.
    tiles = []
    for i, (body_index, vi, direction, selected) in enumerate(zip(body_indices, values, dirs, active)):
        radius = int(rng.integers(5 if selected else 4, 7))
        # Two successful training requests explicitly reveal variable doubled
        # distance. Reach queries extend this to one new successful radius 7.
        if witness and body_index in (0, 1):
            radius = 5 + body_index
        if not witness and split == 'test' and parameters['shift'] == 'reach' and i == source:
            radius = 7
        body, val = identities[body_index], identities[vi]
        tile = _motif(rng, body, val, direction, identities, witness)
        tiles.append(dict(body=body, value=val, direction=direction, radius=radius,
                          active=selected, tile=tile))
    # Choose bounds from the actual input and successful destinations. One input
    # margin keeps role recognition clear; packets can touch any output boundary.
    input_points, output_points = {(0, 0)}, {(-1, -1), (1, 1)}
    lookup = {t['body']: t for t in tiles}
    for t in tiles:
        dr, dc = t['direction']
        radius = t['radius']
        input_points.update((radius*dr+y, radius*dc+x) for y in (-1, 0, 1) for x in (-1, 0, 1))
        if t['active']:
            input_points.update((k*dr, k*dc) for k in range(2, radius-1))
        donor = lookup.get(t['value'])
        if t['active'] and donor is not None and donor is not t:
            output_points.update((2*radius*dr+y, 2*radius*dc+x) for y in (-1, 0, 1) for x in (-1, 0, 1))
    mins, maxes = [], []
    for axis in range(2):
        mins.append(min(min(p[axis] for p in output_points), min(p[axis] for p in input_points)-1))
        maxes.append(max(max(p[axis] for p in output_points), max(p[axis] for p in input_points)+1))
    sizes = [int(rng.integers(max(12, hi-lo+1), 31)) for lo, hi in zip(mins, maxes)]
    hub = tuple(-lo + int(rng.integers(size-(hi-lo+1)+1)) for lo, hi, size in zip(mins, maxes, sizes))
    a = np.full(tuple(sizes), bg, dtype=np.int8)
    a[hub] = neutral
    for t in tiles:
        dr, dc = t['direction']
        r, c = hub[0]+t['radius']*dr, hub[1]+t['radius']*dc
        t['center'] = (r, c)
        a[r-1:r+2, c-1:c+2] = t['tile']
        if t['active']:
            for k in range(2, t['radius']-1):
                a[hub[0]+k*dr, hub[1]+k*dc] = neutral
    info = dict(background=bg, neutral=neutral, hub=hub, tiles=tiles)
    return dict(input=a, output=_render(a, info))


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    nt, ne = _counts(rng, num_train, num_test)
    parameters = dict(stable_roles=bool(rng.random() < .35),
                      roles=tuple(int(v) for v in rng.choice(10, 2, replace=False)),
                      shift=str(rng.choice(['reach', 'sparse'])))
    indices = [int(v) for v in rng.permutation(8)]
    halves = [tuple(DIRECTIONS[i] for i in indices[:4]), tuple(DIRECTIONS[i] for i in indices[4:])]
    train = [_make_pair(rng, parameters, 'train', halves[i]) for i in range(2)]
    train.extend(_make_pair(rng, parameters, 'train') for _ in range(nt-2))
    test = [_make_pair(rng, parameters, 'test') for _ in range(ne)]
    puzzle = dict(train=train, test=test)
    # Palette/layout space is large; duplicate repair is bounded and geometric.
    seen = set()
    for split in ('train', 'test'):
        for i, pair in enumerate(puzzle[split]):
            for attempt in range(32):
                if _key(pair['input']) not in seen:
                    break
                # Preserve mandatory witnesses if a duplicate ever occurs.
                half = halves[i] if split == 'train' and i < 2 else None
                pair = _make_pair(rng, parameters, split, half)
            else:
                raise AssertionError('distinct input construction failed')
            puzzle[split][i] = pair
            seen.add(_key(pair['input']))
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


_ABLATIONS = ('all_tiles', 'requester_payload', 'center', 'outward',
              'reverse_lookup', 'recursive', 'request_hub', 'identity',
              'radial_reflection', 'fixed_distance')


def _coverage(puzzle, infos):
    train_infos = infos['train']
    # Resolve whole-packet orientation independently for each request direction.
    operations = {d: set(range(8)) for d in DIRECTIONS}
    saw = set()
    for pair, info in zip(puzzle['train'], train_infos):
        a, out = np.asarray(pair['input']), np.asarray(pair['output'])
        hr, hc = info['hub']
        for t in info['tiles']:
            if not t['active'] or t['donor'] is None:
                continue
            dr, dc = t['direction']
            r, c = hr+2*t['radius']*dr, hc+2*t['radius']*dc
            packet = out[r-1:r+2, c-1:c+2]
            operations[(dr, dc)].intersection_update(i for i, view in enumerate(_d4(t['donor']['tile']))
                                                     if np.array_equal(packet, view))
            saw.add((dr, dc))
    _require(saw == set(DIRECTIONS) and all(v == {2} for v in operations.values()),
             'training must identify the half-turn in every direction')
    for alternative in _ABLATIONS:
        _require(any(not np.array_equal(_render(np.asarray(p['input']), info, alternative), p['output'])
                     for p, info in zip(puzzle['train'], train_infos)),
                 'undistinguished shortcut: ' + alternative)
    features = [_features(i) for i in train_infos]
    for feature in ('failed', 'unselected_compatible', 'remote', 'shared', 'retained_donor', 'unchanged'):
        _require(any(f[feature] for f in features), 'missing training evidence: ' + feature)
    # At least one documented shift is measurable from returned cells alone.
    tr_r = {t['radius'] for i in train_infos for t in i['tiles'] if t['active']}
    te_r = {t['radius'] for i in infos['test'] for t in i['tiles'] if t['active']}
    tr_n = [len(i['tiles']) for i in train_infos]
    te_n = [len(i['tiles']) for i in infos['test']]
    reach = tr_r == {5, 6} and 7 in te_r
    if reach:
        _require(all(any(t['active'] and t['radius'] == 7 and t['donor'] is not None for t in i['tiles'])
                     for i in infos['test']), 'reach query needs a successful radius-7 request')
    _require(max(te_n) < min(tr_n) or reach, 'missing structural train/test shift')


def _features(info):
    tiles = info['tiles']
    success = [t for t in tiles if t['active'] and t['donor'] is not None]
    return dict(tiles=len(tiles), active=sum(t['active'] for t in tiles), success=len(success),
                failed=sum(t['active'] and t['donor'] is None for t in tiles),
                unselected_compatible=sum(not t['active'] and t['donor'] is not None for t in tiles),
                remote=sum(not t['donor']['active'] for t in success),
                shared=len(success)-len({t['donor']['body'] for t in success}),
                retained_donor=sum(t['donor']['active'] for t in success),
                unchanged=sum(t['value'] == t['donor']['value'] for t in success),
                body_value_equal=sum(t['body'] == t['value'] for t in tiles),
                repeated_sample=sum(np.count_nonzero(t['tile'] == t['value']) > 1 for t in tiles),
                decorations=sum(np.count_nonzero(t['tile'] != t['body']) - int(t['value'] != t['body']) for t in tiles),
                radii=tuple(sorted(t['radius'] for t in tiles)),
                layout=tuple(sorted((*t['direction'], t['radius'], t['active']) for t in tiles)),
                background=info['background'], neutral=info['neutral'])


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'split lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'pair counts')
    seen, infos = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            a, info = _parse(pair['input'])
            b = _grid(pair['output'])
            _require(a.shape == b.shape, 'dimension preservation')
            _require(_key(a) not in seen, 'duplicate input')
            seen.add(_key(a))
            # Scalar coordinate validation, including all unassigned pixels.
            expected, assigned = a.copy(), set()
            hr, hc = info['hub']
            successes = 0
            for t in info['tiles']:
                if not t['active'] or t['donor'] is None:
                    continue
                successes += 1
                donor = t['donor']
                dr, dc = t['direction']
                point = hr+dr, hc+dc
                expected[point] = donor['value']
                assigned.add(point)
                r, c = hr+2*t['radius']*dr, hc+2*t['radius']*dc
                for y in (-1, 0, 1):
                    for x in (-1, 0, 1):
                        expected[r+y, c+x] = donor['tile'][1-y, 1-x]
                        assigned.add((r+y, c+x))
            _require(successes >= 1, 'example has no successful request')
            _require(np.array_equal(expected, b), 'wrong samples, payloads, or copying geometry')
            _require(len(assigned) == 10*successes, 'packet/hub overlap')
            _require(int(np.count_nonzero(a != b)) == 10*successes, 'changed-cell count')
            _require(np.array_equal(a[a != info['background']], b[a != info['background']]),
                     'original tiles/spokes/hub changed')
            _require(set(np.unique(a)) == set(np.unique(b)), 'new color introduced')
            infos[split].append(info)
    _coverage(puzzle, infos)


def self_test():
    """Bounded semantic and diversity audit over 42 ordinary seeds."""
    puzzles, rows, structures, shapes = [], [], set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                a, info = _parse(pair['input'])
                _require(np.array_equal(solve(a.copy()), pair['output']), 'solve disagreement')
                rows.append(_features(info))
                shapes.add(a.shape)
                labels = {}
                norm = tuple(labels.setdefault(int(v), len(labels)) for v in a.flat)
                structures.add((a.shape, norm))
    signatures = [tuple(_key(q['input']) for s in ('train', 'test') for q in p[s]) for p in puzzles]
    _require(len(set(signatures)) == 42, 'duplicate seeds')
    _require(signatures[0] == tuple(_key(q['input']) for s in ('train', 'test') for q in generate_puzzle(0)[s]),
             'nondeterminism')
    _require(len(structures) >= 150 and len(shapes) >= 45, 'geometry diversity')
    _require({f['tiles'] for f in rows} == set(range(3, 9)), 'tile-count diversity')
    _require({f['active'] for f in rows} == set(range(1, 7)), 'spoke-count diversity')
    _require(len({f['layout'] for f in rows}) >= 150, 'ray-layout diversity')
    _require({f['background'] for f in rows} == set(range(10)), 'background coverage')
    _require({f['neutral'] for f in rows} == set(range(10)), 'neutral coverage')
    _require(len({f['decorations'] for f in rows}) >= 10, 'motif richness diversity')
    for feature in ('failed', 'unselected_compatible', 'remote', 'shared', 'retained_donor', 'unchanged', 'repeated_sample'):
        _require(sum(f[feature] > 0 for f in rows) >= 20, 'coverage: ' + feature)
    for nt in range(2, 11):
        for ne in range(1, min(nt-1, 4)+1):
            p = generate_puzzle(42000+nt*10+ne, nt, ne)
            _require(len(p['train']) == nt and len(p['test']) == ne, 'counts ignored')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.0, 1), (True, 1), (4, True)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for nt, ne in ((None, 1), (None, 4), (2, None), (10, None), (None, None)):
        validate_generated_puzzle(generate_puzzle(17, nt, ne))
    # Corrupt three semantically different regions; the validator must reject all.
    base = puzzles[0]
    a, info = _parse(base['test'][0]['input'])
    t = next(t for t in info['tiles'] if t['active'] and t['donor'] is not None)
    dr, dc = t['direction']
    hr, hc = info['hub']
    for point in ((hr+dr, hc+dc), (hr+2*t['radius']*dr, hc+2*t['radius']*dc), t['center']):
        bad = {s: [{k: np.asarray(v).copy() for k, v in q.items()} for q in base[s]] for s in base}
        out = bad['test'][0]['output']
        out[point] = (int(out[point])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('corruption accepted')
    return dict(seeds=42, pairs=len(rows), shapes=len(shapes), structures=len(structures),
                layouts=len({f['layout'] for f in rows}),
                tile_counts=sorted({f['tiles'] for f in rows}),
                spoke_counts=sorted({f['active'] for f in rows}),
                semantic_verification='generator-owned checks only')
