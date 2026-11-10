"""Voting petals: inward samples elect length-ranked radial half-turn echoes.

Only visible pixels determine the rule. No parent code or fixtures are imported.
"""
from collections import Counter, deque
from itertools import permutations

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
LENGTHS = (3, 5, 7, 9)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "colors must be 0..9")
    return a


def _signature(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.flat))


def _puzzle_signature(p):
    return tuple(tuple((_signature(q['input']), _signature(q['output']))
                       for q in p[s]) for s in ('train', 'test'))


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        todo, component = deque([first]), []
        while todo:
            r, c = todo.popleft()
            component.append((int(r), int(c)))
            for dr, dc in DIRECTIONS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        result.append(component)
    return result


def _bounds(center, shape):
    r, c = center
    h, w = shape
    return r - h // 2, r + h // 2 + 1, c - w // 2, c + w // 2 + 1


def _analyze(value):
    """Parse components, cap families and votes entirely from an input."""
    a = _grid(value)
    frequencies = np.bincount(a.ravel().astype(int), minlength=10)
    bg = int(np.argmax(frequencies))
    _require(np.count_nonzero(frequencies == frequencies[bg]) == 1,
             "background must be the unique modal color")
    components = _components(a != bg)
    markers = [p[0] for p in components if len(p) == 1]
    _require(len(markers) == 1, "expected one isolated center pixel")
    center = markers[0]
    carrier = int(a[center])
    petals = []
    for points in components:
        if len(points) == 1:
            continue
        r0, c0 = map(int, np.min(points, axis=0))
        r1, c1 = map(int, np.max(points, axis=0) + 1)
        h, w = r1 - r0, c1 - c0
        _require(h % 2 == w % 2 == 1 and h * w == len(points),
                 "petals must be full odd-sided rectangles")
        pc = ((r0 + r1 - 1) // 2, (c0 + c1 - 1) // 2)
        delta = (pc[0] - center[0], pc[1] - center[1])
        _require((delta[0] == 0) != (delta[1] == 0), "petal is off a cardinal ray")
        d = tuple(int(np.sign(n)) for n in delta)
        radius = max(map(abs, delta))
        length = w if d[0] else h
        _require((h if d[0] else w) == 3 and length in LENGTHS,
                 "petal must have thickness three and length 3/5/7/9")
        _require(3 <= radius <= 6 and length <= 2 * radius - 3,
                 "petal length/radius violates separation")
        caps = ((pc[0], c0), (pc[0], c1 - 1)) if d[0] else (
            (r0, pc[1]), (r1 - 1, pc[1]))
        cap = int(a[caps[0]])
        _require(cap == int(a[caps[1]]) and cap not in (bg, carrier),
                 "matching endpoint caps must differ from carrier and background")
        inward = (pc[0] - d[0], pc[1] - d[1])
        destination = (center[0] + 2 * delta[0], center[1] + 2 * delta[1])
        tr0, tr1, tc0, tc1 = _bounds(destination, (h, w))
        _require(0 <= tr0 < tr1 <= a.shape[0] and 0 <= tc0 < tc1 <= a.shape[1],
                 "complete echo must fit without clipping")
        _require(np.all(a[tr0:tr1, tc0:tc1] == bg), "echo destination is occupied")
        petals.append(dict(direction=d, radius=radius, length=length, cap=cap,
                           center=pc, inward=inward, sample=int(a[inward]),
                           source=(r0, r1, c0, c1), target=(tr0, tr1, tc0, tc1),
                           tile=a[r0:r1, c0:c1].copy()))
    _require(2 <= len(petals) <= 4, "expected two to four petals")
    _require(len({p['direction'] for p in petals}) == len(petals), "duplicate ray")
    families = {}
    for p in petals:
        families.setdefault(p['cap'], []).append(p)
    votes = Counter(p['sample'] for p in petals if p['sample'] != carrier)
    _require(set(votes) == set(families), "each family needs votes; samples must be caps or carrier")
    chosen = []
    for cap, group in families.items():
        _require(len(group) >= 2, "every family needs at least two competitors")
        _require(len({p['length'] for p in group}) == len(group), "within-family length tie")
        _require(1 <= votes[cap] <= len(group), "unary rank must exist")
        chosen.append(sorted(group, key=lambda p: -p['length'])[votes[cap] - 1])
    destinations = np.zeros(a.shape, dtype=bool)
    for p in petals:
        q = (center[0] + p['direction'][0], center[1] + p['direction'][1])
        _require(0 <= q[0] < a.shape[0] and 0 <= q[1] < a.shape[1] and a[q] == bg,
                 "central readout arm must be empty")
        r0, r1, c0, c1 = p['target']
        _require(not np.any(destinations[r0:r1, c0:c1]), "echo footprints overlap")
        destinations[r0:r1, c0:c1] = True
    return dict(grid=a, background=bg, carrier=carrier, center=center,
                petals=petals, families=families, ranks=dict(votes), selected=chosen)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compile inward votes, elect each family's ordinal, then draw its echo."""
    info = _analyze(input_grid)
    answer = info['grid'].copy()
    r, c = info['center']
    for p in info['petals']:
        dr, dc = p['direction']
        answer[r + dr, c + dc] = p['sample']
    for p in info['selected']:
        r0, r1, c0, c1 = p['target']
        answer[r0:r1, c0:c1] = p['tile'][::-1, ::-1]
    return np.ascontiguousarray(answer)


def _views(tile):
    return [np.rot90(tile, k) for k in range(4)] + [
        np.rot90(tile[:, ::-1], k) for k in range(4)]


def _check_pair(pair):
    """Check individual destination cells without calling solve's slice builder."""
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
    a, b = _grid(pair['input']), _grid(pair['output'])
    _require(a.shape == b.shape, "output dimensions changed")
    info = _analyze(a)
    r, c = info['center']
    changed = np.zeros(a.shape, dtype=bool)
    for p in info['petals']:
        dr, dc = p['direction']
        q = (r + dr, c + dc)
        _require(b[q] == a[p['inward']], "wrong central directional sample")
        changed[q] = True
    for p in info['selected']:
        sr, sc = p['center']
        tr, tc = r + 2 * (sr - r), c + 2 * (sc - c)
        h, w = p['tile'].shape
        for dy in range(-(h // 2), h // 2 + 1):
            for dx in range(-(w // 2), w // 2 + 1):
                _require(b[tr - dy, tc - dx] == a[sr + dy, sc + dx], "incorrect half-turn echo")
                changed[tr - dy, tc - dx] = True
    _require(np.array_equal(a[~changed], b[~changed]), "source, marker, or unused cells changed")
    expected_count = len(info['petals']) + sum(3 * p['length'] for p in info['selected'])
    _require(int(changed.sum()) == expected_count, "wrong write footprint")
    _require(np.array_equal(changed, a != b), "every written cell must replace background")
    _require(np.array_equal(np.unique(a), np.unique(b)), "palette changed")
    return info


def _training_evidence(infos):
    ranks = {n for info in infos for n in info['ranks'].values()}
    _require({1, 2} <= ranks, "training must show ordinals one and two")
    _require(any(1 < info['ranks'][cap] < len(group) for info in infos
                 for cap, group in info['families'].items()), "missing non-extreme selection")
    histories = {}
    for info in infos:
        for cap, n in info['ranks'].items():
            histories.setdefault(cap, set()).add(n)
    _require(any({1, 2} <= ns for ns in histories.values()), "missing same-cap rank contrast")
    _require(any(len(info['families']) == 2 and any(p['sample'] not in
                 (info['carrier'], p['cap']) for p in info['petals']) for info in infos),
             "training must show a cross-family vote")
    voting_directions = {p['direction'] for info in infos for p in info['petals']
                         if p['sample'] != info['carrier']}
    _require(voting_directions == set(DIRECTIONS), "every direction needs a voting demonstration")
    witnessed_axes = {bool(p['direction'][0]) for info in infos for p in info['selected']
                      if len({(t.shape, t.tobytes()) for t in _views(p['tile'])}) == 8}
    _require(witnessed_axes == {False, True}, "training must distinguish half-turns on both axes")
    _require(not _training_shortcuts(infos), "training permits a spatial/distance ordinal shortcut")
    return ranks


def _training_shortcuts(infos):
    """Test reading orders, circular orders and distance orders with tie orders.

    These are explicit alternative elections, not a claim to exhaust ARC rules.
    Source coordinates and the observed elected footprints are all that is used.
    """
    surviving = []
    for rule in range(26):
        consistent = True
        for info in infos:
            for cap, group in info['families'].items():
                if rule < 2:
                    ordered = sorted(group, key=lambda p: p['center'], reverse=bool(rule))
                else:
                    mode, subrule = divmod(rule - 2, 8)
                    start, reverse = divmod(subrule, 2)
                    def key(p):
                        circular = ((DIRECTIONS.index(p['direction']) - start)
                                    * (-1 if reverse else 1)) % 4
                        return circular if mode == 0 else (
                            p['radius'] * (1 if mode == 1 else -1), circular)
                    ordered = sorted(group, key=key)
                alternative = ordered[info['ranks'][cap] - 1]
                chosen = next(p for p in info['selected'] if p['cap'] == cap)
                consistent &= alternative['direction'] == chosen['direction']
        if consistent:
            surviving.append(rule)
    return surviving


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "invalid pair counts")
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            sig = _signature(pair['input'])
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            infos[split].append(info)
    ranks = _training_evidence(infos['train'])
    for info in infos['test']:
        _require(all(n in ranks or set(range(1, n)) <= ranks for n in info['ranks'].values()),
                 "test ordinal lacks a demonstrated progression")
        _require(max(info['ranks'].values()) > max(ranks)
                 or max(map(len, info['families'].values())) > max(
                     len(group) for tr in infos['train'] for group in tr['families'].values())
                 or max(p['radius'] for p in info['selected']) > max(
                     p['radius'] for tr in infos['train'] for p in tr['petals']),
                 "test must introduce an ordinal, population, or distance shift")
    # Transformation validity is separate from these teaching checks. solve
    # handles every legal scene, including ones outside the sampling curriculum.


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{name} must be an integer')
        if not lo <= int(value) <= hi:
            raise ValueError(f'{name} must be in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _pattern(rng, length, direction, cap, sample, carrier, bg, controls, witness):
    h, w = (3, length) if direction[0] else (length, 3)
    tile = np.full((h, w), carrier, dtype=np.int8)
    middle = (h // 2, w // 2)
    caps = ((h // 2, 0), (h // 2, w - 1)) if direction[0] else (
        (0, w // 2), (h - 1, w // 2))
    inward = (middle[0] - direction[0], middle[1] - direction[1])
    protected = set(caps) | {inward}
    positions = [(r, c) for r in range(h) for c in range(w) if (r, c) not in protected]
    colors = [x for x in range(10) if x not in (bg, carrier)]
    if witness:
        # One unique off-axis mark, positional caps and an inward sample give
        # all eight distinct D4 views, including when the petal is 3x3.
        auxiliary = int(rng.choice([x for x in colors if x not in controls]))
        rng.shuffle(positions)
        for q in positions:
            candidate = tile.copy()
            for p in caps:
                candidate[p] = cap
            candidate[inward] = sample
            candidate[q] = auxiliary
            if len({(v.shape, v.tobytes()) for v in _views(candidate)}) == 8:
                return candidate
        raise AssertionError('asymmetric petal witness unavailable')
    # Repeated votes elsewhere are harmless: only the spatial inward cell votes.
    count = int(rng.integers(0, max(2, len(positions) // 2) + 1))
    mode = int(rng.integers(3))
    rng.shuffle(positions)
    if mode == 1:
        axis = int(rng.integers(2))
        fixed = int(rng.integers((h, w)[axis]))
        positions.sort(key=lambda q: q[axis] != fixed)
    elif mode == 2:
        positions.sort(key=lambda q: abs(q[0] - inward[0]) + abs(q[1] - inward[1]))
    common = int(rng.choice(colors))
    for q in positions[:count]:
        tile[q] = int(rng.choice(colors)) if mode == 0 else common
    for p in caps:
        tile[p] = cap
    tile[inward] = sample
    return tile


def _make_input(rng, controls, populations, ranks, bg, carrier, *, max_radius=6,
                selected_axis=None, must_vote=(), cross_vote=False, witnesses=False,
                force_radius=False, common_radius=False):
    k = sum(populations)
    directions = [DIRECTIONS[int(i)] for i in rng.choice(4, k, replace=False)]
    family_ids = [f for f, count in enumerate(populations) for _ in range(count)]
    rng.shuffle(family_ids)
    if selected_axis is not None and not any(f == 0 and bool(d[0]) == selected_axis
                                            for f, d in zip(family_ids, directions)):
        i = next(i for i, d in enumerate(directions) if bool(d[0]) == selected_axis)
        j = family_ids.index(0)
        family_ids[i], family_ids[j] = family_ids[j], family_ids[i]
    lengths, selected_indices = [0] * k, []
    pool = [n for n in LENGTHS if n <= 2 * max_radius - 3]
    for f, pop in enumerate(populations):
        indices = [i for i in range(k) if family_ids[i] == f]
        values = list(map(int, rng.choice(pool, pop, replace=False)))
        rng.shuffle(values)
        chosen_length = sorted(values, reverse=True)[ranks[f] - 1]
        if f == 0 and selected_axis is not None:
            target_index = int(rng.choice([j for j, i in enumerate(indices)
                                          if bool(directions[i][0]) == selected_axis]))
            j = values.index(chosen_length)
            values[j], values[target_index] = values[target_index], values[j]
        for i, length in zip(indices, values):
            lengths[i] = length
            if length == chosen_length:
                selected_indices.append(i)
    votes = [controls[f] for f, n in enumerate(ranks) for _ in range(n)]
    votes += [carrier] * (k - len(votes))
    _require(len(votes) == k, 'rank plan needs too many votes')
    options = [v for v in sorted(set(permutations(votes)))
               if all(v[directions.index(d)] != carrier for d in must_vote)
               and (not cross_vote or any(v[i] not in (carrier, controls[family_ids[i]])
                                          for i in range(k)))]
    _require(options, 'no compatible vote placement')
    votes = options[int(rng.integers(len(options)))]
    radii = [max_radius if common_radius else int(rng.integers((n + 3) // 2, max_radius + 1))
             for n in lengths]
    if force_radius:
        radii[int(rng.choice(selected_indices))] = max_radius
    records, bounds = [], [(-1, 2, -1, 2)]
    for i, (d, length, radius) in enumerate(zip(directions, lengths, radii)):
        tile = _pattern(rng, length, d, controls[family_ids[i]], votes[i], carrier,
                        bg, controls, witnesses and i in selected_indices)
        pc = (radius * d[0], radius * d[1])
        records.append((pc, tile))
        bounds.extend((_bounds(pc, tile.shape), _bounds((2 * pc[0], 2 * pc[1]), tile.shape)))
    r0, r1 = min(b[0] for b in bounds), max(b[1] for b in bounds)
    c0, c1 = min(b[2] for b in bounds), max(b[3] for b in bounds)
    h, w = r1 - r0, c1 - c0
    # Independent margins include exact footprint fits and long rectangles.
    extra_h = int(rng.integers(0, 31 - h))
    extra_w = int(rng.integers(0, 31 - w))
    top, left = int(rng.integers(extra_h + 1)), int(rng.integers(extra_w + 1))
    cr, cc = top - r0, left - c0
    a = np.full((h + extra_h, w + extra_w), bg, dtype=np.int8)
    a[cr, cc] = carrier
    for (pr, pc), tile in records:
        rr0, rr1, cc0, cc1 = _bounds((cr + pr, cc + pc), tile.shape)
        a[rr0:rr1, cc0:cc1] = tile
    return a


def _distinct(a, seen):
    if _signature(a) not in seen:
        return a
    info = _analyze(a)
    source_order = [info['background'], info['carrier']] + [c for c in range(10)
                    if c not in (info['background'], info['carrier'])]
    for bg in range(10):
        for carrier in range(10):
            if bg == carrier:
                continue
            target_order = [bg, carrier] + [c for c in range(10) if c not in (bg, carrier)]
            mapping = np.empty(10, dtype=np.int8)
            mapping[source_order] = target_order
            candidate = mapping[a]
            if _signature(candidate) not in seen:
                return candidate
    raise AssertionError('90 distinct role assignments exhausted')


def _fallback_anchors(rng, palette, radius, rotation, reflection):
    """A constructive fallback, rather than a failure on an unlucky RNG stream.

    In the physical template below, the first election is south. The only
    circular orders that rank south second in N/E/S contradict one of the
    two elections in the second scene. Row-major orders rank east second.
    Equal radii reduce distance/tie alternatives to the same circular orders.
    Enumerating D4 placements guarantees the final physical template occurs.
    """
    bg, carrier, anchor, other = palette[:4]
    plans = (
        # direction, length, cap, vote, elected
        ((DIRECTIONS[0], 3, anchor, anchor, False),
         (DIRECTIONS[1], 7, anchor, carrier, False),
         (DIRECTIONS[2], 5, anchor, anchor, True)),
        ((DIRECTIONS[0], 3, anchor, carrier, False),
         (DIRECTIONS[1], 5, anchor, other, True),
         (DIRECTIONS[2], 3, other, other, True),
         (DIRECTIONS[3], 5, other, anchor, False)),
    )
    arrays = []
    for plan in plans:
        a = np.full((27, 27), bg, dtype=np.int8)
        a[13, 13] = carrier
        for d, length, cap, vote, elected in plan:
            tile = _pattern(rng, length, d, cap, vote, carrier, bg, [anchor, other], elected)
            pc = (13 + radius * d[0], 13 + radius * d[1])
            r0, r1, c0, c1 = _bounds(pc, tile.shape)
            a[r0:r1, c0:c1] = tile
        arrays.append(a)
    for flip in (False, True):
        for k in range(4):
            candidates = [np.ascontiguousarray(np.rot90(np.fliplr(a) if flip else a, k))
                          for a in arrays]
            physical = [np.rot90(np.fliplr(a) if reflection else a, rotation) for a in candidates]
            if not _training_shortcuts([_analyze(a) for a in physical]):
                return candidates
    raise AssertionError('constructive contrast template is inconsistent')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, m = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    bg, carrier, anchor, other = palette[:4]
    shift = str(rng.choice(('ordinal', 'radius', 'competition')))
    common_radius = bool(rng.integers(2))
    mixed_roles = bool(rng.integers(2))
    rotation, reflection = int(rng.integers(4)), bool(rng.integers(2))
    train_max = 5 if shift == 'radius' else 6
    # Two mandatory but randomly ordered teaching scenes suffice even at 2/1.
    for _ in range(128):
        axis = bool(rng.integers(2))
        a = _make_input(rng, [anchor], [3], [2], bg, carrier, max_radius=train_max,
                        selected_axis=axis, witnesses=True, common_radius=common_radius)
        active = {p['direction'] for p in _analyze(a)['petals'] if p['sample'] != carrier}
        b = _make_input(rng, [anchor, other], [2, 2], [1, 2], bg, carrier,
                        max_radius=train_max, selected_axis=not axis,
                        must_vote=set(DIRECTIONS) - active, cross_vote=True,
                        witnesses=True, common_radius=common_radius)
        physical = [np.rot90(np.fliplr(x) if reflection else x, rotation) for x in (a, b)]
        if not _training_shortcuts([_analyze(x) for x in physical]):
            break
    else:
        a, b = _fallback_anchors(rng, palette, train_max, rotation, reflection)
    train = [a, b]
    # Optional rank-three evidence permits a rank-four query; otherwise queries
    # extrapolate only the demonstrated one/two progression to rank three.
    for index in range(2, n):
        pop = 3 if shift in ('radius', 'competition') else int(rng.integers(3, 5))
        rank = 3 if index == 2 else int(rng.integers(1, min(pop, 3) + 1))
        roles = list(map(int, rng.permutation(10))) if mixed_roles else palette
        train.append(_make_input(rng, [roles[2]], [pop], [rank], roles[0], roles[1],
                                  max_radius=train_max, common_radius=common_radius))
    rng.shuffle(train)
    tests = []
    for _ in range(m):
        roles = list(map(int, rng.permutation(10))) if mixed_roles else palette
        if shift == 'ordinal':
            rank = 4 if n >= 3 else 3
            populations, ranks = [int(rng.integers(max(3, rank), 5))], [rank]
        elif shift == 'competition':
            populations, ranks = [4], [int(rng.integers(1, 4))]
        else:
            k = int(rng.integers(2, 5))
            populations = [2, 2] if k == 4 and rng.random() < .5 else [k]
            ranks = [int(rng.integers(1, min(pop, 3) + 1)) for pop in populations]
            while sum(ranks) > k:
                j = next(j for j, rank in enumerate(ranks) if rank > 1)
                ranks[j] -= 1
        tests.append(_make_input(rng, roles[2:2 + len(populations)], populations, ranks,
                                 roles[0], roles[1], force_radius=shift == 'radius',
                                 common_radius=common_radius))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, arrays in (('train', train), ('test', tests)):
        for a in arrays:
            a = np.ascontiguousarray(np.rot90(np.fliplr(a) if reflection else a, rotation))
            a = _distinct(a, seen)
            seen.add(_signature(a))
            puzzle[split].append(dict(input=a, output=solve(a)))
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    info = _analyze(pair['input'])
    return dict(shape=tuple(np.asarray(pair['input']).shape), center=info['center'],
                background=info['background'], carrier=info['carrier'],
                petals=len(info['petals']), families=len(info['families']),
                lengths=tuple(sorted(p['length'] for p in info['petals'])),
                radii=tuple(sorted(p['radius'] for p in info['petals'])),
                count_rank=tuple(sorted((len(group), info['ranks'][cap])
                                       for cap, group in info['families'].items())),
                abstentions=sum(p['sample'] == info['carrier'] for p in info['petals']),
                cross_votes=sum(p['sample'] not in (info['carrier'], p['cap'])
                                for p in info['petals']),
                selected_directions=tuple(sorted(p['direction'] for p in info['selected'])))


def self_test():
    """Bounded coverage, metamorphic and corruption checks over 42 fixed seeds."""
    puzzles, rows = [], []
    for seed in range(42):
        m = 1 + seed % 4
        n = max(m + 1, 2) + (seed // 4) % (11 - max(m + 1, 2))
        p = generate_puzzle(seed, n, m)
        validate_generated_puzzle(p)
        puzzles.append(p)
        rows.extend(_features(pair) for s in p for pair in p[s])
        for s in p:
            for pair in p[s]:
                a = pair['input'].copy()
                _require(np.array_equal(solve(a), pair['output']), 'solve differs')
                _require(np.array_equal(a, pair['input']), 'solve mutated input')
        # Contrastive ablation: all-petal replication writes into every unchosen
        # blank destination, while ignoring the votes picks a different longest.
        infos = [_analyze(pair['input']) for pair in p['train']]
        _require(any(any(info['ranks'][cap] != 1 for cap in info['families']) for info in infos),
                 'ignoring A never changes the selection')
        _require(all(len(info['selected']) < len(info['petals']) for info in infos),
                 'copy-all unexpectedly matches the election')
        _require(any(not np.array_equal(q['tile'], q['tile'][::-1, ::-1])
                     for info in infos for q in info['selected']),
                 'ignoring B half-turn is undetectable')
    _require(len({_puzzle_signature(p) for p in puzzles}) == 42, 'seed diversity collapsed')
    _require(len({f['shape'] for f in rows}) >= 50, 'canvas diversity collapsed')
    _require({f['petals'] for f in rows} == {2, 3, 4}, 'petal-count variation missing')
    _require({f['families'] for f in rows} == {1, 2}, 'family-count variation missing')
    _require({n for f in rows for n in f['lengths']} == set(LENGTHS), 'length variation missing')
    _require({n for f in rows for n in f['radii']} == {3, 4, 5, 6}, 'ray-distance variation missing')
    _require({f['background'] for f in rows} == set(range(10)), 'background roles omitted')
    _require({f['carrier'] for f in rows} == set(range(10)), 'carrier roles omitted')
    _require({(2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3), (4, 4)}
             <= {q for f in rows for q in f['count_rank']}, 'population/rank combinations omitted')
    _require(any(len(set(f['radii'])) > 1 for f in rows), 'unequal radii missing')
    _require(any(f['abstentions'] == 0 for f in rows) and any(f['abstentions'] >= 2 for f in rows),
             'abstention spectrum missing')
    _require(all(any(_features(q)['cross_votes'] for q in p['train']) for p in puzzles),
             'cross-family voting missing')
    _require(len({f['selected_directions'] for f in rows}) >= 8, 'spatial selections collapsed')
    _require(_puzzle_signature(generate_puzzle(995188254, 10, 4)) ==
             _puzzle_signature(generate_puzzle(995188254, 10, 4)), 'nondeterministic generation')
    # Exercise the normally unused constructive fallback in every final pose.
    for rotation in range(4):
        for reflection in (False, True):
            anchors = _fallback_anchors(np.random.default_rng(67), list(range(10)),
                                        5, rotation, reflection)
            physical = [np.rot90(np.fliplr(a) if reflection else a, rotation) for a in anchors]
            _training_evidence([_analyze(a) for a in physical])
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            validate_generated_puzzle(generate_puzzle(41, n, m))
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4., 1), (4, 1.), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    pair = puzzles[0]['train'][0]
    for k in range(4):
        for reflected in (False, True):
            a = np.rot90(np.fliplr(pair['input']) if reflected else pair['input'], k)
            b = np.rot90(np.fliplr(pair['output']) if reflected else pair['output'], k)
            mapping = np.array([9, 3, 6, 1, 4, 8, 0, 2, 7, 5])
            _require(np.array_equal(solve(mapping[a]), mapping[b]), 'geometric/color equivariance failed')
    # Corrupt a sample, an echo, an unchanged source, and an unused background.
    base = generate_puzzle(0)
    info = _analyze(base['test'][0]['input'])
    r, c = info['center']
    d = info['petals'][0]['direction']
    p = info['selected'][0]
    points = [(r + d[0], c + d[1]), (p['target'][0], p['target'][2]),
              p['center'], (r - 1, c - 1)]
    for point in points:
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in base[s]] for s in base}
        out = bad['test'][0]['output']
        out[point] = (int(out[point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, all legal counts, semantic and metamorphic checks)')
