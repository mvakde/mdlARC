"""Unanimous dye circuits: simultaneous geometric delivery and routed vetoes.

No parent module, saved example, file, or seed is used by solve().
"""
from collections import deque
import hashlib
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
DISTANCE = 5


def _groups(cells, steps=D8):
    left = set(cells)
    result = []
    while left:
        root = min(left)
        left.remove(root)
        group, todo = {root}, [root]
        for r, c in todo:
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in left:
                    left.remove(p)
                    group.add(p)
                    todo.append(p)
        result.append(group)
    return result


def _near(cells, steps=D8):
    cells = set(cells)
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps} - cells


def _arrow(tip, vector):
    r, c = tip
    dr, dc = vector
    if dr == 0:
        wings = ((-1, -dc), (1, -dc))
    elif dc == 0:
        wings = ((-dr, -1), (-dr, 1))
    else:
        wings = ((-dr, 0), (0, -dc))
    return frozenset((r + a, c + b) for a, b in ((0, 0),) + wings)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _decode(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    bg = int(np.bincount(a.ravel(), minlength=10).argmax())
    assert np.count_nonzero(a == bg) > a.size // 2, "Background must be dominant"
    foreground = set(map(tuple, np.argwhere(a != bg).tolist()))
    circuits, arrows = [], []
    for cells in _groups(foreground):
        colors = {int(a[p]) for p in cells}
        assert len(colors) == 1, "All objects must be monochrome and separated"
        color = colors.pop()
        if len(cells) == 3:
            candidates = [(p, d) for p in sorted(cells) for d in D8
                          if _arrow(p, d) == cells]
            assert len(candidates) == 1, "Not a unique arrow glyph"
            tip, vector = candidates[0]
            arrows.append(dict(tip=tip, vector=vector, cells=cells, color=color))
        else:
            assert len(cells) >= 4
            circuits.append(dict(cells=cells, color=color, holes=[]))
    assert circuits and arrows
    colors = {s['color'] for s in circuits}
    assert len(colors) == 1
    tube = colors.pop()
    assert all(s['color'] != tube for s in arrows)
    owner = {p: i for i, s in enumerate(circuits) for p in s['cells']}
    h, w = a.shape
    for cells in _groups(map(tuple, np.argwhere(a == bg).tolist()), D4):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            continue
        boundary = _near(cells, D4)
        owners = {owner.get(p, -1) for p in boundary}
        assert len(owners) == 1 and -1 not in owners, "Ambiguous cavity ownership"
        i = owners.pop()
        circuits[i]['holes'].append(cells)
    assert all(s['holes'] for s in circuits), "Each circuit needs a closed cavity"
    return dict(grid=a, bg=bg, tube=tube, circuits=circuits, arrows=arrows, owner=owner)


def _votes(info, distance=DISTANCE, nearest=False):
    votes = [[] for _ in info['circuits']]
    for arrow in info['arrows']:
        r, c = arrow['tip']
        dr, dc = arrow['vector']
        p = (r + distance * dr, c + distance * dc)
        if nearest:
            # A deliberately wrong direction-free shortcut, used only in checks.
            p = min(info['owner'], key=lambda q: ((q[0]-r)**2 + (q[1]-c)**2, q))
        if p in info['owner']:
            votes[info['owner'][p]].append((arrow['color'], p))
    return votes


def _paint(info, votes, direct_only=False, majority=False):
    out = info['grid'].copy()
    for circuit, arrivals in zip(info['circuits'], votes):
        colors = {col for col, p in arrivals}
        if not arrivals or (len(colors) != 1 and not majority):
            continue
        if majority:
            counts = {c: sum(col == c for col, p in arrivals) for c in colors}
            top = max(counts.values())
            winners = [c for c, n in counts.items() if n == top]
            if len(winners) != 1:
                continue
            color = winners[0]
        else:
            color = next(iter(colors))
        for hole in circuit['holes']:
            if direct_only and not any(p in _near(hole, D4) for col, p in arrivals):
                continue
            for p in hole:
                out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read all conventions from the input; deliver once, then apply unanimity."""
    info = _decode(input_grid)
    return _paint(info, _votes(info))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _cavity(rng):
    """Small but independently proportioned connected row profiles."""
    h, w = int(rng.integers(1, 4)), int(rng.integers(1, 5))
    kind = int(rng.integers(5))
    cells = set()
    shift = 0
    for r in range(h):
        if kind == 1 and r:
            shift += int(rng.integers(-1, 2)) if w >= 2 else 0
        elif kind == 2:
            shift = r // 2 if w > 1 else 0
        width = w - 1 if kind == 3 and h >= 3 and r == 1 and w > 1 else w
        cells.update((r, c) for c in range(shift, shift + width))
    if kind == 4:
        cells = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}
    if rng.random() < .5:
        cells = {(c, r) for r, c in cells}
    boundary = _near(cells, D4)
    # Both minimal shells and square-corner shells are legal.
    if kind == 0 and rng.random() < .5:
        boundary = _near(cells)
    mr, mc = min(r for r, c in boundary), min(c for r, c in boundary)
    move = lambda points: {(r - mr, c - mc) for r, c in points}
    return move(cells), move(boundary)


def _path(rng, starts, goals, allowed):
    starts = sorted(starts)
    rng.shuffle(starts)
    todo = deque(starts)
    previous = {p: None for p in starts}
    steps = [D8[int(i)] for i in rng.permutation(8)]
    end = None
    while todo:
        p = todo.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in steps:
            q = (p[0] + dr, p[1] + dc)
            if q in allowed and q not in previous:
                previous[q] = p
                todo.append(q)
    if end is None:
        return None
    result = set()
    while end is not None:
        result.add(end)
        end = previous[end]
    return result


def _networks(rng, counts, shape):
    canvas = {(r, c) for r in range(shape[0]) for c in range(shape[1])}
    n = len(counts)
    # Random separated hubs constrain packing, not semantic membership or strips.
    hubs = []
    for i in range(n):
        candidates = [(int(rng.integers(4, shape[0]-4)),
                       int(rng.integers(4, shape[1]-4))) for _ in range(50)]
        if not hubs:
            hubs.append(candidates[0])
        else:
            p = max(candidates, key=lambda q: min((q[0]-r)**2 + (q[1]-c)**2 for r, c in hubs))
            if min(max(abs(p[0]-r), abs(p[1]-c)) for r, c in hubs) < 10:
                return None
            hubs.append(p)
    regions, occupied = [], set()
    for i, count in enumerate(counts):
        for j in range(count):
            inner, shell = _cavity(rng)
            placed = False
            for _ in range(75):
                r = hubs[i][0] + int(rng.integers(-7, 8))
                c = hubs[i][1] + int(rng.integers(-7, 8))
                ins = {(r+a, c+b) for a, b in inner}
                edge = {(r+a, c+b) for a, b in shell}
                cells = ins | edge
                if not cells <= canvas or cells & (occupied | _near(occupied)):
                    continue
                # Keep the sampling neighborhoods loose and two-dimensional.
                if any((r-hubs[i][0])**2 + (c-hubs[i][1])**2 >
                       (r-hubs[k][0])**2 + (c-hubs[k][1])**2 + 15
                       for k in range(n) if k != i):
                    continue
                regions.append(dict(inner=ins, shell=edge, owner=i))
                occupied |= cells
                placed = True
                break
            if not placed:
                return None
    all_loops = set().union(*(s['inner'] | s['shell'] for s in regions))
    networks = [set() for _ in counts]
    for i in rng.permutation(n):
        i = int(i)
        own = [s for s in regions if s['owner'] == i]
        order = rng.permutation(len(own))
        tree = set(own[int(order[0])]['shell'])
        other_networks = set().union(*networks)
        for j in order[1:]:
            target = own[int(j)]
            other_loops = all_loops - target['inner'] - target['shell'] - tree
            forbidden = all_loops | _near(other_loops) | other_networks | _near(other_networks)
            allowed = canvas - forbidden
            starts = (_near(target['shell']) & allowed)
            goals = (_near(tree) & allowed)
            path = _path(rng, starts, goals, allowed)
            if path is None:
                return None
            tree |= target['shell'] | path
        # At least one tail provides an injection point separate from all shells.
        own_loops = set().union(*(s['inner'] | s['shell'] for s in own))
        forbidden = all_loops | _near(all_loops - own_loops) | other_networks | _near(other_networks)
        allowed = canvas - forbidden
        goals = _near(tree) & allowed
        candidates = sorted(p for p in allowed if 3 <= min(max(abs(p[0]-r), abs(p[1]-c))
                                                        for r, c in tree) <= 7)
        if not candidates or not goals:
            return None
        rng.shuffle(candidates)
        path = _path(rng, candidates[:1], goals, allowed)
        if path is None:
            return None
        tree |= path
        if tree & (other_networks | _near(other_networks)):
            return None
        networks[i] = tree
    return regions, networks


def _shot_options(networks, regions, shape):
    occupied = set().union(*networks)
    forbidden = occupied | _near(occupied)
    shell = set().union(*(s['shell'] for s in regions))
    options = [[] for _ in networks]
    for i, cells in enumerate(networks):
        for p in sorted(cells):
            for dr, dc in D8:
                tip = (p[0] - DISTANCE*dr, p[1] - DISTANCE*dc)
                glyph = _arrow(tip, (dr, dc))
                if any(not _inside(q, shape) for q in glyph) or glyph & forbidden:
                    continue
                if any((tip[0] + k*dr, tip[1] + k*dc) in occupied for k in (4, 6)):
                    continue
                options[i].append(dict(tip=tip, vector=(dr, dc), cells=glyph,
                                       endpoint=p, shell=p in shell))
    return options


def _scene(rng, counts, plans, miss, palette, shape, coincide=False):
    built = _networks(rng, counts, shape)
    if built is None:
        return None
    regions, networks = built
    options = _shot_options(networks, regions, shape)
    all_cells = set().union(*networks)
    reserved = all_cells | _near(all_cells)
    interiors = set().union(*(s['inner'] for s in regions))
    selected = []
    # Place difficult routes first; every planned vote remains visible and exact.
    for i in sorted(range(len(plans)), key=lambda k: -len(plans[k])):
        endpoints = set()
        for j, color in enumerate(plans[i]):
            candidates = [o for o in options[i] if not (o['cells'] & (reserved | interiors))]
            if j:
                repeat = coincide and rng.random() < .35
                filtered = [o for o in candidates if (o['endpoint'] in endpoints) == repeat]
                candidates = filtered or [o for o in candidates if o['endpoint'] not in endpoints]
            elif i == 0:
                candidates = [o for o in candidates if not o['shell']]
            if not candidates:
                return None
            shot = dict(_pick(rng, candidates), color=color)
            selected.append(shot)
            endpoints.add(shot['endpoint'])
            reserved |= set(shot['cells']) | _near(shot['cells'])
    if miss is not None:
        # A wrong-way arrow near an unsupplied circuit disproves proximity filling.
        owner, color = miss
        candidates = []
        for r in range(shape[0]):
            for c in range(shape[1]):
                if min(max(abs(r-y), abs(c-x)) for y, x in networks[owner]) > 4:
                    continue
                nearest = min(range(len(networks)), key=lambda k:
                              min((r-y)**2 + (c-x)**2 for y, x in networks[k]))
                if nearest != owner:
                    continue
                for dr, dc in D8:
                    endpoint = (r + 5*dr, c + 5*dc)
                    glyph = _arrow((r, c), (dr, dc))
                    if (not _inside(endpoint, shape) or endpoint in all_cells or
                            any(not _inside(q, shape) for q in glyph) or
                            glyph & (reserved | interiors)):
                        continue
                    candidates.append(dict(tip=(r, c), vector=(dr, dc), cells=glyph,
                                           endpoint=endpoint, color=color, shell=False))
        if not candidates:
            return None
        selected.append(_pick(rng, candidates))
    bg, tube = palette[:2]
    a = np.full(shape, bg, dtype=np.int8)
    for cells in networks:
        for p in cells:
            a[p] = tube
    for arrow in selected:
        for p in arrow['cells']:
            a[p] = arrow['color']
    b = a.copy()
    # Construct expected labels from intended ownership and deliveries, not solve.
    for region in regions:
        colors = set(plans[region['owner']])
        if len(colors) == 1:
            for p in region['inner']:
                b[p] = next(iter(colors))
    try:
        info = _decode(a)
        decoded_holes = {frozenset(h) for s in info['circuits'] for h in s['holes']}
        assert decoded_holes == {frozenset(s['inner']) for s in regions}
        assert len(info['circuits']) == len(networks)
        assert np.array_equal(_paint(info, _votes(info)), b)
    except AssertionError:
        return None
    return {'input': a, 'output': b}


def _pair(rng, kind, test, palette):
    paint = palette[2:]
    for attempt in range(180):
        if kind == 'single':
            counts, plans, miss = [2, 1], [[paint[0]], []], (1, paint[1])
        elif kind == 'veto':
            counts = [int(rng.integers(1, 3)), int(rng.integers(1, 3))]
            plans, miss = [[paint[1], paint[1]], [paint[0], paint[0], paint[2]]], None
        else:
            n = int(rng.integers(1 if test else 2, 4))
            counts = [int(rng.integers(1, 3)) for _ in range(n)]
            if test:
                counts[0] = int(rng.integers(3, 5))
            plans = [[paint[0]] * int(rng.integers(1, 4))]
            for i in range(1, n):
                mode = int(rng.integers(3))
                plans.append([] if mode == 0 else [paint[i]] * int(rng.integers(1, 4))
                             if mode == 1 else [paint[i], paint[-1]])
            miss = None
        # Randomize circuit location independently of its logical status.
        floor = 17 if len(counts) == 1 else 19 if sum(counts) <= 3 else 22
        shape = (int(rng.integers(floor, 31)), int(rng.integers(floor, 31)))
        pair = _scene(rng, counts, plans, miss, palette, shape, coincide=kind == 'mixed')
        if pair is not None:
            return pair
    raise RuntimeError('Bounded scene placement exhausted')


def _check_counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _matching_distances(pairs):
    parsed = [(_decode(p['input']), np.asarray(p['output'])) for p in pairs]
    return [d for d in range(-29, 30) if d and all(
        np.array_equal(_paint(info, _votes(info, distance=d)), out) for info, out in parsed)]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    # The two required lessons suffice even for the smallest legal request.
    for retry in range(16):
        palette = [int(c) for c in rng.permutation(10)]
        lessons = [_pair(rng, kind, False, palette) for kind in ('single', 'veto')]
        if _matching_distances(lessons) == [DISTANCE]:
            break
    else:
        raise RuntimeError('Could not expose a unique projection distance')
    puzzle = {'train': lessons, 'test': []}
    seen = {_key(p['input']) for p in lessons}
    for split, count in (('train', int(num_train)-2), ('test', int(num_test))):
        for index in range(count):
            for attempt in range(12):
                # Palette roles are input-visible and vary independently of geometry.
                colors = [int(c) for c in rng.permutation(10)]
                pair = _pair(rng, 'mixed', split == 'test', colors)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not construct distinct inputs')
    order = rng.permutation(int(num_train))
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    info = _decode(pair['input'])
    votes = _votes(info)
    accepted = [i for i, v in enumerate(votes) if len({c for c, p in v}) == 1]
    conflict = [i for i, v in enumerate(votes) if len({c for c, p in v}) > 1]
    majority_veto = any(max(sum(c == col for col, p in votes[i]) for c, q in votes[i])
                        > len(votes[i]) / 2 for i in conflict)
    return dict(shape=tuple(info['grid'].shape), bg=info['bg'], tube=info['tube'],
                circuits=len(votes), arrows=len(info['arrows']),
                holes=[len(s['holes']) for s in info['circuits']],
                areas=[len(h) for s in info['circuits'] for h in s['holes']],
                directions=tuple(sorted({s['vector'] for s in info['arrows']})),
                accepted=len(accepted), conflict=len(conflict), majority_veto=majority_veto,
                empty=sum(not v for v in votes),
                single=any(len(votes[i]) == 1 for i in accepted),
                agree=any(len(votes[i]) >= 2 for i in accepted),
                separated=any(len({p for c, p in v}) >= 2 for v in votes),
                coincident=any(len({p for c, p in v}) < len(v) for v in votes),
                missed=len(info['arrows']) - sum(map(len, votes)),
                remote=not np.array_equal(_paint(info, votes, direct_only=True), _paint(info, votes)),
                proximity_fails=not np.array_equal(_paint(info, _votes(info, nearest=True)), _paint(info, votes)))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = np.asarray(pair['input']), np.asarray(pair['output'])
            info = _decode(a)
            assert b.shape == a.shape and np.issubdtype(b.dtype, np.integer)
            assert np.all((0 <= b) & (b <= 9))
            key = _key(a)
            assert key not in seen
            seen.add(key)
            assert all(_inside((s['tip'][0] + 5*s['vector'][0],
                                s['tip'][1] + 5*s['vector'][1]), a.shape) for s in info['arrows'])
            assert np.array_equal(solve(a.copy()), b), 'Input-derived delivery/fill mismatch'
            assert np.array_equal(a[a != info['bg']], b[a != info['bg']])
            assert not np.array_equal(a, b)
            assert set(np.unique(b)) <= set(np.unique(a))
            features[split].append(_features(pair))
    train = features['train']
    assert len({int(c) for p in puzzle['train'] for c in np.asarray(p['output'])[
        np.asarray(p['input']) != np.asarray(p['output'])]}) >= 2, 'Teach copying two accepted dyes'
    for feature in ('single', 'agree', 'majority_veto', 'empty', 'remote', 'proximity_fails', 'separated', 'missed'):
        assert any(f[feature] for f in train), f'Missing teaching evidence: {feature}'
    assert _matching_distances(puzzle['train']) == [5], 'Projection length is ambiguous'
    assert max(n for f in train for n in f['holes']) <= 2
    for pair in puzzle['test']:
        info = _decode(pair['input'])
        votes = _votes(info)
        assert any(len(s['holes']) >= 3 and len({c for c, p in v}) == 1
                   for s, v in zip(info['circuits'], votes)), 'Missing productive fan-out shift'


def _semantic_regressions():
    """Tiny hand-constructed cases, unrelated to parent or sampled layouts."""
    a = np.zeros((28, 28), dtype=np.int8)
    holes = [{(r, c) for r in range(lo, lo+2) for c in (22, 23)} for lo in (8, 14)]
    detached = {(r, c) for r in (22, 23) for c in (3, 4)}
    for hole in holes + [detached]:
        for p in _near(hole, D4):
            a[p] = 5
    a[8, 12:21] = 5
    a[11:13, 22] = 5
    # Two distinct wire impacts, one shared endpoint, and a two-to-one veto.
    shots = [((8, 9), (0, 1)), ((13, 16), (-1, 0)), ((3, 14), (1, 0))]
    for tip, direction in shots[:2]:
        for p in _arrow(tip, direction):
            a[p] = 2
    out = solve(a)
    assert all(out[p] == 2 for hole in holes for p in hole)
    assert all(out[p] == 0 for p in detached)
    assert np.array_equal(out[a != 0], a[a != 0])
    for p in _arrow(*shots[2]):
        a[p] = 2
        out[p] = 2
    assert np.array_equal(solve(a), out), 'Coincident agreeing vote must not cancel'
    for p in _arrow(*shots[2]):
        a[p] = 7
        out[p] = 7
    assert np.array_equal(solve(a), a), 'Minority disagreement must veto the whole circuit'
    info = _decode(a)
    expected = _paint(info, _votes(info))
    info['arrows'].reverse()
    assert np.array_equal(_paint(info, _votes(info)), expected)
    # A ray crosses one circuit before reaching a different circuit at step five.
    a = np.zeros((24, 24), dtype=np.int8)
    for hole in ({(2, 9)}, {(15, 12)}):
        for p in _near(hole, D4):
            a[p] = 5
    a[2:9, 7] = 5
    a[8:16, 10] = 5
    for p in _arrow((8, 5), (0, 1)):
        a[p] = 2
    out = solve(a)
    assert out[2, 9] == 0 and out[15, 12] == 2
    assert np.count_nonzero(a != out) == 1
    for p in _arrow((8, 5), (0, 1)):
        a[p] = 0
    # The endpoint now lies inside a cavity, after crossing its tube: still a miss.
    for p in _arrow((15, 7), (0, 1)):
        a[p] = 2
    assert np.array_equal(solve(a), a)


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Streaming 42-seed audit, no generated files and no retained puzzle suite."""
    digests, shapes, directions, roles, structures, areas = set(), set(), set(), set(), set(), set()
    circuit_counts, arrow_counts, hole_shapes = set(), set(), set()
    _semantic_regressions()
    totals = dict(puzzles=0, pairs=0, conflict=0, agreement=0, remote=0, misses=0, coincident=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digests.add(_puzzle_digest(puzzle))
        totals['puzzles'] += 1
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                f = _features(pair)
                totals['pairs'] += 1
                totals['conflict'] += bool(f['conflict'])
                totals['agreement'] += bool(f['agree'])
                totals['remote'] += bool(f['remote'])
                totals['misses'] += bool(f['missed'])
                totals['coincident'] += bool(f['coincident'])
                shapes.add(f['shape']); directions.update(f['directions'])
                roles.add((f['bg'], f['tube'])); areas.update(f['areas'])
                circuit_counts.add(f['circuits']); arrow_counts.add(f['arrows'])
                # Store only short hashes, never reconstructible grid suites.
                mapping = {info['bg']: 0}
                norm = bytes(mapping.setdefault(int(c), len(mapping)) for c in info['grid'].flat)
                structures.add(hashlib.sha256(bytes(info['grid'].shape) + norm).hexdigest())
                for s in info['circuits']:
                    for hole in s['holes']:
                        mr = min(r for r, c in hole); mc = min(c for r, c in hole)
                        hole_shapes.add(tuple(sorted((r-mr, c-mc) for r, c in hole)))
                # Rotations/reflections and color bijections must commute with solving.
                transformed = np.rot90(np.fliplr(pair['input']))
                assert np.array_equal(solve(transformed), np.rot90(np.fliplr(pair['output'])))
                perm = np.array((9, 7, 5, 3, 1, 8, 6, 4, 2, 0), dtype=np.int8)
                assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
    assert len(digests) == 42 and len(shapes) >= 25 and len(structures) >= 180
    assert directions == set(D8) and len(roles) >= 30 and len(hole_shapes) >= 15
    assert circuit_counts >= {1, 2, 3} and len(arrow_counts) >= 5 and len(areas) >= 8
    # Determinism, minimum counts, mutation rejection and copy independence.
    puzzle = generate_puzzle(73, 2, 1)
    assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(73, 2, 1))
    bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    p = bad['test'][0]
    r, c = np.argwhere(p['input'] != p['output'])[0]
    p['output'][r, c] = p['input'][r, c]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Partial fill accepted')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid count accepted')
    return dict(totals, distinct_puzzles=len(digests), shapes=len(shapes),
                normalized_structures=len(structures), directions=len(directions),
                palette_roles=len(roles), cavity_shapes=len(hole_shapes),
                circuit_counts=sorted(circuit_counts), arrow_counts=sorted(arrow_counts),
                cavity_areas=sorted(areas))
