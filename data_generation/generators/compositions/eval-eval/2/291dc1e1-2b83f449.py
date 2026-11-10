"""Switchyard manifests: route, join, orient, center, and list cargo ribbons.

Only NumPy and the standard library are used.  The solver reads every convention
from the input's coordinate key; generation state is never used for prediction.
"""

from collections import Counter
import numpy as np


_DIRS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _transform(a, config):
    axis, bottom, right = config
    b = a.T if axis else a
    if bottom:
        b = b[::-1]
    if right:
        b = b[:, ::-1]
    return b.copy()


def _normalize(block, config):
    """Proper physical rotation taking the fast guide direction to the right."""
    axis, bottom, right = config
    rotation = (3 if bottom else 1) if axis else (2 if right else 0)
    return np.rot90(_transform(block, config), rotation).copy()


def _neighbors(cell, cells):
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in _DIRS
            if (r + dr, c + dc) in cells]


def _paths(mask):
    """Trace every component, rejecting branches, singleton cells and cycles."""
    cells = set(map(tuple, np.argwhere(mask)))
    unseen = cells.copy()
    result = []
    while unseen:
        start = min(unseen)
        todo, component = [start], {start}
        unseen.remove(start)
        while todo:
            for q in _neighbors(todo.pop(), cells):
                if q in unseen:
                    unseen.remove(q)
                    component.add(q)
                    todo.append(q)
        degrees = {p: len(_neighbors(p, cells)) for p in component}
        assert all(d in (1, 2) for d in degrees.values()), 'Branch or isolated cell'
        ends = sorted(p for p, degree in degrees.items() if degree == 1)
        assert len(ends) == 2, 'Closed route'
        path, previous, current = [ends[0]], None, ends[0]
        while current != ends[1]:
            choices = [p for p in _neighbors(current, cells) if p != previous]
            assert len(choices) == 1
            previous, current = current, choices[0]
            path.append(current)
        assert len(path) == len(component)
        result.append(path)
    return sorted(result, key=lambda p: (p[-1], p[0]))


def _network_geometry(core, bg, track, bar):
    h, w = core.shape
    assert h >= 3 and h % 2 == 1 and w >= 7
    assert set(map(int, np.unique(core))) <= {bg, track, bar}
    assert np.all(np.isin(core[::2], [bg, track]))
    assert np.all(core[::2, 0] == track) and np.all(core[::2, -1] == track)
    assert np.all(np.isin(core[1::2], [bg, bar]))
    writes, centers, breaks = {}, [], []
    overlaps = 0
    for r in range(1, h, 2):
        columns = set(map(int, np.flatnonzero(core[r] == bar)))
        starts = sorted(c for c in columns if c - 1 not in columns)
        for c0 in starts:
            assert {c0, c0 + 1, c0 + 2} <= columns and c0 + 3 not in columns
            c = c0 + 1
            assert 2 <= c <= w - 3
            assert np.all(core[[r - 1, r + 1], c - 1:c + 2] == track)
            centers.append((r, c))
            for y in range(r - 1, r + 2):
                for x in range(c - 1, c + 2):
                    value = x != c
                    if (y, x) in writes:
                        assert writes[y, x] == value, 'Conflicting switches'
                        overlaps += 1
                    writes[y, x] = value
    assert centers, 'At least one switch is required'
    for r in range(0, h, 2):
        xs = list(map(int, np.flatnonzero(core[r] == bg)))
        assert all(2 <= x <= w - 3 for x in xs)
        assert all(b - a >= 3 for a, b in zip(xs, xs[1:]))
        breaks.extend((r, c) for c in xs)
    switched = core == track
    for cell, value in writes.items():
        switched[cell] = value
    paths = _paths(switched)
    assert len(paths) <= 15
    for p in paths:
        steps = {b[0] - a[0] for a, b in zip(p, p[1:]) if a[0] != b[0]}
        assert len(steps) <= 1, 'Route reverses along collection axis'
        for r, c in (p[0], p[-1]):
            assert r % 2 == 0
            assert c in (0, w - 1) or core[r, c - 1] == bg or core[r, c + 1] == bg
    return dict(paths=paths, centers=centers, breaks=breaks,
                overlaps=overlaps, switched=switched)


def _read_canonical(a, roles, config):
    """Read a coordinate-aligned scene, with fast right and slow down."""
    bg, origin, track, bar = roles
    assert len(set(roles)) == 4
    assert np.all(a[0, 1:] == track) and np.all(a[1:, 0] == bar)
    assert a[0, 0] == origin
    assert np.all(a[-1, 1:] == bg) and np.all(a[1:, -1] == bg)
    interior = a.copy()
    interior[0] = bg
    interior[:, 0] = bg
    locations = np.argwhere((interior == track) | (interior == bar))
    assert len(locations)
    r0, c0 = map(int, locations.min(axis=0))
    r1, c1 = map(int, locations.max(axis=0) + 1)
    assert r0 >= 2 and c0 >= 2 and r1 + 1 < a.shape[0] and c1 <= a.shape[1] - 1
    core = a[r0:r1, c0:c1].copy()
    info = _network_geometry(core, bg, track, bar)
    cargo = {}
    reconstruction = np.full(a.shape, bg, dtype=np.int8)
    reconstruction[0, 0] = origin
    reconstruction[0, 1:] = track
    reconstruction[1:, 0] = bar
    reconstruction[r0:r1, c0:c1] = core
    for y in range(0, core.shape[0], 2):
        r = r0 + y
        for side in (0, 1):
            cursor, direction = (c0 - 1, -1) if side == 0 else (c1, 1)
            columns = []
            while 0 < cursor < a.shape[1] - 1 and a[r, cursor] != bg:
                columns.append(cursor)
                cursor += direction
            if not columns:
                continue
            lo, hi = min(columns), max(columns) + 1
            block = a[r:r + 2, lo:hi].copy()
            assert block.shape[1] >= 2 and block.shape[1] % 2 == 0
            assert not np.any(np.isin(block, roles)), 'Cargo must use data colors'
            port = (y, 0 if side == 0 else core.shape[1] - 1)
            cargo[port] = block
            reconstruction[r:r + 2, lo:hi] = block
    assert cargo and np.array_equal(a, reconstruction), 'Unattached or malformed cargo'
    endpoints = {q for p in info['paths'] for q in (p[0], p[-1])}
    assert set(cargo) <= endpoints
    info.update(canonical=a, core=core, box=(r0, r1, c0, c1),
                roles=roles, config=config, cargo=cargo)
    return info


def _read(input_grid):
    a = _grid(input_grid)
    assert min(a.shape) >= 5
    candidates = []
    for bottom in (False, True):
        for right in (False, True):
            aligned = a[::-1] if bottom else a
            aligned = aligned[:, ::-1] if right else aligned
            origin, horizontal, vertical, bg = map(int,
                (aligned[0, 0], aligned[0, 1], aligned[1, 0], aligned[-1, -1]))
            if len({origin, horizontal, vertical, bg}) != 4:
                continue
            if not (np.all(aligned[0, 1:] == horizontal)
                    and np.all(aligned[1:, 0] == vertical)
                    and np.all(aligned[-1, 1:] == bg)
                    and np.all(aligned[1:, -1] == bg)):
                continue
            for axis in (0, 1):
                canonical = aligned.T.copy() if axis else aligned.copy()
                track, bar = (vertical, horizontal) if axis else (horizontal, vertical)
                try:
                    candidates.append(_read_canonical(canonical, (bg, origin, track, bar),
                                                      (axis, bottom, right)))
                except AssertionError:
                    pass
    assert len(candidates) == 1, 'The scene must have one input-only interpretation'
    return candidates[0]


def _rank(info, cell, mode):
    if mode == 'key':
        return cell
    # Ablation only: use physical top-to-bottom, left-to-right instead of the key.
    r, c = cell[0] + info['box'][0], cell[1] + info['box'][2]
    h, w = info['canonical'].shape
    axis, bottom, right = info['config']
    if axis:
        r, c, h, w = c, r, w, h
    return (h - 1 - r if bottom else r, w - 1 - c if right else c)


def _manifests(info, *, switches=True, rank_mode='key', normalize=True,
               far_first=False, reverse_cargo=False):
    paths = info['paths'] if switches else _paths(info['core'] == info['roles'][2])
    records = []
    for path in paths:
        ends = sorted((path[0], path[-1]), key=lambda p: _rank(info, p, rank_mode))
        sources = [p for p in ends[::-1] if p in info['cargo']]
        if not sources:
            continue
        if far_first:
            sources.reverse()
        blocks = [_normalize(info['cargo'][p], info['config']) if normalize
                  else info['cargo'][p].copy() for p in sources]
        if reverse_cargo:
            blocks = [b[:, ::-1] for b in blocks]
        records.append((ends[-1], np.concatenate(blocks, axis=1), sources))
    return sorted(records, key=lambda item: _rank(info, item[0], rank_mode))


def _render(info, *, center=True, enforce_bounds=True, **kwargs):
    records = _manifests(info, **kwargs)
    assert records
    width = max(block.shape[1] for _, block, _ in records)
    if enforce_bounds:
        assert width <= 30 and len(records) <= 15
    result = np.full((2 * len(records), width), info['roles'][0], dtype=np.int8)
    for i, (_, block, _) in enumerate(records):
        assert (width - block.shape[1]) % 2 == 0
        left = (width - block.shape[1]) // 2 if center else 0
        result[2 * i:2 * i + 2, left:left + block.shape[1]] = block
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible switches, cargo and coordinate key alone."""
    return _render(_read(input_grid))


def _base(lanes, width, centers, breaks, roles):
    bg, _, track, bar = roles
    core = np.full((2 * lanes - 1, width), bg, dtype=np.int8)
    core[::2] = track
    for r, c in centers:
        core[r, c - 1:c + 2] = bar
    for cell in breaks:
        core[cell] = bg
    return core


def _sample_network(rng, lanes, width, roles, level=False):
    centers, previous = [], []
    for separator in range(lanes - 1):
        wanted = int(rng.choice([0, 1, 1, 2, 3, 4]))
        chosen = []
        if not (level and separator == 0):
            for c in map(int, rng.permutation(np.arange(2, width - 2))):
                if len(chosen) >= wanted:
                    break
                if all(abs(c - p) >= 2 for p in previous) and all(abs(c - p) >= 4 for p in chosen):
                    chosen.append(c)
        centers.extend((2 * separator + 1, c) for c in sorted(chosen))
        previous = chosen
    if not centers:
        separator = 1 if level and lanes >= 3 else 0
        centers = [(2 * separator + 1, int(rng.integers(2, width - 2)))]
    breaks = []
    # Cut only segments whose two successive ports turn toward the same side.
    # This prevents hidden U-shaped gravitational traps, rather than forcing a
    # special break whenever a separator contains several bars.
    for row in range(0, 2 * lanes - 1, 2):
        ports = sorted((c, y) for y, c in centers if abs(y - row) == 1)
        for (a, side_a), (b, side_b) in zip(ports, ports[1:]):
            if side_a == side_b:
                breaks.append((row, int(rng.integers(a + 2, b - 1))))
    for _ in range(int(rng.choice([0, 0, 1, 2, 3]))):
        options = [(r, c) for r in range(0, 2 * lanes - 1, 2)
                   for c in range(2, width - 2)
                   if not (level and r == 0)
                   and all(abs(c - x) >= 2 for y, x in centers if abs(y - r) == 1)
                   and all(abs(c - x) >= 3 for y, x in breaks if y == r)]
        if options and lanes + len(breaks) < 15:
            breaks.append(options[int(rng.integers(len(options)))])
    core = _base(lanes, width, centers, breaks, roles)
    try:
        info = _network_geometry(core, roles[0], roles[2], roles[3])
    except AssertionError:
        return None
    return core, info


def _payload(rng, length, palette, patterned):
    mode = int(rng.integers(4))
    if mode == 0 and not patterned:
        return np.full((2, length), rng.choice(palette), dtype=np.int8)
    if mode == 1 and not patterned:
        return np.repeat(rng.choice(palette, size=(1, length)), 2, axis=0).astype(np.int8)
    if mode == 2 and not patterned:
        return np.repeat(rng.choice(palette, size=(2, 1)), length, axis=1).astype(np.int8)
    block = rng.choice(palette, size=(2, length)).astype(np.int8)
    if patterned:
        # Distinct row/column behavior without a repeated seed-level signature.
        block[0, 0], block[1, 0], block[0, -1] = palette[:3]
    return block


def _scene(rng, core, cargo, roles, config):
    bg, origin, track, bar = roles
    left = max((b.shape[1] for (r, c), b in cargo.items() if c == 0), default=0)
    right = max((b.shape[1] for (r, c), b in cargo.items() if c != 0), default=0)
    h, w = core.shape[0] + 4, core.shape[1] + left + right + 3
    assert h <= 30 and w <= 30
    extra_y, extra_x = int(rng.integers(min(4, 30 - h) + 1)), int(rng.integers(min(4, 30 - w) + 1))
    r0, c0 = 2 + int(rng.integers(extra_y + 1)), 2 + left + int(rng.integers(extra_x + 1))
    a = np.full((h + extra_y, w + extra_x), bg, dtype=np.int8)
    a[0, 0], a[0, 1:], a[1:, 0] = origin, track, bar
    a[r0:r0 + core.shape[0], c0:c0 + core.shape[1]] = core
    for (r, c), block in cargo.items():
        x = c0 - block.shape[1] if c == 0 else c0 + core.shape[1]
        a[r0 + r:r0 + r + 2, x:x + block.shape[1]] = block
    return _transform(a, config)


def _features(info):
    collision = empty_sink = level_join = 0
    records = _manifests(info)
    for path in info['paths']:
        sources = {p for p in (path[0], path[-1]) if p in info['cargo']}
        collision += len(sources) == 2 and path[0][0] != path[-1][0]
        empty_sink += len(sources) == 1 and path[0] in sources
        level_join += len(sources) == 2 and path[0][0] == path[-1][0]
    return dict(collision=collision, empty_sink=empty_sink, level_join=level_join,
                widths=tuple(b.shape[1] for _, b, _ in records),
                cargo_count=len(info['cargo']), routes=len(info['paths']),
                lanes=(info['core'].shape[0] + 1) // 2,
                switches=len(info['centers']), breaks=len(info['breaks']))


def _make_pair(rng, roles, config, lane_range, witness=0):
    for attempt in range(120):
        lanes = int(rng.integers(lane_range[0], lane_range[1] + 1))
        cap = int(rng.choice([2, 4, 6, 8]))
        width = int(rng.integers(7, 28 - 2 * cap))
        if attempt >= 100:
            # Constructive safety net: a staggered pair of switches supplies
            # cross-lane collection, and an untouched outer rail supplies a
            # level route. It obeys exactly the same geometry and solving rule.
            lanes = max(lanes, 4 if witness == 2 else 3 if witness else 2)
            cap, width = 2, 11
            centers = [(1, 3), (3, 5)] if lanes >= 3 else [(1, 3)]
            core = _base(lanes, width, centers, [], roles)
            layout = core, _network_geometry(core, roles[0], roles[2], roles[3])
        else:
            layout = _sample_network(rng, lanes, width, roles, level=witness == 2)
        if layout is None:
            continue
        core, topology = layout
        w = core.shape[1]
        outer = lambda p: p[1] in (0, w - 1)
        double_paths = [p for p in topology['paths']
                        if all(outer(q) for q in (p[0], p[-1])) and p[0][0] != p[-1][0]]
        level_paths = [p for p in topology['paths']
                       if all(outer(q) for q in (p[0], p[-1])) and p[0][0] == p[-1][0]]
        if (witness == 1 and not double_paths) or (witness == 2 and not level_paths):
            continue
        probability = float(rng.choice([.2, .4, .65, .9, 1.]))
        chosen = {q for p in topology['paths'] for q in (p[0], p[-1])
                  if outer(q) and rng.random() < probability}
        if witness == 1:
            double = double_paths[int(rng.integers(len(double_paths)))]
            chosen.update((double[0], double[-1]))
            singles = [p for p in topology['paths'] if p is not double and outer(p[0])]
            if not singles:
                continue
            single = singles[int(rng.integers(len(singles)))]
            chosen.discard(single[-1])
            chosen.add(single[0])
        elif witness == 2:
            level = level_paths[int(rng.integers(len(level_paths)))]
            chosen.update((level[0], level[-1]))
            if attempt >= 100:
                single = next(p for p in topology['paths'] if p is not level and outer(p[0]))
                chosen.discard(single[-1])
                chosen.add(single[0])
        if not chosen:
            options = [q for p in topology['paths'] for q in (p[0], p[-1]) if outer(q)]
            chosen.add(options[int(rng.integers(len(options)))])
        pool = [c for c in range(10) if c not in roles]
        rng.shuffle(pool)
        palette = pool[:int(rng.integers(3 if witness else 1, 7))]
        cargo = {}
        patterned_at = int(rng.integers(len(chosen)))
        for i, port in enumerate(sorted(chosen)):
            length = int(rng.choice(np.arange(2, cap + 1, 2)))
            same = [b for b in cargo.values() if b.shape[1] == length]
            if same and not witness and rng.random() < .3:
                cargo[port] = same[int(rng.integers(len(same)))].copy()
            else:
                cargo[port] = _payload(rng, length, palette, bool(witness and i == patterned_at))
        inp = _scene(rng, core, cargo, roles, config)
        info = _read(inp)
        out = _render(info)
        if witness:
            features = _features(info)
            if len(set(features['widths'])) < 2:
                continue
            if witness == 1 and (not features['empty_sink'] or
                                np.array_equal(out, _render(info, switches=False, enforce_bounds=False))):
                continue
        return dict(input=inp, output=out)
    raise RuntimeError('Could not construct a coverage demonstration within 120 attempts')


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _training_evidence(infos, outputs):
    features = [_features(d) for d in infos]
    assert any(f['collision'] for f in features), 'Need cross-lane cargo collection'
    assert any(f['empty_sink'] for f in features), 'Need collection at an empty terminal'
    assert any(f['level_join'] for f in features), 'Need the fast-axis tie rule'
    assert any(len(set(f['widths'])) > 1 for f in features), 'Need symmetric padding'
    for alternative in ({'switches': False}, {'rank_mode': 'physical'},
                        {'normalize': False}, {'center': False},
                        {'far_first': True}, {'reverse_cargo': True}):
        # A shortcut can itself exceed ARC bounds; that still distinguishes it.
        # Only these internal comparison arrays may exceed the output budget.
        assert any(not np.array_equal(out, _render(d, enforce_bounds=False, **alternative))
                   for d, out in zip(infos, outputs)), f'Unresolved shortcut: {alternative}'


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    configs = [(a, b, r) for a in (0, 1) for b in (False, True) for r in (False, True)]
    first = configs[int(rng.integers(8))]
    second = (1 - first[0], not first[1], not first[2])
    familiar = [first, second]
    held = [c for c in configs if c not in familiar]
    shift = int(rng.integers(3))
    if shift == 0:
        train_range = test_range = (3, 10)
    elif shift == 1:
        train_range, test_range = (3, 5), (6, 11)
    else:
        train_range, test_range = (6, 11), (3, 5)
    fixed_roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    varying_palette = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            for _ in range(80):
                roles = tuple(map(int, rng.choice(10, 4, replace=False))) if varying_palette else fixed_roles
                config = familiar[index] if split == 'train' and index < 2 else (
                    (held if split == 'test' and shift == 0 else familiar)[int(rng.integers(
                        len(held) if split == 'test' and shift == 0 else 2))])
                witness = index + 1 if split == 'train' and index < 2 else 0
                lane_range = train_range if split == 'train' else test_range
                # Ordinary orientation-shift examples may also have two lanes.
                if shift == 0 and not witness:
                    lane_range = (2, 11)
                pair = _make_pair(rng, roles, config, lane_range, witness)
                key = _signature(pair['input'])
                if key in seen:
                    continue
                if split == 'train' and index == 1:
                    try:
                        _training_evidence([_read(puzzle['train'][0]['input']), _read(pair['input'])],
                                           [puzzle['train'][0]['output'], pair['output']])
                    except AssertionError:
                        continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError('Could not construct distinct, informative examples')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, outputs = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _signature(inp)
            assert key not in seen, 'Duplicate input'
            seen.add(key)
            info = _read(inp)
            assert np.array_equal(out, _render(info)), 'Incorrect manifest'
            bg = info['roles'][0]
            before = Counter(int(c) for b in info['cargo'].values() for c in b.flat)
            after = Counter(map(int, out[out != bg]))
            assert before == after, 'Cargo cells must be conserved exactly'
            records = _manifests(info)
            assert out.shape == (2 * len(records), max(b.shape[1] for _, b, _ in records))
            assert all(p not in info['breaks'] for p in set(info['cargo']))
            if split == 'train':
                training.append(info)
                outputs.append(out)
    _training_evidence(training, outputs)
    assert {d['config'][0] for d in training} == {0, 1}
    assert {d['config'][1] for d in training} == {False, True}
    assert {d['config'][2] for d in training} == {False, True}


def self_test():
    """Bounded structural, semantic, metamorphic and 42-seed variation checks."""
    signatures, layouts, input_shapes, output_shapes = set(), set(), set(), set()
    features, configurations, roles_seen = [], set(), [set() for _ in range(4)]
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p[k]) for s in ('train', 'test')
                             for p in puzzle[s] for k in ('input', 'output')))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                info = _read(inp)
                assert np.array_equal(solve(inp.copy()), out)
                features.append(_features(info))
                configurations.add(info['config'])
                for i, color in enumerate(info['roles']):
                    roles_seen[i].add(color)
                layouts.add((info['core'].shape, (info['core'] == info['roles'][2]).tobytes(),
                             tuple((q, b.shape[1]) for q, b in sorted(info['cargo'].items()))))
                input_shapes.add(inp.shape)
                output_shapes.add(out.shape)
    assert len(signatures) == 42 and len(layouts) >= 150
    assert len(input_shapes) >= 60 and len(output_shapes) >= 35
    assert len(configurations) == 8 and all(len(s) == 10 for s in roles_seen)
    assert {f['lanes'] for f in features} >= set(range(3, 12))
    assert len({f['switches'] for f in features}) >= 10
    assert len({f['cargo_count'] for f in features}) >= 12
    assert len({f['breaks'] for f in features}) >= 5
    assert any(f['collision'] >= 2 for f in features)
    assert any(f['routes'] > len(f['widths']) for f in features), 'Need empty routes'
    # A direct, tiny route gives hand-checkable content and tie regressions.
    roles, config = (0, 9, 8, 7), (0, False, False)
    core = _base(4, 11, [(1, 3), (3, 5)], [], roles)
    a = np.array([[1, 2], [3, 4]], dtype=np.int8)
    b = np.array([[5, 6, 5, 6], [6, 5, 6, 5]], dtype=np.int8)
    c = np.full((2, 2), 2, dtype=np.int8)
    cargo = {(0, 10): a, (4, 0): b, (6, 0): c, (6, 10): a}
    inp = _scene(np.random.default_rng(100), core, cargo, roles, config)
    assert _read(inp)['overlaps'] == 1
    expected = np.zeros((4, 6), dtype=np.int8)
    expected[:2] = np.concatenate((b, a), axis=1)
    expected[2:, 1:5] = np.concatenate((a, c), axis=1)
    assert np.array_equal(solve(inp), expected)
    # Proper rotation of the entire input leaves the normalized manifest intact.
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(inp, turn)), expected)
    # Recoloring carries every visible convention, including the background.
    perm = np.array([4, 8, 0, 6, 7, 3, 9, 5, 2, 1], dtype=np.int8)
    assert np.array_equal(solve(perm[inp]), perm[expected])
    # No cache or seed dependence, and no mutation of the caller's array.
    frozen = inp.copy()
    solve(inp)
    assert np.array_equal(inp, frozen)
    # Exercise specification-valid members outside the sampler's usual range:
    # thirteen rails, fifteen routes, ten-column cargo and thirty output rows.
    extended_core = _base(13, 7, [(1, 3)], [(4, 3), (6, 3)], roles)
    extended_rng = np.random.default_rng(117)
    extended_cargo = {(r, x): extended_rng.choice([1, 2, 3, 4, 5, 6], size=(2, 10)).astype(np.int8)
                      for r in range(0, 25, 2) for x in (0, 6)}
    counts_before = Counter(int(c) for b in extended_cargo.values() for c in b.flat)
    for axis in (0, 1):
        for bottom in (False, True):
            for right in (False, True):
                extended = _scene(extended_rng, extended_core, extended_cargo, roles,
                                  (axis, bottom, right))
                answer = solve(extended)
                assert answer.shape == (30, 20)
                assert Counter(map(int, answer[answer != roles[0]])) == counts_before
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(9000 + 10 * train + test, train, test)
            assert len(puzzle['train']) == train and len(puzzle['test']) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal count accepted')
    original = generate_puzzle(1729, 2, 1)
    repeated = generate_puzzle(1729, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in ('train', 'test')
               for p, q in zip(original[s], repeated[s]) for k in ('input', 'output'))
    original['test'][0]['output'][0, 0] = (int(original['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(original)
    except AssertionError:
        pass
    else:
        raise AssertionError('Damaged label accepted')
    # Geometry rejection is separate from checking generated answers.
    invalid = [(3, 9, [(1, 2), (3, 3)], []),
               (3, 9, [(1, 3), (3, 3)], []),
               (2, 11, [(1, 3), (1, 7)], [])]
    for lanes, width, centers, breaks in invalid:
        try:
            _network_geometry(_base(lanes, width, centers, breaks, roles), 0, 8, 7)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid rewiring accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'layouts': len(layouts),
            'input_shapes': len(input_shapes), 'output_shapes': len(output_shapes),
            'configurations': len(configurations), 'extended_solver_cases': 8}


if __name__ == '__main__':
    print(self_test())
