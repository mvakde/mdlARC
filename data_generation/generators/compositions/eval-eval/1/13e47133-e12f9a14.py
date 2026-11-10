"""Contour races: incomplete-ring sources compete along erosion-depth tracks.

Only standard-library modules and NumPy are used. No files are read or written
by generation, solving, validation, or self_test.
"""

from collections import deque
import hashlib
import numbers

import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b)
RING = tuple((r, c) for r in range(-1, 3) for c in range(-1, 3)
             if r in (-1, 2) or c in (-1, 2))
INF = 10000
ALTERNATIVES = ('box', 'four', 'flood', 'ring', 'air', 'four_walk', 'opaque')


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int16, copy=False)


def _components(mask, directions=D4):
    h, w = mask.shape
    labels = np.full(mask.shape, -1, dtype=np.int16)
    groups = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        index = len(groups)
        queue = [(r, c)]
        labels[r, c] = index
        for a, b in queue:
            for dr, dc in directions:
                x, y = a + dr, b + dc
                if (0 <= x < h and 0 <= y < w and mask[x, y]
                        and labels[x, y] < 0):
                    labels[x, y] = index
                    queue.append((x, y))
        groups.append(queue)
    return labels, groups


def _erosion(mask, diagonal=True):
    """Repeated erosion; off-canvas positions are outside the room."""
    current = mask.copy()
    result = np.full(mask.shape, -1, dtype=np.int16)
    h, w = mask.shape
    d = 0
    while np.any(current):
        pad = np.pad(current, 1)
        inner = current.copy()
        for dr, dc in (D8 if diagonal else D4):
            inner &= pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        result[current & ~inner] = d
        current = inner
        d += 1
    return result


def _bfs_depth(mask):
    """Distance implementation independent of repeated erosion."""
    h, w = mask.shape
    result = np.full(mask.shape, -1, dtype=np.int16)
    q = deque()
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if any(not (0 <= r + dr < h and 0 <= c + dc < w
                        and mask[r + dr, c + dc]) for dr, dc in D8):
            result[r, c] = 0
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in D8:
            x, y = r + dr, c + dc
            if (0 <= x < h and 0 <= y < w and mask[x, y]
                    and result[x, y] < 0):
                result[x, y] = result[r, c] + 1
                q.append((x, y))
    return result


def _parse(grid):
    a = _grid(grid)
    h, w = a.shape
    wall = int(a[0, 0])
    assert (np.all(a[0] == wall) and np.all(a[-1] == wall)
            and np.all(a[:, 0] == wall) and np.all(a[:, -1] == wall)), "missing outer frame"
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert bg != wall and np.count_nonzero(counts == counts[bg]) == 1, "background must be the unique mode"
    hardware = np.zeros(a.shape, dtype=bool)
    fixed = a != bg
    sources = []
    seeds = []
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(a[r, c])
            if color in (bg, wall) or not np.all(a[r:r + 2, c:c + 2] == color):
                continue
            assert 2 <= r <= h - 4 and 2 <= c <= w - 4, "sources must fit inside the outer frame"
            assert not np.any(hardware[r - 1:r + 3, c - 1:c + 3]), "overlapping sources"
            ports = []
            for dr, dc in RING:
                value = int(a[r + dr, c + dc])
                assert value in (bg, wall), "invalid source ring"
                if value == bg:
                    ports.append((r + dr, c + dc))
            assert 1 <= len(ports) <= 4, "each source needs one to four ports"
            hardware[r - 1:r + 3, c - 1:c + 3] = True
            sources.append((r, c, color, tuple(ports)))
            seeds.extend((x, y, color) for x, y in ports)
    assert sources, "no source cores"
    assert np.all(hardware | (a == bg) | (a == wall)), "unaccounted payload cells"
    walls = (a == wall) & ~hardware
    _, wall_groups = _components(walls)
    assert len(wall_groups) == 1, "walls must connect to the outer frame"
    assert not np.any(walls[:-1, :-1] & walls[1:, :-1]
                      & walls[:-1, 1:] & walls[1:, 1:]), "walls must be one cell thick"
    rooms, groups = _components(~walls)
    depth = np.full(a.shape, -1, dtype=np.int16)
    for index in range(len(groups)):
        m = rooms == index
        local = _erosion(m)
        depth[m] = local[m]
    # A track is a connected component of one room at one exact depth.
    tracks = np.full(a.shape, -1, dtype=np.int16)
    track_groups = []
    for index in range(len(groups)):
        for d in range(int(depth[rooms == index].max()) + 1):
            _, parts = _components((rooms == index) & (depth == d), D8)
            for part in parts:
                k = len(track_groups)
                for x, y in part:
                    tracks[x, y] = k
                track_groups.append(part)
    return dict(grid=a, bg=bg, wall=wall, walls=walls, hardware=hardware,
                fixed=fixed, sources=sources, seeds=seeds, rooms=rooms,
                room_groups=groups, depth=depth, tracks=tracks,
                track_groups=track_groups)


def _race(info):
    """Simultaneous multi-source wavefronts with color ownership bit sets."""
    a, tracks = info['grid'], info['tracks']
    h, w = a.shape
    distance = np.full(a.shape, INF, dtype=np.int16)
    owners = np.zeros(a.shape, dtype=np.int16)
    q = deque()
    for r, c, color in info['seeds']:
        distance[r, c] = 0
        owners[r, c] |= 1 << color
        q.append((r, c))
    while q:
        r, c = q.popleft()
        nd = int(distance[r, c]) + 1
        for dr, dc in D8:
            x, y = r + dr, c + dc
            if not (0 <= x < h and 0 <= y < w and tracks[x, y] == tracks[r, c]):
                continue
            if nd < distance[x, y]:
                distance[x, y] = nd
                owners[x, y] = owners[r, c]
                q.append((x, y))
            elif nd == distance[x, y]:
                combined = int(owners[x, y]) | int(owners[r, c])
                if combined != owners[x, y]:
                    owners[x, y] = combined
                    q.append((x, y))
    answer = np.full(a.shape, info['bg'], dtype=np.int16)
    for color in range(10):
        answer[owners == (1 << color)] = color
    answer[info['fixed']] = a[info['fixed']]
    return answer, distance, owners


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles, transparent hardware, contour tracks, and race winners."""
    return _race(_parse(input_grid))[0]


def _reference(info, mode='correct'):
    """Separate per-color BFS distances; also defines competing-rule checks.

    This shares the input parser, but not the wavefront/ownership algorithm.
    'box', 'four', and 'flood' deliberately replace the inherited A mechanism.
    'ring' replaces B's gap interpretation by treating the whole ring as ports.
    Other checks replace track distance, diagonal steps, or transparent devices.
    """
    a = info['grid']
    h, w = a.shape
    depth = info['depth'].copy()
    if mode == 'four':
        for index in range(len(info['room_groups'])):
            m = info['rooms'] == index
            local = _erosion(m, False)
            depth[m] = local[m]
    elif mode == 'box':
        for part in info['room_groups']:
            rr, cc = np.array(part).T
            depth[rr, cc] = np.minimum.reduce((rr - rr.min(), rr.max() - rr,
                                                cc - cc.min(), cc.max() - cc))
    elif mode == 'flood':
        depth[info['rooms'] >= 0] = 0
    seeds = list(info['seeds'])
    if mode == 'ring':
        seeds = [(r + dr, c + dc, color) for r, c, color, _ in info['sources']
                 for dr, dc in RING]
    fields = {}
    for color in sorted({s[2] for s in seeds}):
        distances = np.full(a.shape, INF, dtype=np.int16)
        q = deque()
        for r, c, col in seeds:
            if col == color:
                if mode == 'air':
                    rr, cc = np.indices(a.shape)
                    valid = ((info['rooms'] == info['rooms'][r, c])
                             & (depth == depth[r, c]))
                    squared = (rr - r) ** 2 + (cc - c) ** 2
                    distances[valid] = np.minimum(distances[valid], squared[valid])
                    continue
                distances[r, c] = 0
                q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in (D4 if mode == 'four_walk' else D8):
                x, y = r + dr, c + dc
                if (0 <= x < h and 0 <= y < w
                        and info['rooms'][x, y] == info['rooms'][r, c]
                        and depth[x, y] == depth[r, c]
                        and (mode != 'opaque' or not info['fixed'][x, y])
                        and distances[x, y] > distances[r, c] + 1):
                    distances[x, y] = distances[r, c] + 1
                    q.append((x, y))
        fields[color] = distances
    best = np.full(a.shape, INF, dtype=np.int16)
    ties = np.zeros(a.shape, dtype=np.int16)
    out = np.full(a.shape, info['bg'], dtype=np.int16)
    for color, field in fields.items():
        smaller, equal = field < best, (field == best) & (field < INF)
        out[smaller] = color
        ties[smaller] = 1
        ties[equal] += 1
        best = np.minimum(best, field)
    out[ties != 1] = info['bg']
    out[info['fixed']] = a[info['fixed']]
    return out


def _features(info, answer, distance, owners):
    a, bg = info['grid'], info['bg']
    available = a == bg
    contested = (owners != 0) & ((owners & (owners - 1)) != 0) & available
    seeded = {int(info['tracks'][r, c]) for r, c, _ in info['seeds']}
    seed_depths = {int(info['depth'][r, c]) for r, c, _ in info['seeds']}
    competitive = 0
    for track in seeded:
        colors = {color for r, c, color in info['seeds'] if info['tracks'][r, c] == track}
        visible = set(int(v) for v in answer[(info['tracks'] == track) & available]) - {bg}
        competitive += len(colors) >= 2 and len(visible) >= 2
    concave = False
    split_track = False
    for i, part in enumerate(info['room_groups']):
        rr, cc = np.array(part).T
        concave |= len(part) != (int(rr.max() - rr.min()) + 1) * (int(cc.max() - cc.min()) + 1)
        for d in range(int(info['depth'][rr, cc].max()) + 1):
            ids = set(int(v) for v in info['tracks'][(info['rooms'] == i) & (info['depth'] == d)])
            split_track |= len(ids) > 1 and bool(ids & seeded)
    tags = set()
    if len(seed_depths) >= 2:
        tags.add('multiple_depths')
    if max(len(s[3]) for s in info['sources']) >= 2:
        tags.add('multiple_ports')
    if contested.any():
        tags.add('tie')
    if competitive:
        tags.add('competition')
    if concave:
        tags.add('concavity')
    if split_track:
        tags.add('split_track')
    if len(info['room_groups']) > 1:
        tags.add('multiple_rooms')
    return dict(tags=tags, ties=int(contested.sum()), competitive=competitive,
                depths=tuple(sorted(seed_depths)), sources=len(info['sources']),
                ports=sum(len(s[3]) for s in info['sources']),
                max_distance=int(distance[distance < INF].max()),
                changed=int(np.count_nonzero(answer != a)),
                unseeded=sum(1 for i, p in enumerate(info['track_groups'])
                             if i not in seeded and any(available[x, y] for x, y in p)))


def _wall_layout(rng, shape, rich):
    h, w = shape
    wall = np.zeros(shape, dtype=bool)
    wall[0] = wall[-1] = True
    wall[:, 0] = wall[:, -1] = True
    if rich and min(shape) >= 24 and rng.random() < .4:
        if rng.integers(2):
            wall[:, int(rng.integers(10, w - 10))] = True
        else:
            wall[int(rng.integers(10, h - 10)), :] = True
    # Compose connected straight or elbow-shaped wall peninsulas.
    budget = int(rng.integers(1, 5)) if rich else int(rng.integers(0, 2))
    for _ in range(budget):
        for _attempt in range(20):
            points = np.argwhere(wall)
            r, c = map(int, points[int(rng.integers(len(points)))])
            dr, dc = D4[int(rng.integers(4))]
            length = int(rng.integers(2, 8))
            extension = [(r + k * dr, c + k * dc) for k in range(1, length + 1)]
            if rng.random() < .65:
                er, ec = ((-dc, dr) if rng.integers(2) else (dc, -dr))
                x, y = extension[-1]
                extension += [(x + k * er, y + k * ec) for k in range(1, int(rng.integers(2, 7)))]
            if any(not (1 <= x < h - 1 and 1 <= y < w - 1) or wall[x, y] for x, y in extension):
                continue
            candidate = wall.copy()
            for x, y in extension:
                candidate[x, y] = True
            if np.any(candidate[:-1, :-1] & candidate[1:, :-1]
                      & candidate[:-1, 1:] & candidate[1:, 1:]):
                continue
            _, groups = _components(~candidate)
            if any(len(p) < 16 for p in groups):
                continue
            wall = candidate
            break
    return wall


def _make_input(rng, wall, count, multi, roles):
    h, w = wall.shape
    bg, structure, *palette = roles
    rooms, groups = _components(~wall)
    depth = np.full(wall.shape, -1, dtype=np.int16)
    for i in range(len(groups)):
        m = rooms == i
        ds = _erosion(m)
        depth[m] = ds[m]
    candidates = [(r, c) for r in range(2, h - 3) for c in range(2, w - 3)
                  if not wall[r - 1:r + 3, c - 1:c + 3].any()]
    assert len(candidates) >= count
    rng.shuffle(candidates)
    chosen = []
    occupied = np.zeros(wall.shape, dtype=bool)
    for r, c in candidates:
        if occupied[max(0, r - 2):r + 4, max(0, c - 2):c + 4].any():
            continue
        chosen.append((r, c))
        occupied[r - 1:r + 3, c - 1:c + 3] = True
        if len(chosen) == count:
            break
    assert len(chosen) == count
    # Choose two openings on the same depth and in the same room. Reject if
    # these particular sources have no such pair; resample the whole scene.
    options = []
    r, c = chosen[0]
    x, y = chosen[1]
    for dr, dc in RING:
        for er, ec in RING:
            if (rooms[r + dr, c + dc] == rooms[x + er, y + ec]
                    and depth[r + dr, c + dc] == depth[x + er, y + ec]):
                options.append(((dr, dc), (er, ec)))
    assert options
    first_ports = options[int(rng.integers(len(options)))]
    a = np.full(wall.shape, bg, dtype=np.int16)
    a[wall] = structure
    for i, (r, c) in enumerate(chosen):
        ports = [first_ports[i]] if i < 2 else [RING[int(rng.integers(12))]]
        if multi:
            extra = int(rng.integers(1, 4)) if i == 0 else int(rng.integers(0, 3))
            pool = [p for p in RING if p not in ports]
            ports += [pool[int(j)] for j in rng.choice(len(pool), extra, replace=False)]
        for dr, dc in RING:
            a[r + dr, c + dc] = bg if (dr, dc) in ports else structure
        a[r:r + 2, c:c + 2] = palette[i]
    return a


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _sample_pair(rng, regime, roles, taught=None, seen=None, test_count=None):
    # All rejection is on visible geometry and output witnesses, never on a
    # hidden rule parameter. The search has a finite, documented upper bound.
    for attempt in range(240):
        rich = regime != 'basic'
        if regime == 'basic':
            shape = tuple(int(x) for x in rng.integers(17, 27, 2))
            count, multi = 2, False
        elif regime == 'teaching':
            shape = tuple(int(x) for x in rng.integers(23, 31, 2))
            count, multi = 3, True
        else:
            low = 22 if regime == 'test' else 18
            shape = tuple(int(x) for x in rng.integers(low, 31, 2))
            count = test_count if regime == 'test' else int(rng.integers(2, 5))
            multi = bool(rng.integers(2))
        try:
            walls = _wall_layout(rng, shape, rich)
            a = _make_input(rng, walls, count, multi, roles)
            if seen is not None and _key(a) in seen:
                continue
            info = _parse(a)
            b, distance, owners = _race(info)
            f = _features(info, b, distance, owners)
            assert f['competitive'] and f['changed'] >= 12 and f['unseeded']
            assert all(np.count_nonzero((b == color) & (a == info['bg'])
                                        & ~info['hardware']) >= 3
                       for _, _, color, _ in info['sources']), 'inactive source'
            if regime == 'basic':
                assert f['ties'] and f['max_distance'] >= 5
            elif regime == 'teaching':
                assert {'multiple_depths', 'multiple_ports', 'concavity', 'tie'} <= f['tags']
                assert all(not np.array_equal(b, _reference(info, alt))
                           for alt in ALTERNATIVES)
            if taught is not None:
                assert f['tags'] <= taught, "test introduces an undemonstrated mechanism"
            assert np.array_equal(b, _reference(info))
            return {'input': a, 'output': b}, f
        except AssertionError:
            continue
    raise RuntimeError(f"bounded scene search exhausted for {regime}")


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10),
                                    ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle = {'train': [], 'test': []}
    seen, taught = set(), set()
    max_train_sources = 0
    # Local color roles vary visibly. There is no puzzle-specific color law.
    backgrounds = [int(x) for x in rng.permutation(10)]
    for i in range(nt + ne):
        bg = backgrounds[i % 10]
        rest = [int(c) for c in rng.permutation(10) if c != bg]
        roles = [bg] + rest
        regime = ('basic' if i == 0 else 'teaching' if i == 1 else
                  'extra' if i < nt else 'test')
        pair, features = _sample_pair(rng, regime, roles,
                                      taught if i >= nt else None, seen,
                                      max_train_sources + 1 if i >= nt else None)
        puzzle['train' if i < nt else 'test'].append(pair)
        seen.add(_key(pair['input']))
        if i < nt:
            taught |= features['tags']
            max_train_sources = max(max_train_sources, features['sources'])
    # Shuffle demonstration order so pair index is not an output convention.
    order = rng.permutation(nt)
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, taught, needed = set(), set(), set()
    witnesses = {mode: False for mode in ALTERNATIVES}
    train_roles = set()
    source_counts = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _parse(a)
            predicted, distance, owners = _race(info)
            assert np.array_equal(predicted, b), 'incorrect wavefront result'
            assert np.array_equal(_reference(info), b), 'per-color distance result disagrees'
            for i in range(len(info['room_groups'])):
                m = info['rooms'] == i
                assert np.array_equal(_bfs_depth(m)[m], info['depth'][m])
            assert np.array_equal(a[info['fixed']], b[info['fixed']]), 'hardware changed'
            assert set(np.unique(b)) <= set(np.unique(a)), 'new color invented'
            f = _features(info, b, distance, owners)
            assert f['changed'] >= 12 and f['competitive'] and f['unseeded']
            assert all(np.count_nonzero((b == color) & (a == info['bg'])
                                        & ~info['hardware']) >= 3
                       for _, _, color, _ in info['sources']), 'inactive source'
            source_counts[split].append(f['sources'])
            if split == 'train':
                taught |= f['tags']
                train_roles.add((info['bg'], info['wall']))
                for mode in witnesses:
                    witnesses[mode] |= not np.array_equal(_reference(info, mode), b)
            else:
                needed |= f['tags']
    assert needed <= taught, 'test mechanism absent from demonstrations'
    assert {'competition', 'tie', 'multiple_depths', 'multiple_ports', 'concavity'} <= taught
    assert len(train_roles) >= 2 and all(witnesses.values()), 'training lacks discriminating evidence'
    assert min(source_counts['test']) > max(source_counts['train']), 'missing source-count generalization'


def self_test():
    """42 seeds; bounded diagnostics in memory, without retained puzzle files."""
    # Hand-calculated symmetric race. Only the outer contour is seeded. Each
    # half is colored by its near source, with blank top/bottom midpoint ties.
    fixture = np.zeros((13, 19), dtype=np.int16)
    fixture[0] = fixture[-1] = 8
    fixture[:, 0] = fixture[:, -1] = 8
    for c, color, gap in ((2, 2, (4, 1)), (15, 3, (4, 17))):
        fixture[3:7, c - 1:c + 3] = 8
        fixture[4:6, c:c + 2] = color
        fixture[gap] = 0
    result = solve(fixture)
    assert result[1, 1] == result[4, 1] == 2
    assert result[1, 17] == result[4, 17] == 3
    assert result[1, 9] == result[11, 9] == result[7, 9] == 0
    assert result[4, 2] == 2 and result[3, 2] == 8
    # Palette and orientation equivariance require no latent convention.
    lut = np.array([7, 4, 5, 1, 8, 0, 9, 3, 2, 6])
    assert np.array_equal(solve(lut[fixture]), lut[result])
    assert np.array_equal(solve(np.rot90(fixture)), np.rot90(result))
    shapes, palettes, layouts, counts, depths, tags, signatures = (set() for _ in range(7))
    total_ties = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _parse(pair['input'])
                b, d, o = _race(info)
                f = _features(info, b, d, o)
                digest.update(pair['input'].astype(np.int8).tobytes())
                shapes.add(pair['input'].shape)
                palettes.add((info['bg'], info['wall']))
                layouts.add(hashlib.sha256(info['walls'].tobytes()).digest())
                counts.add(f['sources'])
                depths.update(f['depths'])
                tags |= f['tags']
                total_ties += f['ties']
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(shapes) >= 50
    assert len(palettes) >= 40 and len(layouts) >= 100
    assert {2, 3, 4, 5} <= counts and len(depths) >= 5 and total_ties > 100
    assert {'competition', 'tie', 'multiple_depths', 'multiple_ports', 'concavity'} <= tags
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(501 + nt * 10 + ne, nt, ne)
            validate_generated_puzzle(p)
    p = generate_puzzle(19, 2, 1)
    q = generate_puzzle(19, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in p for x, y in zip(p[s], q[s])
               for k in ('input', 'output'))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.0, 1), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    corrupt = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    info = _parse(corrupt['test'][0]['input'])
    r, c, _ = info['seeds'][0]
    corrupt['test'][0]['output'][r, c] = info['wall']
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted port output accepted')
