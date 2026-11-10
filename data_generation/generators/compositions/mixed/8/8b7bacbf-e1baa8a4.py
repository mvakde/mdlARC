"""Recover an equality-run lattice, then paint terminal-owned cavities.

Only the standard library and NumPy are used. Generation constructs intended
cavity ownership; solve reconstructs it from the input without construction data.
No generated puzzles, execution traces, or logs are stored by this module.
"""

from collections import Counter, deque
import hashlib

import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
HOST_SEEDS = (
    0, 346766963, 87817511, 611597748, 528040477, 966920101,
    564592643, 23987089, 89397686, 31914083, 620393997, 2497132,
    678434234, 998868771, 772147212, 255391839, 738091762,
    574240520, 43322795, 918373901, 578316671, 247918164,
    375445629, 170592023, 283109704, 371484571, 914703966,
    497598499, 383827797, 631610611, 433401404, 53266561,
    77924156, 593046957, 555303445, 399840283, 253637028,
    531840091, 580058031, 998268509, 467077881, 37056586,
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(n < 1 or n > 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer 2D grid of size 1..30, colors 0..9")
    return a


def _starts(a, axis):
    return [0] + [i for i in range(1, a.shape[axis])
                  if not np.array_equal(np.take(a, i, axis=axis),
                                        np.take(a, i - 1, axis=axis))]


def _collapse(a):
    rows, cols = _starts(a, 0), _starts(a, 1)
    return a[np.ix_(rows, cols)].copy(), rows, cols


def _runs(a, axis):
    edges = _starts(a, axis) + [a.shape[axis]]
    return tuple(b - x for x, b in zip(edges, edges[1:]))


def _near(cells, shape, steps=D8):
    h, w = shape
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps
            if 0 <= r + dr < h and 0 <= c + dc < w} - set(cells)


def _components(mask, steps=D8):
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    groups = []
    for r, c in zip(*np.where(mask)):
        p = (int(r), int(c))
        if labels[p] >= 0:
            continue
        index = len(groups)
        labels[p] = index
        group = [p]
        for y, x in group:
            for dy, dx in steps:
                q = (y + dy, x + dx)
                if (0 <= q[0] < h and 0 <= q[1] < w
                        and mask[q] and labels[q] < 0):
                    labels[q] = index
                    group.append(q)
        groups.append(set(group))
    return labels, groups


def _paint_compact(a):
    """Analyze only visible compact geometry; no generation state is consulted."""
    bg = int(a[0, 0])
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    if not np.all(border == bg):
        raise ValueError("this family has a complete uniform exterior border")
    labels, systems = _components(a != bg)
    sources = {}
    source_points = {}
    for owner, cells in enumerate(systems):
        candidates = []
        for p in sorted(cells):
            adjacent = _near({p}, a.shape) & cells
            if len(adjacent) == 1:
                q = next(iter(adjacent))
                if a[p] != a[q]:
                    candidates.append(p)
        if len(candidates) > 1:
            raise ValueError("a system may have only one paint terminal")
        body = cells - set(candidates)
        if len({int(a[p]) for p in body}) != 1:
            raise ValueError("each system has one body color, plus an optional terminal")
        if candidates:
            p = candidates[0]
            sources[owner] = int(a[p])
            source_points[owner] = p

    _, background_regions = _components(a == bg, D4)
    cavities = []
    out = a.copy()
    for cells in background_regions:
        if any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1)
               for r, c in cells):
            continue
        edge = _near(cells, a.shape, D4)
        owners = {int(labels[p]) for p in edge if labels[p] >= 0}
        if len(owners) != 1:
            raise ValueError("a cavity must have one unambiguous system owner")
        owner = owners.pop()
        cavities.append((cells, owner))
        if owner in sources:
            for p in cells:
                out[p] = sources[owner]
    return out, dict(background=bg, systems=systems, labels=labels,
                     sources=sources, source_points=source_points,
                     cavities=cavities)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Collapse maximal identical full-line runs once, then fill owned holes."""
    compact = _collapse(_grid(input_grid))[0]
    return _paint_compact(compact)[0]


def _ri(rng, low, high):
    return int(rng.integers(low, high + 1))


def _pick(rng, choices):
    return choices[_ri(rng, 0, len(choices) - 1)]


def _move(cells, dy, dx):
    return {(r + dy, c + dx) for r, c in cells}


def _contour(rng):
    """A connected row-profile cavity and its minimal four-neighbor shell."""
    kind = _pick(rng, ("box", "diamond", "step", "slant", "waist", "bar"))
    height, width = _ri(rng, 1, 4), _ri(rng, 1, 4)
    if kind == "diamond":
        radius = _ri(rng, 0, 2)
        inner = {(r, c) for r in range(-radius, radius + 1)
                 for c in range(-radius, radius + 1)
                 if abs(r) + abs(c) <= radius}
    else:
        if kind == "bar":
            height, width = 1, _ri(rng, 2, 6)
        inner = set()
        left = 0
        for r in range(height):
            if kind == "slant":
                left = r // 2
            elif kind == "step" and r:
                left += _pick(rng, (-1, 0, 1))
            size = max(1, width - int(kind == "waist" and 0 < r < height - 1))
            if r:
                previous = {c for y, c in inner if y == r - 1}
                # Overlap neighboring intervals: the cavity must be 4-connected.
                left = min(max(left, min(previous) - size + 1), max(previous))
            inner.update((r, c) for c in range(left, left + size))
    boundary = {(r + dr, c + dc) for r, c in inner for dr, dc in D4} - inner
    if rng.random() < .5:
        inner = {(c, r) for r, c in inner}
        boundary = {(c, r) for r, c in boundary}
    y, x = min(r for r, c in boundary), min(c for r, c in boundary)
    return _move(inner, -y, -x), _move(boundary, -y, -x)


def _path(rng, starts, goals, allowed, shape):
    starts = sorted(starts)
    rng.shuffle(starts)
    queue = deque(starts)
    previous = {p: None for p in starts}
    directions = [D8[int(i)] for i in rng.permutation(8)]
    end = None
    while queue:
        p = queue.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in directions:
            q = (p[0] + dr, p[1] + dc)
            if q in allowed and q not in previous:
                previous[q] = p
                queue.append(q)
    if end is None:
        return None
    cells = set()
    while end is not None:
        cells.add(end)
        end = previous[end]
    return cells


def _scene(rng, counts, active, require_diagonal=False, shared_body=False):
    """Construct one scene from cavities and free-space routing, not parent calls."""
    n = len(counts)
    total = sum(counts)
    low = 13 if total <= 3 else 19
    shape = (_ri(rng, low, 25), _ri(rng, low, 25))
    canvas = {(r, c) for r in range(1, shape[0] - 1)
              for c in range(1, shape[1] - 1)}
    palette = [int(v) for v in rng.permutation(10)]
    bg = palette[0]
    bodies = [palette[1]] * n if shared_body or rng.random() < .45 else palette[1:1 + n]
    available = [v for v in palette if v != bg and v not in bodies]
    paints = available[:active]
    # The two-color teaching example stays discriminative; later source colors
    # may repeat across separate systems without changing their ownership.
    if active > 1 and not require_diagonal and rng.random() < .2:
        paints = [paints[0]] * active
    regions = []
    occupied = set()
    owners = [i for i, count in enumerate(counts) for _ in range(count)]
    rng.shuffle(owners)
    for owner in owners:
        inner, edge = _contour(rng)
        height = max(r for r, c in edge) + 1
        width = max(c for r, c in edge) + 1
        placed = False
        for _ in range(65):
            y = _ri(rng, 1, shape[0] - height - 1)
            x = _ri(rng, 1, shape[1] - width - 1)
            ins, boundary = _move(inner, y, x), _move(edge, y, x)
            cells = ins | boundary
            if cells & occupied or _near(cells, shape) & occupied:
                continue
            regions.append((ins, boundary, owner))
            occupied |= cells
            placed = True
            break
        if not placed:
            return None

    endpoints = []
    for owner in range(n):
        forbidden = occupied | _near(occupied, shape)
        for p in endpoints:
            forbidden |= {p} | _near({p}, shape)
        candidates = sorted(canvas - forbidden)
        if not candidates:
            return None
        endpoints.append(_pick(rng, candidates))

    wires = [set() for _ in range(n)]
    all_inner = set().union(*(ins for ins, edge, owner in regions))
    all_edge = set().union(*(edge for ins, edge, owner in regions))
    all_endpoints = set(endpoints)
    for owner in [int(i) for i in rng.permutation(n)]:
        targets = [(ins, edge) for ins, edge, own in regions if own == owner]
        rng.shuffle(targets)
        # Inactive single loops can be detached, rather than always tailed.
        if owner >= active and counts[owner] == 1 and rng.random() < .4:
            continue
        tree = {endpoints[owner]}
        foreign_shapes = set().union(*(ins | edge for ins, edge, own in regions
                                       if own != owner))
        foreign_wires = set().union(*(wire for i, wire in enumerate(wires) if i != owner))
        for target_index, (inner, boundary) in enumerate(targets):
            reserved_ends = all_endpoints - ({endpoints[owner]} if target_index == 0 else set())
            protected = reserved_ends | _near(reserved_ends, shape)
            forbidden = all_inner | all_edge | protected
            forbidden |= foreign_shapes | _near(foreign_shapes, shape)
            forbidden |= foreign_wires | _near(foreign_wires, shape)
            starts = tree if target_index == 0 else tree - {endpoints[owner]}
            allowed = (canvas - forbidden) | starts
            goals = _near(boundary, shape) & allowed
            path = _path(rng, starts, goals, allowed, shape)
            if path is None:
                return None
            wires[owner] |= path - all_edge
            tree |= path | boundary

    raw = np.full(shape, bg, dtype=np.int8)
    expected = raw.copy()
    for inner, boundary, owner in regions:
        for p in boundary:
            raw[p] = bodies[owner]
    for owner, wire in enumerate(wires):
        for p in wire:
            raw[p] = bodies[owner]
        if wire and owner < active:
            raw[endpoints[owner]] = paints[owner]
    expected[:] = raw
    for inner, boundary, owner in regions:
        if owner < active:
            for p in inner:
                expected[p] = paints[owner]

    rotation = _ri(rng, 0, 3)
    raw, expected = np.rot90(raw, rotation), np.rot90(expected, rotation)
    if rng.random() < .5:
        raw, expected = np.fliplr(raw), np.fliplr(expected)
    # Canonicalize accidental equal whole lines before any enlargement. The
    # intended labels undergo the same selection, then visible geometry must
    # still recover every source, system, and complete cavity independently.
    compact, rows, cols = _collapse(raw)
    answer = expected[np.ix_(rows, cols)].copy()
    try:
        prediction, info = _paint_compact(compact)
    except ValueError:
        return None
    if (not np.array_equal(prediction, answer)
            or len(info['systems']) != n or len(info['sources']) != active
            or len(info['cavities']) != total
            or sorted(info['sources'].values()) != sorted(paints)):
        return None
    if require_diagonal and not any(
            all(p[0] != q[0] and p[1] != q[1] for q in
                _near({p}, compact.shape) & info['systems'][owner])
            for owner, p in info['source_points'].items()):
        return None
    return compact, answer


def _band_schedule(rng, length, mode):
    """Positive band lengths, at most 30 physical cells on an axis."""
    if mode == 'none':
        return (1,) * length
    if mode == 'uniform' and length <= 15:
        return (_ri(rng, 2, min(3, 30 // length)),) * length
    values = [1] * length
    if mode == 'shift':
        values[_ri(rng, 0, length - 1)] = _ri(rng, 4, min(6, 31 - length))
    spare = 30 - sum(values)
    count = _ri(rng, 1, min(length, max(1, spare))) if spare else 0
    for index in rng.permutation(length)[:count]:
        index = int(index)
        amount = min(_ri(rng, 1, 2), spare, 3 - values[index])
        if amount > 0:
            values[index] += amount
            spare -= amount
    return tuple(values)


def _resolve_counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10),
                                 ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f'{name} must be an integer from {lo} through {hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _resolve_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            for attempt in range(240):
                lesson = split == 'train' and index < 2
                if lesson:
                    active, counts = (1, [2, 1]) if index == 0 else (2, [1, 1, 1])
                else:
                    n = _ri(rng, 1, 3)
                    active = _ri(rng, 1, n)
                    counts = [_ri(rng, 1, 3) for _ in range(n)]
                    while sum(counts) > 6:
                        counts[int(np.argmax(counts))] -= 1
                # require_diagonal also prevents repeated source paints in the
                # second lesson, which must expose two independent supplies.
                scene = _scene(rng, counts, active, require_diagonal=lesson,
                               shared_body=lesson and index == 0)
                if scene is None:
                    continue
                latent, answer = scene
                if split == 'train' and index == 0:
                    modes = ('mixed', 'mixed')
                elif split == 'test':
                    modes = ('shift', _pick(rng, ('mixed', 'uniform', 'none')))
                    if rng.random() < .5:
                        modes = modes[::-1]
                else:
                    modes = _pick(rng, (('mixed', 'mixed'), ('mixed', 'none'),
                                        ('none', 'mixed'), ('uniform', 'uniform')))
                rows = _band_schedule(rng, latent.shape[0], modes[0])
                cols = _band_schedule(rng, latent.shape[1], modes[1])
                if split == 'train' and index == 0 and (
                        len(set(rows)) == 1 or len(set(cols)) == 1):
                    continue
                expanded = np.repeat(np.repeat(latent, rows, axis=0), cols, axis=1)
                key = (expanded.shape, expanded.tobytes())
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append({'input': expanded.copy(), 'output': answer})
                break
            else:
                raise RuntimeError('bounded scene construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _four_neighbor_ablation(a, info):
    labels, _ = _components(a != info['background'], D4)
    source_by_label = {int(labels[p]): info['sources'][owner]
                       for owner, p in info['source_points'].items()}
    out = a.copy()
    for cells, owner in info['cavities']:
        adjacent = {int(labels[p]) for p in _near(cells, a.shape, D4)}
        if len(adjacent) == 1 and next(iter(adjacent)) in source_by_label:
            color = source_by_label[next(iter(adjacent))]
            for p in cells:
                out[p] = color
    return out


def _features(pair):
    a, b = np.asarray(pair['input']), np.asarray(pair['output'])
    compact = _collapse(a)[0]
    _, info = _paint_compact(compact)
    counts = Counter(owner for cells, owner in info['cavities'])
    shapes = set()
    for cells, owner in info['cavities']:
        y, x = min(r for r, c in cells), min(c for r, c in cells)
        shapes.add(tuple(sorted((r - y, c - x) for r, c in cells)))
    directions = set()
    for owner, p in info['source_points'].items():
        q = next(iter(_near({p}, compact.shape) & info['systems'][owner]))
        directions.add((q[0] - p[0], q[1] - p[1]))
    body_colors = {owner: int(compact[next(iter(cells - {info['source_points'].get(owner)}))])
                   for owner, cells in enumerate(info['systems'])}
    supplied_bodies = {color for owner, color in body_colors.items() if owner in info['sources']}
    unsupplied_bodies = {color for owner, color in body_colors.items() if owner not in info['sources']}
    return dict(input_shape=a.shape, output_shape=b.shape,
                background=info['background'], palette=tuple(int(x) for x in np.unique(a)),
                systems=len(info['systems']), sources=len(info['sources']),
                paints=len(set(info['sources'].values())), cavities=len(info['cavities']),
                unmarked=sum(owner not in info['sources'] for cells, owner in info['cavities']),
                max_owned=max(counts.values()), cavity_shapes=shapes,
                rows=_runs(a, 0), cols=_runs(a, 1), directions=directions,
                changed=int(np.count_nonzero(compact != b)),
                shared_body_contrast=bool(supplied_bodies & unsupplied_bodies),
                four_diff=not np.array_equal(_four_neighbor_ablation(compact, info), b),
                repeated_lines=(len({row.tobytes() for row in compact}) < len(compact)
                                or len({col.tobytes() for col in compact.T}) < compact.shape[1]))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'split schema')
    _resolve_counts(len(puzzle['train']), len(puzzle['test']))
    seen = set()
    features = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            key = (a.shape, a.astype(np.int8).tobytes())
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            compact, _, _ = _collapse(a)
            expected, info = _paint_compact(compact)
            _require(np.array_equal(expected, b), 'input-derived answer mismatch')
            _require(a.shape != b.shape, 'every generated pair must require compression')
            _require(np.any(compact != b), 'every pair must require terminal filling')
            _require(np.all(compact[compact != b] == info['background']), 'nonbackground changed')
            _require(np.array_equal(np.unique(a), np.unique(b)), 'color identity changed')
            _require(np.array_equal(np.repeat(np.repeat(compact, _runs(a, 0), axis=0),
                                              _runs(a, 1), axis=1), a), 'expansion mismatch')
            _require(len(info['systems']) >= 1 and len(info['sources']) >= 1, 'missing systems')
            _require(len({owner for cells, owner in info['cavities']}) == len(info['systems']),
                     'each generated system must own a cavity')
            features[split].append(_features(pair))
    training = features['train']
    _require(any(f['unmarked'] for f in training), 'must demonstrate unmarked preservation')
    _require(any(f['shared_body_contrast'] for f in training),
             'must contrast marked/unmarked systems sharing a body color')
    _require(any(f['max_owned'] >= 2 for f in training), 'must demonstrate multi-loop ownership')
    _require(any(f['paints'] >= 2 for f in training), 'must demonstrate independent paint supplies')
    _require(any(f['four_diff'] for f in training), 'eight-neighbor connectivity not evidenced')
    _require(any(max(f['rows']) > 1 and max(f['cols']) > 1
                 and len(set(f['rows'])) > 1 and len(set(f['cols'])) > 1 for f in training),
             'nonuniform two-axis compression not demonstrated')
    train_max = max(max(f['rows'] + f['cols']) for f in training)
    _require(train_max <= 3, 'training band cap')
    _require(all(max(f['rows'] + f['cols']) > train_max for f in features['test']),
             'test must shift to thicker bands')


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed audit; retain counters/hashes rather than puzzle corpora."""
    # A hand-constructed case fixes ownership without using the scene sampler:
    # a diagonal 7-tipped string supplies the left cross-shaped loop; the
    # same-colored detached right loop must remain hollow.
    raw = np.zeros((9, 13), dtype=np.int8)
    for y, x in ((4, 4), (4, 9)):
        for dy, dx in D4:
            raw[y + dy, x + dx] = 2
    raw[1, 1] = 7
    raw[2, 2] = raw[3, 3] = 2
    expected = raw.copy()
    expected[4, 4] = 7
    compact, rows, cols = _collapse(raw)
    expected = expected[np.ix_(rows, cols)]
    _require(np.array_equal(solve(compact), expected), 'manual compact fixture')
    row_bands = [1 + (i % 3) for i in range(compact.shape[0])]
    col_bands = [1 + ((i + 1) % 2) for i in range(compact.shape[1])]
    enlarged = np.repeat(np.repeat(compact, row_bands, axis=0), col_bands, axis=1)
    _require(np.array_equal(solve(enlarged), expected), 'manual enlarged fixture')
    shapes, palettes, backgrounds, structures = set(), set(), set(), set()
    cavities, row_schedules, col_schedules, directions = set(), set(), set(), set()
    source_counts, system_counts, owned_counts, fingerprints = set(), set(), set(), set()
    one_axis = uniform_axis = repeated = four_diff = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair)
                shapes.add((f['input_shape'], f['output_shape']))
                palettes.add(f['palette'])
                backgrounds.add(f['background'])
                cavities.update(f['cavity_shapes'])
                directions.update(f['directions'])
                row_schedules.add(f['rows'])
                col_schedules.add(f['cols'])
                source_counts.add(f['sources'])
                system_counts.add(f['systems'])
                owned_counts.add(f['max_owned'])
                one_axis += (max(f['rows']) == 1) != (max(f['cols']) == 1)
                uniform_axis += len(set(f['rows'])) == 1 or len(set(f['cols'])) == 1
                repeated += f['repeated_lines']
                four_diff += f['four_diff']
                latent = _collapse(pair['input'])[0]
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in latent.flat)
                structures.add((latent.shape, normalized))
                # solve must neither mutate nor retain its caller's input.
                before = pair['input'].copy()
                _require(np.array_equal(solve(before), pair['output']), 'solve mismatch')
                _require(np.array_equal(before, pair['input']), 'solve mutated input')
    for actual, minimum, label in ((len(shapes), 40, 'shapes'), (len(palettes), 20, 'palettes'),
                                    (len(backgrounds), 8, 'backgrounds'),
                                    (len(structures), 150, 'structures'),
                                    (len(cavities), 15, 'cavity geometries'),
                                    (len(row_schedules), 80, 'row schedules'),
                                    (len(col_schedules), 80, 'column schedules'),
                                    (len(directions), 6, 'terminal directions'),
                                    (one_axis, 1, 'single-axis enlargement'),
                                    (uniform_axis, 1, 'uniform axis'),
                                    (repeated, 42, 'separated repeated lines'),
                                    (four_diff, 42, 'eight-neighbor dependence')):
        _require(actual >= minimum, f'insufficient {label}: {actual}')
    _require(len(fingerprints) == 42, 'cross-seed duplication')
    _require(source_counts >= {1, 2, 3} and system_counts >= {1, 2, 3}, 'count diversity')
    _require(max(owned_counts) >= 3, 'ownership breadth')
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(12000 + 10 * nt + ne, nt, ne)
            _require((len(puzzle['train']), len(puzzle['test'])) == (nt, ne), 'count handling')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    _require(_digest(generate_puzzle(0)) == _digest(generate_puzzle(0)), 'nondeterminism')
    # Reject a wrong answer even if its palette, bounds, and shape still pass.
    bad = generate_puzzle(17, 2, 1)
    pair = bad['test'][0]
    latent = _collapse(pair['input'])[0]
    p = tuple(np.argwhere(latent != pair['output'])[0])
    pair['output'][p] = latent[p]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('partial fill accepted')
    return dict(seeds=42, pairs=210, input_output_shapes=len(shapes),
                palettes=len(palettes), backgrounds=len(backgrounds),
                compact_structures=len(structures), cavity_shapes=len(cavities),
                row_schedules=len(row_schedules), column_schedules=len(col_schedules),
                terminal_directions=len(directions), single_axis_pairs=one_axis,
                uniform_axis_pairs=uniform_axis, separated_repeat_pairs=repeated,
                four_neighbor_ablation_differences=four_diff)
