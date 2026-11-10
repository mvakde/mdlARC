"""T transfers determine an inventory, then frequency-ranked column packing.

Only standard-library facilities and NumPy are used. No parent implementation,
example, generator state, or seed is consulted by solve().
"""

from collections import Counter
import hashlib

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    grid = np.asarray(value)
    _need(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "bad grid dimensions")
    _need(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((grid >= 0) & (grid <= 9))), "bad grid color")
    return grid.astype(np.int8, copy=True)


def _add(point, direction, distance=1):
    return point[0] + distance * direction[0], point[1] + distance * direction[1]


def _inside(point, shape):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def _components(grid):
    labels = np.full(grid.shape, -1, dtype=np.int16)
    colors, cells = [], []
    for start in np.ndindex(grid.shape):
        if labels[start] >= 0:
            continue
        label, color = len(colors), int(grid[start])
        colors.append(color)
        stack, points = [start], []
        labels[start] = label
        while stack:
            point = stack.pop()
            points.append(point)
            for direction in DIRS:
                other = _add(point, direction)
                if _inside(other, grid.shape) and labels[other] < 0 and grid[other] == color:
                    labels[other] = label
                    stack.append(other)
        cells.append(points)
    return labels, colors, cells


def _geometry(center, direction, shape):
    perpendicular = (-direction[1], direction[0])
    arms = (_add(center, perpendicular), _add(center, perpendicular, -1))
    tip = _add(center, direction)
    target = _add(center, direction, 2)
    flanks = (_add(tip, perpendicular), _add(tip, perpendicular, -1))
    write = frozenset((center, tip, *arms))
    context = frozenset((target, *flanks))
    if not all(_inside(p, shape) for p in write | context):
        return None
    return dict(center=center, direction=direction, arms=arms, tip=tip,
                targetcell=target, flanks=flanks, write=write, context=context)


def _decode(input_grid):
    """Recover a unique ink, restore hosts, then label original territories."""
    raw = _grid(input_grid)
    raw_ids, raw_colors, raw_cells = _components(raw)
    answers = []
    for ink in sorted(set(raw_colors)):
        # Ink occurs only in separated T annotations; large fill components
        # cannot be ink. Explicit junctions can disconnect the three ink pixels.
        ink_ids = [i for i, color in enumerate(raw_colors) if color == ink]
        if sum(len(raw_cells[i]) for i in ink_ids) < 3 or any(len(raw_cells[i]) > 4 for i in ink_ids):
            continue
        ink_points = {p for i in ink_ids for p in raw_cells[i]}
        centers = sorted({_add(p, d) for p in ink_points for d in DIRS})
        arrows = []
        for center in centers:
            for direction in DIRS:
                geo = _geometry(center, direction, raw.shape)
                if geo is None or not all(raw[p] == ink for p in (*geo['arms'], geo['tip'])):
                    continue
                # An ink component must belong wholly to this annotation.
                if any(not set(raw_cells[int(raw_ids[p])]) <= geo['write']
                       for p in (*geo['arms'], geo['tip'])):
                    continue
                f0, f1 = geo['flanks']
                host = int(raw[f0])
                if host == ink or raw[f1] != host or raw[geo['targetcell']] in (host, ink):
                    continue
                explicit = int(raw[center]) != ink
                arrows.append(dict(geo, host=host, explicit=explicit,
                                   payload=int(raw[center]) if explicit else host))
        if not arrows:
            continue
        used, covered, valid = set(), set(), True
        for arrow in arrows:
            if used & arrow['write']:
                valid = False
            used.update(arrow['write'])
            covered.update(p for p in arrow['write'] if raw[p] == ink)
        if not valid or covered != ink_points:
            continue
        # Annotation writes cannot touch each other or overwrite context.
        for i, arrow in enumerate(arrows):
            halo = {_add(p, d) for p in arrow['write'] for d in DIRS}
            for other in arrows[:i]:
                if (halo & other['write'] or arrow['context'] & other['write']
                        or other['context'] & arrow['write']):
                    valid = False
        if not valid:
            continue
        clean = raw.copy()
        for arrow in arrows:
            for point in arrow['write']:
                clean[point] = arrow['host']
        ids, colors, cells = _components(clean)
        if ink in colors or any(len(points) < 8 for points in cells):
            continue
        assignments = {}
        for arrow in arrows:
            source, target = int(ids[arrow['center']]), int(ids[arrow['targetcell']])
            arrow.update(source=source, target=target)
            if (source == target or any(ids[p] != source for p in arrow['write'] | set(arrow['flanks']))
                    or arrow['payload'] == colors[target]
                    or (target in assignments and assignments[target] != arrow['payload'])):
                valid = False
            assignments[target] = arrow['payload']
        if not valid:
            continue
        painted = clean.copy()
        for target, payload in assignments.items():
            painted[ids == target] = payload
        answers.append(dict(ink=ink, arrows=arrows, clean=clean, ids=ids,
                            colors=colors, cells=cells, assignments=assignments, painted=painted))
    _need(len(answers) == 1, "T annotations must have exactly one interpretation")
    return answers[0]


def _histogram(grid):
    return tuple(int(n) for n in np.bincount(grid.ravel(), minlength=10))


def _ranked(grid, distinct=True):
    colors, counts = np.unique(grid, return_counts=True)
    if distinct:
        _need(len(set(map(int, counts))) == len(counts), "tied resulting frequencies are outside the family")
    order = np.argsort(-counts, kind='stable')
    return colors[order], counts[order]


def _pack(grid):
    colors, counts = _ranked(grid)
    height, width = grid.shape
    return np.repeat(colors, counts).reshape(width, height)[:, ::-1].T.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Simultaneous original-territory transfers, then exact-count packing."""
    return _pack(_decode(input_grid)['painted'])


def _connected(mask):
    points = set(map(tuple, np.argwhere(mask)))
    if not points:
        return False
    stack = [points.pop()]
    while stack:
        point = stack.pop()
        for direction in DIRS:
            neighbor = _add(point, direction)
            if neighbor in points:
                points.remove(neighbor)
                stack.append(neighbor)
    return not points


def _partition(rng, shape, count, mode):
    """Pixel cuts, variable-width bands, steps, and optional enclosed regions."""
    height, width = shape
    labels = np.zeros(shape, dtype=np.int16)
    rr, cc = np.indices(shape)
    if mode == 'bands':
        axis = int(rng.integers(2))
        length = shape[axis]
        if length < 2 * count:
            return None
        widths = np.full(count, 2, dtype=int) + rng.multinomial(length - 2 * count, np.ones(count) / count)
        start = 0
        for label, size in enumerate(widths):
            if axis == 0:
                labels[start:start + size, :] = label
            else:
                labels[:, start:start + size] = label
            start += int(size)
        return labels
    for new_id in range(1, count):
        for _ in range(55):
            areas = np.bincount(labels.ravel(), minlength=new_id).astype(float)
            label = int(rng.choice(new_id, p=areas / areas.sum()))
            mask = labels == label
            rows, cols = np.nonzero(mask)
            r0, r1, c0, c1 = int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1
            if mode == 'enclosed' and min(r1 - r0, c1 - c0) >= 9 and rng.random() < .65:
                top = int(rng.integers(r0 + 2, r1 - 5))
                bottom = int(rng.integers(top + 3, r1 - 1))
                left = int(rng.integers(c0 + 2, c1 - 5))
                right = int(rng.integers(left + 3, c1 - 1))
                section = (rr >= top) & (rr < bottom) & (cc >= left) & (cc < right)
            else:
                axis = int(rng.integers(2))
                lo, hi = (r0, r1) if axis == 0 else (c0, c1)
                if hi - lo < 5:
                    continue
                cut = int(rng.integers(lo + 2, hi - 1))
                path = np.full(width if axis == 0 else height, cut, dtype=int)
                if mode != 'rectangles':
                    step = int(rng.integers(2, 7))
                    for start in range(0, len(path), step):
                        cut = int(np.clip(cut + rng.integers(-2, 3), lo + 2, hi - 2))
                        path[start:start + step] = cut
                section = rr < path[cc] if axis == 0 else cc < path[rr]
            part, other = mask & section, mask & ~section
            if min(int(part.sum()), int(other.sum())) < 12 or not _connected(part) or not _connected(other):
                continue
            labels[part] = new_id
            break
        else:
            return None
    return labels


def _palette(rng, labels, count, ink, repeated):
    pool = [c for c in range(10) if c != ink]
    rng.shuffle(pool)
    colors = np.array(pool[:count], dtype=np.int8)
    if repeated:
        adjacent = set()
        for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:])):
            adjacent.update((min(int(x), int(y)), max(int(x), int(y))) for x, y in zip(a[a != b], b[a != b]))
        choices = [(a, b) for a in range(count) for b in range(a + 1, count) if (a, b) not in adjacent]
        if not choices:
            return None
        a, b = choices[int(rng.integers(len(choices)))]
        colors[b] = colors[a]
    return colors


def _sites(labels):
    sites = []
    for center in np.ndindex(labels.shape):
        for direction in DIRS:
            targetcell = _add(center, direction, 2)
            if not _inside(targetcell, labels.shape) or labels[center] == labels[targetcell]:
                continue
            geo = _geometry(center, direction, labels.shape)
            if geo is not None and all(labels[p] == labels[center] for p in geo['write'] | set(geo['flanks'])):
                sites.append(dict(geo, source=int(labels[center]), target=int(labels[targetcell])))
    return sites


def _plan(rng, sites, colors, ink, desired, requests):
    sites = [sites[int(i)] for i in rng.permutation(len(sites))]
    chosen, assignments, writes, blocked, context = [], {}, set(), set(), set()

    def legal(site):
        return not (site['write'] & (blocked | context) or site['context'] & writes)

    def append(site, payload, explicit):
        target = site['target']
        if (not legal(site) or payload == colors[target]
                or (target in assignments and assignments[target] != payload)):
            return False
        chosen.append(dict(site, payload=int(payload), explicit=explicit))
        assignments[target] = int(payload)
        writes.update(site['write'])
        blocked.update(site['write'])
        blocked.update(_add(p, d) for p in site['write'] for d in DIRS)
        context.update(site['context'])
        return True

    if 'chain' in requests:
        found = False
        by_source = {}
        for site in sites:
            by_source.setdefault(site['source'], []).append(site)
        for first in sites:
            if colors[first['source']] == colors[first['target']]:
                continue
            halo = first['write'] | {_add(p, d) for p in first['write'] for d in DIRS}
            for second in by_source.get(first['target'], []):
                if (first['source'] == second['target'] or colors[second['source']] == colors[second['target']]
                        or second['write'] & (halo | first['context']) or second['context'] & first['write']):
                    continue
                append(first, colors[first['source']], False)
                append(second, colors[second['source']], False)
                found = True
                break
            if found:
                break
        if not found:
            return None
    for kind in ('new', 'existing'):
        if kind not in requests:
            continue
        found = False
        for site in sites:
            if site['target'] in assignments or not legal(site):
                continue
            choices = ([c for c in range(10) if c != ink and c not in colors] if kind == 'new'
                       else [int(c) for c in set(colors) if c not in (colors[site['source']], colors[site['target']])])
            if choices and append(site, int(rng.choice(choices)), True):
                found = True
                break
        if not found:
            return None
    explicit_rate = float(rng.choice([0., .2, .5, .8, 1.]))
    for site in sites:
        if len(chosen) >= desired:
            break
        if not legal(site):
            continue
        source, target = site['source'], site['target']
        if target in assignments:
            if rng.random() > .2:
                continue
            payload = assignments[target]
            explicit = payload != colors[source]
        else:
            explicit = bool(rng.random() < explicit_rate)
            choices = [c for c in range(10) if c not in (ink, colors[source], colors[target])]
            payload = int(rng.choice(choices)) if explicit else int(colors[source])
        append(site, payload, explicit)
    return (chosen, assignments) if len(chosen) == desired else None


def _features(info):
    colors, cells, arrows = info['colors'], info['cells'], info['arrows']
    final = [info['assignments'].get(i, color) for i, color in enumerate(colors)]
    explicit = [a for a in arrows if a['explicit']]
    chain = any(not a['explicit'] and not b['explicit'] and a['target'] == b['source']
                and a['source'] != b['target'] and a['payload'] != b['payload']
                for a in arrows for b in arrows)
    locality = any(colors[a] == colors[b] and final[a] != final[b]
                   for a in range(len(colors)) for b in range(a))
    nonrectangular = sum(len(points) != (max(r for r, c in points) - min(r for r, c in points) + 1)
                         * (max(c for r, c in points) - min(c for r, c in points) + 1) for points in cells)
    ranked_colors, counts = _ranked(info['painted'])
    boundaries = np.cumsum(counts)[:-1] % info['painted'].shape[0]
    return dict(chain=chain, locality=locality, new=any(a['payload'] not in colors for a in explicit),
                existing=any(a['payload'] in colors and a['payload'] != a['host'] for a in explicit),
                territories=len(colors), arrows=len(arrows), explicit=len(explicit),
                nonrectangular=nonrectangular, repeated=len(colors) != len(set(colors)),
                fanout=len({a['source'] for a in arrows}) < len({(a['source'], a['target']) for a in arrows}),
                multiple_incoming=len(info['assignments']) < len(arrows),
                cycle=any(a['source'] == b['target'] and a['target'] == b['source'] for a in arrows for b in arrows),
                aggregation=len(final) > len(set(final)),
                partial=any(2 <= int(n) <= info['painted'].shape[0] - 2 for n in boundaries),
                aligned=any(int(n) == 0 for n in boundaries),
                palette=len(counts), ranked=tuple(map(int, ranked_colors)))


def _candidate(rng, ink, split, requests, fallback=False):
    if fallback:
        count = 5 if split == 'train' else 7
        shape = (int(rng.integers(15, 22)) if split == 'train' else int(rng.integers(23, 31)),
                 3 * count + int(rng.integers(0, 5)))
        mode = 'bands'
    else:
        low, high = (13, 22) if split == 'train' else (22, 31)
        shape = (int(rng.integers(low, high)), int(rng.integers(low, high)))
        count = int(rng.integers(4, 7) if split == 'train' else rng.integers(6, 9))
        if not requests and split == 'train' and rng.random() < .25:
            count = int(rng.integers(3, 5))
        mode = str(rng.choice(['bands', 'rectangles', 'steps', 'enclosed'], p=[.18, .20, .40, .22]))
    labels = _partition(rng, shape, count, mode)
    if labels is None:
        return None
    colors = _palette(rng, labels, count, ink, 'locality' in requests or rng.random() < .45)
    if colors is None:
        return None
    minimum = max(1, 2 * ('chain' in requests), ('new' in requests) + ('existing' in requests))
    desired = int(rng.integers(minimum, min(8, count + 1) + 1))
    if requests:
        desired = min(desired, minimum + 2)
    plan = _plan(rng, _sites(labels), colors, ink, desired, requests)
    if plan is None:
        return None
    arrows, assignments = plan
    clean = colors[labels]
    raw, painted = clean.copy(), clean.copy()
    for arrow in arrows:
        for p in arrow['write']:
            raw[p] = ink
        if arrow['explicit']:
            raw[arrow['center']] = arrow['payload']
    for target, payload in assignments.items():
        painted[labels == target] = payload
    try:
        ranked, counts = _ranked(painted)
        if len(ranked) < (3 if requests else 2):
            return None
        output = _pack(painted)
        if _histogram(painted) == _histogram(clean) or np.array_equal(output, painted):
            return None
        info = _decode(raw)
        if (not np.array_equal(info['clean'], clean) or not np.array_equal(info['painted'], painted)
                or len(info['arrows']) != desired):
            return None
        features = _features(info)
        if any(not features[k] for k in requests) or (requests and not features['partial']):
            return None
    except ValueError:
        return None
    return dict(input=raw, output=output)


def _signature(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return reproducible examples. All retries and working arrays are bounded."""
    _need(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    _need(_integer(num_train) and 2 <= num_train <= 10, "num_train must be an integer in 2..10")
    _need(_integer(num_test) and 1 <= num_test <= 4, "num_test must be an integer in 1..4")
    _need(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    ink = int(rng.integers(10))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            requests = ({'chain', 'locality'} if index == 0 else {'new', 'existing'}) if split == 'train' and index < 2 else set()
            for attempt in range(160):
                pair = _candidate(rng, ink, split, requests, fallback=attempt >= 120)
                if pair is not None and _signature(pair['input']) not in seen:
                    seen.add(_signature(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction exhausted")
    # Pair order carries no instruction meaning; keep the two teaching cases
    # but vary their displayed order, including for the minimal 2/1 request.
    rng.shuffle(puzzle['train'])
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate returned arrays using reconstructed input-only semantics."""
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "bad puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, "bad pair counts")
    seen, inks, taught = set(), set(), set()
    training_partial = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "bad pair keys")
            raw, output = _grid(pair['input']), _grid(pair['output'])
            _need(raw.shape == output.shape, "packing must preserve shape")
            signature = _signature(raw)
            _need(signature not in seen, "duplicate input")
            seen.add(signature)
            info = _decode(raw)
            inks.add(info['ink'])
            features = _features(info)
            _need(np.array_equal(_pack(info['painted']), output), "wrong transfer/packing output")
            _need(info['ink'] not in output, "ink was not erased")
            # Count by ORIGINAL region IDs, aggregating equal final colors.
            inventory = Counter()
            for label, points in enumerate(info['cells']):
                inventory[info['assignments'].get(label, info['colors'][label])] += len(points)
            _need(inventory == Counter(map(int, output.ravel())), "wrong exact post-transfer inventory")
            _need(_histogram(info['clean']) != _histogram(output), "omitting A would leave the answer unchanged")
            _need(not np.array_equal(info['painted'], output), "omitting B would leave the answer unchanged")
            if split == 'train':
                taught.update(k for k in ('chain', 'locality', 'new', 'existing') if features[k])
                training_partial |= features['partial']
    _need(len(inks) == 1, "generated puzzle must have a consistent T ink")
    _need(taught == {'chain', 'locality', 'new', 'existing'}, "missing distinguishing demonstrations")
    _need(training_partial, "training must reveal bottom-up partial columns")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _fixture_test():
    # Five unequal vertical territories with repeated fill 2 in IDs 0 and 2.
    # Chain 2->3->2 and explicit new 7 into the final region. The original
    # territory areas are 36,48,36,48,24, known without using the decoder.
    raw = np.repeat(np.array([[2] * 3 + [3] * 4 + [2] * 3 + [4] * 4 + [5] * 2], dtype=np.int8), 12, axis=0)
    for center, payload in (((2, 1), 1), ((7, 5), 1), ((3, 12), 7)):
        for p in _geometry(center, (0, 1), raw.shape)['write']:
            raw[p] = 1
        raw[center] = payload
    # Simultaneous final inventory: 2:84,4:48,3:36,7:24.
    stream = [2] * 84 + [4] * 48 + [3] * 36 + [7] * 24
    expected = np.empty_like(raw)
    for index, value in enumerate(stream):
        expected[11 - index % 12, index // 12] = value
    before = raw.copy()
    _need(np.array_equal(solve(raw), expected), "hand-counted regression failed")
    info = _decode(raw)
    _need(info['clean'][7, 5] == 3 and info['painted'][7, 5] == 2, "bad host restoration")
    # Replacing the singleton payload changes an entire 24-cell territory.
    changed = raw.copy()
    changed[3, 12] = 6
    _need(np.count_nonzero(solve(changed) == 6) == 24, "literal payload was not expanded")
    # An existing non-host color aggregates with its original territories.
    changed[3, 12] = 2
    _need(_histogram(solve(changed)) == (0, 0, 108, 36, 48, 0, 0, 0, 0, 0),
          "existing payload was not aggregated")
    # Add a separated second instruction into the same target, contradicting
    # its first incoming payload. There is no priority or last-write rule.
    conflicting = raw.copy()
    for p in _geometry((9, 8), (0, -1), raw.shape)['write']:
        conflicting[p] = 1
    conflicting[9, 8] = 6
    for invalid in (lambda: solve(conflicting),
                    lambda: _pack(np.array([[2, 2, 3, 3]], dtype=np.int8))):
        try:
            invalid()
        except ValueError:
            pass
        else:
            raise AssertionError("conflicting transfers or tied ranks accepted")
    _need(np.array_equal(raw, before), "solver mutated the fixture input")


def self_test():
    """42 seeds, semantic ablations, metamorphic checks, and count boundaries.

    Does not retain generated puzzles, logs, or execution traces. It returns a
    small descriptive metric dictionary so callers can save one compact report.
    """
    _fixture_test()
    shapes, partitions, profiles, fingerprints, inks, directions = set(), set(), set(), set(), set(), set()
    totals = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                raw, output = pair['input'], pair['output']
                info = _decode(raw)
                f = _features(info)
                totals['pairs'] += 1
                for name in ('chain', 'locality', 'new', 'existing', 'nonrectangular', 'aggregation',
                             'aligned', 'partial', 'cycle', 'fanout', 'multiple_incoming'):
                    totals[name] += bool(f[name])
                shapes.add(raw.shape)
                partitions.add(hashlib.sha256(info['ids'].tobytes()).hexdigest())
                profiles.add(tuple(sorted(_histogram(output))))
                inks.add(info['ink'])
                directions.update(a['direction'] for a in info['arrows'])
                for name in ('territories', 'arrows', 'palette'):
                    totals[f'{name}_{f[name]}'] += 1
                _need(np.array_equal(solve(raw), output), "solve mismatch")
                # Geometry rotations change where instructions appear, but
                # leave the inventory invariant. A half-turn keeps dimensions.
                _need(np.array_equal(solve(np.rot90(raw, 2)), output), "geometry leaked into packing")
                # Any bijection on ARC colors preserves rank/count semantics.
                remap = np.array([(int(c) + 3) % 10 for c in range(10)], dtype=np.int8)
                _need(np.array_equal(solve(remap[raw]), remap[output]), "literal color shortcut")
                # Verify stream blocks directly rather than only calling pack.
                colors, counts = _ranked(info['painted'])
                stream = output[::-1, :].T.ravel()
                offset = 0
                for color, count in zip(colors, counts):
                    _need(np.all(stream[offset:offset + count] == color), "bad block/traversal")
                    offset += int(count)
                if f['chain']:
                    first, second = next((a, b) for a in info['arrows'] for b in info['arrows']
                                         if not a['explicit'] and not b['explicit'] and a['target'] == b['source']
                                         and a['source'] != b['target'] and a['payload'] != b['payload'])
                    cascading = info['painted'].copy()
                    cascading[info['ids'] == second['target']] = first['payload']
                    _need(_histogram(cascading) != _histogram(output), "simultaneity witness vanished")
        if seed < 2:
            _need(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterministic generation")
    _need(len(fingerprints) == 42 and len(shapes) >= 50 and len(partitions) >= 150, "insufficient structural variation")
    _need(len(profiles) >= 150 and inks == set(range(10)) and directions == set(DIRS), "insufficient role variation")
    _need(all(totals[name] >= 20 for name in ('chain', 'locality', 'new', 'existing', 'nonrectangular', 'aggregation')), "missing concept variation")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(1234 + 10 * a + b, a, b)
            _need(len(puzzle['train']) == a and len(puzzle['test']) == b, "ignored explicit counts")
            validate_generated_puzzle(puzzle)
    for a, b in ((1, 1), (11, 1), (4, 0), (5, 5), (3, 4), (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    return dict(seeds=42, shapes=len(shapes), partitions=len(partitions), profiles=len(profiles),
                distinct_puzzles=len(fingerprints), metrics=dict(sorted(totals.items())))
