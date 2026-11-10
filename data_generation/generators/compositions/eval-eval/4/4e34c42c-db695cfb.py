"""Stamp registration disambiguated by closing broken diagonal circuits.

Only the standard library and NumPy are used.  In particular, solve() sees no
examples, RNG, generation coordinates, or puzzle-level parameters.
"""
from collections import defaultdict
import itertools

import numpy as np


PRIMARY = 1
SECONDARY = 6


def _check_grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid, dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _groups(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    found.add(p)
                    stack.append(p)
        result.append(found)
    return result


def _origin(points):
    return min(r for r, c in points), min(c for r, c in points)


def _shift(p, delta):
    return p[0]+delta[0], p[1]+delta[1]


def _fingerprint(cells):
    top, left = _origin(cells)
    return tuple(sorted((r-top, c-left, int(v)) for (r, c), v in cells.items()))


def _bounds(points):
    top, left = _origin(points)
    return top, left, max(r for r, c in points), max(c for r, c in points)


def _parse(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert counts.max()*2 > grid.size, "Background must occupy over half the input"
    background = int(values[np.argmax(counts)])
    assert background not in (PRIMARY, SECONDARY)
    candidates = [(int(n), int(v)) for v, n in zip(values, counts)
                  if v not in (background, PRIMARY, SECONDARY)]
    assert candidates
    candidates.sort(reverse=True)
    assert len(candidates) == 1 or candidates[0][0] > candidates[1][0]
    backing = candidates[0][1]
    assert candidates[0][0] > max(int(np.sum(grid == PRIMARY)),
                                int(np.sum(grid == SECONDARY)))
    components = _groups(map(tuple, np.argwhere(grid != background)))
    assert 2 <= len(components) <= 4, "Family has two to four plates"
    parts, stamps = [], defaultdict(list)
    for i, component in enumerate(components):
        top, left = _origin(component)
        cells = {(r-top, c-left): int(grid[r, c]) for r, c in component}
        key_cells = {p: v for p, v in cells.items()
                     if v not in (backing, PRIMARY, SECONDARY)}
        labels = {}
        for group in _groups(key_cells):
            key = _fingerprint({p: key_cells[p] for p in group})
            assert len({v for r, c, v in key}) >= 2, "Stamp must be multicolor"
            anchor = _origin(group)
            stamps[key].append((i, anchor))
            labels.update({p: key for p in group})
        assert key_cells and sum(v == backing for v in cells.values()) > len(cells)//2
        parts.append({'cells': cells, 'labels': labels})
    assert len(stamps) == len(parts)-1
    assert sum(v == PRIMARY for part in parts for v in part['cells'].values()) == 2*(len(parts)-1)
    adjacency = [[] for _ in parts]
    for key, occurrences in sorted(stamps.items()):
        owners = sorted({i for i, p in occurrences})
        assert len(owners) == 2 and len(occurrences) == 3
        aa = [p for i, p in occurrences if i == owners[0]]
        bb = [p for i, p in occurrences if i == owners[1]]
        assert sorted((len(aa), len(bb))) == [1, 2]
        shifts = sorted({(a[0]-b[0], a[1]-b[1]) for a in aa for b in bb})
        i, j = owners
        adjacency[i].append((j, shifts))
        adjacency[j].append((i, [(-r, -c) for r, c in shifts]))
    reached, stack = {0}, [0]
    while stack:
        i = stack.pop()
        for j, shifts in adjacency[i]:
            if j not in reached:
                reached.add(j)
                stack.append(j)
    assert len(reached) == len(parts), "Stamp graph must be a connected tree"
    return background, backing, parts, stamps, adjacency


def _placements(adjacency):
    """Enumerate stamp-supported translations; never rank by overlap or area."""
    offsets = {0: (0, 0)}

    def search():
        if len(offsets) == len(adjacency):
            yield dict(offsets)
            return
        choices = []
        for i in sorted(offsets):
            for j, deltas in adjacency[i]:
                if j not in offsets:
                    choices.append((j, i, deltas))
        assert choices
        j, i, deltas = min(choices, key=lambda x: (x[0], x[1]))
        for delta in deltas:
            offsets[j] = _shift(offsets[i], delta)
            if all(any(offsets[k] == _shift(offsets[j], d) for d in ds)
                   for k, ds in adjacency[j] if k in offsets):
                yield from search()
            del offsets[j]
    yield from search()


def _merge(parts, offsets, backing):
    mask, ink, labels, owners = set(), {}, {}, {}
    for i, part in enumerate(parts):
        for p, color in part['cells'].items():
            q = _shift(p, offsets[i])
            mask.add(q)
            if color == backing:
                continue  # A neutral plate is transparent to an inked marking.
            if q in ink:
                assert ink[q] == color
                assert p in part['labels'] and q in labels
                assert part['labels'][p] == labels[q]
            else:
                ink[q] = color
            if p in part['labels']:
                labels[q] = part['labels'][p]
            elif color == PRIMARY:
                assert q not in owners
                owners[q] = i
    top, left, bottom, right = _bounds(mask)
    assert bottom-top < 30 and right-left < 30
    return mask, ink, set(labels), owners


def _segment(a, b):
    assert a[0] < b[0] and abs(b[0]-a[0]) == abs(b[1]-a[1])
    slope = 1 if b[1] > a[1] else -1
    return tuple((a[0]+t, a[1]+slope*t) for t in range(b[0]-a[0]+1))


def _diagonal(point, slope, bounds):
    top, left, bottom, right = bounds
    return {(r, point[1]+slope*(r-point[0])) for r in range(top, bottom+1)
            if left <= point[1]+slope*(r-point[0]) <= right}


def _wires(mask, ink, keys, owners):
    terminals = sorted(p for p, v in ink.items() if v == PRIMARY)
    edges, degrees = [], {p: 0 for p in terminals}
    for a, b in itertools.combinations(terminals, 2):
        if a[0] != b[0] and abs(a[0]-b[0]) == abs(a[1]-b[1]):
            edges.append((a, b))
            degrees[a] += 1
            degrees[b] += 1
    assert terminals and all(v == 1 for v in degrees.values()), "All terminals need unique mates"
    assert all(owners[a] != owners[b] for a, b in edges), "Wires join different plates"
    segments = [_segment(a, b) for a, b in edges]
    assert all(len(s) >= 4 and set(s) <= mask and not (set(s) & keys) for s in segments)
    return segments


def _geometry(mask, ink, keys, owners):
    segments = _wires(mask, ink, keys, owners)
    terminals = {p for p, v in ink.items() if v == PRIMARY}
    receivers = {p for p, v in ink.items() if v == SECONDARY}
    union = set().union(*map(set, segments))
    active = receivers & union
    rays, trigger_slopes = [], []
    for p in sorted(active):
        containing = [s for s in segments if p in s]
        assert len(containing) == 1, "A two-slope receiver is ambiguous"
        s = containing[0]
        slope = 1 if s[-1][1] > s[0][1] else -1
        ray = _diagonal(p, -slope, _bounds(mask)) & mask
        assert ray & union == {p} and not (ray & keys)
        assert not (ray & set(terminals))
        rays.append(ray)
        trigger_slopes.append(slope)
    return {'segments': segments, 'rays': rays, 'active': active,
            'inactive': receivers-active, 'trigger_slopes': trigger_slopes,
            'slopes': [1 if s[-1][1] > s[0][1] else -1 for s in segments],
            'trigger_counts': [len(set(s) & receivers) for s in segments]}


def _render(mask, ink, backing, background, geometry=None):
    top, left, bottom, right = _bounds(mask)
    out = np.full((bottom-top+1, right-left+1), background, dtype=np.int8)
    for r, c in mask:
        out[r-top, c-left] = backing
    for (r, c), v in ink.items():
        out[r-top, c-left] = v
    if geometry is not None:
        for segment in geometry['segments']:
            for r, c in segment:
                out[r-top, c-left] = PRIMARY
        for ray in geometry['rays']:
            for r, c in ray:
                out[r-top, c-left] = SECONDARY
    return out


def _solve_details(input_grid):
    grid = _check_grid(input_grid)
    background, backing, parts, stamps, adjacency = _parse(grid)
    registrations, solutions = [], []
    for offsets in _placements(adjacency):
        try:
            mask, ink, keys, owners = _merge(parts, offsets, backing)
        except AssertionError:
            continue
        candidate = {'offsets': offsets, 'mask': mask, 'ink': ink,
                     'keys': keys, 'owners': owners,
                     'overlap': sum(len(p['cells']) for p in parts)-len(mask)}
        top, left, bottom, right = _bounds(mask)
        candidate['area'] = (bottom-top+1)*(right-left+1)
        registrations.append(candidate)
        try:
            _wires(mask, ink, keys, owners)
        except AssertionError:
            continue
        solutions.append(candidate)
    assert len(solutions) == 1, "Circuit must select exactly one stamp registration"
    chosen = solutions[0]
    chosen['geometry'] = _geometry(chosen['mask'], chosen['ink'], chosen['keys'], chosen['owners'])
    out = _render(chosen['mask'], chosen['ink'], backing, background, chosen['geometry'])
    return out, {'background': background, 'backing': backing, 'parts': parts,
                 'stamps': stamps, 'registrations': registrations, 'chosen': chosen}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the unique registered circuit using input pixels alone."""
    return _solve_details(input_grid)[0]


def _plate_mask(rng, count):
    h = int(rng.integers(6, 10 if count < 4 else 9))
    w = int(rng.integers(7, 12 if count < 4 else 10))
    mask = {(r, c) for r in range(h) for c in range(w)}
    # Optional transparent corner cutouts vary silhouette, never orientation.
    if rng.random() < .45:
        dh, dw = int(rng.integers(1, 3)), int(rng.integers(1, 4))
        rr = range(dh) if rng.random() < .5 else range(h-dh, h)
        cc = range(dw) if rng.random() < .5 else range(w-dw, w)
        mask -= {(r, c) for r in rr for c in cc}
    return mask


def _stamp(rng, colors):
    h, w = ((2, 2), (2, 3), (3, 2))[int(rng.integers(3))]
    values = rng.choice(colors, size=h*w, replace=True)
    values[0], values[-1] = colors[0], colors[1]
    return {(r, c): int(values[r*w+c]) for r in range(h) for c in range(w)}


def _put_stamp(rng, part, pattern):
    mask, used = part['mask'], part['occupied']
    h, w = max(r for r, c in mask)+1, max(c for r, c in mask)+1
    ph, pw = max(r for r, c in pattern)+1, max(c for r, c in pattern)+1
    feasible = []
    for r in range(h-ph+1):
        for c in range(w-pw+1):
            points = {_shift(p, (r, c)) for p in pattern}
            halo = {(u+dr, v+dc) for u, v in points for dr, dc in
                    ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))}
            if points <= mask and not (halo & used):
                feasible.append((r, c))
    assert feasible
    origin = feasible[int(rng.integers(len(feasible)))]
    key = _fingerprint(pattern)
    for p, color in pattern.items():
        q = _shift(p, origin)
        part['ink'][q] = color
        part['labels'][q] = key
        used.add(q)
    return origin


def _make_parts(rng, count, backing, colors, tree):
    parts = [{'mask': _plate_mask(rng, count), 'occupied': set(),
              'ink': {}, 'labels': {}} for _ in range(count)]
    offsets, identities = {0: (0, 0)}, set()
    for parent, child in tree:
        for _ in range(30):
            pattern = _stamp(rng, colors)
            key = _fingerprint(pattern)
            if key not in identities:
                break
        else:
            raise AssertionError('Distinct stamp budget')
        identities.add(key)
        parent_ports = [_put_stamp(rng, parts[parent], pattern) for _ in range(2)]
        child_port = _put_stamp(rng, parts[child], pattern)
        selected = parent_ports[int(rng.integers(2))]
        offsets[child] = (offsets[parent][0]+selected[0]-child_port[0],
                          offsets[parent][1]+selected[1]-child_port[1])
    for part in parts:
        part['cells'] = {p: part['ink'].get(p, backing) for p in part['mask']}
    return parts, offsets


def _add_circuit(rng, parts, offsets, backing, tree, slopes, trigger_counts, inert):
    mask, ink, keys, owners = _merge(parts, offsets, backing)
    global_masks = [{_shift(p, offsets[i]) for p in part['mask']} for i, part in enumerate(parts)]
    terminals, segments = [], []
    for edge_index, (parent, child) in enumerate(tree):
        possibilities = []
        for a in sorted(global_masks[parent]-keys):
            for b in sorted(global_masks[child]-keys):
                if a[0] == b[0] or abs(a[0]-b[0]) != abs(a[1]-b[1]):
                    continue
                slope = 1 if (a[0]-b[0])*(a[1]-b[1]) > 0 else -1
                if slopes is not None and slope != slopes[edge_index]:
                    continue
                aa, bb = sorted((a, b))
                segment = _segment(aa, bb)
                if (len(segment) < max(4, trigger_counts[edge_index]+3)
                        or not set(segment) <= mask or set(segment) & keys):
                    continue
                if any(abs(p[0]-q[0]) == abs(p[1]-q[1])
                       for p in (a, b) for q in terminals):
                    continue
                possibilities.append((a, b, segment))
        assert possibilities
        a, b, segment = possibilities[int(rng.integers(len(possibilities)))]
        for p, i in ((a, parent), (b, child)):
            local = (p[0]-offsets[i][0], p[1]-offsets[i][1])
            assert parts[i]['cells'][local] == backing
            parts[i]['cells'][local] = PRIMARY
            ink[p] = PRIMARY
            owners[p] = i
        terminals.extend((a, b))
        segments.append(segment)
    primary_union = set().union(*map(set, segments))
    receiver_points, ray_union = [], set()
    for index, segment in enumerate(segments):
        slope = 1 if segment[-1][1] > segment[0][1] else -1
        eligible = []
        for p in segment[1:-1]:
            ray = _diagonal(p, -slope, _bounds(mask)) & mask
            if len(ray) >= 3 and ray & primary_union == {p} and not (ray & keys):
                eligible.append((p, ray))
        number = trigger_counts[index]
        assert len(eligible) >= number
        for j in rng.permutation(len(eligible))[:number]:
            p, ray = eligible[int(j)]
            receiver_points.append(p)
            ray_union |= ray
    # Inert dots outside all rays visibly demonstrate conditional activation.
    pool = sorted(mask-keys-primary_union-set(terminals)-ray_union)
    assert len(pool) >= inert
    receiver_points += [pool[int(j)] for j in rng.permutation(len(pool))[:inert]]
    for p in receiver_points:
        options = [i for i, group in enumerate(global_masks) if p in group]
        i = options[int(rng.integers(len(options)))]
        local = (p[0]-offsets[i][0], p[1]-offsets[i][1])
        assert parts[i]['cells'][local] == backing
        parts[i]['cells'][local] = SECONDARY
    mask, ink, keys, owners = _merge(parts, offsets, backing)
    geometry = _geometry(mask, ink, keys, owners)
    return mask, ink, geometry


def _pack(rng, parts, background):
    shapes = [(max(r for r, c in p['cells'])+1, max(c for r, c in p['cells'])+1) for p in parts]
    total = sum(len(p['cells']) for p in parts)
    for attempt in range(60):
        if attempt < 40:
            h = int(rng.integers(max(s[0] for s in shapes), 31))
            w = int(rng.integers(max(s[1] for s in shapes), 31))
        else:
            h, w = 30, 30
        if h*w <= 2*total:
            continue
        occupied = np.zeros((h, w), dtype=bool)
        grid = np.full((h, w), background, dtype=np.int8)
        success = True
        for i in rng.permutation(len(parts)):
            i = int(i)
            ph, pw = shapes[i]
            feasible = [(r, c) for r in range(h-ph+1) for c in range(w-pw+1)
                        if not occupied[max(0, r-1):min(h, r+ph+1),
                                        max(0, c-1):min(w, c+pw+1)].any()]
            if not feasible:
                success = False
                break
            top, left = feasible[int(rng.integers(len(feasible)))]
            occupied[top:top+ph, left:left+pw] = True
            for (r, c), color in parts[i]['cells'].items():
                grid[top+r, left+c] = color
        if success:
            return grid
    raise AssertionError('Packing budget')


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _clipping(info):
    chosen = info['chosen']
    geometry = chosen['geometry']
    for p in geometry['active']:
        segment = next(s for s in geometry['segments'] if p in s)
        slope = 1 if segment[-1][1] > segment[0][1] else -1
        if _diagonal(p, -slope, _bounds(chosen['mask'])) - chosen['mask']:
            return True
    return False


def _finite_witness(info):
    chosen = info['chosen']
    geometry = chosen['geometry']
    wire_pixels = set().union(*map(set, geometry['segments']))
    ray_pixels = set().union(set(), *geometry['rays'])
    for s in geometry['segments']:
        slope = 1 if s[-1][1] > s[0][1] else -1
        continuation = _diagonal(s[0], slope, _bounds(chosen['mask'])) & chosen['mask']
        if continuation-wire_pixels-ray_pixels-set(chosen['ink']):
            return True
    return False


def _noncompact_witness(info):
    chosen = info['chosen']
    return any(c['overlap'] > chosen['overlap'] and c['area'] < chosen['area']
               for c in info['registrations'])


def _reentry(info):
    for ray in info['chosen']['geometry']['rays']:
        rows = sorted(r for r, c in ray)
        if any(b-a > 1 for a, b in zip(rows, rows[1:])):
            return True
    return False


def _generate_pair(rng, count, background, backing, colors, witness=None,
                   allow_reentry=True):
    for attempt in range(250):
        try:
            # Ordered random-parent trees include chains, stars, and bent merges.
            tree = [(int(rng.integers(i)), i) for i in range(1, count)]
            parts, offsets = _make_parts(rng, count, backing, colors, tree)
            if witness is None:
                slopes = None
                triggers = [int(rng.choice((0, 0, 1, 1, 2))) for _ in tree]
                inert = int(rng.integers(4))
            else:
                slopes, triggers, inert = witness['slopes'], witness['triggers'], witness['inert']
            mask, ink, geometry = _add_circuit(rng, parts, offsets, backing, tree,
                                                slopes, triggers, inert)
            expected = _render(mask, ink, backing, background, geometry)
            grid = _pack(rng, parts, background)
            predicted, info = _solve_details(grid)
            assert np.array_equal(predicted, expected)
            assert len(info['registrations']) >= 2, 'Need genuine registration ambiguity'
            assert allow_reentry or not _reentry(info), 'Gap reappearance needs a demonstration'
            if witness is not None and witness.get('noncompact'):
                assert _noncompact_witness(info)
                assert _clipping(info), 'Teach silhouette clipping'
                assert _finite_witness(info), 'Teach finite primary wires'
                assert any(len(ray) >= 5 for ray in info['chosen']['geometry']['rays'])
            return {'input': grid, 'output': expected}
        except AssertionError:
            continue
    raise RuntimeError('Could not construct a uniquely recoverable circuit in 250 attempts')


def _check_counts(num_train, num_test):
    for name, n, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not lo <= n <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = [int(v) for v in rng.permutation([0, 2, 3, 4, 5, 7, 8, 9])]
    fixed_background = bool(rng.integers(2))
    sign = int(rng.choice((-1, 1)))
    teachers = [
        (2, {'slopes': [sign], 'triggers': [2], 'inert': 1, 'noncompact': True}),
        (3, {'slopes': [-sign, sign], 'triggers': [1, 0], 'inert': 1}),
    ]
    teachers += [(int(rng.integers(2, 4)), None) for _ in range(num_train-2)]
    rng.shuffle(teachers)
    puzzle, used, taught_reentry = {'train': [], 'test': []}, set(), False
    tests = [(4, None)] + [(int(rng.integers(3, 5)), None) for _ in range(num_test-1)]
    for split, specs in (('train', teachers), ('test', tests)):
        for count, witness in specs:
            for _ in range(12):
                background = palette[0] if fixed_background else int(rng.choice(palette))
                available = [v for v in palette if v != background]
                backing = int(rng.choice(available))
                key_colors = [v for v in available if v != backing]
                rng.shuffle(key_colors)
                key_colors = key_colors[:int(rng.integers(2, 4))]
                pair = _generate_pair(rng, count, background, backing, key_colors, witness,
                                      allow_reentry=(split == 'train' or taught_reentry))
                key = _key(pair['input'])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    if split == 'train':
                        taught_reentry |= _reentry(_solve_details(pair['input'])[1])
                    break
            else:
                raise RuntimeError('Distinct-input budget')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    used, train_infos, test_infos = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            grid, output = _check_grid(pair['input']), _check_grid(pair['output'])
            assert _key(grid) not in used
            used.add(_key(grid))
            predicted, info = _solve_details(grid)
            assert np.array_equal(predicted, output), 'Wrong assembled circuit or crop'
            assert len(info['registrations']) >= 2, 'Stamps alone must not determine the answer'
            chosen, backing = info['chosen'], info['backing']
            g = chosen['geometry']
            assert len(g['segments']) == len(info['parts'])-1
            assert any(v == PRIMARY for v in chosen['ink'].values())
            assert np.sum(output == PRIMARY) > sum(v == PRIMARY for v in chosen['ink'].values())
            assert not np.array_equal(output, _render(chosen['mask'], chosen['ink'], backing, info['background']))
            if split == 'train':
                train_infos.append(info)
            else:
                test_infos.append(info)
    assert set().union(*(set(i['chosen']['geometry']['trigger_slopes']) for i in train_infos)) == {-1, 1}
    assert any(0 in i['chosen']['geometry']['trigger_counts'] for i in train_infos)
    assert any(max(i['chosen']['geometry']['trigger_counts']) >= 2 for i in train_infos)
    assert any(i['chosen']['geometry']['inactive'] for i in train_infos)
    assert any(_clipping(i) for i in train_infos)
    assert any(_finite_witness(i) for i in train_infos)
    assert any(_noncompact_witness(i) for i in train_infos)
    assert not any(_reentry(i) for i in test_infos) or any(_reentry(i) for i in train_infos)


def _puzzle_key(p):
    return tuple(tuple((_key(pair['input']), _key(pair['output'])) for pair in p[s]) for s in ('train', 'test'))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    records = []
    for p in puzzles:
        validate_generated_puzzle(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                _, info = _solve_details(pair['input'])
                g = info['chosen']['geometry']
                records.append((pair, info, g))
    assert len({pair['input'].shape for pair, i, g in records}) >= 35
    assert len({pair['output'].shape for pair, i, g in records}) >= 35
    assert {len(i['parts']) for pair, i, g in records} == {2, 3, 4}
    assert len({i['background'] for pair, i, g in records}) == 8
    assert len({i['backing'] for pair, i, g in records}) == 8
    assert len({key for pair, i, g in records for key in i['stamps']}) >= 80
    assert {n for pair, i, g in records for n in g['trigger_counts']} >= {0, 1, 2}
    assert len({len(s) for pair, i, g in records for s in g['segments']}) >= 5
    assert any(len(set(g['slopes'])) == 2 and len(i['parts']) == 4 for pair, i, g in records)
    assert any(not g['active'] for pair, i, g in records)
    assert any(_reentry(i) for pair, i, g in records)
    # Geometry regression: finite primary; active receiver; inert receiver;
    # full opposite diagonal clipped at a concave material silhouette.
    mask = {(r, c) for r in range(7) for c in range(7)}-{(r, c) for r in range(3) for c in range(4, 7)}
    ink = {(1, 1): 1, (5, 5): 1, (3, 3): 6, (6, 5): 6}
    g = _geometry(mask, ink, set(), {(1, 1): 0, (5, 5): 1})
    assert g['active'] == {(3, 3)} and g['inactive'] == {(6, 5)}
    assert g['rays'] == [{(r, 6-r) for r in range(7) if (r, 6-r) in mask}]
    out = _render(mask, ink, 2, 0, g)
    assert out[2, 2] == out[4, 4] == 1 and out[0, 6] == 0 and out[6, 0] == 6
    # Repacking and arbitrary relabeling of nonterminal/nonreceiver colors are
    # semantic invariances; endpoint/receiver colors are fixed family roles.
    pair, info, _ = records[0]
    repacked = _pack(np.random.default_rng(91), info['parts'], info['background'])
    assert np.array_equal(solve(repacked), pair['output'])
    lookup = np.arange(10, dtype=np.int8)
    free = [0, 2, 3, 4, 5, 7, 8, 9]
    lookup[free] = free[::-1]
    assert np.array_equal(solve(lookup[pair['input']]), lookup[pair['output']])
    assert _puzzle_key(generate_puzzle(818, 2, 1)) == _puzzle_key(generate_puzzle(818, 2, 1))
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(900+train*10+test, train, test)
            assert len(p['train']) == train and len(p['test']) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2),
                        (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    corrupted = {s: [{'input': pair['input'].copy(), 'output': pair['output'].copy()}
                     for pair in puzzles[0][s]] for s in ('train', 'test')}
    corrupted['test'][0]['output'][0, 0] = (int(corrupted['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupt output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds and count boundaries)')
