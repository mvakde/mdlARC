"""Modal exceptions supply the vectors for stable whole-stack deduplication.

No parent generator, file, seed, or training example is consulted by solve().
Generation and self_test keep only small arrays and aggregate measurements in
memory. They write no files, logs, JSON, or execution traces.
"""

from collections import Counter
from itertools import permutations
import hashlib

import numpy as np


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, low, high=None):
    _need(isinstance(value, (int, np.integer))
          and not isinstance(value, (bool, np.bool_)), name + " must be an integer")
    value = int(value)
    _need(value >= low and (high is None or value <= high), name + " out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "bad grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "grid must have integer cells")
    _need(bool(np.all((0 <= a) & (a <= 9))), "bad grid colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _columns(a):
    return [tuple(map(int, a[:, j])) for j in range(a.shape[1])]


def _first_indices(a):
    seen, indices = set(), []
    for j, vector in enumerate(_columns(a)):
        if vector not in seen:
            seen.add(vector)
            indices.append(j)
    return indices


def _decode(value):
    """Segment visible lattice lines, identify the mode, select each exception."""
    a = _grid(value)
    lattice = int(a[0, 0])
    rs = list(map(int, np.flatnonzero(np.all(a == lattice, axis=1))))
    cs = list(map(int, np.flatnonzero(np.all(a == lattice, axis=0))))
    _need(len(rs) >= 2 and len(cs) >= 3, "not a bordered panel lattice")
    _need(rs[0] == cs[0] == 0 and rs[-1] == a.shape[0] - 1
          and cs[-1] == a.shape[1] - 1, "incomplete outside border")
    _need(rs[1] >= 2 and cs[1] >= 2, "empty panels")
    _need(all(y - x == rs[1] for x, y in zip(rs, rs[1:]))
          and all(y - x == cs[1] for x, y in zip(cs, cs[1:])), "irregular lattice")
    panels = [[a[r+1:s, c+1:d] for c, d in zip(cs, cs[1:])]
              for r, s in zip(rs, rs[1:])]
    _need(all(np.all(p != lattice) for row in panels for p in row),
          "lattice color occurs inside a panel")
    _need(len(np.unique(a)) == 3, "need a lattice and two interior colors")
    counts = Counter(p.tobytes() for row in panels for p in row)
    maximum = max(counts.values())
    modes = [k for k, count in counts.items() if count == maximum]
    _need(len(modes) == 1, "modal master is tied")
    master_key = modes[0]
    master = next(p for row in panels for p in row if p.tobytes() == master_key)
    selected, positions = [], []
    for row in panels:
        indices = [c for c, p in enumerate(row) if p.tobytes() != master_key]
        _need(len(indices) == 1, "each band needs exactly one visible nonmaster")
        positions.append(indices[0])
        selected.append(row[indices[0]])
    return dict(grid=a, lattice=lattice, rows=rs, cols=cs, panels=panels,
                master=master, selected=selected, positions=positions,
                ph=rs[1]-1, pw=cs[1]-1, bands=len(panels), copies=len(cs)-1)


def _stack(info, selected=None):
    result = np.full((info['grid'].shape[0], info['pw']), info['lattice'], dtype=np.int8)
    for r, s, p in zip(info['rows'], info['rows'][1:],
                       info['selected'] if selected is None else selected):
        result[r+1:s] = p
    return result


def _frame(interior, lattice):
    result = np.full((interior.shape[0], interior.shape[1]+2), lattice, dtype=np.int8)
    result[:, 1:-1] = interior
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """B selects and stacks; A deduplicates full-height interior columns.

    Separator rows participate in vectors and are retained. Side borders are
    structural framing, excluded from deduplication and restored on both sides.
    Malformed, tied, and missing-exception inputs raise ValueError.
    """
    info = _decode(input_grid)
    stacked = _stack(info)
    return _frame(stacked[:, _first_indices(stacked)], info['lattice'])


def _mask(rng, h, k):
    """Mix silhouettes, profiles, lines, islands, and unconstrained occupancy."""
    rr, cc = np.indices((h, k))
    style = int(rng.integers(6))
    if style == 0:
        a = rng.random((h, k)) < float(rng.uniform(0.2, 0.8))
    elif style == 1:
        a = np.zeros((h, k), dtype=bool)
        for j in range(k):
            start = int(rng.integers(h))
            a[start:int(rng.integers(start+1, h+1)), j] = True
    elif style == 2:
        period = int(rng.integers(2, 5))
        a = (rr + int(rng.integers(1, 3))*cc) % period == int(rng.integers(period))
    elif style == 3:
        a = (rr == 0) | (rr == h-1) | (cc == 0) | (cc == k-1)
        if rng.random() < 0.5:
            a = ~a
    elif style == 4:
        a = (abs(rr - (h-1)/2) + abs(cc - (k-1)/2)) <= float(rng.uniform(1, h/2+1))
    else:
        a = ((rr+cc) % 2 == 0) & (rng.random((h, k)) < 0.8)
    # The shared prototype must contain both colors: balanced witnesses swap
    # opposite cells without changing counts. This is a construction condition.
    if not a[:, 0].any():
        a[int(rng.integers(h)), 0] = True
    if a[:, 0].all():
        a[int(rng.integers(h)), 0] = False
    a[:, 1] = a[:, 0]
    return a


def _order(rng, k, w):
    """Unequal multiplicities and arbitrary first/last occurrence order."""
    values = list(range(k)) + [0, 1]
    values += list(map(int, rng.integers(k, size=w-len(values))))
    for _ in range(64):
        rng.shuffle(values)
        first = list(dict.fromkeys(values))
        last = {v: i for i, v in enumerate(values)}
        last_order = [v for i, v in enumerate(values) if last[v] == i]
        separated = any(values[i] == values[j] for i in range(w) for j in range(i+2, w))
        if len(set(values[:k])) < k and first != last_order and separated:
            return values.copy()
    # Guaranteed bounded fallback, with an early separated repeat and the
    # second type last: cropping and last-occurrence ordering both fail.
    return [0, 1, 0] + list(range(2, k)) + [0]*(w-k-2) + [1]


def _closest_vector(rng, original, used, require_change=False):
    h = len(original)
    candidates = []
    for code in range(1 << h):
        v = np.array([(code >> r) & 1 for r in range(h)], dtype=bool)
        distance = int(np.count_nonzero(v != original))
        if tuple(v) not in used and (distance > 0 or not require_change):
            candidates.append((distance, v))
    _need(bool(candidates), "internal vector capacity exhausted")
    distance = min(d for d, _ in candidates)
    nearest = [v for d, v in candidates if d == distance]
    return nearest[int(rng.integers(len(nearest)))].copy()


def _assemble(master, variants, positions, copies, lattice):
    h, w = master.shape
    a = np.full((len(variants)*(h+1)+1, copies*(w+1)+1), lattice, dtype=np.int8)
    for b, variant in enumerate(variants):
        for c in range(copies):
            r0, c0 = 1+b*(h+1), 1+c*(w+1)
            a[r0:r0+h, c0:c0+w] = variant if c == positions[b] else master
    return a


def _make_pair(rng, geometry, palette, balanced, placement):
    h, w, bands, copies, k = geometry
    lattice, carrier, motif = palette
    order = _order(rng, k, w)
    base = _mask(rng, h, k)
    if balanced:
        base[:, 2] = base[:, 0]
    types = [base.copy() for _ in range(bands)]
    used = {tuple(base[:, 0])}
    if balanced:
        changed = base[:, 0].copy()
        for bit in (False, True):
            r = int(rng.choice(np.flatnonzero(base[:, 0] == bit)))
            changed[r] = ~changed[r]
        types[0][:, 2] = changed
        used.add(tuple(changed))
        # Keep other first-band types equal to their master prototypes so the
        # balanced panel's *complete* histogram remains unchanged, also at k>3.
        for j in range(3, k):
            vector = _closest_vector(rng, base[:, j], used)
            base[:, j] = vector
            for t in types:
                t[:, j] = vector
            used.add(tuple(vector))
        start = k
    else:
        start = 2
    # In the first band, types 2..k-1 are distinct from the shared 0/1
    # prototype and one another. Thus no unintended global class merger can
    # occur. There are at least 8 available bit vectors and at most 5 types.
    for j in range(start, k):
        vector = _closest_vector(rng, base[:, j], used, require_change=(j == 2))
        types[0][:, j] = vector
        used.add(tuple(vector))
    for b in range(1, bands):
        for j in range(3 if balanced else 2, k):
            count = int(rng.integers(0, min(2, h//2)+1))
            for r in rng.choice(h, count, replace=False):
                types[b][int(r), j] = ~types[b][int(r), j]
    distinguishing_band = int(rng.integers(1, bands))
    distinguishing_row = int(rng.integers(h))
    types[distinguishing_band][distinguishing_row, 1] ^= True
    # Types 0 and 1 are equal in every other band and differ at exactly one
    # cell globally. Deduplication of individual panels therefore fails.
    master_mask = base[:, order].copy()
    last_one = max(i for i, v in enumerate(order) if v == 1)
    break_row = int(rng.integers(h))
    master_mask[break_row, last_one] ^= True
    if balanced:
        # A second, opposite edit makes the master and a visible exceptional
        # panel have identical color histograms. Count-based selection fails.
        duplicate_zero = [i for i, v in enumerate(order) if v == 0][1]
        opposite = np.flatnonzero(base[:, 0] != base[break_row, 0])
        master_mask[int(rng.choice(opposite)), duplicate_zero] ^= True
    master = np.where(master_mask, motif, carrier).astype(np.int8)
    variants = [np.where(t[:, order], motif, carrier).astype(np.int8) for t in types]
    positions = list(map(int, rng.integers(copies, size=bands)))
    if placement == 'varied':
        positions[1] = (positions[0]+int(rng.integers(1, copies))) % copies
    elif placement == 'repeated':
        positions[1] = positions[0]
    elif rng.random() < 0.2:
        positions = [positions[0]]*bands
    a = _assemble(master, variants, positions, copies, lattice)
    # Construct the label from displayed variants and known class identities,
    # not by calling solve or the input parser.
    s = np.full((a.shape[0], w), lattice, dtype=np.int8)
    for b, v in enumerate(variants):
        s[1+b*(h+1):1+b*(h+1)+h] = v
    first = [order.index(v) for v in dict.fromkeys(order)]
    return {'input': a, 'output': _frame(s[:, first], lattice)}


def _geometry(rng, k, bands):
    h = int(rng.integers(3, min(8, 29//bands-1)+1))
    w = int(rng.integers(k+2, 9))
    copies = int(rng.integers(3, 29//(w+1)+1))
    return h, w, bands, copies, k


def _unique(pair, seen):
    if _key(pair['input']) not in seen:
        return pair
    # At most 13 earlier inputs versus 720 injective three-color assignments.
    colors = np.unique(pair['input'])
    for palette in permutations(range(10), 3):
        lookup = np.arange(10, dtype=np.int8)
        lookup[colors] = palette
        candidate = {role: lookup[a] for role, a in pair.items()}
        if _key(candidate['input']) not in seen:
            return candidate
    raise AssertionError("unreachable palette exhaustion")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    ne = _integer(num_test, 'num_test', 1, 4)
    _need(nt > ne, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(2))  # More bands, or more distinct stack columns.
    shared_palette = rng.random() < 0.2
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    puzzle, seen, first_bands = {'train': [], 'test': []}, set(), None
    for index in range(nt+ne):
        training = index < nt
        if training:
            choices = [2, 3, 4]
            if index == 1:
                choices.remove(first_bands)
            bands = int(rng.choice(choices))
            if index == 0:
                first_bands = bands
            k = 3 if shift else int(rng.integers(3, 6))
        else:
            bands = int(rng.integers(5, 8)) if shift == 0 else int(rng.integers(2, 7))
            k = int(rng.integers(3, 6)) if shift == 0 else int(rng.integers(4, 6))
        if not shared_palette:
            palette = tuple(map(int, rng.choice(10, 3, replace=False)))
        balanced = index == 1 or (index > 1 and rng.random() < 0.4)
        placement = 'varied' if index == 0 else 'repeated' if index == 1 else 'random'
        pair = _unique(_make_pair(rng, _geometry(rng, k, bands), palette,
                                  balanced, placement), seen)
        seen.add(_key(pair['input']))
        puzzle['train' if training else 'test'].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure_pair(pair):
    _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "bad pair keys")
    info = _decode(pair['input'])
    out = _grid(pair['output'])
    s = _stack(info)
    keys = _columns(s)
    first = _first_indices(s)
    expected = _frame(s[:, first], info['lattice'])
    _need(np.array_equal(out, expected), "output does not preserve exact first-seen stack columns")
    _need(out.shape[0] == info['grid'].shape[0], "height changed")
    _need(np.all(out[:, [0, -1]] == info['lattice']), "side borders lost")
    _need(np.all(out[info['rows']] == info['lattice']), "separators changed")
    _need(len(set(_columns(out[:, 1:-1]))) == out.shape[1]-2, "retained duplicate vector")
    # No raw-input or master-only deduplication can replace exception selection.
    master_stack = _stack(info, [info['master']]*info['bands'])
    master_result = _frame(master_stack[:, _first_indices(master_stack)], info['lattice'])
    master_keys = _columns(info['master'])
    unique = [s[:, j] for j in first]
    last = {v: j for j, v in enumerate(keys)}
    last_order = [v for j, v in enumerate(keys) if last[v] == j]
    local = any(keys[j] != keys[k] and np.array_equal(p[:, j], p[:, k])
                for p in info['selected'] for j in range(info['pw']) for k in range(j))
    near = any(np.count_nonzero(a != b) == 1 for i, a in enumerate(unique) for b in unique[i+1:])
    hist = [tuple(np.bincount(v, minlength=10)) for v in unique]
    equal_counts = len(set(hist)) < len(hist)
    balanced = any(np.array_equal(np.bincount(p.ravel(), minlength=10),
                                 np.bincount(info['master'].ravel(), minlength=10))
                   for p in info['selected'])
    splits = any(master_keys[j] == master_keys[k] and keys[j] != keys[k]
                 for j in range(info['pw']) for k in range(j))
    merges = any(master_keys[j] != master_keys[k] and keys[j] == keys[k]
                 for j in range(info['pw']) for k in range(j))
    segments = [out[r+1:t, 1:-1].tobytes() for r, t in zip(info['rows'], info['rows'][1:])]
    flags = dict(nonadjacent=any(keys[j] == keys[k] for j in range(len(keys))
                                for k in range(j+2, len(keys))),
                 crop_fails=not np.array_equal(s[:, :len(first)], out[:, 1:-1]),
                 last_differs=last_order != [keys[j] for j in first],
                 local_collision=local, near=near, count_collision=equal_counts,
                 balanced_panel=balanced, master_splits=splits, master_merges=merges,
                 without_b_diff=not np.array_equal(master_result, out),
                 band_order=segments != segments[::-1],
                 varied_positions=len(set(info['positions'])) > 1,
                 repeated_positions=len(set(info['positions'])) < info['bands'])
    return info, flags, len(first)


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "bad puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal split counts")
    seen, train_info, test_info, evidence = set(), [], [], Counter()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info, flags, u = _measure_pair(pair)
            key = _key(info['grid'])
            _need(key not in seen, "duplicate input")
            seen.add(key)
            if split == 'train':
                train_info.append((info, u))
                evidence.update({k: int(v) for k, v in flags.items()})
            else:
                test_info.append((info, u))
    _need(all(evidence[k] for k in (
        'nonadjacent', 'crop_fails', 'last_differs', 'local_collision', 'near',
        'count_collision', 'balanced_panel', 'master_splits', 'master_merges',
        'without_b_diff', 'band_order', 'varied_positions', 'repeated_positions')),
        "training does not distinguish required alternatives")
    _need(len({(i['ph'], i['pw'], i['bands'], i['copies']) for i, _ in train_info}) > 1,
          "training geometry is constant")
    for info, u in test_info:
        _need(info['bands'] > max(i['bands'] for i, _ in train_info)
              or u > max(n for _, n in train_info), "missing structural train/test shift")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds; aggregate features, count boundaries, and mutations.

    Returns a small JSON-compatible report but never writes it. No execution
    trace or corpus is retained. This is generator-owned testing, not a blind
    model prediction or protected independent solver.
    """
    puzzles, geometries, palettes, structures, orders, input_shapes = set(), set(), set(), set(), set(), set()
    band_counts, heights, widths, unique_counts, copies, shifts, lattice_colors = (set() for _ in range(7))
    features, pairs = Counter(), 0
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.add(_fingerprint(p))
        _need(_fingerprint(p) == _fingerprint(generate_puzzle(seed)), "nondeterministic seed")
        train = [_measure_pair(pair) for pair in p['train']]
        for split in ('train', 'test'):
            for pair in p[split]:
                before = _key(pair['input'])
                _need(np.array_equal(solve(pair['input']), pair['output']), "solver disagreement")
                _need(_key(pair['input']) == before, "solver mutated input")
                info, flags, u = _measure_pair(pair)
                pairs += 1
                features.update({k: int(v) for k, v in flags.items()})
                geometries.add((info['ph'], info['pw'], info['bands'], info['copies'], u))
                band_counts.add(info['bands'])
                heights.add(info['ph'])
                widths.add(info['pw'])
                unique_counts.add(u)
                copies.add(info['copies'])
                lattice_colors.add(info['lattice'])
                palettes.add(tuple(map(int, np.unique(info['grid']))))
                input_shapes.add(info['grid'].shape)
                s = _stack(info)
                keys = _columns(s)
                ordered = list(dict.fromkeys(keys))
                orders.add(tuple(ordered.index(v) for v in keys))
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in info['grid'].flat)
                structures.add((info['grid'].shape, hashlib.sha256(normalized).hexdigest()))
                if split == 'test':
                    shifts.add('bands' if info['bands'] > max(i['bands'] for i, _, _ in train) else 'types')
    _need(len(puzzles) == 42 and len(structures) >= 150, "insufficient structural variation")
    _need(len(geometries) >= 50 and len(input_shapes) >= 30 and len(orders) >= 20,
          "geometry or duplicate-placement variation too narrow")
    _need(band_counts >= {2, 3, 4, 5, 6, 7} and heights >= {3, 4, 5, 6, 7, 8}
          and widths == {5, 6, 7, 8} and unique_counts == {3, 4, 5}
          and copies == {3, 4} and shifts == {'bands', 'types'}
          and lattice_colors == set(range(10)), "missing variation dimension")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(500+10*nt+ne, nt, ne)
            _need((len(p['train']), len(p['test'])) == (nt, ne), "counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3, False), (3.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Metamorphic checks: colors, panel positions, and band order are visible
    # roles. Deduplication itself must remain on the complete aligned stack.
    p = generate_puzzle(0)
    pair = p['train'][0]
    info = _decode(pair['input'])
    lookup = np.array([7, 3, 8, 1, 9, 0, 6, 2, 4, 5], dtype=np.int8)
    _need(np.array_equal(solve(lookup[pair['input']]), lookup[pair['output']]),
          "palette equivariance failed")
    moved = _assemble(info['master'], info['selected'],
                      [(v+1) % info['copies'] for v in info['positions']],
                      info['copies'], info['lattice'])
    _need(np.array_equal(solve(moved), pair['output']), "horizontal position leaked")
    reversed_input = _assemble(info['master'], info['selected'][::-1], info['positions'][::-1],
                               info['copies'], info['lattice'])
    first = _first_indices(_stack(info))
    reversed_interior = _stack(info, info['selected'][::-1])[:, first]
    _need(np.array_equal(solve(reversed_input), _frame(reversed_interior, info['lattice'])),
          "band-order equivariance failed")
    for defect in ('uncompressed', 'border', 'cell', 'missing', 'extra_exception'):
        bad = {s: [{r: a.copy() for r, a in v.items()} for v in p[s]] for s in p}
        target = bad['train'][0]
        binfo = _decode(target['input'])
        if defect == 'uncompressed':
            target['output'] = _frame(_stack(binfo), binfo['lattice'])
        elif defect == 'border':
            target['output'] = target['output'][:, :-1]
        elif defect == 'cell':
            target['output'][1, 1] = (int(target['output'][1, 1])+1) % 10
        else:
            c = binfo['positions'][0]
            if defect == 'extra_exception':
                c = (c+1) % binfo['copies']
            c0 = 1+c*(binfo['pw']+1)
            target['input'][1:1+binfo['ph'], c0:c0+binfo['pw']] = (
                binfo['master'] if defect == 'missing' else binfo['selected'][0])
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted mutation: " + defect)
    # Hand-built members outside the sampler's geometry/class-count schedule.
    # The rule uses visible panels, not knowledge of generated prototypes.
    master = np.array([[0, 0, 2]], dtype=np.int8)
    variants = [np.array([[0, 2, 2]], dtype=np.int8),
                np.array([[2, 2, 0]], dtype=np.int8)]
    small = _assemble(master, variants, [0, 1], 2, 9)
    expected = np.array([[9,9,9,9,9], [9,0,2,2,9], [9,9,9,9,9],
                         [9,2,2,0,9], [9,9,9,9,9]], dtype=np.int8)
    _need(np.array_equal(solve(small), expected), "two-copy/global-master case failed")
    try:
        solve(_assemble(master, [variants[0], variants[0]], [0, 1], 2, 9))
    except ValueError:
        pass
    else:
        raise AssertionError("tied master accepted")
    master = np.array([[0,2,0], [0,0,0]], dtype=np.int8)
    variant = np.array([[2,2,2], [0,0,0]], dtype=np.int8)
    small = _assemble(master, [variant, variant], [1, 1], 3, 9)
    expected = np.array([[9,9,9], [9,2,9], [9,0,9], [9,9,9],
                         [9,2,9], [9,0,9], [9,9,9]], dtype=np.int8)
    _need(np.array_equal(solve(small), expected), "singleton/framing/row preservation failed")
    return dict(seeds=42, default_pairs=pairs, distinct_puzzles=len(puzzles),
                normalized_inputs=len(structures), input_shapes=len(input_shapes),
                geometries=len(geometries), duplicate_orders=len(orders),
                palettes=len(palettes), band_counts=sorted(band_counts),
                panel_heights=sorted(heights), panel_widths=sorted(widths),
                retained_counts=sorted(unique_counts), copies=sorted(copies),
                shifts=sorted(shifts), feature_counts=dict(features),
                legal_count_combinations=30, mutation_checks=5, semantic_boundary_checks=3,
                verification_scope='generator-owned checks only')
