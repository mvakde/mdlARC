"""Endpoint-labelled loop systems seed a transverse nearest-stripe field.

Only NumPy and the standard library are used. No parent modules or fixtures are
loaded. Generation constructs answers from placed geometry; solve reconstructs
ownership and enclosed cells from the input alone.
"""
from collections import deque
import hashlib
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
HOST_SEEDS = (0, 754343015, 803488351, 437272918, 756089574, 535357962,
              591055449, 856487973, 155155153, 86226736, 301809200,
              528877344, 546607734, 26056961, 381385828, 502519595,
              524378688, 577291194, 989055090, 929497804, 208784790,
              438934992, 40122365, 481290871, 33696868, 730494492,
              674035365, 968871142, 654593711, 320670599, 643083074,
              439481733, 391654211, 475346304, 938685152, 361839328,
              899650363, 372025808, 416519039, 561157086, 284247654,
              931230570)


def _int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _ri(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def _pick(rng, values):
    return values[_ri(rng, 0, len(values) - 1)]


def _near(cells, shape, steps=D8):
    h, w = shape
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps
            if 0 <= r + dr < h and 0 <= c + dc < w} - set(cells)


def _components(cells, steps=D8):
    left = set(cells)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        group, queue = {start}, [start]
        for r, c in queue:
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    group.add(q)
                    queue.append(q)
        result.append(group)
    return result


def _cells(mask):
    return set(zip(*np.where(mask)))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(n < 1 or n > 30 for n in a.shape)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a


def _analyze(input_grid):
    """Find the visible guide, then trace systems with the guide removed.

    Junction cells are retained as the only bridges across the guide. The
    ordinary guide never connects systems. Background cavities use four-way
    connectivity, whereas thin strings and shells use eight-way connectivity.
    """
    original = _grid(input_grid)
    candidates = []
    for transpose in (False, True):
        a = original.T if transpose else original
        h, w = a.shape
        bg = int(a[0, 0])
        for g in range(3, w - 3):
            guide_values = set(map(int, a[:, g]))
            if len(guide_values) != 2 or bg in guide_values:
                continue
            guide = int(a[0, g])
            if a[-1, g] != guide:
                continue
            junction = next(iter(guide_values - {guide}))
            positions = tuple(map(int, np.flatnonzero(a[:, g] == junction)))
            if len(positions) < 2 or positions[0] == 0 or positions[-1] == h - 1:
                continue
            border = np.concatenate((np.delete(a[0], g), np.delete(a[-1], g),
                                     a[1:-1, 0], a[1:-1, -1]))
            if np.any(border != bg):
                continue
            outside = a.copy()
            outside[:, g] = bg
            if np.any((outside == guide) | (outside == junction)):
                continue
            stripe_cells = {(r, c) for r in positions
                            for c in (g - 2, g - 1, g + 1, g + 2)}
            stripe_colors = {int(a[p]) for p in stripe_cells}
            if len(stripe_colors) != 1:
                continue
            wall = stripe_colors.pop()
            if wall in (bg, guide, junction):
                continue
            groups = _components(_cells((a == wall) | (a == junction)))
            owner = {p: i for i, group in enumerate(groups) for p in group}
            if len(groups) != len(positions):
                continue
            if any(sum((r, g) in group for r in positions) != 1 for group in groups):
                continue
            marker_cells = _cells(~np.isin(a, (bg, guide, junction, wall)))
            markers = {}
            valid = True
            for p in sorted(marker_cells):
                contacts = _near({p}, a.shape) & _cells(a == wall)
                owners = {owner[q] for q in contacts}
                # A rotated 2x2 corner cup: three wall cells, one paint cell.
                offsets = {(r - p[0], c - p[1]) for r, c in contacts}
                cups = [{(dr, 0), (0, dc), (dr, dc)}
                        for dr in (-1, 1) for dc in (-1, 1)]
                if len(owners) != 1 or offsets not in cups:
                    valid = False
                    break
                i = owners.pop()
                if i in markers:
                    valid = False
                    break
                markers[i] = (p, int(a[p]))
            if not valid or len(markers) != len(groups):
                continue
            cavities = []
            for region in _components(_cells(a == bg), D4):
                if any(r in (0, h - 1) or c in (0, w - 1) for r, c in region):
                    continue
                boundary = _near(region, a.shape, D4)
                if not boundary or any(a[p] != wall for p in boundary):
                    valid = False
                    break
                owners = {owner[p] for p in boundary}
                if len(owners) != 1:
                    valid = False
                    break
                cavities.append((region, owners.pop()))
            if not valid or {i for _, i in cavities} != set(range(len(groups))):
                continue
            labels = tuple(markers[owner[(r, g)]][1] for r in positions)
            candidates.append(dict(a=a, transpose=transpose, bg=bg, guide=guide,
                                   junction=junction, wall=wall, g=g,
                                   positions=positions, labels=labels,
                                   groups=groups, owner=owner, markers=markers,
                                   cavities=cavities, stripe_cells=stripe_cells))
    if len(candidates) != 1:
        raise ValueError("The guide, source bars and endpoint ownership must be unique")
    return candidates[0]


def _nearest_field(shape, positions, labels, g, junction):
    out = np.empty(shape, dtype=np.int8)
    for r in range(shape[0]):
        distances = [abs(r - p) for p in positions]
        nearest = {labels[i] for i, d in enumerate(distances) if d == min(distances)}
        out[r, :] = next(iter(nearest)) if len(nearest) == 1 else junction
    out[:, g] = junction
    return out


def _interval_field(shape, positions, labels, g, junction):
    """Generation/validation construction using adjacent intervals, not argmin."""
    out = np.empty(shape, dtype=np.int8)
    out[:positions[0] + 1, :] = labels[0]
    for left, right, lc, rc in zip(positions, positions[1:], labels, labels[1:]):
        for r in range(left + 1, right + 1):
            dl, dr = r - left, right - r
            out[r, :] = junction if dl == dr and lc != rc else (lc if dl <= dr else rc)
    out[positions[-1]:, :] = labels[-1]
    out[:, g] = junction
    return out


def _overlay(field, a, wall, stripe_cells, markers, cavities):
    out = field.copy()
    for p in _cells(a == wall) - stripe_cells:
        out[p] = wall
    for p, color in markers.values():
        out[p] = color
    for region, owner in cavities:
        color = markers[owner][1]
        for p in region:
            out[p] = color
    return out


def _nearest_marker_wrong(positions, labels, g, markers):
    """A spatially nearest paint cup cannot replace tracing to a source bar."""
    for r, label in zip(positions, labels):
        distances = [(abs(r - p[0]) + abs(g - p[1]), color) for p, color in markers.values()]
        best = min(d for d, color in distances)
        if label not in {color for d, color in distances if d == best}:
            return True
    return False


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the input; literal colors, source labels and ownership are visible."""
    d = _analyze(input_grid)
    field = _nearest_field(d['a'].shape, d['positions'], d['labels'], d['g'], d['junction'])
    out = _overlay(field, d['a'], d['wall'], d['stripe_cells'], d['markers'], d['cavities'])
    return out.T.copy() if d['transpose'] else out.copy()


def _contour(rng, simple=False):
    """A connected row-interval cavity with a minimal closed four-way shell."""
    kind = 'box' if simple else _pick(rng, ('box', 'box', 'step', 'slant', 'diamond', 'thin'))
    hh, ww = (1, 1) if simple else (_ri(rng, 1, 3), _ri(rng, 1, 4))
    inner = set()
    if kind == 'diamond':
        inner = {(r, c) for r in range(3) for c in range(3) if abs(r - 1) + abs(c - 1) <= 1}
    else:
        if kind == 'thin':
            hh, ww = 1, _ri(rng, 2, 5)
        for r in range(hh):
            left = r // 2 if kind == 'slant' else (int(r == 1) if kind == 'step' else 0)
            width = max(1, ww - int(kind == 'step' and r == 1))
            inner.update((r, c) for c in range(left, left + width))
    # All sampled row intervals overlap; the shell remains unambiguous.
    if len(_components(inner, D4)) != 1:
        inner = {(r, c) for r in range(hh) for c in range(ww)}
    shell = {(r + dr, c + dc) for r, c in inner for dr, dc in D4} - inner
    r0, c0 = min(r for r, c in shell), min(c for r, c in shell)
    return ({(r - r0, c - c0) for r, c in inner},
            {(r - r0, c - c0) for r, c in shell})


def _place(rng, cells, bounds, occupied, shape, attempts=60):
    lo, hi, left, right = bounds
    hh, ww = max(r for r, c in cells) + 1, max(c for r, c in cells) + 1
    if hh > hi - lo + 1 or ww > right - left + 1:
        return None
    for _ in range(attempts):
        dr, dc = _ri(rng, lo, hi - hh + 1), _ri(rng, left, right - ww + 1)
        placed = {(r + dr, c + dc) for r, c in cells}
        if not (placed | _near(placed, shape)) & occupied:
            return dr, dc
    return None


def _path(rng, starts, goals, allowed):
    directions = [D8[i] for i in rng.permutation(8)]
    queue = deque(sorted(starts))
    previous = {p: None for p in starts}
    end = None
    while queue:
        p = queue.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in directions:
            q = p[0] + dr, p[1] + dc
            if q in allowed and q not in previous:
                previous[q] = p
                queue.append(q)
    if end is None:
        return None
    path = set()
    while end is not None:
        path.add(end)
        end = previous[end]
    return path


def _scene(rng, symbols, parities, transpose, rich, sparse, teach=False, simple=False):
    count = len(symbols)
    if sparse:
        gaps = [_ri(rng, 5, 9) * 2 + parities[0]]
    else:
        base = 4 if count == 4 else 6
        gaps = [base + p + 2 * _ri(rng, 0, 1) for p in parities]
    span = sum(gaps)
    h = _ri(rng, max(18, span + 5), 30)
    start = _ri(rng, 2, h - 3 - span)
    positions = [start]
    for gap in gaps:
        positions.append(positions[-1] + gap)
    w = 30 if simple else _ri(rng, 24, 30)
    g = _ri(rng, 10, w - 11)
    colors = list(map(int, rng.permutation(10)))
    bg, guide, junction, wall = colors[:4]
    palette = {s: colors[4 + i] for i, s in enumerate(dict.fromkeys(symbols))}
    labels = [palette[s] for s in symbols]
    a = np.full((h, w), bg, dtype=np.int8)
    a[:, g] = guide
    stripe_cells = {(r, c) for r in positions for c in (g - 2, g - 1, g + 1, g + 2)}
    roots = [{(r, c) for c in range(g - 2, g + 3)} for r in positions]
    for i, r in enumerate(positions):
        for p in roots[i]:
            a[p] = wall
        a[r, g] = junction
    side_phase = _ri(rng, 0, 1)
    sides = [(i + side_phase) % 2 for i in range(count)]
    bounds, allowed_cores = [], []
    for i, side in enumerate(sides):
        same = [j for j in range(count) if sides[j] == side]
        rank = same.index(i)
        lo = 1 if rank == 0 else (positions[same[rank - 1]] + positions[i]) // 2 + 2
        hi = h - 2 if rank == len(same) - 1 else (positions[i] + positions[same[rank + 1]]) // 2 - 2
        left, right = (1, g - 4) if side == 0 else (g + 4, w - 2)
        bounds.append((lo, hi, left, right))
        allowed_cores.append({(r, c) for r in range(lo, hi + 1) for c in range(left, right + 1)})
    regions, cups, markers = [], [], {}
    occupied = set().union(*roots)
    for i in range(count):
        nloops = 2 if teach and i == count // 2 else (1 if simple else _ri(rng, 1, 2 if rich else 1))
        for _ in range(nloops):
            inner, shell = _contour(rng, simple)
            offset = _place(rng, inner | shell, bounds[i], occupied, a.shape)
            if offset is None:
                return None
            dr, dc = offset
            inner = {(r + dr, c + dc) for r, c in inner}
            shell = {(r + dr, c + dc) for r, c in shell}
            regions.append((inner, shell, i))
            occupied |= inner | shell
        # Rotate the source cup independently of the stripe and paint color.
        dr, dc = _pick(rng, (-1, 1)), _pick(rng, (-1, 1))
        marker = (int(dr == -1), int(dc == -1))
        cup = {(0, 0), (0, 1), (1, 0), (1, 1)} - {marker}
        offset = _place(rng, cup | {marker}, bounds[i], occupied, a.shape)
        if offset is None:
            return None
        rr, cc = offset
        cup = {(r + rr, c + cc) for r, c in cup}
        marker = marker[0] + rr, marker[1] + cc
        cups.append(cup)
        markers[i] = marker, labels[i]
        occupied |= cup | {marker}
    loop_cells = set().union(*(inner | shell for inner, shell, _ in regions))
    marker_cells = {p for p, col in markers.values()}
    all_cups = set().union(*cups)
    wires = [root.copy() for root in roots]
    for i in map(int, rng.permutation(count)):
        side, r = sides[i], positions[i]
        sign = -1 if side == 0 else 1
        connector = {(r, g + sign * k) for k in (2, 3, 4)}
        domain = allowed_cores[i] | connector | roots[i]
        targets = [(cups[i], set())]
        indices = [j for j, region in enumerate(regions) if region[2] == i]
        rng.shuffle(indices)
        targets += [(regions[j][1], regions[j][0]) for j in indices]
        own = wires[i]
        for target, inner in targets:
            foreign_wires = set().union(*(wires[j] for j in range(count) if j != i))
            blocked = loop_cells | all_cups | marker_cells
            blocked |= _near(marker_cells, a.shape)
            blocked |= foreign_wires | _near(foreign_wires, a.shape)
            # Do not touch any target except the current one or connected ones.
            protected = (loop_cells | all_cups) - own - target - inner
            blocked |= _near(protected, a.shape)
            allowed = (domain - blocked) | own | target
            goals = target
            path = _path(rng, own, goals, allowed)
            if path is None:
                return None
            own |= path | target
        wires[i] = own
    for own in wires:
        for p in own:
            a[p] = wall
    for inner, shell, i in regions:
        for p in shell:
            a[p] = wall
    for p, color in markers.values():
        a[p] = color
    for r in positions:
        a[r, g] = junction
    field = _interval_field(a.shape, positions, labels, g, junction)
    cavities = [(inner, i) for inner, shell, i in regions]
    out = _overlay(field, a, wall, stripe_cells, markers, cavities)
    # Every pair makes endpoint ownership observable beyond local field color.
    foreign = sum(field[p] not in (labels[i], junction) for inner, i in cavities for p in inner)
    endpoint_field = _nearest_field(a.shape, [markers[i][0][0] for i in range(count)], labels, g, junction)
    if not foreign or np.array_equal(field[:, 0], endpoint_field[:, 0]):
        return None
    if teach and not _nearest_marker_wrong(positions, labels, g, markers):
        return None
    inp = a.T.copy() if transpose else a.copy()
    out = out.T.copy() if transpose else out.copy()
    try:
        analysis = _analyze(inp)
        reconstructed = solve(inp)
    except ValueError:
        return None
    intended = {(frozenset(inner), positions[i]) for inner, i in cavities}
    discovered = {(frozenset(inner), next(r for r in positions
                   if analysis['owner'][(r, g)] == i)) for inner, i in analysis['cavities']}
    if intended != discovered:
        return None
    if not np.array_equal(reconstructed, out):
        return None
    return {"input": inp, "output": out}


def _pattern(rng, count):
    if count == 2:
        return (0, _ri(rng, 0, 1))
    if count == 3:
        return _pick(rng, ((0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 1, 2)))
    return _pick(rng, ((0, 1, 0, 1), (0, 0, 1, 2), (0, 1, 1, 0),
                       (0, 1, 2, 3), (0, 1, 2, 0), (0, 0, 0, 1)))


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _int(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _int(num_train) or not _int(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("Require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    regime = _ri(rng, 0, 2)
    base_orientation = _ri(rng, 0, 1)
    mixed = bool(_ri(rng, 0, 1))
    specs = []
    # Even and odd anchors jointly teach all equality/parity combinations.
    for parity in (0, 1):
        labels = (0, 0, 1) if _ri(rng, 0, 1) else (1, 0, 0)
        specs.append((labels, (parity, parity), True, False, parity == 0))
    for _ in range(num_train - 2):
        count = _ri(rng, 2, 3)
        labels = _pattern(rng, count)
        # All-equal examples cannot exhibit a foreign-label cavity.
        if len(set(labels)) == 1:
            labels = (0, 1)
        specs.append((labels, tuple(map(int, rng.integers(0, 2, count - 1))), bool(_ri(rng, 0, 1)), False, False))
    rng.shuffle(specs)
    test_specs = []
    for _ in range(num_test):
        count = 4 if regime == 0 else (2 if regime == 2 else _ri(rng, 2, 4))
        labels = _pattern(rng, count)
        if len(set(labels)) == 1:
            labels = (0, 1)
        test_specs.append((labels, tuple(map(int, rng.integers(0, 2, count - 1))), True, regime == 2, False))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, plan in (("train", specs), ("test", test_specs)):
        for labels, parities, rich, sparse, teach in plan:
            orientation = _ri(rng, 0, 1) if mixed else base_orientation
            for attempt in range(160):
                pair = _scene(rng, labels, parities, orientation, rich, sparse, teach, simple=attempt >= 100)
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Bounded placement budget exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(d):
    field = _nearest_field(d['a'].shape, d['positions'], d['labels'], d['g'], d['junction'])
    equality_parity = {(a == b, (y - x) % 2) for a, b, x, y in
                       zip(d['labels'], d['labels'][1:], d['positions'], d['positions'][1:])}
    foreign = sum(field[p] not in (d['markers'][i][1], d['junction'])
                  for cells, i in d['cavities'] for p in cells)
    return equality_parity, foreign


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train_features, train_multiloop = set(), set(), False
    train_tracing = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            d = _analyze(a)
            assert all(1 <= sum(i == owner for cells, i in d['cavities']) <= 2
                       for owner in range(len(d['groups'])))
            field = _interval_field(d['a'].shape, d['positions'], d['labels'], d['g'], d['junction'])
            expected = _overlay(field, d['a'], d['wall'], d['stripe_cells'], d['markers'], d['cavities'])
            observed = b.T if d['transpose'] else b
            assert np.array_equal(observed, expected), "Incorrect field, source bar or cavity"
            assert np.array_equal(solve(a.copy()), b)
            assert d['bg'] not in b and d['guide'] not in b
            assert np.all(observed[:, d['g']] == d['junction'])
            for region, i in d['cavities']:
                assert all(observed[p] == d['markers'][i][1] for p in region)
            for p, color in d['markers'].values():
                assert observed[p] == color
            assert all(observed[p] == d['wall'] for p in _cells(d['a'] == d['wall']) - d['stripe_cells'])
            features, foreign = _features(d)
            assert foreign > 0, "Cavity copying must differ from nearest-line fill"
            ordered_markers = [d['markers'][d['owner'][(r, d['g'])]] for r in d['positions']]
            endpoint_field = _nearest_field(d['a'].shape, [p[0] for p, color in ordered_markers],
                                            d['labels'], d['g'], d['junction'])
            assert not np.array_equal(field[:, 0], endpoint_field[:, 0]), "Endpoint rows are not stripe coordinates"
            # Removing A leaves exterior background; removing B gives wrong holes.
            cavities = set().union(*(cells for cells, i in d['cavities']))
            exterior = _cells(d['a'] == d['bg']) - cavities
            assert exterior and all(observed[p] != d['bg'] for p in exterior)
            if split == 'train':
                train_features |= features
                train_multiloop |= len(d['cavities']) > len(d['positions'])
                train_tracing |= _nearest_marker_wrong(d['positions'], d['labels'], d['g'], d['markers'])
    assert {(False, 0), (False, 1), (True, 0), (True, 1)} <= train_features
    assert train_multiloop, "Demonstrate one endpoint supplying several contours"
    assert train_tracing, "Demonstrate tracing rather than assigning the closest paint cup"


def self_test():
    """Stream 42 seeds; keep only small measurements and SHA-256 fingerprints."""
    fingerprints, shapes, patterns, counts, cavity_shapes = set(), set(), set(), set(), set()
    backgrounds, roles, orientations, seams = set(), set(), set(), set()
    foreign_pairs = multiloop = shifted = total = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _analyze(pair['input'])
                total += 1
                digest.update(repr(pair['input'].shape).encode())
                digest.update(pair['input'].tobytes())
                digest.update(pair['output'].tobytes())
                shapes.add(pair['input'].shape)
                names = {}
                patterns.add(tuple(names.setdefault(c, len(names)) for c in d['labels']))
                counts.add(len(d['positions']))
                backgrounds.add(d['bg'])
                roles.add((d['bg'], d['guide'], d['junction'], d['wall']))
                orientations.add(d['transpose'])
                seams.add(sum(a != b and (y - x) % 2 == 0 for a, b, x, y in
                              zip(d['labels'], d['labels'][1:], d['positions'], d['positions'][1:])))
                foreign_pairs += _features(d)[1] > 0
                multiloop += len(d['cavities']) > len(d['positions'])
                for cells, i in d['cavities']:
                    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
                    cavity_shapes.add(tuple(sorted((r - r0, c - c0) for r, c in cells)))
                # Changing an endpoint changes its stripe territory AND its holes.
                if split == 'test':
                    chosen, (p, paint) = next(iter(d['markers'].items()))
                    replacement = next(c for c in d['labels'] if c != paint)
                    changed = d['a'].copy()
                    changed[p] = replacement
                    changed = changed.T.copy() if d['transpose'] else changed
                    prediction = solve(changed)
                    normalized = prediction.T if d['transpose'] else prediction
                    r = next(r for r in d['positions'] if d['owner'][(r, d['g'])] == chosen)
                    assert normalized[r, 0] == replacement
                    for cells, i in d['cavities']:
                        assert all(normalized[q] == (replacement if i == chosen else d['markers'][i][1])
                                   for q in cells)
                    assert not np.array_equal(prediction, pair['output'])
                    # Geometry and role-based colors survive symmetries/recoloring.
                    perm = np.array([(3 * c + 1) % 10 for c in range(10)], dtype=np.int8)
                    assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
                    assert np.array_equal(solve(pair['input'].T), pair['output'].T)
                    assert np.array_equal(solve(np.fliplr(pair['input'])), np.fliplr(pair['output']))
                    shifted += 1
        fingerprints.add(digest.hexdigest())
    assert len(fingerprints) == 42 and total == foreign_pairs == 210 and shifted == 42
    assert len(shapes) >= 35 and counts == {2, 3, 4} and len(patterns) >= 9
    assert len(cavity_shapes) >= 15 and multiloop >= 20
    assert len(backgrounds) >= 8 and len(roles) >= 100 and orientations == {False, True}
    assert {0, 1, 2} <= seams
    first, second = generate_puzzle(42, 10, 4), generate_puzzle(42, 10, 4)
    for split in first:
        for a, b in zip(first[split], second[split]):
            assert all(np.array_equal(a[k], b[k]) for k in ('input', 'output'))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generate_puzzle(0, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Corrupt one cavity to ensure the validator rejects a partial fill.
    puzzle = generate_puzzle(0, 2, 1)
    pair = puzzle['train'][0]
    d = _analyze(pair['input'])
    p = next(iter(d['cavities'][0][0]))
    if d['transpose']:
        p = p[1], p[0]
    old = int(pair['output'][p])
    pair['output'][p] = d['junction']
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incorrect cavity accepted")
    pair['output'][p] = old
    return {"seeds": 42, "pairs": total, "distinct_puzzles": len(fingerprints),
            "shapes": len(shapes), "label_patterns": len(patterns),
            "cavity_shapes": len(cavity_shapes), "multiloop_pairs": multiloop,
            "foreign_territory_pairs": foreign_pairs, "role_palettes": len(roles)}
