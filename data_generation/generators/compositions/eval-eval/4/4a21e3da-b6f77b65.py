"""Seam demolition: a border color key fractures a scaffold before gravity.

Only standard Python and NumPy are used. No parent data or seed is read by solve.
Rods are tuples (orientation, row, column, length, color).
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _cells(rod):
    o, r, c, n, _ = rod
    return frozenset((r + (k if o == 'V' else 0),
                      c + (k if o == 'H' else 0)) for k in range(n))


def _adjacent(a, b):
    return any((r + dr, c + dc) in b for r, c in a
               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))


def _parse(value):
    grid = _grid(value)
    h, w = grid.shape
    if min(h, w) < 7:
        raise ValueError("a seam scaffold needs at least a 7 by 7 tray")
    rim = np.zeros(grid.shape, dtype=bool)
    rim[[0, -1], :] = True
    rim[:, [0, -1]] = True
    colors, counts = np.unique(grid[rim], return_counts=True)
    bg = int(colors[np.argmax(counts)])
    marks = np.argwhere(rim & (grid != bg))
    if len(marks) != 1:
        raise ValueError("expected exactly one nonbackground rim marker")
    mr, mc = map(int, marks[0])
    if mr in (0, h - 1) and mc in (0, w - 1):
        raise ValueError("the marker cannot be a corner")
    key = int(grid[mr, mc])
    if mr == 0:
        direction, axis, side = (1, 0), (1, mc), 'top'
    elif mr == h - 1:
        direction, axis, side = (-1, 0), (1, mc), 'bottom'
    elif mc == 0:
        direction, axis, side = (0, 1), (0, mr), 'left'
    else:
        direction, axis, side = (0, -1), (0, mr), 'right'

    todo = {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
            if grid[r, c] != bg}
    rods = []
    while todo:
        p = min(todo)
        color = int(grid[p])
        component, stack = {p}, [p]
        todo.remove(p)
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in todo and grid[q] == color:
                    todo.remove(q)
                    component.add(q)
                    stack.append(q)
        rs, cs = zip(*component)
        if len(component) < 2 or (len(set(rs)) > 1 and len(set(cs)) > 1):
            raise ValueError("each source color component must be a straight rod")
        o = 'H' if len(set(rs)) == 1 else 'V'
        rod = (o, min(rs), min(cs), len(component), color)
        if _cells(rod) != component:
            raise ValueError("a source rod is not contiguous")
        rods.append(rod)
    if not rods:
        raise ValueError("empty scaffold")
    hits = sorted({p for rod in rods for p in _cells(rod) if p[axis[0]] == axis[1]})
    crossings = [p for p in rods if any(q[axis[0]] < axis[1] for q in _cells(p))
                 and any(q[axis[0]] > axis[1] for q in _cells(p))]
    if not hits or not crossings:
        raise ValueError("the seam must cross at least one rod")
    return {'grid': grid, 'bg': bg, 'key': key, 'marker': (mr, mc),
            'direction': direction, 'axis': axis, 'side': side,
            'rods': rods, 'hits': hits, 'crossings': crossings}


def _settle(bodies, shape, direction, pins=()):
    """Greatest simultaneous moving set; bodies are (footprint, color)."""
    h, w = shape
    dr, dc = direction
    pins = set(pins)
    result = [(frozenset(s), int(color)) for s, color in bodies]
    distances = [0] * len(result)
    # Each successful step strictly decreases the finite gravity potential.
    for _ in range(max(shape) * len(result) + 1):
        targets = [frozenset((r + dr, c + dc) for r, c in s) for s, _ in result]
        moving = {i for i, target in enumerate(targets)
                  if all(1 <= r < h - 1 and 1 <= c < w - 1 for r, c in target)
                  and not target & pins}
        while True:
            blocked = {i for i in moving if any(targets[i] & result[j][0]
                       for j in range(len(result)) if j not in moving)}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return result, distances
        for i in moving:
            result[i] = (targets[i], result[i][1])
            distances[i] += 1
    raise AssertionError("gravity potential did not converge")


def _parts(info, delete=True, fracture=True):
    pins, bodies = {}, []
    ai, av = info['axis']
    for rod in info['rods']:
        if delete and rod[4] == info['key']:
            continue
        cells = _cells(rod)
        if not fracture:
            bodies.append((cells, rod[4]))
            continue
        pins.update({p: rod[4] for p in cells if p[ai] == av})
        for sign in (-1, 1):
            part = frozenset(p for p in cells if (p[ai] - av) * sign > 0)
            if part:
                bodies.append((part, rod[4]))
    return pins, bodies


def _ray(info, nearest=False, full=False):
    dr, dc = info['direction']
    mr, mc = info['marker']
    distance = lambda p: (p[0] - mr) * dr + (p[1] - mc) * dc
    stop = min(map(distance, info['hits'])) if nearest else max(map(distance, info['hits']))
    if full:
        stop = info['grid'].shape[0 if dr else 1] - 1
    return [(mr + k * dr, mc + k * dc) for k in range(stop + 1)]


def _transform(info, delete=True, fracture=True, nearest=False, full=False):
    pins, bodies = _parts(info, delete, fracture)
    settled, distances = _settle(bodies, info['grid'].shape, info['direction'], pins)
    out = np.full(info['grid'].shape, info['bg'], dtype=np.int8)
    for cells, color in settled:
        for p in cells:
            out[p] = color
    # These overlays also give deliberately incorrect alternatives a comparable
    # completed guide, so the no-fracture check detects a mechanical difference.
    for p in _ray(info, nearest, full):
        out[p] = info['key']
    for p, color in pins.items():
        out[p] = color
    return out, pins, bodies, settled, distances


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer every convention from the input; no generation state is consulted."""
    info = _parse(input_grid)
    return _transform(info)[0]


def _drop(geometry, orientation, column, length, shape):
    h, w = shape
    rod = (orientation, 1, column, length, -1)
    occupied = set().union(*(_cells(p) for p in geometry)) if geometry else set()
    if not all(1 <= r < h - 1 and 1 <= c < w - 1 for r, c in _cells(rod)):
        return
    if _cells(rod) & occupied:
        return
    while True:
        target = (orientation, rod[1] + 1, column, length, -1)
        if any(r >= h - 1 for r, c in _cells(target)) or _cells(target) & occupied:
            geometry.append(rod)
            return
        rod = target


def _scaffold(rng, levels, kind, allow_beam_selection=False):
    """Compose stable geometry, with two physically diagnostic teaching cases."""
    base = int(rng.integers(3, 8))
    gaps = [int(rng.integers(2, 4 if levels >= 4 else 5)) for _ in range(levels - 1)]
    headroom = 7 if kind == 'command' else 4
    minimum = base + sum(g + 1 for g in gaps) + headroom
    # Five levels can otherwise need one more row than permitted.
    while minimum > 30:
        i = next((i for i, g in enumerate(gaps) if g > 2), None)
        if i is not None:
            gaps[i] -= 1
        else:
            base -= 1
        minimum = base + sum(g + 1 for g in gaps) + headroom
    h = int(rng.integers(max(9, minimum), 31))
    w = int(rng.integers(11, 31))
    seam = int(rng.integers(2, w - 2)) if kind == 'general' else int(rng.integers(4, w - 4))
    floor = h - 2
    target_side = int(rng.choice([-1, 1]))
    geometry, selected, beams = [], set(), []
    rows = [floor - base]
    for gap in gaps:
        rows.append(rows[-1] - gap - 1)

    def add(o, r, c, n, command=False):
        rod = (o, int(r), int(c), int(n), -1)
        assert n >= 2
        assert all(1 <= rr < h - 1 and 1 <= cc < w - 1 for rr, cc in _cells(rod))
        assert not any(_cells(rod) & _cells(q) for q in geometry)
        index = len(geometry)
        geometry.append(rod)
        if command:
            assert not any(_adjacent(_cells(rod), _cells(geometry[j])) for j in selected)
            selected.add(index)
        return index

    margin = 1 if kind == 'general' else 2
    intervals = [(int(rng.integers(1, seam - margin + 1)),
                  int(rng.integers(seam + margin, w - 1))) for _ in rows]
    support_columns = []
    for level, (r, (a, b)) in enumerate(zip(rows, intervals)):
        beam = add('H', r, a, b - a + 1)
        beams.append(beam)
        if level == 0:
            lo, hi, n = a, b, base
        else:
            lo = max(a, intervals[level - 1][0])
            hi = min(b, intervals[level - 1][1])
            n = gaps[level - 1]
        xs = {-1: int(rng.integers(lo, seam)), 1: int(rng.integers(seam + 1, hi + 1))}
        support_columns.append(xs)
        if kind == 'fracture':
            sides = [target_side]
        elif kind == 'command':
            sides = [-1, 1]
        else:
            sides = [-1, 1] if level == 0 or rng.random() < .72 else [int(rng.choice([-1, 1]))]
        for side in sides:
            if kind == 'command':
                chosen = (side == target_side if level == 0 else side == -target_side)
            elif kind == 'fracture':
                chosen = False
            else:
                chosen = False if allow_beam_selection else (
                    side == target_side if level == 0 else rng.random() < .28)
            add('V', r + 1, xs[side], n, chosen)
            if kind == 'general' and rng.random() < .38:
                candidates = [x for x in range(lo, seam) if abs(x - xs[side]) >= 2] if side == -1 else [x for x in range(seam + 1, hi + 1) if abs(x - xs[side]) >= 2]
                if candidates:
                    x = int(rng.choice(candidates))
                    # Multiple legs can be redundant after the cut. Their
                    # command identities need not be the same as the first leg.
                    add('V', r + 1, x, n, bool(not allow_beam_selection and rng.random() < .3))
        if kind == 'general' and level and n >= 3 and rng.random() < .4:
            used = {p[2] for p in geometry if p[0] == 'V' and p[1] == r + 1}
            candidates = [x for x in range(lo, hi + 1) if x != seam and x not in used]
            if candidates:
                length = int(rng.integers(2, n))
                add('V', rows[level - 1] - length, int(rng.choice(candidates)), length)

    if kind == 'command':
        # Demonstrate redundancy within a severed half, even with only two
        # training examples. The same key removes a harmless leg on this side
        # and the sole full-height support on the opposite side.
        a, b = intervals[0]
        side = -target_side
        candidates = [c for c in range(a, seam) if c != support_columns[0][side]] if side == -1 else [c for c in range(seam + 1, b + 1) if c != support_columns[0][side]]
        add('V', rows[0] + 1, int(rng.choice(candidates)), base, True)

    # A catch below the released base half establishes support-dependent landing.
    if kind == 'command' or (kind == 'general' and rng.random() < .72):
        a, b = intervals[0]
        used = {p[2] for p in geometry if p[0] == 'V' and p[1] == rows[0] + 1}
        candidates = [c for c in range(a, seam) if c not in used] if target_side == -1 else [c for c in range(seam + 1, b + 1) if c not in used]
        if candidates:
            x = int(rng.choice(candidates))
            lengths = [n for n in range(2, base) if not gaps or base - n != gaps[0]]
            n = int(rng.choice(lengths))
            add('V', floor - n + 1, x, n)

    if kind == 'command':
        # Same commanded identity on a harmless horizontal cap and load supports.
        top = rows[-1]
        a, b = intervals[-1]
        side = -target_side
        low, high = (a, seam - 1) if side == -1 else (seam + 1, b)
        n = int(rng.integers(2, min(4, high - low + 1) + 1))
        x = int(rng.integers(low, high - n + 2))
        post_x = int(rng.integers(x, x + n))
        add('V', top - 2, post_x, 2)
        add('H', top - 3, x, n, True)

    if kind == 'general':
        # Extra rods are placed by support, not in fixed horizontal modules.
        for _ in range(int(rng.integers(1, 9))):
            o = str(rng.choice(['H', 'V']))
            n = int(rng.integers(2, min(8, w - 2 if o == 'H' else h - 2) + 1))
            c = int(rng.integers(1, w - n)) if o == 'H' else int(rng.integers(1, w - 1))
            # Keep the controlled crossing-count shift observable.
            if o == 'H' and c <= seam <= c + n - 1:
                continue
            _drop(geometry, o, c, n, (h, w))
        # Choose commands by identity, including occasional beam removals.
        if allow_beam_selection:
            eligible = [i for i in beams if not any(_adjacent(_cells(geometry[i]), _cells(geometry[j])) for j in selected)]
            if eligible:
                # The teaching case deletes the farthest original axial hit,
                # distinguishing original-hit completion from a new endpoint
                # computed after deletion. Tests can select any visible beam.
                selected.add(eligible[0] if allow_beam_selection == 'endpoint' else int(rng.choice(eligible)))
        for i in range(len(geometry)):
            if i not in selected and i not in beams and rng.random() < .12:
                if not any(_adjacent(_cells(geometry[i]), _cells(geometry[j])) for j in selected):
                    selected.add(i)
    return (h, w), seam, geometry, selected


def _color(rng, geometry, selected, palette):
    bg, key = palette[:2]
    available = list(palette[2:])
    rng.shuffle(available)
    # Sampling a palette size does not sacrifice unambiguous rod parsing.
    preferred = available[:int(rng.integers(2, len(available) + 1))]
    footprints = [_cells(p) for p in geometry]
    assigned = {i: key for i in selected}
    order = sorted((i for i in range(len(geometry)) if i not in selected),
                   key=lambda i: -sum(_adjacent(footprints[i], s) for s in footprints))
    for i in order:
        forbidden = {assigned[j] for j in assigned if _adjacent(footprints[i], footprints[j])}
        choices = [c for c in preferred if c not in forbidden]
        if not choices:
            choices = [c for c in available if c not in forbidden]
        assert choices, "too many adjacent rod identities"
        assigned[i] = int(rng.choice(choices))
    return [(*p[:4], assigned[i]) for i, p in enumerate(geometry)]


def _make_pair(rng, levels, kind, turn, palette, beam_selection=False):
    shape, seam, geometry, selected = _scaffold(rng, levels, kind, beam_selection)
    rods = _color(rng, geometry, selected, palette)
    inp = np.full(shape, palette[0], dtype=np.int8)
    inp[0, seam] = palette[1]
    for rod in rods:
        for p in _cells(rod):
            inp[p] = rod[4]
    info = {'grid': inp, 'bg': palette[0], 'key': palette[1], 'marker': (0, seam),
            'axis': (1, seam), 'direction': (1, 0), 'rods': rods,
            'hits': sorted({p for rod in rods for p in _cells(rod) if p[1] == seam})}
    out = _transform(info)[0]
    assert set(_parse(inp)['rods']) == set(rods)
    return {'input': np.rot90(inp, turn).copy(), 'output': np.rot90(out, turn).copy()}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    fixed_palette = tuple(map(int, rng.permutation(10)))
    varied = bool(rng.random() < .7)
    base_turn = int(rng.integers(4))
    turns = [base_turn]
    if rng.random() < .48:
        turns.append((base_turn + int(rng.integers(1, 4))) % 4)
    puzzle, seen = {'train': [], 'test': []}, set()
    # Beam selection is demonstrated before tests that use it.
    teach_beam = num_train >= 3 and bool(rng.random() < .55)
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            if split == 'train' and i < 2:
                levels, kind = i + 1, ('fracture' if i == 0 else 'command')
                turn = turns[i % len(turns)]
            else:
                levels = int(rng.integers(1, 3)) if split == 'train' else int(rng.integers(3, 6))
                kind, turn = 'general', int(rng.choice(turns))
            beam_selection = teach_beam and ((split == 'train' and i == 2) or (split == 'test' and rng.random() < .4))
            if beam_selection and split == 'train':
                levels = 2
                beam_selection = 'endpoint'
            for _ in range(32):
                palette = tuple(map(int, rng.permutation(10))) if varied else fixed_palette
                pair = _make_pair(rng, levels, kind, turn, palette, beam_selection)
                k = _key(pair['input'])
                if k not in seen:
                    seen.add(k)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("bounded distinct-input construction exhausted")
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(pair):
    info = _parse(pair['input'])
    out, pins, bodies, settled, distances = _transform(info)
    ai, av = info['axis']
    dr, dc = info['direction']
    mr, mc = info['marker']
    depth = lambda p: (p[0] - mr) * dr + (p[1] - mc) * dc
    hit_depths = sorted({depth(p) for p in info['hits']})
    removed = [p for p in info['rods'] if p[4] == info['key']]
    landings, rigid_overhangs, shorter_post_landings = 0, 0, 0
    for i, ((initial, color), (final, _), distance) in enumerate(zip(bodies, settled, distances)):
        if not distance:
            continue
        target = {(r + dr, c + dc) for r, c in final}
        contacts = [j for j, (s, _) in enumerate(settled) if i != j and target & s]
        landings += bool(contacts)
        rigid_overhangs += bool(contacts) and len(final) >= 2 and all(not target <= settled[j][0] for j in contacts)
        perpendicular = len({p[ai] for p in initial}) > 1
        shorter_post_landings += perpendicular and any(
            distances[j] == 0 and len({p[ai] for p in bodies[j][0]}) == 1
            and min(depth(p) for p in bodies[j][0]) > max(depth(p) for p in initial) + 1
            for j in contacts)
    offaxis = np.indices(out.shape)[ai] != av
    intact_parts = _parts(info, delete=False)[1]
    remaining = {s: d for (s, color), d in zip(bodies, distances)}
    redundant_fragments = harmless_support_removal = 0
    for i, (s, color) in enumerate(intact_parts):
        target = {(r + dr, c + dc) for r, c in s}
        support = [j for j, (t, _) in enumerate(intact_parts) if i != j and target & t]
        if len(support) >= 2:
            redundant_fragments += 1
            harmless_support_removal += (s in remaining and remaining[s] == 0
                                        and any(intact_parts[j][1] == info['key'] for j in support))
    return {'side': info['side'], 'shape': list(info['grid'].shape),
            'background': info['bg'], 'key': info['key'], 'rods': len(info['rods']),
            'crossings': len(info['crossings']), 'bodies': len(bodies),
            'rod_lengths': [p[3] for p in info['rods']],
            'removed_rods': len(removed), 'removed_cells': sum(p[3] for p in removed),
            'removed_orientations': sorted({p[0] for p in removed}),
            'selected_crossing': any(p[4] == info['key'] for p in info['crossings']),
            'deleted_endpoint': bool(info['grid'][max(info['hits'], key=depth)] == info['key']),
            'pins': len(pins), 'moved': sum(d > 0 for d in distances),
            'fall_distances': sorted(set(distances) - {0}), 'landings': landings,
            'rigid_overhangs': rigid_overhangs, 'shorter_post_landings': shorter_post_landings,
            'redundant_fragments': redundant_fragments, 'harmless_support_removal': harmless_support_removal,
            'axis_gap': any(b > a + 1 for a, b in zip(hit_depths, hit_depths[1:])),
            'short_endpoint': max(hit_depths) < max(depth(p) for rod in info['rods'] for p in _cells(rod)),
            'no_fracture_differs': not np.array_equal(out[offaxis], _transform(info, fracture=False)[0][offaxis]),
            'no_deletion_differs': not np.array_equal(out, _transform(info, delete=False)[0]),
            'nearest_differs': not np.array_equal(out, _transform(info, nearest=True)[0]),
            'full_axis_differs': not np.array_equal(out, _transform(info, full=True)[0])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            original = [(_cells(p), p[4]) for p in info['rods']]
            assert _settle(original, inp.shape, info['direction'])[1] == [0] * len(original)
            predicted, pins, bodies, settled, distances = _transform(info)
            assert np.array_equal(out, predicted)
            assert not np.array_equal(inp, out)
            # Explicit mechanics invariants supplement whole-grid agreement.
            assert _settle(settled, inp.shape, info['direction'], pins)[1] == [0] * len(settled)
            occupied = set(pins)
            dr, dc = info['direction']
            for (before, color), (after, after_color), d in zip(bodies, settled, distances):
                assert d >= 0 and color == after_color and color != info['key']
                assert after == {(r + d * dr, c + d * dc) for r, c in before}
                assert not occupied & after
                occupied.update(after)
            for p, color in pins.items():
                assert inp[p] == out[p] == color and color != info['key']
            original_kept = sum(p[3] for p in info['rods'] if p[4] != info['key'])
            assert len(occupied) == original_kept
            ray = set(_ray(info))
            assert all(out[p] == info['key'] for p in ray - set(pins))
            assert {tuple(map(int, p)) for p in np.argwhere(out == info['key'])} == ray - set(pins)
            rim = np.zeros(inp.shape, dtype=bool)
            rim[[0, -1]] = True
            rim[:, [0, -1]] = True
            assert np.array_equal(inp[rim], out[rim])
            features[split].append(pair_features(pair))
    tr, te = features['train'], features['test']
    assert any(f['no_fracture_differs'] and not f['removed_rods'] for f in tr)
    assert any(f['no_deletion_differs'] and len(f['removed_orientations']) == 2
               and f['removed_rods'] >= 3 for f in tr)
    assert any(f['axis_gap'] and f['nearest_differs'] for f in tr)
    assert any(f['short_endpoint'] and f['full_axis_differs'] for f in tr)
    assert any(len(f['fall_distances']) >= 2 and f['shorter_post_landings']
               and f['rigid_overhangs'] for f in tr)
    assert any(f['harmless_support_removal'] for f in tr)
    assert max(f['crossings'] for f in tr) <= 2
    assert min(f['crossings'] for f in te) >= 3
    assert {f['side'] for f in te} <= {f['side'] for f in tr}
    if any(f['selected_crossing'] for f in te):
        assert any(f['selected_crossing'] for f in tr)
    if any(f['deleted_endpoint'] for f in te):
        assert any(f['deleted_endpoint'] for f in tr)


def _fingerprint(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for split in ('train', 'test') for p in puzzle[split])


def _mechanics_test():
    # The solver supports the smallest specified tray, independently of the
    # generator's headroom and scaffold-depth sampling choices.
    small = np.zeros((7, 7), dtype=np.int8)
    small[0, 3] = 8
    small[3, 1:6] = 2
    small[4:6, 2] = 3
    expected_small = np.zeros_like(small)
    expected_small[:4, 3] = 8
    expected_small[3, 1:4] = 2
    expected_small[4:6, 2] = 3
    expected_small[5, 4:6] = 2
    assert np.array_equal(solve(small), expected_small)
    # A one-sided bridge: the seam breaks mechanical support transfer.
    inp = np.zeros((9, 11), dtype=np.int8)
    inp[0, 5] = 8
    inp[3, 2:9] = 2
    inp[4:8, 3] = 3
    expected = np.zeros_like(inp)
    expected[:4, 5] = 8
    expected[3, 2:6] = 2
    expected[4:8, 3] = 3
    expected[7, 6:9] = 2
    assert np.array_equal(solve(inp), expected)
    # A deleted crossing still determines the guide endpoint. Another survivor
    # crossing keeps the family input valid; its axial cell remains fixed.
    inp = np.zeros((12, 11), dtype=np.int8)
    inp[0, 5] = 8
    inp[4, 2:9] = 2
    inp[5:8, 3] = 3
    inp[7, 2:9] = 8
    inp[8:11, 2] = 4
    out = solve(inp)
    assert out[7, 5] == 8 and out[8, 5] == 0 and out[4, 5] == 2
    assert all(out[r, 5] == 8 for r in (0, 1, 2, 3, 5, 6, 7))
    # Simultaneous settling must move a stack together and preserve overhangs.
    bodies = [(frozenset({(3, 2), (3, 3), (3, 4)}), 2),
              (frozenset({(4, 3), (5, 3)}), 3)]
    settled, distances = _settle(bodies, (10, 8), (1, 0))
    assert distances == [3, 3]
    assert settled[0][0] == frozenset({(6, 2), (6, 3), (6, 4)})
    # Whole-scene rotation, including inward gravity and the pinned cells.
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(inp, turn)), np.rot90(out, turn))
    palette = np.array([4, 0, 7, 8, 6, 2, 3, 9, 1, 5], dtype=np.int8)
    assert np.array_equal(solve(palette[inp]), palette[out])


def self_test():
    """Bounded 42-seed structural and family-specific audit; no files written."""
    _mechanics_test()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(generate_puzzle(671628239)) == _fingerprint(generate_puzzle(671628239))
    records = [pair_features(pair) for p in puzzles for s in ('train', 'test') for pair in p[s]]
    assert {f['side'] for f in records} == {'top', 'bottom', 'left', 'right'}
    assert {f['background'] for f in records} == set(range(10))
    assert len({tuple(f['shape']) for f in records}) >= 60
    assert len({f['rods'] for f in records}) >= 10
    assert {f['crossings'] for f in records} == {1, 2, 3, 4, 5}
    assert len({d for f in records for d in f['fall_distances']}) >= 8
    assert max(n for f in records for n in f['rod_lengths']) >= 20
    assert sum(f['selected_crossing'] for f in records) >= 3
    assert sum(f['deleted_endpoint'] for f in records) >= 3
    assert sum(len(f['fall_distances']) >= 3 for f in records) >= 10
    assert sum(f['no_fracture_differs'] for f in records) >= 100
    assert sum(f['no_deletion_differs'] for f in records) >= 120
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            p = generate_puzzle(34436953, ntrain, ntest)
            assert len(p['train']) == ntrain and len(p['test']) == ntest
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, all legal explicit counts, mechanics and diversity')
