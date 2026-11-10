"""Scoped relay locks: geometry decodes relations before simultaneous lookup.

No parent module, source puzzle, seed, or generation state is used by solve().
"""
from __future__ import annotations

import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(cells):
    remaining = set(cells)
    parts = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, part = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in D4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    part.add(q)
        parts.append(frozenset(part))
    return parts


def _box(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr), min(cc), max(cc)


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _transform(regions, operation):
    """Four rotations, optionally preceded by a reflection; normalize jointly."""
    transformed = []
    for region in regions:
        cells = []
        for r, c in region:
            if operation >= 4:
                c = -c
            for _ in range(operation % 4):
                r, c = c, -r
            cells.append((r, c))
        transformed.append(cells)
    r0 = min(r for cells in transformed for r, c in cells)
    c0 = min(c for cells in transformed for r, c in cells)
    return tuple(frozenset((r - r0, c - c0) for r, c in cells)
                 for cells in transformed)


def _shape(cells):
    r0, _, c0, _ = _box(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _canonical(cells):
    return min(_shape(_transform((cells,), op)[0]) for op in range(8))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('expected an integer ARC grid with dimensions 1..30')
    return a.astype(np.int8, copy=True)


def _decode(grid):
    """Find the unique full, colored *virtual* assembly, using only the input."""
    values, counts = np.unique(grid, return_counts=True)
    if sum(counts == counts.max()) != 1:
        raise ValueError('background must be uniquely most frequent')
    bg = int(values[np.argmax(counts)])
    fragments, keys = [], []
    for cells in _components(map(tuple, np.argwhere(grid != bg))):
        colors = tuple(sorted({int(grid[p]) for p in cells}))
        if len(colors) == 1:
            fragments.append((cells, colors[0]))
        elif len(colors) == 2:
            regions = tuple(frozenset(p for p in cells if grid[p] == color)
                            for color in colors)
            if any(len(_components(region)) != 1 for region in regions):
                raise ValueError('a key color region is disconnected')
            keys.append(dict(cells=cells, regions=regions, colors=colors))
        else:
            raise ValueError('each loose object must have one or two colors')
    if not keys or len(fragments) != 2 * len(keys):
        raise ValueError('each key requires two fixed lock fragments')
    frame_colors = {color for _, color in fragments}
    if any(frame_colors.intersection(key['colors']) for key in keys):
        raise ValueError('frame labels and relation identities must be distinct')

    locks = []
    for i, (one, color) in enumerate(fragments):
        for j in range(i + 1, len(fragments)):
            two, other_color = fragments[j]
            if color != other_color:
                continue
            union = one | two
            r0, r1, c0, c1 = _box(union)
            rect = frozenset((r, c) for r in range(r0, r1 + 1)
                             for c in range(c0, c1 + 1))
            if frozenset(p for p in rect if grid[p] != bg) != union:
                continue
            slot = rect - union
            if slot and len(_components(slot)) == 1:
                locks.append(dict(parts=frozenset((i, j)), frame=color,
                                  rect=rect, slot=slot, box=(r0, r1, c0, c1)))

    options = []
    for ki, key in enumerate(keys):
        candidates = {}
        for lock in locks:
            sr, _, sc, _ = _box(lock['slot'])
            for op in range(8):
                regions = _transform(key['regions'], op)
                colors = key['colors'] if op < 4 else key['colors'][::-1]
                for plug_index in (0, 1):
                    pr, _, pc, _ = _box(regions[plug_index])
                    placed = tuple(_shift(region, sr - pr, sc - pc) for region in regions)
                    if placed[plug_index] != lock['slot']:
                        continue
                    tail = placed[1 - plug_index]
                    if tail & lock['rect']:
                        continue
                    if any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1])
                           or grid[r, c] != bg for r, c in tail):
                        continue
                    source, value = colors[plug_index], colors[1 - plug_index]
                    signature = (lock['parts'], placed[plug_index], tail, source, value)
                    candidates[signature] = dict(
                        key=ki, parts=lock['parts'], frame=lock['frame'],
                        rect=lock['rect'], slot=lock['slot'], tail=tail,
                        source=source, value=value, operation=op,
                        full=placed[0] | placed[1])
        if not candidates:
            raise ValueError('a loose key has no exact whole-key fit')
        options.append(list(candidates.values()))

    order = sorted(range(len(keys)), key=lambda i: len(options[i]))
    solutions = []
    signatures = set()

    def search(depth, used_parts, used_rects, occupied, chosen):
        if len(solutions) > 1:
            return
        if depth == len(order):
            if len(used_parts) == len(fragments):
                signature = tuple(sorted((m['frame'], tuple(sorted(m['slot'])),
                                          tuple(sorted(m['tail'])), m['source'], m['value'])
                                         for m in chosen))
                if signature not in signatures:
                    signatures.add(signature)
                    solutions.append(chosen)
            return
        for move in options[order[depth]]:
            if move['parts'] & used_parts or move['full'] & occupied:
                continue
            if any(move['rect'] & rect for rect in used_rects):
                continue
            search(depth + 1, used_parts | move['parts'],
                   used_rects + [move['rect']], occupied | move['full'], chosen + [move])

    search(0, frozenset(), [], frozenset(), [])
    if len(solutions) != 1:
        raise ValueError('virtual colored assembly must be unique')
    moves = sorted(solutions[0], key=lambda m: m['key'])
    mapping = {}
    for move in moves:
        address = move['frame'], move['source']
        if address in mapping and mapping[address] != move['value']:
            raise ValueError('conflicting source values within one lock-color group')
        mapping[address] = move['value']
    for i, move in enumerate(moves):
        move['donors'] = [j for j, other in enumerate(moves)
                          if j != i and other['frame'] == move['frame']
                          and other['source'] == move['value']]
        move['result'] = moves[move['donors'][0]]['value'] if move['donors'] else None
    return bg, fragments, keys, moves


def _paint(grid, decoded, policy='intended'):
    """Policies other than intended are explicit shortcut checks, never solvers."""
    bg, fragments, keys, moves = decoded
    out = np.full(grid.shape, bg, dtype=np.int8)
    for cells, color in fragments:
        for p in cells:
            out[p] = color
    for i, move in enumerate(moves):
        donors = move['donors']
        if policy == 'unscoped':
            donors = [j for j, other in enumerate(moves)
                      if j != i and other['source'] == move['value']]
        elif policy == 'reverse':
            donors = [j for j, other in enumerate(moves)
                      if j != i and other['frame'] == move['frame']
                      and other['value'] == move['source']]
        if policy == 'fit_only':
            value = move['value']
        elif not donors:
            continue
        elif policy == 'reverse':
            value = moves[donors[0]]['source']
        else:
            value = moves[donors[0]]['value']
            if policy == 'recursive' and moves[donors[0]]['result'] is not None:
                value = moves[donors[0]]['result']
        for p in move['slot']:
            out[p] = move['source']
        for p in move['tail']:
            out[p] = value
    return out


def _ignore_reflection(decoded):
    bg, fragments, keys, moves = decoded
    altered = [dict(m) for m in moves]
    for m in altered:
        if m['operation'] >= 4:
            m['source'], m['value'] = m['value'], m['source']
    for i, m in enumerate(altered):
        m['donors'] = [j for j, other in enumerate(altered)
                       if j != i and other['frame'] == m['frame']
                       and other['source'] == m['value']]
        m['result'] = altered[m['donors'][0]]['value'] if m['donors'] else None
    return bg, fragments, keys, altered


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode geometry and reflection parity, then compose once in each scope."""
    grid = _grid(input_grid)
    return _paint(grid, _decode(grid))


def _locally_unique(shape):
    frame, slot, tail = shape
    rect = frame | slot
    sr, _, sc, _ = _box(slot)
    results = set()
    for op in range(8):
        transformed = _transform((slot, tail), op)
        for pi in (0, 1):
            pr, _, pc, _ = _box(transformed[pi])
            placed = tuple(_shift(cells, sr - pr, sc - pc) for cells in transformed)
            if placed[pi] != slot or placed[1 - pi] & rect:
                continue
            # Include transformed region labels: opposite parity is a different
            # decoded relation even when its binary footprint happens to agree.
            results.add(tuple(sorted((p, i ^ (op >= 4)) for i, cells in enumerate(placed)
                                     for p in cells)))
    return len(results) == 1


def _new_shape(rng, forbidden, small):
    for _ in range(300):
        h = int(rng.integers(3, 6 if small else 7))
        w = int(rng.integers(3, 6 if small else 7))
        y = int(rng.integers(1, h - 1))
        slot = {(y, 0)}
        for c in range(1, w):
            new_y = int(np.clip(y + rng.integers(-1, 2), 1, h - 2))
            slot.update((r, c) for r in range(min(y, new_y), max(y, new_y) + 1))
            y = new_y
        # Optional thickening changes contours without changing the rule.
        for _ in range(int(rng.integers(3))):
            r, c = sorted(slot)[int(rng.integers(len(slot)))]
            dr, dc = D4[int(rng.integers(4))]
            if 0 <= r + dr < h and 0 <= c + dc < w:
                slot.add((r + dr, c + dc))
        rect = {(r, c) for r in range(h) for c in range(w)}
        frame = rect - slot
        parts = _components(frame)
        if (len(parts) != 2 or min(map(len, parts)) < 2
                or _box(frame) != (0, h - 1, 0, w - 1)
                or _canonical(slot) in forbidden):
            continue
        frontier = sorted({(r + dr, c + dc) for r, c in slot for dr, dc in D4} - rect)
        start = frontier[int(rng.integers(len(frontier)))]
        tail = {start}
        relation = int(rng.integers(3))
        size = (int(rng.integers(2, max(3, len(slot)))) if relation == 0 else
                len(slot) if relation == 1 else len(slot) + int(rng.integers(1, 5)))
        size = min(size, 9 if small else 14)
        style = int(rng.integers(3))
        for _ in range(size - 1):
            frontier = sorted({(r + dr, c + dc) for r, c in tail for dr, dc in D4
                               if -3 <= r + dr < h + 3 and -3 <= c + dc < w + 3}
                              - rect - tail)
            if not frontier:
                break
            weights = np.array([(1 + sum((r + dr, c + dc) in tail for dr, dc in D4))
                                ** (2 if style == 0 else -1)
                                / (1 + abs(r - start[0]) + abs(c - start[1]))
                                ** (1 if style != 2 else 0) for r, c in frontier])
            tail.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
        if len(tail) != size:
            continue
        regions = _transform((frame, slot, tail), int(rng.integers(8)))
        all_cells = frozenset.union(*regions)
        _, r1, _, c1 = _box(all_cells)
        if max(r1, c1) >= (9 if small else 11):
            continue
        if _locally_unique(regions):
            return regions
    raise RuntimeError('shape sampling exhausted')


def _pack(rng, footprints):
    area = sum(len(cells) for cells in footprints)
    for attempt in range(20):
        if attempt >= 8:
            h = w = 30
        else:
            h, w = map(int, rng.integers(16, 31, size=2))
            if h * w < area * 2.1:
                continue
        occupied, origins = set(), [None] * len(footprints)
        order = sorted(range(len(footprints)),
                       key=lambda i: -(_box(footprints[i])[1] + 2)
                       * (_box(footprints[i])[3] + 2) + float(rng.random()))
        for i in order:
            cells = footprints[i]
            _, r1, _, c1 = _box(cells)
            choices = [(r, c) for r in range(h - r1) for c in range(w - c1)]
            rng.shuffle(choices)
            if rng.random() < .35:
                choices.sort(key=lambda p: not (p[0] in (0, h - r1 - 1)
                                               or p[1] in (0, w - c1 - 1)))
            for r, c in choices:
                placed = _shift(cells, r, c)
                if not placed.intersection(occupied):
                    origins[i] = r, c
                    occupied.update((y + dy, x + dx) for y, x in placed
                                    for dy, dx in ((0, 0),) + D4)
                    break
            else:
                break
        else:
            return (h, w), origins
    return None


def _relations(rng, mode, test=False):
    # Identity names are abstract indices, recolored independently below.
    ids = list(map(int, rng.permutation(5)))
    a, b, c, d, _ = ids
    if mode == 'chain':
        return [(0, a, b), (0, b, c), (0, c, d), (1, b, a), (1, c, b)]
    if mode == 'cycle':
        return [(0, a, b), (0, b, a), (1, a, c), (1, a, c), (1, c, d)]
    for _ in range(100):
        n = int(rng.integers(6, 8) if test else rng.integers(3, 6))
        groups = 2 if n >= 4 and (test or rng.random() < .75) else 1
        split = int(rng.integers(2, n - 1)) if groups == 2 else n
        sizes = [split, n - split] if groups == 2 else [n]
        relations = []
        for group, k in enumerate(sizes):
            bodies = list(map(int, rng.integers(5, size=k)))
            x, y, z = map(int, rng.choice(5, 3, replace=False))
            bodies[0], bodies[1] = x, y
            mapping = {body: int(rng.choice([v for v in range(5) if v != body]))
                       for body in set(bodies)}
            mapping[x], mapping[y] = y, z
            relations.extend((group, body, mapping[body]) for body in bodies)
        results = [[j for j, (g, x, y) in enumerate(relations)
                    if j != i and g == group and x == value]
                   for i, (group, body, value) in enumerate(relations)]
        if any(results) and not all(results):
            return relations
    raise RuntimeError('relation sampling exhausted')


def _make_pair(rng, palette, mode, test=False):
    relations = _relations(rng, mode, test)
    n = len(relations)
    bg, frames, identities = palette[0], palette[1:3], palette[3:8]
    for attempt in range(100):
        shapes, forbidden = [], set()
        for _ in range(n):
            shape = _new_shape(rng, forbidden, small=n >= 5)
            shapes.append(shape)
            forbidden.add(_canonical(shape[1]))
        operations = list(map(int, rng.integers(8, size=n)))
        # Both training witnesses include both parities among successful keys.
        if mode in ('chain', 'cycle'):
            operations[0] = int(rng.integers(4))
            operations[1] = 4 + int(rng.integers(4))
        keys = [_transform(shape[1:], op) for shape, op in zip(shapes, operations)]
        footprints = [frozenset.union(*shape) for shape in shapes]
        footprints += [key[0] | key[1] for key in keys]
        packing = _pack(rng, footprints)
        if packing is None:
            continue
        dimensions, origins = packing
        inp = np.full(dimensions, bg, dtype=np.int8)
        out = inp.copy()
        for i, ((group, source, value), shape, key, op) in enumerate(
                zip(relations, shapes, keys, operations)):
            ar, ac = origins[i]
            kr, kc = origins[n + i]
            for p in _shift(shape[0], ar, ac):
                inp[p] = out[p] = frames[group]
            key_colors = (identities[source], identities[value])
            assert source != value  # distinct regions remain visibly separable
            if op >= 4:
                key_colors = key_colors[::-1]
            for region, color in zip(key, key_colors):
                for p in _shift(region, kr, kc):
                    inp[p] = color
            donors = [j for j, (g, x, y) in enumerate(relations)
                      if j != i and g == group and x == value]
            if donors:
                result = relations[donors[0]][2]
                for p in _shift(shape[1], ar, ac):
                    out[p] = identities[source]
                for p in _shift(shape[2], ar, ac):
                    out[p] = identities[result]
        try:
            decoded = _decode(inp)
            if np.array_equal(_paint(inp, decoded), out):
                return dict(input=inp, output=out)
        except ValueError:
            pass
    raise RuntimeError('unique scene construction exhausted')


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _is_int(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (not _is_int(num_train) or not _is_int(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError('counts require 2..10 train, 1..4 test, and train > test')
    rng = np.random.default_rng(seed)
    stable = bool(rng.random() < .25)
    shared_palette = tuple(map(int, rng.permutation(10)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            mode = ('chain', 'cycle')[i] if split == 'train' and i < 2 else 'random'
            for _ in range(20):
                palette = shared_palette if stable else tuple(map(int, rng.permutation(10)))
                pair = _make_pair(rng, palette, mode, split == 'test')
                signature = _key(pair['input'])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input sampling exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, train_decoded = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            decoded = _decode(inp)
            assert np.array_equal(_paint(inp, decoded), out), 'incorrect scoped composition'
            bg, fragments, keys, moves = decoded
            assert (3 <= len(keys) <= 5 if split == 'train' else 6 <= len(keys) <= 7)
            if split == 'test':
                assert len({m['frame'] for m in moves}) == 2
            assert any(m['donors'] for m in moves) and any(not m['donors'] for m in moves)
            for key in keys:
                assert all(out[p] == bg for p in key['cells']), 'loose key was not erased'
            for cells, color in fragments:
                assert all(out[p] == color for p in cells), 'fixed fragment changed'
            for m in moves:
                assert len(_components(m['slot'])) == len(_components(m['tail'])) == 1
                assert len(_components(m['rect'] - m['slot'])) == 2
                assert not (m['tail'] & m['rect'])
                expected = m['source'] if m['donors'] else bg
                assert all(out[p] == expected for p in m['slot'])
                assert all(out[p] == (m['result'] if m['donors'] else bg) for p in m['tail'])
            if split == 'train':
                train_decoded.append((inp, decoded))
    for policy in ('fit_only', 'unscoped', 'recursive', 'reverse'):
        assert any(not np.array_equal(_paint(g, d), _paint(g, d, policy))
                   for g, d in train_decoded), 'training permits shortcut: ' + policy
    assert any(not np.array_equal(_paint(g, d), _paint(g, _ignore_reflection(d)))
               for g, d in train_decoded), 'reflection must change the relational answer'
    train_parities = {m['operation'] >= 4 for _, d in train_decoded for m in d[3] if m['donors']}
    assert train_parities == {False, True}, 'successful fits must teach both parities'
    assert any(a['value'] == b['value'] and a['frame'] != b['frame']
               and a['result'] is not None and b['result'] is not None
               and a['result'] != b['result']
               for _, d in train_decoded for a in d[3] for b in d[3]), 'scope conflict is not demonstrated'
    assert any(m['donors'] and any(not d[3][j]['donors'] for j in m['donors'])
               for _, d in train_decoded for m in d[3]), 'no disappearing-donor witness'


def self_test():
    shapes, slots, tails, operations, relations, sizes, counts, palettes = set(), set(), set(), set(), set(), set(), set(), set()
    metrics = {name: 0 for name in ('pairs', 'symmetric_slots', 'edge_locks', 'shared_donors',
                                   'disappearing_donors', 'updated_donors', 'solid_results',
                                   'repeated_sources', 'cross_group_conflicts', 'equal_area_slots')}
    group_counts, outcomes = set(), set()
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                decoded = _decode(inp)
                assert np.array_equal(solve(inp), pair['output'])
                shapes.add(inp.shape)
                palettes.add(tuple(map(int, np.unique(inp))))
                moves = decoded[3]
                counts.add(len(moves))
                group_counts.add(len({m['frame'] for m in moves}))
                outcomes.add(sum(bool(m['donors']) for m in moves))
                metrics['pairs'] += 1
                metrics['shared_donors'] += any(sum(j in m['donors'] for m in moves) > 1
                                               for j in range(len(moves)))
                metrics['repeated_sources'] += len({(m['frame'], m['source']) for m in moves}) < len(moves)
                metrics['cross_group_conflicts'] += any(a['frame'] != b['frame']
                    and a['source'] == b['source'] and a['value'] != b['value'] for a in moves for b in moves)
                metrics['equal_area_slots'] += any(len(a['slot']) == len(b['slot'])
                    and _canonical(a['slot']) != _canonical(b['slot']) for i, a in enumerate(moves) for b in moves[i+1:])
                for m in moves:
                    metrics['symmetric_slots'] += len({_shape(_transform((m['slot'],), op)[0]) for op in range(8)}) < 8
                    r0, r1, c0, c1 = _box(m['rect'])
                    metrics['edge_locks'] += r0 == 0 or c0 == 0 or r1 == inp.shape[0]-1 or c1 == inp.shape[1]-1
                    metrics['disappearing_donors'] += any(not moves[j]['donors'] for j in m['donors'])
                    metrics['updated_donors'] += any(moves[j]['result'] is not None
                        and moves[j]['result'] != moves[j]['value'] for j in m['donors'])
                    metrics['solid_results'] += m['result'] == m['source']
                for m in moves:
                    slots.add(_canonical(m['slot']))
                    tails.add(_canonical(m['tail']))
                    sizes.add(int(np.sign(len(m['slot']) - len(m['tail']))))
                    operations.add(m['operation'])
                    relations.add((m['frame'], m['source'], m['value']))
    assert len(set(puzzles)) == 42
    assert len(shapes) >= 20 and len(slots) >= 40 and len(tails) >= 40
    assert operations == set(range(8)) and sizes == {-1, 0, 1}
    assert group_counts == {1, 2} and len(outcomes) >= 3
    assert all(v > 0 for v in metrics.values()), metrics
    assert {3, 4, 5, 6, 7} <= counts and len(palettes) >= 10 and len(relations) >= 40
    assert puzzles[0] == tuple(_key(p['input']) for s in ('train', 'test')
                               for p in generate_puzzle(0)[s])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(100 + nt * 5 + ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # The unique-arrangement invariant rejects an actually ambiguous key.
    ambiguous = np.zeros((12, 14), dtype=np.int8)
    ambiguous[4, 2:5] = ambiguous[6, 2:5] = 2
    ambiguous[2, 9:12] = 3
    ambiguous[2:4, 12] = 4
    try:
        solve(ambiguous)
    except ValueError as error:
        assert 'virtual colored assembly must be unique' in str(error)
    else:
        raise AssertionError('ambiguous reflected outputs were accepted')
    damaged = generate_puzzle(3, 2, 1)
    damaged['train'][0]['output'][0, 0] = (int(damaged['train'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged output was accepted')
    return dict(seeds=42, distinct_puzzles=len(set(puzzles)), grid_shapes=len(shapes),
                slot_shapes=len(slots), tail_shapes=len(tails), key_counts=sorted(counts),
                operations=sorted(operations), plug_minus_tail_signs=sorted(sizes),
                group_counts=sorted(group_counts), survivor_counts=sorted(outcomes), features=metrics)
