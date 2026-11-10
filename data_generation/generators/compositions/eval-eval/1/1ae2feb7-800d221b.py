"""Clocked consensus along an oriented trail.

Only the input defines the roles, path order, panel votes, clock periods and
phases. Construction uses its own records; solve reconstructs all of them.
"""
from itertools import permutations

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors(p):
    return [(p[0] + dr, p[1] + dc) for dr, dc in DIRS]


def _halo(cells):
    return set(cells) | {q for p in cells for q in _neighbors(p)}


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        todo, part = [first], {first}
        while todo:
            for q in _neighbors(todo.pop()):
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    todo.append(q)
        result.append(part)
    return result


def _cells(a, color):
    return set(zip(*np.nonzero(a == color)))


def _grid(a):
    a = np.asarray(a)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _order_path(track, start):
    all_cells = track | {start}
    if len(all_cells) < 15:
        raise ValueError("Trail too short")
    degree = {p: sum(q in all_cells for q in _neighbors(p)) for p in all_cells}
    if degree[start] != 1 or sorted(degree.values()) != [1, 1] + [2] * (len(all_cells) - 2):
        raise ValueError("Trail must be a single unbranched path with a marked endpoint")
    path, previous, current = [], None, start
    while True:
        path.append(current)
        following = [q for q in _neighbors(current) if q in all_cells and q != previous]
        if not following:
            break
        previous, current = current, following[0]
        if len(path) > len(all_cells):
            raise ValueError("Trail contains a cycle")
    if len(path) != len(all_cells):
        raise ValueError("Disconnected trail")
    return path


def _interpret(a, background, marker, track_color, clock_color, signals):
    start = next(iter(_cells(a, marker)))
    track = _cells(a, track_color)
    path = _order_path(track, start)
    index = {p: k for k, p in enumerate(path)}
    panels = _components(_cells(a, signals[0]) | _cells(a, signals[1]))
    if not 2 <= len(panels) <= 6:
        raise ValueError("Need 2..6 separate panels")
    owner, labels, boxes = {}, [], []
    for i, cells in enumerate(panels):
        rows, cols = zip(*cells)
        box = min(rows), max(rows), min(cols), max(cols)
        if len(cells) != (box[1] - box[0] + 1) * (box[3] - box[2] + 1):
            raise ValueError("Panel must be a filled rectangle")
        n0 = sum(a[p] == signals[0] for p in cells)
        if n0 == 0 or n0 == len(cells) or 2 * n0 == len(cells):
            raise ValueError("Every panel needs two colors and a strict majority")
        if _halo(cells) & set(path):
            raise ValueError("Panels may contact the trail only through clocks")
        labels.append(signals[int(2 * n0 < len(cells))])
        boxes.append(box)
        owner.update({p: i for p in cells})
    if set(labels) != set(signals):
        raise ValueError("Both local majority labels must occur")
    rods = []
    for cells in _components(_cells(a, clock_color)):
        rows, cols = zip(*cells)
        if len(set(rows)) != 1 and len(set(cols)) != 1:
            raise ValueError("Clocks must be straight, separate bars")
        if not 1 <= len(cells) <= 8:
            raise ValueError("Clock length must be 1..8")
        panel_contacts = {(p, q) for p in cells for q in _neighbors(p) if q in owner}
        trail_contacts = {(p, q) for p in cells for q in _neighbors(p) if q in index}
        if len(panel_contacts) != 1 or len(trail_contacts) != 1:
            raise ValueError("Each clock must have one panel endpoint and one trail endpoint")
        end, panel_cell = next(iter(panel_contacts))
        begin, tap = next(iter(trail_contacts))
        if tap == start:
            raise ValueError("Start marker cannot be a tap")
        # Panel, bar and tap must be collinear, including length-one bars.
        chain = cells | {panel_cell, tap}
        if not (len({p[0] for p in chain}) == 1 or len({p[1] for p in chain}) == 1):
            raise ValueError("Clock endpoints must extend its axis")
        if abs(panel_cell[0] - tap[0]) + abs(panel_cell[1] - tap[1]) != len(cells) + 1:
            raise ValueError("Clock endpoints must be at opposite ends")
        i = owner[panel_cell]
        rods.append(dict(cells=cells, panel=i, label=labels[i], length=len(cells),
                         tap=index[tap], contact=int(a[panel_cell])))
    if not 3 <= len(rods) <= 12 or {r['panel'] for r in rods} != set(range(len(panels))):
        raise ValueError("Need 3..12 clocks, with every panel connected")
    if len({r['tap'] for r in rods}) != len(rods):
        raise ValueError("Only one clock may attach to a trail cell")
    return dict(background=background, marker=marker, track_color=track_color,
                clock_color=clock_color, signals=signals, path=path,
                panels=panels, labels=labels, boxes=boxes,
                rods=sorted(rods, key=lambda r: r['tap']))


def _analyze(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 6 or sum(counts == 1) != 1 or sum(counts == counts.max()) != 1:
        raise ValueError("Need six roles, a unique singleton start and dominant background")
    background = int(colors[counts.argmax()])
    marker = int(colors[np.flatnonzero(counts == 1)[0]])
    other = sorted(set(map(int, colors)) - {background, marker})
    interpretations = []
    # The marked endpoint usually identifies the trail color immediately.
    start = next(iter(_cells(a, marker)))
    adjacent_colors = {int(a[q]) for q in _neighbors(start)
                       if 0 <= q[0] < a.shape[0] and 0 <= q[1] < a.shape[1]}
    for track_color, clock_color in permutations(other, 2):
        if track_color not in adjacent_colors:
            continue
        signals = tuple(c for c in other if c not in (track_color, clock_color))
        try:
            interpretations.append(_interpret(a, background, marker, track_color, clock_color, signals))
        except ValueError:
            continue
    if len(interpretations) != 1:
        raise ValueError("Input must have exactly one geometric role interpretation")
    return a, interpretations[0]


def _ballots(info, k, alternative=None):
    active = []
    for rod in info['rods']:
        distance = k - rod['tap']
        if alternative == 'bidirectional':
            distance = abs(distance)
        elif distance < 0:
            continue
        period = rod['length']
        if alternative == 'no_period':
            fires = True
        elif alternative == 'global_phase':
            fires = k % period == 0
        elif alternative == 'manhattan':
            x, y = info['path'][k], info['path'][rod['tap']]
            fires = (abs(x[0] - y[0]) + abs(x[1] - y[1])) % period == 0
        elif alternative == 'phase_late':
            fires = (distance - 1) % period == 0
        else:
            fires = distance % period == 0
        if fires:
            active.append(rod)
    return active


def _value(info, active, alternative=None):
    if alternative == 'nearest':
        return active[-1]['label'] if active else info['track_color']
    if alternative == 'one_per_panel':
        active = list({r['panel']: r for r in active}.values())
    totals = dict.fromkeys(info['signals'], 0)
    for r in active:
        if alternative == 'contact':
            label = r['contact']
        elif alternative == 'minority':
            label = next(s for s in info['signals'] if s != r['label'])
        else:
            label = r['label']
        weight = len(info['panels'][r['panel']]) if alternative == 'area' else 1
        totals[label] += weight
    s0, s1 = info['signals']
    if totals[s0] == totals[s1]:
        return info['track_color']
    return s0 if totals[s0] > totals[s1] else s1


def _prediction(a, info, alternative=None):
    out = a.copy()
    for k, p in enumerate(info['path'][1:], 1):
        out[p] = _value(info, _ballots(info, k, alternative), alternative)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer roles and ordered geometry from input, then tally periodic ports."""
    a, info = _analyze(input_grid)
    return _prediction(a, info)


RIVALS = ('no_period', 'one_per_panel', 'contact', 'area', 'nearest',
          'global_phase', 'manhattan', 'bidirectional', 'phase_late')


def _evidence(a, info, out):
    first, second = info['rods'][:2]
    isolated = (first['length'] >= 2 and second['tap'] - first['tap'] > first['length'] + 1)
    ties, silence, winners, mixed = False, False, set(), False
    for k in range(1, len(info['path'])):
        active = _ballots(info, k)
        labels = [r['label'] for r in active]
        n0, n1 = (labels.count(s) for s in info['signals'])
        ties |= n0 == n1 and n0 > 0
        silence |= not active and k > first['tap']
        mixed |= min(n0, n1) > 0 and n0 != n1
        value = int(out[info['path'][k]])
        if value in info['signals']:
            winners.add(value)
    excluded = {rule for rule in RIVALS
                if not np.array_equal(out, _prediction(a, info, rule))}
    return dict(isolated=isolated, ties=ties, silence=silence,
                mixed=mixed, winners=winners, excluded=excluded)


def _line(a, b):
    dr = int(np.sign(b[0] - a[0]))
    dc = int(np.sign(b[1] - a[1]))
    if dr and dc:
        raise ValueError("Segment must be orthogonal")
    n = abs(b[0] - a[0]) + abs(b[1] - a[1])
    return [(a[0] + i * dr, a[1] + i * dc) for i in range(n + 1)]


def _trail(rng, rails, width=None):
    h = int(rng.integers(23, 31)) if rails == 3 else int(rng.integers(18, 28))
    w = int(rng.integers(22, 31)) if width is None else width
    left, right = int(rng.integers(1, 4)), w - int(rng.integers(2, 5))
    if rails == 3:
        rows = [int(rng.integers(2, 5)), int(rng.integers(10, 14)), h - int(rng.integers(2, 5))]
    else:
        rows = [int(rng.integers(2, 6)), h - int(rng.integers(3, 7))]
    waypoints = [(rows[0], left)]
    for i, r in enumerate(rows):
        end = right if i % 2 == 0 else left
        waypoints.append((r, end))
        if i + 1 < len(rows):
            waypoints.append((rows[i + 1], end))
    # Unequal endpoints and a short extra tail vary the geometry further.
    if rng.random() < .45:
        r, c = waypoints[-1]
        waypoints.append((r + int(rng.integers(1, min(4, h - r))), c))
    path = [waypoints[0]]
    for a, b in zip(waypoints, waypoints[1:]):
        path.extend(_line(a, b)[1:])
    if rng.random() < .5:
        path.reverse()
    return (h, w), path


def _place_panel(rng, shape, path, existing, wanted_ports, max_period=6):
    h, w = shape
    track = set(path)
    indices = {p: k for k, p in enumerate(path)}
    for _ in range(250):
        tap = path[int(rng.integers(2, len(path) - 2))]
        dr, dc = DIRS[int(rng.integers(4))]
        tr, tc = -dc, dr
        period = int(rng.integers(1, max_period + 1))
        span = int(rng.integers(max(3, 2 * wanted_ports - 1), 8))
        depth = int(rng.integers(2, 5))
        offset = int(rng.integers(span))
        corner = (tap[0] + (period + 1) * dr - offset * tr,
                  tap[1] + (period + 1) * dc - offset * tc)
        panel = {(corner[0] + d * dr + t * tr, corner[1] + d * dc + t * tc)
                 for d in range(depth) for t in range(span)}
        if (any(not (0 <= r < h and 0 <= c < w) for r, c in panel)
                or panel & _halo(existing | track)):
            continue
        candidates = []
        # Look outward from every face, so one panel can send clocks of unequal
        # lengths to different parts of the trail, including different sides.
        existing_halo = _halo(existing)
        for terminal in sorted(panel):
            for vr, vc in DIRS:
                cells = set()
                for length in range(1, max_period + 1):
                    p = terminal[0] + length * vr, terminal[1] + length * vc
                    if (not (0 <= p[0] < h and 0 <= p[1] < w)
                            or p in panel or p in track or p in existing_halo):
                        break
                    cells.add(p)
                    target = terminal[0] + (length + 1) * vr, terminal[1] + (length + 1) * vc
                    if target not in indices:
                        continue
                    if (indices[target] != 0
                            and {q for p in cells for q in _neighbors(p) if q in track} == {target}
                            and {q for p in cells for q in _neighbors(p) if q in panel} == {terminal}):
                        candidates.append(dict(cells=set(cells), length=length,
                                               tap=indices[target], terminal=terminal))
                    break
        rng.shuffle(candidates)
        rods, occupied = [], set()
        for candidate in candidates:
            if (not candidate['cells'] & _halo(occupied)
                    and candidate['tap'] not in {r['tap'] for r in rods}):
                rods.append(candidate)
                occupied |= candidate['cells']
                if len(rods) == wanted_ports:
                    return panel, rods
    return None


def _pattern(rng, cells, majority, minority):
    cells = sorted(cells)
    area = len(cells)
    band = int(rng.integers(3))
    nminor = (area - 1) // 2 if band == 0 else 1 if band == 1 else max(1, area // 4)
    style = int(rng.integers(3))
    if style == 0:
        order = rng.permutation(area)
    elif style == 1:
        center = cells[int(rng.integers(area))]
        order = sorted(range(area), key=lambda i: (abs(cells[i][0] - center[0])
                                                 + abs(cells[i][1] - center[1]), i))
    else:
        order = sorted(range(area), key=lambda i: ((sum(cells[i]) % 2), i))
    minority_cells = {cells[int(i)] for i in order[:nminor]}
    return {p: minority if p in minority_cells else majority for p in cells}


def _construct(rng, roles, rails=2, count=None, lesson=False, min_length=0):
    bg, marker, trail, clock, s0, s1 = roles
    for _ in range(140):
        shape, path = _trail(rng, rails)
        if len(path) <= min_length:
            continue
        n = int(rng.integers(2, 5 if rails == 2 else 7)) if count is None else count
        weights = [int(rng.choice([1, 2, 3], p=[.40, .45, .15])) for _ in range(n)]
        if sum(weights) < 3:
            weights[int(rng.integers(n))] = 2
        if lesson:
            weights[int(rng.integers(n))] = 1
            if max(weights) == 1:
                weights[(weights.index(1) + 1) % n] = 2
        if sum(weights) > 12:
            continue
        existing, panels, records = set(), [], []
        failed = False
        for i, weight in enumerate(weights):
            placed = _place_panel(rng, shape, path, existing, weight)
            if placed is None:
                failed = True
                break
            panel, rods = placed
            panels.append(panel)
            existing |= panel | set().union(*(r['cells'] for r in rods))
            records.extend(dict(r, panel=i) for r in rods)
        if failed:
            continue
        labels = [int(rng.choice([s0, s1])) for _ in panels]
        if len(set(labels)) == 1:
            labels[int(rng.integers(n))] = s1 if labels[0] == s0 else s0
        a = np.full(shape, bg, dtype=np.int8)
        for p in path:
            a[p] = trail
        a[path[0]] = marker
        for panel, label in zip(panels, labels):
            values = _pattern(rng, panel, label, s1 if label == s0 else s0)
            for p, value in values.items():
                a[p] = value
        for r in records:
            for p in r['cells']:
                a[p] = clock
        # Labels are computed from construction records, never by solve.
        out = a.copy()
        for k, p in enumerate(path[1:], 1):
            votes = [labels[r['panel']] for r in records
                     if k >= r['tap'] and (k - r['tap']) % r['length'] == 0]
            n0, n1 = votes.count(s0), votes.count(s1)
            if n0 != n1:
                out[p] = s0 if n0 > n1 else s1
        try:
            _, info = _analyze(a)
        except ValueError:
            continue
        if lesson:
            evidence = _evidence(a, info, out)
            if not (evidence['isolated'] and evidence['ties'] and evidence['silence']
                    and evidence['mixed'] and len(evidence['winners']) == 2
                    and evidence['excluded'] == set(RIVALS)):
                continue
        return {'input': a, 'output': out}
    raise RuntimeError("Bounded geometric sampling exhausted")


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.flat))


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.choice(10, 6, replace=False)))
    mode = int(rng.integers(3))
    base_rotation = int(rng.integers(4))
    varying_palette = bool(rng.integers(2))
    lesson_slot = int(rng.integers(num_train))
    puzzle, seen, train_lengths = {'train': [], 'test': []}, set(), []
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            roles = list(map(int, rng.choice(10, 6, replace=False))) if varying_palette else palette
            rails = 3 if split == 'test' and mode in (0, 1) else 2
            panels = int(rng.integers(4, 7)) if split == 'test' and mode == 1 else None
            rotation = (base_rotation + 1) % 4 if split == 'test' and mode == 2 else base_rotation
            if split == 'train' and i != lesson_slot and rng.random() < .3:
                rotation = (base_rotation + 2) % 4
            for _ in range(10):
                pair = _construct(rng, roles, rails, panels, split == 'train' and i == lesson_slot,
                                  max(train_lengths) if split == 'test' and mode == 0 else 0)
                pair = {k: np.rot90(v, rotation).copy() for k, v in pair.items()}
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("Distinct input sampling exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                train_lengths.append(len(_analyze(pair['input'])[1]['path']))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, out = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            _, info = _analyze(a)
            # Check individual cells, including ties, blanks, and all preserved roles.
            trail_cells = set(info['path'][1:])
            for p in np.ndindex(a.shape):
                if p not in trail_cells:
                    assert out[p] == a[p], 'Non-trail cell changed'
            for k, p in enumerate(info['path'][1:], 1):
                totals = [sum(k >= r['tap'] and (k - r['tap']) % r['length'] == 0
                              for r in info['rods'] if r['label'] == s) for s in info['signals']]
                expected = (info['track_color'] if totals[0] == totals[1]
                            else info['signals'][int(totals[1] > totals[0])])
                assert out[p] == expected, 'Period, phase, direction, local vote or port tally error'
            assert not np.array_equal(a, out)
            if split == 'train':
                evidence.append(_evidence(a, info, out))
    assert any(e['isolated'] for e in evidence), 'No isolated recurrence demonstration'
    for field in ('ties', 'silence', 'mixed'):
        assert any(e[field] for e in evidence), f'Missing {field} evidence'
    assert set().union(*(e['excluded'] for e in evidence)) == set(RIVALS), 'A simpler rival fits training'
    assert any(e['isolated'] and e['ties'] and e['silence'] and e['mixed']
               and len(e['winners']) == 2 and e['excluded'] == set(RIVALS)
               for e in evidence), 'Need one complete interacting teaching scene'


def self_test():
    """Bounded semantic, metamorphic, teaching and 42-seed diversity checks."""
    shapes, paths, palettes, periods, panel_counts, rod_counts = set(), set(), set(), set(), set(), set()
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in puzzle for p in puzzle[s]))
        for split in puzzle:
            for pair in puzzle[split]:
                a, info = _analyze(pair['input'])
                assert np.array_equal(solve(a), pair['output'])
                shapes.add(a.shape)
                paths.add(tuple(info['path']))
                palettes.add((info['background'], info['marker'], info['track_color'],
                              info['clock_color'], *info['signals']))
                periods.update(r['length'] for r in info['rods'])
                panel_counts.add(len(info['panels']))
                rod_counts.add(len(info['rods']))
    assert len(signatures) == 42
    assert len(shapes) >= 45 and len(paths) >= 130 and len(palettes) >= 35
    assert periods == set(range(1, 7))
    assert {2, 3, 4, 5, 6} <= panel_counts and len(rod_counts) >= 7
    # A hand-counted clock tally, separate from generation and its geometry.
    info = dict(signals=(1, 2), track_color=8,
                panels=[set(range(6)), set(range(15))],
                rods=[dict(tap=1, length=2, panel=0, label=1, contact=2),
                      dict(tap=3, length=2, panel=0, label=1, contact=1),
                      dict(tap=4, length=3, panel=1, label=2, contact=2)])
    # k=1..12: two ports of panel 0 overpower panel 1 at k=7.
    expected = [1, 8, 1, 2, 1, 8, 1, 8, 1, 2, 1, 8]
    assert [_value(info, _ballots(info, k)) for k in range(1, 13)] == expected
    # A manually drawn, straight-trail member outside the sampling templates.
    # Its clocks of lengths 8, 8 and 7 test the full specified period range.
    a = np.zeros((18, 30), dtype=np.int8)
    a[15, 1:29] = 8
    a[15, 1] = 3
    a[5:7, 5:10] = 1
    a[5, 5] = a[6, 9] = 2
    a[6:8, 17:22] = 2
    a[6, 17] = a[7, 21] = 1
    a[7:15, 5] = a[7:15, 9] = 5
    a[8:15, 19] = 5
    expected = a.copy()
    expected[15, [5, 9, 13, 17, 21, 25]] = 1
    expected[15, [19, 26]] = 2
    assert np.array_equal(solve(a), expected)
    sample = generate_puzzle(31415, 2, 1)
    for pair in sample['train'] + sample['test']:
        for turn in range(4):
            assert np.array_equal(solve(np.rot90(pair['input'], turn)), np.rot90(pair['output'], turn))
        mapping = np.arange(9, -1, -1, dtype=np.int8)
        assert np.array_equal(solve(mapping[pair['input']]), mapping[pair['output']])
        # Permuting panel pixels preserves votes and all trail decisions.
        a, info = _analyze(pair['input'])
        changed, expected = a.copy(), pair['output'].copy()
        for cells in info['panels']:
            positions = sorted(cells)
            values = sorted(int(a[p]) for p in positions)
            for p, value in zip(positions, values):
                changed[p] = value
                expected[p] = value
        assert np.array_equal(solve(changed), expected)
    corrupt = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]] for s in sample}
    a, info = _analyze(corrupt['test'][0]['input'])
    corrupt['test'][0]['output'][info['path'][-1]] = info['marker']
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('Wrong output accepted')
    for counts in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    assert tuple(_key(p['input']) for s in sample for p in sample[s]) == tuple(
        _key(p['input']) for s in sample for p in generate_puzzle(31415, 2, 1)[s])


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, semantic and geometric invariants')
