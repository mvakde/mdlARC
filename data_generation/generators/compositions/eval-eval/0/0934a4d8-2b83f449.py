"""Mirror repair zones that activate switches and determine bead settling.

0 is background, 9 is the rectangular repair zone. The other three colors
are inferred geometrically. No generation parameters are used by solve.
"""
from itertools import permutations

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _box(mask):
    rr, cc = np.nonzero(mask)
    assert len(rr)
    r, s, c, d = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
    assert len(rr) == (s - r) * (d - c), "Repair zone must be a solid rectangle"
    return r, s, c, d


def _neighbors(mask, cell):
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in _DIRS
            if 0 <= r + dr < mask.shape[0] and 0 <= c + dc < mask.shape[1]
            and mask[r + dr, c + dc]]


def _paths(mask):
    """Require complete, nonbranching paths, monotone across the lane axis."""
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        component, todo = {start}, [start]
        for cell in todo:
            for mate in _neighbors(mask, cell):
                if mate not in component:
                    component.add(mate)
                    todo.append(mate)
        unseen.difference_update(component)
        degrees = {cell: len(_neighbors(mask, cell)) for cell in component}
        assert all(degree in (1, 2) for degree in degrees.values()), "Branch or isolated track"
        ends = sorted(cell for cell, degree in degrees.items() if degree == 1)
        assert len(ends) == 2, "Track cycle"
        ordered = [ends[0]]
        previous = None
        while ordered[-1] != ends[1]:
            choices = [x for x in _neighbors(mask, ordered[-1]) if x != previous]
            assert len(choices) == 1
            previous = ordered[-1]
            ordered.append(choices[0])
        assert len(ordered) == len(component)
        vertical_steps = {b[0] - a[0] for a, b in zip(ordered, ordered[1:]) if b[0] != a[0]}
        assert len(vertical_steps) <= 1, "Internal turn against settling direction"
        result.append(ordered)
    return result


def _centers(full, track, divider):
    h, w = full.shape
    assert h >= 5 and h % 2 == 1 and w >= 9
    assert np.all(np.isin(full[::2], [0, track]))
    assert np.all(np.isin(full[1::2], [0, divider]))
    assert np.all(full[::2, [0, w - 1]] == track)
    centers, breaks = [], []
    for r in range(1, h, 2):
        xs = set(map(int, np.flatnonzero(full[r] == divider)))
        for start in sorted(x for x in xs if x - 1 not in xs):
            assert {start, start + 1, start + 2} <= xs and start + 3 not in xs
            assert 1 <= start and start + 3 < w
            c = start + 1
            assert np.all(full[[r - 1, r + 1], c - 1:c + 2] == track)
            centers.append((r, c))
    for r in range(0, h, 2):
        xs = list(map(int, np.flatnonzero(full[r] == 0)))
        assert all(2 <= c <= w - 3 for c in xs)
        assert all(b - a >= 3 for a, b in zip(xs, xs[1:]))
        breaks.extend((r, c) for c in xs)
    assert centers, "Blueprint must contain visible switch evidence"
    return centers, breaks


def _turn(full, active, track, divider):
    writes = {}
    for r, c in active:
        for y in range(r - 1, r + 2):
            for x in range(c - 1, c + 2):
                value = divider if x == c else track
                assert (y, x) not in writes or writes[y, x] == value, "Conflicting switch writes"
                writes[y, x] = value
    out = full.copy()
    for cell, value in writes.items():
        out[cell] = value
    return out, writes


def _network(full, hidden, track, divider):
    centers, breaks = _centers(full, track, divider)
    active = [cell for cell in centers if hidden[cell]]
    structural, writes = _turn(full, active, track, divider)
    paths = _paths(structural == track)
    for path in paths:
        for r, c in (path[0], path[-1]):
            assert r % 2 == 0
            assert c in (0, full.shape[1] - 1) or full[r, c - 1] == 0 or full[r, c + 1] == 0
    return dict(full=full, centers=centers, breaks=breaks, active=active,
                structural=structural, writes=writes, paths=paths)


def _settle(info, beads, bead, sign):
    endpoints = {cell for p in info['paths'] for cell in (p[0], p[-1])}
    assert beads <= endpoints, "Every bead must start at an exposed terminal"
    answer = info['structural'].copy()
    destinations = set()
    singles = collisions = levels = 0
    for path in info['paths']:
        here = beads & {path[0], path[-1]}
        if not here:
            continue
        if path[0][0] == path[-1][0]:
            targets = here
            levels += 1
        else:
            order = path if sign * path[0][0] > sign * path[-1][0] else path[::-1]
            targets = set(order[:len(here)])
            singles += len(here) == 1 and targets != here
            collisions += len(here) == 2
        destinations.update(targets)
        for cell in targets:
            answer[cell] = bead
    assert np.count_nonzero(answer == bead) == len(beads)
    return answer, dict(beads=beads, destinations=destinations, singles=singles,
                        collisions=collisions, levels=levels, moved=beads != destinations)


def _completion(core, track, bead, divider):
    """Remove the movable layer, then recover each erased reflection orbit."""
    h, w = core.shape
    assert h >= 5 and h % 2 == 1 and w >= 9
    hidden = core == 9
    box = _box(hidden)
    assert np.all(np.isin(core[::2], [0, track, bead, 9]))
    assert np.all(np.isin(core[1::2], [0, divider, 9]))
    assert np.all(np.isin(core[::2, [0, w - 1]], [track, bead, 9]))
    full = core.copy()
    full[full == bead] = track
    for r, c in np.argwhere(hidden):
        orbit = {(int(r), int(c)), (h - 1 - int(r), int(c)),
                 (int(r), w - 1 - int(c)), (h - 1 - int(r), w - 1 - int(c))}
        values = {int(full[cell]) for cell in orbit if not hidden[cell]}
        assert len(values) == 1, "Missing or conflicting reflection witnesses"
        full[r, c] = next(iter(values))
    assert np.array_equal(full, full[::-1]) and np.array_equal(full, full[:, ::-1])
    info = _network(full, hidden, track, divider)
    r, s, c, d = box
    assert r + s != h, "Zone center cannot be on the lane-normal reflection axis"
    sign = 1 if r + s > h else -1
    beads = set(map(tuple, np.argwhere(core == bead)))
    answer, traffic = _settle(info, beads, bead, sign)
    info.update(traffic)
    info.update(hidden=hidden, box=box, sign=sign, palette=(track, bead, divider),
                row_unavailable=int(np.count_nonzero(hidden & hidden[::-1])),
                col_unavailable=int(np.count_nonzero(hidden & hidden[:, ::-1])))
    return answer, info


def _infer(input_grid):
    inp = _grid(input_grid)
    _box(inp == 9)
    colors = sorted(set(map(int, np.unique(inp))) - {0, 9})
    assert len(colors) == 3 and 0 in inp and 9 in inp
    locations = np.argwhere(inp != 0)
    r0, c0 = map(int, locations.min(axis=0))
    r1, c1 = map(int, locations.max(axis=0) + 1)
    crop = inp[r0:r1, c0:c1]
    candidates = []
    for axis in (0, 1):
        core = crop if axis == 0 else crop.T
        for track, bead, divider in permutations(colors):
            try:
                answer, info = _completion(core, track, bead, divider)
            except AssertionError:
                continue
            out = np.zeros(inp.shape, dtype=np.int8)
            out[r0:r1, c0:c1] = answer if axis == 0 else answer.T
            info.update(axis=axis, core_box=(r0, r1, c0, c1), core=core.copy())
            candidates.append((out, info))
    assert len(candidates) == 1, "Input must identify exactly one lane axis and role assignment"
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from input pixels under the fixed family conventions."""
    return _infer(input_grid)[0]


def _orbit(cell, shape):
    r, c = cell
    h, w = shape
    return {(r, c), (h - 1 - r, c), (r, w - 1 - c), (h - 1 - r, w - 1 - c)}


def _sample_centers(rng, h, w, richness):
    pool = [(r, c) for r in range(1, h, 2) for c in range(2, w - 2)
            if r <= h - 1 - r and c <= w - 1 - c]
    rng.shuffle(pool)
    chosen = set()
    wanted = int(rng.integers(1, 3 + richness))
    orbits = 0
    for cell in pool:
        proposed = chosen | _orbit(cell, (h, w))
        cells = sorted(proposed)
        if any((r == y and abs(c - x) < 4) or (abs(r - y) == 2 and abs(c - x) < 2)
               for i, (r, c) in enumerate(cells) for y, x in cells[i + 1:]):
            continue
        chosen = proposed
        orbits += 1
        if orbits >= wanted:
            break
    assert chosen
    return sorted(chosen)


def _window(rng, shape, centers, mode, sign, edge):
    h, w = shape
    middle = h // 2
    if mode == 'row':
        # Hide a column-seam crossing patch wholly in one row half.
        possible = [r for r, c in centers if r < middle]
        assert possible
        anchor = int(rng.choice(possible))
        r = 0 if edge else int(rng.integers(1, anchor + 1))
        s = int(rng.integers(anchor + 1, middle + 1))
        if sign > 0:
            r, s = h - s, h - r
        candidates = [(c, d) for c in range(1, w // 2) for d in range(w // 2 + 2, w)
                      if any(r <= y < s and c <= x < d for y, x in centers)]
        assert candidates
        c, d = candidates[int(rng.integers(len(candidates)))]
    else:
        # Hide a row-seam crossing patch wholly in one column half.
        candidates = [(r, s) for r in range(1, middle) for s in range(middle + 2, h)
                      if (r + s - h) * sign > 0]
        rng.shuffle(candidates)
        windows = []
        for r, s in candidates:
            possible = [x for y, x in centers if r <= y < s and x < w // 2]
            for anchor in possible:
                c = 0 if edge else int(rng.integers(1, anchor + 1))
                d = int(rng.integers(anchor + 1, w // 2 + 1))
                windows.append((r, s, c, d))
        assert windows
        r, s, c, d = windows[int(rng.integers(len(windows)))]
        if rng.random() < .5:
            c, d = w - d, w - c
    hidden = np.zeros(shape, dtype=bool)
    hidden[r:s, c:d] = True
    assert any(hidden[cell] for cell in centers)
    return hidden


def _sample_breaks(rng, shape, centers, active, hidden, richness):
    h, w = shape
    breaks = set()

    def add(cell):
        proposed = breaks | _orbit(cell, shape)
        if any(c < 2 or c > w - 3 or any(abs(y - r) == 1 and abs(x - c) <= 1 for y, x in centers)
               for r, c in proposed):
            return False
        cells = sorted(proposed)
        if any(r == y and abs(c - x) < 3 for i, (r, c) in enumerate(cells) for y, x in cells[i + 1:]):
            return False
        breaks.update(proposed)
        return True

    for r in range(0, h, 2):
        ports = sorted((c, y) for y, c in active if abs(y - r) == 1)
        for (a, side_a), (b, side_b) in zip(ports, ports[1:]):
            if side_a == side_b and not any(y == r and a < c < b for y, c in breaks):
                pool = list(range(a + 2, b - 1))
                rng.shuffle(pool)
                assert any(add((r, c)) for c in pool), "Cannot open a symmetric U turn"
    pool = [(r, c) for r in range(0, h, 2) for c in range(2, w - 2)]
    rng.shuffle(pool)
    # Sometimes prioritize erasure of a real track break, not only switches.
    if rng.random() < .45:
        pool.sort(key=lambda cell: not any(hidden[x] for x in _orbit(cell, shape)))
    wanted = int(rng.integers(richness + 1))
    added = 0
    for cell in pool:
        if added >= wanted:
            break
        if cell not in breaks and add(cell):
            added += 1
    return sorted(breaks)


def _base(shape, centers, breaks, track, divider):
    full = np.zeros(shape, dtype=np.int8)
    full[::2] = track
    for r, c in centers:
        full[r, c - 1:c + 2] = divider
    for cell in breaks:
        full[cell] = 0
    return full


def _beads(rng, info, hidden, sign, density, demonstration):
    beads = set()
    probability = float(rng.uniform(*density))
    for path in info['paths']:
        beads.update(cell for cell in (path[0], path[-1]) if not hidden[cell] and rng.random() < probability)
    nonlevel = [p for p in info['paths'] if p[0][0] != p[-1][0]]
    if demonstration == 'double':
        eligible = [p for p in nonlevel if not hidden[p[0]] and not hidden[p[-1]]
                    and _switches_on_path(p, info['active']) >= 2]
    else:
        eligible = [p for p in nonlevel if not hidden[min((p[0], p[-1]), key=lambda cell: sign * cell[0])]]
    assert eligible
    path = eligible[int(rng.integers(len(eligible)))]
    beads.difference_update((path[0], path[-1]))
    beads.add(min((path[0], path[-1]), key=lambda cell: sign * cell[0]))
    if demonstration == 'double':
        beads.update((path[0], path[-1]))
    if demonstration in ('single', 'double'):
        levels = [p for p in info['paths'] if p[0][0] == p[-1][0]
                  and any(not hidden[cell] for cell in (p[0], p[-1]))]
        assert levels
        p = levels[int(rng.integers(len(levels)))]
        eligible = [cell for cell in (p[0], p[-1]) if not hidden[cell]]
        beads.add(eligible[int(rng.integers(len(eligible)))])
    return beads


def _switches_on_path(path, active):
    cells = set(path)
    return sum((r, c - 1) in cells or (r, c + 1) in cells for r, c in active)


def _inactive_overlap(info):
    return [(r, c) for r, c in info['centers'] if not info['hidden'][r, c]
            and np.any(info['hidden'][r, c - 1:c + 2])]


def _make_pair(rng, palette, mode, sign, edge, demonstration, richness, rotation, seen):
    track, bead, divider = palette
    for _ in range(250):
        try:
            lanes = int(rng.integers(4, 14))
            w = int(rng.integers(11, 30))
            shape = (2 * lanes - 1, w)
            centers = _sample_centers(rng, *shape, richness)
            hidden = _window(rng, shape, centers, mode, sign, edge)
            active = [cell for cell in centers if hidden[cell]]
            breaks = _sample_breaks(rng, shape, centers, active, hidden, richness)
            full = _base(shape, centers, breaks, track, divider)
            info = _network(full, hidden, track, divider)
            density = ((.08, .30), (.30, .65), (.75, 1.0))[int(rng.integers(3))]
            beads = _beads(rng, info, hidden, sign, density, demonstration)
            output, traffic = _settle(info, beads, bead, sign)
            assert traffic['moved']
            inp = full.copy()
            inp[hidden] = 9
            for cell in beads:
                inp[cell] = bead
            margins = []
            for n in shape:
                total = int(rng.integers(min(3, 30 - n) + 1))
                before = int(rng.integers(total + 1))
                margins.append((before, total - before))
            inp, output = (np.rot90(np.pad(a, margins), rotation).copy() for a in (inp, output))
            inferred, record = _infer(inp)
            assert np.array_equal(inferred, output)
            assert record['active'] and len(record['centers']) > len(record['active'])
            if demonstration == 'single':
                assert _inactive_overlap(record), 'Demonstrate center rather than overlap activation'
                assert any(not record['hidden'][cell] for cell in record['writes'])
            if demonstration == 'double':
                assert any(np.all(record['hidden'][r, c - 1:c + 2]) for r, c in record['active']), 'Hide a complete bar, requiring reflection evidence'
            if _key(inp) in seen:
                continue
            return {'input': inp, 'output': output}
        except AssertionError:
            continue
    raise RuntimeError('Bounded sampler could not construct a valid repair network')


def _count(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lower <= int(value) <= upper:
        raise ValueError(f'{name} must be in {lower}..{upper}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    nq = _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    seed = int(seed)
    rng = np.random.default_rng(seed if seed >= 0 else seed % (1 << 64))
    palette = tuple(map(int, rng.choice(np.arange(1, 9), 3, replace=False)))
    vary_palette = rng.random() < .40
    richness = int(rng.integers(1, 5))
    rotation = int(rng.integers(4))
    first_sign = int(rng.choice([-1, 1]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            roles = tuple(map(int, rng.choice(np.arange(1, 9), 3, replace=False))) if vary_palette else palette
            mode = ('row' if i == 0 else 'column') if split == 'train' and i < 2 else str(rng.choice(['row', 'column']))
            sign = first_sign * (1 if i == 0 else -1) if split == 'train' and i < 2 else int(rng.choice([-1, 1]))
            demonstration = ('single' if i == 0 else 'double') if split == 'train' and i < 2 else 'move'
            pair = _make_pair(rng, roles, mode, sign, split == 'test', demonstration,
                              richness, rotation, seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[sp], list) for sp in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            predicted, info = _infer(inp)
            assert np.array_equal(predicted, out), 'Wrong repaired network or traffic label'
            assert inp.shape == out.shape and 9 not in out
            assert info['active'] and len(info['centers']) > len(info['active'])
            assert info['moved'], 'Each generated pair must make routing consequential'
            track, bead, divider = info['palette']
            assert np.count_nonzero(inp == bead) == np.count_nonzero(out == bead)
            assert not any(info['hidden'][cell] for cell in info['beads'])
            # Erased structure includes real dividers, not a cosmetic blank mask.
            assert np.any((info['full'] == divider) & info['hidden'])
            # Counterfactual without quarter-turns: every original lane path is
            # level, so all beads retain their positions. The actual answer differs.
            assert info['destinations'] != info['beads']
            core_out = out[info['core_box'][0]:info['core_box'][1], info['core_box'][2]:info['core_box'][3]]
            if info['axis']:
                core_out = core_out.T
            normalized = core_out.copy()
            normalized[normalized == bead] = track
            assert np.array_equal(normalized, info['structural'])
            # All inactive bars remain in their original state.
            for r, c in set(info['centers']) - set(info['active']):
                assert np.all(normalized[r, c - 1:c + 2] == divider)
            records[split].append(info)
    train = records['train']
    assert {x['sign'] for x in train} == {-1, 1}, 'Demonstrate both patch-directed signs'
    assert any(x['singles'] for x in train) and any(x['collisions'] for x in train)
    assert any(x['levels'] for x in train)
    assert any(_inactive_overlap(x) for x in train)
    assert any(any(not x['hidden'][cell] for cell in x['writes']) for x in train)
    assert any(any(_switches_on_path(p, x['active']) >= 2
                   and {p[0], p[-1]} <= x['beads'] for p in x['paths']) for x in train)
    assert any(any(np.all(x['hidden'][r, c - 1:c + 2]) for r, c in x['active']) for x in train)
    assert any(x['row_unavailable'] == 0 and x['col_unavailable'] > 0 for x in train)
    assert any(x['col_unavailable'] == 0 and x['row_unavailable'] > 0 for x in train)
    # The sampler's educational holdout: interior repair to core-edge repair.
    for split in ('train', 'test'):
        for info in records[split]:
            r, s, c, d = info['box']
            edge = r == 0 or c == 0 or s == info['full'].shape[0] or d == info['full'].shape[1]
            assert edge == (split == 'test')
    return None


def _signature(puzzle):
    return tuple(tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                       for p in puzzle[sp]) for sp in ('train', 'test'))


def self_test():
    """Bounded coverage, counterfactual, mutation and count checks over 42 seeds."""
    signatures, shapes, palettes, layouts = set(), set(), set(), set()
    active_counts, switch_counts, break_counts, bead_counts = set(), set(), set(), set()
    axes, directions, hidden_breaks, overlaps = set(), set(), 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        if seed in (0, 17, 41):
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                prediction, info = _infer(pair['input'])
                assert np.array_equal(prediction, pair['output'])
                shapes.add(pair['input'].shape)
                palettes.add(info['palette'])
                axes.add(info['axis'])
                directions.add((info['axis'], info['sign']))
                active_counts.add(len(info['active']))
                switch_counts.add(len(info['centers']))
                break_counts.add(len(info['breaks']))
                bead_counts.add(len(info['beads']))
                hidden_breaks += any(info['hidden'][cell] for cell in info['breaks'])
                overlaps += len(info['writes']) < 9 * len(info['active'])
                norm = np.zeros(info['core'].shape, dtype=np.int8)
                for code, color in enumerate(info['palette'], 1):
                    norm[info['core'] == color] = code
                norm[info['hidden']] = 4
                layouts.add(_key(norm))
    assert len(signatures) == 42 and len(shapes) >= 35 and len(layouts) >= 180
    assert len(palettes) >= 25 and axes == {0, 1} and len(directions) == 4
    assert len(active_counts) >= 4 and len(switch_counts) >= 6
    assert len(break_counts) >= 4 and len(bead_counts) >= 8
    assert hidden_breaks >= 10 and overlaps >= 5
    # A structural counterfactual: change the intact reflected witnesses of one
    # erased switch orbit. Same window and beads, but a different destination.
    centers = [(1, 5), (1, 11), (3, 3), (3, 13), (9, 3), (9, 13), (11, 5), (11, 11)]
    full = _base((13, 17), centers, [], 8, 7)
    inp = full.copy()
    inp[8:12, 1:7] = 9
    inp[8, 16] = 2
    inp[12, 0] = 2
    inp[0, 0] = 2
    answer, info = _infer(inp)
    assert info['destinations'] == {(12, 0), (12, 1), (0, 0)}
    assert info['active'] == [(9, 3), (11, 5)]
    changed_full = full.copy()
    for r, c in _orbit((9, 3), full.shape):
        changed_full[r, c - 1:c + 2] = 0
    changed = changed_full.copy()
    changed[8:12, 1:7] = 9
    for cell in info['beads']:
        changed[cell] = 2
    different, changed_info = _infer(changed)
    assert changed_info['active'] == [(11, 5)]
    assert changed_info['destinations'] != info['destinations']
    assert not np.array_equal(different, answer)
    # Omit B: repair the same static blueprint but leave all bars parallel.
    repair_only = full.copy()
    for cell in info['beads']:
        repair_only[cell] = 2
    assert not np.array_equal(repair_only, answer)
    # Rotations, pair-specific role recolorings, and input immutability.
    for turn in range(4):
        rotated = np.rot90(inp, turn).copy()
        assert np.array_equal(solve(rotated), np.rot90(answer, turn))
        assert np.array_equal(rotated, np.rot90(inp, turn))
    remap = np.arange(10, dtype=np.int8)
    remap[[8, 2, 7]] = [1, 6, 4]
    assert np.array_equal(solve(remap[inp]), remap[answer])
    # Visible conflicts and a fully erased central orbit must fail.
    conflict = inp.copy()
    conflict[1, 4] = 0
    erased = full.copy()
    erased[5:8, 7:10] = 9
    erased[0, 0] = 2
    for invalid in (conflict, erased):
        try:
            solve(invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid reflection evidence accepted')
    wrong = generate_puzzle(5, 2, 1)
    wrong['test'][0]['output'][0, 0] = 9
    try:
        validate_generated_puzzle(wrong)
    except AssertionError:
        pass
    else:
        raise AssertionError('Wrong output accepted')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            assert len(puzzle['train']) == nt and len(puzzle['test']) == nq
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    return dict(seeds=42, pairs=210, input_shapes=len(shapes), normalized_layouts=len(layouts),
                palettes=len(palettes), active_counts=sorted(active_counts), switch_counts=sorted(switch_counts),
                break_counts=sorted(break_counts), bead_counts=sorted(bead_counts),
                hidden_break_pairs=hidden_breaks, overlapping_switch_pairs=overlaps)
