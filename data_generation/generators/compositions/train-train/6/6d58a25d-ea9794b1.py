"""Shadow-gated transparencies: rays are enabled by exposed trigger dots.

Only Python's standard library and NumPy are used. Source evidence is never read
at runtime. Spatial sheet order, glyph geometry, and visible palettes suffice.
"""

from itertools import permutations
import numpy as np


FLARE = ((0, 0), (1, -1), (1, 0), (1, 1), (2, -2),
         (2, -1), (2, 1), (2, 2), (3, -3), (3, 3))
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
HOST_SEEDS = (0, 112606804, 869211434, 351352533, 332049127, 661023538,
              619128739, 762110802, 100481845, 267761199, 49935109,
              468742032, 320090094, 954072919, 52328920, 682368367,
              554878068, 866378584, 855317019, 265951592, 755286394,
              933364889, 380247000, 706508458, 459383402, 461357792,
              425540528, 262783273, 932487175, 270562913, 546183358,
              510784188, 609012580, 302596005, 43818138, 159606023,
              99227441, 319259781, 137117512, 321150967, 763240095, 466924131)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(name + " is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("invalid ARC grid")
    return a.astype(np.int8, copy=False)


def _point(apex, direction, depth, lane):
    dr, dc = DIRECTIONS[direction]
    tr, tc = (0, 1) if dr else (1, 0)
    return apex[0] + dr * depth + tr * lane, apex[1] + dc * depth + tc * lane


def _coords(shape, apex, direction):
    rows, cols = np.indices(shape)
    dr, dc = DIRECTIONS[direction]
    depth = (rows-apex[0])*dr + (cols-apex[1])*dc
    lane = cols-apex[1] if dr else rows-apex[0]
    return depth, lane


def _glyph(apex, direction):
    return {_point(apex, direction, d, t) for d, t in FLARE}


def _find_glyph(panel, color):
    cells = {tuple(map(int, p)) for p in np.argwhere(panel == color)}
    if len(cells) != 10:
        return []
    return [(apex, direction) for apex in sorted(cells) for direction in range(4)
            if _glyph(apex, direction) == cells]


def _analyze(input_grid):
    a = _grid(input_grid)
    candidates = []
    for axis in (0, 1):
        for color in np.unique(a):
            full = np.all(a == color, axis=1-axis)
            seams = np.flatnonzero(full)
            if len(seams) not in (1, 2):
                continue
            if np.count_nonzero(a == color) != len(seams)*a.shape[1-axis]:
                continue
            count = len(seams)+1
            if (a.shape[axis]-(count-1)) % count:
                continue
            size = (a.shape[axis]-(count-1)) // count
            if size < 7 or list(seams) != [size+i*(size+1) for i in range(count-1)]:
                continue
            panels = [np.take(a, np.arange(i*(size+1), i*(size+1)+size), axis=axis)
                      for i in range(count)]
            sets = [set(map(int, np.unique(p))) for p in panels]
            common = set.intersection(*sets)
            if len(common) != 2 or any(len(s) != 3 for s in sets):
                continue
            for emitter in sorted(common):
                found = [_find_glyph(p, emitter) for p in panels]
                if not all(len(f) == 1 for f in found):
                    continue
                background = next(iter(common-{emitter}))
                inks = [next(iter(s-common)) for s in sets]
                if len(set(inks)) != count:
                    continue
                infos = []
                for ink, f in zip(inks, found):
                    apex, direction = f[0]
                    infos.append(dict(background=background, emitter=emitter,
                                      ink=ink, apex=apex, direction=direction))
                candidates.append((panels, infos, axis, int(color)))
    if len(candidates) != 1:
        raise ValueError("expected one unambiguous transparency strip")
    return candidates[0]


def _sheet(panel, info, covered, gate=True):
    depth, lane = _coords(panel.shape, info['apex'], info['direction'])
    eligible = (panel == info['ink']) & (depth > 3) & (abs(lane) <= 3)
    visible = eligible & ~covered
    active = tuple(sorted(map(int, np.unique(lane[visible if gate else eligible]))))
    rays = np.zeros(panel.shape, dtype=bool)
    for t in active:
        last = max(d for d, u in FLARE if u == t)
        rays |= (lane == t) & (depth > last)
    completed = panel.copy()
    completed[rays] = info['ink']
    opaque = (panel != info['background']) | rays
    return completed, opaque, dict(active=active, rays=rays, eligible=eligible,
                                    visible=visible, covered=covered.copy(), info=info)


def _render(panels, infos, gate=True, raw=False, order=None):
    output = np.full(panels[0].shape, infos[0]['background'], dtype=np.int8)
    covered = np.zeros(output.shape, dtype=bool)
    records = [None]*len(panels)
    for i in (range(len(panels)) if order is None else order):
        completed, opaque, record = _sheet(panels[i], infos[i], covered, gate)
        if raw:
            completed = panels[i]
            opaque = completed != infos[i]['background']
        write = opaque & ~covered
        output[write] = completed[write]
        covered |= opaque
        records[i] = record
    return output, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete and stack the sheets in their visible left-to-right/top-to-bottom order."""
    panels, infos, _, _ = _analyze(input_grid)
    return _render(panels, infos)[0]


def _assemble(panels, seam, axis):
    strip = []
    seam_shape = list(panels[0].shape)
    seam_shape[axis] = 1
    for i, panel in enumerate(panels):
        if i:
            strip.append(np.full(seam_shape, seam, dtype=np.int8))
        strip.append(panel)
    return np.concatenate(strip, axis=axis)


def _put_signals(rng, panel, info, lanes, protected=(), inert_max=5):
    depth, transverse = _coords(panel.shape, info['apex'], info['direction'])
    forbidden = set(protected)
    for t in lanes:
        pool = [tuple(map(int, p)) for p in np.argwhere((depth > 3) & (transverse == t))
                if tuple(map(int, p)) not in forbidden]
        if pool:
            amount = min(len(pool), int(rng.choice((1, 1, 1, 2, 3))))
            for j in rng.choice(len(pool), amount, replace=False):
                panel[pool[int(j)]] = info['ink']
    inert = (panel == info['background']) & ~((depth > 3) & (abs(transverse) <= 3))
    pool = [tuple(map(int, p)) for p in np.argwhere(inert)
            if tuple(map(int, p)) not in forbidden]
    amount = min(len(pool), int(rng.integers(inert_max+1)))
    for j in rng.choice(len(pool), amount, replace=False):
        panel[pool[int(j)]] = info['ink']


def _empty_sheet(shape, palette, layer, apex, direction):
    background, emitter, *rest = palette
    info = dict(background=background, emitter=emitter, ink=rest[layer],
                apex=apex, direction=direction)
    panel = np.full(shape, background, dtype=np.int8)
    for p in _glyph(apex, direction):
        panel[p] = emitter
    return panel, info


def _witness(rng, rescue, rotation, palette):
    """A crossing ray suppresses the middle trigger; redundancy reverses the cascade."""
    h, w = int(rng.integers(11, 31)), 9
    x = int(rng.integers(4, w-3))
    front_row = int(rng.integers(5, h-5))
    middle_row = int(rng.integers(0, front_row-3))
    back_row = int(rng.integers(front_row+2, h-3))
    front_col = int(rng.integers(0, x-3))
    back_col = int(rng.integers(0, x-3))
    setups = (((front_row, front_col), 1), ((middle_row, x), 0), ((back_row, back_col), 1))
    sheets = [_empty_sheet((h, w), palette, i, apex, d)
              for i, (apex, d) in enumerate(setups)]
    front, middle, back = [s[0] for s in sheets]
    front[front_row, int(rng.integers(x+1, w))] = sheets[0][1]['ink']
    middle[front_row, x] = sheets[1][1]['ink']
    if rescue:
        middle[front_row+1, x] = sheets[1][1]['ink']
    back[back_row, x] = sheets[2][1]['ink']
    # Protect the causal witnesses from unrelated original marks and top rays.
    top_pool = [t for t in range(-3, 4) if t not in (0, 1, back_row-front_row)]
    rng.shuffle(top_pool)
    middle_pool = [t for t in range(-3, 4) if t != 0]
    back_pool = middle_pool.copy()
    rng.shuffle(middle_pool)
    rng.shuffle(back_pool)
    extras = (top_pool[:int(rng.integers(0, min(3, len(top_pool))+1))],
              middle_pool[:int(rng.integers(0, 4))],
              back_pool[:int(rng.integers(0, 4))])
    for i, (panel, info) in enumerate(sheets):
        _put_signals(rng, panel, info, extras[i],
                     protected=((front_row, x), (front_row+1, x), (back_row, x)))
    panels = [np.rot90(p, rotation).copy() for p, _ in sheets]
    return _assemble(panels, palette[-1], 0 if rotation % 2 else 1)


def _free_scene(rng, count, axis, palette):
    narrow = 9 if count == 3 else int(rng.integers(9, 15))
    long = int(rng.integers(11, 31))
    shape = (long, narrow) if axis == 1 else (narrow, long)
    panels = []
    covered = np.zeros(shape, dtype=bool)
    for layer in range(count):
        direction = int(rng.integers(4))
        feasible = []
        for r in range(shape[0]):
            for c in range(shape[1]):
                cells = _glyph((r, c), direction)
                end = _point((r, c), direction, 5, 0)
                if (all(0 <= y < shape[0] and 0 <= x < shape[1] for y, x in cells)
                        and 0 <= end[0] < shape[0] and 0 <= end[1] < shape[1]):
                    feasible.append((r, c))
        apex = feasible[int(rng.integers(len(feasible)))]
        panel, info = _empty_sheet(shape, palette, layer, apex, direction)
        lanes = rng.choice(np.arange(-3, 4), int(rng.integers(1, 6)), replace=False)
        _put_signals(rng, panel, info, lanes)
        # Often add a genuinely occluded candidate, without prescribing whether
        # another dot on that lane survives. Its spatial eligibility is visible.
        depth, lane = _coords(shape, apex, direction)
        pool = np.argwhere(covered & (depth > 3) & (abs(lane) <= 3))
        if len(pool) and rng.random() < .8:
            panel[tuple(pool[int(rng.integers(len(pool)))])] = info['ink']
        _, opaque, _ = _sheet(panel, info, covered)
        covered |= opaque
        panels.append(panel)
    return _assemble(panels, palette[-1], axis)


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _program(input_grid):
    panels, infos, _, _ = _analyze(input_grid)
    _, records = _render(panels, infos)
    return tuple((i['direction'], r['active']) for i, r in zip(infos, records))


def _orders_fit(pairs):
    triples = [p for p in pairs if len(_analyze(p['input'])[0]) == 3]
    return [order for order in permutations(range(3)) if all(
        np.array_equal(_render(*_analyze(p['input'])[:2], order=order)[0], p['output'])
        for p in triples)]


def _lane_evidence(pairs):
    lanes, counts, inert = set(), set(), False
    for pair in pairs:
        panels, infos, _, _ = _analyze(pair['input'])
        _, records = _render(panels, infos)
        for panel, info, record in zip(panels, infos, records):
            lanes.update(record['active'])
            counts.add(len(record['active']))
            inert |= bool(np.any((panel == info['ink']) & ~record['eligible']))
    return lanes == set(range(-3, 4)) and 1 in counts and max(counts) > 1 and inert


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    rotation = int(rng.integers(4))
    base = tuple(map(int, rng.choice(10, 6, replace=False)))
    # The second witness swaps the three signal roles; its changed background
    # also rules out hard-coded transparency. Both use exactly the same rule.
    bg2 = int(rng.choice([v for v in range(10) if v not in base]))
    second = (bg2, base[1], base[3], base[4], base[2], base[5])
    for _ in range(160):
        witness_inputs = [_witness(rng, False, rotation, base),
                          _witness(rng, True, (rotation+2) % 4, second)]
        train = [dict(input=a, output=solve(a)) for a in witness_inputs]
        if _lane_evidence(train) and _orders_fit(train) == [(0, 1, 2)]:
            break
    else:
        raise RuntimeError('could not expose the sheet order')
    rng.shuffle(train)
    puzzle = dict(train=train, test=[])
    seen = {_signature(p['input']) for p in train}
    programs = {_program(p['input']) for p in train}
    for split, amount in (('train', num_train-2), ('test', num_test)):
        for index in range(amount):
            # Both card counts are taught before either is requested at test.
            count = 2 if split == 'train' and index == 0 else int(rng.integers(2, 4))
            if split == 'test' and num_train == 2:
                count = 3
            for _ in range(160):
                palette = tuple(map(int, rng.choice(10, 6, replace=False)))
                a = _free_scene(rng, count, 0 if rotation % 2 else 1, palette)
                panels, infos, _, _ = _analyze(a)
                out, _ = _render(panels, infos)
                if np.array_equal(out, _render(panels, infos, raw=True)[0]):
                    continue
                if np.array_equal(out, _render(panels, infos, gate=False)[0]):
                    continue
                if _signature(a) in seen or (split == 'test' and _program(a) in programs):
                    continue
                break
            else:
                raise RuntimeError('could not sample a distinct interactive scene')
            seen.add(_signature(a))
            if split == 'train':
                programs.add(_program(a))
            puzzle[split].append(dict(input=a, output=out))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check geometry, cellwise semantics, causal witnesses, order and test support."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    tr, te = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen, train_counts, train_directions, train_programs = set(), set(), set(), set()
    suppression = redundancy = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, out = _grid(pair['input']), _grid(pair['output'])
            assert _signature(a) not in seen
            seen.add(_signature(a))
            panels, infos, axis, seam = _analyze(a)
            assert out.shape == panels[0].shape
            assert seam not in np.unique(out)
            assert np.array_equal(solve(a), out)
            # A separate cell scan uses exposed ORIGINAL marks for activation,
            # then decides the visible winner at each registered coordinate.
            occupied = np.zeros(out.shape, dtype=bool)
            expected = np.full(out.shape, infos[0]['background'], dtype=np.int8)
            for panel, info in zip(panels, infos):
                assert np.count_nonzero(panel == info['emitter']) == 10
                assert np.count_nonzero(panel == info['background']) > panel.size//2
                depth, lane = _coords(panel.shape, info['apex'], info['direction'])
                marks = (panel == info['ink']) & (depth > 3) & (abs(lane) <= 3)
                active = set(map(int, lane[marks & ~occupied]))
                for t in active:
                    if np.any(marks & (lane == t) & occupied):
                        if split == 'train':
                            redundancy = True
                next_occupied = occupied.copy()
                for r in range(out.shape[0]):
                    for c in range(out.shape[1]):
                        t, d = int(lane[r, c]), int(depth[r, c])
                        ray = t in active and d > max(v for v, u in FLARE if u == t)
                        original = int(panel[r, c])
                        ink = info['ink'] if ray else original
                        if ray or original != info['background']:
                            if not occupied[r, c]:
                                expected[r, c] = ink
                            next_occupied[r, c] = True
                occupied = next_occupied
            assert np.array_equal(expected, out)
            assert not np.array_equal(out, _render(panels, infos, raw=True)[0])
            if split == 'train':
                suppression |= not np.array_equal(out, _render(panels, infos, gate=False)[0])
                train_counts.add(len(panels))
                train_directions.update(i['direction'] for i in infos)
                train_programs.add(_program(a))
            else:
                assert len(panels) in train_counts
                assert all(i['direction'] in train_directions for i in infos)
                assert _program(a) not in train_programs
                assert not np.array_equal(out, _render(panels, infos, gate=False)[0])
    assert suppression and redundancy
    assert _lane_evidence(puzzle['train'])
    assert _orders_fit(puzzle['train']) == [(0, 1, 2)]


def self_test():
    shapes, counts, directions, backgrounds, programs, structures = set(), set(), set(), set(), set(), set()
    puzzle_keys = set()
    pairs = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        q = generate_puzzle(seed)
        key = tuple(_signature(x[k]) for s in ('train', 'test') for x in p[s]
                    for k in ('input', 'output'))
        assert key == tuple(_signature(x[k]) for s in ('train', 'test') for x in q[s]
                            for k in ('input', 'output'))
        puzzle_keys.add(key)
        for split in ('train', 'test'):
            for pair in p[split]:
                a = pair['input']
                panels, infos, axis, _ = _analyze(a)
                shapes.add(a.shape)
                counts.add(len(panels))
                directions.update(i['direction'] for i in infos)
                backgrounds.add(infos[0]['background'])
                programs.add(_program(a))
                roles = {}
                structure = tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)
                structures.add((a.shape, structure))
                pairs += 1
        # These operations preserve the rule while moving beyond the sampled
        # puzzle's palette and strip orientation. No generation is involved.
        probe = p['test'][0]
        assert np.array_equal(solve(9-probe['input']), 9-probe['output'])
        assert np.array_equal(solve(probe['input'].T), probe['output'].T)
    assert len(puzzle_keys) == 42 and counts == {2, 3}
    assert directions == {0, 1, 2, 3} and backgrounds == set(range(10))
    assert len(shapes) >= 35 and len(programs) >= 120
    assert len(structures) >= .95*pairs
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(7100+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, pairs=pairs, input_shapes=len(shapes),
                active_programs=len(programs), normalized_structures=len(structures))


if __name__ == '__main__':
    print(self_test())
