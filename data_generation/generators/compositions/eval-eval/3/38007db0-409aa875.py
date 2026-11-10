"""Anomaly echoes: master-relative arrow differences share a target template.

The solver reads only pixels. Generation builds signals and a compatible
template directly; it does not import or call either parent generator.
"""
from collections import Counter
from functools import lru_cache
import numpy as np


DISTANCE = 3
SINGLE = 2
COLLISION = 4
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1))
INPUT_COLORS = tuple(c for c in range(10) if c not in (SINGLE, COLLISION))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(points):
    """Eight-connected components, including diagonally joined V wings."""
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            near = {(r + dr, c + dc) for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)} & remaining
            remaining -= near
            group |= near
            todo.extend(sorted(near))
        result.append(group)
    return result


def _arrow(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return (r, c, dr, dc, tuple(sorted((r + a, c + b) for a, b in offsets)))


def _endpoint(arrow, distance=DISTANCE):
    r, c, dr, dc, _ = arrow
    return r + distance * dr, c + distance * dc


def _decode_arrow(cells):
    assert len(cells) == 3, "Every difference component must be a triomino"
    matches = [_arrow(p, d) for p in sorted(cells) for d in DIRECTIONS
               if set(_arrow(p, d)[4]) == cells]
    assert len(matches) == 1, "Difference must have one arrow interpretation"
    return matches[0]


def _assemble(rows, lattice):
    ph, pw = rows[0][0].shape
    bands, copies = len(rows), len(rows[0])
    a = np.full((bands * (ph + 1) + 1, copies * (pw + 1) + 1),
                lattice, dtype=np.int8)
    for b, row in enumerate(rows):
        assert len(row) == copies
        for c, panel in enumerate(row):
            assert panel.shape == (ph, pw)
            r0, c0 = 1 + b * (ph + 1), 1 + c * (pw + 1)
            a[r0:r0 + ph, c0:c0 + pw] = panel
    return a


def _read(input_grid):
    a = _grid(input_grid)
    lattice = int(a[0, 0])
    rs = np.flatnonzero(np.all(a == lattice, axis=1))
    cs = np.flatnonzero(np.all(a == lattice, axis=0))
    assert len(rs) >= 2 and len(cs) >= 4
    assert rs[0] == cs[0] == 0
    assert rs[-1] == a.shape[0] - 1 and cs[-1] == a.shape[1] - 1
    assert np.all(np.diff(rs) == rs[1]) and np.all(np.diff(cs) == cs[1])
    ph, pw = int(rs[1] - 1), int(cs[1] - 1)
    panels = [[a[r + 1:s, c + 1:d].copy() for c, d in zip(cs, cs[1:])]
              for r, s in zip(rs, rs[1:])]
    assert all(np.all(p != lattice) for row in panels for p in row)
    counts = Counter(_key(p) for row in panels for p in row)
    maximum = max(counts.values())
    winners = [k for k, v in counts.items() if v == maximum]
    assert len(winners) == 1, "The repeated master must be unique"
    master = next(p for row in panels for p in row if _key(p) == winners[0])
    carrier = int(master[0, 0])
    colors = set(map(int, np.unique(a)))
    assert len(colors) == 3 and not colors & {SINGLE, COLLISION}
    motif = next(iter(colors - {lattice, carrier}))
    assert set(map(int, np.unique(master))) == {carrier, motif}
    assert len({carrier, motif, lattice}) == 3
    arrows, signs, band_ids, columns, variants = [], [], [], [], []
    for b, row in enumerate(panels):
        different = [(c, p) for c, p in enumerate(row) if not np.array_equal(p, master)]
        assert len(different) == 1, "One visible exceptional panel per band"
        column, variant = different[0]
        columns.append(column)
        variants.append(variant)
        for p in row:
            assert set(map(int, np.unique(p))) <= {carrier, motif}
            assert np.all(p[[0, -1], :] == carrier)
            assert np.all(p[:, [0, -1]] == carrier)
        cells = set(map(tuple, np.argwhere(variant != master).tolist()))
        for group in _components(cells):
            arrow = _decode_arrow(group)
            polarities = {int(variant[p] == motif) for p in group}
            assert len(polarities) == 1, "A signal is wholly added or wholly removed"
            ep = _endpoint(arrow)
            assert 1 <= ep[0] < ph - 1 and 1 <= ep[1] < pw - 1
            arrows.append(arrow)
            signs.append(1 if next(iter(polarities)) else -1)
            band_ids.append(b)
    objects = _components(set(map(tuple, np.argwhere(master == motif).tolist())))
    owner = {p: i for i, obj in enumerate(objects) for p in obj}
    endpoints = Counter(_endpoint(o) for o in arrows)
    hit_roles = {}
    for p, n in endpoints.items():
        if p in owner:
            hit_roles.setdefault(owner[p], set()).add(n >= 2)
    assert all(len(v) == 1 for v in hit_roles.values()), "Conflicting target roles"
    return dict(master=master, lattice=lattice, carrier=carrier, motif=motif,
                panels=panels, ph=ph, pw=pw, bands=len(panels), copies=len(panels[0]),
                arrows=arrows, signs=signs, band_ids=band_ids, columns=columns,
                variants=variants, objects=objects, owner=owner, endpoints=endpoints)


def _paint(master, arrows, motif, lattice, distance=DISTANCE):
    """Simultaneous template-coordinate endpoint effects."""
    out = master.copy()
    objects = _components(set(map(tuple, np.argwhere(master == motif).tolist())))
    owner = {p: i for i, obj in enumerate(objects) for p in obj}
    assignments = {}
    for p, count in Counter(_endpoint(o, distance) for o in arrows).items():
        assert 0 <= p[0] < master.shape[0] and 0 <= p[1] < master.shape[1]
        color = COLLISION if count >= 2 else SINGLE
        cells = objects[owner[p]] if p in owner else (p,)
        for cell in cells:
            assert cell not in assignments or assignments[cell] == color
            assignments[cell] = color
    for p, color in assignments.items():
        out[p] = color
    return _assemble([[out]], lattice)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """No seed, training labels, generator parameters, or mutable state."""
    info = _read(input_grid)
    return _paint(info['master'], info['arrows'], info['motif'], info['lattice'])


@lru_cache(maxsize=None)
def _candidates(ph, pw):
    result = []
    for r in range(1, ph - 1):
        for c in range(1, pw - 1):
            for d in DIRECTIONS:
                a = _arrow((r, c), d)
                if all(1 <= rr < ph - 1 and 1 <= cc < pw - 1
                       for rr, cc in list(a[4]) + [_endpoint(a)]):
                    result.append(a)
    return tuple(result)


def _apart(a, b):
    return all(max(abs(r - s), abs(c - d)) > 1
               for r, c in a[4] for s, d in b[4])


@lru_cache(maxsize=None)
def _strike_options(ph, pw):
    candidates = _candidates(ph, pw)
    by_endpoint = {}
    for a in candidates:
        by_endpoint.setdefault(_endpoint(a), []).append(a)
    return tuple((target, shooter) for target in candidates for cell in target[4]
                 if cell != target[:2]
                 for shooter in by_endpoint.get(cell, ())
                 if _apart(target, shooter) and _endpoint(target) not in shooter[4])


@lru_cache(maxsize=None)
def _collision_options(ph, pw, count):
    from itertools import combinations
    by_endpoint = {}
    for a in _candidates(ph, pw):
        by_endpoint.setdefault(_endpoint(a), []).append(a)
    return tuple(group for arrows in by_endpoint.values()
                 for group in combinations(arrows, count)
                 if all(_apart(a, b) for a, b in combinations(group, 2)))


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _strike_scene(rng):
    """A balanced difference plus a wing strike and two isolated endpoints."""
    geometries = [(ph, pw) for ph in range(11, 14) for pw in range(6, 9)]
    for gi in rng.permutation(len(geometries)):
        ph, pw = geometries[int(gi)]
        options = _strike_options(ph, pw)
        for oi in rng.permutation(len(options)):
            a, b = options[int(oi)]
            for ci in rng.permutation(len(_candidates(ph, pw))):
                c = _candidates(ph, pw)[int(ci)]
                if not _apart(a, c) or set(c[4]) & set(b[4]):
                    continue
                if _endpoint(c) in {_endpoint(a), _endpoint(b)}:
                    continue
                if _endpoint(c) in set(a[4]) | set(b[4]) | set(c[4]):
                    continue
                if _endpoint(a) in set(b[4]) | set(c[4]):
                    continue
                # A deletion-only isolated endpoint is a direct ablation witness.
                if _endpoint(a) in (b[:2], c[:2]):
                    continue
                return ph, pw, [[a, c], [b]], [-1, 1, 1], {_endpoint(a), _endpoint(c)}, a
    raise AssertionError('No strike teaching scene exists')


def _collision_scene(rng):
    geometries = [(8, 8, 3)] + [(ph, pw, 2) for ph in range(11, 14) for pw in range(6, 9)]
    feasible = [(ph, pw, bands) for ph, pw, bands in geometries
                if _collision_options(ph, pw, bands)]
    ph, pw, bands = _choose(rng, feasible)
    arrows = list(_choose(rng, _collision_options(ph, pw, bands)))
    rng.shuffle(arrows)
    signs = list(map(int, rng.choice((-1, 1), size=bands)))
    signs[0], signs[-1] = -1, 1
    return ph, pw, [[a] for a in arrows], signs


def _ordinary_scene(rng, test):
    """Independent layouts; queries have one band with three or four heads."""
    cross_band_event = bool(test and rng.random() < .35)
    if test:
        # Each demonstration has at most two heads per exceptional panel.
        # Queries require componentwise generalization to a larger delta group.
        geometries = [(ph, pw, bands) for ph in range(7, 28) for pw in range(5, 9)
                      for bands in range(1, min(3, 29 // (ph + 1)) + 1)
                      if _candidates(ph, pw)]
        if cross_band_event:
            geometries = [(ph, pw, bands) for ph, pw, bands in geometries
                          if bands >= 2 and _collision_options(ph, pw, 2)]
    else:
        geometries = [(ph, pw, bands) for ph in range(7, 22) for pw in range(5, 9)
                      for bands in range(1, min(3, 29 // (ph + 1)) + 1)
                      if _candidates(ph, pw)]
    for gi in rng.permutation(len(geometries)):
        ph, pw, bands = geometries[int(gi)]
        sizes = [int(rng.integers(1, 3)) for _ in range(bands)]
        if test:
            sizes[int(rng.integers(bands))] = int(rng.integers(3, 5))
        candidates = _candidates(ph, pw)
        for _ in range(32):
            groups, used = [[] for _ in range(bands)], set()
            if cross_band_event:
                event = _choose(rng, _collision_options(ph, pw, 2))
                bins = list(map(int, rng.choice(bands, 2, replace=False)))
                for b, a in zip(bins, event):
                    groups[b].append(a)
                    used.update(a[4])
            for b, size in enumerate(sizes):
                group = groups[b]
                if len(group) == size:
                    continue
                for ci in rng.permutation(len(candidates)):
                    a = candidates[int(ci)]
                    if set(a[4]) & used or any(not _apart(a, b) for b in group):
                        continue
                    group.append(a)
                    used.update(a[4])
                    if len(group) == size:
                        break
                if len(group) != size:
                    break
            if all(len(group) == size for group, size in zip(groups, sizes)):
                signs = list(map(int, rng.choice((-1, 1), size=sum(sizes))))
                return ph, pw, groups, signs
    raise AssertionError('No bounded ordinary layout exists')


def _grow(rng, mask, seed_cells, forbidden, target_size):
    """Grow a readable connected silhouette without changing signal polarity."""
    cells = set(seed_cells)
    for p in cells:
        mask[p] = True
    while len(cells) < target_size:
        options = sorted({(r + dr, c + dc) for r, c in cells
                          for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                          if 1 <= r + dr < mask.shape[0] - 1
                          and 1 <= c + dc < mask.shape[1] - 1
                          and (r + dr, c + dc) not in forbidden
                          and (r + dr, c + dc) not in cells})
        if not options:
            break
        p = _choose(rng, options)
        cells.add(p)
        mask[p] = True
    return cells


def _template(rng, ph, pw, groups, signs, mode, protected, target_arrow):
    arrows = [a for group in groups for a in group]
    required_ink = {p for a, sign in zip(arrows, signs) if sign < 0 for p in a[4]}
    forbidden = {p for a, sign in zip(arrows, signs) if sign > 0 for p in a[4]} | set(protected)
    assert not required_ink & forbidden
    mask = np.zeros((ph, pw), dtype=bool)
    for p in required_ink:
        mask[p] = True
    if mode == 'strike':
        cells = _grow(rng, mask, target_arrow[4], forbidden, int(rng.integers(4, 9)))
        if len(cells) < 4:
            return None
    endpoints = Counter(_endpoint(a) for a in arrows)
    if mode == 'collision':
        ep = next(iter(endpoints))
        if rng.random() < .65:
            _grow(rng, mask, {ep}, forbidden, int(rng.integers(2, 7)))
        else:
            forbidden.add(ep)
    # Passive objects are structured templates, not independently solved noise.
    # Different line, corner, ring, cross, and grown shapes supply varied targets.
    shapes = (((0, 0),), ((0, 0), (0, 1), (0, 2)),
              ((0, 0), (1, 0), (2, 0)), ((0, 0), (1, 0), (1, 1)),
              ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
              ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)))
    for _ in range(int(rng.integers(1, 5))):
        for _ in range(25):
            r, c = int(rng.integers(1, ph - 1)), int(rng.integers(1, pw - 1))
            cells = {(r + dr, c + dc) for dr, dc in _choose(rng, shapes)}
            if any(not (1 <= rr < ph - 1 and 1 <= cc < pw - 1)
                   or (rr, cc) in forbidden for rr, cc in cells):
                continue
            # Maintain space when possible; overlapping source silhouettes can
            # also become one larger target, which the pixel solver handles.
            for p in cells:
                mask[p] = True
            if rng.random() < .3:
                _grow(rng, mask, cells, forbidden, int(rng.integers(len(cells), len(cells) + 5)))
            break
    if not mask.any():
        available = [(r, c) for r in range(1, ph - 1) for c in range(1, pw - 1)
                     if (r, c) not in forbidden]
        mask[_choose(rng, available)] = True
    # Object-level conflicting roles are deliberately outside this family.
    objects = _components(set(map(tuple, np.argwhere(mask).tolist())))
    owner = {p: i for i, obj in enumerate(objects) for p in obj}
    roles = {}
    for p, n in endpoints.items():
        if p in owner:
            roles.setdefault(owner[p], set()).add(n >= 2)
    if any(len(v) > 1 for v in roles.values()):
        return None
    return mask


def _pair(rng, mode, palette, test=False):
    lattice, carrier, motif = palette
    for _ in range(80):
        protected, target_arrow = set(), None
        if mode == 'strike':
            ph, pw, groups, signs, protected, target_arrow = _strike_scene(rng)
        elif mode == 'collision':
            ph, pw, groups, signs = _collision_scene(rng)
        else:
            ph, pw, groups, signs = _ordinary_scene(rng, test)
        mask = _template(rng, ph, pw, groups, signs, mode, protected, target_arrow)
        if mask is None:
            continue
        master = np.full((ph, pw), carrier, dtype=np.int8)
        master[mask] = motif
        copies = int(rng.integers(3, 29 // (pw + 1) + 1))
        columns = list(map(int, rng.integers(copies, size=len(groups))))
        # Teach that the selected column is not fixed, with balanced geometry.
        if mode == 'strike' and len(set(columns)) == 1:
            columns[-1] = (columns[-1] + int(rng.integers(1, copies))) % copies
        rows = []
        for b, group in enumerate(groups):
            variant = master.copy()
            for a in group:
                for p in a[4]:
                    variant[p] = carrier if master[p] == motif else motif
            row = [master.copy() for _ in range(copies)]
            row[columns[b]] = variant
            rows.append(row)
        inp = _assemble(rows, lattice)
        arrows = [a for group in groups for a in group]
        out = _paint(master, arrows, motif, lattice)
        info = _read(inp)
        if mode == 'strike':
            positive = [a for a, s in zip(info['arrows'], info['signs']) if s > 0]
            if np.array_equal(out, _paint(master, positive, motif, lattice)):
                continue
            if np.array_equal(out, _paint(master, arrows, motif, lattice, 0)):
                continue
        return {'input': inp, 'output': out}
    raise AssertionError('Bounded template construction exhausted')


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name + ' is out of range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count('num_train', num_train, 2, 10)
    ne = _count('num_test', num_test, 1, 4)
    if nt <= ne:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed))
    shared_palette = bool(rng.random() < .3)
    palette = tuple(map(int, rng.choice(INPUT_COLORS, 3, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    train_modes = ['strike', 'collision'] + ['ordinary'] * (nt - 2)
    rng.shuffle(train_modes)
    for split, modes in (('train', train_modes), ('test', ['ordinary'] * ne)):
        for mode in modes:
            for _ in range(80):
                if not shared_palette:
                    palette = tuple(map(int, rng.choice(INPUT_COLORS, 3, replace=False)))
                pair = _pair(rng, mode, palette, test=split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('Distinct input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    info = _read(pair['input'])
    endpoints, owner = info['endpoints'], info['owner']
    info['balanced'] = any(np.count_nonzero(v == info['motif']) ==
                           np.count_nonzero(info['master'] == info['motif'])
                           for v in info['variants'])
    info['struck'] = {owner[p] for p in endpoints if p in owner}
    info['cross_band_collision'] = any(len({b for a, b in zip(info['arrows'], info['band_ids'])
                                          if _endpoint(a) == p}) > 1
                                       for p, n in endpoints.items() if n >= 2)
    return info


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, teaching = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _features(pair)
            assert out.shape == (info['ph'] + 2, info['pw'] + 2)
            assert np.array_equal(solve(inp), out)
            # Every template object hit is recolored in full; all other pixels
            # are copied from the template, never from the exceptional panel.
            canvas = out[1:-1, 1:-1]
            painted = set()
            for p, n in info['endpoints'].items():
                cells = info['objects'][info['owner'][p]] if p in info['owner'] else {p}
                for q in cells:
                    assert canvas[q] == (COLLISION if n >= 2 else SINGLE)
                painted.update(cells)
            unchanged = np.ones(info['master'].shape, dtype=bool)
            for p in painted:
                unchanged[p] = False
            assert np.array_equal(canvas[unchanged], info['master'][unchanged])
            assert np.all(out[[0, -1], :] == info['lattice'])
            assert np.all(out[:, [0, -1]] == info['lattice'])
            if split == 'train':
                teaching.append(info)
    assert any(t['balanced'] for t in teaching), 'Teach full bitmap comparison'
    assert any(t['cross_band_collision'] for t in teaching), 'Teach shared coordinates'
    assert any(n == 1 and p not in t['owner'] for t in teaching
               for p, n in t['endpoints'].items()), 'Teach isolated three-step impacts'
    assert any(len(t['objects'][i]) >= 4 for t in teaching for i in t['struck']), \
        'Teach whole-object targeting beyond arrowheads'
    assert any(len(set(t['columns'])) > 1 for t in teaching), 'Teach variable selection columns'
    negative_witness = False
    vector_witness = False
    bitmap_witness = False
    for t in teaching:
        expected = _paint(t['master'], t['arrows'], t['motif'], t['lattice'])
        positive = [a for a, sign in zip(t['arrows'], t['signs']) if sign > 0]
        # An ablated event set can itself assign conflicting roles. Only valid
        # ablations count as our explicit counterexamples; the teachers suffice.
        try:
            negative_witness |= not np.array_equal(expected,
                                                  _paint(t['master'], positive, t['motif'], t['lattice']))
        except AssertionError:
            pass
        vector_witness |= not np.array_equal(expected,
                                            _paint(t['master'], t['arrows'], t['motif'], t['lattice'], 0))
        balanced_bands = {b for b, v in enumerate(t['variants'])
                          if np.count_nonzero(v == t['motif']) ==
                          np.count_nonzero(t['master'] == t['motif'])}
        count_only = [a for a, b in zip(t['arrows'], t['band_ids']) if b not in balanced_bands]
        try:
            bitmap_witness |= not np.array_equal(expected,
                                                _paint(t['master'], count_only, t['motif'], t['lattice']))
        except AssertionError:
            pass
    assert negative_witness, 'Deleting a head must still launch a visible echo'
    assert vector_witness, 'Projection must differ from marking tips in place'
    assert bitmap_witness, 'Count-based exception detection must change an answer'


def _regressions():
    """Small hand-computed pixel fixtures, separate from generated labels."""
    def fixture(master, groups, lattice):
        rows = []
        motif = int(master[groups[0][0][4][0]])
        carrier = int(master[0, 0])
        for b, group in enumerate(groups):
            variant = master.copy()
            for a in group:
                for p in a[4]:
                    variant[p] = carrier if master[p] == motif else motif
            row = [master.copy() for _ in range(3)]
            row[b % 3] = variant
            rows.append(row)
        return _assemble(rows, lattice)

    # A removed N head and an added S head form a balanced exception. An
    # additional N head hits a wing of a larger template object. Its two
    # distinct hit cells receive SINGLE, even though they share one object.
    master = np.zeros((11, 8), dtype=np.int8)
    target = {(4, 3), (5, 2), (5, 4), (4, 4), (3, 4), (3, 3), (5, 5)}
    passive = {(8, 6), (9, 6)}
    for p in target | passive:
        master[p] = 3
    a = _arrow((4, 3), (-1, 0))
    b = _arrow((8, 2), (-1, 0))
    c = _arrow((2, 5), (1, 0))
    inp = fixture(master, [[a, c], [b]], 9)
    expected = master.copy()
    for p in target:
        expected[p] = SINGLE
    expected[1, 3] = SINGLE
    assert np.array_equal(solve(inp), _assemble([[expected]], 9))
    # No path is drawn: (2,3) lies between the removed tip and its endpoint.
    assert solve(inp)[3, 4] == 0
    assert all(solve(inp)[r + 1, col + 1] == 3 for r, col in passive)

    # Three distinct directions in separate bands converge on one target.
    master = np.zeros((8, 8), dtype=np.int8)
    arrows = [_arrow((5, 2), (-1, 0)), _arrow((2, 5), (0, -1)),
              _arrow((5, 5), (-1, -1))]
    target = {(2, 2), (2, 3), (3, 2)}
    for p in set(arrows[0][4]) | target:
        master[p] = 3
    inp = fixture(master, [[a] for a in arrows], 9)
    expected = master.copy()
    for p in target:
        expected[p] = COLLISION
    assert np.array_equal(solve(inp), _assemble([[expected]], 9))
    # Negative arrows are restored in the output template, not erased there.
    assert all(solve(inp)[r + 1, col + 1] == 3 for r, col in arrows[0][4])


def self_test() -> dict:
    _regressions()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures = {tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test')
                        for p in task[s]) for task in puzzles}
    assert len(signatures) == 42
    features = [_features(p) for task in puzzles for s in ('train', 'test') for p in task[s]]
    shapes = {(t['ph'], t['pw'], t['bands'], t['copies']) for t in features}
    directions = {(a[2], a[3]) for t in features for a in t['arrows']}
    templates = {_key(t['master'] == t['motif']) for t in features}
    assert len(shapes) >= 25 and len(templates) >= 100
    assert directions == set(DIRECTIONS)
    assert {t['lattice'] for t in features} == set(INPUT_COLORS)
    assert any(t['bands'] == 1 for t in features) and any(t['bands'] == 3 for t in features)
    assert any(t['copies'] == 4 for t in features)
    assert any(max(t['endpoints'].values()) >= 3 for t in features)
    assert any(len(t['struck']) > 1 for t in features)
    assert any(t['bands'] > 1 and len(set(t['columns'])) == 1 for t in features)
    first, second = generate_puzzle(8128, 2, 1), generate_puzzle(8128, 2, 1)
    assert all(_key(a[k]) == _key(b[k]) for s in first for a, b in zip(first[s], second[s])
               for k in ('input', 'output'))
    # Coordinate registration is invariant to band permutations and relocating
    # exceptional panels. These metamorphic checks use only decoded input data.
    base = puzzles[0]['train'][0]
    t = _read(base['input'])
    moved = _assemble(t['panels'][::-1], t['lattice'])
    assert np.array_equal(solve(moved), base['output'])
    rows = [row[1:] + row[:1] for row in t['panels']]
    assert np.array_equal(solve(_assemble(rows, t['lattice'])), base['output'])
    turned = _assemble([[np.rot90(p, 2) for p in row] for row in t['panels']], t['lattice'])
    assert np.array_equal(solve(turned), np.rot90(base['output'], 2))
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in puzzles[0]}
    damaged['test'][0]['output'][0, 0] = SINGLE
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('Altered output accepted')
    # Removing one exceptional panel is a structural error, never completion.
    inp = base['input'].copy()
    col = t['columns'][0]
    c0 = 1 + col * (t['pw'] + 1)
    inp[1:1 + t['ph'], c0:c0 + t['pw']] = t['master']
    try:
        solve(inp)
    except AssertionError:
        pass
    else:
        raise AssertionError('Missing exception accepted')
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            task = generate_puzzle(500 + nt * 10 + ne, nt, ne)
            assert len(task['train']) == nt and len(task['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    return dict(seeds=42, distinct_puzzles=42, geometries=len(shapes),
                distinct_templates=len(templates), directions=len(directions))


if __name__ == '__main__':
    print(self_test())
