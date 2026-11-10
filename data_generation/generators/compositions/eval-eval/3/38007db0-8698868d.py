"""Cavity-addressed patch exchange: infer, route, and transplant binary edits.

The rule is fixed. Sampling policies affect examples, never interpretation.
Only Python's standard library and NumPy are used.
"""

from collections import Counter
import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask, diagonal=False):
    """Return four-components (or eight-components for diagnostic tests)."""
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    steps = _STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _STEPS
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        group, pending = [start], [start]
        while pending:
            r, c = pending.pop()
            for dr, dc in steps:
                point = r + dr, c + dc
                if point in unseen:
                    unseen.remove(point)
                    group.append(point)
                    pending.append(point)
        groups.append(group)
    return groups


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'bad dimensions'
    assert np.issubdtype(a.dtype, np.integer), 'noninteger cells'
    assert np.all((0 <= a) & (a <= 9)), 'bad colors'
    return a.astype(np.int8, copy=False)


def _assemble(rows, lattice):
    ph, pw = rows[0][0].shape
    out = np.full((len(rows) * (ph + 1) + 1,
                   len(rows[0]) * (pw + 1) + 1), lattice, dtype=np.int8)
    for r, row in enumerate(rows):
        assert len(row) == len(rows[0])
        for c, panel in enumerate(row):
            assert panel.shape == (ph, pw)
            r0, c0 = 1 + r * (ph + 1), 1 + c * (pw + 1)
            out[r0:r0 + ph, c0:c0 + pw] = panel
    return out


def _perimeter(panel):
    return np.concatenate((panel[0], panel[-1], panel[1:-1, 0], panel[1:-1, -1]))


def _toggle(panel, patch, wall, void):
    result = panel.copy()
    result[patch] = np.where(panel[patch] == wall, void, wall)
    return result


def _analyze_input(input_grid):
    """Recover all roles, templates, edit masks, and addresses from pixels."""
    a = _grid(input_grid)
    lattice = int(a[0, 0])
    rs = np.flatnonzero(np.all(a == lattice, axis=1))
    cs = np.flatnonzero(np.all(a == lattice, axis=0))
    assert len(rs) >= 3 and len(cs) >= 4, 'need bands and at least three copies'
    assert rs[0] == cs[0] == 0
    assert rs[-1] == a.shape[0] - 1 and cs[-1] == a.shape[1] - 1
    assert np.all(np.diff(rs) == rs[1]) and np.all(np.diff(cs) == cs[1])
    ph, pw = int(rs[1] - 1), int(cs[1] - 1)
    assert ph >= 3 and pw >= 3
    colors = set(map(int, np.unique(a)))
    assert len(colors) == 3, 'one lattice color and two panel colors'
    templates, patches, columns, counts, areas, panels = [], [], [], [], [], []
    wall = void = None
    for r0, r1 in zip(rs, rs[1:]):
        row = [a[r0 + 1:r1, c0 + 1:c1].copy() for c0, c1 in zip(cs, cs[1:])]
        assert all(np.all(p != lattice) for p in row)
        freq = Counter(_key(p) for p in row)
        repeated = [key for key, n in freq.items() if n == len(row) - 1]
        assert len(freq) == 2 and len(repeated) == 1, 'one visible bitmap exception per band'
        master = next(p for p in row if _key(p) == repeated[0])
        w = int(master[0, 0])
        if wall is None:
            wall = w
            void = next(iter(colors - {lattice, wall}))
        assert w == wall and wall != lattice
        assert all(np.all(_perimeter(p) == wall) for p in row), 'edits cannot touch the rim'
        assert all(set(map(int, np.unique(p))) <= {wall, void} for p in row)
        col = next(i for i, p in enumerate(row) if not np.array_equal(p, master))
        patch = row[col] != master
        groups = _components(master == void)
        # The complete wall rim makes every void component an enclosed cavity.
        assert groups, 'zero-cavity addresses cannot receive a nonempty patch'
        templates.append(master)
        patches.append(patch)
        columns.append(col)
        counts.append(len(groups))
        areas.append(int(np.count_nonzero(master == void)))
        panels.append(row)
    assert len(set(counts)) == len(counts), 'cavity addresses must be unique'
    sizes = list(map(lambda p: int(p.sum()), patches))
    assert sorted(sizes) == sorted(counts), 'one incoming patch for every address'
    routes = [counts.index(k) for k in sizes]
    return dict(lattice=lattice, wall=wall, void=void, ph=ph, pw=pw,
                bands=len(templates), copies=len(cs) - 1, templates=templates,
                patches=patches, columns=columns, counts=counts, areas=areas,
                routes=routes, panels=panels)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the current input; never use generation or examples."""
    info = _analyze_input(input_grid)
    result = [p.copy() for p in info['templates']]
    for source, target in enumerate(info['routes']):
        result[target] = _toggle(info['templates'][target], info['patches'][source],
                                 info['wall'], info['void'])
    return _assemble([[p] for p in result], info['lattice'])


def _symmetries(mask):
    variants = [mask[::-1], mask[:, ::-1], mask[::-1, ::-1]]
    if mask.shape[0] == mask.shape[1]:
        variants.extend((mask.T, mask.T[::-1], mask.T[:, ::-1], mask.T[::-1, ::-1]))
    return variants


def _asymmetric(mask):
    return all(not np.array_equal(mask, other) for other in _symmetries(mask))


def _random_transform(rng, masks):
    turns = int(rng.integers(4)) if masks[0].shape[0] == masks[0].shape[1] else 2 * int(rng.integers(2))
    flip = bool(rng.integers(2))
    return [(np.rot90(m, turns)[:, ::-1] if flip else np.rot90(m, turns)).copy() for m in masks]


def _hole_mask(rng, h, w, count):
    """Construct arbitrary connected growth, isolated holes, or rectangle cuts."""
    assert 1 <= count <= (h * w + 1) // 2
    mode = int(rng.integers(4))
    if count == 1 and mode == 0:
        return np.ones((h, w), dtype=bool)
    if mode == 1:
        boxes = [(0, h, 0, w)]
        while len(boxes) < count:
            cuts = []
            for i, (r0, r1, c0, c1) in enumerate(boxes):
                cuts.extend((i, 0, p) for p in range(r0 + 1, r1 - 1))
                cuts.extend((i, 1, p) for p in range(c0 + 1, c1 - 1))
            if not cuts:
                break
            i, axis, p = _choose(rng, cuts)
            r0, r1, c0, c1 = boxes.pop(i)
            boxes.extend([(r0, p, c0, c1), (p + 1, r1, c0, c1)] if axis == 0
                         else [(r0, r1, c0, p), (r0, r1, p + 1, c1)])
        if len(boxes) == count:
            mask = np.zeros((h, w), dtype=bool)
            for r0, r1, c0, c1 in boxes:
                # Independent shrinking makes cavity areas and offsets unequal.
                if rng.random() < .5:
                    r0 = int(rng.integers(r0, r1))
                    r1 = int(rng.integers(r0 + 1, r1 + 1))
                    c0 = int(rng.integers(c0, c1))
                    c1 = int(rng.integers(c0 + 1, c1 + 1))
                mask[r0:r1, c0:c1] = True
            return _random_transform(rng, [mask])[0]
    # Greedy independent sets with a guaranteed checkerboard fallback.
    chosen = []
    for _ in range(12):
        chosen = []
        for flat in rng.permutation(h * w):
            r, c = divmod(int(flat), w)
            if all(abs(r - rr) + abs(c - cc) != 1 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    break
        if len(chosen) == count:
            break
    if len(chosen) < count:
        cells = [(r, c) for r in range(h) for c in range(w) if (r + c) % 2 == 0]
        chosen = [cells[int(i)] for i in rng.permutation(len(cells))[:count]]
    labels = np.zeros((h, w), dtype=int)
    for label, cell in enumerate(chosen, 1):
        labels[cell] = label
    target = count if mode == 3 else int(rng.integers(count, h * w + 1))
    while np.count_nonzero(labels) < target:
        candidates = []
        for r in range(h):
            for c in range(w):
                if labels[r, c]:
                    continue
                touching = {int(labels[r + dr, c + dc]) for dr, dc in _STEPS
                            if 0 <= r + dr < h and 0 <= c + dc < w and labels[r + dr, c + dc]}
                if len(touching) == 1:
                    candidates.append((r, c, next(iter(touching))))
        if not candidates:
            break
        r, c, label = _choose(rng, candidates)
        labels[r, c] = label
    return labels > 0


def _patch(rng, h, w, count):
    """Mix connected strokes, lines, and scattered differences."""
    mask = np.zeros((h, w), dtype=bool)
    mode = int(rng.integers(3))
    if mode == 0:
        mask.flat[rng.choice(h * w, count, replace=False)] = True
    elif mode == 1 and count <= max(h, w):
        if count <= w and (count > h or rng.random() < .5):
            r, c = int(rng.integers(h)), int(rng.integers(w - count + 1))
            mask[r, c:c + count] = True
        else:
            r, c = int(rng.integers(h - count + 1)), int(rng.integers(w))
            mask[r:r + count, c] = True
    else:
        mask[int(rng.integers(h)), int(rng.integers(w))] = True
        while int(mask.sum()) < count:
            edge = [(r, c) for r in range(h) for c in range(w) if not mask[r, c]
                    and any(0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc]
                            for dr, dc in _STEPS)]
            mask[_choose(rng, edge)] = True
    return mask


def _capacity(ph, pw):
    return min(7, ((ph - 2) * (pw - 2) + 1) // 2)


_GEOMETRIES = tuple((ph, pw, bands, copies)
                    for ph in range(4, 10) for pw in range(4, 9)
                    for bands in range(2, 6) for copies in range(3, 6)
                    if bands * (ph + 1) + 1 <= 30
                    and copies * (pw + 1) + 1 <= 30
                    and bands <= _capacity(ph, pw))


def _make_pair(rng, geometry, palette, witness=False, long_cycle=False):
    ph, pw, bands, copies = geometry
    h, w = ph - 2, pw - 2
    lattice, wall, void = palette
    cap = _capacity(ph, pw)
    if witness:
        assert bands == 3 and h >= 3 and w >= 3
        counts = [1, 2, int(rng.integers(4, cap + 1))]
        one = np.ones((h, w), dtype=bool)
        one[0, 0] = one[1, 0] = False  # A notch; the large cavity stays connected.
        two = np.zeros((h, w), dtype=bool)
        two[0, 0] = two[1, 1] = two[1, 2] = True  # Two 4-components, one 8-component.
        balanced = np.zeros((h, w), dtype=bool)
        balanced[0, 0] = balanced[0, 1] = True
        one, two, balanced = _random_transform(rng, [one, two, balanced])
        holes = [one, two, _hole_mask(rng, h, w, counts[2])]
        routes = [1, 0, 2]  # A swap and a self-addressed patch.
        patches = [balanced, _patch(rng, h, w, 1), _patch(rng, h, w, counts[2])]
    else:
        counts = list(map(int, rng.choice(np.arange(1, cap + 1), bands, replace=False)))
        holes = [_hole_mask(rng, h, w, k) for k in counts]
        if long_cycle:
            order = list(map(int, rng.permutation(bands)))
            routes = [0] * bands
            for source, target in zip(order, order[1:] + order[:1]):
                routes[source] = target
        else:
            routes = list(map(int, rng.permutation(bands)))
        patches = [_patch(rng, h, w, counts[target]) for target in routes]
    # A common shuffle permutes the directed patch network and the templates.
    order = list(map(int, rng.permutation(bands)))
    if witness and order in ([0, 1, 2], [2, 1, 0]):
        order = [1, 0, 2] if rng.random() < .5 else [0, 2, 1]
    inverse = {old: new for new, old in enumerate(order)}
    holes = [holes[i] for i in order]
    patches = [patches[i] for i in order]
    routes = [inverse[routes[i]] for i in order]
    masters = []
    full_patches = []
    for mask, patch in zip(holes, patches):
        master = np.full((ph, pw), wall, dtype=np.int8)
        master[1:-1, 1:-1][mask] = void
        masters.append(master)
        full_patches.append(np.pad(patch, 1))
    columns = list(map(int, rng.integers(copies, size=bands)))
    if witness and len(set(columns)) == 1:
        columns[1] = (columns[1] + 1) % copies
    rows = []
    for master, patch, col in zip(masters, full_patches, columns):
        row = [master.copy() for _ in range(copies)]
        row[col] = _toggle(master, patch, wall, void)
        rows.append(row)
    result = [m.copy() for m in masters]
    for source, target in enumerate(routes):
        result[target] = _toggle(masters[target], full_patches[source], wall, void)
    return {'input': _assemble(rows, lattice),
            'output': _assemble([[p] for p in result], lattice)}


def _check_counts(num_train, num_test):
    for name, n, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not lo <= n <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    shift = _choose(rng, ('population', 'geometry'))
    shared_palette = rng.random() < .2
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    puzzle, used, training_shapes = {'train': [], 'test': []}, set(), set()
    for split, number in (('train', int(num_train)), ('test', int(num_test))):
        for i in range(number):
            witness = split == 'train' and i == 0
            choices = list(_GEOMETRIES)
            if split == 'train':
                choices = [g for g in choices if g[2] <= 3]
                if shift == 'geometry':
                    choices = [g for g in choices if g[0] == g[1] and g[0] >= 5]
                if witness:
                    choices = [g for g in choices if g[2] == 3 and min(g[:2]) >= 5]
                elif i == 1:
                    choices = [g for g in choices if g[2] == 3 and g[:2] not in training_shapes]
            elif shift == 'population':
                choices = [g for g in choices if g[2] >= 4]
                # Sample the population size before geometry, so five-band
                # scenes are not drowned out by the many four-band shapes.
                target_bands = int(rng.integers(4, 6))
                choices = [g for g in choices if g[2] == target_bands]
            else:
                choices = [g for g in choices if g[0] != g[1]]
            geometry = _choose(rng, choices)
            if not shared_palette:
                palette = tuple(map(int, rng.choice(10, 3, replace=False)))
            # A collision is overwhelmingly unlikely. A bounded recoloring
            # fallback still guarantees uniqueness (720 role assignments).
            pair = _make_pair(rng, geometry, palette, witness,
                              long_cycle=(split == 'test' or i == 1))
            if _key(pair['input']) in used:
                original = pair
                found = False
                for a in range(10):
                    if found:
                        break
                    for b in range(10):
                        if found:
                            break
                        for c in range(10):
                            if len({a, b, c}) < 3:
                                continue
                            lookup = np.arange(10, dtype=np.int8)
                            lookup[list(palette)] = (a, b, c)
                            candidate = {k: lookup[v] for k, v in original.items()}
                            if _key(candidate['input']) not in used:
                                pair, found = candidate, True
                                break
                assert found
            used.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                training_shapes.add(geometry[:2])
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    masters, patches = info['templates'], info['patches']
    wall, void = info['wall'], info['void']
    counts, areas = info['counts'], info['areas']
    balanced = any(np.count_nonzero(p & (m == wall)) == np.count_nonzero(p & (m == void))
                   for p, m in zip(patches, masters))
    reversed_effect = any(np.any(patches[s] & (masters[s] != masters[t]))
                          for s, t in enumerate(info['routes']) if s != t)
    return dict(area_inversion=any(k < l and areas[i] > areas[j]
                                  for i, k in enumerate(counts) for j, l in enumerate(counts)),
                connectivity=any(len(_components(m == void, True)) < k
                                 for m, k in zip(masters, counts)),
                edited_topology=any(len(_components(row[col] == void)) != k
                                    for row, col, k in zip(info['panels'], info['columns'], counts)),
                balanced=balanced, reversed_effect=reversed_effect,
                orientation=any(_asymmetric(p[1:-1, 1:-1]) for p in patches),
                displaced=any(s != t for s, t in enumerate(info['routes'])),
                self_address=any(s == t for s, t in enumerate(info['routes'])),
                varied_columns=len(set(info['columns'])) > 1,
                nonconsecutive=max(counts) - min(counts) + 1 > len(counts))


def validate_generated_puzzle(puzzle):
    """Check actual grids, routed edits, demonstrations, and split challenges."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(a)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info = _analyze_input(a)
            ph, pw = info['ph'], info['pw']
            assert out.shape == (a.shape[0], pw + 2)
            assert np.all(out[:, (0, -1)] == info['lattice'])
            assert np.all(out[::ph + 1] == info['lattice'])
            for source, target in enumerate(info['routes']):
                r = 1 + target * (ph + 1)
                result = out[r:r + ph, 1:pw + 1]
                master, patch = info['templates'][target], info['patches'][source]
                assert np.array_equal(result != master, patch), 'wrong patch destination or geometry'
                assert np.array_equal(_toggle(result, patch, info['wall'], info['void']), master)
                assert np.array_equal(_perimeter(result), _perimeter(master))
            assert np.array_equal(solve(a), out), 'input-only solver disagrees'
            analyses[split].append(info)
    train_features = [_features(a) for a in analyses['train']]
    for name in ('area_inversion', 'connectivity', 'edited_topology', 'balanced', 'reversed_effect',
                 'orientation', 'displaced', 'self_address', 'varied_columns', 'nonconsecutive'):
        assert any(f[name] for f in train_features), 'missing training evidence: ' + name
    assert any(a['bands'] >= 3 and all(s != t for s, t in enumerate(a['routes']))
               for a in analyses['train']), 'demonstrate routing beyond a two-way swap'
    assert len({(a['ph'], a['pw']) for a in analyses['train']}) >= 2
    return None


def _puzzle_key(p):
    return tuple(tuple((_key(pair['input']), _key(pair['output'])) for pair in p[s])
                 for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed variation audit, interface tests, and negative checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(generate_puzzle(519115241)) == _puzzle_key(generate_puzzle(519115241))
    geometries, templates, cavity_shapes, palettes, count_sets, patch_shapes = set(), set(), set(), set(), set(), set()
    bands, copies, colors, areas = set(), set(), set(), set()
    population_shifts = geometry_shifts = signed_transfers = 0
    pairs = 0
    for p in puzzles:
        data = {s: [_analyze_input(pair['input']) for pair in p[s]] for s in p}
        population_shifts += all(a['bands'] > max(t['bands'] for t in data['train']) for a in data['test'])
        geometry_shifts += all(a['ph'] != a['pw'] for a in data['test']) and all(t['ph'] == t['pw'] for t in data['train'])
        for split in p:
            for pair, a in zip(p[split], data[split]):
                pairs += 1
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
                geometries.add((a['ph'], a['pw'], a['bands'], a['copies']))
                bands.add(a['bands'])
                copies.add(a['copies'])
                palettes.add((a['lattice'], a['wall'], a['void']))
                colors.add(a['lattice'])
                count_sets.add(tuple(sorted(a['counts'])))
                signed_transfers += _features(a)['reversed_effect']
                for m, patch in zip(a['templates'], a['patches']):
                    templates.add(_key(m == a['void']))
                    patch_shapes.add(_key(patch))
                    for group in _components(m == a['void']):
                        areas.add(len(group))
                        rr, cc = zip(*group)
                        shape = np.zeros((max(rr) - min(rr) + 1, max(cc) - min(cc) + 1), dtype=bool)
                        for r, c in group:
                            shape[r - min(rr), c - min(cc)] = True
                        cavity_shapes.add(_key(shape))
    assert len(geometries) >= 30 and len(templates) >= 100
    assert len(palettes) >= 50 and colors == set(range(10))
    assert len(count_sets) >= 20 and len(patch_shapes) >= 100
    assert len(cavity_shapes) >= 15 and max(areas) >= 12
    assert bands == {2, 3, 4, 5} and copies == {3, 4, 5}
    assert population_shifts >= 5 and geometry_shifts >= 5 and signed_transfers >= 100
    # Every legal explicit count combination, including the minimum curriculum.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + ne, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # Counterfactuals: neither leave patches at source nor copy exceptional
    # panels can satisfy even the minimum training curriculum.
    minimum = generate_puzzle(0, 2, 1)
    for p in (minimum, puzzles[0]):
        assert any(not np.array_equal(_assemble([[a['panels'][r][a['columns'][r]]]
                                                for r in range(a['bands'])], a['lattice']), pair['output'])
                   for pair in p['train'] for a in [_analyze_input(pair['input'])])
    witness = next(pair for pair in minimum['train']
                   if _features(_analyze_input(pair['input']))['area_inversion'])
    a = _analyze_input(witness['input'])
    # An area-rank rule has a complete assignment, but gives the wrong answer.
    wrong = [m.copy() for m in a['templates']]
    source_order = sorted(range(a['bands']), key=lambda s: int(a['patches'][s].sum()))
    target_order = sorted(range(a['bands']), key=lambda t: a['areas'][t])
    for source, target in zip(source_order, target_order):
        wrong[target] = _toggle(a['templates'][target], a['patches'][source], a['wall'], a['void'])
    assert not np.array_equal(_assemble([[m] for m in wrong], a['lattice']), witness['output'])
    # Keep an address unchanged but move one payload cell: only the receiver's
    # two corresponding output cells must change. This directly tests A's role.
    source = next(s for s, t in enumerate(a['routes']) if s != t
                  and np.count_nonzero(a['patches'][s] & (a['templates'][s] == a['wall']))
                  == np.count_nonzero(a['patches'][s] & (a['templates'][s] == a['void'])))
    patch = a['patches'][source]
    old = tuple(np.argwhere(patch)[0])
    eligible = ~patch.copy()
    eligible[[0, -1], :] = False
    eligible[:, [0, -1]] = False
    new = tuple(np.argwhere(eligible)[0])
    changed = witness['input'].copy()
    r0 = 1 + source * (a['ph'] + 1)
    c0 = 1 + a['columns'][source] * (a['pw'] + 1)
    for r, c in (old, new):
        changed[r0 + r, c0 + c] = a['void'] if changed[r0 + r, c0 + c] == a['wall'] else a['wall']
    changed_output = solve(changed)
    difference = changed_output != witness['output']
    expected_difference = np.zeros_like(difference)
    dest_r0 = 1 + a['routes'][source] * (a['ph'] + 1)
    for r, c in (old, new):
        expected_difference[dest_r0 + r, 1 + c] = True
    assert np.array_equal(difference, expected_difference)
    # Reject malformed repetition and an incorrect output, rather than merely
    # rechecking dimensions. No protected evidence is read or modified.
    for defect in ('output', 'missing_exception', 'rim'):
        bad = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
        pair = bad['train'][0]
        a = _analyze_input(pair['input'])
        if defect == 'output':
            pair['output'][1, 1] = a['void']
        elif defect == 'missing_exception':
            c = 1 + a['columns'][0] * (a['pw'] + 1)
            pair['input'][1:1 + a['ph'], c:c + a['pw']] = a['templates'][0]
        else:
            c = 1 + a['columns'][0] * (a['pw'] + 1)
            pair['input'][1, c] = a['void']
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('validator accepted ' + defect)
    return dict(seeds=42, pairs=pairs, geometries=len(geometries), templates=len(templates),
                cavity_shapes=len(cavity_shapes), count_sets=len(count_sets),
                patch_shapes=len(patch_shapes), palettes=len(palettes),
                population_shifts=int(population_shifts), geometry_shifts=int(geometry_shifts))


if __name__ == '__main__':
    print(self_test())
