"""Independent, color-matched unary rotations of an identifiable square."""
import numpy as np


def _is_int_scalar(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _resolve_counts(seed, rng, num_train, num_test):
    if num_train is not None:
        if not _is_int_scalar(num_train) or not 2 <= num_train <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _is_int_scalar(num_test) or not 1 <= num_test <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _array_key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_array_key(p['input']), _array_key(p['output']))
                       for p in puzzle[s]) for s in ('train', 'test'))


def _rotate_coord(r, c, size, turns):
    for _ in range(turns % 4):
        r, c = c, size - 1 - r
    return r, c


def _apply_transform(source, background, counts):
    # Generation uses array rotations; checking below uses coordinate arithmetic.
    result = np.full(source.shape, background, dtype=np.int8)
    occupied = np.zeros(source.shape, dtype=bool)
    for color in np.unique(source):
        if color == background:
            continue
        mask = np.rot90(source == color, -counts.get(int(color), 0) % 4)
        assert not np.any(occupied & mask), "conflicting rotated masks"
        result[mask] = color
        occupied |= mask
    return result


def _symmetry(mask):
    if np.array_equal(mask, np.rot90(mask)):
        return 'quarter'
    if np.array_equal(mask, np.rot90(mask, 2)):
        return 'half'
    return 'asymmetric'


def _orbits(n):
    remaining = {(r, c) for r in range(n) for c in range(n)}
    result = []
    while remaining:
        r, c = min(remaining)
        orbit = list(dict.fromkeys(_rotate_coord(r, c, n, k) for k in range(4)))
        remaining.difference_update(orbit)
        result.append(orbit)
    return result


def _make_source_square(rng, n, bg, colors, required_asym, geometry):
    """Allocate rotation-invariant territories, then compose shapes within them.

    Territories are sets of orbits, not visible frames. They guarantee disjoint
    masks for every possible counter value without a color overwrite priority.
    Radial bands and spatial Voronoi patches provide different shape grammars.
    """
    orbits = _orbits(n)
    count = len(colors)
    center = (n - 1) / 2
    depth = np.array([min(r, c, n-1-r, n-1-c) for r, c in (o[0] for o in orbits)])
    jitter = rng.random(len(orbits))
    if geometry == 'shell':
        order = np.argsort(depth + .85 * jitter)
    else:
        # Distance to a rotating off-center anchor is invariant but not radial:
        # territories can span many radii, corners, arms, and separated islands.
        ar, ac = rng.uniform(0, center, 2)
        anchor = [(ar, ac), (ac, n-1-ar), (n-1-ar, n-1-ac), (n-1-ac, ar)]
        scores = [min((r-u)**2 + (c-v)**2 for u, v in anchor) for r, c in (o[0] for o in orbits)]
        order = np.argsort(np.asarray(scores) + jitter * rng.uniform(.3, 2))
    # Every layer has room for both a shape and an asymmetry witness.
    allocations = np.full(count, 2, dtype=int)
    remainder = len(orbits) - int(allocations.sum())
    assert remainder >= 0
    allocations += rng.multinomial(remainder, rng.dirichlet(np.full(count, 1.8)))
    owners = [[] for _ in colors]
    cursor = 0
    for k in rng.permutation(count):
        owners[int(k)] = [int(i) for i in order[cursor:cursor + allocations[k]]]
        cursor += allocations[k]
    extent = int(rng.choice(np.flatnonzero(depth == 0)))
    # A tiny territory containing only the center and the extent orbit cannot
    # be both asymmetric and extent-preserving. Exchange its center orbit for
    # a noncentral orbit; the donor still retains a noncentral orbit of its own.
    for k in required_asym:
        if not any(i != extent and len(orbits[i]) == 4 for i in owners[k]):
            small = next(i for i in owners[k] if len(orbits[i]) == 1)
            donor = next(j for j in range(count) if j != k and
                         sum(len(orbits[i]) == 4 for i in owners[j]) >= 2)
            exchanged = next(i for i in owners[donor] if i != extent and len(orbits[i]) == 4)
            owners[k].remove(small)
            owners[k].append(exchanged)
            owners[donor].remove(exchanged)
            owners[donor].append(small)
    # Occasionally isolate a central singleton, as in the original test.
    central = next((i for i, o in enumerate(orbits) if len(o) == 1), None)
    eligible = [k for k in range(count) if k not in required_asym]
    if central is not None and eligible and rng.random() < .22:
        k = int(rng.choice(eligible))
        previous = next(j for j in range(count) if central in owners[j])
        recipient = (k + 1) % count
        owners[recipient].extend(i for i in owners[k] if i != central)
        owners[k] = []
        if previous != k:
            owners[previous].remove(central)
        owners[k] = [central]
    source = np.full((n, n), bg, dtype=np.int8)
    for k, color in enumerate(colors):
        ids = owners[k]
        kind = 'asymmetric' if k in required_asym else str(rng.choice(['asymmetric', 'half', 'quarter'], p=[.45, .27, .28]))
        if len(ids) == 1 and len(orbits[ids[0]]) == 1:
            kind = 'quarter'
        density = float(rng.uniform(.24, .96))
        motif = int(rng.integers(0, 5))
        seed_r, seed_c = rng.uniform(0, n, 2)
        phase = int(rng.integers(0, 4))
        thick = int(rng.integers(1, max(2, n // 3)))
        units = []
        scores = []
        for i in ids:
            orbit = orbits[i]
            if kind == 'quarter' or len(orbit) == 1:
                groups = [orbit]
            elif kind == 'half':
                groups = [orbit[::2], orbit[1::2]]
            else:
                groups = [[p] for p in orbit]
            for group in groups:
                r, c = group[0]
                if motif == 0:  # Filled bands with irregular protrusions.
                    score = depth[i] + rng.uniform(0, 1.4)
                elif motif == 1:  # Checkerboard and interrupted shells.
                    score = ((r + c + phase) % 2) * 3 + rng.random()
                elif motif == 2:  # Thick, offset bars and teeth.
                    score = ((r + phase) % (thick + 1)) + rng.random() * .5
                elif motif == 3:  # Connected patches within the territory.
                    score = (r-seed_r)**2 + (c-seed_c)**2 + rng.random()
                else:  # Multiple separated patches; density remains variable.
                    score = rng.random()
                units.append(group)
                scores.append(score)
        wanted = max(1, int(round(sum(len(g) for g in units) * density)))
        chosen = set()
        for j in np.argsort(scores):
            chosen.update(units[int(j)])
            if len(chosen) >= wanted:
                break
        if extent in ids:
            chosen.update(orbits[extent])
        # Preserve the requested symmetry exactly, including when dense filling
        # accidentally completed every phase of an orbit.
        changeable = [i for i in ids if i != extent and len(orbits[i]) == 4]
        if kind != 'quarter' and changeable:
            i = int(rng.choice(changeable))
            orbit = orbits[i]
            chosen.difference_update(orbit)
            start = int(rng.integers(0, 4))
            chosen.add(orbit[start])
            if kind == 'half':
                chosen.add(orbit[(start + 2) % 4])
        for r, c in chosen:
            source[r, c] = color
    assert all(np.any(source == color) for color in colors)
    assert all(_symmetry(source == colors[k]) == 'asymmetric' for k in required_asym)
    return source


def _square_candidates(inp):
    """Locate from input alone: tight multicolor square, clear halo, unary bars.

    Integral images test all sizes/positions without consulting an output size,
    border color, generation parameters, or seed. Fragmented source layers are
    allowed: no connected-component assumption is made inside the square.
    """
    values, counts = np.unique(inp, return_counts=True)
    if np.sum(counts == counts.max()) != 1:
        return []
    bg = int(values[counts.argmax()])
    occupied = inp != bg
    h, w = inp.shape
    prefix = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    total = int(occupied.sum())
    candidates = []
    def mass(r, c, e, f):
        return prefix[e, f] - prefix[r, f] - prefix[e, c] + prefix[r, c]
    for n in range(3, min(h, w) + 1):
        rr, cc = np.indices((h-n+1, w-n+1))
        ee, ff = rr+n, cc+n
        inside = mass(rr, cc, ee, ff)
        valid = (inside >= 5) & (inside < total)
        valid &= mass(rr, cc, rr+1, ff) > 0
        valid &= mass(ee-1, cc, ee, ff) > 0
        valid &= mass(rr, cc, ee, cc+1) > 0
        valid &= mass(rr, ff-1, ee, ff) > 0
        valid &= mass(np.maximum(0, rr-1), np.maximum(0, cc-1),
                      np.minimum(h, ee+1), np.minimum(w, ff+1)) == inside
        for r, c in np.argwhere(valid):
            r, c = int(r), int(c)
            source = inp[r:r+n, c:c+n]
            palette = tuple(int(x) for x in np.unique(source) if x != bg)
            if len(palette) < 2:
                continue
            ext = occupied.copy()
            ext[r:r+n, c:c+n] = False
            markers = {}
            good = True
            for color in np.unique(inp[ext]):
                color = int(color)
                coords = np.argwhere(ext & (inp == color))
                rows, cols = coords.T
                horizontal = np.ptp(rows) == 0 and np.ptp(cols) + 1 == len(rows)
                vertical = np.ptp(cols) == 0 and np.ptp(rows) + 1 == len(rows)
                if color not in palette or not (horizontal or vertical):
                    good = False
                    break
                markers[color] = {
                    'length': len(rows),
                    'orientation': 'single' if len(rows) == 1 else ('horizontal' if horizontal else 'vertical'),
                    'coords': tuple(map(tuple, coords.tolist())),
                }
            if not good:
                continue
            # Separate bars even when their colors differ.
            coords_by_color = [(color, p) for color, m in markers.items() for p in m['coords']]
            if any(color != other and max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 1
                   for i, (color, p) in enumerate(coords_by_color)
                   for other, q in coords_by_color[i+1:]):
                continue
            candidates.append({'background': bg, 'source_row': r, 'source_col': c,
                               'size': n, 'source': source.copy(), 'palette': palette,
                               'markers': markers})
    return candidates


def _locate(inp):
    candidates = _square_candidates(inp)
    assert len(candidates) == 1, 'source square must be uniquely localizable from the input'
    return candidates[0]


def _predict_from_info(info, direction=1, absent=0):
    n = info['size']
    result = np.full((n, n), info['background'], dtype=np.int8)
    for color in info['palette']:
        k = info['markers'].get(color, {}).get('length', absent)
        for r, c in np.argwhere(info['source'] == color):
            y, x = _rotate_coord(int(r), int(c), n, direction * k)
            assert result[y, x] == info['background'], 'rotated masks overlap'
            result[y, x] = color
    return result


def _analyze_pair(pair):
    info = _locate(pair['input'])
    expected = _predict_from_info(info)
    assert np.array_equal(pair['output'], expected), 'incorrect rotation or crop'
    for color in info['palette']:
        assert np.sum(info['source'] == color) == np.sum(pair['output'] == color)
    return info


def _bar_candidates(h, w, r, c, n, length, blocked, required=None):
    groups = []
    for orientation in ('horizontal', 'vertical'):
        if required is not None and orientation != required:
            continue
        bh, bw = (1, length) if orientation == 'horizontal' else (length, 1)
        rr, cc = np.indices((h-bh+1, w-bw+1))
        clear = np.ones(rr.shape, dtype=bool)
        for i in range(length):
            clear &= ~blocked[rr + (i if bh > 1 else 0), cc + (i if bw > 1 else 0)]
        sides = [rr+bh <= r-1, rr >= r+n+1, cc+bw <= c-1, cc >= c+n+1]
        for side, predicate in enumerate(sides):
            starts = np.argwhere(clear & predicate)
            if len(starts):
                groups.append((orientation, side, starts))
    return groups


def _put_bars(rng, source, bg, colors, counts, forced_orientations):
    n = len(source)
    for attempt in range(45):
        if attempt >= 35:
            h = w = 30
            r = c = 0
        else:
            h = int(rng.integers(max(n+2, 12), 31))
            w = int(rng.integers(max(n+2, 12), 31))
            r = int(rng.choice([0, h-n])) if rng.random() < .2 else int(rng.integers(1, h-n))
            c = int(rng.choice([0, w-n])) if rng.random() < .2 else int(rng.integers(1, w-n))
        inp = np.full((h, w), bg, dtype=np.int8)
        inp[r:r+n, c:c+n] = source
        blocked = np.zeros((h, w), dtype=bool)
        blocked[max(0,r-1):min(h,r+n+1), max(0,c-1):min(w,c+n+1)] = True
        # Long bars first makes packing predictable without fixing their side.
        marked = [k for k in range(len(colors)) if counts.get(colors[k], 0)]
        marked.sort(key=lambda k: -counts[colors[k]])
        success = True
        for k in marked:
            length = counts[colors[k]]
            groups = _bar_candidates(h, w, r, c, n, length, blocked, forced_orientations.get(k))
            if not groups:
                success = False
                break
            orient, _, starts = groups[int(rng.integers(len(groups)))]
            y, x = starts[int(rng.integers(len(starts)))]
            for i in range(length):
                yy, xx = int(y + (i if orient == 'vertical' else 0)), int(x + (i if orient == 'horizontal' else 0))
                inp[yy, xx] = colors[k]
                blocked[max(0,yy-1):min(h,yy+2), max(0,xx-1):min(w,xx+2)] = True
        if not success:
            continue
        vals, nums = np.unique(inp, return_counts=True)
        if int(vals[nums.argmax()]) != bg or np.sum(nums == nums.max()) != 1:
            continue
        candidates = _square_candidates(inp)
        if len(candidates) == 1 and (candidates[0]['source_row'], candidates[0]['source_col'], candidates[0]['size']) == (r, c, n):
            return inp
    return None


def _fallback_pair(bg, colors, counts, forced):
    """Guaranteed packing after bounded attempts; no random retries here.

    A simple complete frame is reserved for this rare fallback. All bars lie
    within the square's row span and to its right, so adding a bar cannot yield
    another tight square. Separate interior orbits keep every layer asymmetric.
    """
    n = 13
    source = np.full((n, n), bg, dtype=np.int8)
    source[0] = source[-1] = colors[0]
    source[:, 0] = source[:, -1] = colors[0]
    interior = [orbit for orbit in _orbits(n) if len(orbit) == 4 and
                all(0 < r < n-1 and 0 < c < n-1 for r,c in orbit)]
    for color, orbit in zip(colors, interior):
        r, c = orbit[0]
        source[r,c] = color
    inp = np.full((15, 30), bg, dtype=np.int8)
    inp[:n, :n] = source
    row = 0
    for k, color in enumerate(colors):
        length = counts.get(color, 0)
        if not length:
            continue
        if forced.get(k) == 'vertical':
            inp[:length, 26] = color
        else:
            inp[row, 15:15+length] = color
            row += 2
    return {'input': inp, 'output': _apply_transform(source, bg, counts)}


def _sample_counts(rng, colors, regime, mode, witness, witness_counts):
    cap = 3 if mode == 1 and regime == 'source_train' else (4 if mode == 0 else 9)
    marked = int(rng.integers(1, len(colors)+1))
    selected = rng.choice(len(colors), marked, replace=False)
    lengths = [0] * len(colors)
    for k in selected:
        lengths[int(k)] = int(rng.integers(1, cap+1))
    forced = {}
    required_asym = set()
    first, repeated = witness_counts
    if witness == 0:
        # Repetition in different orientations directly defeats location/ranking
        # shortcuts. The actual missing residue varies between puzzles.
        lengths[:3] = [first, repeated, repeated]
        forced[1], forced[2] = ('horizontal', 'vertical') if rng.random() < .5 else ('vertical', 'horizontal')
        required_asym = {0, 1, 2}
    elif witness == 1:
        shown = {first, repeated}
        missing = min({1, 2, 3} - shown)
        lengths[:2] = [missing, 0]
        required_asym = {0, 1}
        if mode in (2, 3):
            lengths[2] = int(rng.integers(4, 10))
    elif rng.random() < .03 and cap >= 4:
        # Meaningful identity crops after the informative demonstrations.
        lengths = [int(rng.choice([0, 4] if cap == 4 else [0, 4, 8])) for _ in colors]
        if not any(lengths):
            lengths[int(rng.integers(len(colors)))] = 4
    if regime == 'target_test' and mode in (1, 3):
        lengths[int(rng.integers(len(colors)))] = int(rng.integers(5, 10))
    return {color: length for color, length in zip(colors, lengths) if length}, forced, required_asym


def _generate_pair(rng, regime, mode, witness, witness_counts, previous):
    choices = list(range(3 if witness == 0 or (witness == 1 and mode in (2, 3)) else 2, 7))
    if previous is not None:
        choices = [n for n in choices if n != len(previous['palette'])]
    layer_count = int(rng.choice(choices))
    bg_choices = [v for v in range(10) if previous is None or v != previous['background']]
    bg = int(rng.choice(bg_choices))
    colors = [int(x) for x in rng.choice([c for c in range(10) if c != bg], layer_count, replace=False)]
    counts, forced, required_asym = _sample_counts(rng, colors, regime, mode, witness, witness_counts)
    minimum = max(5, int(np.ceil(np.sqrt(8*layer_count))))
    n = int(rng.integers(minimum, 24))
    if mode == 3:
        geometry = 'patch' if regime == 'target_test' else ('shell' if max(counts.values()) > 3 else 'patch')
    else:
        geometry = str(rng.choice(['shell', 'patch'], p=[.65, .35]))
    for attempt in range(24):
        source = _make_source_square(rng, n, bg, colors, required_asym, geometry)
        inp = _put_bars(rng, source, bg, colors, counts, forced)
        if inp is not None:
            return {'input': inp, 'output': _apply_transform(source, bg, counts)}
    return _fallback_pair(bg, colors, counts, forced)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not _is_int_scalar(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    # The semantic convention is fixed by the source family. Only curriculum
    # varies at puzzle level; palettes and backgrounds carry no hidden meaning.
    mode = int(rng.integers(0, 4))
    repeated = int(rng.choice([2, 3]))
    witness_counts = (int(rng.choice([v for v in (1,2,3) if v != repeated])), repeated)
    puzzle = {'train': [], 'test': []}
    previous = None
    used = set()
    for split, regime, count in [('train', 'source_train', num_train), ('test', 'target_test', num_test)]:
        for index in range(count):
            witness = index if split == 'train' and index < 2 else -1
            pair = _generate_pair(rng, regime, mode, witness, witness_counts, previous)
            # Pair-local palettes, backgrounds, and layer counts genuinely vary.
            assert _array_key(pair['input']) not in used
            used.add(_array_key(pair['input']))
            puzzle[split].append(pair)
            previous = _locate(pair['input'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= dim <= 30 for dim in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))


def _training_evidence(train, infos):
    witnesses = {'clockwise': False, 'independent': False, 'unmarked': False, 'orientation': False}
    informative_lengths = set()
    for pair, info in zip(train, infos):
        residues = set()
        orientations = {}
        for color in info['palette']:
            mask = info['source'] == color
            if _symmetry(mask) != 'asymmetric':
                continue
            if color not in info['markers']:
                witnesses['unmarked'] = True
                continue
            marker = info['markers'][color]
            length = marker['length']
            informative_lengths.add(length)
            residues.add(length % 4)
            if length % 2:
                witnesses['clockwise'] = True
            if length > 1:
                orientations.setdefault(length, set()).add(marker['orientation'])
        witnesses['orientation'] |= any(len(v) == 2 for v in orientations.values())
        witnesses['independent'] |= len(residues) >= 2
    return witnesses, informative_lengths


def solve(train_pairs, test_input):
    """Prediction from visible demonstrations and a test input only.

    This is an internal check, not an independent protected reference solver.
    Infer clockwise versus counterclockwise and the unmarked default using
    mask correspondences in training, then count the test's exterior bars.
    """
    infos = [_locate(pair['input']) for pair in train_pairs]
    conventions = []
    for direction in (1, -1):
        for absent in range(4):
            good = True
            for pair, info in zip(train_pairs, infos):
                for color in info['palette']:
                    k = info['markers'].get(color, {}).get('length', absent)
                    predicted = np.rot90(info['source'] == color, -direction*k % 4)
                    if not np.array_equal(predicted, pair['output'] == color):
                        good = False
                        break
                if not good:
                    break
            if good:
                conventions.append((direction, absent))
    assert conventions == [(1, 0)], 'training must determine direction and the unmarked default'
    return _predict_from_info(_locate(test_input), *conventions[0])


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs = set(), set()
    analyses = {'train': [], 'test': []}
    original_keys = {(_array_key(a), _array_key(b)) for a, b in _original_pairs()}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            _check_grid(pair['input'])
            _check_grid(pair['output'])
            ik = _array_key(pair['input'])
            pk = ik, _array_key(pair['output'])
            assert ik not in seen_inputs and pk not in seen_pairs and pk not in original_keys
            seen_inputs.add(ik)
            seen_pairs.add(pk)
            info = _analyze_pair(pair)
            assert 2 <= len(info['palette']) <= 6
            assert 1 <= len(info['markers']) <= len(info['palette'])
            assert all(1 <= m['length'] <= 9 for m in info['markers'].values())
            analyses[split].append(info)
    evidence, lengths = _training_evidence(puzzle['train'], analyses['train'])
    assert all(evidence.values()), 'insufficient asymmetric training evidence'
    assert {1, 2, 3}.issubset(lengths), 'demonstrate unary counts before extrapolating'
    all_info = analyses['train'] + analyses['test']
    assert len({i['background'] for i in all_info}) >= 2
    assert len({len(i['palette']) for i in all_info}) >= 2
    assert len({i['palette'] for i in all_info}) >= 2
    for pair in puzzle['test']:
        prediction = solve(puzzle['train'], pair['input'].copy())
        assert np.array_equal(prediction, pair['output'])


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    sizes = []
    while remaining:
        stack = [remaining.pop()]
        size = 0
        while stack:
            r, c = stack.pop()
            size += 1
            for q in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        sizes.append(size)
    return sorted(sizes)


def measure_pair(pair):
    """Review metrics reconstructed from the returned grids, not sampled flags."""
    info = _analyze_pair(pair)
    n, source, bg = info['size'], info['source'], info['background']
    r, c = info['source_row'], info['source_col']
    h, w = pair['input'].shape
    layers = []
    perimeter = np.zeros((n,n), dtype=bool)
    perimeter[0] = perimeter[-1] = True
    perimeter[:,0] = perimeter[:,-1] = True
    yy, xx = np.indices((n,n))
    depth = np.minimum.reduce([yy, xx, n-1-yy, n-1-xx])
    for color in info['palette']:
        mask = source == color
        coords = np.argwhere(mask)
        dims = np.ptp(coords, axis=0)+1
        area = int(mask.sum())
        components = _components(mask)
        eroded = mask.copy()
        thickness = 0
        while eroded.any():
            thickness += 1
            padded = np.pad(eroded, 1)
            eroded = padded[1:-1,1:-1] & padded[:-2,1:-1] & padded[2:,1:-1] & padded[1:-1,:-2] & padded[1:-1,2:]
        layers.append({'color': color, 'area': area, 'symmetry': _symmetry(mask),
                       'components': len(components), 'largest_component': max(components),
                       'density': round(area / int(np.prod(dims)), 4),
                       'depth_span': int(np.ptp(depth[mask])), 'thickness': thickness,
                       'min_depth': int(depth[mask].min()), 'max_depth': int(depth[mask].max()),
                       'border_cells': int(np.sum(mask & perimeter)),
                       'full_perimeter': bool(np.all(mask[perimeter])),
                       'marked': color in info['markers'],
                       'length': info['markers'].get(color, {}).get('length', 0),
                       'unchanged': bool(np.array_equal(mask, pair['output'] == color))})
    markers = []
    for color, marker in info['markers'].items():
        rows, cols = np.array(marker['coords']).T
        sides = [name for name, test in [('top', max(rows)<r), ('bottom', min(rows)>=r+n),
                                         ('left', max(cols)<c), ('right', min(cols)>=c+n)] if test]
        distance = min(max(max(r-y,0,y-(r+n-1)),max(c-x,0,x-(c+n-1))) for y,x in marker['coords']) - 1
        markers.append({'color': color, 'length': marker['length'], 'orientation': marker['orientation'],
                        'sides': sides, 'distance': int(distance)})
    lengths = [m['length'] for m in markers]
    orientations = {m['orientation'] for m in markers} - {'single'}
    return {'size': n, 'input_shape': list(pair['input'].shape), 'position': [r,c],
            'margins': [r,h-r-n,c,w-c-n], 'background': bg, 'palette': list(info['palette']),
            'layer_count': len(layers), 'layers': layers, 'markers': markers,
            'area': int(np.sum(source != bg)), 'density': round(float(np.mean(source != bg)),4),
            'marked_count': len(markers), 'unmarked_count': len(layers)-len(markers),
            'lengths': sorted(lengths), 'residues': sorted({k%4 for k in lengths}),
            'repeated_counts': len(lengths) != len(set(lengths)),
            'mixed_orientation': len(orientations) == 2,
            'same_side_bars': any(sum(side in m['sides'] for m in markers) > 1 for side in ('top','bottom','left','right')),
            'full_frame': any(l['full_perimeter'] for l in layers),
            'identity_crop': bool(np.array_equal(source, pair['output'])),
            'edge_touching': min(r,h-r-n,c,w-c-n) == 0,
            'mixed_short_long': min(lengths) <= 3 and max(lengths) >= 5}


def self_test() -> None:
    # Original solved pairs are regression evidence, including broken borders,
    # the unchanged half-turn layer, and the marked central singleton.
    original = _original_pairs()
    for inp, out in original:
        _analyze_pair({'input': inp, 'output': out})
    train = [{'input': a.copy(), 'output': b.copy()} for a,b in original[:3]]
    inferred = solve(train, original[3][0].copy())
    assert np.array_equal(inferred, original[3][1])
    # Small territories must preserve both crop extent and asymmetry.
    for seed in range(32):
        source = _make_source_square(np.random.default_rng(seed), 5, 0,
                                     [1,2,3], {0,1,2}, 'patch')
        assert all(_symmetry(source == c) == 'asymmetric' for c in (1,2,3))
    fallback = _fallback_pair(0, [1,2,3,4,5,6],
                              {1:9,2:3,3:3,4:8,5:7,6:6}, {1:'horizontal',2:'vertical'})
    _analyze_pair(fallback)
    records, puzzles, train_counts, test_counts = [], set(), set(), set()
    observed_regimes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        local = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = measure_pair(pair)
                f['split'] = split
                records.append(f)
                local.append(f)
        assert len({f['background'] for f in local}) >= 2
        assert len({f['layer_count'] for f in local}) >= 2
        assert len({tuple(f['palette']) for f in local}) >= 2
        training = [f for f in local if f['split'] == 'train']
        testing = [f for f in local if f['split'] == 'test']
        train_max = max(k for f in training for k in f['lengths'])
        test_max = max(k for f in testing for k in f['lengths'])
        if train_max <= 3 and test_max >= 5:
            observed_regimes.add('extrapolation')
        if train_max <= 4 and test_max <= 4:
            observed_regimes.add('shared_short')
        if train_max >= 5:
            observed_regimes.add('long_demonstrated')
        def geometry_bin(f):
            return (f['size'] // 6, f['edge_touching'])
        combinations = {(geometry_bin(f), max(f['lengths']) >= 5) for f in training}
        if any((geometry_bin(f), max(f['lengths']) >= 5) not in combinations
               and geometry_bin(f) in {geometry_bin(t) for t in training}
               and (max(f['lengths']) >= 5) in {max(t['lengths']) >= 5 for t in training}
               for f in testing):
            observed_regimes.add('held_out_combination')
    assert len(puzzles) == 42 and len(train_counts) >= 4 and len(test_counts) >= 3
    assert observed_regimes == {'extrapolation', 'shared_short', 'long_demonstrated', 'held_out_combination'}
    assert len({tuple(f['input_shape']) for f in records}) >= 50
    assert len({f['size'] for f in records}) >= 15
    assert {f['background'] for f in records} == set(range(10))
    assert {f['layer_count'] for f in records} == set(range(2,7))
    layers = [l for f in records for l in f['layers']]
    assert {l['symmetry'] for l in layers} == {'quarter','half','asymmetric'}
    assert min(l['area'] for l in layers) == 1 and max(l['area'] for l in layers) >= 80
    assert any(l['components'] == 1 and l['area'] >= 12 for l in layers)
    assert any(l['components'] >= 10 for l in layers)
    assert any(max(l['area'] for l in f['layers']) >= 30 and min(l['area'] for l in f['layers']) <= 5 for f in records)
    assert any(sum(l['area'] >= 20 for l in f['layers']) >= 2 for f in records)
    assert max(l['thickness'] for l in layers) >= 3
    assert max(l['depth_span'] for l in layers) >= 5
    assert min(f['density'] for f in records) < .4 and max(f['density'] for f in records) > .7
    assert sum(f['full_frame'] for f in records) < len(records) // 3
    assert any(min(f['margins']) >= 4 for f in records)
    assert sum(not f['edge_touching'] for f in records) > len(records) // 3
    assert any(l['symmetry']=='half' and l['length'] % 4 == 2 and l['unchanged'] and l['marked'] for l in layers)
    assert any(l['symmetry']=='half' and l['length'] % 2 and not l['unchanged'] for l in layers)
    assert any(l['symmetry']=='quarter' and l['marked'] and l['unchanged'] for l in layers)
    for name in ('repeated_counts','mixed_orientation','same_side_bars','identity_crop','edge_touching','mixed_short_long'):
        assert sum(f[name] for f in records) >= 5, name
    assert any(f['unmarked_count'] >= 2 for f in records)
    assert any(f['marked_count'] == 1 for f in records)
    assert any(f['unmarked_count'] == 0 for f in records)
    assert any(len({l['symmetry'] for l in f['layers']}) == 3 for f in records)
    markers = [m for f in records for m in f['markers']]
    for length in range(2,10):
        assert {m['orientation'] for m in markers if m['length'] == length} == {'horizontal', 'vertical'}
        assert len({s for m in markers if m['length'] == length for s in m['sides']}) >= 3
    assert max(m['distance'] for m in markers) >= 10
    for split in ('train','test'):
        fs = [f for f in records if f['split']==split]
        assert set(k for f in fs for k in f['lengths']) == set(range(1,10))
        assert sum(f['mixed_short_long'] for f in fs) >= 5
        assert any(f['edge_touching'] and f['size'] >= 17 and f['mixed_orientation'] for f in fs)
        assert len({tuple(f['residues']) for f in fs}) >= 7
        for side in ('top','bottom','left','right'):
            assert any(side in m['sides'] for f in fs for m in f['markers'])
        assert any(f['size'] <= 8 and max(f['lengths']) >= 5 for f in fs)
        assert any(f['size'] >= 18 and max(f['lengths']) <= 3 for f in fs)
        assert any(f['unmarked_count'] >= 1 and max(f['lengths']) >= 5 for f in fs)
    for a,b in ((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None)):
        p = generate_puzzle(710, a, b)
        assert a is None or len(p['train']) == a
        assert b is None or len(p['test']) == b
    for a,b in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(True,1),(3,1.5)):
        try:
            generate_puzzle(710,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    assert _puzzle_key(generate_puzzle(777,6,3)) == _puzzle_key(generate_puzzle(777,6,3))
    assert _puzzle_key(generate_puzzle(778,6,3)) != _puzzle_key(generate_puzzle(777,6,3))
    # Corrupt an output and an exterior counter: checks must reject both.
    p = generate_puzzle(711,2,1)
    pair = p['test'][0]
    bad = {'input': pair['input'].copy(), 'output': pair['output'].copy()}
    bad['output'][0,0] = (int(bad['output'][0,0])+1)%10
    try:
        _analyze_pair(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    bad = {name: grid.copy() for name, grid in p['train'][0].items()}
    info = _locate(bad['input'])
    r, c = next(iter(info['markers'].values()))['coords'][0]
    absent_color = next(c for c in range(10) if c != info['background'] and c not in info['palette'])
    bad['input'][r,c] = absent_color
    try:
        _analyze_pair(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('unmatched exterior counter accepted')


def _original_pairs():
    return [(np.array([[int(c) for c in row] for row in a], dtype=np.int8),
             np.array([[int(c) for c in row] for row in b], dtype=np.int8)) for a,b in _SOURCE_ROWS]


# Immutable regression fixtures from the staged original task.
_SOURCE_ROWS = ((('00000000303000000000', '00000003333333333300', '00000033880888080300', '00000003808808888300', '00000003080060080300', '00000003886666088330', '00000003800606008300', '00000003880666688330', '00000003080060080300', '00000033888808888300', '00000003080888080300', '00000003333333333300', '00000000000330000000', '00000000000000000000', '00000000000000000000', '00000000000000000000', '08000000000000000000', '08000000000000000000', '00000300066000000000', '00000000000000000000'), ('0003000000300', '0333333333330', '0308088808033', '0388880888830', '0308006008033', '3388666608830', '3380060600830', '0388066668830', '0308006008030', '0388880880830', '0308088808830', '0333333333330', '0000030300000')), (('00000000000000000000', '00000000000000000000', '00000050505555050000', '00000000030000000000', '00000050033330050000', '00000003304003050000', '00000050304403300000', '00000050304443050000', '00000050300003050000', '00000050033330050000', '00000000003000000000', '00000050550555050000', '00000000000000000000', '00000000000000000000', '00000000000000000000', '03000000000000000000', '00004000000000000000', '00000000000000000000', '00000000000000000000', '00000000000000000000'), ('5050555505', '0000003000', '5003333005', '0030000335', '5330444300', '5030440305', '5030400305', '5003333005', '0000030000', '5055055505')), (('00000000000000000000', '00000000000000000000', '00000000111000000000', '00000000000000400000', '00000000000000000000', '00000000000000000000', '00000000000000000000', '00000000000000000000', '00000000000000000000', '04040404040000000000', '00404040404000000000', '04000010040000000000', '00401001104000000000', '04001220140000000000', '00410221004000000000', '04001100040000000000', '00400110004000000000', '04040404040000000000', '00404040404000000000', '00000000000000000000'), ('0404040404', '4040404040', '0401100004', '4001010040', '0410220104', '4000221140', '0401101004', '4000010040', '0404040404', '4040404040')), (('00005000000000000000', '00005004000000000000', '03305000000000000000', '00005000000000000000', '00005000000000000000', '00000000000000000000', '00000000000055500000', '00000000000555550000', '00000000000000000000', '00000000000033000000', '00000005503333330055', '00000000000300033055', '00000005503304033055', '00000000000300033055', '00000005503333330055', '00000000000000300000', '00000000000000000000', '00000000000505050000', '00000000000050500000', '00000000000000000000'), ('0000505050000', '0000505050000', '0000000000000', '0000030000000', '0500333333050', '5003300030055', '0503304033055', '5003300030055', '0500333333050', '0000003300000', '0000000000000', '0000555550000', '0000555550000')))
