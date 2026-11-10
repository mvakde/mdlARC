"""Companions choose the mirror: relational selection of symmetry hypotheses.

Only standard-library modules and NumPy are used. Generation paints independently
sampled cores; solve reconstructs candidates from input cells alone. No file I/O,
external data, random choices, or generation metadata enters solve.
"""
from collections import Counter, defaultdict
from hashlib import sha256

import numpy as np


def _normal(cells):
    """Translation-normalized coordinates, with orientation preserved."""
    y0 = min(y for y, x in cells)
    x0 = min(x for y, x in cells)
    return tuple(sorted((y-y0, x-x0) for y, x in cells))


def _rotate(cells, turn):
    result = set(cells)
    for _ in range(turn % 4):
        result = {(x, -y) for y, x in result}
    return frozenset(result)


def _canonical(cells):
    return min(_normal(_rotate(cells, turn)) for turn in range(4))


def _candidates(cells):
    """All maximum-support axes PER orientation; identical subsets deduplicated.

    Axis coordinates are doubled: (dimension=1, a) reflects x to a-x;
    (dimension=0, a) reflects y to a-y. Ties are retained, never broken.
    """
    result = {}
    for dimension in (0, 1):
        low = 2*min(p[dimension] for p in cells)
        high = 2*max(p[dimension] for p in cells)
        scored = []
        for axis in range(low, high+1):
            if dimension == 1:
                subset = frozenset((y, x) for y, x in cells
                                   if (y, axis-x) in cells)
            else:
                subset = frozenset((y, x) for y, x in cells
                                   if (axis-y, x) in cells)
            scored.append((axis, subset))
        maximum = max(len(subset) for _, subset in scored)
        for axis, subset in scored:
            if len(subset) == maximum:
                result.setdefault(subset, []).append((dimension, axis))
    return result


def _classify(figures, equivalence=_canonical):
    """Largest witnessed subset per figure; support requires ANOTHER color."""
    choices = [_candidates(figure) for figure in figures]
    owners = defaultdict(set)
    for i, options in enumerate(choices):
        for subset in options:
            owners[equivalence(subset)].add(i)
    retained = []
    for options in choices:
        winners = [subset for subset in options
                   if len(owners[equivalence(subset)]) >= 2]
        if winners:
            maximum = max(map(len, winners))
            winners = [subset for subset in winners if len(subset) == maximum]
        if len(winners) > 1:
            raise ValueError('distinct equally largest witnessed subsets for a figure')
        retained.append(winners[0] if winners else frozenset())
    return choices, retained


def _as_grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((grid >= 0) & (grid <= 9))):
        raise ValueError('expected a 1..30 by 1..30 integer grid, colors 0..9')
    return grid


def _read_figures(grid):
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    background = int(counts.argmax())
    if counts[background] <= grid.size//2:
        raise ValueError('background must occupy a strict majority')
    colors = [int(c) for c in np.unique(grid) if c != background]
    figures = [frozenset(map(tuple, np.argwhere(grid == c).tolist())) for c in colors]
    return background, colors, figures


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _as_grid(input_grid)
    background, colors, figures = _read_figures(grid)
    if not figures:
        return grid.copy()
    _, retained = _classify(figures)
    output = np.full(grid.shape, background, dtype=int)
    for color, subset in zip(colors, retained):
        for cell in subset:
            output[cell] = color
    return output


def _core(rng, competing=False, parity=None):
    """Mirrored geometry: profiles, contours, bars, holes, supported islands."""
    height = int(rng.integers(3, 7))
    widths = [w for w in range(2, 8) if parity is None or (w-1) % 2 == parity]
    width = int(rng.choice(widths))
    kind = int(rng.integers(4))
    mask = np.zeros((height, width), dtype=bool)
    if kind == 0:
        mask = rng.random((height, width)) < float(rng.uniform(.18, .62))
        mask |= mask[:, ::-1]
    elif kind == 1:
        # Continuous, independently changing widths; even-width central stems.
        for row in range(height):
            depth = int(rng.integers(1, (width+1)//2+1))
            left = (width+1)//2-depth
            mask[row, left:width-left] = True
    elif kind == 2:
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        if width % 2 and rng.random() < .5:
            mask[:, width//2] = True
    else:
        mask[:, (width-1)//2:width//2+1] = True
        for row in rng.choice(height, int(rng.integers(1, height+1)), replace=False):
            mask[int(row), :] = True
    if competing:
        mask |= mask[::-1, :]
    row = int(rng.integers(height))
    mask[row, 0] = mask[row, -1] = True
    if competing:
        mask[height-1-row, 0] = mask[height-1-row, -1] = True
    # Ensure real extents without forcing an off-axis pair on every row.
    middle = (width-1)//2
    mask[0, middle] = mask[-1, middle] = True
    mask |= mask[:, ::-1]
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    if competing:
        # An axis tip (two central cells for even width) disrupts the other
        # orientation. One-sided paired
        # additions can then make that other orientation locally stronger.
        cells.add((height, width//2))
        cells.add((height, (width-1)//2))
    return frozenset(cells)


def _noisy(rng, core, competing=False, clean=False):
    height = max(y for y, x in core)+1
    width = max(x for y, x in core)+1
    for _ in range(64):
        noise = set()
        if competing:
            body_height = height-1
            row = int(rng.integers(max(1, body_height//2)))
            side = -int(rng.integers(1, 3)) if rng.random() < .5 else width-1+int(rng.integers(1, 3))
            noise = {(row, side), (body_height-1-row, side)}
            if width % 2 == 0:
                side2 = side-1 if side < 0 else side+1
                # Keep the entire raw box within the nine-cell packing bound.
                side = -1 if side < 0 else width
                side2 = -2 if side2 < 0 else width+1
                noise = {(r, c) for r in (row, body_height-1-row) for c in (side, side2)}
        elif not clean:
            amount = int(rng.integers(1, 7))
            pool = [(y, x) for y in range(-1, height+1) for x in range(-1, width+1)
                    if (y, x) not in core and (y, width-1-x) not in core
                    and 2*x != width-1]
            for j in rng.permutation(len(pool)):
                p = pool[int(j)]
                if (p[0], width-1-p[1]) not in noise:
                    noise.add(p)
                if len(noise) >= amount:
                    break
        raw = core | noise
        options = _candidates(raw)
        if core in options and (not competing or max(map(len, options)) > len(core)):
            return frozenset(raw)
    # A clean mirrored core is always a legal maximum in its own orientation.
    # Competing cases are resampled by the scene builder if this is insufficient.
    return core


def _local_object(raw, core, turn):
    raw = _rotate(raw, turn)
    core = _rotate(core, turn)
    y0 = min(y for y, x in raw)
    x0 = min(x for y, x in raw)
    return (frozenset((y-y0, x-x0) for y, x in raw),
            frozenset((y-y0, x-x0) for y, x in core))


_FALLBACK = (
    (
        '02222200/22222221/22222221/02222200/00020000',
        '0000001/0002200/0022220/1022220/0222220/0022220/0022220/1002200',
        '001000000/002222211/122222220/022222220/002222210/000020000/000000010',
        '111011/010101/000111/100101/000111/010101/111011',
        '111/111/111/111',
    ),
    (
        '022222/222022/222022/022222/010001/010001',
        '0220/2222/2222/2002/2222/2222',
        '222220/220222/220222/222220',
        '1111111/1000001/1000001/1000001/1111111',
        '11111/11111/11111/00100/11111',
    ),
    (
        '0001011/0120220/0222220/0222220/1020220/0100000',
        '0000010/0021220/0222220/0222220/1020221/1000001',
        '000001/022220/022220/002200/022220/102200',
        '00100010/12222220/00222200/02022021/10222200',
        '0100000/1222200/2122020/0222201/2222220/0000001',
        '000100000/111111110/011010010/010011010/010010010/011111110/000000001',
        '0011/1011/0111/0111/0111/0011/0011',
    ),
)


def _scene_fallback(rng, sizes, singles, parity=None):
    """Finite prechecked SAME-RULE scenes, trimmed to the requested profile.

    Each template has three versions of class 0, optionally two of class 1,
    then two unrelated figures. Digits encode empty/noise/retained locally;
    they are construction data, never read by solve. There is no rule exception.
    """
    multi = len(sizes) == 2
    data = _FALLBACK[2 if multi else int(parity or 0)]
    indices = list(range(max(sizes)))
    if multi:
        indices += [3, 4]
    orphan_start = 5 if multi else 3
    indices += list(range(orphan_start, orphan_start+singles))
    objects, expected = [], []
    turn = int(rng.integers(4))
    for i in indices:
        rows = data[i].split('/')
        raw = frozenset((y, x) for y, row in enumerate(rows) for x, c in enumerate(row) if c != '0')
        keep = frozenset((y, x) for y, row in enumerate(rows) for x, c in enumerate(row) if c == '2')
        raw, keep = _local_object(raw, keep, turn)
        objects.append(raw)
        expected.append(keep)
    assert _classify(objects)[1] == expected
    return objects, expected


def _make_scene(rng, sizes, singles, teaching=False, equal_area=False, parity=None):
    """Construct expected cores BEFORE calling the input-only classification."""
    for _ in range(120):
        objects, expected, used = [], [], set()
        repeated_areas = []
        good = True
        for class_index, count in enumerate(sizes):
            base = _core(rng, competing=teaching and class_index == 0,
                         parity=parity if class_index == 0 else None)
            key = _canonical(base)
            if key in used or (equal_area and class_index == 0 and len(_candidates(base)) < 2):
                good = False
                break
            used.add(key)
            repeated_areas.append(len(base))
            turns = list(map(int, rng.integers(0, 4, size=count)))
            if teaching and class_index == 0:
                turns[1] = (turns[0]+1) % 4
            for j, turn in enumerate(turns):
                hard = (teaching and class_index == 0 and j == 0)
                raw = _noisy(rng, base, competing=hard,
                             clean=(equal_area and class_index == 0 and j > 0)
                             or (not teaching and rng.random() < .18))
                raw, keep = _local_object(raw, base, turn)
                objects.append(raw)
                expected.append(keep)
        if not good:
            continue
        for j in range(singles):
            base = None
            for _ in range(160):
                proposal = _core(rng)
                if _canonical(proposal) in used:
                    continue
                if equal_area and j == 0 and len(proposal) not in repeated_areas:
                    continue
                base = proposal
                break
            if base is None:
                good = False
                break
            used.add(_canonical(base))
            # The teaching orphan is perfectly symmetric, yet must disappear.
            raw = _noisy(rng, base, clean=teaching or rng.random() < .25)
            raw, _ = _local_object(raw, base, int(rng.integers(4)))
            objects.append(raw)
            expected.append(frozenset())
        if not good:
            continue
        try:
            options, inferred = _classify(objects)
        except ValueError:
            continue
        if inferred != expected:
            continue
        if teaching and not any(keep and len(keep) < max(map(len, opts))
                                for keep, opts in zip(expected, options)):
            continue
        # Raw-mask matching cannot explain a teaching scene.
        raw_counts = Counter(_canonical(raw) for raw in objects)
        raw_answer = [raw if raw_counts[_canonical(raw)] >= 2 else frozenset()
                      for raw in objects]
        if teaching and raw_answer == expected:
            continue
        return objects, expected
    return _scene_fallback(rng, sizes, singles, parity)


def _place(rng, objects):
    """Free placement with cell-level separation, then guaranteed roomy packing."""
    count = len(objects)
    dimensions = [(max(y for y, x in raw)+1, max(x for y, x in raw)+1)
                  for raw in objects]
    for attempt in range(18):
        low = 12 if count <= 3 else 17 if count <= 5 else 22
        height, width = map(int, rng.integers(low, 31, size=2))
        if rng.random() < .2:
            width = height
        if 2*sum(map(len, objects)) >= height*width:
            continue
        occupied, offsets = set(), [None]*count
        for i in sorted(range(count), key=lambda i: -len(objects[i])):
            h, w = dimensions[i]
            for _ in range(80):
                y = int(rng.integers(height-h+1))
                x = int(rng.integers(width-w+1))
                shifted = {(a+y, b+x) for a, b in objects[i]}
                if not shifted & occupied:
                    offsets[i] = (y, x)
                    occupied |= {(a+dy, b+dx) for a, b in shifted
                                 for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
                    break
            if offsets[i] is None:
                break
        if all(p is not None for p in offsets):
            return (height, width), offsets
    # At most seven raw boxes, each <=9x9. A randomized 3x3 packing fits 30x30.
    slots = list(map(int, rng.permutation(9)[:count]))
    offsets = []
    for slot, (h, w) in zip(slots, dimensions):
        offsets.append((10*(slot//3)+int(rng.integers(10-h)),
                        10*(slot % 3)+int(rng.integers(10-w))))
    return (30, 30), offsets


def _pair(rng, sizes, singles, teaching=False, equal_area=False, parity=None, fallback=False):
    if fallback:
        objects, expected = _scene_fallback(rng, sizes, singles, parity)
    else:
        objects, expected = _make_scene(rng, sizes, singles, teaching, equal_area, parity)
    order = list(map(int, rng.permutation(len(objects))))
    objects = [objects[i] for i in order]
    expected = [expected[i] for i in order]
    shape, offsets = _place(rng, objects)
    palette = list(map(int, rng.permutation(10)))
    background, colors = palette[0], palette[1:len(objects)+1]
    inp = np.full(shape, background, dtype=int)
    out = inp.copy()
    for raw, keep, (top, left), color in zip(objects, expected, offsets, colors):
        for y, x in raw:
            inp[y+top, x+left] = color
        for y, x in keep:
            out[y+top, x+left] = color
    return {'input': inp, 'output': out}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _fingerprint(grid):
    return tuple(grid.shape), grid.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train_count = _count(num_train, 'num_train', 2, 10)
    test_count = _count(num_test, 'num_test', 1, 4)
    if train_count <= test_count:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    anchors = list(map(int, rng.choice(train_count, 2, replace=False)))
    for split, count in (('train', train_count), ('test', test_count)):
        for i in range(count):
            teaching = split == 'train' and i in anchors
            equal_area = teaching and i == anchors[1]
            if split == 'test':
                sizes = [2, int(rng.choice([2, 3]))]
                singles = int(rng.integers(0, 3))
            elif teaching:
                sizes = [2 if i == anchors[0] else 3]
                singles = int(rng.integers(1, 3))
            else:
                sizes = [int(rng.choice([2, 3]))]
                singles = int(rng.integers(0, 3))
            for _ in range(16):
                parity = int(i == anchors[1]) if teaching else None
                pair = _pair(rng, sizes, singles, teaching, equal_area, parity)
                # Same-orientation ties are a valid relational extension, but
                # never test them before any training figure demonstrates one.
                if (split == 'test' and _features(pair)['ambiguous_axis_resolved']
                        and not any(_features(p)['ambiguous_axis_resolved'] for p in puzzle['train'])):
                    continue
                fingerprint = _fingerprint(pair['input'])
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    puzzle[split].append(pair)
                    break
            else:
                pair = _pair(rng, sizes, singles, teaching, equal_area, parity, fallback=True)
                # At most thirteen prior inputs. Twenty distinct affine palette
                # permutations suffice even if ordinary retries all duplicate or
                # produce a test tie not yet demonstrated. No new marker is used.
                for sign in (1, -1):
                    for shift in range(10):
                        mapping = (sign*np.arange(10)+shift) % 10
                        candidate = {k: mapping[v] for k, v in pair.items()}
                        key = _fingerprint(candidate['input'])
                        if key not in seen:
                            seen.add(key)
                            puzzle[split].append(candidate)
                            break
                    else:
                        continue
                    break
                else:
                    raise AssertionError('twenty palettes cannot all be among thirteen prior inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    inp, out = (np.asarray(pair[k]) for k in ('input', 'output'))
    background, colors, figures = _read_figures(inp)
    choices, retained = _classify(figures)
    classes = Counter(_canonical(keep) for keep in retained if keep)
    owners = defaultdict(set)
    for i, options in enumerate(choices):
        for subset in options:
            owners[_canonical(subset)].add(i)
    # Compare explicit ablations, marking ambiguous alternative outputs as wrong.
    alternatives = {}
    for name, key in (('identity', _normal), ('area', len)):
        try:
            _, alternative = _classify(figures, key)
            alternatives[name] = alternative != retained
        except ValueError:
            alternatives[name] = True
    raw_counts = Counter(_canonical(raw) for raw in figures)
    raw_answer = [raw if raw_counts[_canonical(raw)] >= 2 else frozenset()
                  for raw in figures]
    all_local = [frozenset().union(*options) for options in choices]
    selected_axes = [axis for options, keep in zip(choices, retained) if keep
                     for axis in options[keep]]
    return {
        'shape': tuple(inp.shape), 'background': background, 'figures': len(figures),
        'classes': len(classes), 'multiplicities': tuple(sorted(classes.values())),
        'orphans': sum(not keep for keep in retained),
        'clean_orphan': any(not keep and raw in options
                            for raw, options, keep in zip(figures, choices, retained)),
        'weaker': sum(bool(keep) and len(keep) < max(map(len, options))
                      for options, keep in zip(choices, retained)),
        'smaller_witness': any(keep and any(len(subset) < len(keep)
                                            and len(owners[_canonical(subset)]) >= 2
                                            for subset in options)
                               for options, keep in zip(choices, retained)),
        'ambiguous_axis_resolved': any(keep and any(sum(dim == d for axes in options.values()
                                                      for dim, a in axes) > 1
                                                  for d, axis in options[keep])
                                       for options, keep in zip(choices, retained)),
        'no_A_differs': all_local != retained, 'no_B_differs': raw_answer != retained,
        'identity_differs': alternatives['identity'], 'area_differs': alternatives['area'],
        'orientations': tuple(sorted({d for d, a in selected_axes})),
        'parities': tuple(sorted({a % 2 for d, a in selected_axes})),
        'areas': tuple(sorted(len(keep) for keep in retained if keep)),
        'deleted': int(np.count_nonzero(inp != out)),
        'survivor_noise': sum(len(raw)-len(keep) for raw, keep in zip(figures, retained) if keep),
        'edge_contact': any(y in (0, inp.shape[0]-1) or x in (0, inp.shape[1]-1)
                            for figure in figures for y, x in figure),
    }


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = (_as_grid(pair[k]) for k in ('input', 'output'))
            assert inp.shape == out.shape
            key = _fingerprint(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            bg, colors, figures = _read_figures(inp)
            assert 2 <= len(colors) <= 7
            assert np.all((inp == out) | (out == bg)), 'only erase to background'
            assert np.array_equal(solve(inp), out), 'incorrect witnessed symmetry subset'
            _, retained = _classify(figures)
            for raw, keep in zip(figures, retained):
                assert keep <= raw
                if keep:
                    assert keep in _candidates(raw)
                    assert len(keep) >= 2
                    assert any(i != j and _canonical(keep) == _canonical(other)
                               for i, item in enumerate(retained) if item == keep
                               for j, other in enumerate(retained) if other)
            f = _features(pair)
            assert f['classes'] >= 1
            features[split].append(f)
    training = features['train']
    # These are generation evidence requirements, not hidden solve conventions.
    assert any(2 in f['multiplicities'] for f in training), 'pair witness missing'
    assert any(3 in f['multiplicities'] for f in training), 'triplet witness missing'
    assert sum(f['weaker'] > 0 for f in training) >= 2, 'joint selection not taught'
    assert any(f['clean_orphan'] for f in training), 'clean unpartnered figure not taught'
    assert any(f['smaller_witness'] for f in training), 'largest corroborated subset not taught'
    assert {a for f in training for a in f['parities']} == {0, 1}, 'both axis parities not taught'
    for name in ('no_A_differs', 'no_B_differs', 'identity_differs', 'area_differs'):
        assert any(f[name] for f in training), f'{name} alternative fits all training'
    assert all(f['classes'] == 1 for f in training)
    assert all(f['classes'] == 2 for f in features['test']), 'class-count shift missing'
    if any(f['ambiguous_axis_resolved'] for f in features['test']):
        assert any(f['ambiguous_axis_resolved'] for f in training), 'axis ties not taught'


def self_test():
    """42 seeds, streaming statistics, no disk writes or retained puzzle corpus."""
    shapes, palettes, structures, profiles, areas = set(), set(), set(), set(), set()
    signatures, flags = set(), Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                f = _features(pair)
                shapes.add(tuple(inp.shape))
                palettes.add(f['background'])
                profiles.add((f['figures'], f['multiplicities'], f['orphans']))
                areas.update(f['areas'])
                mapping = {}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in inp.ravel())
                structures.add(sha256(bytes(inp.shape)+normalized).digest())
                signature.update(bytes(inp.shape)+inp.astype(np.uint8).tobytes())
                for name in ('weaker', 'clean_orphan', 'ambiguous_axis_resolved',
                             'edge_contact', 'no_A_differs', 'no_B_differs'):
                    flags[name] += bool(f[name])
                flags['half_axis'] += 1 in f['parities']
        signatures.add(signature.digest())
        if seed in (0, 41):
            repeated = generate_puzzle(seed)
            assert all(np.array_equal(p[r], q[r]) for s in puzzle
                       for p, q in zip(puzzle[s], repeated[s]) for r in p)
        if seed == 0:
            # Inference must commute with rotations, palette permutation and
            # background padding, without consulting examples or generation.
            inp, out = (puzzle['test'][0][r] for r in ('input', 'output'))
            for turn in (1, 2, 3):
                assert np.array_equal(solve(np.rot90(inp, turn)), np.rot90(out, turn))
            palette_map = np.array([3, 9, 0, 8, 1, 5, 4, 7, 2, 6])
            assert np.array_equal(solve(palette_map[inp]), palette_map[out])
            if max(inp.shape) <= 28:
                bg, _, _ = _read_figures(inp)
                assert np.array_equal(solve(np.pad(inp, 1, constant_values=bg)),
                                      np.pad(out, 1, constant_values=bg))
    assert len(signatures) == 42 and len(shapes) >= 35 and len(structures) >= 190
    assert palettes == set(range(10)) and len(profiles) >= 12 and len(areas) >= 16
    assert flags['weaker'] >= 84 and flags['clean_orphan'] >= 84
    assert flags['edge_contact'] and flags['half_axis']
    # Equally large witnessed interpretations must be rejected, not resolved by
    # color order, leftmost axis, or first enumeration result.
    ambiguous = np.zeros((10, 16), dtype=int)
    elbow = {(0, 0), (0, 1), (1, 0)}
    for color, left in ((1, 1), (2, 9)):
        for y, x in elbow:
            ambiguous[y+2, x+left] = color
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError('unresolved relational ambiguity was accepted')
    # Clean copies corroborate both a whole figure and a smaller secondary
    # symmetry. The largest corroborated interpretation must win.
    clean = np.zeros((12, 18), dtype=int)
    mask = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)}
    for color, top, left, turn in ((1, 1, 1, 0), (7, 7, 9, 1)):
        for y, x in _normal(_rotate(mask, turn)):
            clean[y+top, x+left] = color
    assert np.array_equal(solve(clean), clean)
    for parity in (0, 1):
        for size in (2, 3):
            for singles in range(3):
                raw, keep = _scene_fallback(np.random.default_rng(9), [size], singles, parity)
                options, prediction = _classify(raw)
                assert prediction == keep
                assert any(k and len(k) < max(map(len, o)) for o, k in zip(options, keep))
    for sizes in ([2, 2], [2, 3]):
        for singles in range(3):
            raw, keep = _scene_fallback(np.random.default_rng(9), sizes, singles)
            assert _classify(raw)[1] == keep
    # Explicit count boundaries and unusual types; host also checks all legal counts.
    for tr, te in ((2, 1), (10, 4), (10, 1), (5, 4)):
        p = generate_puzzle(1000+tr+te, tr, te)
        assert len(p['train']) == tr and len(p['test']) == te
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            continue
        raise AssertionError('invalid counts accepted')
    bad = generate_puzzle(43, 2, 1)
    out = bad['test'][0]['output']
    bg, _, _ = _read_figures(bad['test'][0]['input'])
    y, x = np.argwhere(out != bg)[0]
    out[y, x] = bg
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('deleted core cell was accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds; no artifacts written)')
