"""Periodic fault atlas: recover a stamp, fold its faults, print signed imprints.

The public solver reads only its argument. Generation uses latent bitmaps and
Kronecker products; validation checks recovered commands one output block at a
time. No parent code, files, seeds, or examples enter the solving path.
"""
from __future__ import annotations

import numpy as np


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, low, high=None):
    _need(not isinstance(value, (bool, np.bool_))
          and isinstance(value, (int, np.integer)), name + " must be an integer")
    value = int(value)
    _need(value >= low and (high is None or value <= high), name + " out of range")
    return value


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    _need(a > b, "num_train must exceed num_test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((0 <= a) & (a <= 9))), "grid colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _recover(motif, structure, label):
    """Nearest strict-majority periodic field with >=3 samples of every phase.

    Period aliases are harmless only when they give exactly the same repaired
    field. The shortest alias then defines the coordinate quotient.
    """
    _need(structure != label and bool(np.all((motif == structure) | (motif == label))),
          "periodic field must be binary")
    bits = motif == label
    h, length = bits.shape
    candidates = []
    for period in range(1, min(5, length // 3) + 1):
        tile = np.empty((h, period), dtype=bool)
        for c in range(period):
            phase = bits[:, c::period]
            ones = phase.sum(axis=1)
            if np.any(2 * ones == phase.shape[1]):
                break
            tile[:, c] = 2 * ones > phase.shape[1]
        else:
            clean = tile[:, np.arange(length) % period]
            candidates.append((int(np.sum(bits != clean)), period, tile, clean))
    _need(bool(candidates), "no supported periodic field")
    candidates.sort(key=lambda item: (item[0], item[1]))
    errors, period, tile, clean = candidates[0]
    _need(all(np.array_equal(item[3], clean) for item in candidates if item[0] == errors),
          "competing minimum-error repairs")
    _need(2 <= period <= 5, "stamp must have primitive period 2..5")
    _need(tile.any() and not tile.all(), "stamp must contain ink and holes")
    _need(1 <= errors <= max(2, motif.size // 8), "faults must be sparse")
    differences = bits != clean
    folded = np.stack([differences[:, c::period].any(axis=1)
                       for c in range(period)], axis=1)
    _need(bool(np.any(folded & tile)) and bool(np.any(folded & ~tile)),
          "each lane needs a missing-ink and a surplus-ink fault")
    return dict(period=period, tile=tile, clean=clean, differences=differences,
                folded=folded, errors=errors,
                missing=folded & tile, surplus=folded & ~tile)


def _parse_canonical(a):
    height, width = a.shape
    _need(height >= 6 and width >= 13, "scene too small")
    structure = int(a[0, 0])
    _need(bool(np.all(a[0] == structure) and np.all(a[-1] == structure)
               and np.all(a[:, 0] == structure) and np.all(a[:, -1] == structure)
               and np.all(a[:, 3] == structure)), "frame and key divider")
    dividers = np.flatnonzero(np.all(a == structure, axis=1)).tolist()
    gaps = np.diff(dividers)
    _need(len(gaps) >= 1 and bool(np.all(gaps == gaps[0])), "equal lane thickness")
    h = int(gaps[0]) - 3
    k = len(gaps)
    length = width - 7
    _need(2 <= h <= 5 and 1 <= k <= 4 and 6 <= length <= 23, "lane geometry")
    background = int(a[1, 1])
    _need(background != structure, "frame and padding must differ")
    structural = np.zeros(a.shape, dtype=bool)
    structural[:, [0, 3, width - 1]] = True
    structural[dividers, :] = True
    content = np.zeros(a.shape, dtype=bool)
    labels, lanes, key_order, key_targets = [], [], [], []
    for first in dividers[:-1]:
        row = first + 2
        rect = (slice(row, row + h), slice(5, width - 2))
        content[rect] = True
        content[row, 1:3] = True
        motif = a[rect]
        colors = [int(c) for c in np.unique(motif) if c != structure]
        _need(len(colors) == 1 and colors[0] != background, "binary lane identity")
        label = colors[0]
        _need(label not in labels and bool(np.any(motif == structure)), "unique lane label")
        labels.append(label)
        lane = _recover(motif, structure, label)
        lane.update(label=label, motif=motif.copy(), rectangle=rect)
        lanes.append(lane)
        key_order.append(int(a[row, 1]))
        key_targets.append(int(a[row, 2]))
    _need(bool(np.all(a[structural] == structure)), "damaged structural line")
    _need(bool(np.all(a[~(structural | content)] == background)), "damaged padding")
    _need(len(set(key_order)) == k and set(key_order) == set(labels), "key identities")
    _need(all(c != background for c in key_targets), "invisible target swatch")
    periods = {lane['period'] for lane in lanes}
    _need(len(periods) == 1, "lanes must share a stamp width")
    period = periods.pop()
    occupied = np.zeros((h, period), dtype=bool)
    for lane in lanes:
        _need(not bool(np.any(occupied & lane['folded'])), "folded fault collision")
        occupied |= lane['folded']
    return dict(h=h, period=period, k=k, length=length, background=background,
                structure=structure, labels=labels, lanes=lanes, occupied=occupied,
                key_order=key_order, key_targets=key_targets,
                targets=dict(zip(key_order, key_targets)))


def _parse(input_grid):
    a = _grid(input_grid)
    readings = []
    for transposed in (False, True):
        try:
            parsed = _parse_canonical(a.T if transposed else a)
        except ValueError:
            continue
        parsed['transposed'] = transposed
        readings.append(parsed)
    _need(len(readings) == 1, "expected one framed/key interpretation")
    return readings[0]


def _render(parsed, variant=None):
    h, p, bg = parsed['h'], parsed['period'], parsed['background']
    result = np.full((h * h, p * p), bg, dtype=np.int8)
    for index, lane in enumerate(parsed['lanes']):
        tile = lane['tile']
        if variant == 'first_copy':
            tile = lane['motif'][:, :p] == lane['label']
        if variant == 'key_position':
            target = parsed['key_targets'][index]
        elif variant == 'key_cyclic':
            shift = parsed['key_order'].index(parsed['labels'][0])
            target = parsed['key_targets'][(shift + index) % parsed['k']]
        else:
            target = parsed['targets'][lane['label']]
        for r, c in zip(*np.nonzero(lane['folded'])):
            missing = bool(lane['tile'][r, c])
            stamp = tile if missing or variant == 'always_positive' else ~tile
            if variant == 'solid':
                stamp = np.ones((h, p), dtype=bool)
            result[r*h:(r+1)*h, c*p:(c+1)*p] = np.where(stamp, target, bg)
    return result.T.copy() if parsed.get('transposed', False) else result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the fault atlas inferred entirely from the visible input."""
    return _render(_parse(input_grid))


def _encode(tiles, noisy, labels, targets, key_order, bg, structure):
    """Canonical framing with a two-dot identity/target key per lane slot."""
    h, length = noisy[0].shape
    k = len(tiles)
    a = np.full((1 + k * (h + 3), length + 7), bg, dtype=np.int8)
    a[:, [0, 3, a.shape[1] - 1]] = structure
    a[::h + 3, :] = structure
    for i, field in enumerate(noisy):
        row = i * (h + 3) + 2
        a[row:row+h, 5:-2] = field
        a[row, 1] = key_order[i]
        a[row, 2] = targets[key_order[i]]
    return a


def _primitive(tile, length):
    field = tile[:, np.arange(length) % tile.shape[1]]
    return not any(np.array_equal(field[:, q:], field[:, :-q])
                   for q in range(1, tile.shape[1]))


def _sample_tile(rng, h, p, groups, length):
    true_r, true_c = divmod(groups[0], p)
    false_r, false_c = divmod(groups[1], p)
    rr, cc = np.indices((h, p))
    for _ in range(32):
        style = int(rng.integers(7))
        if style == 0:
            tile = rng.random((h, p)) < rng.uniform(.2, .8)
        elif style == 1:
            tile = (rr + cc) % 2 == int(rng.integers(2))
        elif style == 2:
            tile = (rr == 0) | (rr == h-1) | (cc == 0) | (cc == p-1)
        elif style == 3:
            tile = cc == (rr + int(rng.integers(p))) % p
        elif style == 4:
            tile = np.broadcast_to(rng.integers(0, 2, p).astype(bool), (h, p)).copy()
        elif style == 5:
            tile = (rr <= int(rng.integers(h))) & (cc <= int(rng.integers(p)))
        else:
            tile = (rr == int(rng.integers(h))) | (cc == int(rng.integers(p)))
        tile[true_r, true_c] = True
        tile[false_r, false_c] = False
        if _primitive(tile, length):
            return tile
    # A one-hot row has primitive width p and satisfies the two forced bits.
    tile = np.zeros((h, p), dtype=bool)
    tile[true_r, true_c] = True
    return tile


def _ordered_positions(rng, h, p):
    coords = [(r, c) for r in range(h) for c in range(p)]
    style = int(rng.integers(5))
    if style == 0:
        return [int(x) for x in rng.permutation(h*p)]
    if style == 1:
        coords.sort(key=lambda x: (sum(x), x[0]))
    elif style == 2:
        cr, cc = int(rng.integers(h)), int(rng.integers(p))
        coords.sort(key=lambda x: (abs(x[0]-cr) + abs(x[1]-cc), x))
    elif style == 3:
        coords.sort(key=lambda x: ((sum(x) % 2), x))
    else:
        coords = [(r, c) for r in range(h)
                  for c in (range(p) if r % 2 == 0 else range(p-1, -1, -1))]
    flip_r, flip_c = bool(rng.integers(2)), bool(rng.integers(2))
    return [(h-1-r if flip_r else r)*p + (p-1-c if flip_c else c) for r, c in coords]


def _make_pair(rng, h, p, k, test, transposed, bg, structure, witness=False, sparse=False):
    train_cap = min(4, 23 // p - 1)
    copies = int(rng.integers(train_cap + 1, 23 // p + 1) if test
                 else rng.integers(3, train_cap + 1))
    length = copies*p + int(rng.integers(min(p-1, 23-copies*p) + 1))
    labels = [int(c) for c in rng.choice([c for c in range(10)
                                        if c not in (bg, structure)], k, replace=False)]
    if witness:
        # Distinct visible recolorings, without binding color numbers to roles.
        for _ in range(128):
            target_values = [int(c) for c in rng.choice([c for c in range(10) if c != bg],
                                                      k, replace=False)]
            if all(a != b for a, b in zip(labels, target_values)):
                break
        else:
            target_values = labels[1:] + labels[:1]
    else:
        target_values = [int(c) for c in rng.choice([c for c in range(10) if c != bg], k)]
    targets = dict(zip(labels, target_values))
    if witness:
        key_order = labels.copy()
        key_order[0], key_order[1] = key_order[1], key_order[0]
    else:
        key_order = [int(c) for c in rng.permutation(labels)]
    cap = min(4, max(2, h*length // 8))
    total = int(rng.integers(2*k, min(h*p-int(sparse), cap*k) + 1))
    sizes = [2] * k
    for _ in range(total - 2*k):
        choices = [j for j in range(k) if sizes[j] < cap]
        sizes[int(rng.choice(choices))] += 1
    order = _ordered_positions(rng, h, p)
    groups, offset = [], 0
    for size in sizes:
        groups.append(order[offset:offset+size])
        offset += size
    tiles, noisy = [], []
    expected = np.full((h*h, p*p), bg, dtype=np.int8)
    for label, locations in zip(labels, groups):
        tile = _sample_tile(rng, h, p, locations, length)
        tiles.append(tile)
        field = np.where(tile[:, np.arange(length) % p], label, structure).astype(np.int8)
        budget = max(2, field.size // 8)
        used_errors = len(locations)
        for j, location in enumerate(locations):
            r, c = divmod(location, p)
            occurrences = list(range(c, length, p))
            # First copy is deliberately imperfect; later copies also err.
            chosen = (occurrences[0] if j == 0 else occurrences[-1] if j == 1
                      else int(rng.choice(occurrences)))
            positions = [chosen]
            if len(occurrences) >= 5 and used_errors < budget and rng.random() < .3:
                positions.append(int(rng.choice([x for x in occurrences if x != chosen])))
                used_errors += 1
            for column in positions:
                field[r, column] = structure if tile[r, c] else label
        noisy.append(field)
        # Latent ground truth uses two signed Kronecker imprints.
        folded = np.zeros((h, p), dtype=bool)
        folded.flat[locations] = True
        expected[np.kron(folded & tile, tile)] = targets[label]
        expected[np.kron(folded & ~tile, ~tile)] = targets[label]
    a = _encode(tiles, noisy, labels, targets, key_order, bg, structure)
    pair = dict(input=a.T.copy() if transposed else a,
                output=expected.T.copy() if transposed else expected)
    parsed = _parse(pair['input'])
    _need(parsed['period'] == p and all(np.array_equal(x, lane['tile'])
          for x, lane in zip(tiles, parsed['lanes'])), "construction failed recovery")
    _need(np.array_equal(_render(parsed), pair['output']), "construction disagreement")
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    a, b = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(10)]
    presentation = int(rng.choice(3, p=[.12, .12, .76]))
    axes = ([False] * (a+b) if presentation == 0 else [True] * (a+b)
            if presentation == 1 else [bool(c) for c in rng.integers(2, size=a+b)])
    if presentation == 2:
        axes[1] = not axes[0]
    p0 = int(rng.choice([2, 3]))
    h0 = int(rng.integers(2, 5))
    k0 = int(rng.integers(1, min(2, (h0*p0-1)//2) + 1))
    p1 = 5 - p0
    h1 = int(rng.choice([h for h in range(2, 5) if h != h0 and h*p1 >= 6]))
    k1 = int(rng.integers(3, min(4, 29//(h1+3), h1*p1//2) + 1))
    schedule = [(h0, p0, k0), (h1, p1, k1)]
    for serial in range(2, a+b):
        h = int(rng.integers(2, 6 if serial >= a else 5))
        p = int(rng.integers(2, 6))
        k = int(rng.integers(1, min(4, 29//(h+3), h*p//2) + 1))
        schedule.append((h, p, k))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for serial, (h, p, k) in enumerate(schedule):
        split = 'train' if serial < a else 'test'
        bg, structure = palette[serial % 10], palette[(serial+1) % 10]
        for _ in range(96):
            try:
                pair = _make_pair(rng, h, p, k, split == 'test', axes[serial],
                                  bg, structure, witness=serial == 1, sparse=serial == 0)
            except ValueError:
                continue
            if _key(pair['input']) not in seen:
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("bounded unambiguous scene sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check geometry, inferred faults, signed blocks, evidence, and split shift."""
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "split schema")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, descriptors = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _parse(inp)
            h, p, bg = d['h'], d['period'], d['background']
            canonical = out.T if d['transposed'] else out
            _need(canonical.shape == (h*h, p*p), "atlas size")
            # Block checks are separate from the Kronecker construction.
            for r in range(h):
                for c in range(p):
                    owners = [lane for lane in d['lanes'] if lane['folded'][r, c]]
                    block = canonical[r*h:(r+1)*h, c*p:(c+1)*p]
                    if not owners:
                        _need(bool(np.all(block == bg)), "uncommanded stamp")
                    else:
                        _need(len(owners) == 1, "atlas ownership")
                        lane = owners[0]
                        mask = lane['tile'] if lane['tile'][r, c] else ~lane['tile']
                        target = d['targets'][lane['label']]
                        _need(bool(np.all(block[mask] == target)
                                   and np.all(block[~mask] == bg)), "wrong signed imprint")
            copies = d['length'] // p
            train_cap = min(4, 23//p - 1)
            _need(copies > train_cap if split == 'test' else 3 <= copies <= train_cap,
                  "repeat-count train/test shift")
            _need(all(np.any(lane['differences'][:, p:]) for lane in d['lanes']),
                  "no later-cycle fault evidence")
            descriptors[split].append(d)
    train = descriptors['train']
    _need(len({d['h'] for d in train}) >= 2 and len({d['period'] for d in train}) >= 2,
          "training must vary stamp geometry and period")
    _need(len({d['k'] for d in train}) >= 2, "training must vary lane count")
    _need(len({(d['background'], d['structure']) for d in train}) >= 2,
          "training must vary observable palette roles")
    _need(any(not d['occupied'].all() for d in train), "training needs blank macro-sites")
    for shortcut in ('first_copy', 'always_positive', 'key_position', 'key_cyclic', 'solid'):
        _need(any(not np.array_equal(_render(d), _render(d, shortcut)) for d in train),
              "training permits shortcut " + shortcut)


def self_test():
    """Bounded 42-seed variation checks plus concrete semantic regressions."""
    shapes, geometries, palettes, structures, signatures = set(), set(), set(), set(), set()
    heights, periods, counts, axes, styles = set(), set(), set(), set(), set()
    repeats, duplicates, partial, shared_targets = set(), 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(_grid(pair[role])) for s in ('train', 'test')
                             for pair in puzzle[s] for role in ('input', 'output')))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                before = inp.copy()
                d = _parse(inp)
                assert np.array_equal(solve(inp), pair['output'])
                assert np.array_equal(inp, before)
                shapes.add((inp.shape, pair['output'].shape))
                h, p = d['h'], d['period']
                heights.add(h); periods.add(p); counts.add(d['k']); axes.add(d['transposed'])
                repeats.add(d['length'] // p)
                palettes.add((d['background'], d['structure']))
                geometries.add((h, p, d['k'], d['length'], d['occupied'].tobytes()))
                normalized = inp.copy()
                roles = {}
                for i, color in enumerate(inp.flat):
                    normalized.flat[i] = roles.setdefault(int(color), len(roles))
                structures.add(_key(normalized))
                partial += d['length'] % p != 0
                shared_targets += len(set(d['key_targets'])) < d['k']
                for lane in d['lanes']:
                    styles.add((h, p, lane['tile'].tobytes()))
                    duplicates += lane['errors'] > int(lane['folded'].sum())
    assert len(signatures) == 42 and len(structures) >= 150
    assert heights == {2, 3, 4, 5} and periods == {2, 3, 4, 5}
    assert counts == {1, 2, 3, 4} and axes == {False, True}
    assert len(shapes) >= 50 and len(geometries) >= 150 and len(styles) >= 80
    assert len(palettes) >= 30 and len(repeats) >= 5
    assert duplicates >= 5 and partial >= 20 and shared_targets >= 5
    # Explicit hand-built signed/folding example; this expected image is not
    # obtained from the rendering helper or sampled latent parameters.
    tile = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    field = np.where(tile[:, np.arange(9) % 3], 2, 5).astype(np.int8)
    field[0, 0] = 5                # Missing ink.
    field[1, 8] = 2                # Surplus ink in a later repetition.
    inp = _encode([tile], [field], [2], {2: 8}, [2], 0, 5)
    answer = np.array([[int(c) for c in row] for row in
                       ('808000000', '080000000', '000000080', '000000808')])
    assert np.array_equal(solve(inp), answer)
    assert np.array_equal(solve(inp.T), answer.T)
    # Multiple faults at the same phase coordinate collapse to one imprint.
    field2 = np.where(tile[:, np.arange(15) % 3], 2, 5).astype(np.int8)
    field2[0, [0, 12]] = 5
    field2[1, 8] = 2
    assert np.array_equal(solve(_encode([tile], [field2], [2], {2: 8}, [2], 0, 5)), answer)
    # Same geometry, different source identity, independently permuted key.
    other = np.where(tile[:, np.arange(9) % 3], 3, 5).astype(np.int8)
    other[0, 2] = 5
    other[1, 6] = 3
    paired = _encode([tile, tile], [field, other], [2, 3], {2: 8, 3: 7}, [3, 2], 0, 5)
    paired_answer = np.array([[int(c) for c in row] for row in
                              ('808000707', '080000070', '070000080', '707000808')])
    assert np.array_equal(solve(paired), paired_answer)
    repeated = generate_puzzle(777, 10, 4)
    assert all(np.array_equal(x[r], y[r]) for s in repeated for x, y in
               zip(repeated[s], generate_puzzle(777, 10, 4)[s]) for r in ('input', 'output'))
    for a in range(2, 11):
        for b in range(1, min(4, a-1) + 1):
            validate_generated_puzzle(generate_puzzle(17, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{r: grid.copy() for r, grid in pair.items()} for pair in repeated[s]]
           for s in repeated}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('bad output accepted')
    bad_frame, bad_key = inp.copy(), inp.copy()
    bad_frame[0, 4] = 0
    bad_key[2, 1] = 9
    collision = _encode([tile, tile], [field, np.where(field == 2, 3, field)],
                        [2, 3], {2: 8, 3: 7}, [3, 2], 0, 5)
    tie = np.array([[1, 1, 0, 0, 0, 0, 1, 1]] * 2, dtype=np.int8)
    for function, arg in ((solve, bad_frame), (solve, bad_key), (solve, collision),
                          (lambda x: _recover(x, 0, 1), tie)):
        try:
            function(arg)
        except ValueError:
            pass
        else:
            raise AssertionError('malformed or tied input accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
