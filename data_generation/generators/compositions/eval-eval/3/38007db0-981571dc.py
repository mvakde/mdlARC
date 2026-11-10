"""Decode reflection-symmetric messages written as changes to an odd panel.

No file access, external state or training labels are used by solve. Construction
creates complete messages first; outputs are not obtained by calling solve.
"""
from collections import Counter
from hashlib import sha256
from itertools import permutations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _count(name, value, low, high):
    _require(isinstance(value, (int, np.integer)) and
             not isinstance(value, (bool, np.bool_)), name + " must be an integer")
    value = int(value)
    _require(low <= value <= high, name + " outside legal range")
    return value


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= x <= 30 for x in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must have integer cells")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "colors must be 0..9")
    return grid


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _orbits(height, width):
    return [tuple(sorted({(r, c), (height - 1 - r, c),
                          (r, width - 1 - c),
                          (height - 1 - r, width - 1 - c)}))
            for r in range((height + 1) // 2)
            for c in range((width + 1) // 2)]


def _assemble(rows, lattice):
    bands, copies = len(rows), len(rows[0])
    ph, pw = rows[0][0].shape
    result = np.full((bands * (ph + 1) + 1, copies * (pw + 1) + 1),
                     lattice, dtype=np.int8)
    for b, row in enumerate(rows):
        for c, panel in enumerate(row):
            r0, c0 = 1 + b * (ph + 1), 1 + c * (pw + 1)
            result[r0:r0 + ph, c0:c0 + pw] = panel
    return result


def _parse(value):
    grid = _grid(value)
    lattice = int(grid[0, 0])
    rs = np.flatnonzero(np.all(grid == lattice, axis=1)).tolist()
    cs = np.flatnonzero(np.all(grid == lattice, axis=0)).tolist()
    _require(len(rs) >= 2 and len(cs) >= 2, "missing lattice boundaries")
    _require(rs[0] == cs[0] == 0 and rs[-1] == grid.shape[0] - 1 and
             cs[-1] == grid.shape[1] - 1, "incomplete outer border")
    dr, dc = rs[1], cs[1]
    _require(dr >= 3 and dc >= 3, "panels must be at least 2x2")
    _require(all(b - a == dr for a, b in zip(rs, rs[1:])) and
             all(b - a == dc for a, b in zip(cs, cs[1:])), "irregular lattice")
    panels = [[grid[r + 1:s, c + 1:d] for c, d in zip(cs, cs[1:])]
              for r, s in zip(rs, rs[1:])]
    _require(all(not np.any(p == lattice) for row in panels for p in row),
             "lattice color inside panel")
    _require(len(np.unique(grid)) >= 3, "need at least two payload colors")
    return dict(lattice=lattice, ph=dr - 1, pw=dc - 1,
                bands=len(panels), copies=len(panels[0]), panels=panels)


def _reference(info):
    _require(info['copies'] >= 3, "need at least three copies per band")
    counts, representatives = Counter(), {}
    for row in info['panels']:
        for panel in row:
            key = _key(panel)
            counts[key] += 1
            representatives[key] = panel
    maximum = max(counts.values())
    winners = [k for k, v in counts.items() if v == maximum]
    _require(len(winners) == 1, "ambiguous reference")
    master = representatives[winners[0]]
    _require(maximum == info['bands'] * (info['copies'] - 1),
             "incorrect reference multiplicity")
    return master


def _decode(master, variant):
    changed = master != variant
    result = np.empty(master.shape, dtype=np.int8)
    for orbit in _orbits(*master.shape):
        values = {int(variant[p]) for p in orbit if changed[p]}
        _require(len(values) == 1, "message orbit empty or contradictory")
        color = next(iter(values))
        for p in orbit:
            result[p] = color
    return result


def _analyze_input(input_grid):
    info = _parse(input_grid)
    master = _reference(info)
    variants, columns, messages = [], [], []
    for row in info['panels']:
        odd = [c for c, p in enumerate(row) if not np.array_equal(p, master)]
        _require(len(odd) == 1, "every band needs exactly one visible exception")
        columns.append(odd[0])
        variant = row[odd[0]]
        variants.append(variant)
        messages.append(_decode(master, variant))
    info.update(master=master, variants=variants, columns=columns, messages=messages)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only reference subtraction followed by two-fold completion."""
    info = _analyze_input(input_grid)
    return _assemble([[message] for message in info['messages']], info['lattice'])


def _draw_master(rng, shape, colors, balanced=False):
    h, w = shape
    orbits = _orbits(h, w)
    if balanced:
        master = np.empty(shape, dtype=np.int8)
        for orbit in orbits:
            values = np.array([colors[0], colors[0], colors[1], colors[1]], dtype=np.int8)
            rng.shuffle(values)
            for p, value in zip(orbit, values):
                master[p] = value
    else:
        k = len(colors)
        rr, cc = np.indices(shape)
        style = int(rng.integers(5))
        if style == 0:
            labels = rng.integers(k, size=shape)
        elif style == 1:
            labels = (rr // int(rng.integers(1, 3)) +
                      cc // int(rng.integers(1, 4)) + int(rng.integers(k))) % k
        elif style == 2:
            labels = (rr * int(rng.choice([-2, -1, 1, 2])) + cc) % k
        elif style == 3:
            labels = np.maximum(abs(rr - int(rng.integers(h))),
                                abs(cc - int(rng.integers(w)))) % k
        else:
            labels = np.zeros(shape, dtype=int)
            for _ in range(int(rng.integers(2, 6))):
                a, b = sorted(map(int, rng.integers(h, size=2)))
                c, d = sorted(map(int, rng.integers(w, size=2)))
                value = int(rng.integers(k))
                labels[a:b + 1, c] = value
                labels[a:b + 1, d] = value
                labels[a, c:d + 1] = value
                labels[b, c:d + 1] = value
        noise = rng.random(shape) < float(rng.uniform(0.02, 0.16))
        labels[noise] = rng.integers(k, size=int(noise.sum()))
        master = np.array(colors, dtype=np.int8)[labels]
        # Payload participation, rather than a hidden fixed color dictionary.
        for index, color in zip(rng.permutation(h * w), colors):
            master.flat[int(index)] = color
    # A variable-location diagonal pair guarantees a real two-fold repair.
    four = [o for o in orbits if len(o) == 4]
    anchor = four[int(rng.integers(len(four)))]
    a, b = map(int, rng.choice(colors, size=2, replace=False))
    for p, value in zip(anchor, (a, b, b, a)):
        master[p] = value
    if not balanced:
        # The anchor might have covered a rare payload; restore it elsewhere.
        free = [p for p in np.ndindex(shape) if p not in anchor]
        for p, color in zip(free, colors):
            master[p] = color
    return master


def _message(rng, master, colors, balanced=False):
    orbits = _orbits(*master.shape)
    k = len(colors)
    if balanced:
        targets = np.array([colors[0]] * (len(orbits) // 2) +
                           [colors[1]] * (len(orbits) // 2), dtype=np.int8)
        rng.shuffle(targets)
    else:
        style = int(rng.integers(5))
        targets = []
        sh, sw = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        for orbit in orbits:
            r, c = orbit[0]
            label = (int(rng.integers(k)) if style == 0 else
                     (r // sh + c // sw) % k if style == 1 else
                     max(r, c) % k if style == 2 else
                     (r + 2 * c) % k if style == 3 else
                     (r // sh + 2 * (c // sw)) % k)
            color = int(colors[label])
            if all(master[p] == color for p in orbit):
                options = [int(x) for x in colors if any(master[p] != x for p in orbit)]
                color = options[int(rng.integers(len(options)))]
            targets.append(color)
        if len(set(map(int, targets))) == 1:
            for i, orbit in enumerate(orbits):
                alternatives = [int(x) for x in colors if x != targets[i] and
                                any(master[p] != x for p in orbit)]
                if alternatives:
                    targets[i] = alternatives[int(rng.integers(len(alternatives)))]
                    break
    result = np.empty(master.shape, dtype=np.int8)
    for orbit, color in zip(orbits, targets):
        for p in orbit:
            result[p] = color
    return result


def _write_message(rng, master, message, sparse=False, balanced=False, single_orbit=None):
    variant = master.copy()
    orbits = _orbits(*master.shape)
    eligible = [[p for p in orbit if master[p] != message[p]] for orbit in orbits]
    # Pick a diagonal pair of different reference cells, and write only one.
    hard = []
    h, w = master.shape
    for i, (orbit, cells) in enumerate(zip(orbits, eligible)):
        if len(orbit) == 4:
            for p in cells:
                opposite = h - 1 - p[0], w - 1 - p[1]
                if opposite in cells:
                    hard.append((i, p))
    _require(bool(hard), "construction lacks a diagonal witness orbit")
    pivot, witness = hard[int(rng.integers(len(hard)))]
    for i, cells in enumerate(eligible):
        if i == pivot:
            chosen = [witness]
        else:
            count = 1 if balanced or i == single_orbit or (sparse and rng.random() < 0.85) else int(rng.integers(1, len(cells) + 1))
            chosen = [cells[int(j)] for j in rng.choice(len(cells), count, replace=False)]
        for p in chosen:
            variant[p] = message[p]
    return variant


def _geometry(rng, test=False, witness=False, avoid_shape=None, contrast=False):
    if witness:
        shapes = [(4, 4), (4, 6), (6, 4)]
        ph, pw = shapes[int(rng.integers(len(shapes)))]
    else:
        shapes = [(h, w) for h in (range(7, 9) if test else range(3, 7))
                  for w in (range(3, 9) if test else range(3, 7))
                  if (h, w) != avoid_shape and (not contrast or (h // 2) * (w // 2) >= 2)]
        ph, pw = shapes[int(rng.integers(len(shapes)))]
    bands = int(rng.integers(2 if witness else 1,
                             min(4 if witness else 7, 29 // (ph + 1)) + 1))
    copies = int(rng.integers(3, min(6, 29 // (pw + 1)) + 1))
    return ph, pw, bands, copies


def _make_pair(rng, geometry, witness=False, test=False, contrast=False):
    ph, pw, bands, copies = geometry
    k = 2 if witness else int(rng.integers(2, 5))
    palette = list(map(int, rng.choice(10, k + 1, replace=False)))
    lattice, colors = palette[0], palette[1:]
    master = _draw_master(rng, (ph, pw), colors, balanced=witness)
    contrast_index = None
    if contrast:
        orbits = _orbits(ph, pw)
        four = [i for i, orbit in enumerate(orbits) if len(orbit) == 4]
        anchor_index, contrast_index = map(int, rng.choice(four, size=2, replace=False))
        a, b = map(int, rng.choice(colors, size=2, replace=False))
        for p, color in zip(orbits[anchor_index], (a, b, b, a)):
            master[p] = color
        for p in orbits[contrast_index]:
            master[p] = a
        free = [p for p in np.ndindex(master.shape)
                if p not in orbits[anchor_index] and p not in orbits[contrast_index]]
        for p, color in zip(free, colors):
            master[p] = color
    messages, variants = [], []
    for _ in range(bands):
        message = _message(rng, master, colors, balanced=witness)
        if contrast:
            for p in orbits[contrast_index]:
                message[p] = b
        if witness and any(np.array_equal(message, previous) for previous in messages):
            # Enumerate balanced subsets: at least six choices for <=4 bands.
            from itertools import combinations
            orbits = _orbits(ph, pw)
            for subset in combinations(range(len(orbits)), len(orbits) // 2):
                candidate = np.full((ph, pw), colors[1], dtype=np.int8)
                for i in subset:
                    for p in orbits[i]:
                        candidate[p] = colors[0]
                if all(not np.array_equal(candidate, old) for old in messages):
                    message = candidate
                    break
        messages.append(message)
        variants.append(_write_message(rng, master, message, sparse=test, balanced=witness,
                                       single_orbit=contrast_index))
    columns = list(map(int, rng.integers(copies, size=bands)))
    if witness and len(set(columns)) == 1:
        index = int(rng.integers(bands))
        columns[index] = (columns[index] + int(rng.integers(1, copies))) % copies
    rows = []
    for column, variant in zip(columns, variants):
        row = [master.copy() for _ in range(copies)]
        row[column] = variant
        rows.append(row)
    return {'input': _assemble(rows, lattice),
            'output': _assemble([[m] for m in messages], lattice)}


def _unique(pair, seen):
    if _key(pair['input']) not in seen:
        return pair
    colors = np.unique(pair['input']).tolist()
    for palette in permutations(range(10), len(colors)):
        lookup = np.arange(10, dtype=np.int8)
        lookup[colors] = palette
        candidate = {role: lookup[grid] for role, grid in pair.items()}
        if _key(candidate['input']) not in seen:
            return candidate
    raise AssertionError("at least 720 palettes cannot be exhausted by 14 inputs")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    _require(nt > ne, "num_train must exceed num_test")
    _require(isinstance(seed, (int, np.integer)) and
             not isinstance(seed, (bool, np.bool_)) and int(seed) >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    witness_geometry = _geometry(rng, witness=True)
    geometries = [witness_geometry, _geometry(rng, avoid_shape=witness_geometry[:2], contrast=True)]
    geometries += [_geometry(rng) for _ in range(nt - 2)]
    train = [_make_pair(rng, geometry, witness=i == 0, contrast=i == 1)
             for i, geometry in enumerate(geometries)]
    rng.shuffle(train)
    puzzle, seen = {'train': [], 'test': []}, set()
    for pair in train:
        pair = _unique(pair, seen)
        seen.add(_key(pair['input']))
        puzzle['train'].append(pair)
    for _ in range(ne):
        pair = _unique(_make_pair(rng, _geometry(rng, test=True), test=True), seen)
        seen.add(_key(pair['input']))
        puzzle['test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _histogram(panel):
    return tuple(map(int, np.bincount(panel.ravel(), minlength=10)))


def _features(info):
    master = info['master']
    h, w = master.shape
    diagonal_repairs = []
    densities, equal_counts, overruled_majorities = [], [], []
    for variant, message in zip(info['variants'], info['messages']):
        changed = master != variant
        densities.append(float(changed.mean()))
        equal_counts.append(_histogram(master) == _histogram(variant))
        minority_orbits = 0
        for orbit in _orbits(h, w):
            frequencies = Counter(int(variant[p]) for p in orbit)
            color, count = frequencies.most_common(1)[0]
            minority_orbits += int(count > len(orbit) / 2 and color != message[orbit[0]])
        overruled_majorities.append(minority_orbits)
        repairs = 0
        for r, c in np.ndindex(master.shape):
            if (not changed[r, c] and variant[r, c] != message[r, c] and
                    not changed[h - 1 - r, c] and not changed[r, w - 1 - c] and
                    changed[h - 1 - r, w - 1 - c]):
                repairs += 1
        diagonal_repairs.append(repairs)
    # A reference-only orbit-majority reconstruction is a concrete ablation.
    baseline = np.empty(master.shape, dtype=np.int8)
    for orbit in _orbits(h, w):
        frequencies = Counter(int(master[p]) for p in orbit)
        value = min(frequencies, key=lambda c: (-frequencies[c], c))
        for p in orbit:
            baseline[p] = value
    return dict(diagonal_repairs=diagonal_repairs, densities=densities,
                equal_counts=equal_counts,
                overruled_majorities=overruled_majorities,
                distinct_messages=len({_key(m) for m in info['messages']}),
                master_only_wrong=sum(not np.array_equal(baseline, m) for m in info['messages']))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _count('num_train', nt, 2, 10)
    _count('num_test', ne, 1, 4)
    _require(nt > ne, "illegal split counts")
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "bad pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze_input(inp)
            expected = _assemble([[m] for m in info['messages']], info['lattice'])
            _require(np.array_equal(out, expected), "incorrect message or output framing")
            _require(out.shape == (inp.shape[0], info['pw'] + 2), "bad output dimensions")
            for variant, message in zip(info['variants'], info['messages']):
                changed = variant != info['master']
                _require(np.array_equal(variant[changed], message[changed]), "changed witness not preserved")
                _require(np.array_equal(message, message[::-1]) and
                         np.array_equal(message, message[:, ::-1]), "message lacks both folds")
            features = _features(info)
            _require(all(n > 0 for n in features['diagonal_repairs']),
                     "each generated band must need an actual diagonal-only repair")
            records[split].append((info, features))
    training = records['train']
    _require(any(any(f['equal_counts']) for _, f in training), "missing equal-count comparison evidence")
    _require(any(f['distinct_messages'] > 1 and len(set(i['columns'])) > 1
                 for i, f in training), "missing multi-position / different-message evidence")
    _require(any(f['master_only_wrong'] > 0 for _, f in training), "reference-only shortcut not distinguished")
    _require(any(any(f['overruled_majorities']) for _, f in training),
             "must demonstrate a message witness overruling a visible orbit majority")
    _require(len({(i['ph'], i['pw']) for i, _ in training}) > 1, "need training geometry variation")
    _require(min(i['ph'] for i, _ in records['test']) > max(i['ph'] for i, _ in training),
             "generated tests must expand panel height")


def _normalized_key(grid):
    mapping = {}
    return grid.shape, bytes(mapping.setdefault(int(x), len(mapping)) for x in grid.flat)


def _puzzle_digest(puzzle):
    digest = sha256()
    for split in ('train', 'test'):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded, streaming variation checks; creates no files or execution traces."""
    signatures, shapes, geometry, structures, masters, messages = set(), set(), set(), set(), set(), set()
    palettes, parity, payload_counts = set(), set(), set()
    bands_checked = pairs_checked = diagonal_cells = equal_count_bands = repeat_column_pairs = 0
    overruled_majority_orbits = 0
    min_density, max_density = 1.0, 0.0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        _require(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed)), "nondeterministic seed")
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                before = inp.copy()
                _require(np.array_equal(solve(inp), out), "solve disagrees")
                _require(np.array_equal(inp, before), "solve mutated input")
                # Nested-list inputs carry no generation metadata.
                _require(np.array_equal(solve(inp.tolist()), out), "list input failed")
                info = _analyze_input(inp)
                features = _features(info)
                pairs_checked += 1
                bands_checked += info['bands']
                diagonal_cells += sum(features['diagonal_repairs'])
                equal_count_bands += sum(features['equal_counts'])
                overruled_majority_orbits += sum(features['overruled_majorities'])
                repeat_column_pairs += len(set(info['columns'])) < info['bands']
                min_density = min(min_density, *features['densities'])
                max_density = max(max_density, *features['densities'])
                shapes.add(inp.shape)
                geometry.add((info['ph'], info['pw'], info['bands'], info['copies']))
                structures.add(_normalized_key(inp))
                masters.add(_normalized_key(info['master']))
                messages.update(_normalized_key(m) for m in info['messages'])
                palettes.add(tuple([info['lattice']] + sorted(set(map(int, np.unique(inp))) - {info['lattice']})))
                payload_counts.add(len(np.unique(inp)) - 1)
                parity.add((info['ph'] % 2, info['pw'] % 2))
    _require(len(signatures) == 42, "cross-seed collapse")
    _require(len(shapes) >= 20 and len(geometry) >= 60, "geometry diversity collapsed")
    _require(len(structures) >= 180 and len(masters) >= 90, "bitmap diversity collapsed")
    _require(len(messages) >= 100 and len(palettes) >= 80, "message/palette diversity collapsed")
    _require(payload_counts == {2, 3, 4} and len(parity) == 4, "payload/parity coverage missing")
    _require(equal_count_bands >= 84 and repeat_column_pairs >= 20, "comparison evidence collapsed")
    _require(overruled_majority_orbits >= 42, "orbit-majority counterevidence missing")
    _require(min_density <= 0.25 and max_density >= 0.45, "witness density variation collapsed")
    # All 30 legal explicit count combinations, one puzzle in memory per call.
    count_cases = 0
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(10000 + 10 * nt + ne, nt, ne)
            _require((len(p['train']), len(p['test'])) == (nt, ne), "count mismatch")
            count_cases += 1
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exercise the same rule beyond normal sampler heights and palettes.
    rng = np.random.default_rng(923)
    for shape in ((12, 6), (9, 5), (2, 8)):
        colors = [1, 2, 3, 4, 5, 6]
        master = _draw_master(rng, shape, colors)
        message = _message(rng, master, colors)
        variant = _write_message(rng, master, message, sparse=True)
        inp = _assemble([[master, variant, master]], 0)
        _require(np.array_equal(solve(inp), _assemble([[message]], 0)), "broader member failed")
    # Identical exceptional bitmap, different references, different answers.
    # This isolates A's role in defining evidence even after selection.
    odd = np.array([[1, 2], [2, 1]], dtype=np.int8)
    for background in (1, 2):
        master = np.full((2, 2), background, dtype=np.int8)
        inp = _assemble([[master, odd, master]], 0)
        message = np.full((2, 2), 3 - background, dtype=np.int8)
        _require(np.array_equal(solve(inp), _assemble([[message]], 0)),
                 "reference-relative evidence ignored")
    base = generate_puzzle(0)
    for pair in base['train'] + base['test']:
        inp, out = pair['input'], pair['output']
        # These are noncontiguous views. Horizontal reflection only relocates
        # the odd columns; vertical reflection also reverses band order.
        _require(np.array_equal(solve(inp[:, ::-1]), out), "column-position dependence")
        _require(np.array_equal(solve(inp[::-1]), out[::-1]), "band-order dependence")
        lookup = np.array([9, 5, 2, 7, 0, 4, 8, 1, 6, 3], dtype=np.int8)
        _require(np.array_equal(solve(lookup[inp]), lookup[out]), "literal-color dependence")
    for defect in ('output', 'border', 'empty_orbit', 'second_exception', 'contradiction'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in base[s]] for s in base}
        pair = bad['train'][0]
        info = _analyze_input(pair['input'])
        if defect == 'output':
            pair['output'][1, 1] = (int(pair['output'][1, 1]) + 1) % 10
        elif defect == 'border':
            pair['input'][0, 1] = (info['lattice'] + 1) % 10
        else:
            column = info['columns'][0]
            c0 = 1 + column * (info['pw'] + 1)
            target = pair['input'][1:1 + info['ph'], c0:c0 + info['pw']]
            if defect == 'empty_orbit':
                for point in _orbits(info['ph'], info['pw'])[0]:
                    target[point] = info['master'][point]
            elif defect == 'second_exception':
                other = (column + 1) % info['copies']
                start = 1 + other * (info['pw'] + 1)
                pair['input'][1:1 + info['ph'], start:start + info['pw']] = target.copy()
            else:
                orbit = next(o for o in _orbits(info['ph'], info['pw']) if len(o) == 4)
                # A third color distinct from the reference and first witness.
                a, b = orbit[:2]
                available = [x for x in range(10) if x not in
                             {info['lattice'], int(info['master'][a]), int(info['master'][b])}]
                target[a], target[b] = available[0], available[1]
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("validator accepted " + defect)
    return dict(seeds=42, pairs=pairs_checked, bands=bands_checked,
                distinct_puzzles=len(signatures), input_shapes=len(shapes),
                geometries=len(geometry), normalized_inputs=len(structures),
                normalized_references=len(masters), normalized_messages=len(messages),
                palettes=len(palettes), payload_cardinalities=sorted(payload_counts),
                fold_parities=len(parity), equal_count_bands=equal_count_bands,
                repeated_column_pairs=repeat_column_pairs, diagonal_only_repairs=diagonal_cells,
                overruled_majority_orbits=overruled_majority_orbits,
                witness_density_range=[min_density, max_density], legal_count_cases=count_cases,
                negative_cases=5, reference_counterfactuals=2, metamorphic_cases=15)
