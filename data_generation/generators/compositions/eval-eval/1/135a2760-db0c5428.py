"""Broken rhythm emits echoes: periodic repair controls radial replication.

The eight upright 3x3 tiles are read clockwise from the direction indicated
inside their central hole. Their concatenation is a noisy 3x24 periodic ribbon.
Only tiles containing an error emit a repaired half-turn at doubled distance.
"""

from __future__ import annotations

import itertools
import numpy as np


# Clockwise, in screen coordinates. This is a geometric convention, not colors.
DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, 1),
              (1, 1), (1, 0), (1, -1), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a


def _key(a):
    return a.shape, tuple(map(int, a.flat))


def _d4(tile):
    return tuple(np.rot90(view, k) for view in (tile, tile[:, ::-1])
                 for k in range(4))


def _periodic_repair(ribbon):
    """Compare every strict-majority explanation having >=3 copies per phase.

    The error budget is nine of 72 cells. No generation-time error locations,
    selected period, or one-flip-per-phase assumption enter this function.
    Redundant periods are fine only when their complete repairs agree.
    """
    explanations = []
    for period in range(1, ribbon.shape[1] // 3 + 1):
        unit = np.empty((3, period), dtype=int)
        supported = True
        for row in range(3):
            for phase in range(period):
                values = ribbon[row, phase::period]
                colors, counts = np.unique(values, return_counts=True)
                if int(counts.max()) * 2 <= len(values):
                    supported = False
                    break
                unit[row, phase] = colors[np.argmax(counts)]
            if not supported:
                break
        if not supported:
            continue
        repaired = unit[:, np.arange(24) % period]
        errors = int(np.count_nonzero(repaired != ribbon))
        if errors <= ribbon.size // 8 and len(np.unique(unit)) >= 2:
            explanations.append((period, repaired))
    _require(bool(explanations), "ribbon has no supported periodic repair")
    clean = explanations[0][1]
    _require(all(np.array_equal(clean, e[1]) for e in explanations),
             "competing periods imply different repairs")
    return clean, explanations[0][0], tuple(e[0] for e in explanations)


def _analyze(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    cells = np.argwhere(a != bg)
    _require(len(cells) == 73, "expected a full eight-tile ring and one cue")
    top, left = map(int, cells.min(axis=0))
    bottom, right = map(int, cells.max(axis=0))
    _require((bottom - top, right - left) == (8, 8), "expected a 9x9 ring")
    _require(3 <= top <= a.shape[0] - 12 and
             3 <= left <= a.shape[1] - 12, "echo footprint would be clipped")
    cr, cc = top + 4, left + 4
    ring_mask = np.zeros(a.shape, dtype=bool)
    ring_mask[top:top + 9, left:left + 9] = True
    ring_mask[cr - 1:cr + 2, cc - 1:cc + 2] = False
    _require(bool(np.all(a[ring_mask] != bg)), "source tiles must be full")
    remainder = (a != bg) & ~ring_mask
    point, = np.argwhere(remainder)
    delta = tuple(map(int, point - (cr, cc)))
    _require(delta in DIRECTIONS, "cue must occupy a directional hole cell")
    start = DIRECTIONS.index(delta)
    cue = int(a[tuple(point)])
    colors, counts = np.unique(a[ring_mask], return_counts=True)
    _require(cue not in colors, "cue color must be absent from the ribbon")
    _require(int(counts.max()) > 36, "carrier must be a strict ring majority")
    carrier = int(colors[np.argmax(counts)])
    tiles = [a[cr + 3 * dr - 1:cr + 3 * dr + 2,
               cc + 3 * dc - 1:cc + 3 * dc + 2].copy()
             for dr, dc in DIRECTIONS]
    order = [(start + j) % 8 for j in range(8)]
    ribbon = np.concatenate([tiles[i] for i in order], axis=1)
    clean, period, supported = _periodic_repair(ribbon)
    clean_tiles = [None] * 8
    for j, i in enumerate(order):
        clean_tiles[i] = clean[:, 3 * j:3 * j + 3].copy()
    faults = tuple(i for i in range(8) if not np.array_equal(tiles[i], clean_tiles[i]))
    errors = int(np.count_nonzero(ribbon != clean))
    _require(1 <= errors <= 9, "at least one sparse error is required")
    _require(bg not in clean and cue not in clean, "repair introduced a role color")
    _require(np.count_nonzero(clean == carrier) > 36,
             "repaired ribbon must retain the common carrier")
    return dict(a=a, background=bg, carrier=carrier, cue=cue, center=(cr, cc),
                start=start, tiles=tiles, clean_tiles=clean_tiles, faults=faults,
                errors=errors, period=period, supported=supported, ribbon=ribbon,
                clean=clean, ring_mask=ring_mask)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver for every member of the specified family."""
    d = _analyze(input_grid)
    out = d['a'].copy()
    cr, cc = d['center']
    out[cr - 1:cr + 2, cc - 1:cc + 2] = d['carrier']
    for i, (dr, dc) in enumerate(DIRECTIONS):
        tile = d['clean_tiles'][i]
        out[cr + 3 * dr - 1:cr + 3 * dr + 2,
            cc + 3 * dc - 1:cc + 3 * dc + 2] = tile
        out[cr + dr, cc + dc] = tile[1 - dr, 1 - dc]
        if i in d['faults']:
            out[cr + 6 * dr - 1:cr + 6 * dr + 2,
                cc + 6 * dc - 1:cc + 6 * dc + 2] = tile[::-1, ::-1]
    return out


# Constructive backups for each period. These are small repeating units,
# not stored puzzles. 0=carrier; positive values name interchangeable details.
# Periods 4 and 5 have eight distinct D4 images for every 3-column window.
_BACKUP_UNITS = {
    2: ((0, 1), (0, 0), (1, 0)),
    3: ((1, 0, 0), (0, 2, 0), (0, 0, 1)),
    4: ((0, 1, 3, 0), (2, 0, 0, 1), (0, 0, 0, 0)),
    5: ((1, 0, 3, 2, 0), (1, 0, 0, 0, 0), (0, 2, 0, 0, 0)),
    6: ((1, 0, 0, 2, 0, 0), (0, 0, 3, 0, 0, 0), (0, 1, 0, 0, 2, 0)),
}


def _noise(rng, clean, period, start, fault_ids, extra, inward):
    noisy = clean.copy()
    used_groups, chosen = set(), []
    targets = list(fault_ids)
    rng.shuffle(targets)
    if inward:
        i = targets[0]
        dr, dc = DIRECTIONS[i]
        j = (i - start) % 8
        row, col = 1 - dr, 3 * j + 1 - dc
        chosen.append((row, col))
        used_groups.add((row, col % period))
        targets.remove(i)
    for i in targets:
        j = (i - start) % 8
        choices = [(r, c) for r in range(3) for c in range(3*j, 3*j+3)
                   if (r, c % period) not in used_groups]
        r, c = choices[int(rng.integers(len(choices)))]
        chosen.append((r, c))
        used_groups.add((r, c % period))
    # Extras are concentrated into existing faulty tiles, never new directions.
    for _ in range(extra):
        choices = [(r, c) for i in fault_ids for r in range(3)
                   for c in range(3*((i-start) % 8), 3*((i-start) % 8)+3)
                   if (r, c % period) not in used_groups]
        _require(bool(choices), "noise groups exhausted")
        r, c = choices[int(rng.integers(len(choices)))]
        chosen.append((r, c))
        used_groups.add((r, c % period))
    values, counts = np.unique(clean, return_counts=True)
    carrier = int(values[np.argmax(counts)])
    details = [int(v) for v in values if v != carrier]
    for r, c in chosen:
        noisy[r, c] = int(rng.choice(details)) if clean[r, c] == carrier else carrier
    return noisy


def _make_ribbon(rng, period, start, fault_ids, extra, inward,
                 carrier, details, witness):
    # Most units are sampled freely. The bounded backup search has one flip
    # in each requested tile plus the requested extras; it is rechecked too.
    for attempt in range(600):
        if attempt < 160:
            unit = np.full((3, period), carrier, dtype=int)
            amount = int(rng.integers(period, max(period + 1, (3*period*2)//5 + 1)))
            positions = rng.choice(3 * period, amount, replace=False)
            active_details = details[:int(rng.integers(1, min(4, amount) + 1))]
            for j, position in enumerate(positions):
                unit.flat[position] = active_details[j % len(active_details)]
        else:
            labels = np.array(_BACKUP_UNITS[period])
            mapping = [carrier] + details
            unit = np.asarray(mapping)[labels]
            # Change row order and phase without relying on a bank of grids.
            unit = np.roll(unit[rng.permutation(3)], int(rng.integers(period)), axis=1)
        clean = unit[:, np.arange(24) % period]
        if np.count_nonzero(clean == carrier) < 44:
            continue
        tiles = [clean[:, 3*j:3*j+3] for j in range(8)]
        if witness and any(len({v.tobytes() for v in _d4(tiles[(i-start) % 8])}) != 8
                           for i in fault_ids):
            continue
        noisy = _noise(rng, clean, period, start, fault_ids, extra, inward)
        try:
            recovered, shortest, _ = _periodic_repair(noisy)
        except AssertionError:
            continue
        if shortest == period and np.array_equal(recovered, clean):
            return noisy, clean
    # A bounded deterministic backup over noise proposals handles an unlucky RNG.
    # Extra noise remains visible but its specific locations are free to vary.
    labels = np.array(_BACKUP_UNITS[period])
    clean = np.asarray([carrier] + details)[labels][:, np.arange(24) % period]
    for backup_seed in range(2048):
        noisy = _noise(np.random.default_rng(backup_seed), clean, period, start,
                       fault_ids, extra, inward)
        try:
            recovered, shortest, _ = _periodic_repair(noisy)
        except AssertionError:
            continue
        if shortest == period and np.array_equal(recovered, clean):
            return noisy, clean
    raise RuntimeError("bounded periodic construction exhausted")


def _embed_pair(rng, noisy, clean, start, faults, bg, carrier, cue, test):
    # A moderate geometry shift leaves the same 15x15 construction intact.
    # Test canvases may be tight or elongated too; their error distribution
    # is the universal shift, rather than a universally larger canvas.
    if rng.random() < .16:
        h, w = (15, 30) if rng.random() < .5 else (30, 15)
    elif rng.random() < .12:
        h = w = 15
    else:
        low = 19 if test and rng.random() < .6 else 15
        h, w = map(int, rng.integers(low, 31, size=2))
    margins = [int(rng.integers(n - 14)) for n in (h, w)]
    cr, cc = margins[0] + 7, margins[1] + 7
    a = np.full((h, w), bg, dtype=int)
    b = a.copy()
    for j in range(8):
        i = (start + j) % 8
        dr, dc = DIRECTIONS[i]
        sl = (slice(cr + 3*dr - 1, cr + 3*dr + 2),
              slice(cc + 3*dc - 1, cc + 3*dc + 2))
        a[sl] = noisy[:, 3*j:3*j+3]
        b[sl] = clean[:, 3*j:3*j+3]
        if i in faults:
            b[cr + 6*dr - 1:cr + 6*dr + 2,
              cc + 6*dc - 1:cc + 6*dc + 2] = clean[:, 3*j:3*j+3][::-1, ::-1]
    dr, dc = DIRECTIONS[start]
    a[cr + dr, cc + dc] = cue
    b[cr - 1:cr + 2, cc - 1:cc + 2] = carrier
    for j in range(8):
        dr, dc = DIRECTIONS[(start + j) % 8]
        b[cr + dr, cc + dc] = clean[1-dr, 3*j + 1-dc]
    return {'input': a, 'output': b}


def _make_pair(rng, period, start, faults, extra, inward, witness, test):
    colors = list(map(int, rng.permutation(10)))
    bg, carrier, cue = colors[:3]
    noisy, clean = _make_ribbon(rng, period, start, faults, extra, inward,
                              carrier, colors[3:], witness)
    return _embed_pair(rng, noisy, clean, start, faults, bg, carrier, cue, test)


def _sample_candidates(train):
    """Learn nine local sampling hypotheses per direction from training labels."""
    positions = [set(range(9)) for _ in DIRECTIONS]
    nongrouped = False
    for pair in train:
        d = _analyze(pair['input'])
        output = np.asarray(pair['output'])
        cr, cc = d['center']
        samples = []
        for i, (dr, dc) in enumerate(DIRECTIONS):
            value = int(output[cr+dr, cc+dc])
            samples.append(value)
            positions[i] &= {j for j in range(9) if d['clean_tiles'][i].flat[j] == value}
        nongrouped |= (len({samples[i] for i in (0, 2, 4, 6)}) > 1 or
                       len({samples[i] for i in (1, 3, 5, 7)}) > 1)
    _require(all(positions), "training sampling rules contradict each other")
    return positions, nongrouped


def _sampling_supported(candidates, d):
    positions, nongrouped = candidates
    samples = []
    for i, (dr, dc) in enumerate(DIRECTIONS):
        tile = d['clean_tiles'][i]
        intended = int(tile[1-dr, 1-dc])
        if {int(tile.flat[j]) for j in positions[i]} != {intended}:
            return False
        samples.append(intended)
    query_nongrouped = (len({samples[i] for i in (0, 2, 4, 6)}) > 1 or
                        len({samples[i] for i in (1, 3, 5, 7)}) > 1)
    return nongrouped or not query_nongrouped


def _familiar_test(rng, train, faults, extra):
    """Finite fallback: reuse a demonstrated clean rhythm, with new damage.

    This guarantees sampling-coordinate agreement; geometry, corruption and
    literal palette still change. It does not consult any proposed test label.
    """
    d = _analyze(train[int(rng.integers(len(train)))]['input'])
    for noise_seed in range(2048):
        noisy = _noise(np.random.default_rng(noise_seed), d['clean'], d['period'],
                       d['start'], faults, extra, True)
        try:
            repaired, _, _ = _periodic_repair(noisy)
        except AssertionError:
            continue
        if np.array_equal(repaired, d['clean']):
            break
    else:
        raise RuntimeError("familiar rhythm noise proposals exhausted")
    mapping = rng.permutation(10)
    return _embed_pair(rng, mapping[noisy], mapping[d['clean']], d['start'], faults,
                       int(mapping[d['background']]), int(mapping[d['carrier']]),
                       int(mapping[d['cue']]), True)


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, 'num_train', 2, 10)
    m = _count(num_test, 'num_test', 1, 4)
    if n <= m:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    # Two query-independent witnesses cover all eight directions even at 2/1.
    partition = list(map(int, rng.permutation(8)))
    cut = int(rng.integers(3, 6))
    seams = list(map(int, rng.choice(8, 2, replace=False)))
    # A partial-cycle witness must move the seam away from NW.
    if seams[1] == 0:
        seams.reverse()
    plans = [(4, seams[0], partition[:cut], 0, True, True, False),
             (5, seams[1], partition[cut:], 1, True, True, False)]
    for _ in range(n - 2):
        count = int(rng.integers(3, 7))
        plans.append((int(rng.integers(2, 7)), int(rng.integers(8)),
                      list(map(int, rng.choice(8, count, replace=False))),
                      int(rng.integers(7 - count)), bool(rng.integers(2)), False, False))
    train = [_make_pair(rng, *p) for p in plans]
    candidates = _sample_candidates(train)
    test = []
    for _ in range(m):
        count = int(rng.integers(1, 3))
        faults = list(map(int, rng.choice(8, count, replace=False)))
        extra = int(rng.integers(1, min(4, 6-count) + 1))
        for _attempt in range(64):
            pair = _make_pair(rng, int(rng.integers(2, 7)), int(rng.integers(8)),
                              faults, extra, True, False, True)
            if _sampling_supported(candidates, _analyze(pair['input'])):
                break
        else:
            pair = _familiar_test(rng, train, faults, extra)
        test.append(pair)
    rng.shuffle(train)
    # A bounded palette permutation fallback guarantees distinct inputs.
    seen = set()
    for pair in train + test:
        if _key(pair['input']) in seen:
            for permutation in itertools.permutations(range(10), 3):
                roles = list(map(int, np.unique(pair['input'])))
                targets = list(permutation) + [c for c in range(10) if c not in permutation]
                mapping = np.arange(10)
                mapping[roles] = targets[:len(roles)]
                a = mapping[pair['input']]
                if _key(a) not in seen:
                    pair['input'], pair['output'] = a, mapping[pair['output']]
                    break
            else:
                raise AssertionError("distinctness fallback exhausted")
        seen.add(_key(pair['input']))
    puzzle = {'train': train, 'test': test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair")
    a, b = _grid(pair['input']), _grid(pair['output'])
    _require(a.shape == b.shape, "dimensions changed")
    d = _analyze(a)
    cr, cc = d['center']
    # Scalar correspondence verification, separate from solve's slice writes.
    assigned = d['ring_mask'].copy()
    assigned[cr-1:cr+2, cc-1:cc+2] = True
    _require(b[cr, cc] == d['carrier'], "wrong hub carrier")
    inward_witness = False
    for i, (dr, dc) in enumerate(DIRECTIONS):
        tile = d['clean_tiles'][i]
        _require(b[cr+dr, cc+dc] == tile[1-dr, 1-dc], "wrong inward sample")
        inward_witness |= tile[1-dr, 1-dc] != d['tiles'][i][1-dr, 1-dc]
        for r in range(3):
            for c in range(3):
                _require(b[cr+3*dr+r-1, cc+3*dc+c-1] == tile[r, c],
                         "source repair is wrong")
                y, x = cr+6*dr+1-r, cc+6*dc+1-c
                if i in d['faults']:
                    _require(b[y, x] == tile[r, c], "wrong echo orientation/content")
                    assigned[y, x] = True
                else:
                    _require(b[y, x] == d['background'], "clean tile emitted an echo")
    _require(np.array_equal(a[~assigned], b[~assigned]), "unused background changed")
    _require(np.count_nonzero(a != b) == d['errors'] + 9 + 9*len(d['faults']),
             "incorrect change count")
    _require(set(map(int, np.unique(b))) <= set(map(int, np.unique(a))), "new color")
    _require(d['cue'] not in b, "seam cue must be consumed")
    d['inward_witness'] = inward_witness
    return d


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "invalid splits")
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "illegal counts")
    seen, descriptors = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            d = _check_pair(pair)
            key = _key(d['a'])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            descriptors[split].append(d)
    train = descriptors['train']
    _require({4, 5} <= {d['period'] for d in train}, "missing rhythm witnesses")
    _require(len({d['start'] for d in train}) >= 2, "seam does not move in training")
    _require(any(d['period'] == 5 and d['start'] != 0 for d in train),
             "partial ribbon must demonstrate a non-NW seam")
    _require(any(d['inward_witness'] for d in train), "raw inward sampling shortcut survives")
    _require(any(d['errors'] > len(d['faults']) for d in train), "clustered damage is not taught")
    for direction in range(8):
        survivors = set(range(8))
        for d, pair in zip(train, puzzle['train']):
            if direction not in d['faults']:
                continue
            dr, dc = DIRECTIONS[direction]
            cr, cc = d['center']
            echo = np.asarray(pair['output'])[cr+6*dr-1:cr+6*dr+2, cc+6*dc-1:cc+6*dc+2]
            survivors &= {k for k, view in enumerate(_d4(d['clean_tiles'][direction]))
                          if np.array_equal(view, echo)}
        _require(survivors == {2}, "training must distinguish each directional half-turn")
    _require(all(len(d['faults']) >= 3 for d in train), "training damage should be distributed")
    _require(all(1 <= len(d['faults']) <= 2 and d['errors'] > len(d['faults'])
                 for d in descriptors['test']), "missing concentration shift")
    candidates = _sample_candidates(puzzle['train'])
    _require(all(_sampling_supported(candidates, d) for d in descriptors['test']),
             "training leaves a query-relevant sampling ambiguity")
    return None


def self_test():
    """Bounded 42-seed diversity, count, determinism and adversarial checks."""
    signatures, shapes, structures, palettes, starts, periods = set(), set(), set(), set(), set(), set()
    fault_counts, error_counts, layouts = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p[k]) for s in ('train', 'test') for p in puzzle[s]
                             for k in ('input', 'output')))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _check_pair(pair)
                predicted = solve(np.asarray(pair['input']).copy())
                _require(np.array_equal(predicted, pair['output']), "solve disagrees")
                shapes.add(d['a'].shape)
                layouts.add((d['a'].shape, d['center'], d['start']))
                roles = {}
                structures.add(tuple(roles.setdefault(int(v), len(roles)) for v in d['ribbon'].flat))
                palettes.add((d['background'], d['carrier'], d['cue']))
                starts.add(d['start'])
                periods.add(d['period'])
                fault_counts.add(len(d['faults']))
                error_counts.add(d['errors'])
    _require(len(signatures) == 42 and len(structures) >= 150, "structural variation collapsed")
    _require(len(shapes) >= 70 and len(layouts) >= 150 and len(palettes) >= 120,
             "geometry or roles do not vary enough")
    _require(starts == set(range(8)) and periods == set(range(2, 7)), "seam/period coverage gap")
    _require(fault_counts == set(range(1, 7)) and error_counts == set(range(2, 7)),
             "error/echo count coverage gap")
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(1000 + 10*n + m, n, m)
            _require((len(p['train']), len(p['test'])) == (n, m), "count request ignored")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (True, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first, second = generate_puzzle(314159, 2, 1), generate_puzzle(314159, 2, 1)
    _require(all(np.array_equal(p[k], q[k]) for s in first
                 for p, q in zip(first[s], second[s]) for k in p), "nondeterminism")
    pair = first['test'][0]
    original = pair['input'].copy()
    solve(pair['input'])
    _require(np.array_equal(original, pair['input']), "solve mutates its input")
    # A wrong ring repair, hub sample, echo cell, or unused cell must be rejected.
    d = _analyze(pair['input'])
    cr, cc = d['center']
    i = d['faults'][0]
    dr, dc = DIRECTIONS[i]
    points = [(cr+3*dr, cc+3*dc), (cr+dr, cc+dc),
              (cr+6*dr, cc+6*dc), (cr-6, cc-3)]
    for point in points:
        broken = {k: v.copy() for k, v in pair.items()}
        broken['output'][point] = (int(broken['output'][point]) + 1) % 10
        try:
            _check_pair(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted incorrect output")
    return {'seeds': 42, 'puzzles': len(signatures), 'shapes': len(shapes),
            'ribbon_structures': len(structures), 'layouts': len(layouts),
            'role_assignments': len(palettes), 'periods': sorted(periods),
            'seams': sorted(starts), 'echo_counts': sorted(fault_counts),
            'error_counts': sorted(error_counts)}


if __name__ == '__main__':
    print(self_test())
