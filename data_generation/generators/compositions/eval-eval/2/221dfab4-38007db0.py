"""Resonant exception scanner: periodic observation gates template comparison.

Only standard-library modules and NumPy are used. The solver reads its complete
convention from the input; sampling preferences never enter the solution.
"""

from collections import Counter
from itertools import permutations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= a) & (a <= 9)), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _uncanonical(a, rotation, reflected):
    if reflected:
        a = a[:, ::-1]
    return np.rot90(a, -rotation).copy()


def _read_canonical(g):
    """Read the visible clock, lattice and baseline in a proposed orientation."""
    carrier = int(g[0, -1])
    cap_positions = np.flatnonzero(g[0] != carrier)
    length = len(cap_positions)
    _require(3 <= length <= 5, "clock length")
    _require(np.array_equal(cap_positions, np.arange(length)), "clock location")
    values, counts = np.unique(g[0, :length], return_counts=True)
    _require(len(values) == 2 and sorted(counts.tolist()) == [1, length - 1],
             "one bead in an ordinary bar")
    special = int(values[np.argmin(counts)])
    ordinary = int(values[np.argmax(counts)])
    bead = int(np.flatnonzero(g[0, :length] == special)[0])
    _require(0 < bead < length - 1, "bead must be interior")
    _require(np.count_nonzero(g == special) == 1 and
             np.count_nonzero(g == ordinary) == length - 1, "reserved clock colors")
    _require(np.all(g[1:, :length + 1] == carrier), "blank rail and gap")
    block = g[1:, length + 1:]
    _require(min(block.shape) >= 2, "lattice rectangle")
    lattice = int(block[0, 0])
    _require(lattice not in (carrier, ordinary, special), "distinct roles")
    rs = np.flatnonzero(np.all(block == lattice, axis=1))
    cs = np.flatnonzero(np.all(block == lattice, axis=0))
    _require(len(rs) >= 2 and len(cs) >= 4, "lattice separators")
    _require(rs[0] == cs[0] == 0 and rs[-1] == len(block) - 1 and
             cs[-1] == block.shape[1] - 1, "complete lattice frame")
    _require(np.all(np.diff(rs) == rs[1]) and np.all(np.diff(cs) == cs[1]),
             "regular lattice")
    ph, pw = int(rs[1] - 1), int(cs[1] - 1)
    bands, copies = len(rs) - 1, len(cs) - 1
    _require(4 <= ph <= 12 and 3 <= pw <= 7 and 1 <= bands <= 5 and
             3 <= copies <= 6, "family geometry")
    panels = [[block[r + 1:s, c + 1:d].copy()
               for c, d in zip(cs, cs[1:])] for r, s in zip(rs, rs[1:])]
    all_colors = set(map(int, np.unique(g)))
    motif_colors = all_colors - {carrier, lattice, ordinary, special}
    _require(len(all_colors) == 5 and len(motif_colors) == 1, "five color roles")
    motif = next(iter(motif_colors))
    for row in panels:
        for panel in row:
            _require(np.all(np.isin(panel, [carrier, motif])), "panel colors")
            _require(np.all(panel[[0, -1]] == carrier) and
                     np.all(panel[:, [0, -1]] == carrier), "carrier perimeter")
    frequency = Counter(_key(p) for row in panels for p in row)
    maximum = max(frequency.values())
    modes = [key for key, count in frequency.items() if count == maximum]
    _require(len(modes) == 1, "unique repeated master")
    master = next(p for row in panels for p in row if _key(p) == modes[0]).copy()
    _require(np.any(master == motif), "master must contain motif")
    winners, masks = [], []
    for b, row in enumerate(panels):
        raw = [p for p in row if not np.array_equal(p, master)]
        _require(len(raw) <= 2 and copies - len(raw) >= 2, "repeated copies per band")
        _require(len({_key(p) for p in raw}) == len(raw), "distinct raw variants")
        distances = 2 + b * (ph + 1) + np.arange(ph)
        mask = distances % 6 == 4
        resonant = [c for c, p in enumerate(row) if np.any(p[mask] != master[mask])]
        _require(len(resonant) <= 1, "at most one resonant candidate")
        winners.append(resonant[0] if resonant else -1)
        masks.append(mask)
    return dict(g=g, length=length, bead=bead, carrier=carrier, lattice=lattice,
                motif=motif, ordinary=ordinary, special=special, ph=ph, pw=pw,
                bands=bands, copies=copies, panels=panels, master=master,
                winners=winners, masks=masks)


def _parse(value):
    g = _grid(value)
    matches = []
    for rotation in range(4):
        turned = np.rot90(g, rotation)
        for reflected in (False, True):
            canonical = turned[:, ::-1] if reflected else turned
            try:
                info = _read_canonical(canonical)
            except ValueError:
                continue
            info.update(rotation=rotation, reflected=reflected)
            matches.append(info)
    _require(len(matches) == 1, "one unambiguous clock orientation required")
    return matches[0]


def _selected(info, winners=None):
    if winners is None:
        winners = info['winners']
    return [info['master'] if c < 0 else info['panels'][b][c]
            for b, c in enumerate(winners)]


def _assemble(chosen, length, carrier, lattice, ordinary, special, motif):
    """Draw the compressed lattice and the uninterrupted clock in canonical space."""
    ph, pw = chosen[0].shape
    height = len(chosen) * (ph + 1) + 2
    out = np.full((height, length + pw + 3), carrier, dtype=np.int8)
    out[1:, length + 1:] = lattice
    for b, panel in enumerate(chosen):
        r0 = 2 + b * (ph + 1)
        out[r0:r0 + ph, length + 2:length + 2 + pw] = panel
    for d in range(height):
        color = carrier if d % 2 else special if d % 6 == 4 else ordinary
        out[d, :length] = color
        if d % 6 == 4:
            out[d, out[d] == motif] = special
    return out


def _render(info, winners=None):
    return _assemble(_selected(info, winners), info['length'], info['carrier'],
                     info['lattice'], info['ordinary'], info['special'], info['motif'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from this input, with no training, seed or cached state."""
    info = _parse(input_grid)
    return _uncanonical(_render(info), info['rotation'], info['reflected'])


_GEOMETRIES = tuple(
    (ph, pw, bands, copies, length)
    for ph in range(4, 13) for pw in range(3, 8)
    for bands in range(1, 6) for copies in range(3, 7) for length in range(3, 6)
    if bands * (ph + 1) + 2 <= 30 and length + 2 + copies * (pw + 1) <= 30
)


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _master_mask(rng, h, w):
    """Different spatial vocabularies, including symmetric and disconnected masks."""
    style = int(rng.integers(6))
    mask = np.zeros((h, w), dtype=bool)
    if style == 0:
        mask = rng.random((h, w)) < rng.uniform(.18, .82)
    elif style == 1:
        root = int(rng.integers(h)), int(rng.integers(w))
        mask[root] = True
        target = int(rng.integers(1, h * w + 1))
        for _ in range(target - 1):
            frontier = [(r, c) for r in range(h) for c in range(w)
                        if not mask[r, c] and any(0 <= rr < h and 0 <= cc < w and mask[rr, cc]
                                                 for rr, cc in ((r-1, c), (r+1, c),
                                                                (r, c-1), (r, c+1)))]
            if not frontier:
                break
            mask[_choose(rng, frontier)] = True
    elif style == 2 and min(h, w) >= 3:
        rh, rw = int(rng.integers(3, h + 1)), int(rng.integers(3, w + 1))
        r, c = int(rng.integers(h - rh + 1)), int(rng.integers(w - rw + 1))
        mask[r:r + rh, c] = mask[r:r + rh, c + rw - 1] = True
        mask[r, c:c + rw] = mask[r + rh - 1, c:c + rw] = True
    elif style == 3:
        c = int(rng.integers(w))
        mask[:, c] = True
        for r in rng.choice(h, int(rng.integers(1, h + 1)), replace=False):
            end = int(rng.integers(w))
            mask[r, min(c, end):max(c, end) + 1] = True
    elif style == 4:
        rr, cc = np.indices((h, w))
        period = int(rng.integers(2, 4))
        coord = rr + int(rng.integers(3)) * cc
        mask = coord % period == int(rng.integers(period))
    else:
        rr, cc = np.indices((h, w))
        mask = ((rr + cc) % 2 == int(rng.integers(2))) & (rng.random((h, w)) < .85)
    if rng.random() < .25:
        mask |= mask[:, ::-1]
    if rng.random() < .2:
        mask |= mask[::-1]
    if rng.random() < .2:
        mask = ~mask
    if not mask.any():
        mask.flat[int(rng.integers(mask.size))] = True
    return mask


def _mixed_row(rng, mask, row):
    """Teaching constraints give a selected row both colors at variable coordinates."""
    cols = rng.permutation(mask.shape[1])
    mask[row, int(cols[0])] = False
    mask[row, int(cols[1])] = True


def _toggle(master, cells, carrier, motif):
    result = master.copy()
    for r, c in cells:
        result[r, c] = carrier if master[r, c] == motif else motif
    return result


def _core_points(ph, pw, b, observed):
    return [(r, c) for r in range(1, ph - 1) for c in range(1, pw - 1)
            if ((2 + b * (ph + 1) + r) % 6 == 4) == observed]


def _sample_cells(rng, points, count):
    return [points[int(i)] for i in rng.choice(len(points), count, replace=False)]


def _silent_variants(rng, master, pool, count, carrier, motif, minimum=1):
    """Finite, distinct silent variants, even when the off-phase core is tiny."""
    result, used = [], set()
    for _ in range(count):
        candidate = None
        for _ in range(12):
            size = int(rng.integers(minimum, min(len(pool), max(minimum, 8)) + 1))
            trial = _toggle(master, _sample_cells(rng, pool, size), carrier, motif)
            if _key(trial) not in used:
                candidate = trial
                break
        if candidate is None:
            # Single flips, then pairs, exhaust more than the two needed variants.
            for size in (1, 2):
                for i in range(len(pool)):
                    cells = [pool[(i + j) % len(pool)] for j in range(size)]
                    trial = _toggle(master, cells, carrier, motif)
                    if _key(trial) not in used:
                        candidate = trial
                        break
                if candidate is not None:
                    break
        if candidate is not None:
            used.add(_key(candidate))
            result.append(candidate)
    return result


def _make_pair(rng, geometry, palette, rotation, reflected, teaching=None):
    ph, pw, bands, copies, length = geometry
    carrier, lattice, motif, ordinary, special = palette
    mask = _master_mask(rng, ph - 2, pw - 2)
    active = [_core_points(ph, pw, b, True) for b in range(bands)]
    silent = [_core_points(ph, pw, b, False) for b in range(bands)]
    plans = {}
    if teaching == 'balanced':
        b = int(_choose(rng, [b for b in range(bands) if active[b]]))
        vr = int(_choose(rng, sorted({r for r, c in active[b]})))
        sr = int(_choose(rng, sorted({r for r, c in silent[b]})))
        _mixed_row(rng, mask, vr - 1)
        _mixed_row(rng, mask, sr - 1)
        plans[b] = ('balanced', vr, sr)
    elif teaching == 'polarity':
        for b, mode in ((0, 'add'), (1, 'remove')):
            rows = sorted({r for r, c in active[b]})
            assert len(rows) >= 2
            r = rows[-1] if b == 0 else rows[0]
            _mixed_row(rng, mask, r - 1)
            plans[b] = (mode, r, None)
    master = np.full((ph, pw), carrier, dtype=np.int8)
    master[1:-1, 1:-1][mask] = motif
    panel_rows, chosen, previous_winner = [], [], None
    for b in range(bands):
        row = [master.copy() for _ in range(copies)]
        winner = None
        if active[b] and (teaching is not None or rng.random() < .85):
            if b in plans:
                mode, vr, sr = plans[b]
                if mode == 'balanced':
                    cells = []
                    for r in (vr, sr):
                        for color in (carrier, motif):
                            pool = [(rr, c) for rr, c in active[b] + silent[b]
                                    if rr == r and master[rr, c] == color]
                            cells.append(_choose(rng, pool))
                else:
                    color = carrier if mode == 'add' else motif
                    pool = [(r, c) for r, c in active[b]
                            if r == vr and master[r, c] == color]
                    cells = _sample_cells(rng, pool, int(rng.integers(1, len(pool) + 1)))
            else:
                cells = _sample_cells(rng, active[b], int(rng.integers(1, min(4, len(active[b])) + 1)))
                if silent[b] and rng.random() < .55:
                    cells += _sample_cells(rng, silent[b],
                                           int(rng.integers(1, min(4, len(silent[b])) + 1)))
            winner = _toggle(master, cells, carrier, motif)
        if winner is not None:
            choices = list(range(copies))
            if teaching == 'polarity' and b == 1:
                choices.remove(previous_winner)
            wc = int(_choose(rng, choices))
            row[wc] = winner
            previous_winner = wc
            # Three copies leave two exact templates; larger bands also hold a
            # dormant distractor. Both teaching constructions use >=4 copies.
            if copies >= 4:
                minimum = 5 if b in plans and plans[b][0] == 'balanced' else 1
                decoys = _silent_variants(rng, master, silent[b], 1, carrier, motif, minimum)
                dc = int(_choose(rng, [c for c in range(copies) if c != wc]))
                row[dc] = decoys[0]
            chosen.append(winner.copy())
        else:
            count = int(rng.integers(0, 3)) if teaching is None else 2
            count = min(count, copies - 2, (1 << min(len(silent[b]), 2)) - 1)
            decoys = _silent_variants(rng, master, silent[b], count, carrier, motif)
            columns = rng.choice(copies, len(decoys), replace=False)
            for c, decoy in zip(columns, decoys):
                row[int(c)] = decoy
            chosen.append(master.copy())
        panel_rows.append(row)
    height, width = bands * (ph + 1) + 2, length + 2 + copies * (pw + 1)
    inp = np.full((height, width), carrier, dtype=np.int8)
    inp[1:, length + 1:] = lattice
    for b, row in enumerate(panel_rows):
        for c, panel in enumerate(row):
            r0, c0 = 2 + b * (ph + 1), length + 2 + c * (pw + 1)
            inp[r0:r0 + ph, c0:c0 + pw] = panel
    inp[0, :length] = ordinary
    inp[0, int(rng.integers(1, length - 1))] = special
    out = _assemble(chosen, length, carrier, lattice, ordinary, special, motif)
    return {'input': _uncanonical(inp, rotation, reflected),
            'output': _uncanonical(out, rotation, reflected)}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name + ' outside legal range')
    return int(value)


def _unique(pair, seen):
    if _key(pair['input']) not in seen:
        return pair
    colors = sorted(map(int, np.unique(pair['input'])))
    # There are 30240 role palettes and fewer than 14 previous inputs.
    for palette in permutations(range(10), 5):
        lookup = np.arange(10, dtype=np.int8)
        lookup[colors] = palette
        inp = lookup[pair['input']]
        if _key(inp) not in seen:
            return {'input': inp, 'output': lookup[pair['output']]}
    raise AssertionError('palette space exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    ne = _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed))
    balanced = [g for g in _GEOMETRIES if g[0] in (6, 7) and g[2] == 3 and g[1] >= 4 and g[3] >= 4]
    polarity = [g for g in _GEOMETRIES if g[0] in (11, 12) and g[2] == 2 and g[1] >= 4 and g[3] >= 4]
    normal = [g for g in _GEOMETRIES if g[2] <= 3]
    expanded = [g for g in _GEOMETRIES if g[2] >= 4]
    plan = [('balanced', _choose(rng, balanced)), ('polarity', _choose(rng, polarity))]
    if rng.integers(2):
        plan.reverse()
    plan += [(None, _choose(rng, normal)) for _ in range(nt - 2)]
    plan += [(None, _choose(rng, expanded)) for _ in range(ne)]
    base_rotation = int(rng.integers(4))
    shared_palette = rng.random() < .25
    palette = tuple(map(int, rng.choice(10, 5, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for i, (teaching, geometry) in enumerate(plan):
        if not shared_palette:
            palette = tuple(map(int, rng.choice(10, 5, replace=False)))
        rotation = (base_rotation + i) % 4 if i < 2 else int(rng.integers(4))
        pair = _make_pair(rng, geometry, palette, rotation, bool(rng.integers(2)), teaching)
        pair = _unique(pair, seen)
        seen.add(_key(pair['input']))
        puzzle['train' if i < nt else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _coordinate_output(info):
    """Check output pixels by original source coordinates, rather than painting."""
    ph, pw, length = info['ph'], info['pw'], info['length']
    out = np.empty((len(info['g']), length + pw + 3), dtype=np.int8)
    chosen = _selected(info)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            if c < length:
                color = (info['carrier'] if r % 2 else
                         info['special'] if r % 6 == 4 else info['ordinary'])
            elif c == length or r == 0:
                color = info['carrier']
            else:
                lr = (r - 1) % (ph + 1)
                lc = c - length - 1
                if lr == 0 or lc in (0, pw + 1):
                    color = info['lattice']
                else:
                    b = (r - 1) // (ph + 1)
                    color = int(chosen[b][lr - 1, lc - 1])
                    if r % 6 == 4 and color == info['motif']:
                        color = info['special']
            out[r, c] = color
    return out


def _measure(info):
    master, motif = info['master'], info['motif']
    record = dict(geometry=(info['ph'], info['pw'], info['bands'], info['copies'], info['length']),
                  shape=info['g'].shape, orientation=(info['rotation'], info['reflected']),
                  palette=tuple(info[k] for k in ('carrier', 'lattice', 'motif', 'ordinary', 'special')),
                  master=_key(master == motif), density=float(np.mean(master[1:-1, 1:-1] == motif)),
                  bead=info['bead'], columns=tuple(info['winners']), raw_counts=[],
                  additions=[], removals=[], edits=[], balanced=False, silent=False,
                  larger_decoy=False, dormant=False, later_only=False, remote_highlight=False)
    all_difference_winners = []
    for b, row in enumerate(info['panels']):
        differences = [int(np.count_nonzero(p != master)) for p in row]
        all_difference_winners.append(int(np.argmax(differences)) if max(differences) else -1)
        record['raw_counts'].append(sum(n > 0 for n in differences))
        wc, observed = info['winners'][b], info['masks'][b]
        if wc < 0:
            record['dormant'] |= max(differences) > 0
            continue
        winner = row[wc]
        add = int(np.count_nonzero((master != motif) & (winner == motif)))
        remove = int(np.count_nonzero((master == motif) & (winner != motif)))
        record['additions'].append(add)
        record['removals'].append(remove)
        record['edits'].append(add + remove)
        view_add = int(np.count_nonzero((master[observed] != motif) & (winner[observed] == motif)))
        view_remove = int(np.count_nonzero((master[observed] == motif) & (winner[observed] != motif)))
        record['balanced'] |= view_add == view_remove > 0 and add == remove
        record['silent'] |= bool(np.any(winner[~observed] != master[~observed]))
        record['larger_decoy'] |= any(n > differences[wc] for c, n in enumerate(differences) if c != wc)
        core_rows = [r for r in range(1, info['ph'] - 1) if observed[r]]
        record['later_only'] |= (len(core_rows) >= 2 and
                                not np.any(winner[core_rows[0]] != master[core_rows[0]]) and
                                any(np.any(winner[r] != master[r]) for r in core_rows[1:]))
        record['remote_highlight'] |= bool(np.any(winner[observed] == motif))
    correct = _render(info)
    record['ablate_A_changes'] = not np.array_equal(correct, _render(info, all_difference_winners))
    record['ablate_B_changes'] = not np.array_equal(correct, _render(info, [-1] * info['bands']))
    return record


def validate_generated_puzzle(puzzle):
    """Validate observable semantics, teaching coverage, uniqueness and shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, measurements = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            try:
                inp, out = _grid(pair['input']), _grid(pair['output'])
                info = _parse(inp)
            except ValueError as error:
                raise AssertionError(str(error)) from error
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            expected = _uncanonical(_coordinate_output(info), info['rotation'], info['reflected'])
            assert np.array_equal(out, expected), 'incorrect selected contents, phase, or framing'
            assert not np.array_equal(inp, out)
            measurements[split].append(_measure(info))
    train = measurements['train']
    for feature in ('balanced', 'silent', 'larger_decoy', 'dormant', 'later_only',
                    'remote_highlight', 'ablate_A_changes', 'ablate_B_changes'):
        assert any(m[feature] for m in train), 'missing training witness: ' + feature
    polarities = [(a, r) for m in train for a, r in zip(m['additions'], m['removals'])]
    assert any(a > 0 and r == 0 for a, r in polarities), 'teach additions'
    assert any(r > 0 and a == 0 for a, r in polarities), 'teach removals'
    assert any(len({c for c in m['columns'] if c >= 0}) >= 2 for m in train), 'varied winning columns'
    assert len({m['geometry'][:2] for m in train}) >= 2, 'varied panel shape'
    assert len({m['orientation'][0] for m in train}) >= 2, 'teach inward direction'
    assert all(m['geometry'][2] <= 3 for m in train)
    assert all(m['geometry'][2] >= 4 for m in measurements['test']), 'expanded band-count shift'


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


def self_test():
    """A bounded 42-seed diversity audit plus count, covariance and mutation checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(generate_puzzle(731, 7, 3)) == _puzzle_key(generate_puzzle(731, 7, 3))
    measurements = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                saved = pair['input'].copy()
                answer = solve(saved)
                assert np.array_equal(answer, pair['output'])
                assert np.array_equal(saved, pair['input']), 'solve mutated its input'
                measurements.append((split, _measure(_parse(pair['input']))))
    all_m = [m for _, m in measurements]
    assert len({m['geometry'] for m in all_m}) >= 35
    assert len({m['shape'] for m in all_m}) >= 30
    assert len({m['master'] for m in all_m}) >= 70
    assert {m['orientation'] for m in all_m} == {(k, f) for k in range(4) for f in (False, True)}
    assert {m['geometry'][0] for m in all_m} == set(range(4, 13))
    assert {m['geometry'][1] for m in all_m} == {3, 4, 5, 6, 7}
    assert {m['geometry'][2] for m in all_m} == {1, 2, 3, 4, 5}
    assert {m['geometry'][3] for m in all_m} == {3, 4, 5, 6}
    assert {m['geometry'][4] for m in all_m} == {3, 4, 5}
    for role in range(5):
        assert {m['palette'][role] for m in all_m} == set(range(10))
    assert min(m['density'] for m in all_m) < .25
    assert max(m['density'] for m in all_m) > .75
    assert {n for m in all_m for n in m['raw_counts']} == {0, 1, 2}
    assert len({n for m in all_m for n in m['edits']}) >= 6
    assert any(m['geometry'][0] != m['geometry'][1] for m in all_m)
    assert any(len([c for c in m['columns'] if c >= 0]) >
               len({c for c in m['columns'] if c >= 0}) for m in all_m)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(100 + 10 * nt + ne, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (None, 1), (4, None), (True, 1), (4, False), (2.5, 1), (4, 1.5)):
        try:
            generate_puzzle(1, nt, ne)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    # The solver also supports legal family members outside the sampled shift:
    # all-master inputs, empty observation sets, and raw exceptions in many bands.
    base_info = _parse(puzzles[0]['train'][0]['input'])
    canonical = base_info['g'].copy()
    for b, row in enumerate(base_info['panels']):
        for c in range(base_info['copies']):
            r0, c0 = 2 + b * (base_info['ph'] + 1), base_info['length'] + 2 + c * (base_info['pw'] + 1)
            canonical[r0:r0 + base_info['ph'], c0:c0 + base_info['pw']] = base_info['master']
    all_master = _read_canonical(canonical)
    assert all(c == -1 for c in all_master['winners'])
    prediction = _coordinate_output(all_master)
    for k in range(4):
        for f in (False, True):
            inp = _uncanonical(canonical, k, f)
            assert np.array_equal(solve(inp), _uncanonical(prediction, k, f))
    # Reject damaged output, loss of a selected silent edit, and multiple winners.
    original = puzzles[0]
    for defect in ('pixel', 'silent', 'two_winners'):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
        pair = next(p for p in bad['train'] if _measure(_parse(p['input']))['balanced'])
        info = _parse(pair['input'])
        if defect == 'pixel':
            pair['output'][0, 0] = (int(pair['output'][0, 0]) + 1) % 10
        elif defect == 'silent':
            b = next(b for b, c in enumerate(info['winners']) if c >= 0 and
                     np.any((info['panels'][b][c] != info['master']) & ~info['masks'][b][:, None]))
            c = info['winners'][b]
            cells = np.argwhere((info['panels'][b][c] != info['master']) & ~info['masks'][b][:, None])
            r, col = map(int, cells[0])
            out = _render(info)
            out[2 + b * (info['ph'] + 1) + r, info['length'] + 2 + col] = info['master'][r, col]
            pair['output'] = _uncanonical(out, info['rotation'], info['reflected'])
        else:
            b = next(b for b, c in enumerate(info['winners']) if c >= 0)
            wc = info['winners'][b]
            dc = next(c for c, p in enumerate(info['panels'][b]) if c != wc and
                      not np.array_equal(p, info['master']))
            r, col = _core_points(info['ph'], info['pw'], b, True)[0]
            inp = info['g'].copy()
            inp[2 + b * (info['ph'] + 1) + r,
                info['length'] + 2 + dc * (info['pw'] + 1) + col] = (
                    info['carrier'] if info['master'][r, col] == info['motif'] else info['motif'])
            pair['input'] = _uncanonical(inp, info['rotation'], info['reflected'])
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError('validator accepted ' + defect)
    return dict(seeds=42, pairs=len(all_m), geometries=len({m['geometry'] for m in all_m}),
                canonical_shapes=len({m['shape'] for m in all_m}),
                master_bitmaps=len({m['master'] for m in all_m}),
                palettes=len({m['palette'] for m in all_m}), orientations=8,
                independent_solver=False)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
