"""Bridge-limited shuttles: repair cues, gauge matching, and finite support.

No files, parent modules, examples, or random state are used by solve().
Generation constructs labels separately from the input parser and simulator.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _runs(values):
    runs = []
    for x in sorted(values):
        if not runs or x != runs[-1][-1] + 1:
            runs.append([x])
        else:
            runs[-1].append(x)
    return runs


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        stack, cells = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _rectangle(cells):
    rs, cs = zip(*cells)
    return len(cells) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)


def _cell(f, k, q):
    a, b = f['k0'] + k, f['q0'] + q
    return (a * f['out'][0] + b * f['perp'][0],
            a * f['out'][1] + b * f['perp'][1])


def _local(f, p):
    r, c = p
    return (r * f['out'][0] + c * f['out'][1] - f['k0'],
            r * f['perp'][0] + c * f['perp'][1] - f['q0'])


def _find_forks(cells):
    """Find an intact base and its three positive, centered attachment runs.

    Interrupted rails are NOT required to be connected to the base. Their
    outward rays must be disjoint from all foreign structures in a legal input.
    """
    forks = []
    for out in _DIRS:
        perp = (-out[1], out[0])
        projected = {(r * out[0] + c * out[1], r * perp[0] + c * perp[1])
                     for r, c in cells}
        by_k = {}
        for k, q in projected:
            by_k.setdefault(k, set()).add(q)
        for k0 in sorted(by_k):
            for base in _runs(by_k[k0]):
                width = len(base)
                if width < 5:
                    continue
                q0 = base[0]
                attached = {q - q0 for q in by_k.get(k0 + 1, ())
                            if q0 <= q < q0 + width}
                arms = _runs(attached)
                if len(arms) != 3 or arms[0] != [0] or arms[2] != [width - 1]:
                    continue
                middle = arms[1]
                span, center = len(middle), middle[0]
                if not 1 <= span <= 3 or 2 * center + span != width:
                    continue
                if center < 2:
                    continue
                stop = 1
                while all((k0 + stop + 1, q0 + q) in projected for q in middle):
                    stop += 1
                depths = [max(k - k0 for k, q in projected
                              if q == q0 + edge and k > k0)
                          for edge in (0, width - 1)]
                if min(depths) < stop + 5:
                    continue
                f = dict(out=out, perp=perp, k0=k0, q0=q0, width=width,
                         span=span, center=center, stop=stop, depths=tuple(depths))
                forks.append(f)
    return forks


def _parse(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    assert 0 not in colors, 'black is reserved for the output trail'
    assert sum(counts == counts.max()) == 1
    bg = int(colors[np.argmax(counts)])
    candidates = []
    for color in colors:
        if int(color) == bg:
            continue
        cells = set(map(tuple, np.argwhere(grid == color).tolist()))
        forks = _find_forks(cells)
        if forks:
            candidates.append((int(color), cells, forks))
    assert len(candidates) == 1, 'one structural color must identify the forks'
    line, cells, forks = candidates[0]
    skeleton, fixed, repairs, gaps, spurs = set(), set(), [], [], set()
    for f in forks:
        base = {_cell(f, 0, q) for q in range(f['width'])}
        prong = {_cell(f, k, q) for k in range(1, f['stop'] + 1)
                 for q in range(f['center'], f['center'] + f['span'])}
        frame = base | prong
        assert frame <= cells
        rails = []
        for edge, depth in zip((0, f['width'] - 1), f['depths']):
            rail = {_cell(f, k, edge) for k in range(1, depth + 1)}
            frame |= rail
            present = {k for k in range(1, depth + 1) if _cell(f, k, edge) in cells}
            for missing in _runs(set(range(1, depth + 1)) - present):
                assert len(missing) == 3, 'rail breaks have exactly three cells'
                g = missing[0]
                assert g >= f['stop'] + 2 and g + 3 <= depth
                evidence = []
                for anchor, toward in ((g - 1, 1), (g + 3, -1)):
                    for side in (-1, 1):
                        off = {_cell(f, anchor + toward * j, edge + side * j)
                               for j in (1, 2)}
                        third = _cell(f, anchor + toward * 3, edge + side * 3)
                        if off <= cells and third not in cells:
                            evidence.append(off)
                assert len(evidence) <= 1, 'ambiguous deviation'
                gap_cells = {_cell(f, k, edge) for k in missing}
                gaps.append((f, edge, g, bool(evidence)))
                if evidence:
                    repairs.append(gap_cells)
                    spurs |= evidence[0]
                    present.update(missing)
            rails.append(present)
        assert skeleton.isdisjoint(frame), 'fork ray envelopes must be disjoint'
        skeleton |= frame
        fixed |= base | prong
        f['rails'] = tuple(rails)
    assert spurs.isdisjoint(skeleton)
    noise = cells - skeleton
    assert spurs <= noise
    # The legal off-frame objects are short 45-degree chains. Base anchors
    # belong to the frame; they are never erased along with their deviations.
    for comp in _diagonal_components(noise):
        assert len(comp) in (2, 3)
        ordered = sorted(comp)
        assert all(ordered[i + 1][0] - ordered[i][0] == 1
                   and abs(ordered[i + 1][1] - ordered[i][1]) == 1
                   for i in range(len(ordered) - 1))
        assert len({r - c for r, c in comp}) == 1 or len({r + c for r, c in comp}) == 1
    clean = (cells & skeleton) | set().union(set(), *repairs)
    tokens = []
    for color in colors:
        if int(color) in (bg, line):
            continue
        for comp in _components(grid == color):
            assert _rectangle(comp), 'each token is a solid rectangle'
            tokens.append(dict(cells=comp, color=int(color)))
    assert forks and tokens
    matches, used = [], set()
    for fi, f in enumerate(forks):
        possible = []
        for ti, token in enumerate(tokens):
            local = {_local(f, p) for p in token['cells']}
            qs = {q for _, q in local}
            if qs != set(range(f['center'], f['center'] + f['span'])):
                continue
            start = min(k for k, _ in local)
            end = max(k for k, _ in local)
            length = end - start + 1
            if not 1 <= length <= 5 or start <= f['stop'] + 1:
                continue
            if end > min(f['depths']):
                continue
            if not all(any(k in rail for k in range(start, end + 1)) for rail in f['rails']):
                continue
            possible.append(dict(fi=fi, ti=ti, start=start, length=length))
        assert len(possible) <= 1, 'at most one gauge-matching shuttle per fork'
        if possible:
            m = possible[0]
            assert m['ti'] not in used, 'one shuttle cannot match two forks'
            used.add(m['ti'])
            matches.append(m)
    return dict(grid=grid, bg=bg, line=line, forks=forks, tokens=tokens,
                matches=matches, clean=clean, noise=noise, repairs=repairs,
                gaps=gaps, skeleton=skeleton, fixed=fixed)


def _diagonal_components(cells):
    unseen = set(cells)
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, comp = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 1):
                for dc in (-1, 1):
                    p = r + dr, c + dc
                    if p in unseen:
                        unseen.remove(p)
                        comp.add(p)
                        stack.append(p)
        result.append(comp)
    return result


def _render(a, *, repair=True, support=True, point_support=False):
    """Simulate inward unit steps; the two switches are ablation checks only."""
    output = np.full_like(a['grid'], a['bg'])
    structural = a['clean'] if repair else set(map(tuple, np.argwhere(a['grid'] == a['line']).tolist())) - a['noise']
    for p in structural:
        output[p] = a['line']
    occupied_routes = set()
    motions = []
    for m in a['matches']:
        f, token = a['forks'][m['fi']], a['tokens'][m['ti']]
        k, length = m['start'], m['length']
        while k > f['stop'] + 1:
            trial = k - 1
            contact_length = 1 if point_support else length
            if support and not all(any(_cell(f, depth, edge) in structural
                                       for depth in range(trial, trial + contact_length))
                                   for edge in (0, f['width'] - 1)):
                break
            k = trial
        final = {_cell(f, depth, q) for depth in range(k, k + length)
                 for q in range(f['center'], f['center'] + f['span'])}
        route = {_cell(f, depth, q) for depth in range(k, m['start'] + length)
                 for q in range(f['center'], f['center'] + f['span'])}
        assert all(0 <= r < output.shape[0] and 0 <= c < output.shape[1] for r, c in route)
        assert route.isdisjoint(a['clean']) and route.isdisjoint(occupied_routes)
        occupied_routes |= route
        for p in route - final:
            output[p] = 0
        for p in final:
            output[p] = token['color']
        motions.append(dict(m, final_start=k, distance=m['start'] - k,
                            final=final, route=route, docked=k == f['stop'] + 1))
    return output, motions


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Clean the rails, select shuttles geometrically, and move while supported."""
    return _render(_parse(input_grid))[0]


def _module(rng, span, length, mode, complex_gaps=False, active=True):
    """Construct a local fork; rows are outward depth, columns cross-section."""
    clearance = int(rng.integers(1, 3))
    width = span + 2 * (clearance + 1)
    center = clearance + 1
    stop = int(rng.integers(1, 4))
    first = stop + int(rng.integers(2, 4))
    slots = [first, first + int(rng.integers(5, 8))]
    if complex_gaps:
        slots = [stop + 2, stop + 7, stop + 12]
    repair_g = slots[-1]
    start = repair_g + int(rng.integers(4, 6))
    rail_depths = [start + length - 1 + int(rng.integers(0, 3)) for _ in range(2)]
    depth = max(rail_depths)
    assert depth < 30
    clean = np.zeros((depth + 1, width), bool)
    clean[0, :] = True
    clean[1:rail_depths[0] + 1, 0] = True
    clean[1:rail_depths[1] + 1, -1] = True
    clean[1:stop + 1, center:center + span] = True
    side = int(rng.integers(2))
    permanent = []
    if mode != 'clear':
        permanent.append((side, slots[0]))
        if complex_gaps:
            permanent.append((1 - side, slots[1]))
    for which, g in permanent:
        clean[g:g + 3, 0 if which == 0 else width - 1] = False
    inp = clean.copy()
    repair_side = int(rng.integers(2))
    edge = 0 if repair_side == 0 else width - 1
    inp[repair_g:repair_g + 3, edge] = False
    high_anchor = bool(rng.integers(2))
    anchor, toward = (repair_g + 3, -1) if high_anchor else (repair_g - 1, 1)
    for j in (1, 2):
        inp[anchor + toward * j, edge + (1 if edge == 0 else -1) * j] = True
    # An offset near-miss lives beside a permanent break, with its extrapolated
    # anchor missing. It cannot authorize a repair. It is away from all tokens.
    if permanent and rng.random() < .5:
        which, g = permanent[0]
        e = 0 if which == 0 else width - 1
        inward = 1 if e == 0 else -1
        inp[g + 1, e + inward] = inp[g + 2, e + 2 * inward] = True
    original = {(k, q) for k in range(start, start + length)
                for q in range(center, center + span)} if active else set()
    # A nearer, centered but too-wide rectangle defeats nearest-object and
    # color-only matching. It is discarded before motion is simulated.
    decoy_k = slots[0] + 3
    decoy = {(decoy_k, q) for q in range(center - 1, center + span + 1)}
    assert all(not inp[p] for p in original | decoy)
    # Independent construction formula: a <=3-cell shuttle loses all contact
    # at the first 3-cell break; a >=4-cell shuttle bridges every such break.
    final_start = stop + 1
    if length <= 3 and permanent:
        final_start = max(g + 4 - length for _, g in permanent)
    final = {(k, q) for k in range(final_start, final_start + length)
             for q in range(center, center + span)} if active else set()
    route = {(k, q) for k in range(final_start, start + length)
             for q in range(center, center + span)} if active else set()
    return dict(inp=inp, clean=clean, original=original, decoy=decoy,
                final=final, route=route, span=span, length=length,
                permanent=permanent, stop=stop)


def _make_pair(rng, index, test, palette):
    if not test:
        count = 2 if index < 2 else int(rng.integers(1, 4))
        lengths = ([3, int(rng.integers(1, 4))] if index == 0
                   else [4, int(rng.integers(1, 4))] if index == 1
                   else [int(rng.integers(1, 6)) for _ in range(count)])
    else:
        count = int(rng.integers(2, 4))
        lengths = [int(rng.integers(1, 4)), int(rng.integers(4, 6))]
        lengths += [int(rng.integers(1, 6)) for _ in range(count - 2)]
    spans = list(map(int, rng.integers(1, 4, count)))
    if count >= 2 and spans[0] == spans[1]:
        spans[1] = spans[1] % 3 + 1
    modules = []
    for j in range(count):
        mode = 'clear' if not test and index == 0 and j == 1 else 'broken'
        active = j < 2 or bool(rng.integers(2))
        modules.append(_module(rng, spans[j], lengths[j], mode,
                               complex_gaps=test, active=active))
    # Resample an over-wide whole scene; never change its intended labels.
    total = sum(m['inp'].shape[1] for m in modules) + count - 1
    if total > 30:
        return None
    height = max(m['inp'].shape[0] for m in modules)
    h = min(30, height + int(rng.integers(0, 4)))
    w = min(30, total + int(rng.integers(0, 5)))
    bg, line, token1, token2 = palette
    inp = np.full((h, w), bg, dtype=np.int8)
    output = inp.copy()
    remaining = w - total
    gaps = [1] * (count - 1)
    # Distribute padding freely between and outside receiver envelopes.
    padding = rng.multinomial(remaining, [1 / (count + 1)] * (count + 1))
    c0 = int(padding[0])
    for j, m in enumerate(modules):
        mh, mw = m['inp'].shape
        r0 = int(rng.integers(h - mh + 1))
        local_in = np.full((mh, mw), bg, dtype=np.int8)
        local_out = local_in.copy()
        local_in[m['inp']] = line
        local_out[m['clean']] = line
        token_color = token1 if j % 2 == 0 else token2
        for p in m['original']:
            local_in[p] = token_color
        for p in m['decoy']:
            local_in[p] = token_color  # Same color as the valid shuttle.
        for p in m['route'] - m['final']:
            local_out[p] = 0
        for p in m['final']:
            local_out[p] = token_color
        if rng.integers(2):
            local_in, local_out = local_in[::-1], local_out[::-1]
        if rng.integers(2):
            local_in, local_out = local_in[:, ::-1], local_out[:, ::-1]
        inp[r0:r0 + mh, c0:c0 + mw] = local_in
        output[r0:r0 + mh, c0:c0 + mw] = local_out
        c0 += mw + (gaps[j] if j < count - 1 else 0) + int(padding[j + 1])
    turns = int(rng.integers(4))
    return dict(input=np.rot90(inp, turns).copy(), output=np.rot90(output, turns).copy())


def _count(value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('example counts must be integers')
    if not lo <= int(value) <= hi:
        raise ValueError('example count out of range')
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, t = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if n <= t:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    base_palette = tuple(map(int, rng.choice(np.arange(1, 10), 4, replace=False)))
    varied = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', n), ('test', t)):
        for index in range(count):
            for _ in range(64):
                palette = (tuple(map(int, rng.choice(np.arange(1, 10), 4, replace=False)))
                           if varied else base_palette)
                pair = _make_pair(rng, index, split == 'test', palette)
                if pair is None or _key(pair['input']) in seen:
                    continue
                # Reject accidental inter-object cue or component interactions;
                # independently constructed labels must survive the full parser.
                try:
                    a = _parse(pair['input'])
                    assert np.array_equal(_render(a)[0], pair['output'])
                except AssertionError:
                    continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError('bounded scene construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    n, t = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, train_features = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            a = _parse(inp)
            expected, motions = _render(a)
            assert np.array_equal(out, expected), 'incorrect repair, movement, or trail'
            assert a['repairs'] and a['noise'] and motions
            assert len(a['tokens']) > len(motions), 'missing mismatch evidence'
            used = {m['ti'] for m in motions}
            # There is a nearer, centered rectangle in the SAME color as a
            # genuine shuttle, but with a different perpendicular span.
            gauge_contrast = False
            for m in motions:
                f = a['forks'][m['fi']]
                for ti, token in enumerate(a['tokens']):
                    if ti in used or token['color'] != a['tokens'][m['ti']]['color']:
                        continue
                    local = {_local(f, p) for p in token['cells']}
                    qs = {q for _, q in local}
                    if (len(qs) != f['span'] and min(qs) + max(qs) == f['width'] - 1
                            and f['stop'] < min(k for k, _ in local)
                            and max(k for k, _ in local) < m['start']):
                        gauge_contrast = True
            assert gauge_contrast, 'missing size-matching counterexample'
            assert all(out[p] == a['line'] for p in a['clean'])
            assert all(out[p] == a['bg'] or p in set().union(set(), *(m['route'] for m in motions))
                       for p in a['noise'])
            assert all(m['distance'] > 0 for m in motions)
            for m in motions:
                f = a['forks'][m['fi']]
                # A finite object needs one support cell on EACH side at its
                # final position. A stopped object's next position loses one.
                assert all(any(k in rail for k in range(m['final_start'], m['final_start'] + m['length']))
                           for rail in f['rails'])
                if not m['docked']:
                    assert any(not any(k in rail for k in range(m['final_start'] - 1,
                                                               m['final_start'] - 1 + m['length']))
                               for rail in f['rails'])
            if split == 'train':
                train_features.append((a, motions))
    # These contrasts are guaranteed even at the minimum 2/1 counts.
    assert any(not m['docked'] and m['length'] <= 3 for _, ms in train_features for m in ms)
    assert any(m['docked'] and m['length'] >= 4 for _, ms in train_features for m in ms)
    assert any(m['docked'] and m['length'] <= 3 for _, ms in train_features for m in ms)
    assert any(not np.array_equal(_render(a, repair=False)[0], _render(a)[0])
               and any(x['final_start'] != y['final_start']
                       for x, y in zip(_render(a, repair=False)[1], _render(a)[1]))
               for a, _ in train_features), 'repair must affect a shuttle destination'
    assert any(not np.array_equal(_render(a, support=False)[0], _render(a)[0])
               for a, _ in train_features), 'permanent breaks must affect motion'
    assert any(any(x['final_start'] != y['final_start']
                   for x, y in zip(_render(a, point_support=True)[1], _render(a)[1]))
               for a, _ in train_features), 'finite token extent must affect motion'


def self_test():
    """42 puzzles, streamed in memory; no traces or bulk samples are saved."""
    # Hand-specified counterfactual pair: same rails, length three versus four.
    # This fixture uses neither _module nor its closed-form label construction.
    for length, target in ((3, 6), (4, 3)):
        inp = np.full((24, 15), 9, dtype=np.int8)
        inp[0, 2:9] = 1
        inp[1:23, 2] = inp[1:22, 8] = 1
        inp[1:3, 5] = 1
        inp[5:8, 2] = 9  # Permanent hole on the left.
        expected = inp.copy()
        inp[12:15, 8] = 9  # Cued hole on the right.
        inp[12, 7] = inp[13, 6] = 1
        inp[4, 11] = inp[5, 12] = inp[6, 13] = 1  # Free triple.
        inp[6, 3] = inp[7, 4] = 1  # Missing-anchor near miss.
        inp[18:18 + length, 5] = 4
        inp[8, 4:7] = 4  # Nearer wrong gauge, same color.
        expected[target:target + length, 5] = 4
        expected[target + length:18 + length, 5] = 0
        assert np.array_equal(solve(inp), expected)
        # All nonblack literal roles may be exchanged without changing behavior.
        remap = np.arange(10, dtype=np.int8)
        remap[[1, 4, 9]] = [9, 1, 4]
        assert np.array_equal(solve(remap[inp]), remap[expected])
    signatures, shapes, geometry, palettes, directions = set(), set(), set(), set(), set()
    lengths, spans, final_modes, counts = set(), set(), set(), set()
    normalized = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(hashlib.sha256(repr(tuple(_key(p['input']) for s in puzzle for p in puzzle[s])).encode()).digest())
        for split in puzzle:
            for pair in puzzle[split]:
                a = _parse(pair['input'])
                predicted, motions = _render(a)
                assert np.array_equal(predicted, pair['output'])
                shapes.add(pair['input'].shape)
                palettes.add((a['bg'], a['line']))
                counts.add(len(a['forks']))
                roles = {a['bg']: 0, a['line']: 1}
                flat = tuple(roles.setdefault(int(c), len(roles)) for c in pair['input'].flat)
                normalized.add(hashlib.sha256(bytes(flat)).digest())
                for f in a['forks']:
                    directions.add(f['out'])
                    spans.add(f['span'])
                    geometry.add((f['width'], f['stop'], f['depths'],
                                  tuple(tuple(sorted(x)) for x in f['rails'])))
                for m in motions:
                    lengths.add(m['length'])
                    final_modes.add((m['length'] >= 4, m['docked']))
        # Direction and reflection covariance test on a genuinely new rule.
        pair = puzzle['test'][0]
        turns = seed % 4
        rotated = np.rot90(pair['input'], turns)[:, ::-1]
        assert np.array_equal(solve(rotated), np.rot90(pair['output'], turns)[:, ::-1])
    assert len(signatures) == 42 and len(shapes) >= 30 and len(normalized) >= 180
    assert len(geometry) >= 150 and len(palettes) >= 25
    assert directions == set(_DIRS) and spans == {1, 2, 3} and lengths == {1, 2, 3, 4, 5}
    assert counts == {1, 2, 3}
    assert {(False, False), (False, True), (True, True)} <= final_modes
    for n in range(2, 11):
        for t in range(1, min(4, n - 1) + 1):
            generate_puzzle(12345 + 10 * n + t, n, t)
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    first, second = generate_puzzle(937, 2, 1), generate_puzzle(937, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in first
               for x, y in zip(first[s], second[s]) for k in ('input', 'output'))
    # Reject a plausible short-token label that always docks at the prong.
    bad = generate_puzzle(17, 2, 1)
    p = bad['train'][0]
    p['output'] = _render(_parse(p['input']), support=False)[0]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('always-dock corruption accepted')
    return {'seeds': 42, 'pairs': 210, 'distinct_puzzles': len(signatures),
            'grid_shapes': len(shapes), 'normalized_inputs': len(normalized),
            'rail_geometries': len(geometry), 'role_palettes': len(palettes),
            'token_lengths': sorted(lengths), 'spans': sorted(spans),
            'receiver_counts': sorted(counts)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
