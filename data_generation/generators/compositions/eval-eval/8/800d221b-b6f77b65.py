"""Load-bearing ballots: local votes, removable props, and contact voting.

No files, parent modules, generation records, or random state are used by solve.
Generation uses cell-body simulation; inference uses tower heights instead.
"""
from __future__ import annotations

import hashlib
import itertools
import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _runs(mask):
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return list(zip(map(int, starts), map(int, ends)))


def _parse(value):
    """Decode the marker, rectangles, prongs, and fully visible floor props."""
    a = _grid(value)
    h, w = a.shape
    bg = int(a[1, 0])
    assert np.all(a[1] == bg)
    commands = set(map(int, a[0][a[0] != bg]))
    assert 1 <= len(commands) <= 2
    assert np.count_nonzero(a[0] != bg) == len(commands)
    occupied = np.argwhere(a[2:] != bg)
    assert len(occupied)
    top = int(occupied[:, 0].min()) + 2
    xs = np.flatnonzero(a[top] != bg)
    assert len(xs) == 3 and xs[-1] - xs[0] == 2
    left = int(xs[0])
    neutral = int(a[top, left])
    assert neutral != bg and neutral not in commands
    beam = top + 4
    assert beam + 3 < h
    expected = np.full((5, w), bg, dtype=np.int8)
    expected[:3, left:left + 3] = neutral
    expected[3, left + 1] = neutral
    spans = _runs(a[beam] == neutral)
    assert len(spans) == 1
    lo, hi = spans[0]
    assert lo <= left and left + 2 <= hi and hi - lo >= 6
    expected[4, lo:hi + 1] = neutral
    assert np.array_equal(a[top:beam + 1], expected)
    assert np.all(a[2:top] == bg)

    # A panel is a maximal rectangle of horizontal runs of length >=3.
    # Props are separated columns, so they cannot create such a run.
    row_runs = {}
    for r in range(beam + 1, h):
        mask = (a[r] != bg) & (a[r] != neutral)
        for x, y in _runs(mask):
            if y - x + 1 >= 3:
                row_runs.setdefault((x, y), []).append(r)
    assert 2 <= len(row_runs) <= 5
    panels = []
    signals = set()
    explained = np.zeros(a.shape, dtype=bool)
    explained[:beam + 1] = a[:beam + 1] != bg
    for (x, y), rows in sorted(row_runs.items()):
        r0, r1 = min(rows), max(rows)
        assert rows == list(range(r0, r1 + 1)) and 2 <= len(rows) <= 4
        assert lo <= x <= y <= hi
        block = a[r0:r1 + 1, x:y + 1]
        colors, counts = np.unique(block, return_counts=True)
        assert len(colors) == 2 and bg not in colors and neutral not in colors
        assert counts[0] != counts[1] and min(counts) > 0
        signals.update(map(int, colors))
        majority = int(colors[np.argmax(counts)])
        explained[r0:r1 + 1, x:y + 1] = True
        prongs = []
        prong_tops = set()
        for c in range(x, y + 1):
            rs = np.flatnonzero(a[beam + 1:r0, c] == neutral) + beam + 1
            if len(rs):
                assert np.array_equal(rs, np.arange(rs[0], r0))
                prongs.append(c)
                prong_tops.add(int(rs[0]))
                explained[rs, c] = True
        assert 1 <= len(prongs) <= 4 and len(prong_tops) == 1
        assert all(b - c >= 2 for c, b in zip(prongs, prongs[1:]))
        props = []
        for c in range(x, y + 1):
            rs = np.flatnonzero(a[r1 + 1:, c] != bg) + r1 + 1
            if len(rs):
                assert np.array_equal(rs, np.arange(rs[0], h))
                color = int(a[rs[0], c])
                assert np.all(a[rs, c] == color)
                props.append((c, len(rs), color))
                explained[rs, c] = True
        assert props and all(b[0] - c[0] >= 2 for c, b in zip(props, props[1:]))
        assert max(p[1] for p in props) == h - r1 - 1
        panels.append(dict(box=(r0, r1, x, y), majority=majority,
                           prongs=prongs, prong_top=prong_tops.pop(), props=props,
                           block=block.copy()))
    assert len(signals) == 2 and not signals & commands
    assert all(p['box'][3] + 1 < q['box'][2] for p, q in zip(panels, panels[1:]))
    prop_colors = {c for p in panels for _, _, c in p['props']}
    assert not prop_colors & (signals | {neutral, bg})
    assert not commands & (signals | {neutral, bg})
    assert np.array_equal(explained, a != bg), 'unexplained scene cells'
    assert min(p['prong_top'] for p in panels) == beam + 1
    return dict(grid=a, background=bg, neutral=neutral, commands=commands,
                signals=tuple(sorted(signals)), panels=panels,
                marker=(top, left), beam=beam, span=(lo, hi))


def _landings(info):
    h = info['grid'].shape[0]
    displacements, tops = [], []
    for p in info['panels']:
        r0, r1, _, _ = p['box']
        support = max((n for _, n, c in p['props'] if c not in info['commands']), default=0)
        d = h - support - r1 - 1
        assert d >= 0
        displacements.append(d)
        tops.append(p['prong_top'] + d)
    beam = min(tops) - 1
    active = [i for i, t in enumerate(tops) if t == beam + 1]
    votes = {c: 0 for c in info['signals']}
    for i in active:
        p = info['panels'][i]
        votes[p['majority']] += len(p['prongs'])
    assert len(set(votes.values())) == 2, 'global tie is outside the family'
    winner = max(votes, key=votes.get)
    return displacements, beam, active, votes, winner


def _draw_answer(info, displacements, beam, winner):
    a = info['grid']
    out = np.full(a.shape, info['background'], dtype=np.int8)
    out[:2] = a[:2]
    top, left = info['marker']
    delta = beam - info['beam']
    lo, hi = info['span']
    out[top + delta:top + delta + 3, left:left + 3] = info['neutral']
    out[top + delta + 3, left + 1] = info['neutral']
    out[beam, lo:hi + 1] = info['neutral']
    out[top + delta + 1, left + 1] = winner
    for p, d in zip(info['panels'], displacements):
        r0, r1, x, y = p['box']
        out[r0 + d:r1 + d + 1, x:y + 1] = p['block']
        for c in p['prongs']:
            out[p['prong_top'] + d:r0 + d, c] = p['majority']
        for c, n, color in p['props']:
            if color not in info['commands']:
                out[-n:, c] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute solely from this input; arbitrary legal tower dimensions work."""
    info = _parse(input_grid)
    distances, beam, _, _, winner = _landings(info)
    return _draw_answer(info, distances, beam, winner)


def _body(cells, values, tag):
    return dict(cells=tuple(cells), values=tuple(map(int, values)), tag=tag)


def _simulate(bodies, height):
    """Independent cell-footprint rigid gravity for construction and checking."""
    bodies = [dict(b) for b in bodies]
    for _ in range(height):
        cells = [set(b['cells']) for b in bodies]
        targets = [{(r + 1, c) for r, c in s} for s in cells]
        moving = {i for i, s in enumerate(targets) if all(r < height for r, _ in s)}
        while True:
            blocked = {i for i in moving for j in range(len(bodies))
                       if j not in moving and targets[i] & cells[j]}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return bodies
        for i in moving:
            bodies[i]['cells'] = tuple((r + 1, c) for r, c in bodies[i]['cells'])
    raise AssertionError('gravity did not settle within the canvas height')


def _render(shape, bg, band, bodies):
    out = np.full(shape, bg, dtype=np.int8)
    out[0] = band
    occupied = set()
    for b in bodies:
        assert len(b['cells']) == len(b['values'])
        for cell, value in zip(b['cells'], b['values']):
            assert cell not in occupied
            occupied.add(cell)
            out[cell] = value
    return out


def _make_pair(rng, roles, *, witness=False, winner_index=0,
               count=None, max_bump=4, long_drop=False):
    bg, neutral, s0, s1, *prop_palette = roles
    signals = sorted((s0, s1))
    selected_count = 2 if witness else int(rng.integers(1, 3))
    selected = prop_palette[:selected_count]
    safe = prop_palette[2:]
    n = 4 if witness else (int(rng.integers(2, 5)) if count is None else count)
    # Logical records are shuffled before spatial allocation.
    if witness:
        k = int(rng.integers(3, 5))
        weights = [k, 1, 1, k - 1]
        active = {0, 1, 2}
        labels = [winner_index, 1 - winner_index, 1 - winner_index, 1 - winner_index]
        widths = [max(3, 2 * v - 1) for v in weights]
    else:
        widths = [3] * n
        for _ in range(int(rng.integers(0, 11))):
            i = int(rng.integers(n))
            if widths[i] < 7 and sum(widths) + n - 1 < 27:
                widths[i] += 1
        weights = [int(rng.integers(1, min(4, (v + 1) // 2) + 1)) for v in widths]
        na = int(rng.integers(1, n + 1))
        active = set(map(int, rng.choice(n, na, replace=False)))
        if na == 2:
            i, j = sorted(active)
            if weights[i] == weights[j]:
                weights[i], weights[j] = 2, 1
        options = []
        for bits in itertools.product((0, 1), repeat=n):
            tallies = [sum(weights[i] for i in active if bits[i] == v) for v in (0, 1)]
            if tallies[winner_index] > tallies[1 - winner_index] and (na == 1 or min(tallies) > 0):
                options.append(bits)
        labels = list(options[int(rng.integers(len(options)))])
    heights = list(map(int, rng.integers(2, 5, size=n)))
    lengths = list(map(int, rng.integers(1, 5, size=n)))
    bases = [p + l for p, l in zip(heights, lengths)]
    final_height = max(bases) + int(rng.integers(2, 5))
    final_supports = [final_height - bases[i] - (0 if i in active else int(rng.integers(1, 3))) for i in range(n)]
    bumps = list(map(int, rng.integers(0, max_bump + 1, size=n)))
    if witness:
        bumps[0] = int(rng.integers(1, 4))
        # Initially only the last, losing block holds the collector.
        bumps[3] = int(rng.integers(5, 7))
        for i in active:
            bumps[i] = min(bumps[i], 2)
        final_supports[3] = final_height - bases[3] - 2
    if long_drop:
        # One tower determines the initial beam, regardless of final membership.
        i = int(rng.integers(n))
        deficit = final_height - (final_supports[i] + bases[i])
        bumps[i] = int(rng.integers(6, 9)) + deficit
    for i in range(n):
        if final_supports[i] == 0 and bumps[i] == 0:
            bumps[i] = 1
    initial_supports = [t + b for t, b in zip(final_supports, bumps)]
    max_tower = max(s + b for s, b in zip(initial_supports, bases))
    h = max_tower + int(rng.integers(7, min(9, 30 - max_tower) + 1))
    assert h <= 30
    # Reserve unequal side margins, unequal gaps, and a variable collector span.
    order = list(map(int, rng.permutation(n)))
    gaps = [1] * (n - 1)
    spare = 30 - sum(widths) - sum(gaps)
    margins = [int(rng.integers(0, min(2, spare) + 1)), 0]
    spare -= margins[0]
    margins[1] = int(rng.integers(0, min(2, spare) + 1))
    spare -= margins[1]
    for j in range(len(gaps)):
        extra = int(rng.integers(0, min(2, spare) + 1))
        gaps[j] += extra
        spare -= extra
    w = sum(widths) + sum(gaps) + sum(margins)
    beam = h - max_tower - 1
    beam_left = int(rng.integers(0, margins[0] + 1))
    beam_right = w - 1 - int(rng.integers(0, margins[1] + 1))
    marker_left = int(rng.integers(beam_left, beam_right - 1))
    marker_top = beam - 4
    collector = [(r, c) for r in range(marker_top, marker_top + 3)
                 for c in range(marker_left, marker_left + 3)]
    collector += [(beam - 1, marker_left + 1)]
    collector += [(beam, c) for c in range(beam_left, beam_right + 1)]
    bodies = [_body(collector, [neutral] * len(collector), ('collector',))]
    panel_records = []
    col = margins[0]
    for rank, i in enumerate(order):
        pw, ph, pl = widths[i], heights[i], lengths[i]
        support = initial_supports[i]
        r1 = h - support - 1
        r0 = r1 - ph + 1
        majority, minority = signals[labels[i]], signals[1 - labels[i]]
        area = pw * ph
        if witness:
            minority_n = (area - 1) // 2 if i == 0 else max(1, area // 5)
        else:
            minority_n = int(rng.integers(1, (area - 1) // 2 + 1))
        pattern = np.full(area, majority, dtype=np.int8)
        style = int(rng.integers(3))
        if style == 0:
            indices = rng.permutation(area)
        elif style == 1:
            indices = np.arange(area)
        else:
            indices = np.arange(area).reshape(ph, pw).T.ravel()
        pattern[indices[:minority_n]] = minority
        if witness:
            desired_corner = minority if i == 0 else majority
            j = int(np.flatnonzero(pattern == desired_corner)[0])
            pattern[0], pattern[j] = pattern[j], pattern[0]
        block = pattern.reshape(ph, pw)
        candidates = list(range(col, col + pw, 2))
        prongs = sorted(map(int, rng.choice(candidates, weights[i], replace=False)))
        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(col, col + pw)]
        values = list(map(int, block.flat))
        for c in prongs:
            cells += [(r, c) for r in range(r0 - pl, r0)]
            values += [neutral] * pl
        bodies.append(_body(cells, values, ('panel', i)))
        panel_records.append((i, prongs, r0 - pl, majority))
        # Every prop is a separate, visible floor-grounded vertical stick.
        rng.shuffle(candidates)
        t = final_supports[i]
        bump = bumps[i]
        specs = []
        if bump:
            specs.append((support, selected[i % selected_count]))
            if t:
                specs.append((t, int(rng.choice(safe))))
        else:
            specs.append((support, int(rng.choice(safe))))
        while len(specs) < len(candidates) and rng.random() < .6:
            if bump and rng.random() < .6:
                # Another tall leg may require the other command as well.
                specs.append((support, selected[(i + 1) % selected_count]))
            elif t:
                specs.append((int(rng.integers(1, t + 1)), int(rng.choice(safe))))
            else:
                specs.append((support, selected[i % selected_count]))
        for c, (length, color) in zip(candidates, specs):
            cells = [(r, c) for r in range(h - length, h)]
            bodies.append(_body(cells, [color] * length, ('prop', color)))
        col += pw + (gaps[rank] if rank < n - 1 else 0)
    band = np.full(w, bg, dtype=np.int8)
    for c, color in zip(rng.choice(w, selected_count, replace=False), selected):
        band[c] = color
    inp = _render((h, w), bg, band, bodies)
    # Ground truth uses the complete rigid bodies, not the inference formula.
    initial = _simulate(bodies, h)
    assert all(a['cells'] == b['cells'] for a, b in zip(bodies, initial))
    survivors = [b for b in bodies if not (b['tag'][0] == 'prop' and b['tag'][1] in selected)]
    landed = _simulate(survivors, h)
    beam_body = next(b for b in landed if b['tag'][0] == 'collector')
    beam_drop = beam_body['cells'][0][0] - collector[0][0]
    final_beam = beam + beam_drop
    votes = {c: 0 for c in signals}
    for b in landed:
        if b['tag'][0] != 'panel':
            continue
        i = b['tag'][1]
        _, prongs, oldtop, majority = next(p for p in panel_records if p[0] == i)
        old = next(p for p in survivors if p['tag'] == b['tag'])
        d = b['cells'][0][0] - old['cells'][0][0]
        if oldtop + d == final_beam + 1:
            votes[majority] += len(prongs)
        vals = list(b['values'])
        panel_area = widths[i] * heights[i]
        vals[panel_area:] = [majority] * (len(vals) - panel_area)
        b['values'] = tuple(vals)
    assert votes[signals[0]] != votes[signals[1]]
    winner = max(votes, key=votes.get)
    vals = list(beam_body['values'])
    vals[4] = winner  # Row-major center of the nine-cell indicator.
    beam_body['values'] = tuple(vals)
    out = _render((h, w), bg, band, landed)
    pair = {'input': inp, 'output': out}
    assert np.array_equal(solve(inp), out)
    return pair


def _features(pair):
    info = _parse(pair['input'])
    ds, beam, active, votes, winner = _landings(info)
    panels = info['panels']
    def tally(ids, weight):
        return sum(weight(panels[i]) * (1 if panels[i]['majority'] == winner else -1) for i in ids)
    initial = [i for i, p in enumerate(panels) if p['prong_top'] == info['beam'] + 1]
    raw = sum(int(np.sum(p['block'] == winner)) - int(np.sum(p['block'] != winner)) for p in panels)
    return dict(panels=len(panels), active=len(active), prongs=sum(len(p['prongs']) for p in panels),
                contacts=sum(votes.values()), margin=abs(votes[info['signals'][0]] - votes[info['signals'][1]]),
                commands=len(info['commands']), beam_fall=beam - info['beam'],
                falls=tuple(sorted(set(ds))),
                weights=tuple(sorted(len(p['prongs']) for p in panels)),
                raw_wrong=raw < 0, block_wrong=tally(active, lambda p: 1) < 0,
                all_wrong=tally(range(len(panels)), lambda p: len(p['prongs'])) < 0,
                before_wrong=tally(initial, lambda p: len(p['prongs'])) < 0,
                corner_wrong=sum(len(panels[i]['prongs']) * (1 if int(panels[i]['block'][0, 0]) == winner else -1) for i in active) < 0,
                winner_index=info['signals'].index(winner), background=info['background'])


def _integer(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not lo <= value <= hi:
        raise ValueError(name + ' is out of range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    seed = int(seed)
    tr = _integer(num_train, 'num_train', 2, 10)
    te = _integer(num_test, 'num_test', 1, 4)
    if tr <= te:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.choice(10, 8, replace=False)))
    varying = bool(rng.integers(2))
    mode = int(rng.integers(2))
    phase = int(rng.integers(2))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', tr), ('test', te)):
        for i in range(count):
            roles = list(map(int, rng.choice(10, 8, replace=False))) if varying else palette
            for _ in range(32):
                pair = _make_pair(rng, roles, witness=split == 'train' and i == 0,
                                  winner_index=phase if i == 0 else 1 - phase if split == 'train' and i == 1 else int(rng.integers(2)),
                                  count=5 if split == 'test' and mode == 0 else None,
                                  long_drop=split == 'test' and mode == 1)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError('bounded distinct-input generation exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    tr, te = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen, train_features, test_features = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            distances, beam, active, votes, winner = _landings(info)
            assert np.array_equal(out, _draw_answer(info, distances, beam, winner))
            assert np.array_equal(inp[:2], out[:2])
            assert not np.array_equal(inp, out)
            assert beam >= info['beam'] and all(d >= 0 for d in distances)
            assert active and sum(votes.values()) >= 1
            # Exact minorities and majority pixels move as one unchanged block.
            for p, d in zip(info['panels'], distances):
                r0, r1, x, y = p['box']
                assert np.array_equal(out[r0 + d:r1 + d + 1, x:y + 1], p['block'])
                support = max((n for _, n, c in p['props'] if c not in info['commands']), default=0)
                assert r1 + d == inp.shape[0] - support - 1
            (train_features if split == 'train' else test_features).append(_features(pair))
    assert {f['winner_index'] for f in train_features} == {0, 1}
    assert any(all(f[k] for k in ('raw_wrong', 'block_wrong', 'all_wrong', 'before_wrong', 'corner_wrong')) for f in train_features)
    assert any(f['commands'] == 2 for f in train_features)
    # A visible increase in count or collector fall is the chosen split shift.
    assert (min(f['panels'] for f in test_features) > max(f['panels'] for f in train_features)
            or min(f['beam_fall'] for f in test_features) > max(f['beam_fall'] for f in train_features))


def self_test():
    """42 seeds, streaming measurements, no files or retained puzzle gallery."""
    shapes, signatures, structures, weights = set(), set(), set(), set()
    counts, active_counts, falls, backgrounds, margins = set(), set(), set(), set(), set()
    totals = dict(puzzles=0, pairs=0, witnesses=0, multiple_falls=0, stationary_beams=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        totals['puzzles'] += 1
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _parse(pair['input'])
                f = _features(pair)
                totals['pairs'] += 1
                totals['witnesses'] += all(f[k] for k in ('raw_wrong', 'block_wrong', 'all_wrong', 'before_wrong', 'corner_wrong'))
                totals['multiple_falls'] += len([d for d in f['falls'] if d > 0]) >= 2
                totals['stationary_beams'] += f['beam_fall'] == 0
                shapes.add(pair['input'].shape)
                weights.add(f['weights'])
                counts.add(f['panels'])
                active_counts.add(f['active'])
                falls.update(f['falls'])
                backgrounds.add(f['background'])
                margins.add(f['margin'])
                # Palette-normalized geometry hash; signatures retain no grids.
                normalized = np.full(info['grid'].shape, 0, dtype=np.int8)
                normalized[info['grid'] != info['background']] = 1
                structures.add(hashlib.sha256(normalized.tobytes()).digest())
                digest.update(pair['input'].tobytes())
                digest.update(pair['output'].tobytes())
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                # Reconstruct rigid bodies from input for a second mechanics check.
                bodies = []
                neutral = info['neutral']
                top, left = info['marker']
                lo, hi = info['span']
                cells = [(r, c) for r in range(top, top + 3) for c in range(left, left + 3)]
                cells += [(info['beam'] - 1, left + 1)]
                cells += [(info['beam'], c) for c in range(lo, hi + 1)]
                bodies.append(_body(cells, [neutral] * len(cells), ('collector',)))
                for i, p in enumerate(info['panels']):
                    r0, r1, x, y = p['box']
                    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(x, y + 1)]
                    cells += [(r, c) for c in p['prongs'] for r in range(p['prong_top'], r0)]
                    bodies.append(_body(cells, [neutral] * len(cells), ('panel', i)))
                    for c, n, color in p['props']:
                        if color not in info['commands']:
                            cells = [(r, c) for r in range(info['grid'].shape[0] - n, info['grid'].shape[0])]
                            bodies.append(_body(cells, [color] * n, ('prop', color)))
                settled = _simulate(bodies, info['grid'].shape[0])
                ds, beam, _, _, _ = _landings(info)
                expected_d = [beam - info['beam']] + ds
                actual_d = [b['cells'][0][0] - a['cells'][0][0] for a, b in zip(bodies, settled) if a['tag'][0] != 'prop']
                assert actual_d == expected_d
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(shapes) >= 35 and len(structures) >= 190
    assert counts == {2, 3, 4, 5} and active_counts >= {1, 2, 3, 4}
    assert len(weights) >= 15 and len(falls) >= 8 and len(margins) >= 4
    assert len(backgrounds) == 10 and totals['witnesses'] >= 42
    assert totals['multiple_falls'] >= 42 and totals['stationary_beams'] >= 1
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            p = generate_puzzle(123, tr, te)
            assert len(p['train']) == tr and len(p['test']) == te
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    first, second = generate_puzzle(7), generate_puzzle(7)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], second[s]) for k in a)
    p = generate_puzzle(5)
    p['test'][0]['output'] = p['test'][0]['output'].copy()
    info = _parse(p['test'][0]['input'])
    _, beam, _, _, _ = _landings(info)
    top, left = info['marker']
    p['test'][0]['output'][top + beam - info['beam'] + 1, left + 1] = info['neutral']
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return {**totals, 'shapes': len(shapes), 'structures': len(structures),
            'weight_profiles': len(weights), 'fall_distances': sorted(falls),
            'margins': sorted(margins)}
