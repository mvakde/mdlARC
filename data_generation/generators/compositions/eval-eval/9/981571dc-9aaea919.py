"""Mirror-addressed abacus: unary addition restricted by two reflection folds.

No files, network, learned state or parent generators are used. All retries and
tests are bounded. JSON, when requested by running this file, is compact.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import product
import json

import numpy as np


def _runs(bits):
    edges = np.diff(np.r_[False, bits, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def _connected(mask):
    points = set(zip(*np.nonzero(mask)))
    if not points:
        return False
    todo = [points.pop()]
    while todo:
        r, c = todo.pop()
        for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if q in points:
                points.remove(q)
                todo.append(q)
    return not points


def _orbits(nr, nc):
    seen, result = set(), []
    for r in range(nr):
        for c in range(nc):
            if (r, c) not in seen:
                group = tuple(sorted(product({r, nr - 1 - r},
                                             {c, nc - 1 - c})))
                seen.update(group)
                result.append(group)
    return result


def _capacity(height, token_height):
    return 1 + (height - 3 - token_height) // (token_height + 1)


def _check_grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _parse(input_grid):
    """Parse only visible input geometry; roles do not depend on color IDs."""
    a = _check_grid(input_grid)
    frame = int(a[0, 0])
    assert np.all(a[[0, -1]] == frame)
    assert np.all(a[:, [0, -1]] == frame)
    rows = np.flatnonzero(np.all(a == frame, axis=1)).tolist()
    cols = np.flatnonzero(np.all(a == frame, axis=0)).tolist()
    heights = [b - x - 1 for x, b in zip(rows, rows[1:])]
    widths = [b - x - 1 for x, b in zip(cols, cols[1:])]
    assert heights and widths and min(heights) >= 4 and min(widths) >= 1
    assert heights == heights[::-1] and widths == widths[::-1]
    nr, nc = len(heights), len(widths)
    bg = int(a[rows[1] - 3, cols[0] + 1])
    assert bg != frame
    wells, inks = {}, set()
    for r, h in enumerate(heights):
        for c, w in enumerate(widths):
            y, x = rows[r] + 1, cols[c] + 1
            tile = a[y:y + h, x:x + w]
            assert not np.any(tile == frame), 'frame color inside an input well'
            assert np.all(tile[-3] == bg), 'missing gap above controls'
            control = tile[-2:]
            if np.all(control == bg):
                role = 'unmarked'
            else:
                ink = int(control[-1, 0])
                assert ink not in (bg, frame) and np.all(control[-1] == ink)
                inks.add(ink)
                if np.all(control[0] == bg):
                    role = 'donor'
                else:
                    assert np.all(control[0] == ink)
                    role = 'receiver'
            body = tile[:-3]
            groups = _runs(np.any(body != bg, axis=1))
            if not groups:
                assert role == 'unmarked'
                wells[r, c] = dict(r=r, c=c, y=y, x=x, h=h, w=w,
                                   role=role, count=0, mask=None)
                continue
            # A stack's tight horizontal span and mask are locally observable.
            xx = np.flatnonzero(np.any(body != bg, axis=0))
            left, right = int(xx[0]), int(xx[-1]) + 1
            first, end = groups[0]
            block = body[first:end, left:right]
            mask = block != bg
            assert _connected(mask)
            assert np.all(mask.any(axis=0)) and np.all(mask.any(axis=1))
            values = np.unique(body[body != bg])
            assert len(values) == 1
            color = int(values[0])
            th = mask.shape[0]
            assert groups[-1][1] == h - 3, 'stack does not meet its baseline'
            assert all(b - s == th for s, b in groups)
            assert all(t - s == th + 1 for (s, _), (t, _) in zip(groups, groups[1:]))
            for start, stop in groups:
                assert np.array_equal(body[start:stop, left:right] != bg, mask)
            wells[r, c] = dict(r=r, c=c, y=y, x=x, h=h, w=w,
                               role=role, count=len(groups), mask=mask.copy(),
                               color=color, left=left, bottom=h - 3 - th)
    assert len(inks) == 1, 'exactly one geometric control color is required'
    ink = inks.pop()
    assert all(s.get('color') != ink for s in wells.values())
    accounts = []
    for orbit in _orbits(nr, nc):
        donors = [wells[p] for p in orbit if wells[p]['role'] == 'donor']
        receivers = [wells[p] for p in orbit if wells[p]['role'] == 'receiver']
        if donors or receivers:
            assert len(receivers) == 1 and donors
            dst = receivers[0]
            total = sum(s['count'] for s in donors)
            assert dst['count'] + total <= _capacity(dst['h'], dst['mask'].shape[0])
            accounts.append(dict(orbit=orbit, donors=donors, receiver=dst, total=total))
    assert accounts
    return dict(grid=a, frame=frame, bg=bg, ink=ink, heights=heights,
                widths=widths, nr=nr, nc=nc, wells=wells, accounts=accounts)


def _stamp(grid, well, level, color):
    mask = well['mask']
    th, tw = mask.shape
    y = well['y'] + well['bottom'] - level * (th + 1)
    x = well['x'] + well['left']
    assert y >= well['y'] and x + tw <= well['x'] + well['w']
    grid[y:y + th, x:x + tw][mask] = color


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from this input alone, without mutating the input."""
    p = _parse(input_grid)
    out = p['grid'].copy()
    for well in p['wells'].values():
        y, x, h, w = (well[k] for k in ('y', 'x', 'h', 'w'))
        out[y + h - 2:y + h, x:x + w] = p['bg']
        if well['role'] == 'donor':
            for k in range(well['count']):
                _stamp(out, well, k, p['ink'])
    for account in p['accounts']:
        dst = account['receiver']
        for k in range(dst['count'], dst['count'] + account['total']):
            _stamp(out, dst, k, dst['color'])
    return out


def _mask(rng, width, force_short=False, force_wide=False):
    h = 1 if force_short else int(rng.integers(1, 3))
    w = int(rng.integers(2 if force_wide else 1, width + 1))
    mask = np.ones((h, w), dtype=bool)
    if h == 2 and w > 1 and rng.random() < 0.65:
        mask[0, int(rng.integers(1, w + 1)):] = False
        if rng.integers(2):
            mask = mask[:, ::-1]
        if rng.integers(2):
            mask = mask[::-1]
    return mask.copy()


def _geometry(rng, three_rows):
    nc = int(rng.integers(3, 6))
    if three_rows:
        a, b = (int(v) for v in rng.integers(6, 9, size=2))
        heights = [a, b, a]
    else:
        h = int(rng.integers(11, 14))
        heights = [h, h]
    half = rng.integers(2, 6, size=(nc + 1) // 2).tolist()
    widths = half + (half[:-1] if nc % 2 else half)[::-1]
    # Shrink symmetric pairs, with a strict finite bound, if width exceeds 30.
    for i in range(nc):
        if sum(widths) + nc + 1 <= 30:
            break
        j = i % ((nc + 1) // 2)
        widths[j] = widths[nc - 1 - j] = 2
    assert sum(heights) + len(heights) + 1 <= 30
    assert sum(widths) + nc + 1 <= 30
    return heights, widths


def _draw_scene(heights, widths, wells, colors):
    """Ground truth comes from construction descriptors, never from solve()."""
    bg, frame, ink = colors
    shape = (sum(heights) + len(heights) + 1,
             sum(widths) + len(widths) + 1)
    inp = np.full(shape, frame, dtype=np.int8)
    out = inp.copy()
    nr, nc = len(heights), len(widths)
    extra = {}
    for orbit in _orbits(nr, nc):
        destinations = [p for p in orbit if wells[p]['role'] == 'receiver']
        if destinations:
            extra[destinations[0]] = sum(wells[p]['count'] for p in orbit
                                        if wells[p]['role'] == 'donor')
    for pos, s in wells.items():
        h, w = s['h'], s['w']
        a, b = np.full((h, w), bg, np.int8), np.full((h, w), bg, np.int8)
        mask = s['mask']
        th, tw = mask.shape
        bottom = h - 3 - th
        for k in range(s['count']):
            yy = bottom - k * (th + 1)
            a[yy:yy + th, s['left']:s['left'] + tw][mask] = s['color']
        for k in range(s['count'] + extra.get(pos, 0)):
            yy = bottom - k * (th + 1)
            assert yy >= 0
            b[yy:yy + th, s['left']:s['left'] + tw][mask] = (
                ink if s['role'] == 'donor' else s['color'])
        if s['role'] == 'donor':
            a[-1] = ink
        elif s['role'] == 'receiver':
            a[-2:] = ink
        y, x = s['y'], s['x']
        inp[y:y + h, x:x + w], out[y:y + h, x:x + w] = a, b
    return dict(input=inp, output=out)


def _make_pair(rng, colors, *, rich=False, three_rows=False, three_donors=False):
    heights, widths = _geometry(rng, three_rows)
    nr, nc = len(heights), len(widths)
    palette = [c for c in range(10) if c not in colors]
    palette = rng.choice(palette, int(rng.integers(1, 8)), replace=False).tolist()
    wells = {}
    y = 1
    for r, h in enumerate(heights):
        x = 1
        for c, w in enumerate(widths):
            mask = _mask(rng, w)
            wells[r, c] = dict(y=y, x=x, h=h, w=w, mask=mask,
                               left=int(rng.integers(w - mask.shape[1] + 1)),
                               color=int(rng.choice(palette)), role='unmarked',
                               count=int(rng.integers(_capacity(h, mask.shape[0]) + 1)))
            x += w + 1
        y += h + 1
    groups = [g for g in _orbits(nr, nc) if len(g) > 1]
    fours = [g for g in groups if len(g) == 4]
    anchor = fours[int(rng.integers(len(fours)))]
    chosen = {anchor}
    chosen.update(g for g in groups if rng.random() < 0.70)
    if len(chosen) < 2:
        others = [g for g in groups if g != anchor]
        chosen.add(others[int(rng.integers(len(others)))])
    # Iterate in canonical order: hash randomization cannot change the RNG stream.
    for orbit in groups:
        if orbit not in chosen:
            continue
        dst_pos = orbit[int(rng.integers(len(orbit)))]
        dst = wells[dst_pos]
        if orbit == anchor or _capacity(dst['h'], dst['mask'].shape[0]) < 2:
            # Short destination tokens make room for the arithmetic witness.
            dst['mask'] = _mask(rng, dst['w'], force_short=True)
            dst['left'] = int(rng.integers(dst['w'] - dst['mask'].shape[1] + 1))
        capacity = _capacity(dst['h'], dst['mask'].shape[0])
        maximum = min(len(orbit) - 1, capacity - 1, 2)
        nsrc = 3 if three_donors and orbit == anchor else (
            2 if rich and orbit == anchor else int(rng.integers(1, maximum + 1)))
        candidates = [p for p in orbit if p != dst_pos]
        rng.shuffle(candidates)
        if orbit == anchor:
            opposite = (nr - 1 - dst_pos[0], nc - 1 - dst_pos[1])
            candidates.remove(opposite)
            candidates.insert(0, opposite)
        src_positions = candidates[:nsrc]
        if rich and orbit == anchor:
            s = wells[src_positions[0]]
            s['mask'] = _mask(rng, s['w'], force_wide=True)
            s['left'] = int(rng.integers(s['w'] - s['mask'].shape[1] + 1))
        options = []
        ranges = [range(1, _capacity(wells[p]['h'], wells[p]['mask'].shape[0]) + 1)
                  for p in src_positions]
        for counts in product(*ranges):
            if rich and orbit == anchor and len(set(counts)) == 1:
                continue
            for d in range(1, capacity - sum(counts) + 1):
                options.append((counts, d))
        assert options, 'chosen geometry cannot fit its intended operands'
        counts, d = options[int(rng.integers(len(options)))]
        dst['role'], dst['count'] = 'receiver', d
        for pos, count in zip(src_positions, counts):
            wells[pos]['role'], wells[pos]['count'] = 'donor', count
        if rich and orbit == anchor:
            passive = [p for p in orbit if wells[p]['role'] == 'unmarked']
            assert passive
            p = passive[0]
            wells[p]['count'] = int(rng.integers(1, _capacity(
                wells[p]['h'], wells[p]['mask'].shape[0]) + 1))
    return _draw_scene(heights, widths, wells, colors)


def _evidence(parses):
    accounts = [a for p in parses for a in p['accounts']]
    operands = [(tuple(s['count'] for s in a['donors']), a['receiver']['count'])
                for a in accounts]
    totals, dests = [sum(s) for s, d in operands], [d for s, d in operands]
    assert len(set(totals)) > 1 and len(set(dests)) > 1
    assert len({s + d for s, d in zip(totals, dests)}) > 1
    assert any(s * dests[0] != totals[0] * d for s, d in zip(totals, dests))
    rivals = (lambda s, d: 2 * d, lambda s, d: 2 * sum(s),
              lambda s, d: d * sum(s), lambda s, d: max(d, sum(s)),
              lambda s, d: sum(s), lambda s, d: d + len(s),
              lambda s, d: d + max(s))
    assert all(any(f(s, d) != d + sum(s) for s, d in operands) for f in rivals)
    assert any(len(s) > 1 and len(set(s)) > 1 for s, d in operands)
    # Quantities, not donor pixel areas; omitted controls and either omitted fold.
    pixel_rival = passive_rival = row_only = col_only = global_rival = False
    for p in parses:
        all_donors = [s for s in p['wells'].values() if s['role'] == 'donor']
        for a in p['accounts']:
            dst, total = a['receiver'], a['total']
            pixel_rival |= sum(s['count'] * int(s['mask'].sum()) for s in a['donors']) != total
            passive_rival |= any(p['wells'][q]['role'] == 'unmarked' and
                                 p['wells'][q]['count'] > 0 for q in a['orbit'])
            row_only |= sum(s['count'] for s in a['donors'] if s['c'] == dst['c']) != total
            col_only |= sum(s['count'] for s in a['donors'] if s['r'] == dst['r']) != total
            global_rival |= sum(s['count'] for s in all_donors) != total
    assert pixel_rival and passive_rival and row_only and col_only and global_rival
    # Odd columns teach two-member through-axis accounts before three-row tests.
    assert any(len(a['orbit']) == 2 for a in accounts)


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError('pair counts must be integers in the legal ranges')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    colors = tuple(int(v) for v in rng.choice(10, 3, replace=False))
    three_rows = bool(rng.integers(2))
    for _ in range(64):
        first = [_make_pair(rng, colors, rich=True), _make_pair(rng, colors)]
        try:
            _evidence([_parse(q['input']) for q in first])
        except AssertionError:
            continue
        if _key(first[0]['input']) != _key(first[1]['input']):
            break
    else:
        raise RuntimeError('bounded demonstration construction exhausted')
    if rng.integers(2):
        first.reverse()
    puzzle = dict(train=first, test=[])
    seen = {_key(q['input']) for q in first}
    for split, number in (('train', nt - 2), ('test', ns)):
        for _ in range(number):
            for _attempt in range(32):
                pair = _make_pair(rng, colors,
                                  three_rows=split == 'test' and three_rows,
                                  three_donors=split == 'test' and not three_rows)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('bounded distinct-input construction exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    a, b = (_check_grid(pair[k]) for k in ('input', 'output'))
    assert a.shape == b.shape
    p = _parse(a)
    assert len(p['accounts']) >= 2
    assert np.array_equal(solve(a), b), 'output does not implement orbit addition'
    assert np.any(a != b), 'answer must perform a nonempty transfer'
    final = {(s['r'], s['c']): s['count'] + account['total']
             for account in p['accounts'] for s in [account['receiver']]}
    # Independently inspect each returned well's body and control rows. Counting
    # returned row-components catches a missing complete token, not only a color.
    for pos, s in p['wells'].items():
        y, x, h, w = (s[k] for k in ('y', 'x', 'h', 'w'))
        old, new = a[y:y + h, x:x + w], b[y:y + h, x:x + w]
        assert np.all(new[-3:] == p['bg'])
        if s['role'] == 'unmarked':
            assert np.array_equal(old, new), 'unmarked well changed'
            continue
        count = final[pos] if s['role'] == 'receiver' else s['count']
        groups = _runs(np.any(new[:-3] != p['bg'], axis=1))
        assert len(groups) == count, 'wrong token count'
        th, tw = s['mask'].shape
        assert groups == [(h - 3 - th - k * (th + 1), h - 3 - k * (th + 1))
                          for k in range(count - 1, -1, -1)]
        color = p['ink'] if s['role'] == 'donor' else s['color']
        for start, stop in groups:
            block = new[start:stop, s['left']:s['left'] + tw]
            assert np.array_equal(block != p['bg'], s['mask'])
            assert np.all(block[s['mask']] == color)
    # Both original frame and all original background outside wells are retained.
    assert np.array_equal(b[a == p['frame']], a[a == p['frame']])
    return p


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[k], list) for k in ('train', 'test'))
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, parses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            p = _validate_pair(pair)
            key = _key(p['grid'])
            assert key not in seen, 'duplicate input'
            seen.add(key)
            parses[split].append(p)
    roles = {(p['bg'], p['frame'], p['ink']) for v in parses.values() for p in v}
    assert len(roles) == 1, 'scene conventions changed within a puzzle'
    _evidence(parses['train'])
    assert all(p['nr'] == 2 for p in parses['train'])
    assert all(len(a['donors']) <= 2 for p in parses['train'] for a in p['accounts'])
    assert all(p['nr'] == 3 for p in parses['test']) or all(
        p['nr'] == 2 and any(len(a['donors']) == 3 for a in p['accounts'])
        for p in parses['test'])


def _fingerprint(puzzle):
    h = sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for key in ('input', 'output'):
                a = np.asarray(pair[key], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded semantic/variation study, retaining metrics rather than grids."""
    signatures, shapes, band_shapes, masks, roles = set(), set(), set(), set(), set()
    donor_counts, operands, shifts = set(), set(), set()
    empty = repeated = nonuniform = diagonal = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = _fingerprint(puzzle)
        assert signature == _fingerprint(generate_puzzle(seed))
        signatures.add(signature)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = np.asarray(pair['input'])
                old = a.copy()
                assert np.array_equal(solve(a), pair['output'])
                assert np.array_equal(a, old), 'solve mutated its input'
                p = _parse(a)
                shapes.add(a.shape)
                band_shapes.add((tuple(p['heights']), tuple(p['widths'])))
                roles.add((p['bg'], p['frame'], p['ink']))
                nonuniform += len(set(p['widths'])) > 1 or len(set(p['heights'])) > 1
                colors = [s['color'] for s in p['wells'].values() if s['count']]
                repeated += len(set(colors)) < len(colors)
                empty += any(s['count'] == 0 for s in p['wells'].values())
                for s in p['wells'].values():
                    if s['count']:
                        masks.add((s['mask'].shape, s['mask'].tobytes()))
                for account in p['accounts']:
                    dst = account['receiver']
                    source = account['donors']
                    donor_counts.add(len(source))
                    operands.add((tuple(s['count'] for s in source), dst['count']))
                    diagonal += any(s['r'] != dst['r'] and s['c'] != dst['c'] for s in source)
                if split == 'test':
                    shifts.add('three_rows' if p['nr'] == 3 else 'three_donors')
    assert len(signatures) == 42 and len(shapes) >= 20 and len(band_shapes) >= 60
    assert len(masks) >= 20 and len(roles) >= 30 and len(operands) >= 12
    assert donor_counts == {1, 2, 3} and shifts == {'three_rows', 'three_donors'}
    assert min(min(s) for s in shapes) <= 15 and max(max(s) for s in shapes) >= 28
    assert empty >= 10 and repeated >= 30 and nonuniform >= 30 and diagonal >= 80
    legal = 0
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(101, nt, ns)
            assert (len(p['train']), len(p['test'])) == (nt, ns)
            legal += 1
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Color roles and horizontal correspondence must be input-derived. A taller
    # token below also checks the solver beyond the sampler's 1/2-row masks.
    sample = generate_puzzle(0, 2, 1)['test'][0]
    assert np.array_equal(solve(sample['input'][:, ::-1]), sample['output'][:, ::-1])
    lookup = np.array([7, 9, 4, 0, 6, 2, 1, 8, 5, 3], dtype=np.int8)
    assert np.array_equal(solve(lookup[sample['input']]), lookup[sample['output']])
    tall_mask = np.array([[False, True, False], [True, True, True],
                          [False, True, False]])
    wells = {(r, c): dict(y=1 + 14 * r, x=1 + 4 * c, h=13, w=3,
                          mask=tall_mask.copy(), left=0, color=3,
                          role='unmarked', count=1)
             for r in range(2) for c in range(3)}
    for pos in ((0, 0), (0, 1)):
        wells[pos]['role'], wells[pos]['color'] = 'receiver', 4
    for pos in ((1, 2), (1, 1)):
        wells[pos]['role'] = 'donor'
    _validate_pair(_draw_scene([13, 13], [3, 3, 3], wells, (0, 5, 8)))
    # Mutations exercise arithmetic, highlighting, marker removal, passive-well
    # preservation, frame preservation and the destination payload template.
    original = generate_puzzle(0, 2, 1)
    p = _parse(original['train'][0]['input'])
    dst = p['accounts'][0]['receiver']
    source = p['accounts'][0]['donors'][0]
    unmarked = next(s for s in p['wells'].values()
                    if s['role'] == 'unmarked' and s['count'])
    for defect in ('sum', 'highlight', 'control', 'unmarked', 'frame', 'receiver_payload'):
        broken = {k: [{r: v.copy() for r, v in pair.items()} for pair in pairs]
                  for k, pairs in original.items()}
        out = broken['train'][0]['output']
        if defect == 'sum':
            _stamp(out, dst, dst['count'], p['bg'])
        elif defect == 'highlight':
            _stamp(out, source, 0, source['color'])
        elif defect == 'control':
            out[source['y'] + source['h'] - 1,
                source['x']:source['x'] + source['w']] = p['ink']
        elif defect == 'unmarked':
            _stamp(out, unmarked, 0, p['ink'])
        elif defect == 'frame':
            out[0, 0] = p['bg']
        else:
            other = p['accounts'][1]['receiver']
            _stamp(out, dst, 0, other['color'] if other['color'] != dst['color'] else p['bg'])
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError('damaged output accepted: ' + defect)
    return dict(seeds=42, distinct_puzzles=len(signatures), grid_shapes=len(shapes),
                band_geometries=len(band_shapes), token_masks=len(masks),
                role_palettes=len(roles), operand_vectors=len(operands),
                shifts=sorted(shifts), empty_pairs=empty, repeated_color_pairs=repeated,
                nonuniform_boards=nonuniform, diagonal_accounts=diagonal,
                legal_count_combinations=legal, mutation_checks=6,
                input_only_generalization_checks=3)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
