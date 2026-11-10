"""Cavity resonance: count-gated radial echoes and inward signal collection.

Only standard-library modules and NumPy are used. All solving conventions are
visible geometric/color roles; random policies affect examples, not the rule.
"""
from __future__ import annotations

import json

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask, diagonal=False):
    points = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    steps = STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else STEPS
    while points:
        todo = [min(points)]
        points.remove(todo[0])
        group = []
        while todo:
            r, c = todo.pop()
            group.append((r, c))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in points:
                    points.remove(q)
                    todo.append(q)
        groups.append(group)
    return groups


def _perimeter(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _d4(a):
    return [np.rot90(a, k) for k in range(4)] + [np.rot90(a.T, k) for k in range(4)]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _mask_catalog(n):
    """All interior masks, rather than a bank of parent patterns."""
    cavities, clues = {}, {}
    for bits in range(1 << (n * n)):
        mask = np.array([(bits >> j) & 1 for j in range(n * n)],
                        dtype=bool).reshape(n, n)
        k = len(_components(mask))
        cavities.setdefault(k, []).append(mask)
        if int(mask.sum()) == k:
            clues.setdefault(k, []).append(mask)
    return cavities, clues


_CATALOGS = {t: _mask_catalog(t - 2) for t in (3, 5)}


def _analyze(input_grid):
    """Find the hub by its exclusive carrier and intact uniform perimeter."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not np.all(_perimeter(a) == bg):
        raise ValueError("the source construction needs a background margin")
    interpretations = []
    for ink in map(int, np.unique(a)):
        if ink == bg:
            continue
        rr, cc = np.nonzero(a == ink)
        hr, hc = int(rr.min()), int(cc.min())
        t = int(rr.max()) - hr + 1
        if t not in (3, 5) or int(cc.max()) - hc + 1 != t:
            continue
        hub = a[hr:hr + t, hc:hc + t]
        if not (np.all(_perimeter(hub) == ink)
                and set(map(int, np.unique(hub))) <= {bg, ink}):
            continue
        dots = hub == bg
        k = int(dots.sum())
        if k != len(_components(dots)):
            continue
        if not (2 * t <= hr <= a.shape[0] - 3 * t
                and 2 * t <= hc <= a.shape[1] - 3 * t):
            continue
        panels = {}
        occupied = np.zeros(a.shape, dtype=bool)
        occupied[hr:hr + t, hc:hc + t] = True
        valid = True
        m = t // 2
        for d in DIRECTIONS:
            dr, dc = d
            r, c = hr + t * dr, hc + t * dc
            tile = a[r:r + t, c:c + t]
            if np.all(tile == bg):
                continue
            values, counts = np.unique(tile, return_counts=True)
            body = int(values[int(np.argmax(counts))])
            sample = int(tile[m - m * dr, m - m * dc])
            if (np.any(tile == ink) or np.any(_perimeter(tile) == bg)
                    or int(counts.max()) <= t * t // 2
                    or body == bg or sample in (bg, ink, body)):
                valid = False
                break
            holes = tile == bg
            panels[d] = dict(tile=tile.copy(), body=body, sample=sample,
                             holes=holes, count=len(_components(holes)))
            occupied[r:r + t, c:c + t] = True
        if (valid and 2 <= len(panels) <= 8 and np.all(a[~occupied] == bg)
                and any(p['count'] == k for p in panels.values())):
            interpretations.append(dict(background=bg, ink=ink, t=t, hr=hr,
                                        hc=hc, k=k, panels=panels, dots=dots))
    if len(interpretations) != 1:
        raise ValueError("input must identify exactly one dotted hub and radial construction")
    return interpretations[0]


def _render(a, info):
    out = a.copy()
    t, r, c, ink, bg, k = (info[x] for x in ('t', 'hr', 'hc', 'ink', 'background', 'k'))
    m = t // 2
    out[r:r + t, c:c + t] = ink
    for (dr, dc), p in info['panels'].items():
        if p['count'] != k:
            continue
        out[r + m + dr, c + m + dc] = p['sample']
        rr, cc = r + 2 * t * dr, c + 2 * t * dc
        echo = p['tile'][::-1, ::-1]
        out[rr:rr + t, cc:cc + t] = np.where(echo == bg, ink, echo)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only equality gate, inward sampling, sealing and radial half-turn."""
    a = _grid(input_grid)
    return _render(a, _analyze(a))


def _mask(rng, t, k, kind=None):
    n = t - 2
    if kind == 'large':
        assert k == 1
        inner = np.ones((n, n), dtype=bool)
    elif kind == 'diagonal':
        assert t == 5 and k == 2
        inner = np.zeros((3, 3), dtype=bool)
        inner[0, 0] = inner[1, 1] = inner[1, 2] = True
        inner = _choice(rng, _d4(inner)).copy()
    elif kind == 'connected_three':
        inner = _choice(rng, [v for v in _CATALOGS[t][0][1] if v.sum() == 3]).copy()
    elif kind == 'diagonal_pair':
        inner = np.zeros((3, 3), dtype=bool)
        inner[0, 0] = inner[1, 1] = True
        inner = _choice(rng, _d4(inner)).copy()
    elif k == 1 and rng.random() < .30:
        inner = np.ones((n, n), dtype=bool)
    else:
        inner = _choice(rng, _CATALOGS[t][0][k]).copy()
    return np.pad(inner, 1)


def _panel(rng, t, d, bg, palette, k, witness=False, kind=None, repeat=False):
    body = int(_choice(rng, palette))
    signal = int(_choice(rng, [x for x in palette if x != body]))
    tile = np.full((t, t), body, dtype=np.int8)
    holes = _mask(rng, t, k, kind)
    tile[holes] = bg
    dr, dc = d
    m = t // 2
    inward = (m - m * dr, m - m * dc)
    tile[inward] = signal
    positions = [(r, c) for r in range(t) for c in range(t)
                 if not holes[r, c] and (r, c) != inward]
    if repeat:
        q = _choice(rng, positions)
        tile[q] = signal
        positions.remove(q)
    if witness:
        aux = int(_choice(rng, [x for x in palette if x not in (body, signal)]))
        for j in rng.permutation(len(positions)):
            q = positions[int(j)]
            tile[q] = aux
            if len({v.tobytes() for v in _d4(tile)}) == 8:
                break
            tile[q] = body
        else:
            raise AssertionError("no directional witness can be constructed")
    else:
        # Repeated signals, several auxiliary colors, or no decoration. Budget
        # depends on the visible cavity area so body majority is never lost.
        room = int(np.count_nonzero(tile == body)) - (t * t // 2 + 1)
        extras = int(rng.integers(min(5, room) + 1))
        for j in rng.permutation(len(positions))[:extras]:
            q = positions[int(j)]
            tile[q] = signal if rng.random() < .45 else int(_choice(rng, palette))
    return dict(tile=tile, holes=holes, count=k, body=body, sample=signal)


def _pair(rng, t, target, directions, active, bg, witness=False, repeat_direction=None):
    # Ink is exclusive to the central hub. Body and signal roles can be reused
    # across panels; the same numeric color can change roles in another pair.
    colors = list(map(int, rng.permutation([x for x in range(10) if x != bg])))
    ink = colors[0]
    palette = colors[1:int(rng.integers(4 if witness else 3, 10))]
    h, w = map(int, rng.integers(5 * t, 31, size=2))
    # Choose the complete 5-by-5 tile footprint's offset, including edge fit.
    r = 2 * t + int(_choice(rng, (0, h - 5 * t, int(rng.integers(h - 5 * t + 1)))))
    c = 2 * t + int(_choice(rng, (0, w - 5 * t, int(rng.integers(w - 5 * t + 1)))))
    a = np.full((h, w), bg, dtype=np.int8)
    hub = np.full((t, t), ink, dtype=np.int8)
    dots = _choice(rng, _CATALOGS[t][1][target])
    hub[1:-1, 1:-1][dots] = bg
    a[r:r + t, c:c + t] = hub
    panels = {}
    cap = 1 if t == 3 else 5
    inactive_index = active_index = 0
    for d in DIRECTIONS:
        if d not in directions:
            continue
        is_active = d in active
        k = target if is_active else int(_choice(rng, [j for j in range(cap + 1) if j != target]))
        kind = None
        if witness:
            if is_active:
                if target == 1:
                    kind = 'large' if active_index == 0 else 'connected_three' if active_index == 1 else None
                else:
                    kind = 'diagonal' if active_index == 0 else 'diagonal_pair' if active_index == 1 else None
                active_index += 1
            elif inactive_index == 0:
                # An actual count/area inversion, in both directions of the gate.
                k, kind = (2, 'diagonal') if target == 1 else (1, 'large')
            inactive_index += not is_active
        p = _panel(rng, t, d, bg, palette, k, witness=is_active and witness,
                   kind=kind, repeat=d == repeat_direction)
        panels[d] = p
        dr, dc = d
        a[r + t * dr:r + t * dr + t, c + t * dc:c + t * dc + t] = p['tile']
    if witness and len({panels[d]['sample'] for d in active}) == 1:
        # Guarantee a multicolor center without sacrificing the unique sample
        # or the asymmetric witness. A color swap preserves both properties.
        d = next(d for d in DIRECTIONS if d in active)
        p = panels[d]
        old = p['sample']
        new = int(_choice(rng, [x for x in palette if x not in (p['body'], old)]))
        old_mask, new_mask = p['tile'] == old, p['tile'] == new
        p['tile'][old_mask], p['tile'][new_mask] = new, old
        p['sample'] = new
        dr, dc = d
        a[r + t * dr:r + t * dr + t, c + t * dc:c + t * dc + t] = p['tile']
    info = dict(t=t, hr=r, hc=c, ink=ink, background=bg, k=target, panels=panels)
    return dict(input=a, output=_render(a, info))


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _fresh_pair(rng, used, *args, **kwargs):
    for _ in range(32):
        pair = _pair(rng, *args, **kwargs)
        if _key(pair['input']) not in used:
            used.add(_key(pair['input']))
            return pair
    raise RuntimeError("distinct input sampling exhausted its bounded attempts")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    mode = str(_choice(rng, ('scale', 'sparse', 'new_count')))
    fixed_bg = bool(rng.random() < .25)
    base_bg = int(rng.integers(10))

    def background():
        return base_bg if fixed_bg else int(rng.integers(10))

    # Complementary directions teach every compass direction; one overlap shows
    # a repeated signal after its unique-position witness. Neither ring is all
    # active, and the second also demonstrates empty source slots.
    corners = [d for d in DIRECTIONS if all(d)]
    cardinals = [d for d in DIRECTIONS if not all(d)]
    first_active = {corners[int(j)] for j in rng.permutation(4)[:2]}
    first_active |= {cardinals[int(j)] for j in rng.permutation(4)[:2]}
    overlap = _choice(rng, [d for d in DIRECTIONS if d in first_active])
    second_active = (set(DIRECTIONS) - first_active) | {overlap}
    extra = [d for d in DIRECTIONS if d in first_active]
    extra.remove(overlap)
    extra = [extra[int(j)] for j in rng.permutation(3)[:int(rng.integers(1, 3))]]
    used = set()
    train = [
        _fresh_pair(rng, used, 5, 1, set(DIRECTIONS), first_active, background(), witness=True),
        _fresh_pair(rng, used, 5, 2, second_active | set(extra), second_active,
                    background(), witness=True, repeat_direction=overlap),
    ]
    for _ in range(nt - 2):
        t = 5 if mode == 'scale' else int(_choice(rng, (3, 5)))
        cap = 1 if t == 3 else 5
        target = int(_choice(rng, (1, 2) if mode == 'new_count' and t == 5
                             else (1,) if mode == 'new_count' else tuple(range(cap + 1))))
        n = int(rng.integers(5 if mode == 'sparse' else 2, 9))
        ds = {DIRECTIONS[int(j)] for j in rng.permutation(8)[:n]}
        order = [d for d in DIRECTIONS if d in ds]
        count = int(rng.integers(1, n + 1))
        active = {order[int(j)] for j in rng.permutation(n)[:count]}
        train.append(_fresh_pair(rng, used, t, target, ds, active, background()))
    tests = []
    for _ in range(ne):
        t = 3 if mode == 'scale' else 5
        cap = 1 if t == 3 else 5
        target = int(_choice(rng, (0, 3, 4, 5) if mode == 'new_count' else tuple(range(cap + 1))))
        n = int(rng.integers(2, 5 if mode == 'sparse' else 9))
        ds = {DIRECTIONS[int(j)] for j in rng.permutation(8)[:n]}
        order = [d for d in DIRECTIONS if d in ds]
        active = {order[int(j)] for j in rng.permutation(n)[:int(rng.integers(1, n))]}
        tests.append(_fresh_pair(rng, used, t, target, ds, active, background()))
    train = [train[int(j)] for j in rng.permutation(nt)]
    puzzle = dict(train=train, test=tests)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    """Scalar destination/preservation checks separate from the slice renderer."""
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys'
    a, b = _grid(pair['input']), _grid(pair['output'])
    assert a.shape == b.shape, 'canvas shape changed'
    info = _analyze(a)
    t, r, c, ink, bg, k = (info[x] for x in ('t', 'hr', 'hc', 'ink', 'background', 'k'))
    m = t // 2
    assigned = np.zeros(a.shape, dtype=bool)
    assigned[r:r + t, c:c + t] = True
    hub_values = {(m + dr, m + dc): p['sample'] for (dr, dc), p in info['panels'].items()
                  if p['count'] == k}
    for y in range(t):
        for x in range(t):
            assert b[r + y, c + x] == hub_values.get((y, x), ink), 'wrong hub signal or uncleared dot'
    for (dr, dc), p in info['panels'].items():
        sr, sc = r + t * dr, c + t * dc
        assert np.array_equal(a[sr:sr + t, sc:sc + t], b[sr:sr + t, sc:sc + t]), 'source panel changed'
        if p['count'] != k:
            continue
        rr, cc = r + 2 * t * dr, c + 2 * t * dc
        for y in range(t):
            for x in range(t):
                value = int(a[sr + y, sc + x])
                value = ink if value == bg else value
                assert b[rr + t - 1 - y, cc + t - 1 - x] == value, 'wrong sealed half-turn echo'
        assigned[rr:rr + t, cc:cc + t] = True
    assert np.array_equal(a[~assigned], b[~assigned]), 'inactive slot or unused background changed'
    assert not np.array_equal(a, b), 'identity pair'
    assert set(map(int, np.unique(b))) <= set(map(int, np.unique(a))), 'invented color'
    return info


def _training_coverage(infos):
    # These are observable evidence requirements, not literal-color schedules.
    witnesses = set()
    area = equal_area = connectivity = nonpositional = repeated_signal = False
    for info in infos:
        k = info['k']
        active = [p for p in info['panels'].values() if p['count'] == k]
        inactive = [p for p in info['panels'].values() if p['count'] != k]
        area |= any(p['count'] < q['count'] and p['holes'].sum() > q['holes'].sum()
                    for p in active for q in inactive)
        equal_area |= any(p['holes'].sum() == q['holes'].sum() for p in active for q in inactive)
        nonpositional |= any(not np.array_equal(info['dots'], p['holes']) for p in active)
        repeated_signal |= any(np.count_nonzero(p['tile'] == p['sample']) > 1 for p in active)
        for d, p in info['panels'].items():
            if p['count'] != k:
                continue
            connectivity |= len(_components(p['holes'], True)) < p['count']
            if (int(np.count_nonzero(p['tile'] == p['sample'])) == 1
                    and len({v.tobytes() for v in _d4(p['tile'])}) == 8):
                witnesses.add(d)
    assert area, 'training fails to distinguish cavity count from area order'
    assert equal_area, 'training lacks equal-area panels with different activation'
    assert connectivity, 'training fails to distinguish four/eight connectivity'
    assert nonpositional, 'training permits geometric clue matching'
    assert repeated_signal, 'training fails to expose repeated inward signals'
    assert witnesses == set(DIRECTIONS), 'training leaves sample/rotation directions undemonstrated'
    assert any(len({p['sample'] for p in i['panels'].values() if p['count'] == i['k']}) > 1
               for i in infos), 'directional signals need a multicolor demonstration'
    assert any(sum(p['count'] == i['k'] for p in i['panels'].values()) > 1 for i in infos), 'ties must activate together'
    assert any(any(p['count'] != i['k'] for p in i['panels'].values()) for i in infos), 'gating is undemonstrated'
    assert any(len(i['panels']) < 8 for i in infos), 'empty source slots are undemonstrated'


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys'
    assert all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists'
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(np.asarray(pair['input']))
            assert key not in seen, 'duplicate input'
            seen.add(key)
            infos[split].append(info)
    _training_coverage(infos['train'])
    train, test = infos['train'], infos['test']
    scale = all(i['t'] == 5 for i in train) and all(i['t'] == 3 for i in test)
    sparse = all(len(i['panels']) >= 5 for i in train) and all(len(i['panels']) <= 4 for i in test)
    new_count = (all(i['k'] in (1, 2) for i in train)
                 and all(i['k'] in (0, 3, 4, 5) for i in test))
    assert scale or sparse or new_count, 'the documented test shift is absent'


def _fingerprint(puzzle):
    return tuple(_key(np.asarray(p[k])) for s in ('train', 'test')
                 for p in puzzle[s] for k in ('input', 'output'))


def _alternative(a, mode):
    """Focused competing rules used by self_test, never by solve."""
    info = _analyze(a)
    altered = dict(info, panels={d: dict(p) for d, p in info['panels'].items()})
    for p in altered['panels'].values():
        if mode == 'all_panels':
            p['count'] = info['k']
        elif mode == 'area':
            p['count'] = int(p['holes'].sum())
        elif mode == 'eight_connected':
            p['count'] = len(_components(p['holes'], True))
        elif mode == 'sample_center':
            p['sample'] = int(p['tile'][info['t'] // 2, info['t'] // 2])
    out = _render(a, altered)
    if mode in ('identity_copy', 'radial_reflection'):
        t, r, c = (info[x] for x in ('t', 'hr', 'hc'))
        for (dr, dc), p in info['panels'].items():
            if p['count'] != info['k']:
                continue
            tile = p['tile']
            if mode == 'radial_reflection':
                tile = tile[::-1, :] if dc == 0 else tile[:, ::-1] if dr == 0 else tile[::-1, ::-1]
            rr, cc = r + 2 * t * dr, c + 2 * t * dc
            out[rr:rr + t, cc:cc + t] = np.where(tile == info['background'], info['ink'], tile)
    if mode in ('unsealed', 'body_fill'):
        t, r, c = (info[x] for x in ('t', 'hr', 'hc'))
        for (dr, dc), p in info['panels'].items():
            if p['count'] == info['k']:
                rr, cc = r + 2 * t * dr, c + 2 * t * dc
                fill = info['background'] if mode == 'unsealed' else p['body']
                out[rr:rr + t, cc:cc + t][p['holes'][::-1, ::-1]] = fill
    return out


def self_test() -> dict:
    """Bounded 42-seed structural/semantic checks; returns measured variation."""
    stats = {name: set() for name in ('shapes', 'sizes', 'occupancy', 'active_patterns',
                                     'targets', 'backgrounds', 'cavity_masks', 'structures')}
    fingerprints = set()
    partial = repeated = sealed = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                info = _check_pair(pair)
                assert np.array_equal(solve(a.copy()), b)
                assert np.array_equal(solve(a.tolist()), b)
                active = tuple(d for d in DIRECTIONS if d in info['panels']
                               and info['panels'][d]['count'] == info['k'])
                for name, value in (('shapes', a.shape), ('sizes', info['t']),
                                    ('occupancy', len(info['panels'])), ('active_patterns', active),
                                    ('targets', info['k']), ('backgrounds', info['background'])):
                    stats[name].add(value)
                labels = {}
                stats['structures'].add((a.shape, tuple(labels.setdefault(int(v), len(labels)) for v in a.flat)))
                partial += len(info['panels']) < 8
                for d, p in info['panels'].items():
                    stats['cavity_masks'].add((info['t'], p['holes'].tobytes()))
                    repeated += int(np.count_nonzero(p['tile'] == p['sample'])) > 1
                    sealed += d in active and p['holes'].any()
    assert len(fingerprints) == 42
    assert stats['sizes'] == {3, 5}
    assert stats['targets'] == set(range(6))
    assert stats['backgrounds'] == set(range(10))
    assert stats['occupancy'] == set(range(2, 9))
    assert len(stats['shapes']) >= 30 and len(stats['active_patterns']) >= 35
    assert len(stats['cavity_masks']) >= 65 and len(stats['structures']) >= 100
    assert partial >= 42 and repeated >= 25 and sealed >= 80
    assert _fingerprint(generate_puzzle(99, 2, 1)) == _fingerprint(generate_puzzle(99, 2, 1))
    # Tight footprints and both extreme rectangular axes, translated to either
    # edge, exercise geometry beyond incidental random-seed coverage.
    boundary_rng = np.random.default_rng(4321)
    for t in (3, 5):
        pair = _pair(boundary_rng, t, 1, set(DIRECTIONS), {(-1, 0), (0, 1)}, 0)
        i = _analyze(pair['input'])
        r, c = i['hr'] - 2 * t, i['hc'] - 2 * t
        tight = {k: v[r:r + 5 * t, c:c + 5 * t].copy() for k, v in pair.items()}
        _check_pair(tight)
        assert np.array_equal(solve(tight['input']), tight['output'])
        for axis in (0, 1):
            for edge in (0, 1):
                pads = [(0, 0), (0, 0)]
                pads[axis] = (30 - 5 * t, 0) if edge else (0, 30 - 5 * t)
                rectangle = {k: np.pad(v, pads, constant_values=0) for k, v in tight.items()}
                _check_pair(rectangle)
                assert np.array_equal(solve(rectangle['input']), rectangle['output'])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + nt * 10 + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False),
                   (2.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # Visible-rule ablations must change a demonstrated answer.
    sample = generate_puzzle(0, 2, 1)
    ablations = {}
    for mode in ('all_panels', 'area', 'eight_connected', 'identity_copy', 'radial_reflection',
                 'sample_center', 'unsealed', 'body_fill'):
        assert any(not np.array_equal(_alternative(p['input'], mode), p['output'])
                   for p in sample['train']), 'unrefuted shortcut: ' + mode
        ablations[mode] = [int(np.count_nonzero(_alternative(p['input'], mode) != p['output']))
                           for p in sample['train']]
    lookup = np.random.default_rng(45).permutation(10)
    for p in sample['train']:
        before = p['input'].copy()
        assert np.array_equal(solve(lookup[before]), lookup[p['output']])
        solve(p['input'])
        assert np.array_equal(before, p['input']), 'solve mutated its input'
    # Check rejection of an echo mutation and of missing training evidence.
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]] for s in sample}
    q = broken['test'][0]
    qi = _analyze(q['input'])
    d = next(d for d, p in qi['panels'].items() if p['count'] == qi['k'])
    y, x = qi['hr'] + 2 * qi['t'] * d[0], qi['hc'] + 2 * qi['t'] * d[1]
    q['output'][y, x] = (int(q['output'][y, x]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt echo accepted')
    try:
        _training_coverage([_analyze(sample['train'][0]['input'])] * 2)
    except AssertionError:
        pass
    else:
        raise AssertionError('complementary direction evidence not required')
    return dict(seeds=42, pairs=210, distinct_puzzles=len(fingerprints),
                variation={k: len(v) for k, v in stats.items()}, partial_rings=partial,
                repeated_signals=repeated, echoes_with_sealed_cavities=int(sealed),
                ablation_changed_cells=ablations)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
