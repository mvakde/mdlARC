"""Mirror-orbit transfer: reflected connected islands are unary credits.

Only standard-library modules and NumPy are used. All role colors, axes and
receiver geometry are read from the input; generation metadata never solves it.
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _neighbors(points):
    return {(r + dr, c + dc) for r, c in points
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))} - set(points)


def _components(points):
    todo, result = set(points), []
    while todo:
        start = todo.pop()
        part, frontier = {start}, [start]
        while frontier:
            r, c = frontier.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    frontier.append(q)
        result.append(frozenset(part))
    return result


def _cells(a, color):
    return set(map(tuple, np.argwhere(a == color).tolist()))


def _bbox(points):
    rows, cols = zip(*points)
    return min(rows), max(rows), min(cols), max(cols)


def _axis_core(points, width):
    """Unique strongest vertical reflection; axis is stored doubled."""
    candidates = [(s, {p for p in points if (p[0], s - p[1]) in points})
                  for s in range(2 * width - 1)]
    best = max(len(core) for _, core in candidates)
    winners = [(s, core) for s, core in candidates if len(core) == best]
    if len(winners) != 1:
        raise ValueError("reflection support has a tie")
    return winners[0]


def _orbits(core, axis):
    parts = set(_components(core))
    self_count = pair_count = 0
    while parts:
        part = parts.pop()
        reflected = frozenset((r, axis - c) for r, c in part)
        if reflected == part:
            self_count += 1
        else:
            if reflected not in parts:
                raise ValueError("core component has no reflected component")
            parts.remove(reflected)
            pair_count += 1
    return self_count + pair_count, self_count, pair_count


def _parse_marker(a, bg, marker):
    parts = _components(_cells(a, marker))
    bars = [p for p in parts if len(p) >= 2]
    if len(bars) != 1:
        raise ValueError("need exactly one receiver bar")
    bar = bars[0]
    y0, y1, x0, x1 = _bbox(bar)
    if y0 != y1 or len(bar) != x1 - x0 + 1:
        raise ValueError("receiver marker must be a solid horizontal bar")
    dots = [next(iter(p)) for p in parts if len(p) == 1]
    if not dots:
        raise ValueError("need source dots")
    sources = {}
    for r, c in dots:
        nearby = {int(a[rr, cc]) for rr, cc in _neighbors({(r, c)})
                  if 0 <= rr < a.shape[0] and 0 <= cc < a.shape[1]
                  and int(a[rr, cc]) not in (bg, marker)}
        if len(nearby) != 1:
            raise ValueError("dot must touch exactly one payload color")
        color = nearby.pop()
        if color in sources:
            raise ValueError("only one dot per donor")
        sources[color] = (r, c)
    stripe = a[:, x0:x1 + 1]
    payloads = set(map(int, np.unique(stripe))) - {bg, marker}
    if len(payloads) != 1:
        raise ValueError("receiver corridor must have one payload color")
    receiver = payloads.pop()
    if receiver in sources:
        raise ValueError("receiver cannot be a donor")
    tokens = _components(_cells(a, receiver))
    if len(tokens) < 2:
        raise ValueError("receiver must visibly repeat")
    tokens.sort(key=lambda p: _bbox(p)[0])
    boxes = [_bbox(p) for p in tokens]
    tops = [b[0] for b in boxes]
    h, w = boxes[0][1] - boxes[0][0] + 1, x1 - x0 + 1
    mask = frozenset((r - tops[0], c - x0) for r, c in tokens[0])
    pitch = tops[1] - tops[0]
    if pitch <= h or any(tops[i] - tops[i - 1] != pitch for i in range(1, len(tops))):
        raise ValueError("receiver needs constant positive spacing")
    for p, b in zip(tokens, boxes):
        if b[2:] != (x0, x1) or frozenset((r - b[0], c - x0) for r, c in p) != mask:
            raise ValueError("receiver token shapes must agree")
    if y0 <= boxes[-1][1] + 1:
        raise ValueError("receiver bar needs a blank gap below tokens")
    expected_stripe = np.full_like(stripe, bg)
    for p in tokens:
        for r, c in p:
            expected_stripe[r, c - x0] = receiver
    expected_stripe[y0] = marker
    if not np.array_equal(expected_stripe, stripe):
        raise ValueError("receiver corridor contains unrelated pixels")
    donors = []
    for color, dot in sources.items():
        points = _cells(a, color)
        axis, core = _axis_core(points, a.shape[1])
        credit, self_count, pair_count = _orbits(core, axis)
        donors.append(dict(color=color, dot=dot, points=points, core=core,
                           axis=axis, credit=credit, self=self_count, paired=pair_count))
    total = sum(d['credit'] for d in donors)
    if tops[0] - total * pitch < 0:
        raise ValueError("receiver extension would be clipped")
    return dict(bg=bg, marker=marker, bar=bar, donors=donors, total=total,
                receiver=receiver, tokens=tokens, mask=mask, x=x0,
                top=tops[0], pitch=pitch, height=h, width=w)


def _parse(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= a.size / 2:
        raise ValueError("background must be a strict majority")
    choices = []
    for marker in set(map(int, np.unique(a))) - {bg}:
        try:
            choices.append(_parse_marker(a, bg, marker))
        except ValueError:
            pass
    if len(choices) != 1:
        raise ValueError("control geometry must uniquely identify roles")
    return choices[0]


def _render_solution(a, parsed, total=None):
    b = a.copy()
    bg, marker = parsed['bg'], parsed['marker']
    for p in parsed['bar']:
        b[p] = bg
    for donor in parsed['donors']:
        b[donor['dot']] = bg
        for p in donor['points']:
            b[p] = marker if p in donor['core'] else bg
    count = parsed['total'] if total is None else total
    for k in range(1, count + 1):
        y = parsed['top'] - k * parsed['pitch']
        for r, c in parsed['mask']:
            if 0 <= y + r < b.shape[0]:
                b[y + r, parsed['x'] + c] = parsed['receiver']
    return b


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve any well-formed family input without training, seed or hidden state."""
    a = _grid(input_grid)
    return _render_solution(a, _parse(a))


def _connected_mask(rng, h, w):
    # A spine and varying row segments: connected, full span, sometimes hollow.
    cells = {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}
    for r in range(h - 1):
        cells |= {(r, c) for c in range(int(rng.integers(1, w + 1)))}
    if rng.integers(2):
        cells = {(r, w - 1 - c) for r, c in cells}
    if rng.integers(2):
        cells = {(h - 1 - r, c) for r, c in cells}
    return cells


def _self_mask(rng, h, w):
    if h >= 3 and w >= 3 and rng.random() < .35:
        return {(r, c) for r in range(h) for c in range(w)
                if r in (0, h - 1) or c in (0, w - 1)}
    cells = {(h // 2, c) for c in range(w)}
    for r in range(h):
        extent = int(rng.integers((w + 1) // 2))
        cells |= {(r, c) for c in range(extent, w - extent)}
    return cells


def _sample_core(rng, credits, kind='random'):
    """Construct component orbits; known credits are independent ground truth."""
    width = int(rng.integers(5, 10))
    core, bridge = set(), set()
    if kind == 'bridge':
        # Two pairs, joined only on the left by additive asymmetric noise.
        h, w = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        for y in (0, h + 1):
            left = {(y + r, c) for r in range(h) for c in range(w)}
            core |= left | {(r, width - 1 - c) for r, c in left}
        bridge = {(h, 0)}
        if credits == 3:
            sw = 3 if width % 2 else 2
            left = (width - sw) // 2
            core |= {(2 * h + 3, left + c) for c in range(sw)}
    else:
        for index in range(credits):
            paired = index == 0 if kind == 'mixed' else rng.random() < .55
            placed = False
            for _ in range(90):
                h = int(rng.integers(1, 4))
                y = int(rng.integers(0, 10 - h + 1))
                if paired:
                    w = int(rng.integers(1, min(3, (width - 1) // 2) + 1))
                    x = int(rng.integers(0, (width - 1) // 2 - w + 1))
                    left = {(y + r, x + c) for r, c in _connected_mask(rng, h, w)}
                    island = left | {(r, width - 1 - c) for r, c in left}
                else:
                    choices = [w for w in range(1, width + 1) if w % 2 == width % 2]
                    w = int(rng.choice(choices))
                    x = (width - w) // 2
                    island = {(y + r, x + c) for r, c in _self_mask(rng, h, w)}
                if not island & (core | _neighbors(core)):
                    core |= island
                    placed = True
                    break
            if not placed:
                return None
    r0, r1, c0, c1 = _bbox(core)
    cropped = {(r - r0, c - c0) for r, c in core}
    axis = width - 1 - 2 * c0
    bridge = {(r - r0, c - c0) for r, c in bridge}
    if _orbits(cropped, axis)[0] != credits:
        return None
    return cropped, axis, bridge


def _noise(rng, core, axis, shape, required, amount):
    points = core | required
    try:
        s, retained = _axis_core(points, shape[1])
        if s != axis or retained != core:
            return None
    except ValueError:
        return None
    candidates = [(r, c) for r in range(shape[0]) for c in range(shape[1])
                  if (r, c) not in points and (r, axis - c) not in points]
    rng.shuffle(candidates)
    for p in candidates:
        if len(points - core) >= amount:
            break
        if (p[0], axis - p[1]) in points:
            continue
        try:
            s, retained = _axis_core(points | {p}, shape[1])
        except ValueError:
            continue
        if s == axis and retained == core:
            points.add(p)
    return points - core


def _token(rng, max_height):
    h = int(rng.integers(1, max_height + 1))
    w = int(rng.integers(2, 6))
    return h, w, _connected_mask(rng, h, w)


def _build_pair(rng, source_credits, initial, *, bridge=False, mixed=False, ignored=0):
    total = sum(source_credits)
    count = len(source_credits) + ignored
    for attempt in range(180):
        max_h = min(3, 27 // (initial + total) - 1)
        max_h = max(1, max_h)
        h, w, mask = _token(rng, max_h)
        possible = [p for p in range(h + 1, h + 3)
                    if h + (initial + total - 1) * p + 2 <= 30]
        pitch = int(rng.choice(possible))
        minimum = h + (initial + total - 1) * pitch + 2
        height = int(rng.integers(max(16, minimum), 31))
        width = int(rng.integers(23 if count >= 4 else 17, 31))
        x = int(rng.integers(0, width - w + 1))
        gap = int(rng.integers(1, min(3, height - minimum + 1) + 1))
        bar_y = height - 1 - int(rng.integers(0, min(2, height - minimum - gap + 1) + 1))
        bottom = bar_y - gap - h
        if bottom - (initial + total - 1) * pitch < 0:
            continue
        palette = rng.choice(10, count + 3, replace=False).tolist()
        bg, marker, receiver, *colors = map(int, palette)
        a = np.full((height, width), bg, dtype=np.int8)
        b = a.copy()
        for k in range(initial + total):
            for rr, cc in mask:
                b[bottom - k * pitch + rr, x + cc] = receiver
                if k < initial:
                    a[bottom - k * pitch + rr, x + cc] = receiver
        a[bar_y, x:x + w] = marker
        reserved = {(r, c) for r in range(height) for c in range(x, x + w)}
        occupied = set(reserved)
        records = []
        specs = [(k, True, 'bridge' if bridge and i == 0 else
                  'mixed' if mixed and i == 0 else 'random')
                 for i, k in enumerate(source_credits)]
        specs += [(int(rng.integers(1, 4)), False, 'random') for _ in range(ignored)]
        ok = True
        for color, (credits, active, kind) in zip(colors, specs):
            local = _sample_core(rng, credits, kind)
            if local is None:
                ok = False
                break
            core, axis, req = local
            ch, cw = _bbox(core)[1] + 1, _bbox(core)[3] + 1
            # Expand a local halo on all sides, permitting external noise.
            core = {(r + 1, c + 1) for r, c in core}
            req = {(r + 1, c + 1) for r, c in req}
            axis += 2
            shape = (ch + 2, cw + 2)
            amount = len(req) if kind == 'bridge' else int(rng.integers(1, 5))
            additions = _noise(rng, core, axis, shape, req, amount)
            if additions is None:
                ok = False
                break
            points = core | additions
            placed = False
            for _ in range(100):
                y = int(rng.integers(0, height - shape[0] + 1))
                left = int(rng.integers(0, width - shape[1] + 1))
                shifted = {(r + y, c + left) for r, c in points}
                if shifted & (occupied | _neighbors(occupied)):
                    continue
                shifted_core = {(r + y, c + left) for r, c in core}
                dot = None
                if active:
                    candidates = sorted(p for p in _neighbors(shifted_core)
                                        if 0 <= p[0] < height and 0 <= p[1] < width
                                        and p not in shifted and p not in occupied
                                        and not _neighbors({p}) & occupied)
                    if not candidates:
                        continue
                    dot = candidates[int(rng.integers(len(candidates)))]
                occupied |= shifted | ({dot} if dot else set())
                for p in shifted:
                    a[p] = color
                    b[p] = marker if active and p in shifted_core else bg if active else color
                if dot:
                    a[dot] = marker
                records.append((color, shifted_core, axis + 2 * left, active, credits))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok or np.count_nonzero(a == bg) <= a.size / 2:
            continue
        try:
            parsed = _parse(a)
        except ValueError:
            continue
        if parsed['marker'] != marker or parsed['total'] != total:
            continue
        if not np.array_equal(_render_solution(a, parsed), b):
            continue
        return {'input': a, 'output': b}
    raise RuntimeError("bounded scene packing exhausted")


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError("illegal example counts")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    # Minimal demonstrations: one noisy bridge and one mixed-topology,
    # unequal multi-source sum. Arithmetic operands and presentation order vary.
    options = [(a, b, d1, d2) for a in (2, 3) for b in (4, 5)
               for d1, d2 in ((2, 3), (3, 2))
               if a != b and a + d1 != b + d2 and a * d2 != b * d1]
    a, b, d1, d2 = options[int(rng.integers(len(options)))]
    anchors = [([a], d1, True, False, 1), ([3, b - 3], d2, False, True, int(rng.integers(0, 2)))]
    rng.shuffle(anchors)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, length in (('train', num_train), ('test', num_test)):
        for index in range(length):
            if split == 'train' and index < 2:
                credits, initial, bridge, mixed, ignored = anchors[index]
            else:
                n = int(rng.integers(1, 3)) if split == 'train' else int(rng.integers(3, 5))
                # Total <= 6 bounds receiver extension while preserving unequal sources.
                total = int(rng.integers(n, min(6, 3 * n) + 1))
                credits = [1] * n
                for _ in range(total - n):
                    candidates = [i for i, k in enumerate(credits) if k < 3]
                    credits[int(rng.choice(candidates))] += 1
                rng.shuffle(credits)
                initial = int(rng.integers(2, 5))
                bridge, mixed = False, False
                ignored = int(rng.integers(0, 3))
            for _ in range(12):
                pair = _build_pair(rng, credits, initial, bridge=bridge, mixed=mixed, ignored=ignored)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness retries exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure(a):
    p = _parse(a)
    donors = p['donors']
    return dict(shape=a.shape, bg=p['bg'], marker=p['marker'],
                sources=len(donors), credits=tuple(d['credit'] for d in donors),
                total=p['total'], initial=len(p['tokens']), final=len(p['tokens']) + p['total'],
                self=sum(d['self'] for d in donors), paired=sum(d['paired'] for d in donors),
                noise=sum(len(d['points'] - d['core']) for d in donors),
                bridge=any(len(_components(d['points'])) < len(_components(d['core'])) for d in donors),
                parities=tuple(d['axis'] % 2 for d in donors),
                ignored=len(set(map(int, np.unique(a))) - {p['bg'], p['marker'], p['receiver']}
                            - {d['color'] for d in donors}),
                token=(p['height'], p['width'], tuple(sorted(p['mask']))), pitch=p['pitch'])


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            parsed = _parse(a)
            assert np.array_equal(_render_solution(a, parsed), b), "wrong transfer output"
            # Check deletion, recoloring, full copy geometry and untouched objects.
            marker, bg = parsed['marker'], parsed['bg']
            assert all(b[p] == bg for p in parsed['bar'])
            for d in parsed['donors']:
                assert b[d['dot']] == bg
                assert all(b[p] == marker for p in d['core'])
                assert all(b[p] == bg for p in d['points'] - d['core'])
                assert _orbits(d['core'], d['axis'])[0] == d['credit']
            assert np.array_equal(a == parsed['receiver'], b == parsed['receiver']) is False
            before = len(parsed['tokens'])
            after = len(_components(_cells(b, parsed['receiver'])))
            assert after == before + parsed['total']
            untouched = set(map(int, np.unique(a))) - {bg, marker, parsed['receiver']} - {d['color'] for d in parsed['donors']}
            for color in untouched:
                assert _cells(a, color) == _cells(b, color)
            if split == 'train':
                training.append(_measure(a))
    assert any(m['paired'] and m['self'] for m in training), "teach both orbit kinds"
    assert any(d['self'] >= 2 and d['paired'] for p in puzzle['train']
               for d in _parse(_grid(p['input']))['donors']), "distinguish halving components"
    assert any(m['bridge'] for m in training), "teach topological cleanup"
    assert any(m['ignored'] for m in training), "teach selective processing"
    assert any(m['sources'] > 1 and len(set(m['credits'])) > 1 for m in training)
    assert len({m['initial'] for m in training}) > 1
    assert len({m['total'] for m in training}) > 1
    assert len({(m['total'], m['initial']) for m in training}) > 1
    rivals = (lambda s, d: 2 * d, lambda s, d: 2 * s,
              lambda s, d: s * d, lambda s, d: max(s, d), lambda s, d: s,
              lambda s, d: d + 1)
    for rival in rivals:
        assert any(rival(m['total'], m['initial']) != m['final'] for m in training)
    assert any(m['total'] * training[0]['initial'] != training[0]['total'] * m['initial'] for m in training)
    # Omitting A gives a different number of receiver tokens, even if cleanup
    # and recoloring are granted for free to the alternative rule.
    assert any(sum(len(_components(d['core'])) for d in _parse(_grid(p['input']))['donors'])
               != _parse(_grid(p['input']))['total'] for p in puzzle['train'])
    # Omitting B leaves a missing receiver extension in every pair.
    assert all(_parse(_grid(p['input']))['total'] > 0 for p in puzzle['train'] + puzzle['test'])


def self_test():
    """Bounded feature coverage, counts, repeatability and mutation checks."""
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(_grid(p['input'])) for p in puzzle['train'] + puzzle['test']))
        for p in puzzle['train'] + puzzle['test']:
            assert np.array_equal(solve(p['input']), p['output'])
            records.append(_measure(_grid(p['input'])))
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], again[s]) for k in p)
    assert len(signatures) == 42
    assert len({m['shape'] for m in records}) >= 35
    assert {m['bg'] for m in records} == set(range(10))
    assert {m['marker'] for m in records} == set(range(10))
    assert {1, 2, 3, 4} <= {m['sources'] for m in records}
    assert {0, 1, 2} <= {m['ignored'] for m in records}
    assert {0, 1} == {p for m in records for p in m['parities']}
    assert len({m['token'] for m in records}) >= 20
    assert len({m['pitch'] for m in records}) >= 3
    assert any(m['self'] >= 2 and m['paired'] >= 2 for m in records)
    assert any(m['final'] >= 9 for m in records)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(73, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    good = generate_puzzle(31, 2, 1)
    for defect in ('retained', 'noise', 'bar', 'sum', 'ignored'):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in good[s]] for s in good}
        changed = False
        for pair in bad['train']:
            p = _parse(pair['input'])
            if defect == 'retained':
                pair['output'][next(iter(p['donors'][0]['core']))] = p['bg']
            elif defect == 'noise':
                d = next((d for d in p['donors'] if d['points'] - d['core']), None)
                if d is None:
                    continue
                pair['output'][next(iter(d['points'] - d['core']))] = p['marker']
            elif defect == 'bar':
                pair['output'][next(iter(p['bar']))] = p['marker']
            elif defect == 'sum':
                rr, cc = next(iter(p['mask']))
                pair['output'][p['top'] - p['pitch'] + rr, p['x'] + cc] = p['bg']
            else:
                colors = set(map(int, np.unique(pair['input']))) - {p['bg'], p['marker'], p['receiver']} - {d['color'] for d in p['donors']}
                if not colors:
                    continue
                pair['output'][next(iter(_cells(pair['input'], min(colors))))] = p['marker']
            changed = True
            break
        assert changed
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("validator accepted " + defect)


if __name__ == '__main__':
    self_test()
