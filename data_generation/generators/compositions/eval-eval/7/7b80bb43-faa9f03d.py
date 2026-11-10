"""Reroute and energize: geometric repair controls first contact and reachability.

Only the standard library and NumPy are used. No parent module is imported.
The singleton annotation is a direction marker; every other color role is
derived from occupancy and the marker's geometrical relationship to a path.
"""
from __future__ import annotations

from collections import deque
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _add(p, d, k=1):
    return p[0] + k * d[0], p[1] + k * d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _has(mask, p):
    return _inside(p, mask.shape) and bool(mask[p])


def _neighbors(p, mask):
    return [_add(p, d) for d in DIRS if _has(mask, _add(p, d))]


def _cells(mask):
    return {tuple(map(int, p)) for p in np.argwhere(mask)}


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    d = (int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))
    return [_add(a, d, k) for k in range(max(abs(b[0]-a[0]), abs(b[1]-a[1])) + 1)]


def _polyline(points):
    result = [points[0]]
    for a, b in zip(points, points[1:]):
        result.extend(_line(a, b)[1:])
    return result


def _mask(shape, cells):
    result = np.zeros(shape, dtype=bool)
    for p in cells:
        result[p] = True
    return result


def _components(mask):
    unseen = _cells(mask)
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component, pending = {start}, [start]
        while pending:
            p = pending.pop()
            for d in DIRS:
                q = _add(p, d)
                if q in unseen:
                    unseen.remove(q)
                    component.add(q)
                    pending.append(q)
        result.append(component)
    return result


def _thin(mask):
    return not np.any(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:])


def _halo(mask):
    padded = np.pad(mask, 1)
    return np.logical_or.reduce([padded[r:r+mask.shape[0], c:c+mask.shape[1]]
                                 for r in range(3) for c in range(3)])


def _gap_evidence(mask):
    """Simultaneous, visible three-cell gaps with isolated two-cell deviations."""
    result = []
    for p in sorted(_cells(mask)):
        for d in ((1, 0), (0, 1)):
            other = _add(p, d, 4)
            gap = tuple(_add(p, d, k) for k in (1, 2, 3))
            if not _has(mask, other) or any(_has(mask, q) for q in gap):
                continue
            if not (_has(mask, _add(p, d, -1)) or _has(mask, _add(other, d))):
                continue
            matches = []
            for anchor, toward in ((p, 1), (other, -1)):
                for side in (-1, 1):
                    diagonal = (toward*d[0] + side*d[1], toward*d[1] + side*d[0])
                    spur = tuple(_add(anchor, diagonal, k) for k in (1, 2))
                    if all(_has(mask, q) and not _neighbors(q, mask) for q in spur):
                        if not _has(mask, _add(anchor, diagonal, 3)):
                            matches.append(spur)
            if matches:
                assert len(matches) == 1, 'ambiguous gap evidence'
                result.append((gap, matches[0], (p, other)))
    assert len({p for _, spur, _ in result for p in spur}) == 2*len(result), 'shared repair spur'
    return result


def _remove_diagonals(mask, protected=()):
    # Only cells without any same-color orthogonal neighbor are noise candidates.
    # Genuine isolated gap endpoints are protected by their repair brackets.
    weak = {p for p in _cells(mask) if not _neighbors(p, mask)} - set(protected)
    removed = set()
    for p in weak:
        if any(_add(p, d) in weak for d in ((1, 1), (1, -1), (-1, 1), (-1, -1))):
            removed.add(p)
    return mask & ~_mask(mask.shape, removed), removed


def _clean(mask):
    repairs = _gap_evidence(mask)
    protected = {p for _, _, ends in repairs for p in ends}
    cleaned, removed = _remove_diagonals(mask, protected)
    added = set()
    for gap, spur, _ in repairs:
        assert set(spur) <= removed
        added.update(gap)
    cleaned |= _mask(mask.shape, added)
    return cleaned, repairs, added, removed


def _walk_old(mask, pivot, direction):
    previous, current = pivot, _add(pivot, direction, -1)
    walked = []
    while _has(mask, current) and current not in walked:
        walked.append(current)
        following = [q for q in _neighbors(current, mask) if q != previous]
        if not following:
            if current[0] in (0, mask.shape[0]-1) or current[1] in (0, mask.shape[1]-1):
                return walked
            return None
        if len(following) != 1:
            return None
        previous, current = current, following[0]
    return None


def _roles_and_paths(inp):
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 4, 'four distinct semantic colors are required'
    bg = int(values[counts.argmax()])
    assert int(counts.max())*2 > inp.size, 'background must be a strict majority'
    singleton = [int(c) for c, n in zip(values, counts) if n == 1 and c != bg]
    assert len(singleton) == 1, 'one singleton direction annotation is required'
    annotation = singleton[0]
    marker = tuple(map(int, np.argwhere(inp == annotation)[0]))
    colors = sorted(set(map(int, values)) - {bg, annotation})
    records = {c: _clean(inp == c) for c in colors}
    candidates = []
    for color in colors:
        path = records[color][0]
        for d in DIRS:
            pivot = _add(marker, d, -1)
            if not _has(path, pivot) or not _has(path, _add(pivot, d, -1)):
                continue
            perp = (-d[1], d[0])
            if not (_has(path, _add(pivot, perp)) or _has(path, _add(pivot, perp, -1))):
                continue
            old = _walk_old(path, pivot, d)
            if old:
                candidates.append((color, pivot, d, old))
    assert len(candidates) == 1, 'marker must identify one terminal launcher branch'
    signal, pivot, direction, old = candidates[0]
    receiver = next(c for c in colors if c != signal)
    return bg, annotation, signal, receiver, records, pivot, direction, old


def _flow(source, receiver, pivot, direction, old):
    """Delete the old arm; first contact captures one entire receiver component."""
    kept = source.copy()
    for p in old:
        kept[p] = False
    ray, contact = [], None
    q = _add(pivot, direction)
    while _inside(q, source.shape):
        ray.append(q)
        if receiver[q]:
            contact = q
            break
        q = _add(q, direction)
    active = set()
    if contact is not None:
        active = {contact}
        queue = deque([contact])
        while queue:
            p = queue.popleft()
            for q in _neighbors(p, receiver):
                if q not in active:
                    active.add(q)
                    queue.append(q)
    return kept, ray, active, contact


def _render(shape, bg, signal, receiver_color, source, receiver, pivot, direction, old):
    kept, ray, active, contact = _flow(source, receiver, pivot, direction, old)
    out = np.full(shape, bg, dtype=np.int8)
    out[receiver] = receiver_color
    out[kept] = signal
    for p in set(ray) | active:
        out[p] = signal
    return out, (kept, ray, active, contact)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, repair, old route, contact and connected set from input."""
    inp = np.asarray(input_grid)
    assert inp.ndim == 2 and np.issubdtype(inp.dtype, np.integer)
    assert all(1 <= n <= 30 for n in inp.shape) and np.all((0 <= inp) & (inp <= 9))
    bg, _, signal, receiver_color, records, pivot, direction, old = _roles_and_paths(inp)
    out, _ = _render(inp.shape, bg, signal, receiver_color, records[signal][0],
                     records[receiver_color][0], pivot, direction, old)
    return out


def _draw(mask, points):
    for p in _polyline(points):
        mask[p] = True


def _source_scene(rng, shape, row, col, bends):
    h, _ = shape
    side = int(rng.choice((-1, 1)))
    # The retained tail varies independently from the old boundary route.
    limit = row if side < 0 else h-1-row
    length = int(rng.integers(3, min(8, limit)+1))
    tail = [(row, col), (row+side*length, col)]
    if bends == 0:
        points = [(row, col), (row, 0)]
    else:
        x = int(rng.integers(4 if bends == 3 else 1, col-2))
        edge = int(rng.choice((0, h-1)))
        if bends == 1:
            points = [(row, col), (row, x), (edge, x)]
        else:
            ys = [y for y in range(2, h-2) if abs(y-row) >= 3]
            y = int(rng.choice(ys))
            if bends == 2:
                points = [(row, col), (row, x), (y, x), (y, 0)]
            else:
                x2 = int(rng.integers(0, x-2))
                points = [(row, col), (row, x), (y, x), (y, x2), (edge, x2)]
    old = _polyline(points)[1:]
    source = _mask(shape, old + _polyline(tail))
    if rng.random() < .35:
        other_limit = h-1-row if side < 0 else row
        other_length = int(rng.integers(2, min(6, other_limit)+1))
        _draw(source, [(row, col), (row-side*other_length, col)])
    assert _walk_old(source, (row, col), (0, 1)) == old
    return source, old


def _receiver_scene(rng, shape, row, col, gate, source, hit, witness, degree):
    h, w = shape
    top = int(rng.integers(0, row-3))
    bottom = int(rng.integers(row+4, h))
    if witness:
        bottom = max(bottom, row+10)
    receiver = np.zeros(shape, bool)
    receiver[top:bottom+1, gate] = True
    # The long left arm is a repairable off-axis connection, not an output slot.
    left = col+2
    receiver[top, left:gate+1] = True
    intentional = []
    if not hit:
        intentional.append(tuple((r, gate) for r in range(row-1, row+2)))
    if witness or rng.random() < .55:
        options = [r for r in range(top+3, bottom-2) if abs(r-row) >= 6 and abs(r-top) >= 4]
        if options:
            center = row+6 if witness else int(rng.choice(options))
            intentional.append(tuple((r, gate) for r in range(center-1, center+2)))
    for gap in intentional:
        for p in gap:
            receiver[p] = False
    # Grow a rectilinear graph: varying arms, secondary turns, occasional loops.
    levels = [r for r in range(top, bottom+1) if abs(r-row) >= 4
              and all(abs(r-p[0]) >= 2 for gap in intentional for p in gap)]
    rng.shuffle(levels)
    chosen = [top]
    for r in levels:
        if any(abs(r-y) < 3 for y in chosen):
            continue
        end = int(rng.integers(left, w))
        if abs(end-gate) < 3:
            continue
        trial = receiver.copy()
        _draw(trial, [(r, gate), (r, end)])
        if not (trial & _halo(source)).any() and _thin(trial):
            receiver = trial
            chosen.append(r)
            if len(chosen) >= degree+2:
                break
    for _ in range(degree+1):
        endpoints = [p for p in _cells(receiver) if len(_neighbors(p, receiver)) == 1 and abs(p[0]-row) >= 4]
        if not endpoints:
            break
        p = endpoints[int(rng.integers(len(endpoints)))]
        ys = [y for y in range(h) if abs(y-p[0]) >= 3 and (y-row)*(p[0]-row) > 0 and abs(y-row) >= 3]
        if not ys:
            continue
        y = int(rng.choice(ys))
        trial = receiver.copy()
        _draw(trial, [p, (y, p[1])])
        if rng.random() < .35:
            _draw(trial, [(y, p[1]), (y, gate)])
        if (trial & _halo(source)).any() or not _thin(trial):
            continue
        if any(trial[p] for gap in intentional for p in gap):
            continue
        receiver = trial
    # One behind-contact receiver makes stopping and component selection visible.
    if hit:
        behind = int(rng.integers(gate+4, w))
        trial = receiver.copy()
        trial[row-2:row+3, behind] = True
        if not (_halo(receiver)[row-2:row+3, behind]).any():
            receiver = trial
    # Extra cold paths can be anywhere compatible, with independently chosen bends.
    for _ in range(int(rng.integers(degree+2))):
        for _ in range(18):
            a = (int(rng.integers(h)), int(rng.integers(w)))
            b = (a[0], int(rng.integers(w))) if rng.integers(2) else (int(rng.integers(h)), a[1])
            points = [a, b]
            if rng.random() < .5:
                points.append((int(rng.integers(h)), b[1]) if a[0] == b[0] else (b[0], int(rng.integers(w))))
            cells = _polyline(points)
            if len(cells) < 3 or len(set(cells)) != len(cells):
                continue
            fragment = _mask(shape, cells)
            if (fragment & _halo(receiver | source)).any() or not _thin(fragment):
                continue
            if any(fragment[p] for gap in intentional for p in gap):
                continue
            if not hit and fragment[row, col+1:].any():
                continue
            receiver |= fragment
            break
    assert not (source & receiver).any()
    assert receiver[row, gate] if hit else not receiver[row, col+1:].any()
    return receiver, intentional, top


def _corrupt_gap(rng, inp, color, intended, anchor, d):
    gap = tuple(_add(anchor, d, k) for k in (1, 2, 3))
    if not all(_has(intended, _add(anchor, d, k)) for k in range(5)):
        return None
    if any(any(_has(intended, _add(p, t)) for t in ((-d[1], d[0]), (d[1], -d[0]))) for p in gap):
        return None
    options = [(end, side) for end in (0, 4) for side in (-1, 1)]
    rng.shuffle(options)
    bg = int(np.bincount(inp.ravel(), minlength=10).argmax())
    for end, side in options:
        a = _add(anchor, d, end)
        toward = 1 if end == 0 else -1
        diag = (toward*d[0]+side*d[1], toward*d[1]+side*d[0])
        spur = tuple(_add(a, diag, k) for k in (1, 2))
        if not all(_inside(p, inp.shape) and inp[p] == bg for p in spur):
            continue
        trial = inp.copy()
        for p in gap:
            trial[p] = bg
        for p in spur:
            trial[p] = color
        try:
            recovered, _, _, _ = _clean(trial == color)
        except AssertionError:
            continue
        if np.array_equal(recovered, intended):
            return trial
    return None


def _add_repairs(rng, inp, color, intended, wanted):
    candidates = [(p, d) for p in sorted(_cells(intended)) for d in ((1, 0), (0, 1))]
    rng.shuffle(candidates)
    done = 0
    for p, d in candidates:
        if any(inp[_add(p, d, k)] != color for k in range(5) if _inside(_add(p, d, k), inp.shape)):
            continue
        trial = _corrupt_gap(rng, inp, color, intended, p, d)
        if trial is not None:
            inp = trial
            done += 1
            if done >= wanted:
                break
    return inp, done


def _add_clutter(rng, inp, intended_by_color, bg, color, cells):
    if not all(_inside(p, inp.shape) and inp[p] == bg for p in cells):
        return inp
    trial = inp.copy()
    for p in cells:
        trial[p] = color
    try:
        if all(np.array_equal(_clean(trial == c)[0], mask) for c, mask in intended_by_color.items()):
            return trial
    except AssertionError:
        pass
    return inp


def _build_pair(rng, palette, hit, witness, degree, bends, contact_repair):
    bg, signal, receiver_color, annotation = palette
    h = int(rng.integers(24 if witness else 19, 31))
    col = int(rng.integers(7 if bends == 3 else 5, 10))
    gate = col + int(rng.integers(7, 11))
    w = int(rng.integers(max(22, gate+5), 31))
    row = int(rng.integers(5, h-(11 if witness else 5)))
    shape, pivot = (h, w), (row, col)
    source, old = _source_scene(rng, shape, row, col, bends)
    receiver, intentional, top = _receiver_scene(rng, shape, row, col, gate, source, hit, witness, degree)
    inp = np.full(shape, bg, dtype=np.int8)
    inp[source], inp[receiver] = signal, receiver_color
    inp[row, col+1] = annotation
    if hit and contact_repair:
        trial = _corrupt_gap(rng, inp, receiver_color, receiver, (row-2, gate), (1, 0))
        assert trial is not None, 'unable to displace the contact'
        inp = trial
    # Always provide a connection repair away from the contact when it is intact
    # or open. Its location is chosen among all legal strokes, not a fixed hole.
    wanted = int(rng.integers(1, degree+3))
    inp, repairs = _add_repairs(rng, inp, receiver_color, receiver, wanted)
    assert repairs or (hit and contact_repair), 'receiver needs a supported repair'
    if rng.random() < .5:
        inp, _ = _add_repairs(rng, inp, signal, source, int(rng.integers(1, 3)))
    masks = {signal: source, receiver_color: receiver}
    if not hit:
        # A near miss extrapolates to a missing anchor inside the deliberate gap.
        inp = _add_clutter(rng, inp, masks, bg, receiver_color, [(row, gate-1), (row+1, gate-2)])
    # A closer diagonal receiver pixel must not capture the beam.
    inp = _add_clutter(rng, inp, masks, bg, receiver_color, [(row, col+3), (row+1, col+4)])
    for _ in range(int(rng.integers(1, degree+4))):
        r, c = int(rng.integers(h)), int(rng.integers(w))
        dr, dc = int(rng.choice((-1, 1))), int(rng.choice((-1, 1)))
        length = int(rng.choice((2, 3)))
        cells = [(r+k*dr, c+k*dc) for k in range(length)]
        inp = _add_clutter(rng, inp, masks, bg, int(rng.choice((signal, receiver_color))), cells)
    out, (_, ray, active, contact) = _render(shape, bg, signal, receiver_color, source,
                                          receiver, pivot, (0, 1), old)
    assert (contact is not None) == hit
    if witness:
        assert contact == (row, gate), 'witness must touch the displaced contact first'
        assert receiver[row, gate+1:].any(), 'witness needs another receiver behind the contact'
        assert (row+10, gate) not in active, 'intentional break must isolate the far trunk'
        assert receiver_color in np.unique(out), 'witness needs an unenergized component'
        assert len(active) > 5 and len(ray) >= 4
    # Construction renders from the undamaged circuit, then checks public evidence.
    assert np.array_equal(solve(inp), out), 'visible drawing did not recover the construction'
    assert _thin(receiver) and _thin(source)
    assert np.count_nonzero(inp == bg)*2 > inp.size
    return {'input': inp, 'output': out}


def _transform(pair, turns, reflect):
    return {k: np.rot90(np.fliplr(a) if reflect else a, turns).copy() for k, a in pair.items()}


def _fallback_pair(palette, hit, bends, serial):
    """Finite constructive safety net; same rule, with 24 distinct cold arms.

    This is not a parent template. It guarantees generation can complete even
    when all procedural proposals collide or produce an unintended cue.
    """
    bg, signal, receiver_color, annotation = palette
    shape, pivot = (30, 30), (8, 9)
    points = {
        0: [(8, 9), (8, 0)],
        1: [(8, 9), (8, 4), (0, 4)],
        2: [(8, 9), (8, 4), (22, 4), (22, 0)],
        3: [(8, 9), (8, 6), (20, 6), (20, 2), (0, 2)],
    }[bends]
    old = _polyline(points)[1:]
    source = _mask(shape, old + _line(pivot, (13, 9)))
    receiver = np.zeros(shape, bool)
    receiver[2:29, 18] = True
    receiver[2, 11:19] = True
    receiver[13:16, 18] = False
    if hit:
        receiver[6:11, 25] = True
    else:
        receiver[7:10, 18] = False
    # Across all serials these pixel sets are distinct, even when the arm joins
    # the lower trunk. Twenty-four choices exceed the maximum 13 earlier inputs.
    row, start = (25 if serial < 14 else 27), 11+serial % 14
    receiver[row, start:start+3] = True
    inp = np.full(shape, bg, dtype=np.int8)
    inp[source], inp[receiver] = signal, receiver_color
    inp[8, 10] = annotation
    inp[2, 13:16] = bg
    inp[3, 13] = inp[4, 14] = receiver_color
    if hit:
        inp[7:10, 18] = bg
        inp[7, 17] = inp[8, 16] = receiver_color
    else:
        inp[8, 17] = inp[9, 16] = receiver_color
    inp[8, 12] = inp[9, 13] = receiver_color
    out, _ = _render(shape, bg, signal, receiver_color, source, receiver, pivot, (0, 1), old)
    assert np.array_equal(solve(inp), out)
    return {'input': inp, 'output': out}


def _check_count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be in {lo}..{hi}')
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _check_count(num_train, 'num_train', 2, 10)
    t = _check_count(num_test, 'num_test', 1, 4)
    if n <= t:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    varied_palette = bool(rng.integers(2))
    degree = int(rng.choice((0, 1, 2), p=(.2, .5, .3)))
    regime, orientation = int(rng.integers(2)), int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', n), ('test', t)):
        for index in range(count):
            witness = split == 'train' and index == 0
            hit = True if witness else (False if split == 'train' and index == 1 else bool(rng.random() < .72))
            extra = split == 'test' and bool(rng.integers(2))
            pair_degree = min(3, degree + int(extra))
            bends = 0 if witness else (int(rng.integers(1, 3)) if split == 'train' and index == 1
                                       else int(rng.integers(4 if split == 'test' and regime == 1 else 3)))
            if split == 'test' and regime == 1 and index == 0:
                bends = 3  # A visible route-complexity shift, not just a wider prior.
            contact_repair = witness or bool(rng.integers(2))
            turns = (orientation + (index % 2)*2) % 4 if split == 'train' and regime == 0 else int(rng.integers(4))
            if split == 'test' and regime == 0:
                turns = (orientation + int(rng.choice((1, 3)))) % 4
            for attempt in range(64):
                colors = tuple(map(int, rng.choice(10, 4, replace=False))) if varied_palette else palette
                try:
                    pair = _build_pair(rng, colors, hit, witness, pair_degree, bends, contact_repair)
                    pair = _transform(pair, turns, bool(rng.integers(2)))
                    if _key(pair['input']) in seen:
                        continue
                    _analyze_pair(pair)
                    break
                except AssertionError:
                    continue
            else:
                # Geometry enumeration, rather than palette changes, resolves
                # any safety-net duplicate while respecting fixed palettes.
                for serial in range(24):
                    pair = _transform(_fallback_pair(palette, hit, bends, serial), turns, False)
                    if _key(pair['input']) not in seen:
                        _analyze_pair(pair)
                        break
                else:
                    raise AssertionError('finite uniqueness guarantee failed')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    inp, out = (np.asarray(pair[k]) for k in ('input', 'output'))
    for a in (inp, out):
        assert a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
        assert all(1 <= n <= 30 for n in a.shape) and np.all((0 <= a) & (a <= 9))
    assert inp.shape == out.shape
    bg, annotation, signal, receiver_color, records, pivot, direction, old = _roles_and_paths(inp)
    source, receiver = records[signal][0], records[receiver_color][0]
    assert not (source & receiver).any(), 'input circuits must not overlap'
    assert _thin(source) and _thin(receiver), 'unit-width drawing required'
    assert len(_components(source)) == 1, 'one launcher required'
    assert np.all(~(source | receiver)[inp == annotation]), 'annotation is outside the drawing'
    for color in (signal, receiver_color):
        assert np.count_nonzero(inp == color) >= 5
        cleaned, repairs, added, removed = records[color]
        assert not any(not _neighbors(p, cleaned) for p in _cells(cleaned)), 'isolated non-noise dot'
        # Removed cells form only isolated two/three-cell 45-degree fragments.
        remaining = set(removed)
        while remaining:
            p = min(remaining)
            remaining.remove(p)
            component, stack = {p}, [p]
            while stack:
                q = stack.pop()
                for d in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    neighbor = _add(q, d)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            points = sorted(component)
            assert len(points) in (2, 3)
            differences = {(b[0]-a[0], b[1]-a[1]) for a, b in zip(points, points[1:])}
            assert differences in ({(1, 1)}, {(1, -1)}), 'clutter must be straight diagonal fragments'
    expected, (kept, ray, active, contact) = _render(inp.shape, bg, signal, receiver_color,
                                                  source, receiver, pivot, direction, old)
    assert np.array_equal(out, expected), 'incorrect repair, deletion, ray or component recolor'
    assert annotation not in np.unique(out)
    assert all(out[p] == signal for p in active | set(ray) | _cells(kept))
    assert all(out[p] == receiver_color for p in _cells(receiver)-active)
    assert all(out[p] == bg for p in set(old)-set(ray)-active)
    repaired_receiver = records[receiver_color][2]
    assert repaired_receiver, 'receiver reconstruction must be demonstrated'
    # Strong counterfactuals: delete only the repaired receiver pixels, preserving
    # launcher inference, and independently omit the entire routing operation.
    unrepaired_receiver = receiver & ~_mask(inp.shape, repaired_receiver)
    no_a, _ = _render(inp.shape, bg, signal, receiver_color, source, unrepaired_receiver,
                      pivot, direction, old)
    no_b = np.full(inp.shape, bg, np.int8)
    no_b[receiver], no_b[source] = receiver_color, signal
    gaps = []
    for p in sorted(_cells(receiver)):
        for d in ((1, 0), (0, 1)):
            if _has(receiver, _add(p, d, 4)) and not any(_has(receiver, _add(p, d, k)) for k in (1, 2, 3)):
                gaps.append((p, d))
    ds = [(b[0]-a[0], b[1]-a[1]) for a, b in zip([pivot]+old, old)]
    bends = sum(a != b for a, b in zip(ds, ds[1:]))
    return {'shape': inp.shape, 'palette': (bg, signal, receiver_color, annotation),
            'direction': direction, 'repairs': sum(len(x[1]) for x in records.values()),
            'receiver_repairs': len(records[receiver_color][1]),
            'components': len(_components(receiver)), 'bends': bends,
            'contact': contact, 'ray_length': len(ray), 'active_size': len(active),
            'active_repairs': len(active & repaired_receiver),
            'intentional_gaps': len(gaps), 'a_changes_answer': not np.array_equal(no_a, out),
            'b_changes_answer': not np.array_equal(no_b, out),
            'clutter': sum(len(x[3]) for x in records.values()),
            'branches': sum(len(_neighbors(p, receiver)) >= 3 for p in _cells(receiver)),
            'structure': (inp.shape, (inp == signal).tobytes(), (inp == receiver_color).tobytes())}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    n, t = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            assert _key(pair['input']) not in seen, 'duplicate input'
            seen.add(_key(pair['input']))
            info = _analyze_pair(pair)
            assert info['a_changes_answer'] and info['b_changes_answer']
            if split == 'train':
                train.append(info)
    assert any(x['contact'] is None and x['intentional_gaps'] for x in train), 'open-contact witness missing'
    assert any(x['contact'] is not None and x['active_repairs'] and x['intentional_gaps']
               and x['components'] >= 2 for x in train), 'closed-contact and isolation witness missing'
    assert any(x['bends'] for x in train), 'bent old-branch witness missing'


def self_test() -> dict:
    """Bounded 42-seed semantic checks, symmetries, negatives and diversity."""
    for hit in (False, True):
        fallback_keys = set()
        for serial in range(24):
            pair = _fallback_pair((0, 1, 3, 2), hit, 3, serial)
            _analyze_pair(pair)
            fallback_keys.add(_key(pair['input']))
        assert len(fallback_keys) == 24
        for bends in range(3):
            _analyze_pair(_fallback_pair((0, 1, 3, 2), hit, bends, 0))
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    infos = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                infos.append(_analyze_pair(pair))
                assert np.array_equal(solve(np.asarray(pair['input']).copy()), pair['output'])
        # Geometrical roles support all rotations/reflections without training state.
        for pair in puzzle['test']:
            for turns in range(4):
                transformed = _transform(pair, turns, True)
                assert np.array_equal(solve(transformed['input']), transformed['output'])
    assert len({tuple(_key(p['input']) for p in q['train']+q['test']) for q in puzzles}) == 42
    assert len({x['structure'] for x in infos}) >= 190
    assert len({x['shape'] for x in infos}) >= 60
    assert len({x['palette'] for x in infos}) >= 30
    assert {x['direction'] for x in infos} == set(DIRS)
    assert {x['bends'] for x in infos} >= {0, 1, 2, 3}
    assert max(x['repairs'] for x in infos) >= 3
    assert max(x['components'] for x in infos) >= 4
    assert max(x['branches'] for x in infos) >= 3
    assert any(x['contact'] is None for x in infos) and any(x['active_repairs'] for x in infos)
    assert any(x['shape'][0] == 30 or x['shape'][1] == 30 for x in infos)
    # Complete-grid mutation checks include routing, circuit repair and preservation.
    for puzzle in puzzles[:3]:
        pair = puzzle['test'][0]
        for point in (tuple(np.argwhere(pair['input'] == pair['output'])[0]),
                      tuple(np.argwhere(pair['input'] != pair['output'])[0])):
            bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
            bad['test'][0]['output'][point] = (int(bad['test'][0]['output'][point])+1) % 10
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError('corrupt output accepted')
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            validate_generated_puzzle(generate_puzzle(31415, n, t))
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    first, again = generate_puzzle(0), generate_puzzle(0)
    assert all(_key(a[k]) == _key(b[k]) for s in first for a, b in zip(first[s], again[s]) for k in a)
    return {'seeds': 42, 'pairs': len(infos), 'shapes': len({x['shape'] for x in infos}),
            'structures': len({x['structure'] for x in infos}),
            'palettes': len({x['palette'] for x in infos}),
            'repair_range': [min(x['repairs'] for x in infos), max(x['repairs'] for x in infos)],
            'receiver_component_range': [min(x['components'] for x in infos), max(x['components'] for x in infos)],
            'old_bends': sorted({x['bends'] for x in infos}),
            'independent_semantic_verification': False}
