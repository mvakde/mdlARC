"""ARC family: collision-activated relay spotlights inside a widening field.

Only NumPy and the standard library are used. All semantic roles and geometric
parameters are recovered from the input. No reference files are imported.
"""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid grid color")
    return a


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    pending = [cells.pop()]
    while pending:
        r, c = pending.pop()
        for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if q in cells:
                cells.remove(q)
                pending.append(q)
    return not cells


def _terminal_fit(left, right):
    """Read synchronized thin stairs, with two complete widening steps.

    The longest periodic terminal suffix is the terminal flare. Rear chamber
    shape is deliberately not extrapolated. The least period resolves ties.
    """
    end = len(left)-1
    times = np.arange(len(left)) - end
    choices = []
    for period in (1, 2, 3):
        for phase in range(period):
            offset = (times + phase) // period
            same = ((np.asarray(left) == left[-1] - offset) &
                    (np.asarray(right) == right[-1] + offset))
            length = 0
            for equal in same[::-1]:
                if not equal:
                    break
                length += 1
            if length >= 2*period+1:
                choices.append((length, -period, phase))
    if not choices:
        return None
    length, negative_period, phase = max(choices)
    return -negative_period, phase, length


def _inspect(input_grid):
    """Recognize one closed cap, paired rails, payload and mouth filaments."""
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel().astype(int), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > inp.size/2, "background must occupy over half the input")
    matches = []
    for rotation in range(4):
        a = np.rot90(inp, -rotation)
        h, w = a.shape
        for wall in map(int, np.unique(a)):
            if wall == bg:
                continue
            positions = np.argwhere(a == wall)
            rear, mouth = int(positions[:, 0].min()), int(positions[:, 0].max())
            cap = np.flatnonzero(a[rear] == wall)
            if len(cap) < 3 or np.any(np.diff(cap) != 1) or mouth-rear < 4:
                continue
            left, right = [], []
            for r in range(rear+1, mouth+1):
                rails = np.flatnonzero(a[r] == wall)
                if len(rails) != 2 or rails[1]-rails[0] < 2:
                    break
                left.append(int(rails[0]))
                right.append(int(rails[1]))
            else:
                if left[0] != cap[0] or right[0] != cap[-1]:
                    continue
                if np.any(np.abs(np.diff(left)) > 1) or np.any(np.abs(np.diff(right)) > 1):
                    continue
                fit = _terminal_fit(left, right)
                if fit is None:
                    continue
                period, phase, flare_length = fit
                bounds = [(0, -1)] * h
                field = np.zeros(a.shape, dtype=bool)
                for r in range(rear+1, h):
                    if r <= mouth:
                        lo, hi = left[r-rear-1], right[r-rear-1]
                    else:
                        step = (r-mouth+phase)//period
                        lo, hi = left[-1]-step, right[-1]+step
                    bounds[r] = (lo, hi)
                    field[r] = (np.arange(w) > lo) & (np.arange(w) < hi)
                chamber = field.copy()
                chamber[mouth+1:] = False
                if not _connected(chamber):
                    continue
                interior_colors = set(map(int, np.unique(a[chamber]))) - {bg}
                if len(interior_colors) != 2:
                    continue
                for guide in interior_colors:
                    light = next(c for c in interior_colors if c != guide)
                    gm, lm = a == guide, a == light
                    if np.any((gm | lm) & ~chamber) or not _connected(lm):
                        continue
                    gp, lp = np.argwhere(gm), np.argwhere(lm)
                    entries = sorted(set(map(int, gp[:, 1])))
                    if not 1 <= len(entries) <= 4 or any(b-a < 2 for a, b in zip(entries, entries[1:])):
                        continue
                    lengths, valid = [], True
                    for c in entries:
                        rows = np.flatnonzero(gm[:, c])
                        n = len(rows)
                        if not 2 <= n <= 4 or not np.array_equal(rows, np.arange(mouth-n+1, mouth+1)):
                            valid = False
                            break
                        lengths.append(n)
                    if not valid or len(lp) > 6 or int(lp[:, 0].max()) >= int(gp[:, 0].min()):
                        continue
                    extras = (a != bg) & (a != wall) & ~gm & ~lm
                    extra_colors = set(map(int, np.unique(a[extras])))
                    if len(extra_colors) > 1 or np.any(extras[:mouth+1]):
                        continue
                    matches.append(dict(a=a, rotation=rotation, bg=bg, wall=wall,
                                        guide=guide, light=light, obstacle=extras,
                                        rear=rear, mouth=mouth, period=period, phase=phase,
                                        flare_length=flare_length, field=field, bounds=bounds,
                                        entries=entries, stub_lengths=lengths))
    _require(len(matches) == 1, "input must identify exactly one funnel and role assignment")
    return matches[0]


def _runs(row):
    intervals = []
    c = 0
    while c < len(row):
        if not row[c]:
            c += 1
            continue
        lo = c
        while c+1 < len(row) and row[c+1]:
            c += 1
        intervals.append((lo, c))
        c += 1
    return intervals


def _component_intervals(mask):
    """Deliberately wrong whole-object interpretation, used only in checks."""
    cells = set(map(tuple, np.argwhere(mask)))
    result = {}
    while cells:
        start = cells.pop()
        group, pending = {start}, [start]
        while pending:
            r, c = pending.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in cells:
                    cells.remove(q)
                    group.add(q)
                    pending.append(q)
        interval = min(c for _, c in group), max(c for _, c in group)
        result.update((point, interval) for point in group)
    return result


def _trace(d, mode="normal"):
    """Simultaneous layer propagation, on immutable original baffle cells."""
    a, field, obstacle = d['a'], d['field'], d['obstacle']
    h, w = a.shape
    path = a == d['guide']
    active = set(d['entries'])
    sources, events, incoming = set(), [], {}
    bounded = mode != 'unbounded'
    objects = _component_intervals(obstacle) if mode == 'component_bbox' else None
    for r in range(d['mouth']+1, h):
        active = {c for c in active if not bounded or field[r, c]}
        incoming[r] = set(active)
        hits = {}
        following = set()
        intervals = _runs(obstacle[r])
        for c in sorted(active):
            if obstacle[r, c]:
                run = objects[(r, c)] if objects is not None else next((lo, hi) for lo, hi in intervals if lo <= c <= hi)
                hits.setdefault(run, []).append(c)
            else:
                path[r, c] = True
                following.add(c)
        for index, ((lo, hi), columns) in enumerate(sorted(hits.items())):
            if mode == 'stop' or (mode == 'one_hit' and index > 0):
                continue
            if mode == 'ignore_singletons' and lo == hi:
                continue
            flanks = []
            for c in range(max(0, lo-1), min(w, hi+2)):
                if (not bounded or field[r-1, c]) and a[r-1, c] == d['bg']:
                    path[r-1, c] = True
            for c in (lo-1, hi+1):
                if 0 <= c < w and (not bounded or field[r, c]):
                    following.add(c)
                    path[r, c] = True
                    sources.add((r, c))
                    flanks.append(c)
            events.append((r, lo, hi, tuple(columns), tuple(flanks)))
        active = following
    return path, sources, events, incoming


def _illumination(d, sources, mode="normal"):
    """Union light from the original chamber and independent point relays.

    Each source has its own set of blocked lanes. An object outside that
    source's prospective footprint never starts a shadow for that source.
    """
    field, obstacle = d['field'], d['obstacle']
    h, w = field.shape
    lit = np.zeros(field.shape, dtype=bool)
    footprints = [(d['rear']+1, field)]
    if mode != 'no_relays':
        for birth, c in sorted(sources):
            cone = np.zeros(field.shape, dtype=bool)
            for r in range(birth, h):
                radius = 0 if mode == 'no_spread' else (r-birth)//d['period']
                cone[r, max(0, c-radius):min(w, c+radius+1)] = True
            footprints.append((birth, cone & field))
    for birth, footprint in footprints:
        blocked = np.zeros(w, dtype=bool)
        for r in range(birth, h):
            if mode != 'no_shadows':
                blocked |= footprint[r] & obstacle[r]
            lit[r] |= footprint[r] & ~blocked & ~obstacle[r]
    return lit


def _render(d, shortcut="normal"):
    trace_mode = {'no_branch': 'stop', 'no_clip': 'unbounded',
                  'one_hit': 'one_hit', 'ignore_singletons': 'ignore_singletons',
                  'component_bbox': 'component_bbox'}.get(shortcut, 'normal')
    path, sources, events, incoming = _trace(d, trace_mode)
    if shortcut == 'all_baffles':
        for r in range(d['mouth']+1, d['a'].shape[0]):
            for lo, hi in _runs(d['obstacle'][r]):
                for c in (lo-1, hi+1):
                    if 0 <= c < d['a'].shape[1] and d['field'][r, c]:
                        sources.add((r, c))
    if shortcut == 'continuous_relays':
        sources |= set(map(tuple, np.argwhere(path & d['field'])))
    lit = _illumination(d, sources, shortcut)
    out = d['a'].copy()
    out[lit & (d['a'] == d['bg'])] = d['light']
    out[path & (d['a'] == d['bg'])] = d['guide']
    return out, dict(path=path, sources=sources, events=events, incoming=incoming, lit=lit)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique family output using only visible input evidence."""
    d = _inspect(input_grid)
    out, _ = _render(d)
    return np.rot90(out, d['rotation']).copy()


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _witnesses(d, out):
    names = ('no_branch', 'no_relays', 'no_shadows', 'no_spread', 'no_clip',
             'all_baffles', 'one_hit', 'ignore_singletons', 'continuous_relays', 'component_bbox')
    return {name for name in names if not np.array_equal(_render(d, name)[0], out)}


def _patch(rng, candidates, count):
    cells = set(map(tuple, np.argwhere(candidates)))
    start = sorted(cells)[int(rng.integers(len(cells)))]
    chosen = {start}
    for _ in range(count-1):
        frontier = sorted({q for r, c in chosen for q in
                           ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                           if q in cells and q not in chosen})
        if not frontier:
            break
        chosen.add(frontier[int(rng.integers(len(frontier)))])
    return chosen


def _canvas(rng, style, projection_range):
    period = int(rng.integers(1, 4))
    rear = int(rng.integers(0, 4))
    length = int(rng.integers(2, 6))
    profiles = []
    for _ in range(2):
        end = int(rng.integers(3, 5))
        profile = [end]
        for _ in range(length-1):
            profile.append(max(2, min(5, profile[-1]+int(rng.integers(-1, 2)))))
        profiles.append(list(reversed(profile)))
    flare = 2*period+1+int(rng.integers(period))
    left_widths = profiles[0] + [profiles[0][-1]+t//period for t in range(1, flare+1)]
    right_widths = profiles[1] + [profiles[1][-1]+t//period for t in range(1, flare+1)]
    mouth = rear+len(left_widths)
    low, high = projection_range
    high = min(high, 29-mouth)
    low = min(low, high)
    projection = int(rng.integers(low, high+1))
    h = mouth+1+projection
    w = int(rng.integers(max(left_widths)+max(right_widths)+1, 31))
    center = int(rng.integers(max(left_widths), w-max(right_widths)))
    a = np.zeros((h, w), dtype=np.int8)
    a[rear, center-profiles[0][0]:center+profiles[1][0]+1] = 1
    field = np.zeros(a.shape, dtype=bool)
    for j, (lw, rw) in enumerate(zip(left_widths, right_widths), rear+1):
        a[j, center-lw] = a[j, center+rw] = 1
        field[j, center-lw+1:center+rw] = True
    phase = flare % period
    for r in range(mouth+1, h):
        step = (r-mouth+phase)//period
        field[r, max(0, center-left_widths[-1]-step+1):min(w, center+right_widths[-1]+step)] = True
    stub_length = 2 if style in ('teach_relay', 'teach_clip') else int(rng.integers(2, 5))
    allowed = np.flatnonzero(np.all(field[mouth-stub_length+1:mouth+1], axis=0))
    if style == 'teach_relay':
        entries = [int(allowed[len(allowed)//2])]
    elif style == 'teach_clip':
        entries = [int(allowed[0]), int(allowed[-1])]
    else:
        wanted = int(rng.integers(1, min(4, (len(allowed)+1)//2)+1))
        entries = []
        for c in rng.permutation(allowed):
            if all(abs(int(c)-x) >= 2 for x in entries):
                entries.append(int(c))
                if len(entries) == wanted:
                    break
        entries.sort()
    a[mouth-stub_length+1:mouth+1, entries] = 2
    candidates = field.copy()
    candidates[rear+length+1:] = False
    candidates[mouth-stub_length+1:] = False
    for r, c in _patch(rng, candidates, int(rng.integers(1, 7))):
        a[r, c] = 3
    return a, field, mouth, entries


def _add_random_events(rng, a, field, mouth, budget, start=None):
    """Schedule local runs around actual active rays, allowing disjoint hits."""
    h, w = a.shape
    start = mouth+1 if start is None else start
    eligible = list(range(start, h))
    rows = set(map(int, rng.choice(eligible, min(budget, len(eligible)), replace=False))) if eligible else set()
    active = set(map(int, np.flatnonzero(a[mouth] == 2)))
    for r in range(mouth+1, h):
        active = {c for c in active if field[r, c]}
        if r in rows and active:
            pivots = list(sorted(active))
            rng.shuffle(pivots)
            for pivot in pivots[:int(rng.integers(1, min(3, len(pivots))+1))]:
                if a[r, pivot] != 0:
                    continue
                lo = hi = pivot
                available = [c for c in range(w) if a[r, c] == 0 and
                             (c == 0 or a[r, c-1] != 4) and (c == w-1 or a[r, c+1] != 4)]
                while lo-1 in available:
                    lo -= 1
                while hi+1 in available:
                    hi += 1
                limit = hi-lo+1
                n = int(rng.integers(1, min(limit, 5)+1)) if rng.random() < .75 else int(rng.integers(1, limit+1))
                left = int(rng.integers(max(lo, pivot-n+1), min(pivot, hi-n+1)+1))
                a[r, left:left+n] = 4
        following = set()
        for c in active:
            if a[r, c] == 4:
                lo = hi = c
                while lo and a[r, lo-1] == 4:
                    lo -= 1
                while hi+1 < w and a[r, hi+1] == 4:
                    hi += 1
                following.update(x for x in (lo-1, hi+1) if 0 <= x < w and field[r, x])
            else:
                following.add(c)
        active = following


def _scene(rng, style, projection_range):
    a, field, mouth, entries = _canvas(rng, style, projection_range)
    h, w = a.shape
    if style == 'teach_relay':
        c = entries[0]
        r = mouth+int(rng.integers(2, 4))
        n = int(rng.integers(1, 4))
        lo = c-int(rng.integers(n))
        a[r, lo:lo+n] = 4
        # Same-length uncontacted baffle in an earlier illuminated layer.
        inert_row = mouth+1
        options = [x for x in range(w-n+1) if np.all(field[inert_row, x:x+n])
                   and not x-1 <= c <= x+n]
        if not options:
            return None
        x = int(rng.choice(options))
        a[inert_row, x:x+n] = 4
    elif style == 'teach_clip':
        r = mouth+int(rng.integers(1, 4))
        c, other = entries
        leftmost = int(np.flatnonzero(field[r])[0])
        lo = max(0, leftmost-1-int(rng.integers(0, 3)))
        end = min(c+int(rng.integers(0, 2)), other-3)
        a[r, lo:end+1] = 4
        other_end = min(w-1, other+int(rng.integers(1, 3))-1)
        a[r, other:other_end+1] = 4
        # Two or more descendant rays strike one later run, then continue.
        descendants = [x for x in (end+1, other-1, other_end+1) if 0 <= x < w and field[r, x]]
        bridge_row = min(h-3, r+int(rng.integers(2, 5)))
        # A thin stem joins the earlier bar to a wider downstream crossbar.
        # Its component spans a different interval than the run first struck.
        a[r+1:bridge_row, end] = 4
        a[bridge_row, min(end, min(descendants)):max(descendants)+1] = 4
        _add_random_events(rng, a, field, mouth, int(rng.integers(0, 3)), bridge_row+2)
    else:
        budget = int(rng.integers(0, 7))
        _add_random_events(rng, a, field, mouth, budget)
        # Extra objects are placed without causal labels: their effect is
        # determined by the final input, not by the placement intention.
        for _ in range(int(rng.integers(0, 5))):
            r, c = int(rng.integers(mouth+1, h)), int(rng.integers(w))
            length = int(rng.integers(1, min(6, w-c)+1))
            a[r, c:c+length] = 4
            if rng.random() < .4:
                for rr in range(r+1, min(h, r+int(rng.integers(2, 5)))):
                    a[rr, c] = 4
    # Optional inert object beyond the virtual field, using the same color.
    outside = np.argwhere(~field & (a == 0) & (np.indices(a.shape)[0] > mouth))
    if len(outside) and rng.random() < .6:
        r, c = outside[int(rng.integers(len(outside)))]
        a[r, c] = 4
    return a


def _make_pair(rng, palette, rotation, style, projection_range, seen, demonstrated=None):
    required = ({'no_branch', 'no_relays', 'no_shadows', 'no_spread', 'all_baffles', 'continuous_relays'}
                if style == 'teach_relay' else
                {'no_clip', 'one_hit', 'no_branch', 'component_bbox'} if style == 'teach_clip' else set())
    for _ in range(160):
        canonical = _scene(rng, style, projection_range)
        if canonical is None:
            continue
        if rng.random() < .5:
            canonical = np.fliplr(canonical).copy()
        inp = np.rot90(np.asarray(palette, dtype=np.int8)[canonical], rotation).copy()
        if _key(inp) in seen:
            continue
        try:
            d = _inspect(inp)
        except AssertionError:
            continue
        out, details = _render(d)
        if np.array_equal(out, d['a']):
            continue
        if required and not required <= _witnesses(d, out):
            continue
        if style == 'teach_clip':
            virtual_loss = any(0 <= c < d['a'].shape[1] and not d['field'][r, c]
                               for r, lo, hi, _, _ in details['events'] for c in (lo-1, hi+1))
            if (len(details['events']) < 3 or not virtual_loss or
                    not any(len(columns) >= 2 for _, _, _, columns, _ in details['events'])):
                continue
        # The singleton shortcut is an optional case extension. Never demand
        # a test distinction that the actual demonstrations failed to expose.
        if demonstrated is not None and 'ignore_singletons' not in demonstrated:
            if not np.array_equal(_render(d, 'ignore_singletons')[0], out):
                continue
        return {'input': inp, 'output': np.rot90(out, d['rotation']).copy()}
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shared_palette = bool(rng.random() < .35)
    palette = rng.choice(10, 5, replace=False)
    preferred = int(rng.integers(4))
    regime = int(rng.integers(3))
    puzzle, seen, demonstrated = {'train': [], 'test': []}, set(), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            style = ('teach_relay' if i == 0 else 'teach_clip') if split == 'train' and i < 2 else 'general'
            colors = palette if shared_palette else rng.choice(10, 5, replace=False)
            if regime == 0:
                rotation = (preferred + (2*(i % 2) if split == 'train' else 1+2*(i % 2))) % 4
            elif regime == 1:
                rotation = preferred
            else:
                rotation = int(rng.integers(4))
            projection = (12, 17) if regime == 1 and split == 'test' else (9, 11) if regime == 1 else (9, 16)
            pair = _make_pair(rng, colors, rotation, style, projection, seen,
                              demonstrated if split == 'test' else None)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                d = _inspect(pair['input'])
                demonstrated |= _witnesses(d, np.rot90(pair['output'], -d['rotation']))
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal pair counts")
    seen, training_witnesses = set(), set()
    virtual_loss, multi_hit = False, False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape and not np.array_equal(inp, out), "invalid or unchanged output")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            d = _inspect(inp)
            expected, detail = _render(d)
            actual = np.rot90(out, -d['rotation'])
            _require(np.array_equal(actual, expected), "incorrect relay illumination")
            _require(np.array_equal(out[inp != d['bg']], inp[inp != d['bg']]), "original cells changed")
            changed = actual != d['a']
            _require(np.all(d['field'][changed]), "new material outside the virtual field")
            _require(set(map(int, np.unique(actual[changed]))) <= {d['guide'], d['light']}, "wrong added color")
            _require(all(d['field'][r, c] for r, c in detail['sources']), "relay outside field")
            for r, lo, hi, columns, flanks in detail['events']:
                _require(any(lo <= c <= hi for c in columns), "uncausal collision")
                _require((lo == 0 or not d['obstacle'][r, lo-1]) and
                         (hi == d['a'].shape[1]-1 or not d['obstacle'][r, hi+1]),
                         "collision run not maximal")
            if split == 'train':
                training_witnesses |= _witnesses(d, expected)
                virtual_loss |= any(0 <= c < d['a'].shape[1] and not d['field'][r, c]
                                    for r, lo, hi, _, _ in detail['events'] for c in (lo-1, hi+1))
                multi_hit |= any(len(columns) >= 2 for _, _, _, columns, _ in detail['events'])
            elif 'ignore_singletons' not in training_witnesses:
                _require(np.array_equal(_render(d, 'ignore_singletons')[0], expected),
                         "test requires an undemonstrated singleton distinction")
    required = {'no_branch', 'no_relays', 'no_shadows', 'no_spread', 'no_clip',
                'all_baffles', 'one_hit', 'continuous_relays', 'component_bbox'}
    _require(required <= training_witnesses, "training fails to distinguish a core shortcut")
    _require(virtual_loss and multi_hit, "training lacks virtual flank loss or same-run multi-hit")


def self_test() -> dict:
    """Bounded 42-seed coverage, semantic fixtures and mutation checks."""
    # Hand-written p=1 funnel: three-cell baffle, dark wake, then relighting.
    a = np.zeros((17, 17), dtype=np.int8)
    a[1, 6:11] = 1
    for r, lo, hi in ((2, 6, 10), (3, 6, 10), (4, 5, 11), (5, 4, 12), (6, 3, 13)):
        a[r, lo] = a[r, hi] = 1
    a[2, 8] = 3
    a[5:7, 8] = 2
    a[8, 7:10] = 4
    out = solve(a)
    _require(np.all(out[7, 6:11] == 2), "upstream crosspiece regression")
    _require(out[9, 8] == 0 and out[10, 8] == 3, "independent source shadow/relighting regression")
    _require(out[7, 2] == 0 and out[7, 3] == 3, "thin virtual boundary regression")
    _require(out[8, 6] == out[8, 10] == 2, "flank source regression")
    b = a.copy()
    b[5:7, 8] = 0
    b[5:7, 5] = 2
    b[8, 7:10] = 0
    b[8, :6] = 4
    clipped = solve(b)
    _require(clipped[8, 6] == 2 and clipped[9, 0] != 2, "virtual clipping regression")

    signatures, structures, shapes, palettes, directions, periods = set(), set(), set(), set(), set(), set()
    outlines, entry_counts, event_counts, relay_counts, reached_lengths = set(), set(), set(), set(), set()
    feature_counts = {name: 0 for name in ('shadow', 'relit', 'clipped', 'simultaneous', 'multi_hit', 'inert', 'connected')}
    pairs = 0
    for seed in range(42):
        nt = 2+seed % 9
        nq = 1+(seed//9) % min(4, nt-1)
        puzzle = generate_puzzle(seed, nt, nq)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                pairs += 1
                inp = pair['input']
                d = _inspect(inp)
                expected, details = _render(d)
                _require(np.array_equal(solve(inp), pair['output']), "input-only solver mismatch")
                _require(np.array_equal(solve(np.rot90(inp)), np.rot90(pair['output'])), "rotation covariance")
                shapes.add(inp.shape)
                directions.add(d['rotation'])
                periods.add(d['period'])
                palettes.add((d['bg'], d['wall'], d['guide'], d['light']))
                normalized = np.zeros(d['a'].shape, dtype=np.int8)
                for code, color in enumerate((d['wall'], d['guide'], d['light']), 1):
                    normalized[d['a'] == color] = code
                normalized[d['obstacle']] = 4
                structures.add(_key(normalized))
                coords = np.argwhere(d['a'] == d['wall'])
                crop = (d['a'][coords[:, 0].min():coords[:, 0].max()+1,
                               coords[:, 1].min():coords[:, 1].max()+1] == d['wall']).astype(np.int8)
                outlines.add(_key(crop))
                entry_counts.add(len(d['entries']))
                events = details['events']
                event_counts.add(len(events))
                relay_counts.add(len(details['sources']))
                reached_lengths.update(hi-lo+1 for _, lo, hi, _, _ in events)
                reached = {(r, lo, hi) for r, lo, hi, _, _ in events}
                feature_counts['inert'] += any((r, lo, hi) not in reached for r in range(d['mouth']+1, d['a'].shape[0]) for lo, hi in _runs(d['obstacle'][r]))
                feature_counts['clipped'] += any(len(flanks) < 2 for _, _, _, _, flanks in events)
                feature_counts['simultaneous'] += len({r for r, _, _, _, _ in events}) < len(events)
                feature_counts['multi_hit'] += any(len(columns) > 1 for _, _, _, columns, _ in events)
                feature_counts['connected'] += bool(np.any(d['obstacle'][:-1] & d['obstacle'][1:]))
                no_shadows = _render(d, 'no_shadows')[0]
                no_relays = _render(d, 'no_relays')[0]
                feature_counts['shadow'] += bool(np.any((expected == d['bg']) & (no_shadows == d['light'])))
                feature_counts['relit'] += bool(np.any((expected == d['light']) & (no_relays == d['bg'])))
        if seed in (0, 13, 41):
            again = generate_puzzle(seed, nt, nq)
            _require(all(_key(x[k]) == _key(y[k]) for s in puzzle for x, y in zip(puzzle[s], again[s])
                         for k in ('input', 'output')), "nondeterminism")
    _require(len(signatures) == 42 and len(structures) >= 150, "structural diversity collapsed")
    _require(len(shapes) >= 40 and len(outlines) >= 60, "geometry diversity collapsed")
    _require(directions == {0, 1, 2, 3} and periods == {1, 2, 3}, "orientation/cadence coverage")
    _require(len(palettes) >= 70 and {1, 2, 3} <= entry_counts, "palette/entry variation")
    _require(len(event_counts) >= 6 and len(relay_counts) >= 6 and len(reached_lengths) >= 10, "branching variation")
    _require(all(n > 0 for n in feature_counts.values()), "missing semantic coverage")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            generate_puzzle(73019, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(123, 2, 1)
    p = bad['test'][0]
    r, c = np.argwhere(p['input'] != p['output'])[0]
    p['output'][r, c] = p['input'][r, c]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, pairs=pairs, structures=len(structures), shapes=len(shapes),
                outlines=len(outlines), palettes=len(palettes), entry_counts=sorted(entry_counts),
                event_counts=sorted(event_counts), relay_counts=sorted(relay_counts),
                reached_lengths=sorted(reached_lengths), features=feature_counts)
