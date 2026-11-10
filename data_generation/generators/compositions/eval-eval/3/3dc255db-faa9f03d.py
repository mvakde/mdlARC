"""Unary-indexed route straightening, with an input-only geometric reader.

No parent code, fixtures, learned palette, or generation metadata is used by
solve. The constructor paints labels from ordered paths; the reader recovers
paths from colored pixels and locally witnessed repairs.
"""
from __future__ import annotations

import json
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 635365967, 354744295, 620679783, 187102769, 557224622,
              713391366, 899025107, 694699253, 410409580, 123920916,
              762953703, 1386589, 632663803, 55472006, 152865552,
              307940170, 491565849, 390537607, 326491900, 821705781,
              559849435, 358378141, 607887281, 366350430, 72486635,
              705867239, 954047467, 760180395, 170069961, 454672263,
              478259497, 553611745, 393245538, 743424610, 816891681,
              76495051, 155360148, 779707719, 573528304, 850976199,
              271942403)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _neighbors(p, cells):
    return [_add(p, d) for d in DIRS if _add(p, d) in cells]


def _components(cells, diagonal=False):
    remaining, result = set(cells), []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            p = stack.pop()
            for d in steps:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "noninteger grid"
    assert np.all((0 <= a) & (a <= 9)), "invalid colors"
    return a.astype(np.int8, copy=True)


def _background(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] * 2 > a.size, "background must occupy a strict majority"
    return bg


def _local(p, direction):
    # y increases toward the arrow's open end; x is transverse.
    dr, dc = direction
    return -p[0] * dr - p[1] * dc, -p[0] * dc + p[1] * dr


def _world(p, direction):
    y, x = p
    dr, dc = direction
    return -y * dr - x * dc, -y * dc + x * dr


def _read_envelope(cells, direction):
    """Read thin monotone arms, including stems and asymmetric shoulders."""
    local = {_local(p, direction) for p in cells}
    y0, y1 = min(y for y, x in local), max(y for y, x in local)
    rows = [sorted(x for y, x in local if y == row) for row in range(y0, y1 + 1)]
    if len(rows) < 4 or len(rows[0]) != 1:
        return None
    tip_x = rows[0][0]
    lo = hi = tip_x
    split, cavity = False, set()
    for i, row in enumerate(rows[1:], 1):
        if not split and row == [tip_x]:
            continue
        if len(row) != 2:
            return None
        left, right = row
        if not (left < tip_x < right and 0 <= lo-left <= 1 and 0 <= right-hi <= 1):
            return None
        split = True
        lo, hi = left, right
        cavity.update(_world((y0 + i, x), direction) for x in range(lo + 1, hi))
    if not cavity:
        return None
    return _world((y0, tip_x), direction), cavity


def _frames(a, bg):
    found = []
    for color in sorted(set(map(int, np.unique(a))) - {bg}):
        for comp in _components(map(tuple, np.argwhere(a == color)), diagonal=True):
            if len(comp) < 7:
                continue
            for d in DIRS:
                envelope = _read_envelope(comp, d)
                if envelope is None:
                    continue
                tip, cavity = envelope
                if any(not _inside(p, a.shape) for p in cavity):
                    continue
                payloads = {int(a[p]) for p in cavity} - {bg}
                if len(payloads) != 1 or color in payloads:
                    continue
                trace = next(iter(payloads))
                launch = _add(tip, d)
                if not _inside(launch, a.shape) or int(a[launch]) != trace:
                    continue
                rs, cs = zip(*comp)
                box = {(r, c) for r in range(min(rs), max(rs) + 1)
                       for c in range(min(cs), max(cs) + 1)}
                found.append(dict(frame=comp, cavity=cavity, tip=tip, direction=d,
                                  launch=launch, trace=trace, frame_color=color, box=box,
                                  payload={p for p in cavity if a[p] == trace}))
    assert found, "no arrow with an attached route"
    assert len({frozenset(f['frame']) for f in found}) == len(found), "ambiguous arrow direction"
    assert len({f['frame_color'] for f in found}) == 1, "one frame role per scene"
    occupied = set()
    for f in found:
        assert not occupied & f['box'], "overlapping arrows"
        occupied.update(f['box'])
    return sorted(found, key=lambda f: f['tip'])


def _ray(pivot, d, shape):
    cells, p = [], _add(pivot, d)
    while _inside(p, shape):
        cells.append(p)
        p = _add(p, d)
    return cells


def _analyze(input_grid):
    a = _grid(input_grid)
    bg = _background(a)
    frames = _frames(a, bg)
    fc = frames[0]['frame_color']
    traces = {f['trace'] for f in frames}
    assert fc not in traces
    roles = set(map(int, np.unique(a))) - {bg, fc} - traces
    assert len(roles) == 1, "a single distinct repair color must be visible"
    repair = next(iter(roles))
    boxes = set().union(*(f['box'] for f in frames))
    assert {p for p in map(tuple, np.argwhere(a == fc))} == set().union(*(f['frame'] for f in frames))
    layers = {c: {p for p in map(tuple, np.argwhere(a == c)) if p not in boxes} for c in traces}
    markers = set(map(tuple, np.argwhere(a == repair)))
    assert markers.isdisjoint(boxes)
    # Marked bends as well as straight markers have two same-color neighbors.
    # A marker is never adjacent to another marker or an omission.
    for p in sorted(markers):
        owners = [c for c, cells in layers.items() if len(_neighbors(p, cells)) == 2]
        assert len(owners) == 1, "marker lacks two uniquely colored route neighbors"
        layers[owners[0]].add(p)
    # Bridge only a single visibly witnessed straight hole. Infer simultaneously
    # from the original marked layers; do not grow strokes from their endpoints.
    additions = {c: set() for c in layers}
    for c, cells in layers.items():
        for p in sorted(cells):
            for d in ((1, 0), (0, 1)):
                gap, end = _add(p, d), _add(_add(p, d), d)
                if end in cells and gap not in cells and gap not in boxes and a[gap] == bg:
                    additions[c].add(gap)
    for c in layers:
        layers[c].update(additions[c])
    used = set()
    for f in frames:
        cells = layers[f['trace']]
        start = f['launch']
        assert len(_neighbors(start, cells)) == 1, "launch is not a route endpoint"
        path, previous, current = [], None, start
        while True:
            assert current not in path, "route cycle"
            path.append(current)
            following = [p for p in _neighbors(current, cells) if p != previous]
            assert len(following) <= 1, "route branch or unintended adjacency"
            if not following:
                break
            previous, current = current, following[0]
        assert not used & set(path), "two arrows own one route"
        used.update(path)
        hits = [i for i, p in enumerate(path) if p in markers]
        n = len(f['payload'])
        assert 1 <= n <= len(hits), "payload index exceeds repair sites"
        at = hits[n - 1]
        assert 0 < at < len(path)-1, "selected marker must be internal"
        p, before = path[at], path[at-1]
        incoming = (p[0]-before[0], p[1]-before[1])
        assert _add(f['tip'], f['direction']) == path[0]
        assert (path[1][0]-path[0][0], path[1][1]-path[0][1]) == f['direction']
        f.update(path=path, hits=hits, quantity=n, at=at, pivot=p,
                 incoming=incoming, prefix=set(path[:at+1]), suffix=set(path[at+1:]),
                 ray=set(_ray(p, incoming, a.shape)), gaps=additions[f['trace']] & set(path))
    assert used == set().union(*layers.values()), "unowned trace component"
    assert markers <= used
    for i, f in enumerate(frames):
        assert f['ray'] and not f['ray'] & boxes, "ray is empty or enters an arrow"
        assert not f['ray'] & f['prefix'], "ray returns through its retained route"
        for other in frames[i+1:]:
            assert not f['ray'] & other['ray'], "different commands have colliding rays"
        for other in frames:
            if other is not f:
                assert not f['ray'] & other['prefix'], "ray enters another retained route"
    return a, bg, repair, frames


def _render_read(a, bg, frames):
    # Clear all original routes together, then repaint retained prefixes/rays.
    out = a.copy()
    for f in frames:
        for p in f['payload'] | set(f['path']):
            out[p] = bg
    for f in frames:
        for p in f['prefix'] | f['ray']:
            out[p] = f['trace']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using this input's shapes, counts, colors, and route topology."""
    a, bg, repair, frames = _analyze(input_grid)
    return _render_read(a, bg, frames)


def _sample_frame(rng, shape, direction, min_capacity=9):
    for _ in range(40):
        left, right = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        stem, tail = int(rng.integers(0, 3)), int(rng.integers(2, 6))
        widths = [(0, 0)] * (stem+1) + [(-1, 1)]
        lo, hi = -1, 1
        while -lo < left or hi < right:
            dl = -lo < left and (hi == right or rng.random() < .75)
            dh = hi < right and (-lo == left or rng.random() < .75)
            if not dl and not dh:
                dl = -lo < left
                dh = not dl
            lo, hi = lo-int(dl), hi+int(dh)
            widths.append((lo, hi))
        widths.extend([(lo, hi)] * tail)
        frame, cavity = set(), set()
        for y, (lo, hi) in enumerate(widths):
            frame.update(_world((y, x), direction) for x in {lo, hi})
            cavity.update(_world((y, x), direction) for x in range(lo+1, hi))
        if len(cavity) < min_capacity:
            continue
        rs, cs = zip(*(frame | cavity))
        dr, dc = -min(rs), -min(cs)
        frame = {(r+dr, c+dc) for r, c in frame}
        cavity = {(r+dr, c+dc) for r, c in cavity}
        fh, fw = max(r for r, c in frame)+1, max(c for r, c in frame)+1
        if fh > shape[0] or fw > shape[1]:
            continue
        r0 = int(rng.integers(shape[0]-fh+1))
        c0 = int(rng.integers(shape[1]-fw+1))
        shift = lambda points: {(r+r0, c+c0) for r, c in points}
        tip = (dr+r0, dc+c0)
        return dict(frame=shift(frame), cavity=shift(cavity), tip=tip, direction=direction,
                    box={(r, c) for r in range(r0, r0+fh) for c in range(c0, c0+fw)})
    return None


def _line(a, b):
    dr, dc = int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1]))
    return [(a[0]+i*dr, a[1]+i*dc) for i in range(max(abs(b[0]-a[0]), abs(b[1]-a[1]))+1)]


def _simple(path):
    cells = set(path)
    return (len(cells) == len(path) and all(len(_neighbors(p, cells)) ==
            (1 if i in (0, len(path)-1) else 2) for i, p in enumerate(path)))


def _grow_path(rng, f, shape, blocked, desired_segments):
    start = _add(f['tip'], f['direction'])
    if not _inside(start, shape) or start in blocked or start in f['box']:
        return None
    path = [start]
    d = f['direction']
    forbidden = blocked | f['box']
    for seg in range(desired_segments):
        choices = []
        if seg == 0:
            for length in range(3, 8):
                choices.append((path[-1][0]+length*d[0], path[-1][1]+length*d[1]))
        else:
            axis = 1 if d[0] else 0
            for v in range(shape[axis]):
                if abs(v-path[-1][axis]) >= 4:
                    q = list(path[-1])
                    q[axis] = v
                    choices.append(tuple(q))
        rng.shuffle(choices)
        for end in choices:
            if not _inside(end, shape):
                continue
            extension = _line(path[-1], end)[1:]
            candidate = path + extension
            if not set(extension) & forbidden and _simple(candidate):
                d = (extension[0][0]-path[-1][0], extension[0][1]-path[-1][1])
                path = candidate
                break
        else:
            break
    return path if len(path) >= 13 else None


def _bends(path):
    return [i for i in range(1, len(path)-1)
            if (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]) !=
               (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])]


def _marker_indices(rng, path, quantity, anchor=False):
    bends = _bends(path)
    if anchor:
        if len(bends) < 3:
            return None
        chosen = [2, bends[0], bends[1]]
        remaining = list(range(bends[1]+3, len(path)-2))
        rng.shuffle(remaining)
        for i in remaining:
            if all(abs(i-j) >= 3 for j in chosen):
                chosen.append(i)
                if len(chosen) == 5:
                    break
        return sorted(chosen) if len(chosen) >= 4 else None
    desired = max(quantity+1, int(rng.integers(4, 9)))
    candidates = list(range(2, len(path)-2))
    # Mix bend and straight sites. Select one site at a time with intact witnesses.
    rng.shuffle(candidates)
    if rng.random() < .65:
        front = list(bends)
        rng.shuffle(front)
        candidates = front + candidates
    chosen = []
    for i in candidates:
        if all(abs(i-j) >= 3 for j in chosen):
            chosen.append(i)
        if len(chosen) == desired:
            break
    return sorted(chosen) if len(chosen) >= quantity+1 else None


def _command(f, path, hits, q):
    at = hits[q-1]
    p, before = path[at], path[at-1]
    return dict(f, path=path, hits=hits, quantity=q, at=at, pivot=p,
                incoming=(p[0]-before[0], p[1]-before[1]), prefix=set(path[:at+1]),
                suffix=set(path[at+1:]))


def _compatible_commands(commands, shape):
    boxes = set().union(*(f['box'] for f in commands))
    for f in commands:
        f['ray'] = set(_ray(f['pivot'], f['incoming'], shape))
        if not f['ray'] or f['ray'] & boxes:
            return False
        if any(f['ray'] & g['prefix'] for g in commands):
            return False
    return not any(f['ray'] & g['ray'] for i, f in enumerate(commands) for g in commands[i+1:])


def _payload(rng, cavity, q, compact):
    ordered = sorted(cavity)
    if compact:
        start = ordered[int(rng.integers(len(ordered)))]
    else:
        # Start at a transverse/axial extreme, so two scattered cells cannot
        # accidentally remain diagonally connected in a compact cavity.
        extremes = {min(ordered), max(ordered), min(ordered, key=lambda p: (p[1], p[0])),
                    max(ordered, key=lambda p: (p[1], p[0]))}
        options = sorted(extremes)
        start = options[int(rng.integers(len(options)))]
    chosen = {start}
    while len(chosen) < q:
        available = sorted(cavity-chosen)
        distances = [min((abs(p[0]-s[0])+abs(p[1]-s[1]) if compact else
                          max(abs(p[0]-s[0]), abs(p[1]-s[1]))) for s in chosen) for p in available]
        target = min(distances) if compact else max(distances)
        candidates = [p for p, distance in zip(available, distances) if distance == target]
        chosen.add(candidates[int(rng.integers(len(candidates)))])
    return chosen


def _paint(rng, commands, shape, palette, patterns=None):
    bg, fc, repair = palette[:3]
    inp = np.full(shape, bg, dtype=np.int8)
    out = inp.copy()
    for j, f in enumerate(commands):
        trace = palette[3+j]
        f['trace'] = trace
        for p in f['frame']:
            inp[p] = out[p] = fc
        compact = bool(rng.integers(2)) if patterns is None else patterns[j]
        payload = _payload(rng, f['cavity'], f['quantity'], compact)
        for p in payload:
            inp[p] = trace
        for p in f['path']:
            inp[p] = trace
        for i in f['hits']:
            inp[f['path'][i]] = repair
        # Optional one-cell holes on straight segments, away from marker witnesses.
        choices = [i for i in range(2, len(f['path'])-2) if i not in _bends(f['path'])
                   and all(abs(i-j) >= 3 for j in f['hits'])]
        rng.shuffle(choices)
        gaps = []
        for i in choices:
            if all(abs(i-j) >= 3 for j in gaps):
                gaps.append(i)
            if len(gaps) >= int(rng.integers(1, 4)):
                break
        if rng.random() < .2:
            gaps = []
        for i in gaps:
            inp[f['path'][i]] = bg
        for p in f['prefix'] | f['ray']:
            out[p] = trace
    # Labels above came from ordered construction, before the reader is invoked.
    predicted = solve(inp)
    assert np.array_equal(predicted, out), "constructed label disagrees with pixel reconstruction"
    return {'input': inp, 'output': out}


def _layout(rng, shape, count, quantities, directions, anchor=False):
    commands, blocked = [], set()
    for j in range(count):
        for _ in range(80):
            d = directions[int(rng.integers(len(directions)))]
            f = _sample_frame(rng, shape, d, min_capacity=max(9, quantities[j]+1))
            if f is None or f['box'] & blocked:
                continue
            path = _grow_path(rng, f, shape, blocked, int(rng.integers(5, 9)) if anchor else int(rng.integers(3, 9)))
            if path is None:
                continue
            hits = _marker_indices(rng, path, quantities[j], anchor=anchor and j == 0)
            if hits is None:
                continue
            command = _command(f, path, hits, quantities[j])
            commands.append(command)
            occupied = f['box'] | set(path)
            blocked.update(_add(p, d) for p in occupied for d in ((0, 0),)+DIRS)
            break
        else:
            return None
    return commands


def _anchors(rng, directions):
    # Matched scenes teach indexing with fixed geometry and simultaneously show
    # a ray crossing another command's deleted suffix. No crossing priority is needed.
    for _ in range(1600):
        shape = (int(rng.integers(26, 31)), int(rng.integers(26, 31)))
        low = _layout(rng, shape, 2, [2, 2], directions, anchor=True)
        if low is None:
            continue
        high = [_command(f, f['path'], f['hits'], 3 if i == 0 else 2) for i, f in enumerate(low)]
        if not _compatible_commands(low, shape) or not _compatible_commands(high, shape):
            continue
        if _index_mask(low[1], 1, shape) == low[1]['prefix'] | low[1]['ray']:
            # Equal quantities in connected/scattered hollows must actually
            # contradict component counting, not just differ decoratively.
            continue
        if not any(f['ray'] & g['suffix'] for commands in (low, high)
                   for f in commands for g in commands if f is not g):
            continue
        palette = list(map(int, rng.permutation(10)))
        try:
            a = _paint(rng, low, shape, palette, [True, False])
            b = _paint(rng, high, shape, palette, [False, True])
            # Keep every visible route and omission identical in the matched
            # demonstrations. Only the contents of the hollows change.
            matched = a['input'].copy()
            for f in high:
                for p in f['cavity']:
                    matched[p] = b['input'][p]
            b['input'] = matched
            assert np.array_equal(solve(matched), b['output'])
            _evidence([a, b])
        except AssertionError:
            continue
        if not np.array_equal(a['input'], b['input']):
            return [a, b]
    raise AssertionError("bounded search failed to find paired interacting demonstrations")


def _ordinary(rng, directions, test, regime):
    for _ in range(700):
        count = int(rng.choice([1, 1, 2, 2, 3]))
        if test and regime == 2:
            count = 3
        low = 13 if count == 1 else 22 if count == 2 else 27
        shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
        quantities = [int(rng.integers(1, 4)) for j in range(count)]
        if test and regime != 3:
            # Every non-rotation query exercises an unseen larger ordinal.
            # Other commands need not all be long or difficult in the same scene.
            quantities[0] = int(rng.integers(4, 9))
        elif test:
            quantities[0] = 3
        commands = _layout(rng, shape, count, quantities, directions)
        if commands is None or not _compatible_commands(commands, shape):
            continue
        if test and not any(all(_index_mask(f, j, shape) != f['prefix'] | f['ray']
                                    for j in (1, 2)) for f in commands):
            continue
        if test and regime == 1 and not any(len(_bends(f['path'])) >= 4 for f in commands):
            continue
        try:
            return _paint(rng, commands, shape, list(map(int, rng.permutation(10))))
        except AssertionError:
            continue
    raise AssertionError("bounded ordinary scene construction failed")


def _count(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in {lower}..{upper}")
    return value


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _index_mask(f, n, shape):
    at = f['hits'][n-1]
    path = f['path']
    p, before = path[at], path[at-1]
    incoming = (p[0]-before[0], p[1]-before[1])
    return set(path[:at+1]) | set(_ray(p, incoming, shape))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, 'num_train', 2, 10)
    m = _count(num_test, 'num_test', 1, 4)
    if n <= m:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(4))
    directions = [DIRS[int(i)] for i in rng.permutation(4)]
    train_directions = directions[:3] if regime == 3 else directions
    test_directions = directions[3:] if regime == 3 else directions
    train = _anchors(rng, train_directions)
    seen = {_key(p['input']) for p in train}
    puzzle = {'train': train, 'test': []}
    for split, count in (('train', n-2), ('test', m)):
        for _ in range(count):
            for attempt in range(20):
                pair = _ordinary(rng, test_directions if split == 'test' else train_directions,
                                 split == 'test', regime)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise AssertionError("distinct-input search exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(train):
    by_route, repeated, quantities = {}, {}, set()
    interaction = straight_marker = bend_selected = gaps = component_counterexample = long_ray = False
    for pair in train:
        a, bg, repair, frames = _analyze(pair['input'])
        for f in frames:
            n = f['quantity']
            quantities.add(n)
            signature = (a.shape, tuple(f['path']), tuple(f['hits']), frozenset(f['frame']), f['trace'])
            by_route.setdefault(signature, set()).add(n)
            repeated.setdefault((signature, n), set()).add(len(_components(f['payload'], diagonal=True)))
            component_count = len(_components(f['payload'], diagonal=True))
            component_counterexample |= (_index_mask(f, component_count, a.shape) !=
                                         f['prefix'] | f['ray'])
            long_ray |= len(f['ray']) > n
            straight_marker |= any(i not in _bends(f['path']) for i in f['hits'])
            bend_selected |= f['at'] in _bends(f['path'])
            gaps |= bool(f['gaps'] & f['prefix'])
            interaction |= any(f['ray'] & g['suffix'] for g in frames if f is not g)
    assert {2, 3} <= quantities, "two observed ordinal quantities are required"
    assert any({2, 3} <= ns for ns in by_route.values()), "missing matched count witness"
    assert any(len(cs) > 1 for cs in repeated.values()), "missing rearranged equal-count witness"
    assert component_counterexample, "component counting predicts every witnessed pivot"
    assert straight_marker and bend_selected, "markers must not merely encode bends"
    assert interaction, "no demonstration of simultaneous deletion/drawing"
    assert gaps, "no demonstration repairs a retained blank hole"
    assert long_ray, "training hides boundary continuation behind the payload quantity"


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    n, m = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, out = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == out.shape
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            a, bg, repair, frames = _analyze(a)
            assert np.array_equal(_render_read(a, bg, frames), out), "wrong complete output"
            assert not np.array_equal(a, out)
            assert not np.any(out == repair), "annotation survived"
            for f in frames:
                assert all(out[p] == bg for p in f['payload']), "payload not consumed"
                assert all(out[p] == f['frame_color'] for p in f['frame']), "frame changed"
                assert all(out[p] == f['trace'] for p in f['prefix'] | f['ray'])
                all_rays = set().union(*(g['ray'] for g in frames))
                assert all(out[p] == bg for p in f['suffix']-all_rays), "suffix retained"
            changed_region = set().union(*(f['payload'] | set(f['path']) | f['ray'] for f in frames))
            assert all(a[p] == out[p] for p in np.ndindex(a.shape) if p not in changed_region)
            if split == 'test':
                assert any(len(f['hits']) >= 2 and all(
                    _index_mask(f, j, a.shape) != f['prefix'] | f['ray'] for j in (1, 2))
                           for f in frames), "query hides the counting-dependent pivot"
    _evidence(puzzle['train'])


def self_test():
    """Bounded 42-seed variation, metamorphic checks, and semantic negatives."""
    signatures, shapes, counts, qs, turns, profiles, directions = set(), set(), set(), set(), set(), set(), set()
    marker_types, gap_counts, ray_lengths, palettes = set(), set(), set(), set()
    interactions = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        signatures.add(tuple(_key(q['input']) for s in ('train', 'test') for q in p[s]))
        for s in ('train', 'test'):
            for pair in p[s]:
                a, bg, repair, fs = _analyze(pair['input'])
                shapes.add(a.shape)
                counts.add(len(fs))
                palettes.add(tuple(map(int, np.unique(a))))
                for f in fs:
                    qs.add(f['quantity'])
                    turns.add(len(_bends(f['path'])))
                    directions.add(f['direction'])
                    local = [_local(x, f['direction']) for x in f['frame']]
                    y0, x0 = min(y for y, x in local), min(x for y, x in local)
                    profiles.add(tuple(sorted((y-y0, x-x0) for y, x in local)))
                    gap_counts.add(len(f['gaps']))
                    ray_lengths.add(len(f['ray']))
                    marker_types.update('bend' if i in _bends(f['path']) else 'straight' for i in f['hits'])
                    interactions += any(f['ray'] & g['suffix'] for g in fs if f is not g)
                # Rotation/reflection and arbitrary color permutations preserve semantics.
                if seed == 0:
                    for k in range(4):
                        x = np.rot90(np.fliplr(a), k).copy()
                        y = np.rot90(np.fliplr(pair['output']), k).copy()
                        remap = np.array([7, 4, 0, 8, 1, 9, 6, 2, 5, 3], dtype=np.int8)
                        assert np.array_equal(solve(remap[x]), remap[y])
        if seed == 0:
            same = generate_puzzle(seed)
            assert all(np.array_equal(a[r], b[r]) for s in p for a, b in zip(p[s], same[s]) for r in a)
            # Full-grid corruption rejection includes preserved cells and command edits.
            query = p['test'][0]
            a, bg, repair, fs = _analyze(query['input'])
            sites = [fs[0]['pivot'], min(fs[0]['ray']), min(fs[0]['payload']), min(fs[0]['frame']), (0, 0)]
            for site in sites:
                bad = {'train': p['train'], 'test': [dict(query)]}
                bad['test'][0]['output'] = query['output'].copy()
                bad['test'][0]['output'][site] = (int(query['output'][site])+1) % 10
                try:
                    validate_generated_puzzle(bad)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("corrupt output accepted")
    assert len(signatures) == 42 and len(shapes) >= 45 and counts == {1, 2, 3}
    assert len(profiles) >= 35 and len(palettes) >= 25
    assert {1, 2, 3, 4, 5} <= qs and {2, 3, 4} <= turns
    assert directions == set(DIRS) and marker_types == {'straight', 'bend'}
    assert 0 in gap_counts and max(gap_counts) >= 2
    assert min(ray_lengths) <= 3 and max(ray_lengths) >= 20 and interactions >= 42
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(27+n*5+m, n, m)
            assert len(p['train']) == n and len(p['test']) == m
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': 42}, separators=(',', ':')))
