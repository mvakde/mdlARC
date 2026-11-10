"""Equal travel: normalize routed strokes to the ordinal middle stroke.

Only visible input cells are used by solve.  Internal canonical colors are
permuted independently in every example; the key supplies all turn meanings.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

_D = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), 'invalid grid size')
    _need(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), 'invalid colors')
    return a.astype(np.int8, copy=False)


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _header_info(grid):
    frame = int(grid[0, 0])
    _need(np.all(grid[0] == frame), 'missing top frame')
    ends = np.flatnonzero(np.all(grid[1:] == frame, axis=1)) + 1
    _need(len(ends) > 0, 'missing bottom frame')
    end = int(ends[0])
    _need(2 <= end < len(grid) - 1, 'bad key height')
    rows = grid[1:end]
    _need(np.all(rows == rows[0]), 'inconsistent key rows')
    row = rows[0]
    _need(row[0] == row[-1] == frame, 'missing key walls')
    runs = []
    c = 0
    while c < len(row):
        if row[c] == frame:
            c += 1
        else:
            start = c
            while c < len(row) and row[c] != frame:
                c += 1
            runs.append((start, c))
    _need(runs and all(b - a >= 3 for a, b in runs), 'narrow key panels')
    background = int(row[runs[0][0] + 1])
    turns = {}
    for a, b in runs:
        _need(np.all(row[a + 1:b - 1] == background), 'inconsistent panel background')
        left, right = int(row[a]), int(row[b - 1])
        _need((left == background) != (right == background), 'ambiguous bar')
        color, side = (right, 1) if left == background else (left, -1)
        _need(color not in turns and color not in (background, frame), 'conflicting key')
        turns[color] = side
    _need(2 <= len(turns) <= 4, 'need two to four keys')
    return end + 1, frame, background, turns, tuple(b - a + 2 for a, b in runs)


def _routes(field, anchors, direction, background, payload, port, turns):
    """Trace through immutable geometry, ignoring all payload paint.

    A collision changes heading at the last free cell. Turns cost zero cells;
    moving into a free cell costs one. Event indices count cells before a turn.
    Full routes must be simple and must eventually leave the field.
    """
    h, w = field.shape
    paths, all_events = [], []
    for anchor in anchors:
        r, c = anchor
        d = direction
        path, events, states, visited = [], [], set(), {anchor}
        last_was_turn = False
        while True:
            state = (r, c, d)
            _need(state not in states, 'cyclic trajectory')
            states.add(state)
            dr, dc = _D[d]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            color = int(field[nr, nc])
            if color in turns:
                _need(not last_was_turn, 'no clear exit from a turning corner')
                nd = (d + turns[color]) % 4
                events.append((len(path), d, nd, color, nr, nc))
                d = nd
                last_was_turn = True
                continue
            _need(color in (background, payload) and color != port, 'route strikes a launch marker')
            _need((nr, nc) not in visited, 'self-intersecting trajectory')
            visited.add((nr, nc))
            path.append((nr, nc))
            r, c = nr, nc
            last_was_turn = False
        paths.append(path)
        all_events.append(events)
    return paths, all_events


def _rectangles(field, colors):
    h, w = field.shape
    seen, rects = set(), []
    for rr, cc in np.argwhere(np.isin(field, list(colors))):
        start = int(rr), int(cc)
        if start in seen:
            continue
        color = int(field[start])
        cells, todo = [], [start]
        seen.add(start)
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in _D:
                p = r + dr, c + dc
                if 0 <= p[0] < h and 0 <= p[1] < w and p not in seen and int(field[p]) == color:
                    seen.add(p)
                    todo.append(p)
        a, e = np.min(cells, axis=0)
        b, f = np.max(cells, axis=0) + 1
        _need(len(cells) == (b - a) * (f - e), 'blocker component must be rectangular')
        rects.append((int(a), int(b), int(e), int(f), color))
    return rects


def _analyze(value):
    grid = _grid(value)
    crop, frame, bg, turns, widths = _header_info(grid)
    field = grid[crop:]
    other = set(map(int, np.unique(field))) - set(turns) - {bg}
    _need(len(other) == 2 and frame not in other, 'need marker and path colors')
    candidates = []
    h, w = field.shape
    for port in sorted(other):
        cells = [tuple(map(int, p)) for p in np.argwhere(field == port)]
        n = len(cells)
        if n not in (3, 5, 7, 9):
            continue
        same_row = len({r for r, c in cells}) == 1
        same_col = len({c for r, c in cells}) == 1
        if same_row == same_col:
            continue
        anchors = sorted(cells, key=lambda p: p[1] if same_row else p[0])
        coords = [p[1] if same_row else p[0] for p in anchors]
        if min(np.diff(coords)) < 2:
            continue
        payload = next(iter(other - {port}))
        for d in ((0, 2) if same_row else (1, 3)):
            dr, dc = _D[d]
            if all(0 <= r + 2 * dr < h and 0 <= c + 2 * dc < w and
                   field[r + dr, c + dc] == payload and field[r + 2 * dr, c + 2 * dc] == payload
                   for r, c in anchors):
                candidates.append((port, payload, anchors, d, coords))
    _need(len(candidates) == 1, 'ambiguous launch array or heading')
    port, payload, anchors, d, coords = candidates[0]
    paths, events = _routes(field, anchors, d, bg, payload, port, turns)
    lengths, accounted = [], set()
    for path in paths:
        length = 0
        for p in path:
            if int(field[p]) != payload:
                break
            length += 1
        _need(2 <= length < len(path), 'each path needs a visible prefix and an empty next cell')
        lengths.append(length)
        accounted.update(path[:length])
    _need(accounted == {tuple(map(int, p)) for p in np.argwhere(field == payload)}, 'paint outside path prefixes')
    reference = lengths[len(lengths) // 2]
    _need(reference >= 3 and all(len(p) > reference for p in paths), 'target does not fit')
    rects = _rectangles(field, turns)
    return dict(grid=grid, field=field, crop=crop, frame=frame, background=bg,
                payload=payload, port=port, anchors=anchors, direction=d,
                coords=coords, paths=paths, events=events, lengths=lengths,
                reference=reference, turns=turns, rectangles=rects, panel_widths=widths)


def _paint(info, lengths):
    out = info['field'].copy()
    out[out == info['payload']] = info['background']
    for path, length in zip(info['paths'], lengths):
        for p in path[:length]:
            out[p] = info['payload']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the cropped field with all route prefixes of equal travel length."""
    info = _analyze(input_grid)
    return _paint(info, [info['reference']] * len(info['paths']))


def _make_header(rng, width, height, turns):
    keys = list(map(int, rng.permutation(list(turns))))
    sizes = np.full(len(keys), 5, dtype=int)
    if width > sizes.sum():
        if width % len(keys) == 0 and rng.random() < .3:
            sizes[:] = width // len(keys)
        else:
            sizes += rng.multinomial(width - int(sizes.sum()), np.full(len(keys), 1 / len(keys)))
    header = np.zeros((height, width), dtype=np.int8)
    header[[0, -1]] = 9
    c = 0
    for color, size in zip(keys, sizes):
        header[:, c] = header[:, c + size - 1] = 9
        header[1:-1, c + 1 if turns[color] == -1 else c + size - 2] = color
        c += int(size)
    return header


def _scene(rng, count, key_count, band_height, witness, mincap):
    limit = 30 - band_height
    low_h = max(5 * key_count, 2 * count + 1, 14)
    low_w = max(5 * key_count, mincap + 3, 19)
    h = int(rng.integers(low_h, limit + 1))
    w = int(rng.integers(low_w, limit + 1))
    c = int(rng.integers(0, min(4, w - mincap)))
    gaps = np.full(count - 1, 2, dtype=int)
    if witness:
        # A unique geometric-midpoint member differs from the ordinal middle.
        gaps[0 if rng.random() < .5 else -1] += int(rng.choice([5, 7]))
        _need(int(gaps.sum()) < h, 'witness does not fit')
    else:
        for _ in range(int(rng.integers(h - 1 - int(gaps.sum())))):
            gaps[int(rng.integers(count - 1))] += 1
    rows = np.r_[0, np.cumsum(gaps)]
    rows += int(rng.integers(h - int(rows[-1])))
    anchors = [(int(r), c) for r in rows]
    field = np.zeros((h, w), dtype=np.int8)
    for p in anchors:
        field[p] = 2
    sides = [-1, 1] + [int(rng.choice([-1, 1])) for _ in range(key_count - 2)]
    rng.shuffle(sides)
    turns = dict(zip(range(3, 3 + key_count), sides))
    paths, events = _routes(field, anchors, 1, 0, 1, 2, turns)
    rectangles = []
    target = int(rng.integers(3, 9))
    for _ in range(180):
        if len(rectangles) >= target:
            break
        path = paths[int(rng.integers(count))]
        t = int(rng.integers(2, len(path) - 1))
        x, y = path[t]
        px, py = path[t - 1]
        d = _D.index((x - px, y - py))
        color = int(rng.choice(list(turns)))
        depth = int(rng.integers(1, 5))
        before, after = map(int, rng.integers(0, 6, size=2))
        if d == 1:
            a, b, e, f = max(0, x - before), min(h, x + after + 1), y, min(w, y + depth)
        elif d == 3:
            a, b, e, f = max(0, x - before), min(h, x + after + 1), max(0, y - depth + 1), y + 1
        elif d == 2:
            a, b, e, f = x, min(h, x + depth), max(0, y - before), min(w, y + after + 1)
        else:
            a, b, e, f = max(0, x - depth + 1), x + 1, max(0, y - before), min(w, y + after + 1)
        if np.any(field[a:b, e:f]):
            continue
        # Equal-colored components must remain separate solid rectangles.
        if np.any(field[max(0, a - 1):min(h, b + 1), max(0, e - 1):min(w, f + 1)] == color):
            continue
        trial = field.copy()
        trial[a:b, e:f] = color
        try:
            pp, ee = _routes(trial, anchors, 1, 0, 1, 2, turns)
            _need(min(map(len, pp)) >= mincap, 'too little travel space')
            _need(all(p[:2] == [(r, c + 1), (r, c + 2)] for p, (r, c) in zip(pp, anchors)), 'early turn')
        except ValueError:
            continue
        field, paths, events = trial, pp, ee
        rectangles.append((a, b, e, f, color))
    _need(np.count_nonzero(field == 0) > field.size / 2, 'crowded field')
    return field, anchors, turns, paths, events


def _length_choice(rng, paths, events, references, strong, mode):
    n, middle = len(paths), len(paths) // 2
    refs = [int(x) for x in rng.permutation(references) if x <= min(map(len, paths)) - 2]
    for ref in refs:
        if strong and not any(e[0] < ref - 1 for e in events[middle]):
            continue
        for _ in range(55 if strong else 25):
            lengths = [ref] * n
            others = [int(i) for i in rng.permutation(n) if i != middle]
            if strong:
                majority_short = bool(rng.integers(2))
                for i in others:
                    lengths[i] = int(rng.integers(2, ref)) if majority_short else int(rng.integers(ref + 1, len(paths[i])))
                i = others[0]
                lengths[i] = int(rng.integers(ref + 1, len(paths[i]))) if majority_short else int(rng.integers(2, ref))
            else:
                for i in others:
                    delta = 'same' if rng.random() < .2 else mode if mode in ('extend', 'trim') else str(rng.choice(['extend', 'trim']))
                    if delta == 'extend':
                        lengths[i] = int(rng.integers(2, ref))
                    elif delta == 'trim':
                        lengths[i] = int(rng.integers(ref + 1, len(paths[i])))
                if lengths == [ref] * n:
                    lengths[others[0]] = 2
            painted = {p for path, size in zip(paths, lengths) for p in path[:size]}
            # Shared routes may not obscure anyone's stopping point.
            if any(path[size] in painted for path, size in zip(paths, lengths)):
                continue
            active = [[e for e in ev if e[0] < max(ref, size) - 1] for ev, size in zip(events, lengths)]
            sides = {((e[2] - e[1]) % 4) for ev in active for e in ev}
            if not active or not any(active):
                continue
            if strong:
                extend_turn = any(size < ref and any(size <= e[0] < ref - 1 for e in ev) for size, ev in zip(lengths, events))
                trim_turn = any(size > ref and any(ref + 1 <= e[0] < size - 1 for e in ev) for size, ev in zip(lengths, events))
                if not (extend_turn and trim_turn and sides == {1, 3} and max(map(len, active)) >= 2):
                    continue
            return ref, lengths
    raise ValueError('no informative unoccluded lengths')


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _make_pair(rng, count, rotation, references, strong=False, avoid_reference=None):
    for _ in range(160):
        keys = int(rng.integers(2, 5))
        hh = int(rng.integers(3, 6))
        try:
            field, anchors, turns, paths, events = _scene(rng, count, keys, hh, strong, 14 if strong else max(14, min(references) + 2))
            ref, lengths = _length_choice(rng, paths, events, [x for x in references if x != avoid_reference], strong,
                                          str(rng.choice(['mixed', 'mixed', 'extend', 'trim'])))
            inp, out = field.copy(), field.copy()
            for path, size in zip(paths, lengths):
                for p in path[:size]:
                    inp[p] = 1
                for p in path[:ref]:
                    out[p] = 1
            _need(not np.array_equal(out, inp), 'normalization hidden by shared paint')
            inp = np.rot90(inp, rotation).copy()
            out = np.rot90(out, rotation).copy()
            _need(np.count_nonzero(inp == 0) > inp.size / 2, 'insufficient blank field')
            header = _make_header(rng, inp.shape[1], hh, turns)
            palette = rng.permutation(10).astype(np.int8)
            pair = dict(input=palette[np.vstack((header, inp))], output=palette[out])
            info = _analyze(pair['input'])
            if strong:
                _need(_is_teaching(family_features(pair['input'])), 'missing visible integrated evidence')
            _need(sorted(info['lengths']) == sorted(lengths) and info['reference'] == ref, 'obscured lengths')
            return pair, info
        except ValueError:
            continue
    raise RuntimeError('bounded scene search exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    base = int(rng.integers(4))
    # All tests extrapolate the budget; orientation is either familiar or a
    # quarter-turn transfer from the two opposed teaching launch headings.
    transfer = bool(rng.integers(2))
    puzzle, seen, train_refs = dict(train=[], test=[]), set(), []
    for split, count_pairs in (('train', nt), ('test', nq)):
        for i in range(count_pairs):
            strong = split == 'train' and i == 0
            count = 5 if strong else int(rng.choice([3, 7] if split == 'train' and i == 1 else [3, 5, 7, 9]))
            if split == 'train':
                rotation = (base + 2 * (i % 2)) % 4
                refs = list(range(7, 12))
            else:
                rotation = (base + (1 if transfer else 0) + 2 * (i % 2)) % 4
                refs = list(range(12, 18))
            for _ in range(12):
                pair, info = _make_pair(rng, count, rotation, refs, strong,
                                         train_refs[0] if split == 'train' and i == 1 else None)
                sig = _signature(pair['input'])
                if sig not in seen:
                    break
            else:
                raise RuntimeError('could not produce a distinct input')
            seen.add(sig)
            puzzle[split].append(pair)
            if split == 'train':
                train_refs.append(info['reference'])
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _straight_output(q):
    out = q['field'].copy()
    out[out == q['payload']] = q['background']
    dr, dc = _D[q['direction']]
    for r, c in q['anchors']:
        for step in range(1, q['reference'] + 1):
            x, y = r + step * dr, c + step * dc
            if 0 <= x < out.shape[0] and 0 <= y < out.shape[1] and out[x, y] == q['background']:
                out[x, y] = q['payload']
    return out


def family_features(input_grid):
    q = _analyze(input_grid)
    ref, ls, evs = q['reference'], q['lengths'], q['events']
    active = [[e for e in ev if e[0] < max(ref, size) - 1] for ev, size in zip(evs, ls)]
    coords = np.array(q['coords'])
    distances = abs(2 * coords - coords[0] - coords[-1])
    near = np.flatnonzero(distances == distances.min())
    geom = (len(near) == 1 and int(near[0]) != len(ls) // 2 and ls[int(near[0])] != ref)
    shared_blocker = any(sum(any(a <= e[4] < b and c <= e[5] < d for e in ev) for ev in active) > 1
                         for a, b, c, d, color in q['rectangles'])
    used = [set(p[:max(ref, size)]) for p, size in zip(q['paths'], ls)]
    shared = sum(len(used[i] & used[j]) for i in range(len(ls)) for j in range(i))
    return dict(count=len(ls), reference=ref, lengths=tuple(ls), direction=q['direction'],
                shape=tuple(q['field'].shape), background=q['background'], payload=q['payload'],
                gaps=tuple(map(int, np.diff(coords))), blockers=len(q['rectangles']),
                keys=len(q['turns']), header_height=q['crop'],
                max_turns=max(map(len, active)), center_turns=sum(e[0] < ref - 1 for e in evs[len(ls) // 2]),
                extend_turn=any(size < ref and any(size <= e[0] < ref - 1 for e in ev) for size, ev in zip(ls, evs)),
                trim_turn=any(size > ref and any(ref + 1 <= e[0] < size - 1 for e in ev) for size, ev in zip(ls, evs)),
                sides=tuple(sorted({q['turns'][e[3]] for ev in active for e in ev})),
                nonmedian=sorted(ls)[len(ls) // 2] != ref, geometric_witness=bool(geom),
                shared_cells=shared, shared_blocker=shared_blocker,
                straight_diff=not np.array_equal(_straight_output(q), _paint(q, [ref] * len(ls))),
                median_diff=not np.array_equal(_paint(q, [sorted(ls)[len(ls) // 2]] * len(ls)),
                                               _paint(q, [ref] * len(ls))),
                repeat_blocker_color=len(q['rectangles']) > len({r[4] for r in q['rectangles']}),
                unchanged_noncentral=sum(size == ref for i, size in enumerate(ls) if i != len(ls) // 2),
                extension=max(ref - size for size in ls), truncation=max(size - ref for size in ls))


def _is_teaching(f):
    return (all(f[k] for k in ('nonmedian', 'geometric_witness', 'center_turns',
                              'extend_turn', 'trim_turn', 'straight_diff', 'median_diff', 'shared_cells'))
            and f['max_turns'] >= 2 and f['sides'] == (-1, 1))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            sig = _signature(inp)
            assert sig not in seen, 'duplicate input'
            seen.add(sig)
            q = _analyze(inp)
            ref, bg, payload = q['reference'], q['background'], q['payload']
            assert out.shape == q['field'].shape, 'bad crop'
            expected = _paint(q, [ref] * len(q['paths']))
            assert np.array_equal(expected, out), 'incorrect finite route prefixes'
            # Blockers, launch markers and empty regions cannot be recolored.
            fixed = (q['field'] != bg) & (q['field'] != payload)
            assert np.array_equal(out[fixed], q['field'][fixed])
            assert np.all(np.isin(out[(q['field'] == bg) | (q['field'] == payload)], [bg, payload]))
            assert not np.array_equal(out, q['field']), 'no normalization change'
            # Route prefixes preserved up to min(old,new); excess tails vanish
            # unless another normalized path also traverses those cells.
            union = {p for path in q['paths'] for p in path[:ref]}
            for path, size in zip(q['paths'], q['lengths']):
                assert all(out[p] == payload for p in path[:min(size, ref)])
                assert all(out[p] == (payload if p in union else bg) for p in path[ref:size])
            features[split].append(family_features(inp))
    train = features['train']
    assert len({f['reference'] for f in train}) >= 2
    assert len({f['count'] for f in train}) >= 2
    assert any(_is_teaching(f) for f in train), 'missing integrated teaching witness'
    assert min(f['reference'] for f in features['test']) > max(f['reference'] for f in train), 'missing budget shift'


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _manual_fixture():
    """Small explicit cell-count fixture; expected cells do not use the tracer."""
    field = np.zeros((15, 22), dtype=np.int8)
    field[3, 1] = field[7, 1] = field[11, 1] = 2
    field[7, 7] = 3  # east -> north, at (7,6)
    field[2, 6] = 4  # north -> east, at (3,6)
    field[11, 12] = 4  # east -> south, at (11,11)
    inp, out = field.copy(), field.copy()
    inp[3, 2:5] = 1  # three cells
    inp[7, 2:7] = 1
    inp[4:7, 6] = 1  # central: five east + three north = eight
    inp[11, 2:12] = 1
    inp[12, 11] = 1  # lower: ten east + one south = eleven
    out[3, 2:10] = 1
    out[7, 2:7] = 1
    out[4:7, 6] = 1
    out[11, 2:10] = 1
    return inp, out


def self_test():
    """Bounded 42-seed structural, semantic, equivariance and ablation checks."""
    signatures, rows = set(), []
    rng = np.random.default_rng(4313)
    fixture_input, fixture_output = _manual_fixture()
    for rotation in range(4):
        fi, fo = np.rot90(fixture_input, rotation), np.rot90(fixture_output, rotation)
        header = _make_header(rng, fi.shape[1], 3, {3: -1, 4: 1})
        full = np.vstack((header, fi))
        assert np.array_equal(solve(full.tolist()), fo), 'explicit corner-count fixture'
        q = _analyze(full)
        assert q['lengths'] == ([3, 8, 11] if rotation in (0, 1) else [11, 8, 3])
        assert q['reference'] == 8
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        if seed < 3:
            assert _digest(puzzle) == _digest(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                q = _analyze(pair['input'])
                rows.append(family_features(pair['input']))
                assert np.array_equal(solve(pair['input']), pair['output'])
                # Palette roles must be local to the current input.
                palette = rng.permutation(10).astype(np.int8)
                assert np.array_equal(solve(palette[pair['input']]), palette[pair['output']])
        witness = next(p for p in puzzle['train'] if _is_teaching(family_features(p['input'])))
        q = _analyze(witness['input'])
        # Omitting A: leaving the original lengths (even though correctly routed).
        assert not np.array_equal(q['field'], witness['output'])
        median = sorted(q['lengths'])[len(q['lengths']) // 2]
        assert not np.array_equal(_paint(q, [median] * len(q['paths'])), witness['output'])
        # Omitting B: equal straight lengths ignore visibly encoded detours.
        straight = _straight_output(q)
        assert not np.array_equal(straight, witness['output'])
        # Changing the central visible endpoint alters the common budget. The
        # change is used only when shared paint does not obscure the new stop.
        mid = len(q['paths']) // 2
        p = q['paths'][mid][q['reference'] - 1]
        altered = witness['input'].copy()
        altered[q['crop'] + p[0], p[1]] = q['background']
        try:
            changed = _analyze(altered)
        except ValueError:
            pass
        else:
            assert changed['reference'] == q['reference'] - 1
            assert not np.array_equal(solve(altered), witness['output'])
    assert len(signatures) == 42
    for field, minimum in (('shape', 35), ('reference', 10), ('lengths', 120), ('gaps', 55), ('blockers', 5)):
        assert len({r[field] for r in rows}) >= minimum, field
    assert {r['count'] for r in rows} == {3, 5, 7, 9}
    assert {r['direction'] for r in rows} == {0, 1, 2, 3}
    assert {r['background'] for r in rows} == set(range(10))
    assert {r['keys'] for r in rows} == {2, 3, 4}
    assert {r['header_height'] for r in rows} == {3, 4, 5}
    assert any(r['shared_blocker'] for r in rows) and any(r['shared_cells'] for r in rows)
    assert any(r['max_turns'] >= 3 for r in rows)
    assert any(r['unchanged_noncentral'] for r in rows)
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        generate_puzzle(2048, nt, nq)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # A damaged marker must be caught without trusting solve agreement alone.
    broken = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    q = _analyze(broken['test'][0]['input'])
    r, c = q['anchors'][0]
    broken['test'][0]['output'][r, c] = q['background']
    try:
        validate_generated_puzzle(broken)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError('validator accepted a damaged launch marker')
    return dict(seeds=42, pairs=len(rows), distinct_puzzles=len(signatures),
                field_shapes=len({r['shape'] for r in rows}),
                route_length_vectors=len({r['lengths'] for r in rows}),
                gap_vectors=len({r['gaps'] for r in rows}),
                reference_lengths=sorted({r['reference'] for r in rows}),
                counts=sorted({r['count'] for r in rows}),
                max_turns=max(r['max_turns'] for r in rows),
                shared_blocker_pairs=sum(r['shared_blocker'] for r in rows),
                shared_path_pairs=sum(r['shared_cells'] > 0 for r in rows))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
