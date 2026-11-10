"""Latent pinwheel docking: shape fitting chooses and decorates a winding.

Only NumPy and the standard library are used. No parent module or fixture is
loaded. See task_spec.md for the family, including its fixed recurrence.
"""
from __future__ import annotations

from collections import Counter
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('grids must be integer arrays with sides 1..30 and colors 0..9')
    return a.astype(np.int8, copy=True)


def _components(cells):
    left = set(cells)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        todo, part = [start], {start}
        while todo:
            y, x = todo.pop()
            for dy, dx in DIRS:
                p = y + dy, x + dx
                if p in left:
                    left.remove(p)
                    part.add(p)
                    todo.append(p)
        result.append(frozenset(part))
    return result


def _bbox(cells):
    return (min(y for y, x in cells), max(y for y, x in cells),
            min(x for y, x in cells), max(x for y, x in cells))


def _rectangle(box):
    y0, y1, x0, x1 = box
    return frozenset((y, x) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1))


def _halo(cells):
    return set(cells) | {(y + dy, x + dx) for y, x in cells for dy, dx in DIRS}


def _shift(cells, dy, dx):
    return frozenset((y + dy, x + dx) for y, x in cells)


def _transform(regions, op):
    # 0..3 = clockwise quarter turns. 4..7 = mirror columns, then turn.
    raw = []
    for region in regions:
        part = []
        for y, x in region:
            if op >= 4:
                x = -x
            for _ in range(op % 4):
                y, x = x, -y
            part.append((y, x))
        raw.append(part)
    ymin = min(y for part in raw for y, x in part)
    xmin = min(x for part in raw for y, x in part)
    return tuple(frozenset((y - ymin, x - xmin) for y, x in part) for part in raw)


def _signature(cells):
    y, _, x, _ = _bbox(cells)
    return tuple(sorted((r - y, c - x) for r, c in cells))


def _canonical(cells):
    return min(_signature(_transform((cells,), op)[0]) for op in range(8)) if cells else ()


def _orbit(cells):
    return len({_signature(_transform((cells,), op)[0]) for op in range(8)})


def _seed_cells(length, center):
    y, x = center
    return frozenset((y + d * dy, x + d * dx)
                     for dy, dx in DIRS for d in range(1, length + 1))


def _winding(length, shape, center, hand):
    """Geometric continuation on the plane; segment clipping permits re-entry."""
    radius = max(2, length)
    h, w = shape
    cy, cx = center
    result = np.zeros(shape, dtype=bool)

    def paint(y0, x0, y1, x1):
        y0, y1, x0, x1 = y0 + cy, y1 + cy, x0 + cx, x1 + cx
        if y0 == y1 and 0 <= y0 < h:
            lo, hi = max(0, min(x0, x1)), min(w - 1, max(x0, x1))
            if lo <= hi:
                result[y0, lo:hi + 1] = True
        elif x0 == x1 and 0 <= x0 < w:
            lo, hi = max(0, min(y0, y1)), min(h - 1, max(y0, y1))
            if lo <= hi:
                result[lo:hi + 1, x0] = True

    distance = max(cy, cx, h - 1 - cy, w - 1 - cx)
    for dy, dx in DIRS:
        y, x = radius * dy, radius * dx
        paint(dy, dx, y, x)
        dy, dx = hand * dx, -hand * dy
        for k in range(4 * (distance // 8 + 2)):
            for vy, vx, n in ((dy, dx, 2 + 4 * k),
                               (hand * dx, -hand * dy, radius - 2),
                               (dy, dx, radius)):
                ny, nx = y + vy * n, x + vx * n
                paint(y, x, ny, nx)
                y, x = ny, nx
            dy, dx = hand * dx, -hand * dy
    return result


def _interval_winding(length, shape, center, hand):
    """Closed-form intervals, used to check the walking calculation."""
    radius = max(2, length)
    yy, xx = np.indices(shape)
    yy, xx = yy - center[0], hand * (xx - center[1])
    mask = np.zeros(shape, dtype=bool)
    bound = max(int(np.abs(yy).max()), int(np.abs(xx).max()))
    for _ in range(4):
        mask |= (xx == 0) & (-radius <= yy) & (yy <= -1)
        for q in range(bound // 8 + 3):
            starts = ((-radius - 8*q, -8*q), (-2 - 8*q, radius + 8*q + 2),
                      (radius + 8*q + 4, 8*q + 4), (8*q + 6, -radius - 8*q - 6))
            for stage, ((y, x), (dy, dx)) in enumerate(zip(
                    starts, ((0, 1), (1, 0), (0, -1), (-1, 0)))):
                along = (yy - y) * dy + (xx - x) * dx
                across = (yy - y) * dx - (xx - x) * dy
                n = 16*q + 4*stage + 2
                mask |= (across == 0) & (0 <= along) & (along <= n)
                mask |= (along == n) & (0 <= across) & (across <= radius - 2)
                mask |= (across == radius - 2) & (n <= along) & (along <= n + radius)
        yy, xx = xx, -yy
    return mask


def _inspect(a):
    values, counts = np.unique(a, return_counts=True)
    order = np.argsort(counts)
    if len(values) < 5 or counts[order[-1]] <= counts[order[-2]]:
        raise ValueError('background must be the unique most frequent input color')
    bg = int(values[order[-1]])
    frames, keys, spoke_parts = [], [], []
    for cells in _components(map(tuple, np.argwhere(a != bg))):
        colors = tuple(sorted({int(a[p]) for p in cells}))
        if len(colors) == 2:
            regions = tuple(frozenset(p for p in cells if a[p] == c) for c in colors)
            if any(len(_components(s)) != 1 for s in regions):
                raise ValueError('both key regions must be connected')
            keys.append({'regions': regions, 'colors': colors, 'cells': cells})
        elif len(colors) == 1:
            y0, y1, x0, x1 = _bbox(cells)
            rect = _rectangle((y0, y1, x0, x1))
            interior = (_rectangle((y0 + 1, y1 - 1, x0 + 1, x1 - 1))
                        if y1 - y0 >= 3 and x1 - x0 >= 3 else frozenset())
            if interior and cells == rect - interior and all(a[p] == bg for p in interior):
                frames.append({'interior': interior, 'box': (y0 + 1, y1 - 1, x0 + 1, x1 - 1),
                               'cells': cells, 'color': colors[0]})
            else:
                spoke_parts.append((cells, colors[0]))
        else:
            raise ValueError('objects must be monochrome markers or two-colored keys')
    if not 1 <= len(frames) <= 4 or len(frames) != len(keys):
        raise ValueError('one key is required for each of 1..4 docking windows')
    if len({f['color'] for f in frames}) != 1 or len(spoke_parts) != 4:
        raise ValueError('frames share a color and four isolated spokes are required')
    foregrounds = {c for s, c in spoke_parts}
    if len(foregrounds) != 1:
        raise ValueError('all spokes must have one color')
    fg = foregrounds.pop()
    frame_color = frames[0]['color']
    if fg == frame_color or any(set(k['colors']) & {bg, fg, frame_color} for k in keys):
        raise ValueError('background, seed, frame, and key colors have distinct local roles')
    seed = frozenset(p for s, c in spoke_parts for p in s)
    if len(seed) % 4:
        raise ValueError('unequal spokes')
    length = len(seed) // 4
    y0, y1, x0, x1 = _bbox(seed)
    center = ((y0 + y1) // 2, (x0 + x1) // 2)
    if not 1 <= length <= 6 or seed != _seed_cells(length, center) or a[center] != bg:
        raise ValueError('a fully visible equal-spoke hollow cross is required')
    frames.sort(key=lambda f: f['box'])
    keys.sort(key=lambda k: min(k['cells']))
    return {'bg': bg, 'fg': fg, 'frame_color': frame_color, 'length': length,
            'center': center, 'seed': seed, 'frames': frames, 'keys': keys}


def _candidates(info, mask, shape):
    frames, keys = info['frames'], info['keys']
    interiors = frozenset(p for f in frames for p in f['interior'])
    candidates = [[] for k in keys]
    for j, frame in enumerate(frames):
        slot = frozenset(p for p in frame['interior'] if mask[p])
        if not slot or len(_components(slot)) != 1:
            return None
        sy, _, sx, _ = _bbox(slot)
        for i, key in enumerate(keys):
            unique = set()
            for op in range(8):
                regions = _transform(key['regions'], op)
                for pi in (0, 1):
                    if len(regions[pi]) != len(slot):
                        continue
                    py, _, px, _ = _bbox(regions[pi])
                    absolute = tuple(_shift(s, sy - py, sx - px) for s in regions)
                    if absolute[pi] != slot:
                        continue
                    handle = absolute[1 - pi]
                    if (handle & interiors or info['center'] in handle or any(
                            not (0 <= r < shape[0] and 0 <= c < shape[1]) or mask[r, c]
                            for r, c in handle)):
                        continue
                    colors = key['colors'][::-1] if op >= 4 else key['colors']
                    placed = tuple(sorted((p, int(c)) for s, c in zip(absolute, colors) for p in s))
                    sig = j, placed
                    if sig not in unique:
                        unique.add(sig)
                        candidates[i].append({'frame': j, 'key': i, 'op': op, 'plug_index': pi,
                                              'slot': slot, 'handle': handle,
                                              'full': absolute[0] | absolute[1], 'placed': placed})
    return candidates


def _reconstruct(a, swap_policy=(0, 1)):
    info = _inspect(a)
    solutions = {}
    for hand in (1, -1):
        mask = _winding(info['length'], a.shape, info['center'], hand)
        choices = _candidates(info, mask, a.shape)
        if choices is None or any(not x for x in choices):
            continue
        order = sorted(range(len(choices)), key=lambda i: len(choices[i]))

        def search(n, used, occupied, moves):
            if len(solutions) >= 2:
                return
            if n == len(order):
                out = np.where(mask, info['fg'], info['bg']).astype(np.int8)
                for move in moves:
                    key = info['keys'][move['key']]
                    # Candidate placements use the intended policy. Alternative
                    # policies are only used by explicit shortcut checks.
                    flip = bool(swap_policy[int(move['op'] >= 4)]) != bool(move['op'] >= 4)
                    exchange = dict(zip(key['colors'], key['colors'][::-1]))
                    for p, color in move['placed']:
                        out[p] = exchange[color] if flip else color
                sig = out.tobytes()
                solutions.setdefault(sig, (out, hand, tuple(moves), info, mask))
                return
            for move in choices[order[n]]:
                if move['frame'] in used or move['full'] & occupied:
                    continue
                search(n + 1, used | {move['frame']}, occupied | move['full'], moves + [move])
        search(0, set(), frozenset(), [])
    if len(solutions) != 1:
        raise ValueError('docking must determine exactly one complete colored winding')
    return next(iter(solutions.values()))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover winding handedness and every rigid placement from the input."""
    return _reconstruct(_grid(input_grid))[0]


def _grow_handle(rng, slot, interior, mask, forbidden, center, size):
    h, w = mask.shape
    y0, y1, x0, x1 = _bbox(interior)

    def allowed(p):
        y, x = p
        return (0 <= y < h and 0 <= x < w and not mask[p]
                and p not in interior and p not in forbidden and p != center
                and y0 - 5 <= y <= y1 + 5 and x0 - 5 <= x <= x1 + 5)

    starts = sorted({(y + dy, x + dx) for y, x in slot for dy, dx in DIRS
                     if allowed((y + dy, x + dx))})
    rng.shuffle(starts)
    for start in starts:
        handle = {start}
        for _ in range(size - 1):
            frontier = sorted({(y + dy, x + dx) for y, x in handle for dy, dx in DIRS
                               if (y + dy, x + dx) not in handle and allowed((y + dy, x + dx))})
            if not frontier:
                break
            handle.add(frontier[int(rng.integers(len(frontier)))])
        if len(handle) == size:
            return frozenset(handle)
    return None


def _choose_windows(rng, length, shape, center, hand, count, equal_witness):
    mask = _winding(length, shape, center, hand)
    other = _winding(length, shape, center, -hand)
    seed = _seed_cells(length, center) | {center}
    blocked = _halo(seed)
    interiors, final_cells, canonical = set(), set(), set()
    systems = []
    h, w = shape
    for i in range(count):
        for attempt in range(6000):
            bh, bw = int(rng.integers(2, 8)), int(rng.integers(2, 11))
            y, x = int(rng.integers(1, h - bh)), int(rng.integers(1, w - bw))
            box = (y, y + bh - 1, x, x + bw - 1)
            interior = _rectangle(box)
            whole = _rectangle((y - 1, y + bh, x - 1, x + bw))
            if whole & blocked or whole & final_cells:
                continue
            coords = np.argwhere(mask[y:y + bh, x:x + bw])
            if not 4 <= len(coords) <= 20:
                continue
            slot = frozenset((y + int(r), x + int(c)) for r, c in coords)
            if len(_components(slot)) != 1 or _orbit(slot) != 8:
                continue
            canon = _canonical(slot)
            if canon in canonical:
                continue
            opposite = frozenset(p for p in interior if other[p])
            if i == 0:
                if _canonical(opposite) == canon:
                    continue
                if equal_witness and len(slot) != len(opposite):
                    continue
            # A sizable fraction of scenes require exact shape matching
            # between two same-area, noncongruent plugs.
            if i == 1 and equal_witness and length == 3 and len(slot) != len(systems[0]['slot']):
                continue
            relation = int(rng.integers(3))
            size = (int(rng.integers(2, len(slot))) if relation == 0 else len(slot)
                    if relation == 1 else len(slot) + int(rng.integers(1, 5)))
            handle = _grow_handle(rng, slot, interior, mask, interiors | final_cells,
                                  center, min(size, 22))
            if handle is None or (i == 0 and _canonical(handle) == _canonical(opposite)):
                continue
            systems.append({'box': box, 'interior': interior, 'whole': whole,
                            'ring': whole - interior, 'slot': slot, 'handle': handle})
            interiors.update(interior)
            final_cells.update(slot | handle)
            blocked.update(_halo(whole))
            canonical.add(canon)
            break
        else:
            raise ValueError('could not sample compatible docking windows')
    return systems, mask


def _build_pair(rng, length, count, hand, witness, stable_palette):
    for attempt in range(90):
        shape = (int(rng.integers(max(19, 2*length + 9), 31)),
                 int(rng.integers(max(20, 2*length + 9), 31)))
        # Canvases and positions are independent of spoke length except fit.
        margin = length + 2
        center = tuple(int(rng.integers(margin, side - margin)) for side in shape)
        if rng.random() < .25:
            center = tuple((side - 1) // 2 for side in shape)
        palette = stable_palette if stable_palette is not None else tuple(int(c) for c in rng.permutation(10))
        bg, fg, marker = palette[:3]
        try:
            systems, mask = _choose_windows(rng, length, shape, center, hand, count, witness)
            a = np.full(shape, bg, dtype=np.int8)
            out = np.where(mask, fg, bg).astype(np.int8)
            cross = _seed_cells(length, center)
            for p in cross:
                a[p] = fg
            occupied = _halo(cross)
            for system in systems:
                for p in system['ring']:
                    a[p] = marker
                occupied.update(_halo(system['whole']))
            keys = []
            for i, system in enumerate(systems):
                parity = i % 2 if witness else int(rng.integers(2))
                op = int(rng.integers(4)) + 4 * parity
                inverse = (-op) % 4 if op < 4 else op
                source = _transform((system['slot'], system['handle']), inverse)
                colors = tuple(int(c) for c in rng.choice(palette[3:], 2, replace=False))
                final_colors = colors[::-1] if parity else colors
                for region, color in zip((system['slot'], system['handle']), final_colors):
                    for p in region:
                        out[p] = color
                keys.append((source, colors))
            rng.shuffle(keys)
            for source, colors in keys:
                cells = source[0] | source[1]
                _, ymax, _, xmax = _bbox(cells)
                options = [(y, x) for y in range(shape[0] - ymax) for x in range(shape[1] - xmax)]
                rng.shuffle(options)
                if rng.random() < .4:
                    options.sort(key=lambda p: not (p[0] in (0, shape[0] - ymax - 1)
                                                   or p[1] in (0, shape[1] - xmax - 1)))
                for y, x in options:
                    dest = _shift(cells, y, x)
                    if dest & occupied:
                        continue
                    for region, color in zip(source, colors):
                        for p in _shift(region, y, x):
                            a[p] = color
                    occupied.update(_halo(dest))
                    break
                else:
                    raise ValueError('source keys do not pack')
            solved, actual_hand, moves, info, _ = _reconstruct(a)
            if actual_hand != hand or not np.array_equal(solved, out):
                continue
            if witness and {int(m['op'] >= 4) for m in moves} != {0, 1}:
                continue
            return {'input': a, 'output': out}
        except ValueError:
            continue
    raise ValueError('bounded docking construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError('counts require 2..10 train, 1..4 test, and train > test')
    rng = np.random.default_rng(seed)
    stable = tuple(int(c) for c in rng.permutation(10)) if rng.random() < .25 else None
    train_lengths = [2, 3] + [int(rng.integers(1, 5)) for _ in range(num_train - 2)]
    extrapolate = bool(rng.integers(2))
    test_lengths = ([max(train_lengths) + 1] * num_test if extrapolate else
                    [int(rng.choice(sorted(set(train_lengths)))) for _ in range(num_test)])
    train_hands = [1, -1] + [int(rng.choice((-1, 1))) for _ in range(num_train - 2)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, lengths in (('train', train_lengths), ('test', test_lengths)):
        for i, length in enumerate(lengths):
            witness = split == 'train' and i < 2
            count = 2 if witness else int(rng.integers(1, 3)) if split == 'train' else int(rng.integers(3, 5))
            hand = train_hands[i] if split == 'train' else int(rng.choice((-1, 1)))
            for _ in range(12):
                pair = _build_pair(rng, length, count, hand, witness, stable)
                sig = _array_key(pair['input'])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError('distinct-input construction exhausted')
    rng.shuffle(puzzle['train'])
    return puzzle


def _array_key(a):
    a = np.ascontiguousarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _features(a):
    out, hand, moves, info, mask = _reconstruct(a)
    opposite = _winding(info['length'], a.shape, info['center'], -hand)
    return {'hand': hand, 'length': info['length'], 'count': len(moves), 'shape': a.shape,
            'center': info['center'], 'palette': (info['bg'], info['fg'], info['frame_color']),
            'operations': {m['op'] for m in moves},
            'areas': [len(m['slot']) for m in moves],
            'relations': {int(np.sign(len(m['slot']) - len(m['handle']))) for m in moves},
            'slots': {_canonical(m['slot']) for m in moves},
            'handles': {_canonical(m['handle']) for m in moves},
            'equal_hand_witness': any(len(m['slot']) == sum(opposite[p] for p in info['frames'][m['frame']]['interior'])
                                     and _canonical(m['slot']) != _canonical(frozenset(
                                         p for p in info['frames'][m['frame']]['interior'] if opposite[p]))
                                     for m in moves),
            'equal_area_keys': len({len(m['slot']) for m in moves}) < len(moves)}


def _unguided_alternative(info, moves, shape):
    """Find a different geometric docking if the latent track is ignored.

    Other solved keys stay fixed. A plug need only fit inside its blank
    window; its attached region must be outside all windows and the seed.
    This is an ablation witness, never part of solving or generation.
    """
    interiors = frozenset(p for f in info['frames'] for p in f['interior'])
    for move in moves:
        key = info['keys'][move['key']]
        box = info['frames'][move['frame']]['box']
        forbidden = info['seed'] | {info['center']} | frozenset(
            p for m in moves if m is not move for p in m['full'])
        for op in range(8):
            regions = _transform(key['regions'], op)
            for pi in (0, 1):
                y0, y1, x0, x1 = _bbox(regions[pi])
                for dy in range(box[0] - y0, box[1] - y1 + 1):
                    for dx in range(box[2] - x0, box[3] - x1 + 1):
                        parts = tuple(_shift(s, dy, dx) for s in regions)
                        full = parts[0] | parts[1]
                        if full == move['full'] or full & forbidden:
                            continue
                        handle = parts[1 - pi]
                        if handle & interiors or any(not (0 <= r < shape[0] and 0 <= c < shape[1])
                                                     for r, c in handle):
                            continue
                        return True
    return False


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train_features, test_features = set(), [], []
    unguided_witness = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and _array_key(a) not in seen
            seen.add(_array_key(a))
            expected, hand, moves, info, mask = _reconstruct(a)
            assert np.array_equal(expected, b), 'wrong or ambiguous colored docking output'
            assert np.array_equal(mask, _interval_winding(info['length'], a.shape, info['center'], hand))
            assert b[info['center']] == info['bg']
            assert all(b[p] == info['fg'] for p in info['seed'])
            assert info['frame_color'] not in np.unique(b), 'window frames must disappear'
            stamp = {p for m in moves for p in m['full']}
            untouched = np.ones(a.shape, bool)
            for p in stamp:
                untouched[p] = False
            assert np.array_equal(b[untouched], np.where(mask, info['fg'], info['bg'])[untouched])
            for move in moves:
                assert move['slot'] == frozenset(p for p in info['frames'][move['frame']]['interior'] if mask[p])
                assert all(not mask[p] for p in move['handle'])
                assert len(_components(move['handle'])) == len(_components(move['slot'])) == 1
                assert len(move['full']) == len(info['keys'][move['key']]['cells'])
            # Omitting A yields the undecorated labyrinth and is always wrong.
            assert not np.array_equal(b, np.where(mask, info['fg'], info['bg']))
            f = _features(a)
            (train_features if split == 'train' else test_features).append(f)
            if split == 'train':
                unguided_witness |= _unguided_alternative(info, moves, a.shape)
    assert {f['hand'] for f in train_features} == {-1, 1}
    assert {2, 3} <= {f['length'] for f in train_features}
    for hand in (-1, 1):
        assert any(f['hand'] == hand and f['equal_hand_witness']
                   and {int(op >= 4) for op in f['operations']} == {0, 1} for f in train_features)
    assert any(f['equal_area_keys'] for f in train_features)
    assert unguided_witness, 'training must show that window geometry alone is insufficient'
    assert all(f['count'] <= 2 for f in train_features)
    assert all(3 <= f['count'] <= 4 for f in test_features)
    assert max(f['length'] for f in test_features) <= max(f['length'] for f in train_features) + 1


def self_test():
    """Bounded semantic, mutation, equivariance, and 42-seed diversity checks."""
    for length in range(1, 7):
        for hand in (-1, 1):
            for shape, center in (((30, 30), (length, length)), ((19, 30), (10, 14)),
                                  ((30, 21), (23, 11))):
                a = _winding(length, shape, center, hand)
                assert np.array_equal(a, _interval_winding(length, shape, center, hand))
                assert np.array_equal(np.rot90(a), _winding(length, shape[::-1],
                                      (shape[1] - 1 - center[1], center[0]), hand))
                assert np.array_equal(np.fliplr(a), _winding(length, shape,
                                      (center[0], shape[1] - 1 - center[1]), -hand))
    assert np.array_equal(_winding(1, (30, 30), (15, 15), 1), _winding(2, (30, 30), (15, 15), 1))
    shapes, positions, slots, handles, palettes, operations, relations = (set() for _ in range(7))
    lengths, counts, signatures = set(), set(), set()
    totals = Counter()
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if first is None:
            first = puzzle
        signatures.add(tuple(_array_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair['input'])
                shapes.add(f['shape']); positions.add((f['shape'], f['center']))
                slots.update(f['slots']); handles.update(f['handles']); palettes.add(f['palette'])
                operations.update(f['operations']); relations.update(f['relations'])
                lengths.add(f['length']); counts.add(f['count'])
                totals['equal_area_keys'] += f['equal_area_keys']
                totals['equal_hand_witness'] += f['equal_hand_witness']
                totals['next_length_tests'] += split == 'test' and f['length'] > max(
                    _inspect(p['input'])['length'] for p in puzzle['train'])
    assert len(signatures) == 42 and len(shapes) >= 35 and len(positions) >= 100
    assert len(slots) >= 10 and len(handles) >= 40 and len(palettes) >= 30
    assert operations == set(range(8)) and relations == {-1, 0, 1}
    assert {1, 2, 3, 4} <= lengths and counts == {1, 2, 3, 4}
    assert min(totals.values()) > 0
    repeated = generate_puzzle(0)
    assert all(np.array_equal(a['input'], b['input']) and np.array_equal(a['output'], b['output'])
               for s in first for a, b in zip(first[s], repeated[s]))
    for counts_pair in ((2, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(101, *counts_pair))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Swapping no keys or all keys must differ on the mixed-parity witnesses.
    for pair in first['train']:
        _, _, moves, _, _ = _reconstruct(pair['input'])
        if {int(m['op'] >= 4) for m in moves} == {0, 1}:
            for policy in ((0, 0), (1, 1), (1, 0)):
                assert not np.array_equal(_reconstruct(pair['input'], policy)[0], pair['output'])
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]] for s in first}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('output mutation accepted')
    # A symmetric three-cell elbow admits both a rotation and a reflection
    # with different colors. This is a genuine tie, not an arbitrary choice.
    ambiguous = np.zeros((26, 26), dtype=np.int8)
    center = (13, 13)
    mask = _winding(2, ambiguous.shape, center, 1)
    interior = _rectangle((1, 2, 3, 4))
    whole = _rectangle((0, 3, 2, 5))
    slot = frozenset(p for p in interior if mask[p])
    for p in _seed_cells(2, center):
        ambiguous[p] = 1
    for p in whole - interior:
        ambiguous[p] = 2
    regions = _transform((slot, frozenset(((0, 3),))), 0)
    for region, color in zip(regions, (3, 4)):
        for p in _shift(region, 3, 0):
            ambiguous[p] = color
    try:
        solve(ambiguous)
    except ValueError as error:
        assert 'exactly one' in str(error)
    else:
        raise AssertionError('competing colored dockings were accepted')
    # Transforming both source and destination conjugates the rigid operation;
    # its parity stays unchanged, even when the whole scene is reflected.
    a = first['test'][0]['input']
    assert np.array_equal(solve(np.rot90(a)), np.rot90(first['test'][0]['output']))
    assert np.array_equal(solve(np.fliplr(a)), np.fliplr(first['test'][0]['output']))
    perm = np.array([7, 4, 9, 0, 6, 3, 8, 2, 5, 1], dtype=np.int8)
    assert np.array_equal(solve(perm[a]), perm[first['test'][0]['output']])
    return {'seeds': 42, 'shapes': len(shapes), 'positions': len(positions), 'slot_shapes': len(slots),
            'handle_shapes': len(handles), 'palettes': len(palettes), 'operations': sorted(operations),
            'size_relations': sorted(relations), 'spoke_lengths': sorted(lengths),
            'docking_counts': sorted(counts), **dict(totals)}
