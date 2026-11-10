"""Pinwheel loom: coordinate-ordered cards carried by four stepped windings.

Only NumPy and Python's standard library are used. Parsing is input-only.
Generation constructs labels from sampled cards; validation uses a separate
closed-form stage calculation. Neither calculation imports a parent generator.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer rectangles, sides 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    candidates = []
    for bottom in (False, True):
        for right in (False, True):
            g = a[::-1] if bottom else a
            g = g[:, ::-1] if right else g
            if min(g.shape) < 7:
                continue
            marker, horizontal, vertical, bg = map(int, (g[0, 0], g[0, 1], g[1, 0], g[-1, -1]))
            if len({marker, horizontal, vertical, bg}) != 4:
                continue
            if not (np.all(g[0, 1:-1] == horizontal)
                    and np.all(g[1:-1, 0] == vertical)
                    and np.all(g[-1, 1:] == bg)
                    and np.all(g[1:, -1] == bg)):
                # The fast arm's cap is on a keyed corner, not the opposite
                # diagonal corner, so it is excluded from these far edges.
                continue
            ends = (int(g[0, -1]), int(g[-1, 0]))
            if ends == (marker, vertical):
                axis = 0
            elif ends == (horizontal, marker):
                axis = 1
            else:
                continue
            df = ((-1 if bottom else 1), 0) if axis else (0, (-1 if right else 1))
            ds = (0, (-1 if right else 1)) if axis else ((-1 if bottom else 1), 0)
            candidates.append(dict(axis=axis, bottom=bottom, right=right,
                                   marker=marker, bg=bg, df=df, ds=ds,
                                   fast=vertical if axis else horizontal,
                                   slow=horizontal if axis else vertical))
    if len(candidates) != 1:
        raise ValueError("one L key with exactly one marker-colored fast cap is required")
    return candidates[0]


def _boxes(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    boxes = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, points = [start], [start]
        while todo:
            r, c = todo.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
                    points.append(p)
        r0, c0 = np.min(points, axis=0)
        r1, c1 = np.max(points, axis=0) + 1
        if len(points) != (r1-r0)*(c1-c0):
            raise ValueError("every pattern card must be a full rectangle")
        boxes.append(tuple(map(int, (r0, r1, c0, c1))))
    return boxes


def _parse(input_grid):
    a = _grid(input_grid)
    d = _key(a)
    interior = a[1:-1, 1:-1]
    seed = interior == d['marker']
    points = np.argwhere(seed)
    length = len(points) // 4
    if not len(points) or len(points) % 4 or not 1 <= length <= 6:
        raise ValueError("one equal-spoke hollow cross of length 1..6 is required")
    rows = np.flatnonzero(seed.sum(axis=1) == 2*length)
    cols = np.flatnonzero(seed.sum(axis=0) == 2*length)
    if len(rows) != 1 or len(cols) != 1:
        raise ValueError("cross center is ambiguous")
    cy, cx = int(rows[0])+1, int(cols[0])+1
    expected = np.zeros(a.shape, dtype=bool)
    if min(cy-length, cx-length, a.shape[0]-1-cy-length, a.shape[1]-1-cx-length) < 1:
        raise ValueError("seed spokes must be fully interior")
    for z in range(1, length+1):
        expected[cy-z, cx] = expected[cy+z, cx] = True
        expected[cy, cx-z] = expected[cy, cx+z] = True
    if not np.array_equal(expected[1:-1, 1:-1], seed):
        raise ValueError("extra marker pixels or malformed cross")
    mask = (a != d['bg']) & (a != d['marker'])
    mask[[0, -1], :] = False
    mask[:, [0, -1]] = False
    boxes = _boxes(mask)
    if not 1 <= len(boxes) <= 15:
        raise ValueError("one to fifteen cards are required")
    lanes = {}
    for box in boxes:
        r0, r1, c0, c1 = box
        if (c1-c0 if d['axis'] else r1-r0) != 2:
            raise ValueError("card thickness must be two along the slow axis")
        width = r1-r0 if d['axis'] else c1-c0
        if width < 2:
            raise ValueError("card reading length must be at least two")
        # A clear cell separates every card from the cross's actual strokes.
        if np.any(expected[max(0, r0-1):r1+1, max(0, c0-1):c1+1]):
            raise ValueError("cards must be separated from seed strokes")
        v = c0 if d['axis'] else r0
        lanes.setdefault(v, []).append(box)
    vcoords = sorted(lanes)
    if any(v2-v1 < 3 for v1, v2 in zip(vcoords, vcoords[1:])):
        raise ValueError("card lanes must be aligned and separated")
    slow_reverse = d['right'] if d['axis'] else d['bottom']
    fast_reverse = d['bottom'] if d['axis'] else d['right']
    rotation = (3 if d['bottom'] else 1) if d['axis'] else (2 if d['right'] else 0)
    blocks, ordered_boxes, counts = [], [], []
    for v in sorted(lanes, reverse=slow_reverse):
        group = sorted(lanes[v], key=lambda b: b[0] if d['axis'] else b[2], reverse=fast_reverse)
        counts.append(len(group))
        for r0, r1, c0, c1 in group:
            blocks.append(np.rot90(a[r0:r1, c0:c1], rotation).copy())
            ordered_boxes.append((r0, r1, c0, c1))
    if len({b.shape[1] % 2 for b in blocks}) != 1:
        raise ValueError("all card lengths must have the same parity for exact centering")
    d.update(shape=a.shape, center=(cy, cx), length=length, blocks=blocks,
             boxes=ordered_boxes, lane_counts=counts, rotation=rotation)
    return d


def _tape(blocks, bg, centered=True):
    width = max(b.shape[1] for b in blocks)
    words = []
    for block in blocks:
        if centered:
            word = np.full((2, width), bg, dtype=np.int8)
            offset = (width-block.shape[1]) // 2
            word[:, offset:offset+block.shape[1]] = block
        else:
            word = block
        words.append(word)
    return np.concatenate(words, axis=1)


def _steps(length, shape, center, radial, hand=1, straight=False, ordinary=False):
    """Yield new destinations and zero-based tape times; corners occur once."""
    radius = max(2, length)
    dy, dx = radial
    y, x = length*dy, length*dx
    t = 0
    for _ in range(radius-length):
        y, x = y+dy, x+dx
        yield center[0]+y, center[1]+x, t
        t += 1
    dy, dx = hand*dx, -hand*dy
    distance = max(center[0], center[1], shape[0]-1-center[0], shape[1]-1-center[1])
    if straight:
        dy, dx = radial
        segments = [(dy, dx, 2*max(shape))]
    else:
        segments = []
        for k in range(4*(distance//8+2)):
            if ordinary:
                segments.append((dy, dx, 4*k+2*radius))
            else:
                segments.extend(((dy, dx, 4*k+2), (hand*dx, -hand*dy, radius-2), (dy, dx, radius)))
            dy, dx = hand*dx, -hand*dy
    for vy, vx, count in segments:
        for _ in range(count):
            y, x = y+vy, x+vx
            yield center[0]+y, center[1]+x, t
            t += 1


def _weave(d, tape=None, *, hand=1, straight=False, stop_at_exit=False,
           ordinary=False, visible_time=False, one_use=False):
    """Procedural generation/solving calculation, with no label or seed state."""
    tape = _tape(d['blocks'], d['bg']) if tape is None else tape
    out = np.full(d['shape'], d['bg'], dtype=np.int8)
    cy, cx = d['center']
    length = d['length']
    for z in range(1, length+1):
        out[cy-z, cx] = out[cy+z, cx] = d['marker']
        out[cy, cx-z] = out[cy, cx+z] = d['marker']
    for radial in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        row = 0 if radial[0]*d['df'][0]+radial[1]*d['df'][1] else 1
        visible_count = 0
        for y, x, t in _steps(length, d['shape'], d['center'], radial, hand, straight, ordinary):
            if 0 <= y < out.shape[0] and 0 <= x < out.shape[1]:
                time = visible_count if visible_time else t
                out[y, x] = d['bg'] if one_use and time >= tape.shape[1] else tape[row, time % tape.shape[1]]
                visible_count += 1
            elif stop_at_exit:
                break
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, card, direction, scale and tape phase from this input."""
    return _weave(_parse(input_grid))


def _analytic_weave(d):
    """Check labels via closed-form stage starts and interval/time membership.

    This is an author-owned cross-check, not an independent ARC solver.
    Unlike _steps, it does not accumulate positions or walk path destinations.
    """
    tape = _tape(d['blocks'], d['bg'])
    out = np.full(d['shape'], d['bg'], dtype=np.int8)
    rr, cc = np.indices(d['shape'])
    rr, cc = rr-d['center'][0], cc-d['center'][1]
    radius, length = max(2, d['length']), d['length']
    bound = max(int(abs(rr).max()), int(abs(cc).max()))
    occupied = np.zeros(d['shape'], dtype=bool)

    def paint(mask, time, row):
        assert not np.any(occupied & mask), "four winding phases must not intersect"
        out[mask] = tape[row, np.asarray(time)[mask] % tape.shape[1]]
        occupied[mask] = True

    for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        yy, xx = -(rr*dy+cc*dx), rr*dx-cc*dy
        row = 0 if dy*d['df'][0]+dx*d['df'][1] else 1
        mask = (xx == 0) & (yy <= -1) & (yy >= -length)
        assert not np.any(occupied & mask)
        out[mask] = d['marker']
        occupied[mask] = True
        paint((xx == 0) & (yy < -length) & (yy >= -radius), -yy-length-1, row)
        for q in range(bound//8+3):
            starts = ((-radius-8*q, -8*q), (-2-8*q, radius+8*q+2),
                      (radius+8*q+4, 8*q+4), (8*q+6, -radius-8*q-6))
            for s, ((y, x), (vy, vx)) in enumerate(zip(starts, ((0, 1), (1, 0), (0, -1), (-1, 0)))):
                k = 4*q+s
                a = 4*k+2
                base = radius-length+2*k*(k+radius-1)
                along = (yy-y)*vy+(xx-x)*vx
                across = (yy-y)*vx-(xx-x)*vy
                paint((across == 0) & (along >= 1) & (along <= a), base+along-1, row)
                paint((along == a) & (across >= 1) & (across <= radius-2), base+a+across-1, row)
                paint((across == radius-2) & (along > a) & (along <= a+radius),
                      base+radius-2+along-1, row)
    assert out[d['center']] == d['bg']
    return out


def _alternative_outputs(d):
    blocks = d['blocks']
    groups, cursor = [], 0
    for n in d['lane_counts']:
        groups.append(list(range(cursor, cursor+n)))
        cursor += n
    faxis = 0 if d['axis'] else 2
    reverse = d['bottom'] if d['axis'] else d['right']
    orders = {
        'reverse_fast': [i for group in groups for i in group[::-1]],
        'reverse_lanes': [i for group in groups[::-1] for i in group],
        'fast_major': sorted(range(len(blocks)), key=lambda i: d['boxes'][i][faxis], reverse=reverse),
        'screen_order': sorted(range(len(blocks)), key=lambda i: d['boxes'][i]),
    }
    alternatives = {name: _weave(d, _tape([blocks[i] for i in order], d['bg']))
                    for name, order in orders.items()}
    alternatives.update({
        'no_centering': _weave(d, _tape(blocks, d['bg'], False)),
        'reflect_rows': _weave(d, _tape([b[::-1] for b in blocks], d['bg'])),
        'reflect_columns': _weave(d, _tape([b[:, ::-1] for b in blocks], d['bg'])),
        'straight_rays': _weave(d, straight=True),
        'counterclockwise': _weave(d, hand=-1),
        'first_exit': _weave(d, stop_at_exit=True),
        'ordinary_spirals': _weave(d, ordinary=True),
        'visible_only_time': _weave(d, visible_time=True),
        'one_use_tape': _weave(d, one_use=True),
        'deduplicate': _weave(d, _tape([b for i, b in enumerate(blocks)
                                      if all(not np.array_equal(b, prior) for prior in blocks[:i])], d['bg'])),
    })
    return alternatives


def _key_of(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _evidence(training):
    descriptors = [_parse(pair['input']) for pair in training]
    assert {d['axis'] for d in descriptors} == {0, 1}, "both source axes need demonstrations"
    assert {d['bottom'] for d in descriptors} == {False, True}
    assert {d['right'] for d in descriptors} == {False, True}
    assert {2, 3}.issubset({d['length'] for d in descriptors}), "show zero and positive inward jogs"
    assert any(max(d['lane_counts']) >= 2 and len(d['lane_counts']) >= 2 for d in descriptors)
    assert any(len({b.shape[1] for b in d['blocks']}) > 1 for d in descriptors)
    assert any(len({_key_of(b) for b in d['blocks']}) < len(d['blocks']) for d in descriptors)
    distinguished = Counter()
    for pair, d in zip(training, descriptors):
        for name, alt in _alternative_outputs(d).items():
            distinguished[name] += int(not np.array_equal(pair['output'], alt))
    assert all(distinguished.values()), f"unresolved shortcut: {dict(distinguished)}"
    return descriptors, dict(distinguished)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape
            key = _key_of(a)
            assert key not in seen, "duplicate input"
            seen.add(key)
            d = _parse(a)
            assert np.array_equal(b, _analytic_weave(d)), "incorrect tape transport or winding"
            assert np.array_equal(b, solve(a))
            # Key markers disappear; only interior cross markers are retained.
            assert np.array_equal(b[1:-1, 1:-1] == d['marker'], a[1:-1, 1:-1] == d['marker'])
            assert b[d['center']] == d['bg']
            assert not np.array_equal(a, b)
            assert set(map(int, b.flat)).issubset({d['bg'], d['marker']} | set(map(int, _tape(d['blocks'], d['bg']).flat)))
    _evidence(puzzle['train'])


def _partition(rng, n, lanes, witness):
    counts = [2 if witness else 1]*lanes
    while sum(counts) < n:
        eligible = [i for i, count in enumerate(counts) if count < 5]
        counts[int(rng.choice(eligible))] += 1
    return counts


def _make_pair(rng, config, length, roles, witness, regime, test=False):
    axis, bottom, right = config
    bg, marker, fast, slow = roles
    lo, hi = (25, 30) if test and regime == 'radius' else ((18, 24) if regime == 'radius' else (18, 30))
    h, w = map(int, rng.integers(lo, hi+1, size=2))
    if not witness and rng.random() < .3:
        h = int(rng.integers(max(2*length+5, 12), hi+1))
    safe = length+3 if witness else length+1
    cy, cx = int(rng.integers(safe, h-safe)), int(rng.integers(safe, w-safe))
    if test and regime == 'clipping':
        cy = int(rng.choice([length+1, h-length-2]))
        cx = int(rng.choice([length+1, w-length-2]))
    if witness:
        n = int(rng.integers(4, 7))
        lanes = 2 if n < 6 else int(rng.integers(2, 4))
    else:
        n = int(rng.integers(1, 9))
        lanes = int(rng.integers(max(1, (n+4)//5), min(n, 5)+1))
    counts = _partition(rng, n, lanes, witness)
    parity = int(rng.integers(2))
    base = 3 if parity else 2
    lengths = rng.choice([base, base+2, base+4], size=n).astype(int).tolist()
    duplicate = None
    if witness:
        a, b, asymmetric = map(int, rng.choice(n, size=3, replace=False))
        lengths[a] = lengths[b] = base
        lengths[asymmetric] = base+2
        duplicate = (a, b, asymmetric)
    # Fit actual row budgets; no compulsory first longest card.
    cursor = 0
    for count in counts:
        while sum(lengths[cursor:cursor+count])+count-1 > w-2:
            large = [i for i in range(cursor, cursor+count) if lengths[i] > base]
            if not large:
                return None
            lengths[int(rng.choice(large))] -= 2
        cursor += count
    if witness and len(set(lengths)) == 1:
        return None
    pool = [c for c in range(10) if c not in (bg, marker)]
    palette = rng.choice(pool, size=int(rng.integers(3 if witness else 1, 7)), replace=False)
    blocks = []
    for width in lengths:
        previous = [b for b in blocks if b.shape[1] == width]
        mode = int(rng.integers(4))
        if previous and rng.random() < .25:
            block = previous[int(rng.integers(len(previous)))].copy()
        elif mode == 0:
            block = np.full((2, width), rng.choice(palette), dtype=np.int8)
        elif mode == 1:
            block = np.repeat(rng.choice(palette, size=(2, 1)), width, axis=1).astype(np.int8)
        elif mode == 2:
            motif = rng.choice(palette, size=(2, int(rng.integers(1, min(3, width)+1))))
            block = np.tile(motif, (1, width))[:, :width].astype(np.int8)
        else:
            block = rng.choice(palette, size=(2, width)).astype(np.int8)
        blocks.append(block)
    if witness:
        a, b, i = duplicate
        blocks[b] = blocks[a].copy()
        for _ in range(32):
            blocks[i] = rng.choice(palette, size=blocks[i].shape).astype(np.int8)
            if all(not np.array_equal(blocks[i], other) for other in
                   (blocks[i][::-1], blocks[i][:, ::-1], blocks[i][::-1, ::-1])):
                break
        else:
            return None
    g = np.full((h, w), bg, dtype=np.int8)
    for z in range(1, length+1):
        g[cy-z, cx] = g[cy+z, cx] = marker
        g[cy, cx-z] = g[cy, cx+z] = marker
    blocked = np.zeros((h, w), dtype=bool)
    # Reserve only the cross and a one-cell clearance, not a target panel.
    for y, x in np.argwhere(g == marker):
        blocked[y-1:y+2, x-1:x+2] = True
    chosen_lanes = []
    placements = []
    cursor = 0
    for count in counts:
        fitted = False
        for v in rng.permutation(np.arange(1, h-2)):
            v = int(v)
            if any(abs(v-other) < 3 for other in chosen_lanes):
                continue
            widths = lengths[cursor:cursor+count]
            # Enumerate feasible starts left-to-right, then backtrack over
            # shuffled candidates. This packs cards around the actual cross.
            def fit(index, minimum, starts):
                if index == count:
                    return starts
                width = widths[index]
                remaining = sum(widths[index+1:])+len(widths[index+1:])
                choices = list(range(minimum, w-1-width-remaining+1))
                rng.shuffle(choices)
                for u in choices:
                    if not np.any(blocked[v:v+2, u:u+width]):
                        result = fit(index+1, u+width+1, starts+[u])
                        if result is not None:
                            return result
                return None
            starts = fit(0, 1, [])
            if starts is not None:
                chosen_lanes.append(v)
                placements.extend((v, u, i) for u, i in zip(starts, range(cursor, cursor+count)))
                cursor += count
                fitted = True
                break
        if not fitted:
            return None
    # Sort the randomly located lane groups to obtain the actual tape order.
    placements.sort()
    ordered = []
    reverse_thickness = (bottom == right) if axis else (bottom != right)
    for v, u, i in placements:
        block = blocks[i]
        g[v:v+2, u:u+block.shape[1]] = block[::-1] if reverse_thickness else block
        ordered.append(block)
    g[0, 0], g[0, 1:], g[1:, 0], g[0, -1] = marker, fast, slow, marker
    if axis:
        g = g.T
    if bottom:
        g = g[::-1]
    if right:
        g = g[:, ::-1]
    g = g.copy()
    # Labels use the sampled cards and the known physical seed position. The
    # input reader is not involved in constructing the output descriptor.
    center = (cx, cy) if axis else (cy, cx)
    if bottom:
        center = (g.shape[0]-1-center[0], center[1])
    if right:
        center = (center[0], g.shape[1]-1-center[1])
    df = ((-1 if bottom else 1), 0) if axis else (0, (-1 if right else 1))
    d = dict(shape=g.shape, center=center, length=length, blocks=ordered,
             bg=bg, marker=marker, df=df)
    return {'input': g, 'output': _weave(d)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    first = (int(rng.integers(2)), bool(rng.integers(2)), bool(rng.integers(2)))
    second = (1-first[0], not first[1], not first[2])
    configs = [(a, b, r) for a in (0, 1) for b in (False, True) for r in (False, True)]
    regime = str(rng.choice(['recombine', 'clipping', 'radius']))
    common_roles = tuple(map(int, rng.choice(10, size=4, replace=False)))
    varied_roles = rng.random() < .65
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            witness = split == 'train' and index < 2
            if witness:
                config = (first, second)[index]
                length = 2+index
            elif split == 'train':
                pool = (first, second) if regime == 'recombine' else configs
                config = pool[int(rng.integers(len(pool)))]
                length = int(rng.choice([1, 2, 3]))
            else:
                used_configs = {(d['axis'], d['bottom'], d['right']) for d in map(lambda p: _parse(p['input']), puzzle['train'])}
                pool = [c for c in configs if c not in used_configs] if regime == 'recombine' else sorted(used_configs)
                config = pool[int(rng.integers(len(pool)))]
                length = 4 if regime == 'radius' else int(rng.choice(sorted({_parse(p['input'])['length'] for p in puzzle['train']})))
            roles = tuple(map(int, rng.choice(10, 4, replace=False))) if varied_roles else common_roles
            for _ in range(128):
                pair = _make_pair(rng, config, length, roles, witness, regime, split == 'test')
                if pair is None or _key_of(pair['input']) in seen:
                    continue
                if witness and index == 0 and config != (0, False, False):
                    # With a horizontal top-left second demonstration, screen
                    # order is structurally identical there. The first must
                    # already expose the guide; repeated cards can otherwise
                    # hide its reversal in an accidentally symmetric tape.
                    d = _parse(pair['input'])
                    if np.array_equal(pair['output'], _alternative_outputs(d)['screen_order']):
                        continue
                if witness and index == 1:
                    try:
                        _evidence(puzzle['train']+[pair])
                    except AssertionError:
                        continue
                seen.add(_key_of(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded card packing/evidence construction failed")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple((_key_of(p['input']), _key_of(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def family_metrics(puzzles):
    """Observed geometry and mechanisms, reconstructed from input grids."""
    features = {k: set() for k in ('shapes', 'configs', 'cross_lengths', 'card_counts', 'lane_counts', 'widths', 'periods', 'palettes', 'placements')}
    totals = Counter()
    for puzzle in puzzles:
        _, evidence = _evidence(puzzle['train'])
        for name, count in evidence.items():
            totals['puzzles_distinguishing_'+name] += int(count > 0)
        for split in ('train', 'test'):
            for p in puzzle[split]:
                d = _parse(p['input'])
                widths = [b.shape[1] for b in d['blocks']]
                tape = _tape(d['blocks'], d['bg'])
                values = {'shapes': d['shape'], 'configs': (d['axis'], d['bottom'], d['right']),
                          'cross_lengths': d['length'], 'card_counts': len(widths),
                          'lane_counts': tuple(d['lane_counts']), 'widths': tuple(widths),
                          'periods': tape.shape[1], 'palettes': tuple(sorted(set(map(int, p['input'].flat)))),
                          'placements': (d['shape'], d['center'], tuple(d['boxes']))}
                for name, value in values.items():
                    features[name].add(value)
                totals[split+'_pairs'] += 1
                totals['with_padding'] += int(min(widths) < max(widths))
                totals['with_duplicates'] += int(len({_key_of(b) for b in d['blocks']}) < len(widths))
                totals['odd_widths'] += int(widths[0] % 2)
                totals['first_exit_changes_answer'] += int(not np.array_equal(p['output'], _weave(d, stop_at_exit=True)))
    return {'distinct': {k: len(v) for k, v in features.items()}, 'totals': dict(totals)}


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    metrics = family_metrics(puzzles)
    for name, minimum in (('shapes', 55), ('configs', 8), ('cross_lengths', 4), ('card_counts', 7),
                          ('lane_counts', 15), ('widths', 80), ('periods', 15), ('placements', 190)):
        assert metrics['distinct'][name] >= minimum, (name, metrics)
    for name in ('with_padding', 'with_duplicates', 'odd_widths', 'first_exit_changes_answer'):
        assert metrics['totals'][name] > 0
    # Explicit broader-family fixtures, generated by simple placement rather
    # than the normal random sampler. Squares rely solely on the visible cap.
    def fixture(widths, counts, length, config):
        axis, bottom, right = config
        rng = np.random.default_rng(800+length+sum(widths))
        blocks = [rng.choice([1, 2, 3, 4, 5, 6, 7, 9], size=(2, width)).astype(np.int8)
                  for width in widths]
        g = np.zeros((30, 30), dtype=np.int8)
        for z in range(1, length+1):
            g[15-z, 15] = g[15+z, 15] = g[15, 15-z] = g[15, 15+z] = 8
        lane_rows = [1, 25] if len(counts) == 2 else [1, 4, 25]
        reverse = (bottom == right) if axis else (bottom != right)
        cursor = 0
        for v, count in zip(lane_rows, counts):
            u = 1
            for block in blocks[cursor:cursor+count]:
                g[v:v+2, u:u+block.shape[1]] = block[::-1] if reverse else block
                u += block.shape[1]+1
            cursor += count
        g[0, 0], g[0, 1:], g[1:, 0], g[0, -1] = 8, 1, 2, 8
        if axis:
            g = g.T
        if bottom:
            g = g[::-1]
        if right:
            g = g[:, ::-1]
        d = _parse(g)
        assert all(np.array_equal(a, b) for a, b in zip(blocks, d['blocks']))
        assert np.array_equal(solve(g), _analytic_weave(d))
        return g.copy()
    for length in range(1, 7):
        for axis in (0, 1):
            for bottom in (False, True):
                for right in (False, True):
                    for widths, counts in (([26, 2, 2], [1, 2]), ([27, 3, 3], [1, 2]),
                                           ([2]*15, [5, 5, 5])):
                        fixture(widths, counts, length, (axis, bottom, right))
    # Reject ambiguous keys, damaged rectangles, and inexact padding directly
    # from input, without asking an output which interpretation was intended.
    a = fixture([26, 2, 2], [1, 2], 6, (0, False, False))
    for mutation in ('cap', 'second_cap', 'card_hole', 'mixed_parity', 'unequal_spoke'):
        bad_input = a.copy()
        if mutation == 'cap':
            bad_input[0, -1] = 1
        elif mutation == 'second_cap':
            bad_input[-1, 0] = 8
        elif mutation == 'card_hole':
            bad_input[1, 1] = 0
        elif mutation == 'mixed_parity':
            bad_input[1:3, 27] = 3
        else:
            bad_input[9, 15] = 0
        try:
            solve(bad_input)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid input accepted: '+mutation)
    # Explicit time and bend checks: L3 north tip (-3,0), east two,
    # inward south one, east three; corners do not consume twice.
    steps = list(_steps(3, (20, 20), (10, 10), (-1, 0)))
    assert steps[:6] == [(7, 11, 0), (7, 12, 1), (8, 12, 2), (8, 13, 3), (8, 14, 4), (8, 15, 5)]
    assert list(_steps(1, (20, 20), (10, 10), (-1, 0)))[:1] == [(8, 10, 0)]
    # Crop/translation must keep every later re-entry and its tape phase.
    # Rotating the whole loom rotates its answer; color permutations commute.
    for seed in (0, 7, 19):
        pair = puzzles[seed]['test'][0]
        a, b = pair['input'], pair['output']
        for k in range(4):
            assert np.array_equal(solve(np.rot90(a, k)), np.rot90(b, k))
        perm = np.random.default_rng(seed+100).permutation(10)
        assert np.array_equal(solve(perm[a]), perm[b])
        d = _parse(a)
        large = dict(d, shape=(90, 90), center=(45, 45))
        big = _weave(large)
        top, left = 45-d['center'][0], 45-d['center'][1]
        assert np.array_equal(b, big[top:top+a.shape[0], left:left+a.shape[1]])
    assert _puzzle_key(generate_puzzle(123)) == _puzzle_key(generate_puzzle(123))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    d = _parse(bad['test'][0]['input'])
    bad['test'][0]['output'][d['center']] = d['marker']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("filled center hole accepted")
    return metrics
