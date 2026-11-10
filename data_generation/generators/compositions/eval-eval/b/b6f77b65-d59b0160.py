"""Contact-completed collapse. Standard library + NumPy; no external state.

No execution traces are saved. Diagnostics are aggregate sets/counters, and
self_test streams 42 seeds instead of retaining their grid payloads.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


# A plank is (row, left, tuple_of_cell_colors). Its index is its persistent ID.
def _shift(p, row):
    return (int(row), p[1], p[2])


def _key(a):
    a = np.asarray(a)
    return (a.shape, a.astype(np.int8).tobytes())


def _grid(a):
    a = np.asarray(a)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _render(shape, bg, header, posts, planks):
    a = np.full(shape, bg, dtype=np.int8)
    a[:5] = header
    for top, col, color in posts:
        a[top:, col] = color
    for row, col, cells in planks:
        assert row >= 5 and col >= 0 and col + len(cells) <= shape[1]
        assert np.all(a[row, col:col+len(cells)] == bg), 'overlapping pieces'
        a[row, col:col+len(cells)] = cells
    return a


def _decode(a):
    """Recover all roles and complete pieces using the input alone."""
    a = _grid(a)
    h, w = a.shape
    assert h >= 8 and w >= 5
    bg = int(a[4, 0])
    assert np.all(a[4] == bg)
    options = []
    for right in (False, True):
        v = a[:, ::-1] if right else a
        frame = int(v[3, 3])
        if frame == bg or not (np.all(v[3, :4] == frame) and np.all(v[:4, 3] == frame)):
            continue
        vals = [int(c) for c in v[:3, :3].flat if c != bg]
        if len(vals) != 3 or len(set(vals)) != 3 or frame in vals:
            continue
        header = np.full((5, w), bg, dtype=np.int8)
        view = header[:, ::-1] if right else header
        view[:4, :4] = v[:4, :4]
        if np.array_equal(header, a[:5]):
            options.append((frame, frozenset(vals), right))
    assert len(options) == 1, 'unique top-corner selector required'
    frame, query, right = options[0]
    used = np.zeros(a.shape, dtype=bool)
    planks = []
    for r in range(5, h):
        c = 0
        while c < w:
            if a[r, c] == bg:
                c += 1
                continue
            end = c + 1
            while end < w and a[r, end] != bg:
                end += 1
            cells = tuple(int(x) for x in a[r, c:end])
            if frame in cells:
                assert len(cells) >= 5 and cells[0] == cells[-1] == frame
                planks.append((r, c, cells))
                used[r, c:end] = True
            c = end
    remainder = (a != bg) & ~used
    remainder[:5] = False
    posts = []
    for c in range(w):
        rows = np.flatnonzero(remainder[:, c])
        if not len(rows):
            continue
        top = int(rows[0])
        assert top >= 5 and len(rows) >= 2
        assert np.array_equal(rows, np.arange(top, h)), 'posts must reach the floor'
        color = int(a[top, c])
        assert color not in (bg, frame) and np.all(a[top:, c] == color)
        posts.append((top, c, color))
    assert planks and posts
    assert all(b[1] - a0[1] >= 2 for a0, b in zip(posts, posts[1:])), 'separate posts'
    assert np.array_equal(a, _render(a.shape, bg, a[:5], posts, planks))
    assert _settle(planks, posts, h) == planks, 'input must be mechanically stable'
    return {'grid': a, 'bg': bg, 'frame': frame, 'query': query,
            'right': right, 'posts': posts, 'planks': planks}


def _settle(planks, posts, height):
    """Lowest-first rigid drop. Overlapping horizontal intervals cannot pass."""
    result = list(planks)
    landed = []
    for i in sorted(range(len(result)), key=lambda k: (-result[k][0], result[k][1])):
        r, c, cells = result[i]
        end = c + len(cells)
        obstacles = [top for top, x, _ in posts if c <= x < end and top > r]
        obstacles += [p[0] for p in landed
                      if p[0] > r and max(c, p[1]) < min(end, p[1]+len(p[2]))]
        target = min(obstacles, default=height) - 1
        assert target >= r
        result[i] = _shift(result[i], target)
        landed.append(result[i])
    return result


def _settle_ticks(planks, posts, height):
    """Different gravity calculation used only for bounded self-checks."""
    result = list(planks)
    fixed = {(r, c) for top, c, _ in posts for r in range(top, height)}
    for _ in range(height * (len(planks)+1)):
        footprints = [{(p[0], p[1]+k) for k in range(len(p[2]))} for p in result]
        moving = {i for i, p in enumerate(result) if p[0]+1 < height}
        while True:
            blocked = set()
            for i in moving:
                target = {(r+1, c) for r, c in footprints[i]}
                if target & fixed or any(target & footprints[j] for j in range(len(result)) if j not in moving):
                    blocked.add(i)
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return result
        result = [_shift(p, p[0]+1) if i in moving else p for i, p in enumerate(result)]
    raise AssertionError('gravity did not terminate')


def _signatures(info, planks):
    a = _render(info['grid'].shape, info['bg'], info['grid'][:5], info['posts'], planks)
    signatures = []
    for r, c, cells in planks:
        own = set(cells) - {info['frame']}
        contact = set(int(x) for x in a[r+1, c:c+len(cells)]) if r+1 < len(a) else set()
        contact -= {info['bg'], info['frame']}
        signatures.append((frozenset(own), frozenset(contact)))
    return signatures


def _react(info, *, contacts=True, gravity=True, once=False, ticks=False, any_query=False):
    """Return surviving pieces and bounded, in-memory object-level diagnostics."""
    planks = list(info['planks'])
    ids = list(range(len(planks)))
    metrics = {'waves': [], 'removed': set(), 'moved': set(), 'distances': set(),
               'contact_only': False, 'extra_match': False, 'repeat_match': False,
               'exact_three': False, 'hard_missing': set(), 'count_match': False,
               'supported_by_plank_match': False}
    seen_colors = {i: set() for i in ids}
    initial = _signatures(info, planks)
    positives = []
    negatives = []
    for i, (own, under) in enumerate(initial):
        sig = own | (under if contacts else frozenset())
        nmarks = sum(x != info['frame'] for x in planks[i][2])
        item = (nmarks, len(own))
        if info['query'] <= sig:
            positives.append(item)
        else:
            negatives.append(item)
            if len(info['query'] - sig) == 1:
                metrics['hard_missing'].update(info['query'] - sig)
    metrics['count_match'] = bool(set(positives) & set(negatives))
    for _ in range(len(planks)+1):
        signatures = _signatures(info, planks)
        for ident, (own, under) in zip(ids, signatures):
            seen_colors[ident].update(own | under)
        gone = []
        for i, (own, under) in enumerate(signatures):
            signature = own | (under if contacts else frozenset())
            if (bool(info['query'] & signature) if any_query else info['query'] <= signature):
                gone.append(i)
        if not gone:
            metrics['forgetting_witness'] = any(info['query'] <= seen_colors[i] for i in ids)
            return planks, ids, metrics
        metrics['waves'].append(tuple(ids[i] for i in gone))
        for i in gone:
            own, under = signatures[i]
            metrics['removed'].add(ids[i])
            metrics['contact_only'] |= not info['query'] <= own
            metrics['extra_match'] |= bool(own - info['query'])
            marks = [x for x in planks[i][2] if x != info['frame']]
            metrics['repeat_match'] |= len(marks) > len(set(marks))
            metrics['exact_three'] |= len(marks) == 3 and set(marks) == info['query']
            p = planks[i]
            metrics['supported_by_plank_match'] |= any(
                j != i and q[0] == p[0]+1 and max(q[1], p[1]) < min(q[1]+len(q[2]), p[1]+len(p[2]))
                and not info['query'] <= own for j, q in enumerate(planks))
        remain = [i for i in range(len(planks)) if i not in gone]
        before = [planks[i] for i in remain]
        ids = [ids[i] for i in remain]
        planks = ((_settle_ticks if ticks else _settle)(before, info['posts'], len(info['grid']))
                  if gravity else before)
        for ident, p, q in zip(ids, before, planks):
            if q[0] > p[0]:
                metrics['moved'].add(ident)
                metrics['distances'].add(q[0]-p[0])
        if once:
            return planks, ids, metrics
    raise AssertionError('each reaction must remove at least one plank')


def _answer(info, **options):
    pieces, _, _ = _react(info, **options)
    return _render(info['grid'].shape, info['bg'], info['grid'][:5], info['posts'], pieces)


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _answer(_decode(input_grid))


def _colors(rng):
    palette = [int(x) for x in rng.permutation(10)]
    # Two extras are available for same-count and richer hard negatives.
    return palette[:2] + palette[2:2+int(rng.integers(5, 9))]


def _beam(rng, row, col, length, marks, frame):
    assert len(marks) <= length-2
    cells = [frame]*length
    for pos, mark in zip(rng.choice(np.arange(1, length-1), len(marks), replace=False), rng.permutation(marks)):
        cells[int(pos)] = int(mark)
    return (row, col, tuple(cells))


def _header(rng, width, colors, right):
    bg, frame = colors[:2]
    a = np.full((5, width), bg, dtype=np.int8)
    v = a[:, ::-1] if right else a
    v[3, :4] = frame
    v[:4, 3] = frame
    for p, color in zip(rng.choice(9, 3, replace=False), rng.permutation(colors[2:5])):
        v[int(p)//3, int(p)%3] = color
    return a


def _identity_scene(rng, colors):
    """One exact match, one matched negative, all three missing identities."""
    bg, frame, a, b, c, x, y = colors[:7]
    width = int(rng.integers(26, 31))
    height = int(rng.integers(15, 28))
    lengths = [5, 5, 6, 5]
    for _ in range(width-24):
        choices = [i for i in (2, 3) if lengths[i] < 9]
        if choices and rng.random() < .65:
            lengths[int(rng.choice(choices))] += 1
    # Exact and incomplete records are always matched on dimensions and counts.
    specs = [[a, b, c], [b, c, x], [a, c, x, y], [a, b, x]]
    order = [int(i) for i in rng.permutation(4)]
    slack = width - sum(lengths) - 3
    offsets = [0]*5
    for _ in range(slack):
        offsets[int(rng.integers(5))] += 1
    col = offsets[0]
    posts, planks = [], []
    for k, ident in enumerate(order):
        length = lengths[ident]
        ph = int(rng.integers(2, min(9, height-9)))
        row = height-ph-1
        pc = col+int(rng.integers(1, length-1))
        # Incomplete records may stand on an already present query or an extra,
        # but never on their missing query identity.
        donor = x if ident == 0 else int(rng.choice(specs[ident]))
        posts.append((height-ph, pc, donor))
        planks.append(_beam(rng, row, col, length, specs[ident], frame))
        if ident == 0:
            # Top incomplete record borrows the missing identity from the lower
            # exact match. Both must disappear in the SAME reaction round.
            planks.append(_beam(rng, row-1, col, length, [a, b, x], frame))
            # This extra-only cap survives and drops two rows after that round.
            planks.append(_beam(rng, row-2, col, length, [x, x, x], frame))
        col += length+1+offsets[k+1]
    return (height, width), posts, planks


def _feedback_scene(rng, colors, deep=False):
    """Shared removable bridge with reactive and inert overhang branches."""
    bg, frame = colors[:2]
    q = list(colors[2:5])
    rng.shuffle(q)
    a, b, missing = q
    extras = colors[5:]
    x = extras[0]
    branches = int(rng.choice([1, 2])) if deep else 2
    widths = [int(rng.integers(5, 9)) for _ in range(branches)]
    bridge_len = sum(widths) + (3 if branches == 2 else 4) + int(rng.integers(0, 4))
    width = min(30, bridge_len + int(rng.integers(0, 5)))
    left = int(rng.integers(0, width-bridge_len+1))
    depth = [int(rng.integers(3, 6) if deep else rng.integers(1, 3)) for _ in widths]
    tall = int(rng.integers(5, 11))
    # The reactive training branch is extended to two planks below; reserve
    # its full height before selecting the canvas, even if both draws were 1.
    height = int(rng.integers(max(17, 7+tall+max(2, max(depth))), 31))
    row = height-tall-1
    starts = [left] if branches == 1 else [left, left+bridge_len-widths[1]]
    tall_x = left+bridge_len-1 if branches == 1 else left+widths[0]+1
    posts = [(height-tall, tall_x, missing)]
    marks = [a, b, a, x, x]
    if len(extras) > 1 and rng.random() < .6:
        marks += [extras[1]]
    planks = [_beam(rng, row, left, bridge_len, marks, frame)]
    reactive = int(rng.integers(branches))
    if not deep:
        depth[reactive] = 2
    for j, (col, length) in enumerate(zip(starts, widths)):
        short = int(rng.integers(2, tall-1))
        donor = missing if j == reactive or (deep and rng.random() < .45) else x
        pc = col+int(rng.integers(1, length-1))
        posts.append((height-short, pc, donor))
        for k in range(depth[j]):
            own = [a, b, x]
            if length >= 7 and rng.random() < .65:
                own += [int(rng.choice([a, b, x]))]
            planks.append(_beam(rng, row-k-1, col, length, own, frame))
        # Missing TWO identities after landing: the cap remains even when its
        # support-connected component contains the complete query.
        cap = [a, x, x] if rng.random() < .5 else [x, x, x]
        planks.append(_beam(rng, row-depth[j]-1, col, length, cap, frame))
    return (height, width), sorted(posts, key=lambda p: p[1]), planks


def _free_scene(rng, colors):
    """Free deposition: spans and heights are unrelated to membership."""
    height, width = int(rng.integers(13, 31)), int(rng.integers(10, 31))
    xs = [int(x) for x in rng.permutation(np.arange(1, width-1))]
    selected = []
    for x in xs:
        if all(abs(x-y) >= 2 for y in selected):
            selected.append(x)
        if len(selected) == int(min(7, max(2, width//4))):
            break
    posts = [(height-int(rng.integers(2, min(12, height-6))), x, int(rng.choice(colors[2:])))
             for x in sorted(selected)]
    return (height, width), posts, []


def _add_random(rng, shape, colors, header, posts, planks, number):
    """Bounded deposit attempts; never retain failed payloads or retry logs."""
    bg, frame = colors[:2]
    for _ in range(number):
        for attempt in range(14):
            length = int(rng.integers(5, min(shape[1], 22)+1))
            col = int(rng.integers(0, shape[1]-length+1))
            nmarks = int(rng.integers(0, min(length-2, 8)+1))
            marks = [int(rng.choice(colors[2:])) for _ in range(nmarks)]
            p = _beam(rng, 5, col, length, marks, frame)
            # No insertion through an existing beam or post.
            if any(q[0] == 5 and max(col, q[1]) < min(col+length, q[1]+len(q[2])) for q in planks):
                continue
            if any(top == 5 and col <= x < col+length for top, x, _ in posts):
                continue
            trial = _settle(planks+[p], posts, shape[0])
            # Adding a new top piece cannot move the old stable scaffold.
            if trial[:-1] != planks:
                continue
            try:
                a = _render(shape, bg, header, posts, trial)
                info = _decode(a)
                assert set(info['planks']) == set(trial)
            except AssertionError:
                continue
            planks = trial
            break
    return planks


def _make_pair(rng, colors, kind, right=False):
    if kind == 'identity':
        shape, posts, planks = _identity_scene(rng, colors)
        extra = int(rng.integers(0, 3))
    elif kind in ('feedback', 'deep'):
        shape, posts, planks = _feedback_scene(rng, colors, deep=kind == 'deep')
        extra = int(rng.integers(0, 5 if kind == 'deep' else 3))
    else:
        shape, posts, planks = _free_scene(rng, colors)
        extra = int(rng.integers(4, 13))
    header = _header(rng, shape[1], colors, right)
    planks = _add_random(rng, shape, colors, header, posts, planks, extra)
    if not planks:
        # A full-width beam always has an unambiguous landing above all posts.
        row = min(p[0] for p in posts)-1
        planks = [_beam(rng, row, 0, shape[1], colors[2:5], colors[1])]
    a = _render(shape, colors[0], header, posts, planks)
    info = _decode(a)
    return {'input': a, 'output': _answer(info)}


def _integer(value, lower, upper, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+' must be an integer')
    if not lower <= value <= upper:
        raise ValueError(name+' is outside the legal range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt = _integer(num_train, 2, 10, 'num_train')
    ne = _integer(num_test, 1, 4, 'num_test')
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(int(seed))
    base = _colors(rng)
    vary_colors = bool(rng.integers(2))
    vary_corner = bool(rng.integers(2))
    base_right = bool(rng.integers(2))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, n in (('train', nt), ('test', ne)):
        for i in range(n):
            kind = ('identity' if i == 0 else 'feedback' if i == 1 else
                    str(rng.choice(['free', 'feedback', 'identity']))) if split == 'train' else (
                    'deep' if i == 0 else str(rng.choice(['free', 'deep', 'feedback'])))
            colors = _colors(rng) if vary_colors else base
            right = bool(i % 2) if vary_corner and split == 'train' else (
                    bool(rng.integers(2)) if vary_corner else base_right)
            for _ in range(16):
                pair = _make_pair(rng, colors, kind, right)
                k = _key(pair['input'])
                if k not in seen:
                    break
            else:
                raise AssertionError('bounded distinct-input construction exhausted')
            seen.add(k)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and np.array_equal(a[:5], b[:5])
            assert _key(a) not in seen
            seen.add(_key(a))
            info = _decode(a)
            result, ids, m = _react(info)
            assert np.array_equal(b, _render(a.shape, info['bg'], a[:5], info['posts'], result))
            assert not any(info['query'] <= own | under for own, under in _signatures(info, result))
            assert _settle(result, info['posts'], len(a)) == result
            assert len(m['removed'])+len(ids) == len(info['planks'])
            for ident, p in zip(ids, result):
                old = info['planks'][ident]
                assert p[0] >= old[0] and p[1:] == old[1:], 'rigid vertical translation only'
            for top, c, color in info['posts']:
                assert np.all(b[top:, c] == color), 'ground posts are fixed'
            m['no_contact_diff'] = not np.array_equal(b, _answer(info, contacts=False))
            m['no_gravity_diff'] = not np.array_equal(b, _answer(info, gravity=False))
            m['one_pass_diff'] = not np.array_equal(b, _answer(info, once=True))
            m['any_query_diff'] = not np.array_equal(b, _answer(info, any_query=True))
            records[split].append((info, m))
    train = records['train']
    for name in ('contact_only', 'extra_match', 'repeat_match', 'exact_three', 'count_match',
                 'supported_by_plank_match', 'no_contact_diff', 'no_gravity_diff', 'one_pass_diff',
                 'any_query_diff', 'forgetting_witness'):
        assert any(m[name] for _, m in train), 'missing training witness: '+name
    assert any(m['hard_missing'] == set(i['query']) for i, m in train)
    assert any(len(m['waves']) >= 3 and m['moved'] for _, m in train)
    assert any(len(m['waves']) >= 4 for _, m in records['test']), 'test must extend reaction depth'
    corners = {i['right'] for i, _ in train}
    assert all(i['right'] in corners for i, _ in records['test'])


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        digest.update(split.encode())
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds; no files, trace logs, or retained puzzle collection."""
    values = {k: set() for k in ('shapes', 'lengths', 'planks', 'posts', 'waves', 'falls', 'backgrounds', 'frames', 'layouts')}
    fingerprints = set()
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        if seed < 3:
            assert _fingerprint(generate_puzzle(seed)) == _fingerprint(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                pairs += 1
                info = _decode(pair['input'])
                a, ids, m = _react(info)
                alt, alt_ids, alt_m = _react(info, ticks=True)
                assert a == alt and ids == alt_ids and m['waves'] == alt_m['waves']
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                # Removing an order cue from the selector cannot change physics.
                changed = pair['input'].copy()
                view = changed[:, ::-1] if info['right'] else changed
                view[:3, :3] = np.roll(view[:3, :3].copy(), 1, axis=0)
                expected = pair['output'].copy()
                expected[:5] = changed[:5]
                assert np.array_equal(solve(changed), expected)
                if seed < 3:
                    reordered = dict(info, planks=list(reversed(info['planks'])))
                    assert np.array_equal(_answer(reordered), pair['output'])
                values['shapes'].add(info['grid'].shape)
                values['lengths'].update(len(p[2]) for p in info['planks'])
                values['planks'].add(len(info['planks']))
                values['posts'].add(len(info['posts']))
                values['waves'].add(len(m['waves']))
                values['falls'].update(m['distances'])
                values['backgrounds'].add(info['bg'])
                values['frames'].add(info['frame'])
                signature = (info['grid'] != info['bg']).tobytes()
                values['layouts'].add(hashlib.sha256(bytes(info['grid'].shape)+signature).hexdigest())
    assert len(fingerprints) == 42 and pairs == 210
    assert len(values['shapes']) >= 50 and len(values['layouts']) >= 180
    assert len(values['lengths']) >= 12 and len(values['planks']) >= 9
    assert len(values['posts']) >= 5 and len(values['waves']) >= 5
    assert len(values['falls']) >= 7
    assert values['backgrounds'] == values['frames'] == set(range(10))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(173, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (4.0, 1), (4, None), (4, False), (4, 1.5)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # A changed output cell must be rejected, even when its colors stay legal.
    bad = generate_puzzle(0, 2, 1)
    bad['train'][0]['output'][0, 0] = (int(bad['train'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged header accepted')
    return {'seeds': 42, 'pairs': pairs, 'distinct_puzzles': len(fingerprints),
            'variation': {k: len(v) for k, v in values.items()}}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
