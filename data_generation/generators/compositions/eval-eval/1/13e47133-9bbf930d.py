"""Contour sluices: tokens travel on tracks gated by invisible contour colors.

Only standard library and NumPy are used. No parent code or data is imported.
The diagonal key and all geometry are recovered afresh by solve(input_grid).
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import numpy as np


VOID, TRACK, TOKEN = 0, 8, 9
D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer grids required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "ARC colors required")
    return g.astype(np.int8, copy=False)


def _inside(mask, p):
    return 0 <= p[0] < mask.shape[0] and 0 <= p[1] < mask.shape[1] and bool(mask[p])


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, pending = {start}, [start]
        while pending:
            p = pending.pop()
            for d in D4:
                q = _add(p, d)
                if q in unseen:
                    unseen.remove(q)
                    cells.add(q)
                    pending.append(q)
        result.append(cells)
    return result


def _depths(mask, diagonal=True):
    """Successive simultaneous erosion; off-grid and holes are outside."""
    h, w = mask.shape
    remaining = mask.copy()
    depth = np.full(mask.shape, -1, dtype=np.int16)
    level = 0
    while remaining.any():
        padded = np.pad(remaining, 1)
        interior = remaining.copy()
        for dr, dc in D8 if diagonal else D4:
            interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        depth[remaining & ~interior] = level
        remaining = interior
        level += 1
    return depth


def _bfs_depths(mask):
    """A different computation used only by validation."""
    depth = np.full(mask.shape, -1, dtype=np.int16)
    pending = deque()
    for p in map(tuple, np.argwhere(mask).tolist()):
        if any(not _inside(mask, _add(p, d)) for d in D8):
            depth[p] = 0
            pending.append(p)
    while pending:
        p = pending.popleft()
        for d in D8:
            q = _add(p, d)
            if _inside(mask, q) and depth[q] < 0:
                depth[q] = depth[p] + 1
                pending.append(q)
    return depth


def _box_depths(mask):
    rr, cc = np.nonzero(mask)
    r, c = np.indices(mask.shape)
    return np.minimum.reduce((r - rr.min(), rr.max() - r, c - cc.min(), cc.max() - c)).astype(np.int16)


def _key_paths(mask, depth, length):
    choices = []
    for r, c in map(tuple, np.argwhere(mask & (depth == 0)).tolist()):
        for dr in (-1, 1):
            for dc in (-1, 1):
                if _inside(mask, (r - dr, c)) or _inside(mask, (r, c - dc)):
                    continue
                path = [(r + i * dr, c + i * dc) for i in range(length)]
                if all(_inside(mask, p) and depth[p] == i for i, p in enumerate(path)):
                    choices.append(path)
    return choices


def _banks(mask, depth, q, direction):
    n = (-direction[1], direction[0])
    a, b = _add(q, n), _add(q, (-n[0], -n[1]))
    if not (_inside(mask, a) and _inside(mask, b)):
        return None
    return int(depth[a]), int(depth[b])


def _gate(mask, depth, period, q, direction, exact_depth=False):
    banks = _banks(mask, depth, q, direction)
    if banks is None:
        return False
    a, b = banks
    return a == b if exact_depth else period[a % len(period)] == period[b % len(period)]


def _walk(path, mask, depth, period, exact_depth=False):
    last = 0
    for i in range(1, len(path)):
        p, q = path[i - 1], path[i]
        d = (q[0] - p[0], q[1] - p[1])
        if not _gate(mask, depth, period, q, d, exact_depth):
            break
        last = i
    return last


def _read(inp):
    """All roles, the palette period, and track ordering come from this input."""
    g = _grid(inp)
    mask = g != VOID
    _require(len(_components(mask)) == 1, "one connected chamber required")
    _require(not mask[0].any() and not mask[-1].any() and not mask[:, 0].any()
             and not mask[:, -1].any(), "chamber needs a void margin")
    counts = np.bincount(g.ravel(), minlength=10)
    base = max(range(1, 8), key=lambda c: int(counts[c]))
    cues = list(map(tuple, np.argwhere((g != VOID) & (g != TRACK) & (g != TOKEN) & (g != base)).tolist()))
    _require(2 <= len(cues) <= 4 and counts[base] > len(cues), "visible short diagonal key")
    depth = _depths(mask)
    cues.sort(key=lambda p: int(depth[p]))
    _require([int(depth[p]) for p in cues] == list(range(len(cues))), "one key cell per successive layer")
    dr, dc = cues[1][0] - cues[0][0], cues[1][1] - cues[0][1]
    _require(abs(dr) == abs(dc) == 1 and all(p == (cues[0][0] + i * dr, cues[0][1] + i * dc)
             for i, p in enumerate(cues)), "inward diagonal key")
    _require(cues in _key_paths(mask, depth, len(cues)), "key starts at a convex corner")
    period = tuple(int(g[p]) for p in cues)
    paths = []
    for cells in _components((g == TRACK) | (g == TOKEN)):
        sources = [p for p in sorted(cells) if g[p] == TOKEN]
        _require(len(sources) == 1 and len(cells) >= 5, "one token per track")
        neighbors = {p: [q for d in D4 if (q := _add(p, d)) in cells] for p in cells}
        ends = [p for p in cells if len(neighbors[p]) == 1]
        _require(len(ends) == 2 and sources[0] in ends and all(1 <= len(v) <= 2 for v in neighbors.values()),
                 "tracks are simple open paths with token at an end")
        path, previous, p = [], None, sources[0]
        while True:
            path.append(p)
            onward = [q for q in neighbors[p] if q != previous]
            if not onward:
                break
            previous, p = p, onward[0]
        _require(len(path) == len(cells), "track must not cycle")
        paths.append(path)
    _require(1 <= len(paths) <= 6, "one to six tracks")
    return dict(grid=g, mask=mask, base=base, depth=depth, period=period, cues=cues, paths=paths)


def _answer(a, depth=None, exact_depth=False, ignore_gates=False):
    out = a['grid'].copy()
    for path in a['paths']:
        last = len(path) - 1 if ignore_gates else _walk(path, a['mask'], a['depth'] if depth is None else depth,
                                                      a['period'], exact_depth)
        out[path[0]] = TRACK
        out[path[last]] = TOKEN
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Move tokens to the ends of their maximal matching-bank track prefixes."""
    return _answer(_read(input_grid))


def _shape(rng, h, w, is_test, length, force_hole=False):
    mask = np.zeros((h, w), dtype=bool)
    mask[1:-1, 1:-1] = True
    # Excisions, rather than a bank of room templates, generate orthogonal bays.
    notch_count = int(rng.integers(1, 4 if is_test else 3))
    for _ in range(notch_count):
        work = mask.T if rng.integers(2) else mask
        if rng.integers(2):
            work = work[::-1]
        span = int(rng.integers(2, min(7, work.shape[1] - 5)))
        start = int(rng.integers(3, work.shape[1] - span - 1))
        amount = int(rng.integers(2, min(6, work.shape[0] // 3)))
        work[1:1 + amount, start:start + span] = False
    if force_hole or rng.random() < .35:
        r = int(rng.integers(5, h - 5))
        c = int(rng.integers(5, w - 5))
        mask[r:r + int(rng.integers(1, 3)), c:c + int(rng.integers(1, 3))] = False
    if len(_components(mask)) != 1:
        return None
    depth = _depths(mask)
    if int(depth.max()) < length + 1 or not _key_paths(mask, depth, length):
        return None
    return mask, depth


def _clearance(cells, shape):
    blocked = set(cells)
    for p in cells:
        blocked.update(_add(p, d) for d in D4)
    return {p for p in blocked if 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]}


def _path_candidate(rng, mask, depth, period, forbidden, kind, requirement=None, is_test=False):
    """Grow a simple track around an intentional pass/fail witness.

    Construction never supplies a destination to the solver. The finished input
    is decoded independently after drawing the track.
    """
    available = [p for p in map(tuple, np.argwhere(mask).tolist()) if p not in forbidden]
    if not available:
        return None
    starts = []
    alt = _box_depths(mask) if requirement == 'contour' else None
    for p in available:
        for d in D4:
            q = _add(p, d)
            if not _inside(mask, q) or q in forbidden:
                continue
            gate = _gate(mask, depth, period, q, d)
            if gate != (kind != 'stalled'):
                continue
            if requirement == 'alias':
                banks = _banks(mask, depth, q, d)
                if banks is None or banks[0] == banks[1] or max(banks) < len(period):
                    continue
            if requirement == 'contour' and gate == _gate(mask, alt, period, q, d):
                continue
            starts.append((p, q))
    if not starts:
        return None
    path = list(starts[int(rng.integers(len(starts)))])
    used = set(path)
    prefix = 0 if kind == 'stalled' else int(rng.integers(3, 11 if is_test else 8))
    # Complete tracks contain only passes. Halted tracks include a first failure
    # after the pass prefix; stalled tracks fail immediately.
    suffix = int(rng.integers(3, 9 if is_test else 7))
    steps = prefix if kind == 'complete' else prefix + 1 + suffix
    for edge in range(2, steps + 1):
        candidates, weights = [], []
        previous_d = (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])
        for d in D4:
            q = _add(path[-1], d)
            if not _inside(mask, q) or q in forbidden or q in used:
                continue
            if any(_add(q, n) in used and _add(q, n) != path[-1] for n in D4):
                continue
            wanted = None
            if kind == 'complete' or edge <= prefix:
                wanted = True
            elif kind == 'halted' and edge == prefix + 1:
                wanted = False
            if wanted is not None and _gate(mask, depth, period, q, d) != wanted:
                continue
            candidates.append(q)
            weights.append(5 if d == previous_d else 2)
        if not candidates:
            return None
        weights = np.asarray(weights, dtype=float)
        q = candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
        path.append(q)
        used.add(q)
    if len(path) < 5:
        return None
    last = _walk(path, mask, depth, period)
    if kind == 'complete' and last != len(path) - 1:
        return None
    if kind == 'halted' and not 2 <= last < len(path) - 2:
        return None
    if kind == 'stalled' and last != 0:
        return None
    return path


def _features(a):
    statuses, turns, alias, repeat = set(), 0, False, False
    lengths, moved = [], []
    for path in a['paths']:
        last = _walk(path, a['mask'], a['depth'], a['period'])
        statuses.add('stalled' if last == 0 else 'complete' if last == len(path) - 1 else 'halted')
        lengths.append(len(path))
        moved.append(last)
        directions = [(q[0] - p[0], q[1] - p[1]) for p, q in zip(path, path[1:])]
        turns += sum(x != y for x, y in zip(directions[:last], directions[1:last]))
        for q, d in zip(path[1:last + 1], directions[:last]):
            banks = _banks(a['mask'], a['depth'], q, d)
            if banks and banks[0] != banks[1]:
                alias = True
                repeat |= max(banks) >= len(a['period'])
    actual = _answer(a)
    return dict(statuses=statuses, turns=turns, alias=alias, repeat=repeat, lengths=lengths, moved=moved,
                contour=not np.array_equal(actual, _answer(a, _box_depths(a['mask']))),
                diagonal=not np.array_equal(actual, _answer(a, _depths(a['mask'], False))),
                cycling=not np.array_equal(actual, _answer(a, np.minimum(a['depth'], len(a['period']) - 1))),
                phase=not np.array_equal(actual, _answer(dict(a, period=a['period'][1:] + a['period'][:1]))))


def _make_pair(rng, index, is_test):
    length = 3 if index == 0 and not is_test else 4 if index == 1 and not is_test else int(rng.integers(2, 5))
    if is_test:
        length = int(rng.integers(3, 5))
    low = 26 if is_test else 21 if index < 2 else 15 + length
    h, w = map(int, rng.integers(low, 31 if is_test else 26, size=2))
    shape = _shape(rng, h, w, is_test, length, force_hole=not is_test and index == 1)
    if shape is None:
        return None
    mask, depth = shape
    base, x, y, z = map(int, rng.choice(np.arange(1, 8), size=4, replace=False))
    skeleton = [x, y] if length == 2 else [x, x, y] if length == 3 else [x, y, x, z]
    period = tuple(np.roll(skeleton, int(rng.integers(length))).tolist())
    keys = _key_paths(mask, depth, length)
    key = keys[int(rng.integers(len(keys)))]
    forbidden = _clearance(key, mask.shape)
    paths = []
    # The first two demonstrations cover the complete operation even for (2,1).
    if not is_test and index == 0:
        requests = [('complete', 'alias'), ('stalled', 'contour'), ('halted', None)]
    elif not is_test and index == 1:
        requests = [('halted', 'alias'), ('stalled', 'contour')]
    else:
        requests = [('halted', None), ('stalled', None)]
    for _ in range(int(rng.integers(0, 5 if is_test else 3))):
        requests.append((str(rng.choice(['complete', 'halted', 'stalled'])), None))
    for kind, requirement in requests:
        path = None
        for _ in range(35):
            path = _path_candidate(rng, mask, depth, period, forbidden, kind, requirement, is_test)
            if path is not None:
                break
        if path is None:
            return None
        paths.append(path)
        forbidden |= _clearance(path, mask.shape)
    inp = np.where(mask, base, VOID).astype(np.int8)
    for p, color in zip(key, period):
        inp[p] = color
    for path in paths:
        for p in path:
            inp[p] = TRACK
        inp[path[0]] = TOKEN
    rotation = int(rng.integers(4))
    inp = np.rot90(inp, rotation).copy()
    a = _read(inp)
    f = _features(a)
    if not is_test and index < 2 and (not f['repeat'] or not f['turns'] or not f['diagonal']
                                    or not f['cycling'] or not f['phase']):
        return None
    return {'input': inp, 'output': _answer(a)}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def _signature(g):
    return g.shape, g.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            for _ in range(180):
                pair = _make_pair(rng, index, split == 'test')
                if pair is not None and _signature(pair['input']) not in seen:
                    puzzle[split].append(pair)
                    seen.add(_signature(pair['input']))
                    break
            else:
                raise RuntimeError('bounded chamber construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'pair counts')
    seen, train_features, train_lengths, shapes = set(), [], set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'shape preservation')
            sig = _signature(inp)
            _require(sig not in seen, 'distinct inputs')
            seen.add(sig)
            a = _read(inp)
            _require(np.array_equal(a['depth'], _bfs_depths(a['mask'])), 'erosion and BFS disagree')
            expected = _answer(a)
            _require(np.array_equal(expected, out), 'incorrect token transfers')
            track_mask = (inp == TRACK) | (inp == TOKEN)
            _require(np.array_equal(inp[~track_mask], out[~track_mask]), 'terrain and key preservation')
            _require(np.count_nonzero(inp == TOKEN) == np.count_nonzero(out == TOKEN) == len(a['paths']),
                     'token conservation')
            f = _features(a)
            _require(any(f['moved']) and any(m < n - 1 for m, n in zip(f['moved'], f['lengths'])),
                     'moving and blocked tracks required')
            _require(np.count_nonzero(inp != out) == 2 * sum(m > 0 for m in f['moved']), 'only token endpoints change')
            shapes.add(inp.shape)
            if split == 'train':
                train_features.append(f)
                train_lengths.add(len(a['period']))
            else:
                _require(min(inp.shape) >= 26, 'test size shift')
                _require(len(a['period']) in train_lengths, 'test key length demonstrated')
    _require({3, 4} <= train_lengths, 'training demonstrates two palette periods')
    _require(set.union(*(f['statuses'] for f in train_features)) == {'stalled', 'halted', 'complete'}, 'three stopping cases')
    _require(any(f['repeat'] for f in train_features), 'cyclic unequal-depth match demonstrated')
    _require(any(f['cycling'] for f in train_features) and any(f['phase'] for f in train_features),
             'key order and cycle repetition must affect demonstrated transfers')
    _require(any(f['turns'] for f in train_features), 'bent movement demonstrated')
    _require(any(f['contour'] for f in train_features) and any(f['diagonal'] for f in train_features),
             'nonrectangular eight-neighbor contour witness')
    _require(len(shapes) >= 2, 'dimensions vary')


def self_test():
    """Bounded 42-seed checks; aggregate summaries only, no on-disk payloads."""
    shapes, masks, palettes, counts, lengths, turns, signatures = (set() for _ in range(7))
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(hashlib.sha256(p['train'][0]['input'].tobytes()).hexdigest())
        for split in ('train', 'test'):
            for pair in p[split]:
                a = _read(pair['input'])
                _require(np.array_equal(solve(pair['input']), pair['output']), 'solve agreement')
                shapes.add(a['grid'].shape)
                masks.add(hashlib.sha256(a['mask'].tobytes()).hexdigest())
                palettes.add((a['base'], a['period']))
                counts.add(len(a['paths']))
                lengths.update(map(len, a['paths']))
                turns.add(_features(a)['turns'])
    _require(len(signatures) == 42 and len(masks) >= 100, 'structural variation')
    _require(len(shapes) >= 35 and (30, 30) in shapes, 'dimension variation')
    _require(len(palettes) >= 100 and len(counts) >= 3 and len(lengths) >= 10 and len(turns) >= 4, 'family variation')
    first = generate_puzzle(70, 2, 1)
    again = generate_puzzle(70, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in first for x, y in zip(first[s], again[s])
                 for k in ('input', 'output')), 'determinism')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt transfer accepted')
    return {'seeds': 42, 'shapes': len(shapes), 'terrain_geometries': len(masks), 'palettes': len(palettes),
            'track_counts': sorted(counts), 'track_lengths': sorted(lengths), 'moving_turn_counts': sorted(turns)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
