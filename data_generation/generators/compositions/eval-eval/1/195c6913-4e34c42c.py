"""Synchronize cyclic paint across overlapping, independently placed trail views.

Only NumPy and the standard library are used. Generation paints from latent path
indices; solve reconstructs all indices from input pixels. No runtime file I/O.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_REQUIRED = {"cycle", "repeated_word", "bend_phase", "local_restart",
             "duplicate_time", "unseeded", "relay"}


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _groups(points):
    todo = set(points)
    groups = []
    while todo:
        first = min(todo)
        todo.remove(first)
        group, stack = {first}, [first]
        while stack:
            p = stack.pop()
            for d in _DIRECTIONS:
                q = _add(p, d)
                if q in todo:
                    todo.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _origin(cells):
    return min(r for r, c in cells), min(c for r, c in cells)


def _normalize(cells):
    r0, c0 = _origin(cells)
    return {(r-r0, c-c0): int(v) for (r, c), v in cells.items()}


def _shape(cells):
    r0, c0 = _origin(cells)
    return max(r for r, c in cells)-r0+1, max(c for r, c in cells)-c0+1


def _pattern(cells):
    return tuple((r, c, v) for (r, c), v in sorted(_normalize(cells).items()))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "integer grids required"
    assert np.all((0 <= a) & (a <= 9)), "color range"
    return a.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _decode(input_grid):
    """Input-only role decoding and exact landmark translation equations."""
    g = _grid(input_grid)
    colors, counts = np.unique(g, return_counts=True)
    assert counts.max()*2 > g.size, "background must have a majority"
    background = int(colors[counts.argmax()])
    components = _groups(map(tuple, np.argwhere(g != background)))
    legend, pieces = [], []
    for cells in components:
        if len(cells) == 4 and _shape(cells) == (2, 2) and len({int(g[p]) for p in cells}) == 1:
            legend.append(cells)
        else:
            pieces.append(cells)
    assert 2 <= len(legend) <= 6 and len(pieces) >= 2, "program and views required"
    origins = [_origin(s) for s in legend]
    horizontal = len({r for r, c in origins}) == 1
    vertical = len({c for r, c in origins}) == 1
    assert horizontal != vertical, "legend must have a single reading axis"
    axis = 1 if horizontal else 0
    legend.sort(key=lambda s: _origin(s)[axis])
    assert all(_origin(b)[axis]-_origin(a)[axis] >= 3 for a, b in zip(legend, legend[1:]))
    word = tuple(int(g[min(s)]) for s in legend)
    annotation = set().union(*legend)
    starts = [p for part in pieces for p in part if int(g[p]) in word]
    assert len(starts) == 1 and int(g[starts[0]]) == word[0], "unique phase-zero endpoint"
    start = starts[0]
    neighbors = [_add(start, d) for d in _DIRECTIONS]
    neighbors = [p for p in neighbors if 0 <= p[0] < g.shape[0] and
                 0 <= p[1] < g.shape[1] and int(g[p]) != background]
    assert len(neighbors) == 1, "start must have one foreground neighbor"
    neutral = int(g[neighbors[0]])
    assert neutral not in set(word) | {background}
    routes, features, occurrences = [], [], defaultdict(list)
    start_piece = None
    for i, part in enumerate(pieces):
        route = {p for p in part if int(g[p]) == neutral or p == start}
        assert len(route) >= 3 and len(_groups(route)) == 1, "connected trail interval"
        degree = {p: sum(_add(p, d) in route for d in _DIRECTIONS) for p in route}
        assert set(degree.values()) <= {1, 2} and sum(v == 1 for v in degree.values()) == 2
        if start in route:
            start_piece = i
        keys = []
        for cells in _groups(part-route):
            feature = {p: int(g[p]) for p in cells}
            assert len(feature) >= 3 and len(set(feature.values())) >= 2, "whole nonmonochrome landmarks"
            assert not (set(feature.values()) & (set(word) | {neutral, background}))
            signature = _pattern(feature)
            assert all(signature != old[0] for old in keys), "key repeated within a view"
            anchor = _origin(cells)
            keys.append((signature, anchor))
            occurrences[signature].append((i, anchor))
        assert keys, "every view must be registered"
        routes.append(route)
        features.append(keys)
    assert start_piece is not None
    adjacency = [[] for _ in pieces]
    for signature, items in occurrences.items():
        assert len(items) >= 2, "unmatched landmark"
        for a in range(len(items)):
            i, p = items[a]
            for j, q in items[a+1:]:
                delta = p[0]-q[0], p[1]-q[1]
                adjacency[i].append((j, delta))
                adjacency[j].append((i, (-delta[0], -delta[1])))
    offsets = {start_piece: (0, 0)}
    depths = {start_piece: 0}
    queue = [start_piece]
    for i in queue:
        for j, delta in adjacency[i]:
            candidate = _add(offsets[i], delta)
            if j in offsets:
                assert offsets[j] == candidate, "inconsistent registration"
            else:
                offsets[j] = candidate
                depths[j] = depths[i]+1
                queue.append(j)
    assert len(offsets) == len(pieces), "disconnected registration"
    union, trail = {}, set()
    for i, part in enumerate(pieces):
        for p in part:
            q, color = _add(p, offsets[i]), int(g[p])
            assert q not in union or union[q] == color, "foreground collision"
            union[q] = color
        trail.update(_add(p, offsets[i]) for p in routes[i])
    assert max(_shape(union)) <= 30, "recovered scene exceeds family bounds"
    assert len(_groups(trail)) == 1
    degree = {p: sum(_add(p, d) in trail for d in _DIRECTIONS) for p in trail}
    assert set(degree.values()) <= {1, 2} and sum(v == 1 for v in degree.values()) == 2
    assert degree[start] == 1, "global start must be an endpoint"
    order, previous, current = [], None, start
    while True:
        order.append(current)
        following = [_add(current, d) for d in _DIRECTIONS
                     if _add(current, d) in trail and _add(current, d) != previous]
        if not following:
            break
        assert len(following) == 1 and len(order) < len(trail), "non-simple route"
        previous, current = current, following[0]
    assert len(order) == len(trail)
    index = {p: n for n, p in enumerate(order)}
    intervals, local_indices = [], []
    for i, route in enumerate(routes):
        numbers = {p: index[_add(p, offsets[i])] for p in route}
        lo, hi = min(numbers.values()), max(numbers.values())
        assert hi-lo+1 == len(numbers), "view is not one contiguous interval"
        intervals.append((lo, hi))
        local_indices.append(numbers)
    assert any(len(routes[i]) + len(routes[j]) >
               len({_add(p, offsets[i]) for p in routes[i]} |
                   {_add(p, offsets[j]) for p in routes[j]})
               for i in range(len(routes)) for j in range(i)), "no trail overlap"
    # A matched landmark must identify shared trail material as well as a key.
    for items in occurrences.values():
        for a, (i, _) in enumerate(items):
            left = {_add(p, offsets[i]) for p in routes[i]}
            for j, _ in items[a+1:]:
                right = {_add(p, offsets[j]) for p in routes[j]}
                assert len(left & right) >= 3, "landmark without a shared trail interval"
    return dict(grid=g, background=background, neutral=neutral, word=word,
                annotation=annotation, pieces=pieces, routes=routes,
                features=features, offsets=offsets, depths=depths,
                order=order, local_indices=local_indices, intervals=intervals,
                key_count=len(occurrences), start_piece=start_piece)


def _paint(info, phase=None, word=None):
    out = info['grid'].copy()
    for p in info['annotation']:
        out[p] = info['background']
    word = info['word'] if word is None else word
    for i, numbers in enumerate(info['local_indices']):
        for p, n in numbers.items():
            out[p] = word[(n if phase is None else phase(i, n)) % len(word)]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the shared clock from this input alone, then paint in place."""
    return _paint(_decode(input_grid))


def _coverage(info):
    correct = _paint(info)
    evidence = {'unseeded'} if any(i != info['start_piece'] for i in range(len(info['pieces']))) else set()
    n, word = len(info['order']), info['word']
    if n >= 2*len(word):
        evidence.add('cycle')
    unique = tuple(dict.fromkeys(word))
    if not np.array_equal(correct, _paint(info, word=unique)):
        evidence.add('repeated_word')
    if not np.array_equal(correct, _paint(info, lambda i, k: k-info['intervals'][i][0])):
        evidence.add('local_restart')
    # Reset to phase zero at every change of direction, counting a bend once.
    phases, local, old_direction = [0], 0, None
    for p, q in zip(info['order'], info['order'][1:]):
        direction = q[0]-p[0], q[1]-p[1]
        local = 0 if old_direction is not None and direction != old_direction else local+1
        phases.append(local)
        old_direction = direction
    if not np.array_equal(correct, _paint(info, lambda i, k: phases[k])):
        evidence.add('bend_phase')
    # Process maximal view intervals upstream-to-downstream, naively counting
    # the whole next observation rather than subtracting its shared cells.
    maximal = [i for i, (a, b) in enumerate(info['intervals']) if not any(
        c <= a and b <= d and ((a, b) != (c, d) or j < i)
        for j, (c, d) in enumerate(info['intervals']) if i != j)]
    clock, wrong = 0, {}
    for i in sorted(maximal, key=lambda j: info['intervals'][j]):
        lo, hi = info['intervals'][i]
        wrong[i] = clock-lo
        clock += hi-lo+1
    if any(word[k % len(word)] != word[(k+wrong[i]) % len(word)]
           for i in maximal for k in info['local_indices'][i].values()):
        evidence.add('duplicate_time')
    if max(info['depths'].values()) >= 2:
        evidence.add('relay')
    return evidence


def _walk(rng, length, winding):
    """Bounded self-avoiding thin trails with no unintended path contacts."""
    for _ in range(100):
        path, used = [(0, 0)], {(0, 0)}
        heading = int(rng.integers(4))
        while len(path) < length:
            candidates = []
            for direction in (heading, (heading-1) % 4, (heading+1) % 4):
                q = _add(path[-1], _DIRECTIONS[direction])
                if q in used or any(_add(q, d) in used and _add(q, d) != path[-1]
                                    for d in _DIRECTIONS):
                    continue
                rr = [p[0] for p in path]+[q[0]]
                cc = [p[1] for p in path]+[q[1]]
                if max(rr)-min(rr) > 23 or max(cc)-min(cc) > 23:
                    continue
                candidates.append(direction)
            if not candidates:
                break
            weights = np.array([1.0 if d != heading else winding for d in candidates])
            heading = int(rng.choice(candidates, p=weights/weights.sum()))
            path.append(_add(path[-1], _DIRECTIONS[heading]))
            used.add(path[-1])
        if len(path) == length:
            return path
    raise AssertionError('bounded trail construction exhausted')


_KEY_SHAPES = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    tuple((r, c) for r in range(2) for c in range(3)),
    tuple((r, c) for r in range(3) for c in range(2)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
)


def _landmark(rng, pivot, path_set, occupied, colors, signatures):
    for _ in range(120):
        footprint = _KEY_SHAPES[int(rng.integers(len(_KEY_SHAPES)))]
        values = [int(rng.choice(colors)) for _ in footprint]
        if len(set(values)) == 1:
            values[-1] = next(c for c in colors if c != values[0])
        local = dict(zip(footprint, values))
        signature = _pattern(local)
        if signature in signatures:
            continue
        contact = footprint[int(rng.integers(len(footprint)))]
        neighbor = _add(pivot, _DIRECTIONS[int(rng.integers(4))])
        delta = neighbor[0]-contact[0], neighbor[1]-contact[1]
        key = {_add(p, delta): v for p, v in local.items()}
        if set(key) & path_set:
            continue
        if any(p in occupied or any(_add(p, d) in occupied for d in _DIRECTIONS) for p in key):
            continue
        return key, signature
    raise AssertionError('no clear landmark attachment')


def _word(rng, palette, witness=False):
    length = int(rng.integers(3 if witness else 2, 7))
    distinct = int(rng.integers(2, min(4, length-1 if witness else length)+1))
    labels = list(range(distinct)) + list(map(int, rng.integers(distinct, size=length-distinct)))
    rng.shuffle(labels)
    if witness:
        # Reject words whose deduplication happens to generate the same stream.
        for _ in range(20):
            unique = tuple(dict.fromkeys(labels))
            if any(labels[k % length] != unique[k % len(unique)] for k in range(2*length)):
                break
            rng.shuffle(labels)
        else:
            labels = [0]*(length-1)+[1]
    return tuple(palette[k] for k in labels)


def _pack(rng, parts, word, background):
    """Random separated placement, using shape masks rather than fixed slots."""
    normalized = [_normalize(p) for p in parts]
    horizontal = bool(rng.integers(2))
    gaps = list(map(int, rng.integers(1, 3, size=len(word)-1)))
    span = 2*len(word)+sum(gaps)
    swatch_origins = [0]
    for gap in gaps:
        swatch_origins.append(swatch_origins[-1]+2+gap)
    total_foreground = sum(map(len, parts))+4*len(word)
    for attempt in range(45):
        h = int(rng.integers(18, 31)) if attempt < 30 else 30
        w = int(rng.integers(18, 31)) if attempt < 30 else 30
        if span > (w if horizontal else h) or h*w <= 2*total_foreground:
            continue
        g = np.full((h, w), background, dtype=np.int8)
        blocked = np.zeros((h, w), dtype=bool)
        strip = int(rng.integers(4, 7))
        far = bool(rng.integers(2))
        if horizontal:
            blocked[h-strip:] = True if far else False
            if not far:
                blocked[:strip] = True
            r0 = h-int(rng.integers(2, strip)) if far else int(rng.integers(strip-2))
            c0 = int(rng.integers(w-span+1))
        else:
            blocked[:, w-strip:] = True if far else False
            if not far:
                blocked[:, :strip] = True
            c0 = w-int(rng.integers(2, strip)) if far else int(rng.integers(strip-2))
            r0 = int(rng.integers(h-span+1))
        for offset, color in zip(swatch_origins, word):
            r, c = (r0, c0+offset) if horizontal else (r0+offset, c0)
            g[r:r+2, c:c+2] = color
        order = list(map(int, rng.permutation(len(parts))))
        # Large extents first; random ties and positions carry no logical order.
        order.sort(key=lambda i: -max(_shape(normalized[i])))
        placements = [None]*len(parts)
        success = True
        for i in order:
            part = normalized[i]
            ph, pw = _shape(part)
            if ph > h or pw > w:
                success = False
                break
            mask = np.zeros((ph, pw), dtype=bool)
            for p in part:
                mask[p] = True
            windows = np.lib.stride_tricks.sliding_window_view(blocked, (ph, pw))
            feasible = np.argwhere(~np.any(windows & mask, axis=(-2, -1)))
            if len(feasible) == 0:
                success = False
                break
            top, left = map(int, feasible[int(rng.integers(len(feasible)))])
            old_origin = _origin(parts[i])
            placements[i] = top-old_origin[0], left-old_origin[1]
            for (r, c), color in part.items():
                r, c = r+top, c+left
                g[r, c] = color
                blocked[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = True
        if success:
            return g, placements
    raise AssertionError('bounded packing exhausted')


def _make_pair(rng, primary_count, redundancy, key_color_count, winding, witness):
    for _ in range(100):
        palette = list(map(int, rng.permutation(10)))
        background, neutral = palette[:2]
        key_colors = palette[2:2+key_color_count]
        program_colors = palette[2+key_color_count:]
        word = _word(rng, program_colors, witness)
        length = 8*primary_count+int(rng.integers(6, 12))
        try:
            path = _walk(rng, length, winding)
            cuts = [int(round(length*j/primary_count))+int(rng.integers(-1, 2))
                    for j in range(1, primary_count)]
            before = list(map(int, rng.integers(1, 4, size=len(cuts))))
            after = list(map(int, rng.integers(1, 4, size=len(cuts))))
            intervals = [(0 if j == 0 else cuts[j-1]-before[j-1],
                          length-1 if j == primary_count-1 else cuts[j]+after[j])
                         for j in range(primary_count)]
            landmarks, occupied, signatures = [], set(), set()
            for cut in cuts:
                cells, signature = _landmark(rng, path[cut], set(path), occupied,
                                             key_colors, signatures)
                landmarks.append(cells)
                occupied.update(cells)
                signatures.add(signature)
            if redundancy:
                j = int(rng.integers(len(cuts)))
                cut = cuts[j]
                intervals.append((cut-int(rng.integers(1, before[j]+1)),
                                  cut+int(rng.integers(1, after[j]+1))))
            parts = []
            for lo, hi in intervals:
                part = {path[k]: word[0] if k == 0 else neutral for k in range(lo, hi+1)}
                for cut, key in zip(cuts, landmarks):
                    if lo <= cut <= hi:
                        part.update(key)
                parts.append(part)
            whole = {p: neutral for p in path}
            for key in landmarks:
                whole.update(key)
            if max(_shape(whole)) > 30:
                continue
            inp, placements = _pack(rng, parts, word, background)
            # Ground truth is direct indexing of the constructed trail. It
            # does not call solve or use reconstructed placement information.
            out = inp.copy()
            for color in set(word):
                out[inp == color] = background
            for i, (lo, hi) in enumerate(intervals):
                for k in range(lo, hi+1):
                    out[_add(path[k], placements[i])] = word[k % len(word)]
            info = _decode(inp)
            assert len(info['order']) == length
            assert set(info['intervals']) == set(intervals)
            needed = {'cycle', 'bend_phase', 'local_restart', 'duplicate_time', 'unseeded'}
            if witness:
                needed.add('repeated_word')
            if primary_count >= 3:
                needed.add('relay')
            if not needed <= _coverage(info):
                continue
            assert np.array_equal(out, _paint(info)), "latent/pixel reconstruction mismatch"
            return {'input': inp, 'output': out}
        except AssertionError:
            continue
    raise AssertionError('bounded pair construction exhausted')


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not _integer(num_train) or not _integer(num_test):
        raise ValueError('example counts must be integers')
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('require 2..10 train, 1..4 test and num_train > num_test')
    rng = np.random.default_rng(seed)
    redundancy = bool(rng.integers(2))
    key_color_count = int(rng.integers(2, 4))
    winding = float(rng.choice([2.0, 4.0, 7.0]))
    opening = list(map(int, rng.permutation(2)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for j in range(count):
            witness = split == 'train' and j == opening.index(0)
            relay = split == 'train' and j == opening.index(1)
            primary = 4 if split == 'test' else (2 if witness else 3 if relay else int(rng.integers(2, 4)))
            redundant = redundancy and (split == 'test' or relay or (j >= 2 and rng.random() < .4))
            for _ in range(8):
                pair = _make_pair(rng, primary, redundant, key_color_count, winding, witness)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('distinct inputs exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, evidence = set(), set()
    redundant = {'train': False, 'test': False}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(out, _paint(info)), 'incorrect synchronized paint'
            trail_points = set().union(*info['routes'])
            preserved = np.ones(inp.shape, dtype=bool)
            for p in trail_points | info['annotation']:
                preserved[p] = False
            assert np.array_equal(inp[preserved], out[preserved]), 'keys or background changed'
            assert all(out[p] == info['background'] for p in info['annotation'])
            assert not np.any(out == info['neutral']), 'unpainted trail'
            assert all(len(numbers) >= 3 for numbers in info['local_indices'])
            intervals = info['intervals']
            redundant[split] |= any(a >= c and b <= d for i, (a, b) in enumerate(intervals)
                                    for j, (c, d) in enumerate(intervals) if i != j)
            if split == 'train':
                evidence |= _coverage(info)
    assert _REQUIRED <= evidence, 'missing training evidence: '+str(_REQUIRED-evidence)
    assert not redundant['test'] or redundant['train'], 'redundancy not demonstrated'


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                grid = _grid(pair[role])
                h.update(bytes(grid.shape))
                h.update(grid.tobytes())
    return h.hexdigest()


def _structure(grid):
    roles = {}
    return grid.shape, bytes(roles.setdefault(int(c), len(roles)) for c in grid.flat)


def self_test():
    """42 seeds; bounded memory, no files, pixel-level invariants and mutations."""
    shapes, structures, words, turns, backgrounds, counts = set(), set(), set(), set(), set(), set()
    signatures, key_shapes, phase_patterns = set(), set(), set()
    redundant_views = 0
    pairs_checked = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                shapes.add(pair['input'].shape)
                structures.add(hashlib.sha256(repr(_structure(pair['input'])).encode()).hexdigest())
                labels = {}
                words.add(tuple(labels.setdefault(c, len(labels)) for c in info['word']))
                backgrounds.add(info['background'])
                counts.add(len(info['pieces']))
                directions = [(q[0]-p[0], q[1]-p[1]) for p, q in zip(info['order'], info['order'][1:])]
                turns.add(sum(a != b for a, b in zip(directions, directions[1:])))
                phase_patterns.add(tuple(sorted((lo % len(info['word']), hi-lo+1)
                                                for lo, hi in info['intervals'])))
                for keys in info['features']:
                    for pattern, origin in keys:
                        key_shapes.add(tuple((r, c) for r, c, color in pattern))
                redundant_views += sum(any(a >= c and b <= d for j, (c, d) in enumerate(info['intervals'])
                                           if i != j) for i, (a, b) in enumerate(info['intervals']))
                pairs_checked += 1
        if seed < 3:
            assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed))
    assert len(signatures) == 42 and len(structures) == pairs_checked
    assert len(shapes) >= 35 and len(words) >= 20 and len(turns) >= 10
    assert backgrounds == set(range(10)) and counts >= {2, 3, 4, 5}
    assert len(key_shapes) >= 7 and redundant_views > 0 and len(phase_patterns) >= 100
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(6151+nt*10+ne, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3, False), (3.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Palette equivariance includes changing which literal color is background.
    p = generate_puzzle(29, 2, 1)
    perm = np.array([8, 4, 0, 6, 9, 2, 1, 7, 5, 3], dtype=np.int8)
    for pair in p['train']+p['test']:
        assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
    # Repack observations independently: phase follows the view, not its slot.
    info = _decode(p['test'][0]['input'])
    parts = [{q: int(info['grid'][q]) for q in part} for part in info['pieces']]
    moved, shifts = _pack(np.random.default_rng(121), parts, info['word'], info['background'])
    expected = moved.copy()
    for color in set(info['word']):
        expected[moved == color] = info['background']
    for i, numbers in enumerate(info['local_indices']):
        for q, n in numbers.items():
            expected[_add(q, shifts[i])] = info['word'][n % len(info['word'])]
    assert np.array_equal(solve(moved), expected)
    # A changed word alters remote view paint while leaving landmark pixels.
    info = _decode(p['train'][0]['input'])
    changed = info['grid'].copy()
    # A global program-color permutation preserves decoding even with repeats.
    used = list(dict.fromkeys(info['word']))
    mapping = {c: used[(j+1) % len(used)] for j, c in enumerate(used)}
    for c, replacement in mapping.items():
        changed[info['grid'] == c] = replacement
    actual = solve(changed)
    assert any(actual[q] != p['train'][0]['output'][q]
               for i, route in enumerate(info['routes']) if i != info['start_piece'] for q in route)
    assert all(actual[q] == changed[q] for part in info['pieces'] for q in part
               if int(changed[q]) not in set(mapping.values()) | {info['neutral']})
    wrong = {'train': [dict(pair) for pair in p['train']], 'test': p['test']}
    wrong['train'][0] = dict(p['train'][0])
    wrong['train'][0]['output'] = _paint(info, lambda i, k: k-info['intervals'][i][0])
    assert not np.array_equal(wrong['train'][0]['output'], p['train'][0]['output'])
    try:
        validate_generated_puzzle(wrong)
    except AssertionError:
        pass
    else:
        raise AssertionError('local phase restart accepted')
    repeated_pair = next(pair for pair in p['train']
                         if 'repeated_word' in _coverage(_decode(pair['input'])))
    repeated_info = _decode(repeated_pair['input'])
    deduplicated = _paint(repeated_info, word=tuple(dict.fromkeys(repeated_info['word'])))
    assert not np.array_equal(deduplicated, repeated_pair['output'])
    bad = {'train': [dict(pair) for pair in p['train']], 'test': p['test']}
    slot = next(j for j, pair in enumerate(p['train']) if pair is repeated_pair)
    bad['train'][slot]['output'] = deduplicated
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('deduplicated cycle accepted')
    return {'seeds': 42, 'pairs': pairs_checked, 'distinct_structures': len(structures),
            'input_shapes': len(shapes), 'word_patterns': len(words),
            'turn_counts': len(turns), 'view_counts': sorted(counts),
            'landmark_shapes': len(key_shapes), 'redundant_views': redundant_views,
            'phase_interval_patterns': len(phase_patterns), 'independent_solver': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
