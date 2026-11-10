"""Counterweight walk: rank-paired square instructions cancel before drawing.

Only NumPy and the standard library are used. No files, generation state, or
training labels are consulted by solve(). Tests and generation keep no traces.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _tile(side, color, bg, kind):
    a = np.full((side, side), color, dtype=np.int8)
    if kind == 0:  # Hollow is the negative queue; solid is positive.
        a[1:-1, 1:-1] = bg
    return a


def _parse(value):
    """Recover the two queues and drawing origin from pixels alone."""
    a = _grid(value)
    height, width = a.shape
    assert height >= 5 and width >= 3
    values, counts = np.unique(a[0], return_counts=True)
    bg = int(values[np.argmax(counts)])
    marks = np.flatnonzero(a[0] != bg)
    assert len(marks) == 1 and np.all(a[1] == bg), "missing isolated top-row origin"
    start = int(marks[0])
    marker = int(a[0, start])
    todo = set((int(r), int(c)) for r, c in np.argwhere(a != bg))
    todo.remove((0, start))
    objects = []
    while todo:
        first = min(todo)
        todo.remove(first)
        stack, component = [first], [first]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
                    component.append(p)
        top = min(p[0] for p in component)
        left = min(p[1] for p in component)
        bottom = max(p[0] for p in component) + 1
        right = max(p[1] for p in component) + 1
        side = bottom - top
        assert top >= 2 and 3 <= side <= 6 and right-left == side
        block = a[top:bottom, left:right]
        color = int(block[0, 0])
        assert color not in (bg, marker), "origin ink is reserved"
        kind = int(np.all(block == color))
        assert np.array_equal(block, _tile(side, color, bg, kind)), "malformed square"
        # Full rectangular gap, including diagonal neighbors; holes are empty.
        r0, c0 = max(0, top-1), max(0, left-1)
        around = a[r0:min(height, bottom+1), c0:min(width, right+1)].copy()
        around[top-r0:bottom-r0, left-c0:right-c0] = bg
        assert np.all(around == bg), "tiles must have a background gap"
        objects.append(dict(top=top, left=left, side=side, color=color, kind=kind))
    assert 2 <= len(objects) <= 10
    queues = []
    for kind in (0, 1):
        queue = sorted((o for o in objects if o['kind'] == kind), key=lambda o: o['top'])
        assert 1 <= len(queue) <= 7
        assert len({o['top'] for o in queue}) == len(queue), "same-class top-row tie"
        queues.append(queue)
    assert max(map(len, queues)) + 1 <= height
    return dict(shape=a.shape, bg=bg, marker=marker, start=start,
                objects=objects, queues=queues)


def _pairs(queues):
    return [(queues[0][i] if i < len(queues[0]) else None,
             queues[1][i] if i < len(queues[1]) else None)
            for i in range(max(map(len, queues)))]


def _draw(info, pairs=None, mode='normal'):
    """None means an alternative walk would leave the visible canvas."""
    if pairs is None:
        pairs = _pairs(info['queues'])
    height, width = info['shape']
    if len(pairs) + 1 > height:
        return None
    out = np.full(info['shape'], info['bg'], dtype=np.int8)
    out[0, info['start']] = info['marker']
    col = info['start']
    for row, (hollow, solid) in enumerate(pairs, 1):
        h = 0 if hollow is None else hollow['side']
        s = 0 if solid is None else solid['side']
        delta = s-h
        if mode == 'area':
            delta = s*s-h*h
        elif mode == 'no_cancel' and delta:
            delta = s if delta > 0 else -h
        if mode == 'no_chain':
            col = info['start']
        end = col + delta
        if not 0 <= end < width:
            return None
        winner = solid if delta > 0 else hollow
        color = info['marker'] if delta == 0 else winner['color']
        out[row, min(col, end):max(col, end)+1] = color
        col = end
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Interpret squares as opposed displacement instructions, then draw."""
    info = _parse(input_grid)
    answer = _draw(info)
    if answer is None:
        raise ValueError("counterweight walk leaves the input canvas")
    return answer


def _alternative_outputs(info):
    """Finite, explicitly documented shortcut checks, not an external oracle."""
    q = info['queues']
    alternatives = {}
    for name, key, reverse in (
            ('horizontal', lambda o: (o['left'], o['top']), False),
            ('horizontal_reverse', lambda o: (o['left'], o['top']), True),
            ('vertical_reverse', lambda o: o['top'], True)):
        queues = [sorted(queue, key=key, reverse=reverse) for queue in q]
        alternatives[name] = _draw(info, _pairs(queues))
    # Parent B's tempting greedy scan never returns to fill an earlier hole.
    greedy = [[None, None]]
    for obj in sorted(info['objects'], key=lambda o: (o['top'], o['left'])):
        k = obj['kind']
        if greedy[-1][k] is not None:
            greedy.append([None, None])
        greedy[-1][k] = obj
    alternatives['greedy'] = _draw(info, greedy)
    # Pair each successive hollow with the nearest still unused solid.
    available = list(q[1])
    nearest = []
    for hollow in q[0]:
        if available:
            solid = min(available, key=lambda o: (
                abs(2*o['top']+o['side']-2*hollow['top']-hollow['side'])
                + abs(2*o['left']+o['side']-2*hollow['left']-hollow['side']),
                o['top'], o['left']))
            available.remove(solid)
        else:
            solid = None
        nearest.append((hollow, solid))
    nearest.extend((None, solid) for solid in available)
    alternatives['nearest'] = _draw(info, nearest)
    for name in ('no_chain', 'no_cancel', 'area'):
        alternatives[name] = _draw(info, mode=name)
    alternatives['hollow_only'] = _draw(info, [(h, None) for h in q[0]])
    alternatives['solid_only'] = _draw(info, [(None, s) for s in q[1]])
    return alternatives


def _different(answer, alternative):
    return alternative is None or not np.array_equal(answer, alternative)


def _sizes(rng, nh, ns, demo):
    if demo is not None:
        if demo == 0:
            # Expose all four side lengths, unequal opposite residuals, a tie.
            h, s = [3, 5, 4], [6, 3, 4]
            if rng.integers(2):
                h, s = s, h
        else:
            low1, low2, tie = (int(x) for x in rng.integers(3, 6, size=3))
            h, s = [low1, low2+1, tie], [low1+1, low2, tie]
        order = rng.permutation(3)
        h, s = [h[int(i)] for i in order], [s[int(i)] for i in order]
        if nh > 3:
            h.append(int(rng.integers(3, 7)))
        if ns > 3:
            s.append(int(rng.integers(3, 7)))
        return h, s
    # Bound retries only for excessive net horizontal excursion, never packing.
    for _ in range(64):
        h = [int(x) for x in rng.integers(3, 7, size=nh)]
        s = [int(x) for x in rng.integers(3, 7, size=ns)]
        deltas = [(s[i] if i < ns else 0)-(h[i] if i < nh else 0)
                  for i in range(max(nh, ns))]
        positions = np.cumsum([0] + deltas)
        if int(np.ptp(positions)) <= 27:
            return h, s
    # Always feasible: at most six unmatched side-three squares => span <=18.
    return [3]*nh, [3]*ns


def _partitions(total, prefix=()):
    if total == 0:
        yield prefix
    elif len(prefix) < 4:
        for count in range(1, min(4, total)+1):
            yield from _partitions(total-count, prefix+(count,))


def _positions(rng, items, span, ranks):
    """Construct a feasible shelf packing, then relocate over all free sites."""
    order = [int(i) for i in rng.permutation(len(items))]
    # Equal reservation footprints allow whole payloads to be assigned to
    # same-class top ranks without ever overlapping differently sized tiles.
    footprint = max(item['side'] for item in items)
    candidates = []
    for partition in _partitions(len(items)):
        offset, shelves = 0, []
        for count in partition:
            shelves.append(order[offset:offset+count])
            offset += count
        height = 2 + len(shelves)-1 + sum(footprint+len(shelf)-1 for shelf in shelves)
        width = max(len(shelf)*(footprint+1)-1 for shelf in shelves)
        if height <= 30 and width <= 30:
            candidates.append((height, width, shelves))
    # With <=10 tiles of side <=6, a 4/3/3 partition fits within 29x27.
    assert candidates
    min_h, min_w, shelves = candidates[int(rng.integers(len(candidates)))]
    height = int(rng.integers(max(min_h, ranks+1, 10), 31))
    width = int(rng.integers(max(min_w, span+1, 12), 31))
    offsets = [rng.permutation(len(shelf)) for shelf in shelves]
    shelf_heights = [footprint+len(shelf)-1 for shelf in shelves]
    slack = height-2-sum(shelf_heights)-(len(shelves)-1)
    gaps = rng.multinomial(slack, np.full(len(shelves)+1, 1/(len(shelves)+1)))
    sites = [None]*len(items)
    top = 2+int(gaps[0])
    for j, (shelf, ds) in enumerate(zip(shelves, offsets)):
        used_width = len(shelf)*(footprint+1)-1
        spaces = rng.multinomial(width-used_width,
                                np.full(len(shelf)+1, 1/(len(shelf)+1)))
        left = int(spaces[0])
        for k, (i, d) in enumerate(zip(shelf, ds)):
            sites[i] = (top+int(d), left)
            left += footprint+1+int(spaces[k+1])
        top += shelf_heights[j]+1+int(gaps[j+1])
    for kind in (0, 1):
        indices = [i for i, item in enumerate(items) if item['kind'] == kind]
        ordered_sites = sorted((sites[i] for i in indices), key=lambda p: p[0])
        for i, site in zip(indices, ordered_sites):
            sites[i] = site
    # Initial placement has globally unique tops. Relocation allows cross-class
    # top ties but preserves a unique top row within each individual queue.
    for _ in range(2):
        for index in rng.permutation(len(items)):
            index = int(index)
            size = items[index]['side']
            rr, cc = np.indices((height-size-1, width-size+1))
            rr += 2
            legal = np.ones(rr.shape, dtype=bool)
            for j, (r, c) in enumerate(sites):
                if j == index:
                    continue
                other = items[j]['side']
                legal &= ((rr+size < r) | (r+other < rr)
                          | (cc+size < c) | (c+other < cc))
                if items[j]['kind'] == items[index]['kind']:
                    legal &= rr > r if j < index else rr < r
            choices = np.argwhere(legal)
            assert len(choices), "current site is always a feasible fallback"
            r, c = choices[int(rng.integers(len(choices)))]
            sites[index] = (int(r)+2, int(c))
    return (height, width), sites


def _make_pair(rng, nh, ns, demo=None):
    h, s = _sizes(rng, nh, ns, demo)
    deltas = [(s[i] if i < ns else 0)-(h[i] if i < nh else 0)
              for i in range(max(nh, ns))]
    endpoints = np.cumsum([0]+deltas)
    span = int(np.ptp(endpoints))
    roles = [int(x) for x in rng.permutation(10)]
    bg, marker = roles[:2]
    palette = roles[2:]
    count = nh+ns
    if demo is not None:
        colors = palette[:count]
        # One ink visibly moves both ways: geometry and size, not ink, decide.
        positive = next(i for i in range(3) if s[i] > h[i])
        negative = next(i for i in range(3) if h[i] > s[i])
        colors[nh+positive] = colors[negative]
    elif rng.random() < 0.45 and count <= 8:
        colors = palette[:count]
    else:
        pool = palette[:int(rng.integers(1, 9))]
        colors = [int(x) for x in rng.choice(pool, size=count)]
    items = [dict(kind=kind, side=size, color=colors[index])
             for index, (kind, size) in enumerate([(0, x) for x in h]+[(1, x) for x in s])]
    for _ in range(64):
        shape, sites = _positions(rng, items, span, max(nh, ns))
        grid = np.full(shape, bg, dtype=np.int8)
        for item, (top, left) in zip(items, sites):
            side = item['side']
            grid[top:top+side, left:left+side] = _tile(side, item['color'], bg, item['kind'])
        # Verify the visible queues before choosing a feasible drawing origin.
        grid[0, 0] = marker
        info = _parse(grid)
        actual_pairs = _pairs(info['queues'])
        actual_deltas = [(0 if so is None else so['side'])-(0 if ho is None else ho['side'])
                         for ho, so in actual_pairs]
        ends = np.cumsum([0]+actual_deltas)
        lo, hi = -int(ends.min()), shape[1]-1-int(ends.max())
        if lo > hi:
            continue
        start = int(rng.integers(lo, hi+1))
        grid[0, 0] = bg
        grid[0, start] = marker
        info = _parse(grid)
        answer = _draw(info)
        assert answer is not None
        if demo is not None:
            d = actual_deltas
            if not (0 in d and any(x > 0 for x in d) and any(x < 0 for x in d)):
                continue
            if not all(_different(answer, a) for a in _alternative_outputs(info).values()):
                continue
            directions = {}
            for ho, so in actual_pairs:
                if ho is not None and so is not None and ho['side'] != so['side']:
                    winner = so if so['side'] > ho['side'] else ho
                    directions.setdefault(winner['color'], set()).add(winner['kind'])
            if not any(len(v) == 2 for v in directions.values()):
                continue
        return dict(input=grid, output=answer)
    raise RuntimeError("bounded demonstration search exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer('seed', seed, 0)
    nt = _integer('num_train', num_train, 2, 10)
    nq = _integer('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    result = dict(train=[], test=[])
    seen = set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            demo = index if split == 'train' and index < 2 else None
            if demo is not None:
                nh, ns = (4, 3) if index == 0 else (3, 4)
            else:
                largest = int(rng.integers(2, 5) if split == 'train' else rng.integers(5, 8))
                other = int(rng.integers(1, min(largest, 10-largest)+1))
                nh, ns = (largest, other) if rng.integers(2) else (other, largest)
            pair = _make_pair(rng, nh, ns, demo)
            # Ten cyclic palette assignments suffice: at most nine earlier
            # training inputs or three earlier tests can have these counts.
            for _ in range(10):
                key = _key(pair['input'])
                if key not in seen:
                    break
                pair = {k: ((v.astype(np.int16)+1) % 10).astype(np.int8)
                        for k, v in pair.items()}
            assert key not in seen
            seen.add(key)
            result[split].append(pair)
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, training, test_ranks = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            assert np.array_equal(out, solve(inp.copy())), "incorrect walk"
            ranks = max(map(len, info['queues']))
            # Row shape, ink ownership, conservation of displacement and cells.
            col = info['start']
            ink_cells = 1
            for row, (h, s) in enumerate(_pairs(info['queues']), 1):
                delta = (0 if s is None else s['side'])-(0 if h is None else h['side'])
                end = col+delta
                nonbg = np.flatnonzero(out[row] != info['bg'])
                assert nonbg.tolist() == list(range(min(col, end), max(col, end)+1))
                expected_ink = info['marker'] if delta == 0 else (s if delta > 0 else h)['color']
                assert np.all(out[row, nonbg] == expected_ink)
                col = end
                ink_cells += abs(delta)+1
            assert np.all(out[ranks+1:] == info['bg'])
            assert int(np.count_nonzero(out != info['bg'])) == ink_cells
            if split == 'train':
                training.append((info, out))
            else:
                test_ranks.append(ranks)
    assert min(test_ranks) > max(max(map(len, a['queues'])) for a, _ in training)
    pairs = [p for info, _ in training for p in _pairs(info['queues'])]
    assert any(h is None for h, s in pairs) and any(s is None for h, s in pairs)
    assert any(h is not None and s is not None and h['side'] == s['side'] for h, s in pairs)
    assert {o['side'] for info, _ in training for o in info['objects']} == {3, 4, 5, 6}
    for name in _alternative_outputs(training[0][0]):
        assert any(_different(out, _alternative_outputs(info)[name]) for info, out in training), name
    moving_roles = {}
    for h, s in pairs:
        if h is not None and s is not None and h['side'] != s['side']:
            winner = s if s['side'] > h['side'] else h
            moving_roles.setdefault(winner['color'], set()).add(winner['kind'])
    assert any(len(v) == 2 for v in moving_roles.values()), "color-direction shortcut"


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Stream 42 seeds; retain only bounded signatures and small measurements."""
    digests, layouts, shapes, counts, backgrounds, programs, sizes = (set() for _ in range(7))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digests.add(_digest(puzzle))
        assert _digest(puzzle) == _digest(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _parse(pair['input'])
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
                q = info['queues']
                counts.add(tuple(map(len, q)))
                shapes.add(info['shape'])
                backgrounds.add(info['bg'])
                sizes.update(o['side'] for o in info['objects'])
                layouts.add(tuple(sorted((o['top'], o['left'], o['side'], o['kind'])
                                         for o in info['objects'])))
                programs.add(tuple((0 if s is None else s['side'])-(0 if h is None else h['side'])
                                   for h, s in _pairs(q)))
    assert len(digests) == 42
    assert len(layouts) >= 150 and len(shapes) >= 40
    assert len(counts) >= 15 and len(programs) >= 100
    assert backgrounds == set(range(10)) and sizes == {3, 4, 5, 6}
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(2026, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = generate_puzzle(314159, 2, 1)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, distinct_puzzles=len(digests), layouts=len(layouts),
                input_shapes=len(shapes), count_patterns=len(counts),
                displacement_programs=len(programs), sizes=sorted(sizes),
                backgrounds=sorted(backgrounds))
