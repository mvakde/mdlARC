"""Feedback dye circuits: select a recurrent loop and transfer original dyes.

Only Python's standard library and NumPy are used. No files, parent modules,
training labels, random state, or construction metadata are read by solve().
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


D = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _need(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _add(p, direction, distance=1):
    dr, dc = D[direction]
    return p[0] + distance * dr, p[1] + distance * dc


def _components(points):
    remaining, result = set(points), []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], {first}
        while stack:
            p = stack.pop()
            for d in range(4):
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    cells.add(q)
        result.append(cells)
    return result


def _direction(a, b):
    delta = (b[0] - a[0], b[1] - a[1])
    return D.index(delta)


def _t_cells(tip, direction):
    center = _add(tip, (direction + 2) % 4)
    arms = {_add(center, (direction + 1) % 4), _add(center, (direction + 3) % 4)}
    return center, arms | {tip, center}


def _decode(input_grid):
    """Recover components, T junctions, and all first-hit edges from input."""
    a = _grid(input_grid)
    h, w = a.shape
    bg = int(a[0, 0])
    _need(all(int(a[p]) == bg for p in ((0, w - 1), (h - 1, 0), (h - 1, w - 1))),
          "corners must reveal one background")
    parts = _components(map(tuple, np.argwhere(a != bg)))
    edge = [s for s in parts if any(r in (0, h - 1) or c in (0, w - 1) for r, c in s)]
    _need(len(edge) == 1, "one boundary probe required")
    launcher = edge[0]
    boundary = [p for p in launcher if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
    _need(len(boundary) == 1 and 3 <= len(launcher) <= 5, "invalid launcher")
    start = boundary[0]
    initial = 2 if start[0] == 0 else 0 if start[0] == h - 1 else 1 if start[1] == 0 else 3
    line = [_add(start, initial, k) for k in range(len(launcher))]
    _need(set(line) == launcher, "launcher must be straight")
    cap, probe = line[-1], int(a[start])
    ink = int(a[cap])
    _need(len({bg, ink, probe}) == 3 and all(int(a[p]) == probe for p in line[:-1]),
          "launcher tail/cap roles are not distinct")
    objects = []
    for cells in parts:
        if cells == launcher:
            continue
        r0, r1 = min(r for r, c in cells), max(r for r, c in cells)
        c0, c1 = min(c for r, c in cells), max(c for r, c in cells)
        _need(0 < r0 <= r1 < h - 1 and 0 < c0 <= c1 < w - 1, "islands must be interior")
        colors, counts = np.unique([int(a[p]) for p in cells], return_counts=True)
        host = int(colors[int(np.argmax(counts))])
        _need(int(max(counts)) > len(cells) / 2 and host not in (bg, ink, probe),
              "island needs a visible majority fill")
        ink_cells = {p for p in cells if int(a[p]) == ink}
        choices = []
        for tip in sorted(ink_cells):
            sides = [d for d, ok in enumerate((tip[0] == r0, tip[1] == c1,
                                               tip[0] == r1, tip[1] == c0)) if ok]
            for direction in sides:
                center, stencil = _t_cells(tip, direction)
                if not stencil <= cells or not ink_cells <= stencil:
                    continue
                if any(int(a[p]) != ink for p in stencil - {center}):
                    continue
                value = int(a[center])
                explicit = value != ink
                if explicit and value in (bg, probe, host):
                    continue
                if ink_cells != stencil - ({center} if explicit else set()):
                    continue
                if any(int(a[p]) != host for p in cells - stencil):
                    continue
                choices.append(dict(cells=cells, host=host, payload=value if explicit else host,
                                    explicit=explicit, center=center, tip=tip, direction=direction,
                                    bbox=(r0, r1, c0, c1)))
        _need(len(choices) == 1, "island must contain one uniquely readable outward T")
        objects.append(choices[0])
    _need(bool(objects), "no islands")
    # A separating background cell is required even between diagonal neighbors.
    owners = {p: i for i, obj in enumerate(objects) for p in obj['cells']}
    for p, owner in owners.items():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                _need(owners.get((p[0] + dr, p[1] + dc), owner) == owner,
                      "islands touch diagonally")

    def first(origin, direction):
        # Project every original footprint onto the ray; do not mutate islands.
        dr, dc = D[direction]
        best = None
        for index, obj in enumerate(objects):
            for r, c in obj['cells']:
                rr, cc = r - origin[0], c - origin[1]
                distance = rr * dr + cc * dc
                if distance > 0 and rr * dc - cc * dr == 0:
                    if best is None or distance < best[0]:
                        best = distance, index
        return None if best is None else best[1]

    successors = [first(obj['tip'], obj['direction']) for obj in objects]
    entry = first(cap, initial)
    route, cycle = _walk(entry, successors)
    return dict(grid=a, bg=bg, ink=ink, probe=probe, cap=cap, initial=initial,
                launcher=launcher, objects=objects, successors=successors,
                entry=entry, route=route, cycle=cycle)


def _walk(start, successors):
    positions, order = {}, []
    while start is not None and start not in positions:
        positions[start] = len(order)
        order.append(start)
        start = successors[start]
    return order, [] if start is None else order[positions[start]:]


def _paint(info, cycle=None, implicit=False, cascade=False):
    cycle = info['cycle'] if cycle is None else cycle
    out = np.full_like(info['grid'], info['bg'])
    objects = info['objects']
    assignments = {}
    current = {i: obj['host'] for i, obj in enumerate(objects)}
    for source in cycle:
        target = info['successors'][source]
        obj = objects[source]
        value = obj['host'] if implicit else obj['payload']
        if cascade and not obj['explicit']:
            value = current[source]
        assignments[target] = value
        current[target] = value
    for target, value in assignments.items():
        for p in objects[target]['cells']:
            out[p] = value
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Follow the probe; keep its recurrent loop and copy original edge dyes."""
    return _paint(_decode(input_grid))


def _random_cycle(rng, rows, cols, length):
    vertices = [(r, c) for r in range(rows) for c in range(cols)]
    first = vertices[int(rng.integers(len(vertices)))]
    if length == 2:
        neighbors = [q for d in range(4) if (q := _add(first, d)) in vertices]
        return [first, neighbors[int(rng.integers(len(neighbors)))]]
    budget = [4000]

    def search(path, used):
        budget[0] -= 1
        if budget[0] < 0:
            return None
        p = path[-1]
        remaining = length - len(path) + 1
        distance = abs(p[0] - first[0]) + abs(p[1] - first[1])
        if distance > remaining or (remaining - distance) % 2:
            return None
        if len(path) == length:
            return path if distance == 1 else None
        for d in rng.permutation(4):
            q = _add(p, int(d))
            if 0 <= q[0] < rows and 0 <= q[1] < cols and q not in used:
                answer = search(path + [q], used | {q})
                if answer is not None:
                    return answer
        return None

    return search([first], {first})


def _prefix_options(cycle, vertices, count):
    options = []
    cycle_set = set(cycle)
    for index, entry in enumerate(cycle):
        def grow(backwards):
            if len(backwards) == count + 1:
                prefix = list(reversed(backwards[1:]))
                start = prefix[0] if prefix else entry
                if not any(p[0] == start[0] and p[1] < start[1] for p in cycle_set | set(prefix)):
                    options.append((index, prefix))
                return
            for d in range(4):
                q = _add(backwards[-1], d)
                if q in vertices and q not in cycle_set and q not in backwards:
                    grow(backwards + [q])
        grow([entry])
    return options


def _behind(cycle, prefix, nodes):
    path = prefix + cycle
    following = path[1:] + [cycle[0]]
    for source, target in zip(path, following):
        d = _direction(source, target)
        dr, dc = D[d]
        if any((p[0] - target[0]) * dc - (p[1] - target[1]) * dr == 0
               and (p[0] - target[0]) * dr + (p[1] - target[1]) * dc > 0 for p in nodes):
            return True
    return False


def _arrange(rng, rows, cols, cycle, prefix_count, inactive_pair, extras, require_behind):
    vertices = {(r, c) for r in range(rows) for c in range(cols)}
    options = _prefix_options(cycle, vertices, prefix_count)
    for k in rng.permutation(len(options)):
        index, prefix = options[int(k)]
        ordered = cycle[index:] + cycle[:index]
        start = prefix[0] if prefix else ordered[0]
        available = vertices - set(ordered) - set(prefix)
        available = {p for p in available if p[0] != start[0] or p[1] > start[1]}
        pairs = [(a, b) for a in sorted(available) for d in (1, 2)
                 if (b := _add(a, d)) in available] if inactive_pair else [()]
        for j in rng.permutation(len(pairs)):
            pair = pairs[int(j)]
            free = sorted(available - set(pair))
            if len(free) < extras:
                continue
            selected = [free[int(i)] for i in rng.permutation(len(free))[:extras]]
            decoys = list(pair) + selected
            if require_behind and not _behind(ordered, prefix, set(ordered + prefix + decoys)):
                continue
            return dict(rows=rows, cols=cols, cycle=ordered, prefix=prefix,
                        decoys=decoys, inactive_pair=list(pair))
    return None


def _topology(rng, length, prefix_count, inactive_pair, extras, require_behind=False, fallback=False):
    for _ in range(0 if fallback else 90):
        rows = int(rng.integers(2 if length == 2 else 3, 5)) if length <= 6 else 4
        cols = int(rng.integers(2 if length == 2 else 3, 5)) if length <= 6 else 4
        if length + prefix_count + 2 * inactive_pair + extras > rows * cols:
            continue
        cycle = _random_cycle(rng, rows, cols, length)
        if cycle is not None:
            plan = _arrange(rng, rows, cols, cycle, prefix_count, inactive_pair, extras, require_behind)
            if plan is not None:
                return plan
    # A constructive safety net retains every requested topological count.
    fallback_cycles = {
        2: [(1, 1), (1, 2)],
        4: [(1, 1), (1, 2), (2, 2), (2, 1)],
        6: [(1, 1), (1, 2), (2, 2), (3, 2), (3, 1), (2, 1)],
        8: [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (2, 1)],
        10: [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (2, 1), (1, 1)],
    }
    plan = _arrange(rng, 4, 4, fallback_cycles[length], prefix_count, inactive_pair, extras, require_behind)
    _need(plan is not None, "constructive topology failed")
    return plan


def _axis(rng, count, launching=False):
    sizes = list(map(int, rng.integers(4, 7, size=count)))
    gaps = list(map(int, rng.integers(1, 4, size=count - 1)))
    before = int(rng.integers(5, 8) if launching else rng.integers(1, 4))
    after = int(rng.integers(1, 4))
    # Shrinking is bounded, with minimum total <=25 for four tracks.
    while sum(sizes) + sum(gaps) + before + after > 30:
        choices = [('s', i) for i, v in enumerate(sizes) if v > 4]
        choices += [('g', i) for i, v in enumerate(gaps) if v > 1]
        choices += [('b', 0)] if before > (5 if launching else 1) else []
        choices += [('a', 0)] if after > 1 else []
        kind, i = choices[int(rng.integers(len(choices)))]
        if kind == 's':
            sizes[i] -= 1
        elif kind == 'g':
            gaps[i] -= 1
        elif kind == 'b':
            before -= 1
        else:
            after -= 1
    starts, anchors, position = [], [], before
    for i, size in enumerate(sizes):
        starts.append(position)
        anchors.append(position + int(rng.integers(1, size - 1)))
        position += size + (gaps[i] if i < len(gaps) else 0)
    return sizes, starts, anchors, position + after


def _footprint(rng, bounds, anchor, direction):
    r0, r1, c0, c1 = bounds
    tip = ((r0, anchor[1]), (anchor[0], c1), (r1, anchor[1]), (anchor[0], c0))[direction]
    center, stencil = _t_cells(tip, direction)
    cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    protected = stencil | {(anchor[0], c) for c in range(c0, c1 + 1)}
    protected |= {(r, anchor[1]) for r in range(r0, r1 + 1)}
    # Random corner clips and concave edge notches, while preserving both axes
    # of possible contact and the complete outward T.
    for _ in range(int(rng.integers(0, 6))):
        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        if rng.random() < .65:
            p = corners[int(rng.integers(4))]
            cut = {p}
            if rng.random() < .35:
                cut.add((p[0] + (1 if p[0] == r0 else -1), p[1]))
        else:
            side = int(rng.integers(4))
            p = ((r0, int(rng.integers(c0, c1 + 1))),
                 (int(rng.integers(r0, r1 + 1)), c1),
                 (r1, int(rng.integers(c0, c1 + 1))),
                 (int(rng.integers(r0, r1 + 1)), c0))[side]
            cut = {p}
        candidate = cells - cut
        if not cut & protected and len(candidate) >= 12 and len(_components(candidate)) == 1:
            cells = candidate
    return cells, tip, center, stencil


def _make_pair(rng, palette, length, prefix_count, inactive_pair, extras, explicit_count,
               demonstrate=False, fallback=False):
    plan = _topology(rng, length, prefix_count, inactive_pair, extras, demonstrate, fallback)
    cycle, prefix, decoys = plan['cycle'], plan['prefix'], plan['decoys']
    nodes = cycle + prefix + decoys
    bg, ink, probe = map(int, palette[:3])
    pool = list(map(int, palette[3:]))
    original = {p: int(rng.choice(pool[:-1])) for p in nodes}
    dye_sequence = list(map(int, rng.permutation(pool[:-1])))
    for i, p in enumerate(cycle):
        original[p] = dye_sequence[i % len(dye_sequence)]
    # Repeated fills do not identify an island: a matching feeder/decoy is erased.
    if prefix:
        original[prefix[0]] = original[cycle[0]]
    if decoys:
        original[decoys[0]] = original[cycle[0]]
    following = {p: cycle[(i + 1) % length] for i, p in enumerate(cycle)}
    route = prefix + [cycle[0]]
    following.update(zip(route, route[1:]))
    pair = plan['inactive_pair']
    if pair:
        following[pair[0]], following[pair[1]] = pair[1], pair[0]
    directions = {p: _direction(p, following[p]) if p in following else int(rng.integers(4)) for p in nodes}
    payloads = dict(original)
    explicit = set(cycle[int(i)] for i in rng.permutation(length)[:explicit_count])
    for i, p in enumerate(sorted(explicit)):
        choices = [c for c in pool if c not in (original[p], original[following[p]])]
        payloads[p] = pool[-1] if i == 0 else int(rng.choice(choices))
    # Off-route payloads obey precisely the same interpretation but do not act.
    for p in prefix + decoys:
        if rng.random() < .25:
            explicit.add(p)
            payloads[p] = int(rng.choice([c for c in pool if c != original[p]]))
    hs, rs, ra, h = _axis(rng, plan['rows'])
    ws, cs, ca, w = _axis(rng, plan['cols'], launching=True)
    inp, out = np.full((h, w), bg, np.int8), np.full((h, w), bg, np.int8)
    footprints = {}
    for p in nodes:
        r, c = p
        bounds = rs[r], rs[r] + hs[r] - 1, cs[c], cs[c] + ws[c] - 1
        cells, tip, center, stencil = _footprint(rng, bounds, (ra[r], ca[c]), directions[p])
        footprints[p] = cells
        for cell in cells:
            inp[cell] = original[p]
        for cell in stencil:
            inp[cell] = ink
        if p in explicit:
            inp[center] = payloads[p]
    start = prefix[0] if prefix else cycle[0]
    launch_row = ra[start[0]]
    tail_length = int(rng.integers(2, min(5, cs[0] - 1)))
    inp[launch_row, :tail_length] = probe
    inp[launch_row, tail_length] = ink
    # Construction labels come from the planned cyclic graph, not solve().
    for source in cycle:
        for cell in footprints[following[source]]:
            out[cell] = payloads[source]
    if rng.random() < .5:
        inp, out = np.fliplr(inp), np.fliplr(out)
    turn = int(rng.integers(4))
    return {'input': np.rot90(inp, turn).copy(), 'output': np.rot90(out, turn).copy()}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if not _integer(value) or not lo <= value <= hi:
            raise ValueError('counts must be integers in their legal ranges')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    base = list(map(int, rng.permutation(10)))
    fixed_palette = bool(rng.random() < .3)
    puzzle, seen = {'train': [], 'test': []}, set()
    training_lengths = [2, 4] + [int(rng.choice([2, 4, 6])) for _ in range(num_train - 2)]
    test_lengths = [v for v in (6, 8, 10) if v > max(training_lengths)]
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            demo = split == 'train' and index == 1
            simple = split == 'train' and index == 0
            length = training_lengths[index] if split == 'train' else int(rng.choice(test_lengths))
            prefix = 0 if simple else int(rng.integers(1, 3) if demo else rng.integers(0, 3))
            inactive_pair = bool(demo or (not simple and rng.random() < .55))
            extras = int(rng.integers(0, 2))
            explicit = 0 if simple else int(rng.integers(1, 3) if demo else rng.integers(0, min(3, length) + 1))
            palette = base if fixed_palette or simple else list(map(int, rng.permutation(10)))
            for attempt in range(12):
                pair = _make_pair(rng, palette, length, prefix, inactive_pair, extras, explicit,
                                  demonstrate=demo, fallback=attempt == 11)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('distinct input construction exhausted')
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _facts(info):
    objects, cycle = info['objects'], info['cycle']
    all_cycles, covered = [], set()
    for start in range(len(objects)):
        _, found = _walk(start, info['successors'])
        if found and not set(found) & covered:
            all_cycles.append(found)
            covered.update(found)
    loop_cells = set().union(*(objects[i]['cells'] for i in cycle)) if cycle else set()
    clean_colors = {obj['host'] for obj in objects}
    behind = False
    for source in info['route']:
        target = info['successors'][source]
        if target is None:
            continue
        origin, direction = objects[source]['tip'], objects[source]['direction']
        dr, dc = D[direction]
        distances = []
        for i, obj in enumerate(objects):
            hits = [(r - origin[0]) * dr + (c - origin[1]) * dc for r, c in obj['cells']
                    if (r - origin[0]) * dc - (c - origin[1]) * dr == 0
                    and (r - origin[0]) * dr + (c - origin[1]) * dc > 0]
            if hits:
                distances.append((min(hits), i))
        behind |= len(distances) >= 2
    nonrect = sum(len(o['cells']) != (o['bbox'][1] - o['bbox'][0] + 1)
                  * (o['bbox'][3] - o['bbox'][2] + 1) for o in objects)
    return dict(length=len(cycle), prefix=len(info['route']) - len(cycle),
                decoys=len(objects) - len(info['route']),
                inactive_cycles=len(all_cycles) - bool(cycle), all_cycles=all_cycles,
                active_explicit=sum(objects[i]['explicit'] for i in cycle),
                fresh=any(objects[i]['explicit'] and objects[i]['payload'] not in clean_colors for i in cycle),
                behind=behind, nonrect=nonrect, loop_cells=loop_cells,
                directions={o['direction'] for o in objects},
                shape=info['grid'].shape, palette=(info['bg'], info['ink'], info['probe']))


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'illegal counts')
    seen, records = set(), {'train': [], 'test': []}
    cascade_witness = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            _need(key not in seen, 'duplicate input')
            seen.add(key)
            info = _decode(inp)
            facts = _facts(info)
            _need(facts['length'] >= 2, 'generated probe must reach a cycle')
            _need(np.array_equal(_paint(info), out), 'incorrect recurrent transfer')
            _need(out.shape == inp.shape, 'output shape changed')
            support = set(map(tuple, np.argwhere(out != info['bg'])))
            _need(support == facts['loop_cells'], 'output must preserve exactly the recurrent footprints')
            _need(not np.any(out == info['ink']) and not np.any(out == info['probe']),
                  'annotations or probe survived')
            records[split].append(facts)
            if split == 'train':
                cascade_witness |= not np.array_equal(_paint(info, cascade=True), out)
    train = records['train']
    _need(any(f['length'] == 2 and f['active_explicit'] == 0 for f in train), 'missing implicit swap')
    _need(cascade_witness, 'training does not distinguish original colors from cascading')
    _need(any(f['fresh'] and f['prefix'] >= 1 and f['inactive_cycles'] >= 1 and f['behind'] for f in train),
          'missing combined payload, transient, inactive-loop, and first-hit demonstration')
    _need(all(f['length'] > max(g['length'] for g in train) for f in records['test']),
          'test does not extend demonstrated cycle length')


def self_test():
    """Bounded 42-seed checks; retain hashes/features, never a grid corpus."""
    signatures, structures, shapes, lengths, prefixes, decoys, directions = set(), set(), set(), set(), set(), set(), set()
    palettes, explicit_counts = set(), set()
    nonrect = inactive = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                before = inp.copy()
                _need(np.array_equal(solve(inp), out) and np.array_equal(inp, before), 'solver mismatch or mutation')
                info = _decode(inp)
                f = _facts(info)
                digest.update(str(inp.shape).encode())
                digest.update(inp.tobytes())
                shapes.add(f['shape'])
                lengths.add(f['length'])
                prefixes.add(f['prefix'])
                decoys.add(f['decoys'])
                directions.update(f['directions'])
                palettes.add(f['palette'])
                explicit_counts.add(f['active_explicit'])
                nonrect += bool(f['nonrect'])
                inactive += f['inactive_cycles'] > 0
                rename = {}
                normalized = bytes(rename.setdefault(int(c), len(rename)) for c in inp.ravel())
                structures.add((inp.shape, hashlib.sha256(normalized).digest()))
                if seed < 3:
                    _need(np.array_equal(solve(np.rot90(inp)), np.rot90(out)), 'rotation equivariance failed')
                    permutation = np.array([7, 3, 8, 1, 9, 0, 6, 2, 5, 4], dtype=np.int8)
                    _need(np.array_equal(solve(permutation[inp]), permutation[out]), 'palette equivariance failed')
        signatures.add(digest.digest())
        if seed == 0:
            repeat = generate_puzzle(seed)
            _need(all(_key(a[k]) == _key(b[k]) for s in puzzle for a, b in zip(puzzle[s], repeat[s])
                      for k in ('input', 'output')), 'nondeterministic puzzle')
            broken = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
            broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
            try:
                validate_generated_puzzle(broken)
            except AssertionError:
                pass
            else:
                raise AssertionError('validator accepted corrupt output')
    _need(len(signatures) == 42 and len(structures) >= 180, 'insufficient structural variation')
    _need(len(shapes) >= 25 and lengths == {2, 4, 6, 8, 10}, 'collapsed geometry or loop sizes')
    _need(prefixes == {0, 1, 2} and len(decoys) >= 3 and directions == set(range(4)), 'collapsed topology')
    _need(len(palettes) >= 20 and {0, 1, 2, 3} <= explicit_counts, 'collapsed palette/payload roles')
    _need(nonrect >= 100 and inactive >= 42, 'missing footprint or inactive-loop variety')
    # Directly exercise the fallback for every combination the planner requests.
    fallback_rng = np.random.default_rng(90210)
    for length in (2, 4, 6, 8, 10):
        for prefix in (0, 1, 2):
            for pair in (False, True):
                plan = _topology(fallback_rng, length, prefix, pair, 1, fallback=True)
                _need(len(plan['cycle']) == length and len(plan['prefix']) == prefix
                      and len(plan['decoys']) == 1 + 2 * pair, 'fallback changed counts')
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(4000 + 10 * a + b, a, b)
            _need(len(p['train']) == a and len(p['test']) == b, 'count request ignored')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), color_normalized_inputs=len(structures),
                shapes=len(shapes), cycle_lengths=sorted(lengths), prefix_lengths=sorted(prefixes),
                decoy_counts=sorted(decoys), palettes=len(palettes), nonrectangular_pairs=nonrect,
                pairs_with_inactive_cycles=inactive)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
