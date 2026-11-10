"""Coupled free-flight rays and individually traceable cables.

No files, parent modules, generation state, or seed are used by solve().
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np

_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
_CHECKER = ~_DIAMOND


def _add(p, d, length=1):
    dr, dc = _DIRS[d]
    return p[0] + length * dr, p[1] + length * dc


def _square(p):
    return {(p[0] + r, p[1] + c) for r in (-1, 0, 1) for c in (-1, 0, 1)}


def _inside(p, shape, margin=0):
    return margin <= p[0] < shape[0] - margin and margin <= p[1] < shape[1] - margin


def _components(cells):
    unseen, result = set(cells), []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        stack, found = [p], {p}
        while stack:
            for d in range(4):
                q = _add(stack[-1], d)
                if q in unseen:
                    unseen.remove(q)
                    found.add(q)
                    stack.append(q)
                    break
            else:
                stack.pop()
        result.append(found)
    return result


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer) and np.all((g >= 0) & (g <= 9))
    return g.astype(np.int8, copy=False)


def _trace_field(wires, centers, inward):
    """Trace the actual cell graph; four-way vertices preserve direction."""
    graph = {p: {d: _add(p, d) for d in range(4) if _add(p, d) in wires} for p in wires}
    stub_owner = {_add(p, d, 2): i for i, (p, d) in enumerate(zip(centers, inward))}
    assert len(stub_owner) == len(centers)
    for i, (p, d) in enumerate(zip(centers, inward)):
        sq = _square(p)
        contacts = {q for a in sq for k in range(4) if (q := _add(a, k)) in wires}
        assert contacts == {_add(p, d, 2)}, 'plug has extra or missing wire contact'
    crossings = set()
    for p, edges in graph.items():
        if p in stub_owner:
            assert len(edges) == 1
        else:
            assert len(edges) in (2, 4), 'broken wire or unmarked branch'
            if len(edges) == 4:
                crossings.add(p)
    assert all(not any(_add(p, d) in crossings for d in range(4)) for p in crossings)
    paired, routes, covered = {}, [], set()
    for i, center in enumerate(centers):
        if i in paired:
            continue
        p, direction = _add(center, inward[i], 2), inward[i]
        path, states = [], set()
        while True:
            assert (p, direction) not in states, 'cycle in cable graph'
            states.add((p, direction))
            path.append(p)
            if p in stub_owner and stub_owner[p] != i:
                j = stub_owner[p]
                assert j not in paired
                break
            edges = graph[p]
            if len(edges) == 4:
                outgoing = direction
            else:
                options = [d for d in edges if d != (direction + 2) % 4]
                assert len(options) == 1
                outgoing = options[0]
            p, direction = edges[outgoing], outgoing
        assert len(set(path)) == len(path), 'self-crossing cable excluded'
        paired[i], paired[j] = j, i
        route = {'ends': (i, j), 'cells': set(path), 'path': path}
        routes.append(route)
        covered.update(path)
    assert covered == wires, 'unattached cable fragment'
    for p in crossings:
        assert sum(p in route['cells'] for route in routes) == 2
    return paired, routes, crossings


def _decode(input_grid):
    g = _grid(input_grid)
    h, w = g.shape
    bg = int(g[0, 0])
    assert all(int(g[r, c]) == bg for r, c in ((0, w - 1), (h - 1, 0), (h - 1, w - 1)))
    boundary = {(r, c) for r in range(h) for c in range(w)
                if (r in (0, h - 1) or c in (0, w - 1)) and g[r, c] != bg}
    assert len(boundary) == 1, 'unique boundary launcher required'
    start = next(iter(boundary))
    initial = 2 if start[0] == 0 else 0 if start[0] == h - 1 else 1 if start[1] == 0 else 3
    signal = int(g[start])
    tail, p = [], start
    while _inside(p, g.shape) and g[p] == signal:
        tail.append(p)
        p = _add(p, initial)
    cap = p
    assert 2 <= len(tail) <= 4 and _inside(cap, g.shape)
    cap_color = int(g[cap])
    assert len({bg, signal, cap_color}) == 3
    assert {tuple(map(int, x)) for x in np.argwhere(g == signal)} == set(tail)
    assert np.count_nonzero(g == cap_color) == 1
    structural = set(map(int, np.unique(g))) - {bg, signal, cap_color}
    candidates = []
    for color in sorted(structural):
        components = _components(map(tuple, np.argwhere(g == color)))
        if len(components) < 2:
            continue
        centers = []
        for component in components:
            r0, c0 = min(p[0] for p in component), min(p[1] for p in component)
            center = (r0 + 1, c0 + 1)
            if component != _square(center):
                break
            centers.append(center)
        else:
            candidates.append((color, centers))
    assert len(candidates) == 1, 'unique solid plug role required'
    plug, centers = candidates[0]
    assert len(centers) % 2 == 0
    assert all(_inside(p, g.shape, 2) for p in centers)
    wire_colors = structural - {plug}
    assert 1 <= len(wire_colors) <= 2
    wires = {tuple(map(int, p)) for p in np.argwhere(np.isin(g, sorted(wire_colors)))}
    assert all(_inside(p, g.shape, 1) for p in wires)
    owner = {}
    for i, p in enumerate(centers):
        for q in _square(p):
            assert q not in owner
            owner[q] = i
    inward = []
    for p in centers:
        sides = [d for d in range(4) if _add(p, d, 2) in wires]
        assert len(sides) == 1
        inward.append(sides[0])
    paired, routes, crossings = _trace_field(wires, centers, inward)
    ordinary_colors = {int(g[p]) for p in wires - crossings}
    assert len(ordinary_colors) == 1
    carrier = next(iter(ordinary_colors))
    cross_colors = {int(g[p]) for p in crossings}
    assert len(cross_colors) <= 1 and carrier not in cross_colors
    assert {carrier} | cross_colors == wire_colors
    for i, p in enumerate(centers):
        assert not any(_add(q, d) in owner and owner[_add(q, d)] != i
                       for q in _square(p) for d in range(4))
    route_of = {i: k for k, route in enumerate(routes) for i in route['ends']}
    return dict(grid=g, bg=bg, signal=signal, cap_color=cap_color, plug=plug,
                carrier=carrier, cross=next(iter(cross_colors), None), tail=tail, cap=cap,
                initial=initial, centers=centers, inward=inward, owner=owner,
                wires=wires, paired=paired, routes=routes, crossings=crossings, route_of=route_of)


def _propagate(info):
    """Flight scans original plugs, then switches to the paired cable."""
    shape, owner = info['grid'].shape, info['owner']
    origin, d = info['cap'], info['initial']
    flight = set(info['tail']) | {origin}
    events, used, segments = [], set(), []
    for _ in range(len(info['routes']) + 1):
        segment, p = [], _add(origin, d)
        while _inside(p, shape) and p not in owner:
            segment.append(p)
            p = _add(p, d)
        flight.update(segment)
        segments.append(segment)
        if not _inside(p, shape):
            return flight, events, segments
        entry = owner[p]
        center = info['centers'][entry]
        assert p == _add(center, (d + 2) % 4), 'off-center plug collision'
        assert info['inward'][entry] == d, 'flight enters plug from wrong side'
        route_index = info['route_of'][entry]
        assert route_index not in used, 'reactivation or flight cycle'
        used.add(route_index)
        exit_index = info['paired'][entry]
        events.append((entry, exit_index, route_index))
        d = (info['inward'][exit_index] + 2) % 4
        # Skip the entire three-cell exit footprint, rendered separately.
        origin = _add(info['centers'][exit_index], d)
    raise AssertionError('unterminated signal')


def _render(info, flight, events):
    out = np.full(info['grid'].shape, info['bg'], dtype=np.int8)
    for p in flight:
        out[p] = info['signal']
    for entry, exit_index, route in events:
        for p in info['routes'][route]['cells']:
            out[p] = info['signal']
        for i, pattern in ((entry, _DIAMOND), (exit_index, _CHECKER)):
            r, c = info['centers'][i]
            out[r - 1:r + 2, c - 1:c + 2] = np.where(pattern, info['signal'], info['bg'])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    flight, events, _ = _propagate(info)
    return _render(info, flight, events)


def _features(info, flight, events, segments):
    active = {event[2] for event in events}
    crossed_decoy = any(any(p in info['routes'][a]['cells'] for a in active)
                        and any(p in route['cells'] for k, route in enumerate(info['routes']) if k not in active)
                        for p in info['crossings'])
    flight_decoy = any(p in route['cells'] for segment in segments for p in segment
                       for k, route in enumerate(info['routes']) if k not in active)
    near_wrong = any(any(j not in (i, partner) and
                            abs(p[0] - info['centers'][j][0]) + abs(p[1] - info['centers'][j][1])
                            < abs(p[0] - info['centers'][partner][0]) + abs(p[1] - info['centers'][partner][1])
                            for j in range(len(info['centers'])))
                     for i, partner, _ in events for p in [info['centers'][i]])
    turns, lengths = [], []
    for _, _, k in events:
        path = info['routes'][k]['path']
        headings = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
        turns.append(sum(a != b for a, b in zip(headings, headings[1:])))
        lengths.append(len(path))
    return dict(shape=info['grid'].shape, active=len(events), decoys=len(info['routes']) - len(events),
                crossings=len(info['crossings']), crossed_decoy=crossed_decoy,
                flight_decoy=flight_decoy, nearest_wrong=near_wrong,
                turns=tuple(turns), lengths=tuple(lengths), initial=info['initial'],
                palette=(info['bg'], info['signal'], info['cap_color'], info['plug'], info['carrier'], info['cross']),
                entry_sides=tuple(info['inward'][i] for i, _, _ in events),
                gaps=tuple(len(segment) for segment in segments))


def _geometry(cables, shape, tail, cap):
    """Construction validation uses explicit polylines and their ownership."""
    centers, inward, wire_sets = [], [], []
    squares = set()
    launch = set(tail) | {cap}
    for cable in cables:
        path = cable['path']
        assert len(path) == len(set(path))
        a, b = path[0], path[-1]
        assert _inside(a, shape, 2) and _inside(b, shape, 2)
        assert all(_inside(p, shape, 1) for p in path)
        first = _DIRS.index((path[1][0] - a[0], path[1][1] - a[1]))
        last_in = _DIRS.index((path[-2][0] - b[0], path[-2][1] - b[1]))
        for p in (a, b):
            sq = _square(p)
            assert not sq & squares
            assert not any(_add(q, d) in squares for q in sq for d in range(4))
            squares |= sq
        centers.extend((a, b))
        inward.extend((first, last_in))
        wire_sets.append(set(path) - _square(a) - _square(b))
    wires = set().union(*wire_sets)
    assert not wires & squares
    assert all(not any(_add(q, d) in wires | squares for d in range(4)) for q in launch)
    assert not launch & (wires | squares)
    paired, routes, crossings = _trace_field(wires, centers, inward)
    assert all(paired[2 * i] == 2 * i + 1 for i in range(len(cables)))
    assert all(routes[i]['cells'] == wire_sets[i] for i in range(len(cables)))
    for p in crossings:
        assert sum(p in wire for wire in wire_sets) == 2
    return centers, inward, wire_sets, crossings


def _construct_route(cables, shape, tail, cap, initial, geometry):
    centers, inward, wires, _ = geometry
    owner = {p: i for i, center in enumerate(centers) for p in _square(center)}
    p, d = cap, initial
    flight, events = set(tail) | {cap}, []
    for k in range(len(cables) + 1):
        p = _add(p, d)
        while _inside(p, shape) and p not in owner:
            flight.add(p)
            p = _add(p, d)
        if not _inside(p, shape):
            return flight, events
        i = owner[p]
        assert i % 2 == 0 and i // 2 == k, 'not the intended active order'
        assert p == _add(centers[i], (d + 2) % 4) and inward[i] == d
        events.append(i // 2)
        d = (inward[i + 1] + 2) % 4
        p = _add(centers[i + 1], d)
    raise AssertionError('construction cycle')


def _random_path(rng, start, initial, shape, bends=None):
    if bends is None:
        bends = int(rng.choice(4, p=[.14, .38, .34, .14]))
    path, p, d = [start], start, initial
    for segment in range(bends + 1):
        limit = 0
        while _inside(_add(p, d, limit + 1), shape, 2):
            limit += 1
        if limit < 4:
            return None
        length = int(rng.integers(4, min(limit, 14) + 1))
        path.extend(_add(p, d, k) for k in range(1, length + 1))
        p = path[-1]
        if segment < bends:
            d = (d + int(rng.choice([-1, 1]))) % 4
    if len(set(path)) != len(path):
        return None
    return {'path': path}


def _make_layout(rng, n, lesson):
    """Backtrack without changing requested active count. No disk diagnostics."""
    for canvas_attempt in range(90):
        shape = tuple(map(int, rng.integers(22, 31, size=2)))
        row = int(rng.integers(3, shape[0] - 3))
        length = int(rng.integers(2, 5))
        tail, cap, initial = [(row, c) for c in range(length)], (row, length), 1
        budget = [1000]

        def grow(cables, origin, d):
            if len(cables) == n:
                return cables
            distances = [k for k in range(5, 31) if _inside(_add(origin, d, k), shape, 2)]
            if not distances:
                return None
            for _ in range(100):
                if budget[0] <= 0:
                    return None
                budget[0] -= 1
                center = _add(origin, d, int(rng.choice(distances)))
                cable = _random_path(rng, center, d, shape)
                if cable is None:
                    continue
                trial = cables + [cable]
                try:
                    geometry = _geometry(trial, shape, tail, cap)
                    _, events = _construct_route(trial, shape, tail, cap, initial, geometry)
                    assert events == list(range(len(trial)))
                except (AssertionError, KeyError):
                    continue
                path = cable['path']
                out_d = _DIRS.index((path[-1][0] - path[-2][0], path[-1][1] - path[-2][1]))
                result = grow(trial, path[-1], out_d)
                if result is not None:
                    return result
            return None

        cables = grow([], cap, initial)
        if cables is None:
            continue
        geometry = _geometry(cables, shape, tail, cap)
        flight, _ = _construct_route(cables, shape, tail, cap, initial, geometry)
        # Add a decoy crossing a used cable or a free-flight segment. Its
        # endpoints must not intercept the flight; wire intersections may.
        decoys = []
        targets = sorted(set().union(*geometry[2]) if lesson == 'strand' else flight - set(tail) - {cap})
        rng.shuffle(targets)
        for _ in range(700):
            if not targets:
                break
            p = targets[int(rng.integers(len(targets)))]
            axis = int(rng.integers(2))
            neg, pos = ((0, 2) if axis == 0 else (3, 1))
            a = _add(p, neg, int(rng.integers(3, 9)))
            b = _add(p, pos, int(rng.integers(3, 9)))
            if not _inside(a, shape, 2) or not _inside(b, shape, 2):
                continue
            distance = abs(a[0] - b[0]) + abs(a[1] - b[1])
            path = [_add(a, pos, k) for k in range(distance + 1)]
            # Inactive strands may have the same bends and lengths as used
            # strands; being straight is not an inactivity convention.
            for end in (0, 1):
                if rng.random() < .55:
                    turn = (pos + int(rng.choice([-1, 1]))) % 4
                    anchor = path[0] if end == 0 else path[-1]
                    extension = int(rng.integers(4, 11))
                    if _inside(_add(anchor, turn, extension), shape, 2):
                        added = [_add(anchor, turn, k) for k in range(1, extension + 1)]
                        path = list(reversed(added)) + path if end == 0 else path + added
            cable = {'path': path}
            trial = cables + decoys + [cable]
            try:
                geom = _geometry(trial, shape, tail, cap)
                _, events = _construct_route(trial, shape, tail, cap, initial, geom)
                assert events == list(range(n))
            except (AssertionError, KeyError):
                continue
            decoys.append(cable)
            if len(decoys) >= 1:
                break
        if not decoys:
            continue
        if rng.random() < .5:
            for _ in range(180):
                start = (int(rng.integers(2, shape[0] - 2)), int(rng.integers(2, shape[1] - 2)))
                extra = _random_path(rng, start, int(rng.integers(4)), shape)
                if extra is None:
                    continue
                try:
                    trial = cables + decoys + [extra]
                    geom = _geometry(trial, shape, tail, cap)
                    _, events = _construct_route(trial, shape, tail, cap, initial, geom)
                    assert events == list(range(n))
                except (AssertionError, KeyError):
                    continue
                decoys.append(extra)
                break
        return shape, tail, cap, initial, cables + decoys, n
    raise RuntimeError('bounded geometry search exhausted')


def _build_pair(rng, n, lesson, palette, rotation):
    shape, tail, cap, initial, cables, n = _make_layout(rng, n, lesson)
    geometry = _geometry(cables, shape, tail, cap)
    centers, inward, wires, crossings = geometry
    flight, events = _construct_route(cables, shape, tail, cap, initial, geometry)
    bg, signal, cap_color, plug, carrier, cross = palette[:6]
    inp = np.full(shape, bg, dtype=np.int8)
    for wire in wires:
        for p in wire:
            inp[p] = cross if p in crossings else carrier
    for center in centers:
        r, c = center
        inp[r - 1:r + 2, c - 1:c + 2] = plug
    for p in tail:
        inp[p] = signal
    inp[cap] = cap_color
    out = np.full(shape, bg, dtype=np.int8)
    for p in flight:
        out[p] = signal
    for k in events:
        for p in wires[k]:
            out[p] = signal
        for center, pattern in ((centers[2 * k], _DIAMOND), (centers[2 * k + 1], _CHECKER)):
            r, c = center
            out[r - 1:r + 2, c - 1:c + 2] = np.where(pattern, signal, bg)
    if rng.random() < .5:
        inp, out = inp[:, ::-1], out[:, ::-1]
    inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
    info = _decode(inp)
    f, ev, segments = _propagate(info)
    assert np.array_equal(_render(info, f, ev), out), 'construction / input parser disagreement'
    facts = _features(info, f, ev, segments)
    return {'input': inp, 'output': out}, facts


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer from {low} through {high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 2, 10, 'num_train'), _count(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    base = list(map(int, rng.permutation(10)))
    vary_palette = bool(rng.integers(2))
    fixed_orientation = bool(rng.integers(2))
    base_rotation = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            lesson = 'strand' if split == 'train' and i == 0 else 'flight' if split == 'train' and i == 1 else str(rng.choice(['strand', 'flight']))
            n = (2 if i == 0 else 1 if i == 1 else int(rng.integers(1, 3))) if split == 'train' else 3
            for attempt in range(40):
                palette = list(map(int, rng.permutation(10))) if vary_palette else base
                rotation = base_rotation if fixed_orientation else int(rng.integers(4))
                pair, facts = _build_pair(rng, n, lesson, palette, rotation)
                if lesson == 'strand' and (not facts['crossed_decoy'] or not facts['nearest_wrong'] or not any(facts['turns'])):
                    continue
                if lesson == 'flight' and not facts['flight_decoy']:
                    continue
                key = _input_key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('bounded evidence/distinctness search exhausted')
            puzzle[split].append(pair)
            seen.add(key)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _input_key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _input_key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info = _decode(inp)
            flight, events, segments = _propagate(info)
            assert np.array_equal(_render(info, flight, events), out), 'incorrect energized route'
            assert not np.array_equal(inp, out)
            facts = _features(info, flight, events, segments)
            records[split].append(facts)
            # A cable contributes exclusive cells, so erasing it is observable.
            active = {e[2] for e in events}
            for k, route in enumerate(info['routes']):
                if k not in active:
                    assert all(out[info['centers'][i]] == info['bg'] for i in route['ends'])
            straight = np.full_like(inp, info['bg'])
            for p in info['tail']:
                straight[p] = info['signal']
            p = info['cap']
            while _inside(p, inp.shape):
                straight[p] = info['signal']
                p = _add(p, info['initial'])
            assert not np.array_equal(straight, out), 'cable-ignoring shortcut survives'
    train = records['train']
    assert any(f['active'] == 2 for f in train)
    assert any(f['crossed_decoy'] for f in train), 'missing strand-selection lesson'
    assert any(f['flight_decoy'] for f in train), 'missing transparent-wire lesson'
    assert any(f['nearest_wrong'] for f in train), 'nearest-plug shortcut survives'
    assert any(any(f['turns']) for f in train), 'missing bend lesson'
    assert all(f['active'] == 3 for f in records['test'])
    assert max(f['active'] for f in train) == 2


def self_test():
    """42 seeds, bounded summaries, no files and no retained grid population."""
    signatures, shapes, structures, palettes, sides = set(), set(), set(), set(), set()
    turns, crossings, gaps, lengths, decoy_counts = set(), set(), set(), set(), set()
    cases = Counter()
    reversed_cases = bent_decoys = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                digest.update(str(inp.shape).encode())
                digest.update(inp.tobytes())
                info = _decode(inp)
                flight, events, segments = _propagate(info)
                f = _features(info, flight, events, segments)
                assert np.array_equal(solve(inp.copy()), pair['output'])
                shapes.add(inp.shape)
                mapping = {}
                normalized = bytes(mapping.setdefault(int(v), len(mapping)) for v in inp.flat)
                structures.add((inp.shape, hashlib.sha256(normalized).digest()))
                palettes.add(f['palette'])
                sides.add(f['initial'])
                turns.update(f['turns'])
                crossings.add(f['crossings'])
                gaps.update(f['gaps'])
                lengths.update(f['lengths'])
                decoy_counts.add(f['decoys'])
                active = {event[2] for event in events}
                for k, route in enumerate(info['routes']):
                    path = route['path']
                    directions = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
                    if k not in active and any(a != b for a, b in zip(directions, directions[1:])):
                        bent_decoys += 1
                cases[(split, f['active'])] += 1
                # Reverse the actual route by moving the launcher to the
                # terminal ray's boundary end. No generation metadata is used.
                if (split == 'test' and len(segments[-1]) >= 4
                        and all(inp[p] == info['bg'] for p in segments[-1][-3:])):
                    reverse = inp.copy()
                    for p in info['tail'] + [info['cap']]:
                        reverse[p] = info['bg']
                    boundary = segments[-1][-1]
                    exit_index = events[-1][1]
                    direction = info['inward'][exit_index]
                    for offset in range(2):
                        reverse[_add(boundary, direction, offset)] = info['signal']
                    reverse[_add(boundary, direction, 2)] = info['cap_color']
                    expected = pair['output'].copy()
                    for i, j, _ in events:
                        for endpoint, pattern in ((i, _CHECKER), (j, _DIAMOND)):
                            r, c = info['centers'][endpoint]
                            expected[r - 1:r + 2, c - 1:c + 2] = np.where(pattern, info['signal'], info['bg'])
                    assert np.array_equal(solve(reverse), expected), 'flow reversal failed'
                    reversed_cases += 1
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(shapes) >= 20 and len(structures) >= 150
    assert len(palettes) >= 30 and sides == set(range(4))
    assert {0, 1, 2} <= turns and max(crossings) >= 2 and max(gaps) >= 10
    assert len(lengths) >= 15 and cases['test', 3] == 42
    assert decoy_counts == {1, 2} and bent_decoys >= 20 and reversed_cases >= 5
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(240 + 10 * nt + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    a, b = generate_puzzle(9981), generate_puzzle(9981)
    assert all(_input_key(p[k]) == _input_key(q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    pair = a['test'][0]
    for rotation in range(4):
        inp, expected = np.rot90(pair['input'], rotation), np.rot90(pair['output'], rotation)
        assert np.array_equal(solve(inp), expected)
    lut = np.array([3, 8, 1, 9, 6, 0, 4, 2, 5, 7], dtype=np.int8)
    assert np.array_equal(solve(lut[pair['input']]), lut[pair['output']])
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted answer accepted')
    return {'seeds': 42, 'pairs': sum(cases.values()), 'distinct_puzzles': len(signatures),
            'shapes': len(shapes), 'normalized_structures': len(structures),
            'palettes': len(palettes), 'initial_sides': sorted(sides),
            'turn_counts': sorted(turns), 'crossing_counts': sorted(crossings),
            'max_flight_length': max(gaps), 'wire_lengths': len(lengths),
            'decoy_counts': sorted(decoy_counts), 'bent_decoys': bent_decoys,
            'reverse_route_checks': reversed_cases}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
