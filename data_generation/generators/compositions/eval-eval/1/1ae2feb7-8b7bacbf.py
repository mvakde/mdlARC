"""Routed pulse waves: run programs compete by intrinsic cavity distance.

The public solver reads only its argument. Construction paints wavefronts from
intended rooms and routes; validation reconstructs those objects from pixels.
"""
from __future__ import annotations

from collections import Counter, deque
import heapq
import math
import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors(p):
    r, c = p
    return [(r + dr, c + dc) for dr, dc in D4]


def _halo(cells):
    return {q for p in cells for q in _neighbors(p)} - set(cells)


def _components(cells):
    left = set(cells)
    groups = []
    while left:
        root = min(left)
        left.remove(root)
        group, queue = {root}, [root]
        for p in queue:
            for q in _neighbors(p):
                if q in left:
                    left.remove(q)
                    group.add(q)
                    queue.append(q)
        groups.append(group)
    return groups


def _distances(cells, start):
    distance, queue = {start: 0}, deque([start])
    while queue:
        p = queue.popleft()
        for q in _neighbors(p):
            if q in cells and q not in distance:
                distance[q] = distance[p] + 1
                queue.append(q)
    assert len(distance) == len(cells), "Disconnected cavity"
    return distance


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9))
    return a


def _key(a):
    return a.shape, tuple(map(int, a.flat))


def _nonrectangular(cells):
    r0, r1 = min(r for r, c in cells), max(r for r, c in cells)
    c0, c1 = min(c for r, c in cells), max(c for r, c in cells)
    return len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1)


def _analyze(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel().astype(int), minlength=10)
    assert np.count_nonzero(counts == counts.max()) == 1, "Ambiguous background"
    bg = int(counts.argmax())
    h, w = a.shape
    inside = lambda p: 0 <= p[0] < h and 0 <= p[1] < w
    sources = []
    # A gate is an isolated three-cell transverse bar with exactly one
    # foreground neighbor, at its center. Its opposite ray holds the program.
    for color in range(10):
        if color == bg:
            continue
        for cells in _components(zip(*np.where(a == color))):
            if len(cells) != 3:
                continue
            ordered = sorted(cells)
            center = ordered[1]
            if not (len({r for r, c in cells}) == 1 or len({c for r, c in cells}) == 1):
                continue
            neighbors = {q for q in _halo(cells) if inside(q) and a[q] != bg}
            if len(neighbors) != 1:
                continue
            anchor = next(iter(neighbors))
            if anchor not in _neighbors(center):
                continue
            dr, dc = center[0] - anchor[0], center[1] - anchor[1]
            # The neighbor must be perpendicular to the bar.
            if (dr and len({r for r, c in cells}) != 1) or (dc and len({c for r, c in cells}) != 1):
                continue
            cursor = (center[0] + dr, center[1] + dc)
            if not inside(cursor) or a[cursor] != bg:
                continue
            runs = []
            distance = 1
            while inside(cursor):
                value = int(a[cursor])
                if value == bg:
                    cursor = (cursor[0] + dr, cursor[1] + dc)
                    distance += 1
                    continue
                length, proximity = 0, distance
                while inside(cursor) and a[cursor] == value:
                    length += 1
                    cursor = (cursor[0] + dr, cursor[1] + dc)
                    distance += 1
                runs.append((length, value, proximity))
            if runs:
                sources.append(dict(gate=color, center=center, anchor=anchor,
                                    wire=int(a[anchor]), direction=(dr, dc), runs=runs))
    assert sources, "No pulse program"
    gates = {s['gate'] for s in sources}
    wires = {s['wire'] for s in sources}
    assert len(gates) == len(wires) == 1
    gate, wire = next(iter(gates)), next(iter(wires))
    assert len({bg, gate, wire}) == 3
    wire_groups = _components(zip(*np.where(a == wire)))
    owner = {}
    for index, cells in enumerate(wire_groups):
        supplied = [i for i, s in enumerate(sources) if s['anchor'] in cells]
        assert len(supplied) <= 1, "Merged programs"
        for p in cells:
            owner[p] = supplied[0] if supplied else None
    holes = [cells for cells in _components(zip(*np.where(a == bg)))
             if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)]
    assert holes, "No enclosed target"
    edge_counts = Counter(int(a[p]) for cells in holes for p in _halo(cells)
                          if int(a[p]) != wire)
    assert edge_counts
    boundary = edge_counts.most_common(1)[0][0]
    assert boundary not in (bg, gate, wire)
    cavities = []
    for cells in holes:
        edge = _halo(cells)
        assert {int(a[p]) for p in edge} <= {boundary, wire}, "Non-contour enclosure"
        ports = []
        for p in sorted(edge):
            if a[p] != wire:
                continue
            adjacent = cells.intersection(_neighbors(p))
            assert len(adjacent) == 1, "Each port must enter exactly one interior cell"
            entry = next(iter(adjacent))
            ports.append(dict(cell=p, entry=entry, source=owner[p],
                              distance=_distances(cells, entry)))
        cavities.append(dict(cells=cells, edge=edge, ports=ports))
    roles = {bg, gate, wire, boundary}
    assert all(1 <= len(s['runs']) <= 3 and
               all(1 <= length <= 6 and color not in roles for length, color, gap in s['runs'])
               for s in sources)
    return dict(background=bg, gate=gate, wire=wire, boundary=boundary,
                sources=sources, cavities=cavities, wire_groups=wire_groups)


def _predict(a, info, alternative=None):
    out = a.copy()
    for cavity in info['cavities']:
        ports = [p for p in cavity['ports'] if p['source'] is not None]
        if alternative == 'all_cavities':
            ports = []
            for i, s in enumerate(info['sources']):
                entry = min(cavity['cells'], key=lambda q: (abs(q[0] - s['center'][0]) +
                            abs(q[1] - s['center'][1]), q))
                ports.append(dict(source=i, entry=entry, distance=_distances(cavity['cells'], entry)))
        for q in cavity['cells']:
            emissions = []
            for port in ports:
                source = info['sources'][port['source']]
                d = port['distance'][q]
                if alternative == 'manhattan':
                    entry = port['entry']
                    d = abs(q[0] - entry[0]) + abs(q[1] - entry[1])
                runs = source['runs']
                hits = [run for run in runs if d % run[0] == 0]
                if alternative == 'uniform':
                    hits = runs[:1]
                if not hits:
                    continue
                if alternative == 'far_run':
                    winner = hits[-1]
                elif alternative == 'short_run':
                    winner = min(hits, key=lambda run: (run[0], run[2]))
                elif alternative == 'long_run':
                    winner = min(hits, key=lambda run: (-run[0], run[2]))
                else:
                    winner = hits[0]
                emissions.append((d, winner[1]))
            if not emissions:
                continue
            target_distance = (max if alternative == 'far_port' else min)(d for d, color in emissions)
            colors = {color for d, color in emissions if d == target_distance}
            if alternative == 'nearest_even_if_silent' and ports:
                if min(p['distance'][q] for p in ports) < target_distance:
                    continue
            if alternative == 'cancel_all_ties' and sum(d == target_distance for d, c in emissions) > 1:
                continue
            if len(colors) == 1:
                out[q] = next(iter(colors))
            elif alternative == 'no_cancel':
                out[q] = min(colors)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer programs, route ownership, entrances and intrinsic wave distances."""
    a = _grid(input_grid)
    return _predict(a, _analyze(a))


def _shape(rng, kind, h=None, w=None):
    h = int(rng.integers(3, 7)) if h is None else h
    w = int(rng.integers(3, 8)) if w is None else w
    cells = {(r, c) for r in range(h) for c in range(w)}
    if kind == 'bay':
        # A notch in the *interior*, with a thin contour around the notch.
        arm = max(2, w // 3)
        base = int(rng.integers(1, min(3, h - 2)))
        cells -= {(r, c) for r in range(h - base) for c in range(arm, w - arm)}
    elif kind == 'steps':
        cells = set()
        left = 0
        for r in range(h):
            left += int(rng.integers(-1, 2)) if r else 0
            width = max(2, w + int(rng.integers(-1, 2)))
            cells.update((r, c) for c in range(left, left + width))
    elif kind == 'diamond':
        radius = int(rng.integers(1, 4))
        cells = {(r, c) for r in range(-radius, radius + 1)
                 for c in range(-radius, radius + 1) if abs(r) + abs(c) <= radius}
    elif kind == 'waist':
        cells -= {(r, c) for r in range(1, h - 1) for c in (0, w - 1)}
    assert len(_components(cells)) == 1
    return cells


def _translate(cells, dr, dc):
    return {(r + dr, c + dc) for r, c in cells}


def _port_options(cells):
    edge = _halo(cells)
    options = []
    for p in sorted(edge):
        neighbors = cells.intersection(_neighbors(p))
        if len(neighbors) == 1:
            q = next(iter(neighbors))
            exterior = (2 * p[0] - q[0], 2 * p[1] - q[1])
            if exterior not in cells | edge:
                options.append((p, q, exterior))
    return options


def _route(rng, starts, goal, allowed):
    starts = sorted(starts)
    queue = [(0, p) for p in starts]
    heapq.heapify(queue)
    distance = {p: 0 for p in starts}
    previous = {p: None for p in starts}
    ordered = sorted(allowed)
    costs = dict(zip(ordered, map(int, rng.integers(1, 9, size=len(ordered)))))
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != distance[p]:
            continue
        if p == goal:
            path = set()
            while p is not None:
                path.add(p)
                p = previous[p]
            return path
        for q in _neighbors(p):
            if q not in allowed:
                continue
            new_cost = cost + costs[q]
            if new_cost < distance.get(q, 10**9):
                distance[q], previous[q] = new_cost, p
                heapq.heappush(queue, (new_cost, q))
    return None


def _paint_waves(a, rooms, programs):
    """Construction uses successive wavefront sets, without input parsing."""
    out = a.copy()
    for room in rooms:
        arrivals = {p: [] for p in room['cells']}
        for owner, entry, port in room['supplies']:
            if owner is None:
                continue
            visited, frontier, layer = set(), {entry}, 0
            while frontier:
                emission = next((color for length, color in programs[owner] if layer % length == 0), None)
                if emission is not None:
                    for p in frontier:
                        arrivals[p].append((layer, emission))
                visited |= frontier
                frontier = _halo(frontier).intersection(room['cells']) - visited
                layer += 1
        for p, hits in arrivals.items():
            if hits:
                closest = min(layer for layer, color in hits)
                colors = {color for layer, color in hits if layer == closest}
                if len(colors) == 1:
                    out[p] = next(iter(colors))
    return out


def _scene(rng, lesson, periods, payload, roles, max_period=6, source_count=None):
    bg, boundary, wire, gate = roles
    a_period, b_period = periods
    x, y, z = payload[:3]
    if lesson == 0:
        programs = [[(a_period, x), (b_period, y)]]
        kinds = [('bay', int(rng.integers(6, 10)), int(rng.integers(8, 12))),
                 (str(rng.choice(['box', 'steps', 'diamond', 'waist'])), None, None), ('box', 2, 3)]
        owners = [[0], [0], []]
    elif lesson == 1:
        programs = [[(b_period, y), (a_period, x)], [(a_period, y)]]
        kinds = [('box', 2 * a_period + 1, math.lcm(*periods) - a_period + 2), ('box', 2, 3)]
        owners = [[0, 1], []]
    else:
        number = 2 if lesson == 2 else source_count
        programs = []
        for owner in range(number):
            nrun = int(rng.choice([1, 2, 3], p=[.25, .55, .20])) if number < 3 else 1
            limit = min(3, max_period) if nrun == 3 else max_period
            lengths = [int(rng.integers(1, limit + 1)) for _ in range(nrun)]
            while sum(lengths) + 2 * len(lengths) > 13:
                lengths.pop()
            colors = [int(rng.choice(payload)) for _ in lengths]
            programs.append(list(zip(lengths, colors)))
        if lesson == 2:
            kinds = [('bay', int(rng.integers(5, 9)), int(rng.integers(7, 12))),
                     (str(rng.choice(['box', 'diamond', 'steps', 'waist'])), None, None), ('box', 2, 2)]
            owners = [[0, 1], [int(rng.integers(number))], []]
            if rng.random() < .4:
                kinds.pop(1)
                owners.pop(1)
        else:
            # Extra demonstrations revisit one of the two learned mechanisms;
            # tests recombine concavity and competing entrances.
            kinds = [('box' if number > 1 else 'bay', int(rng.integers(5, 9)), int(rng.integers(7, 12))),
                     ('box', 2, 3)]
            owners = [list(range(number)), []]
            if rng.random() < .5:
                kinds.append((str(rng.choice(['box', 'steps', 'diamond'])), None, None))
                owners.append([int(rng.integers(number))])
    minimum_size = 25 if lesson == 2 or len(programs) == 3 else 20
    h, w = int(rng.integers(minimum_size, 31)), int(rng.integers(minimum_size, 31))
    canvas = {(r, c) for r in range(h) for c in range(w)}
    rooms, occupied = [], set()
    for index, (kind, rh, rw) in enumerate(kinds):
        cells = _shape(rng, kind, rh, rw)
        if kind != 'box' and rng.random() < .5:
            cells = {(c, r) for r, c in cells}
        edge = _halo(cells)
        r0, c0 = min(r for r, c in edge), min(c for r, c in edge)
        cells, edge = _translate(cells, -r0, -c0), _translate(edge, -r0, -c0)
        sh, sw = max(r for r, c in edge) + 1, max(c for r, c in edge) + 1
        placed = False
        placement_blocked = occupied | _halo(occupied)
        for _ in range(100):
            dr, dc = int(rng.integers(1, h - sh)), int(rng.integers(1, w - sw))
            moved, shell = _translate(cells, dr, dc), _translate(edge, dr, dc)
            full = moved | shell
            if full & placement_blocked:
                continue
            rooms.append(dict(cells=moved, edge=shell, supplies=[]))
            occupied |= full
            placed = True
            break
        if not placed:
            return None
    source_objects, ray_reserved, source_occupied = [], set(), set()
    all_occupied = set(occupied)
    for program in programs:
        placed = False
        placement_blocked = all_occupied | _halo(all_occupied) | ray_reserved
        for _ in range(350):
            dr, dc = D4[int(rng.integers(4))]
            center = (int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))
            bar = {center, (center[0] + dc, center[1] - dr), (center[0] - dc, center[1] + dr)}
            anchor = (center[0] - dr, center[1] - dc)
            cursor, run_cells = int(rng.integers(2, 5)), {}
            for length, color in program:
                for d in range(cursor, cursor + length):
                    run_cells[(center[0] + d * dr, center[1] + d * dc)] = color
                cursor += length + int(rng.integers(1, 3))
            footprint = bar | set(run_cells) | {anchor}
            if not footprint <= canvas:
                continue
            ray = set()
            p = (center[0] + dr, center[1] + dc)
            while p in canvas:
                ray.add(p)
                p = (p[0] + dr, p[1] + dc)
            if footprint & placement_blocked or ray & all_occupied:
                continue
            source_objects.append(dict(bar=bar, anchor=anchor, run_cells=run_cells, ray=ray))
            all_occupied |= footprint
            source_occupied |= bar | set(run_cells)
            ray_reserved |= ray
            placed = True
            break
        if not placed:
            return None
    wire_sets = [{s['anchor']} for s in source_objects]
    targets = []
    for index, (room, supplied_by) in enumerate(zip(rooms, owners)):
        options = [p for p in _port_options(room['cells']) if p[2] in canvas]
        if index == 0 and lesson == 1:
            # Opposed aligned entrances expose different-color cancellation at
            # d=a and same-color coalescence at d=lcm(a,b) in the middle row.
            top = min(r for r, c in room['cells'])
            bottom = max(r for r, c in room['cells'])
            col = (min if rng.random() < .5 else max)(c for r, c in room['cells'])
            chosen = [next(p for p in options if p[1] == (top, col) and p[0][0] == top - 1),
                      next(p for p in options if p[1] == (bottom, col) and p[0][0] == bottom + 1)]
        elif index == 0 and lesson == 0:
            first = min(room['cells'])
            chosen = [p for p in options if p[1] == first][:1]
        else:
            rng.shuffle(options)
            chosen = []
            for _ in supplied_by:
                viable = [p for p in options if p not in chosen]
                if not viable:
                    break
                if chosen:
                    # Separate independent routes at their target, rather than
                    # repeatedly requesting two adjacent, unrouteable ports.
                    viable.sort(key=lambda p: min(abs(p[0][0] - q[0][0]) + abs(p[0][1] - q[0][1])
                                                  for q in chosen), reverse=True)
                    viable = viable[:max(1, len(viable) // 3)]
                chosen.append(viable[int(rng.integers(len(viable)))])
        if len(chosen) != len(supplied_by):
            return None
        for owner, (port, entry, exterior) in zip(supplied_by, chosen):
            targets.append((owner, index, port, entry, exterior))
    rng.shuffle(targets)
    for owner, index, port, entry, exterior in targets:
        others = set().union(*(cells for j, cells in enumerate(wire_sets) if j != owner))
        blocked = occupied | source_occupied | _halo(source_occupied) | ray_reserved | others | _halo(others)
        own = wire_sets[owner]
        allowed = (canvas - blocked) | own
        if exterior not in allowed:
            return None
        path = _route(rng, own, exterior, allowed)
        if path is None or port in others or port in _halo(others):
            return None
        wire_sets[owner] |= path | {port}
        rooms[index]['supplies'].append((owner, entry, port))
    a = np.full((h, w), bg, dtype=np.int8)
    for room in rooms:
        for p in room['edge']:
            a[p] = boundary
    for s in source_objects:
        for p in s['bar']:
            a[p] = gate
        for p, color in s['run_cells'].items():
            a[p] = color
    for cells in wire_sets:
        for p in cells:
            a[p] = wire
    b = _paint_waves(a, rooms, programs)
    try:
        info = _analyze(a)
        if len(info['cavities']) != len(rooms) or len(info['sources']) != len(programs):
            return None
        source_lookup = {s['anchor']: i for i, s in enumerate(info['sources'])}
        for owner, intended in enumerate(programs):
            observed = info['sources'][source_lookup[source_objects[owner]['anchor']]]
            if [(length, color) for length, color, proximity in observed['runs']] != intended:
                return None
        for room in rooms:
            observed = next(c for c in info['cavities'] if c['cells'] == room['cells'])
            actual_ports = {p['cell']: p['source'] for p in observed['ports']}
            intended_ports = {port: source_lookup[source_objects[owner]['anchor']]
                              for owner, entry, port in room['supplies']}
            if actual_ports != intended_ports:
                return None
        if not np.array_equal(_predict(a, info), b) or np.array_equal(a, b):
            return None
    except (AssertionError, KeyError):
        return None
    if lesson == 2 and not any(any(d > abs(q[0] - p['entry'][0]) + abs(q[1] - p['entry'][1])
                                  for q, d in p['distance'].items())
                              for c in info['cavities'] for p in c['ports'] if p['source'] is not None):
        return None
    return dict(input=a, output=b)


def _fallback(lesson, periods, payload, roles, source_count=None):
    """Finite, explicitly routed layout; preserves the same teaching witnesses."""
    bg, boundary, wire, gate = roles
    a_period, b_period = periods
    x, y, z = payload[:3]
    shared = lesson in (1, 2) or (lesson == 3 and source_count > 1)
    programs = [[(b_period, y), (a_period, x)], [(a_period, y)]] if shared else [[(a_period, x), (b_period, y)]]
    three_sources = lesson == 3 and source_count == 3
    if three_sources:
        programs.append([(b_period, z)])
    if lesson in (0, 2):
        inner = {(r, c) for r in range(10, 17) for c in range(17, 26)
                 if r >= 15 or c <= 19 or c >= 23}
        entries = [(10, 18), (16, 18)]
    else:
        hh, ww = 2 * a_period + 1, math.lcm(*periods) - a_period + 2
        inner = {(r, c) for r in range(10, 10 + hh) for c in range(15, 15 + ww)}
        entries = [(10, 15), (9 + hh, 15)]
    rooms = [dict(cells=inner, edge=_halo(inner), supplies=[])]
    detached_start = 23 if three_sources else 4
    detached = {(r, c) for r in range(20, 22) for c in range(detached_start, detached_start + 3)}
    rooms.append(dict(cells=detached, edge=_halo(detached), supplies=[]))
    if not shared:
        other = {(r, c) for r in range(10, 13) for c in range(4, 8)}
        rooms.append(dict(cells=other, edge=_halo(other), supplies=[(0, (10, 6), (9, 6))]))
    a = np.full((30, 30), bg, dtype=np.int8)
    for room in rooms:
        for p in room['edge']:
            a[p] = boundary
    for owner, program in enumerate(programs):
        if owner == 2:
            row = 10 + a_period
            a[15, 3:6] = gate
            a[17:17 + b_period, 4] = z
            a[row:15, 4] = wire
            a[row, 4:15] = wire
            rooms[0]['supplies'].append((2, (row, 15), (row, 14)))
            continue
        row = 3 if owner == 0 else 25
        a[row - 1:row + 2, 11] = gate
        cursor = 9
        for length, color in program:
            a[row, cursor - length + 1:cursor + 1] = color
            cursor -= length + 1
        entry = entries[owner]
        port = (entry[0] - 1, entry[1]) if owner == 0 else (entry[0] + 1, entry[1])
        col = port[1]
        a[row, 12:col + 1] = wire
        lo, hi = sorted((row, port[0]))
        a[lo:hi + 1, col] = wire
        rooms[0]['supplies'].append((owner, entry, port))
    if not shared:
        a[6, 6:19] = wire
        a[6:10, 6] = wire
    b = _paint_waves(a, rooms, programs)
    assert np.array_equal(solve(a), b)
    return dict(input=a, output=b)


def _counts(num_train, num_test):
    for name, value, lo, hi in [('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    roles, payload = palette[:4], palette[4:]
    periods = [(2, 3), (2, 5), (3, 4)][int(rng.integers(3))]
    base_rotation = int(rng.integers(4))
    mixed_rotation = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    taught_maximum = max(periods)
    for split, count in [('train', num_train), ('test', num_test)]:
        for index in range(count):
            lesson = index if split == 'train' and index < 2 else 2 if split == 'test' else 3
            source_count = int(rng.integers(1, 4)) if lesson == 3 else None
            pair = None
            for _ in range(55):
                candidate = _scene(rng, lesson, periods, payload, roles,
                                   taught_maximum if split == 'test' else 6, source_count)
                if candidate is not None:
                    pair = candidate
                    break
            if pair is None:
                pair = _fallback(lesson, periods, payload, roles, source_count)
            rotation = int(rng.integers(4)) if mixed_rotation else base_rotation
            reflected = bool(rng.integers(2))
            pair = {role: np.rot90(np.fliplr(grid) if reflected else grid, rotation).copy()
                    for role, grid in pair.items()}
            # A finite role-preserving payload permutation supplies uniqueness
            # even in the unlikely case of repeated fallback layouts.
            original = pair
            for shift in range(10):
                if _key(pair['input']) not in seen:
                    break
                mapping = (np.arange(10, dtype=np.int8) + shift + 1) % 10
                pair = {role: mapping[grid] for role, grid in original.items()}
            else:
                raise AssertionError('Exhausted distinct inputs')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                taught_maximum = max(taught_maximum, max(run[0] for s in _analyze(pair['input'])['sources']
                                                        for run in s['runs']))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _analyze(a)
            assert np.array_equal(_predict(a, info), b), 'Incorrect wave, ownership, priority or tie'
            assert not np.array_equal(a, b)
            interiors = set().union(*(c['cells'] for c in info['cavities']))
            changed = set(map(tuple, np.argwhere(a != b)))
            assert changed <= interiors and all(a[p] == info['background'] for p in changed)
            assert set(map(int, b.flat)) <= set(map(int, a.flat))
            assert 1 <= len(info['sources']) <= 3
            assert all(len(c['ports']) <= 3 for c in info['cavities'])
            assert any(not any(p['source'] is not None for p in c['ports']) for c in info['cavities'])
            infos[split].append(info)
    # These alternatives are tested on *visible demonstrations*, including the
    # minimum two-example curriculum, rather than on construction parameters.
    for alternative in ('uniform', 'all_cavities', 'manhattan', 'far_run', 'short_run',
                        'long_run', 'no_cancel', 'cancel_all_ties', 'far_port', 'nearest_even_if_silent'):
        assert any(not np.array_equal(_predict(np.asarray(pair['input']), info, alternative), pair['output'])
                   for pair, info in zip(puzzle['train'], infos['train'])), f'Teaching gap: {alternative}'
    assert any(any(sum(p['source'] is not None for p in c['ports']) >= 2 for c in info['cavities'])
               for info in infos['train'])
    for info in infos['test']:
        assert any(sum(p['source'] is not None for p in c['ports']) >= 2 and _nonrectangular(c['cells'])
                   for c in info['cavities']), 'Test must recombine competing ports and concavity'


def _features(pair):
    a = np.asarray(pair['input'])
    info = _analyze(a)
    cavity_shapes = []
    detours = 0
    for c in info['cavities']:
        r0, col0 = min(r for r, col in c['cells']), min(col for r, col in c['cells'])
        cavity_shapes.append(tuple(sorted((int(r - r0), int(col - col0)) for r, col in c['cells'])))
        for port in c['ports']:
            entry = port['entry']
            detours += sum(bool(d > abs(q[0] - entry[0]) + abs(q[1] - entry[1])) for q, d in port['distance'].items())
    route_lengths = []
    for source_index, source in enumerate(info['sources']):
        cells = next(cells for cells in info['wire_groups'] if source['anchor'] in cells)
        distance = _distances(cells, source['anchor'])
        route_lengths.extend(int(distance[p['cell']] + 1) for c in info['cavities'] for p in c['ports']
                             if p['source'] == source_index)
    all_wire = set().union(*info['wire_groups'])
    junctions = sum(len(all_wire.intersection(_neighbors(p))) >= 3 for p in all_wire)
    return dict(shape=list(a.shape), roles=[info[k] for k in ('background', 'boundary', 'wire', 'gate')],
                sources=len(info['sources']), cavities=len(info['cavities']),
                runs=[len(s['runs']) for s in info['sources']],
                periods=sorted({run[0] for s in info['sources'] for run in s['runs']}),
                ports=[len(c['ports']) for c in info['cavities']],
                branch=any(sum(p['source'] == i for c in info['cavities'] for p in c['ports']) > 1
                           for i in range(len(info['sources']))),
                detours=detours, cavity_shapes=cavity_shapes,
                wire_cells=len(all_wire), route_lengths=route_lengths, junctions=junctions,
                source_gaps=[int(s['runs'][0][2] - 1) for s in info['sources']],
                directions=sorted({tuple(map(int, s['direction'])) for s in info['sources']}))


def _hand_diagnostics():
    # Two different exterior route lengths feed the same hand-labeled strip.
    for row in (1, 4):
        a = np.zeros((9, 21), dtype=np.int8)
        a[row - 1:row + 2, 7] = 1
        a[row, 0:2] = 4
        a[row, 3:6] = 7
        a[3, 11:20] = a[5, 11:20] = 8
        a[4, 20] = 8
        a[row, 8:11] = 5
        a[row:5, 10] = 5
        expected = a.copy()
        expected[4, 11:20] = [7, 0, 4, 7, 4, 0, 7, 0, 4]
        assert np.array_equal(solve(a), expected)
        # Omit B's actual route: an inactive port cannot receive the program.
        disconnected = a.copy()
        disconnected[row, 9] = 0
        assert np.array_equal(solve(disconnected), disconnected)
        # Change A's period while keeping the gate, route and target fixed.
        shorter = a.copy()
        shorter[row, 3] = 0
        shortened_expected = shorter.copy()
        shortened_expected[4, 11:20] = [7, 0, 7, 0, 7, 0, 7, 0, 7]
        assert np.array_equal(solve(shorter), shortened_expected)
    # U-shaped room: (4,16) has intrinsic distance 9, direct distance 5.
    a = np.zeros((9, 20), dtype=np.int8)
    a[3:6, 7] = 1
    a[4, 0:2], a[4, 3:6], a[4, 8:11] = 4, 7, 5
    cells = {(r, c) for r in range(4, 7) for c in range(11, 18)
             if r == 6 or c <= 12 or c >= 16}
    for p in _halo(cells):
        a[p] = 8
    a[4, 10] = 5
    expected = a.copy()
    for p, color in {(4, 11): 7, (5, 12): 4, (6, 11): 4, (6, 12): 7,
                     (6, 13): 4, (6, 15): 7, (6, 17): 4, (5, 16): 4,
                     (5, 17): 7, (4, 16): 7, (4, 17): 4}.items():
        expected[p] = color
    assert np.array_equal(solve(a), expected)
    # Aligned opposed ports: unequal colors cancel at d=2; equal colors
    # survive at d=6. A farther firing port paints despite a silent near port.
    a = np.zeros((16, 22), dtype=np.int8)
    for row in (1, 13):
        a[row - 1:row + 2, 7] = 1
        a[row, 8:13] = 5
    a[1, 0:2], a[1, 3:6], a[13, 4:6] = 4, 7, 7
    cells = {(r, c) for r in range(4, 9) for c in range(12, 17)}
    for p in _halo(cells):
        a[p] = 8
    a[2:4, 12] = a[9:13, 12] = 5
    result = solve(a)
    assert result[6, 12:17].tolist() == [0, 7, 0, 0, 7]
    assert result[4, 14] == 4 and result[7, 12] == 7


def self_test():
    """Bounded semantic diagnostics, metamorphisms and 42-seed diversity audit."""
    _hand_diagnostics()
    puzzles, features, structures = [], [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                features.append(_features(pair))
                colors, normalized = {}, []
                for value in a.flat:
                    normalized.append(colors.setdefault(int(value), len(colors)))
                structures.add((a.shape, tuple(normalized)))
                assert np.array_equal(solve(a.tolist()), b)
    assert len(structures) >= 180
    assert len({tuple(f['shape']) for f in features}) >= 25
    assert len({tuple(f['roles']) for f in features}) >= 30
    assert {1, 2, 3} <= {f['sources'] for f in features}
    assert {1, 2, 3} <= {n for f in features for n in f['runs']}
    assert set(range(1, 7)) <= {n for f in features for n in f['periods']}
    assert len({s for f in features for s in f['cavity_shapes']}) >= 50
    assert sum(f['branch'] for f in features) >= 42
    assert sum(f['detours'] > 0 for f in features) >= 84
    assert len({d for f in features for d in f['directions']}) == 4
    first = puzzles[0]
    for pair in first['train'] + first['test']:
        a, b = pair['input'], pair['output']
        for turn in range(4):
            assert np.array_equal(solve(np.rot90(a, turn)), np.rot90(b, turn))
        assert np.array_equal(solve(np.fliplr(a)), np.fliplr(b))
        permutation = np.array([7, 3, 9, 0, 1, 6, 4, 8, 2, 5], dtype=np.int8)
        assert np.array_equal(solve(permutation[a]), permutation[b])
    again = generate_puzzle(0)
    assert all(_key(p[role]) == _key(q[role]) for s in first for p, q in zip(first[s], again[s]) for role in p)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for counts in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)]:
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    corrupted = {s: [{role: a.copy() for role, a in pair.items()} for pair in first[s]] for s in first}
    pair = corrupted['test'][0]
    p = tuple(np.argwhere(pair['input'] != pair['output'])[0])
    pair['output'][p] = pair['input'][p]
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError('Missing wave cell accepted')
    # Hand-computed pulse sequence: near 3/Y, far 2/X. No generated answer
    # supplies these expected values.
    assert [next((c for length, c in [(3, 7), (2, 4)] if d % length == 0), 0)
            for d in range(9)] == [7, 0, 4, 7, 4, 0, 7, 0, 4]
    return dict(seeds=42, pairs=len(features), structures=len(structures),
                shapes=len({tuple(f['shape']) for f in features}),
                cavity_shapes=len({s for f in features for s in f['cavity_shapes']}))


if __name__ == '__main__':
    print(self_test())
