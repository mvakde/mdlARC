"""Assemble a block-scaled road, then run a unary-fueled traveler along it.

Only 4 (available charge) and 7 (spent charge / trail) have fixed meanings.
All other colors are recovered from geometry in each input. No parent code,
examples, random state, or training labels are used by solve().
"""
from __future__ import annotations

from collections import Counter
from math import isqrt
import numpy as np


AVAILABLE, SPENT = 4, 7
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # U, R, D, L
HOST_SEEDS = (0, 667948882, 449611914, 192175767, 558524913,
              970461115, 871117863, 30880057, 886672981, 452219795,
              893964504, 845061192, 857923972, 764615324, 852546793,
              244515029, 481517509, 504822992, 368486086, 411035198,
              621468689, 498762161, 317631913, 873254842, 678174541,
              404321902, 272449280, 867522389, 678827668, 437722335,
              313464368, 323719863, 959990086, 178268052, 118433663,
              351193531, 421230328, 554868608, 390769402, 993847226,
              63422450, 497567990)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((0 <= a) & (a <= 9))), "invalid ARC colors")
    return a.astype(int, copy=False)


def _neighbors(p):
    return [(p[0] + dr, p[1] + dc) for dr, dc in DELTAS]


def _components(mask):
    pending = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while pending:
        p = min(pending)
        pending.remove(p)
        stack, cells = [p], []
        while stack:
            p = stack.pop()
            cells.append(p)
            for q in _neighbors(p):
                if q in pending:
                    pending.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells) + 1,
            min(c for r, c in cells), max(c for r, c in cells) + 1)


def _normal(cells):
    t, b, l, r = _bounds(cells)
    return tuple(sorted((y - t, x - l) for y, x in cells))


def _pattern_normal(pattern):
    t = min(y for y, x, v in pattern)
    l = min(x for y, x, v in pattern)
    return tuple(sorted((y - t, x - l, v) for y, x, v in pattern))


def _pattern_turn(pattern):
    return _pattern_normal([(x, -y, v) for y, x, v in pattern])


def _variants(pattern, reflections=True):
    bases = [pattern]
    if reflections:
        bases.append(_pattern_normal([(y, -x, v) for y, x, v in pattern]))
    result = set()
    for base in bases:
        for _ in range(4):
            base = _pattern_normal(base)
            result.add(base)
            base = _pattern_turn(base)
    return tuple(sorted(result))


def _scan(t, b, l, r, direction):
    """Raster order after rotating the leading bar to the top."""
    if direction == 0:
        return [(y, x) for y in range(t + 1, b - 1) for x in range(l + 1, r - 1)]
    if direction == 1:
        return [(y, x) for x in range(r - 2, l, -1) for y in range(t + 1, b - 1)]
    if direction == 2:
        return [(y, x) for y in range(b - 2, t, -1) for x in range(r - 2, l, -1)]
    return [(y, x) for x in range(l + 1, r - 1) for y in range(b - 2, t, -1)]


def _bar(t, b, l, r, direction):
    if direction == 0:
        return [(t - 1, x) for x in range(l, r)]
    if direction == 1:
        return [(y, r) for y in range(t, b)]
    if direction == 2:
        return [(b, x) for x in range(l, r)]
    return [(y, l - 1) for y in range(t, b)]


def _gauges(board, background, structural):
    result = []
    for color in np.unique(board):
        color = int(color)
        if color in (background, structural, AVAILABLE, SPENT):
            continue
        for cells in _components(board == color):
            t, b, l, r = _bounds(cells)
            h, w = b - t, r - l
            if min(h, w) < 3 or len(cells) != 2 * (h + w) - 4:
                continue
            if any(y not in (t, b - 1) and x not in (l, r - 1) for y, x in cells):
                continue
            if not np.all(np.isin(board[t + 1:b - 1, l + 1:r - 1], (AVAILABLE, SPENT))):
                continue
            bars = []
            for direction in range(4):
                bar = _bar(t, b, l, r, direction)
                if any(not (0 <= y < board.shape[0] and 0 <= x < board.shape[1]) for y, x in bar):
                    continue
                marker = int(board[bar[0]])
                if marker in (background, structural, AVAILABLE, SPENT, color):
                    continue
                if all(board[p] == marker for p in bar):
                    bars.append((direction, marker, bar))
            if len(bars) != 1:
                continue
            direction, marker, bar = bars[0]
            scan = _scan(t, b, l, r, direction)
            advanced = sum(board[p] == SPENT for p in scan)
            if [int(board[p]) for p in scan] != [SPENT] * advanced + [AVAILABLE] * (len(scan) - advanced):
                continue
            result.append(dict(bounds=(t, b, l, r), frame=color, marker=marker,
                               bar=tuple(bar), direction=direction, scan=scan,
                               advanced=advanced, remaining=len(scan) - advanced,
                               capacity=len(scan)))
    return result


def _decode_piece(grid, cells, scale, background, structural, road):
    t, b, l, r = _bounds(cells)
    crop = grid[t:b, l:r]
    if (b - t) % scale or (r - l) % scale:
        return None
    pattern = []
    for y in range(0, b - t, scale):
        for x in range(0, r - l, scale):
            block = crop[y:y + scale, x:x + scale]
            color = int(block[0, 0])
            if not np.all(block == color):
                return None
            if color == background:
                continue
            if color not in (structural, road):
                return None
            pattern.append((y // scale, x // scale, int(color == road)))
    if sum(v for y, x, v in pattern) < 2 or not any(v == 0 for y, x, v in pattern):
        return None
    if not 3 <= len(pattern) <= 8 or max(max(y for y, x, v in pattern),
                                        max(x for y, x, v in pattern)) >= 4:
        return None
    road_cells = {(y, x) for y, x, v in pattern if v}
    # A module is a connected road fragment with two ends, surrounded by some wall.
    if sum(sum(q in road_cells for q in _neighbors(p)) == 1 for p in road_cells) != 2:
        return None
    if any(sum(q in road_cells for q in _neighbors(p)) not in (1, 2) for p in road_cells):
        return None
    return _pattern_normal(pattern)


def _walk(track, start, goal):
    """Reject branches, loops, disconnected fragments and unintended endpoints."""
    for p in track:
        required = 1 if p in (start, goal) else 2
        if sum(q in track for q in _neighbors(p)) != required:
            return None
    path, previous, current = [start], None, start
    while current != goal:
        nxt = [q for q in _neighbors(current) if q in track and q != previous]
        if len(nxt) != 1 or nxt[0] in path:
            return None
        previous, current = current, nxt[0]
        path.append(current)
    return path if len(path) == len(track) else None


def _assemble(board, holes, pieces, structural, road, start, goal):
    fixed = set(map(tuple, np.argwhere(board == road).tolist())) | {start, goal}
    fixed_degrees = {p: sum(q in fixed for q in _neighbors(p)) for p in fixed}
    requirements = {p: 1 if p in (start, goal) else 2 for p in fixed}
    variants = [_variants(p) for p in pieces]
    keys = [v[0] for v in variants]
    options = []
    geometric_ambiguity = False
    for hole in holes:
        t, b, l, r = _bounds(hole)
        shape = _normal(hole)
        choices = []
        for index, vv in enumerate(variants):
            fits = [p for p in vv if tuple((y, x) for y, x, v in p) == shape]
            geometric_ambiguity |= len(fits) > 1
            for p in fits:
                inside = {(y + t, x + l) for y, x, v in p if v}
                # Different holes never touch. Every module road cell therefore
                # already has all its possible neighbors in this fragment + fixed road.
                if any(sum(q in inside or q in fixed for q in _neighbors(cell)) != 2 for cell in inside):
                    continue
                additions = {cell: sum(q in inside for q in _neighbors(cell)) for cell in fixed}
                if any(fixed_degrees[cell] + additions[cell] > requirements[cell] for cell in fixed):
                    continue
                choices.append((index, p, inside, additions))
        _need(bool(choices), "a cavity has no shape-and-road-compatible piece")
        options.append(choices)
    order = sorted(range(len(holes)), key=lambda i: len(options[i]))
    solutions = {}

    def search(depth, used, track, degrees, assignment):
        if len(solutions) > 1:
            return
        if depth == len(order):
            if degrees != requirements:
                return
            path = _walk(track, start, goal)
            if path is not None:
                solutions.setdefault(frozenset(track), (path, assignment.copy()))
            return
        hi = order[depth]
        for pi, pattern, inside, additions in options[hi]:
            if pi in used:
                continue
            # Identical loose modules are interchangeable. Suppress their index
            # permutations, while retaining every distinct orientation.
            if any(j not in used and keys[j] == keys[pi] for j in range(pi)):
                continue
            new_degrees = {p: degrees[p] + additions[p] for p in fixed}
            if any(new_degrees[p] > requirements[p] for p in fixed):
                continue
            assignment[hi] = (pi, pattern)
            search(depth + 1, used | {pi}, track | inside, new_degrees, assignment)

    search(0, set(), fixed, fixed_degrees, {})
    _need(len(solutions) == 1, "assembly lacks a unique complete road")
    path, assignment = next(iter(solutions.values()))
    logical = board.copy()
    reflection = rotation = 0
    class_counts = Counter(keys)
    for hi, (pi, pattern) in assignment.items():
        t, b, l, r = _bounds(holes[hi])
        for y, x, v in pattern:
            logical[y + t, x + l] = road if v else structural
        # Report only witnesses whose piece class is not interchangeable.
        if class_counts[keys[pi]] == 1:
            if pattern not in _variants(pieces[pi], False):
                reflection += 1
            elif pattern != pieces[pi]:
                rotation += 1
    return logical, path, dict(reflection=reflection, rotation=rotation,
                              geometric_ambiguity=geometric_ambiguity,
                              piece_classes=tuple(sorted(set(keys))))


def _recover(input_grid):
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    _need(np.all(grid[0] == background) and np.all(grid[-1] == background)
          and np.all(grid[:, 0] == background) and np.all(grid[:, -1] == background),
          "the input needs an empty background perimeter")
    _need(background not in (AVAILABLE, SPENT), "counter state cannot be background")
    candidates = []
    for structural in np.unique(grid):
        structural = int(structural)
        if structural in (background, AVAILABLE, SPENT):
            continue
        for cells in _components(grid == structural):
            t, b, l, r = _bounds(cells)
            if min(b - t, r - l) < 7:
                continue
            board = grid[t:b, l:r]
            if not (np.all(board[0] == structural) and np.all(board[-1] == structural)
                    and np.all(board[:, 0] == structural) and np.all(board[:, -1] == structural)):
                continue
            gauges = _gauges(board, background, structural)
            if len(gauges) == 1:
                candidates.append((structural, (t, b, l, r), board.copy(), gauges[0]))
    _need(len(candidates) == 1, "expected one bordered blueprint with one gauge")
    structural, bounds, board, gauge = candidates[0]
    t, b, l, r = bounds
    outside = grid.copy()
    outside[t:b, l:r] = background
    external = _components(outside != background)
    colors = set(map(int, np.unique(outside))) - {background, structural}
    _need(len(colors) == 1, "loose modules must use one road color plus wall color")
    road = colors.pop()
    _need(road not in (AVAILABLE, SPENT, gauge['frame'], gauge['marker']), "conflicting roles")
    holes = _components(board == background)
    _need(2 <= len(holes) <= 4 and len(external) == len(holes), "one loose module per cavity required")
    hole_area = sum(map(len, holes))
    external_area = sum(map(len, external))
    ratio, remainder = divmod(external_area, hole_area)
    scale = isqrt(ratio)
    _need(not remainder and scale in (2, 3) and scale * scale == ratio, "invalid common macrocell scale")
    _need(all(n * scale <= 30 for n in board.shape), "enlarged board is too large")
    pieces = [_decode_piece(grid, cells, scale, background, structural, road) for cells in external]
    _need(all(p is not None for p in pieces), "broken macrocell or invalid module")
    known = {background, structural, road, gauge['frame'], gauge['marker'], AVAILABLE, SPENT}
    cursor_colors = set(map(int, np.unique(board))) - known
    _need(len(cursor_colors) == 1, "missing unique traveler color")
    cursor = cursor_colors.pop()
    starts = np.argwhere(board == cursor)
    _need(len(starts) == 1, "traveler must occupy one logical cell")
    start = tuple(map(int, starts[0]))
    goals = set(map(tuple, np.argwhere(board == gauge['marker']).tolist())) - set(gauge['bar'])
    _need(len(goals) == 1, "expected one goal in addition to the gauge bar")
    goal = goals.pop()
    # Explain every counter/frame pixel and keep the gauge apart from the road.
    gt, gb, gl, gr = gauge['bounds']
    _need(np.count_nonzero(board == gauge['frame']) == 2 * (gb - gt + gr - gl) - 4,
          "unexplained gauge-frame cells")
    _need(np.count_nonzero(np.isin(board, (AVAILABLE, SPENT))) == gauge['capacity'],
          "counter state appears outside gauge")
    _need(all(0 < y < board.shape[0] - 1 and 0 < x < board.shape[1] - 1
              for hole in holes for y, x in hole), "cavity touches border")
    logical, path, assembly = _assemble(board, holes, pieces, structural, road, start, goal)
    heading = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    _need(heading == DELTAS[gauge['direction']], "gauge bar must agree with initial heading")
    movement = min(gauge['remaining'], len(path) - 1)
    for p in path[:movement]:
        logical[p] = SPENT
    logical[path[movement]] = cursor  # On arrival the traveler covers the goal.
    for p in gauge['scan'][:gauge['advanced'] + movement]:
        logical[p] = SPENT
    output = np.repeat(np.repeat(logical, scale, axis=0), scale, axis=1)
    missing_road = {p for p in path if board[p] == background}
    first_gap = min(path.index(p) for p in missing_road)
    turns = sum((path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]) !=
                (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
                for i in range(1, len(path) - 1))
    info = dict(background=background, structural=structural, road=road, cursor=cursor,
                marker=gauge['marker'], frame=gauge['frame'], scale=scale,
                board_shape=tuple(board.shape), board_bounds=bounds, input_shape=tuple(grid.shape),
                module_count=len(holes), module_areas=tuple(sorted(map(len, holes))),
                module_road_lengths=tuple(sorted(sum(v for y, x, v in p) for p in pieces)),
                wall_fill=sum(sum(not v for y, x, v in p) for p in pieces),
                direction=gauge['direction'], advanced=gauge['advanced'],
                capacity=gauge['capacity'], remaining=gauge['remaining'],
                gauge_shape=(gb-gt-2, gr-gl-2), distance=len(path)-1,
                movement=movement, turns=turns, crosses_gap=movement >= first_gap,
                stop='zero' if gauge['remaining'] == 0 else
                'destination' if gauge['remaining'] > len(path)-1 else
                'exact' if gauge['remaining'] == len(path)-1 else 'fuel',
                path=tuple(path), holes=holes, gauge=gauge, **assembly)
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from this input only."""
    return _recover(input_grid)[0]


def _payload_counterfactual(input_grid, info):
    """Change a ring module's short road to its seven-cell perimeter detour.

    Blueprint, support silhouettes, scale, placement, palette and charge all
    remain identical. This is a diagnostic, never part of solve or generation.
    """
    grid = np.asarray(input_grid)
    outside = grid != info['background']
    t, b, l, r = info['board_bounds']
    outside = outside.copy()
    outside[t:b, l:r] = False
    ring_shape = tuple((y, x) for y in range(3) for x in range(3) if (y, x) != (1, 1))
    for cells in _components(outside):
        pattern = _decode_piece(grid, cells, info['scale'], info['background'],
                                info['structural'], info['road'])
        if pattern is None or tuple((y, x) for y, x, v in pattern) != ring_shape:
            continue
        road_cells = sorted((y, x) for y, x, v in pattern if v)
        if len(road_cells) != 3 or not (len({y for y, x in road_cells}) == 1 or
                                       len({x for y, x in road_cells}) == 1):
            continue
        alternative = grid.copy()
        y0, y1, x0, x1 = _bounds(cells)
        middle = road_cells[1]
        k = info['scale']
        for y, x, v in pattern:
            color = info['structural'] if (y, x) == middle else info['road']
            alternative[y0+y*k:y0+(y+1)*k, x0+x*k:x0+(x+1)*k] = color
        return alternative
    return None


def _grow_path(rng, allowed, length, direction, initial_length=2):
    starts = [p for p in sorted(allowed)
              if all((p[0] + j * DELTAS[direction][0], p[1] + j * DELTAS[direction][1]) in allowed
                     for j in range(initial_length))]
    rng.shuffle(starts)
    for start in starts[:16]:
        path = [(start[0] + j * DELTAS[direction][0], start[1] + j * DELTAS[direction][1])
                for j in range(initial_length)]
        occupied, budget = set(path), [2200]

        def extend():
            budget[0] -= 1
            if len(path) == length:
                return True
            if budget[0] <= 0:
                return False
            choices = [p for p in _neighbors(path[-1]) if p in allowed and p not in occupied
                       and sum(q in occupied for q in _neighbors(p)) == 1]
            rng.shuffle(choices)
            for p in choices:
                path.append(p)
                occupied.add(p)
                if extend():
                    return True
                occupied.remove(p)
                path.pop()
            return False

        if extend():
            return path
    return None


def _module_supports(rng, path, count, allowed, bypass_witness=False):
    lengths = [2] * count
    if bypass_witness:
        lengths[0] = 3
    extra = len(path) - (3 * count + 1)
    if bypass_witness:
        extra -= 1
    if extra < 0:
        return None
    for _ in range(int(rng.integers(extra + 1))):
        i = int(rng.integers(count))
        if bypass_witness and i == 0:
            continue
        if lengths[i] < 4:
            lengths[i] += 1
    gaps = [1] * (count + 1)
    for _ in range(len(path) - sum(lengths) - sum(gaps)):
        gaps[int(rng.integers(1 if bypass_witness else 0, count + 1))] += 1
    supports, roads, index = [], [], gaps[0]
    for i, length in enumerate(lengths):
        cells = set(path[index:index + length])
        supports.append(cells.copy())
        roads.append(cells)
        index += length + gaps[i + 1]
    full_path = set(path)
    if bypass_witness:
        rr, cc = {y for y, x in roads[0]}, {x for y, x in roads[0]}
        if not (len(rr) == 1 or len(cc) == 1):
            return None
        candidates = []
        t, b, l, r = _bounds(roads[0])
        origins = [(t-2, l), (t, l)] if len(rr) == 1 else [(t, l-2), (t, l)]
        for y0, x0 in origins:
            ring = {(y0+y, x0+x) for y in range(3) for x in range(3) if (y, x) != (1, 1)}
            center = (y0+1, x0+1)
            other_modules = set().union(*supports[1:])
            if not ring <= allowed or ring & full_path != roads[0] or center in full_path:
                continue
            if ring & other_modules or any(q in other_modules for p in ring for q in _neighbors(p)):
                continue
            middle = sorted(roads[0])[1]
            alternate_road = (full_path - {middle}) | (ring - roads[0])
            if _walk(alternate_road, path[0], path[-1]) is not None:
                candidates.append(ring)
        if not candidates:
            return None
        supports[0] = candidates[int(rng.integers(len(candidates)))]
    for i in rng.permutation(count):
        i = int(i)
        if bypass_witness and i == 0:
            continue
        desired = len(supports[i]) + int(rng.integers(1, 5))
        while len(supports[i]) < desired:
            others = set().union(*(supports[j] for j in range(count) if j != i))
            choices = sorted({q for p in supports[i] for q in _neighbors(p)} - supports[i])
            choices = [p for p in choices if p in allowed and p not in full_path
                       and p not in others and not any(q in others for q in _neighbors(p))
                       and max(_bounds(supports[i] | {p})[1] - _bounds(supports[i] | {p})[0],
                               _bounds(supports[i] | {p})[3] - _bounds(supports[i] | {p})[2]) <= 4]
            if not choices:
                break
            supports[i].add(choices[int(rng.integers(len(choices)))])
        if supports[i] == roads[i]:
            return None
    return supports, roads


def _free_origins(blocked, h, w):
    H, W = blocked.shape
    if h > H or w > W:
        return np.empty((0, 2), dtype=int)
    integral = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    return np.argwhere(sums == 0)


def _pack(rng, board, pieces, scale, palette):
    bg, wall, frame, marker, road, cursor = palette
    objects = [board]
    for pattern in pieces:
        h, w = max(y for y, x, v in pattern) + 1, max(x for y, x, v in pattern) + 1
        a = np.full((h, w), bg, dtype=int)
        for y, x, v in pattern:
            a[y, x] = road if v else wall
        objects.append(np.repeat(np.repeat(a, scale, 0), scale, 1))
    for attempt in range(28):
        H, W = ((int(rng.integers(23, 31)), int(rng.integers(23, 31)))
                if attempt < 12 else (30, 30))
        canvas = np.full((H, W), bg, dtype=int)
        blocked = np.zeros((H, W), dtype=bool)
        blocked[0] = blocked[-1] = True
        blocked[:, 0] = blocked[:, -1] = True
        order = [0] + sorted(range(1, len(objects)), key=lambda i: -objects[i].size)
        for index in order:
            obj = objects[index]
            h, w = obj.shape
            positions = _free_origins(blocked, h, w)
            if not len(positions):
                break
            top, left = map(int, positions[int(rng.integers(len(positions)))])
            canvas[top:top+h, left:left+w] = obj
            blocked[max(0, top-1):top+h+1, max(0, left-1):left+w+1] = True
        else:
            t, b, l, r = _bounds(list(map(tuple, np.argwhere(canvas != bg).tolist())))
            t, l = max(0, t-int(rng.integers(1, 4))), max(0, l-int(rng.integers(1, 4)))
            b, r = min(H, b+int(rng.integers(1, 4))), min(W, r+int(rng.integers(1, 4)))
            return canvas[t:b, l:r].copy()
    return None


def _make_pair(rng, scale, module_count, palette, direction, lesson=None):
    bg, wall, frame, marker, road, cursor = palette
    for attempt in range(150):
        bypass = lesson == 'fuel' or (lesson == 'query' and rng.random() < .3)
        upper = 30 // scale
        low = 10 if scale == 2 else 9
        H, W = int(rng.integers(low, upper+1)), int(rng.integers(low, upper+1))
        if lesson == 'destination':
            a, b = int(rng.integers(3, 5 if scale == 2 else 4)), int(rng.integers(3, 6 if scale == 2 else 5))
            capacity = a * b
            distance = int(rng.integers(6, min(capacity-3, 12)+1))
            # A positive initial prefix and at least two unspent cells after
            # arrival make the two-dimensional raster update observable.
            remaining = int(rng.integers(distance+2, capacity))
        else:
            a, b = int(rng.integers(1, 5)), int(rng.integers(1, 6))
            if lesson == 'fuel':
                a, b = ((2, 2) if scale == 3 else
                        (int(rng.integers(2, 4)), int(rng.integers(2, 5))))
            capacity = a * b
            minimum = 3 * module_count
            if bypass:
                minimum += 1
            extra_distance = 3 if scale == 3 and module_count == 4 else 8
            distance = int(rng.integers(minimum, min(25, minimum+extra_distance)+1))
            if lesson == 'fuel':
                remaining = int(rng.integers(2, min(capacity-1, distance-1)+1))
            elif lesson == 'query':
                if scale == 3:
                    a, b = int(rng.integers(1, 3)), int(rng.integers(3, 5))
                    capacity = a * b
                if capacity < 3:
                    continue
                remaining = int(rng.integers(3, capacity+1))
            else:
                remaining = int(rng.integers(capacity+1))
                if rng.random() < .22:
                    remaining = capacity  # A fresh counter is legal.
            if lesson == 'exact':
                a, b, capacity, distance, remaining = 2, 3, 6, 6, 6
            elif lesson == 'zero':
                remaining = 0
        ih, iw = (a, b) if direction in (0, 2) else (b, a)
        gh, gw = ih + 2, iw + 2
        choices = []
        for t in range(1, H-gh):
            for l in range(1, W-gw):
                bar = _bar(t, t+gh, l, l+gw, direction)
                if all(0 < y < H-1 and 0 < x < W-1 for y, x in bar):
                    choices.append((t, l, bar))
        if not choices:
            continue
        # Mostly near an interior edge, sometimes inset; never a fixed corner.
        if rng.random() < .8:
            edges = [p for p in choices if p[0] in (min(q[0] for q in choices), max(q[0] for q in choices))
                     or p[1] in (min(q[1] for q in choices), max(q[1] for q in choices))]
            choices = edges
        t, l, bar = choices[int(rng.integers(len(choices)))]
        btm, right = t+gh, l+gw
        meter = {(y, x) for y in range(t, btm) for x in range(l, right)} | set(bar)
        reserved = meter | {q for p in meter for q in _neighbors(p)}
        allowed = {(y, x) for y in range(1, H-1) for x in range(1, W-1)} - reserved
        if len(allowed) < distance + 1 + module_count:
            continue
        path = _grow_path(rng, allowed, distance+1, direction, 4 if bypass else 2)
        if path is None:
            continue
        made = _module_supports(rng, path, module_count, allowed, bypass)
        if made is None:
            continue
        supports, roads = made
        if bypass and lesson == 'query':
            middle = sorted(roads[0])[1]
            long_road = (set(path) - {middle}) | (supports[0] - roads[0])
            path = _walk(long_road, path[0], path[-1])
            roads[0] = supports[0] - {middle}
            distance = len(path) - 1
        board = np.full((H, W), wall, dtype=int)
        board[t:btm, l:right] = frame
        scan = _scan(t, btm, l, right, direction)
        advanced = capacity - remaining
        for i, p in enumerate(scan):
            board[p] = SPENT if i < advanced else AVAILABLE
        for p in bar:
            board[p] = marker
        for p in path:
            board[p] = road
        board[path[0]], board[path[-1]] = cursor, marker
        logical = board.copy()
        movement = min(remaining, distance)
        for p in path[:movement]:
            logical[p] = SPENT
        logical[path[movement]] = cursor
        for p in scan[:advanced+movement]:
            logical[p] = SPENT
        pieces = []
        for i, (support, road_cells) in enumerate(zip(supports, roads)):
            pattern = _pattern_normal([(y, x, int((y, x) in road_cells)) for y, x in support])
            variants = _variants(pattern)
            if i == 0 and lesson == 'destination':
                mirrored = [v for v in variants if v not in _variants(pattern, False)]
                variants = mirrored or variants
            elif i == 0 and lesson == 'fuel':
                turned = [v for v in _variants(pattern, False) if v != pattern]
                variants = turned or variants
            pieces.append(variants[int(rng.integers(len(variants)))])
            for p in support:
                board[p] = bg
        canvas = _pack(rng, board, pieces, scale, palette)
        if canvas is None:
            continue
        expected = np.repeat(np.repeat(logical, scale, 0), scale, 1)
        try:
            recovered, info = _recover(canvas)
        except ValueError:
            continue
        if not np.array_equal(expected, recovered):
            continue
        if lesson == 'destination' and (not info['reflection'] or not info['geometric_ambiguity']):
            continue
        if lesson == 'fuel' and (not info['rotation'] or not info['turns']):
            continue
        if lesson == 'query' and not info['crosses_gap']:
            continue
        if not info['turns']:
            continue
        return dict(input=canvas, output=expected)
    return _fallback_pair(rng, scale, module_count, palette, direction, lesson)


def _fallback_pair(rng, scale, count, palette, direction, lesson):
    """A feasible small construction of exactly the same rule, after rejection.

    These paths are used only as a bounded fallback, not as the main vocabulary.
    Even here the input-only assembly search must find one colored road.
    """
    bg, wall, frame, marker, road, cursor = palette
    board = np.full((10, 10), wall, dtype=int)
    if lesson in ('destination', 'exact'):
        ih, iw, remaining = (3, 3, 8) if lesson == 'destination' else (2, 3, 6)
        path = [(5, 8), (6, 8), (7, 8), (8, 8), (8, 7), (8, 6), (8, 5)]
        supports = [{(5, 7), (6, 7), (6, 8), (7, 8)}, {(7, 6), (8, 6), (8, 7)}]
    else:
        ih, iw = (2, 2) if lesson in ('query', 'fuel') else (2, 3)
        remaining = ih * iw if lesson == 'query' else 2 if lesson == 'fuel' else 0 if lesson == 'zero' else 3
        path = [(y, 8) for y in range(1, 9)] + [(8, x) for x in range(7, 0, -1)]
        if lesson == 'fuel':
            supports = [{(y, x) for y in range(2, 5) for x in range(6, 9) if (y, x) != (3, 7)},
                        {(7, 4), (7, 5), (8, 5), (8, 6)}]
        else:
            supports = [{(1, 7), (2, 7), (2, 8), (3, 8)},
                        {(5, 7), (5, 8), (6, 8)},
                        {(7, 6), (8, 6), (8, 7)},
                        {(7, 3), (8, 3), (8, 4)}][:count]
    for p in path:
        board[p] = road
    board[path[0]], board[path[-1]] = cursor, marker
    t, b, l, r = 1, ih+3, 1, iw+3
    board[t:b, l:r] = frame
    scan = _scan(t, b, l, r, 2)
    advanced = ih * iw - remaining
    for i, p in enumerate(scan):
        board[p] = SPENT if i < advanced else AVAILABLE
    for p in _bar(t, b, l, r, 2):
        board[p] = marker
    full = board.copy()
    movement = min(remaining, len(path)-1)
    for p in path[:movement]:
        full[p] = SPENT
    full[path[movement]] = cursor
    for p in scan[:advanced+movement]:
        full[p] = SPENT
    turns = (2 - direction) % 4
    # Extract module contents before rotating the blueprint. Loose orientation
    # remains independent of its slot's orientation.
    pieces = []
    for i, support in enumerate(supports):
        pattern = _pattern_normal([(y, x, int(board[y, x] == road)) for y, x in support])
        options = _variants(pattern)
        if lesson == 'destination' and i == 0:
            options = tuple(p for p in options if p not in _variants(pattern, False))
        elif lesson == 'fuel' and i == 0:
            target = pattern
            for _ in range(turns):
                target = _pattern_normal([(-x, y, v) for y, x, v in target])
            options = tuple(p for p in _variants(target, False) if p != target)
        pieces.append(options[int(rng.integers(len(options)))])
        for p in support:
            board[p] = bg
    board, full = np.rot90(board, turns).copy(), np.rot90(full, turns).copy()
    canvas = _pack(rng, board, pieces, scale, palette)
    if canvas is None:
        canvas = np.full((30, 30), bg, dtype=int)
        canvas[1:11, 1:11] = board
        for pattern, (y0, x0) in zip(pieces, ((13, 1), (13, 13), (1, 13), (1, 23))):
            for y, x, v in pattern:
                canvas[y0+y*scale:y0+(y+1)*scale, x0+x*scale:x0+(x+1)*scale] = road if v else wall
    expected = np.repeat(np.repeat(full, scale, 0), scale, 1)
    recovered, info = _recover(canvas)
    _need(np.array_equal(expected, recovered), "fallback does not obey assembly and motion")
    return dict(input=canvas, output=expected)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    _need(num_train > num_test, "num_train must exceed num_test")
    _need(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)) and seed >= 0,
          "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))
    base_scale = int(rng.choice((2, 3)))
    if mode == 0:
        train_scales, test_scales = [2]*num_train, [3]*num_test
    elif mode == 1:
        train_scales = [2, 3] + [int(rng.choice((2, 3))) for _ in range(num_train-2)]
        rng.shuffle(train_scales)
        test_scales = [int(rng.choice((2, 3))) for _ in range(num_test)]
    else:
        train_scales, test_scales = [base_scale]*num_train, [base_scale]*num_test
    shared_palette = bool(rng.integers(2))
    colors = [c for c in range(10) if c not in (AVAILABLE, SPENT)]
    palette = tuple(map(int, rng.choice(colors, 6, replace=False)))
    lesson_positions = list(map(int, rng.choice(num_train, 2, replace=False)))
    initial_heading = int(rng.integers(4))
    lessons = {lesson_positions[0]: ('destination', initial_heading),
               lesson_positions[1]: ('fuel', (initial_heading + int(rng.choice((1, 3)))) % 4)}
    puzzle, seen = dict(train=[], test=[]), set()
    for split, scales in (('train', train_scales), ('test', test_scales)):
        for i, scale in enumerate(scales):
            lesson, direction = lessons[i] if split == 'train' and i in lessons else (None, int(rng.integers(4)))
            if split == 'train' and lesson is None:
                lesson = rng.choice((None, 'exact', 'zero'), p=(.5, .25, .25))
            if split == 'test' and i == 0:
                lesson = 'query'
            count = 2 if lesson in ('destination', 'fuel', 'exact') else int(rng.choice((2, 3) if split == 'train' else ((4,) if mode == 2 else (3, 4))))
            if split == 'test' and i == 0 and mode != 0:
                count = 4
            for _ in range(8):
                local_palette = palette if shared_palette else tuple(map(int, rng.choice(colors, 6, replace=False)))
                pair = _make_pair(rng, scale, count, local_palette, direction, lesson)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _count(len(puzzle['train']), 'train count', 2, 10)
    _count(len(puzzle['test']), 'test count', 1, 4)
    _need(len(puzzle['train']) > len(puzzle['test']), "invalid split counts")
    seen, train, test, payload_witness, counter_order_witness = set(), [], [], False, False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            _need(key not in seen, "duplicate input")
            seen.add(key)
            expected, info = _recover(inp)
            _need(np.array_equal(out, expected), "incorrect road, traveler or counter output")
            # The output remains made entirely of uniform inferred-scale blocks.
            k = info['scale']
            _need(np.array_equal(out, np.repeat(np.repeat(out[::k, ::k], k, 0), k, 1)), "output breaks macrocell scale")
            _need(np.count_nonzero(out == SPENT) == (info['advanced'] + 2 * info['movement']) * k * k,
                  "meter and trail do not conserve spent units")
            _need(np.count_nonzero(out == AVAILABLE) == (info['remaining'] - info['movement']) * k * k,
                  "available charge is not conserved")
            _need(np.count_nonzero(out == info['cursor']) == k * k, "output needs exactly one macrocell traveler")
            _need(not np.any(out == info['background']), "unfilled background remains in output board")
            _need(info['wall_fill'] >= info['module_count'], "module walls were not meaningful")
            _need(info['turns'] >= 1, "road must bend")
            if split == 'train':
                bt, bb, bl, br = info['board_bounds']
                gt, gb, gl, gr = info['gauge']['bounds']
                initial = inp[bt+gt+1:bt+gb-1, bl+gl+1:bl+gr-1]
                canonical = np.rot90(initial, info['direction']).copy()
                pending = info['movement']
                for x in range(canonical.shape[1]):
                    for y in range(canonical.shape[0]):
                        if pending and canonical[y, x] == AVAILABLE:
                            canonical[y, x], pending = SPENT, pending - 1
                correct = out[::k, ::k][gt+1:gb-1, gl+1:gr-1]
                counter_order_witness |= not np.array_equal(canonical, np.rot90(correct, info['direction']))
            if split == 'train' and info['stop'] == 'fuel':
                counterfactual = _payload_counterfactual(inp, info)
                if counterfactual is not None:
                    changed, changed_info = _recover(counterfactual)
                    payload_witness |= (changed_info['distance'] == info['distance'] + 4
                                        and changed_info['remaining'] == info['remaining']
                                        and changed_info['path'][info['movement']] != info['path'][info['movement']]
                                        and not np.array_equal(changed, out))
            (train if split == 'train' else test).append(info)
    _need({'destination', 'fuel'} <= {i['stop'] for i in train}, "training needs both stopping causes")
    _need(len({i['direction'] for i in train if i['movement']}) >= 2, "training must rotate the counter")
    _need(any(i['crosses_gap'] and i['movement'] > 0 for i in train), "training must travel through an assembled module")
    _need(any(i['reflection'] for i in train), "training must teach reflected modules")
    _need(any(i['rotation'] for i in train), "training must teach rotated modules")
    _need(any(i['geometric_ambiguity'] for i in train), "training must demonstrate road continuity resolving shape symmetry")
    _need(payload_witness, "training must show a road chosen by module payload, not just its silhouette")
    _need(counter_order_witness, "training must distinguish row-major from column-major spending")
    _need(any(min(i['gauge_shape']) > 1 and 0 < i['advanced'] < i['capacity'] for i in train),
          "training must expose a two-dimensional counter prefix")
    _need(min(i['module_count'] for i in test) >= 3, "tests should recombine more modules")
    _need(test[0]['movement'] >= 3 and test[0]['crosses_gap'], "first test must move through a repaired gap")


def self_test():
    """Bounded 42-seed checks of actual recovered geometry and semantics."""
    signatures, infos, normalized, structures = set(), [], set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(tuple((_key(p['input']), _key(p['output']))
                             for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                out, info = _recover(pair['input'])
                assert np.array_equal(out, pair['output'])
                infos.append((split, info))
                roles = {}
                normalized.add((pair['input'].shape, tuple(roles.setdefault(int(c), len(roles))
                                                           for c in pair['input'].flat)))
                structures.update(info['piece_classes'])
    assert len(signatures) == 42 and len(normalized) >= 180
    assert len({i['input_shape'] for s, i in infos}) >= 25
    assert len({i['board_shape'] for s, i in infos}) >= 20
    assert len(structures) >= 40
    assert {i['scale'] for s, i in infos} == {2, 3}
    assert {i['module_count'] for s, i in infos} == {2, 3, 4}
    assert {i['direction'] for s, i in infos} == set(range(4))
    assert {'zero', 'fuel', 'exact', 'destination'} <= {i['stop'] for s, i in infos}
    assert any(i['advanced'] == 0 for s, i in infos)
    assert any(min(i['gauge_shape']) == 1 for s, i in infos)
    assert any(i['distance'] >= 20 for s, i in infos)
    assert any(i['geometric_ambiguity'] for s, i in infos if s == 'test')
    assert any(7 in i['module_road_lengths'] for s, i in infos if s == 'test')
    for split in ('train', 'test'):
        local = [i for s, i in infos if s == split]
        assert all(any(i[f] for i in local) for f in ('reflection', 'rotation', 'crosses_gap'))
        assert {i['direction'] for i in local} == set(range(4))
    # Determinism and every legal count combination, including the minimum
    # demonstration set. Count checks do not rely on the external verifier.
    a, b = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(12345, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Negative output and block-corruption diagnostics.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    inp = a['test'][0]['input'].copy()
    _, inf = _recover(inp)
    t, b, l, r = inf['board_bounds']
    external = inp != inf['background']
    external[t:b, l:r] = False
    p = tuple(map(int, np.argwhere(external)[0]))
    inp[p] = inf['background']
    try:
        solve(inp)
    except ValueError:
        pass
    else:
        raise AssertionError("broken loose macrocell accepted")
    # A ring with opposite-corner ports and a five-cell L road has two
    # reflected valid wirings. Reject the ambiguity, rather than choosing one.
    diagnostic = _fallback_pair(np.random.default_rng(234), 2, 2, (0, 5, 1, 9, 6, 8), 2, 'fuel')['input']
    _, inf = _recover(diagnostic)
    bt, bb, bl, br = inf['board_bounds']
    board = diagnostic[bt:bb, bl:br]
    board[board == 6] = 5
    fixed_path = [(1, 8), (5, 6), (6, 6), (7, 6)] + [(8, x) for x in range(6, 0, -1)]
    for p in fixed_path:
        if board[p] == 5:
            board[p] = 6
    outside = diagnostic != 0
    outside[bt:bb, bl:br] = False
    for cells in _components(outside):
        pattern = _decode_piece(diagnostic, cells, 2, 0, 5, 6)
        if len(pattern) != 8:
            continue
        y0, y1, x0, x1 = _bounds(cells)
        for y, x, v in pattern:
            diagnostic[y0+2*y:y0+2*(y+1), x0+2*x:x0+2*(x+1)] = 6 if x == 2 or y == 2 else 5
        break
    try:
        solve(diagnostic)
    except ValueError as error:
        assert 'unique complete road' in str(error)
    else:
        raise AssertionError('two different complete wirings were accepted')
    # The bounded fallback obeys the same counterfactual rule in every orientation.
    for scale in (2, 3):
        for direction in range(4):
            for count, lesson in ((2, 'destination'), (2, 'fuel'), (2, 'exact'), (3, None), (4, 'query')):
                pair = _fallback_pair(np.random.default_rng(430+direction), scale, count,
                                      (0, 5, 1, 9, 6, 8), direction, lesson)
                assert np.array_equal(solve(pair['input']), pair['output'])
                if lesson == 'fuel':
                    _, inf = _recover(pair['input'])
                    _, changed = _recover(_payload_counterfactual(pair['input'], inf))
                    assert changed['distance'] == inf['distance'] + 4
                    assert changed['path'][changed['movement']] != inf['path'][inf['movement']]


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds and all legal counts')
