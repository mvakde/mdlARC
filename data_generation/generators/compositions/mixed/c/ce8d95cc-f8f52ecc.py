"""Canonical checkpoint routing: equality-run contraction precedes path search.

Only the standard library and NumPy are used. Search state is transient; no
examples, generation parameters, or results are stored for use by solve().
"""
from collections import deque
from heapq import heappop, heappush
import hashlib
import json

import numpy as np


BARRIER = 8
HOST_SEEDS = (0, 960990893, 98175810, 15177078, 481067368, 373821712,
              81500023, 771683996, 213815889, 732241601, 640461358,
              798685517, 372205815, 646228181, 836815361, 406859342,
              985681334, 487556669, 178210270, 482606795, 120913659,
              254695843, 897928391, 210112663, 682721787, 791932396,
              355153489, 945723330, 795952754, 896591937, 709322537,
              539695921, 912644825, 221551365, 247275771, 448918214,
              563553113, 984847691, 219743356, 954754696, 583797301, 705935769)


class _SearchBudget(Exception):
    """Reject an expensive construction attempt, never choose a partial route."""


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _compress(a, indices=False):
    # Equality is of entire vectors, and only consecutive equality counts.
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    cols = np.r_[True, np.any(a[:, 1:] != a[:, :-1], axis=0)]
    rr, cc = np.flatnonzero(rows), np.flatnonzero(cols)
    result = a[np.ix_(rr, cc)].copy()
    return (result, rr, cc) if indices else result


def _background(a):
    counts = np.bincount(a.ravel(), minlength=10)
    color = int(np.argmax(counts))
    if color == BARRIER or int(counts[color]) * 2 <= a.size:
        raise ValueError("the background must occupy a strict majority")
    return color


def _neighbors(shape):
    h, w = shape
    return [tuple(rr * w + cc for rr, cc in
                  ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                  if 0 <= rr < h and 0 <= cc < w)
            for r in range(h) for c in range(w)]


def _components(mask):
    neighbors = _neighbors(mask.shape)
    left = set(map(int, np.flatnonzero(mask.ravel())))
    parts = []
    while left:
        start = min(left)
        left.remove(start)
        todo, part = [start], []
        while todo:
            point = todo.pop()
            part.append(point)
            for other in neighbors[point]:
                if other in left:
                    left.remove(other)
                    todo.append(other)
        parts.append(tuple(sorted(part)))
    return parts


def _ordered(part, neighbors):
    cells = set(part)
    if len(cells) == 1:
        return tuple(part)
    ends = sorted(p for p in cells if sum(q in cells for q in neighbors[p]) == 1)
    if len(ends) != 2 or any(sum(q in cells for q in neighbors[p]) > 2 for p in cells):
        raise ValueError("checkpoint/output component is not a simple nonbranching path")
    path, previous, point = [], None, ends[0]
    while True:
        path.append(point)
        options = [q for q in neighbors[point] if q in cells and q != previous]
        if not options:
            break
        previous, point = point, options[0]
        if point in path:
            raise ValueError("cycle")
    if len(path) != len(cells):
        raise ValueError("disconnected path")
    return tuple(path)


def _bends(path, width):
    directions = [(b // width - a // width, b % width - a % width)
                  for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(directions, directions[1:]))


def _route(a, color, bg, budget=None, rank_bends=True):
    """Exact lexicographic search of induced simple paths through all fragments.

    The heuristic is a relaxed background-cell distance to a remaining fragment
    plus a minimum spanning tree on remaining fragments. It is a lower bound,
    not a route choice. Reversal counts as the same path. No bend need be marked.
    A construction-only budget raises rather than silently breaking a tie.
    rank_bends=False is a validator ablation: it rejects ties in length alone.
    """
    h, w = a.shape
    neighbors = _neighbors(a.shape)
    values = a.ravel()
    fragments = [_ordered(part, neighbors) for part in _components(a == color)]
    if len(fragments) <= 1:
        return fragments[0] if fragments else ()
    count = len(fragments)
    all_mask = (1 << count) - 1
    endpoints = {}
    for i, part in enumerate(fragments):
        endpoints[part[0]] = endpoints[part[-1]] = i

    # 0/1 distances ignore path self-contact and fragment visitation order.
    # Entering a background cell costs one; entering a checkpoint costs zero.
    distances = []
    for part in fragments:
        dist = [10 ** 6] * a.size
        queue = deque(set((part[0], part[-1])))
        for point in queue:
            dist[point] = 0
        while queue:
            point = queue.popleft()
            for other in neighbors[point]:
                if int(values[other]) not in (bg, color):
                    continue
                cost = int(values[other] == bg)
                if dist[point] + cost < dist[other]:
                    dist[other] = dist[point] + cost
                    if cost:
                        queue.append(other)
                    else:
                        queue.appendleft(other)
        distances.append(dist)
    pair_dist = [[min(distances[i][fragments[j][0]],
                          distances[i][fragments[j][-1]])
                  for j in range(count)] for i in range(count)]
    mst_cache, bound_cache = {}, {}

    def mst(mask):
        if mask in mst_cache:
            return mst_cache[mask]
        left = [i for i in range(count) if mask & (1 << i)]
        if len(left) <= 1:
            return 0
        tree = [left.pop()]
        cost = 0
        while left:
            distance, chosen = min((pair_dist[i][j], j) for i in tree for j in left)
            cost += distance
            tree.append(chosen)
            left.remove(chosen)
        mst_cache[mask] = cost
        return cost

    def bound(point, mask):
        if mask == all_mask:
            return 0
        key = point, mask
        if key not in bound_cache:
            remaining = all_mask ^ mask
            closest = min(distances[i][point] for i in range(count)
                          if remaining & (1 << i)) - int(values[point] == bg)
            bound_cache[key] = max(0, closest) + mst(remaining)
        return bound_cache[key]

    def append(path, visited, cells, added, turns, direction):
        for point in cells:
            if visited & (1 << point):
                return None
            if path:
                previous = path[-1]
                if (previous not in neighbors[point]
                        or sum(bool(visited & (1 << q)) for q in neighbors[point]) != 1):
                    return None
                new_dir = (point // w - previous // w, point % w - previous % w)
                turns += int(rank_bends and direction is not None and new_dir != direction)
                direction = new_dir
            added += int(values[point] == bg)
            path += (point,)
            visited |= 1 << point
        return path, visited, added, turns, direction

    heap, serial, expanded = [], 0, 0
    seen = {}

    def push(mask, state):
        nonlocal serial
        path, visited, added, turns, direction = state
        state_key = mask, visited, path[-1], direction
        if seen.get(state_key, 10 ** 6) < turns:
            return
        seen[state_key] = turns
        heappush(heap, (added + bound(path[-1], mask), turns, serial,
                        added, mask, path, visited, direction))
        serial += 1

    for i, part in enumerate(fragments):
        for oriented in sorted(set((part, part[::-1]))):
            state = append((), 0, oriented, 0, 0, None)
            if state is not None:
                push(1 << i, state)
    best, answers = None, set()
    while heap:
        lower, turns, _, added, mask, path, visited, direction = heappop(heap)
        if best is not None and (lower, turns) > best:
            break
        if mask == all_mask:
            candidate = min(path, path[::-1])
            best = added, turns
            answers.add(candidate)
            if len(answers) > 1:
                raise ValueError("tied shortest minimum-bend checkpoint paths")
            continue
        expanded += 1
        if budget is not None and (expanded > budget or len(heap) > 4 * budget):
            raise _SearchBudget()
        for other in neighbors[path[-1]]:
            if visited & (1 << other):
                continue
            if int(values[other]) == bg:
                cells, new_mask = (other,), mask
            elif int(values[other]) == color and other in endpoints:
                owner = endpoints[other]
                if mask & (1 << owner):
                    continue
                part = fragments[owner]
                cells = part if other == part[0] else part[::-1]
                new_mask = mask | (1 << owner)
            else:
                continue
            state = append(path, visited, cells, added, turns, direction)
            if state is not None:
                if best is None or (state[2] + bound(state[0][-1], new_mask), state[3]) <= best:
                    push(new_mask, state)
    if len(answers) != 1:
        raise ValueError("no nonbranching checkpoint completion")
    return next(iter(answers))


def _complete(a, budget=None):
    bg = _background(a)
    result = a.copy()
    claimed = set()
    for color in sorted(set(map(int, a.ravel())) - {bg, BARRIER}):
        if len(_components(a == color)) <= 1:
            continue
        path = _route(a, color, bg, budget)
        if claimed.intersection(path):
            raise ValueError("independently optimal paths overlap")
        claimed.update(path)
        for point in path:
            result.flat[point] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Contract the input once, then complete paths in that canonical metric."""
    canonical = _compress(_array(input_grid))
    return _complete(canonical)


def _transform(a, rotation, mirror):
    return np.rot90(np.fliplr(a) if mirror else a, rotation).copy()


def _expand(a, rng, heavy=False, both=False):
    runs = []
    for n in a.shape:
        extra = int(rng.integers(7 if heavy else 1, min(30 - n, 18 if heavy else 11) + 1))
        lengths = np.ones(n, dtype=int)
        mode = int(rng.integers(3))
        if mode == 0:
            lengths += rng.multinomial(extra, np.ones(n) / n)
        else:
            target = int(rng.choice((0, n - 1))) if mode == 1 else int(rng.integers(n))
            bulk = int(rng.integers(1, extra + 1))
            lengths[target] += bulk
            lengths += rng.multinomial(extra - bulk, np.ones(n) / n)
        runs.append(lengths)
    heavy_axis = None
    if heavy:
        heavy_axis = int(rng.integers(2))
        n = a.shape[heavy_axis]
        lengths = np.ones(n, dtype=int)
        target = int(rng.integers(n))
        lengths[target] = int(rng.integers(13, min(16, 31 - n) + 1))
        remainder = int(rng.integers(0, min(3, 30 - int(lengths.sum())) + 1))
        lengths += rng.multinomial(remainder, np.ones(n) / n)
        runs[heavy_axis] = lengths
    if not both and rng.random() < 0.2:
        # Select one no-op axis without coupling its length to the other axis.
        axis = 1 - heavy_axis if heavy else int(rng.integers(2))
        runs[axis] = np.ones(a.shape[axis], dtype=int)
    return np.repeat(np.repeat(a, runs[0], axis=0), runs[1], axis=1)


def _fork(rng, bg, color):
    # Right is shorter in canonical cells; stretch one right-hand band until
    # left is shorter in raw cells. Alternating teeth prevent accidental runs.
    left = int(rng.integers(4, 7))
    right = 2
    a = np.full((5, left + right + 1), bg, dtype=np.int8)
    a[2, 1:-1] = BARRIER
    a[0, 1:left:2] = BARRIER
    # Block the direct first step on the near side. Its shortest route now
    # has three turns; the farther side has only two, but uses more cells.
    a[1, left + 1] = BARRIER
    a[1, left] = a[3, left] = color
    rows = np.array([int(rng.integers(2, 5)), 1, 1, 1, int(rng.integers(1, 5))])
    cols = np.ones(a.shape[1], dtype=int)
    cols[left + 1] += left - right + int(rng.integers(2, 6))
    raw = np.repeat(np.repeat(a, rows, axis=0), cols, axis=1)
    rot, mirror = int(rng.integers(4)), bool(rng.integers(2))
    return _transform(raw, rot, mirror)


def _bend_anchor(rng, bg, colors):
    c, d, inert = colors[:3]
    a = np.full((5, 6), bg, dtype=np.int8)
    a[0, 1] = a[2, 3] = a[4, 2] = BARRIER
    a[1, 1:3] = c
    a[3, 5] = c
    a[0::2, 0] = d
    a[4, 5] = inert
    a = _transform(a, int(rng.integers(4)), bool(rng.integers(2)))
    return _expand(a, rng, both=True)


def _polyline(rng, occupied):
    h, w = occupied.shape
    neighbors = _neighbors(occupied.shape)
    for _ in range(120):
        r, c = int(rng.integers(h)), int(rng.integers(w))
        path = [r * w + c]
        axis = int(rng.integers(2))
        for segment in range(int(rng.integers(1, 6))):
            length = int(rng.integers(2, 6))
            sign = int(rng.choice((-1, 1)))
            for _ in range(length):
                r, c = (r + sign, c) if axis == 0 else (r, c + sign)
                if not (0 <= r < h and 0 <= c < w):
                    break
                path.append(r * w + c)
            else:
                axis = 1 - axis
                continue
            break
        else:
            if len(set(path)) != len(path) or any(occupied.flat[p] for p in path):
                continue
            try:
                _ordered(path, neighbors)
            except ValueError:
                continue
            return tuple(path)
    return None


def _scene(rng, bg, colors, test=False):
    h, w = int(rng.integers(9, 16)), int(rng.integers(9, 16))
    a = np.full((h, w), bg, dtype=np.int8)
    latent = a.copy()
    routes = []
    n_routes = int(rng.integers(2 if test else 1, 4))
    for color in colors[:n_routes]:
        path = _polyline(rng, latent != bg)
        if path is None:
            return None
        for p in path:
            latent.flat[p] = color
        routes.append((color, path))
    # Bars and occasional bent obstacles, independent of path orientation.
    for _ in range(int(rng.integers(0, 5))):
        r, c = int(rng.integers(h)), int(rng.integers(w))
        dr, dc = ((1, 0) if rng.integers(2) else (0, 1))
        cells = [(r + dr * k, c + dc * k) for k in range(int(rng.integers(1, 6)))]
        if rng.random() < 0.3:
            rr, cc = cells[-1]
            cells += [(rr + dc * k, cc + dr * k) for k in range(1, int(rng.integers(2, 4)))]
        if all(0 <= rr < h and 0 <= cc < w and latent[rr, cc] == bg for rr, cc in cells):
            for cell in cells:
                a[cell] = latent[cell] = BARRIER
    for color in colors[n_routes:n_routes + int(rng.integers(0, 3))]:
        free = np.flatnonzero((latent == bg).ravel())
        p = int(rng.choice(free))
        a.flat[p] = latent.flat[p] = color
        # A single connected inert fragment can contain more than one cell.
        if rng.random() < 0.4:
            choices = [q for q in _neighbors(a.shape)[p] if latent.flat[q] == bg]
            if choices:
                q = int(rng.choice(choices))
                a.flat[q] = latent.flat[q] = color
    for color, path in routes:
        length = len(path)
        marked = {0, length - 1}
        if rng.random() < 0.18:
            marked = set(range(length)) - {int(rng.integers(1, length - 1))}
        else:
            front, back = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            marked.update(range(min(front, length)))
            marked.update(range(max(0, length - back), length))
            for index in range(2, length - 2):
                if rng.random() < 0.16:
                    marked.add(index)
        for index in marked:
            a.flat[path[index]] = color
    canonical, rr, cc = _compress(a, indices=True)
    target = latent[np.ix_(rr, cc)]
    if any(len(_components(canonical == color)) < 2 for color, _ in routes):
        return None
    if any(len(_components(canonical == color)) > 5 for color, _ in routes):
        return None
    try:
        result = _complete(canonical, budget=12000)
        if not np.array_equal(result, target):
            return None
        for color, _ in routes:
            parts = _components(result == color)
            if len(parts) != 1:
                return None
            _ordered(parts[0], _neighbors(result.shape))
        raw = _expand(canonical, rng, heavy=test)
        if _background(raw) != bg:
            return None
        return raw, result
    except (ValueError, _SearchBudget):
        return None


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    nonbarrier = [int(c) for c in rng.permutation([c for c in range(10) if c != BARRIER])]
    backgrounds, ordinary = nonbarrier[:2], nonbarrier[2:]
    puzzle, seen = {"train": [], "test": []}, set()
    anchor_color = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            bg = backgrounds[index] if split == "train" and index < 2 else int(rng.choice(backgrounds))
            colors = [int(c) for c in rng.permutation(ordinary)]
            if split == "train" and index == 1:
                other_colors = [c for c in colors if c != anchor_color]
                colors = other_colors[:2] + [anchor_color] + other_colors[2:]
            for _ in range(240):
                if split == "train" and index == 0:
                    raw = _fork(rng, bg, colors[0])
                    pair = raw, solve(raw)
                elif split == "train" and index == 1:
                    raw = _bend_anchor(rng, bg, colors)
                    pair = raw, solve(raw)
                else:
                    pair = _scene(rng, bg, colors, test=split == "test")
                    if pair is None:
                        continue
                raw, result = pair
                try:
                    if _background(raw) != bg:
                        continue
                except ValueError:
                    continue
                key = _key(raw)
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append({"input": raw, "output": result})
                    if split == "train" and index == 0:
                        anchor_color = colors[0]
                    break
            else:
                raise RuntimeError("bounded attempts exhausted while constructing a unique scene")
    return puzzle


def _analyze(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, output = _array(pair["input"]), _array(pair["output"])
    canonical = _compress(a)
    bg = _background(canonical)
    assert _background(a) == bg
    assert output.shape == canonical.shape
    assert a.shape != canonical.shape
    assert np.array_equal(output[canonical != bg], canonical[canonical != bg])
    assert np.array_equal(output == BARRIER, canonical == BARRIER)
    assert set(map(int, output.ravel())) <= set(map(int, canonical.ravel()))
    assert np.array_equal(solve(a.copy()), output)
    neighbors = _neighbors(output.shape)
    active, bends, fragments, hidden, visible_runs = [], [], [], 0, []
    for color in sorted(set(map(int, canonical.ravel())) - {bg, BARRIER}):
        parts = _components(canonical == color)
        if len(parts) == 1:
            assert np.array_equal(canonical == color, output == color)
            continue
        active.append(color)
        completed = _components(output == color)
        assert len(completed) == 1
        path = _ordered(completed[0], neighbors)
        assert set(_route(canonical, color, bg)) == set(path)
        assert len(path) > sum(map(len, parts))
        bends.append(_bends(path, output.shape[1]))
        fragments.append(len(parts))
        visible_runs.append(max(map(len, parts)))
        for x, y, z in zip(path, path[1:], path[2:]):
            if ((y // output.shape[1] - x // output.shape[1], y % output.shape[1] - x % output.shape[1])
                    != (z // output.shape[1] - y // output.shape[1], z % output.shape[1] - y % output.shape[1])):
                hidden += int(canonical.flat[y] == bg)
    assert active
    return {"background": bg, "active": len(active), "bends": bends,
            "fragments": fragments, "hidden_turns": hidden, "visible_runs": visible_runs,
            "canonical": canonical, "raw": a}


def _axis_runs(a, axis):
    vectors = a if axis == 0 else a.T
    starts = np.flatnonzero(np.r_[True, np.any(vectors[1:] != vectors[:-1], axis=1)])
    return tuple(map(int, np.diff(np.r_[starts, len(vectors)])))


def _longer_smoother_witness(a, output):
    """Look for an explicit legal, longer, two-turn alternative in anchor 1."""
    bg, w = _background(a), a.shape[1]
    colors = sorted(set(map(int, a.ravel())) - {bg, BARRIER})
    if len(colors) != 1:
        return False
    color = colors[0]
    checkpoints = list(map(int, np.flatnonzero((a == color).ravel())))
    if len(checkpoints) != 2:
        return False
    chosen = _ordered(_components(output == color)[0], _neighbors(a.shape))
    u, v = (divmod(p, w) for p in checkpoints)
    corners = ([(u, (u[0], c), (v[0], c), v) for c in range(a.shape[1])]
               + [(u, (r, u[1]), (r, v[1]), v) for r in range(a.shape[0])])
    for keys in corners:
        points = [keys[0]]
        for x, y in zip(keys, keys[1:]):
            distance = abs(x[0] - y[0]) + abs(x[1] - y[1])
            dr = (y[0] > x[0]) - (y[0] < x[0])
            dc = (y[1] > x[1]) - (y[1] < x[1])
            points.extend((x[0] + dr * k, x[1] + dc * k) for k in range(1, distance + 1))
        path = tuple(r * w + c for r, c in points)
        if len(path) != len(set(path)) or any(int(a.flat[p]) not in (bg, color) for p in path):
            continue
        try:
            _ordered(path, _neighbors(a.shape))
        except ValueError:
            continue
        if len(path) > len(chosen) and _bends(path, w) < _bends(chosen, w):
            return True
    return False


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), []
    for pair in puzzle["train"] + puzzle["test"]:
        analysis = _analyze(pair)
        key = _key(analysis["raw"])
        assert key not in seen
        seen.add(key)
        analyses.append(analysis)
    assert analyses[0]["background"] != analyses[1]["background"]
    assert analyses[0]["hidden_turns"] == 3 and analyses[0]["bends"] == [3]
    assert _longer_smoother_witness(analyses[0]["canonical"], puzzle["train"][0]["output"])
    assert analyses[1]["active"] == 2
    first_bg = analyses[0]["background"]
    first_color = next(iter(set(map(int, analyses[0]["canonical"].ravel())) - {first_bg, BARRIER}))
    assert len(_components(analyses[1]["canonical"] == first_color)) == 1
    assert 1 in analyses[1]["bends"] and 3 in analyses[1]["fragments"]
    bend_demo = analyses[1]["canonical"]
    bg = analyses[1]["background"]
    saw_length_tie = False
    for color in sorted(set(map(int, bend_demo.ravel())) - {bg, BARRIER}):
        if len(_components(bend_demo == color)) <= 1:
            continue
        try:
            _route(bend_demo, color, bg, rank_bends=False)
        except ValueError as error:
            assert str(error) == "tied shortest minimum-bend checkpoint paths"
            saw_length_tie = True
    assert saw_length_tie, "training must expose a length tie resolved by bends"
    assert all(a["active"] >= 2 for a in analyses[len(puzzle["train"]):])
    training_max = max(max(_axis_runs(a["raw"], axis))
                       for a in analyses[:len(puzzle["train"])] for axis in (0, 1))
    assert training_max <= 12
    assert all(max(max(_axis_runs(a["raw"], axis)) for axis in (0, 1)) > training_max
               for a in analyses[len(puzzle["train"]):])
    # Concrete causal ablation: solving in expanded coordinates, then
    # compressing the completed output, must disagree with the first answer.
    wrong_order = _compress(_complete(analyses[0]["raw"]))
    assert not np.array_equal(wrong_order, puzzle["train"][0]["output"])


def self_test():
    """Bounded, disk-free coverage scan; returns compact serializable metrics."""
    shapes, canon_shapes, structures, palettes, bend_set, fragment_set = set(), set(), set(), set(), set(), set()
    puzzle_hashes, active_counts = set(), set()
    hidden = pairs = no_op = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for pair in puzzle["train"] + puzzle["test"]:
            a = _analyze(pair)
            shapes.add(a["raw"].shape)
            canon_shapes.add(a["canonical"].shape)
            roles = {}
            normalized = bytes(roles.setdefault(int(c), len(roles)) for c in a["canonical"].ravel())
            structures.add((a["canonical"].shape, normalized))
            palettes.add(a["background"])
            bend_set.update(a["bends"])
            fragment_set.update(a["fragments"])
            active_counts.add(a["active"])
            hidden += a["hidden_turns"]
            pairs += 1
            no_op += sum(x == y for x, y in zip(a["raw"].shape, a["canonical"].shape)) == 1
            digest.update(a["raw"].tobytes())
        puzzle_hashes.add(digest.hexdigest())
    assert len(puzzle_hashes) == 42 and len(shapes) >= 30 and len(canon_shapes) >= 15
    assert len(structures) >= 100 and len(palettes) >= 7
    assert {0, 1, 2, 3} <= bend_set and {2, 3} <= fragment_set
    assert {1, 2, 3} <= active_counts and hidden >= 84 and no_op >= 1
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(314159, train, test)
            validate_generated_puzzle(p)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
    first, again = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(x[k], y[k]) for split in first
               for x, y in zip(first[split], again[split]) for k in ("input", "output"))
    return {"seeds": 42, "pairs": pairs, "input_shapes": len(shapes),
            "canonical_shapes": len(canon_shapes), "normalized_structures": len(structures),
            "background_colors": len(palettes), "bend_counts": sorted(bend_set),
            "fragment_counts": sorted(fragment_set), "active_counts": sorted(active_counts),
            "hidden_turns": hidden, "one_no_op_axis": no_op,
            "independent_semantic_verification": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
