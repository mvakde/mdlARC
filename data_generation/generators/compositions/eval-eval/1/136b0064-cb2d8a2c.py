"""Threaded seals: spatial glyph programs executed by unary-clearance bypasses.

No parent code, examples, files, seeds, or learned state enter solve().
Directions below run clockwise in screen coordinates: down, left, up, right.
"""
from collections import deque

import numpy as np


_DIRS = ((1, 0), (0, -1), (-1, 0), (0, 1))
_DOWN = np.array([[1, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=bool)
_MASKS = tuple(np.rot90(_DOWN, -h) for h in range(4))


def _add(p, v, scale=1):
    return p[0] + scale * v[0], p[1] + scale * v[1]


def _dot(p, q):
    return p[0] * q[0] + p[1] * q[1]


def _sub(p, q):
    return p[0] - q[0], p[1] - q[1]


def _line(p, q):
    delta = _sub(q, p)
    assert delta[0] == 0 or delta[1] == 0
    size = abs(delta[0]) + abs(delta[1])
    v = (int(delta[0] > 0) - int(delta[0] < 0),
         int(delta[1] > 0) - int(delta[1] < 0))
    return [_add(p, v, k) for k in range(size + 1)]


def _inside(p, shape, margin=0):
    return all(margin <= x < n - margin for x, n in zip(p, shape))


def _halo(cells):
    return {(r + dy, c + dx) for r, c in cells
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)}


def _bar(q, axis, length, glyph_index, heading, n, marker_low, color):
    center = _add(q, axis, glyph_index)
    line = {_add(q, axis, k) for k in range(length)}
    glyph = {(center[0] + r - 1, center[1] + c - 1)
             for r, c in zip(*np.nonzero(_MASKS[heading]))}
    markers = {_add(q, axis, 2 * k if marker_low else length - 1 - 2 * k)
               for k in range(n)}
    return dict(q=q, axis=axis, length=length, g=glyph_index, center=center,
                heading=heading, n=n, low=marker_low, color=color,
                line=line, glyph=glyph, markers=markers, cells=line | glyph)


def _ray_hit(p, heading, bar):
    f, side = _DIRS[heading], _DIRS[(heading + 1) % 4]
    distances = [_dot(_sub(q, p), f) for q in bar["cells"]
                 if _dot(_sub(q, p), side) == 0 and _dot(_sub(q, p), f) > 0]
    return min(distances) if distances else None


def _maneuver(p, heading, bar, clearance=None, side_override=None):
    f, right = _DIRS[heading], _DIRS[(heading + 1) % 4]
    assert _dot(f, bar["axis"]) == 0, "approached a longitudinal bar"
    assert bar["heading"] != (heading + 2) % 4, "a seal cannot send the pen backward"
    d = bar["n"] + 1 if clearance is None else clearance
    a = _dot(_sub(bar["q"], p), f)
    endpoints = (bar["q"], _add(bar["q"], bar["axis"], bar["length"] - 1))
    offsets = [_dot(_sub(q, p), right) for q in endpoints]
    lo, hi = min(offsets), max(offsets)
    assert lo <= 0 <= hi and a >= d + 2, "insufficient approach room"
    marked = endpoints[0 if bar["low"] else 1]
    side = 1 if _dot(_sub(marked, p), right) == lo else -1
    if side_override is not None:
        side = side_override
    turn = _add(p, f, a - 1 - d)
    bend = _add(turn, right, hi + d if side == 1 else lo - d)
    landing = _add(bend, f, 2 * d + 2)
    approach = _line(p, turn)
    execution = _line(turn, bend) + _line(bend, landing)[1:]
    return approach, execution, landing, side


def _boundary_ray(p, heading, shape):
    v = _DIRS[heading]
    cells = [p]
    while _inside(_add(cells[-1], v), shape):
        cells.append(_add(cells[-1], v))
    return cells


def _check_grid(grid):
    a = np.asarray(grid)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.bool_)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        component, todo = {first}, deque([first])
        while todo:
            r, c = todo.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (r + dy, c + dx)
                    if q in remaining:
                        remaining.remove(q)
                        component.add(q)
                        todo.append(q)
        yield component


def _read(input_grid):
    """Recognize geometry and color roles using only the queried input."""
    a = _check_grid(input_grid)
    h, w = a.shape
    bg = int(a[0, 0])
    assert all(int(a[r, c]) == bg for r, c in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1)))
    assert np.count_nonzero(a == bg) > a.size // 2
    boundary = {(r, c) for r in range(h) for c in range(w)
                if (r in (0, h-1) or c in (0, w-1)) and a[r, c] != bg}
    assert len(boundary) == 1, "expected one non-corner boundary seed"
    seed = boundary.pop()
    heading = next(k for k, match in enumerate(
        (seed[0] == 0, seed[1] == w-1, seed[0] == h-1, seed[1] == 0)) if match)
    arrows = []
    glyph_cells = set()
    for raw_color in np.unique(a):
        color = int(raw_color)
        if color == bg:
            continue
        for component in _components(a == color):
            if seed in component:
                assert component == {seed}
                continue
            if len(component) != 7:
                continue
            r0, c0 = map(min, zip(*component))
            r1, c1 = map(max, zip(*component))
            if (r1-r0, c1-c0) != (2, 2):
                continue
            mask = np.zeros((3, 3), dtype=bool)
            for r, c in component:
                mask[r-r0, c-c0] = True
            matches = [k for k, m in enumerate(_MASKS) if np.array_equal(mask, m)]
            if matches:
                arrows.append(((r0+1, c0+1), matches[0], color, component))
                glyph_cells.update(component)
    assert 1 <= len(arrows) <= 8, "expected one glyph per bar"
    residual = set(map(tuple, np.argwhere(a != bg))) - glyph_cells - {seed}
    assert len({int(a[p]) for p in residual}) == 2
    bars, covered, roles = [], set(), set()
    for center, outgoing, color, component in sorted(arrows):
        axes = [v for v in ((0, 1), (1, 0))
                if _add(center, v, -2) in residual and _add(center, v, 2) in residual]
        assert len(axes) == 1, "glyph must sit inside one straight bar"
        axis = axes[0]
        low, high = -2, 2
        while _add(center, axis, low-1) in residual:
            low -= 1
        while _add(center, axis, high+1) in residual:
            high += 1
        q, length, g = _add(center, axis, low), high-low+1, -low
        external = [_add(q, axis, k) for k in range(length) if abs(k-g) > 1]
        values, counts = np.unique([a[p] for p in external], return_counts=True)
        assert len(values) == 2 and counts[0] != counts[1]
        body, marker = int(values[counts.argmax()]), int(values[counts.argmin()])
        markers = {k for k in range(length) if abs(k-g) > 1 and a[_add(q, axis, k)] == marker}
        n = len(markers)
        assert 1 <= n <= 3 and 2*n + 4 <= length
        low_match = markers == set(range(0, 2*n, 2))
        high_match = markers == {length-1-2*k for k in range(n)}
        assert low_match != high_match, "ambiguous marker endpoint"
        assert (g >= 2*n+1 if low_match else g <= length-2*n-2)
        bar = _bar(q, axis, length, g, outgoing, n, low_match, color)
        assert bar["glyph"] == component
        assert all(_inside(p, a.shape, 1) for p in bar["cells"])
        assert not covered & _halo(bar["cells"]), "instruction objects must have a blank gap"
        assert all(int(a[p]) == (color if p in component else marker if p in bar["markers"] else body)
                   for p in bar["cells"])
        bars.append(bar)
        covered.update(bar["cells"])
        roles.add((body, marker))
    assert len(roles) == 1
    assert covered | {seed} == set(map(tuple, np.argwhere(a != bg)))
    body, marker = roles.pop()
    assert not {bg, body, marker} & {int(a[seed]), *(b["color"] for b in bars)}
    return dict(grid=a, bg=bg, body=body, marker=marker, seed=seed,
                seed_color=int(a[seed]), heading=heading, bars=bars)


def _trace(info, ignore_arrows=False, constant_clearance=None, forced_side=None):
    """Execute reachable seals; keep an ordered route and its changing ink."""
    a, bars = info["grid"], info["bars"]
    p, heading, carry = info["seed"], info["heading"], info["seed_color"]
    ordered, path, paint = [p], {p}, {p: carry}
    visited, events = [], []
    obstacles = set().union(*(b["cells"] for b in bars))

    def extend(cells, color):
        assert cells[0] == ordered[-1]
        for q in cells[1:]:
            assert _inside(q, a.shape) and q not in path and q not in obstacles, "route collision"
            path.add(q)
            ordered.append(q)
            paint[q] = color

    for _ in range(len(bars)+1):
        hits = [(distance, i) for i, b in enumerate(bars)
                if (distance := _ray_hit(p, heading, b)) is not None]
        if not hits:
            extend(_boundary_ray(p, heading, a.shape), carry)
            break
        distance, i = min(hits)
        assert i not in visited, "a route cannot execute a seal twice"
        bar = bars[i]
        approach, execution, landing, side = _maneuver(
            p, heading, bar, constant_clearance, forced_side)
        extend(approach, carry)
        paint[execution[0]] = bar["color"]
        extend(execution, bar["color"])
        events.append(dict(index=i, incoming=heading, outgoing=bar["heading"],
                           n=bar["n"], side=side, turn=execution[0], landing=landing,
                           approach=len(approach)-1, hit=distance))
        visited.append(i)
        p, heading, carry = landing, heading if ignore_arrows else bar["heading"], bar["color"]
    else:
        raise AssertionError("execution did not terminate")
    result = a.copy()
    for i in visited:
        b = bars[i]
        for q in b["glyph"] - b["line"]:
            result[q] = info["bg"]
        for q in b["line"]:
            result[q] = info["body"]
    for q, color in paint.items():
        result[q] = color
    assert result[info["seed"]] == info["seed_color"]
    return result, dict(events=events, visited=visited, path=path, ordered=ordered)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver, including reachable/unreachable instruction selection."""
    return _trace(_read(input_grid))[0]


def _search_chain(rng, shape, ns, rels, sides, palette, budget=45):
    """Bounded geometric search. It constructs objects, not a solver lookup table."""
    height, width = shape
    start = (0, int(rng.integers(6, width-6)))
    nodes = 0

    def search(depth, p, heading, bars, route, queries):
        nonlocal nodes
        if nodes >= budget:
            return None
        nodes += 1
        occupied = set().union(*(b["cells"] for b in bars)) if bars else set()
        occupied_halo = _halo(occupied)
        if depth == len(ns):
            if any(_ray_hit(p, heading, b) is not None for b in bars):
                return None
            exit_cells = _boundary_ray(p, heading, shape)
            if set(exit_cells[1:]) & (occupied | route):
                return None
            return bars, route | set(exit_cells), queries, start
        n, rel = ns[depth], rels[depth]
        d = n+1
        f, right = _DIRS[heading], _DIRS[(heading+1) % 4]
        ink = int(palette[depth % len(palette)])
        for _ in range(45):
            side = int(rng.choice((-1, 1))) if sides[depth] is None else sides[depth]
            length = int(rng.integers(2*n+4, 2*n+9))
            if rng.random() < .75:
                inset = int(rng.choice(3, p=(.6, .3, .1)))
                cross = length-1-inset if side == 1 else inset
            else:
                cross = int(rng.integers(length))
            a = d+2+int(rng.choice(6, p=(.32, .25, .19, .12, .08, .04)))
            intersection = _add(p, f, a)
            q = _add(intersection, right, -cross)
            low = side == 1
            g = (int(rng.integers(2*n+1, length-2)) if low else
                 int(rng.integers(2, length-2*n-1)))
            b = _bar(q, right, length, g, (heading+rel) % 4, n, low, ink)
            if any(not _inside(q, shape, 1) for q in b["cells"]):
                continue
            if b["cells"] & (route | occupied_halo):
                continue
            # New objects may not steal an earlier instruction's incoming ray.
            if any((hit := _ray_hit(s, hd, b)) is not None and hit <= old_hit
                   for s, hd, old_hit in queries):
                continue
            hit = _ray_hit(p, heading, b)
            if hit is None or any((old := _ray_hit(p, heading, ob)) is not None and old <= hit
                                  for ob in bars):
                continue
            approach, execution, landing, _ = _maneuver(p, heading, b)
            fresh = approach[1:] + execution[1:]
            if (len(fresh) != len(set(fresh)) or any(not _inside(q, shape) for q in fresh)
                    or set(fresh) & (route | occupied | b["cells"])):
                continue
            answer = search(depth+1, landing, b["heading"], bars+[b], route | set(fresh),
                            queries+[(p, heading, hit)])
            if answer is not None:
                return answer
        return None

    return search(0, start, 0, [], {start}, [])


def _add_decoys(rng, shape, bars, route, queries, number, max_n, palette):
    result = list(bars)
    for _ in range(number):
        occupied = set().union(*(b["cells"] for b in result))
        occupied_halo = _halo(occupied)
        for _ in range(1500):
            n = int(rng.integers(1, max_n+1))
            length = int(rng.integers(2*n+4, 2*n+9))
            axis = ((0, 1), (1, 0))[int(rng.integers(2))]
            if shape[0 if axis[0] else 1] < length+2:
                continue
            q = (int(rng.integers(2 if axis[1] else 1, shape[0]-1-(length-1)*axis[0])),
                 int(rng.integers(2 if axis[0] else 1, shape[1]-1-(length-1)*axis[1])))
            low = bool(rng.integers(2))
            g = (int(rng.integers(2*n+1, length-2)) if low else
                 int(rng.integers(2, length-2*n-1)))
            b = _bar(q, axis, length, g, int(rng.integers(4)), n, low, int(rng.choice(palette)))
            if any(not _inside(p, shape, 1) for p in b["cells"]):
                continue
            if b["cells"] & (route | occupied_halo):
                continue
            if any((hit := _ray_hit(s, hd, b)) is not None and hit <= old_hit
                   for s, hd, old_hit in queries):
                continue
            result.append(b)
            break
        else:
            return None
    return result


def _render(shape, bars, start, roles):
    bg, body, marker, seed_color = roles
    a = np.full(shape, bg, dtype=np.int8)
    a[start] = seed_color
    for b in bars:
        for p in b["line"]:
            a[p] = body
        for p in b["markers"]:
            a[p] = marker
        for p in b["glyph"]:
            a[p] = b["color"]
    return a


def _make_pair(rng, ns, rels, sides, decoys, max_n, uniform_ink=False):
    colors = list(map(int, rng.permutation(10)))
    roles = tuple(colors[:4])
    palette = list(map(int, rng.choice(colors[4:], int(rng.integers(2, 5)), replace=False)))
    if uniform_ink:
        palette = palette[:1]
    for _ in range(3):
        shape = tuple(map(int, rng.integers(23, 31, size=2)))
        answer = _search_chain(rng, shape, ns, rels, sides, palette)
        if answer is None:
            continue
        bars, route, queries, start = answer
        all_bars = _add_decoys(rng, shape, bars, route, queries, decoys, max_n, palette)
        if all_bars is None:
            continue
        a = _render(shape, all_bars, start, roles)
        info = _read(a)
        b, trace = _trace(info)
        assert len(trace["events"]) == len(ns) and trace["path"] == route
        # Constructor labels: consume exactly the planned objects, paint each
        # maneuver using the corresponding seal, and extend the final ray.
        expected = a.copy()
        for bar in bars:
            for p in bar["glyph"]-bar["line"]:
                expected[p] = roles[0]
            for p in bar["line"]:
                expected[p] = roles[1]
        p, hd, carry = start, 0, roles[3]
        for bar in bars:
            approach, execution, p, _ = _maneuver(p, hd, bar)
            for q in approach:
                expected[q] = carry
            for q in execution:
                expected[q] = bar["color"]
            hd, carry = bar["heading"], bar["color"]
        for q in _boundary_ray(p, hd, shape):
            expected[q] = carry
        assert np.array_equal(b, expected)
        return {"input": a, "output": expected}
    raise ValueError("bounded search exhausted for this instruction sequence")


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def _sample_ns(rng, active, maximum):
    ns = list(map(int, rng.integers(1, maximum+1, size=active)))
    # This is a packing preference, not a solving restriction: large unary
    # values are covered in short demonstrations and can occur in tests.
    while sum(ns) > active+2:
        choices = [i for i, n in enumerate(ns) if n > 1]
        ns[int(rng.choice(choices))] -= 1
    return ns


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    nt = _integer("num_train", num_train, 2, 10)
    nq = _integer("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    max_n = int(rng.integers(2, 4))
    train_limit = 3 if max_n == 2 and rng.random() < .4 else 2
    ns = list(range(1, max_n+1)) + list(map(int, rng.integers(1, max_n+1, size=4-max_n)))
    ns = [ns[int(i)] for i in rng.permutation(4)]
    # Four visible calls across two demonstrations expose the whole vocabulary,
    # counting range, both marker ends, and a repeated-side transition.
    for _ in range(100):
        rels = [-1, 0, 1, int(rng.integers(-1, 2))]
        rels = [rels[int(i)] for i in rng.permutation(4)]
        if rels[1] != 0 and rels[3] != 0:
            break
    else:
        rels = [0, 1, 0, -1]
    puzzle, seen, first_headings = {"train": [], "test": []}, set(), None
    first_sides = []
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            active = (2 if index < 2 else int(rng.integers(2, train_limit+1))) if split == "train" else train_limit+1
            if split == "train" and index < 2:
                sl = slice(0, 2) if index == 0 else slice(2, 4)
                local_ns, local_rels = ns[sl], rels[sl]
            else:
                local_ns = _sample_ns(rng, active, max_n)
                local_rels = list(map(int, rng.integers(-1, 2, size=active)))
                if sum(x != 0 for x in local_rels) < 2:
                    for j in rng.choice(active, 2, replace=False):
                        local_rels[int(j)] = int(rng.choice((-1, 1)))
            decoys = int(rng.integers(1, 3))
            for _ in range(40):
                try:
                    pair = _make_pair(rng, local_ns, local_rels, [None]*active, decoys, max_n,
                                      uniform_ink=split == "train" and index == 0)
                except ValueError:
                    # Counts and geometric opcodes are unchanged; another
                    # ordering of the same obligations can fit the canvas.
                    if split == "train" and index < 2:
                        order = list(map(int, rng.permutation(active)))
                        local_ns = [local_ns[j] for j in order]
                        if active == 3:
                            local_rels = [local_rels[j] for j in order]
                    else:
                        local_ns = _sample_ns(rng, active, max_n)
                        local_rels = list(map(int, rng.choice((-1, 0, 1), active, p=(.4, .2, .4))))
                    continue
                info = _read(pair["input"])
                trace = _trace(info)[1]
                heads = {e["outgoing"] for e in trace["events"]}
                current_sides = [e["side"] for e in trace["events"]]
                rotations = list(range(4))
                if split == "train" and index == 1:
                    rotations = [k for k in rotations if first_headings | {(h-k) % 4 for h in heads} == set(range(4))]
                    if (set(first_sides+current_sides) != {-1, 1}
                            or not any(a == b for ss in (first_sides, current_sides) for a, b in zip(ss, ss[1:]))):
                        continue
                if not rotations:
                    # Resample an order, retaining every required relative turn.
                    rng.shuffle(local_rels)
                    continue
                k = int(rng.choice(rotations))
                pair = {role: np.rot90(a, k).copy() for role, a in pair.items()}
                key = _key(pair["input"])
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train" and index == 0:
                    first_headings = {(h-k) % 4 for h in heads}
                    first_sides = current_sides
                break
            else:
                raise AssertionError("could not make distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _different_ablation(info, expected, **kwargs):
    try:
        return not np.array_equal(_trace(info, **kwargs)[0], expected)
    except AssertionError:
        # The simplified rule can also crash into an object or retrace itself.
        return True


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _check_grid(pair["input"]), _check_grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _read(a)
            predicted, trace = _trace(info)
            assert np.array_equal(predicted, b) and not np.array_equal(a, b)
            assert len(trace["events"]) >= 1
            assert len(trace["visited"]) < len(info["bars"]), "missing inactive instruction"
            assert b[trace["ordered"][-1]] != info["bg"]
            assert _inside(trace["ordered"][-1], b.shape) and not _inside(trace["ordered"][-1], b.shape, 1)
            for i, bar in enumerate(info["bars"]):
                if i in trace["visited"]:
                    assert all(b[p] == info["body"] for p in bar["line"])
                    assert all(b[p] == info["bg"] for p in bar["glyph"]-bar["line"])
                else:
                    assert all(a[p] == b[p] for p in bar["cells"])
            assert {tuple(p) for p in np.argwhere((a == info["bg"]) & (b != info["bg"]))} == trace["path"]-{info["seed"]}
            analyses[split].append((info, trace, b))
    train = analyses["train"]
    events = [e for _, tr, _ in train for e in tr["events"]]
    train_counts = {e["n"] for e in events}
    assert {1, 2} <= train_counts
    assert {e["outgoing"] for e in events} == set(range(4))
    color_meanings = {}
    for info, tr, _ in train:
        for e in tr["events"]:
            color_meanings.setdefault(info["bars"][e["index"]]["color"], set()).add(e["outgoing"])
    assert any(len(heads) > 1 for heads in color_meanings.values()), "color-only command shortcut"
    assert {(e["outgoing"]-e["incoming"]) % 4 for e in events} == {0, 1, 3}
    assert {e["side"] for e in events} == {-1, 1}
    assert any(any(a["side"] == b["side"] for a, b in zip(tr["events"], tr["events"][1:])) for _, tr, _ in train)
    assert any(_different_ablation(info, b, ignore_arrows=True) for info, _, b in train)
    assert any(_different_ablation(info, b, constant_clearance=2) for info, _, b in train)
    assert any(_different_ablation(info, b, forced_side=1) for info, _, b in train)
    assert all(e["n"] in train_counts for _, tr, _ in analyses["test"] for e in tr["events"])
    assert min(len(tr["events"]) for _, tr, _ in analyses["test"]) > max(len(tr["events"]) for _, tr, _ in train)
    return True


def self_test():
    """Bounded 42-seed variation, semantic, metamorphic and damage checks."""
    # Literal fixture: marker count two means clearance three from the
    # seal's full rows 4..6, hence turn row 1 and landing row 9.
    a = np.zeros((12, 18), dtype=np.int8)
    a[0, 9] = 2
    a[5, 5:14] = 5
    a[5, [5, 7]] = 1
    a[4:7, 9:12] = np.array([[3, 3, 0], [3, 3, 3], [3, 3, 0]])
    expected = np.zeros_like(a)
    expected[0, 9] = 2
    expected[5, 5:14] = 5
    expected[1, 9:17] = 3
    expected[1:10, 16] = 3
    expected[9, 16:18] = 3
    assert np.array_equal(solve(a), expected)
    literal_info = _read(a)
    assert not np.array_equal(_trace(literal_info, ignore_arrows=True)[0], expected)
    assert not np.array_equal(_trace(literal_info, constant_clearance=2)[0], expected)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[7]) == _puzzle_key(generate_puzzle(7))
    shapes, palettes, programs, lengths, marker_counts, transitions = set(), set(), set(), set(), set(), set()
    mixed, repeated, inactive, ablation_a, ablation_b = 0, 0, 0, 0, 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _read(pair["input"])
                expected, trace = _trace(info)
                events = trace["events"]
                shapes.add(pair["input"].shape)
                palettes.add((info["bg"], info["body"], info["marker"], info["seed_color"]))
                lengths.update(b["length"] for b in info["bars"])
                marker_counts.update(e["n"] for e in events)
                programs.add(tuple((e["incoming"], e["outgoing"], e["n"], e["side"]) for e in events))
                transitions.update((a["side"], b["side"]) for a, b in zip(events, events[1:]))
                mixed += len({b["axis"] for b in info["bars"]}) > 1
                repeated += any(a["side"] == b["side"] for a, b in zip(events, events[1:]))
                inactive += len(trace["visited"]) < len(info["bars"])
                ablation_a += _different_ablation(info, expected, ignore_arrows=True)
                ablation_b += _different_ablation(info, expected, constant_clearance=2)
    assert len(shapes) >= 35 and len(palettes) >= 100 and len(programs) >= 150
    assert marker_counts == {1, 2, 3} and len(lengths) >= 8
    assert transitions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    assert min(mixed, repeated, ablation_a, ablation_b) >= 60 and inactive == 210
    # Boundary counts; verify.py additionally exercises all 30 legal combinations.
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(9182+nt*7+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Rotation, reflection and arbitrary bijective recoloring preserve semantics.
    pair = puzzles[0]["test"][0]
    permutation = np.array([8, 2, 9, 0, 5, 7, 3, 1, 6, 4])
    for k in range(4):
        for reflection in (False, True):
            a, b = (np.rot90(pair[v], k) for v in ("input", "output"))
            if reflection:
                a, b = a[:, ::-1], b[:, ::-1]
            assert np.array_equal(solve(permutation[a]), permutation[b])
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    info = _read(damaged["test"][0]["input"])
    _, trace = _trace(info)
    damaged["test"][0]["output"][trace["ordered"][1]] = info["bg"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing route cell accepted")
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    info = _read(damaged["test"][0]["input"])
    _, trace = _trace(info)
    inactive_bar = next(b for i, b in enumerate(info["bars"]) if i not in trace["visited"])
    damaged["test"][0]["output"][next(iter(inactive_bar["markers"]))] = info["body"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("executing an inactive marker was accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "palettes": len(palettes),
            "programs": len(programs), "bar_lengths": sorted(lengths),
            "mixed_axis_pairs": mixed, "repeated_side_pairs": repeated,
            "arrow_ablation_witnesses": ablation_a, "count_ablation_witnesses": ablation_b}
