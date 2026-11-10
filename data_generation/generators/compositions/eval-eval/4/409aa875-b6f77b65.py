"""Demolition gates: endpoint convergence repeatedly erases settled sticks.

The solver reads only the supplied grid. Arrowheads are anchored control marks,
not physical bodies. Rod identities are retained through every gravity round.
"""
from collections import Counter
import itertools
import numpy as np


DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1))
DISTANCE = 5


def _arrow(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return tuple(sorted((r + a, c + b) for a, b in offsets))


def _endpoint(arrow):
    tip, direction, _, _ = arrow
    return tuple(tip[k] + DISTANCE * direction[k] for k in (0, 1))


# Rod: (H/V, top row, left column, length, color).
def _cells(rod):
    orientation, r, c, length, _ = rod
    return frozenset((r + (k if orientation == "V" else 0),
                      c + (k if orientation == "H" else 0))
                     for k in range(length))


def _shift(rod, amount):
    return rod[0], rod[1] + amount, rod[2], rod[3], rod[4]


def _settle(rods, height):
    """Each tick moves the greatest set of rods able to descend together."""
    result = list(rods)
    for _ in range(height + 1):
        footprints = [_cells(p) for p in result]
        targets = [{(r + 1, c) for r, c in cells} for cells in footprints]
        moving = {i for i, target in enumerate(targets)
                  if all(r < height for r, _ in target)}
        while True:
            blocked = {i for i in moving
                       if any(targets[i] & footprints[j]
                              for j in range(len(result)) if j not in moving)}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return result
        result = [_shift(p, 1) if i in moving else p for i, p in enumerate(result)]
    raise AssertionError("Gravity did not converge")


def _evolve(rods, gates, height):
    """Return final (original id, rod) pairs and a trace of removal rounds."""
    active = list(enumerate(rods))
    trace = []
    for _ in range(len(rods) + 1):
        removed = [(i, p) for i, p in active if _cells(p) & gates]
        if not removed:
            return active, trace
        survivors = [(i, p) for i, p in active if not _cells(p) & gates]
        settled = _settle([p for _, p in survivors], height)
        after = [(i, q) for (i, _), q in zip(survivors, settled)]
        trace.append({"removed": removed, "before": survivors, "after": after})
        active = after
    raise AssertionError("A removal round must strictly decrease the rod count")


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return grid.astype(np.int8, copy=True)


def _parse(value):
    grid = _grid(value)
    frequencies = np.bincount(grid.ravel(), minlength=10)
    background = int(frequencies.argmax())
    if frequencies[background] * 2 <= grid.size:
        raise ValueError("Background must occupy more than half the input")
    remaining = set(map(tuple, np.argwhere(grid != background).tolist()))
    rods, arrows = [], []
    while remaining:
        start = min(remaining)
        color = int(grid[start])
        component, todo = {start}, [start]
        remaining.remove(start)
        while todo:
            r, c = todo.pop()
            neighbors = {(r + a, c + b) for a in (-1, 0, 1)
                         for b in (-1, 0, 1)} & remaining
            neighbors = {p for p in neighbors if grid[p] == color}
            remaining -= neighbors
            component |= neighbors
            todo.extend(sorted(neighbors))
        matches = [(tip, direction, _arrow(tip, direction), color)
                   for tip in sorted(component) for direction in DIRECTIONS
                   if len(component) == 3
                   and set(_arrow(tip, direction)) == component]
        if matches:
            if len(matches) != 1:
                raise ValueError("Ambiguous arrowhead")
            arrows.append(matches[0])
        else:
            rows, cols = {r for r, _ in component}, {c for _, c in component}
            if len(component) < 2 or (len(rows) != 1 and len(cols) != 1):
                raise ValueError("Every other component must be a straight stick")
            orientation = "H" if len(rows) == 1 else "V"
            rod = (orientation, min(rows), min(cols), len(component), color)
            if _cells(rod) != component:
                raise ValueError("Gapped stick")
            rods.append(rod)
    if not arrows or not rods:
        raise ValueError("Need arrow controls and a scaffold")
    # Controls have a blank king-neighbor gap from every other object.
    rod_cells = set().union(*map(_cells, rods))
    for i, arrow in enumerate(arrows):
        halo = {(r + a, c + b) for r, c in arrow[2]
                for a in (-1, 0, 1) for b in (-1, 0, 1)}
        if halo & rod_cells or any(halo & set(other[2]) for other in arrows[:i]):
            raise ValueError("Controls must be visibly isolated")
    counts = Counter(_endpoint(a) for a in arrows)
    if not all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] for r, c in counts):
        raise ValueError("Every five-step endpoint must be in bounds")
    if set(counts) & set().union(*(set(a[2]) for a in arrows)):
        raise ValueError("Endpoints may not occupy a control mark")
    return grid, background, sorted(rods), arrows, counts


def _render(shape, background, rods, arrows):
    grid = np.full(shape, background, dtype=np.int8)
    occupied = set()
    for rod in rods:
        cells = _cells(rod)
        if occupied & cells:
            raise AssertionError("Overlapping physical sticks")
        occupied |= cells
        for p in cells:
            grid[p] = rod[4]
    for arrow in arrows:
        if occupied & set(arrow[2]):
            raise ValueError("Final sticks must not cover control marks")
        occupied.update(arrow[2])
        for p in arrow[2]:
            grid[p] = arrow[3]
    return grid


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid, background, rods, arrows, counts = _parse(input_grid)
    if _settle(rods, grid.shape[0]) != rods:
        raise ValueError("The input scaffold must initially be stable")
    gates = {p for p, n in counts.items() if n >= 2}
    active, _ = _evolve(rods, gates, grid.shape[0])
    return _render(grid.shape, background, [p for _, p in active], arrows)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < low or (high is not None and value > high)):
        raise ValueError(f"Illegal {name}")
    return int(value)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _drop(rods, orientation, col, length, height):
    rod = (orientation, 0, col, length, 0)
    if any(_cells(rod) & _cells(p) for p in rods):
        return False
    for d in range(height):
        target = _cells(_shift(rod, d + 1))
        if any(r >= height for r, _ in target) or any(target & _cells(p) for p in rods):
            result = _shift(rod, d)
            if result[1] < 2:
                return False
            rods.append(result)
            return True
    return False


def _extras(rng, rods, shape, maximum):
    h, w = shape
    desired = int(rng.integers(maximum + 1))
    for _ in range(desired * 5):
        if desired <= 0:
            break
        orientation = _pick(rng, ("H", "V"))
        length = int(rng.integers(2, min(w if orientation == "H" else h - 5, 12) + 1))
        col = int(rng.integers(w - length + 1 if orientation == "H" else w))
        if _drop(rods, orientation, col, length, h):
            desired -= 1


def _geometry(rng, mode, rich):
    """Targeted witnesses plus randomly inserted pieces, without slot partitions."""
    if mode in ("compact", "quiet"):
        h, w = int(rng.integers(12, 21)), int(rng.integers(15, 25))
        g = int(rng.integers(6, h - 3))
        floor_gate = mode == "compact" and rng.random() < .25
        c = int(rng.integers(6, w - 6 if floor_gate else w - 3))
        left = int(rng.integers(max(0, c - 5), c))
        right = int(rng.integers(c + 1, min(w, c + 6)))
        rods = [("V", g + 1, c, h - g - 1, 0),
                ("H", g, left, right - left + 1, 0)]
        if mode == "quiet":
            gates, singles = set(), [(g, int(rng.integers(left, right + 1)))]
        else:
            gates, singles = {(h - 1 if floor_gate else g + 2, c)}, []
        return (h, w), rods, gates, singles, None
    h = int(rng.integers(19 if rich else 17, 31))
    w = int(rng.integers(22 if mode == "escape" or rich else 17, 31))
    if mode == "cascade":
        g = int(rng.integers(10, h - 4))
        left = int(rng.integers(1, w - 12))
        right = int(rng.integers(left + 9, min(w - 2, left + 18) + 1))
        c = int(rng.integers(left + 1, left + 4))
        posts = [c, int(rng.integers(c + 3, right + 1))]
        rods = [("V", g + 1, x, h - g - 1, 0) for x in posts]
        if rng.random() < .45:
            shorter = int(rng.integers(2, h - g - 1))
            x = int(rng.integers(c + 2, right + 1))
            if x not in posts:
                rods.append(("V", h - shorter, x, shorter, 0))
        feeds = int(rng.integers(2, 5 if rich else 4))
        for k in range(feeds):
            start = int(rng.integers(left, c + 1))
            end = int(rng.integers(max(c + 3, posts[1]), right + 1))
            rods.append(("H", g - k, start, end - start + 1, 0))
        # This cap misses the gate but shares a real post with the feeder stack.
        cap_left = int(rng.integers(c + 2, posts[1] + 1))
        cap_right = int(rng.integers(max(cap_left + 1, posts[1]), right + 2))
        rods.append(("H", g - feeds, cap_left, cap_right - cap_left + 1, 0))
        # A passenger on the cap shares the bottom feeder's color, yet survives.
        # This makes geometry-based selection distinguishable from color commands.
        feeder_index = next(i for i, p in enumerate(rods) if p[0] == "H")
        passenger_index = len(rods)
        rods.append(("V", g - feeds - 2, posts[1], 2, 0))
        reuse_pair = (passenger_index, feeder_index)
        gates = {(g, c)}
        singles = []
    elif mode == "escape":
        reuse_pair = None
        top = int(rng.integers(8, h - 6))
        left = int(rng.integers(1, w - 17))
        right = int(rng.integers(left + 14, w - 1))
        a = int(rng.integers(left + 1, left + 4))
        b = int(rng.integers(max(a + 8, right - 4), right))
        rods = [("V", top + 1, a, h - top - 1, 0),
                ("V", top + 1, b, h - top - 1, 0),
                ("H", top, left, right - left + 1, 0)]
        gate_row = top + int(rng.integers(2, min(5, h - top - 2)))
        gates = {(gate_row, a), (gate_row, b)}
        # Two distinct, harmless single endpoints on the SAME surviving beam.
        single_cols = rng.choice(np.arange(a + 2, b - 1), size=2, replace=False)
        singles = [(top, int(c)) for c in single_cols]
        if rich and rng.random() < .65:
            length = int(rng.integers(2, 5))
            x = int(rng.integers(a + 1, b))
            rods.append(("V", top - length, x, length, 0))
    else:
        reuse_pair = None
        rods = []
        target = int(rng.integers(4, 15 if rich else 10))
        for _ in range(target * 6):
            if len(rods) >= target:
                break
            orientation = "H" if rng.random() < .55 else "V"
            length = int(rng.integers(2, min(w - 3 if orientation == "H" else h - 5, 15) + 1))
            col = int(rng.integers(w - length + 1 if orientation == "H" else w))
            _drop(rods, orientation, col, length, h)
        if not rods:
            return None
        # Geometry, rather than a predetermined identity, selects demolition sites.
        eligible = [p for p in rods if p[1] >= 5]
        if not eligible:
            return None
        chosen = rng.choice(len(eligible), size=min(len(eligible), int(rng.integers(1, 4))), replace=False)
        gates = {_pick(rng, sorted(_cells(eligible[int(i)]))) for i in chosen}
        singles = []
        if rng.random() < .6:
            available = sorted(set().union(*map(_cells, rods)) - gates)
            if available:
                singles = [_pick(rng, available)]
    _extras(rng, rods, (h, w), 5 if rich else 2)
    if len(set().union(*map(_cells, rods))) * 2 >= h * w:
        return None
    if _settle(rods, h) != rods:
        return None
    return (h, w), rods, gates, singles, reuse_pair


def _near(first, second):
    return any(max(abs(r - a), abs(c - b)) <= 1
               for r, c in first for a, b in second)


def _color_rods(rng, rods, background, reuse_pair=None):
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c != background])]
    size = int(rng.integers(3, 10))
    footprints = [_cells(p) for p in rods]
    result = []
    for i, rod in enumerate(rods):
        forbidden = {q[4] for j, q in enumerate(result) if _near(footprints[i], footprints[j])}
        if reuse_pair is not None and i == reuse_pair[0]:
            value = result[reuse_pair[1]][4]
            if value in forbidden:
                return None
            result.append((*rod[:4], value))
            continue
        choices = [c for c in palette[:size] if c not in forbidden]
        if not choices:
            choices = [c for c in palette if c not in forbidden]
        if not choices:
            return None
        result.append((*rod[:4], int(_pick(rng, choices))))
    return result


def _place_arrows(rng, shape, rods, final_rods, gates, singles, force_double=False):
    initial = set().union(*map(_cells, rods))
    final = set().union(*map(_cells, final_rods)) if final_rods else set()
    endpoints = gates | set(singles)
    groups = []
    for point in sorted(endpoints):
        candidates = []
        for direction in DIRECTIONS:
            tip = tuple(point[k] - DISTANCE * direction[k] for k in (0, 1))
            cells = _arrow(tip, direction)
            if (all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)
                    and not _near(cells, initial) and not set(cells) & (final | endpoints)):
                candidates.append((tip, direction, cells, 0))
        need = 2 if point in gates else 1
        if (point in gates and len(candidates) >= 3 and rng.random() < .2
                and not (force_double and point == min(gates))):
            need = 3
        if len(candidates) < need:
            return None
        options = []
        # At most eight candidates: complete finite enumeration, not unbounded search.
        for subset in itertools.combinations(candidates, need):
            if all(not _near(a[2], b[2]) for i, a in enumerate(subset) for b in subset[:i]):
                options.append(subset)
        if not options:
            return None
        groups.append((point, [options[int(i)] for i in rng.permutation(len(options))]))
    # Harder sites first. Backtracking is capped and retried with new geometry.
    groups.sort(key=lambda item: len(item[1]))
    budget = [2000]

    def search(index, selected):
        budget[0] -= 1
        if budget[0] < 0:
            return None
        if index == len(groups):
            return selected
        for option in groups[index][1]:
            if all(not _near(a[2], b[2]) for a in option for b in selected):
                answer = search(index + 1, selected + list(option))
                if answer is not None:
                    return answer
        return None

    return search(0, [])


def _features(grid):
    inp, bg, rods, arrows, counts = _parse(grid)
    gates = {p for p, n in counts.items() if n >= 2}
    active, trace = _evolve(rods, gates, inp.shape[0])
    final = dict(active)
    initial_owner = {p: i for i, rod in enumerate(rods) for p in _cells(rod)}
    single_hits = Counter(initial_owner[p] for p, n in counts.items()
                          if n == 1 and p in initial_owner)
    traversed = False
    for event in trace:
        for (i, before), (_, after) in zip(event["before"], event["after"]):
            if i not in final:
                continue
            displacement = after[1] - before[1]
            traversed |= any(_cells(_shift(before, d)) & gates
                             for d in range(1, displacement))
    edges = [(i, j) for i, p in enumerate(rods) for j, q in enumerate(rods)
             if i != j and _cells(_shift(p, 1)) & _cells(q)]
    supports = {i: [j for a, j in edges if a == i] for i in range(len(rods))}

    def depth(i, path=()):
        assert i not in path, "Cyclic support graph"
        return 1 + max((depth(j, path + (i,)) for j in supports[i]), default=0)

    profile = tuple(sorted((p[0], len(supports[i]), sum(b == i for _, b in edges),
                            int(any(r == inp.shape[0] - 1 for r, _ in _cells(p))), depth(i))
                           for i, p in enumerate(rods)))
    return {"shape": inp.shape, "background": bg, "rods": len(rods),
            "arrows": len(arrows), "gates": len(gates), "rounds": len(trace),
            "max_convergence": max(counts.values()), "directions": {a[1] for a in arrows},
            "double_convergence": any(n == 2 for n in counts.values()),
            "h_lengths": [p[3] for p in rods if p[0] == "H"],
            "v_lengths": [p[3] for p in rods if p[0] == "V"],
            "deleted": len(rods) - len(active), "survivors": len(active),
            "shared_removed_survivor_color": bool(
                {rods[i][4] for i in range(len(rods)) if i not in final}
                & {p[4] for p in final.values()}),
            "final_falls": [q[1] - rods[i][1] for i, q in active],
            "through_gate_survivor": bool(traversed),
            "two_singles_on_survivor": any(n >= 2 and i in final for i, n in single_hits.items()),
            "support_edges": len(edges),
            "support_profile": profile,
            "redundant_support": any(sum(a == i for a, _ in edges) >= 2 for i in range(len(rods))),
            "rod_palette": len({p[4] for p in rods})}


def _make_pair(rng, mode, rich, background, teach_double=False):
    for _ in range(100):
        geometry = _geometry(rng, mode, rich)
        if geometry is None:
            continue
        shape, base_rods, gates, singles, reuse_pair = geometry
        rods = _color_rods(rng, base_rods, background, reuse_pair)
        if rods is None:
            continue
        active, trace = _evolve(rods, gates, shape[0])
        if mode != "quiet" and not trace:
            continue
        if mode == "cascade" and (len(trace) < 2 or not any(q[1] > rods[i][1] for i, q in active)):
            continue
        final_rods = [p for _, p in active]
        arrows = _place_arrows(rng, shape, rods, final_rods, gates, singles, teach_double)
        if arrows is None:
            continue
        palette = [c for c in range(10) if c != background]
        mono = bool(rng.random() < .5)
        base = int(_pick(rng, palette))
        arrows = [(*a[:3], base if mono else int(_pick(rng, palette))) for a in arrows]
        inp = _render(shape, background, rods, arrows)
        if np.sum(inp != background) * 2 >= inp.size:
            continue
        try:
            _, _, decoded, _, _ = _parse(inp)
        except ValueError:
            continue
        if set(decoded) != set(rods):
            continue
        f = _features(inp)
        if mode == "escape" and not (f["through_gate_survivor"] and f["two_singles_on_survivor"]):
            continue
        return {"input": inp, "output": _render(shape, background, final_rods, arrows)}
    raise RuntimeError(f"Could not construct a readable {mode} witness")


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    backgrounds = [int(c) for c in rng.permutation(10)]
    ordinary = ("cascade", "escape", "pile", "pile", "compact", "quiet")
    modes = ["cascade", "escape"] + [_pick(rng, ordinary)
                                     for _ in range(num_train - 2)]
    puzzle, seen = {"train": [], "test": []}, set()
    for index, mode in enumerate(modes):
        for _ in range(10):
            pair = _make_pair(rng, mode, False, backgrounds[index % 10], index < 2)
            if _key(pair["input"]) not in seen:
                seen.add(_key(pair["input"]))
                puzzle["train"].append(pair)
                break
        else:
            raise RuntimeError("Distinct training inputs exhausted")
    train_profiles = {_features(p["input"])["support_profile"] for p in puzzle["train"]}
    for index in range(num_test):
        # More inserted rods and one extra possible feeder layer; no new mechanics.
        for _ in range(60):
            mode = _pick(rng, ordinary)
            pair = _make_pair(rng, mode, True, backgrounds[(num_train + index) % 10])
            if (_key(pair["input"]) not in seen
                    and _features(pair["input"])["support_profile"] not in train_profiles):
                seen.add(_key(pair["input"]))
                puzzle["test"].append(pair)
                break
        else:
            raise RuntimeError("Distinct test inputs exhausted")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            _, bg, rods, arrows, counts = _parse(inp)
            assert _settle(rods, inp.shape[0]) == rods, "Initially unsupported stick"
            gates = {p for p, n in counts.items() if n >= 2}
            active, trace = _evolve(rods, gates, inp.shape[0])
            final_rods = [p for _, p in active]
            assert _settle(final_rods, inp.shape[0]) == final_rods
            assert all(not _cells(p) & gates for p in final_rods)
            assert all(q[1] >= rods[i][1] and q[0] == rods[i][0]
                       and q[2:] == rods[i][2:] for i, q in active)
            assert all(len(t["removed"]) >= 1 for t in trace)
            assert np.array_equal(out, _render(inp.shape, bg, final_rods, arrows))
            assert np.array_equal(out, solve(inp))
            # A without endpoint convergence leaves the stable input untouched.
            assert np.array_equal(_render(inp.shape, bg, _settle(rods, inp.shape[0]), arrows), inp)
            if split == "train":
                training.append(_features(inp))
            else:
                testing.append(_features(inp))
    assert len({f["background"] for f in training}) >= 2
    assert any(f["rounds"] >= 2 and any(f["final_falls"]) for f in training), "Teach recurrence"
    assert any(f["through_gate_survivor"] and f["two_singles_on_survivor"]
               for f in training), "Teach resting snapshots and coordinate counting"
    assert any(f["double_convergence"] for f in training), "Teach the two-shot threshold"
    assert any(f["shared_removed_survivor_color"] for f in training), "Teach geometric selection"
    assert not ({f["support_profile"] for f in training}
                & {f["support_profile"] for f in testing}), "Require a structural query shift"


def _mechanics_tests():
    # Exact two-round feeder and a cap that survives on the second post.
    rods = [("V", 11, 7, 7, 2), ("V", 11, 12, 7, 3),
            ("H", 10, 5, 10, 4), ("H", 9, 5, 10, 5), ("H", 8, 10, 5, 6)]
    specifications = [((5, 7), (1, 0)), ((5, 2), (1, 1))]
    arrows = [(tip, direction, _arrow(tip, direction), 1) for tip, direction in specifications]
    inp = _render((18, 20), 0, rods, arrows)
    expected = _render((18, 20), 0, rods[:2] + [_shift(rods[4], 2)], arrows)
    assert np.array_equal(solve(inp), expected)
    active, trace = _evolve(rods, {(10, 7)}, 18)
    assert len(trace) == 2 and len(active) == 3
    # Ignoring gravity erases only the bottom feeder, leaving the upper one.
    no_gravity = _render(inp.shape, 0, [p for p in rods if not _cells(p) & {(10, 7)}], arrows)
    assert not np.array_equal(no_gravity, expected)
    # A one-round answer also leaves the upper feeder alive at the gate.
    one_round = _render(inp.shape, 0, [p for _, p in trace[0]["after"]], arrows)
    assert not np.array_equal(one_round, expected)

    rods = [("V", 9, 6, 9, 2), ("V", 9, 16, 9, 3), ("H", 8, 4, 15, 4)]
    specifications = [((6, 6), (1, 0)), ((6, 1), (1, 1)),
                      ((6, 16), (1, 0)), ((6, 21), (1, -1)),
                      ((3, 10), (1, 1)), ((3, 13), (1, -1))]
    arrows = [(tip, direction, _arrow(tip, direction), 1) for tip, direction in specifications]
    inp = _render((18, 23), 0, rods, arrows)
    expected = _render((18, 23), 0, [_shift(rods[2], 9)], arrows)
    assert np.array_equal(solve(inp), expected)
    f = _features(inp)
    assert f["through_gate_survivor"] and f["two_singles_on_survivor"]
    assert f["gates"] == 2 and f["rounds"] == 1
    # Colors are identities for drawing, never commands or distance encodings.
    palette = np.array([7, 0, 9, 5, 2, 8, 1, 4, 6, 3], dtype=np.int8)
    assert np.array_equal(solve(palette[inp]), palette[expected])
    assert np.array_equal(solve(np.fliplr(inp)), np.fliplr(expected))
    assert _settle(list(reversed(rods)), 18) == list(reversed(rods))


def self_test():
    """Bounded semantic fixtures, 42-seed diversity, replay and count checks."""
    _mechanics_tests()
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple((_key(p["input"]), _key(p["output"]))
                             for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair["input"])
                f["split"] = split
                records.append(f)
                inp = np.asarray(pair["input"])
                snapshot = inp.copy()
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, snapshot), "Solver mutated its input"
    assert len(signatures) == 42
    assert len({f["shape"] for f in records}) >= 40
    assert len({f["rods"] for f in records}) >= 8
    assert len({f["arrows"] for f in records}) >= 5
    assert {f["background"] for f in records} == set(range(10))
    assert set().union(*(f["directions"] for f in records)) == set(DIRECTIONS)
    assert len({n for f in records for n in f["h_lengths"]}) >= 10
    assert len({n for f in records for n in f["v_lengths"]}) >= 8
    assert max(f["rounds"] for f in records) >= 4
    assert any(f["gates"] >= 3 for f in records)
    assert any(f["max_convergence"] >= 3 for f in records)
    assert any(len(set(f["final_falls"]) - {0}) >= 2 for f in records)
    assert any(f["redundant_support"] for f in records)
    assert min(min(f["shape"]) for f in records) <= 15
    assert any(f["rounds"] == 0 for f in records)
    assert any(f["survivors"] == 0 for f in records)
    for split in ("train", "test"):
        subset = [f for f in records if f["split"] == split]
        assert any(f["rounds"] >= 2 for f in subset)
        assert any(f["through_gate_survivor"] for f in subset)
    for counts in ((2, 1), (10, 1), (10, 4), (6, 3)):
        a, b = generate_puzzle(918266237, *counts), generate_puzzle(918266237, *counts)
        assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Illegal counts accepted: {counts}")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
