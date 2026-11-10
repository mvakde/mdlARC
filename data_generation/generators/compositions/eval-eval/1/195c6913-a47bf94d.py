"""Cyclic echoes: input-only strand distances control complementary sockets.

Only the standard library and NumPy are used. Generation computes answers from
constructed polylines; solve parses the actual input without construction data.
"""
from collections import deque
from copy import deepcopy

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
CHECKER = ~DIAMOND
REQUIRED_EVIDENCE = {
    "clock", "multiplicity", "detour", "overpass", "bend_phase",
    "crossing_tick", "mouth_tick", "unequal_ends", "multiple_beacons",
    "outward_order_0", "outward_order_1",
}


def _add(p, v, scale=1):
    return p[0] + scale * v[0], p[1] + scale * v[1]


def _delta(a, b):
    return b[0] - a[0], b[1] - a[1]


def _neighbors(p):
    return [_add(p, d) for d in DIRECTIONS]


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= d <= 30 for d in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((g >= 0) & (g <= 9))
    return g.astype(np.int8, copy=False)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        found, stack = {p}, [p]
        while stack:
            for q in _neighbors(stack.pop()):
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _bbox(cells):
    return (min(p[0] for p in cells), max(p[0] for p in cells),
            min(p[1] for p in cells), max(p[1] for p in cells))


def _square(center):
    r, c = center
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _parse(input_grid):
    """Parse sockets, visible role grammar, and directional strand connectivity."""
    g = _grid(input_grid)
    corners = [int(g[p]) for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    assert len(set(corners)) == 1, "background must be visible at corners"
    bg = corners[0]
    components = {int(color): _components(tuple(map(int, p)) for p in
                                          np.argwhere(g == color))
                  for color in np.unique(g) if color != bg}
    swatches = []
    for color, groups in components.items():
        for cells in groups:
            r0, r1, c0, c1 = _bbox(cells)
            if len(cells) == 4 and r1 - r0 == c1 - c0 == 1:
                swatches.append((r0, c0, color, cells))
    assert 2 <= len(swatches) <= 6, "expected one complete swatch word"
    assert len({s[0] for s in swatches}) == 1
    swatches.sort(key=lambda s: s[1])
    assert all(b[1] - a[1] >= 3 for a, b in zip(swatches, swatches[1:]))
    word = tuple(s[2] for s in swatches)
    assert len(set(word)) >= 2
    legend = set().union(*(s[3] for s in swatches))
    for p in legend:
        for q in _neighbors(p):
            if q not in legend and 0 <= q[0] < g.shape[0] and 0 <= q[1] < g.shape[1]:
                assert g[q] == bg, "swatches need a clear surrounding field"

    candidates = []
    for color, groups in components.items():
        if color in word:
            continue
        sites = []
        for cells in groups:
            r0, r1, c0, c1 = _bbox(cells)
            if r1 - r0 != 2 or c1 - c0 != 2:
                break
            center = r0 + 1, c0 + 1
            square = _square(center)
            if cells == square:
                kind = "solid"
            elif cells == square - {center} and g[center] == bg:
                kind = "hollow"
            else:
                break
            sites.append(dict(center=center, kind=kind, cells=square))
        else:
            if 2 <= len(sites) <= 8 and len(sites) % 2 == 0:
                candidates.append((color, sites))
    assert len(candidates) == 1, "socket color must be shape identifiable"
    socket_color, sockets = candidates[0]
    footprint = set().union(*(s["cells"] for s in sockets))
    assert sum(len(s["cells"]) for s in sockets) == len(footprint)
    assert min(p[0] for p in footprint) > swatches[0][0] + 2
    visible = {tuple(map(int, p)) for p in np.argwhere(g != bg)
               if tuple(p) not in footprint and tuple(p) not in legend}
    assert visible
    adjacency = {p: [q for q in _neighbors(p) if q in visible] for p in visible}
    ports = {}
    for i, socket in enumerate(sockets):
        center = socket["center"]
        possible = [_add(center, d, 2) for d in DIRECTIONS]
        actual = [p for p in possible if p in visible]
        touching = {q for p in socket["cells"] for q in _neighbors(p) if q in visible}
        assert len(actual) == 1 and touching == set(actual), "socket has one midpoint port"
        p = actual[0]
        assert p not in ports and len(adjacency[p]) == 1
        ports[p] = i
        socket["port"] = p
    crossings = {p for p in visible if len(adjacency[p]) == 4}
    cross_colors = {int(g[p]) for p in crossings}
    assert len(cross_colors) <= 1
    cross = next(iter(cross_colors)) if crossings else None
    assert cross not in set(word) | {bg, socket_color}
    beacons = {p for p in visible if g[p] == word[0]}
    assert beacons and all(len(adjacency[p]) == 2 for p in beacons)
    assert all(g[p] == bg or tuple(p) in legend for color in set(word[1:]) - {word[0]}
               for p in map(tuple, np.argwhere(g == color)))
    carrier_colors = {int(g[p]) for p in visible - crossings - beacons}
    assert len(carrier_colors) == 1
    carrier = carrier_colors.pop()
    assert carrier not in set(word) | {bg, socket_color, cross}
    for p, neighbors in adjacency.items():
        assert len(neighbors) == (1 if p in ports else 4 if p in crossings else 2)
        assert g[p] == (cross if p in crossings else word[0] if p in beacons else carrier)
    assert all(len(c) == 1 for c in _components(beacons)), "beacons are singletons"

    routes, used_ports, covered = [], set(), set()
    cross_axes = {p: [] for p in crossings}
    for port in sorted(ports):
        if port in used_ports:
            continue
        path, states = [], set()
        previous, current = None, port
        while True:
            assert (previous, current) not in states, "strand cycle"
            states.add((previous, current))
            path.append(current)
            if current in ports and current != port:
                break
            if previous is None:
                following = adjacency[current][0]
            elif current in crossings:
                direction = _delta(previous, current)
                following = _add(current, direction)
                assert following in adjacency[current]
                cross_axes[current].append(abs(direction[0]))
            else:
                assert previous in adjacency[current]
                following = next(q for q in adjacency[current] if q != previous)
            previous, current = current, following
            assert len(path) <= 2 * len(visible), "nonterminating strand"
        assert current not in used_ports
        used_ports.update((port, current))
        covered.update(path)
        pair = ports[port], ports[current]
        assert {sockets[i]["kind"] for i in pair} == {"solid", "hollow"}
        launch = [k for k, p in enumerate(path) if p in beacons]
        assert len(launch) == 1, "one beacon on each strand"
        k = launch[0]
        assert 0 < k < len(path) - 1
        routes.append(dict(sockets=pair, path=tuple(path), beacon_index=k,
                           distances=(k + 1, len(path) - k)))
    assert covered == visible and used_ports == set(ports), "unattached fragment"
    assert all(sorted(axes) == [0, 1] for axes in cross_axes.values())
    assert len(routes) == len(beacons) and 1 <= len(routes) <= 4
    return dict(grid=g, background=bg, socket_color=socket_color, carrier=carrier,
                crossing_color=cross, sockets=sockets, footprint=footprint,
                word=word, legend=legend, swatches=swatches, visible=visible,
                adjacency=adjacency, crossings=crossings, beacons=beacons, routes=routes)


def _render(info, terminal_colors=None):
    out = info["grid"].copy()
    colors = {}
    if terminal_colors is None:
        for route in info["routes"]:
            for index, distance in zip(route["sockets"], route["distances"]):
                colors[index] = info["word"][distance % len(info["word"])]
    else:
        colors = terminal_colors
    for index, socket in enumerate(info["sockets"]):
        r, c = socket["center"]
        pattern = DIAMOND if socket["kind"] == "solid" else CHECKER
        out[r - 1:r + 2, c - 1:c + 2] = np.where(pattern, colors[index], info["background"])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive both terminal phases entirely from this input's visible strands."""
    return _render(_parse(input_grid))


def _evidence(info):
    word = info["word"]
    unique = tuple(dict.fromkeys(word))
    result, readouts = set(), []
    # Wrong junction rule: echoes spread across every edge of an ordinary graph.
    distances = {p: 0 for p in info["beacons"]}
    queue = deque(sorted(distances))
    while queue:
        p = queue.popleft()
        for q in info["adjacency"][p]:
            if q not in distances:
                distances[q] = distances[p] + 1
                queue.append(q)
    for route in info["routes"]:
        path, k = route["path"], route["beacon_index"]
        colors = []
        for side, (socket, distance) in enumerate(zip(route["sockets"], route["distances"])):
            color = word[distance % len(word)]
            colors.append(color)
            if color != word[0]:
                result.add("clock")
            if color != unique[distance % len(unique)]:
                result.add("multiplicity")
            if color != word[(distance - 1) % len(word)]:
                result.add("mouth_tick")
            # Both ends must teach advancing the word outward, rather than
            # decreasing the phase on one direction of an oriented cable.
            if color != word[(-distance) % len(word)]:
                result.add("outward_order_" + str(side))
            branch = path[k::-1] if side == 0 else path[k:]
            manhattan = sum(abs(a - b) for a, b in zip(branch[0], branch[-1])) + 1
            if color != word[manhattan % len(word)]:
                result.add("detour")
            if color != word[(distances[branch[-1]] + 1) % len(word)]:
                result.add("overpass")
            count_cross = sum(p in info["crossings"] for p in branch[1:])
            if color != word[(distance - count_cross) % len(word)]:
                result.add("crossing_tick")
            phase, old_direction = 0, None
            for a, b in zip(branch, branch[1:]):
                direction = _delta(a, b)
                phase = 0 if old_direction is not None and direction != old_direction else phase + 1
                old_direction = direction
            # A socket mouth is on the last leg, so no additional bend is made.
            if color != word[(phase + 1) % len(word)]:
                result.add("bend_phase")
        if colors[0] != colors[1]:
            result.add("unequal_ends")
        readouts.append(tuple(colors[i] for i in sorted(range(2), key=lambda i:
                         info["sockets"][route["sockets"][i]]["kind"])))
    if len(set(readouts)) >= 2:
        result.add("multiple_beacons")
    return result


def _polyline(points):
    path = [points[0]]
    for a, b in zip(points, points[1:]):
        dr, dc = _delta(a, b)
        assert (dr == 0) != (dc == 0)
        direction = int(np.sign(dr)), int(np.sign(dc))
        path.extend(_add(a, direction, k) for k in range(1, abs(dr) + abs(dc) + 1))
    return tuple(path)


def _socket_centers(path):
    return (_add(path[0], _delta(path[1], path[0]), 2),
            _add(path[-1], _delta(path[-2], path[-1]), 2))


def _geometry_ok(paths, shape):
    """Check construction geometry without colors, clues, or a solution parser."""
    h, w = shape
    sockets = [center for path in paths for center in _socket_centers(path)]
    blocks = [_square(center) for center in sockets]
    for block in blocks:
        if any(not (5 <= r <= h - 2 and 1 <= c <= w - 2) for r, c in block):
            return False
    for i, block in enumerate(blocks):
        for other in blocks[i + 1:]:
            if any(q in other for p in block for q in [p] + _neighbors(p)):
                return False
    visible = set().union(*(set(path) for path in paths))
    if any(set(path) & set().union(*blocks) for path in paths):
        return False
    owners = {p: [i for i, path in enumerate(paths) if p in path] for p in visible}
    endpoints = {path[j] for path in paths for j in (0, -1)}
    for path in paths:
        if len(path) < 6 or len(set(path)) != len(path):
            return False
        for k, p in enumerate(path):
            local = {path[j] for j in (k - 1, k + 1) if 0 <= j < len(path)}
            if {q for q in _neighbors(p) if q in path} != local:
                return False
    for p in visible:
        neighbors = {q for q in _neighbors(p) if q in visible}
        if len(owners[p]) == 1:
            if len(neighbors) != (1 if p in endpoints else 2):
                return False
        elif len(owners[p]) == 2 and len(neighbors) == 4 and p not in endpoints:
            axes = []
            for i in owners[p]:
                path = paths[i]
                k = path.index(p)
                if not 0 < k < len(path) - 1:
                    return False
                before, after = _delta(path[k - 1], p), _delta(p, path[k + 1])
                if before != after:
                    return False
                axes.append(abs(before[0]))
            if sorted(axes) != [0, 1]:
                return False
        else:
            return False
    for block, port in zip(blocks, [p for path in paths for p in (path[0], path[-1])]):
        if {q for p in block for q in _neighbors(p) if q in visible} != {port}:
            return False
    return True


def _random_path(rng, shape, maximum_segments):
    h, w = shape
    p = int(rng.integers(8, h - 3)), int(rng.integers(3, w - 3))
    points = [p]
    direction = int(rng.integers(4))
    segments = int(rng.integers(1, maximum_segments + 1))
    for j in range(segments):
        if j:
            direction = (direction + int(rng.choice((-1, 1)))) % 4
        d = DIRECTIONS[direction]
        cap = 0
        for k in range(1, 13):
            r, c = _add(p, d, k)
            if not (8 <= r <= h - 4 and 3 <= c <= w - 4):
                break
            cap = k
        if cap < 3:
            return None
        p = _add(p, d, int(rng.integers(3, cap + 1)))
        points.append(p)
    return _polyline(points)


def _witness_paths(rng):
    """A detour and a strand crossing it twice; dimensions and delays vary."""
    h, w = int(rng.integers(28, 31)), int(rng.integers(24, 31))
    upper = int(rng.integers(12, 16))
    lower = int(rng.integers(20, h - 5))
    bend = int(rng.integers(17, w - 4))
    vertical = int(rng.integers(10, bend - 2))
    paths = [_polyline(((lower, 4), (lower, bend), (upper, bend), (upper, 4))),
             _polyline(((8, vertical), (h - 5, vertical)))]
    if rng.random() < .5:
        paths = [tuple((r, w - 1 - c) for r, c in path) for path in paths]
    assert _geometry_ok(paths, (h, w))
    return (h, w), paths


def _sample_geometry(rng, test=False, simple=False):
    for _ in range(12):
        shape = (int(rng.integers(22 if test else 20, 31)),
                 int(rng.integers(22 if test else 20, 31)))
        n = 1 if simple else int(rng.choice((2, 3, 4), p=(.45, .40, .15))) if test else int(rng.integers(1, 3))
        paths = []
        for i in range(n):
            for _ in range(450):
                path = _random_path(rng, shape, 2 if simple else 6 if test else 3)
                if path is not None and _geometry_ok(paths + [path], shape):
                    paths.append(path)
                    break
            else:
                break
        if len(paths) == n:
            return shape, paths
    # Visible valid backstop; never an exception to the family rule.
    return _witness_paths(rng)


def _word_pattern(rng, length, repeated=False):
    if repeated:
        choices = ((0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 1, 1, 0),
                   (0, 0, 1, 1), (0, 1, 0, 2), (0, 1, 2, 1), (0, 0, 1, 2))
        return choices[int(rng.integers(len(choices)))]
    distinct = int(rng.integers(2, length + 1))
    symbols = list(range(distinct)) + list(map(int, rng.integers(distinct, size=length - distinct)))
    rng.shuffle(symbols)
    return tuple(symbols)


def _make_pair(rng, lesson=None, test=False):
    for attempt in range(1000):
        shape, paths = _witness_paths(rng) if lesson == "witness" else _sample_geometry(rng, test, lesson == "simple")
        h, w = shape
        pattern = _word_pattern(rng, int(rng.integers(5, 7) if test else rng.integers(2, 5)),
                                lesson == "witness")
        palette = list(map(int, rng.permutation(10)))
        bg, socket_color, carrier, cross = palette[:4]
        word = tuple(palette[4 + symbol] for symbol in pattern)
        inp = np.full(shape, bg, dtype=np.int8)
        visits = {}
        for path in paths:
            for p in path:
                inp[p] = carrier
                visits[p] = visits.get(p, 0) + 1
        for p, count in visits.items():
            if count == 2:
                inp[p] = cross
        launch_candidates = [
            [k for k in range(1, len(path) - 1) if visits[path[k]] == 1
             and all(visits.get(q, 0) < 2 for q in _neighbors(path[k]))]
            for path in paths
        ]
        if any(not candidates for candidates in launch_candidates):
            # Connectivity alone does not guarantee room for a readable launch
            # clue, especially on a short strand between nearby overpasses.
            continue
        beacon_indices, kinds = [], []
        for path, candidates in zip(paths, launch_candidates):
            k = int(rng.choice(candidates))
            beacon_indices.append(k)
            inp[path[k]] = word[0]
            order = ("solid", "hollow") if rng.random() < .5 else ("hollow", "solid")
            kinds.append(order)
            for center, kind in zip(_socket_centers(path), order):
                r, c = center
                inp[r - 1:r + 2, c - 1:c + 2] = socket_color
                if kind == "hollow":
                    inp[center] = bg
        # The legend is visually grouped, in a clear strip above all sockets.
        row = int(rng.integers(1, 3))
        slack = w - (3 * len(word) - 1) - 2
        left = int(rng.integers(1, min(5, slack) + 1))
        cols = [left]
        spare = w - 1 - (left + 3 * len(word) - 1)
        for j in range(len(word) - 1):
            extra = int(rng.integers(min(2, spare) + 1))
            cols.append(cols[-1] + 3 + extra)
            spare -= extra
        for c, color in zip(cols, word):
            inp[row:row + 2, c:c + 2] = color
        # Construction-based answer: no input parser or solve calls used here.
        out = inp.copy()
        for path, k, order in zip(paths, beacon_indices, kinds):
            for center, kind, distance in zip(_socket_centers(path), order, (k + 1, len(path) - k)):
                r, c = center
                motif = DIAMOND if kind == "solid" else CHECKER
                out[r - 1:r + 2, c - 1:c + 2] = np.where(motif, word[distance % len(word)], bg)
        info = _parse(inp)
        assert np.array_equal(out, _render(info)), "visible trace and construction disagree"
        if lesson == "witness" and not REQUIRED_EVIDENCE <= _evidence(info):
            continue
        return {"input": inp, "output": out}
    raise AssertionError("bounded witness sampling failed")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test) and
            2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("need integer counts: 2..10 train, 1..4 test, train > test")
    return int(num_train), int(num_test)


def _key(grid):
    g = np.asarray(grid, dtype=np.int8)
    return g.shape, g.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if not _integer(seed):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    lesson_slots = list(map(int, rng.choice(nt, size=2, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            lesson = ("witness" if i == lesson_slots[0] else
                      "simple" if i == lesson_slots[1] else None) if split == "train" else None
            for _ in range(30):
                pair = _make_pair(rng, lesson, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise AssertionError("could not obtain distinct inputs")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, evidence = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _parse(inp)
            expected = _render(info)
            assert np.array_equal(out, expected), "incorrect terminal colors or stencil"
            untouched = np.ones(inp.shape, dtype=bool)
            for p in info["footprint"]:
                untouched[p] = False
            assert np.array_equal(inp[untouched], out[untouched]), "diagram changed"
            if split == "train":
                evidence |= _evidence(info)
    assert REQUIRED_EVIDENCE <= evidence, REQUIRED_EVIDENCE - evidence


def _puzzle_key(puzzle):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"])) for pair in puzzle[s])
                 for s in ("train", "test"))


def _equality_pattern(word):
    labels = {}
    return tuple(labels.setdefault(c, len(labels)) for c in word)


def self_test():
    """Bounded 42-seed structural audit plus count and mutation checks."""
    shapes, patterns, counts, lengths, delays, crossings, palettes, structures = (set() for _ in range(8))
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_key(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            info = _parse(pair["input"])
            assert np.array_equal(solve(pair["input"]), pair["output"])
            shapes.add(info["grid"].shape)
            patterns.add(_equality_pattern(info["word"]))
            counts.add(len(info["routes"]))
            lengths.add(len(info["word"]))
            crossings.add(len(info["crossings"]))
            palettes.add((info["background"], info["socket_color"], info["carrier"]))
            role_map = {}
            normalized = tuple(role_map.setdefault(int(x), len(role_map)) for x in info["grid"].flat)
            structures.add((info["grid"].shape, normalized))
            for route in info["routes"]:
                delays.update(route["distances"])
    assert len(signatures) == 42
    assert len(shapes) >= 25 and len(structures) >= 150
    assert lengths == {2, 3, 4, 5, 6}
    assert counts >= {1, 2, 3}
    assert {0, 1, 2} <= crossings
    assert len(patterns) >= 20 and len(delays) >= 25 and len(palettes) >= 80
    assert _puzzle_key(generate_puzzle(112358)) == _puzzle_key(generate_puzzle(112358))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(193 + 7 * nt + ne, nt, ne)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (3.0, 1), (True, 1), (4, None), (4, False)):
        try:
            generate_puzzle(1, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(0, 2, 1)
    broken = deepcopy(puzzle)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("nonterminal corruption accepted")
    broken = deepcopy(puzzle)
    info = _parse(broken["train"][0]["input"])
    broken["train"][0]["output"] = _render(info, {i: info["word"][0] for i in range(len(info["sockets"]))})
    # If this particular simple lesson already agrees with copying, mutate the witness instead.
    if np.array_equal(broken["train"][0]["output"], puzzle["train"][0]["output"]):
        info = _parse(broken["train"][1]["input"])
        broken["train"][1]["output"] = _render(info, {i: info["word"][0] for i in range(len(info["sockets"]))})
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("ignoring cyclic advancement accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal counts, semantic mutations)")
