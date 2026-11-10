"""Recover a run-expanded cycle, then highlight its minimum-station arc.

Input-only conventions: a blank border identifies the background; stations are
isolated single cells AFTER whole-line compression. Exactly two stations have
the endpoint color. Output colors 3 (stations) and 5 (connectors) are fixed.
Only the standard library and NumPy are used. No parent generator is imported.
"""
from collections import defaultdict
from itertools import combinations
import hashlib
import json

import numpy as np


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_INPUT_COLORS = (0, 1, 2, 4, 6, 7, 8, 9)
_CHECK_SEEDS = (0, 747625288, 738406813, 166931442, 153875752, 198707609,
                746528497, 901888583, 777396233, 381178174, 290485731,
                965661961, 864378517, 386499810, 40850419, 750688068,
                669681538, 745228707, 618743952, 303531087, 720052446,
                28415371, 839748847, 306220869, 206739704, 654693855,
                826284958, 930459937, 818467480, 919296059, 83663774,
                412193373, 796813215, 551872744, 648318683, 801350946,
                415183485, 415612746, 857723760, 168199741, 841275580,
                481496326)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "invalid color")
    return grid


def _starts(grid, axis):
    """Compare full original vectors, never individual colored rectangles."""
    if axis == 0:
        return np.r_[0, np.flatnonzero(np.any(grid[1:] != grid[:-1], axis=1)) + 1]
    return np.r_[0, np.flatnonzero(np.any(grid[:, 1:] != grid[:, :-1], axis=0)) + 1]


def _compress(grid):
    return grid[np.ix_(_starts(grid, 0), _starts(grid, 1))].copy()


def _runs(grid, axis):
    return tuple(int(x) for x in np.diff(np.r_[_starts(grid, axis), grid.shape[axis]]))


def _expand(grid, rows, columns):
    return np.repeat(np.repeat(grid, rows, axis=0), columns, axis=1)


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _cycle(grid, background):
    """Trace a four-neighbor induced pixel cycle, including all connector bends."""
    foreground = set(map(tuple, np.argwhere(grid != background).tolist()))
    _require(len(foreground) >= 10, "cycle too small")
    adjacency = {p: [(p[0] + dr, p[1] + dc) for dr, dc in _DIRECTIONS
                     if (p[0] + dr, p[1] + dc) in foreground] for p in foreground}
    _require(all(len(v) == 2 for v in adjacency.values()),
             "foreground must be an unbranched four-neighbor cycle")
    start = min(foreground)
    order, seen, previous, current = [start], {start}, None, start
    while True:
        nxt = next(p for p in adjacency[current] if p != previous)
        if nxt == start:
            break
        _require(nxt not in seen, "self-contact")
        order.append(nxt)
        seen.add(nxt)
        previous, current = current, nxt
    _require(len(order) == len(foreground), "multiple cycles")
    return order


def _arc_choices(grid, order, ordinary):
    sites = [i for i, p in enumerate(order) if grid[p] == ordinary]
    choices = []
    for a, b in combinations(range(len(sites)), 2):
        first = order[sites[a] + 1:sites[b]]
        second = order[sites[b] + 1:] + order[:sites[a]]
        counts = (b - a - 1, len(sites) - b + a - 1)
        if counts[0] == counts[1] or min(counts) == 0:
            continue
        arcs = (first, second)
        winner = int(counts[1] < counts[0])
        areas = tuple(len(arc) - count for arc, count in zip(arcs, counts))
        choices.append((sites[a], sites[b], min(counts),
                        areas[winner] > areas[1 - winner]))
    return choices


def _decoy_station(grid, order, a, b):
    """Can one selected station's 3x3 expansion outweigh every alternative node?"""
    arcs = (order[a + 1:b], order[b + 1:] + order[:a])
    nodes = [[p for p in arc if grid[p] == 2] for arc in arcs]
    winner = int(len(nodes[1]) < len(nodes[0]))
    if len(nodes[winner]) != 1:
        return None
    target = nodes[winner][0]
    other_area = sum((3 if p[0] == target[0] else 1)
                     * (3 if p[1] == target[1] else 1) for p in nodes[1 - winner])
    return target if other_area < 9 else None


def _read_compact(grid):
    colors = set(int(x) for x in np.unique(grid))
    _require(len(colors) == 4 and colors.isdisjoint((3, 5)), "invalid input palette")
    background = int(grid[0, 0])
    _require(bool(np.all(grid[0] == background) and np.all(grid[-1] == background)
                  and np.all(grid[:, 0] == background) and np.all(grid[:, -1] == background)),
             "a complete blank border is required")
    endpoint_colors = [c for c in colors - {background} if np.count_nonzero(grid == c) == 2]
    _require(len(endpoint_colors) == 1, "endpoint color must mark exactly two cells")
    endpoint = endpoint_colors[0]
    ordinary_candidates = []
    for color in colors - {background, endpoint}:
        cells = set(map(tuple, np.argwhere(grid == color).tolist()))
        if len(cells) >= 3 and all((r + dr, c + dc) not in cells
                                  for r, c in cells for dr, dc in _DIRECTIONS):
            ordinary_candidates.append(color)
    _require(len(ordinary_candidates) == 1, "ordinary station color is ambiguous")
    ordinary = ordinary_candidates[0]
    connector = next(iter(colors - {background, endpoint, ordinary}))
    order = _cycle(grid, background)
    station_indices = [i for i, p in enumerate(order) if grid[p] in (ordinary, endpoint)]
    length = len(order)
    _require(all((station_indices[(i + 1) % len(station_indices)] - a) % length >= 2
                 for i, a in enumerate(station_indices)), "adjacent stations")
    endpoints = [i for i, p in enumerate(order) if grid[p] == endpoint]
    a, b = endpoints
    arcs = (order[a + 1:b], order[b + 1:] + order[:a])
    counts = tuple(sum(grid[p] == ordinary for p in arc) for arc in arcs)
    _require(counts[0] != counts[1], "tied station counts are outside this family")
    winner = int(counts[1] < counts[0])
    _require(min(counts) >= 1, "generated family uses positive selected station counts")
    return {"roles": (background, connector, ordinary, endpoint), "order": order,
            "arcs": arcs, "counts": tuple(int(x) for x in counts), "winner": winner,
            "station_count": len(station_indices),
            "areas": tuple(int(sum(grid[p] == connector for p in arc)) for arc in arcs)}


def _paint(grid, data):
    out = grid.copy()
    _, connector, ordinary, _ = data["roles"]
    for p in data["arcs"][data["winner"]]:
        out[p] = 3 if grid[p] == ordinary else 5
        _require(grid[p] in (ordinary, connector), "endpoint included in open arc")
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only this input: whole-line collapse, visible cycle, station counts."""
    compact = _compress(_grid(input_grid))
    return _paint(compact, _read_compact(compact))


def _boundary(rng):
    """A random polyomino supplies a single orthogonal polygon, not a puzzle."""
    nr, nc = map(int, rng.integers(2, 5, size=2))
    target = int(rng.integers(2, min(nr * nc, 11) + 1))
    cells = {(int(rng.integers(nr)), int(rng.integers(nc)))}
    while len(cells) < target:  # at most ten additions
        choices = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _DIRECTIONS
                          if 0 <= r + dr < nr and 0 <= c + dc < nc} - cells)
        cells.add(choices[int(rng.integers(len(choices)))])
    edges = set()
    for r, c in sorted(cells):
        corners = [(r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c)]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            edge = tuple(sorted((a, b)))
            if edge in edges:
                edges.remove(edge)
            else:
                edges.add(edge)
    adjacency = defaultdict(list)
    for a, b in sorted(edges):
        adjacency[a].append(b)
        adjacency[b].append(a)
    if any(len(v) != 2 for v in adjacency.values()):
        return None  # reject pinches
    start = min(adjacency)
    order, seen, previous, current = [start], {start}, None, start
    while True:
        nxt = next(p for p in adjacency[current] if p != previous)
        if nxt == start:
            break
        if nxt in seen:
            return None
        order.append(nxt)
        seen.add(nxt)
        previous, current = current, nxt
    if len(order) != len(adjacency):
        return None  # holes or disconnected boundary components
    pitch = int(rng.integers(3, 6))
    points = []
    for a, b in zip(order, order[1:] + order[:1]):
        points.extend((a[0] * pitch + (b[0] - a[0]) * t,
                       a[1] * pitch + (b[1] - a[1]) * t) for t in range(pitch))
    low = np.min(points, axis=0) - 1
    return [tuple(int(v) for v in np.asarray(p) - low) for p in points]


def _fallback(k, selected):
    """Guaranteed long sparse lower arc versus dense upper arc, for every k."""
    width, height = 2 * (k - selected - 1), 6
    g = np.zeros((height + 3, width + 3), dtype=np.int8)
    g[1, 1:width + 2] = g[height + 1, 1:width + 2] = 1
    g[1:height + 2, 1] = g[1:height + 2, width + 1] = 1
    g[1, 1] = g[1, width + 1] = 2
    for j in range(k - selected - 2):
        g[1, 3 + 2 * j] = 2
    for j in range(selected):
        g[height + 1, 3 + 2 * j] = 2
    return _compress(g)


def _latent(rng, k, witness, minimum):
    for attempt in range(241):
        if attempt == 240:
            g = _fallback(k, minimum)
        else:
            points = _boundary(rng)
            if points is None or len(points) < 2 * k:
                continue
            shape = tuple(np.max(points, axis=0) + 2)
            g = np.zeros(shape, dtype=np.int8)
            for p in points:
                g[p] = 1
            gaps = rng.multinomial(len(points) - 2 * k, [1 / k] * k) + 2
            position = int(rng.integers(len(points)))
            for gap in gaps:
                g[points[position % len(points)]] = 2
                position += int(gap)
            # Normalize the drawing before defining the latent member. Verify
            # that this removes only redundant corridor length, never a station.
            g = _compress(g)
        _require(np.count_nonzero(g == 2) == k, "normalization erased stations")
        order = _cycle(g, 0)
        if not any(g[p] == g[order[(i + 1) % len(order)]] == 1
                   for i, p in enumerate(order)):
            continue  # retain a visible distinction between paths and stations
        choices = [x for x in _arc_choices(g, order, 2)
                   if x[2] >= minimum and (not witness or x[3])]
        if witness:
            choices = [x for x in choices if x[2] == minimum
                       and _decoy_station(g, order, x[0], x[1]) is not None]
        if not choices:
            continue
        a, b, _, _ = choices[int(rng.integers(len(choices)))]
        g[order[a]] = g[order[b]] = 4
        _require(np.array_equal(_compress(g), g), "hidden latent line boundary")
        g = np.rot90(g, int(rng.integers(4)))
        if rng.random() < .5:
            g = g[:, ::-1]
        return g.copy()
    raise AssertionError("constructive fallback failed")


def _schedule(rng, length, mode):
    runs = np.ones(length, dtype=int)
    if mode == "none":
        return tuple(int(x) for x in runs)
    if mode == "test_spike":
        j = int(rng.integers(length))
        runs[j] = int(rng.integers(4, min(8, 31 - length) + 1))
        budget = min(30 - int(runs.sum()), int(rng.integers(0, 5)))
    elif mode == "uniform" and 2 * length <= 30:
        return (2,) * length
    else:
        budget = int(rng.integers(2, min(10, 30 - length) + 1))
    for _ in range(budget):
        options = np.flatnonzero(runs < 3)
        if not len(options):
            break
        runs[int(options[int(rng.integers(len(options)))])] += 1
    return tuple(int(x) for x in runs)


def _make_pair(rng, k, witness, minimum, operation, test):
    symbolic = _latent(rng, k, witness, minimum)
    roles = tuple(int(x) for x in rng.choice(_INPUT_COLORS, 4, replace=False))
    lookup = np.zeros(5, dtype=np.int8)
    lookup[[0, 1, 2, 4]] = roles
    compact = lookup[symbolic]
    data = _read_compact(compact)
    out = _paint(compact, data)
    modes = ["mixed", "mixed"]
    if operation == "rows":
        modes[1] = "none"
    elif operation == "columns":
        modes[0] = "none"
    elif operation == "free":
        modes = [str(rng.choice(("mixed", "uniform", "none"))) for _ in range(2)]
        if modes == ["none", "none"]:
            modes[int(rng.integers(2))] = "mixed"
    if test:
        modes[int(rng.integers(2))] = "test_spike"
    rows = _schedule(rng, compact.shape[0], modes[0])
    columns = _schedule(rng, compact.shape[1], modes[1])
    if witness:
        # A second, genuinely joint witness: counting raw station pixels prefers
        # the wrong arc. Blank-border run variations do not affect this decoy.
        target = next(p for p in data["arcs"][data["winner"]] if compact[p] == roles[2])
        rows, columns = [1] * len(compact), [1] * compact.shape[1]
        rows[target[0]] = columns[target[1]] = 3
        for runs in (rows, columns):
            runs[0], runs[-1] = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    return {"input": _expand(compact, rows, columns), "output": out}


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "test":
                k, witness, minimum, operation = int(rng.integers(11, 15)), False, 1, "both"
            elif i == 0:
                k, witness, minimum, operation = int(rng.integers(6, 9)), True, 1, "both"
            elif i == 1:
                k, witness, minimum = int(rng.integers(8, 11)), False, 2
                operation = str(rng.choice(("rows", "columns")))
            else:
                k, witness, minimum, operation = int(rng.integers(5, 11)), False, 1, "free"
            for _ in range(20):
                pair = _make_pair(rng, k, witness, minimum, operation, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("distinct-input retry budget exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    inp = np.asarray(pair["input"])
    compact = _compress(inp)
    data = _read_compact(compact)
    win = data["winner"]
    order = data["order"]
    directions = [(order[(i + 1) % len(order)][0] - p[0],
                   order[(i + 1) % len(order)][1] - p[1]) for i, p in enumerate(order)]
    corners = sum(directions[i] != directions[i - 1] for i in range(len(order)))
    rows, columns = _runs(inp, 0), _runs(inp, 1)
    pixel_areas = [sum(rows[p[0]] * columns[p[1]] for p in arc
                       if compact[p] == data["roles"][2]) for arc in data["arcs"]]
    return {"shape": compact.shape, "input_shape": inp.shape, "roles": data["roles"],
            "stations": data["station_count"], "selected": min(data["counts"]),
            "gap": abs(data["counts"][0] - data["counts"][1]), "corners": corners,
            "witness": data["areas"][win] > data["areas"][1 - win],
            "station_area_conflict": pixel_areas[win] > pixel_areas[1 - win],
            "rows": rows, "columns": columns}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "pair lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            compact = _compress(inp)
            data = _read_compact(compact)
            rows, columns = _runs(inp, 0), _runs(inp, 1)
            _require(np.array_equal(_expand(compact, rows, columns), inp), "lossy compression")
            _require(np.array_equal(_compress(compact), compact), "noncanonical latent")
            _require(inp.shape != compact.shape, "missing parent B")
            _require(out.shape == compact.shape, "output must stay compact")
            # Check each cell independently of the painter's assignment loop.
            selected = set(data["arcs"][data["winner"]])
            bg, connector, ordinary, endpoint = data["roles"]
            for p in np.ndindex(compact.shape):
                expected = (3 if compact[p] == ordinary else 5) if p in selected else compact[p]
                _require(out[p] == expected, "incorrect arc or preservation")
            _require(np.array_equal(out[compact == endpoint], compact[compact == endpoint]),
                     "endpoint changed")
            _require(np.array_equal(out[compact == bg], compact[compact == bg]), "background changed")
            _require(not np.array_equal(compact, out), "missing parent A")
            _require(3 in out and 5 in out, "both highlight roles must be demonstrated")
            _require(np.array_equal(solve(inp), out), "input-only solver mismatch")
            features[split].append(_features(pair))
    training = features["train"]
    _require(any(f["witness"] and f["station_area_conflict"] for f in training),
             "no joint station/connector-area training witness")
    _require(len({f["selected"] for f in training}) >= 2, "constant answer-count shortcut")
    _require(any(max(f["rows"]) > 1 and max(f["columns"]) > 1
                 and len(set(f["rows"])) > 1 and len(set(f["columns"])) > 1
                 for f in training), "both-axis nonuniform evidence missing")
    _require(any((max(f["rows"]) == 1) != (max(f["columns"]) == 1) for f in training),
             "single-axis evidence missing")
    for f in features["test"]:
        _require(f["stations"] > max(x["stations"] for x in training), "station-count shift")
        _require(max(f["rows"] + f["columns"]) >
                 max(max(x["rows"] + x["columns"]) for x in training), "band-width shift")


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                array = np.asarray(pair[name], dtype=np.int8)
                h.update(bytes(array.shape))
                h.update(array.tobytes())
    return h.hexdigest()


def self_test():
    """42 required seeds, all legal counts, mutations and compact diversity stats.

    Nothing is saved here. At most two puzzles plus small signature sets remain
    live. Reports contain counts/hashes, never duplicated reconstructible grids.
    """
    signatures, shapes, input_shapes, structures, roles, schedules = (set() for _ in range(6))
    stations, selected, gaps, corners = (set() for _ in range(4))
    single_axis = repeated_nonblank = witnesses = station_area_conflicts = 0
    for seed in _CHECK_SEEDS:
        puzzle = generate_puzzle(seed)
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterminism")
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                before = inp.copy()
                _require(np.array_equal(solve(inp), out) and np.array_equal(inp, before),
                         "solve mutated its input or disagreed")
                compact = _compress(inp)
                # Scalar full-vector comparison cross-checks both axis masks.
                rr = [0] + [i for i in range(1, len(inp)) if tuple(inp[i]) != tuple(inp[i - 1])]
                cc = [0] + [j for j in range(1, inp.shape[1])
                            if tuple(inp[:, j]) != tuple(inp[:, j - 1])]
                _require(np.array_equal(compact, inp[np.ix_(rr, cc)]), "compression mask")
                _require(np.array_equal(_compress(inp.T).T, compact), "axis asymmetry")
                _require(np.array_equal(_compress(inp[rr]), compact), "collapse order")
                f = _features(pair)
                shapes.add(f["shape"])
                input_shapes.add(f["input_shape"])
                roles.add(f["roles"])
                schedules.add((f["rows"], f["columns"]))
                stations.add(f["stations"])
                selected.add(f["selected"])
                gaps.add(f["gap"])
                corners.add(f["corners"])
                single_axis += (max(f["rows"]) == 1) != (max(f["columns"]) == 1)
                witnesses += f["witness"]
                station_area_conflicts += f["station_area_conflict"]
                bg = f["roles"][0]
                for axis in (0, 1):
                    vectors = compact if axis == 0 else compact.T
                    repeated_nonblank += any(np.any(vectors[a] != bg)
                        and np.array_equal(vectors[a], vectors[b])
                        for a in range(len(vectors)) for b in range(a + 2, len(vectors)))
                semantic = np.zeros(compact.shape, dtype=np.int8)
                for i, c in enumerate(f["roles"]):
                    semantic[compact == c] = i
                structures.add(_key(semantic))
    _require(len(signatures) == 42, "seed diversity")
    _require(len(shapes) >= 30 and len(input_shapes) >= 40, "shape diversity")
    _require(len(structures) >= 150 and len(schedules) >= 150, "structural diversity")
    _require(len(roles) >= 80 and len(stations) >= 9 and len(selected) >= 4, "role/count diversity")
    _require(max(corners) >= 12 and len(corners) >= 5, "cycle geometry diversity")
    _require(single_axis >= 42 and witnesses >= 42 and station_area_conflicts >= 42
             and repeated_nonblank >= 42,
             "missing conceptual witnesses")
    # Every legal explicit count and several illegal types/ranges.
    count_cases = 0
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(193 + nt * 10 + ne, nt, ne)
            _require((len(p["train"]), len(p["test"])) == (nt, ne), "explicit counts")
            count_cases += 1
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.2), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    for k in range(6, 15):
        for minimum in (1, 2):
            if 2 * minimum >= k - 2:
                continue
            g = _fallback(k, minimum)
            choices = _arc_choices(g, _cycle(g, 0), 2)
            _require(any(x[2] == minimum and x[3] for x in choices), "fallback witness")
    p = generate_puzzle(0, 2, 1)
    p["test"][0]["output"][0, 0] = 3
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    # Tie rejection on a visibly marked four-corner cycle plus two extra nodes.
    tied = np.zeros((7, 7), dtype=np.int8)
    tied[1, 1:6] = tied[5, 1:6] = 1
    tied[1:6, 1] = tied[1:6, 5] = 1
    for point in ((1, 1), (1, 5), (5, 1), (5, 5)):
        tied[point] = 2
    tied[1, 3] = tied[5, 3] = 4
    try:
        solve(tied)
    except AssertionError:
        pass
    else:
        raise AssertionError("tied arcs accepted")
    return {"seeds": 42, "pairs": 210, "legal_count_cases": count_cases,
            "distinct_puzzles": len(signatures), "latent_shapes": len(shapes),
            "input_shapes": len(input_shapes), "semantic_structures": len(structures),
            "role_assignments": len(roles), "band_schedules": len(schedules),
            "station_counts": sorted(stations), "selected_counts": sorted(selected),
            "count_gaps": sorted(gaps), "corner_counts": sorted(corners),
            "single_axis_pairs": single_axis, "pixel_conflict_pairs": witnesses,
            "station_area_conflict_pairs": station_area_conflicts,
            "repeated_nonblank_axis_cases": repeated_nonblank}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
