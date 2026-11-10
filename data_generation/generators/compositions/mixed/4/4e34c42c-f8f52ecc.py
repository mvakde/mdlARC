"""Register overlapping maze observations, then complete optimal checkpoint paths.

All solving information comes from the input. No parent module is imported.
See task_spec.md for the observation-window and unique-optimum conventions.
"""
from collections import defaultdict
from itertools import combinations
import hashlib
import json
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _neighbors(p):
    return tuple((p[0] + dr, p[1] + dc) for dr, dc in DIRS)


def _groups(points):
    remaining = set(points)
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            for q in _neighbors(stack.pop()):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _points(mask):
    return {tuple(map(int, p)) for p in np.argwhere(mask)}


def _ordered(points):
    points = set(points)
    if len(points) == 1:
        return (next(iter(points)),)
    degree = {p: sum(q in points for q in _neighbors(p)) for p in points}
    ends = sorted(p for p in points if degree[p] == 1)
    if len(ends) != 2 or any(d not in (1, 2) for d in degree.values()):
        raise ValueError("checkpoint/path cells branch or form a loop")
    path, previous = [ends[0]], None
    while path[-1] != ends[1]:
        choices = [q for q in _neighbors(path[-1]) if q in points and q != previous]
        if len(choices) != 1:
            raise ValueError("not one path")
        previous = path[-1]
        path.append(choices[0])
    if len(path) != len(points):
        raise ValueError("disconnected path")
    return tuple(path)


def _bends(path):
    directions = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return int(sum(a != b for a, b in zip(directions, directions[1:])))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _background(a):
    colors, counts = np.unique(a, return_counts=True)
    i = int(counts.argmax())
    if int(counts[i]) * 2 <= a.size:
        raise ValueError("exterior background must occupy more than half the input")
    bg = int(colors[i])
    if not all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])):
        raise ValueError("input must have a background margin")
    return bg


def _barrier(a, bg):
    candidates = []
    for color in sorted(set(map(int, np.unique(a))) - {bg}):
        mask = a == color
        if np.any(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]):
            candidates.append(color)
    if len(candidates) != 1:
        raise ValueError("exactly one color must have a solid 2 by 2 wall block")
    return candidates[0]


def _observations(a, bg):
    """Merge component bounding boxes contained in an outer wall's bounding box.

    A map window includes an outer corner of the common field. Its connected
    outer walls span its rectangle; enclosed wall islands and checkpoints are
    therefore grouped with it. Separate windows have disjoint rectangles.
    """
    components = _groups(_points(a != bg))
    boxes = []
    for part in components:
        boxes.append((min(r for r, c in part), min(c for r, c in part),
                      max(r for r, c in part) + 1, max(c for r, c in part) + 1))
    roots = []
    for box in sorted(set(boxes)):
        r, c, s, t = box
        if not any(other != box and other[0] <= r and other[1] <= c
                   and other[2] >= s and other[3] >= t for other in boxes):
            roots.append(box)
    for x, y in combinations(roots, 2):
        if max(x[0], y[0]) < min(x[2], y[2]) and max(x[1], y[1]) < min(x[3], y[3]):
            raise ValueError("observation rectangles overlap on the input canvas")
    if not 2 <= len(roots) <= 4:
        raise ValueError("expected two to four map windows")
    return [a[r:s, c:t].copy() for r, c, s, t in roots]


def _keys(a, bg, wall):
    keys = defaultdict(list)
    h, w = a.shape
    for r, c in sorted(_points((a != bg) & (a != wall))):
        if 0 < r < h - 1 and 0 < c < w - 1:
            patch = a[r-1:r+2, c-1:c+2]
            if np.any(patch == bg) and np.any(patch == wall):
                keys[tuple(map(int, patch.ravel()))].append((r, c))
    return keys


def _agree(a, b, shift):
    """Full observed foreground AND background must agree in the intersection."""
    dr, dc = shift
    r0, c0 = max(0, dr), max(0, dc)
    r1, c1 = min(a.shape[0], dr + b.shape[0]), min(a.shape[1], dc + b.shape[1])
    if r0 >= r1 or c0 >= c1:
        return True
    return np.array_equal(a[r0:r1, c0:c1], b[r0-dr:r1-dr, c0-dc:c1-dc])


def _register(a, bg, wall):
    parts = _observations(a, bg)
    keys = [_keys(p, bg, wall) for p in parts]
    edges = defaultdict(set)
    for i, j in combinations(range(len(parts)), 2):
        for key in keys[i].keys() & keys[j].keys():
            for p in keys[i][key]:
                for q in keys[j][key]:
                    delta = (p[0] - q[0], p[1] - q[1])
                    if _agree(parts[i], parts[j], delta):
                        edges[i, j].add(delta)
                        edges[j, i].add((-delta[0], -delta[1]))
    solutions, states = {}, set()

    def search(offsets):
        state = tuple(sorted(offsets.items()))
        if state in states or len(solutions) > 1:
            return
        states.add(state)
        if len(states) > 6000:
            raise ValueError("registration search exceeds this family's bound")
        if len(offsets) == len(parts):
            r0 = min(offsets[i][0] for i in offsets)
            c0 = min(offsets[i][1] for i in offsets)
            h = max(offsets[i][0] + parts[i].shape[0] for i in offsets) - r0
            w = max(offsets[i][1] + parts[i].shape[1] for i in offsets) - c0
            if max(h, w) > 30:
                return
            field = np.full((h, w), bg, dtype=np.int8)
            covered = np.zeros((h, w), dtype=bool)
            for i, (r, c) in offsets.items():
                ph, pw = parts[i].shape
                field[r-r0:r-r0+ph, c-c0:c-c0+pw] = parts[i]
                covered[r-r0:r-r0+ph, c-c0:c-c0+pw] = True
            # The views observe the complete rectangular field, with closed walls.
            if not np.all(covered):
                return
            if not all(np.all(edge == wall) for edge in
                       (field[0], field[-1], field[:, 0], field[:, -1])):
                return
            solutions[state] = field
            return
        # Try every supported frontier, not a greedy largest-overlap merge.
        for j in range(len(parts)):
            if j in offsets:
                continue
            positions = set()
            for i, (r, c) in offsets.items():
                positions.update((r + dr, c + dc) for dr, dc in edges[i, j])
            for r, c in sorted(positions):
                if all(_agree(parts[i], parts[j], (r - p[0], c - p[1]))
                       for i, p in offsets.items()):
                    search({**offsets, j: (r, c)})

    search({0: (0, 0)})
    if len(solutions) != 1:
        raise ValueError("registration is disconnected, inconsistent, or ambiguous")
    return next(iter(solutions.values())), parts


def _route_options(field, color, bg):
    """Enumerate induced simple paths in a compressed corridor graph.

    Degree-two corridor interiors are contracted only for search. Cost and bends
    are counted at pixel resolution. Every checkpoint is a graph node. Endpoints
    of an optimum must be original checkpoint cells (trim any free end otherwise).
    """
    required = _points(field == color)
    fragments = _groups(required)
    for fragment in fragments:
        _ordered(fragment)
    if len(fragments) <= 1:
        return [(0, _bends(_ordered(required)), _ordered(required))]
    free = _points((field == bg) | (field == color))
    neighbors = {p: tuple(q for q in _neighbors(p) if q in free) for p in free}
    nodes = {p for p in free if len(neighbors[p]) != 2 or p in required}
    graph = defaultdict(list)
    for p in sorted(nodes):
        for q in neighbors[p]:
            trace, previous, current = [q], p, q
            while current not in nodes:
                following = [x for x in neighbors[current] if x != previous]
                previous, current = current, following[0]
                trace.append(current)
            if current != p:
                graph[p].append(tuple(trace))
    starts = sorted(p for p in required if sum(q in required for q in _neighbors(p)) <= 1)
    solutions = {}
    visits = 0

    def walk(path, visited, remaining):
        nonlocal visits
        visits += 1
        if visits > 80000:
            raise ValueError("routing search exceeds this family's bound")
        if not remaining:
            canonical = min(path, path[::-1])
            solutions[canonical] = (len(path) - len(required), _bends(path), canonical)
            return
        for trace in graph[path[-1]]:
            extended = set(visited)
            valid = True
            for point in trace:
                if point in extended or sum(q in extended for q in neighbors[point]) != 1:
                    valid = False
                    break
                extended.add(point)
            if valid:
                walk(path + trace, extended, remaining - set(trace))

    for p in starts:
        walk((p,), {p}, required - {p})
    if not solutions:
        raise ValueError("no nonbranching checkpoint path")
    return sorted(solutions.values())


def _complete(field, bg, wall):
    out = field.copy()
    used = set()
    records = []
    for color in sorted(set(map(int, np.unique(field))) - {bg, wall}):
        options = _route_options(field, color, bg)
        best = options[0]
        if len(options) > 1 and options[1][:2] == best[:2]:
            raise ValueError("tied length-and-bend optimum")
        cells = set(best[2])
        if used & cells:
            raise ValueError("individually optimal routes are incompatible")
        used.update(cells)
        for p in cells:
            out[p] = color
        records.append({"color": color, "added": best[0], "bends": best[1],
                        "components": len(_groups(_points(field == color))),
                        "length_wins": any(x[0] > best[0] and x[1] < best[1] for x in options),
                        "bend_wins": any(x[0] == best[0] and x[1] > best[1] for x in options),
                        "different_lengths": len({x[0] for x in options}) > 1,
                        "hidden_turns": int(sum(field[best[2][i]] == bg
                            for i in range(1, len(best[2])-1)
                            if (best[2][i][0]-best[2][i-1][0], best[2][i][1]-best[2][i-1][1])
                            != (best[2][i+1][0]-best[2][i][0], best[2][i+1][1]-best[2][i][1])))})
    return out, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    bg = _background(a)
    wall = _barrier(a, bg)
    field, _ = _register(a, bg, wall)
    # Closed outer walls are original foreground and make this already tight.
    return _complete(field, bg, wall)[0]


def _line(a, b):
    dr = int(np.sign(b[0] - a[0]))
    dc = int(np.sign(b[1] - a[1]))
    if dr and dc:
        raise ValueError("nonorthogonal edge")
    return [(a[0] + k*dr, a[1] + k*dc)
            for k in range(abs(b[0]-a[0]) + abs(b[1]-a[1]) + 1)]


def _maze(rng, bg, wall, large):
    nr = int(rng.integers(3, 5))
    nc = int(rng.integers(3, 6 if large else 5))
    rows = [2]
    cols = [2]
    for _ in range(nr - 1):
        rows.append(rows[-1] + int(rng.integers(2, 4)))
    for _ in range(nc - 1):
        cols.append(cols[-1] + int(rng.integers(2, 4)))
    vertices = [(r, c) for r in rows for c in cols]
    edges = []
    for i in range(nr):
        for j in range(nc):
            if i + 1 < nr:
                edges.append(((rows[i], cols[j]), (rows[i+1], cols[j])))
            if j + 1 < nc:
                edges.append(((rows[i], cols[j]), (rows[i], cols[j+1])))
    rng.shuffle(edges)
    parent = {p: p for p in vertices}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    selected, rejected = [], []
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            selected.append((a, b))
        else:
            rejected.append((a, b))
    selected.extend(rejected[:int(rng.integers(2, min(5, len(rejected)) + 1))])
    field = np.full((rows[-1] + 3, cols[-1] + 3), wall, dtype=np.int8)
    for a, b in selected:
        for p in _line(a, b):
            field[p] = bg
    return field


def _sample_route(rng, free):
    """Random self-avoiding visible-mask source; not the solving algorithm."""
    start = sorted(free)[int(rng.integers(len(free)))]
    path, visited = [start], {start}
    desired = int(rng.integers(7, 23))
    while len(path) < desired:
        options = [q for q in _neighbors(path[-1]) if q in free and q not in visited
                   and sum(x in visited for x in _neighbors(q)) == 1]
        if not options:
            break
        q = options[int(rng.integers(len(options)))]
        path.append(q)
        visited.add(q)
    return path if len(path) >= 5 else None


def _checkpoint_field(rng, bg, wall, colors, active, large):
    field = _maze(rng, bg, wall, large)
    available = _points(field == bg)
    for color in colors[:active]:
        path = _sample_route(rng, available)
        if path is None:
            raise ValueError("insufficient disjoint route room")
        # Reserve the whole sampled path while placing other colors. The final
        # pixel-derived optimum need not equal this random walk.
        available -= set(path)
        front, back = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        if front + back >= len(path) - 1:
            front = back = 1
        marked = set(range(front)) | set(range(len(path)-back, len(path)))
        if len(path) >= 9 and rng.random() < .55:
            marked.add(int(rng.integers(front+1, len(path)-back-1)))
        for i in marked:
            field[path[i]] = color
    if available and rng.random() < .7:
        inert = colors[active]
        p = sorted(available)[int(rng.integers(len(available)))]
        field[p] = inert
        if rng.random() < .35:
            choices = [q for q in _neighbors(p) if q in available]
            if choices:
                field[choices[int(rng.integers(len(choices)))]] = inert
    if rng.random() < .5:
        field = field.T.copy()
    if rng.random() < .5:
        field = field[::-1].copy()
    if rng.random() < .5:
        field = field[:, ::-1].copy()
    return field


def _split_views(field, bg, wall, count, rng):
    """Overlapping rectangular observations, each retaining an outer corner."""
    boxes = [(0, 0, *field.shape)]
    active = [c for c in sorted(set(map(int, np.unique(field))) - {bg, wall})
              if len(_groups(_points(field == c))) > 1]
    checkpoint_counts = {c: int(np.count_nonzero(field == c)) for c in active}
    for _ in range(count - 1):
        proposals = []
        for i, (r0, c0, r1, c1) in enumerate(boxes):
            for axis in (0, 1):
                lo, hi = (r0, r1) if axis == 0 else (c0, c1)
                for width in (3, 4, 5):
                    for start in range(lo + 2, hi - width - 1):
                        if axis == 0:
                            a, b = (r0, c0, start+width, c1), (start, c0, r1, c1)
                        else:
                            a, b = (r0, c0, r1, start+width), (r0, start, r1, c1)
                        if count >= 3 and len(boxes) >= 2:
                            # T / four-corner windows rather than many long strips.
                            if not ((a[0] == 0 or a[2] == field.shape[0])
                                    and (a[1] == 0 or a[3] == field.shape[1])
                                    and (b[0] == 0 or b[2] == field.shape[0])
                                    and (b[1] == 0 or b[3] == field.shape[1])):
                                continue
                        pa = field[a[0]:a[2], a[1]:a[3]]
                        pb = field[b[0]:b[2], b[1]:b[3]]
                        if not (_keys(pa, bg, wall).keys() & _keys(pb, bg, wall).keys()):
                            continue
                        if len(boxes) == 1 and not any(
                                max(int(np.count_nonzero(pa == c)), int(np.count_nonzero(pb == c)))
                                < checkpoint_counts[c] for c in active):
                            continue
                        proposals.append((i, a, b))
        if not proposals:
            raise ValueError("no supported observation cut")
        i, a, b = proposals[int(rng.integers(len(proposals)))]
        boxes[i:i+1] = [a, b]
    return [field[r:s, c:t].copy() for r, c, s, t in boxes]


def _spread_colors(field, parts, bg, wall):
    """Active checkpoint sets that cannot be read from any one observation."""
    return [color for color in sorted(set(map(int, np.unique(field))) - {bg, wall})
            if len(_groups(_points(field == color))) > 1
            and max(int(np.count_nonzero(part == color)) for part in parts)
            < int(np.count_nonzero(field == color))]


def _pack(parts, bg, rng):
    total = sum(int(np.count_nonzero(p != bg)) for p in parts)
    maxh, maxw = max(p.shape[0] for p in parts), max(p.shape[1] for p in parts)
    canvases = [(int(rng.integers(maxh+2, 31)), int(rng.integers(maxw+2, 31)))
                for _ in range(10)] + [(30, 30)] * 10
    for h, w in canvases:
        if h * w <= 2 * total:
            continue
        occupied = np.zeros((h, w), dtype=bool)
        out = np.full((h, w), bg, dtype=np.int8)
        success = True
        for i in sorted(range(len(parts)), key=lambda k: -parts[k].size + float(rng.random()) * 20):
            ph, pw = parts[i].shape
            positions = [(r, c) for r in range(1, h-ph) for c in range(1, w-pw)
                         if not occupied[r:r+ph, c:c+pw].any()]
            if not positions:
                success = False
                break
            r, c = positions[int(rng.integers(len(positions)))]
            out[r:r+ph, c:c+pw] = parts[i]
            occupied[r-1:r+ph+1, c-1:c+pw+1] = True
        if success:
            return out
    raise ValueError("map windows do not fit")


def _make_pair(rng, palette, index, test):
    bg, wall, *colors = palette
    rng.shuffle(colors)
    # Mandatory demonstrations: length takes precedence, then bends break ties.
    witness = "length_wins" if not test and index == 0 else "bend_wins" if not test and index == 1 else None
    for _ in range(1200):
        active = 1 if witness == "length_wins" else 2 if witness == "bend_wins" else int(rng.integers(1, 4 if test else 3))
        try:
            field = _checkpoint_field(rng, bg, wall, colors, active, test)
            out, records = _complete(field, bg, wall)
            if sum(x["added"] > 0 for x in records) != active:
                continue
            if witness and not any(x[witness] for x in records):
                continue
            count = 2 if witness == "length_wins" else 3 if witness == "bend_wins" else int(rng.integers(3, 5) if test else rng.integers(2, 4))
            parts = _split_views(field, bg, wall, count, rng)
            if not _spread_colors(field, parts, bg, wall):
                continue
            inp = _pack(parts, bg, rng)
            recovered, extracted = _register(inp, bg, wall)
            if not np.array_equal(recovered, field) or len(extracted) != count:
                continue
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise RuntimeError("could not construct a unique registered routing example")


def _count(value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError("illegal example count")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    result, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            for _ in range(8):
                try:
                    pair = _make_pair(rng, palette, i, split == "test")
                except RuntimeError as error:
                    raise RuntimeError(f"seed {seed}, {split} example {i + 1}: {error}") from error
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("duplicate input budget exhausted")
    return result


def _analyze(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    bg = _background(inp)
    wall = _barrier(inp, bg)
    field, parts = _register(inp, bg, wall)
    expected, records = _complete(field, bg, wall)
    assert np.array_equal(expected, out), "incorrect registration, routing, or crop"
    assert np.array_equal(out[field != bg], field[field != bg]), "original cells changed"
    assert np.array_equal(out == wall, field == wall), "walls changed"
    assert any(x["added"] for x in records), "no completion"
    spread = _spread_colors(field, parts, bg, wall)
    assert spread, "registration does not combine an active checkpoint set"
    assert sum(np.count_nonzero(p != bg) for p in parts) > np.count_nonzero(field != bg), "no repeated material"
    assert set(map(int, np.unique(inp))) - {bg} == set(map(int, np.unique(out))) - {bg}
    for record in records:
        color = record["color"]
        _ordered(_points(out == color))
        if record["components"] == 1:
            assert np.array_equal(field == color, out == color), "inert checkpoint extended"
    return {"background": bg, "barrier": wall, "windows": len(parts), "records": records,
            "spread_colors": spread,
            "field": field, "active": sum(x["added"] > 0 for x in records)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a = _grid(pair["input"])
            key = (a.shape, a.tobytes())
            assert key not in seen, "duplicate inputs"
            seen.add(key)
            infos[split].append(_analyze(pair))
    all_infos = infos["train"] + infos["test"]
    assert len({(x["background"], x["barrier"]) for x in all_infos}) == 1
    assert any(r["length_wins"] for x in infos["train"] for r in x["records"]), "length priority not demonstrated"
    assert any(r["bend_wins"] for x in infos["train"] for r in x["records"]), "bend tie-break not demonstrated"
    assert any(x["active"] >= 2 for x in infos["train"]), "multiple routes not demonstrated"


def self_test():
    shapes, outputs, windows, bends, active, palettes, fingerprints = set(), set(), set(), set(), set(), set(), set()
    hidden = inert = length = bend = 0
    sample = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed == 0:
            sample = puzzle["train"][0]
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair)
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                shapes.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                windows.add(info["windows"])
                active.add(info["active"])
                palettes.add((info["background"], info["barrier"]))
                for r in info["records"]:
                    bends.add(r["bends"])
                    hidden += r["hidden_turns"]
                    inert += r["components"] == 1
                    length += r["length_wins"]
                    bend += r["bend_wins"]
                digest.update(pair["input"].tobytes())
                digest.update(pair["output"].tobytes())
        fingerprints.add(digest.digest())
    assert len(fingerprints) == 42
    assert len(shapes) >= 12 and len(outputs) >= 15 and len(palettes) >= 20
    assert windows == {2, 3, 4} and {1, 2, 3} <= active
    assert len(bends) >= 5 and hidden > 30 and inert > 10
    assert length >= 42 and bend >= 42
    # These changes preserve the rule while breaking input-position and literal
    # palette shortcuts. They do not rely on any generation coordinates.
    for transform in (lambda a: a.T.copy(), lambda a: a[::-1].copy(),
                      lambda a: a[:, ::-1].copy()):
        assert np.array_equal(solve(transform(sample["input"])), transform(sample["output"]))
    mapping = np.array([3, 8, 5, 0, 7, 2, 9, 1, 6, 4], dtype=np.int8)
    assert np.array_equal(solve(mapping[sample["input"]]), mapping[sample["output"]])
    bg = _background(sample["input"])
    wall = _barrier(sample["input"], bg)
    _, parts = _register(sample["input"], bg, wall)
    repacked = _pack(parts, bg, np.random.default_rng(142))
    assert np.array_equal(solve(repacked), sample["output"])
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    bg = _background(bad["train"][0]["input"])
    wall = _barrier(bad["train"][0]["input"], bg)
    field, _ = _register(bad["train"][0]["input"], bg, wall)
    bad["train"][0]["output"] = field  # Omit B while keeping A correct.
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("registration-only answer accepted")
    tiny = np.ones((5, 5), dtype=np.int8)
    tiny[1:4, 1:4] = 0
    tiny[1, 1] = tiny[3, 3] = 2
    try:
        _complete(tiny, 0, 1)
    except ValueError as error:
        assert "tied" in str(error)
    else:
        raise AssertionError("symmetric tied routes accepted")
    tiny[tiny == 2] = 0
    tiny[2, 1:3] = 2
    assert np.array_equal(_complete(tiny, 0, 1)[0], tiny), "connected fragment extended"
    tiny[2, 2] = 0
    assert np.array_equal(_complete(tiny, 0, 1)[0], tiny), "singleton extended"
    a = generate_puzzle(99, 2, 1)
    b = generate_puzzle(99, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print(json.dumps({"self_test": "passed", "seeds": 42}, separators=(",", ":")))
