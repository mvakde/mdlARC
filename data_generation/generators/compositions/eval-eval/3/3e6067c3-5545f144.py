"""Color itineraries through silhouette-sensitive docking stations.

No parent module, generated example, seed, or construction metadata is used by
solve. All eight arrow geometries below are sampling choices; the decoder reads
and rotates the actual input cells without identifying a motif from this list.
"""
from __future__ import annotations

from copy import deepcopy
import json

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _rotate(point, turns):
    r, c = point
    for _ in range(turns % 4):
        r, c = c, -r
    return int(r), int(c)


def _arrow(span, extension, solid, tail):
    cells = {(0, -j) for j in range(extension)}
    for j in range(1, span + 1):
        rows = range(-j, j + 1) if solid else (-j, j)
        cells.update((r, 1 - extension - j) for r in rows)
    for j in range(1, tail + 1):
        cells.update((r, 1 - extension - span - j) for r in (-span, span))
    return tuple(sorted(cells))


MASKS = tuple(dict.fromkeys(
    _arrow(span, extension, solid, tail)
    for span in (1, 2) for extension in (1, 2)
    for solid in (False, True) for tail in (0, 1)
    if span + extension - 1 + tail <= 2
))


def _cells(mask, tip, direction):
    return {(tip[0] + r, tip[1] + c)
            for r, c in (_rotate(p, direction - 1) for p in mask)}


def _identify(cells):
    """Infer the symmetry axis and the single projecting tip from pixels."""
    matches = []
    for tip in sorted(cells):
        for direction in range(4):
            offsets = {_rotate((r - tip[0], c - tip[1]), 1 - direction)
                       for r, c in cells}
            if (any(c > 0 for r, c in offsets)
                    or {p for p in offsets if p[1] == 0} != {(0, 0)}
                    or max(abs(r) for r, c in offsets) == 0
                    or {(-r, c) for r, c in offsets} != offsets):
                continue
            matches.append((tuple(sorted(offsets)), tip, direction))
    assert len(matches) == 1, "socket/mover must have one geometric pointing tip"
    return matches[0]


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    components = []
    while remaining:
        start = remaining.pop()
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        components.append(component)
    return components


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    assert (dr == 0) != (dc == 0)
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _band(a, b, size):
    """Tip-aligned, one-cell-wide open gap, excluding both station squares."""
    radius = size // 2
    if a[0] == b[0] and abs(a[1] - b[1]) > size:
        return {(a[0] + radius, x)
                for x in range(min(a[1], b[1]) + size, max(a[1], b[1]))}
    if a[1] == b[1] and abs(a[0] - b[0]) > size:
        return {(y, a[1] + radius)
                for y in range(min(a[0], b[0]) + size, max(a[0], b[0]))}
    return set()


def _parse(grid):
    bg = int(grid[0, 0])
    assert all(np.all(e == bg) for e in
               (grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    stations, tokens, size = [], [], None
    for component in _components(grid != bg):
        if len(component) == 1:
            tokens.append(next(iter(component)))
            continue
        ys, xs = zip(*component)
        y, x = int(min(ys)), int(min(xs))
        s = int(max(ys) - y + 1)
        assert s >= 5 and s % 2 and max(xs) - x + 1 == s
        assert size is None or size == s
        size = s
        block = grid[y:y+s, x:x+s]
        label = int(block[0, 0])
        assert label != bg
        assert all(np.all(e == label) for e in
                   (block[0], block[-1], block[:, 0], block[:, -1]))
        holes = {(y + int(r), x + int(c)) for r, c in zip(*np.where(block != label))}
        assert len(holes) >= 3
        mask, tip, facing = _identify(holes)
        assert tip == (y + s//2, x + s//2), "all noses are square-centered"
        hole_colors = {int(grid[q]) for q in holes}
        assert len(hole_colors) == 1
        stations.append(dict(y=y, x=x, label=label, holes=holes,
                             hole_color=hole_colors.pop(), mask=mask,
                             tip=tip, facing=facing))
    stations.sort(key=lambda s: (s["y"], s["x"]))
    assert 3 <= len(stations) and 3 <= len(tokens) <= len(stations)
    tokens.sort(key=lambda q: q[1])
    key_y = tokens[0][0]
    assert all(y == key_y for y, x in tokens)
    assert all(b[1] - a[1] == 2 for a, b in zip(tokens, tokens[1:]))
    assert key_y > max(s["y"] + size for s in stations)
    key = [int(grid[q]) for q in tokens]
    active = [i for i, s in enumerate(stations) if s["hole_color"] != bg]
    assert len(active) == 1, "one filled socket identifies the initial courier"
    start = active[0]
    ink = stations[start]["hole_color"]
    labels = {s["label"] for s in stations}
    assert ink not in labels and bg not in labels
    assert set(key) <= labels and key[0] == stations[start]["label"]
    assert len(set(key)) >= 2
    graph, bands = [[] for _ in stations], {}
    for i, a in enumerate(stations):
        for j, b in enumerate(stations):
            if i == j:
                continue
            band = _band((a["y"], a["x"]), (b["y"], b["x"]), size)
            if band and all(grid[q] == bg for q in band):
                graph[i].append(j)
                bands[i, j] = band
    return dict(bg=bg, ink=ink, size=size, stations=stations, key=key,
                start=start, mask=stations[start]["mask"],
                initial=stations[start]["facing"], graph=graph, bands=bands,
                key_y=int(key_y), key_x=int(tokens[0][1]))


def _routes(info, *, use_key=True, use_shape=True, last_only=False, limit=2):
    """Enumerate full routes, with optional concept ablations for validation."""
    ss, key = info["stations"], info["key"]
    solutions = []

    def visit(path, painted):
        if len(solutions) >= limit:
            return
        if len(path) == len(key):
            solutions.append(tuple(path))
            return
        i = path[-1]
        for j in info["graph"][i]:
            check_color = use_key and (not last_only or len(path) == len(key)-1)
            if j in path or (check_color and ss[j]["label"] != key[len(path)]):
                continue
            d = _direction(ss[i]["tip"], ss[j]["tip"])
            # Initial travel follows the visible arrow in all ablations too.
            if len(path) == 1 and d != info["initial"]:
                continue
            if use_shape and _cells(info["mask"], ss[j]["tip"], d) != ss[j]["holes"]:
                continue
            band = info["bands"][i, j]
            if painted & band:
                continue
            visit(path + [j], painted | band)

    visit([info["start"]], set())
    return solutions


def _render_solution(grid, info, route):
    out = grid.copy()
    ss = info["stations"]
    for q in ss[info["start"]]["holes"]:
        out[q] = info["bg"]
    for i, j in zip(route, route[1:]):
        for q in info["bands"][i, j]:
            out[q] = ss[i]["label"]
    d = _direction(ss[route[-2]]["tip"], ss[route[-1]]["tip"])
    for q in _cells(info["mask"], ss[route[-1]]["tip"], d):
        out[q] = info["ink"]
    return out


def _analyze(grid):
    info = _parse(grid)
    routes = _routes(info)
    assert len(routes) == 1, "key plus rigid socket fit must define one full route"
    info["route"] = routes[0]
    return _render_solution(grid, info, routes[0]), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the supplied pixels; do not mutate the input."""
    return _analyze(_grid(input_grid))[0]


def _geometry_graph(cells):
    graph = [[] for _ in cells]
    for i, a in enumerate(cells):
        for j, b in enumerate(cells):
            if i == j:
                continue
            if a[0] == b[0]:
                clear = not any(c[0] == a[0] and min(a[1], b[1]) < c[1] < max(a[1], b[1])
                                for c in cells)
            elif a[1] == b[1]:
                clear = not any(c[1] == a[1] and min(a[0], b[0]) < c[0] < max(a[0], b[0])
                                for c in cells)
            else:
                continue
            if clear:
                graph[i].append(j)
    return graph


def _branch_plans(graph, path, deep):
    free = [j for j in graph[path[-2]] if j not in path]
    if not deep:
        return [(a, b) for a in free for b in free if a != b]
    # A two-hop false branch diverges one step earlier, but ends on the SAME
    # terminal key color. It defeats endpoint-color-only route decoding.
    return [(a, b, c) for a in free for b in graph[path[-3]] if b not in path and b != a
            for c in graph[b] if c not in path and c not in (a, b)]


def _path(rng, cells, graph, length, traps, need_turn=False, deep=False):
    budget = 3500

    def visit(path, used_bands):
        nonlocal budget
        budget -= 1
        if budget <= 0:
            return None
        if len(path) == length:
            # Counterfactual branches guarantee complete alternative routes
            # when either inherited mechanism is removed.
            if traps and not _branch_plans(graph, path, deep):
                return None
            if need_turn and len({_direction(cells[i], cells[j])
                                  for i, j in zip(path, path[1:])}) < 2:
                return None
            return path
        for j in rng.permutation(graph[path[-1]]):
            j = int(j)
            if j in path:
                continue
            # Coordinates in the abstract layout are spaced by ten solely for
            # detecting crossings; real independent axis spacings come later.
            a, b = cells[path[-1]], cells[j]
            band = _band((a[0]*10, a[1]*10), (b[0]*10, b[1]*10), 1)
            if band & used_bands:
                continue
            found = visit(path + [j], used_bands | band)
            if found is not None:
                return found
        return None

    for i in rng.permutation(len(cells)):
        result = visit([int(i)], set())
        if result is not None:
            return result
        if budget <= 0:
            break
    return None


def _coordinates(rng, count, size, limit):
    gaps = np.ones(count - 1, dtype=int)
    spare = limit - count*size - len(gaps)
    assert spare >= 0
    # Independently distributed positive gaps, including skipped empty levels.
    for i in rng.permutation(len(gaps)):
        extra = int(rng.integers(0, min(6, spare) + 1))
        gaps[i] += extra
        spare -= extra
    return [0] + [int(x) for x in np.cumsum(size + gaps)]


def _construct(rng, size, mask, length, traps, stable_palette=None, need_turn=False, deep=False):
    max_levels = (26 + 1)//(size + 1)
    options = [(r, c) for r in range(2, max_levels + 1)
               for c in range(2, (28 + 1)//(size + 1) + 1)
               if r*c >= length + (3 if deep else 2 if traps else 0)
               and (not traps or (r >= 3 and c >= 3))]
    assert options
    found = None
    for _ in range(70):
        rows, cols = options[int(rng.integers(len(options)))]
        universe = [(r, c) for r in range(rows) for c in range(cols)]
        low = length + (3 if deep else 2 if traps else 0)
        high = len(universe) if rng.random() < .20 else min(len(universe), low + 3)
        n = int(rng.integers(low, high + 1))
        cells = sorted(universe[int(j)] for j in rng.choice(len(universe), n, replace=False))
        graph = _geometry_graph(cells)
        path = _path(rng, cells, graph, length, traps, need_turn, deep)
        if path is not None:
            found = (rows, cols, cells, graph, path)
            break
    if found is None:
        # Full sampled lattice guarantees an admissible path for our supported
        # lengths. It is still searched; no stored route or output template.
        rows = cols = 3 if size == 7 else 4
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        graph = _geometry_graph(cells)
        path = _path(rng, cells, graph, length, traps, need_turn, deep)
        assert path is not None
    else:
        rows, cols, cells, graph, path = found
    # Remove unoccupied outside levels while preserving internal empty levels.
    rmin, cmin = min(r for r, c in cells), min(c for r, c in cells)
    cells = [(r-rmin, c-cmin) for r, c in cells]
    rows, cols = max(r for r, c in cells)+1, max(c for r, c in cells)+1
    ys = _coordinates(rng, rows, size, 26)
    xs = _coordinates(rng, cols, size, 28)
    bh, bw = ys[-1] + size, xs[-1] + size
    h = int(rng.integers(bh + 4, 31))
    w = int(rng.integers(max(bw + 2, 2*length + 1), 31))
    yoff = int(rng.integers(1, h - bh - 2))
    xoff = int(rng.integers(1, w - bw))
    positions = [(ys[r] + yoff, xs[c] + xoff) for r, c in cells]
    key_y = int(rng.integers(yoff + bh + 1, h - 1))
    key_x = int(rng.integers(1, w - (2*length - 1)))
    if stable_palette is None:
        bg, ink = (int(x) for x in rng.choice(10, 2, replace=False))
    else:
        bg, ink = stable_palette
    pool = [int(x) for x in rng.permutation([c for c in range(10) if c not in (bg, ink)])]
    k = int(rng.integers(3, min(8, length+2) + 1))
    pool = pool[:k]
    feasible_masks = [m for m in MASKS if max(max(abs(r), abs(c)) for r, c in m) < size//2]
    others = [m for m in feasible_masks if m != mask]
    # Assign geometry, labels, and the two counterfactual branches jointly.
    for _ in range(90):
        labels = [int(rng.choice(pool)) for _ in cells]
        for a, b in zip(path, path[1:]):
            # At least the first departure differs from the next station color.
            if a == path[0] and labels[a] == labels[b]:
                labels[b] = next(c for c in pool if c != labels[a])
        key = [labels[i] for i in path]
        if len(set(key)) < 2:
            continue
        masks = [feasible_masks[int(rng.integers(len(feasible_masks)))] for _ in cells]
        directions = [int(rng.integers(4)) for _ in cells]
        directions[path[0]] = _direction(cells[path[0]], cells[path[1]])
        masks[path[0]] = mask
        for i, j in zip(path, path[1:]):
            masks[j] = mask
            directions[j] = _direction(cells[i], cells[j])
        if traps:
            plans = _branch_plans(graph, path, deep)
            plan = plans[int(rng.integers(len(plans)))]
            wrong_shape, wrong_color = plan[:2]
            labels[wrong_shape] = labels[path[-1]]
            masks[wrong_shape] = others[int(rng.integers(len(others)))]
            directions[wrong_shape] = _direction(cells[path[-2]], cells[wrong_shape])
            labels[wrong_color] = int(rng.choice([c for c in pool if c != key[-2 if deep else -1]]))
            masks[wrong_color] = mask
            directions[wrong_color] = _direction(cells[path[-3 if deep else -2]], cells[wrong_color])
            if deep:
                same_end_color = plan[2]
                labels[same_end_color] = key[-1]
                masks[same_end_color] = mask
                directions[same_end_color] = _direction(cells[wrong_color], cells[same_end_color])
        grid = np.full((h, w), bg, dtype=np.int8)
        for j, ((y, x), label) in enumerate(zip(positions, labels)):
            grid[y:y+size, x:x+size] = label
            tip = (y+size//2, x+size//2)
            for q in _cells(masks[j], tip, directions[j]):
                grid[q] = ink if j == path[0] else bg
        grid[key_y, key_x:key_x+2*length:2] = key
        # Build labels directly from construction, then decode visible pixels
        # afresh to check the intended path and all alternative assignments.
        out = grid.copy()
        start_tip = tuple(p + size//2 for p in positions[path[0]])
        for q in _cells(mask, start_tip, directions[path[0]]):
            out[q] = bg
        for i, j in zip(path, path[1:]):
            for q in _band(positions[i], positions[j], size):
                out[q] = labels[i]
        end_tip = tuple(p + size//2 for p in positions[path[-1]])
        for q in _cells(mask, end_tip, directions[path[-1]]):
            out[q] = ink
        info = _parse(grid)
        routes = _routes(info)
        if len(routes) != 1 or routes[0] != tuple(path):
            continue
        if not np.array_equal(out, _render_solution(grid, info, routes[0])):
            continue
        if traps and (len(_routes(info, use_key=False)) < 2
                      or len(_routes(info, use_shape=False)) < 2):
            continue
        if deep and len(_routes(info, last_only=True)) < 2:
            continue
        return dict(input=grid, output=out)
    return None


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _signature(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    stable = tuple(int(c) for c in rng.choice(10, 2, replace=False)) if rng.random() < .20 else None
    scale_shift = bool(rng.random() < .30)
    demo_size = 5 if scale_shift else int(rng.choice((5, 7)))
    fitting = [m for m in MASKS if max(max(abs(r), abs(c)) for r, c in m) < demo_size//2]
    demo_masks = [fitting[int(i)] for i in rng.choice(len(fitting), 2, replace=False)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            # Both anchors have full-route decoys for both concepts. The second
            # also defeats decoding with only the terminal key color.
            traps = split == "test" or i < 2 or rng.random() < .70
            size = (demo_size if split == "train" and i < 2 else
                    5 if split == "train" and scale_shift else
                    7 if split == "test" and scale_shift else
                    5 if split == "test" else int(rng.choice((5, 7))))
            fitting = [m for m in MASKS if max(max(abs(r), abs(c)) for r, c in m) < size//2]
            if split == "test" and scale_shift:
                fitting = [m for m in fitting if max(max(abs(r), abs(c)) for r, c in m) == 2]
            mask = demo_masks[i] if split == "train" and i < 2 else fitting[int(rng.integers(len(fitting)))]
            # The shift is composition over longer sequences, with all locks,
            # color filtering, and rotations already demonstrated in the anchors.
            length = (int(rng.integers(3, 5)) if size == 7 else
                      int(rng.integers(6, 10)) if split == "test" else
                      int(rng.integers(3, 6)))
            deep = (split == "train" and i == 1) or (split == "test" and rng.random() < .50)
            if deep and (split == "train" or length == 3):
                length = 4
            for attempt in range(20):
                pair = _construct(rng, size, mask, length, traps, stable,
                                  need_turn=split == "train" and i == 0, deep=deep)
                if pair is not None and _signature(pair["input"]) not in seen:
                    seen.add(_signature(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded sampler failed to find a distinct docking scene")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    expected, info = _analyze(inp)
    assert np.array_equal(out, expected), "wrong route, bridge color, or rigid terminal mask"
    ss, route, ink = info["stations"], info["route"], info["ink"]
    assert np.count_nonzero(inp == ink) == np.count_nonzero(out == ink) == len(info["mask"])
    start, end, size = ss[route[0]], ss[route[-1]], info["size"]
    initial_bitmap = inp[start["y"]:start["y"]+size, start["x"]:start["x"]+size] == ink
    final_bitmap = out[end["y"]:end["y"]+size, end["x"]:end["x"]+size] == ink
    facing = _direction(ss[route[-2]]["tip"], end["tip"])
    assert np.array_equal(final_bitmap, np.rot90(initial_bitmap, -(facing-info["initial"]) % 4))
    assert not np.array_equal(inp, out)
    changed = set(zip(*np.where(inp != out)))
    bridges = set().union(*(info["bands"][i, j] for i, j in zip(route, route[1:])))
    allowed = bridges | ss[route[0]]["holes"] | ss[route[-1]]["holes"]
    assert changed == allowed, "preserve every shell, unused socket, key, and margin"
    for i, j in zip(route, route[1:]):
        assert all(out[q] == ss[i]["label"] for q in info["bands"][i, j])
    # Compare changed answers, not just route counts, for both concept ablations.
    ablations = {}
    for name, kwargs in (("no_key", {"use_key": False}), ("no_shape", {"use_shape": False})):
        alternatives = _routes(info, limit=80, **kwargs)
        ablations[name] = any(not np.array_equal(_render_solution(inp, info, r), out)
                              for r in alternatives)
    ablations["last_only"] = len(_routes(info, last_only=True)) > 1
    info.update(ablations)
    return info


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    infos, seen = {}, set()
    for split in ("train", "test"):
        infos[split] = []
        for pair in puzzle[split]:
            info = _check_pair(pair)
            sig = _signature(_grid(pair["input"]))
            assert sig not in seen
            seen.add(sig)
            infos[split].append(info)
    train = infos["train"]
    assert len({a["mask"] for a in train}) >= 2
    assert train[0]["size"] == train[1]["size"]
    assert all(a["no_key"] and a["no_shape"] for a in train[:2])
    assert train[1]["last_only"], "an earlier key token must matter beyond the final color"
    assert all(a["no_key"] and a["no_shape"] for a in infos["test"])
    assert any(len({_direction(a["stations"][i]["tip"], a["stations"][j]["tip"])
                    for i, j in zip(a["route"], a["route"][1:])}) > 1 for a in train)
    assert all(any(a["stations"][i]["label"] != a["stations"][j]["label"]
                   for i, j in zip(a["route"], a["route"][1:])) for a in train)


def measure_pair(pair):
    """Compact, input-derived variation evidence for audit scripts."""
    a = _check_pair(pair)
    ss, route = a["stations"], a["route"]
    points = [ss[i]["tip"] for i in route]
    ds = [_direction(u, v) for u, v in zip(points, points[1:])]
    gaps = [len(a["bands"][i, j]) for i, j in zip(route, route[1:])]
    ys, xs = {s["y"] for s in ss}, {s["x"] for s in ss}
    return dict(shape=list(np.asarray(pair["input"]).shape), size=a["size"], mask=a["mask"],
                stations=len(ss), length=len(route), unused=len(ss)-len(route),
                layout=tuple((s["y"]-min(ys), s["x"]-min(xs)) for s in ss),
                sparse=len(ss) < len(ys)*len(xs), gaps=gaps,
                turns=sum(x != y for x, y in zip(ds, ds[1:])),
                initial=ds[0], final=ds[-1], directions=sorted(set(ds)),
                key=a["key"], roles=(a["bg"], a["ink"]),
                no_key=a["no_key"], no_shape=a["no_shape"], last_only=a["last_only"])


def _puzzle_signature(puzzle):
    return tuple((_signature(_grid(p["input"])), _signature(_grid(p["output"])))
                 for s in ("train", "test") for p in puzzle[s])


def self_test() -> dict:
    """Bounded 42-seed semantic/diversity checks and targeted negative tests."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    metrics = [dict(split=s, **measure_pair(pair)) for p in puzzles
               for s in ("train", "test") for pair in p[s]]
    assert len({_puzzle_signature(p) for p in puzzles}) == 42
    assert _puzzle_signature(puzzles[0]) == _puzzle_signature(generate_puzzle(0))
    assert len({tuple(m["shape"]) for m in metrics}) >= 20
    assert len({m["layout"] for m in metrics}) >= 80
    assert {m["size"] for m in metrics} == {5, 7}
    assert len({m["mask"] for m in metrics}) == len(MASKS)
    assert {m["initial"] for m in metrics} == {0, 1, 2, 3}
    assert {m["final"] for m in metrics} == {0, 1, 2, 3}
    assert len({m["length"] for m in metrics}) >= 5
    assert any(m["sparse"] for m in metrics) and any(not m["sparse"] for m in metrics)
    assert any(len(set(m["gaps"])) > 1 for m in metrics)
    assert any(m["key"][0] in m["key"][1:] for m in metrics)
    assert any(x == y for m in metrics for x, y in zip(m["key"], m["key"][1:]))
    assert any(m["last_only"] for m in metrics if m["split"] == "test")
    assert len({m["roles"] for m in metrics}) >= 30
    # Legal count boundaries; verify.py exhausts the entire legal Cartesian set.
    for nt, nq in ((2, 1), (10, 1), (10, 4), (6, 3)):
        validate_generated_puzzle(generate_puzzle(1000 + nt*10 + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Pixel-level equivariance: solving uses equality and geometry, not literal
    # colors or an orientation-specific route table. Keep the key horizontal.
    pair = puzzles[0]["train"][0]
    mapping = (3*np.arange(10) + 4) % 10
    assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
    inp = pair["input"].copy()
    solve(inp)
    assert np.array_equal(inp, pair["input"])
    # A qualifying mask absent from the sampling catalog must also solve. Replace
    # every socket of one actual-mask identity, preserving the compatibility graph.
    novel = tuple(sorted({(0, 0), (-1, -1), (1, -1), (-2, -2),
                          (2, -2), (0, -1), (0, -2)}))
    assert novel not in MASKS
    wide_pair = next(pair for p in puzzles for pair in p["train"]
                     if _parse(pair["input"])["size"] == 7)
    _, wide = _analyze(wide_pair["input"])
    new_input = wide_pair["input"].copy()
    for j, s in enumerate(wide["stations"]):
        if s["mask"] == wide["mask"]:
            for q in s["holes"]:
                new_input[q] = s["label"]
            for q in _cells(novel, s["tip"], s["facing"]):
                new_input[q] = wide["ink"] if j == wide["start"] else wide["bg"]
    new_output, new_info = _analyze(new_input)
    assert new_info["mask"] == novel and new_info["route"] == wide["route"]
    assert np.count_nonzero(new_output == wide["ink"]) == len(novel)
    _check_pair(dict(input=new_input, output=new_output))
    broken = deepcopy(puzzles[0])
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("changed preserved cell accepted")
    # An invalid terminal silhouette must not silently substitute a canned mask.
    _, info = _analyze(pair["input"])
    terminal = info["stations"][info["route"][-1]]
    bad = pair["input"].copy()
    q = next(q for q in terminal["holes"] if q != terminal["tip"])
    bad[q] = terminal["label"]
    try:
        solve(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("broken unique terminal socket accepted")
    return dict(seeds=42, pairs=len(metrics), masks=len(MASKS),
                layouts=len({m["layout"] for m in metrics}),
                shapes=len({tuple(m["shape"]) for m in metrics}),
                lengths=sorted({m["length"] for m in metrics}),
                no_key_counterexamples=sum(m["no_key"] for m in metrics),
                no_shape_counterexamples=sum(m["no_shape"] for m in metrics),
                earlier_key_counterexamples=sum(m["last_only"] for m in metrics),
                novel_mask_regression=True)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
