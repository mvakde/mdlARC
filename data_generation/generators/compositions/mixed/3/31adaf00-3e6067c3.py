"""Key-routed maximal blank-square interiors. No filesystem or hidden solver state."""
from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _parse(a):
    bg = int(a[0, 0])
    _require(all(np.all(e == bg) for e in (a[0], a[-1], a[:, 0], a[:, -1])),
             "background frame required")
    occupied_rows = np.flatnonzero(np.any(a != bg, axis=1))
    _require(bool(len(occupied_rows)), "empty scene")
    key_y = int(occupied_rows[-1])
    xs = np.flatnonzero(a[key_y] != bg)
    _require(3 <= len(xs) <= 9 and bool(np.all(np.diff(xs) == 2)), "invalid bottom key")
    key = [int(a[key_y, x]) for x in xs]
    _require(len(set(key)) >= 2 and key_y >= 3, "invalid key labels")
    _require(np.all(a[key_y-1] == bg), "key must be detached")
    field = a[:key_y-1]
    options = []
    for carrier in sorted(set(map(int, np.unique(field))) - {bg} - set(key)):
        stations, signature = [], None
        valid = True
        for component in _components(field == carrier):
            ys, cs = zip(*component)
            y, x = min(ys), min(cs)
            size = max(ys) - y + 1
            if not (4 <= size <= 7 and max(cs)-x+1 == size):
                valid = False
                break
            block = field[y:y+size, x:x+size]
            py, px = np.nonzero(block != carrier)
            if not len(py):
                valid = False
                break
            offset, patch = int(py.min()), int(py.max()-py.min()+1)
            label = int(block[offset, offset])
            if not (offset > 0 and 2*offset+patch == size
                    and int(px.min()) == offset and int(px.max()) == offset+patch-1
                    and len(py) == patch*patch and label not in (bg, carrier)
                    and np.all(block[offset:offset+patch, offset:offset+patch] == label)):
                valid = False
                break
            if signature is not None and signature != (size, patch):
                valid = False
                break
            signature = (size, patch)
            stations.append((y, x, label))
        if valid and 3 <= len(stations) <= 12:
            options.append((carrier, signature, sorted(stations)))
    _require(len(options) == 1, "station carrier must be unique")
    carrier, (size, patch), stations = options[0]
    rebuilt = np.full(a.shape, bg, dtype=a.dtype)
    offset = (size-patch)//2
    for y, x, label in stations:
        _require(y+size < key_y, "station touches key")
        rebuilt[y:y+size, x:x+size] = carrier
        rebuilt[y+offset:y+offset+patch, x+offset:x+offset+patch] = label
    rebuilt[key_y, xs] = key
    remainder = a != rebuilt
    _require(np.all(rebuilt[remainder] == bg), "malformed station or key")
    debris = set(map(int, np.unique(a[remainder])))
    labels = {s[2] for s in stations}
    _require(len(debris) <= 1 and not debris & ({bg, carrier} | labels | set(key)),
             "obstacles must have a separate color")
    _require(not np.any(remainder[key_y-1:]), "obstacle below station field")
    return dict(background=bg, carrier=carrier, obstacle=next(iter(debris), None),
                size=size, patch=patch, stations=stations, key=key, key_y=key_y)


def _gap(a, b, size):
    """Half-open full-outer-face rectangle, excluding the endpoint stations."""
    if a[0] == b[0] and abs(a[1]-b[1]) > size:
        return (a[0], min(a[1], b[1])+size, a[0]+size, max(a[1], b[1]))
    if a[1] == b[1] and abs(a[0]-b[0]) > size:
        return (min(a[0], b[0])+size, a[1], max(a[0], b[0]), a[1]+size)
    return None


def _slice(rect):
    y, x, yy, xx = rect
    return slice(y, yy), slice(x, xx)


def _band(a, b, size, patch):
    rect = _gap(a, b, size)
    if rect is None:
        return None
    y, x, yy, xx = rect
    offset = (size-patch)//2
    return ((y+offset, x, y+offset+patch, xx) if a[0] == b[0]
            else (y, x+offset, yy, x+offset+patch))


def _overlap(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _geometry_edges(positions, size):
    edges = {}
    for i, a in enumerate(positions):
        for j in range(i+1, len(positions)):
            rect = _gap(a, positions[j], size)
            if rect is None:
                continue
            if any(k not in (i, j) and _overlap(rect, (c[0], c[1], c[0]+size, c[1]+size))
                   for k, c in enumerate(positions)):
                continue
            edges[i, j] = rect
    return edges


def _graph_from_edges(n, edges):
    graph = [[] for _ in range(n)]
    for i, j in edges:
        graph[i].append(j)
        graph[j].append(i)
    return graph


def _routes(graph, labels, key, limit=2):
    answers = []
    choices = [[i for i, color in enumerate(labels) if color == k] for k in key]
    def visit(path, used):
        if len(answers) >= limit:
            return
        if len(path) == len(key):
            answers.append(tuple(path))
            return
        for j in choices[len(path)]:
            if j not in used and (not path or j in graph[path[-1]]):
                visit(path+[j], used | {j})
    visit([], set())
    return answers


def _maximal_squares(mask):
    """Local maximality as a RECTANGLE, not just non-enlargement as a square.

    The mask boundary clips the search; cells outside the gap cannot extend it.
    Integral sums enumerate blank windows; a full adjacent side disqualifies one.
    """
    h, w = mask.shape
    summed = np.pad(mask.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    result = []
    for side in range(2, min(h, w)+1):
        counts = summed[side:, side:] - summed[:-side, side:] - summed[side:, :-side] + summed[:-side, :-side]
        for y, x in zip(*np.nonzero(counts == side*side)):
            y, x = int(y), int(x)
            if y and mask[y-1, x:x+side].all():
                continue
            if y+side < h and mask[y+side, x:x+side].all():
                continue
            if x and mask[y:y+side, x-1].all():
                continue
            if x+side < w and mask[y:y+side, x+side].all():
                continue
            result.append((y, x, side))
    return result


def _winner(squares):
    if not squares:
        return None
    side = max(q[2] for q in squares)
    best = [q for q in squares if q[2] == side]
    _require(len(best) == 1, "equal-largest squares are outside this family")
    return best[0]


def _interior_mask(shape, square):
    mask = np.zeros(shape, dtype=bool)
    if square is not None:
        y, x, side = square
        mask[y+1:y+side-1, x+1:x+side-1] = True
    return mask


def _analyze(input_grid):
    a = _grid(input_grid)
    info = _parse(a)
    stations, size, patch, bg = (info[k] for k in ("stations", "size", "patch", "background"))
    edges = _geometry_edges(stations, size)
    edges = {edge: rect for edge, rect in edges.items()
             if np.all(a[_slice(_band(stations[edge[0]], stations[edge[1]], size, patch))] == bg)}
    graph = _graph_from_edges(len(stations), edges)
    routes = _routes(graph, [s[2] for s in stations], info["key"])
    _require(len(routes) == 1, "key must specify exactly one distinct-station route")
    route = routes[0]
    chosen_edges = {tuple(sorted((i, j))) for i, j in zip(route, route[1:])}
    output = a.copy()
    painted = np.zeros(a.shape, dtype=bool)
    windows, directions = [], []
    for i, j in zip(route, route[1:]):
        rect = edges[tuple(sorted((i, j)))]
        mask = a[_slice(rect)] == bg
        candidates = _maximal_squares(mask)
        winner = _winner(candidates)
        interior = _interior_mask(mask.shape, winner)
        _require(not np.any(painted[_slice(rect)] & interior), "routed interiors overlap")
        painted[_slice(rect)] |= interior
        output[_slice(rect)][interior] = stations[i][2]
        windows.append(dict(edge=(i, j), rect=rect, squares=candidates, winner=winner))
        first, second = stations[i], stations[j]
        directions.append("R" if second[1] > first[1] else "L" if second[1] < first[1]
                          else "D" if second[0] > first[0] else "U")
    info.update(route=route, graph=graph, edges=edges, windows=windows,
                chosen_edges=chosen_edges, directions=directions, painted=painted)
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the visible key and geometry, then paint from the original input."""
    return _analyze(input_grid)[0]


def _sample_path(rng, graph, length):
    budget = 5000
    def visit(path):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            return path
        for j in rng.permutation(graph[path[-1]]):
            j = int(j)
            if j not in path:
                found = visit(path+[j])
                if found is not None:
                    return found
        return None
    for start in rng.permutation(len(graph)):
        found = visit([int(start)])
        if found is not None:
            return found
    return None


def _assign_labels(rng, graph, route, colors, repeated_start=False):
    n, length = len(graph), len(route)
    for attempt in range(160):
        pool_size = min(len(colors), length) if attempt >= 100 else int(rng.integers(2, min(len(colors), length)+1))
        pool = list(map(int, rng.permutation(colors)[:pool_size]))
        key = [int(rng.choice(pool)) for _ in route]
        # Some distinct keys coexist with repetitions; no privileged unique start.
        if attempt >= 100 or (not repeated_start and rng.random() < .35):
            key = pool + [int(rng.choice(pool)) for _ in range(length-len(pool))]
            rng.shuffle(key)
        if repeated_start:
            key[int(rng.integers(1, length-1))] = key[0]
        if len(set(key)) < 2:
            continue
        labels = [int(rng.choice(colors)) for _ in range(n)]
        for i, color in zip(route, key):
            labels[i] = color
        if _routes(graph, labels, key) == [tuple(route)]:
            return labels, key
    return None


def _has_largest_witness(mask, squares, winner):
    if winner is None or winner[2] < 3:
        return False
    selected = _interior_mask(mask.shape, winner)
    return any(q[2] >= 3 and q[2] < winner[2]
               and np.any(_interior_mask(mask.shape, q) & ~selected) for q in squares)


def _gap_mask(rng, size, length, patch, mode, allowed_max=7):
    """Canonical horizontal gap; vertical gaps use its transpose."""
    offset = (size-patch)//2
    if mode == "nonsquare":
        _require(size != length, "nonsquare teaching gap must be nonsquare")
        return np.ones((size, length), dtype=bool)
    attempts = 256 if mode == "competition" else 40
    for attempt in range(attempts):
        mask = rng.random((size, length)) < rng.uniform(.67, .95)
        mask[offset:offset+patch] = True
        if mode == "competition" and attempt % 2 == 0:
            # A varied planted scaffold accelerates the rare unequal-size case.
            # Its partial branches are checked afresh, never assumed eligible.
            mask = rng.random((size, length)) < rng.uniform(0, .38)
            mask[3] = True
            mask[0:4, 0:4] = True
            mask[3:6, length-3:length] = True
            if rng.random() < .5:
                mask = mask[::-1].copy()
            if rng.random() < .5:
                mask = mask[:, ::-1].copy()
        squares = _maximal_squares(mask)
        if not squares:
            continue
        max_side = max(q[2] for q in squares)
        if max_side > allowed_max or sum(q[2] == max_side for q in squares) != 1:
            continue
        winner = max(squares, key=lambda q: q[2])
        if mode == "competition" and not _has_largest_witness(mask, squares, winner):
            continue
        if mode == "active" and max_side < 3:
            continue
        return mask
    if mode == "competition":
        mask = np.zeros((7, length), dtype=bool)
        mask[3] = True
        mask[:4, :4] = True
        mask[3:6, length-3:length] = True
        _require(_has_largest_witness(mask, _maximal_squares(mask), _winner(_maximal_squares(mask))),
                 "competition scaffold failed")
        return mask
    # Guaranteed single square, with optional partial central-band protrusions.
    side = min(size, length, allowed_max)
    _require(side >= patch and side >= 2, "gap too narrow")
    mask = np.zeros((size, length), dtype=bool)
    mask[offset:offset+patch] = True
    tops = [r for r in range(size-side+1) if r <= offset and r+side >= offset+patch]
    y = int(rng.choice(tops))
    x = int(rng.integers(0, length-side+1))
    mask[y:y+side, x:x+side] = True
    squares = _maximal_squares(mask)
    winner = _winner(squares)
    _require(winner is not None and (mode != "active" or winner[2] >= 3), "gap fallback failed")
    return mask


def _coordinates(rng, count, size, limit, min_gap=3, max_gap=8):
    gaps = [min_gap]*(count-1)
    spare = limit-count*size-sum(gaps)
    if spare < 0:
        return None
    for k in rng.permutation(len(gaps)):
        extra = int(rng.integers(0, min(max_gap-min_gap, spare)+1))
        gaps[int(k)] += extra
        spare -= extra
    return [0] + list(map(int, np.cumsum([size+g for g in gaps])))


def _layout(rng, anchor, test):
    for attempt in range(64):
        if anchor == 0:
            size, patch, rows, cols, n, length = 7, 1, 2, 2, 4, 3
            ys = [0, size+int(rng.integers(6, 10))]
            xs = [0, size+int(rng.integers(6, 10))]
        elif anchor == 1:
            size = int(rng.choice((5, 6)))
            patch = 3 if size == 5 else 2
            rows, cols, n, length = 2, 3, int(rng.integers(5, 7)), 5
            ys = _coordinates(rng, rows, size, 26, 3, 4)
            xs = _coordinates(rng, cols, size, 28, 3, 4)
        else:
            size = int(rng.choice((4, 5, 5, 6, 7)))
            patch = int(rng.choice(list(range(1 if size % 2 else 2, min(4, size-1), 2))))
            length = int(rng.integers(6, 10)) if test else int(rng.integers(3, 9))
            n = length+int(rng.integers(0, 4))
            options = [(r, c) for r in range(1, 4) for c in range(1, 5)
                       if n <= r*c <= 12 and r*size+3*(r-1) <= 26
                       and c*size+3*(c-1) <= 28]
            if not options:
                continue
            rows, cols = options[int(rng.integers(len(options)))]
            ys = _coordinates(rng, rows, size, 26)
            xs = _coordinates(rng, cols, size, 28)
        universe = [(y, x) for y in ys for x in xs]
        if n > len(universe):
            continue
        positions = sorted(universe[int(i)] for i in rng.choice(len(universe), n, replace=False))
        edges = _geometry_edges(positions, size)
        # Prevent crossing search windows, including unused gaps. This keeps
        # obstacle sampling independent and gives every marked cell one owner.
        rectangles = list(edges.values())
        if any(_overlap(a, b) for i, a in enumerate(rectangles) for b in rectangles[i+1:]):
            continue
        graph = _graph_from_edges(n, edges)
        route = _sample_path(rng, graph, length)
        if route is None:
            continue
        route_edges = {tuple(sorted(e)) for e in zip(route, route[1:])}
        if anchor in (0, 1) and not set(edges)-route_edges:
            continue
        return size, patch, positions, edges, graph, route
    # Same semantics, guaranteed full 2x3/3x3 visibility graph after bounded search.
    size, patch = (5, 1)
    positions = [(8*r, 8*c) for r in range(3 if test else 2) for c in range(3)]
    route = [0, 1, 2, 5, 4, 3, 6, 7, 8] if test else [0, 1, 2, 5, 4]
    edges = _geometry_edges(positions, size)
    return size, patch, positions, edges, _graph_from_edges(len(positions), edges), route


def _make_pair(rng, params, anchor, test, allowed_max):
    for attempt in range(48):
        size, patch, positions, edges, graph, route = _layout(rng, anchor, test)
        roles = (tuple(map(int, rng.choice(10, 3, replace=False))) if params["varied"]
                 else params["roles"])
        bg, carrier, obstacle = roles
        colors = [c for c in range(10) if c not in roles]
        assigned = _assign_labels(rng, graph, route, colors, repeated_start=anchor == 1)
        if assigned is None:
            continue
        labels, key = assigned
        bh = max(y for y, x in positions)+size
        bw = max(x for y, x in positions)+size
        height = int(rng.integers(bh+4, 31))
        width = int(rng.integers(max(bw+2, 2*len(key)+1), 31))
        top = int(rng.integers(1, height-bh-2))
        left = int(rng.integers(1, width-bw))
        a = np.full((height, width), bg, dtype=np.int8)
        offset = (size-patch)//2
        for (y, x), label in zip(positions, labels):
            y, x = y+top, x+left
            a[y:y+size, x:x+size] = carrier
            a[y+offset:y+offset+patch, x+offset:x+offset+patch] = label
        key_y = int(rng.integers(top+bh+1, height-1))
        key_x = int(rng.integers(1, width-2*len(key)+1))
        a[key_y, key_x:key_x+2*len(key):2] = key
        route_edges = [tuple(sorted(e)) for e in zip(route, route[1:])]
        active_unused = next((e for e in edges if e not in route_edges), None)
        for edge, rect in edges.items():
            horizontal = positions[edge[0]][0] == positions[edge[1]][0]
            length = rect[3]-rect[1] if horizontal else rect[2]-rect[0]
            if anchor == 0 and edge == route_edges[0]:
                mode = "competition"
            elif anchor == 1 and edge == route_edges[0]:
                mode = "nonsquare"
            elif edge == active_unused or (anchor in (0, 1) and edge in route_edges):
                mode = "active"
            elif size != length and rng.random() < .22:
                mode = "nonsquare"
            else:
                mode = "ordinary"
            # Anchor 0 contrasts a competing side >=4 with a separate side 3,
            # guaranteeing visible size variation even with only two demos.
            local_max = 3 if anchor == 0 and edge == route_edges[1] else allowed_max
            mask = _gap_mask(rng, size, length, patch, mode, local_max)
            if not horizontal:
                mask = mask.T
            y, x, yy, xx = rect
            a[y+top:yy+top, x+left:xx+left][~mask] = obstacle
        output, info = _analyze(a)
        if not np.any(output != a):
            continue
        # Anchor 1 and the tests must retain at least one changed source/target
        # contrast; a same-label link alone cannot teach departure coloring.
        if not any(w["winner"] is not None and w["winner"][2] >= 3
                   and info["stations"][w["edge"][0]][2] != info["stations"][w["edge"][1]][2]
                   for w in info["windows"]):
            continue
        return {"input": a, "output": output}
    raise RuntimeError("bounded pair construction exhausted")


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError("illegal example counts")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _signature(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    params = dict(roles=tuple(map(int, rng.choice(10, 3, replace=False))),
                  varied=bool(rng.integers(0, 2)))
    puzzle, seen, maximum = {"train": [], "test": []}, set(), 7
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            anchor = index if split == "train" and index < 2 else -1
            for attempt in range(12):
                pair = _make_pair(rng, params, anchor, split == "test", maximum)
                signature = _signature(pair["input"])
                if signature not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
            seen.add(signature)
            puzzle[split].append(pair)
        if split == "train":
            maximum = max(w["winner"][2] for p in puzzle[split]
                          for w in _analyze(p["input"])[1]["windows"] if w["winner"] is not None)
    validate_generated_puzzle(puzzle)
    return puzzle


def _witnesses(a, info):
    bg = info["background"]
    competition = False
    no_candidate = False
    for w in info["windows"]:
        mask = a[_slice(w["rect"])] == bg
        competition |= _has_largest_witness(mask, w["squares"], w["winner"])
        no_candidate |= not w["squares"] and bool(np.all(mask)) and mask.shape[0] != mask.shape[1]
    unrouted = False
    for edge, rect in info["edges"].items():
        if edge not in info["chosen_edges"]:
            q = _maximal_squares(a[_slice(rect)] == bg)
            if q and sum(s[2] == max(t[2] for t in q) for s in q) == 1:
                unrouted |= max(s[2] for s in q) >= 3
    return competition, no_candidate, unrouted


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "shape changed")
            signature = _signature(a)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            expected, info = _analyze(a)
            _require(np.array_equal(expected, b), "incorrect output")
            _require(np.any(a != b), "pair must show a change")
            _require(np.all(a[info["painted"]] == info["background"]), "non-background changed")
            _require(np.array_equal(a[~info["painted"]], b[~info["painted"]]), "preservation failed")
            # Full search windows do not cross in generated family members.
            rects = list(info["edges"].values())
            _require(not any(_overlap(x, y) for i, x in enumerate(rects) for y in rects[i+1:]),
                     "crossing gap windows")
            infos[split].append(info)
    train = infos["train"]
    witnesses = [_witnesses(np.asarray(p["input"]), i) for p, i in zip(puzzle["train"], train)]
    _require(any(w[0] for w in witnesses), "training lacks unequal-square selection")
    _require(any(w[1] for w in witnesses), "training lacks maximality counterexample")
    _require(any(w[2] for w in witnesses), "training lacks unrouted square distractor")
    _require(any(i["key"][0] in i["key"][1:] for i in train), "training lacks repeated start label")
    _require({"h", "v"} <= {"h" if d in "LR" else "v" for i in train for d in i["directions"]},
             "training must show both axes")
    sides = {w["winner"][2] for i in train for w in i["windows"] if w["winner"] is not None}
    _require(len(sides - {2}) >= 2, "training lacks changing square sizes")
    largest = max(sides)
    for i in infos["test"]:
        _require(len(i["route"]) >= 6, "test must recombine into a longer route")
        _require(all(w["winner"] is None or w["winner"][2] <= largest for w in i["windows"]),
                 "test exceeds demonstrated square scale")
    return None


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def _check_teaching_ablations(puzzle):
    """Competing interpretations must visibly fail on the demonstrations."""
    failed = set()
    for pair in puzzle["train"]:
        a, correct = pair["input"], pair["output"]
        _, info = _analyze(a)
        band_only, all_maxima, whole_square, later_color = (a.copy() for _ in range(4))
        for window in info["windows"]:
            i, j = window["edge"]
            first, second = info["stations"][i], info["stations"][j]
            rect = window["rect"]
            band_only[_slice(_band(first, second, info["size"], info["patch"]))] = first[2]
            for q in window["squares"]:
                all_maxima[_slice(rect)][_interior_mask(all_maxima[_slice(rect)].shape, q)] = first[2]
            q = window["winner"]
            if q is not None:
                y, x, side = q
                whole_square[rect[0]+y:rect[0]+y+side, rect[1]+x:rect[1]+x+side] = first[2]
                later_color[_slice(rect)][_interior_mask(later_color[_slice(rect)].shape, q)] = second[2]
        for name, wrong in (("omit_A", band_only), ("every_maximum", all_maxima),
                            ("whole_square", whole_square), ("target_color", later_color)):
            if not np.array_equal(wrong, correct):
                failed.add(name)
        # Remove B's scope restriction: an unselected facing gap would change.
        without_b = correct.copy()
        for edge, rect in info["edges"].items():
            if edge in info["chosen_edges"]:
                continue
            candidates = _maximal_squares(a[_slice(rect)] == info["background"])
            if not candidates or sum(q[2] == max(t[2] for t in candidates) for q in candidates) != 1:
                continue
            winner = _winner(candidates)
            without_b[_slice(rect)][_interior_mask(without_b[_slice(rect)].shape, winner)] = info["stations"][edge[0]][2]
        if not np.array_equal(without_b, correct):
            failed.add("omit_B")
        # A blank nonsquare routed rectangle contains square windows >=3, yet
        # none survives full-side extension. This refutes square-window filling.
        for window in info["windows"]:
            field = a[_slice(window["rect"])]
            if not window["squares"] and np.all(field == info["background"]) and min(field.shape) >= 3:
                failed.add("ignore_local_maximality")
    assert failed == {"omit_A", "omit_B", "every_maximum", "whole_square", "target_color", "ignore_local_maximality"}


def self_test():
    """42 streamed seeds, local counterexamples, all legal counts; no disk writes."""
    rng = np.random.default_rng(1742)
    # A separate direct window enumeration checks the optimized candidate sums.
    for trial in range(24):
        mask = rng.random((int(rng.integers(2, 8)), int(rng.integers(2, 9)))) < .8
        expected = []
        h, w = mask.shape
        for side in range(2, min(h, w)+1):
            for y in range(h-side+1):
                for x in range(w-side+1):
                    blank = mask[y:y+side, x:x+side].all()
                    extended = ((y > 0 and mask[y-1, x:x+side].all())
                                or (y+side < h and mask[y+side, x:x+side].all())
                                or (x > 0 and mask[y:y+side, x-1].all())
                                or (x+side < w and mask[y:y+side, x+side].all()))
                    if blank and not extended:
                        expected.append((y, x, side))
        assert expected == _maximal_squares(mask)
    assert _maximal_squares(np.ones((4, 7), dtype=bool)) == []
    assert _maximal_squares(np.ones((4, 4), dtype=bool)) == [(0, 0, 4)]
    assert not _interior_mask((2, 2), (0, 0, 2)).any()
    try:
        _winner([(0, 0, 3), (4, 0, 3)])
    except ValueError:
        pass
    else:
        raise AssertionError("equal-area tie accepted")
    shapes, layouts, palettes, structures, sizes, lengths = set(), set(), set(), set(), set(), set()
    flags = dict(competition=0, nonsquare=0, unrouted=0, repeated=0, adjacent_equal=0, unused=0, sparse=0)
    fingerprints = set()
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _check_teaching_ablations(puzzle)
        fingerprint = hashlib.sha256()
        if seed == 0:
            first = puzzle
        for split in ("train", "test"):
            for p in puzzle[split]:
                a = p["input"]
                old = a.copy()
                predicted, info = _analyze(a)
                assert np.array_equal(a, old) and np.array_equal(predicted, p["output"])
                fingerprint.update(a.tobytes())
                shapes.add(a.shape)
                ys = [s[0] for s in info["stations"]]
                xs = [s[1] for s in info["stations"]]
                layouts.add((info["size"], tuple((y-min(ys), x-min(xs)) for y, x, c in info["stations"])))
                palettes.add((info["background"], info["carrier"], info["obstacle"]))
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)
                structures.add((a.shape, normalized))
                sizes.update(w["winner"][2] for w in info["windows"] if w["winner"] is not None)
                lengths.add(len(info["route"]))
                c, n, u = _witnesses(a, info)
                flags["competition"] += c
                flags["nonsquare"] += n
                flags["unrouted"] += u
                flags["repeated"] += len(set(info["key"])) < len(info["key"])
                flags["adjacent_equal"] += any(x == y for x, y in zip(info["key"], info["key"][1:]))
                flags["unused"] += len(info["stations"]) > len(info["route"])
                flags["sparse"] += len(info["stations"]) < len(set(ys))*len(set(xs))
        fingerprints.add(fingerprint.digest())
    assert len(fingerprints) == 42
    assert len(shapes) >= 30 and len(layouts) >= 70 and len(structures) >= 190
    assert len(palettes) >= 25 and {3, 4, 5} <= sizes and {3, 5, 6, 7, 8} <= lengths
    assert all(v >= 5 for v in flags.values()), flags
    assert _equal(first, generate_puzzle(0))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(12345+10*nt+nq, nt, nq)
            assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (4.0, 1), (None, 1), (4, None), (4, False), ("4", 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        return
    raise AssertionError("output corruption accepted")
