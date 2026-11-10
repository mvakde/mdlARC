"""Border-woven relays: an itinerary gives edge roles; territory selects ink.

Standard library and NumPy only. No file access, hidden puzzle state, or parent
generator imports. Generation uses sampled routes; solving redecodes the input.
"""
from hashlib import sha256

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= a) & (a <= 9)), "ARC colors required")
    return a


def _components(mask, diagonal=False):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    result = []
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack, cells = [(y, x)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                yy, xx = r + dr, c + dc
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _mask(shape, cells):
    a = np.zeros(shape, dtype=bool)
    for p in cells:
        a[p] = True
    return a


def _cut_regions(cut):
    h, w = cut.shape
    cells = {tuple(map(int, p)) for p in np.argwhere(cut)}
    _require(len(cells) >= 3, "short cut")
    neighbors = {p: [q for q in ((p[0]-1, p[1]), (p[0]+1, p[1]),
                                 (p[0], p[1]-1), (p[0], p[1]+1)) if q in cells]
                 for p in cells}
    ends = {p for p, ns in neighbors.items() if len(ns) == 1}
    _require(len(ends) == 2 and all(1 <= len(ns) <= 2 for ns in neighbors.values()),
             "cut must be a nonbranching orthogonal path")
    border = {p for p in cells if p[0] in (0, h-1) or p[1] in (0, w-1)}
    _require(border == ends, "only endpoints may meet boundary")
    _require(len(_components(cut)) == 1, "disconnected cut")
    parts = _components(~cut)
    _require(len(parts) == 2, "cut must leave two territories")
    return [_mask(cut.shape, p) for p in parts]


def _band(a, b, size, patch):
    d = (size - patch) // 2
    if a[0] == b[0] and abs(a[1] - b[1]) > size:
        return (slice(a[0]+d, a[0]+d+patch),
                slice(min(a[1], b[1])+size, max(a[1], b[1])))
    if a[1] == b[1] and abs(a[0] - b[0]) > size:
        return (slice(min(a[0], b[0])+size, max(a[0], b[0])),
                slice(a[1]+d, a[1]+d+patch))
    return None


def _graph(positions, size, patch, shape):
    occupied = np.zeros(shape, dtype=bool)
    for r, c in positions:
        occupied[r:r+size, c:c+size] = True
    graph, bands = [[] for _ in positions], {}
    for i, a in enumerate(positions):
        for j in range(i+1, len(positions)):
            band = _band(a, positions[j], size, patch)
            if band is not None and not np.any(occupied[band]):
                graph[i].append(j)
                graph[j].append(i)
                bands[i, j] = bands[j, i] = band
    return graph, bands, occupied


def _route_solutions(graph, labels, key):
    answers = []
    candidates = [[i for i, value in enumerate(labels) if value == k] for k in key]

    def visit(path, used):
        if len(answers) == 2:
            return
        if len(path) == len(key):
            answers.append(tuple(path))
            return
        for i in candidates[len(path)]:
            if i not in used and (not path or i in graph[path[-1]]):
                visit(path + [i], used | {i})

    visit([], set())
    return answers


def _sites(cut, region, forbidden):
    near = cut.copy()
    h, w = cut.shape
    for r, c in np.argwhere(cut):
        near[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = True
    return [tuple(map(int, p)) for p in np.argwhere(region & ~near & ~forbidden)]


def _decode(input_grid):
    grid = _grid(input_grid)
    frame, bg = int(grid[0, 0]), int(grid[-1, 0])
    _require(frame != bg and np.all(grid[0] == frame) and np.all(grid[-1] == bg),
             "frame/background missing")
    bottoms = [r for r in range(1, grid.shape[0]) if np.all(grid[r] == frame)]
    _require(len(bottoms) == 1, "map bottom must be unique")
    bottom = bottoms[0]
    _require(grid.shape[0] == bottom+4 and np.all(grid[bottom+1] == bg), "key layout")
    _require(np.all(grid[:bottom+1, 0] == frame)
             and np.all(grid[:bottom+1, -1] == frame), "broken map frame")
    board = grid[1:bottom, 1:-1]
    h, w = board.shape
    _require(h >= 5 and w >= 5 and not np.any(board == frame), "invalid map interior")
    boundary = np.concatenate((board[0], board[-1], board[:, 0], board[:, -1]))
    controls = set(map(int, boundary)) - {bg}
    _require(len(controls) == 1, "boundary must identify one control color")
    control = next(iter(controls))
    parts = sorted(_components(board == control, diagonal=True), key=len)
    _require(len(parts) == 2 and len(parts[0]) == 1 and len(parts[1]) > 1,
             "one isolated marker and one cut required")
    marker = parts[0][0]
    cut = _mask(board.shape, parts[1])
    regions = _cut_regions(cut)
    side = 0 if regions[0][marker] else 1
    stations, geometry = [], None
    for component in _components((board != bg) & (board != control)):
        rs, cs = zip(*component)
        r, c = min(rs), min(cs)
        size = max(rs)-r+1
        _require(size in (3, 4, 5) and max(cs)-c+1 == size
                 and len(component) == size*size, "station must be a filled square")
        block = board[r:r+size, c:c+size]
        carrier = int(block[0, 0])
        yy, xx = np.nonzero(block != carrier)
        _require(len(yy) > 0, "label patch missing")
        offset, patch = int(yy.min()), int(yy.max()-yy.min()+1)
        _require(offset >= 1 and 2*offset+patch == size
                 and int(xx.min()) == offset and int(xx.max()) == offset+patch-1
                 and len(yy) == patch*patch, "patch must be centered and square")
        label = int(block[offset, offset])
        _require(np.all(block[offset:offset+patch, offset:offset+patch] == label), "mixed label")
        _require(len({frame, bg, control, carrier, label}) == 5, "role collision")
        current = size, patch, carrier
        _require(geometry is None or geometry == current, "unequal station geometry")
        geometry = current
        _require(0 < r and 0 < c and r+size < h and c+size < w, "station on boundary")
        stations.append((r, c, label))
    _require(5 <= len(stations) <= 16, "station count")
    size, patch, carrier = geometry
    stations.sort()
    positions = [(r, c) for r, c, label in stations]
    labels = [label for r, c, label in stations]
    key_y = bottom+2
    xs = np.flatnonzero(grid[key_y] != bg)
    _require(4 <= len(xs) <= 10 and xs[0] >= 1 and xs[-1] < grid.shape[1]-1
             and np.all(np.diff(xs) == 2), "spaced horizontal itinerary required")
    key = [int(grid[key_y, c]) for c in xs]
    _require(len(set(key)) >= 3 and not set(key) & {bg, frame, carrier, control}, "key labels")
    graph, bands, occupied = _graph(positions, size, patch, board.shape)
    routes = _route_solutions(graph, labels, key)
    _require(len(routes) == 1, "itinerary must identify a unique distinct-station route")
    route = routes[0]
    ribbons = np.zeros_like(cut)
    for edge in zip(route, route[1:]):
        band = bands[edge]
        _require(not np.any(ribbons[band]), "ribbons overlap")
        ribbons[band] = True
    _require(marker in _sites(cut, regions[side], occupied | ribbons), "marker touches cut or content")
    _require(all(_sites(cut, reg, occupied | ribbons) for reg in regions),
             "both marker selections must be available")
    return dict(grid=grid, board=board, bottom=bottom, background=bg, frame=frame,
                carrier=carrier, control=control, cut=cut, regions=regions,
                marker=marker, side=side, positions=positions, labels=labels,
                size=size, patch=patch, key=key, key_y=key_y, key_x=int(xs[0]),
                graph=graph, bands=bands, route=route, occupied=occupied, ribbons=ribbons)


def _cellwise(d, policy="territory"):
    """Input-derived cellwise solve/validation path; no generation metadata."""
    out = d["grid"].copy()
    for i, j in zip(d["route"], d["route"][1:]):
        band = d["bands"][i, j]
        for r in range(band[0].start, band[0].stop):
            for c in range(band[1].start, band[1].stop):
                departure = d["cut"][r, c] or d["regions"][d["side"]][r, c]
                if policy == "departure":
                    departure = True
                elif policy == "arrival":
                    departure = False
                elif policy == "swapped" and not d["cut"][r, c]:
                    departure = not departure
                elif policy == "arrival_seam" and d["cut"][r, c]:
                    departure = False
                out[r+1, c+1] = d["labels"][i if departure else j]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output cell from the input alone."""
    return _cellwise(_decode(input_grid))


def _coordinates(rng, count, size, limit, minimum=1):
    gaps = [minimum] * (count-1)
    spare = limit - count*size - sum(gaps) - 2
    if spare < 0:
        raise ValueError("geometry does not fit")
    for index in rng.permutation(len(gaps)):
        extra = int(rng.integers(0, min(4-minimum, spare)+1))
        gaps[int(index)] += extra
        spare -= extra
    margin_extra = int(rng.integers(0, min(3, spare)+1))
    leading = 1+int(rng.integers(0, margin_extra+1))
    positions = [leading]
    for gap in gaps:
        positions.append(positions[-1]+size+gap)
    return positions, count*size+sum(gaps)+2+margin_extra


def _geometry(rng, size, style, anchor=False, fallback=False):
    options = [(nr, nc) for nr in range(2, 5) for nc in range(2, 5)
               if nr*nc >= 6 and nr*size+nr+1 <= 25 and nc*size+nc+1 <= 28
               and (style != "step" or nc >= 3)]
    nr, nc = (3, 3) if anchor or fallback else options[int(rng.integers(len(options)))]
    # Introductory whole-territory ribbons have a genuine near and far end,
    # distinguishing territorial ink from nearest-endpoint coloring.
    minimum = 2 if anchor or fallback else 1
    ys, h = _coordinates(rng, nr, size, 25, minimum)
    xs, w = _coordinates(rng, nc, size, 28, minimum)
    cells = [(r, c) for r in range(nr) for c in range(nc)]
    if anchor or fallback:
        chosen = cells
    else:
        count = int(rng.integers(5, len(cells)+1))
        chosen = [cells[int(i)] for i in sorted(rng.choice(len(cells), count, replace=False))]
    positions = [(ys[r], xs[c]) for r, c in chosen]
    xg = [int(rng.integers(xs[c]+size, xs[c+1])) for c in range(nc-1)]
    yg = [int(rng.integers(ys[r]+size, ys[r+1])) for r in range(nr-1)]
    cut = np.zeros((h, w), dtype=bool)
    c = xg[0] if fallback else int(rng.choice(xg))
    if style == "straight":
        cut[:, c] = True
    elif style == "elbow":
        r = yg[0] if fallback else int(rng.choice(yg))
        cut[:r+1, c] = True
        cut[r, c:] = True
    elif style == "step":
        c1 = xg[-1] if fallback else int(rng.choice([x for x in xg if x != c]))
        r = yg[0] if fallback else int(rng.choice(yg))
        cut[:r+1, c] = True
        cut[r, min(c, c1):max(c, c1)+1] = True
        cut[r:, c1] = True
    else:
        raise ValueError("unknown cut style")
    # A uniform map transform preserves roles and changes endpoint arrangements.
    # The fallback keeps its known coordinate indexing for the constructive path.
    if not fallback:
        if rng.random() < .5:
            cut = cut.T.copy()
            positions = [(c, r) for r, c in positions]
            h, w = w, h
        # Transposition must still leave room for the detached key beneath map.
        if h > 25:
            cut = cut.T.copy()
            positions = [(c, r) for r, c in positions]
            h, w = w, h
        if rng.random() < .5:
            cut = cut[::-1].copy()
            positions = [(h-r-size, c) for r, c in positions]
        if rng.random() < .5:
            cut = cut[:, ::-1].copy()
            positions = [(r, w-c-size) for r, c in positions]
    return positions, cut


def _sample_path(rng, positions, graph, bands, cut, regions, length, anchor):
    budget = 5000
    used_pixels = np.zeros_like(cut)

    def visit(path):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            edges = list(zip(path, path[1:]))
            cross = any(np.any(cut[bands[e]]) for e in edges)
            if not cross:
                return None
            if anchor:
                axes = {positions[i][0] == positions[j][0] for i, j in edges}
                whole = [any(np.all(region[bands[e]]) for e in edges) for region in regions]
                if len(axes) < 2 or not all(whole):
                    return None
            return tuple(path)
        for nxt in rng.permutation(graph[path[-1]]):
            nxt = int(nxt)
            if nxt in path:
                continue
            band = bands[path[-1], nxt]
            if np.any(used_pixels[band]):
                continue
            used_pixels[band] = True
            found = visit(path+[nxt])
            used_pixels[band] = False
            if found is not None:
                return found
        return None

    for start in rng.permutation(len(positions)):
        result = visit([int(start)])
        if result is not None:
            return result
        if budget < 0:
            break
    return None


def _assign_labels(rng, graph, route, colors, bands, cut, anchor):
    length = len(route)
    unused = [i for i in range(len(graph)) if i not in route]
    for _ in range(128):
        k = int(rng.integers(3, min(6, length-1 if anchor else length)+1))
        pool = list(map(int, rng.permutation(colors)[:k]))
        key = pool+[int(rng.choice(pool)) for _ in range(length-k)]
        key = [int(x) for x in rng.permutation(key)]
        if anchor and any(a == b for a, b in zip(key, key[1:])):
            continue
        labels = [0]*len(graph)
        for i, value in zip(route, key):
            labels[i] = value
        for i in unused:
            labels[i] = int(rng.choice(colors if rng.random() < .5 else pool))
        if not any(labels[i] != labels[j] and np.any(cut[bands[i, j]])
                   for i, j in zip(route, route[1:])):
            continue
        if _route_solutions(graph, labels, key) == [tuple(route)]:
            return labels, key
    return None


def _pack(board, key, background, frame, key_x):
    h, w = board.shape
    grid = np.full((h+5, w+2), background, dtype=np.int8)
    grid[:h+2] = frame
    grid[1:h+1, 1:w+1] = board
    grid[h+3, key_x:key_x+2*len(key):2] = key
    return grid


def _render(positions, size, patch, labels, key, route, cut, regions,
            marker, side, roles, key_x):
    bg, frame, carrier, control = roles
    board = np.full(cut.shape, bg, dtype=np.int8)
    board[cut] = control
    board[marker] = control
    offset = (size-patch)//2
    for (r, c), label in zip(positions, labels):
        board[r:r+size, c:c+size] = carrier
        board[r+offset:r+offset+patch, c+offset:c+offset+patch] = label
    output_board = board.copy()
    # Construction paints vector rectangles from the sampled route. The solver
    # instead parses the packed input, resolves its route, and visits each pixel.
    departure_zone = cut | regions[side]
    for i, j in zip(route, route[1:]):
        band = _band(positions[i], positions[j], size, patch)
        output_board[band] = np.where(departure_zone[band], labels[i], labels[j])
    return {"input": _pack(board, key, bg, frame, key_x),
            "output": _pack(output_board, key, bg, frame, key_x)}


def _sample_pair(rng, size, patch, style, roles, anchor=False, force_fallback=False):
    colors = [c for c in range(10) if c not in roles]
    attempts = 0 if force_fallback else 64
    for _ in range(attempts):
        positions, cut = _geometry(rng, size, style, anchor)
        regions = _cut_regions(cut)
        graph, bands, occupied = _graph(positions, size, patch, cut.shape)
        maximum = min(10, len(positions)-1 if anchor else len(positions), (cut.shape[1]+1)//2)
        if maximum < (5 if anchor else 4):
            continue
        length = int(rng.integers(5 if anchor else 4, min(7, maximum)+1 if anchor else maximum+1))
        route = _sample_path(rng, positions, graph, bands, cut, regions, length, anchor)
        if route is None:
            continue
        assigned = _assign_labels(rng, graph, route, colors, bands, cut, anchor)
        if assigned is None:
            continue
        labels, key = assigned
        exposed = [False, False, False]
        for i, j in zip(route, route[1:]):
            if labels[i] != labels[j]:
                for k, zone in enumerate([*regions, cut]):
                    exposed[k] |= bool(np.any(zone[bands[i, j]]))
        if not all(exposed):
            continue
        ribbon = np.zeros_like(cut)
        for edge in zip(route, route[1:]):
            ribbon[bands[edge]] = True
        sites = [_sites(cut, region, occupied | ribbon) for region in regions]
        if not all(sites):
            continue
        side = int(rng.integers(2))
        marker = sites[side][int(rng.integers(len(sites[side])))]
        key_x = int(rng.integers(1, cut.shape[1]-(2*len(key)-1)+2))
        return _render(positions, size, patch, labels, key, route, cut, regions,
                       marker, side, roles, key_x)
    # Known unique repeated-label route. Colors absent from the key identify
    # unused stations, while the two occurrences of a still require alignment.
    positions, cut = _geometry(rng, size, style, fallback=True)
    route = (2, 1, 0, 3, 4) if style == "elbow" else (0, 3, 4, 5, 8)
    a, b, c, d, e, f = [int(x) for x in rng.permutation(colors)]
    key = [a, b, c, a, d]
    labels = [e if i % 2 else f for i in range(9)]
    for i, value in zip(route, key):
        labels[i] = value
    graph, bands, occupied = _graph(positions, size, patch, cut.shape)
    _require(_route_solutions(graph, labels, key) == [route], "fallback route uniqueness")
    regions = _cut_regions(cut)
    ribbon = np.zeros_like(cut)
    for edge in zip(route, route[1:]):
        ribbon[bands[edge]] = True
    _require(np.any(ribbon & cut), "fallback must cross border")
    sites = [_sites(cut, region, occupied | ribbon) for region in regions]
    _require(all(sites), "fallback marker sites")
    side = int(rng.integers(2))
    marker = sites[side][int(rng.integers(len(sites[side])))]
    return _render(positions, size, patch, labels, key, route, cut, regions,
                   marker, side, roles, 1)


def _opposite(pair, rng):
    d = _decode(pair["input"])
    side = 1-d["side"]
    sites = _sites(d["cut"], d["regions"][side], d["occupied"] | d["ribbons"])
    marker = sites[int(rng.integers(len(sites)))]
    roles = d["background"], d["frame"], d["carrier"], d["control"]
    return _render(d["positions"], d["size"], d["patch"], d["labels"], d["key"],
                   d["route"], d["cut"], d["regions"], marker, side, roles, d["key_x"])


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("example counts must be integers")
        if not low <= int(value) <= high:
            raise ValueError("example count outside legal range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    base_roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    varied = bool(rng.integers(2))
    size = int(rng.choice((3, 4, 5)))
    patch = 1 if size == 3 else 2 if size == 4 else int(rng.choice((1, 3)))
    puzzle, seen, taught = {"train": [], "test": []}, set(), {patch}
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            if split == "train" and index == 1:
                pair = _opposite(puzzle["train"][0], rng)
            else:
                roles = tuple(map(int, rng.choice(10, 4, replace=False))) if varied else base_roles
                if split == "train" and index == 0:
                    current_size, current_patch, style, anchor = size, patch, "straight", True
                else:
                    if split == "train" and index == 2 and size == 5:
                        current_size, current_patch = 5, 4-patch
                    else:
                        choices = [(s, p) for s, p in ((3, 1), (4, 2), (5, 1), (5, 3)) if p in taught]
                        current_size, current_patch = choices[int(rng.integers(len(choices)))]
                    # A visible structural shift, independent of ink palettes:
                    # straight -> elbow for minimal puzzles; elbow -> two turns
                    # once a bent training demonstration is available.
                    style = "elbow" if split == "train" or nt == 2 else "step"
                    anchor = False
                pair = _sample_pair(rng, current_size, current_patch, style, roles, anchor)
                if split == "train":
                    taught.add(current_patch)
            # At most 13 previous examples; 40 invertible palette maps suffice.
            if _signature(pair["input"]) in seen:
                for multiplier in (1, 3, 7, 9):
                    found = False
                    for shift in range(10):
                        lookup = (np.arange(10)*multiplier+shift) % 10
                        candidate = {k: lookup[v].astype(np.int8) for k, v in pair.items()}
                        if _signature(candidate["input"]) not in seen:
                            pair, found = candidate, True
                            break
                    if found:
                        break
                _require(found, "cannot separate example")
            seen.add(_signature(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(d):
    marked = d["regions"][d["side"]]
    other = d["regions"][1-d["side"]]
    result = {"marked": False, "other": False, "seam": False,
              "whole_marked": False, "whole_other": False, "far_end": False}
    selected = {frozenset(e) for e in zip(d["route"], d["route"][1:])}
    result["unused_edge"] = any(frozenset((i, j)) not in selected
                                for i, ns in enumerate(d["graph"]) for j in ns)
    result["unused_station"] = len(d["positions"]) > len(d["route"])
    result["repeat"] = len(set(d["key"])) < len(d["key"])
    for i, j in zip(d["route"], d["route"][1:]):
        if d["labels"][i] == d["labels"][j]:
            continue
        band = d["bands"][i, j]
        for name, zone in (("marked", marked), ("other", other), ("seam", d["cut"])):
            result[name] |= bool(np.any(zone[band]))
        result["whole_marked"] |= bool(np.all(marked[band]))
        result["whole_other"] |= bool(np.all(other[band]))
        a, b = d["positions"][i], d["positions"][j]
        gap_length = abs(a[0]-b[0])+abs(a[1]-b[1])-d["size"]
        result["far_end"] |= gap_length >= 2 and bool(np.all(marked[band]) or np.all(other[band]))
    return result


def _turns(cut):
    cells = {tuple(map(int, p)) for p in np.argwhere(cut)}
    graph = {p: [q for q in ((p[0]-1, p[1]), (p[0]+1, p[1]),
                             (p[0], p[1]-1), (p[0], p[1]+1)) if q in cells] for p in cells}
    start = min(p for p, ns in graph.items() if len(ns) == 1)
    steps, previous, current = [], None, start
    while True:
        following = [p for p in graph[current] if p != previous]
        if not following:
            break
        nxt = following[0]
        steps.append((nxt[0]-current[0], nxt[1]-current[1]))
        previous, current = current, nxt
    return sum(a != b for a, b in zip(steps, steps[1:]))


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    infos, seen, taught, evidence = {"train": [], "test": []}, set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "output shape")
            signature = _signature(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            d = _decode(inp)
            _require(np.array_equal(out, _cellwise(d)), "wrong ribbon or changed preserved cell")
            _require(np.any(inp != out), "empty transformation")
            flags = _evidence(d)
            _require(flags["marked"] and flags["other"] and flags["seam"], "ineffective partition")
            if split == "train":
                taught.add(d["patch"])
                evidence |= {k for k, v in flags.items() if v}
            else:
                _require(d["patch"] in taught, "test patch width was not taught")
            infos[split].append(d)
    _require(evidence >= {"marked", "other", "seam", "whole_marked", "whole_other",
                          "repeat", "unused_edge", "unused_station", "far_end"}, "insufficient training evidence")
    train_turns = max(_turns(d["cut"]) for d in infos["train"])
    _require(all(_turns(d["cut"]) == train_turns+1 for d in infos["test"]),
             "test must extend the demonstrated cut by one bend")
    a, b = infos["train"][:2]
    differences = {tuple(map(int, p)) for p in np.argwhere(a["grid"] != b["grid"])}
    _require(a["side"] != b["side"] and differences == {(a["marker"][0]+1, a["marker"][1]+1),
                                                          (b["marker"][0]+1, b["marker"][1]+1)},
             "first demonstrations must isolate opposite marker selection")
    _require(not np.array_equal(puzzle["train"][0]["output"], puzzle["train"][1]["output"]),
             "marker twins must change answer")


def _puzzle_digest(puzzle):
    h = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, payload = _signature(pair[role])
                h.update(bytes(shape))
                h.update(payload)
    return h.hexdigest()


def _reflect_map(grid, axis):
    d = _decode(grid)
    out = grid.copy()
    out[1:d["bottom"], 1:-1] = np.flip(d["board"], axis)
    return out


def self_test():
    """Bounded, streaming checks; return small descriptive measurements."""
    signatures, structures, shapes, sizes, patches = set(), set(), set(), set(), set()
    cut_shapes, route_shapes, palettes, counts, sides = set(), set(), set(), set(), set()
    pair_count = sparse = unequal_gaps = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        _require(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed)), "determinism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _decode(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve mismatch")
                pair_count += 1
                shapes.add(d["grid"].shape)
                sizes.add(d["size"])
                patches.add(d["patch"])
                sides.add(d["side"])
                counts.add((len(d["positions"]), len(d["route"])))
                palettes.add((d["background"], d["frame"], d["carrier"], d["control"]))
                cut_shapes.add((d["cut"].shape, d["cut"].tobytes()))
                route_shapes.add(tuple(d["positions"][i] for i in d["route"]))
                mapping = {}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in d["grid"].flat)
                structures.add(sha256(bytes(d["grid"].shape)+normalized).hexdigest())
                rs = {r for r, c in d["positions"]}
                cs = {c for r, c in d["positions"]}
                sparse += len(d["positions"]) < len(rs)*len(cs)
                lengths = [max(b[0].stop-b[0].start, b[1].stop-b[1].start)
                           for b in (d["bands"][e] for e in zip(d["route"], d["route"][1:]))]
                unequal_gaps += len(set(lengths)) > 1
        first = puzzle["train"][0]
        d = _decode(first["input"])
        twin = _opposite(first, np.random.default_rng(seed+9100))
        prediction = solve(twin["input"])[1:d["bottom"], 1:-1]
        original = first["output"][1:d["bottom"], 1:-1]
        _require(np.array_equal(prediction[d["cut"]], original[d["cut"]]), "marker changed seam")
        _require(np.any(prediction != original), "marker ineffective")
        for policy in ("departure", "arrival", "swapped", "arrival_seam"):
            _require(not np.array_equal(first["output"], _cellwise(d, policy)), "shortcut survived")
        reversed_input = first["input"].copy()
        reversed_input[d["key_y"], d["key_x"]:d["key_x"]+2*len(d["key"]):2] = d["key"][::-1]
        rd = _decode(reversed_input)
        _require(rd["route"] == d["route"][::-1], "reversed itinerary geometry")
        reverse_result = solve(reversed_input)
        _require(np.any(reverse_result[1:d["bottom"], 1:-1] != original), "key order ineffective")
        reverse_board = reverse_result[1:d["bottom"], 1:-1]
        off_cut = d["ribbons"] & ~d["cut"]
        _require(np.array_equal(reverse_board[off_cut], prediction[off_cut]),
                 "reversing itinerary must exchange off-cut endpoint inks")
        for i, j in zip(d["route"], d["route"][1:]):
            band = d["bands"][i, j]
            seam = d["cut"][band]
            _require(np.all(reverse_board[band][seam] == d["labels"][j]),
                     "reverse seam must use the new departure")
        selected = {frozenset(e) for e in zip(d["route"], d["route"][1:])}
        unused_pixels = np.zeros_like(d["cut"])
        for i, ns in enumerate(d["graph"]):
            for j in ns:
                if frozenset((i, j)) not in selected:
                    unused_pixels[d["bands"][i, j]] = True
        unused_pixels &= ~d["ribbons"]
        _require(np.any(unused_pixels) and np.array_equal(original[unused_pixels], d["board"][unused_pixels]),
                 "ignoring itinerary must fill currently blank unselected gaps")
        for axis in (0, 1):
            reflected = _reflect_map(first["input"], axis)
            expected = first["output"].copy()
            expected[1:d["bottom"], 1:-1] = np.flip(original, axis)
            _require(np.array_equal(solve(reflected), expected), "reflection equivariance")
        lookup = np.random.default_rng(seed+12000).permutation(10)
        _require(np.array_equal(solve(lookup[first["input"]]), lookup[first["output"]]), "palette equivariance")
    _require(len(signatures) == 42 and len(structures) >= 140, "structural diversity")
    _require(len(shapes) >= 20 and len(cut_shapes) >= 80 and len(route_shapes) >= 100, "geometric diversity")
    _require(sizes == {3, 4, 5} and patches == {1, 2, 3} and sides == {0, 1}, "scale/side diversity")
    _require(len(counts) >= 15 and len(palettes) >= 30 and sparse >= 20 and unequal_gaps >= 80,
             "counts, gaps, palettes or occupancy collapsed")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(8000+10*nt+nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exercise constructive fallbacks rather than only hoping random search uses them.
    for style in ("straight", "elbow", "step"):
        p = _sample_pair(np.random.default_rng(880), 5, 3, style, (0, 1, 2, 3),
                         anchor=style == "straight", force_fallback=True)
        _require(np.array_equal(solve(p["input"]), p["output"]), "fallback output")
    for style in ("branch", "loop", "three_regions"):
        malformed = np.zeros((9, 9), dtype=bool)
        if style == "branch":
            malformed[:, 4] = True
            malformed[4, 4:7] = True
        elif style == "loop":
            malformed[2:7, 2] = malformed[2:7, 6] = True
            malformed[2, 2:7] = malformed[6, 2:7] = True
        else:
            malformed[:, 3] = malformed[:, 5] = True
        try:
            _cut_regions(malformed)
        except AssertionError:
            pass
        else:
            raise AssertionError("malformed cut accepted")
    p = generate_puzzle(19)
    d = _decode(p["train"][0]["input"])
    for i, j in zip(d["route"], d["route"][1:]):
        seam = np.argwhere(d["cut"] & _mask(d["cut"].shape,
                            [(r, c) for r in range(d["bands"][i, j][0].start, d["bands"][i, j][0].stop)
                             for c in range(d["bands"][i, j][1].start, d["bands"][i, j][1].stop)]))
        if len(seam) and d["labels"][i] != d["labels"][j]:
            r, c = map(int, seam[0])
            p["train"][0]["output"][r+1, c+1] = d["labels"][j]
            break
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect seam priority accepted")
    bad = generate_puzzle(0)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("changed frame accepted")
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(signatures),
            "normalized_inputs": len(structures), "canvas_shapes": len(shapes),
            "cut_geometries": len(cut_shapes), "route_geometries": len(route_shapes),
            "station_sizes": sorted(sizes), "patch_widths": sorted(patches),
            "station_route_counts": len(counts), "sparse_pairs": sparse,
            "unequal_gap_pairs": unequal_gaps, "role_palettes": len(palettes)}
