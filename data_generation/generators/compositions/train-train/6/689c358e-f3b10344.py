"""Arrow valves in implicit, same-color rectangle trees.

The input encodes every color role. No examples, seed, or external files are
consulted by solve(). See task_spec.md for the intended transformation.
"""

import numpy as np


DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "illegal grid dimensions")
    _check(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _check(np.all((0 <= a) & (a <= 9)), "illegal grid colors")
    return a.astype(np.int8, copy=False)


def _arrow_cells(head, direction, shaft=0, wing=1):
    r, c = head
    dr, dc = direction
    pr, pc = -dc, dr
    cells = {(r + k * pr, c + k * pc) for k in range(-wing, wing + 1)}
    cells.add((r + dr, c + dc))
    cells.update((r - k * dr, c - k * dc) for k in range(1, shaft + 1))
    return cells


def _recognize_arrow(cells):
    candidates = []
    for wing in (1, 2):
        shaft = len(cells) - 2 * wing - 2
        if shaft not in (0, 2, 3, 4, 5, 6) or (wing == 2 and shaft == 0):
            continue
        for head in sorted(cells):
            for direction in DIRECTIONS:
                if _arrow_cells(head, direction, shaft, wing) == cells:
                    candidates.append((head, direction, shaft, wing))
    return candidates[0] if len(candidates) == 1 else None


def _objects(g, background, frame):
    unseen = {(int(r), int(c)) for r, c in np.argwhere((g != background) & (g != frame))}
    rectangles, arrows = [], []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        color = int(g[start])
        cells, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in DIRECTIONS:
                nxt = r + dr, c + dc
                if nxt in unseen and int(g[nxt]) == color:
                    unseen.remove(nxt)
                    cells.add(nxt)
                    todo.append(nxt)
        r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
        c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
        if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1) and min(r1 - r0, c1 - c0) >= 2:
            rectangles.append((r0, r1, c0, c1, color))
        else:
            recognized = _recognize_arrow(cells)
            _check(recognized is not None, "object is neither a rectangle nor a readable arrow")
            head, direction, shaft, wing = recognized
            arrows.append(dict(head=head, direction=direction, shaft=shaft, wing=wing,
                               color=color, cells=cells))
    rectangles.sort()
    return rectangles, arrows


def _edges(rectangles, clean, background, match_color=True):
    """Consecutive exact-center neighbors; test visibility in the inset strip."""
    result = []
    for axis in (0, 1):
        groups = {}
        for i, rect in enumerate(rectangles):
            center2 = rect[0] + rect[1] if axis == 0 else rect[2] + rect[3]
            groups.setdefault(center2, []).append(i)
        for center2 in sorted(groups):
            members = sorted(groups[center2], key=lambda i: rectangles[i][2 if axis == 0 else 0])
            for a, b in zip(members, members[1:]):
                x, y = rectangles[a], rectangles[b]
                if match_color and x[4] != y[4]:
                    continue
                if axis == 0:
                    bounds = (max(x[0], y[0]) + 1, min(x[1], y[1]) - 1, x[3] + 1, y[2] - 1)
                else:
                    bounds = (x[1] + 1, y[0] - 1, max(x[2], y[2]) + 1, min(x[3], y[3]) - 1)
                r0, r1, c0, c1 = bounds
                if r0 <= r1 and c0 <= c1 and np.all(clean[r0:r1 + 1, c0:c1 + 1] == background):
                    result.append((axis, a, b, bounds))
    return result


def _reach(adjacency, start, forbidden):
    seen, todo = {start}, [start]
    while todo:
        a = todo.pop()
        for b, edge in adjacency[a]:
            if edge != forbidden and b not in seen:
                seen.add(b)
                todo.append(b)
    return seen


def _port(rect, neighbor, shape):
    r0, r1, c0, c1, _ = rect
    if r0 + r1 == neighbor[0] + neighbor[1]:
        col = 0 if c0 < neighbor[2] else shape[1] - 1
        return {(r, col) for r in range(r0 + 1, r1)}
    row = 0 if r0 < neighbor[0] else shape[0] - 1
    return {(row, c) for c in range(c0 + 1, c1)}


def _analyze(value):
    g = _grid(value)
    _check(min(g.shape) >= 7, "canvas is too small")
    frame = int(g[0, 0])
    border = np.concatenate((g[0], g[-1], g[1:-1, 0], g[1:-1, -1]))
    _check(np.all(border == frame), "input frame must be uniform")
    interior = g[1:-1, 1:-1]
    _check(not np.any(interior == frame), "frame color must be exclusive to the border")
    colors, counts = np.unique(interior, return_counts=True)
    _check(np.count_nonzero(counts == counts.max()) == 1, "background must be uniquely modal")
    background = int(colors[np.argmax(counts)])
    rectangles, arrows = _objects(g, background, frame)
    _check(len(arrows) == 1 and len(rectangles) >= 3, "one valve and at least three rectangles are required")
    arrow = arrows[0]
    _check(arrow["color"] not in {r[4] for r in rectangles}, "signal color must be exclusive to the arrow")
    for r0, r1, c0, c1, _ in rectangles:
        _check(1 <= r0 <= r1 < g.shape[0] - 1 and 1 <= c0 <= c1 < g.shape[1] - 1,
               "rectangle touches frame")
        _check((r1 - r0) % 2 == 0 and (c1 - c0) % 2 == 0, "rectangles have odd side lengths")
    clean = g.copy()
    for cell in arrow["cells"]:
        clean[cell] = background
    edges = _edges(rectangles, clean, background)
    adjacency = [[] for _ in rectangles]
    cover = np.zeros(g.shape, dtype=np.int8)
    for k, (_, a, b, (r0, r1, c0, c1)) in enumerate(edges):
        adjacency[a].append((b, k))
        adjacency[b].append((a, k))
        cover[r0:r1 + 1, c0:c1 + 1] += 1
    _check(np.max(cover) <= 1, "implicit bridges must not cross")
    # All components are trees or isolated rectangles, including inactive ones.
    unseen = set(range(len(rectangles)))
    while unseen:
        component = _reach(adjacency, min(unseen), -1)
        _check(sum(len(adjacency[i]) for i in component) == 2 * (len(component) - 1), "network contains a cycle")
        unseen -= component
    cut_candidates = []
    hr, hc = arrow["head"]
    dr, dc = arrow["direction"]
    for k, (axis, a, b, bounds) in enumerate(edges):
        x, y = rectangles[a], rectangles[b]
        if axis == 0 and dr == 0 and 2 * hr == x[0] + x[1]:
            full_gap = (max(x[0], y[0]), min(x[1], y[1]), x[3] + 1, y[2] - 1)
        elif axis == 1 and dc == 0 and 2 * hc == x[2] + x[3]:
            full_gap = (x[1] + 1, y[0] - 1, max(x[2], y[2]), min(x[3], y[3]))
        else:
            continue
        r0, r1, c0, c1 = full_gap
        if all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in arrow["cells"]):
            cut_candidates.append(k)
    _check(len(cut_candidates) == 1, "arrow must lie on exactly one matching network edge")
    cut = cut_candidates[0]
    for k, (_, _, _, (r0, r1, c0, c1)) in enumerate(edges):
        if k != cut:
            _check(not any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in arrow["cells"]),
                   "valve overlaps a different pipe")
    axis, a, b, _ = edges[cut]
    positive_root, negative_root = (b, a) if dr + dc > 0 else (a, b)
    positive = _reach(adjacency, positive_root, cut)
    negative = _reach(adjacency, negative_root, cut)
    _check(not positive & negative, "valve does not separate the network")
    _check(len(positive | negative) >= 3, "active network must have at least three rectangles")
    ports, claims = [], set()
    for i in sorted(positive | negative):
        if len(adjacency[i]) == 1:
            neighbor = adjacency[i][0][0]
            cells = _port(rectangles[i], rectangles[neighbor], g.shape)
            _check(not cells & claims and cells, "terminal frame projections overlap")
            claims |= cells
            ports.append((i, cells, i in positive))
    return dict(grid=g, frame=frame, background=background, arrow=arrow, rectangles=rectangles,
                edges=edges, adjacency=adjacency, cut=cut, positive=positive, negative=negative, ports=ports)


def _render(info):
    out = info["grid"].copy()
    signal = info["arrow"]["color"]
    for k, (_, a, b, (r0, r1, c0, c1)) in enumerate(info["edges"]):
        if k != info["cut"] and a in info["positive"] and b in info["positive"]:
            out[r0:r1 + 1, c0:c1 + 1] = signal
    for _, cells, positive in info["ports"]:
        for cell in cells:
            out[cell] = signal if positive else info["background"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the virtual tree, split it at the arrow, and route both signs."""
    return _render(_analyze(input_grid))


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _bounds(r, c, h, w, color):
    return (r - h // 2, r + h // 2, c - w // 2, c + w // 2, color)


def _draw(shape, rectangles, background, frame):
    g = np.full(shape, background, dtype=np.int8)
    g[0, :] = g[-1, :] = frame
    g[:, 0] = g[:, -1] = frame
    for r0, r1, c0, c1, color in rectangles:
        if not (1 <= r0 <= r1 < shape[0] - 1 and 1 <= c0 <= c1 < shape[1] - 1):
            return None
        if np.any(g[r0 - 1:r1 + 2, c0 - 1:c1 + 2] != background):
            return None
        g[r0:r1 + 1, c0:c1 + 1] = color
    return g


def _tracks(rng, count, length):
    slack = length - (7 + 7 * (count - 1))
    extras = rng.multinomial(slack, np.full(count + 1, 1 / (count + 1)))
    centers = [3 + int(extras[0])]
    for i in range(1, count):
        centers.append(centers[-1] + 7 + int(extras[i]))
    return centers


def _teaching_layout(rng, mode, body, other):
    if mode == "straight":
        widths = [int(rng.choice((3, 5))) for _ in range(4)]
        gaps = [int(rng.integers(1, 4)), int(rng.integers(4, 8)), int(rng.integers(1, 4))]
        used_width = sum(widths) + sum(gaps)
        if used_width > 26:
            return None
        width = int(rng.integers(used_width + 4, 31))
        height = int(rng.integers(15, 28))
        row = 4
        left = int(rng.integers(2, width - used_width - 1))
        rects = []
        for i, w in enumerate(widths):
            rects.append(_bounds(row, left + w // 2, 3, w, body))
            left += w + (gaps[i] if i < 3 else 0)
        # Exactly centered, different-color non-link next to a forward rectangle.
        target = rects[2 + int(rng.integers(0, 2))]
        col = (target[2] + target[3]) // 2
        rects.append(_bounds(int(rng.integers(10, height - 3)), col, 3, 3, other))
        return (height, width), rects, (0, 1), (rects[1], rects[2])
    # A returning fork on the front, a two-exit drain on the rear. This one
    # demonstration teaches branching for both polarities, even at counts 2/1.
    height, width = int(rng.integers(28, 31)), int(rng.integers(28, 31))
    rows, cols = _tracks(rng, 4, height), _tracks(rng, 4, width)
    slots = ((1, 2), (1, 3), (2, 3), (2, 0), (3, 3), (1, 1), (0, 2))
    rects = []
    for i, (ri, ci) in enumerate(slots):
        h = 7 if i == 3 else 5 if i == 2 else 3
        w = 5 if i == 2 else 3
        rects.append(_bounds(rows[ri], cols[ci], h, w, body))
    # Same color but one row off the top leaf and one column off the left leaf.
    # It must stay inactive.
    rects.append(_bounds(rows[0] + 1, cols[0] + 1, 3, 3, body))
    return (height, width), rects, (0, 1), (rects[0], rects[1])


def _random_layout(rng, body, other):
    nr, nc = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    if nr == nc == 2:
        if rng.random() < .5:
            nr = 3
        else:
            nc = 3
    height = int(rng.integers(7 + 7 * (nr - 1), 31))
    width = int(rng.integers(7 + 7 * (nc - 1), 31))
    rows, cols = _tracks(rng, nr, height), _tracks(rng, nc, width)
    n = int(rng.integers(4, min(7, nr + nc - 1) + 1))
    slots = sorted(int(x) for x in rng.choice(nr * nc, n, replace=False))
    rects = []
    for slot in slots:
        ri, ci = divmod(slot, nc)
        r, c = rows[ri], cols[ci]
        hs = [s for s in (3, 5, 7) if r - s // 2 >= 2 and r + s // 2 <= height - 3]
        ws = [s for s in (3, 5, 7) if c - s // 2 >= 2 and c + s // 2 <= width - 3]
        rects.append(_bounds(r, c, int(rng.choice(hs)), int(rng.choice(ws)), body))
    return (height, width), rects


def _place_arrow(rng, g, rectangles, background, signal, direction=None, forced_endpoints=None,
                 interior_cut=False, allowed_shapes=None):
    edges = _edges(rectangles, g, background)
    adjacency = [[] for _ in rectangles]
    for k, (_, a, b, _) in enumerate(edges):
        adjacency[a].append((b, k))
        adjacency[b].append((a, k))
    if forced_endpoints is None:
        component = _reach(adjacency, 0, -1)
        if len(component) != len(rectangles) or len(edges) != len(rectangles) - 1:
            return None
    candidates = list(range(len(edges)))
    rng.shuffle(candidates)
    for cut in candidates:
        axis, a, b, _ = edges[cut]
        x, y = rectangles[a], rectangles[b]
        if forced_endpoints is not None and {x, y} != set(forced_endpoints):
            continue
        if interior_cut and min(len(_reach(adjacency, a, cut)), len(_reach(adjacency, b, cut))) < 2:
            continue
        forward = direction if direction is not None else ((0, 1) if axis == 0 else (1, 0))
        if direction is None and rng.random() < .5:
            forward = (-forward[0], -forward[1])
        lo, hi = (x[3] + 1, y[2] - 1) if axis == 0 else (x[1] + 1, y[0] - 1)
        options = []
        for shaft in (0, 2, 3, 4, 5, 6):
            for wing in (1, 2):
                if wing == 2 and shaft == 0:
                    continue
                if allowed_shapes is not None and (shaft, wing) not in allowed_shapes:
                    continue
                for along in range(lo + 1, hi):
                    head = ((x[0] + x[1]) // 2, along) if axis == 0 else (along, (x[2] + x[3]) // 2)
                    cells = _arrow_cells(head, forward, shaft, wing)
                    if not all(0 <= r < g.shape[0] and 0 <= c < g.shape[1] and g[r, c] == background for r, c in cells):
                        continue
                    axial = [c if axis == 0 else r for r, c in cells]
                    perp = [r if axis == 0 else c for r, c in cells]
                    low_perp = max(x[0], y[0]) if axis == 0 else max(x[2], y[2])
                    high_perp = min(x[1], y[1]) if axis == 0 else min(x[3], y[3])
                    if min(axial) >= lo + 1 and max(axial) <= hi - 1 and min(perp) >= low_perp and max(perp) <= high_perp:
                        options.append(cells)
        if options:
            result = g.copy()
            for cell in options[int(rng.integers(len(options)))]:
                result[cell] = signal
            return result
    return None


def _add_inert(rng, g, background, frame, body, other):
    # Additional objects share the scene rather than occupying fixed panels.
    base = _analyze(g)
    def edge_keys(info):
        rs = info["rectangles"]
        return {(axis, rs[a], rs[b], bounds) for axis, a, b, bounds in info["edges"]}
    original_edges = edge_keys(base)
    for _ in range(int(rng.integers(0, 4))):
        for _ in range(20):
            h, w = int(rng.choice((3, 5))), int(rng.choice((3, 5)))
            r, c = int(rng.integers(2, g.shape[0] - 2)), int(rng.integers(2, g.shape[1] - 2))
            r0, r1, c0, c1, color = _bounds(r, c, h, w, int(rng.choice((body, other))))
            if not (2 <= r0 <= r1 <= g.shape[0] - 3 and 2 <= c0 <= c1 <= g.shape[1] - 3):
                continue
            if np.any(g[r0 - 1:r1 + 2, c0 - 1:c1 + 2] != background):
                continue
            candidate = g.copy()
            candidate[r0:r1 + 1, c0:c1 + 1] = color
            try:
                info = _analyze(candidate)
            except AssertionError:
                continue
            index = info["rectangles"].index((r0, r1, c0, c1, color))
            if info["adjacency"][index] or edge_keys(info) != original_edges:
                continue
            g = candidate
            break
    return g


def _features(info):
    rects, edges = info["rectangles"], info["edges"]
    p, n = info["positive"], info["negative"]
    positive_edges = [e for k, e in enumerate(edges) if k != info["cut"] and e[1] in p and e[2] in p]
    negative_edges = [e for k, e in enumerate(edges) if k != info["cut"] and e[1] in n and e[2] in n]
    thicknesses = {e[3][1] - e[3][0] + 1 if e[0] == 0 else e[3][3] - e[3][2] + 1 for e in positive_edges}
    dr, dc = info["arrow"]["direction"]
    hr, hc = info["arrow"]["head"]
    behind = any(((rects[i][0] + rects[i][1]) / 2 - hr) * dr +
                 ((rects[i][2] + rects[i][3]) / 2 - hc) * dc < 0 for i in p)
    branch = any(sum(j in p for j, _ in info["adjacency"][i]) >= 3 for i in p)
    internal_turn = any(len({e[0] for k, e in enumerate(edges) if k != info["cut"] and
                             e[1] in side and e[2] in side}) == 2 for side in (p, n))
    cut_edge = edges[info["cut"]]
    cut_degrees = tuple(sorted((len(info["adjacency"][cut_edge[1]]), len(info["adjacency"][cut_edge[2]]))))
    return dict(positive_edges=len(positive_edges), negative_edges=len(negative_edges),
                thicknesses=thicknesses, behind=behind, branch=branch,
                internal_turn=internal_turn, cut_degrees=cut_degrees)


def _make_pair(rng, mode, palette, allowed_shapes=None):
    background, frame, signal, body, other = palette
    for _ in range(1600):
        if mode in ("straight", "returning"):
            layout = _teaching_layout(rng, mode, body, other)
            if layout is None:
                continue
            shape, rectangles, direction, endpoints = layout
        else:
            shape, rectangles = _random_layout(rng, body, other)
            direction = endpoints = None
        g = _draw(shape, rectangles, background, frame)
        if g is None:
            continue
        g = _place_arrow(rng, g, rectangles, background, signal, direction, endpoints, mode == "test", allowed_shapes)
        if g is None:
            continue
        try:
            info = _analyze(g)
        except AssertionError:
            continue
        features = _features(info)
        if not features["positive_edges"]:
            continue
        if mode == "returning" and not (features["behind"] and features["branch"] and max(features["thicknesses"], default=0) >= 3):
            continue
        transfer_case = features["cut_degrees"] == (2, 2) and features["internal_turn"]
        if mode == "test" and not transfer_case:
            continue
        if mode == "free" and transfer_case:
            continue
        if mode not in ("straight", "returning"):
            g = _add_inert(rng, g, background, frame, body, other)
        # Every example may rotate/reflect; source direction is never tied to color.
        g = np.rot90(g, int(rng.integers(0, 4)))
        if rng.random() < .5:
            g = np.fliplr(g)
        g = g.copy()
        return {"input": g, "output": solve(g)}
    raise RuntimeError(f"could not construct {mode} valve network")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    roles = tuple(int(x) for x in rng.choice(10, 3, replace=False))
    remaining = [c for c in range(10) if c not in roles]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            mode = ("straight" if i == 0 else "returning" if i == 1 else "free") if split == "train" else "test"
            for _ in range(20):
                body, other = (int(x) for x in rng.choice(remaining, 2, replace=False))
                allowed_shapes = None
                if split == "test":
                    allowed_shapes = set()
                    for demo in puzzle["train"]:
                        arrow = _analyze(demo["input"])["arrow"]
                        allowed_shapes.add((arrow["shaft"], arrow["wing"]))
                pair = _make_pair(rng, mode, (*roles, body, other), allowed_shapes)
                if split == "train" and i == 1 and _analyze(pair["input"])["arrow"]["direction"] == _analyze(puzzle["train"][0]["input"])["arrow"]["direction"]:
                    pair = {key: np.rot90(value).copy() for key, value in pair.items()}
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _near_miss(info):
    rs = info["rectangles"]
    active = info["positive"] | info["negative"]
    for i, a in enumerate(rs):
        if i in active:
            continue
        for j in active:
            b = rs[j]
            if a[4] == b[4] and (
                (abs(a[0] + a[1] - b[0] - b[1]) == 2 and (a[3] < b[2] or b[3] < a[2])) or
                (abs(a[2] + a[3] - b[2] - b[3]) == 2 and (a[1] < b[0] or b[1] < a[0]))):
                return True
    return False


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _check(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, roles, training = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            g, out = _grid(pair["input"]), _grid(pair["output"])
            info = _analyze(g)
            _check(g.shape == out.shape and np.array_equal(out, _render(info)), "incorrect transformation")
            key = (g.shape, g.tobytes())
            _check(key not in seen, "duplicate input")
            seen.add(key)
            roles.add((info["background"], info["frame"], info["arrow"]["color"]))
            # Check the two kinds of change separately, using exact cell sets.
            bridge_cells = set()
            for k, (_, a, b, (r0, r1, c0, c1)) in enumerate(info["edges"]):
                if k != info["cut"] and a in info["positive"] and b in info["positive"]:
                    bridge_cells.update((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
            port_cells = set().union(*(cells for _, cells, _ in info["ports"]))
            changed = {tuple(int(x) for x in p) for p in np.argwhere(g != out)}
            _check(changed == bridge_cells | port_cells, "change footprint is not bridges plus terminal ports")
            _check(all(g[cell] == info["background"] for cell in bridge_cells), "bridge overwrites an object")
            _check(all(out[cell] == g[cell] for cell in info["arrow"]["cells"]), "valve was modified")
            _check(any(p for _, _, p in info["ports"]) and any(not p for _, _, p in info["ports"]), "both polarities need an exit")
            _check(set(np.unique(out)) <= set(np.unique(g)), "output invented an unencoded color")
            if split == "train":
                training.append(info)
            else:
                _check(min(len(info["positive"]), len(info["negative"])) >= 2, "test must move the valve into the network interior")
                f = _features(info)
                _check(f["cut_degrees"] == (2, 2) and f["internal_turn"], "test lacks the intended routing transfer")
    _check(len(roles) == 1, "background, frame, and signal roles change within a puzzle")
    _check(len({info["arrow"]["direction"] for info in training}) >= 2, "training must show geometric direction rather than a color-direction code")
    features = [_features(i) for i in training]
    _check(any(f["positive_edges"] and f["negative_edges"] for f in features), "training must teach propagation on both sides")
    _check(any(f["behind"] and f["branch"] for f in features), "training must teach a returning fork")
    _check(1 in set().union(*(f["thicknesses"] for f in features)) and
           any(max(f["thicknesses"], default=0) >= 3 for f in features), "training must teach thin and thick inset bridges")
    _check(any(_near_miss(i) for i in training), "training lacks an exact-center negative")
    _check(any(sum(not p for _, _, p in i["ports"]) >= 2 and sum(p for _, _, p in i["ports"]) >= 2
               for i in training), "training must show branching exits for both signs")
    aperture_witness = False
    for info in training:
        for i, cells, positive in info["ports"]:
            if positive:
                edge = info["edges"][info["adjacency"][i][0][1]]
                r0, r1, c0, c1 = edge[3]
                incoming_width = r1 - r0 + 1 if edge[0] == 0 else c1 - c0 + 1
                aperture_witness |= len(cells) != incoming_width
    _check(aperture_witness, "training must distinguish terminal aperture from incoming bridge width")
    _check(all(not (_features(i)["cut_degrees"] == (2, 2) and _features(i)["internal_turn"]) for i in training),
           "training has consumed the reserved test combination")
    taught_shapes = {(info["arrow"]["shaft"], info["arrow"]["wing"]) for info in training}
    for pair in puzzle["test"]:
        arrow = _analyze(pair["input"])["arrow"]
        _check((arrow["shaft"], arrow["wing"]) in taught_shapes, "test introduces an untaught arrow geometry")
    mixed_witness = False
    for info in training:
        clean = info["grid"].copy()
        for cell in info["arrow"]["cells"]:
            clean[cell] = info["background"]
        for _, a, b, _ in _edges(info["rectangles"], clean, info["background"], False):
            if info["rectangles"][a][4] != info["rectangles"][b][4] and (a in info["positive"] or b in info["positive"]):
                mixed_witness = True
    _check(mixed_witness, "training lacks a color-matching negative")
    # Explicit parent ablations: pure projection misses a positive bridge;
    # direction-free B would fill the negative bridge shown in the first demo.
    _check(any(f["positive_edges"] > 0 for f in features), "A-only projection could match training")
    _check(any(f["negative_edges"] > 0 for f in features), "B-only undirected connection could match training")
    for info in training:
        original, signal = info["grid"], info["arrow"]["color"]
        a_only, b_only = original.copy(), original.copy()
        hr, hc = info["arrow"]["head"]
        dr, dc = info["arrow"]["direction"]
        tip = (original.shape[0] - 1 if dr > 0 else 0, hc) if dr else (hr, original.shape[1] - 1 if dc > 0 else 0)
        tail = (0 if dr > 0 else original.shape[0] - 1, hc) if dr else (hr, 0 if dc > 0 else original.shape[1] - 1)
        a_only[tip], a_only[tail] = signal, info["background"]
        for _, _, _, (r0, r1, c0, c1) in info["edges"]:
            b_only[r0:r1 + 1, c0:c1 + 1] = signal
        for cell in info["arrow"]["cells"]:
            b_only[cell] = signal
        _check(not np.array_equal(a_only, _render(info)), "parent-A-only shortcut survived")
        _check(not np.array_equal(b_only, _render(info)), "parent-B-only shortcut survived")
    return None


def self_test():
    """Bounded 42-seed diversity and geometric metamorphic checks."""
    shapes, layouts, counts, leaves, cuts, shafts, wings, colors, sizes, directions = (set() for _ in range(10))
    samples = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g, out = pair["input"], pair["output"]
                info = _analyze(g)
                samples += 1
                shapes.add(g.shape)
                counts.add(len(info["rectangles"]))
                leaves.add(len(info["ports"]))
                cuts.add((len(info["positive"]), len(info["negative"])))
                shafts.add(info["arrow"]["shaft"])
                wings.add(info["arrow"]["wing"])
                colors.add((info["background"], info["frame"], info["arrow"]["color"]))
                directions.add(info["arrow"]["direction"])
                sizes.update((r[1] - r[0] + 1, r[3] - r[2] + 1) for r in info["rectangles"])
                layouts.add((g.shape, tuple(r[:4] for r in info["rectangles"]), info["arrow"]["head"], info["arrow"]["direction"]))
                # Rotation, reflection, and an arbitrary bijective palette change
                # must commute with the transformation.
                _check(np.array_equal(solve(np.rot90(g)), np.rot90(out)), "rotation covariance failed")
                if split == "test":
                    _check(np.array_equal(solve(np.fliplr(g)), np.fliplr(out)), "reflection covariance failed")
                    permutation = np.random.default_rng(seed + 5000).permutation(10)
                    _check(np.array_equal(solve(permutation[g]), permutation[out]), "palette covariance failed")
                    # Reverse the valve, preserving its axis and full network.
                    reversed_grid = g.copy()
                    for cell in info["arrow"]["cells"]:
                        reversed_grid[cell] = info["background"]
                    axis, a, b, _ = info["edges"][info["cut"]]
                    x, y = info["rectangles"][a], info["rectangles"][b]
                    lo, hi = (x[3] + 1, y[2] - 1) if axis == 0 else (x[1] + 1, y[0] - 1)
                    head = ((x[0] + x[1]) // 2, (lo + hi) // 2) if axis == 0 else ((lo + hi) // 2, (x[2] + x[3]) // 2)
                    reverse = tuple(-d for d in info["arrow"]["direction"])
                    # A compact valve always fits in the original cut gap.
                    if (reverse[1] > 0 and head[1] == hi) or (reverse[0] > 0 and head[0] == hi):
                        head = (head[0] - reverse[0], head[1] - reverse[1])
                    for cell in _arrow_cells(head, reverse):
                        reversed_grid[cell] = info["arrow"]["color"]
                    rev_info = _analyze(reversed_grid)
                    _check(rev_info["positive"] == info["negative"] and rev_info["negative"] == info["positive"], "valve reversal did not swap sides")
    _check(samples == 210 and len(layouts) >= 200, "insufficient structural diversity")
    _check(len(shapes) >= 70 and any(30 in s for s in shapes), "insufficient canvas variation")
    _check(len(counts) >= 5 and len(leaves) >= 3 and len(cuts) >= 8, "insufficient network variation")
    _check(len(shafts) >= 3 and directions == set(DIRECTIONS) and len(colors) >= 35 and len(sizes) >= 6,
           "insufficient geometry/palette variation")
    a, b = generate_puzzle(731, 7, 3), generate_puzzle(731, 7, 3)
    _check(all(np.array_equal(x[k], y[k]) for s in ("train", "test") for x, y in zip(a[s], b[s]) for k in ("input", "output")), "nondeterminism")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=samples, shapes=len(shapes), layouts=len(layouts), rectangle_counts=sorted(counts),
                leaf_counts=sorted(leaves), valve_splits=len(cuts), shafts=sorted(shafts), wing_sizes=sorted(wings), palettes=len(colors))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
