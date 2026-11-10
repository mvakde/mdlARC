"""Port-fed stencil networks: visible macro plans, direct keys, seam reachability.

Only Python's standard library and NumPy are used. No parent code, examples,
training labels, seed, or sampled parameters are consulted by solve().
"""
from collections import deque
import numpy as np


_ARROW = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)
_SIDES = ("top", "right", "bottom", "left")
_COUNTERFACTUALS = ("all", "solid", "isolated", "same_source", "diagonal", "no_key", "whole_tile")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _transform(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _box(cells):
    a = np.asarray(cells)
    return (int(a[:, 0].min()), int(a[:, 0].max()) + 1,
            int(a[:, 1].min()), int(a[:, 1].max()) + 1)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], []
        while stack:
            r, c = stack.pop()
            group.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _array_key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple((split, _array_key(p["input"]), _array_key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def _draw_frame(grid, r, c, h, w, color):
    grid[r, c:c+w] = grid[r+h-1, c:c+w] = color
    grid[r:r+h, c] = grid[r:r+h, c+w-1] = color


def _port_position(side, index, h, w):
    if side == "top":
        return 0, index+1
    if side == "bottom":
        return h+1, index+1
    if side == "left":
        return index+1, 0
    return index+1, w+1


def _adjacent_macro(side, index, h, w):
    if side == "top":
        return 0, index
    if side == "bottom":
        return h-1, index
    if side == "left":
        return index, 0
    return index, w-1


def _parse(input_grid):
    """Parse the complete visible convention, without generation metadata."""
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(bool(np.all(grid[0] == bg) and np.all(grid[-1] == bg)
                  and np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)),
             "input needs its blank outer rim")
    frames = []
    for f in map(int, np.unique(grid)):
        if f == bg:
            continue
        cells = np.argwhere(grid == f)
        r, e, c, d = _box(cells)
        h, w = e-r, d-c
        if not (4 <= h <= 9 and 4 <= w <= 9):
            continue
        ring = np.zeros(grid.shape, dtype=bool)
        ring[r, c:d] = ring[e-1, c:d] = True
        ring[r:e, c] = ring[r:e, d-1] = True
        if np.any((grid == f) & ~ring):
            continue
        others = np.unique(grid[ring & (grid != f)])
        if len(others) != 1:
            continue
        port = int(others[0])
        if port == bg or np.any((grid == port) & ~ring):
            continue
        count = int(np.sum(grid == port))
        if not (1 <= count <= 4 and len(cells) > count):
            continue
        if not all(grid[a, b] == f for a, b in ((r, c), (r, d-1), (e-1, c), (e-1, d-1))):
            continue
        frames.append((r, e, c, d, f, port))
    _require(len(frames) == 1, "need one unique port-bearing frame")
    r, e, c, d, frame, port = frames[0]
    macro = grid[r+1:e-1, c+1:d-1].copy()
    h, w = macro.shape
    sources = sorted(map(int, np.unique(macro[macro != bg])))
    _require(1 <= len(sources) <= 4 and not set(sources) & {frame, port}, "invalid macro identities")
    ports = []
    for j in range(w):
        if grid[r, c+j+1] == port:
            ports.append(("top", j))
        if grid[e-1, c+j+1] == port:
            ports.append(("bottom", j))
    for i in range(h):
        if grid[r+i+1, c] == port:
            ports.append(("left", i))
        if grid[r+i+1, d-1] == port:
            ports.append(("right", i))
    _require(len(ports) == int(np.sum(grid == port)), "invalid port positions")
    for side, index in ports:
        _require(macro[_adjacent_macro(side, index, h, w)] != bg,
                 "generated family ports border occupied macro cells")

    exterior = grid != bg
    exterior[r:e, c:d] = False
    keys, key_boxes = {}, {}
    key_pixels = np.zeros(grid.shape, dtype=bool)
    arrow_shapes = [_transform(_ARROW, code) for code in range(8)]
    for cells in _components(exterior):
        values = {int(grid[a, b]) for a, b in cells}
        if len(values) == 1:
            continue
        a, b, x, y = _box(cells)
        tile = grid[a:b, x:y]
        _require(len(values) == 2 and len(cells) == 12 and tile.size == 12,
                 "exterior multicolor object must be a filled key")
        counts = {v: int(np.sum(tile == v)) for v in values}
        _require(sorted(counts.values()) == [5, 7], "key must have seven source and five target pixels")
        source = next(v for v in values if counts[v] == 7)
        target = next(v for v in values if counts[v] == 5)
        _require(any(m.shape == tile.shape and np.array_equal(m, tile == source) for m in arrow_shapes),
                 "unknown key arrow")
        _require(source in sources and source not in keys and target not in (bg, frame, port),
                 "key is not a functional direct mapping")
        keys[source], key_boxes[source] = target, (a, b, x, y)
        key_pixels[a:b, x:y] = True
    _require(set(keys) == set(sources), "missing key for a macro identity")
    remaining = exterior & ~key_pixels
    _require(set(map(int, np.unique(grid[remaining]))) == set(sources), "unexplained exterior pixels")
    masks, mask_boxes, sizes = {}, {}, set()
    for source in sources:
        a, b, x, y = _box(np.argwhere(remaining & (grid == source)))
        n = b-a
        _require(2 <= n <= 5 and n == y-x, "stencil must recover a square of side 2..5")
        tile = grid[a:b, x:y]
        _require(bool(np.all(np.isin(tile, [bg, source]))), "stencil bounding square overlaps a clue")
        mask = tile == source
        _require(bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()),
                 "stencil must expose all four bounds")
        masks[source], mask_boxes[source] = mask.copy(), (a, b, x, y)
        sizes.add(n)
    _require(len(sizes) == 1, "one common stencil size is required")
    n = next(iter(sizes))
    _require(h*n+2 <= 30 and w*n+2 <= 30, "expanded board exceeds 30")
    boxes = [(r, e, c, d), *mask_boxes.values(), *key_boxes.values()]
    for i, (a, b, x, y) in enumerate(boxes):
        for aa, bb, xx, yy in boxes[i+1:]:
            _require(b < aa or bb < a or y < xx or yy < x,
                     "complete clue rectangles need a blank separating cell")
    return dict(background=bg, frame=frame, port=port, macro=macro, n=n,
                sources=sources, masks=masks, mapping=keys, ports=ports,
                frame_box=(r, e, c, d), mask_boxes=mask_boxes, key_boxes=key_boxes,
                stencil_order=sorted(sources, key=lambda s: (mask_boxes[s][0], mask_boxes[s][2])))


def _expanded_border(scene):
    h, w = scene["macro"].shape
    n = scene["n"]
    output = np.full((h*n+2, w*n+2), scene["background"], dtype=np.int8)
    _draw_frame(output, 0, 0, *output.shape, scene["frame"])
    for side, index in scene["ports"]:
        start, end = 1+index*n, 1+(index+1)*n
        if side in ("top", "bottom"):
            output[0 if side == "top" else -1, start:end] = scene["port"]
        else:
            output[start:end, 0 if side == "left" else -1] = scene["port"]
    return output


def _expanded_sources(scene, alternative=None, stamp_transform=0):
    macro, n, bg = scene["macro"], scene["n"], scene["background"]
    original = np.full((macro.shape[0]*n, macro.shape[1]*n), bg, dtype=np.int8)
    sources, order = scene["sources"], scene.get("stencil_order", scene["sources"])
    for source in sources:
        chosen = order[sources.index(source)] if alternative == "position" else source
        mask = (np.ones((n, n), bool) if alternative == "solid"
                else _transform(scene["masks"][chosen], stamp_transform))
        original[np.kron(macro == source, mask)] = chosen
    return original


def _flood(original, border, scene, alternative=None):
    bg, p, n = scene["background"], scene["port"], scene["n"]
    active = original != bg
    seeds = np.zeros(active.shape, dtype=bool)
    seeds[0] |= border[0, 1:-1] == p
    seeds[-1] |= border[-1, 1:-1] == p
    seeds[:, 0] |= border[1:-1, 0] == p
    seeds[:, -1] |= border[1:-1, -1] == p
    live = seeds & active
    queue = deque(map(tuple, np.argwhere(live).tolist()))
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if alternative == "diagonal":
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    h, w = active.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in offsets:
            a, b = r+dr, c+dc
            if not (0 <= a < h and 0 <= b < w) or live[a, b] or not active[a, b]:
                continue
            if alternative == "isolated" and (r//n, c//n) != (a//n, b//n):
                continue
            if alternative == "same_source" and original[r, c] != original[a, b]:
                continue
            live[a, b] = True
            queue.append((a, b))
    return live


def _render(scene, alternative=None, stamp_transform=0):
    output = _expanded_border(scene)
    original = _expanded_sources(scene, alternative, stamp_transform)
    live = (original != scene["background"] if alternative == "all"
            else _flood(original, output, scene, alternative))
    if alternative == "whole_tile":
        n = scene["n"]
        for r in range(0, original.shape[0], n):
            for c in range(0, original.shape[1], n):
                if live[r:r+n, c:c+n].any():
                    live[r:r+n, c:c+n] = original[r:r+n, c:c+n] != scene["background"]
    interior = output[1:-1, 1:-1]
    for source in scene["sources"]:
        target = scene["mapping"][source]
        if alternative == "no_key":
            target = source
        elif alternative == "two_maps":
            target = scene["mapping"].get(target, target)
        interior[live & (original == source)] = target
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct an enlarged, powered board using this input alone."""
    return _render(_parse(input_grid))


def _coordinate_graph_output(scene):
    """Validator reconstruction via tile-local coordinates and a set graph.

    This is a second generator-owned check, not an independent solver claim.
    It does not use the array expansion or the renderer's flood-fill function.
    """
    n, bg = scene["n"], scene["background"]
    h, w = scene["macro"].shape
    nodes = {}
    for mr in range(h):
        for mc in range(w):
            source = int(scene["macro"][mr, mc])
            if source == bg:
                continue
            for lr, lc in np.argwhere(scene["masks"][source]):
                nodes[(mr*n+int(lr)+1, mc*n+int(lc)+1)] = source
    seeds = set()
    for side, index in scene["ports"]:
        for offset in range(n):
            q = ((1 if side == "top" else h*n, 1+index*n+offset)
                 if side in ("top", "bottom") else
                 (1+index*n+offset, 1 if side == "left" else w*n))
            if q in nodes:
                seeds.add(q)
    visited, stack = set(seeds), list(seeds)
    while stack:
        r, c = stack.pop()
        for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if q in nodes and q not in visited:
                visited.add(q)
                stack.append(q)
    result = _expanded_border(scene)
    for q in visited:
        result[q] = scene["mapping"][nodes[q]]
    return result


def _sample_mask(rng, n, informative=False):
    yy, xx = np.indices((n, n))
    kind = 0 if informative else int(rng.integers(9))
    if kind == 0:
        mask = rng.random((n, n)) < rng.uniform(.32, .72)
    elif kind == 1:
        mask = (xx == 0) | (xx == n-1) | (yy == 0) | (yy == n-1)
    elif kind == 2:
        mask = (xx == n//2) | (yy == n//2)
    elif kind == 3:
        mask = xx == yy
        if rng.random() < .5:
            mask |= xx+yy == n-1
    elif kind == 4:
        mask = (xx+yy) % 2 == int(rng.integers(2))
    elif kind == 5:
        mask = yy % 2 == 0
    elif kind == 6:
        mask = np.ones((n, n), dtype=bool)
    elif kind == 7:
        mask = rng.random((n, n)) < rng.uniform(.75, .95)
    else:
        mask = np.zeros((n, n), dtype=bool)
        for _ in range(int(rng.integers(2, n+2))):
            mask[int(rng.integers(n)), int(rng.integers(n))] = True
    for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]):
        if not edge.any():
            edge[int(rng.integers(n))] = True
    return _transform(mask, int(rng.integers(8)))


def _sample_macro(rng, h, w, sources, bg, informative=False):
    k, total = len(sources), h*w
    count = int(rng.integers(max(k, (total*3)//4), total+1)) if informative else int(rng.integers(k, total+1))
    yy, xx = np.indices((h, w))
    kind = int(rng.integers(6))
    if kind == 0 or informative:
        order = rng.permutation(total)
    elif kind == 1:
        order = np.array([r*w+c for r in range(h)
                          for c in (range(w) if r % 2 == 0 else range(w-1, -1, -1))])
    elif kind == 2:
        cr, cc = int(rng.integers(h)), int(rng.integers(w))
        order = np.argsort((abs(yy-cr)+abs(xx-cc)+rng.random((h, w))*.2).ravel())
    elif kind == 3:
        depth = np.minimum.reduce([yy, xx, h-1-yy, w-1-xx])
        order = np.argsort((depth+rng.random((h, w))*.2).ravel())
    elif kind == 4:
        order = np.argsort(((xx+yy) % 2+rng.random((h, w))*.2).ravel())
    else:
        order = np.argsort((yy+xx+rng.random((h, w))*.2).ravel())
    weights = rng.dirichlet(np.full(k, rng.uniform(.4, 2)))
    assignments = list(sources) + list(map(int, rng.choice(sources, count-k, p=weights)))
    if kind == 0 or informative or rng.random() < .4:
        rng.shuffle(assignments)
    else:
        assignments = sorted(assignments, key=lambda s: sources.index(s))
    macro = np.full((h, w), bg, dtype=np.int8)
    macro.flat[order[:count]] = assignments
    return macro


def _sample_mapping(rng, sources, pool, kind):
    outside = [v for v in pool if v not in sources]
    if kind == "disjoint" or len(sources) == 1:
        return {s: int(rng.choice(outside)) for s in sources}
    if kind == "chain":
        order = list(map(int, rng.permutation(sources)))
        return {s: order[i+1] if i+1 < len(order) else int(rng.choice(outside))
                for i, s in enumerate(order)}
    if kind == "cycle":
        order = list(map(int, rng.permutation(sources)))
        return {s: order[(i+1) % len(order)] for i, s in enumerate(order)}
    if kind == "many":
        target = int(rng.choice(pool))
        return {s: target if target != s else int(rng.choice(outside)) for s in sources}
    return {s: int(rng.choice([v for v in pool if v != s])) for s in sources}


def _sample_scene(rng, n, h, w, k, port_count, allowed_sides, informative=False, chain=False):
    colors = list(map(int, rng.permutation(10)))
    bg, frame, port = colors[:3]
    sources = sorted(colors[3:3+k])
    pool = colors[3:]
    masks = {}
    for i, source in enumerate(sources):
        if i and not informative and rng.random() < .15:
            masks[source] = masks[int(rng.choice(sources[:i]))].copy()
        else:
            masks[source] = _sample_mask(rng, n, informative)
    kind = "chain" if chain else "disjoint" if informative else str(rng.choice(["disjoint", "chain", "cycle", "many", "mixed"]))
    mapping = _sample_mapping(rng, sources, pool, kind)
    macro = _sample_macro(rng, h, w, sources, bg, informative)
    options = [(side, j) for side in allowed_sides
               for j in range(w if side in ("top", "bottom") else h)]
    chosen = [options[int(i)] for i in rng.choice(len(options), port_count, replace=False)]
    # The first port is guaranteed to power the head of the chain witness.
    head = next((s for s in sources if s not in mapping.values()), sources[0])
    for i, (side, index) in enumerate(chosen):
        r, c = _adjacent_macro(side, index, h, w)
        if macro[r, c] == bg:
            macro[r, c] = int(rng.choice(sources))
        if chain and i == 0:
            macro[r, c] = head
    # Re-establish categories displaced by a head placement, without erasing
    # a singleton category or a port's required occupied neighbor.
    for source in sources:
        if not np.any(macro == source):
            counts = {s: int(np.sum(macro == s)) for s in sources}
            protected = {_adjacent_macro(side, j, h, w) for side, j in chosen}
            available = [(r, c) for r in range(h) for c in range(w)
                         if (r, c) not in protected and
                         (macro[r, c] == bg or counts[int(macro[r, c])] > 1)]
            if not available:
                return None
            r, c = available[int(rng.integers(len(available)))]
            macro[r, c] = source
    return dict(background=bg, frame=frame, port=port, macro=macro, n=n,
                sources=sources, masks=masks, mapping=mapping, ports=chosen)


def _available(occupied, h, w):
    if h > occupied.shape[0] or w > occupied.shape[1]:
        return np.empty((0, 2), dtype=int)
    summed = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = summed[h:, w:]-summed[:-h, w:]-summed[h:, :-w]+summed[:-h, :-w]
    return np.argwhere(sums == 0)


def _reserve(occupied, r, c, h, w):
    occupied[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True


def _pack_scene(rng, scene):
    n, sources = scene["n"], scene["sources"]
    mh, mw = scene["macro"].shape
    board = np.full((mh+2, mw+2), scene["background"], dtype=np.int8)
    _draw_frame(board, 0, 0, *board.shape, scene["frame"])
    board[1:-1, 1:-1] = scene["macro"]
    for side, j in scene["ports"]:
        board[_port_position(side, j, mh, mw)] = scene["port"]
    objects = [("board", None, board)]
    for source in sources:
        stencil = np.where(scene["masks"][source], source, scene["background"]).astype(np.int8)
        key = _transform(np.where(_ARROW, source, scene["mapping"][source]).astype(np.int8), int(rng.integers(8)))
        objects.extend([("stencil", source, stencil), ("key", source, key)])
    for attempt in range(40):
        # Some compact, some large rectangular canvases; the final attempts
        # use ample space but retain randomized object positions and orders.
        height, width = (30, 30) if attempt >= 12 else tuple(map(int, rng.integers(17, 31, size=2)))
        occupied = np.zeros((height, width), dtype=bool)
        occupied[[0, -1], :] = True
        occupied[:, [0, -1]] = True
        # Random ties give independent stencil/key ordering even in compact packing.
        priorities = rng.random(len(objects))
        order = sorted(range(len(objects)), key=lambda i: (-objects[i][2].size, priorities[i]))
        positions = {}
        for i in order:
            tile = objects[i][2]
            candidates = _available(occupied, *tile.shape)
            if not len(candidates):
                break
            r, c = map(int, candidates[int(rng.integers(len(candidates)))])
            positions[i] = r, c
            _reserve(occupied, r, c, *tile.shape)
        if len(positions) != len(objects):
            continue
        grid = np.full((height, width), scene["background"], dtype=np.int8)
        for i, (_, _, tile) in enumerate(objects):
            r, c = positions[i]
            grid[r:r+tile.shape[0], c:c+tile.shape[1]] = tile
        # Always retain at least one blank cell around the actual clue extent.
        r, e, c, d = _box(np.argwhere(grid != scene["background"]))
        r, c = max(0, r-int(rng.integers(1, 4))), max(0, c-int(rng.integers(1, 4)))
        e, d = min(height, e+int(rng.integers(1, 4))), min(width, d+int(rng.integers(1, 4)))
        return grid[r:e, c:d].copy()
    raise RuntimeError("bounded clue packing exhausted")


def _strong_witness(scene):
    correct = _render(scene)
    return (all(not np.array_equal(correct, _render(scene, alternative)) for alternative in _COUNTERFACTUALS)
            and all(not np.array_equal(correct, _render(scene, stamp_transform=t)) for t in range(1, 8)))


def _ports_witness(scene):
    """No single port alone explains the multiport demonstration's interior."""
    correct = _render(scene)[1:-1, 1:-1]
    return len(scene["ports"]) > 1 and all(
        not np.array_equal(correct, _render(dict(scene, ports=[port]))[1:-1, 1:-1])
        for port in scene["ports"])


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
                 and low <= int(value) <= high, f"{name} is not a legal integer count")
    _require(num_train > num_test, "num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _schedule(rng, train_count, test_count):
    regime = int(rng.integers(4))
    side_order = [str(x) for x in rng.permutation(_SIDES)]
    records = []
    for split, count in (("train", train_count), ("test", test_count)):
        for i in range(count):
            training = split == "train"
            strong, chain = training and i == 0, training and i == 1
            n = (int(rng.choice([3, 4])) if training else 5) if regime == 0 else int(rng.integers(2, 6))
            if strong:
                n = int(rng.choice([3, 4]))
            limit = min(7, 28//n)
            if regime == 1:
                h, w = map(int, rng.integers(3 if strong else 2, 5, size=2))
                if not training:
                    if rng.random() < .5:
                        h = int(rng.integers(5, limit+1))
                    else:
                        w = int(rng.integers(5, limit+1))
            else:
                h, w = map(int, rng.integers(3 if strong else 2, min(4, limit)+1 if strong else limit+1, size=2))
            k = 3 if strong else int(rng.integers(2 if chain else 1, 5))
            k = min(k, h*w)
            port_count = 1 if strong else 2 if chain else int(rng.integers(1, 5))
            if regime == 3:
                port_count = (1 if i == 0 else 2) if training else int(rng.integers(3, 5))
            allowed = side_order[:3] if regime == 2 and training else side_order[3:] if regime == 2 else list(_SIDES)
            port_count = min(port_count, sum(w if side in ("top", "bottom") else h for side in allowed))
            records.append((split, n, h, w, k, port_count, allowed, strong, chain))
    return records


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return a deterministic puzzle with fixed, input-visible conventions."""
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    train_count, test_count = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, n, h, w, k, pc, sides, strong, chain in _schedule(rng, train_count, test_count):
        for attempt in range(384):
            scene = _sample_scene(rng, n, h, w, k, pc, sides, strong, chain)
            if scene is None or (strong and not _strong_witness(scene)):
                continue
            output = _render(scene)
            if chain and np.array_equal(output, _render(scene, "two_maps")):
                continue
            if chain and not _ports_witness(scene):
                continue
            grid = _pack_scene(rng, scene)
            parsed = _parse(grid)
            if strong and np.array_equal(output, _render(parsed, "position")):
                continue
            key = _array_key(grid)
            if key in seen:
                continue
            seen.add(key)
            puzzle[split].append({"input": grid, "output": output})
            break
        else:
            raise RuntimeError("bounded informative scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check rule invariants and actual training evidence from returned grids."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle needs exactly train/test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair needs exactly input/output")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            key = _array_key(grid)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            scene = _parse(grid)
            _require(np.array_equal(output, _coordinate_graph_output(scene)), "output violates the coordinate-graph rule")
            _require(np.array_equal(output, _render(scene)), "output violates input-only rendering")
            _require(not np.array_equal(grid, output), "identity pair is not informative")
            bg, n = scene["background"], scene["n"]
            h, w = scene["macro"].shape
            _require(output.shape == (h*n+2, w*n+2), "incorrect expansion dimensions")
            _require(set(map(int, np.unique(output))) <= {bg, scene["frame"], scene["port"], *scene["mapping"].values()},
                     "unexplained output colors")
            _require(np.any(output[1:-1, 1:-1] != bg), "ports must power some foreground")
            for mr in range(h):
                for mc in range(w):
                    tile = output[1+mr*n:1+(mr+1)*n, 1+mc*n:1+(mc+1)*n]
                    source = int(scene["macro"][mr, mc])
                    if source == bg:
                        _require(bool(np.all(tile == bg)), "empty macro cell must remain a background block")
                    else:
                        _require(bool(np.all(tile[~scene["masks"][source]] == bg)), "paint outside the stencil")
                        _require(bool(np.all(np.isin(tile, [bg, scene["mapping"][source]]))), "tile is not directly mapped")
            if split == "train":
                train.append((scene, output))
    for alternative in (*_COUNTERFACTUALS, "position", "two_maps"):
        _require(any(not np.array_equal(out, _render(scene, alternative)) for scene, out in train),
                 f"training fails to distinguish {alternative}")
    for t in range(1, 8):
        _require(any(not np.array_equal(out, _render(scene, stamp_transform=t)) for scene, out in train),
                 "training fails to establish stencil orientation")
    _require(any(_ports_witness(scene) for scene, _ in train),
             "training fails to establish union of multiple ports")


def measure_puzzle(puzzle):
    """Small descriptive records reconstructed from the visible inputs."""
    records = []
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            a = _parse(pair["input"])
            original = _expanded_sources(a)
            live = _flood(original, _expanded_border(a), a)
            n = a["n"]
            partial_tiles = sum(0 < int(live[r:r+n, c:c+n].sum()) <
                                int(np.sum(original[r:r+n, c:c+n] != a["background"]))
                                for r in range(0, original.shape[0], n)
                                for c in range(0, original.shape[1], n))
            records.append(dict(split=split, pair=index, n=a["n"], macro_shape=list(a["macro"].shape),
                                categories=len(a["sources"]), occupied=int(np.sum(a["macro"] != a["background"])),
                                ports=len(a["ports"]), sides=sorted({s for s, _ in a["ports"]}),
                                input_shape=list(np.asarray(pair["input"]).shape), output_shape=list(np.asarray(pair["output"]).shape),
                                roles=[a["background"], a["frame"], a["port"]],
                                powered=int(live.sum()), unpowered=int(((original != a["background"]) & ~live).sum()),
                                partial_tiles=partial_tiles,
                                disconnected=sum(len(_components(m)) > 1 for m in a["masks"].values()),
                                solid=sum(bool(m.all()) for m in a["masks"].values()),
                                shared_masks=len(a["sources"])-len({m.tobytes() for m in a["masks"].values()}),
                                overlapping_keys=sum(t in a["sources"] for t in a["mapping"].values()),
                                shared_targets=len(a["sources"])-len(set(a["mapping"].values())),
                                alternative_errors={mode: int(np.sum(np.asarray(pair["output"]) != _render(a, mode)))
                                                    for mode in (*_COUNTERFACTUALS, "position", "two_maps")}))
    return records


def self_test():
    """Bounded 42-seed structural, semantic, and mutation checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seed diversity failed")
    _require(_puzzle_key(generate_puzzle(0)) == _puzzle_key(puzzles[0]), "determinism failed")
    records, layouts, masks, normalized, shapes = [], set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        records.extend(measure_puzzle(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                scene = _parse(pair["input"])
                bg = scene["background"]
                macro = np.full(scene["macro"].shape, -1, dtype=np.int8)
                for i, s in enumerate(scene["sources"]):
                    macro[scene["macro"] == s] = i
                layouts.add(_array_key(macro))
                masks.update(_array_key(m) for m in scene["masks"].values())
                shapes.add((np.asarray(pair["input"]).shape, np.asarray(pair["output"]).shape))
                roles = {}
                a = np.array([[roles.setdefault(int(v), len(roles)) for v in row] for row in pair["input"]])
                normalized.add(_array_key(a))
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve disagreement")
    _require({r["n"] for r in records} == {2, 3, 4, 5}, "scale diversity failed")
    _require({r["categories"] for r in records} == {1, 2, 3, 4}, "identity count diversity failed")
    _require({r["ports"] for r in records} == {1, 2, 3, 4}, "port multiplicity diversity failed")
    _require(set().union(*(set(r["sides"]) for r in records)) == set(_SIDES), "port side diversity failed")
    _require(len(shapes) >= 40 and len(layouts) >= 100 and len(masks) >= 80 and len(normalized) >= 150,
             "structural variation is insufficient")
    _require(any(r["macro_shape"][0] != r["macro_shape"][1] for r in records), "no rectangular boards")
    _require(any(r["occupied"] < np.prod(r["macro_shape"]) for r in records), "no blank macro cells")
    _require(any(r["solid"] for r in records) and any(r["shared_masks"] for r in records), "no mask degeneracies")
    _require(sum(r["disconnected"] for r in records) >= 30, "missing disconnected stencils")
    _require(any(r["shared_targets"] for r in records), "missing many-to-one mappings")
    for role in range(3):
        _require({r["roles"][role] for r in records} == set(range(10)), "palette role diversity failed")
    for puzzle in puzzles:
        local = measure_puzzle(puzzle)
        tr, te = local[:4], local[4:]
        scale_shift = max(r["n"] for r in tr) < min(r["n"] for r in te)
        extent_shift = max(max(r["macro_shape"]) for r in tr) < min(max(r["macro_shape"]) for r in te)
        side_shift = bool(set().union(*(set(r["sides"]) for r in te)) - set().union(*(set(r["sides"]) for r in tr)))
        count_shift = max(r["ports"] for r in tr) < min(r["ports"] for r in te)
        _require(scale_shift or extent_shift or side_shift or count_shift, "missing visible train/test shift")
    # Exercise all legal explicit counts at one seed, including the minimum.
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(701, tr, te)
            _require(len(p["train"]) == tr and len(p["test"]) == te, "wrong explicit counts")
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(3, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Recolor every role independently of its meaning.
    rng = np.random.default_rng(8387)
    for puzzle in puzzles[:4]:
        permutation = rng.permutation(10)
        changed = {split: [{role: permutation[np.asarray(value)] for role, value in p.items()}
                           for p in puzzle[split]] for split in ("train", "test")}
        validate_generated_puzzle(changed)
    # Move entire disconnected stencil squares by swapping two legend slots.
    pair = puzzles[0]["train"][0]
    scene = _parse(pair["input"])
    a, b = scene["sources"][:2]
    ar, ae, ac, ad = scene["mask_boxes"][a]
    br, be, bc, bd = scene["mask_boxes"][b]
    moved = pair["input"].copy()
    first, second = moved[ar:ae, ac:ad].copy(), moved[br:be, bc:bd].copy()
    moved[ar:ae, ac:ad], moved[br:be, bc:bd] = second, first
    _require(np.array_equal(solve(moved), pair["output"]), "legend position affected the answer")
    # A deliberately disconnected stencil: diagonal contact must not conduct.
    raw = dict(background=0, frame=3, port=4, n=3, sources=[6], mapping={6:7},
               macro=np.array([[6, 0], [0, 0]], dtype=np.int8), ports=[("left", 0)],
               masks={6:np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=bool)})
    result = _render(raw)
    _require(int(np.sum(result == 7)) == 5 and result[1, 3] == 0, "partial-tile topology regression")
    _require(int(np.sum(_render(raw, "diagonal") == 7)) == 6, "diagonal topology regression")
    _require(np.array_equal(result, _coordinate_graph_output(raw)), "graph topology regression")
    packed = _pack_scene(rng, raw)
    _require(np.array_equal(solve(packed), result), "one-source input-only topology regression")
    # A bridge connects a remote tile through a different source identity.
    # The second copy of source 5 keeps its legend valid after bridge deletion.
    ring = np.ones((3, 3), dtype=bool)
    ring[1, 1] = False
    yy, xx = np.indices((3, 3))
    bridge = dict(background=0, frame=2, port=3, n=3, sources=[4, 5],
                  mapping={4:5, 5:6}, macro=np.array([[4, 5, 4], [5, 0, 0]], dtype=np.int8),
                  ports=[("left", 0)], masks={4:ring, 5:(yy == 1) | (xx == 1)})
    bridged = _render(bridge)
    _require(int(np.sum(bridged[1:-1, 1:-1] != 0)) == 26, "bridge should power all four tiles")
    _require(np.all(bridged[1:4, 7:10][ring] == 5), "remote tile must use its own direct key")
    _require(np.array_equal(solve(_pack_scene(rng, bridge)), bridged), "visible bridge regression")
    removed = dict(bridge, macro=bridge["macro"].copy())
    removed["macro"][0, 1] = 0
    disconnected = _render(removed)
    _require(int(np.sum(disconnected[1:-1, 1:-1] != 0)) == 13
             and bool(np.all(disconnected[1:4, 7:10] == 0)), "bridge deletion must erase the remote tile")
    _require(np.array_equal(solve(_pack_scene(rng, removed)), disconnected), "visible bridge deletion regression")
    reseeded = dict(removed, ports=[("left", 0), ("right", 0)])
    _require(int(np.sum(_render(reseeded)[1:-1, 1:-1] != 0)) == 21, "second port must power the remote island")
    _require(np.array_equal(solve(_pack_scene(rng, reseeded)), _render(reseeded)), "multiple-port union regression")
    for mapping in ({4:5, 5:4}, {4:6, 5:6}):
        changed = dict(bridge, mapping=mapping)
        _require(np.array_equal(solve(_pack_scene(rng, changed)), _coordinate_graph_output(changed)),
                 "cycle/shared-target direct substitution regression")
    # Corrupt an output and a visible key; each should be rejected.
    bad = {split: [{role: np.asarray(v).copy() for role, v in p.items()} for p in puzzles[0][split]]
           for split in ("train", "test")}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    broken = pair["input"].copy()
    r, e, c, d = scene["key_boxes"][a]
    broken[r:e, c:d] = scene["background"]
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("missing key accepted")
    return dict(seeds=42, pairs=len(records), input_output_shapes=len(shapes),
                macro_layouts=len(layouts), stencil_masks=len(masks), normalized_inputs=len(normalized),
                independent_solver=False, blind_model_prediction=False)


if __name__ == "__main__":
    print(self_test())
