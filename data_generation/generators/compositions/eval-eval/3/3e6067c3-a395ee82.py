"""Anchored, directional stencil printing along a uniquely keyed coarse route.

Only NumPy and the standard library are used. Semantic decoding reads pixels;
sampling parameters are never used by solve(). See task_spec.md for the rule.
"""
from collections import Counter
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((a >= 0) & (a <= 9)), "bad colors")
    return a.astype(np.int8, copy=False)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _between(p, a, b):
    return ((a[0] == b[0] == p[0] and min(a[1], b[1]) < p[1] < max(a[1], b[1]))
            or (a[1] == b[1] == p[1] and min(a[0], b[0]) < p[0] < max(a[0], b[0])))


def _graph(points):
    return [[j for j, b in enumerate(points) if j != i
             and (a[0] == b[0] or a[1] == b[1])
             and not any(_between(p, a, b) for p in points)]
            for i, a in enumerate(points)]


def _routes(points, labels, key, limit=2):
    graph = _graph(points)
    choices = [[i for i, label in enumerate(labels) if label == color] for color in key]
    answers = []
    def visit(path, used):
        if len(answers) >= limit:
            return
        if len(path) == len(key):
            answers.append(tuple(path))
            return
        for i in choices[len(path)]:
            if i not in used and (not path or i in graph[path[-1]]):
                visit(path + [i], used | {i})
    visit([], set())
    return answers


def _heading(a, b):
    if a[0] == b[0] and a[1] != b[1]:
        return "E" if b[1] > a[1] else "W"
    _require(a[1] == b[1] and a[0] != b[0], "nonorthogonal segment")
    return "S" if b[0] > a[0] else "N"


_TURN = {"E": 0, "N": 1, "W": 2, "S": 3}


def _stamps(path, colors):
    """Chronological (coarse coordinate, color, heading) instruction stream."""
    stamps = []
    for a, b, color in zip(path, path[1:], colors):
        direction = _heading(a, b)
        dr, dc = int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1]))
        for step in range(abs(b[0]-a[0]) + abs(b[1]-a[1])):
            stamps.append(((a[0]+step*dr, a[1]+step*dc), int(color), direction))
    stamps.append((path[-1], int(colors[-1]), _heading(path[-2], path[-1])))
    return stamps


def _decode(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])),
             "input requires a background frame")
    foreground = np.argwhere(a != bg)
    _require(len(foreground) > 0, "empty input")
    # The detached key is the lowest nonbackground row, below both other objects.
    key_r = int(foreground[:, 0].max())
    key_cs = np.flatnonzero(a[key_r] != bg).tolist()
    _require(3 <= len(key_cs) <= 10 and all(y-x == 2 for x, y in zip(key_cs, key_cs[1:])),
             "malformed left-to-right key")
    key = [int(a[key_r, c]) for c in key_cs]
    body = a != bg
    body[key_r] = False
    parts = _components(body)
    glyphs = [p for p in parts if len(p) > 1]
    _require(len(glyphs) == 1, "expected one connected stencil")
    glyph = glyphs[0]
    tr, tc = min(r for r, c in glyph), min(c for r, c in glyph)
    br, bc = max(r for r, c in glyph), max(c for r, c in glyph)
    n = br-tr+1
    _require(n >= 2 and bc-tc+1 == n, "stencil must have a tight square box")
    ink = int(a[glyph[0]])
    _require(all(int(a[p]) == ink for p in glyph), "stencil must be monochrome")
    mask = a[tr:br+1, tc:bc+1] == ink
    _require(np.all((a[tr:br+1, tc:bc+1] == bg) | mask), "foreign stencil cells")
    point_cells = sorted(p[0] for p in parts if len(p) == 1)
    _require(len(point_cells) >= len(key), "not enough map points")
    pr, pc = min(r for r, c in point_cells), min(c for r, c in point_cells)
    er, ec = max(r for r, c in point_cells), max(c for r, c in point_cells)
    _require(all((r-pr) % 2 == 0 and (c-pc) % 2 == 0 for r, c in point_cells),
             "map points must share a pitch-two lattice")
    _require(br+1 < pr or er+1 < tr or bc+1 < pc or ec+1 < tc,
             "stencil and map boxes require a blank separating row or column")
    _require(key_r >= max(br, er)+2, "key must be detached below both objects")
    points = [((r-pr)//2, (c-pc)//2) for r, c in point_cells]
    labels = [int(a[p]) for p in point_cells]
    anchors = [i for i, color in enumerate(labels) if color == ink]
    _require(len(anchors) == 1, "expected one stencil-colored map anchor")
    routes = _routes(points, labels, key)
    _require(len(routes) == 1, "key must identify a unique distinct-point route")
    route = routes[0]
    _require(anchors[0] in route, "route must include the anchor")
    path = [points[i] for i in route]
    anchor = points[anchors[0]]
    stamps = _stamps(path, key)
    for (r, c), color, direction in stamps:
        y, x = tr+(r-anchor[0])*n, tc+(c-anchor[1])*n
        _require(0 <= y and y+n <= a.shape[0] and 0 <= x and x+n <= a.shape[1],
                 "anchored print is out of bounds")
    return dict(grid=a, background=bg, ink=ink, mask=mask, size=n,
                template=(tr, tc), key=key, key_row=key_r, key_cols=key_cs,
                map_cells=point_cells, points=points, labels=labels, route=route,
                path=path, anchor=anchor, stamps=stamps)


def _render(shape, bg, mask, template, anchor, stamps):
    out = np.full(shape, bg, dtype=np.int8)
    n = mask.shape[0]
    rotations = {d: np.rot90(mask, k).copy() for d, k in _TURN.items()}
    for (r, c), color, direction in stamps:
        y, x = template[0]+(r-anchor[0])*n, template[1]+(c-anchor[1])*n
        tile = out[y:y+n, x:x+n]
        tile[rotations[direction]] = color  # blank stencil cells are transparent
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from the input; do not modify the input."""
    m = _decode(input_grid)
    return _render(m["grid"].shape, m["background"], m["mask"],
                   m["template"], m["anchor"], m["stamps"])


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _asymmetric(mask):
    variants = [np.rot90(mask, k) for k in range(1, 4)]
    variants += [np.rot90(np.fliplr(mask), k) for k in range(4)]
    return all(not np.array_equal(mask, other) for other in variants)


def _mask(rng, n, informative=False):
    """Connected masks with a tight box, varied density and orientation."""
    mask = np.zeros((n, n), dtype=bool)
    r, c = 0, 0
    mask[r, c] = True
    steps = [0]*(n-1) + [1]*(n-1)
    rng.shuffle(steps)
    for axis in steps:
        r += axis == 0
        c += axis == 1
        mask[r, c] = True
    goal = int(rng.integers(int(mask.sum()), n*n))  # at least one transparent cell
    while int(mask.sum()) < goal:
        frontier = sorted({(rr, cc) for r, c in np.argwhere(mask)
                           for rr, cc in ((int(r)-1, int(c)), (int(r)+1, int(c)),
                                          (int(r), int(c)-1), (int(r), int(c)+1))
                           if 0 <= rr < n and 0 <= cc < n and not mask[rr, cc]})
        mask[_pick(rng, frontier)] = True
    if np.array_equal(mask, np.rot90(mask)):
        mask[:] = True
        mask[0, 0] = False
    if informative and not _asymmetric(mask):
        # Connected, chiral fallback at n>=3; this distinguishes rotations from
        # reflections even in the minimum two-demonstration puzzle.
        mask[:] = True
        mask[0, 0] = mask[0, 1] = mask[-1, -1] = False
    return np.rot90(mask, int(rng.integers(4))).copy()


def _can_append(path, q):
    if q in path or any(_between(q, a, b) for a, b in zip(path, path[1:])):
        return False
    a = path[-1]
    return (a[0] == q[0] or a[1] == q[1]) and not any(_between(p, a, q) for p in path)


def _extend(rng, path, dims, target):
    """Bounded random DFS of orthogonal visibility routes; crossings are allowed."""
    budget = 1200
    universe = [(r, c) for r in range(dims[0]) for c in range(dims[1])]
    def visit(p):
        nonlocal budget
        budget -= 1
        if len(p) == target:
            return p
        if budget <= 0:
            return None
        choices = [q for q in universe if _can_append(p, q)]
        rng.shuffle(choices)
        for q in choices:
            found = visit(p+[q])
            if found is not None:
                return found
            if budget <= 0:
                break
        return None
    return visit(list(path))


def _cross_route(rng, dims, target):
    # A variable-spaced four-heading crossing scaffold. Extensions are sampled
    # from the complete visibility geometry, not from a stored route bank.
    h, w = dims
    midr, midc = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
    path = [(midr, 0), (midr, w-1), (0, w-1), (0, midc), (h-1, midc)]
    path = _extend(rng, path, dims, target) or path
    turns = int(rng.integers(4))
    for _ in range(turns):
        path = [(c, h-1-r) for r, c in path]
        h, w = w, h
    if rng.random() < .5:
        path = [(r, w-1-c) for r, c in path]
    if rng.random() < .5:
        path.reverse()
    return path, (h, w)


def _plain_route(rng):
    for _ in range(30):
        dims = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        start = int(rng.integers(dims[0])), int(rng.integers(dims[1]))
        path = _extend(rng, [start], dims, int(rng.integers(3, 7)))
        if path and len({_heading(a, b) in "EW" for a, b in zip(path, path[1:])}) == 2:
            return path, dims
    return [(0, 0), (0, 2), (2, 2)], (3, 3)


def _crossing_mixed(mask, stamps, bg):
    tiles = {}
    for p, color, d in stamps:
        tile = tiles.setdefault(p, np.full(mask.shape, bg, dtype=np.int8))
        tile[np.rot90(mask, _TURN[d])] = color
    return any(len(set(tile[tile != bg].tolist())) >= 2 for tile in tiles.values())


def _labels(rng, path, points, anchor_index, bg, ink, repeated, mask):
    available = [c for c in range(10) if c not in (bg, ink)]
    graph_order = {p: i for i, p in enumerate(points)}
    intended = tuple(graph_order[p] for p in path)
    for _ in range(160):
        if repeated:
            pool = list(map(int, rng.choice(available, int(rng.integers(2, min(5, len(path)-1)+1)), replace=False)))
            key = [int(rng.choice(pool)) for p in path]
        else:
            pool = list(map(int, rng.permutation(available)))
            key = pool[:len(path)-1]
            key.insert(anchor_index, ink)
        key[anchor_index] = ink
        if repeated and len(set(key)) == len(key):
            continue
        if not any(a != b for a, b in zip(key, key[1:])):
            continue
        if repeated and not _crossing_mixed(mask, _stamps(path, key), bg):
            continue
        labels = [int(rng.choice(available)) for p in points]
        for p, color in zip(path, key):
            labels[graph_order[p]] = color
        # One unused duplicate is always present, making route selection matter.
        unused = [i for i, p in enumerate(points) if p not in path]
        if unused:
            labels[unused[0]] = int(rng.choice([c for c in key if c != ink]))
        answers = _routes(points, labels, key)
        if answers == [intended]:
            return labels, key
    return None


def _place(rng, path, points, labels, key, mask, anchor_index, bg, ink):
    n = len(mask)
    lo = (min(r for r, c in path), min(c for r, c in path))
    hi = (max(r for r, c in path), max(c for r, c in path))
    ah, aw = (hi[0]-lo[0]+1)*n, (hi[1]-lo[1]+1)*n
    anchor = path[anchor_index]
    # The print lies on a plane with origin (0,0); register the glyph inside it.
    tr, tc = (anchor[0]-lo[0])*n, (anchor[1]-lo[1])*n
    pr, pc = min(r for r, c in points), min(c for r, c in points)
    ph = 2*(max(r for r, c in points)-pr)+1
    pw = 2*(max(c for r, c in points)-pc)+1
    placements = []
    for mr in range(-ph-2, ah+3):
        for mc in range(-pw-2, aw+3):
            if not (mr+ph < tr or tr+n < mr or mc+pw < tc or tc+n < mc):
                continue
            minr, minc = min(0, mr), min(0, mc)
            maxr, maxc = max(ah, mr+ph), max(aw, mc+pw)
            # A frame and a detached bottom key fit in the union's spare space.
            body_bottom = max(tr+n, mr+ph)
            need_h = max(maxr-minr+2, body_bottom-minr+4)
            need_w = max(maxc-minc+2, 2*len(key)+1)
            if need_h <= 30 and need_w <= 30:
                placements.append((mr, mc, minr, minc, need_h, need_w))
    if not placements:
        return None
    if rng.random() < .55:
        smallest = min(p[4]*p[5] for p in placements)
        placements = [p for p in placements if p[4]*p[5] <= smallest+80]
    mr, mc, minr, minc, base_h, base_w = _pick(rng, placements)
    h = int(rng.integers(base_h, min(30, base_h+4)+1))
    w = int(rng.integers(base_w, min(30, base_w+4)+1))
    # Move union without moving its registration relations.
    top = 1-minr+int(rng.integers(h-base_h+1))
    left = 1-minc+int(rng.integers(w-base_w+1))
    template = tr+top, tc+left
    map_top, map_left = mr+top, mc+left
    body_bottom = max(template[0]+n, map_top+ph)
    key_r = int(rng.integers(body_bottom+1, h-1))
    key_c = int(rng.integers(1, w-(2*len(key)-1)))
    inp = np.full((h, w), bg, dtype=np.int8)
    inp[template[0]:template[0]+n, template[1]:template[1]+n][mask] = ink
    for (r, c), color in zip(points, labels):
        inp[map_top+2*(r-pr), map_left+2*(c-pc)] = color
    inp[key_r, key_c:key_c+2*len(key):2] = key
    out = _render(inp.shape, bg, mask, template, anchor, _stamps(path, key))
    return {"input": inp, "output": out}


def _sample_pair(rng, crossing, test, n, previous_anchor=None, roles=None, teaching=False):
    for _ in range(100):
        if crossing:
            dims = _pick(rng, [(6, k) for k in range(4, 7)]) if test else (
                int(rng.integers(4, 6)), int(rng.integers(4, 6)))
            target = int(rng.integers(7, 10)) if test else int(rng.integers(5, 8))
            path, dims = _cross_route(rng, dims, target)
        else:
            path, dims = _plain_route(rng)
        anchor_indices = [i for i, p in enumerate(path)
                          if (previous_anchor is None or p != previous_anchor)
                          and (not teaching or (0 < i < len(path)-1
                               and p != ((dims[0]-1)//2, (dims[1]-1)//2)))]
        anchor_index = int(_pick(rng, anchor_indices or list(range(len(path)))))
        mask = _mask(rng, n, informative=teaching)
        bg, ink = roles if roles is not None else tuple(map(int, rng.choice(10, 2, replace=False)))
        # Distractors cannot block any selected edge; they can share route labels.
        low_r, high_r = min(r for r, c in path), max(r for r, c in path)
        low_c, high_c = min(c for r, c in path), max(c for r, c in path)
        possible = [(r, c) for r in range(dims[0]) for c in range(dims[1])
                    if (r, c) not in path
                    and low_r <= r <= high_r and low_c <= c <= high_c
                    and not any(_between((r, c), a, b) for a, b in zip(path, path[1:]))]
        if not possible:
            continue
        count = min(len(possible), int(rng.integers(1, 4 if test else 3)))
        indices = rng.choice(len(possible), count, replace=False)
        points = sorted(path+[possible[int(i)] for i in indices])
        assignment = _labels(rng, path, points, anchor_index, bg, ink, crossing, mask)
        if assignment is None:
            continue
        labels, key = assignment
        pair = _place(rng, path, points, labels, key, mask, anchor_index, bg, ink)
        if pair is not None and (previous_anchor is None
                                 or _decode(pair["input"])["anchor"] != previous_anchor):
            return pair
    # Same semantics, deterministic geometry, random mask and palette: bounded
    # retries do not silently weaken the teaching requirements or test extent.
    extent = 6 if test else 4
    path = [(1, 0), (1, extent-1), (0, extent-1), (0, 2), (extent-1, 2)]
    points = sorted(path+[(extent-1, 0)])
    mask = np.ones((n, n), dtype=bool)
    mask[0, 0] = False
    if teaching:
        mask[0, 1] = mask[-1, -1] = False
    bg, ink = roles if roles is not None else tuple(map(int, rng.choice(10, 2, replace=False)))
    colors = [c for c in range(10) if c not in (bg, ink)]
    anchor_index = 1 if previous_anchor == path[2] else 2
    key = [colors[0], colors[1], colors[2], colors[1], colors[3]]
    key[anchor_index] = ink
    # If the second occurrence was replaced, repeat a different pair of labels.
    if len(set(key)) == len(key):
        key[4] = key[2]
    labels = [key[path.index(p)] if p in path else colors[0] for p in points]
    pair = _place(rng, path, points, labels, key, mask, anchor_index, bg, ink)
    assert pair is not None
    return pair


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    # Only distribution choices are puzzle-level. All solving conventions are fixed.
    stable = rng.random() < .25
    roles = tuple(map(int, rng.choice(10, 2, replace=False))) if stable else None
    sizes = list(map(int, rng.choice([2, 3, 4], 2, replace=False)))
    if sizes[1] == 2:
        sizes.reverse()  # the crossing lesson needs a chiral n>=3 stencil
    puzzle, seen = {"train": [], "test": []}, set()
    previous_anchor = None
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            test = split == "test"
            crossing = test or i == 1 or (i > 1 and rng.random() < .65)
            n = sizes[i] if not test and i < 2 else int(rng.integers(2, 5))
            pair = _sample_pair(rng, crossing, test, n,
                                previous_anchor if i == 1 and not test else None,
                                roles, teaching=not test and i == 1)
            if _signature(pair["input"]) in seen:
                # Forty distinct affine palettes outnumber at most thirteen
                # previous inputs. Each scene has at least four colors, so
                # these permutations have distinct effects on its pixels.
                fresh = None
                for multiplier in (1, 3, 7, 9):
                    for delta in range(10):
                        variant = {k: ((multiplier*v.astype(int)+delta) % 10).astype(np.int8) for k, v in pair.items()}
                        if _signature(variant["input"]) not in seen:
                            fresh = variant
                            break
                    if fresh is not None:
                        break
                if fresh is None:
                    raise RuntimeError("distinct-input fallback exhausted")
                pair = fresh
            seen.add(_signature(pair["input"]))
            puzzle[split].append(pair)
            if not test and i == 0:
                previous_anchor = _decode(pair["input"])["anchor"]
    validate_generated_puzzle(puzzle)
    return puzzle


def _inverse_render(m):
    """Pixel-to-tile invariant check, separate from the forward stamp renderer."""
    bg, n = m["background"], m["size"]
    out = np.full(m["grid"].shape, bg, dtype=np.int8)
    layers = {}
    for p, color, direction in m["stamps"]:
        layers.setdefault(p, []).append((color, direction))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            dy, dx = y-m["template"][0], x-m["template"][1]
            p = dy//n+m["anchor"][0], dx//n+m["anchor"][1]
            r, c = dy % n, dx % n
            for color, d in reversed(layers.get(p, [])):
                # Invert the heading's rotation to a prototype coordinate.
                rr, cc = r, c
                for _ in range(_TURN[d]):
                    rr, cc = cc, n-1-rr
                if m["mask"][rr, cc]:
                    out[y, x] = color
                    break
    return out


def _shortcut(m, mode):
    """Finite plausible alternatives for demonstration checks, not another solver."""
    mask, bg = m["mask"], m["background"]
    stamps = m["stamps"]
    if mode == "all_points":
        stamps = [(p, color, "E") for p, color in zip(m["points"], m["labels"])]
    elif mode == "no_rotation":
        stamps = [(p, color, "E") for p, color, d in stamps]
    elif mode == "reverse_layers":
        stamps = list(reversed(stamps))
    elif mode == "no_interpolation":
        stamps = [(p, color, _heading(p, q)) for p, q, color in zip(m["path"], m["path"][1:], m["key"])]
        stamps.append((m["path"][-1], m["key"][-1], _heading(m["path"][-2], m["path"][-1])))
    elif mode == "target_color":
        stamps = _stamps(m["path"], m["key"][1:]+m["key"][-1:])
    elif mode == "solid_tiles":
        mask = np.ones_like(mask)
    elif mode == "mirror_turns":
        # Treat south/north as transposes and west as a horizontal reflection.
        out = np.full(m["grid"].shape, bg, dtype=np.int8)
        n = m["size"]
        shapes = {"E": mask, "S": mask.T, "W": np.fliplr(mask), "N": np.flipud(mask.T)}
        for (r, c), color, d in stamps:
            y, x = m["template"][0]+(r-m["anchor"][0])*n, m["template"][1]+(c-m["anchor"][1])*n
            out[y:y+n, x:x+n][shapes[d]] = color
        return out
    elif mode == "opaque":
        out = np.full(m["grid"].shape, bg, dtype=np.int8)
        n = m["size"]
        for (r, c), color, d in stamps:
            y, x = m["template"][0]+(r-m["anchor"][0])*n, m["template"][1]+(c-m["anchor"][1])*n
            out[y:y+n, x:x+n] = np.where(np.rot90(mask, _TURN[d]), color, bg)
        return out
    else:
        raise ValueError("unknown shortcut")
    return _render(m["grid"].shape, bg, mask, m["template"], m["anchor"], stamps)


_SHORTCUTS = ("all_points", "no_rotation", "reverse_layers", "no_interpolation",
              "target_color", "solid_tiles", "mirror_turns", "opaque")


def validate_generated_puzzle(puzzle):
    """Check schema, exact printing, routing, transparency and teaching witnesses."""
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert all(isinstance(puzzle[s], list) for s in puzzle)
        _counts(len(puzzle["train"]), len(puzzle["test"]))
        seen, infos = set(), {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                inp, out = _grid(pair["input"]), _grid(pair["output"])
                assert inp.shape == out.shape
                signature = _signature(inp)
                assert signature not in seen, "duplicate input"
                seen.add(signature)
                m = _decode(inp)
                assert np.array_equal(out, _inverse_render(m)), "wrong anchored transparent printing"
                assert np.array_equal(out, solve(inp)), "forward solver disagrees"
                assert not np.array_equal(inp, out)
                assert m["size"] in (2, 3, 4)
                assert not np.array_equal(m["mask"], np.rot90(m["mask"])), "uninformative stencil"
                # Every unused station must remain an unprinted macro-cell.
                printed = {p for p, color, direction in m["stamps"]}
                assert all(p not in printed for i, p in enumerate(m["points"]) if i not in m["route"])
                assert len(m["route"]) < len(m["points"]), "missing route-selection witness"
                assert any(color in m["key"] for i, color in enumerate(m["labels"]) if i not in m["route"])
                infos[split].append(m)
        train = infos["train"]
        assert len({m["size"] for m in train}) >= 2, "scale contrast not taught"
        assert len({m["anchor"] for m in train}) >= 2, "anchor contrast not taught"
        assert {d for m in train for p, color, d in m["stamps"]} == set("ENSW")
        assert any(len(set(m["key"])) < len(m["key"]) for m in train), "repeated keys not taught"
        assert any(_crossing_mixed(m["mask"], m["stamps"], m["background"]) for m in train), "transparent crossings not taught"
        assert any(abs(a[0]-b[0])+abs(a[1]-b[1]) >= 2 for m in train for a, b in zip(m["path"], m["path"][1:])), "interpolation not taught"
        assert all(max(r for r, c in m["path"])-min(r for r, c in m["path"])+1 <= 5
                   and max(c for r, c in m["path"])-min(c for r, c in m["path"])+1 <= 5 for m in train)
        assert any(_asymmetric(m["mask"]) for m in train), "rotation versus reflection not taught"
        assert any(m["anchor"] != m["path"][0] for m in train), "first-point anchor shortcut"
        for mode in _SHORTCUTS:
            assert any(not np.array_equal(_shortcut(m, mode), _inverse_render(m)) for m in train), f"undistinguished shortcut: {mode}"
        assert all(max(max(r for r, c in m["path"])-min(r for r, c in m["path"])+1,
                       max(c for r, c in m["path"])-min(c for r, c in m["path"])+1) == 6
                   for m in infos["test"])
    except (ValueError, TypeError, IndexError, KeyError) as exc:
        raise AssertionError(f"malformed generated puzzle: {exc}") from exc


def self_test():
    """Bounded 42-seed checks of semantics and actual structural diversity."""
    signatures, shapes, masks, layouts, anchors, headings, colors = set(), set(), set(), set(), set(), set(), set()
    repeats, mixed, double_crossings, repainted = 0, 0, 0, 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        sig = tuple((_signature(p["input"]), _signature(p["output"])) for s in ("train", "test") for p in puzzle[s])
        signatures.add(sig)
        if seed == 0:
            first = puzzle
        for split in ("train", "test"):
            for pair in puzzle[split]:
                m = _decode(pair["input"])
                shapes.add(pair["input"].shape)
                masks.add((m["size"], m["mask"].tobytes()))
                layouts.add(tuple(m["path"]))
                anchors.add(m["anchor"])
                headings.update(d for p, color, d in m["stamps"])
                colors.update(map(int, np.unique(pair["input"])))
                repeats += len(set(m["key"])) < len(m["key"])
                mixed += _crossing_mixed(m["mask"], m["stamps"], m["background"])
                double_crossings += sum(v > 1 for v in Counter(p for p, color, d in m["stamps"]).values()) >= 2
                repainted += any(pair["output"][p] != m["background"] for p in m["map_cells"])
                # Omitting B: replace the mask by a solid tile. Must change a pixel.
                solid = _render(pair["input"].shape, m["background"], np.ones_like(m["mask"]),
                                m["template"], m["anchor"], m["stamps"])
                assert not np.array_equal(solid, pair["output"])
                # Omitting A: B-like substitution of every point, ignoring the key.
                assert not np.array_equal(_shortcut(m, "all_points"), pair["output"])
    assert len(signatures) == 42
    assert len(shapes) >= 30 and len(masks) >= 30 and len(layouts) >= 100
    assert len(anchors) >= 15 and headings == set("ENSW") and colors == set(range(10))
    assert repeats >= 100 and mixed >= 100 and double_crossings >= 10 and repainted >= 10
    again = generate_puzzle(0)
    assert all(np.array_equal(p[k], q[k]) for s in ("train", "test")
               for p, q in zip(first[s], again[s]) for k in ("input", "output"))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(42000+10*nt+nq, nt, nq)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]] for s in first}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, count coverage, printing invariants and diversity")
