"""Self-mapping patchwork: two microscopic maps activate complementary tile pieces.

Only NumPy and the standard library are used. All solving conventions are visible
in the header or fixed by the family. Generation caches contain geometry, never
solutions or examples, and are not consulted by solve.
"""
from functools import lru_cache
from itertools import combinations

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "bad ARC color")
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    directions = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                  if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, component = [start], []
        while todo:
            r, c = todo.pop()
            component.append((int(r), int(c)))
            for dr, dc in directions:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        a = np.zeros(mask.shape, dtype=bool)
        for p in component:
            a[p] = True
        result.append(a)
    return result


def _partition(cut):
    """Validate a digital path and recover its two four-connected sides."""
    cells = set(map(tuple, np.argwhere(cut)))
    _require(len(cells) >= 2, "cut is too short")
    graph = {}
    for r, c in cells:
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = (r + dr, c + dc)
                if (dr or dc) and q in cells:
                    # Do not introduce a diagonal shortcut across a right angle.
                    if dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells):
                        continue
                    neighbors.append(q)
        graph[r, c] = neighbors
    ends = [p for p in cells if len(graph[p]) == 1]
    _require(len(ends) == 2 and all(1 <= len(v) <= 2 for v in graph.values()),
             "cut branches or loops")
    _require(len(_components(cut, True)) == 1, "disconnected cut")
    h, w = cut.shape
    border = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _require(all(border(p) for p in ends), "cut must join boundaries")
    _require(all(not border(p) for p in cells - set(ends)), "extra boundary contact")
    regions = _components(~cut)
    _require(len(regions) == 2, "cut must separate exactly two nonempty sides")
    return regions


def _line(p, q):
    r, c = p
    r1, c1 = q
    dr, dc = abs(r1 - r), abs(c1 - c)
    sr, sc = (1 if r < r1 else -1), (1 if c < c1 else -1)
    error, cells = dc - dr, []
    while True:
        cells.append((r, c))
        if (r, c) == (r1, c1):
            return cells
        twice = 2 * error
        if twice > -dr:
            error -= dr
            c += sc
        if twice < dc:
            error += dc
            r += sr


@lru_cache(maxsize=9)
def _cuts(h, w):
    """Finite procedural geometric candidates; no fixed stencil vocabulary."""
    candidates = []
    for r in range(1, h - 1):
        a = np.zeros((h, w), bool)
        a[r] = True
        candidates.append(("horizontal", a))
    for c in range(1, w - 1):
        a = np.zeros((h, w), bool)
        a[:, c] = True
        candidates.append(("vertical", a))
    ports = [(r, c) for r in range(h) for c in range(w)
             if r in (0, h - 1) or c in (0, w - 1)]
    for i, p in enumerate(ports):
        for q in ports[i + 1:]:
            if p[0] != q[0] and p[1] != q[1]:
                a = np.zeros((h, w), bool)
                for cell in _line(p, q):
                    a[cell] = True
                candidates.append(("diagonal", a))
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            a = np.zeros((h, w), bool)
            a[:r + 1, c] = True
            a[r, c:] = True
            for fy in (False, True):
                for fx in (False, True):
                    b = a[::-1] if fy else a
                    b = b[:, ::-1] if fx else b
                    candidates.append(("elbow", b.copy()))
    if h >= 4 and w >= 4:
        a = np.zeros((h, w), bool)
        a[:2, 1] = a[1, 1:3] = a[1:, 2] = True
        for b in (a, a.T, a[::-1], a[:, ::-1]):
            candidates.append(("stepped", b.copy()))
    result, seen = [], set()
    for kind, a in candidates:
        if _key(a) in seen:
            continue
        try:
            _partition(a)
        except AssertionError:
            continue
        a.flags.writeable = False
        result.append((kind, a))
        seen.add(_key(a))
    _require(result, "no usable cuts")
    return tuple(result)


def _templates(ordinary, a, b, bg, cut, marked):
    """Indices 0/1/2/3 mean neither/A/B/both microscopic activations."""
    other = ~cut & ~marked
    ta, tb, both = ordinary.copy(), ordinary.copy(), ordinary.copy()
    ta[marked | cut] = a[marked | cut]
    tb[other | cut] = b[other | cut]
    both[marked] = a[marked]
    both[other] = b[other]
    both[cut] = np.where(a[cut] != bg, a[cut], b[cut])
    return (ordinary.copy(), ta, tb, both)


def _origins(labels, rows, cols, observations, untyped=False):
    h, w = labels.shape
    result = []
    for r in range(rows - h + 1):
        for c in range(cols - w + 1):
            if all(0 <= y - r < h and 0 <= x - c < w
                   and (labels[y - r, x - c] != 0 if untyped
                        else labels[y - r, x - c] == code)
                   for (y, x), code in observations.items()):
                result.append((r, c))
    return result


def _runs(indices):
    result = []
    for i in map(int, indices):
        if result and result[-1][1] + 1 == i:
            result[-1] = (result[-1][0], i)
        else:
            result.append((i, i))
    return result


def _decode(input_grid):
    grid = _grid(input_grid)
    divider = int(grid[0, 0])
    ry = _runs(np.flatnonzero(np.all(grid == divider, axis=1)))
    cx = _runs(np.flatnonzero(np.all(grid == divider, axis=0)))
    _require(len(ry) >= 4 and len(cx) >= 4, "header and field lattice missing")
    _require(ry[0][0] == cx[0][0] == 0 and ry[-1][1] == grid.shape[0] - 1
             and cx[-1][1] == grid.shape[1] - 1, "missing outer frame")
    ts = {b - a + 1 for a, b in ry + cx}
    hs = {ry[i + 1][0] - ry[i][1] - 1 for i in range(len(ry) - 1)}
    ws = {cx[i + 1][0] - cx[i][1] - 1 for i in range(len(cx) - 1)}
    _require(len(ts) == len(hs) == len(ws) == 1, "irregular lattice")
    t, h, w = ts.pop(), hs.pop(), ws.pop()
    rows, cols = len(ry) - 2, len(cx) - 1
    _require(t in (1, 2) and 2 <= h <= 4 and 2 <= w <= 4,
             "unreadable tile geometry")
    _require(rows >= h and cols >= max(3, w), "blueprint window does not fit")
    tile = lambda r, c: grid[t + r * (h + t):t + r * (h + t) + h,
                             t + c * (w + t):t + c * (w + t) + w]
    for r in range(rows + 1):
        for c in range(cols):
            _require(not np.any(tile(r, c) == divider), "divider inside a tile")
    control, a, b = [tile(0, c) for c in range(3)]
    ac, bc = set(map(int, np.unique(a))), set(map(int, np.unique(b)))
    _require(len(ac) == len(bc) == 2 and len(ac & bc) == 1, "source roles ambiguous")
    bg = next(iter(ac & bc))
    ia, ib = next(iter(ac - {bg})), next(iter(bc - {bg}))
    values, counts = np.unique(control, return_counts=True)
    roles = [(int(v), int(n)) for v, n in zip(values, counts) if v != bg]
    _require(len(roles) == 2, "control needs cut and dot colors")
    dot = [v for v, n in roles if n == 1]
    path = [v for v, n in roles if n > 1]
    _require(len(dot) == len(path) == 1, "dot and path not distinguishable")
    cut = control == path[0]
    marker = tuple(map(int, np.argwhere(control == dot[0])[0]))
    regions = _partition(cut)
    side = 0 if regions[0][marker] else 1
    _require(regions[side][marker], "marker must lie strictly off the cut")
    _require(len({divider, bg, ia, ib, dot[0], path[0]}) == 6, "overlapping color roles")
    records = {}
    for r in range(rows):
        for c in range(cols):
            p = tile(r + 1, c)
            records.setdefault(_key(p), [p.copy(), []])[1].append((r, c))
    ordered = sorted(records.values(), key=lambda q: len(q[1]), reverse=True)
    ordinary = ordered[0][0]
    _require(len(ordered[0][1]) > rows * cols / 2, "neutral tiles need strict majority")
    oc = set(map(int, np.unique(ordinary)))
    _require(bg in oc and len(oc) == 2 and not (oc - {bg}) &
             {divider, ia, ib, dot[0], path[0]}, "neutral texture role ambiguous")
    _require(all(np.array_equal(tile(0, c), ordinary) for c in range(3, cols)),
             "unused header tiles must be neutral")
    marked = regions[side]
    _require(np.any(marked & (a == ia)) and np.any((~cut & ~marked) & (b == ib)),
             "each source must have visible ink in its own piece")
    patterns = _templates(ordinary, a, b, bg, cut, marked)
    _require(len({_key(p) for p in patterns}) == 4, "patch types collide")
    codes = {_key(p): i for i, p in enumerate(patterns)}
    observations = {}
    for key, (_, positions) in records.items():
        _require(key in codes, "unrecognized torn tile")
        if codes[key]:
            observations.update({p: codes[key] for p in positions})
    labels = (a == ia).astype(np.uint8) + 2 * (b == ib).astype(np.uint8)
    _require(set(map(int, labels.flat)) == {0, 1, 2, 3}, "blueprints need all four activations")
    _require(len(observations) >= 2, "too few supplied patches")
    origins = _origins(labels, rows, cols, observations)
    _require(len(origins) == 1, "registration must be unique; ties are invalid")
    origin = origins[0]
    complete = {(origin[0] + int(r), origin[1] + int(c)): int(labels[r, c])
                for r, c in np.argwhere(labels != 0)}
    missing = {p: code for p, code in complete.items() if p not in observations}
    _require(missing, "at least one patch must need restoration")
    return dict(grid=grid, geometry=(h, w, t, rows, cols), divider=divider,
                control=control, a=a, b=b, bg=bg, ia=ia, ib=ib,
                path_color=path[0], dot_color=dot[0], marker=marker, cut=cut,
                regions=regions, side=side, marked=marked, ordinary=ordinary,
                patterns=patterns, labels=labels, observations=observations,
                origin=origin, complete=complete, missing=missing)


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _decode(input_grid)
    out = d["grid"].copy()
    h, w, t, _, _ = d["geometry"]
    for (r, c), code in d["missing"].items():
        y, x = t + (r + 1) * (h + t), t + c * (w + t)
        out[y:y + h, x:x + w] = d["patterns"][code]
    return out


def _fragment_options(labels, rows, cols, origin, evidence=False):
    """Enumerate strict fragments; colored clues register the two maps jointly."""
    active = np.argwhere(labels != 0)
    n = len(active)
    codes = np.arange(1, (1 << n) - 1, dtype=np.uint32)
    counts = np.zeros(len(codes), dtype=np.uint8)
    for i in range(n):
        counts += ((codes >> i) & 1).astype(np.uint8)
    codes = codes[(counts >= 2) & (counts <= (rows * cols - 1) // 2)]
    if evidence:
        for kind in (1, 2, 3):
            bits = sum(1 << i for i, p in enumerate(active) if labels[tuple(p)] == kind)
            codes = codes[(codes & bits) != bits]  # leave one missing of each kind
    untyped_ambiguous = np.zeros(len(codes), dtype=bool)
    h, w = labels.shape
    for r in range(rows - h + 1):
        for c in range(cols - w + 1):
            if (r, c) == origin:
                continue
            cover, union_cover = 0, 0
            for i, (y, x) in enumerate(active):
                dy, dx = int(y) + origin[0] - r, int(x) + origin[1] - c
                if 0 <= dy < h and 0 <= dx < w:
                    if labels[dy, dx] != 0:
                        union_cover |= 1 << i
                    if labels[dy, dx] == labels[y, x]:
                        cover |= 1 << i
            keep = (codes & cover) != codes
            untyped_ambiguous |= (codes & union_cover) == codes
            codes, untyped_ambiguous = codes[keep], untyped_ambiguous[keep]
    if evidence:
        codes = codes[untyped_ambiguous]
    return active, codes


def _pick_fragment(rng, labels, geometry, origin, evidence=False, excluded=()):
    _, _, _, rows, cols = geometry
    active, codes = _fragment_options(labels, rows, cols, origin, evidence)
    if excluded:
        codes = np.array([int(code) for code in codes if tuple(
            tuple(map(int, active[i])) for i in range(len(active)) if int(code) & (1 << i))
            not in excluded], dtype=np.uint32)
    if not len(codes):
        return None
    sizes = np.array([int(code).bit_count() for code in codes])
    size = int(rng.choice(np.unique(sizes)))
    code = int(rng.choice(codes[sizes == size]))
    return [tuple(map(int, active[i])) for i in range(len(active)) if code & (1 << i)]


def _render(model, fragment, origin):
    h, w, t, rows, cols = model["geometry"]
    cut, marked = model["cut"], model["regions"][model["side"]]
    patterns = _templates(model["ordinary"], model["a"], model["b"], model["bg"], cut, marked)
    control = np.full((h, w), model["bg"], np.int8)
    control[cut], control[model["marker"]] = model["path_color"], model["dot_color"]
    grid = np.full(((rows + 1) * (h + t) + t, cols * (w + t) + t),
                   model["divider"], np.int8)
    for r in range(rows + 1):
        for c in range(cols):
            p = (control, model["a"], model["b"])[c] if r == 0 and c < 3 else patterns[0]
            y, x = t + r * (h + t), t + c * (w + t)
            grid[y:y + h, x:x + w] = p
    supplied = {(origin[0] + r, origin[1] + c) for r, c in fragment}
    out = grid.copy()
    for r, c in np.argwhere(model["labels"] != 0):
        ar, ac = origin[0] + int(r), origin[1] + int(c)
        y, x = t + (ar + 1) * (h + t), t + ac * (w + t)
        patch = patterns[int(model["labels"][r, c])]
        out[y:y + h, x:x + w] = patch
        if (ar, ac) in supplied:
            grid[y:y + h, x:x + w] = patch
    return {"input": grid, "output": out}


def _geometry(rng, evidence=False):
    shapes = ((3, 4), (4, 3), (4, 4)) if evidence else tuple(
        (h, w) for h in range(2, 5) for w in range(2, 5))
    h, w = shapes[int(rng.integers(len(shapes)))]
    thicknesses = [t for t in (1, 2) if (h + 1) * (h + t) + t <= 30
                   and max(3, w) * (w + t) + t <= 30]
    t = 1 if evidence else int(rng.choice(thicknesses))
    max_rows, max_cols = (30 - t) // (h + t) - 1, (30 - t) // (w + t)
    rows = int(rng.integers(h, max_rows + 1))
    cols = int(rng.integers(max(3, w + int(evidence)), max_cols + 1))
    return h, w, t, rows, cols


def _new_model(rng, palette, evidence=False, bent=False):
    geometry = _geometry(rng, evidence)
    h, w, _, rows, cols = geometry
    choices = list(_cuts(h, w))
    if evidence:
        choices = [(kind, cut) for kind, cut in choices
                   if kind in ("horizontal", "vertical") and cut.sum() == 4]
    elif bent:
        choices = [(kind, cut) for kind, cut in choices if kind not in ("horizontal", "vertical")]
    kind, cut = choices[int(rng.integers(len(choices)))]
    regions = _partition(cut)
    side = int(rng.integers(2))
    sites = np.argwhere(regions[side])
    marker = tuple(map(int, sites[int(rng.integers(len(sites)))]))
    divider, bg, path_color, dot_color, ia, ib, texture = palette
    for _ in range(128):
        # Sparse, dense, periodic and free source maps share the same vocabulary.
        mode = int(rng.integers(5))
        rr, cc = np.indices((h, w))
        if mode == 0:
            am = (rr + cc + int(rng.integers(2))) % 2 == 0
            bm = (rr + 2 * cc + int(rng.integers(3))) % 3 != 0
            labels = am.astype(np.uint8) + 2 * bm.astype(np.uint8)
        else:
            p = (.3, .5, .7, .5)[mode - 1]
            labels = (rng.random((h, w)) < p).astype(np.uint8)
            labels += 2 * (rng.random((h, w)) < (1 - p if mode == 3 else p)).astype(np.uint8)
        if evidence:
            labels[cut] = rng.permutation(4)
        if set(map(int, labels.flat)) != {0, 1, 2, 3}:
            continue
        if evidence and not all(np.any(edge != 0) for edge in
                                (labels[0], labels[-1], labels[:, 0], labels[:, -1])):
            continue
        am, bm = labels % 2 == 1, labels >= 2
        needed_regions = regions if evidence else [regions[side]]
        if evidence and not all(np.any(reg & am) and np.any(reg & ~am)
                                and np.any(reg & bm) and np.any(reg & ~bm)
                                for reg in needed_regions):
            continue
        if not np.any(regions[side] & am) or not np.any(regions[1 - side] & bm):
            continue
        # A genuinely asymmetric colored map rules out an extra rotation/reflection.
        if evidence and any(np.array_equal(labels, trial) for trial in
                            [np.rot90(labels, k) for k in (1, 2, 3)] +
                            [np.fliplr(np.rot90(labels, k)) for k in range(4)]):
            continue
        a, b = np.where(am, ia, bg).astype(np.int8), np.where(bm, ib, bg).astype(np.int8)
        neutral_mask = rng.random((h, w)) < float(rng.uniform(.25, .75))
        if rng.random() < .35:
            neutral_mask = (rr + cc + int(rng.integers(2))) % 2 == 0
        if evidence:
            neutral_mask[cut & (labels == 0)] = True
            for reg in regions:
                for mask in (am, bm):
                    cells = np.argwhere(reg & ~mask)
                    neutral_mask[tuple(cells[int(rng.integers(len(cells)))])] = True
        if neutral_mask.all() or not neutral_mask.any():
            continue
        ordinary = np.where(neutral_mask, texture, bg).astype(np.int8)
        origin = (int(rng.integers(rows - h + 1)), int(rng.integers(cols - w + 1)))
        fragment = _pick_fragment(rng, labels, geometry, origin, evidence)
        if fragment is None:
            continue
        model = dict(geometry=geometry, divider=divider, bg=bg, path_color=path_color,
                     dot_color=dot_color, ia=ia, ib=ib, cut=cut.copy(), regions=regions,
                     side=side, marker=marker, a=a, b=b, ordinary=ordinary, labels=labels,
                     kind=kind, origin=origin, fragment=fragment)
        return model
    return None


def _fallback(rng, palette, evidence):
    """Constructive readable fallback, still sampled over palette/marker/fragment."""
    divider, bg, path_color, dot_color, ia, ib, texture = palette
    labels = np.array([[1, 3, 0, 2], [0, 1, 2, 3], [2, 0, 3, 1]], np.uint8)
    cut = np.zeros((3, 4), bool)
    cut[1] = True
    regions = _partition(cut)
    side = int(rng.integers(2))
    sites = np.argwhere(regions[side])
    origin = (1, 1)
    geometry = (3, 4, 1, 5, 5)
    ordinary = np.full((3, 4), texture, np.int8)
    ordinary[0, 0] = bg
    fragment = _pick_fragment(rng, labels, geometry, origin, evidence)
    _require(fragment is not None, "constructive fragment failed")
    return dict(geometry=geometry, divider=divider, bg=bg, path_color=path_color,
                dot_color=dot_color, ia=ia, ib=ib, cut=cut, regions=regions, side=side,
                marker=tuple(map(int, sites[int(rng.integers(len(sites)))])),
                a=np.where(labels % 2, ia, bg).astype(np.int8),
                b=np.where(labels >= 2, ib, bg).astype(np.int8), ordinary=ordinary,
                labels=labels, kind="horizontal", origin=origin, fragment=fragment)


def _model(rng, palette, evidence=False, bent=False):
    for _ in range(64):
        model = _new_model(rng, palette, evidence, bent)
        if model is not None:
            return model
    return _fallback(rng, palette, evidence)


def _count(n, name, low, high):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(n) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(n)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base_palette = tuple(map(int, rng.choice(10, 7, replace=False)))
    stable_palette = bool(rng.random() < .35)
    shift = str(rng.choice(("blueprint", "fragment", "cut")))
    puzzle, models, used = {"train": [], "test": []}, [], set()
    first = _model(rng, base_palette, evidence=True)
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            for attempt in range(64):
                palette = base_palette if stable_palette else tuple(map(int, rng.choice(10, 7, replace=False)))
                if split == "train" and index == 0:
                    m = first
                elif split == "train" and index == 1:
                    # Same maps and clues, opposite marked side: controlled evidence.
                    m = dict(first)
                    m["side"] = 1 - first["side"]
                    cells = np.argwhere(m["regions"][m["side"]])
                    m["marker"] = tuple(map(int, cells[int(rng.integers(len(cells)))]))
                elif split == "test" and shift == "fragment":
                    m = dict(first)
                    h, w, _, rows, cols = m["geometry"]
                    m["origin"] = (int(rng.integers(rows - h + 1)), int(rng.integers(cols - w + 1)))
                    excluded = [tuple(old["fragment"]) for old in models
                                if _key(old["labels"]) == _key(m["labels"])]
                    fragment = _pick_fragment(rng, m["labels"], m["geometry"], m["origin"], excluded=excluded)
                    if fragment is None:
                        continue
                    m["fragment"] = fragment
                    m["side"] = int(rng.integers(2))
                    cells = np.argwhere(m["regions"][m["side"]])
                    m["marker"] = tuple(map(int, cells[int(rng.integers(len(cells)))]))
                else:
                    m = _model(rng, palette, bent=(split == "test" and shift == "cut"))
                    if split == "test" and _key(m["labels"]) in {_key(old["labels"]) for old in models}:
                        continue
                pair = _render(m, m["fragment"], m["origin"])
                if _key(pair["input"]) not in used:
                    break
            else:
                # More candidate textures than used inputs guarantee uniqueness.
                m = _fallback(rng, palette, False)
                texture = next(v for v in np.unique(m["ordinary"]) if v != m["bg"])
                for code in range(1, len(used) + 2):
                    m["ordinary"] = np.array([texture if code & (1 << i) else m["bg"]
                                              for i in range(12)], np.int8).reshape(3, 4)
                    pair = _render(m, m["fragment"], m["origin"])
                    if _key(pair["input"]) not in used:
                        break
                else:
                    raise AssertionError("bounded uniqueness fallback exhausted")
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
            models.append(m)
    validate_generated_puzzle(puzzle)
    return puzzle


def _cellwise_expected(d):
    """Specification-level check, including untouched texture and background copies."""
    out = d["grid"].copy()
    h, w, t, _, _ = d["geometry"]
    for (r, c), code in d["missing"].items():
        y, x = t + (r + 1) * (h + t), t + c * (w + t)
        for dr in range(h):
            for dc in range(w):
                p = dr, dc
                value = d["ordinary"][p]
                if d["cut"][p]:
                    if code & 1 and d["a"][p] != d["bg"]:
                        value = d["a"][p]
                    elif code & 2:
                        value = d["b"][p]
                    elif code & 1:
                        value = d["a"][p]
                elif d["marked"][p]:
                    if code & 1:
                        value = d["a"][p]
                elif code & 2:
                    value = d["b"][p]
                out[y + dr, x + dc] = value
    return out


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal pair counts")
    used, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions must be preserved")
            _require(_key(inp) not in used, "duplicate input")
            used.add(_key(inp))
            d = _decode(inp)
            _require(np.array_equal(out, _cellwise_expected(d)), "incorrect repair or changed header/divider")
            _require(not np.array_equal(inp, out), "identity example")
            if split == "train":
                train.append(d)
    missing_types = set().union(*(set(d["missing"].values()) for d in train))
    _require(missing_types == {1, 2, 3}, "training must repair single and overlapping activations")
    _require({d["side"] for d in train} == {0, 1}, "both marked sides must be demonstrated")
    seam_states = set().union(*(set(map(int, d["labels"][d["cut"]])) for d in train))
    _require(seam_states == {0, 1, 2, 3}, "training must expose all seam states")
    _require(any(len(_origins(d["labels"], *d["geometry"][3:], d["observations"], True)) > 1
                 for d in train), "training must need colored joint registration")
    _require(any(np.any((d["ordinary"] != d["a"]) & (d["a"] == d["bg"]) & d["marked"])
                 and np.any((d["ordinary"] != d["b"]) & (d["b"] == d["bg"]) &
                            ~d["cut"] & ~d["marked"]) for d in train), "background erasure not taught")


def _equal(p, q):
    return all(len(p[s]) == len(q[s]) and all(
        all(np.array_equal(a[k], b[k]) for k in ("input", "output"))
        for a, b in zip(p[s], q[s])) for s in ("train", "test"))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(_equal(puzzles[0], generate_puzzle(0)), "seed reproducibility failed")
    _require(_equal(generate_puzzle(567, 10, 4), generate_puzzle(567, 10, 4)), "count determinism failed")
    _require(len({tuple(_key(p["input"]) for p in q["train"] + q["test"]) for q in puzzles}) == 42,
             "cross-seed diversity collapsed")
    geometry, designs, textures, cuts, palettes, origins, fragments = (set() for _ in range(7))
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["train"] + puzzle["test"]:
            d = _decode(pair["input"])
            before = pair["input"].copy()
            _require(np.array_equal(solve(before), pair["output"]), "solve disagreement")
            _require(np.array_equal(before, pair["input"]), "solve mutated input")
            geometry.add(d["geometry"])
            designs.add(_key(d["labels"]))
            textures.add(_key(d["ordinary"] == d["bg"]))
            cuts.add(_key(d["cut"]))
            palettes.add((d["divider"], d["bg"], d["ia"], d["ib"], d["path_color"], d["dot_color"]))
            origins.add(d["origin"])
            fragments.add((_key(d["labels"]), tuple(sorted(d["observations"]))))
        first, twin = [_decode(p["input"]) for p in puzzle["train"][:2]]
        _require(np.array_equal(first["a"], twin["a"]) and np.array_equal(first["b"], twin["b"])
                 and np.array_equal(first["cut"], twin["cut"]) and first["side"] != twin["side"],
                 "controlled marker demonstration lost")
        # Removing either activation map changes an actually repaired tile.
        for code in (1, 2):
            patch = first["patterns"][code]
            _require(code in first["missing"].values() and not np.array_equal(patch, first["ordinary"]),
                     "a parent source is ineffective")
        _require(not np.array_equal(first["patterns"][3], first["a"])
                 and not np.array_equal(first["patterns"][3], first["b"]), "full-copy shortcut survives")
        for code in (1, 2, 3):
            _require(np.array_equal(first["patterns"][code][first["cut"]],
                                   twin["patterns"][code][first["cut"]]),
                     "marker swap changed seam behavior")
            _require(not np.array_equal(first["patterns"][code], twin["patterns"][code]),
                     "marker is ineffective")
        _require(not np.array_equal(first["patterns"][1], first["a"])
                 and not np.array_equal(first["patterns"][2], first["b"]),
                 "ignoring the partition does not change the answer")
        # Color permutation equivariance: roles have no literal color identities.
        perm = np.array([9, 0, 7, 2, 8, 1, 6, 3, 5, 4], np.int8)
        p = puzzle["test"][0]
        _require(np.array_equal(solve(perm[p["input"]]), perm[p["output"]]), "palette equivariance failed")
        # Reflect the microscopic coordinates and the coarse field together,
        # retaining the header's control/A/B role order.
        d = _decode(p["input"])
        h, w, t, rows, cols = d["geometry"]
        reflected = {}
        for role in ("input", "output"):
            original = p[role]
            reflected[role] = original.copy()
            for r in range(rows + 1):
                for c in range(cols):
                    source_c = c if r == 0 else cols - 1 - c
                    y, x = t + r * (h + t), t + c * (w + t)
                    sx = t + source_c * (w + t)
                    reflected[role][y:y + h, x:x + w] = original[y:y + h, sx:sx + w][:, ::-1]
        _require(np.array_equal(solve(reflected["input"]), reflected["output"]),
                 "two-scale reflection equivariance failed")
    _require(len(geometry) >= 35 and len(designs) >= 80 and len(textures) >= 55,
             "structural variation too narrow")
    _require(len(cuts) >= 20 and len(palettes) >= 40 and len(origins) >= 8,
             "control, palette or placement variation too narrow")
    _require({g[:2] for g in geometry} == {(h, w) for h in range(2, 5) for w in range(2, 5)},
             "tile shapes collapsed")
    _require({g[2] for g in geometry} == {1, 2}, "divider thickness variation lost")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "count ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Targeted bad labels: divider corruption, copying whole sources, ink-only copying.
    original = puzzles[0]
    d = _decode(original["train"][0]["input"])
    h, w, t, _, _ = d["geometry"]
    for mutation in ("divider", "whole_source", "ink_only", "wrong_priority"):
        bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in original[s]]
               for s in ("train", "test")}
        out = bad["train"][0]["output"]
        if mutation == "divider":
            out[0, 0] = (int(out[0, 0]) + 1) % 10
        else:
            code = 1 if mutation == "whole_source" else 3
            r, c = next(p for p, q in d["missing"].items() if q == code)
            y, x = t + (r + 1) * (h + t), t + c * (w + t)
            if mutation == "whole_source":
                out[y:y + h, x:x + w] = d["a"]
            elif mutation == "ink_only":
                dr, dc = np.argwhere(d["marked"] & (d["a"] == d["bg"]) &
                                     (d["ordinary"] != d["bg"]))[0]
                out[y + dr, x + dc] = d["ordinary"][dr, dc]
            else:
                dr, dc = np.argwhere(d["cut"] & (d["labels"] == 3))[0]
                out[y + dr, x + dc] = d["ib"]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"validator accepted {mutation}")
    # Two legitimate patches can still be insufficient: reject translation ties.
    ambiguous = None
    for puzzle in puzzles:
        d = _decode(puzzle["train"][0]["input"])
        h, w, t, rows, cols = d["geometry"]
        for positions in combinations(d["complete"], 2):
            observations = {p: d["complete"][p] for p in positions}
            if len(_origins(d["labels"], rows, cols, observations)) <= 1:
                continue
            ambiguous = d["grid"].copy()
            for r in range(rows):
                for c in range(cols):
                    y, x = t + (r + 1) * (h + t), t + c * (w + t)
                    ambiguous[y:y + h, x:x + w] = d["patterns"][observations.get((r, c), 0)]
            break
        if ambiguous is not None:
            break
    _require(ambiguous is not None, "translation ambiguity fixture missing")
    try:
        solve(ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous registration accepted")
    for kind in ("branch", "loop", "extra_region"):
        cut = np.zeros((4, 4), bool)
        if kind == "branch":
            cut[:, 1] = cut[1, 2] = True
        elif kind == "loop":
            cut[1:3, 1:3] = True
        else:
            cut[:, 1:3] = True
        try:
            _partition(cut)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"invalid {kind} cut accepted")
    lists = {s: [{k: v.tolist() for k, v in p.items()} for p in puzzles[0][s]]
             for s in ("train", "test")}
    validate_generated_puzzle(lists)
    return {"seeds": 42, "pairs": sum(len(p[s]) for p in puzzles for s in p),
            "geometry_tuples": len(geometry), "colored_blueprints": len(designs),
            "neutral_patterns": len(textures), "cut_shapes": len(cuts),
            "palette_roles": len(palettes), "origins": len(origins),
            "blueprint_fragments": len(fragments)}
