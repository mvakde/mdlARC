"""Square inspectors: access-limited repair and delivery on a rail forest.

Only NumPy and the standard library are used. Construction retains its intended
rails to make labels; solve reconstructs rails and access from the input alone.
"""

from collections import deque
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a


def _neighbors(p, shape):
    r, c = p
    for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
        if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
            yield q


def _components(mask, diagonal=False):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = ((1, 1), (1, -1), (-1, 1), (-1, -1)) if diagonal else ((1, 0), (-1, 0), (0, 1), (0, -1))
    while cells:
        p = min(cells)
        cells.remove(p)
        queue, part = [p], []
        for r, c in queue:
            part.append((r, c))
            for dr, dc in steps:
                q = r+dr, c+dc
                if q in cells:
                    cells.remove(q)
                    queue.append(q)
        result.append(tuple(part))
    return result


def _at(mask, p):
    return 0 <= p[0] < mask.shape[0] and 0 <= p[1] < mask.shape[1] and bool(mask[p])


def _offset(p, d, t=1):
    return p[0]+t*d[0], p[1]+t*d[1]


def _strong(mask):
    """Membership in any intact orthogonal run of at least three pixels."""
    result = np.zeros_like(mask)
    for axis in (0, 1):
        m, s = (mask, result) if axis == 0 else (mask.T, result.T)
        triples = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for t in range(3):
            s[:, t:t+triples.shape[1]] |= triples
    return result


def _gaps(mask):
    result = []
    for d in ((0, 1), (1, 0)):
        m = mask if d[1] else mask.T
        found = m[:, :-4] & m[:, 4:]
        found &= ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1])
        for r, c in np.argwhere(found):
            p = (int(r), int(c)) if d[1] else (int(c), int(r))
            result.append((p, d))
    return result


def _reconstruct(mask):
    """Return virtual rails, exact repair bundles, and unsupported diagonals.

    The two off-axis pixels are evidence for a bracketed THREE-pixel gap.
    Corrections are discovered simultaneously, never by iterative gap filling.
    """
    strong = _strong(mask)
    repairs, added = [], np.zeros_like(mask)
    anchors = np.zeros_like(mask)
    for p, d in _gaps(mask):
        if not (_at(mask, _offset(p, d, -1)) or _at(mask, _offset(p, d, 5))):
            continue
        matches = []
        for endpoint in (0, 4):
            a = _offset(p, d, endpoint)
            toward = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                step = toward*d[0]+side*d[1], toward*d[1]+side*d[0]
                spur = tuple(_offset(a, step, t) for t in (1, 2))
                if all(_at(mask, q) and not strong[q] for q in spur) and not _at(mask, _offset(a, step, 3)):
                    matches.append(spur)
        if matches:
            _require(len(matches) == 1, "ambiguous repair attachment")
            gap = tuple(_offset(p, d, t) for t in (1, 2, 3))
            for q in gap:
                added[q] = True
            anchors[p] = anchors[_offset(p, d, 4)] = True
            repairs.append((gap, matches[0]))
    weak = mask & ~_strong(mask | added) & ~anchors
    erased = np.zeros_like(mask)
    for part in _components(weak, diagonal=True):
        if len(part) >= 2:
            for q in part:
                erased[q] = True
    parts = _components(erased, diagonal=True)
    repair_spurs = {frozenset(spur) for _, spur in repairs}
    _require(len(repair_spurs) == len(repairs), "reused repair evidence")
    _require(repair_spurs <= {frozenset(p) for p in parts}, "repair spur mixed with other geometry")
    clutter = [p for p in parts if frozenset(p) not in repair_spurs]
    return (mask & ~erased) | added, repairs, clutter


def _parse(inp):
    inp = _grid(inp)
    values = set(map(int, np.unique(inp)))
    _require(len(values) == 4, "four color roles required")
    border = np.concatenate((inp[0], inp[-1], inp[:, 0], inp[:, -1]))
    _require(len(np.unique(border)) == 1, "background frame required")
    bg = int(border[0])
    candidates = []
    for color in values - {bg}:
        points = np.argwhere(inp == color)
        lo, hi = points.min(axis=0), points.max(axis=0)
        k = int(hi[0]-lo[0]+1)
        if k < 3 or k % 2 == 0 or hi[1]-lo[1]+1 != k or len(points) != k*k-1:
            continue
        center = tuple(map(int, (lo+hi)//2))
        box = inp[lo[0]:hi[0]+1, lo[1]:hi[1]+1].copy()
        rail = int(inp[center])
        box[k//2, k//2] = color
        if rail != color and rail != bg and np.all(box == color):
            candidates.append((color, rail, center, k))
    _require(len(candidates) == 1, "source square must uniquely identify cart and rail")
    cart, rail, root, k = candidates[0]
    floor = next(iter(values - {bg, cart, rail}))
    support = inp != bg
    mask = inp == rail
    half = k//2
    # A source chamber has only straight radial rail spokes beneath its marker.
    for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if _at(mask, _offset(root, d, half+1)):
            for t in range(1, half+1):
                mask[_offset(root, d, t)] = True
    _require(sum(_at(mask, q) for q in _neighbors(root, mask.shape)) >= 1, "source has no rail outlet")
    return {"bg": bg, "floor": floor, "rail": rail, "cart": cart,
            "root": root, "k": k, "support": support, "raw": mask}


def _access(rails, support, root, k):
    half = k//2
    windows = np.lib.stride_tricks.sliding_window_view(support, (k, k))
    valid = np.zeros_like(support)
    valid[half:support.shape[0]-half, half:support.shape[1]-half] = windows.all(axis=(-1, -2))
    _require(valid[root] and rails[root], "source footprint is invalid")
    allowed = rails & valid
    reached = np.zeros_like(rails)
    reached[root] = True
    queue = deque([root])
    while queue:
        p = queue.popleft()
        for q in _neighbors(p, rails.shape):
            if allowed[q] and not reached[q]:
                reached[q] = True
                queue.append(q)
    swept = np.zeros_like(rails)
    for r, c in np.argwhere(reached):
        swept[r-half:r+half+1, c-half:c+half+1] = True
    tips = []
    for p in map(tuple, np.argwhere(reached).tolist()):
        if p != root and sum(bool(rails[q]) for q in _neighbors(p, rails.shape)) == 1:
            tips.append(p)
    return reached, swept, tuple(sorted(tips))


def _render(inp, roles, repairs, clutter, reached, swept, tips):
    out = inp.copy()
    active = 0
    for gap, spur in repairs:
        if all(reached[p] for p in gap):
            active += 1
            for p in gap:
                out[p] = roles["rail"]
            for p in spur:
                out[p] = roles["floor"]
    deleted_clutter = 0
    for part in clutter:
        if all(swept[p] for p in part):
            deleted_clutter += 1
            for p in part:
                out[p] = roles["floor"]
    half = roles["k"]//2
    for r, c in tips:
        out[r-half:r+half+1, c-half:c+half+1] = roles["cart"]
        out[r, c] = roles["rail"]
    return out, active, deleted_clutter


def _analyze(inp):
    inp = _grid(inp)
    roles = _parse(inp)
    rails, repairs, clutter = _reconstruct(roles["raw"])
    reached, swept, tips = _access(rails, roles["support"], roles["root"], roles["k"])
    out, active, deleted = _render(inp, roles, repairs, clutter, reached, swept, tips)
    return out, dict(roles, rails=rails, repairs=repairs, clutter=clutter,
                     reached=reached, swept=swept, tips=tips, active=active,
                     deleted_clutter=deleted)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive repair access and terminal deliveries entirely from this input."""
    return _analyze(np.asarray(input_grid))[0].astype(np.int8)


def _tree(rng, nr, nc, count):
    """Random embedded tree, with a cut separating a genuine downstream arm."""
    for _ in range(48):
        start = (int(rng.integers(nr)), int(rng.integers(nc)))
        nodes, edges = {start}, []
        while len(nodes) < count:
            choices = [(a, b) for a in sorted(nodes) for b in _neighbors(a, (nr, nc)) if b not in nodes]
            a, b = choices[int(rng.integers(len(choices)))]
            nodes.add(b)
            edges.append((a, b))
        adjacent = {a: [] for a in nodes}
        for a, b in edges:
            adjacent[a].append(b)
            adjacent[b].append(a)
        leaves = [a for a in sorted(nodes) if len(adjacent[a]) == 1]
        if len(leaves) < 3:
            continue
        root = leaves[int(rng.integers(len(leaves)))]
        parent, order = {root: None}, [root]
        for a in order:
            for b in sorted(adjacent[a]):
                if b not in parent:
                    parent[b] = a
                    order.append(b)
        choices = []
        for b in order[1:]:
            descendants, queue = {b}, [b]
            for a in queue:
                for child in adjacent[a]:
                    if parent.get(child) == a:
                        descendants.add(child)
                        queue.append(child)
            other_leaves = set(leaves) - descendants - {root}
            if len(descendants) >= 2 and other_leaves and parent[b] != root:
                choices.append((parent[b], b, descendants))
        if choices:
            a, b, downstream = choices[int(rng.integers(len(choices)))]
            root_edge = next(e for e in edges if root in e)
            cut_edge = next(e for e in edges if set(e) == {a, b})
            far_edges = [e for e in edges if set(e) <= downstream]
            near_edges = [e for e in edges if not (set(e) & downstream) and e != root_edge]
            return nodes, edges, root, root_edge, cut_edge, far_edges, near_edges
    raise AssertionError("could not sample a branched tree with a downstream arm")


def _coordinates(rng, n, q):
    minimum = max(8, q+2)
    gaps = np.full(n-1, minimum, dtype=int)
    budget = 28-q-int(gaps.sum())
    for j in rng.permutation(n-1):
        extra = int(rng.integers(0, min(3, budget)+1))
        gaps[j] += extra
        budget -= extra
    return q//2 + np.concatenate(([0], np.cumsum(gaps)))


def _edge_points(a, b):
    d = int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1]))
    distance = abs(b[0]-a[0])+abs(b[1]-a[1])
    return [_offset(a, d, t) for t in range(distance+1)], d


def _candidate_gaps(clean, floor, points, root, root_half=2, with_spur=True):
    d = (points[1][0]-points[0][0], points[1][1]-points[0][1])
    candidates = []
    for t in range(1, len(points)-5):
        p = points[t]
        gap = tuple(points[t+1:t+4])
        if not all(clean[z] for z in points[t:t+5]):
            continue
        if any(max(abs(z[0]-root[0]), abs(z[1]-root[1])) <= root_half for z in points[t:t+5]):
            continue
        # Missing segments never contain a turn or branch.
        if any(_at(clean, _offset(z, (d[1], d[0]), side)) for z in gap for side in (-1, 1)):
            continue
        if not with_spur:
            candidates.append((gap, p, d))
            continue
        for endpoint in (0, 4):
            a = points[t+endpoint]
            toward = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                step = toward*d[0]+side*d[1], toward*d[1]+side*d[0]
                spur = tuple(_offset(a, step, j) for j in (1, 2))
                if all(_at(floor, z) and not clean[z] and
                       max(abs(z[0]-root[0]), abs(z[1]-root[1])) > root_half for z in spur):
                    candidates.append((gap, spur))
    return candidates


def _scene(rng, q, large=False, extra_repairs=0, intentional=True):
    nr, nc = (3, 3) if large else (2, 3)
    count = int(rng.integers(7, 10)) if large else int(rng.integers(5, 7))
    nodes, edges, root_node, root_edge, cut_edge, far_edges, near_edges = _tree(rng, nr, nc, count)
    rows, cols = _coordinates(rng, nr, q), _coordinates(rng, nc, q)
    centers = {a: (int(rows[a[0]]), int(cols[a[1]])) for a in nodes}
    floor = np.zeros((int(rows[-1])+q//2+1, int(cols[-1])+q//2+1), bool)
    rails = np.zeros_like(floor)
    half = q//2
    for r, c in centers.values():
        floor[r-half:r+half+1, c-half:c+half+1] = True
    edge_paths = {}
    for e in edges:
        a, b = map(centers.get, e)
        points, d = _edge_points(a, b)
        width = 3 if e == cut_edge else int(rng.choice([5, q]))
        # More throats are compatible in the larger extrapolation, but the
        # root-side demonstration path and designated far repair stay wide.
        if large and e not in (root_edge, cut_edge) and rng.random() < .18:
            width = int(rng.choice([1, 3]))
        for r, c in points:
            rails[r, c] = True
            if d[1]:
                floor[r-width//2:r+width//2+1, c] = True
            else:
                floor[r, c-width//2:c+width//2+1] = True
        edge_paths[e] = (points, width)
    root = centers[root_node]
    clutter = []
    if intentional:
        choices = []
        for e in near_edges:
            points, width = edge_paths[e]
            if width >= 5:
                choices.extend(_candidate_gaps(rails, floor, points, root, with_spur=False))
        _require(choices, "no room for intentional gap")
        gap, anchor, d = choices[int(rng.integers(len(choices)))]
        for z in gap:
            rails[z] = False
        # Offset by one cell ALONG the run: its apparent anchor does not
        # bracket a three-cell gap. This is a visible near miss, not a label.
        side = int(rng.choice([-1, 1]))
        step = d[0]+side*d[1], d[1]+side*d[0]
        fake_anchor = _offset(anchor, d, -1)
        part = tuple(_offset(fake_anchor, step, t) for t in (1, 2))
        if all(_at(floor, z) and not rails[z] for z in part):
            clutter.append(part)
    clean = rails.copy()
    raw = clean.copy()
    repairs = []
    designated = [root_edge, far_edges[int(rng.integers(len(far_edges)))]]
    remainder = [e for e in edges if e not in designated and e != cut_edge and edge_paths[e][1] >= 5]
    rng.shuffle(remainder)
    for e in designated + remainder[:extra_repairs]:
        points, width = edge_paths[e]
        if width < 5:
            continue
        candidates = _candidate_gaps(clean, floor, points, root)
        rng.shuffle(candidates)
        for gap, spur in candidates:
            trial = raw.copy()
            for z in gap:
                trial[z] = False
            for z in spur:
                trial[z] = True
            try:
                recovered, found, _ = _reconstruct(trial)
                if np.array_equal(recovered, clean) and len(found) == len(repairs)+1:
                    raw = trial
                    repairs.append((gap, spur))
                    break
            except AssertionError:
                continue
        else:
            if e in designated:
                raise AssertionError("required repair did not fit")
    _require(len(repairs) >= 2, "two separated repairs required")
    # Try small diagonal clutter at arbitrary floor locations, keeping the
    # full scene reconstruction exact. No region identity decides deletion.
    for _ in range(int(rng.integers(1, 4))):
        for _ in range(30):
            p = (int(rng.integers(raw.shape[0])), int(rng.integers(raw.shape[1])))
            d = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
            part = tuple(_offset(p, d, t) for t in range(int(rng.choice([2, 3]))))
            if not all(_at(floor, z) and not raw[z] and
                       max(abs(z[0]-root[0]), abs(z[1]-root[1])) > 2 for z in part):
                continue
            trial = raw.copy()
            for z in part:
                trial[z] = True
            try:
                recovered, found, free = _reconstruct(trial)
                if np.array_equal(recovered, clean) and len(found) == len(repairs) and all(_diagonal(x) for x in free):
                    raw = trial
                    clutter.append(part)
                    break
            except AssertionError:
                continue
    for part in clutter:
        for z in part:
            raw[z] = True
    recovered, found, free = _reconstruct(raw)
    _require(np.array_equal(recovered, clean) and len(found) == len(repairs), "clutter caused accidental repair")
    _require({frozenset(p) for p in free} == {frozenset(p) for p in clutter}, "clutter merged")
    padding = []
    for size in floor.shape:
        total = int(rng.integers(2, min(7, 30-size)+1))
        before = int(rng.integers(1, total))
        padding.append((before, total-before))
    def shift(p):
        return p[0]+padding[0][0], p[1]+padding[1][0]
    return {"floor": np.pad(floor, padding), "clean": np.pad(clean, padding), "raw": np.pad(raw, padding),
            "root": shift(root), "repairs": [(tuple(map(shift, a)), tuple(map(shift, b))) for a, b in repairs],
            "clutter": [tuple(map(shift, part)) for part in clutter], "nodes": len(nodes)}


def _diagonal(part):
    points = sorted(part)
    return (len(points) in (2, 3) and all(points[j+1][0]-points[j][0] == 1 for j in range(len(points)-1))
            and len({points[j+1][1]-points[j][1] for j in range(len(points)-1)}) == 1
            and abs(points[1][1]-points[0][1]) == 1)


def _pair(scene, k, palette, turns, flip):
    bg, floor, rail, cart = palette
    inp = np.full(scene["floor"].shape, bg, dtype=np.int8)
    inp[scene["floor"]] = floor
    inp[scene["raw"]] = rail
    r, c = scene["root"]
    half = k//2
    inp[r-half:r+half+1, c-half:c+half+1] = cart
    inp[r, c] = rail
    roles = {"floor": floor, "rail": rail, "cart": cart, "k": k}
    reached, swept, tips = _access(scene["clean"], scene["floor"], scene["root"], k)
    out = _render(inp, roles, scene["repairs"], scene["clutter"], reached, swept, tips)[0]
    def transform(a):
        a = np.rot90(a, turns)
        return np.array(np.fliplr(a) if flip else a, dtype=np.int8, copy=True)
    pair = {"input": transform(inp), "output": transform(out)}
    _validate_pair(pair)
    return pair


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, t = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if n <= t:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    q = int(rng.choice([5, 7]))
    puzzle, seen = {"train": [], "test": []}, set()
    # Controlled contrast: the same damaged branched network admits a 3-square
    # through a width-three passage but excludes a 5-square. Both still make
    # a reachable repair and delivery on the root side of that passage.
    for _ in range(96):
        try:
            scene = _scene(rng, q)
            turns, flip = int(rng.integers(4)), bool(rng.integers(2))
            first = [_pair(scene, k, palette, turns, flip) for k in (3, 5)]
            small, big = [_analyze(p["input"])[1] for p in first]
            _require(small["active"] == len(small["repairs"]) and big["active"] < len(big["repairs"]),
                     "intentional break obscured the width contrast")
            _require(small["tips"] != big["tips"], "size contrast must change terminal deliveries")
            for pair in first:
                seen.add(_key(pair["input"]))
            puzzle["train"].extend(first)
            break
        except AssertionError:
            continue
    else:
        raise AssertionError("bounded training contrast construction exhausted")
    for split, count in (("train", n-2), ("test", t)):
        for _ in range(count):
            for _ in range(96):
                try:
                    large = split == "test"
                    scene = _scene(rng, q, large=large, extra_repairs=int(rng.integers(1, 3)) if large else int(rng.integers(2)),
                                   intentional=large or bool(rng.integers(2)))
                    pair = _pair(scene, int(rng.choice([3, 5])), palette, int(rng.integers(4)), bool(rng.integers(2)))
                    _require(_key(pair["input"]) not in seen, "duplicate input")
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
                except AssertionError:
                    continue
            else:
                raise AssertionError("bounded pair construction exhausted")
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(n)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.int8).tobytes()


def _validate_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "shape must be preserved")
    expected, info = _analyze(inp)
    _require(np.array_equal(out, expected), "incorrect repair access or terminal delivery")
    rails, support, k = info["rails"], info["support"], info["k"]
    _require(not np.any(rails[:-1, :-1] & rails[1:, :-1] & rails[:-1, 1:] & rails[1:, 1:]), "thick rails")
    for part in _components(rails):
        cells = set(part)
        edges = sum((r+1, c) in cells for r, c in part) + sum((r, c+1) in cells for r, c in part)
        _require(edges == len(part)-1, "rails must form a forest")
    _require(len(info["repairs"]) >= 1 and info["active"] >= 1, "reachable repair evidence required")
    _require(info["tips"], "at least one terminal delivery required")
    _require(all(_diagonal(p) for p in info["clutter"]), "unsupported diagonal must have two or three pixels")
    for gap, spur in info["repairs"]:
        _require(all(support[p] and inp[p] == info["floor"] for p in gap), "gap must be on floor")
        _require(all(support[p] for p in spur), "spur must be on floor")
        reach = [bool(info["reached"][p]) for p in gap]
        _require(all(reach) or not any(reach), "access boundary may not bisect a repair")
    centers = (info["root"],) + info["tips"]
    _require(all(max(abs(a[0]-b[0]), abs(a[1]-b[1])) >= k for j, a in enumerate(centers) for b in centers[j+1:]),
             "source and delivered squares overlap")
    _require(int(np.sum(out == info["cart"])) == len(centers)*(k*k-1), "square pattern or count changed")
    _require(np.array_equal(inp == info["bg"], out == info["bg"]), "background collision")
    _require(not np.array_equal(inp, out), "identity pair")
    return info


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, t = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= t <= 4 and n > t, "invalid pair counts")
    seen, conventions, training = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _validate_pair(pair)
            conventions.add(tuple(info[name] for name in ("bg", "floor", "rail", "cart")))
            if split == "train":
                training.append(info)
    _require(len(conventions) == 1, "palette roles must be shared within a puzzle")
    _require({i["k"] for i in training} >= {3, 5}, "training needs a size contrast")
    _require(any(i["active"] < len(i["repairs"]) for i in training), "training needs an inaccessible repair")
    _require(any(_gaps(i["rails"]) for i in training), "training needs an intentional break")
    # Counterfactuals use the same input geometry. One-cell access ignores A;
    # erasing deviations without restoring their gaps ignores B.
    no_a, no_b = False, False
    for i in training:
        no_a |= _access(i["rails"], i["support"], i["root"], 1)[2] != i["tips"]
        no_repair = i["rails"].copy()
        for gap, _ in i["repairs"]:
            for p in gap:
                no_repair[p] = False
        no_b |= _access(no_repair, i["support"], i["root"], i["k"])[2] != i["tips"]
    _require(no_a and no_b, "training does not distinguish both parent ablations")


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def _worked_example(k):
    """A hand-specified T: a repair before a throat, one after it, and a break.

    Expected terminal coordinates are specified directly, not found by solve.
    """
    inp = np.zeros((18, 30), dtype=np.int8)
    inp[2:7, 1:29] = 1
    inp[2:17, 11:16] = 1
    inp[2:7, 18:21] = 0
    inp[3:6, 18:21] = 1
    inp[4, 3:27] = 2
    inp[4:15, 13] = 2
    inp[8:11, 13] = 1  # Intentional break: no attached diagonal evidence.
    inp[4, 7:10] = inp[4, 22:25] = 1
    inp[5, 7] = inp[6, 8] = 2
    inp[5, 22] = inp[6, 23] = 2
    h = k//2
    inp[4-h:5+h, 3-h:4+h] = 3
    inp[4, 3] = 2
    out = inp.copy()
    out[4, 7:10] = 2
    out[5, 7] = out[6, 8] = 1
    if k == 3:
        out[4, 22:25] = 2
        out[5, 22] = out[6, 23] = 1
    tips = ((7, 13), (4, 26)) if k == 3 else ((7, 13),)
    for r, c in tips:
        out[r-h:r+h+1, c-h:c+h+1] = 3
        out[r, c] = 2
    return {"input": inp, "output": out}


def self_test() -> dict:
    """Bounded 42-seed coverage, visible invariants, ablations, and mutations."""
    for k in (3, 5):
        pair = _worked_example(k)
        _require(np.array_equal(solve(pair["input"]), pair["output"]), "hand-worked width/repair contrast")
        for turns in range(4):
            for flip in (False, True):
                inp, out = np.rot90(pair["input"], turns), np.rot90(pair["output"], turns)
                if flip:
                    inp, out = np.fliplr(inp), np.fliplr(out)
                _require(np.array_equal(solve(inp), out), "rotation/reflection changes semantics")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    rows, shapes, normalized = [], set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                i = _validate_pair(pair)
                shapes.add(pair["input"].shape)
                norm = np.zeros_like(pair["input"])
                for code, name in enumerate(("bg", "floor", "rail", "cart")):
                    norm[pair["input"] == i[name]] = code
                normalized.add(_key(norm))
                junctions = sum(sum(bool(i["rails"][q]) for q in _neighbors(p, i["rails"].shape)) >= 3
                                for p in map(tuple, np.argwhere(i["rails"]).tolist()))
                rows.append({"split": split, "k": i["k"], "shape": pair["input"].shape,
                             "repairs": len(i["repairs"]), "active": i["active"], "tips": len(i["tips"]),
                             "junctions": junctions, "clutter": len(i["clutter"]),
                             "axes": tuple(sorted({"h" if gap[0][0] == gap[1][0] else "v" for gap, _ in i["repairs"]})),
                             "palette": tuple(i[name] for name in ("bg", "floor", "rail", "cart"))})
                _require(np.array_equal(solve(pair["input"].copy()), pair["output"]), "solve regression")
    _require(len({_fingerprint(p) for p in puzzles}) == 42, "seed variation collapsed")
    _require(len(normalized) >= 140 and len(shapes) >= 25, "structural diversity collapsed")
    _require({r["k"] for r in rows} == {3, 5}, "size coverage")
    _require({axis for r in rows for axis in r["axes"]} == {"h", "v"}, "repair axis coverage")
    _require(any(len(r["axes"]) == 2 for r in rows), "mixed-axis repairs missing")
    _require(max(r["junctions"] for r in rows) >= 3, "branching coverage")
    _require(max(r["tips"] for r in rows) >= 3, "multiple deliveries missing")
    _require(max(r["repairs"] for r in rows) >= 3, "repair count collapsed")
    _require(any(r["active"] < r["repairs"] for r in rows if r["split"] == "test"), "inaccessible test repairs missing")
    _require(max(max(r["shape"]) for r in rows) == 30, "large canvas missing")
    for role in range(4):
        _require({r["palette"][role] for r in rows} == set(range(10)), "palette role coverage")
    # Check every legal count, and deterministic regeneration at maximal count.
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            p = generate_puzzle(901+n*10+t, n, t)
            _require((len(p["train"]), len(p["test"])) == (n, t), "count ignored")
    _require(_fingerprint(generate_puzzle(97, 10, 4)) == _fingerprint(generate_puzzle(97, 10, 4)), "nondeterminism")
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # The full supplied output, not only a list of edits, must be checked.
    for kind in ("omit_repair", "repair_inaccessible", "erase_break", "wall", "square"):
        bad = {s: [{name: a.copy() for name, a in p.items()} for p in puzzles[0][s]] for s in ("train", "test")}
        pair = next(p for p in bad["train"] if _analyze(p["input"])[1]["active"] < len(_analyze(p["input"])[1]["repairs"])
                    and (kind != "erase_break" or _gaps(_analyze(p["input"])[1]["rails"])))
        i = _analyze(pair["input"])[1]
        if kind in ("omit_repair", "repair_inaccessible"):
            gap, spur = next((g, s) for g, s in i["repairs"] if all(i["reached"][p] for p in g) == (kind == "omit_repair"))
            for p in gap+spur:
                pair["output"][p] = pair["input"][p] if kind == "omit_repair" else (i["rail"] if p in gap else i["floor"])
        elif kind == "erase_break":
            point = next(_offset(p, d, j) for p, d in _gaps(i["rails"]) for j in (1, 2, 3)
                         if pair["output"][_offset(p, d, j)] == i["floor"])
            pair["output"][point] = i["rail"]
        elif kind == "wall":
            pair["output"][0, 0] = i["floor"]
        else:
            r, c = i["tips"][0]
            pair["output"][r-i["k"]//2, c-i["k"]//2] = i["floor"]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted mutation: "+kind)
    return {"seeds": 42, "pairs": len(rows), "normalized_inputs": len(normalized), "shapes": len(shapes),
            "square_sizes": [3, 5], "repair_counts": sorted({r["repairs"] for r in rows}),
            "delivery_counts": sorted({r["tips"] for r in rows}), "max_junctions": max(r["junctions"] for r in rows)}
