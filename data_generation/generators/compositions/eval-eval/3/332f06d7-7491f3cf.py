"""A marked, two-material shuttle slides through selective gates.

Only Python's standard library and NumPy are used. No files are read or written.
The square's cut and dot travel with it; gates and walls stay fixed.
"""
from __future__ import annotations

import hashlib
import numpy as np

WALL, FLOOR, BODY = 0, 1, 8
MATERIALS = (2, 3, 4, 5, 6, 7, 9)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(np.all((0 <= a) & (a <= 9)), "invalid color")
    return a.astype(np.int8, copy=True)


def _neighbors(p, shape, diagonal=False):
    r, c = p
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if not (dr or dc) or (not diagonal and abs(dr) + abs(dc) != 1):
                continue
            q = r + dr, c + dc
            if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
                yield q


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        cells = [p]
        for p in cells:
            for q in _neighbors(p, mask.shape, diagonal):
                if q in unseen:
                    unseen.remove(q)
                    cells.append(q)
        result.append(cells)
    return result


def _cut_info(cut):
    """Digital path: diagonal edges do not shortcut an orthogonal elbow."""
    points = set(map(tuple, np.argwhere(cut)))
    graph = {}
    for r, c in points:
        graph[r, c] = [q for q in _neighbors((r, c), cut.shape, True) if q in points
                       and not (q[0] != r and q[1] != c
                                and ((q[0], c) in points or (r, q[1]) in points))]
    ends = [p for p in points if len(graph[p]) == 1]
    _require(len(points) >= 3 and len(ends) == 2
             and all(1 <= len(v) <= 2 for v in graph.values()), "cut must be a simple path")
    border = lambda p: p[0] in (0, cut.shape[0]-1) or p[1] in (0, cut.shape[1]-1)
    _require(all(border(p) == (p in ends) for p in points), "only cut ends touch its boundary")
    _require(len(_components(cut, True)) == 1, "disconnected cut")
    regions = _components(~cut)
    _require(len(regions) == 2, "cut must leave exactly two regions")
    masks, sites = [], []
    for cells in regions:
        m = np.zeros(cut.shape, dtype=bool)
        for p in cells:
            m[p] = True
        masks.append(m)
        sites.append([p for p in cells if not any(cut[q] for q in _neighbors(p, cut.shape, True))])
    _require(all(sites), "each region needs a separated dot site")
    return masks, sites


def _route(mask, k, start):
    """The geometric lane has a single path of full-square placements."""
    valid = np.lib.stride_tricks.sliding_window_view(mask, (k, k)).all(axis=(-1, -2))
    points = set(map(tuple, np.argwhere(valid)))
    graph = {p: [q for q in _neighbors(p, valid.shape) if q in points] for p in points}
    _require(start in graph and len(graph[start]) == 1, "shuttle must start at a lane terminal")
    _require(sum(len(v) == 1 for v in graph.values()) == 2
             and all(1 <= len(v) <= 2 for v in graph.values()), "ambiguous geometric lane")
    path, previous, p = [], None, start
    while p is not None:
        _require(p not in path, "lane loop")
        path.append(p)
        following = [q for q in graph[p] if q != previous]
        previous, p = p, following[0] if following else None
    _require(len(path) == len(points), "unrelated square placements")
    return path


def _decode(value):
    inp = _grid(value)
    cells = np.argwhere(inp == BODY)
    _require(len(cells) > 0, "missing shuttle body")
    lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
    r, c = map(int, lo)
    h, w = map(int, hi-lo)
    _require(h == w and h >= 5, "body bounds must be a square of side at least five")
    template = inp[r:r+h, c:c+h].copy()
    ac = set(map(int, np.unique(template))) - {BODY}
    _require(len(ac) == 1, "shuttle needs one control color")
    a = ac.pop()
    _require(a in MATERIALS, "invalid control material")
    comps = sorted(_components(template == a, True), key=len)
    _require(len(comps) == 2 and len(comps[0]) == 1 and len(comps[1]) > 1,
             "shuttle needs a cut and one isolated dot")
    dot = comps[0][0]
    cut = np.zeros((h, h), dtype=bool)
    for p in comps[1]:
        cut[p] = True
    regions, sites = _cut_info(cut)
    side = 0 if regions[0][dot] else 1
    _require(dot in sites[side], "dot touches cut")
    other_colors = set(map(int, np.unique(inp))) - {WALL, FLOOR, BODY, a}
    _require(len(other_colors) == 1, "exactly two gate materials required")
    b = other_colors.pop()
    _require(b in MATERIALS and b != a, "invalid second material")
    terrain = inp.copy()
    terrain[r:r+h, c:c+h] = FLOOR
    _require(np.any(terrain == a) and np.any(terrain == b), "both materials must occur as gates")
    path = _route(terrain != WALL, h, (r, c))
    allow = np.where(cut, 3, np.where(regions[side], 1, 2)).astype(np.uint8)
    return dict(inp=inp, terrain=terrain, template=template, k=h, start=(r, c),
                a=a, b=b, dot=dot, cut=cut, regions=regions, sites=sites,
                side=side, allow=allow, path=path)


def _accept(d, p, mode="full"):
    r, c = p
    k = d["k"]
    patch = d["terrain"][r:r+k, c:c+k]
    if mode == "ignore_materials":
        return bool(np.all(patch != WALL))
    good = ((patch == FLOOR)
            | ((patch == d["a"]) & ((d["allow"] & 1) != 0))
            | ((patch == d["b"]) & ((d["allow"] & 2) != 0)))
    if mode == "point":
        return bool(good[k//2, k//2])
    return bool(good.all())


def _stop(d, mode="full"):
    end = 0
    for index, p in enumerate(d["path"]):
        if not _accept(d, p, mode):
            break
        end = index
    return end


def _entry_stop(d):
    """Counterfactual that checks only the new leading strip, ignoring old cells."""
    k = d["k"]
    for index in range(1, len(d["path"])):
        p, old = d["path"][index], d["path"][index-1]
        for u in range(k):
            for v in range(k):
                r, c = p[0]+u, p[1]+v
                if old[0] <= r < old[0]+k and old[1] <= c < old[1]+k:
                    continue
                color = d["terrain"][r, c]
                if color == FLOOR:
                    continue
                bit = 1 if color == d["a"] else 2
                if not d["allow"][u, v] & bit:
                    return index-1
    return len(d["path"])-1


def _render(d, index):
    out = d["terrain"].copy()
    r, c = d["path"][index]
    out[r:r+d["k"], c:c+d["k"]] = d["template"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, region, placement and collision from this input only."""
    d = _decode(input_grid)
    return _render(d, _stop(d))


def _cut(rng, k, kind):
    cut = np.zeros((k, k), dtype=bool)
    if kind == "straight":
        cut[k//2, :] = True
    elif kind == "diagonal":
        cut[np.arange(k), np.arange(k)] = True
    else:
        # A left-to-right stepped separator with a single interior vertical run.
        first, second = (2, k-3) if k >= 6 else (1, 3)
        bend = int(rng.integers(1, k-1))
        cut[first, :bend+1] = True
        cut[min(first, second):max(first, second)+1, bend] = True
        cut[second, bend:] = True
    _cut_info(cut)
    return cut


def _expand(nodes):
    path = [nodes[0]]
    for p, q in zip(nodes, nodes[1:]):
        dr, dc = int(np.sign(q[0]-p[0])), int(np.sign(q[1]-p[1]))
        while path[-1] != q:
            path.append((path[-1][0]+dr, path[-1][1]+dc))
    return path


def _layout(rng, k, turns):
    if turns == 0:
        length = int(rng.integers(k+7, 29-k))
        return _expand([(0, 0), (0, length)])
    # A random self-avoiding walk on a spaced lattice. Unused edges stay wall.
    gap = k + int(rng.integers(2, 5))
    side = 1 + (28-k)//gap
    nr, nc = int(rng.integers(2, side+1)), int(rng.integers(2, side+1))
    for _ in range(32):
        node = (int(rng.integers(nr)), int(rng.integers(nc)))
        nodes, used = [node], {node}
        limit = int(rng.integers(3, min(nr*nc, turns+3)+1))
        while len(nodes) < limit:
            options = [q for q in _neighbors(nodes[-1], (nr, nc)) if q not in used]
            if not options:
                break
            node = options[int(rng.integers(len(options)))]
            nodes.append(node)
            used.add(node)
        headings = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(nodes, nodes[1:])]
        if any(a != b for a, b in zip(headings, headings[1:])):
            return _expand([(r*gap, c*gap) for r, c in nodes])
    return _expand([(0, 0), (0, gap), (gap, gap)])


def _canvas(path, k):
    points = np.array(path)
    low = points.min(axis=0)
    path = [tuple(map(int, p-low)) for p in points]
    shape = tuple(map(int, points.max(axis=0)-low+k))
    terrain = np.zeros(shape, dtype=np.int8)
    for r, c in path:
        terrain[r:r+k, c:c+k] = FLOOR
    _require(_route(terrain != WALL, k, path[0]) == path, "layout has a shortcut")
    return terrain, path


def _pack(terrain, cut, dot, start, a):
    template = np.full(cut.shape, BODY, dtype=np.int8)
    template[cut] = a
    template[dot] = a
    inp = terrain.copy()
    r, c = start
    inp[r:r+len(cut), c:c+len(cut)] = template
    return inp


def _orient(rng, inp):
    # Padding, reflection and whole-scene rotation also transform the shuttle.
    budget = [30-n for n in inp.shape]
    pads = []
    for b in budget:
        extra = int(rng.integers(min(3, b)+1))
        before = int(rng.integers(extra+1))
        pads.append((before, extra-before))
    inp = np.pad(inp, pads, constant_values=WALL)
    if rng.integers(2):
        inp = inp[:, ::-1]
    return np.rot90(inp, int(rng.integers(4))).copy()


def _twins(rng, a, b):
    k = 5
    cut = _cut(rng, k, "straight")
    regions, sites = _cut_info(cut)
    side = int(rng.integers(2))
    terrain, path = _canvas(_layout(rng, k, 0), k)
    # Two full bars test both regional roles and both seam materials. They are
    # far enough from the initial footprint to guarantee visible movement.
    for j, col in enumerate((k+2, k+5)):
        width = int(rng.integers(1, 3))
        for r in range(k):
            material = (a if regions[side][r, 0] else b) if not cut[r, 0] else (a, b)[j]
            terrain[r, col:col+width] = material
    inputs = []
    # Use one transform for the exact controlled comparison.
    dot0 = sites[side][int(rng.integers(len(sites[side])))]
    dot1 = sites[1-side][int(rng.integers(len(sites[1-side])))]
    x = _pack(terrain, cut, dot0, path[0], a)
    y = _pack(terrain, cut, dot1, path[0], a)
    # Stack-free transform specification avoids retaining intermediate traces.
    rot, flip = int(rng.integers(4)), bool(rng.integers(2))
    pads = [(int(rng.integers(2)), int(rng.integers(2))) for _ in range(2)]
    for inp in (x, y):
        inp = np.pad(inp, pads, constant_values=WALL)
        inp = np.rot90(inp[:, ::-1] if flip else inp, rot).copy()
        inputs.append(inp)
    if rng.integers(2):
        inputs.reverse()
    return [{"input": x, "output": solve(x)} for x in inputs]


def _new_pair(rng, k, kind, turns, blocked, a, b, interior=False):
    for _ in range(64):
        try:
            cut = _cut(rng, k, kind)
            if interior or rng.integers(2):
                cut = cut.T.copy()
            if rng.integers(2):
                cut = cut[:, ::-1].copy()
            regions, sites = _cut_info(cut)
            side = int(rng.integers(2))
            dot = sites[side][int(rng.integers(len(sites[side])))]
            allow = np.where(cut, 3, np.where(regions[side], 1, 2)).astype(np.uint8)
            terrain, path = _canvas(_layout(rng, k, turns), k)
            capacity = np.full(terrain.shape, 3, dtype=np.uint8)
            first = np.full(terrain.shape, -1, dtype=np.int16)
            planned_stop = int(rng.integers(k+1 if interior else 2, len(path)-1)) if blocked else len(path)-1
            for index, (r, c) in enumerate(path):
                if index <= planned_stop:
                    capacity[r:r+k, c:c+k] &= allow
                patch = first[r:r+k, c:c+k]
                patch[patch < 0] = index
            if interior:
                r, c = path[planned_stop+1]
                patch = capacity[r:r+k, c:c+k]
                entered = first[r:r+k, c:c+k] == planned_stop+1
                patch[entered] &= allow[entered]
            for bit, color in ((1, a), (2, b)):
                candidates = (first > 0) & ((capacity & bit) != 0) & (terrain == FLOOR)
                cells = list(map(tuple, np.argwhere(candidates)))
                _require(cells, "no traversable site for a material")
                # A few connected gate fragments, not a random pixel carpet.
                for _ in range(int(rng.integers(1, 5))):
                    p = cells[int(rng.integers(len(cells)))]
                    terrain[p] = color
                    for _ in range(int(rng.integers(0, k+2))):
                        nxt = [q for q in _neighbors(p, terrain.shape) if candidates[q]]
                        if not nxt:
                            break
                        p = nxt[int(rng.integers(len(nxt)))]
                        terrain[p] = color
            if blocked:
                # Earlier footprint constraints guarantee a precise first failure.
                # Interior evidence accepts a gate at entry, then rejects it when
                # the translating partition brings the opposite region over it.
                index = planned_stop + 1
                r, c = path[index]
                options = [(u, v, bit) for u in range(k) for v in range(k) for bit in (1, 2)
                           if (0 < first[r+u, c+v] < index if interior else first[r+u, c+v] == index)
                           and capacity[r+u, c+v] & bit and not allow[u, v] & bit]
                _require(options, "no unambiguous first blocker")
                u, v, bit = options[int(rng.integers(len(options)))]
                terrain[r+u, c+v] = a if bit == 1 else b
            inp = _orient(rng, _pack(terrain, cut, dot, path[0], a))
            d = _decode(inp)
            stop = _stop(d)
            _require(stop == planned_stop and stop > 0, "wrong requested outcome")
            _require(not interior or _entry_stop(d) > stop, "interior collision also blocked at entry")
            return {"input": inp, "output": _render(d, stop)}
        except AssertionError:
            continue
    raise RuntimeError("bounded gate construction exhausted")


def _counts(num_train, num_test):
    for x, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or not low <= x <= high:
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    a, b = map(int, rng.choice(MATERIALS, 2, replace=False))
    puzzle = {"train": _twins(rng, a, b), "test": []}
    used = {(p["input"].shape, p["input"].tobytes()) for p in puzzle["train"]}
    regime = int(rng.integers(3))
    for split, count in (("train", nt-2), ("test", nq)):
        for pair_index in range(count):
            for _ in range(64):
                test = split == "test"
                k = 7 if test and regime == 2 else int(rng.integers(5, 7))
                kind = str(rng.choice(("diagonal", "stepped"))) if test and regime == 0 else "straight"
                turns = (4 if test and regime == 1 else int(rng.integers(0, 4)))
                if regime == 0 or (regime == 1 and not test):
                    turns = 0
                interior = not test and pair_index == 0
                if interior:
                    kind, turns = "straight", 0
                ca, cb = (a, b) if rng.random() < .4 else map(int, rng.choice(MATERIALS, 2, replace=False))
                blocked = interior or bool(rng.integers(2))
                # Buried collisions are taught explicitly before appearing in tests.
                if test and nt >= 3 and regime != 1 and blocked and rng.random() < .35:
                    interior, kind, turns = True, "straight", 0
                    if regime == 0:
                        interior, kind = False, "diagonal"
                pair = _new_pair(rng, k, kind, turns, blocked, ca, cb, interior)
                key = pair["input"].shape, pair["input"].tobytes()
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _scalar_accept(d, p):
    """Validator checks each local cell, independently of the vector predicate."""
    for u in range(d["k"]):
        for v in range(d["k"]):
            color = int(d["terrain"][p[0]+u, p[1]+v])
            if color == WALL:
                return False
            if color == FLOOR or d["cut"][u, v]:
                continue
            marked = d["regions"][d["side"]][u, v]
            if color != (d["a"] if marked else d["b"]):
                return False
    return True


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = inp.shape, inp.tobytes()
            _require(key not in seen and inp.shape == out.shape, "duplicate or changed dimensions")
            seen.add(key)
            d = _decode(inp)
            end = 0
            for index, p in enumerate(d["path"]):
                if not _scalar_accept(d, p):
                    break
                end = index
            _require(end > 0, "examples must demonstrate actual motion")
            # Full comparison covers gate preservation, restoration, intact stamp,
            # no wall overlap, no rotation, and stopping before the FIRST failure.
            expected = d["terrain"].copy()
            r, c = d["path"][end]
            expected[r:r+d["k"], c:c+d["k"]] = d["template"]
            _require(np.array_equal(out, expected), "incorrect movement or altered terrain")
            _require(np.sum(out == BODY) == np.sum(inp == BODY), "shuttle body changed")
            if split == "train":
                training.append((d, end))
    _require(any(t == len(d["path"])-1 for d, t in training), "missing successful demonstration")
    _require(any(t < len(d["path"])-1 for d, t in training), "missing collision demonstration")
    _require(any(_stop(d, "point") != t for d, t in training), "training does not refute a point mover")
    _require(any(_stop(d, "ignore_materials") != t for d, t in training), "training does not require regional materials")
    _require({d["side"] for d, _ in training} == {0, 1}, "both dot selections must be taught")
    if len(training) >= 3:
        _require(any(_entry_stop(d) > t for d, t in training), "missing demonstrated interior collision")
    return None


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for p in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(p[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, streamed in memory; no artifact or execution-trace storage."""
    signatures, shapes, structures, cuts, sizes, materials, outcomes, bends = [set() for _ in range(8)]
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_fingerprint(p))
        _require(_fingerprint(p) == _fingerprint(generate_puzzle(seed)), "nondeterministic puzzle")
        for split in ("train", "test"):
            for pair in p[split]:
                inp = pair["input"]
                d = _decode(inp)
                _require(np.array_equal(solve(inp.tolist()), pair["output"]), "input-only solve failure")
                shapes.add(inp.shape)
                sizes.add(d["k"])
                materials.update((d["a"], d["b"]))
                cuts.add((d["k"], d["cut"].tobytes()))
                outcomes.add((split, _stop(d) == len(d["path"])-1))
                steps = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(d["path"], d["path"][1:])]
                bends.add(sum(a != b for a, b in zip(steps, steps[1:])))
                normalized = inp.copy()
                normalized[inp == d["a"]] = 2
                normalized[inp == d["b"]] = 3
                structures.add((inp.shape, normalized.tobytes()))
                # Rotation/reflection covariance includes the stencil and gates.
                for transformed, expected in ((np.rot90(inp), np.rot90(pair["output"])),
                                              (inp[:, ::-1], pair["output"][:, ::-1])):
                    _require(np.array_equal(solve(transformed), expected), "geometric covariance failed")
        # Change only the isolated dot of one controlled demonstration.
        first, second = p["train"][:2]
        d0, d1 = _decode(first["input"]), _decode(second["input"])
        _require(np.array_equal(d0["terrain"], d1["terrain"])
                 and np.array_equal(d0["cut"], d1["cut"]) and d0["side"] != d1["side"], "controlled marker comparison lost")
        _require(_stop(d0) != _stop(d1), "marker has no causal effect")
    _require(len(signatures) == 42 and len(structures) >= 150, "insufficient structural variation")
    _require(len(shapes) >= 25 and sizes == {5, 6, 7}, "scale variation missing")
    _require(materials == set(MATERIALS) and len(cuts) >= 12, "material or separator variation missing")
    _require(len(outcomes) == 4 and max(bends) >= 3 and 0 in bends, "outcome/route variation missing")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(42, nt, nq)
            _require((len(p["train"]), len(p["test"])) == (nt, nq), "count ignored")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    p = generate_puzzle(0)
    p["train"][0]["output"][0, 0] = (int(p["train"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "puzzles": len(signatures), "shapes": len(shapes),
            "structures": len(structures), "separator_masks": len(cuts),
            "sizes": sorted(sizes), "route_bends": sorted(bends)}
