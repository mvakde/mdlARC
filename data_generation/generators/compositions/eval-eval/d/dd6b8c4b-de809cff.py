"""Geodesic catchments around reconstructed holes.

Black (0) is background/erasure; azure (8) is a movable token and the
permanent hole highlight. Two other visible colors fill two rectangles.
All conventions are fixed and solve reads only its argument.
"""
from itertools import permutations

import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SLOTS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
         (0, 1), (1, -1), (1, 0), (1, 1))
INF = 10000


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "Grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _signature(a):
    return a.shape, a.tobytes()


def _paint(shape, boxes):
    clean = np.zeros(shape, dtype=np.int8)
    for r0, c0, r1, c1, color in boxes:
        clean[r0:r1, c0:c1] = color
    return clean


def _bounds(mask):
    points = np.argwhere(mask)
    _require(len(points), "Missing fill")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return tuple(map(int, (*lo, *hi)))


def _ring(p):
    return [(p[0] + dr, p[1] + dc) for dr, dc in SLOTS]


def _reconstruct(a):
    """Extremal support fixes boxes; intact overlap cells fix visible order.

    Unlike a general denoiser, this family has no wrong-fill noise. Sparse
    holes and tokens may remove fill evidence but never invent a fill cell.
    """
    fills = tuple(int(x) for x in np.unique(a) if x not in (0, 8))
    _require(len(fills) == 2 and np.any(a == 0) and np.any(a == 8),
             "Expected background, tokens, and exactly two rectangle fills")
    boxes = [(*_bounds(a == color), color) for color in fills]
    _require(all(min(b[2]-b[0], b[3]-b[1]) >= 4 for b in boxes), "Rectangle too thin")
    evidence = (a != 0) & (a != 8)
    candidates = {}
    for order in permutations(boxes):
        clean = _paint(a.shape, order)
        if np.all(clean[evidence] == a[evidence]):
            candidates[clean.tobytes()] = clean
    _require(len(candidates) == 1, "Rectangle covering order is not input-determined")
    clean = next(iter(candidates.values()))
    for b in boxes:
        color = b[4]
        visible = clean == color
        _require(np.count_nonzero(a == color) >= .70 * visible.sum(), "Insufficient intact fill")
        _require(_bounds(visible) == b[:4], "Rectangle needs visible extremal support")
    holes = [tuple(map(int, p)) for p in np.argwhere((a == 0) & (clean != 0))]
    _require(1 <= len(holes) <= 6, "Expected one to six erasures")
    h, w = a.shape
    for i, p in enumerate(holes):
        _require(0 < p[0] < h-1 and 0 < p[1] < w-1, "Hole needs eight in-bounds slots")
        _require(all(a[q] != 8 for q in _ring(p)), "Tokens cannot occupy a ring slot")
        for q in holes[:i]:
            separation = max(abs(p[0]-q[0]), abs(p[1]-q[1]))
            _require(separation >= (2 if clean[p] == clean[q] else 3), "Incompatible hole spacing")
    return clean, fills, boxes, holes


def _distance(clean, p):
    d = np.full(clean.shape, INF, dtype=np.int16)
    d[p] = 0
    queue = [p]
    h, w = clean.shape
    for r, c in queue:
        for dr, dc in STEPS:
            nr, nc = r+dr, c+dc
            if (0 <= nr < h and 0 <= nc < w and clean[nr, nc] == clean[p]
                    and d[nr, nc] == INF):
                d[nr, nc] = d[r, c] + 1
                queue.append((nr, nc))
    return d


def _wave_distance(clean, p):
    """Separate array wavefront for bounded self-test auditing."""
    seen = np.zeros(clean.shape, bool)
    seen[p] = True
    front = seen.copy()
    d = np.full(clean.shape, INF, dtype=np.int16)
    d[p] = 0
    for depth in range(1, clean.size + 1):
        spread = np.zeros(clean.shape, bool)
        spread[1:] |= front[:-1]
        spread[:-1] |= front[1:]
        spread[:, 1:] |= front[:, :-1]
        spread[:, :-1] |= front[:, 1:]
        front = spread & ~seen & (clean == clean[p])
        if not front.any():
            break
        d[front] = depth
        seen |= front
    return d


def _ownership(clean, holes, geometric=False):
    if geometric:
        rr, cc = np.indices(clean.shape)
        ds = np.stack([np.where(clean == clean[p], abs(rr-p[0])+abs(cc-p[1]), INF)
                       for p in holes])
    else:
        ds = np.stack([_distance(clean, p) for p in holes])
    best = ds.min(axis=0)
    tied = (ds == best).sum(axis=0) > 1
    owner = ds.argmin(axis=0).astype(np.int8)
    owner[tied | (best == INF)] = -1
    return ds, owner, tied & (best < INF)


def _collect(a, holes, ds, owner, strict=True):
    selected = np.zeros(a.shape, bool)
    loads, supplies = [], []
    for i in range(len(holes)):
        pool = (a == 8) & (owner == i)
        distances = np.sort(ds[i][pool])
        supply = len(distances)
        if supply > 8:
            if strict:
                _require(distances[7] < distances[8], "Tie straddles capacity frontier")
            # Non-strict mode is only a diagnostic for a geometric shortcut.
            if not strict and distances[7] == distances[8]:
                ranked = sorted((int(ds[i][tuple(p)]), *map(int, p)) for p in np.argwhere(pool))
                for _, r, c in ranked[:8]:
                    selected[r, c] = True
            else:
                selected |= pool & (ds[i] <= distances[7])
        else:
            selected |= pool
        loads.append(min(8, supply))
        supplies.append(supply)
    return selected, loads, supplies


def _render(a, clean, fills, holes, selected, loads, complementary=True):
    result = a.copy()
    result[selected] = clean[selected]
    demands = {}
    for p, load in zip(holes, loads):
        color = next(c for c in fills if c != clean[p]) if complementary else int(clean[p])
        for q in _ring(p)[:load]:
            _require(q not in demands or demands[q] == color, "Conflicting ring colors")
            demands[q] = color
    for q, color in demands.items():
        result[q] = color
    for p in holes:
        result[p] = 8
    return result


def _analyze(a):
    clean, fills, boxes, holes = _reconstruct(a)
    ds, owner, ties = _ownership(clean, holes)
    selected, loads, supplies = _collect(a, holes, ds, owner)
    expected = _render(a, clean, fills, holes, selected, loads)
    return dict(clean=clean, fills=fills, boxes=boxes, holes=holes, distances=ds,
                owner=owner, ties=ties, selected=selected, loads=loads,
                supplies=supplies, expected=expected)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct, partition, collect, and encode, entirely from the input."""
    return _analyze(_grid(input_grid))["expected"]


def _overlap(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def _scene(rng, shape, fills, route):
    h, w = shape
    if route:
        # An interior occluder creates routes around both ends, rather than
        # independent chambers or a decorative maze attached to the scene.
        r0, c0 = map(int, rng.integers(0, 3, 2))
        r1, c1 = h-int(rng.integers(0, 3)), w-int(rng.integers(0, 3))
        vertical = bool(rng.integers(2))
        if vertical:
            br0, br1 = r0+int(rng.integers(3, 6)), r1-int(rng.integers(3, 6))
            width = int(rng.integers(4, 8))
            bc0 = int(rng.integers(c0+5, c1-width-4)); bc1 = bc0+width
        else:
            bc0, bc1 = c0+int(rng.integers(3, 6)), c1-int(rng.integers(3, 6))
            height = int(rng.integers(4, 8))
            br0 = int(rng.integers(r0+5, r1-height-4)); br1 = br0+height
        boxes = [(r0, c0, r1, c1, fills[0]), (br0, bc0, br1, bc1, fills[1])]
    else:
        boxes = []
        for color in fills:
            rh = int(rng.integers(6, h-1)); rw = int(rng.integers(6, w-1))
            r0 = int(rng.integers(0, h-rh+1)); c0 = int(rng.integers(0, w-rw+1))
            boxes.append((r0, c0, r0+rh, c0+rw, color))
        if rng.integers(2):
            boxes.reverse()
    clean = _paint(shape, boxes)
    if any(np.count_nonzero(clean == b[4]) < 30 or _bounds(clean == b[4]) != b[:4] for b in boxes):
        return None
    if np.count_nonzero(clean == 0) < 10:
        return None
    return clean


def _choose_holes(rng, clean, fills, count, mode):
    h, w = clean.shape
    available = [tuple(map(int, p)) for p in np.argwhere(clean != 0)
                 if 0 < p[0] < h-1 and 0 < p[1] < w-1]
    rng.shuffle(available)
    holes = []
    if mode in ("partial", "route"):
        wanted = [fills[0], fills[0], fills[1]]
    else:
        wanted = [int(rng.choice(fills)) for _ in range(count)]
    for color in wanted:
        candidates = [p for p in available if clean[p] == color]
        if mode == "partial":
            candidates = [p for p in candidates if all(clean[q] == color for q in _ring(p))]
        if mode == "route" and len(holes) == 1:
            candidates = [p for p in candidates if (sum(p)-sum(holes[0])) % 2 == 0]
        # A subset of ordinary examples uses same-fill sharing at distance two.
        if mode == "random" and holes and rng.random() < .35:
            candidates.sort(key=lambda p: min(max(abs(p[0]-q[0]), abs(p[1]-q[1])) for q in holes))
        for p in candidates:
            if all(max(abs(p[0]-q[0]), abs(p[1]-q[1])) >=
                   (5 if mode == "partial" else 2 if clean[p] == clean[q] else 3) for q in holes):
                holes.append(p)
                break
        else:
            return None
    return holes


def _pick(rng, points, count, forced=()):
    forced = list(dict.fromkeys(forced))
    if count < len(forced) or count > len(points):
        return None
    remaining = [p for p in points if p not in forced]
    if count == len(forced):
        return forced
    # Dispersed, compact and almost linear supplies, independently sampled.
    style = int(rng.integers(3))
    anchor = forced[0] if forced else points[int(rng.integers(len(points)))]
    if style == 0:
        rng.shuffle(remaining)
    elif style == 1:
        remaining.sort(key=lambda p: (p[0]-anchor[0])**2+(p[1]-anchor[1])**2)
    else:
        axis = int(rng.integers(2))
        remaining.sort(key=lambda p: abs(p[axis]-anchor[axis])*31+abs(p[1-axis]-anchor[1-axis]))
    return forced + remaining[:count-len(forced)]


def _place_supply(rng, points, distance, count, forced=()):
    if count <= 8:
        return _pick(rng, points, count, forced)
    levels = sorted(set(int(distance[p]) for p in points))
    rng.shuffle(levels)
    for cutoff in levels:
        near = [p for p in points if distance[p] <= cutoff]
        far = [p for p in points if distance[p] > cutoff]
        if len(near) >= 8 and len(far) >= count-8 and all(distance[p] <= cutoff for p in forced):
            return _pick(rng, near, 8, forced) + _pick(rng, far, count-8)
    return None


def _make_pair(rng, fills, mode, count, test=False):
    for attempt in range(260):
        low = 24 if mode == "route" or test else 17
        shape = tuple(map(int, rng.integers(low, 31, 2)))
        clean = _scene(rng, shape, fills, mode == "route")
        if clean is None:
            continue
        holes = _choose_holes(rng, clean, fills, count, mode)
        if holes is None:
            continue
        ds, owner, ties = _ownership(clean, holes)
        reserved = set(holes)
        for p in holes:
            reserved.update(_ring(p))
        pools = [[tuple(map(int, p)) for p in np.argwhere(owner == i) if tuple(p) not in reserved]
                 for i in range(len(holes))]
        witness = None
        tie_points = [tuple(map(int, p)) for p in np.argwhere(ties) if tuple(p) not in reserved]
        if mode == "partial":
            counts = [int(rng.choice([2, 3, 4])), 0, int(rng.choice([5, 6, 7]))]
        elif mode == "route":
            _, geometric, _ = _ownership(clean, holes, True)
            reversals = [tuple(map(int, p)) for p in np.argwhere((owner >= 0) & (geometric >= 0)
                         & (owner != geometric)) if tuple(p) not in reserved]
            if not reversals or not tie_points:
                continue
            witness = reversals[int(rng.integers(len(reversals)))]
            partial = int(owner[witness])
            full = int(geometric[witness])
            counts = [int(rng.integers(2, 7)) for _ in holes]
            counts[full] = int(rng.integers(9, 15))
            counts[partial] = int(rng.integers(2, 7))
        else:
            counts = [int(rng.integers(9, 14)) if rng.random() < .20 else int(rng.integers(0, 9)) for _ in holes]
            counts = [min(n, len(pool)) for n, pool in zip(counts, pools)]
            if not any(counts):
                counts[0] = min(1, len(pools[0]))
        tokens = []
        for i, (pool, n) in enumerate(zip(pools, counts)):
            forced = [witness] if witness is not None and owner[witness] == i else []
            chosen = _place_supply(rng, pool, ds[i], n, forced)
            if chosen is None:
                break
            tokens.extend(chosen)
        else:
            exterior = [tuple(map(int, p)) for p in np.argwhere((clean == 0) & (owner < 0))
                        if tuple(p) not in reserved]
            if not exterior:
                continue
            tokens.extend(_pick(rng, exterior, int(rng.integers(1, min(5, len(exterior))+1))))
            if tie_points and (mode == "route" or rng.random() < .65):
                tokens.extend(_pick(rng, tie_points, int(rng.integers(1, min(3, len(tie_points))+1))))
            # Unserved components of a fill can supply additional distractions.
            inactive = [tuple(map(int, p)) for p in np.argwhere((clean != 0) & (ds.min(axis=0) == INF))
                        if tuple(p) not in reserved]
            if inactive and rng.random() < .65:
                tokens.extend(_pick(rng, inactive, int(rng.integers(1, min(5, len(inactive))+1))))
            a = clean.copy()
            for p in holes:
                a[p] = 0
            for p in tokens:
                a[p] = 8
            try:
                info = _analyze(a)
            except ValueError:
                continue
            if not np.array_equal(info["clean"], clean):
                continue
            pair = {"input": a, "output": info["expected"]}
            if mode == "route" and not _features(a, info)["geometric_differs"]:
                continue
            return pair
    raise RuntimeError("Bounded construction could not satisfy the requested evidence")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    _require(_integer(num_train) and 2 <= num_train <= 10, "num_train must be 2..10")
    _require(_integer(num_test) and 1 <= num_test <= 4, "num_test must be 1..4")
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    available = [1, 2, 3, 4, 5, 6, 7, 9]
    base_fills = tuple(map(int, rng.choice(available, 2, replace=False)))
    changing_palette = bool(rng.integers(2))
    modes = ["partial", "route"] + ["random"] * (num_train-2)
    modes = [modes[int(i)] for i in rng.permutation(num_train)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, total in (("train", num_train), ("test", num_test)):
        for i in range(total):
            mode = modes[i] if split == "train" else "random"
            count = 3 if mode != "random" else int(rng.integers(4, 7) if split == "test" else rng.integers(1, 4))
            fills = tuple(map(int, rng.choice(available, 2, replace=False))) if changing_palette else base_fills
            for _ in range(12):
                pair = _make_pair(rng, fills, mode, count, split == "test")
                sig = _signature(pair["input"])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct a distinct input")
    return puzzle


def _features(a, info=None):
    x = _analyze(a) if info is None else info
    clean, holes = x["clean"], x["holes"]
    gd, go, _ = _ownership(clean, holes, True)
    gs, gl, _ = _collect(a, holes, gd, go, strict=False)
    geometric = _render(a, clean, x["fills"], holes, gs, gl)
    full = _render(a, clean, x["fills"], holes, x["selected"], [8]*len(holes))
    wrong_color = _render(a, clean, x["fills"], holes, x["selected"], x["loads"], False)
    token = a == 8
    box_sizes = tuple(sorted((b[2]-b[0], b[3]-b[1]) for b in x["boxes"]))
    ring_cells = [(p, q) for p, n in zip(holes, x["loads"]) for q in _ring(p)[:n]]
    painted_slots = [set(_ring(p)[:n]) for p, n in zip(holes, x["loads"])]
    return dict(shape=a.shape, box_sizes=box_sizes, overlap=_overlap(*x["boxes"]),
                holes=len(holes), loads=tuple(x["loads"]), supplies=tuple(x["supplies"]),
                tokens=int(token.sum()), collected=int(x["selected"].sum()),
                excess=sum(max(0, n-8) for n in x["supplies"]),
                tied_tokens=int((token & x["ties"]).sum()),
                unreachable=int((token & (x["distances"].min(axis=0) == INF)).sum()),
                geometric_differs=not np.array_equal(geometric, x["expected"]),
                full_ring_differs=not np.array_equal(full, x["expected"]),
                same_fill_ring_differs=not np.array_equal(wrong_color, x["expected"]),
                ring_crosses_seam=any(clean[q] != 0 and clean[q] != clean[p] for p, q in ring_cells),
                ring_into_background=any(clean[q] == 0 for _, q in ring_cells),
                shared_rings=any(bool(p & q) for i, p in enumerate(painted_slots) for q in painted_slots[:i]),
                fill_pair=x["fills"])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Wrong puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "Illegal counts")
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Wrong pair schema")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "Shape changed")
            sig = _signature(a)
            _require(sig not in seen, "Duplicate input")
            seen.add(sig)
            x = _analyze(a)
            _require(np.array_equal(b, x["expected"]), "Output disagrees with input-derived rule")
            # Cell conservation distinguishes remaining tokens from new centers.
            outside_centers = np.ones(a.shape, bool)
            for p in x["holes"]:
                outside_centers[p] = False
                _require(b[p] == 8, "Hole center not highlighted")
            _require(np.count_nonzero((b == 8) & outside_centers) ==
                     np.count_nonzero(a == 8)-int(x["selected"].sum()), "Token conservation failed")
            _require(all(n <= 8 for n in x["loads"]), "Capacity exceeded")
            _require(np.array_equal(b[(a == 8) & ~x["selected"]], a[(a == 8) & ~x["selected"]]),
                     "Uncollected token changed")
            f = _features(a, x)
            if split == "train":
                train.append((x, f))
            else:
                _require(4 <= len(x["holes"]) <= 6, "Test must transfer to more collectors")
    _require(any(f["excess"] and f["tied_tokens"] and f["geometric_differs"] for _, f in train),
             "Missing capacity/tie/detour demonstration")
    _require(any(0 in x["loads"] and any(2 <= n <= 4 for n in x["loads"])
                 and {int(x["clean"][p]) for p, n in zip(x["holes"], x["loads"]) if n > 0} == set(x["fills"])
                 and f["full_ring_differs"] and f["same_fill_ring_differs"] for x, f in train),
             "Missing partial, empty, complementary-ring demonstration")
    _require(all(f["unreachable"] for _, f in train), "Training must preserve unreachable supply")


def self_test():
    features, signatures, scenes = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                x = _analyze(pair["input"])
                for p, d in zip(x["holes"], x["distances"]):
                    _require(np.array_equal(d, _wave_distance(x["clean"], p)), "BFS/wavefront disagreement")
                _require(sum(x["loads"]) == int(x["selected"].sum()), "Collection/count disagreement")
                f = _features(pair["input"], x)
                f["split"] = split
                features.append(f)
                canonical = np.zeros(x["clean"].shape, np.int8)
                for i, color in enumerate(x["fills"], 1):
                    canonical[x["clean"] == color] = i
                scenes.add(_signature(canonical))
    _require(len(signatures) == 42 and len(scenes) >= 120, "Structural variation collapsed")
    _require(len({f["shape"] for f in features}) >= 50, "Insufficient size variation")
    _require(len({f["box_sizes"] for f in features}) >= 100, "Insufficient rectangle geometry variation")
    _require(len({f["overlap"] for f in features}) >= 25, "Insufficient occlusion variation")
    _require(len({f["fill_pair"] for f in features}) >= 12, "Insufficient palettes")
    _require(set(n for f in features for n in f["loads"]) == set(range(9)), "Some counter loads absent")
    _require({f["holes"] for f in features} == set(range(1, 7)), "Collector-count variation absent")
    for name in ("geometric_differs", "shared_rings", "ring_crosses_seam", "ring_into_background"):
        _require(sum(f[name] for f in features) >= 5, "Missing interaction: "+name)
    a = generate_puzzle(12345, 2, 1)
    b = generate_puzzle(12345, 2, 1)
    _require(all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p),
             "Determinism failed")
    for nt, ne in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Mutated label accepted")
    # A tied eighth/ninth boundary must fail rather than use coordinate order.
    probe = np.full((7, 7), 1, np.int8)
    hole = (3, 3)
    distance = _distance(probe, hole)
    probe[distance == 3] = 8
    try:
        _collect(probe, [hole], distance[None], np.zeros(probe.shape, np.int8))
    except ValueError:
        pass
    else:
        raise AssertionError("Capacity tie accepted")
    return {"seeds": 42, "pairs": len(features), "distinct_clean_scenes": len(scenes),
            "shapes": len({f["shape"] for f in features}),
            "rectangle_size_pairs": len({f["box_sizes"] for f in features}),
            "overlap_areas": len({f["overlap"] for f in features}),
            "features": {name: sum(bool(f[name]) for f in features) for name in
                         ("excess", "tied_tokens", "geometric_differs", "shared_rings",
                          "ring_crosses_seam", "ring_into_background")}}
