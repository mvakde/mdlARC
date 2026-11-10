"""Inspection cuts select holes whose Moore halos travel through material.

Only Python's standard library and NumPy are used. Labels are painted from
construction masks; solve reconstructs those masks from the input alone.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int64, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _canonical_mask(mask):
    points = np.argwhere(mask)
    y, x = points.min(axis=0)
    v, u = points.max(axis=0)+1
    cropped = mask[y:v, x:u]
    return min(_key(np.rot90(np.fliplr(cropped) if reflected else cropped, k))
               for k in range(4) for reflected in (False, True))


def _roles(a):
    values, counts = np.unique(a[a != 0], return_counts=True)
    if len(values) != 2:
        raise ValueError("field 0, material, and probe colors must all be visible")
    i = int(np.argmax(counts))
    solid, probe = int(values[i]), int(values[1-i])
    if int(counts[i]) < 40 or int(counts[i]) <= 4*int(counts[1-i]):
        raise ValueError("the extensive material must distinguish the point role")
    return solid, probe


def _cuts(a):
    """All boundaries supported by a straight signed run of three cells."""
    delta = np.diff(a.astype(np.int8), axis=0)
    supported = np.zeros(len(delta), dtype=bool)
    for c in range(a.shape[1]-2):
        run = delta[:, c:c+3]
        supported |= np.all(run == 1, axis=1) | np.all(run == -1, axis=1)
    return [0] + (np.flatnonzero(supported)+1).tolist() + [a.shape[0]]


def _recover(a):
    # Probes occupy intended material and count as intact support, not damage.
    observed = a != 0
    yy, xx = _cuts(observed), _cuts(observed.T)
    base = np.zeros_like(observed)
    for y, v in zip(yy, yy[1:]):
        for x, u in zip(xx, xx[1:]):
            tile = observed[y:v, x:u]
            n = int(tile.sum())
            if 2*n == tile.size:
                raise ValueError("a rectangular evidence tile has tied support")
            base[y:v, x:u] = 2*n > tile.size
    if not base.any() or base.all():
        raise ValueError("both intended material and field must be present")
    return base


def _spans(points):
    points = sorted(points)
    degrees = Counter()
    spans = []
    for i, a in enumerate(points):
        for b in points[i+1:]:
            d = b[0]-a[0]
            if d and abs(d) == abs(b[1]-a[1]):
                if d < 3:
                    raise ValueError("inspection spans need at least four cells")
                s = 1 if b[1] > a[1] else -1
                spans.append((a, b, s))
                degrees[a] += 1
                degrees[b] += 1
    if not spans or any(n > 1 for n in degrees.values()):
        raise ValueError("inspection endpoints need unique diagonal partners")
    return spans


def _on_span(p, span):
    a, b, s = span
    return a[0] <= p[0] <= b[0] and p[1]-a[1] == s*(p[0]-a[0])


def _span_mask(shape, spans):
    rows, cols = np.indices(shape)
    count = np.zeros(shape, dtype=np.int8)
    for a, b, s in spans:
        count += ((rows >= a[0]) & (rows <= b[0])
                  & (cols-a[1] == s*(rows-a[0])))
    return count


def _dilate(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[y:y+h, x:x+w]
                                 for y in range(3) for x in range(3)])


def _diagonal_run(base, p, slope):
    """Input solver: select the contiguous true run on an ordered diagonal."""
    rows = np.arange(base.shape[0])
    cols = p[1] + slope*(rows-p[0])
    valid = (cols >= 0) & (cols < base.shape[1])
    rows, cols = rows[valid], cols[valid]
    flags = base[rows, cols]
    index = int(np.flatnonzero(rows == p[0])[0])
    lo = hi = index
    while lo > 0 and flags[lo-1]:
        lo -= 1
    while hi+1 < len(flags) and flags[hi+1]:
        hi += 1
    return [(int(r), int(c)) for r, c in zip(rows[lo:hi+1], cols[lo:hi+1])]


def _analyze(input_grid):
    a = _grid(input_grid)
    solid, probe = _roles(a)
    base = _recover(a)
    markers = set(map(tuple, np.argwhere(a == probe)))
    if any(not base[p] for p in markers):
        raise ValueError("probes must lie in intended material")
    spans = _spans(markers)
    holes = base & (a == 0)
    extras = ~base & (a == solid)
    membership = _span_mask(a.shape, spans)
    if np.any(holes & (membership > 1)):
        raise ValueError("a hole at two inspection slopes has no unique launch")
    active = holes & (membership == 1)
    paths, center = [], np.zeros_like(base)
    for point in map(tuple, np.argwhere(active)):
        span = next(t for t in spans if _on_span(point, t))
        path = _diagonal_run(base, point, -span[2])
        paths.append((point, -span[2], path))
        for q in path:
            center[q] = True
    ribbon = _dilate(center) & ~holes
    return {"input": a, "solid": solid, "probe": probe, "base": base,
            "markers": markers, "spans": spans, "holes": holes,
            "extras": extras, "active": active, "center": center,
            "ribbon": ribbon, "paths": paths, "membership": membership}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, cuts, defects, and plume limits from this input only."""
    z = _analyze(input_grid)
    out = np.where(z["base"], z["solid"], 0).astype(np.int64)
    out[z["ribbon"]] = z["probe"]
    out[z["holes"]] = 0
    for p in z["markers"]:
        out[p] = z["probe"]
    return out


def _constructed_pair(base, holes, extras, markers, spans, solid, probe):
    """Paint labels from explicit geometry, without calling solve/recovery."""
    inp = np.where(base, solid, 0).astype(np.int64)
    inp[holes] = 0
    inp[extras] = solid
    for q in markers:
        inp[q] = probe
    out = np.where(base, solid, 0).astype(np.int64)
    h, w = base.shape
    for r, c in np.argwhere(holes):
        owners = [s for s in spans if _on_span((r, c), s)]
        if len(owners) != 1:
            continue
        slope = -owners[0][2]
        # Construction instead walks from the source in both directions and
        # explicitly stamps its transported 3x3 neighborhood at each center.
        out[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = probe
        for sign in (-1, 1):
            y, x = int(r)+sign, int(c)+sign*slope
            while 0 <= y < h and 0 <= x < w and base[y, x]:
                out[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)] = probe
                y, x = y+sign, x+sign*slope
    out[holes] = 0
    for q in markers:
        out[q] = probe
    return {"input": inp, "output": out}


def _box(rng, n, minimum):
    length = int(rng.integers(minimum, n+1))
    spare = n-length
    start = int(rng.integers(spare+1))
    if rng.random() < .35:
        start = int(rng.choice((0, spare)))
    return start, start+length


def _make_base(rng, kind, band):
    low, high = ((16, 25), (23, 31), (14, 31))[band]
    h, w = map(int, rng.integers(low, high, size=2))
    if kind == "cavity":
        h, w = max(20, h), max(20, w)
    if kind == "separate":
        w = max(17, w)
    base = np.zeros((h, w), dtype=bool)
    if kind == "rectangle":
        y, v = _box(rng, h, 12)
        x, u = _box(rng, w, 12)
        base[y:v, x:u] = True
    elif kind == "union":
        y, v = _box(rng, h, 12)
        x, u = _box(rng, w, 12)
        thick = int(rng.integers(4, min(9, u-x-3)))
        left = int(rng.integers(x, u-thick+1))
        base[y:v, left:left+thick] = True
        for _ in range(int(rng.integers(1, 4))):
            tall = int(rng.integers(4, min(9, v-y-3)))
            top = int(rng.integers(y, v-tall+1))
            lo = int(rng.integers(x, left+1))
            hi = int(rng.integers(left+thick, u+1))
            base[top:top+tall, lo:hi] = True
    elif kind == "cavity":
        y, v = _box(rng, h, 18)
        x, u = _box(rng, w, 18)
        base[y:v, x:u] = True
        cy = int(rng.integers(y+4, v-7))
        cx = int(rng.integers(x+4, u-7))
        cv = int(rng.integers(cy+4, v-3))
        cu = int(rng.integers(cx+4, u-3))
        base[cy:cv, cx:cu] = False
        if min(cv-cy, cu-cx) >= 10 and rng.random() < .5:
            base[cy+3:cv-3, cx+3:cu-3] = True
    elif kind == "bands":
        y = int(rng.integers(0, 3))
        while y < h-3:
            thick = int(rng.integers(4, min(9, h-y)+1))
            base[y:y+thick, :] = True
            y += thick+int(rng.integers(3, 7))
    elif kind == "separate":
        cut = int(rng.integers(7, w-8))
        for lo, hi in ((0, cut), (cut+3, w)):
            y, v = _box(rng, h, 8)
            x, u = _box(rng, hi-lo, 5)
            base[y:v, lo+x:lo+u] = True
    else:
        raise ValueError("unknown material geometry")
    base = np.rot90(base, int(rng.integers(4)))
    if rng.random() < .5:
        base = np.fliplr(base)
    return base.copy()


def _sample_spans(rng, base, count, forced_slope=None):
    h, w = base.shape
    spans, endpoints = [], set()
    solid_points = list(map(tuple, np.argwhere(base)))
    for i in range(count):
        for _ in range(180):
            s = forced_slope if i == 0 and forced_slope is not None else int(rng.choice((-1, 1)))
            a = solid_points[int(rng.integers(len(solid_points)))]
            steps = int(rng.integers(4, min(h, w)))
            b = (a[0]+steps, a[1]+s*steps)
            if not (0 <= b[0] < h and 0 <= b[1] < w and base[b]):
                continue
            if any(abs(p[0]-q[0]) == abs(p[1]-q[1])
                   for p in (a, b) for q in endpoints):
                continue
            spans.append((a, b, s))
            endpoints.update((a, b))
            break
        else:
            return None
    return spans, endpoints


def _choose(rng, points):
    return points[int(rng.integers(len(points)))] if points else None


def _latent_centers(base, holes, spans):
    center = np.zeros_like(base)
    for p in map(tuple, np.argwhere(holes)):
        owners = [s for s in spans if _on_span(p, s)]
        if len(owners) == 1:
            for q in _diagonal_run(base, p, -owners[0][2]):
                center[q] = True
    return center


def _features(z):
    base, holes = z["base"], z["holes"]
    inactive = holes & ~z["active"]
    support = _dilate(z["center"])
    through = False
    stop_reentry = False
    for p, slope, path in z["paths"]:
        for sign in (-1, 1):
            side = sorted([q for q in path if sign*(q[0]-p[0]) > 0],
                          key=lambda q: sign*(q[0]-p[0]))
            for i, q in enumerate(side):
                if inactive[q] and i+3 < len(side) and i >= 2:
                    through = True
            end = side[-1] if side else p
            r, c = end[0]+sign, end[1]+sign*slope
            met_gap = False
            while 0 <= r < base.shape[0] and 0 <= c < base.shape[1]:
                if not base[r, c]:
                    met_gap = True
                elif met_gap:
                    stop_reentry = True
                    break
                r, c = r+sign, c+sign*slope
    beyond = False
    for p in map(tuple, np.argwhere(inactive & ~support)):
        for a, b, s in z["spans"]:
            if p[1]-a[1] == s*(p[0]-a[0]) and not a[0] <= p[0] <= b[0]:
                beyond = True
    cover = np.zeros_like(base, dtype=np.int16)
    for _, _, path in z["paths"]:
        path_mask = np.zeros_like(base)
        for p in path:
            path_mask[p] = True
        cover += _dilate(path_mask)
    return {"shape": tuple(base.shape), "spans": len(z["spans"]),
            "lengths": tuple(b[0]-a[0]+1 for a, b, _ in z["spans"]),
            "triggers": tuple(int(sum(_on_span(p, span)
                             for p in map(tuple, np.argwhere(z["active"]))))
                              for span in z["spans"]),
            "slopes": tuple(s for _, _, s in z["spans"]),
            "holes": int(holes.sum()), "active": int(z["active"].sum()),
            "extras": int(z["extras"].sum()), "through_hole": through,
            "beyond_endpoint": beyond, "stop_reentry": stop_reentry,
            "inactive_uncovered": int((inactive & ~support).sum()),
            "latent_intact": int(((z["membership"] > 0) &
                                  (z["input"] == z["solid"]) & ~support).sum()),
            "visible_erasure": int((z["extras"] & ~z["ribbon"]).sum()),
            "spill": int((z["ribbon"] & ~base).sum()),
            "overlap": bool(np.any((cover > 1) & ~holes)),
            "long_plume": max((len(p) for _, _, p in z["paths"]), default=0),
            "roles": (z["solid"], z["probe"]),
            "unpaired": len(z["markers"])-2*len(z["spans"]),
            "cut_across_field": bool(np.any((z["membership"] > 0) & ~base))}


def _ablation(z, omit):
    """Two specific shortcuts used to check teaching evidence, not a solver."""
    center = np.zeros_like(z["base"])
    if omit == "A":
        # Give the shortcut its original sources, but let observed missing
        # material obstruct propagation instead of virtually repairing it.
        route = z["input"] != 0
        route[z["active"]] = True
        sources = list(map(tuple, np.argwhere(z["active"])))
    elif omit in ("B", "gap"):
        # On a one-span witness, launching every hole in that span's transverse
        # direction removes only the finite-span gate.
        if omit == "B":
            assert len(z["spans"]) == 1
        route = z["base"]
        sources = list(map(tuple, np.argwhere(z["holes"] if omit == "B" else z["active"])))
    else:
        raise ValueError("unknown ablation")
    for p in sources:
        owner = (next(s for s in z["spans"] if _on_span(p, s))
                 if omit in ("A", "gap") else z["spans"][0])
        if omit == "gap":
            # Ignore only the first-boundary stop; paint every recovered solid
            # center on this infinite transverse diagonal, even after a gap.
            path = [q for q in map(tuple, np.argwhere(route))
                    if q[1]-p[1] == -owner[2]*(q[0]-p[0])]
        else:
            path = _diagonal_run(route, p, -owner[2])
        for q in path:
            center[q] = True
    out = np.where(z["base"], z["solid"], 0)
    out[_dilate(center) & ~z["holes"]] = z["probe"]
    out[z["holes"]] = 0
    for p in z["markers"]:
        out[p] = z["probe"]
    return out


def _make_pair(rng, solid, probe, kinds, band, span_count, anchor=False,
               slope=None, multi=False, reentry=False):
    # Bounded proposals vary actual geometry; there is no fixed sample bank.
    for _ in range(350):
        base = _make_base(rng, str(rng.choice(kinds)), band)
        if int(base.sum()) < 70 or int((~base).sum()) < 8:
            continue
        try:
            if not np.array_equal(_recover(base.astype(int)), base):
                continue
        except ValueError:
            continue
        geometry = _sample_spans(rng, base, span_count, slope)
        if geometry is None:
            continue
        spans, markers = geometry
        membership = _span_mask(base.shape, spans)
        holes = np.zeros_like(base)
        active_pools = []
        for span in spans:
            points = [p for p in map(tuple, np.argwhere(base & (membership == 1)))
                      if _on_span(p, span) and p not in markers]
            active_pools.append(points)
        failed = False
        for i, pool in enumerate(active_pools):
            floor = (2 if multi else 1) if anchor and i == 0 else 0
            wanted = max(floor, int(rng.choice((0, 1, 1, 2, 3))))
            if len(pool) < wanted:
                failed = True
                break
            for index in rng.permutation(len(pool))[:wanted]:
                holes[pool[int(index)]] = True
        if failed or not holes.any():
            continue
        centers = _latent_centers(base, holes, spans)
        ribbon = _dilate(centers)
        inert_pool = [p for p in map(tuple, np.argwhere(base & (membership == 0)))
                      if p not in markers]
        if anchor:
            # A second missing center on a plume, with enough material after it
            # to distinguish virtual repair from stopping at observed zeros.
            through_pool = []
            for p in map(tuple, np.argwhere(holes)):
                owner = next(s for s in spans if _on_span(p, s))
                path = _diagonal_run(base, p, -owner[2])
                for q in path:
                    distance = abs(q[0]-p[0])
                    sign = 1 if q[0] > p[0] else -1
                    tail = (q[0]+3*sign, q[1]-3*sign*owner[2])
                    if distance >= 3 and q in inert_pool and tail in path:
                        through_pool.append(q)
            q = _choose(rng, through_pool)
            if q is None:
                continue
            holes[q] = True
            beyond_pool = [p for p in inert_pool if not ribbon[p] and any(
                p[1]-a[1] == s*(p[0]-a[0]) and not a[0] <= p[0] <= b[0]
                for a, b, s in spans)]
            q = _choose(rng, beyond_pool)
            if q is None:
                continue
            holes[q] = True
        # Optional isolated holes, dimers and L motifs; only off-cut damage is
        # added here, so the independently sampled trigger vector is retained.
        for _ in range(int(rng.integers(1, 5))):
            p = _choose(rng, inert_pool)
            if p is None:
                break
            motifs = (((0, 0),), ((0, 0), (1, 0)),
                      ((0, 0), (1, 1)), ((0, 0), (1, 0), (1, 1)))
            motif = motifs[int(rng.integers(len(motifs)))]
            for dy, dx in motif:
                q = (p[0]+dy, p[1]+dx)
                if q in inert_pool:
                    holes[q] = True
        # Optional unpaired probes remain unchanged, including when a plume
        # passes through them. They must have no diagonal partner at all.
        for _ in range(int(rng.choice((0, 0, 1, 2)))):
            pool = [p for p in map(tuple, np.argwhere(base & ~holes))
                    if p not in markers and not any(
                        abs(p[0]-q[0]) == abs(p[1]-q[1]) for q in markers)]
            q = _choose(rng, pool)
            if q is not None:
                markers.add(q)
        extras = np.zeros_like(base)
        outside = list(map(tuple, np.argwhere(~base)))
        wanted = int(rng.integers(1 if anchor else 0, 7))
        if anchor:
            q = _choose(rng, [p for p in outside if not ribbon[p]])
            if q is None:
                continue
            extras[q] = True
        for index in rng.permutation(len(outside)):
            if int(extras.sum()) >= wanted:
                break
            extras[outside[int(index)]] = True
        if int(holes.sum()) > max(20, int(base.sum())//5):
            continue
        pair = _constructed_pair(base, holes, extras, markers, spans, solid, probe)
        try:
            z = _analyze(pair["input"])
        except ValueError:
            continue
        if not np.array_equal(z["base"], base):
            continue
        f = _features(z)
        if f["inactive_uncovered"] == 0 or f["long_plume"] < 4:
            continue
        if anchor and not (f["through_hole"] and f["beyond_endpoint"]
                           and f["visible_erasure"] and f["spill"]):
            continue
        if anchor and any(np.array_equal(_ablation(z, concept), pair["output"])
                          for concept in ("A", "B")):
            continue
        if anchor and multi and not f["overlap"]:
            continue
        if anchor and multi and not f["latent_intact"]:
            continue
        if reentry and (not f["stop_reentry"] or
                        np.array_equal(_ablation(z, "gap"), pair["output"])):
            continue
        if not np.array_equal(solve(pair["input"]), pair["output"]):
            raise AssertionError("latent painter and input reconstruction disagree")
        return pair
    raise RuntimeError("bounded geometric proposals exhausted")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(np.arange(1, 10))))
    probe, default_solid = palette[:2]
    varying_material = bool(rng.random() < .35)
    material_choices = palette[1:]
    regime = int(rng.integers(3))
    puzzle, used = {"train": [], "test": []}, set()
    # Two shuffled witnesses teach both slopes and all restrictions, even at
    # the minimum count. No feature is attached to a visible example index.
    specs = [{"anchor": True, "slope": int(s), "multi": i == 0}
             for i, s in enumerate(rng.permutation((-1, 1)))]
    specs += [{} for _ in range(int(num_train)-2)]
    all_kinds = ("rectangle", "bands", "union", "cavity", "separate")
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(count):
            spec = specs[i] if split == "train" else {}
            anchor = spec.get("anchor", False)
            solid = (material_choices[i % len(material_choices)] if split == "train"
                     else int(rng.choice(material_choices[2:]))) if varying_material else default_solid
            if anchor:
                # The second witness has a true gap followed by more material.
                # This rules out painting every material cell on an infinite
                # diagonal, which ordinary rectangles cannot distinguish.
                kinds = ("rectangle",) if i == 0 else ("cavity", "separate")
                band, span_count = 0, 1
            elif split == "train":
                kinds = ("rectangle", "bands", "separate") if regime == 0 else all_kinds
                band = 0 if regime == 2 else 2
                span_count = int(rng.integers(1, 3))
            else:
                kinds = ("cavity", "union") if regime == 0 else all_kinds
                band = 1 if regime == 2 or regime == 0 else 2
                span_count = (int(rng.integers(3, 5)) if regime == 1 else
                              int(rng.integers(2 if regime == 0 else 1, 4)))
            for _ in range(12):
                pair = _make_pair(rng, solid, probe, kinds, band, span_count,
                                  reentry=(anchor and i == 1) or
                                  (split == "test" and regime == 0), **spec)
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not obtain distinct inputs")
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise AssertionError("puzzle schema")
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records, probe_colors = set(), {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            z = _analyze(a)
            probe_colors.add(z["probe"])
            assert np.array_equal(solve(a), out), "incorrect plume output"
            assert not np.array_equal(a, out)
            # Family-specific preservation, propagation, and erasure checks.
            assert np.all(out[z["holes"]] == 0), "missing centers must survive"
            assert all(out[p] == z["probe"] for p in z["markers"])
            assert np.all(out[z["extras"] & ~z["ribbon"]] == 0)
            assert np.all(out[z["ribbon"]] == z["probe"])
            unchanged = ~(z["extras"] | z["ribbon"])
            assert np.array_equal(out[unchanged], a[unchanged])
            assert np.all(z["center"] <= z["base"])
            assert int(z["holes"].sum()) <= max(20, int(z["base"].sum())//5)
            f = _features(z)
            assert f["active"] and f["inactive_uncovered"] and f["long_plume"] >= 4
            records[split].append(f)
    assert len(probe_colors) == 1, "plume/probe hue is puzzle-wide"
    teaching = [f for f in records["train"] if f["spans"] == 1 and f["through_hole"]
                and f["beyond_endpoint"] and f["visible_erasure"] and f["spill"]]
    assert len(teaching) >= 2
    assert set(s for f in teaching for s in f["slopes"]) == {-1, 1}
    assert any(f["active"] >= 2 for f in teaching)
    assert any(f["overlap"] for f in teaching), "teach union of transported halos"
    assert any(f["latent_intact"] for f in teaching), "inspection spans remain latent"
    assert any(f["stop_reentry"] for f in teaching), "teach stopping before a true gap"
    assert any(not np.array_equal(_ablation(_analyze(p["input"]), "gap"), p["output"])
               for p in puzzle["train"]), "teach non-reentry observably"
    # Removing A's virtual repair and removing B's finite activation gate must
    # each change the answer on actual one-span teaching inputs.
    for pair in puzzle["train"]:
        z = _analyze(pair["input"])
        f = _features(z)
        if f in teaching:
            for concept in ("A", "B"):
                assert not np.array_equal(_ablation(z, concept), pair["output"])


def self_test():
    """Bounded 42-seed diversity and semantic regression checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, shapes, structures, palettes, scaffolds = [], set(), set(), set(), set()
    for p in puzzles:
        for split in ("train", "test"):
            for pair in p[split]:
                z = _analyze(pair["input"])
                f = _features(z)
                records.append(f)
                shapes.add(f["shape"])
                palettes.add(f["roles"])
                scaffolds.add(_canonical_mask(z["base"]))
                normalized = np.where(z["base"], 1, 0).astype(np.int8)
                normalized[z["holes"]] = 2
                normalized[z["extras"]] = 3
                for q in z["markers"]:
                    normalized[q] = 4
                structures.add(_key(normalized))
        assert len({_key(pair["input"]) for pair in p["train"]+p["test"]}) == 5
    assert len(shapes) >= 45 and len(structures) >= 190 and len(palettes) >= 25
    assert len(scaffolds) >= 100, "geometry diversity after cropping and symmetries"
    assert {1, 2, 3, 4} <= {f["spans"] for f in records}
    assert len({f["holes"] for f in records}) >= 9
    assert len({f["active"] for f in records}) >= 5
    assert {0, 1, 2, 3, 4, 5, 6} <= {f["extras"] for f in records}
    for flag in ("through_hole", "beyond_endpoint", "stop_reentry", "overlap",
                 "cut_across_field"):
        assert any(f[flag] for f in records), flag
    assert any(f["unpaired"] for f in records)
    assert any(0 in f["triggers"] for f in records if f["spans"] > 1)
    assert max(f["long_plume"] for f in records) >= 20
    for seed in (0, 17, 41):
        a, b = generate_puzzle(seed), generate_puzzle(seed)
        assert all(np.array_equal(x[k], y[k]) for s in a
                   for x, y in zip(a[s], b[s]) for k in x)
    # Symmetries must commute with reconstruction and propagation.
    for p in puzzles[:4]:
        for pair in p["train"]+p["test"]:
            for k in range(4):
                for reflect in (False, True):
                    def transform(a):
                        return np.rot90(np.fliplr(a) if reflect else a, k).copy()
                    assert np.array_equal(solve(transform(pair["input"])),
                                          transform(pair["output"]))
    # Meaningful frozen geometry fixture, rather than implementation-mirroring
    # random equality tests: a cavity interrupts an active opposite diagonal.
    base = np.zeros((20, 20), dtype=bool)
    base[1:19, 1:19] = True
    base[9:13, 2:6] = False
    holes, extras = np.zeros_like(base), np.zeros_like(base)
    holes[6, 8] = True
    holes[3, 11] = True  # transparent off-cut hole along the plume
    holes[15, 17] = True  # collinear beyond the inspection endpoint
    extras[0, 3] = True
    markers = {(3, 5), (11, 13)}
    spans = [((3, 5), (11, 13), 1)]
    pair = _constructed_pair(base, holes, extras, markers, spans, 2, 6)
    out = solve(pair["input"])
    assert np.array_equal(out, pair["output"])
    assert out[3, 11] == 0 and out[2, 12] == 6
    assert out[15, 17] == 0 and out[15, 16] == 2
    assert out[0, 3] == 0
    assert out[10, 4] == 0 and out[13, 1] == 2, "stop before cavity, do not reenter"
    # Damage a held output and ensure validation detects it.
    bad = {s: [{k: a.copy() for k, a in pair.items()} for pair in puzzles[0][s]]
           for s in ("train", "test")}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")


if __name__ == "__main__":
    self_test()
    print(json.dumps({"self_test": "passed", "seeds": 42}, separators=(",", ":")))
