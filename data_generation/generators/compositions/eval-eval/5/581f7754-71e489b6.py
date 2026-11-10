"""Negative-space docking: rectilinear defects act as virtual registration marks.

The decoder uses only visible pixels. Construction labels use explicit local
masks and a separate renderer, never solve(). No parent code is imported.
"""
from collections import Counter
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(((0 <= a) & (a <= 9)).all()), "invalid color")
    return a


def _signature(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _spans(points, axis):
    """Fill between the first and last occupied cell on every row/column."""
    runs = {}
    for p in points:
        runs.setdefault(p[1 - axis], []).append(p[axis])
    filled = set()
    for fixed, values in runs.items():
        for moving in range(min(values), max(values) + 1):
            filled.add((moving, fixed) if axis == 0 else (fixed, moving))
    return filled


def _convex(points):
    return _spans(points, 0) == points and _spans(points, 1) == points


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _shift_for(holes, axis, guide):
    coordinates = {p[axis] for p in holes}
    _require(coordinates, "a plate needs at least one missing cell")
    distance = min(abs(guide - k) for k in coordinates)
    winners = [k for k in coordinates if abs(guide - k) == distance]
    _require(len(winners) == 1, "opposite-side nearest-defect tie")
    delta = [0, 0]
    delta[axis] = guide - winners[0]
    return tuple(delta)


def _translated(points, delta, shape):
    return {(r + delta[0], c + delta[1]) for r, c in points
            if _inside((r + delta[0], c + delta[1]), shape)}


def _decode(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    _require(min(h, w) >= 5, "scene too small for plates and guides")
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    counts = Counter(int(a[p]) for p in corners)
    _require(sorted(counts.values()) == [1, 3], "three field corners and one halo cue required")
    field = next(c for c, n in counts.items() if n == 3)
    highlight = next(c for c, n in counts.items() if n == 1)
    cue = next(p for p in corners if int(a[p]) == highlight)
    anchors, plates, extras = {}, [], set()
    groups = _components(tuple(map(int, p)) for p in np.argwhere(a != field))
    for group in groups:
        colors = {int(a[p]) for p in group}
        _require(len(colors) == 1, "input objects must be separated and monochrome")
        color = next(iter(colors))
        if cue in group:
            _require(group == {cue}, "halo cue must be isolated")
            continue
        _require(color != highlight, "halo cue color cannot be payload")
        if len(group) == 1:
            r, c = next(iter(group))
            if r in (0, h - 1) or c in (0, w - 1):
                _require(color not in anchors, "one guide marker per identity required")
                axis = 1 if r in (0, h - 1) else 0
                guide = c if axis == 1 else r
                anchors[color] = {"point": (r, c), "axis": axis, "guide": guide}
                continue
        if len(group) <= 2:
            _require(all(0 < r < h - 1 and 0 < c < w - 1 for r, c in group),
                     "specks cannot touch the canvas edge")
            extras.update(group)
            continue
        _require(len(group) >= 12, "unsupported small component")
        base = _spans(group, 0) | _spans(group, 1)
        _require(_convex(base), "plate must be convex on both row and column runs")
        _require(all(int(a[p]) in (field, color) for p in base), "plate includes another color")
        holes = base - group
        _require(1 <= len(holes) <= 6, "plate needs one to six readable defects")
        _require(len(base) <= 100 and len(holes) * 4 <= len(base), "excessive plate damage")
        plates.append({"color": color, "observed": group, "base": base, "holes": holes})
    _require(anchors and plates, "missing guide or plate")
    _require({p["color"] for p in plates} == set(anchors), "every guide needs matching plates")
    _require(all(int(a[p]) in anchors for p in extras), "specks need a plate identity")
    occupied = {cue} | {s["point"] for s in anchors.values()}
    source = set(occupied)
    for plate in plates:
        _require(not source & plate["base"], "recovered input footprints overlap")
        source.update(plate["base"])
        system = anchors[plate["color"]]
        delta = _shift_for(plate["holes"], system["axis"], system["guide"])
        plate["delta"] = delta
        plate["target"] = _translated(plate["base"], delta, a.shape)
        plate["target_holes"] = _translated(plate["holes"], delta, a.shape)
        _require(not occupied & plate["target"], "docked footprints overlap a plate or marker")
        occupied.update(plate["target"])
    return a, field, highlight, cue, anchors, plates, extras


def _render_decoded(decoded, variant="rule"):
    a, field, highlight, cue, anchors, plates, extras = decoded
    out = np.full(a.shape, field, dtype=np.int8)
    holes = set()
    for p in plates:
        delta = p["delta"]
        if variant == "ignore_A":
            delta = (0, 0)
        elif variant in ("center", "near_edge", "farthest"):
            s = anchors[p["color"]]
            axis, guide = s["axis"], s["guide"]
            lo, hi = min(q[axis] for q in p["base"]), max(q[axis] for q in p["base"])
            if variant == "center":
                registration = (lo + hi) // 2
            elif variant == "near_edge":
                registration = min((lo, hi), key=lambda k: (abs(k - guide), k))
            else:
                registration = max({q[axis] for q in p["holes"]}, key=lambda k: (abs(k - guide), k))
            delta = (guide - registration, 0) if axis == 0 else (0, guide - registration)
        target = _translated(p["base"], delta, a.shape)
        missing = _translated(p["holes"], delta, a.shape)
        for q in target:
            out[q] = p["color"]
        holes.update(missing)
    for q in holes:
        out[q] = field
    if variant != "ignore_B":
        halo = { (r + dr, c + dc) for r, c in holes for dr, dc in _D8
                 if _inside((r + dr, c + dc), a.shape)} - holes
        if variant == "own_plate_halo":
            occupied = set().union(*(p["target"] for p in plates))
            halo = set().union(*((( _neighbors(p["target_holes"]) - holes) &
                                  (p["target"] | (halo - occupied))) for p in plates)) & halo
        for q in halo:
            out[q] = highlight
    else:
        # A-only shortcut: retain the dirt and move raw pixels with each plate.
        for q in extras:
            out[q] = a[q]
    out[cue] = highlight
    for color, s in anchors.items():
        out[s["point"]] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover plates, dock their nearest defects, then draw global defect halos."""
    return _render_decoded(_decode(input_grid))


def _grow_shape(rng, kinds):
    kind = str(rng.choice(kinds))
    h, w = int(rng.integers(4, 10)), int(rng.integers(4, 10))
    b = np.zeros((h, w), dtype=bool)
    if kind == "rectangle":
        b[:] = True
    elif kind == "L":
        b[:int(rng.integers(2, h - 1)), :] = True
        b[:, :int(rng.integers(2, w - 1))] = True
    elif kind in ("tee", "cross"):
        thick = int(rng.integers(2, w - 1))
        left = 0 if kind == "tee" else int(rng.integers(1, w - thick))
        b[:, left:left + thick] = True
        top = int(rng.integers(1, h - 2))
        b[top:top + int(rng.integers(2, h - top + 1)), :] = True
    elif kind == "stair":
        widths = np.sort(rng.integers(2, w + 1, size=h))
        widths[-1] = w
        for r, width in enumerate(widths):
            b[r, :width] = True
    else:
        # Offset overlapping rectangles sharing an intact vertical spine.
        thick = int(rng.integers(2, w - 1))
        b[:, :thick] = True
        for _ in range(int(rng.integers(2, 5))):
            r0 = int(rng.integers(h - 1))
            r1 = int(rng.integers(r0 + 2, h + 1))
            b[r0:r1, :int(rng.integers(thick, w + 1))] = True
    b = np.rot90(b, int(rng.integers(4)))
    if rng.random() < .5:
        b = np.fliplr(b)
    points = set(tuple(map(int, p)) for p in np.argwhere(b))
    if not _convex(points) or len(points) < 16:
        return None
    return points


def _carrier(rng, kinds, mode, boundary=False):
    for _ in range(90):
        base = _grow_shape(rng, kinds)
        if base is None:
            continue
        ordered = sorted(base)
        if mode == "cluster":
            motifs = (((0, 0), (0, 1), (1, 0)), ((0, 0), (1, 1), (2, 2)))
            offsets = motifs[int(rng.integers(2))]
            origin = ordered[int(rng.integers(len(ordered)))]
            holes = {(origin[0] + dr, origin[1] + dc) for dr, dc in offsets}
        else:
            n = 2 if mode == "separated" else int(rng.integers(1, min(6, len(base)//6) + 1))
            holes = {ordered[int(i)] for i in rng.choice(len(ordered), size=n, replace=False)}
        if not holes <= base or len(holes) * 4 > len(base):
            continue
        if mode == "separated" and any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 1
                                       for p in holes for q in holes if p != q):
            continue
        observed = base - holes
        if len(observed) < 12 or len(_components(observed)) != 1:
            continue
        if (_spans(observed, 0) | _spans(observed, 1)) != base:
            continue
        if boundary and not any((r + dr, c + dc) not in base
                                for r, c in holes for dr, dc in _D8):
            continue
        # Crop only the local mask, not the halo that will later be global.
        lo = min(r for r, _ in base), min(c for _, c in base)
        return ({(r-lo[0], c-lo[1]) for r, c in base},
                {(r-lo[0], c-lo[1]) for r, c in holes})
    return None


def _neighbors(points, diagonal=True):
    offsets = _D8 if diagonal else _D4
    return set(points) | {(r+dr, c+dc) for r, c in points for dr, dc in offsets}


def _place_carrier(rng, shape, system, canvas_shape, source, target,
                   require_move=False, clipping=None, distinguish=False):
    if shape is None:
        return None
    base, holes = shape
    h, w = canvas_shape
    bh, bw = max(r for r, _ in base)+1, max(c for _, c in base)+1
    if bh > h or bw > w:
        return None
    rr, cc = np.meshgrid(np.arange(h-bh+1), np.arange(w-bw+1), indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    axis, guide = system["axis"], system["guide"]
    origins = rr if axis == 0 else cc
    hole_coords = np.array(sorted({p[axis] for p in holes}))
    distances = np.abs(origins[:, None] + hole_coords[None, :] - guide)
    closest = distances.min(axis=1)
    nearest = hole_coords[distances.argmin(axis=1)]
    displacement = guide-origins-nearest
    valid = (distances == closest[:, None]).sum(axis=1) == 1
    if require_move:
        valid &= displacement != 0
    if distinguish:
        low, high = min(p[axis] for p in base), max(p[axis] for p in base)
        valid &= nearest != (low+high)//2
        valid &= nearest != low
        valid &= nearest != high
        valid &= (np.abs(origins[:, None]+hole_coords[None, :]-guide).max(axis=1) > closest)
    source_blocked = np.zeros((h, w), dtype=bool)
    for p in _neighbors(source):
        if _inside(p, canvas_shape):
            source_blocked[p] = True
    target_blocked = np.zeros((h, w), dtype=bool)
    for p in target:
        target_blocked[p] = True
    clipped = np.zeros(len(rr), dtype=bool)
    for dr, dc in base:
        valid &= ~source_blocked[rr+dr, cc+dc]
        tr = rr+dr+(displacement if axis == 0 else 0)
        tc = cc+dc+(displacement if axis == 1 else 0)
        inside = (tr >= 0) & (tr < h) & (tc >= 0) & (tc < w)
        clipped |= ~inside
        valid &= ~inside | ~target_blocked[np.clip(tr, 0, h-1), np.clip(tc, 0, w-1)]
    if clipping is not None:
        valid &= clipped == clipping
    choices = np.flatnonzero(valid)
    if not len(choices):
        return None
    i = int(rng.choice(choices))
    origin = int(rr[i]), int(cc[i])
    delta = (int(displacement[i]), 0) if axis == 0 else (0, int(displacement[i]))
    absolute = {(r+origin[0], c+origin[1]) for r, c in base}
    missing = {(r+origin[0], c+origin[1]) for r, c in holes}
    docked = _translated(absolute, delta, canvas_shape)
    source.update(absolute)
    target.update(docked)
    return {"color": system["color"], "base": absolute, "holes": missing, "delta": delta}


def _paint_constructed(shape, field, highlight, cue, systems, records, extras):
    """Labels from construction geometry; no recovery or prediction calls."""
    inp = np.full(shape, field, np.int8)
    out = np.full(shape, field, np.int8)
    surviving_holes = set()
    for p in records:
        for q in p["base"]:
            inp[q] = p["color"]
            tr, tc = q[0]+p["delta"][0], q[1]+p["delta"][1]
            if 0 <= tr < shape[0] and 0 <= tc < shape[1]:
                out[tr, tc] = p["color"]
        for q in p["holes"]:
            inp[q] = field
            tr, tc = q[0]+p["delta"][0], q[1]+p["delta"][1]
            if 0 <= tr < shape[0] and 0 <= tc < shape[1]:
                surviving_holes.add((tr, tc))
    for q, color in extras.items():
        inp[q] = color
    for r, c in surviving_holes:
        out[max(0, r-1):min(shape[0], r+2), max(0, c-1):min(shape[1], c+2)] = highlight
    for q in surviving_holes:
        out[q] = field
    inp[cue] = out[cue] = highlight
    for s in systems:
        inp[s["point"]] = out[s["point"]] = s["color"]
    return {"input": inp, "output": out}


def _scene(rng, palette, plan):
    n = plan["systems"]
    if n == 1:
        h, w = int(rng.integers(10, 26)), int(rng.integers(10, 31))
    elif n == 2:
        h, w = int(rng.integers(16, 31)), int(rng.integers(16, 31))
    else:
        h, w = int(rng.integers(22, 31)), int(rng.integers(22, 31))
    shape = h, w
    field, highlight = palette[:2]
    colors = palette[2:2+n]
    corners = {(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)}
    cue = sorted(corners)[int(rng.integers(4))]
    source, target = set(corners), set(corners)
    systems = []
    for j, color in enumerate(colors):
        axis = plan["axis"] if not plan["mixed"] else (j % 2 if j < 2 else int(rng.integers(2)))
        end = int(rng.integers(2))
        candidates = list(range(1, shape[axis]-1))
        rng.shuffle(candidates)
        if plan["clip"] and j == 0:
            candidates.sort(key=lambda v: min(v, shape[axis]-1-v))
        for guide in candidates:
            point = ((guide, 0 if end == 0 else w-1) if axis == 0
                     else (0 if end == 0 else h-1, guide))
            if point not in _neighbors(source):
                systems.append({"color": color, "axis": axis, "guide": guide, "point": point})
                source.add(point)
                target.add(point)
                break
        else:
            return None
    records = []
    for j, system in enumerate(systems):
        count = int(rng.integers(1, 3)) if j == 0 else 1
        if n == 1 and rng.random() < .25:
            count = int(rng.integers(2, 4))
        for k in range(count):
            compulsory = k == 0
            mode = plan["mode"] if j == k == 0 else "random"
            desired_clip = plan["clip"] if j == k == 0 else None
            placed = None
            for _ in range(36 if compulsory else 9):
                body = _carrier(rng, plan["kinds"], mode, boundary=plan["boundary"] and j == k == 0)
                placed = _place_carrier(rng, body, system, shape, source, target,
                                        require_move=compulsory, clipping=desired_clip,
                                        distinguish=plan["distinguish"] and j == k == 0)
                if placed is not None:
                    break
            if placed is None and compulsory:
                return None
            if placed is not None:
                records.append(placed)
    # Specks are small, separate components, never registration candidates.
    blocked = _neighbors(source)
    options = [(r, c) for r in range(1, h-1) for c in range(1, w-1) if (r, c) not in blocked]
    rng.shuffle(options)
    extras = {}
    wanted = int(rng.integers(1 if plan["extras"] else 0, 6))
    for q in options:
        if q in blocked:
            continue
        group = {q}
        if wanted-len(extras) >= 2 and rng.random() < .3:
            adjacent = [(q[0]+dr, q[1]+dc) for dr, dc in _D4
                        if 0 < q[0]+dr < h-1 and 0 < q[1]+dc < w-1
                        and (q[0]+dr, q[1]+dc) not in blocked]
            if adjacent:
                group.add(adjacent[int(rng.integers(len(adjacent)))])
        color = int(rng.choice(colors))
        extras.update({p: color for p in group})
        blocked.update(_neighbors(group))
        if len(extras) >= wanted:
            break
    if wanted == 0:
        extras = {}
    if plan["extras"] and not extras:
        return None
    pair = _paint_constructed(shape, field, highlight, cue, systems, records, extras)
    if plan["extras"] and not any(int(pair["output"][q]) == field for q in extras):
        return None
    try:
        decoded = _decode(pair["input"])
    except AssertionError:
        return None
    # Gate on the visible recovery; do not silently accept a different scaffold.
    recovered = {(p["color"], frozenset(p["base"]), frozenset(p["holes"]), p["delta"])
                 for p in decoded[5]}
    intended = {(p["color"], frozenset(p["base"]), frozenset(p["holes"]), p["delta"])
                for p in records}
    if recovered != intended:
        return None
    f = _features(decoded)
    if plan["interact"] and (not f["cross_plate_halo"] or
                            np.array_equal(pair["output"], _render_decoded(decoded, "own_plate_halo"))):
        return None
    if plan["nonrect"] and not f["nonrectangular"]:
        return None
    if plan["distinguish"] and any(np.array_equal(pair["output"], _render_decoded(decoded, v))
                                   for v in ("ignore_A", "ignore_B", "center", "near_edge", "farthest")):
        return None
    return pair


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " outside legal range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    stable = bool(rng.random() < .35)
    palette = tuple(map(int, rng.permutation(10)))
    regime = int(rng.integers(3))
    axis = int(rng.integers(2))
    clip_demo = bool(rng.random() < .6)
    simple = ("rectangle",)
    rich = ("L", "tee", "cross", "stair", "arms")
    all_kinds = ("rectangle",) + rich
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(num_train + num_test):
        test = i >= num_train
        local_palette = palette if stable else tuple(map(int, rng.permutation(10)))
        plan = {"axis": (1-axis if test and regime == 2 else axis),
                "mixed": regime != 2 and i != 0,
                "systems": (3 if test and regime == 1 else 1 if i == 0 else 2),
                "kinds": rich if test and regime == 0 else simple if not test and regime == 0 else all_kinds,
                "mode": "separated" if i == 0 else "cluster" if i == 1 else "random",
                "clip": clip_demo if i == 1 else bool(test and clip_demo and rng.random() < .45),
                "boundary": i == 1,
                "extras": i == 0,
                "distinguish": i == 0,
                "interact": i == 1,
                "nonrect": test and regime == 0}
        for _ in range(100):
            pair = _scene(rng, local_palette, plan)
            if pair is not None and _signature(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("bounded construction could not expose the requested evidence")
        seen.add(_signature(pair["input"]))
        puzzle["test" if test else "train"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(decoded):
    a, field, highlight, cue, anchors, plates, extras = decoded
    halos = []
    targets = [p["target"] for p in plates]
    holes = set().union(*(p["target_holes"] for p in plates))
    for p in plates:
        halos.append((_neighbors(p["target_holes"]) - holes) &
                     {(r, c) for r in range(a.shape[0]) for c in range(a.shape[1])})
    return {"shape": a.shape, "field": field, "highlight": highlight,
            "systems": len(anchors), "plates": len(plates),
            "axes": tuple(sorted({s["axis"] for s in anchors.values()})),
            "holes": sum(len(p["holes"]) for p in plates), "extras": len(extras),
            "clipped": sum(len(p["target"]) < len(p["base"]) for p in plates),
            "aligned": sum(p["delta"] == (0, 0) for p in plates),
            "nonrectangular": sum(len(p["base"]) !=
                (max(r for r, _ in p["base"])-min(r for r, _ in p["base"])+1) *
                (max(c for _, c in p["base"])-min(c for _, c in p["base"])+1) for p in plates),
            "cross_plate_halo": any(bool(halos[i] & targets[j]) for i in range(len(plates))
                                    for j in range(len(plates)) if i != j),
            "halo_merge": any(bool(halos[i] & halos[j]) for i in range(len(plates))
                              for j in range(i+1, len(plates))),
            "edge_hole": any(r in (0, a.shape[0]-1) or c in (0, a.shape[1]-1) for r, c in holes),
            "source_edge": any(r in (0, a.shape[0]-1) or c in (0, a.shape[1]-1)
                               for p in plates for r, c in p["base"])}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid example counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas dimensions changed")
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            d = _decode(inp)
            _require(np.array_equal(_render_decoded(d), out), "wrong docking, cleanup, or halo")
            _require(not np.array_equal(inp, out), "unchanged example")
            # Exact family invariants, beyond comparison to solve().
            field, highlight, cue, anchors, plates = d[1:6]
            _require(int(out[cue]) == highlight, "corner cue moved")
            for color, s in anchors.items():
                _require(int(out[s["point"]]) == color, "guide marker changed")
            for p in plates:
                s = anchors[p["color"]]
                _require(any(q[s["axis"]] == s["guide"] for q in p["target_holes"]),
                         "no surviving registration defect on guide")
                _require(all(int(out[q]) == field for q in p["target_holes"]), "defect filled or highlighted")
                _require(p["delta"][1-s["axis"]] == 0, "along-guide displacement")
            if split == "train":
                training.append((d, out))
    for variant in ("ignore_A", "ignore_B", "center", "near_edge", "farthest", "own_plate_halo"):
        _require(any(not np.array_equal(_render_decoded(d, variant), out) for d, out in training),
                 "training does not distinguish " + variant)
    _require(any(int(out[q]) == d[1] for d, out in training for q in d[6]),
             "training must visibly erase a speck outside docked bodies and halos")
    _require(any(len(p["holes"]) >= 3 for d, _ in training for p in d[5]),
             "training must show a multiple-hole neighborhood union")


def _semantic_fixtures():
    # Hand-specified rectangles and expected docking coordinates; neither
    # generation nor the construction renderer supplies these expected arrays.
    inp = np.zeros((14, 20), np.int8)
    inp[8:13, 4:10] = 1
    inp[2:7, 14:19] = 2
    inp[9, 6] = inp[11, 9] = inp[4, 16] = inp[6, 17] = 0
    inp[1, 4:6] = 1  # a dimer of removable dirt
    inp[0, 0], inp[5, 0], inp[0, 12] = 9, 1, 2
    expected = np.zeros_like(inp)
    expected[4:9, 4:10] = 1
    expected[2:7, 10:15] = 2
    holes = {(5, 6), (7, 9), (4, 12), (6, 13)}
    for r, c in holes:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expected[r+dr, c+dc] = 9
    for q in holes:
        expected[q] = 0
    expected[0, 0], expected[5, 0], expected[0, 12] = 9, 1, 2
    _require(np.array_equal(solve(inp), expected), "shared halo / nearest-defect fixture")
    _require(int(expected[6, 10]) == 9 and int(inp[1, 4]) == 1,
             "fixture must distinguish cross-plate painting and dirt erasure")
    altered = inp.copy()
    altered[9, 6] = 1
    old_delta = next(p["delta"] for p in _decode(inp)[5] if p["color"] == 1)
    new_delta = next(p["delta"] for p in _decode(altered)[5] if p["color"] == 1)
    _require(old_delta == (-4, 0) and new_delta == (-6, 0),
             "removing the nearest defect must change the entire plate's movement")
    _require(int(expected[8, 4]) == 1 and int(solve(altered)[8, 4]) == 0,
             "defect change must affect material far beyond that defect's halo")
    # One translated defect leaves the canvas: it contributes no ghost halo.
    inp = np.zeros((10, 14), np.int8)
    inp[1:10, 4:9] = 1
    inp[1, 6] = inp[9, 6] = 0
    inp[2, 0], inp[0, 13] = 1, 9
    expected = np.zeros_like(inp)
    expected[2:10, 4:9] = 1
    expected[1:4, 5:8] = 9
    expected[2, 6] = 0
    expected[2, 0], expected[0, 13] = 1, 9
    _require(np.array_equal(solve(inp), expected), "clipping and surviving-hole fixture")
    _require(int(expected[9, 6]) == 1, "clipped holes must not emit halos")
    # A fixed guide is not swallowed when a docked boundary hole touches it.
    inp = np.zeros((16, 10), np.int8)
    inp[9:14, 1:6] = 1
    inp[10, 1] = 0
    inp[5, 0], inp[0, 0] = 1, 9
    actual = solve(inp)
    _require(int(actual[5, 0]) == 1 and int(actual[4, 0]) == 9 and int(actual[5, 1]) == 0,
             "fixed-marker priority and halo crossing the plate edge")


def self_test():
    """Bounded geometry/semantic/variation checks across exactly 42 base seeds."""
    _semantic_fixtures()
    features, fingerprints, shapes, local_masks = [], set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprints.add(tuple(_signature(pair[k]) for s in ("train", "test")
                               for pair in p[s] for k in ("input", "output")))
        for pair in p["train"] + p["test"]:
            d = _decode(pair["input"])
            features.append(_features(d))
            shapes.add(d[0].shape)
            for body in d[5]:
                points = body["base"]
                lo = min(r for r, _ in points), min(c for _, c in points)
                local_masks.add(tuple(sorted((r-lo[0], c-lo[1]) for r, c in points)))
        if seed < 3:
            q = generate_puzzle(seed)
            _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                         for k in x), "seed replay changed")
    _require(len(fingerprints) == 42, "puzzles repeated")
    _require(len(shapes) >= 60 and len(local_masks) >= 100, "insufficient geometric diversity")
    _require({f["field"] for f in features} == set(range(10)), "field palette coverage")
    _require({f["highlight"] for f in features} == set(range(10)), "halo palette coverage")
    _require({1, 2, 3} <= {f["systems"] for f in features}, "guide count variation")
    _require({(0,), (1,), (0, 1)} <= {f["axes"] for f in features}, "orientation variation")
    _require(len({f["holes"] for f in features}) >= 8, "defect counts are too uniform")
    _require(len({f["plates"] for f in features}) >= 4, "plate counts are too uniform")
    for key in ("clipped", "aligned", "nonrectangular", "edge_hole", "source_edge",
                "cross_plate_halo", "halo_merge"):
        _require(any(f[key] for f in features), "missing structural interaction: " + key)
    # Family is equivariant to every square-grid symmetry and palette permutation.
    p = generate_puzzle(7, 2, 1)
    rng = np.random.default_rng(713)
    for turns in range(4):
        for reflection in (False, True):
            colors = rng.permutation(10)
            q = {s: [{k: colors[np.rot90(np.fliplr(v) if reflection else v, turns)]
                       for k, v in pair.items()} for pair in p[s]] for s in p}
            validate_generated_puzzle(q)
    # Mutations must be rejected; labels are not used to decode the input.
    for kind in ("hole", "anchor", "halo"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = bad["test"][0]
        d = _decode(pair["input"])
        point = (next(iter(d[5][0]["target_holes"])) if kind == "hole" else
                 next(iter(d[4].values()))["point"] if kind == "anchor" else
                 next(tuple(map(int, z)) for z in np.argwhere(pair["output"] == d[2]) if tuple(z) != d[3]))
        pair["output"][point] = (int(pair["output"][point])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("bad " + kind + " output accepted")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                 (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    try:
        _shift_for({(3, 2), (3, 6)}, 1, 4)
    except AssertionError:
        pass
    else:
        raise AssertionError("opposite-side tie accepted")
    _require(_shift_for({(3, 2), (4, 2)}, 1, 5) == (0, 3), "same-coordinate ties must agree")
    return {"seeds": 42, "pairs": len(features), "shapes": len(shapes),
            "local_plate_shapes": len(local_masks),
            "cross_plate_halos": sum(f["cross_plate_halo"] for f in features),
            "merged_plate_halos": sum(f["halo_merge"] for f in features)}
