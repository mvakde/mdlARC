"""Sealed-air shadow sorter: topology supplies registered sortable profiles.

Only standard-library modules and NumPy are used. Generation has its own
geometric renderer; solve reconstructs all roles and templates from input cells.
"""
from __future__ import annotations

import itertools
import json

import numpy as np

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "illegal grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(bool(((a >= 0) & (a <= 9)).all()), "illegal color")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _regions(mask, neighbors=N4):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        cells, stack = [p], [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    cells.append(q)
                    stack.append(q)
        result.append(cells)
    return result


def _holes(mask, neighbors=N4):
    """Outside padding means canvas edges never count as enclosing walls."""
    p = np.pad(mask, 1)
    exterior = np.zeros_like(p)
    exterior[0, 0] = True
    stack = [(0, 0)]
    h, w = p.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not p[rr, cc] and not exterior[rr, cc]:
                exterior[rr, cc] = True
                stack.append((rr, cc))
    return (~p & ~exterior)[1:-1, 1:-1]


def _unambiguous(mask):
    a, b = mask[:-1, :-1], mask[:-1, 1:]
    c, d = mask[1:, :-1], mask[1:, 1:]
    return not np.any((a == d) & (b == c) & (a != b))


def _dominant(values):
    counts = np.bincount(values.ravel(), minlength=10)
    v = int(np.argmax(counts))
    _need(np.count_nonzero(counts == counts[v]) == 1, "ambiguous dominant color")
    return v


def _separated_runs(mask):
    for a in (mask, mask.T):
        for row in a:
            x = np.flatnonzero(row)
            if len(x) > 1 and np.any(np.diff(x) > 1):
                return True
    return False


def _analyze(value):
    """Cell-only parser. Canonical orientation is a horizontal rail."""
    grid = _grid(value)
    bg = _dominant(grid)
    rows = np.flatnonzero(np.all(grid != bg, axis=1))
    cols = np.flatnonzero(np.all(grid != bg, axis=0))
    _need(len(rows) + len(cols) == 1, "rail must be unique")
    vertical = bool(len(cols))
    a = grid.T if vertical else grid
    rail = int(cols[0] if vertical else rows[0])
    carrier = _dominant(a[rail])
    _need(carrier != bg and np.count_nonzero(a[rail] == carrier) >= 3, "unreadable rail color")
    _need(np.count_nonzero(a == carrier) == np.count_nonzero(a[rail] == carrier),
          "input rail color occurs off the rail")
    foreground = (a != bg) & (a != carrier)
    rr, _ = np.nonzero(foreground)
    offsets = rr[rr != rail] - rail
    _need(len(offsets) > 0 and (np.all(offsets < 0) or np.all(offsets > 0)),
          "chambers must occupy one source side")
    side = -1 if offsets[0] < 0 else 1
    records = []
    explained = np.zeros_like(foreground)
    for color in sorted(set(map(int, a[foreground]))):
        for cells in _regions(a == color):
            coords = np.array(cells)
            r, c = coords.T
            hits = np.sort(c[r == rail])
            _need(len(hits) > 0 and np.all(np.diff(hits) == 1), "component needs one contiguous slot")
            start, width = int(hits[0]), len(hits)
            _need(1 <= width <= 3, "slot width outside family")
            r0, c0 = int(r.min()), int(c.min())
            local = np.zeros((int(r.max()) - r0 + 1, int(c.max()) - c0 + 1), bool)
            local[r - r0, c - c0] = True
            _need(_unambiguous(local) and len(_regions(local, N8)) == 1, "ambiguous component")
            cavity = _holes(local)
            _need(np.array_equal(cavity, _holes(local, N8)), "diagonal cavity leak")
            _need(len(_regions(cavity)) == len(_regions(cavity, N8)), "diagonal cavity contact")
            body = np.zeros(a.shape, bool)
            body[r, c] = True
            holes = np.zeros(a.shape, bool)
            hr, hc = np.nonzero(cavity)
            holes[hr + r0, hc + c0] = True
            _need(np.all(a[holes] == bg), "cavity includes another role or nested object")
            profile = tuple(sorted(set(abs(int(x) - rail) for x in hr + r0)))
            _need(all(d > 0 and 0 <= rail - side * d < a.shape[0] for d in profile),
                  "reflected template would be clipped")
            hole_regions = _regions(cavity)
            summed_heights = sum(max(x for x, _ in region) - min(x for x, _ in region) + 1
                                 for region in hole_regions)
            records.append({"start": start, "width": width, "color": color,
                            "body": body, "holes": holes, "profile": profile,
                            "local": local, "area": int(cavity.sum()),
                            "hole_count": len(hole_regions), "sum_heights": summed_heights,
                            "extent": max(abs(int(x) - rail) for x in r),
                            "box": (r0, c0, local.shape[0], local.shape[1])})
            explained |= body
    records.sort(key=lambda o: o["start"])
    _need(2 <= len(records) <= 5, "family has 2 to 5 chambers")
    _need(np.array_equal(explained, foreground), "unexplained foreground")
    _need(len(set(o["color"] for o in records)) >= 2, "need distinguishable accent roles")
    for left, right in zip(records, records[1:]):
        _need(left["box"][1] + left["box"][3] < right["box"][1],
              "parallel component boxes need a background gap")
    all_holes = np.zeros(a.shape, bool)
    for o in records:
        _need(not np.any(all_holes & o["holes"]), "shared cavity ownership")
        all_holes |= o["holes"]
    # Treat the carrier as passable, since it is not a chamber wall.
    _need(np.array_equal(_holes(foreground), all_holes), "collective enclosure")
    return {"grid": a, "background": bg, "carrier": carrier, "rail": rail,
            "side": side, "vertical": vertical, "records": records, "holes": all_holes}


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _analyze(input_grid)
    out = a["grid"].copy()
    out[a["holes"]] = a["carrier"]
    eligible = [o for o in a["records"] if o["profile"]]
    templates = sorted((o["profile"] for o in eligible), key=len)  # Python sort is stable.
    for destination, profile in zip(eligible, templates):
        for depth in profile:
            row = a["rail"] - a["side"] * depth
            x = destination["start"]
            out[row, x:x + destination["width"]] = destination["color"]
    return out.T.copy() if a["vertical"] else out


def _object_valid(mask, target):
    return (_unambiguous(mask) and len(_regions(mask)) == 1
            and np.array_equal(_holes(mask), target)
            and np.array_equal(_holes(mask, N8), target))


def _sculpt(rng, mask, target):
    """Optional exterior corner/bay cuts, accepting only unchanged topology."""
    h, w = mask.shape
    for _ in range(int(rng.integers(0, 5))):
        candidate = mask.copy()
        side = int(rng.integers(2))
        depth = int(rng.integers(1, min(3, h - 1) + 1))
        thick = int(rng.integers(1, min(2, w - 1) + 1))
        if side == 0:
            candidate[-depth:, :thick] = False
        else:
            candidate[-depth:, -thick:] = False
        if candidate[-1].any() and _object_valid(candidate, target):
            mask = candidate
    return mask


def _profile_object(rng, profile, width, slot_width=None, fixed_hole_width=None,
                    dual=False, extent=None, sculpt=True, forced_irregular=False):
    """Depth-indexed body: row 0 is the rail; positive rows are source distances."""
    profile = tuple(sorted(profile))
    lo = min(profile) - int(rng.integers(1, min(3, min(profile) - 1) + 1))
    hi = (max(profile) + int(rng.integers(1, min(3, 14 - max(profile)) + 1))
          if extent is None else extent)
    sw = int(rng.integers(1, min(3, width) + 1)) if slot_width is None else slot_width
    neck = int(rng.integers(0, width - sw + 1))
    mask = np.zeros((hi + 1, width), bool)
    mask[lo:] = True
    mask[:lo, neck:neck + sw] = True
    holes = np.zeros_like(mask)
    if forced_irregular:
        _need(width >= 4 and len(profile) >= 3, "stepped pocket needs space")
        holes[list(profile), 1] = True
        holes[profile[len(profile) // 2], 2] = True
    elif dual:
        _need(width >= 5, "dual pocket needs five columns")
        holes[list(profile), 1] = True
        holes[list(profile), width - 2] = True
    elif fixed_hole_width is not None:
        left = int(rng.integers(1, width - fixed_hole_width))
        holes[list(profile), left:left + fixed_hole_width] = True
    else:
        groups = []
        for d in profile:
            if not groups or d != groups[-1][-1] + 1:
                groups.append([])
            groups[-1].append(d)
        for group in groups:
            backbone = int(rng.integers(1, width - 1))
            left = int(rng.integers(1, backbone + 1))
            right = int(rng.integers(backbone + 1, width))
            irregular = len(group) > 1 and width > 3 and rng.random() < .55
            for d in group:
                l = int(rng.integers(1, backbone + 1)) if irregular else left
                r = int(rng.integers(backbone + 1, width)) if irregular else right
                holes[d, l:r] = True
    mask &= ~holes
    if sculpt:
        mask = _sculpt(rng, mask, holes)
    _need(_object_valid(mask, holes), "invalid constructed chamber")
    return {"mask": mask, "holes": holes, "profile": profile,
            "slot_width": sw, "neck": neck, "width": width, "extent": hi}


def _open_copy(obj):
    """Open every pocket to outside with vertical doorways, preserving the neck."""
    mask = obj["mask"].copy()
    rows, cols = np.nonzero(obj["holes"])
    for c in set(map(int, cols)):
        mask[int(rows.min()):, c] = False
    empty = np.zeros_like(mask)
    _need(_object_valid(mask, empty), "open doorway must not split the body")
    return dict(obj, mask=mask, holes=empty, profile=())


def _plain_object(rng, width):
    hi = int(rng.integers(3, 13))
    lo = int(rng.integers(1, min(4, hi - 1) + 1))
    sw = int(rng.integers(1, min(3, width) + 1))
    neck = int(rng.integers(0, width - sw + 1))
    mask = np.zeros((hi + 1, width), bool)
    mask[:lo, neck:neck + sw] = True
    if rng.random() < .5:
        mask[lo:] = True
    else:
        spine = int(rng.integers(1, width - 1))
        mask[lo:min(hi + 1, lo + 2)] = True
        mask[lo:, spine] = True
        for d in range(lo + 2, hi + 1):
            if rng.random() < .55:
                if rng.random() < .5:
                    mask[d, :spine + 1] = True
                else:
                    mask[d, spine:] = True
    empty = np.zeros_like(mask)
    mask = _sculpt(rng, mask, empty)
    _need(_object_valid(mask, empty), "invalid solid body")
    return {"mask": mask, "holes": empty, "profile": (), "slot_width": sw,
            "neck": neck, "width": width, "extent": hi}


def _run(rng, length, cap=9):
    first = int(rng.integers(2, cap - length + 2))
    return tuple(range(first, first + length))


def _anchor_objects(rng, kind, count):
    capw = 5 if count == 5 else 6
    if kind == 1:
        profiles = {length: _run(rng, length) for length in (1, 2, 3)}
        hi = max(max(p) for p in profiles.values()) + 1
        long = _profile_object(rng, profiles[3], int(rng.integers(3, capw + 1)),
                               1, 1, extent=hi, sculpt=False)
        mid = _profile_object(rng, profiles[2], int(rng.integers(4, capw + 1)),
                              2, 2, extent=hi, sculpt=False)
        short = _profile_object(rng, profiles[1], int(rng.integers(3, capw + 1)),
                                extent=hi, sculpt=False)
        objects = [long, short, mid] if rng.random() < .5 else [mid, long, short]
        opened = _open_copy(long)
    else:
        gap_first = bool(rng.integers(2))
        if gap_first:
            first = int(rng.integers(4, 7))
            gap = (first, first + int(rng.integers(3, 10 - first)))
            consecutive = (2, 3)
        else:
            first = int(rng.integers(2, 4))
            gap = (first, first + int(rng.integers(3, min(5, 9 - first) + 1)))
            first_run = int(rng.integers(6, 9))
            consecutive = (first_run, first_run + 1)
        gapped = _profile_object(rng, gap, int(rng.integers(3, capw + 1)), 1, 1,
                                 sculpt=False)
        dual = _profile_object(rng, consecutive, int(rng.integers(5, capw + 1)),
                               2, dual=True, sculpt=False)
        long = _profile_object(rng, _run(rng, 3), int(rng.integers(4, capw + 1)),
                               forced_irregular=True)
        objects = [long, gapped, dual] if gap_first else [dual, long, gapped]
        opened = _open_copy(gapped)
    objects.insert(int(rng.integers(0, len(objects) + 1)), opened)
    if count == 5:
        objects.insert(int(rng.integers(0, len(objects) + 1)),
                       _plain_object(rng, int(rng.integers(3, capw + 1))))
    return objects


def _random_profile(rng, cap, length=None):
    length = int(rng.integers(1, min(6, cap - 1) + 1)) if length is None else length
    if rng.random() < .6:
        return _run(rng, length, cap)
    return tuple(sorted(map(int, rng.choice(np.arange(2, cap + 1), length, replace=False))))


def _ordinary_objects(rng, test):
    count = int(rng.integers(2, 6))
    capw = 8 if count <= 3 else 6 if count == 4 else 5
    cap = 12 if test else 9
    closed_probability = float(rng.uniform(.15, .95))
    kinds = [rng.random() < closed_probability for _ in range(count)]
    if test:
        for index in rng.choice(count, 2, replace=False):
            kinds[int(index)] = True
    profiles = []
    objects = []
    deepest_index = next((i for i, closed in enumerate(kinds) if closed), None)
    for i, closed in enumerate(kinds):
        width = int(rng.integers(3, capw + 1))
        if closed:
            length = int(rng.choice([len(p) for p in profiles])) if profiles and rng.random() < .35 else None
            if test and len(profiles) == 1:
                length = int(rng.choice([n for n in range(1, 7) if n != len(profiles[0])]))
            p = _random_profile(rng, cap, length)
            if test and i == deepest_index and max(p) < 10:
                shift = int(rng.integers(10, 13)) - max(p)
                p = tuple(d + shift for d in p)
            profiles.append(p)
            objects.append(_profile_object(rng, p, width, dual=width >= 5 and rng.random() < .25))
        elif rng.random() < .6:
            proto = _profile_object(rng, _random_profile(rng, cap), width, sculpt=False)
            objects.append(_open_copy(proto))
        else:
            objects.append(_plain_object(rng, width))
    if test:
        eligible = [i for i, obj in enumerate(objects) if obj["profile"]]
        moved = [objects[i] for i in eligible]
        rng.shuffle(moved)
        lengths = [len(obj["profile"]) for obj in moved]
        if lengths == sorted(lengths):
            moved = moved[1:] + moved[:1]
        for i, obj in zip(eligible, moved):
            objects[i] = obj
    return objects


def _render_scene(rng, objects, bg, carrier, vertical, side, anchor=False):
    count = len(objects)
    pool = [c for c in range(10) if c not in (bg, carrier)]
    palette_size = int(rng.integers(3 if anchor else 2, min(count, 5) + 1))
    palette = list(map(int, rng.choice(pool, palette_size, replace=False)))
    colors = palette + list(map(int, rng.choice(palette, count - palette_size)))
    rng.shuffle(colors)
    if anchor:
        eligible = [i for i, o in enumerate(objects) if o["profile"]]
        distinct = list(map(int, rng.permutation(palette)))[:3]
        for i, color in zip(eligible, distinct):
            colors[i] = color
    total_slots = sum(o["slot_width"] for o in objects)
    biggest_accent_run = max(sum(o["slot_width"] for o, c in zip(objects, colors) if c == color)
                             for color in set(colors))
    min_width = max(sum(o["width"] for o in objects) + count - 1,
                    total_slots + max(3, biggest_accent_run + 1))
    _need(min_width <= 30, "parallel packing overflow")
    width = int(rng.integers(min_width, 31))
    spare = width - sum(o["width"] for o in objects) - (count - 1)
    gaps = list(map(int, rng.multinomial(spare, [1 / (count + 1)] * (count + 1))))
    for i in range(1, count):
        gaps[i] += 1
    extent = max(o["extent"] for o in objects)
    depths = [d for o in objects for d in o["profile"]]
    echo_extent = max(depths, default=1)
    min_height = extent + echo_extent + 1
    _need(min_height <= 30, "perpendicular packing overflow")
    height = int(rng.integers(min_height, min(30, min_height + 7) + 1))
    rail = extent + int(rng.integers(0, height - extent - echo_extent))
    inp = np.full((height, width), bg, np.int8)
    inp[rail] = carrier
    out = inp.copy()
    destinations = []
    x = gaps[0]
    # Ground truth uses the explicitly constructed air masks, not solve/parser.
    for i, (obj, color) in enumerate(zip(objects, colors)):
        r, c = np.nonzero(obj["mask"])
        inp[rail - r, x + c] = color
        out[rail - r, x + c] = color
        hr, hc = np.nonzero(obj["holes"])
        out[rail - hr, x + hc] = carrier
        if obj["profile"]:
            destinations.append((x + obj["neck"], obj["slot_width"], color, obj["profile"]))
        x += obj["width"] + gaps[i + 1]
    ordered = sorted((d[3] for d in destinations), key=len)
    for (start, sw, color, _), profile in zip(destinations, ordered):
        out[rail + np.array(profile), start:start + sw] = color
    if side > 0:
        inp, out = np.flipud(inp).copy(), np.flipud(out).copy()
    if vertical:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def _resolve_count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _unique_coloring(pair, seen):
    if _key(pair["input"]) not in seen:
        return pair
    a = _analyze(pair["input"])
    old = sorted(set(o["color"] for o in a["records"]))
    pool = [c for c in range(10) if c not in (a["background"], a["carrier"])]
    # At least two accents give >=56 colorings, versus at most 13 exclusions.
    for new in itertools.permutations(pool, len(old)):
        mapping = np.arange(10, dtype=np.int8)
        mapping[old] = new
        candidate = {k: mapping[v] for k, v in pair.items()}
        if _key(candidate["input"]) not in seen:
            return candidate
    raise AssertionError("finite uniqueness fallback exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _resolve_count("num_train", num_train, 2, 10)
    nq = _resolve_count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    roles = list(map(int, rng.choice(10, 2, replace=False)))
    fixed_roles = rng.random() < .3
    mode = int(rng.choice(3, p=[.3, .2, .5]))  # H, V, or both demonstrated.
    if mode < 2:
        train_axes, test_axes = [bool(mode)] * nt, [bool(mode)] * nq
    else:
        train_axes = [False, True] + list(map(bool, rng.integers(0, 2, nt - 2)))
        test_axes = list(map(bool, rng.integers(0, 2, nq)))
    first_side = -1 if rng.integers(2) == 0 else 1
    train_sides = [first_side, -first_side] + [(-1 if rng.integers(2) == 0 else 1) for _ in range(nt - 2)]
    test_sides = [(-1 if rng.integers(2) == 0 else 1) for _ in range(nq)]
    first_count = int(rng.integers(4, 6))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, axes, sides in (("train", train_axes, train_sides), ("test", test_axes, test_sides)):
        for i, (axis, side) in enumerate(zip(axes, sides)):
            anchor = split == "train" and i < 2
            objects = (_anchor_objects(rng, i + 1, first_count if i == 0 else 9 - first_count)
                       if anchor else _ordinary_objects(rng, split == "test"))
            bg, carrier = roles if fixed_roles else list(map(int, rng.choice(10, 2, replace=False)))
            pair = _render_scene(rng, objects, bg, carrier, axis, side, anchor)
            pair = _unique_coloring(pair, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _profiles_for(a, mode="correct"):
    closed = [o for o in a["records"] if o["profile"]]
    if mode == "unsorted":
        return closed, [o["profile"] for o in closed]
    if mode == "all_slots":
        objs = a["records"]
        ps = [o["profile"] or tuple(range(1, o["extent"] + 1)) for o in objs]
        return objs, sorted(ps, key=len)
    if mode == "reverse":
        return closed, [o["profile"] for o in reversed(closed)]
    key = {"correct": lambda o: len(o["profile"]),
           "area": lambda o: o["area"],
           "hull": lambda o: max(o["profile"]) - min(o["profile"]) + 1,
           "sum": lambda o: o["sum_heights"],
           "body": lambda o: o["extent"],
           "depth_tie": lambda o: (len(o["profile"]), min(o["profile"]))}
    if mode == "reverse_tie":
        indices = sorted(range(len(closed)), key=lambda i: (len(closed[i]["profile"]), -i))
        ps = [closed[i]["profile"] for i in indices]
    else:
        ps = [o["profile"] for o in sorted(closed, key=key.get(mode, key["correct"]))]
    if mode == "dense":
        ps = [tuple(range(min(p), min(p) + len(p))) for p in ps]
    elif mode == "packed":
        ps = [tuple(range(1, len(p) + 1)) for p in ps]
    return closed, ps


def _features(a):
    eligible, correct = _profiles_for(a)
    original = [o["profile"] for o in eligible]
    features = {mode: _profiles_for(a, mode)[1] != correct
                for mode in ("unsorted", "reverse", "area", "hull", "sum", "body", "reverse_tie", "depth_tie")}
    features.update({"axis": a["vertical"], "side": a["side"], "count": len(a["records"]),
                     "closed": len(eligible), "shape": a["grid"].shape,
                     "widths": tuple(o["width"] for o in a["records"]),
                     "profiles": tuple(original),
                     "open_notch": any(not o["profile"] and _separated_runs(o["local"]) for o in a["records"]),
                     "holes": tuple(o["hole_count"] for o in eligible),
                     "max_depth": max((max(p) for p in original), default=0),
                     "gapped": any(max(p) - min(p) + 1 > len(p) for p in original),
                     "palette": (a["background"], a["carrier"]),
                     "layout": (a["vertical"], a["side"], a["grid"].shape, a["rail"],
                                tuple((o["start"], o["width"], o["extent"], o["profile"],
                                       _key(o["local"])) for o in a["records"]))})
    return features


def _check_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _need(inp.shape == out.shape, "canvas dimensions changed")
    a = _analyze(inp)
    b = out.T if a["vertical"] else out
    _need(np.array_equal(b[a["rail"]], a["grid"][a["rail"]]), "rail or slot identity changed")
    _need(np.all(b[a["holes"]] == a["carrier"]), "incomplete or wrong cavity filling")
    echo_mask = np.zeros(b.shape, bool)
    observed = []
    eligible = [o for o in a["records"] if o["profile"]]
    echo_rows = [r for r in range(b.shape[0]) if (r - a["rail"]) * a["side"] < 0]
    for o in eligible:
        distances = []
        x, width = o["start"], o["width"]
        for r in echo_rows:
            segment = b[r, x:x + width]
            if np.any(segment != a["background"]):
                _need(np.all(segment == o["color"]), "echo does not retain destination color/width")
                distances.append(abs(r - a["rail"]))
                echo_mask[r, x:x + width] = True
        observed.append(tuple(sorted(distances)))
    # Relational template conservation and stable rank selection, no solve/render call.
    remaining = [(i, o["profile"]) for i, o in enumerate(eligible)]
    previous = 0
    for p in observed:
        _need(bool(p) and len(p) >= previous, "missing or nonascending echo")
        chosen = min(remaining, key=lambda t: (len(t[1]), t[0]))
        _need(p == chosen[1], "wrong full template, registration, or stable tie order")
        remaining.remove(chosen)
        previous = len(p)
    _need(not remaining, "template lost")
    protected = ~a["holes"] & ~echo_mask
    _need(np.array_equal(b[protected], a["grid"][protected]),
          "body, open bay, background, or ineligible echo changed")
    return a, _features(a)


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid example counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            key = _key(_grid(pair["input"]))
            _need(key not in seen, "duplicate input")
            seen.add(key)
            _, f = _check_pair(pair)
            features[split].append(f)
    train = features["train"]
    for feature in ("unsorted", "area", "hull", "sum", "body", "reverse_tie", "depth_tie", "open_notch", "gapped"):
        _need(any(f[feature] for f in train), "training lacks distinction: " + feature)
    _need(any(f["unsorted"] and f["reverse"] for f in train), "training permits reversal")
    _need(len({f["count"] for f in train}) >= 2, "training needs different object counts")
    axes = {f["axis"] for f in train if f["closed"] >= 3}
    sides = {f["side"] for f in train if f["closed"] >= 3}
    train_depth = max(f["max_depth"] for f in train)
    _need(train_depth <= 9, "training outside registration regime")
    for f in features["test"]:
        _need(f["axis"] in axes and f["side"] in sides, "undemonstrated reflection/orientation")
        _need(f["closed"] >= 2 and f["unsorted"], "test must exercise template reassignment")
        _need(10 <= f["max_depth"] <= 12 and f["max_depth"] > train_depth,
              "test lacks deeper registration shift")


def _mutated_output(inp, mode):
    a = _analyze(inp)
    out = a["grid"].copy()
    out[a["holes"]] = a["carrier"]
    destinations, profiles = _profiles_for(a, mode)
    for o, profile in zip(destinations, profiles):
        for depth in profile:
            r = a["rail"] - a["side"] * depth
            if 0 <= r < out.shape[0]:
                out[r, o["start"]:o["start"] + o["width"]] = o["color"]
    return out.T.copy() if a["vertical"] else out


def self_test():
    """Bounded authored checks, including geometry diversity over 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signature = lambda p: tuple((_key(q["input"]), _key(q["output"]))
                                for s in ("train", "test") for q in p[s])
    _need(signature(puzzles[0]) == signature(generate_puzzle(0)), "nondeterministic puzzle")
    _need(len({signature(p) for p in puzzles}) == 42, "repeated puzzles across seeds")
    rows = []
    for p in puzzles:
        validate_generated_puzzle(p)
        for split in p:
            for pair in p[split]:
                _, f = _check_pair(pair)
                f["split"] = split
                rows.append(f)
                original = pair["input"].copy()
                _need(np.array_equal(solve(original), pair["output"]), "input-only solve disagrees")
                _need(np.array_equal(original, pair["input"]), "solve mutated its input")
    shapes = {f["shape"] for f in rows}
    layouts = {f["layout"] for f in rows}
    _need(len(shapes) >= 50 and len(layouts) >= .9 * len(rows), "insufficient geometric diversity")
    _need({f["axis"] for f in rows} == {False, True} and {f["side"] for f in rows} == {-1, 1}, "missing orientation")
    _need({f["count"] for f in rows} == {2, 3, 4, 5}, "missing object counts")
    _need({w for f in rows for w in f["widths"]} == {1, 2, 3}, "missing slot widths")
    _need({len(p) for f in rows for p in f["profiles"]} >= {1, 2, 3, 4, 5, 6}, "missing shadow lengths")
    _need(any(f["closed"] == 0 for f in rows) and any(f["closed"] == 1 for f in rows), "missing simple class mixtures")
    _need(any(f["closed"] == 5 for f in rows), "missing all-sealed large scene")
    _need(len({f["palette"] for f in rows}) >= 50, "palette diversity")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(42000 + 10 * nt + nq, nt, nq)
            _need(len(p["train"]) == nt and len(p["test"]) == nq, "count mismatch")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    modes = ("unsorted", "all_slots", "area", "hull", "sum", "body", "reverse",
             "reverse_tie", "depth_tie", "dense", "packed")
    for mode in modes:
        witness = next(pair for p in puzzles for pair in p["train"]
                       if not np.array_equal(_mutated_output(pair["input"], mode), pair["output"]))
        try:
            _check_pair({"input": witness["input"], "output": _mutated_output(witness["input"], mode)})
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted " + mode)
    # Missing fill and changed original bodies must also fail independently.
    witness = next(q for q in puzzles[0]["train"] if _analyze(q["input"])["holes"].any())
    a = _analyze(witness["input"])
    body_zone = a["records"][0]["body"].copy()
    body_zone[a["rail"]] = False
    for zone in (a["holes"], body_zone):
        out = witness["output"].copy()
        canonical = out.T if a["vertical"] else out
        r, c = np.argwhere(zone)[0]
        canonical[r, c] = (int(canonical[r, c]) + 1) % 10
        try:
            _check_pair({"input": witness["input"], "output": out})
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted source corruption")
    return {"seeds": 42, "pairs": len(rows), "canvas_shapes": len(shapes),
            "structural_layouts": len(layouts), "palettes": len({f["palette"] for f in rows}),
            "mutation_rules_rejected": list(modes), "independent_semantic_approval": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
