"""Folded stencils: anchor-relative substitution with exclusive coverage.

Only NumPy and the standard library are used. No parent files are imported.
The forward construction and input-only inverse renderer use different loops.
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import permutations
import numpy as np


def _pick(rng, choices):
    return choices[int(rng.integers(len(choices)))]


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int16, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], [start]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    component.append(q)
        result.append(component)
    return result


def _box(points):
    a = np.asarray(points, dtype=int)
    r0, c0 = a.min(0)
    r1, c1 = a.max(0) + 1
    return int(r0), int(r1), int(c0), int(c1)


def _gap(a, b):
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _decode(input_grid):
    """Recover the template, its calibration square, and the entire map."""
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    assert len(values) == 3, "three local color roles required"
    assert sorted(counts)[-1] > sorted(counts)[-2], "background must be uniquely dominant"
    bg = int(values[np.argmax(counts)])
    components = _components(map(tuple, np.argwhere(a != bg)))
    mixed = [p for p in components if len({int(a[q]) for q in p}) > 1]
    assert len(mixed) == 1, "one connected two-color glyph required"
    glyph = mixed[0]
    sites = [p for p in components if p is not glyph]
    assert len(sites) >= 3
    boxes, site_colors = [], []
    for p in sites:
        box = _box(p)
        r0, r1, c0, c1 = box
        assert r1-r0 == c1-c0 and len(p) == (r1-r0)**2
        color = int(a[p[0]])
        assert all(a[q] == color for q in p)
        boxes.append(box)
        site_colors.append(color)
    sizes = {b[1]-b[0] for b in boxes}
    assert len(sizes) == 1
    s = sizes.pop()
    assert 1 <= s <= 3
    frequencies = Counter(site_colors)
    assert len(frequencies) == 2 and sorted(frequencies.values())[0] == 1
    ink = next(c for c, n in frequencies.items() if n == 1)
    guide = next(c for c in frequencies if c != ink)
    assert {int(a[q]) for q in glyph} == {ink, guide}
    calibration = [q for q in glyph if a[q] == guide]
    qr0, qr1, qc0, qc1 = _box(calibration)
    t = qr1-qr0
    assert t == qc1-qc0 and len(calibration) == t*t and 1 <= t <= 3
    gr0, gr1, gc0, gc1 = _box(glyph)
    h, w = (gr1-gr0)//t, (gc1-gc0)//t
    assert 2 <= h <= 5 and 2 <= w <= 5
    assert gr1-gr0 == h*t and gc1-gc0 == w*t
    assert (qr0-gr0) % t == 0 and (qc0-gc0) % t == 0
    patch = a[gr0:gr1, gc0:gc1]
    blocks = patch.reshape(h, t, w, t)
    assert np.array_equal(blocks.min((1, 3)), blocks.max((1, 3))), "exact calibration lattice"
    small = blocks[:, 0, :, 0]
    template = np.zeros((h, w), dtype=np.int8)
    template[small == ink] = 1
    template[small == guide] = 2
    assert int(np.count_nonzero(template == 2)) == 1
    assert int(np.count_nonzero(template == 1)) >= 3
    assert len(_components(map(tuple, np.argwhere(template != 0)))) == 1
    pr, pc = min(b[0] for b in boxes), min(b[2] for b in boxes)
    points = []
    anchor = None
    for b, color in zip(boxes, site_colors):
        assert (b[0]-pr) % (2*s) == 0 and (b[2]-pc) % (2*s) == 0
        point = ((b[0]-pr)//(2*s), (b[2]-pc)//(2*s))
        points.append(point)
        if color == ink:
            anchor = point
    points = tuple(sorted(points))
    mh, mw = max(r for r, c in points)+1, max(c for r, c in points)+1
    assert 1 <= mh <= 6 and 1 <= mw <= 6
    map_box = (pr, pr+(2*mh-1)*s, pc, pc+(2*mw-1)*s)
    assert _gap((gr0, gr1, gc0, gc1), map_box), "a blank band separates map and glyph boxes"
    return {"bg": bg, "ink": ink, "guide": guide, "t": t, "s": s,
            "template": template, "guide_cell": ((qr0-gr0)//t, (qc0-gc0)//t),
            "origin": (qr0, qc0), "points": points, "anchor": anchor,
            "map_shape": (mh, mw), "map_box": map_box,
            "glyph_box": (gr0, gr1, gc0, gc1)}


def _predict(a, d, *, folding="relative", scaling=True, pitch="box",
             swap=True, coverage="exclusive", solid=False, anchor_mode="color"):
    """Inverse renderer. Pull each canvas cell through each virtual stencil.

    Nondefault options are explicit, finite shortcut hypotheses for validation.
    They are never consulted by solve to select a rule.
    """
    template = d["template"].copy()
    if solid:
        template[template == 0] = 1
    h, w = template.shape
    gr, gc = d["guide_cell"]
    s = d["s"] if scaling else d["t"]
    ar, ac = d["anchor"]
    if anchor_mode == "first":
        ar, ac = d["points"][0]
    elif anchor_mode == "center":
        mh, mw = d["map_shape"]
        ar, ac = (mh-1)//2, (mw-1)//2
    rows = sorted({r for r, c in d["points"]})
    cols = sorted({c for r, c in d["points"]})
    if pitch == "swapped":
        ph, pw = w, h
    elif pitch == "map":
        ph = pw = 2
    elif pitch == "padded":
        ph, pw = h+1, w+1
    else:
        ph, pw = h, w
    yy, xx = np.indices(a.shape)
    count = np.zeros(a.shape, dtype=np.int16)
    color = np.full(a.shape, d["bg"], dtype=np.int16)
    all_inside = True
    for r, c in d["points"]:
        dr, dc = r-ar, c-ac
        if pitch == "compressed":
            dr, dc = rows.index(r)-rows.index(ar), cols.index(c)-cols.index(ac)
        fr, fc = bool(dr % 2), bool(dc % 2)
        if folding == "none":
            fr = fc = False
        elif folding == "row":
            fc = False
        elif folding == "col":
            fr = False
        elif folding == "absolute":
            fr, fc = bool(r % 2), bool(c % 2)
        orow, ocol = d["origin"][0]+dr*ph*s, d["origin"][1]+dc*pw*s
        rr, cc = (yy-orow)//s, (xx-ocol)//s
        # Reflect about the guide square, rather than the glyph's box center.
        rr = -rr if fr else rr
        cc = -cc if fc else cc
        if folding == "box":
            rr, cc = (yy-orow)//s, (xx-ocol)//s
            rr = h-1-(rr+gr)-gr if fr else rr
            cc = w-1-(cc+gc)-gc if fc else cc
        tr, tc = rr+gr, cc+gc
        inside = (tr >= 0) & (tr < h) & (tc >= 0) & (tc < w)
        values = template[np.clip(tr, 0, h-1), np.clip(tc, 0, w-1)]
        occupied = inside & (values != 0)
        is_anchor = (r, c) == d["anchor"]
        local_ink, local_guide = d["ink"], d["guide"]
        if swap and is_anchor:
            local_ink, local_guide = local_guide, local_ink
        mapped = np.where(values == 2, local_guide, local_ink)
        count[occupied] += 1
        color[occupied] = mapped[occupied]
        # Complete-support bounds are checked even if overlap would erase ink.
        for br, bc in np.argwhere(template != 0):
            ur, uc = int(br)-gr, int(bc)-gc
            if folding == "box":
                ur = h-1-int(br)-gr if fr else ur
                uc = w-1-int(bc)-gc if fc else uc
            else:
                ur = -ur if fr else ur
                uc = -uc if fc else uc
            y, x = orow+ur*s, ocol+uc*s
            all_inside &= 0 <= y and y+s <= a.shape[0] and 0 <= x and x+s <= a.shape[1]
    keep = count == 1 if coverage == "exclusive" else count % 2 == 1 if coverage == "xor" else count > 0
    out = np.full(a.shape, d["bg"], dtype=np.int16)
    out[keep] = color[keep]
    return out, {"count": count, "inside": all_inside}


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    d = _decode(a)
    out, stats = _predict(a, d)
    assert stats["inside"], "all virtual stamps must fit; this family has no clipping"
    return out.astype(np.int8)


_ALTERNATIVES = {
    "no_reflections": {"folding": "none"},
    "horizontal_only": {"folding": "col"},
    "vertical_only": {"folding": "row"},
    "absolute_parity": {"folding": "absolute"},
    "box_center_reflection": {"folding": "box"},
    "ignore_marker_size": {"scaling": False},
    "use_map_pitch": {"pitch": "map"},
    "swap_box_axes": {"pitch": "swapped"},
    "padded_box": {"pitch": "padded"},
    "no_color_exchange": {"swap": False},
    "paint_union": {"coverage": "union"},
    "odd_coverage": {"coverage": "xor"},
    "solid_glyph": {"solid": True},
    "first_site_anchor": {"anchor_mode": "first"},
    "geometric_center_anchor": {"anchor_mode": "center"},
}


def _features(pair):
    a, out = _grid(pair["input"]), _grid(pair["output"])
    assert a.shape == out.shape
    d = _decode(a)
    expected, stats = _predict(a, d)
    assert stats["inside"] and np.array_equal(expected, out), "incorrect folded exclusive overlay"
    assert not np.array_equal(a, out)
    assert set(map(int, np.unique(out))) == {d["bg"], d["ink"], d["guide"]}
    assert int(stats["count"].sum()) == len(d["points"])*np.count_nonzero(d["template"])*d["s"]**2
    assert np.count_nonzero(out != d["bg"]) == np.count_nonzero(stats["count"] == 1)
    assert not np.array_equal(_predict(a, d, pitch="map", swap=False)[0], out), "A ablation is invisible"
    assert not np.array_equal(_predict(a, d, folding="none", scaling=False)[0], out), "B ablation is invisible"
    d["collision"] = int(np.count_nonzero(stats["count"] >= 2))
    d["triple"] = int(np.count_nonzero(stats["count"] == 3))
    d["max_coverage"] = int(stats["count"].max())
    assert d["max_coverage"] <= 4
    d["gaps"] = (d["map_shape"][0]-len({r for r, c in d["points"]}),
                 d["map_shape"][1]-len({c for r, c in d["points"]}))
    d["fails"] = {name for name, options in _ALTERNATIVES.items()
                  if not np.array_equal(_predict(a, d, **options)[0], out)}
    if any(d["gaps"]):
        assert not np.array_equal(_predict(a, d, pitch="compressed")[0], out)
    return d


def _teaching_ok(train, require_gaps=False):
    failed = set().union(*(d["fails"] for d in train))
    return (failed == set(_ALTERNATIVES)
            and all(len({d["anchor"][axis] for d in train}) >= 2 for axis in range(2))
            and all(len({d["template"].shape[axis] for d in train}) >= 2 for axis in range(2))
            and any(d["triple"] for d in train)
            and any(d["s"] != d["t"] for d in train)
            and (not require_gaps or all(any(d["gaps"][axis] for d in train) for axis in range(2))))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, metas = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(_grid(pair["input"]))
            assert key not in seen
            seen.add(key)
            metas[split].append(_features(pair))
    assert _teaching_ok(metas["train"]), "demonstrations leave a listed shortcut possible"
    # Gapped coordinates in a test require observed blank bands in training.
    for axis in range(2):
        if any(d["gaps"][axis] for d in metas["test"]):
            assert any(d["gaps"][axis] for d in metas["train"])


def _template(rng, h, w, teaching):
    guide = _pick(rng, [(r, c) for r in range(h) for c in range(w)])
    if teaching:
        guide = _pick(rng, [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)])
    mask = np.zeros((h, w), dtype=bool)
    current = guide
    mask[current] = True
    corners = [(0, 0), (h-1, w-1)]
    if rng.random() < .5:
        corners = [(0, w-1), (h-1, 0)]
    for target in corners:
        while current != target:
            options = []
            r, c = current
            if r != target[0]:
                options.append((r+int(np.sign(target[0]-r)), c))
            if c != target[1]:
                options.append((r, c+int(np.sign(target[1]-c))))
            current = _pick(rng, options)
            mask[current] = True
    density = float(rng.uniform(.45, .95)) if not teaching else float(rng.uniform(.75, .98))
    goal = max(4, int(round(h*w*density)), int(mask.sum()))
    while mask.sum() < goal:
        frontier = set()
        for r, c in np.argwhere(mask):
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= q[0] < h and 0 <= q[1] < w and not mask[q]:
                    frontier.add(q)
        mask[_pick(rng, sorted(frontier))] = True
    if teaching and mask.all():
        removable = [q for q in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)] if q != guide]
        mask[_pick(rng, removable)] = False
    template = mask.astype(np.int8)
    template[guide] = 2
    return template, guide


def _layout(rng, mh, mw, gapped=False, teaching=False):
    rows, cols = list(range(mh)), list(range(mw))
    if gapped:
        if mh >= 4:
            rows.remove(int(rng.integers(1, mh-1)))
        if mw >= 4:
            cols.remove(int(rng.integers(1, mw-1)))
    all_points = [(r, c) for r in rows for c in cols]
    anchor = _pick(rng, all_points)
    points = {anchor}
    # Every permitted band is occupied; omitted bands are deliberate and taught.
    for r in rows:
        points.add((r, _pick(rng, cols)))
    for c in cols:
        points.add((_pick(rng, rows), c))
    # Direct pitch witnesses in both available axes; blanks are not compressed.
    for values, axis in ((rows, 0), (cols, 1)):
        adjacent = [v for v in values if v+1 in values]
        if adjacent:
            v = _pick(rng, adjacent)
            other = _pick(rng, cols if axis == 0 else rows)
            points.update(((v, other), (v+1, other)) if axis == 0 else ((other, v), (other, v+1)))
    remaining = [p for p in all_points if p not in points]
    rng.shuffle(remaining)
    fraction = float(rng.uniform(.7, 1)) if teaching else float(rng.uniform(.2, 1))
    goal = max(3, len(points), int(round(len(all_points)*fraction)))
    points.update(remaining[:goal-len(points)])
    return tuple(sorted(points)), anchor


def _forward(template, guide_cell, points, anchor, s, colors):
    """Construction: push scaled source blocks onto an unbounded plane."""
    bg, ink, guide_color = colors
    h, w = template.shape
    gr, gc = guide_cell
    layers, last = Counter(), {}
    for r, c in points:
        dr, dc = r-anchor[0], c-anchor[1]
        for br, bc in np.argwhere(template != 0):
            ur, uc = int(br)-gr, int(bc)-gc
            ur = -ur if dr % 2 else ur
            uc = -uc if dc % 2 else uc
            y, x = (dr*h+ur)*s, (dc*w+uc)*s
            color = guide_color if template[br, bc] == 2 else ink
            if (r, c) == anchor:
                color = ink if color == guide_color else guide_color
            for i in range(s):
                for j in range(s):
                    q = y+i, x+j
                    layers[q] += 1
                    last[q] = color
    return layers, last


def _place(rng, template, guide_cell, points, anchor, t, s, colors):
    h, w = template.shape
    gr, gc = guide_cell
    layers, last = _forward(template, guide_cell, points, anchor, s, colors)
    glyph_box = (-gr*t, (h-gr)*t, -gc*t, (w-gc)*t)
    br0, br1, bc0, bc1 = _box(list(layers))
    br0, br1 = min(br0, glyph_box[0]), max(br1, glyph_box[1])
    bc0, bc1 = min(bc0, glyph_box[2]), max(bc1, glyph_box[3])
    mh, mw = max(r for r, c in points)+1, max(c for r, c in points)+1
    ph, pw = (2*mh-1)*s, (2*mw-1)*s
    if max(br1-br0, bc1-bc0, ph, pw) > 30:
        return None
    placements = []
    for pr in range(br0-ph-1, br1+2):
        for pc in range(bc0-pw-1, bc1+2):
            if not _gap(glyph_box, (pr, pr+ph, pc, pc+pw)):
                continue
            top, bottom = min(br0, pr), max(br1, pr+ph)
            left, right = min(bc0, pc), max(bc1, pc+pw)
            if bottom-top <= 30 and right-left <= 30:
                placements.append((pr, pc, top, bottom, left, right))
    if not placements:
        return None
    if rng.random() < .45:
        areas = [(p[3]-p[2])*(p[5]-p[4]) for p in placements]
        cut = np.quantile(areas, .25)
        placements = [p for p, area in zip(placements, areas) if area <= cut]
    pr, pc, top, bottom, left, right = _pick(rng, placements)
    gh, gw = bottom-top, right-left
    # Keep the background a recoverable color role, even on dense inputs.
    body = np.count_nonzero(template == 1)*t*t + s*s
    marker = t*t + (len(points)-1)*s*s
    area_needed = body+marker+max(body, marker)+1
    dimension_choices = [(a, b) for a in range(gh, 31) for b in range(gw, 31) if a*b >= area_needed]
    if not dimension_choices:
        return None
    if rng.random() < .12:
        gh, gw = 30, 30
    else:
        minimum = min(a*b for a, b in dimension_choices)
        slack = int(rng.integers(0, 101))
        small = [p for p in dimension_choices if p[0]*p[1] <= minimum+slack]
        gh, gw = _pick(rng, small)
    dy, dx = int(rng.integers(gh-(bottom-top)+1))-top, int(rng.integers(gw-(right-left)+1))-left
    bg, ink, guide_color = colors
    inp, out = np.full((gh, gw), bg, dtype=np.int8), np.full((gh, gw), bg, dtype=np.int8)
    for br, bc in np.argwhere(template != 0):
        y, x = dy+(int(br)-gr)*t, dx+(int(bc)-gc)*t
        inp[y:y+t, x:x+t] = guide_color if template[br, bc] == 2 else ink
    for r, c in points:
        y, x = dy+pr+2*r*s, dx+pc+2*c*s
        inp[y:y+s, x:x+s] = ink if (r, c) == anchor else guide_color
    for (r, c), multiplicity in layers.items():
        if multiplicity == 1:
            out[dy+r, dx+c] = last[r, c]
    return {"input": inp, "output": out}


def _sample_pair(rng, palette, *, teaching=False, gap=False, long_axis=None, long_min=4, previous=None):
    dims = [(h, w) for h in range(2, 6) for w in range(2, 6)
            if not teaching or (h != w and h <= 4 and w <= 4)]
    if previous is not None:
        dims = [(h, w) for h, w in dims if h != previous["template"].shape[0]
                and w != previous["template"].shape[1]]
    h, w = _pick(rng, dims)
    template, guide_cell = _template(rng, h, w, teaching)
    if teaching:
        mh = mw = 4 if gap else 3
    elif long_axis is None:
        mh, mw = _pick(rng, [(a, b) for a in range(1, 4) for b in range(1, 4) if a*b >= 3])
        if gap:
            mh = mw = 4
    else:
        long = 5 if gap else int(rng.integers(long_min, 6))
        short = int(rng.integers(2 if gap else 1, 4))
        mh, mw = (long, short) if long_axis == 0 else (short, long)
        if gap:
            # Both blank-band axes were exposed in the second demonstration.
            short = 4
            mh, mw = (long, short) if long_axis == 0 else (short, long)
    points, anchor = _layout(rng, mh, mw, gap, teaching)
    if previous is not None and (anchor[0] == previous["anchor"][0] or anchor[1] == previous["anchor"][1]):
        return None
    t, s = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    if teaching and previous is None and t == s:
        return None
    colors = palette if palette is not None else tuple(map(int, rng.choice(10, 3, replace=False)))
    pair = _place(rng, template, guide_cell, points, anchor, t, s, colors)
    if pair is None:
        return None
    try:
        features = _features(pair)
    except AssertionError:
        return None
    return pair, features


def _fallback(rng, palette, second=False, gap=False, long_axis=None):
    # Finite capacity safeguard; these are generated geometry, not parent pairs.
    colors = palette if palette is not None else tuple(map(int, rng.choice(10, 3, replace=False)))
    if not second:
        template = np.array([[2, 1, 0], [1, 1, 1]], dtype=np.int8)
        guide, anchor, t, s = (0, 0), (0, 0), 2, 1
    else:
        template = np.array([[0, 1], [1, 1], [1, 2]], dtype=np.int8)
        guide, anchor, t, s = (2, 1), (1, 2), 1, 2 if long_axis is None and not gap else 1
    if long_axis is not None:
        mh, mw = (5, 3) if long_axis == 0 else (3, 5)
        if gap:
            mh, mw = (5, 4) if long_axis == 0 else (4, 5)
    else:
        mh = mw = 4 if gap else 3
    points = tuple((r, c) for r in range(mh) for c in range(mw)
                   if ((r != 2 and c != 2) if gap else (r, c) != (1, 1)))
    if anchor not in points:
        anchor = (mh-1, mw-1)
    pair = _place(rng, template, guide, points, anchor, t, s, colors)
    assert pair is not None
    return pair, _features(pair)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    long_axis, extensions = int(rng.integers(2)), bool(rng.random() < .35)
    palette = tuple(map(int, rng.choice(10, 3, replace=False))) if rng.random() < .2 else None
    # Plan the minimal two-example teaching set before any optional examples.
    first = second = None
    for _ in range(192):
        first = _sample_pair(rng, palette, teaching=True)
        if first is None or not first[1]["triple"]:
            continue
        for _ in range(64):
            second = _sample_pair(rng, palette, teaching=True, gap=extensions, previous=first[1])
            if second is not None and _teaching_ok([first[1], second[1]], extensions):
                break
        else:
            continue
        break
    else:
        first = _fallback(rng, palette)
        second = _fallback(rng, palette, second=True, gap=extensions)
        assert _teaching_ok([first[1], second[1]], extensions)
    puzzle = {"train": [first[0], second[0]], "test": []}
    seen = {_key(first[0]["input"]), _key(second[0]["input"])}
    for split, count in (("train", num_train-2), ("test", num_test)):
        for _ in range(count):
            candidate = None
            for _attempt in range(192):
                candidate = _sample_pair(rng, palette, gap=extensions and rng.random() < .65,
                                         long_axis=long_axis if split == "test" else None,
                                         long_min=5 if extensions else 4)
                if candidate is not None and _key(candidate[0]["input"]) not in seen:
                    break
            else:
                candidate = _fallback(rng, palette, second=True, gap=extensions,
                                      long_axis=long_axis if split == "test" else None)
            pair = candidate[0]
            if _key(pair["input"]) in seen:
                # Color bijections preserve all local semantics and are finite.
                old = sorted(map(int, np.unique(pair["input"])))
                for roles in permutations(range(10), 3):
                    remap = np.arange(10, dtype=np.int8)
                    remap[old] = roles
                    fresh = {k: remap[v] for k, v in pair.items()}
                    if _key(fresh["input"]) not in seen:
                        pair = fresh
                        break
                else:
                    raise AssertionError("finite uniqueness safeguard exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(p):
    return tuple((_key(np.asarray(pair["input"])), _key(np.asarray(pair["output"])))
                 for split in ("train", "test") for pair in p[split])


def self_test():
    """Bounded checks on 42 seeds, real geometry, ablations and bad outputs."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    features = [_features(pair) for p in puzzles for split in ("train", "test") for pair in p[split]]
    shapes = {np.asarray(pair["input"]).shape for p in puzzles for split in p for pair in p[split]}
    assert len(shapes) >= 40
    assert len({(d["template"].shape, d["template"].tobytes()) for d in features}) >= 50
    assert len({(d["map_shape"], d["points"], d["anchor"]) for d in features}) >= 70
    assert len({d["guide_cell"] for d in features}) >= 8
    assert len({d["anchor"] for d in features}) >= 10
    assert {d["s"] for d in features} == {1, 2, 3}
    assert {d["t"] for d in features} == {1, 2, 3}
    assert any(d["s"] < d["t"] for d in features) and any(d["s"] > d["t"] for d in features)
    assert any(d["map_shape"][0] == 1 or d["map_shape"][1] == 1 for d in features)
    assert any(all(d["gaps"]) for d in features)
    assert max(len(d["points"]) for d in features) >= 12
    assert {3, 4} <= {d["max_coverage"] for d in features}
    for seed, nt, nq in ((1001, 2, 1), (1002, 10, 4), (1003, 6, 3)):
        p = generate_puzzle(seed, nt, nq)
        assert len(p["train"]) == nt and len(p["test"]) == nq
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(seed, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4., 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for gap in (False, True):
        rng = np.random.default_rng(700)
        a, b = _fallback(rng, None), _fallback(rng, None, second=True, gap=gap)
        assert _teaching_ok([a[1], b[1]], gap)
        for axis in (0, 1):
            _fallback(rng, None, second=True, gap=gap, long_axis=axis)
    # Extent six is valid semantics even though the ordinary test schedule ends
    # at five. Exercise both orientations through returned pixels.
    rng = np.random.default_rng(701)
    for mh, mw in ((6, 2), (2, 6)):
        template = np.array([[2, 1, 0], [1, 1, 1]], dtype=np.int8)
        points = tuple((r, c) for r in range(mh) for c in range(mw))
        pair = _place(rng, template, (0, 0), points, (1, 1), 2, 1, (0, 1, 2))
        assert pair is not None
        _features(pair)
        assert np.array_equal(solve(pair["input"]), pair["output"])
    broken = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # Input-only sensitivity: moving the exceptional site changes the answer.
    pair = puzzles[0]["train"][0]
    inp = np.asarray(pair["input"]).copy()
    d = _decode(inp)
    pr, _, pc, _ = d["map_box"]
    valid_change = False
    for other in d["points"]:
        if other == d["anchor"]:
            continue
        swapped = inp.copy()
        for p, color in ((d["anchor"], d["guide"]), (other, d["ink"])):
            y, x = pr+2*p[0]*d["s"], pc+2*p[1]*d["s"]
            swapped[y:y+d["s"], x:x+d["s"]] = color
        changed, stats = _predict(swapped, _decode(swapped))
        if stats["inside"] and not np.array_equal(changed, pair["output"]):
            assert np.array_equal(solve(swapped), changed)
            valid_change = True
            break
    assert valid_change


if __name__ == "__main__":
    self_test()
    print(json.dumps({"self_test": "passed", "seeds": 42}, separators=(",", ":")))
