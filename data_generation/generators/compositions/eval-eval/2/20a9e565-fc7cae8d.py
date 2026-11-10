"""Age-indexed glyph growth in independently oriented local frames.

Only NumPy and the standard library are used. Labels are drawn from explicit
latent strokes; solve decodes the input, fits arithmetic measurements, and uses
cell predicates. No parent code, files, random state, or examples are consulted
by solve. See task_spec.md for the full, input-visible convention.
"""

from functools import lru_cache

import numpy as np


SIDES = ("T", "R", "B", "L")
NORMALS = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
MATRICES = (
    ((1, 0), (0, 1)), ((0, -1), (1, 0)),
    ((-1, 0), (0, -1)), ((0, 1), (-1, 0)),
    ((1, 0), (0, -1)), ((-1, 0), (0, 1)),
    ((0, 1), (1, 0)), ((0, -1), (-1, 0)),
)
STYLES = ("elbow", "fork", "comb", "hollow")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grids must have two dimensions in 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             "grid colors must be integers in 0..9")
    return a.astype(np.int8, copy=True)


def _counts(num_train, num_test):
    for name, count, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        _require(not isinstance(count, (bool, np.bool_)) and
                 isinstance(count, (int, np.integer)) and low <= count <= high,
                 name + " is an illegal count")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _adjacent(i):
    return SIDES[i], SIDES[(i + 1) % 4]


def _side(a, side):
    return {"T": a[0, :], "R": a[:, -1], "B": a[-1, :], "L": a[:, 0]}[side]


def _bar(a, side, r, c, s):
    if side == "T":
        return a[r - 1, c:c + s]
    if side == "R":
        return a[r:r + s, c + s]
    if side == "B":
        return a[r + s, c:c + s]
    return a[r:r + s, c - 1]


def _operation(guides, outer):
    """The two correspondences specify a signed orthogonal coordinate matrix."""
    m = np.zeros((2, 2), dtype=int)
    for side, color in guides.items():
        target = next(s for s, v in outer.items() if v == color)
        m += np.outer(NORMALS[target], NORMALS[side])
    matches = [i for i, candidate in enumerate(MATRICES)
               if np.array_equal(m, candidate)]
    _require(len(matches) == 1, "guides do not specify one dihedral symmetry")
    return matches[0]


def _array_transform(a, op):
    # Solver-side array operations. Generation instead moves individual cells.
    if op < 4:
        return np.rot90(a, op).copy()
    if op == 4:
        return a[:, ::-1].copy()
    if op == 5:
        return a[::-1, :].copy()
    if op == 6:
        return a.T.copy()
    return a.T[::-1, ::-1].copy()


def _move_cells(a, matrix):
    p = np.indices(a.shape).reshape(2, -1)
    q = np.asarray(matrix, dtype=int) @ p
    q -= q.min(axis=1, keepdims=True)
    result = np.empty(tuple(q.max(axis=1) + 1), dtype=a.dtype)
    result[q[0], q[1]] = a.ravel()
    return result


def _components(a, color):
    remaining = set(map(tuple, np.argwhere(a == color)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], []
        while stack:
            r, c = stack.pop()
            component.append((int(r), int(c)))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(component)
    return result


def _decode(a):
    """Recover outer controls, all square frames, ages, and normalized patches."""
    H, W = a.shape
    _require(min(H, W) >= 8, "canvas too small for separated keys and frames")
    uniform = {}
    for side in SIDES:
        line = _side(a, side)[1:-1]
        _require(np.all(line == line[0]), "outer sides must be uninterrupted")
        uniform[side] = int(line[0])
    candidates = []
    for i in range(4):
        control_sides = _adjacent(i)
        other = [s for s in SIDES if s not in control_sides]
        bg = uniform[other[0]]
        if (uniform[other[1]] == bg and uniform[control_sides[0]] != bg and
                uniform[control_sides[1]] != bg and
                uniform[control_sides[0]] != uniform[control_sides[1]]):
            candidates.append((bg, {s: uniform[s] for s in control_sides}))
    _require(len(candidates) == 1, "need two distinct adjacent outer controls")
    bg, outer = candidates[0]
    key_colors = set(outer.values())
    lines = []
    covered = np.zeros(a.shape, dtype=bool)
    for side in SIDES:
        if side == "T":
            covered[0, :] = True
        elif side == "B":
            covered[-1, :] = True
        elif side == "L":
            covered[:, 0] = True
        else:
            covered[:, -1] = True
    # Corner overwriting is immaterial, but corners must belong to the sides.
    for r, c in ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)):
        allowed = {uniform["T" if r == 0 else "B"],
                   uniform["L" if c == 0 else "R"]}
        _require(int(a[r, c]) in allowed, "invalid outer corner")
    for color in sorted(key_colors):
        for comp in _components(a, color):
            if any(r in (0, H - 1) or c in (0, W - 1) for r, c in comp):
                _require(all(r in (0, H - 1) or c in (0, W - 1) for r, c in comp),
                         "outer controls must not have inward branches")
                continue
            rows = sorted({r for r, _ in comp})
            cols = sorted({c for _, c in comp})
            if len(rows) == 1 and len(cols) >= 4 and cols == list(range(cols[0], cols[-1] + 1)):
                lines.append(("H", rows[0], cols[0], cols[-1], color))
            elif len(cols) == 1 and len(rows) >= 4 and rows == list(range(rows[0], rows[-1] + 1)):
                lines.append(("V", cols[0], rows[0], rows[-1], color))
            else:
                raise ValueError("internal key components must be complete straight guides")
    panels, used, ages, tag_color = [], set(), set(), None
    for hi, h in enumerate(lines):
        if h[0] != "H":
            continue
        matches = []
        for vi, v in enumerate(lines):
            if v[0] != "V" or h[4] == v[4] or h[3] - h[2] != v[3] - v[2]:
                continue
            hs = "T" if h[1] == v[2] - 1 else "B" if h[1] == v[3] + 1 else None
            vs = "L" if v[1] == h[2] - 1 else "R" if v[1] == h[3] + 1 else None
            if hs is not None and vs is not None:
                matches.append((vi, v[2], h[2], h[3] - h[2] + 1,
                                {hs: h[4], vs: v[4]}))
        _require(len(matches) == 1, "a guide must have one perpendicular mate")
        vi, r, c, s, guides = matches[0]
        _require(hi not in used and vi not in used, "guide reused")
        used.update((hi, vi))
        _require(3 <= r and 3 <= c and r + s <= H - 3 and c + s <= W - 3,
                 "frames need space between their annotation rings and the border")
        _require(not np.any(covered[r - 1:r + s + 1, c - 1:c + s + 1]),
                 "frame rectangles or annotations overlap")
        free = [side for side in SIDES if side not in guides]
        tags = []
        for side in free:
            line = _bar(a, side, r, c, s)
            active = np.flatnonzero(line != bg)
            if len(active):
                colors = np.unique(line[active])
                _require(len(colors) == 1 and int(colors[0]) not in key_colors and
                         np.array_equal(active, np.arange(active[0], active[-1] + 1)),
                         "age must be one contiguous bar in a separate color")
                tags.append((side, int(colors[0]), len(active)))
        _require(len(tags) == 1, "each frame needs one age bar on an unused side")
        tag_side, color, age = tags[0]
        if tag_color is None:
            tag_color = color
        _require(color == tag_color and 1 <= age <= 6 and age not in ages,
                 "age colors must match and ages must be unique integers in 1..6")
        ages.add(age)
        for rr, cc in ((r - 1, c - 1), (r - 1, c + s),
                       (r + s, c - 1), (r + s, c + s)):
            _require(a[rr, cc] == bg, "annotation ring corners must be background")
        raw = a[r:r + s, c:c + s].copy()
        _require(not set(map(int, np.unique(raw))) & (key_colors | {tag_color}),
                 "content, age, and key roles must be distinct")
        op = _operation(guides, outer)
        normalized = _array_transform(raw, op)
        panels.append(dict(r=r, c=c, s=s, guides=guides, age=age,
                           tag_side=tag_side, op=op, raw=raw, patch=normalized,
                           observed=bool(np.any(raw != bg))))
        covered[r - 1:r + s + 1, c - 1:c + s + 1] = True
    _require(len(used) == len(lines) and 4 <= len(panels) <= 6,
             "need four to six frames, with all guides accounted for")
    _require(np.all(a[~covered] == bg), "unassigned foreground outside frames")
    panels.sort(key=lambda p: p["age"])
    return dict(bg=bg, outer=outer, tag=tag_color, panels=panels, shape=a.shape)


def _measure(patch, bg):
    points = np.argwhere(patch != bg)
    _require(len(points) > 0, "empty observation")
    r, c = map(int, points.min(axis=0))
    r1, c1 = map(int, points.max(axis=0))
    glyph = patch[r:r1 + 1, c:c1 + 1]
    values = np.unique(glyph[glyph != bg])
    _require(len(values) == 1, "each observed glyph has one cyclic content color")
    h, w = glyph.shape
    s = patch.shape[0]
    return dict(h=h, w=w, u=2 * r + h - s, v=2 * c + w - s,
                mask=glyph != bg, color=int(values[0]))


def _linear(ages, values):
    da = ages[1] - ages[0]
    dv = values[1] - values[0]
    _require(dv % da == 0, "growth increments must be integral")
    slope = dv // da
    intercept = values[0] - slope * ages[0]
    _require(all(v == intercept + slope * age for age, v in zip(ages, values)),
             "observations do not fit one arithmetic progression")
    return int(intercept), int(slope)


def _mask(h, w, style, orient, spacing):
    """Solver-side geometry: vectorized cell predicates, without latent strokes."""
    bh, bw = (w, h) if orient in (1, 3, 6, 7) else (h, w)
    rr, cc = np.indices((bh, bw))
    if style == "elbow":
        m = (cc == 0) | (rr == bh - 1)
    elif style == "fork":
        m = (cc == 0) | (rr == 0) | (rr == bh - 1)
    elif style == "comb":
        m = (cc == 0) | (rr % spacing == 0)
    else:
        m = (cc == 0) | (cc == bw - 1) | (rr == 0) | (rr == bh - 1)
    return _array_transform(m, orient)


def _fit(info):
    obs = [p for p in info["panels"] if p["observed"]]
    missing = [p for p in info["panels"] if not p["observed"]]
    _require(3 <= len(obs) <= 4 and 1 <= len(missing) <= 3,
             "need three or four observations and one to three empty stages")
    ages = [p["age"] for p in obs]
    measurements = [_measure(p["patch"], info["bg"]) for p in obs]
    fits = {k: _linear(ages, [m[k] for m in measurements]) for k in ("h", "w", "u", "v")}
    colors = {m["color"] for m in measurements}
    period = len(colors)
    _require(1 <= period <= 3 and len(obs) >= max(3, period + 1),
             "need a repeated observation of the complete color cycle")
    cycle = {}
    for age, m in zip(ages, measurements):
        residue = age % period
        _require(residue not in cycle or cycle[residue] == m["color"], "inconsistent color phase")
        cycle[residue] = m["color"]
    _require(len(cycle) == period and set(cycle.values()) == colors, "unexposed color phase")
    descriptors = []
    for style in STYLES:
        for orient in range(8):
            for spacing in ((2, 3, 4) if style == "comb" else (0,)):
                if all(np.array_equal(_mask(m["h"], m["w"], style, orient, spacing), m["mask"])
                       for m in measurements):
                    descriptors.append((style, orient, spacing))
    _require(descriptors, "unsupported glyph growth grammar")
    # Check intermediate ages too: skipped stages must remain valid objects.
    for age in range(min(p["age"] for p in info["panels"]),
                     max(p["age"] for p in info["panels"]) + 1):
        _require(all(fits[k][0] + age * fits[k][1] >= 2 for k in ("h", "w")),
                 "extrapolation passes through a degenerate glyph")
    predictions = []
    for desc in descriptors:
        preds = {}
        for p in missing:
            preds[p["age"]] = _predicted_patch(p, fits, cycle, period, desc, info["bg"])
        predictions.append(preds)
    expected = predictions[0]
    _require(all(all(np.array_equal(candidate[age], expected[age]) for age in expected)
                 for candidate in predictions), "observations admit incompatible missing glyphs")
    return dict(fits=fits, cycle=cycle, period=period, desc=descriptors[0],
                predictions=expected, measurements=measurements, observed_ages=ages,
                missing_ages=sorted(expected), models=len(descriptors))


def _predicted_patch(p, fits, cycle, period, desc, bg, freeze_age=None):
    age, s = p["age"], p["s"]
    geometric_age = age if freeze_age is None else freeze_age
    h, w, u, v = (fits[k][0] + geometric_age * fits[k][1] for k in ("h", "w", "u", "v"))
    _require(h >= 2 and w >= 2 and (s - h + u) % 2 == 0 and (s - w + v) % 2 == 0,
             "predicted glyph does not lie on the pixel lattice")
    r, c = (s - h + u) // 2, (s - w + v) // 2
    _require(r >= 0 and c >= 0 and r + h <= s and c + w <= s,
             "predicted glyph exceeds its guide-defined frame")
    patch = np.full((s, s), bg, dtype=np.int8)
    region = patch[r:r + h, c:c + w]
    region[_mask(h, w, *desc)] = cycle[age % period]
    return patch


def _assemble(info, model, raw=False):
    a = np.full(info["shape"], info["bg"], dtype=np.int8)
    for p in info["panels"]:
        patch = p["patch"] if p["observed"] else model["predictions"][p["age"]]
        if raw:
            patch = _move_cells(patch, np.asarray(MATRICES[p["op"]]).T)
        a[p["r"]:p["r"] + p["s"], p["c"]:p["c"] + p["s"]] = patch
    return a[1:-1, 1:-1].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Normalize every local frame and complete the age-indexed growth sequence."""
    a = _grid(input_grid)
    info = _decode(a)
    return _assemble(info, _fit(info))


def _draw_glyph(h, w, style, orient, spacing, color, bg):
    """Generation-side explicit strokes, independent of _mask."""
    bh, bw = (w, h) if orient in (1, 3, 6, 7) else (h, w)
    a = np.full((bh, bw), bg, dtype=np.int8)
    for r in range(bh):
        a[r, 0] = color
    if style == "elbow":
        for c in range(bw):
            a[-1, c] = color
    elif style == "fork":
        for c in range(bw):
            a[0, c] = a[-1, c] = color
    elif style == "comb":
        for r in range(0, bh, spacing):
            for c in range(bw):
                a[r, c] = color
    else:
        for r in range(bh):
            a[r, -1] = color
        for c in range(bw):
            a[0, c] = a[-1, c] = color
    return _move_cells(a, MATRICES[orient])


def _pick(rng, choices):
    return choices[int(rng.integers(len(choices)))]


@lru_cache(maxsize=6)
def _partitions(n):
    # Bell(6)=203: exhaustive shelf feasibility is tiny and avoids lucky packing.
    if n == 1:
        return (((0,),),)
    result = []
    for groups in _partitions(n - 1):
        result.append(groups + ((n - 1,),))
        for i in range(len(groups)):
            result.append(groups[:i] + (groups[i] + (n - 1,),) + groups[i + 1:])
    return tuple(result)


def _layout(rng, sizes):
    """Pack freely shuffled, unequal frames in irregular shelves; no slot grid."""
    n = len(sizes)
    feasible = []
    for rows in _partitions(len(sizes)):
        height = sum(max(sizes[i] + 2 for i in row) for row in rows) + len(rows) - 1 + 4
        width = max(sum(sizes[i] + 2 for i in row) + len(row) - 1 for row in rows) + 4
        if max(height, width) <= 30:
            feasible.append(rows)
    if feasible:
        chosen = _pick(rng, feasible)
        rows = [list(map(int, rng.permutation(chosen[int(j)]))) for j in rng.permutation(len(chosen))]
        row_count = len(rows)
        heights = [max(sizes[i] + 2 for i in row) for row in rows]
        vgap = [1] * (row_count - 1)
        for i in rng.permutation(row_count - 1):
            vgap[int(i)] += int(rng.integers(min(2, 30 - sum(heights) - sum(vgap) - 4) + 1))
        gaps = []
        for row in rows:
            g = [1] * (len(row) - 1)
            for i in rng.permutation(len(g)):
                g[int(i)] += int(rng.integers(min(2, 30 - sum(sizes[j] + 2 for j in row) - sum(g) - 4) + 1))
            gaps.append(g)
        widths = [sum(sizes[i] + 2 for i in row) + sum(g) for row, g in zip(rows, gaps)]
        base_h, base_w = sum(heights) + sum(vgap) + 4, max(widths) + 4
        extra_h = int(rng.integers(min(4, 30 - base_h) + 1))
        extra_w = int(rng.integers(min(4, 30 - base_w) + 1))
        H, W = base_h + extra_h, base_w + extra_w
        y = 2 + int(rng.integers(extra_h + 1))
        positions = [None] * n
        for ri, (row, height, width) in enumerate(zip(rows, heights, widths)):
            x = 2 + int(rng.integers(W - 4 - width + 1))
            for j, index in enumerate(row):
                s = sizes[index]
                yy = y + int(rng.integers(height - s - 2 + 1))
                positions[index] = yy + 1, x + 1
                x += s + 2 + (gaps[ri][j] if j < len(row) - 1 else 0)
            y += height + (vgap[ri] if ri < row_count - 1 else 0)
        return (H, W), positions
    return None


def _make_pair(rng, palette, outer_index, observed, missing, period, witness=False, fallback=False):
    ages = sorted(observed + missing)
    lo, hi = min(ages), max(ages)
    bg, ka, kb, tag = palette[:4]
    body = palette[4:4 + period]
    if fallback and witness:
        style, orient, spacing = "elbow", 0, 0
        # Both axes are observed beyond length four, distinguishing an elbow
        # from a comb whose second tooth would first appear in the hidden stage.
        dh, dw, hh, ww, u0, v0, du, dv = 0, 1, 5, 4, -1, 0, 1, 0
    elif fallback:
        style, orient, spacing = "elbow", 0, 0
        dh, dw, hh, ww, u0, v0, du, dv = 0, 0, 3, 4, 1, 0, 0, 0
    else:
        style = _pick(rng, STYLES if not witness else ("elbow", "comb"))
        orient = int(rng.integers(8))
        spacing = int(rng.integers(2, 5)) if style == "comb" else 0
        dh, dw = _pick(rng, [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                            if not witness or a != 0 or b != 0])
        dimensions = [(h, w) for h in range(2, 8) for w in range(2, 8)
                      if all(2 <= h + (age - lo) * dh <= 7 and
                             2 <= w + (age - lo) * dw <= 7 for age in range(lo, hi + 1))]
        hh, ww = _pick(rng, dimensions)
        offsets = [(u, v, a, b) for u in (-1, 0, 1) for v in (-1, 0, 1)
                   for a in (-1, 0, 1) for b in (-1, 0, 1)
                   if (hh - u - ww + v) % 2 == 0 and (dh - a - dw + b) % 2 == 0]
        if len(ages) == 6:
            # Compact offsets leave room for six genuine local coordinate frames.
            cost = [1 + max(abs(u + (age - lo) * a) + abs(v + (age - lo) * b)
                            for age in ages) ** 3 for u, v, a, b in offsets]
            weights = 1 / np.asarray(cost, dtype=float)
            u0, v0, du, dv = offsets[int(rng.choice(len(offsets), p=weights / weights.sum()))]
        else:
            u0, v0, du, dv = _pick(rng, offsets)
    canonical, sizes = [], []
    for age in ages:
        h, w = hh + (age - lo) * dh, ww + (age - lo) * dw
        u, v = u0 + (age - lo) * du, v0 + (age - lo) * dv
        lower = max(4, age, h + abs(u), w + abs(v))
        if lower % 2 != (h - u) % 2:
            lower += 1
        options = [s for s in range(lower, 9, 2)]
        if not options:
            return None
        s = options[0] if fallback or len(ages) == 6 else _pick(rng, options[:2])
        r, c = (s - h + u) // 2, (s - w + v) // 2
        patch = np.full((s, s), bg, dtype=np.int8)
        patch[r:r + h, c:c + w] = _draw_glyph(h, w, style, orient, spacing,
                                            body[age % period], bg)
        canonical.append(patch)
        sizes.append(s)
    layout = _layout(rng, sizes)
    if layout is None:
        return None
    shape, positions = layout
    a = np.full(shape, bg, dtype=np.int8)
    latent = a.copy()
    outer = dict(zip(_adjacent(outer_index), (ka, kb)))
    for side, color in outer.items():
        _side(a, side)[:] = color
    for age, s, patch, (r, c) in zip(ages, sizes, canonical, positions):
        local_sides = _adjacent(int(rng.integers(4)))
        guide_colors = (ka, kb) if rng.integers(2) else (kb, ka)
        guides = dict(zip(local_sides, guide_colors))
        op = _operation(guides, outer)
        if age in observed:
            a[r:r + s, c:c + s] = _move_cells(patch, np.asarray(MATRICES[op]).T)
        for side, color in guides.items():
            _bar(a, side, r, c, s)[:] = color
        age_side = _pick(rng, [side for side in SIDES if side not in guides])
        start = int(rng.integers(s - age + 1))
        _bar(a, age_side, r, c, s)[start:start + age] = tag
        latent[r:r + s, c:c + s] = patch
    pair = {"input": a, "output": latent[1:-1, 1:-1].copy()}
    try:
        info = _decode(a)
        model = _fit(info)
        if not np.array_equal(_assemble(info, model), pair["output"]):
            return None
        if witness and not _teaching_witness(info, model):
            return None
    except ValueError:
        return None
    return pair


def _teaching_witness(info, model):
    if not any(model["fits"][k][1] for k in ("h", "w")):
        return False
    if np.array_equal(_assemble(info, model), _assemble(info, model, raw=True)):
        return False
    growth_matters, phase_matters = False, False
    for p in info["panels"]:
        if p["observed"]:
            continue
        nearest = min(model["observed_ages"], key=lambda age: abs(age - p["age"]))
        truth = model["predictions"][p["age"]]
        try:
            frozen = _predicted_patch(p, model["fits"], model["cycle"], model["period"],
                                      model["desc"], info["bg"], freeze_age=nearest)
            growth_matters |= not np.array_equal(truth, frozen)
        except ValueError:
            # A constant-size/constant-position hypothesis cannot fit this frame.
            growth_matters = True
        phase_matters |= model["cycle"][nearest % model["period"]] != model["cycle"][p["age"] % model["period"]]
    return growth_matters and phase_matters


def _swap_outer(pair):
    """Exact contrast: only identities on the two occupied outer sides change."""
    a = pair["input"].copy()
    info = _decode(a)
    values = list(info["outer"].values())
    for side, color in info["outer"].items():
        _side(a, side)[1:-1] = values[1] if color == values[0] else values[0]
    H, W = a.shape
    # Recreate legal corner overwrites after the identity reversal.
    for r, c in ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)):
        touching = [s for s in info["outer"] if
                    (s == "T" and r == 0) or (s == "B" and r == H - 1) or
                    (s == "L" and c == 0) or (s == "R" and c == W - 1)]
        a[r, c] = (values[1] if info["outer"][touching[-1]] == values[0] else values[0]) if touching else info["bg"]
    new_info = _decode(a)
    # Labels move explicit old canonical cells by the change of reference frame.
    out = np.full(a.shape, info["bg"], dtype=np.int8)
    for old, new in zip(info["panels"], new_info["panels"]):
        r, c, s = old["r"], old["c"], old["s"]
        patch = pair["output"][r - 1:r - 1 + s, c - 1:c - 1 + s]
        relative = np.asarray(MATRICES[new["op"]]) @ np.asarray(MATRICES[old["op"]]).T
        out[r:r + s, c:c + s] = _move_cells(patch, relative)
    return {"input": a, "output": out[1:-1, 1:-1].copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    stable_palette = bool(rng.random() < 0.28)
    palette = list(map(int, rng.permutation(10)))
    shift = _pick(rng, ("distance", "backward", "multiple"))
    first_outer = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(nt + ne):
        split = "train" if index < nt else "test"
        if index == 1:
            pair = _swap_outer(puzzle["train"][0])
        else:
            period = 2 if index == 0 else int(rng.integers(1, 4))
            nobs = max(3, period + 1)
            observed = list(range(1, nobs + 1))
            missing = [nobs + 1]
            if split == "test":
                if shift == "distance":
                    missing = [nobs + 2]
                elif shift == "backward":
                    start = int(rng.integers(2, 7 - nobs))
                    observed = list(range(start, start + nobs))
                    missing = [1]
                else:
                    period = int(rng.integers(1, 3))
                    observed = [1, 2, 3]
                    missing = list(range(4, int(rng.integers(5, 7)) + 1))
            local_palette = palette if stable_palette or index == 0 else list(map(int, rng.permutation(10)))
            outer_index = first_outer if index == 0 else int(rng.integers(4))
            if split == "test" and index == nt:
                outer_index = (first_outer + int(rng.integers(1, 4))) % 4
            pair = None
            for attempt in range(200):
                candidate = _make_pair(rng, local_palette, outer_index, observed, missing,
                                       period, witness=index == 0)
                if candidate is None or _key(candidate["input"]) in seen:
                    continue
                if index == 0 and np.array_equal(candidate["output"], _swap_outer(candidate)["output"]):
                    continue
                pair = candidate
                break
            if pair is None:
                # Constructive small fallback, preserving the requested shift.
                # Its arithmetic dimensions use lo, so backward ages remain valid.
                for _ in range(40):
                    candidate = _make_pair(rng, local_palette, outer_index, observed, missing,
                                           period, witness=index == 0, fallback=True)
                    if candidate is not None and _key(candidate["input"]) not in seen:
                        if index != 0 or not np.array_equal(candidate["output"], _swap_outer(candidate)["output"]):
                            pair = candidate
                            break
            _require(pair is not None, "bounded layout construction exhausted")
        _require(_key(pair["input"]) not in seen, "unexpected duplicate input")
        seen.add(_key(pair["input"]))
        puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            info = _decode(a)
            model = _fit(info)
            _require(np.array_equal(_assemble(info, model), b), "incorrect completion or normalization")
            _require(b.shape == (a.shape[0] - 2, a.shape[1] - 2), "incorrect border removal")
            _require(not set(map(int, np.unique(b))) & (set(info["outer"].values()) | {info["tag"]}),
                     "annotations leaked into output")
            # Check exact observed-cell transport and the missing stages separately.
            for p in info["panels"]:
                r, c, s = p["r"] - 1, p["c"] - 1, p["s"]
                out_patch = b[r:r + s, c:c + s]
                if p["observed"]:
                    _require(np.array_equal(np.sort(out_patch.ravel()), np.sort(p["raw"].ravel())),
                             "normalization must preserve all observed cells")
                else:
                    _require(np.any(out_patch != info["bg"]), "empty stage was not reconstructed")
            records[split].append((info, model))
    a, am = records["train"][0]
    b, _ = records["train"][1]
    _require(_teaching_witness(a, am), "first demonstration must expose growth, color phase, and orientation")
    _require(a["outer"].keys() == b["outer"].keys() and
             all(a["outer"][s] != b["outer"][s] for s in a["outer"]),
             "training needs a reversal of outer identities with fixed occupied sides")
    ia, ib = _grid(puzzle["train"][0]["input"]), _grid(puzzle["train"][1]["input"])
    _require(ia.shape == ib.shape and np.array_equal(ia[1:-1, 1:-1], ib[1:-1, 1:-1]),
             "contrast must preserve the entire interior, not just its side layout")
    _require(not np.array_equal(puzzle["train"][0]["output"], puzzle["train"][1]["output"]),
             "outer identity contrast must change the answer")
    for info, model in records["train"]:
        _require(model["observed_ages"] == list(range(1, len(model["observed_ages"]) + 1)) and
                 model["missing_ages"] == [max(model["observed_ages"]) + 1],
                 "generated training should teach immediate forward continuation")
    for info, model in records["test"]:
        obs, miss = model["observed_ages"], model["missing_ages"]
        _require(min(miss) < min(obs) or max(miss) >= max(obs) + 2 or len(miss) >= 2,
                 "test must extend the demonstrated age relation")


def self_test():
    """Bounded checks of semantic witnesses, equivariance, counts, and 42 seeds."""
    metrics = {"seeds": 42, "pairs": 0, "growth": 0, "shrink": 0,
               "drift": 0, "backward": 0, "skipped": 0, "multiple": 0,
               "comb_growth": 0, "blank_edges": 0}
    shapes, sizes, operations, styles, periods, palettes, layouts, signatures = (set() for _ in range(8))
    examples = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        if seed < 3:
            other = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], other[s]) for k in p)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                info = _decode(a)
                model = _fit(info)
                assert np.array_equal(solve(a), pair["output"])
                metrics["pairs"] += 1
                slopes = [model["fits"][k][1] for k in ("h", "w")]
                metrics["growth"] += any(x > 0 for x in slopes)
                metrics["shrink"] += any(x < 0 for x in slopes)
                metrics["drift"] += any(model["fits"][k][1] for k in ("u", "v"))
                obs, miss = model["observed_ages"], model["missing_ages"]
                metrics["backward"] += min(miss) < min(obs)
                metrics["skipped"] += max(miss) > max(obs) + 1
                metrics["multiple"] += len(miss) > 1
                metrics["comb_growth"] += model["desc"][0] == "comb" and any(slopes)
                styles.add(model["desc"][0]); periods.add(model["period"])
                palettes.add((info["bg"], tuple(sorted(info["outer"].values())), info["tag"]))
                shapes.add((a.shape, pair["output"].shape))
                layouts.add(tuple((p["r"], p["c"], p["s"], p["age"]) for p in info["panels"]))
                for p in info["panels"]:
                    sizes.add(p["s"]); operations.add(p["op"])
                    patch = p["patch"] if p["observed"] else model["predictions"][p["age"]]
                    metrics["blank_edges"] += any(np.all(_side(patch, s) == info["bg"]) for s in SIDES)
                if seed < 2:
                    examples.append(pair)
    assert len(signatures) == 42 and len(shapes) >= 30 and len(layouts) >= 120
    assert operations == set(range(8)) and styles == set(STYLES) and periods == {1, 2, 3}
    assert len(sizes) >= 4 and len(palettes) >= 50
    assert all(metrics[k] >= 5 for k in ("growth", "shrink", "drift", "backward", "skipped", "multiple", "comb_growth"))
    assert metrics["blank_edges"] >= 100
    # Exercise the constructive fallback directly, rather than relying on it
    # being reached by these seeds. Six small frames always have a feasible pack.
    fallback_rng = np.random.default_rng(440)
    for observed, missing, period, witness in (([1, 2, 3], [4], 2, True),
                                              ([1, 2, 3, 4], [6], 3, False),
                                              ([3, 4, 5], [1], 2, False),
                                              ([1, 2, 3], [4, 5, 6], 2, False)):
        for outer in range(4):
            pair = _make_pair(fallback_rng, list(range(10)), outer, observed, missing,
                              period, witness=witness, fallback=True)
            assert pair is not None and np.array_equal(solve(pair["input"]), pair["output"])
            if witness:
                swapped = _swap_outer(pair)
                assert not np.array_equal(pair["output"], swapped["output"])
                assert np.array_equal(solve(swapped["input"]), swapped["output"])
    # Equivariance checks exercise input-only parsing, local frames, age bars,
    # arbitrary background identities, and all eight orientation operations.
    permutation = np.array([7, 3, 9, 1, 6, 0, 8, 4, 2, 5], dtype=np.int8)
    for pair in examples:
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
        for op in range(8):
            assert np.array_equal(solve(_array_transform(pair["input"], op)),
                                  _array_transform(pair["output"], op))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(3100 + nt * 31 + ne, nt, ne)
            validate_generated_puzzle(puzzle)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    original = generate_puzzle(0)
    for kind in ("cell", "crop", "guide", "age", "phase"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in original[s]] for s in original}
        pair = bad["test"][0]
        info = _decode(pair["input"])
        p = info["panels"][0]
        if kind == "cell":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif kind == "crop":
            pair["output"] = pair["output"][1:, :]
        elif kind == "guide":
            side = next(iter(p["guides"]))
            _bar(pair["input"], side, p["r"], p["c"], p["s"])[p["s"] // 2] = info["bg"]
        elif kind == "age":
            line = _bar(pair["input"], p["tag_side"], p["r"], p["c"], p["s"])
            line[:] = info["bg"]
        else:
            p = next(q for q in info["panels"] if q["observed"])
            region = pair["input"][p["r"]:p["r"] + p["s"], p["c"]:p["c"] + p["s"]]
            rr, cc = np.argwhere(region != info["bg"])[0]
            region[rr, cc] = info["tag"]
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupted " + kind + " accepted")
    metrics.update(distinct_puzzles=len(signatures), grid_shapes=len(shapes),
                   layouts=len(layouts), frame_sizes=sorted(sizes),
                   operations=sorted(operations), styles=sorted(styles), periods=sorted(periods),
                   role_palettes=len(palettes))
    return metrics
