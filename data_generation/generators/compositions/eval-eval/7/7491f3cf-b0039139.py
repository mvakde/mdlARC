"""Cut-and-transcribe: a spatial cut determines an ordered word of glyphs.

Only standard-library modules and NumPy are used. Nothing is written on import,
generation, validation, or self-test. See task_spec.md for the complete rule.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "bad ARC colors")
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.ascontiguousarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask, diagonal=False):
    """Components in deterministic first-cell order, independent of set order."""
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = [p], [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    group.append(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _mask(shape, cells):
    result = np.zeros(shape, dtype=bool)
    for p in cells:
        result[p] = True
    return result


def _crop(array, occupied):
    points = np.argwhere(occupied)
    _require(len(points) > 0, "empty glyph")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return array[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _geometry(cut):
    """Validate a simple digital path and recover its two four-connected sides."""
    h, w = cut.shape
    cells = {tuple(map(int, p)) for p in np.argwhere(cut)}
    graph = {}
    for r, c in cells:
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = r + dr, c + dc
                if not (dr or dc) or q not in cells:
                    continue
                # Do not shortcut across the diagonal of an orthogonal elbow.
                if dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells):
                    continue
                neighbors.append(q)
        graph[r, c] = neighbors
    ends = sorted(p for p in graph if len(graph[p]) == 1)
    _require(len(cells) >= 3 and len(ends) == 2
             and all(1 <= len(ns) <= 2 for ns in graph.values()), "cut is not a simple path")
    border = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _require(all(border(p) for p in ends), "cut endpoints must reach the boundary")
    _require(all(not border(p) for p in cells - set(ends)), "extra boundary contact")
    visited, stack = set(), [ends[0]]
    while stack:
        p = stack.pop()
        if p not in visited:
            visited.add(p)
            stack.extend(graph[p])
    _require(visited == cells, "disconnected cut")
    groups = _components(~cut)
    _require(len(groups) == 2, "cut must leave exactly two regions")
    regions = [_mask(cut.shape, g) for g in groups]
    near = cut.copy()
    for r, c in cells:
        near[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    sites = [[tuple(map(int, p)) for p in np.argwhere(region & ~near)] for region in regions]
    _require(all(sites), "both sides need an isolated marker site")
    return regions, sites


def _counter_details(specimen, cut, regions):
    surviving = specimen & ~cut
    fragments = _components(surviving)
    _require(len(fragments) == len(_components(surviving, True)), "diagonally touching fragments")
    original = _components(specimen)
    _require(len(original) == len(_components(specimen, True)), "diagonally touching original objects")
    sides = []
    for group in fragments:
        values = {0 if regions[0][p] else 1 for p in group}
        _require(len(values) == 1, "fragment crosses the cut")
        sides.append(values.pop())
    _require(set(sides) == {0, 1}, "both regions must supply a fragment")
    column_order = sorted(range(len(fragments)), key=lambda i: min((c, r) for r, c in fragments[i]))
    original_ids = {p: i for i, group in enumerate(original) for p in group}
    parent_sides = [(original_ids[group[0]], side) for group, side in zip(fragments, sides)]
    parent_order = sorted(range(len(fragments)), key=lambda i: (parent_sides[i][0], min(fragments[i])))
    return {"fragments": fragments, "sides": sides, "original": original,
            "surviving": surviving, "column_sides": [sides[i] for i in column_order],
            "parent_grouped_sides": [sides[i] for i in parent_order],
            "same_parent_same_side_split": len(parent_sides) > len(set(parent_sides))}


def _pack(panels, frame):
    h, w = panels[0].shape
    out = np.full((h + 2, 4 * w + 5), frame, dtype=np.int8)
    for i, panel in enumerate(panels):
        out[1:-1, 1 + i * (w + 1):1 + i * (w + 1) + w] = panel
    return out


def _decode(input_grid):
    """All roles, orientation, fragments, and glyphs come from the input alone."""
    source = _grid(input_grid)
    candidates = []
    for vertical in (False, True):
        g = source.T if vertical else source
        h, width = g.shape
        if h < 5 or (width - 5) % 4:
            continue
        w = (width - 5) // 4
        if w < 3:
            continue
        frame = int(g[0, 0])
        if not (np.all(g[[0, -1]] == frame) and np.all(g[:, ::w + 1] == frame)):
            continue
        panels = [g[1:-1, 1 + i * (w + 1):1 + i * (w + 1) + w] for i in range(4)]
        colors = [set(map(int, np.unique(p))) for p in panels]
        if not all(len(cs) == 2 and frame not in cs for cs in colors):
            continue
        common = set.intersection(*colors)
        if len(common) != 1:
            continue
        bg = next(iter(common))
        inks = [next(iter(cs - {bg})) for cs in colors]
        if len({frame, bg, *inks}) != 6:
            continue
        try:
            control_groups = sorted(_components(panels[0] != bg, True), key=len)
            _require(len(control_groups) == 2 and len(control_groups[0]) == 1
                     and len(control_groups[1]) >= 3, "cut/marker ambiguity")
            marker = control_groups[0][0]
            cut = _mask(panels[0].shape, control_groups[1])
            regions, sites = _geometry(cut)
            side = 0 if regions[0][marker] else 1
            _require(marker in sites[side], "marker touches cut")
            specimen = panels[1] != bg
            d = _counter_details(specimen, cut, regions)
            glyphs = [_crop(p, p != bg) for p in panels[2:]]
            _require(glyphs[0].shape == glyphs[1].shape, "glyph crop shapes must match")
            n = len(d["fragments"])
            th, tw = glyphs[0].shape
            _require(n * (tw + 1) - 1 <= 30 and th <= 30, "tape exceeds ARC limits")
            symbols = [0 if region_side == side else 1 for region_side in d["sides"]]
            d.update({"vertical": vertical, "panels": panels, "frame": frame, "bg": bg,
                      "inks": inks, "cut": cut, "regions": regions, "sites": sites,
                      "marker": marker, "side": side, "specimen": specimen,
                      "glyphs": glyphs, "symbols": symbols})
            candidates.append(d)
        except AssertionError:
            continue
    _require(len(candidates) == 1, "input has no unique valid four-sheet parse")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _decode(input_grid)
    h, w = d["glyphs"][0].shape
    # Cellwise rendering is separate from the concatenation used in generation.
    out = np.full((h, len(d["symbols"]) * (w + 1) - 1), d["bg"], dtype=np.int8)
    for r in range(h):
        for c in range(out.shape[1]):
            index, offset = divmod(c, w + 1)
            if offset < w:
                out[r, c] = d["glyphs"][d["symbols"][index]][r, offset]
    return out.T.copy() if d["vertical"] else out


def _cut_kind(cut):
    rows, cols = np.where(cut)
    if len(set(rows)) == 1:
        return "horizontal"
    if len(set(cols)) == 1:
        return "vertical"
    if np.all(cut.sum(axis=1) == 1):
        return "diagonal"
    h, w = cut.shape
    top, bottom = bool(cut[0].any()), bool(cut[-1].any())
    left, right = bool(cut[:, 0].any()), bool(cut[:, -1].any())
    return "stepped" if (top and bottom) or (left and right) else "elbow"


def _sample_cut(rng, h, w, kind):
    for _ in range(48):
        cut = np.zeros((h, w), dtype=bool)
        if kind == "vertical":
            cut[:, int(rng.integers(2, w - 2))] = True
        elif kind == "horizontal":
            cut[int(rng.integers(2, h - 2)), :] = True
        elif kind == "diagonal":
            a, b = [int(x) for x in rng.choice(np.arange(1, w - 1), 2, replace=False)]
            cols = np.rint(np.linspace(a, b, h)).astype(int)
            cut[np.arange(h), cols] = True
        elif kind == "stepped":
            if w < 6:
                continue
            col, last = int(rng.integers(2, 4)), 0
            possible = list(range(3, h - 3, 3))
            turns = sorted(map(int, rng.choice(possible, int(rng.integers(1, min(3, len(possible)) + 1)), replace=False)))
            for row in turns:
                cut[last:row + 1, col] = True
                next_col = 5 - col
                cut[row, min(col, next_col):max(col, next_col) + 1] = True
                col, last = next_col, row
            cut[last:, col] = True
        elif kind == "elbow":
            r, c = int(rng.integers(3, h - 3)), int(rng.integers(1, w - 2))
            cut[:r + 1, c] = True
            cut[r, c:] = True
        else:
            raise ValueError("unknown cut kind")
        if rng.random() < .5:
            cut = cut[:, ::-1].copy()
        if rng.random() < .5:
            cut = cut[::-1].copy()
        try:
            _geometry(cut)
            return cut
        except AssertionError:
            pass
    raise RuntimeError("bounded cut construction exhausted")


def _grow(allowed, area, rng):
    sites = np.argwhere(allowed)
    _require(len(sites) > 0, "no growth site")
    p = tuple(map(int, sites[int(rng.integers(len(sites)))]))
    frontier, result = {p}, np.zeros(allowed.shape, dtype=bool)
    h, w = allowed.shape
    for _ in range(area):
        if not frontier:
            break
        choices = sorted(frontier)
        r, c = choices[int(rng.integers(len(choices)))]
        frontier.remove((r, c))
        result[r, c] = True
        for x, y in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= x < h and 0 <= y < w and allowed[x, y] and not result[x, y]:
                frontier.add((x, y))
    return result


def _band_specimen(cut, jobs, rng):
    h, w = cut.shape
    regions, _ = _geometry(cut)
    heights = [int(rng.integers(1, 4)) for job in jobs]
    extra = h - sum(heights) - len(jobs) + 1
    if extra < 0:
        heights = [1] * len(jobs)
        extra = h - 2 * len(jobs) + 1
    blanks = rng.multinomial(extra, np.full(len(jobs) + 1, 1 / (len(jobs) + 1)))
    specimen = np.zeros(cut.shape, dtype=bool)
    r0 = int(blanks[0])
    for i, (job, height) in enumerate(zip(jobs, heights)):
        r1 = r0 + height
        if job == "cross":
            obj = np.ones((height, w), dtype=bool)
            style = int(rng.integers(3))
            if style == 1 and height >= 3:
                obj[1:-1, 1:-1] = False  # clean ring
            elif style == 2 and height >= 2:
                obj[:] = False
                pivot = int(rng.integers(height))
                obj[pivot] = True
                obj[:, 0] = True
                obj[:, -1] = True
            specimen[r0:r1] |= obj
        else:
            allowed = regions[int(job)].copy()
            allowed[:r0] = allowed[r1:] = False
            specimen |= _grow(allowed, int(rng.integers(1, min(10, int(allowed.sum())) + 1)), rng)
        r0 = r1 + 1 + int(blanks[i + 1])
    return specimen


def _local_shape(rng, h, w):
    style = int(rng.integers(6))
    m = np.ones((h, w), dtype=bool)
    if style == 1 and h >= 3 and w >= 3:
        m[1:-1, 1:-1] = False
    elif style == 2:
        m[:] = False
        m[:, 0] = m[0] = True
    elif style == 3:
        m = _grow(m, int(rng.integers(1, h * w + 1)), rng)
    elif style == 4:
        m[:] = False
        m[h // 2] = m[:, w // 2] = True
    elif style == 5:
        m[:] = False
        m[::2] = True
        for r in range(1, h, 2):
            m[r, 0 if (r // 2) % 2 else w - 1] = True
    return m


def _free_specimen(cut, rng):
    h, w = cut.shape
    result = np.zeros((h, w), dtype=bool)
    for _ in range(int(rng.integers(2, 7))):
        for _ in range(14):
            oh, ow = int(rng.integers(1, min(7, h) + 1)), int(rng.integers(1, w + 1))
            r, c = int(rng.integers(h - oh + 1)), int(rng.integers(w - ow + 1))
            if result[max(0, r - 1):r + oh + 1, max(0, c - 1):c + ow + 1].any():
                continue
            result[r:r + oh, c:c + ow] = _local_shape(rng, oh, ow)
            break
    return result


def _staggered(groups):
    """Partly overlapping row ranges, excluding simple nesting and row ties."""
    bounds = [(min(r for r, c in g), max(r for r, c in g)) for g in groups]
    return any(a[0] < b[0] <= a[1] < b[1] or b[0] < a[0] <= b[1] < a[1]
               for i, a in enumerate(bounds) for b in bounds[i + 1:])


def _asymmetric(mask):
    return all(not np.array_equal(mask, np.rot90(mask[:, ::-1] if k >= 4 else mask, k % 4))
               for k in range(1, 8))


def _glyph_mask(rng, h, w, witness):
    for _ in range(96):
        mask = _local_shape(rng, h, w)
        # Some glyphs are disconnected; their whole bounding box is one symbol.
        if rng.random() < .15:
            rr, cc = np.indices((h, w))
            mask = (rr + cc) % 2 == int(rng.integers(2))
        if _crop(mask, mask).shape == (h, w) and (not witness or (_asymmetric(mask) and not mask.all())):
            return mask
    # A fully asymmetric connected branch, including for square crop shapes.
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = mask[0, :] = True
    if h == w:
        mask[-1, 1] = True
    _require(not witness or _asymmetric(mask), "asymmetric glyph fallback failed")
    return mask


def _glyph_panels(rng, shape, bg, inks, count, witness):
    h, w = shape
    th = int(rng.integers(3 if witness else 2, min(h - 2, 7 if witness else 10) + 1))
    tw = int(rng.integers(2, min(w - 2, 31 // count - 1) + 1))
    panels = []
    for ink in inks:
        m = _glyph_mask(rng, th, tw, witness)
        panel = np.full(shape, bg, dtype=np.int8)
        top = int(rng.integers(1 if witness else 0, h - th if witness else h - th + 1))
        left = int(rng.integers(1 if witness else 0, w - tw if witness else w - tw + 1))
        panel[top:top + th, left:left + tw][m] = ink
        panels.append(panel)
    return panels


def _make_pair(cut, specimen, marker_side, palette, glyph_panels, vertical, rng):
    frame, bg, control, ink, _, _ = palette
    regions, sites = _geometry(cut)
    marker = sites[marker_side][int(rng.integers(len(sites[marker_side])))]
    ctrl = np.full(cut.shape, bg, dtype=np.int8)
    ctrl[cut] = ctrl[marker] = control
    objects = np.where(specimen, ink, bg).astype(np.int8)
    inp = _pack([ctrl, objects, *glyph_panels], frame)
    d = _counter_details(specimen, cut, regions)
    tiles = [_crop(p, p != bg) for p in glyph_panels]
    gap = np.full((tiles[0].shape[0], 1), bg, dtype=np.int8)
    pieces = []
    for i, side in enumerate(d["sides"]):
        if i:
            pieces.append(gap)
        pieces.append(tiles[0 if side == marker_side else 1])
    out = np.concatenate(pieces, axis=1)
    return {"input": inp.T.copy() if vertical else inp,
            "output": out.T.copy() if vertical else out}


def _nested_specimen(cut, rng):
    """A U is severed around a separate object in its cavity.

    Its lower fragment must be read AFTER the nested object's fragment, so
    grouping output letters by original object cannot imitate the rule.
    """
    h, w = cut.shape
    _require(w == 6, "nested witness needs six columns")
    starts = [r for r in range(1, h - 7) if int(cut[r + 2].sum()) == 1]
    _require(starts, "no nested-object site")
    r = starts[int(rng.integers(len(starts)))]
    cut_col = int(np.flatnonzero(cut[r + 2])[0])
    connected_right = cut_col <= 2
    specimen = np.zeros_like(cut)
    specimen[[r, r + 4]] = True
    specimen[r:r + 5, -1 if connected_right else 0] = True
    specimen[r + 2, 3 if connected_right else 2] = True
    regions, _ = _geometry(cut)
    allowed = regions[1].copy()
    allowed[:h - 2] = False
    specimen |= _grow(allowed, int(rng.integers(1, min(5, int(allowed.sum())) + 1)), rng)
    return specimen


def _witness_specimen(cut, rng, first):
    regions, _ = _geometry(cut)
    jobs = ["cross", 1, 0]
    target = 5 if first else 4
    for _ in range(96):
        specimen = _nested_specimen(cut, rng) if first else _band_specimen(cut, jobs, rng)
        try:
            d = _counter_details(specimen, cut, regions)
        except AssertionError:
            continue
        s = d["sides"]
        if len(s) != target or len(d["original"]) == target:
            continue
        if first and not d["same_parent_same_side_split"]:
            continue
        if first and (s == sorted(s) or s == sorted(s, reverse=True)
                      or s == s[::-1] or s == d["column_sides"]
                      or s == d["parent_grouped_sides"]
                      or all(a != b for a, b in zip(s, s[1:]))):
            continue
        return specimen
    # A U and two side objects fix a five-fragment witness; a bar and
    # two side objects fix the four-fragment witness.
    specimen = np.zeros(cut.shape, dtype=bool)
    rows = [1, cut.shape[0] // 2, cut.shape[0] - 2]
    if first:
        specimen = _nested_specimen(cut, rng)
    else:
        specimen[rows[0]] = True
        right = np.flatnonzero(regions[1][rows[1]])
        specimen[rows[1], int(right[-1])] = True
        left = np.flatnonzero(regions[0][rows[2]])
        specimen[rows[2], int(left[0])] = True
    d = _counter_details(specimen, cut, regions)
    _require(len(d["sides"]) == target and (not first or d["same_parent_same_side_split"]),
             "witness fallback failed")
    return specimen


def _ordinary_pair(rng, params, split):
    h, w = int(rng.integers(11, 25)), int(rng.integers(5, 7))
    kinds = ["vertical", "diagonal"] + (["stepped"] if w == 6 else [])
    if split == "test" and params["regime"] == "bend":
        kind = "elbow"
    elif params["regime"] == "layout" or (split == "test" and params["regime"] == "count"):
        kind = str(rng.choice(kinds))
    else:
        kind = str(rng.choice(kinds + ["horizontal"] + ([] if params["regime"] == "bend" else ["elbow"])))
    cut = _sample_cut(rng, h, w, kind)
    regions, _ = _geometry(cut)
    desired = int(rng.integers(6, 8)) if split == "test" and params["regime"] == "count" else None
    max_count = 7 if split == "test" else 5
    scattered = split == "test" and params["regime"] == "layout"
    specimen = None
    for _ in range(96):
        if desired is not None:
            jobs = ["cross"] * 3 + ([int(rng.integers(2))] if desired == 7 else [])
            rng.shuffle(jobs)
            candidate = _band_specimen(cut, jobs, rng)
        elif params["regime"] == "layout" and split == "train":
            jobs = list(rng.choice([0, 1, "cross"], int(rng.integers(2, 5))))
            jobs = [x if x == "cross" else int(x) for x in jobs]
            candidate = _band_specimen(cut, jobs, rng)
        else:
            candidate = _free_specimen(cut, rng)
        try:
            info = _counter_details(candidate, cut, regions)
        except AssertionError:
            continue
        n = len(info["sides"])
        if not 2 <= n <= max_count or (desired is not None and n != desired):
            continue
        if scattered and not _staggered(info["original"]):
            continue
        specimen = candidate
        break
    if specimen is None:
        specimen = np.zeros(cut.shape, dtype=bool)
        if desired is not None:
            rows = [1, h // 2, h - 2]
            specimen[rows] = True
            if desired == 7:
                # Keep an isolated side object between the first two bars.
                specimen[(rows[0] + rows[1]) // 2, 0] = True
        elif kind == "horizontal":
            specimen[:, [0, -1]] = True
        elif kind == "elbow":
            specimen[[1, h - 2]] = True
        else:
            specimen[1:4, 0] = True
            specimen[2:5, -1] = True
            specimen[h - 2] = True
        info = _counter_details(specimen, cut, regions)
        _require(2 <= len(info["sides"]) <= max_count, "ordinary fallback failed")
        _require(desired is None or len(info["sides"]) == desired, "count fallback failed")
    n = len(_counter_details(specimen, cut, regions)["sides"])
    palette = params["palette"] if params["fixed_palette"] else tuple(map(int, rng.choice(10, 6, replace=False)))
    glyph_panels = _glyph_panels(rng, cut.shape, palette[1], palette[4:], n, False)
    vertical = bool(rng.integers(2)) if params["mixed_orientation"] else params["vertical"]
    return _make_pair(cut, specimen, int(rng.integers(2)), palette, glyph_panels, vertical, rng)


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    params = {"palette": tuple(map(int, rng.choice(10, 6, replace=False))),
              "fixed_palette": bool(rng.random() < .3), "vertical": bool(rng.integers(2)),
              "mixed_orientation": nt >= 3 and bool(rng.random() < .65),
              "regime": str(rng.choice(["bend", "count", "layout"]))}
    h, w = int(rng.integers(14, 25)), 6
    kind = str(rng.choice(["vertical", "diagonal"] + (["stepped"] if w == 6 else [])))
    cut = _sample_cut(rng, h, w, kind)
    glyphs = _glyph_panels(rng, cut.shape, params["palette"][1], params["palette"][4:], 5, True)
    first_side = int(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    for i in range(2):
        specimen = _witness_specimen(cut, rng, i == 0)
        pair = _make_pair(cut, specimen, first_side if i == 0 else 1 - first_side,
                          params["palette"], glyphs, params["vertical"], rng)
        puzzle["train"].append(pair)
        seen.add(_key(pair["input"]))
    for split, amount in (("train", nt - 2), ("test", nq)):
        for i in range(amount):
            for _ in range(32):
                pair = _ordinary_pair(rng, params, split)
                if split == "train" and i == 0 and params["mixed_orientation"]:
                    # Establish the second orientation before it can appear at test.
                    if _decode(pair["input"])["vertical"] == params["vertical"]:
                        pair = {key: value.T.copy() for key, value in pair.items()}
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_evidence(ds):
    """Explicitly reject common simpler explanations of the demonstrations."""
    return {
        "opposite_marker_sides": {d["side"] for d in ds} == {0, 1},
        "different_counts": len({len(d["symbols"]) for d in ds}) >= 2,
        "cut_splits_objects": any(len(d["fragments"]) > len(d["original"]) for d in ds),
        "not_one_per_object_per_side": any(d["same_parent_same_side_split"] for d in ds),
        "not_grouped_by_original_object": any(d["sides"] != d["parent_grouped_sides"] for d in ds),
        "count_not_area": any(int(d["surviving"].sum()) > len(d["fragments"]) for d in ds),
        "not_grouped": any(d["sides"] != sorted(d["sides"])
                           and d["sides"] != sorted(d["sides"], reverse=True) for d in ds),
        "not_alternating": any(any(a == b for a, b in zip(d["symbols"], d["symbols"][1:])) for d in ds),
        "not_reverse": any(d["symbols"] != d["symbols"][::-1] for d in ds),
        "not_column_first": any(d["sides"] != d["column_sides"] for d in ds),
        "tight_crop": any(any(p.shape != g.shape for p, g in zip(d["panels"][2:], d["glyphs"])) for d in ds),
        "orientation": any(all(_asymmetric(g != d["bg"]) for g in d["glyphs"]) for d in ds),
    }


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[k], list) for k in puzzle), "splits must be lists")
    try:
        _counts(len(puzzle["train"]), len(puzzle["test"]))
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            _require(np.array_equal(solve(inp), out), "incorrect transcription")
            canonical_out = out.T if d["vertical"] else out
            h, w = d["glyphs"][0].shape
            n = len(d["fragments"])
            _require(canonical_out.shape == (h, n * (w + 1) - 1), "incorrect tape dimensions")
            _require(np.all(canonical_out[:, w::w + 1] == d["bg"]), "incorrect separators")
            _require(set(map(int, np.unique(out))) <= {d["bg"], *d["inks"][2:]}, "control/specimen leaked into tape")
            for i, symbol in enumerate(d["symbols"]):
                _require(np.array_equal(canonical_out[:, i * (w + 1):i * (w + 1) + w], d["glyphs"][symbol]),
                         "cropped glyph changed")
            if split == "train":
                training.append(d)
    flags = _training_evidence(training)
    _require(all(flags.values()), "missing training evidence: " + ",".join(k for k, v in flags.items() if not v))


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, bounded memory, targeted metamorphic checks; no disk output."""
    puzzle_hashes, shapes, glyph_shapes, cuts, words, layouts = set(), set(), set(), set(), set(), set()
    directions, counts, kinds = Counter(), Counter(), Counter()
    split_cases = staggered_pairs = severed_pairs = disappeared = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzle_hashes.add(_digest(puzzle))
        if seed in (0, 41):
            _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterministic seed")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _decode(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solver mismatch")
                shapes.add((pair["input"].shape, pair["output"].shape))
                glyph_shapes.add(d["glyphs"][0].shape)
                cuts.add(_key(d["cut"]))
                words.add(tuple(d["symbols"]))
                layouts.add((_key(d["cut"]), _key(d["specimen"])))
                directions[d["vertical"]] += 1
                counts[len(d["fragments"])] += 1
                kinds[_cut_kind(d["cut"])] += 1
                severed_pairs += len(d["fragments"]) > len(d["original"])
                disappeared += any(all(d["cut"][p] for p in group) for group in d["original"])
                staggered_pairs += _staggered(d["original"])
        # Move only the marker: symbol identities switch while fragments stay fixed.
        pair = puzzle["train"][0]
        d = _decode(pair["input"])
        panels = [p.copy() for p in d["panels"]]
        panels[0][d["marker"]] = d["bg"]
        marker = d["sites"][1 - d["side"]][0]
        panels[0][marker] = d["inks"][0]
        moved = _pack(panels, d["frame"])
        if d["vertical"]:
            moved = moved.T.copy()
        other = _decode(moved)
        _require(other["symbols"] == [1 - x for x in d["symbols"]], "marker fails to switch sources")
        _require(not np.array_equal(solve(moved), pair["output"]), "marker has no visible effect")
        # A whole-grid transpose exchanges reading/repetition axes without rotating a glyph relative to its input.
        _require(np.array_equal(solve(pair["input"].T.copy()), pair["output"].T), "transpose covariance failed")
        # Arbitrary palette bijections preserve geometry and exact source colors.
        perm = np.random.default_rng(seed + 9000).permutation(10).astype(np.int8)
        _require(np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]]), "color covariance failed")
        split_cases += 1
    _require(len(puzzle_hashes) == 42 and len(layouts) >= 100, "structural diversity collapsed")
    _require(len(shapes) >= 30 and len(glyph_shapes) >= 15 and len(cuts) >= 50, "geometry diversity collapsed")
    _require(len(words) >= 15 and len(counts) >= 5 and set(directions) == {False, True}, "word/count/axis diversity collapsed")
    _require(set(kinds) == {"horizontal", "vertical", "diagonal", "stepped", "elbow"}, "cut family missing")
    _require(severed_pairs >= 84 and staggered_pairs >= 5, "cutting or staggered layouts missing")
    for nt, nq in ((2, 1), (10, 1), (10, 4), (3, 2)):
        p = generate_puzzle(7000 + nt * 10 + nq, nt, nq)
        _require((len(p["train"]), len(p["test"])) == (nt, nq), "ignored counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("accepted illegal counts")
    bad = generate_puzzle(17, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("accepted corrupted output")
    # An explicit surgery: one six-cell bar becomes two fragments of sizes 2/3.
    cut = np.zeros((7, 6), dtype=bool)
    cut[:, 2] = True
    specimen = np.zeros_like(cut)
    specimen[3] = True
    regions, _ = _geometry(cut)
    info = _counter_details(specimen, cut, regions)
    _require(len(info["original"]) == 1 and len(info["fragments"]) == 2
             and int(info["surviving"].sum()) == 5, "topological surgery failed")
    # A cut may erase a separate object completely, without adding a letter.
    specimen[0, 2] = True
    info = _counter_details(specimen, cut, regions)
    _require(len(info["original"]) == 2 and len(info["fragments"]) == 2, "cut-only object survived")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(puzzle_hashes),
            "input_output_shapes": len(shapes), "glyph_shapes": len(glyph_shapes),
            "cut_geometries": len(cuts), "specimen_cut_layouts": len(layouts), "words": len(words),
            "fragment_counts": dict(sorted(counts.items())), "cut_families": dict(sorted(kinds.items())),
            "severed_pairs": severed_pairs, "staggered_pairs": staggered_pairs,
            "whole_object_erasure_pairs": disappeared, "marker_swap_checks": split_cases,
            "semantic_verification": "generator-owned checks; no independent or blind solver"}
