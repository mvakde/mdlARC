"""Reciprocal grafting: transport a marked cut into scaffold-aligned motif copies.

The framed header contains control / state 0 / state 1. The field underneath
contains copies to edit. References are data, not examples to edit themselves.
Only the standard library and NumPy are used; no parent code is imported.
"""

from math import gcd
from functools import reduce
import numpy as np


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(int, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _counts(a, b):
    for value, low, high in ((a, 2, 10), (b, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return int(a), int(b)


def _components(mask, diagonal=False):
    todo = {tuple(map(int, p)) for p in np.argwhere(mask)}
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in todo:
                    todo.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _mask(shape, cells):
    a = np.zeros(shape, dtype=bool)
    for p in cells:
        a[p] = True
    return a


def _bounds(mask):
    points = np.argwhere(mask)
    _require(len(points), "empty support")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _cut_info(cut):
    """A digital path: diagonal edges cannot shortcut an orthogonal elbow."""
    cells = {tuple(map(int, p)) for p in np.argwhere(cut)}
    graph = {}
    for r, c in cells:
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = r + dr, c + dc
                if (dr or dc) and q in cells:
                    if dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells):
                        continue
                    neighbors.append(q)
        graph[r, c] = neighbors
    ends = sorted(p for p, ns in graph.items() if len(ns) == 1)
    _require(len(cells) >= 3 and len(ends) == 2
             and all(1 <= len(ns) <= 2 for ns in graph.values()), "branch or loop in cut")
    h, w = cut.shape
    border = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _require(all(border(p) for p in ends)
             and all(not border(p) for p in cells - set(ends)), "invalid cut endpoints")
    order, previous, current = [], None, ends[0]
    while current is not None and current not in order:
        order.append(current)
        following = [p for p in graph[current] if p != previous]
        previous, current = current, following[0] if following else None
    _require(len(order) == len(cells), "disconnected cut")
    groups = _components(~cut)
    _require(len(groups) == 2, "cut must leave exactly two regions")
    near = cut.copy()
    for r, c in cells:
        near[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    regions = [_mask(cut.shape, g) for g in groups]
    sites = [[tuple(map(int, p)) for p in np.argwhere(region & ~near)] for region in regions]
    _require(all(sites), "both regions must admit an isolated marker")
    return regions, sites, order


def _sample_cut(rng, h, w, kind):
    for _ in range(160):
        cut = np.zeros((h, w), dtype=bool)
        if kind == "horizontal":
            cut[int(rng.integers(2, h - 2)), :] = True
        elif kind == "vertical":
            cut[:, int(rng.integers(2, w - 2))] = True
        elif kind == "diagonal":
            if h == w and rng.random() < .55:
                cut[np.arange(h), np.arange(w)] = True
            else:
                across = w > h
                hh, ww = (w, h) if across else (h, w)
                ends = rng.choice(np.arange(1, ww - 1), 2, replace=False)
                line = np.zeros((hh, ww), dtype=bool)
                line[np.arange(hh), np.rint(np.linspace(*ends, hh)).astype(int)] = True
                cut = line.T.copy() if across else line
        elif kind == "elbow":
            r, c = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
            cut[:r + 1, c] = True
            cut[r, c:] = True
        elif kind == "stepped":
            # Opposite-boundary path, two or four elbows, with clear run spacing.
            across = bool(rng.integers(2))
            hh, ww = (w, h) if across else (h, w)
            line = np.zeros((hh, ww), dtype=bool)
            rows = [2] if hh == 6 else ([2, 4] if rng.random() < .5 else [3])
            columns = list(range(2, ww - 2))
            c, previous = int(rng.choice(columns)), 0
            for r in rows:
                line[previous:r + 1, c] = True
                new = int(rng.choice([x for x in columns if x != c]))
                line[r, min(c, new):max(c, new) + 1] = True
                c, previous = new, r
            line[previous:, c] = True
            cut = line.T.copy() if across else line
        else:
            raise ValueError("unknown cut kind")
        if rng.integers(2):
            cut = cut[::-1].copy()
        if rng.integers(2):
            cut = cut[:, ::-1].copy()
        try:
            _cut_info(cut)
            return cut
        except AssertionError:
            pass
    raise AssertionError("bounded cut sampling exhausted")


def _primitive(crop):
    lengths = []
    for array in (crop, crop.T):
        for row in array:
            stops = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, len(row)]
            lengths.extend(map(int, np.diff(stops)))
    scale = reduce(gcd, lengths)
    small = crop[::scale, ::scale]
    _require(np.array_equal(crop, np.repeat(np.repeat(small, scale, 0), scale, 1)),
             "nonuniform pixel scale")
    return scale, small


def _sample_templates(rng, h, w, bg, colors):
    """Grow two connected colored designs around a shared procedural scaffold."""
    center = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
    core = {center}
    mode = int(rng.integers(4))
    r, c = center
    if mode == 0:
        core |= {(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                 if 0 < r + dr < h - 1 and 0 < c + dc < w - 1}
    elif mode == 1:
        core |= {(r + d, c + d) for d in (-1, 1)
                 if 0 < r + d < h - 1 and 0 < c + d < w - 1}
    elif mode == 2:
        core |= {(r + 1, c), (r + 1, c + 1)}
        core = {p for p in core if 0 < p[0] < h - 1 and 0 < p[1] < w - 1}
    desired = int(rng.integers(3, min(11, (h - 2) * (w - 2)) + 1))
    while len(core) < desired:
        frontier = sorted({(r + dr, c + dc) for r, c in core
                           for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1))
                           if 0 < r + dr < h - 1 and 0 < c + dc < w - 1} - core)
        core.add(frontier[int(rng.integers(len(frontier)))])
    stable = None
    if len(core) >= 5 and rng.random() < .4:
        stable = sorted(core)[int(rng.integers(len(core)))]
    templates = []
    accent_counts = [int(rng.integers(1, 3)), int(rng.integers(1, 3))]
    offset = 2
    for state in range(2):
        occupied = set(core)
        desired = int(rng.integers(len(core) + 3, max(len(core) + 4, int(h * w * .68)) + 1))
        while len(occupied) < desired:
            frontier = sorted({(r + dr, c + dc) for r, c in occupied
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if (dr or dc) and 0 <= r + dr < h and 0 <= c + dc < w} - occupied)
            occupied.add(frontier[int(rng.integers(len(frontier)))])
        template = np.full((h, w), bg, dtype=int)
        for p in core:
            template[p] = colors[0]
        if stable is not None:
            template[stable] = colors[1]
        accents = colors[offset:offset + accent_counts[state]]
        offset += accent_counts[state]
        for p in sorted(occupied - core):
            template[p] = accents[(p[0] + 2 * p[1]) % len(accents)]
        templates.append(template)
    return templates


def _flags(templates, cut, regions, bg):
    a, b = templates
    different = a != b
    return {
        "region0": bool(np.any(different & regions[0])),
        "region1": bool(np.any(different & regions[1])),
        "collision": bool(np.any(cut & (a != bg) & (b != bg) & different)),
        "only0": bool(np.any(cut & (a != bg) & (b == bg))),
        "only1": bool(np.any(cut & (a == bg) & (b != bg))),
        "blank": bool(np.any(cut & (a == bg) & (b == bg))),
        "erase0": bool(np.any(~cut & (a != bg) & (b == bg))),
        "erase1": bool(np.any(~cut & (a == bg) & (b != bg))),
        "shifted_boxes": _bounds(a != bg) != _bounds(b != bg),
        "shifted_origins": _bounds(a != bg)[:2] != _bounds(b != bg)[:2],
    }


def _separated(a, b):
    y, x, h, w = a
    yy, xx, hh, ww = b
    return y + h + 1 <= yy or yy + hh + 1 <= y or x + w + 1 <= xx or xx + ww + 1 <= x


def _place(rng, height, width, sizes):
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    for _ in range(48):
        rectangles = [None] * len(sizes)
        placed = []
        for i in order:
            h, w = sizes[i]
            candidates = [(y, x, h, w) for y in range(height - h + 1)
                          for x in range(width - w + 1)
                          if all(_separated((y, x, h, w), other) for other in placed)]
            if not candidates:
                break
            edge = [q for q in candidates if q[0] in (0, height - h) or q[1] in (0, width - w)]
            pool = edge if edge and rng.random() < .7 else candidates
            rect = pool[int(rng.integers(len(pool)))]
            rectangles[i] = rect
            placed.append(rect)
        else:
            return rectangles
    return None


def _layout(rng, h, w, union_box, first, scale_shift):
    y0, x0, y1, x1 = union_box
    sh, sw = y1 - y0, x1 - x0
    height_cap, width = 27 - h, 3 * w + 2
    for attempt in range(36):
        if first:
            scales = [1, 1, 2]
        elif scale_shift:
            scales = [1, 3] + [1] * int(rng.integers(0, 3))
        else:
            count = int(rng.integers(2, 7))
            scales = [1, 1] + [int(rng.choice([1, 1, 1, 2])) for _ in range(count - 2)]
        if attempt >= 24:
            scales = [1, 1, 2] if first else [1, 3] if scale_shift else [1, 1]
        sizes = [(sh * s, sw * s) for s in scales]
        minimum = max(a for a, b in sizes)
        if minimum > height_cap or max(b for a, b in sizes) > width:
            continue
        field_h = height_cap if attempt >= 18 else int(rng.integers(minimum, height_cap + 1))
        rectangles = _place(rng, field_h, width, sizes)
        if rectangles is not None:
            states = [0, 1] + [int(rng.random() < float(rng.choice([.2, .5, .8])))
                               for _ in scales[2:]]
            # No association of state with size or position.
            if rng.integers(2):
                states = [1 - s for s in states]
            placements = [(y - y0 * s, x - x0 * s, s, state)
                          for (y, x, _, _), s, state in zip(rectangles, scales, states)]
            return field_h, placements
    raise AssertionError("bounded packing exhausted")


def _enlarge(a, scale):
    return np.repeat(np.repeat(a, scale, 0), scale, 1)


def _put(field, template, anchor, scale, union_box):
    y0, x0, y1, x1 = union_box
    y, x = anchor[0] + y0 * scale, anchor[1] + x0 * scale
    field[y:y + (y1 - y0) * scale, x:x + (x1 - x0) * scale] = _enlarge(template[y0:y1, x0:x1], scale)


def _assemble(control, templates, field, frame):
    h, w = control.shape
    grid = np.full((h + field.shape[0] + 3, 3 * w + 4), frame, dtype=int)
    for i, panel in enumerate([control] + list(templates)):
        grid[1:h + 1, 1 + i * (w + 1):1 + i * (w + 1) + w] = panel
    grid[h + 2:-1, 1:-1] = field
    return grid


def _make_pair(templates, cut, marker, field_h, placements, frame, bg, control_color):
    regions, _, _ = _cut_info(cut)
    marked = regions[0] if regions[0][marker] else regions[1]
    control = np.full(cut.shape, bg, dtype=int)
    control[cut] = control_color
    control[marker] = control_color
    h, w = cut.shape
    union_box = _bounds((templates[0] != bg) | (templates[1] != bg))
    inp = np.full((field_h, 3 * w + 2), bg, dtype=int)
    out = inp.copy()
    # Ground truth uses the sampled placements and a scalar logical-cell rule;
    # solve below has to recover every state, scale and anchor from pixels.
    for y, x, scale, state in placements:
        _put(inp, templates[state], (y, x), scale, union_box)
        hybrid = templates[state].copy()
        for r in range(h):
            for c in range(w):
                opposite = int(templates[1 - state][r, c])
                if marked[r, c] or (cut[r, c] and opposite != bg):
                    hybrid[r, c] = opposite
        _put(out, hybrid, (y, x), scale, union_box)
    return {"input": _assemble(control, templates, inp, frame),
            "output": _assemble(control, templates, out, frame)}


def _new_pair(rng, kind, first=False, scale_shift=False):
    h, w = (5, 5) if scale_shift else (int(rng.integers(6, 8)), int(rng.integers(6, 8)))
    cut = _sample_cut(rng, h, w, kind)
    regions, sites, _ = _cut_info(cut)
    frame, bg, control_color, *colors = map(int, rng.permutation(10))
    for _ in range(1800 if first else 200):
        templates = _sample_templates(rng, h, w, bg, colors)
        flags = _flags(templates, cut, regions, bg)
        if not (flags["region0"] and flags["region1"]):
            continue
        if first and not all(flags.values()):
            continue
        boxes = [_bounds(t != bg) for t in templates]
        if any(_primitive(t[y0:y1, x0:x1])[0] != 1
               for t, (y0, x0, y1, x1) in zip(templates, boxes)):
            continue
        union_box = _bounds((templates[0] != bg) | (templates[1] != bg))
        field_h, placements = _layout(rng, h, w, union_box, first, scale_shift)
        side = int(rng.integers(2))
        marker = sites[side][int(rng.integers(len(sites[side])))]
        return _make_pair(templates, cut, marker, field_h, placements, frame, bg, control_color)
    raise AssertionError("bounded template sampling exhausted")


def _read(input_grid):
    grid = _grid(input_grid)
    height, width = grid.shape
    _require((width - 4) % 3 == 0, "header must have three equal panels")
    w = (width - 4) // 3
    frame = int(grid[0, 0])
    _require(np.all(grid[[0, -1], :] == frame) and np.all(grid[:, [0, -1]] == frame), "broken outer frame")
    separators = [r for r in range(1, height - 1) if np.all(grid[r] == frame)]
    _require(len(separators) == 1, "missing or ambiguous header separator")
    separator = separators[0]
    h = separator - 1
    _require(h >= 3 and w >= 3 and separator + 2 < height, "unreadable header/field")
    _require(np.all(grid[:separator + 1, ::w + 1] == frame), "broken header dividers")
    panels = [grid[1:separator, 1 + i * (w + 1):1 + i * (w + 1) + w] for i in range(3)]
    ctrl, a, b = panels
    values = list(map(int, np.unique(ctrl)))
    _require(len(values) == 2 and frame not in values, "control needs exactly two colors")
    candidates = []
    for bg in values:
        groups = sorted(_components(ctrl != bg, diagonal=True), key=len)
        if len(groups) != 2 or len(groups[0]) != 1 or len(groups[1]) < 3:
            continue
        marker = min(groups[0])
        cut = _mask(ctrl.shape, groups[1])
        try:
            regions, sites, path = _cut_info(cut)
            side = 0 if regions[0][marker] else 1
            if marker not in sites[side]:
                continue
            candidates.append((bg, marker, cut, regions, sites, path, side))
        except AssertionError:
            continue
    _require(len(candidates) == 1, "ambiguous control/background")
    bg, marker, cut, regions, sites, path, side = candidates[0]
    control_color = next(c for c in values if c != bg)
    templates = [a, b]
    for t in templates:
        _require(not np.any(np.isin(t, [frame, control_color])) and bg in t, "unreadable reference palette")
        _require(len(_components(t != bg, diagonal=True)) == 1, "reference motif must be eight-connected")
    colors = set(map(int, np.unique(a))) & set(map(int, np.unique(b))) - {bg}
    carriers = [c for c in sorted(colors) if np.array_equal(a == c, b == c) and np.sum(a == c) >= 3]
    _require(len(carriers) == 1, "need one shared carrier color with at least three cells")
    carrier = carriers[0]
    carrier_y, carrier_x, _, _ = _bounds(a == carrier)
    boxes = [_bounds(t != bg) for t in templates]
    crops = [t[y0:y1, x0:x1] for t, (y0, x0, y1, x1) in zip(templates, boxes)]
    _require(not np.array_equal(crops[0], crops[1]), "states cannot be translation-equivalent")
    _require(all(_primitive(c)[0] == 1 for c in crops), "references must expose primitive pixels")
    union_box = _bounds((a != bg) | (b != bg))
    field = grid[separator + 1:-1, 1:-1]
    _require(not np.any(np.isin(field, [frame, control_color])), "frame/control color in field")
    objects, rectangles = [], []
    for group in _components(field != bg, diagonal=True):
        y0, x0, y1, x1 = _bounds(_mask(field.shape, group))
        crop = field[y0:y1, x0:x1]
        _require(np.count_nonzero(crop != bg) == len(group), "interleaved motifs")
        scale, small = _primitive(crop)
        matches = [state for state in (0, 1) if np.array_equal(small, crops[state])]
        _require(len(matches) == 1, "target is not a unique scaled reference state")
        state = matches[0]
        observed_y, observed_x, _, _ = _bounds(crop == carrier)
        anchor = (y0 + observed_y - carrier_y * scale,
                  x0 + observed_x - carrier_x * scale)
        by, bx, _, _ = boxes[state]
        _require(anchor == (y0 - by * scale, x0 - bx * scale), "carrier/state registration disagreement")
        uy0, ux0, uy1, ux1 = union_box
        rect = (anchor[0] + uy0 * scale, anchor[1] + ux0 * scale,
                (uy1 - uy0) * scale, (ux1 - ux0) * scale)
        ry, rx, rh, rw = rect
        _require(0 <= ry and ry + rh <= field.shape[0] and 0 <= rx and rx + rw <= field.shape[1], "replacement clipped")
        _require(all(_separated(rect, other) for other in rectangles), "reserved union rectangles overlap")
        rectangles.append(rect)
        objects.append({"anchor": anchor, "scale": scale, "state": state, "rect": rect})
    _require(len(objects) >= 2, "need at least two field motifs")
    return {"grid": grid, "separator": separator, "panels": panels, "field": field,
            "templates": templates, "background": bg, "frame": frame, "control_color": control_color,
            "cut": cut, "regions": regions, "marked": regions[side], "sites": sites,
            "side": side, "marker": marker, "path": path, "carrier": carrier,
            "objects": objects, "union_box": union_box, "flags": _flags(templates, cut, regions, bg)}


def _prediction(info, mode="graft"):
    result = info["grid"].copy()
    field = result[info["separator"] + 1:-1, 1:-1]
    a, b = info["templates"]
    bg, cut, marked = info["background"], info["cut"], info["marked"]
    for obj in info["objects"]:
        state = obj["state"]
        current, opposite = (a, b) if state == 0 else (b, a)
        if mode == "bbox_aligned":
            # Deliberately wrong alternative: align occupied corners instead
            # of the common carrier. Used only as a training-evidence probe.
            cy, cx, _, _ = _bounds(current != bg)
            oy, ox, _, _ = _bounds(opposite != bg)
            dy, dx = cy - oy, cx - ox
            shifted = np.full(opposite.shape, bg, dtype=int)
            for r, c in np.ndindex(opposite.shape):
                rr, cc = r + dy, c + dx
                if 0 <= rr < shifted.shape[0] and 0 <= cc < shifted.shape[1]:
                    shifted[rr, cc] = opposite[r, c]
            opposite = shifted
        if mode == "whole_exchange":
            hybrid = opposite
        elif mode == "identity":
            hybrid = current
        else:
            hybrid = np.where(marked, opposite, current)
            seam = np.where(opposite != bg, opposite, current)
            if mode == "current_priority":
                seam = np.where(current != bg, current, opposite)
            elif mode == "opposite_only_seam":
                seam = opposite
            elif mode == "fixed_zero":
                hybrid = np.where(marked, a, current)
                seam = np.where(a != bg, a, current)
            elif mode == "keep_erased":
                hybrid = np.where(marked & (opposite != bg), opposite, current)
            hybrid = np.where(cut, seam, hybrid)
        _put(field, hybrid, obj["anchor"], obj["scale"], info["union_box"])
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver: header parsing, scale normalization, registered grafting."""
    return _prediction(_read(input_grid))


def _opposite_marker(pair, rng):
    info = _read(pair["input"])
    g = pair["input"].copy()
    panel = g[1:info["separator"], 1:info["panels"][0].shape[1] + 1]
    panel[info["marker"]] = info["background"]
    sites = info["sites"][1 - info["side"]]
    panel[sites[int(rng.integers(len(sites)))]] = info["control_color"]
    # Preserve independent scalar ground truth for the controlled twin too.
    field_h = info["field"].shape[0]
    new_marker = tuple(map(int, np.argwhere((panel == info["control_color"]) & ~info["cut"])[0]))
    placements = [(*o["anchor"], o["scale"], o["state"]) for o in info["objects"]]
    return _make_pair(info["templates"], info["cut"], new_marker, field_h, placements,
                      info["frame"], info["background"], info["control_color"])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    # Sampling regimes are not solution parameters. All semantics are visible.
    regime = str(rng.choice(["cut", "scale"]))
    first_kind = str(rng.choice(["horizontal", "vertical", "diagonal"]))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            if split == "train" and index == 1:
                pair = _opposite_marker(puzzle["train"][0], rng)
            else:
                first = split == "train" and index == 0
                scale_shift = split == "test" and regime == "scale"
                kind = (first_kind if first or scale_shift else
                        str(rng.choice(["elbow", "stepped"])) if split == "test" else
                        str(rng.choice(["horizontal", "vertical", "diagonal", "elbow"])))
                for _ in range(24):
                    pair = _new_pair(rng, kind, first, scale_shift)
                    if _key(pair["input"]) not in seen:
                        break
                else:
                    raise AssertionError("unique input construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _read(inp)
            _require(inp.shape == out.shape and np.array_equal(_prediction(info), out), "incorrect graft or changed context")
            _require(info["flags"]["region0"] and info["flags"]["region1"], "mask must make a meaningful partial exchange")
            _require({o["state"] for o in info["objects"]} == {0, 1}, "field must show both directions")
            _require(not np.array_equal(inp, out), "identity example")
            # A's omission swaps the whole motif; B's omission leaves it alone.
            _require(not np.array_equal(_prediction(info, "whole_exchange"), out), "A concept is ineffective")
            _require(not np.array_equal(_prediction(info, "identity"), out), "B concept is ineffective")
            common = (info["templates"][0] == info["templates"][1]) & (info["templates"][0] != info["background"])
            before, after = inp[info["separator"] + 1:-1, 1:-1], out[info["separator"] + 1:-1, 1:-1]
            for obj in info["objects"]:
                for r, c in np.argwhere(common):
                    y, x = obj["anchor"][0] + int(r) * obj["scale"], obj["anchor"][1] + int(c) * obj["scale"]
                    s = obj["scale"]
                    _require(np.array_equal(before[y:y + s, x:x + s], after[y:y + s, x:x + s]), "scaffold moved")
            if split == "train":
                training.append(info)
    _require(all(any(d["flags"][name] for d in training) for name in training[0]["flags"]), "training evidence incomplete")
    _require({d["side"] for d in training} == {0, 1}, "training must expose marker reversal")
    _require(any(any(o["scale"] >= 2 for o in d["objects"]) for d in training), "training omits scaling")
    for mode in ("whole_exchange", "identity", "current_priority", "opposite_only_seam", "fixed_zero", "keep_erased", "bbox_aligned"):
        _require(any(not np.array_equal(_prediction(d, mode), puzzle["train"][i]["output"])
                     for i, d in enumerate(training)), "training admits shortcut: " + mode)


def _reflect_pair(pair, axis):
    info = _read(pair["input"])
    result = {}
    h, w = info["panels"][0].shape
    for role in ("input", "output"):
        g = _grid(pair[role])
        panels = [np.flip(g[1:h + 1, 1 + i * (w + 1):1 + i * (w + 1) + w], axis) for i in range(3)]
        field = np.flip(g[h + 2:-1, 1:-1], axis)
        result[role] = _assemble(panels[0], panels[1:], field, info["frame"])
    return result


def self_test():
    """Bounded structural, semantic, counterfactual and diversity checks: 42 seeds."""
    shapes, layouts, designs, cuts, palettes, scales, counts, puzzles = set(), set(), set(), set(), set(), set(), set(), set()
    modes_seen, expansion, changed_cells = set(), 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        again = generate_puzzle(seed)
        signature = tuple(_key(pair[k]) for split in ("train", "test") for pair in p[split] for k in ("input", "output"))
        _require(signature == tuple(_key(pair[k]) for split in ("train", "test") for pair in again[split] for k in ("input", "output")), "nondeterministic puzzle")
        puzzles.add(signature)
        for split in ("train", "test"):
            for pair in p[split]:
                d = _read(pair["input"])
                _require(np.array_equal(solve(pair["input"].copy()), pair["output"]), "solver disagrees")
                shapes.add(pair["input"].shape)
                cuts.add((d["cut"].shape, d["cut"].tobytes()))
                roles = {}
                normalized = tuple(tuple(roles.setdefault(int(c), len(roles)) for c in t.flat) for t in d["templates"])
                designs.add((d["cut"].shape, normalized))
                layouts.add(tuple((o["anchor"], o["scale"], o["state"]) for o in d["objects"]))
                palettes.add(tuple(map(int, np.unique(pair["input"]))))
                schedule = tuple(sorted(o["scale"] for o in d["objects"]))
                scales.add(schedule)
                counts.add(len(schedule))
                modes_seen.update(schedule)
                changed_cells += int(np.count_nonzero(pair["input"] != pair["output"]))
                field_out = pair["output"][d["separator"] + 1:-1, 1:-1]
                for o in d["objects"]:
                    y, x, hh, ww = o["rect"]
                    before = d["field"][y:y + hh, x:x + ww] != d["background"]
                    after = field_out[y:y + hh, x:x + ww] != d["background"]
                    expansion += bool(np.any(after & ~before))
        first, second = p["train"][:2]
        _require(np.count_nonzero(first["input"] != second["input"]) == 2, "controlled marker twin changed content")
        _require(not np.array_equal(first["output"], second["output"]), "marker has no effect")
        d, e = _read(first["input"]), _read(second["input"])
        # Inspect actual output blocks, rather than comparing two rule formulas.
        for obj in d["objects"]:
            state, scale = obj["state"], obj["scale"]
            current, other = d["templates"][state], d["templates"][1 - state]
            fields = [pair["output"][d["separator"] + 1:-1, 1:-1] for pair in (first, second)]
            uy0, ux0, uy1, ux1 = d["union_box"]
            inside_union = np.zeros(d["cut"].shape, dtype=bool)
            inside_union[uy0:uy1, ux0:ux1] = True
            for r, c in np.argwhere((d["cut"] | (current != other)) & inside_union):
                y, x = obj["anchor"][0] + int(r) * scale, obj["anchor"][1] + int(c) * scale
                if d["cut"][r, c]:
                    value = other[r, c] if other[r, c] != d["background"] else current[r, c]
                    _require(all(np.all(f[y:y + scale, x:x + scale] == value) for f in fields), "wrong seam or marker-dependent priority")
                else:
                    expected = [other[r, c], current[r, c]] if d["marked"][r, c] else [current[r, c], other[r, c]]
                    _require(all(np.all(f[y:y + scale, x:x + scale] == value)
                                 for f, value in zip(fields, expected)), "marker twin did not reverse strict regions")
        for axis in (0, 1):
            reflected = _reflect_pair(first, axis)
            _require(np.array_equal(solve(reflected["input"]), reflected["output"]), "reflection covariance failed")
        permutation = np.random.default_rng(seed + 10000).permutation(10)
        _require(np.array_equal(solve(permutation[first["input"]]), permutation[first["output"]]), "literal-color shortcut")
    _require(len(puzzles) == 42 and len(designs) >= 100 and len(cuts) >= 60, "structural diversity collapsed")
    _require(len(shapes) >= 18 and len(layouts) >= 120 and len(palettes) >= 25, "layout/palette diversity collapsed")
    _require(modes_seen == {1, 2, 3} and counts >= {2, 3, 4, 5, 6}, "scale/count variation missing")
    _require(len(scales) >= 10 and expansion >= 100, "scale schedules or support growth missing")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(9100 + a * 10 + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    sample = generate_puzzle(17)
    for point in ((0, 0), tuple(np.argwhere(sample["test"][0]["input"] != sample["test"][0]["output"])[0])):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in sample[s]] for s in sample}
        g = bad["test"][0]["output"]
        g[point] = (int(g[point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError("corrupted output accepted")
    d = _read(sample["train"][0]["input"])
    malformed = d["grid"].copy()
    malformed[1 + d["marker"][0], 1 + d["marker"][1]] = d["background"]
    try:
        solve(malformed)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing marker accepted")
    malformed = d["grid"].copy()
    large = next(o for o in d["objects"] if o["scale"] == 2)
    r, c = np.argwhere(d["templates"][large["state"]] == d["carrier"])[0]
    y = d["separator"] + 1 + large["anchor"][0] + int(r) * 2
    x = 1 + large["anchor"][1] + int(c) * 2
    malformed[y, x] = d["background"]
    try:
        solve(malformed)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged scale block accepted")
    return {"seeds": 42, "distinct_puzzles": len(puzzles), "template_designs": len(designs),
            "cuts": len(cuts), "canvas_shapes": len(shapes), "layouts": len(layouts),
            "palettes": len(palettes), "scale_schedules": len(scales),
            "target_counts": sorted(counts), "support_growth_occurrences": expansion,
            "changed_cells": changed_cells}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
