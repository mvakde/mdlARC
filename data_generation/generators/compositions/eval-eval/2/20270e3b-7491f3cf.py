"""Registered seam sheets: registration constructs the regional compositor.

Only Python's standard library and NumPy are used. No parent code is imported.
The parser and solver recover all roles and translations from input pixels.
"""

from __future__ import annotations

import json

import numpy as np


HOST_SEEDS = (0, 743711410, 393735176, 981909223, 507044670, 558949687,
              424703447, 9545667, 864335840, 66514002, 143033085, 45748050,
              409000051, 306012886, 540432059, 275007501, 241125266,
              617171249, 23300403, 91144323, 603584846, 123184953, 784299742,
              267013240, 60107900, 71152593, 3189760, 111553039, 822301278,
              728980860, 505973270, 232419145, 783796221, 429193687,
              464355743, 676253360, 572373911, 295667259, 979671462,
              474219355, 135888895, 963378519)


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grids must have two dimensions in 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(np.all((0 <= a) & (a <= 9)), "grid colors must be in 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.ascontiguousarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(cells, diagonal=False):
    remaining = set(cells)
    directions = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or not (r and c))]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _cells(a, color):
    return {tuple(map(int, p)) for p in np.argwhere(a == color)}


def _bbox(cells):
    _require(bool(cells), "empty cell set")
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _perimeter(t, b, l, r):
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)
            if y in (t, b) or x in (l, r)}


def _shift(cells, dr, dc):
    return {(r + dr, c + dc) for r, c in cells}


def _path_ok(cells):
    """A digital path; diagonal shortcuts across an orthogonal elbow are omitted."""
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
    ends = [p for p, ns in graph.items() if len(ns) == 1]
    return (len(cells) >= 3 and len(ends) == 2
            and all(1 <= len(ns) <= 2 for ns in graph.values())
            and len(_components(cells, True)) == 1)


def _path_degree(cells, point):
    r, c = point
    return sum(1 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
               if (dr or dc) and (r + dr, c + dc) in cells
               and not (dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells)))


def _extract(input_grid):
    """Find frames, shared control roles, donor roles, and the two backing sides."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(np.all(a[0, :] == bg) and np.all(a[-1, :] == bg)
             and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg),
             "a blank exterior border is required")
    frames = []
    for color in map(int, np.unique(a)):
        if color == bg:
            continue
        comps = _components(_cells(a, color))
        if len(comps) != 2:
            continue
        boxes = [_bbox(c) for c in comps]
        if all(b - t >= 4 and r - l >= 4 and comp == _perimeter(t, b, l, r)
               for comp, (t, b, l, r) in zip(comps, boxes)):
            frames.append((color, boxes))
    _require(len(frames) == 1, "two uniquely identifiable rectangular sheet frames required")
    frame, boxes = frames[0]
    sheet_cells = set()
    cards = []
    for t, b, l, r in boxes:
        box = {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}
        _require(sheet_cells.isdisjoint(box), "sheets must be disjoint before assembly")
        sheet_cells |= box
        cards.append(a[t + 1:b, l + 1:r].copy())
    _require(all(a[y, x] == bg for y in range(a.shape[0]) for x in range(a.shape[1])
                 if (y, x) not in sheet_cells), "no pixels outside the two sheets")
    sets = [set(map(int, np.unique(card))) - {bg} for card in cards]
    shared = sets[0] & sets[1]
    _require(len(shared) == 2 and all(len(s) == 3 for s in sets),
             "each sheet needs shared cut/guide colors and its own pigment")
    _require(frame not in set.union(*sets), "frame color cannot occur inside a sheet")
    candidates = []
    for guide in sorted(shared):
        control = next(c for c in shared if c != guide)
        for axis in (0, 1):
            oriented = [card if axis == 0 else card.T for card in cards]
            records = []
            length = None
            for card in oriented:
                bars = _cells(card, guide)
                t, b, l, r = _bbox(bars)
                if t != b or bars != {(t, x) for x in range(l, r + 1)}:
                    break
                if length is not None and length != len(bars):
                    break
                length = len(bars)
                parts = _components(_cells(card, control), True)
                paths = [p for p in parts if len(p) >= 3]
                dots = [p for p in parts if len(p) == 1]
                if len(paths) != 1 or len(parts) != len(paths) + len(dots) or len(dots) > 1:
                    break
                path = paths[0]
                if not _path_ok(path):
                    break
                negative = any((t - 1, x) in path for x in range(l, r + 1))
                positive = any((t + 1, x) in path for x in range(l, r + 1))
                if negative == positive:
                    break
                sign = -1 if negative else 1
                if not all((y - t) * sign > 0 for y, x in path):
                    break
                h, w = card.shape
                edge = 0 if sign == -1 else h - 1
                boundary = [p for p in path if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
                if len(boundary) != 1 or boundary[0][0] != edge:
                    break
                contacts = [(t + sign, x) for x in range(l, r + 1) if (t + sign, x) in path]
                if (len(contacts) != 1 or _path_degree(path, contacts[0]) != 1
                        or _path_degree(path, boundary[0]) != 1
                        or (t + 2 * sign, contacts[0][1]) not in path):
                    break
                pigment = next(c for c in set(map(int, np.unique(card))) - {bg, guide, control})
                marker = next(iter(dots[0])) if dots else None
                records.append({"card": card, "bar": (t, l), "sign": sign,
                                "path": path, "marker": marker, "pigment": pigment})
            if (len(records) == 2 and records[0]["sign"] != records[1]["sign"]
                    and sum(rec["marker"] is not None for rec in records) == 1):
                records.sort(key=lambda rec: rec["sign"])
                candidates.append({"background": bg, "frame": frame, "control": control,
                                   "guide": guide, "length": length, "axis": axis,
                                   "records": records, "boxes": boxes})
    _require(len(candidates) == 1, "guide axis and backing sides must be unambiguous")
    return candidates[0]


def _register(d, seam=-1, tangent=0):
    """Register above-backed bar at row 0; below-backed bar goes to row -1."""
    bounds, controls, pigments, markers = [], [], [], []
    bg = d["background"]
    for index, rec in enumerate(d["records"]):
        card = rec["card"]
        row, col = rec["bar"]
        dr = -row + (seam if index else 0)
        dc = -col + (tangent if index else 0)
        h, w = card.shape
        bounds.append((dr, dr + h - 1, dc, dc + w - 1))
        bar = {(row, col + x) for x in range(d["length"])}
        controls.append(_shift(rec["path"] | bar, dr, dc))
        pigments.append(_shift(_cells(card, rec["pigment"]), dr, dc))
        markers.append(None if rec["marker"] is None else
                       (rec["marker"][0] + dr, rec["marker"][1] + dc))
    t = min(b[0] for b in bounds)
    b = max(b[1] for b in bounds)
    l = min(bd[2] for bd in bounds)
    r = max(bd[3] for bd in bounds)
    _require(b - t + 1 <= 30 and r - l + 1 <= 30, "assembled domain too large")
    cut = set.union(*controls)
    domain = {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}
    _require(cut <= domain, "cut lies outside sheet extent")
    parts = _components(domain - cut)
    _require(len(parts) == 2, "registered cut must leave exactly two regions")
    # The controls themselves need not be a thin path: paired bars form a thick seam.
    _require(len(_components(cut, True)) == 1, "cut halves must join")
    preferred = next(i for i, m in enumerate(markers) if m is not None)
    marker = markers[preferred]
    _require(all(max(abs(marker[0] - y), abs(marker[1] - x)) >= 2 for y, x in cut),
             "the registered selector must be separate from the whole cut")
    chosen = next((p for p in parts if marker in p), None)
    _require(chosen is not None, "selector is on the cut")
    other = next(p for p in parts if p is not chosen)
    return {"cut": cut, "regions": parts, "marked": chosen, "other": other,
            "pigments": pigments, "controls": controls, "bounds": bounds,
            "preferred": preferred, "marker": marker, "bbox": (t, b, l, r),
            "domain": domain, "background": bg}


def _visible(d, reg, mode="regional"):
    p, q = reg["preferred"], 1 - reg["preferred"]
    first, second = reg["pigments"][p], reg["pigments"][q]
    if mode == "union":
        # A plausible shortcut: preferred pigment wins every overlap.
        return {cell: d["records"][p]["pigment"] if cell in first else d["records"][q]["pigment"]
                for cell in first | second}
    if mode == "reverse":
        first, second = second, first
        p, q = q, p
    visible = {cell: d["records"][p]["pigment"] for cell in first & reg["marked"]}
    visible.update({cell: d["records"][q]["pigment"] for cell in second & reg["other"]})
    # Cut and guide controls are transparent. At least one donor has a control at
    # each cut coordinate, so the other donor is the only possible ordinary pixel.
    for index in range(2):
        for cell in reg["pigments"][index] & reg["cut"]:
            visible[cell] = d["records"][index]["pigment"]
    return visible


def _render(d, visible):
    _require(bool(visible), "empty composites are outside the generated family")
    t, b, l, r = _bbox(visible)
    out = np.full((b - t + 1, r - l + 1), d["background"], dtype=np.int8)
    for (y, x), color in visible.items():
        out[y - t, x - l] = color
    return out.T.copy() if d["axis"] else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive registration, reconstructed regions and the crop from input alone."""
    d = _extract(input_grid)
    return _render(d, _visible(d, _register(d)))


def _motif(rng, h, w, style):
    rr, cc = np.indices((h, w))
    mask = np.zeros((h, w), dtype=bool)
    if style in ("solid", "ring", "u"):
        y0, y1 = sorted(map(int, rng.choice(h, 2, replace=False)))
        x0, x1 = sorted(map(int, rng.choice(w, 2, replace=False)))
        mask[y0:y1 + 1, x0:x1 + 1] = True
        if style != "solid":
            mask[y0 + 1:y1, x0 + 1:x1] = False
        if style == "u":
            mask[y0, x0 + 1:x1] = False
    elif style == "cross":
        y, x = int(rng.integers(h)), int(rng.integers(w))
        mask[y, :] = True
        mask[:, x] = True
    elif style == "diagonal":
        columns = np.rint(np.linspace(0, w - 1, h)).astype(int)
        mask[np.arange(h), columns] = True
        if rng.random() < .5:
            mask[np.arange(h), w - 1 - columns] = True
    elif style == "diamond":
        radius = int(rng.integers(1, max(2, min(h, w) // 2)))
        distance = abs(rr - h // 2) + abs(cc - w // 2)
        mask = distance == radius if rng.random() < .5 else distance <= radius
    elif style == "stripes":
        period = int(rng.integers(2, 4))
        axis = rr if rng.random() < .5 else cc
        mask = (axis + int(rng.integers(period))) % period == 0
    elif style == "checker":
        mask = (rr + cc + int(rng.integers(3))) % int(rng.integers(2, 4)) == 0
    elif style == "branch":
        x = int(rng.integers(w))
        mask[:, x] = True
        for y in range(0, h, int(rng.integers(2, 4))):
            lo, hi = sorted(map(int, rng.choice(w, 2, replace=False)))
            mask[y, lo:hi + 1] = True
    elif style == "blocks":
        for _ in range(int(rng.integers(2, 5))):
            y, x = int(rng.integers(h)), int(rng.integers(w))
            mask[y:min(h, y + int(rng.integers(1, 4))),
                 x:min(w, x + int(rng.integers(1, 4)))] = True
    else:
        mask = rng.random((h, w)) < float(rng.uniform(.12, .6))
        if style == "symmetric":
            mask |= mask[:, ::-1]
    return mask


STYLES = ("solid", "ring", "u", "cross", "diagonal", "diamond", "stripes",
          "checker", "branch", "blocks", "symmetric", "irregular")


def _half_path(rng, depth, width, attach, turns):
    """Top edge to bar backing row, ending with a straight normal approach."""
    if turns == 0:
        return {(y, attach) for y in range(depth)}
    rows = [1] if turns == 1 else [1, 3]
    _require(rows[-1] <= depth - 3, "insufficient depth for requested bends")
    cols = [attach]
    for _ in rows:
        choices = [x for x in range(2, width - 2) if x != cols[0]]
        cols.insert(0, choices[int(rng.integers(len(choices)))])
    path, last = set(), 0
    for index, row in enumerate(rows):
        col, new = cols[index:index + 2]
        path.update((y, col) for y in range(last, row + 1))
        path.update((row, x) for x in range(min(col, new), max(col, new) + 1))
        last = row
    path.update((y, attach) for y in range(last, depth))
    return path


def _geometry(rng, length, scale, bends):
    cards, bars = [], []
    for i in range(2):
        w = int(rng.integers(7, 10) if scale == "small" else rng.integers(9, 12)
                if scale == "large" else rng.integers(7, 12))
        n = int(rng.integers(4, 6) if scale == "small" else rng.integers(6, 8)
                if scale == "large" else rng.integers(4, 8))
        n = max(n, 6 if bends == 2 else 4)
        tail = int(rng.integers(1, 3))
        offset = int(rng.integers(2, w - length - 1))
        attach = offset + int(rng.integers(length))
        path = _half_path(rng, n, w, attach, bends)
        if i == 0:
            bar_row = n
        else:
            path = {(n + tail - y, x) for y, x in path}
            bar_row = tail
        shape = n + tail + 1, w
        cards.append({"shape": shape, "path": path, "bar": (bar_row, offset)})
        bars.append((bar_row, offset))
    return cards


def _frame_card(card, frame):
    h, w = card.shape
    result = np.full((h + 2, w + 2), frame, dtype=np.int8)
    result[1:-1, 1:-1] = card
    return result


def _stage(rng, cards, frame, bg, staging):
    cards = [_frame_card(card, frame) for card in cards]
    if rng.integers(2):
        cards.reverse()
    (h0, w0), (h1, w1) = [card.shape for card in cards]
    gap = int(rng.integers(1, 4))
    if staging == "horizontal":
        dy, dx = int(rng.integers(-h1 + 1, h0)), w0 + gap
    elif staging == "vertical":
        dy, dx = h0 + gap, int(rng.integers(-w1 + 1, w0))
    else:
        dy, dx = h0 + gap, w0 + gap
    t, b, l, r = min(0, dy), max(h0, dy + h1), min(0, dx), max(w0, dx + w1)
    h, w = b - t, r - l
    if max(h, w) > 28:
        return None
    extra_y, extra_x = 30 - h - 2, 30 - w - 2
    pads = [1 + int(rng.integers(min(3, extra_y) + 1)), 1,
            1 + int(rng.integers(min(3, extra_x) + 1)), 1]
    pads[1] += int(rng.integers(min(3, 30 - h - pads[0] - 1) + 1))
    pads[3] += int(rng.integers(min(3, 30 - w - pads[2] - 1) + 1))
    result = np.full((h + pads[0] + pads[1], w + pads[2] + pads[3]), bg, dtype=np.int8)
    for card, (y, x) in zip(cards, ((-t + pads[0], -l + pads[2]),
                                 (dy - t + pads[0], dx - l + pads[2]))):
        result[y:y + card.shape[0], x:x + card.shape[1]] = card
    return result


def _construct(rng, profile, teaching=False):
    bg, frame, control, guide, ca, cb = profile["palette"]
    length = profile["length"]
    for attempt in range(360):
        geometry = _geometry(rng, length, profile["scale"], profile["bends"])
        cards = []
        for i, geom in enumerate(geometry):
            h, w = geom["shape"]
            style = str(rng.choice(STYLES))
            mask = _motif(rng, h, w, style)
            card = np.full((h, w), bg, dtype=np.int8)
            card[mask] = (ca, cb)[i]
            for p in geom["path"]:
                card[p] = control
            row, col = geom["bar"]
            card[row, col:col + length] = guide
            if (ca, cb)[i] not in card:
                break
            cards.append(card)
        if len(cards) != 2:
            continue
        # Build the assembled coordinates from visible geometry before choosing a
        # selector. The selector will occupy an existing blank donor cell.
        records = [{"card": card, "bar": geom["bar"], "sign": -1 if i == 0 else 1,
                    "path": geom["path"], "marker": None, "pigment": (ca, cb)[i]}
                   for i, (card, geom) in enumerate(zip(cards, geometry))]
        d = {"background": bg, "frame": frame, "control": control, "guide": guide,
             "length": length, "axis": 0, "records": records}
        carrier = profile["carrier"]
        row, col = records[carrier]["bar"]
        dr, dc = -row - carrier, -col
        controls = [_shift(rec["path"] | {(rec["bar"][0], rec["bar"][1] + x)
                                         for x in range(length)},
                           -rec["bar"][0] - i, -rec["bar"][1])
                    for i, rec in enumerate(records)]
        cut = set.union(*controls)
        bounds = [(-rec["bar"][0] - i, rec["card"].shape[0] - rec["bar"][0] - i - 1,
                   -rec["bar"][1], rec["card"].shape[1] - rec["bar"][1] - 1)
                  for i, rec in enumerate(records)]
        t, b = min(q[0] for q in bounds), max(q[1] for q in bounds)
        l, r = min(q[2] for q in bounds), max(q[3] for q in bounds)
        regions = _components({(y, x) for y in range(t, b + 1) for x in range(l, r + 1)} - cut)
        if len(regions) != 2:
            continue
        sites = []
        for region in regions:
            candidates = []
            for y, x in sorted(region):
                ly, lx = y - dr, x - dc
                if (0 <= ly < cards[carrier].shape[0] and 0 <= lx < cards[carrier].shape[1]
                        and cards[carrier][ly, lx] == bg
                        and all(max(abs(y - cy), abs(x - cx)) >= 2 for cy, cx in cut)):
                    candidates.append((ly, lx))
            sites.append(candidates)
        if not all(sites):
            continue
        markers = [s[int(rng.integers(len(s)))] for s in sites]
        chosen = profile["side"]
        records[carrier]["marker"] = markers[chosen]
        cards[carrier][markers[chosen]] = control
        reg = _register(d)
        visible = _visible(d, reg)
        if not visible or np.array_equal(_render(d, visible), _render(d, _visible(d, reg, "union"))):
            continue
        if teaching:
            flags = _evidence(d, reg, visible)
            if not all(flags.values()):
                continue
            # Unequal offsets defeat aligning the card corners or centers.
            if records[0]["bar"][1] == records[1]["bar"][1]:
                continue
            if not _registration_evidence(d, _render(d, visible)):
                continue
        inp = _stage(rng, cards, frame, bg, profile["staging"])
        if inp is None:
            continue
        rotation = profile["rotation"]
        output = _render(d, visible)
        pair = {"input": np.rot90(inp, rotation).copy(),
                "output": np.rot90(output, rotation).copy()}
        try:
            parsed = _extract(pair["input"])
            assembled = _register(parsed)
        except ValueError:
            continue
        _require(np.array_equal(_render(parsed, _visible(parsed, assembled)), pair["output"]),
                 "constructed coordinates disagree with pixel registration")
        if teaching and not _registration_evidence(parsed, pair["output"]):
            continue
        if teaching and np.array_equal(_opposite_marker(pair)["output"], pair["output"]):
            continue
        return pair
    raise RuntimeError("bounded sheet construction exhausted")


def _opposite_marker(pair):
    d = _extract(pair["input"])
    reg = _register(d)
    carrier = reg["preferred"]
    rec = d["records"][carrier]
    row, col = rec["bar"]
    dr, dc = -row - carrier, -col
    options = []
    for y, x in sorted(reg["other"]):
        ly, lx = y - dr, x - dc
        if (0 <= ly < rec["card"].shape[0] and 0 <= lx < rec["card"].shape[1]
                and rec["card"][ly, lx] == d["background"]
                and all(max(abs(y - cy), abs(x - cx)) >= 2 for cy, cx in reg["cut"])):
            options.append((ly, lx))
    _require(bool(options), "no opposite selector site")
    # Locate the carrier in the un-oriented staged grid by its unique pigment.
    inp = pair["input"].copy()
    native_card = rec["card"] if d["axis"] == 0 else rec["card"].T
    carrier_box = next(box for box in d["boxes"]
                       if np.array_equal(inp[box[0] + 1:box[1], box[2] + 1:box[3]], native_card))
    t, _, l, _ = carrier_box
    old, new = rec["marker"], options[0]
    if d["axis"]:
        old, new = old[::-1], new[::-1]
    inp[t + 1 + old[0], l + 1 + old[1]] = d["background"]
    inp[t + 1 + new[0], l + 1 + new[1]] = d["control"]
    return {"input": inp, "output": solve(inp)}


def _evidence(d, reg, visible):
    p, q = reg["preferred"], 1 - reg["preferred"]
    a, b = reg["pigments"][p], reg["pigments"][q]
    m, o, cut = reg["marked"], reg["other"], reg["cut"]
    guides = {(y, x) for y in (-1, 0) for x in range(d["length"])}
    vt, vb, vl, vr = _bbox(visible)
    inside = {(y, x) for y in range(vt, vb + 1) for x in range(vl, vr + 1)}
    return {"marked_copies": bool(a & m), "other_copies": bool(b & o),
            "marked_suppresses": bool((b - a) & m),
            "other_suppresses": bool((a - b) & o),
            "guide_transparent": bool(guides & (a | b)),
            "guide_blank": bool((guides - a - b) & inside),
            "path_transparent": bool((cut - guides) & (a | b)),
            "cut_blank": bool((cut - a - b) & inside)}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.choice(10, 6, replace=False)))
    change_palette = bool(rng.integers(2))
    rotation = int(rng.integers(4))
    mixed_axis = bool(rng.integers(2))
    carrier = int(rng.integers(2))
    side = int(rng.integers(2))
    regime = str(rng.choice(("more_bends", "size", "staging")))
    base_staging = str(rng.choice(("horizontal", "vertical", "diagonal")))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            if split == "train" and index == 1:
                pair = _opposite_marker(puzzle["train"][0])
            else:
                is_test = split == "test"
                bends = int(rng.integers(2))
                if regime == "more_bends" and is_test:
                    bends = 2
                elif regime == "staging":
                    bends = int(rng.integers(3))
                scale = ("large" if is_test else "small") if regime == "size" else "mixed"
                staging = (str(rng.choice([s for s in ("horizontal", "vertical", "diagonal")
                                          if s != base_staging])) if is_test else base_staging)
                if regime != "staging":
                    staging = str(rng.choice(("horizontal", "vertical", "diagonal")))
                role = carrier
                if nt >= 3 and not (split == "train" and index == 0):
                    role = 1 - carrier if split == "train" and index == 2 else int(rng.integers(2))
                # Minimal two-demo puzzles retain the taught assembled carrier;
                # neither palette nor initial staging gives the donor its role.
                profile = {"palette": tuple(map(int, rng.choice(10, 6, replace=False)))
                           if change_palette and (is_test or index > 1) else palette,
                           "length": int(rng.integers(2, 4)) if split == "train" and index == 0
                           else int(rng.integers(1, 4)), "carrier": role, "side": side
                           if split == "train" and index == 0 else int(rng.integers(2)),
                           "rotation": (rotation + 2 * int(rng.integers(2))) % 4,
                           "bends": bends, "scale": scale, "staging": staging}
                if mixed_axis and nt >= 3 and (is_test or index >= 2):
                    profile["rotation"] = (rotation + (1 if split == "train" and index == 2
                                                       else int(rng.integers(2)))) % 4
                for _ in range(32):
                    pair = _construct(rng, profile, teaching=split == "train" and index == 0)
                    if _key(pair["input"]) not in seen:
                        break
                else:
                    raise RuntimeError("unable to make distinct inputs")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _cellwise_expected(d, reg):
    """Validator compositor: explicit donor lookups instead of set intersection."""
    values = {}
    for cell in sorted(reg["domain"]):
        if cell in reg["cut"]:
            contributors = [i for i in range(2) if cell in reg["pigments"][i]]
            _require(len(contributors) <= 1, "seam cannot have two ordinary contributors")
            if contributors:
                values[cell] = d["records"][contributors[0]]["pigment"]
        else:
            donor = reg["preferred"] if cell in reg["marked"] else 1 - reg["preferred"]
            rec = d["records"][donor]
            row, col = rec["bar"]
            y, x = cell[0] + row + donor, cell[1] + col
            if 0 <= y < rec["card"].shape[0] and 0 <= x < rec["card"].shape[1]:
                value = int(rec["card"][y, x])
                if value == rec["pigment"]:
                    values[cell] = value
    return _render(d, values), values


def _registration_evidence(d, output):
    """Reject same-line, opposite-order, and sheet-edge registration shortcuts."""
    edge_offset = d["records"][1]["bar"][1] - d["records"][0]["bar"][1]
    if edge_offset == 0:
        return False
    for seam, tangent in ((0, 0), (1, 0), (-1, edge_offset)):
        try:
            wrong = _render(d, _visible(d, _register(d, seam, tangent)))
        except ValueError:
            # This proposed join fails to reconstruct the required partition.
            continue
        if np.array_equal(wrong, output):
            return False
    return True


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "bad example counts")
    seen, coverage, train_roles = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            d = _extract(inp)
            reg = _register(d)
            expected, visible = _cellwise_expected(d, reg)
            _require(np.array_equal(out, expected), "incorrect registered regional composite or crop")
            _require(np.array_equal(solve(inp), out), "solver mismatch")
            colors = {d["background"]} | {rec["pigment"] for rec in d["records"]}
            _require(set(map(int, np.unique(out))) <= colors, "temporary controls leaked into output")
            # A tight foreground crop has a pigment on each of its four edges.
            _require(all(np.any(edge != d["background"]) for edge in
                         (out[0, :], out[-1, :], out[:, 0], out[:, -1])), "crop is not tight")
            _require(not np.array_equal(_render(d, _visible(d, reg, "union")), out),
                     "dropping regional selection would not change this answer")
            if split == "train":
                coverage |= {name for name, present in _evidence(d, reg, visible).items() if present}
                train_roles.add(d["axis"])
            else:
                _require(d["axis"] in train_roles,
                         "test bar orientation was not demonstrated")
    _require(coverage == set(_evidence(d, reg, visible)), "training lacks regional/transparency evidence")
    first, twin = puzzle["train"][:2]
    first_d = _extract(first["input"])
    first_reg = _register(first_d)
    _require(all(_evidence(first_d, first_reg, _visible(first_d, first_reg)).values()),
             "first demonstration lacks complete copying/suppression/transparency evidence")
    _require(_registration_evidence(first_d, np.asarray(first["output"])),
             "training does not distinguish registration shortcuts")
    _require(np.asarray(first["input"]).shape == np.asarray(twin["input"]).shape
             and np.count_nonzero(np.asarray(first["input"]) != np.asarray(twin["input"])) == 2,
             "first demonstrations must isolate moving the marker")
    _require(not np.array_equal(first["output"], twin["output"]), "marker twins have equal answers")


def self_test():
    """Bounded 42-seed survey plus counts and meaningful metamorphic checks."""
    fingerprints, shapes, structures, palettes = set(), set(), set(), set()
    lengths, axes, roles, bends = set(), set(), set(), set()
    overlap_sizes, output_shapes = set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        repeat = generate_puzzle(seed)
        fingerprint = tuple(_key(pair[k]) for split in ("train", "test")
                            for pair in puzzle[split] for k in ("input", "output"))
        _require(fingerprint == tuple(_key(pair[k]) for split in ("train", "test")
                                     for pair in repeat[split] for k in ("input", "output")),
                 "whole-puzzle determinism failed")
        fingerprints.add(fingerprint)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _extract(pair["input"])
                reg = _register(d)
                shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                lengths.add(d["length"])
                axes.add(d["axis"])
                roles.add(reg["preferred"])
                overlap_sizes.add(len(reg["pigments"][0] & reg["pigments"][1]))
                palettes.add((d["background"], d["frame"], d["control"], d["guide"],
                              *(rec["pigment"] for rec in d["records"])))
                normalized = pair["input"].copy()
                colors = {d["background"]: 0, d["frame"]: 1, d["control"]: 2, d["guide"]: 3,
                          d["records"][0]["pigment"]: 4, d["records"][1]["pigment"]: 5}
                for color, role in colors.items():
                    normalized[pair["input"] == color] = role
                structures.add(_key(normalized))
                bends.add(tuple(len({x for y, x in rec["path"]}) for rec in d["records"]))
        # Transposing the entire scene transposes the output; donor roles are
        # recovered from backing rather than global top/left source order.
        pair = puzzle["train"][0]
        _require(np.array_equal(solve(pair["input"].T), pair["output"].T), "transpose covariance")
        # Any bijective palette permutation preserves geometry and exact copying.
        permutation = np.random.default_rng(seed + 17).permutation(10)
        _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]),
                 "palette covariance")
        d = _extract(pair["input"])
        reg = _register(d)
        restaged = _stage(np.random.default_rng(seed + 29),
                          [rec["card"] for rec in d["records"]],
                          d["frame"], d["background"], "vertical")
        _require(restaged is not None, "could not restage a generated sheet pair")
        native_output = pair["output"].T if d["axis"] else pair["output"]
        _require(np.array_equal(solve(restaged), native_output), "staging changed the answer")
        _require(not np.array_equal(_render(d, _visible(d, reg, "reverse")), pair["output"]),
                 "reversed region ownership is indistinguishable")
        # Same-line and the opposite adjacent-bar order are geometrically possible
        # for these sheets but have different crops/pixels (or invalid topology).
        for seam in (0, 1):
            try:
                wrong = _render(d, _visible(d, _register(d, seam)))
            except ValueError:
                continue
            _require(not np.array_equal(wrong, pair["output"]), "wrong registration shortcut")
    _require(len(fingerprints) == 42, "seeds collapsed")
    _require(len(shapes) >= 35 and len(output_shapes) >= 25, "size variation collapsed")
    _require(len(structures) >= 160 and len(palettes) >= 35, "structural/palette variation collapsed")
    _require(lengths == {1, 2, 3} and axes == {0, 1} and roles == {0, 1}, "role/guide variation")
    _require(len(overlap_sizes) >= 8 and len(bends) >= 8, "overlap/cut variation collapsed")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(901 + 10 * nt + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(0)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt answer accepted")
    valid_input = generate_puzzle(0)["train"][0]["input"]
    parsed = _extract(valid_input)
    corruptions = []
    broken_frame = valid_input.copy()
    t, b, l, r = parsed["boxes"][0]
    broken_frame[t, l] = parsed["background"]
    corruptions.append(broken_frame)
    mismatched_bar = valid_input.copy()
    guide_cell = min(_cells(valid_input[t + 1:b, l + 1:r], parsed["guide"]))
    mismatched_bar[t + 1 + guide_cell[0], l + 1 + guide_cell[1]] = parsed["background"]
    corruptions.append(mismatched_bar)
    for malformed in corruptions:
        try:
            solve(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed frame or unmatched guide accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "input_shapes": len(shapes),
            "output_shapes": len(output_shapes), "normalized_structures": len(structures),
            "palettes": len(palettes), "guide_lengths": sorted(lengths),
            "overlap_counts": sorted(overlap_sizes), "cut_width_combinations": len(bends)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
