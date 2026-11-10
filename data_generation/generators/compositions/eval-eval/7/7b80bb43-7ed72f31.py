"""Selective line repair certified by a global reflection or half-turn.

Only standard library and NumPy are used. Construction records expected local
edits; the public solver reconstructs all edits from the supplied array alone.
"""

from __future__ import annotations

import numpy as np


MARKER = 5


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its legal range")
    return value


def _counts(num_train, num_test):
    n = _integer(num_train, "num_train", 2, 10)
    t = _integer(num_test, "num_test", 1, 4)
    if n <= t:
        raise ValueError("num_train must exceed num_test")
    return n, t


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((grid >= 0) & (grid <= 9))):
        raise ValueError("Expected an integer ARC grid, dimensions 1..30 and colors 0..9")
    return grid.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    components = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while cells:
        start = min(cells)
        cells.remove(start)
        stack, found = [start], []
        while stack:
            r, c = stack.pop()
            found.append((r, c))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in cells:
                    cells.remove(q)
                    stack.append(q)
        components.append(found)
    return components


def _at(mask, p):
    r, c = p
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _strong(mask):
    result = np.zeros_like(mask)
    for source, target in ((mask, result), (mask.T, result.T)):
        triples = source[:, :-2] & source[:, 1:-1] & source[:, 2:]
        for offset in range(3):
            target[:, offset:offset + triples.shape[1]] |= triples
    return result


def _brackets(mask, empty):
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        m, e = (mask, empty) if dc else (mask.T, empty.T)
        hits = m[:, :-4] & m[:, 4:] & e[:, 1:-3] & e[:, 2:-2] & e[:, 3:-1]
        for y, x in np.argwhere(hits):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            segment = tuple((r + k*dr, c + k*dc) for k in range(5))
            continued = _at(mask, (r-dr, c-dc)) or _at(mask, (r+5*dr, c+5*dc))
            result.append({"a": (r, c), "d": (dr, dc), "segment": segment,
                           "gap": segment[1:4], "continued": continued})
    return result


def _diagonal_links(weak):
    linked = np.zeros_like(weak)
    pairs = weak[:-1, :-1] & weak[1:, 1:]
    linked[:-1, :-1] |= pairs
    linked[1:, 1:] |= pairs
    pairs = weak[:-1, 1:] & weak[1:, :-1]
    linked[:-1, 1:] |= pairs
    linked[1:, :-1] |= pairs
    return linked


def _color_features(mask, empty):
    brackets = _brackets(mask, empty)
    protected = _strong(mask)
    for b in brackets:
        protected[b["segment"][0]] = protected[b["segment"][4]] = True
    weak = mask & ~protected
    candidates = []
    for b in brackets:
        if not b["continued"]:
            continue
        dr, dc = b["d"]
        matches = []
        for endpoint in (0, 4):
            ar, ac = b["segment"][endpoint]
            toward = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                sr, sc = toward*dr + side*dc, toward*dc + side*dr
                spur = tuple((ar + k*sr, ac + k*sc) for k in (1, 2))
                if (all(_at(weak, p) for p in spur)
                        and not _at(mask, (ar + 3*sr, ac + 3*sc))):
                    matches.append((endpoint, side, spur))
        if len(matches) > 1:
            raise ValueError("A requested gap has multiple competing deviations")
        if matches:
            endpoint, side, spur = matches[0]
            candidates.append(dict(b, endpoint=endpoint, side=side, spur=spur))
    return brackets, candidates, _diagonal_links(weak), protected


def _operator(grid):
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[counts.argmax()])
    if np.count_nonzero(grid == bg) <= grid.size // 2 or bg == MARKER:
        raise ValueError("Background must be a unique majority, distinct from marker 5")
    groups = _components(grid == MARKER, diagonal=True)
    if len(groups) != 1:
        raise ValueError("Expected exactly one marker component")
    marker = sorted(groups[0])
    rows, cols = {r for r, _ in marker}, {c for _, c in marker}
    if len(marker) == 1:
        kind = "P"
    elif len(rows) == 1 and len(marker) == max(cols) - min(cols) + 1:
        kind = "H"
    elif len(cols) == 1 and len(marker) == max(rows) - min(rows) + 1:
        kind = "V"
    else:
        raise ValueError("Marker must be a singleton or a contiguous straight segment")
    return bg, kind, marker[0], marker


def _transform(points, kind, center):
    result = np.asarray(points, dtype=int).reshape(-1, 2).copy()
    if kind in ("H", "P"):
        result[:, 0] = 2*center[0] - result[:, 0]
    if kind in ("V", "P"):
        result[:, 1] = 2*center[1] - result[:, 1]
    return result


def _same_interval(grid, points, color):
    a = np.asarray(points, dtype=int).reshape(-1, 2)
    h, w = grid.shape
    return bool(((a[:, 0] >= 0) & (a[:, 0] < h) & (a[:, 1] >= 0)
                 & (a[:, 1] < w)).all() and (grid[a[:, 0], a[:, 1]] == color).all())


def _read(grid):
    bg, kind, center, marker = _operator(grid)
    brackets, requests = [], []
    removed = np.zeros(grid.shape, dtype=bool)
    protected = np.zeros_like(removed)
    for color in sorted(set(map(int, np.unique(grid))) - {bg, MARKER}):
        local_b, local_r, erased, keep = _color_features(grid == color, grid == bg)
        removed |= erased
        protected |= keep
        for b in local_b:
            b = dict(b, color=color)
            b["image"] = _transform(b["segment"], kind, center)
            b["confirmed"] = _same_interval(grid, b["image"], color)
            brackets.append(b)
        for r in local_r:
            r = dict(r, color=color)
            r["image"] = _transform(r["segment"], kind, center)
            r["confirmed"] = _same_interval(grid, r["image"], color)
            requests.append(r)
    gap_cells, spur_cells = set(), set()
    for r in requests:
        if gap_cells.intersection(r["gap"]) or spur_cells.intersection(r["spur"]):
            raise ValueError("Requests overlap")
        gap_cells.update(r["gap"])
        spur_cells.update(r["spur"])
        if not all(removed[p] for p in r["spur"]):
            raise ValueError("Deviation is not removable diagonal evidence")
    return {"bg": bg, "kind": kind, "center": center, "marker": marker,
            "brackets": brackets, "requests": requests,
            "removed": removed, "protected": protected}


def _prediction(grid, info, mode="normal"):
    result = grid.copy()
    result[info["removed"]] = info["bg"]
    candidates = info["brackets"] if mode == "no_a" else info["requests"]
    for candidate in candidates:
        accepted = candidate["confirmed"] if mode != "no_b" else True
        if accepted and candidate["continued"]:
            for p in candidate["gap"]:
                result[p] = candidate["color"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every edit from one input, with no seed or generation state."""
    grid = _grid(input_grid)
    return _prediction(grid, _read(grid))


def _translation_support(grid, info, request):
    """Competing unchanged-shape translation, inferred from a local clean figure."""
    mask = (grid == request["color"]) & ~info["removed"]
    for p in request["gap"]:
        mask[p] = True
    component = next(c for c in _components(mask) if request["a"] in c)
    a = np.asarray(component, dtype=int)
    lo, hi = a.min(axis=0), a.max(axis=0)
    corners = [tuple(lo), tuple(hi)]
    target_lo = _transform(corners, info["kind"], info["center"]).min(axis=0)
    translated = np.asarray(request["segment"]) + target_lo - lo
    return _same_interval(grid, translated, request["color"])


def _overhang(info, request):
    if info["kind"] == "P":
        return False
    coord = 1 if info["kind"] == "H" else 0
    lo = min(p[coord] for p in info["marker"])
    hi = max(p[coord] for p in info["marker"])
    return min(p[coord] for p in request["segment"]) < lo or max(p[coord] for p in request["segment"]) > hi


def _facts(grid, info):
    requests = info["requests"]
    keys = {(r["color"], frozenset(r["gap"])) for r in requests}
    silent = [b for b in info["brackets"] if b["continued"] and b["confirmed"]
              and (b["color"], frozenset(b["gap"])) not in keys]
    mutual, wrong = [], []
    for r in requests:
        image_gap = frozenset(map(tuple, _transform(r["gap"], info["kind"], info["center"]).tolist()))
        if (r["color"], image_gap) in keys:
            mutual.append(r)
        image = r["image"]
        h, w = grid.shape
        if ((image[:, 0] >= 0) & (image[:, 0] < h) & (image[:, 1] >= 0) & (image[:, 1] < w)).all():
            values = grid[image[:, 0], image[:, 1]]
            if len(set(map(int, values))) == 1 and int(values[0]) not in (info["bg"], MARKER, r["color"]):
                wrong.append(r)
    return {"confirmed": [r for r in requests if r["confirmed"]],
            "unsupported": [r for r in requests if not r["confirmed"]],
            "silent": silent, "mutual": mutual, "wrong": wrong}


def _complementary_witness(grid, info):
    clean = _prediction(grid, info)
    confirmed = [r for r in info["requests"] if r["confirmed"]]
    for color in {r["color"] for r in confirmed}:
        components = _components(clean == color)
        for r in (q for q in confirmed if q["color"] == color):
            component = next(c for c in components if r["a"] in c)
            image = set(map(tuple, _transform(component, info["kind"], info["center"]).tolist()))
            if any(q["color"] == color and q["d"] != r["d"] and q["a"] in image for q in confirmed):
                return True
    return False


def _thin(mask):
    return not bool((mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]).any())


def _fingerprint(grid):
    return tuple(grid.shape), grid.astype(np.int8).tobytes()


def _palette(rng, params, need_two):
    pool = list(params["payloads"])
    bg = params["background"] if params["steady_bg"] else int(rng.choice([c for c in range(10) if c != MARKER]))
    if bg in pool and (len(pool) == 1 or (need_two and len(pool) == 2)):
        bg = int(rng.choice([c for c in range(10) if c != MARKER and c not in pool]))
    return bg, [c for c in pool if c != bg]


def _base_figure(rng, axis, compact):
    """Draw variable open and closed thin scaffolds and a legal local request."""
    if compact:
        # Aligned little Ls have no accidental brackets through request pixels.
        base = {(0, c) for c in range(6)} | {(r, 0) for r in range(1, 4)}
        gap = {(0, c) for c in (1, 2, 3)}
        spur = {(1, 1), (2, 2)}
        if axis == "v":
            base, gap, spur = [{(c, r) for r, c in cells} for cells in (base, gap, spur)]
        return base, gap, spur
    for _ in range(24):
        length = int(rng.integers(6, 13))
        gap_start = int(rng.integers(1, length-3))
        gap = {(0, c) for c in range(gap_start, gap_start+3)}
        base = {(0, c) for c in range(length)}
        form = str(rng.choice(["L", "T", "branch", "loop"], p=[.35, .25, .25, .15]))
        retained = [c for c in range(length) if (0, c) not in gap]
        column = int(rng.choice([0, length-1])) if form == "loop" else int(rng.choice(retained))
        sign = int(rng.choice([-1, 1]))
        depth = int(rng.integers(3, 7))
        for k in range(1, depth+1):
            base.add((sign*k, column))
        if form == "T":
            depth2 = int(rng.integers(3, 6))
            base.update({(-sign*k, column) for k in range(1, depth2+1)})
        if form in ("branch", "loop"):
            choices = [c for c in retained if abs(c-column) >= 3]
            if choices:
                other = (length-1 if column == 0 else 0) if form == "loop" else int(rng.choice(choices))
                depth2 = depth if form == "loop" else int(rng.integers(3, 7))
                sign2 = sign if form == "loop" else int(rng.choice([-1, 1]))
                base.update({(sign2*k, other) for k in range(1, depth2+1)})
                if form == "loop":
                    base.update({(sign*depth, c) for c in range(min(column, other), max(column, other)+1)})
        endpoint, bend = int(rng.choice([0, 4])), int(rng.choice([-1, 1]))
        anchor = gap_start-1 + endpoint
        toward = 1 if endpoint == 0 else -1
        spur = {(bend*k, anchor+toward*k) for k in (1, 2)}
        if base.intersection(spur):
            continue
        whole = base | spur
        r0, c0 = min(r for r, c in whole), min(c for r, c in whole)
        move = lambda cells: {(r-r0, c-c0) for r, c in cells}
        base, gap, spur = move(base), move(gap), move(spur)
        if axis == "v":
            base, gap, spur = [{(c, r) for r, c in cells} for cells in (base, gap, spur)]
        cells = base | spur
        h, w = max(r for r, c in cells)+1, max(c for r, c in cells)+1
        mask = np.zeros((h, w), bool)
        for p in (base-gap) | spur:
            mask[p] = True
        _, requests, removed, _ = _color_features(mask, ~mask)
        clean = mask & ~removed
        if len(requests) == 1 and set(requests[0]["gap"]) == gap and set(requests[0]["spur"]) == spur:
            if _thin(clean) and not _color_features(_mask(base, h, w), ~_mask(base, h, w))[2].any():
                return base, gap, spur
    raise ValueError("Could not draw a unique thin local request")


def _mask(cells, h, w):
    mask = np.zeros((h, w), bool)
    for p in cells:
        mask[p] = True
    return mask


def _complementary_figure(rng, axis, compact):
    """Two different damaged arms can certify each other without a clean figure."""
    width = 6 if compact else int(rng.integers(6, 13))
    depth = 6 if compact else int(rng.integers(6, 10))
    base = {(0, c) for c in range(width)} | {(r, 0) for r in range(1, depth+1)}
    primary = {(0, c) for c in (1, 2, 3)}
    secondary = {(r, 0) for r in (1, 2, 3)}
    spur = {(1, 1), (2, 2)}
    all_sets = [base, primary, secondary, spur]
    if not compact:
        flip_r, flip_c = bool(rng.integers(2)), bool(rng.integers(2))
        all_sets = [{(depth-r if flip_r else r, width-1-c if flip_c else c)
                     for r, c in cells} for cells in all_sets]
    if axis == "v":
        all_sets = [{(c, r) for r, c in cells} for cells in all_sets]
    return all_sets


def _prototype(rng, spec, pool, kind, compact):
    axis = spec.get("axis")
    if axis is None:
        axis = ("v" if kind == "V" else "h") if compact or spec.get("witness") else str(rng.choice(["h", "v"]))
    state = spec["state"]
    if state == "complementary":
        base, gap, second_gap, spur = _complementary_figure(rng, axis, compact)
    else:
        base, gap, spur = _base_figure(rng, axis, compact)
    color = int(rng.choice(pool))
    partner_color = int(rng.choice([c for c in pool if c != color])) if state == "wrong" else color
    source_in = base-gap if state == "silent" else (base-gap) | spur
    source_out = base if state in ("confirmed", "complementary") else base-gap
    partner_in = base if state in ("confirmed", "silent", "wrong") else base-gap
    partner_out = partner_in.copy()
    if state == "complementary":
        partner_in = (base-second_gap) | spur
        partner_out = base.copy()
    if state == "unsupported" and spec.get("mutual"):
        partner_in = partner_in | spur
    return {"source_in": source_in, "source_out": source_out,
            "partner_in": partner_in, "partner_out": partner_out,
            "foot": sorted(base | spur), "color": color, "partner_color": partner_color,
            "side": 1 if compact else int(rng.choice([-1, 1])),
            "half_axis": (1 if kind == "V" else 0 if compact else int(rng.integers(2)) if kind == "P" else 0)}


def _halo(blocked, cells, radius=1):
    h, w = blocked.shape
    for r, c in cells:
        blocked[max(0, r-radius):min(h, r+radius+1), max(0, c-radius):min(w, c+radius+1)] = True


def _pack(rng, prototypes, shape, kind, interior, edge, compact):
    h, w = shape
    center = (h//2 + (-1 if compact else int(rng.integers(-2, 3))),
              w//2 + (-1 if compact else int(rng.integers(-2, 3))))
    if kind == "P":
        marker = [center]
    else:
        length = 2 if compact else int(rng.integers(2, 9))
        span = w if kind == "H" else h
        start = int(rng.integers(1 if interior else 0, span-length-(1 if interior else 0)+1))
        marker = [(center[0], c) for c in range(start, start+length)] if kind == "H" else [(r, center[1]) for r in range(start, start+length)]
    blocked = np.zeros(shape, bool)
    _halo(blocked, marker)
    placed = []
    order = sorted(range(len(prototypes)), key=lambda i: -len(prototypes[i]["foot"]))
    for index in order:
        p = prototypes[index]
        local = np.asarray(p["foot"], dtype=int)
        sh, sw = local.max(axis=0)+1
        if compact:
            # Conservative reservation prevents remote motifs from protecting
            # a spur through incidental brackets. Normal sampling uses footprints.
            local = np.asarray([(r, c) for r in range(sh) for c in range(sw)], dtype=int)
        options = [(r, c) for r in range(h-sh+1) for c in range(w-sw+1)]
        if not compact:
            rng.shuffle(options)
        if edge:
            options.sort(key=lambda q: min(q[0], q[1], h-sh-q[0], w-sw-q[1]))
        chosen = None
        for origin in options:
            source = local + origin
            coord = source[:, p["half_axis"]]
            axis_center = center[p["half_axis"]]
            if not (coord.max() < axis_center if p["side"] < 0 else coord.min() > axis_center):
                continue
            if compact and coord.min() < axis_center+2:
                continue
            partner = _transform(source, kind, center)
            all_cells = np.concatenate((source, partner))
            margin = 1 if interior else 0
            if not ((all_cells[:, 0] >= margin) & (all_cells[:, 0] < h-margin)
                    & (all_cells[:, 1] >= margin) & (all_cells[:, 1] < w-margin)).all():
                continue
            if blocked[all_cells[:, 0], all_cells[:, 1]].any():
                continue
            if (np.abs(source[:, None, :] - partner[None, :, :]).max(axis=2) <= 1).any():
                continue
            chosen = origin
            _halo(blocked, all_cells.tolist(), radius=3 if compact else 1)
            break
        if chosen is None:
            return None
        placed.append((p, chosen))
    return center, marker, placed, blocked


def _paint(shape, bg, kind, placement):
    center, marker, placed, blocked = placement
    inp = np.full(shape, bg, dtype=np.int8)
    out = inp.copy()
    for q in marker:
        inp[q] = out[q] = MARKER
    for p, origin in placed:
        for name, grid in (("source_in", inp), ("source_out", out)):
            for r, c in p[name]:
                grid[r+origin[0], c+origin[1]] = p["color"]
        for name, grid in (("partner_in", inp), ("partner_out", out)):
            points = [(r+origin[0], c+origin[1]) for r, c in sorted(p[name])]
            for q in _transform(points, kind, center):
                grid[tuple(q)] = p["partner_color"]
    return inp, out, blocked


def _scatter(rng, inp, out, blocked, pool, richness, interior=False):
    for _ in range(int(rng.integers(richness+3))):
        diagonal = bool(rng.integers(2))
        length = int(rng.choice([2, 3])) if diagonal else int(rng.integers(3, 7))
        dr, dc = (int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))) if diagonal else ((1, 0) if rng.integers(2) else (0, 1))
        color = int(rng.choice(pool))
        for _ in range(24):
            r, c = int(rng.integers(inp.shape[0])), int(rng.integers(inp.shape[1]))
            cells = [(r+k*dr, c+k*dc) for k in range(length)]
            margin = 1 if interior else 0
            if not all(margin <= a < inp.shape[0]-margin and margin <= b < inp.shape[1]-margin and not blocked[a, b] for a, b in cells):
                continue
            trial, target = inp.copy(), out.copy()
            for q in cells:
                trial[q] = color
                if not diagonal:
                    target[q] = color
            try:
                if not np.array_equal(solve(trial), target):
                    continue
            except ValueError:
                continue
            inp, out = trial, target
            _halo(blocked, cells)
            break
    return inp, out


def _has_edge(grid, bg):
    return bool((grid[0] != bg).any() or (grid[-1] != bg).any()
                or (grid[:, 0] != bg).any() or (grid[:, -1] != bg).any())


def _acceptable(inp, out, contrast, require_wrong, edge):
    try:
        info = _read(inp)
        if not np.array_equal(_prediction(inp, info), out):
            return False
        facts = _facts(inp, info)
        if not facts["confirmed"]:
            return False
        if contrast and not (facts["unsupported"] and facts["silent"] and facts["mutual"]):
            return False
        if require_wrong and not facts["wrong"]:
            return False
        if contrast:
            if not any(not _translation_support(inp, info, r) for r in facts["confirmed"]):
                return False
            if info["kind"] != "P" and not any(_overhang(info, r) for r in facts["confirmed"]):
                return False
            if info["kind"] == "P" and not any(
                not _same_interval(inp, _transform(r["segment"], "H", info["center"]), r["color"])
                and not _same_interval(inp, _transform(r["segment"], "V", info["center"]), r["color"])
                for r in facts["confirmed"]):
                return False
        if edge and not _has_edge(out, info["bg"]):
            return False
        return all(_thin(out == c) for c in np.unique(out) if c not in (info["bg"], MARKER))
    except ValueError:
        return False


def _make_pair(rng, params, kind, specs, contrast=False, require_wrong=False,
               test=False, familiar_shape=None):
    edge = test and params["shift"] == "edge"
    interior = not test and params["shift"] == "edge"
    bg, pool = _palette(rng, params, any(s["state"] == "wrong" for s in specs))
    for attempt in range(18):
        compact = attempt >= 14
        if compact:
            shape = (30, 30)
        elif attempt >= 8:
            shape = (int(rng.integers(27, 31)), int(rng.integers(27, 31)))
        elif familiar_shape is not None and attempt < 3:
            shape = familiar_shape
        else:
            low = 22 if len(specs) >= 3 else 16
            shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
        try:
            prototypes = [_prototype(rng, spec, pool, kind, compact) for spec in specs]
        except ValueError:
            continue
        placement = _pack(rng, prototypes, shape, kind, interior, edge, compact)
        if placement is None:
            continue
        inp, out, blocked = _paint(shape, bg, kind, placement)
        if not _acceptable(inp, out, contrast, require_wrong, edge):
            continue
        if not compact:
            inp, out = _scatter(rng, inp, out, blocked, pool, params["richness"], interior)
        if interior and _has_edge(inp, bg):
            # Optional scatter must honor the interior training regime too.
            continue
        if _acceptable(inp, out, contrast, require_wrong, edge):
            return {"input": inp, "output": out}
    raise RuntimeError("Bounded construction could not place a readable repair scene")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    n, t = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    order = list(map(int, rng.permutation([c for c in range(10) if c != MARKER])))
    palette_size = int(rng.integers(1, 5))
    teach_wrong = palette_size >= 2 and bool(rng.random() < .55)
    k = int(rng.integers(1, min(3, n-int(teach_wrong))+1))
    kinds = list(map(str, rng.choice(["H", "V", "P"], k, replace=False)))
    params = {"payloads": order[1:1+palette_size], "background": order[0],
              "steady_bg": bool(rng.random() < .50), "richness": int(rng.integers(3)),
              "shift": str(rng.choice(["count", "edge", "recombine"]))}
    states = ["confirmed", "unsupported", "silent"] + (["wrong"] if teach_wrong else [])
    plans = []
    for kind in kinds:
        axis = "v" if kind == "V" or (kind == "P" and "H" in kinds and "V" not in kinds) else "h"
        plans.append((kind, [{"state": "complementary", "witness": True, "axis": axis},
                             {"state": "unsupported", "mutual": True},
                             {"state": "silent"}], True, False))
    if teach_wrong:
        plans.append((str(rng.choice(kinds)), [{"state": "confirmed", "witness": True},
                                              {"state": "wrong"}, {"state": "silent"}], False, True))
    while len(plans) < n:
        count = int(rng.integers(1, 4))
        specs = [{"state": "confirmed"}] + [{"state": str(rng.choice(states)), "mutual": bool(rng.integers(2))}
                                              for _ in range(count-1)]
        plans.append((str(rng.choice(kinds)), specs, False, False))
    # Two confirmed repair orientations are taught, even in one-operation puzzles.
    # When only one operation is selected there is always a supplementary pair.
    forced_axes = {p[1][0].get("axis") for p in plans if p[1][0].get("axis")}
    if len(forced_axes) == 1:
        missing = "v" if "h" in forced_axes else "h"
        plans[len(kinds)][1][0]["axis"] = missing
    rng.shuffle(plans)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", n), ("test", t)):
        for index in range(count):
            if split == "train":
                kind, specs, contrast, wrong = plans[index]
            else:
                kind = str(rng.choice(kinds))
                motif_count = 4 if params["shift"] == "count" else int(rng.integers(1, 4))
                specs = [{"state": "confirmed"}] + [{"state": str(rng.choice(states)), "mutual": bool(rng.integers(2))}
                                                     for _ in range(motif_count-1)]
                contrast, wrong = False, False
            familiar = None
            if split == "test" and params["shift"] == "recombine" and rng.random() < .65:
                familiar = puzzle["train"][int(rng.integers(n))]["input"].shape
            for _ in range(12):
                pair = _make_pair(rng, params, kind, specs, contrast, wrong, split == "test", familiar)
                key = _fingerprint(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, taught, witnesses, contrasts, point_witnesses, overhang_kinds = set(), set(), set(), set(), set(), set()
    train_wrong, test_wrong, mutual_train, no_a_diff, no_b_diff = False, False, False, False, False
    train_axes, test_axes = set(), set()
    complementary_kinds = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _fingerprint(inp)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            info = _read(inp)
            facts = _facts(inp, info)
            assert facts["confirmed"], "No actual certified repair"
            assert np.array_equal(_prediction(inp, info), out), "Incomplete or incorrect reconstruction"
            assert np.array_equal(inp[inp == MARKER], out[inp == MARKER])
            added = set(map(tuple, np.argwhere((inp == info["bg"]) & (out != info["bg"])).tolist()))
            assert added == {p for r in facts["confirmed"] for p in r["gap"]}
            assert np.array_equal((inp != info["bg"]) & (out == info["bg"]), info["removed"])
            for cells in _components(info["removed"], diagonal=True):
                cells = sorted(cells)
                assert len(cells) in (2, 3)
                steps = {(b[0]-a[0], b[1]-a[1]) for a, b in zip(cells, cells[1:])}
                assert steps in ({(1, 1)}, {(1, -1)}), "Erased cells are not short diagonal strokes"
            for r in info["requests"]:
                assert len(r["gap"]) == 3 and len(r["spur"]) == 2
                assert all(out[p] == (r["color"] if r["confirmed"] else info["bg"]) for p in r["gap"])
                assert all(out[p] == info["bg"] for p in r["spur"])
            for color in np.unique(out):
                if color not in (info["bg"], MARKER):
                    assert _thin(out == color), "Thick scaffold"
            if split == "train":
                train_axes.update(r["d"] for r in facts["confirmed"])
                if _complementary_witness(inp, info):
                    complementary_kinds.add(info["kind"])
                taught.add(info["kind"])
                if facts["unsupported"] and facts["silent"]:
                    contrasts.add(info["kind"])
                if any(not _translation_support(inp, info, r) for r in facts["confirmed"]):
                    witnesses.add(info["kind"])
                if any(_overhang(info, r) for r in facts["confirmed"]):
                    overhang_kinds.add(info["kind"])
                if info["kind"] == "P" and any(
                    not _same_interval(inp, _transform(r["segment"], "H", info["center"]), r["color"])
                    and not _same_interval(inp, _transform(r["segment"], "V", info["center"]), r["color"])
                    for r in facts["confirmed"]):
                    point_witnesses.add("P")
                train_wrong |= bool(facts["wrong"])
                mutual_train |= bool(facts["mutual"])
                no_a_diff |= not np.array_equal(_prediction(inp, info, "no_a"), out)
                no_b_diff |= not np.array_equal(_prediction(inp, info, "no_b"), out)
            else:
                test_axes.update(r["d"] for r in facts["confirmed"])
                assert info["kind"] in taught, "Test operation has no training demonstration"
                test_wrong |= bool(facts["wrong"])
    assert taught <= witnesses and taught <= contrasts, "An active operation lacks discriminating training"
    assert taught <= complementary_kinds, "Complementary damage and repair multiplicity were not taught"
    assert taught - {"P"} <= overhang_kinds, "Finite-axis overhang was not taught"
    assert "P" not in taught or "P" in point_witnesses
    assert mutual_train, "Training must expose mutually damaged counterparts"
    assert train_axes == {(0, 1), (1, 0)} and test_axes <= train_axes, "Repair direction was not taught"
    assert no_a_diff and no_b_diff, "Training does not distinguish both parent ablations"
    assert not test_wrong or train_wrong, "Wrong-color rejection must be taught"


def _copy_puzzle(puzzle):
    return {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzle[s]] for s in ("train", "test")}


def _puzzle_key(puzzle):
    return tuple((_fingerprint(p["input"]), _fingerprint(p["output"])) for s in ("train", "test") for p in puzzle[s])


def self_test() -> dict:
    """Bounded structural/semantic regressions, including 42-seed diversity."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, structures, palettes, kinds, axes, bends, endpoints = set(), set(), set(), set(), set(), set(), set()
    request_counts, repair_counts, bg_roles, edge_count, wrong_count, loops = set(), set(), set(), 0, 0, 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _read(inp)
                facts = _facts(inp, info)
                assert np.array_equal(solve(inp.tolist()), out)
                shapes.add(inp.shape)
                role_map = {info["bg"]: 0, MARKER: 1}
                normalized = tuple(role_map.setdefault(int(c), len(role_map)) for c in inp.flat)
                structures.add((inp.shape, normalized))
                palettes.add(tuple(map(int, np.unique(inp))))
                kinds.add(info["kind"])
                bg_roles.add(info["bg"])
                request_counts.add(len(info["requests"]))
                repair_counts.add(len(facts["confirmed"]))
                wrong_count += bool(facts["wrong"])
                edge_count += _has_edge(out, info["bg"])
                for r in info["requests"]:
                    axes.add(r["d"])
                    bends.add(r["side"])
                    endpoints.add(r["endpoint"])
                for color in set(map(int, np.unique(out))) - {info["bg"], MARKER}:
                    for comp in _components(out == color):
                        cells = set(comp)
                        edges = sum((r+1, c) in cells for r, c in cells) + sum((r, c+1) in cells for r, c in cells)
                        loops += edges >= len(cells)
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len(shapes) >= 20 and len(structures) >= 150 and len(palettes) >= 20
    assert kinds == {"H", "V", "P"} and axes == {(0, 1), (1, 0)}
    assert bends == {-1, 1} and endpoints == {0, 4}
    assert len(request_counts) >= 4 and len(bg_roles) >= 6
    assert {1, 2, 3} <= repair_counts, "Confirmed repair multiplicity does not vary"
    assert edge_count > 5 and wrong_count > 5 and loops > 0
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            p = generate_puzzle(9183+n*7+t, n, t)
            assert len(p["train"]) == n and len(p["test"]) == t
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # A label which omits even one part of the rule must be rejected.
    original = puzzles[0]
    for mutation in ("omitted", "silent", "unsupported", "spur"):
        bad = _copy_puzzle(original)
        changed = False
        for pair in bad["train"]:
            info = _read(pair["input"])
            facts = _facts(pair["input"], info)
            if mutation == "omitted" and len(facts["confirmed"]) < 2:
                continue
            candidates = facts["silent"] if mutation == "silent" else facts["unsupported"] if mutation == "unsupported" else facts["confirmed"]
            if not candidates:
                continue
            r = candidates[0]
            if mutation == "spur":
                for q in r["spur"]:
                    pair["output"][q] = r["color"]
            else:
                for q in r["gap"]:
                    pair["output"][q] = info["bg"] if mutation == "omitted" else r["color"]
                if mutation == "omitted":
                    for q in r["spur"]:
                        pair["output"][q] = r["color"]
            changed = True
            break
        assert changed
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"Accepted {mutation} label mutation")
    # Coordinate transforms are involutions, and out-of-bounds evidence is false.
    points = np.array([[0, 2], [3, 8], [7, 4]])
    for kind in ("H", "V", "P"):
        assert np.array_equal(_transform(_transform(points, kind, (4, 5)), kind, (4, 5)), points)
    assert not _same_interval(np.zeros((3, 3), int), [(-1, 0)], 0)
    # Hand-built counterfactuals isolate the conjunction and frozen evidence.
    fixture = np.zeros((15, 13), dtype=np.int8)
    fixture[7, :2] = MARKER
    fixture[2, 1:10] = fixture[12, 1:10] = 2
    fixture[:3, 9] = fixture[12:, 9] = 2
    fixture[2, 3:6] = 0
    fixture[3, 3] = fixture[4, 4] = 2
    expected = fixture.copy()
    expected[2, 3:6] = 2
    expected[3, 3] = expected[4, 4] = 0
    assert np.array_equal(solve(fixture), expected)
    silent = fixture.copy()
    silent[3, 3] = silent[4, 4] = 0
    assert np.array_equal(solve(silent), silent)
    broken = fixture.copy()
    broken[12, 3:6] = 0
    broken_expected = broken.copy()
    broken_expected[3, 3] = broken_expected[4, 4] = 0
    assert np.array_equal(solve(broken), broken_expected)
    twin = broken.copy()
    twin[11, 3] = twin[10, 4] = 2
    assert np.array_equal(solve(twin), broken_expected)
    wrong = fixture.copy()
    wrong[12, 1:10] = wrong[12:, 9] = 3
    wrong_expected = wrong.copy()
    wrong_expected[3, 3] = wrong_expected[4, 4] = 0
    assert np.array_equal(solve(wrong), wrong_expected)
    clipped = twin.copy()
    clipped[7, :2] = 0
    clipped[1, :2] = MARKER
    clipped_expected = broken_expected.copy()
    clipped_expected[7, :2] = 0
    clipped_expected[1, :2] = MARKER
    assert np.array_equal(solve(clipped), clipped_expected)
    # Rotating an actual scene also rotates its marker convention and its answer.
    for puzzle in puzzles[:3]:
        for pair in puzzle["train"]:
            for turns in range(4):
                assert np.array_equal(solve(np.rot90(pair["input"], turns)), np.rot90(pair["output"], turns))
                assert np.array_equal(solve(np.rot90(pair["input"][:, ::-1], turns)), np.rot90(pair["output"][:, ::-1], turns))
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "structures": len(structures),
            "palettes": len(palettes), "request_counts": sorted(request_counts),
            "repair_counts": sorted(repair_counts),
            "edge_pairs": edge_count, "wrong_color_pairs": wrong_count, "loop_components": loops}


if __name__ == "__main__":
    print(self_test())
