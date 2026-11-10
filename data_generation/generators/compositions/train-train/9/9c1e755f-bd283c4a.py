"""Scaffold-settled frequency tiles.

Composition of ARC 9c1e755f (a straight scaffold bar carries a terminal tile
that is repeated along the bar, phase-anchored at the tile) and bd283c4a
(cells are ranked by color frequency and packed column by column, bottom to
top, left to right).

Rule: every assembly is a straight one-cell-wide scaffold bar with a scrambled
rectangular sample block beside one terminal end.  The block's cells are
re-packed by descending local color frequency in the scaffold's own frame: the
bar is the floor (fill each cross-section from the bar outward) and the
anchored terminal end is the start (advance one cross-section at a time away
from that end).  The packed block is then repeated along the full extent of
the bar, phase-anchored at the block, filling the band beside the bar.
"""

from __future__ import annotations

import numpy as np

FRAMES = tuple((axis, side, anchor) for axis in ("H", "V")
               for side in (-1, 1) for anchor in ("lo", "hi"))
# Bar under the block, anchored at the left: bd283c4a's own absolute traversal.
CANONICAL_FRAME = ("H", -1, "lo")
MAX_PERIOD, MAX_DEPTH, MAX_BAND = 4, 7, 18


def _traversal_family():
    """Candidate block traversals (outer, along_dir, across_dir, col_sign, row_sign, snake).

    outer names the dimension visited in the outer loop, either relative to the
    scaffold ("along"/"across") or absolute ("col"/"row").  Each dimension's
    direction is relative ("rel+" = from the terminal end / from the bar,
    "rel-" = reversed) or absolute ("abs", using col_sign/row_sign).  snake
    reverses the inner direction on every other outer step (boustrophedon).
    """
    family = []
    dirs = ("rel+", "rel-", "abs")
    for outer in ("along", "across", "col", "row"):
        for along_dir in dirs:
            for across_dir in dirs:
                signs = ((1, 1), (1, -1), (-1, 1), (-1, -1)) if "abs" in (along_dir, across_dir) else ((1, 1),)
                family += [(outer, along_dir, across_dir, cs, rs, False) for cs, rs in signs]
    family += [(outer, a, b, 1, 1, True) for outer in ("along", "across")
               for a in ("rel+", "rel-") for b in ("rel+", "rel-")]
    return tuple(family)


TRAVERSALS = _traversal_family()
TRUE_TRAVERSAL = ("along", "rel+", "rel+", 1, 1, False)
# bd283c4a's literal convention: columns left to right, each bottom to top.
PARENT_B_TRAVERSAL = ("col", "abs", "abs", 1, -1, False)
TRUE_VARIANT = (TRUE_TRAVERSAL, "local", "block", "sort")


# --------------------------------------------------------------------------
# Geometry shared by rendering and solving
# --------------------------------------------------------------------------

def _cells(frame, top, left, U, d, u, v):
    """Map scaffold-local (u, v) to grid (row, col).

    u is the distance along the bar from the block's terminal slice, v the
    distance from the bar (v = -1 is the bar itself, v = 0 touches it).
    """
    axis, side, anchor = frame
    u = np.asarray(u)
    v = np.asarray(v)
    along = u if anchor == "lo" else U - 1 - u
    across = d - 1 - v if side == -1 else 1 + v
    if axis == "H":
        return top + across, left + along
    return top + along, left + across


def _bbox(frame, U, d):
    return (d + 1, U) if frame[0] == "H" else (U, d + 1)


def _sorted_tile(raw):
    """Pack the block's own cells by descending count, cross-section by cross-section."""
    colors, counts = np.unique(raw, return_counts=True)
    order = np.lexsort((colors, -counts))
    return np.repeat(colors[order], counts[order]).reshape(raw.shape)


def _extend(tile, U, phase="block"):
    """Repeat a tile along the bar.  Only "block" is the rule; the other
    phases are alternative readings used to check the demonstrations."""
    us = np.arange(U)
    p = tile.shape[0]
    if phase == "block":
        return tile[us % p]
    if phase == "far":
        return tile[(us - U) % p]
    if isinstance(phase, tuple):  # ("bar", o): restart the period where the bar starts
        return tile[(us - phase[1]) % p]
    k = us % (2 * p)  # "mirror": reflect the tile at every copy
    return tile[np.where(k < p, k, 2 * p - 1 - k)]


def _local_grid(U, d):
    return np.meshgrid(np.arange(U), np.arange(d), indexing="ij")


def _render(shape, bg, scaffold, objects):
    inp = np.full(shape, bg, dtype=np.int64)
    out = inp.copy()
    for obj in objects:
        frame, top, left = obj["frame"], obj["top"], obj["left"]
        U, d, p, o = obj["U"], obj["d"], obj["p"], obj["o"]
        bar_u = np.arange(o, U)
        br, bc = _cells(frame, top, left, U, d, bar_u, np.full(bar_u.shape, -1))
        inp[br, bc] = scaffold
        out[br, bc] = scaffold
        us, vs = _local_grid(p, d)
        rr, cc = _cells(frame, top, left, U, d, us, vs)
        inp[rr, cc] = obj["raw"]
        us, vs = _local_grid(U, d)
        rr, cc = _cells(frame, top, left, U, d, us, vs)
        out[rr, cc] = _extend(_sorted_tile(obj["raw"]), U)
    return inp, out


# --------------------------------------------------------------------------
# Input-only parsing and solving
# --------------------------------------------------------------------------

def _components(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            seen[r0, c0] = True
            stack, cells = [(r0, c0)], []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(cells)
    return comps


def _parse(grid, bg, scaffold):
    """Return the assemblies of a grid under the given roles, or None."""
    h, w = grid.shape
    bars = _components(grid == scaffold)
    blocks = _components((grid != bg) & (grid != scaffold))
    if not bars or len(bars) != len(blocks):
        return None
    block_id = np.full(grid.shape, -1, dtype=np.int64)
    rects = []
    for k, cells in enumerate(blocks):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
            return None
        rects.append((r0, r1, c0, c1))
        for r, c in cells:
            block_id[r, c] = k
    objects, used = [], set()
    claimed = np.zeros(grid.shape, dtype=bool)
    for bar in bars:
        if len(bar) < 3:
            return None
        rows = {r for r, _ in bar}
        cols = {c for _, c in bar}
        if len(rows) == 1:
            axis, line = "H", next(iter(rows))
            a, b = min(cols), max(cols)
        elif len(cols) == 1:
            axis, line = "V", next(iter(cols))
            a, b = min(rows), max(rows)
        else:
            return None
        touching = set()
        for r, c in bar:
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and block_id[rr, cc] >= 0:
                    touching.add(int(block_id[rr, cc]))
        if len(touching) != 1:
            return None
        k = touching.pop()
        if k in used:
            return None
        used.add(k)
        r0, r1, c0, c1 = rects[k]
        # (x0, x1) is the block's extent along the bar, (y0, y1) across it.
        x0, x1, y0, y1 = (c0, c1, r0, r1) if axis == "H" else (r0, r1, c0, c1)
        if y1 == line - 1:
            side = -1
        elif y0 == line + 1:
            side = 1
        else:
            return None
        lo_ok = x0 <= a <= x1 and b > x1
        hi_ok = x0 <= b <= x1 and a < x0
        if lo_ok == hi_ok:
            return None
        p, d = x1 - x0 + 1, y1 - y0 + 1
        if lo_ok:
            anchor, o, U, start = "lo", a - x0, b - x0 + 1, x0
        else:
            anchor, o, U, start = "hi", x1 - b, x1 - a + 1, a
        across0 = y0 if side == -1 else line
        top, left = (across0, start) if axis == "H" else (start, across0)
        frame = (axis, side, anchor)
        us, vs = _local_grid(p, d)
        rr, cc = _cells(frame, top, left, U, d, us, vs)
        raw = grid[rr, cc].copy()
        us, vs = _local_grid(U, d)
        rr, cc = _cells(frame, top, left, U, d, us, vs)
        band_in = grid[rr, cc]
        if np.any(band_in[p:] != bg) or np.any(claimed[rr, cc]):
            return None
        claimed[rr, cc] = True
        objects.append({"frame": frame, "top": int(top), "left": int(left), "U": int(U),
                        "d": int(d), "p": int(p), "o": int(o), "raw": raw})
    if len(used) != len(blocks):
        return None
    return objects


def _roles(grid):
    """Background is the strict modal color; the scaffold color is the unique parse."""
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if np.count_nonzero(counts == counts[bg]) != 1:
        raise ValueError("background is not a strict mode")
    found = []
    for s in range(10):
        if s == bg or counts[s] == 0:
            continue
        objects = _parse(grid, bg, s)
        if objects:
            found.append((s, objects))
    if len(found) != 1:
        raise ValueError("scaffold color is not uniquely identifiable")
    return bg, found[0][0], found[0][1]


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid).astype(np.int64)
    bg, scaffold, objects = _roles(grid)
    out = grid.copy()
    for obj in objects:
        us, vs = _local_grid(obj["U"], obj["d"])
        rr, cc = _cells(obj["frame"], obj["top"], obj["left"], obj["U"], obj["d"], us, vs)
        out[rr, cc] = _extend(_sorted_tile(obj["raw"]), obj["U"])
    return out


# --------------------------------------------------------------------------
# Alternative-rule family used to prove the demonstrations are sufficient
# --------------------------------------------------------------------------

def _traversal_positions(obj):
    """Stream index of every local block cell for each candidate traversal."""
    p, d = obj["p"], obj["d"]
    us, vs = _local_grid(p, d)
    rr, cc = _cells(obj["frame"], obj.get("top", 0), obj.get("left", 0), obj.get("U", p + 2), d, us, vs)
    horizontal = obj["frame"][0] == "H"
    result = {}
    for trav in TRAVERSALS:
        outer, along_dir, across_dir, cs, rs, snake = trav
        along_abs = cs * cc if horizontal else rs * rr
        across_abs = rs * rr if horizontal else cs * cc
        k_along = us if along_dir == "rel+" else -us if along_dir == "rel-" else along_abs
        k_across = vs if across_dir == "rel+" else -vs if across_dir == "rel-" else across_abs
        if outer in ("col", "row"):
            along_first = (outer == "col") == horizontal
        else:
            along_first = outer == "along"
        primary, secondary = (k_along, k_across) if along_first else (k_across, k_along)
        if snake:
            step = np.searchsorted(np.unique(primary), primary)
            secondary = np.where(step % 2 == 1, -secondary, secondary)
        order = np.lexsort((secondary.ravel(), primary.ravel()))
        pos = np.empty(p * d, dtype=np.int64)
        pos[order] = np.arange(p * d)
        result[trav] = pos.reshape(p, d)
    return result


def _stream(colors, counts, scope, global_counts, reading=None):
    """Ranked color stream of one block under an alternative ranking scope."""
    if scope == "local":
        key = -counts
    elif scope == "global":
        key = -global_counts[colors]
    else:  # "first_seen": order of first appearance along the same traversal
        key = np.array([int(np.argmax(reading == c)) for c in colors])
    idx = np.lexsort((colors, key))
    return np.repeat(colors[idx], counts[idx])


def _all_variants():
    # Ascending order along a traversal equals descending order along the fully
    # reversed traversal (counts are distinct), so the reversed traversals
    # already cover every ascending reading.
    variants = [(t, s, ph, "sort") for t in TRAVERSALS
                for s in ("local", "global", "first_seen") for ph in ("block", "far", "mirror")]
    variants += [(None, None, ph, "copy") for ph in ("block", "far", "mirror")]
    variants.append((TRUE_TRAVERSAL, "local", "block", "noext"))
    return tuple(variants)


# Core alternatives must all be refuted by the training pairs.
VARIANTS = _all_variants()
# Secondary alternatives must be refuted or agree with every test output:
# restarting the period at the bar's own end differs only when a block overhangs.
EXTRA_VARIANTS = ((TRUE_TRAVERSAL, "local", "bar", "sort"),)


def _variant_band(obj, variant, global_counts, before, positions, colors, counts):
    trav, scope, phase, mode = variant
    U, p, raw = obj["U"], obj["p"], obj["raw"]
    if phase == "bar":
        phase = ("bar", obj["o"])
    if mode == "copy":
        return _extend(raw, U, phase)
    pos = positions[trav]
    reading = None
    if scope == "first_seen":
        reading = np.empty(raw.size, dtype=np.int64)
        reading[pos.ravel()] = raw.ravel()
    tile = _stream(colors, counts, scope, global_counts, reading)[pos]
    if mode == "noext":
        band = before.copy()
        band[:p] = tile
        return band
    return _extend(tile, U, phase)


def _object_context(inp, out, obj):
    us, vs = _local_grid(obj["U"], obj["d"])
    rr, cc = _cells(obj["frame"], obj["top"], obj["left"], obj["U"], obj["d"], us, vs)
    colors, counts = np.unique(obj["raw"], return_counts=True)
    return rr, cc, out[rr, cc], inp[rr, cc], _traversal_positions(obj), colors, counts


def _matching_variants(inp, out, bg, scaffold, objects, alive=None):
    """Subset of the alternative-rule family that reproduces this pair."""
    global_counts = np.bincount(inp[(inp != bg) & (inp != scaffold)].ravel(), minlength=10)
    alive = set(VARIANTS + EXTRA_VARIANTS) if alive is None else set(alive)
    for obj in objects:
        if not alive:
            break
        _, _, target, before, positions, colors, counts = _object_context(inp, out, obj)
        verdict = {}
        for variant in list(alive):
            trav, scope, phase, mode = variant
            key = (None if trav is None else positions[trav].tobytes(), scope, phase, mode)
            if key not in verdict:
                band = _variant_band(obj, variant, global_counts, before, positions, colors, counts)
                verdict[key] = np.array_equal(band, target)
            if not verdict[key]:
                alive.discard(variant)
    return alive


def _variant_output(inp, bg, scaffold, objects, variant):
    global_counts = np.bincount(inp[(inp != bg) & (inp != scaffold)].ravel(), minlength=10)
    out = inp.copy()
    for obj in objects:
        rr, cc, _, before, positions, colors, counts = _object_context(inp, inp, obj)
        out[rr, cc] = _variant_band(obj, variant, global_counts, before, positions, colors, counts)
    return out


def _fixed_order_possible(objects_lists):
    """True if one fixed color order (e.g. by color index) could explain every
    block's ranking; demonstrations must make this impossible."""
    edges = set()
    for objects in objects_lists:
        for obj in objects:
            colors, counts = np.unique(obj["raw"], return_counts=True)
            ranked = [int(c) for c in colors[np.argsort(-counts)]]
            edges.update((a, b) for i, a in enumerate(ranked) for b in ranked[i + 1:])
    graph = {}
    for a, b in edges:
        graph.setdefault(a, set()).add(b)
    state = {}

    def cyclic(node):
        state[node] = 1
        for nxt in graph.get(node, ()):
            if state.get(nxt) == 1 or (nxt not in state and cyclic(nxt)):
                return True
        state[node] = 2
        return False

    return not any(node not in state and cyclic(node) for node in list(graph))


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _resolve_counts(rng, num_train, num_test):
    def check(name, value, low, high):
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        value = int(value)
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
        return value

    num_train = check("num_train", num_train, 2, 10)
    num_test = check("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_train = int(rng.integers(3, 7))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _max_colors(n):
    k = 1
    while (k + 1) * (k + 2) // 2 <= n:
        k += 1
    return min(k, 5)


def _sample_counts(rng, n, k):
    for _ in range(300):
        roll = rng.random()
        alpha = rng.uniform(4, 12) if roll < .3 else rng.uniform(1, 3) if roll < .7 else rng.uniform(.35, 1)
        counts = rng.multinomial(n - k, rng.dirichlet(np.full(k, alpha))) + 1
        if len(set(counts.tolist())) == k:
            return counts.astype(np.int64)
    counts = np.arange(1, k + 1, dtype=np.int64)
    counts[-1] += n - counts.sum()
    return rng.permutation(counts)


def _make_raw(rng, frame, p, d, colors, counts):
    material = np.repeat(np.asarray(colors, dtype=np.int64), counts)
    target = _sorted_tile(material.reshape(p, d))
    roll = rng.random()
    if roll < .22:
        # A decoy that is already "sorted", but in a wrong frame or direction.
        positions = _traversal_positions({"frame": frame, "p": p, "d": d})
        options = [PARENT_B_TRAVERSAL, ("across", "rel+", "rel+", 1, 1, False),
                   ("along", "rel+", "rel-", 1, 1, False), ("along", "rel-", "rel+", 1, 1, False)]
        trav = options[int(rng.integers(len(options)))]
        raw = target.ravel()[positions[trav]]
        if not np.array_equal(raw, target):
            return raw
    elif roll < .40:
        raw = target.copy().ravel()
        for _ in range(int(rng.integers(1, 4))):
            a, b = rng.choice(p * d, size=2, replace=False)
            raw[a], raw[b] = raw[b], raw[a]
        raw = raw.reshape(p, d)
        if not np.array_equal(raw, target):
            return raw
    for _ in range(50):
        raw = rng.permutation(material).reshape(p, d)
        if not np.array_equal(raw, target):
            return raw
    raise ValueError("could not scramble block")


def _sample_spec(rng, cfg, *, frame, p, pool, compact, overhang_ok,
                 partial=None, palette=None, counts=None, d=None):
    if d is None:
        d_hi = 4 if compact else MAX_DEPTH
        d = int(rng.integers(2, d_hi + 1))
    if p * d < 3:
        d = 3
    n = p * d
    if palette is None:
        kmax = _max_colors(n)
        k = int(rng.integers(2, kmax + 1)) if kmax > 2 else 2
        palette = [int(x) for x in rng.choice(pool, size=k, replace=False)]
        counts = _sample_counts(rng, n, k)
    o = int(rng.integers(1, p)) if overhang_ok and p >= 2 and rng.random() < .45 else 0
    u_hi = 10 if compact else MAX_BAND
    lengths = [U for U in range(p + 2, max(p + 2, u_hi) + 1)
               if partial is None or p == 1 or (U % p != 0) == partial]
    U = int(lengths[int(rng.integers(len(lengths)))])
    raw = _make_raw(rng, frame, p, d, palette, counts)
    return {"frame": frame, "p": int(p), "d": int(d), "U": U, "o": o, "raw": raw}


def _choose_p(rng, max_p, low=1):
    weights = np.array([.18, .36, .3, .16])[low - 1:max_p]
    return int(rng.choice(np.arange(low, max_p + 1), p=weights / weights.sum()))


def _place_and_render(rng, cfg, specs, compact):
    boxes = [_bbox(s["frame"], s["U"], s["d"]) for s in specs]
    need_h = max(5, max(b[0] for b in boxes))
    need_w = max(5, max(b[1] for b in boxes))
    area = sum((h + 1) * (w + 1) for h, w in boxes)
    order = [int(i) for i in rng.permutation(len(specs))]
    for attempt in range(70):
        spread = (2 if compact else 7) + 3 * (len(specs) - 1) + attempt // 5
        H = int(rng.integers(need_h, min(30, need_h + spread) + 1))
        W = int(rng.integers(need_w, min(30, need_w + spread) + 1))
        if H * W < 1.2 * area:
            continue
        occupied = np.zeros((H, W), dtype=bool)
        placed = [None] * len(specs)
        for i in order:
            h, w = boxes[i]
            cands = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)
                     if not occupied[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1].any()]
            if not cands:
                break
            r, c = cands[int(rng.integers(len(cands)))]
            occupied[r:r + h, c:c + w] = True
            placed[i] = dict(specs[i], top=r, left=c)
        else:
            inp, out = _render((H, W), cfg["bg"], cfg["scaffold"], placed)
            counts = np.bincount(inp.ravel(), minlength=10)
            if counts[cfg["bg"]] <= np.max(np.delete(counts, cfg["bg"])):
                continue
            try:
                bg, scaffold, _ = _roles(inp)
            except ValueError:
                continue
            if bg != cfg["bg"] or scaffold != cfg["scaffold"] or not np.array_equal(solve(inp), out):
                continue
            return {"input": inp, "output": out}
    return None


def _random_frame(rng, frames, axis=None):
    options = [f for f in frames if axis is None or f[0] == axis]
    return options[int(rng.integers(len(options)))]


def _conflict_specs(rng, cfg, frames, compact):
    """Two assemblies sharing two colors whose local ranks are opposite."""
    pool = cfg["train_pool"]
    x, y, *rest = [int(c) for c in rng.permutation(pool)]
    axes = ["H", "V"] if rng.random() < .5 else ["V", "H"]
    specs = []
    for i, axis in enumerate(axes):
        p = _choose_p(rng, cfg["max_train_p"], low=2)
        d = int(rng.integers(2, (4 if compact else 6) + 1))
        if p * d < 3:
            d = 3
        n = p * d
        k = int(rng.integers(2, _max_colors(n) + 1))
        extra = [int(c) for c in rng.choice(rest, size=k - 2, replace=False)] if k > 2 else []
        palette = [x, y] + extra
        counts = _sample_counts(rng, n, k)
        if (i == 0) != (counts[0] > counts[1]):
            counts[0], counts[1] = counts[1], counts[0]
        specs.append(_sample_spec(rng, cfg, frame=_random_frame(rng, frames, axis), p=p,
                                  pool=pool, compact=compact, overhang_ok=False,
                                  palette=palette, counts=counts, d=d))
    return specs


def _twin_specs(rng, cfg, frames, compact, max_p):
    """Same histogram, different scramble and frame: identical local result."""
    p = _choose_p(rng, max_p, low=2 if max_p >= 2 else 1)
    d = int(rng.integers(2, (4 if compact else 6) + 1))
    if p * d < 3:
        d = 3
    n = p * d
    k = int(rng.integers(2, _max_colors(n) + 1))
    palette = [int(c) for c in rng.choice(cfg["train_pool"], size=k, replace=False)]
    counts = _sample_counts(rng, n, k)
    first = _random_frame(rng, frames)
    second = _random_frame(rng, [f for f in frames if f != first])
    return [_sample_spec(rng, cfg, frame=f, p=p, pool=cfg["train_pool"], compact=compact,
                         overhang_ok=False, palette=palette, counts=counts, d=d)
            for f in (first, second)]


def _phase_frame(rng, cfg, shown):
    """Frame for the partial-period demonstration that also completes the
    coverage of both bar sides and both anchored ends among multi-slice blocks."""
    sides = {f[1] for f in shown}
    anchors = {f[2] for f in shown}
    options = [f for f in cfg["train_frames"]
               if (len(sides) == 2 or f[1] not in sides) and (len(anchors) == 2 or f[2] not in anchors)]
    preferred = [f for f in options if f != CANONICAL_FRAME]
    options = preferred or options
    return options[int(rng.integers(len(options)))]


def _train_pair(rng, cfg, role, shown=()):
    frames = cfg["train_frames"]
    compact = bool(rng.random() < .35)
    pool = cfg["train_pool"]
    if role == "conflict":
        specs = _conflict_specs(rng, cfg, frames, compact)
    elif role == "phase":
        p = _choose_p(rng, cfg["max_train_p"], low=2)
        specs = [_sample_spec(rng, cfg, frame=_phase_frame(rng, cfg, shown), p=p, pool=pool,
                              compact=compact, overhang_ok=cfg["overhang"], partial=True)]
        if cfg["overhang"] and specs[0]["o"] == 0:
            specs[0]["o"] = int(rng.integers(1, p))
        if cfg["max_train_objects"] >= 2 and rng.random() < .4:
            specs.append(_sample_spec(rng, cfg, frame=_random_frame(rng, frames),
                                      p=_choose_p(rng, cfg["max_train_p"]), pool=pool,
                                      compact=compact, overhang_ok=cfg["overhang"]))
    else:
        weights = np.array([.45, .37, .18])[:cfg["max_train_objects"]]
        count = int(rng.choice(np.arange(1, cfg["max_train_objects"] + 1), p=weights / weights.sum()))
        if count >= 2 and rng.random() < .3:
            specs = _twin_specs(rng, cfg, frames, compact, cfg["max_train_p"])
            count -= 2
        else:
            specs = []
        for _ in range(count):
            specs.append(_sample_spec(rng, cfg, frame=_random_frame(rng, frames),
                                      p=_choose_p(rng, cfg["max_train_p"]), pool=pool,
                                      compact=compact, overhang_ok=cfg["overhang"]))
    return _place_and_render(rng, cfg, specs, compact), specs


def _test_pair(rng, cfg):
    compact = bool(rng.random() < .25)
    shifts = cfg["shifts"]
    pool = cfg["test_pool"]
    max_p = cfg["max_train_p"] + (1 if "longer_period" in shifts else 0)
    if "more_objects" in shifts:
        count = 3
    else:
        weights = np.array([.4, .4, .2])[:cfg["max_train_objects"]]
        count = int(rng.choice(np.arange(1, cfg["max_train_objects"] + 1), p=weights / weights.sum()))
    frames = list(FRAMES)
    specs = []
    if "unseen_frame" in shifts:
        specs.append(_sample_spec(rng, cfg, frame=cfg["excluded_frame"], p=_choose_p(rng, max_p),
                                  pool=pool, compact=compact, overhang_ok=cfg["overhang"]))
    if "longer_period" in shifts:
        specs.append(_sample_spec(rng, cfg, frame=_random_frame(rng, frames), p=max_p, pool=pool,
                                  compact=compact, overhang_ok=cfg["overhang"]))
    while len(specs) < count:
        specs.append(_sample_spec(rng, cfg, frame=_random_frame(rng, frames),
                                  p=_choose_p(rng, max_p), pool=pool, compact=compact,
                                  overhang_ok=cfg["overhang"]))
    if "novel_colors" in shifts:
        # Recolor one color of one block with a color never shown in training.
        target = specs[int(rng.integers(len(specs)))]
        used = {int(c) for s in specs for c in np.unique(s["raw"])}
        if not used & set(cfg["reserved"]):
            fresh = cfg["reserved"][int(rng.integers(len(cfg["reserved"])))]
            old = int(target["raw"].ravel()[int(rng.integers(target["raw"].size))])
            target["raw"] = np.where(target["raw"] == old, fresh, target["raw"])
    return _place_and_render(rng, cfg, specs, compact)


def _puzzle_config(rng):
    bg = 0 if rng.random() < .5 else int(rng.integers(10))
    scaffold = int(rng.choice([c for c in range(10) if c != bg]))
    pool = [int(c) for c in rng.permutation([c for c in range(10) if c not in (bg, scaffold)])]
    # Puzzle-level train/test shift: each test pair carries at least one of these.
    kinds = ["unseen_frame", "longer_period", "more_objects", "novel_colors"]
    weights = np.array([.3, .25, .2, .25])
    shifts = {kinds[int(rng.choice(4, p=weights))]}
    if rng.random() < .4:
        shifts.add(kinds[int(rng.choice(4, p=weights))])
    excluded = FRAMES[int(rng.integers(len(FRAMES)))] if "unseen_frame" in shifts else None
    if excluded == CANONICAL_FRAME and rng.random() < .5:
        excluded = FRAMES[int(rng.integers(1, len(FRAMES)))]
    return {
        "bg": bg, "scaffold": scaffold,
        "train_pool": pool[:6], "reserved": pool[6:],
        "test_pool": pool if "novel_colors" in shifts else pool[:6],
        "shifts": shifts, "excluded_frame": excluded,
        "train_frames": [f for f in FRAMES if f != excluded],
        "max_train_p": int(rng.integers(2, 4)) if "longer_period" in shifts else MAX_PERIOD,
        "max_train_objects": 2 if "more_objects" in shifts else 3,
        "overhang": bool(rng.random() < .3),
    }


def _attempt(rng, num_train, num_test):
    cfg = _puzzle_config(rng)
    roles = ["conflict", "phase"] + ["free"] * (num_train - 2)
    train = []
    seen = set()
    shown = []
    for role in roles:
        for _ in range(25):
            pair, specs = _train_pair(rng, cfg, role, shown)
            if pair is not None and pair["input"].tobytes() + bytes(pair["input"].shape) not in seen:
                break
        else:
            return None
        shown += [s["frame"] for s in specs if s["p"] >= 2]
        seen.add(pair["input"].tobytes() + bytes(pair["input"].shape))
        train.append(pair)
    order = [int(i) for i in rng.permutation(num_train)]
    train = [train[i] for i in order]
    alive = None
    for pair in train:
        bg, scaffold, objects = _roles(pair["input"])
        alive = _matching_variants(pair["input"], pair["output"], bg, scaffold, objects, alive)
    if alive & set(VARIANTS) != {TRUE_VARIANT}:
        return None
    test = []
    for _ in range(num_test):
        for _ in range(25):
            pair = _test_pair(rng, cfg)
            if pair is not None and pair["input"].tobytes() + bytes(pair["input"].shape) not in seen:
                break
        else:
            return None
        seen.add(pair["input"].tobytes() + bytes(pair["input"].shape))
        test.append(pair)
    puzzle = {"train": train, "test": test}
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        return None
    return puzzle


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    for _ in range(60):
        puzzle = _attempt(rng, num_train, num_test)
        if puzzle is not None:
            return puzzle
    raise RuntimeError(f"could not generate a valid puzzle for seed {seed}")


# --------------------------------------------------------------------------
# Validation and self-test
# --------------------------------------------------------------------------

def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen, roles = set(), set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp = np.asarray(pair["input"])
            out = np.asarray(pair["output"])
            for grid in (inp, out):
                assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = inp.astype(np.int64), out.astype(np.int64)
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = (inp.shape, inp.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            try:
                bg, scaffold, objects = _roles(inp)
            except ValueError as error:
                raise AssertionError(str(error)) from error
            roles.add((bg, scaffold))
            assert np.array_equal(solve(inp), out), "output does not follow the rule"
            assert 1 <= len(objects) <= 3
            for obj in objects:
                p, d, U, o = obj["p"], obj["d"], obj["U"], obj["o"]
                assert 1 <= p <= MAX_PERIOD and 2 <= d <= MAX_DEPTH and U <= MAX_BAND
                assert U - p >= 2 and 0 <= o < p and U - o >= 3
                colors, counts = np.unique(obj["raw"], return_counts=True)
                assert len(colors) >= 2 and len(set(counts.tolist())) == len(counts), "tied counts"
                assert bg not in colors and scaffold not in colors
                assert not np.array_equal(obj["raw"], _sorted_tile(obj["raw"])), "block already packed"
            # Omitting either parent concept must change this very pair's answer.
            ignore_b = (None, None, "block", "copy")              # repeat the raw block
            ignore_a = (TRUE_TRAVERSAL, "local", "block", "noext")  # sort but never extend
            for shortcut in (ignore_b, ignore_a):
                assert not np.array_equal(_variant_output(inp, bg, scaffold, objects, shortcut), out)
            assert np.array_equal(_variant_output(inp, bg, scaffold, objects, TRUE_VARIANT), out)
            info[split].append((inp, out, bg, scaffold, objects))
    assert len(roles) == 1, "background/scaffold roles must be puzzle-wide"
    train_objects = [o for _, _, _, _, objs in info["train"] for o in objs]
    assert {o["frame"][0] for o in train_objects} == {"H", "V"}, "training must show both axes"
    assert any(o["p"] >= 2 and o["U"] % o["p"] for o in train_objects), "no partial period in training"
    assert any(len(objs) >= 2 for *_, objs in info["train"]), "no multi-assembly training pair"
    alive = None
    for inp, out, bg, scaffold, objects in info["train"]:
        alive = _matching_variants(inp, out, bg, scaffold, objects, alive)
    core = alive & set(VARIANTS)
    assert core == {TRUE_VARIANT}, f"training leaves alternatives open: {sorted(map(str, core))[:4]}"
    for variant in alive - core:
        for inp, out, bg, scaffold, objects in info["test"]:
            assert np.array_equal(_variant_output(inp, bg, scaffold, objects, variant), out), \
                f"test output not determined by training: {variant}"
    assert not _fixed_order_possible([objs for *_, objs in info["train"]]), "a fixed color order fits"
    train_frames = {o["frame"] for o in train_objects}
    train_max_p = max(o["p"] for o in train_objects)
    train_max_n = max(len(objs) for *_, objs in info["train"])
    train_colors = {int(c) for inp, *_ in info["train"] for c in np.unique(inp)}
    train_overhang = any(o["o"] for o in train_objects)
    for inp, out, bg, scaffold, objects in info["test"]:
        novelty = (any(o["frame"] not in train_frames for o in objects)
                   or any(o["p"] > train_max_p for o in objects)
                   or len(objects) > train_max_n
                   or any(int(c) not in train_colors for c in np.unique(inp)))
        assert novelty, "test pair has no train/test shift"
        assert train_overhang or not any(o["o"] for o in objects), "overhang first appears in test"


def _reference_solve(grid):
    """Independent reconstruction: component scan, explicit sort keys, explicit loops."""
    grid = [list(map(int, row)) for row in grid]
    h, w = len(grid), len(grid[0])
    tally = {}
    for row in grid:
        for c in row:
            tally[c] = tally.get(c, 0) + 1
    bg = max(tally, key=lambda c: tally[c])
    out = [row[:] for row in grid]
    seen = set()
    scaffolds = set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg or (r, c) in seen:
                continue
            comp, stack = set(), [(r, c)]
            seen.add((r, c))
            while stack:
                y, x = stack.pop()
                comp.add((y, x))
                for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= yy < h and 0 <= xx < w and grid[yy][xx] != bg and (yy, xx) not in seen:
                        seen.add((yy, xx))
                        stack.append((yy, xx))
            rmin = min(y for y, _ in comp)
            rmax = max(y for y, _ in comp)
            cmin = min(x for _, x in comp)
            cmax = max(x for _, x in comp)
            found = []
            for edge in ("top", "bottom", "left", "right"):
                if edge in ("top", "bottom"):
                    line_r = rmin if edge == "top" else rmax
                    line = {(y, x) for y, x in comp if y == line_r}
                else:
                    line_c = cmin if edge == "left" else cmax
                    line = {(y, x) for y, x in comp if x == line_c}
                colors = {grid[y][x] for y, x in line}
                rest = comp - line
                if len(colors) != 1 or not rest:
                    continue
                color = colors.pop()
                ry = [y for y, _ in rest]
                rx = [x for _, x in rest]
                full = (max(ry) - min(ry) + 1) * (max(rx) - min(rx) + 1) == len(rest)
                if full and all(grid[y][x] != color for y, x in rest):
                    found.append((edge, color, line, rest))
            assert len(found) == 1
            edge, color, line, block = found[0]
            scaffolds.add(color)
            horizontal = edge in ("top", "bottom")
            pos = (lambda cell: cell[1]) if horizontal else (lambda cell: cell[0])
            dist = (lambda cell: abs(cell[0] - next(iter(line))[0])) if horizontal else \
                (lambda cell: abs(cell[1] - next(iter(line))[1]))
            bar_lo = min(pos(x) for x in line)
            bar_hi = max(pos(x) for x in line)
            blk_lo = min(pos(x) for x in block)
            blk_hi = max(pos(x) for x in block)
            from_lo = blk_lo <= bar_lo
            start = blk_lo if from_lo else blk_hi
            end = bar_hi if from_lo else bar_lo
            step = 1 if from_lo else -1
            depth = max(dist(x) for x in block)
            period = blk_hi - blk_lo + 1
            ordered = sorted(block, key=lambda cell: (abs(pos(cell) - start), dist(cell)))
            freq = {}
            for y, x in block:
                freq[grid[y][x]] = freq.get(grid[y][x], 0) + 1
            stream = []
            for col in sorted(freq, key=lambda col: -freq[col]):
                stream += [col] * freq[col]
            tile = {}
            for cell, col in zip(ordered, stream):
                tile[(abs(pos(cell) - start), dist(cell))] = col
            line_coord = next(iter(line))
            for a in range(start, end + step, step):
                for k in range(1, depth + 1):
                    if horizontal:
                        y = line_coord[0] + (k if edge == "top" else -k)
                        x = a
                    else:
                        y = a
                        x = line_coord[1] + (k if edge == "left" else -k)
                    out[y][x] = tile[(abs(a - start) % period, k)]
    assert len(scaffolds) <= 1
    return np.array(out, dtype=np.int64)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    frames, periods, counts_per_pair, shapes, roles = set(), set(), set(), set(), set()
    overhang = partial = twins = canonical = noncanonical_train = broad = edge = 0
    novelty_kinds = {"frame": 0, "period": 0, "objects": 0, "colors": 0}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_objs = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = np.asarray(pair["input"])
                assert np.array_equal(_reference_solve(inp), pair["output"])
                bg, scaffold, objects = _roles(inp)
                roles.add((bg, scaffold))
                shapes.add(inp.shape)
                counts_per_pair.add(len(objects))
                hist = {}
                for obj in objects:
                    frames.add(obj["frame"])
                    periods.add(obj["p"])
                    overhang += obj["o"] > 0
                    partial += obj["p"] > 1 and obj["U"] % obj["p"] != 0
                    canonical += obj["frame"] == CANONICAL_FRAME
                    broad += obj["d"] >= obj["U"] - obj["p"]
                    rr, cc = _cells(obj["frame"], obj["top"], obj["left"], obj["U"], obj["d"],
                                    *_local_grid(obj["U"], obj["d"]))
                    edge += bool(rr.min() == 0 or cc.min() == 0 or rr.max() == inp.shape[0] - 1
                                 or cc.max() == inp.shape[1] - 1)
                    key = (obj["p"], obj["d"], tuple(sorted(np.unique(obj["raw"], return_counts=True)[1])),
                           tuple(np.unique(obj["raw"])))
                    hist[key] = hist.get(key, 0) + 1
                    if split == "train":
                        train_objs.append(obj)
                        noncanonical_train += obj["frame"] != CANONICAL_FRAME
                twins += any(v > 1 for v in hist.values())
        tf = {o["frame"] for o in train_objs}
        tp = max(o["p"] for o in train_objs)
        tn = max(len(_roles(np.asarray(p["input"]))[2]) for p in puzzle["train"])
        tc = {int(c) for p in puzzle["train"] for c in np.unique(p["input"])}
        for pair in puzzle["test"]:
            _, _, objects = _roles(np.asarray(pair["input"]))
            novelty_kinds["frame"] += any(o["frame"] not in tf for o in objects)
            novelty_kinds["period"] += any(o["p"] > tp for o in objects)
            novelty_kinds["objects"] += len(objects) > tn
            novelty_kinds["colors"] += any(int(c) not in tc for c in np.unique(pair["input"]))
    assert frames == set(FRAMES) and periods == {1, 2, 3, 4} and counts_per_pair == {1, 2, 3}
    assert len(shapes) >= 60 and len(roles) >= 20
    assert overhang >= 5 and partial >= 40 and twins >= 3 and canonical >= 5 and broad >= 10 and edge >= 10
    assert all(v >= 3 for v in novelty_kinds.values()), novelty_kinds
    # Legal explicit counts, determinism and illegal-count rejection.
    for nt, ne in ((2, 1), (10, 4), (5, 3)):
        p = generate_puzzle(7000 + nt * 10 + ne, nt, ne)
        assert len(p["train"]) == nt and len(p["test"]) == ne
    a, b = generate_puzzle(31337), generate_puzzle(31337)
    assert all(np.array_equal(x[k], y[k]) for s in ("train", "test")
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (3, 3), (True, 1)):
        try:
            generate_puzzle(1, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
