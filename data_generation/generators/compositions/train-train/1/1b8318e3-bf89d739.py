"""Fused bar magnet: the aligned magnet pair fuses, then tokens dock on the nearest magnet.

Composition of two ARC families:

* Parent A (1b8318e3): solid 2x2 blocks are fixed magnets; every one-cell token
  slides straight (orthogonally or diagonally) toward its nearest magnet and
  stops in eight-neighbour contact with it, keeping its colour.
* Parent B (bf89d739): among many same-role markers exactly one pair shares a
  row or a column; the cells strictly between them are filled as a trunk and
  every other marker joins that trunk by its perpendicular projection.

New rule: among the 2x2 magnets exactly one pair is aligned (same two rows or
same two columns).  That pair fuses -- the band strictly between them is filled
with the magnet colour -- into one 2-thick bar magnet.  Then every token is
attracted to its nearest magnet, where the whole bar counts as one magnet, and
slides into contact.  Beside the bar's interior that contact point is exactly
B's perpendicular projection; near the bar ends and around the remaining lone
magnets it is A's orthogonal/diagonal docking.

Only the Python standard library and NumPy are used.
"""
from __future__ import annotations

import math

import numpy as np

# A rectangle is (top, bottom, left, right), all inclusive.

_FEATURES = ("move", "stationary", "diag", "ortho_corner", "bar_dep",
             "bar_float", "bar_end", "lone_move", "reassign")
# Token-level behaviours a scene can be asked to contain.
_TOKEN_NEEDS = ("bar_dep", "lone_move", "diag", "reassign", "bar_float",
                "ortho_corner", "bar_end", "stationary")
_MOVING_NEEDS = ("bar_dep", "lone_move", "diag", "ortho_corner", "bar_end")
# Behaviours that a test may only use after some training pair has shown them.
_TAUGHT = ("near_miss", "reassign", "bar_float", "stationary", "diag",
           "ortho_corner", "bar_end")


# --------------------------------------------------------------------------
# Counts
# --------------------------------------------------------------------------
def _checked(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(num_train, num_test):
    if num_test is None:
        num_test = 1
    num_test = _checked(num_test, "num_test", 1, 4)
    if num_train is None:
        num_train = max(4, num_test + 1)
    num_train = _checked(num_train, "num_train", 2, 10)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def _offsets(point, rect):
    """Row and column gaps from a cell to a rectangle (0 when inside the span)."""
    r, c = point
    r0, r1, c0, c1 = rect
    return max(r0 - r, r - r1, 0), max(c0 - c, c - c1, 0)


def _cheb(point, rect):
    dr, dc = _offsets(point, rect)
    return max(dr, dc)


def _landing(point, rect):
    """Nearest contact cell: the token clamped onto the magnet's one-cell halo.

    This is the smallest displacement that brings the token into eight-neighbour
    contact.  Beside a long side it is the perpendicular projection; at a corner
    it may be an orthogonal slide ending in diagonal contact.  Generated tokens
    always reach it along a straight orthogonal or diagonal ray.
    """
    r, c = point
    r0, r1, c0, c1 = rect
    return (min(max(r, r0 - 1), r1 + 1), min(max(c, c0 - 1), c1 + 1))


def _halo(rect, shape):
    r0, r1, c0, c1 = rect
    h, w = shape
    return [(r, c) for r in range(max(0, r0 - 1), min(h, r1 + 2))
            for c in range(max(0, c0 - 1), min(w, c1 + 2))
            if not (r0 <= r <= r1 and c0 <= c <= c1)]


def _separation(a, b):
    """Number of empty rows/columns between two rectangles (Chebyshev gap)."""
    rows = max(0, b[0] - a[1] - 1, a[0] - b[1] - 1)
    cols = max(0, b[2] - a[3] - 1, a[2] - b[3] - 1)
    return max(rows, cols)


def _block_rect(block):
    r, c = block
    return (r, r + 1, c, c + 1)


def _path(start, end):
    """Cells entered while sliding from start to end (excludes start)."""
    dr, dc = end[0] - start[0], end[1] - start[1]
    steps = max(abs(dr), abs(dc))
    if steps == 0:
        return []
    if not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        return None
    sr, sc = (dr > 0) - (dr < 0), (dc > 0) - (dc < 0)
    return [(start[0] + sr * k, start[1] + sc * k) for k in range(1, steps + 1)]


# --------------------------------------------------------------------------
# Input-only parsing and solving
# --------------------------------------------------------------------------
def _components(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack, comp = [(r, c)], []
                while stack:
                    y, x = stack.pop()
                    comp.append((y, x))
                    for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                result.append(comp)
    return result


def _is_square(comp):
    if len(comp) != 4:
        return False
    r = min(y for y, _ in comp)
    c = min(x for _, x in comp)
    return set(comp) == {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}


def _parse(grid):
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")
    values, counts = np.unique(g, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    if len(values) > 1 and counts[order[0]] == counts[order[1]]:
        raise ValueError("no dominant background")
    bg = int(values[order[0]])
    magnet_colors = []
    for value in values:
        value = int(value)
        if value == bg:
            continue
        comps = _components(g == value)
        if len(comps) >= 2 and all(_is_square(comp) for comp in comps):
            magnet_colors.append((value, comps))
    if len(magnet_colors) != 1:
        raise ValueError("expected exactly one colour made of 2x2 magnets")
    magnet, comps = magnet_colors[0]
    blocks = sorted((min(y for y, _ in comp), min(x for _, x in comp)) for comp in comps)
    aligned = [(i, j) for i in range(len(blocks)) for j in range(i + 1, len(blocks))
               if blocks[i][0] == blocks[j][0] or blocks[i][1] == blocks[j][1]]
    if len(aligned) != 1:
        raise ValueError("expected exactly one aligned magnet pair")
    i, j = aligned[0]
    a, b = blocks[i], blocks[j]
    if a[0] == b[0]:
        orientation = "horizontal"
        bar = (a[0], a[0] + 1, min(a[1], b[1]), max(a[1], b[1]) + 1)
    else:
        orientation = "vertical"
        bar = (min(a[0], b[0]), max(a[0], b[0]) + 1, a[1], a[1] + 1)
    lone = [blocks[k] for k in range(len(blocks)) if k not in (i, j)]
    tokens = [(int(r), int(c), int(g[r, c]))
              for r, c in zip(*np.nonzero((g != bg) & (g != magnet)))]
    return {"grid": g, "shape": g.shape, "bg": bg, "magnet": magnet,
            "blocks": blocks, "poles": (a, b), "orientation": orientation,
            "bar": bar, "lone": lone,
            "attractors": [bar] + [_block_rect(x) for x in lone],
            "tokens": tokens}


def _nearest(point, rects):
    distances = [_cheb(point, rect) for rect in rects]
    best = min(distances)
    winners = [k for k, d in enumerate(distances) if d == best]
    return winners


def _paint_magnets(scene, fuse=True):
    out = np.full(scene["shape"], scene["bg"], dtype=int)
    rects = scene["attractors"] if fuse else [_block_rect(x) for x in scene["blocks"]]
    for r0, r1, c0, c1 in rects:
        out[r0:r1 + 1, c0:c1 + 1] = scene["magnet"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fuse the uniquely aligned magnet pair, then dock every token on its nearest magnet."""
    scene = _parse(input_grid)
    out = _paint_magnets(scene, fuse=True)
    rects = scene["attractors"]
    for r, c, color in scene["tokens"]:
        winners = _nearest((r, c), rects)
        if len(winners) != 1:
            raise ValueError("token has no unique nearest magnet")
        end = _landing((r, c), rects[winners[0]])
        if out[end] != scene["bg"]:
            raise ValueError("two tokens dock on the same cell")
        out[end] = color
    return out


# --------------------------------------------------------------------------
# Shortcut predictors (used to prove that both parents matter)
# --------------------------------------------------------------------------
def _shortcut(scene, name):
    """Predicted output for a simpler rule; None when that rule is ambiguous."""
    bg = scene["bg"]
    blocks = [_block_rect(x) for x in scene["blocks"]]
    poles = {scene["blocks"].index(p) for p in scene["poles"]}
    bar = scene["bar"]
    out = _paint_magnets(scene, fuse=(name != "no_fusion"))
    for r, c, color in scene["tokens"]:
        point = (r, c)
        if name == "static_fusion":          # B only: fuse, tokens never move
            end = point
        elif name == "bar_only":             # every token docks on the bar
            end = _landing(point, bar)
        elif name in ("no_fusion", "inert_bar"):   # A only (bar absent / decorative)
            winners = _nearest(point, blocks)
            if len(winners) != 1:
                return None
            end = _landing(point, blocks[winners[0]])
        elif name == "prefusion_choice":     # choose magnet before fusing
            winners = _nearest(point, blocks)
            if len(winners) != 1 and not set(winners) <= poles:
                return None
            if winners[0] in poles:
                end = _landing(point, bar)
            else:
                end = _landing(point, blocks[winners[0]])
        else:
            raise KeyError(name)
        if out[end] != bg and end != point:
            return None
        out[end] = color
    return out


_SHORTCUTS = ("static_fusion", "no_fusion", "inert_bar", "bar_only", "prefusion_choice")
# Axes on which every test must exceed every training pair (canvas area, token
# count, bar length, lone-magnet count); the generator caps training on one.
_SHIFT_AXES = (lambda f: f["shape"][0] * f["shape"][1], lambda f: f["tokens"],
               lambda f: f["bar_len"], lambda f: f["lone"])


# --------------------------------------------------------------------------
# Independent pair analysis (from returned grids only)
# --------------------------------------------------------------------------
def _metric_owner(point, rects):
    """Index of the nearest rectangle if Chebyshev, Manhattan and Euclid agree."""
    owners = []
    for metric in (lambda dr, dc: max(dr, dc), lambda dr, dc: dr + dc,
                   lambda dr, dc: dr * dr + dc * dc):
        values = [metric(*_offsets(point, rect)) for rect in rects]
        best = min(values)
        if values.count(best) != 1:
            return -1
        owners.append(values.index(best))
    return owners[0] if len(set(owners)) == 1 else -1


def _near_miss(blocks):
    return any(abs(a[0] - b[0]) == 1 or abs(a[1] - b[1]) == 1
               for k, a in enumerate(blocks) for b in blocks[k + 1:])


def _closer_pair(scene):
    a, b = scene["poles"]
    aligned = math.hypot(a[0] - b[0], a[1] - b[1])
    blocks = scene["blocks"]
    for k, x in enumerate(blocks):
        for y in blocks[k + 1:]:
            if {x, y} == {a, b}:
                continue
            if math.hypot(x[0] - y[0], x[1] - y[1]) < aligned:
                return True
    return False


def _analyze_pair(input_grid, output_grid):
    inp = np.asarray(input_grid, dtype=int)
    out = np.asarray(output_grid, dtype=int)
    assert inp.shape == out.shape, "shape changed"
    scene = _parse(inp)
    h, w = inp.shape
    bg = scene["bg"]
    counts = np.bincount(inp.ravel(), minlength=10)
    assert counts[bg] * 2 > inp.size, "background must cover most of the input"
    out_counts = np.bincount(out.ravel(), minlength=10)
    assert out_counts[bg] == out_counts.max() and \
        sorted(out_counts).count(out_counts[bg]) == 1, "background not dominant in output"
    rects = scene["attractors"]
    for k, x in enumerate(rects):
        for y in rects[k + 1:]:
            assert _separation(x, y) >= 2, "magnets too close"
    bar = scene["bar"]
    band = inp[bar[0]:bar[1] + 1, bar[2]:bar[3] + 1]
    assert int(np.count_nonzero(band == scene["magnet"])) == 8 and \
        int(np.count_nonzero(band == bg)) == band.size - 8, "fusion band must be empty"
    gap = (bar[3] - bar[2] + 1 - 4) if scene["orientation"] == "horizontal" else (bar[1] - bar[0] + 1 - 4)
    assert gap >= 2, "aligned pair too close"
    tokens = scene["tokens"]
    assert 2 <= len(tokens) <= 16, "token count outside the family"
    starts = {(r, c) for r, c, _ in tokens}
    for r, c, _ in tokens:
        assert not any((r, c) != t and max(abs(r - t[0]), abs(c - t[1])) <= 1
                       for t in starts), "tokens must be isolated"
    expected = solve(inp)
    assert np.array_equal(expected, out), "output disagrees with fused-magnet rule"
    blocks = [_block_rect(x) for x in scene["blocks"]]
    pole_rects = [_block_rect(x) for x in scene["poles"]]
    lone_ids = [k for k, x in enumerate(scene["blocks"]) if x not in scene["poles"]]
    moves = []
    for r, c, color in tokens:
        owner = _metric_owner((r, c), rects)
        assert owner >= 0, "nearest magnet depends on the distance metric"
        end = _landing((r, c), rects[owner])
        path = _path((r, c), end)
        assert path is not None, "token path is not a straight ray"
        moves.append(((r, c), end, owner, color, path))
    ends = [m[1] for m in moves]
    assert len(set(ends)) == len(ends), "landing collision"
    for k, (start, end, owner, color, path) in enumerate(moves):
        others = {m[0] for i, m in enumerate(moves) if i != k} | \
                 {m[1] for i, m in enumerate(moves) if i != k}
        assert not any(p in others for p in path), "token path crosses another token"
        assert all(inp[p] == bg for p in path), "token path is blocked"
    feats = {name: 0 for name in _FEATURES}
    pole_halo = set(_halo(pole_rects[0], inp.shape)) | set(_halo(pole_rects[1], inp.shape))
    max_travel = 0
    for start, end, owner, color, path in moves:
        travel = len(path)
        max_travel = max(max_travel, travel)
        dr, dc = end[0] - start[0], end[1] - start[1]
        feats["move"] += travel > 0
        feats["stationary"] += travel == 0
        feats["diag"] += bool(dr and dc)
        target_rect = rects[owner]
        corner = (end[0] in (target_rect[0] - 1, target_rect[1] + 1) and
                  end[1] in (target_rect[2] - 1, target_rect[3] + 1))
        feats["ortho_corner"] += travel > 0 and corner and not (dr and dc)
        if owner == 0:
            if end not in pole_halo:
                feats["bar_dep" if travel else "bar_float"] += 1
            elif travel:
                feats["bar_end"] += 1
            pre = _nearest(start, blocks)
            if any(x in lone_ids for x in pre):
                feats["reassign"] += 1
        elif travel:
            feats["lone_move"] += 1
    shortcut_differs = {}
    for name in _SHORTCUTS:
        prediction = _shortcut(scene, name)
        shortcut_differs[name] = prediction is None or not np.array_equal(prediction, out)
    feats.update({
        "orientation": scene["orientation"], "shape": inp.shape,
        "bg": bg, "magnet": scene["magnet"], "lone": len(scene["lone"]),
        "tokens": len(tokens), "gap": gap,
        "bar_len": gap + 4, "max_travel": max_travel,
        "token_colors": tuple(sorted({c for _, _, c in tokens})),
        "near_miss": _near_miss(scene["blocks"]),
        "closer_pair": _closer_pair(scene),
        "edge_bar": bar[0] == 0 or bar[2] == 0 or bar[1] == h - 1 or bar[3] == w - 1,
        "full_span": (bar[2] == 0 and bar[3] == w - 1) if scene["orientation"] == "horizontal"
        else (bar[0] == 0 and bar[1] == h - 1),
        "shortcuts": shortcut_differs,
    })
    return feats


# --------------------------------------------------------------------------
# Scene construction
# --------------------------------------------------------------------------
def _dims(rng, regime):
    if regime == "train":
        band = int(rng.choice(4, p=(0.28, 0.44, 0.16, 0.12)))
    else:
        band = int(rng.choice(4, p=(0.04, 0.30, 0.50, 0.16)))
    if band == 0:
        return int(rng.integers(8, 13)), int(rng.integers(8, 14))
    if band == 1:
        return int(rng.integers(11, 21)), int(rng.integers(11, 22))
    if band == 2:
        return int(rng.integers(17, 31)), int(rng.integers(16, 31))
    short, long = int(rng.integers(8, 14)), int(rng.integers(18, 31))
    return (short, long) if rng.random() < 0.5 else (long, short)


def _place_lone(rng, shape, bar, poles, count, near_policy, force_near, want_closer):
    h, w = shape
    blocks = list(poles)
    rects = [bar]
    gap = bar[3] - bar[2] - 3
    aligned_dist = gap + 2

    def acceptable(block):
        r, c = block
        if not (0 <= r <= h - 2 and 0 <= c <= w - 2):
            return False
        rect = _block_rect(block)
        if any(_separation(rect, other) < 2 for other in rects):
            return False
        if any(r == b[0] or c == b[1] for b in blocks):
            return False
        near = any(abs(r - b[0]) == 1 or abs(c - b[1]) == 1 for b in blocks)
        return not (near and near_policy == "forbid")

    def add(block):
        blocks.append(block)
        rects.append(_block_rect(block))

    if force_near and count:
        for _ in range(200):
            anchor = blocks[int(rng.integers(0, len(blocks)))]
            if rng.random() < 0.5:
                block = (anchor[0] + int(rng.choice((-1, 1))), int(rng.integers(0, w - 1)))
            else:
                block = (int(rng.integers(0, h - 1)), anchor[1] + int(rng.choice((-1, 1))))
            if acceptable(block):
                add(block)
                break
        else:
            return None
    if want_closer and len(blocks) - 2 < count:
        for _ in range(200):
            anchor = blocks[int(rng.integers(0, len(blocks)))]
            reach = max(4, min(aligned_dist - 1, 9))
            block = (anchor[0] + int(rng.integers(-reach, reach + 1)),
                     anchor[1] + int(rng.integers(-reach, reach + 1)))
            if (acceptable(block) and
                    math.hypot(block[0] - anchor[0], block[1] - anchor[1]) < aligned_dist):
                add(block)
                break
    tries = 0
    while len(blocks) - 2 < count and tries < 400:
        tries += 1
        block = (int(rng.integers(0, h - 1)), int(rng.integers(0, w - 1)))
        if acceptable(block):
            add(block)
    return blocks[2:]


def _owner_map(shape, rects):
    """Unique nearest-rectangle index agreed by three metrics, else -1."""
    h, w = shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    dr = np.stack([np.broadcast_to(np.maximum(np.maximum(r0 - rows, rows - r1), 0), (h, w))
                   for r0, r1, _, _ in rects])
    dc = np.stack([np.broadcast_to(np.maximum(np.maximum(c0 - cols, cols - c1), 0), (h, w))
                   for _, _, c0, c1 in rects])
    owner = None
    valid = np.ones((h, w), dtype=bool)
    for values in (np.maximum(dr, dc), dr + dc, dr * dr + dc * dc):
        best = values.min(axis=0)
        unique = (values == best).sum(axis=0) == 1
        arg = values.argmin(axis=0)
        valid &= unique
        if owner is None:
            owner = arg
        else:
            valid &= arg == owner
    result = np.where(valid, owner, -1)
    for r0, r1, c0, c1 in rects:
        result[r0:r1 + 1, c0:c1 + 1] = -2
    return result


def _candidates(shape, rects, poles, lone_blocks):
    h, w = shape
    owner = _owner_map(shape, rects)
    block_rects = [_block_rect(p) for p in poles] + [_block_rect(x) for x in lone_blocks]
    pole_halo = set(_halo(block_rects[0], shape)) | set(_halo(block_rects[1], shape))
    found = []
    for index, rect in enumerate(rects):
        r0, r1, c0, c1 = rect
        for end in _halo(rect, shape):
            vr = -1 if end[0] < r0 else (1 if end[0] > r1 else 0)
            vc = -1 if end[1] < c0 else (1 if end[1] > c1 else 0)
            directions = [(0, 0)]
            if vr and vc:
                directions += [(vr, vc), (vr, 0), (0, vc)]
            elif vr:
                directions.append((vr, 0))
            else:
                directions.append((0, vc))
            for dr, dc in directions:
                for k in range(0, 31):
                    if (dr, dc) == (0, 0) and k > 0:
                        break
                    if (dr, dc) != (0, 0) and k == 0:
                        continue
                    start = (end[0] + dr * k, end[1] + dc * k)
                    if not (0 <= start[0] < h and 0 <= start[1] < w):
                        break
                    if owner[start] != index:
                        continue
                    feats = set()
                    if k == 0:
                        feats.add("stationary")
                    else:
                        feats.add("move")
                        if dr and dc:
                            feats.add("diag")
                        elif vr and vc:
                            feats.add("ortho_corner")
                    if index == 0:
                        if end not in pole_halo:
                            feats.add("bar_dep" if k else "bar_float")
                        elif k:
                            feats.add("bar_end")
                        dists = [_cheb(start, rect2) for rect2 in block_rects]
                        best = min(dists)
                        if any(d == best for d in dists[2:]):
                            feats.add("reassign")
                    elif k:
                        feats.add("lone_move")
                    found.append({"start": start, "end": end, "owner": index, "k": k,
                                  "feats": feats, "path": _path(start, end)})
    return found


def _select_tokens(rng, cands, count, needs):
    selected = []
    starts, ends, blocked = set(), set(), set()
    weights = {}
    for cand in cands:
        weights.setdefault(cand["owner"], float(rng.uniform(0.35, 2.0)))

    def compatible(cand):
        s, e = cand["start"], cand["end"]
        if s in starts or s in ends or e in ends or e in starts:
            return False
        if s in blocked or e in blocked:
            return False
        if any(p in starts or p in ends for p in cand["path"]):
            return False
        return not any(max(abs(s[0] - t[0]), abs(s[1] - t[1])) <= 1 for t in starts)

    def take(pool):
        pool = [x for x in pool if compatible(x)]
        if not pool:
            return False
        probabilities = np.array([weights[x["owner"]] for x in pool], dtype=float)
        pick = pool[int(rng.choice(len(pool), p=probabilities / probabilities.sum()))]
        selected.append(pick)
        starts.add(pick["start"])
        ends.add(pick["end"])
        blocked.update(pick["path"])
        return True

    for need in needs:
        if not take([x for x in cands if need in x["feats"]]):
            return None
    attempts = 0
    while len(selected) < count and attempts < 4 * count + 20:
        attempts += 1
        travel = int(rng.choice(3, p=(0.12, 0.40, 0.48)))
        if travel == 0:
            pool = [x for x in cands if x["k"] == 0]
        elif travel == 1:
            pool = [x for x in cands if 1 <= x["k"] <= 2]
        else:
            pool = [x for x in cands if x["k"] >= 3]
        if not take(pool):
            take(cands)
    return selected


def _token_colors(rng, allowed, count):
    size = int(rng.integers(1, min(len(allowed), count, 6) + 1))
    palette = [int(x) for x in rng.choice(allowed, size=size, replace=False)]
    colors = palette + [palette[int(rng.integers(0, size))] for _ in range(count - size)]
    rng.shuffle(colors)
    return colors


def _scene(rng, spec):
    """Build one pair satisfying spec, or return None."""
    regime = spec["regime"]
    needs = spec["needs"]
    caps = spec.get("caps", {})
    for _ in range(60):
        height, width = _dims(rng, regime)
        if height * width >= caps.get("area", 10 ** 6):
            continue
        h, w = (height, width) if spec["orientation"] == "horizontal" else (width, height)
        gmax = w - 4
        if gmax < 3 or h < 6:
            continue
        mode = int(rng.choice(3, p=spec["span_p"]))
        low, high = ((0.3, 0.55), (0.55, 0.8), (0.8, 1.0))[mode]
        gap = int(round(gmax * float(rng.uniform(low, high))))
        gap = max(3 if gmax >= 3 else 2, min(gmax, gap, caps.get("bar_len", 99) - 5))
        if gap < 3 or gap > gmax:
            continue
        roll = rng.random()
        last = w - 4 - gap
        c1 = 0 if roll < 0.12 else (last if roll < 0.24 else int(rng.integers(0, last + 1)))
        roll = rng.random()
        r = 0 if roll < 0.08 else (h - 2 if roll < 0.16 else int(rng.integers(0, h - 1)))
        poles = [(r, c1), (r, c1 + gap + 2)]
        bar = (r, r + 1, c1, c1 + gap + 3)
        lone_count = min(int(spec["lone"](rng)), caps.get("lone", 99) - 1)
        if "lone_move" in needs:
            lone_count = max(1, lone_count)
        lone = _place_lone(rng, (h, w), bar, poles, lone_count, spec["near_policy"],
                           "near_miss" in needs, "closer_pair" in needs)
        if lone is None or len(lone) < (1 if "lone_move" in needs else 0):
            continue
        if "near_miss" in needs and not _near_miss(poles + lone):
            continue
        rects = [bar] + [_block_rect(x) for x in lone]
        cands = _candidates((h, w), rects, poles, lone)
        if spec["near_policy"] == "forbid" and _near_miss(poles + lone):
            continue
        if not spec["reassign_ok"]:
            cands = [x for x in cands if "reassign" not in x["feats"]]
        count = min(int(spec["tokens"](rng)), caps.get("tokens", 99) - 1)
        token_needs = [n for n in _TOKEN_NEEDS if n in needs]
        if not any(n in _MOVING_NEEDS for n in token_needs):
            token_needs.append("move")
        chosen = _select_tokens(rng, cands, max(count, len(token_needs)), token_needs)
        if chosen is None or not 2 <= len(chosen) < caps.get("tokens", 99):
            continue
        canvas = np.full((h, w), spec["bg"], dtype=int)
        for pr, pc in poles + lone:
            canvas[pr:pr + 2, pc:pc + 2] = spec["magnet"]
        result = np.full((h, w), spec["bg"], dtype=int)
        for r0, r1, a0, a1 in rects:
            result[r0:r1 + 1, a0:a1 + 1] = spec["magnet"]
        colors = _token_colors(rng, spec["token_colors"], len(chosen))
        for cand, color in zip(chosen, colors):
            canvas[cand["start"]] = color
            result[cand["end"]] = color
        if spec["orientation"] == "vertical":
            canvas, result = canvas.T.copy(), result.T.copy()
        if canvas.shape != (height, width):
            continue
        try:
            feats = _analyze_pair(canvas, result)
        except (AssertionError, ValueError):
            continue
        if "closer_pair" in needs and not feats["closer_pair"]:
            continue
        if spec["near_policy"] == "forbid" and feats["near_miss"]:
            continue
        if not spec["reassign_ok"] and feats["reassign"]:
            continue
        if any(feats[n] == 0 for n in token_needs):
            continue
        if feats["lone"] >= caps.get("lone", 99) or feats["bar_len"] >= caps.get("bar_len", 99):
            continue
        if regime == "test" and not (feats["shortcuts"]["inert_bar"] and
                                     feats["shortcuts"]["bar_only"]):
            continue
        return {"input": canvas.astype(np.int64), "output": result.astype(np.int64)}, feats
    return None


# --------------------------------------------------------------------------
# Puzzle assembly
# --------------------------------------------------------------------------
def _puzzle_attempt(rng, num_train, num_test):
    level = str(rng.choice(("simple", "moderate", "rich"), p=(0.3, 0.4, 0.3)))
    near_ok = bool(rng.random() < 0.5)
    reassign_ok = bool(rng.random() < 0.65)
    scheme = str(rng.choice(("classic", "fixed", "pair"), p=(0.25, 0.25, 0.5)))
    puzzle_bg = 0 if scheme == "classic" else int(rng.integers(0, 10))
    puzzle_magnet = int(rng.choice([c for c in range(10) if c != puzzle_bg]))
    unseen = int(rng.integers(0, 10)) if rng.random() < 0.4 else None

    lone_train = {"simple": (0.15, 0.55, 0.25, 0.05, 0.0, 0.0),
                  "moderate": (0.08, 0.35, 0.32, 0.17, 0.08, 0.0),
                  "rich": (0.05, 0.2, 0.3, 0.25, 0.12, 0.08)}[level]
    lone_test = {"simple": (0.0, 0.55, 0.35, 0.1, 0.0, 0.0),
                 "moderate": (0.0, 0.3, 0.35, 0.2, 0.15, 0.0),
                 "rich": (0.0, 0.15, 0.25, 0.25, 0.2, 0.15)}[level]
    token_train = {"simple": (2, 6), "moderate": (3, 9), "rich": (4, 12)}[level]
    token_test = {"simple": (3, 8), "moderate": (5, 12), "rich": (6, 16)}[level]

    # Tests are built first; training is then obliged to demonstrate every
    # behaviour the tests rely on (orientation, near misses, reassignment ...).
    test_orient = [str(rng.choice(("horizontal", "vertical"))) for _ in range(num_test)]
    if num_test >= 2 and rng.random() < 0.5:
        test_orient[1] = "vertical" if test_orient[0] == "horizontal" else "horizontal"
    test_needs = []
    for _ in range(num_test):
        need = {"move", "bar_dep", "lone_move"}
        if reassign_ok and rng.random() < 0.5:
            need.add("reassign")
        if near_ok and rng.random() < 0.4:
            need.add("near_miss")
        if rng.random() < 0.3:
            need.add("diag")
        test_needs.append(need)

    colors_used = {"previous": None}

    def pair_colors(split):
        if scheme == "pair":
            bg = int(rng.choice([c for c in range(10) if c != colors_used["previous"]]))
            magnet = int(rng.choice([c for c in range(10) if c != bg]))
            colors_used["previous"] = bg
        else:
            bg, magnet = puzzle_bg, puzzle_magnet
        allowed = [c for c in range(10) if c not in (bg, magnet)]
        if split == "train" and unseen in allowed:
            allowed.remove(unseen)
        return bg, magnet, allowed

    seen = set()

    def build(split, orientation, needs, caps=None):
        bg, magnet, allowed = pair_colors(split)
        low, high = token_train if split == "train" else token_test
        lone_p = lone_train if split == "train" else lone_test
        spec = {
            "regime": split, "orientation": orientation, "needs": needs,
            "bg": bg, "magnet": magnet, "token_colors": allowed,
            "near_policy": "allow" if near_ok else "forbid",
            "reassign_ok": reassign_ok,
            "span_p": (0.3, 0.4, 0.3) if split == "train" else (0.15, 0.4, 0.45),
            "lone": (lambda g, p=lone_p: g.choice(6, p=p)),
            "tokens": (lambda g, lo=low, hi=high: g.integers(lo, hi + 1)),
            "caps": caps or {},
        }
        for _ in range(6):
            built = _scene(np.random.default_rng(int(rng.integers(0, 2 ** 63))), spec)
            if built is None:
                continue
            pair, feats = built
            key = (pair["input"].shape, pair["input"].tobytes())
            if key in seen:
                continue
            seen.add(key)
            return pair, feats
        return None

    tests = []
    for orientation, needs in zip(test_orient, test_needs):
        built = build("test", orientation, needs)
        if built is None:
            return None
        tests.append(built)

    # Train/test shift: every test exceeds all training pairs along one axis.
    least = {"area": min(f["shape"][0] * f["shape"][1] for _, f in tests),
             "tokens": min(f["tokens"] for _, f in tests),
             "bar_len": min(f["bar_len"] for _, f in tests),
             "lone": min(f["lone"] for _, f in tests)}
    feasible = [axis for axis, floor in (("area", 260), ("tokens", 8), ("bar_len", 11),
                                         ("lone", 2)) if least[axis] >= floor]
    if not feasible:
        return None
    axis = feasible[int(rng.integers(0, len(feasible)))]
    caps = {axis: least[axis]}

    first = str(rng.choice(("horizontal", "vertical")))
    other = "vertical" if first == "horizontal" else "horizontal"
    train_orient = [first, other if rng.random() < 0.75 else first]
    train_orient += [str(rng.choice(("horizontal", "vertical"))) for _ in range(num_train - 2)]
    for orientation in sorted(set(test_orient) - set(train_orient)):
        train_orient[int(rng.integers(0, num_train))] = orientation
    if not set(test_orient) <= set(train_orient):
        return None

    train_needs = [{"move"} for _ in range(num_train)]
    train_needs[0] |= {"bar_dep", "lone_move", "diag"}
    for index in range(1, num_train):
        if rng.random() < 0.5:
            train_needs[index].add("bar_dep")
        if rng.random() < 0.5:
            train_needs[index].add("lone_move")
    taught = {"closer_pair"} | {name for name in _TAUGHT
                                 if any(feats[name] for _, feats in tests)}
    if near_ok:
        taught.add("near_miss")
    if reassign_ok and rng.random() < 0.7:
        taught.add("reassign")
    if rng.random() < 0.5:
        taught.add("stationary")
    for name in sorted(taught):
        train_needs[int(rng.integers(0, num_train))].add(name)

    trains = []
    for orientation, needs in zip(train_orient, train_needs):
        built = build("train", orientation, needs, caps)
        if built is None:
            return None
        trains.append(built)
    return {"train": [pair for pair, _ in trains], "test": [pair for pair, _ in tests]}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    for _ in range(40):
        puzzle = _puzzle_attempt(rng, num_train, num_test)
        if puzzle is None:
            continue
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle
    raise RuntimeError("could not build a valid puzzle")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D array"
    assert np.issubdtype(grid.dtype, np.integer), "grid must be integer"
    assert all(1 <= n <= 30 for n in grid.shape), "grid size out of range"
    assert int(grid.min()) >= 0 and int(grid.max()) <= 9, "colour out of range"


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad keys"
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list), "splits must be lists"
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "bad counts"
    seen = set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair"
            a = np.asarray(pair["input"])
            b = np.asarray(pair["output"])
            _check_grid(a)
            _check_grid(b)
            key = (a.shape, a.astype(np.int64).tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            feats = _analyze_pair(a, b)
            assert feats["move"] >= 1, "every pair must move a token"
            assert not np.array_equal(a, b), "identity pair"
            info[split].append(feats)
    tr, te = info["train"], info["test"]

    def trained(name):
        return any(f[name] for f in tr)

    # Training must expose both parents and the interaction before the test uses them.
    for name in ("bar_dep", "lone_move", "diag", "closer_pair"):
        assert trained(name), f"training lacks {name}"
    assert {f["orientation"] for f in te} <= {f["orientation"] for f in tr}, \
        "test orientation not demonstrated"
    for name in _TAUGHT:
        if any(f[name] for f in te):
            assert trained(name), f"test uses untrained behaviour {name}"
    for f in te:
        assert f["bar_dep"] >= 1 and f["lone_move"] >= 1, "test must need both parents"
        for name in ("static_fusion", "no_fusion", "inert_bar", "bar_only"):
            assert f["shortcuts"][name], f"test solvable by shortcut {name}"
    for name in ("static_fusion", "no_fusion", "inert_bar", "bar_only"):
        assert any(f["shortcuts"][name] for f in tr), f"training never refutes {name}"
    assert any(min(key(f) for f in te) > max(key(f) for f in tr) for key in _SHIFT_AXES), \
        "no train/test shift: tests must exceed all training pairs on some axis"
    if any(f["shortcuts"]["prefusion_choice"] for f in te):
        assert any(f["shortcuts"]["prefusion_choice"] for f in tr), \
            "test refutes prefusion_choice but training does not"


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------
def _rows(text):
    return np.array([[int(ch) for ch in row] for row in text.split()], dtype=int)


_REGRESSION_INPUT = """
000000000000
000000000300
000000000000
055000000550
055000000550
000000000000
000040000000
000000000000
000000000055
600000000055
000000000000
000000008000
"""

# Hand-derived: magnets at (3,1) and (3,9) share rows 3-4 and fuse into a bar
# spanning columns 1-10.  Token 3 at (1,9) drops to (2,9); token 4 at (6,4) rises
# to (5,4) beside the bar's interior (perpendicular projection -- unfused it
# would go to the magnet at (3,1) and land on (5,3)); token 6 at (9,0) is nearer
# the bar than the lone magnet at (8,10) and slides up to (5,0), diagonal contact
# with the bar end; token 8 at (11,8) belongs to the lone magnet and slides
# diagonally to (10,9).
_REGRESSION_OUTPUT = """
000000000000
000000000000
000000000300
055555555550
055555555550
600040000000
000000000000
000000000000
000000000055
000000000055
000000000800
000000000000
"""


def self_test() -> None:
    # Independent, hand-derived regression for the rule itself.
    expected = _rows(_REGRESSION_OUTPUT)
    regression_input = _rows(_REGRESSION_INPUT)
    assert np.array_equal(solve(regression_input), expected), "hand regression failed"
    scene = _parse(regression_input)
    for name in ("static_fusion", "no_fusion", "inert_bar", "bar_only"):
        guess = _shortcut(scene, name)
        assert guess is None or not np.array_equal(guess, expected), f"{name} matches regression"
    vertical = solve(regression_input.T.copy())
    assert np.array_equal(vertical, expected.T), "rule is not transpose-invariant"

    for train, test in ((2, 1), (10, 1), (10, 4), (5, 3)):
        p = generate_puzzle(4242, train, test)
        assert (len(p["train"]), len(p["test"])) == (train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (3, 3), (3, 5), (True, 1)):
        try:
            generate_puzzle(1, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(17), generate_puzzle(17)
    assert all(np.array_equal(x[k], y[k]) for s in ("train", "test")
               for x, y in zip(a[s], b[s]) for k in ("input", "output")), "nondeterministic"

    stats = {"shapes": set(), "test_orient": set(), "gaps": set(), "lone": set(),
             "tokens": set(), "bg": set(), "magnet": set(), "near": 0, "reassign": 0,
             "edge": 0, "full": 0, "float": 0, "bar_end": 0, "ortho_corner": 0,
             "prefusion": 0, "mixed_orient": 0, "pair_colors": 0, "signatures": set(),
             "shift": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        feats = {s: [_analyze_pair(p["input"], p["output"]) for p in puzzle[s]]
                 for s in ("train", "test")}
        stats["signatures"].add(puzzle["test"][0]["input"].tobytes())
        stats["mixed_orient"] += len({f["orientation"] for f in feats["train"]}) == 2
        stats["pair_colors"] += len({(f["bg"], f["magnet"]) for s in feats for f in feats[s]}) > 1
        stats["prefusion"] += any(f["shortcuts"]["prefusion_choice"] for f in feats["test"])
        # Which axes carry a strict train/test shift (validation guarantees one).
        stats["shift"] += sum(
            min(key(f) for f in feats["test"]) > max(key(f) for f in feats["train"])
            for key in _SHIFT_AXES) >= 2
        for split in ("train", "test"):
            for f in feats[split]:
                stats["shapes"].add(f["shape"])
                stats["gaps"].add(f["gap"])
                stats["lone"].add(f["lone"])
                stats["tokens"].add(f["tokens"])
                stats["bg"].add(f["bg"])
                stats["magnet"].add(f["magnet"])
                stats["near"] += f["near_miss"]
                stats["reassign"] += f["reassign"] > 0
                stats["edge"] += f["edge_bar"]
                stats["full"] += f["full_span"]
                stats["float"] += f["bar_float"] > 0
                stats["bar_end"] += f["bar_end"] > 0
                stats["ortho_corner"] += f["ortho_corner"] > 0
                if split == "test":
                    stats["test_orient"].add(f["orientation"])
    assert len(stats["signatures"]) == 42, "seeds collide"
    assert stats["test_orient"] == {"horizontal", "vertical"}, "test orientation fixed"
    assert len(stats["shapes"]) >= 60 and len(stats["gaps"]) >= 12, "geometry too narrow"
    assert len(stats["lone"]) >= 4 and len(stats["tokens"]) >= 10, "counts too narrow"
    assert len(stats["bg"]) >= 6 and len(stats["magnet"]) >= 8, "palette too narrow"
    assert stats["near"] >= 10 and stats["reassign"] >= 10, "interaction cases too rare"
    assert stats["edge"] >= 10 and stats["full"] >= 2, "bar placement too narrow"
    assert stats["bar_end"] >= 20 and stats["ortho_corner"] >= 10, "approach variety too narrow"
    assert stats["mixed_orient"] >= 20 and stats["pair_colors"] >= 8, "within-puzzle variety"
    assert stats["prefusion"] >= 5 and stats["shift"] >= 15, "train/test shift too narrow"
