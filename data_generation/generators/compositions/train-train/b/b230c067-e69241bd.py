"""Congruent chambers share paint (v2).

Composition of two ARC families:

* Parent B (e69241bd): walls partition the floor into orthogonally connected
  regions; a one-cell colour seed flood-fills exactly its own wall-bounded
  region and nothing leaks through diagonal pinches.
* Parent A (b230c067): orthogonally connected components are compared by
  their tightly cropped masks under translation only (no rotation or
  reflection); exact copies form a class.

Rule: every floor chamber (4-connected non-wall region) is an object whose
shape is its cropped cell mask, seed cell included.  A seed paints its own
chamber, and the paint is transferred to every other chamber that is an exact
translated copy of the seeded chamber.  Chambers with no seeded exact copy
(rotations, reflections, same-box rearrangements, grown / shrunken versions,
moved or filled pillars, unseeded duplicates, irregular pockets) keep the
floor colour.  Walls never change.

Roles come from each input on its own: the floor is the most frequent colour
(at least 1.2 times as frequent as the wall), the wall is the second most
frequent colour, and every other colour is a one-cell seed.  Floor and wall
colours may therefore change, or even swap, between the pairs of one puzzle.
"""
from __future__ import annotations

import numpy as np

_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_N8 = _N4 + _DIAG

MIN_SEEDED_SIZE = 2
MIN_RATIO = 1.2
MAX_SIDE = 30
FAMILIES = ("blob", "snake", "bar", "rect", "ring", "pillared", "comb", "cross", "stair", "cup")
FILLS = ("field", "maze", "cuts")
TRANSFORMS = ("rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "antitranspose")
ROT_KINDS = ("rot90", "rot180", "rot270")
FLIP_KINDS = ("flip_lr", "flip_ud", "flip_lr", "flip_ud", "transpose", "antitranspose")
REGIMES = ("minimal", "comparable", "copies", "classes", "both", "scale", "compact", "maximal")
REGIME_P = (0.07, 0.13, 0.17, 0.15, 0.12, 0.10, 0.12, 0.14)


class _Retry(Exception):
    """Internal signal: the current random construction must be redrawn."""


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _choose(rng, items):
    return items[int(rng.integers(len(items)))]


# ---------------------------------------------------------------- shapes

def _norm(cells):
    cells = list(cells)
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _dims(shape):
    return 1 + max(r for r, _ in shape), 1 + max(c for _, c in shape)


def _key(shape):
    return tuple(sorted(shape))


def _transform(shape, name):
    h, w = _dims(shape)
    maps = {
        "id": lambda r, c: (r, c),
        "rot90": lambda r, c: (c, h - 1 - r),
        "rot180": lambda r, c: (h - 1 - r, w - 1 - c),
        "rot270": lambda r, c: (w - 1 - c, r),
        "flip_lr": lambda r, c: (r, w - 1 - c),
        "flip_ud": lambda r, c: (h - 1 - r, c),
        "transpose": lambda r, c: (c, r),
        "antitranspose": lambda r, c: (w - 1 - c, h - 1 - r),
    }
    f = maps[name]
    return _norm(f(r, c) for r, c in shape)


def _orbit(shape):
    out = [shape]
    for name in TRANSFORMS:
        s = _transform(shape, name)
        if s not in out:
            out.append(s)
    return out


def _is_connected(cells):
    cells = set(cells)
    if not cells:
        return False
    start = min(cells)
    seen, stack = {start}, [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in _N4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _contains(big, small):
    bh, bw = _dims(big)
    sh, sw = _dims(small)
    for dr in range(bh - sh + 1):
        for dc in range(bw - sw + 1):
            if all((r + dr, c + dc) in big for r, c in small):
                return True
    return False


def _fill_holes(shape):
    """The shape plus every enclosed (4-unreachable from outside) cell of its box."""
    h, w = _dims(shape)
    stack = [(r, c) for r in range(-1, h + 1) for c in (-1, w)] + \
            [(r, c) for c in range(-1, w + 1) for r in (-1, h)]
    outside = set(stack)
    while stack:
        r, c = stack.pop()
        for dr, dc in _N4:
            q = (r + dr, c + dc)
            if -1 <= q[0] <= h and -1 <= q[1] <= w and q not in shape and q not in outside:
                outside.add(q)
                stack.append(q)
    return frozenset((r, c) for r in range(h) for c in range(w) if (r, c) not in outside)


def _interior(shape):
    return [p for p in sorted(shape) if all((p[0] + dr, p[1] + dc) in shape for dr, dc in _N8)]


def _grow_shape(rng, size, max_h, max_w):
    if max_h * max_w < size:
        return None
    for _ in range(30):
        cells = {(int(rng.integers(max_h)), int(rng.integers(max_w)))}
        while len(cells) < size:
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _N4
                               if 0 <= r + dr < max_h and 0 <= c + dc < max_w} - cells)
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        if len(cells) == size:
            return _norm(cells)
    return None


def _snake_shape(rng, length, limit):
    for _ in range(30):
        cells = [(0, 0)]
        direction = _N4[int(rng.integers(4))]
        while len(cells) < length:
            if rng.random() < 0.4:
                direction = (direction[1], direction[0]) if rng.random() < 0.5 else (-direction[1], -direction[0])
            options = [direction] + [d for d in _N4 if d != direction]
            moved = False
            for dr, dc in options:
                nxt = (cells[-1][0] + dr, cells[-1][1] + dc)
                # stay a corridor: the new cell may touch only the previous cell
                if nxt in cells or any((nxt[0] + a, nxt[1] + b) in cells[:-1] for a, b in _N4):
                    continue
                cells.append(nxt)
                direction = (dr, dc)
                moved = True
                break
            if not moved:
                break
        if len(cells) == length:
            shape = _norm(cells)
            h, w = _dims(shape)
            if h <= limit and w <= limit:
                return shape
    return None


def _random_orient(rng, shape):
    return _transform(shape, _choose(rng, ("id",) + TRANSFORMS))


def _random_shape(rng, family, big):
    if family == "blob":
        size = int(rng.integers(4, 15 if big else 10))
        return _grow_shape(rng, size, int(rng.integers(2, 7 if big else 6)), int(rng.integers(2, 7 if big else 6)))
    if family == "snake":
        return _snake_shape(rng, int(rng.integers(4, 19 if big else 13)), 8 if big else 6)
    if family == "bar":
        length = 2 if rng.random() < 0.3 else int(rng.integers(3, 8 if big else 6))
        if rng.random() < 0.5:
            return frozenset((0, c) for c in range(length))
        return frozenset((r, 0) for r in range(length))
    if family == "rect":
        h, w = int(rng.integers(2, 5 if big else 4)), int(rng.integers(2, 6 if big else 5))
        cells = {(r, c) for r in range(h) for c in range(w)}
        if rng.random() < 0.75:
            corner = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)][int(rng.integers(4))]
            cells.discard(corner)
            if h * w >= 9 and rng.random() < 0.4:
                side = [(r, c) for r, c in cells if (r in (0, h - 1)) != (c in (0, w - 1))]
                cells.discard(_choose(rng, side))
        return _norm(cells) if _is_connected(cells) else None
    if family == "ring":
        h, w = int(rng.integers(3, 6 if big else 5)), int(rng.integers(3, 6 if big else 5))
        cells = {(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)}
        roll = rng.random()
        if roll < 0.35:
            side = [(r, c) for r, c in cells if (r in (0, h - 1)) != (c in (0, w - 1))]
            cells.discard(_choose(rng, side))
        elif roll < 0.65:
            options = [(-1, c) for c in range(w)] + [(h, c) for c in range(w)] + \
                      [(r, -1) for r in range(h)] + [(r, w) for r in range(h)]
            cells.add(_choose(rng, options))
        return _norm(cells) if _is_connected(cells) else None
    if family == "pillared":
        h, w = int(rng.integers(3, 7 if big else 6)), int(rng.integers(4, 8 if big else 7))
        cells = {(r, c) for r in range(h) for c in range(w)}
        interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        k = int(rng.integers(1, min(3, max(1, len(interior) // 2)) + 1))
        picks = rng.permutation(len(interior))[:k]
        for i in picks:
            cells.discard(interior[int(i)])
        if rng.random() < 0.35:
            cells.discard([(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)][int(rng.integers(4))])
        shape = _norm(cells)
        if not _is_connected(shape) or _fill_holes(shape) == shape:
            return None
        return shape
    if family == "comb":
        length = int(rng.integers(3, 8 if big else 6))
        cells = {(0, c) for c in range(length)}
        start = int(rng.integers(0, 2))
        teeth = list(range(start, length, 2))
        if len(teeth) > 2 and rng.random() < 0.4:
            teeth.pop(int(rng.integers(len(teeth))))
        alternate = rng.random() < 0.3
        for i, c in enumerate(teeth):
            t = int(rng.integers(1, 4 if big else 3))
            sign = -1 if (alternate and i % 2) else 1
            cells.update((sign * d, c) for d in range(1, t + 1))
        return _random_orient(rng, _norm(cells))
    if family == "cross":
        arms = [int(rng.integers(0, 4 if big else 3)) for _ in range(4)]
        if rng.random() < 0.8:
            for i in rng.permutation(4)[:3]:
                arms[int(i)] = max(arms[int(i)], 1)
        cells = {(0, 0)}
        for (dr, dc), n in zip(_N4, arms):
            cells.update((dr * d, dc * d) for d in range(1, n + 1))
        return _norm(cells)
    if family == "stair":
        steps, run = int(rng.integers(2, 6 if big else 5)), int(rng.integers(1, 3))
        r = c = 0
        cells = {(0, 0)}
        for _ in range(steps):
            for _ in range(run):
                c += 1
                cells.add((r, c))
            r += 1
            cells.add((r, c))
        return _random_orient(rng, _norm(cells))
    if family == "cup":
        h, w = int(rng.integers(2, 5 if big else 4)), int(rng.integers(3, 7 if big else 6))
        cells = {(h - 1, c) for c in range(w)} | {(r, 0) for r in range(h)} | {(r, w - 1) for r in range(h)}
        if h >= 3 and rng.random() < 0.4:
            cells.discard((0, 0))
        return _random_orient(rng, _norm(cells))
    raise ValueError(family)


def _rearrangements(shape):
    h, w = _dims(shape)
    orbit = set(_orbit(shape))
    empty = [(r, c) for r in range(h) for c in range(w) if (r, c) not in shape]
    out = []
    for removed in sorted(shape):
        for added in empty:
            cells = (set(shape) - {removed}) | {added}
            if not _is_connected(cells):
                continue
            cand = _norm(cells)
            if _dims(cand) != (h, w) or cand in orbit or cand in out:
                continue
            out.append(cand)
            if len(out) >= 40:
                return out
    return out


def _samebox_options(shape):
    dims = _dims(shape)
    flips = [s for s in _orbit(shape)[1:] if _dims(s) == dims]
    return flips, _rearrangements(shape)


def _grown_options(shape, limit):
    out = []
    for r, c in sorted(shape):
        for dr, dc in _N4:
            cell = (r + dr, c + dc)
            if cell in shape:
                continue
            cand = _norm(set(shape) | {cell})
            h, w = _dims(cand)
            if h <= limit and w <= limit and cand not in out:
                out.append(cand)
    return out


def _minus_seed_cells(shape):
    return [p for p in sorted(shape) if len(shape) - 1 >= 3 and _is_connected(set(shape) - {p})]


def _hole_options(shape):
    """Pillar contrasts: fill the holes, add a pillar, or move a pillar."""
    filled = _fill_holes(shape)
    holes = sorted(filled - shape)
    orbit = set(_orbit(shape))
    out = []
    if holes:
        out.append(filled)
    for p in _interior(shape):
        cand = frozenset(set(shape) - {p})
        if cand not in orbit and cand not in out:
            out.append(cand)
        for hole in holes:
            moved = frozenset((set(shape) - {p}) | {hole})
            if _is_connected(moved) and _norm(moved) == moved and moved not in orbit and moved not in out:
                out.append(moved)
    return out


# ------------------------------------------------------------- the rule

def _roles(grid):
    values, counts = np.unique(grid, return_counts=True)
    order = sorted(zip(counts.tolist(), values.tolist()), key=lambda t: (-t[0], t[1]))
    if len(order) < 2:
        raise ValueError("input needs a floor and a wall colour")
    return int(order[0][1]), int(order[1][1])


def _components(open_mask, conn=4):
    h, w = open_mask.shape
    label = np.full((h, w), -1, dtype=int)
    nbrs = _N4 if conn == 4 else _N8
    comps = []
    for r in range(h):
        for c in range(w):
            if not open_mask[r, c] or label[r, c] >= 0:
                continue
            index = len(comps)
            label[r, c] = index
            stack, cells = [(r, c)], []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and open_mask[ny, nx] and label[ny, nx] < 0:
                        label[ny, nx] = index
                        stack.append((ny, nx))
            comps.append(sorted(cells))
    return label, comps


def _signature(shape, match):
    if match == "exact":
        return _key(shape)
    if match == "dihedral":
        return min(_key(s) for s in _orbit(shape))
    if match == "rotations":
        return min(_key(_transform(shape, t)) for t in ("id", "rot90", "rot180", "rot270"))
    if match in ("rot180", "flip_lr", "flip_ud"):
        return min(_key(shape), _key(_transform(shape, match)))
    if match == "count":
        return len(shape)
    if match == "box":
        return (_dims(shape), len(shape))
    if match == "box_only":
        return _dims(shape)
    if match == "holes":
        return _key(_fill_holes(shape))
    raise ValueError(match)


def _apply(grid, conn=4, match="exact", seed_in_key=True, color_mode="class", propagate=True, parts=None):
    """Apply the family rule or one of its deliberately wrong shortcut variants."""
    g = np.asarray(grid, dtype=int)
    parts = {} if parts is None else parts
    if "roles" not in parts:
        parts["roles"] = _roles(g)
    bg, wall = parts["roles"]
    if conn not in parts:
        parts[conn] = _components(g != wall, conn)[1]
    comps = parts[conn]
    out = g.copy()
    seed_cells = [(int(r), int(c)) for r, c in np.argwhere((g != bg) & (g != wall))]
    sources, unseeded = [], []
    for cells in comps:
        seeds = [p for p in cells if g[p] != bg]
        colors = {int(g[p]) for p in seeds}
        if not seeds:
            unseeded.append(cells)
            continue
        if len(colors) != 1:
            continue  # out of family: competing seeds, leave untouched
        color = colors.pop()
        for p in cells:
            out[p] = color
        seedset = set(seeds)
        body = cells if seed_in_key else [p for p in cells if p not in seedset]
        if body:
            sources.append((_norm(body), color))
    if not propagate or not sources:
        return out
    if match == "contains":
        ordered = sorted(sources, key=lambda s: (-len(s[0]), _key(s[0]), s[1]))
        lookup = None
    else:
        lookup = {}
        for shape, color in sources:
            lookup.setdefault(_signature(shape, match), set()).add(color)
    for cells in unseeded:
        shape = _norm(cells)
        color = None
        if lookup is None:
            for source, source_color in ordered:
                if len(source) <= len(shape) and _contains(shape, source):
                    color = source_color
                    break
        else:
            colors = lookup.get(_signature(shape, match))
            if colors and len(colors) == 1:
                color = next(iter(colors))
        if color is None:
            continue
        if color_mode == "nearest":
            best = min(((abs(r - sr) + abs(c - sc), int(g[sr, sc]))
                        for r, c in cells for sr, sc in seed_cells))
            color = best[1]
        for p in cells:
            out[p] = color
    return out


def solve(input_grid):
    """Paint each seeded chamber and every exact translated copy of it."""
    grid = np.asarray(input_grid)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("input must be a non-empty 2D grid")
    grid = grid.astype(int)
    if len(np.unique(grid)) < 2:
        return grid.copy()
    return _apply(grid)


VARIANTS = {
    "flood_only": dict(propagate=False),          # parent B alone
    "eight_conn": dict(conn=8),                    # diagonal pinches leak
    "dihedral": dict(match="dihedral"),            # rotations/reflections count
    "rotations": dict(match="rotations"),          # rotations count
    "rot180": dict(match="rot180"),                # half turns count
    "mirror_lr": dict(match="flip_lr"),            # left-right mirrors count
    "mirror_ud": dict(match="flip_ud"),            # up-down mirrors count
    "count_only": dict(match="count"),             # same cell count counts
    "box_count": dict(match="box"),                # same box and count counts
    "box_only": dict(match="box_only"),            # same bounding box counts
    "holes_filled": dict(match="holes"),           # pillars/holes ignored
    "seedless": dict(seed_in_key=False),           # seed cell not part of room
    "nearest": dict(color_mode="nearest"),         # copies take nearest seed colour
    "contains": dict(match="contains"),            # rooms containing the shape count
}
CORE_VARIANTS = tuple(VARIANTS)
REQUIRED_TRAIN = ("flood_only", "eight_conn", "dihedral", "count_only", "box_count")
NEAR_VARIANTS = ("dihedral", "rotations", "rot180", "mirror_lr", "mirror_ud", "count_only", "box_count",
                 "box_only", "holes_filled", "contains")
_TRANSFORM_EXPOSES = {
    "rot90": {"dihedral", "rotations"}, "rot270": {"dihedral", "rotations"},
    "rot180": {"dihedral", "rotations", "rot180"},
    "flip_lr": {"dihedral", "mirror_lr"}, "flip_ud": {"dihedral", "mirror_ud"},
    "transpose": {"dihedral"}, "antitranspose": {"dihedral"},
}


def _exposures(grid, output):
    """Names of the shortcut variants whose answer differs from the true output."""
    grid = np.asarray(grid, dtype=int)
    output = np.asarray(output, dtype=int)
    parts = {}
    return {name for name, kw in VARIANTS.items()
            if not np.array_equal(_apply(grid, parts=parts, **kw), output)}


# ------------------------------------------------------------- generation

def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _sample_params(rng, num_train, num_test):
    regime = REGIMES[int(rng.choice(len(REGIMES), p=REGIME_P))]
    colors = [int(c) for c in rng.permutation(10)]
    if regime == "minimal":
        palette = "fixed"
    elif regime == "maximal":
        palette = ("drift", "swap")[int(rng.integers(2))]
    else:
        palette = ("fixed", "drift", "swap")[int(rng.choice(3, p=(0.45, 0.35, 0.2)))]
    n = num_train + num_test
    f0, w0 = colors[0], colors[1]
    if palette == "fixed":
        roles = [(f0, w0)] * n
    elif palette == "swap":
        roles = [((f0, w0), (w0, f0))[int(rng.integers(2))] for _ in range(n)]
        roles[0] = (f0, w0)
        roles[int(rng.integers(1, num_train))] = (w0, f0)
    else:
        floors = [f0] + ([colors[2]] if rng.random() < 0.6 else [])
        walls = [w0, colors[3]] + ([colors[4]] if rng.random() < 0.4 else [])
        roles = [(_choose(rng, floors), _choose(rng, walls)) for _ in range(n)]
        roles[0] = (f0, w0)
        roles[int(rng.integers(1, num_train))] = (floors[-1], walls[1])
    pool = [int(c) for c in rng.permutation(10)][:int(rng.integers(4, 10))]
    weights = rng.dirichlet(np.full(len(FAMILIES), 0.8))
    weights = 0.2 / len(FAMILIES) + 0.8 * weights
    if regime == "minimal":
        fill_w = np.zeros(len(FILLS))
        fill_w[int(rng.choice(3, p=(0.5, 0.3, 0.2)))] = 1.0
    else:
        fill_w = rng.dirichlet(np.full(len(FILLS), 1.2)) * 0.75 + 0.25 / len(FILLS)
    return dict(regime=regime, palette=palette, roles=roles, pool=pool,
                weights=weights / weights.sum(), fill_w=fill_w / fill_w.sum(),
                p_packed=0.0 if regime == "minimal" else (0.45 if regime == "maximal" else 0.25),
                holes=bool(rng.random() < 0.6),
                reuse=bool(regime != "minimal" and rng.random() < (0.6 if regime == "maximal" else 0.25)),
                transfer_free=bool(regime not in ("minimal",) and num_train >= 3 and rng.random() < 0.18))


def _pair_palette(rng, params, roles):
    floor, wall = roles
    pool = [c for c in params["pool"] if c not in (floor, wall)]
    extra = [c for c in range(10) if c not in pool and c not in (floor, wall)]
    return [pool[i] for i in rng.permutation(len(pool))] + [extra[i] for i in rng.permutation(len(extra))]


def _supports(shape, need, plan, limit):
    if need == "rot":
        return _transform(shape, plan["rot_kind"]) != shape
    if need == "flip":
        return _transform(shape, plan["flip_kind"]) != shape
    if need == "samebox":
        flips, moves = _samebox_options(shape)
        return bool(flips or moves)
    if need == "minus_seed":
        return bool(_minus_seed_cells(shape))
    if need == "holes":
        return bool(_hole_options(shape))
    if need == "grown":
        return bool(_grown_options(shape, limit))
    if need == "gallery":
        return len([t for t in plan["transforms"] if _transform(shape, t) != shape]) >= 3 and len(_orbit(shape)) >= 4
    if need == "sibling":
        return _transform(shape, plan["sib_kind"]) != shape
    if need == "orient":
        return any(_transform(shape, t) != shape for t in plan["transforms"])
    return True


def _pick_shape(rng, params, plan, avoid_orbits, requires=()):
    big = plan["big"]
    limit = 9 if big else 7
    max_cells = plan.get("max_cells", 60)
    limit = min(limit, plan.get("max_box", limit))
    for _ in range(250):
        if "holes" in requires and rng.random() < 0.7:
            family = "pillared"
        else:
            family = FAMILIES[int(rng.choice(len(FAMILIES), p=params["weights"]))]
        shape = _random_shape(rng, family, big)
        if shape is None or not MIN_SEEDED_SIZE <= len(shape) <= max_cells:
            continue
        h, w = _dims(shape)
        if h > limit or w > limit:
            continue
        if any(_key(s) in avoid_orbits for s in _orbit(shape)):
            continue
        if all(_supports(shape, need, plan, limit) for need in requires):
            return shape, family
    raise _Retry("no shape")


def _decoy_shape(rng, kind, target, seed_cell, avoid, params, plan, history=()):
    big = plan["big"]
    limit = min(9 if big else 7, plan.get("max_box", 9))
    tag = None
    if kind == "orient":
        names = [t for t in plan["transforms"] if _transform(target, t) != target]
        options, tags = [], []
        for t in names:
            s = _transform(target, t)
            if _key(s) not in avoid and s not in options:
                options.append(s)
                tags.append(t)
        if not options:
            return None, None
        i = int(rng.integers(len(options)))
        return options[i], tags[i]
    if kind == "samebox":
        flips, moves = _samebox_options(target)
        options = moves if (moves and (not flips or rng.random() < 0.6)) else flips
    elif kind == "grown":
        options = _grown_options(target, limit)
    elif kind == "minus_seed":
        options = [_norm(set(target) - {seed_cell})] if seed_cell in _minus_seed_cells(target) else []
    elif kind == "holes":
        options = _hole_options(target)
    elif kind == "recount":
        orbit = set(_orbit(target))
        options = []
        for _ in range(30):
            cand = _grow_shape(rng, len(target), int(rng.integers(1, 7)), int(rng.integers(1, 7)))
            if cand is not None and cand not in orbit and max(_dims(cand)) <= limit:
                options.append(cand)
                break
    elif kind == "reuse":
        options = list(history)
    else:  # "other" / "dup"
        options = []
        for _ in range(30):
            family = FAMILIES[int(rng.choice(len(FAMILIES), p=params["weights"]))]
            cand = _random_shape(rng, family, big)
            if cand is not None and len(cand) >= 2 and max(_dims(cand)) <= limit:
                options.append(cand)
                break
    options = [s for s in options if _key(s) not in avoid and max(_dims(s)) <= limit]
    return (_choose(rng, options), tag) if options else (None, None)


def _pair_spec(rng, params, plan, roles, history):
    """Turn a pair plan (counts and evidence needs) into concrete shapes."""
    fill = plan.get("fill") or FILLS[int(rng.choice(len(FILLS), p=params["fill_w"]))]
    placement = plan.get("placement") or ("packed" if rng.random() < params["p_packed"] else "scatter")
    tight = bool(rng.random() < {"field": 0.8, "maze": 0.45, "cuts": 0.7}[fill])
    density = float(rng.uniform(0.16, 0.36))
    needs = set(plan["needs"])
    copies = list(plan["copies"])
    n_lone = int(plan.get("lone", 0))
    n_left = int(plan.get("leftover", 0))
    n_seeded = len(copies) + n_lone
    if n_seeded == 0:
        raise _Retry("no seed planned")
    order = _pair_palette(rng, params, roles)
    if n_seeded + n_left > len(order):
        raise _Retry("not enough seed colours")
    colors = order[:n_seeded + n_left]
    spare = order[n_seeded + n_left:]
    sibling = "sibling" in needs and n_seeded >= 2
    limit = min(9 if plan["big"] else 7, plan.get("max_box", 9))

    requires = [n for n in ("rot", "flip", "samebox", "minus_seed", "holes", "grown", "gallery", "orient")
                if n in needs]
    if sibling:
        requires.append("sibling")
    avoid_orbits, classes = set(), []
    for index in range(n_seeded):
        if index == 1 and sibling:
            shape = _transform(classes[0]["shape"], plan["sib_kind"])
            family = classes[0]["family"]
            if shape == classes[0]["shape"]:
                raise _Retry("symmetric sibling")
        else:
            shape, family = _pick_shape(rng, params, plan, avoid_orbits, requires if index == 0 else ())
        avoid_orbits.update(_key(s) for s in _orbit(shape))
        cells = sorted(shape)
        if index == 0 and "minus_seed" in needs:
            seed_cell = _choose(rng, _minus_seed_cells(shape))
        else:
            seed_cell = _choose(rng, cells)
        classes.append(dict(shape=shape, family=family, copies=copies[index] if index < len(copies) else 1,
                            color=colors[index], seed_cell=seed_cell, n_seeds=1))
    shared = False
    if plan.get("shared_colour") and n_seeded >= 2:
        pairs = [(a, b) for a in range(n_seeded) for b in range(a + 1, n_seeded)
                 if not (sibling and {a, b} == {0, 1})]
        if pairs:
            a, b = _choose(rng, pairs)
            classes[b]["color"] = classes[a]["color"]
            shared = True
    multi = False
    if plan.get("multi_seed"):
        big_classes = [c for c in classes if c["copies"] >= 3]
        if big_classes:
            _choose(rng, big_classes)["n_seeds"] = 2
            multi = True

    seeded_keys = {_key(c["shape"]) for c in classes}
    decoys, tags = [], []
    target = classes[0]
    if sibling:
        tags.append(plan["sib_kind"])
    for need, name in (("rot", "rot_kind"), ("flip", "flip_kind")):
        if need in needs:
            shape = _transform(target["shape"], plan[name])
            if shape == target["shape"] or _key(shape) in seeded_keys:
                raise _Retry("orientation decoy collides")
            decoys.append(dict(shape=shape, copies=1, kind=need))
            tags.append(plan[name])
    if "gallery" in needs:
        used = {_key(d["shape"]) for d in decoys}
        options = []
        for t in plan["transforms"]:
            s = _transform(target["shape"], t)
            if s != target["shape"] and _key(s) not in seeded_keys and _key(s) not in used:
                used.add(_key(s))
                options.append((s, t))
        if len(options) < 2:
            raise _Retry("gallery too small")
        take = int(rng.integers(min(3, len(options)), min(6, len(options)) + 1))
        for i in rng.permutation(len(options))[:take]:
            decoys.append(dict(shape=options[int(i)][0], copies=1, kind="orient"))
            tags.append(options[int(i)][1])
    for kind in ("samebox", "grown", "minus_seed", "holes", "orient"):
        if kind in needs:
            shape, tag = _decoy_shape(rng, kind, target["shape"], target["seed_cell"], seeded_keys, params, plan)
            if shape is None:
                raise _Retry("missing near-match")
            decoys.append(dict(shape=shape, copies=1, kind=kind))
            if tag:
                tags.append(tag)
    if "dup" in needs:
        shape, _ = _decoy_shape(rng, "dup", None, None, seeded_keys, params, plan)
        if shape is None:
            raise _Retry("missing duplicate decoy")
        decoys.append(dict(shape=shape, copies=int(rng.choice((2, 2, 3))), kind="dup"))
    reused = False
    if "reuse" in needs and history:
        shape, _ = _decoy_shape(rng, "reuse", None, None, seeded_keys, params, plan, history)
        if shape is not None:
            decoys.append(dict(shape=shape, copies=int(rng.choice((1, 1, 2))), kind="reuse"))
            reused = True
    for _ in range(int(plan["extras"])):
        kind = _choose(rng, plan["kinds"])
        source = _choose(rng, classes)
        shape, tag = _decoy_shape(rng, kind, source["shape"], source["seed_cell"], seeded_keys, params, plan)
        if shape is not None:
            decoys.append(dict(shape=shape, copies=1, kind=kind))
            if tag:
                tags.append(tag)
    pinch = bool("pinch" in needs or rng.random() < 0.35)
    pinch_kind = plan.get("pinch_kind", "decoy")
    if pinch and pinch_kind in ("decoy", "chain") and len(decoys) < (2 if pinch_kind == "chain" else 1):
        for _ in range(2 if pinch_kind == "chain" else 1):
            shape, _ = _decoy_shape(rng, "other", None, None, seeded_keys, params, plan)
            if shape is None:
                raise _Retry("missing pinch partner")
            decoys.append(dict(shape=shape, copies=1, kind="other"))

    nest = None
    if "nest" in needs:
        inner = None
        if classes[0]["copies"] >= 2 and max(_dims(classes[0]["shape"])) <= 5 and rng.random() < 0.6:
            inner = ("class", 0)
        else:
            small = [i for i, d in enumerate(decoys) if max(_dims(d["shape"])) <= 5]
            if small:
                inner = ("decoy", _choose(rng, small))
            elif classes[0]["copies"] >= 2 and max(_dims(classes[0]["shape"])) <= 5:
                inner = ("class", 0)
        if inner is None:
            raise _Retry("nothing to nest")
        ishape = classes[0]["shape"] if inner[0] == "class" else decoys[inner[1]]["shape"]
        ih, iw = _dims(ishape)
        oh, ow = ih + 4, iw + 4
        ring = {(r, c) for r in range(oh) for c in range(ow) if r in (0, oh - 1) or c in (0, ow - 1)}
        if rng.random() < 0.4:
            options = [(-1, c) for c in range(ow)] + [(oh, c) for c in range(ow)] + \
                      [(r, -1) for r in range(oh)] + [(r, ow) for r in range(oh)]
            ring.add(_choose(rng, options))
        outer = _norm(ring)
        if _key(outer) in seeded_keys or any(_key(d["shape"]) == _key(outer) for d in decoys):
            raise _Retry("frame collides")
        if spare and rng.random() < 0.35:
            classes.append(dict(shape=outer, family="frame", copies=1, color=spare[0],
                                seed_cell=_choose(rng, sorted(outer)), n_seeds=1, nest_outer=True))
            seeded_keys.add(_key(outer))
            nest = dict(shape=outer, cls=len(classes) - 1, kind="class", inner=inner)
        else:
            nest = dict(shape=outer, cls=None, kind="frame", inner=inner)
    return dict(fill=fill, placement=placement, tight=tight, density=density, classes=classes, decoys=decoys,
                pinch=pinch, pinch_kind=pinch_kind, nest=nest, terrace=bool(rng.random() < 0.5),
                margin=bool(fill == "field" and rng.random() < 0.6), edge=bool("edge" in needs),
                corner=bool(rng.random() < 0.35), leftover_color=colors[n_seeded] if n_left else None,
                expose=set(plan["expose"]), expose_any=set(plan.get("expose_any", ())),
                size_mode=plan.get("size_mode", "normal"), cap=plan.get("cap"),
                tags=tags, sibling=sibling, shared=shared, multi=multi, reused=reused)


def _grid_size(rng, spec, scale):
    need = 0.0
    max_h = max_w = 1
    groups = spec["classes"] + spec["decoys"]
    for group in groups:
        h, w = _dims(group["shape"])
        max_h, max_w = max(max_h, h + 2), max(max_w, w + 2)
        need += group["copies"] * (h + 1.6) * (w + 1.6)
    mode = spec["size_mode"]
    if mode == "compact":
        factor = rng.uniform(1.0, 1.3)
    elif mode == "dense":
        factor = rng.uniform(0.95, 1.2)
    elif mode == "large":
        factor = rng.uniform(2.0, 3.0)
    else:
        factor = rng.uniform(1.05, 1.65) if spec["fill"] == "field" else rng.uniform(1.2, 1.85)
    if spec["placement"] == "packed" and mode != "large":
        factor *= 0.85
    area = need * factor * scale + 12
    if mode == "large":
        area = max(area, float(rng.uniform(640, 900)))
    aspect = float(np.exp(rng.uniform(-0.45, 0.45)))
    h = int(round(np.sqrt(area * aspect)))
    h = max(h, max_h, 8)
    w = max(int(round(area / h)), max_w, 8)
    cap = MAX_SIDE
    if mode == "compact":
        cap = 10
    elif spec.get("cap"):
        cap = min(MAX_SIDE, spec["cap"])
    return min(h, cap), min(w, cap)


def _place(rng, spec, h, w):
    tight = spec["tight"]
    packed = spec["placement"] == "packed"
    owner = np.full((h, w), -1, dtype=int)
    forced = np.zeros((h, w), dtype=bool)
    soft = np.zeros((h, w), dtype=bool)
    chambers = []

    def fits(cells, allow_pinch):
        cellset = set(cells)
        for r, c in cells:
            if not (0 <= r < h and 0 <= c < w) or owner[r, c] >= 0 or forced[r, c]:
                return False
            if soft[r, c] and not allow_pinch:
                return False
        for r, c in cells:
            for dr, dc in _N4:
                y, x = r + dr, c + dc
                if 0 <= y < h and 0 <= x < w and (y, x) not in cellset and owner[y, x] >= 0:
                    return False
            if tight and not allow_pinch:
                for dr, dc in _DIAG:
                    y, x = r + dr, c + dc
                    if 0 <= y < h and 0 <= x < w and (y, x) not in cellset and owner[y, x] >= 0:
                        return False
        return True

    def commit(cells, meta):
        index = len(chambers)
        cellset = set(cells)
        for r, c in cells:
            owner[r, c] = index
        for r, c in cells:
            for dr, dc in _N4:
                y, x = r + dr, c + dc
                if 0 <= y < h and 0 <= x < w and (y, x) not in cellset:
                    forced[y, x] = True
            if tight:
                for dr, dc in _DIAG:
                    y, x = r + dr, c + dc
                    if 0 <= y < h and 0 <= x < w and (y, x) not in cellset and owner[y, x] < 0:
                        soft[y, x] = True
        chamber = dict(meta, cells=sorted(cells), index=index)
        chambers.append(chamber)
        return chamber

    def shared_walls(cells):
        cellset = set(cells)
        score = 0
        for r, c in cells:
            for dr, dc in _N4:
                y, x = r + dr, c + dc
                if 0 <= y < h and 0 <= x < w and (y, x) not in cellset and forced[y, x]:
                    score += 1
        return score

    def place_free(item, shape, sh, sw):
        inset = 1 if (spec["margin"] and h - sh >= 4 and w - sw >= 4) else 0
        if not packed:
            for _ in range(160):
                r0 = int(rng.integers(inset, h - sh + 1 - inset))
                c0 = int(rng.integers(inset, w - sw + 1 - inset))
                cells = [(r0 + r, c0 + c) for r, c in shape]
                if fits(cells, False):
                    return commit(cells, item)
            return None
        best, found = None, 0
        for _ in range(240):
            r0 = int(rng.integers(inset, h - sh + 1 - inset))
            c0 = int(rng.integers(inset, w - sw + 1 - inset))
            cells = [(r0 + r, c0 + c) for r, c in shape]
            if not fits(cells, False):
                continue
            score = shared_walls(cells)
            if best is None or score > best[0]:
                best = (score, cells)
            found += 1
            if found >= 14:
                break
        return commit(best[1], item) if best else None

    def place_pinch(item, shape, predicate):
        targets = [ch for ch in chambers if predicate(ch)]
        if not targets:
            return None
        for _ in range(100):
            target = _choose(rng, targets)
            qr, qc = _choose(rng, target["cells"])
            dr, dc = _DIAG[int(rng.integers(4))]
            tr, tc = qr + dr, qc + dc
            if not (0 <= tr < h and 0 <= tc < w) or owner[tr, tc] >= 0 or forced[tr, tc]:
                continue
            pr, pc = _choose(rng, shape)
            cells = [(tr - pr + r, tc - pc + c) for r, c in shape]
            if fits(cells, True):
                return commit(cells, item)
        return None

    def place_edge(item, shape, sh, sw):
        for attempt in range(240):
            if spec["corner"] and attempt < 120:
                r0 = 0 if rng.random() < 0.5 else h - sh
                c0 = 0 if rng.random() < 0.5 else w - sw
            else:
                r0 = int(rng.integers(0, h - sh + 1))
                c0 = int(rng.integers(0, w - sw + 1))
                side = int(rng.integers(4))
                r0 = 0 if side == 0 else (h - sh if side == 1 else r0)
                c0 = 0 if side == 2 else (w - sw if side == 3 else c0)
            cells = [(r0 + r, c0 + c) for r, c in shape]
            if fits(cells, False):
                return commit(cells, dict(item, edge=True))
        return None

    painted = []
    for ci, group in enumerate(spec["classes"]):
        if group.get("nest_outer"):
            continue
        for k in range(group["copies"]):
            painted.append(dict(shape=group["shape"], cls=ci, kind="class", uid=len(painted)))
    unpainted = []
    for di, group in enumerate(spec["decoys"]):
        for k in range(group["copies"]):
            unpainted.append(dict(shape=group["shape"], cls=None, kind=group["kind"], decoy=di,
                                  uid=1000 + len(unpainted)))
    painted = [painted[i] for i in rng.permutation(len(painted))]
    unpainted = [unpainted[i] for i in rng.permutation(len(unpainted))]

    nest = spec["nest"]
    if nest:
        kind, ref = nest["inner"]
        if kind == "class":
            inner = next(it for it in painted if it["cls"] == ref)
            painted.remove(inner)
        else:
            inner = next(it for it in unpainted if it["decoy"] == ref)
            unpainted.remove(inner)
        outer = sorted(nest["shape"])
        oh, ow = _dims(nest["shape"])
        if oh > h or ow > w:
            raise _Retry("grid too small for nest")
        filled = _fill_holes(nest["shape"])
        hole = sorted(filled - nest["shape"])
        top = min(r for r, _ in hole)
        left = min(c for _, c in hole)
        done = False
        for _ in range(120):
            r0 = int(rng.integers(0, h - oh + 1))
            c0 = int(rng.integers(0, w - ow + 1))
            ocells = [(r0 + r, c0 + c) for r, c in outer]
            icells = [(r0 + top + 1 + r, c0 + left + 1 + c) for r, c in sorted(inner["shape"])]
            if fits(ocells, False) and fits(icells, False):
                commit(ocells, dict(shape=nest["shape"], cls=nest["cls"], kind=nest["kind"], uid=-1, nest="outer"))
                commit(icells, dict(inner, nest="inner"))
                for r, c in hole:
                    if owner[r0 + r, c0 + c] < 0:
                        forced[r0 + r, c0 + c] = True
                done = True
                break
        if not done:
            raise _Retry("nest placement failed")

    edge_item = None
    if spec["edge"]:
        multi = [it for it in painted if spec["classes"][it["cls"]]["copies"] >= 2]
        if not multi:
            raise _Retry("no class for edge copy")
        edge_item = multi[0]["uid"]
    pinches = {}
    if spec["pinch"]:
        kind = spec["pinch_kind"]
        if kind == "twin":
            seen = set()
            for it in painted:
                if it["cls"] in seen and it["uid"] != edge_item:
                    cls = it["cls"]
                    pinches[it["uid"]] = lambda ch, cls=cls: ch.get("cls") == cls
                    break
                seen.add(it["cls"])
        elif kind == "rival" and painted:
            first = painted[0]["cls"]
            for it in painted[1:]:
                if it["cls"] != first and it["uid"] != edge_item:
                    cls = it["cls"]
                    pinches[it["uid"]] = lambda ch, cls=cls: ch.get("cls") is not None and ch.get("cls") != cls
                    break
        if not pinches and unpainted:
            pinches[unpainted[0]["uid"]] = lambda ch: ch.get("kind") == "class"
            if kind == "chain" and len(unpainted) >= 2:
                anchor = unpainted[0]["uid"]
                pinches[unpainted[1]["uid"]] = lambda ch, anchor=anchor: ch.get("uid") == anchor
        if not pinches:
            raise _Retry("pinch has no partner")
    last_member = {}
    for item in painted + unpainted:
        shape = sorted(item["shape"])
        sh, sw = _dims(item["shape"])
        if sh > h or sw > w:
            raise _Retry("grid too small")
        chamber = None
        if item["uid"] in pinches:
            chamber = place_pinch(item, shape, pinches[item["uid"]])
            if chamber is None:
                raise _Retry("pinch failed")
        elif item["uid"] == edge_item:
            chamber = place_edge(item, shape, sh, sw)
            if chamber is None:
                raise _Retry("edge placement failed")
        else:
            if packed and spec["terrace"] and item["cls"] in last_member:
                prev = last_member[item["cls"]]
                pr0 = min(r for r, _ in prev["cells"])
                pc0 = min(c for _, c in prev["cells"])
                offsets = [(0, sw + 1), (sh + 1, 0), (0, -sw - 1), (-sh - 1, 0)]
                for i in rng.permutation(4):
                    dr, dc = offsets[int(i)]
                    cells = [(pr0 + dr + r, pc0 + dc + c) for r, c in shape]
                    if fits(cells, False):
                        chamber = commit(cells, item)
                        break
            if chamber is None:
                chamber = place_free(item, shape, sh, sw)
        if chamber is None:
            raise _Retry("placement failed")
        if item["cls"] is not None:
            last_member[item["cls"]] = chamber
    return owner, forced, soft, chambers


def _cuts(rng, free, wall, max_walls):
    """B-style straight wall segments across the free floor, stopping at walls."""
    h, w = free.shape
    cut = np.zeros((h, w), dtype=bool)
    avail = free.copy()
    target = int(rng.integers(2, 4 + (h * w) // 70))
    placed = 0
    for _ in range(target * 4):
        if placed >= target:
            break
        cells = np.argwhere(avail)
        if not len(cells):
            break
        r, c = (int(v) for v in cells[int(rng.integers(len(cells)))])
        if rng.random() < 0.5:
            c1 = c
            while c1 - 1 >= 0 and avail[r, c1 - 1]:
                c1 -= 1
            c2 = c
            while c2 + 1 < w and avail[r, c2 + 1]:
                c2 += 1
            seg = [(r, x) for x in range(c1, c2 + 1)]
            beside = sum(1 for y, x in seg if (y > 0 and cut[y - 1, x]) or (y + 1 < h and cut[y + 1, x]))
        else:
            r1 = r
            while r1 - 1 >= 0 and avail[r1 - 1, c]:
                r1 -= 1
            r2 = r
            while r2 + 1 < h and avail[r2 + 1, c]:
                r2 += 1
            seg = [(y, c) for y in range(r1, r2 + 1)]
            beside = sum(1 for y, x in seg if (x > 0 and cut[y, x - 1]) or (x + 1 < w and cut[y, x + 1]))
        if len(seg) < 3 or beside * 3 > len(seg):
            continue
        if len(seg) >= 5 and rng.random() < 0.35:
            seg.pop(int(rng.integers(1, len(seg) - 1)))
        if int(wall.sum() + cut.sum()) + len(seg) > max_walls:
            break
        for p in seg:
            cut[p] = True
            avail[p] = False
        placed += 1
    return cut


def _render(rng, spec, h, w, roles):
    bg, wall_color = roles
    owner, forced, soft, chambers = _place(rng, spec, h, w)
    unowned = owner < 0
    wall = forced | (soft & unowned)
    free = unowned & ~forced & ~soft
    n_seeds = sum(c["n_seeds"] for c in spec["classes"]) + (1 if spec["leftover_color"] is not None else 0)
    # Leave headroom for the frequency margin: floor >= MIN_RATIO * wall.
    max_walls = int((h * w - n_seeds) / (1 + MIN_RATIO * 1.08))
    if spec["fill"] == "maze":
        noise = free & (rng.random((h, w)) < spec["density"])
        extra = int(wall.sum() + noise.sum()) - max_walls
        if extra > 0:
            cells = np.argwhere(noise)
            for i in rng.permutation(len(cells))[:extra]:
                noise[tuple(cells[int(i)])] = False
        wall |= noise
    elif spec["fill"] == "cuts":
        wall |= _cuts(rng, free, wall, max_walls)
    _check(not wall[~unowned].any(), "chamber cell walled")
    seeded_keys = {_key(c["shape"]) for c in spec["classes"]}
    chamber_cells = {p for ch in chambers for p in ch["cells"]}
    # Wall in leftover pockets that would accidentally copy a seeded chamber.
    while True:
        _, comps = _components(~wall)
        changed = False
        for cells in comps:
            if cells[0] in chamber_cells:
                continue
            if _key(_norm(cells)) in seeded_keys:
                for p in cells:
                    wall[p] = True
                changed = True
        if not changed:
            break
    label, comps = _components(~wall)
    for ch in chambers:
        _check(comps[label[ch["cells"][0]]] == ch["cells"], "chamber is not an exact component")
    grid = np.where(wall, wall_color, bg).astype(int)
    for ci, group in enumerate(spec["classes"]):
        members = [ch for ch in chambers if ch["cls"] == ci]
        hosts = [members[int(i)] for i in rng.permutation(len(members))[:group["n_seeds"]]]
        for k, host in enumerate(hosts):
            r0 = min(r for r, _ in host["cells"])
            c0 = min(c for _, c in host["cells"])
            cell = group["seed_cell"] if k == 0 else _choose(rng, sorted(group["shape"]))
            grid[r0 + cell[0], c0 + cell[1]] = group["color"]
    leftover_seeded = False
    if spec["leftover_color"] is not None:
        keys = [_key(_norm(cells)) for cells in comps]
        counts = {}
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
        candidates = [cells for cells, k in zip(comps, keys)
                      if cells[0] not in chamber_cells and len(cells) >= 5 and counts[k] == 1]
        if not candidates:
            raise _Retry("no leftover region to seed")
        cells = _choose(rng, candidates)
        grid[_choose(rng, cells)] = spec["leftover_color"]
        leftover_seeded = True
    if not any(cells[0] not in chamber_cells and all(grid[p] == bg for p in cells) for cells in comps) and \
            not any(ch["kind"] != "class" for ch in chambers):
        raise _Retry("nothing stays floor")
    counts = np.bincount(grid.ravel(), minlength=10)
    seed_max = max(int(counts[c]) for c in range(10) if c not in (bg, wall_color))
    if not (counts[bg] >= MIN_RATIO * counts[wall_color] and counts[wall_color] > seed_max):
        raise _Retry("floor/wall frequency margin violated")
    output = solve(grid)
    for ch in chambers:
        values = {int(output[p]) for p in ch["cells"]}
        if ch["kind"] == "class":
            _check(values == {spec["classes"][ch["cls"]]["color"]}, "class chamber not painted")
        else:
            _check(values == {bg}, "decoy chamber painted")
    if spec["expose"] or spec["expose_any"]:
        parts = {}
        for name in spec["expose"]:
            if np.array_equal(_apply(grid, parts=parts, **VARIANTS[name]), output):
                raise _Retry(f"{name} not exposed")
        if spec["expose_any"] and not any(
                not np.array_equal(_apply(grid, parts=parts, **VARIANTS[name]), output)
                for name in spec["expose_any"]):
            raise _Retry("transfer-free pair shows no near-match contrast")
    meta = dict(fill=spec["fill"], placement=spec["placement"], tight=spec["tight"], shape=(h, w),
                size_mode=spec["size_mode"],
                copies=[c["copies"] for c in spec["classes"]],
                families=sorted({c["family"] for c in spec["classes"]}),
                decoys=sorted(d["kind"] for d in spec["decoys"]), tags=sorted(set(spec["tags"])),
                pinch=spec["pinch"], pinch_kind=spec["pinch_kind"] if spec["pinch"] else None,
                leftover=leftover_seeded, nest=bool(spec["nest"]), sibling=spec["sibling"],
                shared=spec["shared"], multi=spec["multi"], reused=spec["reused"],
                seeded_shapes=[c["shape"] for c in spec["classes"] if c["family"] != "frame"],
                n_lone=sum(1 for c in spec["classes"] if c["copies"] == 1))
    return grid, output, meta


def _make_pair(rng, params, plan, roles, history):
    for attempt in range(40):
        if attempt == 20 and plan.get("size_mode") in ("compact", "dense"):
            plan = dict(plan, size_mode="normal", cap=None)
        try:
            spec = _pair_spec(rng, params, plan, roles, history)
        except _Retry:
            continue
        for grow in range(4):
            h, w = _grid_size(rng, spec, 1.0 + 0.3 * grow)
            try:
                return _render(rng, spec, h, w, roles)
            except _Retry:
                continue
    raise _Retry("pair construction failed")


def _train_plans(rng, params, n):
    regime = params["regime"]
    needs = [set() for _ in range(n)]
    core = ["rot", "flip", "samebox", "pinch", "multicolor", "grown", "edge"]
    if params["holes"]:
        core.append("holes")
    for need in core:
        needs[int(rng.integers(n))].add(need)
    optional = [("dup", 0.6), ("minus_seed", 0.3)]
    if regime != "minimal":
        rich = regime == "maximal"
        optional += [("sibling", 0.8 if rich else 0.4), ("gallery", 0.45 if rich else 0.2),
                     ("nest", 0.6 if rich else 0.25), ("multi_seed", 0.4 if rich else 0.2),
                     ("shared_colour", 0.4 if rich else 0.12)]
    for need, p in optional:
        if rng.random() < p:
            needs[int(rng.integers(n))].add(need)
    if params["reuse"] and n >= 2:
        needs[int(rng.integers(1, n))].add("reuse")
    tf = None
    if params["transfer_free"] and n >= 3:
        tf = int(rng.integers(n))
        others = [i for i in range(n) if i != tf]
        for need in ("edge", "multicolor", "multi_seed", "shared_colour", "nest"):
            if need in needs[tf]:
                needs[tf].discard(need)
                needs[_choose(rng, others)].add(need)
    fills = None
    if regime == "maximal":
        fills = [FILLS[(i + int(rng.integers(3))) % 3] for i in range(n)]
        fills = [FILLS[i % 3] for i in rng.permutation(n)] if n >= 3 else fills
    plans = []
    for i in range(n):
        heavy = len(needs[i] - {"dup", "minus_seed"})
        size_mode = "normal"
        roll = rng.random()
        if roll < 0.16 and heavy <= 2 and i != tf:
            size_mode = "compact"
        elif roll > 0.92 and regime != "scale":
            size_mode = "large"
        if regime == "minimal" or size_mode == "compact":
            n_prop = 1
        else:
            n_prop = int(rng.choice((1, 2, 3), p=(0.55, 0.33, 0.12)))
        if regime == "minimal":
            copies = [int(rng.choice((2, 3)))]
        elif size_mode == "compact":
            copies = [2]
        else:
            copies = [int(rng.choice((2, 3, 4), p=(0.55, 0.33, 0.12))) for _ in range(n_prop)]
        if "multi_seed" in needs[i]:
            copies[0] = max(copies[0], 3)
        lone = leftover = 0
        if "multicolor" in needs[i] and n_prop == 1:
            if rng.random() < 0.5:
                lone = 1
            else:
                leftover = 1
        elif size_mode != "compact":
            lone = int(rng.random() < 0.15)
            leftover = int(rng.random() < 0.15)
        if i == tf:
            copies = []
            lone = max(lone, 1 + int(rng.random() < 0.4))
        shared = "shared_colour" in needs[i] and "multicolor" not in needs[i]
        if ("sibling" in needs[i] or shared) and len(copies) + lone < 2:
            lone += 1
        expose = {"flood_only"} if copies else set()
        rot_kind = _choose(rng, ROT_KINDS)
        flip_kind = _choose(rng, FLIP_KINDS)
        sib_kind = _choose(rng, [t for t in TRANSFORMS if t not in (rot_kind, flip_kind)])
        for k in needs[i]:
            if k == "rot":
                expose |= _TRANSFORM_EXPOSES[rot_kind]
            elif k == "flip":
                expose |= _TRANSFORM_EXPOSES[flip_kind]
            elif k in ("samebox", "pinch", "multicolor", "grown", "holes"):
                expose.add({"samebox": "box_count", "pinch": "eight_conn", "multicolor": "nearest",
                            "grown": "contains", "holes": "holes_filled"}[k])
        kinds = ["orient", "samebox", "recount", "other", "grown", "minus_seed"]
        if params["holes"]:
            kinds.append("holes")
        big = size_mode == "large" or (size_mode == "normal" and rng.random() < 0.2)
        plan = dict(needs=needs[i], copies=copies, lone=lone, leftover=leftover,
                    extras=0 if size_mode == "compact" else int(rng.integers(0, 3 if heavy <= 2 else 2)), kinds=kinds,
                    expose=expose, expose_any=set(NEAR_VARIANTS) if i == tf else set(), big=big,
                    size_mode=size_mode, rot_kind=rot_kind, flip_kind=flip_kind,
                    sib_kind=sib_kind, transforms=list(TRANSFORMS),
                    pinch_kind="decoy" if regime == "minimal" else
                    ("decoy", "twin", "rival", "chain")[int(rng.choice(4, p=(0.3, 0.25, 0.3, 0.15)))],
                    multi_seed="multi_seed" in needs[i], shared_colour=shared, fill=fills[i] if fills else None,
                    placement="scatter" if regime == "minimal" else None)
        if size_mode == "compact":
            plan.update(max_cells=7, max_box=4)
        plans.append(plan)
    return plans


def _test_plan(rng, params, tinfo, first):
    regime = params["regime"]
    max_prop, max_members = tinfo["max_prop"], tinfo["max_members"]
    mode = regime if (first or rng.random() < 0.5) else "comparable"
    if regime == "minimal":
        mode = "comparable"
    n_prop = int(rng.integers(1, max_prop + 1))
    if mode in ("classes", "both", "maximal"):
        n_prop = min(max_prop + 1, 5)
    copies = [int(rng.integers(2, max_members + 1)) for _ in range(n_prop)]
    if mode in ("copies", "both", "maximal"):
        copies[0] = min(max_members + int(rng.integers(1, 3)), 6)
    if mode == "compact":
        n_prop = max(1, max_prop - int(rng.integers(0, 2)))
        copies = [int(rng.integers(2, max_members + 1)) for _ in range(n_prop)]
    size_mode = {"scale": "large", "compact": "dense"}.get(mode, "normal")
    if mode == "maximal" and rng.random() < 0.5:
        size_mode = "large"
    taught = tinfo["kinds"]
    needs = set()
    near = [k for k in ("orient", "samebox", "grown", "holes") if k in taught]
    if near and rng.random() < 0.6:
        needs.add(_choose(rng, near))
    if rng.random() < 0.35:
        needs.add("dup")
    if rng.random() < 0.4:
        needs.add("pinch")
    if "minus_seed" in taught and rng.random() < 0.3:
        needs.add("minus_seed")
    rich = regime == "maximal"
    transforms = sorted(tinfo["tags"])
    if transforms and rng.random() < (0.5 if rich else 0.25):
        needs.add("sibling")
    if len(transforms) >= 3 and rng.random() < (0.3 if rich else 0.12):
        needs.add("gallery")
    if regime != "minimal" and rng.random() < (0.35 if rich else 0.15):
        needs.add("nest")
    if params["reuse"] and rng.random() < 0.4:
        needs.add("reuse")
    lone = int(rng.random() < 0.2)
    shared = bool(tinfo["shared"] and rng.random() < 0.3)
    if ("sibling" in needs or shared) and len(copies) + lone < 2:
        lone += 1
    expose_any = set()
    if not first and tinfo["transfer_free"] and rng.random() < 0.3:
        copies = []
        lone = max(lone, 1)
        expose_any = set(NEAR_VARIANTS) & tinfo["exposed"]
    total = sum(copies) + lone
    plan = dict(needs=needs, copies=copies, lone=lone, leftover=int(rng.random() < 0.3),
                extras=int(rng.integers(2, 6)) if size_mode == "large" else int(rng.integers(0, 4)),
                kinds=sorted(taught - {"minus_seed", "reuse", "dup"}) or ["other"],
                expose={"flood_only"} if copies else set(), expose_any=expose_any,
                big=bool(size_mode != "dense" and rng.random() < 0.7), size_mode=size_mode,
                transforms=transforms or ["rot90"], sib_kind=_choose(rng, transforms) if transforms else "rot90",
                rot_kind="rot90", flip_kind="flip_lr",
                pinch_kind=("decoy", "twin", "rival", "chain")[int(rng.integers(4))],
                multi_seed=bool(tinfo["multi"] and rng.random() < 0.3),
                shared_colour=shared,
                placement="packed" if size_mode == "dense" else ("scatter" if regime == "minimal" else None),
                fill=tinfo["fill"] if regime == "minimal" else None,
                cap=tinfo["max_side"] if size_mode == "dense" else None)
    if total >= 10:
        plan.update(big=False, max_cells=9, max_box=5)
    return plan


def _generate(seed, num_train, num_test):
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    for _ in range(60):
        params = _sample_params(rng, num_train, num_test)
        try:
            plans = _train_plans(rng, params, num_train)
            pairs, metas, seen, history = [], [], set(), []
            for index in range(num_train):
                for _ in range(8):
                    grid, output, meta = _make_pair(rng, params, plans[index], params["roles"][index], history)
                    key = grid.tobytes() + bytes(grid.shape)
                    if key not in seen:
                        break
                else:
                    raise _Retry("duplicate inputs")
                seen.add(key)
                pairs.append({"input": grid, "output": output})
                metas.append(meta)
                history.extend(s for s in meta["seeded_shapes"] if s not in history)
            exposed = set()
            infos = []
            for pair in pairs:
                exposed |= _exposures(pair["input"], pair["output"])
                infos.append(_analyze(pair["input"], pair["output"]))
            if not set(REQUIRED_TRAIN) <= exposed:
                raise _Retry("training misses a core refutation")
            kinds = {"other", "recount"}
            for m in metas:
                kinds.update("orient" if k in ("rot", "flip") else k for k in m["decoys"] if k != "frame")
            tinfo = dict(max_prop=max(max(i["n_prop"] for i in infos), 1),
                         max_members=max(max(i["copies"]) for i in infos),
                         kinds=kinds, tags={t for m in metas for t in m["tags"]}, exposed=exposed,
                         transfer_free=any(i["painted_unseeded"] == 0 for i in infos),
                         multi=any(m["multi"] for m in metas), shared=any(m["shared"] for m in metas),
                         edge=any(i["edge_copy"] for i in infos), fill=metas[0]["fill"],
                         max_side=max(max(p["input"].shape) for p in pairs))
            tinfo["max_members"] = max(tinfo["max_members"], 2)
            for index in range(num_test):
                for _ in range(14):
                    plan = _test_plan(rng, params, tinfo, index == 0)
                    try:
                        grid, output, meta = _make_pair(rng, params, plan, params["roles"][num_train + index],
                                                        history)
                    except _Retry:
                        continue
                    key = grid.tobytes() + bytes(grid.shape)
                    if key in seen or not _exposures(grid, output) <= exposed:
                        continue
                    info = _analyze(grid, output)
                    if info["edge_copy"] and not tinfo["edge"]:
                        continue
                    if index == 0 and info["painted_unseeded"] == 0:
                        continue
                    break
                else:
                    raise _Retry("no acceptable test pair")
                seen.add(key)
                pairs.append({"input": grid, "output": output})
                metas.append(meta)
            puzzle = {"train": pairs[:num_train], "test": pairs[num_train:]}
            validate_generated_puzzle(puzzle)
            return puzzle, dict(params=params, pairs=metas)
        except (_Retry, AssertionError):
            continue
    raise RuntimeError("could not generate a valid puzzle")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    puzzle, _ = _generate(seed, num_train, num_test)
    return puzzle


# ------------------------------------------------------------- validation

def _analyze(grid, output):
    """Independent re-derivation of one pair's structure and expected output."""
    grid = np.asarray(grid, dtype=int)
    output = np.asarray(output, dtype=int)
    _check(grid.ndim == 2 and grid.shape == output.shape, "output must keep input dimensions")
    values, counts = np.unique(grid, return_counts=True)
    ranked = sorted(zip(counts.tolist(), values.tolist()), key=lambda t: (-t[0], t[1]))
    _check(len(ranked) >= 3, "input needs floor, wall and at least one seed")
    _check(ranked[0][0] >= MIN_RATIO * ranked[1][0], "floor must outnumber wall by the 1.2 margin")
    _check(ranked[1][0] > ranked[2][0], "wall must outnumber every seed colour")
    bg, wall = ranked[0][1], ranked[1][1]
    _, comps = _components(grid != wall)
    masks, seeds = [], []
    for cells in comps:
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        mask = np.zeros((max(rows) - min(rows) + 1, max(cols) - min(cols) + 1), dtype=bool)
        for r, c in cells:
            mask[r - min(rows), c - min(cols)] = True
        masks.append(mask)
        seeds.append([(r, c) for r, c in cells if grid[r, c] != bg])
    expected = grid.copy()
    seeded = [i for i, s in enumerate(seeds) if s]
    for i in seeded:
        _check(len(seeds[i]) == 1, "a chamber holds more than one seed")
        _check(len(comps[i]) >= MIN_SEEDED_SIZE, "seeded chamber too small")
    members = {}
    for i in seeded:
        color = int(grid[seeds[i][0]])
        group = [j for j in range(len(comps)) if masks[j].shape == masks[i].shape
                 and np.array_equal(masks[j], masks[i])]
        for j in group:
            _check(j == i or not seeds[j] or int(grid[seeds[j][0]]) == color,
                   "identical chambers carry different seed colours")
            for p in comps[j]:
                expected[p] = color
        members[i] = group
    _check(np.array_equal(expected, output), "output does not follow the congruent-chamber rule")
    _check(np.array_equal(output == wall, grid == wall), "walls must be preserved exactly")
    painted_unseeded = len({j for i in seeded for j in members[i] if not seeds[j]})
    h, w = grid.shape
    edge_copy = any(not seeds[j] and any(r in (0, h - 1) or c in (0, w - 1) for r, c in comps[j])
                    for i in seeded for j in members[i])
    unpainted = sum(1 for j, cells in enumerate(comps) if output[cells[0]] == bg)
    classes = {}
    for i in seeded:
        classes[min(members[i])] = len(members[i])
    return dict(bg=bg, wall=wall, painted_unseeded=painted_unseeded, unpainted=unpainted, edge_copy=edge_copy,
                copies=sorted(classes.values(), reverse=True),
                n_prop=sum(1 for v in classes.values() if v >= 2))


def _as_grid(value):
    array = np.asarray(value)
    _check(array.ndim == 2 and 1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30,
           "grid dimensions must be 1..30")
    _check(np.issubdtype(array.dtype, np.integer), "grid cells must be integers")
    _check(((array >= 0) & (array <= 9)).all(), "colours must be 0..9")
    return array.astype(int)


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    train, test = puzzle["train"], puzzle["test"]
    _check(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _check(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "pair counts")
    grids = {"train": [], "test": []}
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            grid, output = _as_grid(pair["input"]), _as_grid(pair["output"])
            key = grid.tobytes() + bytes(grid.shape)
            _check(key not in seen, "duplicate input")
            seen.add(key)
            grids[split].append((grid, output))
    infos = {s: [_analyze(g, o) for g, o in grids[s]] for s in grids}
    # Roles are read from each input; a test may use an unseen floor/wall
    # assignment only if training already shows that assignments vary.
    train_roles = {(i["bg"], i["wall"]) for i in infos["train"]}
    for info in infos["test"]:
        _check((info["bg"], info["wall"]) in train_roles or len(train_roles) >= 2,
               "test uses new floor/wall colours but training never varies them")
    for split in infos:
        for info in infos[split]:
            _check(info["unpainted"] >= 1, "every pair must keep some chamber unpainted")
    transfer_free = [i for i, info in enumerate(infos["train"]) if info["painted_unseeded"] == 0]
    _check(len(transfer_free) <= (1 if len(train) >= 3 else 0), "too many training pairs without a transfer")
    _check(infos["test"][0]["painted_unseeded"] >= 1, "the first test must transfer paint to a copy")
    exposed = {}
    for split in grids:
        exposed[split] = set()
        for (grid, output), info in zip(grids[split], infos[split]):
            names = _exposures(grid, output)
            if info["painted_unseeded"] == 0:
                _check(names & set(NEAR_VARIANTS), "a transfer-free pair must refute a near-match shortcut")
            exposed[split] |= names
    for name in REQUIRED_TRAIN:
        _check(name in exposed["train"], f"training does not rule out shortcut {name}")
    _check(exposed["test"] <= exposed["train"], "a test exposes a shortcut that training never refutes")
    _check(not any(i["edge_copy"] for i in infos["test"]) or any(i["edge_copy"] for i in infos["train"]),
           "tests paint an edge-bounded copy but training never does")


# ------------------------------------------------------------- measurement

def _measure_pair(grid, output):
    """Grid-measured structural features of one pair (used by self_test)."""
    grid = np.asarray(grid, dtype=int)
    output = np.asarray(output, dtype=int)
    bg, wall = _roles(grid)
    label, comps = _components(grid != wall)
    h, w = grid.shape
    shapes = [_norm(c) for c in comps]
    seeded = [i for i, c in enumerate(comps) if any(grid[p] != bg for p in c)]
    seeded_shapes = {shapes[i] for i in seeded}
    members = {s: [j for j in range(len(comps)) if shapes[j] == s] for s in seeded_shapes}
    colour_of = {}
    for i in seeded:
        colour_of.setdefault(shapes[i], set()).add(int(grid[next(p for p in comps[i] if grid[p] != bg)]))
    prop = [s for s in seeded_shapes if len(members[s]) >= 2]
    sibling = any(a != b and b in _orbit(a) for a in seeded_shapes for b in seeded_shapes)
    gallery = any(len({shapes[j] for j in range(len(comps)) if shapes[j] in set(_orbit(s))}) >= 4
                  for s in prop if len(s) <= 60)
    nested = False
    for i, c in enumerate(comps):
        if len(c) > 120:
            continue
        filled = _fill_holes(shapes[i])
        if filled == shapes[i]:
            continue
        r0 = min(r for r, _ in c)
        c0 = min(x for _, x in c)
        if any(label[r + r0, x + c0] >= 0 for r, x in filled - shapes[i]):
            nested = True
            break
    contacts = set()
    for r in range(h):
        for c in range(w):
            a = label[r, c]
            if a < 0:
                continue
            for dr, dc in _DIAG:
                y, x = r + dr, c + dc
                if 0 <= y < h and 0 <= x < w and label[y, x] >= 0 and label[y, x] != a:
                    ca, cb = int(output[comps[a][0]]), int(output[comps[label[y, x]][0]])
                    if ca != bg and cb != bg:
                        contacts.add("same_colour" if ca == cb else "two_colours")
                    elif ca != bg or cb != bg:
                        contacts.add("painted_unpainted")
    colours = [next(iter(v)) for v in colour_of.values() if len(v) == 1]
    return dict(
        roles=(bg, wall), shape=(h, w), n_prop=len(prop),
        max_members=max((len(members[s]) for s in prop), default=1),
        transfer=any(output[comps[j][0]] != bg and j not in seeded for s in prop for j in members[s]),
        domino=any(len(s) == 2 for s in prop), large_class=any(len(s) >= 14 for s in prop),
        holed_class=any(_fill_holes(s) != s for s in prop), sibling=sibling, gallery=gallery, nested=nested,
        contacts=contacts, multi_seed=any(sum(1 for j in members[s] if j in seeded) >= 2 for s in prop),
        shared_colour=len(colours) != len(set(colours)),
        edge_seeded=any(len(members[shapes[i]]) >= 2 and
                        any(r in (0, h - 1) or c in (0, w - 1) for r, c in comps[i]) for i in seeded),
        rect_share=(sum(1 for s in prop if len(s) == _dims(s)[0] * _dims(s)[1]), len(prop)),
        unpainted_shapes={shapes[j] for j in range(len(comps)) if output[comps[j][0]] == bg},
        seeded_shapes=seeded_shapes)


# ------------------------------------------------------------- self test

def _tally(seeds, num_train=4, num_test=1):
    """Grid-measured feature counts over generated puzzles (shared by self_test and reports)."""
    tally = dict(sizes=set(), fills=set(), regimes=set(), palettes=set(), compact=0, large=0, domino=0,
                 large_class=0, holed=0, sibling=0, gallery=0, nested=0, same=0, rival=0, multi=0, shared=0,
                 edge_seeded=0, transfer_free=0, reuse=0, role_puzzles=0, swap_puzzles=0, four_classes=0,
                 holes_refuted=0, mirror_refuted=0, rect=0, prop=0, packed=0, cuts=0, train_pairs=0,
                 test_not_larger=0, puzzles=0)
    signatures = set()
    for seed in seeds:
        puzzle, meta = _generate(seed, num_train, num_test)
        validate_generated_puzzle(puzzle)
        tally["puzzles"] += 1
        signatures.add(b"".join(p["input"].tobytes() for p in puzzle["train"] + puzzle["test"]))
        tally["regimes"].add(meta["params"]["regime"])
        tally["palettes"].add(meta["params"]["palette"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _check(np.array_equal(solve(pair["input"]), pair["output"]), "solve mismatch")
        measured = {s: [_measure_pair(p["input"], p["output"]) for p in puzzle[s]] for s in ("train", "test")}
        roles = {m["roles"] for s in measured for m in measured[s]}
        tally["role_puzzles"] += len(roles) >= 2
        tally["swap_puzzles"] += any((w, f) in roles for f, w in roles)
        train_exposed = set().union(*[_exposures(p["input"], p["output"]) for p in puzzle["train"]])
        tally["holes_refuted"] += "holes_filled" in train_exposed
        tally["mirror_refuted"] += bool({"mirror_lr", "mirror_ud"} & train_exposed)
        largest = max(m["shape"][0] * m["shape"][1] for m in measured["train"])
        tally["test_not_larger"] += measured["test"][0]["shape"][0] * measured["test"][0]["shape"][1] <= largest
        everything = measured["train"] + measured["test"]
        for i, m in enumerate(everything):
            others = set().union(*[o["seeded_shapes"] for j, o in enumerate(everything) if j != i])
            tally["reuse"] += bool(m["unpainted_shapes"] & (others - m["seeded_shapes"]))
        for split in ("train", "test"):
            for m in measured[split]:
                tally["sizes"].add(m["shape"])
                tally["domino"] += m["domino"]
                tally["holed"] += m["holed_class"]
                tally["sibling"] += m["sibling"]
                tally["gallery"] += m["gallery"]
                tally["nested"] += m["nested"]
                tally["same"] += "same_colour" in m["contacts"]
                tally["rival"] += "two_colours" in m["contacts"]
                tally["multi"] += m["multi_seed"]
                tally["shared"] += m["shared_colour"]
                tally["edge_seeded"] += m["edge_seeded"]
                tally["transfer_free"] += not m["transfer"]
                tally["rect"] += m["rect_share"][0]
                tally["prop"] += m["rect_share"][1]
                if split == "train":
                    tally["train_pairs"] += 1
                    tally["compact"] += max(m["shape"]) <= 10
                    tally["large"] += max(m["shape"]) >= 26
                    tally["large_class"] += m["large_class"]
                else:
                    tally["four_classes"] += m["n_prop"] >= 4
        for m in meta["pairs"]:
            tally["fills"].add((m["fill"], m["tight"]))
            tally["packed"] += m["placement"] == "packed"
            tally["cuts"] += m["fill"] == "cuts"
    tally["distinct"] = len(signatures)
    return tally


def self_test():
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {bad}")
    # Named transforms enumerate exactly the dihedral orbit.
    probe = frozenset({(0, 0), (1, 0), (2, 0), (2, 1), (0, 1), (0, 2)})
    _check(len(_orbit(probe)) == 8 and len({_transform(probe, t) for t in TRANSFORMS}) == 7, "transforms")
    # A hand-made pair: seeded L-room, its exact copy and its mirror, all fully outlined.
    demo = np.ones((7, 12), dtype=int)
    rooms = ([(1, 1), (2, 1), (2, 2)], [(1, 5), (2, 5), (2, 6)], [(1, 10), (2, 10), (2, 9)])
    for room in rooms:
        for r, c in room:
            for dr, dc in _N8:
                y, x = r + dr, c + dc
                if 0 <= y < 7 and 0 <= x < 12 and (y, x) not in room:
                    demo[y, x] = 4
    demo[1, 1] = 3
    answer = solve(demo)
    _check(answer[2, 2] == 3 and answer[2, 6] == 3 and answer[2, 9] == 1 and answer[6, 11] == 1,
           "hand-made pair: copy painted, mirror and open floor kept")
    _check(_apply(demo, **VARIANTS["dihedral"])[2, 9] == 3 and
           _apply(demo, **VARIANTS["flood_only"])[2, 6] == 1, "hand-made pair: shortcuts differ")
    tally = _tally(range(42))
    _check(tally["distinct"] == 42, "seeds must give distinct puzzles")
    _check(len(tally["sizes"]) >= 60, "grid shapes lack variety")
    _check(len(tally["fills"]) == 6, "all fill/outline combinations must occur")
    _check(len(tally["regimes"]) >= 6 and tally["palettes"] == {"fixed", "drift", "swap"}, "regimes lack variety")
    _check(tally["role_puzzles"] >= 10 and tally["swap_puzzles"] >= 3, "floor/wall roles never vary")
    _check(tally["compact"] >= 8 and tally["large"] >= 4, "training grid sizes lack range")
    _check(tally["domino"] >= 2 and tally["large_class"] >= 8 and tally["holed"] >= 15, "shape vocabulary too narrow")
    _check(tally["sibling"] >= 8 and tally["gallery"] >= 3 and tally["nested"] >= 6, "interactions too rare")
    _check(tally["same"] >= 8 and tally["rival"] >= 8, "contact relationships too uniform")
    _check(tally["multi"] >= 2 and tally["shared"] >= 2 and tally["edge_seeded"] >= 10, "seed patterns too uniform")
    _check(tally["transfer_free"] >= 2 and tally["reuse"] >= 5 and tally["four_classes"] >= 2,
           "regime extensions too rare")
    _check(tally["holes_refuted"] >= 15 and tally["mirror_refuted"] >= 25, "orientation/hole contrasts too rare")
    _check(tally["rect"] * 100 <= 22 * tally["prop"], "too many solid rectangles")
    _check(tally["packed"] >= 15 and tally["cuts"] >= 25, "layout styles too rare")
    _check(tally["test_not_larger"] >= 2, "tests are always larger than training")
    again, _ = _generate(7, 4, 1)
    first, _ = _generate(7, 4, 1)
    for split in ("train", "test"):
        for a, b in zip(again[split], first[split]):
            _check(np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"]),
                   "generation is not deterministic")
    big, _ = _generate(3, 10, 4)
    validate_generated_puzzle(big)
    small, _ = _generate(3, 2, 1)
    validate_generated_puzzle(small)
    return True
