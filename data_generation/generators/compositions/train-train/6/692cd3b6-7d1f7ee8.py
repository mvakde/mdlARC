"""Territory-colored shortest-route union through gated nested frames.

Rule (see task_spec.md): two isolated marker cells are terminals.  Rectangular
monochrome frames (possibly nested) are walls; each frame may have zero, one or
two door gaps in its border.  Every background cell on at least one shortest
four-neighbour route between the terminals is painted.  A painted cell lying in
the closed rectangle of an outermost (root) frame takes that root's colour; a
painted cell outside every root takes the marker colour.
"""
from __future__ import annotations

from collections import deque

import numpy as np

FOUR = ((1, 0), (-1, 0), (0, 1), (0, -1))
INF = 10 ** 9


# --------------------------------------------------------------------------
# Input-only parsing and solving
# --------------------------------------------------------------------------

def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != color:
                continue
            seen[r, c] = True
            stack, cells = [(r, c)], []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in FOUR:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and grid[yy, xx] == color:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            out.append(sorted(cells))
    return out


def _bbox(cells):
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _border(rect):
    r0, c0, r1, c1 = rect
    cells = {(r0, c) for c in range(c0, c1 + 1)} | {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _corners(rect):
    r0, c0, r1, c1 = rect
    return {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}


def _as_frame(cells, grid, bg, allowed):
    rect = _bbox(cells)
    if rect[2] - rect[0] < 2 or rect[3] - rect[1] < 2:
        return None
    ring = _border(rect)
    cellset = set(cells)
    if not cellset <= ring:
        return None
    missing = ring - cellset
    if len(missing) not in allowed or missing & _corners(rect):
        return None
    if any(grid[p] != bg for p in missing):
        return None
    return rect, sorted(missing)


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _marker(grid, bg):
    found = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        comps = _components(grid, color)
        if len(comps) == 2 and all(len(c) == 1 for c in comps):
            found.append((color, [comps[0][0], comps[1][0]]))
    if len(found) != 1:
        raise ValueError("expected exactly one colour made of two isolated marker cells")
    return found[0]


def parse(grid):
    """Return background, marker colour, terminals and frames (rect, colour, doors)."""
    grid = np.asarray(grid, dtype=int)
    bg = _background(grid)
    marker, terminals = _marker(grid, bg)
    frames = []
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, marker):
            continue
        leftovers = []
        for comp in _components(grid, color):
            got = _as_frame(comp, grid, bg, (0, 1))
            if got:
                frames.append({"rect": got[0], "color": color, "doors": got[1]})
            else:
                leftovers.append(comp)
        pieces = []
        for comp in leftovers:
            b = _bbox(comp)
            if b[2] > b[0] and b[3] > b[1]:  # two-door frame pieces are never straight
                pieces.append(comp)
        used = set()
        for i in range(len(pieces)):
            if i in used:
                continue
            for j in range(i + 1, len(pieces)):
                if j in used:
                    continue
                got = _as_frame(pieces[i] + pieces[j], grid, bg, (2,))
                if got:
                    frames.append({"rect": got[0], "color": color, "doors": got[1]})
                    used |= {i, j}
                    break
    frames.sort(key=lambda f: f["rect"])
    for f in frames:
        enclosing = [g for g in frames if g is not f and _strictly_inside(f["rect"], g["rect"])]
        f["depth"] = 1 + len(enclosing)
    return bg, marker, terminals, frames


def _strictly_inside(inner, outer):
    return outer[0] < inner[0] and outer[1] < inner[1] and inner[2] < outer[2] and inner[3] < outer[3]


def _distances(passable, start):
    h, w = passable.shape
    dist = np.full((h, w), INF, dtype=np.int64)
    dist[start] = 0
    queue = deque([start])
    while queue:
        y, x = queue.popleft()
        for dy, dx in FOUR:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and passable[yy, xx] and dist[yy, xx] == INF:
                dist[yy, xx] = dist[y, x] + 1
                queue.append((yy, xx))
    return dist


def route_union(grid, bg, marker, terminals):
    grid = np.asarray(grid, dtype=int)
    passable = (grid == bg) | (grid == marker)
    d1 = _distances(passable, terminals[0])
    d2 = _distances(passable, terminals[1])
    total = d1[terminals[1]]
    if total >= INF:
        raise ValueError("terminals are not connected")
    return (grid == bg) & (d1 + d2 == total), int(total)


def territory(shape, frames):
    """-1 outside all roots, else the colour of the root whose closed rectangle holds the cell."""
    terr = np.full(shape, -1, dtype=int)
    for f in frames:
        if f["depth"] == 1:
            r0, c0, r1, c1 = f["rect"]
            terr[r0:r1 + 1, c0:c1 + 1] = f["color"]
    return terr


def solve(input_grid):
    grid = np.asarray(input_grid, dtype=int)
    bg, marker, terminals, frames = parse(grid)
    union, _ = route_union(grid, bg, marker, terminals)
    terr = territory(grid.shape, frames)
    out = grid.copy()
    out[union] = np.where(terr[union] >= 0, terr[union], marker)
    return out


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _near(a, b):
    """True when rect a, grown by one cell, touches rect b."""
    return not (a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1])


def _overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


class _Layout:
    def __init__(self, H, W):
        self.H, self.W = H, W
        self.frames = []      # dicts: rect, color, parent, kind, ndoors
        self.obstacles = []   # dicts: rect, color, parent
        self.terminals = []
        self.kids = {-1: []}  # container index -> blocking rects

    def region(self, cont):
        if cont == -1:
            return (0, 0, self.H - 1, self.W - 1)
        r0, c0, r1, c1 = self.frames[cont]["rect"]
        return (r0 + 2, c0 + 2, r1 - 2, c1 - 2)

    def place(self, rng, cont, h, w, tries=80, region=None):
        reg = region or self.region(cont)
        if reg[2] - reg[0] + 1 < h or reg[3] - reg[1] + 1 < w:
            return None
        for _ in range(tries):
            r = int(rng.integers(reg[0], reg[2] - h + 2))
            c = int(rng.integers(reg[1], reg[3] - w + 2))
            rect = (r, c, r + h - 1, c + w - 1)
            if all(not _near(rect, b) for b in self.kids[cont]):
                return rect
        return None

    def add_frame(self, rect, color, cont, kind, ndoors):
        self.frames.append({"rect": rect, "color": color, "parent": cont, "kind": kind, "ndoors": ndoors})
        idx = len(self.frames) - 1
        self.kids[cont].append(rect)
        self.kids[idx] = []
        return idx

    def root_of(self, idx):
        while self.frames[idx]["parent"] != -1:
            idx = self.frames[idx]["parent"]
        return idx


def _chain_size(levels):
    return 3 + 4 * (levels - 1)


def _frame_color(rng, lay, cont, palette, avoid=()):
    parent_color = lay.frames[cont]["color"] if cont >= 0 else None
    choices = [c for c in palette if c != parent_color and c not in avoid]
    if not choices:
        choices = [c for c in palette if c != parent_color]
    return int(rng.choice(choices))


def _grow(rng, lay, cont, levels, palette, kind="chain", slack=4, root_avoid=()):
    """Place `levels` nested frames inside container; return innermost container."""
    for k in range(levels):
        need = _chain_size(levels - k)
        reg = lay.region(cont)
        ah, aw = reg[2] - reg[0] + 1, reg[3] - reg[1] + 1
        if ah < need or aw < need:
            return None
        h = int(rng.integers(need, min(ah, need + slack) + 1))
        w = int(rng.integers(need, min(aw, need + slack) + 1))
        rect = lay.place(rng, cont, h, w)
        if rect is None:
            return None
        avoid = root_avoid if cont == -1 else ()
        color = _frame_color(rng, lay, cont, palette, avoid)
        nd = 2 if rng.random() < 0.3 else 1
        cont = lay.add_frame(rect, color, cont, kind, nd)
    return cont


def _put_terminal(rng, lay, cont):
    reg = (0, 0, lay.H - 1, lay.W - 1) if cont == -1 else tuple(
        v + (1 if i < 2 else -1) for i, v in enumerate(lay.frames[cont]["rect"]))
    cells = []
    for r in range(reg[0], reg[2] + 1):
        for c in range(reg[1], reg[3] + 1):
            pt = (r, c, r, c)
            if any(_overlap(pt, b) for b in lay.kids[cont]):
                continue
            if any(abs(r - t[0]) + abs(c - t[1]) <= 1 for t in lay.terminals):
                continue
            cells.append((r, c))
    if not cells:
        return False
    r, c = cells[int(rng.integers(len(cells)))]
    lay.terminals.append((r, c))
    lay.kids[cont].append((r, c, r, c))
    return True


def _decorate(rng, lay, cont, count, palette, allow_frames=True):
    for _ in range(count):
        if allow_frames and rng.random() < 0.4:
            h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
            rect = lay.place(rng, cont, h, w, tries=30)
            if rect is not None:
                color = _frame_color(rng, lay, cont, palette)
                nd = int(rng.choice([0, 1, 1, 2]))
                lay.add_frame(rect, color, cont, "decoy", nd)
            continue
        a, b = int(rng.integers(1, 4)), int(rng.integers(2, 5))
        h, w = (a, b) if rng.random() < 0.5 else (b, a)
        rect = lay.place(rng, cont, h, w, tries=30)
        if rect is not None:
            lay.obstacles.append({"rect": rect, "color": int(rng.choice(palette)), "parent": cont})
            lay.kids[cont].append(rect)


def _build(rng, regime, depth_a, depth_b, H, W, palette):
    lay = _Layout(H, W)
    if regime == "same":
        a, b = depth_a - 1, depth_b - 1  # levels below the shared root
        sa, sb = _chain_size(a), (_chain_size(b) if b > 0 else 1)
        long_side, short_side = sa + sb + 5, max(sa, sb) + 4
        if rng.random() < 0.5:
            rh, rw = short_side, long_side
        else:
            rh, rw = long_side, short_side
        rh += int(rng.integers(0, 4))
        rw += int(rng.integers(0, 4))
        if rh > 30 or rw > 30:
            return None
        lay = _Layout(max(H, min(30, rh + int(rng.integers(0, 5)))), max(W, min(30, rw + int(rng.integers(0, 5)))))
    if regime == "via":
        need_a = _chain_size(depth_a) if depth_a > 0 else 1
        need_b = _chain_size(depth_b) if depth_b > 0 else 1
        vw = int(rng.integers(5, 11))
        left_zone = need_a + int(rng.integers(0, 4))
        right_zone = need_b + int(rng.integers(0, 4))
        W = left_zone + 1 + vw + 1 + right_zone
        H = max(H, need_a + 2, need_b + 2)
        if W > 30 or H > 30:
            return None
        lay = _Layout(H, W)
    if regime == "cross":
        roots = []
        for d in sorted((depth_a, depth_b), reverse=True):
            inner = _grow(rng, lay, -1, d, palette, root_avoid=[lay.frames[r]["color"] for r in roots])
            if inner is None:
                return None
            roots.append(lay.root_of(inner))
            _decorate_chain(rng, lay, inner, palette)
            if not _put_terminal(rng, lay, inner):
                return None
    elif regime == "same":
        rect = lay.place(rng, -1, rh, rw)
        if rect is None:
            return None
        root = lay.add_frame(rect, int(rng.choice(palette)), -1, "chain", 2 if rng.random() < 0.3 else 1)
        R = lay.region(root)
        if rh <= rw:   # split interior left/right
            cut = R[1] + sa + int(rng.integers(0, max(1, (R[3] - R[1] + 1) - sa - sb - 1) + 1))
            zones = [(R[0], R[1], R[2], cut - 1), (R[0], cut + 1, R[2], R[3])]
        else:          # split interior top/bottom
            cut = R[0] + sa + int(rng.integers(0, max(1, (R[2] - R[0] + 1) - sa - sb - 1) + 1))
            zones = [(R[0], R[1], cut - 1, R[3]), (cut + 1, R[1], R[2], R[3])]
        if rng.random() < 0.5:
            zones = zones[::-1]
            a, b = b, a
        for levels, zone in zip((a, b), zones):
            if levels > 0:
                inner = _grow_region(rng, lay, zone, levels, palette, (), cont=root)
                if inner is None:
                    return None
                _decorate_chain(rng, lay, inner, palette)
            else:
                inner = root
            if not _put_terminal(rng, lay, inner):
                return None
        _decorate(rng, lay, root, int(rng.integers(0, 3)), palette)
    elif regime == "open":
        inner = _grow(rng, lay, -1, depth_a, palette)
        if inner is None:
            return None
        _decorate_chain(rng, lay, inner, palette)
        if not _put_terminal(rng, lay, inner):
            return None
        if not _put_terminal(rng, lay, -1):
            return None
    elif regime == "via":
        # canonical: left terminal zone | tall two-door root | right terminal zone
        H = lay.H
        vh = H if rng.random() < 0.6 else H - int(rng.integers(1, 4))
        r0 = int(rng.integers(0, H - vh + 1))
        left_w = left_zone + 1
        vrect = (r0, left_w, r0 + vh - 1, left_w + vw - 1)
        v = lay.add_frame(vrect, int(rng.choice(palette)), -1, "via", 2)
        _decorate(rng, lay, v, int(rng.integers(1, 3)), palette)
        left = (0, 0, H - 1, left_zone - 1)
        right = (0, left_w + vw + 1, H - 1, lay.W - 1)
        avoid = [lay.frames[v]["color"]]
        for side, d in ((left, depth_a), (right, depth_b)):
            if d == 0:
                if not _put_terminal_region(rng, lay, side):
                    return None
                continue
            inner = _grow_region(rng, lay, side, d, palette, avoid)
            if inner is None:
                return None
            avoid.append(lay.frames[lay.root_of(inner)]["color"])
            _decorate_chain(rng, lay, inner, palette)
            if not _put_terminal(rng, lay, inner):
                return None
    else:
        raise ValueError(regime)
    # optional decoy roots and loose outside obstacles
    for _ in range(int(rng.integers(0, 3))):
        if rng.random() < 0.5:
            h, w = int(rng.integers(3, 8)), int(rng.integers(3, 8))
            rect = lay.place(rng, -1, h, w, tries=30)
            if rect is not None:
                idx = lay.add_frame(rect, _frame_color(rng, lay, -1, palette), -1, "decoy",
                                    int(rng.choice([0, 1, 2, 2])))
                if h >= 7 and w >= 7 and rng.random() < 0.5:
                    _decorate(rng, lay, idx, 1, palette)
        else:
            _decorate(rng, lay, -1, 1, palette, allow_frames=False)
    return lay


def _grow_region(rng, lay, region, levels, palette, avoid, cont=-1):
    need = _chain_size(levels)
    ah, aw = region[2] - region[0] + 1, region[3] - region[1] + 1
    if ah < need or aw < need:
        return None
    h = int(rng.integers(need, min(ah, need + 4) + 1))
    w = int(rng.integers(need, min(aw, need + 3) + 1))
    rect = lay.place(rng, cont, h, w, region=region)
    if rect is None:
        return None
    color = _frame_color(rng, lay, cont, palette, avoid)
    root = lay.add_frame(rect, color, cont, "chain", 2 if rng.random() < 0.3 else 1)
    if levels == 1:
        return root
    return _grow(rng, lay, root, levels - 1, palette, slack=2)


def _put_terminal_region(rng, lay, region):
    cells = []
    for r in range(region[0], region[2] + 1):
        for c in range(region[1], region[3] + 1):
            pt = (r, c, r, c)
            if any(_near(pt, b) for b in lay.kids[-1]):
                continue
            cells.append((r, c))
    if not cells:
        return False
    r, c = cells[int(rng.integers(len(cells)))]
    lay.terminals.append((r, c))
    lay.kids[-1].append((r, c, r, c))
    return True


def _decorate_chain(rng, lay, inner, palette):
    idx = inner
    while idx != -1:
        if rng.random() < 0.45:
            _decorate(rng, lay, idx, int(rng.integers(1, 3)), palette)
        idx = lay.frames[idx]["parent"]


def _door_options(lay, grid, rect, passable):
    r0, c0, r1, c1 = rect
    opts = []
    H, W = grid.shape
    for side in range(4):
        if side == 0:
            cells, out_d = [(r0, c) for c in range(c0 + 1, c1)], (-1, 0)
        elif side == 1:
            cells, out_d = [(r1, c) for c in range(c0 + 1, c1)], (1, 0)
        elif side == 2:
            cells, out_d = [(r, c0) for r in range(r0 + 1, r1)], (0, -1)
        else:
            cells, out_d = [(r, c1) for r in range(r0 + 1, r1)], (0, 1)
        for (y, x) in cells:
            oy, ox = y + out_d[0], x + out_d[1]
            iy, ix = y - out_d[0], x - out_d[1]
            if not (0 <= oy < H and 0 <= ox < W):
                continue
            if passable[oy, ox] and passable[iy, ix]:
                opts.append((side, (y, x)))
    return opts


def _render(rng, lay, bg, marker):
    H, W = lay.H, lay.W
    grid = np.full((H, W), bg, dtype=int)
    for f in lay.frames:
        for p in _border(f["rect"]):
            grid[p] = f["color"]
    for o in lay.obstacles:
        r0, c0, r1, c1 = o["rect"]
        grid[r0:r1 + 1, c0:c1 + 1] = o["color"]
    for t in lay.terminals:
        grid[t] = marker
    passable = (grid == bg) | (grid == marker)
    for f in lay.frames:
        opts = _door_options(lay, grid, f["rect"], passable)
        k = f["ndoors"]
        doors = None
        for _ in range(25):
            if k == 0:
                doors = []
                break
            if len(opts) < k:
                break
            picks = [opts[i] for i in rng.choice(len(opts), size=k, replace=False)]
            if k == 2 and picks[0][0] == picks[1][0]:
                continue
            if f["kind"] == "via" and {picks[0][0], picks[1][0]} != {2, 3}:
                continue
            cand = [p for _, p in picks]
            if k == 2:
                ring = _border(f["rect"]) - set(cand)
                tmp = np.zeros((H, W), dtype=int)
                for p in ring:
                    tmp[p] = 1
                pieces = _components(tmp, 1)
                if len(pieces) != 2 or any(_bbox(pc)[2] == _bbox(pc)[0] or _bbox(pc)[3] == _bbox(pc)[1]
                                           for pc in pieces):
                    continue
            doors = cand
            break
        if doors is None:
            if f["kind"] == "decoy":
                doors = []
            else:
                return None
        f["doors"] = sorted(doors)
    for f in lay.frames:
        for p in f["doors"]:
            grid[p] = bg
    return grid


def _truth(lay, grid, bg, marker):
    """Expected output computed from the generator's own layout (not from parsing)."""
    union, total = route_union(grid, bg, marker, lay.terminals)
    terr = np.full(grid.shape, -1, dtype=int)
    for f in lay.frames:
        if f["parent"] == -1:
            r0, c0, r1, c1 = f["rect"]
            terr[r0:r1 + 1, c0:c1 + 1] = f["color"]
    out = grid.copy()
    out[union] = np.where(terr[union] >= 0, terr[union], marker)
    return out, union, total


def _features(lay, grid, union, total):
    H, W = grid.shape
    nearest = np.full((H, W), -1, dtype=int)
    nearest_depth = np.zeros((H, W), dtype=int)
    rootcol = np.full((H, W), -1, dtype=int)

    def depth(i):
        d = 1
        while lay.frames[i]["parent"] != -1:
            i = lay.frames[i]["parent"]
            d += 1
        return d

    order = sorted(range(len(lay.frames)), key=depth)
    for i in order:
        r0, c0, r1, c1 = lay.frames[i]["rect"]
        nearest[r0:r1 + 1, c0:c1 + 1] = lay.frames[i]["color"]
        nearest_depth[r0:r1 + 1, c0:c1 + 1] = depth(i)
        if lay.frames[i]["parent"] == -1:
            rootcol[r0:r1 + 1, c0:c1 + 1] = lay.frames[i]["color"]
    deep = union & (nearest_depth >= 2) & (nearest != rootcol)
    term_roots = set()
    for t in lay.terminals:
        for i, f in enumerate(lay.frames):
            r0, c0, r1, c1 = f["rect"]
            if f["parent"] == -1 and r0 <= t[0] <= r1 and c0 <= t[1] <= c1:
                term_roots.add(i)
    through = False
    for i, f in enumerate(lay.frames):
        if f["parent"] == -1 and i not in term_roots:
            r0, c0, r1, c1 = f["rect"]
            if union[r0:r1 + 1, c0:c1 + 1].any():
                through = True
    max_depth = max([depth(i) for i in range(len(lay.frames))] or [0])
    return {
        "outside": bool((union & (rootcol < 0)).any()),
        "inside": bool((union & (rootcol >= 0)).any()),
        "deep": bool(deep.any()),
        "multi": int(union.sum()) > total - 1,
        "through": through,
        "max_depth": max_depth,
        "two_door": any(len(f["doors"]) == 2 for f in lay.frames),
    }


_TRANSFORMS = 8


def _d4(a, k):
    a = np.rot90(a, k % 4)
    return np.fliplr(a) if k >= 4 else a


def _make_pair(rng, params, need, depth_cap, used):
    bg, palette_all = params["bg"], params["palette"]
    for _ in range(400):
        marker = params["marker"] if params["marker"] is not None else int(rng.choice(palette_all))
        palette = [c for c in palette_all if c != marker]
        regimes = list(need.get("regimes") or ["cross", "open", "same", "via"])
        weights = np.array([{"cross": 4, "open": 3, "same": 3, "via": 3}[r] for r in regimes], dtype=float)
        regime = regimes[int(rng.choice(len(regimes), p=weights / weights.sum()))]
        lo = need.get("min_depth", 1)
        da = int(rng.integers(max(lo, 2 if need.get("deep") else 1), depth_cap + 1))
        if regime == "via":
            db = int(rng.integers(0, min(depth_cap, 3) + 1))
            da = min(da, 3)
        elif regime == "same":
            da = max(da, 2)
            db = int(rng.integers(1, max(1, min(depth_cap, 3)) + 1))
        else:
            db = int(rng.integers(1, depth_cap + 1))
        minside = _chain_size(max(da, db, 1)) + 2
        H = int(rng.integers(max(12, minside), 31))
        W = int(rng.integers(max(12, minside), 31))
        if regime == "via":
            W = int(rng.integers(20, 31))
        lay = _build(rng, regime, da, db, H, W, palette)
        if lay is None or len(lay.terminals) != 2:
            continue
        grid = _render(rng, lay, bg, marker)
        if grid is None:
            continue
        try:
            out, union, total = _truth(lay, grid, bg, marker)
        except ValueError:
            continue
        feats = _features(lay, grid, union, total)
        if not feats["inside"]:
            continue
        if need.get("deep") and not feats["deep"]:
            continue
        if need.get("outside") and not feats["outside"]:
            continue
        if (need.get("through") or regime == "via") and not feats["through"]:
            continue
        if need.get("min_depth") and feats["max_depth"] < need["min_depth"]:
            continue
        # background must dominate, palette roles must parse back exactly
        vals, cnts = np.unique(grid, return_counts=True)
        if int(vals[np.argmax(cnts)]) != bg or (len(cnts) > 1 and sorted(cnts)[-1] == sorted(cnts)[-2]):
            continue
        try:
            parsed = parse(grid)
        except ValueError:
            continue
        design = sorted((f["rect"], f["color"], tuple(f["doors"])) for f in lay.frames)
        found = sorted((f["rect"], f["color"], tuple(f["doors"])) for f in parsed[3])
        if parsed[1] != marker or design != found or sorted(parsed[2]) != sorted(lay.terminals):
            continue
        k = int(rng.integers(_TRANSFORMS))
        g2, o2 = _d4(grid, k).copy(), _d4(out, k).copy()
        try:
            pbg, pm, pt, pframes = parse(g2)
            solved = solve(g2)
        except ValueError:
            continue
        if pm != marker or len(pframes) != len(lay.frames):
            continue
        if not np.array_equal(solved, o2):
            continue
        key = g2.tobytes() + bytes(g2.shape)
        if key in used:
            continue
        used.add(key)
        feats["regime"] = regime
        return {"input": g2.astype(int), "output": o2.astype(int)}, feats
    raise RuntimeError("could not generate a pair")


def _check_counts(num_train, num_test):
    for name, v in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("need 2..10 train, 1..4 test and num_train > num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    bg = 0 if rng.random() < 0.7 else int(rng.integers(1, 10))
    others = [c for c in range(10) if c != bg]
    rng.shuffle(others)
    n_colors = int(rng.integers(5, 10))
    palette = [int(c) for c in others[:n_colors]]
    fixed_marker = rng.random() < 0.5
    marker = int(palette[0]) if fixed_marker else None
    pal = palette[1:] if fixed_marker else palette
    params = {"bg": bg, "palette": pal, "marker": marker}
    train_cap = int(rng.choice([2, 3, 3]))
    test_shift = rng.random() < 0.6 and train_cap < 4
    used = set()
    train, feats = [], []
    for i in range(num_train):
        need = {}
        if i < 2:
            need = {"deep": True, "outside": True, "regimes": ["cross", "open", "via"]}
        elif rng.random() < 0.3:
            need = {"deep": True}
        pair, f = _make_pair(rng, params, need, train_cap, used)
        train.append(pair)
        feats.append(f)
    order = rng.permutation(num_train)
    train = [train[i] for i in order]
    test = []
    max_train_depth = max(f["max_depth"] for f in feats)
    for _ in range(num_test):
        cap = min(4, train_cap + 1) if test_shift else train_cap
        need = {"deep": True}
        if test_shift:
            need["min_depth"] = min(cap, max_train_depth + 1)
        pair, _ = _make_pair(rng, params, need, cap, used)
        test.append(pair)
    return {"train": train, "test": test}


# --------------------------------------------------------------------------
# Validation and self test
# --------------------------------------------------------------------------

def _pair_report(inp, out):
    inp = np.asarray(inp, dtype=int)
    out = np.asarray(out, dtype=int)
    bg, marker, terminals, frames = parse(inp)
    union, total = route_union(inp, bg, marker, terminals)
    terr = territory(inp.shape, frames)
    nearest = np.full(inp.shape, -1, dtype=int)
    for f in sorted(frames, key=lambda f: f["depth"]):
        r0, c0, r1, c1 = f["rect"]
        nearest[r0:r1 + 1, c0:c1 + 1] = f["color"]
    plain = inp.copy()
    plain[union] = marker                         # shortcut: ignore territory (drop parent B)
    near = inp.copy()
    near[union] = np.where(nearest[union] >= 0, nearest[union], marker)  # nearest-frame shortcut
    return {
        "bg": bg, "marker": marker, "frames": frames, "union": union, "total": total,
        "inside": bool((union & (terr >= 0)).any()),
        "outside": bool((union & (terr < 0)).any()),
        "beats_plain": not np.array_equal(plain, out),
        "beats_nearest": not np.array_equal(near, out),
        "multi": int(union.sum()) > total - 1,
        "max_depth": max([f["depth"] for f in frames] or [0]),
    }


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must have exactly train/test")
    seen = set()
    reports = {}
    for split in ("train", "test"):
        reports[split] = []
        for pair in puzzle[split]:
            if set(pair) != {"input", "output"}:
                raise ValueError("pair keys")
            inp = np.asarray(pair["input"], dtype=int)
            out = np.asarray(pair["output"], dtype=int)
            if inp.ndim != 2 or inp.shape != out.shape or not all(1 <= n <= 30 for n in inp.shape):
                raise ValueError("bad grid shape")
            if inp.min() < 0 or inp.max() > 9 or out.min() < 0 or out.max() > 9:
                raise ValueError("bad colour")
            key = inp.tobytes() + bytes(inp.shape)
            if key in seen:
                raise ValueError("duplicate input")
            seen.add(key)
            if not np.array_equal(solve(inp), out):
                raise ValueError("output disagrees with rule")
            rep = _pair_report(inp, out)
            changed = inp != out
            if changed.any() and not (inp[changed] == rep["bg"]).all():
                raise ValueError("only background cells may change")
            if not rep["inside"]:
                raise ValueError("route must enter some root territory")
            fr = rep["frames"]
            for i, a in enumerate(fr):
                for b in fr[i + 1:]:
                    ra, rb = a["rect"], b["rect"]
                    if not (_strictly_inside(ra, rb) or _strictly_inside(rb, ra) or not _near(ra, rb)):
                        raise ValueError("frames must be disjoint (with a gap) or strictly nested")
            reports[split].append(rep)
    tr = reports["train"]
    need = min(2, len(tr))
    if sum(r["outside"] for r in tr) < need:
        raise ValueError("training must show the open-field (marker colour) segment")
    if sum(r["beats_nearest"] for r in tr) < need:
        raise ValueError("training must separate root colour from nearest-frame colour")
    if not all(r["beats_plain"] for r in tr + reports["test"]):
        raise ValueError("every pair must need territory colouring")
    if not all(r["beats_nearest"] for r in reports["test"]):
        raise ValueError("test must need root (not nearest) colour")
    if not any(r["multi"] for r in tr):
        raise ValueError("training must show a multi-route union")
    return True


SELF_TEST_SEEDS = list(range(42))


def self_test():
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted {bad}")
    shapes, depths, bgs, sigs = set(), set(), set(), set()
    regimes = {}
    for seed in SELF_TEST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        sigs.add(str([np.asarray(x["input"]).tolist() for x in p["train"]]))
        for pair in p["train"] + p["test"]:
            inp = np.asarray(pair["input"])
            shapes.add(inp.shape)
            rep = _pair_report(inp, pair["output"])
            depths.add(rep["max_depth"])
            bgs.add(rep["bg"])
            key = (rep["outside"], any(len(f["doors"]) == 2 for f in rep["frames"]))
            regimes[key] = regimes.get(key, 0) + 1
    assert len(sigs) == len(SELF_TEST_SEEDS), "seeds must give distinct puzzles"
    assert len(shapes) >= 30, f"too few grid shapes {len(shapes)}"
    assert {2, 3, 4} <= depths, f"nesting depths {depths}"
    assert len(bgs) >= 3, "background colour should vary"
    assert len(regimes) >= 4, f"route regimes {regimes}"
    return True


if __name__ == "__main__":
    self_test()
    print("ok")
