"""Hazard-aware routing through a city of marker-studded blocks.

Composition of parent A (a04b2602: markers embedded in latent carrier rectangles
emit a clipped 3x3 halo, identical markers outside rectangles are inert) and
parent B (b15fca0b: draw the unique shortest orthogonal route between two
terminals through background, around impassable obstacles).

Rule: the carrier rectangles are city blocks separated by background streets.
Every marker embedded in a block is live: its 3x3 neighbourhood (clipped at the
canvas, markers preserved) is painted with the fixed halo colour 1, and those
halo cells, spilling into the street, are impassable.  Markers lying in the
street are inert duds: they are single impassable cells but emit no halo.  The
unique shortest 4-neighbour route between the two terminal cells through the
remaining street cells is painted with the fixed route colour 4.

Only the Python standard library and NumPy are used.
"""
from __future__ import annotations

from collections import deque

import numpy as np

HALO = 1
ROUTE = 4
INPUT_COLORS = (0, 2, 3, 5, 6, 7, 8, 9)
N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


# --------------------------------------------------------------------------
# Shared analysis (used by solve and by validation)
# --------------------------------------------------------------------------

def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack, cells = [(r, c)], []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in N4:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                comps.append(cells)
    return comps


def _bbox(cells):
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1


def _all_rectangular(grid, color, marker):
    """Every 4-component of `color` fills its bounding box together with markers."""
    comps = _components(grid == color)
    if not comps:
        return False
    for cells in comps:
        r0, r1, c0, c1 = _bbox(cells)
        sub = grid[r0:r1, c0:c1]
        if not np.all((sub == color) | (sub == marker)):
            return False
    return True


def _infer_roles(grid):
    grid = np.asarray(grid)
    colors, counts = np.unique(grid, return_counts=True)
    colors = [int(c) for c in colors]
    counts = [int(n) for n in counts]
    if len(colors) != 4:
        raise ValueError("input must use exactly four colors")
    terminals = [c for c, n in zip(colors, counts) if n == 2]
    if len(terminals) != 1:
        raise ValueError("terminal color (exactly two cells) is not unique")
    terminal = terminals[0]
    rest = sorted((n, c) for c, n in zip(colors, counts) if c != terminal)
    if rest[0][0] >= rest[1][0]:
        raise ValueError("marker color is not the unique rarest color")
    marker = rest[0][1]
    a, b = rest[1][1], rest[2][1]
    ra, rb = _all_rectangular(grid, a, marker), _all_rectangular(grid, b, marker)
    if ra == rb:
        raise ValueError("carrier/background roles are ambiguous")
    carrier, background = (a, b) if ra else (b, a)
    return {"background": background, "carrier": carrier, "marker": marker, "terminal": terminal}


def _bfs(passable, start):
    h, w = passable.shape
    dist = np.full((h, w), -1, dtype=int)
    dist[start] = 0
    queue = deque([start])
    order = [start]
    while queue:
        y, x = queue.popleft()
        for dy, dx in N4:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and passable[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = dist[y, x] + 1
                queue.append((ny, nx))
                order.append((ny, nx))
    return dist, order


def _route_stats(passable, s, t):
    """Return (distance, capped path count, dist-from-s, path-count array)."""
    dist, order = _bfs(passable, s)
    if dist[t] < 0:
        return -1, 0, dist, None
    h, w = passable.shape
    count = np.zeros((h, w), dtype=np.int64)
    count[s] = 1
    for y, x in order[1:]:
        total = 0
        for dy, dx in N4:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and dist[ny, nx] == dist[y, x] - 1:
                total += count[ny, nx]
        count[y, x] = min(total, 1 << 40)
    return int(dist[t]), int(count[t]), dist, count


def _trace(dist, count, s, t, rng=None):
    """Walk back from t to s; random predecessor when rng is given."""
    h, w = dist.shape
    path = [t]
    cur = t
    while cur != s:
        y, x = cur
        preds = []
        for dy, dx in N4:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and dist[ny, nx] == dist[y, x] - 1 and count[ny, nx] > 0:
                preds.append((ny, nx))
        cur = preds[int(rng.integers(len(preds)))] if rng is not None else preds[0]
        path.append(cur)
    return path[::-1]


def _analyze(grid, mode="rule"):
    """Full analysis of an input grid.

    mode "rule": the composed rule.  "ignore_halo": halos are not obstacles.
    "all_live": every marker (including duds) emits a halo obstacle.
    """
    grid = np.asarray(grid, dtype=int)
    roles = _infer_roles(grid)
    bg, carrier, marker, terminal = (roles[k] for k in ("background", "carrier", "marker", "terminal"))
    rects = [_bbox(cells) for cells in _components(grid == carrier)]
    marker_mask = grid == marker
    inside = np.zeros_like(marker_mask)
    for r0, r1, c0, c1 in rects:
        inside[r0:r1, c0:c1] = True
    live = marker_mask & inside if mode != "all_live" else marker_mask.copy()
    h, w = grid.shape
    halo = np.zeros_like(marker_mask)
    for y, x in zip(*np.nonzero(live)):
        halo[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
    halo &= ~marker_mask
    halo &= grid != terminal
    terms = [tuple(int(v) for v in p) for p in zip(*np.nonzero(grid == terminal))]
    passable = (grid == bg) | (grid == terminal)
    if mode != "ignore_halo":
        passable &= ~halo
    s, t = terms
    d, n, dist, count = _route_stats(passable, s, t)
    path = _trace(dist, count, s, t) if n >= 1 else []
    return {"roles": roles, "rects": rects, "live": live, "duds": marker_mask & ~inside,
            "halo": halo, "passable": passable, "terminals": terms,
            "distance": d, "count": n, "path": path}


def _render(grid, info):
    out = np.array(grid, dtype=int).copy()
    out[info["halo"]] = HALO
    for cell in info["path"][1:-1]:
        out[cell] = ROUTE
    return out


def solve(input_grid):
    grid = np.asarray(input_grid, dtype=int)
    return _render(grid, _analyze(grid))


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _layout(rng, h, w, margin, maxb):
    blocks = []

    def split(r0, r1, c0, c1):
        hh, ww = r1 - r0, c1 - c0
        can_h, can_v = hh >= 5, ww >= 5
        must = hh > maxb or ww > maxb
        if (can_h or can_v) and (must or rng.random() < 0.3):
            if can_h and can_v:
                horizontal = rng.random() < hh / (hh + ww)
            else:
                horizontal = can_h
            if horizontal:
                s = int(rng.integers(r0 + 2, r1 - 2))
                split(r0, s, c0, c1)
                split(s + 1, r1, c0, c1)
            else:
                s = int(rng.integers(c0 + 2, c1 - 2))
                split(r0, r1, c0, s)
                split(r0, r1, s + 1, c1)
        else:
            blocks.append((r0, r1, c0, c1))

    if margin:
        split(1, h - 1, 1, w - 1)
    else:
        split(0, h, 0, w)
    return blocks


class _Pair:
    """Mutable construction state for one input grid."""

    def __init__(self, h, w, blocks):
        self.h, self.w = h, w
        self.blocks = blocks
        self.block_of = -np.ones((h, w), dtype=int)
        for i, (r0, r1, c0, c1) in enumerate(blocks):
            self.block_of[r0:r1, c0:c1] = i
        self.live = set()
        self.duds = set()
        self.terms = []
        self.halo = np.zeros((h, w), dtype=bool)

    def street(self):
        return self.block_of < 0

    def recompute_halo(self):
        self.halo[:] = False
        for y, x in self.live:
            self.halo[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
        for p in self.live | self.duds:
            self.halo[p] = False

    def passable(self, use_halo=True):
        m = self.street().copy()
        for p in self.duds:
            m[p] = False
        if use_halo:
            m &= ~self.halo
        for p in self.terms:
            m[p] = True
        return m

    def block_ok(self, index, extra):
        r0, r1, c0, c1 = self.blocks[index]
        marks = {p for p in self.live if self.block_of[p] == index} | {extra}
        sub = np.ones((r1 - r0, c1 - c0), dtype=bool)
        for y, x in marks:
            sub[y - r0, x - c0] = False
        for side in (sub[0, :], sub[-1, :], sub[:, 0], sub[:, -1]):
            if side.sum() < max(1, (len(side) + 1) // 2):
                return False
        if sub.sum() < 3:
            return False
        return len(_components(sub)) == 1

    def can_live(self, cell, protect=()):
        y, x = cell
        b = self.block_of[cell]
        if b < 0 or cell in self.live:
            return False
        if not self.block_ok(b, cell):
            return False
        for yy in range(max(0, y - 1), min(self.h, y + 2)):
            for xx in range(max(0, x - 1), min(self.w, x + 2)):
                p = (yy, xx)
                if p in self.duds or p in self.terms or p in protect:
                    return False
        return True

    def add_live(self, cell):
        self.live.add(cell)
        self.recompute_halo()

    def can_dud(self, cell, protect=()):
        return (self.block_of[cell] < 0 and cell not in self.duds and cell not in self.terms
                and not self.halo[cell] and cell not in protect)

    def is_edge(self, cell):
        b = self.block_of[cell]
        if b < 0:
            return False
        r0, r1, c0, c1 = self.blocks[b]
        y, x = cell
        return y in (r0, r1 - 1) or x in (c0, c1 - 1)

    def is_interior(self, cell):
        b = self.block_of[cell]
        if b < 0:
            return False
        r0, r1, c0, c1 = self.blocks[b]
        y, x = cell
        return r0 < y < r1 - 1 and c0 < x < c1 - 1

    def grid(self, roles):
        g = np.full((self.h, self.w), roles["background"], dtype=int)
        g[self.block_of >= 0] = roles["carrier"]
        for p in self.live | self.duds:
            g[p] = roles["marker"]
        for p in self.terms:
            g[p] = roles["terminal"]
        return g


def _cheb_neighbors(cell, h, w):
    y, x = cell
    out = []
    for yy in range(max(0, y - 1), min(h, y + 2)):
        for xx in range(max(0, x - 1), min(w, x + 2)):
            if (yy, xx) != cell:
                out.append((yy, xx))
    return out


def _try_pair(rng, cfg):
    h = int(rng.integers(cfg["hmin"], cfg["hmax"] + 1))
    w = int(rng.integers(cfg["wmin"], cfg["wmax"] + 1))
    margin = cfg["margin"]
    blocks = _layout(rng, h, w, margin, cfg["maxb"])
    if len(blocks) < 3:
        return None
    st = _Pair(h, w, blocks)
    street = st.street()

    # Terminals.
    if cfg["interior_terms"]:
        cands = [tuple(p) for p in np.argwhere(street)]
    else:
        cands = [tuple(p) for p in np.argwhere(street)
                 if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
    cands = [(int(a), int(b)) for a, b in cands]
    if len(cands) < 2:
        return None
    need = max(6, (h + w) // 2)
    for _ in range(40):
        i, j = rng.choice(len(cands), 2, replace=False)
        s, t = cands[int(i)], cands[int(j)]
        if abs(s[0] - t[0]) + abs(s[1] - t[1]) >= need:
            break
    else:
        return None
    st.terms = [s, t]

    edge_cells = [(int(a), int(b)) for a, b in np.argwhere(~street)]

    # Decorative live markers (interior and/or canvas-edge clipped).
    if cfg["want_interior"]:
        inter = [p for p in edge_cells if st.is_interior(p)]
        rng.shuffle(inter)
        for p in inter:
            if st.can_live(p):
                st.add_live(p)
                break
        else:
            return None
    if cfg["want_clip"]:
        clip = [p for p in edge_cells if st.is_edge(p) and (p[0] in (0, h - 1) or p[1] in (0, w - 1))]
        rng.shuffle(clip)
        for p in clip:
            if st.can_live(p):
                st.add_live(p)
                break
        else:
            return None
    for _ in range(int(rng.integers(0, cfg["extra_live"] + 1))):
        p = edge_cells[int(rng.integers(len(edge_cells)))]
        if st.can_live(p):
            st.add_live(p)
    street_cells = [(int(a), int(b)) for a, b in np.argwhere(street)]
    for _ in range(int(rng.integers(0, cfg["extra_duds"] + 1))):
        p = street_cells[int(rng.integers(len(street_cells)))]
        if st.can_dud(p) and all(abs(p[0] - q[0]) + abs(p[1] - q[1]) > 1 for q in st.terms):
            st.duds.add(p)

    d, n, dist, count = _route_stats(st.passable(), s, t)
    if n == 0:
        return None

    # Roadblocks: live edge markers whose halo spills onto a current shortest route.
    want_rb = int(rng.integers(cfg["rb_min"], cfg["rb_max"] + 1))
    placed = 0
    for _ in range(want_rb * 4):
        if placed >= want_rb:
            break
        d, n, dist, count = _route_stats(st.passable(), s, t)
        if n == 0:
            return None
        path = _trace(dist, count, s, t, rng)
        inner = path[2:-2]
        if not inner:
            break
        c = inner[int(rng.integers(len(inner)))]
        opts = [p for p in _cheb_neighbors(c, h, w) if st.is_edge(p) and st.can_live(p)]
        if not opts:
            continue
        p = opts[int(rng.integers(len(opts)))]
        st.add_live(p)
        d2, n2, _, _ = _route_stats(st.passable(), s, t)
        if n2 == 0:
            st.live.discard(p)
            st.recompute_halo()
            continue
        placed += 1
    if placed == 0:
        return None

    d, n, dist, count = _route_stats(st.passable(), s, t)
    d_ignore, _, _, _ = _route_stats(st.passable(use_halo=False), s, t)
    if n == 0 or d <= d_ignore:
        return None

    # Tie-breaking: fix one shortest route P and block alternatives.
    path = _trace(dist, count, s, t, rng)
    pset = set(path)
    for _ in range(cfg["max_ties"]):
        d2, n, dist, count = _route_stats(st.passable(), s, t)
        if n == 1:
            break
        dist_t, _ = _bfs(st.passable(), t)
        dag = [(int(a), int(b)) for a, b in np.argwhere((dist >= 0) & (dist_t >= 0) & (dist + dist_t == d2))]
        dag = [p for p in dag if p not in pset]
        if not dag:
            return None
        c = dag[int(rng.integers(len(dag)))]
        done = False
        if rng.random() < cfg["tie_live"]:
            opts = [p for p in _cheb_neighbors(c, h, w)
                    if st.is_edge(p) and st.can_live(p, protect=pset)]
            if opts:
                st.add_live(opts[int(rng.integers(len(opts)))])
                done = True
        if not done:
            if not st.can_dud(c, protect=pset):
                return None
            st.duds.add(c)
    d2, n, dist, count = _route_stats(st.passable(), s, t)
    if n != 1 or d2 != d:
        return None
    path = _trace(dist, count, s, t)
    pset = set(path)

    # Dud beside the route (would block it if duds were live).
    inner = set(path[1:-1])
    near = any(any(q in inner for q in _cheb_neighbors(p, h, w)) for p in st.duds)
    if not near:
        opts = set()
        for c in path[1:-1]:
            for q in _cheb_neighbors(c, h, w):
                if st.can_dud(q, protect=pset):
                    opts.add(q)
        opts = sorted(opts)
        if not opts:
            return None
        st.duds.add(opts[int(rng.integers(len(opts)))])

    # At least three marker cells overall so the terminal color is the only 2-cell color.
    tries = 0
    while len(st.live) + len(st.duds) < 3 and tries < 50:
        tries += 1
        p = edge_cells[int(rng.integers(len(edge_cells)))]
        if st.can_live(p, protect=pset):
            st.add_live(p)
    if len(st.live) + len(st.duds) < 3:
        return None
    return st


def _features(inp, out):
    info = _analyze(inp)
    h, w = inp.shape
    live = info["live"]
    feats = {"interior": False, "clip": False, "overlap": False, "spill": False}
    cells = [tuple(int(v) for v in p) for p in zip(*np.nonzero(live))]
    for y, x in cells:
        r0, r1, c0, c1 = next(r for r in info["rects"] if r[0] <= y < r[1] and r[2] <= x < r[3])
        if r0 < y < r1 - 1 and c0 < x < c1 - 1:
            feats["interior"] = True
        if y in (0, h - 1) or x in (0, w - 1):
            feats["clip"] = True
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 2:
                feats["overlap"] = True
    bg = info["roles"]["background"]
    feats["spill"] = bool(np.any(info["halo"] & (inp == bg)))
    feats["route_len"] = len(info["path"]) - 2
    feats["margin"] = bool(np.all(np.concatenate([inp[0], inp[-1], inp[:, 0], inp[:, -1]]) != info["roles"]["carrier"]))
    return feats


def _check_counts(num_train, num_test):
    if num_train is None:
        num_train = 4
    if num_test is None:
        num_test = 1
    if isinstance(num_train, bool) or isinstance(num_test, bool):
        raise ValueError("counts must be integers")
    num_train, num_test = int(num_train), int(num_test)
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("need 2..10 train, 1..4 test, train > test")
    return num_train, num_test


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    margin_p = float(rng.choice([0.0, 0.35, 0.65, 1.0]))
    vary_palette = rng.random() < 0.3
    interior_terms_p = float(rng.choice([0.0, 0.0, 0.25]))
    maxb_lo = int(rng.integers(4, 6))
    maxb_hi = maxb_lo + int(rng.integers(0, 3))
    tie_live = float(rng.choice([0.25, 0.5, 0.75]))

    def palette():
        cols = [int(c) for c in rng.permutation(INPUT_COLORS)[:4]]
        if rng.random() < 0.5 and 0 not in cols[1:]:
            cols[0] = 0
        bg, carrier, marker, terminal = cols
        return {"background": bg, "carrier": carrier, "marker": marker, "terminal": terminal}

    base = palette()
    pairs, seen = [], set()
    plans = []
    for i in range(num_train):
        plans.append(("train", i))
    for i in range(num_test):
        plans.append(("test", i))
    clip_test = False
    for split, i in plans:
        test = split == "test"
        for attempt in range(400):
            margin = bool(rng.random() < margin_p)
            want_clip = False
            if split == "train" and i == 1 and margin_p < 1.0 and attempt < 200:
                # the second training pair teaches canvas clipping whenever the
                # puzzle allows blocks to touch the canvas edge
                margin = False
                want_clip = True
            cfg = {
                "hmin": 13 if test else 8, "hmax": 22 if test else 15,
                "wmin": 13 if test else 8, "wmax": 22 if test else 16,
                "margin": margin,
                "maxb": int(rng.integers(maxb_lo, maxb_hi + 1)),
                "interior_terms": bool(rng.random() < interior_terms_p),
                "want_interior": (split == "train" and i == 0) or rng.random() < 0.4,
                "want_clip": want_clip or (not margin and rng.random() < 0.3),
                "extra_live": 3 if test else 2,
                "extra_duds": 2,
                "rb_min": 2 if test else 1, "rb_max": 4 if test else 2,
                "tie_live": tie_live,
                "max_ties": 10,
            }
            st = _try_pair(rng, cfg)
            if st is None:
                continue
            roles = palette() if vary_palette else base
            inp = st.grid(roles)
            try:
                info = _analyze(inp)
                if info["roles"] != roles or info["count"] != 1:
                    continue
                _pair_checks(inp, _render(inp, info))
            except (ValueError, AssertionError):
                continue
            key = inp.tobytes() + bytes(inp.shape)
            if key in seen:
                continue
            out = _render(inp, info)
            feats = _features(inp, out)
            if test and feats["clip"] and not any(f["clip"] for f in [p[2] for p in pairs]):
                # tests may only clip if some training pair already showed clipping
                continue
            seen.add(key)
            pairs.append((inp, out, feats))
            break
        else:
            raise RuntimeError(f"could not build pair {split}{i} for seed {seed}")
    train = [{"input": p[0].tolist(), "output": p[1].tolist()} for p in pairs[:num_train]]
    test = [{"input": p[0].tolist(), "output": p[1].tolist()} for p in pairs[num_train:]]
    return {"train": train, "test": test}


# --------------------------------------------------------------------------
# Validation and self test
# --------------------------------------------------------------------------

def _pair_checks(inp, out):
    inp = np.asarray(inp, dtype=int)
    out = np.asarray(out, dtype=int)
    assert inp.ndim == 2 and inp.shape == out.shape, "shape mismatch"
    assert 1 <= inp.shape[0] <= 30 and 1 <= inp.shape[1] <= 30, "bad size"
    assert not np.any((inp == HALO) | (inp == ROUTE)), "output-only colors in input"
    info = _analyze(inp)
    roles = info["roles"]
    assert info["count"] == 1, "shortest route not unique"
    assert np.array_equal(_render(inp, info), out), "output disagrees with rule"
    # carrier rectangles: connected, bbox exact, each side keeps carrier evidence
    for r0, r1, c0, c1 in info["rects"]:
        sub = inp[r0:r1, c0:c1] == roles["carrier"]
        for side in (sub[0, :], sub[-1, :], sub[:, 0], sub[:, -1]):
            assert side.sum() >= max(1, (len(side) + 1) // 2), "rectangle side too eroded"
    # rectangles separated by background streets (no 8-touching)
    occ = -np.ones(inp.shape, dtype=int)
    for k, (r0, r1, c0, c1) in enumerate(info["rects"]):
        assert np.all(occ[max(0, r0 - 1):r1 + 1, max(0, c0 - 1):c1 + 1] < 0), "touching rectangles"
        occ[r0:r1, c0:c1] = k
    # duds and terminals never inside any halo (no crossing ambiguity)
    live = info["live"]
    full = np.zeros_like(live)
    for y, x in zip(*np.nonzero(live)):
        full[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
    assert not np.any(full & info["duds"]), "dud inside a halo"
    assert not np.any(full & (inp == roles["terminal"])), "terminal inside a halo"
    assert live.sum() >= 1, "no live marker"
    # halo matters: ignoring halos gives a strictly shorter route
    ign = _analyze(inp, "ignore_halo")
    assert ign["distance"] < info["distance"], "halo does not affect the route"
    # duds matter: treating duds as live would invalidate the route
    allv = _analyze(inp, "all_live")
    assert allv["count"] == 0 or allv["path"] != info["path"] or allv["distance"] != info["distance"], \
        "dud-as-live shortcut gives the same answer"
    inner = set(info["path"][1:-1])
    assert inner and all(inp[p] == roles["background"] for p in inner), "route leaves the street"
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad keys"
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "bad counts"
    keys = set()
    feats = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert set(pair) == {"input", "output"}, "bad pair keys"
            inp = np.asarray(pair["input"], dtype=int)
            out = np.asarray(pair["output"], dtype=int)
            k = inp.tobytes() + bytes(inp.shape)
            assert k not in keys, "duplicate input"
            keys.add(k)
            _pair_checks(inp, out)
            feats[split].append(_features(inp, out))
    assert any(f["interior"] for f in feats["train"]), "training never shows an interior halo"
    assert any(f["spill"] for f in feats["train"]), "training never shows halo spill"
    if any(f["clip"] for f in feats["test"]):
        assert any(f["clip"] for f in feats["train"]), "test clips but training never does"


def self_test() -> None:
    seeds = list(range(42))
    shapes, margins, palettes, lens = set(), set(), set(), []
    clip_train = 0
    for seed in seeds:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        q = generate_puzzle(seed)
        assert all(np.array_equal(np.asarray(a["input"]), np.asarray(b["input"]))
                   and np.array_equal(np.asarray(a["output"]), np.asarray(b["output"]))
                   for s in ("train", "test") for a, b in zip(p[s], q[s])), "nondeterministic"
        for s in ("train", "test"):
            for pair in p[s]:
                inp = np.asarray(pair["input"])
                shapes.add(inp.shape)
                f = _features(inp, np.asarray(pair["output"]))
                margins.add(f["margin"])
                lens.append(f["route_len"])
                palettes.add(tuple(sorted(np.unique(inp).tolist())))
                if s == "train" and f["clip"]:
                    clip_train += 1
    assert len(shapes) >= 30, "too few grid shapes"
    assert margins == {True, False}, "margin variation missing"
    assert len(palettes) >= 10, "too few palettes"
    assert len(set(lens)) >= 10, "route lengths do not vary"
    assert clip_train >= 10, "clipping rarely taught"
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (4, 5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted {a}/{b}")
    p = generate_puzzle(7, 10, 4)
    validate_generated_puzzle(p)
    assert len(p["train"]) == 10 and len(p["test"]) == 4


if __name__ == "__main__":
    self_test()
    print("self_test passed")
