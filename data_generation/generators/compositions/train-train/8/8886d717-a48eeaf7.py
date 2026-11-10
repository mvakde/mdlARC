"""Medium-matched attraction: markers join the 2x2 anchor that lives in their hidden substrate.

Composition of
  A (8886d717): a two-substrate field with sparse marker cells; the substrate hidden
     under each marker is inferred from its surroundings, and that latent substrate
     decides the marker's fate (active -> acts, inactive -> erased / restored);
  B (a48eeaf7): solid 2x2 reference blocks; a marker aligned with a block along a
     horizontal, vertical or corner-diagonal ray slides inward to the cell just
     outside the block, its start cell being cleared.

New rule: every 2x2 anchor block sits inside one substrate (its "medium"; the ring of
cells around it is that substrate).  Each marker's hidden substrate is the unanimous
substrate colour of its non-marker orthogonal neighbours.  A marker whose hidden
substrate hosts an anchor slides along its ray into that anchor (just outside it);
a marker whose hidden substrate hosts no anchor dissolves.  Every vacated cell is
restored to its hidden substrate colour.  Blocks and substrates are unchanged.

Standard library + NumPy only.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

# ----------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _rays(r, c):
    """The 12 approach rays of a 2x2 block with top-left (r, c): (direction, destination)."""
    return [
        ((-1, 0), (r - 1, c)), ((-1, 0), (r - 1, c + 1)),
        ((1, 0), (r + 2, c)), ((1, 0), (r + 2, c + 1)),
        ((0, -1), (r, c - 1)), ((0, -1), (r + 1, c - 1)),
        ((0, 1), (r, c + 2)), ((0, 1), (r + 1, c + 2)),
        ((-1, -1), (r - 1, c - 1)), ((-1, 1), (r - 1, c + 2)),
        ((1, -1), (r + 2, c - 1)), ((1, 1), (r + 2, c + 2)),
    ]


def _ray_of(block, cell):
    """Return (ray_index, distance k>=1, destination) if cell lies on a ray of block."""
    r, c = block
    y, x = cell
    for i, ((dr, dc), (ty, tx)) in enumerate(_rays(r, c)):
        oy, ox = y - ty, x - tx
        if dr == 0:
            if oy != 0 or ox == 0 or (ox > 0) != (dc > 0):
                continue
            k = abs(ox)
        elif dc == 0:
            if ox != 0 or oy == 0 or (oy > 0) != (dr > 0):
                continue
            k = abs(oy)
        else:
            if oy == 0 or oy * dr <= 0 or ox * dc <= 0 or abs(oy) != abs(ox):
                continue
            k = abs(oy)
        return i, k, (ty, tx), (dr, dc)
    return None


def _in_ring(block, cell, margin=1):
    r, c = block
    y, x = cell
    return r - margin <= y <= r + 1 + margin and c - margin <= x <= c + 1 + margin


def _components(mask):
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for y in range(H):
        for x in range(W):
            if mask[y, x] and not seen[y, x]:
                stack, comp = [(y, x)], []
                seen[y, x] = True
                while stack:
                    a, b = stack.pop()
                    comp.append((a, b))
                    for dy, dx in N4:
                        p, q = a + dy, b + dx
                        if 0 <= p < H and 0 <= q < W and mask[p, q] and not seen[p, q]:
                            seen[p, q] = True
                            stack.append((p, q))
                comps.append(comp)
    return comps


# ----------------------------------------------------------------------------------
# Input analysis (shared by solve and the validator; input only)
# ----------------------------------------------------------------------------------

def analyze(grid):
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    counts = Counter(g.ravel().tolist())
    order = sorted(counts, key=lambda col: (-counts[col], col))
    substrates = order[:2]
    others = order[2:]
    block_colors, blocks = [], []
    for col in others:
        comps = _components(g == col)
        ok = bool(comps)
        tl = []
        for comp in comps:
            ys = [p[0] for p in comp]
            xs = [p[1] for p in comp]
            if len(comp) != 4 or max(ys) - min(ys) != 1 or max(xs) - min(xs) != 1:
                ok = False
                break
            tl.append((min(ys), min(xs)))
        if ok:
            block_colors.append(col)
            blocks.extend((y, x, col) for y, x in tl)
    marker_colors = [col for col in others if col not in block_colors]
    sub = set(substrates)
    media = []
    for (r, c, _) in blocks:
        ring = [int(g[y, x]) for y in range(r - 1, r + 3) for x in range(c - 1, c + 3)
                if 0 <= y < H and 0 <= x < W and not (r <= y <= r + 1 and c <= x <= c + 1)
                and int(g[y, x]) in sub]
        media.append(Counter(ring).most_common(1)[0][0] if ring else None)
    markers = []
    mset = set(marker_colors)
    for y in range(H):
        for x in range(W):
            if int(g[y, x]) in mset:
                nb = [int(g[y + dy, x + dx]) for dy, dx in N4
                      if 0 <= y + dy < H and 0 <= x + dx < W and int(g[y + dy, x + dx]) in sub]
                if not nb:
                    nb = [int(g[p, q]) for p in range(y - 1, y + 2) for q in range(x - 1, x + 2)
                          if 0 <= p < H and 0 <= q < W and int(g[p, q]) in sub]
                hidden = Counter(nb).most_common(1)[0][0] if nb else substrates[0]
                markers.append({"cell": (y, x), "color": int(g[y, x]), "hidden": hidden,
                                "neighbors": nb})
    return {"grid": g, "substrates": substrates, "blocks": blocks, "media": media,
            "markers": markers, "marker_colors": marker_colors, "block_colors": block_colors}


def _clear_path(info, dest, direction, k):
    g = info["grid"]
    sub = set(info["substrates"])
    for j in range(k):
        y, x = dest[0] + j * direction[0], dest[1] + j * direction[1]
        if int(g[y, x]) not in sub:
            return False
    return True


def _aligned(info, block_index, cell):
    r, c, _ = info["blocks"][block_index]
    hit = _ray_of((r, c), cell)
    if hit is None:
        return None
    i, k, dest, d = hit
    if not _clear_path(info, dest, d, k):
        return None
    return {"ray": i, "k": k, "dest": dest}


def _render(info, moves):
    """moves: list of (marker, destination or None)."""
    out = info["grid"].copy()
    for m, _ in moves:
        out[m["cell"]] = m["hidden"]
    for m, dest in moves:
        if dest is not None:
            out[dest] = m["color"]
    return out


def _plan(info, active_of=None):
    """Rule plan.  active_of(marker) -> list of eligible block indices (default: medium match)."""
    moves = []
    for m in info["markers"]:
        if active_of is None:
            cands = [i for i, med in enumerate(info["media"]) if med == m["hidden"]]
        else:
            cands = active_of(m)
        dest = None
        best = None
        for i in cands:
            a = _aligned(info, i, m["cell"])
            if a is not None and (best is None or a["k"] < best):
                best, dest = a["k"], a["dest"]
        moves.append((m, dest))
    return moves


def solve(input_grid):
    info = analyze(input_grid)
    return _render(info, _plan(info))


# ----------------------------------------------------------------------------------
# Shortcut solvers (used to prove the demonstrations distinguish alternatives)
# ----------------------------------------------------------------------------------

def _shortcut_ray_only(grid):
    """Ignore the hidden substrate: any aligned marker joins its nearest block."""
    info = analyze(grid)
    return _render(info, _plan(info, lambda m: list(range(len(info["blocks"])))))


def _shortcut_fixed_color(grid, color):
    """Parent-A habit: one literal substrate colour is 'active' everywhere."""
    info = analyze(grid)
    return _render(info, _plan(info, lambda m: list(range(len(info["blocks"])))
                               if m["hidden"] == color else []))


def _shortcut_size_role(grid, larger):
    """Active substrate = larger (or smaller) substrate by visible count."""
    info = analyze(grid)
    g = info["grid"]
    s0, s1 = info["substrates"]
    n0, n1 = int((g == s0).sum()), int((g == s1).sum())
    act = (s0 if n0 >= n1 else s1) if larger else (s1 if n0 >= n1 else s0)
    return _render(info, _plan(info, lambda m: list(range(len(info["blocks"])))
                               if m["hidden"] == act else []))


def _shortcut_no_slide(grid):
    """Parent-A-only reading: anchored-medium markers stay put, others dissolve."""
    info = analyze(grid)
    out = info["grid"].copy()
    for m in info["markers"]:
        if m["hidden"] not in info["media"]:
            out[m["cell"]] = m["hidden"]
    return out


def _shortcut_no_substrate(grid):
    """Parent-B-only reading: every marker slides if aligned with any block, else stays."""
    info = analyze(grid)
    out = info["grid"].copy()
    for m, dest in _plan(info, lambda m: list(range(len(info["blocks"])))):
        if dest is not None:
            out[m["cell"]] = m["hidden"]
    for m, dest in _plan(info, lambda m: list(range(len(info["blocks"])))):
        if dest is not None:
            out[dest] = m["color"]
    return out


# ----------------------------------------------------------------------------------
# Substrate fields
# ----------------------------------------------------------------------------------

FIELD_TYPES = ("split", "pockets", "voronoi", "bands", "blobs", "frame")


def _smooth(f, rng, passes=2):
    H, W = f.shape
    for _ in range(passes):
        p = np.pad(f, 1, mode="edge")
        s = sum(p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        f = (s >= 5).astype(int)
    return f


def _field(rng, kind, H, W):
    f = np.zeros((H, W), dtype=int)
    if kind == "split":
        vertical = rng.random() < 0.5
        n, m = (W, H) if vertical else (H, W)
        pos = int(rng.integers(m // 3, m - m // 3 + 1))
        prof = []
        for _ in range(n):
            pos = int(np.clip(pos + rng.choice([-2, -1, 0, 0, 0, 1, 2]), 3, m - 3))
            prof.append(pos)
        for i, p in enumerate(prof):
            if vertical:
                f[p:, i] = 1
            else:
                f[i, p:] = 1
        if rng.random() < 0.5:
            f = 1 - f
    elif kind == "pockets":
        for _ in range(int(rng.integers(1, 4))):
            h = int(rng.integers(4, max(5, H * 2 // 3)))
            w = int(rng.integers(4, max(5, W * 2 // 3)))
            y = int(rng.integers(-1, H - h + 2))
            x = int(rng.integers(-1, W - w + 2))
            f[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = 1
    elif kind == "voronoi":
        k = int(rng.integers(3, 7))
        pts = np.column_stack([rng.integers(0, H, k), rng.integers(0, W, k)])
        lab = rng.permutation(np.arange(k) % 2)
        yy, xx = np.mgrid[0:H, 0:W]
        metric = rng.integers(0, 3)
        d = []
        for (py, px) in pts:
            if metric == 0:
                d.append(np.abs(yy - py) + np.abs(xx - px))
            elif metric == 1:
                d.append(np.maximum(np.abs(yy - py), np.abs(xx - px)))
            else:
                d.append((yy - py) ** 2 + (xx - px) ** 2)
        f = lab[np.argmin(np.stack(d), axis=0)]
    elif kind == "bands":
        mode = int(rng.integers(0, 3))
        yy, xx = np.mgrid[0:H, 0:W]
        coord = yy if mode == 0 else xx if mode == 1 else (yy + xx if rng.random() < 0.5 else yy - xx + W)
        lo, hi = int(coord.min()), int(coord.max())
        cuts, pos = [], lo + int(rng.integers(3, 7))
        while pos < hi - 2:
            cuts.append(pos)
            pos += int(rng.integers(4, 9))
        f = (np.searchsorted(np.array(cuts + [10 ** 6]), coord, side="right") % 2).astype(int)
    elif kind == "blobs":
        for _ in range(int(rng.integers(1, 4))):
            target = int(rng.integers(H * W // 10, H * W // 4 + 2))
            y, x = int(rng.integers(0, H)), int(rng.integers(0, W))
            cells = {(y, x)}
            frontier = [(y, x)]
            while len(cells) < target and frontier:
                a, b = frontier[int(rng.integers(0, len(frontier)))]
                dy, dx = N4[int(rng.integers(0, 4))]
                p, q = a + dy, b + dx
                if 0 <= p < H and 0 <= q < W and (p, q) not in cells:
                    cells.add((p, q))
                    frontier.append((p, q))
            for p, q in cells:
                f[p, q] = 1
    elif kind == "frame":
        t = int(rng.integers(3, 6))
        f[:, :] = 1
        f[t:H - t, t:W - t] = 0
        if rng.random() < 0.5:
            # open the frame on one side
            side = int(rng.integers(0, 4))
            if side == 0:
                f[:t, t:W - t] = 0
            elif side == 1:
                f[H - t:, t:W - t] = 0
            elif side == 2:
                f[t:H - t, :t] = 0
            else:
                f[t:H - t, W - t:] = 0
    f = _smooth(f, rng, passes=1 if kind in ("split", "pockets", "bands", "frame") else 2)
    if rng.random() < 0.5:
        f = 1 - f
    return f


# ----------------------------------------------------------------------------------
# Pair construction
# ----------------------------------------------------------------------------------

def _interior(f, y, x):
    H, W = f.shape
    v = f[y, x]
    for dy, dx in N4:
        p, q = y + dy, x + dx
        if 0 <= p < H and 0 <= q < W and f[p, q] != v:
            return False
    return True


def _block_sites(f, s, edge):
    H, W = f.shape
    sites = []
    for r in range(0, H - 1):
        for c in range(0, W - 1):
            win = f[max(0, r - 1):min(H, r + 3), max(0, c - 1):min(W, c + 3)]
            if not np.all(win == s):
                continue
            touches = r == 0 or c == 0 or r == H - 2 or c == W - 2
            if edge is None or edge == touches:
                sites.append((r, c))
    return sites


def _try_pair(rng, cfg):
    H = int(rng.integers(cfg["hmin"], cfg["hmax"] + 1))
    W = int(rng.integers(cfg["wmin"], cfg["wmax"] + 1))
    kind = cfg["field"]
    f = _field(rng, kind, H, W)
    n1 = int(f.sum())
    if not (0.25 * H * W <= n1 <= 0.75 * H * W):
        return None
    double = cfg["double"]
    hosts = [0, 1] if double else [cfg["host"]]
    blocks = []
    for idx, s in enumerate(hosts):
        edge = cfg["edge"] if idx == 0 else None
        sites = _block_sites(f, s, edge)
        sites = [b for b in sites if all(abs(b[0] - o[0]) >= 4 or abs(b[1] - o[1]) >= 4 for o in blocks)]
        if not sites:
            return None
        if edge is None and rng.random() < 0.7:
            inner = [b for b in sites if 0 < b[0] < H - 2 and 0 < b[1] < W - 2]
            sites = inner or sites
        blocks.append(sites[int(rng.integers(0, len(sites)))])
    block_cells = {(r + a, c + b) for (r, c) in blocks for a in (0, 1) for b in (0, 1)}

    def free(cell):
        return all(not _in_ring(b, cell) for b in blocks)

    def path_ok(dest, d, k):
        for j in range(k):
            p = (dest[0] + j * d[0], dest[1] + j * d[1])
            if p in block_cells:
                return False
        return True

    occupied = {}  # cell -> role
    used_rays = set()
    movers, decoys = [], []

    def ray_candidates(bi, want_s):
        r, c = blocks[bi]
        out = []
        for ri, (d, dest) in enumerate(_rays(r, c)):
            if (bi, ri) in used_rays:
                continue
            k = 1
            while True:
                y, x = dest[0] + k * d[0], dest[1] + k * d[1]
                if not (0 <= y < H and 0 <= x < W):
                    break
                if (f[y, x] == want_s and _interior(f, y, x) and free((y, x))
                        and path_ok(dest, d, k)):
                    out.append((ri, k, (y, x), dest, d))
                k += 1
        return out

    # movers
    for bi, s in enumerate(hosts):
        want = int(rng.integers(cfg["movers"][0], cfg["movers"][1] + 1))
        cands = ray_candidates(bi, s)
        # cross-aligned movers in double pairs: also aligned with the other block, nearer to it
        if double and cfg["cross"] and bi == 0:
            other = blocks[1]
            cross = []
            for cand in cands:
                hit = _ray_of(other, cand[2])
                if hit is not None and path_ok(hit[2], hit[3], hit[1]):
                    cross.append((0 if hit[1] < cand[1] else 1, cand))
            if not cross:
                return None
            cross.sort(key=lambda t: t[0])
            best = [cc for p, cc in cross if p == cross[0][0]]
            pick = best[int(rng.integers(0, len(best)))]
            movers.append((bi, pick))
            used_rays.add((bi, pick[0]))
            occupied[pick[2]] = "m"
            want -= 1
        order = rng.permutation(len(cands))
        for j in order:
            if want <= 0:
                break
            ri, k, cell, dest, d = cands[j]
            if (bi, ri) in used_rays or cell in occupied:
                continue
            if k > cfg["maxk"]:
                continue
            movers.append((bi, cands[j]))
            used_rays.add((bi, ri))
            occupied[cell] = "m"
            want -= 1
        if want > 0 and not any(b == bi for b, _ in movers):
            return None
    if not movers:
        return None

    # dissolvers (single pairs only): aligned decoys first, then free ones
    if not double:
        other_s = 1 - hosts[0]
        n_al = int(rng.integers(cfg["aligned"][0], cfg["aligned"][1] + 1))
        cands = ray_candidates(0, other_s)
        for j in rng.permutation(len(cands)):
            if n_al <= 0:
                break
            ri, k, cell, dest, d = cands[j]
            if (0, ri) in used_rays or cell in occupied:
                continue
            decoys.append(cell)
            used_rays.add((0, ri))
            occupied[cell] = "d"
            n_al -= 1
        if n_al > 0 and cfg["aligned"][0] > 0:
            return None
        n_free = int(rng.integers(cfg["free"][0], cfg["free"][1] + 1))
        pool = [(y, x) for y in range(H) for x in range(W)
                if f[y, x] == other_s and _interior(f, y, x) and free((y, x))
                and (y, x) not in occupied]
        for j in rng.permutation(len(pool))[:n_free * 3]:
            if n_free <= 0:
                break
            cell = pool[j]
            hit = _ray_of(blocks[0], cell)
            if hit is not None and (0, hit[0]) in used_rays:
                continue
            if hit is not None:
                used_rays.add((0, hit[0]))
            decoys.append(cell)
            occupied[cell] = "d"
            n_free -= 1
        if not decoys and cfg["need_dissolve"]:
            return None

    marker_cells = list(occupied)
    mset = set(marker_cells)
    # every mover path must be free of other markers
    for bi, (ri, k, cell, dest, d) in movers:
        for j in range(k):
            if (dest[0] + j * d[0], dest[1] + j * d[1]) in mset:
                return None
    # inference: each marker keeps >= 2 visible, unanimous substrate neighbours
    for (y, x) in marker_cells:
        vis = [(y + dy, x + dx) for dy, dx in N4 if 0 <= y + dy < H and 0 <= x + dx < W
               and (y + dy, x + dx) not in mset]
        if len(vis) < 2:
            return None
    # no 2x2 marker cluster
    for (y, x) in marker_cells:
        if {(y, x + 1), (y + 1, x), (y + 1, x + 1)} <= mset:
            return None
    pal = cfg["palette"]
    s_col = pal["sub"]
    g = np.where(f == 1, s_col[1], s_col[0]).astype(int)
    for (r, c) in blocks:
        g[r:r + 2, c:c + 2] = pal["block"]
    mcols = pal["markers"]
    colors = {}
    for i, cell in enumerate(marker_cells):
        colors[cell] = int(mcols[0] if len(mcols) == 1 else mcols[int(rng.integers(0, len(mcols)))])
    if len(mcols) > 1 and len(set(colors.values())) < 2 and len(marker_cells) >= 2:
        colors[marker_cells[-1]] = int(mcols[1] if colors[marker_cells[-1]] == mcols[0] else mcols[0])
    for cell, col in colors.items():
        g[cell] = col
    # colour-role margins so the solver can read the roles from counts
    cnt = Counter(g.ravel().tolist())
    small = max(cnt[col] for col in cnt if col not in s_col)
    if min(cnt[s_col[0]], cnt[s_col[1]]) < 2 * small + 4:
        return None
    # expected output, built from latent truth
    out = np.where(f == 1, s_col[1], s_col[0]).astype(int)
    for (r, c) in blocks:
        out[r:r + 2, c:c + 2] = pal["block"]
    for bi, (ri, k, cell, dest, d) in movers:
        out[dest] = colors[cell]
    meta = {"double": double, "blocks": blocks, "hosts": [int(s_col[h]) for h in hosts],
            "movers": len(movers), "decoys": len(decoys), "field": kind,
            "edge": any(r == 0 or c == 0 or r == H - 2 or c == W - 2 for r, c in blocks),
            "maxk": max(m[1][1] for m in movers)}
    return g, out, meta


def _make_pair(rng, cfg, tries=400):
    for _ in range(tries):
        res = _try_pair(rng, cfg)
        if res is None:
            continue
        g, out, meta = res
        if not np.array_equal(solve(g), out):
            continue
        try:
            _check_pair(g, out)
        except AssertionError:
            continue
        return g, out, meta
    raise RuntimeError("could not build pair")


# ----------------------------------------------------------------------------------
# Puzzle construction
# ----------------------------------------------------------------------------------

def _palette(rng, n_markers):
    cols = [int(v) for v in rng.permutation(10)[:3 + n_markers]]
    return {"sub": cols[:2], "block": cols[2], "markers": cols[3:]}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = 4 if num_train is None else num_train
    num_test = 1 if num_test is None else num_test
    if not isinstance(num_train, (int, np.integer)) or not isinstance(num_test, (int, np.integer)):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("illegal pair counts")
    rng = np.random.default_rng(seed)
    n_mark_cols = 2 if rng.random() < 0.2 else 1
    palette_mode = "fixed" if rng.random() < 0.7 else "per_pair"
    base_pal = _palette(rng, n_mark_cols)
    field_pool = list(rng.permutation(FIELD_TYPES)[:int(rng.integers(2, len(FIELD_TYPES) + 1))])
    use_double = rng.random() < 0.65
    shift = str(rng.choice(["bigger", "double", "edge", "dense", "sparse", "same"]))
    if shift == "double" and not use_double:
        shift = "bigger"
    lo = int(rng.integers(8, 13))
    hi = lo + int(rng.integers(2, 7))

    kinds = ["single"] * num_train
    if use_double:
        nd = 1 if num_train <= 3 else int(rng.integers(1, num_train // 2 + 1))
        for i in rng.permutation(np.arange(1, num_train))[:nd]:
            kinds[int(i)] = "double"
    # hosts for singles alternate so neither literal colour nor size role can be fixed
    singles = [i for i, k in enumerate(kinds) if k == "single"]
    first_host = int(rng.integers(0, 2))
    hosts = {}
    for j, i in enumerate(singles):
        hosts[i] = first_host if (j % 2 == 0) else 1 - first_host
    if len(singles) >= 3:
        for i in singles[2:]:
            hosts[i] = int(rng.integers(0, 2))

    def cfg_for(kind, host, test=False, slot=0):
        pal = base_pal if palette_mode == "fixed" else _palette(rng, n_mark_cols)
        h0, h1 = lo, hi
        mv = (1, 4)
        al = (1, 2)
        fr = (0, 3)
        edge = None if rng.random() < 0.7 else bool(rng.random() < 0.5)
        if test:
            if shift == "bigger":
                h0, h1 = hi + 1, min(26, hi + 8)
            elif shift == "dense":
                mv, al, fr = (4, 7), (1, 3), (2, 5)
                h0, h1 = max(lo, 11), max(hi, 14)
            elif shift == "sparse":
                mv, al, fr = (1, 2), (1, 1), (0, 1)
            elif shift == "edge":
                edge = True
        if kind == "double":
            h0, h1 = max(h0, 11), max(h1, 14)
        wj = int(rng.integers(-2, 3))
        return {"hmin": h0, "hmax": h1, "wmin": max(8, h0 + wj), "wmax": min(30, max(8, h1 + wj)),
                "field": str(field_pool[int(rng.integers(0, len(field_pool)))]),
                "double": kind == "double", "host": host, "edge": edge,
                "movers": mv, "aligned": al if kind == "single" else (0, 0),
                "free": fr, "cross": kind == "double", "need_dissolve": kind == "single",
                "maxk": 30, "palette": pal}

    for _attempt in range(20):
        train, test, metas, seen = [], [], [], set()
        ok = True
        for i in range(num_train):
            cfg = cfg_for(kinds[i], hosts.get(i, 0), slot=i)
            if i == 0 or kinds[i] == "single":
                cfg["aligned"] = (1, 2)
            try:
                g, out, meta = _make_pair(rng, cfg)
            except RuntimeError:
                ok = False
                break
            key = g.tobytes() + bytes(g.shape)
            if key in seen:
                ok = False
                break
            seen.add(key)
            train.append({"input": g, "output": out})
            metas.append(meta)
        if not ok:
            continue
        for t in range(num_test):
            if shift == "double" and t == 0:
                kind = "double"
            elif use_double and rng.random() < 0.35:
                kind = "double"
            else:
                kind = "single"
            cfg = cfg_for(kind, int(rng.integers(0, 2)), test=True)
            try:
                g, out, meta = _make_pair(rng, cfg)
            except RuntimeError:
                ok = False
                break
            key = g.tobytes() + bytes(g.shape)
            if key in seen:
                ok = False
                break
            seen.add(key)
            test.append({"input": g, "output": out})
        if not ok:
            continue
        puzzle = {"train": train, "test": test}
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle
    raise RuntimeError(f"could not generate puzzle for seed {seed}")


# ----------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------

def _check_pair(inp, out):
    g = np.asarray(inp, dtype=int)
    o = np.asarray(out, dtype=int)
    assert g.shape == o.shape and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
    info = analyze(g)
    assert len(info["substrates"]) == 2 and len(info["block_colors"]) == 1, "role colours unreadable"
    assert info["marker_colors"], "no marker colour"
    nb = len(info["blocks"])
    assert nb in (1, 2), "one or two anchors per pair"
    sub = info["substrates"]
    cnt = Counter(g.ravel().tolist())
    small = max(cnt[c] for c in cnt if c not in sub)
    assert min(cnt[sub[0]], cnt[sub[1]]) >= 2 * small + 4, "substrate count margin"
    # anchors: ring unanimous, distinct media
    H, W = g.shape
    for (r, c, _), med in zip(info["blocks"], info["media"]):
        ring = {int(g[y, x]) for y in range(r - 1, r + 3) for x in range(c - 1, c + 3)
                if 0 <= y < H and 0 <= x < W and not (r <= y <= r + 1 and c <= x <= c + 1)}
        assert ring == {med}, "anchor ring must be a single substrate"
    assert len(set(info["media"])) == nb, "at most one anchor per medium"
    mcells = {m["cell"] for m in info["markers"]}
    plan = _plan(info)
    dests = []
    for m, dest in plan:
        assert len(m["neighbors"]) >= 2 and len(set(m["neighbors"])) == 1, "hidden substrate ambiguous"
        for (r, c, _) in info["blocks"]:
            assert not _in_ring((r, c), m["cell"]), "marker touching an anchor"
        anchored = m["hidden"] in info["media"]
        assert anchored == (dest is not None), "anchored-medium marker must be aligned with a clear ray"
        if dest is not None:
            dests.append(dest)
    assert len(dests) == len(set(dests)), "destinations must be distinct"
    assert not (set(dests) & mcells), "destination occupied"
    assert dests, "at least one marker slides"
    for (y, x) in mcells:
        assert not {(y, x + 1), (y + 1, x), (y + 1, x + 1)} <= mcells, "2x2 marker cluster"
    # one marker per ray
    rays = set()
    for m in info["markers"]:
        for bi, (r, c, _) in enumerate(info["blocks"]):
            hit = _ray_of((r, c), m["cell"])
            if hit is not None:
                assert (bi, hit[0]) not in rays, "two markers on one ray"
                rays.add((bi, hit[0]))
    assert np.array_equal(solve(g), o), "output disagrees with rule"
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    tr, te = puzzle["train"], puzzle["test"]
    assert 2 <= len(tr) <= 10 and 1 <= len(te) <= 4 and len(tr) > len(te)
    seen = set()
    infos = []
    for split in (tr, te):
        for p in split:
            assert set(p) == {"input", "output"}
            g = np.asarray(p["input"], dtype=int)
            key = (g.shape, g.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            infos.append(_check_pair(p["input"], p["output"]))
    tinfo = infos[:len(tr)]
    # the training set must show a dissolve, an anchored slide, and medium decisions
    dissolves = sum(1 for inf in tinfo for m, d in _plan(inf) if d is None)
    assert dissolves >= 1, "training never shows a dissolving marker"
    has_double_train = any(len(inf["blocks"]) == 2 for inf in tinfo)
    has_double_test = any(len(inf["blocks"]) == 2 for inf in infos[len(tr):])
    if has_double_test:
        assert has_double_train, "two-anchor test needs a two-anchor demonstration"

    def beaten(fn):
        return any(not np.array_equal(fn(p["input"]), np.asarray(p["output"])) for p in tr)

    assert beaten(_shortcut_ray_only), "ray-only shortcut not excluded by training"
    assert beaten(_shortcut_no_slide), "no-slide shortcut not excluded"
    assert beaten(_shortcut_no_substrate), "substrate-blind shortcut not excluded"
    assert beaten(lambda g: _shortcut_size_role(g, True)), "larger-substrate shortcut not excluded"
    assert beaten(lambda g: _shortcut_size_role(g, False)), "smaller-substrate shortcut not excluded"
    colors = {c for inf in tinfo for c in inf["substrates"]}
    for col in colors:
        assert beaten(lambda g, col=col: _shortcut_fixed_color(g, col)), "fixed-colour shortcut not excluded"
    return True


# ----------------------------------------------------------------------------------
# Self test
# ----------------------------------------------------------------------------------

def self_test():
    stats = Counter()
    shapes = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        for split in ("train", "test"):
            for pair in p[split]:
                g = np.asarray(pair["input"])
                assert np.array_equal(solve(g), np.asarray(pair["output"]))
                info = analyze(g)
                shapes.add(g.shape)
                stats["double" if len(info["blocks"]) == 2 else "single"] += 1
                stats["multi_marker_colours" if len(info["marker_colors"]) > 1 else "one_marker_colour"] += 1
                for (r, c, _) in info["blocks"]:
                    if r == 0 or c == 0 or r == g.shape[0] - 2 or c == g.shape[1] - 2:
                        stats["edge_block"] += 1
    assert stats["double"] >= 20 and stats["single"] >= 60, stats
    assert stats["edge_block"] >= 5, stats
    assert stats["multi_marker_colours"] >= 5, stats
    assert len(shapes) >= 30, len(shapes)
    for a, b in ((1, 1), (11, 1), (4, 0), (3, 3), (5, 5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return {"stats": dict(stats), "shapes": len(shapes)}


if __name__ == "__main__":
    print(self_test())
