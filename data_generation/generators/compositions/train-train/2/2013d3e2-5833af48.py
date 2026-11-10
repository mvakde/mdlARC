"""Pinwheel Kronecker: a two-scale quarter-turn composition.

Each input shows, on one background:
  * a fine pinwheel  F (2n x 2n): four quarter-turn copies of a base tile T
    (NW = T, NE = rot90cw T, SE = rot180 T, SW = rot90ccw T); tile cells equal to
    the background are placeholders;
  * a coarse placement mask (2k x 2k logical cells, possibly magnified by an
    integer factor s) drawn in a mask ink plus the panel color; it is itself a
    pinwheel of a k x k logical mask M;
  * a solid square panel of side P = 2kn in the panel color.

Output (P x P): every active logical mask cell lying in macro-quadrant q is
replaced by the fine pinwheel's quadrant q (placeholders -> panel color);
inactive cells become solid panel-colored tiles.  Equivalently the output is the
macro pinwheel of kron(M, T).
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# basic helpers


def pinwheel(base: np.ndarray) -> np.ndarray:
    """C4 composite: NW=base, NE=rot cw, SE=rot 180, SW=rot ccw."""
    m = base.shape[0]
    out = np.zeros((2 * m, 2 * m), dtype=base.dtype)
    out[:m, :m] = base
    out[:m, m:] = np.rot90(base, -1)
    out[m:, m:] = np.rot90(base, 2)
    out[m:, :m] = np.rot90(base, 1)
    return out


def _components(nonbg: np.ndarray):
    """8-connected components of a boolean array; list of (rows, cols) arrays."""
    h, w = nonbg.shape
    seen = np.zeros_like(nonbg, dtype=bool)
    comps = []
    for r0 in range(h):
        for c0 in range(w):
            if not nonbg[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < h and 0 <= cc < w and nonbg[rr, cc] and not seen[rr, cc]:
                            seen[rr, cc] = True
                            stack.append((rr, cc))
            arr = np.array(cells)
            comps.append((arr[:, 0], arr[:, 1]))
    return comps


def _border_background(grid: np.ndarray) -> int:
    border = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
    vals, counts = np.unique(border, return_counts=True)
    return int(vals[np.argmax(counts)])


def _parse(grid: np.ndarray):
    """Recover (bg, panel_color, P, fine pinwheel, logical coarse mask, scale, ink, boxes)."""
    grid = np.asarray(grid, dtype=int)
    bg = _border_background(grid)
    nonbg = grid != bg
    comps = _components(nonbg)
    info = []
    for rows, cols in comps:
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        full = len(rows) == (r1 - r0) * (c1 - c0)
        colors = set(np.unique(grid[rows, cols]).tolist())
        info.append(dict(rows=rows, cols=cols, box=(r0, r1, c0, c1), full=full, colors=colors,
                         area=len(rows)))
    panels = [d for d in info if d["full"] and len(d["colors"]) == 1]
    if not panels:
        raise ValueError("no solid panel")
    panel = max(panels, key=lambda d: d["area"])
    pr0, pr1, pc0, pc1 = panel["box"]
    P = pr1 - pr0
    if P != pc1 - pc0:
        raise ValueError("panel is not square")
    pcol = next(iter(panel["colors"]))
    masks = [d for d in info if d is not panel and pcol in d["colors"]]
    if len(masks) != 1:
        raise ValueError("placement mask not unique")
    mask = masks[0]
    if not mask["full"] or len(mask["colors"]) != 2:
        raise ValueError("placement mask must be a two-color rectangle")
    ink = next(iter(mask["colors"] - {pcol}))
    mr0, mr1, mc0, mc1 = mask["box"]
    fine_cells = np.zeros_like(nonbg)
    for d in info:
        if d is panel or d is mask:
            continue
        fine_cells[d["rows"], d["cols"]] = True
    if not fine_cells.any():
        raise ValueError("no fine pinwheel")
    rr, cc = np.nonzero(fine_cells)
    fr0, fr1, fc0, fc1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    F = fr1 - fr0
    if F != fc1 - fc0 or F % 2:
        raise ValueError("fine pinwheel must be an even square")
    fine = grid[fr0:fr1, fc0:fc1].copy()
    n = F // 2
    if P % n:
        raise ValueError("panel side not a multiple of tile side")
    L = P // n
    Q = mr1 - mr0
    if Q != mc1 - mc0 or Q % L:
        raise ValueError("mask size incompatible with panel")
    s = Q // L
    shown = grid[mr0:mr1, mc0:mc1]
    logical = shown[::s, ::s]
    if not np.array_equal(np.kron(logical, np.ones((s, s), dtype=int)), shown):
        raise ValueError("mask blocks not uniform")
    return dict(bg=bg, panel=pcol, P=P, fine=fine, n=n, logical=(logical == ink), scale=s,
                ink=ink, boxes=dict(panel=panel["box"], mask=mask["box"], fine=(fr0, fr1, fc0, fc1)))


def _render(fine: np.ndarray, logical: np.ndarray, bg: int, panel: int) -> np.ndarray:
    n = fine.shape[0] // 2
    L = logical.shape[0]
    half = L // 2
    out = np.full((L * n, L * n), panel, dtype=int)
    for i in range(L):
        for j in range(L):
            if not logical[i, j]:
                continue
            qi, qj = int(i >= half), int(j >= half)
            tile = fine[qi * n:(qi + 1) * n, qj * n:(qj + 1) * n]
            tile = np.where(tile == bg, panel, tile)
            out[i * n:(i + 1) * n, j * n:(j + 1) * n] = tile
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    p = _parse(np.asarray(input_grid, dtype=int))
    if p["logical"].shape[0] % 2:
        raise ValueError("logical mask must have even side")
    return _render(p["fine"], p["logical"], p["bg"], p["panel"])


# --------------------------------------------------------------------------
# generation

TILE_SIZES = (2, 3, 4)
MASK_HALVES = (2, 3, 4)
SCALES = (1, 2, 3)


def _sample_tile(rng, n, palette, bg, *, empty_edge=None, allow_full=True, want_empty_line=False):
    """n x n tile; empty_edge in (None, 'top', 'left')."""
    for _ in range(500):
        kind = rng.choice(["random", "random", "stroke", "blocks", "stair"])
        occ = np.zeros((n, n), dtype=bool)
        if kind == "random":
            d = rng.uniform(0.35, 0.8)
            occ = rng.random((n, n)) < d
        elif kind == "stroke":
            r, c = rng.integers(0, n, 2)
            for _ in range(int(rng.integers(n, n * n))):
                occ[r, c] = True
                dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0)][int(rng.integers(4))]
                r, c = min(max(r + dr, 0), n - 1), min(max(c + dc, 0), n - 1)
        elif kind == "blocks":
            for _ in range(int(rng.integers(1, 4))):
                r0, c0 = rng.integers(0, n, 2)
                h, w = rng.integers(1, n + 1, 2)
                occ[r0:r0 + h, c0:c0 + w] = True
        else:  # staircase
            shift = int(rng.integers(-(n // 2), n // 2 + 1))
            for r in range(n):
                occ[r, :max(0, min(n, r + 1 + shift))] = True
            if rng.random() < 0.5:
                occ = occ[:, ::-1]
        if empty_edge == "top":
            occ[0, :] = False
        elif empty_edge == "left":
            occ[:, 0] = False
        colors = np.array(palette)[rng.integers(0, len(palette), (n, n))]
        if len(palette) > 1 and rng.random() < 0.4:  # color runs by row or column
            line = np.array(palette)[rng.integers(0, len(palette), n)]
            colors = np.tile(line, (n, 1)) if rng.random() < 0.5 else np.tile(line[:, None], (1, n))
        tile = np.where(occ, colors, bg).astype(int)
        nb = tile != bg
        if nb.sum() < 2:
            continue
        if not allow_full and nb.all():
            continue
        if not (nb[0, :].any() or nb[:, 0].any()):
            continue  # composite extent would be invisible
        if empty_edge == "top" and not nb[:, 0].any():
            continue
        if empty_edge == "left" and not nb[0, :].any():
            continue
        if np.array_equal(np.rot90(tile, -1), tile):
            continue  # quarter-turns must be visibly different
        if want_empty_line and not ((~nb).all(axis=1).any() or (~nb).all(axis=0).any()):
            continue
        return tile
    raise RuntimeError("tile sampling failed")


def _sample_mask(rng, k):
    for _ in range(200):
        kind = rng.choice(["random", "random", "line", "diag", "sparse", "dense"])
        if kind == "random":
            m = rng.random((k, k)) < rng.uniform(0.3, 0.7)
        elif kind == "sparse":
            m = np.zeros((k, k), dtype=bool)
            idx = rng.choice(k * k, size=int(rng.integers(1, max(2, k))), replace=False)
            m.flat[idx] = True
        elif kind == "dense":
            m = np.ones((k, k), dtype=bool)
            idx = rng.choice(k * k, size=int(rng.integers(1, max(2, k))), replace=False)
            m.flat[idx] = False
        elif kind == "line":
            m = np.zeros((k, k), dtype=bool)
            if rng.random() < 0.5:
                m[int(rng.integers(k)), :] = True
            else:
                m[:, int(rng.integers(k))] = True
            if rng.random() < 0.5:
                m[int(rng.integers(k)), int(rng.integers(k))] ^= True
        else:
            m = np.eye(k, dtype=bool)
            if rng.random() < 0.5:
                m = m[:, ::-1]
            if rng.random() < 0.5:
                m[int(rng.integers(k)), int(rng.integers(k))] ^= True
        if m.any() and not m.all():
            return m
    raise RuntimeError("mask sampling failed")


def _pack(rng, sizes, max_tries=300):
    """Place square boxes (sides list) with >=1 empty cell between them."""
    for _ in range(max_tries):
        H = int(rng.integers(max(sizes), 31))
        W = int(rng.integers(max(sizes), 31))
        occ = np.zeros((H, W), dtype=bool)
        spots = []
        ok = True
        for side in sizes:
            placed = False
            for _ in range(60):
                if side > H or side > W:
                    break
                r = int(rng.integers(0, H - side + 1))
                c = int(rng.integers(0, W - side + 1))
                if occ[max(r - 1, 0):r + side + 1, max(c - 1, 0):c + side + 1].any():
                    continue
                occ[r:r + side, c:c + side] = True
                spots.append((r, c))
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return H, W, spots
    return None


def _feasible(n, k, s):
    P, F, Q = 2 * k * n, 2 * n, 2 * k * s
    if P > 30 or Q > 30:
        return False
    a, b = sorted((F, Q))
    # Panel beside a column holding the two small boxes (or stacked), or a row arrangement.
    if P + 1 + b <= 30 and a + 1 + b <= 30:
        return True
    if P + 1 + a + 1 + b <= 30:
        return True
    return False


TRIPLES = [(n, k, s) for n in TILE_SIZES for k in MASK_HALVES for s in SCALES if _feasible(n, k, s)]


def _make_pair(rng, triple, bg, *, empty_edge=None, want_empty_line=False, allow_full=True):
    n, k, s = triple
    for _ in range(200):
        others = [c for c in range(10) if c != bg]
        panel = int(rng.choice(others))
        free = [c for c in others if c != panel]
        ncol = int(rng.choice([1, 2, 2, 3, 3, 4])) if n >= 3 else int(rng.choice([1, 2, 2, 3]))
        palette = [int(c) for c in rng.choice(free, size=ncol, replace=False)]
        ink = int(rng.choice(free))
        tile = _sample_tile(rng, n, palette, bg, empty_edge=empty_edge, allow_full=allow_full,
                            want_empty_line=want_empty_line)
        M = _sample_mask(rng, k)
        fine = pinwheel(tile)
        coarse = pinwheel(M.astype(int)).astype(bool)
        shown = np.kron(np.where(coarse, ink, panel), np.ones((s, s), dtype=int))
        P = 2 * k * n
        sizes = [P, 2 * n, 2 * k * s]
        order = list(rng.permutation(3))
        packed = _pack(rng, [sizes[i] for i in order])
        if packed is None:
            continue
        H, W, spots = packed
        if any((r == 0 and sizes[idx] == H) or (c == 0 and sizes[idx] == W)
                    for idx, (r, c) in zip(order, spots)):
            continue  # no object may span the whole canvas height or width
        grid = np.full((H, W), bg, dtype=int)
        objs = [np.full((P, P), panel, dtype=int), fine, shown]
        for idx, (r, c) in zip(order, spots):
            o = objs[idx]
            grid[r:r + o.shape[0], c:c + o.shape[1]] = o
        try:
            if _border_background(grid) != bg:
                continue
            border = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
            if np.mean(border == bg) < 0.6:
                continue  # background must clearly dominate the canvas border
            p = _parse(grid)
        except ValueError:
            continue
        if p["n"] != n or p["scale"] != s or not np.array_equal(p["logical"], coarse):
            continue
        out = _render(fine, coarse, bg, panel)
        meta = dict(n=n, k=k, s=s, tile=tile, M=M, bg=bg, panel=panel, ink=ink)
        return grid, out, meta
    raise RuntimeError("pair construction failed")


def _check_counts(num_train, num_test):
    if not isinstance(num_train, (int, np.integer)) or not isinstance(num_test, (int, np.integer)):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("need 2<=num_train<=10, 1<=num_test<=4, num_train>num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(50):
        try:
            return _generate(rng, int(num_train), int(num_test))
        except RuntimeError:
            continue
    raise RuntimeError("generation failed")


def _generate(rng, num_train, num_test):
    vary_bg = rng.random() < 0.35
    base_bg = int(rng.choice([0] * 5 + list(range(1, 10))))

    def pick_bg():
        return int(rng.integers(0, 10)) if vary_bg else base_bg

    # test triples: held-out combinations, possibly larger or smaller than training ones
    triples = list(TRIPLES)
    test_triples = [triples[i] for i in rng.choice(len(triples), size=num_test, replace=num_test > len(triples))]
    pool = [t for t in triples if t not in test_triples] or triples
    # training triples: ensure both an unscaled and a magnified mask
    unscaled = [t for t in pool if t[2] == 1]
    scaled = [t for t in pool if t[2] >= 2]
    train_triples = [unscaled[int(rng.integers(len(unscaled)))], scaled[int(rng.integers(len(scaled)))]]
    while len(train_triples) < num_train:
        train_triples.append(pool[int(rng.integers(len(pool)))])
    order = rng.permutation(num_train)
    train_triples = [train_triples[i] for i in order]

    # which train pair must demonstrate an internal/edge empty line in its tile
    line_idx = int(rng.integers(num_train))
    test_edge_mode = rng.random() < 0.6

    pairs = []
    seen = set()
    for i, t in enumerate(train_triples + test_triples):
        is_test = i >= num_train
        edge = None
        if is_test and test_edge_mode:
            edge = "top" if rng.random() < 0.5 else "left"
        elif not is_test and rng.random() < 0.25:
            edge = "top" if rng.random() < 0.5 else "left"
        want_line = (not is_test and i == line_idx)
        allow_full = is_test is False and rng.random() < 0.3
        for _ in range(20):
            g, o, meta = _make_pair(rng, t, pick_bg(), empty_edge=edge, want_empty_line=want_line,
                                    allow_full=allow_full)
            key = g.tobytes() + bytes(g.shape)
            if key not in seen:
                seen.add(key)
                break
        else:
            raise RuntimeError("duplicate inputs")
        pairs.append({"input": g.tolist(), "output": o.tolist()})
    puzzle = {"train": pairs[:num_train], "test": pairs[num_train:]}
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError as error:  # resample (deterministically) when evidence is insufficient
        raise RuntimeError(str(error)) from error
    return puzzle


# --------------------------------------------------------------------------
# validation


def _shortcuts(grid):
    """Outputs of plausible simpler rules (None when not same-shaped)."""
    p = _parse(grid)
    fine, logical, bg, panel, n = p["fine"], p["logical"], p["bg"], p["panel"], p["n"]
    T = fine[:n, :n]
    L = logical.shape[0]
    res = {}
    # 1. sequential chaining: extract base tile (A) then plain Kronecker (B), no rotation
    res["no_rotation"] = _render(np.tile(T, (2, 2)), logical, bg, panel)
    # 2. stamp the whole fine pinwheel at each active NW-mask cell (ignores quadrant roles)
    nw = logical[:L // 2, :L // 2]
    big = np.full((L // 2 * 2 * n,) * 2, panel, dtype=int)
    for i in range(L // 2):
        for j in range(L // 2):
            if nw[i, j]:
                big[i * 2 * n:(i + 1) * 2 * n, j * 2 * n:(j + 1) * 2 * n] = np.where(fine == bg, panel, fine)
    res["whole_pinwheel_stamp"] = big
    # 3. keep placeholders as background instead of panel color
    out = np.full((L * n, L * n), panel, dtype=int)
    for i in range(L):
        for j in range(L):
            if logical[i, j]:
                qi, qj = int(i >= L // 2), int(j >= L // 2)
                out[i * n:(i + 1) * n, j * n:(j + 1) * n] = fine[qi * n:(qi + 1) * n, qj * n:(qj + 1) * n]
    res["keep_background"] = out
    # 4. opposite turning sense (NE gets the counter-clockwise copy)
    swapped = fine.copy()
    swapped[:n, n:], swapped[n:, :n] = fine[n:, :n], fine[:n, n:]
    res["wrong_sense"] = _render(swapped, logical, bg, panel)
    return res


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    beaten = {"no_rotation": 0, "whole_pinwheel_stamp": 0, "keep_background": 0, "wrong_sense": 0}
    scales, triples = [], []
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert set(pair) == {"input", "output"}
            g = np.asarray(pair["input"], dtype=int)
            o = np.asarray(pair["output"], dtype=int)
            for a in (g, o):
                assert a.ndim == 2 and 1 <= a.shape[0] <= 30 and 1 <= a.shape[1] <= 30
                assert a.min() >= 0 and a.max() <= 9
            key = g.tobytes() + bytes(g.shape)
            assert key not in seen, "duplicate input"
            seen.add(key)
            p = _parse(g)
            n, fine, bg = p["n"], p["fine"], p["bg"]
            T = fine[:n, :n]
            assert np.array_equal(pinwheel(T), fine), "fine object must be a quarter-turn pinwheel"
            assert not np.array_equal(np.rot90(T, -1), T), "tile must change under a quarter turn"
            L = p["logical"].shape[0]
            assert L % 2 == 0 and L >= 4
            M = p["logical"][:L // 2, :L // 2]
            assert np.array_equal(pinwheel(M.astype(int)).astype(bool), p["logical"]), "mask is a pinwheel"
            assert M.any() and not M.all()
            assert p["P"] == L * n
            assert bg not in (p["panel"], p["ink"])
            assert p["panel"] not in set(np.unique(fine).tolist()), "panel color reserved"
            assert np.array_equal(solve(g), o), "solve mismatch"
            scales.append(p["scale"])
            triples.append((split, n, L // 2, p["scale"]))
            if split == "train":
                for name, alt in _shortcuts(g).items():
                    if alt.shape != o.shape or not np.array_equal(alt, o):
                        beaten[name] += 1
    tr_scales = [t[3] for t in triples if t[0] == "train"]
    assert 1 in tr_scales and any(s >= 2 for s in tr_scales), "train must show unscaled and magnified masks"
    for name, count in beaten.items():
        assert count >= 1, f"shortcut {name} not refuted by training"


def self_test() -> None:
    import collections
    stats = collections.Counter()
    sigs = set()
    for seed in range(42):
        pz = generate_puzzle(seed)
        validate_generated_puzzle(pz)
        sigs.add(repr(pz))
        for split in ("train", "test"):
            for pair in pz[split]:
                p = _parse(np.asarray(pair["input"]))
                stats[("n", p["n"])] += 1
                stats[("k", p["logical"].shape[0] // 2)] += 1
                stats[("s", p["scale"])] += 1
                stats[("bg", p["bg"])] += 1
                stats[("shape", np.asarray(pair["input"]).shape)] += 1
    assert len(sigs) == 42
    for key in (2, 3, 4):
        assert stats[("n", key)] > 0 and stats[("k", key)] > 0
    for key in (1, 2, 3):
        assert stats[("s", key)] > 0
    assert len([k for k in stats if k[0] == "bg"]) >= 4
    assert len([k for k in stats if k[0] == "shape"]) >= 30
    for a, b in ((1, 1), (11, 1), (4, 0), (3, 4)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test ok", len(TRIPLES), TRIPLES)
