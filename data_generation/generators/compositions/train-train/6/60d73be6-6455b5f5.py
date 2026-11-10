"""Mirrored floor plan ranking.

Composition of 60d73be6 (fourfold reflection across a full row/column cross,
with boundary clipping and unmatched margins) and 6455b5f5 (thin walls split
the background into rooms; the unique largest room and every smallest room are
highlighted).

Input: a dominant background, one complete row and one complete column (the
cross) and, in exactly one of the four open quadrants, a one-cell-wide wall
plan attached to the cross / canvas edges.  The other three quadrants are blank.

Output: the wall plan is mirrored across both cross lines (clipped at the canvas
edge, unmatched space left open), then the 4-connected background regions of the
completed plan are ranked by cell count.  The unique largest region becomes 1
(blue) and every region tied for the smallest area becomes 8 (azure).  Walls,
the cross and intermediate regions are unchanged.  Highlight colors are a fixed
family convention and never occur in inputs.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

MAJOR = 1
MINOR = 8
BASE_COLORS = (0, 2, 3, 4, 5, 6, 7, 9)
QUADRANTS = ("UL", "UR", "LL", "LR")


# --------------------------------------------------------------------------
# Shared grid helpers
# --------------------------------------------------------------------------

def _components(free: np.ndarray):
    """4-connected components of True cells. Returns label grid and size list."""
    h, w = free.shape
    labels = -np.ones((h, w), dtype=int)
    sizes = []
    for r in range(h):
        for c in range(w):
            if free[r, c] and labels[r, c] < 0:
                idx = len(sizes)
                stack = [(r, c)]
                labels[r, c] = idx
                n = 0
                while stack:
                    y, x = stack.pop()
                    n += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and free[yy, xx] and labels[yy, xx] < 0:
                            labels[yy, xx] = idx
                            stack.append((yy, xx))
                sizes.append(n)
    return labels, sizes


def _rank_fill(grid: np.ndarray, bg: int):
    """Apply parent B's area ranking to the background of grid (copy)."""
    out = grid.copy()
    labels, sizes = _components(grid == bg)
    if not sizes:
        return out
    mx, mn = max(sizes), min(sizes)
    if sizes.count(mx) == 1 and mn < mx:
        out[labels == sizes.index(mx)] = MAJOR
    if mn < mx:
        for i, s in enumerate(sizes):
            if s == mn:
                out[labels == i] = MINOR
    return out


def _find_cross(grid: np.ndarray):
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] != bg)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] != bg)]
    return bg, rows, cols


# --------------------------------------------------------------------------
# Solver (input only)
# --------------------------------------------------------------------------

def solve(input_grid: np.ndarray) -> np.ndarray:
    g = np.asarray(input_grid, dtype=int)
    h, w = g.shape
    bg, rows, cols = _find_cross(g)
    if len(rows) != 1 or len(cols) != 1:
        raise ValueError("expected exactly one full row and one full column")
    r0, c0 = rows[0], cols[0]
    out = g.copy()
    # Mirror every non-background cell of the open quadrants across both axes.
    for r in range(h):
        if r == r0:
            continue
        for c in range(w):
            if c == c0 or g[r, c] == bg:
                continue
            for rr, cc in ((r, 2 * c0 - c), (2 * r0 - r, c), (2 * r0 - r, 2 * c0 - c)):
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = g[r, c]
    return _rank_fill(out, bg)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _guillotine(rng, sh, sw, nsplits):
    """Local wall plan (sh x sw) made by recursive full splits of rooms."""
    wall = np.zeros((sh, sw), dtype=bool)
    rooms = [(0, 0, sh, sw)]  # (top, left, height, width) in local coords
    for _ in range(nsplits):
        cands = [i for i, (t, l, hh, ww) in enumerate(rooms) if hh >= 3 or ww >= 3]
        if not cands:
            break
        weights = np.array([rooms[i][2] * rooms[i][3] for i in cands], dtype=float)
        i = cands[int(rng.choice(len(cands), p=weights / weights.sum()))]
        t, l, hh, ww = rooms.pop(i)
        horiz_ok, vert_ok = hh >= 3, ww >= 3
        if horiz_ok and vert_ok:
            horiz = bool(rng.random() < hh / (hh + ww))
        else:
            horiz = horiz_ok
        if horiz:
            k = int(rng.integers(1, hh - 1))
            wall[t + k, l:l + ww] = True
            rooms += [(t, l, k, ww), (t + k + 1, l, hh - k - 1, ww)]
        else:
            k = int(rng.integers(1, ww - 1))
            wall[t:t + hh, l + k] = True
            rooms += [(t, l, hh, k), (t, l + k + 1, hh, ww - k - 1)]
    return wall, rooms


def _add_spur(rng, wall, rooms):
    """A dead-end wall stub that dents a room without splitting it."""
    cands = [r for r in rooms if r[2] >= 4 and r[3] >= 3 or r[3] >= 4 and r[2] >= 3]
    if not cands:
        return False
    t, l, hh, ww = cands[int(rng.integers(len(cands)))]
    if hh >= 4 and ww >= 3 and (ww < 4 or rng.random() < 0.5):
        # vertical stub from the top or bottom side of the room
        c = l + int(rng.integers(1, ww - 1))
        n = int(rng.integers(1, hh - 2))
        if rng.random() < 0.5:
            wall[t:t + n, c] = True
        else:
            wall[t + hh - n:t + hh, c] = True
    else:
        r = t + int(rng.integers(1, hh - 1))
        n = int(rng.integers(1, ww - 2))
        if rng.random() < 0.5:
            wall[r, l:l + n] = True
        else:
            wall[r, l + ww - n:l + ww] = True
    return True


def _place(local, quad, r0, c0, h, w, value, grid):
    """Write local plan (distance-from-axis coords) into one quadrant, clipping."""
    ys, xs = np.nonzero(local)
    for i, j in zip(ys, xs):
        r = r0 - 1 - i if quad[0] == "U" else r0 + 1 + i
        c = c0 - 1 - j if quad[1] == "L" else c0 + 1 + j
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = value


def _quad_slices(quad, r0, c0, h, w):
    rs = slice(0, r0) if quad[0] == "U" else slice(r0 + 1, h)
    cs = slice(0, c0) if quad[1] == "L" else slice(c0 + 1, w)
    return rs, cs


def _seed_local_shortcut(inp, quad, r0, c0, bg):
    """Shortcut: rank rooms of the seed quadrant only, then mirror the colors."""
    h, w = inp.shape
    rs, cs = _quad_slices(quad, r0, c0, h, w)
    sub = inp[rs, cs]
    ranked = _rank_fill(sub, bg)
    tmp = inp.copy()
    tmp[rs, cs] = ranked
    out = tmp.copy()
    for r in range(rs.start, rs.stop):
        for c in range(cs.start, cs.stop):
            if tmp[r, c] == bg:
                continue
            for rr, cc in ((r, 2 * c0 - c), (2 * r0 - r, c), (2 * r0 - r, 2 * c0 - c)):
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = tmp[r, c]
    return out


def _analyze(inp, out, quad, r0, c0, bg):
    """Facts used for puzzle-level diversity and shortcut checks."""
    h, w = inp.shape
    walls = out != bg
    walls &= (out != MAJOR) & (out != MINOR)
    labels, sizes = _components(~walls)
    mx, mn = max(sizes), min(sizes)
    imax = sizes.index(mx)
    rs, cs = _quad_slices(quad, r0, c0, h, w)
    seed_mask = np.zeros((h, w), dtype=bool)
    seed_mask[rs, cs] = True
    max_region = labels == imax
    max_kind = "seed" if np.any(max_region & seed_mask) else "mirror"
    return {
        "max_kind": max_kind,
        "min_ties": sizes.count(mn),
        "min_area": mn,
        "regions": len(sizes),
        "shortcut_seed_local_fails": not np.array_equal(
            _seed_local_shortcut(inp, quad, r0, c0, bg), out),
        "shortcut_input_rank_fails": not np.array_equal(_rank_fill(inp, bg), out),
    }


def _make_pair(rng, palette, big=False, want=None):
    """Sample one pair; returns (input, output, facts) or None on rejection."""
    lo, hi = (4, 11) if big else (3, 9)
    sh = int(rng.integers(lo, hi + 1))
    sw = int(rng.integers(lo, hi + 1))

    def opposite(s):
        # An exactly centred axis makes the whole output mirror-symmetric
        # about it, so no region could be the unique maximum: always clip or
        # leave an unmatched margin.
        if rng.random() < 0.5:
            return int(rng.integers(max(1, s // 3), s))
        return s + int(rng.integers(1, 6))

    oh, ow = opposite(sh), opposite(sw)
    h, w = sh + oh + 1, sw + ow + 1
    if h > 30 or w > 30:
        return None
    quad = QUADRANTS[int(rng.integers(4))] if want is None or "quad" not in want else want["quad"]
    top = sh if quad[0] == "U" else oh
    left = sw if quad[1] == "L" else ow
    r0, c0 = top, left
    nsplits = int(rng.integers(2, 5 + (2 if big else 0)))
    local, rooms = _guillotine(rng, sh, sw, nsplits)
    if not local.any():
        return None
    if rng.random() < 0.3:
        _add_spur(rng, local, rooms)
    bg, cross, inner = palette
    inp = np.full((h, w), bg, dtype=int)
    inp[r0, :] = cross
    inp[:, c0] = cross
    _place(local, quad, r0, c0, h, w, inner, inp)
    # Expected output built in local coordinates (independent of solve()).
    full = inp.copy()
    for q in QUADRANTS:
        _place(local, q, r0, c0, h, w, inner, full)
    # every mirrored quadrant must actually receive some wall
    for q in QUADRANTS:
        rs, cs = _quad_slices(q, r0, c0, h, w)
        if not np.any(full[rs, cs] == inner):
            return None
    labels, sizes = _components(full == bg)
    mx, mn = max(sizes), min(sizes)
    if sizes.count(mx) != 1 or mn >= mx:
        return None
    if not any(mn < s < mx for s in sizes):
        return None
    if sizes.count(mn) > 6:
        return None
    out = _rank_fill(full, bg)
    facts = _analyze(inp, out, quad, r0, c0, bg)
    facts.update(quad=quad, shape=(h, w), vclip=oh < sh, hclip=ow < sw,
                 vmargin=oh > sh, hmargin=ow > sw)
    if want:
        for k, v in want.items():
            if k != "quad" and facts.get(k) != v:
                return None
    return inp, out, facts


def _palette(rng):
    cols = [int(c) for c in rng.permutation(BASE_COLORS)]
    bg, cross = cols[0], cols[1]
    inner = cross if rng.random() < 0.6 else cols[2]
    return bg, cross, inner


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = 4 if num_train is None else num_train
    num_test = 1 if num_test is None else num_test
    if isinstance(num_train, bool) or isinstance(num_test, bool):
        raise ValueError("counts must be integers")
    if not (isinstance(num_train, (int, np.integer)) and isinstance(num_test, (int, np.integer))):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("illegal train/test counts")
    rng = np.random.default_rng(seed)
    shared = rng.random() < 0.5
    base_palette = _palette(rng)

    def pal():
        return base_palette if shared else _palette(rng)

    # Training demonstrations: guarantee the key distinctions are visible.
    plans = [
        {"shortcut_seed_local_fails": True, "max_kind": "mirror"},
        {"shortcut_seed_local_fails": True, "max_kind": "seed"},
    ]
    if rng.random() < 0.5:
        plans.reverse()
    while len(plans) < num_train:
        plans.append(None)
    test_plans = [{"shortcut_seed_local_fails": True} for _ in range(num_test)]

    pairs, seen = [], set()
    facts_all = []
    for idx, want in enumerate(plans + test_plans):
        is_test = idx >= num_train
        for _attempt in range(5000):
            res = _make_pair(rng, pal(), big=is_test and rng.random() < 0.7, want=want)
            if res is None:
                continue
            inp, out, facts = res
            key = inp.tobytes() + bytes(inp.shape)
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"input": inp, "output": out})
            facts_all.append(facts)
            break
        else:
            raise RuntimeError("generation failed")
    train = pairs[:num_train]
    order = rng.permutation(num_train)
    train = [train[i] for i in order]
    return {"train": train, "test": pairs[num_train:]}


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _check_pair(inp, out):
    inp = np.asarray(inp, dtype=int)
    out = np.asarray(out, dtype=int)
    assert inp.ndim == 2 and inp.shape == out.shape
    h, w = inp.shape
    assert 1 <= h <= 30 and 1 <= w <= 30
    assert inp.min() >= 0 and inp.max() <= 9
    assert MAJOR not in inp and MINOR not in inp, "highlight colors reserved"
    bg, rows, cols = _find_cross(inp)
    assert len(rows) == 1 and len(cols) == 1, "one full row and one full column"
    r0, c0 = rows[0], cols[0]
    assert 0 < r0 < h - 1 and 0 < c0 < w - 1, "four nonempty quadrants"
    cross = int(inp[r0, 0])
    assert np.all(inp[r0] == cross) and np.all(inp[:, c0] == cross), "uniform cross"
    active = [q for q in QUADRANTS
              if np.any(inp[_quad_slices(q, r0, c0, h, w)] != bg)]
    assert len(active) == 1, "exactly one populated quadrant"
    quad = active[0]
    sub = inp[_quad_slices(quad, r0, c0, h, w)]
    inner_colors = set(np.unique(sub).tolist()) - {bg}
    assert len(inner_colors) == 1, "single wall color in seed quadrant"
    # No 2x2 wall blocks (walls are one cell thick) in the seed quadrant + cross.
    wmask = inp != bg
    assert not np.any(wmask[:-1, :-1] & wmask[1:, :-1] & wmask[:-1, 1:] & wmask[1:, 1:]), "thin walls"
    # Output: independent reconstruction via quadrant-local reflection.
    out_walls = (out != MAJOR) & (out != MINOR) & (out != bg)
    exp = inp.copy()
    rs, cs = _quad_slices(quad, r0, c0, h, w)
    ys, xs = np.nonzero(inp[rs, cs] != bg)
    for y, x in zip(ys + rs.start, xs + cs.start):
        di, dj = abs(y - r0), abs(x - c0)
        for sy in (-1, 1):
            for sx in (-1, 1):
                yy, xx = r0 + sy * di, c0 + sx * dj
                if 0 <= yy < h and 0 <= xx < w:
                    exp[yy, xx] = inp[y, x]
    assert np.array_equal(out_walls, exp != bg), "walls are the clipped fourfold mirror"
    assert np.array_equal(out[out_walls], exp[out_walls]), "wall colors preserved"
    labels, sizes = _components(~out_walls)
    mx, mn = max(sizes), min(sizes)
    assert sizes.count(mx) == 1 and mn < mx, "unique max, distinct min"
    assert any(mn < s < mx for s in sizes), "intermediate region exists"
    for i, s in enumerate(sizes):
        vals = set(np.unique(out[labels == i]).tolist())
        want = MAJOR if s == mx else MINOR if s == mn else bg
        assert vals == {want}, "region coloring"
    assert np.array_equal(solve(inp), out), "solve agrees"
    for q in QUADRANTS:
        assert np.any(out_walls[_quad_slices(q, r0, c0, h, w)]), "mirrored walls visible"
    return quad, bg, r0, c0


def validate_generated_puzzle(puzzle) -> bool:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    tr, te = puzzle["train"], puzzle["test"]
    assert 2 <= len(tr) <= 10 and 1 <= len(te) <= 4 and len(tr) > len(te)
    seen = set()
    shortcut_fail_train = 0
    kinds = set()
    for split in (tr, te):
        for p in split:
            assert set(p) == {"input", "output"}
            inp = np.asarray(p["input"], dtype=int)
            out = np.asarray(p["output"], dtype=int)
            quad, bg, r0, c0 = _check_pair(inp, out)
            key = (inp.shape, inp.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            facts = _analyze(inp, out, quad, r0, c0, bg)
            assert facts["shortcut_input_rank_fails"]
            if split is tr:
                shortcut_fail_train += facts["shortcut_seed_local_fails"]
                kinds.add(facts["max_kind"])
    assert shortcut_fail_train >= 2, "training must refute the seed-local shortcut"
    assert len(kinds) == 2, "training shows the maximum both in and outside the seed quadrant"
    return True


def self_test() -> bool:
    for a, b in ((1, 1), (11, 1), (4, 0), (3, 4), (2, 2), (5, 5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    quads, shapes, palettes, ties, test_kinds = Counter(), set(), set(), Counter(), Counter()
    clips = Counter()
    sigs = set()
    for seed in range(42):
        pz = generate_puzzle(seed)
        validate_generated_puzzle(pz)
        again = generate_puzzle(seed)
        for s in ("train", "test"):
            for p, q in zip(pz[s], again[s]):
                assert np.array_equal(p["input"], q["input"]) and np.array_equal(p["output"], q["output"])
        sigs.add(b"".join(np.asarray(p["input"]).tobytes() for p in pz["train"]))
        for s in ("train", "test"):
            for p in pz[s]:
                inp, out = np.asarray(p["input"]), np.asarray(p["output"])
                quad, bg, r0, c0 = _check_pair(inp, out)
                f = _analyze(inp, out, quad, r0, c0, bg)
                quads[quad] += 1
                shapes.add(inp.shape)
                palettes.add((bg, int(inp[r0, 0])))
                ties[min(f["min_ties"], 4)] += 1
                h, w = inp.shape
                clips[(r0 * 2 + 1 != h, c0 * 2 + 1 != w)] += 1
                if s == "test":
                    test_kinds[f["max_kind"]] += 1
    assert len(sigs) == 42
    assert len(quads) == 4 and min(quads.values()) >= 10, quads
    assert len(shapes) >= 40, len(shapes)
    assert len(palettes) >= 15
    assert len(ties) >= 3, ties
    assert len(test_kinds) == 2, test_kinds
    for n_tr in range(2, 11):
        for n_te in range(1, min(4, n_tr - 1) + 1):
            validate_generated_puzzle(generate_puzzle(7, n_tr, n_te))
    return True


if __name__ == "__main__":
    self_test()
    print("ok")
