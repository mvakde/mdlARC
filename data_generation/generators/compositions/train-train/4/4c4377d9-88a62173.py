"""Mirror-tiling odd-one-out (composition of 4c4377d9 and 88a62173).

The input is a grid of R x C equal tiles cut apart by uniform separator lines.
Neighbouring tiles are meant to be mirror images of each other across every
seam: crossing a horizontal separator reverses row order (the vertical
reflection of 4c4377d9), crossing a vertical separator reverses column order.
Hence tile (r, c) should equal the top-left master flipped vertically when r is
odd and horizontally when c is odd.  Exactly one tile breaks this mirror
tiling.  Undoing each tile's seam reflections makes all tiles but one
identical (the odd-one-out of 88a62173); the output is that defective tile,
cropped exactly as it appears in the input (no repair, no normalisation).
"""
from __future__ import annotations

import json

import numpy as np

# ---------------------------------------------------------------------------
# Parent pairs (exact copies are forbidden).  The family's outputs are single
# tiles of a >=3 tile mirror layout, so none can coincide, but we check anyway.
_PARENT_INPUTS = {
    "9494\n9944\n4444", "4114\n1111\n4441", "3355\n3553\n5533", "9959\n5599\n9599",
    "4499\n4444\n4499", "10010\n01001\n00000\n10010\n11001",
    "02002\n22022\n00000\n02022\n22020", "88008\n80080\n00000\n88088\n80080",
    "55050\n05005\n00000\n55055\n05005",
}

_LAYOUTS = [(1, 3), (3, 1), (1, 4), (4, 1), (2, 2), (2, 3), (3, 2), (3, 3),
            (2, 4), (4, 2), (3, 4), (4, 3)]
_LAYOUT_WEIGHTS = np.array([6, 6, 3, 3, 14, 7, 7, 5, 3, 3, 2, 2], dtype=float)
_LAYOUT_WEIGHTS /= _LAYOUT_WEIGHTS.sum()

_DEFECTS = ("orient", "swap", "toggle", "multi")


def _text(grid):
    return "\n".join("".join(str(int(v)) for v in row) for row in np.asarray(grid))


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------

def _runs(flags):
    """Maximal runs of False entries (tile bands) as (start, stop)."""
    runs, start = [], None
    for i, f in enumerate(list(flags) + [True]):
        if not f and start is None:
            start = i
        elif f and start is not None:
            runs.append((start, i))
            start = None
    return runs


def parse(grid):
    """Return (row_bands, col_bands, separator_color) of a mirror tiling."""
    g = np.asarray(grid, dtype=int)
    urow = np.all(g == g[:, :1], axis=1)
    ucol = np.all(g == g[:1, :], axis=0)
    sep_colors = {int(g[i, 0]) for i in np.flatnonzero(urow)} | {int(g[0, j]) for j in np.flatnonzero(ucol)}
    if len(sep_colors) != 1:
        raise ValueError("separator lines must exist and share one colour")
    rows, cols = _runs(urow), _runs(ucol)
    if not rows or not cols:
        raise ValueError("no tiles")
    if len({b - a for a, b in rows}) != 1 or len({b - a for a, b in cols}) != 1:
        raise ValueError("tiles must have equal sizes")
    if len(rows) * len(cols) < 3:
        raise ValueError("need at least three tiles")
    return rows, cols, sep_colors.pop()


def _normalise(tile, r, c):
    t = tile
    if r % 2:
        t = t[::-1, :]
    if c % 2:
        t = t[:, ::-1]
    return t


def _tiles(g, rows, cols):
    return {(r, c): g[a:b, x:y] for r, (a, b) in enumerate(rows) for c, (x, y) in enumerate(cols)}


def _all_but_one(patterns):
    """patterns: dict pos -> array.  Return the unique pos whose pattern differs
    while all others are identical, else None."""
    keys = {pos: _text(p) for pos, p in patterns.items()}
    counts = {}
    for k in keys.values():
        counts[k] = counts.get(k, 0) + 1
    if len(counts) != 2:
        return None
    n = len(keys)
    major = [k for k, v in counts.items() if v == n - 1]
    minor = [k for k, v in counts.items() if v == 1]
    if not major or not minor or major[0] == minor[0]:
        return None
    return next(pos for pos, k in keys.items() if k == minor[0])


def find_odd(grid):
    g = np.asarray(grid, dtype=int)
    rows, cols, _ = parse(g)
    tiles = _tiles(g, rows, cols)
    pos = _all_but_one({p: _normalise(t, *p) for p, t in tiles.items()})
    if pos is None:
        raise ValueError("mirror tiling does not have exactly one odd tile")
    return pos, tiles


def solve(input_grid):
    pos, tiles = find_odd(input_grid)
    return np.array(tiles[pos], dtype=int).copy()


# ---------------------------------------------------------------------------
# Shortcut solvers (used to prove the demonstrations are informative)
# ---------------------------------------------------------------------------

def _shortcut_literal(g):
    """88a62173 alone: literal odd-one-out, ignoring the seam reflections."""
    rows, cols, _ = parse(g)
    return _all_but_one(_tiles(g, rows, cols))


def _shortcut_partial(g, axis):
    """Only one seam direction reflects (axis 'v': rows only, 'h': cols only)."""
    rows, cols, _ = parse(g)
    tiles = _tiles(g, rows, cols)
    pats = {p: _normalise(t, p[0] if axis == "v" else 0, p[1] if axis == "h" else 0)
            for p, t in tiles.items()}
    return _all_but_one(pats)


def _shortcut_histogram(g):
    """Odd colour histogram (reflections preserve colour counts)."""
    rows, cols, _ = parse(g)
    tiles = _tiles(g, rows, cols)
    pats = {p: np.bincount(t.ravel(), minlength=10)[None, :] for p, t in tiles.items()}
    return _all_but_one(pats)


def _shortcut_master(g):
    """Compare every tile with the (normalised) top-left tile; the first that
    differs is called odd (ignores the majority vote)."""
    rows, cols, _ = parse(g)
    tiles = _tiles(g, rows, cols)
    ref = _text(tiles[(0, 0)])
    for p in sorted(tiles):
        if _text(_normalise(tiles[p], *p)) != ref:
            return p
    return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    for name, v in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if not 2 <= num_train <= 10:
        raise ValueError("num_train must be in 2..10")
    if not 1 <= num_test <= 4:
        raise ValueError("num_test must be in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _lines_ok(t):
    return (np.all(np.any(t != t[:, :1], axis=1)) and np.all(np.any(t != t[:1, :], axis=0)))


def _sym(t):
    return bool(np.array_equal(t, t[:, ::-1])), bool(np.array_equal(t, t[::-1, :]))


def _sample_motif(rng, h, w, colors, symmetry):
    """colors[0] is the tile background; colors[1:] are motif colours."""
    bg, fgs = colors[0], colors[1:]
    for _ in range(300):
        hh = (h + 1) // 2 if symmetry in ("v", "full") else h
        ww = (w + 1) // 2 if symmetry in ("h", "full") else w
        kind = rng.integers(0, 4)
        t = np.full((hh, ww), bg, dtype=int)
        if kind == 0:  # random scatter
            dens = rng.uniform(0.3, 0.65)
            m = rng.random((hh, ww)) < dens
            t[m] = rng.choice(fgs, size=int(m.sum()))
        elif kind == 1:  # random walk stroke
            r, c = int(rng.integers(hh)), int(rng.integers(ww))
            col = int(rng.choice(fgs))
            for _ in range(int(rng.integers(hh * ww // 2, hh * ww + 2))):
                t[r, c] = col
                if rng.random() < 0.25:
                    col = int(rng.choice(fgs))
                d = rng.integers(4)
                r = int(np.clip(r + (d == 0) - (d == 1), 0, hh - 1))
                c = int(np.clip(c + (d == 2) - (d == 3), 0, ww - 1))
        elif kind == 2:  # rectangles
            for _ in range(int(rng.integers(2, 5))):
                a, b = sorted(rng.integers(0, hh, 2))
                x, y = sorted(rng.integers(0, ww, 2))
                t[a:b + 1, x:y + 1] = int(rng.choice(fgs))
            m = rng.random((hh, ww)) < 0.2
            t[m] = bg
        else:  # staircase / diagonal flavour
            off = int(rng.integers(0, max(hh, ww)))
            for i in range(hh):
                for j in range(ww):
                    if (i + j + off) % int(rng.integers(2, 4)) == 0 or rng.random() < 0.15:
                        t[i, j] = int(rng.choice(fgs))
        if symmetry in ("h", "full"):
            t = np.concatenate([t, t[:, : w - ww][:, ::-1]], axis=1)
        if symmetry in ("v", "full"):
            t = np.concatenate([t, t[: h - hh][::-1]], axis=0)
        if t.shape != (h, w) or not _lines_ok(t):
            continue
        hs, vs = _sym(t)
        want = {"none": (False, False), "h": (True, False), "v": (False, True), "full": (True, True)}[symmetry]
        if (hs, vs) != want:
            continue
        return t
    return None


def _expected(master, r, c):
    return _normalise(master, r, c)


def _make_defect(rng, expected, kind, colors):
    h, w = expected.shape
    bg = colors[0]
    d = expected.copy()
    if kind == "orient":
        opts = [expected[:, ::-1], expected[::-1, :], expected[::-1, ::-1]]
        opts = [o for o in opts if not np.array_equal(o, expected)]
        if not opts:
            return None
        return opts[int(rng.integers(len(opts)))].copy()
    if kind == "swap":
        cells = [(i, j) for i in range(h) for j in range(w)]
        for _ in range(40):
            a = cells[int(rng.integers(len(cells)))]
            if rng.random() < 0.6:  # adjacent swap (close difference)
                nb = [(a[0] + di, a[1] + dj) for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0))
                      if 0 <= a[0] + di < h and 0 <= a[1] + dj < w]
                b = nb[int(rng.integers(len(nb)))]
            else:
                b = cells[int(rng.integers(len(cells)))]
            if d[a] != d[b]:
                d[a], d[b] = d[b], d[a]
                return d
        return None
    if kind == "toggle":
        i, j = int(rng.integers(h)), int(rng.integers(w))
        others = [c for c in colors if c != d[i, j]]
        d[i, j] = int(rng.choice(others)) if d[i, j] != bg or rng.random() < 0.3 else int(rng.choice(colors[1:]))
        return d
    # multi: 2-4 independent changes
    for _ in range(int(rng.integers(2, 5))):
        i, j = int(rng.integers(h)), int(rng.integers(w))
        d[i, j] = int(rng.choice([c for c in colors if c != d[i, j]]))
    return d


def _assemble(tiles, R, C, h, w, sr, sc, sep):
    H, W = R * h + (R - 1) * sr, C * w + (C - 1) * sc
    g = np.full((H, W), sep, dtype=int)
    for (r, c), t in tiles.items():
        a, x = r * (h + sr), c * (w + sc)
        g[a:a + h, x:x + w] = t
    return g


def _sample_palette(rng, ncol, sep_mode):
    perm = [int(v) for v in rng.permutation(10)]
    colors = perm[:ncol]
    sep = colors[0] if sep_mode == "bg" else perm[ncol]
    return colors, sep


def _sample_pair(rng, spec):
    """spec keys: layout, tile, symmetry, defect, pos, sep_mode, ncol."""
    for _ in range(200):
        R, C = spec.get("layout") or _LAYOUTS[int(rng.choice(len(_LAYOUTS), p=_LAYOUT_WEIGHTS))]
        sr = int(rng.integers(1, 3)) if rng.random() < 0.4 else 1
        sc = int(rng.integers(1, 3)) if rng.random() < 0.4 else 1
        hmax = min(8, (30 - (R - 1) * sr) // R)
        wmax = min(8, (30 - (C - 1) * sc) // C)
        if hmax < 2 or wmax < 2:
            continue
        if spec.get("tile"):
            h, w = spec["tile"]
            if h > hmax or w > wmax:
                continue
        else:
            h, w = int(rng.integers(2, hmax + 1)), int(rng.integers(2, wmax + 1))
        symmetry = spec.get("symmetry") or str(rng.choice(["none", "none", "none", "h", "v"]))
        if symmetry in ("h", "full") and w < 3:
            continue
        if symmetry in ("v", "full") and h < 3:
            continue
        ncol = spec.get("ncol") or int(rng.choice([2, 2, 2, 3, 3, 4]))
        sep_mode = spec.get("sep_mode") or ("bg" if rng.random() < 0.3 else "own")
        colors, sep = _sample_palette(rng, ncol, sep_mode)
        master = _sample_motif(rng, h, w, colors, symmetry)
        if master is None:
            continue
        positions = [(r, c) for r in range(R) for c in range(C)]
        pos = spec.get("pos")
        if callable(pos):
            pos = pos(R, C, rng)
        if pos is None or pos not in positions:
            pos = positions[int(rng.integers(len(positions)))]
        kind = spec.get("defect") or str(rng.choice(_DEFECTS, p=[0.3, 0.3, 0.2, 0.2]))
        tiles = {p: _expected(master, *p) for p in positions}
        if kind == "copy":  # forgot to flip: literal copy of a seam neighbour
            nbs = [(pos[0] + dr, pos[1] + dc) for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0))
                   if (pos[0] + dr, pos[1] + dc) in tiles]
            nb = nbs[int(rng.integers(len(nbs)))]
            defect = tiles[nb].copy()
        else:
            defect = _make_defect(rng, tiles[pos], kind, colors)
        if defect is None or np.array_equal(defect, tiles[pos]):
            continue
        tiles[pos] = defect
        if sep_mode == "own" and np.any(defect == sep):
            continue
        g = _assemble(tiles, R, C, h, w, sr, sc, sep)
        try:
            rows, cols, s = parse(g)
        except ValueError:
            continue
        if len(rows) != R or len(cols) != C or s != sep:
            continue
        try:
            out = solve(g)
        except ValueError:
            continue
        if not np.array_equal(out, defect):
            continue
        check = spec.get("check")
        if check is not None and not check(g, pos):
            continue
        return {"input": g, "output": out}, {"layout": (R, C), "pos": pos, "defect": kind,
                                             "symmetry": symmetry, "tile": (h, w)}
    return None


def _literal_wrong(g, pos):
    p = _shortcut_literal(g)
    return p is not None and p != pos


def _orientation_matters(g, pos):
    if pos == (0, 0):
        return False
    rows, cols, _ = parse(g)
    t = _tiles(g, rows, cols)[pos]
    return not np.array_equal(_normalise(t, *pos), t)


def _non_master_mixed(g, pos):
    rows, cols, _ = parse(g)
    return len(rows) >= 2 and len(cols) >= 2 and _orientation_matters(g, pos)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _attempt in range(60):
        quiet = rng.random() < 0.25
        quiet_layout = (2, 2)
        train, metas, seen = [], [], set()

        def add(spec, bucket, meta_bucket):
            for _ in range(30):
                res = _sample_pair(rng, spec)
                if res is None:
                    continue
                pair, meta = res
                key = _text(pair["input"])
                if key in seen or key in _PARENT_INPUTS:
                    continue
                seen.add(key)
                bucket.append(pair)
                meta_bucket.append(meta)
                return True
            return False

        layout_q = quiet_layout if quiet else None
        # 1: a mixed (both seam directions) layout whose odd tile is the top-left
        #    master and whose defect keeps colour counts -> majority vote matters.
        specs = [
            {"layout": layout_q or [(2, 2), (2, 3), (3, 2), (3, 3)][int(rng.integers(4))],
             "pos": (0, 0), "defect": str(rng.choice(["orient", "swap"])), "symmetry": "none"},
            # 2: "forgot to flip" -- literal odd-one-out names a wrong tile.
            {"layout": layout_q, "defect": "copy", "check": _literal_wrong,
             "symmetry": None if not quiet else str(rng.choice(["h", "v"]))},
        ]
        # 3: odd tile in a flipped position of a 2-D layout (output keeps the
        #    flipped orientation; partial-axis shortcuts fail).
        specs.append({"layout": layout_q or [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2)][int(rng.integers(6))],
                      "symmetry": "none", "check": _non_master_mixed,
                      "defect": str(rng.choice(["swap", "toggle", "multi", "orient"]))})
        while len(specs) < num_train:
            specs.append({"layout": layout_q if quiet and rng.random() < 0.8 else None})
        order = list(rng.permutation(num_train))
        specs = [specs[i] for i in order] if num_train >= 3 else specs[:num_train]
        ok = all(add(s, train, metas) for s in specs)
        if not ok:
            continue
        train_layouts = {m["layout"] for m in metas}
        test, tmetas = [], []
        for k in range(num_test):
            spec = {}
            u = rng.random()
            if u < 0.5:  # unseen layout when possible
                fresh = [l for l in _LAYOUTS if l not in train_layouts]
                spec["layout"] = fresh[int(rng.integers(len(fresh)))]
            elif u < 0.75:
                ls = sorted(train_layouts)
                spec["layout"] = ls[int(rng.integers(len(ls)))]
            if rng.random() < 0.12:
                spec["symmetry"] = "full"
            if rng.random() < 0.3:  # interior / far tiles
                spec["pos"] = lambda R, C, r_: (int(r_.integers(R)), C - 1) if r_.random() < 0.5 else (R - 1, int(r_.integers(C)))
            if not add(spec, test, tmetas) and not add({}, test, tmetas):
                break
        if len(test) != num_test:
            continue
        puzzle = {"train": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in train],
                  "test": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in test]}
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle
    raise RuntimeError(f"could not generate puzzle for seed {seed}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    ntr, nte = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntr <= 10 and 1 <= nte <= 4 and ntr > nte
    seen = set()
    info = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert set(pair) == {"input", "output"}
            g = np.asarray(pair["input"])
            o = np.asarray(pair["output"])
            for a in (g, o):
                assert a.ndim == 2 and 1 <= a.shape[0] <= 30 and 1 <= a.shape[1] <= 30
                assert np.issubdtype(a.dtype, np.integer) and a.min() >= 0 and a.max() <= 9
            key = _text(g)
            assert key not in seen, "duplicate input"
            assert key not in _PARENT_INPUTS, "parent input copied"
            seen.add(key)
            rows, cols, sep = parse(g)
            tiles = _tiles(g, rows, cols)
            # parse() treats uniform full lines as separators: equal tiles, N >= 3
            pos, _ = find_odd(g)
            assert np.array_equal(o, tiles[pos]), "output must be the odd tile, unchanged"
            norm = {p: _normalise(t, *p) for p, t in tiles.items()}
            ref = [p for p in tiles if p != pos][0]
            for p in tiles:
                if p != pos:
                    assert np.array_equal(norm[p], norm[ref])
            info.append((split, g, pos, len(rows), len(cols)))
    tr = [x for x in info if x[0] == "train"]
    # Demonstrations must defeat the simpler alternatives.
    assert any(_shortcut_literal(g) not in (None, p) for _, g, p, _, _ in tr), "literal shortcut never wrong"
    assert any(_shortcut_literal(g) != p for _, g, p, _, _ in tr)
    assert any(_shortcut_histogram(g) != p for _, g, p, _, _ in tr), "histogram shortcut not refuted"
    assert any(_shortcut_partial(g, "v") != p for _, g, p, _, _ in tr), "rows-only mirror not refuted"
    assert any(_shortcut_partial(g, "h") != p for _, g, p, _, _ in tr), "cols-only mirror not refuted"
    assert any(_shortcut_master(g) != p for _, g, p, _, _ in tr), "master-reference shortcut not refuted"
    assert any(_orientation_matters(g, p) for _, g, p, _, _ in tr), "output orientation never matters"
    assert len({p for _, _, p, _, _ in tr}) >= 2, "odd position constant in training"
    assert any(R >= 2 and C >= 2 for _, _, _, R, C in tr)
    return True


def self_test():
    sigs, layouts, shapes, positions = set(), set(), set(), set()
    for seed in range(42):
        pz = generate_puzzle(seed)
        validate_generated_puzzle(pz)
        assert pz == generate_puzzle(seed)
        sigs.add(json.dumps(pz, separators=(",", ":")))
        for split in ("train", "test"):
            for pair in pz[split]:
                g = np.asarray(pair["input"])
                assert np.array_equal(solve(g), np.asarray(pair["output"]))
                rows, cols, _ = parse(g)
                layouts.add((len(rows), len(cols)))
                shapes.add(g.shape)
                positions.add(find_odd(g)[0])
    assert len(sigs) == 42
    assert len(layouts) >= 8, layouts
    assert len(shapes) >= 30
    assert len(positions) >= 9
    for a, b in ((2, 1), (10, 4), (5, 4)):
        validate_generated_puzzle(generate_puzzle(7, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (3, 3), (6, 5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return True


if __name__ == "__main__":
    self_test()
    print("ok")
