"""Corner-keyed lattice quadrants.

Each lattice block (a solid rectangle of compartment color crossed by complete
one-cell divider rows/columns) carries four singleton corner markers placed
diagonally outside its corners.  The compartments of the block are split into
four quadrants *by ordinal rank* along each axis: the first half of the row
bands versus the last half, and likewise for column bands.  When an axis has an
odd number of bands, its unique middle band belongs to neither half and stays
unchanged (a neutral cross).  Every compartment in a quadrant is filled with the
colour of the marker at the matching outer corner; dividers and neutral
compartments are preserved and the four markers are erased to background.

Parent A (941d9a10) contributes divider lattices, compartment indexing and the
first/middle/last ordinal structure (the middle band).  Parent B (e9ac8c9e)
contributes the diagonal corner-marker encoding, the four-quadrant transfer,
marker removal and independent multi-object processing.
"""
from __future__ import annotations

import numpy as np

MAX_SIDE = 30


# ----------------------------------------------------------------------------
# Parsing / solving (input only)
# ----------------------------------------------------------------------------

def _border_background(grid):
    border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
    values, counts = np.unique(border, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask):
    """4-connected components of a boolean mask; returns list of cell lists."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                comps.append(cells)
    return comps


def _bands_from_flags(flags):
    """Maximal runs of False in a boolean divider-flag list -> [(start, end)]."""
    bands, start = [], None
    for i, f in enumerate(list(flags) + [True]):
        if not f and start is None:
            start = i
        elif f and start is not None:
            bands.append((start, i - 1))
            start = None
    return bands


def _half(index, count):
    """0 = first half, 1 = last half, None = the neutral middle band."""
    if count % 2 == 1 and index == count // 2:
        return None
    return 0 if index < count / 2 else 1


def _parse(grid):
    """Return (background, list of block descriptions)."""
    grid = np.asarray(grid, dtype=int)
    bg = _border_background(grid)
    blocks = []
    for cells in _components(grid != bg):
        if len(cells) == 1:
            continue  # marker candidate
        rows = [y for y, _ in cells]
        cols = [x for _, x in cells]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
            raise ValueError("non-rectangular object")
        sub = grid[r0:r1 + 1, c0:c1 + 1]
        row_div = [bool(np.all(sub[i] == sub[i, 0])) for i in range(sub.shape[0])]
        col_div = [bool(np.all(sub[:, j] == sub[0, j])) for j in range(sub.shape[1])]
        div_colors = {int(sub[i, 0]) for i, f in enumerate(row_div) if f}
        div_colors |= {int(sub[0, j]) for j, f in enumerate(col_div) if f}
        if len(div_colors) != 1:
            raise ValueError("lattice needs one divider colour on both axes")
        divider = div_colors.pop()
        row_bands = _bands_from_flags(row_div)
        col_bands = _bands_from_flags(col_div)
        comp_vals = {int(v) for v in np.unique(sub) if int(v) != divider}
        if len(comp_vals) != 1:
            raise ValueError("lattice needs one compartment colour")
        corners = {}
        for key, (y, x) in {"tl": (r0 - 1, c0 - 1), "tr": (r0 - 1, c1 + 1),
                            "bl": (r1 + 1, c0 - 1), "br": (r1 + 1, c1 + 1)}.items():
            if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]) or grid[y, x] == bg:
                raise ValueError("missing corner marker")
            corners[key] = (y, x, int(grid[y, x]))
        blocks.append({"box": (r0, r1, c0, c1), "divider": divider,
                       "compartment": comp_vals.pop(), "row_bands": row_bands,
                       "col_bands": col_bands, "row_div": row_div, "col_div": col_div,
                       "corners": corners})
    return bg, blocks


def solve(input_grid):
    grid = np.array(input_grid, dtype=int)
    bg, blocks = _parse(grid)
    out = grid.copy()
    keys = (("tl", "tr"), ("bl", "br"))
    for b in blocks:
        r0, _, c0, _ = b["box"]
        for y, x, _ in b["corners"].values():
            out[y, x] = bg
        nr, nc = len(b["row_bands"]), len(b["col_bands"])
        for i, (ra, rb) in enumerate(b["row_bands"]):
            hr = _half(i, nr)
            if hr is None:
                continue
            for j, (ca, cb) in enumerate(b["col_bands"]):
                hc = _half(j, nc)
                if hc is None:
                    continue
                color = b["corners"][keys[hr][hc]][2]
                out[r0 + ra:r0 + rb + 1, c0 + ca:c0 + cb + 1] = color
    return out


# ----------------------------------------------------------------------------
# Shortcut solvers used to prove the demonstrations are informative
# ----------------------------------------------------------------------------

def _shortcut_output(grid, mode):
    grid = np.array(grid, dtype=int)
    bg, blocks = _parse(grid)
    out = grid.copy()
    keys = (("tl", "tr"), ("bl", "br"))
    for b in blocks:
        r0, r1, c0, c1 = b["box"]
        for y, x, _ in b["corners"].values():
            out[y, x] = bg
        H, W = r1 - r0 + 1, c1 - c0 + 1
        nr, nc = len(b["row_bands"]), len(b["col_bands"])

        def geo(a, z, length):
            centre = (length - 1) / 2.0
            if z < centre:
                return 0
            if a > centre:
                return 1
            return None

        for i, (ra, rb) in enumerate(b["row_bands"]):
            for j, (ca, cb) in enumerate(b["col_bands"]):
                if mode == "geometric":
                    hr, hc = geo(ra, rb, H), geo(ca, cb, W)
                elif mode == "diagonal":
                    # Only the corner-most quadrant compartments (whole-diagonal style).
                    hr = 0 if i == 0 else (1 if i == nr - 1 else None)
                    hc = 0 if j == 0 else (1 if j == nc - 1 else None)
                elif mode == "no_cross":
                    # every band belongs to a half; an odd middle band joins the first half
                    hr = 0 if i <= (nr - 1) // 2 else 1
                    hc = 0 if j <= (nc - 1) // 2 else 1
                else:
                    raise ValueError(mode)
                if hr is None or hc is None:
                    continue
                out[r0 + ra:r0 + rb + 1, c0 + ca:c0 + cb + 1] = b["corners"][keys[hr][hc]][2]
    return out


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

def _band_sizes(rng, n, style, max_len):
    """Sizes of n bands (each >= 1) whose lattice length sum+n-1 <= max_len."""
    budget = max_len - (n - 1)
    if budget < n:
        return None
    for _ in range(40):
        if style == "regular":
            s = int(rng.integers(1, min(4, budget // n) + 1))
            sizes = [s] * n
        elif style == "skew":
            big = int(rng.integers(2, 5))
            sizes = [int(rng.integers(1, 3)) for _ in range(n)]
            side = rng.integers(0, 2)
            for k in range(n // 2):
                idx = k if side == 0 else n - 1 - k
                sizes[idx] = big + int(rng.integers(0, 2))
        else:  # uneven
            sizes = [int(rng.integers(1, 5)) for _ in range(n)]
        if sum(sizes) <= budget:
            return sizes
    return [1] * n


def _render_block(row_sizes, col_sizes, comp, div):
    h = sum(row_sizes) + len(row_sizes) - 1
    w = sum(col_sizes) + len(col_sizes) - 1
    block = np.full((h, w), comp, dtype=int)
    pos = 0
    for s in row_sizes[:-1]:
        pos += s
        block[pos, :] = div
        pos += 1
    pos = 0
    for s in col_sizes[:-1]:
        pos += s
        block[:, pos] = div
        pos += 1
    return block


def _quartet(rng, pool, allow_repeat):
    pool = list(pool)
    if allow_repeat and rng.random() < 0.5 and len(pool) >= 2:
        k = int(rng.integers(1, 4))  # 1..3 distinct values
        chosen = list(rng.choice(pool, size=k, replace=False))
        vals = chosen + [chosen[int(rng.integers(0, k))] for _ in range(4 - k)]
        rng.shuffle(vals)
        return [int(v) for v in vals]
    return [int(v) for v in rng.choice(pool, size=4, replace=False)]


def _place(rng, footprints):
    """Place footprints (h+2, w+2 incl. marker ring); return (H, W, origins) or None."""
    need_h = max(f[0] for f in footprints)
    need_w = max(f[1] for f in footprints)
    area = sum((f[0] + 1) * (f[1] + 1) for f in footprints)
    for _ in range(30):
        H = int(rng.integers(need_h, MAX_SIDE + 1))
        W = int(rng.integers(need_w, MAX_SIDE + 1))
        if H * W < area * 1.2:
            continue
        if len(footprints) == 1 and rng.random() < 0.6:
            H = int(min(MAX_SIDE, need_h + rng.integers(0, 6)))
            W = int(min(MAX_SIDE, need_w + rng.integers(0, 6)))
        occ = np.zeros((H, W), dtype=bool)
        origins = []
        ok = True
        for fh, fw in footprints:
            placed = False
            for _ in range(80):
                y = int(rng.integers(0, H - fh + 1))
                x = int(rng.integers(0, W - fw + 1))
                y0, y1 = max(0, y - 1), min(H, y + fh + 1)
                x0, x1 = max(0, x - 1), min(W, x + fw + 1)
                if occ[y0:y1, x0:x1].any():
                    continue
                occ[y:y + fh, x:x + fw] = True
                origins.append((y, x))
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return H, W, origins
    return None


def _counts_for(rng, parity, lo, hi):
    if parity == "odd":
        opts = [n for n in (3, 5, 7, 9) if lo <= n <= hi]
    else:
        opts = [n for n in (2, 3, 4, 5, 6, 7) if lo <= n <= hi]
    return int(rng.choice(opts))


def _make_pair(rng, specs, palette, allow_repeat):
    """specs: list of (nr, nc, row_style, col_style).  Returns pair dict + meta."""
    bg, comp, div = palette
    pool = [c for c in range(10) if c not in (bg, comp, div)]
    k = len(specs)
    max_len = {1: 24, 2: 16, 3: 12, 4: 11}.get(k, 11)
    blocks = []
    for nr, nc, rs, cs in specs:
        rows = _band_sizes(rng, nr, rs, max_len)
        cols = _band_sizes(rng, nc, cs, max_len)
        if rows is None or cols is None:
            return None
        blocks.append((rows, cols))
    fps = [(sum(r) + len(r) - 1 + 2, sum(c) + len(c) - 1 + 2) for r, c in blocks]
    placed = _place(rng, fps)
    if placed is None:
        return None
    H, W, origins = placed
    grid = np.full((H, W), bg, dtype=int)
    quartets = []
    for (rows, cols), (y, x) in zip(blocks, origins):
        blk = _render_block(rows, cols, comp, div)
        h, w = blk.shape
        grid[y + 1:y + 1 + h, x + 1:x + 1 + w] = blk
        q = _quartet(rng, pool, allow_repeat)
        quartets.append(q)
        grid[y, x], grid[y, x + w + 1], grid[y + h + 1, x], grid[y + h + 1, x + w + 1] = q
    # background must dominate the border for an unambiguous reading
    border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
    if np.sum(border == bg) * 2 <= border.size:
        return None
    return {"input": grid, "output": solve(grid)}, quartets


def _palette(rng, fixed=None):
    if fixed is not None:
        return fixed
    bg = 0 if rng.random() < 0.5 else int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    comp, div = (int(v) for v in rng.choice(others, size=2, replace=False))
    return bg, comp, div


SCHEMES = ("objects", "bands", "combo", "palette", "free")


def _pair_meta(pair):
    bg, blocks = _parse(pair["input"])
    axes = []
    for b in blocks:
        axes.extend([len(b["row_bands"]), len(b["col_bands"])])
    return bg, blocks, axes


def _plan(rng, scheme, parity, num_train, num_test):
    """Return lists of block-count specs for train and test pairs."""
    styles = ("regular", "uneven", "skew")

    def st():
        return str(rng.choice(styles, p=[0.25, 0.4, 0.35]))

    def spec(lo, hi):
        return (_counts_for(rng, parity, lo, hi), _counts_for(rng, parity, lo, hi), st(), st())

    train, test = [], []
    if scheme == "objects":
        for i in range(num_train):
            k = 2 if (i == num_train - 1 and rng.random() < 0.4) else 1
            train.append([spec(2, 7) for _ in range(k)])
        for _ in range(num_test):
            k = int(rng.integers(2, 4))
            test.append([spec(2, 5) for _ in range(k)])
    elif scheme == "bands":
        for _ in range(num_train):
            k = 1 if rng.random() < 0.7 else 2
            train.append([spec(2, 5) for _ in range(k)])
        for _ in range(num_test):
            nr, nc, a, b = spec(2, 5)
            big = 7 if parity == "odd" else int(rng.choice([6, 7]))
            if rng.random() < 0.5:
                nr = big
            else:
                nc = big
            if rng.random() < 0.3:
                nr = nc = big
            test.append([(nr, nc, a, b)])
    elif scheme == "combo":
        for _ in range(num_train):
            k = 1 if rng.random() < 0.75 else 2
            blocks = []
            for _ in range(k):
                nr, nc, a, b = spec(2, 7)
                blocks.append((nr, nc, a, b))
            train.append(blocks)
        for _ in range(num_test):
            k = 1 if rng.random() < 0.6 else 2
            test.append([spec(2, 7) for _ in range(k)])
    elif scheme == "palette":
        for _ in range(num_train):
            k = int(rng.integers(1, 3))
            train.append([spec(2, 7) for _ in range(k)])
        for _ in range(num_test):
            k = int(rng.integers(1, 3))
            test.append([spec(2, 7) for _ in range(k)])
    else:  # free
        for _ in range(num_train):
            k = int(rng.integers(1, 4))
            train.append([spec(2, 7) for _ in range(k)])
        for _ in range(num_test):
            k = int(rng.integers(1, 4))
            test.append([spec(2, 7) for _ in range(k)])
    return train, test


def _requirements(train_pairs, test_pairs, scheme):
    """Training must teach every convention the tests rely on."""
    tr_axes, tr_blocks = [], []
    for p in train_pairs:
        _, blocks, axes = _pair_meta(p)
        tr_axes += axes
        tr_blocks += blocks
    te_axes = []
    te_counts = []
    for p in test_pairs:
        _, blocks, axes = _pair_meta(p)
        te_axes += axes
        te_counts += [(len(b["row_bands"]), len(b["col_bands"])) for b in blocks]
    # parity coverage
    if any(a % 2 == 1 for a in te_axes) and not any(a % 2 == 1 for a in tr_axes):
        return "odd axis not taught"
    if any(a % 2 == 0 for a in te_axes) and not any(a % 2 == 0 for a in tr_axes):
        return "even axis not taught"
    if not any(a % 2 == 1 for a in tr_axes):
        return "no neutral cross in training"
    # some quadrant holds >= 2 compartments along an axis (rules out corner-only)
    if not any(a >= 4 for a in tr_axes):
        return "no multi-compartment quadrant in training"
    for mode in ("geometric", "diagonal", "no_cross"):
        if not any(not np.array_equal(_shortcut_output(p["input"], mode), p["output"])
                   for p in train_pairs):
            return f"shortcut {mode} not refuted"
    # some training quartet has four distinct colours
    if not any(len({c[2] for c in b["corners"].values()}) == 4 for b in tr_blocks):
        return "no four-colour quartet in training"
    # repeated test quartets only after a demonstration
    def has_rep(pairs):
        return any(len({c[2] for c in b["corners"].values()}) < 4
                   for p in pairs for b in _pair_meta(p)[1])
    if has_rep(test_pairs) and not has_rep(train_pairs):
        return "repeated quartet not taught"
    if scheme == "bands" and max(te_axes) <= max(tr_axes):
        return "bands shift missing"
    if scheme == "objects":
        tr_k = max(len(_pair_meta(p)[1]) for p in train_pairs)
        te_k = min(len(_pair_meta(p)[1]) for p in test_pairs)
        if te_k <= tr_k:
            return "objects shift missing"
    if scheme == "combo":
        tr_set = {(len(b["row_bands"]), len(b["col_bands"])) for b in tr_blocks}
        if all(c in tr_set for c in te_counts):
            return "combo shift missing"
    return None


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not isinstance(num_train, (int, np.integer)) or not isinstance(num_test, (int, np.integer)):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("illegal train/test counts")
    rng = np.random.default_rng(seed)
    scheme = str(rng.choice(SCHEMES, p=[0.25, 0.2, 0.2, 0.15, 0.2]))
    parity = "odd" if rng.random() < 0.4 else "mixed"
    per_pair_palette = scheme == "palette" or rng.random() < 0.25
    fixed_palette = None if per_pair_palette else _palette(rng)
    allow_repeat = rng.random() < 0.45
    for _attempt in range(400):
        train_plan, test_plan = _plan(rng, scheme, parity, num_train, num_test)
        pairs, seen, ok = [], set(), True
        train_has_rep = False
        for idx, specs in enumerate(train_plan + test_plan):
            is_test = idx >= num_train
            made = None
            for _ in range(25):
                pal = _palette(rng, fixed_palette)
                made = _make_pair(rng, specs, pal,
                                  allow_repeat and (not is_test or train_has_rep))
                if made is None:
                    continue
                key = made[0]["input"].tobytes() + bytes(made[0]["input"].shape)
                if key in seen:
                    made = None
                    continue
                seen.add(key)
                break
            if made is None:
                ok = False
                break
            pair, quartets = made
            if not is_test and any(len(set(q)) < 4 for q in quartets):
                train_has_rep = True
            pairs.append(pair)
        if not ok:
            continue
        train, test = pairs[:num_train], pairs[num_train:]
        if _requirements(train, test, scheme) is not None:
            continue
        if scheme == "palette":
            tr_pal = {tuple(_pal_of(p["input"])) for p in train}
            if all(tuple(_pal_of(p["input"])) in tr_pal for p in test):
                continue
        puzzle = {"train": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in train],
                  "test": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in test]}
        validate_generated_puzzle(puzzle)
        return puzzle
    raise RuntimeError(f"could not generate puzzle for seed {seed}")


def _pal_of(grid):
    bg, blocks = _parse(grid)
    return (bg,) + tuple(sorted({(b["compartment"], b["divider"]) for b in blocks}))


# ----------------------------------------------------------------------------
# Validation and self-test
# ----------------------------------------------------------------------------

def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    norm = []
    for pair in train + test:
        assert set(pair) == {"input", "output"}
        gi = np.asarray(pair["input"], dtype=int)
        go = np.asarray(pair["output"], dtype=int)
        assert gi.ndim == 2 and 1 <= gi.shape[0] <= 30 and 1 <= gi.shape[1] <= 30
        assert gi.shape == go.shape
        assert gi.min() >= 0 and gi.max() <= 9 and go.min() >= 0 and go.max() <= 9
        key = gi.tobytes() + bytes(gi.shape)
        assert key not in seen, "duplicate input"
        seen.add(key)
        bg, blocks = _parse(gi)
        assert blocks, "no lattice block"
        border = np.concatenate([gi[0, :], gi[-1, :], gi[1:-1, 0], gi[1:-1, -1]])
        assert np.sum(border == bg) * 2 > border.size, "background must dominate border"
        claimed = np.zeros(gi.shape, dtype=bool)
        for b in blocks:
            r0, r1, c0, c1 = b["box"]
            nr, nc = len(b["row_bands"]), len(b["col_bands"])
            assert nr >= 2 and nc >= 2, "need at least two bands per axis"
            # dividers: one cell wide, not on the block edge, non-adjacent
            for flags in (b["row_div"], b["col_div"]):
                assert not flags[0] and not flags[-1]
                assert all(not (flags[i] and flags[i + 1]) for i in range(len(flags) - 1))
            marks = {c[2] for c in b["corners"].values()}
            assert not marks & {bg, b["compartment"], b["divider"]}, "marker colour clash"
            # markers are 4-isolated
            for y, x, _ in b["corners"].values():
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < gi.shape[0] and 0 <= nx < gi.shape[1]:
                        assert gi[ny, nx] == bg, "marker not isolated"
            y0, y1, x0, x1 = r0 - 1, r1 + 1, c0 - 1, c1 + 1
            assert not claimed[y0:y1 + 1, x0:x1 + 1].any(), "overlapping footprints"
            claimed[max(0, y0 - 1):y1 + 2, max(0, x0 - 1):x1 + 2] = True
            # output: markers erased, dividers kept, cross kept
            for y, x, _ in b["corners"].values():
                assert go[y, x] == bg
            sub_o = go[r0:r1 + 1, c0:c1 + 1]
            for i, f in enumerate(b["row_div"]):
                if f:
                    assert np.all(sub_o[i] == b["divider"])
            for j, f in enumerate(b["col_div"]):
                if f:
                    assert np.all(sub_o[:, j] == b["divider"])
        # every non-background cell belongs to a block or is one of its markers
        expected = np.zeros(gi.shape, dtype=bool)
        for b in blocks:
            r0, r1, c0, c1 = b["box"]
            expected[r0:r1 + 1, c0:c1 + 1] = True
            for y, x, _ in b["corners"].values():
                expected[y, x] = True
        assert np.array_equal(expected, gi != bg), "stray cells"
        assert np.array_equal(solve(gi), go), "output disagrees with rule"
        assert not np.array_equal(gi, go)
        norm.append(go)
    tr = [{"input": np.asarray(p["input"]), "output": np.asarray(p["output"])} for p in train]
    te = [{"input": np.asarray(p["input"]), "output": np.asarray(p["output"])} for p in test]
    problem = _requirements(tr, te, None)
    assert problem is None, problem
    return True


def self_test():
    shapes, counts, schemes_seen, objs = set(), set(), set(), set()
    geo_diff = 0
    for seed in range(42):
        pz = generate_puzzle(seed, 4, 1)
        validate_generated_puzzle(pz)
        for pair in pz["train"] + pz["test"]:
            gi = np.asarray(pair["input"])
            shapes.add(gi.shape)
            _, blocks = _parse(gi)
            objs.add(len(blocks))
            for b in blocks:
                counts.add((len(b["row_bands"]), len(b["col_bands"])))
            if not np.array_equal(_shortcut_output(gi, "geometric"), np.asarray(pair["output"])):
                geo_diff += 1
        assert generate_puzzle(seed, 4, 1) == pz
    assert len(shapes) >= 40, shapes
    assert len(counts) >= 15, counts
    assert {1, 2, 3} <= objs, objs
    assert geo_diff >= 30
    for a, b in ((2, 1), (10, 4), (5, 3)):
        validate_generated_puzzle(generate_puzzle(7, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (3, 4), (2, 2)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return True


if __name__ == "__main__":
    self_test()
    print("self_test ok")
