"""Odd cavity ledger: find the panel whose enclosed-area code is the odd one out.

The input is a grid of equal panels separated by full divider lines.  Every panel
holds two differently coloured hollow contours (left and right).  Each panel is
summarised by parent A's enclosed-area code: the left contour's enclosed
background count followed by the right contour's count.  All panels except one
share the same ordered code although their contour shapes, thicknesses and
colours differ; parent B's odd-one-out selection picks the single panel whose
code differs.  The output is that panel's code written as A's 3x3 unary ledger
(three columns, left contour first, right contour on a fresh row) in that panel's
literal colours on the shared background.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

# ----------------------------------------------------------------------------
# Generic grid helpers
# ----------------------------------------------------------------------------

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                    for dy, dx in _DIRS:
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                            seen[yy, xx] = True
                            stack.append((yy, xx))
                comps.append(cells)
    return comps


def _enclosed(free):
    """Components of `free` cells not 4-connected to the array border."""
    h, w = free.shape
    return [cells for cells in _components(free)
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)]


def _connected(mask):
    return len(_components(mask)) == 1


# ----------------------------------------------------------------------------
# Unary ledger (parent A)
# ----------------------------------------------------------------------------

def _rows_needed(n):
    return (n + 2) // 3


def _code_ok(code):
    a, b = code
    return 1 <= a <= 6 and 1 <= b <= 6 and _rows_needed(a) + _rows_needed(b) <= 3


ALL_CODES = tuple((a, b) for a in range(1, 7) for b in range(1, 7) if _code_ok((a, b)))


def _ledger(code, colors, bg):
    out = np.full((3, 3), bg, dtype=int)
    row = 0
    for n, color in zip(code, colors):
        for i in range(n):
            out[row + i // 3, i % 3] = color
        row += _rows_needed(n)
    return out


# ----------------------------------------------------------------------------
# Input analysis (shared by solve and the validator)
# ----------------------------------------------------------------------------

def _find_divider(g):
    h, w = g.shape
    found = []
    for color in np.unique(g):
        color = int(color)
        rows = [r for r in range(h) if np.all(g[r] == color)]
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if not rows and not cols:
            continue
        mask = np.zeros_like(g, dtype=bool)
        mask[rows, :] = True
        mask[:, cols] = True
        if np.array_equal(mask, g == color):
            found.append((color, rows, cols))
    if len(found) != 1:
        raise ValueError("Divider colour is not unique")
    return found[0]


def _segments(lines, n):
    segs, start = [], 0
    for x in sorted(lines) + [n]:
        if x > start:
            segs.append((start, x))
        elif x < start:
            raise ValueError("bad divider")
        start = x + 1
    return segs


def _analyse(grid):
    g = np.asarray(grid, dtype=int)
    divider, rows, cols = _find_divider(g)
    rsegs, csegs = _segments(rows, g.shape[0]), _segments(cols, g.shape[1])
    panels = [(r0, r1, c0, c1) for r0, r1 in rsegs for c0, c1 in csegs]
    if len(panels) < 3:
        raise ValueError("Need at least three panels")
    border = []
    for r0, r1, c0, c1 in panels:
        p = g[r0:r1, c0:c1]
        border.extend(p[0].tolist() + p[-1].tolist() + p[:, 0].tolist() + p[:, -1].tolist())
    bg = Counter(border).most_common(1)[0][0]
    infos = []
    for box in panels:
        r0, r1, c0, c1 = box
        p = g[r0:r1, c0:c1]
        colors = sorted(set(int(v) for v in np.unique(p)) - {bg})
        if len(colors) != 2:
            raise ValueError("Each panel needs exactly two contour colours")
        counts = {c: 0 for c in colors}
        ph, pw = p.shape
        for cells in _enclosed(p == bg):
            adj = set()
            for y, x in cells:
                for dy, dx in _DIRS:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < ph and 0 <= xx < pw and p[yy, xx] != bg:
                        adj.add(int(p[yy, xx]))
            if len(adj) != 1:
                raise ValueError("Cavity bounded by more than one contour")
            counts[adj.pop()] += len(cells)
        spans = {c: (int(np.where(p == c)[1].min()), int(np.where(p == c)[1].max())) for c in colors}
        ordered = sorted(colors, key=lambda c: spans[c][0])
        infos.append({
            "box": box,
            "colors": tuple(ordered),
            "code": tuple(counts[c] for c in ordered),
            "spans": tuple(spans[c] for c in ordered),
            "mask": (p != bg),
            "fg": int(np.sum(p != bg)),
        })
    return {"divider": divider, "bg": int(bg), "panels": infos,
            "shape": (len(rsegs), len(csegs))}


def _odd_index(codes):
    tally = Counter(codes)
    singles = [i for i, c in enumerate(codes) if tally[c] == 1]
    if len(tally) != 2 or len(singles) != 1:
        raise ValueError("No unique odd panel")
    return singles[0]


def solve(input_grid):
    info = _analyse(input_grid)
    panels = info["panels"]
    k = _odd_index([p["code"] for p in panels])
    odd = panels[k]
    return _ledger(odd["code"], odd["colors"], info["bg"])


# ----------------------------------------------------------------------------
# Shape bank
# ----------------------------------------------------------------------------

def _norm(cells):
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return tuple(sorted((r - mr, c - mc) for r, c in cells))


def _make_bank():
    bank = {1: [((0, 0),)]}
    for n in range(2, 7):
        shapes = set()
        for shape in bank[n - 1]:
            occ = set(shape)
            for r, c in shape:
                for dr, dc in _DIRS:
                    cell = (r + dr, c + dc)
                    if cell not in occ:
                        shapes.add(_norm(list(occ | {cell})))
        bank[n] = sorted(shapes)
    return bank


_BANK = _make_bank()


def _dims(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


# ----------------------------------------------------------------------------
# Panel construction
# ----------------------------------------------------------------------------

def _thin_contour(shape):
    """Return (wall, cavity) boolean arrays with a one-cell 8-neighbour shell."""
    h, w = _dims(shape)
    wall = np.zeros((h + 2, w + 2), dtype=bool)
    cav = np.zeros_like(wall)
    for r, c in shape:
        cav[r + 1, c + 1] = True
    for r, c in shape:
        wall[r:r + 3, c:c + 3] = True
    wall &= ~cav
    return wall, cav


def _feasible_shapes(n, maxh, maxw):
    return [s for s in _BANK[n] if _dims(s)[0] + 2 <= maxh and _dims(s)[1] + 2 <= maxw]


def _code_feasible(code, ph, pw):
    ih, iw = ph - 2, pw - 2
    a, b = code
    wa = [(_dims(s)[1] + 2) for s in _feasible_shapes(a, ih, iw)]
    wb = [(_dims(s)[1] + 2) for s in _feasible_shapes(b, ih, iw)]
    return bool(wa) and bool(wb) and min(wa) + min(wb) <= iw


def _panel_ok(lab, cavs):
    """lab: 0 bg, 1 left wall, 2 right wall.  cavs: list of two cell sets."""
    found = _enclosed(lab == 0)
    if len(found) != 2:
        return False
    got = sorted(frozenset(c) for c in found)
    want = sorted(frozenset(c) for c in cavs)
    if got != want:
        return False
    for k in (1, 2):
        if not _connected(lab == k):
            return False
    c1 = np.where(lab == 1)[1]
    c2 = np.where(lab == 2)[1]
    return c1.max() < c2.min()


def _decorate(lab, cavs, rng, style):
    ph, pw = lab.shape
    cav_all = set().union(*cavs)
    if style == "plain":
        return lab
    steps = int(rng.integers(1, 7)) if style != "thick" else 1
    for _ in range(steps):
        k = int(rng.integers(1, 3))
        trial = lab.copy()
        if style == "thick":
            for kk in ((1, 2) if rng.random() < 0.5 else (k,)):
                cand = []
                mask = trial == kk
                grown = mask.copy()
                grown[1:, :] |= mask[:-1, :]
                grown[:-1, :] |= mask[1:, :]
                grown[:, 1:] |= mask[:, :-1]
                grown[:, :-1] |= mask[:, 1:]
                grown[1:, 1:] |= mask[:-1, :-1]
                grown[:-1, :-1] |= mask[1:, 1:]
                grown[1:, :-1] |= mask[:-1, 1:]
                grown[:-1, 1:] |= mask[1:, :-1]
                for r, c in zip(*np.where(grown & (trial == 0))):
                    if 1 <= r < ph - 1 and 1 <= c < pw - 1 and (r, c) not in cav_all:
                        if rng.random() < 0.8:
                            cand.append((r, c))
                for r, c in cand:
                    trial[r, c] = kk
                if not _panel_ok(trial, cavs):
                    trial = lab.copy()
        elif style == "bulge":
            mask = trial == k
            cand = []
            for r in range(1, ph - 1):
                for c in range(1, pw - 1):
                    if trial[r, c] == 0 and (r, c) not in cav_all and any(
                            mask[r + dy, c + dx] for dy, dx in _DIRS):
                        cand.append((r, c))
            if cand:
                r, c = cand[int(rng.integers(len(cand)))]
                trial[r, c] = k
        elif style == "trim":
            cand = []
            for r, c in zip(*np.where(trial == k)):
                if not any((r + dy, c + dx) in cav_all for dy, dx in _DIRS):
                    cand.append((r, c))
            if cand:
                r, c = cand[int(rng.integers(len(cand)))]
                trial[r, c] = 0
        if _panel_ok(trial, cavs):
            lab = trial
    return lab


def _build_panel(code, ph, pw, rng, style):
    ih, iw = ph - 2, pw - 2
    for _ in range(60):
        sa = _feasible_shapes(code[0], ih, iw - 3)
        if not sa:
            return None
        s1 = sa[int(rng.integers(len(sa)))]
        w1 = _dims(s1)[1] + 2
        sb = _feasible_shapes(code[1], ih, iw - w1)
        if not sb:
            continue
        s2 = sb[int(rng.integers(len(sb)))]
        wall1, cav1 = _thin_contour(s1)
        wall2, cav2 = _thin_contour(s2)
        h1, w1 = wall1.shape
        h2, w2 = wall2.shape
        slack = iw - w1 - w2
        gap = int(rng.integers(0, slack + 1))
        left_off = int(rng.integers(0, slack - gap + 1))
        x1 = 1 + left_off
        x2 = x1 + w1 + gap
        y1 = 1 + int(rng.integers(0, ih - h1 + 1))
        y2 = 1 + int(rng.integers(0, ih - h2 + 1))
        lab = np.zeros((ph, pw), dtype=int)
        lab[y1:y1 + h1, x1:x1 + w1][wall1] = 1
        lab[y2:y2 + h2, x2:x2 + w2][wall2] = 2
        cavs = [set((int(r) + y1, int(c) + x1) for r, c in zip(*np.where(cav1))),
                set((int(r) + y2, int(c) + x2) for r, c in zip(*np.where(cav2)))]
        if not _panel_ok(lab, cavs):
            continue
        return _decorate(lab, cavs, rng, style)
    return None


# ----------------------------------------------------------------------------
# Puzzle sampling
# ----------------------------------------------------------------------------

LAYOUTS = ("H3", "V3", "V4", "V5", "G22", "G23", "G32", "G33")


def _layout_grid(name):
    if name[0] == "H":
        return 1, 3
    if name[0] == "V":
        return int(name[1]), 1
    return int(name[1]), int(name[2])


def _dim_ranges(name):
    r, c = _layout_grid(name)
    hmax = min((30 - (r - 1)) // r, 10)
    wmax = min((30 - (c - 1)) // c, 15)
    return (5, hmax), (8, wmax)


def _relation(m, o):
    if o == m:
        return "same"
    if o == (m[1], m[0]):
        return "swap"
    if o[0] == m[0]:
        return "left_same"
    if o[1] == m[1]:
        return "right_same"
    if sum(o) == sum(m):
        return "same_total"
    return "free"


def _sample_codes(rng, relation, ph, pw, avoid_odd=(), prefer_major=()):
    feas = [c for c in ALL_CODES if _code_feasible(c, ph, pw)]
    options = []
    for m in feas:
        for o in feas:
            if o != m and (relation is None or _relation(m, o) == relation):
                if o not in avoid_odd:
                    options.append((m, o))
    if not options:
        return None
    if relation == "swap":
        # the swap witness must also show a spill and a partial row + fresh row
        options = [(m, o) for m, o in options if max(m) >= 4 and o[0] % 3]
        if not options:
            return None
    if prefer_major:
        pref = [(m, o) for m, o in options if m in prefer_major]
        if pref and rng.random() < 0.6:
            options = pref
    return options[int(rng.integers(len(options)))]


def _make_pair(rng, layout, relation, bg, colour_mode, witness, avoid_odd, prefer_major):
    rows, cols = _layout_grid(layout)
    (hl, hh), (wl, wh) = _dim_ranges(layout)
    for _ in range(40):
        ph = int(rng.integers(hl, hh + 1))
        pw = int(rng.integers(wl, wh + 1))
        codes = _sample_codes(rng, relation, ph, pw, avoid_odd, prefer_major)
        if codes is None:
            continue
        major, odd = codes
        n = rows * cols
        odd_at = int(rng.integers(n))
        style_pool = ("plain", "bulge", "trim", "thick")
        panels = []
        for i in range(n):
            code = odd if i == odd_at else major
            style = style_pool[int(rng.integers(len(style_pool)))]
            lab = _build_panel(code, ph, pw, rng, style)
            if lab is None:
                break
            panels.append(lab)
        if len(panels) != n:
            continue
        masks = [p.astype(bool) for p in panels]
        keys = set(m.tobytes() for m in masks)
        if len(keys) != n:
            continue
        fg = [int(m.sum()) for m in masks]
        tally = Counter(fg)
        if tally[fg[odd_at]] == 1 and len(tally) == 2:
            continue  # wall-cell count would single out the answer
        # colours
        divider = int(rng.choice([c for c in range(10) if c != bg]))
        palette = [c for c in range(10) if c not in (bg, divider)]
        if colour_mode == "uniform":
            pair = tuple(int(x) for x in rng.choice(palette, 2, replace=False))
            pcols = [pair] * n
        elif colour_mode == "pool":
            pool = [int(x) for x in rng.choice(palette, 3, replace=False)]
            pcols = [tuple(int(x) for x in rng.choice(pool, 2, replace=False)) for _ in range(n)]
        else:
            pcols = [tuple(int(x) for x in rng.choice(palette, 2, replace=False)) for _ in range(n)]
        if witness:
            donor = int(rng.choice([i for i in range(n) if i != odd_at]))
            pcols[odd_at] = pcols[donor]
        H = rows * ph + rows - 1
        W = cols * pw + cols - 1
        grid = np.full((H, W), divider, dtype=int)
        for i, lab in enumerate(panels):
            r, c = divmod(i, cols)
            y, x = r * (ph + 1), c * (pw + 1)
            block = np.full((ph, pw), bg, dtype=int)
            block[lab == 1] = pcols[i][0]
            block[lab == 2] = pcols[i][1]
            grid[y:y + ph, x:x + pw] = block
        out = _ledger(odd, pcols[odd_at], bg)
        return {"input": grid, "output": out}, (major, odd)
    raise RuntimeError("could not build pair")


def _check_counts(num_train, num_test):
    for v in (num_train, num_test):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("illegal pair counts")


def _pick_bg(rng):
    return 0 if rng.random() < 0.45 else int(rng.integers(0, 10))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    total = num_train + num_test
    families = {"H": ["H3"], "V": ["V3", "V4", "V5"], "G": ["G22", "G23", "G32", "G33"]}
    mode = ("novel_layout", "larger", "mixed")[int(rng.integers(3))]
    if mode == "novel_layout":
        test_fam = ("H", "V", "G")[int(rng.integers(3))]
        train_pool = [l for f, ls in families.items() if f != test_fam for l in ls]
        test_pool = families[test_fam]
    elif mode == "larger":
        train_pool = ["H3", "V3", "V4", "G22"]
        test_pool = ["V5", "G23", "G32", "G33"]
    else:
        train_pool = list(LAYOUTS)
        test_pool = list(LAYOUTS)
    bg_mode = rng.random() < 0.5  # True: background varies by pair
    shared_bg = _pick_bg(rng)
    # training relation schedule: witnesses first, then shuffled
    if num_train >= 3:
        witnesses = ["swap", "left_same", "right_same"]
        if num_train >= 4 and rng.random() < 0.5:
            witnesses.append("same_total")
    else:
        witnesses = ["swap", ("left_same" if rng.random() < 0.5 else "right_same")]
    rels = witnesses + [None] * (num_train - len(witnesses))
    order = rng.permutation(num_train)
    train_rel = [rels[i] for i in order]
    colour_witness = int(rng.integers(num_train))
    pairs, seen, train_codes = [], set(), []
    for idx in range(total):
        is_test = idx >= num_train
        for attempt in range(30):
            pool = test_pool if is_test else train_pool
            layout = pool[int(rng.integers(len(pool)))]
            relation = None if is_test else train_rel[idx]
            bg = _pick_bg(rng) if bg_mode else shared_bg
            cmode = ("uniform", "pool", "free")[int(rng.integers(3))]
            avoid = ()
            prefer = ()
            if is_test:
                avoid = tuple(o for _, o in train_codes) if attempt < 20 else ()
                prefer = tuple(o for _, o in train_codes)
                avoid = avoid + tuple(o for _, o in train_codes[num_train:])
            try:
                pair, codes = _make_pair(rng, layout, relation, bg, cmode,
                                         (not is_test and idx == colour_witness) or rng.random() < 0.3,
                                         avoid, prefer)
            except RuntimeError:
                continue
            key = pair["input"].tobytes() + bytes(pair["input"].shape)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(pair)
            train_codes.append(codes)
            break
        else:
            raise RuntimeError("failed to build distinct pair")
    return {"train": pairs[:num_train], "test": pairs[num_train:]}


# ----------------------------------------------------------------------------
# Validation and self test
# ----------------------------------------------------------------------------

def _raw_mask_outlier(info):
    keys = [p["mask"].tobytes() for p in info["panels"]]
    tally = Counter(keys)
    singles = [i for i, k in enumerate(keys) if tally[k] == 1]
    return singles[0] if len(tally) == 2 and len(singles) == 1 else None


def _unique_by(values):
    tally = Counter(values)
    singles = [i for i, v in enumerate(values) if tally[v] == 1]
    return singles[0] if len(tally) == 2 and len(singles) == 1 else None


def validate_generated_puzzle(puzzle):
    assert set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    rels, total_fail, multiset_fail, colour_witness = [], False, False, False
    partial_row, spill = False, False
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            g = np.asarray(pair["input"], dtype=int)
            o = np.asarray(pair["output"], dtype=int)
            assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
            assert g.min() >= 0 and g.max() <= 9
            key = (g.shape, g.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _analyse(g)
            panels = info["panels"]
            assert len(panels) >= 3
            assert info["divider"] != info["bg"]
            codes = [p["code"] for p in panels]
            k = _odd_index(codes)
            assert all(_code_ok(c) for c in codes)
            for p in panels:
                (a0, a1), (b0, b1) = p["spans"]
                assert a1 < b0, "contours must be ordered left to right"
                assert p["colors"][0] != p["colors"][1]
                assert info["divider"] not in p["colors"]
                ph, pw = p["mask"].shape
                m = p["mask"]
                assert not m[0].any() and not m[-1].any() and not m[:, 0].any() and not m[:, -1].any()
            # parent-B raw mask comparison must not reveal the answer
            assert len(set(p["mask"].tobytes() for p in panels)) == len(panels)
            assert _raw_mask_outlier(info) is None
            # wall-cell count must not reveal the answer
            assert _unique_by([p["fg"] for p in panels]) != k
            expected = _ledger(panels[k]["code"], panels[k]["colors"], info["bg"])
            assert o.shape == (3, 3) and np.array_equal(o, expected)
            assert np.array_equal(np.asarray(solve(g)), o)
            if split == "train":
                major = codes[(k + 1) % len(codes)]
                odd = codes[k]
                rels.append(_relation(major, odd))
                if sum(odd) == sum(major):
                    total_fail = True
                if sorted(odd) == sorted(major):
                    multiset_fail = True
                if any(panels[i]["colors"] == panels[k]["colors"] for i in range(len(panels)) if i != k):
                    colour_witness = True
                if odd[0] % 3:
                    partial_row = True
                if max(odd) > 3:
                    spill = True
    assert "swap" in rels and total_fail and multiset_fail, "training must defeat total/multiset shortcuts"
    if len(train) >= 3:
        assert "left_same" in rels and "right_same" in rels, "training must defeat single-count shortcuts"
    else:
        assert "left_same" in rels or "right_same" in rels
    assert colour_witness, "training must show odd colours reused by a majority panel"
    assert partial_row and spill
    return True


def self_test():
    layouts, rels, bgs, shapes, odd_codes = Counter(), Counter(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed, 4, 1)
        validate_generated_puzzle(p)
        for split in ("train", "test"):
            for pair in p[split]:
                info = _analyse(pair["input"])
                layouts[info["shape"]] += 1
                bgs.add(info["bg"])
                shapes.add(np.asarray(pair["input"]).shape)
                codes = [q["code"] for q in info["panels"]]
                k = _odd_index(codes)
                odd_codes.add(codes[k])
                rels[_relation(codes[(k + 1) % len(codes)], codes[k])] += 1
                # ablation: encoding a majority panel instead changes the answer
                j = (k + 1) % len(codes)
                alt = _ledger(codes[j], info["panels"][j]["colors"], info["bg"])
                assert not np.array_equal(alt, np.asarray(pair["output"]))
    assert len(layouts) >= 6, layouts
    assert len(rels) >= 5, rels
    assert len(bgs) >= 6
    assert len(shapes) >= 20
    assert len(odd_codes) >= 18
    # small and large counts
    validate_generated_puzzle(generate_puzzle(7, 2, 1))
    validate_generated_puzzle(generate_puzzle(8, 10, 4))
    for bad in ((1, 1), (11, 1), (4, 0), (3, 4), (2, 2)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return True


if __name__ == "__main__":
    self_test()
    print("ok")
