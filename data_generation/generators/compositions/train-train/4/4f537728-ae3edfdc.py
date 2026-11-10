"""Satellite-selected lattice beams.

A grid is a periodic lattice of filled blocks (ordinary fill colour) separated by
one-cell background gutters (parent A).  A few blocks are recoloured: each
"group" has one singleton anchor block and two to four satellite marker blocks
of its own marker colour, each lying somewhere along one of the anchor's four
cardinal block-rays (parent B).  The output fires a beam from the anchor along
exactly the rays that carry one of its own markers: every filled cell of that
half block-band, from the block next to the anchor to the canvas edge
(including clipped edge blocks), is painted with the marker colour, while
gutters stay background.  The marker's distance is discarded; unmarked rays,
the anchor itself and everything else are unchanged.

Only Python's standard library and NumPy are used.
"""
from __future__ import annotations

import numpy as np

DIRS = ("U", "D", "L", "R")
OPP = {"U": "D", "D": "U", "L": "R", "R": "L"}
_ALL_SUBSETS = [("U", "D"), ("L", "R"), ("U", "L"), ("U", "R"), ("D", "L"), ("D", "R"),
                ("U", "D", "L"), ("U", "D", "R"), ("U", "L", "R"), ("D", "L", "R"),
                ("U", "D", "L", "R")]
SHIFTS = ("recombine", "more_groups", "decoy", "geometry")


# ----------------------------------------------------------------------------
# Lattice helpers (shared by construction and the input-only solver)
# ----------------------------------------------------------------------------

def _axis(n, k, phase):
    """Return (bands, filled-mask) for one axis of length n, band size k."""
    filled = [((i + phase) % (k + 1)) < k for i in range(n)]
    bands, i = [], 0
    while i < n:
        if filled[i]:
            j = i
            while j < n and filled[j]:
                j += 1
            bands.append((i, j))
            i = j
        else:
            i += 1
    return bands, filled


def _runs(mask):
    bands, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            bands.append((i, j))
            i = j
        else:
            i += 1
    return bands


def _ray_blocks(anchor, d, nr, nc):
    r, c = anchor
    if d == "U":
        return [(i, c) for i in range(r - 1, -1, -1)]
    if d == "D":
        return [(i, c) for i in range(r + 1, nr)]
    if d == "L":
        return [(r, j) for j in range(c - 1, -1, -1)]
    return [(r, j) for j in range(c + 1, nc)]


def _direction(anchor, block):
    (r, c), (i, j) = anchor, block
    if i == r and j != c:
        return "L" if j < c else "R"
    if j == c and i != r:
        return "U" if i < r else "D"
    return None


# ----------------------------------------------------------------------------
# Input-only analysis and solver
# ----------------------------------------------------------------------------

def _analyze(grid):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = None
    for row in g:
        if np.all(row == row[0]):
            bg = int(row[0])
            break
    if bg is None:
        raise ValueError("no gutter row")
    row_mask = [not np.all(g[i] == bg) for i in range(h)]
    col_mask = [not np.all(g[:, j] == bg) for j in range(w)]
    rbands, cbands = _runs(row_mask), _runs(col_mask)
    vals, counts = np.unique(g[g != bg], return_counts=True)
    fill = int(vals[np.argmax(counts)])
    blocks = {}  # colour -> set of (rband, cband)
    for bi, (r0, r1) in enumerate(rbands):
        for bj, (c0, c1) in enumerate(cbands):
            cell = g[r0:r1, c0:c1]
            for col in np.unique(cell):
                col = int(col)
                if col not in (bg, fill):
                    blocks.setdefault(col, set()).add((bi, bj))
    anchors = {c: next(iter(s)) for c, s in blocks.items() if len(s) == 1}
    markers = {c: s for c, s in blocks.items() if len(s) >= 2}
    pairing = {}
    for m, mblocks in markers.items():
        cands = []
        for a, apos in anchors.items():
            dirs = [_direction(apos, b) for b in mblocks]
            if None not in dirs and len(set(dirs)) == len(dirs):
                cands.append(a)
        if len(cands) != 1:
            raise ValueError("ambiguous or missing anchor for marker colour %d" % m)
        pairing[m] = cands[0]
    return {"bg": bg, "fill": fill, "rbands": rbands, "cbands": cbands,
            "anchors": anchors, "markers": markers, "pairing": pairing}


def _paint(g, info, beams, *, mask_gutters=True):
    """beams: list of (anchor_pos, direction, colour, stop_block_or_None)."""
    out = np.array(g, dtype=int, copy=True)
    rb, cb = info["rbands"], info["cbands"]
    nr, nc = len(rb), len(cb)
    h, w = out.shape
    for apos, d, col, stop in beams:
        if not mask_gutters:  # solid half-band from the anchor edge to the canvas edge
            (r0, r1), (c0, c1) = rb[apos[0]], cb[apos[1]]
            if d == "U":
                out[0:r0, c0:c1] = col
            elif d == "D":
                out[r1:h, c0:c1] = col
            elif d == "L":
                out[r0:r1, 0:c0] = col
            else:
                out[r0:r1, c1:w] = col
            continue
        ray = _ray_blocks(apos, d, nr, nc)
        if stop is not None:
            ray = ray[: ray.index(stop) + 1]
        for (i, j) in ray:
            (r0, r1), (c0, c1) = rb[i], cb[j]
            out[r0:r1, c0:c1] = col
    return out


def _beams(info):
    beams = []
    for m, a in info["pairing"].items():
        apos = info["anchors"][a]
        for b in info["markers"][m]:
            beams.append((apos, _direction(apos, b), m, None))
    return beams


def solve(input_grid):
    g = np.asarray(input_grid, dtype=int)
    info = _analyze(g)
    return _paint(g, info, _beams(info))


# ----------------------------------------------------------------------------
# Shortcut (alternative-rule) predictions used by validation
# ----------------------------------------------------------------------------

def _shortcut(g, kind):
    g = np.asarray(g, dtype=int)
    info = _analyze(g)
    rb, cb = info["rbands"], info["cbands"]
    nr, nc = len(rb), len(cb)
    if kind == "full_cross":  # parent A alone: every arm fires
        beams = [(info["anchors"][a], d, m, None)
                 for m, a in info["pairing"].items() for d in DIRS]
        return _paint(g, info, beams)
    if kind == "to_marker":  # beam stops at the marker (distance kept)
        beams = [(info["anchors"][a], _direction(info["anchors"][a], b), m, b)
                 for m, a in info["pairing"].items() for b in info["markers"][m]]
        return _paint(g, info, beams)
    if kind == "unmasked":  # beam ignores the lattice gutters
        return _paint(g, info, _beams(info), mask_gutters=False)
    if kind == "compaction":  # parent B alone: markers slide adjacent
        out = g.copy()
        for m, a in info["pairing"].items():
            apos = info["anchors"][a]
            for b in info["markers"][m]:
                (r0, r1), (c0, c1) = rb[b[0]], cb[b[1]]
                out[r0:r1, c0:c1] = info["fill"]
            for b in info["markers"][m]:
                d = _direction(apos, b)
                t = _ray_blocks(apos, d, nr, nc)[0]
                (r0, r1), (c0, c1) = rb[t[0]], cb[t[1]]
                out[r0:r1, c0:c1] = m
        return out
    if kind == "any_marker":  # ignore colour pairing: any marker on a ray fires
        special = set(info["anchors"].values())
        for s in info["markers"].values():
            special |= s
        beams = []
        for m, a in info["pairing"].items():
            apos = info["anchors"][a]
            for d in DIRS:
                if any(b in special for b in _ray_blocks(apos, d, nr, nc)):
                    beams.append((apos, d, m, None))
        return _paint(g, info, beams)
    raise ValueError(kind)


SHORTCUTS = ("full_cross", "to_marker", "unmasked", "compaction", "any_marker")


# ----------------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------------

def _layout_groups(rng, nr, nc, rcomplete, ccomplete, n_groups, decoy, forbid_subsets):
    """Place groups in block coordinates.  Returns list of (anchor, {dir: block})."""
    cells = [(i, j) for i in range(nr) for j in range(nc) if rcomplete[i] and ccomplete[j]]
    for _ in range(400):
        groups, used = [], set()
        ok = True
        for _g in range(n_groups):
            free = [p for p in cells if p not in used]
            if not free:
                ok = False
                break
            anchor = free[int(rng.integers(len(free)))]
            avail = {}
            for d in DIRS:
                opts = [b for b in _ray_blocks(anchor, d, nr, nc)
                        if rcomplete[b[0]] and ccomplete[b[1]] and b not in used]
                if opts:
                    avail[d] = opts
            if len(avail) < 2:
                ok = False
                break
            k = int(rng.choice([2, 2, 3, 3, 4] if len(avail) >= 4 else [2, 3][: len(avail) - 1]))
            k = min(k, len(avail))
            dirs = [str(x) for x in rng.permutation(sorted(avail))[:k]]
            if frozenset(dirs) in forbid_subsets:
                ok = False
                break
            mk = {}
            for d in dirs:
                opts = avail[d]
                mk[d] = opts[int(rng.integers(len(opts)))]
            if len(set(mk.values())) != len(mk):
                ok = False
                break
            groups.append((anchor, mk))
            used.add(anchor)
            used.update(mk.values())
        if not ok:
            continue
        if _check_layout(groups, nr, nc) is None:
            continue
        has_decoy = _has_decoy(groups)
        if decoy == "require" and not has_decoy:
            continue
        if decoy == "forbid" and has_decoy:
            continue
        return groups
    return None


def _check_layout(groups, nr, nc):
    special = {}
    for gi, (a, mk) in enumerate(groups):
        special[a] = gi
        for b in mk.values():
            special[b] = gi
    arms = []
    for gi, (a, mk) in enumerate(groups):
        arm = set()
        for d in mk:
            for b in _ray_blocks(a, d, nr, nc):
                if b in special and not (special[b] == gi and b in mk.values()):
                    return None
                arm.add(b)
        arms.append(arm)
    for x in range(len(arms)):
        for y in range(x + 1, len(arms)):
            if arms[x] & arms[y]:
                return None
    # pairing must be unique by geometry: only own anchor sees all own markers
    for gi, (a, mk) in enumerate(groups):
        for hj, (h, _) in enumerate(groups):
            if hj == gi:
                continue
            dirs = [_direction(h, b) for b in mk.values()]
            if None not in dirs and len(set(dirs)) == len(dirs):
                return None
    return arms


def _has_decoy(groups):
    """True when some marker also lies on a ray of a different group's anchor."""
    for gi, (_a, mk) in enumerate(groups):
        for hj, (h, _) in enumerate(groups):
            if hj != gi and any(_direction(h, b) is not None for b in mk.values()):
                return True
    return False


def _features(groups, nr, nc, rcomplete, ccomplete):
    f = {"n_groups": len(groups), "partial": False, "far": False, "beyond": False,
         "clipped_arm": False, "opposite": False, "adjacent": False,
         "decoy": _has_decoy(groups), "subsets": set()}
    for a, mk in groups:
        f["subsets"].add(frozenset(mk))
        if len(mk) < 4:
            f["partial"] = True
        if set(mk) in ({"U", "D"}, {"L", "R"}):
            f["opposite"] = True
        for d, b in mk.items():
            ray = _ray_blocks(a, d, nr, nc)
            idx = ray.index(b)
            if idx >= 1:
                f["far"] = True
            else:
                f["adjacent"] = True
            if idx < len(ray) - 1:
                f["beyond"] = True
            last = ray[-1]
            if not (rcomplete[last[0]] and ccomplete[last[1]]):
                f["clipped_arm"] = True
    return f


def _make_pair(rng, cfg):
    """cfg keys: shape (kh,kw), size range, n_groups, decoy, forbid_subsets,
    palette (bg, fill) or None, need (set of feature names)."""
    for _ in range(300):
        kh, kw = cfg["shape"]
        lo, hi = cfg["size"]
        h = int(rng.integers(lo, hi + 1))
        w = int(rng.integers(lo, hi + 1))
        pr = int(rng.integers(0, kh + 1))
        pc = int(rng.integers(0, kw + 1))
        rbands, rfill = _axis(h, kh, pr)
        cbands, cfill = _axis(w, kw, pc)
        rcomplete = [b - a == kh for a, b in rbands]
        ccomplete = [b - a == kw for a, b in cbands]
        if sum(rcomplete) < 3 or sum(ccomplete) < 3 or len(rbands) < 4 or len(cbands) < 4:
            continue
        groups = _layout_groups(rng, len(rbands), len(cbands), rcomplete, ccomplete,
                                cfg["n_groups"], cfg["decoy"], cfg["forbid_subsets"])
        if groups is None:
            continue
        feats = _features(groups, len(rbands), len(cbands), rcomplete, ccomplete)
        if any(not feats[n] for n in cfg["need"]):
            continue
        if cfg["palette"] is not None:
            bg, fill = cfg["palette"]
        else:
            bg = 0 if rng.random() < 0.55 else int(rng.integers(0, 10))
            fill = int(rng.choice([c for c in range(10) if c != bg]))
        pool = [c for c in range(10) if c not in (bg, fill)]
        cols = [int(c) for c in rng.permutation(pool)[: 2 * len(groups)]]
        grid = np.full((h, w), bg, dtype=int)
        for i, (r0, r1) in enumerate(rbands):
            for j, (c0, c1) in enumerate(cbands):
                grid[r0:r1, c0:c1] = fill
        out = grid.copy()
        for gi, (a, mk) in enumerate(groups):
            ac, mc = cols[2 * gi], cols[2 * gi + 1]
            (r0, r1), (c0, c1) = rbands[a[0]], cbands[a[1]]
            grid[r0:r1, c0:c1] = ac
            out[r0:r1, c0:c1] = ac
            for d, b in mk.items():
                (r0, r1), (c0, c1) = rbands[b[0]], cbands[b[1]]
                grid[r0:r1, c0:c1] = mc
                for x in _ray_blocks(a, d, len(rbands), len(cbands)):
                    (r0, r1), (c0, c1) = rbands[x[0]], cbands[x[1]]
                    out[r0:r1, c0:c1] = mc
        vals, counts = np.unique(grid[grid != bg], return_counts=True)
        order = np.argsort(-counts)
        if int(vals[order[0]]) != fill or (len(counts) > 1 and counts[order[0]] == counts[order[1]]):
            continue
        feats["shape"] = (kh, kw)
        feats["phase"] = (pr, pc)
        feats["dims"] = (h, w)
        return grid, out, feats
    return None


def _check_counts(num_train, num_test):
    if not isinstance(num_train, (int, np.integer)) or not isinstance(num_test, (int, np.integer)):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("illegal train/test counts")


def _plan(rng, num_train, num_test):
    shift = SHIFTS[int(rng.integers(len(SHIFTS)))]
    shapes_all = [(2, 2), (2, 2), (2, 2), (3, 3), (2, 3), (3, 2)]
    if rng.random() < 0.55:
        shape_mode = "fixed"
        base = (2, 2) if rng.random() < 0.6 else shapes_all[int(rng.integers(len(shapes_all)))]
        shapes = [base]
    else:
        shape_mode = "mixed"
        shapes = [(2, 2), (3, 3), (2, 3), (3, 2)]
    palette = None
    if rng.random() < 0.5:
        bg = 0 if rng.random() < 0.6 else int(rng.integers(0, 10))
        palette = (bg, int(rng.choice([c for c in range(10) if c != bg])))
    return {"shift": shift, "shape_mode": shape_mode, "shapes": shapes, "palette": palette}


def _pick_shape(rng, plan):
    s = plan["shapes"]
    return s[int(rng.integers(len(s)))]


def _try_puzzle(rng, num_train, num_test):
    plan = _plan(rng, num_train, num_test)
    shift = plan["shift"]
    train, tfeats = [], []
    # Training group counts.
    if shift == "more_groups":
        max_tr = 1 if (num_train == 2 or rng.random() < 0.4) else 2
        tr_groups = [max_tr] + [int(rng.integers(1, max_tr + 1)) for _ in range(num_train - 1)]
    else:
        tr_groups = [int(rng.choice([1, 2, 2, 3])) for _ in range(num_train)]
        tr_groups[0] = max(tr_groups[0], 2) if num_train >= 2 else tr_groups[0]
    perm = list(rng.permutation(num_train))
    tr_groups = [tr_groups[i] for i in perm]
    # Required demonstrations in training.
    needs = [set() for _ in range(num_train)]
    order = [int(x) for x in rng.permutation(num_train)]
    needs[order[0]] |= {"partial", "far", "beyond"}
    needs[order[1]] |= {"clipped_arm"}
    if num_train >= 3:
        needs[order[2]] |= {"far"}
    decoy_train = set()
    if shift == "decoy":
        idx = order[1] if num_train >= 2 else order[0]
        decoy_train.add(idx)
        if tr_groups[idx] < 2:
            tr_groups[idx] = 2
    held_out = set()
    if shift == "recombine":
        all_subsets = [frozenset(c) for c in _ALL_SUBSETS]
        pick = rng.permutation(len(all_subsets))[: int(rng.integers(2, 4))]
        held_out = {all_subsets[int(x)] for x in pick}
    train_shape0 = None
    for i in range(num_train):
        shape = _pick_shape(rng, plan)
        if plan["shape_mode"] == "mixed" and i == 1:
            others = [x for x in plan["shapes"] if x != train_shape0]
            shape = others[int(rng.integers(len(others)))]
        if i == 0:
            train_shape0 = shape
        ng = tr_groups[i]
        size = (11, 20) if shift == "geometry" else ((14, 30) if ng == 3 else (10, 26))
        if ng == 3:
            size = (max(size[0], 15), max(size[1], 21))
        if shift == "decoy":
            decoy = "require" if i in decoy_train else ("allow" if ng >= 2 else "forbid")
        elif shift == "more_groups":
            decoy = "forbid"
        else:
            decoy = "allow" if rng.random() < 0.5 else "forbid"
        cfg = {"shape": shape, "size": size, "n_groups": ng, "decoy": decoy,
               "forbid_subsets": held_out, "palette": plan["palette"], "need": needs[i]}
        made = _make_pair(rng, cfg)
        if made is None:
            return None
        train.append(made)
        tfeats.append(made[2])
    train_subsets = set().union(*[f["subsets"] for f in tfeats])
    train_phases = {(f["shape"], f["phase"]) for f in tfeats}
    max_area = max(f["dims"][0] * f["dims"][1] for f in tfeats)
    max_tr_groups = max(f["n_groups"] for f in tfeats)
    test = []
    for _ in range(num_test):
        made = None
        for _a in range(30):
            shape = _pick_shape(rng, plan)
            need = set()
            forbid = set()
            if shift == "recombine":
                ng = int(rng.choice([1, 2, 2, 3]))
                decoy = "allow" if rng.random() < 0.5 else "forbid"
                size = (15, 30) if ng == 3 else (11, 27)
            elif shift == "more_groups":
                ng = min(3, max_tr_groups + 1)
                decoy = "forbid"
                size = (15, 30) if ng == 3 else (12, 28)
            elif shift == "decoy":
                ng = int(rng.choice([2, 3]))
                decoy = "require"
                size = (15, 30) if ng == 3 else (12, 28)
            else:
                ng = int(rng.choice([1, 2, 2, 3]))
                decoy = "allow" if rng.random() < 0.5 else "forbid"
                size = (21, 30)
            cfg = {"shape": shape, "size": size, "n_groups": ng, "decoy": decoy,
                   "forbid_subsets": forbid, "palette": plan["palette"], "need": need}
            cand = _make_pair(rng, cfg)
            if cand is None:
                continue
            f = cand[2]
            if shift == "recombine" and not (f["subsets"] - train_subsets):
                continue
            if shift == "geometry" and ((f["shape"], f["phase"]) in train_phases
                                        or f["dims"][0] * f["dims"][1] <= max_area):
                continue
            made = cand
            break
        if made is None:
            return None
        test.append(made)
    puzzle = {"train": [{"input": a, "output": b} for a, b, _ in train],
              "test": [{"input": a, "output": b} for a, b, _ in test]}
    try:
        validate_generated_puzzle(puzzle)
    except ValueError:
        return None
    return puzzle


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(200):
        puzzle = _try_puzzle(rng, int(num_train), int(num_test))
        if puzzle is not None:
            return puzzle
    raise RuntimeError("could not construct puzzle for seed %r" % (seed,))


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

def _pair_facts(inp, out):
    info = _analyze(inp)
    g = np.asarray(inp)
    rb, cb = info["rbands"], info["cbands"]
    nr, nc = len(rb), len(cb)
    groups = []
    for m, a in info["pairing"].items():
        apos = info["anchors"][a]
        groups.append((apos, {_direction(apos, b): b for b in info["markers"][m]}))
    if len(info["pairing"]) != len(info["anchors"]):
        raise ValueError("anchor without marker colour")
    if len(set(info["pairing"].values())) != len(info["pairing"]):
        raise ValueError("anchor with two marker colours")
    rcomplete = [b - a == max(y - x for x, y in rb) for a, b in rb]
    ccomplete = [b - a == max(y - x for x, y in cb) for a, b in cb]
    return info, groups, _features(groups, nr, nc, rcomplete, ccomplete)


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must have train/test")
    seen = set()
    train_feats = []
    pairs = [("train", p) for p in puzzle["train"]] + [("test", p) for p in puzzle["test"]]
    differs = {k: False for k in SHORTCUTS}
    test_ok_parents = True
    test_beyond = False
    for split, pair in pairs:
        inp = np.asarray(pair["input"], dtype=int)
        out = np.asarray(pair["output"], dtype=int)
        if inp.shape != out.shape or inp.ndim != 2:
            raise ValueError("shape mismatch")
        if not (1 <= inp.shape[0] <= 30 and 1 <= inp.shape[1] <= 30):
            raise ValueError("grid too large")
        if inp.min() < 0 or inp.max() > 9 or out.min() < 0 or out.max() > 9:
            raise ValueError("bad colour")
        key = inp.tobytes() + bytes(inp.shape)
        if key in seen:
            raise ValueError("duplicate input")
        seen.add(key)
        info, groups, feats = _pair_facts(inp, out)
        bg, fill = info["bg"], info["fill"]
        if not 1 <= len(groups) <= 3:
            raise ValueError("group count")
        # lattice: gutters exactly one cell, bands of one shape, >=3 complete
        rb, cb = info["rbands"], info["cbands"]
        for bands, n in ((rb, inp.shape[0]), (cb, inp.shape[1])):
            k = max(y - x for x, y in bands)
            if k not in (2, 3):
                raise ValueError("band size")
            if sum(1 for x, y in bands if y - x == k) < 3:
                raise ValueError("too few complete bands")
            for (x0, x1), (y0, y1) in zip(bands, bands[1:]):
                if y0 - x1 != 1:
                    raise ValueError("gutter width")
            if bands[0][0] > 1 or n - bands[-1][1] > 1:
                raise ValueError("edge gutter")
            for t, (x0, x1) in enumerate(bands):
                if x1 - x0 < k and not ((t == 0 and x0 == 0) or (t == len(bands) - 1 and x1 == n)):
                    raise ValueError("clipped band away from the canvas edge")
        kh = max(y - x for x, y in rb)
        kw = max(y - x for x, y in cb)
        # every special block must be complete and uniform
        for (i, j) in list(info["anchors"].values()) + [b for s in info["markers"].values() for b in s]:
            (r0, r1), (c0, c1) = rb[i], cb[j]
            if r1 - r0 != kh or c1 - c0 != kw or len(np.unique(inp[r0:r1, c0:c1])) != 1:
                raise ValueError("special block not complete/uniform")
        for m, s in info["markers"].items():
            if not 2 <= len(s) <= 4:
                raise ValueError("marker count")
        if not np.array_equal(solve(inp), out):
            raise ValueError("output does not follow rule")
        if np.array_equal(inp, out):
            raise ValueError("identity pair")
        # gutters unchanged
        if np.any(out[inp == bg] != bg):
            raise ValueError("gutter painted")
        vals, counts = np.unique(inp[inp != bg], return_counts=True)
        srt = sorted(counts)
        if len(srt) > 1 and srt[-1] == srt[-2]:
            raise ValueError("fill colour not strict majority")
        alts = {k: _shortcut(inp, k) for k in SHORTCUTS}
        if split == "train":
            train_feats.append(feats)
            for k in SHORTCUTS:
                if not np.array_equal(alts[k], out):
                    differs[k] = True
        else:
            if np.array_equal(alts["full_cross"], out) or np.array_equal(alts["compaction"], out):
                test_ok_parents = False
            if not np.array_equal(alts["to_marker"], out):
                test_beyond = True
    for k in ("full_cross", "to_marker", "unmasked", "compaction"):
        if not differs[k]:
            raise ValueError("training does not refute shortcut %s" % k)
    if not test_ok_parents:
        raise ValueError("a test pair is solved by a single-parent shortcut")
    if not test_beyond:
        raise ValueError("no test pair distinguishes beam-to-edge from beam-to-marker")
    if not any(f["clipped_arm"] for f in train_feats):
        raise ValueError("training never shows a beam entering a clipped edge block")
    test_decoy = False
    for pair in puzzle["test"]:
        _, _, f = _pair_facts(pair["input"], pair["output"])
        test_decoy |= f["decoy"]
    if test_decoy and not differs["any_marker"]:
        raise ValueError("test uses a foreign marker on an unfired ray but training never shows one")
    if not any(f["n_groups"] >= 2 for f in train_feats) and any(
            _pair_facts(p["input"], p["output"])[2]["decoy"] for p in puzzle["test"]):
        raise ValueError("decoy test without multi-group training")
    return True


# ----------------------------------------------------------------------------
# Self test
# ----------------------------------------------------------------------------

def self_test():
    stats = {"groups": set(), "shapes": set(), "decoy": 0, "opposite": 0,
             "adjacent": 0, "bgs": set(), "dims": set(), "subsets": set()}
    sigs = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        p2 = generate_puzzle(seed)
        for s in ("train", "test"):
            for a, b in zip(p[s], p2[s]):
                assert np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        validate_generated_puzzle(p)
        sigs.add(np.asarray(p["test"][0]["input"]).tobytes())
        for pair in p["train"] + p["test"]:
            info, groups, f = _pair_facts(pair["input"], pair["output"])
            stats["groups"].add(f["n_groups"])
            stats["shapes"].add((max(y - x for x, y in info["rbands"]),
                                 max(y - x for x, y in info["cbands"])))
            stats["decoy"] += f["decoy"]
            stats["opposite"] += f["opposite"]
            stats["adjacent"] += f["adjacent"]
            stats["bgs"].add(info["bg"])
            stats["dims"].add(np.asarray(pair["input"]).shape)
            stats["subsets"] |= f["subsets"]
            assert np.array_equal(solve(pair["input"]), np.asarray(pair["output"]))
    assert len(sigs) == 42
    assert stats["groups"] == {1, 2, 3}, stats["groups"]
    assert len(stats["shapes"]) >= 3, stats["shapes"]
    assert stats["decoy"] >= 10 and stats["opposite"] >= 5 and stats["adjacent"] >= 5, stats
    assert len(stats["bgs"]) >= 4 and len(stats["dims"]) >= 60, (stats["bgs"], len(stats["dims"]))
    assert len(stats["subsets"]) == 11, len(stats["subsets"])
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return True


if __name__ == "__main__":
    self_test()
    print("ok")
