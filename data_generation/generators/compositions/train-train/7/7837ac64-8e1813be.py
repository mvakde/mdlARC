"""Stained-glass window: threads dye the pinned panes they pass through.

Composition of parent A (7837ac64: a regular lattice whose substituted
crossings are corners; a cell is decoded when all four of its corners carry the
same payload) and parent B (8e1813be: parallel full-span signal lines, masked by
a solid square plus a one-cell moat, whose colours become complete rows or
columns of a compact output).

Input: a uniform background, one rectangular lattice "window" (structural colour,
regular pitch per axis, one-cell lines, border included) whose crossings may be
replaced by one marker ("pin") colour, and monochrome parallel "threads" that
span the canvas but are hidden by the window and its one-cell background moat.
A thread lies inside one band of panes (strictly between two consecutive
lattice lines).  Threads that never reach the window (decoys) are full lines.

Output: one cell per window pane.  A pane is lit when all four of its corner
crossings are pinned (parent A); it takes the colour of the thread that runs
through its band (parent B, extended by projection).  Unpinned panes and panes
in a band without a thread are background.  Horizontal threads dye rows of
panes, vertical threads dye columns.

Only the Python standard library and NumPy are used.
"""
from __future__ import annotations

import json

import numpy as np

MAX_SIDE = 30
PITCHES = (2, 3, 4, 5)


# --------------------------------------------------------------------------
# Input-only parsing and solving
# --------------------------------------------------------------------------

def _background(g):
    vals, counts = np.unique(g, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    if len(vals) > 1 and counts[order[0]] == counts[order[1]]:
        raise ValueError("ambiguous background")
    return int(vals[order[0]])


def _regular_positions(flags):
    """Indices of True entries if they are 0, p, 2p, ..., last with p >= 2."""
    idx = [i for i, f in enumerate(flags) if f]
    if len(idx) < 2 or idx[0] != 0 or idx[-1] != len(flags) - 1:
        return None
    p = idx[1] - idx[0]
    if p < 2 or idx != list(range(0, len(flags), p)):
        return None
    return p


def _try_frame(g, b, c):
    ys, xs = np.nonzero(g == c)
    if len(ys) == 0:
        return None
    r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    sub = g[r0:r1 + 1, c0:c1 + 1]
    pr = _regular_positions([bool(np.all(row != b)) for row in sub])
    pc = _regular_positions([bool(np.all(col != b)) for col in sub.T])
    if pr is None or pc is None:
        return None
    h, w = sub.shape
    row_line = np.zeros(h, bool)
    row_line[::pr] = True
    col_line = np.zeros(w, bool)
    col_line[::pc] = True
    lattice = row_line[:, None] | col_line[None, :]
    crossing = row_line[:, None] & col_line[None, :]
    if np.any(sub[~lattice] != b):
        return None
    if np.any(sub[lattice & ~crossing] != c):
        return None
    cross_vals = sub[::pr, ::pc]
    markers = set(int(v) for v in np.unique(cross_vals)) - {c}
    if len(markers) > 1 or b in markers:
        return None
    marker = markers.pop() if markers else None
    pins = cross_vals != c
    return {"r0": r0, "r1": r1, "c0": c0, "c1": c1, "pr": pr, "pc": pc,
            "structure": c, "marker": marker, "pins": pins,
            "nr": (r1 - r0) // pr, "nc": (c1 - c0) // pc}


def _decode(pins):
    """Parent A: a pane is open when all four corner crossings are pinned."""
    return pins[:-1, :-1] & pins[:-1, 1:] & pins[1:, :-1] & pins[1:, 1:]


def _horizontal_threads(g, b, f):
    """Read horizontal threads of grid g for frame f (None if inconsistent).

    Returns (band_colors, decoys) where band_colors[i] is the colour of the
    thread inside row band i (or None) and decoys lists (row, colour) of full
    lines outside the window's row span.
    """
    H, W = g.shape
    ry0, ry1 = f["r0"] - 1, f["r1"] + 1
    cx0, cx1 = f["c0"] - 1, f["c1"] + 1
    bands = [None] * f["nr"]
    decoys = []
    for y in range(H):
        if ry0 <= y <= ry1:
            cols = [x for x in range(W) if x < cx0 or x > cx1]
        else:
            cols = list(range(W))
        if not cols:
            continue
        vals = g[y, cols]
        if np.all(vals == b):
            continue
        if np.any(vals == b) or len(set(int(v) for v in vals)) != 1:
            return None
        colour = int(vals[0])
        if f["r0"] < y < f["r1"] and (y - f["r0"]) % f["pr"]:
            band = (y - f["r0"]) // f["pr"]
            if bands[band] is not None:
                return None
            bands[band] = colour
        elif ry0 <= y <= ry1:
            return None  # a line on the frame or moat rows is not part of the family
        else:
            decoys.append((y, colour))
    if all(v is None for v in bands):
        return None
    return bands, decoys


def _transpose_frame(f):
    return {"r0": f["c0"], "r1": f["c1"], "c0": f["r0"], "c1": f["r1"],
            "pr": f["pc"], "pc": f["pr"], "structure": f["structure"],
            "marker": f["marker"], "pins": f["pins"].T,
            "nr": f["nc"], "nc": f["nr"]}


def parse(input_grid):
    """Recover every role of an input grid; raises ValueError if malformed."""
    g = np.asarray(input_grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")
    b = _background(g)
    frames = [f for c in np.unique(g) if int(c) != b
              for f in [_try_frame(g, b, int(c))] if f is not None]
    if len(frames) != 1:
        raise ValueError(f"expected one lattice window, found {len(frames)}")
    f = frames[0]
    H, W = g.shape
    # one-cell background moat around the window
    y0, y1 = max(f["r0"] - 1, 0), min(f["r1"] + 1, H - 1)
    x0, x1 = max(f["c0"] - 1, 0), min(f["c1"] + 1, W - 1)
    region = g[y0:y1 + 1, x0:x1 + 1].copy()
    region[f["r0"] - y0:f["r1"] - y0 + 1, f["c0"] - x0:f["c1"] - x0 + 1] = b
    if np.any(region != b):
        raise ValueError("moat is not clear")
    hor = _horizontal_threads(g, b, f)
    ver = _horizontal_threads(g.T, b, _transpose_frame(f))
    if (hor is None) == (ver is None):
        raise ValueError("thread orientation is ambiguous")
    orientation = "h" if hor is not None else "v"
    bands, decoys = hor if hor is not None else ver
    return {"background": b, "frame": f, "orientation": orientation,
            "bands": bands, "decoys": decoys, "open": _decode(f["pins"])}


def _render(info, bands):
    b = info["background"]
    opened = info["open"]
    out = np.full(opened.shape, b, dtype=int)
    for i, colour in enumerate(bands):
        if colour is None:
            continue
        if info["orientation"] == "h":
            out[i, opened[i, :]] = colour
        else:
            out[opened[:, i], i] = colour
    return out


def solve(input_grid):
    info = parse(input_grid)
    return _render(info, info["bands"])


# --------------------------------------------------------------------------
# Shortcut predictions used to prove the demonstrations discriminate
# --------------------------------------------------------------------------

def _shortcut_outputs(info):
    """Outputs of plausible simpler rules (each ignores part of the true rule)."""
    f, b = info["frame"], info["background"]
    pins = f["pins"]
    res = {}
    # ignore parent A: every pane of a threaded band is dyed (parent B stripes)
    full = dict(info)
    full["open"] = np.ones_like(info["open"])
    res["stripes_ignore_pins"] = _render(full, info["bands"])
    # any pinned corner opens the pane
    anyc = dict(info)
    anyc["open"] = pins[:-1, :-1] | pins[:-1, 1:] | pins[1:, :-1] | pins[1:, 1:]
    res["any_corner"] = _render(anyc, info["bands"])
    # ignore parent B: open panes take the pin colour
    pin_col = np.full(info["open"].shape, b, dtype=int)
    if f["marker"] is not None:
        pin_col[info["open"]] = f["marker"]
    res["pin_colour"] = pin_col
    # parent-B ordering: stack every parallel line in order into the bands
    lines = [(f["r0"] + i, c) if info["orientation"] == "h" else (f["c0"] + i, c)
             for i, c in enumerate(info["bands"]) if c is not None]
    lines = sorted(lines + list(info["decoys"]))
    nb = len(info["bands"])
    stacked = [c for _, c in lines][:nb] + [None] * max(0, nb - len(lines))
    res["ordered_stack"] = _render(info, stacked)
    # fixed orientation: always rows (or always columns)
    flip = dict(info)
    flip["orientation"] = "v" if info["orientation"] == "h" else "h"
    if len(info["bands"]) == (info["open"].shape[1] if flip["orientation"] == "v" else info["open"].shape[0]):
        res["wrong_axis"] = _render(flip, info["bands"])
    else:
        res["wrong_axis"] = None
    return res


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    for v in (num_train, num_test):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError(f"illegal counts num_train={num_train} num_test={num_test}")


def _choose_roles(rng):
    b = 0 if rng.random() < 0.5 else int(rng.integers(0, 10))
    others = [c for c in range(10) if c != b]
    s, m = (int(v) for v in rng.choice(others, 2, replace=False))
    return b, s, m


def _plan(rng, num_train, num_test):
    total = num_train + num_test
    mode = rng.choice(["h", "v", "mixed"], p=[0.25, 0.25, 0.5])
    if mode == "mixed":
        while True:
            train_or = [str(rng.choice(["h", "v"])) for _ in range(num_train)]
            if len(set(train_or)) == 2:
                break
        test_or = [str(rng.choice(["h", "v"])) for _ in range(num_test)]
    else:
        train_or = [str(mode)] * num_train
        test_or = [str(mode)] * num_test
    plan = {
        "orient": train_or + test_or,
        "decoys": rng.random() < 0.45,
        "unlit": rng.random() < 0.4,
        "stable_roles": rng.random() < 0.55,
        "repeat_colours": rng.random() < 0.5,
        "rect": rng.random() < 0.35,
        "aniso": rng.random() < 0.3,
        "free_lane": rng.random() < 0.5,
        "regime": str(rng.choice(["larger", "similar", "smaller"], p=[0.45, 0.4, 0.15])),
        "roles": _choose_roles(rng),
    }
    # size caps per split
    if plan["regime"] == "larger":
        tcap = int(rng.integers(3, 6))
        plan["train_sizes"] = (2, tcap)
        plan["test_sizes"] = (tcap + 1, 7)
    elif plan["regime"] == "smaller":
        plan["train_sizes"] = (3, 6)
        plan["test_sizes"] = (2, 4)
    else:
        plan["train_sizes"] = (2, 6)
        plan["test_sizes"] = (2, 6)

    def feature_flags(enabled):
        if not enabled:
            return [False] * total
        tr = [bool(rng.random() < 0.5) for _ in range(num_train)]
        tr[int(rng.integers(num_train))] = True
        te = [bool(rng.random() < 0.6) for _ in range(num_test)]
        return tr + te

    plan["decoy_flags"] = feature_flags(plan["decoys"])
    plan["unlit_flags"] = feature_flags(plan["unlit"])
    return plan


def _sample_size(rng, lo, hi, plan):
    nr = int(rng.integers(lo, hi + 1))
    nc = int(rng.integers(lo, hi + 1)) if plan["rect"] else nr
    return nr, nc


def _make_pair(rng, plan, k, split):
    """Build one pair in canonical horizontal-thread form, then orient it."""
    lo, hi = plan["train_sizes"] if split == "train" else plan["test_sizes"]
    want_decoy = plan["decoy_flags"][k]
    want_unlit = plan["unlit_flags"][k]
    if plan["stable_roles"]:
        b, s, m = plan["roles"]
    else:
        b, s, m = _choose_roles(rng)
    line_pool = [c for c in range(10) if c not in (b, s, m)]
    for _ in range(80):
        nr, nc = _sample_size(rng, lo, hi, plan)
        if want_unlit and nr < 3:
            nr = 3
        # pitches fitting the canvas (threads need >= 2 extra columns, decoys 2 rows)
        row_room = MAX_SIDE - (2 if want_decoy else 0)
        prs = [p for p in PITCHES if nr * p + 1 <= row_room]
        pcs = [p for p in PITCHES if nc * p + 1 <= MAX_SIDE - 2]
        if not prs or not pcs:
            continue
        pr = int(rng.choice(prs))
        pc = int(rng.choice(pcs)) if plan["aniso"] else (pr if pr in pcs else int(rng.choice(pcs)))
        hw, ww = nr * pr + 1, nc * pc + 1
        # horizontal layout (thread visibility sides)
        slack = MAX_SIDE - ww
        sides = []
        if slack >= 4:
            sides.append("both")
        sides += ["left", "right"]
        side = str(rng.choice(sides))
        if side == "both":
            ml = int(rng.integers(2, slack - 1))
            mr = int(rng.integers(2, slack - ml + 1))
        else:
            vis = int(rng.integers(2, min(slack, 8) + 1))
            hid = int(rng.integers(0, min(2, slack - vis) + 1))
            ml, mr = (vis, hid) if side == "left" else (hid, vis)
        # vertical layout
        vslack = MAX_SIDE - hw
        n_decoy = int(rng.integers(1, 3)) if want_decoy else 0
        if want_decoy:
            above = int(rng.integers(0, n_decoy + 1))
            below = n_decoy - above
            need_t = 0 if above == 0 else 2 * above   # rows r0-2, r0-4, ...
            need_b = 0 if below == 0 else 2 * below
            if need_t + need_b > vslack:
                continue
            t = need_t + int(rng.integers(0, min(3, vslack - need_t - need_b) + 1))
            bt = need_b + int(rng.integers(0, min(3, vslack - t - need_b) + 1))
        else:
            above = below = 0
            t = int(rng.integers(0, min(4, vslack) + 1))
            bt = int(rng.integers(0, min(4, vslack - t) + 1))
        H, W = t + hw + bt, ml + ww + mr
        r0, c0 = t, ml
        r1, c1 = r0 + hw - 1, c0 + ww - 1
        # which bands carry threads
        unlit = set()
        if want_unlit:
            n_un = int(rng.integers(1, max(1, nr // 2) + 1))
            unlit = set(int(v) for v in rng.choice(nr, n_un, replace=False))
        lit = [i for i in range(nr) if i not in unlit]
        # thread colours
        if plan["repeat_colours"]:
            sub = list(rng.choice(line_pool, int(rng.integers(2, 4)), replace=False))
            cols = [int(rng.choice(sub)) for _ in lit]
        else:
            if len(lit) > len(line_pool):
                continue
            cols = [int(v) for v in rng.choice(line_pool, len(lit), replace=False)]
        if len(lit) >= 2 and len(set(cols)) < 2:
            continue
        bands = [None] * nr
        for i, c in zip(lit, cols):
            bands[i] = c
        # lanes
        fixed_lane = int(rng.integers(1, pr))
        lanes = [int(rng.integers(1, pr)) if plan["free_lane"] else fixed_lane for _ in range(nr)]
        # pins (parent A vertex code)
        mask_ok = False
        for _ in range(60):
            d = rng.uniform(0.2, 0.55)
            intended = rng.random((nr, nc)) < d
            pins = np.zeros((nr + 1, nc + 1), bool)
            for i, j in zip(*np.nonzero(intended)):
                pins[i:i + 2, j:j + 2] = True
            opened = _decode(pins)
            lit_rows = opened[lit]
            frac = lit_rows.mean()
            if not lit_rows.any() or lit_rows.all() or not 0.2 <= frac <= 0.8:
                continue
            if unlit and not all(opened[i].any() for i in unlit):
                continue
            mask_ok = True
            break
        if not mask_ok:
            continue
        g = np.full((H, W), b, dtype=int)
        for i, c in enumerate(bands):
            if c is None:
                continue
            y = r0 + i * pr + lanes[i]
            for x in range(W):
                if not (c0 - 1 <= x <= c1 + 1):
                    g[y, x] = c
        decoy_cols = [int(rng.choice(line_pool)) for _ in range(n_decoy)]
        rows = [r0 - 2 * (q + 1) for q in range(above)] + [r1 + 2 * (q + 1) for q in range(below)]
        for y, c in zip(rows, decoy_cols):
            g[y, :] = c
        for q in range(nr + 1):
            g[r0 + q * pr, c0:c1 + 1] = s
        for q in range(nc + 1):
            g[r0:r1 + 1, c0 + q * pc] = s
        for i, j in zip(*np.nonzero(pins)):
            g[r0 + i * pr, c0 + j * pc] = m
        out = np.full((nr, nc), b, dtype=int)
        for i, c in enumerate(bands):
            if c is not None:
                out[i, opened[i]] = c
        if plan["orient"][k] == "v":
            g, out = g.T.copy(), out.T.copy()
        try:
            if not np.array_equal(solve(g), out):
                continue
        except ValueError:
            continue
        return g, out
    return None


def _pair_facts(inp, out):
    info = parse(inp)
    sc = _shortcut_outputs(info)
    diff = {k: (v is None or not np.array_equal(v, out)) for k, v in sc.items()}
    return {
        "orientation": info["orientation"],
        "decoys": len(info["decoys"]) > 0,
        "unlit": any(c is None for c in info["bands"]),
        "diff": diff,
        "colours": len(set(int(v) for v in np.unique(out)) - {info["background"]}),
    }


def _puzzle_ok(train, test):
    tr = [_pair_facts(p["input"], p["output"]) for p in train]
    te = [_pair_facts(p["input"], p["output"]) for p in test]
    for t in te:
        if t["orientation"] not in {x["orientation"] for x in tr}:
            return False
        if t["decoys"] and not any(x["decoys"] for x in tr):
            return False
        if t["unlit"] and not any(x["unlit"] for x in tr):
            return False
    for key in ("stripes_ignore_pins", "any_corner", "pin_colour"):
        if not any(x["diff"][key] for x in tr):
            return False
    if any(t["diff"]["ordered_stack"] for t in te) and not any(x["diff"]["ordered_stack"] for x in tr):
        return False
    if not any(x["colours"] >= 2 for x in tr):
        return False
    return True


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = 4 if num_train is None else num_train
    num_test = 1 if num_test is None else num_test
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(400):
        plan = _plan(rng, num_train, num_test)
        pairs, seen, ok = [], set(), True
        for k in range(num_train + num_test):
            split = "train" if k < num_train else "test"
            made = None
            for _ in range(6):
                made = _make_pair(rng, plan, k, split)
                if made is not None and json.dumps(made[0].tolist()) not in seen:
                    break
                made = None
            if made is None:
                ok = False
                break
            seen.add(json.dumps(made[0].tolist()))
            pairs.append({"input": made[0].tolist(), "output": made[1].tolist()})
        if not ok:
            continue
        train, test = pairs[:num_train], pairs[num_train:]
        if _puzzle_ok(train, test):
            return {"train": train, "test": test}
    raise RuntimeError("could not build puzzle")


# --------------------------------------------------------------------------
# Validation and self test
# --------------------------------------------------------------------------

def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    ntr, nte = len(puzzle["train"]), len(puzzle["test"])
    _check_counts(ntr, nte)
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert set(pair) == {"input", "output"}
            inp = np.asarray(pair["input"], dtype=int)
            out = np.asarray(pair["output"], dtype=int)
            for a in (inp, out):
                assert a.ndim == 2 and 1 <= a.shape[0] <= 30 and 1 <= a.shape[1] <= 30
                assert a.min() >= 0 and a.max() <= 9
            key = json.dumps(inp.tolist())
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = parse(inp)
            f = info["frame"]
            b = info["background"]
            assert f["nr"] >= 2 and f["nc"] >= 2
            assert f["structure"] != b and f["marker"] is not None
            assert out.shape == info["open"].shape
            assert np.array_equal(solve(inp), out), "solve mismatch"
            # thread colours never reuse background, frame or pin roles
            used = [c for c in info["bands"] if c is not None] + [c for _, c in info["decoys"]]
            assert all(c not in (b, f["structure"], f["marker"]) for c in used)
            # every lit band has both an open and a closed pane overall
            lit = [i for i, c in enumerate(info["bands"]) if c is not None]
            opened = info["open"] if info["orientation"] == "h" else info["open"].T
            assert opened[lit].any() and not opened[lit].all()
            for i, c in enumerate(info["bands"]):
                if c is None:
                    assert opened[i].any(), "unlit band must contain an open pane"
    assert _puzzle_ok(puzzle["train"], puzzle["test"]), "demonstration coverage failed"
    return True


def self_test():
    seeds = list(range(42))
    stats = {"orient": set(), "shapes": set(), "pitch": set(), "decoys": 0, "unlit": 0,
             "rect": 0, "roles": set(), "test_larger": 0}
    for seed in seeds:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        if seed < 4:
            assert json.dumps(p) == json.dumps(generate_puzzle(seed)), "nondeterministic"
        tr_max = 0
        for split in ("train", "test"):
            for pair in p[split]:
                info = parse(pair["input"])
                f = info["frame"]
                stats["orient"].add(info["orientation"])
                stats["shapes"].add(np.asarray(pair["output"]).shape)
                stats["pitch"].add((f["pr"], f["pc"]))
                stats["decoys"] += bool(info["decoys"])
                stats["unlit"] += any(c is None for c in info["bands"])
                stats["rect"] += f["nr"] != f["nc"]
                stats["roles"].add((info["background"], f["structure"], f["marker"]))
                size = max(f["nr"], f["nc"])
                if split == "train":
                    tr_max = max(tr_max, size)
                elif size > tr_max:
                    stats["test_larger"] += 1
    assert stats["orient"] == {"h", "v"}
    assert len(stats["shapes"]) >= 10 and len(stats["pitch"]) >= 6
    assert stats["decoys"] >= 10 and stats["unlit"] >= 10 and stats["rect"] >= 10
    assert len(stats["roles"]) >= 20 and stats["test_larger"] >= 8
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    for a, b in ((2, 1), (10, 4), (5, 4)):
        q = generate_puzzle(7, a, b)
        assert len(q["train"]) == a and len(q["test"]) == b
        validate_generated_puzzle(q)
    return True


if __name__ == "__main__":
    self_test()
    print("self_test ok")
