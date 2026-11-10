"""Hidden diagonal parity revealed by a constant-rate sweep.

Composition of 351d6448 (A: four chronological panels -> next frame of a
constant-rate process, including an advancing recolouring frontier) and
a5f85a15 (B: every down-right diagonal run alternates from its upper-left
endpoint; even positions take a highlight colour).

Each input holds four snapshots of one panel, separated by divider lines and
read in row-major order (stacked, side by side, or 2x2).  Every panel contains
the same monochrome down-right diagonal runs.  A straight frontier sweeps the
panel along one axis at a constant number of rows/columns per frame; every
cell it has passed that is an even-numbered cell of its run (counted from the
run's upper-left endpoint) shows the highlight colour.  The output is the fifth
snapshot alone.
"""
from __future__ import annotations

import numpy as np

DIRECTIONS = ("LR", "RL", "TB", "BT")
NEAR_SIDE = ("LR", "TB")      # the frontier meets every run at its upper-left end first
FAR_SIDE = ("RL", "BT")       # the frontier meets every run at its lower-right end first
AXIS = {"LR": "h", "RL": "h", "TB": "v", "BT": "v"}
LAYOUTS = ("stack", "row", "quad")
SHIFTS = ("fresh_far", "new_direction", "scale", "faster", "gaps")
_BIG = 64


class _Retry(Exception):
    pass


# ----------------------------------------------------------------------------
# shared geometry
# ----------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    for name, value in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    num_train, num_test = int(num_train), int(num_test)
    if not 2 <= num_train <= 10:
        raise ValueError("num_train must be in 2..10")
    if not 1 <= num_test <= 4:
        raise ValueError("num_test must be in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _segments(mask):
    """Maximal down-right chains of foreground cells, each listed from its upper-left end."""
    h, w = mask.shape
    runs = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not (r > 0 and c > 0 and mask[r - 1, c - 1]):
                cells, rr, cc = [], r, c
                while rr < h and cc < w and mask[rr, cc]:
                    cells.append((rr, cc))
                    rr += 1
                    cc += 1
                runs.append(cells)
    return runs


def _keys(cells, direction):
    """Sweep coordinate: a cell is swept at frontier x when key <= x."""
    arr = np.asarray(cells, dtype=int).reshape(-1, 2)
    return {"LR": arr[:, 1], "RL": -arr[:, 1], "TB": arr[:, 0], "BT": -arr[:, 0]}[direction]


def _pack(frames, divider, layout):
    h, w = frames[0].shape
    if layout == "stack":
        grid = np.full((4 * h + 3, w), divider, dtype=int)
        for i, f in enumerate(frames):
            grid[i * (h + 1):i * (h + 1) + h, :] = f
    elif layout == "row":
        grid = np.full((h, 4 * w + 3), divider, dtype=int)
        for i, f in enumerate(frames):
            grid[:, i * (w + 1):i * (w + 1) + w] = f
    else:
        grid = np.full((2 * h + 1, 2 * w + 1), divider, dtype=int)
        for i, f in enumerate(frames):
            r, c = divmod(i, 2)
            grid[r * (h + 1):r * (h + 1) + h, c * (w + 1):c * (w + 1) + w] = f
    return grid


def _split(grid):
    """Return (layout, divider, [frame1..frame4]) or None; frames are read row-major."""
    H, W = grid.shape
    found = []
    if H >= 7 and (H - 3) % 4 == 0:
        h = (H - 3) // 4
        d = int(grid[h, 0])
        frames = [grid[i * (h + 1):i * (h + 1) + h, :] for i in range(4)]
        if all(np.all(grid[r, :] == d) for r in (h, 2 * h + 1, 3 * h + 2)) and \
                not any(np.any(f == d) for f in frames):
            found.append(("stack", d, frames))
    if W >= 7 and (W - 3) % 4 == 0:
        w = (W - 3) // 4
        d = int(grid[0, w])
        frames = [grid[:, i * (w + 1):i * (w + 1) + w] for i in range(4)]
        if all(np.all(grid[:, c] == d) for c in (w, 2 * w + 1, 3 * w + 2)) and \
                not any(np.any(f == d) for f in frames):
            found.append(("row", d, frames))
    if H >= 3 and W >= 3 and H % 2 == 1 and W % 2 == 1:
        h, w = (H - 1) // 2, (W - 1) // 2
        d = int(grid[h, 0])
        frames = [grid[:h, :w], grid[:h, w + 1:], grid[h + 1:, :w], grid[h + 1:, w + 1:]]
        if np.all(grid[h, :] == d) and np.all(grid[:, w] == d) and \
                not any(np.any(f == d) for f in frames):
            found.append(("quad", d, frames))
    if len(found) != 1:
        return None
    layout, d, frames = found[0]
    return layout, d, [np.array(f, dtype=int) for f in frames]


def _parse(grid):
    """Read everything the rule needs from the input alone."""
    grid = np.asarray(grid, dtype=int)
    split = _split(grid)
    if split is None:
        raise ValueError("input is not four divided panels")
    layout, divider, frames = split
    values, counts = np.unique(np.concatenate([f.ravel() for f in frames]), return_counts=True)
    background = int(values[np.argmax(counts)])
    fg = frames[0] != background
    if not all(np.array_equal(f != background, fg) for f in frames):
        raise ValueError("panel geometry changes between frames")
    new_colors = set()
    for t in range(3):
        changed = frames[t] != frames[t + 1]
        new_colors.update(int(v) for v in frames[t + 1][changed])
    if len(new_colors) != 1:
        raise ValueError("expected exactly one highlight colour")
    highlight = new_colors.pop()
    runs = _segments(fg)
    eligible = [cell for run in runs for cell in run[1::2]]
    conv = [np.array([f[r, c] == highlight for r, c in eligible], dtype=bool) for f in frames]
    return {"layout": layout, "divider": divider, "frames": frames, "background": background,
            "highlight": highlight, "runs": runs, "eligible": eligible, "conv": conv, "fg": fg}


def _hypotheses(parsed):
    """All constant-rate straight sweeps consistent with frames 1-4, grouped by prediction."""
    eligible, conv = parsed["eligible"], parsed["conv"]
    result = {}
    if not eligible:
        return result
    for direction in DIRECTIONS:
        keys = _keys(eligible, direction)
        intervals = []
        for m in conv:
            lo = int(keys[m].max()) if m.any() else -_BIG
            hi = int(keys[~m].min()) - 1 if (~m).any() else _BIG
            if lo > hi:
                break
            intervals.append((lo, hi))
        if len(intervals) != 4:
            continue
        lo4, hi4 = intervals[3]
        for x4 in range(max(lo4, -_BIG), min(hi4, _BIG) + 1):
            for s in range(1, 2 * _BIG):
                xs = [x4 - (3 - t) * s for t in range(3)]
                if xs[1] < intervals[1][0]:
                    break  # frame 2 would fall behind its observed conversions
                if all(intervals[t][0] <= xs[t] <= intervals[t][1] for t in range(3)):
                    pred = keys <= x4 + s
                    entry = result.setdefault(pred.tobytes(), [pred, []])
                    entry[1].append((direction, x4, s))
    return result


def _apply(frame, cells, mask, highlight):
    out = frame.copy()
    for (r, c), m in zip(cells, mask):
        if m:
            out[r, c] = highlight
    return out


def solve(input_grid):
    grid = np.asarray(input_grid, dtype=int)
    parsed = _parse(grid)
    hyps = _hypotheses(parsed)
    last = parsed["frames"][3].copy()
    if not hyps:
        return last
    pred = next(iter(hyps.values()))[0]
    return _apply(last, parsed["eligible"], pred, parsed["highlight"])


# ----------------------------------------------------------------------------
# analysis used by validation and generation
# ----------------------------------------------------------------------------

def _report(inp, out):
    """Check one pair against the rule and measure how shortcuts would fail."""
    inp = np.asarray(inp, dtype=int)
    out = np.asarray(out, dtype=int)
    P = _parse(inp)
    hyps = _hypotheses(P)
    if len(hyps) != 1:
        raise AssertionError(f"next frame not uniquely determined ({len(hyps)} predictions)")
    pred, hlist = next(iter(hyps.values()))
    frames, hl, elig, runs = P["frames"], P["highlight"], P["eligible"], P["runs"]
    f4 = frames[3]
    truth = _apply(f4, elig, pred, hl)
    if out.shape != f4.shape or not np.array_equal(truth, out):
        raise AssertionError("output is not the uniquely extrapolated fifth frame")
    for f in frames + [out]:
        bad = (f == hl) & P["fg"]
        allowed = np.zeros_like(bad)
        for r, c in elig:
            allowed[r, c] = True
        if np.any(bad & ~allowed):
            raise AssertionError("highlight on an odd-positioned run cell")
        if np.any((f == hl) & ~P["fg"]):
            raise AssertionError("highlight outside the runs")
    counts = [int(m.sum()) for m in P["conv"]] + [int(pred.sum())]
    if any(counts[t + 1] <= counts[t] for t in range(4)):
        raise AssertionError("every transition, including the predicted one, must convert a cell")
    dirs = {h[0] for h in hlist}
    speeds = {h[2] for h in hlist}
    d, x4, s = hlist[0]
    x5 = x4 + s
    fg_cells = [(r, c) for run in runs for (r, c) in run]
    fg_keys = _keys(fg_cells, d)
    elig_keys = _keys(elig, d)
    k4 = int(elig_keys[P["conv"][3]].max())
    band = (fg_keys > x4) & (fg_keys <= x5)

    def with_cells(cells):
        return _apply(f4, cells, [True] * len(cells), hl)

    def band_cells(eligible_set):
        return [cell for cell, b in zip(fg_cells, band) if b and cell in eligible_set]

    shortcuts = {}
    shortcuts["copy"] = f4
    shortcuts["full"] = _apply(f4, elig, [True] * len(elig), hl)
    shortcuts["unit"] = _apply(f4, elig, elig_keys <= k4 + 1, hl)
    shortcuts["all_cells"] = with_cells([cell for cell, b in zip(fg_cells, band) if b])
    entry = set()
    for run in runs:
        seq = run[::-1] if d in FAR_SIDE else run
        entry.update(seq[1::2])
    shortcuts["entry"] = with_cells(band_cells(entry))
    gap = set()
    first = {}
    for r, c in sorted(fg_cells):
        first.setdefault(c - r, r)
        if (r - first[c - r]) % 2 == 1:
            gap.add((r, c))
    shortcuts["gap"] = with_cells(band_cells(gap))
    for p in (0, 1):
        shortcuts[f"row{p}"] = with_cells(band_cells({x for x in fg_cells if x[0] % 2 == p}))
        shortcuts[f"col{p}"] = with_cells(band_cells({x for x in fg_cells if x[1] % 2 == p}))
    differs = {name: not np.array_equal(g, out) for name, g in shortcuts.items()}

    lit4 = f4 == hl
    lit5 = out == hl
    far_even = any(len(run) % 2 == 0 and lit4[run[-1]] for run in runs)
    far_odd = any(len(run) % 2 == 1 and len(run) >= 3 and lit4[run[-2]] for run in runs)
    fresh_far = any(len(run) % 2 == 0 and not lit4[run[-1]] and lit5[run[-1]] for run in runs)
    hl_cells = np.argwhere(lit5)
    diagonals = {}
    for run in runs:
        diagonals.setdefault(run[0][1] - run[0][0], []).append(run)
    return {
        "layout": P["layout"], "shape": out.shape, "dirs": dirs, "speeds": speeds,
        "far": dirs <= set(FAR_SIDE), "near": dirs <= set(NEAR_SIDE),
        "runs": len(runs), "max_len": max(len(run) for run in runs),
        "gapped": sum(1 for v in diagonals.values() if len(v) > 1),
        "incomplete": bool(not pred.all()), "counts": counts,
        "far_anchor": bool(dirs <= set(FAR_SIDE) and far_even and far_odd),
        "fresh_far": bool(dirs <= set(FAR_SIDE) and fresh_far),
        "parity_mix": bool(len(set(hl_cells[:, 0] % 2)) == 2 and len(set(hl_cells[:, 1] % 2)) == 2),
        "differs": differs, "background": P["background"], "highlight": hl,
        "divider": P["divider"],
    }


def _signature(grid):
    g = np.asarray(grid, dtype=int)
    return g.shape, g.tobytes()


def _test_shift_properties(train_reports, rep):
    """Which train->test novelties a test pair exhibits."""
    tdirs = set().union(*(r["dirs"] for r in train_reports))
    tmax_len = max(r["max_len"] for r in train_reports)
    tmax_speed = max(max(r["speeds"]) for r in train_reports)
    tmax_gap = max(r["gapped"] for r in train_reports)
    props = set()
    if rep["fresh_far"] and rep["differs"]["entry"]:
        props.add("fresh_far")
    if not (rep["dirs"] & tdirs):
        props.add("new_direction")
    if rep["max_len"] > tmax_len:
        props.add("scale")
    if min(rep["speeds"]) > tmax_speed and rep["differs"]["unit"]:
        props.add("faster")
    if rep["gapped"] > tmax_gap and rep["differs"]["gap"]:
        props.add("gaps")
    return props


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise AssertionError("puzzle must have exactly train and test")
    train, test = puzzle["train"], puzzle["test"]
    if not (isinstance(train, list) and isinstance(test, list)):
        raise AssertionError("train/test must be lists")
    _check_counts(len(train), len(test))
    seen = set()
    reports = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise AssertionError("pair must have exactly input and output")
            inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            for g in (inp, out):
                if g.ndim != 2 or not (1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30):
                    raise AssertionError("grid dimensions must be 1..30")
                if not np.issubdtype(g.dtype, np.integer) or g.min() < 0 or g.max() > 9:
                    raise AssertionError("colours must be integers 0..9")
            sig = _signature(inp)
            if sig in seen:
                raise AssertionError("duplicate input")
            seen.add(sig)
            rep = _report(inp, out)
            if not np.array_equal(solve(inp), out):
                raise AssertionError("solve disagrees with the stored output")
            reports[split].append(rep)
    tr, te = reports["train"], reports["test"]
    if not any(r["far_anchor"] for r in tr):
        raise AssertionError("training must show far-side entry anchored at the upper-left end")
    if not any(r["near"] for r in tr):
        raise AssertionError("training must show a sweep entering runs at their upper-left end")
    if not any(r["differs"]["unit"] for r in tr):
        raise AssertionError("training must show a multi-cell rate that a unit step misses")
    if not any(r["differs"]["full"] for r in tr):
        raise AssertionError("training must show an unfinished sweep")
    if not any(r["parity_mix"] for r in tr):
        raise AssertionError("training must rule out global row/column parity")
    if not any(r["differs"]["entry"] for r in tr):
        raise AssertionError("training must separate endpoint anchoring from entry anchoring")
    test_layouts = {r["layout"] for r in te}
    if not test_layouts <= {r["layout"] for r in tr}:
        raise AssertionError("test panel layout must appear in training")
    for rep in te:
        for name in ("copy", "full", "all_cells"):
            if not rep["differs"][name]:
                raise AssertionError(f"test shortcut '{name}' matches the answer")
        if not _test_shift_properties(tr, rep):
            raise AssertionError("test pair shows no train->test shift")
    return True


# ----------------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------------

def _dims(rng, layout, size):
    if layout == "stack":
        return int(rng.integers(4, 7)), int(rng.integers(9, 31))
    if layout == "row":
        return int(rng.integers(9, 31)), int(rng.integers(4, 7))
    lo, hi = {"small": (5, 9), "big": (10, 15), None: (5, 15)}[size]
    return int(rng.integers(lo, hi)), int(rng.integers(lo, hi))


def _place(rng, h, w, n, lmax, gaps=0, long_len=None, touch=False):
    """Monochrome down-right runs; separated by background unless `touch` allows
    corner contact between neighbouring diagonals (never along the same diagonal)."""
    occ = np.zeros((h, w), dtype=bool)
    block = np.zeros((h, w), dtype=bool)
    runs = []
    side = min(h, w)
    lmax = max(2, min(lmax, side))

    def free(cells):
        return all(not block[r, c] for r, c in cells)

    def add(cells):
        for r, c in cells:
            occ[r, c] = True
        steps = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
        if not touch:
            steps += [(1, -1), (-1, 1)]
        for r, c in cells:
            for dr, dc in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    block[rr, cc] = True
        runs.append(cells)

    def length():
        hi = lmax if rng.random() < 0.5 else min(lmax, 6)
        lo = 2 if rng.random() < 0.2 else 3
        return int(rng.integers(lo, hi + 1)) if hi >= lo else 2

    if long_len is not None:
        if long_len > side:
            raise _Retry
        L = int(rng.integers(long_len, side + 1))
        r0, c0 = int(rng.integers(0, h - L + 1)), int(rng.integers(0, w - L + 1))
        add([(r0 + i, c0 + i) for i in range(L)])
    for _ in range(gaps):
        for _try in range(40):
            L1, L2 = length(), length()
            g = int(rng.integers(1, 4))
            if rng.random() < 0.75 and (L1 + g) % 2 == 0:
                g = g + 1 if g < 3 else g - 1
            total = L1 + g + L2
            if total > side:
                continue
            r0, c0 = int(rng.integers(0, h - total + 1)), int(rng.integers(0, w - total + 1))
            a = [(r0 + i, c0 + i) for i in range(L1)]
            b = [(r0 + L1 + g + i, c0 + L1 + g + i) for i in range(L2)]
            if free(a) and free(b):
                add(a)
                add(b)
                break
        else:
            raise _Retry
    tries = 0
    while len(runs) < n and tries < 80:
        tries += 1
        L = length()
        r0, c0 = int(rng.integers(0, h - L + 1)), int(rng.integers(0, w - L + 1))
        cells = [(r0 + i, c0 + i) for i in range(L)]
        if free(cells):
            add(cells)
    if occ.sum() > 0.3 * h * w or not runs:
        raise _Retry
    parsed = _segments(occ)
    if sorted(map(tuple, parsed)) != sorted(map(tuple, runs)):
        raise _Retry
    return runs, occ


def _palette(rng, policy, n_sources):
    hl = policy["hl"] if policy["hl"] is not None else int(rng.integers(10))
    pool = [c for c in range(10) if c != hl]
    div = policy["div"] if policy["div"] is not None else \
        int(rng.choice([c for c in pool if c != policy["bg"]]))
    pool = [c for c in pool if c != div]
    bg = policy["bg"] if policy["bg"] is not None else int(rng.choice(pool))
    pool = [c for c in pool if c != bg]
    sources = [int(c) for c in rng.choice(pool, size=n_sources, replace=False)]
    return bg, div, hl, sources


def _choose_layout(rng, layouts, direction):
    fit = {"stack": "h", "row": "v"}
    weights = np.array([3.0 if l == "quad" or fit.get(l) == AXIS[direction] else 0.6
                        for l in layouts])
    return layouts[int(rng.choice(len(layouts), p=weights / weights.sum()))]


def _make_pair(rng, req, policy):
    for _attempt in range(300):
        direction = str(rng.choice(req["dirs"]))
        layout = _choose_layout(rng, req["layouts"], direction)
        h, w = _dims(rng, layout, req.get("size"))
        axis_len = w if AXIS[direction] == "h" else h
        area = h * w
        n = int(rng.integers(2, max(3, min(8, area // 11)) + 1))
        if rng.random() < 0.08:
            n = 1
        try:
            runs, occ = _place(rng, h, w, n, req.get("lmax", 14), req.get("gaps", 0),
                               req.get("long_len"), touch=bool(rng.random() < 0.1))
        except _Retry:
            continue
        elig = [cell for run in runs for cell in run[1::2]]
        keys = _keys(elig, direction)
        speeds = [s for s in req["speeds"] if 4 * s <= axis_len + 2]
        if not speeds:
            continue
        options = []
        for s in speeds:
            for x1 in range(int(keys.min()) - s, int(keys.max()) + 1):
                counts = [int((keys <= x1 + s * t).sum()) for t in range(5)]
                if any(counts[t + 1] <= counts[t] for t in range(4)):
                    continue
                if req.get("incomplete", True) and counts[4] == len(keys):
                    continue
                options.append((x1, s))
        if not options:
            continue
        n_src = int(rng.choice([1, 1, 2, 2, 3])) if len(runs) > 1 else 1
        bg, div, hl, sources = _palette(rng, policy, n_src)
        base = np.full((h, w), bg, dtype=int)
        for i, run in enumerate(runs):
            colour = sources[i % len(sources)] if rng.random() < 0.6 else int(rng.choice(sources))
            for r, c in run:
                base[r, c] = colour
        # pick a speed uniformly first so slow sweeps are not crowded out
        by_speed = {}
        for x1, s in options:
            by_speed.setdefault(s, []).append((x1, s))
        speed_order = [int(v) for v in rng.permutation(sorted(by_speed))]
        ordered = []
        for s in speed_order:
            group = by_speed[s]
            ordered += [group[int(i)] for i in rng.permutation(len(group))[:4]]
        if rng.random() < 0.5:   # sometimes insist that frame 1 already shows highlights
            ordered = [o for o in ordered if o[0] >= keys.min()] + \
                      [o for o in ordered if o[0] < keys.min()]
        for x1, s in ordered[:10]:
            frames = [_apply(base, elig, keys <= x1 + s * t, hl) for t in range(5)]
            inp = _pack(frames[:4], div, layout)
            out = frames[4]
            if inp.shape[0] > 30 or inp.shape[1] > 30:
                break
            try:
                rep = _report(inp, out)
            except (AssertionError, ValueError):
                continue
            if all(req_ok(rep) for req_ok in req.get("checks", [])):
                return {"input": inp, "output": out}, rep
    raise _Retry


def _flag(name):
    return lambda rep: bool(rep[name])


def _differs(name):
    return lambda rep: bool(rep["differs"][name])


def _plan(rng, num_train, num_test):
    shift = str(rng.choice(SHIFTS))
    # palette policy: some puzzles keep roles fixed, others vary them per pair
    hl = int(rng.integers(10)) if rng.random() < 0.6 else None
    rest = [c for c in range(10) if c != hl]
    div = int(rng.choice(rest)) if rng.random() < 0.5 else None
    rest = [c for c in rest if c != div]
    bg = int(rng.choice(rest)) if rng.random() < 0.35 else None
    policy = {"hl": hl, "div": div, "bg": bg}
    # layout policy
    roll = rng.random()
    if shift in ("scale", "new_direction"):
        layouts = ["quad"] + [l for l in ("stack", "row") if rng.random() < 0.4]
    elif roll < 0.45:
        layouts = [str(rng.choice(LAYOUTS))]
    else:
        k = int(rng.integers(2, 4))
        layouts = [str(x) for x in rng.choice(LAYOUTS, size=k, replace=False)]
    return {"shift": shift, "policy": policy, "layouts": layouts}


def _build(rng, plan, num_train, num_test):
    shift, policy, layouts = plan["shift"], plan["policy"], plan["layouts"]
    axis = str(rng.choice(["h", "v"]))
    if shift == "new_direction":
        train_dirs = [d for d in DIRECTIONS if AXIS[d] == axis]
        test_dirs = [d for d in DIRECTIONS if AXIS[d] != axis]
    else:
        train_dirs = list(DIRECTIONS)
        test_dirs = list(DIRECTIONS)
    train_speeds = [1, 2] if shift == "faster" else [1, 2, 3]
    lmax = 6 if shift == "scale" else 14
    base = {"layouts": layouts, "speeds": train_speeds, "lmax": lmax}
    if shift == "scale":
        base["size"] = "small"
    roles = [
        dict(base, dirs=[d for d in train_dirs if d in FAR_SIDE],
             checks=[_flag("far_anchor"), _differs("entry"), _differs("full")]),
        dict(base, dirs=[d for d in train_dirs if d in NEAR_SIDE],
             speeds=[s for s in train_speeds if s >= 2],
             checks=[_differs("unit"), _flag("parity_mix"), _differs("full")]),
    ]
    if shift == "scale":
        roles[1]["layouts"] = ["quad"]
    if shift == "gaps":
        if num_train > 2:
            roles.append(dict(base, dirs=train_dirs, gaps=1, checks=[_differs("gap")]))
        else:
            roles[1]["gaps"] = 1
            roles[1]["checks"] = roles[1]["checks"] + [_differs("gap")]
    while len(roles) < num_train:
        roles.append(dict(base, dirs=train_dirs, incomplete=bool(rng.random() < 0.85),
                          gaps=int(rng.random() < 0.15) if shift != "gaps" else 0))
    train, reports = [], []
    seen = set()
    for req in roles:
        for _ in range(6):
            pair, rep = _make_pair(rng, req, policy)
            if _signature(pair["input"]) not in seen:
                break
        else:
            raise _Retry
        seen.add(_signature(pair["input"]))
        train.append(pair)
        reports.append(rep)
    order = rng.permutation(len(train))
    train = [train[i] for i in order]
    reports = [reports[i] for i in order]
    used_layouts = sorted({r["layout"] for r in reports})
    tmax_len = max(r["max_len"] for r in reports)
    tmax_gap = max(r["gapped"] for r in reports)
    test = []
    common = [_differs("copy"), _differs("full"), _differs("all_cells")]
    for _ in range(num_test):
        req = {"layouts": used_layouts, "dirs": test_dirs, "speeds": [1, 2, 3, 4], "lmax": 14}
        checks = list(common)
        if shift == "fresh_far":
            req["dirs"] = list(FAR_SIDE)
            checks += [_flag("fresh_far"), _differs("entry")]
        elif shift == "scale":
            req["layouts"] = ["quad"]
            req["size"] = "big"
            req["long_len"] = max(8, tmax_len + 2)
            checks.append(lambda rep, m=tmax_len: rep["max_len"] > m)
        elif shift == "faster":
            req["speeds"] = [3, 4, 5]
            checks.append(_differs("unit"))
        elif shift == "gaps":
            req["gaps"] = tmax_gap + 1
            checks += [_differs("gap"), lambda rep, m=tmax_gap: rep["gapped"] > m]
        req["checks"] = checks
        for _ in range(6):
            pair, rep = _make_pair(rng, req, policy)
            if _signature(pair["input"]) not in seen:
                break
        else:
            raise _Retry
        if not _test_shift_properties(reports, rep):
            raise _Retry
        seen.add(_signature(pair["input"]))
        test.append(pair)
    return {"train": train, "test": test}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(60):
        plan = _plan(rng, num_train, num_test)
        try:
            puzzle = _build(rng, plan, num_train, num_test)
        except _Retry:
            continue
        puzzle = {split: [{"input": p["input"].astype(int).tolist(),
                           "output": p["output"].astype(int).tolist()} for p in pairs]
                  for split, pairs in puzzle.items()}
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle
    raise RuntimeError("could not build a puzzle for this seed")


# ----------------------------------------------------------------------------
# self test
# ----------------------------------------------------------------------------

def self_test():
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (4.0, 1), (True, 1)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {bad}")
    layouts, dirs, speeds, shifts, shapes, structures = set(), set(), set(), set(), set(), set()
    gapped = fresh = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert puzzle == generate_puzzle(seed), "nondeterministic"
        reports = {s: [_report(p["input"], p["output"]) for p in puzzle[s]] for s in ("train", "test")}
        for split, reps in reports.items():
            for pair, rep in zip(puzzle[split], reps):
                assert np.array_equal(solve(np.array(pair["input"])), np.array(pair["output"]))
                layouts.add(rep["layout"])
                dirs.update(rep["dirs"])
                speeds.update(rep["speeds"])
                shapes.add(np.array(pair["input"]).shape)
                gapped += rep["gapped"] > 0
                fresh += rep["fresh_far"]
        for rep in reports["test"]:
            shifts.update(_test_shift_properties(reports["train"], rep))
        structures.add(str(puzzle))
    assert layouts == set(LAYOUTS), layouts
    assert dirs == set(DIRECTIONS), dirs
    assert len(speeds) >= 4, speeds
    assert shifts == set(SHIFTS), shifts
    assert len(shapes) >= 30 and gapped >= 10 and fresh >= 5
    assert len(structures) == 42
    for a, b in ((2, 1), (10, 4), (3, 2)):
        p = generate_puzzle(7, a, b)
        assert len(p["train"]) == a and len(p["test"]) == b
        validate_generated_puzzle(p)
    return True


if __name__ == "__main__":
    self_test()
    print("self_test passed")
