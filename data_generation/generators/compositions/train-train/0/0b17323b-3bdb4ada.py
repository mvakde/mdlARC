"""Forecast rails: complete every colour's progression of solid bars; forecast terms are dashed.

Rule (input -> output)
    The input is a uniform background holding solid bars that are three cells thick
    and an odd number (>= 3) of cells long, horizontal (3 x L) or vertical (L x 3).
    Bars of one colour are identical and are consecutive terms of an arithmetic
    progression of positions (constant displacement vector).  Every further term of
    that progression -- in both directions -- whose whole bar lies inside the grid is
    added in the colour of its family, drawn as a *dashed rail*: the bar's two long
    edges and both end caps stay solid while every second cell of its centre line,
    starting one cell inside, becomes background.  Given bars stay solid.  A colour
    with a single bar defines no displacement and is left unchanged.  A term that
    would cross the grid edge (even partly) is not drawn and ends the progression.

Parent A (0b17323b) supplies the progression: equally spaced identical seeds encode a
displacement which is repeated until the next term leaves the canvas, and the added
terms carry a distinct continuation role.  Parent B (3bdb4ada) supplies the objects and
the continuation role: solid 3-thick odd bars, independent per object, and the
alternating centre-line punching whose holes take the local background colour.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# geometry shared by solver, generator and validator
# ---------------------------------------------------------------------------

LINE_CLASSES = ("row", "col", "diag", "anti")


def _dims(orient, length):
    return (3, length) if orient == "H" else (length, 3)


def _fits(r, c, h, w, H, W):
    return r >= 0 and c >= 0 and r + h <= H and c + w <= W


def _meets_grid(r, c, h, w, H, W):
    return r < H and c < W and r + h > 0 and c + w > 0


def _apart(a, b):
    """True when two rectangles are separated by at least one empty row or column."""
    (r1, c1, h1, w1), (r2, c2, h2, w2) = a, b
    return r1 + h1 < r2 or r2 + h2 < r1 or c1 + w1 < c2 or c2 + w2 < c1


def _paint(grid, rect, color, hole=None):
    """Paint a solid bar; when ``hole`` is given punch its centre line as a dashed rail."""
    r, c, h, w = rect
    grid[r:r + h, c:c + w] = color
    if hole is not None:
        if h == 3:  # horizontal bar (and 3x3): middle row, odd offsets
            grid[r + 1, c + 1:c + w - 1:2] = hole
        else:  # vertical bar: middle column, odd offsets
            grid[r + 1:r + h - 1:2, c + 1] = hole


def _orient_of(rect):
    _, _, h, w = rect
    if h == 3 and w == 3:
        return "S"
    return "H" if h == 3 else "V"


def _length_of(rect):
    return max(rect[2], rect[3])


def _line_class(v):
    dr, dc = v
    if dr == 0:
        return "row"
    if dc == 0:
        return "col"
    return "diag" if dr * dc > 0 else "anti"


def _components(grid, bg):
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or grid[r, c] == bg:
                continue
            color = int(grid[r, c])
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and grid[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append((color, cells))
    return comps


def _box(cells):
    rows = [y for y, _ in cells]
    cols = [x for _, x in cells]
    r, c = min(rows), min(cols)
    return r, c, max(rows) - r + 1, max(cols) - c + 1


def _is_bar_shape(h, w):
    return (h == 3 and w >= 3 and w % 2 == 1) or (w == 3 and h >= 3 and h % 2 == 1)


def _solid_bars(grid, bg):
    """Return [(color, rect)] if every non-background component is a solid odd bar."""
    bars = []
    for color, cells in _components(grid, bg):
        rect = _box(cells)
        if len(cells) != rect[2] * rect[3] or not _is_bar_shape(rect[2], rect[3]):
            return None
        bars.append((color, rect))
    return bars


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    order = sorted(range(len(values)), key=lambda i: (-int(counts[i]), int(values[i])))
    for i in order:
        if _solid_bars(grid, int(values[i])) is not None:
            return int(values[i])
    return int(values[order[0]])


def _families(bars):
    """Group bars by colour.  Returns (families, lone) or None if a group is not a progression."""
    groups = {}
    for color, rect in bars:
        groups.setdefault(color, []).append(rect)
    families, lone = [], []
    for color in sorted(groups):
        rects = sorted(groups[color])
        if len(rects) == 1:
            lone.append((color, rects[0]))
            continue
        if len({(h, w) for _, _, h, w in rects}) != 1:
            return None
        v = (rects[1][0] - rects[0][0], rects[1][1] - rects[0][1])
        for a, b in zip(rects, rects[1:]):
            if (b[0] - a[0], b[1] - a[1]) != v:
                return None
        families.append((color, rects, v))
    return families, lone


def _complete(rects, v, H, W):
    """All fitting terms before the first and after the last given bar."""
    dr, dc = v
    r0, c0, h, w = rects[0]
    before, t = [], -1
    while _fits(r0 + t * dr, c0 + t * dc, h, w, H, W):
        before.append((r0 + t * dr, c0 + t * dc, h, w))
        t -= 1
    rl, cl = rects[-1][:2]
    after, t = [], 1
    while _fits(rl + t * dr, cl + t * dc, h, w, H, W):
        after.append((rl + t * dr, cl + t * dc, h, w))
        t += 1
    return before[::-1], after


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.array(input_grid, dtype=int)
    H, W = grid.shape
    bg = _background(grid)
    bars = _solid_bars(grid, bg)
    if bars is None:
        return grid
    parsed = _families(bars)
    if parsed is None:
        return grid
    out = grid.copy()
    for color, rects, v in parsed[0]:
        before, after = _complete(rects, v, H, W)
        for rect in before + after:
            _paint(out, rect, color, hole=bg)
    return out


# ---------------------------------------------------------------------------
# pair analysis (used by the generator's acceptance test and by the validator)
# ---------------------------------------------------------------------------

def _analyze_pair(inp, out):
    inp = np.asarray(inp, dtype=int)
    out = np.asarray(out, dtype=int)
    H, W = inp.shape
    _require(out.shape == inp.shape, "output shape differs from input")
    bg = _background(inp)
    for grid, name in ((inp, "input"), (out, "output")):
        vals, cnts = np.unique(grid, return_counts=True)
        bg_count = int(cnts[list(vals).index(bg)])
        _require(all(bg_count > int(n) for v, n in zip(vals, cnts) if v != bg),
                 f"background is not the strict majority colour of the {name}")
    bars = _solid_bars(inp, bg)
    _require(bars is not None, "input contains a non-bar component")
    parsed = _families(bars)
    _require(parsed is not None, "a colour group is not an arithmetic progression of equal bars")
    families, lone = parsed
    _require(families, "input has no progression (every pair must change)")
    all_rects = []
    fam_info = []
    for color, rects, v in families:
        before, after = _complete(rects, v, H, W)
        _require(before or after, "a progression has no fitting continuation")
        terms = before + rects + after
        r0, c0, h, w = terms[0]
        rl, cl = terms[-1][:2]
        prev_term = (r0 - v[0], c0 - v[1], h, w)
        next_term = (rl + v[0], cl + v[1], h, w)
        partial = _meets_grid(*prev_term, H, W) or _meets_grid(*next_term, H, W)
        fam_info.append({
            "color": color, "orient": _orient_of(rects[0]), "length": _length_of(rects[0]),
            "v": v, "line": _line_class(v), "seeds": len(rects), "before": len(before),
            "after": len(after), "cont": len(before) + len(after),
            "middle": bool(before and after), "partial": bool(partial),
            "cont_rects": before + after,
        })
        all_rects.extend(terms)
    for color, rect in lone:
        all_rects.append(rect)
    for i in range(len(all_rects)):
        for j in range(i + 1, len(all_rects)):
            _require(_apart(all_rects[i], all_rects[j]), "bars touch or overlap in the output")
    _require(np.array_equal(solve(inp), out), "output does not follow the forecast-rail rule")
    return {"shape": (H, W), "bg": bg, "families": fam_info,
            "lone": [{"color": c, "orient": _orient_of(r), "length": _length_of(r)} for c, r in lone]}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------

_VTYPES = ("stack", "inline", "stair", "slide", "diag")
_VWEIGHTS = {
    "stacked": (0.45, 0.05, 0.30, 0.05, 0.15),
    "inline": (0.10, 0.40, 0.10, 0.30, 0.10),
    "stairs": (0.10, 0.05, 0.40, 0.15, 0.30),
    "any": (0.24, 0.16, 0.24, 0.16, 0.20),
}
_BANDS = {"compact": (9, 16), "medium": (12, 23), "large": (18, 31)}
_SHIFTS = ("reach", "crowd", "held_out_line", "length", "orientation_mix")


def _choice(rng, values, p=None):
    if p is None:
        return values[int(rng.integers(len(values)))]
    p = np.asarray(p, dtype=float)
    return values[int(rng.choice(len(values), p=p / p.sum()))]


def _check_counts(num_train, num_test):
    for name, value in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if not 2 <= int(num_train) <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if not 1 <= int(num_test) <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _vector(rng, orient, length, vtype):
    if vtype == "stack":
        across, along = int(_choice(rng, (4, 5, 6, 7), (0.35, 0.3, 0.2, 0.15))), 0
    elif vtype == "inline":
        across, along = 0, int(rng.integers(length + 1, length + 4))
    elif vtype == "stair":
        across, along = int(rng.integers(4, 7)), int(rng.integers(1, min(length, 5) + 1))
    elif vtype == "slide":
        across, along = int(rng.integers(1, 4)), int(rng.integers(length + 1, length + 4))
    else:  # diag: separated along both axes
        across, along = int(rng.integers(4, 7)), int(rng.integers(length + 1, length + 3))
    if across and along and rng.random() < 0.5:
        along = -along
    return (across, along) if orient == "H" else (along, across)


def _vector_for_line(rng, orient, length, line):
    """Displacement whose line class is ``line`` (row, col, diag or anti)."""
    axis = {"H": {"row": "inline", "col": "stack"}, "V": {"row": "stack", "col": "inline"}}
    if line in ("row", "col"):
        return _vector(rng, orient, length, axis[orient][line])
    dr, dc = _vector(rng, orient, length, _choice(rng, ("stair", "slide", "diag")))
    dc = abs(dc) if line == "diag" else -abs(dc)
    return abs(dr), dc


def _odd_length(rng, lo, hi):
    options = [L for L in (3, 5, 7, 9, 11, 13) if lo <= L <= hi]
    weights = {3: 0.12, 5: 0.30, 7: 0.25, 9: 0.15, 11: 0.10, 13: 0.08}
    return _choice(rng, options, [weights[L] for L in options])


def _place_family(rng, H, W, orient, length, v, n, middle, occupied, min_cont, max_cont):
    h, w = _dims(orient, length)
    dr, dc = v
    need = n + (2 if middle else 1)
    starts = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            if _fits(r - dr, c - dc, h, w, H, W):
                continue
            N = 1
            while _fits(r + N * dr, c + N * dc, h, w, H, W):
                N += 1
            if N >= need and min_cont <= N - n <= max_cont:
                starts.append((r, c, N))
    if not starts:
        return None
    order = [int(i) for i in rng.permutation(len(starts))]
    if rng.random() < 0.55:  # favour long runs so the boundary, not a habit, sets the count
        order.sort(key=lambda i: -starts[i][2])
    for idx in order[:30]:
        r, c, N = starts[idx]
        if middle:
            s = int(rng.integers(1, N - n))
        else:
            s = 0 if rng.random() < 0.5 else N - n
        terms = [(r + t * dr, c + t * dc, h, w) for t in range(N)]
        if all(_apart(a, b) for a in terms for b in occupied):
            return {"terms": terms, "seeds": terms[s:s + n], "cont": terms[:s] + terms[s + n:]}
    return None


def _make_pair(rng, spec):
    """Rejection-sample one pair satisfying the pair specification."""
    for _ in range(120):
        lo_f, hi_f = spec["fam"]
        n_fam = hi_f if rng.random() < 0.45 else int(rng.integers(lo_f, hi_f + 1))
        band = spec["band"] if spec["band"] != "mixed" else _choice(rng, ("compact", "medium", "large"))
        if n_fam >= 3 and band != "large":
            band = "large" if rng.random() < 0.7 else "medium"
        lo, hi = _BANDS[band]
        H, W = int(rng.integers(lo, hi)), int(rng.integers(lo, hi))
        bg = spec["bg"] if spec["bg"] is not None else _choice(rng, tuple(range(10)), [3] + [1] * 9)
        slots = []
        for o in sorted(spec["need_long"]):
            slots.append({"orient": o, "min_len": 5})
        if spec["need_len_ge"]:
            slots.append({"min_len": spec["need_len_ge"]})
        if spec["need_line"]:
            slots.append({"line": spec["need_line"]})
        if spec["need_cont_ge"]:
            slots.append({"min_cont": spec["need_cont_ge"]})
        if spec["need_middle"]:
            slots.append({"middle": True})
        if len(slots) > spec["fam"][1]:
            merged = {}  # requirements use distinct keys, so one family can carry all of them
            for slot in slots:
                merged.update(slot)
            slots = [merged]
        while len(slots) < n_fam:
            slots.append({})
        occupied, families, ok = [], [], True
        for slot in slots:
            placed = None
            goal = int(_choice(rng, (1, 2, 3, 4, 5), (0.3, 0.25, 0.2, 0.15, 0.1)))
            for attempt in range(25):
                orient = slot.get("orient", _choice(rng, spec["orients"]))
                lo_len, hi_len = slot.get("min_len", 3), min(spec["max_len"], 13)
                if lo_len > hi_len:
                    break
                length = _odd_length(rng, lo_len, hi_len)
                if "line" in slot:
                    v = _vector_for_line(rng, orient, length, slot["line"])
                else:
                    v = _vector(rng, orient, length, _choice(rng, _VTYPES, spec["vweights"]))
                if _line_class(v) in spec["forbid_lines"]:
                    continue
                n = int(_choice(rng, spec["seeds"]))
                middle = slot.get("middle", spec["allow_middle"] and rng.random() < 0.5)
                min_cont = slot.get("min_cont", 1)
                if attempt < 12:  # aim for the sampled run length first, then accept any
                    min_cont = max(min_cont, min(goal, spec["max_cont"]))
                placed = _place_family(rng, H, W, orient, length, v, n, middle, occupied,
                                       min_cont, spec["max_cont"])
                if placed is not None:
                    break
            if placed is None:
                ok = False
                break
            families.append(placed)
            occupied.extend(placed["terms"])
        if not ok:
            continue
        lone = []
        n_lone = int(rng.integers(spec["lone"][0], spec["lone"][1] + 1))
        for _ in range(n_lone):
            for _ in range(30):
                orient = _choice(rng, spec["orients"])
                length = _odd_length(rng, 3, min(spec["max_len"], 11))
                h, w = _dims(orient, length)
                if h > H or w > W:
                    continue
                rect = (int(rng.integers(0, H - h + 1)), int(rng.integers(0, W - w + 1)), h, w)
                if all(_apart(rect, b) for b in occupied):
                    lone.append(rect)
                    occupied.append(rect)
                    break
        if len(lone) < spec["lone"][0]:
            continue
        palette = [c for c in range(10) if c != bg]
        colors = [int(x) for x in rng.permutation(palette)[:len(families) + len(lone)]]
        inp = np.full((H, W), bg, dtype=int)
        out = np.full((H, W), bg, dtype=int)
        for fam, color in zip(families, colors):
            for rect in fam["seeds"]:
                _paint(inp, rect, color)
                _paint(out, rect, color)
            for rect in fam["cont"]:
                _paint(out, rect, color, hole=bg)
        for rect, color in zip(lone, colors[len(families):]):
            _paint(inp, rect, color)
            _paint(out, rect, color)
        try:
            info = _analyze_pair(inp, out)
        except AssertionError:
            continue
        if spec["need_partial"] and not any(f["partial"] for f in info["families"]):
            continue
        if spec["single_orient"]:
            orients = {f["orient"] for f in info["families"] if f["orient"] != "S"}
            orients |= {b["orient"] for b in info["lone"] if b["orient"] != "S"}
            if len(orients) > 1:
                continue
        return inp, out, info
    return None


def _base_spec(profile):
    return {
        "band": profile["band"], "bg": profile["bg"], "fam": (1, profile["max_f"]),
        "orients": profile["orients"], "need_long": set(), "need_len_ge": 0, "need_line": None,
        "need_cont_ge": 0, "need_middle": False, "allow_middle": profile["anchor"] == "free",
        "forbid_lines": set(), "max_len": 13, "max_cont": 99, "vweights": profile["vweights"],
        "seeds": profile["seeds"], "lone": (0, 2 if profile["lone"] else 0),
        "need_partial": False, "single_orient": False,
    }


def _profile(rng):
    orient_mode = _choice(rng, ("H", "V", "mixed"), (0.3, 0.25, 0.45))
    if orient_mode == "mixed":
        shift = _choice(rng, _SHIFTS, (0.17, 0.17, 0.17, 0.17, 0.32))
    else:
        shift = _choice(rng, _SHIFTS[:4])
    stable = rng.random() < 0.6
    return {
        "orient_mode": orient_mode,
        "orients": ("H", "V") if orient_mode == "mixed" else (orient_mode,),
        "shift": shift,
        "palette": "stable" if stable else "shifting",
        "bg": (0 if rng.random() < 0.45 else int(rng.integers(1, 10))) if stable else None,
        "vweights": _VWEIGHTS[_choice(rng, tuple(_VWEIGHTS))],
        "seeds": (3,) if rng.random() < 0.3 else (2, 3, 3, 4, 4),
        "anchor": "anchored" if rng.random() < 0.4 else "free",
        "max_f": int(_choice(rng, (1, 2, 3, 4), (0.2, 0.4, 0.3, 0.1))),
        "lone": rng.random() < 0.4,
        "band": _choice(rng, ("compact", "medium", "large", "mixed"), (0.1, 0.3, 0.35, 0.25)),
    }


def _train_specs(rng, profile, num_train):
    specs = [_base_spec(profile) for _ in range(num_train)]
    shift = profile["shift"]
    if shift == "crowd":
        for s in specs:
            s["fam"] = (1, min(profile["max_f"], 2))
    if shift == "reach":
        for s in specs:
            s["max_cont"] = 3
            if s["band"] == "large":
                s["band"] = "medium"
    if shift == "length":
        for s in specs:
            s["max_len"] = 7
    if shift == "held_out_line":
        profile["held_out"] = _choice(rng, LINE_CLASSES)
        for s in specs:
            s["forbid_lines"] = {profile["held_out"]}
    first = _choice(rng, profile["orients"])
    specs[0]["need_long"] = {first}
    specs[0]["need_partial"] = True
    if shift == "orientation_mix":
        second = "V" if first == "H" else "H"
        for i, s in enumerate(specs):
            s["orients"] = (first,) if i % 2 == 0 else (second,)
            s["single_orient"] = True
        specs[1]["need_long"] = {second}
    elif profile["orient_mode"] == "mixed":
        other = "V" if first == "H" else "H"
        specs[1]["need_long"] = specs[1]["need_long"] | {other}
    if profile["anchor"] == "free":
        specs[int(rng.integers(num_train))]["need_middle"] = True
    if profile["lone"]:
        specs[int(rng.integers(num_train))]["lone"] = (1, 2)
    for s in specs:
        if s["need_long"] and s["max_len"] < 5:
            s["max_len"] = 5
    return specs


def _test_spec(rng, profile, train_infos):
    spec = _base_spec(profile)
    fams = [f for info in train_infos for f in info["families"]]
    long_orients = {f["orient"] for f in fams if f["length"] >= 5}
    spec["orients"] = tuple(o for o in profile["orients"] if o in long_orients) or ("H",)
    spec["allow_middle"] = any(f["middle"] for f in fams)
    spec["seeds"] = tuple(sorted({f["seeds"] for f in fams}))
    spec["lone"] = (0, 2 if any(info["lone"] for info in train_infos) else 0)
    shift = profile["shift"]
    if shift == "reach":
        spec["need_cont_ge"] = max(f["cont"] for f in fams) + 1
        spec["band"] = "large"
    elif shift == "crowd":
        k = min(4, max(len(info["families"]) for info in train_infos) + 1)
        spec["fam"] = (k, k)
        spec["band"] = "large"
    elif shift == "held_out_line":
        spec["need_line"] = profile["held_out"]
        spec["fam"] = (1, max(profile["max_f"], 1))
    elif shift == "length":
        spec["max_len"] = 13
        spec["need_len_ge"] = max(9, max(f["length"] for f in fams) + 2)
        spec["band"] = _choice(rng, ("medium", "large"))
    elif shift == "orientation_mix":
        spec["orients"] = ("H", "V")
        spec["need_long"] = {"H", "V"}
        spec["fam"] = (2, max(2, profile["max_f"]))
    return spec


def _shift_kinds(train_infos, test_infos):
    """Measured train->test shifts present in a puzzle (grid evidence only)."""
    tf = [f for i in train_infos for f in i["families"]]
    kinds = set()
    for info in test_infos:
        sf = info["families"]
        if max(f["cont"] for f in sf) > max(f["cont"] for f in tf):
            kinds.add("reach")
        if len(sf) > max(len(i["families"]) for i in train_infos):
            kinds.add("crowd")
        if {f["line"] for f in sf} - {f["line"] for f in tf}:
            kinds.add("held_out_line")
        if max(f["length"] for f in sf) > max(f["length"] for f in tf):
            kinds.add("length")
        train_single = all(len({f["orient"] for f in i["families"] if f["orient"] != "S"}) <= 1
                           for i in train_infos)
        if train_single and {"H", "V"} <= {f["orient"] for f in sf}:
            kinds.add("orientation_mix")
    return kinds


def _coverage(train_infos, test_infos):
    """Raise unless the demonstrations expose every convention the tests need."""
    tf = [f for i in train_infos for f in i["families"]]
    sf = [f for i in test_infos for f in i["families"]]
    long_train = {f["orient"] for f in tf if f["length"] >= 5}
    _require(long_train, "training never shows a dashed rail of length >= 5")
    _require({f["orient"] for f in sf if f["length"] >= 5} <= long_train,
             "a test rail orientation is never demonstrated at length >= 5")
    _require(any(f["partial"] for f in tf), "training never refutes drawing a clipped partial bar")
    _require(len({f["cont"] for f in tf}) >= 2, "training continuation counts never vary")
    _require(max(max(f["before"], f["after"]) for f in tf) >= 2,
             "training never extends one side by two or more terms")
    _require(any(f["before"] for f in tf), "training never extends a progression backward in reading order")
    _require(any(f["after"] for f in tf), "training never extends a progression forward in reading order")
    if any(f["middle"] for f in sf):
        _require(any(f["middle"] for f in tf), "test extends both ways but training never does")
    if any(i["lone"] for i in test_infos):
        _require(any(i["lone"] for i in train_infos), "test has a lone bar but training never does")
    _require({f["seeds"] for f in sf} <= {f["seeds"] for f in tf},
             "a test seed count never appears in training")
    _require(_shift_kinds(train_infos, test_infos), "no measurable train/test shift")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(60):
        profile = _profile(rng)
        pairs, infos, ok = [], [], True
        for spec in _train_specs(rng, profile, num_train):
            made = _make_pair(rng, spec)
            if made is None:
                ok = False
                break
            pairs.append(made)
            infos.append(made[2])
        if not ok:
            continue
        test_pairs = []
        for _ in range(num_test):
            made = _make_pair(rng, _test_spec(rng, profile, infos))
            if made is None:
                ok = False
                break
            test_pairs.append(made)
        if not ok:
            continue
        keys = {p[0].tobytes() + bytes(str(p[0].shape), "ascii") for p in pairs + test_pairs}
        if len(keys) != num_train + num_test:
            continue
        try:
            _coverage(infos, [p[2] for p in test_pairs])
        except AssertionError:
            continue
        return {
            "train": [{"input": p[0].tolist(), "output": p[1].tolist()} for p in pairs],
            "test": [{"input": p[0].tolist(), "output": p[1].tolist()} for p in test_pairs],
        }
    raise RuntimeError(f"could not build a puzzle for seed {seed}")


# ---------------------------------------------------------------------------
# validation and self test
# ---------------------------------------------------------------------------

def _grid(value):
    arr = np.asarray(value)
    _require(arr.ndim == 2 and 1 <= arr.shape[0] <= 30 and 1 <= arr.shape[1] <= 30, "bad grid shape")
    _require(np.issubdtype(arr.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((arr >= 0) & (arr <= 9))), "grid colours must be 0..9")
    return arr.astype(int)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "pair counts")
    seen, infos = set(), {"train": [], "test": []}
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze_pair(inp, out)
            changed = out != inp
            _require(bool(changed.any()), "identity pair")
            # given bars are untouched: every changed cell lies in a forecast rail
            mask = np.zeros(inp.shape, dtype=bool)
            for fam in info["families"]:
                for r, c, h, w in fam["cont_rects"]:
                    mask[r:r + h, c:c + w] = True
                    _require(bool(np.all(inp[r:r + h, c:c + w] == info["bg"])),
                             "a forecast term covers a non-background input cell")
            _require(not bool((changed & ~mask).any()), "a cell outside the forecast rails changed")
            infos[split].append(info)
    _coverage(infos["train"], infos["test"])
    return True


def _stats(puzzle):
    train = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
    test = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
    return train, test


def self_test():
    shapes, lines, orients, seeds, fam_counts, conts, shifts, bgs = (set() for _ in range(8))
    lengths = set()
    varying_bg = middle = lone = stable_bg = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        again = generate_puzzle(seed)
        _require(again == puzzle, f"seed {seed} is not deterministic")
        train, test = _stats(puzzle)
        for info in train + test:
            shapes.add(info["shape"])
            bgs.add(info["bg"])
            fam_counts.add(len(info["families"]))
            lone += bool(info["lone"])
            for f in info["families"]:
                lines.add(f["line"])
                orients.add(f["orient"])
                seeds.add(f["seeds"])
                conts.add(f["cont"])
                lengths.add(f["length"])
                middle += f["middle"]
        if len({i["bg"] for i in train + test}) > 1:
            varying_bg += 1
        else:
            stable_bg += 1
        shifts |= _shift_kinds(train, test)
        for pair in puzzle["train"] + puzzle["test"]:
            _require(np.array_equal(solve(np.array(pair["input"])), np.array(pair["output"])), "solve mismatch")
    _require(len(shapes) >= 60, f"too few grid shapes ({len(shapes)})")
    _require(set(LINE_CLASSES) <= lines, "missing displacement line classes")
    _require({"H", "V", "S"} <= orients, "missing bar orientations")
    _require({2, 3, 4} <= seeds, "missing seed counts")
    _require({1, 2, 3} <= fam_counts, "missing family counts")
    _require(len(conts) >= 5 and len(lengths) >= 5, "continuation counts or lengths too uniform")
    _require(set(_SHIFTS) <= shifts, f"missing shift kinds {set(_SHIFTS) - shifts}")
    _require(len(bgs) >= 6 and varying_bg >= 5 and stable_bg >= 10, "palette roles too uniform")
    _require(middle >= 10 and lone >= 10, "bidirectional or lone-bar cases too rare")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (5, 5)):
        try:
            generate_puzzle(0, num_train=a, num_test=b)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted {a}/{b}")
    for a, b in ((2, 1), (10, 4), (5, 4)):
        validate_generated_puzzle(generate_puzzle(7, num_train=a, num_test=b))
    return True


if __name__ == "__main__":
    import json
    print(json.dumps(generate_puzzle(0), separators=(",", ":")))
