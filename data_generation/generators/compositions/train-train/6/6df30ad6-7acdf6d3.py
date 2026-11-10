"""Pour-by-size, paint-by-proximity.

Each solid placeholder silhouette is melted into the open cup whose cavity
capacity equals the silhouette's cell count (parent B: unary quantity and
capacity matching), and the poured liquid takes the color of the singleton
marker nearest (Euclidean, cellwise) to that silhouette (parent A: nearest
marker selection).  Silhouettes and markers are erased; cups stay in place;
unmatched cups stay empty.

Standard library + NumPy only.
"""
from __future__ import annotations

from itertools import product

import numpy as np

# ---------------------------------------------------------------------------
# counts


def _check_counts(num_train, num_test):
    num_train = 4 if num_train is None else num_train
    num_test = 1 if num_test is None else num_test
    for name, value in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    num_train, num_test = int(num_train), int(num_test)
    if not 2 <= num_train <= 10:
        raise ValueError("num_train must be in 2..10")
    if not 1 <= num_test <= 4:
        raise ValueError("num_test must be in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


# ---------------------------------------------------------------------------
# cup bank (open-top receptacles; capacity = number of cavity cells)


def _cup(left_moves, right_moves, top_width, pointed):
    depth = len(left_moves)
    left, right = 0, top_width + 1
    outline, cavity = set(), set()
    for row in range(depth):
        if right - left < 2:
            return None
        outline.update(((row, left), (row, right)))
        cavity.update((row, c) for c in range(left + 1, right))
        left += left_moves[row]
        right -= right_moves[row]
    if pointed:
        if left != right:
            return None
        outline.add((depth, left))
        style = "point"
    else:
        if right - left < 2:
            return None
        outline.update((depth, c) for c in range(left, right + 1))
        style = "slope" if any(left_moves) or any(right_moves) else "u"
    if not 1 <= len(cavity) <= 16:
        return None
    return {"outline": tuple(sorted(outline)), "cavity": tuple(sorted(cavity)),
            "h": depth + 1, "w": top_width + 2, "capacity": len(cavity), "style": style}


def _make_bank():
    bank, seen = {}, set()
    for depth in range(1, 5):
        for moves in product((0, 1), repeat=2 * (depth - 1)):
            lm = moves[:depth - 1] + (1,)
            rm = moves[depth - 1:] + (1,)
            shape = _cup(lm, rm, sum(lm) + sum(rm) - 1, True)
            if shape is not None and shape["outline"] not in seen:
                seen.add(shape["outline"])
                bank.setdefault(shape["capacity"], {}).setdefault("point", []).append(shape)
        for width in range(1, 8):
            for moves in product((0, 1), repeat=2 * depth):
                shape = _cup(moves[:depth], moves[depth:], width, False)
                if shape is not None and shape["outline"] not in seen:
                    seen.add(shape["outline"])
                    bank.setdefault(shape["capacity"], {}).setdefault(shape["style"], []).append(shape)
    return bank


_BANK = _make_bank()
STYLES = ("u", "slope", "point")


def _caps_for(styles):
    return sorted(c for c, d in _BANK.items() if any(s in d for s in styles))


# ---------------------------------------------------------------------------
# grid analysis shared by solve, validation and generation


def _components(mask):
    """8-connected components of a boolean mask (list of sorted cell lists)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                stack, comp = [(r, c)], []
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    comp.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                stack.append((ny, nx))
                comps.append(sorted(comp))
    return comps


def _row_cavity(cells):
    """Cells strictly between a component's leftmost and rightmost cell of each row."""
    rows = {}
    for r, c in cells:
        rows.setdefault(r, []).append(c)
    cs = set(cells)
    cav = []
    for r, cols in rows.items():
        for c in range(min(cols) + 1, max(cols)):
            if (r, c) not in cs:
                cav.append((r, c))
    return sorted(cav)


def _background(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _analyze(grid):
    grid = np.asarray(grid, dtype=int)
    bg = _background(grid)
    markers, cups, sils = [], [], []
    for color in sorted(int(v) for v in np.unique(grid) if v != bg):
        for comp in _components(grid == color):
            if len(comp) == 1:
                markers.append((comp[0][0], comp[0][1], color))
                continue
            cav = _row_cavity(comp)
            if cav:
                cups.append({"cells": comp, "cavity": cav, "capacity": len(cav), "color": color})
            else:
                sils.append({"cells": comp, "size": len(comp), "color": color})
    return {"bg": bg, "markers": markers, "cups": cups, "sils": sils}


def _d_euclid(dr, dc):
    return dr * dr + dc * dc


def _d_manhattan(dr, dc):
    return abs(dr) + abs(dc)


def _d_chebyshev(dr, dc):
    return max(abs(dr), abs(dc))


def _nearest_colors(cells, markers, metric=_d_euclid):
    """Set of marker colors attaining the minimum metric distance to the cell set."""
    arr = np.asarray(cells)
    best, colors = None, set()
    for r, c, color in markers:
        d = min(metric(int(r - y), int(c - x)) for y, x in arr)
        if best is None or d < best:
            best, colors = d, {color}
        elif d == best:
            colors.add(color)
    return colors


def _bbox_cells(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return [(r, c) for r in range(min(rs), max(rs) + 1) for c in range(min(cs), max(cs) + 1)]


def _plan(info):
    """For each silhouette: (matched cup index or None, fill color or None)."""
    plan = []
    for s in info["sils"]:
        matches = [i for i, cup in enumerate(info["cups"]) if cup["capacity"] == s["size"]]
        colors = _nearest_colors(s["cells"], info["markers"]) if info["markers"] else set()
        plan.append((matches[0] if len(matches) == 1 else None,
                     min(colors) if colors else None))
    return plan


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    info = _analyze(grid)
    out = np.full_like(grid, info["bg"])
    for cup in info["cups"]:
        for r, c in cup["cells"]:
            out[r, c] = cup["color"]
    for (cup_index, color) in _plan(info):
        if cup_index is None or color is None:
            continue
        for r, c in info["cups"][cup_index]["cavity"]:
            out[r, c] = color
    return out


# ---------------------------------------------------------------------------
# shortcut / alternative-rule diagnostics (computed from the input only)


def _diagnostics(grid):
    """Which plausible shortcuts give a different answer on this input."""
    info = _analyze(grid)
    plan = _plan(info)
    cups, markers = info["cups"], info["markers"]
    flags = {k: False for k in ("route_nearest", "color_by_cup", "manhattan", "chebyshev",
                                "bbox", "majority", "global_color", "marker_count")}
    counts = {}
    for _, _, color in markers:
        counts[color] = counts.get(color, 0) + 1
    top = max(counts.values())
    majority = {c for c, n in counts.items() if n == top}
    for s, (ci, color) in zip(info["sils"], plan):
        if ci is None or color is None:
            continue
        cells = np.asarray(s["cells"])
        # route to the nearest cup instead of the equal-capacity cup
        dists = []
        for cup in cups:
            foot = np.asarray(cup["cells"] + cup["cavity"])
            d = int(np.min(((cells[:, None, :] - foot[None, :, :]) ** 2).sum(-1)))
            dists.append(d)
        near = [i for i, d in enumerate(dists) if d == min(dists)]
        if near != [ci]:
            flags["route_nearest"] = True
        # paint with the marker nearest the destination cup
        cup = cups[ci]
        if _nearest_colors(cup["cells"] + cup["cavity"], markers) != {color}:
            flags["color_by_cup"] = True
        if _nearest_colors(s["cells"], markers, _d_manhattan) != {color}:
            flags["manhattan"] = True
        if _nearest_colors(s["cells"], markers, _d_chebyshev) != {color}:
            flags["chebyshev"] = True
        if _nearest_colors(_bbox_cells(s["cells"]), markers) != {color}:
            flags["bbox"] = True
        if majority != {color}:
            flags["majority"] = True
        if counts[color] == s["size"]:
            flags["marker_count"] = True
    # one global nearest marker for every silhouette
    if len(info["sils"]) > 1 and len({color for _, color in plan}) > 1:
        flags["global_color"] = True
    return flags


# ---------------------------------------------------------------------------
# generation


def _silhouette(rng, n):
    """Random 4-connected, row-convex polyomino with n cells (n >= 2)."""
    for _ in range(200):
        kmax = min(n, 5)
        k = int(rng.integers(1, kmax + 1))
        if n > 7 * k:
            continue
        cuts = sorted(rng.choice(np.arange(1, n), size=k - 1, replace=False).tolist()) if k > 1 else []
        lengths = [b - a for a, b in zip([0] + cuts, cuts + [n])]
        if max(lengths) > 7:
            continue
        cells, start = [], 0
        prev = None
        for r, ln in enumerate(lengths):
            if prev is not None:
                ps, pl = prev
                start = int(rng.integers(ps - ln + 1, ps + pl))
            cells.extend((r, start + i) for i in range(ln))
            prev = (start, ln)
        mc = min(c for _, c in cells)
        cells = [(r, c - mc) for r, c in cells]
        if rng.random() < 0.5:  # vertical flip keeps row-convexity
            mr = max(r for r, _ in cells)
            cells = [(mr - r, c) for r, c in cells]
        if max(c for _, c in cells) >= 7:
            continue
        if n >= 4 and (k == 1 and rng.random() < 0.6):
            continue  # prefer 2-D silhouettes over plain bars
        return sorted(cells)
    return [(0, i) for i in range(n)] if n <= 7 else None


def _palette(rng, n_marker_colors, fixed_bg=None):
    colors = list(range(10))
    rng.shuffle(colors)
    if fixed_bg is not None:
        colors.remove(fixed_bg)
        colors.insert(0, fixed_bg)
    return {"bg": colors[0], "outline": colors[1], "sil": colors[2],
            "markers": colors[3:3 + n_marker_colors]}


def _place(rng, occ, h, w, cells, tries=80):
    """Place cell set (relative coords) where its 1-dilation is free; returns offset."""
    ch = max(r for r, _ in cells) + 1
    cw = max(c for _, c in cells) + 1
    if ch > h or cw > w:
        return None
    for _ in range(tries):
        top = int(rng.integers(0, h - ch + 1))
        left = int(rng.integers(0, w - cw + 1))
        ok = True
        for r, c in cells:
            y0, y1 = max(0, top + r - 1), min(h, top + r + 2)
            x0, x1 = max(0, left + c - 1), min(w, left + c + 2)
            if occ[y0:y1, x0:x1].any():
                ok = False
                break
        if ok:
            return top, left
    return None


def _scene(rng, spec, palette):
    """One input/output pair from a spec, or None on failure."""
    h, w = spec["h"], spec["w"]
    caps = spec["matched"] + spec["distractors"]
    shapes = []
    for cap in caps:
        options = [s for s in spec["styles"] if s in _BANK.get(cap, {})]
        if not options:
            return None
        style = options[int(rng.integers(len(options)))]
        bank = _BANK[cap][style]
        shapes.append(bank[int(rng.integers(len(bank)))])
    sil_shapes = [_silhouette(rng, n) for n in spec["matched"]]
    if any(s is None for s in sil_shapes):
        return None
    occ = np.zeros((h, w), dtype=bool)
    grid = np.full((h, w), palette["bg"], dtype=int)
    objects = [("cup", s) for s in shapes] + [("sil", s) for s in sil_shapes]
    order = rng.permutation(len(objects))
    for idx in order:
        kind, shape = objects[idx]
        cells = list(shape["outline"]) + list(shape["cavity"]) if kind == "cup" else shape
        pos = _place(rng, occ, h, w, cells)
        if pos is None:
            return None
        top, left = pos
        for r, c in cells:
            occ[top + r, left + c] = True
        if kind == "cup":
            for r, c in shape["outline"]:
                grid[top + r, left + c] = palette["outline"]
        else:
            for r, c in shape:
                grid[top + r, left + c] = palette["sil"]
    free = []
    for r in range(h):
        for c in range(w):
            if not occ[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any():
                free.append((r, c))
    n_markers = spec["markers"]
    colors = palette["markers"]
    if len(free) < n_markers or n_markers < len(colors):
        return None
    chosen = []
    taken = np.zeros((h, w), dtype=bool)
    perm = rng.permutation(len(free))
    for i in perm:
        r, c = free[int(i)]
        if taken[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any():
            continue
        chosen.append((r, c))
        taken[r, c] = True
        if len(chosen) == n_markers:
            break
    if len(chosen) < n_markers:
        return None
    assign = list(colors) + [colors[int(rng.integers(len(colors)))] for _ in range(n_markers - len(colors))]
    rng.shuffle(assign)
    for (r, c), color in zip(chosen, assign):
        grid[r, c] = color
    out = solve(grid)
    return grid, out


def _pair_ok(grid, out, spec):
    """Structural checks for a generated pair; returns diagnostics or None."""
    info = _analyze(grid)
    counts = np.sort(np.unique(grid, return_counts=True)[1])
    if len(counts) > 1 and counts[-1] <= counts[-2]:
        return None
    if len(info["sils"]) != len(spec["matched"]) or len(info["cups"]) != len(spec["matched"]) + len(spec["distractors"]):
        return None
    caps = [c["capacity"] for c in info["cups"]]
    if len(set(caps)) != len(caps):
        return None
    for s, (ci, color) in zip(info["sils"], _plan(info)):
        if ci is None:
            return None
        if len(_nearest_colors(s["cells"], info["markers"])) != 1:
            return None
    flags = _diagnostics(grid)
    if flags["marker_count"]:
        return None
    return flags


def _spec(rng, kind, ctx):
    """Sample per-pair parameters. kind is 'train' or 'test'."""
    shift = ctx["shift"]
    styles = ctx["train_styles"] if kind == "train" else ctx["test_styles"]
    if kind == "train":
        n_sil = ctx["n_sil"]
    else:
        if shift == "count":
            n_sil = 3
        else:
            n_sil = int(rng.choice([1, 2, 2, 3])) if ctx["train_max_nsil"] >= 2 else 1
    n_dis = int(rng.choice([1, 1, 2])) if n_sil < 3 else 1
    caps_available = _caps_for(styles)
    matched_pool = [c for c in caps_available if 2 <= c <= 12]
    if kind == "test" and shift == "capacity":
        fresh = [c for c in matched_pool if c not in ctx["train_caps"]]
        if not fresh:
            return None
        first = int(rng.choice(fresh))
        rest = [c for c in matched_pool if c != first]
        matched = [first] + [int(x) for x in rng.choice(rest, size=n_sil - 1, replace=False)]
    else:
        matched = [int(x) for x in rng.choice(matched_pool, size=n_sil, replace=False)]
    dis_pool = [c for c in caps_available if c not in matched]
    distractors = [int(x) for x in rng.choice(dis_pool, size=n_dis, replace=False)]
    nobj = n_sil + n_dis + n_sil
    lo = max(12, 7 + 2 * nobj)
    if kind == "test" and shift == "scale":
        lo = max(lo, ctx["train_max_side"] + 3)
    lo = min(lo, 28)
    h = int(rng.integers(lo, min(30, lo + 10) + 1))
    w = int(rng.integers(lo, min(30, lo + 10) + 1))
    n_colors = int(rng.choice(ctx["color_counts"]))
    if kind == "test" and shift == "scale":
        n_colors = max(n_colors, 3)
    n_markers = int(rng.integers(max(n_colors, 2 + n_sil), max(n_colors, 2 + n_sil) + 3 + 3 * n_sil + 1))
    if kind == "test" and shift == "scale":
        n_markers += 4
    return {"h": h, "w": w, "matched": matched, "distractors": distractors, "styles": styles,
            "markers": n_markers, "n_colors": n_colors}


def _make_pair(rng, kind, ctx, need=None, test_pred=None, max_tries=400):
    for _ in range(max_tries):
        spec = _spec(rng, kind, ctx)
        if spec is None:
            continue
        palette = ctx["palette"](spec["n_colors"])
        res = _scene(rng, spec, palette)
        if res is None:
            continue
        grid, out = res
        flags = _pair_ok(grid, out, spec)
        if flags is None:
            continue
        if need and not all(flags[k] for k in need):
            continue
        if test_pred is not None and not test_pred(flags, spec):
            continue
        key = grid.tobytes() + bytes(grid.shape)
        if key in ctx["seen"]:
            continue
        ctx["seen"].add(key)
        return grid, out, flags, spec
    return None


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(50):
        puzzle = _attempt(rng, num_train, num_test)
        if puzzle is not None:
            return puzzle
    raise RuntimeError("generation failed")  # pragma: no cover


def _attempt(rng, num_train, num_test):
    shift = str(rng.choice(["style", "count", "capacity", "scale", "mixed"]))
    all_styles = list(STYLES)
    if shift == "style":
        held = str(rng.choice(all_styles))
        train_styles = [s for s in all_styles if s != held]
        if rng.random() < 0.5:
            train_styles = [train_styles[int(rng.integers(len(train_styles)))]]
        test_styles = [held]
    else:
        k = int(rng.integers(1, 4))
        train_styles = [str(s) for s in rng.choice(all_styles, size=k, replace=False)]
        test_styles = list(all_styles) if rng.random() < 0.5 else list(train_styles)
    palette_mode = str(rng.choice(["fixed", "fixed_bg", "per_pair"]))
    color_counts = [2, 3] if rng.random() < 0.5 else [2, 3, 4]
    fixed = _palette(rng, 4)
    fixed_bg = fixed["bg"]

    def palette(n):
        if palette_mode == "fixed":
            return {**fixed, "markers": fixed["markers"][:n]}
        if palette_mode == "fixed_bg":
            return _palette(rng, n, fixed_bg)
        return _palette(rng, n)

    # training silhouette counts: guarantee at least one multi-silhouette pair
    if shift == "count":
        nsil = [int(rng.choice([1, 2])) for _ in range(num_train)]
    else:
        nsil = [int(rng.choice([1, 1, 2, 2, 3])) for _ in range(num_train)]
    if max(nsil) < 2:
        nsil[int(rng.integers(num_train))] = 2
    ctx = {"shift": shift, "train_styles": train_styles, "test_styles": test_styles,
           "palette": palette, "color_counts": color_counts, "seen": set(),
           "n_sil": 1, "train_max_nsil": max(nsil), "train_caps": set(),
           "train_max_side": 0}
    # witness slots for the two cross-parent shortcuts
    slots = list(rng.permutation(num_train))
    needs = {int(slots[0]): ["route_nearest"], int(slots[1]): ["color_by_cup"]}
    train, train_flags = [], []
    for i in range(num_train):
        ctx["n_sil"] = nsil[i]
        res = _make_pair(rng, "train", ctx, need=needs.get(i))
        if res is None:
            return None
        grid, out, flags, spec = res
        train.append({"input": grid, "output": out})
        train_flags.append(flags)
        ctx["train_caps"].update(spec["matched"])
        ctx["train_max_side"] = max(ctx["train_max_side"], spec["h"], spec["w"])
    seen_alt = {k for f in train_flags for k, v in f.items() if v}

    def test_pred(flags, spec):
        for alt in ("manhattan", "chebyshev", "bbox", "majority", "global_color"):
            if flags[alt] and alt not in seen_alt:
                return False
        return flags["route_nearest"] or flags["color_by_cup"]

    test = []
    for _ in range(num_test):
        res = _make_pair(rng, "test", ctx, test_pred=test_pred)
        if res is None:
            return None
        grid, out, _, _ = res
        test.append({"input": grid, "output": out})
    return {"train": train, "test": test}


# ---------------------------------------------------------------------------
# validation and self test


def _validate_pair(pair):
    inp = np.asarray(pair["input"])
    out = np.asarray(pair["output"])
    assert inp.ndim == 2 and inp.shape == out.shape, "shape mismatch"
    assert all(1 <= n <= 30 for n in inp.shape), "bad size"
    assert inp.min() >= 0 and inp.max() <= 9 and out.min() >= 0 and out.max() <= 9, "bad colors"
    vals, counts = np.unique(inp, return_counts=True)
    top = np.sort(counts)
    assert len(top) == 1 or top[-1] > top[-2], "background not unique"
    info = _analyze(inp)
    bg = info["bg"]
    assert info["sils"], "no silhouette"
    assert len(info["cups"]) > len(info["sils"]), "need at least one distractor cup"
    assert info["markers"], "no markers"
    cup_colors = {c["color"] for c in info["cups"]}
    sil_colors = {s["color"] for s in info["sils"]}
    marker_colors = {m[2] for m in info["markers"]}
    assert len(cup_colors) == 1 and len(sil_colors) == 1, "role colors must be uniform"
    assert not (cup_colors & sil_colors) and not (marker_colors & (cup_colors | sil_colors)), "role colors overlap"
    assert len(marker_colors) >= 2, "need competing marker colors"
    caps = [c["capacity"] for c in info["cups"]]
    assert len(set(caps)) == len(caps), "capacities must be distinct"
    sizes = [s["size"] for s in info["sils"]]
    assert len(set(sizes)) == len(sizes), "silhouette sizes must be distinct"
    occupied = np.zeros(inp.shape, dtype=bool)
    for cup in info["cups"]:
        for r, c in cup["cavity"]:
            assert inp[r, c] == bg, "cavity not empty"
        # cup structure: each non-bottom row has exactly two wall cells
        rows = sorted({r for r, _ in cup["cells"]})
        assert rows == list(range(rows[0], rows[-1] + 1)), "cup rows not contiguous"
        for r in rows[:-1]:
            assert sum(1 for rr, _ in cup["cells"] if rr == r) == 2, "cup wall row malformed"
        assert all(r < rows[-1] for r, _ in cup["cavity"]), "cavity below base"
    for s in info["sils"]:
        assert s["size"] in caps, "silhouette with no equal-capacity cup"
        cs = set(s["cells"])
        # 4-connectivity
        stack, seen = [s["cells"][0]], {s["cells"][0]}
        while stack:
            y, x = stack.pop()
            for n in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if n in cs and n not in seen:
                    seen.add(n)
                    stack.append(n)
        assert seen == cs, "silhouette not 4-connected"
        assert len(_nearest_colors(s["cells"], info["markers"])) == 1, "nearest-marker tie between colors"
    # markers isolated (8-neighborhood) from every other non-background cell
    for r, c, _ in info["markers"]:
        win = inp[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
        assert int((win != bg).sum()) == 1, "marker not isolated"
    expected = solve(inp)
    assert np.array_equal(expected, out), "output disagrees with rule"
    assert not np.array_equal(inp, out), "identity pair"
    # every silhouette and marker erased; cups preserved
    for s in info["sils"]:
        for r, c in s["cells"]:
            assert out[r, c] == bg, "silhouette not erased"
    for cup in info["cups"]:
        for r, c in cup["cells"]:
            assert out[r, c] == cup["color"], "cup outline changed"
    del occupied
    return _diagnostics(inp)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad keys"
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "bad counts"
    keys = set()
    for pair in train + test:
        assert set(pair) == {"input", "output"}, "bad pair keys"
        k = np.asarray(pair["input"]).tobytes() + bytes(np.asarray(pair["input"]).shape)
        assert k not in keys, "duplicate input"
        keys.add(k)
    tflags = [_validate_pair(p) for p in train]
    xflags = [_validate_pair(p) for p in test]
    assert any(f["route_nearest"] for f in tflags), "training never rejects nearest-cup routing"
    assert any(f["color_by_cup"] for f in tflags), "training never rejects cup-proximity painting"
    assert max(len(_analyze(p["input"])["sils"]) for p in train) >= 2, "no multi-silhouette demo"
    seen = {k for f in tflags for k, v in f.items() if v}
    for f in xflags:
        assert f["route_nearest"] or f["color_by_cup"], "test pair does not exercise the composition"
        for alt in ("manhattan", "chebyshev", "bbox", "majority", "global_color"):
            assert not f[alt] or alt in seen, f"test depends on undemonstrated distinction {alt}"
        assert not f["marker_count"], "marker count mimics silhouette size"
    return True


def self_test():
    stats = {"styles": set(), "nsil": set(), "shapes": set(), "caps": set(), "bgs": set(),
             "witness_train": 0, "metric_train": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        again = generate_puzzle(seed)
        for a, b in zip(puzzle["train"] + puzzle["test"], again["train"] + again["test"]):
            assert np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        for pair in puzzle["train"] + puzzle["test"]:
            info = _analyze(pair["input"])
            stats["nsil"].add(len(info["sils"]))
            stats["shapes"].add(np.asarray(pair["input"]).shape)
            stats["bgs"].add(info["bg"])
            for cup in info["cups"]:
                stats["caps"].add(cup["capacity"])
        f = [_diagnostics(p["input"]) for p in puzzle["train"]]
        stats["metric_train"] += any(x["manhattan"] or x["chebyshev"] or x["bbox"] for x in f)
    assert stats["nsil"] >= {1, 2, 3}, stats["nsil"]
    assert len(stats["shapes"]) >= 30, len(stats["shapes"])
    assert len(stats["caps"]) >= 10, stats["caps"]
    assert len(stats["bgs"]) >= 6, stats["bgs"]
    for a, b in ((1, 1), (11, 1), (4, 0), (2, 2), (3, 4), (6, 5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return True


if __name__ == "__main__":
    self_test()
    print("self_test ok")
