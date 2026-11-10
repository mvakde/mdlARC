"""Broken-shear reconstruction: an ARC task family composed from 4364c1c4 and bb52a14b.

Each input holds one or more intact two-color *reference composites* (an upper
section stacked on a lower section that shares its full-width interface row, as
in 4364c1c4).  Elsewhere lie lone single-color *fragments*: exact translated
copies of one section of one reference, with the other section deleted (the
masked-copy idea of bb52a14b).  Single-color distractors that do not exactly
match a reference section (wrong shape, mirrored, wrong color) also appear.

Output: every fragment is completed with its missing section, using the
reference geometry, and then every complete composite (references and
completed fragments) is sheared: the upper section moves one column left and
the lower section one column right.  Distractors stay where they are.

Whether a lone fragment is an "upper" or a "lower" (hence which way it moves
and on which side its partner is rebuilt) is known only by matching it, by
color and exact shape, against the references.  When two references share a
color in opposite roles, shape alone decides.

Puzzle-level orientation convention: vertical stacks, or the same pairs rotated
a quarter-turn clockwise (side-by-side sections; the right section moves up and
the left section moves down), or a mix whose training shows both.  The solver
reads the orientation from the reference composites in the input.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

# ----------------------------------------------------------------------------
# Grid analysis shared by solve() and validate_generated_puzzle()
# ----------------------------------------------------------------------------

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(grid, bg):
    """8-connected components of non-background cells: list of (cells, colors)."""
    h, w = grid.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == bg:
                continue
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in _N8:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] != bg:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append((cells, sorted({int(grid[y, x]) for y, x in cells})))
    return comps


def _norm(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _orientation(grid, comps):
    """'v' if the two-color references are vertical stacks, 'r' if side-by-side."""
    for cells, colors in comps:
        if len(colors) != 2:
            continue
        a, b = colors
        ra = [r for r, c in cells if grid[r, c] == a]
        rb = [r for r, c in cells if grid[r, c] == b]
        if max(ra) < min(rb) or max(rb) < min(ra):
            return "v"
        return "r"
    return None


def _analyze_vertical(grid):
    """Analyze a grid whose references are vertical stacks.

    Returns dict with bg, refs, fragments (matched), unmatched single-color comps.
    """
    bg = _background(grid)
    comps = _components(grid, bg)
    refs, singles = [], []
    for cells, colors in comps:
        if len(colors) == 2:
            a, b = colors
            ra = [r for r, c in cells if grid[r, c] == a]
            rb = [r for r, c in cells if grid[r, c] == b]
            upper_color, lower_color = (a, b) if max(ra) < min(rb) else (b, a)
            r0 = min(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            upper = frozenset((r - r0, c - c0) for r, c in cells if grid[r, c] == upper_color)
            lower = frozenset((r - r0, c - c0) for r, c in cells if grid[r, c] == lower_color)
            hu = max(r for r, _ in upper) + 1
            refs.append({"origin": (r0, c0), "U": upper_color, "L": lower_color,
                         "upper": upper, "lower": lower, "hu": hu, "cells": cells})
        elif len(colors) == 1:
            singles.append((cells, colors[0]))
        else:
            refs.append({"invalid": True, "cells": cells})
    sections = {}
    for k, ref in enumerate(refs):
        if ref.get("invalid"):
            continue
        for role in ("upper", "lower"):
            color = ref["U"] if role == "upper" else ref["L"]
            key = (color, _norm(list(ref[role])))
            sections.setdefault(key, []).append((k, role))
    fragments, unmatched = [], []
    for cells, color in singles:
        key = (color, _norm(cells))
        if key in sections:
            k, role = sections[key][0]
            r0 = min(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            origin = (r0 - (0 if role == "upper" else refs[k]["hu"]), c0)
            fragments.append({"ref": k, "role": role, "origin": origin, "cells": cells,
                              "ambiguous": len(sections[key]) > 1, "color": color})
        else:
            unmatched.append((cells, color))
    return {"bg": bg, "refs": refs, "fragments": fragments, "unmatched": unmatched,
            "sections": sections}


def _solve_vertical(grid):
    info = _analyze_vertical(grid)
    bg = info["bg"]
    out = grid.copy()
    h, w = grid.shape
    composites = []
    for ref in info["refs"]:
        if ref.get("invalid"):
            continue
        for r, c in ref["cells"]:
            out[r, c] = bg
        composites.append((ref["origin"], ref))
    for frag in info["fragments"]:
        for r, c in frag["cells"]:
            out[r, c] = bg
        composites.append((frag["origin"], info["refs"][frag["ref"]]))
    for (r0, c0), ref in composites:
        for dr, dc in ref["upper"]:
            y, x = r0 + dr, c0 + dc - 1
            if 0 <= y < h and 0 <= x < w:
                out[y, x] = ref["U"]
        for dr, dc in ref["lower"]:
            y, x = r0 + dr, c0 + dc + 1
            if 0 <= y < h and 0 <= x < w:
                out[y, x] = ref["L"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.array(input_grid, dtype=int)
    bg = _background(grid)
    orient = _orientation(grid, _components(grid, bg))
    if orient == "r":
        # Side-by-side references: rotate counter-clockwise so the right section
        # becomes the upper one, solve the vertical rule, and rotate back.
        return np.rot90(_solve_vertical(np.rot90(grid, 1)), -1).copy()
    return _solve_vertical(grid)


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------


def _connected4(mask):
    cells = list(zip(*np.nonzero(mask)))
    if not cells:
        return False
    seen = {cells[0]}
    stack = [cells[0]]
    cellset = set(cells)
    while stack:
        y, x = stack.pop()
        for dy, dx in _N4:
            n = (y + dy, x + dx)
            if n in cellset and n not in seen:
                seen.add(n)
                stack.append(n)
    return len(seen) == len(cells)


def _keep_interface_component(mask, iface_row):
    h, w = mask.shape
    keep = np.zeros_like(mask)
    stack = [(iface_row, c) for c in range(w)]
    for y, x in stack:
        keep[y, x] = True
    while stack:
        y, x = stack.pop()
        for dy, dx in _N4:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not keep[ny, nx]:
                keep[ny, nx] = True
                stack.append((ny, nx))
    return keep


def _section_mask(rng, w, h, upper, style=None):
    """Section shape with a full-width interface row (bottom row if upper)."""
    m = np.ones((h, w), bool)
    if h >= 2:
        style = style or rng.choice(["solid", "taper", "notch", "random", "hole", "steps"],
                                    p=[0.17, 0.2, 0.2, 0.2, 0.08, 0.15])
        # Build in "interface at row 0" coordinates then flip for upper sections.
        if style == "taper":
            lo, hi = 0, w
            for row in range(1, h):
                if hi - lo > 1 and rng.random() < 0.75:
                    if rng.random() < 0.5:
                        lo += int(rng.integers(1, max(2, (hi - lo) // 2 + 1)))
                    else:
                        hi -= int(rng.integers(1, max(2, (hi - lo) // 2 + 1)))
                    lo = min(lo, hi - 1)
                m[row, :] = False
                m[row, lo:hi] = True
        elif style == "notch" and w >= 3:
            nw = int(rng.integers(1, w - 1))
            nc = int(rng.integers(1, w - nw)) if w - nw > 1 else 0
            depth = int(rng.integers(1, h))
            m[h - depth:, nc:nc + nw] = False
        elif style == "random":
            rnd = rng.random((h, w)) < 0.35
            rnd[0, :] = False
            m &= ~rnd
        elif style == "hole" and w >= 3 and h >= 3:
            hr = int(rng.integers(1, h - 1))
            hc = int(rng.integers(1, w - 1))
            m[hr, hc] = False
        elif style == "steps":
            for row in range(1, h):
                if rng.random() < 0.6:
                    side = rng.random() < 0.5
                    cut = int(rng.integers(1, max(2, w // 2 + 1)))
                    if side:
                        m[row, :cut] = False
                    else:
                        m[row, w - cut:] = False
        m[0, :] = True
        m = _keep_interface_component(m, 0)
        rows = np.nonzero(m.any(axis=1))[0]
        m = m[: rows.max() + 1]
    if upper:
        m = m[::-1]
    return m


def _mask_cells(mask):
    return frozenset((int(r), int(c)) for r, c in zip(*np.nonzero(mask)))


def _make_ref(rng, max_w=6, max_h=7):
    w = int(rng.integers(2, max_w + 1))
    hu = int(rng.integers(1, 5))
    hl = int(rng.integers(1, 5))
    while hu + hl > max_h:
        if hu > hl:
            hu -= 1
        else:
            hl -= 1
    up = _section_mask(rng, w, hu, True)
    lo = _section_mask(rng, w, hl, False)
    return {"w": w, "upper": up, "lower": lo, "hu": up.shape[0], "hl": lo.shape[0]}


def _distractor_masks(rng, ref, role, kind):
    """Return a single-color mask derived from a section (near miss) or a blob."""
    base = ref[role]
    if kind == "mirror":
        return base[:, ::-1].copy()
    if kind == "flip":
        return base[::-1, :].copy()
    if kind == "remove":
        cells = list(zip(*np.nonzero(base)))
        rng.shuffle(cells)
        for y, x in cells:
            m = base.copy()
            m[y, x] = False
            if m.sum() >= 1 and _connected4(m):
                return _trim(m)
        return None
    if kind == "add":
        h, w = base.shape
        big = np.zeros((h + 2, w + 2), bool)
        big[1:-1, 1:-1] = base
        cand = []
        for y in range(h + 2):
            for x in range(w + 2):
                if big[y, x]:
                    continue
                if any(0 <= y + dy < h + 2 and 0 <= x + dx < w + 2 and big[y + dy, x + dx]
                       for dy, dx in _N4):
                    cand.append((y, x))
        y, x = cand[int(rng.integers(len(cand)))]
        big[y, x] = True
        return _trim(big)
    if kind == "same":
        return base.copy()
    return None


def _trim(m):
    rows = np.nonzero(m.any(axis=1))[0]
    cols = np.nonzero(m.any(axis=0))[0]
    return m[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1].copy()


def _blob(rng):
    h = int(rng.integers(1, 4))
    w = int(rng.integers(1, 4))
    for _ in range(20):
        m = rng.random((h, w)) < 0.65
        if m.any() and _connected4(m):
            return _trim(m)
    m = np.zeros((1, 1), bool)
    m[0, 0] = True
    return m


class _Placer:
    def __init__(self, h, w):
        self.h, self.w = h, w
        self.occ = np.zeros((h, w), bool)

    def place(self, rng, rh, rw, tries=200):
        if rh > self.h or rw > self.w:
            return None
        for _ in range(tries):
            r = int(rng.integers(0, self.h - rh + 1))
            c = int(rng.integers(0, self.w - rw + 1))
            if not self.occ[max(0, r - 1): r + rh + 1, max(0, c - 1): c + rw + 1].any():
                self.occ[r: r + rh, c: c + rw] = True
                return r, c
        return None


def _pick_colors(rng, bg, n, exclude=()):
    pool = [c for c in range(10) if c != bg and c not in exclude]
    return [int(x) for x in rng.choice(pool, size=n, replace=False)]


def _make_pair(rng, cfg):
    """cfg keys: nrefs, roles (list of (ref_idx, role)), ndistract, need_color_distractor,
    need_shape_distractor, shared, bg, ref0_colors (optional), orient."""
    for _attempt in range(80):
        bg = cfg["bg"]
        nrefs = cfg["nrefs"]
        # ---- colors
        colors = []
        if cfg.get("ref0_colors"):
            colors.append(tuple(cfg["ref0_colors"]))
        else:
            colors.append(tuple(_pick_colors(rng, bg, 2)))
        for k in range(1, nrefs):
            prev_used = {c for pair in colors for c in pair}
            u0, l0 = colors[0]
            if cfg.get("shared") and k == 1:
                mode = cfg["shared"]
                if mode == "swap":
                    pair = (l0, u0)
                elif mode == "lower_up":
                    pair = (l0, _pick_colors(rng, bg, 1, exclude=(u0, l0))[0])
                else:
                    pair = (_pick_colors(rng, bg, 1, exclude=(u0, l0))[0], u0)
            else:
                # Same-role reuse is allowed; opposite-role reuse only via cfg["shared"].
                uppers = {p[0] for p in colors}
                lowers = {p[1] for p in colors}
                avail = [c for c in range(10) if c != bg and c not in prev_used]
                if len(avail) >= 2 and rng.random() < 0.7:
                    pair = tuple(int(x) for x in rng.choice(avail, size=2, replace=False))
                else:
                    u_pool = [c for c in range(10) if c != bg and c not in lowers]
                    u = int(rng.choice(u_pool))
                    l_pool = [c for c in range(10) if c not in (bg, u) and c not in uppers]
                    pair = (u, int(rng.choice(l_pool)))
            colors.append(pair)
        if len(set(colors)) != len(colors):
            continue
        # ---- shapes
        max_w = cfg.get("max_w", 6)
        refs = []
        for k in range(nrefs):
            ref = _make_ref(rng, max_w=max_w)
            ref["U"], ref["L"] = colors[k]
            refs.append(ref)
        sections = {}
        for k, ref in enumerate(refs):
            for role in ("upper", "lower"):
                key = (ref["U"] if role == "upper" else ref["L"], _mask_cells(ref[role]))
                sections.setdefault(key, []).append((k, role))
        if any(len(v) > 1 for v in sections.values()):
            continue
        # ---- distractors
        distractors = []
        noise_color = None
        used = {c for pair in colors for c in pair}
        free = [c for c in range(10) if c != bg and c not in used]
        if free and rng.random() < 0.4:
            noise_color = int(rng.choice(free))
        kinds = []
        if cfg.get("need_wrongcolor"):
            kinds.append("wrongcolor")
        if cfg.get("need_shape_distractor"):
            kinds.append(rng.choice(["mirror", "remove", "add", "wrongcolor", "flip"]))
        if cfg.get("need_color_distractor"):
            kinds.append(rng.choice(["remove", "add", "blobsec"]))
        while len(kinds) < cfg["ndistract"]:
            kinds.append(rng.choice(["mirror", "remove", "add", "wrongcolor", "blobsec", "blobsec",
                                     "noise", "flip"]))
        ok = True
        for kind in kinds:
            made = None
            for _t in range(25):
                k = int(rng.integers(nrefs))
                role = "upper" if rng.random() < 0.5 else "lower"
                ref = refs[k]
                color = ref["U"] if role == "upper" else ref["L"]
                if kind in ("mirror", "remove", "add", "flip"):
                    m = _distractor_masks(rng, ref, role, kind)
                elif kind == "wrongcolor":
                    m = ref[role].copy()
                    color = ref["L"] if role == "upper" else ref["U"]
                elif kind == "blobsec":
                    m = _blob(rng)
                else:
                    if noise_color is None:
                        m = _blob(rng)
                    else:
                        m = _blob(rng)
                        color = noise_color
                if m is None or not m.any() or not _connected4(m):
                    continue
                key = (int(color), _mask_cells(_trim(m)))
                if key in sections:
                    continue
                made = (int(color), _trim(m), kind)
                break
            if made is None:
                ok = False
                break
            distractors.append(made)
        if not ok:
            continue
        # ---- placement
        items = []
        for k, ref in enumerate(refs):
            items.append(("ref", k, None))
        for k, role in cfg["roles"]:
            items.append(("frag", k, role))
        for d in distractors:
            items.append(("dis", d, None))

        def dims(item):
            if item[0] in ("ref", "frag"):
                ref = refs[item[1]]
                return ref["hu"] + ref["hl"], ref["w"] + 2
            return item[1][1].shape

        area = sum((dims(it)[0] + 1) * (dims(it)[1] + 1) for it in items)
        maxh = max(dims(it)[0] for it in items)
        maxw = max(dims(it)[1] for it in items)
        dens = cfg.get("density", 2.6)
        target = area * dens
        aspect = float(np.exp(rng.uniform(-0.5, 0.5)))
        H = int(np.clip(round(np.sqrt(target * aspect)), max(maxh, 6), 30))
        W = int(np.clip(round(target / max(H, 1)), max(maxw, 6), 30))
        H = int(np.clip(H + rng.integers(-1, 3), max(maxh, 5), 30))
        W = int(np.clip(W + rng.integers(-1, 3), max(maxw, 5), 30))
        placed = None
        for _grow in range(6):
            placer = _Placer(H, W)
            order = sorted(range(len(items)), key=lambda i: -dims(items[i])[0] * dims(items[i])[1])
            pos = {}
            for i in order:
                p = placer.place(rng, *dims(items[i]))
                if p is None:
                    break
                pos[i] = p
            if len(pos) == len(items):
                placed = pos
                break
            H = min(30, H + 2)
            W = min(30, W + 2)
        if placed is None:
            continue
        inp = np.full((H, W), bg, int)
        out = np.full((H, W), bg, int)
        for i, it in enumerate(items):
            r, c = placed[i]
            if it[0] in ("ref", "frag"):
                ref = refs[it[1]]
                col = c + 1  # unsheared left column
                up_cells = list(zip(*np.nonzero(ref["upper"])))
                lo_cells = list(zip(*np.nonzero(ref["lower"])))
                if it[0] == "ref" or it[2] == "upper":
                    for y, x in up_cells:
                        inp[r + y, col + x] = ref["U"]
                if it[0] == "ref" or it[2] == "lower":
                    for y, x in lo_cells:
                        inp[r + ref["hu"] + y, col + x] = ref["L"]
                for y, x in up_cells:
                    out[r + y, col + x - 1] = ref["U"]
                for y, x in lo_cells:
                    out[r + ref["hu"] + y, col + x + 1] = ref["L"]
            else:
                color, m, _kind = it[1]
                for y, x in zip(*np.nonzero(m)):
                    inp[r + y, c + x] = color
                    out[r + y, c + x] = color
        if (inp == bg).sum() <= max((inp != bg).sum(), (out != bg).sum()):
            continue
        meta = {"colors": colors, "kinds": [d[2] for d in distractors], "roles": list(cfg["roles"]),
                "nrefs": nrefs}
        if cfg["orient"] == "r":
            inp = np.rot90(inp, -1).copy()
            out = np.rot90(out, -1).copy()
        # Safety: the input-only solver must reproduce the constructed output.
        if not np.array_equal(solve(inp), out):
            continue
        return inp, out, meta
    raise RuntimeError("could not construct pair")


def _check_counts(num_train, num_test):
    for value in (num_train, num_test):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10) or not (1 <= num_test <= 4) or num_train <= num_test:
        raise ValueError("need 2 <= num_train <= 10, 1 <= num_test <= 4 and num_train > num_test")


def _roles_for(rng, nrefs, nfrag, force=None):
    roles = []
    for i in range(nfrag):
        k = i % nrefs if i < nrefs else int(rng.integers(nrefs))
        role = "upper" if rng.random() < 0.5 else "lower"
        roles.append((k, role))
    if force:
        roles[0] = (roles[0][0], force)
    return roles


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = 4 if num_train is None else num_train
    num_test = 1 if num_test is None else num_test
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    orient_mode = str(rng.choice(["v", "r", "m"], p=[0.5, 0.2, 0.3]))
    multi = bool(rng.random() < 0.55)
    # Shared colors in opposite roles across two references: the role of a lone
    # fragment of that color is then decided by its shape alone.  If a puzzle
    # uses it at all, training demonstrates it on the multi-reference pair.
    shared_any = multi and bool(rng.random() < 0.7)
    total = num_train + num_test

    # Orientation per pair.
    if orient_mode == "v":
        orients = ["v"] * total
    elif orient_mode == "r":
        orients = ["r"] * total
    else:
        tr = [str(x) for x in rng.choice(["v", "r"], size=num_train)]
        tr[0], tr[1] = ("v", "r") if rng.random() < 0.5 else ("r", "v")
        te = [str(x) for x in rng.choice(["v", "r"], size=num_test)]
        orients = tr + te

    multi_train_index = int(rng.integers(1, num_train)) if multi else None
    shape_dis_index = int(rng.integers(0, num_train))
    pairs, seen = [], set()
    first_colors = None
    first_bg = None
    for i in range(total):
        is_test = i >= num_train
        bg = int(rng.choice(range(10), p=[0.4] + [0.6 / 9] * 9))
        if i == 1 and first_bg is not None and rng.random() < 0.5:
            # Role exchange: a previous section color becomes background.
            bg = int(rng.choice(first_colors))
        if not is_test:
            if not multi:
                nrefs = 1
            elif i == 0:
                nrefs = 1
            elif i == multi_train_index:
                nrefs = 2
            else:
                nrefs = int(rng.choice([1, 2, 3], p=[0.35, 0.45, 0.2]))
            nfrag = int(rng.integers(max(1, nrefs), nrefs + 3))
            if i == 0:
                force = "upper"
            elif i == 1:
                force = "lower"
            else:
                force = None
            ndis = int(rng.integers(1, 4))
            shared = shared_any and nrefs >= 2 and (i == multi_train_index or rng.random() < 0.5)
        else:
            nrefs = int(rng.choice([1, 2, 3], p=[0.2, 0.5, 0.3])) if multi else 1
            nfrag = int(rng.integers(max(2, nrefs), nrefs + 4))
            force = None
            ndis = int(rng.integers(1, 5))
            shared = shared_any and nrefs >= 2 and rng.random() < 0.8
        roles = _roles_for(rng, nrefs, nfrag, force)
        if nfrag >= 2 and len({r for _, r in roles}) == 1 and (is_test or rng.random() < 0.5):
            roles[-1] = (roles[-1][0], "lower" if roles[-1][1] == "upper" else "upper")
        if shared:
            shared = str(rng.choice(["swap", "lower_up", "upper_down"]))
            if shared == "lower_up" or (shared == "swap" and rng.random() < 0.5):
                need = [(0, "lower"), (1, "upper")]
            else:
                need = [(0, "upper"), (1, "lower")]
            roles = need + roles[2:] if len(roles) >= 2 else need
        else:
            shared = None
        ref0_colors = None
        if i == 1 and first_colors is not None and rng.random() < 0.6 and bg not in first_colors:
            ref0_colors = (first_colors[1], first_colors[0])  # same colors, swapped roles
        cfg = {"nrefs": nrefs, "roles": roles, "ndistract": ndis, "bg": bg,
               "need_color_distractor": i == 0 or rng.random() < 0.5,
               "need_wrongcolor": i == shape_dis_index,
               "need_shape_distractor": rng.random() < 0.4,
               "shared": shared, "ref0_colors": ref0_colors, "orient": orients[i],
               "density": float(rng.uniform(2.0, 3.6)), "max_w": int(rng.choice([4, 5, 6, 6]))}
        for _retry in range(30):
            try:
                inp, out, meta = _make_pair(rng, cfg)
            except RuntimeError:
                cfg["density"] = cfg["density"] + 0.5
                cfg["ndistract"] = max(1, cfg["ndistract"] - 1)
                continue
            key = inp.tobytes() + bytes(inp.shape)
            if key in seen:
                continue
            seen.add(key)
            break
        else:
            raise RuntimeError("generation failed")
        if i == 0:
            first_colors = meta["colors"][0]
            first_bg = bg
        pairs.append({"input": inp.astype(int), "output": out.astype(int)})
    return {"train": pairs[:num_train], "test": pairs[num_train:]}


# ----------------------------------------------------------------------------
# Validation and self test
# ----------------------------------------------------------------------------


def _analyze_any(grid):
    grid = np.asarray(grid, dtype=int)
    bg = _background(grid)
    orient = _orientation(grid, _components(grid, bg))
    g = np.rot90(grid, 1) if orient == "r" else grid
    return orient, _analyze_vertical(g), g


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    seen = set()
    train_roles, train_orients = set(), set()
    train_section_distractor = False
    train_wrongcolor = False
    train_role_conflict = test_role_conflict = False
    stats = Counter()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert set(pair) == {"input", "output"}
            inp = np.asarray(pair["input"])
            out = np.asarray(pair["output"])
            for g in (inp, out):
                assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
                assert np.issubdtype(g.dtype, np.integer) and g.min() >= 0 and g.max() <= 9
            assert inp.shape == out.shape
            key = inp.tobytes() + bytes(inp.shape)
            assert key not in seen, "duplicate input"
            seen.add(key)
            assert not np.array_equal(inp, out), "identity pair"
            assert np.array_equal(solve(inp), out), "solve mismatch"
            orient, info, g = _analyze_any(inp)
            assert orient in ("v", "r"), "no reference composite"
            refs = info["refs"]
            assert refs and not any(r.get("invalid") for r in refs), "bad reference"
            bg = info["bg"]
            # Every reference: two distinct colors stacked with a full shared interface.
            for ref in refs:
                hu = ref["hu"]
                up_cols = sorted(c for r, c in ref["upper"] if r == hu - 1)
                lo_cols = sorted(c for r, c in ref["lower"] if r == hu)
                w = max(c for _, c in list(ref["upper"]) + list(ref["lower"])) + 1
                assert up_cols == list(range(w)) == lo_cols, "interface not full width"
                assert _connected4(_cells_to_mask(ref["upper"])), "upper section disconnected"
                assert _connected4(_cells_to_mask(_shift(ref["lower"], -hu))), "lower disconnected"
            # Each section key unique, so lone fragments have exactly one role.
            assert all(len(v) == 1 for v in info["sections"].values()), "ambiguous section"
            assert info["fragments"], "no fragment to rebuild"
            # Fragment completion area must be background and inside the grid.
            gh, gw = g.shape
            for frag in info["fragments"]:
                ref = refs[frag["ref"]]
                r0, c0 = frag["origin"]
                other = ref["lower"] if frag["role"] == "upper" else ref["upper"]
                for dr, dc in other:
                    y, x = r0 + dr, c0 + dc
                    assert 0 <= y < gh and 0 <= x < gw and g[y, x] == bg, "blocked completion"
                assert c0 >= 1 and c0 + max(c for _, c in ref["upper"]) + 2 <= gw, "no shear room"
            for ref in refs:
                r0, c0 = ref["origin"]
                assert c0 >= 1 and c0 + max(c for _, c in ref["upper"]) + 2 <= gw, "no shear room"
            # Distractors are preserved in the output.
            og = np.rot90(out, 1) if orient == "r" else out
            for cells, color in info["unmatched"]:
                assert all(og[r, c] == color for r, c in cells), "distractor changed"
            roles = {f["role"] for f in info["fragments"]}
            uppers = {r["U"] for r in refs}
            lowers = {r["L"] for r in refs}
            sec_colors = uppers | lowers
            sec_shapes = {key[1] for key in info["sections"]}
            # A fragment whose color is an upper color of one reference and a lower color
            # of another: its role (and shear direction) follows from its shape only.
            conflict = any(f["color"] in uppers and f["color"] in lowers for f in info["fragments"])
            wrongcolor = any(_norm(cells) in sec_shapes for cells, _ in info["unmatched"])
            if split == "train":
                train_roles |= roles
                train_orients.add(orient)
                train_role_conflict |= conflict
                train_wrongcolor |= wrongcolor
                if any(color in sec_colors for _, color in info["unmatched"]):
                    train_section_distractor = True
            else:
                assert orient in train_orients, "test orientation not demonstrated"
                test_role_conflict |= conflict
            stats[(split, "refs", len(refs))] += 1
    assert train_roles == {"upper", "lower"}, "training must show both fragment roles"
    assert train_section_distractor, "training must show an unmoved section-colored distractor"
    assert train_wrongcolor, "training must show a section shape in a non-matching color"
    assert train_role_conflict or not test_role_conflict, "shared-color roles not demonstrated"
    return True


def _cells_to_mask(cells):
    cells = list(cells)
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    m = np.zeros((h, w), bool)
    for r, c in cells:
        m[r, c] = True
    return m


def _shift(cells, dr):
    return frozenset((r + dr, c) for r, c in cells)


def _puzzle_stats(puzzle):
    s = Counter()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            orient, info, _ = _analyze_any(pair["input"])
            s[f"orient_{orient}"] += 1
            s[f"refs_{len(info['refs'])}"] += 1
            s["fragments"] += len(info["fragments"])
            for f in info["fragments"]:
                s[f"role_{f['role']}"] += 1
            colors_u = Counter(r["U"] for r in info["refs"])
            colors_l = Counter(r["L"] for r in info["refs"])
            if set(colors_u) & set(colors_l):
                s["shared_opposite_role"] += 1
            s["distractors"] += len(info["unmatched"])
            s[f"shape_{np.asarray(pair['input']).shape}"] += 1
    return s


def self_test():
    seeds = list(range(42))
    total = Counter()
    sigs = set()
    for seed in seeds:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        q = generate_puzzle(seed)
        for split in ("train", "test"):
            for a, b in zip(p[split], q[split]):
                assert np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        sigs.add(np.asarray(p["test"][0]["input"]).tobytes())
        total.update(_puzzle_stats(p))
    assert len(sigs) == len(seeds)
    for key in ("orient_v", "orient_r", "refs_1", "refs_2", "refs_3", "role_upper", "role_lower",
                "shared_opposite_role"):
        assert total[key] > 0, f"missing variation {key}"
    shapes = [k for k in total if k.startswith("shape_")]
    assert len(shapes) >= 30, "too few grid shapes"
    for a, b in ((2, 1), (10, 4), (5, 3)):
        p = generate_puzzle(7, num_train=a, num_test=b)
        assert len(p["train"]) == a and len(p["test"]) == b
        validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (3, 4), (2, 2), (6, 5)):
        try:
            generate_puzzle(0, num_train=a, num_test=b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return True


if __name__ == "__main__":
    self_test()
    print("self_test passed")
