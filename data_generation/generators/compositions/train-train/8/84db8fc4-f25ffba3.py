"""Mirror-sealed flooding: complete a half-built (or quarter-built) symmetric
wall plan, then let the 'sea' color of the blank part flood every floor cell
that can reach the outer border orthogonally.

Family conventions (fixed, visible in every input):
* 0 is the floor (background) colour.
* The blank destination region is one uniform non-zero colour V (the sea).
  It is either one congruent half of the grid (top/bottom/left/right) or the
  three quadrants other than one source quadrant.  V never appears in the
  source region.
* Every other non-zero colour is an impermeable, preserved wall colour.

Rule: mirror the source region across the seam(s) so the whole grid becomes
symmetric (floor cells are copied too), then flood every floor cell that is
four-neighbour connected to the outer grid border with V.  Floor cells that
cannot reach the border stay 0.  The seam is NOT a border: pockets that open
only onto the seam are sealed by their own mirror image.
"""
from __future__ import annotations

from collections import deque

import numpy as np

FLOOR = 0
HALF_SIDES = ("top", "bottom", "left", "right")
QUAD_SIDES = ("tl", "tr", "bl", "br")


# ----------------------------------------------------------------------------
# Shared geometry
# ----------------------------------------------------------------------------

def _source_mask(shape, name):
    h, w = shape
    m = np.zeros(shape, dtype=bool)
    if name == "top":
        m[: h // 2, :] = True
    elif name == "bottom":
        m[h // 2:, :] = True
    elif name == "left":
        m[:, : w // 2] = True
    elif name == "right":
        m[:, w // 2:] = True
    else:
        rows = slice(0, h // 2) if name[0] == "t" else slice(h // 2, h)
        cols = slice(0, w // 2) if name[1] == "l" else slice(w // 2, w)
        m[rows, cols] = True
    return m


def _candidate_names(shape):
    h, w = shape
    names = []
    if h % 2 == 0 and h >= 2:
        names += ["top", "bottom"]
    if w % 2 == 0 and w >= 2:
        names += ["left", "right"]
    if h % 2 == 0 and w % 2 == 0 and h >= 2 and w >= 2:
        names += list(QUAD_SIDES)
    return names


def _detect(grid):
    """Return list of (name, sea_colour) whose blank part is uniform non-zero
    colour V and whose source part is free of V."""
    found = []
    for name in _candidate_names(grid.shape):
        src = _source_mask(grid.shape, name)
        blank = grid[~src]
        v = int(blank[0])
        if v == FLOOR or not np.all(blank == v):
            continue
        if np.any(grid[src] == v):
            continue
        found.append((name, v))
    return found


def _complete(grid, name):
    """Mirror the source region over the seam(s) into the whole grid."""
    h, w = grid.shape
    if name == "top":
        s = grid[: h // 2]
        return np.vstack([s, s[::-1]])
    if name == "bottom":
        s = grid[h // 2:]
        return np.vstack([s[::-1], s])
    if name == "left":
        s = grid[:, : w // 2]
        return np.hstack([s, s[:, ::-1]])
    if name == "right":
        s = grid[:, w // 2:]
        return np.hstack([s[:, ::-1], s])
    rows = slice(0, h // 2) if name[0] == "t" else slice(h // 2, h)
    cols = slice(0, w // 2) if name[1] == "l" else slice(w // 2, w)
    q = grid[rows, cols]
    if name[0] == "b":
        q = q[::-1]
    if name[1] == "r":
        q = q[:, ::-1]
    top = np.hstack([q, q[:, ::-1]])
    return np.vstack([top, top[::-1]])


def _flood_from_border(passable):
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    queue = deque()
    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and passable[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def _components(mask):
    h, w = mask.shape
    labels = np.full(mask.shape, -1, dtype=int)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and labels[r, c] < 0:
                idx = len(comps)
                cells = []
                labels[r, c] = idx
                queue = deque([(r, c)])
                while queue:
                    y, x = queue.popleft()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] < 0:
                            labels[ny, nx] = idx
                            queue.append((ny, nx))
                comps.append(cells)
    return labels, comps


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    found = _detect(grid)
    if not found:
        raise ValueError("No uniform blank destination region found")
    # Prefer the largest blank region if several were ever possible.
    found.sort(key=lambda item: -int((~_source_mask(grid.shape, item[0])).sum()))
    name, sea = found[0]
    full = _complete(grid, name)
    exterior = _flood_from_border(full == FLOOR)
    out = full.copy()
    out[exterior] = sea
    return out


# ----------------------------------------------------------------------------
# Analysis helpers (used by validation and self-test)
# ----------------------------------------------------------------------------

def _analyze(inp, out=None):
    inp = np.asarray(inp, dtype=int)
    found = _detect(inp)
    name, sea = found[0]
    src = _source_mask(inp.shape, name)
    full = _complete(inp, name)
    floor = full == FLOOR
    exterior = _flood_from_border(floor)
    enclosed = floor & ~exterior
    labels, comps = _components(floor)
    seam_pockets = interior_pockets = seam_exterior = 0
    for cells in comps:
        rr = np.array([c[0] for c in cells])
        cc = np.array([c[1] for c in cells])
        in_src = src[rr, cc]
        crosses = bool(in_src.any() and (~in_src).any())
        ext = bool(exterior[rr[0], cc[0]])
        if ext and crosses:
            seam_exterior += 1
        if not ext and crosses:
            seam_pockets += 1
        if not ext and in_src.all():
            interior_pockets += 1
    h, w = full.shape
    diag = False
    ys, xs = np.nonzero(enclosed)
    for y, x in zip(ys, xs):
        for dy, dx in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and exterior[ny, nx]:
                diag = True
                break
        if diag:
            break
    walls = sorted(set(int(v) for v in np.unique(full)) - {FLOOR})
    # Shortcut: treat the seam as border (flood the source alone, then mirror).
    src_rows = np.nonzero(src.any(axis=1))[0]
    src_cols = np.nonzero(src.any(axis=0))[0]
    sub = inp[src_rows[0]:src_rows[-1] + 1, src_cols[0]:src_cols[-1] + 1]
    sub_ext = _flood_from_border(sub == FLOOR)
    sub_out = sub.copy()
    sub_out[sub_ext] = sea
    shortcut_in = inp.copy()
    shortcut_in[src_rows[0]:src_rows[-1] + 1, src_cols[0]:src_cols[-1] + 1] = sub_out
    seam_border_shortcut = _complete(shortcut_in, name)
    # Shortcut: flood the raw input (blank sea leaks through the seam).
    raw = inp.copy()
    raw[_flood_from_border((inp == FLOOR) | (inp == sea))] = sea
    return {
        "mode": "quad" if name in QUAD_SIDES else "half",
        "side": name,
        "axis": "quad" if name in QUAD_SIDES else ("h" if name in ("top", "bottom") else "v"),
        "sea": sea,
        "walls": walls,
        "full": full,
        "exterior": int(exterior.sum()),
        "enclosed": int(enclosed.sum()),
        "seam_pockets": seam_pockets,
        "interior_pockets": interior_pockets,
        "seam_exterior": seam_exterior,
        "diag": diag,
        "mirror_only": full,
        "seam_border_shortcut": seam_border_shortcut,
        "raw_flood": raw,
        "n_candidates": len(found),
    }


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must be in {low}..{high}")
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _place_free(occ, r0, r1, c0, c1):
    """True if rectangle rows r0..r1, cols c0..c1 (inclusive) plus a 1-cell
    margin is unoccupied."""
    h, w = occ.shape
    return not occ[max(0, r0 - 1):min(h, r1 + 2), max(0, c0 - 1):min(w, c1 + 2)].any()


def _stamp(rng, wall, occ, r0, r1, c0, c1, open_bottom, open_right,
           gate, island, interior_noise):
    """Hollow rectangle; bottom/right sides optionally left open (seam side)."""
    h, w = wall.shape
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_top = r == r0
            on_left = c == c0
            on_bottom = r == r1 and not open_bottom
            on_right = c == c1 and not open_right
            if on_top or on_left or on_bottom or on_right:
                wall[r, c] = True
            else:
                wall[r, c] = rng.random() < interior_noise
            occ[r, c] = True
    # Keep the seam-row / seam-column interior open so the pocket reaches the seam.
    if open_bottom:
        wall[r1, c0 + 1:(c1 if not open_right else c1 + 1)] = False
    if open_right:
        wall[r0 + 1:(r1 if not open_bottom else r1 + 1), c1] = False
    if island:
        ir0, ir1 = r0 + 1, (r1 - 1 if not open_bottom else r1)
        ic0, ic1 = c0 + 1, (c1 - 1 if not open_right else c1)
        if ir1 - ir0 >= 2 and ic1 - ic0 >= 2:
            rr = int(rng.integers(ir0 + 1, ir1))
            cc = int(rng.integers(ic0 + 1, ic1))
            if not ((open_bottom and rr == r1) or (open_right and cc == c1)):
                wall[rr, cc] = True
    if gate:
        # remove one non-corner cell from the top or left wall
        top_end = c1 if open_right else c1 - 1
        left_end = r1 if open_bottom else r1 - 1
        choices = [(r0, c) for c in range(c0 + 1, top_end + 1)]
        choices += [(r, c0) for r in range(r0 + 1, left_end + 1)]
        if choices:
            gr, gc = choices[int(rng.integers(len(choices)))]
            wall[gr, gc] = False


def _canonical_source(rng, h, w, quad, style, force_showcase):
    """Barrier mask of a canonical source region with the seam at the bottom
    (and also at the right in quadrant mode).  Outer sides: top, left (and
    right in half mode)."""
    if style == "sparse":
        p = float(rng.uniform(0.04, 0.16))
    elif style == "maze":
        p = float(rng.uniform(0.38, 0.56))
    else:  # rooms
        p = float(rng.uniform(0.12, 0.32))
    wall = rng.random((h, w)) < p
    occ = np.zeros((h, w), dtype=bool)
    structs = []

    def try_cup(open_right_cup=False, corner=False, gate=False):
        for _ in range(30):
            if corner:
                ch = int(rng.integers(2, min(6, h) + 1))
                cw = int(rng.integers(2, min(6, w) + 1))
                r0, r1, c0, c1 = h - ch, h - 1, w - cw, w - 1
                ob, orr = True, True
            elif open_right_cup:
                cw = int(rng.integers(2, min(6, w) + 1))
                ch = int(rng.integers(3, min(7, h) + 1))
                c0, c1 = w - cw, w - 1
                r0 = int(rng.integers(0, h - ch + 1))
                r1 = r0 + ch - 1
                if r1 >= h - 1:
                    continue
                ob, orr = False, True
            else:
                ch = int(rng.integers(2, min(6, h) + 1))
                cw = int(rng.integers(3, min(8, w if quad else w) + 1))
                r0, r1 = h - ch, h - 1
                hi = (w - cw) if not quad else (w - cw - 1)
                if hi < 0:
                    continue
                c0 = int(rng.integers(0, hi + 1))
                c1 = c0 + cw - 1
                ob, orr = True, False
            if r0 < 0 or c0 < 0 or not _place_free(occ, r0, r1, c0, c1):
                continue
            _stamp(rng, wall, occ, r0, r1, c0, c1, ob, orr, gate,
                   rng.random() < 0.35, 0.0 if rng.random() < 0.6 else 0.15)
            structs.append((r0, r1, c0, c1))
            return True
        return False

    def try_room(gate=False):
        for _ in range(30):
            rh = int(rng.integers(3, min(7, h) + 1))
            rw = int(rng.integers(3, min(8, w) + 1))
            if rh > h or rw > w:
                continue
            r0 = int(rng.integers(0, h - rh + 1))
            c0 = int(rng.integers(0, w - rw + 1))
            r1, c1 = r0 + rh - 1, c0 + rw - 1
            if not _place_free(occ, r0, r1, c0, c1):
                continue
            _stamp(rng, wall, occ, r0, r1, c0, c1, False, False, gate,
                   rng.random() < 0.35, 0.0 if rng.random() < 0.6 else 0.15)
            structs.append((r0, r1, c0, c1))
            return True
        return False

    n_cups = int(rng.integers(1, 3 if style != "rooms" else 4))
    if force_showcase:
        n_cups = max(n_cups, 1)
    for i in range(n_cups):
        if quad:
            kind = int(rng.integers(3))
            try_cup(open_right_cup=(kind == 1), corner=(kind == 2),
                    gate=(i > 0 and rng.random() < 0.4))
        else:
            try_cup(gate=(i > 0 and rng.random() < 0.4))
    n_rooms = int(rng.integers(0, 3)) + (1 if force_showcase else 0)
    for _ in range(n_rooms):
        try_room(gate=rng.random() < 0.3)
    if force_showcase:
        # An extra cup with a one-cell gate if there is room.
        try_cup(gate=True)
    # Clear a little floor along the outer border so the sea can enter.
    outer = np.zeros((h, w), dtype=bool)
    outer[0, :] = True
    outer[:, 0] = True
    if not quad:
        outer[:, w - 1] = True
    if not (~wall & outer).any():
        r, c = np.argwhere(outer & ~occ)[0] if (outer & ~occ).any() else (0, 0)
        wall[r, c] = False
    return wall, structs


def _colour_walls(rng, wall, structs, colours, appearance):
    h, w = wall.shape
    out = np.zeros((h, w), dtype=int)
    k = len(colours)
    if k == 1 or appearance == "single":
        out[wall] = colours[0]
        return out
    weights = rng.dirichlet(np.ones(k) * (0.6 if rng.random() < 0.5 else 2.0))
    if appearance == "mottled":
        pick = rng.choice(k, size=(h, w), p=weights)
        out[wall] = np.array(colours)[pick][wall]
    elif appearance == "objects":
        base = rng.choice(k, size=(h, w), p=weights)
        cols = np.array(colours)[base]
        for (r0, r1, c0, c1) in structs:
            cols[r0:r1 + 1, c0:c1 + 1] = colours[int(rng.integers(k))]
        out[wall] = cols[wall]
    else:  # regions (voronoi patches)
        n = int(rng.integers(2, 5))
        seeds_r = rng.integers(0, h, size=n)
        seeds_c = rng.integers(0, w, size=n)
        lab = rng.integers(0, k, size=n)
        lab[: min(n, k)] = rng.permutation(k)[: min(n, k)]
        yy, xx = np.mgrid[0:h, 0:w]
        d = (yy[None] - seeds_r[:, None, None]) ** 2 + (xx[None] - seeds_c[:, None, None]) ** 2
        cols = np.array(colours)[lab[np.argmin(d, axis=0)]]
        out[wall] = cols[wall]
    return out


SIZE_CLASSES = {
    "half": {"small": ((3, 5), (5, 9)), "medium": ((5, 9), (8, 15)), "large": ((9, 15), (13, 30))},
    "quad": {"small": ((3, 5), (3, 6)), "medium": ((5, 9), (5, 10)), "large": ((9, 15), (9, 15))},
}


def _make_pair(rng, mode, side, size_class, sea, wall_colours, style, appearance, showcase):
    lo_h, lo_w = SIZE_CLASSES[mode][size_class]
    for _ in range(200):
        h = int(rng.integers(lo_h[0], lo_h[1] + 1))
        w = int(rng.integers(lo_w[0], lo_w[1] + 1))
        if mode == "half" and side in ("left", "right") and w > 30:
            continue
        wall, structs = _canonical_source(rng, h, w, mode == "quad", style, showcase)
        canon = _colour_walls(rng, wall, structs, wall_colours, appearance)
        if mode == "half":
            full = np.vstack([canon, canon[::-1]])
            if side in ("left", "right"):
                full = full.T.copy()
        else:
            top = np.hstack([canon, canon[:, ::-1]])
            full = np.vstack([top, top[::-1]])
        if max(full.shape) > 30:
            continue
        src = _source_mask(full.shape, side)
        inp = full.copy()
        inp[~src] = sea
        if not (inp[src] == FLOOR).any():
            continue
        found = _detect(inp)
        if len(found) != 1 or found[0] != (side, sea):
            continue
        out = solve(inp)
        info = _analyze(inp, out)
        if info["seam_pockets"] < 1 or info["exterior"] < 3:
            continue
        if showcase and not (info["interior_pockets"] >= 1 and info["seam_exterior"] >= 1 and info["diag"]):
            continue
        return inp, out, info
    return None


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _attempt in range(50):
        puzzle = _try_generate(rng, num_train, num_test)
        if puzzle is not None:
            return puzzle
    raise RuntimeError("generation failed")


def _try_generate(rng, num_train, num_test):
    total = num_train + num_test
    schemes = ["fixed", "rotating", "quadrant", "mixed"]
    scheme = schemes[int(rng.choice(4, p=[0.2, 0.35, 0.2, 0.25]))]
    if scheme == "mixed" and num_train < 3:
        scheme = "rotating"
    # --- layout plan -----------------------------------------------------
    plan = []
    if scheme == "fixed":
        side = HALF_SIDES[int(rng.integers(4))]
        plan = [("half", side)] * total
    elif scheme == "quadrant":
        corners = list(QUAD_SIDES)
        plan = [("quad", corners[int(rng.integers(4))]) for _ in range(total)]
        # training shows at least two corners
        if len(set(p[1] for p in plan[:num_train])) < 2:
            plan[1] = ("quad", [c for c in corners if c != plan[0][1]][int(rng.integers(3))])
    else:
        h_side = ("top", "bottom")[int(rng.integers(2))]
        v_side = ("left", "right")[int(rng.integers(2))]
        train = [("half", h_side), ("half", v_side)]
        if scheme == "mixed":
            train.append(("quad", QUAD_SIDES[int(rng.integers(4))]))
        while len(train) < num_train:
            if scheme == "mixed" and rng.random() < 0.35:
                train.append(("quad", QUAD_SIDES[int(rng.integers(4))]))
            else:
                train.append(("half", HALF_SIDES[int(rng.integers(4))]))
        order = rng.permutation(len(train))
        train = [train[i] for i in order]
        test = []
        for _ in range(num_test):
            if scheme == "mixed" and rng.random() < 0.5:
                test.append(("quad", QUAD_SIDES[int(rng.integers(4))]))
            else:
                test.append(("half", HALF_SIDES[int(rng.integers(4))]))
        plan = train + test
    # --- palette plan ----------------------------------------------------
    colours = [int(c) for c in rng.permutation(np.arange(1, 10))]
    sea_policy = "fixed" if rng.random() < 0.4 else "varying"
    if sea_policy == "fixed":
        sea_pool = colours[:1]
        rest = colours[1:]
    else:
        n_sea = int(rng.integers(2, 4))
        sea_pool = colours[:n_sea]
        rest = colours[n_sea:]
    n_wall = int(rng.integers(1, 5))
    wall_palette = rest[:n_wall]
    spare_wall = rest[n_wall] if len(rest) > n_wall else None
    seas = []
    for i in range(total):
        seas.append(sea_pool[int(rng.integers(len(sea_pool)))])
    if sea_policy == "varying":
        if len(sea_pool) == 3 and rng.random() < 0.6:
            # reserve a sea colour that only the test pairs use
            seas = [sea_pool[int(rng.integers(2))] for _ in range(num_train)]
            seas += [sea_pool[2] if rng.random() < 0.75 else sea_pool[int(rng.integers(2))]
                     for _ in range(num_test)]
        if len(set(seas[:num_train])) < 2:
            seas[0], seas[1] = sea_pool[0], sea_pool[1]
    # --- per-pair generation ---------------------------------------------
    styles = ["maze", "rooms", "sparse"]
    appearances = ["single", "mottled", "objects", "regions"]
    sizes = ["small", "medium", "large"]
    pairs, seen = [], set()
    train_infos = []
    for i, (mode, side) in enumerate(plan):
        is_test = i >= num_train
        showcase = i == 0
        style = "rooms" if showcase else styles[int(rng.choice(3, p=[0.45, 0.35, 0.2]))]
        if is_test and rng.random() < 0.5:
            style = "maze"
        size_class = sizes[int(rng.choice(3, p=[0.3, 0.45, 0.25]))]
        if showcase and size_class == "small":
            size_class = "medium"
        if is_test and rng.random() < 0.4:
            size_class = "large"
        k = int(rng.integers(1, len(wall_palette) + 1))
        if showcase:
            wall_colours = list(wall_palette)
        else:
            wall_colours = [wall_palette[j] for j in sorted(rng.choice(len(wall_palette), size=k, replace=False))]
        if is_test and spare_wall is not None and len(wall_palette) >= 2 and rng.random() < 0.3:
            wall_colours = wall_colours + [spare_wall]
        appearance = appearances[int(rng.integers(4))]
        made = None
        for _ in range(8):
            made = _make_pair(rng, mode, side, size_class, seas[i], wall_colours, style, appearance, showcase)
            if made is not None:
                key = made[0].tobytes() + bytes(made[0].shape)
                if key not in seen:
                    seen.add(key)
                    break
                made = None
        if made is None:
            return None
        inp, out, info = made
        if not is_test:
            train_infos.append(info)
        pairs.append({"input": inp.astype(int).tolist(), "output": out.astype(int).tolist()})
    puzzle = {"train": pairs[:num_train], "test": pairs[num_train:]}
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        return None
    return puzzle


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    keys = set()
    infos = []
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert set(pair) == {"input", "output"}
            inp = np.asarray(pair["input"], dtype=int)
            out = np.asarray(pair["output"], dtype=int)
            assert inp.ndim == 2 and 1 <= inp.shape[0] <= 30 and 1 <= inp.shape[1] <= 30
            assert inp.shape == out.shape
            assert inp.min() >= 0 and inp.max() <= 9 and out.min() >= 0 and out.max() <= 9
            key = inp.tobytes() + bytes(inp.shape)
            assert key not in keys, "duplicate input"
            keys.add(key)
            found = _detect(inp)
            assert len(found) == 1, "blank destination must be unique"
            assert np.array_equal(solve(inp), out), "output disagrees with rule"
            info = _analyze(inp, out)
            name = info["side"]
            src = _source_mask(inp.shape, name)
            # source preserved, output symmetric
            sea = info["sea"]
            keep = inp[src] != FLOOR
            assert np.array_equal(out[src][keep], inp[src][keep]), "walls changed"
            assert set(np.unique(out[src][~keep]).tolist()) <= {FLOOR, sea}
            if info["mode"] == "half" and info["axis"] == "h":
                assert np.array_equal(out, out[::-1])
            elif info["mode"] == "half":
                assert np.array_equal(out, out[:, ::-1])
            else:
                assert np.array_equal(out, out[::-1]) and np.array_equal(out, out[:, ::-1])
            assert info["seam_pockets"] >= 1, "every pair needs a seam-sealed pocket"
            assert info["exterior"] >= 3
            # Shortcuts must fail on every pair.
            assert not np.array_equal(info["mirror_only"], out), "mirror-only shortcut works"
            assert not np.array_equal(info["seam_border_shortcut"], out), "seam-as-border shortcut works"
            assert not np.array_equal(info["raw_flood"], out), "raw flood shortcut works"
            infos.append((split, info))
    tr = [i for s, i in infos if s == "train"]
    te = [i for s, i in infos if s == "test"]
    assert any(i["interior_pockets"] >= 1 for i in tr), "training needs a closed interior room"
    assert any(i["seam_exterior"] >= 1 for i in tr), "training needs a seam-crossing exterior region"
    assert any(i["diag"] for i in tr), "training needs a diagonal near-miss"
    train_axes = {i["axis"] for i in tr}
    for i in te:
        if i["axis"] not in train_axes:
            # a new half-axis in test is only allowed if training already shows both half axes
            raise AssertionError("test axis/mode unseen in training")
    train_seas = {i["sea"] for i in tr}
    if len(train_seas) == 1:
        assert all(i["sea"] in train_seas for i in te), "new sea colour without varying training seas"
    train_walls = set().union(*[set(i["walls"]) for i in tr])
    test_walls = set().union(*[set(i["walls"]) for i in te])
    if not test_walls <= train_walls:
        assert any(len(i["walls"]) >= 2 for i in tr), "new wall colour needs multicolour training walls"
    all_seas = {i["sea"] for _, i in infos}
    all_walls = set().union(*[set(i["walls"]) for _, i in infos])
    assert not (all_seas & all_walls), "sea and wall colours must be disjoint within a puzzle"


# ----------------------------------------------------------------------------
# Self test
# ----------------------------------------------------------------------------

def self_test() -> None:
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (4, 5)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted {bad}")
    shapes, sides, sigs, seas_vary, quad_puzzles, multi_axis = set(), set(), set(), 0, 0, 0
    wall_counts = set()
    new_test_sea = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        q = generate_puzzle(seed)
        assert p == q, "nondeterministic"
        validate_generated_puzzle(p)
        sigs.add(repr(p))
        infos = [_analyze(np.array(pr["input"])) for pr in p["train"] + p["test"]]
        for pr, info in zip(p["train"] + p["test"], infos):
            shapes.add(np.array(pr["input"]).shape)
            sides.add(info["side"])
            wall_counts.add(len(info["walls"]))
        if len({i["sea"] for i in infos}) > 1:
            seas_vary += 1
        n_tr = len(p["train"])
        if any(i["sea"] not in {t["sea"] for t in infos[:n_tr]} for i in infos[n_tr:]):
            new_test_sea += 1
        if any(i["mode"] == "quad" for i in infos):
            quad_puzzles += 1
        if len({i["axis"] for i in infos}) > 1:
            multi_axis += 1
    assert len(sigs) == 42
    assert len(shapes) >= 40, len(shapes)
    assert sides == set(HALF_SIDES) | set(QUAD_SIDES), sides
    assert seas_vary >= 10 and quad_puzzles >= 8 and multi_axis >= 10, (seas_vary, quad_puzzles, multi_axis)
    assert wall_counts >= {1, 2, 3}, wall_counts
    assert new_test_sea >= 3, new_test_sea
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(7, a, b))


if __name__ == "__main__":
    self_test()
    print("self_test passed")
