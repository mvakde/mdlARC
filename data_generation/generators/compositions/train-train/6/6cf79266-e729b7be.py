"""Pivot squares: hidden blank 3x3 squares fix the centre of a half-turn copy.

Composition of ARC 6cf79266 (find every all-blank 3x3 window hidden in a
two-colour noise field and highlight it) and ARC e729b7be (complete a figure's
half-turn symmetry about a marked centre by copying every payload cell to its
180-degree image).

Rule (input-only):
  * The input is a two-colour noise field (blank + texture) with one to four
    hidden all-blank 3x3 squares and one to three multicoloured payload pieces
    (colours other than the two noise colours).
  * The hidden squares are the only monochrome 3x3 windows.  Their centres form
    a point-symmetric constellation (one square; two squares; a middle square
    with two partners; or two crossing pairs).  The constellation's centre of
    symmetry is the pivot.
  * Output: every hidden square is recoloured with the family highlight colour
    (1), and every payload cell is copied, with its colour, to its half-turn
    image about the pivot (r, c) -> (Sr - r, Sc - c), overwriting the noise
    there.  All other cells are unchanged, so the coloured content of the
    output is point-symmetric about the pivot.
"""
from __future__ import annotations

import numpy as np

HIGHLIGHT = 1
MIN_SIDE = 12
MAX_SIDE = 30
SHIFTS = ("more_pieces", "bigger", "new_color", "richer", "plain")
MASK_STYLES = ("solid", "frame", "corner", "cross", "stair", "blob", "comb", "zig")
COLOR_STYLES = ("rows", "cols", "checker", "diag", "blobs", "rim", "random")
DECOY_KINDS = ("8/9", "2x3", "3x2")


# ----------------------------------------------------------------------------
# Shared grid analysis (used by solve, validation and self-test)
# ----------------------------------------------------------------------------

def _mono_windows(grid):
    """All 3x3 windows whose nine cells share one colour: [(row, col, colour)]."""
    h, w = grid.shape
    if h < 3 or w < 3:
        return []
    base = grid[:h - 2, :w - 2]
    same = np.ones((h - 2, w - 2), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            same &= grid[dr:h - 2 + dr, dc:w - 2 + dc] == base
    return [(int(r), int(c), int(base[r, c])) for r, c in np.argwhere(same)]


def _window_counts(mask):
    h, w = mask.shape
    counts = np.zeros((max(h - 2, 0), max(w - 2, 0)), dtype=np.int16)
    if h < 3 or w < 3:
        return counts
    m = mask.astype(np.int16)
    for dr in range(3):
        for dc in range(3):
            counts += m[dr:h - 2 + dr, dc:w - 2 + dc]
    return counts


def _roles(grid):
    """Recover (blank, texture, square starts) from an input grid alone."""
    windows = _mono_windows(grid)
    if not windows:
        return None
    colours = [col for _, _, col in windows]
    blank = max(sorted(set(colours)), key=colours.count)
    starts = sorted((r, c) for r, c, col in windows if col == blank)
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    others = [(int(counts[c]), -c, c) for c in range(10) if c != blank and counts[c] > 0]
    texture = max(others)[2] if others else blank
    return blank, texture, starts


def _pivot_sum(starts):
    """Twice the centroid of the square centres, i.e. S with image p -> S - p."""
    n = len(starts)
    sr = sum(2 * (r + 1) for r, _ in starts)
    sc = sum(2 * (c + 1) for _, c in starts)
    return sr, sc, n


def _render(grid, starts, payload_cells, dest_fn):
    out = grid.copy()
    h, w = grid.shape
    for r, c in starts:
        out[r:r + 3, c:c + 3] = HIGHLIGHT
    for (r, c), (rr, cc) in zip(payload_cells, dest_fn(payload_cells)):
        if 0 <= rr < h and 0 <= cc < w:
            out[rr, cc] = grid[r, c]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.array(input_grid, dtype=int)
    roles = _roles(grid)
    if roles is None:
        return grid.copy()
    blank, texture, starts = roles
    sr, sc, n = _pivot_sum(starts)
    pivot = (int(round(sr / n)), int(round(sc / n)))
    payload = [tuple(map(int, p)) for p in np.argwhere((grid != blank) & (grid != texture))]
    return _render(grid, starts, payload,
                   lambda cells: [(pivot[0] - r, pivot[1] - c) for r, c in cells])


# ----------------------------------------------------------------------------
# Payload pieces
# ----------------------------------------------------------------------------

def _trim(mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _connected8(mask):
    cells = [tuple(map(int, p)) for p in np.argwhere(mask)]
    if not cells:
        return False
    todo, seen = [cells[0]], {cells[0]}
    present = set(cells)
    while todo:
        r, c = todo.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = (r + dr, c + dc)
                if q in present and q not in seen:
                    seen.add(q)
                    todo.append(q)
    return len(seen) == len(cells)


def _piece_mask(rng, style, limit):
    hi = limit + 1

    def dim(low=2):
        # Mostly substantial pieces; occasionally a thin one.
        return int(rng.integers(low, hi)) if rng.random() < 0.25 else int(rng.integers(max(low, 3), hi))

    for _ in range(60):
        if style == "solid":
            a, b = int(rng.integers(2, 4)), dim(3)
            mask = np.ones((a, b) if rng.random() < 0.5 else (b, a), dtype=bool)
        elif style == "frame":
            ph, pw = int(rng.integers(3, hi)), int(rng.integers(3, hi))
            mask = np.zeros((ph, pw), dtype=bool)
            mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
            if rng.random() < 0.5:
                side = int(rng.integers(4))
                if side == 0:
                    mask[0, 1:-1] = False
                elif side == 1:
                    mask[-1, 1:-1] = False
                elif side == 2:
                    mask[1:-1, 0] = False
                else:
                    mask[1:-1, -1] = False
        elif style == "corner":
            ph, pw = dim(), dim()
            t = 2 if min(ph, pw) >= 4 and rng.random() < 0.7 else 1
            mask = np.zeros((ph, pw), dtype=bool)
            mask[:, :t] = True
            mask[-t:, :] = True
            mask = np.rot90(mask, int(rng.integers(4)))
        elif style == "cross":
            ph, pw = int(rng.integers(3, hi)), int(rng.integers(3, hi))
            mask = np.zeros((ph, pw), dtype=bool)
            t = 2 if min(ph, pw) >= 4 and rng.random() < 0.4 else 1
            r, c = int(rng.integers(ph - t + 1)), int(rng.integers(pw - t + 1))
            mask[r:r + t, :] = True
            mask[:, c:c + t] = True
        elif style == "stair":
            t = int(rng.integers(1, 3)) if limit < 6 else int(rng.integers(2, 4))
            n = int(rng.integers(3, max(4, limit - t + 2)))
            mask = np.zeros((n, n + t - 1), dtype=bool)
            for i in range(n):
                mask[i, i:i + t] = True
            if rng.random() < 0.5:
                mask = mask[:, ::-1]
        elif style == "blob":
            ph, pw = dim(), dim()
            target = max(4, int(round(ph * pw * float(rng.uniform(0.55, 0.9)))))
            mask = np.zeros((ph, pw), dtype=bool)
            frontier = [(int(rng.integers(ph)), int(rng.integers(pw)))]
            mask[frontier[0]] = True
            while mask.sum() < target:
                r, c = frontier[int(rng.integers(len(frontier)))]
                dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
                q = (r + dr, c + dc)
                if 0 <= q[0] < ph and 0 <= q[1] < pw and not mask[q]:
                    mask[q] = True
                    frontier.append(q)
        elif style == "comb":
            ph, pw = int(rng.integers(3, hi)), int(rng.integers(3, hi))
            mask = np.zeros((ph, pw), dtype=bool)
            mask[0, :] = True
            for c in range(int(rng.integers(2)), pw, 2):
                mask[:int(rng.integers(2, ph + 1)), c] = True
            mask = np.rot90(mask, int(rng.integers(4)))
        else:  # zig: a short random lattice walk
            ph, pw = int(rng.integers(3, hi)), int(rng.integers(3, hi))
            mask = np.zeros((ph, pw), dtype=bool)
            r, c = int(rng.integers(ph)), int(rng.integers(pw))
            for _ in range(int(rng.integers(ph * pw // 2 + 2, ph * pw + 3))):
                mask[r, c] = True
                dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1))[int(rng.integers(6))]
                r, c = min(max(r + dr, 0), ph - 1), min(max(c + dc, 0), pw - 1)
        mask = _trim(mask)
        if 4 <= int(mask.sum()) and max(mask.shape) <= limit and _connected8(mask):
            return mask
    return np.ones((1, 4), dtype=bool)


def _color_piece(rng, mask, palette):
    """Colour a piece mask; returns an int array with -1 outside the mask."""
    k = len(palette)
    cells = np.argwhere(mask)
    rows, cols = cells[:, 0], cells[:, 1]
    style = "single" if k == 1 else COLOR_STYLES[int(rng.integers(len(COLOR_STYLES)))]
    if style == "single":
        idx = np.zeros(len(cells), dtype=np.int64)
    elif style == "rows":
        idx = (rows // int(rng.integers(1, 3))) % k
    elif style == "cols":
        idx = (cols // int(rng.integers(1, 3))) % k
    elif style == "checker":
        idx = (rows + cols) % k
    elif style == "diag":
        idx = (rows - cols) % k
    elif style == "blobs":
        seeds = cells[rng.choice(len(cells), size=min(k, len(cells)), replace=False)]
        dist = (rows[:, None] - seeds[None, :, 0]) ** 2 + (cols[:, None] - seeds[None, :, 1]) ** 2
        idx = np.argmin(dist, axis=1)
    elif style == "rim":
        padded = np.pad(mask, 1)
        interior = np.array([padded[r, c + 1] and padded[r + 2, c + 1] and padded[r + 1, c]
                             and padded[r + 1, c + 2] for r, c in cells])
        idx = np.where(interior, 1, 0) % k
        if k > 2:
            idx = np.where(interior, 1 + (rows + cols) % (k - 1), 0)
    else:
        idx = rng.integers(0, k, size=len(cells))
    idx = np.asarray(idx, dtype=np.int64)
    if len(cells) >= k:
        # Every requested palette colour is visible.
        for colour_index, place in enumerate(rng.choice(len(cells), size=k, replace=False)):
            idx[int(place)] = colour_index
    piece = np.full(mask.shape, -1, dtype=np.int64)
    for (r, c), i in zip(cells, idx):
        piece[r, c] = palette[int(i)]
    return piece


# ----------------------------------------------------------------------------
# Pair construction
# ----------------------------------------------------------------------------

def _cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _constellation(rng, h, w, ctype):
    """Square centres (a point-symmetric set) and S, twice its centre.

    The pivot is sampled first, off the plain grid centre but close enough to
    it that the grid and its half-turn image overlap substantially (room for
    payload pieces and their copies)."""
    def centre():
        return (int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))

    def valid(p):
        return 1 <= p[0] <= h - 2 and 1 <= p[1] <= w - 2

    for _ in range(600):
        dr = int(rng.integers(-(9 * h) // 20, (9 * h) // 20 + 1))
        dc = int(rng.integers(-(9 * w) // 20, (9 * w) // 20 + 1))
        if max(abs(dr), abs(dc)) < 2:
            continue  # the pivot is visibly away from the grid centre
        total = (h - 1 + dr, w - 1 + dc)
        if ctype in (1, 3) and (total[0] % 2 or total[1] % 2):
            continue  # a middle square needs a whole-cell pivot
        if ctype == 1:
            centres = [(total[0] // 2, total[1] // 2)]
        elif ctype == 2:
            a = centre()
            centres = [a, (total[0] - a[0], total[1] - a[1])]
        elif ctype == 3:
            a = centre()
            centres = [a, (total[0] // 2, total[1] // 2), (total[0] - a[0], total[1] - a[1])]
        else:
            a1, a2 = centre(), centre()
            centres = [a1, (total[0] - a1[0], total[1] - a1[1]), a2, (total[0] - a2[0], total[1] - a2[1])]
        if not all(valid(p) for p in centres):
            continue
        if any(_cheb(p, q) < 4 for i, p in enumerate(centres) for q in centres[i + 1:]):
            continue
        return centres, total
    return None


def _dilated_hits(blocked, r0, c0, ph, pw):
    h, w = blocked.shape
    return bool(blocked[max(0, r0 - 1):min(h, r0 + ph + 1), max(0, c0 - 1):min(w, c0 + pw + 1)].any())


def _place_piece(rng, blocked, total, ph, pw):
    h, w = blocked.shape
    sr, sc = total
    lo_r, hi_r = max(0, sr - (h - 1)), min(h - ph, sr - ph + 1)
    lo_c, hi_c = max(0, sc - (w - 1)), min(w - pw, sc - pw + 1)
    options = []
    for r0 in range(lo_r, hi_r + 1):
        ir0 = sr - (r0 + ph - 1)
        for c0 in range(lo_c, hi_c + 1):
            ic0 = sc - (c0 + pw - 1)
            # The piece and its image must be separated by at least one cell.
            if (r0 - 1 <= ir0 + ph - 1 and ir0 <= r0 + ph and
                    c0 - 1 <= ic0 + pw - 1 and ic0 <= c0 + pw):
                continue
            if _dilated_hits(blocked, r0, c0, ph, pw) or _dilated_hits(blocked, ir0, ic0, ph, pw):
                continue
            options.append((r0, c0, ir0, ic0))
    if not options:
        return None
    return options[int(rng.integers(len(options)))]


def _ring_sides(shape, start):
    r, c = start
    h, w = shape
    sides = []
    if r > 0:
        sides.append([(r - 1, c + x) for x in range(3)])
    if r + 3 < h:
        sides.append([(r + 3, c + x) for x in range(3)])
    if c > 0:
        sides.append([(r + x, c - 1) for x in range(3)])
    if c + 3 < w:
        sides.append([(r + x, c + 3) for x in range(3)])
    return sides


def _decoy_patch(rng, kind):
    patch = np.ones((3, 3), dtype=bool)  # True = blank
    if kind == "2x3":
        patch[0 if rng.random() < 0.5 else 2, :] = False
    elif kind == "3x2":
        patch[:, 0 if rng.random() < 0.5 else 2] = False
    else:
        patch[int(rng.integers(3)), int(rng.integers(3))] = False
    return patch


def _repair(rng, grid, fixed, blank, texture, targets):
    for _ in range(grid.size * 3):
        bad = [(r, c, col) for r, c, col in _mono_windows(grid)
               if not (col == blank and (r, c) in targets)]
        if not bad:
            return True
        r, c, col = bad[int(rng.integers(len(bad)))]
        if col not in (blank, texture):
            return False
        cells = [(rr, cc) for rr in range(r, r + 3) for cc in range(c, c + 3) if not fixed[rr, cc]]
        if not cells:
            return False
        rr, cc = cells[int(rng.integers(len(cells)))]
        grid[rr, cc] = texture if col == blank else blank
    return False


def _make_pair(rng, spec, blank, texture):
    ctype, n_pieces = spec["ctype"], spec["pieces"]
    for attempt in range(120):
        grow = attempt // 30
        lo_h, hi_h = spec["rows"]
        lo_w, hi_w = spec["cols"]
        h = int(rng.integers(min(lo_h + 2 * grow, hi_h), hi_h + 1))
        w = int(rng.integers(min(lo_w + 2 * grow, hi_w), hi_w + 1))
        made = _constellation(rng, h, w, ctype)
        if made is None:
            continue
        centres, total = made
        starts = sorted((r - 1, c - 1) for r, c in centres)
        blocked = np.zeros((h, w), dtype=bool)
        for r, c in starts:
            blocked[r:r + 3, c:c + 3] = True
        side = min(h, w)
        limit = 4 if side < 14 else 5 if side < 18 else 6 if side < 23 else 7
        pieces = []
        ok = True
        for index in range(n_pieces):
            for _ in range(12):
                mask = _piece_mask(rng, spec["styles"][index], limit)
                if rng.random() < 0.5:
                    mask = mask.T.copy()
                piece = _color_piece(rng, mask, spec["palettes"][index])
                if any(len(set(piece[r:r + 3, c:c + 3].ravel())) == 1 and piece[r, c] >= 0
                       for r in range(piece.shape[0] - 2) for c in range(piece.shape[1] - 2)):
                    continue  # a monochrome 3x3 block would read as a hidden square
                spot = _place_piece(rng, blocked, total, *piece.shape)
                if spot is not None:
                    break
            else:
                ok = False
                break
            r0, c0, ir0, ic0 = spot
            ph, pw = piece.shape
            blocked[r0:r0 + ph, c0:c0 + pw] = True
            blocked[ir0:ir0 + ph, ic0:ic0 + pw] = True
            pieces.append((r0, c0, piece))
        if not ok:
            continue
        decoys = []
        for kind in spec["decoys"]:
            options = [(r, c) for r in range(h - 2) for c in range(w - 2)
                       if not _dilated_hits(blocked, r, c, 3, 3)]
            if not options:
                break
            r, c = options[int(rng.integers(len(options)))]
            blocked[r:r + 3, c:c + 3] = True
            decoys.append((r, c, _decoy_patch(rng, kind)))
        if len(decoys) < spec["min_decoys"]:
            continue
        grid = np.where(rng.random((h, w)) < spec["density"], blank, texture).astype(np.int64)
        fixed = np.zeros((h, w), dtype=bool)
        for r, c in starts:
            grid[r:r + 3, c:c + 3] = blank
            fixed[r:r + 3, c:c + 3] = True
        footprint = fixed.copy()
        for r, c in starts:
            sides = _ring_sides((h, w), (r, c))
            for side in sides:
                for p in side:
                    if not footprint[p]:
                        grid[p] = texture
                        fixed[p] = True
            if sides and rng.random() < 0.6:  # attach short blank runs to the square
                for s in rng.permutation(len(sides))[:int(rng.integers(1, min(3, len(sides)) + 1))]:
                    side = [p for p in sides[int(s)] if not footprint[p]]
                    if side:
                        take = rng.choice(len(side), size=min(len(side), int(rng.integers(1, 3))),
                                          replace=False)
                        for t in np.atleast_1d(take):
                            grid[side[int(t)]] = blank
        for r, c, patch in decoys:
            grid[r:r + 3, c:c + 3] = np.where(patch, blank, texture)
            fixed[r:r + 3, c:c + 3] = True
        for r0, c0, piece in pieces:
            inside = piece >= 0
            region = grid[r0:r0 + piece.shape[0], c0:c0 + piece.shape[1]]
            region[inside] = piece[inside]
            fixed[r0:r0 + piece.shape[0], c0:c0 + piece.shape[1]] |= inside
        if not _repair(rng, grid, fixed, blank, texture, set(starts)):
            continue
        roles = _roles(grid)
        if roles is None or roles[0] != blank or roles[1] != texture or roles[2] != starts:
            continue
        counts = np.bincount(grid.ravel(), minlength=10)
        payload_colours = [c for c in range(10) if c not in (blank, texture) and counts[c]]
        top = max(int(counts[c]) for c in payload_colours)
        if counts[texture] < max(2 * top, 0.15 * grid.size) or counts[blank] < max(2 * top, 0.15 * grid.size):
            continue
        output = solve(grid)
        return {"input": grid, "output": output}
    raise RuntimeError("could not construct a pair")


# ----------------------------------------------------------------------------
# Puzzle construction
# ----------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    for value, name in ((num_train, "num_train"), (num_test, "num_test")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    num_train, num_test = int(num_train), int(num_test)
    if not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _min_side(ctype, pieces):
    return min(MAX_SIDE, 12 + 2 * (ctype - 1) + 2 * (pieces - 1))


def _plan(rng, num_train, num_test):
    """Puzzle-level choices: palette roles, constellation types, shift and pair specs."""
    blank = 0 if rng.random() < 0.5 else int(rng.choice([c for c in range(2, 10)]))
    free = [c for c in range(10) if c not in (blank, HIGHLIGHT)]
    rng.shuffle(free)
    if rng.random() < 0.55:
        textures = [int(free[0])]
    else:
        textures = [int(c) for c in free[:int(rng.integers(2, 4))]]
    payload_colours = [int(c) for c in free[len(textures):]]

    r = rng.random()
    if r < 0.3:
        types = [2]
    elif r < 0.5:
        types = [1, 2]
    else:
        size = int(rng.integers(2, 5))
        types = sorted(int(t) for t in rng.choice([1, 2, 3, 4], size=size, replace=False))
    while len(types) > num_train:
        types.remove(types[int(rng.integers(len(types)))])
    if max(types) < 2:
        types = [1, 2]

    shift = SHIFTS[int(rng.choice(len(SHIFTS), p=[0.24, 0.2, 0.2, 0.22, 0.14]))]
    if shift == "richer" and len(types) < 2:
        shift = "more_pieces"
    reserved = None
    if shift == "new_color":
        reserved = payload_colours.pop()

    # Constellation types: training covers every type; tests reuse them.
    train_types = list(types)
    while len(train_types) < num_train:
        if shift == "richer":
            train_types.append(int(rng.choice(types[:-1])))
        else:
            train_types.append(int(rng.choice(types)))
    first = min(train_types)
    train_types.remove(first)
    rng.shuffle(train_types)
    train_types = [first] + [int(t) for t in train_types]
    if shift == "richer":
        test_types = [max(types)] * num_test
    else:
        test_types = [int(rng.choice(types)) for _ in range(num_test)]

    # Piece counts.
    if shift == "more_pieces":
        train_pieces = [1] * num_train
        if num_train >= 3 and rng.random() < 0.5:
            train_pieces[int(rng.integers(1, num_train))] = 2
        test_pieces = [int(rng.integers(max(train_pieces) + 1, 4)) for _ in range(num_test)]
    else:
        weights = [0.42, 0.38, 0.2]
        train_pieces = [1] + [int(rng.choice([1, 2, 3], p=weights)) for _ in range(num_train - 1)]
        test_pieces = [int(rng.choice([1, 2, 3], p=weights)) for _ in range(num_test)]

    # Decoys: at least one training pair carries an eight-of-nine near miss.
    def decoy_list(force):
        n = int(rng.choice([0, 1, 2, 3], p=[0.2, 0.35, 0.3, 0.15]))
        kinds = [DECOY_KINDS[int(rng.integers(3))] for _ in range(n)]
        if force and "8/9" not in kinds:
            kinds.insert(0, "8/9")
        return kinds

    forced = int(rng.integers(num_train))
    specs = []
    for split, types_, pieces_ in (("train", train_types, train_pieces), ("test", test_types, test_pieces)):
        for index, (ctype, n_pieces) in enumerate(zip(types_, pieces_)):
            need = _min_side(ctype, n_pieces)
            if shift == "bigger":
                if split == "train":
                    rows = cols = (need, max(need, 19))
                else:
                    big = (22, 30)
                    other = (max(need, 14), 30)
                    rows, cols = (big, other) if rng.random() < 0.5 else (other, big)
            else:
                top = int(rng.choice([20, 24, 28, 30]))
                rows = cols = (need, max(need + 2, top))
            styles = [MASK_STYLES[int(rng.integers(len(MASK_STYLES)))] for _ in range(n_pieces)]
            palettes = []
            pool = list(payload_colours)
            for p in range(n_pieces):
                k = int(rng.choice([1, 2, 3, 4], p=[0.2, 0.35, 0.3, 0.15]))
                palette = [int(c) for c in rng.choice(pool, size=min(k, len(pool)), replace=False)]
                if split == "test" and reserved is not None and p == 0:
                    palette[int(rng.integers(len(palette)))] = reserved
                    if len(palette) == 1 and rng.random() < 0.5:
                        palette.append(int(rng.choice(pool)))
                palettes.append(palette)
            kinds = decoy_list(split == "train" and index == forced)
            specs.append((split, {
                "ctype": ctype, "pieces": n_pieces, "rows": rows, "cols": cols,
                "styles": styles, "palettes": palettes, "decoys": kinds,
                "min_decoys": 1 if "8/9" in kinds[:1] and split == "train" and index == forced else 0,
                "density": float(rng.uniform(0.32, 0.6)),
                "texture": textures[int(rng.integers(len(textures)))] if split == "test" or index >= len(textures)
                else textures[index],
            }))
    return blank, specs


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = 4 if num_train is None else num_train
    num_test = 1 if num_test is None else num_test
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    last_problems = []
    for _ in range(40):
        sub = np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1)))
        blank, specs = _plan(sub, num_train, num_test)
        puzzle = {"train": [], "test": []}
        try:
            for split, spec in specs:
                pair_rng = np.random.default_rng(int(sub.integers(0, 2 ** 63 - 1)))
                puzzle[split].append(_make_pair(pair_rng, spec, blank, spec["texture"]))
        except RuntimeError:
            continue
        last_problems = _puzzle_problems(puzzle)
        if not last_problems:
            return {split: [{"input": p["input"].copy(), "output": p["output"].copy()} for p in puzzle[split]]
                    for split in ("train", "test")}
    raise RuntimeError(f"could not build puzzle: {last_problems[:3]}")


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

def _components8(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for start in map(tuple, np.argwhere(mask)):
        if seen[start]:
            continue
        stack, comp = [start], []
        seen[start] = True
        while stack:
            r, c = stack.pop()
            comp.append((int(r), int(c)))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if 0 <= q[0] < mask.shape[0] and 0 <= q[1] < mask.shape[1] and mask[q] and not seen[q]:
                        seen[q] = True
                        stack.append(q)
        comps.append(comp)
    return comps


def _analyze(grid):
    """Structural reading of an input grid (raises AssertionError on violations)."""
    roles = _roles(grid)
    assert roles is not None, "no hidden square"
    blank, texture, starts = roles
    windows = _mono_windows(grid)
    assert all(col == blank for _, _, col in windows), "monochrome window of a non-blank colour"
    assert HIGHLIGHT not in set(grid.ravel().tolist()), "highlight colour present in input"
    assert 1 <= len(starts) <= 4, "square count"
    centres = [(r + 1, c + 1) for r, c in starts]
    assert all(_cheb(p, q) >= 4 for i, p in enumerate(centres) for q in centres[i + 1:]), "overlapping squares"
    sr, sc, n = _pivot_sum(starts)
    assert sr % n == 0 and sc % n == 0, "pivot not on the half-cell lattice"
    total = (sr // n, sc // n)
    assert sorted(centres) == sorted((total[0] - r, total[1] - c) for r, c in centres), \
        "squares are not point-symmetric"
    h, w = grid.shape
    assert total != (h - 1, w - 1), "pivot coincides with grid centre"
    payload = (grid != blank) & (grid != texture)
    cells = [tuple(map(int, p)) for p in np.argwhere(payload)]
    assert len(cells) >= 3, "payload too small"
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    top = max(int(counts[c]) for c in range(10) if c not in (blank, texture) and counts[c])
    assert counts[texture] >= 2 * top and counts[blank] >= 2 * top, "noise colours not dominant"
    footprint = np.zeros(grid.shape, dtype=bool)
    for r, c in starts:
        footprint[r:r + 3, c:c + 3] = True
    for r, c in cells:
        rr, cc = total[0] - r, total[1] - c
        assert 0 <= rr < h and 0 <= cc < w, "payload image leaves the grid"
        assert not payload[rr, cc] and not footprint[rr, cc], "payload image collides"
        nb = footprint[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
        assert not nb.any(), "payload touches a square"
    near8 = 0
    counts9 = _window_counts(grid == blank)
    for r, c in np.argwhere(counts9 == 8):
        if not footprint[r:r + 3, c:c + 3].any():
            near8 += 1
    return {"blank": blank, "texture": texture, "starts": starts, "total": total,
            "cells": cells, "payload": payload, "near8": near8,
            "pieces": _components8(payload), "colours": sorted(set(grid[payload].tolist()))}


def _alternatives(grid, info):
    """Outputs of plausible shortcut rules (each drops or distorts one concept)."""
    starts, cells, (sr, sc) = info["starts"], info["cells"], info["total"]
    h, w = grid.shape
    alts = {"no_copy": _render(grid, starts, [], lambda cs: []),
            "grid_centre": _render(grid, starts, cells, lambda cs: [(h - 1 - r, w - 1 - c) for r, c in cs]),
            "mirror_lr": _render(grid, starts, cells, lambda cs: [(r, sc - c) for r, c in cs]),
            "mirror_ud": _render(grid, starts, cells, lambda cs: [(sr - r, c) for r, c in cs])}
    # Translate each piece unrotated onto its image bounding box.
    moves = {}
    for piece in info["pieces"]:
        rs = [r for r, _ in piece]
        cs_ = [c for _, c in piece]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs_), max(cs_)
        dr, dc = (sr - r1) - r0, (sc - c1) - c0
        for r, c in piece:
            moves[(r, c)] = (r + dr, c + dc)
    alts["translate"] = _render(grid, starts, cells, lambda cs: [moves[p] for p in cs])
    # Rotate each piece about its nearest square instead of the constellation centre.
    near = {}
    centres = [(r + 1, c + 1) for r, c in starts]
    for piece in info["pieces"]:
        pr = sum(r for r, _ in piece) / len(piece)
        pc = sum(c for _, c in piece) / len(piece)
        m = min(centres, key=lambda q: ((q[0] - pr) ** 2 + (q[1] - pc) ** 2, q))
        for r, c in piece:
            near[(r, c)] = (2 * m[0] - r, 2 * m[1] - c)
    alts["nearest_square"] = _render(grid, starts, cells, lambda cs: [near[p] for p in cs])
    # Loose detection: windows with at least eight blank cells also count as squares.
    loose = [tuple(map(int, p)) for p in np.argwhere(_window_counts(grid == info["blank"]) >= 8)]
    lr, lc, ln = _pivot_sum(loose)
    lt = (int(round(lr / ln)), int(round(lc / ln)))
    alts["loose_squares"] = _render(grid, starts, cells, lambda cs: [(lt[0] - r, lt[1] - c) for r, c in cs])
    return alts


def _expected(grid, info):
    sr, sc = info["total"]
    return _render(grid, info["starts"], info["cells"], lambda cs: [(sr - r, sc - c) for r, c in cs])


def _pair_problems(pair):
    problems = []
    if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
        return ["pair keys"]
    grids = []
    for key in ("input", "output"):
        g = np.asarray(pair[key])
        if g.ndim != 2 or not np.issubdtype(g.dtype, np.integer):
            return [f"{key} not an integer matrix"]
        if not (MIN_SIDE <= g.shape[0] <= MAX_SIDE and MIN_SIDE <= g.shape[1] <= MAX_SIDE):
            return [f"{key} size {g.shape}"]
        if g.min() < 0 or g.max() > 9:
            return [f"{key} colours"]
        grids.append(g.astype(np.int64))
    inp, out = grids
    if inp.shape != out.shape:
        return ["shape changed"]
    try:
        info = _analyze(inp)
    except AssertionError as error:
        return [f"input: {error}"]
    expected = _expected(inp, info)
    if not np.array_equal(out, expected):
        problems.append("output differs from the rule")
    if not np.array_equal(solve(inp), out):
        problems.append("solve disagrees")
    coloured = (out != info["blank"]) & (out != info["texture"])
    sr, sc = info["total"]
    pts = [tuple(map(int, p)) for p in np.argwhere(coloured)]
    if any(out[sr - r, sc - c] != out[r, c] for r, c in pts if 0 <= sr - r < out.shape[0]
           and 0 <= sc - c < out.shape[1]) or any(not (0 <= sr - r < out.shape[0] and 0 <= sc - c < out.shape[1])
                                                    for r, c in pts):
        problems.append("output coloured content is not point-symmetric about the pivot")
    return problems


def _puzzle_problems(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        return ["puzzle keys"]
    train, test = puzzle["train"], puzzle["test"]
    if not isinstance(train, list) or not isinstance(test, list):
        return ["pair lists"]
    if not (2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)):
        return ["pair counts"]
    problems = []
    infos = []
    seen = set()
    for split, pairs in (("train", train), ("test", test)):
        for i, pair in enumerate(pairs):
            issues = _pair_problems(pair)
            if issues:
                problems += [f"{split}[{i}]: {x}" for x in issues]
                continue
            inp = np.asarray(pair["input"]).astype(np.int64)
            key = (inp.shape, inp.tobytes())
            if key in seen:
                problems.append(f"{split}[{i}]: duplicate input")
            seen.add(key)
            infos.append((split, inp, np.asarray(pair["output"]).astype(np.int64), _analyze(inp)))
    if problems:
        return problems
    if len({info["blank"] for *_, info in infos}) != 1:
        problems.append("blank colour changes within the puzzle")
    train_infos = [x for x in infos if x[0] == "train"]
    train_counts = {len(info["starts"]) for *_, info in train_infos}
    if max(train_counts) < 2:
        problems.append("no training pair with two or more squares")
    for split, _, _, info in infos:
        if split == "test" and len(info["starts"]) not in train_counts:
            problems.append("test constellation size unseen in training")
    if not any(info["near8"] for *_, info in train_infos):
        problems.append("no eight-of-nine near miss in training")
    for name in ("no_copy", "grid_centre", "mirror_lr", "mirror_ud", "translate",
                 "nearest_square", "loose_squares"):
        if not any(not np.array_equal(_alternatives(inp, info)[name], out)
                   for _, inp, out, info in train_infos):
            problems.append(f"training never refutes shortcut {name}")
    for split, inp, out, info in infos:
        alts = _alternatives(inp, info)
        if np.array_equal(alts["no_copy"], out) or np.array_equal(alts["grid_centre"], out):
            problems.append(f"{split}: pair does not distinguish both concepts")
    irregular = False
    for *_, info in train_infos:
        for piece in info["pieces"]:
            rs = [r for r, _ in piece]
            cs = [c for _, c in piece]
            if len(piece) < (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1):
                irregular = True
    if not irregular:
        problems.append("no non-rectangular payload in training")
    return problems


def validate_generated_puzzle(puzzle: dict) -> None:
    problems = _puzzle_problems(puzzle)
    if problems:
        raise AssertionError("; ".join(problems[:5]))


# ----------------------------------------------------------------------------
# Self test
# ----------------------------------------------------------------------------

def self_test() -> None:
    shapes, types, piece_counts, blanks, textures, colours = set(), set(), set(), set(), set(), set()
    half_cell = whole_cell = multi_texture = test_new_colour = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        again = generate_puzzle(seed)
        assert all(np.array_equal(a[k], b[k]) for s in ("train", "test")
                   for a, b in zip(puzzle[s], again[s]) for k in ("input", "output")), "nondeterministic"
        signatures.add(b"".join(p["input"].tobytes() for p in puzzle["train"] + puzzle["test"]))
        train_colours, puzzle_textures = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                info = _analyze(inp)
                assert np.array_equal(solve(inp), pair["output"])
                shapes.add(inp.shape)
                types.add(len(info["starts"]))
                piece_counts.add(len(info["pieces"]))
                blanks.add(info["blank"])
                textures.add(info["texture"])
                puzzle_textures.add(info["texture"])
                colours.update(info["colours"])
                if info["total"][0] % 2 or info["total"][1] % 2:
                    half_cell += 1
                else:
                    whole_cell += 1
                if split == "train":
                    train_colours.update(info["colours"])
                elif set(info["colours"]) - train_colours:
                    test_new_colour += 1
        multi_texture += len(puzzle_textures) > 1
    assert len(signatures) == 42
    assert types == {1, 2, 3, 4}, types
    assert piece_counts >= {1, 2, 3}, piece_counts
    assert len(shapes) >= 60, len(shapes)
    assert len(blanks) >= 4 and len(textures) >= 6 and len(colours) >= 7
    assert half_cell and whole_cell and multi_texture and test_new_colour
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (4, 5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {a}/{b}")
    small = generate_puzzle(7, 2, 1)
    big = generate_puzzle(7, 10, 4)
    assert len(small["train"]) == 2 and len(big["train"]) == 10 and len(big["test"]) == 4
    # A damaged output must be rejected.
    broken = generate_puzzle(3)
    broken["train"][0]["output"] = broken["train"][0]["input"].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged puzzle accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
