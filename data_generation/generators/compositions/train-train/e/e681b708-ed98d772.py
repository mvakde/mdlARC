"""Pinwheel-furnished rooms with boundary-marker plurality voting.

Composition of parent A (e681b708: one-cell divider network, coloured markers at
network vertices vote on every incident region, isolated divider-coloured query
dots take the unique plurality colour of their region) and parent B (ed98d772:
the four quarter-turn states of a square laid out as a fixed pinwheel -- original
upper-left, counter-clockwise upper-right, half turn lower-left, clockwise
lower-right).

Rule.  An odd square canvas is split by a full one-cell plus-shaped divider into
four square rooms.  Exactly one room is furnished with a divider network whose
vertices may carry coloured markers; the plus may carry fixed markers at its
ends, centre and junctions.  All rooms may contain isolated divider-coloured
query dots.  Furnish the three empty rooms so that the four rooms show the
pinwheel (moving one room right = quarter turn counter-clockwise, moving one room
down = half turn); query dots stay where they are and are not copied.  Then every
query dot takes the unique most frequent marker colour among the markers that
touch (8-neighbourhood) its region of the completed picture.  Markers on the plus
are fixed and not rotated, so a rotated copy of a region can win a different
colour than its template.

Only the standard library and NumPy are used.
"""
from __future__ import annotations

import numpy as np

ROOMS = ("TL", "TR", "BL", "BR")
MIN_ROOM, MAX_ROOM = 7, 14

# A layout maps each room to a dihedral transform (flip, k) of the canonical
# upper-left content: apply fliplr first when flip is set, then np.rot90(k)
# (counter-clockwise quarter turns).  "pinwheel" is the family rule (parent B);
# the others are plausible wrong layouts used only for evidence checks.
LAYOUTS = {
    "pinwheel": {"TL": (False, 0), "TR": (False, 1), "BL": (False, 2), "BR": (False, 3)},
    "center_turn": {"TL": (False, 0), "TR": (False, 3), "BL": (False, 1), "BR": (False, 2)},
    "mirror": {"TL": (False, 0), "TR": (True, 0), "BL": (True, 2), "BR": (False, 2)},
    "copy": {"TL": (False, 0), "TR": (False, 0), "BL": (False, 0), "BR": (False, 0)},
}
DOT_SHORTCUTS = ("no_pinwheel", "ignore_cross", "rotate_colors", "nearest", "room_vote",
                 "orthogonal_touch")
LAYOUT_SHORTCUTS = ("center_turn", "mirror", "copy", "copy_template_dots")


# --------------------------------------------------------------------------
# Small geometric helpers
# --------------------------------------------------------------------------

def _room_origin(room, n):
    return (0 if room in ("TL", "TR") else n + 1, 0 if room in ("TL", "BL") else n + 1)


def _room_slice(room, n):
    r0, c0 = _room_origin(room, n)
    return slice(r0, r0 + n), slice(c0, c0 + n)


def _room_of(r, c, n):
    if r == n or c == n:
        return None
    return ("T" if r < n else "B") + ("L" if c < n else "R")


def _apply(transform, a):
    flip, k = transform
    return np.rot90(np.fliplr(a) if flip else a, k)


def _inverse(transform):
    flip, k = transform
    return (True, k) if flip else (False, (-k) % 4)


def _preimage(i, j, n, transform):
    """Cell of the source square that `transform` moves to (i, j)."""
    probe = np.zeros((n, n), dtype=int)
    probe[i, j] = 1
    back = _apply(_inverse(transform), probe)
    r, c = np.argwhere(back)[0]
    return int(r), int(c)


def _compose(t1, t2):
    """Transform equal to applying t2 and then t1."""
    probe = np.arange(9).reshape(3, 3)
    target = _apply(t1, _apply(t2, probe))
    for flip in (False, True):
        for k in range(4):
            if np.array_equal(_apply((flip, k), probe), target):
                return (flip, k)
    raise AssertionError("dihedral composition failed")


def _key(grid):
    grid = np.asarray(grid)
    return grid.tobytes() + bytes(grid.shape)


def _dihedral_views(a):
    return [np.rot90(a, k) for k in range(4)] + [np.rot90(np.fliplr(a), k) for k in range(4)]


def _is_chiral(a):
    keys = {_key(np.ascontiguousarray(v)) for v in _dihedral_views(a)}
    return len(keys) == 8


def _isolated(g, bg, div):
    """Divider-coloured cells whose eight neighbours are all background."""
    h, w = g.shape
    padded = np.pad(g, 1, constant_values=bg)
    ok = g == div
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                ok &= padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] == bg
    return ok


def _dilate(mask):
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    out = np.zeros_like(mask)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out |= padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return out


def _label(free):
    """4-connected components of the free (non-network) cells."""
    h, w = free.shape
    lab = np.full((h, w), -1, dtype=int)
    count = 0
    for r0 in range(h):
        for c0 in range(w):
            if not free[r0, c0] or lab[r0, c0] >= 0:
                continue
            lab[r0, c0] = count
            stack = [(r0, c0)]
            while stack:
                r, c = stack.pop()
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < h and 0 <= cc < w and free[rr, cc] and lab[rr, cc] < 0:
                        lab[rr, cc] = count
                        stack.append((rr, cc))
            count += 1
    return lab, count


def _winner(tally):
    if not tally:
        return None
    best = max(tally.values())
    tops = [color for color, votes in tally.items() if votes == best]
    return tops[0] if len(tops) == 1 else None


def _legible(tally):
    """Generator-side readability policy for queried regions (not part of the rule)."""
    if _winner(tally) is None:
        return False
    votes = sorted(tally.values(), reverse=True)
    total = sum(votes)
    margin = votes[0] - (votes[1] if len(votes) > 1 else 0)
    return total <= 9 and (margin >= 2 or total <= 5)


def _vote_map(g, net, bg, div, include=None, diagonal=True):
    """Each marker votes once for every region it touches (8-neighbourhood)."""
    lab, count = _label(~net)
    tallies = [dict() for _ in range(count)]
    h, w = g.shape
    for r, c in np.argwhere(net & (g != bg) & (g != div)):
        r, c = int(r), int(c)
        if include is not None and not include[r, c]:
            continue
        color = int(g[r, c])
        touched = {int(lab[y, x]) for y in range(max(0, r - 1), min(h, r + 2))
                   for x in range(max(0, c - 1), min(w, c + 2))
                   if lab[y, x] >= 0 and (diagonal or y == r or x == c)}
        for face in touched:
            tallies[face][color] = tallies[face].get(color, 0) + 1
    return lab, tallies, [_winner(t) for t in tallies]


# --------------------------------------------------------------------------
# Input-only reading and solving
# --------------------------------------------------------------------------

def _read(grid):
    g = np.array(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or g.shape[0] % 2 == 0 or g.shape[0] < 5:
        raise ValueError("expected an odd square canvas split by a central plus")
    n = g.shape[0] // 2
    cross = np.zeros(g.shape, dtype=bool)
    cross[n, :] = True
    cross[:, n] = True
    values, counts = np.unique(g[cross], return_counts=True)
    div = int(values[np.argmax(counts)])
    values, counts = np.unique(g[~cross], return_counts=True)
    bg = int(values[np.argmax(counts)])
    dots = _isolated(g, bg, div) & ~cross
    net = (g != bg) & ~dots
    furnished = [room for room in ROOMS if net[_room_slice(room, n)].any()]
    return {"g": g, "n": n, "bg": bg, "div": div, "cross": cross, "dots": dots,
            "net": net, "furnished": furnished}


def _template(info):
    if len(info["furnished"]) != 1:
        return None, None
    room = info["furnished"][0]
    sl = _room_slice(room, info["n"])
    return room, np.where(info["net"][sl], info["g"][sl], info["bg"])


def _furnish(info, layout="pinwheel", with_dots=False):
    """Fill the empty rooms from the single furnished room under `layout`."""
    g, n, bg = info["g"], info["n"], info["bg"]
    out = g.copy()
    room, content = _template(info)
    if room is None:
        return out
    sl = _room_slice(room, n)
    if with_dots:
        content = g[sl].copy()
    canon = _apply(_inverse(LAYOUTS[layout][room]), content)
    for other in ROOMS:
        if other == room:
            continue
        copy = _apply(LAYOUTS[layout][other], canon)
        region = out[_room_slice(other, n)]
        paint = copy != bg
        region[paint] = copy[paint]
    return out


def _recolor(out, dots, bg, div, include=None):
    dots = dots & _isolated(out, bg, div)
    net = (out != bg) & ~dots
    lab, _, winners = _vote_map(out, net, bg, div, include)
    result = out.copy()
    for r, c in np.argwhere(dots):
        win = winners[lab[r, c]]
        if win is not None:
            result[r, c] = win
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read(input_grid)
    furnished = _furnish(info)
    return _recolor(furnished, info["dots"], info["bg"], info["div"])


# --------------------------------------------------------------------------
# Shortcut predictions (input-only), used to prove the demonstrations decide
# --------------------------------------------------------------------------

def _shortcut_dot_colors(input_grid):
    """Dot colours predicted by plausible simpler rules (None = left unchanged)."""
    info = _read(input_grid)
    g, n, bg, div, dots = info["g"], info["n"], info["bg"], info["div"], info["dots"]
    furnished = _furnish(info)
    dot_cells = [(int(r), int(c)) for r, c in np.argwhere(dots)]
    preds = {}

    # A without B: vote on the unfurnished input picture.
    lab_in, _, win_in = _vote_map(g, info["net"], bg, div)
    preds["no_pinwheel"] = {p: win_in[lab_in[p]] for p in dot_cells}

    # B + A but the fixed plus markers are ignored.
    net_f = (furnished != bg) & ~dots
    lab_x, _, win_x = _vote_map(furnished, net_f, bg, div, include=~info["cross"])
    preds["ignore_cross"] = {p: win_x[lab_x[p]] for p in dot_cells}

    # B + A but markers only vote for regions they touch orthogonally.
    lab_o, _, win_o = _vote_map(furnished, net_f, bg, div, diagonal=False)
    preds["orthogonal_touch"] = {p: win_o[lab_o[p]] for p in dot_cells}

    # Vote only in the template room, then rotate those region colours outward.
    room, _ = _template(info)
    rotated = {}
    rel_of = {q: _compose(LAYOUTS["pinwheel"][q], _inverse(LAYOUTS["pinwheel"][room]))
              for q in ROOMS} if room is not None else {}
    for (r, c) in dot_cells:
        q = _room_of(r, c, n)
        if room is None or q is None:
            rotated[(r, c)] = None
            continue
        rel = rel_of[q]
        r0, c0 = _room_origin(q, n)
        i, j = _preimage(r - r0, c - c0, n, rel)
        s0, t0 = _room_origin(room, n)
        face = lab_in[s0 + i, t0 + j]
        rotated[(r, c)] = win_in[face] if face >= 0 else None
    preds["rotate_colors"] = rotated

    # Nearest marker (Euclidean) in the furnished picture.
    markers = np.argwhere(net_f & (furnished != bg) & (furnished != div))
    nearest = {}
    for (r, c) in dot_cells:
        if len(markers) == 0:
            nearest[(r, c)] = None
            continue
        d2 = (markers[:, 0] - r) ** 2 + (markers[:, 1] - c) ** 2
        colors = {int(furnished[tuple(m)]) for m in markers[d2 == d2.min()]}
        nearest[(r, c)] = colors.pop() if len(colors) == 1 else None
    preds["nearest"] = nearest

    # Whole-room plurality instead of region plurality.
    lab_f, _, _ = _vote_map(furnished, net_f, bg, div)
    room_tally = {q: {} for q in ROOMS}
    h, w = furnished.shape
    for r, c in markers:
        color = int(furnished[r, c])
        rooms = {_room_of(y, x, n) for y in range(max(0, r - 1), min(h, r + 2))
                 for x in range(max(0, c - 1), min(w, c + 2)) if lab_f[y, x] >= 0}
        for q in rooms - {None}:
            room_tally[q][color] = room_tally[q].get(color, 0) + 1
    preds["room_vote"] = {p: _winner(room_tally[_room_of(*p, n)]) for p in dot_cells}
    return preds


def _shortcut_grids(input_grid):
    info = _read(input_grid)
    grids = {}
    for layout in ("center_turn", "mirror", "copy"):
        grids[layout] = _recolor(_furnish(info, layout), info["dots"], info["bg"], info["div"])
    copied = _furnish(info, "pinwheel", with_dots=True)
    grids["copy_template_dots"] = _recolor(copied, _isolated(copied, info["bg"], info["div"]) & ~info["cross"],
                                           info["bg"], info["div"])
    return grids


def _evidence(input_grid, output_grid):
    """Which shortcuts give a different answer on this pair (True = refuted)."""
    out = np.asarray(output_grid, dtype=int)
    info = _read(input_grid)
    result = {}
    for name, colors in _shortcut_dot_colors(input_grid).items():
        result[name] = any((info["div"] if col is None else col) != out[p] for p, col in colors.items())
    for name, grid in _shortcut_grids(input_grid).items():
        result[name] = not np.array_equal(grid, out)
    return result


def _pair_facts(input_grid, output_grid):
    """Family facts measured from a pair's grids only."""
    info = _read(input_grid)
    out = np.asarray(output_grid, dtype=int)
    n, bg, div = info["n"], info["bg"], info["div"]
    room, content = _template(info)
    canon = _apply(_inverse(LAYOUTS["pinwheel"][room]), content) if room else None
    net_out = (out != bg) & ~info["dots"]
    lab, tallies, winners = _vote_map(np.where(info["dots"], bg, out), net_out, bg, div)
    queried = {int(lab[r, c]) for r, c in np.argwhere(info["dots"])}
    contested = any(len(tallies[f]) >= 2 for f in queried)
    winners_q = {winners[f] for f in queried}
    dot_rooms = {_room_of(int(r), int(c), n) for r, c in np.argwhere(info["dots"])}
    cross_markers = int(np.sum(info["cross"] & (info["g"] != div)))
    hidden_junction_markers = 0
    for r, c in np.argwhere(info["cross"] & (info["g"] != div)):
        if r == n and c != n and 0 < c < 2 * n and not (info["net"][r - 1, c] or info["net"][r + 1, c]):
            hidden_junction_markers += 1
        if c == n and r != n and 0 < r < 2 * n and not (info["net"][r, c - 1] or info["net"][r, c + 1]):
            hidden_junction_markers += 1
    marker_colors = sorted(int(v) for v in np.unique(info["g"][info["net"]]) if v not in (bg, div))
    return {
        "n": n, "bg": bg, "div": div, "room": room, "canon": canon,
        "chiral": bool(canon is not None and _is_chiral(canon != bg)),
        "dots": int(info["dots"].sum()), "dot_rooms": dot_rooms,
        "template_dots": room in dot_rooms, "queried_faces": len(queried),
        "contested": contested, "winners": winners_q, "marker_colors": marker_colors,
        "cross_markers": cross_markers, "hidden_junction_markers": hidden_junction_markers,
        "fragment_cells": int(np.sum(content != bg)) if content is not None else 0,
        "vertices": len(_vertices(canon != bg)) if canon is not None else 0,
        "kinds": _vertex_kinds(canon != bg) if canon is not None else {},
    }


# --------------------------------------------------------------------------
# Fragment (furnishing) construction
# --------------------------------------------------------------------------

def _vertices(mask):
    n = mask.shape[0]
    found = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        up = r > 0 and bool(mask[r - 1, c])
        down = r < n - 1 and bool(mask[r + 1, c])
        left = c > 0 and bool(mask[r, c - 1])
        right = c < n - 1 and bool(mask[r, c + 1])
        degree = up + down + left + right
        if degree != 2 or not ((up and down) or (left and right)):
            found.append((r, c))
    return found


def _vertex_kinds(mask):
    """Counts of free ends, room-border contacts, corners, T and plus junctions."""
    n = mask.shape[0]
    kinds = {"free": 0, "border": 0, "corner": 0, "T": 0, "plus": 0}
    for r, c in _vertices(mask):
        degree = sum(1 for y, x in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                     if 0 <= y < n and 0 <= x < n and mask[y, x])
        if degree == 1:
            kinds["border" if r in (0, n - 1) or c in (0, n - 1) else "free"] += 1
        else:
            kinds[{2: "corner", 3: "T", 4: "plus"}[degree]] += 1
    return kinds


def _segment_cells(seg):
    orient, coord, a, b = seg[:4]
    return [(coord, x) if orient == 0 else (x, coord) for x in range(a, b + 1)]


def _mask_clean(mask):
    """One-cell-wide, no diagonal-only contacts, border cells are perpendicular ends."""
    n = mask.shape[0]
    if np.any(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]):
        return False
    a, b, c, d = mask[:-1, :-1], mask[:-1, 1:], mask[1:, :-1], mask[1:, 1:]
    if np.any(a & d & ~b & ~c) or np.any(b & c & ~a & ~d):
        return False
    if mask[0, 0] or mask[0, -1] or mask[-1, 0] or mask[-1, -1]:
        return False
    for side in (mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]):
        if np.any(side[:-1] & side[1:]):
            return False
    inward = ((mask[0, :], mask[1, :]), (mask[-1, :], mask[-2, :]),
              (mask[:, 0], mask[:, 1]), (mask[:, -1], mask[:, -2]))
    return all(not np.any(edge & ~inner) for edge, inner in inward) and n >= 5


def _segments_clean(segs, n):
    mask = np.zeros((n, n), dtype=bool)
    for seg in segs:
        for r, c in _segment_cells(seg):
            mask[r, c] = True
    if not _mask_clean(mask):
        return None
    for i, s in enumerate(segs):
        for t in segs[i + 1:]:
            if s[0] == t[0] and s[2] - 1 <= t[3] and t[2] <= s[3] + 1 and abs(s[1] - t[1]) < 3:
                return None
    for seg in segs:
        orient, coord, a, b, free_a, free_b = seg
        for end, step, free in ((a, -1, free_a), (b, 1, free_b)):
            if not free:
                continue
            ahead = end + step
            if not (0 <= end + 2 * step < n):
                return None
            fy, fx = (coord, ahead) if orient == 0 else (ahead, coord)
            ey, ex = (coord, end) if orient == 0 else (end, coord)
            box = mask[max(0, fy - 1):fy + 2, max(0, fx - 1):fx + 2].copy()
            box[ey - max(0, fy - 1), ex - max(0, fx - 1)] = False
            if box.any():
                return None
    return mask


def _propose_segment(rng, n, segs):
    free_ends = [(s, s[2] if s[4] else s[3]) for s in segs if s[4] or s[5]]
    grow = None
    if free_ends and rng.random() < 0.3:
        # Grow a perpendicular segment out of an existing free end (corner or T).
        old, end = free_ends[int(rng.integers(len(free_ends)))]
        orient, coord, grow = 1 - old[0], end, old[1]
        if any(s[0] == orient and s[1] == coord for s in segs) or not 2 <= coord <= n - 3:
            return None
    else:
        orient = int(rng.integers(2))
        used = {s[1] for s in segs if s[0] == orient}
        options = [c for c in range(2, n - 2) if c not in used]
        if not options:
            return None
        coord = int(rng.choice(options))
    perp = sorted({s[1] for s in segs if s[0] != orient and s[2] <= coord <= s[3]})
    anchors = [0] + perp + [n - 1]
    if grow is None and (not segs or rng.random() < 0.58):
        i, j = sorted(int(x) for x in rng.choice(len(anchors), 2, replace=False))
        a, b = anchors[i], anchors[j]
        if b - a < 2:
            return None
        return (orient, coord, a, b, False, False)
    i = anchors.index(grow) if grow is not None else int(rng.integers(len(anchors)))
    start = anchors[i]
    if grow is not None and rng.random() < 0.5:
        j = i + (1 if (i == 0 or (i < len(anchors) - 1 and rng.random() < 0.5)) else -1)
        a, b = sorted((start, anchors[j]))
        return (orient, coord, a, b, False, False) if b - a >= 2 else None
    if i == 0:
        step = 1
    elif i == len(anchors) - 1:
        step = -1
    else:
        step = 1 if rng.random() < 0.5 else -1
    nxt = anchors[i + step]
    barrier = n if (step == 1 and nxt == n - 1) else (-1 if (step == -1 and nxt == 0) else nxt)
    lo, hi = (start + 2, barrier - 3) if step == 1 else (barrier + 3, start - 2)
    if lo > hi:
        return None
    end = int(rng.integers(lo, hi + 1))
    a, b = sorted((start, end))
    return (orient, coord, a, b, a == end, b == end)


def _segment_cap(n):
    return 3 + (n - MIN_ROOM) // 2


def _attach(segs, seg):
    """Add `seg`; a perpendicular free end lying on it becomes a corner or T."""
    orient, coord, a, b = seg[:4]
    updated = []
    for old in segs:
        o, c, a0, b0, fa, fb = old
        if o != orient and a <= c <= b:
            if fa and a0 == coord:
                fa = False
            if fb and b0 == coord:
                fb = False
        updated.append((o, c, a0, b0, fa, fb))
    return updated + [seg]


def _sample_fragment(rng, n, target):
    target = min(target, _segment_cap(n))
    best = None
    for _ in range(12):
        segs = []
        tries = 0
        while len(segs) < target and tries < 40:
            tries += 1
            seg = _propose_segment(rng, n, segs)
            if seg is not None:
                trial = _attach(segs, seg)
                if _segments_clean(trial, n) is not None:
                    segs = trial
        if len(segs) < 2:
            continue
        mask = _segments_clean(segs, n)
        if mask is None or len(_vertices(mask)) < 3:
            continue
        if len(segs) >= target:
            return mask, segs
        if best is None or len(segs) > len(best[1]):
            best = (mask, segs)
    return best


# --------------------------------------------------------------------------
# Pair construction
# --------------------------------------------------------------------------

def _cross_vertex_candidates(skel, n):
    size = 2 * n + 1
    cands = [(0, n), (size - 1, n), (n, 0), (n, size - 1), (n, n)]
    for r in range(1, size - 1):
        if r != n and (skel[r, n - 1] or skel[r, n + 1]):
            cands.append((r, n))
    for c in range(1, size - 1):
        if c != n and (skel[n - 1, c] or skel[n + 1, c]):
            cands.append((n, c))
    return cands


def _place_dots(rng, faces, clear_cells, per_face):
    chosen = []
    for face in faces:
        cells = list(clear_cells[face])
        order = rng.permutation(len(cells))
        want, got = int(per_face(face)), 0
        for idx in order:
            r, c = cells[int(idx)]
            if all(max(abs(r - y), abs(c - x)) >= 2 for y, x in chosen):
                chosen.append((r, c))
                got += 1
                if got >= want:
                    break
    return chosen


def _build_pair(rng, spec):
    n, room = spec["n"], spec["room"]
    bg, div, colors = spec["bg"], spec["div"], list(spec["colors"])
    size = 2 * n + 1
    for _ in range(spec.get("attempts", 60)):
        frag = _sample_fragment(rng, n, spec["segments"])
        if frag is None:
            continue
        mask, _segs = frag
        if spec["chiral"] and not _is_chiral(mask):
            continue
        verts = _vertices(mask)
        marked = [v for v in verts if rng.random() < spec["marker_prob"]]
        if len(marked) < 2:
            pick = rng.choice(len(verts), min(len(verts), 3), replace=False)
            marked = [verts[int(i)] for i in pick]
        skel = np.zeros((size, size), dtype=bool)
        skel[n, :] = True
        skel[:, n] = True
        for q in ROOMS:
            skel[_room_slice(q, n)] |= _apply(LAYOUTS["pinwheel"][q], mask)
        cands = _cross_vertex_candidates(skel, n)
        order = rng.permutation(len(cands))
        cross_marked = []
        for idx in order:
            cell = cands[int(idx)]
            if len(cross_marked) < spec["cross_count"] and all(
                    abs(cell[0] - y) + abs(cell[1] - x) > 1 for y, x in cross_marked):
                cross_marked.append(cell)
        for _color_try in range(14):
            canon = np.full((n, n), bg, dtype=int)
            canon[mask] = div
            for v in marked:
                canon[v] = int(rng.choice(colors))
            full = np.full((size, size), bg, dtype=int)
            full[n, :] = div
            full[:, n] = div
            for q in ROOMS:
                copy = _apply(LAYOUTS["pinwheel"][q], canon)
                region = full[_room_slice(q, n)]
                region[copy != bg] = copy[copy != bg]
            for cell in cross_marked:
                full[cell] = int(rng.choice(colors))
            present = {int(v) for v in np.unique(full)} - {bg, div}
            if len(present) < min(2, len(colors)):
                continue
            net = full != bg
            lab, tallies, winners = _vote_map(full, net, bg, div)
            clear = ~_dilate(net)
            clear_cells = {}
            for r, c in np.argwhere(clear):
                clear_cells.setdefault(int(lab[r, c]), []).append((int(r), int(c)))
            eligible = [f for f in clear_cells if _legible(tallies[f])]
            face_room = {f: _room_of(*clear_cells[f][0], n) for f in eligible}
            others = [q for q in ROOMS if q != room]
            if sum(any(face_room[f] == q for f in eligible) for q in others) < 2:
                continue
            queried = []
            for q in others:
                pool = [f for f in eligible if face_room[f] == q]
                if not pool:
                    continue
                if spec["dense"]:
                    queried += pool
                else:
                    k = int(rng.integers(1, len(pool) + 1))
                    queried += [pool[int(i)] for i in rng.choice(len(pool), k, replace=False)]
            tpool = [f for f in eligible if face_room[f] == room]
            if spec["template_dots"]:
                if not tpool:
                    continue
                k = len(tpool) if spec["dense"] else int(rng.integers(1, min(2, len(tpool)) + 1))
                queried += [tpool[int(i)] for i in rng.choice(len(tpool), k, replace=False)]
            hi = 4 if spec["dense"] else 3
            dots = _place_dots(rng, queried, clear_cells, lambda f: rng.integers(1, hi))
            if spec.get("novel") is not None and spec["novel"] not in {winners[lab[d]] for d in dots}:
                continue
            inp = np.full((size, size), bg, dtype=int)
            inp[n, :] = full[n, :]
            inp[:, n] = full[:, n]
            sl = _room_slice(room, n)
            inp[sl] = full[sl]
            out = full.copy()
            for r, c in dots:
                inp[r, c] = div
                out[r, c] = winners[lab[r, c]]
            if not np.array_equal(solve(inp), out):
                continue
            ev = _evidence(inp, out)
            facts = _pair_facts(inp, out)
            need = spec.get("require", ())
            if not all(ev.get(k, facts.get(k)) for k in need):
                continue
            return {"input": inp, "output": out}, ev, facts
    return None


# --------------------------------------------------------------------------
# Puzzle construction
# --------------------------------------------------------------------------

def _checked_count(name, value, default, low, high):
    if value is None:
        value = default
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _palette(rng):
    order = [int(x) for x in rng.permutation(10)]
    bg_main = 0 if rng.random() < 0.6 else order[0]
    rest = [c for c in order if c != bg_main]
    div_main, bg_alt, div_alt = rest[0], rest[1], rest[2]
    pool = rest[3:3 + int(rng.integers(3, 6))]
    leftover = [c for c in rest[3:] if c not in pool]
    novel = leftover[0] if leftover and rng.random() < 0.55 else None
    return {"bg": bg_main, "div": div_main, "bg_alt": bg_alt, "div_alt": div_alt,
            "pool": pool, "novel": novel}


def _room_plan(rng, num_train, num_test):
    if rng.random() < 0.5:
        fixed = str(rng.choice(ROOMS, p=[0.46, 0.18, 0.18, 0.18]))
        return [fixed] * num_train, [fixed] * num_test, "fixed"
    first = [str(x) for x in rng.choice(ROOMS, 2, replace=False)]
    train = first + [str(rng.choice(ROOMS)) for _ in range(num_train - 2)]
    order = rng.permutation(num_train)
    train = [train[int(i)] for i in order]
    seen = sorted(set(train))
    tests = [str(rng.choice(seen)) for _ in range(num_test)]
    return train, tests, "mixed"


def _pair_spec(rng, palette, room, *, test, first_test, evidence, size_hint=None):
    if test:
        n = int(rng.integers(9, MAX_ROOM + 1)) if first_test else int(rng.integers(MIN_ROOM, MAX_ROOM + 1))
        segments = int(rng.integers(4, 7)) if first_test else int(rng.integers(3, 6))
    else:
        n = size_hint if size_hint is not None else int(rng.integers(MIN_ROOM, 13))
        segments = int(rng.integers(2, 5))
    bg, div = palette["bg"], palette["div"]
    if not evidence and rng.random() < 0.22:
        bg = palette["bg_alt"]
    if not evidence and rng.random() < 0.22:
        div = palette["div_alt"]
    pool = list(palette["pool"])
    k = int(rng.integers(2, min(4, len(pool)) + 1))
    colors = [pool[int(i)] for i in rng.choice(len(pool), k, replace=False)]
    novel = None
    if first_test and palette["novel"] is not None:
        novel = palette["novel"]
        colors = colors[:max(1, k - 1)] + [novel]
    spec = {"n": n, "room": room, "bg": bg, "div": div, "colors": colors,
            "segments": segments, "marker_prob": float(rng.uniform(0.7, 1.0)),
            "cross_count": int(rng.choice([0, 1, 2, 3, 4], p=[0.08, 0.2, 0.3, 0.26, 0.16])),
            "dense": bool(rng.random() < 0.5), "template_dots": bool(rng.random() < 0.6),
            "chiral": evidence, "novel": novel, "require": ("no_pinwheel",)}
    if evidence:
        spec["template_dots"] = True
        spec["cross_count"] = int(rng.integers(2, 5))
        spec["require"] = ("no_pinwheel", "rotate_colors", "ignore_cross", "nearest",
                           "room_vote", "contested", "template_dots", "chiral")
    elif test and first_test:
        spec["cross_count"] = max(spec["cross_count"], 1)
        spec["require"] = ("no_pinwheel", "rotate_colors")
    return spec


def _try_puzzle(rng, num_train, num_test):
    palette = _palette(rng)
    train_rooms, test_rooms, _mode = _room_plan(rng, num_train, num_test)
    sizes = [int(x) for x in rng.integers(MIN_ROOM, 13, size=num_train)]
    if len(set(sizes)) == 1 and num_train > 1:
        sizes[-1] = MIN_ROOM + (sizes[-1] - MIN_ROOM + 3) % (13 - MIN_ROOM)
    evidence_slot = int(rng.integers(num_train))
    pairs = {"train": [], "test": []}
    seen = set()
    for split, rooms in (("train", train_rooms), ("test", test_rooms)):
        for index, room in enumerate(rooms):
            spec = _pair_spec(rng, palette, room, test=split == "test",
                              first_test=split == "test" and index == 0,
                              evidence=split == "train" and index == evidence_slot,
                              size_hint=sizes[index] if split == "train" else None)
            pair = None
            for attempt in range(4):
                if attempt == 2 and spec["novel"] is not None:
                    # Fall back to familiar marker colours if the new colour cannot win.
                    spec["novel"] = None
                    spec["colors"] = [c for c in spec["colors"] if c != palette["novel"]]
                    if len(spec["colors"]) < 2:
                        spec["colors"] = list(palette["pool"][:2])
                built = _build_pair(rng, spec)
                if built is not None and _key(built[0]["input"]) not in seen:
                    pair = built[0]
                    break
            if pair is None:
                return None
            seen.add(_key(pair["input"]))
            pairs[split].append({"input": pair["input"].tolist(), "output": pair["output"].tolist()})
    return pairs


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _checked_count("num_train", num_train, 4, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    for _ in range(30):
        puzzle = _try_puzzle(rng, num_train, num_test)
        if puzzle is None:
            continue
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle
    raise RuntimeError(f"could not construct a puzzle for seed {seed}")


# --------------------------------------------------------------------------
# Validation and self test
# --------------------------------------------------------------------------

def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "train/test keys")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "pair counts")
    seen = set()
    facts = {"train": [], "test": []}
    evidence = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split} {index}"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, f"{label} keys")
            inp = np.asarray(pair["input"])
            out = np.asarray(pair["output"])
            for grid in (inp, out):
                _require(grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer), f"{label} integer grid")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, f"{label} size")
                _require(grid.min() >= 0 and grid.max() <= 9, f"{label} colours")
            _require(inp.shape == out.shape, f"{label} same shape")
            _require(inp.shape[0] == inp.shape[1] and inp.shape[0] % 2 == 1, f"{label} odd square")
            n = inp.shape[0] // 2
            _require(MIN_ROOM <= n <= MAX_ROOM, f"{label} room size")
            key = _key(inp.astype(int))
            _require(key not in seen, f"{label} duplicate input")
            seen.add(key)
            info = _read(inp)
            bg, div = info["bg"], info["div"]
            _require(bg != div, f"{label} roles")
            cross = info["cross"]
            # The plus is complete: every plus cell is divider or a marker, never background.
            _require(not np.any(inp[cross] == bg), f"{label} plus intact")
            _require(np.array_equal(inp[cross], out[cross]), f"{label} plus unchanged")
            # Exactly one furnished room; others hold only isolated divider dots.
            _require(len(info["furnished"]) == 1, f"{label} one furnished room")
            room, content = _template(info)
            canon = _apply(_inverse(LAYOUTS["pinwheel"][room]), content)
            _require(_mask_clean(canon != bg), f"{label} clean one-cell network")
            _require(len(_vertices(canon != bg)) >= 3, f"{label} network vertices")
            # Markers only occupy network vertices of the room network or of the plus.
            verts = set(_vertices(canon != bg))
            for r, c in np.argwhere((canon != bg) & (canon != div)):
                _require((int(r), int(c)) in verts, f"{label} marker off vertex")
            dots = info["dots"]
            _require(dots.sum() >= 2, f"{label} query dots")
            # Output: rooms are the pinwheel of the canonical room network plus untouched dots.
            for q in ROOMS:
                sl = _room_slice(q, n)
                expect = _apply(LAYOUTS["pinwheel"][q], canon)
                got = out[sl].copy()
                qdots = dots[sl]
                _require(np.all(expect[qdots] == bg), f"{label} dot collides with furnishing")
                got[qdots] = bg
                _require(np.array_equal(got, expect), f"{label} pinwheel room {q}")
            _require(np.all(_isolated(np.where(dots, div, out), bg, div)[dots]), f"{label} dots isolated in output")
            skel = (out != bg) & ~dots
            plus_vertices = set(_cross_vertex_candidates(skel, n))
            for r, c in np.argwhere(cross & (inp != div)):
                _require((int(r), int(c)) in plus_vertices, f"{label} plus marker off vertex")
            _require(np.all(out[dots] != div) and np.all(out[dots] != bg), f"{label} every dot recoloured")
            net_out = (out != bg) & ~dots
            lab, tallies, _ = _vote_map(np.where(dots, bg, out), net_out, bg, div)
            for r, c in np.argwhere(dots):
                _require(_legible(tallies[lab[r, c]]), f"{label} queried vote not legible")
            _require(int(np.sum(cross & (inp != div))) <= 4, f"{label} at most four plus markers")
            _require(np.array_equal(solve(inp), out), f"{label} solve mismatch")
            facts[split].append(_pair_facts(inp, out))
            evidence[split].append(_evidence(inp, out))
            _require(evidence[split][-1]["no_pinwheel"], f"{label} solvable without the pinwheel")
            _require(len(facts[split][-1]["dot_rooms"] - {room}) >= 2, f"{label} dots in empty rooms")
    # Training demonstrations must refute each simpler alternative at least once.
    for name in DOT_SHORTCUTS + LAYOUT_SHORTCUTS:
        _require(any(ev[name] for ev in evidence["train"]), f"training does not refute {name}")
    _require(any(f["contested"] for f in facts["train"]), "training lacks a contested plurality")
    _require(any(f["chiral"] for f in facts["train"]), "training lacks a chiral furnishing")
    _require(any(f["template_dots"] for f in facts["train"]), "training never shows template dots staying")
    train_rooms = {f["room"] for f in facts["train"]}
    _require(all(f["room"] in train_rooms for f in facts["test"]), "test template room unseen in training")


def _canonical_puzzle(puzzle):
    return tuple(tuple((np.asarray(p["input"]).tobytes(), np.asarray(p["output"]).tobytes())
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test() -> None:
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (4.0, 1), (True, 1)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {bad}")
    stats = {"n": set(), "rooms": set(), "mixed": 0, "novel_win": 0, "palettes": set(),
             "segments": set(), "hidden": 0, "cross_free": 0, "sizes_per_puzzle": [],
             "rotate_test": 0, "dense_dots": 0, "kinds": {}}
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed % 14 == 0:
            _require(_canonical_puzzle(puzzle) == _canonical_puzzle(generate_puzzle(seed)), "determinism")
        signatures.add(_canonical_puzzle(puzzle))
        train_colors = set()
        rooms = set()
        sizes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _pair_facts(pair["input"], pair["output"])
                stats["n"].add(f["n"])
                rooms.add(f["room"])
                sizes.add(f["n"])
                stats["palettes"].add((f["bg"], f["div"], tuple(f["marker_colors"])))
                stats["segments"].add(f["fragment_cells"])
                stats["hidden"] += f["hidden_junction_markers"] > 0
                stats["cross_free"] += f["cross_markers"] == 0
                stats["dense_dots"] += f["dots"] >= 12
                for kind, count in f["kinds"].items():
                    stats["kinds"][kind] = stats["kinds"].get(kind, 0) + (count > 0)
                if split == "train":
                    train_colors |= set(f["marker_colors"])
                else:
                    if f["winners"] - train_colors:
                        stats["novel_win"] += 1
                    stats["rotate_test"] += _evidence(pair["input"], pair["output"])["rotate_colors"]
        stats["rooms"] |= rooms
        stats["mixed"] += len({_pair_facts(p["input"], p["output"])["room"] for p in puzzle["train"]}) > 1
        stats["sizes_per_puzzle"].append(len(sizes))
    _require(len(signatures) == 42, "distinct puzzles")
    _require(len(stats["n"]) >= 7, "room sizes vary")
    _require(stats["rooms"] == set(ROOMS), "every room serves as a template")
    _require(stats["mixed"] >= 10, "some puzzles mix template rooms")
    _require(stats["novel_win"] >= 8, "tests sometimes introduce a winning new marker colour")
    _require(len(stats["palettes"]) >= 100, "palettes vary")
    _require(stats["hidden"] >= 20, "plus markers sometimes sit on junctions created by furnishing")
    _require(stats["rotate_test"] >= 30, "tests usually depend on fixed plus markers")
    _require(min(stats["sizes_per_puzzle"]) >= 2, "sizes vary inside each puzzle")
    _require(stats["dense_dots"] >= 40, "dense query pairs occur")
    floors = {"border": 150, "T": 80, "plus": 30, "free": 30, "corner": 12}
    for kind, floor in floors.items():
        _require(stats["kinds"].get(kind, 0) >= floor, f"furnishing topology {kind} too rare")
    for counts in ((2, 1), (10, 4), (5, 3)):
        puzzle = generate_puzzle(7, *counts)
        validate_generated_puzzle(puzzle)
        _require((len(puzzle["train"]), len(puzzle["test"])) == counts, "explicit counts honoured")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
