"""Shattering mirror (v2): a composition of ARC tasks 11dc524f and 27a28665.

Parent A (11dc524f): a patterned object approaches a solid marker until it
touches it; the marker is replaced by the object's reflection across the
shared face line, drawn in the marker colour.  Parts of the object outside
the marker's own lanes are reflected as well (A train 1 and test).

Parent B (27a28665): one-colour foreground cells are partitioned into
four-neighbour (edge-connected) components; diagonal contact does not join
cells, and the literal foreground colour is irrelevant.

Composed rule: every edge-connected ink piece is a separate rigid body pulled
toward the solid mirror rectangle.  A piece stops when its next step would
enter the mirror or a piece that has already stopped, so pieces stack; pieces
beyond the mirror's side rest on other pieces.  The old ink and the whole
mirror are erased, the settled pile is drawn in the ink colour, and its
reflection across the mirror's facing edge is drawn in the mirror colour.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


DIRECTIONS = ("down", "up", "right", "left")
_AXIS = {"down": "vertical", "up": "vertical", "right": "horizontal", "left": "horizontal"}
_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_N8 = _N4 + _DIAG
B_PROFILES = ((5,), (4, 1), (3, 1, 1), (1, 1, 1, 1, 1))
SHORTCUTS = ("rigid_whole", "rigid_8", "sand", "no_stack", "touching_only", "copy",
             "keep_mirror", "clip", "lane_clip", "facing_only")
# Every test must refute these: ignoring B (one rigid object, eight-connected
# bodies, or no bodies at all) or ignoring A's flip gives a wrong answer.
TEST_REQUIRED = frozenset(("rigid_whole", "rigid_8", "sand", "copy"))
# Every training set demonstrates these distinctions.
TRAIN_REQUIRED = frozenset(("rigid_whole", "rigid_8", "sand", "copy", "keep_mirror"))
MAX_THICKNESS = 5
MAX_INK = 40

# sha256(compact JSON)[:16] of every parent input grid (task and selected
# variation of both parents).  Generated inputs must never reproduce them.
_PARENT_INPUT_HASHES = frozenset((
    "1a663d01296783c0", "a010bf5458e1fc45", "48b72aa8528cccf3", "af8011e58ae75712",
    "06e66a3b93f3e0d4", "09d9db18dcd643ad", "42d4e6562ee4adbb", "6a4ca8ad4990acc0",
    "648000dd73797ab6", "22dcf1e33011e401", "c505d334d66dc515", "783af6e74f4a61ad",
    "b1ddfb986219c5ce", "eaf6bc992fe89d11", "c6340fa112ae2e8b", "12126a70633c78b3",
    "367373b90e8c87a1", "459046e62cfe5ff6", "9fdd1e2f2a8ff3f8", "6d94189b422f18bf",
    "b1df540048805132", "553652323cfcbc33", "e1573576c36bc44c", "3fdce12c7e5bd387",
))


# ---------------------------------------------------------------------------
# Generic geometry
# ---------------------------------------------------------------------------

def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), max(rows), min(cols), max(cols)


def _components(cells, offsets):
    remaining = set(cells)
    found = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        piece, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in offsets:
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    piece.add(point)
                    stack.append(point)
        found.append(piece)
    return found


def _is_filled_rectangle(cells):
    r0, r1, c0, c1 = _bbox(cells)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _union(bodies):
    cells = set()
    for body in bodies:
        cells |= set(body)
    return cells


def _to_direction(grid, direction):
    """Map a canonical (mirror below, falling down) grid to `direction`."""
    if direction == "down":
        return grid
    if direction == "up":
        return grid[::-1, :]
    if direction == "right":
        return grid.T
    return grid.T[:, ::-1]


def _from_direction(grid, direction):
    """Inverse of `_to_direction`: bring a grid into the canonical frame."""
    if direction == "down":
        return grid
    if direction == "up":
        return grid[::-1, :]
    if direction == "right":
        return grid.T
    return grid[:, ::-1].T


# ---------------------------------------------------------------------------
# Input-only reading and solving
# ---------------------------------------------------------------------------

def _direction(ink, mirror):
    ir0, ir1, ic0, ic1 = _bbox(ink)
    mr0, mr1, mc0, mc1 = _bbox(mirror)
    options = []
    if ir1 < mr0:
        options.append("down")
    if ir0 > mr1:
        options.append("up")
    if ic1 < mc0:
        options.append("right")
    if ic0 > mc1:
        options.append("left")
    if len(options) != 1:
        raise ValueError("ink and mirror are not separated along exactly one axis")
    return options[0]


def _parse(grid):
    """Read the three colour roles and the motion direction from an input."""
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError("grid must be two-dimensional")
    values, counts = np.unique(grid, return_counts=True)
    if len(values) != 3:
        raise ValueError("expected exactly background, ink and mirror colours")
    order = np.argsort(-counts, kind="stable")
    if counts[order[0]] == counts[order[1]]:
        raise ValueError("background is not the strict majority colour")
    background = int(values[order[0]])
    foreground = [int(v) for v in values if int(v) != background]
    cells = {v: {(int(r), int(c)) for r, c in np.argwhere(grid == v)} for v in foreground}
    rectangles = [v for v in foreground if _is_filled_rectangle(cells[v])]
    if len(rectangles) != 1:
        raise ValueError("exactly one foreground colour must form a filled rectangle")
    mirror = rectangles[0]
    ink = foreground[0] if foreground[1] == mirror else foreground[1]
    return {
        "background": background, "ink": ink, "mirror": mirror,
        "ink_cells": cells[ink], "mirror_cells": cells[mirror],
        "direction": _direction(cells[ink], cells[mirror]),
    }


def _read(grid):
    """Parse a grid and rotate it into the canonical frame (mirror below)."""
    grid = np.asarray(grid)
    info = _parse(grid)
    canonical = np.ascontiguousarray(_from_direction(grid, info["direction"]))
    cinfo = _parse(canonical)
    if cinfo["direction"] != "down":
        raise ValueError("canonical frame is not downward")
    cinfo["source_direction"] = info["direction"]
    cinfo["grid"] = canonical
    return cinfo


def _fall(bodies, blocked, height, stacking=True):
    """Slide rigid bodies down until each is blocked (iterative relaxation).

    A body stops on a blocked cell, on the bottom edge, or (when stacking) on
    the current cells of another body.  Bodies blocked by bodies that later
    move get another chance in the next sweep.  The order of bodies is kept.
    """
    bodies = [set(body) for body in bodies]
    owner = {}
    if stacking:
        for index, body in enumerate(bodies):
            for cell in body:
                owner[cell] = index
    moved = True
    while moved:
        moved = False
        order = sorted(range(len(bodies)),
                       key=lambda i: -max(r for r, _ in bodies[i]) if bodies[i] else 0)
        for index in order:
            body = bodies[index]
            if not body:
                continue
            drop = height
            # Scan below every cell: a body's lane may have gaps holding
            # another body (for example eight-connected shortcut bodies).
            for r, c in body:
                rr = r + 1
                while rr < height and (rr, c) not in blocked and (rr, c) not in owner:
                    rr += 1
                if rr < height and owner.get((rr, c)) == index:
                    continue
                drop = min(drop, rr - r - 1)
                if drop == 0:
                    break
            if drop > 0:
                if stacking:
                    for cell in body:
                        del owner[cell]
                body = {(r + drop, c) for r, c in body}
                bodies[index] = body
                if stacking:
                    for cell in body:
                        owner[cell] = index
                moved = True
    return [frozenset(body) for body in bodies]


def _reflect(cells, face):
    return {(2 * face - 1 - r, c) for r, c in cells}


def _render(shape, background, ink, mirror, pile, image, keep=()):
    out = np.full(shape, background, dtype=np.int64)
    height, width = shape
    for r, c in keep:
        out[r, c] = mirror
    for r, c in image:
        if 0 <= r < height and 0 <= c < width:
            out[r, c] = mirror
    for r, c in pile:
        out[r, c] = ink
    return out


def _canonical_solution(cinfo):
    pieces = _components(cinfo["ink_cells"], _N4)
    settled = _fall(pieces, cinfo["mirror_cells"], cinfo["grid"].shape[0])
    pile = _union(settled)
    face = _bbox(cinfo["mirror_cells"])[0]
    return pieces, settled, pile, _reflect(pile, face), face


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the output from the input grid alone."""
    grid = np.asarray(input_grid)
    if grid.ndim != 2:
        raise ValueError("grid must be two-dimensional")
    grid = grid.astype(np.int64)
    try:
        cinfo = _read(grid)
    except ValueError:
        return grid.copy()
    _, _, pile, image, _ = _canonical_solution(cinfo)
    out = _render(cinfo["grid"].shape, cinfo["background"], cinfo["ink"], cinfo["mirror"],
                  pile, image)
    return np.ascontiguousarray(_to_direction(out, cinfo["source_direction"]))


# ---------------------------------------------------------------------------
# Shortcut rules (all computed from the input alone, canonical frame)
# ---------------------------------------------------------------------------

def _shortcut_outputs(cinfo):
    shape = cinfo["grid"].shape
    height = shape[0]
    ink, mirror = cinfo["ink_cells"], cinfo["mirror_cells"]
    face, _, lo, hi = _bbox(mirror)

    def draw(pile, image, keep=False):
        return _render(shape, cinfo["background"], cinfo["ink"], cinfo["mirror"], pile, image,
                       mirror if keep else ())

    def reflected(pile):
        return draw(pile, _reflect(pile, face))

    pieces = _components(ink, _N4)
    settled = _fall(pieces, mirror, height)
    pile = _union(settled)
    image = _reflect(pile, face)
    depth = face - min(r for r, _ in pile)
    touching = _union(body for body in settled if any((r + 1, c) in mirror for r, c in body))
    facing = [piece for piece in pieces if any(lo <= c <= hi for _, c in piece)]
    facing_pile = _union(_fall(facing, mirror, height)) if facing else set()
    return {
        # Parent A alone: the whole ink pattern translates as one rigid object.
        "rigid_whole": reflected(_union(_fall([ink], mirror, height))),
        # Diagonal contact treated as joining (eight-neighbour bodies).
        "rigid_8": reflected(_union(_fall(_components(ink, _N8), mirror, height))),
        # No partition at all: every cell falls on its own (lane compaction).
        "sand": reflected(_union(_fall([{cell} for cell in sorted(ink)], mirror, height))),
        # Pieces pass through one another.
        "no_stack": reflected(_union(_fall(pieces, mirror, height, stacking=False))),
        # Only pieces resting directly on the mirror are reflected.
        "touching_only": draw(pile, _reflect(touching, face)),
        # Parent A's flip ignored: the pile is translated across the face.
        "copy": draw(pile, {(r + depth, c) for r, c in pile}),
        # Parent A's marker replacement ignored: uncovered mirror cells remain.
        "keep_mirror": draw(pile, image, keep=True),
        # Reflection limited to the mirror rectangle.
        "clip": draw(pile, image & mirror),
        # Reflection limited to the mirror's lanes (A reflects overhangs too).
        "lane_clip": draw(pile, {(r, c) for r, c in image if lo <= c <= hi}),
        # Pieces with no lane over the mirror are ignored.
        "facing_only": reflected(facing_pile),
    }


# ---------------------------------------------------------------------------
# Independent canonical settle used by construction and validation
# ---------------------------------------------------------------------------

def _topo_settle(pieces, face, lanes):
    """Topological settle in the canonical frame.

    A piece settles after every piece below one of its cells in a shared
    lane.  Lanes inside `lanes` (inclusive) end at the mirror face; other lanes
    have no floor.  Returns None for interlocked pieces, unsupported pieces
    (nothing below them) or settled cells reaching the face line.
    """
    lo, hi = lanes
    count = len(pieces)
    columns = {}
    for index, piece in enumerate(pieces):
        for r, c in piece:
            columns.setdefault(c, []).append((r, index))
    below = [set() for _ in range(count)]
    for items in columns.values():
        items.sort()
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                upper, lower = items[a][1], items[b][1]
                if upper != lower:
                    below[upper].add(lower)
    order, done, remaining = [], set(), set(range(count))
    while remaining:
        ready = sorted(i for i in remaining if below[i] <= done)
        if not ready:
            return None
        for index in ready:
            order.append(index)
            done.add(index)
            remaining.discard(index)
    top = {}
    settled = [None] * count
    for index in order:
        drop = None
        for r, c in pieces[index]:
            limit = top.get(c, face if lo <= c <= hi else None)
            if limit is None:
                continue
            value = limit - r - 1
            drop = value if drop is None else min(drop, value)
        if drop is None or drop < 0:
            return None
        moved = {(r + drop, c) for r, c in pieces[index]}
        if any(r >= face for r, _ in moved):
            return None
        for r, c in moved:
            top[c] = min(top.get(c, face), r)
        settled[index] = moved
    return settled


def _support_levels(settled, mirror):
    """Shortest support chain to the mirror for each settled piece."""
    owner = {}
    for index, body in enumerate(settled):
        for cell in body:
            owner[cell] = index
    level = [None] * len(settled)
    supporters = []
    for index, body in enumerate(settled):
        found = set()
        for r, c in body:
            if (r + 1, c) in mirror:
                level[index] = 0
            other = owner.get((r + 1, c))
            if other is not None and other != index:
                found.add(other)
        supporters.append(found)
    changed = True
    while changed:
        changed = False
        for index in range(len(settled)):
            if level[index] == 0:
                continue
            options = [level[j] + 1 for j in supporters[index] if level[j] is not None]
            if options and (level[index] is None or min(options) < level[index]):
                level[index] = min(options)
                changed = True
    return level


def _pattern_key(cells):
    """Canonical-frame ink pattern, identified with its lane mirror image."""
    r0, _, c0, c1 = _bbox(cells)
    first = tuple(sorted((r - r0, c - c0) for r, c in cells))
    second = tuple(sorted((r - r0, c1 - c) for r, c in cells))
    return min(first, second)


# ---------------------------------------------------------------------------
# Pair analysis (grids only)
# ---------------------------------------------------------------------------

def _analyze_pair(inp, out):
    """Check the family invariants of one pair from its grids and measure it."""
    inp, out = np.asarray(inp), np.asarray(out)
    assert inp.shape == out.shape, "canvas size changes"
    try:
        cinfo = _read(inp)
    except ValueError as error:
        raise AssertionError(f"unreadable input: {error}") from None
    direction = cinfo["source_direction"]
    grid = cinfo["grid"]
    height, width = grid.shape
    cout = np.ascontiguousarray(_from_direction(out, direction))
    ink, mirror = cinfo["ink_cells"], cinfo["mirror_cells"]
    values, counts = np.unique(inp, return_counts=True)
    count = dict(zip(values.tolist(), counts.tolist()))
    assert count[cinfo["background"]] > max(count[cinfo["ink"]], count[cinfo["mirror"]])
    assert not _is_filled_rectangle(ink), "the ink is never a filled rectangle"
    assert len(mirror) >= 2, "the mirror is never a single cell"
    face, bottom, lo, hi = _bbox(mirror)
    thickness, length = bottom - face + 1, hi - lo + 1
    assert 1 <= thickness <= MAX_THICKNESS
    ir0, ir1, ic0, ic1 = _bbox(ink)
    gap = face - ir1 - 1
    assert gap >= 0
    assert ir1 > ir0 and ic1 > ic0, "the ink spans at least two rows and two lanes"
    lanes = sorted({c for _, c in ink})
    assert any(lo <= c <= hi for c in lanes), "some ink lane meets the mirror"
    pieces = _components(ink, _N4)
    settled = _fall(pieces, mirror, height)
    topo = _topo_settle(pieces, face, (lo, hi))
    assert topo is not None, "pieces interlock, are unsupported or pass the face line"
    assert [frozenset(body) for body in topo] == list(settled), "settle methods disagree"
    levels = _support_levels(settled, mirror)
    assert all(level is not None for level in levels), "a piece is not supported"
    pile = _union(settled)
    assert all(r < face for r, _ in pile), "the pile lies in front of the face"
    assert pile != ink, "something must move"
    image = _reflect(pile, face)
    assert all(0 <= r < height for r, _ in image), "the reflection must fit"
    expected = _render(grid.shape, cinfo["background"], cinfo["ink"], cinfo["mirror"], pile, image)
    assert np.array_equal(cout, expected), "output differs from the rule"
    assert np.array_equal(solve(inp), out)
    assert int((out == cinfo["ink"]).sum()) == len(ink) == int((out == cinfo["mirror"]).sum())
    assert not np.array_equal(inp, out)
    shortcuts = _shortcut_outputs(cinfo)
    refuted = frozenset(name for name, guess in shortcuts.items() if not np.array_equal(guess, cout))
    depth = face - min(r for r, _ in pile)
    holes = 0
    for lane in lanes:
        rows = [r for r, c in pile if c == lane]
        holes += (face - min(rows)) - len(rows)
    starts = [max(r for r, _ in piece) for piece in pieces]
    ends = [max(r for r, _ in body) for body in settled]
    overtaking = any(starts[i] < starts[j] and ends[i] > ends[j]
                     for i in range(len(pieces)) for j in range(len(pieces)))
    sizes = sorted((len(piece) for piece in pieces), reverse=True)
    occupied_rows = {r for r, _ in ink}
    band = run = 0
    for r in range(ir0, ir1 + 1):
        run = 0 if r in occupied_rows else run + 1
        band = max(band, run)
    return {
        "direction": direction, "axis": _AXIS[direction], "pieces": len(pieces),
        "profile": tuple(sizes), "cells": len(ink), "lanes": len(lanes), "width": ic1 - ic0 + 1,
        "rows": ir1 - ir0 + 1, "gap": gap, "thickness": thickness, "length": length,
        "depth": depth, "holes": holes, "level": max(levels), "overtaking": overtaking,
        "groups8": len(_components(ink, _N8)),
        "overhang": sum(1 for lane in lanes if not lo <= lane <= hi),
        "outside_pieces": sum(1 for piece in pieces if not any(lo <= c <= hi for _, c in piece)),
        "wall": lo == 0 and hi == width - 1, "square": thickness == length,
        "contact": gap == 0, "colours": (cinfo["background"], cinfo["ink"], cinfo["mirror"]),
        "shape": tuple(inp.shape), "key": _pattern_key(ink), "refuted": refuted,
        "band": band, "isolates": sizes.count(1),
        "b_profile": len(ink) == 5 and tuple(sizes) in B_PROFILES,
        "b_mask": len(ink) == 5 and tuple(sizes) in B_PROFILES and ir1 - ir0 == 2 and ic1 - ic0 == 2,
    }


# ---------------------------------------------------------------------------
# Piece vocabulary and debris layouts (canonical frame: falling down)
# ---------------------------------------------------------------------------

SHAPE_KINDS = ("iso", "domino", "bar", "L", "T", "U", "plus", "S", "comb", "stair", "blob")
ANCHOR_KINDS = ("L", "T", "U", "plus", "S", "comb", "stair", "blob")
LAYOUTS = ("clump", "scatter", "tower", "lattice", "straggler")


def _norm(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _transform(cells, k):
    out = set()
    for r, c in cells:
        if k & 1:
            r, c = c, r
        if k & 2:
            r = -r
        if k & 4:
            c = -c
        out.add((r, c))
    return _norm(out)


def _grow(rng, size, max_h, max_w):
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _N4} - cells)
        allowed = []
        for point in frontier:
            r0, r1, c0, c1 = _bbox(cells | {point})
            if r1 - r0 < max_h and c1 - c0 < max_w:
                allowed.append(point)
        if not allowed:
            return None
        cells.add(allowed[int(rng.integers(len(allowed)))])
    return _norm(cells)


def _shape(rng, kind, max_h, max_w):
    if kind == "iso":
        cells = {(0, 0)}
    elif kind == "domino":
        cells = _transform({(0, 0), (0, 1)}, int(rng.integers(8)))
    elif kind == "bar":
        if rng.random() < 0.65:
            cells = {(0, c) for c in range(int(rng.integers(3, 7)))}
        else:
            cells = {(r, 0) for r in range(int(rng.integers(3, 6)))}
    elif kind == "L":
        a, b = int(rng.integers(2, 5)), int(rng.integers(2, 4))
        cells = _transform({(r, 0) for r in range(a)} | {(a - 1, c) for c in range(b)},
                           int(rng.integers(8)))
    elif kind == "T":
        w = int(rng.choice((3, 3, 5)))
        stem = int(rng.integers(1, 3))
        base = {(0, c) for c in range(w)} | {(r, w // 2) for r in range(1, stem + 1)}
        cells = _transform(base, int(rng.integers(8)))
    elif kind == "U":
        w, h = int(rng.integers(3, 6)), int(rng.integers(2, 4))
        base = ({(h - 1, c) for c in range(w)} | {(r, 0) for r in range(h)}
                | {(r, w - 1) for r in range(h)})
        u = rng.random()
        # Open toward incoming pieces (a cup), away from them (a roof over a
        # sheltered hole), or sideways.
        cells = base if u < 0.42 else (_transform(base, 2) if u < 0.84
                                       else _transform(base, int(rng.choice((1, 3, 5, 7)))))
    elif kind == "plus":
        cells = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}
    elif kind == "S":
        cells = _transform({(0, 1), (0, 2), (1, 0), (1, 1)}, int(rng.integers(8)))
    elif kind == "comb":
        w, t = int(rng.integers(3, 7)), int(rng.integers(1, 3))
        base = {(0, c) for c in range(w)} | {(r, c) for c in range(0, w, 2) for r in range(1, t + 1)}
        cells = _transform(base, int(rng.choice((0, 2, 0, 2, 1, 3))))
    elif kind == "stair":
        n = int(rng.integers(2, 4))
        base = set()
        for i in range(n):
            base |= {(i, i), (i, i + 1)}
        cells = _transform(base, int(rng.integers(8)))
    else:
        cells = _grow(rng, int(rng.integers(3, 9)), max_h, max_w)
        if cells is None:
            return None
    cells = _norm(cells)
    r0, r1, c0, c1 = _bbox(cells)
    if r1 - r0 + 1 > max_h or c1 - c0 + 1 > max_w:
        return None
    return cells


def _pick(rng, weights):
    keys = list(weights)
    probs = np.array([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=probs / probs.sum()))]


def _crop(pieces):
    r0, _, c0, _ = _bbox(_union(pieces))
    return [{(r - r0, c - c0) for r, c in piece} for piece in pieces]


def _touches_diag(cells, occupied):
    return any((r + dr, c + dc) in occupied for r, c in cells for dr, dc in _DIAG)


def _prefer(rng, candidates, occupied, layout, diag_bias):
    def pick(pool):
        return pool[int(rng.integers(len(pool)))]

    if not occupied:
        return pick(candidates)
    if layout == "tower":
        lanes = {c for _, c in occupied}
        stack = [x for x in candidates if any(c in lanes for _, c in x)]
        if stack and rng.random() < 0.88:
            diagonal = [x for x in stack if _touches_diag(x, occupied)]
            if diagonal and rng.random() < diag_bias * 0.7:
                return pick(diagonal)
            return pick(stack)
        return pick(candidates)
    if layout == "scatter":
        u = rng.random()
        if u < 0.4:
            apart = [x for x in candidates if not _touches_diag(x, occupied)]
            if apart:
                return pick(apart)
        elif u < 0.4 + 0.5 * diag_bias:
            diagonal = [x for x in candidates if _touches_diag(x, occupied)]
            if diagonal:
                return pick(diagonal)
        return pick(candidates)
    if rng.random() < diag_bias:
        diagonal = [x for x in candidates if _touches_diag(x, occupied)]
        if diagonal:
            return pick(diagonal)
    near = [x for x in candidates
            if any((r + dr, c + dc) in occupied for r, c in x for dr in (-2, -1, 0, 1, 2)
                   for dc in (-2, -1, 0, 1, 2))]
    return pick(near if near else candidates)


def _place(rng, shapes, height, width, layout, diag_bias, initial=()):
    """Place shapes in a field without edge contact between pieces."""
    occupied, halo, pieces = set(), set(), []
    for piece in initial:
        pieces.append(set(piece))
        occupied |= set(piece)
        for r, c in piece:
            for dr, dc in _N4:
                halo.add((r + dr, c + dc))
    for shape in shapes:
        _, sh, _, sw = _bbox(shape)
        sh, sw = sh + 1, sw + 1
        if sh > height or sw > width:
            return None
        candidates = []
        for dr in range(height - sh + 1):
            for dc in range(width - sw + 1):
                cells = frozenset((r + dr, c + dc) for r, c in shape)
                if any(p in occupied or p in halo for p in cells):
                    continue
                candidates.append(cells)
        if not candidates:
            return None
        chosen = _prefer(rng, candidates, occupied, layout, diag_bias)
        pieces.append(set(chosen))
        occupied |= chosen
        for r, c in chosen:
            for dr, dc in _N4:
                halo.add((r + dr, c + dc))
    return _crop(pieces)


def _vocab_shape(rng, plan, max_h, max_w):
    for _ in range(10):
        shape = _shape(rng, _pick(rng, plan["vocab"]), max_h, max_w)
        if shape is not None:
            return shape
    return frozenset({(0, 0)})


def _width_ok(pieces, width_range):
    _, _, c0, c1 = _bbox(_union(pieces))
    return width_range[0] <= c1 - c0 + 1 <= width_range[1]


def _shape_ok(pieces):
    cells = _union(pieces)
    r0, r1, c0, c1 = _bbox(cells)
    return r1 > r0 and c1 > c0 and not _is_filled_rectangle(cells) and len(cells) <= MAX_INK


def _free_debris(rng, plan, k, width_range, layout, dims=(5, 6)):
    w_lo, w_hi = width_range
    diag_bias = plan["diag_bias"]
    for _ in range(12):
        if layout == "lattice" and k >= 5:
            pieces = _lattice(rng, plan, k, width_range)
        else:
            shapes = [_vocab_shape(rng, plan, dims[0], max(2, min(w_hi, dims[1])))
                      for _ in range(k)]
            if k == 1 and (len(shapes[0]) < 3 or _is_filled_rectangle(shapes[0])):
                continue
            if layout == "straggler" and k >= 2:
                shapes, straggler = shapes[:-1], shapes[-1]
            else:
                straggler = None
            if rng.random() < 0.5:
                shapes.sort(key=len, reverse=True)
            total = sum(len(s) for s in shapes)
            widest = max(_bbox(s)[3] + 1 for s in shapes)
            tallest = max(_bbox(s)[1] + 1 for s in shapes)
            if layout == "tower":
                width = max(widest, min(w_hi, int(rng.integers(2, 6))))
                height = min(18, max(tallest, sum(_bbox(s)[1] + 1 for s in shapes)
                                     + len(shapes) // 2 + int(rng.integers(0, 3))))
            else:
                width = max(widest, int(rng.integers(w_lo, w_hi + 1)))
                need = 2.4 * total + 2 * len(shapes)
                height = int(np.ceil(need / width)) + int(rng.integers(0, 3))
                if layout == "scatter":
                    height += int(rng.integers(1, 4))
                height = int(min(max(height, tallest, 2), 14))
            if width > w_hi + 2:
                continue
            pieces = _place(rng, shapes, height, width, layout, diag_bias)
            if pieces is not None and straggler is not None:
                pieces = _add_straggler(rng, pieces, straggler)
        if pieces is None or len(pieces) != k:
            continue
        if not _width_ok(pieces, width_range) or not _shape_ok(pieces):
            continue
        return pieces
    return None


def _lattice(rng, plan, k, width_range):
    """A diagonal checkerboard patch of isolates (B's X, enlarged) plus extras."""
    w_lo, w_hi = width_range
    lw = int(rng.integers(3, max(4, min(8, w_hi + 1))))
    lh = int(rng.integers(2, 5))
    parity = int(rng.integers(2))
    cells = [(r, c) for r in range(lh) for c in range(lw) if (r + c + parity) % 2 == 0]
    drop = int(rng.integers(0, 3))
    for _ in range(drop):
        if len(cells) > 4:
            cells.pop(int(rng.integers(len(cells))))
    extra = k - len(cells)
    if extra < 0 or extra > 4:
        return None
    width = max(lw + int(rng.integers(0, 4)), w_lo)
    height = lh + int(rng.integers(2, 6))
    off_r = int(rng.integers(0, height - lh + 1))
    off_c = int(rng.integers(0, width - lw + 1))
    initial = [{(r + off_r, c + off_c)} for r, c in cells]
    shapes = [_vocab_shape(rng, plan, 4, 5) for _ in range(extra)]
    return _place(rng, shapes, height, width, "clump", plan["diag_bias"], initial=initial)


def _add_straggler(rng, pieces, shape):
    """Add one piece far behind the cluster, sharing at least one lane."""
    cells = _union(pieces)
    lanes = sorted({c for _, c in cells})
    _, sh, _, sw = _bbox(shape)
    sh, sw = sh + 1, sw + 1
    anchor = lanes[int(rng.integers(len(lanes)))]
    dc = anchor - int(rng.integers(0, sw))
    separation = int(rng.integers(4, 12))
    straggler = {(r - sh - separation, c + dc) for r, c in shape}
    return _crop(pieces + [straggler])


def _b_debris(rng, plan, profiles):
    """Five cells with one of parent B's component profiles."""
    weights = {(5,): 0.2, (4, 1): 0.32, (3, 1, 1): 0.32, (1, 1, 1, 1, 1): 0.16}
    allowed = {p: weights[p] for p in profiles}
    profile = _pick(rng, allowed)
    if rng.random() < plan["b_spread"]:
        for _ in range(12):
            shapes = []
            for size in profile:
                shape = frozenset({(0, 0)}) if size == 1 else _grow(rng, size, 4, 4)
                if shape is None or (size == 5 and _is_filled_rectangle(shape)):
                    shapes = None
                    break
                shapes.append(shape)
            if shapes is None:
                continue
            height, width = int(rng.integers(2, 6)), int(rng.integers(2, 6))
            if height * width < 11:
                continue
            layout = "clump" if rng.random() < 0.7 else "scatter"
            pieces = _place(rng, shapes, height, width, layout, plan["diag_bias"])
            if pieces is None or not _shape_ok(pieces):
                continue
            if tuple(sorted((len(p) for p in pieces), reverse=True)) != profile:
                continue
            return pieces
    pool = _SOURCE_BY_PROFILE[profile]
    return [set(p) for p in pool[int(rng.integers(len(pool)))]]


def _anchor_debris(rng):
    """A single parent-A-like rigid object (3-12 cells, never a rectangle)."""
    for _ in range(20):
        shape = _shape(rng, ANCHOR_KINDS[int(rng.integers(len(ANCHOR_KINDS)))], 5, 6)
        if shape is None or len(shape) < 3 or _is_filled_rectangle(shape):
            continue
        r0, r1, c0, c1 = _bbox(shape)
        if r1 > r0 and c1 > c0:
            return [set(shape)]
    pool = _SOURCE_BY_PROFILE[(5,)]
    return [set(pool[int(rng.integers(len(pool)))][0])]


def _build_source_masks():
    """Every five-cell 3x3 mask with one of parent B's four component profiles."""
    masks = []
    for bits in range(512):
        if bin(bits).count("1") != 5:
            continue
        cells = {(i // 3, i % 3) for i in range(9) if bits >> i & 1}
        pieces = _components(cells, _N4)
        profile = tuple(sorted((len(p) for p in pieces), reverse=True))
        if profile not in B_PROFILES:
            continue
        r0, r1, c0, c1 = _bbox(cells)
        if r1 == r0 or c1 == c0:
            continue
        masks.append([{(r - r0, c - c0) for r, c in p} for p in pieces])
    return masks


_SOURCE_MASKS = _build_source_masks()
_SOURCE_BY_PROFILE = {}
for _mask in _SOURCE_MASKS:
    _SOURCE_BY_PROFILE.setdefault(tuple(sorted((len(p) for p in _mask), reverse=True)),
                                  []).append(_mask)


# ---------------------------------------------------------------------------
# Pair construction (canonical frame, then oriented)
# ---------------------------------------------------------------------------

def _sample_gap(rng, contact_p):
    if contact_p and rng.random() < contact_p:
        return 0
    band = int(rng.choice(4, p=(0.30, 0.35, 0.25, 0.10)))
    low, high = ((1, 1), (2, 3), (4, 6), (7, 10))[band]
    return int(rng.integers(low, high + 1))


def _sample_thickness(rng, minimal):
    if minimal:
        return int(rng.choice((1, 2, 3), p=(0.35, 0.4, 0.25)))
    return int(rng.choice((1, 2, 3, 4, 5), p=(0.27, 0.30, 0.20, 0.13, 0.10)))


def _sample_margin_split(rng, span):
    available = 30 - span
    band = int(rng.choice(4, p=(0.27, 0.38, 0.25, 0.10)))
    if band == 0:
        extra = int(rng.integers(0, min(available, 2) + 1))
    elif band == 1:
        extra = int(rng.integers(min(available, 2), min(available, 7) + 1))
    elif band == 2:
        extra = int(rng.integers(min(available, 6), min(available, 14) + 1))
    else:
        extra = int(rng.integers(0, available + 1))
    before = int(rng.integers(0, extra + 1))
    return before, extra - before


def _mirror_lanes(rng, width, style, overhang, minimal):
    """Return (lo, hi, thickness, wall) in field lane coordinates."""
    thickness = _sample_thickness(rng, minimal)
    if overhang:
        options = np.arange(1, min(width - 1, 6) + 1)
        uncovered = int(rng.choice(options, p=(1.0 / options) / (1.0 / options).sum()))
        u_left = int(rng.integers(0, uncovered + 1))
        u_right = uncovered - u_left
        lo, hi = u_left, width - 1 - u_right
        if u_left == 0:
            lo -= int(rng.integers(0, 4))
        if u_right == 0:
            hi += int(rng.integers(0, 4))
        if hi == lo and thickness == 1:
            thickness = int(rng.integers(2, MAX_THICKNESS + 1))
        return lo, hi, thickness, False
    if style == "square" and width <= 4:
        side = int(rng.integers(max(2, width), 5))
        left = int(rng.integers(0, side - width + 1))
        return -left, -left + side - 1, side, False
    if style == "wall":
        lo = -int(rng.integers(0, 9))
        hi = width - 1 + int(rng.integers(0, 9))
        return lo, hi, min(thickness, 4), True
    ext = (0, 1, 2, 3, 4)
    p = (0.4, 0.27, 0.15, 0.1, 0.08)
    lo = -int(rng.choice(ext, p=p))
    hi = width - 1 + int(rng.choice(ext, p=p))
    return lo, hi, thickness, False


def _construct_pair(rng, pieces, plan, direction, colours, style, overhang):
    """Build one pair in the canonical downward frame, then orient it."""
    cells = _union(pieces)
    rows = max(r for r, _ in cells) + 1
    width = max(c for _, c in cells) + 1
    lo, hi, thickness, wall = _mirror_lanes(rng, width, style, overhang, plan["minimal"])
    gap = _sample_gap(rng, plan["contact_p"])
    face = rows + gap
    settled = _topo_settle(pieces, face, (lo, hi))
    if settled is None:
        return None
    pile = _union(settled)
    if pile == cells:
        return None
    depth = face - min(r for r, _ in pile)
    image = _reflect(pile, face)
    lane_lo, lane_hi = min(0, lo), max(width - 1, hi)
    span_h, span_w = face + max(thickness, depth), lane_hi - lane_lo + 1
    if span_h > 30 or span_w > 30:
        return None
    top, bottom = _sample_margin_split(rng, span_h)
    left, right = (0, 0) if wall else _sample_margin_split(rng, span_w)
    mirror_count = thickness * (hi - lo + 1)
    ink_count = len(cells)
    for _ in range(80):
        height, full_width = top + span_h + bottom, left + span_w + right
        if height * full_width - mirror_count - ink_count > max(mirror_count, ink_count):
            break
        grow_rows = height < 30 and (wall or full_width >= 30 or rng.random() < 0.5)
        if grow_rows:
            if rng.random() < 0.5:
                top += 1
            else:
                bottom += 1
        elif full_width < 30 and not wall:
            if rng.random() < 0.5:
                left += 1
            else:
                right += 1
        else:
            return None
    else:
        return None
    height, full_width = top + span_h + bottom, left + span_w + right
    background, ink, mirror = colours
    inp = np.full((height, full_width), background, dtype=np.int64)
    out = np.full((height, full_width), background, dtype=np.int64)
    shift = left - lane_lo
    for r, c in cells:
        inp[top + r, shift + c] = ink
    for r in range(face, face + thickness):
        for c in range(lo, hi + 1):
            inp[top + r, shift + c] = mirror
    for r, c in image:
        out[top + r, shift + c] = mirror
    for r, c in pile:
        out[top + r, shift + c] = ink
    inp = np.ascontiguousarray(_to_direction(inp, direction))
    out = np.ascontiguousarray(_to_direction(out, direction))
    return {"input": inp, "output": out}


def _input_hash(grid):
    text = json.dumps(np.asarray(grid).astype(int).tolist(), separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _input_key(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.astype(np.int64).tobytes()


# ---------------------------------------------------------------------------
# Puzzle-level planning
# ---------------------------------------------------------------------------

class _Retry(Exception):
    pass


def _choose_directions(rng, num_train, num_test, minimal):
    if minimal:
        mode = "simple"
    else:
        mode = str(rng.choice(("simple", "familiar", "heldout", "mixed"),
                              p=(0.12, 0.28, 0.30, 0.30)))
    horizontal, vertical = ("right", "left"), ("down", "up")
    if mode == "simple":
        direction = str(rng.choice(DIRECTIONS))
        return mode, [direction] * num_train, [direction] * num_test
    if mode == "heldout":
        held = str(rng.choice(DIRECTIONS))
        allowed = tuple(d for d in DIRECTIONS if d != held)
        train = [str(rng.choice([d for d in horizontal if d != held])),
                 str(rng.choice([d for d in vertical if d != held]))]
        train += [str(rng.choice(allowed)) for _ in range(num_train - 2)]
        test = [held] * num_test
        if num_test > 1 and rng.random() < 0.35:
            test[-1] = str(rng.choice(train))
    else:
        train = [str(rng.choice(horizontal)), str(rng.choice(vertical))]
        train += [str(rng.choice(DIRECTIONS)) for _ in range(num_train - 2)]
        pool = sorted(set(train)) if mode == "familiar" else list(DIRECTIONS)
        test = [str(rng.choice(pool)) for _ in range(num_test)]
    rng.shuffle(train)
    rng.shuffle(test)
    return mode, train, test


def _choose_palettes(rng, total, minimal):
    if minimal:
        mode = "fixed"
    else:
        mode = str(rng.choice(("fixed", "fixed_bg", "swap", "free"), p=(0.18, 0.22, 0.25, 0.35)))
    base = tuple(int(v) for v in rng.choice(10, size=3, replace=False))
    palettes = []
    for index in range(total):
        if mode == "fixed":
            palettes.append(base)
        elif mode == "swap":
            palettes.append(base if index % 2 == 0 else (base[0], base[2], base[1]))
        elif mode == "fixed_bg":
            others = [v for v in range(10) if v != base[0]]
            pair = rng.choice(len(others), size=2, replace=False)
            palettes.append((base[0], others[int(pair[0])], others[int(pair[1])]))
        else:
            palettes.append(tuple(int(v) for v in rng.choice(10, size=3, replace=False)))
    return mode, palettes


_K_WEIGHTS = {1: 0.2, 2: 1.0, 3: 1.0, 4: 0.9, 5: 0.75, 6: 0.55, 7: 0.4, 8: 0.3, 9: 0.2, 10: 0.15}


def _plan(rng, num_train, num_test, minimal):
    direction_mode, train_dirs, test_dirs = _choose_directions(rng, num_train, num_test, minimal)
    palette_mode, palettes = _choose_palettes(rng, num_train + num_test, minimal)
    plan = {"minimal": minimal, "direction_mode": direction_mode, "train_dirs": train_dirs,
            "test_dirs": test_dirs, "palette_mode": palette_mode, "palettes": palettes}
    if minimal:
        regime = "b_profile" if rng.random() < 0.3 else "free"
        plan.update(regime=regime, mirror_style=str(rng.choice(("slab", "square"))),
                    overhang="none", overhang_rate=0.0, shift="none", contact_p=0.0,
                    layouts={"clump": 1.0}, diag_bias=float(rng.uniform(0.6, 0.9)),
                    b_spread=0.3, anchor=True,
                    k_train=(1, 4), k_test=(2, 4), w_train=(2, 6), w_test=(2, 6))
    else:
        regime = str(rng.choice(("free", "b_profile", "mixed"), p=(0.5, 0.2, 0.3)))
        layouts = {name: float(rng.uniform(0.3, 1.7)) * base for name, base in
                   zip(LAYOUTS, (1.0, 0.7, 0.55, 0.35, 0.35))}
        shift = str(rng.choice(("none", "more_pieces", "wider", "deeper"),
                               p=(0.38, 0.24, 0.18, 0.20)))
        plan.update(regime=regime,
                    mirror_style=str(rng.choice(("square", "slab", "wall", "mixed"),
                                                p=(0.12, 0.30, 0.12, 0.46))),
                    overhang=str(rng.choice(("none", "some", "test"), p=(0.35, 0.4, 0.25))),
                    overhang_rate=float(rng.uniform(0.25, 0.5)),
                    shift=shift, contact_p=0.12 if rng.random() < 0.3 else 0.0,
                    layouts=layouts, diag_bias=float(rng.uniform(0.35, 0.95)),
                    b_spread=float(rng.uniform(0.3, 0.8)), anchor=rng.random() < 0.55,
                    k_train=(1, 7), k_test=(2, 9), w_train=(2, 10), w_test=(3, 12))
        if shift == "more_pieces":
            cap = int(rng.integers(2, 5))
            plan.update(cap=cap, k_train=(1, cap), k_test=(cap + 1, min(cap + 5, 10)))
        elif shift == "wider":
            threshold = int(rng.integers(5, 9))
            plan.update(wt=threshold, w_train=(2, threshold - 1),
                        w_test=(threshold, min(threshold + 6, 14)))
        elif shift == "deeper":
            plan.update(lt=int(rng.integers(2, 4)), k_test=(3, 9))
    vocab = {kind: float(rng.uniform(0.3, 1.7)) for kind in SHAPE_KINDS}
    for kind in rng.choice(SHAPE_KINDS, size=int(rng.integers(0, 5)), replace=False):
        vocab[str(kind)] *= 0.1
    if minimal:
        for kind in ("comb", "stair", "U", "bar"):
            vocab[kind] *= 0.2
    plan["vocab"] = vocab
    plan["showcase"] = (not minimal) and rng.random() < 0.4
    if minimal:
        plan["scales"] = ({"small": 0.6, "medium": 0.4}, {"small": 0.6, "medium": 0.4})
    else:
        plan["scales"] = ({"small": 0.4, "medium": 0.38, "large": 0.22},
                          {"small": 0.3, "medium": 0.42, "large": 0.28})
    plan["b_rate"] = {"free": (0.0, 0.0), "b_profile": (0.8, 0.6),
                      "mixed": (0.35, 0.3)}[plan["regime"]]
    return plan


def _accept(plan, role, analysis, used_keys, flat):
    if analysis["key"] in used_keys:
        return False
    shift = plan["shift"]
    test = role == "test"
    if test:
        if not TEST_REQUIRED <= analysis["refuted"]:
            return False
        if plan["overhang"] == "test" and not analysis["overhang"]:
            return False
    elif analysis["depth"] == 1 and flat:
        return False
    if plan["overhang"] == "none" and analysis["overhang"]:
        return False
    if shift == "more_pieces" and (analysis["pieces"] <= plan["cap"]) == test:
        return False
    if shift == "wider" and (analysis["width"] < plan["wt"]) == test:
        return False
    if shift == "deeper" and (analysis["level"] < plan["lt"]) == test:
        return False
    return True


def _candidate(rng, plan, role, direction, colours, needs):
    test = role == "test"
    if role == "anchor":
        if rng.random() < 0.4 and plan["k_train"][1] >= 2:
            pieces = _free_debris(rng, plan, 2, (2, 5), "clump", (3, 3))
        else:
            pieces = _anchor_debris(rng)
    else:
        k_lo, k_hi = plan["k_test"] if test else plan["k_train"]
        widths = plan["w_test"] if test else plan["w_train"]
        b_rate = plan["b_rate"][1 if test else 0]
        profiles = [p for p in B_PROFILES if k_lo <= len(p) <= k_hi
                    and (not test or p in ((4, 1), (3, 1, 1)))
                    and widths[0] <= 5]
        if profiles and rng.random() < b_rate:
            pieces = _b_debris(rng, plan, profiles)
        else:
            # Pair scale keeps a spectrum from small scenes to rich debris fields.
            scale = _pick(rng, plan["scales"][1 if test else 0])
            cap, dims = {"small": (4, (3, 3)), "medium": (6, (4, 4)), "large": (10, (5, 6))}[scale]
            if k_lo > cap:
                cap, dims = 10, (5, 6)
            weights = {k: _K_WEIGHTS[k] for k in range(k_lo, min(k_hi, cap) + 1)}
            if needs & {"no_stack", "touching_only"} or (test and plan["shift"] == "deeper"):
                weights.pop(1, None)
            if not weights:
                return None
            k = _pick(rng, weights)
            layouts = dict(plan["layouts"])
            if test and plan["shift"] == "deeper":
                layouts["tower"] = layouts.get("tower", 0.0) + 2.0
            if k < 5:
                layouts.pop("lattice", None)
            if k < 2:
                layouts.pop("straggler", None)
            if not layouts:
                layouts = {"clump": 1.0}
            pieces = _free_debris(rng, plan, k, widths, _pick(rng, layouts), dims)
    if pieces is None:
        return None
    if plan["overhang"] == "none":
        overhang = False
    elif test and plan["overhang"] == "test":
        overhang = True
    elif needs & {"lane_clip", "facing_only"}:
        overhang = rng.random() < 0.85
    else:
        rate = plan["overhang_rate"] if plan["overhang"] == "some" else 0.15
        overhang = rng.random() < rate
    width = max(c for _, c in _union(pieces)) + 1
    overhang = overhang and width >= 2
    style = plan["mirror_style"]
    if style == "mixed":
        style = str(rng.choice(("square", "slab", "wall"), p=(0.3, 0.5, 0.2)))
    return _construct_pair(rng, pieces, plan, direction, colours, style, overhang)


def _coverage(analysis):
    covered = set(analysis["refuted"])
    if "copy" in analysis["refuted"]:
        covered.add(("copy", analysis["axis"]))
    return covered


def _draw(rng, plan, role, direction, colours, used_inputs, used_keys, flat, needs, attempts,
          prefer=None):
    best, best_gain = None, -1
    for _ in range(attempts):
        pair = _candidate(rng, plan, role, direction, colours, needs)
        if pair is None:
            continue
        key = _input_key(pair["input"])
        if key in used_inputs or _input_hash(pair["input"]) in _PARENT_INPUT_HASHES:
            continue
        analysis = _analyze_pair(pair["input"], pair["output"])
        if not _accept(plan, role, analysis, used_keys, flat):
            continue
        if prefer is not None:
            # Keep the first acceptable pair, but look for a showcase pair.
            if best is None:
                best = (pair, analysis)
            if prefer(analysis):
                return pair, analysis
            continue
        gain = len(_coverage(analysis) & needs)
        if gain > best_gain:
            best, best_gain = (pair, analysis), gain
        if not needs or needs <= _coverage(analysis):
            break
    return best


def _build_puzzle(rng, num_train, num_test, minimal):
    plan = _plan(rng, num_train, num_test, minimal)
    used_inputs, used_keys = set(), set()
    tests = []
    # Showcase puzzles look for a test where a piece overtakes a nearer one.
    prefer = (lambda a: a["overtaking"]) if plan["showcase"] else None
    for index, direction in enumerate(plan["test_dirs"]):
        built = _draw(rng, plan, "test", direction, plan["palettes"][num_train + index],
                      used_inputs, used_keys, False, frozenset(), 60 if prefer is None else 30,
                      prefer)
        if built is None:
            raise _Retry()
        used_inputs.add(_input_key(built[0]["input"]))
        used_keys.add(built[1]["key"])
        tests.append(built)

    required = set(TRAIN_REQUIRED)
    for _, analysis in tests:
        required |= analysis["refuted"]
        required.add(("copy", analysis["axis"]))
    remaining = set(required)
    trains, flat, anchored = [], 0, False
    want_anchor = num_train >= 4 or (plan["anchor"] and num_train == 3)
    for index, direction in enumerate(plan["train_dirs"]):
        colours = plan["palettes"][index]
        built = None
        if want_anchor and not anchored:
            # The first slot is an easy anchor: one A-like object or a small
            # two-piece clump.
            built = _draw(rng, plan, "anchor", direction, colours, used_inputs, used_keys,
                          flat, frozenset(), 15)
            anchored = True
        if built is None:
            attempts = 45 if remaining else 15
            built = _draw(rng, plan, "train", direction, colours, used_inputs, used_keys,
                          flat, frozenset(remaining), attempts)
        if built is None:
            raise _Retry()
        pair, analysis = built
        used_inputs.add(_input_key(pair["input"]))
        used_keys.add(analysis["key"])
        flat += analysis["depth"] == 1
        remaining -= _coverage(analysis)
        trains.append(built)
    if remaining:
        raise _Retry()
    order = rng.permutation(len(trains))
    trains = [trains[int(i)] for i in order]
    summary = {k: plan[k] for k in ("minimal", "direction_mode", "palette_mode", "regime",
                                    "mirror_style", "overhang", "shift")}
    return trains, tests, summary


def _check_counts(num_train, num_test):
    num_train = 4 if num_train is None else num_train
    num_test = 1 if num_test is None else num_test
    for name, value in (("num_train", num_train), ("num_test", num_test)):
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


def _generate(seed, num_train, num_test):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    minimal = bool(rng.random() < 1 / 14)
    for attempt in range(120):
        try:
            trains, tests, plan = _build_puzzle(rng, num_train, num_test,
                                                minimal or attempt >= 100)
        except _Retry:
            continue
        puzzle = {
            "train": [{"input": p["input"], "output": p["output"]} for p, _ in trains],
            "test": [{"input": p["input"], "output": p["output"]} for p, _ in tests],
        }
        return puzzle, [a for _, a in trains], [a for _, a in tests], plan
    raise RuntimeError("could not construct a puzzle")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    puzzle, _, _, _ = _generate(seed, num_train, num_test)
    validate_generated_puzzle(puzzle)
    return puzzle


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _as_array(value):
    array = np.asarray(value)
    assert array.ndim == 2 and all(1 <= n <= 30 for n in array.shape)
    assert np.issubdtype(array.dtype, np.integer)
    assert int(array.min()) >= 0 and int(array.max()) <= 9
    return array.astype(np.int64)


def _puzzle_analyses(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    num_train, num_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _as_array(pair["input"]), _as_array(pair["output"])
            key = _input_key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            assert _input_hash(inp) not in _PARENT_INPUT_HASHES, "parent input copied"
            analyses[split].append(_analyze_pair(inp, out))
    return analyses


def validate_generated_puzzle(puzzle: dict) -> None:
    analyses = _puzzle_analyses(puzzle)
    train, test = analyses["train"], analyses["test"]
    train_refuted = set()
    for item in train:
        train_refuted |= item["refuted"]
    assert TRAIN_REQUIRED <= train_refuted, "training misses a core distinction"
    train_keys = [item["key"] for item in train]
    assert len(set(train_keys)) == len(train_keys), "repeated training ink pattern"
    assert sum(item["depth"] == 1 for item in train) <= 1, "more than one flat training pair"
    test_keys = set()
    for item in test:
        # Ignoring parent B (one rigid object, eight-connected bodies, no
        # bodies at all) or parent A's flip must give wrong test answers.
        assert TEST_REQUIRED <= item["refuted"], "test is solved by a shortcut"
        # Every distinction a test relies on is demonstrated in training.
        assert item["refuted"] <= train_refuted, "test relies on an undemonstrated distinction"
        assert any(t["axis"] == item["axis"] and "copy" in t["refuted"] for t in train)
        assert item["key"] not in train_keys, "test ink pattern repeats a training pattern"
        assert item["key"] not in test_keys, "repeated test ink pattern"
        test_keys.add(item["key"])


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        for a, b in zip(first[s], second[s])) for s in ("train", "test"))


def self_test() -> None:
    plans, records = [], []
    puzzles = []
    for seed in range(42):
        # The returned analyses are measured from the returned grids.
        puzzle, train_analyses, test_analyses, plan = _generate(seed, 4, 1)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        plans.append(plan)
        records.append((train_analyses, test_analyses))
    assert _puzzles_equal(_generate(7, 4, 1)[0], puzzles[7])
    signatures = {json.dumps([[p["input"].tolist(), p["output"].tolist()] for s in ("train", "test")
                              for p in pz[s]]) for pz in puzzles}
    assert len(signatures) == 42
    pairs = [a for tr, te in records for a in tr + te]
    tests = [a for _, te in records for a in te]
    # Measured on the returned grids.
    assert {a["direction"] for a in pairs} == set(DIRECTIONS)
    assert {a["direction"] for a in tests} == set(DIRECTIONS)
    assert all(TEST_REQUIRED <= a["refuted"] for a in tests)
    assert sum(a["overhang"] > 0 for a in pairs) >= 35
    assert sum(a["overhang"] > 0 for a in tests) >= 12
    assert sum(a["outside_pieces"] > 0 for a in pairs) >= 5
    assert sum(a["overtaking"] for a in tests) >= 10
    assert sum(a["level"] >= 2 for a in tests) >= 15
    assert sum(a["level"] >= 3 for a in pairs) >= 12
    assert sum(a["holes"] >= 2 for a in tests) >= 28
    assert sum(a["groups8"] >= 2 for a in pairs) >= 70
    assert sum(a["groups8"] >= 2 for a in tests) >= 22
    assert sum(a["band"] >= 4 for a in pairs) >= 3
    assert sum(a["isolates"] >= 5 for a in pairs) >= 3
    assert max(a["width"] for a in pairs) >= 11
    assert max(a["pieces"] for a in pairs) >= 8
    assert max(a["cells"] for a in pairs) >= 32
    assert {a["thickness"] for a in pairs} == {1, 2, 3, 4, 5}
    assert sum(a["contact"] for a in pairs) >= 4
    assert sum(a["wall"] for a in pairs) >= 20
    assert sum(a["square"] for a in pairs) >= 20
    assert sum(a["b_profile"] for a in pairs) >= 25
    assert sum(a["b_profile"] and not a["b_mask"] for a in pairs) >= 10
    assert 15 <= sum(a["pieces"] == 1 for a in pairs) <= 70
    assert sum(a["cells"] <= 12 for a in tests) >= 3
    assert sum(a["cells"] > 25 for a in tests) >= 5
    assert len({a["shape"] for a in pairs}) >= 120
    assert len({a["colours"] for a in pairs}) >= 90
    # Puzzle-level regimes and their train/test relations.
    assert {p["shift"] for p in plans} == {"none", "more_pieces", "wider", "deeper"}
    assert {p["overhang"] for p in plans} == {"none", "some", "test"}
    assert {p["palette_mode"] for p in plans} == {"fixed", "fixed_bg", "swap", "free"}
    assert len({p["direction_mode"] for p in plans}) == 4
    assert len({p["regime"] for p in plans}) == 3
    assert sum(p["minimal"] for p in plans) >= 1
    for plan, (tr, te) in zip(plans, records):
        if plan["shift"] == "more_pieces":
            assert min(a["pieces"] for a in te) > max(a["pieces"] for a in tr)
        if plan["shift"] == "wider":
            assert min(a["width"] for a in te) > max(a["width"] for a in tr)
        if plan["shift"] == "deeper":
            assert min(a["level"] for a in te) > max(a["level"] for a in tr)
        if plan["overhang"] == "test":
            assert all(a["overhang"] for a in te)
        if plan["overhang"] == "none":
            assert not any(a["overhang"] for a in tr + te)
    heldout = sum(not ({a["direction"] for a in te} <= {a["direction"] for a in tr})
                  for tr, te in records)
    assert heldout >= 8
    # Explicit legal counts, including the extremes, and default counts.
    for num_train, num_test in ((2, 1), (3, 2), (10, 4), (5, 4), (10, 1)):
        puzzle = generate_puzzle(1000 + 10 * num_train + num_test, num_train, num_test)
        assert len(puzzle["train"]) == num_train and len(puzzle["test"]) == num_test
    assert _puzzles_equal(generate_puzzle(3, None, None), generate_puzzle(3))
    for num_train, num_test in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (4.0, 1),
                                (True, 1), (4, "1")):
        try:
            generate_puzzle(0, num_train, num_test)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
