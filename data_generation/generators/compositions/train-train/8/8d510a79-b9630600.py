"""Anchored and outbound room networks.

Composition of ARC 8d510a79 (parent A: seeds grow perpendicular to a
full-length separator, one role toward it and the other away to the outer
edge) and ARC b9630600 (parent B: hollow rooms are joined by the unique
minimum spanning tree of clear, interior-overlapping orthogonal gaps, each
selected gap rendered as a walled corridor).

Rule
----
The input has a background, exactly one full-length separator line, and
one-cell-thick hollow rectangular rooms on both sides of it.  A room drawn in
the separator's colour is *anchored*; a room drawn in the single other room
colour is *outbound*.  Each class forms its own network with a root:

* anchored rooms are rooted at the separator (A's "toward" role);
* outbound rooms are rooted at the outer grid edge parallel to the separator
  on the room's own side (A's "away" role).

Candidate links (B's eligibility rule, extended to the roots):

* room-room: two rooms of the same class that are separated horizontally or
  vertically, whose open interiors overlap on the perpendicular axis, and
  whose gap over that overlap is entirely background;
* room-root: the straight band perpendicular to the separator, from the
  room's facing wall to its root, over the room's whole interior span
  (the separator / edge is an infinitely wide room), entirely background.

The weight of a link is the number of background cells it spans.  Each class
keeps the unique minimum spanning tree of its rooms plus its root.  Every
selected link becomes a corridor in the class colour whose cross-section is
the interior overlap: its two extreme lines are walls, the cells between them
are cleared (carving a door through the facing walls), and widths one or two
are solid.  Anchored corridors stop beside the separator, which is never
recoloured; outbound corridors run through the grid edge.  Everything else is
unchanged.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np

_MIN_TRAIN, _MAX_TRAIN = 2, 10
_MIN_TEST, _MAX_TEST = 1, 4

# A room: rows r0..r1, columns c0..c1 (outer frame), colour, side
# (0 = before the separator, 1 = after it) in canonical orientation, where
# the separator is a row.
_Room = namedtuple("_Room", "r0 r1 c0 c1 color side")

# A candidate link.  ``b`` is a room index or -1 for the class root.
# axis "v": corridor runs along rows; cross-section columns lo..hi; rendered
#   rows e0..e1; background gap rows g0..g1.
# axis "h": corridor runs along columns; cross-section rows lo..hi; rendered
#   columns e0..e1; background gap columns g0..g1.
_Edge = namedtuple("_Edge", "weight a b axis lo hi e0 e1 g0 g1 color")

ROOT = -1


# --------------------------------------------------------------------------
# Input analysis (shared by solve, the validator and the generator)
# --------------------------------------------------------------------------

def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    if len(order) > 1 and counts[order[0]] == counts[order[1]]:
        raise ValueError("ambiguous background")
    return int(values[order[0]])


def _find_separator(grid, bg):
    height, width = grid.shape
    found = []
    for r in range(height):
        value = int(grid[r, 0])
        if value != bg and np.all(grid[r, :] == value):
            found.append(("h", r, value))
    for c in range(width):
        value = int(grid[0, c])
        if value != bg and np.all(grid[:, c] == value):
            found.append(("v", c, value))
    if len(found) != 1:
        raise ValueError("expected exactly one full-length separator")
    return found[0]


def _find_rooms(canon, bg, k):
    """Segment non-background cells (separator row excluded) into frames."""
    height, width = canon.shape
    mask = canon != bg
    mask[k, :] = False
    seen = np.zeros_like(mask)
    rooms = []
    for sr in range(height):
        for sc in range(width):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            stack, cells = [(sr, sc)], []
            seen[sr, sc] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            colors = {int(canon[r, c]) for r, c in cells}
            if len(colors) != 1:
                raise ValueError("a room must have one colour")
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
            if r1 - r0 < 2 or c1 - c0 < 2:
                raise ValueError("rooms are at least 3x3")
            perimeter = 2 * (r1 - r0 + 1) + 2 * (c1 - c0 + 1) - 4
            if len(cells) != perimeter:
                raise ValueError("rooms are hollow one-cell frames")
            if not (np.all(canon[r0, c0:c1 + 1] != bg) and np.all(canon[r1, c0:c1 + 1] != bg)
                    and np.all(canon[r0:r1 + 1, c0] != bg) and np.all(canon[r0:r1 + 1, c1] != bg)):
                raise ValueError("rooms are rectangular frames")
            if not np.all(canon[r0 + 1:r1, c0 + 1:c1] == bg):
                raise ValueError("room interiors are empty")
            side = 0 if r1 < k else 1
            rooms.append(_Room(r0, r1, c0, c1, colors.pop(), side))
    rooms.sort()
    return rooms


def _parse(grid):
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError("grid must be 2-D")
    grid = grid.astype(int)
    bg = _background(grid)
    orientation, k, sep = _find_separator(grid, bg)
    canon = grid if orientation == "h" else grid.T
    canon = canon.copy()
    rooms = _find_rooms(canon, bg, k)
    others = {room.color for room in rooms if room.color != sep}
    if len(others) > 1:
        raise ValueError("at most one outbound colour")
    return {"bg": bg, "orientation": orientation, "k": k, "sep": sep,
            "canon": canon, "rooms": rooms}


# --------------------------------------------------------------------------
# Candidate graph, minimum spanning tree and rendering
# --------------------------------------------------------------------------

def _clear(canon, bg, r0, r1, c0, c1):
    if r0 > r1 or c0 > c1:
        return True
    return bool(np.all(canon[r0:r1 + 1, c0:c1 + 1] == bg))


def _root_edges(canon, bg, k, index, room, role, side_exits=False):
    """Candidate links from a room to the root of its role."""
    height, width = canon.shape
    lo, hi = room.c0 + 1, room.c1 - 1
    if role == "anchor":
        if room.side == 0:
            g0, g1, e0, e1 = room.r1 + 1, k - 1, room.r1, k - 1
        else:
            g0, g1, e0, e1 = k + 1, room.r0 - 1, k + 1, room.r0
    else:
        if room.side == 0:
            g0, g1, e0, e1 = 0, room.r0 - 1, 0, room.r0
        else:
            g0, g1, e0, e1 = room.r1 + 1, height - 1, room.r1, height - 1
    edges = []
    if _clear(canon, bg, g0, g1, lo, hi):
        edges.append(_Edge(max(0, g1 - g0 + 1), index, ROOT, "v", lo, hi, e0, e1, g0, g1, room.color))
    if side_exits and role == "outbound":
        # Shortcut variant only: outbound rooms may also leave sideways.
        rlo, rhi = room.r0 + 1, room.r1 - 1
        if _clear(canon, bg, rlo, rhi, 0, room.c0 - 1):
            edges.append(_Edge(room.c0, index, ROOT, "h", rlo, rhi, 0, room.c0, 0, room.c0 - 1, room.color))
        if _clear(canon, bg, rlo, rhi, room.c1 + 1, width - 1):
            edges.append(_Edge(width - 1 - room.c1, index, ROOT, "h", rlo, rhi, room.c1, width - 1,
                               room.c1 + 1, width - 1, room.color))
    return edges


def _pair_edge(canon, bg, i, a, j, b):
    """B's candidate link between two same-class rooms, or None."""
    if a.c1 < b.c0 - 1 or b.c1 < a.c0 - 1:
        left, right = (a, b) if a.c1 < b.c0 else (b, a)
        lo, hi = max(left.r0, right.r0) + 1, min(left.r1, right.r1) - 1
        g0, g1 = left.c1 + 1, right.c0 - 1
        if lo <= hi and _clear(canon, bg, lo, hi, g0, g1):
            return _Edge(g1 - g0 + 1, i, j, "h", lo, hi, left.c1, right.c0, g0, g1, a.color)
    if a.r1 < b.r0 - 1 or b.r1 < a.r0 - 1:
        top, bottom = (a, b) if a.r1 < b.r0 else (b, a)
        lo, hi = max(top.c0, bottom.c0) + 1, min(top.c1, bottom.c1) - 1
        g0, g1 = top.r1 + 1, bottom.r0 - 1
        if lo <= hi and _clear(canon, bg, g0, g1, lo, hi):
            return _Edge(g1 - g0 + 1, i, j, "v", lo, hi, top.r1, bottom.r0, g0, g1, a.color)
    return None


def _candidates(info, roles=None, include_roots=True, side_exits=False):
    """All candidate links.  ``roles`` maps colour -> 'anchor'/'outbound'."""
    canon, bg, k, rooms = info["canon"], info["bg"], info["k"], info["rooms"]
    if roles is None:
        roles = {room.color: ("anchor" if room.color == info["sep"] else "outbound") for room in rooms}
    edges = []
    for i, a in enumerate(rooms):
        for j in range(i + 1, len(rooms)):
            b = rooms[j]
            if a.color != b.color:
                continue
            edge = _pair_edge(canon, bg, i, a, j, b)
            if edge is not None:
                edges.append(edge)
        if include_roots:
            edges.extend(_root_edges(canon, bg, k, i, a, roles[a.color], side_exits))
    edges.sort(key=lambda e: (e.weight, e.a, e.b, e.axis, e.e0))
    return edges


def _vertex(edge_end, color):
    return ("root", color) if edge_end == ROOT else ("room", edge_end)


def _kruskal(edges, priority=None):
    parent = {}

    def find(v):
        parent.setdefault(v, v)
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    order = edges if priority is None else sorted(edges, key=priority)
    chosen = []
    for edge in order:
        u, v = find(_vertex(edge.a, edge.color)), find(_vertex(edge.b, edge.color))
        if u != v:
            parent[u] = v
            chosen.append(edge)
    return chosen, find


def _spanning(info, edges):
    """Kruskal per class; report connectivity and uniqueness."""
    chosen, find = _kruskal(edges)
    rooms = info["rooms"]
    connected = all(find(("room", i)) == find(("root", room.color)) for i, room in enumerate(rooms))
    adjacency = {}
    for edge in chosen:
        u, v = _vertex(edge.a, edge.color), _vertex(edge.b, edge.color)
        adjacency.setdefault(u, []).append((v, edge.weight))
        adjacency.setdefault(v, []).append((u, edge.weight))
    chosen_set = set(chosen)
    unique = True
    for edge in edges:
        if edge in chosen_set:
            continue
        source, target = _vertex(edge.a, edge.color), _vertex(edge.b, edge.color)
        stack, best = [(source, None, -1)], None
        while stack:
            vertex, previous, maximum = stack.pop()
            if vertex == target:
                best = maximum
                break
            for neighbour, weight in adjacency.get(vertex, ()):
                if neighbour != previous:
                    stack.append((neighbour, vertex, max(maximum, weight)))
        if best is None or best >= edge.weight:
            unique = False
    return chosen, connected, unique


def _render(canon, bg, chosen):
    out = canon.copy()
    for e in chosen:
        if e.axis == "v":
            out[e.e0:e.e1 + 1, e.lo:e.hi + 1] = bg
            out[e.e0:e.e1 + 1, e.lo] = e.color
            out[e.e0:e.e1 + 1, e.hi] = e.color
        else:
            out[e.lo:e.hi + 1, e.e0:e.e1 + 1] = bg
            out[e.lo, e.e0:e.e1 + 1] = e.color
            out[e.hi, e.e0:e.e1 + 1] = e.color
    return out


def _uncanon(info, canon_grid):
    return canon_grid if info["orientation"] == "h" else canon_grid.T


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the output from the input alone."""
    info = _parse(input_grid)
    edges = _candidates(info)
    chosen, _, _ = _spanning(info, edges)
    return np.array(_uncanon(info, _render(info["canon"], info["bg"], chosen)), dtype=int)


# --------------------------------------------------------------------------
# Measurements used by the generator, validator and self-test
# --------------------------------------------------------------------------

def _footprint(e):
    if e.g0 > e.g1:
        return set()
    if e.axis == "v":
        return {(r, c) for r in range(e.g0, e.g1 + 1) for c in range(e.lo, e.hi + 1)}
    return {(r, c) for r in range(e.lo, e.hi + 1) for c in range(e.g0, e.g1 + 1)}


def _halo(e, shape):
    height, width = shape
    cells = set()
    if e.g0 > e.g1:
        return cells
    if e.axis == "v":
        for r in range(e.g0, e.g1 + 1):
            for c in (e.lo - 1, e.hi + 1):
                if 0 <= c < width:
                    cells.add((r, c))
    else:
        for c in range(e.g0, e.g1 + 1):
            for r in (e.lo - 1, e.hi + 1):
                if 0 <= r < height:
                    cells.add((r, c))
    return cells


def _clean_rendering(info, edges, chosen):
    """Selected corridors are isolated: no crossings, no touching strangers."""
    canon, bg = info["canon"], info["bg"]
    prints = {e: _footprint(e) for e in edges}
    for e in chosen:
        for f in edges:
            if f != e and prints[e] & prints[f]:
                return False
        others = set().union(*(prints[f] for f in chosen if f != e)) if len(chosen) > 1 else set()
        for r, c in _halo(e, canon.shape):
            if canon[r, c] != bg or (r, c) in others:
                return False
    return True


def _band_blockers(info, index, role):
    """Colours of rooms that obstruct a room's straight band to its root."""
    canon, bg, k = info["canon"], info["bg"], info["k"]
    room = info["rooms"][index]
    height = canon.shape[0]
    if role == "anchor":
        rows = range(room.r1 + 1, k) if room.side == 0 else range(k + 1, room.r0)
    else:
        rows = range(0, room.r0) if room.side == 0 else range(room.r1 + 1, height)
    found = set()
    for r in rows:
        for c in range(room.c0 + 1, room.c1):
            if canon[r, c] != bg:
                found.add(int(canon[r, c]))
    return found


def _variant_output(info, variant):
    rooms = info["rooms"]
    sep = info["sep"]
    if variant == "rays_only":       # A alone: every clear root band, no graph
        chosen = [e for e in _candidates(info) if e.b == ROOT]
    elif variant == "no_roots":      # B alone: per-colour MST between rooms only
        chosen, _ = _kruskal(_candidates(info, include_roots=False))
    elif variant == "direct_first":  # use every clear root link, then connect the rest
        chosen, _ = _kruskal(_candidates(info), priority=lambda e: (e.b != ROOT, e.weight, e.a, e.b))
    elif variant == "nearest":       # every room keeps its own shortest link
        edges = _candidates(info)
        chosen = []
        for i in range(len(rooms)):
            mine = [e for e in edges if e.a == i or e.b == i]
            if mine and mine[0] not in chosen:
                chosen.append(mine[0])
    elif variant in ("all_to_separator", "swap_roles"):
        roles = {}
        for room in rooms:
            if variant == "all_to_separator":
                roles[room.color] = "anchor"
            else:
                roles[room.color] = "outbound" if room.color == sep else "anchor"
        chosen, _ = _kruskal(_candidates(info, roles=roles))
    elif variant == "side_exits":    # outbound rooms may leave through any edge
        chosen, _ = _kruskal(_candidates(info, side_exits=True))
    else:
        raise ValueError(variant)
    return _render(info["canon"], info["bg"], chosen)


_VARIANTS = ("rays_only", "no_roots", "direct_first", "nearest", "all_to_separator",
             "swap_roles", "side_exits")


def _features(info, edges, chosen, truth, variants=True):
    rooms = info["rooms"]
    sep = info["sep"]
    role = {room.color: ("anchor" if room.color == sep else "outbound") for room in rooms}
    root_candidates = {e.a for e in edges if e.b == ROOT}
    root_selected = [e for e in chosen if e.b == ROOT]
    direct = {e.a for e in root_selected}
    depth = {}
    adjacency = {}
    for e in chosen:
        u, v = _vertex(e.a, e.color), _vertex(e.b, e.color)
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
    for color in {room.color for room in rooms}:
        frontier, seen = [("root", color)], {("root", color): 0}
        while frontier:
            v = frontier.pop()
            for w in adjacency.get(v, ()):
                if w not in seen:
                    seen[w] = seen[v] + 1
                    frontier.append(w)
        for v, d in seen.items():
            if v[0] == "room":
                depth[v[1]] = d
    cross_block = 0
    blocked = 0
    for i, room in enumerate(rooms):
        if i not in root_candidates:
            blocked += 1
            if any(c not in (room.color,) for c in _band_blockers(info, i, role[room.color])):
                cross_block += 1
    widths = [e.hi - e.lo + 1 for e in chosen]
    feats = {
        "rooms": len(rooms),
        "anchor_rooms": sum(role[r.color] == "anchor" for r in rooms),
        "outbound_rooms": sum(role[r.color] == "outbound" for r in rooms),
        "sides": {r.side for r in rooms},
        "root_links": {(role[rooms[e.a].color], rooms[e.a].side) for e in root_selected},
        "relay": sum(1 for i in range(len(rooms)) if i not in direct),
        "rejected_root": len(root_candidates - direct),
        "rejected_pair": sum(1 for e in edges if e.b != ROOT and e not in set(chosen)),
        "blocked": blocked,
        "cross_block": cross_block,
        "depth": max(depth.values()) if depth else 0,
        "hollow": sum(w >= 3 for w in widths),
        "solid": sum(w <= 2 for w in widths),
        "links": len(chosen),
    }
    if variants:
        for variant in _VARIANTS:
            feats["diff_" + variant] = not np.array_equal(_variant_output(info, variant), truth)
    return feats


def _analyze_pair(grid):
    """Full analysis of an input grid: parse, graph, tree, render, features."""
    info = _parse(grid)
    edges = _candidates(info)
    chosen, connected, unique = _spanning(info, edges)
    truth = _render(info["canon"], info["bg"], chosen)
    return info, edges, chosen, connected, unique, truth


def _margins_ok(info):
    """Rooms keep a one-cell gap from each other, the separator and the
    parallel outer edges."""
    rooms, k = info["rooms"], info["k"]
    height = info["canon"].shape[0]
    for room in rooms:
        if room.side == 0 and not (room.r0 >= 1 and room.r1 <= k - 2):
            return False
        if room.side == 1 and not (room.r0 >= k + 2 and room.r1 <= height - 2):
            return False
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            if not (a.r1 < b.r0 - 1 or b.r1 < a.r0 - 1 or a.c1 < b.c0 - 1 or b.c1 < a.c0 - 1):
                return False
    return True


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _size(rng, low, high):
    """Room side length, favouring compact rooms but allowing long ones."""
    if high < low:
        return None
    if rng.random() < 0.7:
        return int(rng.integers(low, min(high, low + 3) + 1))
    return int(rng.integers(low, high + 1))


def _gap(rng):
    return int(rng.choice([1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]))


def _fits(room, rooms, height, width, k):
    if room is None:
        return False
    r0, r1, c0, c1 = room.r0, room.r1, room.c0, room.c1
    # Never as wide as the canvas: a full-width wall would read as a second
    # separator.
    if c0 < 0 or c1 >= width or r1 - r0 < 2 or c1 - c0 < 2 or c1 - c0 + 1 >= width:
        return False
    if room.side == 0:
        if r0 < 1 or r1 > k - 2:
            return False
    elif r0 < k + 2 or r1 > height - 2:
        return False
    for other in rooms:
        if not (r1 < other.r0 - 1 or other.r1 < r0 - 1 or c1 < other.c0 - 1 or other.c1 < c0 - 1):
            return False
    return True


def _region_rows(side, height, k):
    return (1, k - 2) if side == 0 else (k + 2, height - 2)


def _random_room(rng, height, width, k, color, sides):
    side = int(rng.choice(sides))
    lo, hi = _region_rows(side, height, k)
    h = _size(rng, 3, min(9, hi - lo + 1))
    w = _size(rng, 3, min(12, width - 1))
    if h is None or w is None:
        return None
    r0 = int(rng.integers(lo, hi - h + 2))
    c0 = int(rng.integers(0, width - w + 1))
    return _Room(r0, r0 + h - 1, c0, c0 + w - 1, color, side)


def _neighbour_room(rng, height, width, k, x, color, direction=None):
    """A room beside x (0 up, 1 down, 2 left, 3 right) sharing interior lanes."""
    lo, hi = _region_rows(x.side, height, k)
    if direction is None:
        direction = int(rng.integers(4))
    g = _gap(rng)
    h = _size(rng, 3, min(9, hi - lo + 1))
    w = _size(rng, 3, min(12 if direction < 2 else 10, width - 1))
    if h is None or w is None:
        return None
    if direction < 2:
        if direction == 0:
            r1 = x.r0 - 1 - g
            r0 = r1 - h + 1
        else:
            r0 = x.r1 + 1 + g
            r1 = r0 + h - 1
        c0 = int(rng.integers(x.c0 + 3 - w, x.c1 - 1))
        return _Room(r0, r1, c0, c0 + w - 1, color, x.side)
    if direction == 2:
        c1 = x.c0 - 1 - g
        c0 = c1 - w + 1
    else:
        c0 = x.c1 + 1 + g
        c1 = c0 + w - 1
    r0 = int(rng.integers(x.r0 + 3 - h, x.r1 - 1))
    return _Room(r0, r0 + h - 1, c0, c1, color, x.side)


def _blocker_room(rng, height, width, k, x, x_role, color):
    """A room placed inside x's straight band toward x's root."""
    downward = (x_role == "anchor") == (x.side == 0)
    if downward:
        low = x.r1 + 2
        high = k - 2 if x.side == 0 else height - 2
    else:
        low = 1 if x.side == 0 else k + 2
        high = x.r0 - 2
    h = _size(rng, 3, min(7, high - low + 1))
    w = _size(rng, 3, min(10, width - 1))
    if h is None or w is None:
        return None
    r0 = int(rng.integers(low, high - h + 2))
    c0 = int(rng.integers(x.c0 + 2 - w, x.c1))
    return _Room(r0, r0 + h - 1, c0, c0 + w - 1, color, x.side)


def _conflict_motif(rng, height, width, k, palette, sides, rooms):
    """Parent A's conflicting lane, resolved by parent B's graph.

    An anchored room lies farther from the separator than an outbound room in
    an overlapping lane.  Each then blocks the other's straight band (blocking
    is symmetric), so both rooms must reach their roots through same-class
    neighbours, which are placed beside them on opposite flanks.
    """
    _, sep, out = palette
    options = [s for s in sides if _region_rows(s, height, k)[1] - _region_rows(s, height, k)[0] >= 8]
    if not options:
        return rooms
    side = int(rng.choice(options))
    lo, hi = _region_rows(side, height, k)
    hx, hy, gap = _size(rng, 3, 5), _size(rng, 3, 5), int(rng.integers(1, 4))
    span = hx + gap + hy
    if span > hi - lo + 1:
        return rooms
    start = int(rng.integers(lo, hi - span + 2))
    wx, wy = _size(rng, 3, min(9, width - 1)), _size(rng, 3, min(9, width - 1))
    cx = int(rng.integers(0, width - wx + 1))
    cy_lo, cy_hi = max(0, cx + 2 - wy), min(width - wy, cx + wx - 2)
    if cy_lo > cy_hi:
        return rooms
    cy = int(rng.integers(cy_lo, cy_hi + 1))
    if side == 0:      # separator below: the anchored room is on top
        x_rows, y_rows = (start, start + hx - 1), (start + hx + gap, start + span - 1)
    else:              # separator above: the anchored room is at the bottom
        y_rows, x_rows = (start, start + hy - 1), (start + hy + gap, start + span - 1)
    far = _Room(x_rows[0], x_rows[1], cx, cx + wx - 1, sep, side)
    near = _Room(y_rows[0], y_rows[1], cy, cy + wy - 1, out, side)
    placed = list(rooms)
    for room in (far, near):
        if not _fits(room, placed, height, width, k):
            return rooms
        placed.append(room)
    left_first = rng.random() < 0.5
    for base, color, to_left in ((far, sep, left_first), (near, out, not left_first)):
        for _ in range(8):
            room = _neighbour_room(rng, height, width, k, base, color, 2 if to_left else 3)
            if _fits(room, placed, height, width, k):
                placed.append(room)
                break
    return placed


def _paint(height, width, k, bg, sep, rooms):
    grid = np.full((height, width), bg, dtype=int)
    grid[k, :] = sep
    for room in rooms:
        grid[room.r0, room.c0:room.c1 + 1] = room.color
        grid[room.r1, room.c0:room.c1 + 1] = room.color
        grid[room.r0:room.r1 + 1, room.c0] = room.color
        grid[room.r0:room.r1 + 1, room.c1] = room.color
    return grid


def _layout(rng, height, width, k, palette, target, outbound_share, sides, motifs):
    _, sep, out = palette
    rooms = []
    for _ in range(motifs):
        rooms = _conflict_motif(rng, height, width, k, palette, sides, rooms)
    for _ in range(target * 10):
        if len(rooms) >= target:
            break
        color = out if rng.random() < outbound_share else sep
        move = rng.random()
        if not rooms or move < 0.3:
            room = _random_room(rng, height, width, k, color, sides)
        elif move < 0.75:
            base = rooms[int(rng.integers(len(rooms)))]
            same = rng.random() < 0.8
            room = _neighbour_room(rng, height, width, k, base, base.color if same else color)
        else:
            base = rooms[int(rng.integers(len(rooms)))]
            base_role = "anchor" if base.color == sep else "outbound"
            other = out if base.color == sep else sep
            room = _blocker_room(rng, height, width, k, base, base_role,
                                 other if rng.random() < 0.7 else base.color)
        if _fits(room, rooms, height, width, k):
            rooms.append(room)
    return rooms


def _pair_dims(rng, size_class):
    if size_class == "small":
        return int(rng.integers(9, 16)), int(rng.integers(8, 16))
    if size_class == "large":
        return int(rng.integers(20, 31)), int(rng.integers(18, 31))
    return int(rng.integers(12, 27)), int(rng.integers(10, 27))


def _instance(palette, orientation, height, width, k, rooms, variants):
    """Build and check one pair from a canonical layout (separator = row k)."""
    bg, sep, _ = palette
    canon = _paint(height, width, k, bg, sep, rooms)
    info = {"bg": bg, "orientation": orientation, "k": k, "sep": sep,
            "canon": canon, "rooms": sorted(rooms)}
    edges = _candidates(info)
    chosen, connected, unique = _spanning(info, edges)
    if not (connected and unique and _clean_rendering(info, edges, chosen)):
        return None
    truth = _render(canon, bg, chosen)
    try:
        if _background(canon) != bg or _background(truth) != bg:
            return None
    except ValueError:
        return None
    feats = _features(info, edges, chosen, truth, variants)
    feats["orientation"] = orientation
    feats["palette"] = palette
    grid, output = (canon, truth) if orientation == "h" else (canon.T.copy(), truth.T.copy())
    feats["shape"] = grid.shape
    return grid, output, feats


def _try_pair(rng, palette, orientation, request, variants):
    size_class, target, outbound_share, thin, motifs = request(rng)
    height, width = _pair_dims(rng, size_class)
    if thin:
        depth = int(rng.integers(1, 5))
        k = depth if rng.random() < 0.5 else height - 1 - depth
        sides = [1] if k < height // 2 else [0]
        if height - 1 - depth < 5:
            return None
    else:
        if height < 11:
            return None
        k = int(rng.integers(5, height - 5))
        sides = [0, 1]
    rooms = _layout(rng, height, width, k, palette, target, outbound_share, sides, motifs)
    if len(rooms) < 2:
        return None
    return _instance(palette, orientation, height, width, k, rooms, variants)


def _key(grid):
    return grid.tobytes() + bytes(grid.shape)


def _sample(rng, palette, orientation, request, seen, accept, attempts=300, variants=False):
    for _ in range(attempts):
        item = _try_pair(rng, palette, orientation, request, variants)
        if item is not None and _key(item[0]) not in seen and accept(item[2]):
            return item
    return None


# Training evidence required jointly from the two demonstration pairs: root
# links of both roles on both sides, a relay, a rejected direct link, a
# blocked band and a cross-class lane conflict, both corridor styles, and a
# visible refutation of every shortcut rule in _VARIANTS.
_COVERAGE = ("anchor0", "anchor1", "outbound0", "outbound1", "relay", "rejected_root",
             "blocked", "cross_block", "hollow", "solid") + tuple("diff_" + v for v in _VARIANTS)


def _coverage(feats):
    items = {f"{role}{side}" for role, side in feats["root_links"]}
    for key in ("relay", "rejected_root", "blocked", "cross_block", "hollow", "solid"):
        if feats[key]:
            items.add(key)
    for variant in _VARIANTS:
        if feats.get("diff_" + variant):
            items.add("diff_" + variant)
    return frozenset(items)


def _palettes(rng, mode, count):
    def fresh():
        bg = 0 if rng.random() < 0.5 else int(rng.integers(10))
        sep, out = (int(v) for v in rng.choice([c for c in range(10) if c != bg], 2, replace=False))
        return bg, sep, out

    if mode == "fixed":
        return [fresh()] * count
    if mode == "swap":
        # Two room colours trade roles as the separator colour changes.
        bg, a, b = fresh()
        roles = [(a, b), (b, a)] + [((a, b), (b, a))[int(rng.integers(2))] for _ in range(count - 2)]
        return [(bg, s, o) for s, o in roles]
    palettes = [fresh()]
    while len(palettes) < count:
        p = fresh()
        if len(palettes) == 1 and p[1] == palettes[0][1]:
            continue
        palettes.append(p)
    return palettes


def _orientations(rng, mode, num_train, num_test):
    first = "h" if rng.random() < 0.5 else "v"
    second = "v" if first == "h" else "h"
    if mode == "single":
        return [first] * (num_train + num_test)
    if mode == "rotated":
        return [first] * num_train + [second] * num_test
    rest = ["h" if rng.random() < 0.5 else "v" for _ in range(num_train + num_test - 2)]
    return [first, second] + rest


def _check_counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, _MIN_TRAIN, _MAX_TRAIN),
                                   ("num_test", num_test, _MIN_TEST, _MAX_TEST)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must be in {low}..{high}")
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _novelties(feats, train_feats):
    """Ways in which a test pair goes beyond every training pair."""
    found = set()
    if feats["orientation"] not in {f["orientation"] for f in train_feats}:
        found.add("orientation")
    if feats["rooms"] > max(f["rooms"] for f in train_feats):
        found.add("more_rooms")
    if feats["depth"] > max(f["depth"] for f in train_feats):
        found.add("deeper_chain")
    if feats["cross_block"] > max(f["cross_block"] for f in train_feats):
        found.add("more_conflicts")
    if len(feats["sides"]) == 1 and all(len(f["sides"]) == 2 for f in train_feats):
        found.add("one_side")
    if min(feats["anchor_rooms"], feats["outbound_rooms"]) == 0 and all(
            min(f["anchor_rooms"], f["outbound_rooms"]) > 0 for f in train_feats):
        found.add("one_class")
    if feats["shape"][0] * feats["shape"][1] > max(f["shape"][0] * f["shape"][1] for f in train_feats):
        found.add("larger_canvas")
    return found


def _demo_request(rng):
    return (str(rng.choice(["medium", "large"], p=[0.55, 0.45])), int(rng.integers(4, 9)),
            0.45, False, int(rng.choice([0, 1, 1, 2])))


def _free_request(rng):
    size_class = str(rng.choice(["small", "medium", "large"], p=[0.3, 0.45, 0.25]))
    target = {"small": int(rng.integers(2, 4)), "medium": int(rng.integers(3, 7)),
              "large": int(rng.integers(4, 10))}[size_class]
    share = float(rng.choice([0.0, 0.3, 0.5, 0.7, 1.0], p=[0.08, 0.27, 0.3, 0.27, 0.08]))
    motifs = int(rng.random() < 0.3) if 0.0 < share < 1.0 else 0
    return size_class, target, share, bool(rng.random() < 0.12), motifs


_SHIFT_REQUESTS = {
    "more_rooms": lambda r: ("large", int(r.integers(8, 13)), 0.5, False, int(r.integers(0, 2))),
    "deeper_chain": lambda r: (str(r.choice(["medium", "large"])), int(r.integers(4, 10)), 0.4, False, 0),
    "one_class": lambda r: (str(r.choice(["small", "medium", "large"])), int(r.integers(2, 7)),
                            float(r.choice([0.0, 1.0])), False, 0),
    "one_side": lambda r: (str(r.choice(["medium", "large"])), int(r.integers(3, 8)), 0.5, True,
                           int(r.integers(0, 2))),
    "more_conflicts": lambda r: ("large", int(r.integers(6, 11)), 0.5, False, 2),
}


def _demonstrations(rng, palettes, orientations, seen):
    """Pick two demonstration pairs that jointly cover _COVERAGE."""
    def first_ok(f):
        return (f["anchor_rooms"] and f["outbound_rooms"] and len(f["sides"]) == 2 and f["relay"]
                and {"anchor", "outbound"} <= {role for role, _ in f["root_links"]})

    def second_ok(f):
        return f["anchor_rooms"] and f["outbound_rooms"] and (f["rejected_root"] or f["rejected_pair"])

    pools = ([], [])
    need = frozenset(_COVERAGE)
    for _ in range(8):
        for slot, accept in ((0, first_ok), (1, second_ok)):
            for _ in range(4):
                item = _sample(rng, palettes[slot], orientations[slot], _demo_request, seen, accept,
                               attempts=60, variants=True)
                if item is not None:
                    pools[slot].append((item, _coverage(item[2])))
        for a, cover_a in pools[0]:
            for b, cover_b in pools[1]:
                if need <= (cover_a | cover_b) and _key(a[0]) != _key(b[0]):
                    return a, b
    return None


def _try_puzzle(rng, num_train, num_test):
    palette_mode = str(rng.choice(["fixed", "varied", "swap"], p=[0.4, 0.35, 0.25]))
    orientation_mode = str(rng.choice(["single", "mixed", "rotated"], p=[0.35, 0.4, 0.25]))
    palettes = _palettes(rng, palette_mode, num_train + num_test)
    orientations = _orientations(rng, orientation_mode, num_train, num_test)
    seen = set()
    demos = _demonstrations(rng, palettes, orientations, seen)
    if demos is None:
        return None
    pairs = list(demos)
    seen.update(_key(item[0]) for item in pairs)
    for index in range(2, num_train):
        item = _sample(rng, palettes[index], orientations[index], _free_request, seen,
                       lambda f: f["links"] >= 1)
        if item is None:
            return None
        pairs.append(item)
        seen.add(_key(item[0]))
    train_feats = [item[2] for item in pairs]
    for index in range(num_train, num_train + num_test):
        goal = str(rng.choice(sorted(_SHIFT_REQUESTS)))
        if orientations[index] not in {f["orientation"] for f in train_feats}:
            accept = lambda f: True  # the rotation itself is the shift
        else:
            accept = lambda f, g=goal: g in _novelties(f, train_feats)
        item = _sample(rng, palettes[index], orientations[index], _SHIFT_REQUESTS[goal], seen, accept)
        if item is None:
            item = _sample(rng, palettes[index], orientations[index], _free_request, seen,
                           lambda f: bool(_novelties(f, train_feats)))
        if item is None:
            return None
        pairs.append(item)
        seen.add(_key(item[0]))
    return {
        "train": [{"input": p[0], "output": p[1]} for p in pairs[:num_train]],
        "test": [{"input": p[0], "output": p[1]} for p in pairs[num_train:]],
    }


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(60):
        puzzle = _try_puzzle(rng, num_train, num_test)
        if puzzle is not None:
            return puzzle
    raise RuntimeError("could not build a puzzle for this seed")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _brute_force_tree(info, edges):
    """Independent check of the minimum spanning forest for small graphs."""
    from itertools import combinations

    results = {}
    for color in {room.color for room in info["rooms"]}:
        mine = [e for e in edges if e.color == color]
        n = sum(room.color == color for room in info["rooms"])
        if len(mine) > 16:
            results[color] = None
            continue
        best, best_sets = None, []
        for subset in combinations(mine, n):
            parent = {}

            def find(v):
                parent.setdefault(v, v)
                while parent[v] != v:
                    v = parent[v]
                return v

            ok = True
            for e in subset:
                u, v = find(_vertex(e.a, color)), find(_vertex(e.b, color))
                if u == v:
                    ok = False
                    break
                parent[u] = v
            if not ok:
                continue
            total = sum(e.weight for e in subset)
            if best is None or total < best:
                best, best_sets = total, [set(subset)]
            elif total == best:
                best_sets.append(set(subset))
        results[color] = best_sets
    return results


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert _MIN_TRAIN <= len(train) <= _MAX_TRAIN and _MIN_TEST <= len(test) <= _MAX_TEST
    assert len(train) > len(test)
    seen = set()
    train_feats, all_feats = [], []
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            for g in (inp, out):
                assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
                assert np.issubdtype(g.dtype, np.integer) and g.min() >= 0 and g.max() <= 9
            assert inp.shape == out.shape
            key = inp.tobytes() + bytes(inp.shape)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info, edges, chosen, connected, unique, truth = _analyze_pair(inp)
            assert connected, "every room must reach its root"
            assert unique, "minimum spanning tree must be unique"
            assert _margins_ok(info) and _clean_rendering(info, edges, chosen)
            assert len(info["rooms"]) >= 2
            expected = np.array(_uncanon(info, truth))
            assert np.array_equal(out, expected) and np.array_equal(solve(inp), out)
            assert not np.array_equal(inp, out)
            # Independent spanning-tree check where the graph is small enough.
            for color, best_sets in _brute_force_tree(info, edges).items():
                if best_sets is None:
                    continue
                mine = {e for e in chosen if e.color == color}
                assert len(best_sets) == 1 and best_sets[0] == mine
            # Only corridor rectangles change; the separator never changes.
            canon_out = out if info["orientation"] == "h" else out.T
            changed = np.argwhere(canon_out != info["canon"])
            for r, c in changed:
                assert r != info["k"]
                assert any((e.lo <= c <= e.hi and e.e0 <= r <= e.e1) if e.axis == "v"
                           else (e.lo <= r <= e.hi and e.e0 <= c <= e.e1) for e in chosen)
            # Corridors use the colour of their class and the separator's
            # colour exactly for anchored rooms.
            for e in chosen:
                room = info["rooms"][e.a]
                assert e.color == room.color
                if e.b == ROOT:
                    if room.color == info["sep"]:
                        assert (e.e1 == info["k"] - 1) if room.side == 0 else (e.e0 == info["k"] + 1)
                    else:
                        limit = info["canon"].shape[0] - 1
                        assert (e.e0 == 0) if room.side == 0 else (e.e1 == limit)
            feats = _features(info, edges, chosen, truth)
            feats["orientation"] = info["orientation"]
            feats["shape"] = inp.shape
            all_feats.append(feats)
            if split == "train":
                train_feats.append(feats)
    covered = set().union(*(_coverage(f) for f in train_feats))
    assert set(_COVERAGE) <= covered, f"training lacks {sorted(set(_COVERAGE) - covered)}"
    for feats in all_feats[len(train):]:
        assert _novelties(feats, train_feats), "each test pair must extend the training set"


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _expect_value_error(**kwargs):
    try:
        generate_puzzle(5, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"accepted illegal counts {kwargs}")


def self_test() -> None:
    shapes, palettes, orientations, room_counts = set(), set(), set(), set()
    novelty_kinds = set()
    stats = {"relay": 0, "cross_block": 0, "deep": 0, "hollow": 0, "solid": 0,
             "one_class": 0, "one_side": 0, "mixed_orientation": 0, "rejected_root": 0,
             "side_exit_trap": 0, "constant_palette": 0, "role_swap": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        pair_orientations = set()
        pair_palettes, anchored_colors, outbound_colors = set(), set(), set()
        train_feats = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info, edges, chosen, _, _, truth = _analyze_pair(pair["input"])
                feats = _features(info, edges, chosen, truth)
                feats["orientation"] = info["orientation"]
                feats["shape"] = np.asarray(pair["input"]).shape
                shapes.add(feats["shape"])
                palettes.add((info["bg"], info["sep"]))
                pair_palettes.add((info["bg"], info["sep"],
                                   frozenset(room.color for room in info["rooms"])))
                for room in info["rooms"]:
                    (anchored_colors if room.color == info["sep"] else outbound_colors).add(room.color)
                orientations.add(info["orientation"])
                pair_orientations.add(info["orientation"])
                room_counts.add(feats["rooms"])
                stats["relay"] += feats["relay"] > 0
                stats["cross_block"] += feats["cross_block"] > 0
                stats["deep"] += feats["depth"] >= 3
                stats["hollow"] += feats["hollow"] > 0
                stats["solid"] += feats["solid"] > 0
                stats["one_class"] += min(feats["anchor_rooms"], feats["outbound_rooms"]) == 0
                stats["one_side"] += len(feats["sides"]) == 1
                stats["rejected_root"] += feats["rejected_root"] > 0
                stats["side_exit_trap"] += feats["diff_side_exits"]
                if split == "train":
                    train_feats.append(feats)
                else:
                    novelty_kinds |= _novelties(feats, train_feats)
        stats["mixed_orientation"] += len(pair_orientations) == 2
        stats["constant_palette"] += len({p[:2] for p in pair_palettes}) == 1
        stats["role_swap"] += bool(anchored_colors & outbound_colors)
    assert orientations == {"h", "v"}
    assert stats["relay"] >= 60 and stats["rejected_root"] >= 30
    assert stats["hollow"] >= 60 and stats["solid"] >= 60
    assert stats["constant_palette"] >= 5 and stats["role_swap"] >= 5
    assert len(shapes) >= 60 and len(palettes) >= 25
    assert min(room_counts) <= 2 and max(room_counts) >= 9
    assert stats["mixed_orientation"] >= 10
    assert stats["cross_block"] >= 20 and stats["deep"] >= 10
    assert stats["one_class"] >= 5 and stats["one_side"] >= 5
    assert stats["side_exit_trap"] >= 10
    assert len(novelty_kinds) >= 5
    # Determinism and count handling.
    a, b = generate_puzzle(123, 5, 2), generate_puzzle(123, 5, 2)
    for split in ("train", "test"):
        for x, y in zip(a[split], b[split]):
            assert np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"])
    for num_train, num_test in ((2, 1), (10, 4), (5, 4)):
        p = generate_puzzle(77, num_train, num_test)
        assert len(p["train"]) == num_train and len(p["test"]) == num_test
    for bad in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
                {"num_train": 4.0}, {"num_train": True}):
        _expect_value_error(**bad)


if __name__ == "__main__":
    self_test()
    print("self_test passed")
