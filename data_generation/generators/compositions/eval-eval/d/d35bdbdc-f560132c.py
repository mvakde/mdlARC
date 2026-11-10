"""Seam-mediated relations: packing creates the context for color composition.

Only standard-library modules and NumPy are used. Nothing from parents is
imported. The solver reads footprints, marks, and relations from input pixels.
"""
from __future__ import annotations

import heapq
import itertools
import math

import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    if grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(grid.dtype, np.integer) or not np.all((0 <= grid) & (grid <= 9)):
        raise ValueError("grid must contain integer colors 0..9")
    return grid.astype(np.int8, copy=True)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, part = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D4:
                q = (r + dr, c + dc)
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
                    part.append(q)
        result.append(tuple(sorted(part)))
    return result


def _normalize(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _dims(shape):
    return max(r for r, c in shape) + 1, max(c for r, c in shape) + 1


def _rotate(shape, turns):
    for _ in range(turns % 4):
        h, _ = _dims(shape)
        shape = tuple(sorted((c, h - 1 - r) for r, c in shape))
    return shape


def _mask(shape, side, row=0, col=0):
    return sum(1 << ((r + row) * side + c + col) for r, c in shape)


def _pack(shapes, side, limit=2):
    """Enumerate unique placements, including piece identities, from shapes.

    Corner ownership fixes bounding-box translation. Deduplicate symmetric
    orientations only when they put that same piece in the same cells.
    """
    if len(shapes) != 4 or sum(map(len, shapes)) != side * side:
        return []
    corner_bits = [1 << p for p in (0, side - 1, side * (side - 1), side * side - 1)]
    if (0, 0) not in shapes[0] or max(_dims(shapes[0])) > side:
        return []
    anchor = _mask(shapes[0], side)
    if [bool(anchor & bit) for bit in corner_bits] != [True, False, False, False]:
        return []
    candidates = {}
    for piece in range(1, 4):
        for corner in range(1, 4):
            placements = {}
            for k in range(4):
                rotated = _rotate(shapes[piece], k)
                h, w = _dims(rotated)
                if h > side or w > side:
                    continue
                r = 0 if corner < 2 else side - h
                c = 0 if corner % 2 == 0 else side - w
                m = _mask(rotated, side, r, c)
                hits = [bool(m & bit) for bit in corner_bits]
                if sum(hits) == 1 and hits[corner]:
                    placements.setdefault(m, (k, r, c))
            candidates[piece, corner] = tuple(placements.items())
    full = (1 << (side * side)) - 1
    masks, moves = [anchor, 0, 0, 0], [(0, 0, 0), None, None, None]
    solutions, seen = [], set()

    def visit(corner, remaining, occupied):
        if len(solutions) >= limit:
            return
        if corner == 4:
            signature = tuple(masks)
            if occupied == full and signature not in seen:
                seen.add(signature)
                owner = np.empty((side, side), dtype=np.int8)
                for p, m in enumerate(masks):
                    for pos in range(side * side):
                        if (m >> pos) & 1:
                            owner[pos // side, pos % side] = p
                solutions.append({"owner": owner, "masks": signature, "moves": tuple(moves)})
            return
        for p in remaining:
            for m, move in candidates[p, corner]:
                if not m & occupied:
                    masks[p], moves[p] = m, move
                    visit(corner + 1, tuple(q for q in remaining if q != p), occupied | m)
                    if len(solutions) >= limit:
                        return

    visit(1, (1, 2, 3), anchor)
    return solutions


def _parse(grid):
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    if not np.all(border == border[0]):
        raise ValueError("input needs a uniform background border")
    bg = int(border[0])
    components = _components(grid != bg)
    if len(components) != 4:
        raise ValueError("input needs four disconnected footprints")
    pieces, anchors = [], []
    for cells in components:
        coords = np.asarray(cells)
        values = grid[coords[:, 0], coords[:, 1]]
        colors, counts = np.unique(values, return_counts=True)
        body = int(colors[np.argmax(counts)])
        if np.count_nonzero(counts == counts.max()) != 1:
            raise ValueError("piece body must be uniquely dominant")
        singleton = colors[counts == 1]
        if len(singleton) != 1:
            raise ValueError("each piece needs one differently colored singleton badge")
        badge = int(singleton[0])
        bp = cells[int(np.flatnonzero(values == badge)[0])]
        others = [int(c) for c in colors if c not in (body, badge)]
        piece = {"cells": cells, "shape": _normalize(cells), "body": body,
                 "badge": badge, "badge_point": bp, "marker": None}
        if others:
            if len(others) != 1 or np.count_nonzero(values == others[0]) != 4 or counts.max() < 6:
                raise ValueError("anchor must have one four-cell marker and at least six body cells")
            marker_cells = [p for p in cells if grid[p] == others[0]]
            r, c = min(p[0] for p in marker_cells), min(p[1] for p in marker_cells)
            if set(marker_cells) != {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}:
                raise ValueError("anchor marker must be a uniform 2 by 2 block")
            piece["marker"] = others[0]
            piece["marker_point"] = (r, c)
            anchors.append(piece)
        elif counts.max() < 5:
            raise ValueError("ordinary pieces need at least five body cells")
        pieces.append(piece)
    if len(anchors) != 1:
        raise ValueError("exactly one marked anchor required")
    anchor = anchors[0]
    marker = anchor["marker"]
    if any(p["body"] == marker or p["badge"] == marker for p in pieces):
        raise ValueError("marker cannot also be a relation identity")
    for a, b in itertools.combinations(pieces, 2):
        if min(max(abs(r - u), abs(c - v)) for r, c in a["cells"] for u, v in b["cells"]) < 2:
            raise ValueError("footprints need a full background gap")
    # All other pieces remain in input reading order. This has no solving role.
    pieces = [anchor] + [p for p in pieces if p is not anchor]
    area = sum(len(p["cells"]) for p in pieces)
    side = math.isqrt(area)
    if side * side != area or side > 30:
        raise ValueError("combined footprint area must be a square")
    return {"background": bg, "marker": marker, "pieces": pieces, "side": side}


def _seams(owner):
    seams = np.zeros((4, 4), dtype=int)
    for a, b in ((owner[:-1], owner[1:]), (owner[:, :-1], owner[:, 1:])):
        for x, y in zip(a.ravel(), b.ravel()):
            x, y = int(x), int(y)
            if x != y:
                seams[x, y] += 1
                seams[y, x] += 1
    return seams


def _result_colors(bodies, badges, seams, bg):
    colors, winners = [], []
    for i in range(4):
        candidates = [j for j in range(4) if j != i and bodies[j] == badges[i] and seams[i, j] > 0]
        if not candidates:
            colors.append(bg)
            winners.append(())
            continue
        length = max(int(seams[i, j]) for j in candidates)
        best = tuple(j for j in candidates if seams[i, j] == length)
        values = {badges[j] for j in best}
        if len(values) != 1:
            raise ValueError("conflicting badges tied at the longest matching seam")
        colors.append(values.pop())
        winners.append(best)
    return np.asarray(colors, dtype=np.int8), tuple(winners)


def _analyze(input_grid):
    grid = _grid(input_grid)
    a = _parse(grid)
    solutions = _pack([p["shape"] for p in a["pieces"]], a["side"])
    if len(solutions) != 1:
        raise ValueError("footprints need one unique corner-constrained packing")
    a.update(solutions[0])
    a["seams"] = _seams(a["owner"])
    bodies = [p["body"] for p in a["pieces"]]
    badges = [p["badge"] for p in a["pieces"]]
    a["colors"], a["winners"] = _result_colors(bodies, badges, a["seams"], a["background"])
    a["output"] = a["colors"][a["owner"]].copy()
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the fitted, simultaneously composed square using input only."""
    return _analyze(input_grid)["output"]


def _growth(rng, side):
    owner = np.full((side, side), -1, dtype=np.int8)
    corners = ((0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1))
    reserved = {q: p for p, q in enumerate(corners)}
    weights = rng.uniform(.5, 2.5, (4, side, side))
    speeds = rng.uniform(.7, 1.6, 4)
    heap = [(0., p, *q) for p, q in enumerate(corners)]
    heapq.heapify(heap)
    while heap:
        distance, p, r, c = heapq.heappop(heap)
        if owner[r, c] != -1 or ((r, c) in reserved and reserved[r, c] != p):
            continue
        owner[r, c] = p
        for dr, dc in _D4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < side and 0 <= cc < side and owner[rr, cc] == -1:
                heapq.heappush(heap, (distance + float(weights[p, rr, cc] / speeds[p]), p, rr, cc))
    return owner


def _cuts(rng, side):
    owner = np.empty((side, side), dtype=np.int8)
    cut_row = int(rng.integers(2, side - 1))
    for start, end, left, right in ((0, cut_row, 0, 1), (cut_row, side, 2, 3)):
        cut = int(rng.integers(1, side))
        wandering = float(rng.choice([0., .35, .7, 1.]))
        for r in range(start, end):
            if r > start and rng.random() < wandering:
                cut = int(np.clip(cut + rng.choice([-1, 1]), 1, side - 1))
            owner[r, :cut], owner[r, cut:] = left, right
    return owner


def _exchange(rng, owner):
    side = len(owner)
    corners = {(0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1)}
    for _ in range(int(rng.integers(0, 2 * side + 1))):
        r, c = int(rng.integers(side)), int(rng.integers(side))
        if (r, c) in corners:
            continue
        before = int(owner[r, c])
        neighbors = sorted({int(owner[r + dr, c + dc]) for dr, dc in _D4
                            if 0 <= r + dr < side and 0 <= c + dc < side} - {before})
        if not neighbors:
            continue
        after = int(rng.choice(neighbors))
        owner[r, c] = after
        if np.count_nonzero(owner == before) < (11 if before == 0 else 6) or len(_components(owner == before)) != 1:
            owner[r, c] = before
    return owner


def _patches(owner):
    return [(r, c) for r in range(len(owner) - 1) for c in range(len(owner) - 1)
            if np.all(owner[r:r + 2, c:c + 2] == 0)]


def _geometry(owner, asymmetric=False):
    if tuple(int(owner[p]) for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))) != (0, 1, 2, 3):
        return None
    shapes = []
    for p in range(4):
        parts = _components(owner == p)
        if len(parts) != 1 or len(parts[0]) < (11 if p == 0 else 6):
            return None
        shape = _normalize(parts[0])
        if asymmetric and len({_rotate(shape, k) for k in range(4)}) != 4:
            return None
        shapes.append(shape)
    if not _patches(owner):
        return None
    solutions = _pack(shapes, len(owner))
    if len(solutions) != 1 or not np.array_equal(solutions[0]["owner"], owner):
        return None
    return shapes


def _relations(rng, seams, mode, identities, bg):
    palette = list(map(int, rng.permutation(identities)))
    if mode == "chain":
        bodies = palette[:4]
        candidates = []
        for b in (1, 2, 3):
            for c in (1, 2, 3):
                if b == c or not seams[0, b] or not seams[b, c]:
                    continue
                d = ({1, 2, 3} - {b, c}).pop()
                for e in range(4):
                    if e == d or seams[d, e]:
                        continue
                    badges = [0] * 4
                    badges[0], badges[b], badges[c], badges[d] = bodies[b], bodies[c], palette[4], bodies[e]
                    colors, winners = _result_colors(bodies, badges, seams, bg)
                    # A source must also refute longest-unrestricted-neighbor.
                    wrong_priority = any(winners[i] and max(seams[i]) > max(seams[i, j] for j in winners[i])
                                         for i in range(4))
                    if wrong_priority:
                        candidates.append((bodies, badges))
        return candidates[int(rng.integers(len(candidates)))] if candidates else None
    if mode == "competition":
        choices = []
        for b in (1, 2, 3):
            for c in (1, 2, 3):
                if b == c:
                    continue
                d = ({1, 2, 3} - {b, c}).pop()
                # Two sources requesting the same identity must prefer opposite
                # donors. A fixed input-order preference cannot explain both.
                if (min(seams[0, b], seams[0, c], seams[d, b], seams[d, c]) > 0
                        and (seams[0, b] - seams[0, c]) * (seams[d, b] - seams[d, c]) < 0):
                    choices.append((b, c))
        if not choices:
            return None
        b, c = choices[int(rng.integers(len(choices)))]
        d = ({1, 2, 3} - {b, c}).pop()
        x, y, z, w = palette[:4]
        bodies, badges = [x] * 4, [y] * 4
        bodies[b], bodies[c], bodies[d] = y, y, z
        badges[b], badges[c], badges[d] = z, w, y
        return bodies, badges
    for _ in range(64):
        k = int(rng.integers(3, 7))
        pool = palette[:k]
        n_bodies = int(rng.integers(2, min(4, k) + 1))
        source_pool = pool[:n_bodies]
        if n_bodies == 4 and rng.random() < .6:
            bodies = list(map(int, rng.permutation(source_pool)))
        else:
            bodies = list(map(int, rng.choice(source_pool, 4)))
        badges = []
        for body in bodies:
            available = [v for v in pool if v != body]
            matching = [v for v in available if v in bodies]
            badges.append(int(rng.choice(matching if matching and rng.random() < .75 else available)))
        try:
            colors, _ = _result_colors(bodies, badges, seams, bg)
        except ValueError:
            continue
        if np.any(colors != bg):
            return bodies, badges
    return None


def _shelf_scatter(rng, shapes, style):
    gap = int(rng.integers(1, 4))
    layouts = []
    for transpose in (False, True):
        dimensions = [tuple(reversed(_dims(s))) if transpose else _dims(s) for s in shapes]
        for order in itertools.permutations(range(4)):
            for breaks in range(8):
                groups, group = [], []
                for j, p in enumerate(order):
                    group.append(p)
                    if j == 3 or (breaks >> j) & 1:
                        groups.append(group)
                        group = []
                widths = [sum(dimensions[p][1] for p in g) + gap * (len(g) - 1) for g in groups]
                heights = [max(dimensions[p][0] for p in g) for g in groups]
                h, w = sum(heights) + gap * (len(groups) - 1), max(widths)
                if h <= 28 and w <= 28:
                    layouts.append((transpose, dimensions, groups, heights, widths, h, w))
    if not layouts:
        # Four generated footprints are at most 12 by 12: two shelves
        # with even the largest sampled gap fit inside the 28-cell extent.
        raise RuntimeError("footprints exceed the supported scatter extent")
    linear = [v for v in layouts if len(v[2]) == 1]
    pool = linear if style == 1 and linear else layouts
    transpose, dims, groups, heights, widths, h, w = pool[int(rng.integers(len(pool)))]
    offsets, row = [None] * 4, 0
    for group, height, width in zip(groups, heights, widths):
        col = int(rng.integers(w - width + 1))
        for p in group:
            rr = row + int(rng.integers(height - dims[p][0] + 1))
            offsets[p] = (col, rr) if transpose else (rr, col)
            col += dims[p][1] + gap
        row += height + gap
    extent = (w, h) if transpose else (h, w)
    pads = []
    for length in extent:
        room = 30 - length
        before = int(rng.integers(1, min(4, room - 1) + 1))
        after = int(rng.integers(1, min(5, room - before) + 1))
        pads.append((before, after))
    return ([(r + pads[0][0], c + pads[1][0]) for r, c in offsets],
            tuple(extent[a] + sum(pads[a]) for a in (0, 1)))


def _scatter(rng, shapes, style):
    if style == 0:
        h, w = int(rng.integers(20, 31)), int(rng.integers(20, 31))
        blocked = np.zeros((h, w), dtype=bool)
        offsets = [None] * 4
        for p in sorted(range(4), key=lambda i: -len(shapes[i])):
            coords = np.asarray(shapes[p])
            ph, pw = _dims(shapes[p])
            for _ in range(80):
                r, c = int(rng.integers(1, h - ph)), int(rng.integers(1, w - pw))
                rr, cc = coords[:, 0] + r, coords[:, 1] + c
                if blocked[rr, cc].any():
                    continue
                offsets[p] = (r, c)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        blocked[rr + dr, cc + dc] = True
                break
            if offsets[p] is None:
                break
        if all(p is not None for p in offsets):
            return offsets, (h, w)
    return _shelf_scatter(rng, shapes, style)


def _evidence(a):
    pieces, seams, results = a["pieces"], a["seams"], a["colors"]
    bg = a["background"]
    f = {"nonneighbor": 0, "competition": 0, "simultaneous": 0,
         "erased_donor": 0, "input_order_wrong": 0, "input_reverse_wrong": 0, "priority_wrong": 0,
         "own_badge_wrong": 0, "own_body_wrong": 0}
    for i, p in enumerate(pieces):
        matches = [j for j, q in enumerate(pieces) if j != i and q["body"] == p["badge"]]
        adjacent = [j for j in matches if seams[i, j] > 0]
        if results[i] == bg and matches and not adjacent:
            f["nonneighbor"] += 1
        if adjacent:
            values = {pieces[j]["badge"] for j in adjacent}
            if len(values) > 1 and max(seams[i, j] for j in adjacent) > min(seams[i, j] for j in adjacent):
                f["competition"] += 1
            first = min(adjacent, key=lambda j: pieces[j]["cells"][0])
            last = max(adjacent, key=lambda j: pieces[j]["cells"][0])
            f["input_order_wrong"] += pieces[first]["badge"] != results[i]
            f["input_reverse_wrong"] += pieces[last]["badge"] != results[i]
            if max(seams[i]) > max(seams[i, j] for j in a["winners"][i]):
                f["priority_wrong"] += 1
        for j in a["winners"][i]:
            if results[j] != bg and results[j] != pieces[j]["badge"]:
                f["simultaneous"] += 1
            if results[j] == bg:
                f["erased_donor"] += 1
        f["own_badge_wrong"] += results[i] != p["badge"]
        f["own_body_wrong"] += results[i] != p["body"]
    return {k: int(v) for k, v in f.items()}


def _make_pair(rng, side, mode, roles, style, growth_weight):
    bg, marker = roles
    identities = [c for c in range(10) if c not in roles]
    # Bounded stochastic construction. A finite deterministic cut search below
    # handles any exceptional failure, without using precomputed puzzle pairs.
    def owners():
        for _ in range(240):
            base = _growth(rng, side) if rng.random() < growth_weight else _cuts(rng, side)
            yield _exchange(rng, base)
        for y in range(2, side - 1):
            for x in range(2, side - 1):
                for z in range(1, side - 1):
                    for t in range(1, side - 1):
                        owner = np.empty((side, side), dtype=np.int8)
                        owner[:y, :x], owner[:y, x:] = 0, 1
                        owner[y:, :z], owner[y:, z:] = 2, 3
                        owner[y - 1, t] = owner[y, t]
                        owner[1, x] = 0
                        owner[side - 2, z] = 2
                        yield owner

    for owner in owners():
        shapes = _geometry(owner, asymmetric=mode == "chain")
        if shapes is None:
            continue
        seams = _seams(owner)
        relations = _relations(rng, seams, mode, identities, bg)
        if relations is None:
            continue
        bodies, badges = relations
        patches = _patches(owner)
        mr, mc = patches[int(rng.integers(len(patches)))]
        marker_cells = {(mr, mc), (mr + 1, mc), (mr, mc + 1), (mr + 1, mc + 1)}
        values = []
        for p in range(4):
            points = [(int(r), int(c)) for r, c in np.argwhere(owner == p)
                      if p != 0 or (r, c) not in marker_cells]
            badge_point = points[int(rng.integers(len(points)))]
            mins = np.argwhere(owner == p).min(axis=0)
            h, w = _dims(shapes[p])
            picture = np.full((h, w), -1, dtype=np.int8)
            for r, c in shapes[p]:
                picture[r, c] = bodies[p]
            br, bc = badge_point[0] - int(mins[0]), badge_point[1] - int(mins[1])
            picture[br, bc] = badges[p]
            if p == 0:
                picture[mr:mr + 2, mc:mc + 2] = marker
            values.append(picture)
        rotations = [0] + (list(map(int, rng.permutation([1, 2, 3]))) if mode == "chain"
                            else list(map(int, rng.integers(0, 4, 3))))
        values = [np.rot90(v, -k).copy() for v, k in zip(values, rotations)]
        oriented = [tuple(map(tuple, np.argwhere(v != -1).tolist())) for v in values]
        # Symmetry deduplication in the input solver must recover the same cells.
        fitted = _pack(oriented, side)
        if len(fitted) != 1:
            continue
        colors, _ = _result_colors(bodies, badges, seams, bg)
        output = colors[owner].copy()
        for _ in range(32):
            offsets, canvas = _scatter(rng, oriented, style)
            grid = np.full(canvas, bg, dtype=np.int8)
            for p, coords in enumerate(oriented):
                r, c = offsets[p]
                for dr, dc in coords:
                    grid[r + dr, c + dc] = values[p][dr, dc]
            a = _analyze(grid)
            if not np.array_equal(a["output"], output):
                raise AssertionError("sampled partition and input reconstruction disagree")
            f = _evidence(a)
            if mode == "competition" and not (f["input_order_wrong"] and f["input_reverse_wrong"]):
                continue
            if mode == "chain" and not (f["nonneighbor"] and f["simultaneous"] and f["priority_wrong"] and f["erased_donor"]):
                continue
            return {"input": grid, "output": output}
    raise RuntimeError("bounded partition search found no admissible geometry")


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    growth_shift = bool(rng.integers(2))
    varied_roles = bool(rng.random() < .65)
    fixed_roles = tuple(map(int, rng.choice(10, 2, replace=False)))
    style = int(rng.integers(3))
    growth_weight = float(rng.uniform(.2, .8))
    modes = ["chain", "competition"] + ["random"] * (nt - 2)
    rng.shuffle(modes)
    puzzle, seen = {"train": [], "test": []}, set()
    first_roles = None
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            large = (split == "test") == growth_shift
            side = int(rng.integers(10, 13) if large else rng.integers(7, 10))
            roles = tuple(map(int, rng.choice(10, 2, replace=False))) if varied_roles else fixed_roles
            if split == "train" and index == 1 and varied_roles:
                while roles == first_roles:
                    roles = tuple(map(int, rng.choice(10, 2, replace=False)))
            if split == "train" and index == 0:
                first_roles = roles
            mode = modes[index] if split == "train" else "random"
            for _ in range(16):
                pair = _make_pair(rng, side, mode, roles, style, growth_weight)
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not generate a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "exact train/test keys required")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    try:
        _counts(len(puzzle["train"]), len(puzzle["test"]))
    except ValueError as error:
        raise AssertionError(str(error)) from error
    seen, evidence, analyses = set(), [], {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair needs exact input/output keys")
            try:
                inp, out = _grid(pair["input"]), _grid(pair["output"])
                a = _analyze(inp)
            except ValueError as error:
                raise AssertionError(str(error)) from error
            signature = _key(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            _require(7 <= a["side"] <= 12, "generated side out of range")
            _require(out.shape == (a["side"], a["side"]), "output area must follow all four footprints")
            _require(np.array_equal(a["output"], out), "wrong simultaneous seam-mediated output")
            _require(np.any(a["colors"] != a["background"]), "generated examples must have a successful source")
            _require(not np.array_equal(inp, out), "input cannot already be output")
            _require(np.array_equal(a["seams"], a["seams"].T) and not np.diag(a["seams"]).any(), "seam graph invalid")
            for i, winners in enumerate(a["winners"]):
                _require(all(j != i and a["seams"][i, j] > 0 and
                             a["pieces"][j]["body"] == a["pieces"][i]["badge"] for j in winners), "ineligible donor")
            analyses[split].append(a)
            if split == "train":
                evidence.append(_evidence(a))
    for name in ("nonneighbor", "competition", "simultaneous", "erased_donor", "input_order_wrong",
                 "input_reverse_wrong", "priority_wrong", "own_badge_wrong", "own_body_wrong"):
        _require(any(f[name] for f in evidence), "missing training contrast: " + name)
    tr_sides = [a["side"] for a in analyses["train"]]
    te_sides = [a["side"] for a in analyses["test"]]
    _require(max(tr_sides) < min(te_sides) or max(te_sides) < min(tr_sides), "missing size shift")
    # Training must show all rotations on genuinely asymmetric non-anchor pieces.
    _require(any({a["moves"][i][0] for i in (1, 2, 3)} == {1, 2, 3} and
                 all(len({_rotate(p["shape"], k) for k in range(4)}) == 4 for p in a["pieces"])
                 for a in analyses["train"]), "quarter turns lack an asymmetric training witness")
    roles = {(a["background"], a["marker"]) for a in analyses["train"]}
    if len(roles) == 1:
        _require({(a["background"], a["marker"]) for a in analyses["test"]} <= roles,
                 "new palette roles need varying training roles")


def measure_pair(pair):
    """Return compact measurements recovered solely from visible input pixels."""
    a = _analyze(pair["input"])
    owner, pieces = a["owner"], a["pieces"]
    corner_ids = [int(owner[p]) for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    recode = np.empty(4, dtype=np.int8)
    for corner, piece in enumerate(corner_ids):
        recode[piece] = corner
    normalized_owner = recode[owner]
    areas = tuple(len(pieces[p]["shape"]) for p in corner_ids)
    fills = [len(p["shape"]) / math.prod(_dims(p["shape"])) for p in pieces]
    edges = tuple(int(a["seams"][corner_ids[i], corner_ids[j]])
                  for i in range(4) for j in range(i + 1, 4))
    return {"side": a["side"], "canvas": tuple(np.asarray(pair["input"]).shape),
            "background": a["background"], "marker": a["marker"], "areas": areas,
            "area_ratio": max(areas) / min(areas), "min_fill": min(fills),
            "contact_edges": sum(n > 0 for n in edges), "seam_lengths": edges,
            "bodies": len({p["body"] for p in pieces}),
            "successes": int(np.count_nonzero(a["colors"] != a["background"])),
            "rotations": tuple(move[0] for move in a["moves"]),
            "layout": (normalized_owner.shape, normalized_owner.tobytes()), **_evidence(a)}


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed structural, semantic, diversity, and argument checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seeds collapsed")
    _require(_puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0)), "seed determinism failed")
    records = [measure_pair(p) for puzzle in puzzles for s in ("train", "test") for p in puzzle[s]]
    _require(len({r["layout"] for r in records}) >= 120, "packed geometry lacks variety")
    _require(len({r["canvas"] for r in records}) >= 45, "scatter dimensions lack variety")
    _require({r["side"] for r in records} == set(range(7, 13)), "assembly sizes missing")
    _require(len({r["areas"] for r in records}) >= 120, "relative areas lack variety")
    _require(len({r["seam_lengths"] for r in records}) >= 120, "seam geometry lacks variety")
    _require({r["background"] for r in records} == set(range(10)), "background roles fixed")
    _require({r["marker"] for r in records} == set(range(10)), "marker roles fixed")
    _require({r["bodies"] for r in records} >= {2, 3, 4}, "source count lacks variety")
    _require({r["successes"] for r in records} >= {1, 2, 3, 4}, "success count lacks variety")
    _require(max(r["area_ratio"] for r in records) >= 4, "unequal silhouettes missing")
    _require(min(r["min_fill"] for r in records) <= .65, "irregular silhouettes missing")
    _require(len({r["contact_edges"] for r in records}) >= 2, "contact graphs fixed")
    _require(sum(r["competition"] > 0 and r["simultaneous"] > 0 for r in records) >= 5,
             "concepts do not combine outside separate witnesses")
    # Equal maximal seams are admissible only when their original values agree.
    tied_seams = np.zeros((4, 4), dtype=int)
    tied_seams[0, 1] = tied_seams[1, 0] = 2
    tied_seams[0, 2] = tied_seams[2, 0] = 2
    colors, winners = _result_colors([1, 2, 2, 3], [2, 3, 3, 4], tied_seams, 0)
    _require(colors[0] == 3 and winners[0] == (1, 2), "consistent longest-seam tie failed")
    try:
        _result_colors([1, 2, 2, 3], [2, 3, 4, 4], tied_seams, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting maximal tie accepted")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (4, 4), (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ("train", "test")}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal explicit counts)")
