"""Shattered stained glass: jointly assemble tiles and match enclosed rooms.

Standard library + NumPy only. Generation paints from a sampled construction;
the input-only solver enumerates visible assemblies and room correspondences.
No files, parent modules, generation state, or seed are consulted by solve().
"""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import math
import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0,985973819,829352081,592885733,911215357,894271013,221462073,189437679,846234993,914679798,868841412,818354527,42341095,617161520,511603562,191801673,458987939,681819530,426403571,933371620,614712281,451167888,819584098,466049439,247025490,154245312,670717259,846582507,740256744,296121833,459771368,113866216,884892076,879196527,733284830,520839513,535473296,476663124,328930495,199129379,509168150,702591447)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not _integer(num_test):
        raise ValueError("example counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must have integer cells")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _components(mask):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while cells:
        start = min(cells)
        cells.remove(start)
        stack, part = [start], []
        while stack:
            r, c = stack.pop()
            part.append((r, c))
            for dr, dc in _STEPS:
                q = (r + dr, c + dc)
                if q in cells:
                    cells.remove(q)
                    stack.append(q)
        result.append(np.asarray(sorted(part), dtype=int))
    return result


def _bbox(points):
    return (*map(int, points.min(0)), *map(int, points.max(0) + 1))


def _border(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _relations(rooms):
    boxes = [_bbox(p) for p in rooms]
    return tuple(tuple(tuple(-1 if a[k+2] <= b[k] else 1 if b[k+2] <= a[k] else 0
                             for k in (0, 1)) for b in boxes) for a in boxes)


def _mappings(a, b):
    """All exact orientation-preserving spatial-graph isomorphisms.

    Uses only bounding-interval relations, never area, color, discovery order,
    hidden coarse coordinates, or the sampled construction's room identities.
    """
    n = len(a)
    if n != len(b):
        return []
    signatures_a = [Counter(row) for row in a]
    signatures_b = [Counter(row) for row in b]
    options = [[j for j in range(n) if signatures_a[i] == signatures_b[j]] for i in range(n)]
    result = []

    def visit(mapping, used):
        if len(mapping) == n:
            result.append(tuple(mapping[i] for i in range(n)))
            return
        remaining = {i: [j for j in options[i] if j not in used and all(
            a[i][k] == b[j][v] and a[k][i] == b[v][j] for k, v in mapping.items())]
                     for i in range(n) if i not in mapping}
        i = min(remaining, key=lambda q: (len(remaining[q]), q))
        for j in remaining[i]:
            mapping[i] = j
            visit(mapping, used | {j})
            del mapping[i]

    visit({}, set())
    return result


def _parse(inp):
    border = _border(inp)
    bg = int(np.bincount(border, minlength=10).argmax())
    _require(np.count_nonzero(border == bg) > len(border) // 2, "background must dominate border")
    parts = _components(inp != bg)
    _require(len(parts) == 5, "expected one model and four tiles")
    objects = []
    for points in parts:
        r, c, R, C = _bbox(points)
        crop = np.full((R-r, C-c), -1, dtype=np.int8)
        local = points - (r, c)
        crop[tuple(local.T)] = inp[tuple(points.T)]
        objects.append((crop, points, (r, c, R, C)))
    donors = [o for o in objects if len(np.unique(o[0][o[0] >= 0])) >= 4]
    _require(len(donors) == 1, "one multicolored floorplan required")
    donor, donor_points, donor_box = donors[0]
    _require(bool(np.all(donor >= 0)), "model must be a complete rectangle")
    wall = int(donor[0, 0])
    _require(bool(np.all(_border(donor) == wall)), "model needs a closed outer frame")
    rooms = _components(donor != wall)
    _require(3 <= len(rooms) <= 6, "expected 3..6 model rooms")
    colors = []
    for room in rooms:
        values = np.unique(donor[tuple(room.T)])
        _require(len(values) == 1 and int(values[0]) != bg, "model rooms must be uniformly colored")
        colors.append(int(values[0]))
    _require(len(set(colors)) >= 3, "model needs at least three fill colors")
    tiles = [o for o in objects if o is not donors[0]]
    values = Counter(int(v) for tile, _, _ in tiles for v in tile[tile >= 0])
    _require(wall in values, "model and tiles must share wall color")
    candidates = [c for c, n in values.items() if c != wall and n == 1]
    _require(len(candidates) == 1, "one singleton anchor pin required")
    pin = candidates[0]
    glass_options = set(values) - {wall, pin}
    _require(len(glass_options) == 1, "one common blank-glass color required")
    glass = glass_options.pop()
    _require(len({bg, wall, glass, pin}) == 4 and not set(colors) & {bg, wall, glass, pin}, "palette roles overlap")
    anchors = [o for o in tiles if pin in o[0]]
    _require(len(anchors) == 1 and anchors[0][0][0, 0] == pin, "pin must mark the unturned upper-left tile corner")
    tiles = anchors + [o for o in tiles if o is not anchors[0]]
    for tile, _, _ in tiles:
        _require(wall in tile and glass in tile, "every tile must carry both wall and glass")
        _require(set(np.unique(tile)) <= {-1, wall, glass, pin}, "unexpected tile content")
    for a, b in itertools.combinations(parts, 2):
        _require(int(np.max(np.abs(a[:, None] - b[None, :]), axis=2).min()) >= 2,
                 "objects must have a background-cell separation")
    total = sum(int(np.count_nonzero(t[0] >= 0)) for t in tiles)
    side = math.isqrt(total)
    _require(side * side == total and 1 <= side <= 30, "tile areas must sum to a square")
    return {"bg": bg, "wall": wall, "glass": glass, "pin": pin,
            "donor": donor, "donor_box": donor_box, "rooms": rooms, "colors": colors,
            "relations": _relations(rooms), "tiles": tiles, "side": side}


def _bits(points, side, offset=(0, 0)):
    return sum(1 << ((int(r) + offset[0]) * side + int(c) + offset[1]) for r, c in points)


def _placements(tile, side, corner, wall, pin, anchor=False):
    corners = (0, side-1, side*(side-1), side*side-1)
    result, seen = [], set()
    for k in ([0] if anchor else range(4)):
        view = np.rot90(tile, k)
        h, w = view.shape
        if h > side or w > side:
            continue
        offset = (0 if corner < 2 else side-h, 0 if corner % 2 == 0 else side-w)
        points = np.argwhere(view >= 0)
        mask = _bits(points, side, offset)
        if [i for i, index in enumerate(corners) if mask & (1 << index)] != [corner]:
            continue
        walls = _bits(np.argwhere((view == wall) | (view == pin)), side, offset)
        signature = mask, walls
        if signature in seen:
            continue
        seen.add(signature)
        result.append((mask, walls, k, offset))
    return result


def _assemblies(info):
    """Enumerate full geometric square packings; keep internal wall rotations.

    Unlike monochrome packing, two rotations with the same footprint may
    contain different walls and must remain different candidates.
    """
    n, wall, pin = info["side"], info["wall"], info["pin"]
    tiles = [t[0] for t in info["tiles"]]
    anchors = _placements(tiles[0], n, 0, wall, pin, True)
    if not anchors:
        return
    table = {(i, c): _placements(tiles[i], n, c, wall, pin) for i in range(1, 4) for c in range(1, 4)}
    full = (1 << (n*n)) - 1
    anchor = anchors[0]

    def search(corner, remaining, occupied, walls, placed):
        if corner == 4:
            if occupied == full:
                yield walls, placed
            return
        for i in remaining:
            for p in table[i, corner]:
                if not occupied & p[0]:
                    yield from search(corner+1, tuple(j for j in remaining if j != i),
                                      occupied | p[0], walls | p[1], placed + ((i, p),))

    yield from search(1, (1, 2, 3), anchor[0], anchor[1], ((0, anchor),))


def _analyze(inp):
    info = _parse(inp)
    n = info["side"]
    answers, wall_patterns, room_panels = {}, set(), set()
    best_placements, even_turn_panels, unturned_panels = {}, set(), set()
    geometry_count = 0
    for wall_bits, placed in _assemblies(info):
        geometry_count += 1
        turns = tuple(p[2] for _, p in placed)
        score = (sum(k != 0 for k in turns), sum(turns), turns)
        if wall_bits not in best_placements or score < best_placements[wall_bits][0]:
            best_placements[wall_bits] = score, placed
        if not any(turns):
            unturned_panels.add(wall_bits)
        if not any(k % 2 for k in turns):
            even_turn_panels.add(wall_bits)
        if wall_bits in wall_patterns:
            continue
        wall_patterns.add(wall_bits)
        wall_mask = np.fromiter(((wall_bits >> i) & 1 for i in range(n*n)), dtype=np.int8).reshape(n, n).astype(bool)
        if not np.all(_border(wall_mask)):
            continue
        rooms = _components(~wall_mask)
        if len(rooms) != len(info["rooms"]):
            continue
        room_panels.add(wall_bits)
        mappings = _mappings(info["relations"], _relations(rooms))
        for mapping in mappings:
            answer = np.full((n, n), info["wall"], dtype=np.int8)
            for i, j in enumerate(mapping):
                answer[tuple(rooms[j].T)] = info["colors"][i]
            signature = _key(answer)
            answers.setdefault(signature, (answer, wall_mask, rooms, mapping, wall_bits))
            _require(len(answers) <= 1, "more than one colored restoration fits the visible model")
    _require(len(answers) == 1, "no restoration matches model rooms")
    answer, wall_mask, rooms, mapping, wall_bits = next(iter(answers.values()))
    placed = best_placements[wall_bits][1]
    info.update(answer=answer, wall_mask=wall_mask, target_rooms=rooms, mapping=mapping,
                placed=placed, geometry_count=geometry_count, room_panel_count=len(room_panels),
                rejected_room_panels=len(room_panels)-1,
                rotation_required=wall_bits not in unturned_panels,
                quarter_turn_required=wall_bits not in even_turn_panels)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Restore and color the square using only this input's visible pixels."""
    return _analyze(_grid(input_grid))["answer"].copy()


def _plan(rng, count):
    options = [(2, 2), (2, 3), (3, 2), (3, 3)]
    options = [s for s in options if s[0] * s[1] >= count]
    rows, cols = options[int(rng.integers(len(options)))]
    labels = np.arange(rows*cols).reshape(rows, cols)
    while len(np.unique(labels)) > count:
        edges = [(int(labels[r, c]), int(labels[r+dr, c+dc]))
                 for r in range(rows) for c in range(cols) for dr, dc in ((0, 1), (1, 0))
                 if r+dr < rows and c+dc < cols and labels[r, c] != labels[r+dr, c+dc]]
        a, b = edges[int(rng.integers(len(edges)))]
        labels[labels == b] = a
    relabel = {int(old): i for i, old in enumerate(np.unique(labels))}
    return np.array([[relabel[int(v)] for v in row] for row in labels], dtype=np.int8)


def _lengths(rng, count, total):
    lengths = np.ones(count, dtype=int)
    for _ in range(total-count):
        lengths[int(rng.integers(count))] += 1
    return lengths


def _render(plan, heights, widths):
    """Render a connected room partition with one-cell separating walls."""
    ys, xs = np.r_[0, np.cumsum(heights+1)], np.r_[0, np.cumsum(widths+1)]
    labels = np.full((int(ys[-1])+1, int(xs[-1])+1), -1, dtype=np.int8)
    for r, c in np.ndindex(plan.shape):
        label = plan[r, c]
        labels[ys[r]+1:ys[r+1], xs[c]+1:xs[c+1]] = label
        if r and plan[r-1, c] == label:
            labels[ys[r], xs[c]+1:xs[c+1]] = label
        if c and plan[r, c-1] == label:
            labels[ys[r]+1:ys[r+1], xs[c]] = label
        if r and c and np.all(plan[r-1:r+1, c-1:c+1] == label):
            labels[ys[r], xs[c]] = label
    return labels


def _paint(labels, wall, colors):
    result = np.full(labels.shape, wall, dtype=np.int8)
    for i, color in enumerate(colors):
        result[labels == i] = color
    return result


def _detail(rng, labels, count):
    """Local room shrinkage preserving the complete spatial graph.

    Trimming the top of only one side room can change room discovery order
    without reversing any meaningful above/below or left/right relation.
    """
    labels = labels.copy()
    reference = _relations([np.argwhere(labels == i) for i in range(count)])
    for i in rng.permutation(count):
        i = int(i)
        if rng.random() < 0.25:
            continue
        points = np.argwhere(labels == i)
        r, c, R, C = _bbox(points)
        style = int(rng.integers(4))
        if style == 0:
            removed = points[points[:, 0] == r]
        elif style == 1:
            removed = points[points[:, 1] == c]
        else:
            q = (r, c) if style == 2 else (R-1, C-1)
            removed = np.asarray([q], dtype=int) if labels[q] == i else np.empty((0, 2), dtype=int)
        if len(removed) == 0 or len(removed) == len(points):
            continue
        labels[tuple(removed.T)] = -1
        rooms = [np.argwhere(labels == j) for j in range(count)]
        if len(_components(labels == i)) != 1 or _relations(rooms) != reference:
            labels[tuple(removed.T)] = i
    return labels


def _cut(rng, side, rough):
    y, x = int(rng.integers(3, side-2)), int(rng.integers(3, side-2))
    owner = np.empty((side, side), dtype=np.int8)
    owner[:y, :x], owner[:y, x:], owner[y:, :x], owner[y:, x:] = 0, 1, 2, 3
    corners = {(0, 0), (0, side-1), (side-1, 0), (side-1, side-1)}
    # Local boundary growth gives tabs, recesses, steps, and unequal areas.
    for _ in range(rough):
        boundary = [(r, c, int(owner[r+dr, c+dc])) for r in range(1, side-1)
                    for c in range(1, side-1) for dr, dc in _STEPS
                    if owner[r, c] != owner[r+dr, c+dc]]
        if not boundary:
            break
        r, c, new = boundary[int(rng.integers(len(boundary)))]
        if (r, c) in corners:
            continue
        old = int(owner[r, c])
        owner[r, c] = new
        valid = True
        for label in (old, new):
            pts = np.argwhere(owner == label)
            if len(pts) < 5 or len(_components(owner == label)) != 1:
                valid = False
                break
            rr, cc, R, C = _bbox(pts)
            if max(R-rr, C-cc) > 9:
                valid = False
                break
        if not valid:
            owner[r, c] = old
    return owner


def _scatter(rng, objects, bg):
    """Random shelf partitions, shuffled independently of output tile corner."""
    choices = []
    for _ in range(96):
        order = list(map(int, rng.permutation(5)))
        cuts = int(rng.integers(1, 16))
        transpose = bool(rng.integers(2))
        gap = int(rng.integers(1, 4))
        dims = [tuple(reversed(o.shape)) if transpose else o.shape for o in objects]
        groups, current = [], []
        for i, item in enumerate(order):
            current.append(item)
            if i == 4 or (cuts >> i) & 1:
                groups.append(current)
                current = []
        widths = [sum(dims[j][1] for j in g)+gap*(len(g)-1) for g in groups]
        heights = [max(dims[j][0] for j in g) for g in groups]
        h, w = sum(heights)+gap*(len(groups)-1), max(widths)
        if h <= 28 and w <= 28:
            choices.append((transpose, gap, groups, dims, heights, widths, h, w))
    if not choices:
        # The four tile boxes are <=9x9, the model <=10x10. Three tiles
        # on one shelf, model+remaining tile on another always fit <=29x20.
        choices = [(False, 1, [[0, 1], [2, 3, 4]], [o.shape for o in objects],
                    [max(objects[0].shape[0], objects[1].shape[0]),
                     max(o.shape[0] for o in objects[2:])],
                    [objects[0].shape[1]+objects[1].shape[1]+1,
                     sum(o.shape[1] for o in objects[2:])+2], 0, 0)]
        entry = choices[0]
        choices[0] = entry[:-2] + (sum(entry[4])+1, max(entry[5]))
    transpose, gap, groups, dims, heights, widths, h, w = choices[int(rng.integers(len(choices)))]
    _require(h <= 30 and w <= 30, "scatter fallback exceeded canvas")
    offsets, row = {}, 0
    for group, height, width in zip(groups, heights, widths):
        col = int(rng.integers(w-width+1))
        for j in group:
            rr = row + int(rng.integers(height-dims[j][0]+1))
            offsets[j] = (col, rr) if transpose else (rr, col)
            col += dims[j][1]+gap
        row += height+gap
    h, w = (w, h) if transpose else (h, w)
    pads = []
    for length in (h, w):
        extra = int(rng.integers(min(6, 30-length)+1))
        before = int(rng.integers(extra+1))
        pads.append((before, extra-before))
    canvas = np.full((h+sum(pads[0]), w+sum(pads[1])), bg, dtype=np.int8)
    for j, obj in enumerate(objects):
        rr, cc = offsets[j]
        rr, cc = rr+pads[0][0], cc+pads[1][0]
        region = canvas[rr:rr+obj.shape[0], cc:cc+obj.shape[1]]
        region[obj >= 0] = obj[obj >= 0]
    return canvas


def _measure(info):
    owners = np.full((info["side"], info["side"]), -1, dtype=np.int8)
    for i, p in info["placed"]:
        mask = p[0]
        for j in range(owners.size):
            if mask & (1 << j):
                owners.flat[j] = i
    rooms, mapping = info["target_rooms"], info["mapping"]
    crossed = sum(len(np.unique(owners[tuple(room.T)])) > 1 for room in rooms)
    mixed = sum(len(np.unique(info["answer"][(owners == i) & ~info["wall_mask"]])) > 1 for i in range(4))
    donor_sizes = [len(r) for r in info["rooms"]]
    target_sizes = [len(rooms[mapping[i]]) for i in range(len(rooms))]
    signs = {int(np.sign(b-a)) for a, b in zip(donor_sizes, target_sizes)}
    reversal = any(donor_sizes[i] < donor_sizes[j] and target_sizes[i] > target_sizes[j]
                   and info["colors"][i] != info["colors"][j]
                   for i in range(len(rooms)) for j in range(len(rooms)))
    seam_excess = int(np.count_nonzero(owners[1:] != owners[:-1])
                      + np.count_nonzero(owners[:, 1:] != owners[:, :-1]) - 2*info["side"])
    irregular = sum(np.count_nonzero(t[0] >= 0) < t[0].size for t in info["tiles"])
    return {"rooms": len(rooms), "side": info["side"], "donor_shape": list(info["donor"].shape),
            "background": info["bg"], "wall": info["wall"], "glass": info["glass"], "pin": info["pin"],
            "fill_colors": len(set(info["colors"])), "reused_color": len(set(info["colors"])) < len(rooms),
            "crossed_rooms": int(crossed), "mixed_tiles": int(mixed), "irregular_tiles": int(irregular),
            "geometry_candidates": info["geometry_count"], "room_panels": info["room_panel_count"],
            "rejected_room_panels": info["rejected_room_panels"], "area_change_signs": sorted(signs),
            "area_rank_reversal": reversal, "scan_order_changed": any(i != j for i, j in enumerate(mapping)),
            "seam_excess": seam_excess, "rotation_required": info["rotation_required"],
            "quarter_turn_required": info["quarter_turn_required"],
            "rotations": [int(p[2]) for _, p in info["placed"]],
            "nonrectangular_rooms": sum(len(r) != (lambda b: (b[2]-b[0])*(b[3]-b[1]))(_bbox(r)) for r in rooms)}


def _make_pair(rng, count, side, rough, require_topology, rotations, excluded_roles=None,
               require_analogy=False, min_seam_excess=None):
    for attempt in range(480):
        plan = _plan(rng, count)
        donor_labels = _render(plan, rng.integers(1, 3, size=plan.shape[0]), rng.integers(1, 3, size=plan.shape[1]))
        target_labels = _render(plan, _lengths(rng, plan.shape[0], side-plan.shape[0]-1),
                                _lengths(rng, plan.shape[1], side-plan.shape[1]-1))
        donor_labels = _detail(rng, donor_labels, count)
        target_labels = _detail(rng, target_labels, count)
        if donor_labels.shape == target_labels.shape and np.array_equal(donor_labels, target_labels):
            continue
        owner = _cut(rng, side, rough)
        palette = list(map(int, rng.permutation(10)))
        bg, wall, glass, pin = palette[:4]
        if excluded_roles is not None and (bg == excluded_roles[0] or wall == excluded_roles[1]):
            continue
        richness = int(rng.integers(3, min(count, 6)+1))
        colors = palette[4:4+richness] + [palette[4+int(rng.integers(richness))] for _ in range(count-richness)]
        rng.shuffle(colors)
        donor = _paint(donor_labels, wall, colors)
        expected = _paint(target_labels, wall, colors)
        blank = np.where(target_labels < 0, wall, glass).astype(np.int8)
        blank[0, 0] = pin
        pieces = []
        for i in range(4):
            pts = np.argwhere(owner == i)
            r, c, R, C = _bbox(pts)
            tile = np.where(owner[r:R, c:C] == i, blank[r:R, c:C], -1).astype(np.int8)
            pieces.append(np.rot90(tile, rotations[i]).copy())
        if any(max(p.shape) > 9 or wall not in p or glass not in p for p in pieces):
            continue
        inp = _scatter(rng, [donor]+pieces, bg)
        try:
            info = _analyze(inp)
            if not np.array_equal(info["answer"], expected):
                continue
            metrics = _measure(info)
            if metrics["crossed_rooms"] < 1 or metrics["mixed_tiles"] < 1:
                continue
            if require_topology and metrics["rejected_room_panels"] < 1:
                continue
            if require_topology and not metrics["quarter_turn_required"]:
                continue
            if require_analogy and not (metrics["scan_order_changed"] and metrics["area_rank_reversal"]):
                continue
            if min_seam_excess is not None and metrics["seam_excess"] < min_seam_excess:
                continue
            return {"input": inp, "output": expected}
        except AssertionError:
            continue
    raise RuntimeError("bounded construction exhausted; no ambiguous restoration was accepted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(3))  # larger square / more rooms / more interlocking cuts
    puzzle, seen, first_roles = {"train": [], "test": []}, set(), None
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            if split == "train" and index == 0:
                count, rough = 3, 0
                rotations = [0, 1, 2, 3]
            elif split == "train" and index == 1:
                count, rough = 5, 3 if shift == 2 else int(rng.integers(3, 9))
                rotations = [0, 0, 2, 1]
            else:
                count = 6 if split == "test" and shift == 1 else int(rng.integers(3, 6))
                rough = int(rng.integers(24, 41)) if split == "test" and shift == 2 else int(rng.integers(0, 4 if shift == 2 else 10))
                rotations = [0] + list(map(int, rng.integers(0, 4, size=3)))
            side = 12 if split == "test" and shift == 0 else int(rng.integers(8, 12))
            for _ in range(12):
                pair = _make_pair(rng, count, side, rough, split == "train" and index == 0,
                                  rotations, first_roles if split == "train" and index == 1 else None,
                                  require_analogy=split == "train" and index == 1,
                                  min_seam_excess=8 if split == "test" and shift == 2 else None)
                signature = _key(pair["input"])
                if signature not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness retries exhausted")
            seen.add(signature)
            puzzle[split].append(pair)
            if split == "train" and index == 0:
                info = _parse(pair["input"])
                first_roles = info["bg"], info["wall"]
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    try:
        _counts(len(puzzle["train"]), len(puzzle["test"]))
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _analyze(inp)
            _require(np.array_equal(info["answer"], out), "incorrect colored restoration")
            _require(out.shape == (info["side"], info["side"]), "square area mismatch")
            _require(np.array_equal(out == info["wall"], info["wall_mask"]), "walls changed")
            _require(int(out[0, 0]) == info["wall"], "pin must become wall")
            _require(not set(np.unique(out)) & {info["bg"], info["glass"], info["pin"]}, "temporary colors leaked")
            m = _measure(info)
            _require(m["crossed_rooms"] >= 1 and m["mixed_tiles"] >= 1, "tile and room partitions must interact")
            _require(info["donor"].shape != out.shape or not np.array_equal(info["donor"], out), "model copied unchanged")
            analyses[split].append(m)
    training = analyses["train"]
    _require(training[0]["rejected_room_panels"] >= 1, "demonstrate topology resolving square assembly")
    _require(training[0]["quarter_turn_required"], "demonstrate a necessary quarter turn")
    _require(training[1]["scan_order_changed"] and training[1]["area_rank_reversal"],
             "demonstrate correspondence beyond area and scan order")
    _require(len({m["rooms"] for m in training}) >= 2, "demonstrate different room structures")
    _require(len({m["background"] for m in training}) >= 2 and len({m["wall"] for m in training}) >= 2,
             "demonstrate local palette roles")


def self_test():
    """42 host seeds, bounded in-memory metrics, counts and corruption checks."""
    shapes, structures, roles, room_counts, sides, rotations = set(), set(), set(), set(), set(), set()
    totals = Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                m = _measure(info)
                shapes.add(pair["input"].shape)
                structures.add(hashlib.sha256(info["wall_mask"].tobytes()+bytes(info["wall_mask"].shape)).hexdigest())
                roles.add((info["bg"], info["wall"], info["glass"], info["pin"]))
                room_counts.add(m["rooms"])
                sides.add(m["side"])
                rotations.update(m["rotations"])
                totals["pairs"] += 1
                for name in ("crossed_rooms", "mixed_tiles", "irregular_tiles", "rejected_room_panels", "nonrectangular_rooms",
                             "scan_order_changed", "area_rank_reversal", "quarter_turn_required"):
                    totals[name] += m[name] > 0
                totals["reused_color"] += m["reused_color"]
                totals["mixed_area_change"] += -1 in m["area_change_signs"] and 1 in m["area_change_signs"]
    _require(room_counts == {3, 4, 5, 6} and sides == set(range(8, 13)), "room/scale variation collapsed")
    _require(rotations == {0, 1, 2, 3}, "rotation spectrum missing")
    _require(len(shapes) >= 35 and len(structures) >= 100 and len(roles) >= 100, "structural/palette diversity collapsed")
    _require(totals["rejected_room_panels"] >= 42, "model must participate in assembly")
    for feature in ("irregular_tiles", "nonrectangular_rooms", "reused_color", "mixed_area_change"):
        _require(totals[feature] >= 10, "missing variation: " + feature)
    _require(totals["scan_order_changed"] >= 42 and totals["area_rank_reversal"] >= 42,
             "spatial analogy counterexamples missing")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(0, a, b))
    for a, b in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (4,1.0), (None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first, again = generate_puzzle(0), generate_puzzle(0)
    _require(all(_key(p[k]) == _key(q[k]) for s in first for p, q in zip(first[s], again[s])
                 for k in ("input", "output")), "nondeterministic puzzle")
    sample = first["train"][0]
    info = _analyze(sample["input"])
    # Color equivariance: literal digits do not determine semantic roles.
    palette = np.array([7, 4, 9, 0, 8, 2, 1, 6, 3, 5], dtype=np.int8)
    _require(np.array_equal(solve(palette[sample["input"]]), palette[sample["output"]]),
             "palette permutation changed the rule")
    # Moving and turning the three free tiles changes no reconstructed room.
    moved = [info["donor"].copy(), info["tiles"][0][0].copy()]
    moved.extend(np.rot90(t[0], i+1).copy() for i, t in enumerate(info["tiles"][1:]))
    relocated = _scatter(np.random.default_rng(1701), moved, info["bg"])
    _require(np.array_equal(solve(relocated), sample["output"]), "tile presentation leaked into answer")
    # A donor-only payload change must propagate through the assembled room.
    changed = sample["input"].copy()
    old = info["colors"][0]
    new = next(c for c in range(10) if c not in set(changed.flat))
    r, c, _, _ = info["donor_box"]
    points = info["rooms"][0] + (r, c)
    changed[tuple(points.T)] = new
    expected = np.where(sample["output"] == old, new, sample["output"])
    _require(np.array_equal(solve(changed), expected), "donor color did not control the restored room")
    # Both an interior color corruption and a wall corruption must be caught.
    for wall_change in (False, True):
        damaged = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
        pair = damaged["test"][0]
        info = _analyze(pair["input"])
        point = tuple(np.argwhere(info["wall_mask"] == wall_change)[0])
        pair["output"][point] = info["glass"]
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupted answer accepted")
    return {"seeds": len(HOST_SEEDS), "pairs": totals["pairs"], "input_shapes": len(shapes),
            "wall_structures": len(structures), "role_palettes": len(roles), "features": dict(totals)}
