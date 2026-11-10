"""Cavity-keyed contour transplants. Standard library and NumPy only.

Label convention is fixed: convex=4, concave=2. Neither occurs in inputs.
No files, generator state, seed, or demonstration pairs are used by solve().
"""
from __future__ import annotations

import hashlib
import json
from collections import deque

import numpy as np


CONVEX, CONCAVE = 4, 2
COLORS = (0, 1, 3, 5, 6, 7, 8, 9)
STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 16464851, 742201583, 71062150, 689253095, 827960147,
              92987162, 759381464, 577444114, 81875623, 478419930,
              872426580, 832643052, 404187723, 268235278, 976047727,
              917832028, 341478964, 753046999, 138309112, 226651626,
              695916845, 885426693, 187844692, 115168203, 105452290,
              470017756, 564387322, 184491360, 46805849, 942334711,
              911447971, 759951907, 490208376, 780754837, 364370331,
              374802399, 390478242, 88951211, 763107331, 430694596,
              788258480)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in STEPS:
                cell = r + dr, c + dc
                if cell in remaining:
                    remaining.remove(cell)
                    cells.add(cell)
                    stack.append(cell)
        result.append(cells)
    return result


def _box(cells):
    rr, cc = zip(*cells)
    return min(rr), min(cc), max(rr) + 1, max(cc) + 1


def _inside(mask):
    """Bounded cells using a padded four-neighbor exterior flood fill."""
    blocked = np.pad(mask, 1)
    exterior = np.zeros_like(blocked)
    exterior[0, 0] = True
    queue = deque([(0, 0)])
    h, w = blocked.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in STEPS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < h and 0 <= nc < w and not blocked[nr, nc]
                    and not exterior[nr, nc]):
                exterior[nr, nc] = True
                queue.append((nr, nc))
    return ((~blocked) & (~exterior))[1:-1, 1:-1]


def _loop_info(cells, shape):
    def neighbors(p):
        r, c = p
        return sorted((r + dr, c + dc) for dr, dc in STEPS
                      if (r + dr, c + dc) in cells)
    assert len(cells) >= 8
    assert all(len(neighbors(p)) == 2 for p in cells), "contour must be a simple cycle"
    start = min(cells)
    path, seen = [start], {start}
    previous, current = start, neighbors(start)[0]
    while current != start:
        assert current not in seen
        path.append(current)
        seen.add(current)
        previous, current = current, next(p for p in neighbors(current) if p != previous)
    assert len(path) == len(cells)
    area = sum(c * path[(i + 1) % len(path)][0]
               - path[(i + 1) % len(path)][1] * r
               for i, (r, c) in enumerate(path))
    assert area
    kinds = {}
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % len(path)]
        turn = (c - pc) * (nr - r) - (r - pr) * (nc - c)
        kinds[r, c] = 0 if turn == 0 else 1 if turn * area > 0 else 2
    assert sum(v == 1 for v in kinds.values()) - sum(v == 2 for v in kinds.values()) == 4
    mask = np.zeros(shape, dtype=bool)
    for p in cells:
        mask[p] = True
    interior = _inside(mask)
    assert interior.any() and len(_components(interior)) == 1
    return {"cells": cells, "path": path, "kinds": kinds, "inside": interior}


def _polygon(vertices):
    cells = []
    for (r, c), (nr, nc) in zip(vertices, vertices[1:] + vertices[:1]):
        dr, dc = nr - r, nc - c
        if (dr == 0) == (dc == 0):
            return None
        length = abs(dr) + abs(dc)
        for j in range(length):
            cells.append((r + j * int(np.sign(dr)), c + j * int(np.sign(dc))))
    if len(set(cells)) != len(cells):
        return None
    r0, c0, r1, c1 = _box(cells)
    shape = r1 - r0, c1 - c0
    normalized = {(r - r0, c - c0) for r, c in cells}
    try:
        _loop_info(normalized, shape)
    except (AssertionError, StopIteration):
        return None
    mask = np.zeros(shape, dtype=bool)
    for p in normalized:
        mask[p] = True
    return mask


def _shape(rng, max_h, max_w, require_concave=False, rectangle=False):
    """Compose local corner cuts and edge detours, rejecting illegal contacts."""
    for _ in range(12):
        low = 5 if require_concave else 3
        h = int(rng.integers(low, max_h + 1))
        w = int(rng.integers(low, max_w + 1))
        vertices = [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)]
        best = _polygon(vertices)
        target = 0 if rectangle else int(rng.integers(int(require_concave), 5))
        for _ in range(35):
            info = _loop_info(_components(best)[0], best.shape)
            if sum(k == 2 for k in info["kinds"].values()) >= target:
                break
            i = int(rng.integers(len(vertices)))
            r, c = vertices[i]
            if rng.random() < .6:
                pr, pc = vertices[i - 1]
                nr, nc = vertices[(i + 1) % len(vertices)]
                a, b = abs(pr - r) + abs(pc - c), abs(nr - r) + abs(nc - c)
                if min(a, b) < 2:
                    continue
                x, y = int(rng.integers(1, a)), int(rng.integers(1, b))
                p = r + x * (pr - r) // a, c + x * (pc - c) // a
                q = r + y * (nr - r) // b, c + y * (nc - c) // b
                middle = p[0] + q[0] - r, p[1] + q[1] - c
                candidate = vertices[:i] + [p, middle, q] + vertices[i + 1:]
            else:
                nr, nc = vertices[(i + 1) % len(vertices)]
                length = abs(nr - r) + abs(nc - c)
                if length < 3:
                    continue
                a = int(rng.integers(1, length - 1))
                b = int(rng.integers(a + 1, length))
                dr, dc = (nr - r) // length, (nc - c) // length
                depth = int(rng.choice((-2, -1, 1, 2)))
                p = r + a * dr, c + a * dc
                q = r + b * dr, c + b * dc
                candidate = vertices[:i + 1] + [p, (p[0] + depth * dc, p[1] - depth * dr),
                                                (q[0] + depth * dc, q[1] - depth * dr), q] + vertices[i + 1:]
            mask = _polygon(candidate)
            if mask is not None and mask.shape[0] <= max_h and mask.shape[1] <= max_w:
                vertices, best = candidate, mask
        info = _loop_info(_components(best)[0], best.shape)
        if not require_concave or 2 in info["kinds"].values():
            # Restrict rotations for nonsquare slots; never clip or rescale.
            variants = [np.rot90(best, k) for k in range(4)]
            variants += [np.fliplr(v) for v in variants]
            return _pick(rng, [v for v in variants if v.shape[0] <= max_h and v.shape[1] <= max_w]).copy()
    # Constructive one-concavity L, with no touching nonconsecutive edges.
    assert require_concave and min(max_h, max_w) >= 5
    return _polygon([(0, 0), (0, 4), (4, 4), (4, 2), (2, 2), (2, 0)])


def _capacity(h, w):
    return ((h + 1) // 4) * ((w + 1) // 4)


def _slots(rng, size, count):
    boxes = [(0, 0, size, size)]
    while len(boxes) < count:
        choices = []
        for i, (r, c, h, w) in enumerate(boxes):
            for axis, length in ((0, h), (1, w)):
                for p in range(3, length - 3):
                    parts = ([(r, c, p, w), (r + p + 1, c, h - p - 1, w)] if axis == 0
                             else [(r, c, h, p), (r, c + p + 1, h, w - p - 1)])
                    trial = boxes[:i] + parts + boxes[i + 1:]
                    if sum(_capacity(b[2], b[3]) for b in trial) >= count:
                        choices.append(trial)
        assert choices, "constructive loop packing capacity violated"
        boxes = _pick(rng, choices)
    return boxes


def _make_motif(rng, size, count, bg, body, stroke, mode="random"):
    motif = np.full((size, size), body, dtype=np.int8)
    loops = []

    def draw(mask, r, c, fill):
        cells = {(r + int(y), c + int(x)) for y, x in np.argwhere(mask)}
        info = _loop_info(cells, motif.shape)
        motif[info["inside"]] = fill
        for p in cells:
            motif[p] = stroke
        loops.append(info)

    if count == 0:
        return motif, loops
    if count == 1 and (mode == "island" or (mode == "random" and size >= 7 and rng.random() < .30)):
        h, w = int(rng.integers(7, size + 1)), int(rng.integers(7, size + 1))
        outer = _polygon([(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)])
        r, c = int(rng.integers(size - h + 1)), int(rng.integers(size - w + 1))
        draw(outer, r, c, bg)
        island = _shape(rng, h - 4, w - 4)
        ir = r + int(rng.integers(2, h - island.shape[0] - 1))
        ic = c + int(rng.integers(2, w - island.shape[1] - 1))
        draw(island, ir, ic, body)
        return motif, loops
    for r, c, h, w in _slots(rng, size, count):
        if mode == "small":
            mask = _polygon([(0, 0), (0, 2), (2, 2), (2, 0)])
        else:
            # Leave some body visible for single non-island contours.
            if count == 1:
                h, w = min(h, size - 1), min(w, size - 1)
            required = mode == "concave" or (min(h, w) >= 5 and rng.random() < .40)
            mask = _shape(rng, h, w, required)
        rr = r + int(rng.integers(h - mask.shape[0] + 1))
        cc = c + int(rng.integers(w - mask.shape[1] + 1))
        draw(mask, rr, cc, bg)
    return motif, loops


def _parse_motif(motif, bg):
    colors = set(map(int, np.unique(motif))) - {bg}
    assert 1 <= len(colors) <= 2
    assert np.all(motif[0] != bg) and np.all(motif[-1] != bg)
    assert np.all(motif[:, 0] != bg) and np.all(motif[:, -1] != bg)
    cavities = _components(motif == bg)
    if not cavities:
        assert len(colors) == 1, "zero-cavity motif must be solid"
        return {"count": 0, "loops": [], "motif": motif.copy(), "areas": (), "islands": 0}
    interpretations = []
    for stroke in sorted(colors):
        try:
            loops = [_loop_info(cells, motif.shape) for cells in _components(motif == stroke)]
            depth = sum((info["inside"].astype(np.int8) for info in loops), np.zeros(motif.shape, dtype=np.int8))
            assert int(depth.max()) <= 2
            nonstroke = motif != stroke
            assert np.array_equal((depth % 2 == 1) & nonstroke, motif == bg)
            body_cells = nonstroke & (depth % 2 == 0)
            body_colors = set(map(int, np.unique(motif[body_cells])))
            assert len(body_colors) <= 1 and bg not in body_colors
            assert colors == {stroke} | body_colors
            islands = sum(any(other["inside"][min(info["cells"])]
                              for other in loops if other is not info) for info in loops)
            interpretations.append({"count": len(cavities), "loops": loops,
                                    "motif": motif.copy(), "areas": tuple(sorted(map(len, cavities))),
                                    "islands": islands})
        except (AssertionError, StopIteration):
            continue
    assert len(interpretations) == 1, "ambiguous contour/body interpretation"
    return interpretations[0]


def _separated(a, b):
    r, c, h, w = a
    rr, cc, hh, ww = b
    return r + h < rr or rr + hh < r or c + w < cc or cc + ww < c


def _decode_bg(grid, bg):
    components = _components(grid != bg)
    boxes = [_box(cells) for cells in components]
    # Filled islands are components inside a motif's complete square perimeter.
    top = [b for b in boxes if not any(b != a and a[0] <= b[0] and a[1] <= b[1]
                                     and b[2] <= a[2] and b[3] <= a[3] for a in boxes)]
    assert 3 <= len(top) <= 10
    top.sort(key=lambda b: (-(b[2] - b[0]) * (b[3] - b[1]), b))
    mr, mc, er, ec = top[0]
    motif_shapes = {(b[2] - b[0], b[3] - b[1]) for b in top[1:]}
    assert len(motif_shapes) == 1
    s, sw = next(iter(motif_shapes))
    assert s == sw and 7 <= s <= 13
    t = s + 2
    assert (er - mr) % t == (ec - mc) % t == 0
    rows, cols = (er - mr) // t, (ec - mc) // t
    assert rows * cols == len(top) - 1 and rows * cols >= 2
    rects = [(r, c, e - r, f - c) for r, c, e, f in top]
    assert all(_separated(a, b) for i, a in enumerate(rects) for b in rects[i + 1:])
    motifs = {}
    ordered = []
    for r, c, e, f in sorted(top[1:]):
        info = _parse_motif(grid[r:e, c:f], bg)
        k = info["count"]
        assert k not in motifs
        motifs[k] = info
        ordered.append(k)
    tiles = []
    for i in range(rows * cols):
        r, c = mr + (i // cols) * t, mc + (i % cols) * t
        tile = grid[r:r + t, c:c + t]
        carrier = int(tile[0, 0])
        assert carrier != bg
        assert np.all(tile[0] == carrier) and np.all(tile[-1] == carrier)
        assert np.all(tile[:, 0] == carrier) and np.all(tile[:, -1] == carrier)
        assert set(map(int, np.unique(tile))) <= {bg, carrier}
        clues = tile == bg
        assert not np.any(clues[:-1] & clues[1:]) and not np.any(clues[:, :-1] & clues[:, 1:])
        k = int(clues.sum())
        assert k in motifs
        # All preserved motif foreground remains visibly distinct from carrier.
        assert carrier not in set(map(int, np.unique(motifs[k]["motif"]))) - {bg}
        tiles.append({"r": (i // cols) * t, "c": (i % cols) * t,
                      "carrier": carrier, "count": k, "clues": clues[1:-1, 1:-1].copy()})
    assert len({tile["count"] for tile in tiles}) == len(tiles)
    return {"bg": bg, "size": s, "t": t, "rows": rows, "cols": cols,
            "box": (mr, mc, er, ec), "motifs": motifs, "tiles": tiles, "ordered": ordered}


def _decode(input_grid):
    grid = np.asarray(input_grid)
    _grid_check(grid)
    assert not ({CONVEX, CONCAVE} & set(map(int, np.unique(grid))))
    candidates = []
    for bg in map(int, np.unique(grid)):
        try:
            candidates.append(_decode_bg(grid, bg))
        except (AssertionError, StopIteration, IndexError, ValueError):
            continue
    assert len(candidates) == 1, "input must determine a unique background, mosaic, and contour interpretation"
    return candidates[0]


def _paint_motif(info, bg, carrier, labels="both"):
    out = np.where(info["motif"] == bg, carrier, info["motif"]).astype(np.int8)
    for loop in info["loops"]:
        for p, kind in loop["kinds"].items():
            if labels == "all":
                out[p] = CONVEX
            elif kind == 1 and labels != "none":
                out[p] = CONVEX
            elif kind == 2 and labels == "both":
                out[p] = CONCAVE
    return out


def _render(a, labels="both", assignment=None):
    t = a["t"]
    out = np.empty((a["rows"] * t, a["cols"] * t), dtype=np.int8)
    for tile in a["tiles"]:
        r, c, carrier, k = (tile[x] for x in ("r", "c", "carrier", "count"))
        out[r:r + t, c:c + t] = carrier
        selected = k if assignment is None else assignment[k]
        out[r + 1:r + t - 1, c + 1:c + t - 1] = _paint_motif(a["motifs"][selected], a["bg"], carrier, labels)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode count matching, transplant without reorientation, label all loops."""
    return _render(_decode(input_grid))


def _free_origins(boxes, h, w):
    rr, cc = np.indices((31 - h, 31 - w))
    free = np.ones(rr.shape, dtype=bool)
    for r, c, bh, bw in boxes:
        free &= (rr + h < r) | (rr > r + bh) | (cc + w < c) | (cc > c + bw)
    return np.argwhere(free)


def _pack_scene(rng, mosaic, motifs, bg, order_mode):
    h, w = mosaic.shape
    s, n = motifs[0].shape[0], len(motifs)
    boxes = None
    for attempt in range(13):
        if attempt == 12:
            r = c = 0
        else:
            r, c = int(rng.integers(31 - h)), int(rng.integers(31 - w))
        trial = [(r, c, h, w)]
        for _ in range(n):
            free = _free_origins(trial, s, s)
            if not len(free):
                break
            rr, cc = map(int, free[0] if attempt == 12 else free[int(rng.integers(len(free)))])
            trial.append((rr, cc, s, s))
        if len(trial) == n + 1:
            boxes = trial
            break
    assert boxes is not None
    positions = sorted(boxes[1:])
    order = (list(range(n)) if order_mode == "same" else list(reversed(range(n)))
             if order_mode == "reverse" else list(map(int, rng.permutation(n))))
    canvas = np.full((30, 30), bg, dtype=np.int8)
    r, c, _, _ = boxes[0]
    canvas[r:r + h, c:c + w] = mosaic
    for (r, c, _, _), i in zip(positions, order):
        canvas[r:r + s, c:c + s] = motifs[i]
    low_r, low_c = min(b[0] for b in boxes), min(b[1] for b in boxes)
    high_r, high_c = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    # Full canvases, zero margins, and independently sampled blank margins.
    if rng.random() < .18:
        return canvas
    low_r, low_c = int(rng.integers(low_r + 1)), int(rng.integers(low_c + 1))
    high_r, high_c = int(rng.integers(high_r, 31)), int(rng.integers(high_c, 31))
    return canvas[low_r:high_r, low_c:high_c].copy()


def _make_pair(rng, s, rows, cols, counts, bg, witness=None):
    n, t = rows * cols, s + 2
    palette = list(map(int, rng.permutation([c for c in COLORS if c != bg])))
    # Witness motifs share colors, preventing color-based matching.
    shared = witness is not None or rng.random() < .35
    motifs, outputs = {}, {}
    order = list(map(int, rng.permutation(counts)))
    mosaic = np.empty((rows * t, cols * t), dtype=np.int8)
    expected = np.empty_like(mosaic)
    for i, k in enumerate(order):
        body, stroke = (palette[:2] if shared else list(map(int, rng.choice(palette, 2, replace=False))))
        carrier = _pick(rng, [c for c in palette if c not in (body, stroke)])
        mode = ("concave" if witness == "bends" and k == 1 else
                "island" if witness == "island" and k == 1 else
                "small" if witness is not None and k >= 2 else "random")
        motif, loops = _make_motif(rng, s, k, bg, body, stroke, mode)
        info = {"motif": motif, "loops": loops}
        motifs[k] = motif
        outputs[k] = _paint_motif(info, bg, carrier)
        r, c = (i // cols) * t, (i % cols) * t
        tile = np.full((t, t), carrier, dtype=np.int8)
        # Arbitrary four-isolated clues: no fixed checkerboard parity.
        chosen = []
        for flat in (rng.permutation(s * s) if k else []):
            rr, cc = divmod(int(flat), s)
            if all(abs(rr - y) + abs(cc - x) != 1 for y, x in chosen):
                chosen.append((rr, cc))
                if len(chosen) == k:
                    break
        if k:
            assert len(chosen) == k
            for rr, cc in chosen:
                tile[rr + 1, cc + 1] = bg
        mosaic[r:r + t, c:c + t] = tile
        expected[r:r + t, c:c + t] = carrier
        expected[r + 1:r + t - 1, c + 1:c + t - 1] = outputs[k]
    mode = "same" if witness == "bends" else "reverse" if witness == "island" else "random"
    inp = _pack_scene(rng, mosaic, [motifs[k] for k in order], bg, mode)
    return {"input": inp, "output": expected}


def _counts_check(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts_check(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    previous_bg = None
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "test":
                # Novel scale; selection, island semantics and labels are already demonstrated.
                s = int(rng.integers(10, 14))
                rows, cols = _pick(rng, ((1, 2), (2, 1)))
                witness = None
                counts = [1, int(rng.integers(2, _capacity(s, s) + 1))]
                if i > 0 and rng.random() < .4:
                    counts[0] = 0
            else:
                s = int(rng.integers(7, 9 if i < 2 else 10))
                layouts = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2)) if s <= 8 else ((1, 2), (2, 1))
                rows, cols = _pick(rng, layouts)
                n = rows * cols
                witness = "bends" if i == 0 else "island" if i == 1 else None
                required = [1, 2] if witness else []
                options = [k for k in range(_capacity(s, s) + 1) if k not in required]
                counts = required + list(map(int, rng.choice(options, n - len(required), replace=False)))
            bg = _pick(rng, [c for c in COLORS if c != previous_bg])
            previous_bg = bg
            for _ in range(8):
                pair = _make_pair(rng, s, rows, cols, counts, bg, witness)
                key = _grid_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness retries exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_check(grid):
    assert grid.ndim == 2 and all(1 <= x <= 30 for x in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((grid >= 0) & (grid <= 9))


def _grid_key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _point_inside(path, row, col):
    # Independent geometric check at noninteger sample points.
    return sum(c == cc and min(r, rr) < row < max(r, rr) and c > col
               for (r, c), (rr, cc) in zip(path, path[1:] + path[:1])) % 2 == 1


def _features(a):
    motifs = a["motifs"]
    areas = {k: sum(m["areas"]) for k, m in motifs.items()}
    return {
        "concave": any(2 in loop["kinds"].values() for m in motifs.values() for loop in m["loops"]),
        "straight": any(0 in loop["kinds"].values() for m in motifs.values() for loop in m["loops"]),
        "island": any(m["islands"] for m in motifs.values()),
        "area_inversion": any(k < j and areas[k] > areas[j] for k in areas for j in areas),
        "nonpositional": any(tile["count"] and not np.array_equal(tile["clues"], motifs[tile["count"]]["motif"] == a["bg"])
                             for tile in a["tiles"]),
        "source_order_differs": a["ordered"] != [t["count"] for t in a["tiles"]],
    }


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts_check(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            _grid_check(inp)
            _grid_check(out)
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            a = _decode(inp)
            assert np.array_equal(_render(a), out), "wrong selection, contour labels, preservation, or crop"
            analyses[split].append(a)
            for motif in a["motifs"].values():
                # Check every bend by the number of enclosed vertex quadrants,
                # independently of the signed turn used by solve().
                for loop in motif["loops"]:
                    for (r, c), kind in loop["kinds"].items():
                        quadrants = sum(_point_inside(loop["path"], r + dr, c + dc)
                                        for dr in (-.25, .25) for dc in (-.25, .25))
                        assert quadrants == (2 if kind == 0 else 1 if kind == 1 else 3)
            for tile in a["tiles"]:
                r, c, t, carrier = tile["r"], tile["c"], a["t"], tile["carrier"]
                region = out[r:r + t, c:c + t]
                assert np.all(region[0] == carrier) and np.all(region[-1] == carrier)
                assert np.all(region[:, 0] == carrier) and np.all(region[:, -1] == carrier)
                info = a["motifs"][tile["count"]]
                interior = region[1:-1, 1:-1]
                assert np.all(interior[info["motif"] == a["bg"]] == carrier)
                for loop in info["loops"]:
                    for p, kind in loop["kinds"].items():
                        if kind == 0:
                            assert interior[p] == info["motif"][p]
    train_features = [_features(a) for a in analyses["train"]]
    for name in ("concave", "straight", "island", "area_inversion", "nonpositional", "source_order_differs"):
        assert any(f[name] for f in train_features), "missing training witness: " + name
    # The same count selects different contour geometry across demonstrations.
    signatures = {}
    for a in analyses["train"]:
        for k, m in a["motifs"].items():
            kinds = tuple(sorted(sum(kind == 2 for kind in loop["kinds"].values()) for loop in m["loops"]))
            signatures.setdefault(k, set()).add(kinds)
    assert any(len(v) > 1 for v in signatures.values())


def self_test():
    """42 streamed seeds, bounded aggregates, and focused semantic mutations."""
    shapes, layouts, count_sets, masks, turn_patterns, backgrounds = set(), set(), set(), set(), set(), set()
    totals = {k: 0 for k in ("pairs", "motifs", "island_motifs", "concave", "straight", "zero_motifs", "area_inversions")}
    fingerprints = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                a = _decode(inp)
                assert np.array_equal(solve(inp.copy()), out)
                digest.update(inp.tobytes())
                shapes.add(inp.shape)
                layouts.add((a["size"], a["rows"], a["cols"]))
                backgrounds.add(a["bg"])
                count_sets.add(tuple(sorted(a["motifs"])))
                totals["pairs"] += 1
                totals["area_inversions"] += _features(a)["area_inversion"]
                for k, m in a["motifs"].items():
                    totals["motifs"] += 1
                    totals["zero_motifs"] += k == 0
                    totals["island_motifs"] += m["islands"] > 0
                    # Hashes only: no retained reconstructible grid collection.
                    masks.add((a["size"], hashlib.sha256((m["motif"] == a["bg"]).tobytes()).hexdigest()))
                    for loop in m["loops"]:
                        turns = [loop["kinds"][p] for p in loop["path"] if loop["kinds"][p]]
                        candidates = [tuple(v[i:] + v[:i]) for v in (turns, turns[::-1]) for i in range(len(v))]
                        turn_patterns.add(min(candidates))
                        totals["concave"] += sum(v == 2 for v in loop["kinds"].values())
                        totals["straight"] += sum(v == 0 for v in loop["kinds"].values())
                # Omitting either mechanism changes training outputs.
                if split == "train" and _features(a)["concave"]:
                    assert not np.array_equal(_render(a, "none"), out)
                    assert not np.array_equal(_render(a, "convex"), out)
                    assert not np.array_equal(_render(a, "all"), out)
                    keys = sorted(a["motifs"])
                    wrong = dict(zip(keys, keys[1:] + keys[:1]))
                    assert not np.array_equal(_render(a, assignment=wrong), out)
        fingerprints.add(digest.hexdigest())
    assert len(fingerprints) == 42 and len(shapes) >= 25 and len(layouts) >= 12
    assert len(count_sets) >= 12 and len(masks) >= 150 and len(turn_patterns) >= 8
    assert backgrounds == set(COLORS)
    assert all(totals[k] > 0 for k in totals)
    first, second = generate_puzzle(0, 2, 1), generate_puzzle(0, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], second[s]) for k in p)
    # An island adds a contour without adding a neutral cavity.
    a = _decode(first["train"][1]["input"])
    assert any(len(m["loops"]) > m["count"] for m in a["motifs"].values())
    # Corrupt a straight stroke, not just a randomly chosen corner.
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    a = _decode(broken["train"][0]["input"])
    tile = next(t for t in a["tiles"] if t["count"] == 1)
    loop = a["motifs"][1]["loops"][0]
    r, c = next(p for p, kind in loop["kinds"].items() if kind == 0)
    broken["train"][0]["output"][tile["r"] + 1 + r, tile["c"] + 1 + c] = CONVEX
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("straight-cell corruption accepted")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "input_shapes": len(shapes), "size_layouts": len(layouts),
            "count_sets": len(count_sets), "cavity_masks": len(masks),
            "cyclic_turn_patterns": len(turn_patterns), "totals": totals,
            "independent_solver": False, "blind_prediction": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
