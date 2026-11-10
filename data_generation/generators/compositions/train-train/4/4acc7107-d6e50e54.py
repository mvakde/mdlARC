"""Polarity docking around a cored magnet rectangle.

Composition of parent A (4acc7107: segment monochrome objects, sort them into
per-identity stacks that rest flush on a wall with one blank line between
consecutive objects, translation only) and parent B (d6e50e54: a solid
rectangle attractor, exterior pieces assigned to the side they face, pulled
straight in while keeping their tangential coordinate and their order, ending
flush against the rectangle).

Rule.  The input holds one framed rectangle (a one-cell ring of frame colour F
around a solid interior of core colour K) and several monochrome objects that
each lie entirely in one of the rectangle's four axial lanes (the band of
cells directly above, below, left or right of it, within its span).  Objects
whose colour equals the core colour K are attracted: they slide straight
toward the rectangle.  Every other object is repelled: it slides straight away
from the rectangle to the grid border of its lane.  Walls (the rectangle and
the grid border) are touched flush; an object that meets an already-settled
object of its own polarity whose span overlaps its own stops with exactly one
blank line between their bounding boxes.  Objects nearest their destination
settle first, so order is preserved; attracted and repelled objects pass
through each other.  The rectangle never changes.
"""
from __future__ import annotations

from collections import deque

import numpy as np

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


# ---------------------------------------------------------------------------
# Parsing helpers (shared by solve, alternative rules and validation)
# ---------------------------------------------------------------------------

def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(grid, bg, exclude=None):
    """4-connected monochrome components of non-background cells."""
    h, w = grid.shape
    seen = grid == bg
    if exclude is not None:
        seen = seen | exclude
    seen = seen.copy()
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            cells = []
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                y, x = queue.popleft()
                cells.append((y, x))
                for dy, dx in _DIRS:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] == color:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            comps.append((color, cells))
    return comps


def _make_object(color, cells):
    ys = [p[0] for p in cells]
    xs = [p[1] for p in cells]
    top, bottom, left, right = min(ys), max(ys), min(xs), max(xs)
    mask = np.zeros((bottom - top + 1, right - left + 1), dtype=bool)
    for y, x in cells:
        mask[y - top, x - left] = True
    return {"color": color, "top": top, "bottom": bottom, "left": left, "right": right, "mask": mask}


def _find_rects(grid, bg):
    """All framed rectangles: ring of one colour around a uniform interior of another."""
    found = []
    for color, cells in _components(grid, bg):
        obj = _make_object(color, cells)
        h = obj["bottom"] - obj["top"] + 1
        w = obj["right"] - obj["left"] + 1
        if h < 3 or w < 3 or len(cells) != 2 * h + 2 * w - 4:
            continue
        block = grid[obj["top"]:obj["bottom"] + 1, obj["left"]:obj["right"] + 1]
        ring = np.ones((h, w), dtype=bool)
        ring[1:-1, 1:-1] = False
        if not np.all(block[ring] == color):
            continue
        inner = block[1:-1, 1:-1]
        core = int(inner[0, 0])
        if core in (bg, color) or not np.all(inner == core):
            continue
        found.append({"top": obj["top"], "bottom": obj["bottom"], "left": obj["left"],
                      "right": obj["right"], "frame": color, "core": core})
    return found


def _top_lane_objects(grid, bg, rect):
    """Objects lying entirely in the lane above the rectangle (current frame)."""
    exclude = np.zeros(grid.shape, dtype=bool)
    exclude[rect["top"]:rect["bottom"] + 1, rect["left"]:rect["right"] + 1] = True
    objs = []
    for color, cells in _components(grid, bg, exclude):
        obj = _make_object(color, cells)
        if obj["bottom"] < rect["top"] and obj["left"] >= rect["left"] and obj["right"] <= rect["right"]:
            objs.append(obj)
    return objs


def _overlap(a, b):
    return not (a["right"] < b["left"] or b["right"] < a["left"])


# ---------------------------------------------------------------------------
# Settling in the canonical frame (the lane is above the rectangle)
# ---------------------------------------------------------------------------

def _settle_bbox(objs, r0, gap=1, projection=True):
    """Return new top row per object index using bounding-box stacking.

    gap: blank lines kept between consecutive objects of one polarity.
    projection: if False, every same-polarity object in the lane queues behind
    the previous one even when their spans do not overlap (shortcut rule).
    """
    new_top = {}
    attract = sorted((i for i, o in enumerate(objs) if o["group"] == "A"), key=lambda i: -objs[i]["bottom"])
    placed = []
    for i in attract:
        o = objs[i]
        a = o["bottom"] - o["top"] + 1
        bottom = r0 - 1
        for j in placed:
            if not projection or _overlap(o, objs[j]):
                bottom = min(bottom, new_top[j] - gap - 1)
        new_top[i] = bottom - a + 1
        placed.append(i)
    repel = sorted((i for i, o in enumerate(objs) if o["group"] == "R"), key=lambda i: objs[i]["top"])
    placed = []
    for i in repel:
        o = objs[i]
        a = o["bottom"] - o["top"] + 1
        top = 0
        for j in placed:
            if not projection or _overlap(o, objs[j]):
                top = max(top, new_top[j] + (objs[j]["bottom"] - objs[j]["top"]) + gap + 1)
        new_top[i] = top
        placed.append(i)
    return new_top


def _settle_cells(objs, r0):
    """Cell-level sliding: one blank cell in the motion direction, walls flush."""
    new_top = {}
    for group, step in (("A", 1), ("R", -1)):
        order = [i for i, o in enumerate(objs) if o["group"] == group]
        order.sort(key=(lambda i: -objs[i]["bottom"]) if step == 1 else (lambda i: objs[i]["top"]))
        occupied = set()
        for i in order:
            o = objs[i]
            cells = [(y + o["top"], x + o["left"]) for y, x in zip(*np.nonzero(o["mask"]))]
            shift = 0

            def ok(s):
                for y, x in cells:
                    ny = y + s
                    if ny < 0 or ny > r0 - 1:
                        return False
                    if (ny, x) in occupied or (ny + step, x) in occupied:
                        return False
                return True

            while ok(shift + step):
                shift += step
            new_top[i] = o["top"] + shift
            occupied.update((y + shift, x) for y, x in cells)
    return new_top


def _transform(input_grid, method="bbox", gap=1, projection=True, polarity=None):
    grid = np.array(input_grid, dtype=int)
    bg = _background(grid)
    rects = _find_rects(grid, bg)
    if len(rects) != 1:
        return grid.copy()
    out = grid.copy()
    for k in range(4):
        rg = np.rot90(out, k).copy()
        rect = _rotated_rect(rects[0], grid.shape, k)
        objs = _top_lane_objects(rg, bg, rect)
        if not objs:
            continue
        for o in objs:
            if polarity is None:
                o["group"] = "A" if o["color"] == rect["core"] else "R"
            else:
                o["group"] = polarity
        if method == "cells":
            new_top = _settle_cells(objs, rect["top"])
        else:
            new_top = _settle_bbox(objs, rect["top"], gap=gap, projection=projection)
        for o in objs:
            rg[o["top"]:o["bottom"] + 1, o["left"]:o["right"] + 1][o["mask"]] = bg
        for i, o in enumerate(objs):
            t = new_top[i]
            h = o["bottom"] - o["top"] + 1
            if t < 0 or t + h > rg.shape[0]:
                continue
            rg[t:t + h, o["left"]:o["right"] + 1][o["mask"]] = o["color"]
        out = np.rot90(rg, -k).copy()
    return out


def _rotated_rect(rect, shape, k):
    """Rectangle bounds after np.rot90(grid, k) (counter-clockwise k times)."""
    top, bottom, left, right = rect["top"], rect["bottom"], rect["left"], rect["right"]
    h, w = shape
    for _ in range(k % 4):
        # rot90 CCW: (r, c) -> (w-1-c, r); new shape (w, h)
        top, bottom, left, right = w - 1 - right, w - 1 - left, top, bottom
        h, w = w, h
    return {"top": top, "bottom": bottom, "left": left, "right": right,
            "frame": rect["frame"], "core": rect["core"]}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the output from the input alone."""
    return _transform(input_grid)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _random_shape(rng, max_a, max_w):
    """Mask of shape (axial, tangential); row 0 is the far side of the lane."""
    max_a = max(1, max_a)
    max_w = max(1, max_w)
    kind = rng.random()
    if kind < 0.22:
        a, w = int(rng.integers(1, max_a + 1)), int(rng.integers(1, max_w + 1))
        return np.ones((a, w), dtype=bool)
    if kind < 0.34:
        if rng.random() < 0.5:
            return np.ones((1, int(rng.integers(2, max_w + 1)) if max_w >= 2 else 1), dtype=bool)
        return np.ones((int(rng.integers(2, max_a + 1)) if max_a >= 2 else 1, 1), dtype=bool)
    a, w = int(rng.integers(1, max_a + 1)), int(rng.integers(1, max_w + 1))
    if a * w <= 2:
        return np.ones((a, w), dtype=bool)
    box = np.zeros((a, w), dtype=bool)
    target = int(rng.integers(max(2, (a * w + 1) // 2), a * w + 1))
    y, x = int(rng.integers(a)), int(rng.integers(w))
    box[y, x] = True
    count = 1
    while count < target:
        ys, xs = np.nonzero(box)
        idx = int(rng.integers(len(ys)))
        dy, dx = _DIRS[int(rng.integers(4))]
        ny, nx = ys[idx] + dy, xs[idx] + dx
        if 0 <= ny < a and 0 <= nx < w and not box[ny, nx]:
            box[ny, nx] = True
            count += 1
    ys, xs = np.nonzero(box)
    return box[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _shared_offset(rng, width, span, column):
    lo = max(0, column - width + 1)
    hi = min(column, span - width)
    return int(rng.integers(lo, hi + 1))


def _recipe(rng, name, span, max_a):
    """Lane content ordered from the rectangle outward: list of (group, mask, offset)."""
    maxw = min(span, 5)
    if name in ("stackA", "stackR", "stack3A", "stack3R"):
        group = name[-1]
        n = 3 if name.startswith("stack3") else 2
        column = int(rng.integers(span))
        items = []
        for _ in range(n):
            m = _random_shape(rng, max_a, maxw)
            items.append((group, m, _shared_offset(rng, m.shape[1], span, column)))
        return items
    if name == "cross":
        column = int(rng.integers(span))
        mr = _random_shape(rng, max_a, maxw)
        ma = _random_shape(rng, max_a, maxw)
        return [("R", mr, _shared_offset(rng, mr.shape[1], span, column)),
                ("A", ma, _shared_offset(rng, ma.shape[1], span, column))]
    if name in ("parallelA", "parallelR"):
        group = name[-1]
        w1 = int(rng.integers(1, max(1, (span - 1) // 2) + 1))
        w2 = int(rng.integers(1, max(1, span - 1 - w1) + 1))
        m1 = _random_shape(rng, max_a, w1)
        m2 = _random_shape(rng, max_a, w2)
        w1, w2 = m1.shape[1], m2.shape[1]
        start1 = int(rng.integers(0, span - w1 - w2 - 1 + 1))
        start2 = int(rng.integers(start1 + w1 + 1, span - w2 + 1))
        items = [(group, m1, start1), (group, m2, start2)]
        if rng.random() < 0.5:
            items.reverse()
        return items
    # random lane: 1-2 objects of random polarity
    n = 1 if rng.random() < 0.55 else 2
    items = []
    for _ in range(n):
        m = _random_shape(rng, max_a, maxw)
        items.append(("A" if rng.random() < 0.5 else "R", m, int(rng.integers(0, span - m.shape[1] + 1))))
    return items


def _lane_layout(rng, items):
    """Assign depth bands (near depth, 1 = adjacent to rectangle) to lane items."""
    placed = []
    depth = 1 + int(rng.integers(1, 4))  # at least one blank between rectangle and object
    for group, mask, off in items:
        a = mask.shape[0]
        placed.append({"group": group, "mask": mask, "off": off, "dn": depth, "df": depth + a - 1})
        depth = depth + a + 1 + int(rng.integers(0, 3))
    return placed, depth - 1  # depth - 1 is the minimal lane length (last gap counted)


def _simulate_lane(objs, L):
    """Depth-coordinate settling; returns new near depth per object (independent of solve)."""
    def ov(a, b):
        return not (a["off"] + a["mask"].shape[1] - 1 < b["off"] or b["off"] + b["mask"].shape[1] - 1 < a["off"])

    res = {}
    att = sorted((i for i, o in enumerate(objs) if o["group"] == "A"), key=lambda i: objs[i]["dn"])
    done = []
    for i in att:
        dn = 1
        for j in done:
            if ov(objs[i], objs[j]):
                dn = max(dn, res[j] + objs[j]["mask"].shape[0] + 1)
        res[i] = dn
        done.append(i)
    rep = sorted((i for i, o in enumerate(objs) if o["group"] == "R"), key=lambda i: -objs[i]["df"])
    done = []
    for i in rep:
        df = L
        for j in done:
            if ov(objs[i], objs[j]):
                df = min(df, res[j] - 2)
        res[i] = df - objs[i]["mask"].shape[0] + 1
        done.append(i)
    return res


def _separated(boxes):
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if not (a[1] + 1 < b[0] or b[1] + 1 < a[0] or a[3] + 1 < b[2] or b[3] + 1 < a[2]):
                return False
    return True


def _make_example(rng, colors, recipes, big):
    bg, frame, core, repel_colors = colors
    rh = int(rng.integers(3, 11 if big else 9))
    rw = int(rng.integers(3, 11 if big else 9))
    sides = list(rng.permutation(4))
    if len(recipes) > 4:
        return None
    max_a = 3 if not big else int(rng.integers(2, 4))
    if rng.random() < 0.15:
        max_a = 4
    lanes = {}
    for side, name in zip(sides, recipes):
        span = rw if side % 2 == 0 else rh
        items = _recipe(rng, name, span, max_a)
        if name != "random" and rng.random() < 0.25:
            m = _random_shape(rng, 2, min(span, 4))
            items.append(("A" if rng.random() < 0.5 else "R", m, int(rng.integers(0, span - m.shape[1] + 1))))
        placed, need = _lane_layout(rng, items)
        L = need + int(rng.integers(0, 4))
        lanes[side] = (placed, L)
    depths = {}
    for side in range(4):
        if side in lanes:
            depths[side] = lanes[side][1]
        else:
            depths[side] = int(rng.integers(0, 5)) if rng.random() < 0.8 else 0
    H = depths[0] + rh + depths[2]
    W = depths[3] + rw + depths[1]
    if H > 30 or W > 30:
        return None
    grid_in = np.full((H, W), bg, dtype=int)
    top, left = depths[0], depths[3]
    grid_in[top:top + rh, left:left + rw] = frame
    grid_in[top + 1:top + rh - 1, left + 1:left + rw - 1] = core
    grid_out = grid_in.copy()
    rect = {"top": top, "bottom": top + rh - 1, "left": left, "right": left + rw - 1, "frame": frame, "core": core}
    flat = np.arange(H * W).reshape(H, W)
    repel_items = [o for placed, _ in lanes.values() for o in placed if o["group"] == "R"]
    for o in repel_items:
        o["color"] = int(repel_colors[int(rng.integers(len(repel_colors)))])
    if repel_items and all(o["color"] != repel_colors[0] for o in repel_items):
        repel_items[int(rng.integers(len(repel_items)))]["color"] = int(repel_colors[0])
    moved_a = moved_r = 0
    for side, (placed, L) in lanes.items():
        rr = _rotated_rect(rect, (H, W), side)
        assert rr["top"] == L
        idx = np.rot90(flat, side)
        res = _simulate_lane(placed, L)
        in_boxes, out_boxes = [], []
        for i, o in enumerate(placed):
            a, w = o["mask"].shape
            new_dn = res[i]
            if new_dn < 1 or new_dn + a - 1 > L:
                return None
            if o["group"] == "R" and new_dn < 2:
                return None
            if new_dn != o["dn"]:
                if o["group"] == "A":
                    moved_a += 1
                else:
                    moved_r += 1
            color = core if o["group"] == "A" else o["color"]
            for dn, store, boxes in ((o["dn"], grid_in, in_boxes), (new_dn, grid_out, out_boxes)):
                top_row = L - (dn + a - 1)
                c0 = rr["left"] + o["off"]
                boxes.append((top_row, top_row + a - 1, c0, c0 + w - 1))
                for y, x in zip(*np.nonzero(o["mask"])):
                    f = idx[top_row + y, c0 + x]
                    store[f // W, f % W] = color
        if not _separated(in_boxes) or not _separated(out_boxes):
            return None
    if moved_a == 0 or moved_r == 0:
        return None
    values, counts = np.unique(grid_in, return_counts=True)
    order = np.argsort(-counts)
    if values[order[0]] != bg or (len(counts) > 1 and counts[order[0]] == counts[order[1]]):
        return None
    if not np.array_equal(solve(grid_in), grid_out):
        return None
    if not np.array_equal(_transform(grid_in, method="cells"), grid_out):
        return None
    return grid_in, grid_out


def _pick(rng, pool, avoid):
    choices = [c for c in pool if c not in avoid]
    return int(choices[int(rng.integers(len(choices)))])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not (isinstance(num_train, (int, np.integer)) and isinstance(num_test, (int, np.integer))):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("illegal train/test counts")
    rng = np.random.default_rng(seed)
    palette = list(range(10))
    puzzle_bg = 0 if rng.random() < 0.5 else int(rng.integers(10))
    vary_bg = rng.random() < 0.25
    puzzle_frame = _pick(rng, palette, {puzzle_bg})
    vary_frame = rng.random() < 0.3
    swap = [c for c in palette if c not in (puzzle_bg, puzzle_frame)]
    rng.shuffle(swap)
    c1, c2 = int(swap[0]), int(swap[1])

    required = ["stackA", "stackR", "cross", "parallelA" if rng.random() < 0.5 else "parallelR"]
    rng.shuffle(required)
    train_recipes = [[] for _ in range(num_train)]
    for i, name in enumerate(required):
        train_recipes[i % num_train].append(name)
    for recipes in train_recipes:
        lanes = int(rng.integers(2, 4))
        while len(recipes) < lanes:
            recipes.append("random")
    test_recipes = []
    for _ in range(num_test):
        special = ["stack3A" if rng.random() < 0.5 else "stack3R"]
        special.append(["cross", "stackA", "stackR", "parallelA", "parallelR"][int(rng.integers(5))])
        lanes = int(rng.integers(3, 5))
        while len(special) < lanes:
            special.append("random")
        test_recipes.append(special)

    def colors_for(index, role_swap):
        bg = puzzle_bg
        if vary_bg and rng.random() < 0.5:
            bg = _pick(rng, palette, {c1, c2, puzzle_frame})
        frame = puzzle_frame
        if vary_frame and rng.random() < 0.5:
            frame = _pick(rng, palette, {bg, c1, c2})
        if role_swap == 0:
            core, repel = c1, [c2]
        elif role_swap == 1:
            core, repel = c2, [c1]
        else:
            core = _pick(rng, palette, {bg, frame})
            repel = [_pick(rng, palette, {bg, frame, core})]
        r = rng.random()
        if r < 0.25:
            repel.append(frame)
        elif r < 0.45:
            repel.append(_pick(rng, palette, {bg, frame, core, *repel}))
        return bg, frame, core, repel

    pairs = []
    seen = set()
    all_specs = [(train_recipes[i], i if i < 2 else None, False) for i in range(num_train)]
    all_specs += [(test_recipes[i], None, True) for i in range(num_test)]
    for recipes, role_swap, big in all_specs:
        for _attempt in range(400):
            colors = colors_for(len(pairs), role_swap)
            order = list(recipes)
            rng.shuffle(order)
            made = _make_example(rng, colors, order, big)
            if made is None:
                continue
            key = made[0].tobytes() + bytes(made[0].shape)
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"input": made[0], "output": made[1]})
            break
        else:
            raise RuntimeError("could not build example")
    train = pairs[:num_train]
    order = rng.permutation(num_train)
    train = [train[int(i)] for i in order]
    return {"train": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in train],
            "test": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in pairs[num_train:]]}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _parse(grid):
    """Return (bg, rect, lane objects with side and polarity) or raise."""
    grid = np.asarray(grid, dtype=int)
    bg = _background(grid)
    rects = _find_rects(grid, bg)
    if len(rects) != 1:
        raise AssertionError("input must contain exactly one framed rectangle")
    rect = rects[0]
    covered = np.zeros(grid.shape, dtype=bool)
    covered[rect["top"]:rect["bottom"] + 1, rect["left"]:rect["right"] + 1] = True
    lanes = []
    for k in range(4):
        rg = np.rot90(grid, k)
        rr = _rotated_rect(rect, grid.shape, k)
        cov = np.rot90(covered, k)
        objs = _top_lane_objects(rg, bg, rr)
        for o in objs:
            o["group"] = "A" if o["color"] == rr["core"] else "R"
            cov[o["top"]:o["bottom"] + 1, o["left"]:o["right"] + 1] |= o["mask"]
            if o["bottom"] > rr["top"] - 2:
                raise AssertionError("input object touches the rectangle")
        covered = np.rot90(cov, -k)
        boxes = [(o["top"], o["bottom"], o["left"], o["right"]) for o in objs]
        if not _separated(boxes):
            raise AssertionError("input objects are not separated by a blank line")
        lanes.append((k, rr, objs))
    if np.any((grid != bg) & ~covered):
        raise AssertionError("cell outside rectangle and lanes")
    return bg, rect, lanes


def _signature(grid, bg):
    sig = []
    rects = _find_rects(grid, bg)
    ex = np.zeros(grid.shape, dtype=bool)
    for r in rects:
        ex[r["top"]:r["bottom"] + 1, r["left"]:r["right"] + 1] = True
    for color, cells in _components(grid, bg, ex):
        o = _make_object(color, cells)
        sig.append((color, o["mask"].shape, o["mask"].tobytes()))
    return sorted(sig)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    evidence = {"stack": False, "gap0": False, "queue": False, "cross": False,
                "all_attract": False, "all_repel": False}
    cores, repelled_colors = [], set()
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert set(pair) == {"input", "output"}
            inp = np.asarray(pair["input"], dtype=int)
            out = np.asarray(pair["output"], dtype=int)
            for g in (inp, out):
                assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
                assert g.min() >= 0 and g.max() <= 9
            key = inp.tobytes() + bytes(inp.shape)
            assert key not in seen, "duplicate input"
            seen.add(key)
            bg, rect, lanes = _parse(inp)
            assert np.array_equal(solve(inp), out), "solve disagrees"
            assert np.array_equal(_transform(inp, method="cells"), out), "bbox/cell ambiguity"
            assert out.shape == inp.shape
            r = rect
            assert np.array_equal(inp[r["top"]:r["bottom"] + 1, r["left"]:r["right"] + 1],
                                  out[r["top"]:r["bottom"] + 1, r["left"]:r["right"] + 1])
            assert _signature(inp, bg) == _signature(out, bg), "objects not conserved"
            moved = {"A": 0, "R": 0}
            for k, rr, objs in lanes:
                ro = np.rot90(out, k)
                after = _top_lane_objects(ro, bg, rr)
                boxes = [(o["top"], o["bottom"], o["left"], o["right"]) for o in after]
                assert _separated(boxes), "output objects touch"
                assert len(after) == len(objs)
                for o in objs:
                    still = any(a["top"] == o["top"] and a["left"] == o["left"] and a["color"] == o["color"]
                                and a["mask"].shape == o["mask"].shape and np.array_equal(a["mask"], o["mask"])
                                for a in after)
                    if not still:
                        moved[o["group"]] += 1
                for a in objs:
                    for b in objs:
                        if a["group"] == "R" and b["group"] == "A" and _overlap(a, b) and a["bottom"] > b["bottom"]:
                            if split == "train":
                                evidence["cross"] = True
                for o in objs:
                    if o["group"] == "R":
                        repelled_colors.add((len(cores), o["color"]))
            assert moved["A"] > 0 and moved["R"] > 0, "each pair must move both polarities"
            cores.append((split, rect["core"]))
            if split == "train":
                for name, kwargs in (("gap0", {"gap": 0}), ("queue", {"projection": False}),
                                     ("all_attract", {"polarity": "A"}), ("all_repel", {"polarity": "R"})):
                    if not np.array_equal(_transform(inp, **kwargs), out):
                        evidence[name] = True
                if not np.array_equal(_transform(inp, gap=0), _transform(inp, gap=5)):
                    evidence["stack"] = True
    missing = [k for k, v in evidence.items() if not v]
    assert not missing, f"training evidence missing: {missing}"
    train_cores = [c for s, c in cores if s == "train"]
    assert len(set(train_cores)) >= 2, "core colour must vary in training"
    swap = any(c == color for i, (s, c) in enumerate(cores) if s == "train"
               for (j, color) in repelled_colors if j != i and j < len(train))
    assert swap, "need a colour attracted in one training pair and repelled in another"
    return True


def self_test():
    shapes, rect_sizes, cores, lane_counts = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for pair in puzzle["train"] + puzzle["test"]:
            g = np.asarray(pair["input"])
            shapes.add(g.shape)
            bg, rect, lanes = _parse(g)
            rect_sizes.add((rect["bottom"] - rect["top"] + 1, rect["right"] - rect["left"] + 1))
            cores.add(rect["core"])
            lane_counts.add(sum(1 for _, _, objs in lanes if objs))
    assert len(shapes) >= 40 and len(rect_sizes) >= 20 and len(cores) >= 8 and len(lane_counts) >= 3
    for bad in ((1, 1), (11, 1), (4, 0), (3, 3), (6, 5)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted illegal counts {bad}")
    return True
