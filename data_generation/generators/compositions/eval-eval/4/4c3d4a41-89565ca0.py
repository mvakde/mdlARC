"""Roof sieve: count complete rooms in partitioned pistons after a keyed press.

Only standard library and NumPy are used. The semantic reader reconstructs
geometry from input pixels; construction labels use original room boundaries.
"""
from collections import deque
import json

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _count(value, lower, upper, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lower <= int(value) <= upper):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


def _walls(shape, boxes):
    mask = np.zeros(shape, dtype=bool)
    for t, l, b, r in boxes:
        mask[t, l:r + 1] = mask[b, l:r + 1] = True
        mask[t:b + 1, l] = mask[t:b + 1, r] = True
    return mask


def _rooms(mask):
    """Bounded four-connected free regions; coordinates are interior boxes.

    Exterior padding ensures that an input edge never seals an open fragment.
    All elementary rooms in this family are rectangles with positive area.
    """
    free = np.pad(~mask, 1, constant_values=True)
    seen = np.zeros_like(free)
    h, w = free.shape
    result = []
    for yy, xx in np.argwhere(free):
        yy, xx = int(yy), int(xx)
        if seen[yy, xx]:
            continue
        todo = deque([(yy, xx)])
        seen[yy, xx] = True
        cells, exterior = [], False
        while todo:
            y, x = todo.popleft()
            cells.append((y - 1, x - 1))
            exterior |= y in (0, h - 1) or x in (0, w - 1)
            for v, u in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= v < h and 0 <= u < w and free[v, u] and not seen[v, u]:
                    seen[v, u] = True
                    todo.append((v, u))
        if not exterior:
            p = np.asarray(cells)
            lo, hi = p.min(axis=0), p.max(axis=0)
            if len(cells) != int(np.prod(hi - lo + 1)):
                raise ValueError("nonrectangular elementary room")
            result.append(tuple(map(int, (*lo, *hi))))
    return result


def _room_walls(rooms):
    return [(t - 1, l - 1, b + 1, r + 1) for t, l, b, r in rooms]


def _is_partial(mask):
    hs = np.flatnonzero(np.any(mask[:, :-1] & mask[:, 1:], axis=1))
    vs = np.flatnonzero(np.any(mask[:-1] & mask[1:], axis=0))
    return not (all(mask[y].all() for y in hs)
                and all(mask[:, x].all() for x in vs))


def _capacity(box):
    t, l, b, r = box
    return ((b - t) // 2) * ((r - l) // 2)


def _partition(rng, height, width, number, partial=False, strip=False):
    """Random recursive splits retain genuine segment endpoints and room area."""
    outer = (0, 0, height - 1, width - 1)
    if not 1 <= number <= _capacity(outer):
        raise ValueError("partition does not fit")
    for attempt in range(80):
        boxes = [outer]
        preferred_axis = int(rng.integers(2)) if strip else None
        for _ in range(number - 1):
            choices = []
            for index, (t, l, b, r) in enumerate(boxes):
                for axis, start, stop in ((0, t, b), (1, l, r)):
                    if preferred_axis is not None and axis != preferred_axis:
                        continue
                    for cut in range(start + 2, stop - 1):
                        children = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else (
                            (t, l, b, cut), (t, cut, b, r))
                        capacity = sum(_capacity(box) for j, box in enumerate(boxes) if j != index)
                        capacity += sum(_capacity(box) for box in children)
                        if capacity >= number:
                            choices.append((index, children))
            if not choices:
                break
            index, children = choices[int(rng.integers(len(choices)))]
            boxes[index:index + 1] = children
        if len(boxes) != number:
            # A strip may be impossible although a mixed partition fits.
            strip = False
            continue
        mask = _walls((height, width), boxes)
        if partial and not _is_partial(mask):
            continue
        return mask, boxes
    # Deterministic finite guillotine search, only for the requested count.
    def search(boxes):
        if len(boxes) == number:
            mask = _walls((height, width), boxes)
            return None if partial and not _is_partial(mask) else (mask, boxes)
        for index, (t, l, b, r) in enumerate(boxes):
            for axis, start, stop in ((0, t, b), (1, l, r)):
                for cut in range(start + 2, stop - 1):
                    children = [(t, l, cut, r), (cut, l, b, r)] if axis == 0 else [
                        (t, l, b, cut), (t, cut, b, r)]
                    nxt = boxes[:index] + children + boxes[index + 1:]
                    if sum(_capacity(box) for box in nxt) < number:
                        continue
                    found = search(nxt)
                    if found is not None:
                        return found
        return None
    found = search([outer])
    if found is None:
        raise ValueError("no partition of the requested kind fits")
    return found


def _sacrificial_partition(rng, height, width, shift, lost, surviving):
    cut = max(2, shift)
    top, top_boxes = _partition(rng, cut + 1, width, lost)
    bottom, bottom_boxes = _partition(rng, height - cut, width, surviving)
    boxes = top_boxes + [(t + cut, l, b + cut, r) for t, l, b, r in bottom_boxes]
    mask = _walls((height, width), boxes)
    assert np.array_equal(mask[:cut + 1], top)
    assert np.array_equal(mask[cut:], bottom)
    return mask, boxes


def _structure(shape, s, t, b, l, r, e, teeth):
    g = np.zeros(shape, dtype=bool)
    g[t, l:r + 1] = True
    g[t:b + 1, l] = g[t:b + 1, r] = True
    g[b, l + 1] = g[b, r - 1] = True
    g[e, l + 1:r] = True
    for x, h in teeth:
        g[e - h:e, x] = True
    return g


def _repair(grid, color, bg, s, t, b, l, r):
    visible = grid == color
    if np.any(visible[:t + 1]) or np.any(visible[b:]):
        return None
    if np.any(visible[:, :l + 2]) or np.any(visible[:, r - 1:]):
        return None
    p = np.argwhere(visible)
    if len(p) < 8:
        return None
    y0, x0 = map(int, p.min(axis=0))
    y1, x1 = map(int, p.max(axis=0))
    if y0 != t + 1 or y1 - y0 < 2 or x1 - x0 < 4 or (x1 - x0) % 2:
        return None
    mask = visible.copy()
    occupied = (grid != bg) & (grid != s) & ~visible
    horizontal = np.zeros_like(visible)
    vertical = np.zeros_like(visible)
    horizontal[:, 1:-1] = visible[:, :-2] & visible[:, 2:]
    vertical[1:-1] = visible[:-2] & visible[2:]
    mask |= occupied & (horizontal | vertical)
    local = mask[y0:y1 + 1, x0:x1 + 1]
    if not (local[0].all() and local[-1].all()
            and local[:, 0].all() and local[:, -1].all()):
        return None
    try:
        rooms = _rooms(local)
    except ValueError:
        return None
    if not rooms or len(rooms) > 30 or not np.array_equal(local, _walls(local.shape, _room_walls(rooms))):
        return None
    return {"color": int(color), "box": (y0, x0, y1, x1),
            "mask": local, "rooms": rooms}


def _canonical_read(g, bg, color, rotation):
    points = np.argwhere(g == color)
    t, e = int(points[:, 0].min()), int(points[:, 0].max())
    roof = np.flatnonzero(g[t] == color)
    l, r = int(roof[0]), int(roof[-1])
    if r - l < 14 or not np.all(g[t, l:r + 1] == color):
        raise ValueError("no roof")
    sides = np.flatnonzero(g[:, l] == color)
    b = int(sides[-1])
    if not (b - t >= 7 and e > b + 2
            and np.all(g[t:b + 1, l] == color)
            and np.all(g[t:b + 1, r] == color)
            and g[b, l + 1] == g[b, r - 1] == color
            and np.all(g[e, l + 1:r] == color)):
        raise ValueError("no dock and external base")
    xs = np.flatnonzero(g[e - 1] == color)
    if not 2 <= len(xs) <= 4 or np.any((xs < l + 2) | (xs > r - 2)):
        raise ValueError("invalid teeth")
    teeth = []
    for x in xs:
        y = e - 1
        while y >= 0 and g[y, x] == color:
            y -= 1
        h = e - 1 - y
        if not (1 <= h <= b - t - 3 and e - h >= b + 2):
            raise ValueError("invalid external tooth")
        teeth.append((int(x), int(h)))
    if not np.array_equal(g == color, _structure(g.shape, color, t, b, l, r, e, teeth)):
        raise ValueError("extraneous structure cells")
    networks, leftovers = [], []
    for c in np.unique(g):
        if c in (bg, color):
            continue
        n = _repair(g, int(c), bg, color, t, b, l, r)
        if n is None:
            leftovers.append(int(c))
        else:
            networks.append(n)
    if len(leftovers) != 1 or len(networks) != len(teeth):
        raise ValueError("ambiguous network/noise roles")
    noise = leftovers[0]
    networks.sort(key=lambda n: n["box"][1])
    previous = l
    for n, (x, h) in zip(networks, teeth):
        y0, x0, y1, x1 = n["box"]
        if x0 < previous + 2 or x != (x0 + x1) // 2:
            raise ValueError("piston/tooth mismatch")
        previous = x1
        shift = y1 - (b - h - 1)
        if not 0 <= shift <= h:
            raise ValueError("unsupported contact gap")
        n.update(height=h, shift=shift, tooth=x)
        hidden = n["mask"] & (g[y0:y1 + 1, x0:x1 + 1] != n["color"])
        if np.any(g[y0:y1 + 1, x0:x1 + 1][hidden] != noise):
            raise ValueError("wall damaged by an unknown role")
        if np.any((g[y0:y1 + 1, x0:x1 + 1] == noise) & ~n["mask"]):
            raise ValueError("noise in a room interior is outside this family")
        # Repair is isolated and never invents a boundary through background.
        if np.any(hidden[:-1] & hidden[1:]) or np.any(hidden[:, :-1] & hidden[:, 1:]):
            raise ValueError("adjacent occlusions outside family")
        for y, x in np.argwhere(hidden):
            y, x = int(y), int(x)
            arms = ((y > 0 and n["mask"][y - 1, x]),
                    (y + 1 < n["mask"].shape[0] and n["mask"][y + 1, x]),
                    (x > 0 and n["mask"][y, x - 1]),
                    (x + 1 < n["mask"].shape[1] and n["mask"][y, x + 1]))
            if arms not in ((True, True, False, False), (False, False, True, True)):
                raise ValueError("occluded corner or junction")
        # In the presented input, tie order follows the horizontal or vertical
        # lane axis. Quarter turns 1/2 reverse canonical lane positions.
        n["spatial"] = -x0 if rotation in (1, 2) else x0
    # All non-noise, non-background cells are accounted for geometrically.
    return {"grid": g, "background": bg, "structure": color, "noise": noise,
            "roof": t, "dock": b, "left": l, "right": r, "base": e,
            "rotation": rotation, "networks": networks}


def _read(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("background must be uniquely modal")
    bg = int(colors[np.argmax(counts)])
    interpretations = []
    for rotation in range(4):
        g = np.rot90(a, -rotation)
        for color in colors:
            if color == bg:
                continue
            try:
                interpretations.append(_canonical_read(g, bg, int(color), rotation))
            except (ValueError, IndexError):
                pass
    if len(interpretations) != 1:
        raise ValueError("input must have exactly one mechanism interpretation")
    return interpretations[0]


def _inventory(noise, items):
    items = sorted(items, key=lambda n: (n["surviving"], n["spatial"]))
    maximum = max(n["surviving"] for n in items)
    if maximum < 1:
        raise ValueError("an inventory needs at least one surviving room")
    out = np.full((len(items), maximum), noise, dtype=np.int8)
    for row, n in enumerate(items):
        out[row, :n["surviving"]] = n["color"]
    return out


def _press_count(f):
    """Stepwise movement and pixel clipping, with no construction room labels."""
    readings = []
    for original in f["networks"]:
        n = dict(original)
        y0, x0, y1, x1 = n["box"]
        cells = {(int(y) + y0, int(x) + x0)
                 for y, x in np.argwhere(n["mask"])}
        bottom = y1
        tooth_top = f["dock"] - n["height"]
        while bottom >= tooth_top:
            cells = {(y - 1, x) for y, x in cells if y - 1 > f["roof"]}
            bottom -= 1
        moved = np.zeros((f["dock"] + 1, x1 - x0 + 1), dtype=bool)
        for y, x in cells:
            moved[y, x - x0] = True
        n["surviving"] = len(_rooms(moved))
        n["pressed_mask"] = moved
        readings.append(n)
    return readings


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all colors, geometry, displacement, topology, and order from input."""
    f = _read(input_grid)
    return _inventory(f["noise"], _press_count(f))


def _piston(rng, depth, width, role, high=None, low=None):
    if role == "reversal_moving":
        h = int(rng.integers(4, min(6, depth - 3) + 1))
        shift = int(rng.integers(4, h + 1))
        height = depth - h + shift - 1
        mask, boxes = _sacrificial_partition(rng, height, width, shift, high, low)
    elif role == "reversal_still":
        h, shift = int(rng.integers(1, 3)), 0
        height = depth - h - 1
        mask, boxes = _partition(rng, height, width, high, partial=True)
    elif role == "zero":
        h = int(rng.integers(1, 5))
        shift = int(rng.integers(1, h + 1))
        height = depth - h + shift - 1
        number = int(rng.integers(1, (width - 1) // 2 + 1))
        # Only vertical splits: all rooms lose their common top boundary.
        cuts = [0, width - 1]
        for _ in range(number - 1):
            choices = [(i, cut) for i, (l, r) in enumerate(zip(cuts, cuts[1:]))
                       for cut in range(l + 2, r - 1)
                       if sum((v - u) // 2 for u, v in zip(cuts, cuts[1:]))
                       - (r - l) // 2 + (cut - l) // 2 + (r - cut) // 2 >= number]
            i, cut = choices[int(rng.integers(len(choices)))]
            cuts.insert(i + 1, cut)
        boxes = [(0, l, height - 1, r) for l, r in zip(cuts, cuts[1:])]
        mask = _walls((height, width), boxes)
    elif role == "tie_moving":
        contacts = [(h, shift) for h in range(2, 5) for shift in range(1, h + 1)
                    if _capacity((0, 0, depth - h + shift - 2 - max(2, shift), width - 1)) >= low]
        h, shift = contacts[int(rng.integers(len(contacts)))]
        height = depth - h + shift - 1
        lost = int(rng.integers(1, min(3, _capacity((0, 0, max(2, shift), width - 1))) + 1))
        mask, boxes = _sacrificial_partition(rng, height, width, shift, lost, low)
    elif role == "tie_still":
        h, shift = int(rng.integers(1, 5)), 0
        height = depth - h - 1
        mask, boxes = _partition(rng, height, width, low)
    else:
        h = int(rng.integers(1, min(6, depth - 3) + 1))
        shift = 0 if rng.random() < 0.3 else int(rng.integers(1, h + 1))
        height = depth - h + shift - 1
        cap = _capacity((0, 0, height - 1, width - 1))
        number = int(rng.integers(1, min(8, cap) + 1))
        mask, boxes = _partition(rng, height, width, number, strip=rng.random() < 0.25)
    return {"height": h, "shift": shift, "mask": mask, "boxes": boxes}


def _damage_sites(mask):
    candidates = []
    h, w = mask.shape
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        arms = ((y > 0 and mask[y - 1, x]), (y + 1 < h and mask[y + 1, x]),
                (x > 0 and mask[y, x - 1]), (x + 1 < w and mask[y, x + 1]))
        if arms in ((True, True, False, False), (False, False, True, True)):
            candidates.append((y, x))
    return candidates


def _make_pair(rng, parameters, split, witness=None):
    rotation = parameters["orientation"] if not parameters["mixed"] else int(rng.integers(4))
    depth_range = (10, 12) if parameters["shift"] == "depth" and split == "train" else (
        (14, 16) if parameters["shift"] == "depth" else (10, 16))
    if witness == "reversal":
        number = int(rng.integers(2, 4))
    elif witness == "tie_zero":
        number = 3 if parameters["shift"] == "lanes" else int(rng.integers(3, 5))
    else:
        number = 4 if parameters["shift"] == "lanes" and split == "test" else int(rng.integers(
            2, 4 if parameters["shift"] == "lanes" else 5))
    widths = [5] * number
    gaps = [1] * (number - 1)
    l = int(rng.integers(0, 3))
    margin = int(rng.integers(0, 3))
    budget = 30 - (sum(widths) + sum(gaps) + 4 + l + margin)
    # Four lanes need a little less margin; preserve requested network count.
    if budget < 0:
        margin = max(0, margin + budget)
        budget = 30 - (sum(widths) + sum(gaps) + 4 + l + margin)
    for i in rng.permutation(number):
        extra = 2 * int(rng.integers(0, min(2, budget // 2) + 1))
        widths[int(i)] += extra
        budget -= extra
    for i in rng.permutation(number - 1):
        extra = int(rng.integers(0, min(2, budget) + 1))
        gaps[int(i)] += extra
        budget -= extra
    depth = int(rng.integers(depth_range[0], depth_range[1] + 1))
    t = int(rng.integers(0, 3))
    b = t + depth
    width = sum(widths) + sum(gaps) + 4
    r = l + width - 1
    high, low = int(rng.integers(3, 5)), int(rng.integers(1, 3))
    if witness == "reversal":
        roles = ["reversal_still", "reversal_moving"]
        if rotation in (1, 2):
            roles.reverse()
        if number == 3:
            roles.insert(int(rng.integers(3)), "ordinary")
    elif witness == "tie_zero":
        low = int(rng.integers(1, 4))
        roles = ["zero", "tie_still", "tie_moving"] + ["ordinary"] * (number - 3)
        rng.shuffle(roles)
    else:
        roles = ["ordinary"] * number
    pistons = []
    for i, (pw, role) in enumerate(zip(widths, roles)):
        n = _piston(rng, depth, pw, role, high, low)
        pistons.append(n)
    hmax = max(n["height"] for n in pistons)
    e = b + hmax + int(rng.integers(1, 4)) + 1
    if e >= 30:
        e = 29
    bottom_margin = int(rng.integers(0, min(2, 29 - e) + 1))
    shape = (e + bottom_margin + 1, r + margin + 1)
    bg, s, noise = parameters["roles"]
    if parameters["local_colors"]:
        bg, s, noise = map(int, rng.choice(10, 3, replace=False))
    palette = [c for c in range(10) if c not in (bg, s, noise)]
    colors = list(map(int, rng.choice(palette, number, replace=False)))
    x0 = l + 2
    teeth, labels = [], []
    for i, n in enumerate(pistons):
        x1 = x0 + widths[i] - 1
        center = (x0 + x1) // 2
        n.update(color=colors[i], left=x0)
        teeth.append((center, n["height"]))
        # Independent analytic construction: a room survives iff every wall
        # row survives. Each room's top is its minimum wall row.
        survivor = sum(box[0] >= n["shift"] for box in n["boxes"])
        labels.append({"color": colors[i], "surviving": survivor,
                       "spatial": -x0 if rotation in (1, 2) else x0})
        if i + 1 < number:
            x0 = x1 + gaps[i] + 1
    if split == "test" and (max(n["surviving"] for n in labels) < 2
                            or all(n["surviving"] == len(p["boxes"])
                                   for n, p in zip(labels, pistons))):
        raise ValueError("query must expose both room loss and subdivision counting")
    g = np.full(shape, bg, dtype=np.int8)
    structure = _structure(shape, s, t, b, l, r, e, teeth)
    g[structure] = s
    occupied_boxes = np.zeros(shape, dtype=bool)
    for n in pistons:
        mask, x0 = n["mask"], n["left"]
        local = g[t + 1:t + 1 + mask.shape[0], x0:x0 + mask.shape[1]]
        local[mask] = n["color"]
        occupied_boxes[t + 1:t + 1 + mask.shape[0], x0:x0 + mask.shape[1]] = True
        candidates = _damage_sites(mask)
        rng.shuffle(candidates)
        limit = int(rng.integers(1, min(5, len(candidates)) + 1)) if candidates else 0
        if witness is None and rng.random() < 0.35:
            limit = 0
        chosen = []
        for y, x in candidates:
            if len(chosen) >= limit:
                break
            if all(abs(y - v) + abs(x - u) >= 3 for v, u in chosen):
                local[y, x] = noise
                chosen.append((y, x))
    # Sparse irrelevant noise is always outside piston boxes and mechanism.
    sites = np.argwhere((g == bg) & ~occupied_boxes & ~structure)
    amount = min(len(sites), int(rng.integers(2, 7)))
    for y, x in sites[rng.choice(len(sites), amount, replace=False)]:
        g[y, x] = noise
    output = _inventory(noise, labels)
    inp = np.rot90(g, rotation).copy()
    pair = {"input": inp, "output": output}
    # Reject accidental role ambiguity, not a change to semantics or labels.
    f = _read(inp)
    prediction = _inventory(f["noise"], _press_count(f))
    if not np.array_equal(prediction, output):
        raise ValueError("pixel reader and construction disagree")
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    ne = _count(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    parameters = {"roles": tuple(map(int, rng.choice(10, 3, replace=False))),
                  "local_colors": bool(rng.random() < 0.65),
                  "orientation": int(rng.integers(4)), "mixed": bool(rng.random() < 0.55),
                  "shift": str(rng.choice(("lanes", "depth")))}
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", nt), ("test", ne)):
        for index in range(number):
            witness = ("reversal", "tie_zero")[index] if split == "train" and index < 2 else None
            for _ in range(100):
                try:
                    pair = _make_pair(rng, parameters, split, witness)
                except ValueError:
                    continue
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction could not produce a distinct pair")
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _lattice_survivors(n):
    m = n["mask"]
    hs = np.flatnonzero(np.any(m[:, :-1] & m[:, 1:], axis=1))
    vs = np.flatnonzero(np.any(m[:-1] & m[1:], axis=0))
    return sum(y >= n["shift"] for y in hs[:-1]) * (len(vs) - 1)


def _features(pair):
    f = _read(pair["input"])
    ns = _press_count(f)
    assert np.array_equal(pair["output"], _inventory(f["noise"], ns))
    original = [{**n, "surviving": len(n["rooms"])} for n in ns]
    lattice = [{**n, "surviving": _lattice_survivors(n)} for n in ns]
    area = [{**n, "surviving": (n["box"][2] - n["box"][0] - 1)
                              * (n["box"][3] - n["box"][1] - 1)} for n in ns]
    one_per_piston = [{**n, "surviving": 1} for n in ns]
    moving = any(n["shift"] > 0 for n in ns)
    stationary = any(n["shift"] == 0 for n in ns)
    counts = [n["surviving"] for n in ns]
    ordering = sorted(ns, key=lambda n: n["spatial"])
    damage = any(np.any(n["mask"] & (f["grid"][n["box"][0]:n["box"][2] + 1,
                                                   n["box"][1]:n["box"][3] + 1] == f["noise"])) for n in ns)
    return f, ns, {"lanes": len(ns), "depth": f["dock"] - f["roof"],
                   "counts": tuple(counts), "original": tuple(len(n["rooms"]) for n in ns),
                   "heights": tuple(n["height"] for n in ns),
                   "shifts": tuple(n["shift"] for n in ns),
                   "moving": moving, "stationary": stationary,
                   "zero": 0 in counts, "tie": len(counts) != len(set(counts)),
                   "positive_tie_mixed": any(a["surviving"] == b["surviving"] > 0
                                             and a["shift"] > 0 == b["shift"] for a in ns for b in ns),
                   "damage": damage, "partial": any(_is_partial(n["mask"]) for n in ns),
                   "not_spatial": [n["surviving"] for n in ordering] != sorted(counts),
                   "no_A_differs": not np.array_equal(_inventory(f["noise"], original), pair["output"]),
                   "no_B_differs": not np.array_equal(_inventory(f["noise"], one_per_piston), pair["output"]),
                   "lattice_differs": not np.array_equal(_inventory(f["noise"], lattice), pair["output"])
                       if max(n["surviving"] for n in lattice) > 0 else True,
                   "area_differs": not np.array_equal(_inventory(f["noise"], area), pair["output"]),
                   "reversal": any(len(a["rooms"]) > len(b["rooms"])
                                   and a["surviving"] < b["surviving"] for a in ns for b in ns)}


def validate_generated_puzzle(puzzle):
    """Validate returned pixels, topology, contacts, labels, and evidence coverage."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, o = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen
            seen.add(_key(a))
            f, ns, features = _features(pair)
            assert o.shape[0] == len(ns) and o.shape[1] == max(n["surviving"] for n in ns)
            assert not np.array_equal(a, o)
            for n in ns:
                # A second room survival criterion using recovered original
                # wall boxes rather than counting the pressed mask.
                analytic = sum(room[0] - 1 >= n["shift"] for room in n["rooms"])
                assert analytic == n["surviving"]
                assert n["box"][2] - n["shift"] == f["dock"] - n["height"] - 1
                assert n["surviving"] <= len(n["rooms"])
                if n["shift"] == 0:
                    assert n["surviving"] == len(n["rooms"])
            records[split].append(features)
    train = records["train"]
    for flag in ("moving", "stationary", "zero", "tie", "partial", "damage",
                 "not_spatial", "no_A_differs", "no_B_differs", "lattice_differs", "area_differs", "reversal"):
        assert any(r[flag] for r in train), f"missing training evidence: {flag}"
    assert any(r["zero"] and r["positive_tie_mixed"] for r in train)
    assert all(r["no_A_differs"] and r["no_B_differs"] and max(r["counts"]) >= 2
               for r in records["test"])
    # Verify the visible train/test transfer rather than trusting a hidden tag.
    lane_transfer = all(r["lanes"] > max(t["lanes"] for t in train) for r in records["test"])
    depth_transfer = all(r["depth"] > max(t["depth"] for t in train) for r in records["test"])
    assert lane_transfer or depth_transfer, "no intended structural transfer"
    return True


def _puzzle_key(puzzle):
    return tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output"))


def _topology_signature(mask):
    # Remove metric room dimensions while retaining segment endpoints and
    # occupied elementary intervals. This is stronger than noisy scene hashes.
    hs = list(np.flatnonzero(np.any(mask[:, :-1] & mask[:, 1:], axis=1)))
    vs = list(np.flatnonzero(np.any(mask[:-1] & mask[1:], axis=0)))
    horizontal = tuple(tuple(bool(mask[y, x1:x2 + 1].all()) for x1, x2 in zip(vs, vs[1:])) for y in hs)
    vertical = tuple(tuple(bool(mask[y1:y2 + 1, x].all()) for x in vs) for y1, y2 in zip(hs, hs[1:]))
    return horizontal, vertical


def self_test():
    """Bounded 42-seed sweep, all counts, equivariance, and focused regressions."""
    signatures, shapes, topologies, heights, profiles = set(), set(), set(), set(), set()
    rotations, roles, lanes, depths = set(), set(), set(), set()
    partial = damage = zero = stationary = moving = reversal = 0
    lane_transfers = depth_transfers = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(seed))
        signatures.add(_puzzle_key(p))
        all_records = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in p[split]:
                f, ns, record = _features(pair)
                all_records[split].append(record)
                shapes.add(np.asarray(pair["input"]).shape)
                rotations.add(f["rotation"])
                roles.add((f["background"], f["structure"], f["noise"]))
                lanes.add(len(ns)); depths.add(record["depth"])
                profiles.add(record["heights"])
                heights.update(record["heights"])
                topologies.update(_topology_signature(n["mask"]) for n in ns)
                partial += record["partial"]; damage += record["damage"]
                zero += record["zero"]; reversal += record["reversal"]
                stationary += sum(n["shift"] == 0 for n in ns)
                moving += sum(n["shift"] > 0 for n in ns)
        tr, te = all_records["train"], all_records["test"]
        lane_transfers += all(r["lanes"] > max(t["lanes"] for t in tr) for r in te)
        depth_transfers += all(r["depth"] > max(t["depth"] for t in tr) for r in te)
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(topologies) >= 40 and len(profiles) >= 80
    assert lanes == {2, 3, 4} and rotations == {0, 1, 2, 3}
    assert heights == set(range(1, 7)) and len(depths) >= 6 and len(roles) >= 40
    assert min(partial, damage, zero, stationary, moving, reversal) >= 42
    assert lane_transfers >= 5 and depth_transfers >= 5
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(500 + 10 * nt + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (2.0, 1), (None, 1), (4, False), (4, 1.0)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Recoloring preserves visible roles. Reflecting a mechanism reverses
    # spatial tie order; compare through the analytic inventory, not a guessed
    # invariant output. A quarter turn can likewise reverse tie positions.
    for pair in generate_puzzle(123, 2, 1)["train"]:
        f = _read(pair["input"])
        for rotation in range(4):
            a = np.rot90(f["grid"], rotation).copy()
            labels = []
            for n in f["networks"]:
                labels.append({"color": n["color"],
                               "surviving": sum(r[0] - 1 >= n["shift"] for r in n["rooms"]),
                               "spatial": -n["box"][1] if rotation in (1, 2) else n["box"][1]})
            expected = _inventory(f["noise"], labels)
            assert np.array_equal(solve(a), expected)
            mapping = np.array([7, 2, 9, 0, 4, 8, 1, 6, 5, 3], dtype=np.int8)
            assert np.array_equal(solve(mapping[a]), mapping[expected])
        reflected = np.fliplr(f["grid"])
        labels = [{"color": n["color"],
                   "surviving": sum(r[0] - 1 >= n["shift"] for r in n["rooms"]),
                   "spatial": -n["box"][1]} for n in f["networks"]]
        assert np.array_equal(solve(reflected), _inventory(f["noise"], labels))
    # The roof is a deleting boundary, never a replacement piston wall.
    m = _walls((7, 5), [(0, 0, 2, 4), (2, 0, 6, 2), (2, 2, 6, 4)])
    assert len(_rooms(m)) == 3
    clipped = m[1:]
    assert len(_rooms(clipped)) == 2
    assert len(_rooms(m[3:])) == 0
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "scale_free_piston_topologies": len(topologies), "tooth_profiles": len(profiles),
            "role_palettes": len(roles), "lane_transfers": lane_transfers,
            "depth_transfers": depth_transfers}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
