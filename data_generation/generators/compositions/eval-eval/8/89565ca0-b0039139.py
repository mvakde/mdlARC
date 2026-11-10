"""Furnish recovered rooms with counted, recolored, wrapping stencil copies.

Only standard Python and NumPy are used. Generation writes no files, keeps one
puzzle at a time, and has finite geometry/uniqueness retries. See task_spec.md.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid shape")
    _require(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= g) & (g <= 9))), "invalid grid color")
    return g


def _components(mask, diagonal=False):
    """Deterministic components, each a sorted list of (row, column)."""
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    steps = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if diagonal:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    while remaining:
        point = min(remaining)
        remaining.remove(point)
        stack, group = [point], []
        while stack:
            r, c = stack.pop()
            group.append((r, c))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        groups.append(sorted(group))
    return groups


def _bounds(mask):
    cells = np.argwhere(mask)
    _require(len(cells) > 0, "empty shape")
    return tuple(map(int, (*cells.min(axis=0), *cells.max(axis=0))))


def _crop(mask):
    t, l, b, r = _bounds(mask)
    return mask[t:b + 1, l:r + 1].copy()


def _wall_mask(shape, boxes):
    result = np.zeros(shape, dtype=bool)
    for t, l, b, r in boxes:
        result[t, l:r + 1] = result[b, l:r + 1] = True
        result[t:b + 1, l] = result[t:b + 1, r] = True
    return result


def _rooms(walls):
    """Elementary interior rectangles, not products of line coordinates."""
    h, w = walls.shape
    _require(walls[0].all() and walls[-1].all() and walls[:, 0].all()
             and walls[:, -1].all(), "outer frame is not closed")
    result = []
    for cells in _components(~walls):
        a = np.asarray(cells)
        t, l = map(int, a.min(axis=0))
        b, r = map(int, a.max(axis=0))
        _require(len(cells) == (b - t + 1) * (r - l + 1), "nonrectangular room")
        _require(0 < t <= b < h - 1 and 0 < l <= r < w - 1, "room reaches outside")
        result.append((t, l, b, r))
    _require(bool(result), "no rooms")
    borders = [(t - 1, l - 1, b + 1, r + 1) for t, l, b, r in result]
    _require(np.array_equal(walls, _wall_mask(walls.shape, borders)), "unexplained walls")
    return sorted(result)


def _capacity(box, tile_shape):
    t, l, b, r = box
    th, tw = tile_shape
    rows = (b - t + 2) // (th + 1)
    cols = (r - l + 2) // (tw + 1)
    return rows, cols


def _decode(input_grid):
    """Infer roles and the complete floorplan from input pixels alone."""
    g = _grid(input_grid)
    candidates = []
    for wall in range(1, 9):
        if not np.any(g == wall):
            continue
        t, l, b, r = _bounds(g == wall)
        if b - t < 2 or r - l < 2:
            continue
        try:
            inside = np.zeros(g.shape, dtype=bool)
            inside[t:b + 1, l:r + 1] = True
            _require(not np.any((g == 9) & ~inside), "exterior damage")
            for y, x in np.argwhere(g == 9):
                horizontal = 0 < x < g.shape[1] - 1 and g[y, x - 1] == wall and g[y, x + 1] == wall
                vertical = 0 < y < g.shape[0] - 1 and g[y - 1, x] == wall and g[y + 1, x] == wall
                _require(horizontal or vertical, "unsupported wall repair")
            local = g[t:b + 1, l:r + 1]
            walls = (local == wall) | (local == 9)
            rooms = _rooms(walls)
            exterior = g[~inside]
            inks = set(map(int, np.unique(exterior))) - {0}
            _require(len(inks) == 1, "exterior needs one source ink")
            source = next(iter(inks))
            _require(source not in (wall, 9), "source role conflict")
            _require(not np.any(local == source), "source ink appears indoors")
            sm = g == source
            st, sl, sb, sr = _bounds(sm)
            _require(sb < t - 1 or st > b + 1 or sr < l - 1 or sl > r + 1,
                     "source crop must be separated outside the building")
            _require(len(_components(sm)) == 1, "source stencil is disconnected")
            tile = _crop(sm)
            room_data = []
            for box in rooms:
                a, c, d, e = box
                region = local[a:d + 1, c:e + 1]
                colors = set(map(int, np.unique(region))) - {0}
                _require(len(colors) <= 1 and not (colors & {wall, source, 9}), "mixed token colors")
                color = next(iter(colors)) if colors else 0
                groups = _components(region != 0)
                _require(len(groups) == len(_components(region != 0, True)), "diagonal token contact")
                nr, nc = _capacity(box, tile.shape)
                _require(nr >= 1 and nc >= 1, "stencil cannot fit room")
                _require(len(groups) <= nr * nc, "too many tokens for room")
                room_data.append({"box": box, "color": color, "count": len(groups),
                                  "area": int(np.count_nonzero(region)), "groups": groups,
                                  "rows": nr, "cols": nc})
            _require(any(d["count"] for d in room_data), "all rooms are empty")
            candidates.append({"wall": wall, "source": source, "tile": tile,
                               "walls": walls, "rooms": room_data,
                               "box": (t, l, b, r), "local": local.copy()})
        except AssertionError:
            continue
    _require(len(candidates) == 1, "input needs one unique building/source parse")
    return candidates[0]


def _render(walls, rooms, tile, wall):
    """Each room has its own row-major lattice, reset at its upper-left cell."""
    out = np.zeros(walls.shape, dtype=np.int8)
    out[walls] = wall
    th, tw = tile.shape
    for d in rooms:
        t, l, _, _ = d["box"]
        for i in range(d["count"]):
            row, col = divmod(i, d["cols"])
            y, x = t + row * (th + 1), l + col * (tw + 1)
            patch = out[y:y + th, x:x + tw]
            patch[tile] = d["color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Repair, segment, count, crop, recolor, wrap; no generation state."""
    d = _decode(input_grid)
    return _render(d["walls"], d["rooms"], d["tile"], d["wall"])


def _transform(mask, k):
    return np.rot90(mask[:, ::-1] if k >= 4 else mask, k % 4)


def _stencil(rng, h, w, asymmetric=False):
    for _ in range(24):
        mask = np.ones((h, w), dtype=bool)
        target = max(h + w - 1, int(rng.uniform(.35, .90) * h * w))
        if not asymmetric and rng.random() < .12:
            target = h * w
        for flat in rng.permutation(h * w):
            if mask.sum() <= target:
                break
            y, x = divmod(int(flat), w)
            mask[y, x] = False
            if not mask.any() or _crop(mask).shape != (h, w) or len(_components(mask)) != 1:
                mask[y, x] = True
        if not asymmetric or all(not np.array_equal(mask, _transform(mask, k)) for k in range(1, 8)):
            return mask
    # A finite asymmetric safety shape; only requested dimensions >=3 use it.
    return np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool)


def _partition(rng, h, w, number, tile_shape, partial=False):
    th, tw = tile_shape
    boxes = [(0, 0, h - 1, w - 1)]
    first_axis = None
    for step in range(number - 1):
        options = []
        for i, (t, l, b, r) in enumerate(boxes):
            for axis, start, end, minimum in ((0, t, b, th), (1, l, r, tw)):
                if partial and step == 1 and axis == first_axis:
                    continue
                for cut in range(start + minimum + 1, end - minimum):
                    halves = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else ((t, l, b, cut), (t, cut, b, r))
                    options.append((i, axis, halves))
        if not options:
            return None
        i, axis, halves = options[int(rng.integers(len(options)))]
        if step == 0:
            first_axis = axis
        boxes[i:i + 1] = halves
    return _wall_mask((h, w), boxes)


def _partial(walls):
    # True internal T-junctions. Rooms have minimum interior size >=1.
    arms = np.zeros(walls.shape, dtype=int)
    arms[1:] += walls[:-1]
    arms[:-1] += walls[1:]
    arms[:, 1:] += walls[:, :-1]
    arms[:, :-1] += walls[:, 1:]
    return bool(np.any(walls[1:-1, 1:-1] & (arms[1:-1, 1:-1] == 3)))


def _safety_partition(number, tile_shape, more_copies=False):
    """Finite 23x23 layouts that fit each requested tile through side length 5."""
    if number == 1:
        boxes = [(0, 0, 22, 22)]
    elif number in (2, 3):
        cut = 10 if more_copies else tile_shape[1] + 2
        ys = [22 * i // (number - 1) for i in range(number)]
        boxes = [(0, 0, 22, cut)] + [(ys[i], cut, ys[i + 1], 22) for i in range(number - 1)]
    elif number == 4:
        boxes = [(a, c, b, e) for a, b in ((0, 11), (11, 22)) for c, e in ((0, 11), (11, 22))]
    else:
        _require(number in (5, 6), "unsupported safety room count")
        boxes = [(a, c, b, e) for a, b in ((0, 11), (11, 22))
                 for c, e in ((0, 7), (7, 14), (14, 22))]
        if number == 5:
            boxes = [b for b in boxes if b[1] != 0] + [(0, 0, 22, 7)]
    return _wall_mask((23, 23), boxes)


def _grow_object(rng, h, w, area):
    mask = np.zeros((h, w), dtype=bool)
    frontier = {(int(rng.integers(h)), int(rng.integers(w)))}
    for _ in range(area):
        choices = sorted(frontier)
        y, x = choices[int(rng.integers(len(choices)))]
        frontier.remove((y, x))
        mask[y, x] = True
        for a, b in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= a < h and 0 <= b < w and not mask[a, b]:
                frontier.add((a, b))
    return mask


def _tokens(rng, h, w, n, tile_shape, area=None):
    """Arbitrary positions and shapes, with eight-neighbor separation.

    A random independent set is grown while protecting other labels. The
    deterministic spaced fallback has ample capacity because tile dims >=2
    for requested multi-object evidence. No grids are retained across retries.
    """
    if n == 0:
        return np.zeros((h, w), dtype=bool)
    if n == 1:
        size = int(rng.integers(1, min(h * w, 16) + 1)) if area is None else area
        return _grow_object(rng, h, w, size)
    desired = n + int(rng.integers(0, 2 * n + 1)) if area is None else area
    for attempt in range(16):
        labels = np.zeros((h, w), dtype=int)
        if attempt == 15:
            choices = [(y, x) for y in range(0, h, 2) for x in range(0, w, 2)]
            rng.shuffle(choices)
        else:
            choices = [divmod(int(k), w) for k in rng.permutation(h * w)]
        used = 0
        for y, x in choices:
            if np.any(labels[max(0, y - 1):y + 2, max(0, x - 1):x + 2]):
                continue
            used += 1
            labels[y, x] = used
            if used == n:
                break
        if used != n:
            continue
        for _ in range(max(0, desired - n)):
            options = []
            for y in range(h):
                for x in range(w):
                    if labels[y, x]:
                        continue
                    neighbors = set(labels[max(0, y - 1):y + 2, max(0, x - 1):x + 2].flat) - {0}
                    if len(neighbors) != 1:
                        continue
                    label = next(iter(neighbors))
                    if any(0 <= a < h and 0 <= b < w and labels[a, b] == label
                           for a, b in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))):
                        options.append((y, x, label))
            if not options:
                break
            y, x, label = options[int(rng.integers(len(options)))]
            labels[y, x] = label
        if area is None or np.count_nonzero(labels) == area:
            return labels != 0
    # Guaranteed exact n+1 evidence: one domino and n-1 singletons, using
    # separated tile-sized cells. Ordinary draws may safely use singleton ink.
    th, tw = tile_shape
    nc = (w + 1) // (tw + 1)
    out = np.zeros((h, w), dtype=bool)
    for i in range(n):
        row, col = divmod(i, nc)
        out[row * (th + 1), col * (tw + 1)] = True
    if area is not None:
        _require(area == n + 1 and tw >= 2, "unsupported token fallback")
        out[0, 1] = True
    return out


def _embed(rng, building, tile, source):
    """The source can occupy any exterior side; there are no panels."""
    h, w = building.shape
    th, tw = tile.shape
    side = int(rng.integers(4))
    available = 30 - (w + tw if side < 2 else h + th)
    gap = int(rng.integers(1, min(3, available) + 1))
    if side < 2:
        canvas = np.zeros((h, w + gap + tw), dtype=np.int8)
        bx = 0 if side == 0 else tw + gap
        sx = w + gap if side == 0 else 0
        sy = int(rng.integers(h - th + 1))
        by = 0
    else:
        canvas = np.zeros((h + gap + th, w), dtype=np.int8)
        by = 0 if side == 2 else th + gap
        sy = h + gap if side == 2 else 0
        sx = int(rng.integers(w - tw + 1))
        bx = 0
    canvas[by:by + h, bx:bx + w] = building
    canvas[sy:sy + th, sx:sx + tw][tile] = source
    padding = []
    for length in canvas.shape:
        extra = int(rng.integers(min(4, 30 - length) + 1))
        first = int(rng.integers(extra + 1))
        padding.append((first, extra - first))
    return np.pad(canvas, padding, constant_values=0)


def _make_pair(rng, *, witness=False, shift=None, high_count=0):
    if shift == "motif":
        th, tw = (5, int(rng.integers(2, 5))) if rng.random() < .5 else (int(rng.integers(2, 5)), 5)
    elif witness:
        th, tw = int(rng.integers(3, 5)), int(rng.integers(3, 5))
    elif shift == "copies":
        th, tw = int(rng.integers(2, 4)), 2
    else:
        th, tw = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        if rng.random() < .16:
            thin_mode = int(rng.integers(3))
            if thin_mode == 0:
                th = 1
            elif thin_mode == 1:
                tw = 1
            else:
                th = tw = 1
    tile = _stencil(rng, th, tw, asymmetric=witness)
    th, tw = tile.shape
    number = int(rng.integers(5, 7)) if shift == "rooms" else int(rng.integers(1, 5))
    if witness or shift == "copies":
        number = 3
    walls = None
    for _ in range(64):
        low_h = 16 if witness else th + 2
        low_w = 16 if witness else tw + 2
        h, w = int(rng.integers(low_h, 24)), int(rng.integers(low_w, 24))
        candidate = _partition(rng, h, w, number, tile.shape, partial=witness or shift == "rooms")
        if candidate is None:
            continue
        boxes = _rooms(candidate)
        caps = [_capacity(b, tile.shape) for b in boxes]
        if witness and (not _partial(candidate) or not any(nr * nc >= nc + 2 for nr, nc in caps)):
            continue
        if witness and not any(((b[0] - 1) % (th + 1) or (b[1] - 1) % (tw + 1))
                               and nr * nc >= 1 for b, (nr, nc) in zip(boxes, caps)):
            continue
        if shift == "copies" and max(nr * nc for nr, nc in caps) < high_count:
            continue
        walls = candidate
        break
    if walls is None:
        walls = _safety_partition(number, tile.shape, shift == "copies")
        boxes = _rooms(walls)
        caps = [_capacity(b, tile.shape) for b in boxes]
    wall, source = map(int, rng.choice(np.arange(1, 9), 2, replace=False))
    palette = [c for c in range(1, 9) if c not in (wall, source)]
    ink_pool = list(map(int, rng.choice(palette, int(rng.integers(2, 5)), replace=False)))
    rooms = []
    for box, (nr, nc) in zip(boxes, caps):
        n = int(rng.integers(0, min(nr * nc, 6) + 1))
        rooms.append({"box": box, "rows": nr, "cols": nc, "count": n,
                      "color": int(rng.choice(ink_pool)), "wanted_area": None})
    if witness:
        eligible = [i for i, d in enumerate(rooms) if d["rows"] * d["cols"] >= d["cols"] + 2]
        i = int(rng.choice(eligible))
        others = [j for j in range(len(rooms)) if j != i]
        j = int(rng.choice(others))
        n = rooms[i]["cols"] + 1
        for d in rooms:
            d["count"] = 0
        rooms[i].update(count=n, wanted_area=n + 1, color=ink_pool[0])
        rooms[j].update(count=1, wanted_area=n + 1, color=ink_pool[1])
        # Ensure an active room visibly resets the stencil phase. If both active
        # choices happened to align, move the count-one witness to an unaligned
        # room; the wrapping and equal-area evidence are preserved.
        def unaligned(d):
            return (d["box"][0] - 1) % (th + 1) or (d["box"][1] - 1) % (tw + 1)
        if not (unaligned(rooms[i]) or unaligned(rooms[j])):
            k = next(k for k in others if unaligned(rooms[k]))
            rooms[j]["count"] = 0
            rooms[k].update(count=1, wanted_area=n + 1, color=ink_pool[1])
    elif shift == "copies":
        eligible = [i for i, d in enumerate(rooms) if d["rows"] * d["cols"] >= high_count]
        rooms[int(rng.choice(eligible))]["count"] = high_count
    if not any(d["count"] for d in rooms):
        rooms[0]["count"] = 1
    building = np.zeros(walls.shape, dtype=np.int8)
    building[walls] = wall
    for d in rooms:
        t, l, b, r = d["box"]
        mask = _tokens(rng, b - t + 1, r - l + 1, d["count"], tile.shape, d["wanted_area"])
        building[t:b + 1, l:r + 1][mask] = d["color"]
    output = _render(walls, rooms, tile, wall)
    # Isolated holes only, supported by two surviving collinear wall neighbors.
    damage_sites = []
    for y, x in np.argwhere(walls):
        arms = [(int(y) + dy, int(x) + dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))]
        hits = [(a, b) for a, b in arms if 0 <= a < walls.shape[0] and 0 <= b < walls.shape[1] and walls[a, b]]
        if len(hits) == 2 and (hits[0][0] == hits[1][0] or hits[0][1] == hits[1][1]):
            damage_sites.append((int(y), int(x)))
    rng.shuffle(damage_sites)
    amount = int(rng.integers(1 if witness else 0, 7))
    damaged = []
    for y, x in damage_sites:
        if len(damaged) == amount:
            break
        if all(abs(y - a) + abs(x - b) > 2 for a, b in damaged):
            building[y, x] = 9
            damaged.append((y, x))
    inp = _embed(rng, building, tile, source)
    return {"input": inp, "output": output}


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError("counts must be integers: 2..10 train, 1..4 test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(g):
    return tuple(g.shape), np.asarray(g, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(["rooms", "motif", "copies"]))
    puzzle, seen = {"train": [], "test": []}, set()
    high_count = 0
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            for _ in range(12):
                pair = _make_pair(rng, witness=split == "train" and index < 2,
                                  shift=shift if split == "test" else None,
                                  high_count=high_count + 1)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise AssertionError("bounded uniqueness retries exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                high_count = max(high_count, max(d["count"] for d in _decode(pair["input"])["rooms"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _audit_pair(pair):
    inp, out = (_grid(pair[k]) for k in ("input", "output"))
    d = _decode(inp)
    _require(out.shape == d["walls"].shape, "output is not the cropped building")
    _require(np.array_equal(out == d["wall"], d["walls"]), "output walls differ")
    th, tw = d["tile"].shape
    # Per-cell modular check differs from the constructor's copy concatenation.
    for room in d["rooms"]:
        t, l, b, r = room["box"]
        for y in range(t, b + 1):
            for x in range(l, r + 1):
                row, iy = divmod(y - t, th + 1)
                col, ix = divmod(x - l, tw + 1)
                i = row * room["cols"] + col
                filled = (col < room["cols"] and row < room["rows"] and i < room["count"]
                          and iy < th and ix < tw and d["tile"][iy, ix])
                expected = room["color"] if filled else 0
                _require(int(out[y, x]) == expected, "incorrect local stamp placement")
        region = out[t:b + 1, l:r + 1]
        _require(np.count_nonzero(region) == room["count"] * int(d["tile"].sum()), "ink conservation")
        _require(len(_components(region != 0)) == room["count"], "copy count differs from token count")
    _require(np.array_equal(solve(inp), out), "solve disagreement")
    return d


def _ablation_witnesses(d):
    """Compare furnished pixels, holding repaired walls and local inks fixed."""
    correct = _render(d["walls"], d["rooms"], d["tile"], d["wall"])
    global_lattice = np.zeros_like(correct)
    global_lattice[d["walls"]] = d["wall"]
    th, tw = d["tile"].shape
    for room in d["rooms"]:
        t, l, b, r = room["box"]
        slots = [(y, x) for y in range(1, correct.shape[0] - th, th + 1)
                 for x in range(1, correct.shape[1] - tw, tw + 1)
                 if t <= y and l <= x and y + th - 1 <= b and x + tw - 1 <= r]
        for y, x in slots[:room["count"]]:
            global_lattice[y:y + th, x:x + tw][d["tile"]] = room["color"]
    area_rooms = [dict(room, count=min(room["area"], room["rows"] * room["cols"]))
                  for room in d["rooms"]]
    pixel_counter = _render(d["walls"], area_rooms, d["tile"], d["wall"])
    solid_stencil = _render(d["walls"], d["rooms"], np.ones_like(d["tile"]), d["wall"])
    return {"local_reset": not np.array_equal(global_lattice, correct),
            "area_ablation": not np.array_equal(pixel_counter, correct),
            "stencil_ablation": not np.array_equal(solid_stencil, correct)}


def _evidence(ds):
    all_rooms = [r for d in ds for r in d["rooms"]]
    result = {
        "partial": any(_partial(d["walls"]) for d in ds),
        "repair": any(np.any(d["local"] == 9) for d in ds),
        "wrap": any(r["count"] > r["cols"] for r in all_rooms),
        "slack": any(0 < r["count"] < r["rows"] * r["cols"] for r in all_rooms),
        "empty": any(r["count"] == 0 for r in all_rooms),
        "color_binding": any(len({r["color"] for r in d["rooms"] if r["count"]}) >= 2 for d in ds),
        "equal_area_different_counts": any(a["area"] == b["area"] > 0 and a["count"] != b["count"]
                                           for d in ds for a in d["rooms"] for b in d["rooms"]),
        "asymmetric": all(any(not np.array_equal(d["tile"], _transform(d["tile"], k)) for d in ds)
                          for k in range(1, 8)),
    }
    ablations = [_ablation_witnesses(d) for d in ds]
    result.update({k: any(a[k] for a in ablations) for k in ablations[0]})
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp = _grid(pair["input"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            analyses[split].append(_audit_pair(pair))
    evidence = _evidence(analyses["train"])
    _require(all(evidence.values()), "missing training evidence: " + str(evidence))
    tr, te = analyses["train"], analyses["test"]
    more_rooms = min(len(d["rooms"]) for d in te) > max(len(d["rooms"]) for d in tr)
    bigger_tile = min(max(d["tile"].shape) for d in te) > max(max(d["tile"].shape) for d in tr)
    more_copies = min(max(r["count"] for r in d["rooms"]) for d in te) > max(r["count"] for d in tr for r in d["rooms"])
    _require(more_rooms or bigger_tile or more_copies, "missing visible train/test shift")


def self_test():
    """42 streamed seeds; no files, traces, journals, indexes or backups."""
    stats = {"seeds": 42, "pairs": 0, "puzzles": set(), "input_shapes": set(),
             "output_shapes": set(), "tiles": set(), "room_layouts": set(),
             "room_counts": set(), "token_counts": set(), "source_sides": set(),
             "palettes": set(), "wrapped_rooms": 0, "empty_rooms": 0,
             "damaged_pairs": 0, "partial_pairs": 0}
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                d = _audit_pair(pair)
                stats["pairs"] += 1
                digest.update(bytes(pair["input"].shape) + pair["input"].tobytes())
                stats["input_shapes"].add(pair["input"].shape)
                stats["output_shapes"].add(pair["output"].shape)
                stats["tiles"].add(_key(d["tile"]))
                stats["room_layouts"].add(tuple(r["box"] for r in d["rooms"]))
                stats["room_counts"].add(len(d["rooms"]))
                stats["token_counts"].update(r["count"] for r in d["rooms"])
                stats["palettes"].add((d["wall"], d["source"]))
                stats["wrapped_rooms"] += sum(r["count"] > r["cols"] for r in d["rooms"])
                stats["empty_rooms"] += sum(r["count"] == 0 for r in d["rooms"])
                stats["damaged_pairs"] += bool(np.any(d["local"] == 9))
                stats["partial_pairs"] += _partial(d["walls"])
                t, l, b, r = d["box"]
                st, sl, sb, sr = _bounds(pair["input"] == d["source"])
                stats["source_sides"].add("top" if sb < t else "bottom" if st > b else "left" if sr < l else "right")
        stats["puzzles"].add(digest.hexdigest())
        if seed in (0, 7, 41):
            again = generate_puzzle(seed)
            _require(all(_key(a[k]) == _key(b[k]) for s in p for a, b in zip(p[s], again[s]) for k in a), "nondeterminism")
    _require(len(stats["puzzles"]) == 42, "duplicate puzzles across seeds")
    _require(len(stats["input_shapes"]) >= 45 and len(stats["output_shapes"]) >= 35, "shape diversity")
    _require(len(stats["tiles"]) >= 80 and len(stats["room_layouts"]) >= 100, "structural diversity")
    _require({3, 4, 5, 6} <= stats["room_counts"] and set(range(7)) <= stats["token_counts"], "count coverage")
    _require(len(stats["source_sides"]) == 4 and len(stats["palettes"]) >= 35, "role/layout coverage")
    _require(stats["wrapped_rooms"] >= 84 and stats["empty_rooms"] >= 84, "packing coverage")
    # Exercise every safety layout directly; random sampling need not reach it.
    for h in range(1, 6):
        for w in range(1, 6):
            for number in range(1, 7):
                walls = _safety_partition(number, (h, w))
                rooms = _rooms(walls)
                _require(len(rooms) == number, "safety room count")
                _require(all(min(_capacity(b, (h, w))) >= 1 for b in rooms), "safety capacity")
    # Input-only boundary case: edge frame, 1x1 source, empty room, a token bar.
    hand = np.zeros((5, 11), dtype=np.int8)
    hand[:5, :7] = np.where(_safety_small_walls(), 2, 0)
    hand[1, 1:3] = 4
    hand[2, 9] = 7
    hand[0, 1] = 9
    expected = np.zeros((5, 7), dtype=np.int8)
    expected[_safety_small_walls()] = 2
    expected[1, 1] = 4
    _require(np.array_equal(solve(hand), expected), "edge/one-cell/empty boundary case")
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(93, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3., 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Meaningful mutation: the validator must reject a wrong output pixel.
    p = generate_puzzle(0, 2, 1)
    p["test"][0]["output"][1, 1] = (int(p["test"][0]["output"][1, 1]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    # Palette permutations preserve semantics if fixed background/damage stay fixed.
    p = generate_puzzle(71, 2, 1)
    perm = np.array([0, 3, 6, 1, 8, 2, 7, 4, 5, 9], dtype=np.int8)
    for pair in p["train"] + p["test"]:
        _require(np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]]), "palette equivariance")
    return {k: len(v) if isinstance(v, set) else int(v) for k, v in stats.items()}


def _safety_small_walls():
    return _wall_mask((5, 7), [(0, 0, 4, 3), (0, 3, 4, 6)])


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
