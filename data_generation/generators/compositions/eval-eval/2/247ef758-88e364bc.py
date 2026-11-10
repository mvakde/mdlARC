"""Rigid stencil constellations: coordinate replication with shared clearance.

The public solver reads one input only. Generation uses per-pixel clearances;
prediction moves a reconstructed union of masks. No parent code is imported.
"""
from itertools import product

import numpy as np


DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
CARDINAL = tuple(d for d in DIRECTIONS if 0 in d)
DIAGONAL = tuple(d for d in DIRECTIONS if 0 not in d)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(np.all((0 <= a) & (a <= 9)), "colors outside 0..9")
    return a.astype(np.int8, copy=False)


def _fingerprint(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _puzzle_fingerprint(puzzle):
    return tuple(tuple((_fingerprint(p["input"]), _fingerprint(p["output"]))
                       for p in puzzle[s]) for s in ("train", "test"))


def _bbox(cells):
    a = np.asarray(list(cells), dtype=int)
    return tuple(map(int, a.min(axis=0))), tuple(map(int, a.max(axis=0)))


def _components(mask, diagonal=False):
    todo = set(map(tuple, np.argwhere(mask)))
    steps = DIRECTIONS if diagonal else CARDINAL
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, found = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
                    found.add(p)
        result.append(found)
    return result


def _decode_glyph(sub, carrier):
    """Infer polarity from complete repeated strokes, without a color dictionary."""
    others = set(map(int, np.unique(sub))) - {carrier}
    if len(others) != 2:
        return []
    interpretations = []
    for tip in sorted(others):
        trail = next(iter(others - {tip}))
        tips = set(map(tuple, np.argwhere(sub == tip)))
        trails = set(map(tuple, np.argwhere(sub == trail)))
        if len(tips) < 2:
            continue
        for dr, dc in DIRECTIONS:
            used, lengths = set(), []
            for r, c in sorted(tips):
                stroke = []
                p = r - dr, c - dc
                while p in trails:
                    stroke.append(p)
                    p = p[0] - dr, p[1] - dc
                if len(stroke) < 2 or used.intersection(stroke):
                    break
                used.update(stroke)
                lengths.append(len(stroke))
            if len(lengths) == len(tips) and used == trails:
                interpretations.append((tip, trail, (dr, dc), tuple(lengths)))
    return interpretations


def _parse(input_grid):
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(sum(counts == counts[bg]) == 1, "background must be uniquely most frequent")
    components = _components(inp != bg)
    boards = []
    for cells in components:
        (r0, c0), (r1, c1) = _bbox(cells)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if min(h, w) < 9:
            continue
        sub = inp[r0:r1 + 1, c0:c1 + 1]
        wall = int(sub[0, 0])
        if wall == bg or any(int(sub[r, c]) != wall
                             for r, c in ((0, -1), (-1, 0), (-1, -1))):
            continue
        if not (np.array_equal(sub[0], sub[-1]) and np.array_equal(sub[:, 0], sub[:, -1])):
            continue
        edge = np.concatenate((sub[0], sub[-1], sub[:, 0], sub[:, -1]))
        if np.any(edge == bg) or not np.all(np.isin(sub[1:-1, 1:-1], [bg, wall])):
            continue
        rows, cols = {}, {}
        for r in range(1, h - 1):
            color = int(sub[r, 0])
            if color != wall:
                rows.setdefault(color, []).append(r)
        for c in range(1, w - 1):
            color = int(sub[0, c])
            if color != wall:
                cols.setdefault(color, []).append(c)
        if rows and set(rows) == set(cols):
            boards.append((r0, c0, h, w, wall, rows, cols))
    _require(len(boards) == 1, "one uniquely identifiable coordinate frame required")
    r0, c0, h, w, wall, rows, cols = boards[0]
    board_box = (r0, c0, h, w)
    excluded = np.zeros(inp.shape, bool)
    excluded[r0:r0 + h, c0:c0 + w] = True
    keys = []
    for cells in components:
        if any(excluded[p] for p in cells):
            continue
        (a, b), (z, y) = _bbox(cells)
        kh, kw = z - a + 1, y - b + 1
        if min(kh, kw) < 4 or len(cells) != kh * kw:
            continue
        sub = inp[a:z + 1, b:y + 1]
        edge = np.concatenate((sub[0], sub[-1], sub[:, 0], sub[:, -1]))
        if np.any(edge != wall):
            continue
        decoded = _decode_glyph(sub, wall)
        if len(decoded) == 1:
            tip, trail, direction, lengths = decoded[0]
            keys.append(dict(box=(a, b, kh, kw), tip=tip, trail=trail,
                             direction=direction, lengths=lengths))
    _require(len(keys) == 1, "one unambiguous wall-carried direction key required")
    key = keys[0]
    a, b, kh, kw = key["box"]
    excluded[a:a + kh, b:b + kw] = True
    remaining = (inp != bg) & ~excluded
    colors = set(map(int, np.unique(inp[remaining])))
    _require(colors == set(rows), "every stencil needs both axes; no unrelated exterior marks")
    _require(not colors.intersection({bg, wall, key["tip"], key["trail"]}), "local roles must differ")
    templates, copies, formation = [], [], {}
    for color in sorted(colors):
        cells = set(map(tuple, np.argwhere(remaining & (inp == color))))
        (lo_r, lo_c), (hi_r, hi_c) = _bbox(cells)
        bh, bw = hi_r - lo_r + 1, hi_c - lo_c + 1
        _require(bh % 2 == bw % 2 == 1, "stencils require odd tight bounding boxes")
        mask = (remaining[lo_r:hi_r + 1, lo_c:hi_c + 1]
                & (inp[lo_r:hi_r + 1, lo_c:hi_c + 1] == color))
        _require(len(_components(mask, diagonal=True)) == 1, "one 8-connected stencil per color")
        center = ((lo_r + hi_r) // 2, (lo_c + hi_c) // 2)
        offsets = {(r - center[0], c - center[1]) for r, c in cells}
        t = dict(color=color, cells=cells, offsets=offsets, mask=mask,
                 rows=rows[color], cols=cols[color], box=(lo_r, lo_c, bh, bw))
        templates.append(t)
        for r, c in product(rows[color], cols[color]):
            launched = {(r + dr, c + dc) for dr, dc in offsets}
            _require(all(1 <= y < h - 1 and 1 <= x < w - 1 for y, x in launched),
                     "launch pixels must fit inside the coordinate frame")
            for p in launched:
                _require(p not in formation or formation[p] == color, "different-color launch collision")
                formation[p] = color
            copies.append(dict(color=color, cells=launched, center=(r, c),
                               shape=(bh, bw)))
    walls = inp[r0:r0 + h, c0:c0 + w] == wall
    walls[[0, -1], :] = True
    walls[:, [0, -1]] = True  # Coordinate labels also belong to the boundary.
    _require(all(not walls[p] for p in formation), "an occupied launch pixel intersects a wall")
    return dict(input=inp, bg=bg, wall=wall, board=board_box, walls=walls,
                key=key, direction=key["direction"], templates=templates,
                copies=copies, formation=formation)


def _free(walls, cells):
    h, w = walls.shape
    return all(0 <= r < h and 0 <= c < w and not walls[r, c] for r, c in cells)


def _translate(cells, delta):
    dr, dc = delta
    return {(r + dr, c + dc) for r, c in cells}


def _common_distance(walls, cells, direction, corner=True):
    """Solver: move the entire union, testing actual footprints at each step."""
    current = set(cells)
    if not _free(walls, current):
        return 0  # Used for deliberately wrong bounding-box/center hypotheses.
    dr, dc = direction
    for distance in range(31):
        following = _translate(current, direction)
        legal = _free(walls, following)
        if legal and corner and dr and dc:
            legal = (_free(walls, _translate(current, (dr, 0)))
                     and _free(walls, _translate(current, (0, dc))))
        if not legal:
            return distance
        current = following
    raise AssertionError("closed frame failed to stop motion")


def _paint(info, colored_cells):
    out = info["input"].copy()
    for t in info["templates"]:
        for p in t["cells"]:
            out[p] = info["bg"]
    r0, c0, _, _ = info["board"]
    for (r, c), color in colored_cells.items():
        out[r0 + r, c0 + c] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode virtual Cartesian copies and slide their union as one rigid body."""
    info = _parse(input_grid)
    distance = _common_distance(info["walls"], info["formation"], info["direction"])
    dr, dc = info["direction"]
    final = {(r + distance * dr, c + distance * dc): color
             for (r, c), color in info["formation"].items()}
    return _paint(info, final)


def _pixel_clearance(walls, start, direction, corner=True):
    """Construction: the distance of one occupied pixel before its first wall."""
    r, c = start
    dr, dc = direction
    h, w = walls.shape
    distance = 0
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w) or walls[nr, nc]:
            return distance
        if corner and dr and dc and (walls[nr, c] or walls[r, nc]):
            return distance
        r, c = nr, nc
        distance += 1


def _feature_info(info):
    walls, formation, direction = info["walls"], info["formation"], info["direction"]
    distance = _common_distance(walls, formation, direction)
    copy_distances = [_common_distance(walls, cp["cells"], direction) for cp in info["copies"]]
    color_distances = [_common_distance(walls, {p for p, col in formation.items() if col == t["color"]},
                                        direction) for t in info["templates"]]
    box_union = set()
    for cp in info["copies"]:
        r, c = cp["center"]
        h, w = cp["shape"]
        box_union.update((r + y, c + x) for y in range(-(h // 2), h // 2 + 1)
                         for x in range(-(w // 2), w // 2 + 1))
    box_distance = _common_distance(walls, box_union, direction)
    centers = {cp["center"] for cp in info["copies"]}
    center_distance = _common_distance(walls, centers, direction)
    no_corner = _common_distance(walls, formation, direction, corner=False)
    return dict(distance=distance, coupling=any(d > distance for d in copy_distances),
                color_coupling=any(d > distance for d in color_distances),
                copy_distances=copy_distances, color_distances=color_distances,
                corner=no_corner != distance,
                mask_vs_box=box_distance != distance, mask_vs_center=center_distance != distance,
                hole_wall=not _free(walls, box_union),
                products=tuple((len(t["rows"]), len(t["cols"])) for t in info["templates"]),
                count=len(info["templates"]), direction=direction,
                shapes=tuple(t["mask"].shape for t in info["templates"]),
                empty_center=any(not t["mask"][t["mask"].shape[0] // 2,
                                                    t["mask"].shape[1] // 2]
                                 for t in info["templates"]),
                merge=sum(len(cp["cells"]) for cp in info["copies"]) - len(formation))


def _alternative(info, kind):
    """Named shortcut predictions used only for evidence and mutation tests."""
    walls, direction = info["walls"], info["direction"]
    formation = dict(info["formation"])
    if kind == "zip":
        formation = {}
        for t in info["templates"]:
            for r, c in zip(t["rows"], t["cols"]):
                for dr, dc in t["offsets"]:
                    formation[r + dr, c + dc] = t["color"]
    if kind == "individual":
        final = {}
        for cp in info["copies"]:
            d = _common_distance(walls, cp["cells"], direction)
            for r, c in sorted(cp["cells"]):
                final[r + d * direction[0], c + d * direction[1]] = cp["color"]
        return _paint(info, final)
    if kind == "by_color":
        final = {}
        for t in info["templates"]:
            cells = {p for p, color in formation.items() if color == t["color"]}
            d = _common_distance(walls, cells, direction)
            for r, c in sorted(cells):
                final[r + d * direction[0], c + d * direction[1]] = t["color"]
        return _paint(info, final)
    if kind == "stationary":
        d = 0
    else:
        if kind == "reverse":
            direction = -direction[0], -direction[1]
        d = _common_distance(walls, formation, direction, corner=kind != "no_corner")
    return _paint(info, {(r + d * direction[0], c + d * direction[1]): color
                         for (r, c), color in formation.items()})


def _mask(rng, h, w, aperture=False):
    mask = np.zeros((h, w), bool)
    mode = 1 if aperture else int(rng.integers(7))
    if min(h, w) == 1 or mode == 0:
        mask[:] = True
    elif mode == 1:
        mask[[0, -1], :] = True
        mask[:, [0, -1]] = True
        if aperture:
            # Corners can disappear without changing the tight odd box.
            for r, c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
                if rng.random() < .35:
                    mask[r, c] = False
    elif mode == 2 and h == w:
        yy, xx = np.indices((h, w))
        mask = abs(yy - h // 2) + abs(xx - w // 2) == h // 2
    elif mode == 3:
        for r in range(h):
            mask[r, round(r * (w - 1) / (h - 1))] = True
        for c in range(w):
            mask[round(c * (h - 1) / (w - 1)), c] = True
    elif mode == 4:
        mask[h // 2] = True
        mask[:, w // 2] = True
    else:
        # Random 8-connected paths join independently sampled extremal pixels.
        points = [(0, int(rng.integers(w))), (h - 1, int(rng.integers(w))),
                  (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)]
        rng.shuffle(points)
        r, c = points[0]
        mask[r, c] = True
        for y, x in points[1:]:
            while (r, c) != (y, x):
                dr, dc = int(np.sign(y - r)), int(np.sign(x - c))
                if dr and dc and rng.random() < .4:
                    if rng.random() < .5:
                        dr = 0
                    else:
                        dc = 0
                r, c = r + dr, c + dc
                mask[r, c] = True
        target = int(rng.uniform(.3, .9) * h * w)
        while mask.sum() < target:
            choices = [(r, c) for r in range(h) for c in range(w) if not mask[r, c]
                       and mask[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any()]
            r, c = choices[int(rng.integers(len(choices)))]
            mask[r, c] = True
    if rng.random() < .5:
        mask = mask[::-1]
    if rng.random() < .5:
        mask = mask[:, ::-1]
    if h == w:
        mask = np.rot90(mask, int(rng.integers(4)))
    return mask.copy()


def _make_key(rng, direction, carrier, trail, tip):
    for _ in range(30):
        dr, dc = direction
        strokes = int(rng.choice([2, 2, 2, 3]))
        length = int(rng.choice([2, 2, 3, 4]))
        points = []
        for i in range(strokes):
            stagger = int(rng.integers(2))
            base_r, base_c = (0, 2 * i) if dr else (2 * i, 0)
            for j in range(length + 1):
                points.append((base_r + (j + stagger) * dr,
                               base_c + (j + stagger) * dc,
                               tip if j == length else trail))
        r0 = min(p[0] for p in points)
        c0 = min(p[1] for p in points)
        r1 = max(p[0] for p in points)
        c1 = max(p[1] for p in points)
        pads = list(map(int, rng.choice([1, 1, 1, 2], 4)))
        h, w = r1 - r0 + 1 + pads[0] + pads[1], c1 - c0 + 1 + pads[2] + pads[3]
        if h > 10 or w > 9:
            continue
        key = np.full((h, w), carrier, np.int8)
        for r, c, color in points:
            key[r - r0 + pads[0], c - c0 + pads[2]] = color
        if len(_decode_glyph(key, carrier)) == 1:
            return key
    # Minimal two-stroke glyph fits in every canonical exterior band.
    dr, dc = direction
    points = [(j * dr + (2 * i if dr == 0 else 0),
               j * dc + (2 * i if dr != 0 else 0), tip if j == 2 else trail)
              for i in range(2) for j in range(3)]
    r0, c0 = min(p[0] for p in points), min(p[1] for p in points)
    h, w = max(p[0] for p in points) - r0 + 3, max(p[1] for p in points) - c0 + 3
    key = np.full((h, w), carrier, np.int8)
    for r, c, color in points:
        key[r - r0 + 1, c - c0 + 1] = color
    return key


def _rot_direction(direction, k):
    r, c = direction
    for _ in range(k % 4):
        r, c = -c, r
    return r, c


def _positions(rng, available, count, spread=False):
    if len(available) < count:
        return None
    for _ in range(30):
        values = sorted(map(int, rng.choice(available, count, replace=False)))
        if not spread or all(b - a >= 5 for a, b in zip(values, values[1:])):
            return values
    if spread and count == 2 and available[-1] - available[0] >= 5:
        return [available[0], available[-1]]
    return None


def _launch_records(rng, h, w, masks, products):
    used_rows, used_cols, claimed = set(), set(), set()
    records = []
    for index, (mask, counts) in enumerate(zip(masks, products)):
        mh, mw = mask.shape
        offsets = {(int(r) - mh // 2, int(c) - mw // 2) for r, c in np.argwhere(mask)}
        available_rows = [r for r in range(1 + mh // 2, h - 1 - mh // 2) if r not in used_rows]
        available_cols = [c for c in range(1 + mw // 2, w - 1 - mw // 2) if c not in used_cols]
        for _ in range(80):
            spread = bool(index == 0 and counts == (2, 2))
            rows = _positions(rng, available_rows, counts[0], spread)
            cols = _positions(rng, available_cols, counts[1], spread)
            if rows is None or cols is None:
                return None
            copies = [{(r + dy, c + dx) for dy, dx in offsets} for r, c in product(rows, cols)]
            cells = set().union(*copies)
            if claimed.isdisjoint(cells):
                break
        else:
            return None
        records.append(dict(mask=mask, offsets=offsets, rows=rows, cols=cols,
                            copies=copies, cells=cells))
        claimed.update(cells)
        used_rows.update(rows)
        used_cols.update(cols)
    return records


def _walls_for_formation(rng, h, w, records, direction, kind, requested_stop=None):
    cells = set().union(*(rec["cells"] for rec in records))
    walls = np.zeros((h, w), bool)
    walls[[0, -1]] = True
    walls[:, [0, -1]] = True
    maximum = min(_pixel_clearance(walls, p, direction) for p in cells)
    if maximum < 2 or (requested_stop is not None and maximum <= requested_stop):
        return None
    dr, dc = direction
    stop = (requested_stop if requested_stop is not None else
            int(rng.integers(1, min(maximum - 1, 5 if kind == "ordinary" else 2) + 1)))
    protected = set()
    for step in range(stop + 1):
        protected.update(_translate(cells, (step * dr, step * dc)))
        if step and dr and dc:
            protected.update(_translate(cells, ((step - 1) * dr + dr, (step - 1) * dc)))
            protected.update(_translate(cells, ((step - 1) * dr, (step - 1) * dc + dc)))
    following = _translate(cells, ((stop + 1) * dr, (stop + 1) * dc))
    if kind == "corner":
        sides = (_translate(cells, ((stop + 1) * dr, stop * dc))
                 | _translate(cells, (stop * dr, (stop + 1) * dc)))
        blockers = sides - following - protected
    else:
        blockers = following - protected
    blockers = sorted(p for p in blockers if 1 <= p[0] < h - 1 and 1 <= p[1] < w - 1)
    if not blockers:
        return None
    blocker = blockers[int(rng.integers(len(blockers)))]
    walls[blocker] = True
    if kind == "aperture":
        boxes = set()
        for rec in records:
            mh, mw = rec["mask"].shape
            boxes.update((r + dy, c + dx) for r, c in product(rec["rows"], rec["cols"])
                         for dy in range(-(mh // 2), mh // 2 + 1)
                         for dx in range(-(mw // 2), mw // 2 + 1))
        holes = sorted(boxes - protected)
        if not holes:
            return None
        walls[holes[int(rng.integers(len(holes)))]] = True
    forbidden = protected | (following if kind == "corner" else set())
    # Jagged corner bays, broken baffles, and interior islands share one wall role.
    for corner in rng.choice(4, int(rng.integers(4)), replace=False):
        depth = int(rng.integers(1, 4))
        for y in range(1, depth + 1):
            for x in range(1, depth + 2 - y):
                p = (y if corner < 2 else h - 1 - y,
                     x if corner % 2 == 0 else w - 1 - x)
                if p not in forbidden:
                    walls[p] = True
    for _ in range(int(rng.integers(4))):
        horizontal = bool(rng.integers(2))
        r, c = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
        length = int(rng.integers(2, (w if horizontal else h) // 2 + 1))
        for j in range(length):
            p = (r, c + j) if horizontal else (r + j, c)
            if 1 <= p[0] < h - 1 and 1 <= p[1] < w - 1 and p not in forbidden:
                walls[p] = True
    for r, c in np.argwhere(rng.random((h, w)) < float(rng.uniform(0, .025))):
        if (r, c) not in forbidden:
            walls[r, c] = True
    distance = min(_pixel_clearance(walls, p, direction) for p in cells)
    _require(distance == stop, "construction lost its protected shared displacement")
    return walls, distance


def _assemble(rng, walls, records, direction, colors, key, distance, rotation):
    bg, wall, _, _ = colors[:4]
    h, w = walls.shape
    sw = max(key.shape[1], max(rec["mask"].shape[1] for rec in records))
    sw = min(29 - w, sw + int(rng.integers(3)))
    if sw < key.shape[1]:
        return None
    objects = [("key", key)] + [(i, rec["mask"]) for i, rec in enumerate(records)]
    order = list(map(int, rng.permutation(len(objects))))
    minimum_h = sum(obj.shape[0] for _, obj in objects) + len(objects) - 1
    height = max(h, minimum_h) + int(rng.integers(0, min(3, 31 - max(h, minimum_h))))
    if height > 30:
        return None
    inp = np.full((height, sw + 1 + w), bg, np.int8)
    board_r = int(rng.integers(height - h + 1))
    board_c = sw + 1
    inp[board_r:board_r + h, board_c:board_c + w][walls] = wall
    slack = height - minimum_h
    gaps = rng.multinomial(slack, np.full(len(objects) + 1, 1 / (len(objects) + 1)))
    row = int(gaps[0])
    source_cells = []
    for slot, index in enumerate(order):
        tag, obj = objects[index]
        oh, ow = obj.shape
        col = int(rng.integers(sw - ow + 1))
        if tag == "key":
            inp[row:row + oh, col:col + ow] = obj
        else:
            color = colors[4 + tag]
            inp[row:row + oh, col:col + ow][obj] = color
            source_cells.extend((row + int(y), col + int(x)) for y, x in np.argwhere(obj))
        row += oh + 1 + int(gaps[slot + 1])
    for rec, color in zip(records, colors[4:]):
        for r in rec["rows"]:
            inp[board_r + r, board_c] = inp[board_r + r, board_c + w - 1] = color
        for c in rec["cols"]:
            inp[board_r, board_c + c] = inp[board_r + h - 1, board_c + c] = color
    out = inp.copy()
    for p in source_cells:
        out[p] = bg
    dr, dc = direction
    for rec, color in zip(records, colors[4:]):
        for r, c in rec["cells"]:
            out[board_r + r + distance * dr, board_c + c + distance * dc] = color
    return dict(input=np.rot90(inp, rotation).copy(), output=np.rot90(out, rotation).copy())


def _make_pair(rng, global_direction, kind="ordinary", test=False, requested_stop=None):
    for attempt in range(240):
        rotation = int(rng.integers(4))
        direction = _rot_direction(global_direction, -rotation)
        n = 2 if kind == "aperture" else int(rng.choice([1, 2, 3], p=[.40, .43, .17]))
        colors = list(map(int, rng.permutation(10)))[:4 + n]
        key = _make_key(rng, direction, colors[1], colors[2], colors[3])
        h = int(rng.integers(18 if kind == "aperture" else 13, 25))
        max_w = min(21, 29 - key.shape[1])
        w = int(rng.integers(16 if kind == "aperture" else 13, max_w + 1))
        masks, multiplicities = [], []
        for i in range(n):
            if i == 0 and kind == "aperture":
                masks.append(_mask(rng, 5, 5, aperture=True))
                multiplicities.append((2, 2))
            else:
                sizes = [(1, 3), (3, 1), (3, 3), (3, 5), (5, 3), (5, 5)]
                weights = np.array([1, 1, 5, 2, 2, 2 if n < 3 else .4], float)
                mh, mw = sizes[int(rng.choice(len(sizes), p=weights / weights.sum()))]
                masks.append(_mask(rng, mh, mw))
                if i == 0 and kind == "corner":
                    counts = [(2, 1), (1, 2)][int(rng.integers(2))]
                elif i == 0 and test:
                    counts = [(3, 1), (1, 3), (2, 3), (3, 2), (2, 2)][int(rng.integers(5))]
                else:
                    counts = [(1, 1), (1, 1), (2, 1), (1, 2), (2, 2)][int(rng.integers(5))]
                multiplicities.append(counts)
        records = _launch_records(rng, h, w, masks, multiplicities)
        if records is None:
            continue
        geometry = _walls_for_formation(rng, h, w, records, direction, kind, requested_stop)
        if geometry is None:
            continue
        walls, distance = geometry
        pair = _assemble(rng, walls, records, direction, colors, key, distance, rotation)
        if pair is None:
            continue
        # Reject only visible ambiguities, missing requested evidence, or packing.
        try:
            info = _parse(pair["input"])
        except AssertionError:
            continue
        features = _feature_info(info)
        if kind == "aperture" and not (features["coupling"] and features["color_coupling"]
                                        and features["mask_vs_box"]
                                        and features["mask_vs_center"] and features["hole_wall"]):
            continue
        if kind == "corner" and not features["corner"]:
            continue
        if np.array_equal(_alternative(info, "reverse"), pair["output"]):
            continue
        _require(np.array_equal(solve(pair["input"]), pair["output"]), "construction/prediction disagreement")
        return pair
    raise RuntimeError("bounded construction failed to supply its geometric witness")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    cardinal = CARDINAL[int(rng.integers(len(CARDINAL)))]
    diagonal = DIAGONAL[int(rng.integers(len(DIAGONAL)))]
    training_pool = [cardinal, diagonal]
    novel = [d for d in DIRECTIONS if d not in training_pool]
    first_stop = int(rng.integers(1, 3))
    plans = [(cardinal, "aperture", False, first_stop),
             (diagonal, "corner", False, 3 - first_stop)]
    plans += [(training_pool[int(rng.integers(2))], "ordinary", False, None) for _ in range(nt - 2)]
    plans += [(novel[int(rng.integers(len(novel)))], "ordinary", True, None) for _ in range(nq)]
    puzzle, seen = {"train": [], "test": []}, set()
    for index, (direction, kind, test, requested_stop) in enumerate(plans):
        for _ in range(12):
            pair = _make_pair(rng, direction, kind, test, requested_stop)
            fp = _fingerprint(pair["input"])
            if fp not in seen:
                break
        else:
            raise RuntimeError("bounded distinct-input construction failed")
        seen.add(fp)
        puzzle["test" if index >= nt else "train"].append(pair)
    order = rng.permutation(nt)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimension preservation")
            fp = _fingerprint(inp)
            _require(fp not in seen, "duplicate input")
            seen.add(fp)
            info = _parse(inp)
            f = _feature_info(info)
            _require(f["distance"] > 0, "generated pairs must show positive collective travel")
            _require(np.array_equal(solve(inp), out), "wrong rigid translation")
            # Verify unchanged structure and the only permitted source erasures.
            allowed = np.zeros(inp.shape, bool)
            r0, c0, h, w = info["board"]
            allowed[r0 + 1:r0 + h - 1, c0 + 1:c0 + w - 1] = True
            for t in info["templates"]:
                for p in t["cells"]:
                    allowed[p] = True
                    _require(out[p] == info["bg"], "occupied source pixels must clear")
                expected_pixels = sum(color == t["color"] for color in info["formation"].values())
                _require(np.count_nonzero(out[r0 + 1:r0 + h - 1, c0 + 1:c0 + w - 1] == t["color"])
                         == expected_pixels, "colored union area must survive rigid motion")
            _require(np.array_equal(inp[~allowed], out[~allowed]), "key, labels, or exterior changed")
            sub_out = out[r0:r0 + h, c0:c0 + w]
            sub_in = inp[r0:r0 + h, c0:c0 + w]
            _require(np.array_equal(sub_out[info["walls"]], sub_in[info["walls"]]), "walls changed")
            dr, dc = info["direction"]
            final = _translate(info["formation"], (f["distance"] * dr, f["distance"] * dc))
            _require(_free(info["walls"], final), "final footprint intersects wall")
            _require(_common_distance(info["walls"], final, info["direction"]) == 0,
                     "constellation did not stop at first obstruction")
            f["info"] = info
            (training if split == "train" else testing).append(f)
    _require(any(f["coupling"] and f["color_coupling"] and f["mask_vs_box"]
                 and f["mask_vs_center"] and f["hole_wall"]
                 and any(r > 1 and c > 1 for r, c in f["products"]) for f in training),
             "missing coupled Cartesian aperture demonstration")
    _require(any(f["corner"] for f in training), "missing diagonal corner demonstration")
    _require(len({f["distance"] for f in training}) >= 2, "missing variable-clearance demonstration")
    training_directions = {f["direction"] for f in training}
    _require(len(training_directions) >= 2 and any(0 in d for d in training_directions)
             and any(0 not in d for d in training_directions), "direction evidence")
    _require(all(f["direction"] not in training_directions for f in testing), "held-out direction shift")
    for kind in ("stationary", "zip", "individual", "by_color", "reverse", "no_corner"):
        _require(any(not np.array_equal(_alternative(f["info"], kind), solve(f["info"]["input"]))
                     for f in training), f"training does not distinguish {kind}")


def _regression_pair(mask, rows, cols, direction, wall_points, expected_distance):
    """Small, explicitly specified geometric regressions, with known distances."""
    rng = np.random.default_rng(100)
    h, w = 15, 18
    walls = np.zeros((h, w), bool)
    walls[[0, -1]] = True
    walls[:, [0, -1]] = True
    for p in wall_points:
        walls[p] = True
    mh, mw = mask.shape
    offsets = {(int(r) - mh // 2, int(c) - mw // 2) for r, c in np.argwhere(mask)}
    copies = [{(r + dy, c + dx) for dy, dx in offsets} for r, c in product(rows, cols)]
    records = [dict(mask=mask, offsets=offsets, rows=rows, cols=cols,
                    copies=copies, cells=set().union(*copies))]
    colors = [0, 8, 2, 4, 6]
    key = _make_key(rng, direction, 8, 2, 4)
    pair = _assemble(rng, walls, records, direction, colors, key, expected_distance, 0)
    _require(pair is not None, "regression packing")
    return pair


def self_test():
    ring3 = np.ones((3, 3), bool)
    ring3[1, 1] = False
    ring5 = np.ones((5, 5), bool)
    ring5[1:-1, 1:-1] = False
    ring7 = np.ones((7, 7), bool)
    ring7[1:-1, 1:-1] = False
    cases = [
        _regression_pair(ring3, [4, 10], [4, 11], (0, 1), [], 4),
        _regression_pair(np.eye(3, dtype=bool), [4], [4], (1, 1), [(7, 6)], 1),
        _regression_pair(ring5, [7], [7], (0, 1), [(7, 7), (6, 11)], 1),
        _regression_pair(ring3, [4, 10], [4, 11], (0, 1), [(4, 13)], 0),
        _regression_pair(ring7, [7], [7], (0, 1), [], 6),
    ]
    for pair in cases:
        _require(np.array_equal(solve(pair["input"]), pair["output"]), "explicit geometric regression")
        for rotation in range(4):
            _require(np.array_equal(solve(np.rot90(pair["input"], rotation)),
                                    np.rot90(pair["output"], rotation)), "rotation equivariance")
        for shift in range(10):
            _require(np.array_equal(solve((pair["input"] + shift) % 10),
                                    (pair["output"] + shift) % 10), "palette equivariance")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_fingerprint(p) for p in puzzles}) == 42, "42-seed puzzle diversity")
    _require(_puzzle_fingerprint(puzzles[0]) == _puzzle_fingerprint(generate_puzzle(0)), "determinism")
    shapes, masks, products_seen, directions, counts, palette_roles, keys, distances = (set() for _ in range(8))
    merged = 0
    for puzzle in puzzles:
        local_shapes, local_counts = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                f = _feature_info(info)
                shapes.add(pair["input"].shape)
                local_shapes.add(pair["input"].shape)
                counts.add(f["count"])
                local_counts.add(f["count"])
                masks.update((t["mask"].shape, t["mask"].tobytes()) for t in info["templates"])
                products_seen.update(f["products"])
                directions.add(f["direction"])
                palette_roles.add((info["bg"], info["wall"], info["key"]["trail"], info["key"]["tip"]))
                keys.add(info["key"]["box"][2:])
                distances.add(f["distance"])
                merged += f["merge"] > 0
        _require(len(local_shapes) >= 2, "within-puzzle scene variation")
    _require(len(shapes) >= 45 and len(masks) >= 50, "structural variation")
    _require(directions == set(DIRECTIONS) and counts == {1, 2, 3}, "direction/count diversity")
    _require({(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3), (2, 3), (3, 2)} <= products_seen,
             "asymmetric Cartesian products")
    _require(len(palette_roles) >= 80 and len(keys) >= 12, "local palette/key variation")
    _require({1, 2, 3, 4} <= distances and merged >= 5, "clearance/merging variation")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            _require(len(puzzle["train"]) == nt and len(puzzle["test"]) == nq, "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (4.0, 1), (4, False), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Mutations change an answer's conceptual content, not only its schema.
    first = puzzles[0]
    for kind in ("stationary", "zip", "individual", "by_color", "reverse", "no_corner"):
        index = next(i for i, p in enumerate(first["train"])
                     if not np.array_equal(_alternative(_parse(p["input"]), kind), p["output"]))
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
        bad["train"][index]["output"] = _alternative(_parse(first["train"][index]["input"]), kind)
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"validator accepted {kind} corruption")
    return {"seeds": 42, "pairs": sum(len(p[s]) for p in puzzles for s in p),
            "canvas_shapes": len(shapes), "stencil_masks": len(masks),
            "key_shapes": len(keys), "palette_roles": len(palette_roles),
            "directions": len(directions), "template_counts": sorted(counts),
            "coordinate_products": sorted(products_seen), "distances": sorted(distances),
            "merging_pairs": merged, "independent_solver": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
