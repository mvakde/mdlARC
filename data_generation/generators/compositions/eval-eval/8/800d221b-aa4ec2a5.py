"""Courtyard elections: topological activation and geodesic rim voting.

Only NumPy and the standard library are used. Generation keeps no global state,
does no I/O, and never reads the parents. See task_spec.md for the full rule.
"""
from collections import deque
import hashlib

import numpy as np

BACKGROUND = 0
SCAFFOLD = 8
SIGNALS = (1, 2, 3, 4, 5, 6, 7, 9)
D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
INF = 10000


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(((a >= 0) & (a <= 9)).all()), "invalid color")
    return a


def _regions(mask, directions=D4):
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    result = []
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        index = len(result)
        labels[r, c] = index
        stack, cells = [(r, c)], []
        while stack:
            p = stack.pop()
            cells.append(p)
            for dr, dc in directions:
                nr, nc = p[0] + dr, p[1] + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] < 0:
                    labels[nr, nc] = index
                    stack.append((nr, nc))
        result.append(cells)
    return result, labels


def _dilate(mask, directions=D8):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    out = np.zeros_like(mask, dtype=bool)
    for dr, dc in directions:
        out |= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return out


def _holes(material, directions=D4):
    # Padding makes outside meaningful even for edge-touching material.
    free = np.pad(~material, 1, constant_values=True)
    _, labels = _regions(free, directions)
    return (~material) & (labels[1:-1, 1:-1] != labels[0, 0])


def _distance(material, sources):
    h, w = material.shape
    distance = np.full((h, w), INF, dtype=np.int16)
    queue = deque()
    for r, c in sources:
        distance[r, c] = 0
        queue.append((int(r), int(c)))
    while queue:
        r, c = queue.popleft()
        d = int(distance[r, c]) + 1
        for dr, dc in D4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and material[nr, nc] and distance[nr, nc] > d:
                distance[nr, nc] = d
                queue.append((nr, nc))
    return distance


def _analyze(input_grid):
    a = _grid(input_grid)
    values = set(map(int, np.unique(a)))
    _require({BACKGROUND, SCAFFOLD} <= values and len(values) == 4, "need black, azure and two signal colors")
    signals = tuple(sorted(values - {BACKGROUND, SCAFFOLD}))
    material = a != BACKGROUND
    objects, object_ids = _regions(material)
    _require(len(objects) == len(_regions(material, D8)[0]), "diagonal-only object contact")
    holes_mask = _holes(material)
    _require(np.array_equal(holes_mask, _holes(material, D8)), "diagonal-only exterior leak")
    holes, _ = _regions(holes_mask)
    _require(len(holes) == len(_regions(holes_mask, D8)[0]), "diagonally touching courtyards")
    panel_mask = material & (a != SCAFFOLD)
    panel_cells, _ = _regions(panel_mask)
    panels = []
    forbidden_panel = _dilate(holes_mask)
    for cells in panel_cells:
        rr, cc = np.array(cells).T
        r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
        _require(len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1) and len(cells) >= 3,
                 "panels must be full rectangles of area at least three")
        counts = [int(np.count_nonzero(a[rr, cc] == s)) for s in signals]
        _require(min(counts) > 0 and counts[0] != counts[1], "panel vote must be mixed and strict")
        _require(not forbidden_panel[rr, cc].any(), "a scaffold layer must separate panels from courtyards")
        panels.append({"cells": cells, "box": (r0, r1, c0, c1), "counts": counts,
                       "majority": signals[int(counts[1] > counts[0])],
                       "body": int(object_ids[cells[0]])})
    _require(len(panels) >= 2, "no panel competition")
    sources = [[p for panel in panels if panel["majority"] == s for p in panel["cells"]]
               for s in signals]
    distances = [_distance(material, cells) for cells in sources]
    wave = np.full(a.shape, SCAFFOLD, dtype=np.int8)
    wave[distances[0] < distances[1]] = signals[0]
    wave[distances[1] < distances[0]] = signals[1]
    active_ids = set()
    courtyards = []
    for cells in holes:
        mask = np.zeros_like(material)
        rr, cc = np.array(cells).T
        mask[rr, cc] = True
        rim = _dilate(mask, D4) & material
        owners = set(map(int, object_ids[rim]))
        _require(len(owners) == 1, "courtyard must belong to one material body")
        owner = owners.pop()
        _require(np.all(a[rim] == SCAFFOLD), "courtyard rim must be scaffold")
        counts = [int(np.count_nonzero(wave[rim] == s)) for s in signals]
        _require(min(counts) > 0 and counts[0] != counts[1], "courtyard needs a strict, contested rim vote")
        active_ids.add(owner)
        courtyards.append({"cells": cells, "rim": rim, "counts": counts,
                           "winner": signals[int(counts[1] > counts[0])], "body": owner})
    _require(courtyards, "at least one sealed courtyard is required")
    for owner in active_ids:
        labels = {p["majority"] for p in panels if p["body"] == owner}
        _require(labels == set(signals), "active body needs both local majority labels")
    active = np.isin(object_ids, sorted(active_ids))
    scaffold = a == SCAFFOLD
    outline = _dilate(material) & ~material & ~holes_mask
    return {"signals": signals, "material": material, "objects": objects,
            "object_ids": object_ids, "panels": panels, "distances": distances,
            "wave": wave, "courtyards": courtyards, "holes": holes_mask,
            "active": active, "active_ids": active_ids, "scaffold": scaffold,
            "outline": outline, "panel_mask": panel_mask}


def _paint(a, info):
    out = np.array(a, dtype=np.int8, copy=True)
    conduct = info["active"] & info["scaffold"]
    out[conduct] = info["wave"][conduct]
    for hole in info["courtyards"]:
        rr, cc = np.array(hole["cells"]).T
        out[rr, cc] = hole["winner"]
    out[info["outline"]] = SCAFFOLD
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover panels, topology, graph distances and rim votes from this input."""
    a = _grid(input_grid)
    return _paint(a, _analyze(a))


def _crop(mask):
    rr, cc = np.nonzero(mask)
    return mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()


def _orient(a, turns, flip):
    a = np.rot90(a, turns)
    return np.fliplr(a).copy() if flip else a.copy()


def _body(rng, count, compact=False):
    """Connected sculpted material, independently carved sealed pockets."""
    if compact:
        hole_widths = [int(rng.integers(3, 6))]
        hole_height = int(rng.integers(3, 5))
        top, bottom, left, right = map(int, rng.integers(2, 4, size=4))
        gaps = []
    else:
        hole_widths = list(map(int, rng.integers(2 if count == 3 else 3, 5 if count == 3 else 7, size=count)))
        hole_height = int(rng.integers(3, 7))
        top, bottom, left, right = map(int, rng.integers(2, 5, size=4))
        gaps = list(map(int, rng.integers(2, 5 if count < 3 else 4, size=count - 1)))
    h, w = hole_height + top + bottom, sum(hole_widths) + sum(gaps) + left + right
    if w > 27:
        return None
    material = np.ones((h, w), dtype=bool)
    x = left
    for index, width in enumerate(hole_widths):
        hh = hole_height if compact else int(rng.integers(2, hole_height + 1))
        y = top if compact else int(rng.integers(2, h - hh - 1))
        cavity = np.ones((hh, width), dtype=bool)
        if min(hh, width) >= 3 and rng.random() < .55:
            # A connected L/stepped pocket, not an outline-shape template.
            cut_h = int(rng.integers(1, hh - 1))
            cut_w = int(rng.integers(1, width - 1))
            cavity[:cut_h, :cut_w] = False
            cavity = _orient(cavity, 2 * int(rng.integers(2)), bool(rng.integers(2)))
        material[y:y + hh, x:x + width] &= ~cavity
        x += width + (gaps[index] if index < len(gaps) else 0)
    original_holes = _holes(material)
    # Exterior corner cuts and narrow bays cannot change the carved topology.
    for _ in range(int(rng.integers(1, 5))):
        candidate = material.copy()
        side = int(rng.integers(4))
        depth, breadth = int(rng.integers(1, 3)), int(rng.integers(1, 4))
        if side == 0:
            col = int(rng.integers(max(1, w - breadth + 1)))
            candidate[:depth, col:col + breadth] = False
        elif side == 1:
            col = int(rng.integers(max(1, w - breadth + 1)))
            candidate[-depth:, col:col + breadth] = False
        elif side == 2:
            row = int(rng.integers(max(1, h - breadth + 1)))
            candidate[row:row + breadth, :depth] = False
        else:
            row = int(rng.integers(max(1, h - breadth + 1)))
            candidate[row:row + breadth, -depth:] = False
        if (len(_regions(candidate)[0]) == 1 and np.array_equal(_holes(candidate), original_holes)
                and np.array_equal(_holes(candidate, D8), original_holes)):
            material = candidate
    if not compact and rng.random() < .6:
        # A bent appendage makes graph travel differ from straight-line distance.
        length = int(rng.integers(2, 6))
        if h + length <= 27:
            extended = np.pad(material, ((0, length), (1, 1)))
            col = int(rng.choice(np.flatnonzero(material[-1]))) + 1
            extended[h - 1:, col] = True
            end = min(extended.shape[1], col + int(rng.integers(2, 5)))
            extended[-1, col:end] = True
            material = _crop(extended)
    return material


def _open_door(material, rng):
    holes = _holes(material)
    for turns in rng.permutation(4):
        m = np.rot90(material, int(turns)).copy()
        hh = np.rot90(holes, int(turns))
        cells = list(zip(*np.nonzero(hh)))
        rng.shuffle(cells)
        for r, c in cells:
            candidate = m.copy()
            candidate[:r + 1, c] = False
            if len(_regions(candidate)[0]) == 1 and not _holes(candidate).any():
                opened = np.rot90(candidate, -int(turns)).copy()
                if opened.shape == material.shape:
                    return opened
    return None


def _panel_candidates(material, allowed, witness):
    holes = _holes(material)
    eligible = allowed & ~_dilate(holes)
    sizes = ((1, 3), (3, 1)) if witness else (
        (1, 3), (3, 1), (1, 4), (4, 1), (1, 5), (5, 1),
        (2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4), (4, 3))
    candidates = []
    for ph, pw in sizes:
        for r in range(material.shape[0] - ph + 1):
            for c in range(material.shape[1] - pw + 1):
                if eligible[r:r + ph, c:c + pw].all():
                    candidates.append((r, c, ph, pw))
    return candidates


def _box_cells(box):
    r, c, h, w = box
    return [(rr, cc) for rr in range(r, r + h) for cc in range(c, c + w)]


def _pick_boxes(rng, candidates, count, shape):
    blocked = np.zeros(shape, dtype=bool)
    boxes = []
    for index in rng.permutation(len(candidates)):
        box = candidates[int(index)]
        r, c, h, w = box
        if not blocked[r:r + h, c:c + w].any():
            boxes.append(box)
            blocked[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1] = True
            if len(boxes) == count:
                return boxes
    return None


def _decorate(rng, material, signals, count, witness=False, opened=None, opposite=False):
    """Place sources first, then choose visible strict votes on that geometry."""
    allowed = material if opened is None else material & opened
    candidates = _panel_candidates(material, allowed, witness)
    hole_cells, _ = _regions(_holes(material))
    rims = []
    for cells in hole_cells:
        mask = np.zeros_like(material)
        rr, cc = np.array(cells).T
        mask[rr, cc] = True
        rims.append(_dilate(mask, D4) & material)
    for _ in range(64):
        boxes = _pick_boxes(rng, candidates, count, material.shape)
        if boxes is None:
            continue
        cells = [_box_cells(box) for box in boxes]
        distance = np.stack([_distance(material, points) for points in cells])
        if witness:
            assignments = [1 << i for i in rng.permutation(count)]
        else:
            assignments = list(range(1, (1 << count) - 1))
            rng.shuffle(assignments)
        for bits in assignments:
            left = [i for i in range(count) if bits & (1 << i)]
            right = [i for i in range(count) if not bits & (1 << i)]
            d0, d1 = distance[left].min(axis=0), distance[right].min(axis=0)
            winners = []
            valid = True
            for rim in rims:
                n0, n1 = int(np.sum((d0 < d1) & rim)), int(np.sum((d1 < d0) & rim))
                if min(n0, n1) == 0 or n0 == n1:
                    valid = False
                    break
                winners.append(int(n1 > n0))
            if not valid or (witness and winners != [0]) or (opposite and len(set(winners)) < 2):
                continue
            image = np.where(material, SCAFFOLD, BACKGROUND).astype(np.int8)
            for index, box in enumerate(boxes):
                r, c, ph, pw = box
                majority = signals[0] if bits & (1 << index) else signals[1]
                minority = next(s for s in signals if s != majority)
                area = ph * pw
                nminor = 1 if witness else int(rng.integers(1, (area - 1) // 2 + 1))
                pattern = np.full(area, majority, dtype=np.int8)
                mode = int(rng.integers(3))
                if mode == 0:
                    order = rng.permutation(area)
                elif mode == 1:
                    order = np.arange(area)
                else:
                    rr, cc = np.indices((ph, pw))
                    order = np.argsort(((rr + cc) % 2 * area + rr * pw + cc).ravel())
                pattern[order[:nminor]] = minority
                image[r:r + ph, c:c + pw] = pattern.reshape(ph, pw)
            return image
    return None


def _small_solid(rng):
    style = int(rng.integers(3))
    if style == 0:
        mask = np.ones((1, int(rng.integers(1, 6))), dtype=bool)
    elif style == 1:
        mask = np.ones((int(rng.integers(3, 6)), int(rng.integers(3, 6))), dtype=bool)
        mask[:-1, 1:-1] = False
    else:
        mask = np.ones((int(rng.integers(1, 4)), int(rng.integers(1, 4))), dtype=bool)
    return np.where(_orient(mask, int(rng.integers(4)), bool(rng.integers(2))), SCAFFOLD, BACKGROUND).astype(np.int8)


def _scene(rng, required, extras=True):
    items = list(required)
    if extras:
        for _ in range(int(rng.integers(3))):
            item = _small_solid(rng)
            # A simple finite shelf bound guarantees a packing fallback.
            proposed = items + [item]
            if (sum(a.shape[0] for a in proposed) + 2 * (len(proposed) - 1) <= 28 or
                    sum(a.shape[1] for a in proposed) + 2 * (len(proposed) - 1) <= 28):
                items.append(item)
    hmax = max(a.shape[0] for a in items)
    wmax = max(a.shape[1] for a in items)
    # Free offsets, canvas aspect and margins precede deterministic shelf fallback.
    area = sum(a.size for a in items)
    for attempt in range(12):
        h = int(rng.integers(min(30, max(hmax, int(np.sqrt(area)))), 31))
        w = int(rng.integers(min(30, max(wmax, (area + h - 1) // h)), 31))
        grid = np.zeros((h, w), dtype=np.int8)
        blocked = np.zeros((h, w), dtype=bool)
        for item in sorted(items, key=lambda a: -a.size):
            mh, mw = item.shape
            valid = []
            for r in range(h - mh + 1):
                for c in range(w - mw + 1):
                    if not blocked[r:r + mh, c:c + mw].any():
                        valid.append((r, c))
            if not valid:
                break
            r, c = valid[int(rng.integers(len(valid)))]
            grid[r:r + mh, c:c + mw] = item
            blocked[max(0, r - 1):r + mh + 1, max(0, c - 1):c + mw + 1] = True
        else:
            if _dilate(grid != 0).any() and ((grid == 0) & ~_holes(grid != 0)).any():
                return grid
    # The maximum required bodies fit side-by-side on at least one axis.
    order = list(rng.permutation(len(items)))
    vertical = sum(a.shape[0] for a in items) + 2 * (len(items) - 1) <= 28
    if vertical:
        h = sum(a.shape[0] for a in items) + 2 * (len(items) - 1) + 2
        w = wmax + 2
    else:
        h = hmax + 2
        w = sum(a.shape[1] for a in items) + 2 * (len(items) - 1) + 2
    _require(h <= 30 and w <= 30, "packing capacity exceeded")
    grid = np.zeros((h, w), dtype=np.int8)
    cursor = 1
    for index in order:
        item = items[int(index)]
        mh, mw = item.shape
        r = cursor if vertical else int(rng.integers(h - mh + 1))
        c = int(rng.integers(w - mw + 1)) if vertical else cursor
        grid[r:r + mh, c:c + mw] = item
        cursor += (mh if vertical else mw) + 2
    return grid


def _make_pair(rng, signals, kind, holes=1, sources=3):
    for _ in range(48):
        mask = _body(rng, holes, compact=kind == "contrast")
        if mask is None:
            continue
        opened = _open_door(mask, rng) if kind == "contrast" else None
        if kind == "contrast" and opened is None:
            continue
        if opened is not None and not (_holes(mask) & ~_dilate(opened)).any():
            # Retain a visibly black interior part of the open bay even after
            # outlining, so cavity filling cannot be confused with dilation.
            continue
        turns, flip = int(rng.integers(4)), bool(rng.integers(2))
        mask = _orient(mask, turns, flip)
        if opened is not None:
            opened = _orient(opened, turns, flip)
        image = _decorate(rng, mask, signals, sources, witness=kind == "contrast",
                          opened=opened, opposite=kind == "multiple")
        if image is None:
            continue
        required = [image]
        if opened is not None:
            other = image.copy()
            other[~opened] = BACKGROUND
            required.append(other)
        inp = _scene(rng, required)
        try:
            info = _analyze(inp)
        except AssertionError:
            continue
        # Teaching evidence is enforced at construction, rather than merely
        # hoping a later random example will disambiguate these two shortcuts.
        if kind in ("contrast", "multiple"):
            feature = _features(inp, info)
            if kind == "contrast" and not feature["raw_pixel_disagreement"]:
                continue
            if kind == "multiple" and (not feature["straight_distance_disagreement"] or not feature["ties"]):
                continue
        return {"input": inp, "output": _paint(inp, info)}
    raise RuntimeError("bounded courtyard construction exhausted")


def _signature(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.choice(SIGNALS, 2, replace=False)))
    variable = bool(rng.integers(2))
    shift = int(rng.integers(2))  # additional courtyards, or additional sources
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            signals = tuple(map(int, rng.choice(SIGNALS, 2, replace=False))) if variable else palette
            if split == "train" and index == 0:
                kind, nholes, nsources = "contrast", 1, 3
            elif split == "train" and index == 1:
                kind, nholes, nsources = "multiple", 2, 4
            elif split == "train":
                kind = "ordinary"
                nholes, nsources = int(rng.integers(1, 3)), int(rng.integers(3, 5))
            else:
                kind = "multiple"
                nholes = 3 if shift == 0 else 2
                nsources = 4 if shift == 0 else int(rng.integers(5, 7))
            for _ in range(16):
                pair = _make_pair(rng, signals, kind, nholes, nsources)
                key = _signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    return puzzle


def _features(a, info=None):
    z = _analyze(a) if info is None else info
    s0, s1 = z["signals"]
    disagreements = 0
    for hole in z["courtyards"]:
        panels = [p for p in z["panels"] if p["body"] == hole["body"]]
        votes = [sum(p["majority"] == s for p in panels) for s in (s0, s1)]
        disagreements += votes[0] != votes[1] and (s1 if votes[1] > votes[0] else s0) != hole["winner"]
    raw0 = _distance(z["material"], list(zip(*np.nonzero(a == s0))))
    raw1 = _distance(z["material"], list(zip(*np.nonzero(a == s1))))
    raw_wave = np.full(a.shape, SCAFFOLD, dtype=np.int8)
    raw_wave[raw0 < raw1] = s0
    raw_wave[raw1 < raw0] = s1
    conduct = z["active"] & z["scaffold"]
    inactive_panel_bodies = {p["body"] for p in z["panels"]} - z["active_ids"]
    matched_open_closed = unfilled_open_bay = False
    crops = {}
    for owner in z["active_ids"] | inactive_panel_bodies:
        points = np.array(z["objects"][owner])
        r0, c0 = points.min(axis=0)
        r1, c1 = points.max(axis=0)
        crop = a[r0:r1 + 1, c0:c1 + 1].copy()
        crop[z["object_ids"][r0:r1 + 1, c0:c1 + 1] != owner] = 0
        crops[owner] = crop
    for closed_id in z["active_ids"]:
        for opened_id in inactive_panel_bodies:
            closed, opened = crops[closed_id], crops[opened_id]
            if closed.shape != opened.shape:
                continue
            changed = closed != opened
            if changed.any() and np.all(closed[changed] == SCAFFOLD) and np.all(opened[changed] == 0):
                matched_open_closed = True
                unfilled_open_bay |= bool((_holes(closed != 0) & ~_dilate(opened != 0)).any())
    opposite = any(len({hole["winner"] for hole in z["courtyards"] if hole["body"] == owner}) > 1
                   for owner in z["active_ids"])
    # Manhattan distance ignores paths around pockets and notches.
    rr, cc = np.indices(a.shape)
    straight = []
    for s in (s0, s1):
        points = [cell for panel in z["panels"] if panel["majority"] == s for cell in panel["cells"]]
        straight.append(np.minimum.reduce([abs(rr - r) + abs(cc - c) for r, c in points]))
    straight_wave = np.full(a.shape, SCAFFOLD, dtype=np.int8)
    straight_wave[straight[0] < straight[1]] = s0
    straight_wave[straight[1] < straight[0]] = s1
    return {"shape": tuple(a.shape), "objects": len(z["objects"]), "holes": len(z["courtyards"]),
            "panels": len(z["panels"]), "panel_areas": [len(p["cells"]) for p in z["panels"]],
            "hole_areas": [len(h["cells"]) for h in z["courtyards"]],
            "rim_counts": [int(h["rim"].sum()) for h in z["courtyards"]],
            "margins": [abs(h["counts"][0] - h["counts"][1]) for h in z["courtyards"]],
            "panel_count_disagreements": int(disagreements), "inactive_panel_bodies": len(inactive_panel_bodies),
            "matched_open_closed": bool(matched_open_closed), "unfilled_open_bay": bool(unfilled_open_bay),
            "opposite_courtyards": bool(opposite),
            "raw_pixel_disagreement": bool(np.any((raw_wave != z["wave"]) & conduct)),
            "straight_distance_disagreement": bool(np.any((straight_wave != z["wave"]) & conduct)),
            "ties": int(np.sum((z["distances"][0] == z["distances"][1]) & conduct)),
            "irregular_holes": sum(len(h["cells"]) != np.prod(np.ptp(np.array(h["cells"]), axis=0) + 1)
                                   for h in z["courtyards"]),
            "signals": z["signals"]}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= tr <= 10 and 1 <= te <= 4 and tr > te, "invalid counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "dimensions changed")
            key = _signature(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            z = _analyze(a)
            conduct = z["active"] & z["scaffold"]
            _require(np.array_equal(b[conduct], z["wave"][conduct]), "incorrect propagated colors or tie cells")
            _require(np.array_equal(b[z["panel_mask"]], a[z["panel_mask"]]), "panel pixels changed")
            _require(np.all(b[z["scaffold"] & ~z["active"]] == SCAFFOLD), "inactive scaffold changed")
            _require(np.all(b[z["outline"]] == SCAFFOLD), "wrong exterior outline")
            far = ~z["material"] & ~z["holes"] & ~z["outline"]
            _require(np.all(b[far] == BACKGROUND), "exterior bay or distant background filled")
            for hole in z["courtyards"]:
                rr, cc = np.array(hole["cells"]).T
                # Count unique original rim cells; no output topology is reused.
                counts = [int(np.sum(z["wave"][hole["rim"]] == s)) for s in z["signals"]]
                winner = z["signals"][int(counts[1] > counts[0])]
                _require(np.all(b[rr, cc] == winner), "wrong courtyard vote or incomplete fill")
            _require(np.array_equal(solve(a.copy()), b), "solve mismatch")
            features[split].append(_features(a, z))
    train = features["train"]
    _require(any(f["inactive_panel_bodies"] for f in train), "no topological activation contrast")
    _require(any(f["matched_open_closed"] and f["unfilled_open_bay"] for f in train),
             "no matched topology contrast with an unfilled open-bay remainder")
    _require(any(f["panel_count_disagreements"] for f in train), "panel-count shortcut not excluded")
    _require(any(f["opposite_courtyards"] for f in train), "independent courtyard votes not demonstrated")
    _require(any(f["raw_pixel_disagreement"] for f in train), "local panel-majority step not necessary")
    _require(any(f["straight_distance_disagreement"] for f in train), "graph distance not distinguished")
    _require(any(f["ties"] for f in train), "neutral distance ties not demonstrated")
    max_train_holes = max(f["holes"] for f in train)
    # Count sources on active bodies: contrast scenes duplicate three panels on
    # an inactive body, which must not conceal the actual source-count shift.
    active_counts = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            z = _analyze(pair["input"])
            active_counts[split].append(sum(p["body"] in z["active_ids"] for p in z["panels"]))
    _require(all(f["holes"] > max_train_holes for f in features["test"]) or
             min(active_counts["test"]) > max(active_counts["train"]), "missing count-based train/test shift")


def _regressions():
    # A fixed, author-inspected new-family input, not a parent fixture. Its
    # stepped upper courtyard exposes the cell-versus-edge distinction.
    rows = (
        "00380008888800000000008888", "00188888888100000000000000",
        "00188888888300000000000000", "00188888888300000000000000",
        "00880008388800000000000000", "00880008388800000000000000",
        "00888008188800000000000000", "00888008888800000000000000",
        "00888008888800000000000000", "00888888888800000000000000",
        "00888888888800000000000000", "00888888888808880000000000",
        "00888880008808880000000000", "00388880008800000000000000",
        "00188880008800000000000000", "00188888888800000000000000",
        "00188888888800000000000000", "00188800088800000000000000")
    a = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
    z = _analyze(a)
    first, second = z["courtyards"]
    _require(first["counts"] == [7, 8] and first["winner"] == 3, "distinct rim-cell regression")
    edge_counts = [sum(bool(z["material"][r + dr, c + dc] and z["wave"][r + dr, c + dc] == s)
                       for r, c in first["cells"] for dr, dc in D4) for s in z["signals"]]
    _require(edge_counts == [8, 8], "fixture must distinguish cell votes from edge votes")
    out = solve(a)
    _require(out[4, 4] == 3 and out[12, 7] == 1 and second["counts"] == [7, 5], "separate courtyard elections")
    _require(out[2, 6] == SCAFFOLD, "distance tie must abstain")
    _require(out[0, 2] == 3 and out[0, 3] == 1, "preserve minority pixel but launch panel majority")
    for signals in ((9, 2), (4, 7), (6, 5)):
        lookup = np.arange(10, dtype=np.int8)
        lookup[1], lookup[3] = signals
        _require(np.array_equal(solve(lookup[a]), lookup[out]), "signal-color renaming failed")
    # Two independent active bodies in one input: the solver must not rely on
    # the sampler's usual one-active-body arrangement.
    body = a[:, 2:12]
    tile_out = solve(body)
    twin = np.concatenate((body, np.zeros((len(body), 1), dtype=np.int8), body), axis=1)
    twin_out = solve(twin)
    _require(len(_analyze(twin)["active_ids"]) == 2, "multi-active-body fixture")
    _require(np.array_equal(twin_out[:, :10], tile_out) and np.array_equal(twin_out[:, 11:], tile_out),
             "signals crossed disconnected material")


def self_test():
    """42 seeds in memory; no files, worker pools, or retained run payloads."""
    _regressions()
    shapes, structures, areas, rims, palettes, margins = set(), set(), set(), set(), set(), set()
    records = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                f = _features(a)
                records.append(f)
                shapes.add(a.shape)
                structures.add(hashlib.sha256((a != 0).astype(np.uint8).tobytes()).digest())
                areas.update(f["hole_areas"])
                rims.update(f["rim_counts"])
                margins.update(f["margins"])
                palettes.add(f["signals"])
                # Translation, reflection and rotation preserve all semantics.
                if split == "test":
                    for turns, flip in ((1, False), (2, True)):
                        inp = _orient(a, turns, flip)
                        expected = _orient(pair["output"], turns, flip)
                        _require(np.array_equal(solve(inp), expected), "orientation covariance failed")
                    padded = np.pad(a, 1)
                    if max(padded.shape) <= 30:
                        predicted = solve(padded)[1:-1, 1:-1]
                        _require(np.array_equal(predicted, pair["output"]), "translation/padding changed interior result")
    _require(len(shapes) >= 25 and len(structures) >= 180, "insufficient structural variation")
    _require(len(areas) >= 12 and len(rims) >= 10 and len(margins) >= 5, "insufficient courtyard variation")
    _require(len(palettes) >= 15, "insufficient signal palette variation")
    _require(any(f["ties"] for f in records) and any(f["irregular_holes"] for f in records), "missing ties or irregular cavities")
    _require({1, 2, 3} <= {f["holes"] for f in records}, "courtyard-count coverage")
    _require(len({f["objects"] for f in records}) >= 3, "object-count variation")
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            puzzle = generate_puzzle(42000 + 10 * tr + te, tr, te)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (tr, te), "explicit count mismatch")
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first, second = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    _require(all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], second[s])
                 for k in ("input", "output")), "nondeterministic puzzle")
    # Corrupt each semantically different output zone and demand rejection.
    base = generate_puzzle(0, 2, 1)
    for role in ("panel_mask", "outline", "holes", "active"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in base[s]] for s in base}
        pair = bad["train"][0]
        z = _analyze(pair["input"])
        mask = z[role] if role != "active" else z["active"] & z["scaffold"]
        r, c = np.argwhere(mask)[0]
        pair["output"][r, c] = (int(pair["output"][r, c]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted corrupted " + role)


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds; no samples saved)")
