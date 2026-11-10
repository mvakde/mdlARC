"""Dye-certified seals: geometry proposes repairs; keyed payload certifies them.

No parent generator is imported. Construction labels use sampled cavity masks;
solve reconstructs walls, cues, payloads and virtual cavities from input pixels.
"""
import json

import numpy as np


FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    seen = np.zeros(mask.shape, bool)
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        cells, stack = [], [(r, c)]
        seen[r, c] = True
        while stack:
            q = stack.pop()
            cells.append(q)
            for dr, dc in EIGHT if diagonal else FOUR:
                nr, nc = q[0] + dr, q[1] + dc
                if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]
                        and mask[nr, nc] and not seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(cells)
    return result


def _dilate(mask):
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[r:r+mask.shape[0], c:c+mask.shape[1]]
                                 for r in range(3) for c in range(3)])


def _orthogonal(mask):
    """Pixels supported by at least one same-color orthogonal neighbor."""
    support = np.zeros(mask.shape, bool)
    pair = mask[1:] & mask[:-1]
    support[1:] |= pair
    support[:-1] |= pair
    pair = mask[:, 1:] & mask[:, :-1]
    support[:, 1:] |= pair
    support[:, :-1] |= pair
    return support


def _at(mask, q):
    r, c = q
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _key(grid, bg):
    candidates = []
    for cells in _components(grid != bg, True):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        h, w = r1-r0+1, c1-c0+1
        if len(cells) != h*w or len({int(grid[q]) for q in cells}) < 2:
            continue
        lanes = []
        if h == 2 and 3 <= w <= 6:
            if r0 == 0:
                lanes.append(("top", grid[r0, c0:c1+1], grid[r1, c0:c1+1]))
            if r1 == grid.shape[0]-1:
                lanes.append(("bottom", grid[r1, c0:c1+1], grid[r0, c0:c1+1]))
        if w == 2 and 3 <= h <= 6:
            if c0 == 0:
                lanes.append(("left", grid[r0:r1+1, c0], grid[r0:r1+1, c1]))
            if c1 == grid.shape[1]-1:
                lanes.append(("right", grid[r0:r1+1, c1], grid[r0:r1+1, c0]))
        for side, source, fill in lanes:
            if len(set(map(int, source))) == len(source):
                candidates.append((side, dict(zip(map(int, source), map(int, fill))), cells))
    if len(candidates) != 1:
        raise ValueError("expected one isolated, unambiguous edge key")
    side, mapping, cells = candidates[0]
    mask = np.zeros(grid.shape, bool)
    for q in cells:
        mask[q] = True
    return side, mapping, mask


def _candidates(grid, bg, walls, debris):
    found = []
    for color in sorted(set(map(int, grid[walls]))):
        same_color = grid == color
        mask = walls & same_color
        weak = debris & same_color
        for r, c in np.argwhere(mask):
            r, c = int(r), int(c)
            for dr, dc in ((0, 1), (1, 0)):
                end = (r+4*dr, c+4*dc)
                gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
                if (not _at(mask, end)
                        or any(not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1])
                               or grid[y, x] != bg for y, x in gap)
                        or not (_at(mask, (r-dr, c-dc)) or _at(mask, (r+5*dr, c+5*dc)))):
                    continue
                matches = []
                for endpoint in (0, 4):
                    ar, ac = r+endpoint*dr, c+endpoint*dc
                    toward = 1 if endpoint == 0 else -1
                    for side in (-1, 1):
                        sr, sc = toward*dr+side*dc, toward*dc+side*dr
                        spur = tuple((ar+k*sr, ac+k*sc) for k in (1, 2))
                        third = (ar+3*sr, ac+3*sc)
                        if all(_at(weak, q) for q in spur) and not _at(same_color, third):
                            matches.append(spur)
                if matches:
                    if len(matches) != 1:
                        raise ValueError("ambiguous diagonal cue")
                    found.append({"color": color, "gap": gap, "spur": matches[0],
                                  "axis": "h" if dc else "v"})
    used_gaps, used_spurs = set(), set()
    for item in found:
        if used_gaps.intersection(item["gap"]) or used_spurs.intersection(item["spur"]):
            raise ValueError("overlapping repair proposals")
        used_gaps.update(item["gap"])
        used_spurs.update(item["spur"])
    return found


def _decode(input_grid, proposals=True):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("background must have a unique largest input count")
    bg = int(colors[np.argmax(counts)])
    side, mapping, key = _key(grid, bg)
    ink = (grid != bg) & ~key
    markers = np.zeros(grid.shape, bool)
    for cells in _components(ink, True):
        if len(cells) == 1:
            markers[cells[0]] = True
    walls = np.zeros(grid.shape, bool)
    for color in colors:
        if color != bg:
            walls |= _orthogonal(ink & (grid == color))
    debris = ink & ~walls & ~markers
    candidates = _candidates(grid, bg, walls, debris) if proposals else []
    virtual = np.full(grid.shape, bg, np.int8)
    virtual[walls] = grid[walls]
    for item in candidates:
        for q in item["gap"]:
            virtual[q] = item["color"]
    rooms = _chambers(virtual, bg, markers, grid, mapping)
    return {"grid": grid, "bg": bg, "side": side, "mapping": mapping, "key": key,
            "markers": markers, "walls": walls, "debris": debris,
            "candidates": candidates, "rooms": rooms, "virtual": virtual}


def _chambers(virtual, bg, markers, grid, mapping):
    rooms = []
    for cells in _components(virtual == bg):
        if any(r in (0, grid.shape[0]-1) or c in (0, grid.shape[1]-1) for r, c in cells):
            continue
        boundary = set()
        for r, c in cells:
            for dr, dc in FOUR:
                q = r+dr, c+dc
                if virtual[q] != bg:
                    boundary.add(q)
        owners = {int(virtual[q]) for q in boundary}
        if len(owners) != 1:
            raise ValueError("virtual chamber has more than one wall color")
        color = owners.pop()
        payload = tuple(q for q in cells if markers[q])
        dyes = {int(grid[q]) for q in payload}
        accepted = color in mapping and bool(payload) and dyes == {mapping[color]}
        rooms.append({"cells": cells, "boundary": boundary, "color": color,
                      "payload": payload, "dyes": dyes, "accepted": accepted})
    return rooms


def _prediction(info, ignore_key=False):
    out = info["grid"].copy()
    out[info["debris"]] = info["bg"]
    accepted = [r for r in info["rooms"] if r["accepted"] or
                (ignore_key and len(r["dyes"]) == 1)]
    for item in info["candidates"]:
        if any(set(item["gap"]) <= room["boundary"] for room in accepted):
            for q in item["gap"]:
                out[q] = item["color"]
    for room in accepted:
        fill = (next(iter(room["dyes"])) if ignore_key else info["mapping"][room["color"]])
        for q in room["cells"]:
            out[q] = fill
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """An input-only, simultaneous virtual-enclosure rule; no generation state."""
    return _prediction(_decode(input_grid))


def _local_rooms(rng, count, regular=False, lesson=False):
    """Grow unequal rooms with shared walls, rather than use fixed canvas slots."""
    if regular:
        # Spacious bounded fallback: only construction placement changes.
        boxes = [(7*(i//2), 7*(i % 2), 6, 6) for i in range(count)]
    else:
        boxes = [(0, 0, int(rng.integers(5, 9)), int(rng.integers(5, 9)))]
    attempts = 0
    while len(boxes) < count and attempts < 180:
        attempts += 1
        r, c, h, w = boxes[int(rng.integers(len(boxes)))]
        nh, nw = int(rng.integers(5, 9)), int(rng.integers(5, 9))
        direction = int(rng.integers(4))
        if direction < 2:
            nr = r+int(rng.integers(-min(2, nh-3), min(3, h-2)))
            nc = c+w+1 if direction == 0 else c-nw-1
        else:
            nr = r+h+1 if direction == 2 else r-nh-1
            nc = c+int(rng.integers(-min(2, nw-3), min(3, w-2)))
        candidate = (nr, nc, nh, nw)
        if any(not (nr+nh < y or y+hh < nr or nc+nw < x or x+ww < nc)
               for y, x, hh, ww in boxes):
            continue
        trial = boxes+[candidate]
        minr, minc = min(b[0] for b in trial), min(b[1] for b in trial)
        maxr, maxc = max(b[0]+b[2] for b in trial), max(b[1]+b[3] for b in trial)
        if maxr-minr <= 23 and maxc-minc <= 23:
            boxes = trial
    if len(boxes) != count:
        return None
    rmin, cmin = min(b[0] for b in boxes)-1, min(b[1] for b in boxes)-1
    height = max(b[0]+b[2] for b in boxes)-rmin+1
    width = max(b[1]+b[3] for b in boxes)-cmin+1
    rooms = []
    for r, c, h, w in boxes:
        mask = np.zeros((height, width), bool)
        patch = np.ones((h, w), bool)
        # Large corner notches create L and stepped chambers and silhouettes.
        if (not regular and h >= 6 and w >= 6 and rng.random() < .65) or (regular and lesson and len(rooms) == count-1):
            nh, nw = int(rng.integers(2, h-2)), int(rng.integers(2, w-2))
            patch[:nh, :nw] = False
            patch = np.rot90(patch, 2*int(rng.integers(2)))
            if rng.integers(2):
                patch = patch[:, ::-1]
        mask[r-rmin:r-rmin+h, c-cmin:c-cmin+w] = patch
        rooms.append(mask)
    union = np.logical_or.reduce(rooms)
    walls = _dilate(union) & ~union
    if len(_components(walls)) != 1:
        return None
    return rooms, walls


def _damage_options(full, rooms):
    ids = np.full(full.shape, -1, int)
    for i, room in enumerate(rooms):
        ids[room] = i
    options = []
    for r, c in np.argwhere(full):
        r, c = int(r), int(c)
        for dr, dc in ((0, 1), (1, 0)):
            gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
            if not all(_at(full, (r+k*dr, c+k*dc)) for k in (-1, 0, 1, 2, 3, 4, 5)):
                continue
            # Repairs occupy thin straight wall sections, clear of junctions.
            if any(_at(full, (y+dc, x+dr)) or _at(full, (y-dc, x-dr)) for y, x in gap):
                continue
            neighbors = set()
            for y, x in gap:
                for sr, sc in FOUR:
                    yy, xx = y+sr, x+sc
                    if 0 <= yy < ids.shape[0] and 0 <= xx < ids.shape[1] and ids[yy, xx] >= 0:
                        neighbors.add(int(ids[yy, xx]))
            owners = {i for i in neighbors if all(any(_at(rooms[i], (y+sr, x+sc))
                                                    for sr, sc in FOUR) for y, x in gap)}
            if not owners:
                continue
            external = len(owners) == 1
            for endpoint in (0, 4):
                ar, ac = r+endpoint*dr, c+endpoint*dc
                toward = 1 if endpoint == 0 else -1
                for side in (-1, 1):
                    sr, sc = toward*dr+side*dc, toward*dc+side*dr
                    spur = tuple((ar+k*sr, ac+k*sc) for k in (1, 2))
                    if not all(0 <= y < full.shape[0] and 0 <= x < full.shape[1]
                               and not full[y, x] for y, x in spur):
                        continue
                    trial = full.copy()
                    for q in gap:
                        trial[q] = False
                    wall = _orthogonal(trial)
                    ink = trial.copy()
                    for q in spur:
                        ink[q] = True
                    if np.array_equal(wall, trial) and np.array_equal(_orthogonal(ink), trial):
                        options.append({"gap": gap, "spur": spur, "owners": owners,
                                        "external": external})
    return options


def _put_key(grid, side, mapping, start):
    for i, (source, fill) in enumerate(mapping.items()):
        p = start+i
        if side == "top":
            grid[0, p], grid[1, p] = source, fill
        elif side == "bottom":
            grid[-1, p], grid[-2, p] = source, fill
        elif side == "left":
            grid[p, 0], grid[p, 1] = source, fill
        else:
            grid[p, -1], grid[p, -2] = source, fill


def _construct_pair(rng, side, statuses, color, force_double=False, force_shared=False,
                    secondary=False, lesson=False, regular=False):
    local = _local_rooms(rng, len(statuses), regular, lesson)
    if local is None:
        return None
    local_rooms, local_wall = local
    lh, lw = local_wall.shape
    # A free, variable margin locates the cluster independently of key position.
    h, w = min(30, lh+int(rng.integers(4, 8))), min(30, lw+int(rng.integers(4, 8)))
    if secondary:
        h, w = max(h, int(rng.integers(26, 31))), max(w, int(rng.integers(26, 31)))
    if side in ("top", "bottom"):
        h = max(h, lh+4)
    else:
        w = max(w, lw+4)
    if h > 30 or w > 30:
        return None
    rlow, rhigh = (3, h-lh) if side == "top" else (0, h-lh-3) if side == "bottom" else (0, h-lh)
    clow, chigh = (3, w-lw) if side == "left" else (0, w-lw-3) if side == "right" else (0, w-lw)
    r, c = int(rng.integers(rlow, rhigh+1)), int(rng.integers(clow, chigh+1))
    rooms = []
    full = np.zeros((h, w), bool)
    full[r:r+lh, c:c+lw] = local_wall
    for mask in local_rooms:
        dest = np.zeros((h, w), bool)
        dest[r:r+lh, c:c+lw] = mask
        rooms.append(dest)
    nkey = int(rng.integers(3, 7))
    mapping = {1: 2, 2: 7, 3: 3}
    for source in range(4, nkey+1):
        mapping[source] = int(rng.integers(1, 10))
    order = rng.permutation(list(mapping))
    mapping = {int(k): mapping[int(k)] for k in order}
    # Color 1 teaches one-step chain lookup; color 3 teaches self-mapping.
    if color is None:
        color = int(rng.choice(list(mapping)))
    fill = mapping[color]
    options = _damage_options(full, rooms)
    selected, occupied = [], np.zeros(full.shape, bool)
    current = full.copy()
    for i, status in enumerate(statuses):
        if status.endswith("closed"):
            continue
        times = 2 if force_double and i == 0 else 1
        for k in range(times):
            pool = [o for o in options if i in o["owners"] and
                    (o["external"] or (force_shared and i == 0 and k == 0))]
            if force_shared and i == 0 and k == 0:
                pool = [o for o in pool if not o["external"]]
            rng.shuffle(pool)
            for option in pool:
                footprint = np.zeros(full.shape, bool)
                for q in option["gap"]+option["spur"]:
                    footprint[q] = True
                if np.any(footprint & _dilate(occupied)):
                    continue
                # Uncued openings must reach exterior, never a neighboring room.
                if status == "match_uncued" and not option["external"]:
                    continue
                trial = current.copy()
                for q in option["gap"]:
                    trial[q] = False
                if not np.array_equal(_orthogonal(trial), trial):
                    continue
                current = trial
                occupied |= footprint
                selected.append((option, status != "match_uncued"))
                break
            else:
                return None
    inp = np.zeros(full.shape, np.int8)
    inp[current] = color
    cue_mask = np.zeros(full.shape, bool)
    for option, cue in selected:
        if cue:
            for q in option["spur"]:
                inp[q] = color
                cue_mask[q] = True
    marker_mask = np.zeros(full.shape, bool)
    for i, status in enumerate(statuses):
        if status.startswith("empty"):
            continue
        colors = [fill]
        if status.startswith("wrong"):
            colors = [int(rng.choice([x for x in range(1, 10) if x != fill]))]
        elif status.startswith("mixed"):
            colors.append(int(rng.choice([x for x in range(1, 10) if x != fill])))
        elif (lesson and i == 0) or rng.random() < .35:
            colors *= 2
        for dye in colors:
            # Singletons remain isolated even after hypothetical completion.
            safe = rooms[i] & ~_dilate(full | cue_mask | marker_mask)
            choices = np.argwhere(safe)
            if not len(choices):
                return None
            q = tuple(map(int, choices[int(rng.integers(len(choices)))]))
            inp[q] = dye
            marker_mask[q] = True
    # Extra detached two/three-cell diagonals are erased, never dye markers.
    for _ in range(int(rng.integers(3))):
        for _ in range(12):
            y, x = int(rng.integers(h)), int(rng.integers(w))
            dy, dx = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
            coords = [(y+k*dy, x+k*dx) for k in range(int(rng.integers(2, 4)))]
            blocked = _dilate((inp != 0) | full)
            if all(0 <= yy < h and 0 <= xx < w and not blocked[yy, xx] for yy, xx in coords):
                for q in coords:
                    inp[q] = color
                break
    axis = w if side in ("top", "bottom") else h
    starts = list(range(axis-len(mapping)+1))
    rng.shuffle(starts)
    for start in starts:
        keygrid = np.zeros(inp.shape, np.int8)
        _put_key(keygrid, side, mapping, start)
        if not np.any(_dilate(keygrid != 0) & (inp != 0)):
            inp[keygrid != 0] = keygrid[keygrid != 0]
            break
    else:
        return None
    # Construction labels know cavities and sampled cue boundaries, not solve().
    accepted_ids = {i for i, s in enumerate(statuses) if s in ("match_repair", "match_closed")}
    out = np.zeros(inp.shape, np.int8)
    out[current] = color
    for option, cue in selected:
        if cue and option["owners"] & accepted_ids:
            for q in option["gap"]:
                out[q] = color
    out[marker_mask] = inp[marker_mask]
    for i in accepted_ids:
        out[rooms[i]] = fill
    out[keygrid != 0] = keygrid[keygrid != 0]
    if secondary:
        # A second mapped wall identity defeats treating the key as one global
        # dye. It shares this rule, while the principal floor plan stays joined.
        inner_h, inner_w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        sh, sw = inner_h+2, inner_w+2
        blocked = _dilate((inp != 0) | full | np.logical_or.reduce(rooms))
        positions = [(y, x) for y in range(h-sh+1) for x in range(w-sw+1)
                     if not blocked[y:y+sh, x:x+sw].any()]
        if not positions:
            return None
        y, x = positions[int(rng.integers(len(positions)))]
        second_color = int(rng.choice([k for k in mapping if k != color]))
        if lesson:
            second_color = 2
        second_fill = mapping[second_color]
        patch = np.full((sh, sw), second_color, np.int8)
        patch[1:-1, 1:-1] = 0
        py, px = 2, 2
        patch[py, px] = second_fill
        inp[y:y+sh, x:x+sw] = patch
        patch_out = patch.copy()
        patch_out[1:-1, 1:-1] = second_fill
        out[y:y+sh, x:x+sw] = patch_out
    palette = rng.permutation(10).astype(np.int8)
    inp, out = palette[inp], palette[out]
    try:
        info = _decode(inp)
        if not np.array_equal(_prediction(info), out):
            return None
        if not any(r["accepted"] for r in info["rooms"]):
            return None
        if force_shared and not any(sum(set(item["gap"]) <= room["boundary"] for room in info["rooms"]) == 2
                                   for item in info["candidates"]):
            return None
        if force_double and not any(room["accepted"] and sum(set(o["gap"]) <= room["boundary"]
                                                             for o in info["candidates"]) >= 2
                                    for room in info["rooms"]):
            return None
        if lesson and not any(room["accepted"] and len(room["cells"]) !=
                              (max(y for y, x in room["cells"])-min(y for y, x in room["cells"])+1)*
                              (max(x for y, x in room["cells"])-min(x for y, x in room["cells"])+1)
                              for room in info["rooms"]):
            return None
    except ValueError:
        return None
    return {"input": inp, "output": out}


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, t = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if n <= t:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    sides = list(map(str, rng.permutation(["top", "bottom", "left", "right"])))
    shift = int(rng.integers(3))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", n), ("test", t)):
        for index in range(count):
            if split == "train" and index == 0:
                statuses, color = ["match_repair", "wrong_repair", "match_closed"], 1
                double, shared = False, True
            elif split == "train" and index == 1:
                statuses, color = ["match_repair", "match_uncued", "mixed_repair", "empty_closed"], 3
                double, shared = True, False
            else:
                number = int(rng.integers(2, 5 if split == "train" else 6))
                statuses = ["match_repair"] + list(map(str, rng.choice(
                    ["wrong_repair", "match_closed", "empty_repair", "mixed_repair", "match_uncued"], number-1)))
                color, double, shared = None, bool(rng.integers(2)), False
            side = sides[index % 2] if split == "train" else sides[2] if shift == 0 else sides[index % 2]
            lesson = split == "train" and index == 0
            secondary = lesson or bool(rng.integers(2))
            if split == "test" and shift == 1:
                # Training separates a shared seal from a two-seal chamber;
                # test recombines them in the same chamber, on a familiar side.
                double, shared = True, True
                statuses = ["match_repair", "wrong_repair"] + list(map(str, rng.choice(
                    ["match_closed", "empty_closed"], int(rng.integers(0, 3)))))
            if split == "test" and shift == 2:
                # A larger joined floor plan recombines the trained contrasts.
                statuses = ["match_repair", "wrong_repair", "match_closed", "match_uncued", "mixed_repair"]
            for attempt in range(220):
                pair = _construct_pair(rng, side, statuses, color, double, shared, secondary, lesson,
                                       regular=attempt >= 120)
                if pair is not None and _fingerprint(pair["input"]) not in seen:
                    seen.add(_fingerprint(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError(f"bounded room construction failed: seed={seed}, {split}[{index}], "
                                   f"statuses={statuses}, double={double}, shared={shared}, secondary={secondary}")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    accepted = [r for r in info["rooms"] if r["accepted"]]
    candidate_accepts = [any(set(o["gap"]) <= r["boundary"] for r in accepted) for o in info["candidates"]]
    matched_uncued = _uncued_contrast(info)
    return {"accepted_repairs": sum(candidate_accepts), "rejected_repairs": sum(not x for x in candidate_accepts),
            "shared": any(sum(set(o["gap"]) <= r["boundary"] for r in info["rooms"]) >= 2 for o in info["candidates"]),
            "double": any(sum(set(o["gap"]) <= r["boundary"] for o in info["candidates"]) >= 2 for r in accepted),
            "chain": any(info["mapping"][r["color"]] in info["mapping"] and
                         info["mapping"][info["mapping"][r["color"]]] != info["mapping"][r["color"]] for r in accepted),
            "self": any(info["mapping"][r["color"]] == r["color"] for r in accepted),
            "empty": any(not r["payload"] for r in info["rooms"]),
            "mixed": any(len(r["dyes"]) > 1 for r in info["rooms"]), "uncued": matched_uncued,
            "room_count": len(info["rooms"]), "side": info["side"], "bg": info["bg"],
            "key_length": len(info["mapping"]),
            "wall_colors": len(set(map(int, info["grid"][info["walls"]]))),
            "same_dye_multiple": any(r["accepted"] and len(r["payload"]) > 1 for r in info["rooms"]),
            "accepted_irregular": any(r["accepted"] and len(r["cells"]) !=
                                      (max(y for y, x in r["cells"])-min(y for y, x in r["cells"])+1)*
                                      (max(x for y, x in r["cells"])-min(x for y, x in r["cells"])+1)
                                      for r in info["rooms"]),
            "irregular": any(len(r["cells"]) != (max(y for y, x in r["cells"])-min(y for y, x in r["cells"])+1)*
                             (max(x for y, x in r["cells"])-min(x for y, x in r["cells"])+1) for r in info["rooms"])}


def _uncued_contrast(info):
    """Check that an unsupported gap would seal a correctly dyed chamber.

    This is a training-witness measurement, never part of solve's repair rule.
    It distinguishes a real leak contrast from a random exterior dye singleton.
    """
    virtual, bg = info["virtual"], info["bg"]
    actual = {frozenset(o["gap"]) for o in info["candidates"]}
    for r, c in np.argwhere(virtual != bg):
        r, c = int(r), int(c)
        color = int(virtual[r, c])
        if color not in info["mapping"]:
            continue
        wall = virtual == color
        for dr, dc in ((0, 1), (1, 0)):
            gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
            if (frozenset(gap) in actual or not _at(wall, (r+4*dr, c+4*dc))
                    or any(virtual[q] != bg or info["markers"][q] for q in gap)
                    or not (_at(wall, (r-dr, c-dc)) or _at(wall, (r+5*dr, c+5*dc)))):
                continue
            trial = virtual.copy()
            for q in gap:
                trial[q] = color
            try:
                rooms = _chambers(trial, bg, info["markers"], info["grid"], info["mapping"])
            except ValueError:
                continue
            if any(room["accepted"] and set(gap) <= room["boundary"] for room in rooms):
                return True
    return False


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    n, t = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            fp = _fingerprint(inp)
            assert fp not in seen
            seen.add(fp)
            info = _decode(inp)
            # Every weak stroke is geometrically diagonal; arbitrary blobs are
            # not quietly discarded as if they were cues or payloads.
            for component in _components(info["debris"], True):
                points = sorted(component)
                assert len(points) in (2, 3), "malformed diagonal debris"
                assert len({int(inp[q]) for q in points}) == 1
                assert all(points[i+1][0]-points[i][0] == 1 and
                           abs(points[i+1][1]-points[i][1]) == 1 for i in range(len(points)-1))
                assert len({points[i+1][1]-points[i][1] for i in range(len(points)-1)}) == 1
            assert np.array_equal(_prediction(info), out), "incorrect complete transformation"
            assert np.array_equal(inp[info["key"]], out[info["key"]]), "key changed"
            assert np.array_equal(inp[info["walls"]], out[info["walls"]]), "wall changed"
            assert np.array_equal(inp[info["markers"]], out[info["markers"]]), "payload changed"
            assert not np.array_equal(inp, out)
            assert not np.array_equal(_prediction(_decode(inp, proposals=False)), out), "A has no effect"
            if split == "train":
                train.append((info, _features(info)))
    # Actual output and input geometry expose the two mechanisms and contrasts.
    for name in ("shared", "double", "chain", "self", "empty", "mixed", "uncued",
                 "accepted_irregular", "same_dye_multiple"):
        assert any(f[name] for _, f in train), "missing training witness: "+name
    assert any(f["wall_colors"] >= 2 for _, f in train)
    assert any(f["rejected_repairs"] for _, f in train)
    assert any(not np.array_equal(_prediction(info, ignore_key=True), _prediction(info)) for info, _ in train), "B has no effect"


def self_test():
    """Bounded 42-seed suite, semantic ablations, symmetry and bad-label checks."""
    puzzles, features, shapes, structures = [], [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                assert np.array_equal(solve(a.copy()), b)
                shapes.add(a.shape)
                info = _decode(a)
                features.append(_features(info))
                roles = {}
                structures.add((a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)))
    assert len({_fingerprint(p["test"][0]["input"]) for p in puzzles}) == 42
    assert len(shapes) >= 30 and len(structures) >= 150
    assert {f["side"] for f in features} == {"top", "bottom", "left", "right"}
    assert {f["bg"] for f in features} == set(range(10))
    assert {f["key_length"] for f in features} == {3, 4, 5, 6}
    assert any(f["irregular"] for f in features)
    assert max(f["room_count"] for f in features) >= 4
    # Same input, different seeds and train/test flags never enter solve.
    repeated = generate_puzzle(0)
    assert all(_fingerprint(puzzles[0][s][i][k]) == _fingerprint(repeated[s][i][k])
               for s in ("train", "test") for i in range(len(puzzles[0][s])) for k in ("input", "output"))
    pair = puzzles[0]["train"][0]
    for turns in range(4):
        for reflected in (False, True):
            a, b = np.rot90(pair["input"], turns), np.rot90(pair["output"], turns)
            if reflected:
                a, b = a[:, ::-1], b[:, ::-1]
            assert np.array_equal(solve(a), b)
    permutation = np.array([7, 3, 1, 9, 0, 6, 8, 2, 5, 4], np.int8)
    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    # Full-label negative regression: omit just one certified seal.
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ("train", "test")}
    info = _decode(broken["train"][0]["input"])
    item = next(o for o in info["candidates"] if any(r["accepted"] and set(o["gap"]) <= r["boundary"] for r in info["rooms"]))
    broken["train"][0]["output"][item["gap"][0]] = info["bg"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing seal accepted")
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            p = generate_puzzle(42+n*5+t, n, t)
            assert len(p["train"]) == n and len(p["test"]) == t
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": len(features), "shapes": len(shapes), "normalized_structures": len(structures),
            "irregular_pairs": sum(f["irregular"] for f in features),
            "shared_wall_pairs": sum(f["shared"] for f in features),
            "double_seal_pairs": sum(f["double"] for f in features),
            "multiple_wall_color_pairs": sum(f["wall_colors"] >= 2 for f in features)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
