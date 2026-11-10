"""Repair floodgates from enlarged keyed patches, then ink the outside rim.

Only standard-library modules and NumPy are used. No examples, seeds, palettes,
or construction metadata are consulted by solve(). No files are written here.
"""

from collections import Counter
from math import isqrt
import hashlib

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a.astype(np.int8, copy=True)


def _flood(mask, start):
    reached = np.zeros(mask.shape, bool)
    if not mask[start]:
        return reached
    reached[start] = True
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in N4:
            x, y = r + dr, c + dc
            if 0 <= x < h and 0 <= y < w and mask[x, y] and not reached[x, y]:
                reached[x, y] = True
                stack.append((x, y))
    return reached


def _components(mask):
    remaining = mask.copy()
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if remaining[r, c]:
            found = _flood(remaining, (r, c))
            remaining[found] = False
            result.append(found)
    return result


def _adj(mask, diagonal=False):
    h, w = mask.shape
    p = np.pad(mask, 1)
    result = np.zeros_like(mask)
    offsets = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
               if (r, c) != (0, 0)] if diagonal else N4
    for dr, dc in offsets:
        result |= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


def _bounds(mask):
    rr, cc = np.nonzero(mask)
    return int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1


def _crop(a, bounds):
    r0, r1, c0, c1 = bounds
    return a[r0:r1, c0:c1]


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _d4(a, reflect=True):
    seen = set()
    for flip in range(2 if reflect else 1):
        base = np.fliplr(a) if flip else a
        for turn in range(4):
            transformed = np.rot90(base, turn).copy()
            key = _key(transformed)
            if key not in seen:
                seen.add(key)
                yield transformed


def _shape_key(mask):
    local = _crop(mask, _bounds(mask))
    return min(_key(m) for m in _d4(local))


def _expand(a, k):
    return np.repeat(np.repeat(a, k, axis=0), k, axis=1)


def _rim(selected):
    external = _flood(np.pad(~selected, 1, constant_values=True), (0, 0))
    return selected & _adj(external, True)[1:-1, 1:-1], external


def _paint(final, roles, k):
    bg, wall, patch, ink = roles
    del patch
    seed = tuple(map(int, np.argwhere(final == ink)[0]))
    selected = _flood((final == bg) | (final == ink), seed)
    physical = _expand(final, k)
    target = _expand(selected, k)
    rim, external = _rim(target)
    kept, erased = [], []
    logical = final.copy()
    for component in _components(final == wall):
        if np.any(_adj(selected) & component):
            kept.append(component)
        else:
            erased.append(component)
            logical[component] = bg
    output = _expand(logical, k)
    output[rim & (physical != ink)] = ink
    orth = _adj(external)[1:-1, 1:-1]
    islands = [m for m in kept if not (m[0].any() or m[-1].any()
                                      or m[:, 0].any() or m[:, -1].any())]
    # A visible floor pixel beside an island that remains uninked.
    island_evidence = any(np.any(_adj(_expand(m, k), True)
                                & target & (output == bg)) for m in islands)
    sizes = [int(m.sum()) for m in _components((final == bg) | (final == ink))]
    area = int(selected.sum())
    selection_class = "small" if area < max(sizes) else "equal" if sizes.count(area) > 1 else "large"
    return output, {
        "selected": selected, "rim": rim, "external": external,
        "kept_islands": len(islands), "erased": len(erased),
        "island_evidence": bool(island_evidence),
        "diagonal": int(np.count_nonzero(rim & ~orth & (physical != ink))),
        "compartments": len(sizes), "selection_class": selection_class,
        "selected_area": area,
    }


def _recover(input_grid):
    """Recover board, patch payloads and scale from the input alone.

    The union footprint of each two-color loose tile is fitted under D4. If
    interchangeable tiles or silhouette symmetries yield different wall/floor
    payloads, the input is rejected rather than resolved by position or color.
    """
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 4, "four visible roles required"
    bg = int(colors[np.argmax(counts)])
    assert counts.max() * 2 > grid.size, "background must occupy a majority"
    singletons = colors[counts == 1]
    assert len(singletons) == 1, "exactly one singleton seed role required"
    ink = int(singletons[0])
    candidates = []
    for wall in map(int, colors):
        if wall in (bg, ink):
            continue
        for component in _components(grid == wall):
            bounds = _bounds(component)
            board = _crop(grid, bounds)
            if min(board.shape) < 5 or np.count_nonzero(board == ink) != 1:
                continue
            if (np.all(board[0] == wall) and np.all(board[-1] == wall)
                    and np.all(board[:, 0] == wall) and np.all(board[:, -1] == wall)):
                candidates.append((wall, bounds, board.copy()))
    assert len(candidates) == 1, "unique framed blueprint required"
    wall, bounds, board = candidates[0]
    patch = next(int(c) for c in colors if c not in (bg, wall, ink))
    roles = (bg, wall, patch, ink)
    r0, r1, c0, c1 = bounds
    outside = np.ones(grid.shape, bool)
    outside[r0:r1, c0:c1] = False
    occupied = outside & (grid != bg)
    halo = np.zeros(grid.shape, bool)
    halo[max(0, r0 - 1):min(grid.shape[0], r1 + 1),
         max(0, c0 - 1):min(grid.shape[1], c1 + 1)] = True
    assert not np.any(occupied & halo), "blueprint must be separated"
    holes = _components(board == patch)
    objects = _components(occupied)
    assert len(holes) >= 2 and len(holes) == len(objects)
    area = sum(int(m.sum()) for m in holes)
    big_area = sum(int(m.sum()) for m in objects)
    ratio, remainder = divmod(big_area, area)
    k = isqrt(ratio)
    assert not remainder and k >= 2 and k * k == ratio, "common square scale required"
    assert max(board.shape) * k <= 30
    groups = {}
    for obj in objects:
        local = _crop(grid, _bounds(obj))
        support = _crop(obj, _bounds(obj))
        h, w = local.shape
        assert h % k == 0 and w % k == 0
        payload = np.full((h // k, w // k), -1, np.int8)
        for r in range(payload.shape[0]):
            for c in range(payload.shape[1]):
                block = local[r*k:(r+1)*k, c*k:(c+1)*k]
                present = support[r*k:(r+1)*k, c*k:(c+1)*k]
                if np.any(present):
                    assert np.all(present) and np.all(block == block[0, 0])
                    assert int(block[0, 0]) in (wall, patch)
                    payload[r, c] = block[0, 0]
        assert np.array_equal(_expand(payload != -1, k), support)
        groups.setdefault(_shape_key(payload != -1), []).append(payload)
    counts_by_shape = Counter(_shape_key(m) for m in holes)
    assert set(counts_by_shape) == set(groups)
    assert all(counts_by_shape[key] == len(groups[key]) for key in groups)
    final = board.copy()
    reflection, rotation = 0, 0
    for hole in holes:
        box = _bounds(hole)
        support = _crop(hole, box)
        pieces = groups[_shape_key(hole)]
        answers = {}
        for piece in pieces:
            for payload in _d4(piece):
                if np.array_equal(payload != -1, support):
                    answers[_key(payload)] = payload
        assert len(answers) == 1, "ambiguous patterned patch assignment"
        payload = next(iter(answers.values()))
        view = _crop(final, box)
        view[support] = np.where(payload[support] == wall, wall, bg)
        if len(pieces) == 1:
            source = pieces[0] != -1
            if not any(np.array_equal(m, support) for m in _d4(source, False)):
                reflection += 1
            elif not np.array_equal(source, support):
                rotation += 1
    seed = tuple(map(int, np.argwhere(board == ink)[0]))
    selected = _flood(final != wall, seed)
    all_open, all_closed = board.copy(), board.copy()
    all_open[board == patch] = bg
    all_closed[board == patch] = wall
    info = {
        "roles": roles, "scale": k, "bounds": bounds, "board": board,
        "final": final, "slot_shapes": tuple(sorted(counts_by_shape)),
        "piece_count": len(objects), "reflection": reflection, "rotation": rotation,
        "slot_areas": tuple(sorted(int(m.sum()) for m in holes)),
        # Count reachability changes outside the repair footprints, not merely
        # their own newly passable/blocked pixels.
        "open_critical": bool(np.any((selected != _flood(all_closed != wall, seed)) & (board != patch))),
        "closed_critical": bool(np.any((selected != _flood(all_open != wall, seed)) & (board != patch))),
        "board_pattern": _key(board == wall),
    }
    return final, info


def _toggle_payload_at(inp, destination):
    """Self-test helper: change one recovered logical tile cell in the input.

    Unlike editing a recovered board, this verifies that an actual loose tile's
    contents control the answer while its silhouette and blueprint stay fixed.
    """
    final, info = _recover(inp)
    bg, wall, patch, ink = info["roles"]
    del bg, ink
    board = info["board"]
    hole = next(m for m in _components(board == patch) if m[destination])
    box = _bounds(hole)
    support = _crop(hole, box)
    label = np.full(support.shape, -1, np.int16)
    for index, (r, c) in enumerate(np.argwhere(support)):
        label[r, c] = index
    label_index = int(label[destination[0] - box[0], destination[1] - box[2]])
    outside = np.ones(inp.shape, bool)
    r0, r1, c0, c1 = info["bounds"]
    outside[r0:r1, c0:c1] = False
    for obj in _components(outside & (inp != info["roles"][0])):
        obr0, obr1, obc0, obc1 = _bounds(obj)
        k = info["scale"]
        local_support = _crop(obj, _bounds(obj))[::k, ::k]
        if _shape_key(local_support) != _shape_key(support):
            continue
        placements = [a for a in _d4(label) if np.array_equal(a != -1, local_support)]
        assert len(placements) == 1  # generated repair silhouettes are asymmetric
        r, c = map(int, np.argwhere(placements[0] == label_index)[0])
        changed = inp.copy()
        view = changed[obr0 + r*k:obr0 + (r+1)*k, obc0 + c*k:obc0 + (c+1)*k]
        view[:] = patch if int(view[0, 0]) == wall else wall
        return changed
    raise AssertionError("matching tile not found")


def solve(input_grid: np.ndarray) -> np.ndarray:
    final, info = _recover(input_grid)
    return _paint(final, info["roles"], info["scale"])[0]


def _partition(rng, h, w, count):
    wall = np.zeros((h, w), bool)
    wall[[0, -1], :] = True
    wall[:, [0, -1]] = True
    rooms = [(1, h - 1, 1, w - 1)]
    while len(rooms) < count:
        choices = []
        for i, (r0, r1, c0, c1) in enumerate(rooms):
            margin = 3 if len(rooms) == 1 else 2
            for axis, length in ((0, r1 - r0), (1, c1 - c0)):
                if length >= 2 * margin + 1:
                    choices.append((i, axis, margin))
        if not choices:
            return None
        i, axis, margin = choices[int(rng.integers(len(choices)))]
        r0, r1, c0, c1 = rooms.pop(i)
        if axis == 0:
            cut = int(rng.integers(r0 + margin, r1 - margin))
            wall[cut, c0:c1] = True
            rooms.extend(((r0, cut, c0, c1), (cut + 1, r1, c0, c1)))
        else:
            cut = int(rng.integers(c0 + margin, c1 - margin))
            wall[r0:r1, cut] = True
            rooms.extend(((r0, r1, c0, cut), (r0, r1, cut + 1, c1)))
    return wall


def _doors(wall):
    h, w = wall.shape
    result = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if not wall[r, c]:
                continue
            if not wall[r - 1, c] and not wall[r + 1, c] and (wall[r, c - 1] or wall[r, c + 1]):
                result.append(((r, c), (1, 0)))
            if not wall[r, c - 1] and not wall[r, c + 1] and (wall[r - 1, c] or wall[r + 1, c]):
                result.append(((r, c), (0, 1)))
    return result


def _island(rng, wall, region):
    available = region & ~_adj(wall, True)
    positions = np.argwhere(available)
    if not len(positions):
        return False
    start = tuple(map(int, positions[int(rng.integers(len(positions)))]))
    cells = {start}
    for _ in range(int(rng.integers(0, 4))):
        neighbors = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in N4}
                           - cells)
        choices = [p for p in neighbors if 0 <= p[0] < wall.shape[0]
                   and 0 <= p[1] < wall.shape[1] and available[p]]
        if not choices:
            break
        cells.add(choices[int(rng.integers(len(choices)))])
    for p in cells:
        wall[p] = True
    return True


def _patch(rng, wall, door, blocked, used_shapes):
    (r, c), (dr, dc) = door
    # Include floor on both sides of a wall and a neighboring wall cell. This
    # makes each gate tile a genuine wall/floor pattern rather than a color key.
    along = [(r - dc, c - dr), (r + dc, c + dr)]
    along = [p for p in along if wall[p]]
    if not along:
        return None
    for _ in range(48):
        cells = {(r, c), (r - dr, c - dc), (r + dr, c + dc),
                 along[int(rng.integers(len(along)))]}
        if any(blocked[p] for p in cells):
            return None
        area = int(rng.integers(5, 9))
        while len(cells) < area:
            frontier = sorted({(x + a, y + b) for x, y in cells for a, b in N4} - cells)
            choices = []
            for x, y in frontier:
                if not (0 < x < wall.shape[0] - 1 and 0 < y < wall.shape[1] - 1) or blocked[x, y]:
                    continue
                rr = [p[0] for p in cells] + [x]
                cc = [p[1] for p in cells] + [y]
                if max(rr) - min(rr) < 4 and max(cc) - min(cc) < 4:
                    choices.append((x, y))
            if not choices:
                break
            cells.add(choices[int(rng.integers(len(choices)))])
        mask = np.zeros(wall.shape, bool)
        for p in cells:
            mask[p] = True
        local = _crop(mask, _bounds(mask))
        key = _shape_key(mask)
        if len(list(_d4(local))) == 8 and key not in used_shapes and np.any(wall[mask]) and np.any(~wall[mask]):
            return mask
    return None


def _scene(rng, k, complex_test, active):
    limit = 30 // k
    h, w = int(rng.integers(9, limit + 1)), int(rng.integers(9, limit + 1))
    if rng.random() < .22:
        w = h
    room_count = int(rng.integers(4, 6)) if complex_test else int(rng.integers(3, 5))
    wall = _partition(rng, h, w, room_count)
    if wall is None:
        return None
    open_doors = []
    for _ in range(2 if complex_test else 1):
        candidates = _doors(wall)
        rng.shuffle(candidates)
        for door in candidates:
            p, (dr, dc) = door
            before = _flood(~wall, (p[0] - dr, p[1] - dc))
            if before[p[0] + dr, p[1] + dc]:
                continue
            if open_doors:
                joined = _flood(~wall, open_doors[0][0])
                if not (joined[p[0] - dr, p[1] - dc] or joined[p[0] + dr, p[1] + dc]):
                    continue
            wall[p] = False
            open_doors.append(door)
            break
        else:
            return None
    regions = _components(~wall)
    # Put islands in independently placed compartments. The presence/size of an
    # island does not encode which compartment the marker will select.
    for region in regions:
        _island(rng, wall, region)
        if rng.random() < .3:
            for component in _components(~wall):
                if np.any(component & region):
                    _island(rng, wall, component)
                    break
    regions = _components(~wall)
    if len(regions) != room_count - len(open_doors):
        return None
    first_open = open_doors[int(rng.integers(len(open_doors)))]
    open_region = next(m for m in regions if m[first_open[0]])
    closed = []
    for door in _doors(wall):
        p, (dr, dc) = door
        a, b = (p[0] - dr, p[1] - dc), (p[0] + dr, p[1] + dc)
        if open_region[a] != open_region[b]:
            closed.append(door)
    if not closed:
        return None
    rng.shuffle(closed)
    slots, used = [], set()
    blocked = np.zeros(wall.shape, bool)
    for door in (first_open, closed[0]):
        mask = _patch(rng, wall, door, blocked, used)
        if mask is None:
            return None
        slots.append(mask)
        used.add(_shape_key(mask))
        blocked |= mask | _adj(mask)
    # Additional repairs combine with the gates, but never become required
    # because of their index. Some tests retain an already visible open door.
    wanted = int(rng.integers(2, 5 if k == 2 else 4))
    options = list(_doors(wall)) + open_doors
    rng.shuffle(options)
    for door in options:
        if len(slots) >= wanted:
            break
        mask = _patch(rng, wall, door, blocked, used)
        if mask is not None:
            slots.append(mask)
            used.add(_shape_key(mask))
            blocked |= mask | _adj(mask)
    chosen = open_region
    if not active:
        p, (dr, dc) = closed[0]
        other = (p[0] + dr, p[1] + dc) if open_region[p[0] - dr, p[1] - dc] else (p[0] - dr, p[1] - dc)
        chosen = next(m for m in regions if m[other])
    occupied = np.logical_or.reduce(slots)
    candidates = chosen & ~occupied
    rim, _ = _rim(chosen)
    preferred = candidates & (~rim if active else rim)
    locations = np.argwhere(preferred if preferred.any() else candidates)
    if not len(locations):
        return None
    seed = tuple(map(int, locations[int(rng.integers(len(locations)))]))
    return wall, slots, seed


def _free_origins(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), int)
    s = np.pad(blocked.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = s[h:, w:] - s[:-h, w:] - s[h:, :-w] + s[:-h, :-w]
    return np.argwhere(totals == 0)


def _pack(rng, board, tiles, bg, k):
    objects = [board] + [_expand(tile, k) for tile in tiles]
    for attempt in range(36):
        h, w = ((int(rng.integers(22, 31)), int(rng.integers(22, 31)))
                if attempt < 20 else (30, 30))
        canvas = np.full((h, w), bg, np.int8)
        blocked = np.zeros((h, w), bool)
        order = [0] + sorted(range(1, len(objects)), key=lambda i: -objects[i].size)
        for i in order:
            obj = objects[i]
            oh, ow = obj.shape
            origins = _free_origins(blocked, oh, ow)
            if not len(origins):
                break
            r, c = map(int, origins[int(rng.integers(len(origins)))])
            view = canvas[r:r + oh, c:c + ow]
            if i == 0:
                view[:] = obj
            else:
                view[obj != -1] = obj[obj != -1]
            blocked[max(0, r - 1):min(h, r + oh + 1), max(0, c - 1):min(w, c + ow + 1)] = True
        else:
            if rng.random() < .6:
                r0, r1, c0, c1 = _bounds(canvas != bg)
                r0, c0 = max(0, r0 - int(rng.integers(4))), max(0, c0 - int(rng.integers(4)))
                r1, c1 = min(h, r1 + int(rng.integers(4))), min(w, c1 + int(rng.integers(4)))
                canvas = canvas[r0:r1, c0:c1].copy()
            if np.count_nonzero(canvas == bg) * 2 > canvas.size:
                return canvas
    return None


def _make_pair(rng, roles, k, complex_test, active, reflected, selected_size=None):
    bg, wall_color, patch, ink = roles
    for _ in range(240):
        scene = _scene(rng, k, complex_test, active)
        if scene is None:
            continue
        wall, slots, seed = scene
        final = np.where(wall, wall_color, bg).astype(np.int8)
        final[seed] = ink
        board = final.copy()
        for slot in slots:
            board[slot] = patch
        turn, flip = int(rng.integers(4)), bool(rng.integers(2))
        board, final = np.rot90(board, turn).copy(), np.rot90(final, turn).copy()
        if flip:
            board, final = np.fliplr(board).copy(), np.fliplr(final).copy()
        tiles = []
        for i, slot in enumerate(_components(board == patch)):
            box = _bounds(slot)
            support = _crop(slot, box)
            payload = np.full(support.shape, -1, np.int8)
            local = _crop(final, box)
            payload[support] = np.where(local[support] == wall_color, wall_color, patch)
            if reflected and i == 0:
                tile = np.rot90(np.fliplr(payload), int(rng.integers(4))).copy()
            else:
                choices = list(_d4(payload))
                tile = choices[int(rng.integers(len(choices)))]
            tiles.append(tile)
        inp = _pack(rng, board, tiles, bg, k)
        if inp is None:
            continue
        out, features = _paint(final, roles, k)
        if selected_size is not None and features["selection_class"] != selected_size:
            continue
        if not (features["island_evidence"] and features["erased"]):
            continue
        if active and not features["diagonal"]:
            continue
        recovered, info = _recover(inp)
        assert np.array_equal(recovered, final)
        if not info["closed_critical"] or (active and not info["open_critical"]):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded floodgate construction exhausted")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    mode = int(rng.integers(4))
    scales = ([2] * num_train, [3] * num_test) if mode == 0 else (
        ([3] * num_train, [2] * num_test) if mode == 1 else (
            ([2] * num_train, [2] * num_test) if mode == 2 else (
                [2, 3] + [int(rng.integers(2, 4)) for _ in range(num_train - 2)],
                [int(rng.integers(2, 4)) for _ in range(num_test)])))
    fixed_roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    stable_palette = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    first_shapes = None
    for split, schedule in zip(("train", "test"), scales):
        for i, k in enumerate(schedule):
            for _ in range(12):
                roles = fixed_roles if stable_palette else tuple(map(int, rng.choice(10, 4, replace=False)))
                active = i != 1 if split == "train" else True
                selected_size = ("large" if i == 0 else "small") if split == "train" and i < 2 else None
                pair = _make_pair(rng, roles, k, split == "test", active,
                                  split == "train" and i == 0, selected_size)
                key = _key(pair["input"])
                info = _recover(pair["input"])[1]
                if key in seen or (split == "train" and i == 1 and info["slot_shapes"] == first_shapes):
                    continue
                if split == "train" and i == 0:
                    first_shapes = info["slot_shapes"]
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
        rng.shuffle(puzzle[split])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            final, info = _recover(inp)
            expected, features = _paint(final, info["roles"], info["scale"])
            assert np.array_equal(expected, out), "incorrect repaired contour"
            bg, wall, patch, ink = info["roles"]
            assert patch not in np.unique(out)
            assert set(map(int, np.unique(out))) == {bg, wall, ink}
            assert features["kept_islands"] and features["island_evidence"]
            assert features["erased"]
            assert info["closed_critical"]
            assert np.count_nonzero(final == ink) == 1
            # Retained wall cells remain whole macrocells. Contouring is the
            # only operation whose thickness is defined in physical pixels.
            k = info["scale"]
            blocks = (out == wall).reshape(final.shape[0], k, final.shape[1], k)
            blocks = blocks.transpose(0, 2, 1, 3)
            retained = blocks[:, :, 0, 0]
            assert np.all(blocks == retained[:, :, None, None])
            assert np.all(~retained | (final == wall))
            if split == "train":
                train.append((info, features))
    assert any(i["open_critical"] and i["closed_critical"] for i, _ in train)
    assert any(f["diagonal"] for _, f in train)
    assert {"small", "large"} <= {f["selection_class"] for _, f in train}
    assert any(i["reflection"] for i, _ in train)
    assert len({i["slot_shapes"] for i, _ in train}) >= 2
    # Both larger connected selections and separate excluded compartments are
    # taught even with two demonstrations. The seed is not a fixed-corner cue.
    assert len({i["board_pattern"] for i, _ in train}) >= 2


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded streaming diagnostics: 42 puzzles; no sample/trace files."""
    digests, shapes, boards, vocab, scales, pieces, selections = set(), set(), set(), set(), set(), set(), set()
    reflected = rotated = open_critical = closed_critical = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                final, info = _recover(inp)
                expected, features = _paint(final, info["roles"], info["scale"])
                assert np.array_equal(expected, pair["output"])
                assert np.array_equal(solve(inp.copy()), pair["output"])
                shapes.add(inp.shape)
                boards.add(final.shape)
                vocab.update(info["slot_shapes"])
                scales.add(info["scale"])
                pieces.add(info["piece_count"])
                selections.add(features["selected_area"])
                reflected += bool(info["reflection"])
                rotated += bool(info["rotation"])
                open_critical += bool(info["open_critical"])
                closed_critical += bool(info["closed_critical"])
    assert len(digests) == 42 and len(shapes) >= 25 and len(boards) >= 15
    assert len(vocab) >= 25 and scales == {2, 3} and len(pieces) >= 2
    assert len(selections) >= 20 and min(reflected, rotated, open_critical, closed_critical) > 0
    p = generate_puzzle(818)
    assert _puzzle_digest(p) == _puzzle_digest(generate_puzzle(818))
    # Input-only covariance under every rigid square symmetry.
    pair = p["train"][0]
    before = pair["input"].copy()
    for flip in (False, True):
        for turn in range(4):
            inp, out = np.rot90(before, turn), np.rot90(pair["output"], turn)
            if flip:
                inp, out = np.fliplr(inp), np.fliplr(out)
            assert np.array_equal(solve(inp), out)
    assert np.array_equal(before, pair["input"])
    # Move the marker within and across compartments without changing repairs.
    final, info = _recover(before)
    bg, wall, patch, ink = info["roles"]
    sr, sc = tuple(map(int, np.argwhere(final == ink)[0]))
    selected = _flood(final != wall, (sr, sc))
    r0, r1, c0, c1 = info["bounds"]
    moved_outputs = []
    for same in (True, False):
        options = (final == bg) & (selected if same else ~selected) & (info["board"] == bg)
        assert options.any()
        r, c = map(int, np.argwhere(options)[0])
        changed = before.copy()
        changed[r0 + sr, c0 + sc], changed[r0 + r, c0 + c] = bg, ink
        changed_final, _ = _recover(changed)
        changed_selection = _flood(changed_final != wall, (r, c))
        assert np.array_equal(changed_selection, selected) == same
        moved_outputs.append(solve(changed))
    assert not np.array_equal(*moved_outputs)
    # A physical one-pixel contour differs from outlining logical cells and
    # enlarging that line into a k-pixel ribbon.
    out = solve(before)
    k = info["scale"]
    assert not np.array_equal(out, _expand(_paint(final, info["roles"], 1)[0], k))
    # Changing the contents of a tile, with its footprint and the blueprint
    # unchanged, changes both reachability and the output. This is the coupling
    # between fitting and flood selection, not an appended decorative contour.
    selected_pair = next(pair for pair in p["train"] if _recover(pair["input"])[1]["open_critical"])
    repaired, details = _recover(selected_pair["input"])
    bg, wall, patch, ink = details["roles"]
    original_seed = tuple(map(int, np.argwhere(repaired == ink)[0]))
    original_selected = _flood(repaired != wall, original_seed)
    for r, c in np.argwhere((details["board"] == patch) & (repaired == bg)):
        trial = _toggle_payload_at(selected_pair["input"], (int(r), int(c)))
        recovered, _ = _recover(trial)
        if not np.array_equal(_flood(recovered != wall, original_seed), original_selected):
            assert np.array_equal(_crop(trial, details["bounds"]), details["board"])
            assert not np.array_equal(solve(trial), selected_pair["output"])
            break
    else:
        raise AssertionError("no tile-content reachability witness")
    # Reject output corruption and partial macrocells.
    corrupted = {s: [{r: a.copy() for r, a in pair.items()} for pair in p[s]] for s in p}
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    damaged = selected_pair["input"].copy()
    mask = damaged != bg
    r0, r1, c0, c1 = details["bounds"]
    mask[r0:r1, c0:c1] = False
    r, c = map(int, np.argwhere(mask)[0])
    damaged[r, c] = bg
    try:
        solve(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("broken tile macrocell accepted")
    # A symmetric footprint with a non-invariant pattern has several valid
    # orientations. Reject that ambiguity; accept the same footprint when its
    # payload is invariant. This exercises broader input-only solver support.
    board = np.zeros((9, 9), np.int8)
    board[[0, -1], :] = 5
    board[:, [0, -1]] = 5
    board[6, 2] = 1
    board[2:4, 2:4] = 8
    for pos in ((2, 5), (3, 5), (4, 5), (4, 6)):
        board[pos] = 8
    square = np.array([[8, 8], [8, 5]], np.int8)
    elbow = np.array([[5, -1], [5, -1], [5, 5]], np.int8)
    ambiguous = _pack(np.random.default_rng(11), board, [square, elbow], 0, 2)
    assert ambiguous is not None
    try:
        solve(ambiguous)
    except AssertionError as error:
        assert "ambiguous" in str(error)
    else:
        raise AssertionError("ambiguous patterned orientation accepted")
    uniform = _pack(np.random.default_rng(11), board, [np.full((2, 2), 5, np.int8), elbow], 0, 2)
    assert uniform is not None and solve(uniform).shape == (18, 18)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds; streaming, no files)")
