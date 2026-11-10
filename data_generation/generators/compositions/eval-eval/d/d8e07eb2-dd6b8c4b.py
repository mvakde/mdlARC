"""Exact glyph keys open pores; shortest routes collect up to nine cells.

All solving conventions are fixed or visible in the input. No parent code,
example labels, file reads, or random state participate in solve().
"""
from collections import deque
import hashlib

import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_MASKS = tuple(np.array([(code >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
               for code in range(1, 511))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid must have integer cells")
    _require(bool(np.all((0 <= a) & (a <= 9))), "Invalid colors")
    return a.astype(np.int8, copy=True)


def _key(patch):
    return tuple(int(v) for v in patch.ravel())


def _digest(a):
    return hashlib.sha256(bytes(a.shape) + a.astype(np.uint8).tobytes()).hexdigest()


def _counter(grid):
    candidates = []
    for color in np.unique(grid):
        rr, cc = np.where(grid == color)
        if len(rr) != 8:
            continue
        r, c = int(rr.min()), int(cc.min())
        if int(rr.max()) != r + 2 or int(cc.max()) != c + 2:
            continue
        block = grid[r:r+3, c:c+3]
        center = int(block[1, 1])
        rim = np.ones((3, 3), bool)
        rim[1, 1] = False
        if np.all(block[rim] == color) and center != color and np.count_nonzero(grid == center) == 1:
            candidates.append((r, c, int(color), center))
    _require(len(candidates) == 1, "Expected one marked collector")
    return candidates[0]


def _parse_canonical(a, allow_empty=False):
    h, w = a.shape
    _require(20 <= h <= 30 and 18 <= w <= 30, "Outside family canvas bounds")
    pane, wall, floor = int(a[0, 0]), int(a[4, 0]), int(a[5, 0])
    _require(len({pane, wall, floor}) == 3, "Legend roles must differ")
    _require(np.all(a[0] == pane) and np.all(a[4] == wall) and np.all(a[5] == floor),
             "Expected key rail, divider and floor gutter")
    _require(np.all(a[6:, 0] == floor), "Expected outside gutter")
    _require(np.all(a[6, 1:] == wall) and np.all(a[-1, 1:] == wall)
             and np.all(a[6:, 1] == wall) and np.all(a[6:, -1] == wall), "Field boundary must be closed")
    r, c, slot, marker = _counter(a)
    _require(7 <= r and r+2 < h-1 and 2 <= c and c+2 < w-1, "Collector outside field")
    _require(len({pane, wall, floor, slot, marker}) == 5, "Structural roles must differ")
    queries, header = [], np.full((4, w), pane, np.int8)
    for start in range(1, w-2, 4):
        patch = a[1:4, start:start+3]
        inks = set(map(int, np.unique(patch))) - {pane}
        if inks:
            _require(len(inks) == 1 and pane in patch and not inks & {wall, floor, slot, marker},
                     "Invalid key glyph")
            queries.append(_key(patch))
        header[1:4, start:start+3] = patch
    _require(np.array_equal(header, a[:4]), "Ink outside key positions")
    _require(1 <= len(queries) <= 5 and len(set(queries)) == len(queries), "Invalid key set")
    excluded = {wall, floor, slot, marker}
    doors, covered = [], np.zeros(a.shape, bool)
    # Every candidate contains a pane pixel. Derive candidate origins from those
    # pixels instead of inspecting every empty square of a large field.
    origins = sorted({(int(rr)-dr, int(cc)-dc) for rr, cc in np.argwhere(a[6:] == pane)
                      for dr in range(3) for dc in range(3)
                      if 7 <= int(rr)+6-dr < h-3 and 2 <= int(cc)-dc < w-3})
    origins = [(rr+6, cc) for rr, cc in origins]
    for rr, cc in origins:
        patch = a[rr:rr+3, cc:cc+3]
        values = set(map(int, np.unique(patch)))
        if pane not in values or values & excluded or len(values) != 2:
            continue
        _require(not covered[rr:rr+3, cc:cc+3].any(), "Overlapping door windows")
        covered[rr:rr+3, cc:cc+3] = True
        doors.append((rr, cc, patch.copy(), _key(patch) in queries))
    _require(2 <= len(doors) <= 12, "Expected 2..12 door glyphs")
    door_keys = [_key(patch) for _, _, patch, _ in doors]
    _require(len(set(door_keys)) == len(door_keys), "Door addresses must be unique")
    _require(set(queries) <= set(door_keys), "Every key needs an exact door copy")
    _require(any(not selected for _, _, _, selected in doors), "Expected an unqueried door")
    _require(np.all(covered[6:][a[6:] == pane]), "Pane outside a door")
    rest = a[6:][~covered[6:]]
    leftovers = set(map(int, np.unique(rest))) - {wall, floor, slot, marker}
    _require(len(leftovers) <= 1 and (allow_empty or len(leftovers) == 1), "Expected one payload color")
    payload = next(iter(leftovers)) if leftovers else None
    inks = {int(patch[patch != pane][0]) for _, _, patch, _ in doors}
    _require(not inks & {pane, wall, floor, slot, marker, payload}, "Role collision")
    _require(2 <= len(inks) <= 4, "Expected 2..4 glyph inks")
    return dict(grid=a, pane=pane, wall=wall, floor=floor, slot=slot, marker=marker,
                counter=(r, c), payload=payload, doors=doors, queries=queries, inks=inks)


def _analyze(input_grid, allow_empty=False):
    a = _grid(input_grid)
    candidates = []
    for k in range(4):
        try:
            info = _parse_canonical(np.rot90(a, k).copy(), allow_empty)
        except AssertionError:
            continue
        info["rotation"] = k
        candidates.append(info)
    _require(len(candidates) == 1, "Key rail and field must have a unique orientation")
    return candidates[0]


def _passable(info, variant="normal"):
    a = info["grid"]
    traversable = a == info["floor"]
    if info["payload"] is not None:
        traversable |= a == info["payload"]
    r, c = info["counter"]
    traversable[r:r+3, c:c+3] = True
    for rr, cc, patch, selected in info["doors"]:
        if variant == "no_keys":
            selected = False
        elif variant == "all_keys":
            selected = True
        if selected:
            traversable[rr:rr+3, cc:cc+3] = (np.ones((3, 3), bool) if variant == "solid_open"
                                            else patch == info["pane"])
    traversable[:6] = False
    traversable[:, 0] = False
    return traversable


def _distances(info, variant="normal"):
    allowed = _passable(info, variant)
    distance = np.full(allowed.shape, -1, np.int16)
    r, c = info["counter"]
    queue = deque((r+i, c+j) for i in range(3) for j in range(3))
    for p in queue:
        distance[p] = 0
    h, w = allowed.shape
    while queue:
        rr, cc = queue.popleft()
        for dr, dc in _STEPS:
            nr, nc = rr+dr, cc+dc
            if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and distance[nr, nc] < 0:
                distance[nr, nc] = distance[rr, cc] + 1
                queue.append((nr, nc))
    return distance


def _selected(info, distance, strict=True):
    a = info["grid"]
    supply = (a == info["payload"]) & (distance >= 0) if info["payload"] is not None else np.zeros(a.shape, bool)
    cells = sorted(map(int, np.flatnonzero(supply)), key=lambda i: (int(distance.flat[i]), i))
    if strict and len(cells) > 9:
        _require(distance.flat[cells[8]] < distance.flat[cells[9]], "Capacity frontier tie")
    result = np.zeros(a.shape, bool)
    result.flat[cells[:9]] = True
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Open exact-key pores, collect nearest reachable cells, serialize in display order."""
    info = _analyze(input_grid)
    a = info["grid"]
    result = a.copy()
    for r, c, patch, selected in info["doors"]:
        if selected:
            block = result[r:r+3, c:c+3]
            block[patch == info["pane"]] = info["floor"]
    chosen = _selected(info, _distances(info))
    result[chosen] = info["floor"]
    result = np.rot90(result, -info["rotation"]).copy()
    # Do not rotate an already filled prefix: serialization is in the displayed grid.
    r, c, _, _ = _counter(_grid(input_grid))
    for i in range(int(chosen.sum())):
        result[r+i//3, c+i%3] = info["payload"]
    return result


def _crosses(mask, axis):
    """Does a pore path join opposite edges? axis=0: left/right; axis=1: top/bottom."""
    pores = ~mask
    starts = [(r, 0) for r in range(3) if pores[r, 0]] if axis == 0 else [(0, c) for c in range(3) if pores[0, c]]
    seen, todo = set(starts), list(starts)
    for r, c in todo:
        for dr, dc in _STEPS:
            p = r+dr, c+dc
            if 0 <= p[0] < 3 and 0 <= p[1] < 3 and pores[p] and p not in seen:
                seen.add(p)
                todo.append(p)
    return any(c == 2 if axis == 0 else r == 2 for r, c in seen)


def _mask(rng, axis, crossing):
    bank = _MASK_BANKS[axis, bool(crossing)]
    return bank[int(rng.integers(len(bank)))].copy()


_MASK_BANKS = {(axis, crossing): tuple(m for m in _MASKS if _crosses(m, axis) == crossing)
               for axis in (0, 1) for crossing in (False, True)}


def _starts(rng, length, count):
    slack = length - 2 - 3*count - (count-1)
    if slack < 0:
        return []
    gaps = rng.multinomial(slack, np.full(count+1, 1/(count+1)))
    positions, start = [], 1+int(gaps[0])
    for i in range(count):
        positions.append(start)
        start += 4 + int(gaps[i+1])
    return positions


def _partition_geometry(rng, shape, lesson):
    """Construct cuts/windows, not a bank of finished mazes."""
    h, w = shape
    wall = np.zeros((h, w), bool)
    wall[[0, -1]] = True
    wall[:, [0, -1]] = True
    doors = []
    count = 4 if lesson else int(rng.integers(2, min(5, (h-1)//4)+1))
    rows = _starts(rng, h, count)
    b = int(rng.integers(5, w-7))
    wall[1:-1, b:b+3] = True
    doors.extend((r, b, 0) for r in rows)
    blocked = int(rng.choice([1, 2])) if lesson else None
    far = (0 if blocked == 2 else count-1) if lesson else None
    near = (2 if blocked == 1 else 1) if lesson else None
    if lesson:
        cr = max(1, min(h-4, rows[blocked]+int(rng.integers(-1, 2))))
        side = bool(rng.integers(2))
        cc = b-4 if side else b+4
        pos = (cr, cc)
    else:
        pos = None
    # Secondary partitions produce branches and series-connected locks.
    mode = "divider" if lesson else str(rng.choice(["divider", "parallel", "rooms", "cross"]))
    if mode == "parallel" and w >= 24:
        other = b+6 if b+9 < w-1 else b-6
        if other >= 1 and other+2 < w-1:
            wall[1:-1, other:other+3] = True
            for r in _starts(rng, h, int(rng.integers(2, 4))):
                doors.append((r, other, 0))
    elif mode in ("rooms", "cross"):
        # Horizontal cuts terminate at the main wall or extend across both rooms.
        y = int(rng.integers(4, h-6))
        sides = [(1, b)] if mode == "rooms" and rng.random() < .5 else [(b+3, w-1)] if mode == "rooms" else [(1, b), (b+3, w-1)]
        for lo, hi in sides:
            if hi-lo < 5:
                continue
            wall[y:y+3, lo:hi] = True
            x = int(rng.integers(lo+1, hi-3))
            doors.append((y, x, 1))
    return wall, doors, pos, dict(blocked=blocked, far=far, near=near, mode=mode)


def _ring_geometry(rng, shape):
    h, w = shape
    wall = np.zeros(shape, bool)
    wall[[0, -1]] = True
    wall[:, [0, -1]] = True
    r0, c0 = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    r1, c1 = h-int(rng.integers(3, 5)), w-int(rng.integers(3, 5))
    wall[r0:r0+3, c0:c1+1] = True
    wall[r1-2:r1+1, c0:c1+1] = True
    wall[r0:r1+1, c0:c0+3] = True
    wall[r0:r1+1, c1-2:c1+1] = True
    x = int(rng.integers(c0+3, c1-4))
    y = int(rng.integers(r0+3, r1-4))
    doors = [(r0, x, 1), (r1-2, x, 1), (y, c0, 0), (y, c1-2, 0)]
    return wall, doors, None, dict(mode="ring")


def _scene(rng, palette, lesson=False, test=False, fallback=False):
    floor, wall_color, pane, slot, marker, payload = palette[:6]
    inks = palette[6:]
    if fallback:
        h, w = 24, 24
    elif lesson:
        h, w = int(rng.integers(20, 25)), int(rng.integers(20, 30))
    else:
        h = int(rng.integers(18 if test else 14, 25))
        w = int(rng.integers(20 if test else 17, 30))
    if not lesson and h >= 20 and w >= 20 and rng.random() < (.35 if test else .2):
        wall, doors, pos, meta = _ring_geometry(rng, (h, w))
    else:
        wall, doors, pos, meta = _partition_geometry(rng, (h, w), lesson)
    # A small sealed pocket is optional except in the two evidence lessons.
    reserved = np.zeros((h, w), bool)
    for r, c, _ in doors:
        reserved[max(0, r-1):r+4, max(0, c-1):c+4] = True
    if pos is not None:
        r, c = pos
        reserved[max(0, r-1):r+4, max(0, c-1):c+4] = True
    if lesson or rng.random() < .45:
        candidates = [(r, c) for r in range(1, h-5) for c in range(1, w-5)
                      if not reserved[r:r+5, c:c+5].any() and not wall[r:r+5, c:c+5].any()]
        if candidates:
            r, c = candidates[int(rng.integers(len(candidates)))]
            wall[r, c:c+5] = wall[r+4, c:c+5] = True
            wall[r:r+5, c] = wall[r:r+5, c+4] = True
    # Variable short walls and turns are allowed, but never cross a door or collector.
    for _ in range(int(rng.integers(0, 4))):
        rr, cc = int(rng.integers(1, h-3)), int(rng.integers(1, w-3))
        sh, sw = (1, int(rng.integers(2, 6))) if rng.random() < .5 else (int(rng.integers(2, 6)), 1)
        if not reserved[rr:min(h-1, rr+sh), cc:min(w-1, cc+sw)].any():
            wall[rr:min(h-1, rr+sh), cc:min(w-1, cc+sw)] = True
    for r, c, _ in doors:
        wall[r:r+3, c:c+3] = False
    if pos is None:
        choices = [(r, c) for r in range(1, h-3) for c in range(1, w-3)
                   if not wall[r:r+3, c:c+3].any() and not reserved[r:r+3, c:c+3].any()]
        if not choices:
            return None
        pos = choices[int(rng.integers(len(choices)))]
    masks, ink_values, selected = [], [], set()
    if lesson:
        block, far, near = meta["blocked"], meta["far"], meta["near"]
        selected = {block, far}
        pass_mask = _mask(rng, 0, True)
        block_mask = _mask(rng, 0, False)
        for i, (_, _, axis) in enumerate(doors):
            if i == block:
                m, ink = block_mask, inks[0]
            elif i == far:
                m, ink = pass_mask, inks[0]
            elif i == near:
                m, ink = pass_mask, inks[1]
            else:
                m, ink = _mask(rng, axis, bool(rng.integers(2))), inks[0]
                while np.array_equal(m, pass_mask) or np.array_equal(m, block_mask):
                    m = _mask(rng, axis, bool(rng.integers(2)))
            masks.append(m.copy())
            ink_values.append(ink)
    else:
        q = int(rng.integers(1, min(5, len(doors)-1, w//4)+1))
        selected = set(map(int, rng.choice(len(doors), q, replace=False)))
        seen = set()
        for i, (_, _, axis) in enumerate(doors):
            for _ in range(100):
                m = _mask(rng, axis, rng.random() < (.75 if i in selected else .5))
                ink = int(rng.choice(inks))
                key = (m.tobytes(), ink)
                if key not in seen:
                    seen.add(key)
                    masks.append(m)
                    ink_values.append(ink)
                    break
            else:
                return None
        if len(set(ink_values)) == 1:
            ink_values[-1] = inks[1] if ink_values[-1] == inks[0] else inks[0]
    # Internal transpose changes barrier directions independently of the key rail.
    transpose = bool(rng.integers(2)) and w <= 24 and h >= 17 and len(selected) <= h//4
    field = np.full((h, w), floor, np.int8)
    field[wall] = wall_color
    patches = []
    for (r, c, _), m, ink in zip(doors, masks, ink_values):
        patch = np.full((3, 3), pane, np.int8)
        patch[m] = ink
        field[r:r+3, c:c+3] = patch
        patches.append(patch)
    r, c = pos
    field[r:r+3, c:c+3] = slot
    field[r+1, c+1] = marker
    if transpose:
        field = field.T.copy()
        patches = [p.T.copy() for p in patches]
    h, w = field.shape
    a = np.full((h+6, w+1), floor, np.int8)
    a[:4] = pane
    a[4] = wall_color
    a[6:, 1:] = field
    slots = list(range(1, a.shape[1]-2, 4))
    query_indices = sorted(selected)
    rng.shuffle(query_indices)
    occupied = rng.choice(len(slots), len(query_indices), replace=False)
    for index, place in zip(query_indices, occupied):
        start = slots[int(place)]
        a[1:4, start:start+3] = patches[index]
    try:
        info = _analyze(a, allow_empty=True)
    except AssertionError:
        return None
    return a, info


def _choose_supply(rng, info, load, evidence):
    a = info["grid"]
    distance = _distances(info)
    available = a == info["floor"]
    available[:6] = False
    available[:, 0] = False
    active, inactive = available & (distance >= 0), available & (distance < 0)
    if active.sum() < load or (evidence and not inactive.any()):
        return None
    r, c = info["counter"]
    rr, cc = np.indices(a.shape)
    geometric = np.maximum(r-rr, 0) + np.maximum(rr-r-2, 0) + np.maximum(c-cc, 0) + np.maximum(cc-c-2, 0)
    alt_all, alt_solid = _distances(info, "all_keys"), _distances(info, "solid_open")
    choices = np.flatnonzero(active)
    if load <= 9:
        no_key = _distances(info, "no_keys")
        across = np.flatnonzero(active & (no_key < 0))
        if evidence and not len(across):
            return None
        forced = [int(rng.choice(across))] if load > 0 and len(across) else []
        pool = [int(x) for x in choices if int(x) not in forced]
        supply = forced + list(map(int, rng.choice(pool, load-len(forced), replace=False)))
    else:
        supply = None
        for i in rng.permutation(choices):
            i = int(i)
            near = active & (distance <= distance.flat[i])
            far = active & (distance > distance.flat[i])
            if near.sum() < 9 or far.sum() < load-9:
                continue
            b_mask = far.copy()
            if evidence:
                b_mask &= (geometric < geometric.flat[i]) & (alt_all >= 0) & (alt_all < alt_all.flat[i])
                b_mask &= (alt_solid >= 0) & (alt_solid < alt_solid.flat[i])
            if not b_mask.any():
                continue
            j = int(rng.choice(np.flatnonzero(b_mask)))
            near_pool = [int(x) for x in np.flatnonzero(near) if x != i]
            far_pool = [int(x) for x in np.flatnonzero(far) if x != j]
            supply = [i] + list(map(int, rng.choice(near_pool, 8, replace=False)))
            supply += [j] + list(map(int, rng.choice(far_pool, load-10, replace=False)))
            break
        if supply is None:
            return None
    distract_count = min(int(inactive.sum()), int(rng.integers(1 if evidence or load == 0 else 0, 8)))
    if load == 0 and distract_count == 0:
        return None
    distractors = list(map(int, rng.choice(np.flatnonzero(inactive), distract_count, replace=False)))
    return supply+distractors


def pair_features(input_grid):
    """Measurements of actual pixels; no construction metadata is used."""
    info = _analyze(input_grid)
    distance = _distances(info)
    chosen = _selected(info, distance)
    a, payload = info["grid"], info["payload"]
    mask = a == payload
    retained = mask & (distance >= 0) & ~chosen
    r, c = info["counter"]
    rr, cc = np.indices(a.shape)
    direct = np.maximum(r-rr, 0)+np.maximum(rr-r-2, 0)+np.maximum(c-cc, 0)+np.maximum(cc-c-2, 0)
    shapes = [tuple((p != info["pane"]).ravel()) for _, _, p, _ in info["doors"]]
    selected_doors = [i for i, d in enumerate(info["doors"]) if d[3]]
    outside_doors = [i for i, d in enumerate(info["doors"]) if not d[3]]
    inks = [int(p[p != info["pane"]][0]) for _, _, p, _ in info["doors"]]
    def changes(variant):
        return not np.array_equal(chosen, _selected(info, _distances(info, variant), strict=False))
    return dict(shape=list(_grid(input_grid).shape), rotation=info["rotation"],
                palette=[info[k] for k in ("floor", "wall", "pane", "slot", "marker", "payload")],
                doors=len(shapes), keys=len(selected_doors), ink_count=len(set(inks)),
                selected=int(chosen.sum()), reachable=int((mask & (distance >= 0)).sum()),
                unreachable=int((mask & (distance < 0)).sum()), excess=int(retained.sum()),
                distance_gap=int(distance[retained].min()-distance[chosen].max()) if retained.any() else 0,
                max_path=int(distance.max()), wall_cells=int((a[6:] == info["wall"]).sum()),
                path_reversal=bool(retained.any() and direct[chosen].max() > direct[retained].min()),
                color_sensitive=any(shapes[i] == shapes[j] and inks[i] != inks[j] for i in selected_doors for j in outside_doors),
                shape_sensitive=any(inks[i] == inks[j] and shapes[i] != shapes[j] for i in selected_doors for j in outside_doors),
                no_keys_changes_collection=changes("no_keys"), all_keys_changes_collection=changes("all_keys"),
                ink_obstacles_change_collection=changes("solid_open"))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not _integer(num_test) or not 2 <= int(num_train) <= 10 or not 1 <= int(num_test) <= 4 or num_train <= num_test:
        raise ValueError("Require 2..10 training, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    ink_count = int(rng.integers(2, 5))
    palette = list(map(int, rng.permutation(10)[:6+ink_count]))
    vary_palette = bool(rng.random() < .5)
    orientation = int(rng.integers(4))
    mixed_orientation = int(num_train) >= 3 and rng.random() < .5
    partial = int(rng.choice([2, 4]))
    train_specs = [(partial, "partial"), (int(rng.integers(10, 20)), "excess")]
    for _ in range(int(num_train)-2):
        load = int(rng.integers(0, 10)) if rng.random() < .75 else int(rng.integers(10, 24))
        train_specs.append((load, "ordinary"))
    rng.shuffle(train_specs)
    # New room/lock configurations and larger fields are weighted in testing;
    # load and output orientation have no test-index convention.
    test_specs = [(int(rng.integers(0, 10)) if rng.random() < .65 else int(rng.integers(10, 24)), "ordinary")
                  for _ in range(int(num_test))]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, specs in (("train", train_specs), ("test", test_specs)):
        for load, lesson in specs:
            accepted = False
            for attempt in range(160):
                colors = list(map(int, rng.permutation(10)[:6+ink_count])) if vary_palette else palette
                scene = _scene(rng, colors, lesson != "ordinary", split == "test", fallback=attempt >= 120)
                if scene is None:
                    continue
                a, info = scene
                cells = _choose_supply(rng, info, load, lesson != "ordinary")
                if cells is None:
                    continue
                a.flat[cells] = colors[5]
                k = int(rng.integers(4)) if mixed_orientation else orientation
                a = np.rot90(a, k).copy()
                try:
                    features = pair_features(a)
                except AssertionError:
                    continue
                if lesson == "partial" and not features["no_keys_changes_collection"]:
                    continue
                if lesson == "excess" and not (features["path_reversal"] and features["all_keys_changes_collection"]
                                                and features["ink_obstacles_change_collection"]):
                    continue
                signature = _digest(a)
                if signature in seen:
                    continue
                seen.add(signature)
                puzzle[split].append({"input": a, "output": solve(a)})
                accepted = True
                break
            if not accepted:
                raise AssertionError("Bounded construction failed: " + lesson)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "Invalid pair counts")
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "Dimensions must be preserved")
            key = _digest(a)
            _require(key not in seen, "Duplicate input")
            seen.add(key)
            info = _analyze(a)
            d = _distances(info)
            chosen = _selected(info, d)
            _require(np.array_equal(b, solve(a)), "Incorrect porous-lock/collection output")
            canonical_out = np.rot90(b, info["rotation"])
            _require(np.array_equal(canonical_out[:6], info["grid"][:6]), "Key rail changed")
            # Ink, walls, unreachable and reachable excess are preserved.
            for r, c, patch, _ in info["doors"]:
                ink = patch != info["pane"]
                _require(np.array_equal(canonical_out[r:r+3, c:c+3][ink], patch[ink]), "Door ink changed")
            supply = info["grid"] == info["payload"]
            _require(np.array_equal(canonical_out[supply & ~chosen], info["grid"][supply & ~chosen]), "Uncollected payload changed")
            _require(np.all(canonical_out[chosen] == info["floor"]), "Collected payload not erased")
            r, c, _, _ = _counter(a)
            prefix = b[r:r+3, c:c+3].ravel()
            original = a[r:r+3, c:c+3].ravel()
            n = int(chosen.sum())
            _require(np.all(prefix[:n] == info["payload"]) and np.array_equal(prefix[n:], original[n:]), "Wrong displayed prefix")
            features = pair_features(a)
            if split == "train":
                train.append(features)
    _require(any(f["selected"] in (2, 4) for f in train), "Missing partial-prefix evidence")
    _require(any(f["excess"] and f["path_reversal"] for f in train), "Missing path/capacity evidence")
    _require(any(f["unreachable"] for f in train), "Missing unreachable distraction")
    for feature in ("color_sensitive", "shape_sensitive", "no_keys_changes_collection", "all_keys_changes_collection", "ink_obstacles_change_collection"):
        _require(any(f[feature] for f in train), "Missing training witness: " + feature)


def self_test():
    """Bounded 42-seed semantic, diversity, determinism and mutation checks."""
    records, signatures, normalized = [], set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        q = generate_puzzle(seed)
        for split in p:
            for a, b in zip(p[split], q[split]):
                _require(all(np.array_equal(a[k], b[k]) for k in a), "Nondeterminism")
                records.append(pair_features(a["input"]))
                roles = {}
                normalized.add(tuple(tuple(roles.setdefault(int(v), len(roles)) for v in row) for row in a["input"]))
        signatures.add(tuple(_digest(pair["input"]) for split in p for pair in p[split]))
    _require(len(signatures) == 42 and len(normalized) >= 180, "Structural variation collapsed")
    for feature, minimum in (("doors", 4), ("keys", 4), ("selected", 8), ("reachable", 15),
                             ("wall_cells", 30), ("max_path", 20), ("unreachable", 5)):
        _require(len({f[feature] for f in records}) >= minimum, "Insufficient variation: " + feature)
    _require({f["rotation"] for f in records} == {0, 1, 2, 3}, "Missing rail orientations")
    _require(len({tuple(f["shape"]) for f in records}) >= 40, "Missing canvas variation")
    _require(len({tuple(f["palette"]) for f in records}) >= 30, "Missing palette variation")
    # Concrete pore topology checks, independent of labels.
    stripe = np.zeros((3, 3), bool)
    stripe[:, 1] = True
    _require(not _crosses(stripe, 0) and _crosses(stripe, 1), "Stripe pore topology")
    _require(_crosses(stripe.T, 0) and not _crosses(stripe.T, 1), "Transposed pore topology")
    # Deliberate strict-frontier ambiguity must be rejected.
    p = generate_puzzle(71, 2, 1)
    info = _analyze(p["train"][0]["input"])
    info["grid"] = info["grid"].copy()
    d = _distances(info)
    floor = info["grid"] == info["floor"]
    levels = [(int(k), np.flatnonzero(floor & (d == k))) for k in np.unique(d[floor]) if k > 0]
    earlier = []
    tested_tie = False
    for k, cells in levels:
        if len(earlier) >= 8 and len(cells) >= 2:
            info["grid"][info["grid"] == info["payload"]] = info["floor"]
            info["grid"].flat[earlier[:8] + list(map(int, cells[:2]))] = info["payload"]
            try:
                _selected(info, d)
            except AssertionError:
                tested_tie = True
            break
        earlier.extend(map(int, cells))
    _require(tested_tie, "Tie rejection not exercised")
    # Reject damaged labels and duplicates rather than merely validating schema.
    for defect in ("door_ink", "prefix", "duplicate"):
        broken = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        if defect == "duplicate":
            broken["train"][1] = {k: v.copy() for k, v in broken["train"][0].items()}
        else:
            pair = broken["test"][0]
            if defect == "prefix":
                r, c, _, _ = _counter(pair["input"])
                pair["output"][r, c] = (int(pair["output"][r, c])+1) % 10
            else:
                i = _analyze(pair["input"])
                r, c, patch, _ = i["doors"][0]
                rr, cc = np.argwhere(patch != i["pane"])[0]
                canonical = np.rot90(pair["output"], i["rotation"])
                canonical[r+rr, c+cc] = i["floor"]
            try:
                validate_generated_puzzle(broken)
            except AssertionError:
                continue
            raise AssertionError("Damaged label accepted: " + defect)
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            continue
        raise AssertionError("Duplicate accepted")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            generate_puzzle(1000+nt*10+nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    return {"seeds": 42, "pairs": len(records), "distinct_puzzles": len(signatures),
            "normalized_inputs": len(normalized), "grid_shapes": len({tuple(f["shape"]) for f in records})}
