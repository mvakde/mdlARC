"""Sealed rooms stretch their 2x2 seed.

Composition of 762cd429 (nearest-neighbor enlargement of a 2x2 template) and
a5313dff (four-neighbor enclosure detection behind one-cell barrier walls).

Rule: the background is the most common color and the wall color is the most
common remaining color.  Every non-wall region that cannot reach the grid edge
through four-neighbor steps over non-wall cells is a sealed room.  Each sealed
room holds exactly one 2x2 seed.  The seed is enlarged by nearest-neighbor
scaling to the room's bounding box (independent vertical and horizontal
factors, so each seed cell becomes an (h/2)x(w/2) block) and painted onto every
non-wall cell of the room.  Wall cells (including marks inside a room), open
(leaky) rooms with their seeds, stray seeds and the exterior stay unchanged.
"""

from __future__ import annotations

from collections import deque

import numpy as np

MODES = ("scale_up", "crowded", "marks", "palette")
SEED_PATTERNS = ("distinct", "three", "row", "column", "diagonal", "three_one", "single")


# ----------------------------------------------------------------------------
# Shared analysis (used by solve and by validation)
# ----------------------------------------------------------------------------

def _roles(grid):
    values, counts = np.unique(grid, return_counts=True)
    order = sorted(zip(counts.tolist(), values.tolist()), reverse=True)
    background = order[0][1]
    wall = order[1][1] if len(order) > 1 else None
    return background, wall, order


def _exterior(passable):
    """Cells of `passable` reachable from the border by four-neighbor steps."""
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    queue = deque()
    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                queue.append((rr, cc))
    return seen


def _components(mask):
    h, w = mask.shape
    label = -np.ones(mask.shape, dtype=int)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and label[r, c] < 0:
                cells = []
                label[r, c] = len(comps)
                queue = deque([(r, c)])
                while queue:
                    y, x = queue.popleft()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and label[yy, xx] < 0:
                            label[yy, xx] = len(comps)
                            queue.append((yy, xx))
                comps.append(cells)
    return comps


def _sealed_rooms(grid):
    """Return (background, wall, rooms); each room is a dict with cells/bbox/seed."""
    grid = np.asarray(grid)
    background, wall, _ = _roles(grid)
    if wall is None:
        return background, wall, []
    passable = grid != wall
    outside = _exterior(passable)
    rooms = []
    for cells in _components(passable & ~outside):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        seed_cells = [(r, c) for r, c in cells if grid[r, c] != background]
        seed = None
        if len(seed_cells) == 4:
            sr = sorted({r for r, _ in seed_cells})
            sc = sorted({c for _, c in seed_cells})
            if len(sr) == 2 and len(sc) == 2 and sr[1] == sr[0] + 1 and sc[1] == sc[0] + 1:
                seed = (sr[0], sc[0], grid[sr[0]:sr[0] + 2, sc[0]:sc[0] + 2].copy())
        rooms.append({"cells": cells, "bbox": (r0, r1, c0, c1), "seed": seed})
    return background, wall, rooms


def _stretch(seed, height, width):
    rows = (np.arange(height) * 2) // height
    cols = (np.arange(width) * 2) // width
    return seed[rows][:, cols]


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.array(input_grid, dtype=int)
    out = grid.copy()
    _, _, rooms = _sealed_rooms(grid)
    for room in rooms:
        if room["seed"] is None:
            continue
        r0, r1, c0, c1 = room["bbox"]
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h % 2 or w % 2:
            continue
        big = _stretch(room["seed"][2], h, w)
        for r, c in room["cells"]:
            out[r, c] = big[r - r0, c - c0]
    return out


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must be in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _seed_block(rng, avoid, pattern):
    pool = np.array([c for c in range(10) if c not in avoid])
    count = {"distinct": 4, "three": 3, "single": 1}.get(pattern, 2)
    colors = rng.choice(pool, size=count, replace=False)
    labels = {
        "distinct": [0, 1, 2, 3], "three": [0, 1, 2, 2], "row": [0, 0, 1, 1],
        "column": [0, 1, 0, 1], "diagonal": [0, 1, 1, 0], "three_one": [0, 0, 0, 1],
        "single": [0, 0, 0, 0],
    }[pattern]
    if pattern in ("three", "three_one"):
        labels = list(rng.permutation(labels))
    return colors[np.array(labels)].reshape(2, 2).astype(int)


class _Room:
    def __init__(self, top, left, ih, iw, closed):
        self.top, self.left, self.ih, self.iw, self.closed = top, left, ih, iw, closed
        self.seed_pos = None
        self.seed = None
        self.marks = []
        self.gap = None

    @property
    def bottom(self):
        return self.top + self.ih + 1

    @property
    def right(self):
        return self.left + self.iw + 1

    def frame_cells(self):
        cells = set()
        for c in range(self.left, self.right + 1):
            cells.add((self.top, c))
            cells.add((self.bottom, c))
        for r in range(self.top, self.bottom + 1):
            cells.add((r, self.left))
            cells.add((r, self.right))
        return cells

    def interior_cells(self):
        return {(r, c) for r in range(self.top + 1, self.bottom) for c in range(self.left + 1, self.right)}


def _room_dims(rng, max_dim, need_big=0, need_nonsquare=False):
    dims = list(range(2, max_dim + 1, 2))
    for _ in range(100):
        ih, iw = int(rng.choice(dims)), int(rng.choice(dims))
        if ih == 2 and iw == 2:
            continue
        if need_big and max(ih, iw) < need_big:
            continue
        if need_nonsquare and ih == iw:
            continue
        return ih, iw
    return (2, max_dim) if rng.integers(2) else (max_dim, 2)


def _place_free(rng, rooms, h, w, ih, iw, closed):
    for _ in range(60):
        top = int(rng.integers(0, h - ih - 1))
        left = int(rng.integers(0, w - iw - 1))
        room = _Room(top, left, ih, iw, closed)
        ok = True
        for other in rooms:
            if not (room.bottom + 1 < other.top or other.bottom + 1 < room.top
                    or room.right + 1 < other.left or other.right + 1 < room.left):
                ok = False
                break
        if ok:
            return room
    return None


def _place_shared(rng, rooms, h, w, ih, iw, closed):
    anchors = [r for r in rooms]
    for _ in range(60):
        a = anchors[int(rng.integers(len(anchors)))]
        side = int(rng.integers(4))
        if side in (0, 1):  # right / left neighbor sharing a vertical wall
            left = a.right if side == 0 else a.left - iw - 1
            lo, hi = a.top - ih + 1, a.bottom - 2
            if lo > hi:
                continue
            top = int(rng.integers(lo, hi + 1))
        else:  # below / above sharing a horizontal wall
            top = a.bottom if side == 2 else a.top - ih - 1
            lo, hi = a.left - iw + 1, a.right - 2
            if lo > hi:
                continue
            left = int(rng.integers(lo, hi + 1))
        room = _Room(top, left, ih, iw, closed)
        if room.top < 0 or room.left < 0 or room.bottom >= h or room.right >= w:
            continue
        frame, inside = room.frame_cells(), room.interior_cells()
        ok = True
        for other in rooms:
            ofr, oin = other.frame_cells(), other.interior_cells()
            if inside & (ofr | oin) or frame & oin:
                ok = False
                break
            if other is not a and frame & ofr:
                ok = False  # only share with the chosen anchor
                break
        if ok and len(frame & a.frame_cells()) >= 3:
            return room
    return None


def _free_seed_spots(room, blocked):
    spots = []
    for r in range(room.top + 1, room.bottom - 1):
        for c in range(room.left + 1, room.right - 1):
            if not ({(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} & blocked):
                spots.append((r, c))
    return spots


def _room_interior_ok(room, wallset):
    free = [(r, c) for (r, c) in room.interior_cells() if (r, c) not in wallset]
    if not free:
        return False
    rows = [r for r, _ in free]
    cols = [c for _, c in free]
    if (min(rows), max(rows), min(cols), max(cols)) != (room.top + 1, room.bottom - 1, room.left + 1, room.right - 1):
        return False
    freeset = set(free)
    start = free[0]
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if n in freeset and n not in seen:
                seen.add(n)
                queue.append(n)
    return len(seen) == len(freeset)


def _add_marks(rng, room, wallset, count):
    seedcells = {(room.seed_pos[0] + dr, room.seed_pos[1] + dc) for dr in (0, 1) for dc in (0, 1)}
    added = 0
    for _ in range(30):
        if added >= count:
            break
        kind = int(rng.integers(3))
        cells = []
        if kind == 0:  # floating dot
            r = int(rng.integers(room.top + 1, room.bottom))
            c = int(rng.integers(room.left + 1, room.right))
            cells = [(r, c)]
        else:  # stub growing inward from one wall
            side = int(rng.integers(4))
            if side < 2:
                if room.ih < 3:
                    continue
                c = int(rng.integers(room.left + 1, room.right))
                length = int(rng.integers(1, max(2, room.ih // 2) + 1))
                r = room.top + 1 if side == 0 else room.bottom - 1
                step = 1 if side == 0 else -1
                cells = [(r + step * k, c) for k in range(length)]
            else:
                if room.iw < 3:
                    continue
                r = int(rng.integers(room.top + 1, room.bottom))
                length = int(rng.integers(1, max(2, room.iw // 2) + 1))
                c = room.left + 1 if side == 2 else room.right - 1
                step = 1 if side == 2 else -1
                cells = [(r, c + step * k) for k in range(length)]
        if any(cell in seedcells or cell in wallset for cell in cells):
            continue
        trial = wallset | set(cells)
        if not _room_interior_ok(room, trial):
            continue
        wallset |= set(cells)
        room.marks.extend(cells)
        added += 1
    return added


def _carve_gap(rng, room, rooms, h, w, wallset):
    other_frames = set()
    other_insides = set()
    for o in rooms:
        if o is not room:
            other_frames |= o.frame_cells()
            other_insides |= o.interior_cells()
    options = []
    for c in range(room.left + 1, room.right):
        options.append(((room.top, c), (room.top - 1, c)))
        options.append(((room.bottom, c), (room.bottom + 1, c)))
    for r in range(room.top + 1, room.bottom):
        options.append(((r, room.left), (r, room.left - 1)))
        options.append(((r, room.right), (r, room.right + 1)))
    good = []
    for cell, out in options:
        if cell in other_frames:
            continue
        if not (0 <= out[0] < h and 0 <= out[1] < w):
            good.append(cell)
            continue
        if out in other_frames or out in other_insides or out in wallset:
            continue
        good.append(cell)
    if not good:
        return False
    cell = good[int(rng.integers(len(good)))]
    wallset.discard(cell)
    room.gap = cell
    return True


def _build_pair(rng, cfg):
    """Try to build one pair from a configuration; return (input, output, info) or None."""
    h = int(rng.integers(cfg["h"][0], cfg["h"][1] + 1))
    w = int(rng.integers(cfg["w"][0], cfg["w"][1] + 1))
    bg, wall = cfg["bg"], cfg["wall"]
    rooms = []
    rest = [True] * (cfg["closed"] - 1) + [False] * cfg["open"]
    kinds = [True] + [rest[i] for i in rng.permutation(len(rest))]
    shared_done = 0
    for index, closed in enumerate(kinds):
        need_big = cfg["need_big"] if (index == 0 and closed) else 0
        need_ns = cfg.get("need_nonsquare", False) and index == 0 and closed
        ih, iw = _room_dims(rng, cfg["max_dim"], need_big, need_ns)
        if ih + 2 > h or iw + 2 > w:
            return None
        want_shared = rooms and (cfg["shared"] > shared_done if index > 0 else False)
        if want_shared or (rooms and rng.random() < cfg["shared_prob"]):
            room = _place_shared(rng, rooms, h, w, ih, iw, closed)
            if room is not None:
                shared_done += 1
        else:
            room = None
        if room is None:
            if want_shared:
                return None
            room = _place_free(rng, rooms, h, w, ih, iw, closed)
        if room is None:
            return None
        rooms.append(room)

    wallset = set()
    for room in rooms:
        wallset |= room.frame_cells()
    # Seeds
    for room in rooms:
        spots = _free_seed_spots(room, wallset)
        if not spots:
            return None
        if cfg.get("offcenter") and room.closed:
            center = (room.top + room.ih // 2, room.left + room.iw // 2)
            off = [s for s in spots if s != center]
            spots = off or spots
        room.seed_pos = spots[int(rng.integers(len(spots)))]
        pattern = cfg["patterns"][int(rng.integers(len(cfg["patterns"])))]
        room.seed = _seed_block(rng, {bg, wall}, pattern)
    # Marks inside rooms
    marked = 0
    for room in rooms:
        want = 0
        if rng.random() < cfg["mark_prob"]:
            want = int(rng.integers(1, cfg["max_marks"] + 1))
        if cfg["force_marks"] and room is rooms[0]:
            want = max(want, 1)
        marked += _add_marks(rng, room, wallset, want) if want else 0
    if cfg["force_marks"] and not rooms[0].marks:
        return None
    # Gaps for open rooms
    for room in rooms:
        if not room.closed and not _carve_gap(rng, room, rooms, h, w, wallset):
            return None

    grid = np.full((h, w), bg, dtype=int)
    for r, c in wallset:
        grid[r, c] = wall
    for room in rooms:
        r, c = room.seed_pos
        grid[r:r + 2, c:c + 2] = room.seed

    # Exterior distractor strokes (branches or edge-anchored lines)
    rects = set()
    for room in rooms:
        rects |= room.frame_cells() | room.interior_cells()
    for _ in range(cfg["strokes"]):
        for _attempt in range(20):
            vertical = bool(rng.integers(2))
            length = int(rng.integers(2, 7))
            if vertical:
                c = int(rng.integers(w))
                r = 0 if rng.random() < 0.5 else int(rng.integers(h))
                cells = [(r + k, c) for k in range(length) if r + k < h]
            else:
                r = int(rng.integers(h))
                c = 0 if rng.random() < 0.5 else int(rng.integers(w))
                cells = [(r, c + k) for k in range(length) if c + k < w]
            if any(cell in rects or grid[cell] != bg for cell in cells):
                continue
            near_gap = any(room.gap and abs(room.gap[0] - y) + abs(room.gap[1] - x) <= 2
                           for room in rooms for y, x in cells)
            if near_gap:
                continue
            trial = grid.copy()
            for cell in cells:
                trial[cell] = wall
            passable = trial != wall
            enclosed = passable & ~_exterior(passable)
            expected = np.zeros_like(enclosed)
            for room in rooms:
                if room.closed:
                    for cell in room.interior_cells():
                        if trial[cell] != wall:
                            expected[cell] = True
            if np.array_equal(enclosed, expected):
                grid = trial
                break

    # Stray exterior seed
    if rng.random() < cfg["stray_prob"]:
        for _attempt in range(30):
            r = int(rng.integers(0, h - 1))
            c = int(rng.integers(0, w - 1))
            r0, r1, c0, c1 = max(r - 1, 0), min(r + 3, h), max(c - 1, 0), min(c + 3, w)
            if np.all(grid[r0:r1, c0:c1] == bg) and not any(
                    (y, x) in rects for y in range(r0, r1) for x in range(c0, c1)):
                pattern = cfg["patterns"][int(rng.integers(len(cfg["patterns"])))]
                grid[r:r + 2, c:c + 2] = _seed_block(rng, {bg, wall}, pattern)
                break

    # Structural acceptance checks
    background, detected_wall, order = _roles(grid)
    if background != bg or detected_wall != wall:
        return None
    if len(order) > 2 and order[1][0] == order[2][0]:
        return None
    if order[0][0] == order[1][0]:
        return None
    _, _, sealed = _sealed_rooms(grid)
    closed_rooms = [room for room in rooms if room.closed]
    if len(sealed) != len(closed_rooms):
        return None
    for s in sealed:
        if s["seed"] is None:
            return None
        r0, r1, c0, c1 = s["bbox"]
        match = [room for room in closed_rooms
                 if (room.top + 1, room.bottom - 1, room.left + 1, room.right - 1) == (r0, r1, c0, c1)]
        if len(match) != 1:
            return None
    passable = grid != wall
    outside = _exterior(passable)
    for room in rooms:
        if not room.closed:
            r, c = room.seed_pos
            if not outside[r, c]:
                return None
    out = solve(grid)
    info = {
        "rooms": [(room.ih, room.iw, room.closed, bool(room.marks),
                   room.seed_pos != (room.top + room.ih // 2, room.left + room.iw // 2),
                   len(np.unique(room.seed))) for room in rooms],
        "shared": shared_done,
        "bg": bg, "wall": wall,
    }
    return grid, out, info


def _palette_pair(rng):
    bg = 0 if rng.random() < 0.55 else int(rng.integers(10))
    wall = int(rng.choice([c for c in range(10) if c != bg]))
    return bg, wall


def _config(rng, mode, is_test, index, palette):
    bg, wall = palette
    cfg = {
        "bg": bg, "wall": wall, "h": (10, 18), "w": (10, 20), "max_dim": 8,
        "closed": int(rng.integers(1, 3)), "open": int(rng.integers(0, 2)),
        "shared": 0, "shared_prob": 0.25, "mark_prob": 0.25, "max_marks": 2,
        "force_marks": False, "strokes": int(rng.integers(0, 3)), "stray_prob": 0.3,
        "need_big": 0, "patterns": SEED_PATTERNS[:6], "offcenter": rng.random() < 0.7,
    }
    if index == 0 and not is_test:
        # first demonstration: one sealed and one leaky room, off-center seeds
        cfg.update(closed=1, open=1, offcenter=True, need_nonsquare=True,
                   patterns=("distinct", "three"), mark_prob=0.0, shared_prob=0.0, strokes=0)
    if not is_test and rng.random() < 0.12:
        cfg["patterns"] = SEED_PATTERNS
    if mode == "scale_up":
        if is_test:
            cfg.update(h=(16, 26), w=(16, 28), max_dim=12, need_big=10, closed=int(rng.integers(1, 3)))
        else:
            cfg.update(max_dim=6)
    elif mode == "crowded":
        if is_test:
            cfg.update(h=(16, 26), w=(16, 28), closed=int(rng.integers(2, 4)), open=1,
                       shared=1, shared_prob=0.6, strokes=int(rng.integers(1, 4)), stray_prob=0.6)
        else:
            cfg.update(closed=min(cfg["closed"], 2), shared_prob=0.15 if index else 0.0)
    elif mode == "marks":
        if is_test:
            cfg.update(h=(14, 24), w=(14, 26), max_dim=10, force_marks=True, mark_prob=0.8,
                       max_marks=3, strokes=int(rng.integers(1, 4)), need_big=6)
        else:
            cfg.update(mark_prob=0.3, force_marks=(index == 1))
    elif mode == "palette":
        if is_test:
            cfg.update(h=(12, 24), w=(12, 26), max_dim=10, closed=int(rng.integers(1, 3)), open=1)
    return cfg


def _train_flags(infos):
    rooms = [room for info in infos for room in info["rooms"]]
    closed = [r for r in rooms if r[2]]
    return {
        "open": any(not r[2] for r in rooms),
        "offcenter": any(r[4] for r in closed),
        "nonsquare": any(r[0] != r[1] for r in closed),
        "multicolor": any(r[5] >= 2 and max(r[0], r[1]) >= 4 for r in closed),
        "marks": any(r[3] for r in closed),
        "max_dim": max(max(r[0], r[1]) for r in closed),
    }


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    mode = MODES[int(rng.integers(len(MODES)))]
    for _attempt in range(200):
        if mode == "palette":
            palettes = []
            while len(palettes) < num_train + num_test:
                p = _palette_pair(rng)
                if p not in palettes:
                    palettes.append(p)
        else:
            p = _palette_pair(rng)
            palettes = [p] * (num_train + num_test)
        pairs, infos, seen = [], [], set()
        failed = False
        for index in range(num_train + num_test):
            is_test = index >= num_train
            built = None
            for _try in range(80):
                cfg = _config(rng, mode, is_test, index if not is_test else index - num_train, palettes[index])
                built = _build_pair(rng, cfg)
                if built is not None and built[0].tobytes() + bytes(built[0].shape) not in seen:
                    break
                built = None
            if built is None:
                failed = True
                break
            grid, out, info = built
            seen.add(grid.tobytes() + bytes(grid.shape))
            pairs.append({"input": grid, "output": out})
            infos.append(info)
        if failed:
            continue
        flags = _train_flags(infos[:num_train])
        if not (flags["open"] and flags["offcenter"] and flags["nonsquare"] and flags["multicolor"]):
            continue
        if mode == "marks" and not flags["marks"]:
            continue
        if mode == "scale_up":
            test_max = min(max(max(r[0], r[1]) for r in info["rooms"] if r[2]) for info in infos[num_train:])
            if test_max <= flags["max_dim"]:
                continue
        puzzle = {"train": pairs[:num_train], "test": pairs[num_train:]}
        validate_generated_puzzle(puzzle)
        return puzzle
    raise RuntimeError("could not generate a puzzle")


# ----------------------------------------------------------------------------
# Validation and self test
# ----------------------------------------------------------------------------

def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    stats = {"open_seed": 0, "offcenter": 0, "nonsquare": 0}
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert set(pair) == {"input", "output"}
            inp = np.asarray(pair["input"], dtype=int)
            out = np.asarray(pair["output"], dtype=int)
            assert inp.ndim == 2 and 1 <= inp.shape[0] <= 30 and 1 <= inp.shape[1] <= 30
            assert inp.shape == out.shape
            assert inp.min() >= 0 and inp.max() <= 9 and out.min() >= 0 and out.max() <= 9
            key = inp.tobytes() + bytes(inp.shape)
            assert key not in seen, "duplicate input"
            seen.add(key)
            bg, wall, order = _roles(inp)
            assert wall is not None and order[0][0] > order[1][0], "background must be strict majority"
            assert len(order) < 3 or order[1][0] > order[2][0], "wall must be strict second color"
            _, _, rooms = _sealed_rooms(inp)
            assert rooms, "every pair needs a sealed room"
            changed = np.zeros(inp.shape, dtype=bool)
            for room in rooms:
                assert room["seed"] is not None, "sealed room without exactly one 2x2 seed"
                r0, r1, c0, c1 = room["bbox"]
                hh, ww = r1 - r0 + 1, c1 - c0 + 1
                assert hh % 2 == 0 and ww % 2 == 0 and (hh, ww) != (2, 2)
                # bounding box is surrounded by wall on all four sides
                assert np.all(inp[r0 - 1, c0 - 1:c1 + 2] == wall) and np.all(inp[r1 + 1, c0 - 1:c1 + 2] == wall)
                assert np.all(inp[r0 - 1:r1 + 2, c0 - 1] == wall) and np.all(inp[r0 - 1:r1 + 2, c1 + 1] == wall)
                for r, c in room["cells"]:
                    changed[r, c] = True
                sr, sc, _ = room["seed"]
                if hh != ww:
                    stats["nonsquare"] += split == "train"
                if (sr - r0, sc - c0) != (hh // 2 - 1, ww // 2 - 1):
                    stats["offcenter"] += split == "train"
            # outside sealed rooms nothing changes; wall cells never change
            assert np.array_equal(inp[~changed], out[~changed])
            assert np.all(out[inp == wall] == wall)
            assert not np.array_equal(inp, out)
            assert np.array_equal(solve(inp), out), "solve disagrees"
            # leaky seeds: 2x2 seeds on exterior cells
            passable = inp != wall
            outside = _exterior(passable)
            fg = (inp != bg) & (inp != wall) & outside
            if fg.any() and split == "train":
                stats["open_seed"] += 1
    assert stats["open_seed"] >= 1, "training must show an unchanged seed"
    assert stats["offcenter"] >= 1, "training must show an off-center seed"
    assert stats["nonsquare"] >= 1, "training must show a non-square sealed room"


def self_test() -> None:
    modes, shapes, scales, structures = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        for a, b in zip(puzzle["train"] + puzzle["test"], again["train"] + again["test"]):
            assert np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        modes.add(MODES[int(np.random.default_rng(seed).integers(len(MODES)))])
        for pair in puzzle["train"] + puzzle["test"]:
            inp = np.asarray(pair["input"])
            shapes.add(inp.shape)
            _, _, rooms = _sealed_rooms(inp)
            for room in rooms:
                r0, r1, c0, c1 = room["bbox"]
                scales.add(((r1 - r0 + 1) // 2, (c1 - c0 + 1) // 2))
            structures.add(inp.tobytes())
    assert modes == set(MODES), modes
    assert len(shapes) >= 30 and len(scales) >= 15, (len(shapes), len(scales))
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted {bad}")


if __name__ == "__main__":
    self_test()
    print("self_test ok")
