"""Wall-mounted locks trade colours with the one loose key that fits them.

Parent A (1acc24af) supplies the fit test: a recess opening towards the free
space accepts a loose piece when, after translation and quarter turns, the
piece fills the recess exactly; any extra material of the piece lies beyond
the recess mouth.  Parent B (d511f180) supplies the action: a simultaneous
two-colour transposition that keeps every position and leaves all other
colours inert.

Rule of this family: every *lock* is a solid block mounted on exactly one
border of the grid, with one or two keyholes carved into its free face.  Every
*key* is a free-floating one-colour object.  When a lock is fitted by exactly
one key and that key fits no other lock, the two colours are exchanged
simultaneously: the lock is repainted in the key's colour and the key in the
lock's colour.  Everything else, including unfitted locks and keys and the
background, is unchanged.  All roles are read from geometry, so solve() needs
no puzzle-level convention.
"""
from __future__ import annotations

import numpy as np

SIDES = ("top", "right", "bottom", "left")
# np.rot90 turns that bring the named border to the top of the grid.
_TO_TOP = {"top": 0, "right": 1, "bottom": 2, "left": 3}
_OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
_STYLES = ("exact", "stem", "base", "ring", "hook", "irregular")
_EXTENDED = _STYLES[1:]


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _check_counts(num_train, num_test):
    checked = []
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must be in {low}..{high}")
        checked.append(int(value))
    if checked[0] <= checked[1]:
        raise ValueError("num_train must exceed num_test")
    return checked[0], checked[1]


# --------------------------------------------------------------------------
# Cell-set geometry
# --------------------------------------------------------------------------

def _normalize(cells):
    cells = list(cells)
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _rotate(cells, turns):
    out = frozenset(cells)
    for _ in range(int(turns) % 4):
        out = frozenset((c, -r) for r, c in out)
    return _normalize(out)


def _reflect(cells):
    return _normalize((r, -c) for r, c in cells)


def _orientations(cells, reflect=False):
    shapes = [_rotate(cells, t) for t in range(4)]
    if reflect:
        mirror = _reflect(cells)
        shapes += [_rotate(mirror, t) for t in range(4)]
    return shapes


def _connected(cells):
    cells = set(cells)
    if not cells:
        return False
    start = min(cells)
    seen = {start}
    todo = [start]
    while todo:
        r, c = todo.pop()
        for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if q in cells and q not in seen:
                seen.add(q)
                todo.append(q)
    return len(seen) == len(cells)


def _fits_oriented(oriented, cavity):
    """Parent A's socket test for one fixed orientation of a normalized piece.

    The cavity's lowest row is its mouth.  Every piece cell at or above the
    mouth must be a cavity cell and every cavity cell must be covered; the
    remaining cells lie beyond the mouth, outside the lock.
    """
    top = min(r for r, _ in cavity)
    base = max(r for r, _ in cavity)
    left = min(c for _, c in cavity)
    right = max(c for _, c in cavity)
    width = max(c for _, c in oriented)
    for shift in range(left - width, right + 1):
        upper = frozenset((r + top, c + shift) for r, c in oriented if r + top <= base)
        if upper == cavity:
            return True
    return False


def _fits(piece, cavity, reflect=False):
    return any(_fits_oriented(o, cavity) for o in _orientations(piece, reflect))


def _near_miss_kinds(piece, cavity):
    """'gap': the piece slides in but leaves exactly one keyhole cell empty.
    'collision': the piece fills the keyhole but one extra cell hits the lock."""
    top = min(r for r, _ in cavity)
    base = max(r for r, _ in cavity)
    left = min(c for _, c in cavity)
    right = max(c for _, c in cavity)
    kinds = set()
    for oriented in _orientations(piece):
        width = max(c for _, c in oriented)
        for dr in (-1, 0, 1):
            for shift in range(left - width - 1, right + 2):
                upper = frozenset((r + top + dr, c + shift) for r, c in oriented
                                  if r + top + dr <= base)
                if dr >= 0 and upper < cavity and len(cavity - upper) == 1:
                    kinds.add("gap")
                if dr <= 0 and cavity < upper and len(upper - cavity) == 1:
                    kinds.add("collision")
    return kinds


# --------------------------------------------------------------------------
# Input-only analysis and the solver
# --------------------------------------------------------------------------

def _background(grid):
    return int(np.argmax(np.bincount(grid.ravel(), minlength=10)))


def _components(grid, background):
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    found = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or grid[r, c] == background:
                continue
            color = int(grid[r, c])
            seen[r, c] = True
            todo = [(r, c)]
            cells = []
            while todo:
                y, x = todo.pop()
                cells.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (0 <= yy < height and 0 <= xx < width and not seen[yy, xx]
                            and grid[yy, xx] == color):
                        seen[yy, xx] = True
                        todo.append((yy, xx))
            mask = np.zeros(grid.shape, dtype=bool)
            for y, x in cells:
                mask[y, x] = True
            found.append({"color": color, "cells": frozenset(cells), "mask": mask})
    return found


def _touched_sides(mask):
    hits = (("top", mask[0].any()), ("right", mask[:, -1].any()),
            ("bottom", mask[-1].any()), ("left", mask[:, 0].any()))
    return tuple(side for side, hit in hits if hit)


def _lock_view(mask, side):
    """Turn a border-mounted lock to the top and read its keyholes.

    A keyhole is a connected group of non-lock cells inside the lock's bounding
    box that reaches the box's free (lowest) row, i.e. the face opposite the
    mounting border.
    """
    turns = _TO_TOP[side]
    view = np.rot90(mask, turns)
    rows = np.flatnonzero(view.any(axis=1))
    cols = np.flatnonzero(view.any(axis=0))
    box = view[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    height, width = box.shape
    free = ~box
    seen = np.zeros(box.shape, dtype=bool)
    cavities = []
    for c0 in range(width):
        if not free[height - 1, c0] or seen[height - 1, c0]:
            continue
        seen[height - 1, c0] = True
        todo = [(height - 1, c0)]
        cells = []
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < height and 0 <= xx < width and free[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        cavities.append(frozenset(cells))
    return {"side": side, "turns": turns, "box": box, "cavities": cavities}


def _analyze(grid, fit=None):
    grid = np.asarray(grid, dtype=int)
    background = _background(grid)
    locks, keys, others = [], [], []
    for comp in _components(grid, background):
        sides = _touched_sides(comp["mask"])
        if not sides:
            comp["shape"] = _normalize(comp["cells"])
            keys.append(comp)
        elif len(sides) == 1:
            comp.update(_lock_view(comp["mask"], sides[0]))
            locks.append(comp)
        else:
            others.append(comp)
    test = fit or (lambda key, lock: any(_fits(key["shape"], cav) for cav in lock["cavities"]))
    fits = [[bool(test(key, lock)) for key in keys] for lock in locks]
    pairs = []
    for i, row in enumerate(fits):
        chosen = [j for j, ok in enumerate(row) if ok]
        if len(chosen) == 1 and sum(fits[x][chosen[0]] for x in range(len(locks))) == 1:
            pairs.append((i, chosen[0]))
    return {"background": background, "locks": locks, "keys": keys,
            "others": others, "fits": fits, "pairs": pairs}


def _exchange(grid, info, pairs):
    out = np.array(grid, dtype=int, copy=True)
    for i, j in pairs:
        lock, key = info["locks"][i], info["keys"][j]
        # Simultaneous transposition: both writes read the original colours.
        out[lock["mask"]] = key["color"]
        out[key["mask"]] = lock["color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    info = _analyze(grid)
    return _exchange(grid, info, info["pairs"])


# --------------------------------------------------------------------------
# Alternative rules used to prove that demonstrations distinguish shortcuts
# --------------------------------------------------------------------------

def _key_in_lock_frame(key, lock):
    view = np.rot90(key["mask"], lock["turns"])
    return _normalize(zip(*np.nonzero(view)))


def _translation_only(key, lock):
    shape = _key_in_lock_frame(key, lock)
    return any(_fits_oriented(shape, cav) for cav in lock["cavities"])


def _exact_only(key, lock):
    return any(len(key["shape"]) == len(cav) and _fits(key["shape"], cav)
               for cav in lock["cavities"])


def _contains_keyhole(key, lock):
    """Looser hypothesis: some part of the key has the keyhole's shape."""
    for cav in lock["cavities"]:
        target = _normalize(cav)
        for oriented in _orientations(key["shape"]):
            cells = set(oriented)
            for r0, c0 in oriented:
                tr, tc = min(target)
                dr, dc = r0 - tr, c0 - tc
                if all((r + dr, c + dc) in cells for r, c in target):
                    return True
    return False


def _with_reflection(key, lock):
    return any(_fits(key["shape"], cav, reflect=True) for cav in lock["cavities"])


def _gap(a, b):
    ra, ca = np.nonzero(a)
    rb, cb = np.nonzero(b)
    return int(np.min(np.maximum(np.abs(ra[:, None] - rb[None, :]),
                                 np.abs(ca[:, None] - cb[None, :]))))


def _alternative(grid, name):
    grid = np.asarray(grid, dtype=int)
    if name == "no_rotation":
        info = _analyze(grid, _translation_only)
        return _exchange(grid, info, info["pairs"])
    if name == "exact_plug_only":
        info = _analyze(grid, _exact_only)
        return _exchange(grid, info, info["pairs"])
    if name == "contains_keyhole":
        info = _analyze(grid, _contains_keyhole)
        return _exchange(grid, info, info["pairs"])
    info = _analyze(grid)
    if name == "one_way":
        out = grid.copy()
        for i, j in info["pairs"]:
            out[info["keys"][j]["mask"]] = info["locks"][i]["color"]
        return out
    if name == "lock_only":
        out = grid.copy()
        for i, j in info["pairs"]:
            out[info["locks"][i]["mask"]] = info["keys"][j]["color"]
        return out
    if name == "nearest_lock":
        # Ignore the fit test: every key trades with its nearest free lock.
        order = []
        for j, key in enumerate(info["keys"]):
            for i, lock in enumerate(info["locks"]):
                order.append((_gap(key["mask"], lock["mask"]), j, i))
        used_keys, used_locks, pairs = set(), set(), []
        for _, j, i in sorted(order):
            if j not in used_keys and i not in used_locks:
                used_keys.add(j)
                used_locks.add(i)
                pairs.append((i, j))
        return _exchange(grid, info, pairs)
    raise ValueError(name)


_SHORTCUTS = ("no_rotation", "exact_plug_only", "contains_keyhole", "one_way", "lock_only",
              "nearest_lock")


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

def _profile(rng, max_width, max_depth):
    for _ in range(100):
        width = int(rng.integers(1, max_width + 1))
        if rng.random() < 0.3:
            profile = [int(rng.integers(1, max_depth + 1))] * width
        else:
            profile = [int(rng.integers(1, max_depth + 1))]
            for _ in range(width - 1):
                step = int(rng.integers(-2, 3))
                profile.append(min(max_depth, max(1, profile[-1] + step)))
        if sum(profile) >= 3:
            return tuple(profile)
    return (2, 1)


def _make_lock(rng, notches, max_width, max_depth):
    profiles = [_profile(rng, max_width, max_depth) for _ in range(notches)]
    roof = 1 + int(rng.random() < 0.4)
    depth = max(max(p) for p in profiles) + roof
    walls = [1 + int(rng.random() < 0.4) for _ in range(notches + 1)]
    width = sum(walls) + sum(len(p) for p in profiles)
    block = np.ones((depth, width), dtype=bool)
    cavities = []
    x = walls[0]
    for i, profile in enumerate(profiles):
        cells = set()
        for j, d in enumerate(profile):
            for r in range(depth - d, depth):
                block[r, x + j] = False
                cells.add((r, x + j))
        cavities.append(frozenset(cells))
        x += len(profile) + walls[i + 1]
    return {"block": block, "cavities": cavities}


def _extend(rng, cavity, style):
    """A key built on a keyhole: the plug plus optional material beyond the mouth."""
    shape = set(cavity)
    base = max(r for r, _ in cavity)
    mouth = sorted(c for r, c in cavity if r == base)
    if style == "stem":
        width = int(rng.integers(1, min(3, len(mouth)) + 1))
        first = int(rng.integers(0, len(mouth) - width + 1))
        lo, hi = mouth[first], mouth[first + width - 1]
        for j in range(1, int(rng.integers(1, 4)) + 1):
            shape.update((base + j, c) for c in range(lo, hi + 1))
    elif style == "base":
        lo, hi = mouth[0], mouth[-1]
        for j in range(1, int(rng.integers(1, 3)) + 1):
            lo -= int(rng.random() < 0.6)
            hi += int(rng.random() < 0.6)
            shape.update((base + j, c) for c in range(lo, hi + 1))
    elif style == "ring":
        col = mouth[int(rng.integers(len(mouth)))]
        stem = int(rng.integers(1, 3))
        shape.update((base + j, col) for j in range(1, stem + 1))
        ring_w = int(rng.choice((3, 3, 4)))
        ring_h = int(rng.choice((3, 3, 4)))
        left = col - int(rng.integers(0, ring_w))
        top = base + stem + 1
        for r in range(top, top + ring_h):
            for c in range(left, left + ring_w):
                if r in (top, top + ring_h - 1) or c in (left, left + ring_w - 1):
                    shape.add((r, c))
    elif style == "hook":
        col = mouth[int(rng.integers(len(mouth)))]
        stem = int(rng.integers(1, 4))
        shape.update((base + j, col) for j in range(1, stem + 1))
        direction = 1 if rng.random() < 0.5 else -1
        arm = int(rng.integers(1, 4))
        shape.update((base + stem, col + direction * j) for j in range(1, arm + 1))
    elif style == "irregular":
        lo = mouth[int(rng.integers(len(mouth)))]
        hi = min(mouth[-1], lo + int(rng.integers(0, 2)))
        for j in range(1, int(rng.integers(2, 5)) + 1):
            if j > 1:
                step = int(rng.integers(-1, 2))
                old_lo, old_hi = lo, hi
                lo, hi = lo + step, hi + step
                lo, hi = min(lo, old_hi), max(hi, old_lo)
                if rng.random() < 0.3:
                    hi += 1
            shape.update((base + j, c) for c in range(lo, hi + 1))
    return shape


def _negative(rng, cavities, style, max_width, max_depth, forced=None):
    """A decoy that resembles a key but fits no keyhole (checked by the caller)."""
    kinds = ("remove", "add", "remove", "add", "phantom")
    kind = forced or kinds[int(rng.integers(len(kinds)))]
    if kind == "phantom":
        profile = _profile(rng, max_width, max_depth)
        cavity = frozenset((max(profile) - d + r, x) for x, d in enumerate(profile)
                           for r in range(d))
        return _normalize(_extend(rng, cavity, style)), kind
    cavity = cavities[int(rng.integers(len(cavities)))]
    shape = _extend(rng, cavity, style)
    base = max(r for r, _ in cavity)
    upper = sorted(cell for cell in shape if cell[0] <= base)
    order = [upper[i] for i in rng.permutation(len(upper))]
    if kind == "remove":
        for cell in order:
            trial = shape - {cell}
            if len(trial) >= 3 and _connected(trial):
                return _normalize(trial), kind
    else:
        for r, c in order:
            options = [(r - 1, c), (r, c - 1), (r, c + 1)]
            q = options[int(rng.integers(3))]
            if q not in shape and q[0] <= base:
                return _normalize(shape | {q}), kind
    return None, kind


def _relation(lock_cavities, keys, reflect):
    return [[any(_fits(key, cav, reflect) for cav in cavs) for key in keys]
            for cavs in lock_cavities]


def _free(taken, r0, c0, h, w):
    return not taken[max(0, r0 - 1):r0 + h + 1, max(0, c0 - 1):c0 + w + 1].any()


def _free_spots(taken, h, w):
    """Every interior position whose box, grown by one cell, avoids taken cells."""
    height, width = taken.shape
    table = np.pad(taken.astype(int).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    spots = []
    for r0 in range(1, height - h):
        for c0 in range(1, width - w):
            a, b = r0 - 1, c0 - 1
            c, d = min(height, r0 + h + 1), min(width, c0 + w + 1)
            if table[c, d] - table[a, d] - table[c, b] + table[a, b] == 0:
                spots.append((r0, c0))
    return spots


def _canvas(rng, spec, blocks, sides, keys):
    """Pick a canvas whose size follows the content, then a random density."""
    mounted = [np.rot90(b, -_TO_TOP[s]).shape for b, s in zip(blocks, sides)]
    padded = sum((h + 1) * (w + 2) for h, w in mounted)
    padded += sum((max(r for r, _ in k) + 2) * (max(c for _, c in k) + 2) for k in keys)
    along = {s: 0 for s in SIDES}
    depth = {s: 0 for s in SIDES}
    for (h, w), side in zip(mounted, sides):
        along[side] += (w if side in ("top", "bottom") else h) + 1
        depth[side] = max(depth[side], h if side in ("top", "bottom") else w)
    key_h = max(max(r for r, _ in k) + 1 for k in keys)
    key_w = max(max(c for _, c in k) + 1 for k in keys)
    min_h = max(along["left"] + 2, along["right"] + 2,
                depth["top"] + depth["bottom"] + key_h + 4, spec["size"][0])
    min_w = max(along["top"] + 2, along["bottom"] + 2,
                depth["left"] + depth["right"] + key_w + 4, spec["size"][0])
    density = float(rng.uniform(*spec["density"]))
    side = (padded / density) ** 0.5
    aspect = float(np.exp(rng.uniform(-0.25, 0.25)))
    height = int(round(max(min_h, min(spec["size"][1], side * aspect))))
    width = int(round(max(min_w, min(spec["size"][1], side / aspect))))
    return min(30, height), min(30, width)


def _place(rng, spec, blocks, sides, keys):
    """Mount locks on their borders and scatter keys; returns placements or None."""
    base_h, base_w = _canvas(rng, spec, blocks, sides, keys)
    for grow in range(16):
        height = min(30, base_h + (grow + 1) // 4)
        width = min(30, base_w + (grow + 3) // 4)
        taken = np.zeros((height, width), dtype=bool)
        lock_spots, key_spots = [], []
        ok = True
        for block, side in zip(blocks, sides):
            mounted = np.rot90(block, -_TO_TOP[side])
            h, w = mounted.shape
            if side in ("top", "bottom"):
                if h > height - 3 or w > width - 2:
                    ok = False
                    break
                r0 = 0 if side == "top" else height - h
                spots = [(r0, c) for c in range(1, width - w)]
            else:
                if w > width - 3 or h > height - 2:
                    ok = False
                    break
                c0 = 0 if side == "left" else width - w
                spots = [(r, c0) for r in range(1, height - h)]
            spots = [spots[i] for i in rng.permutation(len(spots))]
            spot = next((s for s in spots if _free(taken, s[0], s[1], h, w)), None)
            if spot is None:
                ok = False
                break
            taken[spot[0]:spot[0] + h, spot[1]:spot[1] + w] = True
            lock_spots.append((spot, mounted))
        if not ok:
            continue
        key_spots = [None] * len(keys)
        order = sorted(range(len(keys)), key=lambda j: -(max(r for r, _ in keys[j]) + 1)
                       * (max(c for _, c in keys[j]) + 1))
        for j in order:
            shape = keys[j]
            h = max(r for r, _ in shape) + 1
            w = max(c for _, c in shape) + 1
            spots = _free_spots(taken, h, w)
            if not spots:
                ok = False
                break
            spot = spots[int(rng.integers(len(spots)))]
            taken[spot[0]:spot[0] + h, spot[1]:spot[1] + w] = True
            key_spots[j] = spot
        if ok:
            return height, width, lock_spots, key_spots
    return None


def _choose_style(rng, weights, extended=False):
    if extended:
        sub = np.array([weights[i] for i in range(1, len(_STYLES))], dtype=float)
        return _EXTENDED[int(rng.choice(len(_EXTENDED), p=sub / sub.sum()))]
    return _STYLES[int(rng.choice(len(_STYLES), p=weights))]


def _make_pair(rng, spec, params):
    n_locks, n_keys, n_matched = spec["locks"], spec["keys"], spec["matched"]
    for _ in range(120):
        locks = [_make_lock(rng, spec["notches"][i], spec["max_width"], spec["max_depth"])
                 for i in range(n_locks)]
        matched = sorted(int(x) for x in rng.choice(n_locks, size=n_matched, replace=False))
        shapes, partner = [], []
        for n, li in enumerate(matched):
            cavs = locks[li]["cavities"]
            cavity = cavs[int(rng.integers(len(cavs)))]
            style = _choose_style(rng, params["styles"], extended=(n == 0 and spec["need_extension"]))
            shapes.append(_normalize(_extend(rng, cavity, style)))
            partner.append(li)
        all_cavities = [cav for lock in locks for cav in lock["cavities"]]
        failed = False
        for n in range(n_keys - n_matched):
            style = _choose_style(rng, params["styles"])
            forced = spec["near_miss"][n] if n < len(spec["near_miss"]) else None
            piece, _ = _negative(rng, all_cavities, style, spec["max_width"], spec["max_depth"],
                                 forced)
            if piece is None:
                failed = True
                break
            shapes.append(piece)
            partner.append(None)
        if failed or len(set(shapes)) < len(shapes) and rng.random() < 0.7:
            continue
        if any(not _connected(s) or max(max(r for r, _ in s), max(c for _, c in s)) > 9
               for s in shapes):
            continue
        intended = [[partner[j] == i for j in range(n_keys)] for i in range(n_locks)]
        cav_lists = [lock["cavities"] for lock in locks]
        if _relation(cav_lists, shapes, False) != intended:
            continue
        if _relation(cav_lists, shapes, True) != intended:
            continue
        order = [int(x) for x in rng.permutation(n_keys)]
        display = [_rotate(shapes[j], int(rng.integers(4))) for j in order]
        placed = _place(rng, spec, [lock["block"] for lock in locks], spec["sides"], display)
        if placed is None:
            continue
        height, width, lock_spots, key_spots = placed
        background = params["background"] if params["background"] is not None else int(rng.integers(10))
        palette = [int(x) for x in rng.permutation([c for c in range(10) if c != background])]
        lock_colors = palette[:n_locks]
        key_colors = palette[n_locks:n_locks + n_keys]
        grid = np.full((height, width), background, dtype=int)
        for (spot, mounted), color in zip(lock_spots, lock_colors):
            h, w = mounted.shape
            region = grid[spot[0]:spot[0] + h, spot[1]:spot[1] + w]
            region[mounted] = color
        for shape, spot, color in zip(display, key_spots, key_colors):
            for r, c in shape:
                grid[spot[0] + r, spot[1] + c] = color
        counts = np.bincount(grid.ravel(), minlength=10)
        if counts[background] <= max(counts[c] for c in range(10) if c != background):
            continue
        info = _analyze(grid)
        if len(info["locks"]) != n_locks or len(info["keys"]) != n_keys or info["others"]:
            continue
        if len(info["pairs"]) != n_matched:
            continue
        if spec["need_rotation"] and not any(
                not _translation_only(info["keys"][j], info["locks"][i]) for i, j in info["pairs"]):
            continue
        if spec.get("min_cavity") and max(len(c) for lock in info["locks"]
                                          for c in lock["cavities"]) <= spec["min_cavity"]:
            continue
        output = _exchange(grid, info, info["pairs"])
        return {"input": grid, "output": output}
    return None


_TIERS = {
    # locks, keys, max keyhole width, max keyhole depth, canvas range
    "simple": {"locks": (1, 3), "keys": (2, 4), "max_width": 3, "max_depth": 3, "size": (8, 16),
               "objects": 6, "density": (0.6, 0.95)},
    "standard": {"locks": (2, 3), "keys": (3, 5), "max_width": 4, "max_depth": 4, "size": (10, 22),
                 "objects": 7, "density": (0.5, 0.85)},
    "rich": {"locks": (2, 4), "keys": (3, 6), "max_width": 4, "max_depth": 4, "size": (13, 26),
             "objects": 8, "density": (0.45, 0.75)},
}


def _puzzle_params(rng):
    tier = str(rng.choice(("simple", "standard", "rich"), p=(0.3, 0.4, 0.3)))
    modes = {"simple": (("single_fixed", "single_varied"), (0.65, 0.35)),
             "standard": (("single_fixed", "single_varied", "opposite", "mixed"),
                          (0.3, 0.2, 0.3, 0.2)),
             "rich": (("single_varied", "opposite", "mixed"), (0.2, 0.35, 0.45))}
    names, probs = modes[tier]
    wall_mode = str(rng.choice(names, p=probs))
    background = None
    roll = rng.random()
    if roll < 0.45:
        background = 0
    elif roll < 0.7:
        background = int(rng.integers(1, 10))
    # Share of plain plugs versus keys with handles, then the handle styles.
    plain = float(rng.uniform(0.05, 0.45))
    handles = rng.dirichlet(np.full(len(_EXTENDED), 1.2)) + 0.04
    styles = np.concatenate(([plain], (1 - plain) * handles / handles.sum()))
    return {"tier": tier, "wall_mode": wall_mode,
            "fixed_side": str(rng.choice(SIDES, p=(0.55, 0.15, 0.15, 0.15))),
            "background": background,
            "multi_notch": {"simple": 0.0, "standard": 0.15, "rich": 0.3}[tier],
            "styles": styles,
            "shift": str(rng.choice(("comparable", "more", "larger"), p=(0.4, 0.35, 0.25)))}


def _sides_for(rng, params, n_locks, allowed):
    mode = params["wall_mode"]
    pool = [s for s in SIDES if allowed is None or s in allowed]
    if mode == "single_fixed":
        return [params["fixed_side"]] * n_locks
    if mode == "single_varied":
        side = pool[int(rng.integers(len(pool)))]
        return [side] * n_locks
    if mode == "opposite":
        axes = [axis for axis in (("top", "bottom"), ("left", "right"))
                if all(s in pool for s in axis)]
        if not axes:
            return [pool[int(rng.integers(len(pool)))] for _ in range(n_locks)]
        axis = axes[int(rng.integers(len(axes)))]
        sides = [axis[i % 2] for i in range(n_locks)]
        return [sides[i] for i in rng.permutation(n_locks)]
    sides = [pool[int(rng.integers(len(pool)))] for _ in range(n_locks)]
    if n_locks >= 2 and len(set(sides)) == 1 and len(pool) > 1:
        sides[-1] = [s for s in pool if s != sides[0]][int(rng.integers(len(pool) - 1))]
    return sides


def _pair_spec(rng, params, role, index, train_stats):
    tier = _TIERS[params["tier"]]
    lo_l, hi_l = tier["locks"]
    lo_k, hi_k = tier["keys"]
    max_width, max_depth = tier["max_width"], tier["max_depth"]
    size = tier["size"]
    teach0 = role == "train" and index == 0
    teach1 = role == "train" and index == 1
    if teach1:
        lo_l = max(lo_l, 2)
    n_locks = int(rng.integers(lo_l, hi_l + 1))
    n_keys = int(rng.integers(lo_k, hi_k + 1))
    min_cavity = 0
    cap = tier["objects"]
    if role == "test" and params["shift"] == "more":
        cap = min(9, cap + 1)
        target = min(cap, train_stats["max_objects"] + int(rng.integers(1, 3)))
        n_locks = min(4, max(n_locks, target - hi_k))
        n_keys = max(2, min(6, target - n_locks))
    if role == "test" and params["shift"] == "larger" and train_stats["max_cavity"] <= 12:
        max_width, max_depth = min(5, max_width + 1), min(5, max_depth + 1)
        min_cavity = train_stats["max_cavity"]
        size = (size[0] + 2, min(28, size[1] + 3))
    if n_locks + n_keys > cap:
        n_keys = max(2, cap - n_locks)
        n_locks = cap - n_keys
    if teach0 and n_keys < 2:
        n_keys = 2
    top = min(n_locks, n_keys)
    if teach0:
        n_matched = int(rng.integers(1, min(top, n_keys - 1) + 1))
    elif teach1:
        n_matched = int(rng.integers(1, min(top, n_locks - 1, n_keys - 1) + 1))
    elif rng.random() < (0.07 if role == "train" else 0.03):
        n_matched = 0
    else:
        n_matched = int(rng.integers(1, top + 1))
    allowed = None if role == "train" else train_stats["sides"]
    sides = _sides_for(rng, params, n_locks, allowed)
    multi_allowed = params["multi_notch"] > 0 and (role == "train" or train_stats["multi"])
    notches = [1 + int(multi_allowed and rng.random() < params["multi_notch"])
               for _ in range(n_locks)]
    if teach1 and params["multi_notch"] > 0 and max(notches) == 1:
        notches[int(rng.integers(n_locks))] = 2
    return {"locks": n_locks, "keys": n_keys, "matched": n_matched, "sides": sides,
            "notches": notches, "max_width": max_width, "max_depth": max_depth,
            "size": size, "density": tier["density"], "min_cavity": min_cavity,
            "need_rotation": teach0, "need_extension": teach1,
            "near_miss": ["remove"] if teach0 else ["add"] if teach1 else []}


def _grid_key(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.astype(np.int8).tobytes()


def _attempt_puzzle(rng, num_train, num_test):
    params = _puzzle_params(rng)
    puzzle = {"train": [], "test": []}
    seen = set()
    stats = {"sides": set(), "multi": False, "max_objects": 0, "max_cavity": 0}
    for role, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            pair = None
            for _ in range(10):
                spec = _pair_spec(rng, params, role, index, stats)
                pair = _make_pair(rng, spec, params)
                if pair is not None and _grid_key(pair["input"]) not in seen:
                    break
                pair = None
            if pair is None:
                return None
            seen.add(_grid_key(pair["input"]))
            puzzle[role].append(pair)
            if role == "train":
                info = _analyze(pair["input"])
                stats["sides"].update(lock["side"] for lock in info["locks"])
                stats["multi"] |= any(len(lock["cavities"]) > 1 for lock in info["locks"])
                stats["max_objects"] = max(stats["max_objects"],
                                           len(info["locks"]) + len(info["keys"]))
                stats["max_cavity"] = max([stats["max_cavity"]] + [
                    len(c) for lock in info["locks"] for c in lock["cavities"]])
    return puzzle


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(40):
        puzzle = _attempt_puzzle(rng, num_train, num_test)
        if puzzle is None:
            continue
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle
    raise AssertionError(f"could not build a valid puzzle for seed {seed}")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _pair_report(inp, out):
    """Strict input-derived checks for one pair; returns measured features."""
    _require(inp.shape == out.shape, "input/output dimensions differ")
    info = _analyze(inp)
    background = info["background"]
    counts = np.bincount(inp.ravel(), minlength=10)
    _require(all(counts[background] > counts[c] for c in range(10) if c != background),
             "background is not strictly the most frequent colour")
    _require(not info["others"], "an object touches two borders")
    objects = info["locks"] + info["keys"]
    colors = [obj["color"] for obj in objects]
    _require(len(colors) == len(set(colors)), "two objects share a colour")
    _require(1 <= len(info["locks"]) <= 4 and 2 <= len(info["keys"]) <= 6,
             "lock/key counts out of range")
    for a in range(len(objects)):
        for b in range(a + 1, len(objects)):
            _require(_gap(objects[a]["mask"], objects[b]["mask"]) >= 2, "objects touch")
    for lock in info["locks"]:
        box = lock["box"]
        _require(box[0].all(), "lock is not mounted flat on its border")
        _require(1 <= len(lock["cavities"]) <= 2, "lock needs one or two keyholes")
        free = int((~box).sum())
        _require(free == sum(len(c) for c in lock["cavities"]), "lock body has a sealed hole")
        width = box.shape[1]
        for cav in lock["cavities"]:
            cols = sorted({c for _, c in cav})
            _require(len(cav) >= 3, "keyhole too small")
            _require(cols[0] > 0 and cols[-1] < width - 1, "keyhole is not walled in")
            _require(cols == list(range(cols[0], cols[-1] + 1)), "keyhole mouth is split")
            base = max(r for r, _ in cav)
            for c in cols:
                rows = sorted(r for r, x in cav if x == c)
                _require(rows[-1] == base and rows == list(range(rows[0], base + 1)),
                         "keyhole has an undercut")
        for key in info["keys"]:
            dist = _gap(key["mask"], lock["mask"])
            _require(dist >= 2, "key touches a lock")
    for i, lock in enumerate(info["locks"]):
        for j, key in enumerate(info["keys"]):
            _require(_with_reflection(key, lock) == info["fits"][i][j],
                     "a key fits only after reflection")
    for i in range(len(info["locks"])):
        _require(sum(info["fits"][i]) <= 1, "a lock is fitted by several keys")
    for j in range(len(info["keys"])):
        _require(sum(row[j] for row in info["fits"]) <= 1, "a key fits several locks")
    expected = solve(inp)
    _require(np.array_equal(expected, out), "output disagrees with the rule")
    changed = out != inp
    paired = np.zeros(inp.shape, dtype=bool)
    for i, j in info["pairs"]:
        lock, key = info["locks"][i], info["keys"][j]
        paired |= lock["mask"] | key["mask"]
        _require(set(np.unique(out[lock["mask"]]).tolist()) == {key["color"]}, "lock not repainted")
        _require(set(np.unique(out[key["mask"]]).tolist()) == {lock["color"]}, "key not repainted")
    _require(np.array_equal(changed, paired), "cells outside traded objects changed")
    features = {"pairs": len(info["pairs"]), "locks": len(info["locks"]),
                "keys": len(info["keys"]), "sides": {lock["side"] for lock in info["locks"]},
                "multi": any(len(lock["cavities"]) > 1 for lock in info["locks"]),
                "max_cavity": max(len(c) for lock in info["locks"] for c in lock["cavities"]),
                "unmatched_keys": len(info["keys"]) - len(info["pairs"]),
                "unmatched_locks": len(info["locks"]) - len(info["pairs"]),
                "rotation": 0, "extension": 0, "exact": 0, "near_miss": 0,
                "gap": 0, "collision": 0}
    for i, j in info["pairs"]:
        lock, key = info["locks"][i], info["keys"][j]
        features["rotation"] += not _translation_only(key, lock)
        plug = min(len(c) for c in lock["cavities"] if _fits(key["shape"], c))
        features["extension"] += len(key["cells"]) > plug
        features["exact"] += len(key["cells"]) == plug
    matched_keys = {j for _, j in info["pairs"]}
    for j, key in enumerate(info["keys"]):
        if j in matched_keys:
            continue
        kinds = set()
        for lock in info["locks"]:
            for cav in lock["cavities"]:
                kinds |= _near_miss_kinds(key["shape"], cav)
        features["near_miss"] += bool(kinds)
        features["gap"] += "gap" in kinds
        features["collision"] += "collision" in kinds
    return features


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    seen = set()
    reports = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair")
            grids = []
            for role in ("input", "output"):
                grid = np.asarray(pair[role])
                _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
                         "grid dimensions must be 1..30")
                _require(np.issubdtype(grid.dtype, np.integer), "grid cells must be integers")
                _require(np.all((grid >= 0) & (grid <= 9)), "grid colours must be 0..9")
                grids.append(grid.astype(int))
            key = _grid_key(grids[0])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            reports[split].append(_pair_report(grids[0], grids[1]))
    tr = reports["train"]
    _require(all(r["pairs"] >= 1 for r in tr[:2]), "first two demonstrations must trade colours")
    _require(any(r["unmatched_keys"] for r in tr), "training never shows a key that fits nothing")
    _require(any(r["unmatched_locks"] for r in tr), "training never shows an unfitted lock")
    _require(any(r["rotation"] for r in tr), "training never needs a quarter turn")
    _require(any(r["extension"] for r in tr), "training never shows a key with a handle")
    _require(any(r["gap"] for r in tr), "training never shows a key that leaves a gap")
    _require(any(r["collision"] for r in tr), "training never shows a key that hits a wall")
    train_sides = set().union(*(r["sides"] for r in tr))
    for r in reports["test"]:
        _require(r["sides"] <= train_sides, "test mounts a lock on an undemonstrated border")
        _require(not r["multi"] or any(t["multi"] for t in tr),
                 "test has a two-keyhole lock that training never shows")
    # Every shortcut must be refuted by at least one demonstration.
    for name in _SHORTCUTS:
        _require(any(not np.array_equal(_alternative(np.asarray(p["input"], dtype=int), name),
                                        np.asarray(p["output"], dtype=int)) for p in train),
                 f"training does not refute shortcut {name}")


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _signature(puzzle):
    return tuple(_grid_key(p[role]) for split in ("train", "test")
                 for p in puzzle[split] for role in ("input", "output"))


def self_test() -> None:
    shapes, cavities, key_shapes, palettes = set(), set(), set(), set()
    side_sets, tiers_seen = set(), set()
    pair_counts, lock_counts, key_counts, train_counts = set(), set(), set(), set()
    multi = zero = full = opposite = mixed_walls = 0
    test_more = test_larger = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        max_train_objects = max_train_cavity = 0
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = np.asarray(pair["input"])
                info = _analyze(inp)
                shapes.add(inp.shape)
                palettes.add(tuple(np.unique(inp).tolist()))
                pair_counts.add(len(info["pairs"]))
                lock_counts.add(len(info["locks"]))
                key_counts.add(len(info["keys"]))
                sides = frozenset(lock["side"] for lock in info["locks"])
                side_sets.add(sides)
                opposite += len(sides) == 2 and any(_OPPOSITE[s] in sides for s in sides)
                mixed_walls += len(sides) >= 2 and not (
                    len(sides) == 2 and any(_OPPOSITE[s] in sides for s in sides))
                multi += any(len(lock["cavities"]) > 1 for lock in info["locks"])
                zero += not info["pairs"]
                full += len(info["pairs"]) == len(info["locks"])
                for lock in info["locks"]:
                    for cav in lock["cavities"]:
                        cavities.add(_normalize(cav))
                for key in info["keys"]:
                    key_shapes.add(min(tuple(sorted(o)) for o in _orientations(key["shape"])))
                objects = len(info["locks"]) + len(info["keys"])
                cav = max(len(c) for lock in info["locks"] for c in lock["cavities"])
                if split == "train":
                    max_train_objects = max(max_train_objects, objects)
                    max_train_cavity = max(max_train_cavity, cav)
                else:
                    test_more += objects > max_train_objects
                    test_larger += cav > max_train_cavity
                # The rule is orientation-free and palette-free.
                for turns in (1, 2, 3):
                    _require(np.array_equal(solve(np.rot90(inp, turns)),
                                            np.rot90(np.asarray(pair["output"]), turns)),
                             "solve is not rotation-equivariant")
                _require(np.array_equal(solve(np.fliplr(inp)),
                                        np.fliplr(np.asarray(pair["output"]))),
                         "solve is not mirror-equivariant")
                perm = np.roll(np.arange(10), 3)
                _require(np.array_equal(solve(perm[inp]), perm[np.asarray(pair["output"])]),
                         "solve is not palette-equivariant")
    _require(len(signatures) == 42, "seeds repeat puzzles")
    _require(len(shapes) >= 40 and len(cavities) >= 40 and len(key_shapes) >= 150,
             "geometry lacks diversity")
    _require(len(palettes) >= 60, "palettes lack diversity")
    _require({0, 1, 2, 3}.issubset(pair_counts) and {1, 2, 3, 4}.issubset(lock_counts)
             and {2, 3, 4, 5, 6}.issubset(key_counts), "counts lack range")
    _require(len(side_sets) >= 8 and opposite > 0 and mixed_walls > 0,
             "lock mounting lacks variety")
    _require(multi > 0 and zero > 0 and full > 0, "pair composition lacks variety")
    _require(test_more > 0 and test_larger > 0, "train/test shift never occurs")
    for n, m in ((2, 1), (10, 4), (5, 4), (7, 3)):
        puzzle = generate_puzzle(1000 + n * 10 + m, n, m)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (n, m), "explicit counts ignored")
    a, b = generate_puzzle(4242, 6, 2), generate_puzzle(4242, 6, 2)
    _require(_signature(a) == _signature(b), "generation is not deterministic")
    for n, m in ((1, 1), (11, 1), (4, 0), (4, 5), (3, 3), (2, 2), (True, 1)):
        try:
            generate_puzzle(5, n, m)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {n}/{m}")
