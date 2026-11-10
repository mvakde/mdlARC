"""A conserved split gate docks to caps, then an edge key fills its cavities.

Only NumPy and the standard library are used. No parent generator or saved
example is consulted by generate_puzzle or by the input-only solve function.
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np


_SIDES = ("top", "left", "bottom", "right")
_STEPS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_STEPS8 = _STEPS4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
_SEEDS = (0, 913631297, 203593611, 72172529, 36519171, 834978114,
          389957279, 806731918, 293030573, 623611196, 152015820,
          422725370, 371150391, 940989434, 883994893, 39198800,
          267840568, 23925201, 399214398, 790072475, 399899230,
          602065177, 91910294, 791353392, 132563015, 564295315,
          756027280, 890721163, 410261556, 847192155, 505001810,
          597030892, 48735279, 90497884, 571918432, 224526815,
          442744031, 797755186, 895632850, 921889249, 241705881,
          337692307)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    """Deterministic connected components; coordinates are ordinary integers."""
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    steps = _STEPS8 if diagonal else _STEPS4
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in steps:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _bounds(mask):
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _rectangle(mask):
    box = _bounds(mask)
    if box is None:
        return None
    y0, y1, x0, x1 = box
    return box if int(mask.sum()) == (y1-y0)*(x1-x0) else None


def _mask(shape, cells):
    a = np.zeros(shape, dtype=bool)
    for q in cells:
        a[q] = True
    return a


def _read_key(a, bg):
    candidates = []
    for cells in _components(a != bg, diagonal=True):
        mask = _mask(a.shape, cells)
        box = _rectangle(mask)
        if box is None:
            continue
        y0, y1, x0, x1 = box
        if (y1-y0, x1-x0) != (2, 2) or len(set(int(a[q]) for q in cells)) < 2:
            continue
        # Noncorner placement supplies exactly one boundary-facing direction.
        sides = []
        if y0 == 0:
            sides.append(("top", a[0, x0:x1], a[1, x0:x1]))
        if y1 == a.shape[0]:
            sides.append(("bottom", a[-1, x0:x1], a[-2, x0:x1]))
        if x0 == 0:
            sides.append(("left", a[y0:y1, 0], a[y0:y1, 1]))
        if x1 == a.shape[1]:
            sides.append(("right", a[y0:y1, -1], a[y0:y1, -2]))
        if len(sides) != 1:
            continue
        side, sources, fills = sides[0]
        sources, fills = tuple(map(int, sources)), tuple(map(int, fills))
        if len(set(sources)) == 2:
            candidates.append({"mask": mask, "side": side,
                               "mapping": dict(zip(sources, fills))})
    if len(candidates) != 1:
        raise ValueError("Expected one isolated noncorner two-entry edge key")
    return candidates[0]


def _canonical_roles(a, bg, key_mask, mapping):
    """Try a rightward interpretation; no generator template is assumed."""
    scene = a.copy()
    scene[key_mask] = bg
    source_mask = np.isin(scene, tuple(mapping))
    groups = _components(source_mask, diagonal=True)
    bodies = [g for g in groups if len({int(scene[q]) for q in g}) == 2]
    if len(bodies) != 1:
        return []
    body = _mask(a.shape, bodies[0])
    y0, y1, b0, b1 = _bounds(body)
    if (y1-y0) % 2 or y1-y0 < 6 or b1-b0 < 5:
        return []
    center = (y0+y1)//2
    upper = body.copy()
    upper[center:] = False
    lower = body & ~upper
    upper_colors, lower_colors = set(map(int, scene[upper])), set(map(int, scene[lower]))
    if len(upper_colors) != 1 or len(lower_colors) != 1 or upper_colors == lower_colors:
        return []
    top, bottom = next(iter(upper_colors)), next(iter(lower_colors))
    if upper.sum() != lower.sum():
        return []
    if len(_components(upper, True)) != 1 or len(_components(lower, True)) != 1:
        return []
    if not np.all(scene[center-1, b0:b1] == top) or not np.all(scene[center, b0:b1] == bottom):
        return []
    others = sorted(set(map(int, scene[~source_mask])) - {bg})
    if len(others) != 2:
        return []
    results = []
    for launcher, projectile in (others, others[::-1]):
        pmask, lmask = scene == projectile, scene == launcher
        pbox = _rectangle(pmask)
        if pbox is None:
            continue
        py0, py1, px0, px1 = pbox
        width = py1-py0
        if width not in (2, 4, 6, 8) or py0+py1 != 2*center or not 1 <= px1-px0 <= 4:
            continue
        lbox = _bounds(lmask)
        if lbox is None or lbox[2] != 0 or lbox[3] != px0 or px0 < 2 or b0 < px1:
            continue
        profile = []
        valid = True
        for x in range(px0):
            ys = np.flatnonzero(lmask[:, x])
            if (len(ys) < width or len(ys) % 2 or int(ys[0])+int(ys[-1])+1 != 2*center
                    or len(ys) != int(ys[-1])-int(ys[0])+1):
                valid = False
                break
            profile.append(len(ys))
        if not valid or any(p < q for p, q in zip(profile, profile[1:])):
            continue
        if px1 > a.shape[1]-(px1-px0):
            continue
        shift = width//2
        cap_groups = [g for g in groups if g is not bodies[0]]
        if len(cap_groups) != 2:
            continue
        cap_masks = [_mask(a.shape, g) for g in cap_groups]
        cap_boxes = [_rectangle(m) for m in cap_masks]
        if any(box is None for box in cap_boxes):
            continue
        cap_masks.sort(key=lambda m: _bounds(m)[0])
        tc, bc = cap_masks
        ty0, ty1, tx0, tx1 = _bounds(tc)
        by0, by1, bx0, bx1 = _bounds(bc)
        if (not 1 <= ty1-ty0 <= 2 or not 1 <= by1-by0 <= 2
                or (tx0, tx1) != (b0, b1) or (bx0, bx1) != (b0, b1)
                or ty1 != y0-shift or by0 != y1+shift
                or not np.all(scene[tc] == top) or not np.all(scene[bc] == bottom)):
            continue
        results.append({"bg": bg, "key_mask": key_mask, "mapping": mapping,
                        "upper": upper, "lower": lower, "body": body,
                        "caps": tc | bc, "launcher_mask": lmask,
                        "projectile_mask": pmask, "projectile": projectile,
                        "launcher": launcher, "top": top, "bottom": bottom,
                        "center": center, "width": width, "shift": shift,
                        "pbox": pbox, "bbox": (y0, y1, b0, b1),
                        "profile": tuple(map(int, profile)),
                        "cap_thickness": (ty1-ty0, by1-by0)})
    return results


def _decode(value):
    original = _grid(value)
    colors, counts = np.unique(original, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("Background must be the unique most frequent color")
    bg = int(colors[counts.argmax()])
    key = _read_key(original, bg)
    candidates = []
    for turns in range(4):
        a = np.rot90(original, turns).copy()
        kmask = np.rot90(key["mask"], turns).copy()
        for info in _canonical_roles(a, bg, kmask, key["mapping"]):
            info.update({"turns": turns, "key_side": key["side"],
                         "direction": (-turns) % 4})
            candidates.append((a, info))
    if len(candidates) != 1:
        raise ValueError("Launcher, projectile, halves, and caps must have unique roles")
    return candidates[0]


def _move(a, info):
    """Stage A. Simultaneous rigid translations, with no trail or clipping."""
    bg, shift = info["bg"], info["shift"]
    out = a.copy()
    out[info["body"] | info["projectile_mask"]] = bg
    for mask, delta, color in ((info["upper"], -shift, info["top"]),
                               (info["lower"], shift, info["bottom"])):
        ys, xs = np.nonzero(mask)
        yy = ys+delta
        if np.any(yy < 0) or np.any(yy >= a.shape[0]) or np.any(out[yy, xs] != bg):
            raise ValueError("Half translation collides or leaves the grid")
        out[yy, xs] = color
    py0, py1, px0, px1 = info["pbox"]
    length = px1-px0
    if np.any(out[py0:py1, -length:] != bg):
        raise ValueError("Projectile destination is obstructed")
    out[py0:py1, -length:] = info["projectile"]
    return out


def _cavities(a, bg, diagonal=False):
    holes = []
    h, w = a.shape
    for cells in _components(a == bg, diagonal=diagonal):
        if any(y == 0 or x == 0 or y == h-1 or x == w-1 for y, x in cells):
            continue
        boundary = set()
        for y, x in cells:
            for dy, dx in _STEPS4:
                yy, xx = y+dy, x+dx
                if a[yy, xx] != bg:
                    boundary.add(int(a[yy, xx]))
        holes.append((cells, boundary))
    return holes


def _fill(a, info, diagonal=False):
    """Stage B. Read each original cavity owner exactly once."""
    out = a.copy()
    for cells, colors in _cavities(a, info["bg"], diagonal):
        if len(colors) != 1 or not colors <= set(info["mapping"]):
            raise ValueError("Cavity ownership must be one listed half color")
        fill = info["mapping"][next(iter(colors))]
        for q in cells:
            out[q] = fill
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, movement, closure, and colors solely from the input."""
    a, info = _decode(input_grid)
    out = _fill(_move(a, info), info)
    return np.rot90(out, -info["turns"]).copy()


def _template(rng, h, d, lesson):
    """Connected notched half; all notches initially reach the outer face."""
    mask = np.ones((h, d), dtype=bool)
    if lesson == "diagonal":
        # Diagonal-only escape after docking: four-connected fill must survive.
        end = int(rng.integers(3, d))
        for x in range(2, end):
            depth = h-2 if x == 2 else int(rng.integers(1, h-1))
            mask[:depth, x] = False
        mask[h-2, :2] = False
    else:
        # Disjoint mouth intervals, separated by walls; depths may be stepped.
        x = 1
        mouths = 0
        while x < d-1:
            max_run = min(3, d-1-x)
            run = int(rng.integers(1, max_run+1))
            for col in range(x, x+run):
                depth = int(rng.integers(1, h-1))
                if lesson == "irregular" and mouths == 0:
                    depth = 1 if col == x else h-2
                mask[:depth, col] = False
            mouths += 1
            x += run+int(rng.integers(1, 3))
        if lesson == "irregular":
            # Guaranteed two different depths in one mouth and a second mouth.
            mask[:, :] = True
            mask[:1, 1] = False
            mask[:h-2, 2] = False
            mask[:int(rng.integers(1, h-1)), d-2] = False
        elif rng.random() < .55:
            # A side bay remains connected to the exterior after docking.
            mask[h-2, :2] = False
    return mask


def _key_positions(shape, side):
    h, w = shape
    length = w if side in ("top", "bottom") else h
    for start in range(1, length-2):
        if side == "top":
            yield [(0, start), (0, start+1), (1, start), (1, start+1)]
        elif side == "bottom":
            yield [(h-1, start), (h-1, start+1), (h-2, start), (h-2, start+1)]
        elif side == "left":
            yield [(start, 0), (start+1, 0), (start, 1), (start+1, 1)]
        else:
            yield [(start, w-1), (start+1, w-1), (start, w-2), (start+1, w-2)]


def _mapping(rng, sources, bg, launcher, projectile, style):
    a, b = sources
    fresh = [c for c in range(10) if c not in (bg, a, b)]
    f, g = map(int, rng.choice(fresh, 2, replace=False))
    if style == "chain":
        return {a: b, b: f}
    if style == "self":
        return {a: a, b: f}
    if style == "swap":
        return {a: b, b: a}
    if style == "shared":
        return {a: f, b: f}
    if style == "roles":
        return {a: launcher, b: projectile}
    return {a: f, b: g}


def _make_pair(rng, turns, absolute_side, width, kind, lesson, style):
    for _attempt in range(64):
        bg, launcher, projectile, top, bottom = map(int, rng.choice(10, 5, replace=False))
        h = int(rng.integers(4 if lesson == "irregular" else 3, 8))
        d = int(rng.integers(7 if lesson == "irregular" else 5, 13))
        cap_top, cap_bottom = map(int, rng.integers(1, 3, size=2))
        margins = list(map(int, rng.integers(0, 5, size=2)))
        shift = width//2
        height = sum(margins)+2*h+width+cap_top+cap_bottom
        if height > 30:
            continue
        center = margins[0]+cap_top+shift+h
        tip_extra = int(rng.integers(0, 3))
        tip = width+2*tip_extra
        if kind == "block":
            profile = [tip]*int(rng.integers(2, 5))
        elif kind == "shoulder":
            profile = [tip+2*int(rng.integers(1, 3))]*int(rng.integers(1, 3))
            profile += [tip]*int(rng.integers(1, 3))
        else:
            profile = [tip+4, tip+2, tip]
            if rng.random() < .5:
                profile.insert(0, tip+4)
        if max(profile) > 2*min(center, height-center):
            continue
        length = int(rng.integers(1, 5))
        gap = int(rng.integers(0, 5))
        tail = int(rng.integers(0, 7))
        b0 = len(profile)+length+gap
        axis = b0+d+tail
        if axis > 30:
            continue
        a = np.full((height, axis), bg, dtype=np.int8)
        for x, span in enumerate(profile):
            a[center-span//2:center+span//2, x] = launcher
        a[center-shift:center+shift, len(profile):len(profile)+length] = projectile
        half = _template(rng, h, d, lesson)
        lower_half = np.flipud(half)
        if rng.random() < .5:
            lower_half = np.fliplr(lower_half)
        a[center-h:center, b0:b0+d][half] = top
        a[center:center+h, b0:b0+d][lower_half] = bottom
        a[margins[0]:margins[0]+cap_top, b0:b0+d] = top
        a[center+h+shift:center+h+shift+cap_bottom, b0:b0+d] = bottom
        mapping = _mapping(rng, (top, bottom), bg, launcher, projectile, style)
        # Reserve local empty halos in BOTH configurations before adding the key.
        dummy = {"bg": bg, "shift": shift, "top": top, "bottom": bottom,
                 "upper": np.zeros(a.shape, dtype=bool), "lower": np.zeros(a.shape, dtype=bool),
                 "body": np.zeros(a.shape, dtype=bool), "projectile_mask": a == projectile,
                 "pbox": (center-shift, center+shift, len(profile), len(profile)+length),
                 "projectile": projectile}
        dummy["upper"][center-h:center, b0:b0+d] = half
        dummy["lower"][center:center+h, b0:b0+d] = lower_half
        dummy["body"] = dummy["upper"] | dummy["lower"]
        moved = _move(a, dummy)
        occupied = (a != bg) | (moved != bg)
        side = _SIDES[(_SIDES.index(absolute_side)-turns) % 4]
        positions = list(_key_positions(a.shape, side))
        rng.shuffle(positions)
        chosen = None
        for cells in positions:
            halo = {(y+dy, x+dx) for y, x in cells for dy, dx in ((0, 0),)+_STEPS8
                    if 0 <= y+dy < height and 0 <= x+dx < axis}
            if all(not occupied[q] for q in halo):
                chosen = cells
                break
        if chosen is None:
            continue
        sources = [top, bottom]
        rng.shuffle(sources)
        for i, source in enumerate(sources):
            a[chosen[i]] = source
            a[chosen[i+2]] = mapping[source]
        colors, counts = np.unique(a, return_counts=True)
        if int(colors[counts.argmax()]) != bg or np.count_nonzero(counts == counts.max()) != 1:
            continue
        inp = np.rot90(a, turns).copy()
        try:
            ca, info = _decode(inp)
            intermediate = _move(ca, info)
            holes = _cavities(intermediate, bg)
            if {next(iter(cs)) for _, cs in holes if len(cs) == 1} != {top, bottom}:
                continue
            if _cavities(ca, bg):
                continue
            output = np.rot90(_fill(intermediate, info), -info["turns"]).copy()
        except ValueError:
            continue
        return {"input": inp, "output": output}
    raise RuntimeError("Bounded scene construction exhausted its 64 attempts")


def _integer(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+" must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name+" outside its legal range")
    return int(value)


def _fingerprint(a):
    a = np.asarray(a, dtype=np.int8)
    return (a.shape, a.tobytes())


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Sample one rule, with pair-local colors and a held-out launch direction.

    The first two demonstrations jointly teach a one-step chain, self-fill,
    diagonal enclosure, open side bays, irregular pockets, and different widths.
    Counts are explicit integers; None and booleans are not count conventions.
    Construction is bounded by 64 scene attempts and 8 uniqueness attempts per
    pair. No files, traces, parent data, or persistent caches are used.
    """
    num_train = _integer("num_train", num_train, 2, 10)
    num_test = _integer("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    held = int(rng.integers(4))
    pool = [k for k in range(4) if k != held]
    first = int(rng.choice([k for k in pool if k % 2 == 0]))
    second = int(rng.choice([k for k in pool if k % 2 == 1]))
    train_turns = [first, second]+list(map(int, rng.choice(pool, num_train-2)))
    test_turns = [held]+list(map(int, rng.choice(range(4), num_test-1)))
    sides = list(map(str, rng.permutation(_SIDES)))
    widths = list(map(int, rng.choice((2, 4, 6, 8), 2, replace=False)))
    kinds = ["block", "shoulder"]
    if num_train >= 3:
        kinds.append("taper")
    puzzle, seen = {"train": [], "test": []}, set()
    for split, directions in (("train", train_turns), ("test", test_turns)):
        for i, direction in enumerate(directions):
            lesson = "diagonal" if split == "train" and i == 0 else (
                "irregular" if split == "train" and i == 1 else "random")
            style = "chain" if split == "train" and i == 0 else (
                "self" if split == "train" and i == 1 else
                str(rng.choice(("chain", "self", "swap", "shared", "roles", "external"))))
            width = widths[i] if split == "train" and i < 2 else int(rng.choice((2, 4, 6, 8)))
            kind = kinds[i] if split == "train" and i < len(kinds) else str(rng.choice(kinds))
            side = sides[i] if split == "train" and i < 2 else str(rng.choice(_SIDES))
            for _retry in range(8):
                pair = _make_pair(rng, direction, side, width, kind, lesson, style)
                fp = _fingerprint(pair["input"])
                if fp not in seen:
                    seen.add(fp)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Bounded distinct-input construction failed")
    return puzzle


def _analyze(pair):
    if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
        raise AssertionError("Each pair must have exactly input/output")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    a, info = _decode(inp)
    mid = _move(a, info)
    holes = _cavities(mid, info["bg"])
    expected = _fill(mid, info)
    actual = np.rot90(out, info["turns"])
    assert np.array_equal(actual, expected), "Incorrect motion or fill"
    assert not _cavities(a, info["bg"]), "Generated notches must initially be open"
    assert np.array_equal(np.bincount(a.ravel(), minlength=10),
                          np.bincount(mid.ravel(), minlength=10)), "Movement must conserve all ink"
    assert all(len(cs) == 1 and cs <= set(info["mapping"]) for _, cs in holes)
    assert {next(iter(cs)) for _, cs in holes} == {info["top"], info["bottom"]}
    assert np.any(mid != expected), "Omitting B must change the answer"
    assert np.array_equal(_fill(a, info), a), "B before A must have no cavity target"
    without_caps = mid.copy()
    without_caps[info["caps"]] = info["bg"]
    assert not _cavities(without_caps, info["bg"]), "Caps must create closure"
    stationary = info["key_mask"] | info["caps"] | info["launcher_mask"]
    assert np.array_equal(a[stationary], actual[stationary]), "Stationary ink changed"
    assert np.all(actual[info["projectile_mask"]] == info["bg"]), "Projectile trail remains"
    # All translated material is preserved, even if a destination equals ink.
    moved_material = np.zeros(a.shape, dtype=bool)
    body_boxes = np.zeros(a.shape, dtype=bool)
    for mask, delta, color in ((info["upper"], -info["shift"], info["top"]),
                               (info["lower"], info["shift"], info["bottom"])):
        ys, xs = np.nonzero(mask)
        assert np.all(actual[ys+delta, xs] == color)
        moved_material[ys+delta, xs] = True
        y0, y1, x0, x1 = _bounds(mask)
        body_boxes[y0+delta:y1+delta, x0:x1] = True
        assert np.any((a[ys+delta, xs] == info["bg"])), "Erase-only shortcut not distinguished"
    hole_mask = np.zeros(a.shape, dtype=bool)
    for cells, _ in holes:
        boundary = {(y+dy, x+dx) for y, x in cells for dy, dx in _STEPS4
                    if mid[y+dy, x+dx] != info["bg"]}
        assert any(info["caps"][q] for q in boundary), "Cavity must use a stationary cap"
        assert any(moved_material[q] for q in boundary), "Cavity must use a moved half"
        for q in cells:
            hole_mask[q] = True
    open_pockets = body_boxes & (mid == info["bg"]) & ~hole_mask
    assert np.all(actual[open_pockets] == info["bg"]), "An exterior-connected pocket was filled"
    y0, y1, b0, b1 = info["bbox"]
    c, s = info["center"], info["shift"]
    channel = mid[c-s:c+s, b0:b1]
    assert np.all((channel == info["bg"]) | (channel == info["projectile"]))
    hole_shapes = []
    for cells, _ in holes:
        ys, xs = zip(*cells)
        hole_shapes.append((len(cells), max(ys)-min(ys)+1, max(xs)-min(xs)+1))
    eight = _fill(mid, info, diagonal=True)
    mapping = info["mapping"]
    profile = info["profile"]
    return {"direction": info["direction"], "key_side": info["key_side"],
            "width": info["width"], "length": info["pbox"][3]-info["pbox"][2],
            "half_height": (y1-y0)//2, "thickness": b1-b0,
            "gap": b0-info["pbox"][3], "tail": a.shape[1]-b1,
            "profile_kind": ("block" if len(set(profile)) == 1 else
                             "shoulder" if len(set(profile)) == 2 else "taper"),
            "chain": any(v in mapping and mapping[v] != v for v in mapping.values()),
            "self_map": any(k == v for k, v in mapping.items()),
            "shared": len(set(mapping.values())) == 1,
            "diagonal_witness": not np.array_equal(eight, expected),
            "irregular": any(area != h*w for area, h, w in hole_shapes),
            "holes": len(holes), "fill_cells": sum(t[0] for t in hole_shapes),
            "bg": info["bg"], "cap_thickness": info["cap_thickness"],
            "boundary_cap": bool(np.any(info["caps"][[0, -1]])),
            "open_cells": int(open_pockets.sum()),
            "open_after": bool(np.any(open_pockets))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            features[split].append(_analyze(pair))
            fp = _fingerprint(pair["input"])
            assert fp not in seen, "Duplicate input"
            seen.add(fp)
    train, test = features["train"], features["test"]
    assert len({f["direction"] % 2 for f in train}) == 2
    assert len({f["width"] for f in train}) >= 2
    assert len({f["key_side"] for f in train}) >= 2
    assert test[0]["direction"] not in {f["direction"] for f in train}
    assert any(f["chain"] and f["diagonal_witness"] for f in train)
    assert any(f["open_after"] for f in train), "Missing open/closed contrast"
    assert any(f["self_map"] for f in train)
    assert any(f["irregular"] and f["holes"] >= 4 for f in train)
    assert all(f["profile_kind"] in {t["profile_kind"] for t in train} for f in test)


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(split.encode())
        for pair in puzzle[split]:
            for name in ("input", "output"):
                shape, data = _fingerprint(pair[name])
                digest.update(bytes(shape))
                digest.update(data)
    return digest.hexdigest()


def _exterior_check(a, info):
    """Supplementary topology check via per-color padded exterior flood.

    This is a different fill algorithm, sharing role discovery and movement;
    it is not an independent full solver or blind prediction.
    """
    result = a.copy()
    for color, fill in info["mapping"].items():
        wall = a == color
        wall[info["key_mask"]] = False
        passable = np.pad(~wall, 1, constant_values=True)
        reached = np.zeros(passable.shape, dtype=bool)
        reached[0, 0] = True
        queue = deque([(0, 0)])
        while queue:
            y, x = queue.popleft()
            for dy, dx in _STEPS4:
                yy, xx = y+dy, x+dx
                if (0 <= yy < reached.shape[0] and 0 <= xx < reached.shape[1]
                        and passable[yy, xx] and not reached[yy, xx]):
                    reached[yy, xx] = True
                    queue.append((yy, xx))
        result[(a == info["bg"]) & ~reached[1:-1, 1:-1]] = fill
    return result


def self_test():
    """Bounded, streaming 42-seed checks; no diagnostic files or run archives."""
    # A small hand-specified regression: width two moves each half one cell;
    # four open notches close against fixed caps. Key 3->4->5 applies once.
    hand = np.zeros((15, 18), dtype=np.int8)
    hand[6:8, :2] = 1
    hand[6:8, 2:4] = 2
    hand[4:7, 7:12] = 3
    hand[7:10, 7:12] = 4
    hand[4, [8, 10]] = 0
    hand[9, [8, 10]] = 0
    hand[2, 7:12] = 3
    hand[11, 7:12] = 4
    hand[:2, 14:16] = [[3, 4], [4, 5]]
    expected = hand.copy()
    expected[6:8, 2:4] = 0
    expected[4:10, 7:12] = 0
    expected[3:6, 7:12] = 3
    expected[3, [8, 10]] = 4
    expected[8:11, 7:12] = 4
    expected[10, [8, 10]] = 5
    expected[6:8, 16:18] = 2
    for turns in range(4):
        assert np.array_equal(solve(np.rot90(hand, turns)), np.rot90(expected, turns))
    _analyze({"input": hand, "output": expected})
    signatures, shapes, structures = set(), set(), set()
    records = {"train": [], "test": []}
    for seed in _SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                records[split].append(_analyze(pair))
                assert np.array_equal(solve(inp.copy()), out)
                a, info = _decode(inp)
                assert np.array_equal(_exterior_check(_move(a, info), info),
                                      np.rot90(out, info["turns"]))
                shapes.add(inp.shape)
                palette = {}
                norm = bytes(palette.setdefault(int(v), len(palette)) for v in inp.flat)
                structures.add((inp.shape, hashlib.sha256(norm).digest()))
        if seed == 0:
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
            # Equivariance checks include new positions, key sides, and black ink.
            pair = puzzle["train"][0]
            perm = np.array([7, 4, 9, 1, 6, 0, 8, 3, 5, 2], dtype=np.int8)
            for k in range(4):
                assert np.array_equal(solve(perm[np.rot90(pair["input"], k)]),
                                      perm[np.rot90(pair["output"], k)])
    assert len(signatures) == 42 and len(shapes) >= 50 and len(structures) >= 200
    for split, rows in records.items():
        assert {f["direction"] for f in rows} == set(range(4)), split
        assert {f["key_side"] for f in rows} == set(_SIDES), split
        assert {f["width"] for f in rows} == {2, 4, 6, 8}, split
        assert len({f["bg"] for f in rows}) >= 8
        assert len({f["thickness"] for f in rows}) >= 6
        assert len({f["half_height"] for f in rows}) >= 4
        assert any(f["boundary_cap"] for f in rows)
    all_rows = records["train"]+records["test"]
    assert {f["length"] for f in all_rows} == {1, 2, 3, 4}
    assert max(f["holes"] for f in all_rows) >= 6
    assert any(f["shared"] for f in all_rows)
    assert any(f["gap"] == 0 for f in all_rows) and any(f["tail"] == 0 for f in all_rows)
    # Every legal explicit count, using one puzzle at a time.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(1000+10*nt+ne, nt, ne)
            validate_generated_puzzle(puzzle)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Reject a semantically wrong output, rather than just illegal dimensions.
    bad = generate_puzzle(0, 2, 1)
    pair = bad["test"][0]
    a, info = _decode(pair["input"])
    pair["output"] = np.rot90(_move(a, info), -info["turns"]).copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Missing cavity fills were accepted")
