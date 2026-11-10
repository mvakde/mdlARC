"""Textured pieces dock on color-matched terrain, using fine-pixel support.

Only NumPy and the standard library are used. Geometry, palettes and texture are
read from the input; generation policies never enter solve().
"""
from __future__ import annotations

import json
import numpy as np

FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(value, message):
    if not value:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=True)


def _components(grid, background, diagonal=False, ignore_colors=False):
    remaining = set(map(tuple, np.argwhere(grid != background)))
    parts = []
    offsets = EIGHT if diagonal else FOUR
    while remaining:
        start = min(remaining)
        color = int(grid[start])
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in offsets:
                p = r + dr, c + dc
                if p in remaining and (ignore_colors or int(grid[p]) == color):
                    remaining.remove(p)
                    stack.append(p)
        parts.append({"color": color, "cells": tuple(sorted(cells))})
    return parts


def _box(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _inside(box, outer):
    return outer[0] <= box[0] <= box[1] <= outer[1] and outer[2] <= box[2] <= box[3] <= outer[3]


def _framed_boxes(grid, background):
    """Maximal complete monochrome borders identify the two reference panels."""
    candidates = []
    for color in sorted(set(map(int, np.unique(grid))) - {background}):
        binary = np.where(grid == color, color, -1)
        boxes = []
        for part in _components(binary, -1):
            r0, r1, c0, c1 = _box(part["cells"])
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue
            crop = grid[r0:r1+1, c0:c1+1]
            if all(np.all(side == color) for side in (crop[0], crop[-1], crop[:, 0], crop[:, -1])):
                boxes.append((r0, r1, c0, c1))
        boxes = [b for b in boxes if not any(b != a and _inside(b, a) for a in boxes)]
        if len(boxes) != 2:
            continue
        coverage = np.zeros(grid.shape, dtype=bool)
        for r0, r1, c0, c1 in boxes:
            coverage[r0:r1+1, c0:c1+1] = True
        if np.all(coverage[grid == color]):
            candidates.append((color, sorted(boxes)))
    _require(len(candidates) == 1, "two uniquely framed panels required")
    return candidates[0]


def _expand(part, stamp, background, ink, rail, solid=False):
    """Keep coordinates in the entire source lattice, rather than recropping it."""
    sh, sw = stamp.shape
    pixels = {}
    for r, c in part["cells"]:
        for y in range(sh):
            for x in range(sw):
                v = int(stamp[y, x])
                if solid or v != background:
                    pixels[(r*sh+y, c*sw+x)] = rail if solid or v == ink else v
    return pixels


def _landing(pixels, well):
    bottom = {}
    for r, c in pixels:
        bottom[c] = max(bottom.get(c, -1), r)
    _require(all(0 <= c < len(well["floor"]) for c in bottom), "lateral overflow")
    shift = min(int(well["floor"][c]) - 1 - r for c, r in bottom.items())
    targets = {(r + shift, c + well["left"] + 1): v for (r, c), v in pixels.items()}
    _require(all(well["top"] <= r < int(well["floor"][c-well["left"]-1])
                 for r, c in targets), "piece does not fit above terrain")
    return shift, targets


def _canonical_read(grid, background):
    ink, boxes = _framed_boxes(grid, background)
    source_candidates, stamp_candidates = [], []
    remaining = grid.copy()
    for box in boxes:
        r0, r1, c0, c1 = box
        inner = grid[r0+1:r1, c0+1:c1].copy()
        colors = set(map(int, np.unique(inner))) - {background}
        if ink in colors and len(colors - {ink}) == 1:
            stamp_candidates.append((box, inner, next(iter(colors - {ink}))))
        elif ink not in colors and len(colors) >= 2:
            source_candidates.append((box, inner))
        remaining[r0:r1+1, c0:c1+1] = background
    _require(len(source_candidates) == len(stamp_candidates) == 1, "panel roles ambiguous")
    source_box, source = source_candidates[0]
    stamp_box, stamp, accent = stamp_candidates[0]
    _require(all(np.any(side != background) for side in
                 (stamp[0], stamp[-1], stamp[:, 0], stamp[:, -1])), "invisible stamp padding")
    wells = []
    for comp in _components(remaining, background, ignore_colors=True):
        top, bottom, left, right = _box(comp["cells"])
        crop = grid[top:bottom+1, left:right+1]
        _require(crop.size == len(comp["cells"]) and right-left >= 3 and bottom-top >= 2,
                 "receptacles must be complete rectangles")
        rail = int(crop[0, 0])
        route = int(crop[0, 1])
        _require(rail != route and rail not in (background, ink, accent) and
                 route not in (background, ink, accent), "invalid receptacle color roles")
        _require(np.all(crop[:, 0] == rail) and np.all(crop[:, -1] == rail) and
                 np.all(crop[-1] == rail) and np.all(crop[0, 1:-1] == route), "invalid side rails")
        _require(right-left-1 == source.shape[1]*stamp.shape[1], "fine lattice width mismatch")
        floor = []
        for col in range(1, crop.shape[1]-1):
            first = int(np.flatnonzero(crop[:, col] == rail)[0])
            _require(first > 0 and np.all(crop[:first, col] == route) and
                     np.all(crop[first:, col] == rail), "terrain must be solid downward")
            floor.append(top+first)
        wells.append(dict(top=top, bottom=bottom, left=left, right=right,
                          route=route, rail=rail, floor=tuple(floor)))
    wells.sort(key=lambda b: b["top"])
    _require(2 <= len(wells) <= 3, "need two or three receptacles")
    _require(len({b["route"] for b in wells}) == len(wells), "routing colors must be unique")
    _require(len({(b["left"], b["right"]) for b in wells}) == 1, "wells must share a lateral origin")
    _require(max(b[1] for b in boxes) + 1 < wells[0]["top"], "panels must be on the near side")
    _require(all(a["bottom"]+1 < b["top"] for a, b in zip(wells, wells[1:])), "missing well separator")
    _require(set(map(int, np.unique(source))) - {background} <= {b["route"] for b in wells},
             "unroutable source color")
    source_parts = _components(source, background)
    lookup = {b["route"]: b for b in wells}
    placements, occupied = [], set()
    for part in source_parts:
        well = lookup[part["color"]]
        pixels = _expand(part, stamp, background, ink, well["rail"])
        shift, targets = _landing(pixels, well)
        _require(not occupied.intersection(targets), "independently docked pieces collide")
        occupied.update(targets)
        placements.append(dict(part=part, well=well, pixels=pixels, shift=shift, targets=targets))
    return dict(grid=grid, background=background, ink=ink, accent=accent,
                source=source, source_box=source_box, stamp=stamp, stamp_box=stamp_box,
                wells=wells, parts=source_parts, placements=placements)


def _read(input_grid):
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    _require(all(np.all(side == background) for side in
                 (grid[0], grid[-1], grid[:, 0], grid[:, -1])), "need a visible background rim")
    interpretations = []
    for turn in range(4):
        canonical = np.rot90(grid, turn).copy()
        try:
            info = _canonical_read(canonical, background)
        except (AssertionError, IndexError, ValueError):
            continue
        info["turn"] = turn
        interpretations.append(info)
    _require(len(interpretations) == 1, "input needs one docking direction")
    return interpretations[0]


def _render(info):
    out = info["grid"].copy()
    r0, r1, c0, c1 = info["source_box"]
    out[r0+1:r1, c0+1:c1] = info["background"]
    for placement in info["placements"]:
        for p, value in placement["targets"].items():
            out[p] = value
    return np.rot90(out, -info["turn"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible panels, routing interiors and terrain alone."""
    return _render(_read(input_grid))


def _asymmetric(a):
    transforms = ([np.rot90(a, k) for k in range(1, 4)] +
                  [np.fliplr(np.rot90(a, k)) for k in range(4)])
    return all(not np.array_equal(a, t) for t in transforms)


def _stamp(rng, shape, bg, ink, accent, witness):
    for _ in range(100):
        labels = rng.choice(3, size=shape, p=[.47, .35, .18])
        a = np.array([ink, accent, bg], dtype=np.int8)[labels]
        if set(map(int, np.unique(a))) - {bg} != {ink, accent}:
            continue
        if not all(np.any(side != bg) for side in (a[0], a[-1], a[:, 0], a[:, -1])):
            continue
        if witness and (not _asymmetric(a) or not np.any(a[-1] == bg)):
            continue
        return a
    # Also asymmetric in the 2 by 2 case, and visibly open on the bottom edge.
    a = np.full(shape, ink, dtype=np.int8)
    a[0, -1] = accent
    a[-1, -1] = bg
    return a


def _small_mask(rng, h, w):
    kind = int(rng.integers(5))
    a = np.zeros((h, w), dtype=bool)
    if kind == 0:
        a[:] = True
    elif kind == 1:
        a[-1, :] = True
        a[:, 0] = True
    elif kind == 2:
        a[:, w//2] = True
        a[int(rng.integers(h)), :] = True
    elif kind == 3:
        r = c = 0
        a[0, 0] = True
        while r < h-1 or c < w-1:
            if r < h-1 and (c == w-1 or rng.random() < .5):
                r += 1
            else:
                c += 1
            a[r, c] = True
    else:
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        for _ in range(int(rng.integers(1, h*w+1))-1):
            fringe = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in FOUR
                             if 0 <= r+dr < h and 0 <= c+dc < w and (r+dr, c+dc) not in cells})
            if not fringe:
                break
            cells.add(fringe[int(rng.integers(len(fringe)))])
        for p in cells:
            a[p] = True
    return a


def _source(rng, shape, routes, bg, sh):
    """A variable touching composite plus lateral lanes for independent pieces."""
    mh, mw = shape
    order = [int(v) for v in rng.permutation(routes)]
    a, b = order[:2]
    source = np.full(shape, bg, dtype=np.int8)
    gw = int(rng.integers(3, mw-len(routes)+1))
    gh = int(rng.integers(2, min(mh, 3 if sh >= 3 else 4)+1))
    top = int(rng.integers(mh-gh+1))
    group = np.full((gh, gw), a, dtype=np.int8)
    family = int(rng.integers(3)) if gh >= 3 else 1
    if family == 0:
        group[1:-1, 1:-1] = bg
        inner = _small_mask(rng, gh-2, gw-2)
        group[1:-1, 1:-1][inner] = b
    elif family == 1:
        for row in range(gh):
            divide = int(rng.integers(1, gw))
            group[row, divide:] = b
    else:
        group[:-1] = b
        group[:-1, ::2] = a
    source[top:top+gh, :gw] = group
    # An air column separates the composite from repeated-route lanes.
    start, stop = gw+1, mw-(1 if len(routes) == 3 else 0)
    eh = int(rng.integers(1, min(2, mh-1)+1))
    bottoms = [v for v in range(eh-1, mh) if v != top+gh-1]
    bottom = int(rng.choice(bottoms))
    extra = _small_mask(rng, eh, stop-start)
    # Ensure its actual far extent differs even for a frontier-grown mask.
    rr, cc = np.nonzero(extra)
    extra = extra[rr.min():rr.max()+1, cc.min():cc.max()+1]
    h, w = extra.shape
    source[bottom-h+1:bottom+1, start:start+w][extra] = a
    if len(routes) == 3:
        h = int(rng.integers(1, min(2, mh)+1))
        r = int(rng.integers(mh-h+1))
        source[r:r+h, -1] = order[2]
    return source


def _floor(rng, width, mode=None, cap=None):
    mode = int(rng.integers(4)) if mode is None else mode
    cap = int(rng.integers(2, 4)) if cap is None else cap
    if mode == 0:
        a = rng.integers(1, cap+1, size=width)
    elif mode == 1:
        period = int(rng.integers(2, 6))
        a = 1 + (np.arange(width)//period) % cap
    elif mode == 2:
        a = np.ones(width, dtype=int)
        for _ in range(int(rng.integers(1, 5))):
            lo = int(rng.integers(width-1))
            hi = int(rng.integers(lo+1, width+1))
            a[lo:hi] = int(rng.integers(2, cap+1))
    else:
        a = np.full(width, int(rng.integers(1, cap+1)), dtype=int)
    # The cap denotes an observed step height in the step-growth regime.
    if cap == 3:
        a[int(rng.integers(width))] = 3
    return a.astype(int)


def _witnesses(info):
    independent = any(x["part"]["color"] == y["part"]["color"] and x["shift"] != y["shift"]
                      for i, x in enumerate(info["placements"]) for y in info["placements"][i+1:])
    contact = False
    source, bg = info["source"], info["background"]
    for r, c in np.argwhere(source != bg):
        for dr, dc in FOUR:
            nr, nc = int(r)+dr, int(c)+dc
            if 0 <= nr < source.shape[0] and 0 <= nc < source.shape[1]:
                contact |= int(source[nr, nc]) not in (bg, int(source[r, c]))
    micro, profile = False, False
    for p in info["placements"]:
        well, pixels = p["well"], p["pixels"]
        solid = _expand(p["part"], info["stamp"], bg, info["ink"], well["rail"], solid=True)
        def raw_shift(q):
            return min(int(well["floor"][c])-1-r for r, c in q)
        micro |= raw_shift(solid) != p["shift"]
        r0, r1, c0, c1 = _box(pixels)
        rectangle_shift = min(well["floor"][c] for c in range(c0, c1+1))-1-r1
        profile |= rectangle_shift != p["shift"]
    return dict(independent=independent, contact=bool(contact), micro=micro, profile=profile)


def _attempt_pair(rng, turn, regime, test, witness, previous_shape=None):
    n = 3 if test and regime == "more_routes" else 2 if regime == "more_routes" else int(rng.choice([2, 2, 3]))
    sh = int(rng.choice([2, 2, 3])) if n == 2 else 2
    sw = int(rng.choice([3, 4])) if test and regime == "wider_tiles" else 2 if regime == "wider_tiles" else int(rng.integers(2, 5))
    mw_max = min(11, 26//sw)
    mw = int(rng.integers(6, mw_max+1))
    mh = int(rng.integers(3, 5))
    palette = [int(v) for v in rng.permutation(10)]
    bg, ink, accent = palette[:3]
    routes = palette[3:3+n]
    # The walls may share a color or reuse another route, never their own.
    rail_pool = palette[3+n:]
    rails = []
    for route in routes:
        choices = rail_pool + [r for r in routes if r != route]
        rails.append(int(rng.choice(choices)))
    stamp = _stamp(rng, (sh, sw), bg, ink, accent, witness)
    source = _source(rng, (mh, mw), routes, bg, sh)
    parts = _components(source, bg)
    # Diagonal-only same-color contact is not introduced without evidence.
    if len(parts) != len(_components(source, bg, diagonal=True)):
        return None
    fw = mw*sw
    cap = (3 if test else 2) if regime == "higher_steps" else None
    floors = [_floor(rng, fw, cap=cap) for _ in routes]
    depths = []
    for route, rail, terrain in zip(routes, rails, floors):
        minimum = int(terrain.max())+1
        for part in parts:
            if part["color"] != route:
                continue
            pixels = _expand(part, stamp, bg, ink, rail)
            raw = min(-int(terrain[c])-1-r for r, c in pixels)
            minimum = max(minimum, -raw-min(r for r, c in pixels))
        depths.append(minimum+int(rng.integers(0, 3)))
    gaps = [int(rng.integers(1, 3)) for _ in routes]
    head = max(mh+2, sh+2)+int(rng.integers(1, 3))
    height = head+sum(depths)+sum(gaps)+1
    width = max(fw+4, mw+sw+7)
    if height > 30 or width > 30:
        return None
    grid = np.full((height, width), bg, dtype=np.int8)
    # Source/stamp positions move independently; swapping them changes no rule.
    gap = int(rng.integers(1, min(4, width-mw-sw-6)+1))
    start = int(rng.integers(1, width-(mw+2+sw+2+gap)))
    if rng.random() < .5:
        sc, tc = start, start+mw+2+gap
    else:
        tc, sc = start, start+sw+2+gap
    sr = int(rng.integers(1, head-(mh+2)+1))
    tr = int(rng.integers(1, head-(sh+2)+1))
    for content, row, col in ((source, sr, sc), (stamp, tr, tc)):
        h, w = content.shape
        grid[row:row+h+2, col:col+w+2] = ink
        grid[row+1:row+h+1, col+1:col+w+1] = content
    left = int(rng.integers(1, width-fw-2))
    cursor = head
    wells = []
    for i in map(int, rng.permutation(n)):
        cursor += gaps[i]
        depth, route, rail, terrain = depths[i], routes[i], rails[i], floors[i]
        right = left+fw+1
        grid[cursor:cursor+depth, left:right+1] = route
        grid[cursor:cursor+depth, [left, right]] = rail
        floor = []
        for c, thickness in enumerate(terrain):
            first = cursor+depth-int(thickness)
            grid[first:cursor+depth, left+1+c] = rail
            floor.append(first)
        wells.append(dict(top=cursor, bottom=cursor+depth-1, left=left, right=right,
                          route=route, rail=rail, floor=tuple(floor)))
        cursor += depth
    # Labels are written from construction geometry, not from the input parser.
    out = grid.copy()
    out[sr+1:sr+mh+1, sc+1:sc+mw+1] = bg
    lookup = {b["route"]: b for b in wells}
    occupied = set()
    for part in parts:
        well = lookup[part["color"]]
        pixels = _expand(part, stamp, bg, ink, well["rail"])
        try:
            _, targets = _landing(pixels, well)
        except AssertionError:
            return None
        if occupied.intersection(targets):
            return None
        occupied.update(targets)
        for p, value in targets.items():
            out[p] = value
    pair = {"input": np.rot90(grid, turn).copy(), "output": np.rot90(out, turn).copy()}
    if previous_shape == pair["input"].shape:
        return None
    try:
        info = _read(pair["input"])
    except AssertionError:
        return None
    features = _witnesses(info)
    if witness and not all(features.values()):
        return None
    return pair


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, "num_train", 2, 10)
    ntest = _count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    turn = int(rng.integers(4))
    regime = str(rng.choice(["wider_tiles", "more_routes", "higher_steps"]))
    puzzle, seen = {"train": [], "test": []}, set()
    source_lessons, stamp_lessons = set(), set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            pair = None
            for _ in range(300):
                previous_shape = puzzle[split][-1]["input"].shape if index else None
                candidate = _attempt_pair(rng, turn, regime, split == "test", split == "train" and index < 2,
                                          previous_shape)
                if candidate is None:
                    continue
                if split == "train" and index < 2:
                    info = _read(candidate["input"])
                    source_key = _source_key(info)
                    stamp_key = _stamp_key(info)
                    if source_key in source_lessons or stamp_key in stamp_lessons:
                        continue
                key = (candidate["input"].shape, candidate["input"].tobytes())
                if key not in seen:
                    pair = candidate
                    seen.add(key)
                    if split == "train" and index < 2:
                        source_lessons.add(source_key)
                        stamp_lessons.add(stamp_key)
                    break
            if pair is None:
                raise RuntimeError("bounded construction exhausted")
            puzzle[split].append(pair)
    # The order of teaching examples is not tied to any output-affecting role.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _source_key(info):
    roles = {info["background"]: 0}
    values = tuple(roles.setdefault(int(v), len(roles)) for v in info["source"].ravel())
    return info["source"].shape, values


def _stamp_key(info):
    stamp = info["stamp"]
    labels = np.where(stamp == info["background"], 0, np.where(stamp == info["ink"], 1, 2))
    return stamp.shape, tuple(map(int, labels.ravel()))


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4 and
             len(puzzle["train"]) > len(puzzle["test"]), "illegal counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas dimensions must be preserved")
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _read(inp)
            _require(np.array_equal(out, _render(info)), "incorrect docking output")
            _require(set(p["color"] for p in info["parts"]) == {b["route"] for b in info["wells"]},
                     "generated wells must all be used")
            _require(len(info["parts"]) == len(_components(info["source"], info["background"], diagonal=True)),
                     "undemonstrated diagonal grouping")
            _require(not np.array_equal(inp, out), "empty transformation")
            # Every piece has at least one pixel immediately above its floor.
            for p in info["placements"]:
                well = p["well"]
                _require(any(r+1 == well["floor"][c-well["left"]-1] for r, c in p["targets"]),
                         "piece must make exact contact")
            infos[split].append(info)
    train = infos["train"]
    _require(len({i["turn"] for s in infos.values() for i in s}) == 1, "generated direction is puzzle-wide")
    for name in ("contact", "independent", "micro", "profile"):
        _require(any(_witnesses(i)[name] for i in train), "training lacks "+name+" witness")
    _require(any(_asymmetric(i["stamp"]) for i in train), "training must fix stamp orientation")
    _require(any(np.any(i["stamp"][-1] == i["background"]) for i in train), "missing edge-air witness")
    _require(len({_source_key(i) for i in train}) >= 2, "repeated source geometry")
    _require(len({_stamp_key(i) for i in train}) >= 2, "repeated stamp geometry")
    test = infos["test"]
    def step_height(i):
        return max(b["bottom"]+1-r for b in i["wells"] for r in b["floor"])
    shifted = (min(i["stamp"].shape[1] for i in test) > max(i["stamp"].shape[1] for i in train) or
               min(len(i["wells"]) for i in test) > max(len(i["wells"]) for i in train) or
               min(step_height(i) for i in test) > max(step_height(i) for i in train))
    _require(shifted, "missing visible train/test structural shift")


def measure_pair(pair):
    info = _read(pair["input"])
    bg, ink, accent = info["background"], info["ink"], info["accent"]
    stamp = np.where(info["stamp"] == bg, 0, np.where(info["stamp"] == ink, 1, 2))
    shapes = []
    for p in info["parts"]:
        r0, r1, c0, c1 = _box(p["cells"])
        shapes.append(tuple((r-r0, c-c0) for r, c in p["cells"]))
    return dict(input_shape=tuple(np.asarray(pair["input"]).shape), direction=info["turn"],
                source_shape=info["source"].shape, stamp_shape=stamp.shape,
                stamp_pattern=(stamp.shape, tuple(map(int, stamp.ravel()))),
                source_masks=tuple(shapes), wells=len(info["wells"]), components=len(info["parts"]),
                background=bg, depths=tuple(b["bottom"]-b["top"]+1 for b in info["wells"]),
                terrain=tuple(tuple(b["bottom"]+1-r for r in b["floor"]) for b in info["wells"]),
                shared_rails=len({b["rail"] for b in info["wells"]}) < len(info["wells"]),
                route_rail_reuse=any(b["rail"] in {a["route"] for a in info["wells"]} for b in info["wells"]),
                holes=int(np.count_nonzero(stamp == 0)), **_witnesses(info))


def self_test() -> dict:
    """Bounded structural and semantic audit of 42 seeds, plus edge requests."""
    records, fingerprints = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        _require(all(np.array_equal(a[k], b[k]) for s in puzzle for a, b in zip(puzzle[s], again[s])
                     for k in ("input", "output")), "nondeterminism")
        fingerprints.add(tuple((p["input"].shape, p["input"].tobytes()) for s in puzzle for p in puzzle[s]))
        records.extend(measure_pair(p) for s in puzzle for p in puzzle[s])
    _require(len(fingerprints) == 42, "cross-seed duplicates")
    diversity = {name: len({r[name] for r in records}) for name in
                 ("input_shape", "source_shape", "stamp_shape", "stamp_pattern", "depths", "terrain")}
    for name, minimum in (("input_shape", 40), ("source_shape", 10), ("stamp_shape", 5),
                          ("stamp_pattern", 70), ("depths", 35), ("terrain", 90)):
        _require(diversity[name] >= minimum, "narrow variation: "+name)
    _require({r["direction"] for r in records} == set(range(4)), "missing direction")
    _require({r["wells"] for r in records} == {2, 3}, "missing well count")
    _require(len({s for r in records for s in r["source_masks"]}) >= 50, "narrow coarse component masks")
    _require(any(r["shared_rails"] for r in records) and any(r["route_rail_reuse"] for r in records),
             "narrow palette relationships")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(710+a*10+b, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "count ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Exact hand-worked support: the lower-right air cell exposes a raised step.
    stamp = np.array([[8, 5], [8, 0]], dtype=np.int8)
    part = {"color": 2, "cells": ((0, 0),)}
    well = dict(top=0, bottom=5, left=0, right=3, route=2, rail=3, floor=(4, 3))
    pixels = _expand(part, stamp, 0, 8, 3)
    shift, targets = _landing(pixels, well)
    _require(shift == 2 and targets == {(2, 1): 3, (2, 2): 5, (3, 1): 3}, "support contact regression")
    solid = _expand(part, stamp, 0, 8, 3, solid=True)
    _require(_landing(solid, well)[0] == 1, "microstructure must change landing")
    # A source translation along the motion axis is erased by independent docking.
    moved = {"color": 2, "cells": ((2, 0),)}
    _require(_landing(_expand(moved, stamp, 0, 8, 3), well)[1] == targets, "source depth leaked into result")
    # Reject a changed label. Also verify solving does not mutate the input.
    p = generate_puzzle(991, 2, 1)
    original = p["test"][0]["input"].copy()
    solve(original)
    _require(np.array_equal(original, p["test"][0]["input"]), "solver mutated input")
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return dict(seeds=42, pairs=len(records), diversity=diversity,
                coarse_component_masks=len({s for r in records for s in r["source_masks"]}))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
