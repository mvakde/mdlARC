"""Telescoping cargo carts: color-addressed loads determine packing geometry.

Only standard-library Python and NumPy are used. No parent modules, saved
examples, random state, or training pairs are consulted by solve().
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions must be 1..30")
    _need(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.tobytes()


def _height(cells):
    return max(r for r, _ in cells) - min(r for r, _ in cells) + 1


def _normalize(cells):
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _components(grid, start, stop, diagonal=False):
    unseen = {tuple(map(int, p)) for p in np.argwhere(grid[start:stop] != grid[0, 1])}
    unseen = {(r+start, c) for r, c in unseen}
    result = []
    steps = EIGHT if diagonal else FOUR
    while unseen:
        first = min(unseen)
        color = int(grid[first])
        unseen.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                p = r+dr, c+dc
                if p in unseen and int(grid[p]) == color:
                    unseen.remove(p)
                    stack.append(p)
        result.append({"color": color, "cells": tuple(sorted(cells))})
    return result


def _rows_overlap(a, b):
    return max(a["r"], b["r"]) < min(a["r"]+a["h"], b["r"]+b["h"])


def _overlap(a, b):
    return _rows_overlap(a, b) and max(a["c"], b["c"]) < min(a["c"]+a["w"], b["c"]+b["w"])


def _disjoint(objects):
    return not any(_overlap(a, b) for i, a in enumerate(objects) for b in objects[i+1:])


def _pack(objects, width):
    """Stable boundary recurrence, used by labels and the public solver."""
    result = [dict(o) for o in objects]
    for direction in (0, 1):
        indices = [i for i, o in enumerate(objects) if o["d"] == direction]
        indices.sort(key=lambda i: (objects[i]["c"] if direction == 0 else
                                   -objects[i]["c"]-objects[i]["w"], objects[i]["r"]))
        settled = []
        for i in indices:
            o = result[i]
            blockers = [p for p in settled if _rows_overlap(o, p)]
            o["c"] = (max((p["c"]+p["w"] for p in blockers), default=0) if direction == 0
                      else min((p["c"] for p in blockers), default=width)-o["w"])
            _need(0 <= o["c"] <= width-o["w"], "packing exceeds the canvas")
            settled.append(o)
    _need(_disjoint(result), "opposite packed groups collide")
    return result


def _simulate(objects, width):
    """Validator check: repeatedly slide individual rectangles by one cell."""
    result = [dict(o) for o in objects]
    for direction, delta in ((0, -1), (1, 1)):
        group = [o for o in result if o["d"] == direction]
        for _ in range(width*len(group)+1):
            moved = False
            for o in group:
                candidate = dict(o, c=o["c"]+delta)
                if (0 <= candidate["c"] <= width-o["w"] and
                        not any(p is not o and _overlap(candidate, p) for p in group)):
                    o["c"] += delta
                    moved = True
            if not moved:
                break
        else:
            raise AssertionError("unit-step compaction failed to converge")
    _need(_disjoint(result), "simulated final carriers collide")
    return result


def _edge_first(objects, width):
    """Deliberately competing overtaking rule, used only for witness checks."""
    result = [dict(o) for o in objects]
    for direction in (0, 1):
        indices = [i for i, o in enumerate(objects) if o["d"] == direction]
        indices.sort(key=lambda i: (objects[i]["c"] if direction == 0 else
                                   -objects[i]["c"]-objects[i]["w"], objects[i]["r"]))
        settled = []
        for i in indices:
            o = result[i]
            positions = range(width-o["w"]+1)
            if direction:
                positions = reversed(positions)
            for c in positions:
                candidate = dict(o, c=c)
                if not any(_overlap(candidate, p) for p in settled):
                    o["c"] = c
                    settled.append(o)
                    break
            else:
                raise ValueError("edge-first overflow")
    return result


def _load(carts, components, line):
    """A load preserves absolute staging columns before the cart moves."""
    lookup = {o["route"]: i for i, o in enumerate(carts)}
    grouped = [[] for _ in carts]
    for comp in components:
        _need(comp["color"] in lookup, "source color has no matching cart")
        grouped[lookup[comp["color"]]].append(comp)
    result = []
    for cart, cargo in zip(carts, grouped):
        _need(cargo, "every cart must receive cargo")
        ht = max(_height(comp["cells"]) for comp in cargo)
        r, c, w, bottom = cart["b"]-ht-1, cart["c"], cart["w"], cart["b"]
        _need(r > line, "expanded carrier crosses the loading line")
        patch = np.full((ht+2, w), cart["frame"], dtype=np.int8)
        patch[1:-1, 1:-1] = cart["route"]
        occupied = set()
        for comp in cargo:
            shift = bottom-1-max(y for y, _ in comp["cells"])
            target = {(y+shift, x) for y, x in comp["cells"]}
            _need(all(r < y < bottom and c < x < c+w-1 for y, x in target), "load crosses a rail")
            _need(not occupied & target, "aligned cargo masks collide")
            occupied.update(target)
            for y, x in target:
                patch[y-r, x-c] = cart["frame"]
        result.append(dict(cart, r=r, h=ht+2, patch=patch, targets=tuple(sorted(occupied))))
    _need(_disjoint(result), "expanded starting carriers overlap")
    return result


def _decode_canonical(grid):
    h, w = grid.shape
    _need(h >= 10 and w >= 8, "scene too small")
    bg, left, right = int(grid[0, 1]), int(grid[0, 0]), int(grid[0, -1])
    _need(len({bg, left, right}) == 3, "header needs distinct tabs and background")
    _need(np.all(grid[0, 1:-1] == bg) and np.all(grid[1] == bg) and np.all(grid[-1] == bg), "invalid header/margins")
    lines = [r for r in range(2, h-1) if grid[r, 0] not in (bg, left, right) and np.all(grid[r] == grid[r, 0])]
    _need(len(lines) == 1, "one loading line is required")
    line = lines[0]
    floor_color = int(grid[line, 0])
    _need(line >= 5 and np.all(grid[line-1] == bg), "missing source separator")
    _need(np.all(grid[2:line, 0] == bg) and np.all(grid[2:line, -1] == bg), "sources must be inset")
    covered = grid == bg
    covered[:line+1] = True
    carts = []
    for r in range(line+1, h-1):
        for c in range(w):
            if covered[r, c]:
                continue
            frame = int(grid[r, c])
            _need(frame in (left, right) and r+2 < h-1 and c+3 < w, "unrecognized collapsed cart")
            route = int(grid[r+1, c+1])
            _need(route not in (bg, left, right, floor_color), "invalid cart interior")
            end = c+2
            while end < w and grid[r+1, end] == route:
                end += 1
            _need(end < w and end-c+1 >= 4, "open or too narrow cart")
            patch = grid[r:r+3, c:end+1]
            _need(np.all(patch[0] == frame) and np.all(patch[2] == frame) and
                  patch[1, 0] == frame and patch[1, -1] == frame, "incomplete cart frame")
            _need(not np.any(covered[r:r+3, c:end+1]), "overlapping collapsed carts")
            covered[r:r+3, c:end+1] = True
            carts.append(dict(b=r+2, c=c, w=end-c+1, route=route, frame=frame, d=int(frame == right)))
    _need(carts and np.all(covered), "unparsed foreground below the loading line")
    _need(len({o["route"] for o in carts}) == len(carts), "duplicate routing color")
    comps = _components(grid, 2, line)
    _need(comps, "empty staging field")
    loaded = _load(carts, comps, line)
    return dict(grid=grid, bg=bg, frames=(left, right), line=line, carts=carts,
                components=comps, loaded=loaded)


def _analyze(value):
    grid = _grid(value)
    candidates = []
    for turn in range(4):
        try:
            a = _decode_canonical(np.rot90(grid, turn).copy())
        except ValueError:
            continue
        a["turn"] = turn
        candidates.append(a)
    _need(len(candidates) == 1, "input must have one loading orientation")
    return candidates[0]


def _draw(a, objects):
    out = np.full_like(a["grid"], a["bg"])
    out[0] = a["grid"][0]
    out[a["line"]] = a["grid"][a["line"]]
    for o in objects:
        r, c, h, w = o["r"], o["c"], o["h"], o["w"]
        _need(0 <= c <= out.shape[1]-w and a["line"] < r and r+h < out.shape[0], "carrier out of bounds")
        out[r:r+h, c:c+w] = o["patch"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every command, route, mask and dimension entirely from the input."""
    a = _analyze(input_grid)
    settled = _pack(a["loaded"], a["grid"].shape[1])
    return np.rot90(_draw(a, settled), -a["turn"]).copy()


def _mask(rng, height, width):
    """Connected masks with exact height; the width is a capacity, not a role."""
    kind = int(rng.integers(6))
    if kind == 0:
        cells = {(r, c) for r in range(height) for c in range(width)}
    elif kind == 1 and height >= 3 and width >= 3:
        cells = {(r, c) for r in range(height) for c in range(width)
                 if r in (0, height-1) or c in (0, width-1)}
    elif kind == 2:
        spine = int(rng.integers(width))
        cells = {(r, spine) for r in range(height)}
        for r in range(height):
            if rng.random() < .65:
                end = int(rng.integers(width))
                cells.update((r, c) for c in range(min(spine, end), max(spine, end)+1))
    elif kind == 3:
        c = int(rng.integers(width))
        cells = {(0, c)}
        for r in range(1, height):
            end = int(rng.integers(width))
            cells.update((r-1, x) for x in range(min(c, end), max(c, end)+1))
            c = end
            cells.add((r, c))
    elif kind == 4:
        cells = {(r, 0) for r in range(height)} | {(height-1, c) for c in range(width)}
    else:
        spine = int(rng.integers(width))
        cells = {(r, spine) for r in range(height)}
        goal = int(rng.integers(height, height*width+1))
        while len(cells) < goal:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in FOUR
                               if 0 <= r+dr < height and 0 <= c+dc < width and (r+dr, c+dc) not in cells})
            cells.add(frontier[int(rng.integers(len(frontier)))])
    return _normalize(cells)


def _palette(rng, frames, count):
    choices = [int(c) for c in rng.permutation(10) if c not in frames]
    return choices[0], choices[1], choices[2:count+2]


def _base(shape, stage, bg, floor_color, frames):
    grid = np.full(shape, bg, dtype=np.int8)
    grid[0, 0], grid[0, -1] = frames
    line = stage+3
    grid[line] = floor_color
    return grid, line


def _try_source(grid, cells, color, cart, components, line):
    bg = int(grid[0, 1])
    cellset = set(cells)
    if not all(2 <= r < line-1 and cart["c"] < c < cart["c"]+cart["w"]-1
               and grid[r, c] == bg for r, c in cells):
        return False
    if any((r+dr, c+dc) not in cellset and 2 <= r+dr < line-1 and
           0 <= c+dc < grid.shape[1] and grid[r+dr, c+dc] == color
           for r, c in cells for dr, dc in EIGHT):
        return False
    bottom = max(r for r, _ in cells)
    targets = {(cart["b"]-1-bottom+r, c) for r, c in cells}
    for old in components:
        if old["color"] == color:
            old_bottom = max(r for r, _ in old["cells"])
            if targets & {(cart["b"]-1-old_bottom+r, c) for r, c in old["cells"]}:
                return False
    for p in cells:
        grid[p] = color
    components.append(dict(color=color, cells=tuple(sorted(cells))))
    return True


def _place_source(rng, grid, mask, cart, components, line):
    ht, wd = _height(mask), max(c for _, c in mask)+1
    if wd > cart["w"]-2 or ht > line-3:
        return False
    positions = [(r, c) for r in range(2, line-ht) for c in
                 range(cart["c"]+1, cart["c"]+cart["w"]-wd)]
    for i in rng.permutation(len(positions)):
        r, c = positions[int(i)]
        cells = tuple((r+y, c+x) for y, x in mask)
        if _try_source(grid, cells, cart["route"], cart, components, line):
            return True
    return False


def _finish(grid, line, carts, components, turn):
    loaded = _load(carts, components, line)
    settled = _pack(loaded, grid.shape[1])
    for o in carts:
        b, c, w = o["b"], o["c"], o["w"]
        _need(np.all(grid[b-2:b+1, c:c+w] == grid[0, 1]), "cart construction overlaps")
        grid[b-2:b+1, c:c+w] = o["frame"]
        grid[b-1, c+1:c+w-1] = o["route"]
    a = dict(grid=grid, bg=int(grid[0, 1]), line=line)
    output = _draw(a, settled)
    return {"input": np.rot90(grid, turn).copy(), "output": np.rot90(output, turn).copy()}


def _split_lesson(rng, frames, turn):
    width, stage = int(rng.integers(20, 31)), int(rng.integers(6, 9))
    bg, floor_color, routes = _palette(rng, frames, 2)
    widths = [int(rng.integers(6, min(11, width//2)+1)) for _ in range(2)]
    c0 = int(rng.integers(1, width-max(widths)))
    c1 = max(1, min(width-widths[1]-1, c0+int(rng.integers(-2, 3))))
    columns = [c0, c1]
    lo, hi = max(columns)+1, min(c+w-1 for c, w in zip(columns, widths))
    _need(hi-lo >= 3, "nested lesson has no shared source span")
    fh, fw = int(rng.integers(3, 6)), int(rng.integers(3, min(7, hi-lo)+1))
    top, left = int(rng.integers(2, 3+stage-fh)), int(rng.integers(lo, hi-fw+1))
    outer = {(top+r, left+c) for r in range(fh) for c in range(fw)
             if r in (0, fh-1) or c in (0, fw-1)}
    inside = _mask(rng, fh-2, fw-2)
    inner = {(top+1+r, left+1+c) for r, c in inside}
    heights = [fh+2, _height(inner)+2]
    order = list(map(int, rng.permutation(2)))
    line = stage+3
    gap = int(rng.integers(3))
    required = sum(heights)+gap
    max_margin = 28-line-required
    _need(max_margin >= 0, "split lesson exceeds height budget")
    margin = int(rng.integers(min(3, max_margin)+1))
    cursor = line+1+margin
    bottoms = [0, 0]
    for i in order:
        bottoms[i] = cursor+heights[i]-1
        cursor += heights[i]+gap
    height = max(bottoms)+2+int(rng.integers(min(3, 29-max(bottoms))))
    grid, line = _base((height, width), stage, bg, floor_color, frames)
    directions = list(map(int, rng.permutation(2)))
    carts = [dict(b=bottoms[i], c=columns[i], w=widths[i], route=routes[i],
                  frame=frames[directions[i]], d=directions[i]) for i in range(2)]
    components = []
    _need(_try_source(grid, outer, routes[0], carts[0], components, line), "outer lesson placement failed")
    _need(_try_source(grid, inner, routes[1], carts[1], components, line), "inner lesson placement failed")
    return _finish(grid, line, carts, components, turn)


def _chain_lesson(rng, frames, turn):
    cargo_h = int(rng.integers(3, 6))
    stage = cargo_h+2
    line, expanded_h = stage+3, cargo_h+2
    max_step = min(expanded_h-1, (28-line-expanded_h)//2)
    step = int(rng.integers(max(3, (expanded_h+1)//2), max_step+1))
    zone = expanded_h+2*step
    first_width = int(rng.integers(6, 9))
    # The third cart must fit in the tempting edge gap left by the first.
    # Otherwise edge-first and stable packing can accidentally coincide.
    widths = [first_width, int(rng.integers(6, 9)), int(rng.integers(6, first_width+1))]
    width = int(rng.integers(max(22, sum(widths)+2), 31))
    free = width-sum(widths)
    gaps = rng.multinomial(free-2, [.25, .25, .25, .25])+np.array([1, 0, 0, 1])
    columns, c = [], int(gaps[0])
    for i, wd in enumerate(widths):
        columns.append(c)
        c += wd+int(gaps[i+1])
    direction = int(rng.integers(2))
    if direction:
        columns = [width-c-w for c, w in zip(columns, widths)]
    margin = int(rng.integers(min(2, 28-line-zone)+1))
    first_top = line+1+margin
    rows = [first_top+i*step for i in range(3)]
    if rng.random() < .5:
        rows.reverse()
    bottoms = [r+expanded_h-1 for r in rows]
    height = max(bottoms)+2+int(rng.integers(min(3, 29-max(bottoms))))
    bg, floor_color, routes = _palette(rng, frames, 3)
    grid, line = _base((height, width), stage, bg, floor_color, frames)
    carts = [dict(b=bottoms[i], c=columns[i], w=widths[i], route=routes[i],
                  frame=frames[direction], d=direction) for i in range(3)]
    components = []
    # First cart: two separated source components with unequal source bottoms.
    main = _mask(rng, cargo_h, 2)
    main_top = 2+int(rng.integers(stage-cargo_h+1))
    cells = {(main_top+r, columns[0]+1+c) for r, c in main}
    _need(_try_source(grid, cells, routes[0], carts[0], components, line), "chain main placement failed")
    source_rows = [r for r in range(2, line-1) if r != main_top+cargo_h-1]
    extra_row = int(rng.choice(source_rows))
    extra = {(extra_row, columns[0]+widths[0]-2)}
    _need(_try_source(grid, extra, routes[0], carts[0], components, line), "independent lesson placement failed")
    for i in (1, 2):
        mask = _mask(rng, cargo_h, int(rng.integers(1, widths[i]-1)))
        _need(_place_source(rng, grid, mask, carts[i], components, line), "chain source placement failed")
    return _finish(grid, line, carts, components, turn)


def _free_scene(rng, frames, turn, testing):
    width = int(rng.integers(17, 31))
    stage = int(rng.integers(6, 10 if testing else 9))
    line = stage+3
    height = int(rng.integers(max(23, line+10), 31))
    requested = int(rng.choice([1, 2, 2, 3, 3, 4, 4, 5]))
    bg, floor_color, routes = _palette(rng, frames, requested)
    grid, line = _base((height, width), stage, bg, floor_color, frames)
    grown, carts, components = [], [], []
    bias = float(rng.choice([0, .2, .5, .5, .8, 1]))
    for route in routes:
        accepted = False
        for _ in range(65):
            wd = int(rng.integers(4, min(12, width-2)+1))
            ht = int(rng.integers(1, min(6 if testing else 5, stage, height-line-4)+1))
            d = int(rng.random() >= bias)
            col = int(rng.integers(0, width-wd+1))
            if rng.random() < .25:
                col = int(rng.choice([0, width-wd]))
            bottom = int(rng.integers(line+ht+2, height-1))
            cart = dict(b=bottom, c=col, w=wd, route=route, frame=frames[d], d=d)
            candidate = dict(cart, r=bottom-ht-1, h=ht+2)
            if any(_overlap(candidate, o) for o in grown):
                continue
            try:
                _pack(grown+[candidate], width)
            except ValueError:
                continue
            mask = _mask(rng, ht, int(rng.integers(1, wd-1)))
            if not _place_source(rng, grid, mask, cart, components, line):
                continue
            grown.append(candidate)
            carts.append(cart)
            accepted = True
            # Extra pieces are independent loads and cannot increase this height.
            for _ in range(int(rng.integers(4 if testing else 3))):
                extra = _mask(rng, int(rng.integers(1, ht+1)), int(rng.integers(1, wd-1)))
                _place_source(rng, grid, extra, cart, components, line)
            break
        if not accepted:
            break  # Feasible count, rather than an unbounded density retry.
    _need(carts, "free placement found no cart")
    return _finish(grid, line, carts, components, turn)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0, 2**128-1)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    frames = tuple(int(c) for c in rng.choice(10, 2, replace=False))
    turn, mixed = int(rng.integers(4)), bool(rng.random() < .4)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            rotation = ((turn+index) % 4 if split == "train" else int(rng.integers(4))) if mixed else turn
            builder = _split_lesson if split == "train" and index == 0 else (
                _chain_lesson if split == "train" and index == 1 else None)
            for _ in range(32):
                try:
                    pair = (builder(rng, frames, rotation) if builder else
                            _free_scene(rng, frames, rotation, split == "test"))
                except ValueError:
                    continue
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    objects, comps, width = a["loaded"], a["components"], a["grid"].shape[1]
    settled = _pack(objects, width)
    collapsed = [dict(o, r=o["b"]-2, h=3) for o in objects]
    old = _pack(collapsed, width)
    edge = _edge_first(objects, width)
    points = {p for comp in comps for p in comp["cells"]}
    contact = any((r+dr, c+dc) in points and a["grid"][r, c] != a["grid"][r+dr, c+dc]
                  for r, c in points for dr, dc in FOUR)
    independent = any(x["color"] == y["color"] and max(r for r, _ in x["cells"]) !=
                      max(r for r, _ in y["cells"]) for i, x in enumerate(comps) for y in comps[i+1:])
    return dict(count=len(objects), components=len(comps), contact=contact, independent=independent,
                growth_block=any(x["c"] != y["c"] for x, y in zip(settled, old)),
                stable_witness=any(x["c"] != y["c"] for x, y in zip(settled, edge)),
                moved=any(x["c"] != y["c"] for x, y in zip(objects, settled)),
                heights=tuple(o["h"] for o in objects), widths=tuple(o["w"] for o in objects),
                classes=tuple(sum(o["d"] == d for o in objects) for d in (0, 1)),
                edges=sum(_rows_overlap(x, y) and x["d"] == y["d"] for i, x in enumerate(objects) for y in objects[i+1:]),
                masks=tuple(_normalize(comp["cells"]) for comp in comps),
                stationary=sum(x["c"] == y["c"] for x, y in zip(objects, settled)),
                shape=a["grid"].shape, background=a["bg"], orientation=a["turn"])


def validate_generated_puzzle(puzzle):
    """Check visible input geometry, loaded pixel conservation and stable motion."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle schema"
    assert all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists"
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, conventions, training = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair schema"
            inp, output = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == output.shape, "changed canvas shape"
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            a = _analyze(inp)
            conventions.add(a["frames"])
            four = {frozenset(comp["cells"]) for comp in a["components"]}
            eight = {frozenset(comp["cells"]) for comp in _components(a["grid"], 2, a["line"], True)}
            assert four == eight, "ambiguous diagonal-only component grouping"
            assert sum(len(o["targets"]) for o in a["loaded"]) == sum(len(c["cells"]) for c in a["components"]), "lost load pixels"
            for o in a["loaded"]:
                inner = o["patch"][1:-1, 1:-1]
                assert np.count_nonzero(inner == o["frame"]) == len(o["targets"])
                assert np.any(inner[0] == o["frame"]), "carrier is not minimally tall"
                assert o["r"]+o["h"]-1 == o["b"], "growth moved the bottom anchor"
            settled = _simulate(a["loaded"], a["grid"].shape[1])
            # Check the recurrence and a separate physical simulation agree.
            assert [o["c"] for o in settled] == [o["c"] for o in _pack(a["loaded"], a["grid"].shape[1])]
            expected = np.rot90(_draw(a, settled), -a["turn"])
            assert np.array_equal(expected, output), "incorrect loaded-carrier transformation"
            if split == "train":
                training.append(_features(a))
    assert len(conventions) == 1, "frame/tab roles changed within the puzzle"
    for feature in ("contact", "independent", "growth_block", "stable_witness", "moved"):
        assert any(m[feature] for m in training), "missing training witness: "+feature


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed structural sweep; no files or per-seed corpus are saved."""
    shapes, masks, widths, counts, heights, orientations, palettes, fingerprints = (set() for _ in range(8))
    observed = {name: False for name in ("one_class", "both_classes", "stationary", "growth", "contact", "repeated")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"])
                m = _features(a)
                shapes.add(pair["input"].shape)
                masks.update(m["masks"])
                widths.update(m["widths"])
                counts.add(m["count"])
                heights.update(m["heights"])
                orientations.add(m["orientation"])
                palettes.add(a["frames"])
                observed["one_class"] |= min(m["classes"]) == 0
                observed["both_classes"] |= min(m["classes"]) > 0
                observed["stationary"] |= m["stationary"] > 0
                observed["growth"] |= m["growth_block"]
                observed["contact"] |= m["contact"]
                observed["repeated"] |= m["components"] > m["count"]
                assert np.array_equal(solve(pair["input"]), pair["output"])
    assert len(fingerprints) == 42 and len(palettes) >= 20
    assert len(shapes) >= 40 and len(masks) >= 80 and len(widths) >= 6
    assert {1, 2, 3, 4} <= counts and {3, 4, 5, 6, 7} <= heights
    assert orientations == {0, 1, 2, 3} and all(observed.values())
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(20261023, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    assert _fingerprint(p) == _fingerprint(generate_puzzle(0, 2, 1))
    # Concrete unit-step regression: first/last disjoint, linked through middle.
    chain = [dict(r=r, c=c, h=5, w=6, d=0) for r, c in ((12, 1), (15, 8), (18, 15))]
    assert [o["c"] for o in _pack(chain, 24)] == [0, 6, 12]
    assert [o["c"] for o in _simulate(chain, 24)] == [0, 6, 12]
    assert [o["c"] for o in _edge_first(chain, 24)] == [0, 6, 0]
    # Hand-computed end-to-end regression, independent of construction helpers.
    # The tall bar and deeper singleton route together but align separately.
    # Growing the two initially row-disjoint carts creates a blocking edge.
    inp = np.zeros((20, 18), dtype=np.int8)
    inp[0, 0], inp[0, -1], inp[8] = 6, 2, 8
    inp[2:5, 3], inp[6, 6] = 3, 3
    inp[2:5, 11], inp[4, 12] = 4, 4
    inp[12:15, 2:8], inp[13, 3:7] = 6, 3
    inp[15:18, 10:15], inp[16, 11:14] = 6, 4
    expected = np.zeros_like(inp)
    expected[0], expected[8] = inp[0], inp[8]
    expected[10:15, 0:6], expected[11:14, 1:5] = 6, 3
    expected[11:14, 1], expected[13, 4] = 6, 6
    expected[13:18, 6:11], expected[14:17, 7:10] = 6, 4
    expected[14:17, 7], expected[16, 8] = 6, 6
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(inp, rotation)), np.rot90(expected, rotation))
    # Mutation checks cover a wrong output, routing-color decoding, and input purity.
    before = p["test"][0]["input"].copy()
    solve(before)
    assert np.array_equal(before, p["test"][0]["input"])
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError("corrupted label accepted")
    malformed = inp.copy()
    malformed[16, 11:14] = 3  # Two carts now claim the same visible route.
    try:
        solve(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate routing color accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "masks": len(masks),
            "cart_widths": sorted(widths), "cart_heights": sorted(heights),
            "cart_counts": sorted(counts), "observed": observed}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
