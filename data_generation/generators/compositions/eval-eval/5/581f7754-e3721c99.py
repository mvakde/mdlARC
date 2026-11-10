"""Cavity docking: topology chooses an edge; that edge chooses a hole handle.

Only NumPy and the standard library are used. Decoding and validation use pixels,
never construction records, random state, demonstrations, or saved parent data.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("left", "top", "right", "bottom")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _holes(cells):
    """Enclosed four-connected complement components, in the same coordinates."""
    r0, r1, c0, c1 = _bounds(cells)
    mask = np.ones((r1-r0+3, c1-c0+3), dtype=bool)
    for r, c in cells:
        mask[r-r0+1, c-c0+1] = False
    result = []
    for empty in _components(mask):
        if any(r in (0, mask.shape[0]-1) or c in (0, mask.shape[1]-1) for r, c in empty):
            continue
        result.append({(r+r0-1, c+c0-1) for r, c in empty})
    return result


def _center(hole):
    r0, r1, c0, c1 = _bounds(hole)
    _require((r1-r0) % 2 == 0 and (c1-c0) % 2 == 0, "hole bounding spans must be odd")
    point = (r0+r1)//2, (c0+c1)//2
    _require(point in hole, "hole center must be a background cell")
    return point


def _socket(holes, side):
    _require(bool(holes), "an addressed carrier needs a hole")
    index = {"top": 0, "bottom": 1, "left": 2, "right": 3}[side]
    distances = [_bounds(h)[index] for h in holes]
    nearest = (max if side in ("bottom", "right") else min)(distances)
    candidates = [h for h, d in zip(holes, distances) if d == nearest]
    _require(len(candidates) == 1, "nearest cavity is tied")
    return _center(candidates[0]), candidates[0]


def _decode(value):
    g = _grid(value)
    h, w = g.shape
    colors, frequencies = np.unique(g, return_counts=True)
    bg = int(colors[np.argmax(frequencies)])
    _require(int(frequencies.max()) > g.size/2, "background must occupy more than half the grid")
    edge_colors = set(map(int, np.concatenate((g[0], g[-1], g[:, 0], g[:, -1])))) - {bg}
    neutral = set(map(int, colors)) - edge_colors - {bg}
    _require(len(neutral) == 1 and bool(edge_colors), "need one interior-only query color and boundary ports")
    query = neutral.pop()
    ports, objects, port_colors = {}, [], set()
    for cells in _components(g != bg):
        local_colors = {int(g[p]) for p in cells}
        _require(len(local_colors) == 1, "foreground components must be monochrome and separate")
        color = local_colors.pop()
        holes = _holes(cells)
        _require(all(int(g[p]) == bg for hole in holes for p in hole), "foreign pixels inside a cavity")
        if color == query:
            _require(all(0 < r < h-1 and 0 < c < w-1 for r, c in cells), "queries cannot touch the boundary")
            for hole in holes:
                _center(hole)
            objects.append({"cells": cells, "holes": holes, "count": len(holes)})
            continue
        boundary = [(r, c) for r, c in cells if r in (0, h-1) or c in (0, w-1)]
        _require(len(boundary) == 1, "each port must have exactly one boundary pin")
        r, c = boundary[0]
        sides = [side for side, hit in (("top", r == 0), ("bottom", r == h-1),
                                        ("left", c == 0), ("right", c == w-1)) if hit]
        _require(len(sides) == 1, "corner pins are ambiguous")
        _require(color not in port_colors and len(holes) not in ports, "port colors and classes must be unique")
        _require(len(holes) > 0, "port prototypes must have at least one cavity")
        side = sides[0]
        axis = 0 if side in ("left", "right") else 1
        ports[len(holes)] = {"color": color, "cells": cells, "pin": (r, c),
                             "side": side, "axis": axis, "guide": (r, c)[axis]}
        port_colors.add(color)
    _require(len(objects) >= 2 and len(port_colors) == len(edge_colors), "incomplete port/query field")
    for obj in objects:
        if obj["count"] in ports:
            port = ports[obj["count"]]
            center, selected = _socket(obj["holes"], port["side"])
            delta = [0, 0]
            delta[port["axis"]] = port["guide"] - center[port["axis"]]
            obj.update(port=port, center=center, selected=selected, delta=tuple(delta))
    return {"grid": g, "background": bg, "query": query, "ports": ports, "objects": objects}


def _render(parsed, handle="socket", move=True):
    g, bg = parsed["grid"], parsed["background"]
    result = np.full_like(g, bg)
    occupied = set()
    for port in parsed["ports"].values():
        for p in port["cells"]:
            result[p] = port["color"]
            occupied.add(p)
    for obj in parsed["objects"]:
        if "port" not in obj:
            continue
        port = obj["port"]
        delta = list(obj["delta"])
        if not move:
            delta = [0, 0]
        elif handle != "socket":
            if handle == "box":
                r0, r1, c0, c1 = _bounds(obj["cells"])
                center = (r0+r1)//2, (c0+c1)//2
            elif handle == "largest":
                # Diagnostic alternative only; tie uses a deterministic pixel order.
                hole = max(obj["holes"], key=lambda z: (len(z), tuple(sorted(z))))
                center = _center(hole)
            else:
                raise ValueError("unknown diagnostic handle")
            delta[port["axis"]] = port["guide"] - center[port["axis"]]
        for r, c in obj["cells"]:
            p = r+delta[0], c+delta[1]
            if 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]:
                _require(p not in occupied, "projected foreground footprints overlap")
                result[p] = port["color"]
                occupied.add(p)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only decoding; no training data or generation state is consulted."""
    return _render(_decode(input_grid))


def _mask_cells(mask):
    return {tuple(map(int, p)) for p in np.argwhere(mask)}


def _trim(mask):
    r, c = np.nonzero(mask)
    return mask[r.min():r.max()+1, c.min():c.max()+1].copy()


def _topology(mask, count):
    return len(_components(mask)) == 1 and len(_holes(_mask_cells(mask))) == count


def _diagonal(count):
    mask = np.ones((2*count+1, 2*count+1), dtype=bool)
    for i in range(count):
        mask[2*i+1, 2*i+1] = False
    return mask


def _witness(rng):
    """Same three-hole carrier in two opposing-edge teaching examples."""
    mask = _diagonal(3)
    for dr, dc in DIRS:
        mask[3+dr, 3+dc] = False
    # Exterior corner edits leave a large central hole and two small end holes.
    for r, c in ((0, 6), (6, 0), (0, 0), (6, 6)):
        if rng.random() < .7:
            trial = mask.copy()
            trial[r, c] = False
            if _topology(trial, 3):
                mask = trial
    return mask


def _center_witness():
    mask = np.ones((5, 5), bool)
    for p in ((2, 2), (1, 2), (3, 2), (2, 1), (2, 3)):
        mask[p] = False
    return mask


def _query_mask(rng, count, side=None, richer=False):
    if count == 0:
        h, w = int(rng.integers(2, 6)), int(rng.integers(2, 7))
        if rng.random() < .5:
            mask = np.zeros((h, w), dtype=bool)
            mask[:, 0] = mask[-1, :] = mask[:, -1] = True
        else:
            mask = np.zeros((h, w), dtype=bool)
            mask[h//2, w//2] = True
            for _ in range(h*w//2):
                neighbors = np.zeros_like(mask, dtype=int)
                neighbors[1:] += mask[:-1]; neighbors[:-1] += mask[1:]
                neighbors[:, 1:] += mask[:, :-1]; neighbors[:, :-1] += mask[:, 1:]
                choices = np.argwhere(~mask & (neighbors == 1))
                if not len(choices):
                    break
                mask[tuple(choices[int(rng.integers(len(choices)))])] = True
        # Four-connected background can form a diagonal enclosure even when
        # foreground growth added only one orthogonal neighbor at each step.
        for hole in _holes(_mask_cells(mask)):
            for point in hole:
                mask[point] = True
        return _trim(np.rot90(mask, int(rng.integers(4))))
    # Odd-span cavity stencils include rectangles, crosses and asymmetric corners.
    cavities = [np.ones((1, 1), bool), np.ones((1, 3), bool), np.ones((3, 1), bool),
                np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool),
                np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], bool), np.ones((3, 3), bool)]
    for _ in range(24):
        limit = 10 if richer else 9
        h, w = int(rng.integers(5, limit+1)), int(rng.integers(5, limit+1))
        mask = np.ones((h, w), bool)
        reserved = np.zeros((h, w), bool)
        ok = True
        for i in range(count):
            cavity = cavities[int(rng.integers(len(cavities)))]
            cavity = np.rot90(cavity, int(rng.integers(4)))
            ch, cw = cavity.shape
            choices = [(r, c) for r in range(1, h-ch) for c in range(1, w-cw)
                       if not reserved[r:r+ch, c:c+cw].any()]
            if not choices:
                ok = False
                break
            r, c = choices[int(rng.integers(len(choices)))]
            mask[r:r+ch, c:c+cw][cavity] = False
            reserved[r-1:r+ch+1, c-1:c+cw+1] = True
        if not ok or not _topology(mask, count):
            continue
        # Carve open bays and corners only if the class and cavity geometry survive.
        for _ in range(5):
            trial = mask.copy()
            side_edit = int(rng.integers(4))
            if side_edit < 2:
                r = 0 if side_edit == 0 else h-1
                trial[r, int(rng.integers(w))] = False
            else:
                c = 0 if side_edit == 2 else w-1
                trial[int(rng.integers(h)), c] = False
            if _topology(trial, count):
                mask = trial
        holes = _holes(_mask_cells(mask))
        try:
            for hole in holes:
                _center(hole)
            if side is not None:
                _socket(holes, side)
        except ValueError:
            continue
        return _trim(mask)
    # Diagonal holes have distinct extreme coordinates on all four sides.
    return _diagonal(count)


def _prototype(rng, count):
    rows = count if rng.random() < .65 else (1 if count == 1 else 2)
    cols = (count+rows-1)//rows
    mask = np.ones((2*rows+1, 2*cols+1), bool)
    for i in range(count):
        mask[2*(i//cols)+1, 2*(i%cols)+1] = False
    if max(mask.shape) <= 5 and rng.random() < .35:
        mask = np.repeat(mask, 2, axis=int(rng.integers(2)))
    # Canonical left port: prefer a long tangential rather than inward extent.
    if mask.shape[1] > mask.shape[0]:
        mask = mask.T.copy()
    return mask


def _clipping_mask(count, port, shape):
    """A hollow lobe with a thin tail crossing a perpendicular canvas edge."""
    body = _diagonal(count)
    center = _socket(_holes(_mask_cells(body)), port["side"])[0]
    axis, guide = port["axis"], port["guide"]
    span, size = body.shape[axis], shape[axis]
    before = guide <= size-1-guide
    length = max(1, guide-center[axis]+2 if before else
                 size+1-guide-span+center[axis])
    extent = list(body.shape)
    extent[axis] += length
    mask = np.zeros(extent, bool)
    if axis == 0:
        start = length if before else 0
        mask[start:start+span, :] = body
        if before:
            mask[:length, 0] = True
        else:
            mask[span:, 0] = True
    else:
        start = length if before else 0
        mask[:, start:start+span] = body
        if before:
            mask[0, :length] = True
        else:
            mask[0, span:] = True
    return mask


def _reserve(free, r, c, h, w, gap=1):
    free[max(0, r-gap):min(free.shape[0], r+h+gap),
         max(0, c-gap):min(free.shape[1], c+w+gap)] = False


def _locations(free, h, w):
    if h > free.shape[0] or w > free.shape[1]:
        return []
    sums = np.pad((~free).astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    total = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
    return np.argwhere(total == 0)


def _place_port(rng, inp, out, source_free, target_free, count, color, side):
    body = _prototype(rng, count)
    stem = int(rng.integers(2, 5))
    hook = int(rng.integers(body.shape[0]))
    patch = np.zeros((body.shape[0], stem+body.shape[1]), bool)
    patch[:, stem:] = body
    patch[hook, :stem] = True
    if side == "right":
        patch = patch[:, ::-1]
    elif side == "top":
        patch = patch.T
    elif side == "bottom":
        patch = patch.T[::-1]
    ph, pw = patch.shape
    if side in ("left", "right"):
        locations = [(r, 0 if side == "left" else inp.shape[1]-pw)
                     for r in range(1, inp.shape[0]-ph)]
    else:
        locations = [(0 if side == "top" else inp.shape[0]-ph, c)
                     for c in range(1, inp.shape[1]-pw)]
    rng.shuffle(locations)
    for r, c in locations:
        if source_free[r:r+ph, c:c+pw].all():
            inp[r:r+ph, c:c+pw][patch] = color
            out[r:r+ph, c:c+pw][patch] = color
            _reserve(source_free, r, c, ph, pw)
            _reserve(target_free, r, c, ph, pw)
            pts = {(r+a, c+b) for a, b in _mask_cells(patch)}
            pin = next(p for p in pts if p[0] in (0, inp.shape[0]-1) or p[1] in (0, inp.shape[1]-1))
            axis = 0 if side in ("left", "right") else 1
            return {"color": color, "side": side, "axis": axis, "guide": pin[axis]}
    return None


def _place_query(rng, inp, out, source_free, target_free, mask, count, ports, query,
                 must_move=False, allow_clip=False, require_aligned=False):
    mh, mw = mask.shape
    pts = sorted(_mask_cells(mask))
    _require(len(_holes(set(pts))) == count, "query constructor changed the class")
    port = ports.get(count)
    center = _socket(_holes(set(pts)), port["side"])[0] if port else None
    choices = _locations(source_free, mh, mw)
    if not len(choices):
        return False
    for j in rng.permutation(len(choices)):
        r, c = map(int, choices[j])
        delta = [0, 0]
        if port:
            axis = port["axis"]
            delta[axis] = port["guide"] - ((r, c)[axis]+center[axis])
            if must_move and delta == [0, 0]:
                continue
            if require_aligned and delta != [0, 0]:
                continue
        tr, tc = r+delta[0], c+delta[1]
        target = [(tr+a, tc+b) for a, b in pts]
        inside = [(a, b) for a, b in target if 0 <= a < inp.shape[0] and 0 <= b < inp.shape[1]]
        if port:
            if len(inside) != len(pts) and not allow_clip:
                continue
            # Reserve destination boxes, including cavity interiors, to avoid nesting.
            if not target_free[max(0, tr):min(inp.shape[0], tr+mh),
                               max(0, tc):min(inp.shape[1], tc+mw)].all():
                continue
            if not inside:
                continue
        inp[r:r+mh, c:c+mw][mask] = query
        _reserve(source_free, r, c, mh, mw)
        if port:
            for p in inside:
                out[p] = port["color"]
            _reserve(target_free, tr, tc, mh, mw)
        return True
    return False


def _scene(rng, plan, palette, witness):
    if rng.random() < .18:
        h, w = int(rng.integers(17, 21)), int(rng.integers(26, 31))
        if rng.random() < .5:
            h, w = w, h
    else:
        h, w = map(int, rng.integers(21 if plan["test"] else 20, 31, size=2))
    bg, query = palette[:2]
    inp = np.full((h, w), bg, np.int8)
    out = inp.copy()
    source_free = np.ones((h, w), bool)
    target_free = source_free.copy()
    ports = {}
    for count, side, color in zip(plan["keys"], plan["sides"], palette[2:]):
        port = _place_port(rng, inp, out, source_free, target_free, count, color, side)
        if port is None:
            return None
        ports[count] = port
    source_free[0] = source_free[-1] = False
    source_free[:, 0] = source_free[:, -1] = False
    mandatory = list(plan["keys"][:2])
    if plan["teach"] or rng.random() < .65:
        mandatory.append(0)
    if plan["teach"]:
        # A hollow unlisted object rules out merely erasing solid shapes.
        mandatory.append(2 if 1 in ports else 1)
    # The first two examples reuse an asymmetric three-hole object, with opposite
    # port sides. This contrasts edge-facing selection with a largest-hole handle.
    requests = [(k, True, "plain") for k in mandatory]
    if not plan["teach"] and rng.random() < .4:
        k = next(k for k in plan["keys"] if k in (1, 2))
        requests.append((k, False, "aligned"))
    if not plan["teach"] and rng.random() < .55:
        k = next(k for k in plan["keys"] if k in (1, 2))
        requests.append((k, False, "clip"))
    absent = [k for k in range(6) if k not in ports]
    for _ in range(int(rng.integers(0, 5 if plan["test"] else 3 if plan["teach"] else 4))):
        k = int(rng.choice(list(ports) if rng.random() < .65 else absent))
        requests.append((k, False, "plain"))
    for k, required, mode in requests:
        side = ports[k]["side"] if k in ports else None
        if mode == "clip":
            mask = _clipping_mask(k, ports[k], inp.shape)
        elif mode == "aligned":
            mask = _diagonal(k)
        elif plan["teach"] and k == 3:
            mask = witness.copy()
        elif plan["teach"] and k == 1 and k in ports:
            mask = _center_witness()
        else:
            mask = _query_mask(rng, k, side, plan["test"])
        placed = _place_query(rng, inp, out, source_free, target_free, mask, k, ports, query,
                              must_move=required and k in ports,
                              allow_clip=mode == "clip" or not required and rng.random() < .4,
                              require_aligned=mode == "aligned")
        teaching_handle = plan["teach"] and (k == 3 or k == 1 and k in ports)
        if not placed and required and not teaching_handle:
            small = np.array([[1, 1], [1, 0]], bool) if k == 0 else _diagonal(k)
            placed = _place_query(rng, inp, out, source_free, target_free, small, k, ports, query,
                                  must_move=k in ports)
        if not placed and required:
            return None
    if np.mean(inp == bg) <= .5:
        return None
    return {"input": inp, "output": out}


def _fallback(plan, palette, witness, ordinal):
    """Finite guaranteed layout: two ports, two addressed and two erased carriers.

    Fourteen singleton positions make the input unique for every legal call.
    The layout is rotated to retain the plan's primary edge orientation.
    """
    bg, query, a, b = palette[:4]
    high = 4 if plan["test"] else 3
    low = next(k for k in plan["keys"] if k in (1, 2))
    first_side = plan["sides"][plan["keys"].index(high)]
    right = first_side in ("right", "bottom")
    turns = 3 if first_side in ("top", "bottom") else 0
    inp = np.full((30, 30), bg, np.int8)
    # Primary port: a vertical tally with a horizontal stem to row 10.
    proto = np.ones((2*high+1, 3), bool)
    proto[1:2*high:2, 1] = False
    start = 10-high
    c = 25 if right else 2
    inp[start:start+proto.shape[0], c:c+3][proto] = a
    if right:
        inp[10, 28:30] = a
    else:
        inp[10, :2] = a
    # Secondary port on the top, with pin column 24.
    glyph = np.ones((3, 2*low+1), bool)
    glyph[1, 1:2*low:2] = False
    left = 24-low
    inp[2:5, left:left+glyph.shape[1]][glyph] = b
    inp[:2, 24] = b
    out = inp.copy()
    ports = {high: {"side": "right" if right else "left", "axis": 0, "guide": 10, "color": a},
             low: {"side": "top", "axis": 1, "guide": 24, "color": b}}
    low_mask = _center_witness() if high == 3 and low == 1 else _diagonal(low)
    for k, mask, origin in ((high, _diagonal(4) if high == 4 else witness, (18, 11)),
                            (low, low_mask, (18, 23))):
        r, c = origin
        inp[r:r+mask.shape[0], c:c+mask.shape[1]][mask] = query
        port = ports[k]
        center = _socket(_holes(_mask_cells(mask)), port["side"])[0]
        delta = [0, 0]
        axis = port["axis"]
        delta[axis] = port["guide"] - ((r, c)[axis]+center[axis])
        for rr, cc in _mask_cells(mask):
            out[r+rr+delta[0], c+cc+delta[1]] = port["color"]
    inp[1+ordinal, 7] = query
    unlisted = _diagonal(2 if low == 1 else 1)
    inp[1:1+unlisted.shape[0], 13:13+unlisted.shape[1]][unlisted] = query
    return {"input": np.rot90(inp, turns).copy(), "output": np.rot90(out, turns).copy()}


def _counts(rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train is None:
        num_train = int(rng.integers((num_test or 1)+1, 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train-1)+1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _signature(grid):
    g = np.asarray(grid)
    return g.shape, g.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    a, b = _counts(rng, num_train, num_test)
    stable = bool(rng.random() < .25)
    base_palette = tuple(map(int, rng.permutation(10)))
    base_side = int(rng.integers(4))
    witness = _witness(rng)
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(a+b):
        test, teach = i >= a, i < 2
        if teach:
            keys = [3, 1 if i == 0 else 2]
            sides = [SIDES[(base_side+2*i) % 4], SIDES[(base_side+1+2*i) % 4]]
        else:
            keys = [4 if test else 3, int(rng.integers(1, 3))]
            if rng.random() < .4:
                keys.append(int(rng.choice([k for k in (1, 2, 3) if k not in keys])))
            sides = [SIDES[int(rng.integers(4))] for k in keys]
        # Class four is held out of training but visibly specified by the test port.
        plan = {"keys": keys, "sides": sides, "test": test, "teach": teach}
        palette = base_palette if stable else tuple(map(int, rng.permutation(10)))
        for attempt in range(40):
            pair = _scene(rng, plan, palette, witness)
            if pair is None or _signature(pair["input"]) in seen:
                continue
            # Construction writes destinations independently; pixel decoding checks it.
            _require(np.array_equal(solve(pair["input"]), pair["output"]), "construction/decoding disagreement")
            break
        else:
            pair = _fallback(plan, palette, witness, i)
        seen.add(_signature(pair["input"]))
        puzzle["test" if test else "train"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative_differs(parsed, output, handle="socket", move=True):
    try:
        return not np.array_equal(_render(parsed, handle, move), output)
    except ValueError:
        return True  # This alternative produces a forbidden collision.


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal example counts")
    seen, witnesses = set(), Counter()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions changed")
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            p = _decode(inp)
            expected = _render(p)
            _require(np.array_equal(out, expected), "wrong docking output")
            _require(not np.array_equal(inp, out), "identity scene")
            for port in p["ports"].values():
                _require(all(out[q] == inp[q] for q in port["cells"]), "fixed port changed")
            if split == "train":
                for obj in p["objects"]:
                    if "port" not in obj:
                        witnesses["erase"] += 1
                        witnesses["hollow_erase"] += obj["count"] > 0
                    else:
                        witnesses["class", obj["count"]] += 1
                        delta = obj["delta"]
                        full = all(0 <= r+delta[0] < inp.shape[0] and 0 <= c+delta[1] < inp.shape[1]
                                   for r, c in obj["cells"])
                        witnesses["full_move"] += bool(full and delta != (0, 0))
                        sr0, sr1, sc0, sc1 = _bounds(obj["selected"])
                        span = (sr1-sr0+1, sc1-sc0+1)[obj["port"]["axis"]]
                        witnesses["wide_socket"] += bool(full and span >= 3 and delta != (0, 0))
                witnesses["ignore_A"] += _alternative_differs(p, out, move=False)
                # Keeping all query cells (ignoring B's topology filter) cannot
                # reproduce an erased zero-hole component, which has no socket.
                witnesses["ignore_B"] += any(o["count"] == 0 for o in p["objects"])
                witnesses["box_handle"] += _alternative_differs(p, out, "box")
                witnesses["largest_handle"] += _alternative_differs(p, out, "largest")
    _require(len([k for k in witnesses if isinstance(k, tuple)]) >= 2, "need two retained training classes")
    for name in ("erase", "hollow_erase", "full_move", "wide_socket", "ignore_A", "ignore_B", "box_handle", "largest_handle"):
        _require(witnesses[name] > 0, f"missing training witness: {name}")
    # Do not enforce sampling templates, literal colors, sides, or held-out counts:
    # the rule and input parser support legal scenes beyond the sampler's shift.


def diversity_features(pair):
    p = _decode(pair["input"])
    objects = p["objects"]
    return {"shape": list(p["grid"].shape), "background": p["background"], "query": p["query"],
            "keys": sorted(p["ports"]), "sides": sorted(v["side"] for v in p["ports"].values()),
            "port_guides": sorted((v["axis"], v["guide"]) for v in p["ports"].values()),
            "objects": len(objects), "counts": sorted(o["count"] for o in objects),
            "sizes": sorted(len(o["cells"]) for o in objects),
            "cavity_areas": sorted(len(h) for o in objects for h in o["holes"]),
            "irregular_holes": sum(len(h) != (_bounds(h)[1]-_bounds(h)[0]+1)*(_bounds(h)[3]-_bounds(h)[2]+1)
                                   for o in objects for h in o["holes"]),
            "erased": sum("port" not in o for o in objects),
            "unused_ports": sum(not any(o["count"] == k for o in objects) for k in p["ports"]),
            "clipped": sum(any(not (0 <= r+o["delta"][0] < p["grid"].shape[0] and
                                     0 <= c+o["delta"][1] < p["grid"].shape[1]) for r, c in o["cells"])
                           for o in objects if "port" in o),
            "aligned": sum(o["delta"] == (0, 0) for o in objects if "port" in o),
            "deltas": [list(o["delta"]) for o in objects if "port" in o]}


def self_test():
    """Bounded suite over 42 seeds, topology fixtures and metamorphic checks."""
    diagonal_escape = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
    assert len(_holes(diagonal_escape)) == 1
    assert not _holes(diagonal_escape - {(0, 1)})
    assert not _holes({(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)})
    w = _witness(np.random.default_rng(0))
    holes = _holes(_mask_cells(w))
    assert [_socket(holes, s)[0] for s in SIDES] == [(1, 1), (1, 1), (5, 5), (5, 5)]
    assert sorted(map(len, holes)) == [1, 1, 5]
    try:
        _socket([{(1, 1)}, {(3, 1)}], "left")
    except ValueError:
        pass
    else:
        raise AssertionError("tied sockets accepted")
    features, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(pair["input"]) for s in puzzle for pair in puzzle[s]))
        local = [diversity_features(pair) for s in puzzle for pair in puzzle[s]]
        features.extend(local)
        assert 4 not in set(k for f in local[:4] for k in f["keys"])
        assert 4 in local[-1]["keys"]
        assert len({tuple(f["shape"]) for f in local}) > 1
    assert len(signatures) == 42
    assert len({tuple(f["shape"]) for f in features}) >= 35
    assert {s for f in features for s in f["sides"]} == set(SIDES)
    assert {f["background"] for f in features} == set(range(10))
    assert {f["query"] for f in features} == set(range(10))
    assert len({tuple(f["sizes"]) for f in features}) >= 100
    assert {2, 3} <= {len(f["keys"]) for f in features}
    assert len({f["objects"] for f in features}) >= 4
    assert sum(f["irregular_holes"] for f in features) >= 50
    assert any(f["clipped"] for f in features)
    assert any(f["unused_ports"] for f in features)
    assert any(f["aligned"] for f in features)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(731, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(19, 2, 1)
    q = generate_puzzle(19, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    rng = np.random.default_rng(97)
    for turns in (1, 2, 3):
        rotated = {s: [{k: np.rot90(v, turns).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        validate_generated_puzzle(rotated)
    recolored = {"train": [], "test": []}
    for s in p:
        for pair in p[s]:
            colors = rng.permutation(10)
            recolored[s].append({k: colors[v].astype(np.int8) for k, v in pair.items()})
    validate_generated_puzzle(recolored)
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    for side in SIDES:
        for high in (3, 4):
            for low in (1, 2):
                plan = {"test": high == 4, "keys": [high, low], "sides": [side, "top"]}
                for ordinal in (0, 13):
                    pair = _fallback(plan, tuple(range(10)), w, ordinal)
                    assert np.array_equal(solve(pair["input"]), pair["output"])
    return {"seeds_checked": 42, "pairs_checked": len(features),
            "distinct_shapes": len({tuple(f["shape"]) for f in features}),
            "distinct_size_populations": len({tuple(f["sizes"]) for f in features}),
            "clipped_pairs": sum(f["clipped"] > 0 for f in features),
            "irregular_holes": sum(f["irregular_holes"] for f in features)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
