"""Folded message exchange: visible dictionaries and symmetry-paired addresses.

Only standard-library modules and NumPy are used. No parent code or data is
imported. The reader derives all roles, pairings, and templates from input cells.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             "grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = _D8 if diagonal else _D4
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        result.append(sorted(cells))
    return result


def _bounds(cells):
    rows, cols = zip(*cells)
    return min(rows), min(cols), max(rows) + 1, max(cols) + 1


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _nonsolid(template):
    r0, c0, r1, c1 = _bounds([(r, c) for r, c, _ in template])
    return (r1 - r0) * (c1 - c0) > len(template)


def _pick(rng, choices):
    return choices[int(rng.integers(len(choices)))]


def _marker(cells):
    if len(cells) == 1:
        return {"kind": "P", "origin": cells[0], "cells": cells}
    r0, c0, r1, c1 = _bounds(cells)
    if r1 - r0 == 1 and len(cells) == c1 - c0:
        return {"kind": "H", "origin": cells[0], "cells": cells}
    if c1 - c0 == 1 and len(cells) == r1 - r0:
        return {"kind": "V", "origin": cells[0], "cells": cells}
    raise ValueError("marker must be a straight segment or a singleton")


def _transform(position, marker):
    r, c = position
    ar, ac = marker["origin"]
    kind = marker["kind"]
    return (2 * ar - r if kind in ("H", "P") else r,
            2 * ac - c if kind in ("V", "P") else c)


def _flip_offset(offset, kind):
    r, c = offset
    return (-r if kind in ("H", "P") else r,
            -c if kind in ("V", "P") else c)


def _read(input_grid):
    """Input-only recognition. Returned coordinates are local to the board crop."""
    grid = _grid(input_grid)
    frames = []
    for color in np.unique(grid):
        for cells in _components(grid == color, diagonal=False):
            r0, c0, r1, c1 = _bounds(cells)
            h, w = r1 - r0, c1 - c0
            if h < 7 or w < 7 or len(cells) != 2 * h + 2 * w - 4:
                continue
            if not all(r in (r0, r1 - 1) or c in (c0, c1 - 1) for r, c in cells):
                continue
            board = grid[r0:r1, c0:c1]
            rim = np.concatenate((board[1, 1:-1], board[-2, 1:-1],
                                  board[1:-1, 1], board[1:-1, -2]))
            if len(np.unique(rim)) == 1 and rim[0] != color:
                frames.append(((r0, c0, r1, c1), int(color), int(rim[0])))
    _require(len(frames) == 1, "expected one isolated rectangular workboard frame")
    box, marker_color, fill = frames[0]
    r0, c0, r1, c1 = box
    exterior = np.ones(grid.shape, dtype=bool)
    exterior[r0:r1, c0:c1] = False
    _require(exterior.any(), "missing exterior dictionary")
    colors, counts = np.unique(grid[exterior], return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1 and counts.max() > exterior.sum() / 2,
             "exterior fill must be the unique majority")
    background = int(colors[counts.argmax()])
    _require(background != marker_color, "exterior fill cannot be the frame color")
    templates, repetitions = {}, Counter()
    for cells in _components(exterior & (grid != background)):
        _require(len(cells) >= 3, "dictionary motifs must have at least three cells")
        a, b, c, d = _bounds(cells)
        _require(c - a <= 5 and d - b <= 5, "dictionary motif exceeds the family envelope")
        colors, counts = np.unique([grid[q] for q in cells], return_counts=True)
        singleton = colors[counts == 1]
        _require(len(singleton) == 1, "dictionary motif needs one unique singleton key color")
        key = int(singleton[0])
        _require(key not in (background, fill, marker_color), "key collides with a structural role")
        anchor = next(q for q in cells if grid[q] == key)
        sig = tuple(sorted((r - anchor[0], c - anchor[1], int(grid[r, c]))
                           for r, c in cells if (r, c) != anchor))
        _require(key not in templates or templates[key] == sig, "conflicting dictionary entries")
        templates[key] = sig
        repetitions[key] += 1
    _require(bool(templates), "empty dictionary")
    keys = set(templates)
    _require(all(color not in keys | {background, fill, marker_color}
                 for sig in templates.values() for _, _, color in sig),
             "decoration color collides with a key or structural role")
    board = grid[r0:r1, c0:c1].copy()
    mask = board == marker_color
    mask[0] = mask[-1] = False
    mask[:, 0] = mask[:, -1] = False
    markers = [_marker(cells) for cells in _components(mask, diagonal=False)]
    _require(bool(markers), "missing exchange markers")
    anchor_mask = (board != fill) & (board != marker_color)
    anchors = {tuple(q): int(board[tuple(q)]) for q in np.argwhere(anchor_mask).tolist()}
    _require(bool(anchors) and set(anchors.values()) <= keys, "unrecognized board address")
    _require(all(len(c) == 1 for c in _components(anchor_mask)), "addresses must be isolated")
    for p in anchors:
        _require(all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1
                     for m in markers for q in m["cells"]), "address touches a marker")
    assignment, partners = {}, {}
    usage = Counter()
    for p in anchors:
        eligible = [(i, _transform(p, m)) for i, m in enumerate(markers)
                    if _transform(p, m) in anchors and _transform(p, m) != p]
        _require(len(eligible) == 1, "address has ambiguous or missing symmetry partner")
        i, q = eligible[0]
        assignment[p], partners[p] = i, q
        usage[i] += 1
    _require(set(usage) == set(range(len(markers))), "unused marker")
    info = {"grid": grid, "box": box, "board": board, "fill": fill,
            "background": background, "marker_color": marker_color,
            "templates": templates, "repetitions": dict(repetitions),
            "markers": markers, "anchors": anchors, "assignment": assignment,
            "partners": partners, "pair_counts": tuple(usage[i] // 2 for i in range(len(markers)))}
    # Validate footprints without consulting an output or generation-time ownership.
    ink = _ink(info)
    for (r, c), color in ink.items():
        _require(2 <= r < board.shape[0] - 2 and 2 <= c < board.shape[1] - 2,
                 "transferred decoration leaves the clear inner rim")
        _require(board[r, c] == fill, "transferred decoration overwrites an address or marker")
    return info


def _ink(info, variant="exchange", only_kind=None):
    """Decoration maps also define explicit competing interpretations for checks."""
    result = {}
    for p, key in sorted(info["anchors"].items()):
        m = info["markers"][info["assignment"][p]]
        if only_kind is not None and m["kind"] != only_kind:
            continue
        sig = info["templates"][key if variant != "uniform" else min(info["templates"])]
        receiver = p if variant in ("local", "local_oriented") else info["partners"][p]
        for dr, dc, color in sig:
            if variant not in ("local", "unturned"):
                dr, dc = _flip_offset((dr, dc), m["kind"])
            q = receiver[0] + dr, receiver[1] + dc
            if variant == "exchange":
                _require(q not in result, "transferred motifs overlap")
            result[q] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read a visible dictionary, exchange its decoration, and crop the board."""
    info = _read(input_grid)
    output = info["board"].copy()
    for q, color in _ink(info).items():
        output[q] = color
    return output


def _template(rng, key, decorations, cap):
    h, w = int(rng.integers(2, cap + 1)), int(rng.integers(2, cap + 1))
    style = int(rng.integers(5))
    if style == 0 and min(h, w) >= 3:
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1)}
    elif style == 1:
        cells = {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}
    elif style == 2 and max(h, w) >= 3:
        # A bouncing diagonal spine stays eight-connected even in narrow bounds.
        col, direction, cells = 0, 1, set()
        for row in range(h):
            cells.add((row, col))
            if not 0 <= col + direction < w:
                direction = -direction
            col += direction
        cells.add((0, 1))
        cells.add((h - 1, max(0, min(w - 1, col))))
    else:
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        n = int(rng.integers(3, min(16, h * w) + 1))
        steps = _D8 if style == 3 else _D4
        while len(cells) < n:
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in steps
                               if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
            cells.add(_pick(rng, frontier))
    if len(cells) < 3:
        cells = {(0, 0), (1, 0), (1, 1)}
    cells = sorted(cells)
    anchor = _pick(rng, cells)
    offsets = [(r - anchor[0], c - anchor[1]) for r, c in cells if (r, c) != anchor]
    ncolors = int(rng.integers(1, min(len(decorations), len(offsets) // 2) + 1))
    chosen = [int(c) for c in rng.choice(decorations, size=ncolors, replace=False)]
    colors = chosen * 2 + [_pick(rng, chosen) for _ in range(len(offsets) - 2 * ncolors)]
    rng.shuffle(colors)
    return tuple(sorted([(0, 0, key)] + [(r, c, color) for (r, c), color in zip(offsets, colors)]))


def _templates(rng, keys, decorations, cap):
    result, shapes = {}, set()
    for key in keys:
        for _ in range(64):
            template = _template(rng, key, decorations, cap)
            sig = {(r, c): color for r, c, color in template if (r, c) != (0, 0)}
            geometry = tuple((r, c) for r, c, _ in template)
            asymmetric = all(sig != {_flip_offset(p, kind): color for p, color in sig.items()}
                             for kind in ("H", "V", "P"))
            if asymmetric and (cap == 2 or geometry not in shapes):
                break
        else:
            # A corner anchor on an L has visible orientation for every operation.
            template = ((0, 0, key), (1, 0, decorations[0]), (1, 1, decorations[0]))
        result[key] = template
        shapes.add(tuple((r, c) for r, c, _ in template))
    return result


def _shelf_options(templates, ids):
    options = []
    for width in range(6, 29):
        row, col, row_height = 1, 1, 0
        places = []
        for key in ids:
            template = templates[key]
            r0, c0, r1, c1 = _bounds([(r, c) for r, c, _ in template])
            h, w = r1 - r0, c1 - c0
            if w + 2 > width:
                break
            if col + w > width - 1:
                row, col, row_height = row + row_height + 1, 1, 0
            places.append((key, row - r0, col - c0))
            col += w + 1
            row_height = max(row_height, h)
        else:
            height = row + row_height + 1
            if height <= 28 and width * height > 2 * sum(len(templates[k]) for k in ids):
                options.append(((height, width), places))
    return options


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _tile(rng, kind, npairs, templates, endpoint_keys, limit, compact=False):
    length = 1 if kind == "P" else int(rng.integers(2, 8))
    start = -(length // 2)
    marker_cells = ([(0, 0)] if kind == "P" else
                    [(0, c) for c in range(start, start + length)] if kind == "H" else
                    [(r, 0) for r in range(start, start + length)])
    m = {"kind": kind, "origin": marker_cells[0], "cells": marker_cells}
    # For lines the origin's other coordinate does not affect the transformation.
    occupied, blocked = set(marker_cells), _halo(marker_cells)
    anchors, ink = {}, {}
    for i in range(npairs):
        key1, key2 = endpoint_keys[2 * i:2 * i + 2]
        if compact:
            positions = [(3, (i - (npairs - 1) / 2) * 6)] if kind == "H" else (
                [((i - (npairs - 1) / 2) * 6, 3)] if kind == "V" else
                [(3, 3), (3, -3), (0, 6)][i:i + 1])
            positions = [(int(r), int(c)) for r, c in positions]
        else:
            reach = 8 if npairs > 1 else 6
            positions = ([(r, c) for r in range(2, reach + 1) for c in range(-reach, reach + 1)]
                         if kind == "H" else
                         [(r, c) for r in range(-reach, reach + 1) for c in range(2, reach + 1)]
                         if kind == "V" else
                         [(r, c) for r in range(0, reach + 1) for c in range(-reach, reach + 1)
                          if r > 0 or c > 0])
            rng.shuffle(positions)
        accepted = None
        # Prefer compact actual footprints, while preserving random ordering of ties.
        if not compact:
            positions.sort(key=lambda p: max(abs(p[0]), abs(p[1])) + float(rng.random()) * 4)
        for p in positions:
            q = _transform(p, m)
            stamps = []
            for sender, receiver, key in ((p, q, key1), (q, p, key2)):
                stamp = {receiver}
                for dr, dc, _ in templates[key]:
                    if (dr, dc) != (0, 0):
                        stamp.add(_transform((sender[0] + dr, sender[1] + dc), m))
                stamps.append(stamp)
            if stamps[0] & _halo(stamps[1]) or any(stamp & blocked for stamp in stamps):
                continue
            footprint = occupied | stamps[0] | stamps[1]
            r0, c0, r1, c1 = _bounds(footprint)
            if r1 - r0 + 2 > limit[0] or c1 - c0 + 2 > limit[1]:
                continue
            accepted = p, q, stamps
            break
        if accepted is None:
            return None
        p, q, stamps = accepted
        anchors[p], anchors[q] = key1, key2
        for sender, receiver, key in ((p, q, key1), (q, p, key2)):
            for dr, dc, color in templates[key]:
                if (dr, dc) != (0, 0):
                    ink[_transform((sender[0] + dr, sender[1] + dc), m)] = color
        occupied |= stamps[0] | stamps[1]
        blocked |= _halo(stamps[0] | stamps[1])
    r0, c0, _, _ = _bounds(occupied)
    move = lambda p: (p[0] - r0, p[1] - c0)
    return {"kind": kind, "marker": [move(p) for p in marker_cells],
            "anchors": {move(p): key for p, key in anchors.items()},
            "ink": {move(p): color for p, color in ink.items()},
            "footprint": {move(p) for p in occupied}}


def _pack_tiles(rng, tiles, shape):
    blocked, placed, axes = set(), [], set()
    ordered = sorted(tiles, key=lambda t: -len(t["footprint"]))
    for tile in ordered:
        _, _, h, w = _bounds(tile["footprint"])
        candidates = [(r, c) for r in range(1, shape[0] - h) for c in range(1, shape[1] - w)]
        rng.shuffle(candidates)
        chosen = None
        for r, c in candidates:
            footprint = {(r + a, c + b) for a, b in tile["footprint"]}
            kind = tile["kind"]
            axis = ((kind, r + tile["marker"][0][0]) if kind == "H" else
                    (kind, c + tile["marker"][0][1]) if kind == "V" else
                    (kind, r + tile["marker"][0][0], c + tile["marker"][0][1]))
            if footprint & blocked or axis in axes:
                continue
            chosen = (r, c, footprint, axis)
            break
        if chosen is None:
            return None
        r, c, footprint, axis = chosen
        axes.add(axis)
        blocked |= _halo(footprint)
        placed.append((tile, r, c))
    return placed


def _free_references(rng, templates, ids, shape, fallback):
    occupied, places = set(), []
    for key in ids:
        r0, c0, r1, c1 = _bounds([(r, c) for r, c, _ in templates[key]])
        h, w = r1 - r0, c1 - c0
        candidates = [(r - r0, c - c0) for r in range(1, shape[0] - h)
                      for c in range(1, shape[1] - w)]
        rng.shuffle(candidates)
        for ar, ac in candidates:
            footprint = {(ar + dr, ac + dc) for dr, dc, _ in templates[key]}
            if not footprint & occupied:
                occupied |= _halo(footprint)
                places.append((key, ar, ac))
                break
        else:
            return fallback
    return places


def _generate_pair(rng, palette, plans, shared, witness, independent_palette):
    for attempt in range(48):
        compact = attempt >= 32
        cap = 2 if compact else (3 if attempt >= 16 else int(rng.integers(2, 6)))
        order = list(map(int, rng.permutation(10))) if independent_palette else list(palette)
        background, fill, marker_color = order[:3]
        if not witness and rng.random() < .25:
            fill = background
        ntypes = 2 if witness else int(rng.integers(2 if shared else 1, 4))
        keys, decorations = order[3:3 + ntypes], order[3 + ntypes:]
        templates = _templates(rng, keys, decorations[:int(rng.integers(1, 4))], cap)
        if (witness or shared) and len({tuple((r, c, color) for r, c, color in t if (r, c) != (0, 0))
                                       for t in templates.values()}) < 2:
            continue
        if witness and not any(_nonsolid(t) for t in templates.values()):
            continue
        refs = [key for key in keys for _ in range(2 if witness else int(rng.choice([1, 1, 1, 2])))]
        rng.shuffle(refs)
        options = _shelf_options(templates, refs)
        layout = int(rng.integers(2))
        feasible = [(shape, places) for shape, places in options if shape[layout] <= 13]
        if not feasible:
            continue
        # Narrow reference fields allow varied board aspect ratios in either layout.
        feasible.sort(key=lambda item: item[0][layout])
        ref_shape, ref_fallback = _pick(rng, feasible[:max(1, len(feasible) // 3)])
        gap = int(rng.integers(1, 4)) if not compact else 1
        maximum = [26, 26]
        maximum[layout] = 26 - ref_shape[layout] - gap
        if min(maximum) < 10:
            continue
        pair_counts = [int(rng.integers(2, 4)) if shared else 1 for _ in plans]
        if compact:
            pair_counts = [2 if shared else 1 for _ in plans]
        endpoint_keys = []
        for count in pair_counts:
            colors = [int(_pick(rng, keys)) for _ in range(2 * count)]
            endpoint_keys.append(colors)
        if ntypes >= 2:
            endpoint_keys[0][:2] = keys[:2]
        if witness:
            # Both visible dictionary entries are requested, with different partners.
            endpoint_keys[0][:2] = keys[:2]
        tiles = [_tile(rng, kind, count, templates, colors, maximum, compact)
                 for kind, count, colors in zip(plans, pair_counts, endpoint_keys)]
        if any(t is None for t in tiles):
            continue
        min_h = max(_bounds(t["footprint"])[2] for t in tiles) + 2
        min_w = max(_bounds(t["footprint"])[3] for t in tiles) + 2
        if compact or attempt >= 24:
            shape = tuple(maximum)
        else:
            shape = (int(rng.integers(min_h, maximum[0] + 1)),
                     int(rng.integers(min_w, maximum[1] + 1)))
        placed = _pack_tiles(rng, tiles, shape)
        if placed is None:
            continue
        board = np.full((shape[0] + 2, shape[1] + 2), fill, dtype=np.int8)
        board[0] = board[-1] = marker_color
        board[:, 0] = board[:, -1] = marker_color
        output = board.copy()
        for tile, r, c in placed:
            for a, b in tile["marker"]:
                board[r + a + 1, c + b + 1] = output[r + a + 1, c + b + 1] = marker_color
            for (a, b), key in tile["anchors"].items():
                board[r + a + 1, c + b + 1] = output[r + a + 1, c + b + 1] = key
            for (a, b), color in tile["ink"].items():
                output[r + a + 1, c + b + 1] = color
        sizes = [ref_shape, board.shape]
        needed = [max(s[0] for s in sizes) + 2, max(s[1] for s in sizes) + 2]
        needed[layout] = sizes[0][layout] + sizes[1][layout] + gap + 2
        if max(needed) > 30:
            continue
        canvas_shape = tuple(int(rng.integers(n, 31)) for n in needed)
        origins = [[0, 0], [0, 0]]
        first = int(rng.integers(2))
        begin = int(rng.integers(0, canvas_shape[layout] - sum(s[layout] for s in sizes) - gap + 1))
        origins[first][layout] = begin
        origins[1 - first][layout] = begin + sizes[first][layout] + gap
        for i in range(2):
            origins[i][1 - layout] = int(rng.integers(0, canvas_shape[1 - layout] - sizes[i][1 - layout] + 1))
        grid = np.full(canvas_shape, background, dtype=np.int8)
        br, bc = origins[1]
        grid[br:br + board.shape[0], bc:bc + board.shape[1]] = board
        positions = _free_references(rng, templates, refs, ref_shape, ref_fallback)
        rr, rc = origins[0]
        for key, ar, ac in positions:
            for dr, dc, color in templates[key]:
                grid[rr + ar + dr, rc + ac + dc] = color
        try:
            info = _read(grid)
            if not np.array_equal(solve(grid), output):
                continue
            if (witness or shared) and (_ink(info) == _ink(info, "local_oriented") or
                                        _ink(info) == _ink(info, "uniform")):
                continue
            if not all(_ink(info, only_kind=kind) != _ink(info, "unturned", only_kind=kind)
                       for kind in plans):
                continue
        except ValueError:
            continue
        return {"input": grid, "output": output}
    raise RuntimeError("bounded construction could not fit a readable exchange board")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Reproducibly construct demonstrations and shared-marker test recombinations."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, ns = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    modes = [("H",), ("V",), ("P",), ("H", "V"), ("H", "P"), ("V", "P"), ("H", "V", "P")]
    kinds = _pick(rng, modes)
    independent_palette = bool(rng.integers(2))
    demonstrations = [[], []]
    for i, kind in enumerate(rng.permutation(kinds)):
        demonstrations[i % 2].append(str(kind))
    for plan in demonstrations:
        if not plan:
            plan.append(str(_pick(rng, kinds)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            if split == "train" and index < 2:
                plans = demonstrations[index]
            else:
                ngroups = int(rng.choice([1, 1, 2])) if split == "test" else int(rng.integers(1, 4))
                plans = [str(_pick(rng, kinds)) for _ in range(ngroups)]
            for _ in range(8):
                pair = _generate_pair(rng, palette, plans, split == "test",
                                      split == "train" and index < 2, independent_palette)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("could not make distinct inputs")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check visible family semantics, preservation, evidence, and the split shift."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid example counts")
    seen, trained, witnessed, exchange_witness, dictionary_witness = set(), set(), set(), False, False
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _read(inp)
            expected = solve(inp)
            _require(np.array_equal(out, expected), "incorrect exchange or workboard crop")
            board = info["board"]
            _require(out.shape == board.shape, "output must include exactly the board frame")
            _require(np.array_equal(out[board != info["fill"]], board[board != info["fill"]]),
                     "existing board cells changed")
            _require(np.count_nonzero(out != board) == sum(len(info["templates"][k])
                                                          for k in info["anchors"].values()),
                     "incorrect number of transferred decoration cells")
            active = {m["kind"] for m in info["markers"]}
            if split == "train":
                _require(set(info["pair_counts"]) == {1}, "training uses one pair per marker")
                if index < 2:
                    _require(len(info["templates"]) == 2 and
                             set(info["anchors"].values()) == set(info["templates"]),
                             "initial demonstrations must request two dictionary keys")
                    _require(min(info["repetitions"].values()) >= 2,
                             "initial demonstrations must repeat the dictionary")
                    _require(info["background"] != info["fill"] and
                             any(_nonsolid(((0, 0, key),) + sig)
                                 for key, sig in info["templates"].items()),
                             "initial demonstrations must expose transparent gaps")
                    _require(_ink(info) != _ink(info, "local_oriented") and
                             _ink(info) != _ink(info, "uniform"),
                             "initial demonstrations must expose key-specific exchange")
                trained |= active
                witnessed |= {kind for kind in active if _ink(info, only_kind=kind) !=
                              _ink(info, "unturned", only_kind=kind)}
                exchange_witness |= _ink(info) != _ink(info, "local_oriented")
                dictionary_witness |= _ink(info) != _ink(info, "uniform")
            else:
                _require(active <= trained, "test uses an undemonstrated marker operation")
                _require(min(info["pair_counts"]) >= 2, "test must recombine pairs at shared markers")
                _require(len(info["templates"]) >= 2 and
                         any(info["anchors"][p] != info["anchors"][q]
                             for p, q in info["partners"].items()),
                         "test must exchange differently keyed messages")
                _require(_ink(info) != _ink(info, "local_oriented") and
                         _ink(info) != _ink(info, "uniform"),
                         "test must visibly require partner-specific lookup")
    _require(trained <= witnessed, "training conceals a symmetry operation")
    _require(exchange_witness and dictionary_witness, "training does not expose keyed exchange")


def _features(pair):
    info = _read(pair["input"])
    return {"shape": tuple(info["grid"].shape), "output_shape": tuple(info["board"].shape),
            "kinds": tuple(sorted(m["kind"] for m in info["markers"])),
            "marker_lengths": tuple(len(m["cells"]) for m in info["markers"]),
            "pair_counts": info["pair_counts"], "types": len(info["templates"]),
            "motif_sizes": tuple(len(t) + 1 for t in info["templates"].values()),
            "geometries": tuple(tuple((r, c) for r, c, _ in t) for t in info["templates"].values()),
            "palette": (info["background"], info["fill"], info["marker_color"]),
            "multicolor": any(len({c for _, _, c in t}) > 1 for t in info["templates"].values()),
            "different_partners": any(info["anchors"][p] != info["anchors"][q]
                                      for p, q in info["partners"].items())}


def self_test():
    """Bounded 42-seed checks with geometric, dictionary, and mutation witnesses."""
    puzzles, records = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                records.append(_features(pair))
                original = pair["input"].copy()
                _require(np.array_equal(solve(original), pair["output"]), "solver mismatch")
                _require(np.array_equal(original, pair["input"]), "solver mutated its input")
                # A whole-scene dihedral transformation carries the answer with it.
                for transformed, expected in ((np.rot90(original), np.rot90(pair["output"])),
                                               (np.fliplr(original), np.fliplr(pair["output"]))):
                    _require(np.array_equal(solve(transformed), expected), "geometric covariance failed")
    _require(len({_key(p["train"][0]["input"]) for p in puzzles}) == 42, "seeds duplicate")
    _require(len({r["shape"] for r in records}) >= 35, "insufficient canvas variation")
    _require(len({r["output_shape"] for r in records}) >= 25, "insufficient board variation")
    _require(len({g for r in records for g in r["geometries"]}) >= 70, "insufficient motif variation")
    _require({r["types"] for r in records} == {1, 2, 3}, "missing dictionary counts")
    _require({k for r in records for k in r["kinds"]} == {"H", "V", "P"}, "missing operations")
    _require({n for r in records for n in r["pair_counts"]} >= {1, 2, 3}, "missing shared-marker variation")
    _require(len({n for r in records for n in r["motif_sizes"]}) >= 7, "missing size variation")
    _require(sum(r["multicolor"] for r in records) >= 10, "missing colored-offset variation")
    _require(len({r["palette"] for r in records}) >= 25, "missing palette-role variation")
    _require(all(_features(p["train"][0])["different_partners"] for p in puzzles),
             "first demonstration must show differently keyed partners")
    a, b = generate_puzzle(991, 6, 3), generate_puzzle(991, 6, 3)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ("input", "output")), "nondeterministic generation")
    for nt, ns in ((2, 1), (10, 4)):
        p = generate_puzzle(829 + nt, nt, ns)
        _require(len(p["train"]) == nt and len(p["test"]) == ns, "counts not honored")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt frame accepted")
    # Erasing one address breaks reciprocal matching, not merely motif counts.
    info = _read(a["train"][0]["input"])
    missing = a["train"][0]["input"].copy()
    p = min(info["anchors"])
    missing[info["box"][0] + p[0], info["box"][1] + p[1]] = info["fill"]
    try:
        solve(missing)
    except ValueError:
        pass
    else:
        raise AssertionError("unpaired address accepted")
    return {"seeds": 42, "pairs": len(records),
            "input_shapes": len({r["shape"] for r in records}),
            "output_shapes": len({r["output_shape"] for r in records}),
            "motif_geometries": len({g for r in records for g in r["geometries"]}),
            "motif_sizes": sorted({n for r in records for n in r["motif_sizes"]}),
            "shared_pair_counts": sorted({n for r in records for n in r["pair_counts"]}),
            "semantic_verification": "generator-owned checks only"}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
