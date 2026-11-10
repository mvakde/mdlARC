"""Collision loom: shape-addressed censuses drive a lattice of XOR stamps.

Only NumPy and the standard library are used. No reference files, generation
metadata, or seed are consulted by solve().
"""
from __future__ import annotations

from collections import Counter
import copy

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "color bounds")
    return a


def _crop(mask):
    points = np.argwhere(mask)
    _require(len(points) > 0, "empty mask")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return np.asarray(mask[lo[0]:hi[0], lo[1]:hi[1]], dtype=bool).copy()


def _key(mask):
    return mask.shape, mask.astype(np.uint8).tobytes()


def _canonical(mask):
    return min(_key(np.rot90(mask, k)) for k in range(4))


def _chiral(mask):
    return _canonical(mask) != _canonical(np.fliplr(mask))


def _components(mask, directions=DIR8):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    cells.append(q)
        groups.append(cells)
    return groups


def _objects(grid, background):
    result = []
    for color in sorted(set(map(int, np.unique(grid))) - {background}):
        for cells in _components(grid == color):
            top = min(r for r, c in cells)
            left = min(c for r, c in cells)
            h = max(r for r, c in cells) - top + 1
            w = max(c for r, c in cells) - left + 1
            mask = np.zeros((h, w), dtype=bool)
            for r, c in cells:
                mask[r-top, c-left] = True
            result.append({"color": color, "top": top, "left": left,
                           "mask": mask, "cells": cells})
    return result


def _outline(h, w):
    a = np.zeros((h, w), dtype=bool)
    a[[0, -1], :] = True
    a[:, [0, -1]] = True
    return a


def _decode(value):
    """Recover all roles from the input, before looking at any output."""
    grid = _grid(value)
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    _require(int(counts.max()) * 2 > grid.size, "background must be dominant")
    objects = _objects(grid, background)
    frames = []
    for index, obj in enumerate(objects):
        h, w = obj["mask"].shape
        if min(h, w) < 5 or h == w or not np.array_equal(obj["mask"], _outline(h, w)):
            continue
        r, c = obj["top"], obj["left"]
        inside = [(i, x) for i, x in enumerate(objects) if i != index
                  and all(r < y < r+h-1 and c < z < c+w-1 for y, z in x["cells"])]
        if len(inside) != 1:
            continue
        ki, key = inside[0]
        kh, kw = key["mask"].shape
        margins = (key["top"] - r - 1, r+h-2 - (key["top"]+kh-1),
                   key["left"] - c - 1, c+w-2 - (key["left"]+kw-1))
        if min(margins) < 1 or max(kh, kw) > 6:
            continue
        interior = grid[r+1:r+h-1, c+1:c+w-1]
        if set(map(int, np.unique(interior))) != {background, key["color"]}:
            continue
        frames.append({"index": index, "key_index": ki, "key": key,
                       "frame": obj, "margins": margins, "wide": w > h})
    _require(len(frames) == 2 and {f["wide"] for f in frames} == {False, True},
             "need exactly one wide and one tall keyed frame")
    horizontal = next(f for f in frames if f["wide"])
    vertical = next(f for f in frames if not f["wide"])
    primary, secondary = horizontal["frame"]["color"], vertical["frame"]["color"]
    _require(primary != secondary, "frame colors must differ")
    excluded = {f[k] for f in frames for k in ("index", "key_index")}
    _require(len(excluded) == 4, "nested/shared keys")
    field = [o for i, o in enumerate(objects) if i not in excluded]
    _require(all(o["color"] not in (primary, secondary) for o in objects
                 if o is not horizontal["frame"] and o is not vertical["frame"]),
             "frame colors are reserved")
    # Enforce the background moat between every distinct object, regardless
    # of color, so same-color copies cannot merge and keys cannot leak out.
    labels = {}
    for i, obj in enumerate(objects):
        for p in obj["cells"]:
            labels[p] = i
    for p, i in labels.items():
        for dr, dc in DIR8:
            q = p[0]+dr, p[1]+dc
            _require(q not in labels or labels[q] == i, "objects touch")
    _require(all(max(o["mask"].shape) <= 6 for o in field), "field extent exceeds six")
    hkey, vkey = horizontal["key"]["mask"], vertical["key"]["mask"]
    hc, vc = _canonical(hkey), _canonical(vkey)
    _require(hc != vc, "keys cannot address the same class")
    groups = {"H": [], "V": [], "noise": []}
    for obj in field:
        key = _canonical(obj["mask"])
        groups["H" if key == hc else "V" if key == vc else "noise"].append(obj)
    columns, rows = len(groups["H"]), len(groups["V"])
    _require(rows > 0 and columns > 0, "both censuses must be positive")
    tile = _xor_tile(hkey, vkey)
    _require(rows*(tile.shape[0]+1)-1 <= 30 and columns*(tile.shape[1]+1)-1 <= 30,
             "census lattice exceeds output budget")
    return {"background": background, "primary": primary, "secondary": secondary,
            "H": hkey, "V": vkey, "tile": tile, "columns": columns, "rows": rows,
            "groups": groups, "frames": (horizontal, vertical), "field": field}


def _pad_keys(a, b, centered=False):
    h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    padded = []
    for mask in (a, b):
        canvas = np.zeros((h, w), dtype=bool)
        r, c = ((h-mask.shape[0])//2, (w-mask.shape[1])//2) if centered else (0, 0)
        canvas[r:r+mask.shape[0], c:c+mask.shape[1]] = mask
        padded.append(canvas)
    return padded


def _xor_tile(a, b):
    aa, bb = _pad_keys(a, b)
    return aa ^ bb


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _decode(input_grid)
    tile, rows, columns = d["tile"], d["rows"], d["columns"]
    h, w = tile.shape
    # Modular coordinates are independent of the concatenation constructor.
    r = np.arange(rows*(h+1)-1) % (h+1)
    c = np.arange(columns*(w+1)-1) % (w+1)
    extended = np.pad(tile, ((0, 1), (0, 1)), constant_values=False)
    return np.where(extended[r[:, None], c[None, :]],
                    d["primary"], d["secondary"]).astype(np.int8)


def _render(tile, columns, rows, primary, secondary):
    colored = np.where(tile, primary, secondary).astype(np.int8)
    gap = np.full((tile.shape[0], 1), secondary, dtype=np.int8)
    strip = np.concatenate([x for i in range(columns)
                            for x in ((colored,) if i == columns-1 else (colored, gap))], axis=1)
    separator = np.full((1, strip.shape[1]), secondary, dtype=np.int8)
    return np.concatenate([x for i in range(rows)
                           for x in ((strip,) if i == rows-1 else (strip, separator))], axis=0)


def _random_mask(rng, limit):
    h, w = map(int, rng.integers(2, limit+1, size=2))
    kind = int(rng.integers(7))
    mask = np.zeros((h, w), dtype=bool)
    if kind == 0:
        return np.ones((1, int(rng.integers(1, limit+1))), dtype=bool)
    if kind == 1:
        mask[:] = True
        if min(h, w) > 2 and rng.random() < .65:
            mask[1:-1, 1:-1] = False
    elif kind == 2:
        mask[:, 0] = True
        mask[-1, :] = True
        if rng.random() < .5:
            mask[int(rng.integers(h)), :] = True
    elif kind == 3:
        mask[int(rng.integers(h)), :] = True
        mask[:, int(rng.integers(w))] = True
    elif kind == 4:
        mask[0, 0] = True
        mask[1:, 1] = True
        mask[-1, 1:] = True
        if h > 3 and w > 3 and rng.random() < .5:
            mask[1:-1, 1:-1] = True
    else:
        # Growth can make branches, holes, concavities and diagonal bridges.
        directions = DIR4 if kind == 5 else DIR8
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        target = int(rng.integers(2, h*w+1))
        for _ in range(target-1):
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in directions
                               if 0 <= r+dr < h and 0 <= c+dc < w
                               and (r+dr, c+dc) not in cells})
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        for p in cells:
            mask[p] = True
    return _crop(np.rot90(mask, int(rng.integers(4))))


def _fallback_keys(mode):
    if mode == "cancel":
        a = np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool)
        b = a.copy()
        b[1, 1] = True
    elif mode in ("unequal", "ordinary"):
        a = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1]], dtype=bool)
        b = np.array([[1, 0, 0], [1, 1, 1]], dtype=bool)
    else:
        a = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool)
        b = np.fliplr(a).copy() if mode == "reflection" else np.array([[1, 1, 1], [1, 0, 0]], dtype=bool)
    return a, b


def _sample_keys(rng, mode, limit):
    for _ in range(500):
        a = _random_mask(rng, limit)
        if mode == "cancel":
            if min(a.shape) < 3 or not _chiral(a):
                continue
            empty = np.argwhere(~a[1:-1, 1:-1]) + 1
            if not len(empty):
                continue
            b = a.copy()
            for i in rng.choice(len(empty), size=int(rng.integers(1, min(2, len(empty))+1)), replace=False):
                b[tuple(empty[int(i)])] = True
            if len(_components(b)) != 1:
                continue
        elif mode == "reflection":
            if not _chiral(a) or max(a.shape) != limit:
                continue
            b = np.fliplr(a).copy()
        else:
            if mode == "unequal" and (not _chiral(a) or len(_components(a, DIR4)) == 1):
                continue
            if mode == "test" and max(a.shape) != limit:
                continue
            b = _random_mask(rng, 3 if mode == "unequal" else limit)
        if _canonical(a) == _canonical(b):
            continue
        aa, bb = _pad_keys(a, b)
        if not (np.any(aa & bb) and np.any(aa ^ bb)):
            continue
        if mode == "unequal":
            ca, cb = _pad_keys(a, b, True)
            if (not _chiral(b) or _canonical(np.fliplr(a)) == _canonical(b)
                    or not np.any(aa & ~bb) or not np.any(bb & ~aa)
                    or np.array_equal(aa ^ bb, ca ^ cb)):
                continue
        return a, b
    return _fallback_keys(mode)


def _frame(mask, color, ink, background, wide, rng, compact=False):
    margins = [1]*4 if compact else list(map(int, rng.integers(1, 3, size=4)))
    top, bottom, left, right = margins
    h, w = mask.shape[0]+top+bottom+2, mask.shape[1]+left+right+2
    if wide:
        right += max(0, h+2-w)
        w = mask.shape[1]+left+right+2
    else:
        bottom += max(0, w+2-h)
        h = mask.shape[0]+top+bottom+2
    a = np.full((h, w), background, dtype=np.int8)
    a[_outline(h, w)] = color
    a[top+1:top+1+mask.shape[0], left+1:left+1+mask.shape[1]][mask] = ink
    return a


class _PackingError(Exception):
    pass


def _pack(items, background, rng, compact=False):
    """Randomized shelves on either axis; disjoint boxes have a blank moat."""
    foreground = sum(int(np.count_nonzero(x != background)) for x in items)
    for attempt in range(100):
        transpose = bool(rng.integers(2))
        shapes = [x.shape[::-1] if transpose else x.shape for x in items]
        width = 30 if attempt > 70 or compact else int(rng.integers(18, 31))
        order = list(map(int, rng.permutation(len(items))))
        if attempt % 3 != 0:
            order.sort(key=lambda i: -shapes[i][0])
        shelves, current, used, height = [], [], 0, 0
        failed = False
        for i in order:
            ih, iw = shapes[i]
            if iw > width:
                failed = True
                break
            gap = 1 if compact else int(rng.integers(1, 3))
            if current and used + gap + iw > width:
                shelves.append((current, used, height))
                current, used, height = [], 0, 0
            x = used + (gap if current else 0)
            current.append((i, x))
            used, height = x+iw, max(height, ih)
        if failed:
            continue
        shelves.append((current, used, height))
        needed = sum(s[2] for s in shelves) + len(shelves)-1
        if needed > 30:
            continue
        canvas_h = min(30, max(needed + int(rng.integers(0, 5)), 2*foreground//width+1))
        if canvas_h < needed or 2*foreground >= canvas_h*width:
            continue
        positions, y = {}, int(rng.integers(canvas_h-needed+1))
        for shelf, used, sh in shelves:
            shift = int(rng.integers(width-used+1))
            for i, x in shelf:
                ih, iw = shapes[i]
                positions[i] = (y + int(rng.integers(sh-ih+1)), x+shift)
            y += sh+1
        out = np.full((canvas_h, width), background, dtype=np.int8)
        for i, item in enumerate(items):
            r, c = positions[i]
            piece = item.T if transpose else item
            out[r:r+piece.shape[0], c:c+piece.shape[1]] = piece
        return out.T.copy() if transpose else out
    raise _PackingError("bounding boxes do not fit")


def _palette(rng, previous=None):
    for _ in range(100):
        colors = list(map(int, rng.permutation(10)))
        if previous is None or all(colors[i] != previous[i] for i in range(3)):
            return colors
    # Cyclic shifting of a permutation preserves all distinctness constraints.
    return [(c+1) % 10 for c in previous]


def _make_pair(rng, mode, counts=None, previous_palette=None, force_fallback=False):
    for attempt in range(70):
        fallback = force_fallback or attempt == 69
        limit = int(rng.integers(5, 7)) if mode in ("test", "reflection") else 4
        hkey, vkey = _fallback_keys(mode) if fallback else _sample_keys(rng, mode, limit)
        tile = _xor_tile(hkey, vkey)
        cap_c = min(6, 31//(tile.shape[1]+1))
        cap_r = min(6, 31//(tile.shape[0]+1))
        if counts is None:
            low = 2 if mode in ("test", "reflection") else 1
            columns = 2 if fallback else int(rng.integers(low, cap_c+1))
            rows = 2 if fallback else int(rng.integers(low, cap_r+1))
        else:
            columns, rows = counts
        if columns > cap_c or rows > cap_r:
            continue
        colors = _palette(rng, previous_palette)
        background, primary, secondary, hi, vi = colors[:5]
        if rng.random() < .25:
            vi = hi
        field_palette = colors[3:]
        masks, roles = [], []
        for name, key, amount in (("H", hkey, columns), ("V", vkey, rows)):
            turns = list(map(int, rng.integers(0, 4, size=amount)))
            if mode in ("cancel", "unequal"):
                turns[0] = 1
            for turn in turns:
                masks.append(np.rot90(key, turn).copy())
                roles.append(name)
        forbidden = {_canonical(hkey), _canonical(vkey)}
        noise = []
        if mode in ("cancel", "unequal"):
            mirror = np.fliplr(hkey).copy()
            if _canonical(mirror) in forbidden or not _chiral(hkey):
                continue
            noise.extend([mirror, np.rot90(mirror, int(rng.integers(4)))])
            if mode == "unequal":
                # Teach exact size rather than scale-invariant matching.
                noise.append(np.repeat(np.repeat(vkey, 2, axis=0), 2, axis=1))
        else:
            for _ in range(int(rng.integers(0 if mode == "ordinary" else 1, 4))):
                for _ in range(100):
                    candidate = _random_mask(rng, 3 if fallback else min(limit, 4))
                    if _canonical(candidate) not in forbidden:
                        noise.append(candidate)
                        break
        masks += noise
        roles += ["noise"]*len(noise)
        inks = [int(rng.choice(field_palette)) for _ in masks]
        if mode in ("cancel", "unequal"):
            # Different-color match, unrelated same-color object, and color
            # shared across populations: no literal ink can select a class.
            inks[0] = colors[5]
            inks[columns] = colors[5]
            inks[-1] = colors[5]
            if columns > 1:
                inks[1] = colors[6]
        items = [_frame(hkey, primary, hi, background, True, rng, fallback),
                 _frame(vkey, secondary, vi, background, False, rng, fallback)]
        items += [np.where(mask, ink, background).astype(np.int8) for mask, ink in zip(masks, inks)]
        try:
            inp = _pack(items, background, rng, fallback)
        except _PackingError:
            continue
        out = _render(tile, columns, rows, primary, secondary)
        # Parsing verifies construction separation; it never selects a parse
        # or a shape based on the generated label.
        d = _decode(inp)
        _require(d["columns"] == columns and d["rows"] == rows, "packing changed census")
        return {"input": inp, "output": out}, colors
    raise AssertionError("bounded pair construction exhausted")


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _fingerprint(a):
    return a.shape, a.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    orientation = bool(rng.integers(2))
    second_counts = (1, 3) if orientation else (3, 1)
    first_counts = ((2, 2) if rng.random() < .6 else ((3, 2) if orientation else (2, 3)))
    previous = None
    for split, amount in (("train", num_train), ("test", num_test)):
        for i in range(amount):
            mode = ("cancel" if i == 0 else "unequal" if i == 1 else "ordinary") if split == "train" else (
                "reflection" if rng.random() < .30 else "test")
            census = first_counts if split == "train" and i == 0 else second_counts if split == "train" and i == 1 else None
            # Later tests occasionally have a single axis, after training has
            # already shown it. Their larger key remains the size shift.
            if split == "test" and i > 0 and rng.random() < .45:
                census = (1, 2) if rng.random() < .5 else (2, 1)
            for _ in range(30):
                pair, palette = _make_pair(rng, mode, census, previous if split == "train" and i == 1 else None)
                signature = _fingerprint(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    if split == "train" and i == 0:
                        previous = palette
                    break
            else:
                raise AssertionError("distinct-input budget exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _different(a, b):
    return a.shape != b.shape or not np.array_equal(a, b)


def _features(d):
    a, b = _pad_keys(d["H"], d["V"])
    ca, cb = _pad_keys(d["H"], d["V"], True)
    field = d["field"]
    kc = [_canonical(o["mask"]) for o in field]
    noise_counts = Counter(_canonical(o["mask"]) for o in d["groups"]["noise"])
    key_colors = [f["key"]["color"] for f in d["frames"]]
    return {"counts": (d["columns"], d["rows"]),
            "key_extents": (d["H"].shape, d["V"].shape),
            "overlap": bool(np.any(a & b)), "exclusive_both": bool(np.any(a & ~b) and np.any(b & ~a)),
            "blank_boundary": bool(not d["tile"][0].any() and not d["tile"][-1].any()
                                   and not d["tile"][:, 0].any() and not d["tile"][:, -1].any()),
            "center_differs": _different(d["tile"], ca ^ cb),
            "diagonal_key": any(len(_components(d[k], DIR4)) > 1 for k in ("H", "V")),
            "reflection_negative": any(_canonical(np.fliplr(d[k])) in noise_counts
                                       and _chiral(d[k]) for k in ("H", "V")),
            "scale_negative": any(_canonical(np.repeat(np.repeat(d[k], 2, axis=0), 2, axis=1)) in noise_counts
                                  for k in ("H", "V")),
            "repeated_noise": any(n > 1 for n in noise_counts.values()),
            "quarter_turn": any(_key(o["mask"]) in (_key(np.rot90(d[k])), _key(np.rot90(d[k], 3)))
                                and _key(o["mask"]) != _key(d[k]) for k in ("H", "V") for o in d["groups"][k]),
            "different_color_match": any(o["color"] != key_colors[i] for i, k in enumerate(("H", "V"))
                                         for o in d["groups"][k]),
            "cross_class_color": any(x["color"] == y["color"] and kc[i] != kc[j]
                                     for i, x in enumerate(field) for j, y in enumerate(field) if i < j),
            "noise": len(d["groups"]["noise"])}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    decoded, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            signature = _fingerprint(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            d = _decode(inp)
            _require(np.array_equal(solve(inp), out), "label violates collision loom")
            h, w = d["tile"].shape
            _require(out.shape == (d["rows"]*(h+1)-1, d["columns"]*(w+1)-1), "lattice dimensions")
            _require(bool(np.all(out[h::h+1] == d["secondary"])), "row separators")
            _require(bool(np.all(out[:, w::w+1] == d["secondary"])), "column separators")
            _require(set(map(int, np.unique(out))) == {d["primary"], d["secondary"]}, "output roles")
            decoded[split].append(d)
    train = decoded["train"]
    fs = [_features(d) for d in train]
    for name in ("overlap", "exclusive_both", "blank_boundary", "center_differs", "diagonal_key",
                 "reflection_negative", "scale_negative", "repeated_noise", "quarter_turn", "different_color_match", "cross_class_color"):
        _require(any(f[name] for f in fs), "training lacks witness: " + name)
    _require(any(d["columns"] > 1 and d["rows"] > 1 for d in train), "two-dimensional teaching")
    _require(any(1 in f["counts"] and 3 in f["counts"] for f in fs), "one/three census teaching")
    for role in ("columns", "rows"):
        _require(len({d[role] for d in train}) > 1, "census must vary on both axes")
    for role in ("background", "primary", "secondary"):
        _require(len({d[role] for d in train}) > 1, "literal colors cannot define roles")
    for axis in (0, 1):
        _require(any(f["frame"]["mask"].shape[axis] != d["columns" if i == 0 else "rows"]
                     for d in train for i, f in enumerate(d["frames"])), "frame-size count ambiguity")
    # Explicit parent ablations and overlay alternatives, measured from grids.
    _require(any(len(d["field"]) != d["columns"] and len(d["field"]) != d["rows"]
                 and len(d["groups"]["noise"]) > 0 for d in train), "A ablation fails to change answer")
    _require(any(d["rows"] > 1 and d["columns"] > 1 and min(d["H"].sum(), d["V"].sum()) > 1
                 for d in train), "B ablation fails to change answer")
    for transform in range(1, 8):
        for role in ("H", "V"):
            _require(any(_different(d["tile"], _xor_tile(
                np.rot90(np.fliplr(d["H"]) if transform >= 4 else d["H"], transform % 4) if role == "H" else d["H"],
                np.rot90(np.fliplr(d["V"]) if transform >= 4 else d["V"], transform % 4) if role == "V" else d["V"]))
                         for d in train), "key orientation ambiguity")
    _require(all(max(max(d["H"].shape), max(d["V"].shape)) <= 4 for d in train), "training size envelope")
    _require(all(max(max(d["H"].shape), max(d["V"].shape)) >= 5 for d in decoded["test"]), "missing size shift")
    _require(decoded["test"][0]["rows"] > 1 and decoded["test"][0]["columns"] > 1, "first test must combine axes")


def _puzzle_signature(p):
    return tuple((_fingerprint(pair["input"]), _fingerprint(pair["output"]))
                 for s in ("train", "test") for pair in p[s])


def self_test():
    """Bounded checks and structural diversity measurements over 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_signature(p) for p in puzzles}) == 42, "seed diversity")
    for seed in (0, 7, 41):
        _require(_puzzle_signature(puzzles[seed]) == _puzzle_signature(generate_puzzle(seed)), "determinism")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(12000+a*10+b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    decoded = [_decode(pair["input"]) for p in puzzles for s in ("train", "test") for pair in p[s]]
    masks = {_key(d[k]) for d in decoded for k in ("H", "V")}
    _require(len(masks) >= 80, "shape diversity")
    _require(len({d["tile"].shape for d in decoded}) >= 12, "tile dimension diversity")
    _require(len({(d["columns"], d["rows"]) for d in decoded}) >= 15, "census diversity")
    _require(len({_key(d["tile"]) for d in decoded}) >= 80, "collision geometry diversity")
    _require({d["background"] for d in decoded} == set(range(10)), "background diversity")
    _require(any(min(d[k].shape) == 1 for d in decoded for k in ("H", "V")), "line keys missing")
    _require(any(len(_components(~np.pad(d[k], 1), DIR4)) > 1 for d in decoded for k in ("H", "V")), "holes missing")
    _require(any(_canonical(np.fliplr(d["H"])) == _canonical(d["V"]) for d in decoded), "mirrored key pairs missing")
    # Palette equivariance: solve must read structural roles, not literal colors.
    mapping = np.array([7, 3, 9, 1, 0, 6, 8, 4, 2, 5], dtype=np.int8)
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        _require(np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]]), "palette equivariance")
    # Exercise the deterministic geometry fallback as actual family members.
    rng = np.random.default_rng(183)
    for mode, census in (("cancel", (2, 2)), ("unequal", (1, 3)), ("test", (2, 2)), ("reflection", (2, 2))):
        pair, _ = _make_pair(rng, mode, census, force_fallback=True)
        _require(np.array_equal(solve(pair["input"]), pair["output"]), "fallback label")
    bad = copy.deepcopy(puzzles[0])
    bad["test"][0]["output"].flat[0] = (int(bad["test"][0]["output"].flat[0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return {"seeds": 42, "pairs": len(decoded), "key_masks": len(masks),
            "tile_shapes": len({d["tile"].shape for d in decoded}),
            "census_pairs": len({(d["columns"], d["rows"]) for d in decoded}),
            "collision_tiles": len({_key(d["tile"]) for d in decoded})}


if __name__ == "__main__":
    print(self_test())
