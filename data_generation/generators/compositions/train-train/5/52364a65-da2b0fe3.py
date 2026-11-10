"""Open all geometric seams and peel their incident object faces.

Only Python's standard library and NumPy are used. Color 3 is the fixed
divider; all other roles are read from the input. No parent code is imported.
"""
from __future__ import annotations

import json

import numpy as np


DIVIDER = 3
_SIDES = ("L", "R", "T", "B")


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _background(a):
    colors, counts = np.unique(a, return_counts=True)
    i = int(np.argmax(counts))
    if int(counts[i]) * 2 <= a.size or int(colors[i]) == DIVIDER:
        raise ValueError("background must be a strict majority and differ from 3")
    return int(colors[i])


def _objects(a, bg):
    """Monochromatic four-connected components, in raster order."""
    seen = np.zeros(a.shape, dtype=bool)
    result = []
    h, w = a.shape
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(a[r, c]) == bg:
                continue
            color = int(a[r, c])
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if (0 <= yy < h and 0 <= xx < w and not seen[yy, xx]
                            and int(a[yy, xx]) == color):
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            coords = np.asarray(cells, dtype=int)
            box = (int(coords[:, 0].min()), int(coords[:, 0].max()),
                   int(coords[:, 1].min()), int(coords[:, 1].max()))
            result.append({"box": box, "cells": coords, "color": color})
    return result


def _relations(boxes):
    """Potential edges: axis, line, first endpoint/face, second endpoint/face."""
    result = []
    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            if max(a[0], b[0]) <= min(a[1], b[1]):
                if a[3] + 2 == b[2]:
                    result.append(("V", a[3]+1, i, "R", j, "L"))
                elif b[3] + 2 == a[2]:
                    result.append(("V", b[3]+1, j, "R", i, "L"))
            if max(a[2], b[2]) <= min(a[3], b[3]):
                if a[1] + 2 == b[0]:
                    result.append(("H", a[1]+1, i, "B", j, "T"))
                elif b[1] + 2 == a[0]:
                    result.append(("H", b[1]+1, j, "B", i, "T"))
    return result


def _edges(a, bg, objects):
    return [e for e in _relations([o["box"] for o in objects])
            if np.all((a[:, e[1]] if e[0] == "V" else a[e[1], :]) == bg)]


def _faces(n, edges):
    result = [set() for _ in range(n)]
    for _, _, i, si, j, sj in edges:
        result[i].add(si)
        result[j].add(sj)
    return result


def _peel_mask(obj, faces, depth=2):
    r0, r1, c0, c1 = obj["box"]
    y, x = obj["cells"].T
    erase = np.zeros(len(y), dtype=bool)
    for side in faces:
        erase |= {"L": x < c0 + depth, "R": x > c1 - depth,
                  "T": y < r0 + depth, "B": y > r1 - depth}[side]
    return erase


def _render(a, bg, objects, edges, depth=2, mode="normal"):
    out = a.copy()
    faces = _faces(len(objects), edges)
    for obj, sides in zip(objects, faces):
        if mode == "all_left":
            sides = {"L"}
        erase = _peel_mask(obj, sides, depth)
        if mode == "pixels":
            # Counterfactual: peel the first two occupied pixels per scanline.
            erase = np.zeros(len(obj["cells"]), dtype=bool)
            y, x = obj["cells"].T
            for side in sides:
                across, along = (y, x) if side in ("L", "R") else (x, y)
                for v in np.unique(across):
                    indices = np.flatnonzero(across == v)
                    order = indices[np.argsort(along[indices])]
                    picked = order[:depth] if side in ("L", "T") else order[-depth:]
                    erase[picked] = True
        if mode != "no_peel":
            cells = obj["cells"][erase]
            out[cells[:, 0], cells[:, 1]] = bg
    for axis, line in sorted({(e[0], e[1]) for e in edges}):
        if axis == "V":
            out[:, line] = DIVIDER
        else:
            out[line, :] = DIVIDER
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input's components and original gap relations only."""
    a = _grid(input_grid)
    if np.any(a == DIVIDER):
        raise ValueError("input color 3 is reserved for output dividers")
    bg = _background(a)
    objects = _objects(a, bg)
    return _render(a, bg, objects, _edges(a, bg, objects))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _bounds(boxes):
    return (min(b[0] for b in boxes), max(b[1] for b in boxes),
            min(b[2] for b in boxes), max(b[3] for b in boxes))


def _contains_line(box, axis, line):
    lo, hi = (box[2], box[3]) if axis == "V" else (box[0], box[1])
    return lo <= line <= hi


def _valid_boxes(boxes, max_sides=4):
    bounds = _bounds(boxes)
    if bounds[1]-bounds[0]+1 > 27 or bounds[3]-bounds[2]+1 > 27:
        return False
    for i, a in enumerate(boxes):
        for b in boxes[i+1:]:
            if not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]):
                return False
            # Leave at least one background cell between entire boxes.
            if not (a[1]+1 < b[0] or b[1]+1 < a[0]
                    or a[3]+1 < b[2] or b[3]+1 < a[2]):
                return False
    relations = _relations(boxes)
    if any(_contains_line(b, e[0], e[1]) for e in relations for b in boxes):
        return False
    faces = _faces(len(boxes), relations)
    return max(map(len, faces), default=0) <= max_sides


def _attach(rng, box, side, h, w, contained=False):
    r0, r1, c0, c1 = box
    if side in ("L", "R"):
        low, high = (r0, r1-h+1) if contained and h <= r1-r0+1 else (r0-h+1, r1)
        top = int(rng.integers(low, high+1))
        left = c0-w-1 if side == "L" else c1+2
    else:
        low, high = (c0, c1-w+1) if contained and w <= c1-c0+1 else (c0-w+1, c1)
        left = int(rng.integers(low, high+1))
        top = r0-h-1 if side == "T" else r1+2
    return top, top+h-1, left, left+w-1


def _grow(rng, boxes, target, max_sides):
    boxes = list(boxes)
    for _ in range(240):
        if len(boxes) >= target:
            break
        source = _pick(rng, boxes)
        side = _pick(rng, _SIDES)
        h, w = int(rng.integers(1, 8)), int(rng.integers(1, 8))
        new = _attach(rng, source, side, h, w)
        if _valid_boxes(boxes + [new], max_sides):
            boxes.append(new)
    return boxes


def _decoy(rng, boxes):
    """Try an unpaired piece; it may share a color with a network endpoint."""
    r0, r1, c0, c1 = _bounds(boxes)
    for _ in range(400):
        h, w = int(rng.integers(2, 6)), int(rng.integers(3, 7))
        top = int(rng.integers(r0-7, r1+7))
        left = int(rng.integers(c0-7, c1+7))
        new = top, top+h-1, left, left+w-1
        candidate = boxes + [new]
        if _valid_boxes(candidate) and len(_relations(candidate)) == len(_relations(boxes)):
            return candidate
    return None


def _mask(rng, h, w, mode, diagnostic_side=None):
    m = np.zeros((h, w), dtype=bool)
    if min(h, w) == 1 or mode == "solid":
        return np.ones((h, w), dtype=bool)
    if diagnostic_side is not None:
        # Canonical normal axis is columns. Sparse row is far from the cut.
        ph, nw = (h, w) if diagnostic_side in ("L", "R") else (w, h)
        q = np.zeros((ph, nw), dtype=bool)
        spine = nw-1 if diagnostic_side in ("L", "T") else 0
        q[:, spine] = True
        bridge = int(rng.integers(1, ph))
        q[bridge, :] = True
        for y in range(1, ph):
            if y != bridge and rng.random() < .6:
                length = int(rng.integers(1, nw+1))
                if spine == 0:
                    q[y, :length] = True
                else:
                    q[y, nw-length:] = True
        return q if diagnostic_side in ("L", "R") else q.T.copy()
    if mode == "ring":
        m[0, :] = m[-1, :] = True
        m[:, 0] = m[:, -1] = True
    elif mode == "u":
        m[:, 0] = m[:, -1] = True
        m[-1, :] = True
    elif mode == "c":
        m[:, 0] = True
        m[0, :] = m[-1, :] = True
    elif mode == "h":
        m[:, 0] = m[:, -1] = True
        m[int(rng.integers(h)), :] = True
    else:
        # A monotone crooked stroke spans the box; connected growth adds branches.
        steps = [0]*(h-1) + [1]*(w-1)
        rng.shuffle(steps)
        y = x = 0
        m[y, x] = True
        for step in steps:
            y += step == 0
            x += step == 1
            m[y, x] = True
        target = int(rng.integers(int(m.sum()), h*w+1))
        while int(m.sum()) < target:
            candidates = []
            for y in range(h):
                for x in range(w):
                    if not m[y, x] and any(0 <= yy < h and 0 <= xx < w and m[yy, xx]
                                           for yy, xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1))):
                        candidates.append((y, x))
            m[_pick(rng, candidates)] = True
    if rng.random() < .5:
        m = np.flipud(m)
    if rng.random() < .5:
        m = np.fliplr(m)
    return m.copy()


def _make_pair(rng, bg, palette, stage):
    """0: one seam; 1: three arms; 2: free training; 3: larger networks."""
    for _ in range(80):
        diagnostic = None
        if stage == 0:
            side = _pick(rng, _SIDES)
            normal, span = int(rng.integers(5, 9)), int(rng.integers(3, 8))
            h, w = (span, normal) if side in ("L", "R") else (normal, span)
            boxes = [(0, h-1, 0, w-1)]
            thin, span2 = int(rng.integers(1, 3)), int(rng.integers(1, span+1))
            h2, w2 = (span2, thin) if side in ("L", "R") else (thin, span2)
            boxes.append(_attach(rng, boxes[0], side, h2, w2))
            diagnostic = side
        else:
            plan = int(rng.integers(3)) if stage == 3 else None
            h, w = int(rng.integers(5, 9)), int(rng.integers(5, 9))
            if stage == 3 and rng.random() < .4:
                if rng.random() < .5:
                    h = int(rng.integers(3, 5))
                else:
                    w = int(rng.integers(3, 5))
            boxes = [(0, h-1, 0, w-1)]
            if stage == 1 or (stage == 3 and plan < 2):
                missing = _pick(rng, _SIDES) if stage == 1 or plan == 1 else None
                sides = [s for s in _SIDES if s != missing]
                rng.shuffle(sides)
                for side in sides:
                    span = int(rng.integers(1, (h if side in ("L", "R") else w)+1))
                    normal = int(rng.integers(1, 7))
                    hh, ww = (span, normal) if side in ("L", "R") else (normal, span)
                    boxes.append(_attach(rng, boxes[0], side, hh, ww, contained=True))
            if stage == 3:
                target = int(rng.integers(5 if plan == 0 else 6, 9))
                boxes = _grow(rng, boxes, target, 4)
                if len(boxes) < target:
                    continue
            elif stage == 2:
                boxes = _grow(rng, boxes, int(rng.integers(2, 6)), 3)
        if not _valid_boxes(boxes, 4 if stage == 3 else 3):
            continue
        if stage == 0 or rng.random() < .65:
            with_decoy = _decoy(rng, boxes)
            if with_decoy is None:
                continue
            boxes = with_decoy
        if not _relations(boxes):
            continue
        bound = _bounds(boxes)
        # Translation, independent margins, and canvas aspect ratio all vary.
        room_h = 30 - (bound[1]-bound[0]+1)
        room_w = 30 - (bound[3]-bound[2]+1)
        mt = int(rng.integers(1 if stage == 0 else 0, min(3, room_h)+1))
        mb = int(rng.integers(0, min(3, room_h-mt)+1))
        ml = int(rng.integers(0, min(3, room_w)+1))
        mr = int(rng.integers(0, min(3, room_w-ml)+1))
        shifted = [(a-bound[0]+mt, b-bound[0]+mt, c-bound[2]+ml, d-bound[2]+ml)
                   for a,b,c,d in boxes]
        gh, gw = bound[1]-bound[0]+1+mt+mb, bound[3]-bound[2]+1+ml+mr
        a = np.full((gh, gw), bg, dtype=np.int8)
        colors = [int(_pick(rng, palette)) for _ in boxes]
        if stage == 0:
            colors[1] = int(_pick(rng, [c for c in palette if c != colors[0]]))
            colors[-1] = colors[0]  # Same color, separate object, no seam edge.
        for i, (r0, r1, c0, c1) in enumerate(shifted):
            mode = "ring" if stage == 1 and i == 0 else _pick(rng, ("solid", "ring", "u", "c", "h", "growth"))
            mask = _mask(rng, r1-r0+1, c1-c0+1, mode, diagnostic if i == 0 else None)
            a[r0:r1+1, c0:c1+1][mask] = colors[i]
        if np.count_nonzero(a == bg) * 2 <= a.size:
            continue
        objects = _objects(a, bg)
        if len(objects) != len(boxes):
            continue
        edges = _edges(a, bg, objects)
        faces = _faces(len(objects), edges)
        if stage == 3 and not (max(map(len, faces)) == 4 or sum(bool(s) for s in faces) >= 6):
            continue
        # Output is defined on the constructed original snapshot, never iterative.
        out = _render(a, bg, objects, edges)
        if stage == 0 and np.array_equal(out, _render(a, bg, objects, edges, mode="pixels")):
            continue
        return {"input": a, "output": out}
    raise RuntimeError("bounded seam layout construction failed")


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    roles = [int(x) for x in rng.permutation([c for c in range(10) if c != DIVIDER])]
    bg, palette = roles[0], roles[1:]
    # A small palette deliberately permits repeated colors in disconnected pieces.
    if rng.random() < .5:
        palette = palette[:int(rng.integers(2, 5))]
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            stage = (min(i, 2) if split == "train" else 3)
            for _ in range(20):
                pair = _make_pair(rng, bg, palette, stage)
                key = _key(pair["input"])
                if key not in used:
                    break
            else:
                raise RuntimeError("could not generate distinct inputs")
            used.add(key)
            puzzle[split].append(pair)
    # Shuffle demonstrations so the rule never depends on example position.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(pair):
    a, out = _grid(pair["input"]), _grid(pair["output"])
    _require(a.shape == out.shape, "canvas changed")
    _require(not np.any(a == DIVIDER), "reserved divider in input")
    bg = _background(a)
    objects = _objects(a, bg)
    _require(len(objects) >= 2, "too few objects")
    boxes = [o["box"] for o in objects]
    for i, b in enumerate(boxes):
        for c in boxes[i+1:]:
            _require(b[1]+1 < c[0] or c[1]+1 < b[0]
                     or b[3]+1 < c[2] or c[3]+1 < b[2],
                     "object boxes lack their separating background gap")
    edges = _edges(a, bg, objects)
    _require(bool(edges), "no qualifying seam")
    faces = _faces(len(objects), edges)
    line_mask = np.zeros(a.shape, dtype=bool)
    for axis, line in {(e[0], e[1]) for e in edges}:
        if axis == "V":
            line_mask[:, line] = True
        else:
            line_mask[line, :] = True
    _require(np.all(a[line_mask] == bg), "seams must be completely empty originally")
    _require(np.array_equal(out == DIVIDER, line_mask), "divider extent or location wrong")
    erase_mask = np.zeros(a.shape, dtype=bool)
    deleted, untouched = 0, 0
    # Cellwise geometric validation is separate from _render's vector masks.
    for obj, sides in zip(objects, faces):
        r0, r1, c0, c1 = obj["box"]
        removed = 0
        for y, x in obj["cells"]:
            cut = (("L" in sides and x-c0 < 2) or ("R" in sides and c1-x < 2)
                   or ("T" in sides and y-r0 < 2) or ("B" in sides and r1-y < 2))
            erase_mask[y, x] = cut
            _require(int(out[y, x]) == (bg if cut else obj["color"]), "incorrect face cut")
            removed += cut
        deleted += removed == len(obj["cells"])
        untouched += not sides
    unchanged = ~(line_mask | erase_mask)
    _require(np.array_equal(a[unchanged], out[unchanged]), "unrelated cells changed")
    _require(np.all(out[erase_mask] == bg), "peeled material was moved or recolored")
    _require(np.any(erase_mask), "no actual peel")
    return {"background": bg, "objects": objects, "edges": edges, "faces": faces,
            "count": len(objects), "active_sides": max(map(len, faces)), "deleted": deleted,
            "participating": sum(bool(s) for s in faces),
            "isolated": untouched, "seams": len({e[:2] for e in edges}),
            "orientations": {e[0] for e in edges},
            "row_witness": not np.array_equal(out, _render(a, bg, objects, edges, mode="pixels")),
            "depth_one": not np.array_equal(out, _render(a, bg, objects, edges, depth=1)),
            "depth_three": not np.array_equal(out, _render(a, bg, objects, edges, depth=3))}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split types")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    used, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            k = _key(_grid(pair["input"]))
            _require(k not in used, "repeated input")
            used.add(k)
            info = _inspect(pair)
            infos[split].append(info)
            a, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            _require(np.array_equal(solve(a), out), "solve mismatch")
            _require(not np.array_equal(out, _render(a, info["background"], info["objects"], info["edges"], mode="no_peel")), "A omission has no effect")
    train = infos["train"]
    _require(len({i["background"] for s in infos.values() for i in s}) == 1, "background convention varies")
    _require(any(i["row_witness"] for i in train), "no bounding-box versus occupied-pixel witness")
    _require(any(i["depth_one"] and i["depth_three"] for i in train), "cut thickness not distinguished")
    _require(any(i["deleted"] for i in train), "no narrow-object deletion demonstration")
    _require(any(i["isolated"] for i in train), "no unpaired piece demonstration")
    _require(any(i["objects"][e[2]]["color"] != i["objects"][e[4]]["color"]
                 for i in train for e in i["edges"]), "no unequal-color seam demonstration")
    _require(any(not sides and any(other_sides and obj["color"] == other["color"]
                                 for other, other_sides in zip(i["objects"], i["faces"]))
                 for i in train for obj, sides in zip(i["objects"], i["faces"])),
             "no repeated-color isolated-object witness")
    _require(any(i["active_sides"] == 3 for i in train), "no multi-face junction demonstration")
    _require(all(i["active_sides"] <= 3 for i in train), "training shift lost")
    _require(all(i["active_sides"] == 4 or i["participating"] >= 6 for i in infos["test"]),
             "test lacks a larger network or four-face combination")
    _require(all(i["participating"] <= 5 for i in train), "training network size shift lost")
    _require(set.union(*(i["orientations"] for i in train)) == {"V", "H"}, "missing training orientation")
    _require(any(not np.array_equal(np.asarray(p["output"]), _render(np.asarray(p["input"]), i["background"], i["objects"], i["edges"], mode="all_left"))
                 for p, i in zip(puzzle["train"], train)), "B-directed selection has no effect")


def self_test():
    """Bounded checks on 42 seeds, all explicit counts, and omission witnesses."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, shapes, structures, backgrounds = set(), set(), set(), set()
    counts, seam_counts, side_sets, deleted, isolated = set(), set(), set(), 0, 0
    test_layouts = set()
    for p in puzzles:
        signature = []
        for split in ("train", "test"):
            for pair in p[split]:
                a = pair["input"]
                i = _inspect(pair)
                if split == "test":
                    test_layouts.add("four_sides" if i["active_sides"] == 4 else "extended")
                shapes.add(a.shape)
                backgrounds.add(i["background"])
                counts.add(i["count"])
                seam_counts.add(i["seams"])
                side_sets.update(tuple(sorted(s)) for s in i["faces"])
                deleted += i["deleted"]
                isolated += i["isolated"]
                # Geometry signature ignores palette and canvas margins.
                boxes = [o["box"] for o in i["objects"]]
                r0, _, c0, _ = _bounds(boxes)
                structures.add(tuple(sorted(tuple(sorted((int(y-r0), int(x-c0)) for y,x in o["cells"])) for o in i["objects"])))
                signature.append((_key(a), _key(pair["output"])))
        signatures.add(tuple(signature))
    _require(len(signatures) == 42 and len(structures) >= 180, "insufficient structural variation")
    _require(len(shapes) >= 50 and len(backgrounds) >= 7, "size or palette roles too narrow")
    _require({3, 4, 5, 6} <= counts and {1, 3, 4} <= seam_counts, "network counts too narrow")
    _require(all((s,) in side_sets for s in _SIDES), "a cut direction is absent")
    _require(tuple(sorted(_SIDES)) in side_sets and () in side_sets, "junction or isolation absent")
    _require(deleted >= 42 and isolated >= 42, "deletion or selection coverage absent")
    _require(test_layouts == {"four_sides", "extended"}, "test network layouts too narrow")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(12345, nt, ns)
            _require(len(p["train"]) == nt and len(p["test"]) == ns, "counts ignored")
    for nt, ns in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(4.0,1),(4,None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p, q = generate_puzzle(271828, 7, 3), generate_puzzle(271828, 7, 3)
    _require(all(_key(a[k]) == _key(b[k]) for s in p for a,b in zip(p[s], q[s]) for k in a), "nondeterministic")
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        a, out = pair["input"], pair["output"]
        original = a.copy()
        solve(a)
        _require(np.array_equal(a, original), "solve mutated its input")
        for turns in range(4):
            rotated = np.rot90(a, turns)
            _require(np.array_equal(solve(rotated), np.rot90(out, turns)), "rotation covariance failed")
        _require(np.array_equal(solve(np.fliplr(a)), np.fliplr(out)), "reflection covariance failed")
        mapping = np.arange(10, dtype=np.int8)
        colors = [c for c in range(10) if c != DIVIDER]
        mapping[colors] = colors[1:] + colors[:1]
        _require(np.array_equal(solve(mapping[a]), mapping[out]), "semantic palette permutation failed")
    # Exact fixtures also cover valid scenes not deliberately sampled by attachment.
    # A distant blocker invalidates an otherwise facing one-cell gap at column 7.
    a = np.zeros((22, 22), dtype=np.int8)
    for r0, r1, c0, c1, color in ((3,5,2,6,7),(3,5,8,11,8),(12,14,6,8,1),
                                  (17,19,14,17,2),(17,19,19,20,4)):
        a[r0:r1+1, c0:c1+1] = color
    expected = a.copy()
    expected[17:20, 16:18] = 0
    expected[17:20, 19:21] = 0
    expected[:, 18] = DIVIDER
    _require(np.array_equal(solve(a), expected), "blocked-gap or wider-gap fixture failed")
    # Two neighbors induce the same seam and the same face cut once, not twice.
    a = np.zeros((16, 17), dtype=np.int8)
    a[2:11, 2:7], a[2:5, 8:12], a[6:11, 8:13] = 7, 8, 9
    expected = a.copy()
    expected[2:11, 5:7] = 0
    expected[2:5, 8:10], expected[6:11, 8:10] = 0, 0
    expected[:, 7] = DIVIDER
    _require(np.array_equal(solve(a), expected), "duplicate seam or repeated face fixture failed")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "geometries": len(structures),
            "canvas_shapes": len(shapes), "backgrounds": len(backgrounds),
            "object_counts": sorted(counts), "seam_counts": sorted(seam_counts),
            "test_layouts": sorted(test_layouts),
            "deleted_objects": int(deleted), "isolated_objects": int(isolated)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
