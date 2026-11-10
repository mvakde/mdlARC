"""Exact-span rectangle anchors driving registered neutral stencils.

No parent code or artifacts are loaded at runtime. Generation and validation
are bounded and do not write files. See task_spec.md for the complete rule.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numbers
import numpy as np

_AXES = (1, 2, 3, 4, 6, 8)
_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _runs(row, bg):
    runs, c = [], 0
    while c < len(row):
        color = int(row[c])
        end = c + 1
        while end < len(row) and int(row[end]) == color:
            end += 1
        if color != bg:
            runs.append((color, c, end))
        c = end
    return runs


def _rectangles(coarse, bg, mode="exact"):
    """Return (color, top, bottom-exclusive, left, right-exclusive)."""
    runs = [_runs(row, bg) for row in coarse]
    if mode == "pixels":
        return [(int(coarse[r, c]), r, r + 1, c, c + 1)
                for r, c in np.argwhere(coarse != bg)]
    if mode == "rows":
        return [(color, r, r + 1, left, right)
                for r, row in enumerate(runs) for color, left, right in row]
    if mode == "overlap":
        # Deliberately wrong ablation: connected overlapping runs -> bounding box.
        nodes = [(color, r, r + 1, left, right)
                 for r, row in enumerate(runs) for color, left, right in row]
        remaining = set(range(len(nodes)))
        result = []
        while remaining:
            start = min(remaining)
            remaining.remove(start)
            group, stack = [nodes[start]], [start]
            while stack:
                i = stack.pop()
                color, r, _, left, right = nodes[i]
                neighbors = [j for j in sorted(remaining)
                             if nodes[j][0] == color and abs(nodes[j][1] - r) == 1
                             and max(left, nodes[j][3]) < min(right, nodes[j][4])]
                for j in neighbors:
                    remaining.remove(j)
                    stack.append(j)
                    group.append(nodes[j])
            result.append((group[0][0], min(x[1] for x in group), max(x[2] for x in group),
                           min(x[3] for x in group), max(x[4] for x in group)))
        return result
    _require(mode == "exact", "unknown rectangle mode")
    result = []
    for r, row in enumerate(runs):
        before = set(runs[r - 1]) if r else set()
        for color, left, right in row:
            if (color, left, right) in before:
                continue
            end = r + 1
            while end < len(runs) and (color, left, right) in runs[end]:
                end += 1
            result.append((color, r, end, left, right))
    return result


def _box(rect, shape):
    _, top, bottom, left, right = rect
    py, px = 24 // shape[0], 24 // shape[1]
    return 2 + top * py, bottom * py, 2 + left * px, right * px


def _center(rect, shape):
    y0, y1, x0, x1 = _box(rect, shape)
    return (y0 + y1 - 1) // 2, (x0 + x1 - 1) // 2


def _components(cells):
    remaining, components = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in remaining:
                        remaining.remove(q)
                        component.add(q)
                        stack.append(q)
        components.append(component)
    return components


def _parse_instructions(panel, bg, neutral):
    components = _components(map(tuple, np.argwhere(panel == neutral)))
    _require(bool(components), "no neutral stencil")
    candidates = [[] for _ in components]
    near = {}
    for i, component in enumerate(components):
        for r, c in component:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    near.setdefault((r + dr, c + dc), set()).add(i)
    h, w = panel.shape
    for sr, sc in np.argwhere((panel != bg) & (panel != neutral)):
        sr, sc = int(sr), int(sc)
        source = int(panel[sr, sc])
        for dr, dc in _DIRECTIONS:
            tr, tc, qr, qc = sr + dr, sc + dc, sr + 2 * dr, sc + 2 * dc
            if not (0 <= qr < h and 0 <= qc < w):
                continue
            target = int(panel[tr, tc])
            if target in (bg, neutral, source) or int(panel[qr, qc]) not in (bg, neutral):
                continue
            for i in sorted(near.get((qr, qc), ())):
                candidates[i].append({"source": source, "target": target, "q": (qr, qc),
                                      "direction": (dr, dc), "labels": ((sr, sc), (tr, tc)),
                                      "cells": frozenset(components[i]),
                                      "offsets": tuple(sorted((int(r - qr), int(c - qc))
                                                              for r, c in components[i]))})
    _require(all(len(x) == 1 for x in candidates), "ambiguous or missing stencil labels")
    records = [x[0] for x in candidates]
    labels = [p for rec in records for p in rec["labels"]]
    _require(len(labels) == len(set(labels)), "shared instruction labels")
    _require(len({r["source"] for r in records}) == len(records), "ambiguous source instruction")
    occupied = set(labels) | set().union(*(set(r["cells"]) for r in records))
    _require(occupied == set(map(tuple, np.argwhere(panel != bg))), "unassigned legend ink")
    return records


def _decode(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    columns = [c for c in range(a.shape[1]) if int(a[0, c]) != bg
               and np.all(a[:, c] == a[0, c])]
    _require(len(columns) == 1, "expected one solid neutral divider")
    divider = columns[0]
    lattice, panel = a[:, :divider], a[:, divider + 1:]
    _require(min(lattice.shape) >= 3 and all(n % 2 for n in lattice.shape), "invalid lattice shape")
    _require(panel.shape[1] >= 3 and np.all(panel[:, 0] == bg), "legend needs a blank left collar")
    _require(np.all(lattice[::2, :] == bg) and np.all(lattice[:, ::2] == bg), "invalid lattice scaffold")
    neutral = int(a[0, divider])
    coarse = lattice[1::2, 1::2].copy()
    _require(all(n in _AXES for n in coarse.shape), "coarse axes must divide 24 with positive cores")
    _require(np.any(coarse != bg) and not np.any(coarse == neutral), "invalid lattice payload")
    records = _parse_instructions(panel, bg, neutral)
    _require(all(np.any(coarse == r["source"]) for r in records), "unused source instruction")
    return coarse, bg, neutral, records


def _compose(coarse, bg, records, mode="exact", registration="labels", stamping=True):
    """Paint from a fixed set of rectangles and original instructions only."""
    rectangles = _rectangles(coarse, bg, mode)
    mapping = {r["source"]: r for r in records}
    out = np.full((26, 26), bg, dtype=np.int8)
    paint, visits, clipped, anchors = {}, Counter(), 0, []
    # Keep inactive rendering exact even in the deliberate anchor ablations:
    # their answer differences must be caused by changed stamp sites.
    for rect in _rectangles(coarse, bg):
        color = rect[0]
        if color not in mapping or not stamping:
            y0, y1, x0, x1 = _box(rect, coarse.shape)
            out[y0:y1, x0:x1] = color
    for rect in rectangles:
        color = rect[0]
        if color not in mapping or not stamping:
            continue
        rec = mapping[color]
        ar, ac = _center(rect, coarse.shape)
        anchors.append((color, ar, ac))
        offsets = rec["offsets"]
        if registration == "mask_center":
            rmid = (min(r for r, c in offsets) + max(r for r, c in offsets)) // 2
            cmid = (min(c for r, c in offsets) + max(c for r, c in offsets)) // 2
            offsets = tuple((r - rmid, c - cmid) for r, c in offsets)
        for dr, dc in offsets:
            p = (ar + dr, ac + dc)
            if not (0 <= p[0] < 26 and 0 <= p[1] < 26):
                clipped += 1
                continue
            _require(p not in paint or paint[p] == rec["target"], "conflicting target ink")
            paint[p] = rec["target"]
            visits[p] += 1
    for p, color in paint.items():
        out[p] = color
    return out, {"rectangles": rectangles, "anchors": anchors, "paint": paint,
                 "clipped": clipped, "overlap": sum(n > 1 for n in visits.values())}


def solve(input_grid: np.ndarray) -> np.ndarray:
    coarse, bg, _, records = _decode(input_grid)
    return _compose(coarse, bg, records)[0]


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _rotate(offsets, direction):
    dr, dc = direction
    return tuple(sorted((r * dc + c * dr, -r * dr + c * dc) for r, c in offsets))


def _registered_mask(rng, wanted, small=False):
    # Generate geometry, then independently enumerate legal registrations.
    for _ in range(50):
        kind = _choose(rng, ("line", "elbow", "branch", "outline", "growth", "chevron"))
        if wanted == "hole":
            kind = "c"
        if small:
            kind = "line"
        length = int(rng.integers(3, 8)) if not small else int(rng.integers(1, 4))
        h, w = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        if kind == "line":
            cells = {(0, c) for c in range(length)}
        elif kind == "elbow":
            cells = {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}
        elif kind == "branch":
            cells = {(r, 1) for r in range(h + 1)} | {(h // 2, c) for c in range(w + 1)}
        elif kind == "outline":
            cells = {(r, c) for r in range(h + 1) for c in range(w + 1)
                     if r in (0, h) or c in (0, w)}
        elif kind == "chevron":
            cells = {(r, h - abs(r)) for r in range(-h, h + 1)}
        elif kind == "c":
            cells = {(r, w - 1) for r in range(-h, h + 1)}
            cells |= {(r, c) for r in (-h, h) for c in range(-1, w)}
        else:
            cells = {(0, 0)}
            for _ in range(int(rng.integers(4, 16))):
                r, c = _choose(rng, sorted(cells))
                dr, dc = _choose(rng, _DIRECTIONS)
                if -3 <= r + dr <= 3 and 0 <= c + dc <= 4:
                    cells.add((r + dr, c + dc))
        near = {(r + dr, c + dc) for r, c in cells
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        options = []
        for qr, qc in sorted(near):
            if (qr, qc - 1) in cells or (qr, qc - 2) in cells:
                continue
            if (qr, qc - 3) in near:
                continue
            offsets = tuple(sorted((r - qr, c - qc) for r, c in cells))
            hole = (0, 0) not in offsets
            if wanted == "hole" and not hole:
                continue
            if wanted == "occupied" and hole:
                continue
            if wanted == "offcenter" and (hole or len(offsets) < 3 or
                    (sum(r for r, c in offsets) == 0 and sum(c for r, c in offsets) == 0)):
                continue
            options.append(offsets)
        if options:
            return _rotate(_choose(rng, options), _choose(rng, _DIRECTIONS))
    raise ValueError("bounded mask construction exhausted")


def _place_legend(rng, shape, bg, neutral, specs):
    panel = np.full(shape, bg, dtype=np.int8)
    reserved, halo = set(), set()
    records = []
    for source, target, direction, offsets in specs:
        dr, dc = direction
        relative = set(offsets) | {(0, 0), (-dr, -dc), (-2 * dr, -2 * dc)}
        rlo, rhi = min(r for r, c in relative), max(r for r, c in relative)
        clo, chi = min(c for r, c in relative), max(c for r, c in relative)
        positions = [(r, c) for r in range(-rlo, shape[0] - rhi)
                     for c in range(max(1, 1 - clo), shape[1] - chi)]
        rng.shuffle(positions)
        for qr, qc in positions:
            mask = {(qr + r, qc + c) for r, c in offsets}
            labels = ((qr - 2 * dr, qc - 2 * dc), (qr - dr, qc - dc))
            cells = mask | set(labels) | {(qr, qc)}
            if cells & reserved or mask & halo:
                continue
            candidate = panel.copy()
            for p in mask:
                candidate[p] = neutral
            candidate[labels[0]], candidate[labels[1]] = source, target
            try:
                parsed = _parse_instructions(candidate, bg, neutral)
            except ValueError:
                continue
            matches = [r for r in parsed if r["source"] == source and r["q"] == (qr, qc)
                       and r["offsets"] == offsets]
            if len(parsed) != len(records) + 1 or len(matches) != 1:
                continue
            panel, records = candidate, parsed
            reserved |= cells
            halo |= {(r + a, c + b) for r, c in mask for a in (-1, 0, 1) for b in (-1, 0, 1)}
            break
        else:
            raise ValueError("legend does not fit uniquely")
    return panel, records


def _paint_run(coarse, row, span, color, bg):
    left, right = span
    coarse[row, left:right] = color
    if left and int(coarse[row, left - 1]) == color:
        coarse[row, left - 1] = bg
    if right < coarse.shape[1] and int(coarse[row, right]) == color:
        coarse[row, right] = bg


def _coarse_sample(rng, shape, bg, sources, inert, depth):
    density = float(_choose(rng, (.12, .25, .45, .65)))
    colors = sources + inert
    coarse = np.full(shape, bg, dtype=np.int8)
    occupied = rng.random(shape) < density
    coarse[occupied] = rng.choice(colors, size=int(occupied.sum()))
    rows, cols = shape
    # Random run operations supplement independent occupancy.
    for _ in range(int(rng.integers(1, 5))):
        r, left = int(rng.integers(rows)), int(rng.integers(cols))
        right = int(rng.integers(left + 1, cols + 1))
        _paint_run(coarse, r, (left, right), _choose(rng, colors), bg)
    if depth:
        top = int(rng.integers(rows - depth))
        spans = [(a, b) for a in range(cols) for b in range(a + 1, cols + 1)]
        choices = [(a, b) for a in spans if a[1] - a[0] >= 2 for b in spans
                   if a != b and max(a[0], b[0]) < min(a[1], b[1])]
        first, changed = _choose(rng, choices)
        # Clear full motif rows so the two spans cannot grow accidentally.
        for r in range(top, top + depth + 1):
            coarse[r, :] = bg
            _paint_run(coarse, r, first if r < top + depth else changed, sources[0], bg)
    for source in sources:
        if not np.any(coarse == source):
            positions = list(map(tuple, np.argwhere(coarse == bg)))
            if not positions:
                raise ValueError("no room for source")
            coarse[_choose(rng, positions)] = source
    if inert and not any(np.any(coarse == color) for color in inert):
        positions = list(map(tuple, np.argwhere(coarse == bg)))
        if not positions:
            raise ValueError("no room for inactive calibration rectangle")
        coarse[_choose(rng, positions)] = inert[0]
    return coarse


def _features(coarse, bg, records):
    active = {r["source"] for r in records}
    runs = [_runs(row, bg) for row in coarse]
    horizontal = any(c in active and right - left > 1 for row in runs for c, left, right in row)
    exact = any(c in active and (c, left, right) in lower
                for upper, lower in zip(runs, runs[1:]) for c, left, right in upper)
    partial = any(ca == cb and ca in active and (la, ra) != (lb, rb)
                  and max(la, lb) < min(ra, rb)
                  for upper, lower in zip(runs, runs[1:])
                  for ca, la, ra in upper for cb, lb, rb in lower)
    depth = max((b - t for c, t, b, l, r in _rectangles(coarse, bg) if c in active), default=0)
    return {"horizontal": horizontal, "exact": exact, "partial": partial, "depth": depth}


def _differs(coarse, bg, records, out, **kwargs):
    try:
        return not np.array_equal(_compose(coarse, bg, records, **kwargs)[0], out)
    except ValueError:
        # A wrong grouping may itself create an incompatible-ink conflict.
        return True


def _pair(rng, palette, role):
    bg, neutral = palette[:2]
    colors = list(palette[2:])
    for _ in range(160):
        try:
            teaching = role in ("occupied", "hole")
            shifted = role == "test"
            rows = int(_choose(rng, (4, 6, 8) if teaching or shifted else (2, 3, 4, 6, 8)))
            cols = int(_choose(rng, (3, 4, 6, 8)))
            count = 2 if role == "hole" else int(_choose(rng, (1, 1, 2, 2, 3)))
            if rows == 2:
                count = 1
            sources = colors[:count] if teaching else [int(x) for x in rng.choice(colors, size=count, replace=False)]
            target_pool = [x for x in colors if x not in sources]
            mode = _choose(rng, ("independent", "chain", "shared", "cycle"))
            if role == "occupied":
                targets = [colors[3]] + [target_pool[-1]] * (count - 1)
            elif role == "hole":
                targets = [sources[1], _choose(rng, [x for x in colors if x not in sources])]
            elif mode == "cycle" and count > 1:
                targets = sources[1:] + sources[:1]
            elif mode == "chain" and count > 1:
                targets = sources[1:] + [target_pool[0]]
            elif mode == "shared":
                targets = [target_pool[0]] * count
            else:
                targets = [int(_choose(rng, [x for x in colors if x != s])) for s in sources]
            specs = []
            for i, (source, target) in enumerate(zip(sources, targets)):
                wanted = ("offcenter" if role == "occupied" and i == 0 else
                          "hole" if role == "hole" and i == 0 else
                          _choose(rng, ("occupied", "occupied", "hole", "free")))
                offsets = _registered_mask(rng, wanted, small=rows == 2)
                # Rotation of the visible mask is already reflected in offsets;
                # choose the matching label ray by testing every direction.
                directions = [d for d in _DIRECTIONS if (-d[0], -d[1]) not in offsets
                              and (-2 * d[0], -2 * d[1]) not in offsets]
                rng.shuffle(directions)
                accepted = None
                for direction in directions:
                    cells = set(offsets)
                    near = {(r + a, c + b) for r, c in cells
                            for a in (-1, 0, 1) for b in (-1, 0, 1)}
                    if (-3 * direction[0], -3 * direction[1]) not in near:
                        accepted = direction
                        break
                if accepted is None:
                    raise ValueError("no unique label ray")
                specs.append((source, target, accepted, offsets))
            panel_width = int(rng.integers(8, min(17, 28 - 2 * cols) + 1))
            panel, records = _place_legend(rng, (2 * rows + 1, panel_width), bg, neutral, specs)
            candidates = [x for x in colors if x not in sources]
            inert = [int(x) for x in rng.choice(candidates, size=int(rng.integers(1, min(3, len(candidates)) + 1)), replace=False)]
            depth = int(rng.integers(3, rows)) if shifted else 2 if teaching else 0
            coarse = _coarse_sample(rng, (rows, cols), bg, sources, inert, depth)
            features = _features(coarse, bg, records)
            if not shifted and features["depth"] > 2:
                continue
            if teaching and not all(features[k] for k in ("horizontal", "exact", "partial")):
                continue
            if shifted and not (features["depth"] >= 3 and features["partial"]):
                continue
            out, info = _compose(coarse, bg, records)
            if not info["paint"] or any(not any(a[0] == s for a in info["anchors"]) for s in sources):
                continue
            if teaching:
                if role == "occupied" and not info["clipped"]:
                    continue
                if not all(_differs(coarse, bg, records, out, mode=mode)
                           for mode in ("rows", "overlap", "pixels")):
                    continue
                if not _differs(coarse, bg, records, out, registration="mask_center"):
                    continue
                inactive_boxes = [_box(rect, coarse.shape) for rect in info["rectangles"]
                                  if rect[0] not in sources]
                if not any(all((r, c) not in info["paint"]
                               for r in range(y0, y1) for c in range(x0, x1))
                           for y0, y1, x0, x1 in inactive_boxes):
                    continue
                # At least one complete, isolated main-source stamp makes
                # registration observable rather than fully hidden by clipping.
                main = next(r for r in records if r["source"] == sources[0])
                if not any(all(0 <= ar + dr < 26 and 0 <= ac + dc < 26 and
                               out[ar + dr, ac + dc] == main["target"]
                               for dr, dc in main["offsets"])
                           for s, ar, ac in info["anchors"] if s == sources[0]):
                    continue
                # Show the source rectangle really disappears, including
                # erased centers for a hole registration.
                if role == "hole" and not any(out[ar, ac] == bg for s, ar, ac in info["anchors"]
                                              if s == sources[0]):
                    continue
            lattice = np.full((2 * rows + 1, 2 * cols + 1), bg, dtype=np.int8)
            lattice[1::2, 1::2] = coarse
            divider = np.full((lattice.shape[0], 1), neutral, dtype=np.int8)
            inp = np.concatenate((lattice, divider, panel), axis=1)
            if not np.array_equal(solve(inp), out):
                raise AssertionError("construction and input decoder disagree")
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise RuntimeError("bounded pair generation exhausted")


def _integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(name + " must be an integer")
    value = int(value)
    _require(value >= low and (high is None or value <= high), name + " out of range")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 0, None, "seed")
    nt, ns = _integer(num_train, 2, 10, "num_train"), _integer(num_test, 1, 4, "num_test")
    _require(nt > ns, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = [int(x) for x in rng.permutation(10)]
    puzzle, seen = {"train": [], "test": []}, set()
    roles = ["occupied", "hole"] + ["free"] * (nt - 2)
    rng.shuffle(roles)
    for split, pair_roles in (("train", roles), ("test", ["test"] * ns)):
        for role in pair_roles:
            for _ in range(12):
                pair = _pair(rng, palette, role)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness retries exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "invalid splits")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid counts")
    seen, conventions, training = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            coarse, bg, neutral, records = _decode(inp)
            conventions.add((bg, neutral))
            expected, info = _compose(coarse, bg, records)
            _require(np.array_equal(expected, out), "incorrect composed output")
            _require(out.shape == (26, 26), "wrong canvas")
            _require(bool(info["paint"]), "no target paint")
            # Rectangles partition the occupied coarse slots exactly once.
            cover = np.zeros(coarse.shape, dtype=int)
            for color, top, bottom, left, right in info["rectangles"]:
                _require(np.all(coarse[top:bottom, left:right] == color), "invalid rectangle")
                cover[top:bottom, left:right] += 1
            _require(np.array_equal(cover, (coarse != bg).astype(int)), "rectangle partition failure")
            for color, ar, ac in info["anchors"]:
                _require(0 <= ar < 26 and 0 <= ac < 26, "bad rectangle center")
            for p, color in info["paint"].items():
                _require(int(out[p]) == color, "paint omitted")
            features = _features(coarse, bg, records)
            if split == "train":
                _require(features["depth"] <= 2, "training depth shift violated")
                training.append((coarse, bg, records, out, features, info))
            else:
                _require(features["depth"] >= 3 and features["partial"], "test lacks depth/composition shift")
    _require(len(conventions) == 1, "background/neutral convention changed")
    _require(any(all(f[k] for k in ("horizontal", "exact", "partial")) for _, _, _, _, f, _ in training),
             "training lacks span-equality contrast")
    for mode in ("rows", "overlap", "pixels"):
        _require(any(_differs(c, b, r, o, mode=mode) for c, b, r, o, _, _ in training),
                 "training does not distinguish " + mode)
    _require(any(_differs(c, b, r, o, registration="mask_center") for c, b, r, o, _, _ in training),
             "registration not observable")
    _require(any(_differs(c, b, r, o, stamping=False) for c, b, r, o, _, _ in training),
             "stencil contribution not observable")
    _require(any(info["clipped"] for c, b, r, o, f, info in training), "missing clipping evidence")
    _require(any(any((0, 0) not in rec["offsets"] for rec in r) and
                 any(o[ar, ac] == b for _, ar, ac in info["anchors"])
                 for c, b, r, o, f, info in training), "missing empty-registration evidence")
    _require(any(any((0, 0) in rec["offsets"] and
                        (sum(dr for dr, dc in rec["offsets"]) != 0 or
                         sum(dc for dr, dc in rec["offsets"]) != 0) for rec in r)
                 for c, b, r, o, f, info in training), "missing off-center evidence")
    _require(any(len({rec["source"] for rec in r} & {rec["target"] for rec in r}) > 0
                 for c, b, r, o, f, info in training), "no simultaneous color-role evidence")


def _manual_regressions():
    # Hand-calculated 4x4 A geometry; a translated line is one column right
    # of its empty registration. Inactive 3 remains a 4x4 rectangle.
    coarse = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 3]], dtype=np.int8)
    panel = np.zeros((9, 10), dtype=np.int8)
    panel[3:6, 6] = 8
    panel[4, 3:5] = (1, 2)
    lattice = np.zeros((9, 9), dtype=np.int8)
    lattice[1::2, 1::2] = coarse
    inp = np.concatenate((lattice, np.full((9, 1), 8, dtype=np.int8), panel), axis=1)
    expected = np.zeros((26, 26), dtype=np.int8)
    expected[5:8, 7] = 2
    expected[14:17, 13] = 2
    expected[20:24, 20:24] = 3
    assert np.array_equal(solve(inp), expected)
    c, b, n, records = _decode(inp)
    assert not np.array_equal(_compose(c, b, records, mode="rows")[0], expected)
    assert not np.array_equal(_compose(c, b, records, mode="overlap")[0], expected)
    assert not np.array_equal(_compose(c, b, records, registration="mask_center")[0], expected)
    # An 8x8 single coarse slot renders to a 1x1 rectangle centered at (2,2).
    edge = np.zeros((8, 8), dtype=np.int8)
    edge[0, 0] = 1
    rec = {"source": 1, "target": 2, "offsets": tuple((r, 0) for r in range(-4, 1))}
    result, info = _compose(edge, 0, [rec])
    expected_edge = np.zeros((26, 26), dtype=np.int8)
    expected_edge[0:3, 2] = 2
    assert np.array_equal(result, expected_edge) and info["clipped"] == 2
    # All anchors survive being covered by another stamp; generated source
    # colors are never used as an additional instruction pass.
    linked = np.array([[1, 2, 0], [0, 0, 0]], dtype=np.int8)
    a = {"source": 1, "target": 2, "offsets": ((0, 0), (0, 8))}
    z = {"source": 2, "target": 3, "offsets": ((1, 0),)}
    result, _ = _compose(linked, 0, [a, z])
    assert result[6, 4] == 2 and result[6, 12] == 2 and result[7, 12] == 3
    assert result[7, 4] == 0
    # Wider solver domain: a one-by-one coarse lattice has center (12,12).
    tiny_lattice = np.zeros((3, 3), dtype=np.int8)
    tiny_lattice[1, 1] = 1
    tiny_panel = np.zeros((3, 7), dtype=np.int8)
    tiny_panel[1, 2:5] = (1, 2, 8)
    tiny_input = np.concatenate((tiny_lattice, np.full((3, 1), 8, dtype=np.int8), tiny_panel), axis=1)
    tiny_expected = np.zeros((26, 26), dtype=np.int8)
    tiny_expected[12, 12] = 2
    assert np.array_equal(solve(tiny_input), tiny_expected)
    # Inactive rectangles are preserved except where simultaneous ink lands.
    overlay = np.array([[1, 3, 0], [0, 0, 0]], dtype=np.int8)
    result, _ = _compose(overlay, 0, [{"source": 1, "target": 2, "offsets": ((0, 8),)}])
    assert result[6, 12] == 2 and result[6, 11] == 3 and result[7, 12] == 3
    # Unlike-target overlap is invalid, not subject to iteration precedence.
    try:
        _compose(linked, 0, [a, {"source": 2, "target": 3, "offsets": ((0, 0),)}])
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting ink was accepted")


def self_test():
    """Stream 42 seeds; retain only hashes and bounded measurement sets."""
    _manual_regressions()
    shapes, coarse_shapes, masks, directions, backgrounds, neutrals = set(), set(), set(), set(), set(), set()
    structures, hashes, depths, rectangle_counts, instruction_counts = set(), set(), set(), set(), set()
    counters = Counter()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp.tolist()), out)
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                coarse, bg, neutral, records = _decode(inp)
                _, info = _compose(coarse, bg, records)
                shapes.add(inp.shape)
                coarse_shapes.add(coarse.shape)
                backgrounds.add(bg)
                neutrals.add(neutral)
                instruction_counts.add(len(records))
                rectangle_counts.add(len(info["rectangles"]))
                depths.add(_features(coarse, bg, records)["depth"])
                counters["clipped_pairs"] += bool(info["clipped"])
                counters["overlapping_pairs"] += bool(info["overlap"])
                counters["inactive_rectangles"] += sum(r[0] not in {rec["source"] for rec in records}
                                                       for r in info["rectangles"])
                counters["hole_instructions"] += sum((0, 0) not in r["offsets"] for r in records)
                rename = {bg: 0}
                normalized = tuple(rename.setdefault(int(x), len(rename)) for x in coarse.flat)
                structures.add((coarse.shape, normalized))
                for rec in records:
                    masks.add(rec["offsets"])
                    directions.add(rec["direction"])
        hashes.add(digest.hexdigest())
    assert len(hashes) == 42 and len(shapes) >= 20 and len(coarse_shapes) >= 10
    assert len(structures) >= 150 and len(masks) >= 50
    assert len(backgrounds) >= 8 and len(neutrals) >= 8 and len(directions) == 4
    assert {1, 2, 3} <= instruction_counts and len(rectangle_counts) >= 10
    assert max(depths) >= 5 and counters["clipped_pairs"] >= 5
    assert counters["overlapping_pairs"] >= 2 and counters["inactive_rectangles"] >= 30
    assert counters["hole_instructions"] >= 42
    first, second = generate_puzzle(98765, 2, 1), generate_puzzle(98765, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in ("train", "test")
               for a, b in zip(first[s], second[s]) for k in ("input", "output"))
    # Every legal explicit count combination, in addition to host checks.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(5000 + 10 * nt + ns, nt, ns)
            assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    bad = generate_puzzle(123, 2, 1)
    bad["test"][0]["output"] = np.zeros((26, 26), dtype=np.int8)
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    return {"seeds": 42, "distinct_puzzles": len(hashes), "input_shapes": len(shapes),
            "coarse_shapes": len(coarse_shapes), "normalized_coarse_layouts": len(structures),
            "registered_masks": len(masks), "directions": len(directions),
            "backgrounds": len(backgrounds), "neutral_colors": len(neutrals),
            "rectangle_counts": sorted(rectangle_counts), "instruction_counts": sorted(instruction_counts),
            "stack_depths": sorted(depths), **dict(counters)}
