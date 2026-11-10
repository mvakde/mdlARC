"""Seam-spliced corner stamps: local reflections meet a global region mask.

The solver reads only the query grid. Generation paints from sampled geometry;
validation reparses that geometry and also tests deliberately simpler rules.
"""
from __future__ import annotations

import json
from collections import Counter
import numpy as np


DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid dimensions")
    _check(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _check(bool(np.all((a >= 0) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    h, w = mask.shape
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if diagonal:
        directions = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, group = [first], {first}
        while stack:
            y, x = stack.pop()
            for dy, dx in directions:
                p = y + dy, x + dx
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _partition(cut):
    groups = _components(~cut)
    _check(len(groups) == 2, "seam must separate exactly two regions")
    labels = np.full(cut.shape, -1, dtype=np.int8)
    for index, group in enumerate(groups):
        for p in group:
            labels[p] = index
    # An 8-connected digital path, suppressing diagonal shortcuts at elbows.
    cells = {tuple(map(int, p)) for p in np.argwhere(cut)}
    graph = {}
    for y, x in cells:
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dy or dx) or (y + dy, x + dx) not in cells:
                    continue
                if dy and dx and ((y + dy, x) in cells or (y, x + dx) in cells):
                    continue
                neighbors.append((y + dy, x + dx))
        graph[y, x] = neighbors
    ends = [p for p in cells if len(graph[p]) == 1]
    _check(len(ends) == 2 and all(1 <= len(v) <= 2 for v in graph.values()), "branched seam")
    h, w = cut.shape
    border = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _check(all(border(p) == (p in ends) for p in cells), "seam boundary contacts")
    _check(len(_components(cut, True)) == 1, "disconnected seam")
    return labels


def _near(mask, radius=1):
    result = np.zeros_like(mask)
    h, w = mask.shape
    for y, x in np.argwhere(mask):
        result[max(0, y-radius):min(h, y+radius+1), max(0, x-radius):min(w, x+radius+1)] = True
    return result


def _positions(anchor, pattern, convention="reflect"):
    y, x, q = anchor
    dy, dx = DIRS[q]
    if convention == "rotate" and q in (1, 2):
        pattern = {(v, u) for u, v in pattern}
    if convention == "translate":
        dy = dx = 1
    return {(y + dy*u, x + dx*v) for u, v in pattern}


def _corners(rect):
    t, l, b, r = rect
    return [(t, l, 0), (t, r, 1), (b, l, 2), (b, r, 3)]


def _outline(rect):
    t, l, b, r = rect
    return ({(t, x) for x in range(l, r+1)} | {(b, x) for x in range(l, r+1)} |
            {(y, l) for y in range(t, b+1)} | {(y, r) for y in range(t, b+1)})


def _bounds(points):
    ys, xs = zip(*points)
    return min(ys), min(xs), max(ys), max(xs)


def _disjoint(a, b, gap=1):
    return a[2]+gap < b[0] or b[2]+gap < a[0] or a[3]+gap < b[1] or b[3]+gap < a[1]


def _l_anchor(cells):
    t, l, b, r = _bounds(cells)
    _check(3 <= b-t <= 5 and 3 <= r-l <= 5, "reference arm length")
    choices = []
    for y, x, q in _corners((t, l, b, r)):
        expected = {(y, c) for c in range(l, r+1)} | {(a, x) for a in range(t, b+1)}
        if cells == expected:
            choices.append((y, x, q))
    _check(len(choices) == 1, "reference support is not an L")
    return choices[0]


def _find_frames(a, frame, cut_color, pegs):
    ys = list(map(int, np.flatnonzero(np.sum(a == frame, axis=1) >= 3)))
    xs = list(map(int, np.flatnonzero(np.sum(a == frame, axis=0) >= 3)))
    rects = []
    allowed = (a == frame) | (a == cut_color)
    for p in pegs:
        allowed[p] = True
    for i, t in enumerate(ys):
        for b in ys[i+1:]:
            if b-t < 6:
                continue
            for j, l in enumerate(xs):
                for r in xs[j+1:]:
                    if r-l < 6:
                        continue
                    if not (np.all(allowed[t, l:r+1]) and np.all(allowed[b, l:r+1]) and
                            np.all(allowed[t:b+1, l]) and np.all(allowed[t:b+1, r])):
                        continue
                    if any(a[y, x] == cut_color for y, x, _ in _corners((t, l, b, r))):
                        continue
                    rects.append((t, l, b, r))
    _check(rects and all(_disjoint(x, y, 0) for i, x in enumerate(rects) for y in rects[:i]),
           "missing or overlapping receiving frames")
    structure = set().union(*(_outline(r) for r in rects))
    _check(all(tuple(p) in structure for p in np.argwhere(a == frame)), "unexplained frame pixels")
    corners = [c for r in rects for c in _corners(r)]
    _check(pegs <= {(y, x) for y, x, _ in corners}, "peg is not a frame corner")
    return rects, [c for c in corners if c[:2] in pegs]


def _read(a):
    """Infer all roles, two observed stencils, anchors and marker from input."""
    a = _grid(a)
    colors, counts = np.unique(a, return_counts=True)
    _check(len(colors) == 6, "six distinct semantic colors required")
    _check(int(np.sum(counts == np.max(counts))) == 1, "background frequency tie")
    bg = int(colors[np.argmax(counts)])
    cuts = []
    for color in map(int, colors):
        if color == bg:
            continue
        mask = a == color
        try:
            labels = _partition(mask)
            cuts.append((color, mask, labels))
        except AssertionError:
            pass
    _check(len(cuts) == 1, "seam role is ambiguous")
    cut_color, cut, labels = cuts[0]
    interpretations = []
    for support in map(int, colors):
        if support in (bg, cut_color):
            continue
        try:
            groups = _components(a == support)
            large = [g for g in groups if len(g) > 1]
            pegs = set().union(*(g for g in groups if len(g) == 1))
            _check(len(large) == 2 and pegs, "two reference brackets and receiving pegs required")
            refs = [_l_anchor(g) for g in large]
            templates, source_refs, reference_points = {}, {}, set()
            for ref in refs:
                y, x, q = ref
                dy, dx = DIRS[q]
                data = [(int(a[y+u, x+v]), (u*dy, v*dx), (y+u, x+v))
                        for u in range(-2, 3) for v in range(-2, 3)
                        if u and v and 0 <= y+u < a.shape[0] and 0 <= x+v < a.shape[1]
                        and a[y+u, x+v] not in (bg, support)]
                role_colors = {c for c, _, _ in data}
                _check(len(role_colors) == 1, "reference stencil is not monochrome")
                color = role_colors.pop()
                _check(color not in (cut_color, *templates), "reference colors must differ")
                pattern = frozenset(p for _, p, _ in data)
                _check(2 <= len(pattern) <= 12, "reference size")
                templates[color] = pattern
                source_refs[color] = ref
                reference_points |= {p for _, _, p in data}
            _check(len(set(templates.values())) == 2, "source stencils must differ")
            remaining = set(map(int, colors)) - {bg, cut_color, support, *templates}
            _check(len(remaining) == 1, "missing frame role")
            frame = remaining.pop()
            rects, anchors = _find_frames(a, frame, cut_color, pegs)
            for rect in rects:
                border = _outline(rect)
                _check(not any(cut[y, x] and cut[y1, x1] for y, x in border
                               for y1, x1 in ((y+1, x), (y, x+1)) if (y1, x1) in border),
                       "seam runs along a frame edge")
            ref_boxes = []
            for g, ref in zip(large, refs):
                y, x, _ = ref
                points = set(g) | {(y-2, x-2), (y+2, x+2)}
                for color, pattern in templates.items():
                    actual = _positions(ref, pattern)
                    if all(a[p] == color for p in actual):
                        points |= actual
                ref_boxes.append(_bounds(points))
            boxes = ref_boxes + [(t-2, l-2, b+2, r+2) for t, l, b, r in rects]
            _check(all(0 <= t <= b < a.shape[0] and 0 <= l <= r < a.shape[1]
                       for t, l, b, r in boxes), "clipped object window")
            _check(all(_disjoint(x, y, 0) for i, x in enumerate(boxes) for y in boxes[:i]),
                   "object neighborhoods overlap")
            for box in ref_boxes:
                t, l, b, r = box
                _check(not np.any(cut[t:b+1, l:r+1]), "seam crosses a reference")
            windows = set().union(*({(y+u, x+v) for u in range(-2, 3) for v in range(-2, 3)
                                      if u and v and 0 <= y+u < a.shape[0] and 0 <= x+v < a.shape[1]}
                                     for y, x, _ in anchors))
            source_points = {tuple(map(int, p)) for p in np.argwhere(np.isin(a, list(templates)))}
            leftover = source_points - reference_points - windows
            _check(len(leftover) == 1, "one isolated source-colored marker required")
            marker = leftover.pop()
            _check(not _near(cut)[marker], "marker touches seam")
            _check(not any(t-1 <= marker[0] <= b+1 and l-1 <= marker[1] <= r+1 for t, l, b, r in boxes),
                   "marker must be outside object boxes")
            primary = int(a[marker])
            other = next(c for c in templates if c != primary)
            info = dict(background=bg, frame=frame, support=support, cut_color=cut_color,
                        cut=cut, labels=labels, marker=marker, side=int(labels[marker]),
                        primary=primary, other=other, templates=templates, refs=refs,
                        source_refs=source_refs, rects=rects, anchors=anchors, boxes=boxes)
            expected = _paint(a, info)
            for p in source_points & windows:
                _check(a[p] == expected[p], "fragment contradicts completed hybrid")
            interpretations.append(info)
        except (AssertionError, IndexError):
            continue
    _check(len(interpretations) == 1, "ambiguous or malformed stamp scene")
    return interpretations[0]


def _paint(a, d, mode="rule"):
    """Stamp using global coordinates; alternate modes are shortcut controls."""
    out = a.copy()
    p, q = d["primary"], d["other"]
    side = d["side"]
    if mode == "fixed_side_0":
        side = 0
    elif mode == "fixed_side_1":
        side = 1
    elif mode == "reverse_regions":
        side = 1-side
    if mode == "min_color":
        p, q = min(p, q), max(p, q)
    elif mode == "max_color":
        p, q = max(p, q), min(p, q)
    elif mode in ("left_source", "right_source"):
        order = sorted((p, q), key=lambda c: (d["source_refs"][c][1], d["source_refs"][c][0]))
        p, q = order if mode == "left_source" else order[::-1]
    for anchor in d["anchors"]:
        convention = mode if mode in ("rotate", "translate") else "reflect"
        positions = []
        for color in (p, q):
            pattern = d["templates"][color]
            ref_q = d["source_refs"][color][2]
            if mode == "rotate" and ref_q in (1, 2):
                pattern = {(v, u) for u, v in pattern}
            elif mode == "translate":
                dy, dx = DIRS[ref_q]
                pattern = {(dy*u, dx*v) for u, v in pattern}
            positions.append(_positions(anchor, pattern, convention))
        first, second = positions
        center_side = d["labels"][anchor[:2]]
        for point in first | second:
            y, x = point
            if not (0 <= y < a.shape[0] and 0 <= x < a.shape[1]):
                continue
            region = int(d["labels"][point])
            if mode == "union":
                value = p if point in first else q
            elif mode == "whole_corner":
                selected = p if center_side == side else q
                value = selected if point in (first if selected == p else second) else None
            elif region == -1:
                if mode == "blank_seam":
                    value = None
                elif mode == "reverse_priority":
                    value = q if point in second else p
                else:
                    value = p if point in first else q
            elif region == side:
                value = p if point in first else None
            else:
                value = q if point in second else None
            if value is not None:
                out[point] = value
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    return _paint(a, _read(a))


def _line(p, q):
    y, x = p
    y1, x1 = q
    dy, dx = abs(y1-y), abs(x1-x)
    sy, sx = (1 if y < y1 else -1), (1 if x < x1 else -1)
    error, cells = dx-dy, []
    while True:
        cells.append((y, x))
        if (y, x) == q:
            return cells
        twice = 2*error
        if twice > -dy:
            error -= dy
            x += sx
        if twice < dx:
            error += dx
            y += sy


def _make_cut(rng, shape, kind):
    h, w = shape
    cut = np.zeros(shape, dtype=bool)
    if kind == "straight":
        cut[:, int(rng.integers(5, w-5))] = True
    elif kind == "diagonal":
        for p in _line((0, int(rng.integers(3, w-3))), (h-1, int(rng.integers(3, w-3)))):
            cut[p] = True
    elif kind == "elbow":
        y, x = int(rng.integers(6, h-6)), int(rng.integers(6, w-6))
        cut[:y+1, x] = True
        cut[y, x:] = True
    else:
        steps = 1 if kind == "step" else 2
        rows = sorted(map(int, rng.choice(np.arange(5, h-5, 4), steps, replace=False)))
        col, last = int(rng.integers(5, w-5)), 0
        for row in rows:
            choices = [x for x in range(5, w-5) if abs(x-col) >= 3]
            nxt = int(rng.choice(choices))
            cut[last:row+1, col] = True
            cut[row, min(col, nxt):max(col, nxt)+1] = True
            last, col = row, nxt
        cut[last:, col] = True
    if rng.random() < .5:
        cut = cut[::-1, :].copy()
    if rng.random() < .5:
        cut = cut[:, ::-1].copy()
    if rng.random() < .5:
        cut = cut.T.copy()
    _partition(cut)
    return cut


def _pattern(rng, radius):
    domain = [(u, v) for u in range(-radius, radius+1) for v in range(-radius, radius+1) if u and v]
    for attempt in range(64):
        mode = int(rng.integers(4))
        if mode == 0:
            # A filled quadrant, short diagonal, or two opposite blocks.
            su, sv = int(rng.choice((-1, 1))), int(rng.choice((-1, 1)))
            p = {(u, v) for u, v in domain if u*su > 0 and v*sv > 0}
            if rng.random() < .6:
                p |= {(-u, -v) for u, v in p}
        elif mode == 1:
            p = {domain[int(i)] for i in rng.choice(len(domain), int(rng.integers(2, min(10, len(domain))+1)), replace=False)}
            p |= {(v, u) for u, v in p}
        elif mode == 2:
            p = {(u, v) for u, v in domain if abs(u) == abs(v)}
            if rng.random() < .5:
                p = {(u, v) for u, v in p if u*v > 0}
        else:
            count = int(rng.integers(2, min(10, len(domain))+1))
            p = {domain[int(i)] for i in rng.choice(len(domain), count, replace=False)}
        if 2 <= len(p) <= 12:
            return frozenset(p)
    return frozenset({(-1, -1), (1, 1)})


def _scene(rng, kind, side, palette, target_count=1, fragments=True, strong=False):
    radius = 2 if strong or rng.random() < .75 else 1
    shape = (int(rng.integers(24 if radius == 2 else 21, 31)),
             int(rng.integers(24 if radius == 2 else 21, 31)))
    cut = _make_cut(rng, shape, kind)
    shape = cut.shape
    h, w = shape
    labels = _partition(cut)
    patterns = [_pattern(rng, radius), _pattern(rng, radius)]
    _check(patterns[0] != patterns[1], "identical source stencils")
    bg, frame, support, cc, ca, cb = palette
    primary = ca if rng.random() < .5 else cb
    secondary = cb if primary == ca else ca
    templates = {ca: patterns[0], cb: patterns[1]}
    rects, boxes, anchors = [], [], []
    inner_cut = [tuple(map(int, p)) for p in np.argwhere(cut)
                 if 3 <= p[0] < h-3 and 3 <= p[1] < w-3]
    for index in range(target_count):
        for attempt in range(90):
            sh, sw = int(rng.integers(7, 13)), int(rng.integers(7, 13))
            if index == 0:
                q = int(rng.integers(4))
                dy, dx = DIRS[q]
                y, x = inner_cut[int(rng.integers(len(inner_cut)))]
                u, v = sorted(patterns[0] | patterns[1])[int(rng.integers(len(patterns[0] | patterns[1])))]
                y, x = y-dy*u, x-dx*v
                t, l = (y if dy == 1 else y-sh+1), (x if dx == 1 else x-sw+1)
            else:
                t, l = int(rng.integers(2, h-sh-1)), int(rng.integers(2, w-sw-1))
            rect = t, l, t+sh-1, l+sw-1
            box = t-2, l-2, t+sh+1, l+sw+1
            if not (0 <= box[0] and box[2] < h and 0 <= box[1] and box[3] < w):
                continue
            if not all(_disjoint(box, b) for b in boxes):
                continue
            corners = _corners(rect)
            if any(cut[c[:2]] for c in corners):
                continue
            # No seam segment may run along a receiving-frame edge.
            if any(cut[y, x] and cut[y1, x1] for y, x in _outline(rect)
                   for y1, x1 in ((y+1, x), (y, x+1)) if (y1, x1) in _outline(rect)):
                continue
            if index == 0:
                crossing = [c for c in corners if len({int(labels[p]) for p in
                            _positions(c, patterns[0] | patterns[1])} - {-1}) == 2]
                if not crossing:
                    continue
                chosen = [crossing[int(rng.integers(len(crossing)))]]
                # Adjacent corners expose the axis-reflection convention.
                if strong:
                    chosen.append(corners[chosen[0][2] ^ int(rng.choice((1, 2)))])
            else:
                chosen = []
            for c in corners:
                if c not in chosen and rng.random() < (.65 if strong else .45):
                    chosen.append(c)
            if not chosen and rng.random() < .8:
                chosen = [corners[int(rng.integers(4))]]
            rects.append(rect)
            boxes.append(box)
            anchors.extend(chosen)
            break
        else:
            raise AssertionError("receiving layout did not fit")
    refs, supports = [], []
    for pattern in patterns:
        for attempt in range(220):
            q = int(rng.integers(4))
            y, x = int(rng.integers(2, h-2)), int(rng.integers(2, w-2))
            dy, dx = DIRS[q]
            arm_y, arm_x = int(rng.integers(3, 6)), int(rng.integers(3, 6))
            ref = y, x, q
            cells = {(y+dy*k, x) for k in range(arm_y+1)} | {(y, x+dx*k) for k in range(arm_x+1)}
            box = _bounds(cells | {(y-2, x-2), (y+2, x+2)})
            t, l, b, r = box
            if not (0 <= t and b < h and 0 <= l and r < w):
                continue
            if not all(_disjoint(box, old) for old in boxes) or np.any(cut[t:b+1, l:r+1]):
                continue
            boxes.append(box)
            refs.append(ref)
            supports.append(cells)
            break
        else:
            raise AssertionError("reference layout did not fit")
    a = np.full(shape, bg, dtype=np.int8)
    for rect in rects:
        for p in _outline(rect):
            a[p] = frame
    a[cut] = cc
    for ref, cells, color, pattern in zip(refs, supports, (ca, cb), patterns):
        for p in cells:
            a[p] = support
        for p in _positions(ref, pattern):
            a[p] = color
    for y, x, _ in anchors:
        a[y, x] = support
    candidates = [tuple(map(int, p)) for p in np.argwhere((labels == side) & ~_near(cut))
                  if not any(t-1 <= p[0] <= b+1 and l-1 <= p[1] <= r+1 for t, l, b, r in boxes)]
    _check(candidates, "no marker site")
    marker = candidates[int(rng.integers(len(candidates)))]
    a[marker] = primary
    # Ground truth is constructed from the sampled local stencils and labels.
    out = a.copy()
    for anchor in anchors:
        first, second = _positions(anchor, templates[primary]), _positions(anchor, templates[secondary])
        for p in first | second:
            if labels[p] == -1:
                out[p] = primary if p in first else secondary
            elif labels[p] == side and p in first:
                out[p] = primary
            elif labels[p] != side and p in second:
                out[p] = secondary
    if fragments and rng.random() < .6:
        additions = [tuple(map(int, p)) for p in np.argwhere((a != out) & ~cut)]
        rng.shuffle(additions)
        count = int(rng.integers(0, max(1, len(additions)//2)+1))
        for p in additions[:count]:
            a[p] = out[p]
    _check(int(np.argmax(np.bincount(a.ravel(), minlength=10))) == bg, "background not dominant")
    return {"input": a, "output": out}


SHORTCUTS = ("union", "whole_corner", "translate", "rotate", "blank_seam",
             "reverse_priority", "fixed_side_0", "fixed_side_1", "reverse_regions",
             "min_color", "max_color", "left_source", "right_source")


def _flags(pair, d=None):
    a, out = pair["input"], pair["output"]
    d = _read(a) if d is None else d
    flags = {m for m in SHORTCUTS if not np.array_equal(_paint(a, d, m), out)}
    p, q = d["primary"], d["other"]
    for anchor in d["anchors"]:
        first = _positions(anchor, d["templates"][p])
        second = _positions(anchor, d["templates"][q])
        for point in first | second:
            if d["cut"][point] and point in second and point not in first:
                flags.add("seam_fallback")
    for color in (p, q):
        if np.any((a != out) & (out == color)):
            flags.add("adds_both_" + ("primary" if color == p else "other"))
    return flags


def _key(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    palette = tuple(map(int, rng.choice(10, 6, replace=False)))
    stable = rng.random() < .25
    first_side = int(rng.integers(2))
    required = set(SHORTCUTS) | {"seam_fallback", "adds_both_primary", "adds_both_other"}
    # Resample the small demonstration set if it leaves a listed shortcut alive.
    for teaching_attempt in range(60):
        puzzle, seen, coverage = {"train": [], "test": []}, set(), set()
        for index in range(num_train):
            kind = str(rng.choice(("straight", "diagonal"))) if index == 0 else (
                "elbow" if index == 1 else str(rng.choice(("straight", "diagonal", "elbow", "step"))))
            side = first_side if index == 0 else 1-first_side if index == 1 else int(rng.integers(2))
            for attempt in range(500):
                roles = palette if stable else tuple(map(int, rng.choice(10, 6, replace=False)))
                try:
                    pair = _scene(rng, kind, side, roles, target_count=1,
                                  fragments=index >= 2, strong=index < 2)
                    d = _read(pair["input"])
                    _check(np.array_equal(_paint(pair["input"], d), pair["output"]), "parser mismatch")
                    _check(_key(pair["input"]) not in seen and np.any(pair["input"] != pair["output"]), "duplicate or identity")
                    flags = _flags(pair, d)
                    if index < 2:
                        _check({"union", "whole_corner", "reverse_priority"} <= flags, "weak integrated witness")
                    break
                except AssertionError:
                    continue
            else:
                raise RuntimeError("bounded training construction exhausted")
            puzzle["train"].append(pair)
            seen.add(_key(pair["input"]))
            coverage |= flags
        if required <= coverage:
            break
    else:
        raise RuntimeError("could not teach all conventions")
    for index in range(num_test):
        for attempt in range(700):
            roles = palette if stable else tuple(map(int, rng.choice(10, 6, replace=False)))
            try:
                pair = _scene(rng, "double_step", int(rng.integers(2)), roles,
                              target_count=int(rng.choice((1, 1, 2))), fragments=True)
                _check(_key(pair["input"]) not in seen, "duplicate test")
                _check(np.array_equal(solve(pair["input"]), pair["output"]), "test parser mismatch")
                _check(np.any(pair["input"] != pair["output"]), "identity test")
                break
            except AssertionError:
                continue
        else:
            raise RuntimeError("bounded test construction exhausted")
        puzzle["test"].append(pair)
        seen.add(_key(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "split type")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "counts")
    seen, coverage, sides = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _check(a.shape == out.shape and np.any(a != out), "identity or changed dimensions")
            _check(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            d = _read(a)
            _check(np.array_equal(_paint(a, d), out), "incorrect completion or changed protected cell")
            changed = a != out
            _check(np.all(np.isin(a[changed], (d["background"], d["cut_color"]))), "erased an input fragment")
            _check(np.all(np.isin(out[changed], (d["primary"], d["other"]))), "wrong new colors")
            _check(set(np.unique(a)) == set(np.unique(out)), "palette changed")
            _check(any(len({int(d["labels"][p]) for p in _positions(c, set().union(*d["templates"].values()))} - {-1}) == 2
                       for c in d["anchors"]), "no corner spans both regions")
            if split == "train":
                coverage |= _flags({"input": a, "output": out}, d)
                sides.add(d["side"])
    _check(set(SHORTCUTS) | {"seam_fallback", "adds_both_primary", "adds_both_other"} <= coverage,
           "demonstrations leave a simpler alternative unexplained")
    _check(sides == {0, 1}, "marker side not demonstrated")


def _signature(puzzle):
    return tuple(_key(np.asarray(p[k], dtype=np.int8)) for s in ("train", "test")
                 for p in puzzle[s] for k in ("input", "output"))


def self_test():
    signatures, shapes, patterns, layouts = set(), set(), set(), set()
    stats = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        if seed < 2:
            _check(_signature(puzzle) == _signature(generate_puzzle(seed)), "nondeterminism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                _check(np.array_equal(solve(a.copy()), out), "solve disagreement")
                d = _read(a)
                shapes.add(a.shape)
                patterns.update(tuple(sorted(p)) for p in d["templates"].values())
                layouts.add((a.shape, tuple(d["rects"]), tuple(d["anchors"]), d["cut"].tobytes()))
                stats["two_frames"] += len(d["rects"]) == 2
                windows = set().union(*(_positions(c, {(u, v) for u in range(-2, 3)
                                      for v in range(-2, 3) if u and v}) for c in d["anchors"]))
                stats["fragments"] += any(a[p] in d["templates"] for p in windows)
                stats["radius_one"] += any(max(max(abs(u), abs(v)) for u, v in p) == 1 for p in d["templates"].values())
                stats["asymmetric"] += any(p != frozenset((v, u) for u, v in p) for p in d["templates"].values())
        # Geometric reflections must commute with the complete rule.
        a, out = puzzle["train"][0]["input"], puzzle["train"][0]["output"]
        for axis in (0, 1):
            _check(np.array_equal(solve(np.flip(a, axis)), np.flip(out, axis)), "reflection equivariance")
        # Color renaming also commutes, including black in any role.
        permutation = np.random.default_rng(seed+700).permutation(10)
        _check(np.array_equal(solve(permutation[a]), permutation[out]), "color equivariance")
        # Relocate the marker to the opposite side without touching any object.
        d = _read(a)
        swapped = a.copy()
        swapped[d["marker"]] = d["background"]
        sites = [tuple(map(int, p)) for p in np.argwhere((d["labels"] == 1-d["side"]) & ~_near(d["cut"]))
                 if not any(t-1 <= p[0] <= b+1 and l-1 <= p[1] <= r+1 for t, l, b, r in d["boxes"])]
        _check(bool(sites), "opposite marker control unavailable")
        swapped[sites[0]] = d["primary"]
        answer = solve(swapped)
        stamp_cells = set().union(*(_positions(c, set().union(*d["templates"].values()))
                                   for c in d["anchors"]))
        _check(any(answer[p] != out[p] for p in stamp_cells if not d["cut"][p]), "marker has no effect")
        _check(np.array_equal(answer[d["cut"]], out[d["cut"]]), "marker position changed seam priority")
        recolored = a.copy()
        recolored[d["marker"]] = d["other"]
        changed_primary = solve(recolored)
        _check(any(changed_primary[p] != out[p] for p in stamp_cells if d["cut"][p]),
               "marker color did not change collision priority")
    _check(len(signatures) == 42 and len(shapes) >= 25 and len(patterns) >= 100 and len(layouts) >= 180,
           "structural variation collapsed")
    _check(stats["two_frames"] > 0 and stats["radius_one"] > 0 and stats["asymmetric"] > 100 and stats["fragments"] > 0,
           "supported variation absent")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1200+nt*5+nq, nt, nq)
            _check(len(p["train"]) == nt and len(p["test"]) == nq, "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(99, 2, 1)
    bad["train"][0]["output"][0, 0] = (int(bad["train"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "stencils": len(patterns), "layouts": len(layouts), "features": dict(stats)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
