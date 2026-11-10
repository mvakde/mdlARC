"""Room-indexed compass docking. Standard library + NumPy; no file I/O.

The exported solver reads only the supplied grid. Construction uses its own
completed wall maps, then checks the resulting observations with that solver.
Self-tests retain bounded sets of signatures rather than generated datasets.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # top, right, bottom, left
_N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and a.dtype.kind != "b", "integer grid required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "grid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _digest(a):
    return hashlib.sha256(bytes(a.shape) + a.astype(np.int8).tobytes()).hexdigest()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = _N8 if diagonal else _DIRS
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        found, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _transform(cells, t):
    result = []
    for (r, c), value in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        result.append((r, c, int(value)))
    r0, c0 = min(x[0] for x in result), min(x[1] for x in result)
    return {(r - r0, c - c0): value for r, c, value in result}


def _signature(cells):
    return tuple(sorted(_transform(cells, 0).items()))


def _shift(cells, origin):
    return {(r + origin[0], c + origin[1]): value for (r, c), value in cells.items()}


def _halo(points):
    return {(r + dr, c + dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _bbox(points):
    return (min(r for r, _ in points), min(c for _, c in points),
            max(r for r, _ in points), max(c for _, c in points))


def _walls(shape, rooms):
    mask = np.zeros(shape, dtype=bool)
    for t, l, b, r in rooms:
        mask[t, l:r + 1] = mask[b, l:r + 1] = True
        mask[t:b + 1, l] = mask[t:b + 1, r] = True
    return mask


def _room_analysis(mask):
    """Flood count plus a separate planar cycle invariant, including T walls."""
    _require(len(_components(mask)) == 1, "disconnected wall network")
    h, w = mask.shape
    _require(bool(mask[0].all() and mask[-1].all()
                  and mask[:, 0].all() and mask[:, -1].all()), "incomplete outer wall")
    rooms = []
    for component in _components(np.pad(~mask, 1, constant_values=True)):
        if any(r in (0, h + 1) or c in (0, w + 1) for r, c in component):
            continue
        t, l, b, r = _bbox(component)
        _require(len(component) == (b - t + 1) * (r - l + 1), "nonrectangular elementary room")
        rooms.append((t - 2, l - 2, b, r))  # inclusive wall box in unpadded mask
    _require(bool(rooms), "no rooms")
    _require(np.array_equal(mask, _walls(mask.shape, rooms)), "unexplained wall geometry")
    e = int((mask[:, :-1] & mask[:, 1:]).sum())
    e += int((mask[:-1] & mask[1:]).sum())
    unit = int((mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]).sum())
    _require(e - int(mask.sum()) + 1 - unit == len(rooms), "flood/Euler disagreement")
    ys = np.flatnonzero(np.any(mask[:, :-1] & mask[:, 1:], axis=1))
    xs = np.flatnonzero(np.any(mask[:-1] & mask[1:], axis=0))
    lattice = (len(ys) - 1) * (len(xs) - 1)
    return len(rooms), rooms, lattice != len(rooms)


def _labels(cells, band_colors):
    core = {p: v for p, v in cells.items() if v not in band_colors}
    t, l, b, r = _bbox(core)
    labels = {}
    for (y, x), value in cells.items():
        if value not in band_colors:
            continue
        sides = [d for d, ok in enumerate((y < t, x > r, y > b, x < l)) if ok]
        _require(len(sides) == 1, "label must face one side")
        d = sides[0]
        _require((y == t - 1 and l <= x <= r) if d == 0 else
                 (x == r + 1 and t <= y <= b) if d == 1 else
                 (y == b + 1 and l <= x <= r) if d == 2 else
                 (x == l - 1 and t <= y <= b), "label too far from wall")
        labels[(y, x)] = d
    directions = set(labels.values())
    _require(len(directions) == 2 and (max(directions) - min(directions)) % 2 == 1,
             "two adjacent label sides required")
    values = []
    for d in directions:
        patch = [p for p in labels if labels[p] == d]
        _require(1 <= len(patch) <= 3, "label patch length")
        indices = sorted(p[1] if d % 2 == 0 else p[0] for p in patch)
        _require(indices == list(range(indices[0], indices[-1] + 1)), "fragmented label")
        cs = {cells[p] for p in patch}
        _require(len(cs) == 1, "mixed label identity")
        values.append(next(iter(cs)))
    _require(len(set(values)) == 2, "label identities must differ")
    return core, labels


def _parse(input_grid):
    g = _grid(input_grid)
    hist = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(hist))
    _require(int(np.count_nonzero(hist == hist[bg])) == 1, "ambiguous background")
    h, w = g.shape
    top = np.flatnonzero(g[0] != bg)
    _require(len(top) > 0, "missing boundary tapes")
    depth = int(top[0])
    _require(1 <= depth <= 8 and min(h, w) > 2 * depth, "bad tape depth")
    _require(top.tolist() == list(range(depth, w - depth)), "bad top tape extent")
    tapes = np.zeros_like(g, dtype=bool)
    tapes[:depth, depth:w - depth] = True
    tapes[h - depth:, depth:w - depth] = True
    tapes[depth:h - depth, :depth] = True
    tapes[depth:h - depth, w - depth:] = True
    field = np.zeros_like(tapes)
    field[depth:h - depth, depth:w - depth] = True
    _require(bool(np.all(g[~(tapes | field)] == bg)), "corners must be background")
    compasses = []
    for k in range(depth):
        stripes = (g[depth - 1 - k, depth:w - depth],
                   g[depth:h - depth, w - depth + k],
                   g[h - depth + k, depth:w - depth],
                   g[depth:h - depth, depth - 1 - k])
        compass = tuple(int(s[0]) for s in stripes)
        _require(all(bool(np.all(s == s[0])) for s in stripes), "nonuniform stripe")
        _require(len(set(compass)) == 4 and bg not in compass, "compass identities")
        compasses.append(compass)
    band_colors = set(compasses[0])
    _require(len(set(compasses)) == depth and all(set(c) == band_colors for c in compasses),
             "compass layers must differ and share identities")
    opposite = {frozenset((compasses[0][0], compasses[0][2])),
                frozenset((compasses[0][1], compasses[0][3]))}
    _require(all({frozenset((c[0], c[2])), frozenset((c[1], c[3]))} == opposite
                 for c in compasses), "layers must be D4 poses of one compass")
    sources, targets, reg_colors = [], [], set()
    for component in _components(field & (g != bg)):
        cells = {p: int(g[p]) for p in component}
        if len(set(cells.values())) == 1:
            targets.append(set(cells))
            reg_colors.update(cells.values())
        else:
            sources.append(cells)
    _require(bool(sources) and len(sources) == len(targets) and len(reg_colors) == 1,
             "source/stencil segmentation")
    reg = next(iter(reg_colors))
    _require(reg not in band_colors and reg != bg, "registration role")
    sizes = [len(x) for x in targets]
    _require(len(sizes) == len(set(sizes)), "registration sizes must distinguish objects")
    for target in targets:
        t, l, b, r = _bbox(target)
        _require(1 <= len(target) <= 4 and (t == b or l == r)
                 and len(target) == max(b - t, r - l) + 1, "straight registration stencil")
    core_hist = Counter(v for s in sources for v in s.values() if v not in band_colors and v != reg)
    _require(bool(core_hist), "missing wall color")
    ranked = core_hist.most_common()
    _require(len(ranked) <= 2 and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]), "wall/damage roles")
    body = ranked[0][0]
    damage = ranked[1][0] if len(ranked) == 2 else None
    info = {"grid": g, "background": bg, "depth": depth, "tapes": tapes,
            "field": field, "compasses": compasses, "band_colors": band_colors,
            "registration": reg, "body": body, "damage": damage,
            "sources": sources, "targets": targets, "analyses": []}
    for cells in sources:
        core, labels = _labels(cells, band_colors)
        _require(reg in core.values() and body in core.values(), "source roles")
        t, l, b, r = _bbox(core)
        mask = np.zeros((b - t + 1, r - l + 1), dtype=bool)
        for y, x in core:
            mask[y - t, x - l] = True
        count, rooms, partial = _room_analysis(mask)
        _require(count <= depth, "not enough compass layers")
        for (y, x), value in core.items():
            if value == damage:
                neighbors = [((y + dy, x + dx) in core) for dy, dx in _DIRS]
                _require(sum(neighbors) == 2 and ((neighbors[0] and neighbors[2])
                         or (neighbors[1] and neighbors[3])), "unsupported wall damage")
        marks = {p for p, value in core.items() if value == reg}
        _require(len(marks) in sizes, "missing matching target")
        info["analyses"].append({"count": count, "rooms": rooms, "partial": partial,
                                 "core_shape": mask.shape, "mask": mask, "marks": marks,
                                 "labels": labels, "damage_cells": sum(v == damage for v in core.values())})
    return info


def _candidates(info, index, use_labels=True, count_override=None):
    source = info["sources"][index]
    n = info["analyses"][index]["count"] if count_override is None else count_override
    compass = info["compasses"][n - 1]
    reg, body, damage = info["registration"], info["body"], info["damage"]
    repaired = {p: body if value == damage else value for p, value in source.items()}
    result, seen = [], set()
    for t in range(8):
        cells = _transform(repaired, t)
        _, labels = _labels(cells, info["band_colors"])
        if use_labels and any(cells[p] != compass[d] for p, d in labels.items()):
            continue
        marks = {p for p, value in cells.items() if value == reg}
        first = min(marks)
        for j, target in enumerate(info["targets"]):
            if len(target) != len(marks):
                continue
            anchor = min(target)
            delta = anchor[0] - first[0], anchor[1] - first[1]
            if {(y + delta[0], x + delta[1]) for y, x in marks} != target:
                continue
            placed = _shift(cells, delta)
            h, w = info["grid"].shape
            d = info["depth"]
            if any(not (d <= y < h - d and d <= x < w - d) for y, x in placed):
                continue
            signature = tuple(sorted(placed.items()))
            if signature not in seen:
                seen.add(signature)
                result.append((j, placed))
    return result


def _maps(info, use_labels=True, overrides=None, limit=2):
    overrides = {} if overrides is None else overrides
    choices = [_candidates(info, i, use_labels, overrides.get(i)) for i in range(len(info["sources"]))]
    order = sorted(range(len(choices)), key=lambda i: len(choices[i]))
    solutions = {}

    def search(k, used, occupied):
        if len(solutions) >= limit:
            return
        if k == len(order):
            solutions[tuple(sorted(occupied.items()))] = occupied
            return
        blocked = _halo(occupied)
        for target, cells in choices[order[k]]:
            if target not in used and not (set(cells) & blocked):
                search(k + 1, used | {target}, {**occupied, **cells})

    search(0, set(), {})
    return list(solutions.values())


def _render(info, cells):
    out = np.full_like(info["grid"], info["background"])
    out[info["tapes"]] = info["grid"][info["tapes"]]
    for p, value in cells.items():
        out[p] = value
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the unique repaired docking output entirely from input pixels."""
    info = _parse(input_grid)
    answers = _maps(info)
    _require(len(answers) == 1, "input does not determine one docking output")
    return _render(info, answers[0])


def _capacity(box):
    t, l, b, r = box
    return ((b - t) // 2) * ((r - l) // 2)


def _partition(rng, h, w, n, partial=False):
    boxes = [(0, 0, h - 1, w - 1)]
    if _capacity(boxes[0]) < n:
        return None
    # Occasionally use genuine strips; recursive splits cover mixed geometries.
    if not partial and n > 1 and rng.random() < .3:
        axes = [a for a, length in enumerate((h, w)) if length >= 2 * n + 1]
        if axes:
            axis = int(rng.choice(axes))
            length = (h, w)[axis] - 1
            gaps = [2] * n
            for _ in range(length - 2 * n):
                gaps[int(rng.integers(n))] += 1
            cuts = [0] + list(np.cumsum(gaps).astype(int))
            boxes = [(cuts[i], 0, cuts[i + 1], w - 1) if axis == 0 else
                     (0, cuts[i], h - 1, cuts[i + 1]) for i in range(n)]
            return _walls((h, w), boxes)
    first_axis = None
    for step in range(n - 1):
        options = []
        for i, (t, l, b, r) in enumerate(boxes):
            for axis, start, end in ((0, t, b), (1, l, r)):
                if partial and step == 1 and axis == first_axis:
                    continue
                for cut in range(start + 2, end - 1):
                    halves = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else (
                        (t, l, b, cut), (t, cut, b, r))
                    cap = sum(_capacity(x) for j, x in enumerate(boxes) if j != i)
                    if cap + sum(_capacity(x) for x in halves) >= n:
                        options.append((i, axis, halves))
        if not options:
            return None
        i, axis, halves = options[int(rng.integers(len(options)))]
        if step == 0:
            first_axis = axis
        boxes[i:i + 1] = halves
    mask = _walls((h, w), boxes)
    count, _, actual_partial = _room_analysis(mask)
    assert count == n
    return None if partial and n >= 3 and not actual_partial else mask


def _piece(rng, palette, compass, n, mark_size, partial, damaged, small=False):
    bg, body, reg, noise = palette[:4]
    sizes = [(h, w) for h in range(5, 10) for w in range(5, 10)
             if ((h - 1) // 2) * ((w - 1) // 2) >= n]
    if small:
        sizes = [(7, 7)]
    h, w = sizes[int(rng.integers(len(sizes)))]
    mask = _partition(rng, h, w, n, partial)
    if mask is None:
        return None
    cells = {tuple(p): body for p in np.argwhere(mask).tolist()}
    if mark_size == 1:
        p = sorted(cells)[int(rng.integers(len(cells)))]
        marks = {p}
    else:
        d = int(rng.integers(4))
        length = w if d % 2 == 0 else h
        start = int(rng.integers(length - mark_size + 1))
        marks = {(0, i) if d == 0 else (i, w - 1) if d == 1 else
                 (h - 1, i) if d == 2 else (i, 0)
                 for i in range(start, start + mark_size)}
    for p in marks:
        cells[p] = reg
    safe = []
    for y, x in sorted(cells):
        ns = [((y + dy, x + dx) in cells) for dy, dx in _DIRS]
        if (y, x) not in marks and sum(ns) == 2 and ((ns[0] and ns[2]) or (ns[1] and ns[3])):
            safe.append((y, x))
    if damaged and safe:
        rng.shuffle(safe)
        chosen = []
        for p in safe:
            if not (set((p,)) & _halo(chosen)):
                chosen.append(p)
                cells[p] = noise
                if len(chosen) >= int(rng.integers(1, 3)):
                    break
    side = int(rng.integers(4))
    for d in (side, (side + 1) % 4):
        length = w if d % 2 == 0 else h
        patch = int(rng.integers(1, 4))
        start = int(rng.integers(length - patch + 1))
        for i in range(start, start + patch):
            p = (-1, i) if d == 0 else (i, w) if d == 1 else (h, i) if d == 2 else (i, -1)
            cells[p] = compass[d]
    return _transform(cells, 0)


def _compass_layers(rng, colors, depth):
    base = {p: int(c) for p, c in zip(_DIRS, colors)}
    layers = []
    for t in rng.permutation(8)[:depth]:
        points = []
        for (r, c), v in base.items():
            if t >= 4:
                c = -c
            for _ in range(int(t) % 4):
                r, c = c, -r
            points.append(((r, c), v))
        table = dict(points)
        layers.append(tuple(table[p] for p in _DIRS))
    return layers


def _canvas(shape, palette, layers):
    h, w = shape
    d = len(layers)
    a = np.full(shape, palette[0], dtype=np.int8)
    for k, (top, right, bottom, left) in enumerate(layers):
        a[d - 1 - k, d:w - d] = top
        a[h - d + k, d:w - d] = bottom
        a[d:h - d, d - 1 - k] = left
        a[d:h - d, w - d + k] = right
    return a


def _origins(rng, cells, shape, depth):
    h, w = shape
    ph = 1 + max(r for r, _ in cells)
    pw = 1 + max(c for _, c in cells)
    origins = [(r, c) for r in range(depth + 1, h - depth - ph)
               for c in range(depth + 1, w - depth - pw)]
    rng.shuffle(origins)
    return origins


def _pose_kind(source, final, body, damage):
    repaired = {p: body if v == damage else v for p, v in source.items()}
    signature = _signature(final)
    ts = [t for t in range(8) if _signature(_transform(repaired, t)) == signature]
    _require(bool(ts), "object not rigidly preserved")
    return "translation" if 0 in ts else "rotation" if any(t < 4 for t in ts) else "reflection"


def _witnesses(info):
    answers = _maps(info)
    _require(len(answers) == 1, "ambiguous complete rule")
    actual = answers[0]
    compass_needed = len(_maps(info, use_labels=False)) >= 2
    count_needed = False
    for i, a in enumerate(info["analyses"]):
        if len(a["marks"]) == 1 and a["count"] > 1:
            alternatives = _maps(info, overrides={i: 1})
            count_needed |= bool(alternatives) and all(c != actual for c in alternatives)
    poses = []
    for i, source in enumerate(info["sources"]):
        candidates = _candidates(info, i)
        _require(len(candidates) == 1, "ambiguous individual docking")
        poses.append(_pose_kind(source, candidates[0][1], info["body"], info["damage"]))
    return {"compass_needed": compass_needed, "count_needed": count_needed,
            "poses": poses, "partial": any(a["partial"] for a in info["analyses"]),
            "damage": sum(a["damage_cells"] for a in info["analyses"])}


def _make_pair(rng, depth, counts, witness=None):
    palette = [int(x) for x in rng.permutation(10)[:8]]
    layers = _compass_layers(rng, palette[4:], depth)
    mark_sizes = [1, int(rng.integers(2, 5))][:len(counts)]
    for attempt in range(240):
        small = attempt >= 160
        shape = (30, 30) if small else tuple(int(rng.integers(max(25, 2 * depth + 16), 31)) for _ in range(2))
        pieces = []
        for i, n in enumerate(counts):
            partial = n >= 3 and ((witness == "reflection" and i == 0) or rng.random() < .65)
            piece = _piece(rng, palette, layers[n - 1], n, mark_sizes[i], partial,
                           damaged=(i == 0 or rng.random() < .55), small=small)
            if piece is None:
                break
            pieces.append(piece)
        if len(pieces) != len(counts):
            continue
        finals, blocked = [], set()
        for piece in pieces:
            for origin in _origins(rng, piece, shape, depth):
                placed = _shift(piece, origin)
                # Bbox separation also leaves all empty rooms visually distinct.
                t, l, b, r = _bbox(placed)
                box = {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}
                if not box & blocked:
                    finals.append(placed)
                    blocked.update(_halo(box))
                    break
            else:
                break
        if len(finals) != len(counts):
            continue
        targets = {p: palette[2] for final in finals for p, v in final.items() if v == palette[2]}
        blocked = _halo(targets)
        sources = []
        for i, final in enumerate(finals):
            transforms = list(map(int, rng.permutation(8)))
            if i == 0 and witness:
                transforms = [t for t in transforms if (0 < t < 4 if witness == "rotation" else t >= 4)]
            found = False
            for t in transforms:
                cells = _transform(final, t)
                for origin in _origins(rng, cells, shape, depth):
                    placed = _shift(cells, origin)
                    a, l, b, r = _bbox(placed)
                    box = {(y, x) for y in range(a, b + 1) for x in range(l, r + 1)}
                    if not box & blocked:
                        sources.append(placed)
                        blocked.update(_halo(box))
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        if len(sources) != len(counts):
            continue
        inp = _canvas(shape, palette, layers)
        out = inp.copy()
        for p, value in targets.items():
            inp[p] = value
        for source in sources:
            for p, value in source.items():
                inp[p] = value
        for final in finals:
            for p, value in final.items():
                out[p] = palette[1] if value == palette[3] else value
        try:
            info = _parse(inp)
            maps = _maps(info)
            if len(maps) != 1 or not np.array_equal(_render(info, maps[0]), out):
                continue
            if witness:
                evidence = _witnesses(info)
                if not evidence["compass_needed"] or not evidence["count_needed"]:
                    continue
                if witness not in evidence["poses"]:
                    continue
                if witness == "reflection" and not evidence["partial"]:
                    continue
        except ValueError:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded construction exhausted")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    depth = int(rng.integers(4, 6))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", num_train), ("test", num_test)):
        for i in range(number):
            witness = None
            if split == "train" and i == 0:
                counts, witness = [2, 1], "rotation"
            elif split == "train" and i == 1:
                counts, witness = [depth - 1, max(2, depth - 2)], "reflection"
            elif split == "test":
                counts = [depth, int(rng.integers(1, depth))]
            else:
                counts = [int(rng.integers(1, depth)) for _ in range(int(rng.integers(1, 3)))]
            for _ in range(12):
                pair = _make_pair(rng, depth, counts, witness)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("duplicate after bounded attempts")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
    inp, out = (_grid(pair[k]) for k in ("input", "output"))
    _require(inp.shape == out.shape, "output dimensions changed")
    info = _parse(inp)
    answers = _maps(info)
    _require(len(answers) == 1 and np.array_equal(_render(info, answers[0]), out), "wrong docking output")
    _require(not np.array_equal(inp, out), "identity pair")
    _require(bool(np.all(out[info["tapes"]] == inp[info["tapes"]])), "changed boundary tapes")
    _require(info["damage"] is None or not np.any(out == info["damage"]), "unrepaired damage")
    output_objects = _components(info["field"] & (out != info["background"]))
    _require(len(output_objects) == len(info["sources"]), "output object count")
    for i, a in enumerate(info["analyses"]):
        placements = _candidates(info, i)
        _require(len(placements) == 1, "nonunique placement")
        target_id, final = placements[0]
        _require({p for p, v in final.items() if v == info["registration"]} == info["targets"][target_id],
                 "registration mismatch")
        _pose_kind(info["sources"][i], final, info["body"], info["damage"])
        core, _ = _labels(final, info["band_colors"])
        t, l, b, r = _bbox(core)
        mask = np.zeros((b - t + 1, r - l + 1), dtype=bool)
        for y, x in core:
            mask[y - t, x - l] = True
        _require(_room_analysis(mask)[0] == a["count"], "topology changed during docking")
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "pair lists required")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "example counts")
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            key = _key(info["grid"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(1 <= len(info["sources"]) <= 2 and info["depth"] in (4, 5), "construction scope")
            (training if split == "train" else testing).append(info)
    depths = {x["depth"] for x in training + testing}
    _require(len(depths) == 1, "puzzle depth changed")
    depth = next(iter(depths))
    trcounts = {a["count"] for info in training for a in info["analyses"]}
    _require(trcounts == set(range(1, depth)), "training ordinal coverage")
    _require(all(max(a["count"] for a in info["analyses"]) == depth for info in testing), "test count shift")
    evidence = [_witnesses(info) for info in training[:2]]
    _require(all(e["compass_needed"] and e["count_needed"] for e in evidence), "missing parent-dependence witness")
    _require("rotation" in evidence[0]["poses"] and "reflection" in evidence[1]["poses"], "missing pose witnesses")
    _require(evidence[1]["partial"] and any(e["damage"] for e in evidence), "missing topology/repair witness")


def _normalized_structure(grid):
    roles = {}
    flat = bytes(roles.setdefault(int(c), len(roles)) for c in grid.ravel())
    return hashlib.sha256(bytes(grid.shape) + flat).hexdigest()


def self_test():
    """42 sequential seeds, bounded metrics, geometry and omission checks."""
    # Causal topology check: remove only a partial divider. Same dimensions,
    # labels, registration, target and tapes; three rooms become two, so the
    # required pose changes. The central anchor makes both poses fit.
    palette = list(range(8))
    layers_demo = [(4, 5, 6, 7), (7, 4, 5, 6), (6, 7, 4, 5), (5, 6, 7, 4)]
    mask = _walls((7, 7), [(0, 0, 3, 3), (3, 0, 6, 3), (0, 3, 6, 6)])
    demo_piece = {tuple(p): 1 for p in np.argwhere(mask).tolist()}
    demo_piece[(3, 3)] = 2
    demo_piece[(0, 1)] = 3
    demo_piece[(-1, 2)] = 6
    demo_piece[(2, 7)] = 7
    demo = _canvas((30, 30), palette, layers_demo)
    for p, v in _shift(demo_piece, (6, 6)).items():
        demo[p] = v
    demo[15, 15] = 2
    changed = demo.copy()
    changed[9, 7:9] = 0
    assert _parse(demo)["analyses"][0]["count"] == 3
    assert _parse(changed)["analyses"][0]["count"] == 2
    assert not np.array_equal(solve(demo), solve(changed))
    repaired_demo = demo.copy()
    repaired_demo[repaired_demo == 3] = 1
    assert np.array_equal(solve(demo), solve(repaired_demo))
    shapes, structures, palettes, backgrounds, layers = set(), set(), set(), set(), set()
    objects, counts, poses, room_sizes = set(), set(), set(), set()
    shape_counts, count_shapes, mark_counts = {}, {}, {}
    signatures = set()
    partials = damaged = overlap = 0
    base = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed == 0:
            base = tuple(_digest(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output"))
        signatures.add(tuple(_digest(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair)
                shapes.add(info["grid"].shape)
                structures.add(_normalized_structure(info["grid"]))
                backgrounds.add(info["background"])
                palettes.add((info["background"], info["body"], info["registration"], tuple(info["compasses"])))
                layers.add(info["depth"])
                objects.add(len(info["sources"]))
                old = set().union(*(set(s) for s in info["sources"]))
                final = _maps(info)[0]
                overlap += bool(old & set(final))
                for i, a in enumerate(info["analyses"]):
                    n, shape = a["count"], tuple(sorted(a["core_shape"]))
                    counts.add(n)
                    partials += a["partial"]
                    damaged += a["damage_cells"] > 0
                    shape_counts.setdefault(shape, set()).add(n)
                    count_shapes.setdefault(n, set()).add(shape)
                    mark_counts.setdefault(len(a["marks"]), set()).add(n)
                    for t, l, b, r in a["rooms"]:
                        room_sizes.add((b - t - 1, r - l - 1))
                    final_i = _candidates(info, i)[0][1]
                    poses.add(_pose_kind(info["sources"][i], final_i, info["body"], info["damage"]))
                # Global geometry and palette transformations must commute with solve.
                if seed < 3:
                    for t in range(8):
                        def array_transform(a):
                            return np.rot90(np.fliplr(a) if t >= 4 else a, -int(t % 4))
                        assert np.array_equal(solve(array_transform(pair["input"])), array_transform(pair["output"]))
                    permutation = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
                    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    again = generate_puzzle(0)
    assert base == tuple(_digest(p[k]) for s in ("train", "test") for p in again[s] for k in ("input", "output"))
    # The host verifier exhausts legal combinations. Keep the embedded test
    # bounded with representative extremes and a non-default legal ratio.
    for nt, ne in ((2, 1), (10, 1), (10, 4), (3, 2)):
        p = generate_puzzle(1000 + nt * 10 + ne, nt, ne)
        assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 16 and len(palettes) >= 150
    assert backgrounds == set(range(10)) and layers == {4, 5}
    assert objects == {1, 2} and counts == {1, 2, 3, 4, 5}
    assert poses == {"translation", "rotation", "reflection"}
    assert partials >= 60 and damaged >= 150 and overlap >= 20
    assert any(len(ns) >= 4 for ns in shape_counts.values())
    assert all(len(ss) >= 6 for ss in count_shapes.values())
    assert len(mark_counts[1]) == 5 and any(len(ns) >= 4 for size, ns in mark_counts.items() if size > 1)
    assert min(min(s) for s in room_sizes) == 1
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes),
            "color_normalized_input_structures": len(structures), "palettes": len(palettes),
            "room_counts": sorted(counts), "object_counts": sorted(objects),
            "core_shapes": len(shape_counts), "room_interior_shapes": len(room_sizes),
            "partial_networks": int(partials), "damaged_networks": int(damaged),
            "pairs_with_old_final_overlap": int(overlap), "poses": sorted(poses),
            "topology_counterfactual": True, "repair_invariance": True,
            "independent_solver": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
