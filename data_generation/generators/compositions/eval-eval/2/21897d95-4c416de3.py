"""Corner transfer circuits: infer a repeated program, then execute its arrows.

Only standard Python and NumPy are used. The constructor paints from a clean
territory plan; the solver reconstructs the frame, program, and original
components from the annotated input. No source generator is imported.
"""

from __future__ import annotations

import json
import numpy as np

_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
HOST_SEEDS = (0, 46969844, 575709140, 763889560, 376216659, 251339703,
              296349019, 648349246, 999298884, 150367982, 223572744,
              885453808, 346357401, 293824685, 654119957, 876636967,
              935176860, 388566790, 42123396, 364774845, 306744702,
              401682154, 403171826, 188702373, 530758703, 918991434,
              880792080, 799239377, 112764375, 51423216, 130326216,
              875738656, 55600882, 999450628, 3360339, 520656337,
              244303098, 751883614, 587742392, 490786086, 731680820,
              576891075)


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid dimensions")
    _need(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((0 <= g) & (g <= 9))), "invalid color")
    return g.astype(np.int8, copy=True)


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def _outline(rect):
    t, l, b, r = rect
    return ({(t, c) for c in range(l, r + 1)} |
            {(b, c) for c in range(l, r + 1)} |
            {(a, l) for a in range(t, b + 1)} |
            {(a, r) for a in range(t, b + 1)})


def _corners(rect):
    t, l, b, r = rect
    return ((t, l), (t, r), (b, l), (b, r))


def _at(corner, q, uv):
    dy, dx = _SIGNS[q]
    return corner[0] + dy * uv[0], corner[1] + dx * uv[1]


def _motif(command):
    """Canonical corner has its interior below/right. out is a boolean."""
    edge, distance, out = command
    a = 1 if out else -1
    if edge == 0:  # top incident wall
        return frozenset(((a, distance - 1), (a, distance),
                          (a, distance + 1), (0, distance)))
    return frozenset(((distance - 1, a), (distance, a),
                      (distance + 1, a), (distance, 0)))


def _stencil(commands):
    return frozenset(p for command in commands for p in _motif(command))


def _arrow(corner, q, command):
    edge, s, out = command
    a = 1 if out else -1
    center = (a, s) if edge == 0 else (s, a)
    target = (-a, s) if edge == 0 else (s, -a)
    back = (2 * a, s) if edge == 0 else (s, 2 * a)
    return {"cells": {_at(corner, q, p) for p in _motif(command)},
            "center": _at(corner, q, center), "targetcell": _at(corner, q, target),
            "back": _at(corner, q, back), "corner": q, "command": command}


def _parse_program(pattern):
    commands = [(edge, s, out) for edge in range(2) for s in range(2, 9)
                for out in (False, True) if _motif((edge, s, out)) <= pattern]
    _need(2 <= len(commands) <= 4, "circuit must contain two to four recognizable Ts")
    used = set()
    for command in commands:
        cells = _motif(command)
        _need(not used & cells, "ambiguous or overlapping Ts")
        used.update(cells)
    _need(used == set(pattern), "reference has unexplained circuit pixels")
    for a in commands:
        for b in commands:
            if a != b and a[0] == b[0]:
                _need(abs(a[1] - b[1]) >= 3, "Ts on one arm need a gap")
    return tuple(sorted(commands))


def _components(g, wall):
    """Flood fill original, four-connected, non-wall components, by color."""
    ids = np.full(g.shape, -1, dtype=np.int16)
    colors, cells = [], []
    h, w = g.shape
    for row, col in np.ndindex(g.shape):
        if g[row, col] == wall or ids[row, col] >= 0:
            continue
        label, color = len(colors), int(g[row, col])
        colors.append(color)
        points, stack = [], [(row, col)]
        ids[row, col] = label
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for dy, dx in _DIRS:
                a, b = y + dy, x + dx
                if (0 <= a < h and 0 <= b < w and ids[a, b] < 0
                        and g[a, b] == color):
                    ids[a, b] = label
                    stack.append((a, b))
        cells.append(points)
    return ids, colors, cells


def _paint(clean, ids, colors, arrows, *, visible_only=False, cascade=False):
    result = clean.copy()
    assignments = {}
    for arrow in sorted(arrows, key=lambda a: a["center"]):
        if visible_only and not arrow["visible_complete"]:
            continue
        source, target = arrow["source"], arrow["target"]
        payload = int(result[arrow["center"]]) if cascade else colors[source]
        if not cascade:
            _need(target not in assignments or assignments[target] == payload,
                  "contradictory incoming transfers")
        assignments[target] = payload
        if cascade:
            result[ids == target] = payload
    if not cascade:
        for target, payload in assignments.items():
            result[ids == target] = payload
    return result, assignments


def _read_with_roles(raw, wall, ink, rotation=False):
    wall_points = set(map(tuple, np.argwhere(raw == wall)))
    _need(len(wall_points) >= 20, "too little frame evidence")
    t = min(p[0] for p in wall_points)
    b = max(p[0] for p in wall_points)
    l = min(p[1] for p in wall_points)
    r = max(p[1] for p in wall_points)
    rect = (t, l, b, r)
    _need(min(b - t, r - l) >= 8, "frame too small")
    _need(t >= 2 and l >= 2 and b <= raw.shape[0] - 3 and r <= raw.shape[1] - 3,
          "frame needs visible source context on every side")
    outline = _outline(rect)
    _need(wall_points <= outline, "wall color occurs away from its frame")
    corners = _corners(rect)
    _need(all(raw[p] == wall for p in corners), "frame corners must remain bare")
    _need(all(raw[p] in (wall, ink) for p in outline), "broken frame")
    clues = {q: set() for q in range(4)}
    for y, x in np.argwhere(raw == ink):
        y, x = int(y), int(x)
        distances = [(y - a) ** 2 + (x - b_) ** 2 for a, b_ in corners]
        minimum = min(distances)
        _need(distances.count(minimum) == 1, "ink equally close to two corners")
        q = distances.index(minimum)
        cy, cx = corners[q]
        dy, dx = _SIGNS[q]
        u, v = (y - cy) * dy, (x - cx) * dx
        if rotation and q in (1, 2):
            u, v = v, u
        clues[q].add((u, v))
    clues = {q: frozenset(p) for q, p in clues.items() if p}
    _need(2 <= len(clues) <= 4, "need a reference and a marked recipient")
    candidates = {p for p in clues.values() if all(other <= p for other in clues.values())}
    _need(len(candidates) == 1, "no unique intact compatible circuit")
    pattern = candidates.pop()
    commands = _parse_program(pattern)
    reach = max(max(abs(u), abs(v)) for u, v in pattern)
    _need(min(b - t + 1, r - l + 1) >= 2 * reach + 3,
          "corner neighborhoods insufficiently separated")
    clean = raw.copy()
    arrows, occupied = [], set()
    for q, clue in clues.items():
        for command in commands:
            actual = (1 - command[0], command[1], command[2]) if rotation and q in (1, 2) else command
            arrow = _arrow(corners[q], q, actual)
            _need(all(0 <= y < raw.shape[0] and 0 <= x < raw.shape[1]
                      for y, x in arrow["cells"] | {arrow["targetcell"], arrow["back"]}),
                  "circuit must be entirely visible")
            _need(not occupied & arrow["cells"], "completed circuits collide")
            occupied.update(arrow["cells"])
            host = int(raw[arrow["back"]])
            _need(host not in (wall, ink), "source context is not visible fill")
            for p in arrow["cells"]:
                original = wall if p in outline else host
                _need(raw[p] in (original, ink), "copy disagrees with underlying source")
                clean[p] = original
            arrow["visible_complete"] = _motif(command) <= clue
            arrow["host"] = host
            arrows.append(arrow)
    _need(not np.any(clean == ink), "unexplained ink remains")
    ids, colors, cells = _components(clean, wall)
    for arrow in arrows:
        source = int(ids[arrow["center"]])
        target = int(ids[arrow["targetcell"]])
        _need(source >= 0 and target >= 0 and source != target, "transfer lacks two territories")
        _need(int(ids[arrow["back"]]) == source, "source context crosses a boundary")
        _need(all(ids[p] == source for p in arrow["cells"] - outline),
              "T bar crosses source components")
        arrow.update(source=source, target=target)
    result, assignments = _paint(clean, ids, colors, arrows)
    return result, {"wall": wall, "ink": ink, "rect": rect, "clues": clues,
                    "stencil": pattern, "commands": commands, "clean": clean,
                    "ids": ids, "colors": colors, "cells": cells,
                    "arrows": arrows, "assignments": assignments}


def _decode(input_grid, rotation=False):
    raw = _grid(input_grid)
    answers = []
    # The perimeter bounding box, four bare corners, full outline accounting,
    # and repeated glyph grammar identify roles; no literal color is privileged.
    for wall in map(int, np.unique(raw)):
        points = np.argwhere(raw == wall)
        if len(points) < 20:
            continue
        t, l = points.min(axis=0)
        b, r = points.max(axis=0)
        if min(b - t, r - l) < 8:
            continue
        if not np.all((points[:, 0] == t) | (points[:, 0] == b) |
                      (points[:, 1] == l) | (points[:, 1] == r)):
            continue
        for ink in map(int, np.unique(raw)):
            if ink == wall:
                continue
            try:
                answers.append(_read_with_roles(raw, wall, ink, rotation))
            except AssertionError:
                continue
    _need(len(answers) == 1, "input does not determine a unique frame/circuit interpretation")
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer entirely from the input; no examples or seed needed."""
    return _decode(input_grid)[0]


def _features(info):
    colors, assignments = info["colors"], info["assignments"]
    final = [assignments.get(i, c) for i, c in enumerate(colors)]
    arrows = info["arrows"]
    visible, _ = _paint(info["clean"], info["ids"], colors, arrows, visible_only=True)
    cascaded, _ = _paint(info["clean"], info["ids"], colors, arrows, cascade=True)
    intended, _ = _paint(info["clean"], info["ids"], colors, arrows)
    boxes = [(max(y for y, _ in p) - min(y for y, _ in p) + 1,
              max(x for _, x in p) - min(x for _, x in p) + 1) for p in info["cells"]]
    return {"shape": info["clean"].shape, "frame_shape": (info["rect"][2] - info["rect"][0] + 1,
                                                         info["rect"][3] - info["rect"][1] + 1),
            "commands": len(info["commands"]), "marked": len(info["clues"]),
            "territories": len(colors), "targeted": len(assignments),
            "stencil": tuple(sorted(info["stencil"])), "wall": info["wall"], "ink": info["ink"],
            "outward": sum(c[2] for c in info["commands"]),
            "partial": sum(p != info["stencil"] for p in info["clues"].values()),
            "references": sum(p == info["stencil"] for p in info["clues"].values()),
            "clue_counts": tuple(len(p) for p in info["clues"].values()),
            "hidden_effect": not np.array_equal(visible, intended),
            "simultaneous": not np.array_equal(cascaded, intended),
            "locality": any(colors[a] == colors[b] and final[a] != final[b]
                            for a in range(len(colors)) for b in range(a)),
            "position_sensitive": any(a["corner"] == b["corner"] and
                                      a["command"][0] == b["command"][0] and
                                      a["command"][2] == b["command"][2] and
                                      a["target"] != b["target"] for a in arrows for b in arrows),
            "nonrectangular": sum(len(p) != h * w for p, (h, w) in zip(info["cells"], boxes)),
            "unmarked": len(info["clues"]) < 4,
            "both_arms": {c[0] for c in info["commands"]} == {0, 1}}


def _random_commands(rng):
    maximum = int(rng.choice((2, 5, 5, 8)))
    wanted = 2 if maximum == 2 else int(rng.integers(2, 5))
    slots = [(edge, s) for edge in range(2) for s in range(2, maximum + 1)]
    rng.shuffle(slots)
    chosen = []
    for edge, s in slots:
        if all(edge != e or abs(s - d) >= 3 for e, d in chosen):
            chosen.append((edge, s))
        if len(chosen) == wanted:
            break
    # Greedy packing may fit fewer than requested; both incident arms still
    # supply at least one slot, so the resulting two-to-four-arrow circuit is valid.
    out_index = int(rng.integers(len(chosen))) if rng.random() < .7 else -1
    return tuple(sorted((edge, s, i == out_index) for i, (edge, s) in enumerate(chosen)))


def _cut_quad(rng, box, count, stepped):
    """Unequal pixel cuts, with optional stair-stepped seams, inside one quadrant."""
    t, l, b, r = box  # half-open
    labels = np.zeros((b - t, r - l), dtype=np.int16)
    rr, cc = np.indices(labels.shape)
    for label in range(1, count):
        possible = [i for i in range(label) if np.count_nonzero(labels == i) >= 12]
        if not possible:
            break
        old = max(possible, key=lambda i: np.count_nonzero(labels == i))
        mask = labels == old
        ys, xs = np.nonzero(mask)
        axis = int(rng.integers(2))
        lo, hi = (int(ys.min()), int(ys.max()) + 1) if axis == 0 else (int(xs.min()), int(xs.max()) + 1)
        if hi - lo < 4:
            axis = 1 - axis
            lo, hi = (int(ys.min()), int(ys.max()) + 1) if axis == 0 else (int(xs.min()), int(xs.max()) + 1)
        if hi - lo < 4:
            break
        cut = int(rng.integers(lo + 2, hi - 1))
        coordinate, along = (rr, cc) if axis == 0 else (cc, rr)
        length = labels.shape[1 - axis]
        path = np.full(length, cut, dtype=int)
        if stepped and length >= 5 and rng.random() < .65:
            split = int(rng.integers(2, length - 1))
            path[split:] = int(np.clip(cut + rng.choice((-1, 1)), lo + 2, hi - 2))
        part = mask & (coordinate < path[along])
        other = mask & ~part
        if min(int(part.sum()), int(other.sum())) < 4:
            continue
        # Connectivity is checked directly for the candidate cut.
        for side in (part, other):
            probe = np.where(side, 0, 9).astype(np.int8)
            _, _, components = _components(probe, 9)
            if len(components) != 1:
                break
        else:
            labels[part] = label
    _, _, groups = _components(labels, -1)
    dense = np.empty(labels.shape, dtype=np.int16)
    for i, group in enumerate(groups):
        for p in group:
            dense[p] = i
    return dense


def _color_labels(rng, labels, palette):
    count = int(labels.max()) + 1
    neighbors = [set() for _ in range(count)]
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:])):
        for x, y in zip(a[a != b], b[a != b]):
            if x >= 0 and y >= 0:
                neighbors[int(x)].add(int(y))
                neighbors[int(y)].add(int(x))
    for _ in range(30):
        colors = [-1] * count
        order = sorted(range(count), key=lambda x: (-len(neighbors[x]), float(rng.random())))
        for label in order:
            available = [c for c in palette if c not in {colors[n] for n in neighbors[label]}]
            if not available:
                break
            colors[label] = int(rng.choice(available))
        else:
            return colors
    return None


def _construct(rng, mode, roles, test=False):
    if mode == "directions":
        commands = ((0, 2, False), (0, 5, True), (1, 3, False))
    elif mode == "positions":
        commands = ((0, 2, False), (0, 5, False), (1, 2, False))
    else:
        commands = _random_commands(rng)
    if mode != "random" and rng.random() < .5:
        commands = tuple(sorted((1 - e, s, out) for e, s, out in commands))
    pattern = _stencil(commands)
    reach = max(max(abs(u), abs(v)) for u, v in pattern)
    minimum = 2 * reach + 3
    fh, fw = int(rng.integers(minimum, 27)), int(rng.integers(minimum, 27))
    h = int(rng.integers(fh + 4, 31))
    w = int(rng.integers(fw + 4, 31))
    t = int(rng.integers(2, h - fh - 1))
    l = int(rng.integers(2, w - fw - 1))
    b, r = t + fh - 1, l + fw - 1
    rect = (t, l, b, r)
    # Seams may move within the middle zone, away from the corner programs.
    my = int(rng.integers(t + reach + 1, b - reach + 1))
    mx = int(rng.integers(l + reach + 1, r - reach + 1))
    wall, ink = roles if roles is not None else map(int, rng.choice(10, 2, replace=False))
    palette = [c for c in range(10) if c not in (wall, ink)]
    rng.shuffle(palette)
    n_outer = 2 if mode == "directions" else int(rng.choice((2, 3, 4)))
    outer_palette, inner_palette = palette[:n_outer], palette[n_outer:]
    inner_palette = inner_palette[:int(rng.integers(3, min(4, len(inner_palette)) + 1))]
    yy, xx = np.indices((h, w))
    outer = (yy >= my).astype(int) * 2 + (xx >= mx).astype(int)
    # A proper coloring of the quadrant adjacency cycle; diagonal reuse is legal.
    outer_colors = [outer_palette[0], outer_palette[1], outer_palette[1], outer_palette[0]]
    if n_outer >= 3:
        outer_colors[2] = outer_palette[2]
    if n_outer == 4:
        outer_colors[3] = outer_palette[3]
    clean = np.asarray(outer_colors, dtype=np.int8)[outer]
    interior_ids = np.full((h, w), -1, dtype=np.int16)
    boxes = ((t + 1, l + 1, my, mx), (t + 1, mx, my, r),
             (my, l + 1, b, mx), (my, mx, b, r))
    next_id = 0
    for q, box in enumerate(boxes):
        a, c, d, e = box
        if mode == "directions":
            local = np.zeros((d - a, e - c), dtype=np.int16)
        elif mode == "positions":
            local = np.zeros((d - a, e - c), dtype=np.int16)
            # Cut between distances 2 and 5 on the arm with two arrows.
            if sum(command[0] == 0 for command in commands) == 2:
                boundary = l + 4 if q % 2 == 0 else r - 3
                local[:, boundary - c:] = 1
            else:
                boundary = t + 4 if q < 2 else b - 3
                local[boundary - a:, :] = 1
        else:
            count = int(rng.integers(1, 5 if test else 4))
            local = _cut_quad(rng, box, count, stepped=True)
        interior_ids[a:d, c:e] = local + next_id
        next_id += int(local.max()) + 1
    if mode == "directions":
        inner_palette = palette[n_outer:n_outer + 4]
        colors = list(map(int, rng.permutation(inner_palette)))
    else:
        colors = _color_labels(rng, interior_ids, inner_palette)
        if colors is None:
            return None
    for label, color in enumerate(colors):
        clean[interior_ids == label] = color
    outline = _outline(rect)
    for p in outline:
        clean[p] = wall
    reference = int(rng.integers(4))
    marked = [reference, reference ^ 1, reference ^ 2] if mode != "random" else [reference]
    if mode == "random":
        marked += [int(q) for q in rng.permutation(4) if q != reference][:int(rng.integers(1, 4))]
    full = {reference}
    if mode == "random" and len(marked) >= 3 and rng.random() < .3:
        full.add(marked[-1])
    corners = _corners(rect)
    ids, original_colors, _ = _components(clean, wall)
    arrows, used = [], set()
    for q in marked:
        for command in commands:
            arrow = _arrow(corners[q], q, command)
            if used & arrow["cells"]:
                return None
            used.update(arrow["cells"])
            source, target = int(ids[arrow["center"]]), int(ids[arrow["targetcell"]])
            if (source < 0 or target < 0 or source == target or ids[arrow["back"]] != source
                    or any(ids[p] != source for p in arrow["cells"] - outline)):
                return None
            arrow.update(source=source, target=target, visible_complete=True)
            arrows.append(arrow)
    try:
        out, _ = _paint(clean, ids, original_colors, arrows)
    except AssertionError:
        return None
    inp = clean.copy()
    for q in marked:
        retained = pattern
        if q not in full:
            points = sorted(pattern)
            rng.shuffle(points)
            if mode in ("directions", "positions") and q == (reference ^ 1):
                first = next(c for c in commands if not c[2])
                retained = _motif(first)
            else:
                maximum = max(2, len(points) // 3) if test else len(points)
                retained = frozenset(points[:int(rng.integers(1, maximum))])
        for p in retained:
            inp[_at(corners[q], q, p)] = ink
    try:
        prediction, info = _decode(inp)
    except AssertionError:
        return None
    if not np.array_equal(prediction, out) or not np.array_equal(info["clean"], clean):
        return None
    feature = _features(info)
    if not feature["hidden_effect"]:
        return None
    if mode == "directions" and not (feature["simultaneous"] and feature["locality"]):
        return None
    if mode == "positions" and not feature["position_sensitive"]:
        return None
    if mode != "random":
        try:
            alternative = _decode(inp, rotation=True)[0]
        except AssertionError:
            alternative = None
        if alternative is not None and np.array_equal(alternative, out):
            return None
    return {"input": inp, "output": out}, info


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not 2 <= int(num_train) <= 10 or
            not _integer(num_test) or not 1 <= int(num_test) <= 4 or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    roles = tuple(map(int, rng.choice(10, 2, replace=False))) if rng.random() < .5 else None
    puzzle, seen, train_stencils = {"train": [], "test": []}, set(), set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            mode = ("directions" if index == 0 else "positions") if split == "train" and index < 2 else "random"
            for _ in range(160):
                built = _construct(rng, mode, roles, test=split == "test")
                if built is None:
                    continue
                pair, info = built
                key = _key(pair["input"])
                stencil = info["stencil"]
                if key in seen or (split == "test" and stencil in train_stencils):
                    continue
                puzzle[split].append(pair)
                seen.add(key)
                if split == "train":
                    train_stencils.add(stencil)
                break
            else:
                raise RuntimeError("bounded circuit construction exhausted")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal example counts")
    seen, infos = set(), {"train": [], "test": []}
    rotation_fits = True
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape and not np.array_equal(inp, out), "invalid output shape or identity")
            key = _key(inp)
            _need(key not in seen, "duplicate input")
            seen.add(key)
            prediction, info = _decode(inp)
            _need(np.array_equal(prediction, out), "incorrect circuit transfer")
            wall, ink = info["wall"], info["ink"]
            _need(not np.any(out == ink), "command ink survives")
            _need(np.array_equal(out == wall, info["clean"] == wall), "frame geometry changed")
            _need(set(map(int, np.unique(out))) <= set(map(int, np.unique(info["clean"]))),
                  "invented output color")
            for label, color in enumerate(info["colors"]):
                expected = info["assignments"].get(label, color)
                _need(np.all(out[info["ids"] == label] == expected), "territory not recolored uniformly")
            feature = _features(info)
            _need(feature["partial"] >= 1 and feature["hidden_effect"], "B completion has no effect")
            _need(not np.array_equal(out, info["clean"]), "A transfers have no effect")
            infos[split].append(info)
            if split == "train" and rotation_fits:
                try:
                    rotation_fits = np.array_equal(_decode(inp, rotation=True)[0], out)
                except AssertionError:
                    rotation_fits = False
    features = [_features(info) for info in infos["train"]]
    for name in ("simultaneous", "locality", "position_sensitive", "unmarked", "both_arms"):
        _need(any(f[name] for f in features), "training lacks witness: " + name)
    _need(not rotation_fits, "axis reflection convention not distinguished in training")
    train_stencils = {info["stencil"] for info in infos["train"]}
    _need(len(train_stencils) >= 2, "training repeats only one circuit")
    _need(all(info["stencil"] not in train_stencils for info in infos["test"]),
          "test does not introduce a fresh input-supplied program")


def _signature(puzzle):
    return tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output"))


def _regression_test():
    """Explicit cases outside constructor preferences, with hand-set effects."""
    wall, ink = 5, 9
    rect = (3, 4, 19, 20)
    yy, xx = np.indices((23, 25))
    quadrants = (yy >= 11).astype(int) * 2 + (xx >= 12).astype(int)
    outside = np.asarray((0, 2, 2, 0), dtype=np.int8)[quadrants]
    inside_mask = (yy > 3) & (yy < 19) & (xx > 4) & (xx < 20)
    inner = np.asarray((6, 7, 3, 4), dtype=np.int8)[quadrants]
    clean = np.where(inside_mask, inner, outside).astype(np.int8)
    for p in _outline(rect):
        clean[p] = wall
    # Two outward Ts at each active corner have the same original source fill.
    commands = ((0, 2, False), (0, 5, True), (1, 3, True))
    pattern, corners = _stencil(commands), _corners(rect)

    def annotate(base):
        inp = base.copy()
        for q, clue in ((0, pattern), (1, {(-1, 2)}), (2, _motif(commands[2]))):
            for p in clue:
                inp[_at(corners[q], q, p)] = ink
        return inp

    expected = clean.copy()
    # Explicit semantic answer: swap the first three quadrant pairs only.
    for q, (outer_color, inner_color) in enumerate(zip((0, 2, 2), (6, 7, 3))):
        mask = (quadrants == q) & (clean != wall)
        expected[mask & inside_mask] = outer_color
        expected[mask & ~inside_mask] = inner_color
    inp = annotate(clean)
    _need(np.array_equal(solve(inp), expected), "consistent multiple outward transfers failed")
    # Same literal color on opposite sides of the membrane still means two
    # components. This pair also lies outside the disjoint-palette preference.
    overlap_input = inp.copy()
    overlap_expected = expected.copy()
    overlap_input[overlap_input == 7] = 2
    overlap_expected[overlap_expected == 7] = 2
    _need(np.array_equal(solve(overlap_input), overlap_expected), "overlapping fill palettes failed")
    permutation = np.asarray((9, 7, 1, 0, 6, 2, 8, 5, 4, 3), dtype=np.int8)
    _need(np.array_equal(solve(permutation[inp]), permutation[expected]), "literal-color shortcut")
    # Split TL's interior between the two outward sources. The two requested
    # colors now disagree for its one exterior recipient, so there is no answer.
    conflict_clean = clean.copy()
    conflict_clean[(quadrants == 0) & inside_mask & (xx >= 8)] = 1
    try:
        solve(annotate(conflict_clean))
    except AssertionError:
        pass
    else:
        raise AssertionError("conflicting recovered transfers were accepted")


def self_test():
    """Bounded coverage, ablations, determinism, and malformed-output checks."""
    _regression_test()
    features, structures, signatures = [], set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                prediction, info = _decode(np.asarray(pair["input"]).copy())
                _need(np.array_equal(prediction, pair["output"]), "input-only solve failed")
                feature = _features(info)
                features.append(feature)
                mapping = {}
                normalized = tuple(mapping.setdefault(int(c), len(mapping)) for c in pair["input"].ravel())
                structures.add((pair["input"].shape, normalized))
                _need(feature["hidden_effect"], "omitting B leaves answer unchanged")
                _need(not np.array_equal(info["clean"], prediction), "omitting A leaves answer unchanged")
    _need(len(signatures) == 42 and len(structures) >= 150, "insufficient structural diversity")
    _need(len({f["shape"] for f in features}) >= 40, "insufficient canvas variation")
    _need(len({f["stencil"] for f in features}) >= 35, "insufficient program variation")
    _need({f["commands"] for f in features} == {2, 3, 4}, "missing command counts")
    _need({f["marked"] for f in features} == {2, 3, 4}, "missing corner counts")
    _need({f["wall"] for f in features} == set(range(10)), "missing wall palettes")
    _need({f["ink"] for f in features} == set(range(10)), "missing ink palettes")
    _need(any(f["references"] > 1 for f in features), "missing multiple references")
    _need(max(f["territories"] for f in features) >= 15, "missing richer partitions")
    _need(any(f["nonrectangular"] > 4 for f in features), "missing irregular interior territories")
    _need(any(min(f["shape"]) < 20 for f in features), "missing smaller canvases")
    _need(any(30 in f["shape"] for f in features), "missing 30-cell axis")
    _need(_signature(generate_puzzle(0)) == _signature(generate_puzzle(0)), "nondeterminism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(991 + nt * 10 + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(12, 2, 1)
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzle[s]]
               for s in puzzle}
    raw = damaged["test"][0]["input"]
    _, info = _decode(raw)
    # Changing a bare frame corner must invalidate the input, not silently move it.
    raw[_corners(info["rect"])[0]] = info["ink"]
    try:
        solve(raw)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged frame was accepted")
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzle[s]]
               for s in puzzle}
    out = damaged["test"][0]["output"]
    out[0, 0] = (int(out[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("unrelated output mutation accepted")
    return {"seeds": 42, "pairs": len(features), "distinct_puzzles": len(signatures),
            "normalized_structures": len(structures),
            "canvas_shapes": len({f["shape"] for f in features}),
            "programs": len({f["stencil"] for f in features}),
            "territory_range": [min(f["territories"] for f in features), max(f["territories"] for f in features)]}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
