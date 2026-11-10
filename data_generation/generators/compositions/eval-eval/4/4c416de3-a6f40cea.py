"""Hidden junction motifs: infer rectangular rails and weave a corner stencil.

Only NumPy and the standard library are used. The renderer constructs labels
from sampled geometry; solve independently parses the input's visible pixels.
"""
from __future__ import annotations

import itertools
import numpy as np

_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _perimeter(box):
    t, l, b, r = box
    return ({(t, x) for x in range(l, r + 1)}
            | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t, b + 1)}
            | {(y, r) for y in range(t, b + 1)})


def _area(box):
    t, l, b, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), min(xs), max(ys), max(xs)


def _corners(box):
    t, l, b, r = box
    return ((t, l), (t, r), (b, l), (b, r))


def _inside(point, box):
    y, x = point
    t, l, b, r = box
    return t < y < b and l < x < r


def _positions(box, q, stencil):
    y, x = _corners(box)[q]
    dy, dx = _DIRS[q]
    return {(y + dy * u, x + dx * v) for u, v in stencil}


def _connected(points):
    seen = {min(points)}
    stack = list(seen)
    while stack:
        u, v = stack.pop()
        for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = u + du, v + dv
            if p in points and p not in seen:
                seen.add(p)
                stack.append(p)
    return seen == set(points)


def _junctions(boxes):
    """A horizontal side and a different loop's vertical side form a corner."""
    result = []
    for hi, a in enumerate(boxes):
        for vi, b in enumerate(boxes):
            if hi == vi:
                continue
            for y, dy in ((a[0], 1), (a[2], -1)):
                for x, dx in ((b[1], 1), (b[3], -1)):
                    if a[1] < x < a[3] and b[0] < y < b[2]:
                        result.append((y, x, dy, dx, hi, vi))
    return sorted(result)


def _weave(main, fill, boxes, colors, stencil):
    t, l, b, r = main
    out = np.full((b - t - 1, r - l - 1), fill, dtype=np.int8)
    occupied = set()
    junctions = _junctions(boxes)
    anchors = {(y, x) for y, x, *_ in junctions}
    assert junctions, "no perpendicular rail junctions"
    clipped = 0
    cells = []
    for y, x, dy, dx, hi, vi in junctions:
        assert _inside((y, x), main), "all rail crossings must be under the window"
        stamp = {}
        for u, v in stencil:
            p = y + dy * u, x + dx * v
            assert _inside(p, boxes[hi]) and _inside(p, boxes[vi]), "motif leaves the mutual interior"
            if _inside(p, main):
                # Distances to the supporting horizontal and vertical sides
                # are u and v. Horizontal pigment wins the equal-distance tie.
                stamp[p] = colors[hi] if u <= v else colors[vi]
        assert stamp, "a whole motif was cropped away"
        assert not (set(stamp) & occupied), "junction motifs overlap"
        assert not (set(stamp) & anchors), "motif touches another junction anchor"
        occupied.update(stamp)
        clipped += int(len(stamp) < len(stencil))
        cells.append(stamp)
        for (yy, xx), color in stamp.items():
            out[yy - t - 1, xx - l - 1] = color
    return out, junctions, cells, clipped


def _read_input(value):
    """Recover mask, loops, source ornaments, and stencil without any label."""
    grid = _grid(value)
    bg = int(grid[0, 0])
    assert all(np.all(edge == bg) for edge in
               (grid[0], grid[-1], grid[:, 0], grid[:, -1])), "outer margin must be background"
    points_by_color = {int(c): {tuple(map(int, p)) for p in np.argwhere(grid == c)}
                       for c in np.unique(grid) if int(c) != bg}
    frames = []
    for color, points in sorted(points_by_color.items()):
        box = _bbox(points)
        t, l, b, r = box
        if b - t < 4 or r - l < 4 or points != _perimeter(box):
            continue
        interior = grid[t + 1:b, l + 1:r]
        if np.all(interior == interior[0, 0]) and int(interior[0, 0]) != color:
            frames.append((box, color, int(interior[0, 0])))
    assert len(frames) == 1, "window must have one unique complete frame color"
    main, frame, fill = frames[0]
    assert frame != bg and fill != frame
    mask = _area(main)
    assert fill == bg or points_by_color[fill] <= mask, "window fill occurs outside its mask"
    boxes, colors, occurrences = [], [], []
    for color, all_points in sorted(points_by_color.items()):
        if color in (frame, fill):
            continue
        points = all_points - mask
        assert points, "loop pigment has no visible evidence"
        box = _bbox(points)
        t, l, b, r = box
        assert b - t >= 6 and r - l >= 6, "loop too small for unambiguous corner clues"
        visible = _perimeter(box) - mask
        assert visible <= points and _bbox(visible) == box, "missing rail extremum or broken exposed rail"
        extras = points - visible
        assigned = {q: set() for q in range(4)}
        for y, x in sorted(extras):
            choices = []
            for q, (cy, cx) in enumerate(_corners(box)):
                dy, dx = _DIRS[q]
                u, v = (y - cy) * dy, (x - cx) * dx
                if 1 <= u <= 3 and 1 <= v <= 3:
                    choices.append((q, (u, v)))
            assert len(choices) == 1, "extra pixel is not a unique inward corner clue"
            q, uv = choices[0]
            assigned[q].add(uv)
        assert extras, "each rail must visibly carry at least one motif clue"
        index = len(boxes)
        boxes.append(box)
        colors.append(color)
        for q, clue in assigned.items():
            if clue:
                occurrences.append((index, q, frozenset(clue)))
    assert 2 <= len(boxes) <= 4, "expected two to four differently colored rails"
    assert len(set(colors)) == len(colors)
    # No coincident parallel sides or corner-touching loops are admitted.
    for a, b in itertools.combinations(boxes, 2):
        assert not ({a[0], a[2]} & {b[0], b[2]})
        assert not ({a[1], a[3]} & {b[1], b[3]})
        proper = {(y, x) for y, x, *_ in _junctions([a, b])}
        assert _perimeter(a) & _perimeter(b) == proper
    patterns = {clue for _, _, clue in occurrences
                if all(other <= clue for _, _, other in occurrences)}
    assert len(patterns) == 1, "no uniquely intact common motif"
    stencil = patterns.pop()
    assert 3 <= len(stencil) <= 9 and _connected(stencil)
    assert {(1, 1), (1, 2), (2, 1)} <= stencil, "missing distance/tie witnesses"
    assert any(clue != stencil for _, _, clue in occurrences), "no partial source copy"
    intact = [(i, q) for i, q, clue in occurrences if clue == stencil]
    # Input-local adjacent references establish axis reflections even when a
    # motif is asymmetric. Nothing about the convention comes from the seed.
    assert any(i == j and q ^ p in (1, 2) for i, q in intact for j, p in intact), "no adjacent intact references"
    reserved = set()
    for i, q, _ in occurrences:
        full = _positions(boxes[i], q, stencil)
        assert not (full & mask), "source copies must be entirely outside the window"
        assert not (full & reserved), "source motifs would overlap"
        assert all(not (full & _perimeter(box)) for box in boxes)
        reserved.update(full)
    out, junctions, cells, clipped = _weave(main, fill, boxes, colors, stencil)
    return out, {"background": bg, "frame": frame, "fill": fill, "main": main,
                 "boxes": boxes, "colors": colors, "stencil": stencil,
                 "occurrences": occurrences, "junctions": junctions,
                 "cells": cells, "clipped": clipped}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the junction map entirely from one input grid."""
    return _read_input(input_grid)[0]


def _sample_stencil(rng, radius, asymmetric=False):
    domain = {(u, v) for u in range(1, radius + 1) for v in range(1, radius + 1)}
    for _ in range(30):
        size = int(rng.integers(3, len(domain) + 1))
        pattern = {(1, 1), (1, 2), (2, 1)}
        while len(pattern) < size:
            frontier = sorted(p for p in domain - pattern
                              if any(abs(p[0] - u) + abs(p[1] - v) == 1 for u, v in pattern))
            pattern.add(frontier[int(rng.integers(len(frontier)))])
        if not asymmetric or pattern != {(v, u) for u, v in pattern}:
            return frozenset(pattern)
    # Only the sampling fallback is fixed; the solver never consults it.
    return frozenset({(1, 1), (1, 2), (2, 1), (1, 3)})


def _random_box(rng, shape, radius):
    h, w = shape
    yy = sorted(map(int, rng.choice(np.arange(1, h - 1), 2, replace=False)))
    xx = sorted(map(int, rng.choice(np.arange(1, w - 1), 2, replace=False)))
    if yy[1] - yy[0] < 2 * radius + 2 or xx[1] - xx[0] < 2 * radius + 2:
        return None
    return yy[0], xx[0], yy[1], xx[1]


def _geometry(rng, stencil, active, full_stamps):
    radius = max(max(p) for p in stencil)
    h, w = map(int, rng.integers(23 if active == 3 else 20, 31, size=2))
    # Independent margins, aspect ratio, and location; no fixed quadrant slots.
    t, l = map(int, rng.integers(4, min(8, min(h, w) // 3), size=2))
    b = h - int(rng.integers(4, min(8, h // 3))) - 1
    r = w - int(rng.integers(4, min(8, w // 3))) - 1
    main = t, l, b, r
    if b - t < 7 or r - l < 7:
        return None
    mask = _area(main)
    boxes = []
    for index in range(active):
        for _ in range(380):
            box = _random_box(rng, (h, w), radius)
            if box is None:
                continue
            perimeter = _perimeter(box)
            visible = perimeter - mask
            if not visible or _bbox(visible) != box or not (perimeter & mask):
                continue
            if any(({a[0], a[2]} & {box[0], box[2]})
                   or ({a[1], a[3]} & {box[1], box[3]}) for a in boxes):
                continue
            candidate = boxes + [box]
            junctions = _junctions(candidate)
            if index and len(junctions) <= len(_junctions(boxes)):
                continue
            if active == 2 and len(junctions) > 2:
                continue
            if any(not _inside((y, x), main) for y, x, *_ in junctions):
                continue
            if any(_perimeter(a) & perimeter != {(y, x) for y, x, *_ in _junctions([a, box])}
                   for a in boxes):
                continue
            source_options = []
            for q in range(4):
                points = _positions(box, q, stencil)
                if not (points & mask):
                    source_options.append(q)
            if not source_options:
                continue
            if index == 0 and not any(q ^ p in (1, 2) for q in source_options for p in source_options):
                continue
            if index:
                try:
                    _, _, cells, clipped = _weave(main, 0, candidate, list(range(1, len(candidate) + 1)), stencil)
                except AssertionError:
                    continue
                if full_stamps and clipped:
                    continue
            boxes.append(box)
            break
        else:
            return None
    # All active loops participate, and test graphs have at least two edges.
    junctions = _junctions(boxes)
    if len(junctions) < (4 if active == 3 else 2):
        return None
    options = []
    all_perimeters = set().union(*map(_perimeter, boxes))
    for box in boxes:
        eligible = [q for q in range(4) if not (_positions(box, q, stencil) & (mask | all_perimeters))]
        if not eligible:
            return None
        options.append(eligible)
    if not any(q ^ p in (1, 2) for q in options[0] for p in options[0]):
        return None
    return (h, w), main, boxes, options


def _render(rng, geometry, stencil, palette):
    shape, main, boxes, options = geometry
    bg, frame, special_fill = map(int, palette[:3])
    fill = bg if rng.random() < .4 else special_fill
    colors = list(map(int, palette[3:3 + len(boxes)]))
    inp = np.full(shape, bg, dtype=np.int8)
    for box, color in zip(boxes, colors):
        for point in _perimeter(box):
            inp[point] = color
    # Labels are painted from the actual sampled geometry before constructing
    # the opaque observation. The input parser is not used to produce labels.
    out, _, _, _ = _weave(main, fill, boxes, colors, stencil)
    t, l, b, r = main
    inp[t:b + 1, l:r + 1] = fill
    for point in _perimeter(main):
        inp[point] = frame
    adjacent = [(q, p) for q in options[0] for p in options[0] if q < p and q ^ p in (1, 2)]
    chosen = adjacent[int(rng.integers(len(adjacent)))]
    marked = [(0, q, True) for q in chosen]
    # A partial copy is mandatory, has arbitrary deletion positions, and may
    # be disconnected. It need not retain any particular source motif cell.
    partial_q = int(rng.choice(options[1]))
    marked.append((1, partial_q, False))
    for i, eligible in enumerate(options):
        for q in eligible:
            if any(j == i and p == q for j, p, _ in marked):
                continue
            if rng.random() < .48 or (i >= 2 and not any(j == i for j, _, _ in marked)):
                marked.append((i, q, bool(rng.random() < .3)))
    occupied = set()
    for i, q, intact in marked:
        full = _positions(boxes[i], q, stencil)
        if full & occupied:
            return None
        occupied.update(full)
        retained = sorted(full)
        if not intact:
            rng.shuffle(retained)
            retained = retained[:int(rng.integers(1, len(full)))]
        for point in retained:
            inp[point] = colors[i]
    return {"input": inp, "output": out}


def _array_key(a):
    a = np.asarray(a, dtype=np.int8)
    return tuple(a.shape), a.tobytes()


def _signature(puzzle):
    return tuple((_array_key(p["input"]), _array_key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    nt, nq = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(seed % (1 << 64))
    stable_palette = rng.permutation(10) if rng.random() < .25 else None
    puzzle, seen = {"train": [], "test": []}, set()
    train_stencils = set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            active = 2 if split == "train" else 3
            # The count shift fits compact motifs into more interacting rails.
            # The rule/parser still support 3x3 motifs on three/four rails.
            radius = 3 if split == "train" and (index < 2 or rng.random() < .5) else 2
            for attempt in range(260):
                stencil = _sample_stencil(rng, radius, asymmetric=(split == "train" and index == 0))
                if split == "train" and index == 1 and stencil in train_stencils:
                    continue
                geometry = _geometry(rng, stencil, active, full_stamps=(split == "train" and index == 0))
                if geometry is None:
                    continue
                palette = rng.permutation(10) if stable_palette is None else stable_palette
                pair = _render(rng, geometry, stencil, palette)
                if pair is None or _array_key(pair["input"]) in seen:
                    continue
                try:
                    prediction, info = _read_input(pair["input"])
                    assert np.array_equal(prediction, pair["output"])
                    if split == "train" and index == 0:
                        assert info["clipped"] == 0
                        assert any(len(set(stamp.values())) == 2 for stamp in info["cells"])
                        assert all(not np.array_equal(alt, pair["output"])
                                   for alt in _alternative_maps(info).values())
                except AssertionError:
                    continue
                seen.add(_array_key(pair["input"]))
                if split == "train":
                    train_stencils.add(stencil)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative_maps(info):
    """Concrete simpler rules used as countermodels, not to generate labels."""
    t, l, b, r = info["main"]
    shape = b - t - 1, r - l - 1
    outlines = np.full(shape, info["fill"], dtype=np.int8)
    corners = outlines.copy()
    mono = outlines.copy()
    points = outlines.copy()
    for box, color in zip(info["boxes"], info["colors"]):
        for y, x in _perimeter(box):
            if _inside((y, x), info["main"]):
                outlines[y - t - 1, x - l - 1] = color
        for q in range(4):
            for y, x in _positions(box, q, info["stencil"]):
                if _inside((y, x), info["main"]):
                    corners[y - t - 1, x - l - 1] = color
    for stamp, junction in zip(info["cells"], info["junctions"]):
        y, x, _, _, hi, _ = junction
        points[y - t - 1, x - l - 1] = info["colors"][hi]
        for yy, xx in stamp:
            mono[yy - t - 1, xx - l - 1] = info["colors"][hi]
    return {"B_only_outlines": outlines, "A_only_true_corners": corners,
            "single_pigment": mono, "no_stencil_points": points}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _array_key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            pred, info = _read_input(inp)
            assert np.array_equal(pred, out), "incorrect junction motif crop"
            t, l, b, r = info["main"]
            assert out.shape == (b - t - 1, r - l - 1)
            assert np.all(inp[t + 1:b, l + 1:r] == info["fill"])
            assert np.any(out != info["fill"])
            allowed = {info["fill"], *info["colors"]}
            assert set(map(int, np.unique(out))) <= allowed
            occupied = set().union(*(set(stamp) for stamp in info["cells"]))
            assert np.count_nonzero(out != info["fill"]) == len(occupied)
            assert all(not np.array_equal(alt, out) for name, alt in _alternative_maps(info).items()
                       if name not in ("A_only_true_corners", "single_pigment")
                       or (name == "single_pigment" and info["clipped"] == 0))
            infos[split].append(info)
    # Every count, including 2/1, retains two informative demonstrations.
    first = infos["train"][0]
    assert first["clipped"] == 0
    assert first["stencil"] != frozenset((v, u) for u, v in first["stencil"])
    assert all(not np.array_equal(alt, puzzle["train"][0]["output"])
               for alt in _alternative_maps(first).values())
    assert len({i["stencil"] for i in infos["train"]}) >= 2
    assert all(len(i["junctions"]) == 2 for i in infos["train"])
    assert all(len(i["junctions"]) >= 4 for i in infos["test"])


def _regression_test():
    """Explicit pixels exercise color ties, four rails, clipping, and causality."""
    s = frozenset({(1, 1), (1, 2), (2, 1)})
    big = s | {(1, 3)}
    main = (6, 6, 23, 23)
    boxes = [(1, 1, 18, 18), (10, 10, 28, 28),
             (3, 3, 16, 16), (12, 12, 26, 26)]

    def observed(rects, motif, window=main):
        inp = np.zeros((30, 30), dtype=np.int8)
        for color, box in enumerate(rects, 1):
            for p in _perimeter(box):
                inp[p] = color
        t, l, b, r = window
        inp[t:b + 1, l:r + 1] = 9
        for p in _perimeter(window):
            inp[p] = 8
        for q in (0, 1):
            for p in _positions(rects[0], q, motif):
                inp[p] = 1
        for i, box in enumerate(rects[1:], 2):
            q = 0 if i == 3 else 3
            for p in _positions(box, q, {(1, 1)}):
                inp[p] = i
        return inp

    def expected(assignments, window=main):
        t, l, b, r = window
        out = np.full((b - t - 1, r - l - 1), 9, dtype=np.int8)
        for y, x, pigment in assignments:
            out[y - t - 1, x - l - 1] = pigment
        return out

    # These assignments are explicit hand-derived pixel fixtures. In each
    # triple the first two cells are horizontal-owned and the third vertical.
    pixels = [(17, 11, 1), (17, 12, 1), (16, 11, 2),
              (11, 17, 2), (11, 16, 2), (12, 17, 1),
              (15, 11, 3), (15, 12, 3), (14, 11, 2),
              (11, 15, 2), (11, 14, 2), (12, 15, 3),
              (17, 13, 1), (17, 14, 1), (16, 13, 4),
              (13, 17, 4), (13, 16, 4), (14, 17, 1),
              (15, 13, 3), (15, 14, 3), (14, 13, 4),
              (13, 15, 4), (13, 14, 4), (14, 15, 3)]
    four = observed(boxes, s)
    answer = expected(pixels)
    assert np.array_equal(solve(four), answer)
    assert np.array_equal(solve(np.fliplr(four)), np.fliplr(answer))
    assert np.array_equal(solve(np.flipud(four)), np.flipud(answer))
    permutation = np.array([7, 6, 5, 4, 3, 2, 1, 0, 9, 8], dtype=np.int8)
    assert np.array_equal(solve(permutation[four]), permutation[answer])

    # A 3x3-domain motif remains valid with four interacting rails. These
    # more widely spaced bounds avoid the collisions forbidden by the family.
    roomy = [(1, 1, 20, 20), (10, 10, 28, 28),
             (3, 3, 17, 17), (13, 13, 26, 26)]
    roomy_pixels = [(19, 11, 1), (19, 12, 1), (18, 11, 2), (19, 13, 1),
                    (11, 19, 2), (11, 18, 2), (12, 19, 1), (11, 17, 2),
                    (16, 11, 3), (16, 12, 3), (15, 11, 2), (16, 13, 3),
                    (11, 16, 2), (11, 15, 2), (12, 16, 3), (11, 14, 2),
                    (19, 14, 1), (19, 15, 1), (18, 14, 4), (19, 16, 1),
                    (14, 19, 4), (14, 18, 4), (15, 19, 1), (14, 17, 4),
                    (16, 14, 3), (16, 15, 3), (15, 14, 4), (16, 16, 3),
                    (14, 16, 4), (14, 15, 4), (15, 16, 3), (14, 14, 4)]
    assert np.array_equal(solve(observed(roomy, big)), expected(roomy_pixels))

    # A changes the answer even when all rail geometry is held fixed.
    two = observed(boxes[:2], s)
    small_answer = expected(pixels[:6])
    assert np.array_equal(solve(two), small_answer)
    larger = observed(boxes[:2], big)
    large_answer = expected(pixels[:6] + [(17, 13, 1), (11, 15, 2)])
    assert np.array_equal(solve(larger), large_answer)
    assert not np.array_equal(large_answer, small_answer)

    # B changes an inferred side coordinate without changing the template.
    moved_boxes = [boxes[0], (11, 10, 28, 28)]
    moved = observed(moved_boxes, big)
    moved_answer = expected([(17, 11, 1), (17, 12, 1), (16, 11, 2),
                             (17, 13, 1), (12, 17, 2), (12, 16, 2),
                             (13, 17, 1), (12, 15, 2)])
    assert np.array_equal(solve(moved), moved_answer)
    assert not np.array_equal(moved_answer, large_answer)

    # Two crossings on the same horizontal side teach clipping, without
    # substituting invisible motif cells from a generator-held template.
    clip_window = (16, 6, 24, 24)
    clipped = observed([(1, 1, 18, 28), (10, 10, 28, 20)], s, clip_window)
    clip_answer = expected([(17, 11, 1), (17, 12, 1),
                            (17, 19, 1), (17, 18, 1)], clip_window)
    assert np.array_equal(solve(clipped), clip_answer)
    assert _read_input(clipped)[1]["clipped"] == 2


def self_test():
    """Bounded variation audit over 42 seeds, plus counts and regressions."""
    signatures, shapes, output_shapes, patterns, layouts, palettes = set(), set(), set(), set(), set(), set()
    sizes, orientations, clue_counts, clipping, backgrounds, frames = set(), set(), set(), set(), set(), set()
    _regression_test()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = pair["input"].copy()
                prediction = solve(before)
                assert np.array_equal(prediction, pair["output"])
                assert np.array_equal(before, pair["input"]), "solve mutated its input"
                _, info = _read_input(before)
                shapes.add(before.shape)
                output_shapes.add(prediction.shape)
                patterns.add(tuple(sorted(info["stencil"])))
                layouts.add((before.shape, info["main"], tuple(info["boxes"])))
                palettes.add((info["background"], info["frame"], info["fill"], tuple(info["colors"])))
                sizes.add(len(info["stencil"]))
                orientations.update((dy, dx) for _, _, dy, dx, _, _ in info["junctions"])
                clue_counts.update(len(clue) for _, _, clue in info["occurrences"] if clue != info["stencil"])
                clipping.add(info["clipped"] > 0)
                backgrounds.add(info["background"])
                frames.add(info["frame"])
    assert len(signatures) == 42 and len(layouts) >= 200
    assert len(shapes) >= 35 and len(output_shapes) >= 35
    assert len(patterns) >= 20 and len(palettes) >= 100
    assert len(sizes) >= 5 and len(clue_counts) >= 5
    assert orientations == set(_DIRS) and clipping == {False, True}
    assert backgrounds == set(range(10)) and frames == set(range(10))
    assert _signature(generate_puzzle(909, 2, 1)) == _signature(generate_puzzle(909, 2, 1))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(10000 + 10 * nt + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(1, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(991, 2, 1)
    a = bad["test"][0]["output"]
    a[0, 0] = (int(a[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes),
            "output_shapes": len(output_shapes), "stencils": len(patterns),
            "layouts": len(layouts), "palettes": len(palettes)}


if __name__ == "__main__":
    print(self_test())
