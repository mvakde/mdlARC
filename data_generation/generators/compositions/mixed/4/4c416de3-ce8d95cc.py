"""Repair frame-corner stencils, then contract equal vector runs on both axes.

No parent code, output labels, generation state, or template bank is used by solve.
All tests and sampling are bounded and write no artifacts.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_SEEDS = (0, 547471253, 170777157, 633337474, 367355583, 673214653,
          830448594, 200720226, 792314692, 751201758, 554127940, 903683967,
          972641480, 354895046, 241910015, 276283917, 480243367, 639095777,
          845513876, 280762238, 523886233, 45948855, 782817487, 183427071,
          604616457, 448811122, 559179432, 893363017, 941626150, 648144856,
          542036559, 120696785, 427748352, 337306690, 521589468, 412754993,
          594180414, 2566126, 46624742, 472111597, 446041640, 735003089)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(not 1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _corners(rect):
    t, l, b, r = rect
    return ((t, l, 0), (t, r, 1), (b, l, 2), (b, r, 3))


def _outline(rect):
    t, l, b, r = rect
    return ({(t, x) for x in range(l, r + 1)}
            | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t, b + 1)}
            | {(y, r) for y in range(t, b + 1)})


def _positions(corner, pattern):
    y, x, q = corner
    dy, dx = _DIRS[q]
    return {(y + dy * u, x + dx * v) for u, v in pattern}


def _connected(pattern):
    todo = [min(pattern)]
    seen = set(todo)
    while todo:
        u, v = todo.pop()
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                p = u + du, v + dv
                if p in pattern and p not in seen:
                    todo.append(p)
                    seen.add(p)
    return len(seen) == len(pattern)


def _geometry(a, bg, frame):
    """Recover closed frames from long, unoccluded incident arms."""
    h, w = a.shape
    corners = []
    for y, x in np.argwhere(a != bg):
        y, x = int(y), int(x)
        for q, (dy, dx) in enumerate(_DIRS):
            if not (0 <= y + 3 * dy < h and 0 <= x + 3 * dx < w):
                continue
            if all(a[y + k * dy, x] == frame and a[y, x + k * dx] == frame
                   for k in (2, 3)):
                if a[y + dy, x] != bg and a[y, x + dx] != bg:
                    corners.append((y, x, q))
    assert corners
    corner_set = set(corners)
    rectangles = []
    for t, l, q in corners:
        if q != 0:
            continue
        rights = [x for y, x, k in corners if y == t and k == 1 and x - l >= 6]
        bottoms = [y for y, x, k in corners if x == l and k == 2 and y - t >= 6]
        assert rights and bottoms
        r, b = min(rights), min(bottoms)
        assert (b, r, 3) in corner_set
        rectangles.append((t, l, b, r))
    assert rectangles
    expected = [c for rect in rectangles for c in _corners(rect)]
    assert len(expected) == len(corners) and set(expected) == corner_set
    structure = set()
    for rect in rectangles:
        border = _outline(rect)
        assert not border & structure
        assert all(a[p] != bg for p in border)
        structure |= border
    assert all(tuple(p) in structure for p in np.argwhere(a == frame))
    return sorted(rectangles), sorted(corners), structure


def _complete(value, rotation=False):
    """Parse input alone. Local coordinates point inward at each frame corner."""
    a = _grid(value)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert np.count_nonzero(counts == counts[bg]) == 1, "background must be unique plurality"
    candidates = []
    for frame in np.flatnonzero(counts):
        frame = int(frame)
        if frame == bg:
            continue
        try:
            rects, corners, structure = _geometry(a, bg, frame)
            assigned = {c: [] for c in corners}
            payload = set()
            for y, x in np.argwhere((a != bg) & (a != frame)):
                y, x = int(y), int(x)
                choices = []
                for c in corners:
                    cy, cx, q = c
                    dy, dx = _DIRS[q]
                    u, v = (y - cy) * dy, (x - cx) * dx
                    if -3 <= u <= 1 and -3 <= v <= 1:
                        choices.append((c, u, v))
                if choices:
                    assert len(choices) == 1, "ambiguous ornament association"
                    c, u, v = choices[0]
                    assigned[c].append((u, v, int(a[y, x])))
                else:
                    assert any(t + 3 <= y <= b - 3 and l + 3 <= x <= r - 3
                               for t, l, b, r in rects), "unexplained foreground"
                    payload.add((y, x))
            occurrences = []
            for c, entries in assigned.items():
                if entries:
                    colors = {k for _, _, k in entries}
                    assert len(colors) == 1
                    clue = frozenset((u, v) for u, v, _ in entries)
                    if rotation and c[2] in (1, 2):
                        clue = frozenset((v, u) for u, v in clue)
                    occurrences.append((c, colors.pop(), clue))
            assert len(occurrences) >= 2
            maxima = {p for _, _, p in occurrences
                      if all(q <= p for _, _, q in occurrences)}
            assert len(maxima) == 1, "no intact common stencil"
            stencil = maxima.pop()
            assert 5 <= len(stencil) <= 25 and _connected(stencil)
            result = a.copy()
            occupied = set()
            for c, color, clue in occurrences:
                local = (frozenset((v, u) for u, v in stencil)
                         if rotation and c[2] in (1, 2) else stencil)
                positions = _positions(c, local)
                assert not positions & occupied
                for y, x in positions:
                    assert 0 <= y < a.shape[0] and 0 <= x < a.shape[1]
                    assert int(a[y, x]) in (bg, frame, color)
                    # Stencils never touch another frame, or the interior payload.
                    own = next(rect for rect in rects if c in _corners(rect))
                    assert (y, x) not in structure - _outline(own)
                    assert (y, x) not in payload
                    result[y, x] = color
                occupied |= positions
            candidates.append((result, {"background": bg, "frame": frame,
                "rectangles": rects, "corners": corners, "occurrences": occurrences,
                "stencil": stencil, "painted": occupied, "payload": payload}))
        except AssertionError:
            continue
    if len(candidates) != 1:
        raise ValueError("input does not have one unambiguous frame/stencil interpretation")
    return candidates[0]


def _keeps(a):
    # Both masks refer to the SAME completed canvas. Neither axis has priority.
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    cols = np.r_[True, np.any(a[:, 1:] != a[:, :-1], axis=0)]
    return rows, cols


def _contract(a):
    rows, cols = _keeps(a)
    return a[np.ix_(rows, cols)].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the intact stencil, repair all marked corners, contract equal runs."""
    completed, _ = _complete(input_grid)
    return _contract(completed)


def _sample_stencil(rng, asymmetric=False):
    """Grow a connected symbol, expanding its exterior coordinate into strips.

    This is NOT a solver vocabulary: solve reads the actual intact occurrence.
    Expanded exterior rows/columns give constructive vector-equality witnesses.
    """
    symmetric = not asymmetric and rng.random() < .35
    ry = int(rng.integers(2, 4))
    rx = ry if symmetric else int(rng.integers(2, 4))
    domain = {(u, v) for u in (-1, 0, 1) for v in (-1, 0, 1)}
    for _ in range(80):
        core = {(-1, -1)}
        target = int(rng.integers(3, 9))
        for _ in range(9):
            frontier = []
            for p in sorted(domain - core):
                extra = {p, p[::-1]} if symmetric else {p}
                if _connected(core | extra):
                    frontier.append(extra)
            if len(core) >= target or not frontier:
                break
            core |= frontier[int(rng.integers(len(frontier)))]
        if not any(u == 1 or v == 1 for u, v in core):
            continue
        points = frozenset((a, b) for u, v in core
                           for a in (range(-ry, 0) if u == -1 else (u,))
                           for b in (range(-rx, 0) if v == -1 else (v,)))
        if len(points) >= 5 and (not asymmetric or points != frozenset((v, u) for u, v in points)):
            return points, ry, rx
    raise RuntimeError("stencil growth budget exhausted")


def _sample_rectangles(rng, ry, rx, witness=False):
    # Free placement with bounding-box clearance, rather than a fixed panel layout.
    target = 1 if witness else int(rng.choice((1, 1, 1, 2, 2, 3, 4)))
    if target >= 3:
        h, w = int(rng.integers(27, 31)), int(rng.integers(27, 31))
    elif target == 2:
        h, w = int(rng.integers(22, 31)), int(rng.integers(22, 31))
    else:
        h, w = int(rng.integers(13, 31)), int(rng.integers(13, 31))
    rects = []
    for _ in range(target):
        for attempt in range(180):
            minimum = 11 if witness else 7
            cap_h = h - 2 * ry
            cap_w = w - 2 * rx
            if target > 1:
                cap_h, cap_w = min(cap_h, 10), min(cap_w, 10)
            if min(cap_h, cap_w) < minimum:
                continue
            sh = int(rng.integers(minimum, cap_h + 1))
            sw = int(rng.integers(minimum, cap_w + 1))
            t = int(rng.integers(ry, h - ry - sh + 1))
            l = int(rng.integers(rx, w - rx - sw + 1))
            rect = (t, l, t + sh - 1, l + sw - 1)
            if all(rect[2] + 2 * ry + 2 <= other[0]
                   or other[2] + 2 * ry + 2 <= rect[0]
                   or rect[3] + 2 * rx + 2 <= other[1]
                   or other[3] + 2 * rx + 2 <= rect[1] for other in rects):
                rects.append(rect)
                break
        else:
            break
    assert rects
    return (h, w), rects


def _add_threads(rng, base, rect, palette, witness=False):
    t, l, b, r = rect
    if min(b - t, r - l) < 8 or (not witness and rng.random() < .45):
        return
    height, width = b - t - 5, r - l - 5
    nh = min(2 if witness else int(rng.integers(1, 4)), (height + 1) // 2)
    nv = min(2 if witness else int(rng.integers(1, 4)), (width + 1) // 2)
    ys = sorted(int(y) for y in rng.choice(np.arange(t + 3, b - 2, 2), nh, replace=False))
    xs = sorted(int(x) for x in rng.choice(np.arange(l + 3, r - 2, 2), nv, replace=False))
    hc, vc = map(int, rng.choice(palette, 2, replace=False))
    for y in ys:
        base[y, l + 3:r - 2] = hc
    for x in xs:
        base[t + 3:b - 2, x] = vc
    mode = int(rng.integers(4))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            horizontal = ((i + j) % 2 == 0 if witness else
                          mode == 0 or (mode > 1 and rng.random() < .5))
            base[y, x] = hc if horizontal else vc


def _new_equalities(a, completed):
    old_rows, old_cols = _keeps(a)
    rows, cols = _keeps(completed)
    # Non-background witness requirement is checked separately in validation.
    return np.flatnonzero(old_rows & ~rows), np.flatnonzero(old_cols & ~cols)


def _make_pair(rng, sparse=False, witness=False, leave_unmarked=False):
    stencil, ry, rx = _sample_stencil(rng, asymmetric=witness)
    shape, rects = _sample_rectangles(rng, ry, rx, witness)
    bg, frame = map(int, rng.choice(10, 2, replace=False))
    palette = [c for c in range(10) if c not in (bg, frame)]
    base = np.full(shape, bg, dtype=np.int8)
    for rect in rects:
        for p in _outline(rect):
            base[p] = frame
        _add_threads(rng, base, rect, palette, witness)
    marked = []
    for index, rect in enumerate(rects):
        cs = _corners(rect)
        if index == 0:
            n = 3 if witness else int(rng.integers(2, 4 if leave_unmarked else 5))
        else:
            n = int(rng.integers(0, 5))
        indices = list(map(int, rng.permutation(4)[:n]))
        marked += [cs[q] for q in indices]
    # Three distinct corners in the witness necessarily include adjacent orientations.
    ref = int(rng.integers(len(marked)))
    ncolors = int(rng.integers(1, min(5, len(marked)) + 1))
    colors = list(map(int, rng.choice(palette, ncolors, replace=False)))
    inp, completed = base.copy(), base.copy()
    for i, c in enumerate(marked):
        color = int(rng.choice(colors))
        positions = sorted(_positions(c, stencil))
        for p in positions:
            completed[p] = color
        if i == ref or (i != (ref + 1) % len(marked) and rng.random() < .15):
            retained = positions
        else:
            # Deliberately separate clue density at train/test, but expose every
            # test's full stencil locally. Deletion never erases a marked copy.
            low, high = ((.10, .35) if sparse else (.50, .85))
            keep = max(1, min(len(positions) - 1, int(len(positions) * rng.uniform(low, high))))
            retained = [positions[int(k)] for k in rng.permutation(len(positions))[:keep]]
        for p in retained:
            inp[p] = color
    assert np.any(inp != completed)
    predicted, info = _complete(inp)
    assert np.array_equal(predicted, completed)
    assert info["stencil"] == stencil and info["rectangles"] == sorted(rects)
    new_rows, new_cols = _new_equalities(inp, completed)
    assert len(new_rows) and len(new_cols), "completion must CREATE equality on both axes"
    if witness:
        try:
            alternate = _contract(_complete(inp, rotation=True)[0])
        except ValueError:
            alternate = None
        assert alternate is None or not np.array_equal(alternate, _contract(completed))
    return {"input": inp, "output": _contract(completed)}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            for _ in range(160):
                try:
                    pair = _make_pair(rng, sparse=split == "test",
                                      witness=split == "train" and i == 0,
                                      leave_unmarked=split == "train" and i == 1)
                    key = _key(pair["input"])
                    assert key not in seen
                    if split == "train" and i == 1:
                        first = _complete(puzzle["train"][0]["input"])[1]
                        current = _complete(pair["input"])[1]
                        assert current["stencil"] != first["stencil"]
                        assert pair["input"].shape != puzzle["train"][0]["input"].shape
                    break
                except (AssertionError, ValueError):
                    continue
            else:
                raise RuntimeError("bounded scene-generation budget exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _has_separated_repeat(a, axis):
    vectors = a if axis == 0 else a.T
    runs = [vectors[0].tobytes()]
    for v in vectors[1:]:
        if v.tobytes() != runs[-1]:
            runs.append(v.tobytes())
    return len(set(runs)) < len(runs)


def _thread_crossings(a, info):
    """Recover generated threads and their literal crossings from input pixels.

    The solver preserves arbitrary inset content; this stricter check describes
    the sampled woven motifs and verifies the training ownership witness.
    """
    crossings, payload = [], set()
    bg = info["background"]
    for t, l, b, r in info["rectangles"]:
        tile = a[t + 3:b - 2, l + 3:r - 2]
        if np.all(tile == bg):
            continue
        ys = np.flatnonzero(np.all(tile != bg, axis=1))
        xs = np.flatnonzero(np.all(tile != bg, axis=0))
        free_y = [i for i in range(tile.shape[0]) if i not in ys]
        free_x = [j for j in range(tile.shape[1]) if j not in xs]
        assert 1 <= len(ys) <= 3 and 1 <= len(xs) <= 3 and free_y and free_x
        assert np.all(np.diff(ys) >= 2) and np.all(np.diff(xs) >= 2)
        hc, vc = int(tile[ys[0], free_x[0]]), int(tile[free_y[0], xs[0]])
        assert len({bg, info["frame"], hc, vc}) == 4
        assert np.all(tile[np.ix_(ys, free_x)] == hc)
        assert np.all(tile[np.ix_(free_y, xs)] == vc)
        assert np.all(tile[np.ix_(free_y, free_x)] == bg)
        for y in ys:
            for x in xs:
                assert int(tile[y, x]) in (hc, vc)
                crossings.append((t + 3 + int(y), l + 3 + int(x), hc, vc))
        payload |= {(t + 3 + int(y), l + 3 + int(x))
                    for y, x in np.argwhere(tile != bg)}
    assert payload == info["payload"]
    return crossings


def _check_training_countermodels(first, second):
    """Check the two required anchors, including alternatives named in the spec."""
    a, out = _grid(first["input"]), _grid(first["output"])
    completed, info = _complete(a)
    crossings = _thread_crossings(a, info)
    assert len(crossings) == 4, "first demonstration needs a 2-by-2 crossing witness"
    assert {int(a[y, x]) == hc for y, x, hc, _ in crossings} == {False, True}
    for owner in (2, 3):
        alternative = completed.copy()
        for crossing in crossings:
            y, x = crossing[:2]
            alternative[y, x] = crossing[owner]
        assert not np.array_equal(_contract(alternative), out), "crossing priority fits training"
    assert info["stencil"] != frozenset((v, u) for u, v in info["stencil"])
    assert any(c[2] ^ d[2] in (1, 2) for c, _, _ in info["occurrences"]
               for d, _, _ in info["occurrences"])
    try:
        alternate = _contract(_complete(a, rotation=True)[0])
    except ValueError:
        alternate = None
    assert alternate is None or not np.array_equal(alternate, out), "reflection not taught"

    b, expected = _grid(second["input"]), _grid(second["output"])
    filled, second_info = _complete(b)
    marked = {c for c, _, _ in second_info["occurrences"]}
    unmarked = set(second_info["corners"]) - marked
    assert unmarked
    color = second_info["occurrences"][0][1]
    for corner in unmarked:
        for point in _positions(corner, second_info["stencil"]):
            filled[point] = color
    assert not np.array_equal(_contract(filled), expected), "filling unmarked corners fits training"


def _features(pair):
    a, out = _grid(pair["input"]), _grid(pair["output"])
    completed, info = _complete(a)
    rows, cols = _keeps(completed)
    new_rows, new_cols = _new_equalities(a, completed)
    partials = [len(p) / len(info["stencil"]) for _, _, p in info["occurrences"]
                if p != info["stencil"]]
    return {"input_shape": list(a.shape), "output_shape": list(out.shape),
            "frames": len(info["rectangles"]), "marks": len(info["occurrences"]),
            "stencil_size": len(info["stencil"]), "stencil": sorted(info["stencil"]),
            "background": info["background"], "frame_color": info["frame"],
            "accents": len({c for _, c, _ in info["occurrences"]}),
            "new_equal_rows": len(new_rows), "new_equal_columns": len(new_cols),
            "partial_fractions": partials, "payload_cells": len(info["payload"]),
            "separated_row_repeat": _has_separated_repeat(completed, 0),
            "separated_column_repeat": _has_separated_repeat(completed, 1),
            "unmarked": len(info["corners"]) - len(info["occurrences"]),
            "rows_removed": int((~rows).sum()), "columns_removed": int((~cols).sum())}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen = set()
    fractions = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(a)
            assert key not in seen
            seen.add(key)
            completed, info = _complete(a)
            rows, cols = _keeps(completed)
            expected = completed[np.ix_(rows, cols)]
            assert np.array_equal(expected, out), "incorrect completed-canvas contraction"
            assert np.all(np.isin(a[a != completed], (info["background"], info["frame"])))
            assert all(tuple(p) in info["painted"] for p in np.argwhere(a != completed))
            assert any(p == info["stencil"] for _, _, p in info["occurrences"])
            assert set(np.unique(a)) == set(np.unique(completed))
            protected = np.ones(a.shape, dtype=bool)
            for point in info["painted"]:
                protected[point] = False
            assert np.array_equal(completed[protected], a[protected])
            _thread_crossings(a, info)
            for axis, new in enumerate(_new_equalities(a, completed)):
                assert len(new), "A did not create full-vector equality on both axes"
                vectors = completed if axis == 0 else completed.T
                assert all(np.any(vectors[i] != info["background"]) for i in new)
                assert np.all(np.any(np.diff(out, axis=axis) != 0, axis=1 - axis))
            assert out.shape[0] < a.shape[0] and out.shape[1] < a.shape[1]
            assert not np.array_equal(_contract(a), out), "omitting A must change the answer"
            assert not np.array_equal(completed, out), "omitting B must change the answer"
            # Exact lossless reconstruction proves retained crossing/other values
            # are copied, and that the two axes were contracted independently.
            ri = np.cumsum(rows) - 1
            ci = np.cumsum(cols) - 1
            assert np.array_equal(out[np.ix_(ri, ci)], completed)
            for _, _, clue in info["occurrences"]:
                if clue != info["stencil"]:
                    fractions[split].append(len(clue) / len(info["stencil"]))
    assert min(fractions["train"]) > max(fractions["test"]), "missing clue-density shift"
    first, second = puzzle["train"][:2]
    assert _grid(first["input"]).shape != _grid(second["input"]).shape
    first_info, second_info = _complete(first["input"])[1], _complete(second["input"])[1]
    assert first_info["stencil"] != second_info["stencil"]
    assert second_info["corners"] and len(second_info["occurrences"]) < len(second_info["corners"])
    assert any(_has_separated_repeat(_complete(p["input"])[0], axis)
               for p in puzzle["train"] for axis in (0, 1)), "teach retention of separated repeats"
    _check_training_countermodels(first, second)


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds in memory, count boundaries, countermodels, mutation checks."""
    signatures, shapes, structures, stencils = set(), set(), set(), set()
    frames, sizes, backgrounds, crossing_pairs = set(), set(), set(), 0
    new_rows = new_cols = total = 0
    for seed in _SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                total += 1
                shapes.add(tuple(f["input_shape"]))
                frames.add(f["frames"])
                sizes.add(f["stencil_size"])
                stencils.add(tuple(map(tuple, f["stencil"])))
                backgrounds.add(f["background"])
                crossing_pairs += f["payload_cells"] > 0
                new_rows += f["new_equal_rows"]
                new_cols += f["new_equal_columns"]
                roles = {}
                normalized = [[roles.setdefault(int(c), len(roles)) for c in row]
                              for row in pair["input"]]
                structures.add(json.dumps(normalized, separators=(",", ":")))
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 50 and len(stencils) >= 40 and len(sizes) >= 10
    assert {1, 2} <= frames and len(backgrounds) == 10 and crossing_pairs >= 42
    assert new_rows >= total and new_cols >= total
    assert _signature(generate_puzzle(71, 2, 1)) == _signature(generate_puzzle(71, 2, 1))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(100 + 10 * nt + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Pure B regressions: nonconsecutive equal vectors survive; literal crossings
    # are never repainted, and row-first/column-first contraction agree.
    raw = np.array([[1, 1, 2, 2, 1], [1, 1, 2, 2, 1],
                    [3, 3, 4, 4, 3], [1, 1, 2, 2, 1]], dtype=np.int8)
    expected = np.array([[1, 2, 1], [3, 4, 3], [1, 2, 1]], dtype=np.int8)
    assert np.array_equal(_contract(raw), expected)
    assert np.array_equal(_contract(raw.T).T, expected)
    equal_histograms = np.array([[1, 1, 2, 2, 1], [1, 2, 1, 2, 1]], dtype=np.int8)
    assert np.array_equal(np.sort(equal_histograms[0]), np.sort(equal_histograms[1]))
    assert np.array_equal(_contract(equal_histograms), equal_histograms)
    sample = generate_puzzle(0, 2, 1)
    validate_generated_puzzle({s: [{k: v.tolist() for k, v in p.items()} for p in sample[s]]
                               for s in sample})
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]] for s in sample}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": len(_SEEDS), "pairs": total, "input_shapes": len(shapes),
            "color_normalized_inputs": len(structures), "stencils": len(stencils),
            "stencil_sizes": sorted(sizes), "frame_counts": sorted(frames),
            "pairs_with_threads": crossing_pairs, "new_equal_rows": new_rows,
            "new_equal_columns": new_cols}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
