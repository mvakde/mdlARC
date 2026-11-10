"""Extreme-height rectangles control duplication of shared canvas center axes.

Only NumPy and the Python standard library are used.  No parent files are read.
"""

from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a


def _measure(a):
    """Return background and inclusive boxes, using only visible input cells."""
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    background = int(border[0])
    if not np.all(border == background):
        raise ValueError("the input perimeter must be uniform background")
    boxes = {}
    for raw in np.unique(a):
        color = int(raw)
        if color != background:
            rr, cc = np.where(a == color)
            boxes[color] = (int(rr.min()), int(cc.min()),
                            int(rr.max()), int(cc.max()))
    if len(boxes) < 3:
        raise ValueError("at least three rectangles are required")
    heights = [b[2] - b[0] + 1 for b in boxes.values()]
    if len(set(heights)) != len(heights):
        raise ValueError("rectangle heights must be distinct")
    if any((b[2] - b[0]) % 2 or (b[3] - b[1]) % 2 for b in boxes.values()):
        raise ValueError("rectangle dimensions must be odd")
    return background, boxes


def _rank(boxes):
    return sorted(boxes, key=lambda color: boxes[color][2] - boxes[color][0])


def _axes(boxes, pins=None, row_anchor="center", col_anchor="center"):
    if pins is None:
        ordered = _rank(boxes)
        pins = [ordered[0], ordered[-1]]
    rows, cols = [], []
    for color in pins:
        top, left, bottom, right = boxes[color]
        rows.append(top if row_anchor == "top" else bottom if row_anchor == "bottom"
                    else (top + bottom) // 2)
        cols.append(left if col_anchor == "left" else right if col_anchor == "right"
                    else (left + right) // 2)
    return rows, cols


def _warp(a, rows, cols, additive=False):
    row_counts = np.ones(a.shape[0], dtype=int)
    col_counts = np.ones(a.shape[1], dtype=int)
    for row in rows if additive else set(rows):
        row_counts[row] += 1
    for col in cols if additive else set(cols):
        col_counts[col] += 1
    return np.repeat(np.repeat(a, row_counts, axis=0), col_counts, axis=1).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the scene's full spans, then copy its visible sheet along pin axes."""
    a = _array(input_grid)
    _, boxes = _measure(a)
    rows, cols = _axes(boxes)
    result = _warp(a, rows, cols)
    if max(result.shape) > 30:
        raise ValueError("expanded grid exceeds ARC dimensions")
    return result


def _layerable(a, background, boxes):
    """Acyclic covering constraints and exact reconstruction of visible boxes."""
    above = {}
    for color, (top, left, bottom, right) in boxes.items():
        window = a[top:bottom + 1, left:right + 1]
        if np.any(window == background):
            return False
        above[color] = set(map(int, np.unique(window))) - {color}
    pending, order = set(boxes), []
    # Peel topmost boxes; reversing this order gives a valid drawing order.
    while pending:
        topmost = sorted(c for c in pending if not (above[c] & pending))
        if not topmost:
            return False
        order.extend(topmost)
        pending.difference_update(topmost)
    rebuilt = np.full(a.shape, background, dtype=a.dtype)
    for color in reversed(order):
        top, left, bottom, right = boxes[color]
        rebuilt[top:bottom + 1, left:right + 1] = color
    return bool(np.array_equal(rebuilt, a))


def _overlaps(boxes):
    bs = list(boxes.values())
    return [((min(a[2], b[2]) - max(a[0], b[0]) + 1)
             * (min(a[3], b[3]) - max(a[1], b[1]) + 1))
            for i, a in enumerate(bs) for b in bs[:i]
            if min(a[2], b[2]) >= max(a[0], b[0])
            and min(a[3], b[3]) >= max(a[1], b[1])]


def _gaps(a, boxes):
    return sum(not np.any(a[row] == color)
               for color, b in boxes.items() for row in range(b[0] + 1, b[2]))


def _hidden_pin_center(a, boxes):
    ordered = _rank(boxes)
    return any(not np.any(a[(boxes[c][0] + boxes[c][2]) // 2] == c)
               for c in (ordered[0], ordered[-1]))


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _external_copy_effects(a, boxes):
    """Do pin lines copy other objects beyond both controlling rectangles?"""
    ordered = _rank(boxes)
    pins = {ordered[0], ordered[-1]}
    background = int(a[0, 0])
    rows, cols = _axes(boxes)
    external = (a != background) & ~np.isin(a, list(pins))
    for color in pins:
        top, left, bottom, right = boxes[color]
        external[top:bottom + 1, left:right + 1] = False
    return (any(np.any(external[r, :]) for r in rows),
            any(np.any(external[:, c]) for c in cols))


def _alternatives(a, boxes):
    """Plausible ablations for checking finite demonstration sufficiency."""
    ranked = _rank(boxes)
    out = {"identity": a, "uniform_scaling": np.repeat(np.repeat(a, 2, 0), 2, 1),
           "canvas_border": _warp(a, [0, a.shape[0] - 1], [0, a.shape[1] - 1])}
    rows, cols = _axes(boxes)
    out["rows_only"] = _warp(a, rows, [])
    out["columns_only"] = _warp(a, [], cols)
    all_rows, all_cols = _axes(boxes, list(boxes))
    out["spatial_outer_centers"] = _warp(a, [min(all_rows), max(all_rows)],
                                        [min(all_cols), max(all_cols)])
    for name, pins in (("shortest_only", ranked[:1]), ("tallest_only", ranked[-1:]),
                       ("all_objects", ranked), ("median_only", [ranked[len(ranked) // 2]])):
        out[name] = _warp(a, *_axes(boxes, pins))
    metrics = {
        "visible_rows": lambda c: len(np.unique(np.where(a == c)[0])),
        "visible_area": lambda c: int(np.count_nonzero(a == c)),
        "width": lambda c: boxes[c][3] - boxes[c][1] + 1,
        "spatial_top": lambda c: boxes[c][0],
        "spatial_left": lambda c: boxes[c][1],
    }
    for name, metric in metrics.items():
        ordered = sorted(boxes, key=lambda c: (metric(c), c))
        out[name] = _warp(a, *_axes(boxes, [ordered[0], ordered[-1]]))
    for name, r, c in (("top_left", "top", "left"),
                       ("bottom_right", "bottom", "right"),
                       ("top_row", "top", "center"),
                       ("bottom_row", "bottom", "center"),
                       ("left_column", "center", "left"),
                       ("right_column", "center", "right")):
        out[name] = _warp(a, *_axes(boxes, row_anchor=r, col_anchor=c))
    out["per_pin_multiplicity"] = _warp(a, *_axes(boxes), additive=True)
    return out


def _defeated(pair):
    a, b = pair["input"], pair["output"]
    _, boxes = _measure(a)
    return {name for name, prediction in _alternatives(a, boxes).items()
            if not np.array_equal(prediction, b)}


def _finish(rng, background, heights, widths, positions, order):
    """Render, while ensuring all true box boundaries remain input-visible."""
    n = len(heights)
    colors = list(map(int, rng.choice([c for c in range(10) if c != background], n,
                                      replace=False)))
    min_r = min(p[0] for p in positions)
    min_c = min(p[1] for p in positions)
    max_r = max(p[0] + h for p, h in zip(positions, heights))
    max_c = max(p[1] + w for p, w in zip(positions, widths))
    extent_r, extent_c = max_r - min_r, max_c - min_c
    if max(extent_r, extent_c) > 26:
        return None
    margins = []
    for extent in (extent_r, extent_c):
        first = int(rng.integers(1, min(5, 27 - extent) + 1))
        second = int(rng.integers(1, min(5, 28 - extent - first) + 1))
        margins.append((first, second))
    (up, down), (left_pad, right_pad) = margins
    a = np.full((extent_r + up + down, extent_c + left_pad + right_pad),
                background, dtype=np.int8)
    true_boxes = {}
    for i in order:
        top = positions[i][0] - min_r + up
        left = positions[i][1] - min_c + left_pad
        a[top:top + heights[i], left:left + widths[i]] = colors[i]
        true_boxes[colors[i]] = (top, left, top + heights[i] - 1, left + widths[i] - 1)
    try:
        _, visible = _measure(a)
    except ValueError:
        return None
    if visible != true_boxes or not _layerable(a, background, visible):
        return None
    if rng.random() < 0.5:
        a = a[::-1, :]
    if rng.random() < 0.5:
        a = a[:, ::-1]
    a = np.ascontiguousarray(a)
    return {"input": a, "output": solve(a)}


def _sample_candidate(rng, background, n, mode, cap, hidden_pin=False):
    odd_widths = np.arange(1, 14, 2)
    widths = list(map(int, rng.choice(odd_widths, n)))
    positions = [None] * n
    if mode == "row_witness":
        # The shortest covers a central band. Acceptance below checks that the
        # missing rows change pin selection. Only the center rows coincide.
        low = int(rng.choice([3, 5]))
        heights = list(range(low, low + 2 * n, 2))
        victim, cover = n - 1, 0
        widths[victim] = int(rng.choice([1, 3, 5]))
        widths[cover] = int(rng.choice(np.arange(widths[victim] + 2, 14, 2)))
        positions[victim] = (0, 0)
        positions[cover] = ((heights[victim] - heights[cover]) // 2,
                            int(rng.choice([0, widths[victim] - widths[cover]])))
        order = [victim, cover]
    else:
        heights = list(map(int, rng.choice(np.arange(1, cap + 1, 2), n, replace=False)))
        eligible = [i for i in range(n) if any(h <= heights[i] - 2 for h in heights)]
        victim = int(np.argmax(heights)) if hidden_pin or mode == "column_witness" else int(rng.choice(eligible))
        lower = [i for i in range(n) if heights[i] <= heights[victim] - 2]
        if mode == "joint" and len(lower) < 2:
            mode = "split"
        widths[victim] = int(rng.choice([3, 5, 7]))
        positions[victim] = (0, 0)
        order = [victim]
        if mode == "joint":
            coverers = list(map(int, rng.choice(lower, 2, replace=False)))
            pivot = int(rng.integers(1, widths[victim] - 1))
            for k, cover in enumerate(coverers):
                center = heights[victim] // 2
                lo = max(1, center - heights[cover] + 1)
                hi = min(center, heights[victim] - heights[cover] - 1)
                if hi < lo:
                    return None
                row = int(rng.integers(lo, hi + 1))
                minimum = pivot + 1 if k == 0 else widths[victim] - pivot
                possible = [w for w in odd_widths if w >= minimum + 1]
                widths[cover] = int(rng.choice(possible))
                col = pivot + 1 - widths[cover] if k == 0 else pivot
                positions[cover] = (row, col)
                order.append(cover)
        else:
            cover = (int(np.argmin(heights)) if mode == "column_witness" else
                     int(rng.choice(lower if mode == "split" else
                                    [i for i in range(n) if i != victim])))
            if mode in ("split", "column_witness"):
                widths[cover] = int(rng.choice(np.arange(widths[victim], 14, 2)))
                lo, hi = 1, heights[victim] - heights[cover] - 1
                if hidden_pin:
                    center = heights[victim] // 2
                    lo, hi = max(lo, center - heights[cover] + 1), min(hi, center)
                if hi < lo:
                    return None
                possible_rows = [r for r in range(lo, hi + 1)
                                 if mode != "column_witness"
                                 or r != (heights[victim] - heights[cover]) // 2]
                if not possible_rows:
                    return None
                row = int(rng.choice(possible_rows))
                offset = widths[cover] - widths[victim]
                col = (-offset // 2 if mode == "column_witness" or rng.random() < 0.3 else
                       -int(rng.integers(0, offset + 1)))
            else:
                row = int(rng.integers(-heights[cover] + 1, heights[victim]))
                col = int(rng.integers(-widths[cover] + 1, widths[victim]))
            positions[cover] = (row, col)
            order.append(cover)

    if mode == "column_witness":
        # Two middle objects expose global effects without relying on rare
        # random arrangements: a lateral arm crosses a selected row outside the
        # pins, and a narrow stalk crosses their shared column above both pins.
        extras = [i for i in range(n) if positions[i] is None]
        rng.shuffle(extras)
        arm, stalk = extras[:2]
        center_col = widths[victim] // 2
        left = center_col + 1
        far_right = max(widths[victim] - 1, positions[cover][1] + widths[cover] - 1)
        widths[arm] = int(rng.choice([w for w in odd_widths if left + w - 1 > far_right]))
        widths[stalk] = 1
        positions[arm] = (heights[victim] // 2 - heights[arm] // 2, left)
        positions[stalk] = (2 - heights[stalk], center_col)
        order = [victim, arm, cover, stalk]

    extras = [i for i in range(n) if positions[i] is None]
    rng.shuffle(extras)
    for i in extras:
        # Independent free groups coexist with dense overlapping clusters.
        if mode != "row_witness" and rng.random() < (0.84 if hidden_pin else 0.62):
            anchor = int(rng.choice(order))
            ar, ac = positions[anchor]
            positions[i] = (ar + int(rng.integers(-heights[i] + 1, heights[anchor])),
                            ac + int(rng.integers(-widths[i] + 1, widths[anchor])))
        else:
            positions[i] = (int(rng.integers(-9, 11)), int(rng.integers(-10, 12)))
        order.append(i)
    return _finish(rng, background, heights, widths, positions, order)


def _make_pair(rng, background, kind, n=None, missing=frozenset()):
    for _ in range(1800):
        count = n if n is not None else int(rng.integers(3, 5 if kind == "first" else 6))
        mode = ("row_witness" if kind == "first" else "column_witness" if kind == "second" else
                str(rng.choice(["partial", "split", "joint"], p=[0.32, 0.46, 0.22])))
        hidden_pin = kind == "test_first"
        if hidden_pin and mode == "partial":
            mode = "split"
        pair = _sample_candidate(rng, background, count, mode,
                                 13 if kind.startswith("test") else 11, hidden_pin)
        if pair is None:
            continue
        a = pair["input"]
        _, boxes = _measure(a)
        rows, cols = _axes(boxes)
        if not _overlaps(boxes):
            continue
        if kind == "first":
            if (len(set(rows)) != 1 or len(set(cols)) != 2
                    or not _hidden_pin_center(a, boxes)
                    or "visible_rows" not in _defeated(pair)):
                continue
        if kind == "second":
            if len(set(rows)) != 2 or len(set(cols)) != 1:
                continue
            if not all(_external_copy_effects(a, boxes)):
                continue
            if not missing <= _defeated(pair):
                continue
        if kind.startswith("test"):
            if _gaps(a, boxes) == 0 or len(_overlaps(boxes)) < 2:
                continue
            if hidden_pin and not _hidden_pin_center(a, boxes):
                continue
        return pair
    raise RuntimeError("could not sample a recoverable scene with the requested evidence")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train_count, test_count = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    background = 0 if rng.random() < 0.7 else int(rng.integers(1, 10))
    first = _make_pair(rng, background, "first")
    _, boxes = _measure(first["input"])
    names = set(_alternatives(first["input"], boxes))
    second = _make_pair(rng, background, "second", n=int(rng.integers(4, 6)),
                        missing=names - _defeated(first))
    puzzle = {"train": [first, second], "test": []}
    used = {_key(first["input"]), _key(second["input"])}
    for split, count in (("train", train_count), ("test", test_count)):
        while len(puzzle[split]) < count:
            if split == "train":
                kind, n = "ordinary", int(rng.integers(3, 6))
            else:
                kind = "test_first" if not puzzle["test"] else "test"
                max_train_objects = max(len(_measure(p["input"])[1]) for p in puzzle["train"])
                n = int(rng.integers(max_train_objects + 1, 7))
            pair = _make_pair(rng, background, kind, n)
            if _key(pair["input"]) in used:
                continue
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split types")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "pair counts")
    seen, backgrounds, train_counts = set(), set(), []
    defeated, all_names = set(), set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            a, b = _array(pair["input"]), _array(pair["output"])
            _require(3 <= min(a.shape) and max(a.shape) <= 28, "input bounds")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            background, boxes = _measure(a)
            backgrounds.add(background)
            _require(3 <= len(boxes) <= 9, "object count")
            _require(_layerable(a, background, boxes), "not a recoverable rectangle layering")
            _require(bool(_overlaps(boxes)), "missing occlusion")
            rows, cols = _axes(boxes)
            _require(b.shape == (a.shape[0] + len(set(rows)), a.shape[1] + len(set(cols))),
                     "wrong expanded dimensions")
            # A separate address construction verifies exact copying and original
            # coordinate semantics, including the Cartesian crossing of seams.
            ri = [r for r in range(a.shape[0]) for _ in range(2 if r in rows else 1)]
            ci = [c for c in range(a.shape[1]) for _ in range(2 if c in cols else 1)]
            _require(np.array_equal(b, a[np.ix_(ri, ci)]), "wrong visible-sheet copying")
            _require(set(map(int, np.unique(a))) == set(map(int, np.unique(b))), "color loss")
            _require(np.array_equal(b, solve(a)), "solve mismatch")
            if split == "train":
                train_counts.append(len(boxes))
                if index < 2:
                    all_names.update(_alternatives(a, boxes))
                    defeated.update(_defeated(pair))
                if index == 0:
                    _require(len(set(rows)) == 1 and len(set(cols)) == 2, "first witness axes")
                    _require(_hidden_pin_center(a, boxes), "first witness lacks hidden center")
                    _require("visible_rows" in _defeated(pair), "first witness lacks A ablation")
                elif index == 1:
                    _require(len(set(rows)) == 2 and len(set(cols)) == 1, "second witness axes")
                    _require(all(_external_copy_effects(a, boxes)), "global copying not demonstrated")
            else:
                _require(len(boxes) > max(train_counts), "missing object-count test shift")
                _require(_gaps(a, boxes) > 0 and len(_overlaps(boxes)) >= 2, "missing dense test occlusion")
                if index == 0:
                    _require(_hidden_pin_center(a, boxes), "test lacks hidden pin center")
    _require(len(backgrounds) == 1, "background changes within puzzle")
    _require(all_names <= defeated, "training leaves a checked shortcut indistinguishable")


def _canonical(a):
    labels = {}
    return a.shape, tuple(labels.setdefault(int(v), len(labels)) for v in a.flat)


def self_test():
    """Bounded structural, ablation, count, and metamorphic checks over 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    fingerprints = {tuple((_key(pair["input"]), _key(pair["output"]))
                          for split in ("train", "test") for pair in p[split])
                    for p in puzzles}
    _require(len(fingerprints) == 42, "different seeds repeated a puzzle")
    structures, shapes, profiles, counts, widths, backgrounds, collisions = (set() for _ in range(7))
    broad, joint, middle_gaps, pin_hidden, long_thin = 0, 0, 0, 0, 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local_profiles = set()
        for pair in puzzle["train"] + puzzle["test"]:
            a, b = pair["input"], pair["output"]
            background, boxes = _measure(a)
            ordered = _rank(boxes)
            h = tuple(boxes[c][2] - boxes[c][0] + 1 for c in ordered)
            w = [box[3] - box[1] + 1 for box in boxes.values()]
            structures.add(_canonical(a))
            shapes.add(a.shape)
            profiles.add(h)
            local_profiles.add(h)
            counts.add(len(boxes))
            widths.update(w)
            backgrounds.add(background)
            r, c = _axes(boxes)
            collisions.add((len(set(r)), len(set(c))))
            broad += max(_overlaps(boxes)) >= 9
            joint += any(not np.any(a[row] == color)
                         and len(np.unique(a[row, box[1]:box[3] + 1])) >= 2
                         for color, box in boxes.items() for row in range(box[0] + 1, box[2]))
            middle_gaps += any(not np.any(a[row] == color)
                               for color in ordered[1:-1]
                               for row in range(boxes[color][0] + 1, boxes[color][2]))
            pin_hidden += _hidden_pin_center(a, boxes)
            long_thin += any((box[2] - box[0] >= 6 and box[3] == box[1])
                             or (box[3] - box[1] >= 8 and box[2] == box[0])
                             for box in boxes.values())
            # Palette permutations preserve every semantic role, including bg.
            permutation = np.array([3, 7, 0, 9, 1, 8, 2, 6, 5, 4])
            _require(np.array_equal(solve(permutation[a]), permutation[b]), "color equivariance")
            # Axis reflection changes locations, not which ranks control them.
            _require(np.array_equal(solve(a[::-1, ::-1]), b[::-1, ::-1]), "reflection equivariance")
        _require(len(local_profiles) >= 3, "little within-puzzle height variation")
    _require(len(structures) >= 180 and len(shapes) >= 60, "weak scene geometry variation")
    _require(len(profiles) >= 45 and counts == {3, 4, 5, 6}, "weak count/profile variation")
    _require(len(backgrounds) >= 4 and widths == {1, 3, 5, 7, 9, 11, 13}, "weak palette/proportions")
    _require(collisions == {(1, 1), (1, 2), (2, 1), (2, 2)}, "missing axis collision patterns")
    _require(min(broad, joint, middle_gaps, long_thin) > 0 and pin_hidden >= 84, "missing occlusion modes")
    # Every legal explicit count combination is exercised, at one bounded seed.
    for n in range(2, 11):
        for m in range(1, min(n - 1, 4) + 1):
            p = generate_puzzle(314159, n, m)
            _require((len(p["train"]), len(p["test"])) == (n, m), "explicit counts ignored")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    again = generate_puzzle(0)
    for split in ("train", "test"):
        for a, b in zip(puzzles[0][split], again[split]):
            _require(all(np.array_equal(a[k], b[k]) for k in ("input", "output")), "nondeterministic")

    # Compatible family members beyond the sampler's small box/count ranges.
    # Nine objects; a 25-row pin and a 25-column pin share only their center row.
    a = np.full((28, 28), 9, dtype=np.int8)
    for i, height in enumerate([3, 5, 7, 9, 11, 13, 15, 25]):
        a[13 - height // 2:14 + height // 2, 3 + 2 * i] = i
    a[13, 1:26] = 8
    background, boxes = _measure(a)
    _require(len(boxes) == 9 and _layerable(a, background, boxes), "wide-family fixture")
    ri = [r for r in range(28) for _ in range(2 if r == 13 else 1)]
    ci = [c for c in range(28) for _ in range(2 if c in (13, 17) else 1)]
    _require(np.array_equal(solve(a), a[np.ix_(ri, ci)]), "nine-object family member")
    # A uniform perimeter, rather than majority frequency, defines background.
    a = np.full((28, 28), 4, dtype=np.int8)
    a[1:26, 1:26] = 7
    a[7:10, 7:10] = 0
    a[13, 1:26] = 2
    _require(np.count_nonzero(a == 7) > np.count_nonzero(a == 4), "nonmajority fixture")
    ci = ri
    _require(np.array_equal(solve(a), a[np.ix_(ri, ci)]), "nonmajority background")
