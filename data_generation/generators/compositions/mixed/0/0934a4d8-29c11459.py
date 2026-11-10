"""Reflect hidden endpoints, partition their full row intervals, then crop.

Only standard library and NumPy are used. No parent module or artifact is loaded.
The input visibly encodes all color roles; solve never receives a puzzle or seed.
Blank rows contain only the local background. Singleton rows name the separator;
two-cell rows contain distinct endpoints of an odd-length horizontal interval.
Reflection is across a horizontal, possibly offset, axis. This is a deliberate
specialization of parent A, not a claim to support its entire reflection family.
"""

from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape),
             "Grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "Grid cells must be integers")
    _require(np.all((a >= 0) & (a <= 9)), "Grid colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _mask(a):
    """The mask color occurs only in one solid rectangle of area at least 4."""
    candidates = []
    for color in np.unique(a):
        rr, cc = np.nonzero(a == color)
        r, s = int(rr.min()), int(rr.max()) + 1
        c, d = int(cc.min()), int(cc.max()) + 1
        if len(rr) >= 4 and len(rr) == (s - r) * (d - c):
            candidates.append((int(color), (r, s, c, d)))
    _require(len(candidates) == 1, "Mask must be the unique rectangular color of area >=4")
    return candidates[0]


def _rows(full, background):
    """Read the completed endpoint field, including the singleton color key."""
    lines, markers, blanks = [], [], []
    for r, row in enumerate(full):
        positions = np.flatnonzero(row != background)
        if len(positions) == 0:
            blanks.append(r)
        elif len(positions) == 1:
            c = int(positions[0])
            markers.append((r, c, int(row[c])))
        elif len(positions) == 2:
            left, right = (int(c) for c in positions)
            x, y = int(row[left]), int(row[right])
            _require(x != y and right - left >= 2 and (right - left) % 2 == 0,
                     "An endpoint interval needs distinct colors and odd inclusive length")
            lines.append((r, left, right, x, y))
        else:
            raise ValueError("Completed rows may contain zero, one or two non-background cells")
    _require(lines and markers and blanks, "Need endpoint rows, a singleton color key and a blank row")
    colors = {x[2] for x in markers}
    _require(len(colors) == 1, "All singleton key rows must use one color")
    separator = next(iter(colors))
    _require(all(separator not in (line[3], line[4]) for line in lines),
             "The key color cannot be an endpoint color")
    return separator, lines, markers, blanks


def _infer(input_grid):
    """Infer the row-reflection axis from visible pixels and recover the field.

    Coordinate sum t means r maps to t-r, restricted to the canvas overlap.
    Two distinct fully visible endpoint-row signatures must witness the axis.
    Every hidden cell needs an unoccluded reflected mate. Agreeing axes are fine.
    """
    a = _grid(input_grid)
    mask, box = _mask(a)
    r, s, c, d = box
    visible = a != mask
    colors, counts = np.unique(a[visible], return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    _require(int(counts.max()) * 2 > int(visible.sum()), "Visible background must be a strict majority")
    h, w = a.shape
    hidden_rows = np.arange(r, s)
    candidates = []
    for total in range(1, 2 * h - 2):
        mates = total - hidden_rows
        if np.any((mates < 0) | (mates >= h)) or np.any((mates >= r) & (mates < s)):
            continue
        row_ids = np.arange(h)
        paired = (total - row_ids >= 0) & (total - row_ids < h) & (row_ids < total - row_ids)
        first = row_ids[paired]
        second = total - first
        if len(first) * 2 < h // 2:
            continue
        aa, bb = a[first], a[second]
        known = (aa != mask) & (bb != mask)
        if np.any(aa[known] != bb[known]):
            continue
        witnesses = set()
        for x, y in zip(first, second):
            if not (r <= x < s or r <= y < s):
                positions = np.flatnonzero(a[x] != background)
                if len(positions) == 2:
                    witnesses.add(a[x].tobytes())
        if len(witnesses) < 2:
            continue
        full = a.copy()
        full[r:s, c:d] = a[mates, c:d]
        try:
            separator, lines, markers, blanks = _rows(full, background)
        except ValueError:
            continue
        candidates.append((full, total, separator, lines, markers, blanks))
    _require(candidates, "No witnessed reflection can recover every hidden cell")
    full, _, separator, lines, markers, blanks = candidates[0]
    _require(all(np.array_equal(full, candidate[0]) for candidate in candidates),
             "Supported reflection completions disagree")
    return {"input": a, "full": full, "box": box, "mask": mask,
            "background": background, "separator": separator, "lines": lines,
            "markers": markers, "blanks": blanks,
            "axes": tuple(candidate[1] for candidate in candidates)}


def _fill(full, lines, separator):
    out = full.copy()
    for row, left, right, x, y in lines:
        middle = (left + right) // 2
        out[row, left:middle] = x
        out[row, middle] = separator
        out[row, middle + 1:right + 1] = y
    return out


def _crop(a, box):
    r, s, c, d = box
    return a[r:s, c:d].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover first, fill FULL intervals second, and return just the query crop."""
    info = _infer(input_grid)
    return _crop(_fill(info["full"], info["lines"], info["separator"]), info["box"])


def _without_recovery(info):
    """A diagnostic ablation: erase the mask, then use only visible endpoints."""
    a = info["input"].copy()
    a[a == info["mask"]] = info["background"]
    lines = []
    for r, row in enumerate(a):
        pos = np.flatnonzero((row != info["background"]) & (row != info["separator"]))
        if len(pos) == 2:
            left, right = (int(v) for v in pos)
            if (right - left) % 2 == 0:
                lines.append((r, left, right, int(row[left]), int(row[right])))
    return _crop(_fill(a, lines, info["separator"]), info["box"])


def _midpoints_only(info):
    """A diagnostic shortcut that inserts separators without growing territories."""
    a = info["full"].copy()
    for row, left, right, _, _ in info["lines"]:
        a[row, (left + right) // 2] = info["separator"]
    return _crop(a, info["box"])


def _features(info):
    r, s, c, d = info["box"]
    affected = [line for line in info["lines"] if r <= line[0] < s
                and line[1] < d and line[2] >= c]
    endpoints = sum(c <= x < d for line in affected for x in line[1:3])
    midpoints = sum(c <= (line[1] + line[2]) // 2 < d for line in affected)
    partial = sum(not (c <= line[1] and line[2] < d) for line in affected)
    clipped_growth = any(not (c <= line[1] and line[2] < d)
                         and max(c, line[1] + 1) < min(d, line[2])
                         for line in affected)
    outside_background = any(c < line[1] or d - 1 > line[2] for line in affected)
    blank = any(r <= x < s for x in info["blanks"])
    marker = any(r <= x < s and c <= y < d for x, y, _ in info["markers"])
    return {"affected": len(affected), "endpoints": int(endpoints),
            "midpoints": int(midpoints), "partial": int(partial),
            "clipped_growth": clipped_growth,
            "outside_background": outside_background, "blank": blank, "marker": marker,
            "halves": tuple((line[2] - line[1]) // 2 for line in affected)}


def _sample_field(rng, separator, maximum_half, forced_half):
    n = int(rng.integers(8, 16))
    w = int(rng.integers(max(9, 2 * forced_half + 3), 31))
    background, mask = (int(x) for x in rng.choice(
        [x for x in range(10) if x != separator], size=2, replace=False))
    legal = [x for x in range(10) if x not in (separator, background, mask)]
    palette = rng.choice(legal, size=int(rng.integers(2, 8)), replace=False)
    count = int(rng.integers(4, min(n - 2, 8) + 1))
    rows = sorted(int(x) for x in rng.choice(n, count, replace=False))
    key_row = int(rng.choice([x for x in range(n) if x not in rows]))
    base = np.full((n, w), background, dtype=np.int8)
    base[key_row, int(rng.integers(w))] = separator
    anchor = int(rng.choice(rows))
    for row in rows:
        maximum = min(maximum_half, (w - 1) // 2)
        half = forced_half if row == anchor else int(rng.integers(1, maximum + 1))
        # Endpoint-color assignments may repeat; only the two ends must differ.
        x, y = (int(v) for v in rng.choice(palette, size=2, replace=False))
        left = int(rng.integers(0, w - 2 * half))
        base[row, left], base[row, left + 2 * half] = x, y
    through_cell = bool(rng.integers(2))
    full = np.concatenate((base, base[-2::-1] if through_cell else base[::-1]))
    total = len(full) - 1
    # Crop the surrounding reflected field, not the query. This offsets its seam.
    top, bottom = (int(x) for x in rng.integers(0, 3, size=2))
    if rng.random() < .3:
        top = bottom = 0
    full = full[top:len(full) - bottom if bottom else len(full)].copy()
    return full, mask, background, total - 2 * top


def _make_pair(rng, separator, mode, forced_half, seen):
    """Bounded rejection uses returned pixels, never hidden solver metadata."""
    is_test = mode == "test"
    for _ in range(100):
        full, mask, background, axis = _sample_field(
            rng, separator, 14 if is_test else 4, forced_half)
        try:
            _, all_lines, all_markers, all_blanks = _rows(full, background)
        except ValueError:
            # Cropping may remove both copies of a peripheral key/blank row.
            continue
        h, w = full.shape
        # Choose either side independently; each query has an intact mirror.
        side = int(rng.integers(2))
        available = [line for line in all_lines if
                     (2 * line[0] < axis if side == 0 else 2 * line[0] > axis)
                     and 0 <= axis - line[0] < h]
        if len(available) < 3:
            continue
        for _ in range(72):
            k = int(rng.integers(2 if is_test else 1,
                                 min(4 if is_test else 3, len(available) - 2) + 1)) if len(available) >= 4 else 1
            if is_test and k < 2:
                continue
            start = int(rng.integers(0, len(available) - k + 1))
            selected = available[start:start + k]
            # A forced-scale row must actually affect the returned rectangle.
            if not any((line[2] - line[1]) // 2 == forced_half for line in selected):
                continue
            previous = available[start - 1][0] + 1 if start else max(0, axis - h + 1)
            following = available[start + k][0] if start + k < len(available) else h
            if side == 0:
                following = min(following, (axis + 1) // 2)
            else:
                previous = max(previous, axis // 2 + 1)
            if previous > selected[0][0] or following <= selected[-1][0]:
                continue
            r = int(rng.integers(previous, selected[0][0] + 1))
            s = int(rng.integers(selected[-1][0] + 1, following + 1))
            if mode == "full":
                c = int(rng.integers(0, min(line[1] for line in selected) + 1))
                d = int(rng.integers(max(line[2] for line in selected) + 1, w + 1))
            else:
                c = int(rng.integers(w))
                d = int(rng.integers(c + 1, w + 1))
            if (s - r) * (d - c) < 4 or (s - r) * (d - c) > full.size * .38:
                continue
            box = r, s, c, d
            # Cheap construction filters precede the input-only inference. The
            # inferred field must still exactly equal this field before return.
            features = _features({"box": box, "lines": all_lines,
                                  "markers": all_markers, "blanks": all_blanks})
            if not features["endpoints"] or not features["midpoints"]:
                continue
            if forced_half not in features["halves"]:
                continue
            if mode == "full" and not (features["outside_background"] and
                                       features["blank"] and features["marker"]):
                continue
            if mode == "partial" and not features["clipped_growth"]:
                continue
            if is_test and (features["affected"] < 2 or not features["clipped_growth"]):
                continue
            a = full.copy()
            a[r:s, c:d] = mask
            if _key(a) in seen:
                continue
            try:
                info = _infer(a)
            except ValueError:
                continue
            if not np.array_equal(info["full"], full):
                continue
            out = _crop(_fill(full, info["lines"], separator), box)
            if np.array_equal(out, _crop(full, box)) or np.array_equal(out, _without_recovery(info)):
                continue
            # A known interior (non-midpoint) background cell must be propagated,
            # so separator insertion by itself is also ruled out.
            interior = False
            for row, left, right, x, y in info["lines"]:
                if r <= row < s:
                    middle = (left + right) // 2
                    interior |= any(c <= col < d and col != middle
                                    for col in range(left + 1, right))
            if not interior:
                continue
            return {"input": a, "output": out}
    raise RuntimeError("Could not construct a witnessed distinct query after bounded retries")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " is outside the legal range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed if seed >= 0 else seed % (1 << 64))
    separator = int(rng.integers(10))
    halves = [int(x) for x in rng.choice([2, 3, 4], size=2, replace=False)]
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(num_train + num_test):
        training = i < num_train
        mode = "full" if i == 0 else "partial" if i == 1 else "mixed" if training else "test"
        forced_half = halves[i] if i < 2 else int(rng.integers(1, 5)) if training else int(rng.integers(5, 11))
        pair = _make_pair(rng, separator, mode, forced_half, seen)
        puzzle["train" if training else "test"].append(pair)
        seen.add(_key(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check semantics AND the documented demonstration/test sampling contract.

    solve has wider support than the sampling contract (e.g. an all-short-span
    query). No construction parameters or stored labels determine its answer.
    """
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Wrong puzzle schema")
    _require(all(isinstance(puzzle[k], list) for k in puzzle), "Splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "Illegal pair counts")
    seen, separators = set(), set()
    train_features, test_features = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Wrong pair schema")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(a) not in seen, "Duplicate input")
            seen.add(_key(a))
            info = _infer(a)
            r, s, c, d = info["box"]
            _require(b.shape == (s - r, d - c), "Output must have the mask's exact shape")
            _require(not np.any(b == info["mask"]), "Mask color cannot survive")
            # Per-cell equation independently checks crop clipping and non-active rows.
            by_row = {line[0]: line for line in info["lines"]}
            for rr in range(r, s):
                for cc in range(c, d):
                    expected = int(info["full"][rr, cc])
                    if rr in by_row:
                        _, left, right, x, y = by_row[rr]
                        if left <= cc <= right:
                            middle = (left + right) // 2
                            expected = x if cc < middle else y if cc > middle else info["separator"]
                    _require(int(b[rr - r, cc - c]) == expected,
                             "Incorrect territory, separator, exterior or non-active row")
            _require(np.array_equal(solve(a), b), "Input-only solver disagrees")
            _require(not np.array_equal(b, _crop(info["full"], info["box"])), "Omitting B leaves the answer unchanged")
            _require(not np.array_equal(b, _without_recovery(info)), "Omitting A leaves the answer unchanged")
            _require(not np.array_equal(b, _midpoints_only(info)),
                     "Midpoint insertion alone explains the answer without territory growth")
            f = _features(info)
            _require(f["endpoints"] > 0 and f["midpoints"] > 0, "Need hidden endpoints and hidden midpoints")
            separators.add(info["separator"])
            (train_features if split == "train" else test_features).append(f)
    _require(len(separators) == 1, "Puzzle-wide separator must be consistent")
    _require(any(f["outside_background"] and f["blank"] and f["marker"] for f in train_features),
             "Training must demonstrate preserved exterior, blank rows and singleton keys")
    _require(any(f["clipped_growth"] for f in train_features), "Training must demonstrate clipping after filling")
    _require(len({half for f in train_features for half in f["halves"]}) >= 2,
             "Training must show varied interval lengths")
    _require(all(max(f["halves"]) <= 4 for f in train_features), "Training scale exceeds its sampling regime")
    _require(all(max(f["halves"]) >= 5 and f["affected"] >= 2 and f["clipped_growth"] for f in test_features),
             "Tests must combine longer intervals, multiple rows and clipping")


def _signature(puzzle):
    return tuple(_key(pair[role]) for split in ("train", "test")
                 for pair in puzzle[split] for role in ("input", "output"))


def self_test():
    """Bounded, streaming checks on 42 seeds; no artifacts or execution traces."""
    # Hand-calculated fixture: clipped intervals, a blank and a hidden color key.
    base = np.zeros((7, 11), dtype=np.int8)
    base[0, 1], base[0, 9] = 2, 3
    base[2, 0], base[2, 6] = 4, 6
    base[3, 8] = 5
    base[4, 2], base[4, 6] = 7, 9
    base[6, 3], base[6, 9] = 2, 6
    fixture = np.concatenate((base, base[::-1]))
    fixture[8:12, 3:10] = 8
    expected = np.array([[int(c) for c in row] for row in
                         ("0000000", "7599000", "0000050", "5666000")], dtype=np.int8)
    _require(np.array_equal(solve(fixture), expected), "Hand-calculated clipped answer failed")
    before = fixture.copy()
    solve(fixture)
    _require(np.array_equal(fixture, before), "solve mutated its input")
    _require(np.array_equal(solve(fixture[:, ::-1]), expected[:, ::-1]),
             "Left/right reversal changed the rule")
    _require(np.array_equal(solve(fixture[::-1]), expected[::-1]),
             "Top/bottom reversal changed the rule")
    permutation = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int8)
    _require(np.array_equal(solve(permutation[fixture]), permutation[expected]),
             "Semantic colors are not permutation equivariant")
    changed = fixture.copy()
    changed[4, 6] = 1
    _require(np.array_equal(solve(changed)[1], np.array([7, 5, 1, 1, 0, 0, 0])),
             "A recovered endpoint must causally change its territory")
    changed = fixture.copy()
    changed[3, 8] = 1
    _require(np.array_equal(solve(changed), np.where(expected == 5, 1, expected)),
             "The recovered singleton key must designate the separator")
    # The input-only solver also handles a three-cell interval even though the
    # sampler requires some longer interior propagation in every returned pair.
    short = fixture.copy()
    short[4, 6], short[4, 4] = 0, 9
    short_expected = expected.copy()
    short_expected[1] = np.array([5, 9, 0, 0, 0, 0, 0], dtype=np.int8)
    _require(np.array_equal(solve(short), short_expected), "Three-cell interval failed")
    malformed = []
    even = fixture.copy()
    even[4, 6], even[4, 5] = 0, 9
    malformed.append(even)
    equal = fixture.copy()
    equal[4, 6] = 7
    malformed.append(equal)
    no_key = fixture.copy()
    no_key[3, 8] = 0
    malformed.append(no_key)
    for invalid in malformed:
        try:
            solve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Even interval, equal endpoints or missing separator key accepted")
    for invalid in (fixture.copy(), np.concatenate((base, base[::-1]))):
        if np.any(invalid == 8):
            invalid[0, 1] = 4  # Contradicts a visible reflected endpoint.
        else:
            invalid[6:8, 3:10] = 8  # Erases both members of several orbits.
        try:
            solve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Conflicting evidence or a completely erased orbit accepted")
    shapes, outputs, layouts, signatures = set(), set(), set(), set()
    masks, backgrounds, separators, endpoints = set(), set(), set(), set()
    stats = {"offset": 0, "through_cell": 0, "blank_preserved": 0,
             "marker_preserved": 0, "thin": 0, "crop_first_differs": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = _signature(puzzle)
        signatures.add(hashlib.sha256(repr(signature).encode()).hexdigest())
        if seed in (0, 17, 41):
            _require(signature == _signature(generate_puzzle(seed)), "Nondeterministic puzzle")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                info = _infer(a)
                f = _features(info)
                shapes.add(a.shape)
                outputs.add(b.shape)
                layouts.add((a.shape, info["box"], tuple(line[:3] for line in info["lines"])))
                masks.add(info["mask"])
                backgrounds.add(info["background"])
                separators.add(info["separator"])
                endpoints.update(x for line in info["lines"] for x in line[3:5])
                stats["offset"] += any(axis != len(a) - 1 for axis in info["axes"])
                stats["through_cell"] += any(axis % 2 == 0 for axis in info["axes"])
                stats["blank_preserved"] += f["blank"]
                stats["marker_preserved"] += f["marker"]
                stats["thin"] += min(b.shape) == 1
                # Fill only endpoint pairs completely contained in the crop.
                r, s, c, d = info["box"]
                small = _crop(info["full"], info["box"])
                local = [(row - r, left - c, right - c, x, y)
                         for row, left, right, x, y in info["lines"]
                         if r <= row < s and c <= left and right < d]
                stats["crop_first_differs"] += not np.array_equal(
                    b, _fill(small, local, info["separator"]))
                _require(np.array_equal(solve(a.tolist()), b), "Nested-list solving failed")
    _require(len(signatures) == 42, "Seed diversity failed")
    _require(len(shapes) >= 40 and len(outputs) >= 40 and len(layouts) >= 180,
             "Geometry diversity failed")
    _require(all(len(colors) == 10 for colors in (masks, backgrounds, separators, endpoints)),
             "Color roles must cover all ten ARC colors")
    _require(all(value > 0 for value in stats.values()), "A required structural feature is absent")
    # Every legal explicit count, including the minimal two-demonstration case.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(900 + nt * 10 + nq, nt, nq)
            _require(len(puzzle["train"]) == nt and len(puzzle["test"]) == nq, "Counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    puzzle = generate_puzzle(7, 2, 1)
    puzzle["test"][0]["output"][0, 0] = (int(puzzle["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(puzzle)
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupt output accepted")
    # A whole-canvas overwrite has no reflection witness.
    try:
        solve(np.full((12, 14), 8, dtype=np.int8))
    except ValueError:
        pass
    else:
        raise AssertionError("Unwitnessed reconstruction accepted")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes),
            "output_shapes": len(outputs), "layouts": len(layouts), "features": stats}
