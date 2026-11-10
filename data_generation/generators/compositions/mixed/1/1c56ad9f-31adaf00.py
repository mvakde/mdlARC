"""Square-gated, locally bottom-anchored row waves on registered layers.

Only NumPy and the standard library are used. No parent generator is imported.
Generation and self-tests retain no puzzles or execution traces on disk.
"""

import hashlib
import numpy as np


NEUTRAL = 0
HIGHLIGHT = 1
DIVIDER = 9
OFFSETS = (0, -1, 0, 1)


def _check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a


def _unpack(value):
    """Map precedes scaffold: left/right, or top/bottom; divider is all 9."""
    a = _grid(value)
    h, w = a.shape
    vertical = (w % 2 == 1 and np.all(a[:, w // 2] == DIVIDER))
    horizontal = (h % 2 == 1 and np.all(a[h // 2, :] == DIVIDER))
    if vertical == horizontal:
        raise ValueError("expected exactly one central divider")
    if vertical:
        geometry, content = a[:, :w // 2], a[:, w // 2 + 1:]
        content_slice = (slice(None), slice(w // 2 + 1, None))
        orientation = "horizontal"
    else:
        geometry, content = a[:h // 2, :], a[h // 2 + 1:, :]
        content_slice = (slice(h // 2 + 1, None), slice(None))
        orientation = "vertical"
    if not geometry.size or not content.size:
        raise ValueError("empty layer")
    blockers = set(map(int, np.unique(geometry))) - {NEUTRAL}
    foregrounds = set(map(int, np.unique(content))) - {NEUTRAL}
    if (len(blockers) != 1 or len(foregrounds) != 1
            or not blockers.isdisjoint({0, 1, 9})
            or not foregrounds.isdisjoint({0, 1, 9})
            or blockers == foregrounds):
        raise ValueError("map/scaffold color conventions violated")
    return a, geometry, content, foregrounds.pop(), content_slice, orientation


def _pack(geometry, content, orientation):
    axis = 1 if orientation == "horizontal" else 0
    shape = (geometry.shape[0], 1) if axis else (1, geometry.shape[1])
    return np.concatenate((geometry, np.full(shape, DIVIDER, dtype=np.int8), content), axis=axis)


def _rectangles(mask):
    """Enumerate locally maximal solid rectangles, with inclusive bounds."""
    h, w = mask.shape
    rectangles = []
    for top in range(h):
        common = np.ones(w, dtype=bool)
        for bottom in range(top, h):
            common &= mask[bottom]
            changes = np.flatnonzero(np.diff(np.r_[False, common, False]))
            for left, stop in zip(changes[::2], changes[1::2]):
                left, stop = int(left), int(stop)
                if top and np.all(mask[top - 1, left:stop]):
                    continue
                if bottom + 1 < h and np.all(mask[bottom + 1, left:stop]):
                    continue
                rectangles.append((top, left, bottom, stop - 1))
    return rectangles


def _squares(mask):
    """Vectorized window sums followed by four complete-side extension tests."""
    h, w = mask.shape
    integral = np.pad(mask.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    result = []
    for s in range(2, min(h, w) + 1):
        sums = integral[s:, s:] - integral[:-s, s:] - integral[s:, :-s] + integral[:-s, :-s]
        for t, l in np.argwhere(sums == s * s):
            t, l = int(t), int(l)
            b, r = t + s - 1, l + s - 1
            if ((t and mask[t - 1, l:r + 1].all())
                    or (b + 1 < h and mask[b + 1, l:r + 1].all())
                    or (l and mask[t:b + 1, l - 1].all())
                    or (r + 1 < w and mask[t:b + 1, r + 1].all())):
                continue
            result.append((t, l, b, r))
    return sorted(result)


def _overlap(a, b):
    return max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3])


def _disjoint(rectangles):
    return not any(_overlap(a, b) for i, a in enumerate(rectangles) for b in rectangles[i + 1:])


def _row_result(row, offset, strict_row, fg, highlight_border=False):
    result = np.zeros_like(row)
    cols = np.flatnonzero(row == fg) + offset
    cols = cols[(cols >= 0) & (cols < len(row))]
    result[cols] = fg
    if offset and strict_row:
        inside = cols if highlight_border else cols[(cols > 0) & (cols < len(row) - 1)]
        result[inside] = HIGHLIGHT
    return result


def _transform(content, squares, fg, anchor="square", highlight_border=False, table=OFFSETS):
    """Simultaneous reads from the input; selected squares never overlap."""
    out = content.copy()
    for t, l, b, r in squares:
        for row in range(t, b + 1):
            distance = (b - row if anchor == "square" else
                        row - t if anchor == "top" else content.shape[0] - 1 - row)
            delta = table[distance % 4]
            out[row, l:r + 1] = _row_result(content[row, l:r + 1], delta,
                                           t < row < b, fg, highlight_border)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover both layers, select literal solid neutral squares, apply local waves."""
    a, geometry, content, fg, target, _ = _unpack(input_grid)
    squares = _squares(geometry == NEUTRAL)
    if not _disjoint(squares):
        raise ValueError("overlapping locally maximal squares are outside this family")
    out = a.copy()
    out[target] = _transform(content, squares, fg)
    return out


def _stamp(mask, reserved, rect):
    t, l, b, r = rect
    mask[t:b + 1, l:r + 1] = True
    reserved[max(0, t - 1):min(mask.shape[0], b + 2),
             max(0, l - 1):min(mask.shape[1], r + 2)] = True


def _place(mask, reserved, rng, height, width):
    h, w = mask.shape
    if height > h or width > w:
        return None
    for _ in range(40):
        t, l = int(rng.integers(h - height + 1)), int(rng.integers(w - width + 1))
        if not reserved[t:t + height, l:l + width].any():
            rect = (t, l, t + height - 1, l + width - 1)
            _stamp(mask, reserved, rect)
            return rect
    return None


def _geometry(rng, primary, orientation, extra_count, touching):
    short = int(rng.integers(max(10, primary + 2), 15))
    long = int(rng.integers(max(17, primary + 7), 31))
    h, w = (long, short) if orientation == "horizontal" else (short, long)
    mask = np.zeros((h, w), dtype=bool)
    reserved = np.zeros_like(mask)
    if touching:
        s = int(rng.integers(2, 4))
        # Two squares touch along one cell of a side, never a complete side.
        gh, gw = primary + s - 1, primary + s
        transpose = bool(rng.integers(2))
        if transpose:
            gh, gw = gw, gh
        if gh > h or gw > w:
            return None
        top, left = int(rng.integers(h - gh + 1)), int(rng.integers(w - gw + 1))
        if transpose:
            other = (top + primary, left + primary - 1,
                     top + primary + s - 1, left + primary + s - 2)
        else:
            other = (top + primary - 1, left + primary,
                     top + primary + s - 2, left + primary + s - 1)
        main = (top, left, top + primary - 1, left + primary - 1)
        _stamp(mask, reserved, main)
        _stamp(mask, reserved, other)
        intended = [main, other]
    else:
        main = _place(mask, reserved, rng, primary, primary)
        if main is None:
            return None
        intended = [main]
    for index in range(extra_count):
        s = 2 if index == 0 else int(rng.integers(2, min(primary, 5)))
        rect = _place(mask, reserved, rng, s, s)
        if rect is not None:
            intended.append(rect)
    if len(intended) < 2:
        return None
    dh, dw = 3, int(rng.integers(4, 8))
    if rng.random() < 0.5:
        dh, dw = dw, dh
    distractor = _place(mask, reserved, rng, dh, dw)
    if distractor is None:
        return None
    # Partial-side attachments retain full-side maximality. No separate islands
    # are assumed by the detector; noise can join neutral components.
    for t, l, b, r in intended:
        if rng.random() < 0.7:
            direction = int(rng.integers(4))
            if direction < 2:
                rr = t - 1 if direction == 0 else b + 1
                cc = int(rng.integers(l, r + 1))
            else:
                rr = int(rng.integers(t, b + 1))
                cc = l - 1 if direction == 2 else r + 1
            if 0 <= rr < h and 0 <= cc < w:
                mask[rr, cc] = True
    density = float(rng.uniform(0.18, 0.62))
    mask |= (rng.random((h, w)) < density) & ~reserved
    squares = _squares(mask)
    if not all(rect in squares for rect in intended) or not _disjoint(squares):
        return None
    # The explicit non-square must stay nonextendable, with subwindows unselected.
    if distractor not in _rectangles(mask):
        return None
    if max(b - t + 1 for t, l, b, r in squares) != primary:
        return None
    return mask, squares


def _scaffold(rng, shape, squares, fg):
    h, w = shape
    rails = rng.random(w) < float(rng.uniform(0.12, 0.36))
    rails[0] = rails[-1] = True
    for t, l, b, r in squares:
        rails[l + 1 if r - l >= 2 else l] = True
    # A single connected rectilinear scaffold: every rail joins the end bars.
    content = np.tile(np.where(rails, fg, NEUTRAL), (h, 1)).astype(np.int8)
    content[0] = content[-1] = fg
    bar_rows = [row for row in range(1, h - 1) if rng.random() < 0.2]
    for row in bar_rows[:max(0, h - 5)]:
        positions = np.flatnonzero(rails)
        i, j = sorted(map(int, rng.choice(len(positions), size=2, replace=False)))
        content[row, positions[i]:positions[j] + 1] = fg
    return content


def _phase_evidence(content, result, squares, fg):
    possible = [set((-1, 0, 1)) for _ in range(4)]
    for t, l, b, r in squares:
        for row in range(t, b + 1):
            choices = {d for d in (-1, 0, 1) if np.array_equal(
                _row_result(content[row, l:r + 1], d, t < row < b, fg),
                result[row, l:r + 1])}
            possible[(b - row) % 4] &= choices
    return possible


def _all_windows_greedy(mask):
    """A specific shortcut: take big square subwindows even in long rectangles."""
    candidates = []
    for side in range(min(mask.shape), 1, -1):
        for t in range(mask.shape[0] - side + 1):
            for l in range(mask.shape[1] - side + 1):
                rect = (t, l, t + side - 1, l + side - 1)
                if mask[t:t + side, l:l + side].all() and not any(
                        _overlap(rect, a) for a in candidates):
                    candidates.append(rect)
    return candidates


def _make_pair(rng, primary, orientation, blocker, fg, teaching=False):
    for _ in range(160):
        made = _geometry(rng, primary, orientation, int(rng.integers(1, 4)),
                         bool(rng.random() < 0.4))
        if made is None:
            continue
        mask, squares = made
        content = _scaffold(rng, mask.shape, squares, fg)
        result = _transform(content, squares, fg)
        if not np.any(result == HIGHLIGHT):
            continue
        if teaching:
            if not any(b - t + 1 == 2 for t, l, b, r in squares):
                continue
            if any(p != {d} for p, d in zip(_phase_evidence(content, result, squares, fg), OFFSETS)):
                continue
            alternatives = [
                _transform(content, squares, fg, anchor="canvas"),
                _transform(content, squares, fg, anchor="top"),
                _transform(content, squares, fg, highlight_border=True),
                _transform(content, [max(squares, key=lambda x: x[2] - x[0])], fg),
                _transform(content, _all_windows_greedy(mask), fg),
                _transform(content, [(0, 0, mask.shape[0] - 1, mask.shape[1] - 1)], fg),
            ]
            if any(np.array_equal(result, alternative) for alternative in alternatives):
                continue
        geometry = np.where(mask, NEUTRAL, blocker).astype(np.int8)
        inp = _pack(geometry, content, orientation)
        return {"input": inp, "output": _pack(geometry, result, orientation)}
    raise RuntimeError("bounded pair construction exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return (a.shape, a.tobytes())


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ntrain = _count(num_train, "num_train", 2, 10)
    ntest = _count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    blocker, fg = map(int, rng.choice(np.arange(2, 9), size=2, replace=False))
    orientation = "horizontal" if rng.integers(2) else "vertical"
    maximum = int(rng.integers(6, 8))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            if split == "train":
                primary = maximum - index if index < 2 else int(rng.integers(3, maximum + 1))
            else:
                primary = int(rng.integers(maximum + 1, min(10, maximum + 3) + 1))
            for _ in range(12):
                pair = _make_pair(rng, primary, orientation, blocker, fg,
                                  teaching=(split == "train" and index < 2))
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded unique-input construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _reference(content, squares, fg):
    """Cell-by-cell transport check, separately from the row-array transform."""
    result = content.copy()
    for t, l, b, r in squares:
        result[t:b + 1, l:r + 1] = NEUTRAL
        for row in range(t, b + 1):
            d = b - row
            shift = 0 if d % 2 == 0 else (-1 if d % 4 == 1 else 1)
            for col in range(l, r + 1):
                landing = col + shift
                if content[row, col] == fg and l <= landing <= r:
                    result[row, landing] = (HIGHLIGHT if shift and t < row < b
                                            and l < landing < r else fg)
    return result


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _check(all(isinstance(puzzle[k], list) for k in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    seen, conventions = set(), set()
    learned = [set((-1, 0, 1)) for _ in range(4)]
    demonstrated = {name: False for name in ("canvas", "top", "border", "largest", "subwindows", "no_selection")}
    sizes = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
            a, geometry, content, fg, target, orientation = _unpack(pair["input"])
            out = _grid(pair["output"])
            _check(out.shape == a.shape, "shape changed")
            key = _key(a)
            _check(key not in seen, "duplicate input")
            seen.add(key)
            conventions.add((orientation, fg, tuple(np.unique(geometry))))
            scaffold = content == fg
            _check(scaffold[0].all() and scaffold[-1].all(), "scaffold end bars missing")
            _check(scaffold[:, 0].all() and scaffold[:, -1].all(), "scaffold outside rails missing")
            rail_mask = scaffold.all(axis=0)
            _check(np.count_nonzero(np.all(scaffold == rail_mask, axis=1)) >= 3,
                   "scaffold needs open rail rows")
            mask = geometry == NEUTRAL
            squares = _squares(mask)
            _check(len(squares) >= 2 and _disjoint(squares), "expected nonoverlapping local squares")
            rectangles = _rectangles(mask)
            _check(squares == sorted(r for r in rectangles if r[2] - r[0] == r[3] - r[1] and r[2] > r[0]),
                   "square window and maximal-rectangle detectors disagree")
            _check(any(r[2] - r[0] >= 2 and r[3] - r[1] >= 2 and r[2] - r[0] != r[3] - r[1]
                       for r in rectangles), "missing non-square distractor")
            expected_content = _reference(content, squares, fg)
            expected = a.copy()
            expected[target] = expected_content
            _check(np.array_equal(out, expected), "cell transport, clipping or recoloring failed")
            _check(np.any(expected_content == HIGHLIGHT), "wave has no interior effect")
            _check(np.array_equal(solve(a), out), "solver mismatch")
            union = np.zeros_like(mask)
            for t, l, b, r in squares:
                union[t:b + 1, l:r + 1] = True
                _check(np.any(content[t:b + 1, l:r + 1] == fg), "empty target scaffold")
                sizes[split].append(b - t + 1)
            _check(np.array_equal(expected_content[~union], content[~union]), "outside selection changed")
            _check(np.any(content[~union] == fg), "no unselected foreground evidence")
            if split == "train":
                evidence = _phase_evidence(content, expected_content, squares, fg)
                learned = [x & y for x, y in zip(learned, evidence)]
                variants = {
                    "canvas": _transform(content, squares, fg, anchor="canvas"),
                    "top": _transform(content, squares, fg, anchor="top"),
                    "border": _transform(content, squares, fg, highlight_border=True),
                    "largest": _transform(content, [max(squares, key=lambda x: x[2] - x[0])], fg),
                    "subwindows": _transform(content, _all_windows_greedy(mask), fg),
                    "no_selection": _transform(content, [(0, 0, mask.shape[0] - 1, mask.shape[1] - 1)], fg),
                }
                for name, alternative in variants.items():
                    demonstrated[name] |= not np.array_equal(alternative, expected_content)
    _check(len(conventions) == 1, "conventions differ within puzzle")
    _check(learned == [{0}, {-1}, {0}, {1}], "training does not identify all wave phases")
    _check(all(demonstrated.values()), "training admits a specified shortcut")
    _check(max(sizes["test"]) > max(sizes["train"]), "missing size extrapolation")
    _check(min(sizes["train"]) == 2 and len(set(sizes["train"])) >= 3, "missing small/mixed size evidence")


def self_test():
    """Streaming, bounded 42-seed sweep plus targeted geometry/transport probes."""
    rng = np.random.default_rng(20261023)
    for _ in range(24):
        mask = rng.random((int(rng.integers(3, 8)), int(rng.integers(3, 8)))) < 0.65
        direct = sorted(r for r in _rectangles(mask) if r[2] > r[0] and r[2] - r[0] == r[3] - r[1])
        _check(_squares(mask) == direct, "square detector probe failed")
    # A maximal 2x2 has no strict interior. All-true 3x5 is NOT a square.
    _check(_squares(np.ones((2, 2), dtype=bool)) == [(0, 0, 1, 1)], "2x2 boundary failed")
    _check(not _squares(np.ones((3, 5), dtype=bool)), "non-square subwindows selected")
    row = np.array([4, 0, 4, 4, 0], dtype=np.int8)
    _check(np.array_equal(_row_result(row, -1, True, 4), [0, 1, 1, 0, 0]), "left clipping failed")
    _check(np.array_equal(_row_result(row, 1, True, 4), [0, 1, 0, 1, 4]), "border highlighting failed")
    shapes, geometry_hashes, orientations, palettes, sides, counts, signatures = (set() for _ in range(7))
    touching = attached = nonzero_top = 0
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, geometry, content, fg, target, orientation = _unpack(pair["input"])
                digest.update(a.tobytes())
                mask = geometry == NEUTRAL
                squares = _squares(mask)
                shapes.add(a.shape)
                geometry_hashes.add(hashlib.sha256(mask.tobytes()).digest())
                orientations.add(orientation)
                palettes.add((int(geometry[geometry != 0][0]), fg))
                sides.update(b - t + 1 for t, l, b, r in squares)
                counts.add(len(squares))
                touching += any((a[3] + 1 == b[1] or b[3] + 1 == a[1])
                                and max(a[0], b[0]) <= min(a[2], b[2])
                                or (a[2] + 1 == b[0] or b[2] + 1 == a[0])
                                and max(a[1], b[1]) <= min(a[3], b[3])
                                for i, a in enumerate(squares) for b in squares[i + 1:])
                nonzero_top += any(t > 0 for t, l, b, r in squares)
                for t, l, b, r in squares:
                    attached += ((t > 0 and mask[t - 1, l:r + 1].any())
                                 or (b + 1 < mask.shape[0] and mask[b + 1, l:r + 1].any())
                                 or (l > 0 and mask[t:b + 1, l - 1].any())
                                 or (r + 1 < mask.shape[1] and mask[t:b + 1, r + 1].any()))
                pairs += 1
        signatures.add(digest.digest())
    _check(len(signatures) == 42 and len(geometry_hashes) >= 180, "insufficient structural variation")
    _check(len(shapes) >= 25 and len(palettes) >= 20 and len(orientations) == 2, "layout/palette variation missing")
    _check(set(range(2, 10)) <= sides and len(counts) >= 4, "square size/count variation missing")
    _check(touching >= 15 and attached >= 40 and nonzero_top >= 100, "interaction variation missing")
    a, b = generate_puzzle(1984, 10, 4), generate_puzzle(1984, 10, 4)
    _check(all(_key(x[k]) == _key(y[k]) for split in ("train", "test")
               for x, y in zip(a[split], b[split]) for k in ("input", "output")), "nondeterministic")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generate_puzzle(42 + nt * 5 + ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": pairs, "shapes": len(shapes),
            "geometry_signatures": len(geometry_hashes), "palettes": len(palettes),
            "square_sides": sorted(sides), "square_counts": sorted(counts),
            "touching_pairs": int(touching), "attached_squares": int(attached)}
