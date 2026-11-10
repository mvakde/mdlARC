"""Global bottom-anchored shear, followed by maximal neutral-square marking.

Only NumPy and the standard library are used. No files, examples, or puzzle-level
state are consulted by solve. Neutral is 0, highlight is 1, blockers are 2..9.
Generation works in destination coordinates and takes exact row preimages;
it does not call or combine either parent's puzzle generator.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return grid


def _shear(grid, padding=0):
    h, w = grid.shape
    if h + w - 1 > 30:
        raise ValueError("the widened canvas would exceed 30 columns")
    field = np.full((h, h + w - 1), padding, dtype=grid.dtype)
    for r in range(h):
        start = h - 1 - r
        field[r, start:start + w] = grid[r]
    return field


def _window_sums(prefix, height, width):
    return (prefix[height:, width:] - prefix[:-height, width:]
            - prefix[height:, :-width] + prefix[:-height, :-width])


def _squares(mask):
    """All locally maximal rectangles which are squares of side >= 2.

    A square is maximal exactly when none of its four complete adjacent sides
    can be added. Partial attachments are allowed. This tests squares directly
    rather than storing the irrelevant non-square maximal rectangles.
    Rectangles use exclusive bottom/right coordinates.
    """
    h, w = mask.shape
    prefix = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    result = []
    for s in range(2, min(h, w) + 1):
        good = _window_sums(prefix, s, s) == s * s
        row_runs = _window_sums(prefix, 1, s) == s
        col_runs = _window_sums(prefix, s, 1) == s
        good[1:] &= ~row_runs[:h - s]
        good[:-1] &= ~row_runs[s:]
        good[:, 1:] &= ~col_runs[:, :w - s]
        good[:, :-1] &= ~col_runs[:, s:]
        for r, c in np.argwhere(good):
            result.append((int(r), int(c), int(r) + s, int(c) + s))
    return result


def _union(shape, rectangles):
    mask = np.zeros(shape, dtype=bool)
    for t, l, b, r in rectangles:
        mask[t:b, l:r] = True
    return mask


def _overlap(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _unambiguous(rectangles):
    return not any(_overlap(a, b) for i, a in enumerate(rectangles)
                   for b in rectangles[i + 1:])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Copy each complete row with offset h-1-r, then mark maximal 0 squares."""
    field = _shear(_grid(input_grid))
    # Evaluate every candidate on the same unchanged field, never iteratively.
    targets = _squares(field == 0)
    field[_union(field.shape, targets)] = 1
    return field


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                    (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _interior(h, w):
    allowed = np.zeros((h, h + w - 1), dtype=bool)
    for r in range(h):
        start = h - 1 - r
        allowed[r, start + 1:start + w - 1] = True
    return allowed


def _place(rng, mask, reserved, allowed, rh, rw):
    h, w = mask.shape
    sites = []
    for t in range(h - rh + 1):
        for l in range(w - rw + 1):
            if (allowed[t:t + rh, l:l + rw].all()
                    and not reserved[t:t + rh, l:l + rw].any()):
                sites.append((t, l))
    if not sites:
        return None
    t, l = sites[int(rng.integers(len(sites)))]
    b, r = t + rh, l + rw
    border = (slice(max(0, t - 1), min(h, b + 1)),
              slice(max(0, l - 1), min(w, r + 1)))
    mask[border] = False
    reserved[border] = True
    mask[t:b, l:r] = True
    return t, l, b, r


def _attach(rng, mask, allowed, protected, rect):
    """A thin branch can join a target to its surroundings without a full side."""
    t, l, b, r = rect
    direction = int(rng.integers(4))
    if direction == 0:
        row, col, dr, dc = t - 1, int(rng.integers(l, r)), -1, 0
    elif direction == 1:
        row, col, dr, dc = b, int(rng.integers(l, r)), 1, 0
    elif direction == 2:
        row, col, dr, dc = int(rng.integers(t, b)), l - 1, 0, -1
    else:
        row, col, dr, dc = int(rng.integers(t, b)), r, 0, 1
    for _ in range(int(rng.integers(1, 5))):
        if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]):
            break
        if not allowed[row, col] or protected[row, col]:
            break
        mask[row, col] = True
        row, col = row + dr, col + dc


def _has_attachment(mask, rect):
    t, l, b, r = rect
    h, w = mask.shape
    return bool((t > 0 and mask[t - 1, l:r].any())
                or (b < h and mask[b, l:r].any())
                or (l > 0 and mask[t:b, l - 1].any())
                or (r < w and mask[t:b, r].any()))


def _flange(rng, mask, allowed, protected, rect):
    """Attach all but one cell of a side, giving overlapping non-square windows."""
    t, l, b, r = rect
    options = []
    for skip_first in (False, True):
        lo, hi = int(skip_first), int(not skip_first)
        options.extend([
            [(t - 1, c) for c in range(l + lo, r - hi)],
            [(b, c) for c in range(l + lo, r - hi)],
            [(row, l - 1) for row in range(t + lo, b - hi)],
            [(row, r) for row in range(t + lo, b - hi)],
        ])
    feasible = [cells for cells in options if all(
        0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]
        and allowed[row, col] and not protected[row, col] for row, col in cells)]
    if not feasible:
        return False
    for row, col in feasible[int(rng.integers(len(feasible)))]:
        mask[row, col] = True
    return True


def _maximal_nonsquares(mask):
    """Diagnostics only: all maximal non-square rectangles with both sides >=2."""
    h, w = mask.shape
    result = []
    for t in range(h):
        common = np.ones(w, dtype=bool)
        for b in range(t + 1, h + 1):
            common &= mask[b - 1]
            if b - t < 2:
                continue
            changes = np.flatnonzero(np.diff(np.r_[False, common, False]))
            for l, r in zip(changes[::2], changes[1::2]):
                l, r = int(l), int(r)
                if r - l < 2 or r - l == b - t:
                    continue
                if t and mask[t - 1, l:r].all():
                    continue
                if b < h and mask[b, l:r].all():
                    continue
                result.append((t, l, b, r))
    return result


def _ablation_differs(inp, field, targets):
    # The tempting reversed order: detect A on the input, then apply B.
    early = _union(inp.shape, _squares(inp == 0))
    return not np.array_equal(_shear(early, False), _union(field.shape, targets))


def _make_pair(rng, palette, max_side, index, is_test, allowed_sides, seen):
    anchor = not is_test and index < 2
    for _ in range(100):
        low_w = max(2 * max_side + 1, 9)
        if is_test:
            h = int(rng.choice(np.arange(15, 32 - low_w, 2)))
        elif index == 0:
            h = int(rng.choice([7, 9, 11]))
        else:
            h = int(rng.choice([9, 11, 13]))
        w = int(rng.integers(low_w, 32 - h))
        allowed = _interior(h, w)
        mask = np.zeros_like(allowed)
        reserved = np.zeros_like(allowed)
        if anchor:
            sides = [2] if index == 0 else [max_side, 2]
        else:
            n = int(rng.integers(2 if is_test else 1, 4))
            sides = [int(rng.choice(sorted(allowed_sides))) for _ in range(n)]
        planted = []
        for side in sorted(sides, reverse=True):
            rect = _place(rng, mask, reserved, allowed, side, side)
            if rect is None:
                break
            planted.append(rect)
        if len(planted) != len(sides):
            continue
        # An explicit wide/tall rectangle refutes marking every square subwindow.
        distractor = None
        if anchor or rng.random() < 0.65:
            rh, rw = [(2, 3), (2, 4), (3, 2), (4, 2)][int(rng.integers(4))]
            distractor = _place(rng, mask, reserved, allowed, rh, rw)
            if anchor and distractor is None:
                continue
        density = float(rng.uniform(0.07, 0.17) if anchor else rng.uniform(0.18, 0.47))
        mask |= (rng.random(mask.shape) < density) & allowed & ~reserved
        protected = _union(mask.shape, planted + ([distractor] if distractor else []))
        for rect in planted:
            if anchor and index == 1:
                _flange(rng, mask, allowed, protected, rect)
            elif rect[2] - rect[0] >= 3 and rng.random() < 0.3:
                _flange(rng, mask, allowed, protected, rect)
            elif rng.random() < 0.7:
                _attach(rng, mask, allowed, protected, rect)
        # Outside the copied payload, padding is always neutral and included in A.
        support = _shear(np.ones((h, w), dtype=bool), False)
        field_mask = mask | ~support
        targets = _squares(field_mask)
        if (not _unambiguous(targets) or not all(r in targets for r in planted)
                or any((r[2] - r[0]) not in allowed_sides for r in targets)
                or (anchor and len(targets) != (1 if index == 0 else 2))
                or (is_test and len(targets) < 2)):
            continue
        larger = max(targets, key=lambda r: r[2] - r[0])
        if anchor and index == 1 and not _has_attachment(field_mask, larger):
            continue
        highlight_mask = _union(field_mask.shape, targets)
        if np.any(highlight_mask & ~support):
            continue
        if anchor:
            non_squares = _maximal_nonsquares(field_mask)
            if not any(support[t:b, l:r].all() and not highlight_mask[t:b, l:r].any()
                       for t, l, b, r in non_squares):
                continue
            if index == 1 and not any(_overlap(larger, r) for r in non_squares):
                continue
        colors = np.asarray(palette, dtype=np.int8)
        # Blockers are information-bearing row payload, with several clean/noisy styles.
        rr, cc = np.indices((h, w))
        style = int(rng.integers(4))
        if style == 0:
            labels = rng.integers(len(colors), size=(h, w))
        elif style == 1:
            labels = np.broadcast_to(rng.integers(len(colors), size=(h, 1)), (h, w))
        elif style == 2:
            labels = (rr + cc + int(rng.integers(len(colors)))) % len(colors)
        else:
            labels = (rr // int(rng.integers(1, 4)) + cc // 2) % len(colors)
        inp = colors[labels].copy()
        for row in range(h):
            start = h - 1 - row
            inp[row, mask[row, start:start + w]] = 0
        key = _key(inp)
        if key in seen:
            continue
        field = _shear(inp)
        if not _ablation_differs(inp, field, targets):
            continue
        out = field.copy()
        out[highlight_mask] = 1
        seen.add(key)
        return {"input": inp, "output": out}
    raise RuntimeError("exhausted 100 in-memory construction attempts")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    max_side = int(rng.choice([3, 4, 5]))
    palette = rng.choice(np.arange(2, 10), size=int(rng.integers(1, 5)), replace=False)
    puzzle = {"train": [], "test": []}
    seen, shown_sides = set(), {2, max_side}
    # Both anchors show the pool, so no hidden color convention is introduced.
    for index in range(num_train):
        chosen = palette if index < 2 else rng.choice(
            palette, size=int(rng.integers(1, len(palette) + 1)), replace=False)
        pair = _make_pair(rng, chosen, max_side, index, False, shown_sides, seen)
        puzzle["train"].append(pair)
        shown_sides.update(r[2] - r[0] for r in _squares(_shear(pair["input"]) == 0))
    for index in range(num_test):
        chosen = rng.choice(palette, size=int(rng.integers(1, len(palette) + 1)), replace=False)
        puzzle["test"].append(_make_pair(rng, chosen, max_side, index, True,
                                         shown_sides, seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, sizes, train_heights = set(), set(), []
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            h, w = inp.shape
            assert h % 2 == 1 and h >= 7 and w >= 5 and h + w - 1 <= 30
            assert not np.any(inp == 1) and np.any(inp == 0)
            assert np.all(inp[:, [0, -1]] >= 2), "neutral touches padding"
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            field = _shear(inp)
            targets = _squares(field == 0)
            assert targets and _unambiguous(targets), "missing or overlapping squares"
            union = _union(field.shape, targets)
            support = _shear(np.ones(inp.shape, dtype=bool), False)
            assert not np.any(union & ~support), "padding became a trivial target"
            expected = field.copy()
            expected[union] = 1
            assert np.array_equal(out, expected)
            assert np.array_equal(solve(inp.copy()), out)
            assert np.any(out == 1), "A has no effect"
            assert _ablation_differs(inp, field, targets), "A and B commute on this pair"
            # Undo only the highlight and recover every complete source row exactly.
            recovered = out.copy()
            recovered[recovered == 1] = 0
            assert np.array_equal(recovered, field)
            if split == "train":
                train_heights.append(h)
                sizes.update(r[2] - r[0] for r in targets)
                if index < 2:
                    assert len(targets) == (1 if index == 0 else 2)
                    assert any(support[t:b, l:r].all() and not union[t:b, l:r].any()
                               for t, l, b, r in _maximal_nonsquares(field == 0)), \
                        "anchor lacks an unmarked payload rectangle distractor"
                if index == 0:
                    assert targets[0][2] - targets[0][0] == 2
                if index == 1:
                    larger = max(targets, key=lambda r: r[2] - r[0])
                    assert {r[2] - r[0] for r in targets} == {2, larger[2] - larger[0]}
                    assert larger[2] - larger[0] >= 3
                    assert _has_attachment(field == 0, larger)
                    assert any(_overlap(larger, r)
                               for r in _maximal_nonsquares(field == 0))
            else:
                assert h > max(train_heights), "missing height extrapolation"
                assert len(targets) >= 2
                assert {r[2] - r[0] for r in targets} <= sizes
    assert 2 in sizes and len(sizes) >= 2


def _direct_squares(mask):
    """Small-grid assertion oracle using literal windows and four side tests."""
    h, w = mask.shape
    result = []
    for s in range(2, min(h, w) + 1):
        for t in range(h - s + 1):
            for l in range(w - s + 1):
                b, r = t + s, l + s
                if not mask[t:b, l:r].all():
                    continue
                if ((t and mask[t - 1, l:r].all())
                        or (b < h and mask[b, l:r].all())
                        or (l and mask[t:b, l - 1].all())
                        or (r < w and mask[t:b, r].all())):
                    continue
                result.append((t, l, b, r))
    return result


def self_test():
    """Bounded, streaming checks; stores no puzzles, logs, or traces on disk."""
    rng = np.random.default_rng(917)
    for _ in range(30):
        mask = rng.random((int(rng.integers(2, 8)), int(rng.integers(2, 8)))) < 0.6
        assert _squares(mask) == _direct_squares(mask)
    # Full-side extension disqualifies, while a one-cell protrusion does not.
    mask = np.zeros((6, 8), dtype=bool)
    mask[1:4, 2:5] = True
    mask[0, 3] = True
    assert _squares(mask) == [(1, 2, 4, 5)]
    mask[0, 2:5] = True
    assert _squares(mask) == []
    # Widening must preserve full row vectors even if edge cells are neutral.
    payload = np.array([[0, 2, 0], [3, 0, 4], [0, 5, 0]])
    assert np.array_equal(_shear(payload), np.array([[0, 0, 0, 2, 0],
                                                    [0, 3, 0, 4, 0],
                                                    [0, 5, 0, 0, 0]]))
    shapes, layouts, palettes, target_counts, sizes, hashes = set(), set(), set(), set(), set(), set()
    attachments = boundary = width30 = nonsquare_overlap = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprint = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                field = _shear(inp)
                targets = _squares(field == 0)
                fingerprint.update(bytes(inp.shape) + inp.tobytes() + out.tobytes())
                shapes.add(inp.shape)
                layouts.add((inp.shape, tuple(targets)))
                palettes.add(tuple(np.unique(inp)))
                target_counts.add(len(targets))
                sizes.update(r[2] - r[0] for r in targets)
                attachments += any(_has_attachment(field == 0, r) for r in targets)
                boundary += any(r[0] == 0 or r[2] == len(inp) for r in targets)
                width30 += out.shape[1] == 30
                nonsquare_overlap += any(_overlap(a, b) for a in targets
                                         for b in _maximal_nonsquares(field == 0))
                before = inp.copy()
                assert np.array_equal(solve(inp), out) and np.array_equal(inp, before)
        hashes.add(fingerprint.digest())
    assert len(hashes) == 42 and len(shapes) >= 25 and len(layouts) >= 150
    assert len(palettes) >= 20 and {1, 2, 3} <= target_counts
    assert {2, 3, 4, 5} <= sizes
    assert attachments >= 42 and boundary >= 5 and width30 >= 3 and nonsquare_overlap >= 5
    a, b = generate_puzzle(91, 2, 1), generate_puzzle(91, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for split in a
               for x, y in zip(a[split], b[split]) for k in x)
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(777, train, test)
            assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
    for train, test in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)]:
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal counts, geometry assertions)")
