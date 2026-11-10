"""Rank blank rectangles first; highlight the winner only if it is square.

Fixed family conventions: background 0, output-only highlight 1, blockers 2..9.
No file I/O, learned state, parent imports, or seed information is used by solve.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


HOST_SEEDS = (0, 50554188, 113392951, 533126671, 90503112, 636418900,
              501040442, 723473349, 939615110, 979820501, 172352310,
              609823581, 683808907, 728527315, 961430434, 621923855,
              245377957, 311884280, 384345668, 76917059, 618413934,
              770303460, 851205843, 804214978, 327044804, 397846352,
              431229014, 250034471, 116466628, 837127950, 615459453,
              545775266, 554677928, 780126628, 950903978, 949831979,
              416843072, 222260944, 436025041, 156227208, 161479638,
              375066706)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= int(num_train) <= 10:
        raise ValueError("num_train must be an integer from 2 through 10")
    if not _integer(num_test) or not 1 <= int(num_test) <= 4:
        raise ValueError("num_test must be an integer from 1 through 4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grids must have 1..30 rows and columns")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid cells must be integer ARC colors")
    return a


def _runs(bits):
    """Half-open column spans of set bits; work is bounded by width <= 30."""
    while bits:
        left = (bits & -bits).bit_length() - 1
        shifted = bits >> left
        length = (shifted ^ (shifted + 1)).bit_length() - 1
        yield left, left + length
        bits &= ~(((1 << length) - 1) << left)


def _maximal_rectangles(mask):
    """Enumerate all side-nonextendable rectangles with h,w >= 2.

    A rectangle is (top, left, height, width), using half-open slices. Runs
    of common row bits supply horizontal maximality; two row tests supply
    vertical maximality. Partial protrusions do not disqualify a rectangle.
    """
    height, width = mask.shape
    rows = [sum(1 << c for c in np.flatnonzero(row).tolist()) for row in mask]
    found = []
    for top in range(height - 1):
        common = rows[top]
        for bottom in range(top + 1, height):
            common &= rows[bottom]
            if not common:
                break
            for left, right in _runs(common):
                if right - left < 2:
                    continue
                span = ((1 << (right - left)) - 1) << left
                if top and rows[top - 1] & span == span:
                    continue
                if bottom + 1 < height and rows[bottom + 1] & span == span:
                    continue
                found.append((top, left, bottom - top + 1, right - left))
    return found


def _area(rect):
    return rect[2] * rect[3]


def _winner(rectangles):
    if not rectangles:
        raise ValueError("family inputs require an eligible background rectangle")
    area = max(map(_area, rectangles))
    winners = [r for r in rectangles if _area(r) == area]
    if len(winners) != 1:
        raise ValueError("greatest eligible area must have a unique rectangle")
    return winners[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Apply the family rule to any valid input, without generation context.

    Tied maxima and absent eligible rectangles are outside the family and
    raise ValueError. All other geometry is allowed, including overlap.
    """
    before = _grid(input_grid)
    if np.any(before == 1):
        raise ValueError("1 is reserved for the output highlight")
    winner = _winner(_maximal_rectangles(before == 0))
    out = before.copy()
    top, left, rh, rw = winner
    if rh == rw:  # min side 2 already enforced in candidate enumeration
        out[top:top + rh, left:left + rw] = 1
    return out


def _overlap(a, b):
    return (a[0] < b[0] + b[2] and b[0] < a[0] + a[2]
            and a[1] < b[1] + b[3] and b[1] < a[1] + a[3])


def _paint(mask, rect):
    top, left, rh, rw = rect
    mask[top:top + rh, left:left + rw] = True


def _rectangle_mask(shape, rect):
    mask = np.zeros(shape, dtype=bool)
    _paint(mask, rect)
    return mask


def _longest_line(mask):
    best = 0
    for array in (mask, mask.T):
        for row in array:
            run = 0
            for cell in row:
                run = run + 1 if cell else 0
                best = max(best, run)
    return best


def _features(mask, rectangles, winner):
    others = [r for r in rectangles if r != winner]
    top, left, rh, rw = winner
    outside = mask & ~_rectangle_mask(mask.shape, winner)
    adjacent = 0
    if top:
        adjacent += int(mask[top - 1, left:left + rw].sum())
    if top + rh < mask.shape[0]:
        adjacent += int(mask[top + rh, left:left + rw].sum())
    if left:
        adjacent += int(mask[top:top + rh, left - 1].sum())
    if left + rw < mask.shape[1]:
        adjacent += int(mask[top:top + rh, left + rw].sum())
    return {"square": rh == rw, "winner_shape": (rh, rw),
            "candidates": len(rectangles), "area": _area(winner),
            "smaller_squares": sum(r[2] == r[3] for r in others),
            "overlaps": sum(_overlap(r, winner) for r in others),
            "attachments": adjacent,
            "runner_up": max(map(_area, others), default=0),
            "long_line": _longest_line(outside) > _area(winner),
            "edge": top == 0 or left == 0 or top + rh == mask.shape[0]
                    or left + rw == mask.shape[1]}


def _anchor(rng, limit):
    if limit == 0:
        return 0
    mode = int(rng.integers(4))
    return 0 if mode == 0 else limit if mode == 1 else int(rng.integers(limit + 1))


def _dimensions(rng, rh, rw, test, long_line):
    if long_line:
        # Teaching squares have side 3..5, so the long 1-D distractor fits.
        height = int(rng.integers(max(rh + 4, 8), max(rh + 5, 16)))
        width = int(rng.integers(max(rw + 6, rh * rw + 1, 16), 31))
    elif test:
        height = int(rng.integers(max(rh + 4, 14), 31))
        width = int(rng.integers(max(rw + 4, 16), 31))
    else:
        height = int(rng.integers(max(rh + 3, 7), 25))
        width = int(rng.integers(max(rw + 3, 9), 29))
    return height, width


def _attach_strip(mask, target, rng):
    """Add a partial side whose overlapping rectangle loses by area."""
    top, left, rh, rw = target
    choices = []
    for side in range(4):
        span, normal = (rw, rh) if side < 2 else (rh, rw)
        available = (top > 0, top + rh < mask.shape[0],
                     left > 0, left + rw < mask.shape[1])[side]
        length_max = min(span - 1, (_area(target) - 1) // (normal + 1))
        if not available or length_max < 2:
            continue
        # Near ties occur naturally: an s-square competes with (s-1)x(s+1).
        for start in range(span - length_max + 1):
            length = length_max if rng.random() < .75 else int(rng.integers(2, length_max + 1))
            if start + length > span:
                continue
            if side < 2:
                rect = (top - 1 if side == 0 else top + rh, left + start, 1, length)
            else:
                rect = (top + start, left - 1 if side == 2 else left + rw, length, 1)
            region = _rectangle_mask(mask.shape, rect)
            if not np.all(mask[region]):
                choices.append(rect)
    # Several individually safe strips can jointly extend a complete side or
    # create a tied maximum, especially around a thin 2x4 winner. Check their
    # union incrementally instead of spending most retries on invalid masks.
    rng.shuffle(choices)
    for rect in choices:
        trial = mask.copy()
        _paint(trial, rect)
        try:
            safe = _winner(_maximal_rectangles(trial)) == target
        except ValueError:
            safe = False
        if safe:
            _paint(mask, rect)
            return True
    return False


def _place_separate(mask, rh, rw, rng):
    height, width = mask.shape
    for _ in range(70):
        top = _anchor(rng, height - rh)
        left = _anchor(rng, width - rw)
        if not mask[max(0, top - 1):min(height, top + rh + 1),
                    max(0, left - 1):min(width, left + rw + 1)].any():
            rect = (top, left, rh, rw)
            _paint(mask, rect)
            return rect
    return None


def _add_line(mask, area, rng):
    height, width = mask.shape
    choices = [axis for axis, limit in ((0, width), (1, height)) if limit > area]
    rng.shuffle(choices)
    for axis in choices:
        limit = width if axis == 0 else height
        length = int(rng.integers(area + 1, min(limit, area + 5) + 1))
        if _place_separate(mask, 1 if axis == 0 else length,
                           length if axis == 0 else 1, rng) is not None:
            return True
    return False


def _branch(mask, target, rng):
    top, left, rh, rw = target
    row = int(rng.integers(top, top + rh))
    col = int(rng.integers(left, left + rw))
    dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
    for step in range(int(rng.integers(2, 12))):
        row, col = row + dr, col + dc
        if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]):
            break
        mask[row, col] = True
        if step > 0 and rng.random() < .32:
            dr, dc = dc, -dr


def _make_pair(rng, shape, blockers, *, teaching=False, test=False,
               long_line=False, forbidden_shapes=()):
    rh, rw = shape
    area = rh * rw
    for _ in range(160):
        height, width = _dimensions(rng, rh, rw, test, long_line)
        mask = np.zeros((height, width), dtype=bool)
        target = (_anchor(rng, height - rh), _anchor(rng, width - rw), rh, rw)
        _paint(mask, target)
        strip_count = int(rng.integers(2, 5)) if test else int(rng.integers(1, 3))
        if rh == rw == 2:
            strip_count = 0
        for _ in range(strip_count):
            _attach_strip(mask, target, rng)
        if teaching:
            max_square = min(5, int(np.sqrt(area - 1)))
            side = int(rng.integers(2, max_square + 1))
            if _place_separate(mask, side, side, rng) is None:
                continue
        if long_line and not _add_line(mask, area, rng):
            continue
        # Variable rivals, not a compulsory count or fixed layout vocabulary.
        rival_count = int(rng.integers(1, 5)) if test else int(rng.integers(0, 3))
        rivals = [(h, w) for h in range(2, min(11, height) + 1)
                  for w in range(2, min(11, width) + 1) if h * w < area]
        if rivals:
            rivals.sort(key=lambda s: s[0] * s[1], reverse=True)
            for _ in range(rival_count):
                pool = rivals[:max(1, len(rivals) // 2)] if rng.random() < .7 else rivals
                h, w = pool[int(rng.integers(len(pool)))]
                _place_separate(mask, h, w, rng)
        if not long_line and rng.random() < .3:
            _add_line(mask, area, rng)
        for _ in range(int(rng.integers(0, 5 if test else 3))):
            _branch(mask, target, rng)
        density = float(rng.uniform(.12, .40) if test else rng.uniform(.04, .30))
        mask |= rng.random(mask.shape) < density
        rectangles = _maximal_rectangles(mask)
        try:
            winner = _winner(rectangles)
        except ValueError:
            continue
        if winner != target:
            continue
        features = _features(mask, rectangles, winner)
        if teaching and (not features["smaller_squares"] or not features["overlaps"]):
            continue
        if long_line and not features["long_line"]:
            continue
        if test and area > 4 and (features["candidates"] < 4 or not features["overlaps"]):
            continue
        # D4 transformations alter edge/corner/orientation conventions together.
        mask = np.rot90(mask, int(rng.integers(4)))
        if rng.random() < .5:
            mask = np.fliplr(mask)
        if mask.shape in forbidden_shapes:
            continue
        grid = rng.choice(blockers, size=mask.shape).astype(np.int8)
        grid[mask] = 0
        positions = np.argwhere(~mask)
        if len(positions) < len(blockers):
            continue
        for index, color in zip(rng.permutation(len(positions)), blockers):
            row, col = positions[int(index)]
            grid[row, col] = color
        return {"input": grid, "output": solve(grid)}
    raise RuntimeError("bounded rectangle construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return distinct pairs sharing the fixed rule and color conventions."""
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    blockers = rng.permutation(np.arange(2, 10))[:int(rng.integers(1, 6))].tolist()
    side = int(rng.integers(3, 6))
    nonsquares = [(h, w) for h in range(2, 7) for w in range(h + 1, 11)
                  if h * w >= 8]
    negative = nonsquares[int(rng.integers(len(nonsquares)))]
    teaching_shapes = [(side, side), negative]
    rng.shuffle(teaching_shapes)
    puzzle = {"train": [], "test": []}
    used = set()
    learned_shapes = []
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            teaching = split == "train" and index < 2
            if teaching:
                shape = teaching_shapes[index]
            elif split == "train":
                if rng.random() < .58:
                    s = int(rng.integers(2, 8))
                    shape = (s, s)
                else:
                    shape = nonsquares[int(rng.integers(len(nonsquares)))]
            else:
                positive_shapes = [s for s in learned_shapes if s[0] == s[1]]
                negative_shapes = [s for s in learned_shapes if s[0] != s[1]]
                # Both branches are demonstrated. Tests recombine competition,
                # attachment and placement at a fresh canvas shape.
                choices = positive_shapes if rng.random() < .5 else negative_shapes
                shape = choices[int(rng.integers(len(choices)))]
            if teaching and index == 0:
                active = blockers
            else:
                n = int(rng.integers(1, len(blockers) + 1))
                active = rng.choice(blockers, size=n, replace=False).tolist()
            for _ in range(12):
                pair = _make_pair(rng, shape, active, teaching=teaching,
                                  test=split == "test",
                                  long_line=teaching and shape[0] == shape[1],
                                  forbidden_shapes={p["input"].shape for p in puzzle["train"]}
                                  if split == "test" else ())
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    if split == "train":
                        learned_shapes.append(shape)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_generated_puzzle(puzzle):
    """Check semantics and teaching coverage on returned grids, not metadata."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    used, train_colors, train_shapes, demonstrated = set(), set(), set(), set()
    teaching_positive = teaching_negative = long_line = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair schema")
            before, after = _grid(pair["input"]), _grid(pair["output"])
            _require(before.shape == after.shape, "shape changed")
            _require(0 in before and 1 not in before, "invalid background/highlight convention")
            colors = set(map(int, np.unique(before))) - {0}
            _require(bool(colors) and colors <= set(range(2, 10)), "invalid blockers")
            key = (before.shape, before.astype(np.int8).tobytes())
            _require(key not in used, "duplicate input")
            used.add(key)
            mask = before == 0
            rectangles = _maximal_rectangles(mask)
            winner = _winner(rectangles)
            top, left, rh, rw = winner
            feature = _features(mask, rectangles, winner)
            _require(np.array_equal(after, solve(before)), "incorrect output")
            changed = before != after
            expected_change = _rectangle_mask(before.shape, winner) if rh == rw else np.zeros_like(mask)
            _require(np.array_equal(changed, expected_change), "conditional recolor differs")
            _require(np.all(after[changed] == 1), "wrong highlight color")
            # Inspect all four complete adjacent sides explicitly.
            for strip in (mask[top - 1:top, left:left + rw] if top else np.array([], dtype=bool),
                          mask[top + rh:top + rh + 1, left:left + rw],
                          mask[top:top + rh, left - 1:left] if left else np.array([], dtype=bool),
                          mask[top:top + rh, left + rw:left + rw + 1]):
                _require(not strip.size or not strip.all(), "winner is extendable")
            if split == "train":
                train_colors |= colors
                train_shapes.add(before.shape)
                demonstrated.add(tuple(sorted((rh, rw))))
                if feature["smaller_squares"]:
                    teaching_positive |= rh == rw and feature["overlaps"] > 0
                    teaching_negative |= rh != rw and feature["overlaps"] > 0
                long_line |= feature["long_line"]
            else:
                _require(colors <= train_colors, "unseen test blocker")
                _require(before.shape not in train_shapes, "test canvas does not shift")
                _require(tuple(sorted((rh, rw))) in demonstrated, "unshown winner dimensions")
                if rh * rw > 4:
                    _require(feature["candidates"] >= 4 and feature["overlaps"] > 0,
                             "test lacks recombined competition")
    _require(teaching_positive and teaching_negative, "training must expose both parent ablations")
    _require(long_line, "training must expose the two-dimensional minimum")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _small_checks():
    # Hand-specified outcomes exercise filter order, area, overlap, edges,
    # 2x2 minimum, identity, full extensions, tie rejection and immutability.
    a = np.full((7, 10), 5, dtype=np.int8)
    a[:3, :3] = 0
    a[:2, 3] = 0  # overlapping maximal 2x4 loses to the 3x3
    a[5:7, 8:10] = 0
    expected = a.copy()
    expected[:3, :3] = 1
    frozen = a.copy()
    assert np.array_equal(solve(a), expected) and np.array_equal(a, frozen)
    a[:3, 3] = 0  # now the winner is 3x4: every cell remains unchanged
    assert np.array_equal(solve(a), a)
    b = np.array([[0, 0, 3, 0, 0, 0, 0, 0], [0, 0, 3, 3, 3, 3, 3, 3]], dtype=np.int8)
    expected = b.copy()
    expected[:, :2] = 1
    assert np.array_equal(solve(b), expected)
    for bad in (np.array([[0, 0, 4, 0, 0], [0, 0, 4, 0, 0]]),
                np.array([[0, 0, 0], [4, 4, 4]])):
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-family ambiguity accepted")
    # Exhaustive windows on bounded 5x6 masks validate candidate enumeration
    # and demonstrate that removing maximality cannot change the area winner.
    rng = np.random.default_rng(8675309)
    for _ in range(24):
        mask = rng.random((5, 6)) < .64
        all_windows, maximal = [], []
        for top in range(4):
            for bottom in range(top + 2, 6):
                for left in range(5):
                    for right in range(left + 2, 7):
                        if not mask[top:bottom, left:right].all():
                            continue
                        rect = (top, left, bottom - top, right - left)
                        all_windows.append(rect)
                        extends = ((top > 0 and mask[top - 1, left:right].all())
                                   or (bottom < 5 and mask[bottom, left:right].all())
                                   or (left > 0 and mask[top:bottom, left - 1].all())
                                   or (right < 6 and mask[top:bottom, right].all()))
                        if not extends:
                            maximal.append(rect)
        actual = _maximal_rectangles(mask)
        assert set(actual) == set(maximal)
        if all_windows:
            best = max(map(_area, all_windows))
            assert {r for r in all_windows if _area(r) == best} == {r for r in actual if _area(r) == best}


def self_test():
    """Bounded 42-seed semantic/variation audit; no samples or traces stored."""
    _small_checks()
    shapes, winner_shapes, fingerprints, structures, blocker_palettes = set(), set(), set(), set(), set()
    stats = {"pairs": 0, "changed": 0, "identity": 0, "overlap": 0,
             "edge": 0, "interior": 0, "long_line": 0,
             "near_tie": 0, "multicolor": 0, "dimension_30": 0}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        if seed == 0:
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        for pair in puzzle["train"] + puzzle["test"]:
            grid = pair["input"]
            mask = grid == 0
            rectangles = _maximal_rectangles(mask)
            win = _winner(rectangles)
            feature = _features(mask, rectangles, win)
            shapes.add(grid.shape)
            winner_shapes.add((win[2], win[3]))
            structures.add(hashlib.sha256(bytes(grid.shape) + np.packbits(mask).tobytes()).hexdigest())
            blocker_palettes.add(tuple(np.unique(grid[~mask]).tolist()))
            assert np.array_equal(solve(grid.copy()), pair["output"])
            # Removing A's square gate (B alone) changes every identity answer.
            without_a = grid.copy()
            _paint_region = (slice(win[0], win[0] + win[2]), slice(win[1], win[1] + win[3]))
            without_a[_paint_region] = 1
            if not feature["square"]:
                assert not np.array_equal(without_a, pair["output"])
            # Removing B's ranking highlights smaller maximal squares too.
            without_b = grid.copy()
            for rect in rectangles:
                if rect[2] == rect[3]:
                    top, left, h, w = rect
                    without_b[top:top + h, left:left + w] = 1
            if feature["smaller_squares"]:
                assert not np.array_equal(without_b, pair["output"])
            stats["pairs"] += 1
            stats["changed" if feature["square"] else "identity"] += 1
            stats["overlap"] += feature["overlaps"] > 0
            stats["edge" if feature["edge"] else "interior"] += 1
            stats["long_line"] += feature["long_line"]
            stats["near_tie"] += feature["runner_up"] >= .8 * feature["area"]
            stats["multicolor"] += len(np.unique(grid)) > 2
            stats["dimension_30"] += 30 in grid.shape
    assert len(fingerprints) == 42 and len(shapes) >= 50
    assert len(structures) == stats["pairs"] and len(winner_shapes) >= 15
    assert len(blocker_palettes) >= 15
    assert stats["changed"] >= 42 and stats["identity"] >= 42
    assert stats["overlap"] >= 100 and stats["long_line"] >= 42
    assert stats["near_tie"] >= 50 and stats["multicolor"] >= 30
    assert stats["edge"] >= 20 and stats["interior"] >= 10 and stats["dimension_30"] > 0
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            puzzle = generate_puzzle(314159, n, m)
            assert len(puzzle["train"]) == n and len(puzzle["test"]) == m
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(stats, seeds=42, canvas_shapes=len(shapes), winner_shapes=len(winner_shapes),
                background_structures=len(structures), blocker_palettes=len(blocker_palettes),
                maximality_independently_necessary=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
