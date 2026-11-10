"""Bottom-anchored shear, followed by the largest empty rectangle.

Fixed family roles: 0 is empty/padding and 6 is the output-only highlight.
No files, examples, seeds, or generation metadata are consulted by solve().
"""
from __future__ import annotations

import hashlib
import json
from collections import deque

import numpy as np


HOST_SEEDS = (
    0, 128340739, 109182927, 325806783, 910744418, 502598503,
    214832728, 51564085, 163333444, 712316958, 54845339, 738146105,
    155649645, 946619870, 907933065, 325028213, 537849762, 626564828,
    625396808, 835331150, 953836180, 771573931, 325749578, 685696201,
    541185820, 678251181, 932420480, 955178367, 414815642, 970047934,
    331591230, 982191377, 43522600, 182098175, 641560882, 186452987,
    363136783, 955658118, 430013472, 248480758, 866763742, 696922795,
)
BLOCKERS = (1, 2, 3, 4, 5, 7, 8, 9)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train")
    b = _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    return a, b


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return grid.astype(np.int8, copy=True)


def _shear(grid):
    h, w = grid.shape
    field = np.zeros((h, h + w - 1), dtype=grid.dtype)
    for r in range(h):
        start = h - 1 - r
        field[r, start:start + w] = grid[r]
    return field


def _rectangles(empty):
    """Yield (area, top, left, height, width) for maximal-width row spans.

    Any greatest-area rectangle must use a whole common run for its row span;
    otherwise extending it horizontally increases its area. Thus this searches
    the true maximum, including overlaps, without storing all subrectangles.
    """
    h, w = empty.shape
    row_bits = [sum(1 << c for c, value in enumerate(row) if value) for row in empty]
    for top in range(h - 1):
        common = (1 << w) - 1
        for bottom in range(top, h):
            common &= row_bits[bottom]
            rh = bottom - top + 1
            if not common:
                break
            if rh < 2:
                continue
            bits = common
            while bits:
                left = (bits & -bits).bit_length() - 1
                shifted = bits >> left
                # Number of trailing one bits = length of this common run.
                rw = (shifted ^ (shifted + 1)).bit_length() - 1
                bits &= ~(((1 << rw) - 1) << left)
                if rw >= 2:
                    yield rh * rw, top, left, rh, rw


def _best(empty):
    area, winners = 0, []
    for candidate in _rectangles(empty):
        if candidate[0] > area:
            area, winners = candidate[0], [candidate[1:]]
        elif candidate[0] == area:
            winners.append(candidate[1:])
    if len(winners) != 1:
        raise ValueError("the sheared field must have one unique largest eligible rectangle")
    return winners[0]


def _mask(shape, rect):
    top, left, rh, rw = rect
    mask = np.zeros(shape, dtype=bool)
    mask[top:top + rh, left:left + rw] = True
    return mask


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    h, w = inp.shape
    if h < 2 or h + w - 1 > 30 or np.any(inp == 6):
        raise ValueError("require height >= 2, h+w-1 <= 30, and no input highlight")
    field = _shear(inp)
    top, left, rh, rw = _best(field == 0)
    field[top:top + rh, left:left + rw] = 6
    return field


def _padding_bound(h):
    # Both triangular margins independently attain this same bound.
    return max((k * (h - k) for k in range(2, h - 1)), default=0)


def _carve(empty, rect):
    """Carve a rectangle in SHEARED coordinates, touching only payload cells.

    This is input construction, never an extra rule of solve(). Padding is
    automatically zero when the completed input is sheared.
    """
    h, w = empty.shape
    top, left, rh, rw = rect
    for r in range(top, top + rh):
        shift = h - 1 - r
        a, b = max(0, left - shift), min(w, left + rw - shift)
        if a < b:
            empty[r, a:b] = True


def _paint(empty, rng, palette, show_all=False):
    h, w = empty.shape
    rr, cc = np.indices((h, w))
    style = int(rng.integers(4))
    if style == 0:
        data = np.full((h, w), int(rng.choice(palette)), dtype=np.int8)
    elif style == 1:
        data = np.asarray(palette, dtype=np.int8)[(rr + cc // int(rng.integers(1, 4))) % len(palette)]
    elif style == 2:
        data = np.asarray(palette, dtype=np.int8)[(rr // int(rng.integers(1, 4))) % len(palette)]
    else:
        data = rng.choice(palette, size=(h, w)).astype(np.int8)
    data[empty] = 0
    if show_all:
        positions = np.argwhere(~empty)
        if len(positions) < len(palette):
            raise ValueError("not enough blocker positions")
        for pos, color in zip(rng.permutation(len(positions)), palette):
            r, c = positions[int(pos)]
            data[r, c] = color
    return data


def _anchor(rng, shape, palette):
    """A rectangular hole with an overlapping smaller rival and a long line."""
    h, w = shape
    rh, rw = 3, h - 2  # h=5 -> 3x3; h=6 -> 3x4.
    tops = [t for t in range(h - rh + 1) if t >= 2 or t + rh <= h - 2]
    top = int(rng.choice(tops))
    left = int(rng.integers(h - 1 - top + 2, h - top - rh + w - rw - 1))
    rect = (top, left, rh, rw)
    empty = np.zeros(shape, dtype=bool)
    _carve(empty, rect)
    wing_left = left - 1 if rng.random() < .5 else left + rw
    _carve(empty, (top + int(rng.integers(2)), wing_left, 2, 1))
    line_row = 0 if top >= 2 else h - 1
    empty[line_row] = True
    inp = _paint(empty, rng, palette, show_all=True)
    assert _best(_shear(inp) == 0) == rect
    return {"input": inp, "output": solve(inp)}


def _bridge(rng, shape, palette):
    """Clean corner rectangle joining original zeros to triangular padding."""
    h, w = shape
    wide = h + w - 1
    choices = [(k, b) for k in range(2, h)
               for b in range(max(2, h - k + 2), min(w - 2, h + 5) + 1)
               if k * b > _padding_bound(h)]
    k, b = choices[int(rng.integers(len(choices)))]
    rect = (0, 0, k, b) if rng.random() < .5 else (h - k, wide - b, k, b)
    empty = np.zeros(shape, dtype=bool)
    _carve(empty, rect)
    inp = _paint(empty, rng, palette)
    assert _best(_shear(inp) == 0) == rect
    return {"input": inp, "output": solve(inp)}


def _shape(rng, low, high, is_test=False):
    h = int(rng.integers(low, high + 1))
    minimum = 3 if h >= 10 else 5
    maximum = 31 - h
    w = maximum if rng.random() < .25 else int(rng.integers(minimum, maximum + 1))
    if is_test and h <= 4:
        w = int(rng.integers(12, maximum + 1))
    return h, w


def _plant(rng, shape):
    h, w = shape
    wide = h + w - 1
    bound = _padding_bound(h)
    inside = [(a, b) for a in range(2, min(h, w - 1) + 1)
              for b in range(2, w - a + 2) if a * b > bound]
    if inside and rng.random() < .5:
        rh, rw = inside[int(rng.integers(len(inside)))]
        top = int(rng.integers(h - rh + 1))
        left = int(rng.integers(h - 1 - top, h - top - rh + w - rw + 1))
        return top, left, rh, rw
    choices = [(k, b) for k in range(2, h + 1)
               for b in range(max(2, h - k + 1), wide)
               if k * b > bound]
    rh, rw = choices[int(rng.integers(len(choices)))]
    if rng.random() < .5:
        return 0, 0, rh, rw
    return h - rh, wide - rw, rh, rw


def _general(rng, low, high, palette, used, is_test=False):
    for _ in range(160):
        shape = _shape(rng, low, high, is_test)
        h, w = shape
        wide = h + w - 1
        density = 0.0 if rng.random() < .3 else float(rng.uniform(.02, .28))
        empty = rng.random(shape) < density
        rect = _plant(rng, shape)
        _carve(empty, rect)
        for _ in range(int(rng.integers(4))):
            if rng.random() < .5:
                rh = int(rng.integers(1, h + 1))
                rw = int(rng.integers(1, wide + 1))
                if rh * rw >= rect[2] * rect[3]:
                    continue
            elif rng.random() < .5:
                rh, rw = 1, int(rng.integers(2, wide + 1))
            else:
                rh, rw = int(rng.integers(2, h + 1)), 1
            top = int(rng.integers(h - rh + 1))
            left = int(rng.integers(wide - rw + 1))
            _carve(empty, (top, left, rh, rw))
        if not np.any(~empty) or np.count_nonzero(empty) < 4:
            continue
        active = rng.choice(palette, size=int(rng.integers(1, len(palette) + 1)), replace=False)
        inp = _paint(empty, rng, active)
        try:
            winner = _best(_shear(inp) == 0)
        except ValueError:
            continue
        carried = np.count_nonzero(_mask((h, wide), winner) & _shear(np.ones(shape, dtype=bool)))
        if carried < max(4, int(.08 * winner[2] * winner[3])):
            continue
        if _signature(inp) in used:
            continue
        return {"input": inp, "output": solve(inp)}
    raise RuntimeError("bounded generation exhausted without an unambiguous example")


def _signature(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    a, b = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = rng.choice(BLOCKERS, size=int(rng.integers(1, 6)), replace=False)
    taller = bool(rng.integers(2))
    h = int(rng.integers(5, 7))
    w = int(rng.integers(16, 32 - h))
    first = _anchor(rng, (h, w), palette)
    second = _bridge(rng, (h, w), palette)
    puzzle = {"train": [first, second], "test": []}
    used = {_signature(first["input"]), _signature(second["input"])}
    for split, count, low, high in (("train", a - 2, 3 if taller else 5, 8 if taller else 10),
                                   ("test", b, 10 if taller else 2, 18 if taller else 4)):
        for _ in range(count):
            pair = _general(rng, low, high, palette, used, split == "test")
            used.add(_signature(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _longest_line(empty):
    longest = 0
    for array in (empty, empty.T):
        for row in array:
            run = 0
            for value in row:
                run = run + 1 if value else 0
                longest = max(longest, run)
    return longest


def _component_size(empty, start):
    seen = {start}
    queue = deque([start])
    h, w = empty.shape
    while queue:
        r, c = queue.popleft()
        for y, x in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= y < h and 0 <= x < w and empty[y, x] and (y, x) not in seen:
                seen.add((y, x))
                queue.append((y, x))
    return len(seen)


def _features(inp):
    h, w = inp.shape
    field = _shear(inp)
    rect = _best(field == 0)
    top, left, rh, rw = rect
    target = _mask(field.shape, rect)
    support = _shear(np.ones(inp.shape, dtype=bool))
    area = rh * rw
    competitors = []
    overlaps = 0
    for value, y, x, a, b in _rectangles(field == 0):
        if top <= y and left <= x and y + a <= top + rh and x + b <= left + rw:
            continue
        competitors.append(value)
        overlaps += y < top + rh and top < y + a and x < left + rw and left < x + b
    unchanged = (field == 0) & ~target
    aligned = np.zeros_like(field)
    aligned[:, :w] = inp
    try:
        aligned_rect = _best(aligned == 0)
    except ValueError:
        aligned_rect = None
    return {"rect": rect, "area": area,
            "carried": int(np.count_nonzero(target & support)),
            "padding": int(np.count_nonzero(target & ~support)),
            "component": _component_size(field == 0, (top, left)),
            "long_line": _longest_line(unchanged),
            "overlaps": int(overlaps), "runner_up": max(competitors, default=0),
            "aligned_differs": aligned_rect != rect}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    used, training, test_heights, train_colors = set(), [], [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            h, w = inp.shape
            assert h >= 2 and h + w - 1 <= 30
            assert np.any(inp == 0) and np.any(inp != 0) and not np.any(inp == 6)
            assert _signature(inp) not in used
            used.add(_signature(inp))
            field = _shear(inp)
            info = _features(inp)
            mask = _mask(field.shape, info["rect"])
            assert out.shape == field.shape
            assert np.array_equal(out == 6, mask)
            assert np.array_equal(out[~mask], field[~mask])
            assert np.all(field[mask] == 0)
            assert info["area"] > _padding_bound(h) and info["carried"] >= 1
            assert np.array_equal(solve(inp), out)
            colors = set(int(c) for c in np.unique(inp) if c != 0)
            if split == "train":
                training.append((inp.shape, info))
                train_colors.update(colors)
            else:
                test_heights.append(h)
    assert all(set(int(c) for c in np.unique(p["input"]) if c != 0) <= train_colors
               for p in puzzle["test"])
    # These are demonstration requirements, not extra selection rules.
    assert any(x["padding"] == 0 for _, x in training)
    assert any(0 < x["padding"] < x["area"] for _, x in training)
    assert any(x["long_line"] > x["area"] and x["overlaps"] >= 1
               and x["component"] > x["area"] and x["runner_up"] >= .65 * x["area"]
               for _, x in training)
    assert any(x["aligned_differs"] for _, x in training)
    assert any(s == t and x["rect"] != y["rect"]
               for i, (s, x) in enumerate(training) for t, y in training[i + 1:])
    heights = [shape[0] for shape, _ in training]
    assert (all(h > max(heights) for h in test_heights)
            or all(h < min(heights) for h in test_heights))


def _digest(puzzle):
    content = {s: [{role: np.asarray(p[role]).tolist() for role in ("input", "output")}
                   for p in puzzle[s]] for s in ("train", "test")}
    return hashlib.sha256(json.dumps(content, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """Bounded, streaming checks: 42 host seeds, all legal counts, no file IO."""
    shapes, target_shapes, positions, signatures, digests, palettes = set(), set(), set(), set(), set(), set()
    totals = {"pairs": 0, "carried_only": 0, "bridged": 0, "overlap": 0,
              "long_line": 0, "boundary_width_30": 0, "taller_puzzles": 0,
              "shorter_puzzles": 0, "square": 0, "wide": 0, "tall": 0}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digests.add(_digest(puzzle))
        heights = [p["input"].shape[0] for p in puzzle["train"]]
        totals["taller_puzzles" if puzzle["test"][0]["input"].shape[0] > max(heights)
               else "shorter_puzzles"] += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _features(inp)
                t, l, rh, rw = info["rect"]
                shapes.add(inp.shape)
                target_shapes.add((rh, rw))
                positions.add(info["rect"])
                signatures.add(hashlib.sha256((inp.shape.__repr__()).encode() + inp.tobytes()).hexdigest())
                palettes.add(tuple(int(c) for c in np.unique(inp) if c != 0))
                totals["pairs"] += 1
                totals["carried_only" if info["padding"] == 0 else "bridged"] += 1
                totals["overlap"] += info["overlaps"] > 0
                totals["long_line"] += info["long_line"] > info["area"]
                totals["boundary_width_30"] += out.shape[1] == 30
                totals["square" if rh == rw else "wide" if rw > rh else "tall"] += 1
        assert _digest(generate_puzzle(seed)) == _digest(puzzle)
    assert len(digests) == 42 and len(signatures) == totals["pairs"]
    assert len(shapes) >= 50 and len(target_shapes) >= 25 and len(positions) >= 80
    assert len(palettes) >= 15
    assert totals["carried_only"] >= 42 and totals["bridged"] >= 42
    assert totals["overlap"] >= 42 and totals["long_line"] >= 42
    assert all(totals[k] >= 5 for k in ("square", "wide", "tall", "taller_puzzles", "shorter_puzzles"))
    assert totals["boundary_width_30"] >= 5
    legal = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(0, a, b)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (a, b)
            legal += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Input-only hand examples; no generator metadata participates in solving.
    small = np.array([[1, 0, 0, 0], [1, 1, 0, 0]], dtype=np.int8)
    expected = np.array([[0, 1, 6, 6, 6], [1, 1, 6, 6, 6]], dtype=np.int8)
    assert np.array_equal(solve(small), expected)
    assert np.array_equal(solve(small.tolist()), expected)
    assert np.array_equal(small, [[1, 0, 0, 0], [1, 1, 0, 0]])
    # The input-only rule extends beyond the sampler's height emphasis and
    # does not impose its minimum copied-cell fraction as a solving filter.
    boundary = np.ones((29, 2), dtype=np.int8)
    boundary[13, 0] = 0
    boundary[14] = 0
    result = solve(boundary)
    assert result.shape == (29, 30)
    assert np.array_equal(result == 6, _mask((29, 30), (0, 0, 15, 16)))
    boundary_wide = np.ones((2, 29), dtype=np.int8)
    boundary_wide[0, 1:9] = 0
    boundary_wide[1, 2:10] = 0
    assert np.array_equal(solve(boundary_wide) == 6, _mask((2, 30), (0, 2, 2, 8)))
    # No backgrounds -> symmetric padding winners tie, never choose a side.
    for invalid in (np.ones((6, 8), dtype=int), np.ones((2, 5), dtype=int)):
        try:
            solve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous/absent rectangle accepted")
    return {"seeds": len(HOST_SEEDS), "legal_counts": legal,
            "input_shapes": len(shapes), "target_shapes": len(target_shapes),
            "target_positions_and_shapes": len(positions), "blocker_palettes": len(palettes),
            **totals}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
