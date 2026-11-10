"""Layer odometers: spatially ordered square digits with a visible carry.

Only standard-library modules and NumPy are used.  Labels use base-five
arithmetic; the input-only solver instead follows the carry square by square.
"""

from __future__ import annotations

import json

import numpy as np


STATES = (1, 3, 5, 7, 9)
HOST_SEEDS = (
    0, 928422180, 168158583, 104331477, 931531501, 359237500,
    657844113, 553762383, 58536459, 104252691, 44957385, 641460002,
    629131702, 474388683, 239025485, 549952343, 30608130, 798360572,
    328893555, 800177618, 145667118, 353948724, 633535319, 56688675,
    122337428, 713900890, 660320561, 126152146, 453207297, 899585721,
    861050793, 525172246, 3346755, 414675234, 194421214, 44647478,
    703273460, 397995713, 837136664, 953537633, 938056746, 633604897,
)


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("Counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _array(grid):
    grid = np.asarray(grid)
    if (grid.ndim != 2 or grid.dtype.kind not in "iu"
            or not all(1 <= d <= 30 for d in grid.shape)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("Expected a 1..30 integer color grid")
    return grid


def _separated(boxes, axis):
    # Boxes are half-open (row start, row end, column start, column end).
    k = 0 if axis == 0 else 2
    intervals = sorted((b[k], b[k + 1]) for b in boxes)
    return all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))


def _read_input(input_grid):
    grid = _array(input_grid)
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("Input background must be uniquely most frequent")
    background = int(counts.argmax())
    objects = []
    for color in np.unique(grid):
        color = int(color)
        if color == background:
            continue
        rows, cols = np.where(grid == color)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        side = r1 - r0
        if (side not in STATES or c1 - c0 != side
                or len(rows) != side * side
                or not np.all(grid[r0:r1, c0:c1] == color)):
            raise ValueError("Each foreground color must be one solid odd square")
        objects.append((color, side, (r0, r1, c0, c1)))
    if not 2 <= len(objects) <= 7:
        raise ValueError("Expected 2..7 square digits")
    boxes = [obj[2] for obj in objects]
    axes = [axis for axis in (0, 1) if _separated(boxes, axis)]
    if len(axes) != 1:
        raise ValueError("The ordering axis must be unique")
    axis = axes[0]
    objects.sort(key=lambda obj: obj[2][0 if axis == 0 else 2])
    sizes = [obj[1] for obj in objects]
    if all(side == 1 for side in sizes):
        raise ValueError("A non-singleton must stop the carry")
    return background, axis, objects


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer order from the input, propagate one carry, and pack the result."""
    background, axis, objects = _read_input(input_grid)
    transformed = []
    carry = True
    for color, side, _ in objects:
        if carry:
            if side == 1:
                new_side = 9
            else:
                new_side = side - 2
                carry = False
        else:
            new_side = side
        transformed.append((color, new_side))
    along = sum(side for _, side in transformed)
    across = max(side for _, side in transformed)
    if along > 30:
        raise ValueError("Packed output exceeds ARC dimensions")
    shape = (along, across) if axis == 0 else (across, along)
    output = np.full(shape, background, dtype=np.int8)
    cursor = 0
    for color, side in transformed:
        offset = (across - side) // 2
        if axis == 0:
            output[cursor:cursor + side, offset:offset + side] = color
        else:
            output[offset:offset + side, cursor:cursor + side] = color
        cursor += side
    return output


def _arithmetic_update(sizes):
    """Independent labeling route: add one to a little-endian base-5 word."""
    value = sum(((9 - side) // 2) * 5 ** i for i, side in enumerate(sizes))
    value += 1
    return [9 - 2 * ((value // (5 ** i)) % 5) for i in range(len(sizes))]


def _reference_strip(sizes, colors, background, axis):
    # Tile construction is separate from solve's slice painting.
    height = max(sizes)
    tiles = []
    for side, color in zip(sizes, colors):
        tile = np.full((height, side), background, dtype=np.int8)
        margin = (height - side) // 2
        tile[margin:margin + side, :] = color
        tiles.append(tile)
    horizontal = np.concatenate(tiles, axis=1)
    return horizontal if axis == 1 else horizontal.T.copy()


def _admissible(sizes):
    return (2 <= len(sizes) <= 7 and any(s > 1 for s in sizes)
            and sum(sizes) <= 30 and sum(_arithmetic_update(sizes)) <= 30)


def _random_word(rng, prefix, stop=None, count=None):
    for _ in range(400):
        n = int(rng.integers(max(2, prefix + 1), 8)) if count is None else count
        head = int(rng.choice((3, 5, 7, 9))) if stop is None else int(stop)
        suffix = [int(x) for x in rng.choice(
            STATES, size=n - prefix - 1, p=(0.32, 0.24, 0.19, 0.15, 0.10))]
        sizes = [1] * prefix + [head] + suffix
        if _admissible(sizes):
            return sizes
    raise RuntimeError("Could not sample a bounded square word")


def _core_words(rng):
    # 1->9, 3->1, 9->7, an inactive singleton and inactive large
    # squares are guaranteed even when only two demonstrations are requested.
    first_count = int(rng.integers(3, 6))
    first = _random_word(rng, prefix=1, stop=3, count=first_count)
    second_counts = [n for n in (4, 5, 6) if n != first_count]
    second_count = int(rng.choice(second_counts))
    suffix = [1, 5, 7] + [int(rng.choice((1, 3)))
                            for _ in range(second_count - 4)]
    rng.shuffle(suffix)
    second = [9] + suffix
    assert _admissible(second)
    words = [first, second]
    rng.shuffle(words)
    return words


def _place(rng, sizes, colors, background, axis):
    count = len(sizes)
    # Margins and gaps share the available along-axis budget.  Adjacent
    # differently colored squares are allowed, as are wide gaps.
    spaces = rng.integers(0, 7, size=count + 1).astype(int)
    spaces[0] = min(int(spaces[0]), 5)
    spaces[-1] = min(int(spaces[-1]), 5)
    budget = 30 - sum(sizes)
    while int(spaces.sum()) > budget:
        choices = np.flatnonzero(spaces)
        spaces[int(rng.choice(choices))] -= 1
    starts = [int(spaces[0])]
    for index in range(1, count):
        starts.append(starts[-1] + sizes[index - 1] + int(spaces[index]))
    along = starts[-1] + sizes[-1] + int(spaces[-1])

    if rng.random() < 0.22:
        centers = [max(sizes) // 2] * count
        offsets = [center - side // 2 for center, side in zip(centers, sizes)]
    else:
        offsets = [int(x) for x in rng.integers(0, 12, size=count)]
        intervals = sorted((v, v + s) for v, s in zip(offsets, sizes))
        if all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])):
            i, j = (int(v) for v in rng.choice(count, size=2, replace=False))
            # Overlap one pair to rule out a second complete ordering axis.
            center = max(offsets[i] + sizes[i] // 2, sizes[j] // 2)
            offsets[i] = center - sizes[i] // 2
            offsets[j] = center - sizes[j] // 2
    lead, trail = (int(v) for v in rng.integers(0, 6, size=2))
    offsets = [v + lead for v in offsets]
    across = max(v + s for v, s in zip(offsets, sizes)) + trail
    # Make the background observable without relying on a fixed literal or
    # on an empty border.  This loop adds only empty perpendicular canvas.
    while along * across - sum(s * s for s in sizes) <= max(s * s for s in sizes):
        across += 1
    if across > 30:
        raise RuntimeError("Perpendicular canvas overflow")
    shape = (along, across) if axis == 0 else (across, along)
    grid = np.full(shape, background, dtype=np.int8)
    for side, color, a, b in zip(sizes, colors, starts, offsets):
        row, col = (a, b) if axis == 0 else (b, a)
        grid[row:row + side, col:col + side] = color
    return grid


def _key(grid):
    array = np.asarray(grid).astype(np.int8, copy=False)
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    first_axis = int(rng.integers(0, 2))
    mixed = bool(rng.integers(0, 2))
    axes = [first_axis, 1 - first_axis if mixed else first_axis]
    axes += [int(rng.choice(axes)) for _ in range(num_train - 2)]
    words = _core_words(rng)
    for index in range(2, num_train):
        stop = (5 if index == 2 else 7 if index == 3 else None)
        words.append(_random_word(rng, prefix=int(rng.integers(0, 2)), stop=stop))
    observed_stops = [word[next(i for i, s in enumerate(word) if s != 1)]
                      for word in words]
    seen_words = {tuple(word) for word in words}
    for _ in range(num_test):
        for attempt in range(400):
            prefix = int(rng.choice((2, 3), p=(0.84, 0.16)))
            stop = int(rng.choice(observed_stops))
            # Three wraps leave room only for a 3->1 or 5->3 stop.
            if prefix == 3 and stop > 5:
                continue
            word = _random_word(rng, prefix=prefix, stop=stop)
            if tuple(word) not in seen_words:
                break
        else:
            raise RuntimeError("Could not sample a novel test word")
        seen_words.add(tuple(word))
        words.append(word)
        axes.append(int(rng.choice(axes[:num_train])))

    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for index, (sizes, axis) in enumerate(zip(words, axes)):
        for _ in range(40):
            background = int(rng.integers(0, 10))
            pool = [c for c in range(10) if c != background]
            colors = [int(v) for v in rng.choice(pool, size=len(sizes), replace=False)]
            if index < 2 and (colors == sorted(colors)
                             or colors == sorted(colors, reverse=True)):
                colors[0], colors[1] = colors[1], colors[0]
            grid = _place(rng, sizes, colors, background, axis)
            if _key(grid) not in seen_inputs:
                break
        else:
            raise RuntimeError("Could not construct a distinct input")
        seen_inputs.add(_key(grid))
        output = _reference_strip(_arithmetic_update(sizes), colors, background, axis)
        puzzle["train" if index < num_train else "test"].append(
            {"input": grid, "output": output})
    validate_generated_puzzle(puzzle)
    return puzzle


def _prefix_length(sizes):
    return next(i for i, side in enumerate(sizes) if side != 1)


def validate_generated_puzzle(puzzle):
    """Check labels, carry locality, geometry, and sufficient demonstrations."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, info = set(), {"train": [], "test": []}
    transitions, inactive, train_states = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            bg, axis, objects = _read_input(inp)
            sizes = [obj[1] for obj in objects]
            colors = [obj[0] for obj in objects]
            new_sizes = _arithmetic_update(sizes)
            assert _admissible(sizes)
            oracle = _reference_strip(new_sizes, colors, bg, axis)
            assert np.array_equal(out, oracle)
            assert np.array_equal(solve(inp), out)
            assert not np.array_equal(inp, out)
            stop = _prefix_length(sizes)
            assert all(new_sizes[i] == 9 for i in range(stop))
            assert new_sizes[stop] == sizes[stop] - 2
            assert new_sizes[stop + 1:] == sizes[stop + 1:]
            # Area, color order, output extent and common centerline all
            # follow from the tiles, including background padding holes.
            for color, side in zip(colors, new_sizes):
                assert int(np.count_nonzero(out == color)) == side * side
            info[split].append((axis, tuple(sizes), stop))
            if split == "train":
                train_states.update(sizes)
                transitions.update(zip(sizes[:stop + 1], new_sizes[:stop + 1]))
                inactive.update(sizes[stop + 1:])
    assert train_states == set(STATES)
    assert {(1, 9), (3, 1), (9, 7)} <= transitions
    assert 1 in inactive and any(s > 1 for s in inactive)
    assert len({len(s) for _, s, _ in info["train"]}) >= 2
    train_axes = {axis for axis, _, _ in info["train"]}
    train_words = {sizes for _, sizes, _ in info["train"]}
    demonstrated_stops = {sizes[stop] for _, sizes, stop in info["train"]}
    assert max(stop for _, _, stop in info["train"]) == 1
    color_orders = [[obj[0] for obj in _read_input(pair["input"])[2]]
                    for pair in puzzle["train"]]
    assert any(order != sorted(order) for order in color_orders)
    assert any(order != sorted(order, reverse=True) for order in color_orders)
    for axis, sizes, stop in info["test"]:
        assert axis in train_axes and sizes not in train_words
        assert stop >= 2 and sizes[stop] in demonstrated_stops
    # Direct ablations are deliberately checked on the demonstrations.
    # No A: copy sizes while packing.  No B: update every square independently.
    assert any(_arithmetic_update(sizes) != list(sizes)
               for _, sizes, _ in info["train"])
    assert any(_arithmetic_update(sizes) != [9 if s == 1 else s - 2 for s in sizes]
               for _, sizes, _ in info["train"])


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        _key(x[k]) == _key(y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed checks plus all legal count combinations."""
    shapes, words, palettes, layouts, backgrounds, axis_counts = (set() for _ in range(6))
    gaps, side_spans, test_prefixes, updated_stops = (set() for _ in range(4))
    aligned = edge = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                bg, axis, objects = _read_input(pair["input"])
                sizes = tuple(obj[1] for obj in objects)
                boxes = tuple(obj[2] for obj in objects)
                k, perp = (0, 2) if axis == 0 else (2, 0)
                gaps.update(b[k] - a[k + 1] for a, b in zip(boxes, boxes[1:]))
                centers = [box[perp] + side // 2 for box, side in zip(boxes, sizes)]
                span = max(centers) - min(centers)
                side_spans.add(span)
                aligned += span == 0
                shape = pair["input"].shape
                edge += any(b[0] == 0 or b[2] == 0 or b[1] == shape[0]
                            or b[3] == shape[1] for b in boxes)
                shapes.add((shape, pair["output"].shape))
                words.add((split, sizes))
                palettes.add((bg, tuple(obj[0] for obj in objects)))
                layouts.add((axis, boxes, shape))
                backgrounds.add(bg)
                axis_counts.add((split, axis, len(objects)))
                stop = _prefix_length(sizes)
                if split == "test":
                    test_prefixes.add(stop)
                updated_stops.add(sizes[stop])
    assert len(shapes) >= 100 and len(words) >= 100
    assert len(palettes) >= 150 and len(layouts) >= 150
    assert backgrounds == set(range(10)) and updated_stops == {3, 5, 7, 9}
    assert {axis for _, axis, _ in axis_counts} == {0, 1}
    assert {n for _, _, n in axis_counts} == set(range(2, 8))
    assert test_prefixes == {2, 3}
    assert 0 in gaps and max(gaps) >= 5 and max(side_spans) >= 8
    assert aligned >= 10 and edge >= 10
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1700 + 10 * train + test, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2),
                        (3, 4), (True, 1), (4, False), (4.0, 1),
                        (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Input-only equivariances: recoloring and transposition cannot affect
    # carry behavior, and empty padding cannot alter the compact result.
    pair = generate_puzzle(871)["train"][0]
    permutation = np.array([7, 3, 9, 0, 4, 8, 1, 6, 2, 5], dtype=np.int8)
    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    assert np.array_equal(solve(pair["input"].T), pair["output"].T)
    bg, _, _ = _read_input(pair["input"])
    pad_axis = next(i for i, d in enumerate(pair["input"].shape) if d < 30)
    padding = [(0, 0), (0, 0)]
    padding[pad_axis] = (0, 1)
    padded = np.pad(pair["input"], padding, constant_values=bg)
    assert np.array_equal(solve(padded), pair["output"])
    return {"seeds_checked": 42, "distinct_shape_pairs": len(shapes),
            "distinct_state_words": len(words), "distinct_layouts": len(layouts),
            "backgrounds": sorted(backgrounds), "test_carry_lengths": sorted(test_prefixes),
            "object_counts": sorted({n for _, _, n in axis_counts})}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
