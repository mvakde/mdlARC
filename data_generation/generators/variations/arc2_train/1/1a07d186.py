"""Same-color, same-side projection onto full-span bars."""

import numpy as np


# Fingerprints of the four source inputs (dimensions and all cells).
_SOURCE_INPUTS = {3863038736611838290, 2051759204340902660,
                  7766833808758427366, 18188601351752782652}


def _resolve_counts(rng, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, bool) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= value <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, (num_train or 10) - 1) + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _fingerprint(grid):
    result = 1469598103934665603
    for value in (grid.shape[0], grid.shape[1], *grid.flat):
        result = ((result ^ int(value)) * 1099511628211) & ((1 << 64) - 1)
    return result


def _signature(grid):
    return grid.shape, grid.tobytes()


def _rc(orientation, perpendicular, parallel):
    if orientation == "horizontal":
        return int(perpendicular), int(parallel)
    return int(parallel), int(perpendicular)


def _shape(rng, orientation, count, anchor):
    # Draw axes independently, with an occasional full-size square.
    if int(rng.integers(0, 14)) == 0:
        height = width = 30
    else:
        bands = ((8, 13), (13, 21), (21, 31))
        height = int(rng.integers(*bands[int(rng.integers(0, 3))]))
        width = int(rng.integers(*bands[int(rng.integers(0, 3))]))
    minimum = max(8, 5 * count if anchor else 3 * count + 5)
    if orientation == "horizontal" and height < minimum:
        height = int(rng.integers(minimum, 31))
    if orientation == "vertical" and width < minimum:
        width = int(rng.integers(minimum, 31))
    return height, width


def _bar_positions(rng, length, count, anchor):
    first = 2 if anchor else 0
    last = length - 3 if anchor else length - 1
    gap = 5 if anchor else 2
    available = last - first + 1 - (gap - 1) * (count - 1)
    if available < count:
        raise RuntimeError("not enough space for bars")
    base = np.sort(rng.choice(available, size=count, replace=False))
    return tuple(int(first + v + (gap - 1) * i) for i, v in enumerate(base))


def _put_marker(rng, orientation, shape, bars, color, destination,
                side, kind, markers, occupied, targets):
    length = shape[0] if orientation == "horizontal" else shape[1]
    parallel_length = shape[1] if orientation == "horizontal" else shape[0]
    positions = tuple(p for p, _ in bars)
    possible = []
    for p in range(length):
        if any(abs(p - line) <= 1 for line in positions):
            continue
        if destination is not None:
            if side * (p - destination) <= 1:
                continue
            crossed = sum(min(p, destination) < line < max(p, destination)
                          for line in positions if line != destination)
            if kind == "cross" and crossed == 0:
                continue
            if kind == "gap" and not any(a < p < b for a, b in zip(positions, positions[1:])):
                continue
            if kind == "near" and abs(p - destination) > 3:
                continue
            if kind == "far" and abs(p - destination) < 5:
                continue
        possible.append(p)
    if not possible:
        raise RuntimeError("no marker row or column")
    for index in rng.permutation(len(possible) * parallel_length):
        p = possible[int(index) // parallel_length]
        parallel = int(index) % parallel_length
        cell = _rc(orientation, p, parallel)
        target = None if destination is None else _rc(orientation, destination + side, parallel)
        if cell in occupied or cell in targets or target in occupied or target in targets:
            continue
        if any(abs(cell[0] - old[0]) + abs(cell[1] - old[1]) <= 1 for old in occupied):
            continue
        occupied.add(cell)
        if target is not None:
            targets.add(target)
        markers.append((cell, int(color), target))
        return
    raise RuntimeError("marker positions conflict")


def _make_pair(rng, regime, orientation, count, anchor=False,
               forced_distractor=None, minimum_matching=0):
    assert regime in ("source_train", "target_test")
    assert not anchor or regime == "source_train"
    for _ in range(80):
        try:
            backgrounds = [c for c in range(10) if c != forced_distractor]
            background = int(rng.choice(backgrounds))
            palette = [c for c in range(10) if c != background and c != forced_distractor]
            colors = [int(c) for c in rng.choice(palette, size=count, replace=False)]
            shape = _shape(rng, orientation, count, anchor)
            length = shape[0] if orientation == "horizontal" else shape[1]
            positions = _bar_positions(rng, length, count, anchor)
            bars = tuple(zip(positions, colors))
            markers, occupied, targets = [], set(), set()

            if anchor:
                # Demonstrate two identities, both sides, an interior origin,
                # and a marker that passes another bar.
                for index, side, kind in ((0, -1, "any"),
                                          (0, 1, "cross"),
                                          (1, -1, "gap")):
                    _put_marker(rng, orientation, shape, bars, colors[index],
                                positions[index], side, kind,
                                markers, occupied, targets)

            # Independent counts on each side permit one-sided and untouched bars.
            for position, color in bars:
                for side in (-1, 1):
                    amount = int(rng.choice(4, p=(0.39, 0.34, 0.19, 0.08)))
                    for _ in range(amount):
                        kind = str(rng.choice(("any", "gap", "cross", "near", "far"),
                                              p=(0.25, 0.23, 0.19, 0.19, 0.14)))
                        try:
                            _put_marker(rng, orientation, shape, bars, color,
                                        position, side, kind,
                                        markers, occupied, targets)
                        except RuntimeError:
                            try:
                                _put_marker(rng, orientation, shape, bars, color,
                                            position, side, "any",
                                            markers, occupied, targets)
                            except RuntimeError:
                                break

            while sum(target is not None for _, _, target in markers) < max(1, minimum_matching):
                if len(markers) >= 25:
                    raise RuntimeError("too many markers")
                placed = False
                for index in rng.permutation(count):
                    index = int(index)
                    for side in rng.permutation((-1, 1)):
                        try:
                            _put_marker(rng, orientation, shape, bars, colors[index],
                                        positions[index], int(side), "any",
                                        markers, occupied, targets)
                            placed = True
                            break
                        except RuntimeError:
                            pass
                    if placed:
                        break
                if not placed:
                    raise RuntimeError("insufficient marker capacity")

            distractor_count = int(rng.integers(0, 5))
            if anchor or forced_distractor is not None:
                distractor_count = max(1, distractor_count)
            unmatched = [c for c in range(10) if c != background and c not in colors]
            for index in range(distractor_count):
                color = (forced_distractor if index == 0 and forced_distractor is not None
                         else int(rng.choice(unmatched)))
                _put_marker(rng, orientation, shape, bars, color, None, 0,
                            "any", markers, occupied, targets)

            inp = np.full(shape, background, dtype=np.int8)
            out = np.full(shape, background, dtype=np.int8)
            for position, color in bars:
                if orientation == "horizontal":
                    inp[position, :] = color
                    out[position, :] = color
                else:
                    inp[:, position] = color
                    out[:, position] = color
            for cell, color, target in markers:
                inp[cell] = color
                if target is not None:
                    out[target] = color
            return inp, out
        except RuntimeError:
            continue
    raise ValueError("unable to construct a pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    # The projection rule is the sole puzzle-level convention. These plans
    # vary presentation; output-affecting roles remain visible in each input.
    orientation_plan = str(rng.choice(("uniform", "transfer", "mixed", "mixed_alt"),
                                      p=(0.4, 0.2, 0.2, 0.2)))
    first_orientation = "horizontal" if int(rng.integers(0, 2)) == 0 else "vertical"
    other_orientation = "vertical" if first_orientation == "horizontal" else "horizontal"
    anchor_index = int(rng.integers(0, num_train))
    switch_index = (anchor_index + int(rng.integers(1, num_train))) % num_train
    shift = int(rng.integers(0, 3))  # larger, mixed, smaller
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max,
                              size=num_train + num_test, dtype=np.uint64)
    puzzle = {"train": [None] * num_train, "test": []}
    used = set()

    def orientation(index, test):
        if orientation_plan == "uniform":
            return first_orientation
        if orientation_plan == "transfer":
            return other_orientation if test else first_orientation
        if orientation_plan == "mixed":
            return other_orientation if index % 2 else first_orientation
        return first_orientation if test else (other_orientation if index % 2 else first_orientation)

    def add_pair(index, test, count, anchor=False, forced=None, minimum=0):
        pair_rng = np.random.default_rng(pair_seeds[index + (num_train if test else 0)])
        for _ in range(30):
            inp, out = _make_pair(pair_rng,
                                  "target_test" if test else "source_train",
                                  orientation(index, test), count,
                                  anchor, forced, minimum)
            key = _signature(inp)
            if key not in used and _fingerprint(inp) not in _SOURCE_INPUTS:
                used.add(key)
                pair = {"input": inp, "output": out}
                if test:
                    puzzle["test"].append(pair)
                else:
                    puzzle["train"][index] = pair
                return
        raise ValueError("unable to construct a distinct pair")

    anchor_rng = np.random.default_rng(pair_seeds[anchor_index])
    anchor_count = int(anchor_rng.choice((2, 3, 4, 5), p=(0.48, 0.30, 0.17, 0.05)))
    add_pair(anchor_index, False, anchor_count, anchor=True)
    anchor_pair = puzzle["train"][anchor_index]
    anchor_analysis = _analyze_pair(anchor_pair["input"], anchor_pair["output"])
    switch_color = int(anchor_rng.choice(anchor_analysis["bar_colors"]))

    for index in range(num_train):
        if index == anchor_index:
            continue
        local_rng = np.random.default_rng(pair_seeds[index])
        count = int(local_rng.choice((1, 2, 3, 4, 5), p=(0.24, 0.32, 0.23, 0.15, 0.06)))
        add_pair(index, False, count,
                 forced=switch_color if index == switch_index else None)

    train_analyses = [_analyze_pair(pair["input"], pair["output"])
                      for pair in puzzle["train"]]
    max_bars = max(len(a["bar_colors"]) for a in train_analyses)
    max_matching = max(a["matching_count"] for a in train_analyses)
    for index in range(num_test):
        local_rng = np.random.default_rng(pair_seeds[num_train + index])
        if shift == 0:
            count = int(local_rng.integers(max_bars + 1, 6)) if max_bars < 5 else 5
            minimum = min(19, max_matching + 1)
        elif shift == 2:
            count = int(local_rng.integers(1, max_bars))
            minimum = 1
        else:
            count = int(local_rng.integers(1, 6))
            minimum = 1
        add_pair(index, True, count, minimum=minimum)

    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(inp, out):
    values, counts = np.unique(inp, return_counts=True)
    assert sum(counts == max(counts)) == 1
    background = int(values[int(np.argmax(counts))])
    rows = [(r, int(inp[r, 0])) for r in range(inp.shape[0])
            if inp[r, 0] != background and np.all(inp[r] == inp[r, 0])]
    cols = [(c, int(inp[0, c])) for c in range(inp.shape[1])
            if inp[0, c] != background and np.all(inp[:, c] == inp[0, c])]
    assert bool(rows) != bool(cols)
    orientation = "horizontal" if rows else "vertical"
    bars = rows if rows else cols
    positions = tuple(p for p, _ in bars)
    colors = tuple(c for _, c in bars)
    assert len(colors) == len(set(colors))
    bar_mask = np.zeros(inp.shape, dtype=bool)
    expected = np.full(inp.shape, background, dtype=inp.dtype)
    for position, color in bars:
        if orientation == "horizontal":
            bar_mask[position] = True
            expected[position] = color
        else:
            bar_mask[:, position] = True
            expected[:, position] = color
    markers = [tuple(map(int, cell)) for cell in np.argwhere((inp != background) & ~bar_mask)]
    assert all(all(abs((r if orientation == "horizontal" else c) - p) > 1
                   for p in positions) for r, c in markers)
    assert all(abs(r - rr) + abs(c - cc) > 1
               for i, (r, c) in enumerate(markers) for rr, cc in markers[:i])
    by_color = dict(zip(colors, positions))
    sides = {color: set() for color in colors}
    marker_counts = {color: {-1: 0, 1: 0} for color in colors}
    targets = set()
    matching = distractors = crossing = gap = 0
    crossing_depths, move_distances = [], []
    distractor_colors = set()
    for row, col in markers:
        color = int(inp[row, col])
        if color not in by_color:
            distractors += 1
            distractor_colors.add(color)
            continue
        matching += 1
        position = by_color[color]
        p, parallel = (row, col) if orientation == "horizontal" else (col, row)
        side = -1 if p < position else 1
        sides[color].add(side)
        marker_counts[color][side] += 1
        target = _rc(orientation, position + side, parallel)
        assert 0 <= target[0] < inp.shape[0] and 0 <= target[1] < inp.shape[1]
        assert not bar_mask[target] and target not in targets
        targets.add(target)
        expected[target] = color
        crossed = sum(min(p, position) < q < max(p, position)
                      for q in positions if q != position)
        crossing_depths.append(crossed)
        move_distances.append(abs(p - position))
        crossing += crossed > 0
        gap += any(a < p < b for a, b in zip(positions, positions[1:]))
    assert matching >= 1 and np.array_equal(expected, out)
    assert not np.array_equal(inp, out)
    return {"background": background, "orientation": orientation,
            "bar_colors": colors, "line_positions": positions,
            "matching_count": matching, "distractor_count": distractors,
            "distractor_colors": distractor_colors, "crossing_count": crossing,
            "gap_count": gap, "sides": sides, "marker_counts": marker_counts,
            "crossing_depths": crossing_depths, "move_distances": move_distances}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            assert _fingerprint(inp) not in _SOURCE_INPUTS
            ikey, okey = _signature(inp), _signature(out)
            assert ikey not in inputs and (ikey, okey) not in pairs
            inputs.add(ikey)
            pairs.add((ikey, okey))
            analyses[split].append(_analyze_pair(inp, out))
    train = analyses["train"]
    assert any(len(a["bar_colors"]) >= 2 for a in train)
    assert any(a["crossing_count"] > 0 for a in train)
    assert any(a["gap_count"] > 0 for a in train)
    assert sum(a["distractor_count"] for a in train) > 0
    assert {-1, 1} <= set().union(*(set().union(*a["sides"].values()) for a in train))
    assert any(set(a["bar_colors"]) & b["distractor_colors"]
               for a in train for b in train if a is not b)


def _equal_puzzles(a, b):
    return all(np.array_equal(x[k], y[k])
               for split in ("train", "test")
               for x, y in zip(a[split], b[split]) for k in ("input", "output"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _equal_puzzles(generate_puzzle(701, 6, 3), generate_puzzle(701, 6, 3))
    assert not _equal_puzzles(generate_puzzle(701, 6, 3), generate_puzzle(702, 6, 3))
    for count in ((2, 1), (10, 1), (10, 4), (None, 1),
                  (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(8100 + (count[0] or 0) + (count[1] or 0), *count))
    for count in ((1, None), (11, None), (None, 0), (None, 5),
                  (2, 2), (3, 4)):
        try:
            generate_puzzle(10001, *count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    all_pairs = [(split, pair, _analyze_pair(pair["input"], pair["output"]))
                 for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split]]
    assert len({a["background"] for _, _, a in all_pairs}) == 10
    assert len({_signature(pair["input"]) for _, pair, _ in all_pairs}) > 170
    assert {a["orientation"] for _, _, a in all_pairs} == {"horizontal", "vertical"}
    assert {len(a["bar_colors"]) for _, _, a in all_pairs} >= {1, 2, 3, 4, 5}
    assert any(a["distractor_count"] == 0 for _, _, a in all_pairs)
    assert any(len(a["distractor_colors"]) >= 2 for _, _, a in all_pairs)
    assert any(any(sum(sides.values()) == 0 for sides in a["marker_counts"].values())
               for _, _, a in all_pairs)
    assert any(any(0 in sides.values() and sum(sides.values()) > 0
                   for sides in a["marker_counts"].values()) for _, _, a in all_pairs)
    assert any(pair["input"].shape == (30, 30) for _, pair, _ in all_pairs)
    assert any(min(pair["input"].shape) <= 10 for _, pair, _ in all_pairs)
    assert any(pair["input"].shape[0] > pair["input"].shape[1] + 8 for _, pair, _ in all_pairs)
    assert any(pair["input"].shape[1] > pair["input"].shape[0] + 8 for _, pair, _ in all_pairs)
    assert {min(depth, 2) for _, _, a in all_pairs
            for depth in a["crossing_depths"]} == {0, 1, 2}
    assert any(distance <= 3 for _, _, a in all_pairs for distance in a["move_distances"])
    assert any(distance >= 5 for _, _, a in all_pairs for distance in a["move_distances"])
    assert any(any(b - c < 5 for c, b in zip(a["line_positions"], a["line_positions"][1:]))
               for _, _, a in all_pairs)
    assert any(0 in a["line_positions"] or
               (pair["input"].shape[0 if a["orientation"] == "horizontal" else 1] - 1)
               in a["line_positions"] for _, pair, a in all_pairs)
    assert any(pair["input"].shape in {(15, 14), (18, 19), (15, 16), (19, 26)}
               for _, pair, _ in all_pairs)
    analyses = []
    for puzzle in puzzles:
        train = [_analyze_pair(pair["input"], pair["output"]) for pair in puzzle["train"]]
        test = [_analyze_pair(pair["input"], pair["output"]) for pair in puzzle["test"]]
        analyses.append((train, test))
    assert all(len({a["background"] for a in train + test}) > 1
               for train, test in analyses)
    assert any(len({a["orientation"] for a in train + test}) == 1
               for train, test in analyses)
    assert any({a["orientation"] for a in train} == {"horizontal"} and
               {a["orientation"] for a in test} == {"vertical"}
               for train, test in analyses)
    assert any({a["orientation"] for a in train} == {"vertical"} and
               {a["orientation"] for a in test} == {"horizontal"}
               for train, test in analyses)
    assert any(len({a["orientation"] for a in train}) > 1
               for train, _ in analyses)
    assert any(set().union(*(set(a["bar_colors"]) for a in test)) -
               set().union(*(set(a["bar_colors"]) for a in train))
               for train, test in analyses)
    assert any(set().union(*(set(a["bar_colors"]) for a in train)) -
               set().union(*(set(a["bar_colors"]) for a in test))
               for train, test in analyses)
    assert any(any(len(t["bar_colors"]) > max(len(a["bar_colors"]) for a in train) and
                   t["matching_count"] > max(a["matching_count"] for a in train)
                   for t in test) for train, test in analyses)
    assert any(any(len(t["bar_colors"]) < max(len(a["bar_colors"]) for a in train) and
                   t["matching_count"] < max(a["matching_count"] for a in train)
                   for t in test) for train, test in analyses)
    assert any(any(len(t["bar_colors"]) == len(a["bar_colors"]) and
                   abs(t["matching_count"] - a["matching_count"]) <= 1
                   for t in test for a in train) for train, test in analyses)
