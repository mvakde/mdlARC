"""Synthetic puzzles for lane-wise marker compaction against a full-span stripe."""

import hashlib
import numpy as np


_SOURCE_PAIR_HASHES = {
    "8cd247f21e680e49a1f3fabb72ba09c903245bed2c983387da52ec6a52aea817",
    "24915aeeb3e7b5ee969dc9051f4c1c23e22a8f5f78d7bca9251d502b1c68b204",
    "a11b6a11e417badab36a6cb7b6b93f46f557ce7bfce27ef81e9a40205162786a",
    "26d931d409044b5b144fd31fdd4242a258cf0a2a653b029e1ffd51e4129a2090",
}


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} out of range")
    return value


def _counts(seed, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _parse(grid):
    height, width = grid.shape
    colors = [int(x) for x in np.unique(grid)]
    assert len(colors) == 3
    candidates = []
    for color in colors:
        mask = grid == color
        for orientation, axis in (("vertical", 0), ("horizontal", 1)):
            full = np.flatnonzero(np.all(mask, axis=axis))
            if not len(full):
                continue
            start, end = int(full[0]), int(full[-1])
            if not np.array_equal(full, np.arange(start, end + 1)):
                continue
            area = height * len(full) if orientation == "vertical" else width * len(full)
            if int(np.count_nonzero(mask)) == area:
                candidates.append((orientation, color, start, end))
    assert len(candidates) == 1
    orientation, stripe, start, end = candidates[0]
    other = [c for c in colors if c != stripe]
    sizes = [int(np.count_nonzero(grid == c)) for c in other]
    assert sizes[0] != sizes[1]
    background, marker = (other[0], other[1]) if sizes[0] > sizes[1] else (other[1], other[0])
    capacity = (start, (width if orientation == "vertical" else height) - end - 1)
    assert min(capacity) >= 1
    return {
        "orientation": orientation, "stripe": stripe, "background": background,
        "marker": marker, "start": start, "end": end, "thickness": end - start + 1,
        "capacity": capacity, "lanes": height if orientation == "vertical" else width,
    }


def _lane_data(grid, info):
    """Return marker distances from the stripe for each lane and side."""
    result = []
    start, end = info["start"], info["end"]
    for lane in range(info["lanes"]):
        if info["orientation"] == "vertical":
            lower = np.flatnonzero(grid[lane, :start] == info["marker"])
            upper = np.flatnonzero(grid[lane, end + 1:] == info["marker"]) + end + 1
        else:
            lower = np.flatnonzero(grid[:start, lane] == info["marker"])
            upper = np.flatnonzero(grid[end + 1:, lane] == info["marker"]) + end + 1
        result.append((tuple(start - int(x) for x in lower),
                       tuple(int(x) - end for x in upper)))
    return result


def _solve(grid):
    info = _parse(grid)
    output = np.full(grid.shape, info["background"], dtype=np.int8)
    start, end = info["start"], info["end"]
    if info["orientation"] == "vertical":
        output[:, start:end + 1] = info["stripe"]
    else:
        output[start:end + 1, :] = info["stripe"]
    for lane, sides in enumerate(_lane_data(grid, info)):
        low, high = len(sides[0]), len(sides[1])
        if info["orientation"] == "vertical":
            output[lane, start - low:start] = info["stripe"]
            output[lane, end + 1:end + 1 + high] = info["stripe"]
        else:
            output[start - low:start, lane] = info["stripe"]
            output[end + 1:end + 1 + high, lane] = info["stripe"]
    return output


def _dimension(rng):
    band = int(rng.integers(0, 4))
    if band == 0:
        return int(rng.integers(5, 11))
    if band == 1:
        return int(rng.integers(11, 19))
    if band == 2:
        return int(rng.integers(19, 26))
    return int(rng.integers(26, 31))


def _shape(rng):
    if int(rng.integers(0, 12)) == 0:
        return 14, 14
    return _dimension(rng), _dimension(rng)


def _geometry(rng, shape, orientation, mode):
    axis = shape[1] if orientation == "vertical" else shape[0]
    if mode == "rich":
        # Leave one side at least three cells wide for a visible count-two move.
        thickness = int(rng.integers(1, min(8, axis - 4) + 1))
        low = int(rng.integers(1, axis - thickness - 2))
    else:
        thickness = int(rng.integers(1, min(8, axis - 2) + 1))
        low = int(rng.integers(1, axis - thickness))
    high = axis - thickness - low
    assert low >= 1 and high >= 1
    return low, low + thickness - 1


def _quantities(rng, lanes, capacities, mode, regime):
    counts = np.zeros((lanes, 2), dtype=np.int16)
    available = lanes * sum(capacities)
    budget = max(1, (available - 1) // 2)
    if mode == "simple":
        active = int(rng.integers(1, min(lanes, 5) + 1))
    elif mode == "sparse":
        active = int(rng.integers(1, min(lanes, 4) + 1))
    elif mode == "dense":
        active = int(rng.integers(max(2, lanes // 3), max(3, 3 * lanes // 4) + 1))
    else:
        active = int(rng.integers(2, max(3, lanes // 2) + 1))
    active = min(active, lanes)
    chosen = [int(x) for x in rng.choice(lanes, size=active, replace=False)]
    favored = int(rng.integers(0, 2))
    for lane in chosen:
        if mode == "one_side":
            sides = [favored]
        elif mode == "simple":
            sides = [int(rng.integers(0, 2))]
            if int(rng.integers(0, 4)) == 0:
                sides.append(1 - sides[0])
        else:
            sides = [0, 1] if int(rng.integers(0, 3)) == 0 else [int(rng.integers(0, 2))]
        for side in sides:
            cap = capacities[side]
            if mode == "simple":
                quantity = 1
            elif mode == "sparse":
                quantity = int(rng.integers(1, min(cap, 2) + 1))
            elif mode == "dense":
                quantity = int(rng.integers(1, min(cap, 8) + 1))
            else:
                quantity = int(rng.integers(1, min(cap, 6) + 1))
            if mode in {"dense", "rich"} and int(rng.integers(0, 5)) == 0:
                quantity = cap
            counts[lane, side] = quantity
    if mode == "rich":
        lane = chosen[0]
        long_side = 0 if capacities[0] >= 3 else 1
        counts[lane, long_side] = 2
        counts[lane, 1 - long_side] = 1
    # The background must remain identifiable even in compact, saturated grids.
    while int(counts.sum()) > budget:
        eligible = np.argwhere(counts > 0)
        pick = eligible[int(rng.integers(0, len(eligible)))]
        if mode == "rich" and int(pick[0]) == chosen[0]:
            other = [x for x in eligible if int(x[0]) != chosen[0]]
            if not other:
                break
            pick = other[int(rng.integers(0, len(other)))]
        counts[int(pick[0]), int(pick[1])] -= 1
    assert int(counts.sum()) < available - int(counts.sum())
    return counts


def _positions(rng, capacity, count, style):
    if count == capacity:
        return np.arange(1, capacity + 1)
    if style == "packed":
        return np.arange(1, count + 1)
    if style == "near":
        pool = np.arange(1, min(capacity, count + 3) + 1)
    elif style == "far":
        pool = np.arange(max(1, capacity - count - 2), capacity + 1)
    elif style == "interior" and capacity >= count + 2:
        pool = np.arange(2, capacity)
    else:
        pool = np.arange(1, capacity + 1)
    if len(pool) < count:
        pool = np.arange(1, capacity + 1)
    if style == "run":
        first = int(rng.integers(1, capacity - count + 2))
        return np.arange(first, first + count)
    if style == "edge":
        others = rng.choice(np.arange(1, capacity), size=count - 1, replace=False)
        return np.append(others, capacity)
    return np.asarray(rng.choice(pool, size=count, replace=False))


def _make_pair(rng, shape, orientation, colors, mode, regime):
    start, end = _geometry(rng, shape, orientation, mode)
    capacities = (start, (shape[1] if orientation == "vertical" else shape[0]) - end - 1)
    lanes = shape[0] if orientation == "vertical" else shape[1]
    counts = _quantities(rng, lanes, capacities, mode, regime)
    background, stripe, marker = colors
    grid = np.full(shape, background, dtype=np.int8)
    if orientation == "vertical":
        grid[:, start:end + 1] = stripe
    else:
        grid[start:end + 1, :] = stripe
    styles = ("packed", "near", "far", "interior", "run", "edge", "random")
    pair_style = "packed" if mode == "packed" else styles[int(rng.integers(0, len(styles)))]
    for lane in range(lanes):
        for side in range(2):
            quantity = int(counts[lane, side])
            if not quantity:
                continue
            style = pair_style if mode == "packed" or int(rng.integers(0, 3)) else styles[int(rng.integers(0, len(styles)))]
            distances = _positions(rng, capacities[side], quantity, style)
            for distance in distances:
                coordinate = start - int(distance) if side == 0 else end + int(distance)
                if orientation == "vertical":
                    grid[lane, coordinate] = marker
                else:
                    grid[coordinate, lane] = marker
    if mode == "rich":
        # Guarantee that at least one marker actually moves, even at low counts.
        side = 0 if capacities[0] >= 3 else 1
        lane = next(l for l in range(lanes)
                    if counts[l, side] == 2 and counts[l, 1 - side] >= 1)
        if counts[lane, side] >= 2 and capacities[side] >= 3:
            if orientation == "vertical":
                region = grid[lane, :start] if side == 0 else grid[lane, end + 1:]
            else:
                region = grid[:start, lane] if side == 0 else grid[end + 1:, lane]
            region[region == marker] = background
            for distance in (2, 3):
                coordinate = start - distance if side == 0 else end + distance
                if orientation == "vertical":
                    grid[lane, coordinate] = marker
                else:
                    grid[coordinate, lane] = marker
    output = _solve(grid)
    return {"input": grid, "output": output}


def _colors(rng, fixed, train_markers, test_style):
    if fixed is not None:
        return fixed
    if test_style == "familiar" and train_markers:
        marker = int(rng.choice(sorted(train_markers)))
        rest = [c for c in range(10) if c != marker]
        chosen = rng.choice(rest, size=2, replace=False)
        return int(chosen[0]), int(chosen[1]), marker
    if test_style == "new" and train_markers:
        choices = [c for c in range(10) if c not in train_markers]
        marker = int(rng.choice(choices))
        rest = [c for c in range(10) if c != marker]
        chosen = rng.choice(rest, size=2, replace=False)
        return int(chosen[0]), int(chosen[1]), marker
    chosen = rng.choice(10, size=3, replace=False)
    return tuple(int(c) for c in chosen)


def _distance_twin(pair):
    """Keep every lane count, but repack one marker group in the input."""
    grid = pair["input"].copy()
    info = _parse(grid)
    for lane, sides in enumerate(_lane_data(grid, info)):
        for side, distances in enumerate(sides):
            count = len(distances)
            if not count or count >= info["capacity"][side]:
                continue
            if tuple(sorted(distances)) == tuple(range(1, count + 1)):
                continue
            if info["orientation"] == "vertical":
                region = grid[lane, :info["start"]] if side == 0 else grid[lane, info["end"] + 1:]
            else:
                region = grid[:info["start"], lane] if side == 0 else grid[info["end"] + 1:, lane]
            region[region == info["marker"]] = info["background"]
            for distance in range(1, count + 1):
                coordinate = info["start"] - distance if side == 0 else info["end"] + distance
                if info["orientation"] == "vertical":
                    grid[lane, coordinate] = info["marker"]
                else:
                    grid[coordinate, lane] = info["marker"]
            return {"input": grid, "output": _solve(grid)}
    raise AssertionError("rich pair had no movable marker group")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed_key = int(seed) % (1 << 64)
    train_count, test_count = _counts(seed_key, num_train, num_test)
    rng = np.random.default_rng(seed_key)
    orientation_mode = ("mixed", "vertical", "horizontal", "v_to_h", "h_to_v")[int(rng.integers(0, 5))]
    fixed_shape = _shape(rng) if int(rng.integers(0, 4)) == 0 else None
    if fixed_shape is not None:
        fixed_shape = (max(6, fixed_shape[0]), max(6, fixed_shape[1]))
    fixed_colors = _colors(rng, None, set(), "random") if int(rng.integers(0, 4)) == 0 else None
    if orientation_mode == "mixed":
        training_orientations = ["vertical", "horizontal"] + [
            ("vertical", "horizontal")[int(rng.integers(0, 2))]
            for _ in range(train_count - 2)
        ]
        rng.shuffle(training_orientations)
        testing_orientations = [
            ("vertical", "horizontal")[int(rng.integers(0, 2))]
            for _ in range(test_count)
        ]
    else:
        training_direction = "vertical" if orientation_mode in {"vertical", "v_to_h"} else "horizontal"
        testing_direction = (
            "horizontal" if orientation_mode == "v_to_h" else
            "vertical" if orientation_mode == "h_to_v" else training_direction
        )
        training_orientations = [training_direction] * train_count
        testing_orientations = [testing_direction] * test_count
    training_modes = ["simple", "rich"] + [
        ("simple", "sparse", "moderate", "one_side", "dense", "rich", "packed")[
            int(rng.integers(0, 7))
        ] for _ in range(train_count - 2)
    ]
    rng.shuffle(training_modes)
    testing_modes = [
        ("sparse", "moderate", "one_side", "dense", "rich", "packed")[
            int(rng.integers(0, 6))
        ] for _ in range(test_count)
    ]
    puzzle = {"train": [], "test": []}
    seen = set()
    train_markers = set()
    for split, orientations, modes in (
        ("train", training_orientations, training_modes),
        ("test", testing_orientations, testing_modes),
    ):
        for orientation, mode in zip(orientations, modes):
            for _ in range(40):
                pair_rng = np.random.default_rng(int(rng.integers(0, 1 << 63)))
                shape = fixed_shape if fixed_shape is not None else _shape(pair_rng)
                if mode == "rich":
                    axis = shape[1] if orientation == "vertical" else shape[0]
                    if axis < 6:
                        continue
                style = ("random", "familiar", "new")[int(pair_rng.integers(0, 3))] if split == "test" else "random"
                colors = _colors(pair_rng, fixed_colors, train_markers, style)
                try:
                    pair = _make_pair(pair_rng, shape, orientation, colors, mode, split)
                    info = _parse(pair["input"])
                except (AssertionError, ValueError):
                    continue
                key = _key(pair["input"])
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train":
                    train_markers.add(info["marker"])
                break
            else:
                raise RuntimeError("bounded pair construction failed")
    if train_count >= 3 and int(rng.integers(0, 3)) == 0:
        rich_index = training_modes.index("rich")
        replace_index = next(index for index, mode in enumerate(training_modes)
                             if index != rich_index and
                             (mode != "simple" or training_modes.count("simple") >= 2))
        twin = _distance_twin(puzzle["train"][rich_index])
        existing_inputs = {
            _key(pair["input"]) for split in ("train", "test")
            for pair in puzzle[split]
        }
        if _key(twin["input"]) not in existing_inputs:
            assert np.array_equal(twin["output"], puzzle["train"][rich_index]["output"])
            puzzle["train"][replace_index] = twin
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    training_features = {"both": False, "multiple": False, "moved": False, "simple": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            before, after = pair["input"], pair["output"]
            for grid in (before, after):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            assert before.shape == after.shape and not np.array_equal(before, after)
            pair_key = (_key(before), _key(after))
            assert pair_key[0] not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(pair_key[0])
            seen_pairs.add(pair_key)
            if before.shape == (14, 14):
                digest = hashlib.sha256(
                    before.astype(np.int8).tobytes() + after.astype(np.int8).tobytes()
                ).hexdigest()
                assert digest not in _SOURCE_PAIR_HASHES
            info = _parse(before)
            lanes = _lane_data(before, info)
            counts = np.array([[len(side) for side in lane] for lane in lanes])
            assert int(counts.sum()) > 0
            assert all(counts[:, side].max() <= info["capacity"][side] for side in range(2))
            assert np.array_equal(after, _solve(before))
            assert set(int(x) for x in np.unique(after)) == {info["background"], info["stripe"]}
            # Compare each output lane with its own input counts; no generated metadata is used.
            for lane, (low, high) in enumerate(lanes):
                for side, distances in enumerate((low, high)):
                    length = len(distances)
                    if info["orientation"] == "vertical":
                        segment = after[lane, :info["start"]] if side == 0 else after[lane, info["end"] + 1:]
                    else:
                        segment = after[:info["start"], lane] if side == 0 else after[info["end"] + 1:, lane]
                    expected = np.full(len(segment), info["background"], dtype=np.int8)
                    if side == 0:
                        expected[len(segment) - length:] = info["stripe"]
                    else:
                        expected[:length] = info["stripe"]
                    assert np.array_equal(segment, expected)
            if split == "train":
                training_features["both"] |= bool(np.any(np.all(counts > 0, axis=1)))
                training_features["multiple"] |= bool(np.any(counts >= 2))
                training_features["moved"] |= any(
                    tuple(sorted(side)) != tuple(range(1, len(side) + 1))
                    for lane in lanes for side in lane if side
                )
                training_features["simple"] |= bool(np.max(counts) <= 1)
    assert all(training_features.values())


def _fingerprint(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, thicknesses, capacities, totals, maximums = set(), set(), set(), set(), set()
    orientation_profiles, palette_profiles = set(), set()
    split_thin = {"train": 0, "test": 0}
    split_thick = {"train": 0, "test": 0}
    split_horizontal = {"train": 0, "test": 0}
    familiar_tests = new_tests = source_sized = edge_sized = 0
    distance_twin_puzzles = 0
    simple_pairs = one_side_pairs = edge_free_pairs = saturated_pairs = run_pairs = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_info = [_parse(p["input"]) for p in puzzle["train"]]
        test_info = [_parse(p["input"]) for p in puzzle["test"]]
        distance_twin_puzzles += int(any(
            _key(a["output"]) == _key(b["output"]) and _key(a["input"]) != _key(b["input"])
            for index, a in enumerate(puzzle["train"])
            for b in puzzle["train"][index + 1:]
        ))
        orientation_profiles.add((
            tuple(sorted(set(i["orientation"] for i in train_info))),
            tuple(sorted(set(i["orientation"] for i in test_info))),
        ))
        palette_profiles.add(len({(i["background"], i["stripe"]) for i in train_info + test_info}))
        known_markers = {i["marker"] for i in train_info}
        familiar_tests += sum(i["marker"] in known_markers for i in test_info)
        new_tests += sum(i["marker"] not in known_markers for i in test_info)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before, after = pair["input"], pair["output"]
                info = _parse(before)
                data = _lane_data(before, info)
                count = np.array([[len(s) for s in lane] for lane in data])
                assert np.array_equal(after, _solve(before))
                shapes.add(before.shape)
                thicknesses.add(info["thickness"])
                split_thin[split] += int(info["thickness"] <= 2)
                split_thick[split] += int(info["thickness"] >= 3)
                split_horizontal[split] += int(info["orientation"] == "horizontal")
                source_sized += int(before.shape == (14, 14))
                edge_sized += int(30 in before.shape)
                capacities.add(info["capacity"])
                totals.add(int(count.sum()))
                maximums.add(int(count.max()))
                simple_pairs += int(count.max() <= 1)
                one_side_pairs += int(bool(np.any(count[:, 0])) != bool(np.any(count[:, 1])))
                edge_free_pairs += int(not np.any(before[0, :] == info["marker"]) and
                                       not np.any(before[-1, :] == info["marker"]) and
                                       not np.any(before[:, 0] == info["marker"]) and
                                       not np.any(before[:, -1] == info["marker"]))
                saturated_pairs += int(any(
                    len(side) == info["capacity"][s] and len(side) > 0
                    for lane in data for s, side in enumerate(lane)
                ))
                run_pairs += int(any(
                    len(side) >= 2 and any(abs(side[j] - side[j - 1]) == 1 for j in range(1, len(side)))
                    for lane in data for side in lane
                ))
    assert len(shapes) >= 30 and min(x[0] for x in shapes) <= 8 and max(x[0] for x in shapes) >= 28
    assert 1 in thicknesses and max(thicknesses) >= 6
    assert any(min(c) <= 3 for c in capacities)
    assert len(totals) >= 15 and max(maximums) >= 5
    assert simple_pairs >= 20 and one_side_pairs >= 10
    assert edge_free_pairs >= 10 and saturated_pairs >= 10 and run_pairs >= 10
    assert len(orientation_profiles) >= 4 and max(palette_profiles) > 1 and min(palette_profiles) == 1
    assert all(split_thin[s] >= 5 and split_thick[s] >= 5 for s in ("train", "test"))
    assert all(split_horizontal[s] >= 5 for s in ("train", "test"))
    assert familiar_tests >= 5 and new_tests >= 5
    assert source_sized >= 1 and edge_sized >= 1
    assert distance_twin_puzzles >= 5
    for train, test in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(500 + train + test, train, test)
        assert (len(p["train"]), len(p["test"])) == (train, test)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(900, train, test)
        assert (train is None or len(p["train"]) == train)
        assert (test is None or len(p["test"]) == test)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    assert _fingerprint(generate_puzzle(789, 6, 3)) == _fingerprint(generate_puzzle(789, 6, 3))
    assert _fingerprint(generate_puzzle(789, 6, 3)) != _fingerprint(generate_puzzle(790, 6, 3))
