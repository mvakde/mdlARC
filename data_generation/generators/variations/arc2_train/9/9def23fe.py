from __future__ import annotations

import hashlib
import numpy as np

# Directions name separate rays, even when they share a row or column.
UP, DOWN, LEFT, RIGHT = range(4)
_SOURCE_PAIR_DIGESTS = {
    "9203a008e9c0d5ba66df61365b259f34dc58bb35ab8852f1dfa48928d6e5c36b",
    "135cdfb35d01bbb56f8a1dbe32897712a0d0e43a51d4bd953c063ca958525268",
    "965e3e3a9294ea38b5b870239729d86d299fe8f54ba96c3cf1ff568701ae0ea0",
    "a6bdc48b5d31fbe77ec58f0bcee82e2017aac7633c080fde7381ce8a81a6a578",
}
_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 1), (5, 3), (6, 2),
                (7, 3), (8, 4), (9, 1), (9, 4), (10, 2), (10, 4))


def _checked_integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, rng, num_train, num_test):
    if num_train is not None:
        num_train = _checked_integer(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _checked_integer(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _dimension(rng, side):
    band = int(rng.integers(3))
    low, high = ((side + 2, min(15, 30)),
                 (max(side + 2, 15), 23),
                 (max(side + 2, 23), 31))[band]
    return int(rng.integers(low, high))


def _margin(rng, extra):
    # Both short and long rays occur, independently on the two axes.
    if int(rng.integers(4)) == 0:
        return int(rng.integers(1, min(2, extra - 1) + 1))
    if int(rng.integers(4)) == 0:
        return extra - int(rng.integers(1, min(2, extra - 1) + 1))
    return int(rng.integers(1, extra))


def _geometry(rng, square_canvas=None):
    side = int(rng.integers(4, 11))
    if square_canvas is None:
        square_canvas = bool(rng.integers(2))
    height = _dimension(rng, side)
    width = height if square_canvas else _dimension(rng, side)
    if not square_canvas and width == height:
        width = width + 1 if width < 30 else width - 1
    top = _margin(rng, height - side)
    left = _margin(rng, width - side)
    return height, width, top, left, side


def _position(direction, lane, depth, top, left, side):
    if direction == UP:
        return top - depth, left + lane
    if direction == DOWN:
        return top + side - 1 + depth, left + lane
    if direction == LEFT:
        return top + lane, left - depth
    return top + lane, left + side - 1 + depth


def _lanes(rng, side, margins, anchor, marker_free, full_side):
    if marker_free:
        return []
    all_lanes = [(direction, lane) for direction in range(4)
                 for lane in range(side)]
    if anchor:
        # Exactly eight distinct lanes, with both states on every side.
        counts = np.ones(4, dtype=int)
        for _ in range(4):
            available = [d for d in range(4) if counts[d] < side - 1]
            counts[int(rng.choice(available))] += 1
        return [(d, int(lane)) for d in range(4)
                for lane in rng.choice(side, size=int(counts[d]), replace=False)]
    selected = set()
    if full_side:
        choices = [d for d in range(4) if margins[d] >= 2]
        if not choices:
            return None
        direction = int(rng.choice(choices))
        selected.update((direction, lane) for lane in range(side))
    total = 8 if int(rng.integers(4)) == 0 else int(rng.integers(2, min(16, 4 * side - 1) + 1))
    total = max(total, len(selected))
    rng.shuffle(all_lanes)
    for candidate in all_lanes:
        if len(selected) >= total:
            break
        selected.add(candidate)
    return sorted(selected)


def _marker_positions(rng, geometry, lanes, repeat, distractors):
    height, width, top, left, side = geometry
    margins = (top, height - top - side, left, width - left - side)
    positions = []
    if repeat:
        candidates = [lane for lane in lanes if margins[lane[0]] >= 3]
        if not candidates:
            return None
        repeated_lane = candidates[int(rng.integers(len(candidates)))]
    else:
        repeated_lane = None

    # A few random orders and depths avoid accidental adjacent markers.
    for _ in range(36):
        positions.clear()
        ordered = list(lanes)
        rng.shuffle(ordered)
        if repeated_lane is not None:
            ordered.remove(repeated_lane)
            ordered.insert(0, repeated_lane)
        successful = True
        for direction, lane in ordered:
            depth_options = list(range(1, margins[direction] + 1))
            rng.shuffle(depth_options)
            choices = ((a, b) for a in depth_options for b in depth_options
                       if abs(a - b) >= 2) if (direction, lane) == repeated_lane else ((a,) for a in depth_options)
            for depths in choices:
                candidates = [_position(direction, lane, depth, top, left, side) for depth in depths]
                if all(all(abs(row - old_row) + abs(col - old_col) != 1
                           for old_row, old_col in positions + candidates[:index])
                       for index, (row, col) in enumerate(candidates)):
                    positions.extend(candidates)
                    break
            else:
                successful = False
                break
        if successful:
            break
    else:
        return None

    if distractors:
        corners = [(row, col) for row in range(height) for col in range(width)
                   if (row < top or row >= top + side)
                   and (col < left or col >= left + side)]
        rng.shuffle(corners)
        wanted = int(rng.integers(1, 4))
        for row, col in corners:
            if all(abs(row - old_row) + abs(col - old_col) != 1
                   for old_row, old_col in positions):
                positions.append((row, col))
                wanted -= 1
                if wanted == 0:
                    break
        if wanted:
            return None
    return positions


def _transform(input_grid, top, left, side, source, marker):
    output = input_grid.copy()
    bottom = top + side
    right = left + side
    for row in range(top, bottom):
        if not np.any(input_grid[row, :left] == marker):
            output[row, :left] = source
        if not np.any(input_grid[row, right:] == marker):
            output[row, right:] = source
    for col in range(left, right):
        if not np.any(input_grid[:top, col] == marker):
            output[:top, col] = source
        if not np.any(input_grid[bottom:, col] == marker):
            output[bottom:, col] = source
    return output


def _make_pair(rng, colors, anchor=False, features=(False, False, False, False),
               square_canvas=None):
    marker_free, full_side, repeat, distractors = features
    for _ in range(256):
        geometry = _geometry(rng, square_canvas)
        height, width, top, left, side = geometry
        margins = (top, height - top - side, left, width - left - side)
        lanes = _lanes(rng, side, margins, anchor, marker_free, full_side)
        if lanes is None or (repeat and not lanes):
            continue
        markers = _marker_positions(rng, geometry, lanes, repeat, distractors)
        if markers is None:
            continue
        if height * width - side * side <= len(markers):
            continue
        background, source, marker = colors
        input_grid = np.full((height, width), background, dtype=np.int8)
        input_grid[top:top + side, left:left + side] = source
        for row, col in markers:
            input_grid[row, col] = marker
        output_grid = _transform(input_grid, top, left, side, source, marker)
        if not np.array_equal(input_grid, output_grid):
            return {"input": input_grid, "output": output_grid}
    raise RuntimeError("could not construct valid marker layout")


def _choose_features(rng, allow_repeat=True, allow_distractors=True):
    marker_free = bool(rng.random() < 0.09)
    if marker_free:
        return True, False, False, False
    return (False, bool(rng.random() < 0.22),
            bool(allow_repeat and rng.random() < 0.24),
            bool(allow_distractors and rng.random() < 0.22))


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _checked_integer(seed, "seed", 0, 2**63 - 1)
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(seed, rng, num_train, num_test)
    # The ray rule is the puzzle-level convention. Color roles are identified
    # by local structure, so a puzzle may keep or vary their literal colors.
    palette = tuple(int(x) for x in rng.permutation(10))
    vary_source_colors = bool(rng.integers(2))
    base_colors = palette[:3]
    puzzle = {"train": [], "test": []}
    used = set()
    seen_marker_colors = set()
    train_color_pairs = []
    train_has_repeat = False
    train_has_distractors = False
    anchor_index = int(rng.integers(num_train))
    train_square_shape = bool(rng.integers(2))

    def colors_for_pair(index, split):
        if vary_source_colors and (split == "test" or index != 0):
            if split == "test" and rng.random() < 0.5:
                background, source = train_color_pairs[int(rng.integers(len(train_color_pairs)))]
            else:
                background, source = [int(x) for x in rng.choice(10, size=2, replace=False)]
        else:
            background, source = base_colors[:2]
        options = [c for c in range(10) if c not in (background, source)]
        if split == "train":
            unused = [c for c in options if c not in seen_marker_colors]
            if index == 0 or (index == 1 and unused) or (unused and rng.random() < 0.35):
                marker = int(rng.choice(unused or options))
            else:
                marker = int(rng.choice(options))
        else:
            reused = [c for c in options if c in seen_marker_colors]
            fresh = [c for c in options if c not in seen_marker_colors]
            marker = int(rng.choice((reused if rng.random() < 0.5 else fresh) or options))
        return background, source, marker

    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            anchor = split == "train" and index == anchor_index
            if anchor:
                features = (False, False, False, False)
            elif split == "train":
                features = _choose_features(rng)
            else:
                features = _choose_features(rng, train_has_repeat, train_has_distractors)
            if split == "train" and index < 2:
                square_canvas = train_square_shape if index == 0 else not train_square_shape
            else:
                square_canvas = None
            colors = colors_for_pair(index, split)
            for _ in range(12):
                pair = _make_pair(rng, colors, anchor, features, square_canvas)
                key = _grid_key(pair["input"])
                if key not in used:
                    break
            else:
                raise RuntimeError("could not create unique pair")
            used.add(key)
            puzzle[split].append(pair)
            if split == "train":
                train_color_pairs.append(colors[:2])
                if features[0] is False:
                    seen_marker_colors.add(colors[2])
                train_has_repeat |= features[2]
                train_has_distractors |= features[3]
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_input(grid):
    candidates = []
    for color in np.unique(grid):
        positions = np.argwhere(grid == color)
        top, left = np.min(positions, axis=0)
        bottom, right = np.max(positions, axis=0)
        side = int(bottom - top + 1)
        if side >= 4 and side == right - left + 1 and len(positions) == side * side:
            candidates.append((int(color), int(top), int(left), side))
    assert len(candidates) == 1
    source, top, left, side = candidates[0]
    assert 1 <= top and top + side < grid.shape[0]
    assert 1 <= left and left + side < grid.shape[1]
    other = [int(c) for c in np.unique(grid) if c != source]
    assert 1 <= len(other) <= 2
    background = max(other, key=lambda c: int(np.count_nonzero(grid == c)))
    marker_colors = [c for c in other if c != background]
    assert len(marker_colors) <= 1
    marker = marker_colors[0] if marker_colors else None
    marker_positions = [] if marker is None else [tuple(map(int, p)) for p in np.argwhere(grid == marker)]
    skeleton = np.full_like(grid, background)
    skeleton[top:top + side, left:left + side] = source
    for row, col in marker_positions:
        skeleton[row, col] = marker
    assert np.array_equal(grid, skeleton)
    for i, (row, col) in enumerate(marker_positions):
        for old_row, old_col in marker_positions[:i]:
            assert abs(row - old_row) + abs(col - old_col) != 1
    margins = (top, grid.shape[0] - top - side, left, grid.shape[1] - left - side)
    lane_positions = {d: {lane: [] for lane in range(side)} for d in range(4)}
    distractors = 0
    for row, col in marker_positions:
        if row < top and left <= col < left + side:
            lane_positions[UP][col - left].append(top - row)
        elif row >= top + side and left <= col < left + side:
            lane_positions[DOWN][col - left].append(row - top - side + 1)
        elif col < left and top <= row < top + side:
            lane_positions[LEFT][row - top].append(left - col)
        elif col >= left + side and top <= row < top + side:
            lane_positions[RIGHT][row - top].append(col - left - side + 1)
        else:
            distractors += 1
    assert all(len(depths) <= 2 and (len(depths) < 2 or abs(depths[0] - depths[1]) >= 2)
               for directions in lane_positions.values() for depths in directions.values())
    return {"source": source, "background": background, "marker": marker,
            "top": top, "left": left, "side": side, "margins": margins,
            "lanes": lane_positions, "distractors": distractors,
            "marker_count": len(marker_positions), "square_canvas": grid.shape[0] == grid.shape[1]}


def _independent_expected(grid, info):
    expected = grid.copy()
    top, left, side = info["top"], info["left"], info["side"]
    source = info["source"]
    h, w = grid.shape
    for direction in range(4):
        for lane in range(side):
            if info["lanes"][direction][lane]:
                continue
            if direction == UP:
                cells = ((row, left + lane) for row in range(top))
            elif direction == DOWN:
                cells = ((row, left + lane) for row in range(top + side, h))
            elif direction == LEFT:
                cells = ((top + lane, col) for col in range(left))
            else:
                cells = ((top + lane, col) for col in range(left + side, w))
            for row, col in cells:
                expected[row, col] = source
    return expected


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert input_grid.shape == output_grid.shape
            input_key = _grid_key(input_grid)
            pair_key = (input_key, _grid_key(output_grid))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            digest = hashlib.sha256(input_grid.astype(np.int8).tobytes() +
                                    output_grid.astype(np.int8).tobytes()).hexdigest()
            assert digest not in _SOURCE_PAIR_DIGESTS
            info = _analyze_input(input_grid)
            assert np.array_equal(output_grid, _independent_expected(input_grid, info))
            assert not np.array_equal(input_grid, output_grid)
            if info["marker"] is not None:
                marker_cells = input_grid == info["marker"]
                assert np.array_equal(input_grid[marker_cells], output_grid[marker_cells])
            infos[split].append(info)
    # The anchor demonstration has blocked and open rays on all four sides.
    assert any(info["marker_count"] == 8 and info["distractors"] == 0 and
               all(0 < sum(bool(depths) for depths in info["lanes"][direction].values()) < info["side"]
                   for direction in range(4)) for info in infos["train"])
    assert any(info["square_canvas"] for info in infos["train"])
    assert any(not info["square_canvas"] for info in infos["train"])
    train_has_repeat = any(len(depths) == 2 for info in infos["train"]
                           for lanes in info["lanes"].values() for depths in lanes.values())
    train_has_distractors = any(info["distractors"] for info in infos["train"])
    for info in infos["test"]:
        if any(len(depths) == 2 for lanes in info["lanes"].values() for depths in lanes.values()):
            assert train_has_repeat
        if info["distractors"]:
            assert train_has_distractors


def _puzzles_equal(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(pa[key], pb[key]) for pa, pb in zip(a[split], b[split])
        for key in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    auto = generate_puzzle(12345)
    validate_generated_puzzle(auto)
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                       (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(500 + i, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    assert _puzzles_equal(generate_puzzle(2345, 8, 3), generate_puzzle(2345, 8, 3))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(700, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    train_counts, test_counts, shapes, layouts, patterns, colors = (set() for _ in range(6))
    coverage = {split: {name: 0 for name in (
        "square", "rectangle", "free", "under_eight", "over_eight",
        "ordinary_eight", "repeat", "distractor", "full_side", "open_side",
        "upper_marker", "margin_one", "margin_two", "long_margin",
        "near_marker", "boundary_marker", "combined")}
        for split in ("train", "test")}
    within = {name: 0 for name in (
        "shapes", "sides", "marker_counts", "train_colors", "familiar_test_side",
        "new_test_side", "reused_test_marker", "new_test_marker")}
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        infos = {"train": [], "test": []}
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(_grid_key(p["input"]) for split in ("train", "test")
                             for p in puzzle[split]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                info = _analyze_input(grid)
                infos[split].append(info)
                assert np.array_equal(pair["output"], _independent_expected(grid, info))
                shapes.add(grid.shape)
                layouts.add((grid.shape, info["top"], info["left"], info["side"],
                             tuple(tuple(len(depths) for depths in info["lanes"][d].values())
                                   for d in range(4))))
                patterns.add(tuple(sum(bool(depths) for depths in info["lanes"][d].values())
                                   for d in range(4)))
                colors.add((info["background"], info["source"], info["marker"]))
                c = coverage[split]
                blocked = [sum(bool(depths) for depths in info["lanes"][d].values())
                           for d in range(4)]
                depths = [(d, depth) for d in range(4)
                          for lane_depths in info["lanes"][d].values()
                          for depth in lane_depths]
                repeat = any(len(lane_depths) == 2 for lanes in info["lanes"].values()
                             for lane_depths in lanes.values())
                c["square"] += info["square_canvas"]
                c["rectangle"] += not info["square_canvas"]
                c["free"] += info["marker_count"] == 0
                c["under_eight"] += 0 < info["marker_count"] < 8
                c["over_eight"] += info["marker_count"] > 8
                c["ordinary_eight"] += info["marker_count"] == 8 and not repeat and not info["distractors"]
                c["repeat"] += repeat
                c["distractor"] += bool(info["distractors"])
                c["full_side"] += any(n == info["side"] for n in blocked)
                c["open_side"] += any(n == 0 for n in blocked)
                c["upper_marker"] += blocked[UP] > 0
                c["margin_one"] += 1 in info["margins"]
                c["margin_two"] += 2 in info["margins"]
                c["long_margin"] += max(info["margins"]) >= 8
                c["near_marker"] += any(depth == 1 for _, depth in depths)
                c["boundary_marker"] += any(depth == info["margins"][d] for d, depth in depths)
                c["combined"] += repeat and info["distractors"] > 0 and min(info["margins"]) <= 2
        training, testing = infos["train"], infos["test"]
        train_sides = {info["side"] for info in training}
        train_markers = {info["marker"] for info in training if info["marker"] is not None}
        within["shapes"] += len({pair["input"].shape for split in ("train", "test")
                                 for pair in puzzle[split]}) > 1
        within["sides"] += len({info["side"] for split in ("train", "test")
                                for info in infos[split]}) > 1
        within["marker_counts"] += len({info["marker_count"] for split in ("train", "test")
                                        for info in infos[split]}) > 1
        within["train_colors"] += len({(info["background"], info["source"])
                                       for info in training}) > 1
        within["familiar_test_side"] += any(info["side"] in train_sides for info in testing)
        within["new_test_side"] += any(info["side"] not in train_sides for info in testing)
        within["reused_test_marker"] += any(info["marker"] in train_markers for info in testing
                                             if info["marker"] is not None)
        within["new_test_marker"] += any(info["marker"] not in train_markers for info in testing
                                          if info["marker"] is not None)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(shapes) >= 35
    assert len(layouts) >= 200 and len(patterns) >= 25 and len(colors) >= 35
    for split in ("train", "test"):
        assert all(value > 0 for value in coverage[split].values()), (split, coverage[split])
    assert within["shapes"] == 42 and within["sides"] == 42
    assert within["marker_counts"] >= 38
    assert all(value >= 10 for name, value in within.items()
               if name not in ("shapes", "sides", "marker_counts")), within
