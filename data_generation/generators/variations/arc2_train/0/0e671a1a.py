"""Compositional examples of the two ordered Manhattan routes ARC rule."""

from itertools import permutations

import numpy as np


def _check_optional_count(name, value, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _choose_counts(rng, num_train, num_test):
    _check_optional_count("num_train", num_train, 2, 10)
    _check_optional_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, int(num_test) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _single_position(grid, color):
    locations = np.argwhere(grid == color)
    if locations.shape != (1, 2):
        raise ValueError("marker must occur exactly once")
    return tuple(int(x) for x in locations[0])


def _construct_output(input_grid, background, pivot, horizontal, vertical, connector):
    pr, pc = _single_position(input_grid, pivot)
    hr, hc = _single_position(input_grid, horizontal)
    vr, vc = _single_position(input_grid, vertical)
    route = set()
    route.update((pr, c) for c in range(min(pc, hc), max(pc, hc) + 1))
    route.update((r, hc) for r in range(min(pr, hr), max(pr, hr) + 1))
    route.update((r, pc) for r in range(min(pr, vr), max(pr, vr) + 1))
    route.update((vr, c) for c in range(min(pc, vc), max(pc, vc) + 1))
    output = input_grid.copy()
    for r, c in route:
        if output[r, c] == background:
            output[r, c] = connector
    return output


def _sample_shape(rng, regime, challenge=False):
    # Independent dimensions include tiny, large, and pronounced aspect ratios.
    choices = ("small", "medium", "large", "wide", "tall")
    weights = (0.19, 0.35, 0.27, 0.095, 0.095) if regime == "source_train" else (
        0.16, 0.31, 0.29, 0.12, 0.12
    )
    bucket = "large" if challenge and rng.random() < 0.45 else str(rng.choice(choices, p=weights))
    if bucket == "small":
        return int(rng.integers(4, 10)), int(rng.integers(4, 10))
    if bucket == "medium":
        return int(rng.integers(10, 23)), int(rng.integers(10, 23))
    if bucket == "large":
        return int(rng.integers(23, 31)), int(rng.integers(23, 31))
    if bucket == "wide":
        return int(rng.integers(4, 10)), int(rng.integers(18, 31))
    return int(rng.integers(18, 31)), int(rng.integers(4, 10))


def _three_coordinates(rng, length, force_edge=False):
    if force_edge:
        edge = int(rng.choice((0, length - 1)))
        others = rng.choice([x for x in range(length) if x != edge], 2, replace=False)
        return sorted((edge, int(others[0]), int(others[1])))
    if length >= 9 and rng.random() < 0.28:
        # Sample a local route span, then place that span independently on
        # the canvas so a short route need not imply a small grid or edge.
        span = int(rng.integers(3, min(9, length) + 1))
        start = int(rng.integers(0, length - span + 1))
        return sorted(int(x) for x in rng.choice(
            np.arange(start, start + span), 3, replace=False
        ))
    return sorted(int(x) for x in rng.choice(length, 3, replace=False))


def _sample_positions(rng, height, width, mode):
    edge = mode == "edge" or (mode != "free" and rng.random() < 0.18)
    rows = _three_coordinates(rng, height, edge)
    cols = _three_coordinates(rng, width, edge)
    if mode in ("free", "edge", "align"):
        rr = rng.permutation(rows)
        cc = rng.permutation(cols)
        points = [(int(rr[i]), int(cc[i])) for i in range(3)]
        if mode == "align":
            # A zero-length leg or two targets on one row/column.
            axis = int(rng.integers(0, 2))
            a, b = rng.choice(3, 2, replace=False)
            changed = list(points[b])
            changed[axis] = points[a][axis]
            points[b] = tuple(changed)
        return tuple(points)

    row_direction = int(rng.choice((-1, 1)))
    col_direction = int(rng.choice((-1, 1)))
    near_row = rows[1]
    near_col = cols[1]
    pr = rows[0] if row_direction == 1 else rows[2]
    pc = cols[0] if col_direction == 1 else cols[2]
    far_row = rows[2] if row_direction == 1 else rows[0]
    far_col = cols[2] if col_direction == 1 else cols[0]
    if mode == "cross":
        return (pr, pc), (far_row, near_col), (near_row, far_col)
    if mode == "shared":
        if rng.random() < 0.5:
            # H lies on V's vertical run; that run is shared beyond the pivot.
            return (pr, pc), (near_row, pc), (far_row, far_col)
        # V lies on H's horizontal run.
        return (pr, pc), (far_row, far_col), (pr, near_col)
    raise ValueError("unknown placement mode")


def _layout_signature(points):
    p, h, v = points
    return tuple(int(np.sign(a[i] - b[i])) for i in range(2) for a, b in ((p, h), (p, v), (h, v)))


def _grid_key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


_SOURCE_POSITIONS = (
    ((1, 1), (4, 11), (10, 5)),
    ((10, 2), (1, 5), (8, 11)),
    ((1, 10), (11, 9), (5, 2)),
    ((6, 10), (11, 3), (2, 1)),
    ((2, 10), (11, 7), (5, 1)),
)


def _matches_source_pair(input_grid, output_grid):
    if input_grid.shape != (13, 13):
        return False
    for pivot, horizontal, vertical in _SOURCE_POSITIONS:
        source = np.zeros((13, 13), dtype=np.int8)
        source[pivot], source[horizontal], source[vertical] = 4, 3, 2
        if np.array_equal(input_grid, source) and np.array_equal(
            output_grid, _construct_output(source, 0, 4, 3, 2, 5)
        ):
            return True
    return False


def _background_and_markers(input_grid):
    colors, counts = np.unique(input_grid, return_counts=True)
    assert len(colors) == 4
    dominant = int(np.argmax(counts))
    assert int(counts[dominant]) == input_grid.size - 3
    assert all(int(n) == 1 for i, n in enumerate(counts) if i != dominant)
    background = int(colors[dominant])
    return background, tuple(sorted(int(c) for c in colors if int(c) != background))


def _infer_roles(training, markers, connector):
    valid = []
    for pivot, horizontal, vertical in permutations(markers):
        if all(np.array_equal(
            _construct_output(pair["input"], _background_and_markers(pair["input"])[0],
                              pivot, horizontal, vertical, connector), pair["output"]
        ) for pair in training):
            valid.append((pivot, horizontal, vertical))
    return valid


def _make_pair(rng, role_colors, background, mode, regime, challenge=False):
    height, width = _sample_shape(rng, regime, challenge)
    positions = _sample_positions(rng, height, width, mode)
    grid = np.full((height, width), background, dtype=np.int8)
    for point, color in zip(positions, role_colors[:3]):
        grid[point] = color
    output = _construct_output(grid, background, *role_colors)
    return {"input": grid, "output": output}, positions


def _fallback_pair(index, role_colors, background, seen_inputs,
                   train_signatures, train_shapes, training, require_novel):
    # Exhaustive rank orders on a reserved, index-specific rectangle make
    # rejection bounded without depending on a lucky random draw.
    height, width = 14 + index, 15 + index
    rows = (0, height // 2, height - 1)
    cols = (0, width // 2, width - 1)
    for row_order in permutations(range(3)):
        for col_order in permutations(range(3)):
            points = tuple((rows[row_order[i]], cols[col_order[i]]) for i in range(3))
            signature = _layout_signature(points)
            if require_novel and signature in train_signatures:
                continue
            if index == 1 and (signature in train_signatures or
                               (height, width) in train_shapes):
                continue
            grid = np.full((height, width), background, dtype=np.int8)
            for point, color in zip(points, role_colors[:3]):
                grid[point] = color
            output = _construct_output(grid, background, *role_colors)
            pair = {"input": grid, "output": output}
            if np.array_equal(grid, output) or _grid_key(grid) in seen_inputs:
                continue
            if index == 1 and len(_infer_roles(training + [pair],
                                               tuple(sorted(role_colors[:3])), role_colors[3])) != 1:
                continue
            return pair, signature
    raise RuntimeError("no informative fallback layout exists")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    colors = [int(x) for x in rng.permutation(10)]
    # Marker and connector meanings are fixed; only the evident background varies.
    role_colors = tuple(colors[:4])  # pivot, horizontal, vertical, connector
    backgrounds = colors[4:]
    regime_style = str(rng.choice(("matched", "novel_layout", "challenge")))
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max,
                              size=train_count + test_count, dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    train_signatures = set()
    train_shapes = set()
    background_offset = int(rng.integers(0, len(backgrounds)))

    for index, pair_seed in enumerate(pair_seeds):
        is_train = index < train_count
        destination = puzzle["train"] if is_train else puzzle["test"]
        regime = "source_train" if is_train else "target_test"
        pair_rng = np.random.default_rng(pair_seed)
        background = backgrounds[(background_offset + index) % len(backgrounds)]
        # The first example is informative; its geometry and direction remain random.
        if index == 0:
            modes = ("free",)
        elif is_train:
            modes = ("free", "cross", "align", "shared", "edge")
        elif regime_style == "challenge":
            modes = ("cross", "shared", "edge", "align", "free")
        else:
            modes = ("free", "cross", "align", "shared", "edge")
        weights = (0.36, 0.22, 0.15, 0.12, 0.15)
        if not is_train and regime_style == "challenge":
            weights = (0.13, 0.27, 0.14, 0.22, 0.24)

        selected = None
        for attempt in range(256):
            mode = modes[0] if index == 0 else str(pair_rng.choice(modes, p=weights))
            pair, positions = _make_pair(pair_rng, role_colors, background, mode,
                                         regime, regime_style == "challenge" and not is_train)
            signature = _layout_signature(positions)
            grid = pair["input"]
            if np.array_equal(grid, pair["output"]):
                continue
            if _grid_key(grid) in seen_inputs or _matches_source_pair(grid, pair["output"]):
                continue
            if index == 1 and (signature in train_signatures or grid.shape in train_shapes):
                continue
            if not is_train and regime_style == "novel_layout" and signature in train_signatures:
                continue
            if index == 1 and len(_infer_roles(puzzle["train"] + [pair],
                                               tuple(sorted(role_colors[:3])), role_colors[3])) != 1:
                continue
            selected = pair, signature
            break
        if selected is None:
            selected = _fallback_pair(index, role_colors, background, seen_inputs,
                                      train_signatures, train_shapes, puzzle["train"],
                                      not is_train and regime_style == "novel_layout")
        pair, signature = selected
        destination.append(pair)
        seen_inputs.add(_grid_key(pair["input"]))
        if is_train:
            train_signatures.add(signature)
            train_shapes.add(pair["input"].shape)

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    marker_colors = None
    connector_color = None
    train_signatures = set()
    train_shapes = set()
    backgrounds = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            assert not _matches_source_pair(inp, out)
            background, markers = _background_and_markers(inp)
            input_colors = set(int(x) for x in np.unique(inp))
            output_colors = set(int(x) for x in np.unique(out))
            new_colors = output_colors - input_colors
            assert len(new_colors) == 1
            connector = next(iter(new_colors))
            assert output_colors == input_colors | {connector}
            if marker_colors is None:
                marker_colors, connector_color = markers, connector
            assert markers == marker_colors and connector == connector_color
            assert background not in markers and background != connector
            for marker in markers:
                location = _single_position(inp, marker)
                assert out[location] == marker
            if split == "train":
                backgrounds.add(background)
                train_shapes.add(inp.shape)
    assert len(backgrounds) >= 2 and len(train_shapes) >= 2
    assignments = _infer_roles(puzzle["train"], marker_colors, connector_color)
    assert len(assignments) == 1
    pivot, horizontal, vertical = assignments[0]
    for split in ("train", "test"):
        for pair in puzzle[split]:
            inp, out = pair["input"], pair["output"]
            background, _ = _background_and_markers(inp)
            assert np.array_equal(out, _construct_output(
                inp, background, pivot, horizontal, vertical, connector_color
            ))
            points = tuple(_single_position(inp, color) for color in assignments[0])
            if split == "train":
                train_signatures.add(_layout_signature(points))
    assert len(train_signatures) >= 2
    assert any(len({point[0] for point in (
        _single_position(p["input"], pivot), _single_position(p["input"], horizontal),
        _single_position(p["input"], vertical))}) == 3 and len({point[1] for point in (
        _single_position(p["input"], pivot), _single_position(p["input"], horizontal),
        _single_position(p["input"], vertical))}) == 3 for p in puzzle["train"])


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
        for k in ("input", "output")
    ) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, signatures, schemes = set(), set(), set(), set(), set()
    train_orders, test_cross_orientations = set(), set()
    per_split = {"train": {"cross": 0, "noncross": 0, "aligned": 0, "shared": 0},
                 "test": {"cross": 0, "noncross": 0, "aligned": 0, "shared": 0}}
    target_swaps = 0
    backgrounds_varied = 0
    puzzle_review = {"seen_test_layout": 0, "novel_test_layout": 0,
                     "longer_test_leg": 0}
    modes = {"cross": 0, "aligned": 0, "shared": 0, "edge": 0,
             "small": 0, "large": 0, "wide": 0, "tall": 0,
             "pivot_middle_row": 0, "pivot_middle_col": 0,
             "above_left": 0, "opposite_quadrants": 0,
             "corner": 0, "edge_bend": 0,
             "padded_short_leg": 0, "long_edge": 0}
    test_signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        bg, markers = _background_and_markers(puzzle["train"][0]["input"])
        connector = next(iter(set(int(x) for x in np.unique(puzzle["train"][0]["output"])) -
                              set(int(x) for x in np.unique(puzzle["train"][0]["input"]))))
        roles = _infer_roles(puzzle["train"], markers, connector)[0]
        schemes.add((*roles, connector))
        train_backgrounds = set()
        row_relations, col_relations = set(), set()
        train_relative, test_relative = set(), set()
        train_departures, test_departures = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                h, w = inp.shape
                shapes.add((h, w))
                points = tuple(_single_position(inp, c) for c in roles)
                signature = _layout_signature(points)
                signatures.add(signature)
                if split == "test":
                    test_signatures.add(signature)
                pr, pc = points[0]
                hr, hc = points[1]
                vr, vc = points[2]
                if split == "train":
                    train_relative.add(signature)
                    train_departures.append((abs(pc - hc), abs(pr - vr)))
                    train_backgrounds.add(_background_and_markers(inp)[0])
                    row_relations.add(int(np.sign(hr - vr)))
                    col_relations.add(int(np.sign(hc - vc)))
                    if len({pr, hr, vr}) == len({pc, hc, vc}) == 3:
                        train_orders.add((
                            tuple(sorted(range(3), key=lambda i: points[i][0])),
                            tuple(sorted(range(3), key=lambda i: points[i][1])),
                        ))
                else:
                    test_relative.add(signature)
                    test_departures.append((abs(pc - hc), abs(pr - vr)))
                aligned = len({pr, hr, vr}) < 3 or len({pc, hc, vc}) < 3
                crossing = (min(pr, hr) < vr < max(pr, hr) and
                            min(pc, vc) < hc < max(pc, vc))
                shared = (pc == hc and (hr - pr) * (vr - hr) >= 0) or (
                    pr == vr and (vc - pc) * (hc - vc) >= 0)
                per_split[split]["cross" if crossing else "noncross"] += 1
                per_split[split]["aligned"] += aligned
                per_split[split]["shared"] += shared
                if split == "test" and crossing:
                    test_cross_orientations.add((int(np.sign(hr - pr)), int(np.sign(vc - pc))))
                modes["aligned"] += len({pr, hr, vr}) < 3 or len({pc, hc, vc}) < 3
                modes["shared"] += (pc == hc and (hr - pr) * (vr - hr) >= 0) or (
                    pr == vr and (vc - pc) * (hc - vc) >= 0)
                modes["cross"] += pr != vr != hr and pc != hc != vc and (
                    min(pr, hr) < vr < max(pr, hr) and min(pc, vc) < hc < max(pc, vc))
                modes["edge"] += any(r in (0, h - 1) or c in (0, w - 1) for r, c in points)
                modes["corner"] += any(r in (0, h - 1) and c in (0, w - 1)
                                       for r, c in points)
                modes["edge_bend"] += hc in (0, w - 1) or vr in (0, h - 1)
                modes["pivot_middle_row"] += min(hr, vr) < pr < max(hr, vr)
                modes["pivot_middle_col"] += min(hc, vc) < pc < max(hc, vc)
                modes["above_left"] += hr < pr and vr < pr and hc < pc and vc < pc
                modes["opposite_quadrants"] += (hr - pr) * (vr - pr) < 0 and (
                    hc - pc) * (vc - pc) < 0
                margin = min(min(pr, hr, vr), h - 1 - max(pr, hr, vr),
                             min(pc, hc, vc), w - 1 - max(pc, hc, vc))
                modes["padded_short_leg"] += h >= 20 and w >= 20 and margin >= 3 and (
                    abs(pc - hc) <= 3 or abs(pr - vr) <= 3)
                modes["long_edge"] += margin == 0 and (
                    abs(pc - hc) >= 8 or abs(pr - vr) >= 8)
                modes["small"] += h <= 9 and w <= 9
                modes["large"] += h >= 23 and w >= 23
                modes["wide"] += w >= 2 * h
                modes["tall"] += h >= 2 * w
                # Independently draw both ordered routes using array slices.
                expected = inp.copy()
                expected[pr, min(pc, hc):max(pc, hc)+1] = connector
                expected[min(pr, hr):max(pr, hr)+1, hc] = connector
                expected[min(pr, vr):max(pr, vr)+1, pc] = connector
                expected[vr, min(pc, vc):max(pc, vc)+1] = connector
                for point, color in zip(points, roles):
                    expected[point] = color
                assert np.array_equal(expected, out)
        backgrounds_varied += len(train_backgrounds) >= 2
        target_swaps += len(row_relations) >= 2 and len(col_relations) >= 2
        puzzle_review["seen_test_layout"] += bool(train_relative & test_relative)
        puzzle_review["novel_test_layout"] += bool(test_relative - train_relative)
        puzzle_review["longer_test_leg"] += any(
            horizontal > max(t[0] for t in train_departures) or
            vertical > max(t[1] for t in train_departures)
            for horizontal, vertical in test_departures)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 70 and len(signatures) >= 25 and len(schemes) >= 30
    assert len(test_signatures) >= 15
    assert all(value >= 5 for value in modes.values()), modes
    assert len(train_orders) == 36 and len(test_cross_orientations) == 4
    assert backgrounds_varied == 42 and target_swaps >= 30
    assert all(per_split[s][k] >= 10 for s in ("train", "test")
               for k in ("cross", "noncross", "aligned", "shared")), per_split
    assert puzzle_review["seen_test_layout"] >= 5
    assert puzzle_review["novel_test_layout"] >= 30
    assert puzzle_review["longer_test_leg"] >= 10
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4),
                                    (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(8100 + index, *counts)
        validate_generated_puzzle(puzzle)
        for actual, requested in zip((len(puzzle["train"]), len(puzzle["test"])), counts):
            assert requested is None or actual == requested
    assert _puzzles_equal(generate_puzzle(8001, 6, 2), generate_puzzle(8001, 6, 2))
    assert not _puzzles_equal(generate_puzzle(8001, 6, 2), generate_puzzle(8002, 6, 2))
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(99173, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {counts}")
