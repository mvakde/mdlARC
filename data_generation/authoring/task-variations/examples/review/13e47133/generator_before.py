import numpy as np


_SOURCE_GRID_SHAPES = {(10, 13), (20, 20), (30, 30)}
_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIR8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _require_optional_int(name, value, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(seed, num_train, num_test):
    _require_optional_int("num_train", num_train, 2, 10)
    _require_optional_int("num_test", num_test, 1, 4)

    supplied_train = num_train is not None
    supplied_test = num_test is not None
    s = int(seed)

    if supplied_train and supplied_test:
        nt = int(num_train)
        ne = int(num_test)
        if nt <= ne:
            raise ValueError("num_train must be greater than num_test")
        return nt, ne

    if not supplied_train and not supplied_test:
        ne = 1 + (s % 4)
        candidate = 2 + ((s // 4) % 9)
        nt = max(candidate, ne + 1)
        return nt, ne

    if not supplied_train:
        ne = int(num_test)
        low = max(2, ne + 1)
        nt = low + (s % (11 - low))
        return nt, ne

    nt = int(num_train)
    high = min(4, nt - 1)
    ne = 1 + (s % high)
    return nt, ne


def _sample_puzzle_parameters(rng):
    colors = [int(x) for x in rng.permutation(10)]
    return {
        "neutral": colors[0],
        "divider": colors[1],
        "band_palette": tuple(colors[2:]),
    }


def _inside(mask, r, c):
    return (
        0 <= r < mask.shape[0]
        and 0 <= c < mask.shape[1]
        and bool(mask[r, c])
    )


def _is_connected(mask):
    starts = np.argwhere(mask)
    if len(starts) == 0:
        return False

    seen = np.zeros(mask.shape, dtype=bool)
    sr, sc = (int(x) for x in starts[0])
    queue = [(sr, sc)]
    seen[sr, sc] = True
    head = 0

    while head < len(queue):
        r, c = queue[head]
        head += 1
        for dr, dc in _DIR4:
            nr, nc = r + dr, c + dc
            if _inside(mask, nr, nc) and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))

    return int(np.count_nonzero(seen)) == int(np.count_nonzero(mask))


def _components(open_mask):
    h, w = open_mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []

    for sr in range(h):
        for sc in range(w):
            if not open_mask[sr, sc] or seen[sr, sc]:
                continue

            queue = [(sr, sc)]
            seen[sr, sc] = True
            coords = []
            head = 0

            while head < len(queue):
                r, c = queue[head]
                head += 1
                coords.append((r, c))

                for dr, dc in _DIR4:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and open_mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            component = np.zeros((h, w), dtype=bool)
            for r, c in coords:
                component[r, c] = True
            result.append(component)

    return result


def _depths(component):
    """Eight-neighbor erosion depth, with off-canvas positions outside."""
    h, w = component.shape
    depth = np.full((h, w), -1, dtype=np.int16)
    queue = []

    for r, c in np.argwhere(component):
        r, c = int(r), int(c)
        boundary = False
        for dr, dc in _DIR8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w and component[nr, nc]):
                boundary = True
                break
        if boundary:
            depth[r, c] = 0
            queue.append((r, c))

    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        next_depth = int(depth[r, c]) + 1

        for dr, dc in _DIR8:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and component[nr, nc]
                and depth[nr, nc] < 0
            ):
                depth[nr, nc] = next_depth
                queue.append((nr, nc))

    assert np.all(depth[component] >= 0)
    return depth


def _candidate_paths(component, depth, allowed_row_direction=None):
    """
    Return convex-corner diagonal paths.

    Each result is (row, col, dr, dc, usable_length), where the position
    at offset d has contour depth d.
    """
    paths = []

    for r, c in np.argwhere(component & (depth == 0)):
        r, c = int(r), int(c)

        for dr in (-1, 1):
            if allowed_row_direction is not None and dr != allowed_row_direction:
                continue

            for dc in (-1, 1):
                # Inward orthogonal and diagonal cells must be in the region.
                if not (
                    _inside(component, r + dr, c)
                    and _inside(component, r, c + dc)
                    and _inside(component, r + dr, c + dc)
                ):
                    continue

                # The two outward orthogonal cells must be outside the region.
                if _inside(component, r - dr, c):
                    continue
                if _inside(component, r, c - dc):
                    continue

                length = 0
                while True:
                    nr, nc = r + length * dr, c + length * dc
                    if not _inside(component, nr, nc):
                        break
                    if int(depth[nr, nc]) != length:
                        break
                    length += 1

                if length >= 1:
                    paths.append((r, c, dr, dc, length))

    return paths


def _choose_path(
    component,
    depth,
    rng,
    needed_length,
    allowed_row_direction=None,
):
    options = [
        p
        for p in _candidate_paths(
            component,
            depth,
            allowed_row_direction=allowed_row_direction,
        )
        if p[4] >= needed_length
    ]
    assert options
    return options[int(rng.integers(0, len(options)))]


def _sample_period(rng, palette, neutral, length, force_gap=False):
    assert 1 <= length <= 4
    ids = rng.choice(len(palette), size=length, replace=False)
    values = [int(palette[int(i)]) for i in np.atleast_1d(ids)]

    if force_gap:
        assert length >= 3
        gap_index = 1 + int(rng.integers(0, length - 2))
        values[gap_index] = int(neutral)
    elif length >= 3 and rng.random() < 0.22:
        gap_index = 1 + int(rng.integers(0, length - 2))
        values[gap_index] = int(neutral)

    assert values[0] != neutral
    assert values[-1] != neutral
    return tuple(values)


def _build_train_mask(rng, pair_index):
    h = int(rng.integers(14, 20))
    w = int(rng.integers(14, 20))
    divider = np.zeros((h, w), dtype=bool)

    if pair_index % 2 == 0:
        # A full vertical divider with one boundary-reaching branch.
        c = int(rng.integers(5, w - 5))
        r = int(rng.integers(5, h - 5))
        divider[:, c] = True

        if rng.random() < 0.5:
            divider[r, c:] = True
        else:
            divider[r, : c + 1] = True
    else:
        # A divider loop attached to a full vertical divider.
        box_width = int(rng.integers(6, min(8, w - 7) + 1))
        box_height = int(rng.integers(6, min(8, h - 6) + 1))
        c = int(rng.integers(5, w - box_width - 1))
        a = int(rng.integers(4, h - box_height - 1))
        b = a + box_height
        d = c + box_width

        divider[:, c] = True
        divider[a, c : d + 1] = True
        divider[b, c : d + 1] = True
        divider[a : b + 1, c] = True
        divider[a : b + 1, d] = True

        if rng.random() < 0.5:
            divider = np.fliplr(divider).copy()

    return divider


def _build_test_mask(rng, pair_index):
    h = int(rng.integers(24, 30))
    w = int(rng.integers(24, 30))
    divider = np.zeros((h, w), dtype=bool)

    r = int(rng.integers(9, h - 9))
    c = int(rng.integers(9, w - 9))
    divider[:, c] = True
    divider[r, :] = True

    quadrants = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    order = [quadrants[int(i)] for i in rng.permutation(4)]
    number_of_boxes = 1 + ((pair_index + int(rng.integers(0, 2))) % 2)

    for vertical_side, horizontal_side in order[:number_of_boxes]:
        if vertical_side < 0:
            b = r - 2
            a = b - 6
        else:
            a = r + 2
            b = a + 6

        if horizontal_side < 0:
            d = c
            left = d - 6
        else:
            left = c
            d = left + 6

        divider[a, left : d + 1] = True
        divider[b, left : d + 1] = True
        divider[a : b + 1, left] = True
        divider[a : b + 1, d] = True

    turns = int(rng.integers(0, 4))
    divider = np.rot90(divider, turns).copy()
    if rng.random() < 0.5:
        divider = np.fliplr(divider).copy()

    return divider


def _build_pair_from_mask(rng, regime, pair_index, parameters, divider_mask):
    neutral = int(parameters["neutral"])
    divider_color = int(parameters["divider"])
    palette = parameters["band_palette"]

    components = _components(~divider_mask)
    depths = [_depths(component) for component in components]
    assignments = []

    if regime == "source_train":
        assert len(components) == 3

        if pair_index == 0:
            # In a single example, visibly establish lengths 3, 2, and 1.
            lengths = (3, 2, 1)
            for i, (component, depth) in enumerate(zip(components, depths)):
                length = lengths[i]
                path = _choose_path(
                    component,
                    depth,
                    rng,
                    needed_length=length,
                    allowed_row_direction=1,
                )
                period = _sample_period(
                    rng,
                    palette,
                    neutral,
                    length,
                    force_gap=(length == 3),
                )
                assignments.append((period, path))
        else:
            # Every puzzle also demonstrates an entirely unmarked region.
            unmarked_index = 1 if pair_index % 2 == 1 else 2

            for i, (component, depth) in enumerate(zip(components, depths)):
                if i == unmarked_index:
                    assignments.append(((neutral,), None))
                    continue

                length = 3 if i == 0 else 2
                path = _choose_path(
                    component,
                    depth,
                    rng,
                    needed_length=length,
                    allowed_row_direction=1,
                )
                period = _sample_period(
                    rng,
                    palette,
                    neutral,
                    length,
                    force_gap=False,
                )
                assignments.append((period, path))

    elif regime == "target_test":
        assert len(components) >= 5

        all_paths = [
            _candidate_paths(component, depth)
            for component, depth in zip(components, depths)
        ]

        upward_options = [
            (i, path)
            for i, paths in enumerate(all_paths)
            for path in paths
            if path[2] == -1 and path[4] >= 3
        ]
        assert upward_options
        upward_index, upward_path = upward_options[
            int(rng.integers(0, len(upward_options)))
        ]

        long_options = [
            (i, path)
            for i, paths in enumerate(all_paths)
            for path in paths
            if i != upward_index and path[4] >= 4
        ]
        assert long_options
        long_index, long_path = long_options[
            int(rng.integers(0, len(long_options)))
        ]

        unmarked_candidates = [
            i
            for i in range(len(components))
            if i not in (upward_index, long_index)
        ]
        assert unmarked_candidates
        unmarked_index = unmarked_candidates[
            int(rng.integers(0, len(unmarked_candidates)))
        ]

        for i, (component, depth) in enumerate(zip(components, depths)):
            if i == upward_index:
                period = _sample_period(
                    rng,
                    palette,
                    neutral,
                    3,
                    force_gap=True,
                )
                assignments.append((period, upward_path))
            elif i == long_index:
                period = _sample_period(
                    rng,
                    palette,
                    neutral,
                    4,
                    force_gap=True,
                )
                assignments.append((period, long_path))
            elif i == unmarked_index:
                assignments.append(((neutral,), None))
            else:
                paths = all_paths[i]
                assert paths
                maximum = min(4, max(path[4] for path in paths))
                length = int(rng.integers(1, maximum + 1))
                path = _choose_path(
                    component,
                    depth,
                    rng,
                    needed_length=length,
                    allowed_row_direction=None,
                )
                period = _sample_period(
                    rng,
                    palette,
                    neutral,
                    length,
                    force_gap=False,
                )
                assignments.append((period, path))
    else:
        raise ValueError("unknown generation regime")

    input_grid = np.full(divider_mask.shape, neutral, dtype=np.int8)
    input_grid[divider_mask] = divider_color

    for period, path in assignments:
        if path is None:
            assert period == (neutral,)
            continue

        r, c, dr, dc, _ = path
        for d, value in enumerate(period):
            if value != neutral:
                input_grid[r + d * dr, c + d * dc] = value

    output_grid = np.full(divider_mask.shape, neutral, dtype=np.int8)
    output_grid[divider_mask] = divider_color

    for component, depth, (period, _) in zip(
        components,
        depths,
        assignments,
    ):
        period_array = np.asarray(period, dtype=np.int8)
        output_grid[component] = period_array[
            depth[component] % len(period)
        ]

    return input_grid, output_grid


def _infer_roles(input_grid):
    counts = np.bincount(input_grid.ravel(), minlength=10)
    order = sorted(range(10), key=lambda color: (-int(counts[color]), color))

    neutral = int(order[0])
    divider = int(order[1])
    third = int(order[2])

    assert counts[neutral] > counts[divider] > 0
    assert counts[divider] > counts[third]
    return neutral, divider


def _decode_region_key(input_grid, component, depth, neutral):
    cue_mask = component & (input_grid != neutral)
    cue_positions = [(int(r), int(c)) for r, c in np.argwhere(cue_mask)]

    if not cue_positions:
        return {
            "period": (int(neutral),),
            "direction": None,
            "anchor": None,
            "marked": False,
            "key_positions": (),
            "component": component,
            "depth": depth,
        }

    anchors = [
        (r, c)
        for r, c in cue_positions
        if int(depth[r, c]) == 0
    ]
    assert len(anchors) == 1
    ar, ac = anchors[0]

    anchor_paths = [
        path
        for path in _candidate_paths(component, depth)
        if path[0] == ar and path[1] == ac
    ]
    assert anchor_paths

    if len(cue_positions) == 1:
        period = (int(input_grid[ar, ac]),)
        assert period[0] != neutral
        return {
            "period": period,
            "direction": None,
            "anchor": (ar, ac),
            "marked": True,
            "key_positions": ((ar, ac),),
            "component": component,
            "depth": depth,
        }

    deepest_cue = max(int(depth[r, c]) for r, c in cue_positions)
    valid_paths = []

    for path in anchor_paths:
        _, _, dr, dc, usable_length = path
        if usable_length < deepest_cue + 1:
            continue

        valid = True
        for r, c in cue_positions:
            d = int(depth[r, c])
            if (r, c) != (ar + d * dr, ac + d * dc):
                valid = False
                break

        if valid:
            valid_paths.append(path)

    directions = {(path[2], path[3]) for path in valid_paths}
    assert len(directions) == 1
    dr, dc = next(iter(directions))

    period = tuple(
        int(input_grid[ar + d * dr, ac + d * dc])
        for d in range(deepest_cue + 1)
    )
    assert period[0] != neutral
    assert period[-1] != neutral

    key_positions = tuple(
        (ar + d * dr, ac + d * dc)
        for d in range(len(period))
    )
    expected_cues = {
        position
        for position, value in zip(key_positions, period)
        if value != neutral
    }
    assert expected_cues == set(cue_positions)

    return {
        "period": period,
        "direction": (dr, dc),
        "anchor": (ar, ac),
        "marked": True,
        "key_positions": key_positions,
        "component": component,
        "depth": depth,
    }


def _edge_touch_count(mask):
    return sum(
        (
            bool(np.any(mask[0, :])),
            bool(np.any(mask[-1, :])),
            bool(np.any(mask[:, 0])),
            bool(np.any(mask[:, -1])),
        )
    )


def _divider_has_cycle(divider_mask):
    nodes = int(np.count_nonzero(divider_mask))
    horizontal_edges = int(
        np.count_nonzero(divider_mask[:, :-1] & divider_mask[:, 1:])
    )
    vertical_edges = int(
        np.count_nonzero(divider_mask[:-1, :] & divider_mask[1:, :])
    )
    return horizontal_edges + vertical_edges >= nodes


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    neutral, divider = _infer_roles(input_grid)
    divider_mask = input_grid == divider

    assert _is_connected(divider_mask)
    assert _edge_touch_count(divider_mask) >= 2

    if input_grid.shape[0] >= 2 and input_grid.shape[1] >= 2:
        solid_blocks = (
            divider_mask[:-1, :-1]
            & divider_mask[1:, :-1]
            & divider_mask[:-1, 1:]
            & divider_mask[1:, 1:]
        )
        assert not np.any(solid_blocks)

    assert np.all(output_grid[divider_mask] == divider)

    components = _components(~divider_mask)
    assert len(components) >= 2

    predicted = np.full(input_grid.shape, neutral, dtype=np.int8)
    predicted[divider_mask] = divider
    region_info = []

    for component in components:
        depth = _depths(component)
        info = _decode_region_key(
            input_grid,
            component,
            depth,
            neutral,
        )
        period = np.asarray(info["period"], dtype=np.int8)
        predicted[component] = period[
            depth[component] % len(period)
        ]
        region_info.append(info)

    assert np.array_equal(predicted, output_grid)
    assert not np.array_equal(input_grid, output_grid)

    # Every non-role output color is supplied by a visible input cue.
    visible_input_colors = {int(x) for x in np.unique(input_grid)}
    for color in (int(x) for x in np.unique(output_grid)):
        if color not in (neutral, divider):
            assert color in visible_input_colors

    return {
        "neutral": neutral,
        "divider": divider,
        "divider_mask": divider_mask,
        "regions": region_info,
        "shape": input_grid.shape,
    }


def _validate_regime(analysis, regime):
    h, w = analysis["shape"]
    regions = analysis["regions"]
    divider_mask = analysis["divider_mask"]

    if regime == "source_train":
        assert 14 <= h <= 19
        assert 14 <= w <= 19
        assert len(regions) == 3

        for region in regions:
            assert len(region["period"]) <= 3
            direction = region["direction"]
            if direction is not None:
                assert direction[0] == 1

    elif regime == "target_test":
        assert 24 <= h <= 29
        assert 24 <= w <= 29
        assert len(regions) >= 5
        assert _edge_touch_count(divider_mask) == 4
        assert _divider_has_cycle(divider_mask)

        assert any(
            region["direction"] is not None
            and region["direction"][0] == -1
            for region in regions
        )
        assert any(
            region["marked"] and len(region["period"]) >= 4
            for region in regions
        )
        assert any(not region["marked"] for region in regions)
    else:
        raise AssertionError("unknown regime")


def _grid_key(grid):
    canonical = grid.astype(np.int8, copy=False)
    return tuple(grid.shape), canonical.tobytes()


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _puzzle_key(puzzle):
    return tuple(
        (
            split,
            tuple(_pair_key(pair) for pair in puzzle[split]),
        )
        for split in ("train", "test")
    )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be nonnegative")

    train_count, test_count = _resolve_counts(
        int(seed),
        num_train,
        num_test,
    )
    rng = np.random.default_rng(int(seed))
    parameters = _sample_puzzle_parameters(rng)

    puzzle = {"train": [], "test": []}
    used_inputs = set()
    used_pairs = set()

    for regime, split, count in (
        ("source_train", "train", train_count),
        ("target_test", "test", test_count),
    ):
        for pair_index in range(count):
            accepted = False

            for _ in range(100):
                child_seed = int(
                    rng.integers(0, np.iinfo(np.int64).max)
                )
                pair_rng = np.random.default_rng(child_seed)

                try:
                    if regime == "source_train":
                        mask = _build_train_mask(pair_rng, pair_index)
                    else:
                        mask = _build_test_mask(pair_rng, pair_index)

                    input_grid, output_grid = _build_pair_from_mask(
                        pair_rng,
                        regime,
                        pair_index,
                        parameters,
                        mask,
                    )

                    candidate = {
                        "input": input_grid,
                        "output": output_grid,
                    }
                    input_key = _grid_key(input_grid)
                    pair_key = _pair_key(candidate)

                    if input_key in used_inputs or pair_key in used_pairs:
                        continue

                    analysis = _analyze_pair(input_grid, output_grid)
                    _validate_regime(analysis, regime)
                except AssertionError:
                    continue

                puzzle[split].append(candidate)
                used_inputs.add(input_key)
                used_pairs.add(pair_key)
                accepted = True
                break

            if not accepted:
                raise ValueError(
                    "unable to construct a unique valid pair for the "
                    "requested puzzle"
                )

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    number_train = len(puzzle["train"])
    number_test = len(puzzle["test"])
    assert 2 <= number_train <= 10
    assert 1 <= number_test <= 4
    assert number_train > number_test

    input_keys = set()
    pair_keys = set()
    train_analyses = []
    test_analyses = []

    for split, regime in (
        ("train", "source_train"),
        ("test", "target_test"),
    ):
        for pair in puzzle[split]:
            assert isinstance(pair, dict)
            assert set(pair.keys()) == {"input", "output"}

            input_grid = pair["input"]
            output_grid = pair["output"]

            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))

            assert input_grid.shape == output_grid.shape

            # Generated dimensions are deliberately disjoint from every
            # source-pair dimension, making exact source copies impossible.
            assert tuple(input_grid.shape) not in _SOURCE_GRID_SHAPES

            input_key = _grid_key(input_grid)
            pair_key = _pair_key(pair)
            assert input_key not in input_keys
            assert pair_key not in pair_keys
            input_keys.add(input_key)
            pair_keys.add(pair_key)

            analysis = _analyze_pair(input_grid, output_grid)
            _validate_regime(analysis, regime)

            if split == "train":
                train_analyses.append(analysis)
            else:
                test_analyses.append(analysis)

    # Neutral and divider identities are the puzzle-level color convention.
    role_pairs = {
        (analysis["neutral"], analysis["divider"])
        for analysis in train_analyses + test_analyses
    }
    assert len(role_pairs) == 1
    neutral, divider = next(iter(role_pairs))
    assert neutral != divider

    train_regions = [
        region
        for analysis in train_analyses
        for region in analysis["regions"]
    ]

    # Training explicitly establishes marked periods of lengths 1, 2, and 3.
    for required_length in (1, 2, 3):
        assert any(
            region["marked"]
            and len(region["period"]) == required_length
            for region in train_regions
        )

    # Training establishes meaningful neutral entries inside a period.
    assert any(
        len(region["period"]) >= 3
        and neutral in region["period"][1:-1]
        for region in train_regions
    )

    # Training also establishes the implicit neutral-only period.
    assert any(not region["marked"] for region in train_regions)

    # At least one multi-cell upper-corner key is explicitly demonstrated.
    assert any(
        region["direction"] is not None
        and region["direction"][0] == 1
        for region in train_regions
    )

    # Test complexity must exceed the fixed three-region training regime.
    assert all(
        len(analysis["regions"]) > 3
        for analysis in test_analyses
    )

    # Recoverability check: every test period was decoded solely from that
    # returned test input, and the independently recomputed output matched.
    for analysis in test_analyses:
        for region in analysis["regions"]:
            if not region["marked"]:
                assert region["period"] == (neutral,)
                continue

            period = region["period"]
            positions = region["key_positions"]
            assert len(period) == len(positions)
            assert period[0] != neutral
            assert period[-1] != neutral


def _puzzles_equal(first, second):
    return _puzzle_key(first) == _puzzle_key(second)


def self_test() -> None:
    generated = []
    train_counts = set()
    test_counts = set()
    complete_signatures = set()
    layout_signatures = set()
    role_assignments = set()
    observed_shapes = set()

    # Forty-two deterministic automatic-count puzzles.
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        generated.append(puzzle)

        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        complete_signatures.add(_puzzle_key(puzzle))

        neutral, divider = _infer_roles(puzzle["train"][0]["input"])
        role_assignments.add((neutral, divider))

        layout_signature = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                input_grid = pair["input"]
                local_neutral, local_divider = _infer_roles(input_grid)
                assert (local_neutral, local_divider) == (
                    neutral,
                    divider,
                )
                divider_mask = (input_grid == divider).astype(np.int8)
                layout_signature.append(_grid_key(divider_mask))
                observed_shapes.add(tuple(input_grid.shape))

                analysis = _analyze_pair(
                    pair["input"],
                    pair["output"],
                )
                _validate_regime(
                    analysis,
                    "source_train" if split == "train"
                    else "target_test",
                )

        layout_signatures.add(tuple(layout_signature))

    assert len(train_counts) >= 4
    assert len(test_counts) >= 3
    assert len(complete_signatures) >= 40
    assert len(layout_signatures) >= 30
    assert len(role_assignments) >= 20
    assert len(observed_shapes) >= 20

    # Whole-puzzle determinism with explicit requested counts.
    first = generate_puzzle(9173, num_train=7, num_test=3)
    second = generate_puzzle(9173, num_train=7, num_test=3)
    assert _puzzles_equal(first, second)

    # Different seeds should normally change the complete puzzle.
    different = generate_puzzle(9174, num_train=7, num_test=3)
    assert not _puzzles_equal(first, different)

    # Legal count boundaries.
    for seed, requested_train, requested_test in (
        (101, 2, 1),
        (102, 10, 1),
        (103, 10, 4),
    ):
        puzzle = generate_puzzle(
            seed,
            num_train=requested_train,
            num_test=requested_test,
        )
        assert len(puzzle["train"]) == requested_train
        assert len(puzzle["test"]) == requested_test
        validate_generated_puzzle(puzzle)

    # Partial count specifications.
    partial_cases = (
        (201, None, 1),
        (202, None, 4),
        (203, 2, None),
        (204, 10, None),
    )
    for seed, requested_train, requested_test in partial_cases:
        puzzle = generate_puzzle(
            seed,
            num_train=requested_train,
            num_test=requested_test,
        )
        if requested_train is not None:
            assert len(puzzle["train"]) == requested_train
        if requested_test is not None:
            assert len(puzzle["test"]) == requested_test
        validate_generated_puzzle(puzzle)

    # Illegal counts and impossible relationships.
    illegal_arguments = (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4},
        {"num_train": True, "num_test": 1},
    )
    for arguments in illegal_arguments:
        raised = False
        try:
            generate_puzzle(300, **arguments)
        except ValueError:
            raised = True
        assert raised

    # Recheck uniqueness, dtype, size, colors, dimensions, and source-copy
    # exclusion independently over all fixed-seed puzzles.
    for puzzle in generated:
        seen_inputs = set()
        seen_pairs = set()

        for split in ("train", "test"):
            for pair in puzzle[split]:
                input_grid = pair["input"]
                output_grid = pair["output"]

                assert input_grid.shape == output_grid.shape
                assert tuple(input_grid.shape) not in _SOURCE_GRID_SHAPES

                for grid in (input_grid, output_grid):
                    assert grid.ndim == 2
                    assert 1 <= grid.shape[0] <= 30
                    assert 1 <= grid.shape[1] <= 30
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert np.all((grid >= 0) & (grid <= 9))

                input_key = _grid_key(input_grid)
                pair_key = _pair_key(pair)
                assert input_key not in seen_inputs
                assert pair_key not in seen_pairs
                seen_inputs.add(input_key)
                seen_pairs.add(pair_key)
