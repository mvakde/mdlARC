import numpy as np


_EDGES = ("top", "bottom", "left", "right")

_SOURCE_INPUTS = frozenset(
    text.strip().replace("\n", "/")
    for text in (
        """000000000000
003000000000
000000000000
003000000000
000000007000
000000006000
000000000000
004000000000
000000008000
000000000000
000000000000
005000005000""",
        """8000000000000
0000000010006
0008008000020
0070000000000
0000000060400
0000000000000
0000400600000
0000000001000
0000003000000
0300000000000
0000800060002
0000000000000
0000500050000""",
        """0000000
0000000
0000000
0006000
0000000
0000000
0005000""",
        """04008000000000000
00000900070000040
00000100000000000
00300000004000000
00000000000000900
00000000000000007
70000009000000009
00030000000000000
02000000003000040
60000000000000000
00000200000000000
00000000004000006
00000000000200000
08000200007000600
00600000040000000
00000000000000000
00000500050000500""",
        """00002309040003008
00300044000000000
00000000000000000
00000000000000000
00000001800094900
00060000101060007
00000800001300000
00007030030000400
00400000000000000
00020019000700120
00030000000000000
04040000100000000
00000000009000300
00000003000002000
00010000600490303
00000000000000100
00000500500500000""",
    )
)


def _validate_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _validate_count(num_train, "num_train", 2, 10)
    num_test = _validate_count(num_test, "num_test", 1, 4)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))

    if num_train <= num_test:
        raise ValueError("could not choose legal pair counts")

    return num_train, num_test


def _sample_puzzle_parameters(rng, num_train, num_test):
    marker = int(rng.integers(0, 10))
    novel_color = None
    if rng.random() < 0.28:
        novel_color = int(rng.choice([color for color in range(1, 10) if color != marker]))
    policy = str(rng.choice(["fixed", "mixed", "heldout"], p=[0.38, 0.43, 0.19]))
    edges = list(_EDGES)
    rng.shuffle(edges)
    if policy == "fixed":
        train_edges = [edges[0]] * num_train
        test_edges = [edges[0]] * num_test
    elif policy == "heldout":
        train_edges = [edges[0], edges[1]]
        train_edges += [edges[int(rng.integers(0, 2))] for _ in range(num_train - 2)]
        test_edges = [edges[int(rng.integers(2, 4))] for _ in range(num_test)]
    else:
        train_edges = [edges[0], edges[1]]
        train_edges += [edges[int(rng.integers(0, 4))] for _ in range(num_train - 2)]
        test_edges = [edges[int(rng.integers(0, 4))] for _ in range(num_test)]
    rng.shuffle(train_edges)
    return marker, train_edges, test_edges, novel_color


def _orient_from_bottom(grid, edge):
    if edge == "bottom":
        return grid.copy()
    if edge == "top":
        return np.flipud(grid).copy()
    if edge == "right":
        return grid.T.copy()
    if edge == "left":
        return np.fliplr(grid.T).copy()
    raise ValueError("unknown edge")


def _canonicalize(grid, edge):
    if edge == "bottom":
        return grid
    if edge == "top":
        return np.flipud(grid)
    if edge == "right":
        return grid.T
    if edge == "left":
        return np.fliplr(grid).T
    raise ValueError("unknown edge")


def _background(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _anchor_candidates(grid, marker, background):
    found = set()
    for edge in _EDGES:
        canonical = _canonicalize(grid, edge)
        boundary = canonical[-1]
        lanes = np.flatnonzero(boundary == marker)
        if not len(lanes):
            continue
        if np.any((boundary != marker) & (boundary != background)):
            continue
        if len(lanes) == 1 and int(lanes[0]) in (0, canonical.shape[1] - 1):
            continue
        found.add(edge)
    return found


def _apply_rule(grid, edge, marker, background):
    canonical = _canonicalize(grid, edge)
    output = canonical.copy()
    for lane in np.flatnonzero(canonical[-1] == marker):
        color = marker
        for row in range(canonical.shape[0] - 1, -1, -1):
            if canonical[row, lane] == background:
                output[row, lane] = color
            else:
                color = int(canonical[row, lane])
    return _orient_from_bottom(output, edge)


def _independent_prediction(grid, edge, marker, background):
    """Assign each blank from the nearest original seed toward the boundary."""
    canonical = _canonicalize(grid, edge)
    output = canonical.copy()
    for lane in np.flatnonzero(canonical[-1] == marker):
        seeds = np.flatnonzero(canonical[:, lane] != background)
        for row in np.flatnonzero(canonical[:, lane] == background):
            toward_anchor = seeds[seeds > row]
            output[row, lane] = canonical[int(toward_anchor[0]), lane]
    return _orient_from_bottom(output, edge)


def _sample_dimension(rng, regime):
    if regime == "source_train":
        band = int(rng.choice(3, p=[0.27, 0.48, 0.25]))
    else:
        band = int(rng.choice(3, p=[0.17, 0.38, 0.45]))
    low, high = ((7, 11), (12, 18), (19, 30))[band]
    return int(rng.integers(low, high + 1))


def _sample_lane_count(rng, width, showcase):
    upper = min(6, width - 1)
    counts = np.arange(2 if showcase else 1, upper + 1)
    weights = np.array([3 if n in (1, 2, 3) else 2 if n == 4 else 1 for n in counts], dtype=float)
    return int(rng.choice(counts, p=weights / weights.sum()))


def _seed_count(rng, depth):
    maximum = min(9, depth - 1)
    band = int(rng.choice(3, p=[0.44, 0.40, 0.16]))
    low, high = ((0, 2), (3, 5), (6, 9))[band]
    low, high = min(low, maximum), min(high, maximum)
    return int(rng.integers(low, high + 1))


def _clutter_density(rng, regime):
    if regime == "source_train":
        band = int(rng.choice(4, p=[0.21, 0.34, 0.28, 0.17]))
    else:
        band = int(rng.choice(4, p=[0.13, 0.23, 0.31, 0.33]))
    if band == 0:
        return 0.0
    low, high = ((0, 0), (0.01, 0.05), (0.05, 0.12), (0.12, 0.23))[band]
    return float(rng.uniform(low, high))


def _add_distractors(grid, lanes, palette, rng, density, background, showcase):
    mask = np.ones(grid.shape, dtype=bool)
    mask[:, lanes] = False
    mask[-1, :] = False
    eligible = np.argwhere(mask)
    count = int(round(density * len(eligible)))
    if showcase:
        count = max(1, count)
    count = min(count, len(eligible))
    if not count:
        return
    chosen = []
    outside = [column for column in range(grid.shape[1]) if column not in set(lanes)]
    if outside:
        lane = int(rng.choice(lanes))
        seed_rows = np.flatnonzero(grid[:-1, lane] != background)
        if len(seed_rows):
            chosen.append((int(rng.choice(seed_rows)), int(rng.choice(outside))))
        if count >= 3:
            column = int(rng.choice(outside))
            rows = rng.choice(grid.shape[0] - 1, size=min(3, grid.shape[0] - 1), replace=False)
            chosen.extend((int(row), column) for row in rows)
    chosen = list(dict.fromkeys(chosen))[:count]
    available = [tuple(map(int, point)) for point in eligible if tuple(map(int, point)) not in chosen]
    if count > len(chosen):
        indexes = rng.choice(len(available), size=count - len(chosen), replace=False)
        chosen.extend(available[int(index)] for index in indexes)
    for row, column in chosen:
        grid[row, column] = int(rng.choice(palette))


def _generate_pair(rng, regime, marker, edge, showcase, requested_shape=None, forbidden_color=None, forced_seed_color=None):
    if requested_shape is None:
        depth = _sample_dimension(rng, regime)
        width = _sample_dimension(rng, regime)
    elif edge in ("top", "bottom"):
        depth, width = requested_shape
    else:
        width, depth = requested_shape
    depth = max(7, depth)
    lane_count = _sample_lane_count(rng, width, showcase)
    if lane_count == 1:
        lanes = np.array([int(rng.integers(1, width - 1))])
    else:
        lanes = np.sort(rng.choice(width, size=lane_count, replace=False))
    if marker != 0 and rng.random() < 0.64:
        background = 0
    else:
        background = int(rng.choice([c for c in range(10) if c != marker and c != forbidden_color and c != forced_seed_color]))
    available = [c for c in range(10) if c != background and c != marker and c != forbidden_color]
    size = int(rng.integers(2 if showcase else 1, len(available) + 1))
    palette = list(map(int, rng.choice(available, size=size, replace=False)))
    if rng.random() < 0.25:
        palette.append(marker)
    if forced_seed_color is not None and forced_seed_color not in palette:
        palette.append(forced_seed_color)
    canonical = np.full((depth, width), background, dtype=np.int8)
    canonical[-1, lanes] = marker
    for index, lane in enumerate(lanes):
        if showcase and index == 0:
            near = int(rng.integers(2, max(3, depth // 3 + 1)))
            far = int(rng.integers(near + 2, depth))
            distances = [near, far]
            first, second = rng.choice([c for c in palette if c != marker], size=2, replace=False)
            values = [int(first), int(second)]
        else:
            count = _seed_count(rng, depth)
            distances = list(map(int, rng.choice(np.arange(1, depth), size=count, replace=False)))
            if count and rng.random() < 0.24:
                values = [int(rng.choice(palette))] * count
            else:
                values = list(map(int, rng.choice(palette, size=count, replace=True)))
        for distance, value in zip(distances, values):
            canonical[depth - 1 - distance, lane] = value
    if forced_seed_color is not None:
        distance = int(rng.integers(2, depth - 1))
        lane = int(rng.choice(lanes))
        canonical[depth - 1 - distance, lane] = forced_seed_color
        canonical[depth - 2 - distance, lane] = background
    density = _clutter_density(rng, regime)
    _add_distractors(canonical, lanes, palette, rng, density, background, showcase)
    input_grid = _orient_from_bottom(canonical, edge)
    output_grid = _apply_rule(input_grid, edge, marker, background)
    return input_grid, output_grid


def _grid_key(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes()


def _compact_grid(grid):
    return "/".join("".join(str(int(value)) for value in row) for row in grid)


def _pair_conventions(input_grid, output_grid):
    background = _background(input_grid)
    candidates = set()
    for marker in range(10):
        if marker == background:
            continue
        for edge in _anchor_candidates(input_grid, marker, background):
            predicted = _independent_prediction(input_grid, edge, marker, background)
            if np.array_equal(predicted, output_grid) and not np.array_equal(input_grid, output_grid):
                candidates.add((edge, marker, background))
    return candidates


def _pair_metrics(input_grid, output_grid, edge, marker, background):
    canonical = _canonicalize(input_grid, edge)
    result = _canonicalize(output_grid, edge)
    depth, width = canonical.shape
    lanes = np.flatnonzero(canonical[-1] == marker)
    assert len(lanes) and len(lanes) < width
    assert np.all((canonical[-1] == background) | (canonical[-1] == marker))
    assert np.array_equal(_independent_prediction(input_grid, edge, marker, background), output_grid)
    assert np.array_equal(result[canonical != background], canonical[canonical != background])
    assert np.all(result[:, lanes] != background)
    outside = np.ones(width, dtype=bool)
    outside[lanes] = False
    assert np.array_equal(result[:, outside], canonical[:, outside])
    seed_rows = [np.flatnonzero(canonical[:-1, lane] != background) for lane in lanes]
    seed_counts = tuple(len(rows) for rows in seed_rows)
    adjacent = any(np.any(np.diff(rows) == 1) for rows in seed_rows)
    far_seed = any(0 in rows for rows in seed_rows)
    near_seed = any(depth - 2 in rows for rows in seed_rows)
    uniform = any(len(rows) == 0 or len(set(int(canonical[r, lane]) for r in rows)) == 1 for lane, rows in zip(lanes, seed_rows))
    aligned = any(canonical[int(row), column] != background for lane, rows in zip(lanes, seed_rows) for row in rows for column in range(width) if column not in lanes)
    eligible = (depth - 1) * int(outside.sum())
    distractors = int(np.count_nonzero(canonical[:-1, outside] != background))
    palette_size = len(np.unique(input_grid))
    return {
        "edge": edge, "marker": marker, "background": background,
        "shape": tuple(input_grid.shape), "depth": depth, "width": width,
        "lanes": tuple(map(int, lanes)), "seed_counts": seed_counts,
        "adjacent": adjacent, "far_seed": far_seed, "near_seed": near_seed,
        "uniform": uniform, "aligned": aligned,
        "distractors": distractors, "density": distractors / eligible,
        "palette_size": palette_size,
        "colors": frozenset(map(int, np.unique(input_grid))),
    }


def _informative(input_grid, metrics):
    canonical = _canonicalize(input_grid, metrics["edge"])
    background = metrics["background"]
    for lane in metrics["lanes"]:
        rows = np.flatnonzero(canonical[:, lane] != background)
        for near, far in zip(rows[::-1], rows[-2::-1]):
            if near - far >= 2 and canonical[near, lane] != canonical[far, lane]:
                if near < canonical.shape[0] - 1 and canonical.shape[0] - 1 - near >= 2:
                    return True
    return False


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    metrics = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for array in (grid, output):
                assert isinstance(array, np.ndarray) and array.ndim == 2
                assert all(1 <= length <= 30 for length in array.shape)
                assert np.issubdtype(array.dtype, np.integer)
                assert np.all((0 <= array) & (array <= 9))
            assert grid.shape == output.shape and not np.array_equal(grid, output)
            assert _compact_grid(grid) not in _SOURCE_INPUTS
            key = _grid_key(grid)
            assert key not in seen
            seen.add(key)
            candidates = _pair_conventions(grid, output)
            assert len(candidates) == 1, candidates
            edge, marker, background = next(iter(candidates))
            assert _anchor_candidates(grid, marker, background) == {edge}
            metrics[split].append(_pair_metrics(grid, output, edge, marker, background))
    markers = {m["marker"] for split in metrics.values() for m in split}
    assert len(markers) == 1
    assert any(_informative(pair["input"], m) and m["aligned"] for pair, m in zip(puzzle["train"], metrics["train"]))
    train_edges = {m["edge"] for m in metrics["train"]}
    test_edges = {m["edge"] for m in metrics["test"]}
    if not test_edges.issubset(train_edges):
        assert len(train_edges) >= 2
    # The shared boundary color plus each test input uniquely identifies its anchor.
    for pair, m in zip(puzzle["test"], metrics["test"]):
        inferred_background = _background(pair["input"])
        assert inferred_background == m["background"]
        assert _anchor_candidates(pair["input"], m["marker"], inferred_background) == {m["edge"]}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    marker, train_edges, test_edges, novel_color = _sample_puzzle_parameters(rng, num_train, num_test)
    showcase_index = int(rng.integers(0, num_train))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, regime, edges in (("train", "source_train", train_edges), ("test", "target_test", test_edges)):
        for index, edge in enumerate(edges):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            requested_shape = None
            if split == "test" and rng.random() < 0.36:
                example = puzzle["train"][int(rng.integers(0, num_train))]
                requested_shape = example["input"].shape
            for attempt in range(100):
                grid, output = _generate_pair(
                    pair_rng, regime, marker, edge, split == "train" and index == showcase_index,
                    requested_shape, novel_color if split == "train" else None,
                    novel_color if split == "test" and index == 0 else None,
                )
                key = _grid_key(grid)
                if key in seen or _compact_grid(grid) in _SOURCE_INPUTS:
                    continue
                candidates = _pair_conventions(grid, output)
                if candidates != {(edge, marker, _background(grid))}:
                    continue
                if _anchor_candidates(grid, marker, _background(grid)) != {edge}:
                    continue
                break
            else:
                raise RuntimeError("could not construct an unambiguous pair")
            seen.add(key)
            puzzle[split].append({"input": grid, "output": output})
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(np.array_equal(a[name], b[name]) for split in ("train", "test") for a, b in zip(first[split], second[split]) for name in ("input", "output")) and all(len(first[split]) == len(second[split]) for split in ("train", "test"))


def _expect_value_error(**kwargs):
    try:
        generate_puzzle(918273, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {kwargs}")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzles_equal(generate_puzzle(12345), generate_puzzle(12345))
    assert not _puzzles_equal(generate_puzzle(12345), generate_puzzle(12346))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(30000 + 10 * train_count + test_count, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
    for index, kwargs in enumerate(({"num_train": None, "num_test": 1}, {"num_train": None, "num_test": 4}, {"num_train": 2, "num_test": None}, {"num_train": 10, "num_test": None})):
        puzzle = generate_puzzle(31000 + index, **kwargs)
        for name, split in (("num_train", "train"), ("num_test", "test")):
            if kwargs[name] is not None:
                assert len(puzzle[split]) == kwargs[name]
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}, {"num_train": 2.0, "num_test": 1}):
        _expect_value_error(**kwargs)
    all_metrics = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                edge, marker, background = next(iter(_pair_conventions(pair["input"], pair["output"])))
                assert np.array_equal(_independent_prediction(pair["input"], edge, marker, background), pair["output"])
                all_metrics.append(_pair_metrics(pair["input"], pair["output"], edge, marker, background))
    assert {m["edge"] for m in all_metrics} == set(_EDGES)
    assert len({m["marker"] for m in all_metrics}) >= 6
    assert len({m["shape"] for m in all_metrics}) >= 60
    assert len({(m["shape"], m["lanes"], m["seed_counts"]) for m in all_metrics}) >= 150
    assert {len(m["lanes"]) for m in all_metrics} >= {1, 2, 3, 4}
    assert any(m["adjacent"] for m in all_metrics)
    assert any(m["far_seed"] for m in all_metrics)
    assert any(m["near_seed"] for m in all_metrics)
    assert any(m["distractors"] == 0 for m in all_metrics)
    assert any(m["density"] > 0.12 for m in all_metrics)
    assert any(m["background"] != 0 for m in all_metrics)
    assert min(m["palette_size"] for m in all_metrics) <= 3
    assert max(m["palette_size"] for m in all_metrics) >= 8
    review = {
        "familiar_size": 0, "size_transfer": 0, "new_test_color": 0,
        "fixed_edges": 0, "mixed_edges": 0, "heldout_edge": 0,
        "background_change": 0, "lane_change": 0,
        "sparse_to_dense": 0, "dense_to_simpler": 0,
    }
    properties = {
        "clean": 0, "light": 0, "dense": 0, "adjacent_seed": 0,
        "far_seed": 0, "near_seed": 0, "terminal_only": 0,
        "uniform_interior": 0, "same_sequence": 0,
        "adjacent_lanes": 0, "separated_lanes": 0, "corner_lane": 0,
        "aligned_distractor": 0, "plausible_unmarked_lane": 0,
        "marker_inside": 0, "zero_as_seed_or_distractor": 0,
        "combined": 0,
    }
    for puzzle in puzzles:
        pair_metrics = {"train": [], "test": []}
        train_markers = {
            next(iter(_pair_conventions(pair["input"], pair["output"])))[1]
            for pair in puzzle["train"]
        }
        assert len(train_markers) == 1
        marker = next(iter(train_markers))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                bg = _background(grid)
                edges = _anchor_candidates(grid, marker, bg)
                assert len(edges) == 1
                edge = next(iter(edges))
                assert np.array_equal(
                    _independent_prediction(grid, edge, marker, bg),
                    pair["output"],
                )
                metric = _pair_metrics(grid, pair["output"], edge, marker, bg)
                pair_metrics[split].append(metric)
                canonical = _canonicalize(grid, edge)
                lanes = metric["lanes"]
                rows = [np.flatnonzero(canonical[:-1, lane] != bg) for lane in lanes]
                sequences = [tuple(int(canonical[row, lane]) for row in selected[::-1]) for lane, selected in zip(lanes, rows)]
                outside = [column for column in range(canonical.shape[1]) if column not in lanes]
                properties["clean"] += metric["distractors"] == 0
                properties["light"] += 0 < metric["density"] < 0.05
                properties["dense"] += metric["density"] >= 0.12
                properties["adjacent_seed"] += metric["adjacent"]
                properties["far_seed"] += metric["far_seed"]
                properties["near_seed"] += metric["near_seed"]
                properties["terminal_only"] += 0 in metric["seed_counts"]
                properties["uniform_interior"] += any(len(selected) >= 2 and len(set(int(canonical[row, lane]) for row in selected)) == 1 for lane, selected in zip(lanes, rows))
                properties["same_sequence"] += len(sequences) >= 2 and len(set(sequences)) < len(sequences)
                properties["adjacent_lanes"] += any(b - a == 1 for a, b in zip(lanes, lanes[1:]))
                properties["separated_lanes"] += any(b - a >= 3 for a, b in zip(lanes, lanes[1:]))
                properties["corner_lane"] += 0 in lanes or canonical.shape[1] - 1 in lanes
                properties["aligned_distractor"] += metric["aligned"]
                properties["plausible_unmarked_lane"] += any(np.count_nonzero(canonical[:-1, column] != bg) >= 2 for column in outside)
                properties["marker_inside"] += np.any(canonical[:-1] == marker)
                properties["zero_as_seed_or_distractor"] += bg != 0 and np.any(canonical != bg) and np.any(canonical == 0)
                properties["combined"] += bg != 0 and metric["adjacent"] and metric["density"] >= 0.05
        train, test = pair_metrics["train"], pair_metrics["test"]
        train_shapes = {m["shape"] for m in train}
        test_shapes = {m["shape"] for m in test}
        train_edges = {m["edge"] for m in train}
        test_edges = {m["edge"] for m in test}
        train_colors = set().union(*(m["colors"] for m in train))
        test_colors = set().union(*(m["colors"] for m in test))
        review["familiar_size"] += bool(train_shapes & test_shapes)
        review["size_transfer"] += bool(test_shapes - train_shapes)
        review["new_test_color"] += bool(test_colors - train_colors)
        review["fixed_edges"] += len(train_edges | test_edges) == 1
        review["mixed_edges"] += len(train_edges | test_edges) > 1
        review["heldout_edge"] += bool(test_edges - train_edges)
        review["background_change"] += len({m["background"] for m in train + test}) > 1
        review["lane_change"] += len({len(m["lanes"]) for m in train + test}) > 1
        review["sparse_to_dense"] += max(m["density"] for m in test) > max(m["density"] for m in train) + 0.05
        review["dense_to_simpler"] += max(m["density"] for m in train) > max(m["density"] for m in test) + 0.05
    assert all(value > 0 for value in review.values()), review
    assert all(value > 0 for value in properties.values()), properties
    assert {len(m["lanes"]) for puzzle in puzzles for pair in puzzle["test"] for m in [_pair_metrics(pair["input"], pair["output"], *next(iter(_pair_conventions(pair["input"], pair["output"]))))]} >= {1, 2, 3, 4}
