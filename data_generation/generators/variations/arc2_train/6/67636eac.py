from __future__ import annotations

import hashlib

import numpy as np


# SHA-256 over (shape, int8 cells) of the four source inputs. The input alone
# identifies a source pair because the transformation is deterministic.
_SOURCE_INPUT_DIGESTS = {
    "2e354980f0165e3bc0eed10858d3a2706117d460cad90678f2b40f71d9170f7c",
    "adde251fcdd0789a454d5704a9c8fecbd55d6079084e8711d0e5574775e86e04",
    "19449dcf4ea7553bee30d61e72d4231ffdae2a1e6eb44afdd77947375606f7f4",
    "472808563bfe9fc22550261817823be04565b08e2ed765fe39ce1c10917a3e0c",
}
_SIZES = ((2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4))
_SIZE_WEIGHTS = (0.08, 0.10, 0.10, 0.43, 0.10, 0.10, 0.09)


def _check_count(name: str, value: int | None, low: int, high: int) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _select_counts(rng: np.random.Generator, num_train: int | None,
                   num_test: int | None) -> tuple[int, int]:
    _check_count("num_train", num_train, 2, 10)
    _check_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif num_train is None:
        test = int(num_test)
        train = int(rng.integers(max(2, test + 1), 11))
    elif num_test is None:
        train = int(num_train)
        test = int(rng.integers(1, min(4, train - 1) + 1))
    else:
        train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return train, test


def _grid_key(grid: np.ndarray) -> tuple:
    data = grid.astype(np.int8, copy=False)
    return data.shape, data.tobytes()


def _source_digest(grid: np.ndarray) -> str:
    data = grid.astype(np.int8, copy=False)
    return hashlib.sha256(str(data.shape).encode() + data.tobytes()).hexdigest()


def _background(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    assert np.count_nonzero(counts == counts.max()) == 1
    return int(np.argmax(counts))


def _separated(boxes: list[tuple[int, int, int, int]], axis: str) -> bool:
    ix = 0 if axis == "vertical" else 2
    intervals = sorted((box[ix], box[ix + 1]) for box in boxes)
    return all(a[1] < b[0] for a, b in zip(intervals, intervals[1:]))


def _connected(mask: np.ndarray) -> bool:
    points = set(map(tuple, np.argwhere(mask)))
    if not points:
        return False
    reached = {next(iter(points))}
    frontier = list(reached)
    while frontier:
        y, x = frontier.pop()
        for point in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if point in points and point not in reached:
                reached.add(point)
                frontier.append(point)
    return len(reached) == len(points)


def _bounded(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any()
                and mask[:, 0].any() and mask[:, -1].any())


def _sample_mask(rng: np.random.Generator, size: tuple[int, int],
                 connectivity: bool | None = None) -> np.ndarray:
    h, w = size
    area = h * w
    for _ in range(120):
        density = float(rng.uniform(0.30, 0.92))
        mask = rng.random(size) < density
        if (2 <= int(mask.sum()) <= area and _bounded(mask)
                and (connectivity is None or _connected(mask) == connectivity)):
            return mask
    if connectivity is False:
        mask = np.zeros(size, dtype=bool)
        mask[0, 0] = True
        mask[-1, -1] = True
        return mask
    return np.ones(size, dtype=bool)


def _choose_size(rng: np.random.Generator,
                 exclude: tuple[int, int] | None = None) -> tuple[int, int]:
    weights = np.array([weight if size != exclude else 0.0
                        for size, weight in zip(_SIZES, _SIZE_WEIGHTS)])
    weights /= weights.sum()
    return _SIZES[int(rng.choice(len(_SIZES), p=weights))]


def _solve_input(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    background = _background(grid)
    colors = [int(c) for c in np.unique(grid) if int(c) != background]
    assert 2 <= len(colors) <= 4
    objects = []
    common_mask = None
    for color in colors:
        ys, xs = np.where(grid == color)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        tile = grid[y0:y1 + 1, x0:x1 + 1]
        assert np.all((tile == color) | (tile == background))
        mask = tile == color
        assert _bounded(mask)
        if common_mask is None:
            common_mask = mask
        else:
            assert np.array_equal(mask, common_mask)
        objects.append((color, (y0, y1, x0, x1), tile.copy()))
    boxes = [item[1] for item in objects]
    vertical = _separated(boxes, "vertical")
    horizontal = _separated(boxes, "horizontal")
    assert vertical != horizontal, "ordering axis is not unique"
    axis = "vertical" if vertical else "horizontal"
    ix = 0 if vertical else 2
    objects.sort(key=lambda item: item[1][ix])
    output = np.concatenate([item[2] for item in objects], axis=0 if vertical else 1)
    return output, {
        "axis": axis, "count": len(objects), "mask": common_mask.copy(),
        "background": background, "colors": tuple(item[0] for item in objects),
        "boxes": tuple(item[1] for item in objects),
    }


def _sample_positions(rng: np.random.Generator, count: int, along: int,
                      perpendicular: int) -> tuple[list[int], list[int], int, int]:
    lead, trail = (int(v) for v in rng.integers(0, 6, size=2))
    if count * along + lead + trail > 30:
        trail = max(0, 30 - count * along - lead)
    budget = 30 - count * along - lead - trail
    gaps = [int(v) for v in rng.integers(0, 7, size=count - 1)]
    while sum(gaps) > budget:
        largest = max(range(len(gaps)), key=lambda i: gaps[i])
        gaps[largest] -= 1
    along_starts = [lead]
    for gap in gaps:
        along_starts.append(along_starts[-1] + along + gap)
    along_size = along_starts[-1] + along + trail

    if rng.random() < 0.14:
        offsets = [0] * count
    else:
        offsets = [int(v) for v in rng.integers(0, 9, size=count)]
        # Only the *global* perpendicular ordering must fail. The middle
        # object may be far away, as in the second source training example.
        boxes = [(v, v + perpendicular - 1, 0, 0) for v in offsets]
        if _separated(boxes, "vertical"):
            offsets[-1] = offsets[0]
    side_lead, side_trail = (int(v) for v in rng.integers(0, 6, size=2))
    side_starts = [side_lead + v for v in offsets]
    side_size = max(side_starts) + perpendicular + side_trail
    if along_size == count * along and all(v == side_starts[0] for v in side_starts):
        along_size += 1
    return along_starts, side_starts, along_size, side_size


def _make_pair(rng: np.random.Generator, axis: str, count: int,
               mask: np.ndarray, background: int, colors: list[int]) -> dict:
    h, w = mask.shape
    along = w if axis == "horizontal" else h
    side = h if axis == "horizontal" else w
    along_starts, side_starts, along_size, side_size = _sample_positions(
        rng, count, along, side)
    if axis == "horizontal":
        shape = (side_size, along_size)
        positions = list(zip(side_starts, along_starts))
    else:
        shape = (along_size, side_size)
        positions = list(zip(along_starts, side_starts))
    grid = np.full(shape, background, dtype=np.int64)
    tiles = []
    for color, (y, x) in zip(colors, positions):
        tile = np.where(mask, color, background).astype(np.int64)
        grid[y:y + h, x:x + w] = tile
        tiles.append(tile)
    output = np.concatenate(tiles, axis=1 if axis == "horizontal" else 0)
    # The dominant background is observable even for dense, edge-touching
    # motifs. Add empty canvas only when the object colors would tie it.
    while np.count_nonzero(grid == background) <= int(mask.sum()):
        if axis == "horizontal":
            grid = np.pad(grid, ((0, 1), (0, 0)), constant_values=background)
        else:
            grid = np.pad(grid, ((0, 0), (0, 1)), constant_values=background)
    assert all(1 <= d <= 30 for d in grid.shape)
    return {"input": grid, "output": output}


def _choose_colors(rng: np.random.Generator, count: int, background: int,
                   familiar: set[int] | None = None) -> list[int]:
    available = [c for c in range(10) if c != background]
    if familiar is not None and len(set(available) & familiar) >= count:
        available = [c for c in available if c in familiar]
    return [int(v) for v in rng.choice(available, size=count, replace=False)]


def _pair_signature(pair: dict) -> tuple:
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _puzzles_equal(a: dict, b: dict) -> bool:
    return all(len(a[s]) == len(b[s]) and
               all(_pair_signature(x) == _pair_signature(y)
                   for x, y in zip(a[s], b[s])) for s in ("train", "test"))


def _validate_regimes(train: list[dict], test: list[dict]) -> None:
    train_axes = {item["axis"] for item in train}
    assert {item["axis"] for item in test} <= train_axes
    assert len({item["mask"].tobytes() for item in train}) >= 2
    assert len({item["mask"].shape for item in train}) >= 2
    assert any(_connected(item["mask"]) for item in train)
    assert any(not _connected(item["mask"]) for item in train)
    assert all(not any(np.array_equal(item["mask"], example["mask"])
                       for example in train) for item in test)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    seen_inputs, seen_pairs = set(), set()
    information = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            input_key, pair_key = _pair_signature(pair)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _source_digest(pair["input"]) not in _SOURCE_INPUT_DIGESTS
            expected, info = _solve_input(pair["input"])
            assert np.array_equal(pair["output"], expected)
            assert not np.array_equal(pair["input"], pair["output"])
            information[split].append(info)
    _validate_regimes(information["train"], information["test"])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _select_counts(rng, num_train, num_test)
    # The spatial packing rule is the only puzzle-wide convention. A single
    # orientation regime or a mixed one determines the demonstrations.
    mixed = bool(rng.integers(0, 2))
    progression = not mixed and bool(rng.random() < 0.35)
    first_axis = "horizontal" if rng.integers(0, 2) else "vertical"
    second_axis = first_axis
    if mixed:
        second_axis = "vertical" if first_axis == "horizontal" else "horizontal"
    train_axes = [first_axis, second_axis]
    train_axes += [str(rng.choice(train_axes if not mixed else
                                  ("horizontal", "vertical")))
                   for _ in range(ntrain - 2)]
    train_sizes = [_choose_size(rng)]
    train_sizes.append(_choose_size(rng, train_sizes[0]))
    train_sizes += [_choose_size(rng) for _ in range(ntrain - 2)]
    first_count = int(rng.choice((2, 3, 4), p=(0.39, 0.39, 0.22)))
    if progression:
        train_counts = [2 + index % 3 for index in range(ntrain)]
    else:
        train_counts = [first_count]
        train_counts.append(int(rng.choice(
            [count for count in (2, 3, 4) if count != first_count])))
        train_counts += [int(rng.choice((2, 3, 4), p=(0.39, 0.39, 0.22)))
                         for _ in range(ntrain - 2)]
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=ntrain + ntest, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    train_masks = []
    training_colors = set()
    first_background = None
    for index in range(ntrain + ntest):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        training = index < ntrain
        if training:
            axis = train_axes[index]
            count = train_counts[index]
            size = train_sizes[index]
            connectivity = True if index == 0 else False if index == 1 else None
        else:
            axis = str(pair_rng.choice(tuple(sorted(set(train_axes)))))
            count = (4 if progression and pair_rng.random() < 0.7 else
                     int(pair_rng.choice((2, 3, 4), p=(0.20, 0.30, 0.50))))
            size = _choose_size(pair_rng)
            connectivity = None
        for attempt in range(40):
            mask = _sample_mask(pair_rng, size, connectivity)
            if training or not any(np.array_equal(mask, old) for old in train_masks):
                break
            size = _SIZES[(index + attempt) % len(_SIZES)]
        else:
            raise AssertionError("could not select a new test motif")
        if training:
            train_masks.append(mask)
        for _ in range(20):
            background = int(pair_rng.integers(0, 10))
            if index != 1 or background != first_background:
                break
        if index == 0:
            first_background = background
        familiar = (training_colors if not training and pair_rng.random() < 0.5
                    else None)
        colors = _choose_colors(pair_rng, count, background, familiar)
        if training:
            training_colors.update(colors)
        for _ in range(20):
            pair = _make_pair(pair_rng, axis, count, mask, background, colors)
            key = _grid_key(pair["input"])
            if key not in seen_inputs and _source_digest(pair["input"]) not in _SOURCE_INPUT_DIGESTS:
                break
        else:
            raise AssertionError("could not construct a distinct pair")
        seen_inputs.add(key)
        puzzle["train" if training else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _expect_value_error(train: int | None, test: int | None) -> None:
    try:
        generate_puzzle(19, train, test)
    except ValueError:
        return
    raise AssertionError("illegal counts were accepted")


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    train_counts, test_counts = set(), set()
    all_axes_counts = set()
    masks, sizes, placements, colors, backgrounds = set(), set(), set(), set(), set()
    connected_splits = {"train": set(), "test": set()}
    motif_sizes = {"train": set(), "test": set()}
    densities = {"train": set(), "test": set()}
    all_motifs = []
    gap_values, side_spans, edge_margins = set(), set(), set()
    aligned = wide_shift = zero_gap = familiar_test = False
    familiar_count = four_train = vertical_four_test = 0
    novel_size_tests = novel_color_tests = progression_puzzles = 0
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_colors = set()
        puzzle_backgrounds = set()
        train_counts_here = set()
        train_sizes_here = set()
        train_info = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                rebuilt, info = _solve_input(pair["input"])
                assert np.array_equal(rebuilt, pair["output"])
                all_axes_counts.add((split, info["axis"], info["count"]))
                masks.add((info["mask"].shape, info["mask"].tobytes()))
                all_motifs.append(info["mask"])
                motif_sizes[split].add(info["mask"].shape)
                densities[split].add(int(info["mask"].sum()))
                sizes.add(pair["input"].shape)
                colors.add(info["colors"])
                backgrounds.add(info["background"])
                puzzle_backgrounds.add(info["background"])
                connected_splits[split].add(_connected(info["mask"]))
                placements.add((info["axis"], info["count"],
                                pair["input"].shape, info["boxes"]))
                boxes = info["boxes"]
                ix = 0 if info["axis"] == "vertical" else 2
                ordered = sorted(boxes, key=lambda box: box[ix])
                gaps = [b[ix] - a[ix + 1] - 1 for a, b in zip(ordered, ordered[1:])]
                gap_values.update(gaps)
                zero_gap |= 0 in gaps
                side = 2 if ix == 0 else 0
                span = max(box[side] for box in boxes) - min(box[side] for box in boxes)
                side_spans.add(span)
                aligned |= span == 0
                wide_shift |= span >= 3
                edge_margins.add(min(box[0] for box in boxes))
                edge_margins.add(min(box[2] for box in boxes))
                if split == "train":
                    train_colors.update(info["colors"])
                    train_counts_here.add(info["count"])
                    train_sizes_here.add(info["mask"].shape)
                    train_info.append(info)
                    four_train += info["count"] == 4
                else:
                    familiar_test |= set(info["colors"]) <= train_colors
                    familiar_count += info["count"] in train_counts_here
                    novel_color_tests += not set(info["colors"]) <= train_colors
                    novel_size_tests += info["mask"].shape not in train_sizes_here
                    vertical_four_test += info["axis"] == "vertical" and info["count"] == 4
        assert len(puzzle_backgrounds) >= 2
        assert len(train_counts_here) >= 2 and len(train_sizes_here) >= 2
        progression_puzzles += (
            len(train_info) >= 3
            and len({item["axis"] for item in train_info}) == 1
            and [item["count"] for item in train_info[:3]] == [2, 3, 4]
        )
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len({_pair_signature(p["train"][0]) for p in automatic}) == 42
    assert len(masks) >= 35 and len(sizes) >= 45 and len(placements) >= 150
    assert len(colors) >= 60 and len(backgrounds) >= 8
    assert connected_splits == {"train": {False, True}, "test": {False, True}}
    assert all(len(motif_sizes[split]) == 7 for split in ("train", "test"))
    assert all(min(densities[split]) <= 3 and max(densities[split]) >= 9
               for split in ("train", "test"))
    assert sum(not np.array_equal(mask, np.fliplr(mask))
               for mask in all_motifs) >= 100
    assert sum((mask.shape, np.fliplr(mask).tobytes()) in masks
               and not np.array_equal(mask, np.fliplr(mask))
               for mask in all_motifs) >= 20
    assert len(all_axes_counts) == 12
    assert zero_gap and wide_shift and aligned
    assert max(gap_values) >= 4 and max(side_spans) >= 5
    assert 0 in edge_margins and max(edge_margins) >= 4
    assert familiar_test and familiar_count >= 10
    assert four_train >= 10 and vertical_four_test >= 5
    assert novel_size_tests >= 10 and novel_color_tests >= 5
    assert progression_puzzles >= 2
    assert _puzzles_equal(automatic[7], generate_puzzle(7))
    assert not _puzzles_equal(automatic[7], automatic[8])
    for seed, train, test in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                              (201, None, 1), (202, None, 4),
                              (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    assert _puzzles_equal(generate_puzzle(9001, 7, 3), generate_puzzle(9001, 7, 3))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4), (True, 1), (3, False)):
        _expect_value_error(train, test)
