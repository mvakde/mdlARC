import numpy as np


_AUTO_COUNT_OPTIONS = (
    (2, 1),
    (3, 1),
    (4, 2),
    (5, 1),
    (5, 2),
    (6, 3),
    (7, 1),
    (7, 2),
    (8, 3),
    (9, 4),
    (10, 1),
    (10, 4),
)


_SOURCE_PAIR_ROWS = frozenset({
    (
        (
            "000008000000",
            "000008000000",
            "000008000000",
            "001111111000",
            "001008001000",
            "001008001000",
            "881888881888",
            "001008001000",
            "001008001000",
            "001111111000",
            "000008000000",
            "000008000000",
        ),
        (
            "000000008000",
            "000000008000",
            "000000008000",
            "881111111888",
            "001000001000",
            "001000001000",
            "001000001000",
            "001000001000",
            "001000001000",
            "001111111000",
            "000000008000",
            "000000008000",
        ),
    ),
    (
        (
            "000030000000",
            "000030000000",
            "002222220000",
            "002030020000",
            "002030020000",
            "332333323333",
            "002030020000",
            "002222220000",
            "000030000000",
            "000030000000",
            "000030000000",
            "000030000000",
        ),
        (
            "000000030000",
            "000000030000",
            "332222223333",
            "002000020000",
            "002000020000",
            "002000020000",
            "002000020000",
            "002222220000",
            "000000030000",
            "000000030000",
            "000000030000",
            "000000030000",
        ),
    ),
    (
        (
            "000000400000",
            "000000400000",
            "000000400000",
            "000033333000",
            "000030403000",
            "444434443444",
            "000030403000",
            "000033333000",
            "000000400000",
            "000000400000",
            "000000400000",
            "000000400000",
        ),
        (
            "000000004000",
            "000000004000",
            "000000004000",
            "444433333444",
            "000030003000",
            "000030003000",
            "000030003000",
            "000033333000",
            "000000004000",
            "000000004000",
            "000000004000",
            "000000004000",
        ),
    ),
    (
        (
            "000000600000",
            "000000600000",
            "007777777700",
            "007000600700",
            "007000600700",
            "667666666766",
            "007000600700",
            "007000600700",
            "007000600700",
            "007777777700",
            "000000600000",
            "000000600000",
        ),
        (
            "000000000600",
            "000000000600",
            "667777777766",
            "007000000700",
            "007000000700",
            "007000000700",
            "007000000700",
            "007000000700",
            "007000000700",
            "007777777700",
            "000000000600",
            "000000000600",
        ),
    ),
})


def _coerce_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")

    value = int(seed)
    if value < 0:
        raise ValueError("seed must be a non-negative integer")
    return value


def _check_optional_count(
    value: int | None,
    name: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")

    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _check_optional_count(
        num_train, "num_train", 2, 10
    )
    test_count = _check_optional_count(
        num_test, "num_test", 1, 4
    )

    if train_count is None and test_count is None:
        return _AUTO_COUNT_OPTIONS[seed % len(_AUTO_COUNT_OPTIONS)]

    if train_count is None:
        assert test_count is not None
        lowest_train = test_count + 1
        number_of_choices = 10 - lowest_train + 1
        train_count = lowest_train + seed % number_of_choices

    if test_count is None:
        highest_test = min(4, train_count - 1)
        test_count = 1 + seed % highest_test

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _rows_signature(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.reshape(-1)),
    )


def _pair_signature(pair: dict) -> tuple:
    return (
        _grid_signature(pair["input"]),
        _grid_signature(pair["output"]),
    )


def _is_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    signature = (
        _rows_signature(input_grid),
        _rows_signature(output_grid),
    )
    return signature in _SOURCE_PAIR_ROWS


_CORNERS = ("UL", "UR", "LL", "LR")


def _sample_colors(rng, fixed_background=None):
    palette = [int(x) for x in rng.permutation(10)]
    if fixed_background is None:
        return tuple(palette[:3])
    others = [color for color in palette if color != fixed_background]
    return fixed_background, others[0], others[1]


def _sample_size(rng):
    band = int(rng.choice(4, p=(0.14, 0.43, 0.25, 0.18)))
    low, high = ((3, 4), (5, 9), (10, 13), (14, 18))[band]
    return int(rng.integers(low, high + 1))


def _sample_frame_dims(rng, shape=None, target=None):
    if target is None:
        target = _sample_size(rng)
    if shape is None:
        shape = "square" if rng.random() < 0.48 else "rectangle"
    if shape == "square" or target == 3:
        return target, target
    minimum = max(3, (2 * target + 4) // 5)
    other = int(rng.integers(minimum, target))
    return (target, other) if rng.random() < 0.5 else (other, target)


def _split_margin(rng, total, flush_first=False, flush_second=False):
    if flush_first:
        return 0, total
    if flush_second:
        return total, 0
    if total >= 7 and rng.random() < 0.28:
        short = int(rng.integers(1, 3))
        return (short, total - short) if rng.random() < 0.5 else (total - short, short)
    first = int(rng.integers(1, total))
    return first, total - first


def _sample_geometry(rng, frame_h, frame_w, allow_flush=True):
    extra_h = int(rng.integers(4, min(16, 30 - frame_h) + 1))
    extra_w = int(rng.integers(4, min(16, 30 - frame_w) + 1))
    flush = int(rng.integers(0, 4)) if allow_flush and rng.random() < 0.23 else -1
    top, bottom_margin = _split_margin(rng, extra_h, flush == 0, flush == 1)
    left, right_margin = _split_margin(rng, extra_w, flush == 2, flush == 3)
    return {
        "height": frame_h + extra_h,
        "width": frame_w + extra_w,
        "top": top,
        "left": left,
        "frame_h": frame_h,
        "frame_w": frame_w,
        "cross_row": top + int(rng.integers(1, frame_h - 1)),
        "cross_col": left + int(rng.integers(1, frame_w - 1)),
    }


def _construct_grids(spec):
    height, width = spec["height"], spec["width"]
    top, left = spec["top"], spec["left"]
    bottom = top + spec["frame_h"] - 1
    right = left + spec["frame_w"] - 1
    background, frame, line = spec["background"], spec["frame"], spec["line"]

    source = np.full((height, width), background, dtype=np.int8)
    source[:, spec["cross_col"]] = line
    source[spec["cross_row"], :] = line
    source[top, left:right + 1] = frame
    source[bottom, left:right + 1] = frame
    source[top:bottom + 1, left] = frame
    source[top:bottom + 1, right] = frame

    result = np.full((height, width), background, dtype=np.int8)
    corner = spec["corner"]
    destination_row = top if corner[0] == "U" else bottom
    destination_col = left if corner[1] == "L" else right
    result[destination_row, :] = line
    result[:, destination_col] = line
    result[top, left:right + 1] = frame
    result[bottom, left:right + 1] = frame
    result[top:bottom + 1, left] = frame
    result[top:bottom + 1, right] = frame
    return source, result


def _infer_input(input_grid):
    height, width = input_grid.shape
    colors = [int(x) for x in np.unique(input_grid)]
    assert len(colors) == 3
    candidates = []
    for frame in colors:
        locations = np.argwhere(input_grid == frame)
        top, left = (int(x) for x in locations.min(axis=0))
        bottom, right = (int(x) for x in locations.max(axis=0))
        frame_h, frame_w = bottom - top + 1, right - left + 1
        if not (3 <= frame_h <= 18 and 3 <= frame_w <= 18):
            continue
        margins = (top, height - bottom - 1, left, width - right - 1)
        if sum(margin == 0 for margin in margins) > 1:
            continue
        if margins[0] + margins[1] == 0 or margins[2] + margins[3] == 0:
            continue
        boundary = np.zeros(input_grid.shape, dtype=bool)
        boundary[top, left:right + 1] = True
        boundary[bottom, left:right + 1] = True
        boundary[top:bottom + 1, left] = True
        boundary[top:bottom + 1, right] = True
        if not np.array_equal(input_grid == frame, boundary):
            continue

        outside_cols = list(range(left)) + list(range(right + 1, width))
        outside_rows = list(range(top)) + list(range(bottom + 1, height))
        for line in colors:
            if line == frame:
                continue
            background = next(x for x in colors if x != frame and x != line)
            row_hits = np.flatnonzero(np.any(input_grid[:, outside_cols] == line, axis=1))
            col_hits = np.flatnonzero(np.any(input_grid[outside_rows, :] == line, axis=0))
            if row_hits.size != 1 or col_hits.size != 1:
                continue
            cross_row, cross_col = int(row_hits[0]), int(col_hits[0])
            if not (top < cross_row < bottom and left < cross_col < right):
                continue
            spec = {
                "height": height, "width": width, "top": top, "left": left,
                "frame_h": frame_h, "frame_w": frame_w,
                "cross_row": cross_row, "cross_col": cross_col,
                "background": background, "frame": frame, "line": line,
                "corner": "UR",
            }
            expected, _ = _construct_grids(spec)
            if np.array_equal(input_grid, expected):
                candidates.append(spec)
    assert len(candidates) == 1
    return candidates[0]


def _predict_from_input(input_grid, spec, corner):
    # Rebuild solely from visible roles and bounds, independent of the
    # output used to validate the prediction.
    top, left = spec["top"], spec["left"]
    bottom = top + spec["frame_h"] - 1
    right = left + spec["frame_w"] - 1
    result = np.full(input_grid.shape, spec["background"], dtype=np.int8)
    row = top if corner[0] == "U" else bottom
    col = left if corner[1] == "L" else right
    result[row, :left] = spec["line"]
    result[row, right + 1:] = spec["line"]
    result[:top, col] = spec["line"]
    result[bottom + 1:, col] = spec["line"]
    result[input_grid == spec["frame"]] = spec["frame"]
    return result


def _analyze_pair(input_grid, output_grid):
    spec = _infer_input(input_grid)
    assert input_grid.shape == output_grid.shape
    assert set(int(x) for x in np.unique(output_grid)) == {
        spec["background"], spec["frame"], spec["line"]
    }
    matching = [
        corner for corner in _CORNERS
        if np.array_equal(_predict_from_input(input_grid, spec, corner), output_grid)
    ]
    assert len(matching) == 1
    spec["corner"] = matching[0]
    return spec


def _make_unique_pair(spec, used_inputs):
    # Finite fallback over visible cross offsets and role colors avoids
    # relying on a lucky rejection loop for small requested puzzles.
    offsets = [(spec["cross_row"], spec["cross_col"])]
    offsets += [
        (r, c)
        for r in range(spec["top"] + 1, spec["top"] + spec["frame_h"] - 1)
        for c in range(spec["left"] + 1, spec["left"] + spec["frame_w"] - 1)
        if (r, c) != offsets[0]
    ]
    colors = [(spec["frame"], spec["line"])]
    colors += [
        (frame, line)
        for frame in range(10) for line in range(10)
        if frame != spec["background"] and line != spec["background"]
        and frame != line and (frame, line) != colors[0]
    ]
    for frame, line in colors:
        for row, col in offsets:
            attempt = dict(spec, frame=frame, line=line, cross_row=row, cross_col=col)
            input_grid, output_grid = _construct_grids(attempt)
            signature = _grid_signature(input_grid)
            if signature not in used_inputs and not _is_source_pair(input_grid, output_grid):
                used_inputs.add(signature)
                spec.update(attempt)
                return {"input": input_grid, "output": output_grid}
    raise RuntimeError("could not construct a unique pair")


def _placement_variant(rng, anchor):
    spec = dict(anchor)
    top_choices = [x for x in range(1, spec["height"] - spec["frame_h"])
                   if x != anchor["top"]]
    left_choices = [x for x in range(1, spec["width"] - spec["frame_w"])
                    if x != anchor["left"]]
    if rng.random() < 0.5:
        spec["top"] = int(rng.choice(top_choices))
        spec["left"] = int(rng.choice(left_choices)) if rng.random() < 0.5 else anchor["left"]
    else:
        spec["left"] = int(rng.choice(left_choices))
        spec["top"] = int(rng.choice(top_choices)) if rng.random() < 0.5 else anchor["top"]
    spec["cross_row"] = spec["top"] + int(rng.integers(1, spec["frame_h"] - 1))
    spec["cross_col"] = spec["left"] + int(rng.integers(1, spec["frame_w"] - 1))
    return spec


def _test_colors(rng, train_specs, fixed_background):
    draw = rng.random()
    if draw < 0.34:
        source = train_specs[int(rng.integers(len(train_specs)))]
        return source["background"], source["frame"], source["line"]
    background, frame, line = _sample_colors(rng, fixed_background)
    if draw < 0.70:
        seen_roles = {item["frame"] for item in train_specs} | {
            item["line"] for item in train_specs
        }
        novel = [color for color in range(10)
                 if color != background and color not in seen_roles]
        if novel:
            frame = int(rng.choice(novel))
            if line == frame:
                line = next(x for x in range(10) if x != background and x != frame)
    return background, frame, line


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed_value = _coerce_seed(seed)
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    corner = str(rng.choice(_CORNERS, p=(0.20, 0.40, 0.20, 0.20)))
    fixed_background = None
    if rng.random() < 0.36:
        fixed_background = int(rng.integers(10))

    child_seeds = rng.integers(0, np.iinfo(np.int64).max,
                               size=train_count + test_count, dtype=np.int64)
    train_specs = []
    first_colors = _sample_colors(rng, fixed_background)
    for index in range(train_count):
        pair_rng = np.random.default_rng(int(child_seeds[index]))
        if index == 0:
            size = int(pair_rng.integers(5, 10))
            frame_h, frame_w = size, size
            colors = first_colors
            geometry = _sample_geometry(pair_rng, frame_h, frame_w, allow_flush=False)
        elif index == 1:
            frame_h, frame_w = _sample_frame_dims(pair_rng, "rectangle",
                                                   max(4, _sample_size(pair_rng)))
            if fixed_background is None:
                colors = (first_colors[1], first_colors[2], first_colors[0])
            else:
                colors = _sample_colors(pair_rng, fixed_background)
                if colors[1] == first_colors[1] or colors[2] == first_colors[2]:
                    available = [x for x in range(10) if x != fixed_background]
                    frame = next(x for x in available if x != first_colors[1])
                    line = next(x for x in available if x != frame and x != first_colors[2])
                    colors = fixed_background, frame, line
            geometry = _sample_geometry(pair_rng, frame_h, frame_w)
        else:
            frame_h, frame_w = _sample_frame_dims(pair_rng)
            colors = _sample_colors(pair_rng, fixed_background)
            geometry = _sample_geometry(pair_rng, frame_h, frame_w)
        train_specs.append(dict(geometry, background=colors[0], frame=colors[1],
                                line=colors[2], corner=corner))

    used_inputs = set()
    train = [_make_unique_pair(spec, used_inputs) for spec in train_specs]
    train_sizes = [max(item["frame_h"], item["frame_w"]) for item in train_specs]
    test_specs = []
    for index in range(test_count):
        pair_rng = np.random.default_rng(int(child_seeds[train_count + index]))
        mode = str(pair_rng.choice(
            ("placement", "overlap", "smaller", "larger", "free"),
            p=(0.19, 0.23, 0.16, 0.25, 0.17),
        ))
        if mode == "placement":
            spec = _placement_variant(pair_rng, train_specs[0])
        else:
            if mode == "overlap":
                target = int(pair_rng.choice(train_sizes))
            elif mode == "smaller" and max(train_sizes) > 3:
                target = int(pair_rng.integers(3, max(train_sizes)))
            elif mode == "larger" and max(train_sizes) < 18:
                target = int(pair_rng.integers(max(train_sizes) + 1, 19))
            else:
                target = _sample_size(pair_rng)
            frame_h, frame_w = _sample_frame_dims(pair_rng, target=target)
            geometry = _sample_geometry(pair_rng, frame_h, frame_w)
            colors = _test_colors(pair_rng, train_specs, fixed_background)
            spec = dict(geometry, background=colors[0], frame=colors[1],
                        line=colors[2], corner=corner)
        test_specs.append(spec)

    test = [_make_unique_pair(spec, used_inputs) for spec in test_specs]
    # Order carries no semantic role. The two required training witnesses
    # remain present after shuffling.
    rng.shuffle(train)
    rng.shuffle(test)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    analyses = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            assert not np.array_equal(pair["input"], pair["output"])
            assert not _is_source_pair(pair["input"], pair["output"])
            signature = _grid_signature(pair["input"])
            pair_signature = _pair_signature(pair)
            assert signature not in seen_inputs and pair_signature not in seen_pairs
            seen_inputs.add(signature)
            seen_pairs.add(pair_signature)
            spec = _analyze_pair(pair["input"], pair["output"])
            analyses[split].append(spec)
            margins = (spec["top"], spec["height"] - spec["top"] - spec["frame_h"],
                       spec["left"], spec["width"] - spec["left"] - spec["frame_w"])
            assert sum(margin == 0 for margin in margins) <= 1
            for grid in pair.values():
                counts = [int(np.count_nonzero(grid == spec[key]))
                          for key in ("background", "frame", "line")]
                assert counts[0] > counts[1] and counts[0] > counts[2]

    all_specs = analyses["train"] + analyses["test"]
    assert len({item["corner"] for item in all_specs}) == 1
    assert any(item["frame_h"] == item["frame_w"] and
               0 < item["top"] and item["top"] + item["frame_h"] < item["height"] and
               0 < item["left"] and item["left"] + item["frame_w"] < item["width"]
               for item in analyses["train"])
    assert any(item["frame_h"] != item["frame_w"] for item in analyses["train"])
    assert len({item["frame"] for item in analyses["train"]}) >= 2
    assert len({item["line"] for item in analyses["train"]}) >= 2

    # Infer the convention exclusively from copies of the demonstrations.
    learned = {_analyze_pair(pair["input"].copy(), pair["output"].copy())["corner"]
               for pair in puzzle["train"]}
    assert len(learned) == 1
    learned_corner = next(iter(learned))
    for pair in puzzle["test"]:
        test_input = pair["input"].copy()
        inferred = _infer_input(test_input)
        prediction = _predict_from_input(test_input, inferred, learned_corner)
        assert np.array_equal(prediction, pair["output"])


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a[role], b[role])
        for a, b in zip(first[split], second[split]) for role in ("input", "output")
    ) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts, test_counts = set(), set()
    corners, shapes, layouts, palettes, inputs = set(), set(), set(), set(), set()
    rectangular = small = large = flush = asymmetric = 0
    varied_background = recycled_background = 0
    smaller_tests = overlapping_tests = larger_tests = familiar_tests = placement_tests = 0
    original_like_tests = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_specs = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test_specs = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        all_specs = train_specs + test_specs
        corners.add(train_specs[0]["corner"])
        backgrounds = {item["background"] for item in all_specs}
        if len(backgrounds) > 1:
            varied_background += 1
        if any(a["background"] in (b["frame"], b["line"])
               for a in all_specs for b in all_specs if a is not b):
            recycled_background += 1
        train_sizes = [max(item["frame_h"], item["frame_w"]) for item in train_specs]
        train_roles = {item["frame"] for item in train_specs} | {
            item["line"] for item in train_specs
        }
        for item in test_specs:
            size = max(item["frame_h"], item["frame_w"])
            smaller_tests += size < max(train_sizes)
            overlapping_tests += size in train_sizes
            larger_tests += size > max(train_sizes)
            familiar_tests += item["frame"] in train_roles and item["line"] in train_roles
            original_like_tests += size > max(train_sizes) and (
                item["frame"] not in train_roles or item["line"] not in train_roles
            )
            placement_tests += any(
                item["height"] == base["height"] and item["width"] == base["width"]
                and item["frame_h"] == base["frame_h"]
                and item["frame_w"] == base["frame_w"]
                and (item["background"], item["frame"], item["line"]) ==
                (base["background"], base["frame"], base["line"])
                for base in train_specs
            )
        for item, pair in zip(all_specs, puzzle["train"] + puzzle["test"]):
            rectangular += item["frame_h"] != item["frame_w"]
            small += max(item["frame_h"], item["frame_w"]) <= 4
            large += max(item["frame_h"], item["frame_w"]) >= 12
            margins = (item["top"], item["height"] - item["top"] - item["frame_h"],
                       item["left"], item["width"] - item["left"] - item["frame_w"])
            flush += any(margin == 0 for margin in margins)
            asymmetric += max(margins) - min(margins) >= 5
            shapes.add(pair["input"].shape)
            palettes.add((item["background"], item["frame"], item["line"]))
            inputs.add(_grid_signature(pair["input"]))
            layouts.add((pair["input"].shape, item["top"], item["left"],
                         item["frame_h"], item["frame_w"],
                         item["cross_row"] - item["top"],
                         item["cross_col"] - item["left"], item["corner"]))

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(corners) == 4 and len(shapes) >= 60 and len(layouts) >= 180
    assert len(palettes) >= 80 and len(inputs) >= 200
    assert rectangular >= 60 and small >= 10 and large >= 25
    assert flush >= 25 and asymmetric >= 40
    assert varied_background >= 10 and recycled_background >= 10
    assert smaller_tests >= 12 and overlapping_tests >= 10 and larger_tests >= 8
    assert familiar_tests >= 10 and placement_tests >= 6 and original_like_tests >= 2

    assert _puzzles_equal(generate_puzzle(19), generate_puzzle(19))
    assert _puzzles_equal(generate_puzzle(123456, 7, 3),
                          generate_puzzle(123456, 7, 3))
    assert not _puzzles_equal(puzzles[0], puzzles[1])
    for seed, requested_train, requested_test in (
        (200, 2, 1), (201, 10, 1), (202, 10, 4),
        (300, None, 1), (301, None, 4), (302, 2, None), (303, 10, None),
    ):
        puzzle = generate_puzzle(seed, requested_train, requested_test)
        validate_generated_puzzle(puzzle)
        if requested_train is not None:
            assert len(puzzle["train"]) == requested_train
        if requested_test is not None:
            assert len(puzzle["test"]) == requested_test
    for bad in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 2, "num_test": 3},
        {"num_train": 3.0, "num_test": 1},
        {"num_train": 3, "num_test": True},
    ):
        try:
            generate_puzzle(400, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count was accepted")
