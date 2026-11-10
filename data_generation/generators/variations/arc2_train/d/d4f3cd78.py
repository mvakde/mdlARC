import numpy as np


def _array_from_rows(rows):
    return np.array(
        [[ord(character) - 48 for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = (
    (
        _array_from_rows(
            (
                "0000000000",
                "0000000000",
                "0000000000",
                "0000000000",
                "0000000000",
                "0055505500",
                "0050000500",
                "0050000500",
                "0050000500",
                "0055555500",
            )
        ),
        _array_from_rows(
            (
                "0000080000",
                "0000080000",
                "0000080000",
                "0000080000",
                "0000080000",
                "0055585500",
                "0058888500",
                "0058888500",
                "0058888500",
                "0055555500",
            )
        ),
    ),
    (
        _array_from_rows(
            (
                "0000000000",
                "0000000000",
                "0055555500",
                "0050000500",
                "0050000500",
                "0050000500",
                "0050000500",
                "0055505500",
                "0000000000",
                "0000000000",
            )
        ),
        _array_from_rows(
            (
                "0000000000",
                "0000000000",
                "0055555500",
                "0058888500",
                "0058888500",
                "0058888500",
                "0058888500",
                "0055585500",
                "0000080000",
                "0000080000",
            )
        ),
    ),
    (
        _array_from_rows(
            (
                "0000000000",
                "0000000000",
                "0055555000",
                "0050005000",
                "0050005000",
                "0050000000",
                "0050005000",
                "0055555000",
                "0000000000",
                "0000000000",
            )
        ),
        _array_from_rows(
            (
                "0000000000",
                "0000000000",
                "0055555000",
                "0058885000",
                "0058885000",
                "0058888888",
                "0058885000",
                "0055555000",
                "0000000000",
                "0000000000",
            )
        ),
    ),
)


def _validate_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")

    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _validate_requested_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = (
        None
        if num_train is None
        else _validate_requested_count(num_train, "num_train", 2, 10)
    )
    test = (
        None
        if num_test is None
        else _validate_requested_count(num_test, "num_test", 1, 4)
    )

    if train is None and test is None:
        legal_pairs = [
            (train_count, test_count)
            for train_count in range(2, 11)
            for test_count in range(
                1, min(4, train_count - 1) + 1
            )
        ]
        train, test = legal_pairs[seed % len(legal_pairs)]
    elif train is None:
        choices = np.arange(
            max(2, test + 1),
            11,
            dtype=np.int64,
        )
        train = int(
            choices[int(rng.integers(0, len(choices)))]
        )
    elif test is None:
        choices = np.arange(
            1,
            min(4, train - 1) + 1,
            dtype=np.int64,
        )
        test = int(
            choices[int(rng.integers(0, len(choices)))]
        )

    if train <= test:
        raise ValueError("num_train must be greater than num_test")

    return train, test


def sample_puzzle_parameters(rng):
    colors = rng.choice(10, size=3, replace=False)
    return {
        "fill": int(colors[2]),
        "fixed_palette": (int(colors[0]), int(colors[1])),
        "vary_palette": bool(rng.random() < 0.45),
        "shared_canvas": (
            (int(rng.integers(12, 31)), int(rng.integers(12, 31)))
            if rng.random() < 0.38
            else None
        ),
        "direction_mode": int(rng.integers(0, 4)),
    }


def _grid_signature(grid):
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _pair_signature(input_grid, output_grid):
    return (_grid_signature(input_grid), _grid_signature(output_grid))


def _is_source_pair(input_grid, output_grid):
    return any(
        np.array_equal(input_grid, old_input)
        and np.array_equal(output_grid, old_output)
        for old_input, old_output in _SOURCE_PAIRS
    )


def _draw_extent(rng, limit):
    choice = rng.random()
    if choice < 0.10:
        return 3
    if choice < 0.27:
        return int(rng.integers(4, min(5, limit) + 1))
    if choice < 0.45:
        return int(rng.integers(max(3, limit - 3), limit + 1))
    return int(rng.integers(3, limit + 1))


def _draw_frame_size(rng, max_height, max_width):
    if rng.random() < 0.08:
        return 3, 3
    if rng.random() < 0.16:
        extent = _draw_extent(rng, min(max_height, max_width))
        return extent, extent
    return _draw_extent(rng, max_height), _draw_extent(rng, max_width)


def _choose_palette(rng, parameters, previous=None):
    if not parameters["vary_palette"]:
        return parameters["fixed_palette"]
    available = [color for color in range(10) if color != parameters["fill"]]
    choices = [(background, frame) for background in available
               for frame in available if background != frame
               and (background, frame) != previous]
    return choices[int(rng.integers(0, len(choices)))]


def _side_schedule(rng, mode, train_count, test_count):
    vertical = ("top", "bottom")
    horizontal = ("left", "right")
    all_sides = vertical + horizontal
    if mode in (0, 1):
        train_options, test_options = (
            (vertical, horizontal) if mode == 0 else (horizontal, vertical)
        )
        train = [train_options[int(rng.integers(0, 2))]
                 for _ in range(train_count)]
        if rng.random() < 0.60:
            train[1] = next(side for side in train_options if side != train[0])
        test = [test_options[int(rng.integers(0, 2))]
                for _ in range(test_count)]
    elif mode == 2:
        train = [vertical[int(rng.integers(0, 2))],
                 horizontal[int(rng.integers(0, 2))]]
        train += [all_sides[int(rng.integers(0, 4))]
                  for _ in range(train_count - 2)]
        test = [all_sides[int(rng.integers(0, 4))]
                for _ in range(test_count)]
    else:
        train = [all_sides[int(rng.integers(0, 4))]
                 for _ in range(train_count)]
        test = [train[int(rng.integers(0, train_count))]
                for _ in range(test_count)]
        test[0] = train[0]
    return train, test


def generate_pair(rng, regime, puzzle_parameters, side=None,
                  canvas=None, frame_size=None, gap_index=None,
                  min_ray=None, zero_ray=False, palette=None):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown generation regime")
    if side is None:
        side = ("top", "bottom", "left", "right")[int(rng.integers(0, 4))]
    if side not in ("top", "bottom", "left", "right"):
        raise ValueError("unknown aperture side")
    if min_ray is None:
        min_ray = 1 if regime == "source_train" else 0
    if zero_ray and (regime == "source_train" or min_ray):
        raise ValueError("training must display an outward ray")

    if canvas is None:
        lower_height = max(6, (frame_size[0] + 2) if frame_size else 6)
        lower_width = max(6, (frame_size[1] + 2) if frame_size else 6)
        height = int(rng.integers(lower_height, 31))
        width = int(rng.integers(lower_width, 31))
    else:
        height, width = canvas
    if frame_size is None:
        reserve = 2 if regime == "source_train" else 1
        frame_height, frame_width = _draw_frame_size(
            rng, height - reserve, width - reserve
        )
    else:
        frame_height, frame_width = frame_size
    if not (3 <= frame_height <= height and 3 <= frame_width <= width):
        raise ValueError("frame does not fit canvas")

    if side in ("top", "bottom"):
        outward_room = height - frame_height
        if outward_room < min_ray:
            raise ValueError("no room for the required ray")
        outward = 0 if zero_ray else int(rng.integers(min_ray, outward_room + 1))
        top = outward if side == "top" else height - frame_height - outward
        left = int(rng.integers(0, width - frame_width + 1))
    else:
        outward_room = width - frame_width
        if outward_room < min_ray:
            raise ValueError("no room for the required ray")
        outward = 0 if zero_ray else int(rng.integers(min_ray, outward_room + 1))
        left = outward if side == "left" else width - frame_width - outward
        top = int(rng.integers(0, height - frame_height + 1))
    bottom = top + frame_height - 1
    right = left + frame_width - 1
    if gap_index is None:
        gap_index = int(rng.integers(1, (
            frame_width if side in ("top", "bottom") else frame_height
        ) - 1))
    if side in ("top", "bottom"):
        aperture_row = top if side == "top" else bottom
        aperture_col = left + gap_index
    else:
        aperture_row = top + gap_index
        aperture_col = left if side == "left" else right
    if palette is None:
        palette = puzzle_parameters["fixed_palette"]
    background, frame = palette
    fill = puzzle_parameters["fill"]

    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[top, left:right + 1] = frame
    input_grid[bottom, left:right + 1] = frame
    input_grid[top:bottom + 1, left] = frame
    input_grid[top:bottom + 1, right] = frame
    input_grid[aperture_row, aperture_col] = background
    output_grid = input_grid.copy()
    output_grid[top + 1:bottom, left + 1:right] = fill
    if side == "top":
        output_grid[:top + 1, aperture_col] = fill
    elif side == "bottom":
        output_grid[bottom:, aperture_col] = fill
    elif side == "left":
        output_grid[aperture_row, :left + 1] = fill
    else:
        output_grid[aperture_row, right:] = fill
    info = {"size": (frame_height, frame_width), "side": side,
            "gap_index": gap_index}
    return input_grid, output_grid, info


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    train_sides, test_sides = _side_schedule(
        rng, parameters["direction_mode"], train_count, test_count
    )
    matched_train = rng.random() < 0.24
    if matched_train:
        options = [s for s in ("top", "bottom", "left", "right")
                   if s != train_sides[0]
                   and (parameters["direction_mode"] not in (0, 1)
                        or (s in (("top", "bottom") if
                                  parameters["direction_mode"] == 0
                                  else ("left", "right"))))]
        train_sides[1] = options[int(rng.integers(0, len(options)))]
    puzzle = {"train": [], "test": []}
    seen = set()
    train_info = []
    previous_palette = None
    for split, sides in (("train", train_sides), ("test", test_sides)):
        for index, side in enumerate(sides):
            forced_size = None
            forced_gap = None
            if split == "train" and index == 1 and matched_train:
                forced_size = train_info[0]["size"]
            if (split == "test" and index == 0
                    and parameters["direction_mode"] == 3):
                forced_size = train_info[0]["size"]
                side = train_info[0]["side"]
                side_length = forced_size[
                    1 if side in ("top", "bottom") else 0
                ]
                offsets = [i for i in range(1, side_length - 1)
                           if i != train_info[0]["gap_index"]]
                if offsets:
                    forced_gap = offsets[int(rng.integers(0, len(offsets)))]
            min_ray = (2 if index == 0 else 1) if split == "train" else 0
            zero_ray = split == "test" and rng.random() < 0.13
            for attempt in range(150):
                palette = _choose_palette(
                    rng, parameters,
                    previous_palette if parameters["vary_palette"] else None
                )
                pair_seed = int(rng.integers(0, np.iinfo(np.uint64).max,
                                             dtype=np.uint64))
                pair_rng = np.random.default_rng(pair_seed)
                input_grid, output_grid, info = generate_pair(
                    pair_rng, "source_train" if split == "train"
                    else "target_test", parameters, side=side,
                    canvas=parameters["shared_canvas"],
                    frame_size=forced_size, gap_index=forced_gap,
                    min_ray=min_ray, zero_ray=zero_ray, palette=palette
                )
                if (split == "train" and index == 0
                        and parameters["direction_mode"] == 3
                        and info["size"][1 if side in ("top", "bottom") else 0] < 4):
                    continue
                signature = _grid_signature(input_grid)
                if signature not in seen and not _is_source_pair(
                    input_grid, output_grid
                ):
                    break
            else:
                raise RuntimeError("unable to construct a distinct pair")
            seen.add(signature)
            puzzle[split].append({"input": input_grid, "output": output_grid})
            if split == "train":
                train_info.append(info)
            previous_palette = palette
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((grid >= 0) & (grid <= 9))


def _parse_input(input_grid):
    _check_grid(input_grid)
    colors = [int(color) for color in np.unique(input_grid)]
    assert len(colors) == 2, "input must have two colors"
    candidates = []
    for frame in colors:
        background = next(color for color in colors if color != frame)
        rows, cols = np.where(input_grid == frame)
        top, bottom = int(rows.min()), int(rows.max())
        left, right = int(cols.min()), int(cols.max())
        frame_height, frame_width = bottom - top + 1, right - left + 1
        if frame_height < 3 or frame_width < 3:
            continue
        boundary = {(top, col) for col in range(left, right + 1)}
        boundary |= {(bottom, col) for col in range(left, right + 1)}
        boundary |= {(row, left) for row in range(top, bottom + 1)}
        boundary |= {(row, right) for row in range(top, bottom + 1)}
        actual = set(zip(rows.tolist(), cols.tolist()))
        missing = boundary - actual
        if not actual.issubset(boundary) or len(missing) != 1:
            continue
        gap_row, gap_col = next(iter(missing))
        if gap_row in (top, bottom) and gap_col in (left, right):
            continue
        expected = np.full(input_grid.shape, background, dtype=input_grid.dtype)
        expected[top, left:right + 1] = frame
        expected[bottom, left:right + 1] = frame
        expected[top:bottom + 1, left] = frame
        expected[top:bottom + 1, right] = frame
        expected[gap_row, gap_col] = background
        if not np.array_equal(input_grid, expected):
            continue
        if gap_row == top:
            side, ray_length, gap_index = "top", top, gap_col - left
        elif gap_row == bottom:
            side, ray_length, gap_index = (
                "bottom", input_grid.shape[0] - bottom - 1, gap_col - left
            )
        elif gap_col == left:
            side, ray_length, gap_index = "left", left, gap_row - top
        else:
            side, ray_length, gap_index = (
                "right", input_grid.shape[1] - right - 1, gap_row - top
            )
        candidates.append({
            "background": background, "frame": frame, "bounds":
            (top, bottom, left, right), "side": side,
            "ray_outside_length": ray_length,
            "frame_height": frame_height, "frame_width": frame_width,
            "aperture": (gap_row, gap_col), "gap_index": gap_index,
            "cavity_area": (frame_height - 2) * (frame_width - 2),
        })
    assert len(candidates) == 1, "frame and aperture must be unique"
    return candidates[0]


def _fill_rule(input_grid, metadata, fill):
    output = input_grid.copy()
    top, bottom, left, right = metadata["bounds"]
    gap_row, gap_col = metadata["aperture"]
    output[top + 1:bottom, left + 1:right] = fill
    if metadata["side"] == "top":
        output[:top + 1, gap_col] = fill
    elif metadata["side"] == "bottom":
        output[bottom:, gap_col] = fill
    elif metadata["side"] == "left":
        output[gap_row, :left + 1] = fill
    else:
        output[gap_row, right:] = fill
    return output


def _analyze_pair(input_grid, output_grid):
    _check_grid(output_grid)
    assert input_grid.shape == output_grid.shape
    metadata = _parse_input(input_grid)
    changed = input_grid != output_grid
    assert np.any(changed)
    assert np.all(input_grid[changed] == metadata["background"])
    fills = np.unique(output_grid[changed])
    assert len(fills) == 1
    fill = int(fills[0])
    assert fill not in np.unique(input_grid)
    assert np.array_equal(output_grid, _fill_rule(input_grid, metadata, fill))
    metadata["fill"] = fill
    return metadata


def _predict_from_examples(train, test_input):
    fills = set()
    for pair in train:
        changed = pair["input"] != pair["output"]
        new_colors = np.unique(pair["output"][changed])
        assert len(new_colors) == 1
        fills.add(int(new_colors[0]))
    assert len(fills) == 1
    return _fill_rule(test_input, _parse_input(test_input), next(iter(fills)))


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list)
        assert isinstance(puzzle["test"], list)
        train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
        assert 2 <= train_count <= 10 and 1 <= test_count <= 4
        assert train_count > test_count
        seen_inputs, seen_pairs, fills = set(), set(), set()
        train_metadata = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {
                    "input", "output"
                }
                input_grid, output_grid = pair["input"], pair["output"]
                metadata = _analyze_pair(input_grid, output_grid)
                input_key = _grid_signature(input_grid)
                pair_key = _pair_signature(input_grid, output_grid)
                assert input_key not in seen_inputs and pair_key not in seen_pairs
                assert not _is_source_pair(input_grid, output_grid)
                seen_inputs.add(input_key)
                seen_pairs.add(pair_key)
                fills.add(metadata["fill"])
                if split == "train":
                    assert metadata["ray_outside_length"] >= 1
                    train_metadata.append(metadata)
                else:
                    assert np.array_equal(
                        output_grid,
                        _predict_from_examples(puzzle["train"], input_grid.copy())
                    )
        assert len(fills) == 1, "fill must be puzzle-wide"
        assert max(item["ray_outside_length"] for item in train_metadata) >= 2
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("malformed generated puzzle") from exc


def _puzzles_equal(first, second):
    return all(
        len(first[split]) == len(second[split]) and all(
            np.array_equal(a[key], b[key])
            for a, b in zip(first[split], second[split])
            for key in ("input", "output")
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({
        tuple(_pair_signature(pair["input"], pair["output"])
              for split in ("train", "test") for pair in puzzle[split])
        for puzzle in puzzles
    }) == 42
    sides, aspects, side_aspects = set(), set(), set()
    rays, frame_sizes, shapes, palettes = set(), set(), set(), set()
    source_like = reverse = mixed = demonstrated = shared = changing = 0
    zero = long_ray = tiny = wide_test = larger_test = matched = 0
    short_large = long_small = new_gap = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        train_sides = {m["side"] for m in train}
        test_sides = {m["side"] for m in test}
        source_like += train_sides <= {"top", "bottom"} and test_sides <= {"left", "right"}
        reverse += train_sides <= {"left", "right"} and test_sides <= {"top", "bottom"}
        mixed += bool(train_sides & {"top", "bottom"}) and bool(train_sides & {"left", "right"})
        demonstrated += bool(train_sides & test_sides)
        all_pairs = puzzle["train"] + puzzle["test"]
        all_meta = train + test
        shared += len({p["input"].shape for p in all_pairs}) == 1
        changing += len({(m["background"], m["frame"]) for m in all_meta}) > 1
        larger_test += max(m["frame_width"] for m in test) >= min(m["frame_width"] for m in train)
        for first in train:
            for second in train:
                matched += (first is not second and
                            (first["frame_height"], first["frame_width"]) ==
                            (second["frame_height"], second["frame_width"]) and
                            first["side"] != second["side"])
        for item in test:
            new_gap += any(
                item["side"] == old["side"] and
                (item["frame_height"], item["frame_width"]) ==
                (old["frame_height"], old["frame_width"]) and
                item["gap_index"] != old["gap_index"]
                for old in train
            )
        for pair, item in zip(all_pairs, all_meta):
            h, w = item["frame_height"], item["frame_width"]
            sides.add(item["side"])
            aspects.add("square" if h == w else "wide" if w > h else "tall")
            side_aspects.add((item["side"], "square" if h == w
                              else "wide" if w > h else "tall"))
            rays.add(item["ray_outside_length"])
            frame_sizes.add((h, w))
            shapes.add(pair["input"].shape)
            palettes.add((item["background"], item["frame"], item["fill"]))
            zero += item["ray_outside_length"] == 0
            long_ray += item["ray_outside_length"] >= 10
            tiny += h == w == 3
            short_large += item["ray_outside_length"] <= 2 and item["cavity_area"] >= 30
            long_small += item["ray_outside_length"] >= 10 and item["cavity_area"] <= 9
        wide_test += any(m["frame_width"] > m["frame_height"] for m in test)
    assert sides == {"top", "bottom", "left", "right"}
    assert aspects == {"square", "wide", "tall"}
    assert len(side_aspects) == 12
    assert len(frame_sizes) >= 70 and len(shapes) >= 60 and len(palettes) >= 30
    assert source_like >= 3 and reverse >= 3 and mixed >= 3 and demonstrated >= 3
    assert shared >= 3 and changing >= 5 and matched >= 2 and new_gap >= 2
    assert zero >= 1 and long_ray >= 8 and tiny >= 1
    assert short_large >= 1 and long_small >= 1 and wide_test >= 3
    assert larger_test >= 8
    assert _puzzles_equal(generate_puzzle(2718), generate_puzzle(2718))
    assert _puzzles_equal(generate_puzzle(314159, 7, 3),
                          generate_puzzle(314159, 7, 3))
    assert not _puzzles_equal(generate_puzzle(100), generate_puzzle(101))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (6, 2),
                                    (None, 1), (None, 4), (2, None),
                                    (10, None)):
        puzzle = generate_puzzle(9000 + (train_count or 0) * 10 +
                                 (test_count or 0), train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(77, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts must raise ValueError")
