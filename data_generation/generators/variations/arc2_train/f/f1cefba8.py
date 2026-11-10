import hashlib

import numpy as np


_FRAME_THICKNESS = 2
_SOURCE_INPUTS = {
    ((15, 17), "e4e57e542f223381f03c22c62c9cd73a35ccc453fbf2c39832ea1281459bed6e"),
    ((19, 13), "e13e317254c103ed9cb50e518c84f169ae6116ae01a1f62afe5e55aee09da187"),
    ((16, 18), "c4728009d6dab58317637f6466b127b3e6148121797a4f447ca591193d8d5e5d"),
    ((19, 18), "ef13286dfd3a7de00ae5a13c0783fc3417ac0aebc00d517b08ec8b08fe6dda17"),
}


def _checked_optional_count(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _checked_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a non-negative integer")

    result = int(seed)
    if result < 0:
        raise ValueError("seed must be a non-negative integer")
    return result


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _checked_optional_count(num_train, "num_train", 2, 10)
    test_count = _checked_optional_count(num_test, "num_test", 1, 4)

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        first_legal_train = test_count + 1
        number_of_choices = 10 - first_legal_train + 1
        train_count = (
            first_legal_train + (seed // 4) % number_of_choices
        )
    elif train_count is None:
        assert test_count is not None
        train_count = int(rng.integers(test_count + 1, 11))
    elif test_count is None:
        test_count = int(
            rng.integers(1, min(4, train_count - 1) + 1)
        )

    assert train_count is not None
    assert test_count is not None
    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _palette(rng, background=None):
    if background is None:
        background = int(rng.integers(0, 10))
    available = [color for color in range(10) if color != background]
    frame, core = rng.choice(available, size=2, replace=False)
    return background, int(frame), int(core)


def _sample_geometry(rng, regime, train_infos=()):
    if regime == "compact_train":
        core_h = int(rng.integers(4, 12))
        core_w = int(rng.integers(4, 12))
    elif regime == "source_train":
        core_h = int(rng.integers(4, 19))
        core_w = int(rng.integers(4, 19))
    elif regime == "small":
        core_h = int(rng.integers(4, 8))
        core_w = int(rng.integers(4, 8))
    elif regime == "large":
        core_h = int(rng.integers(18, 23))
        core_w = int(rng.integers(18, 23))
    elif regime == "matched":
        example = train_infos[int(rng.integers(len(train_infos)))]
        if int(rng.integers(2)):
            core_h = example["core_h"]
            core_w = int(rng.integers(4, 21))
        else:
            core_h = int(rng.integers(4, 21))
            core_w = example["core_w"]
    else:
        core_h = int(rng.integers(4, 21))
        core_w = int(rng.integers(4, 21))

    def margins(length):
        free = 30 - length - 4
        margin_limit = 4 if regime == "compact_train" else 8
        first = int(rng.integers(1, min(margin_limit, free - 1) + 1))
        second = int(rng.integers(1, min(margin_limit, free - first) + 1))
        return first, second

    top, bottom = margins(core_h)
    left, right = margins(core_w)
    return {
        "height": top + core_h + 4 + bottom,
        "width": left + core_w + 4 + right,
        "r0": top,
        "c0": left,
        "core_h": core_h,
        "core_w": core_w,
    }


def _sample_markers(rng, core_h, core_w, mode, required=()):
    markers = {side: [] for side in ("top", "bottom", "left", "right")}
    opposite = {"top": "bottom", "bottom": "top",
                "left": "right", "right": "left"}

    def add(side, offset=None, distinct=True):
        length = core_w if side in ("top", "bottom") else core_h
        choices = [i for i in range(length)
                   if all(abs(i - old) >= 2 for old in markers[side])
                   and (not distinct or i not in markers[opposite[side]])]
        if offset is not None:
            choices = [i for i in choices if i == offset]
        if not choices:
            return False
        markers[side].append(int(rng.choice(choices)))
        return True

    def axis(side):
        if not add(side):
            assert add(opposite[side])

    if mode == "train_required":
        for side in required:
            axis(side)
        for orientation in (("top", "bottom"), ("left", "right")):
            for _ in range(int(rng.integers(0, 3))):
                axis(orientation[int(rng.integers(2))])
    elif mode in ("single_col", "single_row", "multi_col", "multi_row",
                  "mixed", "dense"):
        if mode == "single_col":
            columns, rows = 1, 0
        elif mode == "single_row":
            columns, rows = 0, 1
        elif mode == "multi_col":
            columns, rows = int(rng.integers(2, 4)), 0
        elif mode == "multi_row":
            columns, rows = 0, int(rng.integers(2, 4))
        elif mode == "dense":
            columns, rows = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        else:
            columns, rows = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        for _ in range(columns):
            axis(("top", "bottom")[int(rng.integers(2))])
        for _ in range(rows):
            axis(("left", "right")[int(rng.integers(2))])
    elif mode in ("same_top", "same_bottom", "same_left", "same_right"):
        side = mode[5:]
        axis(side)
        axis(side)
        if int(rng.integers(2)):
            alternate = (("left", "right") if side in ("top", "bottom")
                         else ("top", "bottom"))
            axis(alternate[int(rng.integers(2))])
    elif mode == "all_four":
        for side in ("top", "bottom", "left", "right"):
            axis(side)
    elif mode in ("redundant_col", "redundant_row"):
        first, other = (("top", "bottom") if mode == "redundant_col"
                        else ("left", "right"))
        assert add(first)
        assert add(other, markers[first][0], distinct=False)
        if int(rng.integers(2)):
            alternate = (("left", "right") if first == "top"
                         else ("top", "bottom"))
            axis(alternate[int(rng.integers(2))])
    else:
        raise ValueError("unknown marker mode")
    return markers


def _construct_pair(rng, geometry, mode, palette, required=()):
    height, width = geometry["height"], geometry["width"]
    r0, c0 = geometry["r0"], geometry["c0"]
    core_h, core_w = geometry["core_h"], geometry["core_w"]
    r1, c1 = r0 + core_h + 3, c0 + core_w + 3
    background, frame, core = palette
    markers = _sample_markers(rng, core_h, core_w, mode, required)
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[r0:r1 + 1, c0:c1 + 1] = frame
    input_grid[r0 + 2:r1 - 1, c0 + 2:c1 - 1] = core
    for offset in markers["top"]:
        input_grid[r0 + 1, c0 + 2 + offset] = core
    for offset in markers["bottom"]:
        input_grid[r1 - 1, c0 + 2 + offset] = core
    for offset in markers["left"]:
        input_grid[r0 + 2 + offset, c0 + 1] = core
    for offset in markers["right"]:
        input_grid[r0 + 2 + offset, c1 - 1] = core
    output_grid = input_grid.copy()
    for offset in set(markers["top"] + markers["bottom"]):
        col = c0 + 2 + offset
        output_grid[r0:r1 + 1, col] = frame
        output_grid[:r0, col] = core
        output_grid[r1 + 1:, col] = core
    for offset in set(markers["left"] + markers["right"]):
        row = r0 + 2 + offset
        output_grid[row, c0:c1 + 1] = frame
        output_grid[row, :c0] = core
        output_grid[row, c1 + 1:] = core
    return input_grid, output_grid


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _analyze_input(grid: np.ndarray) -> dict:
    height, width = grid.shape
    background = int(grid[0, 0])

    # Inputs have positive exterior margins on every side.
    assert np.all(grid[0, :] == background)
    assert np.all(grid[-1, :] == background)
    assert np.all(grid[:, 0] == background)
    assert np.all(grid[:, -1] == background)

    non_background = grid != background
    assert bool(np.any(non_background))

    points = np.argwhere(non_background)
    r0, c0 = (int(value) for value in points.min(axis=0))
    r1, c1 = (int(value) for value in points.max(axis=0))

    assert 1 <= r0 < r1 < height - 1
    assert 1 <= c0 < c1 < width - 1

    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    assert outer_h >= 8
    assert outer_w >= 8

    outside = np.ones(grid.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    assert np.all(grid[outside] == background)

    rectangle = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(rectangle != background)

    frame = int(rectangle[0, 0])
    core = int(
        rectangle[_FRAME_THICKNESS, _FRAME_THICKNESS]
    )
    assert len({background, frame, core}) == 3
    assert set(int(value) for value in np.unique(grid)) == {
        background,
        frame,
        core,
    }

    # Compare with a perfect two-cell frame and solid core. Every difference
    # must be a core-colored marker in the inner frame layer.
    canonical = np.full(
        rectangle.shape,
        frame,
        dtype=grid.dtype,
    )
    canonical[
        _FRAME_THICKNESS : -_FRAME_THICKNESS,
        _FRAME_THICKNESS : -_FRAME_THICKNESS,
    ] = core

    differences = np.argwhere(rectangle != canonical)
    markers: dict[str, list[int]] = {
        "top": [],
        "bottom": [],
        "left": [],
        "right": [],
    }

    for rr_raw, cc_raw in differences:
        rr = int(rr_raw)
        cc = int(cc_raw)

        assert int(canonical[rr, cc]) == frame
        assert int(rectangle[rr, cc]) == core

        if (
            rr == 1
            and _FRAME_THICKNESS
            <= cc
            < outer_w - _FRAME_THICKNESS
        ):
            markers["top"].append(c0 + cc)
        elif (
            rr == outer_h - 2
            and _FRAME_THICKNESS
            <= cc
            < outer_w - _FRAME_THICKNESS
        ):
            markers["bottom"].append(c0 + cc)
        elif (
            cc == 1
            and _FRAME_THICKNESS
            <= rr
            < outer_h - _FRAME_THICKNESS
        ):
            markers["left"].append(r0 + rr)
        elif (
            cc == outer_w - 2
            and _FRAME_THICKNESS
            <= rr
            < outer_h - _FRAME_THICKNESS
        ):
            markers["right"].append(r0 + rr)
        else:
            raise AssertionError(
                "a core-colored anomaly is not a valid marker"
            )

    for side in markers:
        markers[side].sort()
        values = markers[side]
        assert len(values) == len(set(values))
        assert all(
            second - first >= 2
            for first, second in zip(values, values[1:])
        )

    selected_cols = sorted(
        set(markers["top"] + markers["bottom"])
    )
    selected_rows = sorted(
        set(markers["left"] + markers["right"])
    )
    assert selected_cols or selected_rows

    return {
        "height": height,
        "width": width,
        "background": background,
        "frame": frame,
        "core": core,
        "r0": r0,
        "r1": r1,
        "c0": c0,
        "c1": c1,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "core_h": outer_h - 2 * _FRAME_THICKNESS,
        "core_w": outer_w - 2 * _FRAME_THICKNESS,
        "markers": markers,
        "selected_cols": selected_cols,
        "selected_rows": selected_rows,
    }


def _expected_output(
    grid: np.ndarray,
    info: dict,
) -> np.ndarray:
    output = grid.copy()

    r0 = info["r0"]
    r1 = info["r1"]
    c0 = info["c0"]
    c1 = info["c1"]
    frame = info["frame"]
    core = info["core"]

    for col in info["selected_cols"]:
        output[r0 : r1 + 1, col] = frame
        output[:r0, col] = core
        output[r1 + 1 :, col] = core

    for row in info["selected_rows"]:
        output[row, c0 : c1 + 1] = frame
        output[row, :c0] = core
        output[row, c1 + 1 :] = core

    return output


def _source_key(grid):
    return tuple(grid.shape), hashlib.sha256(
        grid.astype(np.int8).tobytes()
    ).hexdigest()


def _structure_key(info):
    return (
        info["height"], info["width"], info["r0"], info["c0"],
        info["core_h"], info["core_w"],
        tuple(tuple(value - (info["c0"] + 2 if side in ("top", "bottom")
                           else info["r0"] + 2)
                    for value in info["markers"][side])
              for side in ("top", "bottom", "left", "right")),
    )


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list)
        assert isinstance(puzzle["test"], list)
        assert 2 <= len(puzzle["train"]) <= 10
        assert 1 <= len(puzzle["test"]) <= 4
        assert len(puzzle["train"]) > len(puzzle["test"])
        seen_inputs, seen_pairs, seen_structures = set(), set(), set()
        infos = {"train": [], "test": []}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                for grid in pair.values():
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert int(grid.min()) >= 0 and int(grid.max()) <= 9
                source, target = pair["input"], pair["output"]
                assert source.shape == target.shape
                key = _grid_key(source)
                pair_key = key, _grid_key(target)
                assert key not in seen_inputs and pair_key not in seen_pairs
                seen_inputs.add(key)
                seen_pairs.add(pair_key)
                assert _source_key(source) not in _SOURCE_INPUTS
                info = _analyze_input(source)
                structure = _structure_key(info)
                assert structure not in seen_structures
                seen_structures.add(structure)
                assert np.array_equal(target, _expected_output(source, info))
                assert not np.array_equal(source, target)
                infos[section].append(info)
        demonstrated = {side for info in infos["train"]
                        for side, values in info["markers"].items() if values}
        assert demonstrated == {"top", "bottom", "left", "right"}
        for info in infos["test"]:
            assert {side for side, values in info["markers"].items()
                    if values} <= demonstrated
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("invalid generated puzzle") from exc


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _checked_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    # Color policy is sampled once. In flexible policies, each pair still
    # visibly identifies its own background, frame, and core.
    color_mode = int(rng.integers(3))
    fixed_palette = _palette(rng)
    fixed_background = fixed_palette[0]
    first_vertical = ("top", "bottom")[int(rng.integers(2))]
    first_horizontal = ("left", "right")[int(rng.integers(2))]
    opposite = {"top": "bottom", "bottom": "top",
                "left": "right", "right": "left"}
    extra_modes = ("single_col", "single_row", "multi_col", "multi_row",
                   "mixed", "dense", "all_four", "same_top", "same_bottom",
                   "same_left", "same_right", "redundant_col", "redundant_row")
    train_modes = ["train_required", "train_required"]
    train_modes += [extra_modes[int(rng.integers(len(extra_modes)))]
                    for _ in range(train_count - 2)]
    target_modes = ("single_col", "multi_row", "same_top", "all_four",
                    "redundant_col", "same_right", "single_row", "dense",
                    "same_bottom", "redundant_row", "multi_col", "same_left")
    puzzle = {"train": [], "test": []}
    seen, seen_structures = set(), set()
    train_infos = []
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=train_count + test_count, dtype=np.int64)
    for index, pair_seed in enumerate(pair_seeds):
        pair_rng = np.random.default_rng(int(pair_seed))
        is_train = index < train_count
        section = "train" if is_train else "test"
        local_index = index if is_train else index - train_count
        if color_mode == 0:
            palette = fixed_palette
        else:
            palette = _palette(pair_rng, fixed_background if color_mode == 1 else None)
        if is_train:
            mode = train_modes[index]
            required = ((first_vertical, first_horizontal) if index == 0 else
                        (opposite[first_vertical], opposite[first_horizontal])
                        if index == 1 else ())
            geometry_regime = "compact_train" if seed % 4 == 2 else "source_train"
        else:
            mode = target_modes[(seed + local_index) % len(target_modes)]
            required = ()
            geometry_regime = ("small", "matched", "large", "free")[
                (seed + local_index) % 4
            ]
        for _ in range(40):
            geometry = _sample_geometry(pair_rng, geometry_regime, train_infos)
            source, target = _construct_pair(
                pair_rng, geometry, mode, palette, required
            )
            info = _analyze_input(source)
            if (_grid_key(source) not in seen
                    and _source_key(source) not in _SOURCE_INPUTS
                    and _structure_key(info) not in seen_structures):
                break
        else:
            # Exhaust a finite geometry grid if ordinary sampling collides.
            # Distinct core shapes guarantee room for all 14 possible pairs.
            for core_h in range(4, 23):
                for core_w in range(4, 23):
                    geometry = {"height": core_h + 6, "width": core_w + 6,
                                "r0": 1, "c0": 1,
                                "core_h": core_h, "core_w": core_w}
                    source, target = _construct_pair(
                        pair_rng, geometry, mode, palette, required
                    )
                    info = _analyze_input(source)
                    if (_grid_key(source) not in seen
                            and _source_key(source) not in _SOURCE_INPUTS
                            and _structure_key(info) not in seen_structures):
                        break
                else:
                    continue
                break
            else:
                raise RuntimeError("unable to construct a unique pair")
        seen.add(_grid_key(source))
        seen_structures.add(_structure_key(info))
        puzzle[section].append({"input": source, "output": target})
        if is_train:
            train_infos.append(info)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(
        len(first[section]) == len(second[section])
        and all(
            np.array_equal(a[key], b[key])
            for a, b in zip(first[section], second[section])
            for key in ("input", "output")
        )
        for section in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts, structures, shapes, profiles = (
        set(), set(), set(), set(), set()
    )
    source_keys = set()
    measured = {name: 0 for name in (
        "all_four", "single_axis", "redundant", "core_h4", "background0",
        "large_palette", "varied_background", "stable_palette",
        "smaller", "larger", "matched", "short_margin", "long_margin"
    )}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = {section: [_analyze_input(pair["input"])
                           for pair in puzzle[section]]
                 for section in ("train", "test")}
        colors, palettes = set(), []
        for section in ("train", "test"):
            for pair, info in zip(puzzle[section], infos[section]):
                source = pair["input"]
                assert _source_key(source) not in _SOURCE_INPUTS
                assert np.array_equal(pair["output"], _expected_output(source, info))
                source_keys.add(_grid_key(source))
                shapes.add(source.shape)
                colors.update(map(int, np.unique(source)))
                palettes.append((info["background"], info["frame"], info["core"]))
                pattern = tuple(len(info["markers"][side])
                                for side in ("top", "bottom", "left", "right"))
                profiles.add(pattern)
                structures.add((source.shape, info["r0"], info["c0"],
                                info["core_h"], info["core_w"], pattern,
                                tuple(info["selected_cols"]),
                                tuple(info["selected_rows"])))
                measured["all_four"] += all(pattern)
                measured["single_axis"] += bool(info["selected_cols"]) != bool(info["selected_rows"])
                measured["redundant"] += (
                    sum(pattern[:2]) > len(info["selected_cols"])
                    or sum(pattern[2:]) > len(info["selected_rows"])
                )
                measured["core_h4"] += info["core_h"] == 4
                measured["background0"] += info["background"] == 0
                margins = (info["r0"], info["height"] - info["r1"] - 1,
                           info["c0"], info["width"] - info["c1"] - 1)
                measured["short_margin"] += any(x == 1 for x in margins)
                measured["long_margin"] += any(x >= 6 for x in margins)
        measured["large_palette"] += len(colors) > 4
        measured["varied_background"] += len({p[0] for p in palettes}) > 1
        measured["stable_palette"] += len(set(palettes)) == 1
        max_h = max(x["core_h"] for x in infos["train"])
        max_w = max(x["core_w"] for x in infos["train"])
        for info in infos["test"]:
            measured["smaller"] += info["core_h"] < max_h or info["core_w"] < max_w
            measured["larger"] += info["core_h"] > max_h and info["core_w"] > max_w
            measured["matched"] += (
                info["core_h"] in {x["core_h"] for x in infos["train"]}
                or info["core_w"] in {x["core_w"] for x in infos["train"]}
            )
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(source_keys) >= 220 and len(structures) >= 220
    assert len(shapes) >= 60 and len(profiles) >= 12
    assert all(value > 0 for value in measured.values()), measured
    for seed, train, test in (
        (7001, 2, 1), (7002, 10, 1), (7003, 10, 4),
        (7004, None, 1), (7005, None, 4),
        (7006, 2, None), (7007, 10, None),
    ):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        assert _puzzles_equal(puzzle, generate_puzzle(seed, train, test))
    assert not _puzzles_equal(
        generate_puzzle(7001, 7, 3), generate_puzzle(7002, 7, 3)
    )
    for train, test in (
        (1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)
    ):
        try:
            generate_puzzle(9000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
