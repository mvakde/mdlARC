import numpy as np


_CORNERS = ("TL", "TR", "BL", "BR")

# (corner, input arm length, foreground color) for the three source pairs.
_SOURCE_SPECS = (
    ("BR", 2, 5),
    ("BL", 3, 9),
    ("TR", 4, 8),
)


def _checked_optional_count(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")

    return value


def _select_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _checked_optional_count(
        num_train, "num_train", 2, 10
    )
    num_test = _checked_optional_count(
        num_test, "num_test", 1, 4
    )

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))
    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))
    else:
        largest_test = min(4, num_train - 1)
        if largest_test < 1:
            raise ValueError("no legal num_test exists for num_train")
        num_test = int(rng.integers(1, largest_test + 1))

    return num_train, num_test


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    # The output canvas is a convention inferred from the demonstrations.
    # Source-like 16x16 canvases coexist with independently sized margins.
    if rng.random() < 0.20:
        output_shape = (16, 16)
    elif rng.random() < 0.35:
        side = int(rng.integers(14, 27))
        output_shape = (side, side)
    else:
        output_shape = (int(rng.integers(14, 27)),
                        int(rng.integers(14, 27)))
    if rng.random() < 0.48:
        input_shape = None
        maximum_arm = min(8, min(output_shape) // 2)
    else:
        if output_shape == (16, 16) and rng.random() < 0.35:
            input_shape = (6, 6)
        else:
            input_shape = (int(rng.integers(6, 12)),
                           int(rng.integers(6, 12)))
        maximum_arm = min(8, min(output_shape) // 2, min(input_shape) - 1)
    corner_mode = str(rng.choice(("source_like", "reverse", "mixed"),
                                  p=(0.25, 0.25, 0.50)))
    shift_mode = ("larger" if corner_mode == "source_like" else
                  str(rng.choice(("larger", "smaller", "intermediate",
                                  "familiar", "mixed"))))
    return {
        "output_shape": output_shape,
        "input_shape": input_shape,
        "maximum_arm": maximum_arm,
        "corner_mode": corner_mode,
        "shift_mode": shift_mode,
        "background_mode": str(rng.choice(("shared", "varying"))),
        "test_color_mode": str(rng.choice(("reuse", "new", "mixed"))),
        "palette": [int(x) for x in rng.permutation(10)],
    }


def _make_l_input(
    shape: tuple[int, int],
    background: int,
    foreground: int,
    corner: str,
    arm: int,
) -> np.ndarray:
    height, width = shape
    grid = np.full(
        (height, width),
        background,
        dtype=np.int8,
    )

    if corner == "TL":
        grid[0, :arm] = foreground
        grid[:arm, 0] = foreground
    elif corner == "TR":
        grid[0, width - arm:] = foreground
        grid[:arm, width - 1] = foreground
    elif corner == "BL":
        grid[height - 1, :arm] = foreground
        grid[height - arm:, 0] = foreground
    elif corner == "BR":
        grid[height - 1, width - arm:] = foreground
        grid[height - arm:, width - 1] = foreground
    else:
        raise ValueError("unknown corner")

    return grid


def _make_triangle_output(
    shape: tuple[int, int],
    background: int,
    foreground: int,
    corner: str,
    input_arm: int,
) -> np.ndarray:
    height, width = shape
    leg = 2 * input_arm

    grid = np.full(
        (height, width),
        background,
        dtype=np.int8,
    )
    diagonal_step = np.arange(leg)

    if corner == "TL":
        grid[0, :leg] = foreground
        grid[:leg, 0] = foreground
        grid[
            diagonal_step,
            leg - 1 - diagonal_step,
        ] = foreground

    elif corner == "TR":
        grid[0, width - leg:] = foreground
        grid[:leg, width - 1] = foreground
        grid[
            diagonal_step,
            width - leg + diagonal_step,
        ] = foreground

    elif corner == "BL":
        grid[height - 1, :leg] = foreground
        grid[height - leg:, 0] = foreground
        grid[
            height - leg + diagonal_step,
            diagonal_step,
        ] = foreground

    elif corner == "BR":
        grid[height - 1, width - leg:] = foreground
        grid[height - leg:, width - 1] = foreground
        grid[
            height - leg + diagonal_step,
            width - 1 - diagonal_step,
        ] = foreground

    else:
        raise ValueError("unknown corner")

    return grid


def _geometry_specs(rng, parameters, num_train, num_test):
    maximum = parameters["maximum_arm"]
    shift = parameters["shift_mode"]
    if parameters["corner_mode"] == "source_like":
        low, high = 2, 3
    elif shift == "smaller":
        low = int(rng.integers(3, maximum))
        high = int(rng.integers(low + 1, maximum + 1))
    elif shift == "intermediate":
        low = 2
        high = int(rng.integers(4, maximum + 1))
    else:
        low = int(rng.integers(2, maximum - 1))
        high = int(rng.integers(low + 1, maximum))

    if parameters["corner_mode"] == "source_like":
        corner_order = ["BL", "BR"]
        if rng.random() < 0.5:
            corner_order.reverse()
    elif parameters["corner_mode"] == "reverse":
        corner_order = ["TL", "TR"]
        if rng.random() < 0.5:
            corner_order.reverse()
    else:
        corner_order = [str(_CORNERS[i]) for i in rng.permutation(4)]
    first, second = corner_order[:2]
    train = [(first, low), (second, high)]
    if num_train >= 3:
        train.append((first, high))
    if num_train >= 4:
        train.append((second, low))

    permitted = list(range(2, maximum + 1))
    if shift == "larger":
        permitted.remove(maximum)
    elif shift == "smaller":
        permitted.remove(2)
    elif shift == "intermediate":
        permitted.remove(3)
    candidates = [(corner, arm) for corner in _CORNERS for arm in permitted]
    if (parameters["corner_mode"] != "mixed"
            and num_train <= 2 * len(permitted)):
        candidates = [(corner, arm) for corner in corner_order[:2]
                      for arm in permitted]
    rng.shuffle(candidates)
    while len(train) < num_train:
        available = [item for item in candidates if item not in train]
        assert available
        used_corners = {c for c, _ in train}
        used_arms = {a for _, a in train}
        available.sort(key=lambda item: (item[0] not in used_corners,
                                         item[1] not in used_arms), reverse=True)
        train.append(available[0])

    used = set(train)
    test = []
    train_arms = {arm for _, arm in train}
    for index in range(num_test):
        available = [(corner, arm) for corner in _CORNERS
                     for arm in range(2, maximum + 1) if (corner, arm) not in used]
        assert available
        if index == 0 and shift == "larger":
            target_arms = {maximum}
        elif index == 0 and shift == "smaller":
            target_arms = {2}
        elif index == 0 and shift == "intermediate":
            target_arms = {3}
        elif index == 0 and shift == "familiar":
            target_arms = train_arms
        else:
            target_arms = set(range(2, maximum + 1))
        preferred = [item for item in available if item[1] in target_arms]
        if preferred:
            available = preferred
        if parameters["corner_mode"] == "source_like":
            preferred = [item for item in available if item[0][0] == "T"]
        elif parameters["corner_mode"] == "reverse":
            preferred = [item for item in available if item[0][0] == "B"]
        else:
            unseen = set(_CORNERS) - {c for c, _ in train}
            preferred = [item for item in available if item[0] in unseen]
        if preferred:
            available = preferred
        choice = available[int(rng.integers(len(available)))]
        test.append(choice)
        used.add(choice)
    return train, test


def _sample_colors(rng, parameters, num_train, num_test):
    total = num_train + num_test
    palette = parameters["palette"]
    if parameters["background_mode"] == "shared":
        backgrounds = [palette[0]] * total
    else:
        count = int(rng.integers(2, 6))
        backgrounds = [palette[i % count] for i in range(total)]
    foregrounds = []
    for i in range(total):
        background = backgrounds[i]
        if i < 2 and parameters["background_mode"] == "varying":
            foreground = backgrounds[1 - i]
        elif i < num_train:
            options = [x for x in palette
                       if x != background and (i != 1 or x != foregrounds[0])]
            foreground = options[int(rng.integers(len(options)))]
        else:
            mode = parameters["test_color_mode"]
            if mode == "mixed":
                mode = "reuse" if i % 2 == 0 else "new"
            train_colors = set(foregrounds[:num_train])
            options = ([x for x in train_colors if x != background] if mode == "reuse"
                       else [x for x in palette if x != background and x not in train_colors])
            if not options:
                options = [x for x in palette if x != background]
            foreground = options[int(rng.integers(len(options)))]
        foregrounds.append(foreground)
    return list(zip(backgrounds, foregrounds))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _select_counts(rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    train_specs, test_specs = _geometry_specs(rng, parameters, num_train, num_test)
    colors = _sample_colors(rng, parameters, num_train, num_test)
    puzzle = {"train": [], "test": []}
    for index, (corner, arm) in enumerate(train_specs + test_specs):
        if parameters["input_shape"] is None:
            minimum = max(6, arm + 1)
            shape = (int(rng.integers(minimum, 12)),
                     int(rng.integers(minimum, 12)))
        else:
            shape = parameters["input_shape"]
        background, foreground = colors[index]
        input_grid = _make_l_input(shape, background, foreground, corner, arm)
        output_grid = _make_triangle_output(parameters["output_shape"], background,
                                            foreground, corner, arm)
        if _is_exact_source_pair(input_grid, output_grid):
            blocked = {background, foreground}
            if index < 2:
                blocked.add(colors[1 - index][1])
            foreground = next(x for x in parameters["palette"] if x not in blocked)
            input_grid = _make_l_input(shape, background, foreground, corner, arm)
            output_grid = _make_triangle_output(parameters["output_shape"], background,
                                                foreground, corner, arm)
        split = "train" if index < num_train else "test"
        puzzle[split].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_signature(
    grid: np.ndarray,
) -> tuple:
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_signature(
    pair: dict,
) -> tuple:
    return (
        _grid_signature(pair["input"]),
        _grid_signature(pair["output"]),
    )


def _analyze_l_input(
    grid: np.ndarray,
) -> dict:
    values, counts = np.unique(
        grid,
        return_counts=True,
    )
    assert len(values) == 2

    largest = int(np.argmax(counts))
    assert int(counts[largest]) > int(counts[1 - largest])

    background = int(values[largest])
    foreground = int(values[1 - largest])
    height, width = grid.shape

    matches = []
    for corner in _CORNERS:
        for arm in range(2, min(height, width)):
            expected = _make_l_input(
                (height, width),
                background,
                foreground,
                corner,
                arm,
            )
            if np.array_equal(grid, expected):
                matches.append((corner, arm))

    assert len(matches) == 1
    corner, arm = matches[0]

    return {
        "background": background,
        "foreground": foreground,
        "corner": corner,
        "arm": arm,
    }


def _is_exact_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    if (
        input_grid.shape != (6, 6)
        or output_grid.shape != (16, 16)
    ):
        return False

    for corner, arm, foreground in _SOURCE_SPECS:
        source_input = _make_l_input(
            (6, 6),
            7,
            foreground,
            corner,
            arm,
        )
        source_output = _make_triangle_output(
            (16, 16),
            7,
            foreground,
            corner,
            arm,
        )

        if (
            np.array_equal(input_grid, source_input)
            and np.array_equal(output_grid, source_output)
        ):
            return True

    return False


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, geometries, output_shapes = set(), set(), set()
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for name in ("input", "output"):
                grid = pair[name]
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            input_grid, output_grid = pair["input"], pair["output"]
            assert input_grid.dtype == output_grid.dtype
            assert all(o > i for o, i in zip(output_grid.shape, input_grid.shape))
            signature = _grid_signature(input_grid)
            assert signature not in inputs
            inputs.add(signature)
            info = _analyze_l_input(input_grid)
            assert 2 * info["arm"] <= min(output_grid.shape)
            geometry = (info["corner"], info["arm"])
            assert geometry not in geometries
            geometries.add(geometry)
            assert np.array_equal(output_grid, _make_triangle_output(
                output_grid.shape, info["background"], info["foreground"],
                info["corner"], info["arm"]))
            assert np.count_nonzero(output_grid == info["foreground"]) == 6 * info["arm"] - 3
            assert not _is_exact_source_pair(input_grid, output_grid)
            records[split].append(info)
            output_shapes.add(output_grid.shape)
    assert len(output_shapes) == 1
    assert len({x["corner"] for x in records["train"]}) >= 2
    assert len({x["arm"] for x in records["train"]}) >= 2
    assert len({x["foreground"] for x in records["train"]}) >= 2


def _predict_from_examples(train_pairs, test_input):
    # Infer the shared canvas and check the doubling rule using only
    # training grids. The test grid supplies local corner and color roles.
    shapes = {pair["output"].shape for pair in train_pairs}
    assert len(shapes) == 1
    shape = next(iter(shapes))
    for pair in train_pairs:
        info = _analyze_l_input(pair["input"])
        expected = _make_triangle_output(shape, info["background"], info["foreground"],
                                         info["corner"], info["arm"])
        assert np.array_equal(pair["output"], expected)
    info = _analyze_l_input(test_input)
    return _make_triangle_output(shape, info["background"], info["foreground"],
                                 info["corner"], info["arm"])


def _check_pair_independently(pair):
    input_grid, output_grid = pair["input"], pair["output"]
    info = _analyze_l_input(input_grid)
    h, w = output_grid.shape
    top = info["corner"][0] == "T"
    left = info["corner"][1] == "L"
    r0, c0 = (0 if top else h - 1), (0 if left else w - 1)
    dr, dc = (1 if top else -1), (1 if left else -1)
    leg = 2 * info["arm"]
    cells = {(r0 + dr * j, c0) for j in range(leg)}
    cells.update((r0, c0 + dc * j) for j in range(leg))
    cells.update((r0 + dr * j, c0 + dc * (leg - 1 - j))
                 for j in range(leg))
    actual = set(map(tuple, np.argwhere(output_grid == info["foreground"])))
    assert actual == cells
    assert np.all(output_grid[output_grid != info["foreground"]]
                  == info["background"])


def _puzzles_equal(
    first: dict,
    second: dict,
) -> bool:
    if set(first.keys()) != set(second.keys()):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for pair_a, pair_b in zip(
            first[split],
            second[split],
        ):
            if not np.array_equal(
                pair_a["input"],
                pair_b["input"],
            ):
                return False
            if not np.array_equal(
                pair_a["output"],
                pair_b["output"],
            ):
                return False

    return True


def _puzzle_signature(
    puzzle: dict,
) -> tuple:
    signature = []

    for split in ("train", "test"):
        signature.append(split)
        for pair in puzzle[split]:
            signature.append(_pair_signature(pair))

    return tuple(signature)


def _copy_puzzle(
    puzzle: dict,
) -> dict:
    return {
        split: [
            {
                "input": pair["input"].copy(),
                "output": pair["output"].copy(),
            }
            for pair in puzzle[split]
        ]
        for split in ("train", "test")
    }


def self_test() -> None:
    for args in ((None, None), (2, 1), (10, 1), (10, 4),
                 (None, 1), (None, 4), (2, None), (10, None)):
        a = generate_puzzle(12345, *args)
        b = generate_puzzle(12345, *args)
        validate_generated_puzzle(a)
        for split in ("train", "test"):
            assert len(a[split]) == len(b[split])
            for x, y in zip(a[split], b[split]):
                assert np.array_equal(x["input"], y["input"])
                assert np.array_equal(x["output"], y["output"])
        for pair in a["test"]:
            assert np.array_equal(_predict_from_examples(a["train"], pair["input"]), pair["output"])
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (True, 1), (3.0, 1)):
        try:
            generate_puzzle(99, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    cohort = [generate_puzzle(seed) for seed in range(42)]
    assert not _puzzles_equal(cohort[0], cohort[1])
    train_counts = {len(p["train"]) for p in cohort}
    test_counts = {len(p["test"]) for p in cohort}
    shapes, corners, arms, palettes = set(), set(), set(), set()
    regimes = set()
    pair_varying_background = 0
    pair_varying_input = 0
    reused_test_color = 0
    novel_test_color = 0
    swapped_roles = 0
    top_only = bottom_only = mixed_sides = 0
    familiar_corner = novel_corner = 0
    larger = smaller = intermediate = familiar_arm = 0
    independent_canvas = rectangular_canvas = 0
    for p in cohort:
        validate_generated_puzzle(p)
        train = [_analyze_l_input(pair["input"]) for pair in p["train"]]
        test = [_analyze_l_input(pair["input"]) for pair in p["test"]]
        assert len({(x["corner"], x["arm"]) for x in train}) == len(train)
        assert len({(x["corner"], x["arm"]) for x in test}) == len(test)
        train_corners = {x["corner"] for x in train}
        train_arms = {x["arm"] for x in train}
        test_corners = {x["corner"] for x in test}
        top_only += all(x[0] == "T" for x in train_corners)
        bottom_only += all(x[0] == "B" for x in train_corners)
        mixed_sides += any(x[0] == "T" for x in train_corners) and any(
            x[0] == "B" for x in train_corners)
        familiar_corner += bool(train_corners & test_corners)
        novel_corner += bool(test_corners - train_corners)
        larger += any(x["arm"] > max(train_arms) for x in test)
        smaller += any(x["arm"] < min(train_arms) for x in test)
        intermediate += any(min(train_arms) < x["arm"] < max(train_arms)
                            and x["arm"] not in train_arms for x in test)
        familiar_arm += any(x["arm"] in train_arms for x in test)
        inp_shape = p["train"][0]["input"].shape
        out_shape = p["train"][0]["output"].shape
        independent_canvas += out_shape != (2 * inp_shape[0] + 4,
                                           2 * inp_shape[1] + 4)
        rectangular_canvas += out_shape[0] != out_shape[1]
        regimes.add((tuple(sorted(train_corners)), np.sign(test[0]["arm"] - max(train_arms))))
        if len({x["background"] for x in train + test}) > 1:
            pair_varying_background += 1
        if len({pair["input"].shape for split in ("train", "test") for pair in p[split]}) > 1:
            pair_varying_input += 1
        train_colors = {x["foreground"] for x in train}
        reused_test_color += any(x["foreground"] in train_colors for x in test)
        novel_test_color += any(x["foreground"] not in train_colors for x in test)
        swapped_roles += any(x["foreground"] == y["background"]
                             for x in train + test for y in train + test if x is not y)
        for split in ("train", "test"):
            for pair in p[split]:
                _check_pair_independently(pair)
                info = _analyze_l_input(pair["input"])
                shapes.add((pair["input"].shape, pair["output"].shape))
                corners.add(info["corner"])
                arms.add(info["arm"])
                palettes.add((info["background"], info["foreground"]))
        for pair in p["test"]:
            assert np.array_equal(_predict_from_examples(p["train"], pair["input"]), pair["output"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert corners == set(_CORNERS) and len(arms) >= 5
    assert len(shapes) >= 25 and len(palettes) >= 25 and len(regimes) >= 5
    assert pair_varying_background >= 10 and pair_varying_input >= 10
    assert reused_test_color >= 10 and novel_test_color >= 10 and swapped_roles >= 10
    assert top_only >= 3 and bottom_only >= 3 and mixed_sides >= 10
    assert familiar_corner >= 10 and novel_corner >= 10
    assert larger >= 8 and smaller >= 3 and intermediate >= 3 and familiar_arm >= 10
    assert independent_canvas >= 15 and rectangular_canvas >= 10
