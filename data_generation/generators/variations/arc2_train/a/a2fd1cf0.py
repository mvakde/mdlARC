"""Generate color-directed, horizontal-first elbow puzzles."""

import numpy as np


# (height, width, source row, source column, destination row, destination column)
_SOURCE_SPECS = (
    (10, 16, 7, 1, 1, 11),
    (15, 15, 1, 4, 13, 10),
    (12, 14, 1, 11, 10, 4),
    (12, 14, 2, 1, 10, 10),
)
_SOURCE_PALETTE = (0, 2, 3, 8)
_ORIENTATIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grid_key(grid: np.ndarray) -> tuple:
    return (tuple(grid.shape), tuple(int(value) for value in grid.flat))


def _render_from_positions(
    input_grid: np.ndarray,
    source_pos: tuple[int, int],
    destination_pos: tuple[int, int],
    connector_color: int,
) -> np.ndarray:
    sr, sc = source_pos
    dr, dc = destination_pos
    output = input_grid.copy()
    horizontal = 1 if dc > sc else -1
    vertical = 1 if dr > sr else -1
    for col in range(sc + horizontal, dc + horizontal, horizontal):
        output[sr, col] = connector_color
    for row in range(sr + vertical, dr, vertical):
        output[row, dc] = connector_color
    return output


def _make_pair(
    shape: tuple[int, int],
    source_pos: tuple[int, int],
    destination_pos: tuple[int, int],
    palette: tuple[int, int, int, int],
) -> dict:
    background, source, destination, connector = palette
    input_grid = np.full(shape, background, dtype=np.int8)
    input_grid[source_pos] = source
    input_grid[destination_pos] = destination
    return {
        "input": input_grid,
        "output": _render_from_positions(
            input_grid, source_pos, destination_pos, connector
        ),
    }


def _check_optional_count(name: str, value: int | None, low: int, high: int) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    rng: np.random.Generator, num_train: int | None, num_test: int | None
) -> tuple[int, int]:
    _check_optional_count("num_train", num_train, 2, 10)
    _check_optional_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        # Short teaching sets are common, while the complete count range remains live.
        train_count = int(rng.choice(
            np.arange(2, 11),
            p=(0.18, 0.18, 0.18, 0.10, 0.09, 0.08, 0.07, 0.06, 0.06),
        ))
    elif num_train is None:
        train_count = int(rng.integers(max(2, int(num_test) + 1), 11))
    else:
        train_count = int(num_train)
    if num_test is None:
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    else:
        test_count = int(num_test)
    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")
    return train_count, test_count


def _sample_shape(
    rng: np.random.Generator, previous: list[tuple[int, int]]
) -> tuple[tuple[int, int], bool]:
    if rng.random() < 0.10:
        return (
            (int(rng.integers(20, 31)), int(rng.integers(20, 31))),
            True,
        )
    if previous and rng.random() < 0.22:
        return previous[int(rng.integers(len(previous)))], False

    def dimension() -> int:
        bucket = int(rng.choice(3, p=(0.22, 0.55, 0.23)))
        bounds = ((4, 8), (8, 23), (23, 31))[bucket]
        return int(rng.integers(*bounds))

    return (dimension(), dimension()), False


def _sample_axis_pair(
    rng: np.random.Generator, length: int, sign: int, compact: bool
) -> tuple[int, int]:
    # The two axis gaps are sampled separately, so short and long legs combine.
    if compact:
        gap = int(rng.integers(1, 4))
    else:
        ranges = ((1, 1), (2, 3), (4, 7), (8, length - 1))
        available = [(low, min(high, length - 1)) for low, high in ranges if low < length]
        low, high = available[int(rng.integers(len(available)))]
        gap = int(rng.integers(low, high + 1))
    maximum = length - gap - 1
    if rng.random() < 0.38:
        start = (0, maximum)[int(rng.integers(2))]
    else:
        start = int(rng.integers(maximum + 1))
    return (start, start + gap) if sign == 1 else (start + gap, start)


def _sample_positions(
    rng: np.random.Generator, shape: tuple[int, int],
    orientation: tuple[int, int], compact: bool
) -> tuple[tuple[int, int], tuple[int, int]]:
    sr, dr = _sample_axis_pair(rng, shape[0], orientation[0], compact)
    sc, dc = _sample_axis_pair(rng, shape[1], orientation[1], compact)
    return (sr, sc), (dr, dc)


def _matches_source_pair(pair: dict) -> bool:
    if tuple(
        int(np.count_nonzero(pair["input"] == color))
        for color in _SOURCE_PALETTE[:3]
    ) == (pair["input"].size - 2, 1, 1):
        for height, width, sr, sc, dr, dc in _SOURCE_SPECS:
            if pair["input"].shape != (height, width):
                continue
            source = _make_pair(
                (height, width), (sr, sc), (dr, dc), _SOURCE_PALETTE
            )
            if np.array_equal(pair["input"], source["input"]) and np.array_equal(
                pair["output"], source["output"]
            ):
                return True
    return False


def _new_pair(
    rng: np.random.Generator,
    shape: tuple[int, int],
    orientation: tuple[int, int],
    palette: tuple[int, int, int, int],
    seen_inputs: set[tuple],
    compact: bool,
) -> dict:
    source_pos, destination_pos = _sample_positions(rng, shape, orientation, compact)
    pair = _make_pair(shape, source_pos, destination_pos, palette)
    if _grid_key(pair["input"]) not in seen_inputs and not _matches_source_pair(pair):
        return pair

    # Deterministic fallback for an unlikely duplicate or exact source pair.
    height, width = shape
    backgrounds = [palette[0]] + [
        color for color in range(10) if color not in palette
    ]
    for background in backgrounds:
        other_palette = (background, *palette[1:])
        for sr in range(height):
            for dr in range(height):
                if (dr - sr) * orientation[0] <= 0:
                    continue
                for sc in range(width):
                    for dc in range(width):
                        if (dc - sc) * orientation[1] <= 0:
                            continue
                        pair = _make_pair(
                            shape, (sr, sc), (dr, dc), other_palette
                        )
                        if _grid_key(pair["input"]) not in seen_inputs and not _matches_source_pair(pair):
                            return pair
    raise AssertionError("no distinct endpoint placement is available")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)

    # Foreground colors are fixed conventions. Each input's uniform background
    # is visible locally and may vary independently between pairs.
    source, destination, connector = (
        int(color) for color in rng.choice(10, size=3, replace=False)
    )
    backgrounds = [
        color for color in range(10)
        if color not in (source, destination, connector)
    ]
    first_background = backgrounds[int(rng.integers(len(backgrounds)))]
    second_background = [
        color for color in backgrounds if color != first_background
    ][int(rng.integers(len(backgrounds) - 1))]
    shared_shape = (
        _sample_shape(rng, [])[0] if rng.random() < 0.10 else None
    )

    first_orientation = _ORIENTATIONS[int(rng.integers(4))]
    orientations = [
        first_orientation,
        (-first_orientation[0], -first_orientation[1]),
    ]
    orientations.extend(
        _ORIENTATIONS[int(rng.integers(4))]
        for _ in range(train_count - 2)
    )
    order = [int(index) for index in rng.permutation(train_count)]
    orientations = [orientations[index] for index in order]

    train: list[dict] = []
    test: list[dict] = []
    seen_inputs: set[tuple] = set()
    shapes: list[tuple[int, int]] = []
    for index, orientation in enumerate(orientations):
        shape, compact = (
            (shared_shape, False) if shared_shape is not None
            else _sample_shape(rng, shapes)
        )
        background = (
            first_background if index == 0 else
            second_background if index == 1 else
            backgrounds[int(rng.integers(len(backgrounds)))]
        )
        pair = _new_pair(
            rng, shape, orientation,
            (background, source, destination, connector),
            seen_inputs,
            compact,
        )
        train.append(pair)
        shapes.append(shape)
        seen_inputs.add(_grid_key(pair["input"]))

    familiar = set(orientations)
    unfamiliar = [orientation for orientation in _ORIENTATIONS if orientation not in familiar]
    for _ in range(test_count):
        if unfamiliar and rng.random() < 0.42:
            orientation = unfamiliar[int(rng.integers(len(unfamiliar)))]
        else:
            orientation = _ORIENTATIONS[int(rng.integers(4))]
        shape, compact = (
            (shared_shape, False) if shared_shape is not None
            else _sample_shape(rng, shapes)
        )
        background = backgrounds[int(rng.integers(len(backgrounds)))]
        pair = _new_pair(
            rng, shape, orientation,
            (background, source, destination, connector),
            seen_inputs,
            compact,
        )
        test.append(pair)
        shapes.append(shape)
        seen_inputs.add(_grid_key(pair["input"]))

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    _require(input_grid.shape == output_grid.shape, "input/output shapes differ")
    values, counts = np.unique(input_grid, return_counts=True)
    _require(len(values) == 3, "input must have a background and two marker colors")
    backgrounds = [
        int(value) for value, count in zip(values, counts)
        if count == input_grid.size - 2
    ]
    markers = [
        int(value) for value, count in zip(values, counts)
        if count == 1
    ]
    _require(len(backgrounds) == 1 and len(markers) == 2,
             "input must contain two distinct singleton markers")
    output_colors = {int(value) for value in np.unique(output_grid)}
    new_colors = output_colors - {int(value) for value in values}
    _require(len(new_colors) == 1, "output must introduce one connector color")
    connector = next(iter(new_colors))
    _require(
        output_colors == {int(value) for value in values} | {connector},
        "output has an invalid palette",
    )
    matches = []
    for source, destination in ((markers[0], markers[1]), (markers[1], markers[0])):
        source_pos = tuple(int(v) for v in np.argwhere(input_grid == source)[0])
        destination_pos = tuple(int(v) for v in np.argwhere(input_grid == destination)[0])
        if source_pos[0] == destination_pos[0] or source_pos[1] == destination_pos[1]:
            continue
        expected = _render_from_positions(
            input_grid, source_pos, destination_pos, connector
        )
        if np.array_equal(expected, output_grid):
            matches.append((source, destination, source_pos, destination_pos))
    _require(len(matches) == 1, "pair does not encode a unique directed elbow")
    source, destination, source_pos, destination_pos = matches[0]
    sr, sc = source_pos
    dr, dc = destination_pos
    return {
        "background": backgrounds[0],
        "source": source,
        "destination": destination,
        "connector": connector,
        "source_pos": source_pos,
        "destination_pos": destination_pos,
        "shape": tuple(input_grid.shape),
        "orientation": (1 if dr > sr else -1, 1 if dc > sc else -1),
        "vertical_distance": abs(dr - sr),
        "horizontal_distance": abs(dc - sc),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(all(isinstance(puzzle[split], list) for split in ("train", "test")),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain exactly input and output")
            for grid in (pair["input"], pair["output"]):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                _require(all(1 <= dimension <= 30 for dimension in grid.shape),
                         "grid dimensions must be within ARC limits")
                _require(np.issubdtype(grid.dtype, np.integer),
                         "grid dtype must be integer")
                _require(bool(np.all((grid >= 0) & (grid <= 9))),
                         "grid colors must be between 0 and 9")
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate generated input or pair")
            _require(not _matches_source_pair(pair), "exact source pair reproduced")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            infos[split].append(_analyze_pair(pair["input"], pair["output"]))

    learned = tuple(infos["train"][0][role] for role in ("source", "destination", "connector"))
    for info in infos["train"] + infos["test"]:
        _require(tuple(info[role] for role in ("source", "destination", "connector")) == learned,
                 "foreground color roles are inconsistent")
        _require(info["background"] not in learned,
                 "background collides with a foreground role")
    train_orientations = {info["orientation"] for info in infos["train"]}
    _require({orientation[0] for orientation in train_orientations} == {-1, 1}
             and {orientation[1] for orientation in train_orientations} == {-1, 1},
             "training does not distinguish color roles from geometric position")

    # Infer the answer using only training-derived roles and each test input.
    source, destination, connector = learned
    for pair in puzzle["test"]:
        input_grid = pair["input"]
        _require(np.count_nonzero(input_grid == source) == 1
                 and np.count_nonzero(input_grid == destination) == 1,
                 "test markers do not match trained roles")
        source_pos = tuple(int(v) for v in np.argwhere(input_grid == source)[0])
        destination_pos = tuple(int(v) for v in np.argwhere(input_grid == destination)[0])
        expected = _render_from_positions(
            input_grid, source_pos, destination_pos, connector
        )
        _require(np.array_equal(expected, pair["output"]),
                 "test output is not recoverable from training roles")


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        tuple((_grid_key(pair["input"]), _grid_key(pair["output"])) for pair in puzzle[split])
        for split in ("train", "test")
    )


def self_test() -> None:
    for height, width, sr, sc, dr, dc in _SOURCE_SPECS:
        _require(
            _matches_source_pair(
                _make_pair(
                    (height, width), (sr, sc), (dr, dc), _SOURCE_PALETTE
                )
            ),
            "source-copy guard missed an original pair",
        )
    source_palette_puzzle = {
        "train": [
            _make_pair((8, 9), (1, 2), (6, 7), _SOURCE_PALETTE),
            _make_pair((9, 8), (7, 6), (2, 1), _SOURCE_PALETTE),
        ],
        "test": [
            _make_pair((7, 10), (5, 1), (1, 8), _SOURCE_PALETTE),
        ],
    }
    validate_generated_puzzle(source_palette_puzzle)

    automatic = [generate_puzzle(seed) for seed in range(42)]
    for seed, puzzle in enumerate(automatic):
        validate_generated_puzzle(puzzle)
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)),
                 "generation is nondeterministic")
    _require(len({_puzzle_key(puzzle) for puzzle in automatic}) == 42,
             "different seeds generated identical puzzles")
    _require(len({len(puzzle["train"]) for puzzle in automatic}) >= 4,
             "train counts lack diversity")
    _require(len({len(puzzle["test"]) for puzzle in automatic}) >= 3,
             "test counts lack diversity")

    for index, (train_count, test_count) in enumerate(
        ((2, 1), (10, 1), (10, 4), (6, 3))
    ):
        puzzle = generate_puzzle(1000 + index, train_count, test_count)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count),
                 "explicit counts were ignored")
    for index, (train_count, test_count) in enumerate(
        ((None, 1), (None, 4), (2, None), (10, None))
    ):
        puzzle = generate_puzzle(2000 + index, train_count, test_count)
        _require(train_count is None or len(puzzle["train"]) == train_count,
                 "partial train count was ignored")
        _require(test_count is None or len(puzzle["test"]) == test_count,
                 "partial test count was ignored")
    for train_count, test_count in (
        (1, None), (11, None), (None, 0), (None, 5),
        (2, 2), (3, 4), (2.5, 1),
    ):
        try:
            generate_puzzle(77, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count arguments were accepted")

    shapes = set()
    layouts = set()
    palettes = set()
    aspects = set()
    train_orientations = set()
    test_orientations = set()
    test_regimes = set()
    one_step_tests = 0
    boundary_tests = 0
    compact_tests = 0
    mixed_leg_tests = 0
    new_shape_tests = 0
    reused_shape_tests = 0
    corner_tests = 0
    small_axis_tests = 0
    near30_axis_tests = 0
    extreme_aspect_tests = 0
    compact_large_tests = 0
    combined_shift_tests = 0
    varying_background_puzzles = 0
    single_shape_puzzles = 0
    repeated_train_shape_puzzles = 0
    concise_puzzles = 0
    for puzzle in automatic:
        train_infos = [
            _analyze_pair(pair["input"], pair["output"]) for pair in puzzle["train"]
        ]
        test_infos = [
            _analyze_pair(pair["input"], pair["output"]) for pair in puzzle["test"]
        ]
        backgrounds = {info["background"] for info in train_infos + test_infos}
        varying_background_puzzles += len(backgrounds) > 1
        train_shapes = {info["shape"] for info in train_infos}
        single_shape_puzzles += len({info["shape"] for info in train_infos + test_infos}) == 1
        repeated_train_shape_puzzles += len(train_shapes) < len(train_infos)
        concise_puzzles += len(train_infos) <= 4 and len(test_infos) == 1
        train_orientations_in_puzzle = {info["orientation"] for info in train_infos}
        palettes.add(tuple(train_infos[0][role] for role in
                           ("source", "destination", "connector")))
        for split, pairs, infos in (
            ("train", puzzle["train"], train_infos),
            ("test", puzzle["test"], test_infos),
        ):
            for pair, info in zip(pairs, infos):
                sr, sc = info["source_pos"]
                dr, dc = info["destination_pos"]
                expected_changed = {
                    (sr, col) for col in range(min(sc, dc) + 1, max(sc, dc) + 1)
                } if sc < dc else {
                    (sr, col) for col in range(min(sc, dc), max(sc, dc))
                }
                expected_changed |= {
                    (row, dc) for row in range(min(sr, dr) + 1, max(sr, dr))
                }
                changed = {tuple(int(v) for v in cell)
                           for cell in np.argwhere(pair["input"] != pair["output"])}
                _require(changed == expected_changed,
                         "connector cells do not form the directed elbow")
                _require(all(pair["output"][cell] == info["connector"] for cell in changed),
                         "path has the wrong connector color")
                _require(pair["input"][sr, sc] == pair["output"][sr, sc]
                         and pair["input"][dr, dc] == pair["output"][dr, dc],
                         "endpoint was overwritten")
                _require(not _matches_source_pair(pair), "source pair reproduced")
                shapes.add(info["shape"])
                layouts.add((info["shape"], info["source_pos"],
                             info["destination_pos"]))
                height, width = info["shape"]
                aspects.add("landscape" if width > height + 2 else
                            "portrait" if height > width + 2 else "balanced")
                if split == "train":
                    train_orientations.add(info["orientation"])
                else:
                    test_orientations.add(info["orientation"])
                    test_regimes.add("familiar" if info["orientation"] in train_orientations_in_puzzle
                                     else "new")
                    one_step_tests += (info["vertical_distance"] == 1
                                       or info["horizontal_distance"] == 1)
                    compact_tests += (info["vertical_distance"] <= 3
                                      and info["horizontal_distance"] <= 3)
                    mixed_leg_tests += (
                        (info["vertical_distance"] <= 3 and info["horizontal_distance"] >= 8)
                        or (info["horizontal_distance"] <= 3 and info["vertical_distance"] >= 8)
                    )
                    boundary_tests += (
                        sr in (0, height - 1) or dr in (0, height - 1)
                        or sc in (0, width - 1) or dc in (0, width - 1)
                    )
                    corner_tests += (
                        (sr in (0, height - 1) and sc in (0, width - 1))
                        or (dr in (0, height - 1) and dc in (0, width - 1))
                    )
                    small_axis_tests += min(height, width) <= 7
                    near30_axis_tests += max(height, width) >= 27
                    extreme_aspect_tests += max(height, width) >= 3 * min(height, width)
                    compact_large_tests += (height >= 20 and width >= 20
                                            and info["vertical_distance"] <= 3
                                            and info["horizontal_distance"] <= 3)
                    is_new_shape = info["shape"] not in train_shapes
                    new_shape_tests += is_new_shape
                    reused_shape_tests += not is_new_shape
                    combined_shift_tests += (
                        info["orientation"] not in train_orientations_in_puzzle
                        and is_new_shape
                        and (sr in (0, height - 1) or dr in (0, height - 1)
                             or sc in (0, width - 1) or dc in (0, width - 1))
                    )
    _require(len(palettes) >= 35, "foreground palettes lack diversity")
    _require(len(shapes) >= 75 and len(layouts) >= 160,
             "grid shapes or endpoint layouts lack diversity")
    _require(aspects == {"landscape", "portrait", "balanced"},
             "aspect ratios lack diversity")
    _require(train_orientations == set(_ORIENTATIONS)
             and test_orientations == set(_ORIENTATIONS)
             and test_regimes == {"familiar", "new"},
             "orientation regimes lack diversity")
    _require(varying_background_puzzles >= 35,
             "within-puzzle backgrounds lack diversity")
    _require(min(one_step_tests, compact_tests, mixed_leg_tests,
                 boundary_tests, new_shape_tests) >= 2,
             "reviewed test geometry lacks diversity")
    _require(
        concise_puzzles >= 8 and single_shape_puzzles >= 2
        and repeated_train_shape_puzzles >= 5,
        "count or within-puzzle shape regimes lack diversity",
    )
    _require(
        min(reused_shape_tests, corner_tests, small_axis_tests,
            near30_axis_tests, extreme_aspect_tests,
            compact_large_tests, combined_shift_tests) >= 2,
        "reviewed shape and boundary combinations lack diversity",
    )
