import numpy as np


# Rejecting every exact source input is stronger than rejecting exact source
# input/output pairs: an original pair cannot be duplicated if its input is not.
_SOURCE_INPUT_CODES = frozenset(
    {
        (
            "000020000000/000020000000/000020000000/000020000000/"
            "000020000000/000333000000/000323000000/000333000000/"
            "000020000000/000020000000/000020000000/000020000000"
        ),
        (
            "000200000200/003330000200/003230000200/003330000200/"
            "000200000200/000200000200/000200000200/000200000200/"
            "000200000200/000200000200/000200003330/000200003230/"
            "000200003330/000200000200/000200000200"
        ),
        (
            "00000000000000/00000000000000/00033300000000/"
            "22232322222222/00033300000000/00000000000000/"
            "00000000000000/00000000000000/22222222222222/"
            "00000000000000/00000000000000/00000000000000/"
            "00000000000000/00000000033300/22222222232322/"
            "00000000033300"
        ),
        (
            "0000333000/2222323222/0000333000/0000000000/"
            "0000000000/0000000000/2222222222/0000000000/"
            "0000000000/0000000000"
        ),
        (
            "002000020002000020/033300020002000020/"
            "032300020002000020/033300020002000020/"
            "002000020002000020/002000020002000020/"
            "002000020002000020/002000020002000020/"
            "002000020002000020/002000020002000020/"
            "002000020002000020/002000020033300020/"
            "002000020032300020/002000020033300020/"
            "002000020002000020/002000020002000020"
        ),
    }
)


def _grid_code(grid: np.ndarray) -> str:
    return "/".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _array_key(grid: np.ndarray) -> tuple:
    normalized = grid.astype(np.int8, copy=False)
    return tuple(grid.shape), normalized.tobytes()


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        (
            section,
            tuple(
                (
                    _array_key(pair["input"]),
                    _array_key(pair["output"]),
                )
                for pair in puzzle[section]
            ),
        )
        for section in ("train", "test")
    )


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first) != {"train", "test"}:
        return False
    if set(second) != {"train", "test"}:
        return False

    for section in ("train", "test"):
        if len(first[section]) != len(second[section]):
            return False
        for pair_a, pair_b in zip(first[section], second[section]):
            if set(pair_a) != {"input", "output"}:
                return False
            if set(pair_b) != {"input", "output"}:
                return False
            if not np.array_equal(pair_a["input"], pair_b["input"]):
                return False
            if not np.array_equal(pair_a["output"], pair_b["output"]):
                return False
    return True


def _validate_optional_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_optional_count(
        "num_train", num_train, 2, 10
    )
    num_test = _validate_optional_count(
        "num_test", num_test, 1, 4
    )

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        # This deterministic schedule guarantees broad count diversity for
        # consecutive seeds while remaining wholly seed-dependent.
        num_test = 1 + seed % 4
        number_of_legal_train_counts = 10 - num_test
        num_train = (
            num_test
            + 1
            + (seed // 4) % number_of_legal_train_counts
        )
    elif num_train is None:
        lower = max(2, num_test + 1)
        num_train = int(rng.integers(lower, 11))
    else:
        largest_test_count = min(4, num_train - 1)
        if largest_test_count < 1:
            raise ValueError("no legal test count exists")
        num_test = int(rng.integers(1, largest_test_count + 1))

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError("the requested pair counts are impossible")

    return num_train, num_test


def _sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    # The output-only color is the one convention that cannot be read from a
    # new input. The three visible roles are identified afresh by geometry.
    return {"signal": int(rng.integers(0, 10))}


def _random_inclusive(
    rng: np.random.Generator,
    lower: int,
    upper: int,
) -> int:
    if lower > upper:
        raise ValueError("empty integer range")
    return int(rng.integers(lower, upper + 1))


def _spaced_positions(
    rng: np.random.Generator,
    length: int,
    count: int,
    minimum_gap: int = 4,
) -> tuple[int, ...]:
    """Choose interior positions with the requested minimum separation."""
    if count < 1:
        raise ValueError("at least one position is required")

    # Positions initially lie in [1, length - 2]. Compressing the required
    # gaps turns this into ordinary sampling without replacement.
    compressed_max = (
        length - 2 - (minimum_gap - 1) * (count - 1)
    )
    if compressed_max < count:
        raise ValueError("dimension is too small for spaced positions")

    compressed = np.sort(
        rng.choice(
            np.arange(1, compressed_max + 1),
            size=count,
            replace=False,
        )
    )
    expanded = compressed + (
        np.arange(count) * (minimum_gap - 1)
    )
    return tuple(int(value) for value in expanded)


def _construct_output(
    input_grid: np.ndarray,
    analysis: dict,
    signal: int,
) -> np.ndarray:
    output = input_grid.copy()

    background = analysis["background"]
    orientation = analysis["orientation"]
    carriers = analysis["carriers"]
    sources = analysis["sources"]

    protected_source_cells = set()
    for center_row, center_col in sources:
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                protected_source_cells.add(
                    (center_row + delta_row, center_col + delta_col)
                )

    source_centers = set(sources)
    height, width = input_grid.shape

    def stamp_signal_ring(center_row: int, center_col: int) -> None:
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                if delta_row == 0 and delta_col == 0:
                    continue
                row = center_row + delta_row
                col = center_col + delta_col
                if (row, col) not in protected_source_cells:
                    output[row, col] = signal

    if orientation == "vertical":
        for source_row, source_col in sources:
            for col in range(width):
                if input_grid[source_row, col] == background:
                    output[source_row, col] = signal

            for carrier_col in carriers:
                crossing = (source_row, carrier_col)
                if crossing not in source_centers:
                    stamp_signal_ring(source_row, carrier_col)
    else:
        for source_row, source_col in sources:
            for row in range(height):
                if input_grid[row, source_col] == background:
                    output[row, source_col] = signal

            for carrier_row in carriers:
                crossing = (carrier_row, source_col)
                if crossing not in source_centers:
                    stamp_signal_ring(carrier_row, source_col)

    return output


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    puzzle_parameters: dict,
    spec: dict,
    previous_palette: tuple[int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown generation regime")
    orientation = spec["orientation"]
    carrier_count = spec["carriers"]
    source_count = spec["sources"]
    mode = spec["mode"]
    axis_count = source_count if mode in ("ordinary", "repeated") else source_count - 1

    def dimension(minimum: int) -> int:
        # Small and large grids are independently available in both regimes.
        upper = min(30, 16 if minimum <= 16 and rng.integers(2) else 30)
        return _random_inclusive(rng, minimum, upper)

    carrier_dimension = dimension(max(7, 4 * carrier_count - 1))
    axis_dimension = dimension(max(7, 4 * axis_count - 1))

    if orientation == "vertical":
        height = axis_dimension
        width = carrier_dimension
    else:
        height = carrier_dimension
        width = axis_dimension

    signal = puzzle_parameters["signal"]
    available = [color for color in range(10) if color != signal]
    palette = tuple(int(v) for v in rng.choice(available, 3, replace=False))
    if palette == previous_palette:
        palette = (palette[1], palette[2], palette[0])
    background, carrier, source = palette

    grid = np.full(
        (height, width), background, dtype=np.int8
    )

    carriers = _spaced_positions(
        rng, carrier_dimension, carrier_count
    )
    source_axes = _spaced_positions(rng, axis_dimension, axis_count)
    carrier_order = tuple(int(v) for v in rng.permutation(carriers))
    axis_order = tuple(int(v) for v in rng.permutation(source_axes))
    if mode == "ordinary":
        placements = list(zip(carrier_order, axis_order))
    elif mode == "shared":
        placements = [(carrier_order[0], axis_order[0]),
                      (carrier_order[1], axis_order[0])]
    elif mode == "repeated":
        placements = [(carrier_order[0], axis_order[0]),
                      (carrier_order[0], axis_order[1])]
    else:
        placements = [(carrier_order[0], axis_order[0]),
                      (carrier_order[1], axis_order[0]),
                      (carrier_order[0], axis_order[1])]
    remaining = [(c, a) for c in carrier_order for a in axis_order
                 if (c, a) not in placements]
    rng.shuffle(remaining)
    placements += remaining[:source_count - len(placements)]

    if orientation == "vertical":
        grid[:, carriers] = carrier
        sources = tuple(sorted((a, c) for c, a in placements))
    else:
        grid[carriers, :] = carrier
        sources = tuple(sorted((c, a) for c, a in placements))

    for center_row, center_col in sources:
        grid[
            center_row - 1:center_row + 2,
            center_col - 1:center_col + 2,
        ] = source
        grid[center_row, center_col] = carrier

    analysis = {
        "background": background,
        "carrier": carrier,
        "source": source,
        "orientation": orientation,
        "carriers": tuple(sorted(carriers)),
        "sources": tuple(sorted(sources)),
    }

    output = _construct_output(grid, analysis, signal)
    return grid, output


def _analyze_input(grid: np.ndarray) -> dict:
    """Infer semantic roles and geometry using only an input grid."""
    colors = tuple(int(value) for value in np.unique(grid))
    assert len(colors) == 3, "an input must contain exactly three colors"

    height, width = grid.shape
    candidates = []

    for background in colors:
        for carrier in colors:
            if carrier == background:
                continue

            marker_candidates = [
                value
                for value in colors
                if value not in (background, carrier)
            ]
            if len(marker_candidates) != 1:
                continue
            source = marker_candidates[0]

            for orientation in ("vertical", "horizontal"):
                if orientation == "vertical":
                    carriers = tuple(
                        int(value)
                        for value in np.flatnonzero(
                            np.any(grid == carrier, axis=0)
                        )
                    )
                else:
                    carriers = tuple(
                        int(value)
                        for value in np.flatnonzero(
                            np.any(grid == carrier, axis=1)
                        )
                    )

                if not carriers:
                    continue

                sources = []
                for center_row in range(1, height - 1):
                    for center_col in range(1, width - 1):
                        if grid[center_row, center_col] != carrier:
                            continue

                        ring_matches = True
                        for delta_row in (-1, 0, 1):
                            for delta_col in (-1, 0, 1):
                                if delta_row == 0 and delta_col == 0:
                                    continue
                                if (
                                    grid[
                                        center_row + delta_row,
                                        center_col + delta_col,
                                    ]
                                    != source
                                ):
                                    ring_matches = False
                                    break
                            if not ring_matches:
                                break

                        if ring_matches:
                            sources.append(
                                (center_row, center_col)
                            )

                if not sources:
                    continue

                expected = np.full(
                    grid.shape, background, dtype=grid.dtype
                )
                if orientation == "vertical":
                    expected[:, carriers] = carrier
                else:
                    expected[carriers, :] = carrier

                for center_row, center_col in sources:
                    expected[
                        center_row - 1:center_row + 2,
                        center_col - 1:center_col + 2,
                    ] = source
                    expected[center_row, center_col] = carrier

                if np.array_equal(expected, grid):
                    candidates.append(
                        {
                            "background": background,
                            "carrier": carrier,
                            "source": source,
                            "orientation": orientation,
                            "carriers": tuple(sorted(carriers)),
                            "sources": tuple(sorted(sources)),
                        }
                    )

    assert len(candidates) == 1, (
        "input roles or carrier orientation are ambiguous"
    )
    return candidates[0]


def _validate_analysis_geometry(
    grid: np.ndarray,
    analysis: dict,
) -> None:
    height, width = grid.shape
    orientation = analysis["orientation"]
    carriers = analysis["carriers"]
    sources = analysis["sources"]

    if orientation == "vertical":
        carrier_limit = width
        source_axes = sorted(row for row, _ in sources)
        source_carriers = [col for _, col in sources]
    else:
        carrier_limit = height
        source_axes = sorted(col for _, col in sources)
        source_carriers = [row for row, _ in sources]

    assert all(
        1 <= position < carrier_limit - 1
        for position in carriers
    )
    assert all(
        later - earlier >= 4
        for earlier, later in zip(carriers, carriers[1:])
    )

    assert all(
        later == earlier or later - earlier >= 4
        for earlier, later in zip(source_axes, source_axes[1:])
    )
    assert len(sources) == len(set(sources))
    assert all(
        position in carriers for position in source_carriers
    )


def _validate_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    height, width = grid.shape
    assert 1 <= height <= 30
    assert 1 <= width <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((grid >= 0) & (grid <= 9))


def _validate_pair(pair: dict) -> dict:
    assert isinstance(pair, dict)
    assert set(pair) == {"input", "output"}

    input_grid = pair["input"]
    output_grid = pair["output"]

    _validate_grid(input_grid)
    _validate_grid(output_grid)

    assert input_grid.shape == output_grid.shape
    assert not np.array_equal(input_grid, output_grid)

    analysis = _analyze_input(input_grid)
    _validate_analysis_geometry(input_grid, analysis)

    input_colors = set(
        int(value) for value in np.unique(input_grid)
    )
    output_colors = set(
        int(value) for value in np.unique(output_grid)
    )
    added_colors = output_colors - input_colors

    assert len(added_colors) == 1
    signal = next(iter(added_colors))
    assert output_colors == input_colors | {signal}

    expected_output = _construct_output(
        input_grid, analysis, signal
    )
    assert np.array_equal(output_grid, expected_output), (
        "output does not follow the emitter transformation"
    )

    # Since no original source input is allowed, no original pair can occur.
    assert _grid_code(input_grid) not in _SOURCE_INPUT_CODES

    result = dict(analysis)
    result["signal"] = signal
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    train_count = len(puzzle["train"])
    test_count = len(puzzle["test"])

    assert 2 <= train_count <= 10
    assert 1 <= test_count <= 4
    assert train_count > test_count

    seen_inputs = set()
    seen_pairs = set()
    train_analyses = []
    test_analyses = []
    signal_color = None

    for section in ("train", "test"):
        for pair in puzzle[section]:
            analysis = _validate_pair(pair)

            input_key = _array_key(pair["input"])
            output_key = _array_key(pair["output"])
            pair_key = (input_key, output_key)

            assert input_key not in seen_inputs
            assert pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

            roles = (analysis["background"], analysis["carrier"],
                     analysis["source"], analysis["signal"])
            assert len(set(roles)) == 4
            if signal_color is None:
                signal_color = analysis["signal"]
            assert analysis["signal"] == signal_color

            if section == "train":
                train_analyses.append(analysis)
            else:
                test_analyses.append(analysis)

    assert all(1 <= len(a["carriers"]) <= 4 and
               1 <= len(a["sources"]) <= 4 for a in train_analyses)
    assert all(1 <= len(a["carriers"]) <= 6 and
               1 <= len(a["sources"]) <= 4 for a in test_analyses)
    assert any(len(a["carriers"]) > len({
        c if a["orientation"] == "vertical" else r
        for r, c in a["sources"]
    }) for a in train_analyses), "training must show a copied ring"
    train_orientations = {a["orientation"] for a in train_analyses}
    assert all(a["orientation"] in train_orientations for a in test_analyses)

    def features(a: dict) -> set[str]:
        axes = [r if a["orientation"] == "vertical" else c
                for r, c in a["sources"]]
        used_carriers = [c if a["orientation"] == "vertical" else r
                         for r, c in a["sources"]]
        result = set()
        if len(set(axes)) < len(axes):
            result.add("shared")
        if len(set(used_carriers)) < len(used_carriers):
            result.add("repeated")
        if len(a["sources"]) > len(a["carriers"]):
            result.add("more_sources")
        return result

    train_features = set().union(*(features(a) for a in train_analyses))
    assert all(features(a) <= train_features for a in test_analyses)
    assert len({(a["background"], a["carrier"], a["source"])
                for a in train_analyses}) >= 2


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")

    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(
        seed, rng, num_train, num_test
    )
    puzzle_parameters = _sample_puzzle_parameters(rng)
    orientation_plan = int(rng.integers(3))
    if orientation_plan == 0:
        train_orientations = ["vertical"] * train_count
    elif orientation_plan == 1:
        train_orientations = ["horizontal"] * train_count
    else:
        first = "vertical" if rng.integers(2) else "horizontal"
        other = "horizontal" if first == "vertical" else "vertical"
        train_orientations = [first, other] + [
            (first if rng.integers(2) else other)
            for _ in range(train_count - 2)
        ]

    def pair_features(analysis: dict) -> set[str]:
        axes = [r if analysis["orientation"] == "vertical" else c
                for r, c in analysis["sources"]]
        carriers = [c if analysis["orientation"] == "vertical" else r
                    for r, c in analysis["sources"]]
        features = set()
        if len(set(axes)) < len(axes):
            features.add("shared")
        if len(set(carriers)) < len(carriers):
            features.add("repeated")
        if len(analysis["sources"]) > len(analysis["carriers"]):
            features.add("more_sources")
        return features

    def make_spec(section: str, slot: int, allowed: set[str]) -> dict:
        if section == "train":
            orientation = train_orientations[slot]
            if slot == 0:
                return {"orientation": orientation,
                        "carriers": _random_inclusive(rng, 2, 4),
                        "sources": 1, "mode": "ordinary"}
            mode = ("ordinary", "shared", "repeated", "both")[
                int(rng.integers(4))]
            lower = 2 if mode in ("shared", "both") else 1
            carrier_count = _random_inclusive(rng, lower, 4)
        else:
            orientation = train_orientations[int(rng.integers(train_count))]
            if {"repeated", "more_sources"} <= allowed and rng.integers(5) == 0:
                return {"orientation": orientation,
                        "carriers": _random_inclusive(rng, 1, 2),
                        "sources": _random_inclusive(rng, 3, 4),
                        "mode": "repeated"}
            if rng.integers(5) == 0:
                return {"orientation": orientation,
                        "carriers": _random_inclusive(rng, 4, 6),
                        "sources": 1, "mode": "ordinary"}
            modes = ["ordinary"]
            if "shared" in allowed:
                modes.append("shared")
            if "repeated" in allowed:
                modes.append("repeated")
            if {"shared", "repeated"} <= allowed:
                modes.append("both")
            mode = modes[int(rng.integers(len(modes)))]
            lower = 2 if mode in ("shared", "both") else 1
            carrier_count = _random_inclusive(rng, lower, 6)
        if mode == "ordinary":
            source_count = _random_inclusive(rng, 1, min(4, carrier_count))
        elif mode == "both":
            source_count = _random_inclusive(rng, 3, 4)
        else:
            source_count = _random_inclusive(rng, 2, 4)
        return {"orientation": orientation, "carriers": carrier_count,
                "sources": source_count, "mode": mode}

    total_pairs = train_count + test_count
    pair_tokens = rng.integers(
        0,
        np.iinfo(np.int64).max,
        size=total_pairs,
        dtype=np.int64,
    )

    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    seen_pairs = set()
    global_index = 0
    previous_palette = None
    demonstrated_features: set[str] = set()

    for section, count, regime in (
        ("train", train_count, "source_train"),
        ("test", test_count, "target_test"),
    ):
        for slot in range(count):
            token = int(pair_tokens[global_index])
            # Each pair receives a distinct deterministic child generator.
            pair_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [token, global_index, seed]
                )
            )

            accepted_pair = None
            for _ in range(200):
                spec = make_spec(section, slot, demonstrated_features)
                input_grid, output_grid = _generate_pair(
                    pair_rng,
                    regime,
                    puzzle_parameters,
                    spec,
                    previous_palette,
                )

                analysis = _analyze_input(input_grid)
                features = pair_features(analysis)
                if section == "test" and not features <= demonstrated_features:
                    continue

                input_key = _array_key(input_grid)
                pair_key = (
                    input_key,
                    _array_key(output_grid),
                )

                if input_key in seen_inputs:
                    continue
                if pair_key in seen_pairs:
                    continue
                if _grid_code(input_grid) in _SOURCE_INPUT_CODES:
                    continue

                accepted_pair = {
                    "input": input_grid,
                    "output": output_grid,
                }
                seen_inputs.add(input_key)
                seen_pairs.add(pair_key)
                previous_palette = (analysis["background"],
                                    analysis["carrier"], analysis["source"])
                if section == "train":
                    demonstrated_features |= features
                break

            if accepted_pair is None:
                raise ValueError(
                    "could not generate a unique valid pair"
                )

            puzzle[section].append(accepted_pair)
            global_index += 1

    validate_generated_puzzle(puzzle)
    return puzzle


def _canonical_input(
    grid: np.ndarray,
    analysis: dict,
) -> np.ndarray:
    canonical = np.empty(grid.shape, dtype=np.int8)
    canonical[grid == analysis["background"]] = 0
    canonical[grid == analysis["carrier"]] = 1
    canonical[grid == analysis["source"]] = 2
    return canonical


def self_test() -> None:
    automatic_puzzles = []
    puzzle_fingerprints = set()
    train_counts = set()
    test_counts = set()
    role_assignments = set()
    shapes = set()
    canonical_layouts = set()

    # Automatic counts, task invariants, and 42-seed diversity sweep.
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        automatic_puzzles.append(puzzle)

        puzzle_fingerprints.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))

        first_analysis = _analyze_input(
            puzzle["train"][0]["input"]
        )
        first_output_colors = set(
            int(value)
            for value in np.unique(
                puzzle["train"][0]["output"]
            )
        )
        first_input_colors = set(
            int(value)
            for value in np.unique(
                puzzle["train"][0]["input"]
            )
        )
        first_signal = next(
            iter(first_output_colors - first_input_colors)
        )
        role_assignments.add(
            (
                first_analysis["background"],
                first_analysis["carrier"],
                first_analysis["source"],
                first_signal,
            )
        )

        for section in ("train", "test"):
            for pair in puzzle[section]:
                analysis = _analyze_input(pair["input"])
                shapes.add(tuple(pair["input"].shape))
                canonical_layouts.add(
                    _array_key(
                        _canonical_input(
                            pair["input"], analysis
                        )
                    )
                )

                assert (
                    _grid_code(pair["input"])
                    not in _SOURCE_INPUT_CODES
                )

                added = (
                    set(
                        int(value)
                        for value in np.unique(pair["output"])
                    )
                    - set(
                        int(value)
                        for value in np.unique(pair["input"])
                    )
                )
                assert len(added) == 1
                expected = _construct_output(
                    pair["input"],
                    analysis,
                    next(iter(added)),
                )
                assert np.array_equal(
                    expected, pair["output"]
                )

    assert len(train_counts) >= 4
    assert len(test_counts) >= 3
    assert len(puzzle_fingerprints) >= 40
    assert len(role_assignments) >= 30
    assert len(shapes) >= 35
    assert len(canonical_layouts) >= 120

    # Measure the review axes from returned grids, including combinations.
    observed = {}
    feature_combinations = set()

    def count(name: str, condition: bool = True) -> None:
        if condition:
            observed[name] = observed.get(name, 0) + 1

    for puzzle in automatic_puzzles:
        train = [_analyze_input(pair["input"]) for pair in puzzle["train"]]
        orientations = [a["orientation"] for a in train]
        palettes = {(a["background"], a["carrier"], a["source"])
                    for a in train}
        count("same_orientation", len(set(orientations)) == 1)
        count("mixed_orientation", len(set(orientations)) == 2)
        count("reversed_sequence", orientations[:2] ==
              ["horizontal", "vertical"])
        count("recolored_train", len(palettes) >= 2)
        maximum_train_carriers = max(len(a["carriers"]) for a in train)
        for section in ("train", "test"):
            for pair in puzzle[section]:
                a = _analyze_input(pair["input"])
                orientation = a["orientation"]
                carrier_count = len(a["carriers"])
                source_count = len(a["sources"])
                axes = [r if orientation == "vertical" else c
                        for r, c in a["sources"]]
                used_carriers = [c if orientation == "vertical" else r
                                 for r, c in a["sources"]]
                shared = len(set(axes)) < source_count
                repeated = len(set(used_carriers)) < source_count
                count(section + "_shared", shared)
                count(section + "_repeated", repeated)
                count(section + "_both", shared and repeated)
                count(section + "_more_sources", source_count > carrier_count)
                if section == "test":
                    count("horizontal_test", orientation == "horizontal")
                    count("large_family_test", carrier_count >= 5)
                    count("familiar_or_fewer_test",
                          carrier_count <= maximum_train_carriers)
                    count("single_carrier_test", carrier_count == 1)
                    count("lone_emitter_many_carriers",
                          source_count == 1 and carrier_count >= 4)
                    count("many_emitters_small_family",
                          source_count >= 3 and carrier_count <= 2)
                    count("compact_test", max(pair["input"].shape) <= 16)
                    count("large_test", max(pair["input"].shape) >= 24)
                    count("narrow_test", pair["input"].shape[1] < 18)
                    feature_combinations.add((orientation, carrier_count,
                                              source_count, shared, repeated))

    required = {
        "same_orientation": 8, "mixed_orientation": 8,
        "reversed_sequence": 3, "recolored_train": 42,
        "horizontal_test": 8, "large_family_test": 8,
        "familiar_or_fewer_test": 8, "single_carrier_test": 2,
        "lone_emitter_many_carriers": 5,
        "many_emitters_small_family": 3,
        "compact_test": 5, "large_test": 8, "narrow_test": 8,
        "train_shared": 8, "train_repeated": 8,
        "test_shared": 5, "test_repeated": 5,
        "test_both": 3, "test_more_sources": 3,
    }
    for name, minimum in required.items():
        assert observed.get(name, 0) >= minimum, (name, observed)
    assert len(feature_combinations) >= 25

    # Whole-puzzle determinism with explicit requested counts.
    deterministic_a = generate_puzzle(
        314159, num_train=7, num_test=3
    )
    deterministic_b = generate_puzzle(
        314159, num_train=7, num_test=3
    )
    assert _puzzles_equal(deterministic_a, deterministic_b)

    # Different seeds should produce different complete puzzles.
    different_a = generate_puzzle(
        271828, num_train=6, num_test=2
    )
    different_b = generate_puzzle(
        271829, num_train=6, num_test=2
    )
    assert not _puzzles_equal(different_a, different_b)

    # Legal explicit boundary cases.
    for seed, train_count, test_count in (
        (1001, 2, 1),
        (1002, 10, 1),
        (1003, 10, 4),
    ):
        puzzle = generate_puzzle(
            seed,
            num_train=train_count,
            num_test=test_count,
        )
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)

    # Additional explicitly requested non-boundary counts.
    for seed, train_count, test_count in (
        (1101, 3, 1),
        (1102, 6, 3),
        (1103, 9, 4),
    ):
        puzzle = generate_puzzle(
            seed,
            num_train=train_count,
            num_test=test_count,
        )
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)

    # Partial count specifications.
    partial_cases = (
        (1201, None, 1),
        (1202, None, 4),
        (1203, 2, None),
        (1204, 10, None),
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

    # Illegal individual counts and illegal relationships.
    illegal_arguments = (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4},
        {"num_train": 3.0, "num_test": 1},
        {"num_train": 3, "num_test": True},
    )

    for arguments in illegal_arguments:
        raised = False
        try:
            generate_puzzle(1300, **arguments)
        except ValueError:
            raised = True
        assert raised, f"expected ValueError for {arguments}"
