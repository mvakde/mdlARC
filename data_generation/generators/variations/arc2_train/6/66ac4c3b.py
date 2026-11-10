import hashlib
import numpy as np


_SOURCE_HASHES = {
    "1b8a32436a60a999773130c42dbee6d39be9f128c80dd3e5d0224097803a7b9b",
    "9e8dcf10bb1044846ef4b21243415d6b6525f0f64a7bb81da99ac3c0023cc900",
    "2268da940ec10f893837242aa557abfd0530be9ed7acc2b5d2eb57759fd07bfc",
}


def _source_hash(pair: dict) -> str:
    a = np.asarray(pair["input"], dtype=np.int8)
    b = np.asarray(pair["output"], dtype=np.int8)
    return hashlib.sha256(a.tobytes() + b.tobytes()).hexdigest()


def _grid_key(grid: np.ndarray) -> tuple:
    return (tuple(grid.shape), grid.dtype.str, grid.tobytes())


def _pair_key(pair: dict) -> tuple:
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _puzzle_key(puzzle: dict) -> tuple:
    return (
        tuple(_pair_key(pair) for pair in puzzle["train"]),
        tuple(_pair_key(pair) for pair in puzzle["test"]),
    )


def _validate_optional_count(
    value: int | None,
    name: str,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(
    rng: np.random.Generator,
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_optional_count(num_train, "num_train", 2, 10)
    test_count = _validate_optional_count(num_test, "num_test", 1, 4)

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        # This schedule guarantees count diversity for consecutive seed ranges.
        test_count = 1 + seed % 4
        number_of_legal_train_counts = 10 - test_count
        train_count = (
            test_count
            + 1
            + (seed // 4) % number_of_legal_train_counts
        )
        return train_count, test_count

    if train_count is None:
        assert test_count is not None
        train_count = int(rng.integers(test_count + 1, 11))
        return train_count, test_count

    maximum_test = min(4, train_count - 1)
    test_count = int(rng.integers(1, maximum_test + 1))
    return train_count, test_count


def _analyze_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> list[dict]:
    """Infer all guide/motif interpretations from the input alone."""
    if not isinstance(input_grid, np.ndarray):
        return []
    if output_grid is not None and not isinstance(output_grid, np.ndarray):
        return []
    if input_grid.ndim != 2 or (output_grid is not None and output_grid.ndim != 2):
        return []
    if output_grid is not None and input_grid.shape != output_grid.shape:
        return []
    if input_grid.size == 0:
        return []
    if not np.issubdtype(input_grid.dtype, np.integer):
        return []
    if output_grid is not None and not np.issubdtype(output_grid.dtype, np.integer):
        return []

    input_colors, counts = np.unique(input_grid, return_counts=True)
    if input_colors.size != 3:
        return []
    if output_grid is not None and not np.array_equal(input_colors, np.unique(output_grid)):
        return []
    if np.count_nonzero(counts == counts.max()) != 1:
        return []

    background = int(input_colors[int(np.argmax(counts))])
    candidates = []

    for orientation in ("horizontal", "vertical"):
        if orientation == "horizontal":
            canonical_input = input_grid
            canonical_output = output_grid
        else:
            canonical_input = input_grid.T
            canonical_output = None if output_grid is None else output_grid.T

        perpendicular_size, along_size = canonical_input.shape
        line_candidates = []

        for axis in range(perpendicular_size):
            row = canonical_input[axis]
            positions_array = np.flatnonzero(row != background)

            if positions_array.size < 4:
                continue
            if np.unique(row[positions_array]).size != 1:
                continue

            color = int(row[int(positions_array[0])])
            phase = int(positions_array[0])
            positions = tuple(int(x) for x in positions_array.tolist())

            gaps = np.diff(positions_array)
            if not np.all(gaps == gaps[0]) or int(gaps[0]) not in (3, 4, 5):
                continue
            line_candidates.append(
                {"axis": axis, "color": color, "step": int(gaps[0]),
                 "phase": phase, "positions": positions}
            )

        for line_a in line_candidates:
            for line_b in line_candidates:
                if line_a["axis"] == line_b["axis"]:
                    continue
                if line_a["color"] == line_b["color"]:
                    continue
                if line_a["step"] != line_b["step"]:
                    continue
                if line_a["phase"] != line_b["phase"]:
                    continue
                if line_a["positions"] != line_b["positions"]:
                    continue

                axis_a = int(line_a["axis"])
                axis_b = int(line_b["axis"])
                color_a = int(line_a["color"])
                color_b = int(line_b["color"])
                dash_positions = np.asarray(
                    line_a["positions"], dtype=np.int64
                )

                guide_mask = np.zeros(
                    canonical_input.shape, dtype=np.bool_
                )
                guide_mask[axis_a, dash_positions] = True
                guide_mask[axis_b, dash_positions] = True

                extra_mask = (
                    (canonical_input != background) & ~guide_mask
                )
                source_cells = np.argwhere(extra_mask)

                if source_cells.shape[0] < 2:
                    continue
                if not np.all(
                    canonical_input[
                        source_cells[:, 0], source_cells[:, 1]
                    ]
                    == color_b
                ):
                    continue

                source_perpendicular = source_cells[:, 0]
                source_along = source_cells[:, 1]
                if (np.any(source_along < max(0, dash_positions[0] - line_a["step"] + 1))
                        or np.any(source_along > min(along_size - 1, dash_positions[-1] + line_a["step"] - 1))):
                    continue

                if axis_a < axis_b:
                    if not np.all(source_perpendicular < axis_a):
                        continue
                    source_side = "low"
                else:
                    if not np.all(source_perpendicular > axis_a):
                        continue
                    source_side = "high"

                destination_perpendicular = (
                    axis_a + axis_b - source_perpendicular
                )
                if np.any(destination_perpendicular < 0):
                    continue
                if np.any(
                    destination_perpendicular >= perpendicular_size
                ):
                    continue
                if not np.all(
                    canonical_input[
                        destination_perpendicular, source_along
                    ]
                    == background
                ):
                    continue

                expected_output = canonical_input.copy()
                expected_output[
                    destination_perpendicular, source_along
                ] = color_a

                if canonical_output is not None and not np.array_equal(expected_output, canonical_output):
                    continue

                depth = int(
                    np.max(np.abs(source_perpendicular - axis_a))
                )
                along_minimum = int(source_along.min())
                normalized_motif = tuple(
                    sorted(
                        (
                            abs(int(perpendicular) - axis_a),
                            int(along) - along_minimum,
                        )
                        for perpendicular, along in source_cells
                    )
                )

                candidates.append(
                    {
                        "orientation": orientation,
                        "step": int(line_a["step"]),
                        "phase": int(line_a["phase"]),
                        "background": background,
                        "color_a": color_a,
                        "color_b": color_b,
                        "role_triple": (
                            background,
                            color_a,
                            color_b,
                        ),
                        "axis_a": axis_a,
                        "axis_b": axis_b,
                        "source_side": source_side,
                        "depth": depth,
                        "motif_count": int(source_cells.shape[0]),
                        "motif_signature": normalized_motif,
                        "span": int(source_along.max() - source_along.min() + 1),
                        "guide_start": int(dash_positions[0]),
                        "guide_end": int(dash_positions[-1]),
                        "margins": (min(axis_a, axis_b), perpendicular_size - 1 - max(axis_a, axis_b)),
                        "predicted_output": expected_output.copy() if orientation == "horizontal" else expected_output.T.copy(),
                        "dash_count": int(dash_positions.size),
                        "grid_shape": tuple(input_grid.shape),
                        "perpendicular_size": perpendicular_size,
                        "along_size": along_size,
                    }
                )

    return candidates


def _sample_motif_mask(
    rng: np.random.Generator,
    depth: int,
    span: int,
) -> np.ndarray:
    """Sample connected, branched, or disconnected shapes without a fixed spine."""
    mask = np.zeros((depth, span), dtype=np.bool_)
    mode = int(rng.integers(0, 4))
    if mode == 0:
        # One connected path with optional adjacent branches.
        row = column = 0
        mask[row, column] = True
        while row < depth - 1 or column < span - 1:
            if row == depth - 1:
                column += 1
            elif column == span - 1:
                row += 1
            elif rng.random() < (depth - 1 - row) / (depth + span - 2 - row - column):
                row += 1
            else:
                column += 1
            mask[row, column] = True
        for row, column in np.argwhere(mask.copy()):
            if rng.random() < .3:
                neighbor_row = int(np.clip(row + rng.choice((-1, 1)), 0, depth - 1))
                mask[neighbor_row, column] = True
            if rng.random() < .3:
                neighbor_column = int(np.clip(column + rng.choice((-1, 1)), 0, span - 1))
                mask[row, neighbor_column] = True
    elif mode == 1:
        # Several separated lobes.
        lobes = int(rng.integers(2, min(6, max(3, span // 3 + 1))))
        for index in range(lobes):
            center = int(round((index + 0.5) * (span - 1) / lobes))
            width = int(rng.integers(1, max(2, span // (2 * lobes) + 1)))
            top = int(rng.integers(0, depth))
            bottom = int(rng.integers(top + 1, depth + 1))
            mask[top:bottom, max(0, center-width):min(span, center+width+1)] = True
    elif mode == 2:
        # Sparse cells can make several disconnected components.
        mask = rng.random((depth, span)) < float(rng.uniform(.12, .40))
    else:
        # Independent horizontal branches.
        for row in range(depth):
            for _ in range(int(rng.integers(1, 4))):
                start = int(rng.integers(0, span))
                width = int(rng.integers(1, max(2, span // 3 + 1)))
                mask[row, start:min(span, start+width)] = True
    # Only a few cells establish the requested bounds; no row is filled.
    if mode != 0:
        mask[int(rng.integers(depth)), 0] = True
        mask[int(rng.integers(depth)), -1] = True
        mask[0, int(rng.integers(span))] = True
        mask[-1, int(rng.integers(span))] = True
    return mask


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    orientation: str,
    role_colors: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown generation regime")
    if orientation not in ("horizontal", "vertical"):
        raise ValueError("unknown orientation")

    background, color_a, color_b = role_colors
    if len({background, color_a, color_b}) != 3:
        raise ValueError("role colors must be distinct")

    for _ in range(300):
        probabilities = (.53, .36, .11) if regime == "source_train" else (.37, .43, .20)
        step = int(rng.choice((3, 4, 5), p=probabilities))
        depth = int(rng.integers(1, 8 if regime == "source_train" else 9))
        separation = int(rng.integers(3, 12))
        extra_budget = 30 - (2 * depth + separation + 1)
        low_extra = int(rng.integers(0, min(12, extra_budget) + 1))
        high_extra = int(rng.integers(0, min(12, extra_budget - low_extra) + 1))
        if rng.random() < .5:
            low_extra, high_extra = high_extra, low_extra
        low_margin = depth + low_extra
        high_margin = depth + high_extra
        perpendicular_size = (
            low_margin + separation + 1 + high_margin
        )

        if perpendicular_size > 30:
            continue

        low_axis = low_margin
        high_axis = low_axis + separation

        source_on_low_side = bool(rng.integers(0, 2))
        if source_on_low_side:
            axis_a = low_axis
            axis_b = high_axis
        else:
            axis_a = high_axis
            axis_b = low_axis

        if rng.random() < .35:
            along_size = int(rng.integers(13, 31))
            dash_positions = np.arange(0, along_size, step, dtype=np.int64)
        else:
            along_size = int(rng.integers(15, 31))
            guide_start = int(rng.integers(0, 5))
            guide_stop = along_size - int(rng.integers(0, 5))
            dash_positions = np.arange(guide_start, guide_stop, step, dtype=np.int64)
        if dash_positions.size < 4:
            continue
        motif_end = min(along_size - 1, int(dash_positions[-1]) + step - 1)
        available = int(motif_end - dash_positions[0] + 1)
        span_mode = rng.random()
        if span_mode < .28:
            span = available
        elif span_mode < .60:
            span = int(rng.integers(max(3, int(.65 * available)), available + 1))
        else:
            span = int(rng.integers(3, max(4, int(.65 * available) + 1)))
        along_start = int(rng.integers(int(dash_positions[0]), motif_end - span + 2))
        motif_mask = _sample_motif_mask(rng, depth, span)

        canonical_input = np.full(
            (perpendicular_size, along_size),
            background,
            dtype=np.int8,
        )
        canonical_input[axis_a, dash_positions] = color_a
        canonical_input[axis_b, dash_positions] = color_b

        for distance_index, offset in np.argwhere(motif_mask):
            distance = int(distance_index) + 1
            along = along_start + int(offset)

            if axis_a < axis_b:
                perpendicular = axis_a - distance
            else:
                perpendicular = axis_a + distance

            canonical_input[perpendicular, along] = color_b

        canonical_output = canonical_input.copy()
        source_cells = np.argwhere(
            (canonical_input == color_b)
            & (
                np.indices(canonical_input.shape)[0] != axis_b
            )
        )

        # The predicate above contains exactly the motif because color_b
        # appears elsewhere only on axis_b.
        for perpendicular, along in source_cells:
            destination = (
                axis_a + axis_b - int(perpendicular)
            )
            canonical_output[destination, int(along)] = color_a

        if orientation == "horizontal":
            input_grid = canonical_input.copy()
            output_grid = canonical_output.copy()
        else:
            input_grid = canonical_input.T.copy()
            output_grid = canonical_output.T.copy()

        pair = {"input": input_grid, "output": output_grid}
        if _source_hash(pair) in _SOURCE_HASHES:
            continue
        analyses = _analyze_pair(input_grid, None)
        if len(analyses) != 1:
            continue

        analysis = analyses[0]
        if analysis["orientation"] != orientation:
            continue
        if analysis["step"] != step:
            continue
        if analysis["depth"] != depth:
            continue
        if analysis["role_triple"] != role_colors:
            continue
        if analysis["axis_a"] != axis_a:
            continue
        if analysis["axis_b"] != axis_b:
            continue
        if not np.array_equal(analysis["predicted_output"], output_grid):
            continue

        return input_grid, output_grid

    raise ValueError(
        "could not construct an unambiguous pair after repeated attempts"
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
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")

    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(
        rng, seed, num_train, num_test
    )

    # There is no hidden palette convention in the source rule. Most puzzles
    # vary literal colors by pair; a few retain a stable palette.
    stable_palette = bool(rng.random() < .18)
    palette = tuple(int(x) for x in rng.choice(10, 3, replace=False))
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for _ in range(40):
                colors = palette if stable_palette else tuple(int(x) for x in rng.choice(10, 3, replace=False))
                role_colors = tuple(colors[int(i)] for i in rng.permutation(3))
                orientation = "horizontal" if bool(rng.integers(0, 2)) else "vertical"
                child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(child_seed)
                input_grid, output_grid = _generate_pair(
                    pair_rng, "source_train" if split == "train" else "target_test",
                    orientation, role_colors,
                )
                pair = {"input": input_grid, "output": output_grid}
                if split == "train" and index == 0:
                    example = _analyze_pair(input_grid, None)[0]
                    if example["depth"] < 2 or example["motif_count"] < 6:
                        continue
                input_key = _grid_key(input_grid)
                if input_key in used_inputs:
                    continue
                used_inputs.add(input_key)
                puzzle[split].append(pair)
                break
            else:
                raise ValueError("could not generate a unique pair")

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    train_count = len(puzzle["train"])
    test_count = len(puzzle["test"])
    assert 2 <= train_count <= 10
    assert 1 <= test_count <= 4
    assert train_count > test_count

    input_keys = set()
    pair_keys = set()
    analyses_by_split = {"train": [], "test": []}

    for split in ("train", "test"):
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
                assert grid.size > 0
                assert int(grid.min()) >= 0
                assert int(grid.max()) <= 9

            assert input_grid.shape == output_grid.shape
            assert not np.array_equal(input_grid, output_grid)

            input_key = _grid_key(input_grid)
            pair_key = _pair_key(pair)
            assert input_key not in input_keys
            assert pair_key not in pair_keys
            input_keys.add(input_key)
            pair_keys.add(pair_key)

            assert _source_hash(pair) not in _SOURCE_HASHES

            input_palette = frozenset(
                int(x) for x in np.unique(input_grid)
            )
            output_palette = frozenset(
                int(x) for x in np.unique(output_grid)
            )
            assert len(input_palette) == 3
            assert output_palette == input_palette

            analyses = _analyze_pair(input_grid, None)
            assert len(analyses) == 1
            analysis = analyses[0]
            analyses_by_split[split].append(analysis)
            assert np.array_equal(analysis["predicted_output"], output_grid)

            changed = input_grid != output_grid
            assert int(np.count_nonzero(changed)) == analysis[
                "motif_count"
            ]
            assert np.all(
                input_grid[changed] == analysis["background"]
            )
            assert np.all(
                output_grid[changed] == analysis["color_a"]
            )
            assert analysis["color_a"] != analysis["color_b"]
            assert analysis["dash_count"] >= 4
            assert analysis["motif_count"] >= 2
            assert analysis["step"] in (3, 4, 5)
    # Each input exposes both guide colors, their alignment, and the sole
    # exterior motif. No hidden palette or direction convention is needed.
    assert any(a["depth"] >= 2 and a["motif_count"] >= 6
               for a in analyses_by_split["train"])


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first.keys()) != {"train", "test"}:
        return False
    if set(second.keys()) != {"train", "test"}:
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for pair_a, pair_b in zip(first[split], second[split]):
            for grid_name in ("input", "output"):
                grid_a = pair_a[grid_name]
                grid_b = pair_b[grid_name]
                if grid_a.dtype != grid_b.dtype:
                    return False
                if not np.array_equal(grid_a, grid_b):
                    return False

    return True


def self_test() -> None:
    automatic = generate_puzzle(12345)
    assert _puzzles_equal(automatic, generate_puzzle(12345))
    explicit = generate_puzzle(54321, 6, 3)
    assert _puzzles_equal(explicit, generate_puzzle(54321, 6, 3))
    for seed, train_count, test_count in (
        (7001, 2, 1), (7002, 10, 1), (7003, 10, 4),
        (8001, None, 1), (8002, None, 4),
        (8003, 2, None), (8004, 10, None),
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for train_count, test_count in (
        (1, None), (11, None), (None, 0), (None, 5),
        (2, 2), (3, 3), (3, 4), (2.5, 1), (2, 1.5),
    ):
        try:
            generate_puzzle(9000, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")

    train_counts, test_counts = set(), set()
    shapes, motifs, layouts, palettes, puzzle_keys = set(), set(), set(), set(), set()
    features = []
    within_palette_changes = 0
    within_depth_changes = 0
    within_step_changes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(_puzzle_key(puzzle))
        puzzle_features = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                analysis = _analyze_pair(pair["input"], None)
                assert len(analysis) == 1
                feature = analysis[0]
                assert np.array_equal(feature["predicted_output"], pair["output"])
                assert _source_hash(pair) not in _SOURCE_HASHES
                puzzle_features.append(feature)
                features.append((split, feature))
                shapes.add(pair["input"].shape)
                palettes.add(tuple(int(x) for x in np.unique(pair["input"])))
                motifs.add(feature["motif_signature"])
                layouts.add((feature["orientation"], feature["step"],
                             pair["input"].shape, feature["axis_a"],
                             feature["axis_b"], feature["guide_start"],
                             feature["guide_end"]))
        within_palette_changes += len({f["role_triple"] for f in puzzle_features}) > 1
        within_depth_changes += len({f["depth"] for f in puzzle_features}) > 1
        within_step_changes += len({f["step"] for f in puzzle_features}) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_keys) == 42 and len(shapes) >= 20
    assert len(motifs) >= 80 and len(layouts) >= 80 and len(palettes) >= 20
    assert within_palette_changes >= 25 and within_depth_changes >= 25
    assert within_step_changes >= 20
    assert {f["orientation"] for _, f in features} == {"horizontal", "vertical"}
    assert {f["step"] for _, f in features} == {3, 4, 5}
    assert any(f["span"] == f["along_size"] for _, f in features)
    assert any(f["guide_start"] > 0 for _, f in features)
    assert any(f["margins"][0] != f["margins"][1] for _, f in features)
    assert any(f["depth"] <= 2 for split, f in features if split == "test")
    assert any(f["depth"] >= 5 for split, f in features if split == "test")
    assert any(f["grid_shape"] == (22, 22) for _, f in features)
    assert {abs(f["axis_a"]-f["axis_b"]) for _, f in features} == set(range(3, 12))
    assert sum(f["guide_start"] > 0 or f["guide_end"] < f["along_size"]-3
               for _, f in features) >= 100
    assert sum(abs(f["margins"][0]-f["margins"][1]) >= 4
               for _, f in features) >= 100
    assert any(f["span"] < .5*f["along_size"] for _, f in features)
    assert any(f["span"] >= .75*f["along_size"] and f["guide_start"] > 0
               and abs(f["margins"][0]-f["margins"][1]) >= 4
               for _, f in features)
