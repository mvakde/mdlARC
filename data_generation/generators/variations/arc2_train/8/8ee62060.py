"""Translate repeated diagonal tiles to horizontally opposite lattice slots."""

import numpy as np


_SOURCE_SIGNATURES = {
    (0, 6, "anti", (2, 2, 3, 2)),
    (0, 6, "main", (8, 0, 2, 2)),
    (0, 5, "main", (2, 1, 1, 0)),
    (0, 7, "anti", (1, 8, 8, 1)),
}


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _grid_key(grid):
    array = np.asarray(grid, dtype=np.int64)
    return array.shape, array.tobytes()


def _checkerboard(motif, background):
    return bool(
        motif.shape == (2, 2)
        and np.all(motif != background)
        and motif[0, 0] == motif[1, 1]
        and motif[0, 1] == motif[1, 0]
        and motif[0, 0] != motif[0, 1]
    )


def _motif_valid(motif, background):
    foreground = motif != background
    return bool(
        np.all(np.any(foreground, axis=0))
        and np.all(np.any(foreground, axis=1))
        and 1 <= len(set(int(v) for v in motif[foreground])) <= 3
    )


def _sample_motif(rng, shape, background, force_asymmetric=False):
    height, width = shape
    for _ in range(1500):
        symmetric = not force_asymmetric and bool(rng.random() < 0.22)
        full = bool(rng.random() < 0.53)
        if symmetric and width == 2:
            full = True
        if force_asymmetric and full:
            color_count = int(rng.choice([2, 3]))
        else:
            color_count = int(rng.choice([1, 2, 2, 2, 3]))
        if symmetric and color_count > height * ((width + 1) // 2):
            continue
        palette = rng.choice(
            [color for color in range(10) if color != background],
            size=color_count,
            replace=False,
        )
        motif = np.full(shape, background, dtype=np.int8)
        orbit_width = (width + 1) // 2 if symmetric else width
        for row in range(height):
            for col in range(orbit_width):
                if full or rng.random() < 0.72:
                    color = int(rng.choice(palette))
                    motif[row, col] = color
                    if symmetric:
                        motif[row, width - 1 - col] = color
        if not _motif_valid(motif, background):
            continue
        if bool(np.all(motif != background)) != full:
            continue
        if len(set(int(v) for v in motif.flat if v != background)) != color_count:
            continue
        if bool(np.array_equal(motif, np.fliplr(motif))) != symmetric:
            continue
        return motif
    raise RuntimeError("could not sample motif")


def _sample_geometry(rng, n, shape):
    height, width = shape
    configs, weights = [], []
    gap_weight = (5.0, 3.0, 2.0)
    margin_weight = (5.0, 3.0, 2.0)
    for row_gap in range(3):
        row_pitch = height + row_gap
        if (n - 1) * row_pitch + height > 30:
            continue
        for col_gap in range(3):
            col_pitch = width + col_gap
            for margin in range(3):
                if (n - 1) * col_pitch + width + 2 * margin <= 30:
                    configs.append((row_pitch, col_pitch, margin))
                    weights.append(
                        gap_weight[row_gap] * gap_weight[col_gap] * margin_weight[margin]
                    )
    probabilities = np.asarray(weights, dtype=float)
    probabilities /= probabilities.sum()
    return configs[int(rng.choice(len(configs), p=probabilities))]


def _construct(n, motif, background, orientation, row_pitch, col_pitch, margin):
    height, width = motif.shape
    rows = (n - 1) * row_pitch + height
    cols = (n - 1) * col_pitch + width + 2 * margin
    grid = np.full((rows, cols), background, dtype=np.int8)
    for row_slot in range(n):
        col_slot = row_slot if orientation == "main" else n - 1 - row_slot
        row = row_slot * row_pitch
        col = margin + col_slot * col_pitch
        grid[row:row + height, col:col + width] = motif
    return grid


def _move_tiles(info):
    output = np.full(info["shape"], info["background"], dtype=np.int8)
    motif = info["motif"]
    height, width = motif.shape
    n = info["n"]
    for row_slot in range(n):
        input_col_slot = row_slot if info["orientation"] == "main" else n - 1 - row_slot
        target_col_slot = n - 1 - input_col_slot
        row = row_slot * info["row_pitch"]
        col = info["margin"] + target_col_slot * info["col_pitch"]
        output[row:row + height, col:col + width] = motif
    return output


def _sample_size(rng, regime, train_sizes):
    if regime == "train":
        return int(rng.choice([3, 4, 5, 5, 6, 6, 7]))
    draw = rng.random()
    if train_sizes and draw < 0.35:
        return int(rng.choice(train_sizes))
    if train_sizes and draw < 0.65 and min(train_sizes) > 3:
        return int(rng.integers(3, min(train_sizes)))
    if train_sizes and draw < 0.85 and max(train_sizes) < 9:
        return int(rng.integers(max(train_sizes) + 1, 10))
    return int(rng.choice([3, 4, 5, 6, 7, 7, 8, 9]))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)

    # The preferred background affects sampling, but each pair's roles are
    # visible locally. There is no hidden color map or direction convention.
    preferred_background = int(rng.integers(0, 10))
    change_background = bool(rng.integers(0, 2))
    first_orientation = "main" if rng.integers(0, 2) else "anti"
    varied_slopes = bool(rng.integers(0, 2))
    pair_seeds = rng.integers(
        0, np.iinfo(np.int64).max, size=train_count + test_count, dtype=np.int64
    )
    puzzle = {"train": [], "test": []}
    seen = set()
    train_info = []

    for index in range(train_count + test_count):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        regime = "train" if index < train_count else "test"
        destination = puzzle[regime]
        for attempt in range(200):
            related = regime == "test" and bool(pair_rng.random() < 0.28)
            if related:
                model = train_info[int(pair_rng.integers(0, len(train_info)))]
                background = model["background"]
                motif = model["motif"].copy()
                shape = motif.shape
            else:
                if index == 0 or not change_background:
                    background = preferred_background
                elif index == 1:
                    background = int(pair_rng.choice(
                        [c for c in range(10) if c != preferred_background]
                    ))
                elif pair_rng.random() < 0.53:
                    background = preferred_background
                else:
                    background = int(pair_rng.choice(
                        [c for c in range(10) if c != preferred_background]
                    ))
                shapes = [(2, 2), (2, 3), (3, 2), (3, 3)]
                if index == 1:
                    shapes.remove(train_info[0]["motif"].shape)
                shape = shapes[int(pair_rng.integers(0, len(shapes)))]
                motif = _sample_motif(pair_rng, shape, background, index == 0)
                if regime == "test" and pair_rng.random() < 0.15:
                    shape = (2, 2)
                    colors = pair_rng.choice(
                        [c for c in range(10) if c != background],
                        size=2, replace=False,
                    )
                    a, b = (int(colors[0]), int(colors[1]))
                    motif = np.array([[a, b], [b, a]], dtype=np.int8)
                if index == 1:
                    first_colors = set(int(v) for v in train_info[0]["motif"].flat)
                    first_colors.discard(train_info[0]["background"])
                    colors = set(int(v) for v in motif.flat)
                    colors.discard(background)
                    if colors == first_colors:
                        continue

            train_sizes = [entry["n"] for entry in train_info]
            n = _sample_size(pair_rng, regime, train_sizes)
            if index == 0:
                orientation = first_orientation
            elif index == 1:
                orientation = (
                    ("anti" if first_orientation == "main" else "main")
                    if varied_slopes else first_orientation
                )
            elif regime == "train" and not varied_slopes:
                orientation = first_orientation
            else:
                orientation = "main" if pair_rng.integers(0, 2) else "anti"
            row_pitch, col_pitch, margin = _sample_geometry(pair_rng, n, shape)
            if (
                background, n, orientation, tuple(int(v) for v in motif.flat)
            ) in _SOURCE_SIGNATURES and shape == (2, 2) and (
                row_pitch, col_pitch, margin
            ) == (2, 2, 0):
                continue
            input_grid = _construct(
                n, motif, background, orientation, row_pitch, col_pitch, margin
            )
            key = _grid_key(input_grid)
            if key in seen:
                continue
            info = {
                "n": n, "motif": motif, "background": background,
                "orientation": orientation, "row_pitch": row_pitch,
                "col_pitch": col_pitch, "margin": margin,
                "shape": input_grid.shape,
            }
            output_grid = _move_tiles(info)
            seen.add(key)
            destination.append({"input": input_grid, "output": output_grid})
            if regime == "train":
                train_info.append(info)
            break
        else:
            raise RuntimeError("could not construct distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _infer_pair(input_grid, output_grid):
    values, counts = np.unique(input_grid, return_counts=True)
    assert len(values) >= 2
    assert np.count_nonzero(counts == counts.max()) == 1
    background = int(values[np.argmax(counts)])
    assert int(counts.max()) > input_grid.size // 2
    foreground_positions = np.argwhere(input_grid != background)
    assert len(foreground_positions) > 0
    assert int(foreground_positions[:, 0].min()) == 0
    left = int(foreground_positions[:, 1].min())
    right = input_grid.shape[1] - 1 - int(foreground_positions[:, 1].max())
    assert left == right and 0 <= left <= 2
    margin = left
    candidates = []
    for n in range(3, 10):
        for height in (2, 3):
            for width in (2, 3):
                for row_gap in range(3):
                    row_pitch = height + row_gap
                    if (n - 1) * row_pitch + height != input_grid.shape[0]:
                        continue
                    for col_gap in range(3):
                        col_pitch = width + col_gap
                        if (
                            (n - 1) * col_pitch + width + 2 * margin
                            != input_grid.shape[1]
                        ):
                            continue
                        for orientation in ("main", "anti"):
                            first_col_slot = 0 if orientation == "main" else n - 1
                            col = margin + first_col_slot * col_pitch
                            motif = input_grid[:height, col:col + width].copy()
                            if not _motif_valid(motif, background):
                                continue
                            reconstructed = _construct(
                                n, motif, background, orientation,
                                row_pitch, col_pitch, margin,
                            )
                            if not np.array_equal(reconstructed, input_grid):
                                continue
                            info = {
                                "n": n, "motif": motif, "background": background,
                                "orientation": orientation, "row_pitch": row_pitch,
                                "col_pitch": col_pitch, "margin": margin,
                                "shape": input_grid.shape,
                            }
                            candidates.append(info)
    assert candidates
    predictions = [_move_tiles(info) for info in candidates]
    assert all(np.array_equal(predictions[0], grid) for grid in predictions[1:])
    assert np.array_equal(output_grid, predictions[0])
    assert not np.array_equal(input_grid, output_grid)
    info = candidates[0]
    if (
        info["background"], info["n"], info["orientation"],
        tuple(int(v) for v in info["motif"].flat),
    ) in _SOURCE_SIGNATURES:
        assert not (
            info["motif"].shape == (2, 2)
            and (info["row_pitch"], info["col_pitch"], info["margin"])
            == (2, 2, 0)
        )
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list)
        assert isinstance(puzzle["test"], list)
        assert 2 <= len(puzzle["train"]) <= 10
        assert 1 <= len(puzzle["test"]) <= 4
        assert len(puzzle["train"]) > len(puzzle["test"])
        seen_inputs, seen_pairs = set(), set()
        info_by_regime = {"train": [], "test": []}
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                for grid in pair.values():
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert all(1 <= length <= 30 for length in grid.shape)
                    assert np.all((0 <= grid) & (grid <= 9))
                input_grid, output_grid = pair["input"], pair["output"]
                assert input_grid.shape == output_grid.shape
                input_key = _grid_key(input_grid)
                pair_key = (input_key, _grid_key(output_grid))
                assert input_key not in seen_inputs and pair_key not in seen_pairs
                seen_inputs.add(input_key)
                seen_pairs.add(pair_key)
                info_by_regime[regime].append(_infer_pair(input_grid, output_grid))
        train = info_by_regime["train"]
        assert any(not np.array_equal(i["motif"], np.fliplr(i["motif"])) for i in train)
        train_palettes = {
            frozenset(int(v) for v in i["motif"].flat if v != i["background"])
            for i in train
        }
        assert len(train_palettes) >= 2
        assert len({i["motif"].shape for i in train}) >= 2
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("invalid generated puzzle") from exc


def _puzzles_equal(first, second):
    return all(
        np.array_equal(a[k], b[k])
        for regime in ("train", "test")
        for a, b in zip(first[regime], second[regime])
        for k in ("input", "output")
    ) and all(len(first[r]) == len(second[r]) for r in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, puzzle_keys = set(), set(), set()
    backgrounds, shapes, train_shapes, test_shapes, sizes, gaps, margins, slopes, test_slopes, colors = (
        set(), set(), set(), set(), set(), set(), set(), set(), set(), set()
    )
    changed_backgrounds = repeated_train_slopes = mixed_train_slopes = 0
    repeated_train_sizes = opposite_slope_same_size = same_slope_different_size = 0
    familiar_test = smaller_test = larger_test = 0
    partial_test = full_nonchecker_test = checker_test = 0
    symmetric_train = symmetric_test = train_checker = 0
    related_test = same_size_related_test = contiguous = combined = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(tuple(
            _grid_key(pair["input"])
            for regime in ("train", "test") for pair in puzzle[regime]
        ))
        train = [_infer_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test = [_infer_pair(p["input"], p["output"]) for p in puzzle["test"]]
        all_info = train + test
        changed_backgrounds += len({i["background"] for i in all_info}) > 1
        repeated_train_slopes += len({i["orientation"] for i in train}) == 1
        mixed_train_slopes += len({i["orientation"] for i in train}) == 2
        repeated_train_sizes += len({i["n"] for i in train}) < len(train)
        opposite_slope_same_size += any(
            a["n"] == b["n"] and a["orientation"] != b["orientation"]
            for j, a in enumerate(train) for b in train[j + 1:]
        )
        same_slope_different_size += any(
            a["n"] != b["n"] and a["orientation"] == b["orientation"]
            for j, a in enumerate(train) for b in train[j + 1:]
        )
        train_shapes.update(i["motif"].shape for i in train)
        test_shapes.update(i["motif"].shape for i in test)
        test_slopes.update(i["orientation"] for i in test)
        symmetric_train += sum(
            np.array_equal(i["motif"], np.fliplr(i["motif"])) for i in train
        )
        symmetric_test += sum(
            np.array_equal(i["motif"], np.fliplr(i["motif"])) for i in test
        )
        train_checker += sum(_checkerboard(i["motif"], i["background"]) for i in train)
        for info in all_info:
            backgrounds.add(info["background"])
            shapes.add(info["motif"].shape)
            sizes.add(info["n"])
            gaps.add((info["row_pitch"] - info["motif"].shape[0],
                      info["col_pitch"] - info["motif"].shape[1]))
            margins.add(info["margin"])
            slopes.add(info["orientation"])
            colors.add(len(set(
                int(v) for v in info["motif"].flat if v != info["background"]
            )))
            contiguous += (
                info["row_pitch"] == info["motif"].shape[0]
                and info["col_pitch"] == info["motif"].shape[1]
                and info["margin"] == 0
            )
            combined += (
                info["background"] != 0 and info["motif"].shape != (2, 2)
                and (info["row_pitch"] > info["motif"].shape[0]
                     or info["col_pitch"] > info["motif"].shape[1])
            )
        for info in test:
            familiar_test += info["n"] in {i["n"] for i in train}
            smaller_test += info["n"] < min(i["n"] for i in train)
            larger_test += info["n"] > max(i["n"] for i in train)
            partial_test += np.any(info["motif"] == info["background"])
            full_nonchecker_test += (
                np.all(info["motif"] != info["background"])
                and not _checkerboard(info["motif"], info["background"])
            )
            checker_test += _checkerboard(info["motif"], info["background"])
            related = any(
                info["background"] == example["background"]
                and np.array_equal(info["motif"], example["motif"])
                for example in train
            )
            related_test += related
            same_size_related_test += related and info["n"] in {i["n"] for i in train}
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_keys) == 42
    assert backgrounds == set(range(10))
    assert shapes == {(2, 2), (2, 3), (3, 2), (3, 3)}
    assert train_shapes == shapes and test_shapes == shapes
    assert sizes == set(range(3, 10))
    assert len(gaps) >= 6 and margins == {0, 1, 2}
    assert slopes == {"main", "anti"} and colors == {1, 2, 3}
    assert test_slopes == slopes
    assert changed_backgrounds >= 10
    assert repeated_train_slopes >= 3 and mixed_train_slopes >= 3
    assert repeated_train_sizes >= 10
    assert opposite_slope_same_size >= 3 and same_slope_different_size >= 10
    assert familiar_test >= 5 and smaller_test >= 3 and larger_test >= 3
    assert partial_test >= 5 and full_nonchecker_test >= 5 and checker_test >= 1
    assert symmetric_train >= 3 and symmetric_test >= 3 and train_checker >= 1
    assert related_test >= 5 and same_size_related_test >= 3
    assert contiguous >= 10 and combined >= 10
    assert _puzzles_equal(generate_puzzle(12345), generate_puzzle(12345))
    assert _puzzles_equal(generate_puzzle(54321, 8, 3), generate_puzzle(54321, 8, 3))
    assert not _puzzles_equal(generate_puzzle(100), generate_puzzle(101))
    for index, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(2000 + index, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        validate_generated_puzzle(puzzle)
    for index, (train_count, test_count) in enumerate(
        ((None, 1), (None, 4), (2, None), (10, None))
    ):
        puzzle = generate_puzzle(3000 + index, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for kwargs in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4}, {"num_train": 3.0, "num_test": 1},
    ):
        try:
            generate_puzzle(4000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
