import numpy as np


_SOURCE_PAIR_SIGNATURES = {
    (
        (
            "0000000000", "0000000000", "5555550000", "0000000500",
            "0000000500", "0000000500", "0000000500", "0055500500",
            "0000000000", "0000000000",
        ),
        (
            "0000000000", "0000000000", "1111110000", "0000000400",
            "0000000400", "0000000400", "0000000400", "0022200400",
            "0000000000", "0000000000",
        ),
    ),
    (
        (
            "0000000000", "0000500000", "0000500000", "0500500000",
            "0500500000", "0500500500", "0500500500", "0000000000",
            "0000000000", "0000000000",
        ),
        (
            "0000000000", "0000100000", "0000100000", "0400100000",
            "0400100000", "0400100200", "0400100200", "0000000000",
            "0000000000", "0000000000",
        ),
    ),
    (
        (
            "0000000000", "0000000500", "0555500500", "0000000500",
            "0000000500", "0555550500", "0000000500", "0000000500",
            "0000000000", "0000000000",
        ),
        (
            "0000000000", "0000000100", "0222200100", "0000000100",
            "0000000100", "0444440100", "0000000100", "0000000100",
            "0000000000", "0000000000",
        ),
    ),
    (
        (
            "0000000000", "0500000000", "0500500000", "0500500000",
            "0500500500", "0500500500", "0500500500", "0000000000",
            "0000000000", "0000000000",
        ),
        (
            "0000000000", "0100000000", "0100400000", "0100400000",
            "0100400200", "0100400200", "0100400200", "0000000000",
            "0000000000", "0000000000",
        ),
    ),
    (
        (
            "0000000000", "0005000000", "0005000000", "0005000000",
            "0000000000", "0000555555", "0000000000", "0000000000",
            "5555500000", "0000000000",
        ),
        (
            "0000000000", "0002000000", "0002000000", "0002000000",
            "0000000000", "0000111111", "0000000000", "0000000000",
            "4444400000", "0000000000",
        ),
    ),
}


def _checked_count(name, value):
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)

    if name == "num_train" and not 2 <= value <= 10:
        raise ValueError("num_train must be between 2 and 10")

    if name == "num_test" and not 1 <= value <= 4:
        raise ValueError("num_test must be between 1 and 4")

    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _checked_count("num_train", num_train)
    num_test = _checked_count("num_test", num_test)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        if rng.random() < 0.42:
            num_test = 1
            num_train = int(rng.integers(2, 5))
        else:
            num_test = int(rng.integers(1, 5))
            num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(
            rng.integers(max(2, num_test + 1), 11)
        )
    else:
        maximum_test = min(4, num_train - 1)
        if maximum_test < 1:
            raise ValueError(
                "no legal num_test exists for this num_train"
            )
        num_test = int(rng.integers(1, maximum_test + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _components(mask):
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result = []

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue

            stack = [(row, col)]
            seen[row, col] = True
            component = []

            while stack:
                current_row, current_col = stack.pop()
                component.append((current_row, current_col))

                for dr, dc in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ):
                    next_row = current_row + dr
                    next_col = current_col + dc

                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and mask[next_row, next_col]
                        and not seen[next_row, next_col]
                    ):
                        seen[next_row, next_col] = True
                        stack.append((next_row, next_col))

            result.append(component)

    return result


def _solve_from_input(
    input_grid,
    background,
    source,
    rank_colors,
):
    components = _components(input_grid == source)

    assert len(components) == 3

    components = sorted(
        components,
        key=len,
        reverse=True,
    )

    assert len(
        {len(component) for component in components}
    ) == 3

    output = np.full(
        input_grid.shape,
        background,
        dtype=np.int8,
    )

    for rank, component in enumerate(components):
        color = int(rank_colors[rank])
        for row, col in component:
            output[row, col] = color

    return output


def sample_puzzle_parameters(rng):
    colors = [int(x) for x in rng.permutation(10)]
    return {
        "rank_colors": tuple(colors[:3]),
        "available": tuple(colors[3:]),
        "vary_palette": bool(rng.integers(0, 2)),
    }


_ORIENTATIONS = tuple(
    (a, b, c)
    for a in "hv" for b in "hv" for c in "hv"
)


def _orientation_plan(rng, train_count, test_count):
    # A source-like holdout, a holdout in another direction, and familiar
    # test assignments are all supported by the same length rule.
    regime = int(rng.integers(0, 3))
    if regime == 0:
        held_out = ("h", "h", "v")
        first = ("v", "v", "v")
        second = ("v", "h", "h") if rng.integers(0, 2) else ("h", "v", "h")
    else:
        held_out = _ORIENTATIONS[int(rng.integers(0, 8))]
        choices = [p for p in _ORIENTATIONS if p != held_out]
        first = choices[int(rng.integers(0, len(choices)))]
        different = [p for p in choices if p != first]
        second = different[int(rng.integers(0, len(different)))]
    training = [first, second]
    choices = [p for p in _ORIENTATIONS if regime == 2 or p != held_out]
    while len(training) < train_count:
        training.append(choices[int(rng.integers(0, len(choices)))])
    training = [training[int(i)] for i in rng.permutation(train_count)]
    if regime == 2:
        testing = [training[int(rng.integers(0, train_count))] for _ in range(test_count)]
    else:
        testing = [held_out]
        while len(testing) < test_count:
            testing.append(_ORIENTATIONS[int(rng.integers(0, 8))])
    return training, testing


def _sample_lengths(rng, scale=None):
    if scale is None:
        scale = int(rng.choice(3, p=(0.42, 0.38, 0.20)))
    upper = (8, 11, 18)[scale]
    return tuple(sorted((int(x) for x in rng.choice(np.arange(2, upper + 1), 3, replace=False)), reverse=True))


def _length_plan(rng, train_count, test_count):
    first = _sample_lengths(rng)
    mode = int(rng.integers(0, 3))
    if mode == 0:
        second = first
    elif mode == 1:
        # An overlapping arithmetic pair puts a numeric length in two ranks.
        step = int(rng.integers(1, 4))
        low = int(rng.integers(2, 8))
        first = (low + 2 * step, low + step, low)
        second = (low + 3 * step, low + 2 * step, low + step)
    else:
        second = _sample_lengths(rng)
    training = [first, second]
    while len(training) < train_count:
        if rng.random() < 0.18:
            training.append(training[int(rng.integers(0, len(training)))])
        else:
            training.append(_sample_lengths(rng))
    if rng.random() < 0.5:
        order = rng.permutation(train_count)
        training = [training[int(i)] for i in order]
    testing = []
    for _ in range(test_count):
        if rng.random() < 0.38:
            testing.append(training[int(rng.integers(0, train_count))])
        else:
            candidate = _sample_lengths(rng)
            for _ in range(20):
                if candidate not in training:
                    break
                candidate = _sample_lengths(rng)
            testing.append(candidate)
    return training, testing


def _cells_for_start(
    orientation,
    length,
    row,
    col,
):
    if orientation == "h":
        return [
            (row, col + offset)
            for offset in range(length)
        ]

    return [
        (row + offset, col)
        for offset in range(length)
    ]


def _place_segments(
    rng,
    height,
    width,
    specifications,
):
    for _ in range(35):
        placed = [None] * len(specifications)
        occupied = []
        failed = False

        for index_value in rng.permutation(
            len(specifications)
        ):
            index = int(index_value)
            orientation, length = specifications[index]
            candidates = []

            if orientation == "h":
                row_limit = height
                col_limit = width - length + 1
            else:
                row_limit = height - length + 1
                col_limit = width

            for row in range(row_limit):
                for col in range(col_limit):
                    cells = _cells_for_start(
                        orientation,
                        length,
                        row,
                        col,
                    )

                    if all(
                        abs(cr - old_row) > 1 or abs(cc - old_col) > 1
                        for cr, cc in cells for old_row, old_col in occupied
                    ):
                        candidates.append(cells)

            if not candidates:
                failed = True
                break

            selected = candidates[
                int(rng.integers(0, len(candidates)))
            ]

            placed[index] = selected
            occupied.extend(selected)

        if not failed:
            return placed

    return None


def _generate_pair(
    rng,
    rank_colors,
    background,
    source,
    lengths,
    orientations,
):
    specifications = list(zip(orientations, lengths))

    maximum_vertical = max(
        (
            length
            for orientation, length in specifications
            if orientation == "v"
        ),
        default=1,
    )

    maximum_horizontal = max(
        (
            length
            for orientation, length in specifications
            if orientation == "h"
        ),
        default=1,
    )

    minimum_height = max(6, maximum_vertical)
    minimum_width = max(6, maximum_horizontal)
    density = int(rng.choice(3, p=(0.43, 0.35, 0.22)))
    extra = ((0, 4), (3, 9), (8, 16))[density]
    height = min(30, minimum_height + int(rng.integers(*extra)))
    width = min(30, minimum_width + int(rng.integers(*extra)))
    placements = None
    for expansion in range(12):
        placements = _place_segments(rng, height, width, specifications)
        if placements is not None:
            break
        height = min(30, height + 1)
        width = min(30, width + 1)
    assert placements is not None

    input_grid = np.full(
        (height, width),
        background,
        dtype=np.int8,
    )

    for component in placements:
        for row, col in component:
            input_grid[row, col] = source

    output_grid = _solve_from_input(
        input_grid,
        background,
        source,
        rank_colors,
    )

    return input_grid, output_grid


def _array_key(grid):
    normalized = np.asarray(
        grid,
        dtype=np.int8,
    )
    return normalized.shape, normalized.tobytes()


def _matches_source_pair(
    input_grid,
    output_grid,
):
    if (
        input_grid.shape != (10, 10)
        or output_grid.shape != (10, 10)
    ):
        return False

    input_rows = tuple(
        "".join(
            str(int(value))
            for value in input_grid[row]
        )
        for row in range(10)
    )

    output_rows = tuple(
        "".join(
            str(int(value))
            for value in output_grid[row]
        )
        for row in range(10)
    )

    return (
        input_rows,
        output_rows,
    ) in _SOURCE_PAIR_SIGNATURES


def _analyze_pair(
    input_grid,
    output_grid,
):
    assert input_grid.shape == output_grid.shape

    input_colors, counts = np.unique(
        input_grid,
        return_counts=True,
    )

    assert input_colors.size == 2

    background_index = int(np.argmax(counts))

    assert (
        int(counts[background_index])
        > int(counts[1 - background_index])
    )

    background = int(
        input_colors[background_index]
    )
    source = int(
        input_colors[1 - background_index]
    )

    mask = input_grid == source
    components = _components(mask)

    assert len(components) == 3

    for first_index in range(3):
        for second_index in range(
            first_index + 1,
            3,
        ):
            for (
                first_row,
                first_col,
            ) in components[first_index]:
                for (
                    second_row,
                    second_col,
                ) in components[second_index]:
                    assert not (
                        abs(
                            first_row - second_row
                        ) <= 1
                        and abs(
                            first_col - second_col
                        ) <= 1
                    )

    records = []

    for component in components:
        rows = [
            row
            for row, _ in component
        ]
        cols = [
            col
            for _, col in component
        ]

        assert len(component) >= 2

        if len(set(rows)) == 1:
            orientation = "h"
            assert (
                max(cols) - min(cols) + 1
                == len(component)
            )
        elif len(set(cols)) == 1:
            orientation = "v"
            assert (
                max(rows) - min(rows) + 1
                == len(component)
            )
        else:
            raise AssertionError(
                "a component is not "
                "a straight segment"
            )

        records.append(
            {
                "cells": component,
                "length": len(component),
                "orientation": orientation,
            }
        )

    records.sort(
        key=lambda record: record["length"],
        reverse=True,
    )

    lengths = tuple(
        record["length"]
        for record in records
    )

    assert len(set(lengths)) == 3
    assert np.all(
        output_grid[~mask] == background
    )

    rank_colors = []

    for record in records:
        values = np.array(
            [
                output_grid[row, col]
                for row, col in record["cells"]
            ],
            dtype=np.int8,
        )

        unique_values = np.unique(values)
        assert unique_values.size == 1
        rank_colors.append(
            int(unique_values[0])
        )

    rank_colors = tuple(rank_colors)

    assert len(set(rank_colors)) == 3
    assert background not in rank_colors
    assert source not in rank_colors

    assert set(
        int(value)
        for value in np.unique(output_grid)
    ) == {
        background,
        *rank_colors,
    }

    return {
        "background": background,
        "source": source,
        "rank_colors": rank_colors,
        "lengths": lengths,
        "orientations": tuple(
            record["orientation"]
            for record in records
        ),
    }


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)

    num_train, num_test = _resolve_counts(
        rng,
        num_train,
        num_test,
    )

    puzzle_parameters = sample_puzzle_parameters(rng)
    train_orientations, test_orientations = _orientation_plan(rng, num_train, num_test)
    training_lengths, test_lengths = _length_plan(rng, num_train, num_test)
    available = puzzle_parameters["available"]
    first_palette = (available[0], available[1])
    second_palette = (available[2], available[3])
    palettes = [first_palette]
    if puzzle_parameters["vary_palette"]:
        palettes.append(second_palette)
    else:
        palettes.append(first_palette)
    for _ in range(num_train + num_test - 2):
        if puzzle_parameters["vary_palette"]:
            pair = tuple(int(x) for x in rng.choice(available, 2, replace=False))
            palettes.append(pair)
        else:
            palettes.append(first_palette)

    puzzle = {
        "train": [],
        "test": [],
    }

    input_keys = set()
    pair_keys = set()

    def add_pair(section, lengths, orientations, palette):
        pair_rng = np.random.default_rng(
            int(
                rng.integers(
                    0,
                    np.iinfo(np.int64).max,
                )
            )
        )

        for _ in range(100):
            input_grid, output_grid = (
                _generate_pair(
                    pair_rng,
                    puzzle_parameters["rank_colors"],
                    palette[0],
                    palette[1],
                    lengths,
                    orientations,
                )
            )

            input_key = _array_key(input_grid)
            pair_key = (
                input_key,
                _array_key(output_grid),
            )

            if (
                input_key not in input_keys
                and pair_key not in pair_keys
                and not _matches_source_pair(
                    input_grid,
                    output_grid,
                )
            ):
                input_keys.add(input_key)
                pair_keys.add(pair_key)

                puzzle[section].append(
                    {
                        "input": input_grid,
                        "output": output_grid,
                    }
                )
                return

        raise RuntimeError(
            "could not generate a new unique pair"
        )

    for i in range(num_train):
        add_pair("train", training_lengths[i], train_orientations[i], palettes[i])
    for i in range(num_test):
        add_pair("test", test_lengths[i], test_orientations[i], palettes[num_train + i])

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(
    puzzle: dict,
) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {
        "train",
        "test",
    }
    assert len(puzzle) == 2

    assert isinstance(
        puzzle["train"],
        list,
    )
    assert isinstance(
        puzzle["test"],
        list,
    )

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert (
        len(puzzle["train"])
        > len(puzzle["test"])
    )

    input_keys = set()
    pair_keys = set()

    information = {
        "train": [],
        "test": [],
    }

    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict)
            assert set(pair.keys()) == {
                "input",
                "output",
            }
            assert len(pair) == 2

            input_grid = pair["input"]
            output_grid = pair["output"]

            for grid in (
                input_grid,
                output_grid,
            ):
                assert isinstance(
                    grid,
                    np.ndarray,
                )
                assert grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.issubdtype(
                    grid.dtype,
                    np.integer,
                )
                assert np.all(
                    (0 <= grid) & (grid <= 9)
                )

            assert (
                input_grid.shape
                == output_grid.shape
            )

            input_key = _array_key(
                input_grid
            )
            pair_key = (
                input_key,
                _array_key(output_grid),
            )

            assert input_key not in input_keys
            assert pair_key not in pair_keys

            input_keys.add(input_key)
            pair_keys.add(pair_key)

            assert not np.array_equal(
                input_grid,
                output_grid,
            )

            assert not _matches_source_pair(
                input_grid,
                output_grid,
            )

            information[section].append(
                _analyze_pair(
                    input_grid,
                    output_grid,
                )
            )

    first = information["train"][0]

    learned_rank_colors = (
        first["rank_colors"]
    )

    for section in ("train", "test"):
        for pair, info in zip(
            puzzle[section],
            information[section],
        ):
            assert (
                info["rank_colors"]
                == learned_rank_colors
            )

            reconstructed = (
                _solve_from_input(
                    pair["input"],
                    info["background"],
                    info["source"],
                    learned_rank_colors,
                )
            )

            assert np.array_equal(
                reconstructed,
                pair["output"],
            )

    # Every rank color appears in each training output. Orientation changes
    # across training examples, so a fixed orientation code cannot fit them.
    assert len({info["orientations"] for info in information["train"]}) >= 2


def _puzzles_equal(first, second):
    if (
        set(first.keys())
        != {"train", "test"}
        or set(second.keys())
        != {"train", "test"}
    ):
        return False

    for section in ("train", "test"):
        if (
            len(first[section])
            != len(second[section])
        ):
            return False

        for first_pair, second_pair in zip(
            first[section],
            second[section],
        ):
            for name in ("input", "output"):
                first_grid = first_pair[name]
                second_grid = second_pair[name]

                if (
                    first_grid.dtype
                    != second_grid.dtype
                ):
                    return False

                if not np.array_equal(
                    first_grid,
                    second_grid,
                ):
                    return False

    return True


def _puzzle_signature(puzzle):
    return tuple(
        (
            _array_key(pair["input"]),
            _array_key(pair["output"]),
        )
        for section in ("train", "test")
        for pair in puzzle[section]
    )


def self_test() -> None:
    automatic = []
    train_counts = set()
    test_counts = set()
    puzzle_signatures = set()
    palette_signatures = set()
    shapes = set()
    layouts = set()
    length_triples = set()
    train_patterns = set()
    test_patterns = set()
    review = {
        "small": 0, "single_test": 0, "variable_palette": 0,
        "constant_palette": 0, "repeated_train": 0,
        "overlapping_absolute_lengths": 0, "test_new_lengths": 0,
        "test_replayed_lengths": 0, "test_new_orientation": 0,
        "test_familiar_orientation": 0, "compact": 0,
        "dense": 0, "long": 0, "border": 0,
        "combined": 0, "source_like_shift": 0,
        "opposite_shift": 0, "all_familiar_tests": 0,
    }

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)

        automatic.append(puzzle)
        train_counts.add(
            len(puzzle["train"])
        )
        test_counts.add(
            len(puzzle["test"])
        )
        puzzle_signatures.add(
            _puzzle_signature(puzzle)
        )

        first_info = _analyze_pair(
            puzzle["train"][0]["input"],
            puzzle["train"][0]["output"],
        )

        palette_signatures.add(
            (
                first_info["background"],
                first_info["source"],
                *first_info["rank_colors"],
            )
        )

        training_info = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        testing_info = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        train_orientations = {i["orientations"] for i in training_info}
        train_lengths = {i["lengths"] for i in training_info}
        assert len(train_orientations) >= 2
        roles = {(i["background"], i["source"]) for i in training_info + testing_info}
        review["small"] += len(training_info) <= 4
        review["single_test"] += len(testing_info) == 1
        review["variable_palette" if len(roles) > 1 else "constant_palette"] += 1
        review["repeated_train"] += any(
            a["lengths"] == b["lengths"] and a["orientations"] != b["orientations"]
            for j, a in enumerate(training_info) for b in training_info[j + 1:]
        )
        numeric_ranks = {}
        for info in training_info:
            for rank, length in enumerate(info["lengths"]):
                numeric_ranks.setdefault(length, set()).add(rank)
        review["overlapping_absolute_lengths"] += any(len(x) > 1 for x in numeric_ranks.values())
        new_lengths = any(i["lengths"] not in train_lengths for i in testing_info)
        new_orientation = any(i["orientations"] not in train_orientations for i in testing_info)
        review["test_new_lengths"] += new_lengths
        review["test_replayed_lengths"] += any(i["lengths"] in train_lengths for i in testing_info)
        review["test_new_orientation"] += new_orientation
        review["test_familiar_orientation"] += any(i["orientations"] in train_orientations for i in testing_info)
        review["combined"] += len(roles) > 1 and new_lengths and new_orientation
        test_orientations = {i["orientations"] for i in testing_info}
        review["source_like_shift"] += (
            ("h", "h", "v") in test_orientations
            and ("h", "h", "v") not in train_orientations
            and ("v", "v", "v") in train_orientations
            and bool({("v", "h", "h"), ("h", "v", "h")} & train_orientations)
        )
        review["opposite_shift"] += any(
            p not in train_orientations and p != ("h", "h", "v")
            for p in test_orientations
        )
        review["all_familiar_tests"] += test_orientations <= train_orientations

        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _analyze_pair(
                    pair["input"],
                    pair["output"],
                )

                shapes.add(
                    pair["input"].shape
                )

                source_mask = np.asarray(
                    pair["input"]
                    == info["source"],
                    dtype=np.uint8,
                )

                layouts.add(
                    (
                        source_mask.shape,
                        source_mask.tobytes(),
                    )
                )

                length_triples.add(
                    info["lengths"]
                )
                if section == "train":
                    train_patterns.add(info["orientations"])
                else:
                    test_patterns.add(info["orientations"])
                height, width = pair["input"].shape
                rows, cols = np.nonzero(source_mask)
                review["compact"] += height <= 10 and width <= 10
                review["dense"] += float(source_mask.mean()) >= 0.2
                review["long"] += info["lengths"][0] > 9
                review["border"] += bool(
                    np.any(rows == 0) or np.any(rows == height - 1)
                    or np.any(cols == 0) or np.any(cols == width - 1)
                )

    assert len(train_counts) >= 4
    assert len(test_counts) >= 3
    assert len(puzzle_signatures) >= 40
    assert len(palette_signatures) >= 30
    assert len(shapes) >= 20
    assert len(layouts) >= 100
    assert len(length_triples) >= 25
    assert len(train_patterns) == len(_ORIENTATIONS)
    assert len(test_patterns) == len(_ORIENTATIONS)
    assert review["small"] >= 10 and review["single_test"] >= 10
    assert review["variable_palette"] >= 8 and review["constant_palette"] >= 8
    assert review["repeated_train"] >= 8
    assert review["overlapping_absolute_lengths"] >= 8
    assert review["test_new_lengths"] >= 8 and review["test_replayed_lengths"] >= 8
    assert review["test_new_orientation"] >= 8 and review["test_familiar_orientation"] >= 8
    assert review["compact"] >= 20 and review["dense"] >= 20
    assert review["long"] >= 20 and review["border"] >= 20
    assert review["combined"] >= 2
    assert review["source_like_shift"] >= 4
    assert review["opposite_shift"] >= 4
    assert review["all_familiar_tests"] >= 4

    explicit = generate_puzzle(
        1001,
        num_train=7,
        num_test=3,
    )

    validate_generated_puzzle(explicit)

    assert len(explicit["train"]) == 7
    assert len(explicit["test"]) == 3

    automatic_repeat = generate_puzzle(17)

    assert _puzzles_equal(
        automatic[17],
        automatic_repeat,
    )

    explicit_first = generate_puzzle(
        2027,
        num_train=6,
        num_test=2,
    )
    explicit_second = generate_puzzle(
        2027,
        num_train=6,
        num_test=2,
    )

    assert _puzzles_equal(
        explicit_first,
        explicit_second,
    )

    varied = {
        _puzzle_signature(
            generate_puzzle(
                3000 + offset,
                5,
                2,
            )
        )
        for offset in range(8)
    }

    assert len(varied) >= 7

    boundary_cases = (
        (2, 1),
        (10, 1),
        (10, 4),
    )

    for train_count, test_count in (
        boundary_cases
    ):
        puzzle = generate_puzzle(
            4000
            + train_count * 10
            + test_count,
            num_train=train_count,
            num_test=test_count,
        )

        validate_generated_puzzle(puzzle)

        assert (
            len(puzzle["train"])
            == train_count
        )
        assert (
            len(puzzle["test"])
            == test_count
        )

    partial_cases = (
        {
            "num_train": None,
            "num_test": 1,
        },
        {
            "num_train": None,
            "num_test": 4,
        },
        {
            "num_train": 2,
            "num_test": None,
        },
        {
            "num_train": 10,
            "num_test": None,
        },
    )

    for offset, arguments in enumerate(
        partial_cases
    ):
        puzzle = generate_puzzle(
            5000 + offset,
            **arguments,
        )

        validate_generated_puzzle(puzzle)

        if arguments["num_train"] is not None:
            assert (
                len(puzzle["train"])
                == arguments["num_train"]
            )

        if arguments["num_test"] is not None:
            assert (
                len(puzzle["test"])
                == arguments["num_test"]
            )

    illegal_cases = (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {
            "num_train": 2,
            "num_test": 2,
        },
        {
            "num_train": 3,
            "num_test": 4,
        },
    )

    for arguments in illegal_cases:
        try:
            generate_puzzle(
                6000,
                **arguments,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "illegal counts were accepted: "
                f"{arguments}"
            )
