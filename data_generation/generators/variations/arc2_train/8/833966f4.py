"""Generate five-cell adjacent-pair swap puzzles."""

from itertools import product

import numpy as np


_PERMUTATION = (1, 0, 2, 4, 3)
_SOURCE_INPUTS = {
    (9, 0, 1, 6, 8),
    (4, 3, 6, 2, 8),
    (4, 5, 6, 7, 2),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _normalize_count(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if name == "num_train" and not 2 <= value <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if name == "num_test" and not 1 <= value <= 4:
        raise ValueError("num_test must be between 1 and 4")
    return value


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _normalize_count("num_train", num_train)
    test_count = _normalize_count("num_test", num_test)
    if train_count is None and test_count is None:
        schedule = ((2, 1), (2, 1), (2, 1), (2, 1), (3, 1), (3, 1),
                    (3, 2), (4, 2), (5, 3), (7, 3), (9, 4))
        slot = seed % len(schedule)
        train_count, test_count = schedule[slot]
        if slot >= 8:
            train_count += int(rng.integers(2))
        return train_count, test_count
    if train_count is None:
        choices = np.arange(max(2, (test_count or 0) + 1), 11)
        weights = np.asarray([24, 20, 16, 11, 9, 7, 6, 4, 3], dtype=float)[choices - 2]
        train_count = int(rng.choice(choices, p=weights / weights.sum()))
    if test_count is None:
        choices = np.arange(1, min(4, train_count - 1) + 1)
        weights = np.asarray([35, 30, 25, 10], dtype=float)[choices - 1]
        test_count = int(rng.choice(choices, p=weights / weights.sum()))
    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")
    return train_count, test_count


def _swapped(sequence: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sequence[index] for index in _PERMUTATION)


def _make_pair(sequence: tuple[int, ...], shape: tuple[int, int]) -> dict:
    return {
        "input": np.asarray(sequence, dtype=np.int8).reshape(shape),
        "output": np.asarray(_swapped(sequence), dtype=np.int8).reshape(shape),
    }


def _is_available(
    sequence: tuple[int, ...],
    shape: tuple[int, int],
    used_inputs: set[tuple[int, ...]],
    used_outputs: set[tuple[int, ...]],
) -> bool:
    swapped = _swapped(sequence)
    return (
        sequence != swapped
        and (shape != (5, 1) or sequence not in _SOURCE_INPUTS)
        and sequence not in used_inputs
        and sequence not in used_outputs
        and swapped not in used_inputs
    )


def _draw_sequence(
    rng: np.random.Generator,
    familiar: tuple[int, ...],
    novel: tuple[int, ...],
    unique_count: int,
    novel_count: int,
    exact_colors: tuple[int, ...] | None,
    special_layout: bool,
) -> tuple[int, ...]:
    if exact_colors is not None:
        chosen = exact_colors
    else:
        chosen_novel = tuple(int(x) for x in rng.choice(novel, size=novel_count, replace=False))
        chosen_familiar = tuple(
            int(x) for x in rng.choice(familiar, size=unique_count - novel_count, replace=False)
        )
        chosen = chosen_novel + chosen_familiar

    if special_layout and novel_count == 2 and unique_count == 5:
        layout_kind = int(rng.choice(3, p=[0.8, 0.1, 0.1]))
        if layout_kind == 0:
            positions = (int(rng.choice((0, 1))), int(rng.choice((3, 4))))
        elif layout_kind == 1:
            positions = (0, 1) if int(rng.integers(2)) == 0 else (3, 4)
        else:
            positions = (2, int(rng.choice((0, 1, 3, 4))))
        sequence = [None] * 5
        shuffled_novel = rng.permutation(chosen[:2])
        shuffled_familiar = iter(rng.permutation(chosen[2:]))
        for position, color in zip(positions, shuffled_novel):
            sequence[position] = int(color)
        for position in range(5):
            if sequence[position] is None:
                sequence[position] = int(next(shuffled_familiar))
        return tuple(sequence)

    values = list(chosen)
    if unique_count < 5:
        values.extend(int(x) for x in rng.choice(chosen, size=5 - unique_count))
    return tuple(int(x) for x in rng.permutation(values))


def _sample_pair(
    rng: np.random.Generator,
    shape: tuple[int, int],
    familiar: tuple[int, ...],
    novel: tuple[int, ...],
    unique_count: int,
    novel_count: int,
    used_inputs: set[tuple[int, ...]],
    used_outputs: set[tuple[int, ...]],
    exact_colors: tuple[int, ...] | None = None,
    special_layout: bool = False,
) -> dict:
    for _ in range(256):
        sequence = _draw_sequence(
            rng, familiar, novel, unique_count, novel_count, exact_colors, special_layout
        )
        if _is_available(sequence, shape, used_inputs, used_outputs):
            break
    else:
        allowed = exact_colors if exact_colors is not None else familiar + novel
        for sequence in product(sorted(allowed), repeat=5):
            if (
                len(set(sequence)) == unique_count
                and len(set(sequence) & set(novel)) == novel_count
                and (exact_colors is None or set(sequence) == set(exact_colors))
                and _is_available(sequence, shape, used_inputs, used_outputs)
            ):
                break
        else:
            raise RuntimeError("no unique valid five-cell sequence remains")
    used_inputs.add(sequence)
    used_outputs.add(_swapped(sequence))
    return _make_pair(sequence, shape)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)

    # Shape is the only puzzle-level convention. The vertical source shape dominates.
    shape = (5, 1) if rng.random() < 0.8 else (1, 5)
    first_palette = tuple(int(x) for x in rng.choice(10, size=5, replace=False))
    overlap = seed % 6
    shared = tuple(int(x) for x in rng.choice(first_palette, size=overlap, replace=False))
    remaining = tuple(color for color in range(10) if color not in first_palette)
    added = tuple(int(x) for x in rng.choice(remaining, size=5 - overlap, replace=False))
    second_palette = shared + added
    familiar = tuple(sorted(set(first_palette) | set(second_palette)))
    novel = tuple(color for color in range(10) if color not in familiar)

    used_inputs: set[tuple[int, ...]] = set()
    used_outputs: set[tuple[int, ...]] = set()
    puzzle = {"train": [], "test": []}

    # Two complete demonstrations make all five positional correspondences visible.
    for palette in (first_palette, second_palette):
        puzzle["train"].append(
            _sample_pair(rng, shape, palette, (), 5, 0, used_inputs, used_outputs,
                         exact_colors=palette)
        )
    for _ in range(train_count - 2):
        unique_count = int(rng.choice((3, 4, 5), p=(0.2, 0.3, 0.5)))
        puzzle["train"].append(
            _sample_pair(rng, shape, familiar, (), unique_count, 0,
                         used_inputs, used_outputs)
        )
    for _ in range(test_count):
        possible = np.arange(min(3, len(novel)) + 1)
        novelty_weights = np.asarray((18, 22, 50, 10), dtype=float)[possible]
        novel_count = int(rng.choice(possible, p=novelty_weights / novelty_weights.sum()))
        unique_count = int(rng.choice((3, 4, 5), p=(0.05, 0.10, 0.85)))
        puzzle["test"].append(
            _sample_pair(rng, shape, familiar, novel, unique_count, novel_count,
                         used_inputs, used_outputs, special_layout=True)
        )

    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(grid: np.ndarray) -> tuple:
    return tuple(grid.shape), tuple(int(x) for x in grid.flat)


def _mapping_candidates(train_pairs: list[dict]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            input_position for input_position in range(5)
            if all(
                int(pair["output"].flat[output_position])
                == int(pair["input"].flat[input_position])
                for pair in train_pairs
            )
        )
        for output_position in range(5)
    )


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "invalid pair counts")

    shapes = set()
    inputs = set()
    pairs = set()
    input_sequences = set()
    output_sequences = set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label} must contain exactly input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         f"{label} {name} must be a 2D array")
                _require(all(1 <= dimension <= 30 for dimension in grid.shape),
                         f"{label} {name} has invalid dimensions")
                _require(np.issubdtype(grid.dtype, np.integer),
                         f"{label} {name} must have integer dtype")
                _require(bool(np.all((0 <= grid) & (grid <= 9))),
                         f"{label} {name} has an invalid color")
            input_grid, output_grid = pair["input"], pair["output"]
            _require(input_grid.shape in ((5, 1), (1, 5)) and
                     output_grid.shape == input_grid.shape,
                     f"{label} has invalid five-cell geometry")
            shapes.add(input_grid.shape)
            sequence = tuple(int(x) for x in input_grid.flat)
            result = tuple(int(x) for x in output_grid.flat)
            _require(result == _swapped(sequence), f"{label} violates the swap rule")
            _require(result != sequence, f"{label} is unchanged")
            _require(input_grid.shape != (5, 1) or sequence not in _SOURCE_INPUTS,
                     f"{label} copies a source pair")
            input_sig = _signature(input_grid)
            pair_sig = (input_sig, _signature(output_grid))
            _require(input_sig not in inputs and pair_sig not in pairs,
                     f"{label} duplicates a prior pair")
            _require(sequence not in output_sequences and result not in input_sequences,
                     f"{label} is a redundant inverse of a prior pair")
            inputs.add(input_sig)
            pairs.add(pair_sig)
            input_sequences.add(sequence)
            output_sequences.add(result)

    _require(len(shapes) == 1, "orientation must be shared by the puzzle")
    first_two = puzzle["train"][:2]
    _require(all(len(set(int(x) for x in pair["input"].flat)) == 5 for pair in first_two),
             "the first two training pairs must show both swaps")
    _require(_mapping_candidates(puzzle["train"]) == ((1,), (0,), (2,), (4,), (3,)),
             "training pairs do not uniquely reveal the positional mapping")
    train_colors = {int(x) for pair in puzzle["train"] for x in pair["input"].flat}
    first_two_colors = {int(x) for pair in first_two for x in pair["input"].flat}
    _require(train_colors == first_two_colors,
             "later training pairs introduce colors outside the demonstrated palette")
    for pair in puzzle["test"]:
        _require(len(set(int(x) for x in pair["input"].flat) - train_colors) <= 3,
                 "test uses too many novel colors")


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(
        tuple((_signature(pair["input"]), _signature(pair["output"])) for pair in puzzle[split])
        for split in ("train", "test")
    )


def _expect_value_error(function) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError was not raised")


def self_test() -> None:
    train_counts, test_counts, shapes, overlaps = set(), set(), set(), set()
    signatures, sequences, novel_counts, unique_counts = set(), set(), set(), set()
    novel_positions, combined_cases, opposite_side_masks = set(), set(), set()
    mixed_unique_puzzles = mixed_novelty_puzzles = 0
    unseen_training_centers = horizontal_novel_repeats = 0
    small_puzzles = large_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_puzzle_signature(puzzle) == _puzzle_signature(generate_puzzle(seed)),
                 "generation is not deterministic")
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        small_puzzles += (len(puzzle["train"]), len(puzzle["test"])) == (2, 1)
        large_puzzles += len(puzzle["train"]) >= 8
        signatures.add(_puzzle_signature(puzzle))
        shapes.add(puzzle["train"][0]["input"].shape)
        first, second = (set(int(x) for x in p["input"].flat) for p in puzzle["train"][:2])
        overlaps.add(len(first & second))
        train_colors = {int(x) for pair in puzzle["train"] for x in pair["input"].flat}
        training_centers = {int(pair["input"].flat[2]) for pair in puzzle["train"]}
        local_unique_counts, local_novel_counts = set(), set()
        for pair in puzzle["train"] + puzzle["test"]:
            sequence = tuple(int(x) for x in pair["input"].flat)
            result = tuple(int(x) for x in pair["output"].flat)
            _require(result == (sequence[1], sequence[0], sequence[2], sequence[4], sequence[3]),
                     "self-test found an incorrect output")
            sequences.add((pair["input"].shape, sequence))
            unique_counts.add(len(set(sequence)))
            local_unique_counts.add(len(set(sequence)))
        for pair in puzzle["test"]:
            sequence = tuple(int(x) for x in pair["input"].flat)
            new = set(sequence) - train_colors
            novel_counts.add(len(new))
            local_novel_counts.add(len(new))
            unseen_training_centers += sequence[2] not in training_centers
            horizontal_novel_repeats += (
                pair["input"].shape == (1, 5) and bool(new) and len(set(sequence)) < 5
            )
            positions = tuple(i for i, color in enumerate(sequence) if color in new)
            novel_positions.add(positions)
            combined_cases.add((len(new), len(set(sequence)) < 5))
            if len(new) == 2 and len(set(sequence)) == 5:
                opposite_side_masks.add(positions)
        mixed_unique_puzzles += len(local_unique_counts) > 1
        mixed_novelty_puzzles += len(local_novel_counts) > 1

    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack diversity")
    _require(len(signatures) == 42 and len(sequences) >= 100,
             "cross-seed puzzle or sequence diversity is too low")
    _require(shapes == {(5, 1), (1, 5)}, "orientation regime was not exercised")
    _require(overlaps == set(range(6)), "training palette overlap lacks diversity")
    _require(novel_counts == {0, 1, 2, 3}, "test novelty regimes are missing")
    _require({3, 4, 5}.issubset(unique_counts), "repeated-color regimes are missing")
    _require(any(2 in positions for positions in novel_positions),
             "no novel color appeared at the center")
    _require(any({0, 1}.issubset(positions) or {3, 4}.issubset(positions)
                 for positions in novel_positions),
             "no test put two novel colors on one side")
    _require(any(count > 0 and repeated for count, repeated in combined_cases),
             "novel and repeated colors never combine")
    _require({(0, 3), (0, 4), (1, 3), (1, 4)}.issubset(opposite_side_masks),
             "the four source-like novel-color placements were not all exercised")
    _require(small_puzzles >= 4 and large_puzzles >= 2,
             "compact and large puzzle counts were not both exercised")
    _require(mixed_unique_puzzles >= 5 and mixed_novelty_puzzles >= 2,
             "within-puzzle variation was not exercised")
    _require(unseen_training_centers > 0 and horizontal_novel_repeats > 0,
             "center novelty or a compatible orientation/repetition combination is missing")

    _require(_puzzle_signature(generate_puzzle(8001, 6, 2)) !=
             _puzzle_signature(generate_puzzle(8002, 6, 2)),
             "different seeds produced identical puzzles")
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(2000 + train_count * 10 + test_count,
                                 train_count, test_count)
        validate_generated_puzzle(puzzle)
        _require((len(puzzle["train"]), len(puzzle["test"])) ==
                 (train_count, test_count), "explicit counts were not honored")
    for seed, train_count, test_count in ((3101, None, 1), (3102, None, 4),
                                          (3103, 2, None), (3104, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        _require((train_count is None or len(puzzle["train"]) == train_count) and
                 (test_count is None or len(puzzle["test"]) == test_count),
                 "partial count was not honored")
    for train_count, test_count in ((1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (3, 4)):
        _expect_value_error(lambda train_count=train_count, test_count=test_count:
                            generate_puzzle(1, train_count, test_count))
