"""Generate 3-by-3 counting puzzles with a fixed four-cell output stencil."""

import numpy as np


_STENCIL = (0, 1, 2, 4)
_CORNERS = frozenset((0, 2, 6, 8))
_SOURCE_MASKS = frozenset(
    ((0, 1, 6, 8), (0, 1, 6), (2, 6), (1, 3, 4, 6),
     (0, 5, 7, 8), (1, 3, 4), (1, 3), (1, 5),
     (2,), (3,), (1, 4, 5, 6), (1, 7))
)
_MASKS = {
    count: tuple(
        tuple(position for position in range(9) if bits & (1 << position))
        for bits in range(1 << 9) if bits.bit_count() == count
    ) for count in range(1, 5)
}


def _signature(grid: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in grid.flat)


def _original_flat(mask: tuple[int, ...], color: int) -> tuple[int, ...]:
    return tuple(color if position in mask else 0 for position in range(9))


_SOURCE_PAIRS = frozenset(
    (_original_flat(mask, 1), _original_flat(_STENCIL[:len(mask)], 2))
    for mask in _SOURCE_MASKS
)


def _count_arg(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _pair_counts(rng: np.random.Generator, seed: int,
                 num_train: int | None, num_test: int | None) -> tuple[int, int]:
    num_train = _count_arg(num_train, "num_train", 2, 10)
    num_test = _count_arg(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _count_schedule(rng: np.random.Generator, num_train: int,
                    num_test: int) -> tuple[list[int], list[int]]:
    primary = int(rng.integers(1, 5))
    others = [int(value) for value in rng.permutation(
        [count for count in range(1, 5) if count != primary]
    )]
    if num_train == 2:
        train = [primary, others[0]]
    elif num_train == 3:
        train = ([primary, primary, others[0]] if rng.random() < 0.65
                 else [primary, others[0], others[1]])
    elif num_train == 4:
        train = ([primary, primary, others[0], others[1]]
                 if rng.random() < 0.75 else [1, 2, 3, 4])
    else:
        train = [1, 2, 3, 4, primary]
        while len(train) < num_train:
            train.extend(int(value) for value in rng.permutation([1, 2, 3, 4]))
        train = train[:num_train]
    train = [int(train[index]) for index in rng.permutation(len(train))]
    tested = [primary]
    available = [int(value) for value in rng.permutation(
        [count for count in set(train) if count != primary]
    )]
    tested.extend(available[:num_test - 1])
    assert len(tested) == num_test
    tested = [int(tested[index]) for index in rng.permutation(len(tested))]
    return train, tested


def _connected(mask: tuple[int, ...]) -> bool:
    visited = {mask[0]}
    frontier = [mask[0]]
    locations = set(mask)
    while frontier:
        position = frontier.pop()
        row, col = divmod(position, 3)
        for neighbor in locations - visited:
            nr, nc = divmod(neighbor, 3)
            if abs(row - nr) + abs(col - nc) == 1:
                visited.add(neighbor)
                frontier.append(neighbor)
    return len(visited) == len(mask)


def _gapped_two(mask: tuple[int, ...]) -> bool:
    if len(mask) != 2:
        return False
    r0, c0 = divmod(mask[0], 3)
    r1, c1 = divmod(mask[1], 3)
    return (r0 == r1 and abs(c0 - c1) == 2) or (c0 == c1 and abs(r0 - r1) == 2)


def _contrast(first: tuple[int, ...], second: tuple[int, ...]) -> float:
    return (
        4 * (_connected(first) != _connected(second))
        + 3 * ((4 in first) != (4 in second))
        + 3 * (_gapped_two(first) != _gapped_two(second))
        + 2 * ((sum(p in _CORNERS for p in first) >= 3)
               != (sum(p in _CORNERS for p in second) >= 3))
        + abs(len(set(first) & set(_STENCIL))
              - len(set(second) & set(_STENCIL)))
        + len(set(first) ^ set(second)) / 2
    )


def _choose_mask(rng: np.random.Generator,
                 candidates: list[tuple[int, ...]],
                 same_count_train: list[tuple[int, ...]]) -> tuple[int, ...]:
    if not same_count_train or rng.random() < 0.3:
        return candidates[int(rng.integers(len(candidates)))]
    scores = [min(_contrast(mask, train_mask) for train_mask in same_count_train)
              for mask in candidates]
    highest = max(scores)
    strong = [mask for mask, score in zip(candidates, scores)
              if score >= highest - 1.5]
    return strong[int(rng.integers(len(strong)))]


def _make_grid(mask: tuple[int, ...], background: int,
               marker: int) -> np.ndarray:
    grid = np.full((3, 3), background, dtype=np.int8)
    for position in mask:
        grid.flat[position] = marker
    return grid


def _colors(rng: np.random.Generator, output_color: int,
            previous: tuple[int, int] | None,
            encourage_change: bool) -> tuple[int, int]:
    backgrounds = [color for color in range(10) if color != output_color]
    if previous is not None and encourage_change and rng.random() < 0.7:
        backgrounds = [color for color in backgrounds if color != previous[0]]
    background = int(backgrounds[int(rng.integers(len(backgrounds)))])
    markers = [color for color in range(10)
               if color not in (background, output_color)]
    if previous is not None and encourage_change and rng.random() < 0.7:
        markers = [color for color in markers if color != previous[1]]
    marker = int(markers[int(rng.integers(len(markers)))])
    return background, marker


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    num_train, num_test = _pair_counts(rng, seed, num_train, num_test)
    output_color = int(rng.integers(10))
    train_counts, test_counts = _count_schedule(rng, num_train, num_test)
    canonical_train = (train_counts.count(4) >= 2 and 4 in test_counts
                       and rng.random() < 0.55)
    match_repeated_background = rng.random() < 0.7
    used_masks: set[tuple[int, ...]] = set()
    train_masks: dict[int, list[tuple[int, ...]]] = {count: [] for count in range(1, 5)}
    train_backgrounds: dict[int, int] = {}
    train_marker_colors: set[int] = set()
    puzzle: dict = {"train": [], "test": []}
    previous_colors = None

    for section, counts in (("train", train_counts), ("test", test_counts)):
        for count in counts:
            background, marker = _colors(
                rng, output_color, previous_colors, section == "train"
            )
            if section == "train" and count in train_backgrounds and match_repeated_background:
                background = train_backgrounds[count]
                if marker == background:
                    choices = [color for color in range(10)
                               if color not in (background, output_color)]
                    marker = int(choices[int(rng.integers(len(choices)))])
            if section == "test":
                demonstrated = sorted(color for color in train_marker_colors
                                     if color != background)
                if not demonstrated:
                    only_marker = next(iter(train_marker_colors))
                    choices = [color for color in range(10)
                               if color not in (only_marker, output_color)]
                    background = int(choices[int(rng.integers(len(choices)))])
                    demonstrated = [only_marker]
                marker = int(demonstrated[int(rng.integers(len(demonstrated)))])
            previous_colors = (background, marker)
            forced = None
            if section == "train" and count == 4 and canonical_train:
                if not train_masks[4]:
                    forced = _STENCIL
            elif section == "test" and count == 4 and rng.random() < 0.3:
                forced = _STENCIL

            output_grid = _make_grid(_STENCIL[:count], background, output_color)
            candidates = [mask for mask in _MASKS[count]
                          if mask not in used_masks and (
                              _signature(_make_grid(mask, background, marker)),
                              _signature(output_grid)
                          ) not in _SOURCE_PAIRS]
            if not candidates:
                raise ValueError("no unused non-source layout remains")
            mask = forced if forced in candidates else _choose_mask(
                rng, candidates, train_masks[count]
            )
            used_masks.add(mask)
            pair = {
                "input": _make_grid(mask, background, marker),
                "output": output_grid,
            }
            puzzle[section].append(pair)
            if section == "train":
                train_backgrounds.setdefault(count, background)
                train_marker_colors.add(marker)
                train_masks[count].append(mask)

    validate_generated_puzzle(puzzle)
    return puzzle


def _roles(grid: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    colors, frequencies = np.unique(grid, return_counts=True)
    assert len(colors) == 2
    background = int(colors[int(np.argmax(frequencies))])
    marker = int(colors[int(np.argmin(frequencies))])
    mask = tuple(int(position) for position in np.flatnonzero(grid != background))
    return background, marker, mask


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    num_train, num_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test
    seen_inputs = set()
    seen_pairs = set()
    output_colors = set()
    train_counts = set()
    train_marker_colors = set()
    train_masks: dict[int, set[tuple[int, ...]]] = {count: set() for count in range(1, 5)}
    test_masks: dict[int, set[tuple[int, ...]]] = {count: set() for count in range(1, 5)}

    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert grid.shape == (3, 3)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            input_grid, output_grid = pair["input"], pair["output"]
            assert not np.array_equal(input_grid, output_grid)
            background, input_color, mask = _roles(input_grid)
            output_background, output_color, output_mask = _roles(output_grid)
            assert background == output_background
            assert len({background, input_color, output_color}) == 3
            assert 1 <= len(mask) <= 4 and output_mask == _STENCIL[:len(mask)]
            output_colors.add(output_color)
            input_signature, output_signature = _signature(input_grid), _signature(output_grid)
            assert input_signature not in seen_inputs
            assert (input_signature, output_signature) not in seen_pairs
            assert (input_signature, output_signature) not in _SOURCE_PAIRS
            seen_inputs.add(input_signature)
            seen_pairs.add((input_signature, output_signature))
            if section == "train":
                train_counts.add(len(mask))
                train_marker_colors.add(input_color)
                train_masks[len(mask)].add(mask)
            else:
                assert input_color in train_marker_colors
                test_masks[len(mask)].add(mask)

    assert len(output_colors) == 1
    for count, masks in test_masks.items():
        if masks:
            assert count in train_counts
            assert not masks & train_masks[count]


def _equal_puzzles(first: dict, second: dict) -> bool:
    return all(
        len(first[section]) == len(second[section]) and all(
            np.array_equal(a[key], b[key])
            for a, b in zip(first[section], second[section])
            for key in ("input", "output")
        ) for section in ("train", "test")
    )


def _predict_from_examples(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    # This check deliberately reads only demonstrations and the test input.
    output_colors = {_roles(pair["output"])[1] for pair in train}
    assert len(output_colors) == 1
    output_color = next(iter(output_colors))
    background, _, mask = _roles(test_input)
    demonstrated = {len(_roles(pair["input"])[2]) for pair in train}
    assert len(mask) in demonstrated
    return _make_grid(_STENCIL[:len(mask)], background, output_color)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for seed, train_count, test_count in ((900, 2, 1), (901, 10, 1),
                                          (902, 10, 4), (123456, 7, 3)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        puzzles.append(puzzle)
    for seed, train_count, test_count in ((910, None, 1), (911, None, 4),
                                          (912, 2, None), (913, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
        puzzles.append(puzzle)
    for request in ({"num_train": 1}, {"num_train": 11},
                    {"num_test": 0}, {"num_test": 5},
                    {"num_train": 2, "num_test": 2},
                    {"num_train": 3, "num_test": 4},
                    {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(999, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal pair counts: {request}")
    assert _equal_puzzles(generate_puzzle(123456, 7, 3),
                          generate_puzzle(123456, 7, 3))
    assert not _equal_puzzles(generate_puzzle(7001, 6, 2),
                              generate_puzzle(7002, 6, 2))
    assert len({len(p["train"]) for p in puzzles[:42]}) >= 4
    assert len({len(p["test"]) for p in puzzles[:42]}) >= 3

    train_layouts, test_layouts, test_counts = set(), set(), set()
    backgrounds, output_colors, input_colors = set(), set(), set()
    bg_varied = marker_varied = repeated_test_count = 0
    gapped_splits, corner_splits = set(), set()
    canonical_contrasts = source_shapes = identical_output_contrasts = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_bgs, train_ics = set(), set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                background, input_color, mask = _roles(pair["input"])
                _, output_color, _ = _roles(pair["output"])
                assert np.array_equal(
                    pair["output"],
                    _make_grid(_STENCIL[:len(mask)], background, output_color)
                )
                backgrounds.add(background)
                input_colors.add(input_color)
                output_colors.add(output_color)
                source_shapes += mask in _SOURCE_MASKS
                if len(mask) == 2 and _gapped_two(mask):
                    gapped_splits.add(section)
                if len(mask) == 4 and sum(p in _CORNERS for p in mask) >= 3:
                    corner_splits.add(section)
                if section == "train":
                    train_layouts.add(mask)
                    train_bgs.add(background)
                    train_ics.add(input_color)
                else:
                    test_layouts.add(mask)
                    test_counts.add(len(mask))
                    assert np.array_equal(
                        _predict_from_examples(puzzle["train"], pair["input"]),
                        pair["output"]
                    )
        bg_varied += len(train_bgs) > 1
        marker_varied += len(train_ics) > 1
        train_by_count = {
            count: [_roles(pair["input"])[2] for pair in puzzle["train"]
                    if len(_roles(pair["input"])[2]) == count]
            for count in range(1, 5)
        }
        repeated_test_count += any(
            len(train_by_count[len(_roles(pair["input"])[2])]) >= 2
            for pair in puzzle["test"]
        )
        identical_output_contrasts += any(
            np.array_equal(first["output"], second["output"])
            and _roles(first["input"])[2] != _roles(second["input"])[2]
            for index, first in enumerate(puzzle["train"])
            for second in puzzle["train"][index + 1:]
            if len(_roles(first["input"])[2]) == len(_roles(second["input"])[2])
        )
        canonical_contrasts += (
            _STENCIL in train_by_count[4]
            and len(train_by_count[4]) >= 2
            and any(len(_roles(pair["input"])[2]) == 4 for pair in puzzle["test"])
        )
    assert test_counts == {1, 2, 3, 4}
    assert len(train_layouts) >= 90 and len(test_layouts) >= 45
    assert len(backgrounds) == 10 and len(output_colors) >= 8
    assert len(input_colors) >= 8
    assert bg_varied >= 30 and marker_varied >= 30
    assert repeated_test_count >= 15
    assert identical_output_contrasts >= 10
    assert gapped_splits == {"train", "test"}
    assert corner_splits == {"train", "test"}
    assert canonical_contrasts >= 1 and source_shapes >= 10
