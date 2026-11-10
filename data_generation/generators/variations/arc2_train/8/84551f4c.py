import numpy as np


_TRIGGER = "trigger"
_PASSIVE = "passive"


# Exact source inputs, represented geometrically. Excluding an exact source
# input is stronger than merely excluding an exact source input/output pair,
# because the transformation is deterministic.
_SOURCE_INPUT_SPECS = (
    (
        3,
        16,
        (
            (0, 3, 1),
            (3, 3, 2),
            (6, 3, 2),
            (10, 3, 2),
            (13, 3, 2),
        ),
    ),
    (
        3,
        16,
        (
            (0, 3, 1),
            (4, 3, 2),
            (8, 3, 1),
            (11, 3, 2),
        ),
    ),
    (
        3,
        7,
        (
            (0, 3, 1),
            (3, 3, 2),
        ),
    ),
    (
        9,
        30,
        (
            (0, 2, 1),
            (3, 4, 2),
            (7, 3, 2),
            (11, 3, 1),
            (14, 3, 2),
            (17, 3, 2),
            (20, 4, 2),
            (24, 3, 2),
            (27, 3, 1),
        ),
    ),
    (
        3,
        30,
        (
            (0, 3, 1),
            (3, 3, 2),
            (7, 3, 2),
            (11, 3, 1),
            (14, 3, 2),
            (17, 3, 2),
            (20, 3, 2),
            (24, 3, 2),
            (27, 3, 1),
        ),
    ),
)


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(int(value) for value in grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_signature(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return (_grid_signature(input_grid), _grid_signature(output_grid))


def _is_source_input(grid: np.ndarray) -> bool:
    for height, width, rods in _SOURCE_INPUT_SPECS:
        if grid.shape != (height, width):
            continue

        source = np.zeros((height, width), dtype=np.int8)
        for column, rod_height, color in rods:
            source[height - rod_height :, column] = color

        if np.array_equal(grid, source):
            return True

    return False


def _validate_optional_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, inclusive"
        )
    return value


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_optional_count("num_train", num_train, 2, 10)
    test_count = _validate_optional_count("num_test", num_test, 1, 4)
    seed_key = int(seed)

    if train_count is None and test_count is None:
        test_count = 1 + seed_key % 4
        minimum_train = max(2, test_count + 1)
        train_count = minimum_train + (
            (seed_key // 4) % (11 - minimum_train)
        )
    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        train_count = minimum_train + seed_key % (11 - minimum_train)
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        test_count = 1 + seed_key % maximum_test

    if not (
        2 <= train_count <= 10
        and 1 <= test_count <= 4
        and train_count > test_count
    ):
        raise ValueError(
            "pair counts must satisfy 2 <= num_train <= 10, "
            "1 <= num_test <= 4, and num_train > num_test"
        )

    return train_count, test_count


def _sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    colors = rng.choice(10, size=3, replace=False)
    return {
        "background": int(colors[0]),
        "trigger": int(colors[1]),
        "passive": int(colors[2]),
    }


def _parse_rods(
    grid: np.ndarray,
    background: int,
    trigger: int,
    passive: int,
) -> list[tuple[int, int, str]]:
    assert grid.ndim == 2
    height, width = grid.shape
    allowed = {background, trigger, passive}
    assert set(int(value) for value in np.unique(grid)).issubset(allowed)

    rods: list[tuple[int, int, str]] = []

    for column in range(width):
        occupied_rows = np.flatnonzero(grid[:, column] != background)
        if len(occupied_rows) == 0:
            continue

        rod_height = int(len(occupied_rows))
        assert rod_height >= 2
        expected_rows = np.arange(height - rod_height, height)
        assert np.array_equal(occupied_rows, expected_rows)

        color = int(grid[occupied_rows[0], column])
        assert np.all(grid[occupied_rows, column] == color)

        if color == trigger:
            role = _TRIGGER
        else:
            assert color == passive
            role = _PASSIVE

        assert column + rod_height <= width
        rods.append((column, rod_height, role))

    assert len(rods) > 0

    # No later rod may begin inside the landing interval of an earlier rod.
    for first, second in zip(rods, rods[1:]):
        first_column, first_height, _ = first
        second_column, _, _ = second
        assert second_column - first_column >= first_height

    return rods


def _active_indices(
    rods: list[tuple[int, int, str]],
) -> set[int]:
    by_column = {
        column: index
        for index, (column, _, _) in enumerate(rods)
    }

    active = {
        index
        for index, (_, _, role) in enumerate(rods)
        if role == _TRIGGER
    }
    stack = list(active)

    while stack:
        index = stack.pop()
        column, rod_height, _ = rods[index]
        successor = by_column.get(column + rod_height)

        if successor is not None and successor not in active:
            active.add(successor)
            stack.append(successor)

    return active


def _longest_trigger_chain(
    rods: list[tuple[int, int, str]],
) -> int:
    by_column = {
        column: index
        for index, (column, _, _) in enumerate(rods)
    }
    best = 0

    for start, (_, _, role) in enumerate(rods):
        if role != _TRIGGER:
            continue

        length = 0
        current = start
        visited: set[int] = set()

        while current not in visited:
            visited.add(current)
            length += 1
            column, rod_height, _ = rods[current]
            successor = by_column.get(column + rod_height)
            if successor is None:
                break
            current = successor

        best = max(best, length)

    return best


def _transform(
    input_grid: np.ndarray,
    background: int,
    trigger: int,
    passive: int,
) -> np.ndarray:
    rods = _parse_rods(input_grid, background, trigger, passive)
    active = _active_indices(rods)
    height, _ = input_grid.shape

    output = input_grid.copy()

    # Clear every activated upright rod first.
    for index in active:
        column, rod_height, _ = rods[index]
        output[height - rod_height :, column] = background

    # Then draw each fallen rod along the common baseline.
    for index in active:
        column, rod_height, role = rods[index]
        color = trigger if role == _TRIGGER else passive
        output[height - 1, column : column + rod_height] = color

    return output


def _make_pair_from_rods(
    height: int,
    width: int,
    rods: list[tuple[int, int, str]],
    parameters: dict,
) -> tuple[np.ndarray, np.ndarray]:
    background = parameters["background"]
    trigger = parameters["trigger"]
    passive = parameters["passive"]

    input_grid = np.full(
        (height, width),
        background,
        dtype=np.int8,
    )

    seen_columns: set[int] = set()
    for column, rod_height, role in rods:
        assert role in (_TRIGGER, _PASSIVE)
        assert column not in seen_columns
        assert 2 <= rod_height <= height
        assert 0 <= column and column + rod_height <= width
        seen_columns.add(column)

        color = trigger if role == _TRIGGER else passive
        input_grid[height - rod_height :, column] = color

    output_grid = _transform(
        input_grid,
        background,
        trigger,
        passive,
    )
    return input_grid, output_grid


def _make_anchor_pairs(
    rng: np.random.Generator,
    parameters: dict,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[int, ...]]:
    """A near miss and a recursive hit, with the first rod's reach taught directly."""
    mixed = parameters["mixed"]
    short_height = int(rng.integers(2, 5)) if mixed else 3
    hit_height = short_height + 1 if mixed else 3
    third_height = int(rng.integers(2, 6)) if mixed else 3
    second_height = int(rng.choice((short_height, hit_height, third_height)))
    palette = tuple(sorted({short_height, hit_height, second_height, third_height}))

    start = int(rng.integers(0, 6))
    lead = []
    if start >= 2 and rng.random() < 0.45:
        choices = [height for height in palette if height <= start]
        if choices:
            lead = [(0, int(rng.choice(choices)), _PASSIVE)]
    fourth = rng.random() < 0.45
    tail = rng.random() < 0.35
    pairs = []
    for contact in (False, True):
        first_height = hit_height if contact else short_height
        second_column = start + (hit_height if mixed or contact else short_height + 1)
        third_column = second_column + second_height
        rods = lead + [
            (start, first_height, _TRIGGER),
            (second_column, second_height, _PASSIVE),
            (third_column, third_height, _PASSIVE),
        ]
        end = third_column + third_height
        if fourth:
            fourth_height = int(rng.choice(palette))
            rods.append((end, fourth_height, _PASSIVE))
            end += fourth_height
        if tail:
            extra_height = int(rng.choice(palette))
            extra_column = end + int(rng.integers(1, 4))
            if extra_column + extra_height <= 27:
                extra_role = _TRIGGER if rng.random() < 0.65 else _PASSIVE
                rods.append((extra_column, extra_height, extra_role))
                end = extra_column + extra_height
        width = end + int(rng.integers(0, min(5, 31 - end)))
        height = max(rod_height for _, rod_height, _ in rods) + int(rng.integers(0, 5))
        pairs.append(_make_pair_from_rods(height, width, rods, parameters))

    if pairs[0][0].shape == pairs[1][0].shape:
        inp, out = pairs[1]
        if inp.shape[1] < 30:
            pairs[1] = (
                np.pad(inp, ((0, 0), (0, 1)), constant_values=parameters["background"]),
                np.pad(out, ((0, 0), (0, 1)), constant_values=parameters["background"]),
            )
        else:
            pairs[1] = (
                np.pad(inp, ((1, 0), (0, 0)), constant_values=parameters["background"]),
                np.pad(out, ((1, 0), (0, 0)), constant_values=parameters["background"]),
            )
    rng.shuffle(pairs)
    return pairs, palette


def _sample_general_pair(
    rng: np.random.Generator,
    parameters: dict,
    palette: tuple[int, ...],
    regime: str,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(400):
        if regime == "source_train":
            width = int(rng.integers(8, 27))
            ceiling = 7
        else:
            width = int(rng.integers(8, 19)) if rng.random() < 0.32 else int(rng.integers(19, 31))
            ceiling = 9
        max_count = min(ceiling, width // min(palette))
        if max_count < 2:
            continue
        count = int(rng.integers(2, max_count + 1))
        start = int(rng.integers(0, min(5, width - 1)))
        columns = [start]
        heights = []
        for index in range(count - 1):
            # Some spacings are chosen before heights. At the same distance,
            # a short rod misses while a taller one can make exact contact.
            if rng.random() < 0.38:
                distance = int(rng.integers(2, 8))
                eligible = [h for h in palette if h <= distance]
                if not eligible:
                    break
                height = int(rng.choice(eligible))
                if distance in eligible and rng.random() < 0.55:
                    height = distance
            else:
                height = int(rng.choice(palette))
                gap = int(rng.choice((0, 0, 0, 1, 1, 2, 3)))
                distance = height + gap
            heights.append(height)
            columns.append(columns[-1] + distance)
        if len(columns) != count:
            continue
        heights.append(int(rng.choice(palette)))
        if columns[-1] + heights[-1] > width:
            continue

        roles = [_TRIGGER if rng.random() < 0.34 else _PASSIVE for _ in columns]
        if all(role == _PASSIVE for role in roles):
            roles[int(rng.integers(0, count))] = _TRIGGER
        if regime == "source_train" and all(role == _TRIGGER for role in roles):
            roles[int(rng.integers(0, count))] = _PASSIVE
        if regime == "target_test" and count >= 3 and rng.random() < 0.4:
            candidates = [
                index for index in range(count - 2)
                if columns[index + 1] == columns[index] + heights[index]
                and columns[index + 2] == columns[index + 1] + heights[index + 1]
            ]
            if candidates:
                index = int(rng.choice(candidates))
                roles[index:index + 3] = [_TRIGGER, _PASSIVE, _TRIGGER]
        rods = [(c, h, role) for c, h, role in zip(columns, heights, roles)]
        headroom = int(rng.integers(0, 7 if regime == "target_test" else 5))
        height = max(heights) + headroom
        return _make_pair_from_rods(height, width, rods, parameters)
    raise AssertionError("failed to sample a valid pair")


def _sample_long_test_pair(
    rng: np.random.Generator,
    parameters: dict,
    palette: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(500):
        maximum = min(9, 30 // min(palette))
        count = int(rng.integers(5, maximum + 1))
        chain_length = int(rng.integers(4, min(6, count) + 1))
        before = int(rng.integers(0, count - chain_length + 1))
        heights = [int(rng.choice(palette)) for _ in range(count)]
        columns = [int(rng.integers(0, 4))]
        for index in range(count - 1):
            gap = 0 if before <= index < before + chain_length - 1 else int(rng.integers(1, 4))
            columns.append(columns[-1] + heights[index] + gap)
        end = columns[-1] + heights[-1]
        if end > 30:
            continue
        roles = [_TRIGGER if rng.random() < 0.38 else _PASSIVE for _ in range(count)]
        roles[before:before + chain_length] = [_TRIGGER] + [_PASSIVE] * (chain_length - 1)
        if rng.random() < 0.42:
            roles[before + chain_length - 1] = _TRIGGER
        rods = [(c, h, role) for c, h, role in zip(columns, heights, roles)]
        width = end + int(rng.integers(0, min(5, 31 - end)))
        height = max(heights) + int(rng.integers(0, 7))
        return _make_pair_from_rods(height, width, rods, parameters)
    raise AssertionError("failed to sample a long test chain")


def _infer_puzzle_parameters(train_pairs: list[dict]) -> dict:
    colors = set()
    for pair in train_pairs:
        colors.update(int(v) for v in np.unique(pair["input"]))
        colors.update(int(v) for v in np.unique(pair["output"]))
    assert len(colors) == 3
    candidates = []
    for background in colors:
        for trigger in colors - {background}:
            passive = next(iter(colors - {background, trigger}))
            try:
                if all(np.array_equal(
                    _transform(pair["input"], background, trigger, passive),
                    pair["output"],
                ) for pair in train_pairs):
                    candidates.append({"background": background, "trigger": trigger, "passive": passive})
            except AssertionError:
                pass
    assert len(candidates) == 1
    return candidates[0]


def _has_height_contact_contrast(train_pairs: list[dict], parameters: dict) -> bool:
    rods_by_pair = [
        _parse_rods(pair["input"], parameters["background"],
                    parameters["trigger"], parameters["passive"])
        for pair in train_pairs
    ]
    for short in rods_by_pair:
        for tall in rods_by_pair:
            if short is tall:
                continue
            for column, height, role in short:
                if role != _TRIGGER:
                    continue
                next_passive = next((c for c, _, r in short
                                     if c > column and r == _PASSIVE), None)
                if next_passive != column + height + 1:
                    continue
                if (column, height + 1, _TRIGGER) in tall and any(
                    c == next_passive and r == _PASSIVE for c, _, r in tall
                ):
                    return True
    return False


def _append_pair(puzzle: dict, split: str, pair: tuple[np.ndarray, np.ndarray], seen: set) -> bool:
    inp, out = pair
    signature = _grid_signature(inp)
    if signature in seen or _is_source_input(inp):
        return False
    puzzle[split].append({"input": inp, "output": out})
    seen.add(signature)
    return True


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    parameters["mixed"] = bool(rng.random() < 0.67)
    anchors, palette = _make_anchor_pairs(rng, parameters)
    puzzle = {"train": [], "test": []}
    seen = set()
    for pair in anchors:
        assert _append_pair(puzzle, "train", pair, seen)
    for split, count in (("train", train_count), ("test", test_count)):
        for _ in range(count - len(puzzle[split])):
            for _ in range(600):
                if split == "test" and rng.random() < 0.38:
                    pair = _sample_long_test_pair(rng, parameters, palette)
                else:
                    pair = _sample_general_pair(rng, parameters, palette,
                                                "source_train" if split == "train" else "target_test")
                if _append_pair(puzzle, split, pair, seen):
                    break
            else:
                raise AssertionError("failed to find a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= size <= 30 for size in grid.shape)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
    parameters = _infer_puzzle_parameters(puzzle["train"])
    background, trigger, passive = (parameters[k] for k in ("background", "trigger", "passive"))
    seen = set()
    train_heights = set()
    test_heights = set()
    train_shapes = set()
    active_passive = inactive_passive = chain = free_trigger = False
    exact = miss = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= size <= 30 for size in grid.shape)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
                assert set(int(v) for v in np.unique(grid)) <= {background, trigger, passive}
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            signature = _grid_signature(inp)
            assert signature not in seen and not _is_source_input(inp)
            seen.add(signature)
            rods = _parse_rods(inp, background, trigger, passive)
            assert np.array_equal(_transform(inp, background, trigger, passive), out)
            heights = {h for _, h, _ in rods}
            if split == "train":
                train_heights.update(heights)
                train_shapes.add(inp.shape)
                active = _active_indices(rods)
                by_column = {c: i for i, (c, _, _) in enumerate(rods)}
                for i, (column, rod_height, role) in enumerate(rods):
                    predecessor = any(j in active and c + h == column
                                      for j, (c, h, _) in enumerate(rods[:i]))
                    if role == _TRIGGER and not predecessor:
                        free_trigger = True
                    if role == _PASSIVE:
                        active_passive |= i in active
                        inactive_passive |= i not in active
                    if i in active and role == _PASSIVE and by_column.get(column + rod_height) is not None:
                        chain = True
                    if i in active and i + 1 < len(rods):
                        next_column = rods[i + 1][0]
                        exact |= next_column == column + rod_height
                        miss |= next_column == column + rod_height + 1 and rods[i + 1][2] == _PASSIVE and i + 1 not in active
            else:
                test_heights.update(heights)
    assert len(train_shapes) >= 2
    assert active_passive and inactive_passive and chain and free_trigger and exact and miss
    assert test_heights <= train_heights
    if len(train_heights) > 1:
        assert _has_height_contact_contrast(puzzle["train"], parameters)


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(tuple(_pair_signature(p["input"], p["output"]) for p in puzzle[split])
                 for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len({_puzzle_signature(p) for p in puzzles}) == 42
    shapes = set()
    train_layouts = set()
    test_layouts = set()
    palettes = set()
    test_widths = set()
    test_rod_counts = set()
    test_trigger_counts = set()
    leading_passive = wholly_active = active_passive = inactive_passive = 0
    counterfactual_puzzles = 0
    mixed = simple = later_chain = near_miss = passive_into_trigger = False
    long_chain = False
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        params = _infer_puzzle_parameters(puzzle["train"])
        counterfactual_puzzles += _has_height_contact_contrast(puzzle["train"], params)
        palettes.add(tuple(params[k] for k in ("background", "trigger", "passive")))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                shapes.add((split, inp.shape))
                rods = _parse_rods(inp, params["background"], params["trigger"], params["passive"])
                layout = (inp.shape, tuple(rods))
                (train_layouts if split == "train" else test_layouts).add(layout)
                if split == "train":
                    mixed |= len({h for _, h, _ in rods}) > 1
                    simple |= {h for _, h, _ in rods} == {3}
                else:
                    active = _active_indices(rods)
                    test_widths.add(inp.shape[1])
                    test_rod_counts.add(len(rods))
                    test_trigger_counts.add(sum(role == _TRIGGER for _, _, role in rods))
                    leading_passive += rods[0][2] == _PASSIVE
                    wholly_active += len(active) == len(rods)
                    active_passive += any(i in active and role == _PASSIVE
                                          for i, (_, _, role) in enumerate(rods))
                    inactive_passive += any(i not in active and role == _PASSIVE
                                            for i, (_, _, role) in enumerate(rods))
                    long_chain |= _longest_trigger_chain(rods) >= 5
                    for i in range(len(rods) - 1):
                        c, h, role = rods[i]
                        next_c, _, next_role = rods[i + 1]
                        if i in active and next_c == c + h + 1 and next_role == _PASSIVE:
                            near_miss = True
                        if i in active and role == _PASSIVE and next_c == c + h and next_role == _TRIGGER:
                            passive_into_trigger = True
                    later_chain |= any(i > 0 and rods[i][2] == _TRIGGER
                                       and sum(1 for j in range(i, len(rods))
                                               if all(rods[k + 1][0] == rods[k][0] + rods[k][1]
                                                      for k in range(i, j))) >= 4
                                       for i in range(len(rods)))
    assert len(shapes) >= 25 and len(train_layouts) >= 80 and len(test_layouts) >= 50
    assert len(palettes) >= 20
    assert mixed and simple and later_chain and long_chain and near_miss and passive_into_trigger
    assert counterfactual_puzzles >= 15
    assert len(test_widths) >= 12 and len(test_rod_counts) >= 5
    assert len(test_trigger_counts) >= 3
    assert leading_passive >= 8 and wholly_active >= 8
    assert active_passive >= 8 and inactive_passive >= 8
    assert _puzzle_signature(generate_puzzle(123, 6, 3)) == _puzzle_signature(generate_puzzle(123, 6, 3))
    assert _puzzle_signature(generate_puzzle(123, 6, 3)) != _puzzle_signature(generate_puzzle(124, 6, 3))
    for seed, nt, ns in ((9000, 2, 1), (9001, 10, 1), (9002, 10, 4),
                         (9100, None, 1), (9101, None, 4), (9102, 2, None), (9103, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        assert nt is None or len(puzzle["train"]) == nt
        assert ns is None or len(puzzle["test"]) == ns
        validate_generated_puzzle(puzzle)
    for nt, ns in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count request accepted")
