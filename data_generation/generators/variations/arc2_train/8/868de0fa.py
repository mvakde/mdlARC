import numpy as np
from itertools import permutations


_LEGAL_COUNT_PAIRS = tuple(
    (num_train, num_test)
    for num_test in range(1, 5)
    for num_train in range(max(2, num_test + 1), 11)
)


def _source_text(*rows: str) -> str:
    return "/".join(rows)


# Rejecting an exact source input is stronger than merely rejecting an exact
# source input/output pair.
_SOURCE_INPUT_TEXTS = frozenset(
    {
        _source_text(
            "00000000000",
            "01110000000",
            "01010111100",
            "01110100100",
            "00000100100",
            "00000111100",
            "11111000000",
            "10001000000",
            "10001000000",
            "10001000000",
            "11111000000",
        ),
        _source_text(
            "1110111111",
            "1010100001",
            "1110100001",
            "0000100001",
            "0000100001",
            "0000111111",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
        _source_text(
            "11111000000000000000",
            "10001011110000000000",
            "10001010010000000000",
            "10001010010011111100",
            "11111011110010000100",
            "00000000000010000100",
            "00000000000010000100",
            "00000000000010000100",
            "00000000000011111100",
            "00111111100000000000",
            "00100000100000000000",
            "00100000100000000000",
            "00100000100011111111",
            "00100000100010000001",
            "00100000100010000001",
            "00111111100010000001",
            "00000000000010000001",
            "00000000000010000001",
            "00000000000010000001",
            "00000000000011111111",
        ),
        _source_text(
            "000000000000000",
            "011111110000000",
            "010000010000000",
            "010000010000000",
            "010000010000000",
            "010000010000000",
            "010000010000000",
            "011111110000000",
            "000000000000000",
            "000000111111000",
            "000000100001000",
            "000000100001000",
            "000000100001000",
            "000000100001000",
            "000000111111000",
        ),
        _source_text(
            "1111000000",
            "1001000000",
            "1001001110",
            "1111001010",
            "0000001110",
            "1111100000",
            "1000100000",
            "1000100000",
            "1000100000",
            "1111100000",
        ),
        _source_text(
            "00000000000011111000",
            "00000000000010001000",
            "01111111100010001000",
            "01000000100010001000",
            "01000000100011111000",
            "01000000100000000000",
            "01000000100000000000",
            "01000000101111111111",
            "01000000101000000001",
            "01111111101000000001",
            "00000000001000000001",
            "01111111001000000001",
            "01000001001000000001",
            "01000001001000000001",
            "01000001001000000001",
            "01000001001000000001",
            "01000001001111111111",
            "01111111000000000000",
            "00000000000000000000",
            "00000000000000000000",
        ),
    }
)


def _checked_integer(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, inclusive"
        )
    return result


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    result = int(seed)
    if result < 0:
        raise ValueError("seed must be nonnegative")
    return result


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _checked_integer("num_train", num_train, 2, 10)
    test_count = _checked_integer("num_test", num_test, 1, 4)

    if train_count is None and test_count is None:
        return _LEGAL_COUNT_PAIRS[seed % len(_LEGAL_COUNT_PAIRS)]

    if train_count is None:
        assert test_count is not None
        choices = tuple(range(max(2, test_count + 1), 11))
        if not choices:
            raise ValueError("no legal num_train value exists")
        train_count = choices[seed % len(choices)]

    elif test_count is None:
        choices = tuple(range(1, min(4, train_count - 1) + 1))
        if not choices:
            raise ValueError("no legal num_test value exists")
        test_count = choices[seed % len(choices)]

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    # Only the two output meanings persist across pairs.  Background and
    # border colors describe the current input, so they may change per pair.
    odd_fill, even_fill = rng.choice(10, size=2, replace=False)
    return {"odd_fill": int(odd_fill), "even_fill": int(even_fill)}


def _grid_text(grid: np.ndarray) -> str:
    return "/".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _grid_key(grid: np.ndarray) -> tuple:
    compact = np.asarray(grid, dtype=np.int8)
    return tuple(int(x) for x in compact.shape), compact.tobytes()


def _render_pair(
    height: int,
    width: int,
    rectangles: list[tuple[int, int, int]],
    puzzle_parameters: dict,
    background: int,
    frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    odd_fill = puzzle_parameters["odd_fill"]
    even_fill = puzzle_parameters["even_fill"]

    input_grid = np.full(
        (height, width),
        background,
        dtype=np.int8,
    )
    output_grid = input_grid.copy()

    for row, col, side in rectangles:
        if not (
            0 <= row
            and 0 <= col
            and side >= 3
            and row + side <= height
            and col + side <= width
        ):
            raise ValueError("invalid generated frame placement")

        input_grid[row, col:col + side] = frame
        input_grid[row + side - 1, col:col + side] = frame
        input_grid[row:row + side, col] = frame
        input_grid[row:row + side, col + side - 1] = frame

        fill = odd_fill if side % 2 == 1 else even_fill
        output_grid[row, col:col + side] = frame
        output_grid[row + side - 1, col:col + side] = frame
        output_grid[row:row + side, col] = frame
        output_grid[row:row + side, col + side - 1] = frame
        output_grid[
            row + 1:row + side - 1,
            col + 1:col + side - 1,
        ] = fill

    return input_grid, output_grid



def _separated(first: tuple[int, int, int], second: tuple[int, int, int]) -> bool:
    r1, c1, s1 = first
    r2, c2, s2 = second
    return (r1 + s1 < r2 or r2 + s2 < r1
            or c1 + s1 < c2 or c2 + s2 < c1)


def _place(
    rng: np.random.Generator,
    sides: list[int],
    height: int,
    width: int,
    restarts: int = 90,
    min_margin: int = 0,
) -> list[tuple[int, int, int]] | None:
    # Randomized complete candidate lists make each attempt bounded and give
    # staggered, row, column, corner, and loose constellations naturally.
    order = sorted(sides, reverse=True)
    for _ in range(restarts):
        placed = []
        for side in order:
            usable_height = height - side + 1 - 2 * min_margin
            usable_width = width - side + 1 - 2 * min_margin
            if usable_height <= 0 or usable_width <= 0:
                break
            possibilities = usable_height * usable_width
            for flat in rng.permutation(possibilities):
                row, col = divmod(int(flat), usable_width)
                row += min_margin
                col += min_margin
                candidate = (row, col, side)
                if all(_separated(candidate, old) for old in placed):
                    placed.append(candidate)
                    break
            else:
                break
        if len(placed) == len(sides):
            rng.shuffle(placed)
            return placed
    return None


def _fallback_place(sides: list[int]) -> list[tuple[int, int, int]] | None:
    # At full canvas size, a deterministic shelf search resolves rare greedy
    # placement misses.  This is reached only after varied placement attempts.
    for ordering in permutations(sides):
        for cuts in range(1 << (len(sides) - 1)):
            rows = []
            current = []
            for i, side in enumerate(ordering):
                current.append(side)
                if i == len(sides) - 1 or cuts & (1 << i):
                    rows.append(current)
                    current = []
            if any(sum(row) + len(row) - 1 > 30 for row in rows):
                continue
            if sum(max(row) for row in rows) + len(rows) - 1 > 30:
                continue
            frames = []
            top = 0
            for row in rows:
                left = 0
                for side in row:
                    frames.append((top, left, side))
                    left += side + 1
                top += max(row) + 1
            return frames
    return None


def _dimensions(
    rng: np.random.Generator,
    sides: list[int],
) -> tuple[int, int]:
    area = sum((side + 1) ** 2 for side in sides)
    target = area * float(rng.uniform(1.08, 1.75))
    aspect = float(rng.uniform(0.56, 1.78))
    height = max(max(sides), min(30, int(round((target * aspect) ** .5))))
    width = max(max(sides), min(30, int(round(target / height))))
    height = min(30, height + int(rng.integers(0, 4)))
    width = min(30, width + int(rng.integers(0, 4)))
    return height, width


def _pair_colors(
    rng: np.random.Generator,
    puzzle_parameters: dict,
) -> tuple[int, int]:
    available = [
        color for color in range(10)
        if color not in (puzzle_parameters["odd_fill"],
                         puzzle_parameters["even_fill"])
    ]
    background, frame = rng.choice(available, size=2, replace=False)
    return int(background), int(frame)


def _aligned_layout(
    rng: np.random.Generator,
    sides: list[int],
) -> tuple[int, int, list[tuple[int, int, int]]]:
    ordered = list(sides)
    rng.shuffle(ordered)
    extra_gap = min(4, max(1, (28 - sum(ordered)) // (len(ordered) - 1)))
    gaps = [int(x) for x in rng.integers(1, extra_gap + 1,
                                        size=len(ordered) - 1)]
    used = sum(ordered) + sum(gaps)
    along = used + int(rng.integers(0, min(5, 30 - used) + 1))
    across = max(ordered) + int(
        rng.integers(0, min(8, 30 - max(ordered)) + 1)
    )
    cursor = int(rng.integers(0, along - used + 1))
    anchor = int(rng.integers(0, across))
    row_layout = []
    for index, side in enumerate(ordered):
        low = max(0, anchor - side + 1)
        high = min(anchor, across - side)
        perpendicular = int(rng.integers(low, high + 1))
        row_layout.append((perpendicular, cursor, side))
        cursor += side
        if index < len(gaps):
            cursor += gaps[index]
    if bool(rng.integers(0, 2)):
        return along, across, [
            (col, row, side) for row, col, side in row_layout
        ]
    return across, along, row_layout


def _construct_pair(
    rng: np.random.Generator,
    sides: list[int],
    puzzle_parameters: dict,
    dense20: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if (not dense20 and 3 <= len(sides) <= 4
            and sum(sides) + len(sides) - 1 <= 29
            and rng.random() < .35):
        height, width, frames = _aligned_layout(rng, sides)
        background, frame = _pair_colors(rng, puzzle_parameters)
        return _render_pair(
            height, width, frames, puzzle_parameters, background, frame
        )
    if dense20:
        height = width = 20
    else:
        height, width = _dimensions(rng, sides)
    desired_margin = (
        int(rng.integers(3, 5))
        if not dense20 and len(sides) <= 3 and rng.random() < .28
        else 0
    )
    for _ in range(50):
        frames = _place(
            rng, sides, height, width, min_margin=desired_margin
        )
        if frames is not None:
            background, frame = _pair_colors(rng, puzzle_parameters)
            return _render_pair(
                height, width, frames, puzzle_parameters, background, frame
            )
        if height == width == 30:
            if desired_margin:
                desired_margin = 0
                continue
            frames = _fallback_place(sides)
            if frames is not None:
                background, frame = _pair_colors(rng, puzzle_parameters)
                return _render_pair(
                    30, 30, frames, puzzle_parameters, background, frame
                )
            break
        if height <= width and height < 30 or width == 30:
            height = min(30, height + 1)
        else:
            width = min(30, width + 1)
    raise ValueError("could not place square frames")


def _training_schedule(
    rng: np.random.Generator,
    num_train: int,
) -> list[list[int]]:
    counts = [int(x) for x in rng.integers(2, 6, size=num_train)]
    slots = [(i, j) for i, count in enumerate(counts) for j in range(count)]
    odd_count = int(rng.integers(2, 4))
    even_count = int(rng.integers(2, 4))
    while odd_count + even_count > len(slots):
        if odd_count > even_count:
            odd_count -= 1
        else:
            even_count -= 1
    odds = [int(x) for x in rng.choice(
        [3, 5, 7, 9], size=odd_count, replace=False
    )]
    evens = [int(x) for x in rng.choice(
        [4, 6, 8, 10], size=even_count, replace=False
    )]
    pool = odds + evens
    schedule = [
        [int(x) for x in rng.choice(
            [side for side in pool if side <= (7 if count == 5
                                               else 8 if count == 4
                                               else 10)]
            or [3, 4, 5, 6, 7],
            size=count
        )]
        for count in counts
    ]
    rng.shuffle(slots)
    for size, (i, j) in zip(odds + evens, slots):
        schedule[i][j] = size
    return schedule


def _test_sides(
    rng: np.random.Generator,
    trained: set[int],
) -> tuple[list[int], bool]:
    # A dense 20-square case is one regime among loose and irregular cases.
    if rng.random() < .14:
        choices = [5, 6, 7, 8, 9]
        return [int(x) for x in rng.choice(choices, size=4, replace=False)], True

    count = int(rng.integers(1, 6))
    mode = int(rng.integers(0, 4))
    size_cap = 7 if count == 5 else 8 if count == 4 else 10
    familiar = sorted(side for side in trained if side <= size_cap)
    odd_new = [side for side in (9, 11) if side not in trained]
    even_new = [side for side in (10, 12) if side not in trained]
    sides = [int(x) for x in rng.choice(familiar, size=count)]
    if mode in (1, 3):
        sides[0] = int(rng.choice(odd_new))
    if mode in (2, 3):
        sides[-1] = int(rng.choice(even_new))
    # Single-frame tests can show either novel parity.  For more frames,
    # replacement at opposite ends permits both at once.
    rng.shuffle(sides)
    return sides, False


def _generate_unique_pair(
    rng: np.random.Generator,
    sides: list[int],
    puzzle_parameters: dict,
    seen_inputs: set,
    dense20: bool = False,
    avoid_palette: frozenset[int] | None = None,
) -> dict:
    for _ in range(24):
        try:
            input_grid, output_grid = _construct_pair(
                rng, sides, puzzle_parameters, dense20
            )
        except ValueError:
            if dense20:
                dense20 = False
                continue
            raise
        if (avoid_palette is not None
                and frozenset(int(x) for x in np.unique(input_grid))
                == avoid_palette):
            continue
        key = _grid_key(input_grid)
        if key not in seen_inputs and _grid_text(input_grid) not in _SOURCE_INPUT_TEXTS:
            seen_inputs.add(key)
            return {"input": input_grid, "output": output_grid}
    raise ValueError("could not generate a unique pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _normalize_seed(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle_parameters = sample_puzzle_parameters(rng)
    schedule = _training_schedule(rng, train_count)
    trained = {side for row in schedule for side in row}
    pair_seeds = rng.integers(
        0, np.iinfo(np.int64).max,
        size=train_count + test_count, dtype=np.int64
    )
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for i, sides in enumerate(schedule):
        pair_rng = np.random.default_rng(int(pair_seeds[i]))
        avoid_palette = (
            frozenset(int(x) for x in np.unique(puzzle["train"][0]["input"]))
            if i == 1 else None
        )
        puzzle["train"].append(_generate_unique_pair(
            pair_rng, sides, puzzle_parameters, seen_inputs,
            avoid_palette=avoid_palette
        ))
    for i in range(test_count):
        pair_rng = np.random.default_rng(int(pair_seeds[train_count + i]))
        sides, dense20 = _test_sides(pair_rng, trained)
        puzzle["test"].append(_generate_unique_pair(
            pair_rng, sides, puzzle_parameters, seen_inputs, dense20
        ))
    validate_generated_puzzle(puzzle)
    return puzzle


def _extract_frames(
    grid: np.ndarray,
    background: int,
    frame: int,
) -> list[tuple[int, int, int]]:
    if not np.all((grid == background) | (grid == frame)):
        raise ValueError("input contains an unexpected color")

    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    components = []

    for start_row in range(height):
        for start_col in range(width):
            if grid[start_row, start_col] != frame:
                continue
            if visited[start_row, start_col]:
                continue

            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True
            component = []

            while stack:
                row, col = stack.pop()
                component.append((row, col))

                for next_row, next_col in (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                ):
                    if not (
                        0 <= next_row < height
                        and 0 <= next_col < width
                    ):
                        continue
                    if visited[next_row, next_col]:
                        continue
                    if grid[next_row, next_col] != frame:
                        continue

                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))

            components.append(component)

    frames = []
    for component in components:
        rows = [cell[0] for cell in component]
        cols = [cell[1] for cell in component]
        row_min = min(rows)
        row_max = max(rows)
        col_min = min(cols)
        col_max = max(cols)

        box_height = row_max - row_min + 1
        box_width = col_max - col_min + 1
        if box_height != box_width or box_height < 3:
            raise ValueError("structural component is not a square frame")

        side = box_height
        box = grid[
            row_min:row_max + 1,
            col_min:col_max + 1,
        ]
        border_mask = np.zeros((side, side), dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True

        if not np.all(box[border_mask] == frame):
            raise ValueError("square border is incomplete")
        if not np.all(box[~border_mask] == background):
            raise ValueError("square interior is not hollow")
        if len(component) != 4 * side - 4:
            raise ValueError("frame has extra structural cells")

        frames.append((row_min, col_min, side))

    if not frames:
        raise ValueError("input contains no frames")

    frames.sort()
    return frames


def _frames_are_separated(
    frames: list[tuple[int, int, int]],
) -> bool:
    for first_index in range(len(frames)):
        first_row, first_col, first_side = frames[first_index]
        first_bottom = first_row + first_side - 1
        first_right = first_col + first_side - 1

        for second_index in range(first_index + 1, len(frames)):
            second_row, second_col, second_side = frames[second_index]
            second_bottom = second_row + second_side - 1
            second_right = second_col + second_side - 1

            separated = (
                first_bottom + 1 < second_row
                or second_bottom + 1 < first_row
                or first_right + 1 < second_col
                or second_right + 1 < first_col
            )
            if not separated:
                return False

    return True


def _apply_inferred_rule(
    input_grid: np.ndarray,
    frames: list[tuple[int, int, int]],
    odd_fill: int,
    even_fill: int,
) -> np.ndarray:
    expected = input_grid.copy()
    for row, col, side in frames:
        fill = odd_fill if side % 2 == 1 else even_fill
        expected[
            row + 1:row + side - 1,
            col + 1:col + side - 1,
        ] = fill
    return expected



def _identify_pair(pair: dict) -> tuple[int, int, list[tuple[int, int, int]]]:
    colors = [int(x) for x in np.unique(pair["input"])]
    assert len(colors) == 2
    candidates = []
    for background, frame in ((colors[0], colors[1]),
                              (colors[1], colors[0])):
        try:
            frames = _extract_frames(pair["input"], background, frame)
            if _frames_are_separated(frames):
                candidates.append((background, frame, frames))
        except ValueError:
            pass
    assert len(candidates) == 1
    return candidates[0]


def _analyze_task_invariants(puzzle: dict) -> dict:
    train_pairs = puzzle["train"]
    test_pairs = puzzle["test"]
    all_pairs = train_pairs + test_pairs
    roles_and_frames = [_identify_pair(pair) for pair in all_pairs]
    assert len({
        (background, frame)
        for background, frame, _ in roles_and_frames[:len(train_pairs)]
    }) >= 2
    observations = {0: set(), 1: set()}
    demonstrated = {0: set(), 1: set()}

    for pair, (_, _, frames) in zip(train_pairs, roles_and_frames):
        for row, col, side in frames:
            values = np.unique(pair["output"][
                row + 1:row + side - 1, col + 1:col + side - 1
            ])
            assert len(values) == 1
            observations[side % 2].add(int(values[0]))
            demonstrated[side % 2].add(side)

    assert all(len(observations[parity]) == 1 for parity in (0, 1))
    assert all(len(demonstrated[parity]) >= 2 for parity in (0, 1))
    even_fill = next(iter(observations[0]))
    odd_fill = next(iter(observations[1]))
    assert odd_fill != even_fill
    train_frames = []
    test_frames = []

    for i, (pair, (background, frame, frames)) in enumerate(
        zip(all_pairs, roles_and_frames)
    ):
        assert pair["input"].shape == pair["output"].shape
        assert len({background, frame, odd_fill, even_fill}) == 4
        expected = _apply_inferred_rule(
            pair["input"], frames, odd_fill, even_fill
        )
        assert np.array_equal(pair["output"], expected)
        assert not np.array_equal(pair["input"], pair["output"])
        sides = [side for _, _, side in frames]
        if i < len(train_pairs):
            assert 2 <= len(frames) <= 5
            assert all(3 <= side <= 10 for side in sides)
            train_frames.append(frames)
        else:
            assert 1 <= len(frames) <= 5
            assert all(3 <= side <= 12 for side in sides)
            test_frames.append(frames)

    return {
        "odd_fill": odd_fill,
        "even_fill": even_fill,
        "train_frames": train_frames,
        "test_frames": test_frames,
        "roles_and_frames": roles_and_frames,
        "demonstrated": demonstrated,
    }


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

    seen_inputs = set()
    seen_pairs = set()

    for split_name in ("train", "test"):
        for pair in puzzle[split_name]:
            assert isinstance(pair, dict)
            assert set(pair.keys()) == {"input", "output"}

            input_grid = pair["input"]
            output_grid = pair["output"]

            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                height, width = grid.shape
                assert 1 <= height <= 30
                assert 1 <= width <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))

            input_key = _grid_key(input_grid)
            output_key = _grid_key(output_grid)
            pair_key = (input_key, output_key)

            assert input_key not in seen_inputs
            assert pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

            assert _grid_text(input_grid) not in _SOURCE_INPUT_TEXTS

    _analyze_task_invariants(puzzle)


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first.keys()) != set(second.keys()):
        return False

    for split_name in ("train", "test"):
        if len(first[split_name]) != len(second[split_name]):
            return False
        for first_pair, second_pair in zip(
            first[split_name],
            second[split_name],
        ):
            if set(first_pair.keys()) != set(second_pair.keys()):
                return False
            if not np.array_equal(
                first_pair["input"],
                second_pair["input"],
            ):
                return False
            if not np.array_equal(
                first_pair["output"],
                second_pair["output"],
            ):
                return False

    return True


def _puzzle_fingerprint(puzzle: dict) -> tuple:
    result = []
    for split_name in ("train", "test"):
        for pair in puzzle[split_name]:
            result.append(
                (
                    split_name,
                    _grid_key(pair["input"]),
                    _grid_key(pair["output"]),
                )
            )
    return tuple(result)



def _predict_from_training(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    learned = {0: set(), 1: set()}
    for pair in train:
        _, _, frames = _identify_pair({"input": pair["input"].copy()})
        for row, col, side in frames:
            interior = pair["output"][
                row + 1:row + side - 1, col + 1:col + side - 1
            ]
            learned[side % 2].update(int(x) for x in np.unique(interior))
    assert all(len(learned[parity]) == 1 for parity in (0, 1))
    _, _, frames = _identify_pair({"input": test_input.copy()})
    return _apply_inferred_rule(
        test_input.copy(), frames,
        next(iter(learned[1])), next(iter(learned[0]))
    )


def self_test() -> None:
    automatic = []
    train_counts, test_counts = set(), set()
    role_pairs, train_sets, dimensions, layouts, fingerprints = (
        set(), set(), set(), set(), set()
    )
    seen_counts, size_values = set(), set()
    square_train = non_square_test = dense20 = 0
    unseen_odd = unseen_even = familiar_only = 0
    repeated = single_parity = wide_gap = edge_touch = roomy_margin = 0
    rows = columns = staggered = three_plus_one = 0
    test_rows = test_columns = 0
    within_palette_variation = within_dimension_variation = 0
    within_count_variation = 0

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        analysis = _analyze_task_invariants(puzzle)
        automatic.append(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(_puzzle_fingerprint(puzzle))
        train_sides = {side for frames in analysis["train_frames"]
                       for _, _, side in frames}
        train_sets.add(tuple(sorted(train_sides)))
        palettes = []
        shapes = []
        object_counts = []
        for pair, (background, frame, frames) in zip(
            puzzle["train"] + puzzle["test"], analysis["roles_and_frames"]
        ):
            inp, out = pair["input"], pair["output"]
            palettes.append((background, frame))
            shapes.append(inp.shape)
            object_counts.append(len(frames))
            role_pairs.add((background, frame, analysis["odd_fill"],
                            analysis["even_fill"]))
            dimensions.add(inp.shape)
            seen_counts.add(len(frames))
            sizes = [side for _, _, side in frames]
            size_values.update(sizes)
            if len(sizes) != len(set(sizes)):
                repeated += 1
            if len({side % 2 for side in sizes}) == 1:
                single_parity += 1
            row_overlap = (
                max(row for row, _, _ in frames)
                <= min(row + side - 1 for row, _, side in frames)
            )
            col_overlap = (
                max(col for _, col, _ in frames)
                <= min(col + side - 1 for _, col, side in frames)
            )
            if len(frames) >= 3:
                rows += row_overlap
                columns += col_overlap
            if len(frames) == 3 and not (row_overlap or col_overlap):
                row_pairs = col_pairs = 0
                for i, (r1, c1, s1) in enumerate(frames):
                    for r2, c2, s2 in frames[i + 1:]:
                        row_pairs += max(r1, r2) <= min(
                            r1 + s1 - 1, r2 + s2 - 1
                        )
                        col_pairs += max(c1, c2) <= min(
                            c1 + s1 - 1, c2 + s2 - 1
                        )
                staggered += row_pairs > 0 and col_pairs > 0
            if len(frames) == 4:
                for axis in (0, 1):
                    values = [(frame[axis], frame[axis] + frame[2] - 1)
                              for frame in frames]
                    if any(
                        max(values[i][0] for i in range(4) if i != skip)
                        <= min(values[i][1] for i in range(4) if i != skip)
                        for skip in range(4)
                    ):
                        three_plus_one += 1
                        break
            height, width = inp.shape
            margin = min(
                min(row, col, height - row - side, width - col - side)
                for row, col, side in frames
            )
            edge_touch += margin == 0
            roomy_margin += margin >= 3
            for i, first in enumerate(frames):
                for second in frames[i + 1:]:
                    r1, c1, s1 = first
                    r2, c2, s2 = second
                    row_gap = max(r2 - r1 - s1, r1 - r2 - s2, 0)
                    col_gap = max(c2 - c1 - s1, c1 - c2 - s2, 0)
                    wide_gap += max(row_gap, col_gap) >= 4
            changed = sum((side - 2) ** 2 for side in sizes)
            assert np.count_nonzero(inp != out) == changed
            assert np.array_equal(
                _apply_inferred_rule(inp, frames,
                                     analysis["odd_fill"],
                                     analysis["even_fill"]), out
            )
            layouts.add((inp.shape, tuple(sorted(frames))))
        within_palette_variation += len(set(palettes)) > 1
        within_dimension_variation += len(set(shapes)) > 1
        within_count_variation += len(set(object_counts)) > 1
        square_train += sum(pair["input"].shape[0] == pair["input"].shape[1]
                            for pair in puzzle["train"])
        non_square_test += sum(pair["input"].shape[0] != pair["input"].shape[1]
                               for pair in puzzle["test"])
        for pair, frames in zip(puzzle["test"], analysis["test_frames"]):
            sides = {side for _, _, side in frames}
            if len(frames) >= 3:
                test_rows += (
                    max(row for row, _, _ in frames)
                    <= min(row + side - 1 for row, _, side in frames)
                )
                test_columns += (
                    max(col for _, col, _ in frames)
                    <= min(col + side - 1 for _, col, side in frames)
                )
            unseen = sides - train_sides
            unseen_odd += any(side in (9, 11) for side in unseen)
            unseen_even += any(side in (10, 12) for side in unseen)
            familiar_only += not unseen
            dense20 += pair["input"].shape == (20, 20) and len(frames) == 4
            assert np.array_equal(
                _predict_from_training(puzzle["train"], pair["input"]),
                pair["output"]
            )

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) == 42
    assert len(role_pairs) >= 30 and len(train_sets) >= 15
    assert len(dimensions) >= 25 and len(layouts) >= 100
    assert seen_counts == {1, 2, 3, 4, 5}
    assert {3, 4, 5, 6, 7, 8, 9, 10, 11, 12} <= size_values
    assert square_train >= 3 and non_square_test >= 10
    assert unseen_odd >= 3 and unseen_even >= 3 and familiar_only >= 3
    assert dense20 >= 1 and repeated >= 5 and single_parity >= 5
    assert wide_gap >= 10 and edge_touch >= 10 and roomy_margin >= 3
    assert rows >= 5 and columns >= 5
    assert test_rows >= 1 and test_columns >= 1
    assert staggered >= 10 and three_plus_one >= 5
    assert within_palette_variation >= 35
    assert within_dimension_variation >= 35
    assert within_count_variation >= 35

    for seed, ntrain, ntest in (
        (9001, 7, 3), (1101, 2, 1), (1102, 10, 1), (1103, 10, 4),
        (1201, None, 1), (1202, None, 4),
        (1203, 2, None), (1204, 10, None)
    ):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed, ntrain, ntest))
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    assert len({_puzzle_fingerprint(generate_puzzle(seed, 6, 2))
                for seed in range(100, 112)}) >= 10
    for kwargs in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4}, {"num_train": True},
        {"num_test": False}
    ):
        try:
            generate_puzzle(1300, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request was accepted")
