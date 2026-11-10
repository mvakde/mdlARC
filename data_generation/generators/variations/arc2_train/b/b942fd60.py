import numpy as np


_SOURCE_TEXT = (
    "07000000/00000003/00000000/20800000/00000000/00000000/00007000/00000000",
    "000037000000/008000007003/000000600000/000008000800/200800000000/000007000000/000000000600/007000000000/000000700700/000600000000/030000083000/000070000000",
    "003007/000000/200030/000000/000008/003000",
    "000000/200080/000000/000000",
    "0000800000/0030000070/2000003000/0000000000/0000708006/0000000000/0700060008/0060800000/0000800000/0000007000",
    "000700/600000/000000/200080/000000/000008/000000/700000",
    "000800007003/000000000000/006000060000/200000600070/000000000006/000300080000/000000000600/007000000003/008000700600/000003000000",
)


def _fingerprint(grid):
    return grid.shape, tuple(int(value) for value in grid.flat)


def _source_key(grid):
    return (grid.shape[0], grid.shape[1], tuple(int(value) for value in grid.flat))


def _source_keys():
    keys = set()
    for text in _SOURCE_TEXT:
        rows = text.split("/")
        keys.add((len(rows), len(rows[0]), tuple(int(value) for row in rows for value in row)))
    return keys


_SOURCE_INPUTS = _source_keys()


def _counts(rng, num_train, num_test):
    for value, name, lower, upper in (
        (num_train, "num_train", 2, 10),
        (num_test, "num_test", 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lower <= value <= upper
        ):
            raise ValueError(f"{name} must be an integer from {lower} through {upper}")
    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _direction(edge):
    return ((0, 1), (0, -1), (1, 0), (-1, 0))[edge]


def _edge_seed(rng, height, width, edge):
    if edge == 0:
        return int(rng.integers(1, height - 1)), 0
    if edge == 1:
        return int(rng.integers(1, height - 1)), width - 1
    if edge == 2:
        return 0, int(rng.integers(1, width - 1))
    return height - 1, int(rng.integers(1, width - 1))


def _seed_direction(position, shape):
    row, col = position
    height, width = shape
    choices = []
    if col == 0:
        choices.append((0, 1))
    if col == width - 1:
        choices.append((0, -1))
    if row == 0:
        choices.append((1, 0))
    if row == height - 1:
        choices.append((-1, 0))
    assert len(choices) == 1
    return choices[0]


def _trace(input_grid, background, tracer):
    height, width = input_grid.shape
    seeds = np.argwhere(input_grid == tracer)
    assert seeds.shape == (1, 2)
    seed = tuple(int(value) for value in seeds[0])
    output = input_grid.copy()
    pending = [(seed, _seed_direction(seed, input_grid.shape))]
    seen = set()
    contacts = {}
    open_rays = []
    queue_index = 0
    while queue_index < len(pending):
        start, (delta_row, delta_col) = pending[queue_index]
        queue_index += 1
        state = (start, delta_row, delta_col)
        if state in seen:
            continue
        seen.add(state)
        row, col = start
        path = []
        while True:
            next_row, next_col = row + delta_row, col + delta_col
            if not (0 <= next_row < height and 0 <= next_col < width):
                open_rays.append((start, (delta_row, delta_col), path))
                break
            value = int(input_grid[next_row, next_col])
            if value not in (background, tracer):
                position = (next_row, next_col)
                contacts.setdefault(position, []).append((delta_row, delta_col))
                pivot = (row, col)
                pending.append((pivot, (-delta_col, delta_row)))
                pending.append((pivot, (delta_col, -delta_row)))
                break
            if int(output[next_row, next_col]) == background:
                output[next_row, next_col] = tracer
            path.append((next_row, next_col))
            row, col = next_row, next_col
    return output, contacts, open_rays


def _dimensions(rng, level):
    minimum = 8 if level >= 6 else 6 if level >= 4 else 4
    bands = ((minimum, max(minimum, 7)), (max(minimum, 8), 12), (13, 20))
    dimensions = []
    for _ in range(2):
        low, high = bands[int(rng.integers(0, 3))]
        dimensions.append(int(rng.integers(low, high + 1)))
    return tuple(dimensions)


def _geometry(rng, height, width, edge, requested):
    seed = _edge_seed(rng, height, width, edge)
    grid = np.zeros((height, width), dtype=np.int8)
    grid[seed] = 1
    delta_row, delta_col = _direction(edge)
    span = width - 1 if edge in (0, 1) else height - 1
    first_distance = int(rng.integers(2, span + 1))
    first = (seed[0] + first_distance * delta_row, seed[1] + first_distance * delta_col)
    grid[first] = 2
    for _ in range(requested - 1):
        _, contacts, open_rays = _trace(grid, 0, 1)
        proposals = []
        for _, _, path in open_rays:
            for distance, position in enumerate(path, 1):
                if distance < 2 or grid[position] != 0:
                    continue
                candidate = grid.copy()
                candidate[position] = 2
                _, new_contacts, _ = _trace(candidate, 0, 1)
                if len(new_contacts) == len(contacts) + 1 and all(
                    len(directions) == 1 for directions in new_contacts.values()
                ):
                    proposals.append(position)
        if not proposals:
            return None
        grid[proposals[int(rng.integers(0, len(proposals)))]] = 2
    output, contacts, _ = _trace(grid, 0, 1)
    if len(contacts) != requested or any(len(directions) != 1 for directions in contacts.values()):
        return None
    return grid, output, contacts


def _add_distractors(rng, grid, output, contacts, count):
    height, width = grid.shape
    blockers = []
    for (row, col), directions in contacts.items():
        for delta_row, delta_col in directions:
            for distance in (1, 2, 3):
                position = (row + distance * delta_row, col + distance * delta_col)
                if (0 <= position[0] < height and 0 <= position[1] < width
                    and grid[position] == 0 and output[position] == 0):
                    blockers.append(position)
    if count and blockers and rng.random() < 0.7:
        grid[blockers[int(rng.integers(0, len(blockers)))]] = 2
        count -= 1
    free = [
        (row, col) for row in range(height) for col in range(width)
        if grid[row, col] == 0 and output[row, col] == 0
    ]
    if count:
        for index in np.atleast_1d(rng.choice(len(free), size=min(count, len(free)), replace=False)):
            grid[free[int(index)]] = 2


def _pair(rng, background, tracer, palette, edge, level, cover_palette):
    for _ in range(80):
        height, width = _dimensions(rng, level)
        geometry = _geometry(rng, height, width, edge, level)
        if geometry is None:
            continue
        grid, output, contacts = geometry
        if level == 1 and rng.random() < 0.6:
            distractors = 0
        else:
            maximum = min(12, grid.size // 8)
            distractors = int(rng.integers(0, maximum + 1))
        _add_distractors(rng, grid, output, contacts, distractors)
        output, contacts, _ = _trace(grid, 0, 1)
        if len(contacts) != level or any(len(directions) != 1 for directions in contacts.values()):
            continue
        positions = [tuple(int(value) for value in position) for position in np.argwhere(grid == 2)]
        if cover_palette:
            used = palette
        else:
            size = int(rng.integers(1, len(palette) + 1))
            used = tuple(int(value) for value in rng.choice(palette, size=size, replace=False))
        assignments = {
            position: int(used[int(rng.integers(0, len(used)))])
            for position in positions
        }
        if cover_palette:
            selected = rng.choice(len(contacts), size=len(palette), replace=False)
            active = list(contacts)
            for color, index in zip(palette, selected):
                assignments[active[int(index)]] = int(color)
        colored = np.full(grid.shape, background, dtype=np.int8)
        colored[grid == 1] = tracer
        result = np.full(grid.shape, background, dtype=np.int8)
        result[output == 1] = tracer
        for position, color in assignments.items():
            colored[position] = color
            result[position] = color
        if _source_key(colored) not in _SOURCE_INPUTS:
            return {"input": colored, "output": result}
    raise ValueError("could not construct a ray network")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _counts(rng, num_train, num_test)
    background = int(rng.integers(0, 10))
    tracer = int(rng.choice([color for color in range(10) if color != background]))
    palette_size = int(rng.integers(1, 6))
    palette = tuple(int(color) for color in rng.choice(
        [color for color in range(10) if color not in (background, tracer)],
        size=palette_size, replace=False,
    ))
    orientation_count = int(rng.integers(1, min(3, train_count) + 1))
    orientations = tuple(int(value) for value in rng.choice(4, size=orientation_count, replace=False))
    anchor_index = int(rng.integers(0, train_count))
    train_levels = [int(rng.integers(1, 9)) for _ in range(train_count)]
    train_levels[anchor_index] = max(4, palette_size, train_levels[anchor_index])
    train_edges = list(orientations)
    train_edges.extend(
        int(orientations[int(rng.integers(0, len(orientations)))])
        for _ in range(train_count - len(orientations))
    )
    rng.shuffle(train_edges)
    train, test, seen = [], [], set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            level = train_levels[index] if split == "train" else int(rng.integers(1, 10))
            edge = train_edges[index] if split == "train" else int(orientations[int(rng.integers(0, len(orientations)))])
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                pair = _pair(pair_rng, background, tracer, palette, edge, level, split == "train" and index == anchor_index)
                signature = _fingerprint(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    (train if split == "train" else test).append(pair)
                    break
            else:
                raise ValueError("could not create a distinct pair")
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _roles(pairs):
    changes = set()
    for pair in pairs:
        input_grid, output_grid = pair["input"], pair["output"]
        changes.update(
            (int(old), int(new))
            for old, new in zip(
                input_grid[input_grid != output_grid],
                output_grid[input_grid != output_grid],
            )
        )
    assert len(changes) == 1
    return next(iter(changes))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    pairs = puzzle["train"] + puzzle["test"]
    for pair in pairs:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.all((0 <= grid) & (grid <= 9))
        assert pair["input"].shape == pair["output"].shape
    background, tracer = _roles(pairs)
    inputs, training_orientations, test_orientations = set(), set(), set()
    training_contacts, test_colors = set(), set()
    recursive_evidence = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            input_grid, output_grid = pair["input"], pair["output"]
            signature = _fingerprint(input_grid)
            assert signature not in inputs
            assert _source_key(input_grid) not in _SOURCE_INPUTS
            inputs.add(signature)
            assert np.count_nonzero(input_grid == tracer) == 1
            assert np.count_nonzero(input_grid == background) > input_grid.size // 2
            marker_mask = (input_grid != background) & (input_grid != tracer)
            assert np.count_nonzero(marker_mask) >= 1
            assert np.array_equal(input_grid[marker_mask], output_grid[marker_mask])
            assert np.any(input_grid != output_grid)
            expected, contacts, _ = _trace(input_grid, background, tracer)
            assert np.array_equal(expected, output_grid)
            assert contacts and all(len(directions) == 1 for directions in contacts.values())
            seed = tuple(int(value) for value in np.argwhere(input_grid == tracer)[0])
            orientation = _seed_direction(seed, input_grid.shape)
            delta_row, delta_col = orientation
            row, col = seed
            first = None
            while 0 <= row + delta_row < input_grid.shape[0] and 0 <= col + delta_col < input_grid.shape[1]:
                row += delta_row
                col += delta_col
                if input_grid[row, col] not in (background, tracer):
                    first = (row, col)
                    break
            assert first in contacts
            colors = {int(input_grid[position]) for position in contacts}
            if split == "train":
                training_orientations.add(orientation)
                training_contacts.update(colors)
                recursive_evidence |= len(contacts) >= 4
            else:
                test_orientations.add(orientation)
                test_colors.update(int(value) for value in np.unique(input_grid[marker_mask]))
    assert recursive_evidence
    assert test_orientations <= training_orientations
    assert test_colors <= training_contacts


def _equal(first, second):
    return all(
        len(first[split]) == len(second[split]) and all(
            np.array_equal(first_pair["input"], second_pair["input"])
            and np.array_equal(first_pair["output"], second_pair["output"])
            for first_pair, second_pair in zip(first[split], second[split])
        ) for split in ("train", "test")
    )


def _predict_from_training(train, test_input):
    background, tracer = _roles(train)
    height, width = test_input.shape
    seed = tuple(int(value) for value in np.argwhere(test_input == tracer)[0])
    pending = [(seed, _seed_direction(seed, test_input.shape))]
    seen = set()
    visits = np.zeros(test_input.shape, dtype=np.int16)
    output = test_input.copy()
    index = 0
    while index < len(pending):
        start, (delta_row, delta_col) = pending[index]
        index += 1
        state = (start, delta_row, delta_col)
        if state in seen:
            continue
        seen.add(state)
        row, col = start
        while 0 <= row + delta_row < height and 0 <= col + delta_col < width:
            next_row, next_col = row + delta_row, col + delta_col
            value = int(test_input[next_row, next_col])
            if value not in (background, tracer):
                pending.extend((
                    ((row, col), (-delta_col, delta_row)),
                    ((row, col), (delta_col, -delta_row)),
                ))
                break
            output[next_row, next_col] = tracer
            visits[next_row, next_col] += 1
            row, col = next_row, next_col
    return output, visits


def self_test() -> None:
    assert _equal(generate_puzzle(1234), generate_puzzle(1234))
    assert _equal(generate_puzzle(1234, 6, 3), generate_puzzle(1234, 6, 3))
    for index, (train_count, test_count) in enumerate(
        ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))
    ):
        puzzle = generate_puzzle(9000 + index, train_count, test_count)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for index, arguments in enumerate((
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
    )):
        try:
            generate_puzzle(10000 + index, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair count accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    assert not _equal(puzzles[0], puzzles[1])
    layouts, shapes, roles, contact_counts, marker_counts, orientations, palette_sizes = (
        set(), set(), set(), set(), set(), set(), set()
    )
    minimal = blockers = short_rays = near_edges = 0
    test_simple = test_dense = high_inactive = 0
    squares = tall = wide = crossings = merges = 0
    collinear_blockers = one_sided_forks = two_sided_forks = 0
    palette_variation = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        background, tracer = _roles(puzzle["train"] + puzzle["test"])
        roles.add((background, tracer))
        palettes = set()
        for pair in puzzle["train"] + puzzle["test"]:
            grid = pair["input"]
            predicted, visits = _predict_from_training(puzzle["train"], grid)
            assert np.array_equal(predicted, pair["output"])
            shapes.add(grid.shape)
            squares += grid.shape[0] == grid.shape[1]
            tall += grid.shape[0] > grid.shape[1]
            wide += grid.shape[0] < grid.shape[1]
            normalized = np.full(grid.shape, 2, dtype=np.int8)
            normalized[grid == background] = 0
            normalized[grid == tracer] = 1
            pair_palette = tuple(sorted(int(color) for color in np.unique(grid[normalized == 2])))
            palettes.add(pair_palette)
            palette_sizes.add(len(pair_palette))
            layouts.add(_fingerprint(normalized))
            count = int(np.count_nonzero(normalized == 2))
            marker_counts.add(count)
            _, contacts, open_rays = _trace(grid, background, tracer)
            contact_counts.add(len(contacts))
            minimal += count == 1
            blockers += count > len(contacts)
            high_inactive += count - len(contacts) >= 10
            merges += np.any(visits > 1)
            inactive = {
                tuple(int(value) for value in position)
                for position in np.argwhere(normalized == 2)
            } - set(contacts)
            collinear_blockers += any(
                (contact_row + step * delta_row, contact_col + step * delta_col) in inactive
                for (contact_row, contact_col), directions in contacts.items()
                for delta_row, delta_col in directions
                for step in (1, 2, 3)
            )
            seed = tuple(int(value) for value in np.argwhere(grid == tracer)[0])
            direction = _seed_direction(seed, grid.shape)
            orientations.add(direction)
            near_edges += (
                min(seed[0], grid.shape[0] - 1 - seed[0]) == 1
                if direction[1] else
                min(seed[1], grid.shape[1] - 1 - seed[1]) == 1
            )
            row, col = seed
            distance = 0
            while True:
                row += direction[0]
                col += direction[1]
                distance += 1
                if grid[row, col] not in (background, tracer):
                    break
            short_rays += distance == 2
            pivot = (row - direction[0], col - direction[1])
            first_open = sum(
                start == pivot and ray_direction in (
                    (-direction[1], direction[0]),
                    (direction[1], -direction[0]),
                )
                for start, ray_direction, _ in open_rays
            )
            one_sided_forks += first_open == 1
            two_sided_forks += first_open == 0
            output = pair["output"]
            for row in range(1, grid.shape[0] - 1):
                for col in range(1, grid.shape[1] - 1):
                    crossings += (
                        output[row, col] == tracer
                        and output[row - 1, col] == tracer
                        and output[row + 1, col] == tracer
                        and output[row, col - 1] == tracer
                        and output[row, col + 1] == tracer
                    )
        palette_variation += len(palettes) >= 2
        for pair in puzzle["test"]:
            _, contacts, _ = _trace(pair["input"], background, tracer)
            test_simple += len(contacts) <= 3
            test_dense += len(contacts) >= 7
    assert len(layouts) >= 100 and len(shapes) >= 35
    assert len(roles) >= 15 and len(orientations) == 4
    assert len(contact_counts) >= 7 and len(marker_counts) >= 9
    assert minimal >= 3 and blockers >= 20
    assert palette_sizes == {1, 2, 3, 4, 5}
    assert short_rays >= 10 and near_edges >= 10
    assert squares >= 5 and tall >= 20 and wide >= 20
    assert high_inactive >= 10 and palette_variation >= 15
    assert test_simple >= 5 and test_dense >= 5
    assert crossings >= 10 and merges >= 10
    assert collinear_blockers >= 20
    assert one_sided_forks >= 20 and two_sided_forks >= 20
