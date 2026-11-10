import numpy as np


_SOURCE_PAIRS = {
    (("020200200202", "220220220202"), ("0003000", "0022000", "0022000", "0002200", "0000200", "0000200", "0000200", "0000200")),
    (("200020202002", "220220202202"), ("0003000", "0002200", "0002200", "0002000", "0002000", "0002200", "0000200", "0000200")),
    (("200200200202", "220220220202"), ("0003000", "0002200", "0000220", "0000022", "0000002", "0000002", "0000002", "0000002")),
    (("202002000202", "202202202202"), ("0003000", "0002000", "0002000", "0002200", "0000220", "0000220", "0000200", "0000200")),
}


def _count(name, value, low, high):
    if value is None:
        return None
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _pair_counts(seed, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _sample_mix(rng, mixed=False):
    length = int(rng.integers(3, 10))
    if mixed:
        length = max(4, length)
        straight = int(rng.integers(1, length - 1))
        lateral = length - straight
        if lateral < 2:
            straight -= 1
            lateral += 1
    else:
        kind = int(rng.integers(0, 10))
        if kind == 0:
            straight = length
        elif kind == 1:
            straight = 0
        else:
            straight = int(rng.integers(1, length))
        lateral = length - straight
    return straight, lateral


def _commands(rng, straight, lateral, style):
    if style in ("original_train", "original_test"):
        assert straight >= 1 and lateral >= 2
        length = straight + lateral
        result = [None] * length
        if style == "original_train":
            left_pos = int(rng.integers(0, length // 2))
            result[left_pos] = "L"
            rights = lateral - 1
            straights = straight
            if left_pos != 0:
                result[0] = "R"
                rights -= 1
        else:
            result[0] = "S"
            left_pos = int(rng.integers((length + 1) // 2, length))
            result[left_pos] = "L"
            rights = lateral - 1
            straights = straight - 1
        free = [i for i, value in enumerate(result) if value is None]
        rest = ["R"] * rights + ["S"] * straights
        rng.shuffle(rest)
        for index, command in zip(free, rest):
            result[index] = command
        return tuple(result)

    if style == "zigzag":
        first = "L" if int(rng.integers(2)) else "R"
        directions = [first if i % 2 == 0 else ("R" if first == "L" else "L") for i in range(lateral)]
    elif style == "anchor_s":
        chance = (0.16, 0.5, 0.84)[int(rng.integers(3))]
        directions = ["L", "R"] + ["L" if rng.random() < chance else "R" for _ in range(lateral - 2)]
    else:
        chance = (0.16, 0.5, 0.84)[int(rng.integers(3))]
        directions = ["L" if rng.random() < chance else "R" for _ in range(lateral)]
    if style in ("start_s", "anchor_s") and straight:
        result = ["S"]
        remainder = ["S"] * (straight - 1) + directions
    elif style == "start_lateral" and lateral:
        result = [directions[0]]
        remainder = ["S"] * straight + directions[1:]
    else:
        result = []
        remainder = ["S"] * straight + directions
    rng.shuffle(remainder)
    return tuple(result + remainder)


def _gaps(rng, commands):
    gaps = [int(x) for x in rng.integers(1, 5, size=len(commands) - 1)]
    base = commands.count("S") + 2 * (len(commands) - commands.count("S"))
    while base + sum(gaps) > 30:
        eligible = [i for i, gap in enumerate(gaps) if gap > 1]
        gaps[int(rng.choice(eligible))] -= 1
    return tuple(gaps)


def _encode_input(commands, gaps, background, signal):
    width = commands.count("S") + 2 * (len(commands) - commands.count("S")) + sum(gaps)
    grid = np.full((2, width), background, dtype=np.int8)
    col = 0
    for index, command in enumerate(commands):
        if command == "S":
            grid[:, col] = signal
            col += 1
        elif command == "R":
            grid[0, col] = signal
            grid[1, col:col + 2] = signal
            col += 2
        else:
            grid[0, col + 1] = signal
            grid[1, col:col + 2] = signal
            col += 2
        if index < len(gaps):
            col += gaps[index]
    return grid


def _render_output(commands, background, signal, marker):
    straight = commands.count("S")
    lateral = len(commands) - straight
    grid = np.full((1 + 2 * straight + lateral, 2 * lateral + 1), background, dtype=np.int8)
    row, col = 0, lateral
    grid[row, col] = marker
    for command in commands:
        row += 1
        grid[row, col] = signal
        if command == "S":
            row += 1
            grid[row, col] = signal
        elif command == "L":
            col -= 1
            grid[row, col] = signal
        else:
            col += 1
            grid[row, col] = signal
    return grid


def _signature(grid):
    return tuple(tuple(int(x) for x in row) for row in grid)


def _source_copy(pair):
    return (tuple("".join(str(int(x)) for x in row) for row in pair["input"]),
            tuple("".join(str(int(x)) for x in row) for row in pair["output"])) in _SOURCE_PAIRS


def _decode(input_grid):
    assert input_grid.shape[0] == 2
    colors = set(int(x) for x in np.unique(input_grid))
    assert len(colors) == 2
    signal = int(input_grid[1, 0])
    background = next(iter(colors - {signal}))
    assert int(input_grid[1, -1]) == signal
    active = np.any(input_grid != background, axis=0)
    assert bool(active[0]) and bool(active[-1])
    commands, gaps = [], []
    col = 0
    while col < input_grid.shape[1]:
        assert bool(active[col])
        start = col
        while col < input_grid.shape[1] and bool(active[col]):
            col += 1
        token = input_grid[:, start:col]
        if token.shape[1] == 1 and np.all(token == signal):
            commands.append("S")
        elif token.shape[1] == 2 and np.array_equal(token, [[signal, background], [signal, signal]]):
            commands.append("R")
        elif token.shape[1] == 2 and np.array_equal(token, [[background, signal], [signal, signal]]):
            commands.append("L")
        else:
            raise AssertionError("invalid glyph")
        if col < input_grid.shape[1]:
            start = col
            while col < input_grid.shape[1] and not bool(active[col]):
                col += 1
            assert col < input_grid.shape[1]
            gaps.append(col - start)
    assert 3 <= len(commands) <= 9
    assert len(gaps) == len(commands) - 1
    assert all(1 <= gap <= 4 for gap in gaps)
    return tuple(commands), tuple(gaps), background, signal


def _make_pair(commands, gaps, background, signal, marker):
    return {"input": _encode_input(commands, gaps, background, signal),
            "output": _render_output(commands, background, signal, marker)}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _count("seed", seed, 0, 2**63 - 1)
    train_count, test_count = _pair_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    marker = int(rng.integers(10))
    colors = [color for color in range(10) if color != marker]
    shared_palette = seed % 4 == 0
    fixed_shape = seed % 7 == 0
    challenge = seed % 3
    palette = tuple(int(x) for x in rng.choice(colors, size=2, replace=False))
    anchor_mix = (int(rng.integers(2, 4)), int(rng.integers(3, 5)))
    second_mix = _sample_mix(rng, mixed=challenge == 0)
    if not fixed_shape:
        for _ in range(40):
            if second_mix[0] != anchor_mix[0] and second_mix[1] != anchor_mix[1]:
                break
            second_mix = _sample_mix(rng, mixed=challenge == 0)
        if second_mix[0] == anchor_mix[0] or second_mix[1] == anchor_mix[1]:
            second_mix = (anchor_mix[0] + 1, anchor_mix[1] - 1)
    used_inputs, used_commands = set(), set()
    puzzle = {"train": [], "test": []}
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for attempt in range(300):
                if fixed_shape:
                    straight, lateral = anchor_mix
                elif split == "train" and index == 0:
                    straight, lateral = anchor_mix
                elif split == "train" and index == 1:
                    straight, lateral = second_mix
                else:
                    straight, lateral = _sample_mix(rng, mixed=challenge == 0)
                if challenge == 0:
                    style = "original_train" if split == "train" else "original_test"
                elif split == "train" and index == 0:
                    style = "anchor_s"
                elif challenge == 1 and index == 0:
                    style = "start_s"
                elif challenge == 2 and index == 0:
                    style = "start_s" if split == "train" else "start_lateral"
                else:
                    style = ("free", "start_s", "start_lateral", "zigzag")[int(rng.integers(4))]
                commands = _commands(rng, straight, lateral, style)
                if commands in used_commands and attempt < 100:
                    continue
                gaps = _gaps(rng, commands)
                if shared_palette:
                    background, signal = palette
                else:
                    background, signal = (int(x) for x in rng.choice(colors, size=2, replace=False))
                pair = _make_pair(commands, gaps, background, signal, marker)
                signature = _signature(pair["input"])
                if signature in used_inputs or _source_copy(pair):
                    continue
                puzzle[split].append(pair)
                used_inputs.add(signature)
                used_commands.add(commands)
                break
            else:
                raise RuntimeError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    marker = None
    seen_inputs, seen_pairs = set(), set()
    decoded = {"train": [], "test": []}
    palettes = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= value <= 30 for value in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            commands, gaps, background, signal = _decode(pair["input"])
            input_key = _signature(pair["input"])
            pair_key = (input_key, _signature(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not np.array_equal(pair["input"], pair["output"])
            assert not _source_copy(pair)
            new_colors = set(int(x) for x in np.unique(pair["output"])) - {background, signal}
            assert len(new_colors) == 1
            pair_marker = new_colors.pop()
            if marker is None:
                marker = pair_marker
            assert pair_marker == marker and marker not in (background, signal)
            expected = _render_output(commands, background, signal, marker)
            assert np.array_equal(pair["output"], expected)
            assert int(np.count_nonzero(pair["output"] == marker)) == 1
            assert pair["output"].shape == (1 + 2 * commands.count("S") + len(commands) - commands.count("S"),
                                            2 * (len(commands) - commands.count("S")) + 1)
            decoded[split].append(commands)
            palettes.add((background, signal))
    train_symbols = set().union(*(set(commands) for commands in decoded["train"]))
    test_symbols = set().union(*(set(commands) for commands in decoded["test"]))
    assert train_symbols == {"S", "L", "R"}
    assert test_symbols <= train_symbols
    train_mixes = {(commands.count("S"), len(commands) - commands.count("S"))
                   for commands in decoded["train"]}
    if len(train_mixes) > 1:
        assert len({straight for straight, _ in train_mixes}) > 1
        assert len({lateral for _, lateral in train_mixes}) > 1


def _predict_from_training(train, test_input):
    markers = []
    for pair in train:
        colors = set(int(x) for x in np.unique(pair["output"])) - set(int(x) for x in np.unique(pair["input"]))
        assert len(colors) == 1
        markers.append(colors.pop())
    assert len(set(markers)) == 1
    commands, _, background, signal = _decode(test_input)
    lateral = sum(command != "S" for command in commands)
    height = 1 + sum(2 if command == "S" else 1 for command in commands)
    predicted = np.full((height, 2 * lateral + 1), background, dtype=np.int8)
    y, x = 0, lateral
    predicted[y, x] = markers[0]
    for command in commands:
        if command == "S":
            for _ in range(2):
                y += 1
                predicted[y, x] = signal
        else:
            y += 1
            predicted[y, x] = signal
            x += 1 if command == "R" else -1
            predicted[y, x] = signal
    return predicted


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[role], b[role]) for a, b in zip(first[split], second[split])
                   for role in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(12345), generate_puzzle(12345))
    assert _puzzles_equal(generate_puzzle(54321, 7, 3), generate_puzzle(54321, 7, 3))
    for seed, train, test in ((100, 2, 1), (101, 10, 1), (102, 10, 4),
                              (103, 5, 2), (104, None, 1), (105, None, 4),
                              (106, 2, None), (107, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 2, "num_test": 4}):
        try:
            generate_puzzle(200, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")

    train_counts, test_counts = set(), set()
    commands_seen, shapes, palettes, gap_widths = set(), set(), set(), set()
    first_symbols = {"train": set(), "test": set()}
    varied_train_shapes = mixed_gaps = recolored_puzzles = 0
    pure_straight = pure_lateral = one_hand = long_run = zigzag = 0
    relative_lengths = set()
    original_shift = overlap = other_shift = 0
    all_puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        all_puzzles.append(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_shapes, puzzle_palettes = set(), set()
        train_decoded, test_decoded = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                commands, gaps, bg, signal = _decode(pair["input"])
                (train_decoded if split == "train" else test_decoded).append(commands)
                first_symbols[split].add(commands[0])
                commands_seen.add(commands)
                shapes.add((pair["input"].shape, pair["output"].shape))
                palettes.add((bg, signal))
                puzzle_palettes.add((bg, signal))
                gap_widths.update(gaps)
                mixed_gaps += len(set(gaps)) > 1
                pure_straight += set(commands) == {"S"}
                pure_lateral += "S" not in commands
                one_hand += ("L" in commands) != ("R" in commands)
                long_run += "LLL" in "".join(commands) or "RRR" in "".join(commands)
                lateral_seq = [x for x in commands if x != "S"]
                zigzag += len(lateral_seq) >= 3 and all(a != b for a, b in zip(lateral_seq, lateral_seq[1:]))
                if split == "train":
                    train_shapes.add(pair["output"].shape)
                else:
                    assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]), pair["output"])
        varied_train_shapes += len(train_shapes) > 1
        recolored_puzzles += len(puzzle_palettes) > 1
        relative_lengths.update("shorter" if len(test_commands) < len(train_decoded[0]) else
                                "longer" if len(test_commands) > len(train_decoded[0]) else "equal"
                                for test_commands in test_decoded)
        original_shift += all(c[0] != "S" and all(x != "L" or i < len(c)//2 for i, x in enumerate(c)) for c in train_decoded) and all(c[0] == "S" and all(x != "L" or i >= (len(c)+1)//2 for i, x in enumerate(c)) for c in test_decoded)
        overlap += train_decoded[0][0] == test_decoded[0][0]
        other_shift += train_decoded[0][0] == "S" and test_decoded[0][0] != "S"
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(commands_seen) >= 100 and len(shapes) >= 40 and len(palettes) >= 20
    assert gap_widths == {1, 2, 3, 4}
    assert varied_train_shapes >= 20 and mixed_gaps >= 60 and recolored_puzzles >= 20
    assert pure_straight >= 2 and pure_lateral >= 2 and one_hand >= 20
    assert long_run >= 10 and zigzag >= 5
    assert relative_lengths == {"shorter", "equal", "longer"}
    assert original_shift >= 3 and overlap >= 5 and other_shift >= 5
    assert first_symbols["train"] == first_symbols["test"] == {"S", "L", "R"}
    assert sum(not _puzzles_equal(a, b) for a, b in zip(all_puzzles, all_puzzles[1:])) >= 40
