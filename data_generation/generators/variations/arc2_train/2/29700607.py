"""Solved examples of the boundary-key path-completion ARC family."""

import numpy as np


_SOURCE_INPUT_SPECS = {
    (12, 12, 4, (5, 3, 8), ((3, 8, "L"), (5, 4, "L"), (8, 5, "R"))),
    (15, 15, 5, (6, 5, 3), ((3, 2, "R"), (6, 4, "L"))),
    (11, 11, 3, (4, 6, 2), ((2, 4, "R"), (4, 4, "L"), (6, 9, "R"))),
    (11, 11, 3, (4, 8, 5), ((4, 7, "L"), (5, 9, "R"))),
}


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _resolve_counts(seed, num_train, num_test):
    if num_train is not None:
        if not _integer(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _integer(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be 1 through 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = int(max(2, num_test + 1) + (seed // 4) % (11 - max(2, num_test + 1)))
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = int(low + (seed // 4 + num_test) % (11 - low))
    elif num_test is None:
        num_test = int(1 + (seed + num_train) % min(4, num_train - 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _grid_key(grid):
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _axis(rng, minimum, test=False):
    bands = ((5, 8), (9, 18), (19, 30))
    weights = (0.24, 0.43, 0.33) if test else (0.29, 0.46, 0.25)
    band = int(rng.choice(3, p=weights))
    lo, hi = bands[band]
    lo = max(lo, minimum)
    if lo > hi:
        lo, hi = minimum, 30
    return int(rng.integers(lo, hi + 1))


def _spaced_rows(rng, height, count):
    # Choosing from a compressed interval gives a uniform nonadjacent subset.
    if count == 0:
        return ()
    compressed = np.sort(rng.choice(np.arange(1, height - 1 - (count - 1)), size=count, replace=False))
    return tuple(int(row + index) for index, row in enumerate(compressed))


def _canonical_pair(rng, regime, background, orientation, demonstrator=False):
    test = regime == "target_test"
    lengths = np.arange(2, 7)
    weights = (0.13, 0.25, 0.23, 0.21, 0.18) if test else (0.17, 0.31, 0.23, 0.17, 0.12)
    if demonstrator:
        key_length = int(rng.integers(3, 7))
    else:
        key_length = int(rng.choice(lengths, p=weights))
    height = _axis(rng, 5, test)
    width = _axis(rng, key_length + 2, test)
    start = int(rng.integers(1, width - key_length))
    colors = tuple(int(color) for color in rng.choice(
        np.asarray([color for color in range(10) if color != background]),
        size=key_length, replace=False))

    capacity = (height - 1) // 2
    configurations = [(left, right)
                      for left in range(min(key_length, capacity) + 1)
                      for right in range(min(key_length - left, capacity) + 1)
                      if not demonstrator or (left >= 1 and right >= 1 and left + right < key_length)]
    if demonstrator and not configurations:
        raise AssertionError("no demonstrator configuration")
    # A target first chooses a destination count, then its side allocation.
    if demonstrator:
        left, right = configurations[int(rng.integers(len(configurations)))]
    else:
        possible_counts = sorted({left + right for left, right in configurations})
        count_weights = np.asarray([0.8 if count == 0 else 1.0 for count in possible_counts], dtype=float)
        count_weights /= count_weights.sum()
        terminal_count = int(rng.choice(possible_counts, p=count_weights))
        choices = [(left, right) for left, right in configurations if left + right == terminal_count]
        left, right = choices[int(rng.integers(len(choices)))]

    canonical = np.full((height, width), background, dtype=np.int8)
    canonical[0, start:start + key_length] = colors
    left_rows = _spaced_rows(rng, height, left)
    right_rows = _spaced_rows(rng, height, right)
    for index, row in enumerate(left_rows):
        canonical[row, 0] = colors[index]
    for offset, row in enumerate(right_rows):
        canonical[row, width - 1] = colors[key_length - 1 - offset]
    input_grid = np.rot90(canonical, -orientation).copy()
    output_grid = _derive_output(input_grid)
    return {"input": input_grid, "output": output_grid}


def _analyze_input(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.all((grid >= 0) & (grid <= 9))
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    candidates = []
    for orientation in range(4):
        canonical = np.rot90(grid, orientation)
        height, width = canonical.shape
        if height < 5 or width < 4:
            continue
        top = np.flatnonzero(canonical[0] != background)
        if not 2 <= len(top) <= 6:
            continue
        start = int(top[0])
        if start < 1 or start + len(top) > width - 1:
            continue
        if not np.array_equal(top, np.arange(start, start + len(top))):
            continue
        colors = tuple(int(value) for value in canonical[0, top])
        if len(set(colors)) != len(colors):
            continue
        terminals = {}
        valid = True
        sides = {"L": [], "R": []}
        for row in range(1, height):
            for col in range(width):
                color = int(canonical[row, col])
                if color == background:
                    continue
                if not (1 <= row <= height - 2 and col in (0, width - 1)
                        and color in colors and color not in terminals):
                    valid = False
                    break
                side = "L" if col == 0 else "R"
                terminals[color] = (row, side)
                sides[side].append(row)
            if not valid:
                break
        if not valid or any(any(b - a < 2 for a, b in zip(sorted(rows), sorted(rows)[1:]))
                            for rows in sides.values()):
            continue
        # Only a left prefix and right suffix can turn without crossing live stems.
        status = tuple(terminals.get(color, (None, None))[1] for color in colors)
        left = sum(side == "L" for side in status)
        right = sum(side == "R" for side in status)
        if status != ("L",) * left + (None,) * (len(colors) - left - right) + ("R",) * right:
            continue
        if any(terminals[colors[i]][0] >= terminals[colors[i + 1]][0]
               for i in range(left - 1)):
            continue
        if any(terminals[colors[len(colors) - 1 - i]][0] >=
               terminals[colors[len(colors) - 2 - i]][0] for i in range(right - 1)):
            continue
        candidates.append({"canonical": canonical, "orientation": orientation,
                           "background": background, "height": height, "width": width,
                           "start": start, "key_colors": colors, "terminals": terminals,
                           "left": left, "right": right})
    assert len(candidates) == 1, "ambiguous or malformed key and terminals"
    return candidates[0]


def _derive_output(input_grid):
    info = _analyze_input(input_grid)
    result = info["canonical"].copy()
    background = info["background"]
    for index, color in enumerate(info["key_colors"]):
        column = info["start"] + index
        terminal = info["terminals"].get(color)
        stop = info["height"] - 1 if terminal is None else terminal[0]
        for row in range(stop + 1):
            assert int(result[row, column]) in (background, color), "stem crossing"
            result[row, column] = color
        if terminal is not None:
            row, side = terminal
            edge = 0 if side == "L" else info["width"] - 1
            for col in range(min(edge, column), max(edge, column) + 1):
                assert int(result[row, col]) in (background, color), "branch crossing"
                result[row, col] = color
    return np.rot90(result, -info["orientation"]).copy()


def _is_source_input(info):
    if info["orientation"] != 0 or info["background"] != 0:
        return False
    terminal_spec = tuple(sorted((color, row, side)
                                 for color, (row, side) in info["terminals"].items()))
    return (info["height"], info["width"], info["start"],
            info["key_colors"], terminal_spec) in _SOURCE_INPUT_SPECS


def _orientations(rng, num_train, num_test):
    mode = int(rng.choice(3, p=(0.38, 0.43, 0.19)))
    if mode == 0:
        orientation = int(rng.integers(4))
        return [orientation] * num_train, [orientation] * num_test
    if mode == 2:
        heldout = int(rng.integers(4))
        available = [value for value in range(4) if value != heldout]
        train = [int(value) for value in rng.choice(available, size=num_train)]
        train[0], train[1] = [int(value) for value in rng.choice(available, size=2, replace=False)]
        rng.shuffle(train)
        return train, [heldout] * num_test
    train = [int(value) for value in rng.integers(0, 4, size=num_train)]
    if len(set(train)) == 1:
        train[int(rng.integers(num_train))] = (train[0] + int(rng.integers(1, 4))) % 4
    return train, [int(value) for value in rng.integers(0, 4, size=num_test)]


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    train_orientations, test_orientations = _orientations(rng, num_train, num_test)
    backgrounds = [int(value) for value in rng.integers(0, 10, size=num_train + num_test)]
    if len(set(backgrounds)) == 1:
        backgrounds[-1] = (backgrounds[-1] + int(rng.integers(1, 10))) % 10
    demonstrator = int(rng.integers(num_train))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count, orientations, offset in (("train", num_train, train_orientations, 0),
                                                ("test", num_test, test_orientations, num_train)):
        regime = "source_train" if split == "train" else "target_test"
        for index in range(count):
            for _ in range(100):
                pair = _canonical_pair(rng, regime, backgrounds[offset + index],
                                       orientations[index], split == "train" and index == demonstrator)
                info = _analyze_input(pair["input"])
                key = _grid_key(pair["input"])
                if key not in seen and not _is_source_input(info):
                    break
            else:
                raise AssertionError("could not construct a distinct non-source pair")
            puzzle[split].append(pair)
            seen.add(key)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        assert 2 <= len(puzzle["train"]) <= 10
        assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
        seen_inputs, seen_pairs = set(), set()
        infos = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                for grid in pair.values():
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert all(1 <= size <= 30 for size in grid.shape)
                    assert np.all((grid >= 0) & (grid <= 9))
                inp, out = pair["input"], pair["output"]
                assert inp.shape == out.shape and not np.array_equal(inp, out)
                info = _analyze_input(inp)
                assert not _is_source_input(info)
                assert np.array_equal(out, _derive_output(inp))
                assert set(np.unique(inp)) == set(np.unique(out)) == {info["background"], *info["key_colors"]}
                input_key = _grid_key(inp)
                pair_key = (input_key, _grid_key(out))
                assert input_key not in seen_inputs and pair_key not in seen_pairs
                seen_inputs.add(input_key)
                seen_pairs.add(pair_key)
                infos[split].append(info)
        train = infos["train"]
        assert any(info["left"] and info["right"] and
                   info["left"] + info["right"] < len(info["key_colors"]) for info in train)
        train_orientations = {info["orientation"] for info in train}
        assert all(info["orientation"] in train_orientations or len(train_orientations) >= 2
                   for info in infos["test"])
        assert len({info["background"] for split in infos for info in infos[split]}) >= 2
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("invalid generated puzzle") from exc


def _puzzle_fingerprint(puzzle):
    return tuple((split, _grid_key(pair["input"]), _grid_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_fingerprint(puzzle) for puzzle in puzzles}) == 42
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    assert _puzzle_fingerprint(generate_puzzle(7)) == _puzzle_fingerprint(generate_puzzle(7))
    for seed, train, test in ((7001, 7, 3), (8101, 2, 1), (8102, 10, 1),
                              (8103, 10, 4), (8201, None, 1), (8202, None, 4),
                              (8203, 2, None), (8204, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 3}, {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(8300, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    features = {"shapes": set(), "lengths": set(), "orientations": set(),
                "backgrounds": set(), "terminal_counts": set(), "layouts": set(),
                "starts": set(), "palettes": set()}
    within_palettes = within_shapes = within_orientations = 0
    test_unseen_color = zero_path = same_row = no_terminal = one_terminal = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        all_info = {split: [_analyze_input(pair["input"]) for pair in puzzle[split]]
                    for split in ("train", "test")}
        flat = all_info["train"] + all_info["test"]
        within_palettes += len({frozenset(info["key_colors"]) for info in flat}) > 1
        within_shapes += len({(info["height"], info["width"]) for info in flat}) > 1
        within_orientations += len({info["orientation"] for info in flat}) > 1
        train_colors = {color for info in all_info["train"] for color in info["key_colors"]}
        test_unseen_color += any(set(info["key_colors"]) - train_colors for info in all_info["test"])
        for info in flat:
            k = len(info["key_colors"])
            n = len(info["terminals"])
            features["shapes"].add((info["height"], info["width"]))
            features["lengths"].add(k)
            features["orientations"].add(info["orientation"])
            features["backgrounds"].add(info["background"])
            features["terminal_counts"].add(n)
            features["starts"].add((info["width"], info["start"], k))
            features["palettes"].add(tuple(sorted(info["key_colors"])))
            layout = (info["height"], info["width"], info["start"], k,
                      tuple(sorted((info["key_colors"].index(color), row, side)
                                   for color, (row, side) in info["terminals"].items())),
                      info["orientation"])
            features["layouts"].add(layout)
            zero_path += info["background"] != 0 and 0 in info["key_colors"]
            same_row += len({row for row, _ in info["terminals"].values()}) < n
            no_terminal += n == 0
            one_terminal += n == 1
    assert len(features["shapes"]) >= 40 and len(features["layouts"]) >= 100
    assert features["lengths"] == {2, 3, 4, 5, 6}
    assert features["orientations"] == {0, 1, 2, 3}
    assert features["backgrounds"] == set(range(10))
    assert len(features["terminal_counts"]) >= 6
    assert len(features["palettes"]) >= 50 and len(features["starts"]) >= 30
    assert within_palettes >= 35 and within_shapes >= 35 and within_orientations >= 15
    assert test_unseen_color >= 5 and zero_path >= 10
    assert same_row >= 2 and no_terminal >= 2 and one_terminal >= 2

    train_patterns, test_patterns = set(), set()
    split_counts = {split: {name: 0 for name in
                   ("missing_outer", "two_same_side", "all_terminate", "same_row",
                    "no_terminal", "one_terminal", "compact", "large", "square",
                    "centered", "offcenter", "zero_path", "source_pattern")}
                    for split in ("train", "test")}
    stem_lengths, branch_lengths = [], []
    late_demonstrations = 0
    for puzzle in puzzles:
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                info = _analyze_input(pair["input"])
                k = len(info["key_colors"])
                n = len(info["terminals"])
                left, right = info["left"], info["right"]
                (train_patterns if split == "train" else test_patterns).add(
                    (k, left, right, info["orientation"]))
                counts = split_counts[split]
                counts["missing_outer"] += left == 0 or right == 0
                counts["two_same_side"] += max(left, right) >= 2
                counts["all_terminate"] += n == k
                counts["same_row"] += len({row for row, _ in info["terminals"].values()}) < n
                counts["no_terminal"] += n == 0
                counts["one_terminal"] += n == 1
                counts["compact"] += min(info["height"], info["width"]) <= 8
                counts["large"] += max(info["height"], info["width"]) >= 25
                counts["square"] += info["height"] == info["width"]
                left_margin = info["start"] - 1
                right_margin = info["width"] - info["start"] - k - 1
                counts["centered"] += abs(left_margin - right_margin) <= 1
                counts["offcenter"] += min(left_margin, right_margin) == 0
                counts["zero_path"] += info["background"] != 0 and 0 in info["key_colors"]
                counts["source_pattern"] += k == 3 and left == right == 1 and info["orientation"] == 0
                if split == "train" and index >= 2 and left and right and n < k:
                    late_demonstrations += 1
                for offset, color in enumerate(info["key_colors"]):
                    terminal = info["terminals"].get(color)
                    stem_lengths.append(info["height"] - 1 if terminal is None else terminal[0])
                    if terminal is not None:
                        branch_lengths.append(info["start"] + offset if terminal[1] == "L"
                                              else info["width"] - 1 - info["start"] - offset)
    assert len(train_patterns) >= 80 and len(test_patterns) >= 40
    for split in ("train", "test"):
        counts = split_counts[split]
        assert all(counts[name] > 0 for name in counts)
    assert split_counts["test"]["source_pattern"] < len(test_patterns)
    assert late_demonstrations > 0
    assert min(stem_lengths) == min(branch_lengths) == 1
    assert max(stem_lengths) >= 25 and max(branch_lengths) >= 20
    shared = generate_puzzle(245)
    assert any(len((info := _analyze_input(pair["input"]))["key_colors"]) == 3
               and info["left"] == info["right"] == 1
               and len({row for row, _ in info["terminals"].values()}) == 1
               for pair in shared["test"])
