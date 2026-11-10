from __future__ import annotations

import numpy as np


def _grid_fingerprint(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.reshape(-1)),
    )


def _grid_from_strings(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_TASK_ROWS = (
    (
        (
            "000000000000",
            "002000000000",
            "000000000000",
            "000000000000",
            "030000000000",
            "000000600000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
        ),
        (
            "000000000000",
            "002525252525",
            "000000000000",
            "000000000000",
            "035353535353",
            "000000656565",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
        ),
    ),
    (
        (
            "00000000",
            "00000000",
            "00200000",
            "00000000",
            "00000000",
            "00060000",
            "00000000",
            "00000000",
            "00000000",
            "00000000",
        ),
        (
            "00000000",
            "00000000",
            "00252525",
            "00000000",
            "00000000",
            "00065656",
            "00000000",
            "00000000",
            "00000000",
            "00000000",
        ),
    ),
    (
        (
            "00000000",
            "00000000",
            "00000000",
            "00080000",
            "00000000",
            "00000000",
            "00000000",
        ),
        (
            "00000000",
            "00000000",
            "00000000",
            "00085858",
            "00000000",
            "00000000",
            "00000000",
        ),
    ),
    (
        (
            "00000000",
            "00030000",
            "00400000",
            "00000000",
            "00008000",
            "00000000",
            "00020000",
            "00000000",
            "00000000",
        ),
        (
            "00000000",
            "00035353",
            "00454545",
            "00000000",
            "00008585",
            "00000000",
            "00025252",
            "00000000",
            "00000000",
        ),
    ),
)

_SOURCE_PAIR_FINGERPRINTS = frozenset(
    (
        _grid_fingerprint(_grid_from_strings(input_rows)),
        _grid_fingerprint(_grid_from_strings(output_rows)),
    )
    for input_rows, output_rows in _SOURCE_TASK_ROWS
)


def _is_integer_argument(value: object) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
    )


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None:
        if (
            not _is_integer_argument(num_train)
            or not 2 <= int(num_train) <= 10
        ):
            raise ValueError(
                "num_train must be an integer from 2 through 10"
            )
        num_train = int(num_train)

    if num_test is not None:
        if (
            not _is_integer_argument(num_test)
            or not 1 <= int(num_test) <= 4
        ):
            raise ValueError(
                "num_test must be an integer from 1 through 4"
            )
        num_test = int(num_test)

    if (
        num_train is not None
        and num_test is not None
        and num_train <= num_test
    ):
        raise ValueError("num_train must be greater than num_test")

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        span = 10 - num_test
        num_train = num_test + 1 + (seed // 4) % span
    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, maximum_test + 1))

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError("the requested pair-count combination is impossible")

    return num_train, num_test


def sample_puzzle_parameters(rng: np.random.Generator, num_train: int) -> dict:
    # Only the separator is a convention shared by every pair. Background is
    # the uniform field of each input, so it can change from pair to pair.
    separator = int(rng.integers(0, 10))
    available = [color for color in range(10) if color != separator]
    palette_size = int(rng.integers(3, 5 if num_train == 2 else 7))
    palette = tuple(int(x) for x in rng.choice(available, palette_size, replace=False))
    return {"separator": separator, "train_palette": palette}


def _sample_rows(rng: np.random.Generator, height: int, count: int) -> list[int]:
    # A clustered draw is useful even in a tall grid, where a uniform subset
    # would seldom contain neighbors. The fallback remains bounded.
    mode = int(rng.integers(0, 3))
    best: list[int] | None = None
    best_score = -1
    for _ in range(18):
        rows = sorted(int(x) for x in rng.choice(height, count, replace=False))
        gaps = [b - a for a, b in zip(rows, rows[1:])]
        adjacent = any(gap == 1 for gap in gaps)
        spaced = any(gap >= 3 for gap in gaps)
        score = (int(adjacent) if mode == 1 else
                 int(adjacent) + int(spaced) if mode == 2 and count >= 3 else
                 1)
        if score > best_score:
            best, best_score = rows, score
        if score == (2 if mode == 2 and count >= 3 else 1):
            break
    assert best is not None
    return best


def _sample_dimension(rng: np.random.Generator, minimum: int) -> int:
    band = int(rng.choice(3, p=[0.32, 0.50, 0.18]))
    if band == 0:
        return int(rng.integers(minimum, max(minimum + 1, 10)))
    if band == 1:
        return int(rng.integers(max(minimum, 7), max(minimum + 1, 19)))
    return int(rng.integers(max(minimum, 19), 31))


def _sample_columns(rng: np.random.Generator, width: int, count: int,
                    rows: list[int]) -> list[int]:
    columns: list[int] = []
    for _ in range(count):
        mode = int(rng.choice(4, p=[0.18, 0.27, 0.25, 0.30]))
        if mode == 0:
            column = 0
        elif mode == 1:
            column = width - int(rng.integers(1, min(width, 3) + 1))
        elif mode == 2:
            column = int(rng.integers(0, max(1, width // 3)))
        else:
            column = int(rng.integers(1, width))
        columns.append(column)

    # Neighboring rows should not always have aligned starts.
    for index in range(1, count):
        if rows[index] == rows[index - 1] + 1 and columns[index] == columns[index - 1]:
            columns[index] = (columns[index] + 1) % width
    # Every pair has a visible seed/separator/seed example; other rays may
    # have length one or two. This also guarantees a changed output.
    if max(width - column for column in columns) < 3:
        columns[0] = int(rng.integers(0, width - 2))
    return columns


def _sample_background(rng: np.random.Generator, separator: int,
                       colors: list[int]) -> int:
    allowed = [c for c in range(10) if c != separator and c not in colors]
    if 0 in allowed and rng.random() < 0.48:
        return 0
    return int(rng.choice(allowed))


def _generate_pair(rng: np.random.Generator, regime: str,
                   separator: int, colors: list[int]) -> dict:
    if regime not in {"source_train", "target_test"}:
        raise ValueError("unknown generation regime")
    count = len(colors)
    height = _sample_dimension(rng, max(4, count))
    width = _sample_dimension(rng, 5)
    rows = _sample_rows(rng, height, count)
    columns = _sample_columns(rng, width, count, rows)
    background = _sample_background(rng, separator, colors)
    input_grid = np.full((height, width), background, dtype=np.int8)
    output_grid = input_grid.copy()
    for row, column, color in zip(rows, columns, colors):
        input_grid[row, column] = color
        output_grid[row, column::2] = color
        output_grid[row, column + 1::2] = separator
    return {"input": input_grid, "output": output_grid}


def _training_seed_counts(rng: np.random.Generator, num_train: int,
                          source_like: bool) -> list[int]:
    if source_like:
        counts = [1, 3] if num_train == 2 else [1, 2, 3]
        counts += [int(rng.integers(1, 5)) for _ in range(num_train - len(counts))]
    else:
        counts = [int(rng.integers(2, 6))]
        counts += [int(rng.integers(1, 7)) for _ in range(num_train - 1)]
    rng.shuffle(counts)
    return counts


def _train_color_lists(rng: np.random.Generator, counts: list[int],
                       palette: tuple[int, ...]) -> list[list[int]]:
    # First three appearances expose several seed colors even for two-pair
    # puzzles. Later colors may repeat on rows of the same pair.
    stream = [int(x) for x in rng.permutation(palette[:3])]
    stream += [int(rng.choice(palette)) for _ in range(sum(counts) - 3)]
    lists: list[list[int]] = []
    offset = 0
    for count in counts:
        lists.append(stream[offset:offset + count])
        offset += count
    return lists


def _test_colors(rng: np.random.Generator, count: int, train_colors: set[int],
                 separator: int, source_like: bool) -> list[int]:
    unseen = [c for c in range(10) if c != separator and c not in train_colors]
    if source_like:
        novelty = 1
    else:
        novelty = int(rng.choice(3, p=[0.42, 0.36, 0.22]))
    novelty = min(novelty, count, len(unseen))
    colors = [int(x) for x in rng.choice(unseen, novelty, replace=False)]
    remainder = count - novelty
    familiar = sorted(train_colors)
    if remainder:
        repeat = remainder > 1 and rng.random() < 0.45
        colors.extend(int(x) for x in rng.choice(
            familiar, remainder, replace=repeat or remainder > len(familiar)))
    rng.shuffle(colors)
    return colors


def _background(input_grid: np.ndarray) -> int:
    counts = np.bincount(input_grid.reshape(-1).astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > input_grid.size // 2
    return background


def _analyze_pair(pair: dict, regime: str) -> dict:
    input_grid, output_grid = pair["input"], pair["output"]
    assert input_grid.shape == output_grid.shape
    height, width = input_grid.shape
    assert height >= 4 and width >= 5
    background = _background(input_grid)
    coordinates = [(int(r), int(c)) for r, c in np.argwhere(input_grid != background)]
    count = len(coordinates)
    assert 1 <= count <= (6 if regime == "source_train" else 8)
    assert count <= height
    assert len({r for r, _ in coordinates}) == count
    assert all(np.count_nonzero(input_grid[r] != background) == 1 for r, _ in coordinates)
    assert max(width - c for _, c in coordinates) >= 3

    informative = next((r, c) for r, c in coordinates if c + 1 < width)
    separator = int(output_grid[informative[0], informative[1] + 1])
    assert separator != background
    assert not np.any(input_grid == separator)
    expected = input_grid.copy()
    for row, column in coordinates:
        color = int(input_grid[row, column])
        expected[row, column::2] = color
        expected[row, column + 1::2] = separator
    assert np.array_equal(output_grid, expected)
    assert np.any(input_grid != output_grid)
    return {"background": background, "separator": separator,
            "seed_count": count, "coordinates": tuple(coordinates),
            "seed_colors": tuple(int(input_grid[r, c]) for r, c in coordinates),
            "ray_lengths": tuple(width - c for _, c in coordinates)}


def _predict_from_training(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    # This uses only displayed demonstrations and a test input. It is an
    # internal recoverability check, not an independent reference solver.
    learned: set[int] = set()
    for pair in train:
        inp, out = pair["input"], pair["output"]
        bg = _background(inp)
        for row, column in np.argwhere(inp != bg):
            if column + 1 < inp.shape[1]:
                learned.add(int(out[row, column + 1]))
    assert len(learned) == 1
    separator = next(iter(learned))
    bg = _background(test_input)
    assert separator != bg and not np.any(test_input == separator)
    prediction = test_input.copy()
    for row, column in np.argwhere(test_input != bg):
        color = int(test_input[row, column])
        prediction[row, column::2] = color
        prediction[row, column + 1::2] = separator
    return prediction


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    metadata: dict[str, list[dict]] = {"train": [], "test": []}
    for split, regime in (("train", "source_train"), ("test", "target_test")):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _grid_fingerprint(pair["input"])
            pair_key = (input_key, _grid_fingerprint(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert pair_key not in _SOURCE_PAIR_FINGERPRINTS
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            metadata[split].append(_analyze_pair(pair, regime))
    separators = {m["separator"] for items in metadata.values() for m in items}
    assert len(separators) == 1
    train_colors = {c for m in metadata["train"] for c in m["seed_colors"]}
    assert len(train_colors) >= 3
    assert any(m["seed_count"] >= 2 for m in metadata["train"])
    for pair in puzzle["test"]:
        assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]),
                              pair["output"])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not _is_integer_argument(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng, num_train)
    source_like = rng.random() < 0.20
    train_counts = _training_seed_counts(rng, num_train, source_like)
    train_color_lists = _train_color_lists(rng, train_counts,
                                           parameters["train_palette"])
    puzzle: dict[str, list[dict]] = {"train": [], "test": []}
    seen_inputs: set[tuple] = set()

    def add_pair(split: str, regime: str, colors: list[int]) -> None:
        for _ in range(60):
            candidate = _generate_pair(rng, regime, parameters["separator"], colors)
            input_key = _grid_fingerprint(candidate["input"])
            pair_key = (input_key, _grid_fingerprint(candidate["output"]))
            if input_key not in seen_inputs and pair_key not in _SOURCE_PAIR_FINGERPRINTS:
                puzzle[split].append(candidate)
                seen_inputs.add(input_key)
                return
        raise RuntimeError("could not construct a unique pair")

    for colors in train_color_lists:
        add_pair("train", "source_train", colors)
    train_colors = {c for pair in puzzle["train"] for c in np.unique(pair["input"])
                    if int(c) != _background(pair["input"])}
    for index in range(num_test):
        if source_like and index == 0:
            count = 4
        elif rng.random() < 0.45:
            count = int(rng.choice(train_counts))
        else:
            count = int(rng.integers(1, 9))
        colors = _test_colors(rng, count, {int(x) for x in train_colors},
                              parameters["separator"], source_like and index == 0)
        add_pair("test", "target_test", colors)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_fingerprint(puzzle: dict) -> tuple:
    return tuple(tuple((_grid_fingerprint(pair["input"]),
                        _grid_fingerprint(pair["output"])) for pair in puzzle[split])
                 for split in ("train", "test"))


def self_test() -> None:
    train_counts_seen: set[int] = set()
    test_counts_seen: set[int] = set()
    puzzles: set[tuple] = set()
    shapes: set[tuple[int, int]] = set()
    layouts: set[tuple] = set()
    separators: set[int] = set()
    backgrounds: set[int] = set()
    seed_counts: set[int] = set()
    ray_lengths: set[int] = set()
    measures = {name: 0 for name in (
        "adjacent", "mixed_gaps", "staggered", "left_edge", "short_long",
        "nonzero_background", "background_changes", "zero_seed",
        "repeated_train", "repeated_test", "novel_0", "novel_1",
        "novel_2plus", "familiar_count", "larger_test_count", "dense",
        "adjacent_short", "repeated_nonzero", "source_like")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts_seen.add(len(puzzle["train"]))
        test_counts_seen.add(len(puzzle["test"]))
        puzzles.add(_puzzle_fingerprint(puzzle))
        train_items = [_analyze_pair(p, "source_train") for p in puzzle["train"]]
        test_items = [_analyze_pair(p, "target_test") for p in puzzle["test"]]
        train_colors = {c for m in train_items for c in m["seed_colors"]}
        train_count_set = {m["seed_count"] for m in train_items}
        if len({m["background"] for m in train_items + test_items}) > 1:
            measures["background_changes"] += 1
        if {1, 2, 3}.issubset(train_count_set) and any(
                m["seed_count"] == 4 and len(set(m["seed_colors"]) - train_colors) == 1
                for m in test_items):
            measures["source_like"] += 1
        for split, items in (("train", train_items), ("test", test_items)):
            for pair, m in zip(puzzle[split], items):
                h, w = pair["input"].shape
                shapes.add((h, w))
                layouts.add(((h, w), m["coordinates"]))
                separators.add(m["separator"])
                backgrounds.add(m["background"])
                seed_counts.add(m["seed_count"])
                ray_lengths.update(m["ray_lengths"])
                rows = [r for r, _ in m["coordinates"]]
                columns = [c for _, c in m["coordinates"]]
                gaps = [b - a for a, b in zip(rows, rows[1:])]
                adjacent = 1 in gaps
                repeated = len(set(m["seed_colors"])) < m["seed_count"]
                measures["adjacent"] += adjacent
                measures["mixed_gaps"] += adjacent and any(g >= 3 for g in gaps)
                measures["staggered"] += any(g == 1 and columns[i] != columns[i + 1]
                                             for i, g in enumerate(gaps))
                measures["left_edge"] += 0 in columns
                measures["short_long"] += min(m["ray_lengths"]) <= 3 and max(m["ray_lengths"]) >= 5
                measures["nonzero_background"] += m["background"] != 0
                measures["zero_seed"] += 0 in m["seed_colors"]
                measures["dense"] += m["seed_count"] >= h - 1
                measures["adjacent_short"] += adjacent and min(m["ray_lengths"]) <= 3
                measures["repeated_nonzero"] += repeated and m["background"] != 0
                measures["repeated_" + split] += repeated
                if split == "test":
                    novelty = len(set(m["seed_colors"]) - train_colors)
                    key = "novel_0" if novelty == 0 else "novel_1" if novelty == 1 else "novel_2plus"
                    measures[key] += 1
                    measures["familiar_count"] += m["seed_count"] in train_count_set
                    measures["larger_test_count"] += m["seed_count"] > max(train_count_set)
                    assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]),
                                          pair["output"])
    assert len(train_counts_seen) >= 4 and len(test_counts_seen) >= 3
    assert len(puzzles) >= 40 and len(shapes) >= 60 and len(layouts) >= 100
    assert len(separators) >= 6 and len(backgrounds) >= 6
    assert len(seed_counts) >= 7 and {1, 2, 3}.issubset(ray_lengths)
    assert max(ray_lengths) >= 25 and max(h for h, _ in shapes) >= 25
    assert max(w for _, w in shapes) >= 25 and min(h for h, _ in shapes) <= 6
    for name, minimum in {"adjacent": 20, "mixed_gaps": 5, "staggered": 20,
                          "left_edge": 10, "short_long": 10,
                          "nonzero_background": 30, "background_changes": 10,
                          "zero_seed": 5, "repeated_train": 5,
                          "repeated_test": 5, "novel_0": 3, "novel_1": 3,
                          "novel_2plus": 2, "familiar_count": 5,
                          "larger_test_count": 3, "dense": 3,
                          "adjacent_short": 5, "repeated_nonzero": 3,
                          "source_like": 1}.items():
        assert measures[name] >= minimum, (name, measures[name], minimum)
    deterministic = generate_puzzle(123456, 7, 3)
    assert _puzzle_fingerprint(deterministic) == _puzzle_fingerprint(generate_puzzle(123456, 7, 3))
    for offset, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(2000 + offset, nt, ns)
        assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ns
    for offset, (nt, ns) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(3000 + offset, nt, ns)
        assert nt is None or len(puzzle["train"]) == nt
        assert ns is None or len(puzzle["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4), (3.0, 1)):
        try:
            generate_puzzle(5000, nt, ns)
        except ValueError:
            continue
        raise AssertionError("illegal pair counts accepted")
