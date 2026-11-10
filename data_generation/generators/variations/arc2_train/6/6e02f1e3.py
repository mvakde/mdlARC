"""Whole-puzzle generator for the three-color-count indicator family."""

from itertools import permutations

import numpy as np


_TEMPLATE_CELLS = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 1), (2, 2)),
    ((0, 2), (1, 1), (2, 0)),
)
_TWO_PROFILES = ((1, 8), (2, 7), (3, 6), (4, 5))
_THREE_PROFILES = (
    (1, 1, 7), (1, 2, 6), (1, 3, 5), (1, 4, 4),
    (2, 2, 5), (2, 3, 4), (3, 3, 3),
)
# Exact flattened pairs from the original task, including its solved test.
_SOURCE_PAIR_KEYS = frozenset({
    ((3, 3, 3, 3, 3, 3, 3, 3, 3), (5, 5, 5, 0, 0, 0, 0, 0, 0)),
    ((3, 3, 3, 4, 2, 2, 4, 4, 2), (0, 0, 5, 0, 5, 0, 5, 0, 0)),
    ((4, 4, 4, 4, 4, 4, 4, 4, 4), (5, 5, 5, 0, 0, 0, 0, 0, 0)),
    ((4, 4, 4, 4, 4, 4, 3, 3, 3), (5, 0, 0, 0, 5, 0, 0, 0, 5)),
    ((2, 2, 2, 3, 2, 3, 3, 3, 3), (5, 0, 0, 0, 5, 0, 0, 0, 5)),
    ((4, 4, 4, 2, 3, 2, 3, 2, 3), (0, 0, 5, 0, 5, 0, 5, 0, 0)),
})


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _pair_key(pair):
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _source_key(pair):
    return (tuple(int(x) for x in pair["input"].flat),
            tuple(int(x) for x in pair["output"].flat))


def _optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _optional_count(num_train, "num_train", 2, 10)
    num_test = _optional_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = num_test + 1
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = low + seed % (11 - low)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _parameters(rng):
    roles = [int(x) for x in rng.choice(10, size=2, replace=False)]
    # Source-like palettes remain common; other puzzles demonstrate that an
    # output color is still an ordinary input color when present in the input.
    overlap = bool(rng.integers(0, 4) == 0)
    available = [c for c in range(10) if c not in roles]
    rng.shuffle(available)
    pool_size = int(rng.integers(4, 9))
    if overlap:
        pool = [available[0], roles[int(rng.integers(0, 2))]]
        pool += available[1:pool_size - 1]
    else:
        pool = available[:pool_size]
    mapping = [0, 1, 2]
    if rng.integers(0, 10) < 3:
        while mapping == [0, 1, 2]:
            mapping = [int(x) for x in rng.permutation(3)]
    return roles[0], roles[1], tuple(pool), tuple(mapping)


def _cardinalities(rng, num_train, num_test):
    if num_train == 2:
        counts = [int(x) for x in rng.choice((1, 2, 3), size=2, replace=False)]
    else:
        counts = [1, 2, 3]
        counts += [int(rng.choice((1, 2, 3), p=(.20, .38, .42)))
                   for _ in range(num_train - 3)]
        # Leave room for distinct uniform tests even when output-role
        # colors were deliberately excluded from the input palette.
        for index in range(3, len(counts)):
            if counts[:index + 1].count(1) > 4:
                counts[index] = int(rng.choice((2, 3)))
    rng.shuffle(counts)
    if num_test == 1:
        tests = [int(rng.choice(counts))]
    else:
        unique = list(set(counts))
        rng.shuffle(unique)
        tests = unique[:min(num_test, len(unique))]
        while len(tests) < num_test:
            tests.append(int(rng.choice(unique)))
        rng.shuffle(tests)
    return counts, tests


def _pick_palettes(rng, train_counts, test_counts, pool):
    train = []
    first = set()
    anchor = pool[0]
    for count in train_counts:
        if count not in first:
            if count == 1:
                colors = (anchor,)
            elif count == 2:
                colors = (anchor, pool[1])
            elif 2 in train_counts:
                colors = (anchor, pool[2], pool[3])
            else:
                colors = (anchor, pool[1], pool[2])
            first.add(count)
        else:
            colors = tuple(int(x) for x in rng.choice(pool, size=count, replace=False))
        train.append(colors)
    train_union = tuple(sorted({color for colors in train for color in colors}))
    tests = []
    for count in test_counts:
        choices = [tuple(int(x) for x in rng.choice(train_union, size=count, replace=False))
                   for _ in range(12)]
        used = {frozenset(colors) for colors in train + tests if len(colors) == count}
        different = [colors for colors in choices if frozenset(colors) not in used]
        tests.append(different[0] if different else choices[0])
    return train, tests


def _profile(rng, count, regime, occurrence):
    if count == 1:
        return (9,)
    if count == 2:
        return _TWO_PROFILES[int(rng.integers(0, len(_TWO_PROFILES)))]
    if regime == "source_train" and occurrence == 1:
        return (3, 3, 3) if rng.integers(0, 2) else _THREE_PROFILES[int(rng.integers(0, 6))]
    if regime == "source_train" and occurrence == 2:
        return _THREE_PROFILES[int(rng.integers(0, 6))]
    if regime == "target_test":
        return ((3, 3, 3) if rng.integers(0, 4) == 0
                else _THREE_PROFILES[int(rng.integers(0, 6))])
    return _THREE_PROFILES[int(rng.integers(0, len(_THREE_PROFILES)))]


def _input_grid(rng, colors, profile, layout):
    colors = list(colors)
    profile = list(profile)
    rng.shuffle(colors)
    rng.shuffle(profile)
    if layout == 0:  # horizontal bands when frequencies permit
        order = [(r, c) for r in range(3) for c in range(3)]
    elif layout == 1:  # vertical bands
        order = [(r, c) for c in range(3) for r in range(3)]
    elif layout == 2:  # compact corner block
        order = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2),
                 (1, 2), (2, 0), (2, 1), (2, 2)]
    elif layout == 3:  # separated and irregular cells
        order = [(int(i // 3), int(i % 3)) for i in rng.permutation(9)]
    else:  # checker style distribution
        order = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2),
                 (0, 1), (1, 0), (1, 2), (2, 1)]
    if rng.integers(0, 2):
        order = [(r, 2 - c) for r, c in order]
    if rng.integers(0, 2):
        order = [(2 - r, c) for r, c in order]
    if rng.integers(0, 2):
        order = [(c, r) for r, c in order]
    grid = np.empty((3, 3), dtype=np.int8)
    index = 0
    for color, size in zip(colors, profile):
        for row, column in order[index:index + size]:
            grid[row, column] = color
        index += size
    return grid


def _output_grid(count, background, indicator, mapping):
    grid = np.full((3, 3), background, dtype=np.int8)
    for row, col in _TEMPLATE_CELLS[mapping[count - 1]]:
        grid[row, col] = indicator
    return grid


def _make_pair(rng, count, colors, profile, layout, parameters, used,
               fallback_colors=tuple(range(10))):
    background, indicator, _, mapping = parameters
    output = _output_grid(count, background, indicator, mapping)
    # The input space is large except for uniform grids. The schedule has
    # at most eight uniform pairs, leaving ten possible uniform colors.
    for attempt in range(64):
        actual_colors = colors
        if attempt and count == 1:
            actual_colors = (int(rng.choice(fallback_colors)),)
        grid = _input_grid(rng, actual_colors, profile, layout if attempt < 16 else int(rng.integers(0, 5)))
        pair = {"input": grid, "output": output}
        key = _grid_key(grid)
        if key not in used and _source_key(pair) not in _SOURCE_PAIR_KEYS and not np.array_equal(grid, output):
            used.add(key)
            return pair
    if count == 1:
        for color in fallback_colors:
            grid = np.full((3, 3), color, dtype=np.int8)
            pair = {"input": grid, "output": output}
            key = _grid_key(grid)
            if key not in used and _source_key(pair) not in _SOURCE_PAIR_KEYS:
                used.add(key)
                return pair
    else:
        for ordered_colors in permutations(colors):
            flat = np.repeat(np.asarray(ordered_colors, dtype=np.int8), profile)
            for shift in range(9):
                grid = np.roll(flat, shift).reshape(3, 3)
                pair = {"input": grid, "output": output}
                key = _grid_key(grid)
                if key not in used and _source_key(pair) not in _SOURCE_PAIR_KEYS and not np.array_equal(grid, output):
                    used.add(key)
                    return pair
    raise RuntimeError("unable to construct unique pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    parameters = _parameters(rng)
    train_counts, test_counts = _cardinalities(rng, num_train, num_test)
    train_palettes, test_palettes = _pick_palettes(rng, train_counts, test_counts, parameters[2])
    layouts = [int(x) for x in rng.permutation(5)]
    while len(layouts) < num_train + num_test:
        layouts.extend(int(x) for x in rng.permutation(5))
    used = set()
    train, test = [], []
    three_occurrence = 0
    for i, (count, colors) in enumerate(zip(train_counts, train_palettes)):
        if count == 3:
            three_occurrence += 1
        profile = _profile(rng, count, "source_train", three_occurrence)
        layout = layouts[i]
        if (count == 3 and three_occurrence == 1 and train_counts.count(3) >= 2
                and rng.integers(0, 2) == 0):
            # A clean three-band demonstration sits beside an unequal
            # three-color demonstration in the same training set.
            profile = (3, 3, 3)
            layout = int(rng.integers(0, 2))
        train.append(_make_pair(rng, count, colors, profile, layout, parameters, used))
    for i, (count, colors) in enumerate(zip(test_counts, test_palettes)):
        profile = _profile(rng, count, "target_test", i + 1)
        demonstrated_colors = {int(x) for pair in train for x in np.unique(pair["input"])}
        safe_colors = tuple(c for c in range(10)
                            if c in demonstrated_colors or c not in parameters[:2])
        test.append(_make_pair(rng, count, colors, profile, layouts[num_train + i],
                               parameters, used, safe_colors))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _inferred_roles(output):
    values, counts = np.unique(output, return_counts=True)
    assert len(values) == 2 and tuple(sorted(int(x) for x in counts)) == (3, 6)
    return int(values[int(np.argmax(counts))]), int(values[int(np.argmin(counts))])


def _template_index(output, indicator):
    marked = {(int(r), int(c)) for r, c in np.argwhere(output == indicator)}
    return _TEMPLATE_CELLS.index(tuple(sorted(marked)))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    roles = None
    learned = {}
    train_colors = set()
    test_counts = []
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for matrix in (grid, output):
                assert isinstance(matrix, np.ndarray) and matrix.ndim == 2
                assert 1 <= matrix.shape[0] <= 30 and 1 <= matrix.shape[1] <= 30
                assert np.issubdtype(matrix.dtype, np.integer)
                assert 0 <= int(matrix.min()) and int(matrix.max()) <= 9
            assert grid.shape == output.shape == (3, 3)
            assert not np.array_equal(grid, output)
            key, pair_key = _grid_key(grid), _pair_key(pair)
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert _source_key(pair) not in _SOURCE_PAIR_KEYS
            palette = set(int(x) for x in np.unique(grid))
            count = len(palette)
            assert count in (1, 2, 3)
            if split == "train":
                train_colors.update(palette)
            else:
                test_counts.append(count)
            this_roles = _inferred_roles(output)
            if roles is None:
                roles = this_roles
            assert this_roles == roles
            template = _template_index(output, roles[1])
            expected = np.full((3, 3), roles[0], dtype=np.int8)
            for row, col in _TEMPLATE_CELLS[template]:
                expected[row, col] = roles[1]
            assert np.array_equal(output, expected)
            if split == "train":
                assert count not in learned or learned[count] == template
                learned[count] = template
            else:
                assert count in learned and learned[count] == template
                assert palette.intersection(roles).issubset(train_colors)
    assert len(set(learned.values())) == len(learned)
    if len(train) >= 3:
        assert set(learned) == {1, 2, 3}
    else:
        assert len(learned) == 2
    if len(test) >= 2:
        assert len(set(test_counts)) >= 2


def _signature(puzzle):
    return tuple(tuple(_pair_key(pair) for pair in puzzle[split]) for split in ("train", "test"))


def _canonical(grid):
    labels = {}
    return tuple(labels.setdefault(int(x), len(labels)) for x in grid.flat)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    roles, mappings, palettes, layouts = set(), set(), set(), set()
    two_profiles, train_three, test_three = set(), set(), set()
    test_counts = set()
    overlap, many_colors, mixed_tests, different_triples = 0, 0, 0, 0
    row_bands = col_bands = dispersed = test_role_overlap = balanced_tests = 0
    source_mapping = changed_mapping = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train, test = puzzle["train"], puzzle["test"]
        background, indicator = _inferred_roles(train[0]["output"])
        roles.add((background, indicator))
        mapping = {}
        train_palette = set()
        train_triples = set()
        for pair in train:
            grid = pair["input"]
            count = len(np.unique(grid))
            mapping[count] = _template_index(pair["output"], indicator)
            train_palette.update(int(x) for x in np.unique(grid))
            if count == 3:
                train_triples.add(frozenset(int(x) for x in np.unique(grid)))
        mappings.add(tuple(sorted(mapping.items())))
        source_mapping += mapping == {1: 0, 2: 1, 3: 2}
        changed_mapping += any(template != count - 1 for count, template in mapping.items())
        palettes.add(tuple(sorted(train_palette)))
        many_colors += len(train_palette) > 3
        overlap += bool(train_palette.intersection((background, indicator)))
        mixed_tests += len({len(np.unique(pair["input"])) for pair in test}) > 1
        for index, pair in enumerate(train + test):
            grid = pair["input"]
            count = len(np.unique(grid))
            profile = tuple(sorted(int(x) for x in np.unique(grid, return_counts=True)[1]))
            layouts.add(_canonical(grid))
            if count > 1 and index < len(train):
                row_bands += all(len(set(int(x) for x in grid[row])) == 1 for row in range(3))
                col_bands += all(len(set(int(x) for x in grid[:, col])) == 1 for col in range(3))
                dispersed += (all(len(set(int(x) for x in grid[row])) > 1 for row in range(3))
                              and all(len(set(int(x) for x in grid[:, col])) > 1 for col in range(3)))
            assert _template_index(pair["output"], indicator) == mapping[count]
            assert _source_key(pair) not in _SOURCE_PAIR_KEYS
            if count == 2:
                two_profiles.add(profile)
            if count == 3:
                (train_three if index < len(train) else test_three).add(profile)
            if index >= len(train):
                test_counts.add(count)
                different_triples += count == 3 and frozenset(int(x) for x in np.unique(grid)) not in train_triples
                test_role_overlap += bool(set(int(x) for x in np.unique(grid)) & {background, indicator})
                balanced_tests += count == 3 and profile == (3, 3, 3)
    assert len(roles) >= 15 and len(mappings) >= 3
    assert len(palettes) >= 20 and len(layouts) >= 60
    assert two_profiles == set(_TWO_PROFILES)
    assert (3, 3, 3) in train_three and (3, 3, 3) in test_three
    assert len(train_three) >= 5 and len(test_three) >= 5
    assert test_counts == {1, 2, 3}
    assert overlap >= 4 and many_colors >= 15 and mixed_tests >= 8 and different_triples >= 4
    assert row_bands >= 5 and col_bands >= 5 and dispersed >= 10
    assert test_role_overlap >= 3 and balanced_tests >= 2
    assert source_mapping >= 15 and changed_mapping >= 3
    assert _signature(generate_puzzle(12345, 7, 3)) == _signature(generate_puzzle(12345, 7, 3))
    assert _signature(generate_puzzle(54321)) == _signature(generate_puzzle(54321))
    assert _signature(generate_puzzle(90001, 6, 2)) != _signature(generate_puzzle(90002, 6, 2))
    for seed, ntrain, ntest in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                                (2001, None, 1), (2002, None, 4),
                                (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (True, 1), (4, 1.5)):
        try:
            generate_puzzle(3000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {args}")
