"""ARC puzzles with left-edge markers and learned color destinations."""
import numpy as np

_SOURCE_INPUTS = (
    ("2000000000", "8000000000", "0000000000", "3000000000", "2000000000", "4000000000", "0000000000", "8000000000", "0000000000", "3000000000"),
    ("3000000000", "4000000000", "2000000000", "3000000000", "0000000000", "4000000000", "8000000000", "2000000000", "0000000000", "0000000000"),
    ("8000000000", "3000000000", "2000000000", "4000000000", "3000000000", "8000000000", "0000000000", "3000000000", "8000000000", "0000000000"),
    ("2000000000", "4000000000", "3000000000", "2000000000", "0000000000", "8000000000", "3000000000", "0000000000", "4000000000", "2000000000"),
)
_SOURCE_MAP = {8: 1, 2: 2, 4: 3, 3: 4}


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(v) for v in grid.flat))


def _pair_key(pair):
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _source_keys():
    keys = set()
    for rows in _SOURCE_INPUTS:
        inp = np.array([[int(v) for v in row] for row in rows], dtype=np.int8)
        out = np.zeros_like(inp)
        for r, color in enumerate(inp[:, 0]):
            if int(color) in _SOURCE_MAP:
                out[r, _SOURCE_MAP[int(color)]] = color
        keys.add((_grid_key(inp), _grid_key(out)))
    return frozenset(keys)


_SOURCE_KEYS = _source_keys()


def _counts(seed, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    train = None if num_train is None else int(num_train)
    test = None if num_test is None else int(num_test)
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


def _parameters(rng):
    role_count = int(rng.choice([2, 3, 4, 5], p=[0.14, 0.23, 0.45, 0.18]))
    background_count = 2 if rng.random() < 0.38 else 1
    palette = [int(x) for x in rng.permutation(10)]
    backgrounds = tuple(palette[:background_count])
    colors = tuple(palette[background_count:background_count + role_count])
    if rng.random() < 0.4:
        destinations = tuple(range(1, role_count + 1))
    else:
        farthest = int(rng.integers(max(role_count + 2, 7), 21))
        nearby = [int(x) for x in rng.choice(np.arange(1, farthest), size=role_count - 1, replace=False)]
        destinations = tuple(sorted(nearby + [farthest]))
    return {"colors": colors, "destinations": destinations, "backgrounds": backgrounds}


def _train_role_sets(rng, count, role_count):
    sets = [set() for _ in range(count)]
    if rng.random() < 0.27:
        sets[0].update(range(role_count))
    else:
        order = [int(x) for x in rng.permutation(role_count)]
        for index, role in enumerate(order):
            sets[index % count].add(role)
    for roles in sets:
        if not roles:
            roles.add(int(rng.integers(role_count)))
        if rng.random() < 0.36:
            extras = int(rng.integers(1, min(3, role_count) + 1))
            roles.update(int(x) for x in rng.choice(role_count, size=extras, replace=False))
    return [tuple(sorted(roles)) for roles in sets]


def _test_role_set(rng, role_count):
    size = role_count if rng.random() < 0.32 else int(rng.integers(1, role_count + 1))
    return tuple(sorted(int(x) for x in rng.choice(role_count, size=size, replace=False)))


def _sample_pair(rng, parameters, roles, regime, background):
    destinations, colors = parameters["destinations"], parameters["colors"]
    if regime == "source_train":
        extra = 0 if rng.random() < 0.07 else int(rng.integers(1, 9))
    else:
        extra = 0 if rng.random() < 0.30 else int(rng.integers(0, 11))
    row_roles = list(roles)
    row_roles.extend(int(x) for x in rng.choice(roles, size=extra, replace=True))
    rng.shuffle(row_roles)
    marker_count = len(row_roles)
    long_chance = 0.43 if regime == "target_test" else 0.32
    selector = rng.random()
    if selector < 0.21:
        blanks = 0
    elif selector < 1 - long_chance:
        blanks = int(rng.integers(1, 5))
    else:
        blanks = int(rng.integers(5, 19))
    blanks = min(blanks, 30 - marker_count)
    height = marker_count + blanks
    minimum_width = max(5, max(destinations) + 1)
    width = int(rng.integers(minimum_width, min(30, minimum_width + 12) + 1))
    if len(colors) == 4 and destinations == (1, 2, 3, 4) and rng.random() < 0.17:
        width = 10
        if marker_count <= 10:
            height = 10
            blanks = height - marker_count
    if blanks >= 5 and rng.random() < 0.43:
        insertion = int(rng.integers(0, marker_count + 1))
        layout = row_roles[:insertion] + [-1] * blanks + row_roles[insertion:]
    else:
        positions = [int(x) for x in rng.choice(height, size=marker_count, replace=False)]
        if marker_count >= 2 and height > marker_count and rng.random() < 0.35:
            positions = [0, height - 1] + [int(x) for x in rng.choice(np.arange(1, height - 1), size=marker_count - 2, replace=False)]
        rng.shuffle(row_roles)
        layout = [-1] * height
        for position, role in zip(positions, row_roles):
            layout[position] = role
    inp = np.full((height, width), background, dtype=np.int8)
    out = np.full((height, width), background, dtype=np.int8)
    for row, role in enumerate(layout):
        if role >= 0:
            inp[row, 0] = colors[role]
            out[row, destinations[role]] = colors[role]
    return {"input": inp, "output": out}


def _unique_pair(rng, parameters, roles, regime, background, used):
    for _ in range(96):
        pair = _sample_pair(rng, parameters, roles, regime, background)
        key = _pair_key(pair)
        if key[0] not in used and key not in _SOURCE_KEYS:
            used.add(key[0])
            return pair
    for offset in range(30):
        inp = np.full((30, 30), background, dtype=np.int8)
        out = np.full((30, 30), background, dtype=np.int8)
        for index, role in enumerate(roles):
            row = (offset + index * 3) % 30
            color = parameters["colors"][role]
            inp[row, 0] = color
            out[row, parameters["destinations"][role]] = color
        pair = {"input": inp, "output": out}
        key = _pair_key(pair)
        if key[0] not in used:
            used.add(key[0])
            return pair
    raise RuntimeError("could not construct a unique pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    parameters = _parameters(rng)
    train_sets = _train_role_sets(rng, train_count, len(parameters["colors"]))
    puzzle = {"train": [], "test": []}
    used = set()
    for split, sets in (("train", train_sets),
                        ("test", [_test_role_set(rng, len(parameters["colors"])) for _ in range(test_count)])):
        for index, roles in enumerate(sets):
            backgrounds = parameters["backgrounds"]
            background = (backgrounds[index] if split == "train" and index < len(backgrounds)
                          else backgrounds[int(rng.integers(len(backgrounds)))])
            entropy = int(rng.integers(0, np.iinfo(np.int64).max))
            pair_rng = np.random.default_rng(np.random.SeedSequence([seed, entropy, index, 0 if split == "train" else 1]))
            pair = _unique_pair(pair_rng, parameters, roles,
                                "source_train" if split == "train" else "target_test", background, used)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    assert np.count_nonzero(inp == background) > inp.size // 2
    assert np.count_nonzero(out == background) == np.count_nonzero(inp == background)
    assert np.all(inp[:, 1:] == background)
    mapping, frequencies, marker_rows = {}, {}, []
    for row in range(inp.shape[0]):
        color = int(inp[row, 0])
        changed = np.flatnonzero(out[row] != background)
        if color == background:
            assert len(changed) == 0
            continue
        assert len(changed) == 1
        destination = int(changed[0])
        assert 1 <= destination < inp.shape[1] and int(out[row, destination]) == color
        assert color not in mapping or mapping[color] == destination
        mapping[color] = destination
        frequencies[color] = frequencies.get(color, 0) + 1
        marker_rows.append(row)
    assert marker_rows
    return background, mapping, frequencies, marker_rows


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, learned, backgrounds, test_colors = set(), {}, set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            key = _pair_key(pair)
            assert key[0] not in seen_inputs and key not in _SOURCE_KEYS
            seen_inputs.add(key[0])
            background, mapping, _, _ = _analyze_pair(pair)
            backgrounds.add(background)
            for color, destination in mapping.items():
                if split == "test":
                    test_colors.add(color)
                else:
                    assert color not in learned or learned[color] == destination
                    learned[color] = destination
                if color in learned:
                    assert learned[color] == destination
    assert 2 <= len(learned) <= 5 and len(set(learned.values())) == len(learned)
    assert not (backgrounds & set(learned)) and test_colors <= set(learned)


def _puzzle_key(puzzle):
    return tuple(tuple(_pair_key(pair) for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    explicit = generate_puzzle(12345, 7, 3)
    assert _puzzle_key(explicit) == _puzzle_key(generate_puzzle(12345, 7, 3))
    assert _puzzle_key(explicit) != _puzzle_key(generate_puzzle(12346, 7, 3))
    for seed, train, test in ((31, 2, 1), (32, 10, 1), (33, 10, 4),
                              (34, None, 1), (35, None, 4), (36, 2, None), (37, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(9, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    train_counts, test_counts, role_counts = set(), set(), set()
    destinations, backgrounds, heights, widths = set(), set(), set(), set()
    train_vocab_sizes, test_vocab_sizes, blanks_seen = set(), set(), set()
    marker_totals, structures = set(), set()
    distributed = full_first = zero_marker = edge_markers = long_run = pair_count = 0
    varied_background = varied_width = varied_height = 0
    gapped = consecutive = ten_by_ten = full_vocabulary_tests = 0
    simpler = comparable = denser = 0
    test_frequency_patterns = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        learned = {}
        for pair in puzzle["train"]:
            learned.update(_analyze_pair(pair)[1])
        role_counts.add(len(learned))
        destinations.add(tuple(sorted(learned.values())))
        first_colors = set(_analyze_pair(puzzle["train"][0])[1])
        full_first += first_colors == set(learned)
        distributed += first_colors != set(learned)
        ordered_destinations = tuple(sorted(learned.values()))
        gapped += ordered_destinations != tuple(range(1, len(learned) + 1))
        consecutive += ordered_destinations == tuple(range(1, len(learned) + 1))
        all_pairs = puzzle["train"] + puzzle["test"]
        varied_background += len({_analyze_pair(pair)[0] for pair in all_pairs}) > 1
        varied_width += len({pair["input"].shape[1] for pair in all_pairs}) > 1
        varied_height += len({pair["input"].shape[0] for pair in all_pairs}) > 1
        train_marker_counts = [len(_analyze_pair(pair)[3]) for pair in puzzle["train"]]
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pair_count += 1
                inp, out = pair["input"], pair["output"]
                bg, mapping, frequencies, rows = _analyze_pair(pair)
                backgrounds.add(bg)
                heights.add(inp.shape[0])
                widths.add(inp.shape[1])
                blanks = inp.shape[0] - len(rows)
                blanks_seen.add(blanks)
                marker_totals.add(len(rows))
                ten_by_ten += inp.shape == (10, 10) and len(learned) == 4
                edge_markers += rows[0] == 0 and rows[-1] == inp.shape[0] - 1
                long_run += any(all(r not in rows for r in range(start, start + 5))
                                for start in range(max(0, inp.shape[0] - 4)))
                zero_marker += 0 in mapping
                (train_vocab_sizes if split == "train" else test_vocab_sizes).add(len(mapping))
                assert all(learned[color] == destination for color, destination in mapping.items())
                expected = np.full_like(inp, bg)
                for row in rows:
                    color = int(inp[row, 0])
                    expected[row, learned[color]] = color
                assert np.array_equal(out, expected)
                structures.add((inp.shape, tuple(-1 if int(v) == bg else learned[int(v)] for v in inp[:, 0])))
                if split == "test":
                    full_vocabulary_tests += len(mapping) == len(learned)
                    test_frequency_patterns.add(tuple(sorted(frequencies.values())))
                    simpler += len(rows) < min(train_marker_counts)
                    comparable += min(train_marker_counts) <= len(rows) <= max(train_marker_counts)
                    denser += len(rows) > max(train_marker_counts)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert role_counts == {2, 3, 4, 5}
    assert len(destinations) >= 12 and any(max(d) >= 9 for d in destinations)
    assert len(backgrounds) >= 5 and zero_marker >= 5
    assert 0 in blanks_seen and max(blanks_seen) >= 12
    assert min(heights) <= 3 and max(heights) >= 20
    assert min(widths) <= 6 and max(widths) >= 20
    assert min(marker_totals) == 1 and max(marker_totals) >= 12
    assert 1 in train_vocab_sizes and 2 in train_vocab_sizes
    assert 1 in test_vocab_sizes and max(test_vocab_sizes) >= 4
    assert distributed >= 10 and full_first >= 3
    assert edge_markers >= 10 and long_run >= 10
    assert varied_background >= 5 and varied_width >= 35 and varied_height >= 35
    assert gapped >= 10 and consecutive >= 5 and ten_by_ten >= 2
    assert full_vocabulary_tests >= 10 and len(test_frequency_patterns) >= 20
    assert simpler >= 2 and comparable >= 10 and denser >= 2
    assert len(structures) >= pair_count * 0.85
