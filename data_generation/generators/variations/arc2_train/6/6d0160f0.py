import numpy as np


# Exact original inputs are excluded; the original solved pairs also exercise
# the independent interpretation in self_test.
_SOURCE_INPUT_ROWS = (
    ("07050605700", "83650005080", "00050305000", "55555555555", "08750305007", "00058005086", "00650005300", "55555555555", "06050805000", "80053005408", "07050605067"),
    ("30050205060", "00750005009", "06050105000", "55555555555", "03050005000", "10950065073", "00050005000", "55555555555", "90050905090", "06050045001", "00057005000"),
    ("30050105002", "02050305060", "00050005000", "55555555555", "01050005030", "70652075070", "00050005060", "55555555555", "70050405003", "00050705200", "03050305006"),
    ("30057065807", "00950305060", "72050025030", "55555555555", "70258705023", "00650035007", "30052005060", "55555555555", "03450205207", "70057035001", "00250605030"),
    ("20352005030", "76050705670", "00056035002", "55555555555", "70050005604", "06050275020", "60250305070", "55555555555", "70050605230", "00650205000", "20050705067"),
)
_SOURCE_OUTPUT_ROWS = (
    ("00050005000", "00050005000", "00050005000", "55555555555", "00050005000", "40850005000", "06750005000", "55555555555", "00050005000", "00050005000", "00050005000"),
    ("00050005000", "00050005000", "00050005000", "55555555555", "00050005090", "00050005004", "00050005700", "55555555555", "00050005000", "00050005000", "00050005000"),
    ("00050405000", "00050705000", "00050305000", "55555555555", "00050005000", "00050005000", "00050005000", "55555555555", "00050005000", "00050005000", "00050005000"),
    ("00050005034", "00050005700", "00050005002", "55555555555", "00050005000", "00050005000", "00050005000", "55555555555", "00050005000", "00050005000", "00050005000"),
    ("00050005604", "00050005020", "00050005070", "55555555555", "00050005000", "00050005000", "00050005000", "55555555555", "00050005000", "00050005000", "00050005000"),
)
_SOURCE_KEYS = frozenset(
    (len(rows), len(rows[0]), tuple(int(c) for row in rows for c in row))
    for rows in _SOURCE_INPUT_ROWS
)


def _key(grid):
    return (grid.shape[0], grid.shape[1], tuple(int(x) for x in grid.flat))


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(rng, num_train, num_test):
    train = None if num_train is None else _count("num_train", num_train, 2, 10)
    test = None if num_test is None else _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _size_to_panels(length):
    for count in range(2, 6):
        if count * count + count - 1 == length:
            return count
    raise AssertionError("invalid lattice size")


def _mask(m, n):
    mask = np.zeros((m * m + m - 1, n * n + n - 1), dtype=bool)
    for k in range(1, m):
        mask[k * (m + 1) - 1, :] = True
    for k in range(1, n):
        mask[:, k * (n + 1) - 1] = True
    return mask


def _panel(position, m, n):
    row, col = position
    return (slice(row * (m + 1), row * (m + 1) + m),
            slice(col * (n + 1), col * (n + 1) + n))


def _blank(m, n, background, divider):
    grid = np.full((m * m + m - 1, n * n + n - 1), background, dtype=np.int8)
    grid[_mask(m, n)] = divider
    return grid


def _mode_background(grid, mask, divider):
    values = grid[~mask]
    counts = np.bincount(values.astype(np.int64), minlength=10)
    counts[divider] = 0
    winners = np.flatnonzero(counts == counts.max())
    assert len(winners) == 1
    return int(winners[0])


def _roles_from_input(grid):
    m = _size_to_panels(grid.shape[0])
    n = _size_to_panels(grid.shape[1])
    mask = _mask(m, n)
    divider_values = np.unique(grid[mask])
    assert len(divider_values) == 1
    divider = int(divider_values[0])
    assert not np.any(grid[~mask] == divider)
    background = _mode_background(grid, mask, divider)
    return m, n, mask, background, divider


def _relocate(input_grid, locator):
    m, n, mask, background, divider = _roles_from_input(input_grid)
    locations = np.argwhere(input_grid == locator)
    assert locations.shape == (1, 2)
    row, col = map(int, locations[0])
    assert not mask[row, col]
    source = (row // (m + 1), col // (n + 1))
    address = (row % (m + 1), col % (n + 1))
    assert 0 <= address[0] < m and 0 <= address[1] < n
    payload = input_grid[_panel(source, m, n)].copy()
    output = _blank(m, n, background, divider)
    output[_panel(address, m, n)] = payload
    return output, source, address, payload


def _locator_candidates(pair):
    grid = pair["input"]
    m, n, mask, background, divider = _roles_from_input(grid)
    counts = np.bincount(grid[~mask].astype(np.int64), minlength=10)
    candidates = set()
    for color in np.flatnonzero(counts == 1):
        color = int(color)
        if color in (background, divider):
            continue
        predicted, _, _, _ = _relocate(grid, color)
        if np.array_equal(predicted, pair["output"]):
            candidates.add(color)
    return candidates


def _learn_locator(train):
    candidates = set.intersection(*(_locator_candidates(pair) for pair in train))
    assert len(candidates) == 1, "training does not identify one locator color"
    return next(iter(candidates))


def _sample_positions(rng, m, n, train_count, test_count):
    positions = [(r, c) for r in range(m) for c in range(n)]
    source_order = [positions[int(i)] for i in rng.permutation(len(positions))]
    train_sources = source_order[:min(2, train_count)]
    train_sources += [positions[int(rng.integers(len(positions)))] for _ in range(train_count - len(train_sources))]

    rows = [int(x) for x in rng.permutation(m)]
    cols = [int(x) for x in rng.permutation(n)]
    train_addresses = []
    for i in range(train_count):
        if i < max(m, n):
            train_addresses.append((rows[i % m], cols[i % n]))
        else:
            train_addresses.append((int(rng.integers(m)), int(rng.integers(n))))
    # Shuffle pair order, so the first demonstrations have no fixed addresses.
    order = [int(i) for i in rng.permutation(train_count)]
    train_sources = [train_sources[i] for i in order]
    train_addresses = [train_addresses[i] for i in order]

    seen_rows = {r for r, _ in train_addresses}
    seen_cols = {c for _, c in train_addresses}
    address_pool = [(r, c) for r in seen_rows for c in seen_cols]
    rng.shuffle(address_pool)
    novel = [a for a in address_pool if a not in train_addresses]
    familiar = [a for a in address_pool if a in train_addresses]
    if novel and rng.random() < 0.8:
        first = novel.pop()
        test_addresses = [first]
    else:
        first = familiar.pop() if familiar else novel.pop()
        test_addresses = [first]
    remainder = novel + familiar
    rng.shuffle(remainder)
    test_addresses += remainder[:test_count - 1]
    assert len(test_addresses) == test_count

    train_combinations = set(zip(train_sources, train_addresses))
    test_sources = []
    for address in test_addresses:
        eligible = [p for p in positions if (p, address) not in train_combinations]
        new_sources = [p for p in eligible if p not in train_sources]
        familiar_sources = [p for p in eligible if p in train_sources]
        if new_sources and (not familiar_sources or rng.random() < 0.55):
            pool = new_sources
        else:
            pool = familiar_sources or eligible or positions
        test_sources.append(pool[int(rng.integers(len(pool)))])
    return train_sources, train_addresses, test_sources, test_addresses


def _draw_pair(rng, m, n, locator, background, divider, source, address,
               density_kind, reserved_color=None, force_reserved=False):
    auxiliaries = [c for c in range(10) if c not in (locator, background, divider)]
    if reserved_color in auxiliaries:
        auxiliaries.remove(reserved_color)
    palette_size = int(rng.integers(1, len(auxiliaries) + 1))
    chosen = [int(x) for x in rng.choice(auxiliaries, size=palette_size, replace=False)]
    grid = _blank(m, n, background, divider)
    payload = np.full((m, n), background, dtype=np.int8)
    payload[address] = locator
    available = [(r, c) for r in range(m) for c in range(n) if (r, c) != address]
    if density_kind == "rich":
        mark_count = max(2, int(round((m * n - 1) * float(rng.uniform(0.55, 0.9)))))
    elif density_kind == "sparse":
        mark_count = max(1, int(round((m * n - 1) * float(rng.uniform(0.05, 0.22)))))
    else:
        mark_count = int(rng.integers(1, m * n))
    mark_count = min(mark_count, len(available))
    for index in rng.choice(len(available), size=mark_count, replace=False):
        cell = available[int(index)]
        payload[cell] = chosen[int(rng.integers(len(chosen)))]
    grid[_panel(source, m, n)] = payload

    distractor_positions = [p for p in ((r, c) for r in range(m) for c in range(n)) if p != source]
    rng.shuffle(distractor_positions)
    for position in distractor_positions:
        roll = rng.random()
        if roll < 0.16:
            count = 0
        elif roll < 0.35:
            count = 1
        else:
            probability = float(rng.uniform(0.16, 0.78))
            count = int(rng.binomial(m * n, probability))
        if count:
            panel = grid[_panel(position, m, n)]
            for index in rng.choice(m * n, size=count, replace=False):
                panel.flat[int(index)] = chosen[int(rng.integers(len(chosen)))]
    # A sparse distractor supplies both a misleading singleton and an observed
    # test-only auxiliary when requested. Neither affects locator uniqueness.
    if force_reserved:
        position = distractor_positions[0]
        panel = grid[_panel(position, m, n)]
        cell = int(rng.integers(m * n))
        panel.flat[cell] = reserved_color
    elif rng.random() < 0.52:
        unused = [c for c in auxiliaries if not np.any(grid == c)]
        if unused:
            position = distractor_positions[0]
            panel = grid[_panel(position, m, n)]
            panel.flat[int(rng.integers(m * n))] = unused[int(rng.integers(len(unused)))]

    # The input background is always detectable from the input alone.
    mask = _mask(m, n)
    while True:
        counts = np.bincount(grid[~mask].astype(np.int64), minlength=10)
        if counts[background] > max(counts[c] for c in range(10) if c != background):
            break
        candidates = [(p, k) for p in distractor_positions for k in range(m * n)
                      if grid[_panel(p, m, n)].flat[k] != background]
        assert candidates
        p, k = candidates[int(rng.integers(len(candidates)))]
        grid[_panel(p, m, n)].flat[k] = background
    if not any(np.any(grid[_panel(p, m, n)] != background) for p in distractor_positions):
        grid[_panel(distractor_positions[0], m, n)].flat[0] = chosen[0]
    output = _blank(m, n, background, divider)
    output[_panel(address, m, n)] = payload
    return {"input": grid, "output": output}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    m = int(rng.integers(2, 6))
    n = int(rng.integers(2, 6))
    locator = int(rng.integers(10))
    fixed_roles = bool(rng.integers(2))
    other_colors = [c for c in range(10) if c != locator]
    fixed_pair = [int(x) for x in rng.choice(other_colors, size=2, replace=False)]
    train_sources, train_addresses, test_sources, test_addresses = _sample_positions(
        rng, m, n, train_count, test_count)
    # Sometimes introduce an auxiliary color first in the test. It carries no
    # convention: the marker is established by training and remains unique.
    reserve = int(rng.choice([c for c in other_colors if c not in fixed_pair])) if rng.random() < 0.38 else None
    kinds = ["rich", "sparse"] + ["mixed"] * (train_count - 2)
    rng.shuffle(kinds)
    puzzle = {"train": [], "test": []}
    used = set()
    for split, count, sources, addresses in (
        ("train", train_count, train_sources, train_addresses),
        ("test", test_count, test_sources, test_addresses),
    ):
        for i in range(count):
            for attempt in range(16):
                if fixed_roles:
                    background, divider = fixed_pair
                else:
                    available = [c for c in range(10) if c != locator and c != reserve]
                    background, divider = [int(x) for x in rng.choice(available, size=2, replace=False)]
                pair = _draw_pair(
                    rng, m, n, locator, background, divider, sources[i], addresses[i],
                    kinds[i] if split == "train" else "mixed",
                    reserve,
                    split == "test" and i == 0 and reserve is not None,
                )
                key = _key(pair["input"])
                if key not in used and key not in _SOURCE_KEYS:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    dimensions = set()
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.all((grid >= 0) & (grid <= 9))
        input_grid, output_grid = pair["input"], pair["output"]
        assert input_grid.shape == output_grid.shape
        dimensions.add(input_grid.shape)
        input_key, output_key = _key(input_grid), _key(output_grid)
        assert input_key not in seen_inputs and (input_key, output_key) not in seen_pairs
        assert input_key not in _SOURCE_KEYS
        seen_inputs.add(input_key)
        seen_pairs.add((input_key, output_key))
        m, n, mask, background, divider = _roles_from_input(input_grid)
        assert np.all(output_grid[mask] == divider)
        assert not np.any(output_grid[~mask] == divider)
        assert _mode_background(output_grid, mask, divider) == background
        assert not np.array_equal(input_grid, output_grid)
    assert len(dimensions) == 1
    locator = _learn_locator(train)
    train_sources, train_addresses = [], []
    for index, pair in enumerate(train + test):
        predicted, source, address, payload = _relocate(pair["input"], locator)
        assert np.array_equal(predicted, pair["output"])
        assert np.count_nonzero(pair["input"] == locator) == 1
        assert np.count_nonzero(pair["output"] == locator) == 1
        m, n, _, background, _ = _roles_from_input(pair["input"])
        assert any(np.any(pair["input"][_panel((r, c), m, n)] != background)
                   for r in range(m) for c in range(n) if (r, c) != source)
        if index < len(train):
            train_sources.append(source)
            train_addresses.append(address)
    assert len(set(train_sources)) >= 2
    assert len({r for r, _ in train_addresses}) >= 2
    assert len({c for _, c in train_addresses}) >= 2
    trained_rows = {r for r, _ in train_addresses}
    trained_cols = {c for _, c in train_addresses}
    test_addresses = [_relocate(pair["input"], locator)[2] for pair in test]
    assert len(set(test_addresses)) == len(test_addresses)
    assert all(r in trained_rows and c in trained_cols for r, c in test_addresses)
    # Inputs are distinct even if a small lattice has exhausted a particular
    # source/address combination across many demonstrations.


def _equal_puzzles(left, right):
    return all(len(left[split]) == len(right[split]) and all(
        np.array_equal(a[name], b[name])
        for a, b in zip(left[split], right[split]) for name in ("input", "output"))
        for split in ("train", "test"))


def self_test() -> None:
    source_pairs = []
    for input_rows, output_rows in zip(_SOURCE_INPUT_ROWS, _SOURCE_OUTPUT_ROWS):
        pair = {
            "input": np.array([[int(c) for c in row] for row in input_rows], dtype=np.int8),
            "output": np.array([[int(c) for c in row] for row in output_rows], dtype=np.int8),
        }
        source_pairs.append(pair)
        assert np.array_equal(_relocate(pair["input"], 4)[0], pair["output"])
    assert _learn_locator(source_pairs[:4]) == 4
    assert all(len(np.flatnonzero(np.bincount(source_pairs[i]["input"].ravel(),
                                                  minlength=10) == 1)) > 1
               for i in (1, 3))

    train_counts, test_counts, shapes, roles = set(), set(), set(), set()
    sources, addresses, singleton_decoys = set(), set(), 0
    empty_panels, one_mark_panels, dense_panels = 0, 0, 0
    familiar_test_source, new_test_source, novel_test_address = 0, 0, 0
    downward, sideways, stationary, test_only_colors = 0, 0, 0, 0
    palette_change, background_change, divider_change = 0, 0, 0
    sparse_payloads, rich_payloads, rectangular_decoys = 0, 0, 0
    combined_new_source_address = 0
    input_keys = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _equal_puzzles(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        locator = _learn_locator(puzzle["train"])
        train_info = [_relocate(p["input"], locator) for p in puzzle["train"]]
        test_info = [_relocate(p["input"], locator) for p in puzzle["test"]]
        train_source_set = {x[1] for x in train_info}
        train_address_set = {x[2] for x in train_info}
        pair_roles = [_roles_from_input(p["input"]) for p in puzzle["train"] + puzzle["test"]]
        background_change += len({x[3] for x in pair_roles}) > 1
        divider_change += len({x[4] for x in pair_roles}) > 1
        train_palettes = []
        for p in puzzle["train"]:
            _, _, mask, background, divider = _roles_from_input(p["input"])
            train_palettes.append(frozenset(int(c) for c in p["input"][~mask]
                                            if int(c) not in (background, divider, locator)))
        palette_change += len(set(train_palettes)) > 1
        for p, info in zip(puzzle["test"], test_info):
            familiar_test_source += info[1] in train_source_set
            new_test_source += info[1] not in train_source_set
            novel_test_address += info[2] not in train_address_set
            combined_new_source_address += (info[1] not in train_source_set
                                            and info[2] not in train_address_set)
            train_colors = set(int(v) for q in puzzle["train"] for v in q["input"].flat)
            test_only_colors += bool(set(int(v) for v in p["input"].flat) - train_colors)
        for pair in puzzle["train"] + puzzle["test"]:
            grid = pair["input"]
            assert _key(grid) not in input_keys
            input_keys.add(_key(grid))
            m, n, mask, background, divider = _roles_from_input(grid)
            shapes.add(grid.shape)
            roles.add((background, divider, locator))
            counts = np.bincount(grid[~mask].astype(np.int64), minlength=10)
            singleton_decoys += np.count_nonzero(counts == 1) > 1
            rectangular_decoys += m != n and np.count_nonzero(counts == 1) > 1
            predicted, source, address, payload = _relocate(grid, locator)
            assert np.array_equal(predicted, pair["output"])
            payload_marks = int(np.count_nonzero(payload != background))
            sparse_payloads += payload_marks <= max(3, m * n // 5)
            rich_payloads += payload_marks >= max(3, 3 * m * n // 5)
            sources.add(source)
            addresses.add(address)
            downward += address[0] > source[0]
            sideways += address[1] != source[1]
            stationary += address == source
            for r in range(m):
                for c in range(n):
                    if (r, c) == source:
                        continue
                    marks = int(np.count_nonzero(grid[_panel((r, c), m, n)] != background))
                    empty_panels += marks == 0
                    one_mark_panels += marks == 1
                    dense_panels += marks >= max(2, (m * n) // 2)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 10 and any(h != w for h, w in shapes)
    assert (5, 5) in shapes and (29, 29) in shapes
    assert len(roles) >= 30 and len(sources) >= 15 and len(addresses) >= 15
    assert singleton_decoys >= 20
    assert min(empty_panels, one_mark_panels, dense_panels) >= 10
    assert min(familiar_test_source, new_test_source, novel_test_address) >= 3
    assert min(downward, sideways, stationary, test_only_colors) >= 3
    assert min(palette_change, background_change, divider_change) >= 10
    assert min(sparse_payloads, rich_payloads, rectangular_decoys,
               combined_new_source_address) >= 5

    for train, test in ((2, 1), (10, 1), (10, 4), (7, 3)):
        p = generate_puzzle(1000 + train * 10 + test, train, test)
        assert (len(p["train"]), len(p["test"])) == (train, test)
        validate_generated_puzzle(p)
        assert _equal_puzzles(p, generate_puzzle(1000 + train * 10 + test, train, test))
    for i, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(2000 + i, train, test)
        assert train is None or len(p["train"]) == train
        assert test is None or len(p["test"]) == test
        validate_generated_puzzle(p)
    for i, args in enumerate(((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 3), (2, 4))):
        try:
            generate_puzzle(3000 + i, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert not _equal_puzzles(generate_puzzle(5001), generate_puzzle(5002))
