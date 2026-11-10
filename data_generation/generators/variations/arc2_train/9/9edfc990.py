import numpy as np


_SOURCE_INPUT_ROWS = (
    (
        "002096555021000",
        "304490003900050",
        "892010680008080",
        "604040016169142",
        "777300640401300",
        "760402004080076",
        "004783040050603",
        "080020001020010",
        "331020060860000",
        "000008000904028",
        "004149070105008",
        "720045109006400",
        "000096313390005",
        "050407990000940",
        "098806800080000",
    ),
    (
        "9000028090209",
        "1006000000095",
        "9049300570808",
        "0086060100041",
        "3601039004572",
        "0800000071800",
        "9002000757084",
        "0008750070050",
        "9900005005000",
        "8008065000090",
        "4006079980573",
        "0000000720008",
        "0007505000003",
    ),
    (
        "7440400611100605",
        "1133403853450828",
        "8048850900055858",
        "0260000301080408",
        "8028070009073036",
        "0000005306060457",
        "6600310025000345",
        "7078010009073030",
        "0600566598390700",
        "7500080690071000",
        "6534306941892877",
        "8686327302002100",
        "9000618033020214",
        "0400001000604456",
        "0508321059187027",
        "0901860998090030",
    ),
    (
        "0008050090600005",
        "6760402008362000",
        "0007005410176000",
        "0580900220804007",
        "4004227320649990",
        "0180500020080966",
        "9902800300200580",
        "1301610008000400",
        "0040740040058040",
        "0006060000080144",
        "0900900000150600",
        "6075907000460280",
        "5000102480030908",
        "1002400017000050",
        "6900771020091030",
        "1830097027208900",
    ),
)


def _validate_count(name, value, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(seed, num_train, num_test):
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)
    basis = abs(int(seed))

    if num_train is None and num_test is None:
        num_test = 1 + basis % 4
        lower = max(2, num_test + 1)
        num_train = lower + (basis // 4) % (11 - lower)
    elif num_train is None:
        num_test = int(num_test)
        lower = max(2, num_test + 1)
        num_train = lower + basis % (11 - lower)
    elif num_test is None:
        num_train = int(num_train)
        upper = min(4, num_train - 1)
        num_test = 1 + basis % upper
    else:
        num_train = int(num_train)
        num_test = int(num_test)

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _label_components(mask):
    height, width = mask.shape
    labels = np.full((height, width), -1, dtype=np.int16)
    components = []

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or labels[row, col] != -1:
                continue

            label = len(components)
            labels[row, col] = label
            stack = [(row, col)]
            cells = []

            while stack:
                r, c = stack.pop()
                cells.append((r, c))

                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and mask[nr, nc]
                        and labels[nr, nc] == -1
                    ):
                        labels[nr, nc] = label
                        stack.append((nr, nc))

            components.append(cells)

    return labels, components


def _flood_fill_output(input_grid, passable_color, seed_color):
    output_grid = input_grid.copy()
    traversable = (input_grid == passable_color) | (input_grid == seed_color)
    _, components = _label_components(traversable)

    for component in components:
        if any(input_grid[r, c] == seed_color for r, c in component):
            for r, c in component:
                if input_grid[r, c] == passable_color:
                    output_grid[r, c] = seed_color

    return output_grid


def _has_remote_change(input_grid, change_mask, seed_color):
    height, width = input_grid.shape

    for r, c in np.argwhere(change_mask):
        has_adjacent_seed = False

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = int(r) + dr, int(c) + dc
            if (
                0 <= nr < height
                and 0 <= nc < width
                and input_grid[nr, nc] == seed_color
            ):
                has_adjacent_seed = True
                break

        if not has_adjacent_seed:
            return True

    return False


def _has_diagonal_contrast(labels, seeded_components):
    height, width = labels.shape
    seeded_components = set(seeded_components)

    for r in range(height - 1):
        for c in range(width):
            for dc in (-1, 1):
                nc = c + dc
                if not 0 <= nc < width:
                    continue

                first = int(labels[r, c])
                second = int(labels[r + 1, nc])

                if (
                    first >= 0
                    and second >= 0
                    and first != second
                    and (
                        (first in seeded_components)
                        != (second in seeded_components)
                    )
                ):
                    return True

    return False


def _touches_mask(cell_mask, target_mask):
    height, width = cell_mask.shape

    for r, c in np.argwhere(cell_mask):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = int(r) + dr, int(c) + dc
            if (
                0 <= nr < height
                and 0 <= nc < width
                and target_mask[nr, nc]
            ):
                return True

    return False


def _mask_for_mode(rng, shape, mode):
    height, width = shape
    if mode == "mosaic":
        density = float(rng.uniform(0.34, 0.66))
        return rng.random(shape) < density

    if mode == "walls":
        mask = rng.random(shape) < float(rng.uniform(0.03, 0.15))
        directions = ["vertical", "horizontal"]
        rng.shuffle(directions)
        for direction in directions:
            if direction == "vertical":
                col = int(rng.integers(3, width - 3))
                mask[:, col] = True
                if rng.random() < 0.6:
                    row = int(rng.integers(2, height - 2))
                    mask[row, col] = False
            else:
                row = int(rng.integers(3, height - 3))
                mask[row, :] = True
                if rng.random() < 0.6:
                    col = int(rng.integers(2, width - 2))
                    mask[row, col] = False
        if height >= 13 and width >= 13 and rng.random() < 0.7:
            top = int(rng.integers(1, height - 5))
            left = int(rng.integers(1, width - 5))
            mask[top:top + 5, left:left + 5] = True
            mask[top + 1:top + 4, left + 1:left + 4] = False
        return mask

    # A carved, branching passage with isolated passable pockets.
    mask = np.ones(shape, dtype=bool)
    mask[1:5, 1:5] = False
    mask[height - 5:height - 1, width - 5:width - 1] = False
    bend_col = int(rng.integers(max(5, width // 3), width - 3))
    mask[3, 3:bend_col + 1] = False
    mask[3:height - 3, bend_col] = False
    mask[height - 4, bend_col:width - 3] = False
    branch_col = int(rng.integers(4, bend_col + 1))
    mask[3:min(height - 2, 7), branch_col] = False
    mask[1:3, width - 3:width - 1] = False
    mask[height - 3:height - 1, 1:3] = False
    for _ in range(int(rng.integers(0, 4))):
        row = int(rng.integers(1, height - 1))
        col = int(rng.integers(1, width - 1))
        if not mask[row, col]:
            mask[max(0, row - 1):min(height, row + 2),
                 max(0, col - 1):min(width, col + 2)] = False
    return mask


def _choose_components(rng, components, policy):
    sizes = [len(component) for component in components]
    positive = [index for index, size in enumerate(sizes) if size >= 2]
    if not positive:
        return None
    largest = int(np.argmax(sizes))

    if policy == "single":
        pool = [index for index in positive if index != largest]
        if pool and rng.random() < 0.7:
            return [int(rng.choice(pool))]
        return [int(rng.choice(positive))]

    if policy == "largest_unseeded":
        pool = [index for index in positive if index != largest]
        if not pool or sizes[largest] < 6:
            return None
        rng.shuffle(pool)
        return pool[:int(rng.integers(1, min(4, len(pool)) + 1))]

    if policy == "many":
        if len(components) < 6 or len(positive) < 3:
            return None
        pool = list(range(len(components)))
        rng.shuffle(pool)
        amount = int(rng.integers(5, min(11, len(pool)) + 1))
        selected = pool[:amount]
        if not any(sizes[index] >= 2 for index in selected):
            selected[0] = int(rng.choice(positive))
        return sorted(set(selected))

    if policy == "isolates":
        singles = [index for index, size in enumerate(sizes) if size == 1]
        if len(singles) < 2:
            return None
        main = [index for index in positive if sizes[index] >= 4]
        if not main:
            return None
        rng.shuffle(singles)
        return [int(rng.choice(main))] + singles[:min(len(singles),
            int(rng.integers(2, 5)))]

    pool = list(range(len(components)))
    rng.shuffle(pool)
    amount = int(rng.integers(1, min(9, len(pool)) + 1))
    selected = pool[:amount]
    if not any(sizes[index] >= 2 for index in selected):
        selected[0] = int(rng.choice(positive))
    return sorted(set(selected))


def _paint_pair(rng, barrier_mask, components, chosen, roles,
                barrier_palette, seed_style="mixed"):
    passable, seed = roles
    chosen = set(chosen)
    seed_mask = np.zeros(barrier_mask.shape, dtype=bool)
    for index in chosen:
        cells = components[index]
        length = len(cells)
        if length == 1:
            count = 1
        elif seed_style == "many" and length >= 10:
            count = int(rng.integers(5, min(12, length - 1) + 1))
        elif length >= 6 and rng.random() < 0.25:
            count = int(rng.integers(3, min(12, length - 1) + 1))
        else:
            count = int(rng.integers(1, min(3, length - 1) + 1))
        positions = rng.choice(length, size=count, replace=False)
        for position in np.atleast_1d(positions):
            row, col = cells[int(position)]
            seed_mask[row, col] = True

    grid = np.full(barrier_mask.shape, passable, dtype=np.int8)
    palette = np.asarray(barrier_palette, dtype=np.int8)
    grid[barrier_mask] = rng.choice(palette, size=int(barrier_mask.sum()))
    grid[seed_mask] = seed
    output = _flood_fill_output(grid, passable, seed)
    return grid, output


def _sample_shape(rng, regime, teaching=False):
    if teaching:
        height = int(rng.integers(14, 21))
        width = int(rng.integers(14, 21))
    else:
        band = int(rng.integers(0, 3))
        low, high = ((9, 15), (13, 21), (19, 27))[band]
        height = int(rng.integers(low, high))
        width = int(rng.integers(low, high))
        if rng.random() < 0.35:
            width = height
        elif regime == "target_test" and rng.random() < 0.25:
            width = int(rng.integers(9, 27))
    return height, width


def _sample_palette(rng, available, teaching=False):
    if teaching:
        count = int(rng.integers(2, min(5, len(available)) + 1))
    else:
        count = int(rng.integers(1, min(8, len(available)) + 1))
    return tuple(int(value) for value in rng.choice(
        available, size=count, replace=False))


def _pair_features(grid, output, roles):
    passable, seed = roles
    labels, components = _label_components(
        (grid == passable) | (grid == seed))
    seeded = {index for index, cells in enumerate(components)
              if any(grid[r, c] == seed for r, c in cells)}
    seedless = set(range(len(components))) - seeded
    changes = grid != output
    return labels, components, seeded, seedless, changes


def _generate_pair(rng, regime, roles, available, teaching=False,
                   forced_policy=None, forced_mode=None, shape=None):
    for attempt in range(180):
        this_shape = shape if shape is not None else _sample_shape(
            rng, regime, teaching)
        mode_weights = ([0.48, 0.30, 0.22] if regime == "source_train"
                        else [0.38, 0.27, 0.35])
        mode = forced_mode or str(rng.choice(
            ["mosaic", "walls", "corridor"], p=mode_weights))
        if teaching:
            mode = "mosaic"
        mask = _mask_for_mode(rng, this_shape, mode)
        _, components = _label_components(~mask)
        if len(components) < 2:
            continue
        policy = forced_policy or str(rng.choice(
            ["single", "largest_unseeded", "many", "isolates", "mixed"],
            p=[0.19, 0.23, 0.19, 0.14, 0.25]))
        if teaching:
            policy = "mixed"
        chosen = _choose_components(rng, components, policy)
        if not chosen:
            continue
        palette = _sample_palette(rng, available, teaching)
        grid, output = _paint_pair(rng, mask, components, chosen,
                                   roles, palette,
                                   "many" if rng.random() < 0.25 else "mixed")
        labels, found, seeded, seedless, changes = _pair_features(
            grid, output, roles)
        if not np.any(changes):
            continue
        if teaching:
            if (not seedless or max(len(found[i]) for i in seedless) < 3
                    or not _has_remote_change(grid, changes, roles[1])
                    or not _has_diagonal_contrast(labels, seeded)):
                continue
            adjacent = {int(grid[r, c]) for r, c in np.argwhere(
                (grid != roles[0]) & (grid != roles[1]))
                if any(0 <= r + dr < grid.shape[0]
                       and 0 <= c + dc < grid.shape[1]
                       and changes[r + dr, c + dc]
                       for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))}
            if len(adjacent) < 2:
                continue
        if policy == "largest_unseeded" and max(
                len(found[i]) for i in seeded) >= max(
                len(found[i]) for i in seedless):
            continue
        if policy == "many" and len(seeded) < 5:
            continue
        if policy == "isolates" and not any(
                len(found[i]) == 1 for i in seeded):
            continue
        return grid, output

    # A bounded deterministic escape route with the same teaching evidence.
    if teaching:
        mask = np.ones((17, 19), dtype=bool)
        mask[1:6, 1:6] = False
        mask[6:10, 6:10] = False
        mask[11:15, 11:15] = False
        palette = _sample_palette(rng, available, teaching=True)
        grid = np.full(mask.shape, roles[0], dtype=np.int8)
        grid[mask] = rng.choice(np.asarray(palette, dtype=np.int8),
                                size=int(mask.sum()))
        grid[1, 1] = roles[1]
        grid[12, 12] = roles[1]
        grid[0, 2] = palette[0]
        grid[2, 0] = palette[1]
        return grid, _flood_fill_output(grid, *roles)

    # Other fallback pairs use four partitioned rooms.
    height, width = shape if shape is not None else (17, 19)
    mask = np.zeros((height, width), dtype=bool)
    mask[:, width // 2] = True
    mask[height // 2, :] = True
    mask[height // 2, width // 2] = True
    _, components = _label_components(~mask)
    chosen = [0, 1]
    return _paint_pair(rng, mask, components, chosen, roles,
                       _sample_palette(rng, available, teaching))


def _controlled_pairs(rng, roles, available, kind):
    height = int(rng.integers(12, 24))
    width = int(rng.integers(13, 26))
    wall = width // 2
    door = height // 2
    mask = rng.random((height, width)) < float(rng.uniform(0.02, 0.10))
    mask[:, wall] = True
    mask[door, 1:width - 1] = False
    mask[door, wall] = True
    mask[door - 1:door + 2, 2:wall - 1] = False
    mask[door - 1:door + 2, wall + 2:width - 2] = False
    palette = _sample_palette(rng, available)
    barrier_colors = np.asarray(palette, dtype=np.int8)
    base = np.full((height, width), roles[0], dtype=np.int8)
    base[mask] = rng.choice(barrier_colors, size=int(mask.sum()))
    left_seed = (door, 2)
    right_seed = (door, width - 3)
    pairs = []
    for variant in range(2):
        grid = base.copy()
        if kind == "seed_move":
            seed_position = left_seed if variant == 0 else right_seed
            grid[seed_position] = roles[1]
        else:
            grid[left_seed] = roles[1]
            if variant == 1:
                grid[door, wall] = roles[0]
        pairs.append((grid, _flood_fill_output(grid, *roles)))
    return pairs


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _topology_key(grid, passable, seed):
    return grid.shape, ((grid == passable) | (grid == seed)).tobytes()


def _is_source_input(grid):
    for rows in _SOURCE_INPUT_ROWS:
        if grid.shape != (len(rows), len(rows[0])):
            continue
        observed = tuple("".join(str(int(value)) for value in row)
                         for row in grid)
        if observed == rows:
            return True
    return False


def _transition_set(pairs):
    return {(int(before), int(after))
            for pair in pairs
            for before, after in zip(
                pair["input"][pair["input"] != pair["output"]],
                pair["output"][pair["input"] != pair["output"]])}


def _infer_roles_from_training(training_pairs):
    transitions = _transition_set(training_pairs)
    assert len(transitions) == 1
    passable, seed = next(iter(transitions))
    assert passable != seed
    return passable, seed


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    colors = rng.permutation(10)
    roles = (int(colors[0]), int(colors[1]))
    available = tuple(int(value) for value in colors[2:])
    puzzle = {"train": [], "test": []}
    seen = set()

    def add(split, pair):
        grid, output = pair
        key = _grid_key(grid)
        if key in seen or _is_source_input(grid):
            return False
        seen.add(key)
        puzzle[split].append({"input": grid, "output": output})
        return True

    teaching = _generate_pair(rng, "source_train", roles, available,
                              teaching=True)
    assert add("train", teaching)

    if train_count >= 3 and rng.random() < 0.72:
        kind = str(rng.choice(["seed_move", "gate_toggle"]))
        for pair in _controlled_pairs(rng, roles, available, kind):
            assert add("train", pair)

    for split, count, regime in (("train", train_count, "source_train"),
                                  ("test", test_count, "target_test")):
        while len(puzzle[split]) < count:
            for _ in range(12):
                pair = _generate_pair(rng, regime, roles, available)
                if add(split, pair):
                    break
            else:
                raise AssertionError("could not create distinct pair")

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    pairs = puzzle["train"] + puzzle["test"]
    seen_inputs, seen_pairs = set(), set()
    for pair in pairs:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        grid, output = pair["input"], pair["output"]
        assert isinstance(grid, np.ndarray) and isinstance(output, np.ndarray)
        for item in (grid, output):
            assert item.ndim == 2
            assert all(1 <= size <= 30 for size in item.shape)
            assert np.issubdtype(item.dtype, np.integer)
            assert np.all((item >= 0) & (item <= 9))
        assert grid.shape == output.shape
        assert np.any(grid != output)
        assert not _is_source_input(grid)
        key = _grid_key(grid)
        pair_key = key, _grid_key(output)
        assert key not in seen_inputs and pair_key not in seen_pairs
        seen_inputs.add(key)
        seen_pairs.add(pair_key)

    roles = _infer_roles_from_training(puzzle["train"])
    assert _transition_set(pairs) == {roles}
    for pair in pairs:
        grid, output = pair["input"], pair["output"]
        assert np.array_equal(output, _flood_fill_output(grid, *roles))
        labels, components, seeded, seedless, changes = _pair_features(
            grid, output, roles)
        assert seeded
        assert np.all(grid[changes] == roles[0])
        assert np.all(output[changes] == roles[1])
        assert np.any(grid == roles[1])
        assert np.any(grid == roles[0])
        assert any(len(components[index]) >= 2 for index in seeded)
        assert np.array_equal(output[(grid != roles[0]) & (grid != roles[1])],
                              grid[(grid != roles[0]) & (grid != roles[1])])

    teaching = puzzle["train"][0]
    grid, output = teaching["input"], teaching["output"]
    labels, components, seeded, seedless, changes = _pair_features(
        grid, output, roles)
    assert max(len(components[index]) for index in seedless) >= 3
    assert _has_remote_change(grid, changes, roles[1])
    assert _has_diagonal_contrast(labels, seeded)
    boundary_colors = {int(grid[r, c]) for r, c in np.argwhere(
        (grid != roles[0]) & (grid != roles[1]))
        if any(0 <= r + dr < grid.shape[0]
               and 0 <= c + dc < grid.shape[1]
               and changes[r + dr, c + dc]
               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))}
    assert len(boundary_colors) >= 2


def _puzzles_equal(first, second):
    return all(np.array_equal(a[key], b[key])
               for split in ("train", "test")
               for a, b in zip(first[split], second[split])
               for key in ("input", "output"))


def self_test() -> None:
    first = generate_puzzle(123456, 6, 3)
    assert _puzzles_equal(first, generate_puzzle(123456, 6, 3))
    assert not _puzzles_equal(first, generate_puzzle(123457, 6, 3))
    train_counts, test_counts = set(), set()
    roles_seen, shapes, barrier_counts = set(), set(), set()
    seeded_counts, changed_counts, topology_keys = set(), set(), set()
    single_seeded = large_unseeded = isolated_seed = 0
    rectangles = remote = short_fill = 0
    controlled = gate_controls = new_test_hues = 0
    varied_palettes = short_with_isolate = redundant_seeds = 0
    sparse = dense = diagonal_remote = 0
    test_smaller = test_similar = test_denser = 0
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        roles = _infer_roles_from_training(puzzle["train"])
        roles_seen.add(roles)
        train_barriers = set().union(*(
            set(int(value) for value in np.unique(pair["input"]))
            - set(roles) for pair in puzzle["train"]))
        palettes_in_puzzle = set()
        train_grids = [pair["input"] for pair in puzzle["train"]]
        test_grids = [pair["input"] for pair in puzzle["test"]]
        train_area = float(np.median([grid.size for grid in train_grids]))
        test_area = float(np.median([grid.size for grid in test_grids]))
        test_smaller += test_area < train_area * 0.85
        test_similar += 0.85 <= test_area / train_area <= 1.15
        density = lambda grid: float(np.mean(
            (grid != roles[0]) & (grid != roles[1])))
        test_denser += (np.median([density(grid) for grid in test_grids])
                        > np.median([density(grid) for grid in train_grids])
                        + 0.05)
        train_topologies = [_topology_key(p["input"], *roles)
                            for p in puzzle["train"]]
        if len(set(train_topologies)) < len(train_topologies):
            controlled += 1
        for index, first_pair in enumerate(puzzle["train"]):
            for second_pair in puzzle["train"][index + 1:]:
                first_grid, second_grid = (first_pair["input"],
                                           second_pair["input"])
                if first_grid.shape != second_grid.shape:
                    continue
                first_barrier = ((first_grid != roles[0])
                                 & (first_grid != roles[1]))
                second_barrier = ((second_grid != roles[0])
                                  & (second_grid != roles[1]))
                if (np.count_nonzero(first_barrier != second_barrier) == 1
                        and np.array_equal(first_grid == roles[1],
                                           second_grid == roles[1])):
                    gate_controls += 1
                    break
        for pair_index, pair in enumerate(puzzle["train"] + puzzle["test"]):
            total += 1
            grid, output = pair["input"], pair["output"]
            assert np.array_equal(output, _flood_fill_output(grid, *roles))
            labels, components, seeded, seedless, changes = _pair_features(
                grid, output, roles)
            shapes.add(grid.shape)
            barrier_counts.add(len(set(int(v) for v in np.unique(grid))
                                   - set(roles)))
            barrier_palette = set(int(v) for v in np.unique(grid)) - set(roles)
            palettes_in_puzzle.add(tuple(sorted(barrier_palette)))
            if (pair_index >= len(puzzle["train"])
                    and barrier_palette - train_barriers):
                new_test_hues += 1
            seeded_counts.add(len(seeded))
            changed_counts.add(int(changes.sum()))
            topology_keys.add(_topology_key(grid, *roles))
            single_seeded += len(seeded) == 1
            large_unseeded += bool(seedless) and max(
                len(components[i]) for i in seedless) > max(
                len(components[i]) for i in seeded)
            isolated_seed += any(len(components[i]) == 1 for i in seeded)
            short_with_isolate += (int(changes.sum()) <= 4
                                  and any(len(components[i]) == 1
                                          for i in seeded))
            redundant_seeds += any(
                sum(grid[r, c] == roles[1] for r, c in components[i]) >= 10
                for i in seeded)
            rectangles += grid.shape[0] != grid.shape[1]
            remote += _has_remote_change(grid, changes, roles[1])
            short_fill += int(changes.sum()) <= 4
            diagonal_remote += (_has_diagonal_contrast(labels, seeded)
                                and _has_remote_change(grid, changes,
                                                       roles[1]))
            sparse += density(grid) < 0.25
            dense += density(grid) > 0.5
        varied_palettes += len(palettes_in_puzzle) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(roles_seen) >= 12 and len(shapes) >= 45
    assert len(topology_keys) >= total * 0.85
    assert len(seeded_counts) >= 6 and len(changed_counts) >= 20
    assert len(barrier_counts) >= 6
    assert min(h for h, w in shapes) <= 11
    assert max(h for h, w in shapes) >= 23
    assert single_seeded >= 10 and large_unseeded >= 10
    assert isolated_seed >= 10 and rectangles >= 30
    assert remote >= 30 and short_fill >= 5 and controlled >= 3
    assert gate_controls >= 3 and new_test_hues >= 3
    assert varied_palettes >= 35 and redundant_seeds >= 5
    assert short_with_isolate >= 3 and diagonal_remote >= 20
    assert sparse >= 20 and dense >= 20
    assert test_smaller >= 3 and test_similar >= 2 and test_denser >= 3

    for offset, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                            (None, 1), (None, 4),
                                            (2, None), (10, None))):
        puzzle = generate_puzzle(200000 + offset, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 3), (3, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(300000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
