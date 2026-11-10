"""Generate paired-snapshot ARC change maps."""

import numpy as np


_SOURCE_INPUT_ROWS = {
    (
        "000400000050004000000", "000000800050000008000",
        "000000000050000000000", "090008000050900800000",
        "000000040050000000400", "040000000050400000000",
        "000000000050000000000", "000800000050008009000",
        "000000090050000000000", "000000000050000000000",
    ),
    (
        "000000000050000000000", "000000006050000000060",
        "003003000050030030000", "000000000050000000000",
        "000600000050006000000", "000000300050000000000",
        "000000000050000003000", "000000000050000000000",
        "030000060050300000600", "000000000050000000000",
    ),
    (
        "000000000050000000000", "000000080050000000800",
        "000800000050008000000", "030000000350300003003",
        "000003000050000000000", "000600000050006000000",
        "000000000050000000600", "000000000050000000000",
        "000800060050008000000", "000000000050000000000",
    ),
    (
        "000040000050000000000", "090000000050900004000",
        "000400090050004000900", "000000000050000000000",
        "000900700050009000000", "000000000050000070000",
        "040000000050400000000", "000000000050000000000",
        "000070040050000700400", "090000000050900000000",
    ),
}


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        if high is None:
            raise ValueError(f"{name} must be at least {low}")
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = None if num_train is None else _integer(num_train, "num_train", 2, 10)
    test = None if num_test is None else _integer(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    else:
        test = 1 + seed % min(4, train - 1)
    return train, test


def _key(grid):
    return (grid.shape, grid.astype(np.uint8, copy=False).tobytes())


def _is_source_input(grid):
    return grid.shape == (10, 21) and tuple(
        "".join(str(int(x)) for x in row) for row in grid
    ) in _SOURCE_INPUT_ROWS


def _output(first, second, background, departure, arrival):
    same = first == second
    old = (first != background) & (second == background)
    new = (first == background) & (second != background)
    assert np.all(same | old | new)
    result = np.full(first.shape, background, dtype=np.int8)
    result[same] = first[same]
    result[old] = departure
    result[new] = arrival
    return result


def _shape_event(rng, first, second, color, reserved, maximum_changes):
    height, width = first.shape
    candidates = []
    for axis in (0, 1):
        for length in (2, 3, 4):
            for row in range(height):
                for col in range(width):
                    old = tuple(
                        (row + k, col) if axis == 0 else (row, col + k)
                        for k in range(length)
                    )
                    for delta in (-1, 1):
                        new = tuple(
                            (r + delta, c) if axis == 0 else (r, c + delta)
                            for r, c in old
                        )
                        union = set(old) | set(new)
                        if all(0 <= r < height and 0 <= c < width for r, c in union):
                            if not union & reserved:
                                candidates.append((old, new))
    if maximum_changes >= 2:
        for row in range(height - 1):
            for col in range(width - 1):
                old = ((row, col), (row + 1, col), (row, col + 1), (row + 1, col + 1))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    new = tuple((r + dr, c + dc) for r, c in old)
                    union = set(old) | set(new)
                    if all(0 <= r < height and 0 <= c < width for r, c in union):
                        if not union & reserved:
                            candidates.append((old, new))
    if not candidates:
        raise ValueError("shape cannot fit")
    old, new = candidates[int(rng.integers(len(candidates)))]
    for cell in old:
        first[cell] = color
    for cell in new:
        second[cell] = color
    reserved.update(old)
    reserved.update(new)
    return len(set(old) - set(new))


def _point_event(rng, first, second, color, reserved):
    height, width = first.shape
    free = [(r, c) for r in range(height) for c in range(width) if (r, c) not in reserved]
    mode = int(rng.integers(3))
    candidates = []
    for index, old in enumerate(free):
        for new in free[index + 1:]:
            dr, dc = new[0] - old[0], new[1] - old[1]
            if (mode == 0 and dr == 0) or (mode == 1 and dc == 0) or (mode == 2 and dr != 0 and dc != 0):
                candidates.append((old, new))
                candidates.append((new, old))
    if not candidates:
        raise ValueError("point relocation cannot fit")
    old, new = candidates[int(rng.integers(len(candidates)))]
    first[old] = color
    second[new] = color
    reserved.update((old, new))


def _palette(rng, pool, forced=None):
    sizes = np.array([1, 2, 3, 4], dtype=np.int16)
    size = int(rng.choice(sizes, p=[0.12, 0.30, 0.38, 0.20]))
    size = min(size, len(pool) + (forced is not None))
    candidates = [int(x) for x in pool if int(x) != forced]
    count = size - (forced is not None)
    chosen = [int(x) for x in rng.choice(candidates, size=count, replace=False)] if count else []
    if forced is not None:
        chosen.append(int(forced))
    return tuple(chosen)


def _make_pair(rng, palette, orientation, height, width, background, divider,
               departure, arrival, force_change=False, force_shape=False):
    first = np.full((height, width), background, dtype=np.int8)
    second = np.full_like(first, background)
    reserved = set()
    change_count = int(rng.choice([0, 1, 2, 3, 4, 5], p=[0.12, 0.19, 0.23, 0.22, 0.15, 0.09]))
    if force_change:
        change_count = max(1, change_count)
    if force_shape:
        change_count = max(1, change_count)
    change_count = min(change_count, max(1, height * width // 7))
    remaining = change_count
    if remaining and (force_shape or rng.random() < 0.28):
        color = int(rng.choice(palette))
        remaining -= _shape_event(rng, first, second, color, reserved, remaining)
    for _ in range(remaining):
        color = int(rng.choice(palette))
        _point_event(rng, first, second, color, reserved)

    # A color can occur only in moving cells. Missing colors get one shared token.
    free = [(r, c) for r in range(height) for c in range(width) if (r, c) not in reserved]
    rng.shuffle(free)
    for color in palette:
        if not np.any(first == color):
            cell = free.pop()
            first[cell] = color
            second[cell] = color
    occupied = int(np.count_nonzero(first != background))
    area = height * width
    if rng.random() < 0.42:
        desired = int(rng.integers(1, max(3, area // 12 + 1)))
    else:
        desired = int(rng.integers(max(2, area // 10), max(3, area * 3 // 10 + 1)))
    desired = min(max(desired, occupied), int(area * 0.36), occupied + len(free))
    moving_without_stationary = [
        color for color in palette if np.any(first == color) and not np.any((first == color) & (second == color))
    ]
    eligible = [color for color in palette if color not in moving_without_stationary or rng.random() < 0.25]
    if not eligible:
        eligible = list(palette)
    for _ in range(desired - occupied):
        cell = free.pop()
        color = int(rng.choice(eligible))
        first[cell] = color
        second[cell] = color

    if orientation == "vertical":
        stripe = np.full((height, 1), divider, dtype=np.int8)
        input_grid = np.concatenate((first, stripe, second), axis=1)
    else:
        stripe = np.full((1, width), divider, dtype=np.int8)
        input_grid = np.concatenate((first, stripe, second), axis=0)
    return {"input": input_grid, "output": _output(first, second, background, departure, arrival)}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _integer(seed, "seed", 0, None)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    colors = [int(x) for x in rng.permutation(10)]
    base_background, base_divider, departure, arrival = colors[:4]
    train_pool = colors[4:8]
    held_out = colors[8:]
    style = int(rng.choice(3, p=[0.23, 0.62, 0.15]))
    switch_roles = bool(rng.random() < 0.28)
    total = train_count + test_count
    orientations = ["vertical"] * total
    if style == 1:
        orientations = ["vertical", "horizontal"] + [
            "horizontal" if rng.random() < 0.5 else "vertical" for _ in range(total - 2)
        ]
    elif style == 2:
        orientations = ["horizontal"] * total
    if style == 1:
        # Keep an example of each direction in training, including small puzzles.
        orientations[0], orientations[1] = "vertical", "horizontal"
    swaps = [False] * total
    if switch_roles:
        swaps = [False, True] + [bool(rng.integers(2)) for _ in range(total - 2)]
    anchor = int(rng.integers(train_count))
    required = {anchor}
    for orientation in set(orientations[train_count:]):
        if not any(orientations[i] == orientation for i in required):
            candidates = [i for i in range(train_count) if orientations[i] == orientation]
            required.add(int(rng.choice(candidates)))
    train = []
    test = []
    seen = set()
    for index in range(total):
        is_train = index < train_count
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        if is_train:
            palette = _palette(pair_rng, train_pool)
        else:
            observed = sorted({int(x) for pair in train for x in np.unique(pair["input"])
                               if int(x) in train_pool})
            use_novel = pair_rng.random() < 0.55
            forced = int(pair_rng.choice(held_out)) if use_novel else None
            palette = _palette(pair_rng, observed, forced)
        height = 10 if style == 0 else int(pair_rng.integers(5, 15))
        width = 10 if style == 0 else int(pair_rng.integers(5, 15))
        background, divider = ((base_divider, base_background) if swaps[index]
                               else (base_background, base_divider))
        for _ in range(12):
            pair = _make_pair(pair_rng, palette, orientations[index], height, width,
                              background, divider, departure, arrival,
                              force_change=index in required,
                              force_shape=is_train and index == anchor)
            key = _key(pair["input"])
            if key not in seen and not _is_source_input(pair["input"]):
                seen.add(key)
                break
        else:
            raise ValueError("could not construct a unique pair")
        (train if is_train else test).append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _split_input(grid, output_shape):
    height, width = grid.shape
    candidates = []
    if width % 2 == 1 and output_shape == (height, (width - 1) // 2):
        middle = width // 2
        stripe = grid[:, middle]
        first, second = grid[:, :middle], grid[:, middle + 1:]
        if np.all(stripe == stripe[0]) and not np.any(first == stripe[0]) and not np.any(second == stripe[0]):
            candidates.append(("vertical", first, second, int(stripe[0])))
    if height % 2 == 1 and output_shape == ((height - 1) // 2, width):
        middle = height // 2
        stripe = grid[middle, :]
        first, second = grid[:middle, :], grid[middle + 1:, :]
        if np.all(stripe == stripe[0]) and not np.any(first == stripe[0]) and not np.any(second == stripe[0]):
            candidates.append(("horizontal", first, second, int(stripe[0])))
    assert len(candidates) == 1
    return candidates[0]


def _overlap_shape(first, second, background):
    height, width = first.shape
    for r in range(height):
        for c in range(width):
            color = int(first[r, c])
            if color == background or second[r, c] != background:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                for distance in (2, 3, 4):
                    end = (r + distance * dr, c + distance * dc)
                    if not (0 <= end[0] < height and 0 <= end[1] < width):
                        continue
                    shared = all(
                        first[r + step * dr, c + step * dc] == color
                        and second[r + step * dr, c + step * dc] == color
                        for step in range(1, distance)
                    )
                    if shared and first[end] == background and second[end] == color:
                        return True
    return False


def _analyze_pair(pair):
    grid, result = pair["input"], pair["output"]
    orientation, first, second, divider = _split_input(grid, result.shape)
    values, counts = np.unique(np.concatenate((first.ravel(), second.ravel())), return_counts=True)
    background = int(values[np.argmax(counts)])
    assert background != divider
    assert np.count_nonzero(first == background) > first.size // 2
    assert np.count_nonzero(second == background) > second.size // 2
    same = first == second
    old = (first != background) & (second == background)
    new = (first == background) & (second != background)
    assert np.all(same | old | new)
    assert np.count_nonzero(old) == np.count_nonzero(new)
    changed = int(np.count_nonzero(old))
    departure = arrival = None
    if changed:
        departed = np.unique(result[old])
        arrived = np.unique(result[new])
        assert len(departed) == len(arrived) == 1
        departure, arrival = int(departed[0]), int(arrived[0])
        assert len({background, divider, departure, arrival}) == 4
    expected = np.full(first.shape, background, dtype=np.int8)
    expected[same] = first[same]
    if changed:
        expected[old] = departure
        expected[new] = arrival
    assert np.array_equal(result, expected)
    palette = set(int(x) for x in np.unique(np.concatenate((first.ravel(), second.ravel())))) - {background}
    assert 1 <= len(palette) <= 4
    assert divider not in palette
    if changed:
        assert departure not in palette and arrival not in palette
    for color in palette:
        assert np.count_nonzero(first == color) == np.count_nonzero(second == color)
    occupied = int(np.count_nonzero(first != background))
    assert occupied <= int(first.size * 0.40)
    assert occupied >= 1
    return {"orientation": orientation, "shape": first.shape, "background": background,
            "divider": divider, "departure": departure, "arrival": arrival,
            "palette": frozenset(palette), "changes": changed, "occupied": occupied,
            "overlap_shape": _overlap_shape(first, second, background),
            "stationary": int(np.count_nonzero((first == second) & (first != background))),
            "zero_stationary_movers": sum(
                np.any((first == color) & (second == background)) and
                not np.any((first == color) & (second == color)) for color in palette
            )}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs = set()
    seen_pairs = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _key(pair["input"])
            pair_key = (input_key, _key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert not _is_source_input(pair["input"])
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analyses[split].append(_analyze_pair(pair))
    all_analyses = analyses["train"] + analyses["test"]
    changed_train = [a for a in analyses["train"] if a["changes"]]
    assert changed_train and any(a["overlap_shape"] for a in changed_train)
    assert any(a["stationary"] for a in analyses["train"])
    markers = {(a["departure"], a["arrival"]) for a in all_analyses if a["changes"]}
    assert len(markers) == 1
    departure, arrival = next(iter(markers))
    for a in all_analyses:
        assert departure not in a["palette"] and arrival not in a["palette"]
        assert departure not in {a["background"], a["divider"]}
        assert arrival not in {a["background"], a["divider"]}
    for orientation in {a["orientation"] for a in analyses["test"]}:
        assert any(a["orientation"] == orientation and a["changes"] for a in analyses["train"])
    assert all(a["changes"] <= 5 for a in all_analyses)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    for seed, train_count, test_count in ((4000, 2, 1), (4001, 10, 1), (4002, 10, 4)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
    for seed, train_count, test_count in ((5000, None, 1), (5001, None, 4),
                                          (5002, 2, None), (5003, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(6000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    first = generate_puzzle(314159, 7, 3)
    second = generate_puzzle(314159, 7, 3)
    assert all(np.array_equal(a[k], b[k]) for split in ("train", "test")
               for a, b in zip(first[split], second[split]) for k in ("input", "output"))
    assert len({_key(p["train"][0]["input"]) for p in puzzles}) == 42
    analyses = [(split, _analyze_pair(pair)) for puzzle in puzzles
                for split in ("train", "test") for pair in puzzle[split]]
    assert {a["orientation"] for _, a in analyses} == {"horizontal", "vertical"}
    assert len({a["shape"] for _, a in analyses}) >= 30
    assert {a["changes"] for _, a in analyses} == set(range(6))
    for split in ("train", "test"):
        for index in (0, 1):
            index_changes = {
                _analyze_pair(puzzle[split][index])["changes"]
                for puzzle in puzzles if len(puzzle[split]) > index
            }
            assert index_changes == set(range(6))
    assert {len(a["palette"]) for _, a in analyses} == {1, 2, 3, 4}
    assert any(a["occupied"] > 11 for _, a in analyses)
    assert any(a["occupied"] < 5 for _, a in analyses)
    assert any(a["zero_stationary_movers"] for _, a in analyses)
    assert any(a["overlap_shape"] for split, a in analyses if split == "test")
    novel, familiar = False, False
    palette_relations = set()
    mixed_orientations = varied_sizes = swapped_roles = source_geometry = 0
    easier = harder = repeated_changes = combined = 0
    movement_geometries = set()
    movement_distances = set()
    role_signatures = set()
    for puzzle in puzzles:
        train_info = [_analyze_pair(pair) for pair in puzzle["train"]]
        test_info = [_analyze_pair(pair) for pair in puzzle["test"]]
        info = train_info + test_info
        known = set().union(*(a["palette"] for a in train_info))
        mixed_orientations += len({a["orientation"] for a in info}) > 1
        varied_sizes += len({a["shape"] for a in info}) > 1
        swapped_roles += len({a["background"] for a in info}) > 1
        source_geometry += all(a["shape"] == (10, 10) and a["orientation"] == "vertical" for a in info)
        demonstration = next(a for a in train_info if a["changes"])
        role_signatures.add((demonstration["background"], demonstration["divider"],
                             demonstration["departure"], demonstration["arrival"]))
        for index, first_info in enumerate(train_info):
            for second_info in train_info[index + 1:]:
                first_palette, second_palette = first_info["palette"], second_info["palette"]
                palette_relations.add("disjoint" if not first_palette & second_palette else
                                      "reused" if first_palette == second_palette else "partial")
        for pair in puzzle["test"]:
            unseen = _analyze_pair(pair)["palette"] - known
            novel |= bool(unseen)
            familiar |= not unseen
        for a in test_info:
            easier += a["changes"] < min(t["changes"] for t in train_info)
            harder += a["changes"] > max(t["changes"] for t in train_info)
        for pair, a in zip(puzzle["train"] + puzzle["test"], info):
            combined += (a["orientation"] == "horizontal" and a["shape"][0] != a["shape"][1]
                         and a["changes"] >= 2 and len(a["palette"]) >= 3 and a["overlap_shape"])
            _, first_panel, second_panel, _ = _split_input(pair["input"], pair["output"].shape)
            for color in a["palette"]:
                old = np.argwhere((first_panel == color) & (second_panel == a["background"]))
                new = np.argwhere((first_panel == a["background"]) & (second_panel == color))
                repeated_changes += len(old) >= 2
                if len(old) == len(new) == 1:
                    dr, dc = new[0] - old[0]
                    movement_geometries.add("horizontal" if dr == 0 else
                                            "vertical" if dc == 0 else "diagonal")
                    movement_distances.add("short" if max(abs(dr), abs(dc)) <= 2 else "long")
    assert novel and familiar
    assert palette_relations == {"disjoint", "partial", "reused"}
    assert mixed_orientations >= 5 and varied_sizes >= 10 and source_geometry >= 3
    assert swapped_roles >= 3 and len(role_signatures) >= 20
    assert easier and harder and repeated_changes and combined
    assert movement_geometries == {"horizontal", "vertical", "diagonal"}
    assert movement_distances == {"short", "long"}
