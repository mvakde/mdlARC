"""Scatter nine directional tiles and assemble them by their internal motifs."""

import numpy as np


_DIRECTIONS = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2),
               (2, 0), (2, 1), (2, 2))
_CENTER = (1, 1)
_SLOTS = _DIRECTIONS + (_CENTER,)
_MASKS = {
    (0, 0): ((0, 0), (0, 1), (1, 0)),
    (0, 1): ((0, 0), (0, 1), (0, 2), (1, 1)),
    (0, 2): ((0, 1), (0, 2), (1, 2)),
    (1, 0): ((0, 0), (1, 0), (2, 0), (1, 1)),
    (1, 2): ((0, 2), (1, 2), (2, 2), (1, 1)),
    (2, 0): ((1, 0), (2, 0), (2, 1)),
    (2, 1): ((1, 1), (2, 0), (2, 1), (2, 2)),
    (2, 2): ((1, 2), (2, 1), (2, 2)),
}
_MASK_TO_SLOT = {frozenset(mask): slot for slot, mask in _MASKS.items()}

# The three original inputs are excluded by content, not by shape or palette.
_SOURCE_INPUTS = {
    (16, 14, ("00000000000554", "05550000000544", "03550588000554",
              "03350558000000", "00000555000000", "00000000005550",
              "00000000005590", "00000000005990", "01110055500000",
              "05150055506550", "05550055506650", "00000000006550",
              "00007750000000", "00007550055500", "00005550052500",
              "00000000022200")),
    (14, 14, ("55500000000000", "85500000222000", "88500000525000",
              "00255000555000", "00225000000000", "00255055500000",
              "00000055505550", "05110055505450", "05510000004440",
              "05550055300000", "00000053300000", "55500055306650",
              "55900000006550", "59900000005550")),
    (16, 15, ("000000000055500", "015500000065500", "011502220066500",
              "015505250000000", "000005550000000", "000000000055500",
              "005550000055500", "005850551055500", "008880511000000",
              "000000551000000", "054400000003350", "055400000003550",
              "055500555005550", "000000557000000", "000000577000000",
              "000000000000000")),
}


def _checked_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if name == "num_train" and not 2 <= value <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if name == "num_test" and not 1 <= value <= 4:
        raise ValueError("num_test must be between 1 and 4")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _checked_count(num_train, "num_train")
    num_test = _checked_count(num_test, "num_test")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        if rng.random() < 0.38:
            return 2, 1
        num_test = int(rng.choice((1, 2, 3, 4), p=(.48, .28, .16, .08)))
        return int(rng.integers(num_test + 1, 11)), num_test
    if num_train is None:
        return int(rng.integers(num_test + 1, 11)), num_test
    return num_train, int(rng.integers(1, min(4, num_train - 1) + 1))


def _roles(rng):
    order = rng.permutation(10)
    return int(order[0]), int(order[1])


def _accent_assignment(rng, background, carrier, kind):
    colors = [x for x in range(10) if x not in (background, carrier)]
    if kind == "distinct":
        count = 8
    elif kind == "repeat":
        count = int(rng.choice((1, 2, 3, 4, 5, 6, 7),
                               p=(.09, .10, .12, .14, .16, .18, .21)))
    elif kind == "test":
        count = int(rng.choice((1, 2, 3, 4, 5, 6, 7, 8),
                               p=(.13, .14, .15, .15, .13, .12, .11, .07)))
    else:
        count = int(rng.integers(1, 9))
    chosen = [colors[int(i)] for i in rng.permutation(8)[:count]]
    slots = [_DIRECTIONS[int(i)] for i in rng.permutation(8)]
    labels = list(range(count))
    labels.extend(int(rng.integers(count)) for _ in range(8 - count))
    rng.shuffle(labels)
    return {slot: chosen[label] for slot, label in zip(slots, labels)}


def _tiles(carrier, accents):
    result = {_CENTER: np.full((3, 3), carrier, dtype=np.int8)}
    for slot in _DIRECTIONS:
        tile = np.full((3, 3), carrier, dtype=np.int8)
        for row, column in _MASKS[slot]:
            tile[row, column] = accents[slot]
        result[slot] = tile
    return result


def _dimension(rng, mode):
    if mode == "dense":
        return int(rng.integers(13, 18))
    if mode == "sparse":
        return int(rng.integers(21, 31))
    return int(rng.integers(13, 31))


def _bands(rng, length):
    extra = length - 12
    weights = rng.dirichlet((1.5, 1.5, 1.5))
    sizes = 4 + rng.multinomial(extra, weights)
    return (0, int(sizes[0]), int(sizes[0] + sizes[1]), length)


def _touch_count(positions):
    count = 0
    for i, (r1, c1) in enumerate(positions):
        for r2, c2 in positions[i + 1:]:
            vertical = abs(r1 - r2) == 3 and abs(c1 - c2) < 3
            horizontal = abs(c1 - c2) == 3 and abs(r1 - r2) < 3
            count += int(vertical or horizontal)
    return count


def _sample_positions(rng, height, width, layout):
    for _ in range(120):
        if layout == "touch":
            box_h = int(rng.integers(13, min(height, 18) + 1))
            box_w = int(rng.integers(13, min(width, 18) + 1))
        elif layout == "spread":
            box_h = int(rng.integers(max(13, height - 3), height + 1))
            box_w = int(rng.integers(max(13, width - 3), width + 1))
        else:
            box_h = int(rng.integers(13, height + 1))
            box_w = int(rng.integers(13, width + 1))
        top = int(rng.integers(0, height - box_h + 1))
        left = int(rng.integers(0, width - box_w + 1))
        rows = _bands(rng, box_h)
        cols = _bands(rng, box_w)
        positions = []
        for row_band in range(3):
            for col_band in range(3):
                row = top + int(rng.integers(rows[row_band], rows[row_band + 1] - 2))
                col = left + int(rng.integers(cols[col_band], cols[col_band + 1] - 2))
                positions.append((row, col))
        if min(len({r for r, _ in positions}), len({c for _, c in positions})) < 5:
            continue
        touches = _touch_count(positions)
        if layout == "touch" and touches == 0:
            continue
        if layout == "spread" and touches > 0:
            continue
        return positions
    raise RuntimeError("could not place nine tiles")


def _make_pair(rng, roles, accent_kind, dimension_mode, layout):
    background, carrier = roles
    height = _dimension(rng, dimension_mode)
    width = _dimension(rng, dimension_mode)
    accents = _accent_assignment(rng, background, carrier, accent_kind)
    tiles = _tiles(carrier, accents)
    positions = _sample_positions(rng, height, width, layout)
    input_grid = np.full((height, width), background, dtype=np.int8)
    output_grid = np.empty((9, 9), dtype=np.int8)
    for slot, tile in tiles.items():
        row, col = slot
        output_grid[3 * row:3 * row + 3, 3 * col:3 * col + 3] = tile
    for index, slot_index in enumerate(rng.permutation(9)):
        row, col = positions[index]
        input_grid[row:row + 3, col:col + 3] = tiles[_SLOTS[int(slot_index)]]
    return {"input": input_grid, "output": output_grid}


def _grid_key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _source_key(grid):
    rows = tuple("".join(str(int(cell)) for cell in row) for row in grid)
    return grid.shape[0], grid.shape[1], rows


def _tile_slot(tile, carrier):
    other = tile != carrier
    if not np.any(other):
        return _CENTER, None
    values = np.unique(tile[other])
    if len(values) != 1:
        return None
    mask = frozenset(map(tuple, np.argwhere(other)))
    slot = _MASK_TO_SLOT.get(mask)
    if slot is None:
        return None
    return slot, int(values[0])


def _partition(input_grid, background):
    """Find at most two complete decompositions, using only the input."""
    occupied = np.argwhere(input_grid != background)
    assert len(occupied) == 81
    locations = np.full(input_grid.shape, -1, dtype=np.int16)
    for index, (row, col) in enumerate(occupied):
        locations[row, col] = index
    full = (1 << 81) - 1
    solutions = []
    for carrier in np.unique(input_grid[input_grid != background]):
        carrier = int(carrier)
        candidates = []
        by_cell = [[] for _ in range(81)]
        for row in range(input_grid.shape[0] - 2):
            for col in range(input_grid.shape[1] - 2):
                indices = locations[row:row + 3, col:col + 3]
                if np.any(indices < 0):
                    continue
                tile = input_grid[row:row + 3, col:col + 3]
                parsed = _tile_slot(tile, carrier)
                if parsed is None:
                    continue
                slot, accent = parsed
                bitmask = 0
                for index in indices.flat:
                    bitmask |= 1 << int(index)
                candidate = (bitmask, slot, (row, col), accent)
                candidate_index = len(candidates)
                candidates.append(candidate)
                for index in indices.flat:
                    by_cell[int(index)].append(candidate_index)

        def search(used, slots, selected):
            if len(solutions) >= 2:
                return
            if used == full:
                if len(selected) == 9 and len(slots) == 9:
                    solutions.append((carrier, tuple(candidates[i] for i in selected)))
                return
            best = None
            for cell in range(81):
                if used & (1 << cell):
                    continue
                options = [i for i in by_cell[cell]
                           if not (candidates[i][0] & used) and candidates[i][1] not in slots]
                if not options:
                    return
                if best is None or len(options) < len(best):
                    best = options
                    if len(best) == 1:
                        break
            for index in best:
                mask, slot, _, _ = candidates[index]
                search(used | mask, slots | {slot}, selected + [index])
                if len(solutions) >= 2:
                    return

        search(0, set(), [])
        if len(solutions) >= 2:
            break
    assert len(solutions) == 1, "input has no unique nine-tile partition"
    return solutions[0]


def _analyze_pair(pair):
    input_grid, output_grid = pair["input"], pair["output"]
    counts = np.bincount(input_grid.ravel().astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > int(np.max(np.delete(counts, background)))
    carrier, decomposition = _partition(input_grid, background)
    expected = np.empty((9, 9), dtype=np.int8)
    accents = {}
    positions = {}
    for _, slot, (row, col), accent in decomposition:
        tile = input_grid[row:row + 3, col:col + 3]
        output_row, output_col = slot
        expected[3 * output_row:3 * output_row + 3,
                 3 * output_col:3 * output_col + 3] = tile
        positions[slot] = (row, col)
        if accent is not None:
            assert accent not in (background, carrier)
            accents[slot] = accent
    assert set(positions) == set(_SLOTS)
    assert set(accents) == set(_DIRECTIONS)
    assert output_grid.shape == (9, 9)
    assert np.array_equal(output_grid, expected)
    assert not np.any(output_grid == background)
    return {"background": background, "carrier": carrier,
            "accents": accents, "positions": positions,
            "touches": _touch_count(list(positions.values()))}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    metadata = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            key = _grid_key(pair["input"])
            assert key not in seen_inputs
            seen_inputs.add(key)
            assert _source_key(pair["input"]) not in _SOURCE_INPUTS
            metadata[split].append(_analyze_pair(pair))
    train = metadata["train"]
    test = metadata["test"]
    assert any(len(set(item["accents"].values())) == 8 for item in train)
    assert any(len(set(item["accents"].values())) < 8 for item in train)
    train_roles = {(item["background"], item["carrier"]) for item in train}
    if len(train_roles) == 1:
        assert all((item["background"], item["carrier"]) in train_roles for item in test)
    assert any(item["touches"] > 0 for item in train)
    assert any(item["touches"] == 0 for item in train)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    stable_roles = bool(rng.integers(2))
    common_roles = _roles(rng)
    train_order = [int(i) for i in rng.permutation(num_train)]
    distinct_index, repeat_index = train_order[:2]
    touch_index, spread_index = [int(i) for i in rng.permutation(num_train)[:2]]
    dense_index, sparse_index = [int(i) for i in rng.permutation(num_train)[:2]]
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            if split == "train":
                kind = ("distinct" if index == distinct_index else
                        "repeat" if index == repeat_index else
                        ("distinct" if rng.random() < .22 else "repeat"))
                layout = ("touch" if index == touch_index else
                          "spread" if index == spread_index else
                          ("touch" if rng.random() < .30 else "free"))
                dimension = ("dense" if index == dense_index else
                             "sparse" if index == sparse_index else "any")
            else:
                kind = "test"
                layout = "touch" if rng.random() < .38 else "free"
                dimension = "any"
            for _ in range(80):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                if stable_roles or (split == "train" and index == 0):
                    roles = common_roles
                else:
                    roles = _roles(pair_rng)
                    if split == "train" and index == 1 and roles == common_roles:
                        continue
                try:
                    pair = _make_pair(pair_rng, roles, kind, dimension, layout)
                    key = _grid_key(pair["input"])
                    if key in used_inputs:
                        continue
                    _analyze_pair(pair)
                except (AssertionError, RuntimeError):
                    continue
                puzzle[split].append(pair)
                used_inputs.add(key)
                break
            else:
                raise RuntimeError("could not construct a unique puzzle pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    samples = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(puzzle) for puzzle in samples}) == 42
    assert len({len(puzzle["train"]) for puzzle in samples}) >= 4
    assert len({len(puzzle["test"]) for puzzle in samples}) >= 3
    assert sum(len(puzzle["train"]) == 2 and len(puzzle["test"]) == 1
               for puzzle in samples) >= 8
    shapes, aspects, roles, accent_counts, layouts = set(), set(), set(), set(), set()
    train_shapes, test_shapes = set(), set()
    train_aspects, test_aspects = set(), set()
    review_counts = {name: 0 for name in (
        "dense", "sparse", "source_shape", "all_same", "opposite_repeat",
        "multiple_groups", "nonadjacent_repeat", "row_7", "column_7",
        "touch_dense", "touch_sparse", "touch_heavy", "one_cell_contact", "stable_roles",
        "varying_roles")}
    touches = 0
    subtle_width_puzzles = 0
    dense_sparse_puzzles = 0
    for puzzle in samples:
        validate_generated_puzzle(puzzle)
        puzzle_roles = set()
        subtle_width_puzzles += any(
            test_pair["input"].shape[0] == train_pair["input"].shape[0]
            and abs(test_pair["input"].shape[1] - train_pair["input"].shape[1]) == 1
            for train_pair in puzzle["train"] for test_pair in puzzle["test"])
        train_densities = [81 / pair["input"].size for pair in puzzle["train"]]
        dense_sparse_puzzles += (any(value >= .34 for value in train_densities)
                                 and any(value < .20 for value in train_densities))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                height, width = pair["input"].shape
                density = 81 / (height * width)
                unique_accents = len(set(info["accents"].values()))
                puzzle_roles.add((info["background"], info["carrier"]))
                shapes.add((height, width))
                aspects.add(int(np.sign(height - width)))
                roles.add((info["background"], info["carrier"]))
                accent_counts.add(unique_accents)
                layouts.add((pair["input"].shape,
                             tuple(info["positions"][slot] for slot in _SLOTS)))
                touches += info["touches"] > 0
                (train_shapes if split == "train" else test_shapes).add((height, width))
                (train_aspects if split == "train" else test_aspects).add(
                    int(np.sign(height - width)))
                review_counts["dense"] += density >= .34
                review_counts["sparse"] += density < .20
                review_counts["source_shape"] += (height, width) in {
                    (14, 14), (16, 14), (16, 15)}
                review_counts["all_same"] += unique_accents == 1
                review_counts["opposite_repeat"] += (
                    info["accents"][(1, 0)] == info["accents"][(1, 2)])
                groups = [
                    {slot for slot, color in info["accents"].items() if color == value}
                    for value in set(info["accents"].values())]
                review_counts["multiple_groups"] += sum(
                    len(group) > 1 for group in groups) >= 2
                review_counts["nonadjacent_repeat"] += any(
                    len(group) > 1 and any(
                        abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1
                        for a in group for b in group)
                    for group in groups)
                review_counts["row_7"] += len({
                    row for row, _ in info["positions"].values()}) >= 7
                review_counts["column_7"] += len({
                    col for _, col in info["positions"].values()}) >= 7
                review_counts["touch_dense"] += info["touches"] > 0 and density >= .34
                review_counts["touch_sparse"] += info["touches"] > 0 and density < .20
                review_counts["touch_heavy"] += info["touches"] > 0 and unique_accents <= 3
                positions = list(info["positions"].values())
                review_counts["one_cell_contact"] += any(
                    (abs(r1 - r2) == 3 and abs(c1 - c2) == 2) or
                    (abs(c1 - c2) == 3 and abs(r1 - r2) == 2)
                    for i, (r1, c1) in enumerate(positions)
                    for r2, c2 in positions[i + 1:])
        review_counts["stable_roles"] += len(puzzle_roles) == 1
        review_counts["varying_roles"] += len(puzzle_roles) > 1
    assert len(shapes) >= 60 and aspects == {-1, 0, 1}
    assert len(roles) >= 25 and accent_counts == set(range(1, 9))
    assert len(layouts) >= 150 and touches >= 20
    assert train_aspects == test_aspects == {-1, 0, 1}
    assert len(train_shapes & test_shapes) >= 10
    assert subtle_width_puzzles >= 2
    assert dense_sparse_puzzles >= 15
    for name, minimum in {
        "dense": 20, "sparse": 80, "source_shape": 3,
        "all_same": 5, "opposite_repeat": 15, "multiple_groups": 50,
        "nonadjacent_repeat": 70, "row_7": 40, "column_7": 40,
        "touch_dense": 8, "touch_sparse": 30, "touch_heavy": 10,
        "one_cell_contact": 10,
        "stable_roles": 8, "varying_roles": 8,
    }.items():
        assert review_counts[name] >= minimum, name
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1),
                   (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1000 + len(str(counts)), *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
    assert _puzzle_key(generate_puzzle(314159, 7, 2)) == _puzzle_key(generate_puzzle(314159, 7, 2))
    for bad in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                {"num_test": 5}, {"num_train": 2, "num_test": 2},
                {"num_train": 3, "num_test": 4},
                {"num_train": True, "num_test": 1}):
        try:
            generate_puzzle(999, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count request accepted")
