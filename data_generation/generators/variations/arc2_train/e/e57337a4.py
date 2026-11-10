"""Equal-block occupancy maps, with one output cell per input region."""

import numpy as np


_CELLS = tuple((r, c) for r in range(3) for c in range(3))
_SOURCE_SIGNATURES = frozenset({
    (8, ((2, 1), (7, 7))),
    (9, ((2, 12),)),
    (7, ((2, 3), (2, 11), (9, 2))),
    (6, ((1, 5), (12, 13))),
})


def _resolve_counts(rng, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)
    ):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
    if num_train is None:
        low = 2 if num_test is None else int(num_test) + 1
        num_train = int(rng.integers(low, 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(grid):
    return grid.shape, grid.tobytes()


def _source_signature(grid, marker, field):
    if marker != 0 or grid.shape != (15, 15):
        return None
    return field, tuple(map(tuple, np.argwhere(grid == marker)))


def _choose_cells(rng, count, allowed=_CELLS, require_bottom=False, upper_only=False):
    choices = [cell for cell in allowed if not upper_only or cell[0] < 2]
    if require_bottom:
        bottom = [cell for cell in choices if cell[0] == 2]
        first = bottom[int(rng.integers(len(bottom)))]
        rest = [cell for cell in choices if cell != first]
        picked = [first] + [rest[int(i)] for i in rng.choice(len(rest), count - 1, replace=False)]
    else:
        picked = [choices[int(i)] for i in rng.choice(len(choices), count, replace=False)]
    return frozenset(picked)


def _neighbor(rng, allowed):
    edges = [(a, b) for a in allowed for b in allowed
             if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
    return edges[int(rng.integers(len(edges)))]


def _edge_positions(rng, source, target, size):
    offset = int(rng.integers(size))
    if target[0] == source[0]:
        if target[1] > source[1]:
            return (offset, size - 1), (offset, 0)
        return (offset, 0), (offset, size - 1)
    if target[0] > source[0]:
        return (size - 1, offset), (0, offset)
    return (0, offset), (size - 1, offset)


def _local_positions(rng, size, count):
    positions = set()
    if count >= 3 and rng.random() < 0.48:
        if rng.random() < 0.5:
            row = int(rng.integers(size))
            col = int(rng.integers(size - 2))
            positions.update((row, col + i) for i in range(3))
        else:
            row = int(rng.integers(size - 2))
            col = int(rng.integers(size))
            positions.update((row + i, col) for i in range(3))
    corners = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
    boundary = [(r, c) for r in range(size) for c in range(size)
                if r in (0, size - 1) or c in (0, size - 1)]
    while len(positions) < count:
        roll = rng.random()
        pool = corners if roll < 0.25 else boundary if roll < 0.62 else None
        point = pool[int(rng.integers(len(pool)))] if pool else (
            int(rng.integers(size)), int(rng.integers(size))
        )
        positions.add(point)
    return positions


def _make_pair(rng, size, marker, field, occupied, mode="mixed", forced=None):
    grid = np.full((3 * size, 3 * size), field, dtype=np.int8)
    output = np.full((3, 3), field, dtype=np.int8)
    occupied = sorted(occupied)
    for cell in occupied:
        output[cell] = marker
    forced = forced or {}
    extra = []
    if mode == "multi":
        extra = [occupied[0]]
    elif mode == "mixed" and rng.random() < 0.72:
        count = 1 if rng.random() < 0.72 else 2
        extra = [occupied[int(i)] for i in rng.choice(len(occupied),
                 min(count, len(occupied)), replace=False)]
    for cell in occupied:
        count = 1
        if cell in extra:
            count += int(rng.integers(2, 4)) if mode == "multi" else int(rng.integers(1, 4))
        positions = set(forced.get(cell, ()))
        if len(positions) < count:
            candidates = _local_positions(rng, size, count)
            positions.update(candidates)
            if len(positions) > count:
                positions = set(sorted(positions)[:count])
        for row, col in positions:
            grid[size * cell[0] + row, size * cell[1] + col] = marker
    return {"input": grid, "output": output}


def _straddle(rng, pair, size, marker):
    occupied = set(map(tuple, np.argwhere(pair["output"] == marker)))
    edges = [(a, b) for a in occupied for b in occupied
             if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
    if not edges:
        return
    a, b = edges[int(rng.integers(len(edges)))]
    local_a, local_b = _edge_positions(rng, a, b, size)
    pair["input"][size * a[0] + local_a[0], size * a[1] + local_a[1]] = marker
    pair["input"][size * b[0] + local_b[0], size * b[1] + local_b[1]] = marker


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    size = int(rng.choice(np.array([3, 4, 5, 6, 7])))
    marker = int(rng.integers(10))
    fields = [int(v) for v in rng.permutation([c for c in range(10) if c != marker])]
    if marker != 0 and rng.random() < 0.5:
        fields.remove(0)
        fields.insert(int(rng.integers(2)), 0)
    train_fields = [fields[0], fields[1]]
    for i in range(2, train_count):
        train_fields.append(fields[0] if i == 2 else fields[1] if i == 3
                            else fields[int(rng.integers(min(5, len(fields))))])
    familiar = set(train_fields)
    novel = [c for c in fields if c not in familiar]
    first_familiar = bool(rng.integers(2))
    test_fields = []
    for i in range(test_count):
        use_familiar = first_familiar if i % 2 == 0 else not first_familiar
        pool = sorted(familiar) if use_familiar else novel
        test_fields.append(pool[int(rng.integers(len(pool)))])

    holdout_bottom = rng.random() < 0.36
    allowed = tuple(cell for cell in _CELLS if not holdout_bottom or cell[0] < 2)
    first_count = int(rng.integers(1, 5))
    first_cells = _choose_cells(rng, first_count, allowed,
                                require_bottom=not holdout_bottom)
    second_count = int(rng.choice([n for n in range(1, 5) if n != first_count]))
    if train_count >= 4:
        source, destination = _neighbor(rng, allowed)
        other_allowed = [c for c in allowed if c not in (source, destination)]
        common = _choose_cells(rng, second_count - 1, other_allowed) if second_count > 1 else frozenset()
        second_cells = frozenset(set(common) | {source})
    else:
        source = destination = None
        second_cells = _choose_cells(rng, second_count, allowed)

    puzzle = {"train": [], "test": []}
    seen = set()

    def add(split, pair):
        key = _key(pair["input"])
        colors, counts = np.unique(pair["input"], return_counts=True)
        field = int(colors[int(np.argmax(counts))])
        if key in seen or _source_signature(pair["input"], marker, field) in _SOURCE_SIGNATURES:
            return False
        seen.add(key)
        puzzle[split].append(pair)
        return True

    for index in range(train_count):
        field = train_fields[index]
        if index == 0:
            cells, mode = first_cells, "multi"
        elif index == 1:
            cells, mode = second_cells, "single"
        elif index == 2:
            cells, mode = first_cells, "single"
        elif index == 3:
            base = puzzle["train"][1]
            moved = {"input": base["input"].copy(), "output": base["output"].copy()}
            old_local, new_local = boundary_positions
            moved["input"][size * source[0] + old_local[0],
                           size * source[1] + old_local[1]] = field
            moved["input"][size * destination[0] + new_local[0],
                           size * destination[1] + new_local[1]] = marker
            moved["output"][source] = field
            moved["output"][destination] = marker
            assert add("train", moved)
            continue
        else:
            count = int(rng.integers(1, 7))
            cells = _choose_cells(rng, count, allowed)
            mode = "mixed"
        for _ in range(50):
            forced = None
            if index == 1 and train_count >= 4:
                boundary_positions = _edge_positions(rng, source, destination, size)
                forced = {source: {boundary_positions[0]}}
            pair = _make_pair(rng, size, marker, field, cells, mode, forced)
            if index >= 4 and rng.random() < 0.3:
                _straddle(rng, pair, size, marker)
            if add("train", pair):
                break
        else:
            raise RuntimeError("Could not construct a distinct training pair")

    train_has_bottom = any(np.any(p["output"][2] == marker) for p in puzzle["train"])
    for index in range(test_count):
        field = test_fields[index]
        upper_only = train_has_bottom and rng.random() < 0.47
        require_bottom = not train_has_bottom and index == 0
        limit = 6 if upper_only else 8
        count = int(rng.integers(1, limit + 1))
        if index == 0 and train_count == 2 and field == train_fields[0] and rng.random() < 0.4:
            cells = first_cells
        else:
            cells = _choose_cells(rng, count, require_bottom=require_bottom,
                                  upper_only=upper_only)
        for _ in range(50):
            pair = _make_pair(rng, size, marker, field, cells)
            if rng.random() < 0.45:
                _straddle(rng, pair, size, marker)
            if add("test", pair):
                break
        else:
            raise RuntimeError("Could not construct a distinct test pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair, size, marker):
    grid, output = pair["input"], pair["output"]
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 2
    field = int(colors[int(np.argmax(counts))])
    assert field != marker and int(colors[int(np.argmin(counts))]) == marker
    assert int(np.count_nonzero(grid == marker)) <= grid.size // 4
    occupancy = (grid.reshape(3, size, 3, size) == marker).any(axis=(1, 3))
    expected = np.where(occupancy, marker, field)
    assert np.array_equal(output, expected)
    assert 1 <= int(occupancy.sum()) <= 8
    assert _source_signature(grid, marker, field) not in _SOURCE_SIGNATURES
    return field, occupancy


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    first = puzzle["train"][0]["input"]
    assert first.shape[0] == first.shape[1] and first.shape[0] % 3 == 0
    size = first.shape[0] // 3
    assert 3 <= size <= 7
    colors, counts = np.unique(first, return_counts=True)
    assert len(colors) == 2
    marker = int(colors[int(np.argmin(counts))])
    seen_inputs, seen_pairs = set(), set()
    fields = {"train": [], "test": []}
    occupancies = {"train": [], "test": []}
    repeated_region = False
    isolated_pair = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            grid = pair["input"]
            assert grid.shape == (3 * size, 3 * size)
            assert pair["output"].shape == (3, 3)
            key = _key(grid)
            pair_key = (key, _key(pair["output"]))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            field, occupancy = _inspect_pair(pair, size, marker)
            fields[split].append(field)
            occupancies[split].append(occupancy)
            blocks = (grid == marker).reshape(3, size, 3, size).sum(axis=(1, 3))
            if split == "train":
                repeated_region |= bool(np.any(blocks >= 2))
                isolated_pair |= bool(np.all(blocks <= 1))
                assert int(occupancy.sum()) <= 6
    assert len(set(fields["train"])) >= 2
    assert repeated_region and isolated_pair
    if len(puzzle["train"]) == 2:
        assert int(occupancies["train"][0].sum()) != int(occupancies["train"][1].sum())
    if len(puzzle["train"]) >= 3:
        assert fields["train"][0] == fields["train"][2]
        assert np.array_equal(puzzle["train"][0]["output"], puzzle["train"][2]["output"])
        assert np.count_nonzero(puzzle["train"][0]["input"] == marker) > np.count_nonzero(puzzle["train"][2]["input"] == marker)
    if len(puzzle["train"]) >= 4:
        a, b = puzzle["train"][1], puzzle["train"][3]
        assert fields["train"][1] == fields["train"][3]
        changed = np.argwhere(a["input"] != b["input"])
        assert len(changed) == 2 and int(np.abs(changed[0] - changed[1]).sum()) == 1
        assert np.count_nonzero(a["output"] != b["output"]) == 2
    bottom_train = any(bool(np.any(occ[2])) for occ in occupancies["train"])
    if not bottom_train:
        assert any(bool(np.any(occ[2])) for occ in occupancies["test"])
    if len(puzzle["test"]) >= 2:
        familiar = set(fields["train"])
        assert any(f in familiar for f in fields["test"])
        assert any(f not in familiar for f in fields["test"])


def _fingerprint(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test() -> None:
    fingerprints, train_counts, test_counts = set(), set(), set()
    sizes, markers, layouts, fine = set(), set(), set(), set()
    train_bottom = test_upper = field_zero = many = 0
    familiar_test = novel_test = 0
    first_test_layouts, occupancy_counts = set(), {"train": set(), "test": set()}
    side_hits = {"top": 0, "bottom": 0, "left": 0, "right": 0, "corner": 0}
    multi_train = multi_test = straddle_train = straddle_test = 0
    adjacent_blocks = separated_blocks = 0
    holdout_puzzles = familiar_rows_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fp = _fingerprint(puzzle)
        assert fp == _fingerprint(generate_puzzle(seed))
        fingerprints.add(fp)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        size = puzzle["train"][0]["input"].shape[0] // 3
        sizes.add(size)
        colors, counts = np.unique(puzzle["train"][0]["input"], return_counts=True)
        marker = int(colors[int(np.argmin(counts))])
        markers.add(marker)
        train_fields = set()
        training_bottom_here = False
        test_upper_here = False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                colors, counts = np.unique(grid, return_counts=True)
                field = int(colors[int(np.argmax(counts))])
                if split == "train":
                    train_fields.add(field)
                    train_bottom += bool(np.any(output[2] == marker))
                    training_bottom_here |= bool(np.any(output[2] == marker))
                    field_zero += field == 0 and marker != 0
                else:
                    test_upper += not bool(np.any(output[2] == marker))
                    test_upper_here |= not bool(np.any(output[2] == marker))
                mask = (grid != field).reshape(3, size, 3, size).any(axis=(1, 3))
                assert np.array_equal(output, np.where(mask, marker, field))
                layout = tuple(map(tuple, np.argwhere(mask)))
                layouts.add(layout)
                if split == "test" and pair is puzzle["test"][0]:
                    first_test_layouts.add(layout)
                occupancy_counts[split].add(int(mask.sum()))
                positions = set(map(tuple, np.argwhere(grid == marker)))
                fine.add((size, tuple(sorted(positions))))
                many += int(mask.sum() >= 5)
                blocks = (grid == marker).reshape(3, size, 3, size).sum(axis=(1, 3))
                if split == "train":
                    multi_train += bool(np.any(blocks >= 3))
                else:
                    multi_test += bool(np.any(blocks >= 3))
                for row, col in positions:
                    local_row, local_col = row % size, col % size
                    side_hits["top"] += local_row == 0
                    side_hits["bottom"] += local_row == size - 1
                    side_hits["left"] += local_col == 0
                    side_hits["right"] += local_col == size - 1
                    side_hits["corner"] += local_row in (0, size - 1) and local_col in (0, size - 1)
                    for other in ((row + 1, col), (row, col + 1)):
                        if other in positions:
                            if (row // size, col // size) != (other[0] // size, other[1] // size):
                                if split == "train":
                                    straddle_train += 1
                                else:
                                    straddle_test += 1
                            else:
                                adjacent_blocks += 1
                for macro_row, macro_col in _CELLS:
                    local = [(r, c) for r, c in positions
                             if r // size == macro_row and c // size == macro_col]
                    if len(local) >= 3 and not any(
                        abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                        for a in local for b in local if a != b
                    ):
                        separated_blocks += 1
        holdout_puzzles += not training_bottom_here
        familiar_rows_puzzles += training_bottom_here and test_upper_here
        for pair in puzzle["test"]:
            colors, counts = np.unique(pair["input"], return_counts=True)
            field = int(colors[int(np.argmax(counts))])
            familiar_test += field in train_fields
            novel_test += field not in train_fields
    assert len(fingerprints) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert sizes == {3, 4, 5, 6, 7} and len(markers) >= 7
    assert len(layouts) >= 70 and len(fine) >= 150
    assert train_bottom > 0 and test_upper > 0 and field_zero > 0 and many > 0
    assert familiar_test > 0 and novel_test > 0
    assert len(first_test_layouts) >= 20
    assert len(occupancy_counts["train"]) >= 5 and len(occupancy_counts["test"]) >= 7
    assert all(value > 0 for value in side_hits.values())
    assert multi_train > 0 and multi_test > 0
    assert straddle_train > 0 and straddle_test > 0
    assert adjacent_blocks > 0 and separated_blocks > 0
    assert holdout_puzzles > 0 and familiar_rows_puzzles > 0
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(200 + ntrain + ntest, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (ntrain, ntest)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(200 + ntrain + ntest, ntrain, ntest))
    for ntrain, ntest in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(300 + (ntrain or 0) + (ntest or 0), ntrain, ntest)
        validate_generated_puzzle(puzzle)
        assert ntrain is None or len(puzzle["train"]) == ntrain
        assert ntest is None or len(puzzle["test"]) == ntest
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5),
                          (2, 2), (3, 4), (4, 4), (True, 1)):
        try:
            generate_puzzle(400, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
