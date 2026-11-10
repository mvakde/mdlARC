"""Separated glyph strips whose categorical labels become solid output rows."""

import numpy as np


SOLID, HOLE, BOTTOM, SIDES = range(4)
CLASSES = frozenset(range(4))

# The original side openings are one column deep, not two. These masks are
# also used to detect an exact source pair; generation is not limited to them.
SOURCE_MASKS = (
    np.array([[1, 1, 1, 1]] * 4, dtype=bool),
    np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [0, 1, 1, 0], [0, 1, 1, 0], [1, 1, 1, 1]], dtype=bool),
)
SOURCE_SEQUENCES = frozenset({
    (HOLE, SOLID, BOTTOM),
    (SIDES, BOTTOM, SOLID),
    (SOLID, BOTTOM, SOLID),
    (SOLID, HOLE, SIDES),
    (BOTTOM, SIDES, HOLE),
})
SOURCE_LABELS = (2, 8, 4, 3)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _counts(seed, rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        legal = [(a, b) for a in range(2, 11) for b in range(1, 5) if a > b]
        return legal[int(seed) % len(legal)]
    if train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _mask(class_id, height, width, rng):
    if height == width == 4 and rng.random() < 0.27:
        return SOURCE_MASKS[class_id].copy()
    mask = np.ones((height, width), dtype=bool)
    if class_id == SOLID:
        return mask
    if class_id == HOLE:
        cut_h = int(rng.integers(1, height - 1))
        cut_w = int(rng.integers(1, width - 1))
        row = int(rng.integers(1, height - cut_h))
        col = int(rng.integers(1, width - cut_w))
        mask[row:row + cut_h, col:col + cut_w] = False
    elif class_id == BOTTOM:
        cut_w = int(rng.integers(2, width - 1))
        depth = int(rng.integers(1, height))
        col = int(rng.integers(1, width - cut_w))
        mask[height - depth:, col:col + cut_w] = False
    elif class_id == SIDES:
        depth = int(rng.integers(1, (width - 1) // 2 + 1))
        cut_h = int(rng.integers(1, height - 1))
        row = int(rng.integers(1, height - cut_h))
        mask[row:row + cut_h, :depth] = False
        mask[row:row + cut_h, width - depth:] = False
    else:
        raise AssertionError("unknown glyph class")
    return mask


def _sequence_plan(rng, train_sizes):
    """Cover the codebook across training and show order independently of color."""
    sequences = [[int(rng.integers(4)) for _ in range(k)] for k in train_sizes]
    anchor = int(rng.integers(4))
    first_position = int(rng.integers(train_sizes[0]))
    other_positions = [p for p in range(train_sizes[1]) if p != first_position]
    second_position = other_positions[int(rng.integers(len(other_positions)))]
    sequences[0][first_position] = anchor
    sequences[1][second_position] = anchor
    fixed = {(0, first_position), (1, second_position)}

    others = [c for c in range(4) if c != anchor]
    rng.shuffle(others)
    delayed = others[0] if len(sequences) >= 3 and rng.random() < 0.65 else None
    if delayed is not None:
        for sequence in sequences[:2]:
            for position, value in enumerate(sequence):
                if value == delayed:
                    sequence[position] = int(rng.choice([c for c in range(4) if c != delayed]))
    for class_id in others:
        slots = [
            (i, p) for i, sequence in enumerate(sequences)
            for p in range(len(sequence))
            if (i, p) not in fixed and (delayed != class_id or i >= 2)
        ]
        slot = slots[int(rng.integers(len(slots)))]
        sequences[slot[0]][slot[1]] = class_id
        fixed.add(slot)
    return [tuple(sequence) for sequence in sequences]


def _test_sequence(rng, count):
    mode = int(rng.integers(4))
    if mode == 0:
        if count == 3:
            return tuple(int(c) for c in rng.permutation([HOLE, BOTTOM, SIDES]))
        values = list(int(c) for c in rng.permutation(4))
        return tuple((values + [int(rng.integers(4))])[:count])
    if mode == 1:
        dominant = int(rng.integers(4))
        result = [dominant] * count
        location = int(rng.integers(count))
        result[location] = int(rng.choice([c for c in range(4) if c != dominant]))
        return tuple(result)
    if mode == 2:
        return (int(rng.integers(4)),) * count
    return tuple(int(rng.integers(4)) for _ in range(count))


def _layout(rng, orientation, count, height, width):
    axis = width if orientation == "horizontal" else height
    max_gap = min(3, (30 - count * axis) // (count - 1))
    assert max_gap >= 1
    gap = int(rng.integers(1, max_gap + 1))
    spare = 30 - count * axis - (count - 1) * gap
    leading = int(rng.integers(0, min(2, spare) + 1))
    trailing = int(rng.integers(0, min(2, spare - leading) + 1))
    if orientation == "horizontal":
        return (int(rng.integers(0, 3)), int(rng.integers(0, 3)), leading, trailing), gap
    return (leading, trailing, int(rng.integers(0, 3)), int(rng.integers(0, 3))), gap


def _assemble(sequence, masks, orientation, margins, gap, carrier, void):
    top, bottom, left, right = margins
    height, width = masks[0].shape
    count = len(sequence)
    if orientation == "horizontal":
        grid = np.full((top + height + bottom,
                        left + count * width + (count - 1) * gap + right), void, dtype=np.int64)
        for index, mask in enumerate(masks):
            col = left + index * (width + gap)
            grid[top:top + height, col:col + width] = np.where(mask, carrier, void)
    else:
        grid = np.full((top + count * height + (count - 1) * gap + bottom,
                        left + width + right), void, dtype=np.int64)
        for index, mask in enumerate(masks):
            row = top + index * (height + gap)
            grid[row:row + height, left:left + width] = np.where(mask, carrier, void)
    return grid


def _output(sequence, labels):
    return np.repeat(np.asarray([labels[c] for c in sequence], dtype=np.int64)[:, None], len(sequence), axis=1)


def _key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    label_colors = tuple(int(v) for v in rng.choice(10, 4, replace=False))
    input_colors = [c for c in range(10) if c not in label_colors]
    fixed_roles = tuple(int(v) for v in rng.choice(input_colors, 2, replace=False))
    recolor_pairs = bool(rng.integers(2))
    orientation = "vertical" if rng.random() < 0.28 else "horizontal"
    height = int(rng.choice([4, 5, 6], p=[0.46, 0.33, 0.21]))
    width = int(rng.choice([4, 5, 6], p=[0.46, 0.33, 0.21]))
    axis = width if orientation == "horizontal" else height
    max_count = min(5, (30 + 1) // (axis + 1))

    train_sizes = [int(rng.integers(3, max_count + 1)) for _ in range(train_count)]
    if orientation == "horizontal" and height == width == 4:
        train_sizes[0] = 3
    if max_count >= 4 and train_sizes[0] == train_sizes[1]:
        train_sizes[1] = 3 + (train_sizes[1] - 2) % (max_count - 2)
    if train_count >= 3:
        for i in range(2, train_count):
            train_sizes[i] = int(rng.integers(2, max_count + 1))
    train_sequences = _sequence_plan(rng, train_sizes)
    test_sizes = [int(rng.choice(train_sizes)) for _ in range(test_count)]
    test_sequences = [_test_sequence(rng, k) for k in test_sizes]

    train, test = [], []
    seen_inputs = set()
    trained_masks = {c: [] for c in range(4)}
    previous_roles = None
    for split, sizes, sequences in (("train", train_sizes, train_sequences),
                                    ("test", test_sizes, test_sequences)):
        destination = train if split == "train" else test
        for pair_index, (count, sequence) in enumerate(zip(sizes, sequences)):
            for attempt in range(100):
                if recolor_pairs:
                    possibilities = [(a, b) for a in input_colors for b in input_colors
                                     if a != b and (a, b) != previous_roles]
                    carrier, void = possibilities[int(rng.integers(len(possibilities)))]
                else:
                    carrier, void = fixed_roles
                masks = []
                for class_id in sequence:
                    available = trained_masks[class_id]
                    unique = {(_key(m)) for m in available}
                    if split == "test" and len(unique) == 1:
                        masks.append(available[int(rng.integers(len(available)))].copy())
                    elif split == "test" and available and rng.random() < 0.4:
                        masks.append(available[int(rng.integers(len(available)))].copy())
                    else:
                        masks.append(_mask(class_id, height, width, rng))
                margins, gap = _layout(rng, orientation, count, height, width)
                if split == "train" and pair_index == 0 and orientation == "horizontal" and height == width == 4:
                    margins, gap = (0, 0, 0, 0), 1
                input_grid = _assemble(sequence, masks, orientation, margins, gap, carrier, void)
                output_grid = _output(sequence, label_colors)
                input_key = _key(input_grid)
                if input_key in seen_inputs or _is_source_pair(input_grid, output_grid):
                    continue
                seen_inputs.add(input_key)
                previous_roles = (carrier, void)
                destination.append({"input": input_grid, "output": output_grid})
                if split == "train":
                    for class_id, mask in zip(sequence, masks):
                        trained_masks[class_id].append(mask.copy())
                break
            else:
                raise AssertionError("failed to construct a distinct pair")
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _runs(flags):
    result = []
    start = None
    for index, value in enumerate(flags):
        if value and start is None:
            start = index
        if not value and start is not None:
            result.append((start, index))
            start = None
    if start is not None:
        result.append((start, len(flags)))
    return result


def _classify(mask):
    height, width = mask.shape
    assert 4 <= height <= 6 and 4 <= width <= 6
    assert bool(np.all(mask[0]))
    missing = np.argwhere(~mask)
    if len(missing) == 0:
        return SOLID
    top, left = np.min(missing, axis=0)
    bottom, right = np.max(missing, axis=0)
    if not bool(np.any(~mask[:, 0])) and not bool(np.any(~mask[:, -1])):
        expected = np.ones_like(mask)
        expected[top:bottom + 1, left:right + 1] = False
        assert np.array_equal(mask, expected)
        assert left >= 1 and right <= width - 2
        if bottom == height - 1:
            return BOTTOM
        assert top >= 1 and bottom <= height - 2
        return HOLE
    assert top >= 1 and bottom <= height - 2
    assert left == 0 and right == width - 1
    side_depth = 0
    while side_depth < width and not mask[top, side_depth]:
        side_depth += 1
    assert 1 <= side_depth <= (width - 1) // 2
    expected = np.ones_like(mask)
    expected[top:bottom + 1, :side_depth] = False
    expected[top:bottom + 1, width - side_depth:] = False
    assert np.array_equal(mask, expected)
    return SIDES


def _decode_with_roles(grid, carrier, void):
    row_runs = _runs(np.any(grid == carrier, axis=1))
    col_runs = _runs(np.any(grid == carrier, axis=0))
    if len(col_runs) >= 2 and len(row_runs) == 1:
        orientation = "horizontal"
        axis_runs, cross_runs = col_runs, row_runs
    elif len(row_runs) >= 2 and len(col_runs) == 1:
        orientation = "vertical"
        axis_runs, cross_runs = row_runs, col_runs
    else:
        raise AssertionError("no unique strip axis")
    assert 2 <= len(axis_runs) <= 5
    axis_size = axis_runs[0][1] - axis_runs[0][0]
    assert all(end - start == axis_size for start, end in axis_runs)
    gaps = [axis_runs[i + 1][0] - axis_runs[i][1] for i in range(len(axis_runs) - 1)]
    assert len(set(gaps)) == 1 and 1 <= gaps[0] <= 3
    assert axis_runs[0][0] <= 2
    axis_extent = grid.shape[1 if orientation == "horizontal" else 0]
    assert axis_extent - axis_runs[-1][1] <= 2
    assert cross_runs[0][0] <= 2
    cross_extent = grid.shape[0 if orientation == "horizontal" else 1]
    assert cross_extent - cross_runs[0][1] <= 2
    masks, sequence = [], []
    for start, end in axis_runs:
        if orientation == "horizontal":
            tile = grid[cross_runs[0][0]:cross_runs[0][1], start:end]
        else:
            tile = grid[start:end, cross_runs[0][0]:cross_runs[0][1]]
        mask = tile == carrier
        sequence.append(_classify(mask))
        masks.append(mask)
    height, width = masks[0].shape
    return {"roles": (carrier, void), "orientation": orientation, "height": height,
            "width": width, "count": len(sequence), "gap": gaps[0],
            "margins": (row_runs[0][0], grid.shape[0] - row_runs[-1][1],
                        col_runs[0][0], grid.shape[1] - col_runs[-1][1]),
            "sequence": tuple(sequence), "masks": masks}


def _decode(grid):
    colors = [int(v) for v in np.unique(grid)]
    assert len(colors) == 2
    candidates = []
    for carrier, void in ((colors[0], colors[1]), (colors[1], colors[0])):
        try:
            candidates.append(_decode_with_roles(grid, carrier, void))
        except AssertionError:
            continue
    assert len(candidates) == 1
    return candidates[0]


def _is_source_pair(input_grid, output_grid):
    if input_grid.shape != (4, 14) or output_grid.shape != (3, 3):
        return False
    for sequence in SOURCE_SEQUENCES:
        masks = [SOURCE_MASKS[c] for c in sequence]
        source_input = _assemble(sequence, masks, "horizontal", (0, 0, 0, 0), 1, 5, 0)
        if np.array_equal(input_grid, source_input) and np.array_equal(output_grid, _output(sequence, SOURCE_LABELS)):
            return True
    return False


def _grid_check(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= dimension <= 30 for dimension in grid.shape)
    assert bool(np.all((0 <= grid) & (grid <= 9)))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _grid_check(inp)
            _grid_check(out)
            input_key, pair_key = _key(inp), (_key(inp), _key(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _is_source_pair(inp, out)
            record = _decode(inp)
            count = record["count"]
            assert out.shape == (count, count)
            labels = []
            for row in out:
                assert bool(np.all(row == row[0]))
                labels.append(int(row[0]))
            record["labels"] = tuple(labels)
            record["output"] = out
            records[split].append(record)
    all_records = records["train"] + records["test"]
    assert len({(r["orientation"], r["height"], r["width"]) for r in all_records}) == 1
    mapping, reverse = {}, {}
    for record in records["train"]:
        for class_id, label in zip(record["sequence"], record["labels"]):
            if class_id in mapping:
                assert mapping[class_id] == label
            else:
                mapping[class_id] = label
            if label in reverse:
                assert reverse[label] == class_id
            else:
                reverse[label] = class_id
    assert set(mapping) == CLASSES and len(reverse) == 4
    trained_counts = {r["count"] for r in records["train"]}
    trained_roles = {r["roles"] for r in records["train"]}
    trained_masks = {class_id: set() for class_id in CLASSES}
    for record in records["train"]:
        for class_id, mask in zip(record["sequence"], record["masks"]):
            trained_masks[class_id].add(_key(mask))
    test_record_ids = {id(record) for record in records["test"]}
    for record in all_records:
        assert set(record["roles"]).isdisjoint(reverse)
        assert np.array_equal(record["output"], _output(record["sequence"], mapping))
        if id(record) in test_record_ids:
            assert record["count"] in trained_counts
            assert record["roles"] in trained_roles or len(trained_roles) >= 2
            for class_id, mask in zip(record["sequence"], record["masks"]):
                assert _key(mask) in trained_masks[class_id] or len(trained_masks[class_id]) >= 2
    # At least one glyph recurs at two positions in different demonstrations.
    assert any(
        first_class == second_class and first_position != second_position
        for i, first in enumerate(records["train"])
        for second in records["train"][i + 1:]
        for first_position, first_class in enumerate(first["sequence"])
        for second_position, second_class in enumerate(second["sequence"])
    )


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    orientations, tile_sizes, strip_counts, gaps, margin_patterns = set(), set(), set(), set(), set()
    input_roles, sequences, structures, puzzles = set(), set(), set(), set()
    repeated_hole_or_side = 0
    test_with_solid = 0
    all_same_test = 0
    training_without_solid = 0
    variable_roles = 0
    shape_variant_puzzles = 0
    later_class_puzzles = 0
    source_sized_pairs = 0
    source_test_order = 0
    repeated_non_solid_test = 0
    two_same_test = 0
    three_same_test = 0
    combined_variation_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        records = [_decode(p["input"]) for split in ("train", "test") for p in puzzle[split]]
        role_set = {r["roles"] for r in records}
        variable_roles += len(role_set) > 1
        later_class_puzzles += len(set(records[0]["sequence"]) | set(records[1]["sequence"])) < 4
        source_sized_pairs += sum(pair["input"].shape == (4, 14)
                                  for split in ("train", "test") for pair in puzzle[split])
        shape_sets = {c: set() for c in CLASSES}
        for index, record in enumerate(records):
            orientations.add(record["orientation"])
            tile_sizes.add((record["height"], record["width"]))
            strip_counts.add(record["count"])
            gaps.add(record["gap"])
            margin_patterns.add(record["margins"])
            input_roles.add(record["roles"])
            sequences.add(record["sequence"])
            structures.add((record["orientation"], record["height"], record["width"],
                            record["count"], record["gap"], record["margins"], record["sequence"]))
            if index < len(puzzle["train"]):
                training_without_solid += SOLID not in record["sequence"]
                repeated_hole_or_side += any(record["sequence"].count(c) >= 2 for c in (HOLE, SIDES))
                for c, mask in zip(record["sequence"], record["masks"]):
                    shape_sets[c].add(_key(mask))
            else:
                test_with_solid += SOLID in record["sequence"]
                all_same_test += len(set(record["sequence"])) == 1
        shape_variant_puzzles += any(len(values) >= 2 for values in shape_sets.values())
        combined_variation_puzzles += (len({r["count"] for r in records}) > 1
                                       and len({r["gap"] for r in records}) > 1
                                       and any(len(values) >= 2 for values in shape_sets.values()))
        for record in records[len(puzzle["train"]):]:
            sequence = record["sequence"]
            source_test_order += sequence == (BOTTOM, SIDES, HOLE)
            repeated_non_solid_test += any(sequence.count(c) >= 2 for c in (HOLE, BOTTOM, SIDES))
            two_same_test += len(sequence) == 2 and len(set(sequence)) == 1
            three_same_test += len(sequence) == 3 and len(set(sequence)) == 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzles) >= 40 and len(structures) >= 150
    assert orientations == {"horizontal", "vertical"}
    assert len(tile_sizes) >= 6 and strip_counts == {2, 3, 4, 5}
    assert gaps == {1, 2, 3} and len(margin_patterns) >= 12
    assert len(input_roles) >= 12 and len(sequences) >= 35
    assert min(repeated_hole_or_side, test_with_solid, all_same_test,
               training_without_solid, variable_roles, shape_variant_puzzles) >= 3
    assert later_class_puzzles >= 3 and source_sized_pairs >= 1 and source_test_order >= 1
    assert repeated_non_solid_test >= 3 and two_same_test >= 1 and three_same_test >= 1
    assert combined_variation_puzzles >= 10
    assert _puzzle_key(generate_puzzle(314159)) == _puzzle_key(generate_puzzle(314159))
    assert _puzzle_key(generate_puzzle(271828, 7, 3)) == _puzzle_key(generate_puzzle(271828, 7, 3))
    for offset, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                     (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + offset, *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
    for offset, counts in enumerate(((1, None), (11, None), (None, 0), (None, 5),
                                     (2, 2), (3, 4), (True, 1), (2.0, 1))):
        try:
            generate_puzzle(2000 + offset, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair count was accepted")
