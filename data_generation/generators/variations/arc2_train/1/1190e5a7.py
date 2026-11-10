"""Generate ARC partitions whose rectangular fields compress to one cell each."""

import numpy as np


# Exact source pairs, represented by all information needed to reconstruct
# their two grids. Source dimensions and palettes remain available otherwise.
_SOURCE_SIGNATURES = frozenset({
    ((11, 11), 1, 8, (3, 9), (4,)),
    ((15, 15), 3, 7, (2,), (1, 10, 13)),
    ((27, 27), 3, 1, (2, 7, 16, 21, 23), (6, 21, 23, 25)),
    ((22, 22), 1, 5, (2, 7, 12, 17), (15, 20)),
})


def _coerce_count(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _choose_counts(rng, num_train, num_test):
    num_train = _coerce_count("num_train", num_train, 2, 10)
    num_test = _coerce_count("num_test", num_test, 1, 4)
    legal = [
        (a, b) for a in range(2, 11) for b in range(1, 5)
        if a > b and (num_train is None or a == num_train)
        and (num_test is None or b == num_test)
    ]
    if not legal:
        raise ValueError("num_train must exceed num_test")
    if len(legal) == 1:
        return legal[0]
    return legal[int(rng.integers(len(legal)))]


def _sample_puzzle_parameters(rng):
    # The source varies colors between its pairs. This is a color reuse
    # regime, not a hidden color dictionary: every input reveals its roles.
    colors = rng.choice(10, size=2, replace=False)
    return {
        "base": (int(colors[0]), int(colors[1])),
        "vary_palette": bool(rng.random() < 0.72),
    }


def _choose_shapes(rng, train_count, test_count):
    # Two crossed demonstrations show that row and column counts are
    # independent, even for the minimum legal training count.
    first = (int(rng.integers(2, 8)), int(rng.integers(2, 8)))
    second = (
        int(rng.choice([n for n in range(2, 8) if n != first[0]])),
        int(rng.choice([n for n in range(2, 8) if n != first[1]])),
    )
    train = [first, second]
    for index in range(2, train_count):
        draw = rng.random()
        if index == 2 and draw < 0.55:
            shape = first
        elif draw < 0.16:
            shape = train[int(rng.integers(len(train)))]
        elif draw < 0.29 and train_count >= 4:
            n = int(rng.integers(2, 11))
            shape = (1, n) if rng.random() < 0.5 else (n, 1)
        else:
            shape = (int(rng.integers(2, 11)), int(rng.integers(2, 11)))
        train.append(shape)

    test = []
    one_axis_train = [s for s in train if 1 in s]
    for _ in range(test_count):
        draw = rng.random()
        if draw < 0.23:
            shape = first[0], second[1]
            if rng.random() < 0.5:
                shape = second[0], first[1]
        elif draw < 0.45:
            shape = train[int(rng.integers(len(train)))]
        elif draw < 0.65:
            # New axis counts follow the same demonstrated counting rule.
            shape = (int(rng.integers(8, 16)), int(rng.integers(8, 16)))
        elif draw < 0.75 and one_axis_train:
            demonstration = one_axis_train[int(rng.integers(len(one_axis_train)))]
            n = int(rng.integers(2, 11))
            shape = (1, n) if demonstration[0] == 1 else (n, 1)
        elif draw < 0.87:
            shape = ((2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3))[
                int(rng.integers(6))
            ]
        else:
            shape = (int(rng.integers(2, 10)), int(rng.integers(2, 10)))
        test.append(shape)
    return train, test


def _choose_palettes(rng, parameters, train_count, test_count):
    base = parameters["base"]
    total = train_count + test_count
    if not parameters["vary_palette"]:
        return [base] * total

    train = [base, (base[1], base[0])]
    while len(train) < train_count:
        draw = rng.random()
        if draw < 0.25:
            train.append(train[int(rng.integers(len(train)))])
        elif draw < 0.55:
            retained = int(rng.choice(base))
            other = int(rng.choice([c for c in range(10) if c != retained]))
            train.append((retained, other))
        else:
            colors = rng.choice(10, size=2, replace=False)
            train.append((int(colors[0]), int(colors[1])))

    tests = []
    used_dividers = {pair[1] for pair in train}
    for _ in range(test_count):
        draw = rng.random()
        if draw < 0.36 and len(used_dividers) < 10:
            novel = int(rng.choice([c for c in range(10) if c not in used_dividers]))
            field = int(rng.choice([c for c in range(10) if c != novel]))
            tests.append((field, novel))
        elif draw < 0.65:
            tests.append(train[int(rng.integers(len(train)))])
        else:
            colors = rng.choice(10, size=2, replace=False)
            tests.append((int(colors[0]), int(colors[1])))
    return train + tests


def _choose_dimension(rng, parts, bias):
    minimum = max(3, 2 * parts - 1)
    if bias == "dense":
        upper = min(30, minimum + 3)
        return int(rng.integers(minimum, upper + 1))
    if bias == "spacious":
        lower = max(minimum, 23)
        return int(rng.integers(lower, 31))
    if bias == "compact":
        upper = min(30, minimum + 9)
        return int(rng.integers(minimum, upper + 1))
    return int(rng.integers(minimum, 31))


def _composition(rng, total, parts):
    if parts == 1:
        return (total,)
    cuts = sorted(int(v) for v in rng.choice(np.arange(1, total), parts - 1, replace=False))
    marks = [0] + cuts + [total]
    return tuple(b - a for a, b in zip(marks, marks[1:]))


def _segments(rng, dimension, parts):
    if parts == 1:
        return (dimension,)
    field_cells = dimension - parts + 1
    extra = field_cells - parts
    mode = int(rng.integers(7))
    if mode == 0:
        lengths = list(_composition(rng, field_cells, parts))
    elif mode == 1:
        # Balanced intervals, including exactly regular dividers when the
        # surplus divides evenly.
        lengths = [field_cells // parts] * parts
        for index in rng.permutation(parts)[:field_cells % parts]:
            lengths[int(index)] += 1
    elif mode == 2 and parts >= 3:
        # Equal interior intervals give an arithmetic divider progression.
        max_interior = max(1, (field_cells - 2) // (parts - 2))
        interior = int(rng.integers(1, max_interior + 1))
        edges = field_cells - interior * (parts - 2)
        left = int(rng.integers(1, edges))
        lengths = [left] + [interior] * (parts - 2) + [edges - left]
    elif mode in (3, 4, 6) and extra > 0:
        # A long boundary interval clusters lines at the opposite edge.
        # A long central interval splits them toward both ends.
        if mode == 3:
            focus = 0
        elif mode == 4:
            focus = parts - 1
        else:
            focus = parts // 2
        long_extra = int(rng.integers(max(1, 3 * extra // 4), extra + 1))
        lengths = list(_composition(rng, field_cells - long_extra, parts))
        lengths[focus] += long_extra
    elif mode == 5 and extra > 0:
        # Large outer margins place the divider group near the middle.
        boundary_extra = int(rng.integers(max(1, 3 * extra // 4), extra + 1))
        left_extra = int(rng.integers(boundary_extra + 1))
        lengths = list(_composition(rng, field_cells - boundary_extra, parts))
        lengths[0] += left_extra
        lengths[-1] += boundary_extra - left_extra
    else:
        lengths = list(_composition(rng, field_cells, parts))
    if rng.random() < 0.5:
        lengths.reverse()
    return tuple(lengths)


def _divider_positions(lengths):
    cursor = lengths[0]
    positions = []
    for length in lengths[1:]:
        positions.append(cursor)
        cursor += 1 + length
    return tuple(positions)


def _render_partition(height, width, rows, cols, field, divider):
    inp = np.full((height, width), field, dtype=np.int8)
    if rows:
        inp[list(rows), :] = divider
    if cols:
        inp[:, list(cols)] = divider
    out = np.full((len(rows) + 1, len(cols) + 1), field, dtype=np.int8)
    return {"input": inp, "output": out}


def _sample_pair(rng, output_shape, palette, dimensions=None, bias=None):
    transposed = bool(rng.integers(2))
    raw_shape = output_shape[::-1] if transposed else output_shape
    if dimensions is not None:
        raw_dimensions = dimensions[::-1] if transposed else dimensions
    else:
        raw_dimensions = None
    if bias is None:
        bias = ("dense", "compact", "broad", "spacious")[int(rng.integers(4))]
    dims = tuple(
        raw_dimensions[axis] if raw_dimensions is not None
        else _choose_dimension(rng, raw_shape[axis], bias if rng.random() < 0.65 else "broad")
        for axis in range(2)
    )
    row_lengths = _segments(rng, dims[0], raw_shape[0])
    col_lengths = _segments(rng, dims[1], raw_shape[1])
    pair = _render_partition(
        dims[0], dims[1], _divider_positions(row_lengths),
        _divider_positions(col_lengths), palette[0], palette[1],
    )
    if transposed:
        pair = {key: value.T.copy() for key, value in pair.items()}
    # Reflection acts on the whole solved pair, so geometry and output
    # axes remain aligned even for one-axis partitions.
    if rng.random() < 0.5:
        pair = {key: value[::-1, :].copy() for key, value in pair.items()}
    if rng.random() < 0.5:
        pair = {key: value[:, ::-1].copy() for key, value in pair.items()}
    return pair


def _fallback_pair(output_shape, palette, seen):
    # The random path should nearly always succeed. This finite search
    # guarantees legal count requests even if several tiny layouts collide.
    oh, ow = output_shape
    for height in range(max(3, 2 * oh - 1), 31):
        for width in range(max(3, 2 * ow - 1), 31):
            for row_shift in range(oh):
                row_lengths = [1] * oh
                row_lengths[row_shift] += height - (2 * oh - 1)
                for col_shift in range(ow):
                    col_lengths = [1] * ow
                    col_lengths[col_shift] += width - (2 * ow - 1)
                    pair = _render_partition(
                        height, width, _divider_positions(row_lengths),
                        _divider_positions(col_lengths), palette[0], palette[1],
                    )
                    if _array_key(pair["input"]) in seen:
                        continue
                    try:
                        _analyze_pair(pair)
                    except AssertionError:
                        continue
                    return pair
    raise RuntimeError("no unused partition remains for this output shape")


def _array_key(grid):
    return (grid.shape, grid.tobytes())


def _puzzle_key(puzzle):
    return tuple(
        tuple((_array_key(p["input"]), _array_key(p["output"])) for p in puzzle[side])
        for side in ("train", "test")
    )


def _analyze_input(grid):
    # Field cells occupy all corners because dividers are internal. This
    # remains true when the field is the minority color.
    field = int(grid[0, 0])
    assert all(int(grid[r, c]) == field for r, c in (
        (0, 0), (0, -1), (-1, 0), (-1, -1)
    ))
    colors = {int(v) for v in np.unique(grid)}
    assert len(colors) == 2 and field in colors
    divider = next(iter(colors - {field}))
    rows = tuple(int(v) for v in np.flatnonzero(np.all(grid == divider, axis=1)))
    cols = tuple(int(v) for v in np.flatnonzero(np.all(grid == divider, axis=0)))
    assert rows or cols
    for positions, length in ((rows, grid.shape[0]), (cols, grid.shape[1])):
        assert all(1 <= p <= length - 2 for p in positions)
        assert all(b - a >= 2 for a, b in zip(positions, positions[1:]))
    expected = np.full(grid.shape, field, dtype=grid.dtype)
    if rows:
        expected[list(rows), :] = divider
    if cols:
        expected[:, list(cols)] = divider
    assert np.array_equal(grid, expected)
    return {
        "field": field, "divider": divider, "rows": rows, "cols": cols,
        "input_shape": grid.shape, "output_shape": (len(rows) + 1, len(cols) + 1),
    }


def _analyze_pair(pair):
    info = _analyze_input(pair["input"])
    assert pair["output"].shape == info["output_shape"]
    assert np.all(pair["output"] == info["field"])
    assert pair["input"].shape != pair["output"].shape
    signature = (
        info["input_shape"], info["field"], info["divider"],
        info["rows"], info["cols"],
    )
    assert signature not in _SOURCE_SIGNATURES
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    infos = {"train": [], "test": []}
    for side in ("train", "test"):
        for pair in puzzle[side]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            ikey = _array_key(pair["input"])
            pkey = ikey, _array_key(pair["output"])
            assert ikey not in inputs and pkey not in pairs
            inputs.add(ikey)
            pairs.add(pkey)
            infos[side].append(_analyze_pair(pair))
    # The independent row and column counts must both be demonstrated.
    train_shapes = [info["output_shape"] for info in infos["train"]]
    assert len({shape[0] for shape in train_shapes}) >= 2
    assert len({shape[1] for shape in train_shapes}) >= 2
    # A one-axis test is supported by a demonstration of that orientation.
    for info in infos["test"]:
        shape = info["output_shape"]
        if 1 in shape:
            assert any(
                (shape[0] == 1 and example[0] == 1)
                or (shape[1] == 1 and example[1] == 1)
                for example in train_shapes
            )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    train_shapes, test_shapes = _choose_shapes(rng, train_count, test_count)
    palettes = _choose_palettes(rng, parameters, train_count, test_count)
    child_seeds = rng.integers(0, np.iinfo(np.int64).max, size=train_count + test_count)
    puzzle = {"train": [], "test": []}
    seen = set()
    shared_dimensions = None
    if train_count >= 2 and rng.random() < 0.38:
        shared_dimensions = tuple(
            int(rng.integers(max(2 * train_shapes[0][axis] - 1,
                                 2 * train_shapes[1][axis] - 1, 17), 31))
            for axis in range(2)
        )
    for side, shapes, offset in (
        ("train", train_shapes, 0), ("test", test_shapes, train_count)
    ):
        for index, shape in enumerate(shapes):
            pair_rng = np.random.default_rng(int(child_seeds[offset + index]))
            dimensions = shared_dimensions if side == "train" and index < 2 else None
            bias = None
            if side == "test" and shape[0] >= 8 and shape[1] >= 8:
                bias = "dense" if pair_rng.random() < 0.8 else None
            elif side == "test" and shape[0] * shape[1] <= 12:
                bias = "spacious" if pair_rng.random() < 0.7 else None
            for _ in range(32):
                pair = _sample_pair(
                    pair_rng, shape, palettes[offset + index],
                    dimensions=dimensions, bias=bias,
                )
                key = _array_key(pair["input"])
                if key in seen:
                    continue
                try:
                    _analyze_pair(pair)
                except AssertionError:
                    continue
                seen.add(key)
                puzzle[side].append(pair)
                break
            else:
                pair = _fallback_pair(shape, palettes[offset + index], seen)
                seen.add(_array_key(pair["input"]))
                puzzle[side].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _predict_from_examples(train, test_input):
    # Only visible training pairs and the test input are used here.
    for pair in train:
        info = _analyze_input(pair["input"])
        assert pair["output"].shape == info["output_shape"]
        assert np.all(pair["output"] == info["field"])
    info = _analyze_input(test_input)
    return np.full(info["output_shape"], info["field"], dtype=np.int8)


def self_test() -> None:
    train_counts, test_counts, puzzle_keys = set(), set(), set()
    palettes, layouts, input_shapes, output_shapes = set(), set(), set(), set()
    stable, varying, novel_divider = 0, 0, 0
    repeated_outputs, repeated_new_size, equal_input_new_output = 0, 0, 0
    recombined, familiar, new_count = 0, 0, 0
    test_regular, test_irregular, test_left, test_right = 0, 0, 0, 0
    test_middle, test_broad_narrow = 0, 0
    small, large, thin, dense_test, minority_test, one_axis = 0, 0, 0, 0, 0, 0
    sparse_test, six_columns, portrait, landscape = 0, 0, 0, 0

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(_puzzle_key(puzzle))
        train_infos = [_analyze_pair(pair) for pair in puzzle["train"]]
        test_infos = [_analyze_pair(pair) for pair in puzzle["test"]]
        all_infos = train_infos + test_infos
        local_palettes = {(i["field"], i["divider"]) for i in all_infos}
        stable += len(local_palettes) == 1
        varying += len(local_palettes) > 1
        train_dividers = {i["divider"] for i in train_infos}
        novel_divider += any(i["divider"] not in train_dividers for i in test_infos)
        train_out = {i["output_shape"] for i in train_infos}
        train_heights = {s[0] for s in train_out}
        train_widths = {s[1] for s in train_out}
        repeated_outputs += len(train_out) < len(train_infos)
        repeated_new_size += any(
            a["output_shape"] == b["output_shape"] and a["input_shape"] != b["input_shape"]
            for j, a in enumerate(train_infos)
            for b in train_infos[j + 1:]
        )
        equal_input_new_output += any(
            a["input_shape"] == b["input_shape"] and a["output_shape"] != b["output_shape"]
            for j, a in enumerate(train_infos)
            for b in train_infos[j + 1:]
        )
        for info in test_infos:
            shape = info["output_shape"]
            familiar += shape in train_out
            recombined += shape not in train_out and shape[0] in train_heights and shape[1] in train_widths
            new_count += shape[0] not in train_heights or shape[1] not in train_widths
        for side in ("train", "test"):
            for pair in puzzle[side]:
                info = _analyze_pair(pair)
                palettes.add((info["field"], info["divider"]))
                input_shapes.add(info["input_shape"])
                output_shapes.add(info["output_shape"])
                layouts.add((info["input_shape"], info["rows"], info["cols"]))
                h, w = info["input_shape"]
                small += min(h, w) <= 7
                large += max(h, w) >= 27
                thin += max(h, w) >= 2 * min(h, w)
                one_axis += 1 in info["output_shape"]
                portrait += h > w
                landscape += w > h
                six_columns += info["output_shape"][1] >= 6
                if side == "test":
                    predicted = _predict_from_examples(puzzle["train"], pair["input"].copy())
                    assert np.array_equal(predicted, pair["output"])
                    rows, cols = info["rows"], info["cols"]
                    for positions, length in ((rows, h), (cols, w)):
                        boundaries = (-1,) + positions + (length,)
                        intervals = np.diff(boundaries) - 1
                        test_broad_narrow += max(intervals) >= 5 and min(intervals) <= 2
                        if len(positions) >= 3:
                            steps = np.diff(positions)
                            test_regular += bool(np.all(steps == steps[0]))
                            test_irregular += bool(np.any(steps != steps[0]))
                        if positions:
                            test_left += positions[-1] <= 0.4 * length
                            test_right += positions[0] >= 0.6 * length
                        if len(positions) >= 2:
                            test_middle += positions[0] >= 0.25 * length and positions[-1] <= 0.75 * length
                    dense_test += info["output_shape"][0] >= 8 and info["output_shape"][1] >= 8
                    minority_test += int(np.count_nonzero(pair["input"] == info["field"])) < h * w / 2
                    sparse_test += info["output_shape"][0] * info["output_shape"][1] <= 12 and h * w >= 300

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_keys) >= 40
    assert len(palettes) >= 25 and len(input_shapes) >= 80
    assert len(output_shapes) >= 30 and len(layouts) >= 200
    assert stable >= 4 and varying >= 20 and novel_divider >= 8
    assert repeated_outputs >= 4 and repeated_new_size >= 4 and equal_input_new_output >= 4
    assert familiar >= 5 and recombined >= 4 and new_count >= 5
    assert test_regular >= 5 and test_irregular >= 12
    assert test_left >= 2 and test_right >= 2 and test_middle >= 1
    assert test_broad_narrow >= 8
    assert small >= 8 and large >= 20 and thin >= 8
    assert dense_test >= 5 and minority_test >= 5 and one_axis >= 2
    assert sparse_test >= 2 and six_columns >= 15
    assert portrait >= 20 and landscape >= 20

    for index, (a, b) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(1000 + index, a, b)
        assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
        validate_generated_puzzle(puzzle)
    for index, (a, b) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + index, a, b)
        assert (a is None or len(puzzle["train"]) == a)
        assert (b is None or len(puzzle["test"]) == b)
        validate_generated_puzzle(puzzle)
    assert _puzzle_key(generate_puzzle(314159, 7, 3)) == _puzzle_key(generate_puzzle(314159, 7, 3))
    assert _puzzle_key(generate_puzzle(271828)) == _puzzle_key(generate_puzzle(271828))
    assert len({_puzzle_key(generate_puzzle(s, 6, 2)) for s in range(3000, 3008)}) >= 7
    for request in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4}, {"num_train": 2.5, "num_test": 1},
        {"num_train": 3, "num_test": True},
    ):
        try:
            generate_puzzle(9999, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {request}")
