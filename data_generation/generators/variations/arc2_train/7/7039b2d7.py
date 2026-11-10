"""Generate lattices whose field rectangles collapse to single cells."""

import numpy as np


def _check_count(name, value, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _counts(rng, num_train, num_test):
    _check_count("num_train", num_train, 2, 10)
    _check_count("num_test", num_test, 1, 4)
    if num_test is None:
        if num_train is None:
            num_test = int(rng.choice([1, 2, 3, 4], p=[.55, .23, .14, .08]))
        else:
            num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if num_train is None:
        choices = [n for n in range(2, 11) if n > num_test]
        weights = np.array([1, 7, 4, 3, 2, 1.5, 1, 1, 1], dtype=float)
        weights = weights[np.array(choices) - 2]
        num_train = int(rng.choice(choices, p=weights / weights.sum()))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _sample_shape(rng):
    values = [2, 3, 4, 5, 6, 7, 8]
    probabilities = [.18, .24, .22, .17, .09, .06, .04]
    return (int(rng.choice(values, p=probabilities)),
            int(rng.choice(values, p=probabilities)))


def _different_shape(rng, other, both=False):
    for _ in range(32):
        shape = _sample_shape(rng)
        if shape != other and (not both or (shape[0] != other[0] and shape[1] != other[1])):
            return shape
    return (3 if other[0] != 3 else 4, 4 if other[1] != 4 else 3)


def _shape_plan(rng, train_count, test_count):
    train = [_sample_shape(rng) for _ in range(train_count)]
    test = [_sample_shape(rng) for _ in range(test_count)]
    comparison = str(rng.choice(["free", "same_output", "same_canvas", "combined"],
                                p=[.27, .24, .26, .23]))
    if comparison == "same_output" and train_count == 2:
        comparison = "same_canvas"
    if comparison == "combined" and train_count < 4:
        comparison = "same_canvas" if train_count == 2 else "same_output"
    canvas_partner = None
    if comparison == "same_output":
        train[1] = train[0]
        if train_count >= 3:
            train[2] = _different_shape(rng, train[0], both=True)
    elif comparison == "same_canvas":
        train[1] = _different_shape(rng, train[0], both=True)
        canvas_partner = (0, 1)
    elif comparison == "combined":
        train[1] = train[0]
        train[2] = _different_shape(rng, train[0], both=True)
        canvas_partner = (0, 2)
    else:
        train[1] = _different_shape(rng, train[0], both=True)

    test_mode = str(rng.choice(["free", "familiar", "recombine"], p=[.34, .31, .35]))
    if test_mode == "familiar":
        test[0] = train[int(rng.integers(train_count))]
    elif test_mode == "recombine":
        candidates = [(a[0], b[1]) for a in train for b in train
                      if a[0] != b[0] and a[1] != b[1]]
        if candidates:
            test[0] = candidates[int(rng.integers(len(candidates)))]
    return train, test, canvas_partner


def _palettes(rng, total, train_count):
    mode = str(rng.choice(["fixed", "shared_field", "shared_divider", "varied", "recombine"],
                          p=[.15, .16, .16, .28, .25]))
    colors = [int(x) for x in rng.permutation(10)]
    if mode == "fixed":
        return [(colors[0], colors[1])] * total
    if mode == "shared_field":
        others = [colors[1 + (i % 9)] for i in range(total)]
        return [(colors[0], value) for value in others]
    if mode == "shared_divider":
        others = [colors[1 + (i % 9)] for i in range(total)]
        return [(value, colors[0]) for value in others]
    if mode == "recombine":
        palettes = [(colors[0], colors[1]), (colors[2], colors[3])]
        for _ in range(2, train_count):
            palettes.append((colors[int(rng.choice([0, 2]))],
                             colors[int(rng.choice([1, 3]))]))
        palettes.append((colors[0], colors[3]))
        for _ in range(train_count + 1, total):
            palettes.append((colors[int(rng.choice([0, 2]))],
                             colors[int(rng.choice([1, 3]))]))
        return palettes
    palettes = []
    for _ in range(total):
        selected = rng.choice(10, size=2, replace=False)
        palettes.append((int(selected[0]), int(selected[1])))
    return palettes


def _edge_plan(rng, train_count, test_count):
    # A puzzle may stay interior-only, or show a range of edge placements.
    rate = float(rng.choice([0, .25, .5, .8], p=[.16, .26, .40, .18]))
    plan = [bool(rng.random() < rate) for _ in range(train_count + test_count)]
    if any(plan[train_count:]) and not any(plan[:train_count]):
        plan[int(rng.integers(train_count))] = True
    return plan


def _axis_edges(rng, edge_requested, band_count, target):
    if not edge_requested:
        return False, False
    # Prefer either boundary, but sometimes draw both. A fixed canvas may
    # leave room for only one boundary or none.
    possibilities = [(True, False), (False, True), (True, True)]
    order = rng.permutation(len(possibilities))
    for index in order:
        first, last = possibilities[int(index)]
        overhead = band_count - 1 + int(first) + int(last)
        if target is None or target - overhead >= band_count:
            return first, last
    return False, False


def _axis_lengths(rng, bands, overhead, target, unit, all_unit, regime):
    if all_unit and (target is None or target == bands + overhead):
        return [1] * bands
    minimum = 1 if unit else 2
    if target is not None:
        total = target - overhead
        if total < bands:
            raise ValueError("canvas too small for output bands")
        if total < bands * minimum:
            minimum = 1
        lengths = np.full(bands, minimum, dtype=int)
        pinned = int(rng.integers(bands)) if unit else None
        if pinned is not None:
            lengths[pinned] = 1
        extra = int(total - lengths.sum())
        if extra < 0:
            raise ValueError("canvas too small for band lengths")
        eligible = [i for i in range(bands) if i != pinned]
        if not eligible:
            eligible = [0]
        style = str(rng.choice(["regular", "mild", "uneven", "random"],
                               p=[.22, .29, .26, .23]))
        if style == "regular":
            for index in rng.permutation(eligible):
                lengths[int(index)] += extra // len(eligible)
            for index in rng.permutation(eligible)[:extra % len(eligible)]:
                lengths[int(index)] += 1
        else:
            weights = np.ones(len(eligible), dtype=float)
            if style == "mild":
                weights = rng.uniform(.7, 1.3, len(eligible))
            elif style == "uneven":
                weights[int(rng.integers(len(eligible)))] = 8.0
            else:
                weights = rng.uniform(.1, 2.5, len(eligible))
            additions = rng.multinomial(extra, weights / weights.sum())
            for index, addition in zip(eligible, additions):
                lengths[index] += int(addition)
        return [int(x) for x in lengths]

    styles = ["regular", "mild", "uneven", "random"]
    probabilities = [.25, .28, .25, .22] if regime == "source_train" else [.22, .23, .30, .25]
    for _ in range(80):
        style = str(rng.choice(styles, p=probabilities))
        if style == "regular":
            upper = min(8, (30 - overhead) // bands)
            if upper < minimum:
                continue
            if minimum == 1 and rng.random() < .25:
                size = 1
            else:
                size = int(rng.integers(minimum, upper + 1))
            lengths = [size] * bands
        elif style == "mild":
            base = int(rng.integers(max(2, minimum), 7))
            lengths = [max(minimum, base + int(rng.integers(-1, 2))) for _ in range(bands)]
        elif style == "uneven":
            lengths = [int(rng.integers(minimum, 5)) for _ in range(bands)]
            lengths[int(rng.integers(bands))] = int(rng.integers(9, 17))
        else:
            lengths = [int(rng.integers(minimum, 11)) for _ in range(bands)]
        if unit:
            lengths[int(rng.integers(bands))] = 1
        if sum(lengths) + overhead <= 30:
            return lengths
    # Legal deterministic fallback for dense 7- or 8-band axes.
    lengths = [minimum] * bands
    if unit:
        lengths[int(rng.integers(bands))] = 1
    return lengths


def _axis_layout(lengths, first, last):
    lines = []
    cursor = 0
    if first:
        lines.append(cursor)
        cursor += 1
    for index, length in enumerate(lengths):
        cursor += length
        if index + 1 < len(lengths):
            lines.append(cursor)
            cursor += 1
    if last:
        lines.append(cursor)
        cursor += 1
    return cursor, lines


def _grid(height, width, field, divider, rows, columns):
    grid = np.full((height, width), field, dtype=np.int8)
    grid[np.asarray(rows, dtype=int), :] = divider
    grid[:, np.asarray(columns, dtype=int)] = divider
    return grid


_SOURCE_SPECS = (
    (27, 27, 5, 8, (9, 14, 19, 24), (6, 16, 19, 25), (5, 5)),
    (21, 21, 5, 3, (10, 13), (3, 8, 18), (3, 4)),
    (10, 10, 1, 4, (5,), (0, 2, 8), (2, 3)),
    (16, 16, 1, 3, (6, 14), (2, 4, 6, 9), (3, 5)),
)


def _is_source_copy(inp, out):
    for height, width, field, divider, rows, columns, shape in _SOURCE_SPECS:
        if inp.shape == (height, width) and out.shape == shape:
            source = _grid(height, width, field, divider, rows, columns)
            if np.array_equal(inp, source) and np.all(out == field):
                return True
    return False


def _make_pair(rng, shape, palette, edge_requested, regime, canvas=None):
    row_bands, col_bands = shape
    if canvas is None and rng.random() < (.27 if regime == "source_train" else .19):
        side = int(rng.integers(20, 30)) if rng.random() < .7 else int(rng.integers(10, 20))
        if side >= 2 * max(shape) - 1:
            canvas = (side, side)
    if edge_requested:
        edge_axis = int(rng.integers(2))
    else:
        edge_axis = -1
    edges = []
    for axis, bands in enumerate(shape):
        target = None if canvas is None else canvas[axis]
        edges.append(_axis_edges(rng, axis == edge_axis or
                                 (edge_requested and axis != edge_axis and rng.random() < .35),
                                 bands, target))
    if edge_requested and not any(a or b for a, b in edges):
        # A same-canvas comparison can be too tight for a chosen edge.
        edge_requested = False

    unit_pattern = int(rng.choice(4, p=[.34, .22, .22, .22]))
    force_rows = unit_pattern in (1, 3)
    force_cols = unit_pattern in (2, 3)
    all_unit = canvas is None and rng.random() < .045
    if all_unit:
        force_rows = force_cols = True
    layouts = []
    for axis, bands in enumerate(shape):
        first, last = edges[axis]
        overhead = bands - 1 + int(first) + int(last)
        target = None if canvas is None else canvas[axis]
        lengths = _axis_lengths(rng, bands, overhead, target,
                                force_rows if axis == 0 else force_cols,
                                all_unit, regime)
        layouts.append(_axis_layout(lengths, first, last))
    height, rows = layouts[0]
    width, columns = layouts[1]
    field, divider = palette
    inp = _grid(height, width, field, divider, rows, columns)
    out = np.full(shape, field, dtype=np.int8)
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    train_shapes, test_shapes, canvas_partner = _shape_plan(rng, train_count, test_count)
    palettes = _palettes(rng, train_count + test_count, train_count)
    edge_plan = _edge_plan(rng, train_count, test_count)
    pair_seeds = rng.integers(0, 2**63, size=train_count + test_count, dtype=np.int64)
    pairs = []
    seen = set()
    for index, shape in enumerate(train_shapes + test_shapes):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        canvas = None
        if canvas_partner is not None:
            if index == canvas_partner[0]:
                partner_shape = train_shapes[canvas_partner[1]]
                minimum = [max(13, 2 * max(shape[axis], partner_shape[axis]) + 2)
                           for axis in range(2)]
                if pair_rng.random() < .4:
                    side = int(pair_rng.integers(max(minimum), 30))
                    canvas = (side, side)
                else:
                    canvas = tuple(int(pair_rng.integers(value, 30)) for value in minimum)
            elif index == canvas_partner[1]:
                canvas = pairs[canvas_partner[0]]["input"].shape
        for _ in range(64):
            pair = _make_pair(pair_rng, shape, palettes[index], edge_plan[index],
                              "source_train" if index < train_count else "target_test", canvas)
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen and not _is_source_copy(pair["input"], pair["output"]):
                seen.add(key)
                pairs.append(pair)
                break
        else:
            raise RuntimeError("could not construct a distinct lattice")
    puzzle = {"train": pairs[:train_count], "test": pairs[train_count:]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _runs(mask):
    result = []
    count = 0
    for value in mask:
        if value:
            count += 1
        elif count:
            result.append(count)
            count = 0
    if count:
        result.append(count)
    return tuple(result)


def _analyze(pair):
    inp, out = pair["input"], pair["output"]
    colors = np.unique(inp)
    assert len(colors) == 2
    candidates = []
    for color in colors:
        rows = np.flatnonzero(np.all(inp == color, axis=1))
        cols = np.flatnonzero(np.all(inp == color, axis=0))
        if len(rows) and len(cols):
            candidates.append((int(color), rows, cols))
    assert len(candidates) == 1
    divider, rows, cols = candidates[0]
    field = int(colors[0]) if int(colors[1]) == divider else int(colors[1])
    assert len(rows) >= 1 and len(cols) >= 1
    assert not len(rows) > 1 or np.all(np.diff(rows) > 1)
    assert not len(cols) > 1 or np.all(np.diff(cols) > 1)
    expected = _grid(inp.shape[0], inp.shape[1], field, divider, rows, cols)
    assert np.array_equal(inp, expected)
    row_mask = np.ones(inp.shape[0], dtype=bool)
    col_mask = np.ones(inp.shape[1], dtype=bool)
    row_mask[rows] = False
    col_mask[cols] = False
    row_lengths, col_lengths = _runs(row_mask), _runs(col_mask)
    assert 2 <= len(row_lengths) <= 8 and 2 <= len(col_lengths) <= 8
    assert out.shape == (len(row_lengths), len(col_lengths))
    assert np.all(out == field)
    edge = (int(rows[0]) == 0 or int(rows[-1]) == inp.shape[0] - 1 or
            int(cols[0]) == 0 or int(cols[-1]) == inp.shape[1] - 1)
    return {"field": field, "divider": divider, "rows": row_lengths,
            "cols": col_lengths, "edge": edge}


def _predict_from_training(train, test_input):
    # The predictor receives only demonstrations and an input. It verifies
    # the rule on the demonstrations before applying it to a new grid.
    for pair in train:
        _analyze(pair)
    colors = np.unique(test_input)
    assert len(colors) == 2
    candidates = []
    for color in colors:
        rows = np.flatnonzero(np.all(test_input == color, axis=1))
        cols = np.flatnonzero(np.all(test_input == color, axis=0))
        if len(rows) and len(cols):
            candidates.append((int(color), rows, cols))
    assert len(candidates) == 1
    divider, rows, cols = candidates[0]
    field = int(colors[0]) if int(colors[1]) == divider else int(colors[1])
    row_mask = np.ones(test_input.shape[0], dtype=bool)
    col_mask = np.ones(test_input.shape[1], dtype=bool)
    row_mask[rows] = False
    col_mask[cols] = False
    return np.full((len(_runs(row_mask)), len(_runs(col_mask))), field, dtype=np.int8)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            key = (pair["input"].shape, pair["input"].tobytes())
            assert key not in seen
            seen.add(key)
            assert not _is_source_copy(pair["input"], pair["output"])
            analyses[split].append(_analyze(pair))
    # A test with an edge divider has an edge example in training. The
    # divider role itself is unambiguous from full rows and columns in each
    # input; no hidden palette mapping is needed.
    if any(item["edge"] for item in analyses["test"]):
        assert any(item["edge"] for item in analyses["train"])
    for pair in test:
        predicted = _predict_from_training(train, pair["input"].copy())
        assert np.array_equal(predicted, pair["output"])


def _equal_puzzles(a, b):
    return all(len(a[split]) == len(b[split]) and
               all(np.array_equal(x["input"], y["input"]) and
                   np.array_equal(x["output"], y["output"])
                   for x, y in zip(a[split], b[split]))
               for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert sum(len(p["train"]) == 3 and len(p["test"]) == 1 for p in puzzles) >= 3
    palettes = set()
    canvas_shapes = set()
    output_shapes = set()
    layouts = set()
    band_layouts = set()
    varied_palettes = fixed_palettes = 0
    repeated_output = same_canvas = familiar_test = recombined_test = 0
    square = large_square = equal_axis = all_unit = long_train = 0
    edges = {"train": set(), "test": set()}
    unit_categories = {"train": set(), "test": set()}
    for puzzle in puzzles:
        train, test = puzzle["train"], puzzle["test"]
        train_palettes = {(a["field"], a["divider"]) for a in map(_analyze, train)}
        fixed_palettes += len(train_palettes) == 1 and len({(_analyze(p)["field"], _analyze(p)["divider"]) for p in test}) == 1 and all((_analyze(p)["field"], _analyze(p)["divider"]) in train_palettes for p in test)
        varied_palettes += len(train_palettes) > 1
        train_output_shapes = {p["output"].shape for p in train}
        train_heights = {s[0] for s in train_output_shapes}
        train_widths = {s[1] for s in train_output_shapes}
        repeated_output += any(a["output"].shape == b["output"].shape and a["input"].shape != b["input"].shape for i, a in enumerate(train) for b in train[i+1:])
        same_canvas += any(a["input"].shape == b["input"].shape and a["output"].shape != b["output"].shape for i, a in enumerate(train) for b in train[i+1:])
        familiar_test += any(p["output"].shape in train_output_shapes for p in test)
        recombined_test += any(p["output"].shape not in train_output_shapes and p["output"].shape[0] in train_heights and p["output"].shape[1] in train_widths for p in test)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                item = _analyze(pair)
                palettes.add((item["field"], item["divider"]))
                canvas_shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                layouts.add((pair["input"].shape, tuple(item["rows"]), tuple(item["cols"])))
                band_layouts.add((item["rows"], item["cols"]))
                edges[split].add(item["edge"])
                units = (1 in item["rows"], 1 in item["cols"])
                unit_categories[split].add(sum(units))
                square += pair["input"].shape[0] == pair["input"].shape[1]
                large_square += pair["input"].shape[0] == pair["input"].shape[1] and pair["input"].shape[0] >= 20
                equal_axis += len(set(item["rows"])) == 1 or len(set(item["cols"])) == 1
                all_unit += set(item["rows"]) == {1} and set(item["cols"]) == {1}
                if split == "train":
                    long_train += max(item["rows"] + item["cols"]) >= 9
    assert len(palettes) >= 20 and len(canvas_shapes) >= 40
    assert len(output_shapes) >= 20 and len(layouts) >= 100 and len(band_layouts) >= 100
    assert fixed_palettes >= 2 and varied_palettes >= 15
    assert repeated_output >= 3 and same_canvas >= 3
    assert familiar_test >= 3 and recombined_test >= 3
    assert square >= 10 and large_square >= 4 and equal_axis >= 10 and all_unit >= 2 and long_train >= 10
    assert all(edges[split] == {False, True} for split in edges)
    assert all(unit_categories[split] == {0, 1, 2} for split in unit_categories)
    for seed, counts in enumerate(((2, 1), (10, 1), (10, 4), (7, 3)), 100):
        puzzle = generate_puzzle(seed, *counts)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
        validate_generated_puzzle(puzzle)
    for seed, counts in enumerate(((None, 1), (None, 4), (2, None), (10, None)), 200):
        puzzle = generate_puzzle(seed, *counts)
        assert (counts[0] is None or len(puzzle["train"]) == counts[0])
        assert (counts[1] is None or len(puzzle["test"]) == counts[1])
        validate_generated_puzzle(puzzle)
    assert _equal_puzzles(generate_puzzle(123456, 8, 3), generate_puzzle(123456, 8, 3))
    assert _equal_puzzles(generate_puzzle(654321), generate_puzzle(654321))
    assert not _equal_puzzles(generate_puzzle(300, 6, 2), generate_puzzle(301, 6, 2))
    illegal = ((1, None), (11, None), (None, 0), (None, 5),
               (2, 2), (3, 4), (4, 4), (2.5, 1), (True, 1))
    for counts in illegal:
        try:
            generate_puzzle(999, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {counts}")
