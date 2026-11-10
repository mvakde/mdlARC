"""Complete rectangular tile arrays with interior channels and exterior continuations."""

import numpy as np


# Exact source pairs are excluded; their dimensions and structural categories are valid.
_SOURCE_PAIRS = {
    (
        "0055005500/0055005500/0000000000/0055005500/0055005500/"
        "0000000000/0055005500/0055005500/0000000000/0000000000",
        "0055225500/0055225500/1122222211/0055225500/0055225500/"
        "1122222211/0055225500/0055225500/0000110000/0000110000",
    ),
    (
        "5505505500/5505505500/0000000000/5505505500/5505505500/"
        "0000000000/5505505500/5505505500/0000000000/0000000000",
        "5525525500/5525525500/2222222211/5525525500/5525525500/"
        "2222222211/5525525500/5525525500/0010010000/0010010000",
    ),
    (
        "0550550550/0550550550/0000000000/0000000000/0550550550/"
        "0550550550/0000000000/0000000000/0550550550/0550550550",
        "0552552550/0552552550/1222222221/1222222221/0552552550/"
        "0552552550/1222222221/1222222221/0552552550/0552552550",
    ),
    (
        "0000000000/0550055055/0550055055/0000000000/0550055055/"
        "0550055055/0000000000/0550055055/0550055055/0000000000",
        "0001100100/0552255255/0552255255/1222222222/0552255255/"
        "0552255255/1222222222/0552255255/0552255255/0001100100",
    ),
}


def _validate_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _optional_integer(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    train = _optional_integer(num_train, "num_train")
    test = _optional_integer(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train is not None and test is not None and train <= test:
        raise ValueError("num_train must exceed num_test")
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + (seed + 3 * test) % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    return train, test


def _bands(mask):
    positions = np.flatnonzero(mask)
    if not len(positions):
        return ()
    result = []
    start = previous = int(positions[0])
    for raw in positions[1:]:
        position = int(raw)
        if position != previous + 1:
            result.append((start, previous + 1))
            start = position
        previous = position
    result.append((start, previous + 1))
    return tuple(result)


def _widths(bands):
    return tuple(end - start for start, end in bands)


def _analyze_input(grid):
    """Recover the background and the full Cartesian array from visible geometry."""
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    colors = np.unique(grid)
    assert len(colors) == 2
    candidates = []
    for raw_tile in colors:
        tile = int(raw_tile)
        background = int(colors[0] if colors[1] == raw_tile else colors[1])
        rows = _bands(np.any(grid == tile, axis=1))
        columns = _bands(np.any(grid == tile, axis=0))
        if len(rows) < 2 or len(columns) < 2:
            continue
        if len(set(_widths(rows))) != 1 or len(set(_widths(columns))) != 1:
            continue
        expected = np.full(grid.shape, background, dtype=grid.dtype)
        for r0, r1 in rows:
            for c0, c1 in columns:
                expected[r0:r1, c0:c1] = tile
        if not np.array_equal(grid, expected):
            continue
        row_gaps = tuple((rows[i][1], rows[i + 1][0]) for i in range(len(rows) - 1))
        column_gaps = tuple(
            (columns[i][1], columns[i + 1][0]) for i in range(len(columns) - 1)
        )
        if any(a >= b for a, b in row_gaps + column_gaps):
            continue
        candidates.append({
            "background": background, "tile": tile,
            "rows": rows, "columns": columns,
            "row_gaps": row_gaps, "column_gaps": column_gaps,
            "tile_height": rows[0][1] - rows[0][0],
            "tile_width": columns[0][1] - columns[0][0],
            "r0": rows[0][0], "r1": rows[-1][1],
            "c0": columns[0][0], "c1": columns[-1][1],
            "top": rows[0][0], "bottom": grid.shape[0] - rows[-1][1],
            "left": columns[0][0], "right": grid.shape[1] - columns[-1][1],
        })
    assert len(candidates) == 1
    return candidates[0]


def _masks(grid, analysis):
    height, width = grid.shape
    r0, r1, c0, c1 = (analysis[k] for k in ("r0", "r1", "c0", "c1"))
    interior = np.zeros(grid.shape, dtype=bool)
    exterior = np.zeros(grid.shape, dtype=bool)
    interior[r0:r1, c0:c1] = grid[r0:r1, c0:c1] == analysis["background"]
    for start, end in analysis["row_gaps"]:
        exterior[start:end, :c0] = True
        exterior[start:end, c1:width] = True
    for start, end in analysis["column_gaps"]:
        exterior[:r0, start:end] = True
        exterior[r1:height, start:end] = True
    assert np.any(interior) and not np.any(interior & exterior)
    return interior, exterior


def _output(grid, analysis, connector, extension):
    interior, exterior = _masks(grid, analysis)
    result = grid.copy()
    result[interior] = connector
    result[exterior] = extension
    return result


def _margin(rng):
    return 0 if rng.random() < 0.34 else int(rng.integers(1, 9))


def _gap_sequence(rng, count, maximum):
    if rng.random() < 0.42:
        return (int(rng.integers(1, maximum + 1)),) * (count - 1)
    return tuple(int(x) for x in rng.integers(1, maximum + 1, count - 1))


def _orient(rng, layout):
    # A transpose and independent reflections retain exactly the same rule.
    if rng.random() < 0.5:
        for first, second in (
            ("row_count", "column_count"), ("tile_height", "tile_width"),
            ("row_gaps", "column_gaps"), ("top", "left"), ("bottom", "right"),
        ):
            layout[first], layout[second] = layout[second], layout[first]
    if rng.random() < 0.5:
        layout["top"], layout["bottom"] = layout["bottom"], layout["top"]
        layout["row_gaps"] = layout["row_gaps"][::-1]
    if rng.random() < 0.5:
        layout["left"], layout["right"] = layout["right"], layout["left"]
        layout["column_gaps"] = layout["column_gaps"][::-1]
    return layout


def _sample_layout(rng, regime, mode, index, coverage_index):
    for _ in range(256):
        compact = mode != 3 and rng.random() < 0.13
        if compact:
            low_count, high_count, high_tile, high_gap = 2, 3, 2, 2
        elif mode == 3 and regime == "source_train":
            low_count, high_count, high_tile, high_gap = 2, 4, 3, 4
        elif mode == 3 and regime == "target_test":
            low_count, high_count, high_tile, high_gap = 3, 6, 5, 6
        else:
            low_count, high_count, high_tile, high_gap = 2, 6, 5, 6
        row_count = int(rng.integers(low_count, high_count + 1))
        column_count = int(rng.integers(low_count, high_count + 1))
        layout = {
            "row_count": row_count,
            "column_count": column_count,
            "tile_height": int(rng.integers(1, high_tile + 1)),
            "tile_width": int(rng.integers(1, high_tile + 1)),
            "row_gaps": _gap_sequence(rng, row_count, high_gap),
            "column_gaps": _gap_sequence(rng, column_count, high_gap),
            "top": int(rng.integers(0, 2)) if compact else _margin(rng),
            "bottom": int(rng.integers(0, 2)) if compact else _margin(rng),
            "left": int(rng.integers(0, 2)) if compact else _margin(rng),
            "right": int(rng.integers(0, 2)) if compact else _margin(rng),
        }
        layout = _orient(rng, layout)
        if (regime == "source_train" and index != coverage_index
                and rng.random() < (0.30 if index == 0 else 0.09)):
            for side in ("top", "bottom", "left", "right"):
                layout[side] = 0
        if mode == 1:
            if regime == "source_train":
                layout["top"] = 0
                if index == coverage_index:
                    layout["bottom"] = int(rng.integers(1, 7))
                    layout["left"] = int(rng.integers(1, 7))
                    layout["right"] = int(rng.integers(1, 7))
            else:
                layout["top"] = int(rng.integers(1, 6))
                layout["bottom"] = int(rng.integers(1, 6))
                layout["left"] = int(rng.integers(1, 6))
                layout["right"] = 0
        elif mode == 2:
            if regime == "source_train":
                side = ("top", "bottom", "left", "right")[int(rng.integers(0, 4))]
                layout[side] = 0
            else:
                for side in ("top", "bottom", "left", "right"):
                    layout[side] = int(rng.integers(1, 7))
        if regime == "source_train" and index == coverage_index:
            for first, second in (("top", "bottom"), ("left", "right")):
                if layout[first] == layout[second] == 0:
                    layout[("bottom" if first == "top" else "right")] = int(
                        rng.integers(1, 7)
                    )
        height = (layout["top"] + layout["bottom"]
                  + layout["row_count"] * layout["tile_height"] + sum(layout["row_gaps"]))
        width = (layout["left"] + layout["right"]
                 + layout["column_count"] * layout["tile_width"] + sum(layout["column_gaps"]))
        if 1 <= height <= 30 and 1 <= width <= 30:
            if mode != 1 and rng.random() < 0.045:
                vertical_extra = 30 - height
                horizontal_extra = 30 - width
                top_extra = int(rng.integers(0, vertical_extra + 1))
                left_extra = int(rng.integers(0, horizontal_extra + 1))
                layout["top"] += top_extra
                layout["bottom"] += vertical_extra - top_extra
                layout["left"] += left_extra
                layout["right"] += horizontal_extra - left_extra
            return layout
    raise AssertionError("layout sampling exhausted")


def _make_input(layout, background, tile):
    height = (layout["top"] + layout["bottom"]
              + layout["row_count"] * layout["tile_height"]
              + sum(layout["row_gaps"]))
    width = (layout["left"] + layout["right"]
             + layout["column_count"] * layout["tile_width"]
             + sum(layout["column_gaps"]))
    result = np.full((height, width), background, dtype=np.int8)
    row = layout["top"]
    for r in range(layout["row_count"]):
        column = layout["left"]
        for c in range(layout["column_count"]):
            result[row:row + layout["tile_height"],
                   column:column + layout["tile_width"]] = tile
            column += layout["tile_width"]
            if c + 1 < layout["column_count"]:
                column += layout["column_gaps"][c]
        row += layout["tile_height"]
        if r + 1 < layout["row_count"]:
            row += layout["row_gaps"][r]
    return result


def _signature(grid):
    return (grid.shape, tuple(int(x) for x in grid.ravel()))


def _source_pair(input_grid, output_grid):
    if input_grid.shape != (10, 10) or output_grid.shape != (10, 10):
        return False
    def rows(grid):
        return "/".join("".join(str(int(x)) for x in row) for row in grid)
    return (rows(input_grid), rows(output_grid)) in _SOURCE_PAIRS


def _layout_signature(analysis):
    return (
        len(analysis["rows"]), len(analysis["columns"]),
        analysis["tile_height"], analysis["tile_width"],
        _widths(analysis["row_gaps"]), _widths(analysis["column_gaps"]),
        analysis["top"], analysis["bottom"], analysis["left"], analysis["right"],
    )


def _learn_colors(train):
    connectors = set()
    extensions = set()
    for pair in train:
        grid = pair["input"]
        analysis = _analyze_input(grid)
        interior, exterior = _masks(grid, analysis)
        connectors.update(int(x) for x in np.unique(pair["output"][interior]))
        if np.any(exterior):
            extensions.update(int(x) for x in np.unique(pair["output"][exterior]))
    assert len(connectors) == len(extensions) == 1
    connector = next(iter(connectors))
    extension = next(iter(extensions))
    assert connector != extension
    return connector, extension


def _solve_from_train(train, test_input):
    # Uses returned demonstrations and the visible test input only.
    connector, extension = _learn_colors(train)
    analysis = _analyze_input(test_input)
    assert connector not in (analysis["background"], analysis["tile"])
    assert extension not in (analysis["background"], analysis["tile"])
    return _output(test_input, analysis, connector, extension)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _validate_seed(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    connector, extension = (int(x) for x in rng.choice(10, size=2, replace=False))
    available = np.array([x for x in range(10) if x not in (connector, extension)])
    mode = int(rng.integers(0, 4))
    coverage_index = int(rng.integers(0, train_count))
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    train_layouts = set()
    first_counts = None
    for regime, count in (("source_train", train_count), ("target_test", test_count)):
        destination = "train" if regime == "source_train" else "test"
        for index in range(count):
            # Separate streams make each pair reproducible without repeating a stream.
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            for _ in range(512):
                layout = _sample_layout(pair_rng, regime, mode, index, coverage_index)
                counts = (layout["row_count"], layout["column_count"])
                if regime == "source_train" and index == 1 and counts == first_counts:
                    continue
                background, tile = (int(x) for x in pair_rng.choice(available, 2, replace=False))
                input_grid = _make_input(layout, background, tile)
                signature = _signature(input_grid)
                if signature in seen_inputs:
                    continue
                analysis = _analyze_input(input_grid)
                layout_signature = _layout_signature(analysis)
                if regime == "target_test" and layout_signature in train_layouts:
                    continue
                output_grid = _output(input_grid, analysis, connector, extension)
                if _source_pair(input_grid, output_grid):
                    continue
                puzzle[destination].append({"input": input_grid, "output": output_grid})
                seen_inputs.add(signature)
                if regime == "source_train":
                    train_layouts.add(layout_signature)
                    if index == 0:
                        first_counts = counts
                break
            else:
                raise AssertionError("could not construct distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(np.min(grid)) >= 0 and int(np.max(grid)) <= 9
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert not _source_pair(inp, out)
            input_key = _signature(inp)
            pair_key = (input_key, _signature(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analyses[section].append(_analyze_input(inp))

    connector, extension = _learn_colors(puzzle["train"])
    train_layouts = {_layout_signature(a) for a in analyses["train"]}
    assert (len(analyses["train"][0]["rows"]), len(analyses["train"][0]["columns"])) != (
        len(analyses["train"][1]["rows"]), len(analyses["train"][1]["columns"])
    )
    assert any(
        a["left"] + a["right"] > 0 and a["top"] + a["bottom"] > 0
        for a in analyses["train"]
    )
    for section in ("train", "test"):
        for pair, analysis in zip(puzzle[section], analyses[section]):
            inp, out = pair["input"], pair["output"]
            assert connector not in (analysis["background"], analysis["tile"])
            assert extension not in (analysis["background"], analysis["tile"])
            interior, exterior = _masks(inp, analysis)
            assert np.all(out[interior] == connector)
            if np.any(exterior):
                assert np.all(out[exterior] == extension)
            assert np.array_equal(out, _output(inp, analysis, connector, extension))
            assert set(np.unique(inp).tolist()) == {analysis["background"], analysis["tile"]}
            if section == "test":
                assert _layout_signature(analysis) not in train_layouts
                assert np.array_equal(out, _solve_from_train(puzzle["train"], inp.copy()))


def _puzzles_equal(first, second):
    if set(first) != {"train", "test"} or set(second) != {"train", "test"}:
        return False
    for section in ("train", "test"):
        if len(first[section]) != len(second[section]):
            return False
        for a, b in zip(first[section], second[section]):
            if not all(np.array_equal(a[key], b[key]) for key in ("input", "output")):
                return False
    return True


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert _puzzles_equal(puzzles[17], generate_puzzle(17))
    assert _puzzles_equal(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3))
    for seed, train, test in ((2000, 2, 1), (2001, 10, 1), (2002, 10, 4)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for seed, train, test in ((3000, None, 1), (3001, None, 4),
                              (3002, 2, None), (3003, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(4000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")

    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_signature(p["train"][0]["input"]) for p in puzzles}) == 42
    shapes, layouts, counts, tiles = set(), set(), set(), set()
    color_changes = 0
    all_edge = large_margin = large_grid = small_grid = False
    first_all_edge = 0
    equal_gaps = {"train": set(), "test": set()}
    variable_gaps = {"train": set(), "test": set()}
    extension_sides = set()
    for puzzle in puzzles:
        first = _analyze_input(puzzle["train"][0]["input"])
        first_all_edge += all(
            first[side] == 0 for side in ("top", "bottom", "left", "right")
        )
        pair_colors = set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp = pair["input"]
                a = _analyze_input(inp)
                pair_colors.add((a["background"], a["tile"]))
                shapes.add(inp.shape)
                layouts.add(_layout_signature(a))
                counts.add((len(a["rows"]), len(a["columns"])))
                tiles.add((a["tile_height"], a["tile_width"]))
                margins = (a["top"], a["bottom"], a["left"], a["right"])
                all_edge |= all(x == 0 for x in margins)
                large_margin |= any(x > 3 for x in margins)
                large_grid |= max(inp.shape) >= 25
                small_grid |= max(inp.shape) <= 7
                extension_sides.update(
                    side for side, value in zip(("top", "bottom", "left", "right"), margins)
                    if value > 0
                )
                for axis, gaps in (("row", a["row_gaps"]), ("column", a["column_gaps"])):
                    widths = _widths(gaps)
                    if len(widths) > 1:
                        (equal_gaps if len(set(widths)) == 1 else variable_gaps)[section].add(axis)
        color_changes += len(pair_colors) > 1
    assert color_changes >= 35
    assert len(shapes) >= 60 and len(layouts) >= 100
    assert len(counts) >= 15 and (2, 2) in counts
    assert len(tiles) >= 18 and any(max(x) > 3 for x in tiles)
    assert all_edge and large_margin and large_grid and small_grid
    assert first_all_edge >= 2
    assert extension_sides == {"top", "bottom", "left", "right"}
    assert all(equal_gaps[s] == {"row", "column"} for s in ("train", "test"))
    assert all(variable_gaps[s] == {"row", "column"} for s in ("train", "test"))
