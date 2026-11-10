import numpy as np


_SOURCE_ROWS = (
    ("000000", "004000", "000000", "400040"),
    ("0050", "0000"),
    ("00000", "00000", "00200", "00000", "00000"),
    ("000", "060", "000"),
    ("200", "000", "000"),
)


def _signature(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _source_inputs():
    return [
        np.array([[int(x) for x in row] for row in rows], dtype=np.int8)
        for rows in _SOURCE_ROWS
    ]


def _background_and_seed(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 2
    major = int(np.argmax(counts))
    assert counts[major] > counts[1 - major]
    return int(colors[major]), int(colors[1 - major])


def _masks(grid, background):
    tiled = np.tile(grid, (3, 3))
    seeds = tiled != background
    vacant = tiled == background
    lines = vacant & np.any(seeds, axis=1)[:, None]
    diagonals = np.zeros(tiled.shape, dtype=bool)
    for row, column in np.argwhere(seeds):
        for delta in (-1, 1):
            rr, cc = int(row) + delta, int(column) + delta
            if (0 <= rr < tiled.shape[0] and 0 <= cc < tiled.shape[1]
                    and vacant[rr, cc] and not lines[rr, cc]):
                diagonals[rr, cc] = True
    untouched = vacant & ~lines & ~diagonals
    return seeds, lines, diagonals, untouched


def _construct_output(grid, background, horizontal, diagonal):
    output = np.tile(grid, (3, 3))
    copied_seeds = np.argwhere(output != background).copy()
    for row in np.unique(copied_seeds[:, 0]):
        row = int(row)
        output[row, output[row] == background] = horizontal
    for row, column in copied_seeds:
        for delta in (-1, 1):
            rr, cc = int(row) + delta, int(column) + delta
            if (0 <= rr < output.shape[0] and 0 <= cc < output.shape[1]
                    and output[rr, cc] == background):
                output[rr, cc] = diagonal
    return output


def _check_count(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = _check_count("num_train", num_train, 2, 10)
    test = _check_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = max(2, test + 1) + (seed // 4) % (11 - max(2, test + 1))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _sample_positions(rng, height, width, kind):
    cells = [(r, c) for r in range(height) for c in range(width)]
    cap = min(7, (height * width - 1) // 2)
    if kind == "shared_seam":
        row = int(rng.integers(1, height - 1))
        return ((row, 0), (row, width - 1))
    if kind == "adjacent":
        if int(rng.integers(0, 3)) == 0:
            row, col = 0, 0
        else:
            row = int(rng.integers(0, height - 2))
            col = int(rng.integers(0, width - 1))
        chosen = [(row, col), (row + 1, col + 1)]
        if cap >= 3 and int(rng.integers(0, 2)):
            extras = [(row + 1, c) for c in range(width) if (row + 1, c) not in chosen]
            chosen.append(extras[int(rng.integers(len(extras)))])
        return tuple(sorted(chosen))
    if kind == "no_diagonal":
        return tuple((r, int(rng.integers(width))) for r in range(height))
    if kind == "top_left":
        return ((0, 0),)
    if kind == "edge":
        boundary = [(r, c) for r, c in cells if r in (0, height - 1) or c in (0, width - 1)]
        return (boundary[int(rng.integers(len(boundary)))],)
    if kind == "interior":
        interior = [(r, c) for r, c in cells if 0 < r < height - 1 and 0 < c < width - 1]
        return (interior[int(rng.integers(len(interior)))],)
    if kind == "shared":
        row = int(rng.integers(height))
        count = int(rng.integers(2, min(5, width, cap) + 1))
        columns = rng.choice(width, size=count, replace=False)
        return tuple(sorted((row, int(c)) for c in columns))
    if kind == "different":
        count = int(rng.integers(2, min(5, height, cap) + 1))
        rows = rng.choice(height, size=count, replace=False)
        return tuple(sorted((int(r), int(rng.integers(width))) for r in rows))
    if kind == "near_seam":
        row = int(rng.integers(height))
        col = (0, width - 1)[int(rng.integers(2))]
        chosen = [(row, col)]
        if cap >= 2 and int(rng.integers(0, 4)) < 3:
            rest = [cell for cell in cells if cell not in chosen]
            chosen.append(rest[int(rng.integers(len(rest)))])
        return tuple(sorted(chosen))
    count = int(rng.integers(1, cap + 1))
    indices = rng.choice(len(cells), size=count, replace=False)
    return tuple(sorted(cells[int(i)] for i in np.atleast_1d(indices)))


def _shape_for(rng, kind):
    if kind in ("shared_seam", "adjacent"):
        return int(rng.integers(3, 9)), int(rng.integers(3, 9))
    if kind == "no_diagonal":
        return int(rng.integers(2, 5)), int(rng.integers(3, 8))
    if kind == "interior":
        return int(rng.integers(3, 11)), int(rng.integers(3, 11))
    if kind == "shared":
        return int(rng.integers(2, 11)), int(rng.integers(3, 11))
    if kind == "different":
        return int(rng.integers(3, 11)), int(rng.integers(2, 11))
    if kind == "near_seam":
        width = 2 if int(rng.integers(0, 2)) == 0 else int(rng.integers(3, 11))
        return int(rng.integers(3, 11)), width
    return int(rng.integers(2, 11)), int(rng.integers(2, 11))


def _kind_for(rng, split, index, seed):
    if split == "train" and index == 0:
        return "shared_seam"
    if split == "train" and index == 1:
        return "adjacent"
    kinds = ("top_left", "edge", "interior", "shared", "different",
             "near_seam", "no_diagonal", "random")
    if split == "test" and index == 0:
        return kinds[seed % len(kinds)]
    return kinds[int(rng.integers(len(kinds)))]


def _choose_background(rng, base, palette, split, index, vary_background):
    if not vary_background or (split == "train" and index == 0):
        return base
    if (split == "train" and index == 1) or int(rng.integers(0, 10)) < 4:
        alternatives = [c for c in palette if c != base]
        return alternatives[int(rng.integers(len(alternatives)))]
    return base


def _choose_seed_color(rng, palette, background, train_colors, split, seed):
    available = [c for c in palette if c != background]
    unseen = [c for c in available if c not in train_colors]
    familiar = [c for c in available if c in train_colors]
    if split == "test" and seed % 3 == 0 and unseen:
        pool = unseen
    elif familiar and int(rng.integers(0, 10)) < 5:
        pool = familiar
    else:
        pool = available
    return pool[int(rng.integers(len(pool)))]


def _make_pair(rng, kind, background, seed_color, horizontal, diagonal,
               used_inputs, used_layouts, train_shapes, split):
    for attempt in range(200):
        if split == "train" and len(train_shapes) >= 2 and attempt == 0 and int(rng.integers(0, 3)) == 0:
            height, width = train_shapes[int(rng.integers(len(train_shapes)))]
        elif split == "test" and train_shapes and attempt == 0 and int(rng.integers(0, 2)) == 0:
            height, width = train_shapes[int(rng.integers(len(train_shapes)))]
        else:
            height, width = _shape_for(rng, kind)
        if kind in ("shared_seam", "adjacent") and (height < 3 or width < 3):
            continue
        if kind == "adjacent" and height < 3:
            continue
        if kind == "interior" and (height < 3 or width < 3):
            continue
        if kind == "shared" and (width < 3 or height * width < 6):
            continue
        if kind == "different" and height < 3:
            continue
        if kind == "no_diagonal" and (width < 3 or height > 4):
            continue
        positions = _sample_positions(rng, height, width, kind)
        if (height, width, positions) in used_layouts:
            continue
        if len(positions) * 2 >= height * width:
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        for row, col in positions:
            grid[row, col] = seed_color
        sig = _signature(grid)
        if sig in used_inputs or any(np.array_equal(grid, s) for s in _source_inputs()):
            continue
        output = _construct_output(grid, background, horizontal, diagonal)
        return {"input": grid, "output": output}
    raise AssertionError("bounded pair construction exhausted")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    colors = [int(c) for c in rng.permutation(10)]
    horizontal, diagonal, base_background = colors[:3]
    palette = colors[2:]
    vary_background = bool(rng.integers(0, 3) == 0)
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    used_layouts = set()
    train_shapes = []
    train_colors = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            kind = _kind_for(rng, split, index, seed)
            background = _choose_background(rng, base_background, palette, split, index,
                                            vary_background)
            seed_color = _choose_seed_color(rng, palette, background, train_colors, split, seed)
            pair = _make_pair(rng, kind, background, seed_color, horizontal, diagonal,
                              used_inputs, used_layouts, train_shapes, split)
            puzzle[split].append(pair)
            used_inputs.add(_signature(pair["input"]))
            _, pair_seed = _background_and_seed(pair["input"])
            positions = tuple(map(tuple, np.argwhere(pair["input"] == pair_seed).tolist()))
            used_layouts.add((*pair["input"].shape, positions))
            if split == "train":
                train_shapes.append(pair["input"].shape)
                train_colors.add(seed_color)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    grid, output = pair["input"], pair["output"]
    background, seed_color = _background_and_seed(grid)
    positions = tuple(map(tuple, np.argwhere(grid == seed_color).tolist()))
    seeds, lines, diagonals, untouched = _masks(grid, background)
    assert output.shape == (3 * grid.shape[0], 3 * grid.shape[1])
    assert np.all(output[seeds] == seed_color)
    assert np.all(output[untouched] == background)
    return {
        "background": background, "seed_color": seed_color,
        "positions": positions, "shape": grid.shape,
        "seeds": seeds, "lines": lines, "diagonals": diagonals,
        "untouched": untouched,
    }


def _infer_roles(train):
    roles = []
    for pair in train:
        info = _analyze_pair(pair)
        for mask, label in ((info["lines"], "horizontal"), (info["diagonals"], "diagonal")):
            if mask.any():
                values = np.unique(pair["output"][mask])
                assert len(values) == 1
                roles.append((label, int(values[0])))
    horizontal = {value for label, value in roles if label == "horizontal"}
    diagonal = {value for label, value in roles if label == "diagonal"}
    assert len(horizontal) == len(diagonal) == 1
    horizontal, diagonal = horizontal.pop(), diagonal.pop()
    assert horizontal != diagonal
    return horizontal, diagonal


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, seen_layouts = set(), set(), set()
    analyses = {"train": [], "test": []}
    sources = _source_inputs()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            grid = pair["input"]
            assert 2 <= grid.shape[0] <= 10 and 2 <= grid.shape[1] <= 10
            assert not np.array_equal(grid, pair["output"])
            input_sig = _signature(grid)
            pair_sig = (input_sig, _signature(pair["output"]))
            assert input_sig not in seen_inputs and pair_sig not in seen_pairs
            seen_inputs.add(input_sig)
            seen_pairs.add(pair_sig)
            assert not any(np.array_equal(grid, source) for source in sources)
            info = _analyze_pair(pair)
            layout = (*info["shape"], info["positions"])
            assert layout not in seen_layouts
            seen_layouts.add(layout)
            assert 1 <= len(info["positions"]) <= 7
            assert len(info["positions"]) * 2 < grid.size
            analyses[split].append(info)
    horizontal, diagonal = _infer_roles(puzzle["train"])
    for split in ("train", "test"):
        for pair, info in zip(puzzle[split], analyses[split]):
            assert len({info["background"], info["seed_color"], horizontal, diagonal}) == 4
            expected = _construct_output(pair["input"], info["background"], horizontal, diagonal)
            assert np.array_equal(pair["output"], expected)
    first = analyses["train"][0]
    second = analyses["train"][1]
    h, w = first["shape"]
    assert h >= 3 and w >= 3
    assert any((r, 0) in first["positions"] and (r, w - 1) in first["positions"]
               for r in range(1, h - 1))
    assert first["lines"].any() and first["diagonals"].any()
    assert any((r + 1, c + 1) in second["positions"]
               for r, c in second["positions"])
    assert second["diagonals"].any()


def _predict_from_demonstrations(train, test_input):
    horizontal, diagonal = _infer_roles(train)
    background, seed_color = _background_and_seed(test_input)
    assert len({background, seed_color, horizontal, diagonal}) == 4
    seeds, lines, accents, _ = _masks(test_input, background)
    predicted = np.full((3 * test_input.shape[0], 3 * test_input.shape[1]),
                        background, dtype=np.int8)
    predicted[lines] = horizontal
    predicted[accents] = diagonal
    predicted[seeds] = np.tile(test_input, (3, 3))[seeds]
    return predicted


def _fingerprint(puzzle):
    return tuple((split, tuple((_signature(p["input"]), _signature(p["output"]))
                                for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    shapes, layouts, seed_counts, role_pairs = set(), set(), set(), set()
    categories = set()
    novel_shapes = novel_colors = changed_backgrounds = absent_diagonals = 0
    same_train_shape = top_left_train = crowded_test = 0
    two_columns = large = 0
    changed_train_backgrounds = 0
    other_corner_test = interior_test = noncorner_edge_test = 0
    shared_test = multirow_test = adjacent_rows_test = 0
    seam_test = suppression_test = priority_test = two_col_crowded_test = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        assert np.array_equal(puzzle["test"][0]["output"],
                              _predict_from_demonstrations(puzzle["train"], puzzle["test"][0]["input"]))
        horizontal, diagonal = _infer_roles(puzzle["train"])
        role_pairs.add((horizontal, diagonal))
        train_infos = [_analyze_pair(p) for p in puzzle["train"]]
        train_shapes = {i["shape"] for i in train_infos}
        train_colors = {i["seed_color"] for i in train_infos}
        backgrounds = {i["background"] for i in train_infos}
        changed_train_backgrounds += len(backgrounds) > 1
        same_train_shape += len(train_shapes) < len(train_infos)
        top_left_train += any((0, 0) in i["positions"] for i in train_infos)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                assert np.array_equal(pair["output"],
                                      _predict_from_demonstrations(puzzle["train"], pair["input"]))
                h, w = info["shape"]
                shapes.add((h, w))
                layouts.add((h, w, info["positions"]))
                seed_counts.add(len(info["positions"]))
                two_columns += w == 2
                large += h >= 8 or w >= 8
                assert not any(np.array_equal(pair["input"], source) for source in _source_inputs())
                if split == "test":
                    novel_shapes += info["shape"] not in train_shapes
                    novel_colors += info["seed_color"] not in train_colors
                    changed_backgrounds += info["background"] not in backgrounds
                    absent_diagonals += not info["diagonals"].any()
                    crowded_test += len(info["positions"]) >= 2
                    rows = [r for r, _ in info["positions"]]
                    shared_test += len(rows) != len(set(rows))
                    multirow_test += len(set(rows)) > 1
                    adjacent_rows_test += any(abs(a[0] - b[0]) == 1
                                              for j, a in enumerate(info["positions"])
                                              for b in info["positions"][j + 1:])
                    if len(info["positions"]) == 1:
                        row, column = info["positions"][0]
                        corners = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
                        other_corner_test += (row, column) in corners[1:]
                        interior_test += 0 < row < h - 1 and 0 < column < w - 1
                        noncorner_edge_test += (row in (0, h - 1) or column in (0, w - 1)) and (row, column) not in corners
                    seam_test += any(int(column) % w in (0, w - 1)
                                     for _, column in np.argwhere(info["diagonals"]))
                    for row, column in np.argwhere(info["seeds"]):
                        for delta in (-1, 1):
                            rr, cc = int(row) + delta, int(column) + delta
                            if 0 <= rr < 3 * h and 0 <= cc < 3 * w:
                                suppression_test += bool(info["lines"][rr, cc])
                                priority_test += bool(info["seeds"][rr, cc])
                    two_col_crowded_test += w == 2 and len(info["positions"]) >= 2
                    categories.add(("top_left" if info["positions"] == ((0, 0),) else
                                    "shared" if len(rows) != len(set(rows)) else
                                    "multirow" if len(rows) > 1 else
                                    "single"))
    assert len(shapes) >= 35 and len(layouts) >= 150
    assert max(seed_counts) >= 5 and {1, 2, 3}.issubset(seed_counts)
    assert len(role_pairs) >= 25
    assert two_columns >= 8 and large >= 40
    assert novel_shapes >= 12 and novel_colors >= 8
    assert absent_diagonals >= 3 and crowded_test >= 15
    assert same_train_shape >= 4 and top_left_train >= 5
    assert len(categories) == 4
    assert changed_train_backgrounds >= 4 and changed_backgrounds >= 2
    assert other_corner_test >= 1 and interior_test >= 4 and noncorner_edge_test >= 4
    assert shared_test >= 5 and multirow_test >= 10 and adjacent_rows_test >= 5
    assert seam_test >= 10 and suppression_test >= 5 and priority_test >= 5
    assert two_col_crowded_test >= 1
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (2001, None, 1), (2002, None, 4),
                              (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    assert _fingerprint(generate_puzzle(314159, 7, 3)) == _fingerprint(generate_puzzle(314159, 7, 3))
    for train, test in ((1, 1), (11, 1), (2, 0), (2, 5), (2, 2), (4, 4)):
        try:
            generate_puzzle(99, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
