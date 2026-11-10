"""Synthetic puzzles for diagonal duplication and horizontal pair connectors."""

import numpy as np


_SOURCE_DESCRIPTORS = {
    (3, 0, ((0, 1, 8),), 3),
    (6, 0, ((0, 1, 4), (3, 4, 4)), 3),
    (7, 0, ((0, 2, 7),), 3),
    (4, 0, ((0, 0, 9),), 3),
}
_MIN_SIDE = {"single": 3, "ordinary": 5, "downleft": 5,
             "same_mid": 3, "stacked": 6, "zigzag": 7, "mixed": 7}


def _normalize_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(seed)


def _normalize_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, rng, num_train, num_test):
    train = _normalize_count(num_train, "num_train")
    test = _normalize_count(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train is None and test is None:
        test = 1 + seed % 4
        train = max(test + 1, 2 + (seed // 4) % 9)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    assert int(counts[index]) > grid.size // 2
    return int(colors[index])


def _extract_pairs(grid, background):
    """Read the unique consecutive vertical endpoint pairs in each column."""
    assert grid.ndim == 2 and grid.shape[0] == grid.shape[1]
    assert 3 <= grid.shape[0] <= 15
    foreground = np.argwhere(grid != background)
    assert len(foreground) >= 2 and len(foreground) % 2 == 0
    pairs = []
    for column in range(grid.shape[1]):
        rows = np.flatnonzero(grid[:, column] != background)
        assert len(rows) % 2 == 0
        for index in range(0, len(rows), 2):
            top, bottom = int(rows[index]), int(rows[index + 1])
            color = int(grid[top, column])
            assert bottom == top + 2
            assert int(grid[bottom, column]) == color
            assert color != background
            assert np.all(grid[top + 1, :] == background)
            pairs.append((top, column, color))
    assert len(pairs) * 2 == len(foreground)
    return sorted(pairs)


def _construct_output(grid, connector):
    background = _background(grid)
    pairs = _extract_pairs(grid, background)
    assert connector not in set(int(x) for x in np.unique(grid))
    side = grid.shape[0]
    output = np.full((2 * side, 2 * side), background, dtype=np.int8)
    output[:side, :side] = grid
    output[side:, side:] = grid
    for top in {top for top, _, _ in pairs}:
        output[top + 1, :] = connector
        output[side + top + 1, :] = connector
    return output


def _choose_layout(rng, side, mode):
    tops = range(side - 2)
    if mode == "single":
        return [(int(rng.choice(list(tops))), int(rng.integers(side)))]
    if mode == "same_mid":
        top = int(rng.choice(list(tops)))
        columns = rng.choice(side, size=2, replace=False)
        return [(top, int(column)) for column in columns]
    if mode in ("ordinary", "downleft"):
        choices = [(a, b) for a in tops for b in tops if b - a >= 2]
        a, b = choices[int(rng.integers(len(choices)))]
        columns = sorted(int(x) for x in rng.choice(side, size=2, replace=False))
        if mode == "downleft":
            columns.reverse()
        return [(a, columns[0]), (b, columns[1])]
    if mode == "stacked":
        choices = [(a, b) for a in tops for b in tops if b - a == 3 or b - a >= 5]
        a, b = choices[int(rng.integers(len(choices)))]
        column = int(rng.integers(side))
        return [(a, column), (b, column)]
    if mode == "zigzag":
        choices = [(a, b, c) for a in tops for b in tops for c in tops
                   if b - a >= 2 and c - b >= 2]
        a, b, c = choices[int(rng.integers(len(choices)))]
        columns = sorted(int(x) for x in rng.choice(side, size=3, replace=False))
        return [(a, columns[0]), (b, columns[2]), (c, columns[1])]
    if mode == "mixed":
        count = int(rng.integers(3, 6))
        candidates = [(top, column) for top in tops for column in range(side)]
        rng.shuffle(candidates)
        selected = []
        for top, column in candidates:
            if any(abs(top - other_top) == 1 or
                   (column == other_column and abs(top - other_top) in (2, 4))
                   for other_top, other_column in selected):
                continue
            selected.append((top, column))
            if len(selected) == count:
                break
        assert len(selected) == count
        return selected
    raise ValueError(f"unknown layout mode: {mode}")


def _colors(rng, connector, train_count):
    available = [color for color in range(10) if color != connector]
    rng.shuffle(available)
    background_count = int(rng.integers(1, min(3, train_count) + 1))
    backgrounds = available[:background_count]
    endpoints = available[background_count:]
    return backgrounds, endpoints


def _make_pair(rng, side, mode, background, endpoint_palette, connector,
               multicolor, shared_color=None):
    positions = _choose_layout(rng, side, mode)
    if multicolor and len(positions) > 1:
        colors = [int(x) for x in rng.choice(endpoint_palette, size=len(positions), replace=True)]
        if len(set(colors)) == 1:
            colors[-1] = next(color for color in endpoint_palette if color != colors[0])
    else:
        color = int(shared_color if shared_color is not None else rng.choice(endpoint_palette))
        colors = [color] * len(positions)
    grid = np.full((side, side), background, dtype=np.int8)
    for (top, column), color in zip(positions, colors):
        grid[top, column] = color
        grid[top + 2, column] = color
    _extract_pairs(grid, background)
    return {"input": grid, "output": _construct_output(grid, connector)}


def _descriptor(pair):
    grid = pair["input"]
    background = _background(grid)
    connector = next(iter(set(int(x) for x in np.unique(pair["output"])) -
                          set(int(x) for x in np.unique(grid))))
    return (grid.shape[0], background, tuple(_extract_pairs(grid, background)), connector)


def _pair_info(pair):
    grid, output = pair["input"], pair["output"]
    side = grid.shape[0]
    assert grid.shape == (side, side) and 3 <= side <= 15
    assert output.shape == (2 * side, 2 * side)
    background = _background(grid)
    pairs = _extract_pairs(grid, background)
    input_colors = set(int(x) for x in np.unique(grid))
    output_colors = set(int(x) for x in np.unique(output))
    new_colors = output_colors - input_colors
    assert len(new_colors) == 1
    connector = next(iter(new_colors))
    assert np.array_equal(output, _construct_output(grid, connector))
    assert _descriptor(pair) not in _SOURCE_DESCRIPTORS
    return {"side": side, "background": background, "pairs": pairs,
            "connector": connector,
            "multi_color": len({color for _, _, color in pairs}) > 1}


def _mode_candidates(rng, index):
    if index == 0:
        return "single"
    if index == 1:
        return str(rng.choice(["ordinary", "ordinary", "downleft", "same_mid", "stacked"]))
    return str(rng.choice(["single", "ordinary", "downleft", "same_mid",
                           "stacked", "zigzag", "mixed"]))


def _unique_pair(rng, used, used_layouts, side, mode, background, endpoints, connector,
                 multicolor, forced_column=None, shared_color=None,
                 permitted_edges=None):
    for _ in range(100):
        pair = _make_pair(rng, side, mode, background, endpoints, connector,
                          multicolor, shared_color)
        grid = pair["input"]
        if forced_column is not None and mode == "single":
            old_col = _extract_pairs(grid, background)[0][1]
            if old_col != forced_column:
                color = int(grid[_extract_pairs(grid, background)[0][0], old_col])
                top = _extract_pairs(grid, background)[0][0]
                grid[top, old_col] = background
                grid[top + 2, old_col] = background
                grid[top, forced_column] = color
                grid[top + 2, forced_column] = color
                pair["output"] = _construct_output(grid, connector)
        positions = _extract_pairs(grid, background)
        if permitted_edges is not None:
            edges = {"left" if col == 0 else "right" for _, col, _ in positions
                     if col in (0, side - 1)}
            if not edges.issubset(permitted_edges):
                continue
        key = _grid_key(grid)
        layout = (side, tuple((top, column) for top, column, _ in positions))
        if key in used or layout in used_layouts or _descriptor(pair) in _SOURCE_DESCRIPTORS:
            continue
        used.add(key)
        used_layouts.add(layout)
        return pair
    raise AssertionError("unable to find a distinct legal layout")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _normalize_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    connector = int(rng.integers(10))
    backgrounds, endpoints = _colors(rng, connector, train_count)
    pair_seeds = rng.integers(0, 2**63 - 1, size=train_count + test_count, dtype=np.int64)
    used = set()
    used_layouts = set()
    train = []
    modes = []
    sizes = []
    training_edges = set()
    multicolor_taught = False
    # A compact single pair anchors the rule. Later examples demonstrate denser layouts.
    for index in range(train_count):
        local = np.random.default_rng(int(pair_seeds[index]))
        mode = _mode_candidates(rng, index)
        minimum = _MIN_SIDE[mode]
        if index == 0:
            side = int(rng.integers(3, 8))
            forced_column = int(rng.choice([0, side - 1, int(rng.integers(1, side - 1))]))
        elif index == 1:
            possible_sides = [value for value in range(max(6, minimum), 16)
                              if value != sizes[0]]
            side = int(rng.choice(possible_sides))
            forced_column = None
        else:
            side = int(rng.integers(minimum, 16))
            forced_column = None
        background = int(backgrounds[index % len(backgrounds)])
        multicolor = index >= 2 and mode != "single" and (index == 2 or rng.random() < 0.5)
        pair = _unique_pair(local, used, used_layouts, side, mode, background, endpoints,
                            connector, multicolor, forced_column)
        info = _pair_info(pair)
        if info["multi_color"]:
            multicolor_taught = True
        training_edges.update("left" if col == 0 else "right" for _, col, _ in info["pairs"]
                              if col in (0, side - 1))
        train.append(pair)
        modes.append(mode)
        sizes.append(side)
    test = []
    for index in range(test_count):
        local = np.random.default_rng(int(pair_seeds[train_count + index]))
        mode = str(rng.choice(modes))
        minimum = _MIN_SIDE[mode]
        if rng.random() < 0.5:
            eligible = [side for side in sizes if side >= minimum]
            side = int(rng.choice(eligible))
        else:
            side = int(rng.integers(minimum, 16))
        if side == 3 and not training_edges:
            side = 4
        background = int(rng.choice(backgrounds))
        multicolor = multicolor_taught and mode != "single" and rng.random() < 0.55
        # Literal endpoint colors sometimes recur and sometimes transfer to a new color.
        train_colors = {int(x) for pair in train for x in np.unique(pair["input"])}
        unseen = [color for color in endpoints if color not in train_colors]
        if not multicolor and unseen and rng.random() < 0.5:
            shared_color = int(rng.choice(unseen))
        else:
            shared_color = None
        pair = _unique_pair(local, used, used_layouts, side, mode, background, endpoints,
                            connector, multicolor, shared_color=shared_color,
                            permitted_edges=training_edges)
        test.append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _predict_from_training(train, test_input):
    connectors = set()
    for pair in train:
        colors_in = set(int(x) for x in np.unique(pair["input"]))
        colors_out = set(int(x) for x in np.unique(pair["output"]))
        assert len(colors_out - colors_in) == 1
        connectors.update(colors_out - colors_in)
    assert len(connectors) == 1
    return _construct_output(test_input, next(iter(connectors)))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used = set()
    layouts = set()
    infos = {"train": [], "test": []}
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            key = _grid_key(pair["input"])
            assert key not in used
            used.add(key)
            info = _pair_info(pair)
            layout = (info["side"], tuple((top, col) for top, col, _ in info["pairs"]))
            assert layout not in layouts
            layouts.add(layout)
            infos[regime].append(info)
    all_infos = infos["train"] + infos["test"]
    connectors = {info["connector"] for info in all_infos}
    assert len(connectors) == 1
    connector = next(iter(connectors))
    backgrounds = {info["background"] for info in all_infos}
    endpoints = {color for info in all_infos for _, _, color in info["pairs"]}
    assert connector not in backgrounds | endpoints
    assert backgrounds.isdisjoint(endpoints)
    train_backgrounds = {info["background"] for info in infos["train"]}
    assert all(info["background"] in train_backgrounds for info in infos["test"])
    trained_edges = {"left" if col == 0 else "right" for info in infos["train"]
                     for _, col, _ in info["pairs"] if col in (0, info["side"] - 1)}
    for info, pair in zip(infos["test"], puzzle["test"]):
        test_edges = {"left" if col == 0 else "right" for _, col, _ in info["pairs"]
                      if col in (0, info["side"] - 1)}
        assert test_edges.issubset(trained_edges)
        if info["multi_color"]:
            assert any(x["multi_color"] for x in infos["train"])
        assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]),
                              pair["output"])
    assert len({info["side"] for info in infos["train"]}) >= 2
    assert any(len(info["pairs"]) == 1 for info in infos["train"])
    assert any(len(info["pairs"]) >= 2 for info in infos["train"])


def _puzzle_key(puzzle):
    return tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                 for regime in ("train", "test") for pair in puzzle[regime])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                info = _pair_info(pair)
                side = info["side"]
                output = pair["output"]
                assert np.array_equal(output[:side, :side][[r for r in range(side)
                    if r not in {top + 1 for top, _, _ in info["pairs"]}], :],
                    pair["input"][[r for r in range(side)
                    if r not in {top + 1 for top, _, _ in info["pairs"]}], :])
                quiet_rows = [r for r in range(side)
                              if r not in {top + 1 for top, _, _ in info["pairs"]}]
                assert np.array_equal(output[side:, side:][quiet_rows, :],
                                      pair["input"][quiet_rows, :])
                assert np.all(output[quiet_rows, side:] == info["background"])
                assert np.all(output[[side + r for r in quiet_rows], :side]
                              == info["background"])
                assert all(np.all(output[top + 1, :] == info["connector"])
                           and np.all(output[side + top + 1, :] == info["connector"])
                           for top, _, _ in info["pairs"])
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    all_infos = [_pair_info(pair) for p in puzzles for regime in ("train", "test")
                 for pair in p[regime]]
    assert {info["side"] for info in all_infos} == set(range(3, 16))
    assert len({info["connector"] for info in all_infos}) >= 7
    assert len({info["background"] for info in all_infos}) >= 7
    assert any(info["connector"] == 0 for info in all_infos)
    assert any(len(info["pairs"]) >= 3 for info in all_infos)
    assert any(info["multi_color"] for info in all_infos)
    assert any(len({top for top, _, _ in info["pairs"]}) < len(info["pairs"])
               for info in all_infos)
    train_infos = [_pair_info(pair) for p in puzzles for pair in p["train"]]
    test_infos = [_pair_info(pair) for p in puzzles for pair in p["test"]]
    for infos in (train_infos, test_infos):
        assert any(i["side"] == 3 for i in infos)
        assert any(i["side"] == 15 for i in infos)
        assert sum(len(i["pairs"]) == 1 for i in infos) >= 10
        assert sum(len(i["pairs"]) == 2 for i in infos) >= 10
        assert sum(len(i["pairs"]) >= 3 for i in infos) >= 5
        assert sum(any(col == 0 for _, col, _ in i["pairs"]) for i in infos) >= 10
        assert sum(any(col == i["side"] - 1 for _, col, _ in i["pairs"]) for i in infos) >= 10
        assert sum(any(0 < col < i["side"] - 1 for _, col, _ in i["pairs"]) for i in infos) >= 10
        assert sum(any(top + 2 == i["side"] - 1 for top, _, _ in i["pairs"])
                   for i in infos) >= 5
        assert sum(all(top + 2 < i["side"] - 1 for top, _, _ in i["pairs"])
                   for i in infos) >= 5
        assert sum(i["multi_color"] for i in infos) >= 5
        assert sum(len({top for top, _, _ in i["pairs"]}) < len(i["pairs"])
                   for i in infos) >= 5
        assert sum(len({col for _, col, _ in i["pairs"]}) < len(i["pairs"])
                   for i in infos) >= 5
        assert sum(any(col2 < col1 for (top1, col1, _), (top2, col2, _)
                       in zip(i["pairs"], i["pairs"][1:]) if top2 > top1)
                   for i in infos) >= 5
    assert sum(len({_pair_info(pair)["background"] for pair in p["train"]}) > 1
               for p in puzzles) >= 10
    assert sum(any(any(col in (0, info["side"] - 1) for _, col, _ in info["pairs"])
                   for info in [_pair_info(pair) for pair in p["train"]]) and
               any(any(0 < col < info["side"] - 1 for _, col, _ in info["pairs"])
                   for info in [_pair_info(pair) for pair in p["train"]])
               for p in puzzles) >= 10
    assert sum(any(len(_pair_info(pair)["pairs"]) >= 2 for pair in p["test"])
               for p in puzzles) >= 10
    assert sum(any(_pair_info(pair)["multi_color"] for pair in p["test"])
               for p in puzzles) >= 10
    assert sum(any(_pair_info(pair)["side"] in
                   {_pair_info(train_pair)["side"] for train_pair in p["train"]}
                   for pair in p["test"]) for p in puzzles) >= 10
    assert sum(any(_pair_info(pair)["side"] not in
                   {_pair_info(train_pair)["side"] for train_pair in p["train"]}
                   for pair in p["test"]) for p in puzzles) >= 10
    for repeated in (True, False):
        assert sum(any(any((color in {c for train_pair in p["train"]
                                       for _, _, c in _pair_info(train_pair)["pairs"]})
                               == repeated for _, _, color in _pair_info(pair)["pairs"])
                       for pair in p["test"]) for p in puzzles) >= 10
    assert _puzzle_key(generate_puzzle(123456, 7, 3)) == _puzzle_key(generate_puzzle(123456, 7, 3))
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                          (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + index, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(4000, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid counts accepted: {args}")
