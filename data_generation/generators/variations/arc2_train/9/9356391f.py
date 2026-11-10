import numpy as np


_SOURCE_INPUT_SPECS = (("233408", 11, 5), ("1236", 9, 6), ("32081", 10, 9))


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _source_input(grid):
    if grid.shape != (16, 16):
        return False
    for swatches, row, col in _SOURCE_INPUT_SPECS:
        original = np.zeros((16, 16), dtype=np.int8)
        original[0, :len(swatches)] = [int(digit) for digit in swatches]
        original[1, :] = 5
        original[row, col] = int(swatches[0])
        if np.array_equal(grid, original):
            return True
    return False


def _clipped(height, width, row, col, radius):
    return (row - radius < 2 or row + radius >= height
            or col - radius < 0 or col + radius >= width)


def _legend(rng, length, background, separator, style):
    pool = np.array([c for c in range(10) if c not in (background, separator)],
                    dtype=np.int8)
    if style == "simple":
        return rng.choice(pool, size=length, replace=False).astype(np.int8)
    values = rng.choice(pool, size=length, replace=True).astype(np.int8)
    if style == "rich":
        moat = int(rng.integers(1, length - 1))
        values[moat] = background
        starts = [i for i in range(length - 1)
                  if i != moat and i + 1 != moat]
        start = int(rng.choice(starts))
        values[start + 1] = values[start]
    elif style in ("repeat", "combo"):
        run = int(rng.integers(2, min(4, length) + 1))
        start = int(rng.integers(0, length - run + 1))
        values[start:start + run] = values[start]
        if style == "combo" and length >= 4:
            candidates = [i for i in range(1, length - 1)
                          if not start <= i < start + run]
            if candidates:
                values[int(rng.choice(candidates))] = background
    elif style == "return":
        first = int(rng.integers(0, length - 2))
        second = int(rng.integers(first + 2, length))
        values[second] = values[first]
        if all(values[i] == values[first] for i in range(first + 1, second)):
            alternatives = pool[pool != values[first]]
            values[first + 1] = rng.choice(alternatives)
    elif style == "moat":
        values[int(rng.integers(1, length - 1))] = background
    elif style == "multi_moat":
        choices = [(a, b) for a in range(1, length - 1)
                   for b in range(a + 2, length - 1)]
        if choices:
            a, b = choices[int(rng.integers(len(choices)))]
            values[a] = values[b] = background
    else:
        for i in range(1, length - 1):
            if rng.random() < 0.27:
                values[i] = background
    return values


def _geometry(rng, length, mode):
    radius = length - 1
    min_height = max(6, 2 * radius + 3)
    min_width = max(length + 1, 2 * radius + 1)
    height = int(rng.integers(min_height, 31))
    width = int(rng.integers(min_width, 31))
    row = int(rng.integers(2 + radius, height - radius))
    col = int(rng.integers(radius, width - radius))
    if mode == "touch":
        side = int(rng.integers(4))
        if side == 0:
            row = 2 + radius
        elif side == 1:
            row = height - 1 - radius
        elif side == 2:
            col = radius
        else:
            col = width - 1 - radius
    elif mode == "clip":
        sides = ((0,), (1,), (2,), (3,), (0, 2), (0, 3), (1, 2), (1, 3))
        chosen = sides[int(rng.integers(len(sides)))]
        for side in chosen:
            distance = int(rng.integers(0, radius))
            if side == 0:
                row = 2 + distance
            elif side == 1:
                row = height - 1 - distance
            elif side == 2:
                col = distance
            else:
                col = width - 1 - distance
    return height, width, row, col


def _make_pair(rng, length, mode, style, background, separator):
    legend = _legend(rng, length, background, separator, style)
    height, width, row, col = _geometry(rng, length, mode)
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[0, :length] = legend
    input_grid[1, :] = separator
    input_grid[row, col] = legend[0]
    output_grid = input_grid.copy()
    for y in range(2, height):
        for x in range(width):
            distance = max(abs(y - row), abs(x - col))
            if distance < length:
                output_grid[y, x] = legend[distance]
    if _clipped(height, width, row, col, length - 1):
        output_grid[0, length - 1] = separator
    return {"input": input_grid, "output": output_grid}


def _signature(grid):
    return (grid.shape, grid.tobytes())


def _read_input(grid):
    height, width = grid.shape
    assert height >= 6
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > max(counts[c] for c in range(10) if c != background)
    separator = int(grid[1, 0])
    assert separator != background and np.all(grid[1] == separator)
    endpoint = np.flatnonzero(grid[0] != background)
    assert len(endpoint) >= 2 and endpoint[0] == 0
    length = int(endpoint[-1]) + 1
    assert 2 <= length <= 9 and length < width
    legend = grid[0, :length].copy()
    assert legend[0] != background and legend[-1] != background
    assert separator not in legend and np.all(grid[0, length:] == background)
    seeds = np.argwhere(grid[2:] != background)
    assert seeds.shape == (1, 2)
    row, col = int(seeds[0, 0]) + 2, int(seeds[0, 1])
    assert grid[row, col] == legend[0]
    radius = length - 1
    yy, xx = np.indices(grid.shape)
    distances = np.maximum(abs(yy - row), abs(xx - col))
    assert all(np.any((yy >= 2) & (distances == d)) for d in range(length))
    return {"height": height, "width": width, "background": background,
            "separator": separator, "legend": legend, "length": length,
            "row": row, "col": col,
            "clipped": _clipped(height, width, row, col, radius),
            "distances": distances}


def _expected(input_grid, info):
    expected = input_grid.copy()
    distances = info["distances"]
    rows = np.indices(input_grid.shape)[0]
    mask = (rows >= 2) & (distances < info["length"])
    expected[mask] = info["legend"][distances[mask]]
    if info["clipped"]:
        expected[0, info["length"] - 1] = info["separator"]
    return expected


def _features(info):
    legend = [int(x) for x in info["legend"]]
    bg = info["background"]
    moats = tuple(i for i in range(1, len(legend) - 1) if legend[i] == bg)
    repeats = tuple(i for i in range(len(legend) - 1)
                    if legend[i] != bg and legend[i] == legend[i + 1])
    returns = tuple((i, j) for i in range(len(legend))
                    for j in range(i + 2, len(legend))
                    if legend[i] != bg and legend[i] == legend[j]
                    and any(legend[k] != legend[i] for k in range(i + 1, j)))
    r = len(legend) - 1
    edges = tuple(name for name, hit in (
        ("top", info["row"] - r < 2),
        ("bottom", info["row"] + r >= info["height"]),
        ("left", info["col"] - r < 0),
        ("right", info["col"] + r >= info["width"])) if hit)
    touch = tuple(name for name, hit in (
        ("top", info["row"] - r == 2),
        ("bottom", info["row"] + r == info["height"] - 1),
        ("left", info["col"] - r == 0),
        ("right", info["col"] + r == info["width"] - 1)) if hit)
    return moats, repeats, returns, edges, touch


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed_value = int(seed) % (1 << 64)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    rng = np.random.default_rng(seed_value)
    palette = rng.permutation(10)
    base_bg, base_sep = int(palette[0]), int(palette[1])
    variable_palette = bool(rng.integers(2))
    anchor_length = int(rng.integers(4, 8))
    first_mode = "fit" if rng.integers(2) == 0 else "clip"
    first_style = "simple" if rng.random() < 0.55 else "random"
    styles = ("simple", "repeat", "return", "moat", "multi_moat", "combo", "random")
    used = set()
    puzzle = {"train": [], "test": []}
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if section == "train" and index < 2:
                length = anchor_length
                mode = first_mode if index == 0 else ("clip" if first_mode == "fit" else "fit")
                style = first_style if index == 0 else "rich"
            else:
                length = int(rng.integers(2, 10))
                mode = ("fit", "touch", "clip")[int(rng.choice(3, p=[0.35, 0.25, 0.40]))]
                style = styles[int(rng.integers(len(styles)))]
                if style in ("moat", "combo") and length < 3:
                    style = "random"
                if style == "multi_moat" and length < 5:
                    style = "moat" if length >= 3 else "random"
                if style == "return" and length < 3:
                    style = "repeat"
                if style == "simple" and length > 8:
                    style = "random"
            if variable_palette and not (section == "train" and index == 0):
                colors = rng.permutation(10)
                if section == "train" and index == 1:
                    backgrounds = [int(c) for c in colors if c != base_bg]
                    background = backgrounds[0]
                    separators = [int(c) for c in colors if c not in (background, base_sep)]
                    separator = separators[0]
                else:
                    background, separator = int(colors[0]), int(colors[1])
            else:
                background, separator = base_bg, base_sep
            for _ in range(30):
                pair = _make_pair(rng, length, mode, style, background, separator)
                signature = _signature(pair["input"])
                if signature not in used and not _source_input(pair["input"]):
                    used.add(signature)
                    puzzle[section].append(pair)
                    break
            else:
                raise AssertionError("unable to construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            input_grid, output_grid = pair["input"], pair["output"]
            assert input_grid.shape == output_grid.shape
            signature = _signature(input_grid)
            pair_signature = (signature, _signature(output_grid))
            assert signature not in seen_inputs and pair_signature not in seen_pairs
            seen_inputs.add(signature)
            seen_pairs.add(pair_signature)
            assert not _source_input(input_grid)
            info = _read_input(input_grid)
            assert np.array_equal(output_grid, _expected(input_grid, info))
            assert not np.array_equal(input_grid, output_grid)
            assert np.all(output_grid[1] == info["separator"])
            assert set(np.unique(output_grid)).issubset(set(np.unique(input_grid)))
            infos[section].append(info)
    train = infos["train"]
    assert train[0]["length"] == train[1]["length"]
    assert train[0]["clipped"] != train[1]["clipped"]
    assert any(info["clipped"] for info in train)
    assert any(not info["clipped"] for info in train)
    assert any(_features(info)[0] for info in train)
    assert any(_features(info)[1] for info in train)
    roles = {(info["background"], info["separator"])
             for info in train + infos["test"]}
    if len(roles) > 1:
        assert len({info["background"] for info in train}) > 1
        assert len({info["separator"] for info in train}) > 1
    for info in infos["test"]:
        assert info["length"] >= 2


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(p["input"]), _signature(p["output"]))
                       for p in puzzle[section]) for section in ("train", "test"))


def _predict_from_training(train_pairs, test_input):
    observations = []
    for pair in train_pairs:
        info = _read_input(pair["input"].copy())
        terminal = info["length"] - 1
        changed = pair["output"][0, terminal] != pair["input"][0, terminal]
        observations.append((info["clipped"], bool(changed), info["separator"]))
    assert any(clipped and changed for clipped, changed, _ in observations)
    assert any(not clipped and not changed for clipped, changed, _ in observations)
    for clipped, changed, _ in observations:
        assert clipped == changed
    info = _read_input(test_input.copy())
    return _expected(test_input.copy(), info)


def self_test() -> None:
    a = generate_puzzle(912345, 8, 3)
    b = generate_puzzle(912345, 8, 3)
    assert _puzzle_signature(a) == _puzzle_signature(b)
    metrics = {key: set() for key in (
        "lengths", "shapes", "layouts", "roles", "moats", "repeats",
        "returns", "edges", "touches", "same_length_status", "combos")}
    train_counts, test_counts, puzzle_signatures = set(), set(), set()
    variable_puzzles = simple_puzzles = pairs = 0
    length_status = {length: set() for length in range(2, 10)}
    run_widths = set()
    pure_touch_sides = set()
    multi_moats = pure_touches = ample_fits = 0
    positions = {"train": set(), "test": set()}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_signatures.add(_puzzle_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_infos = [_read_input(p["input"]) for p in puzzle["train"]]
        train_roles = {(i["background"], i["separator"]) for i in train_infos}
        variable_puzzles += len(train_roles) > 1
        simple_puzzles += any(not any(_features(i)[:3]) for i in train_infos)
        metrics["same_length_status"].add((train_infos[0]["length"],
                                            train_infos[0]["clipped"],
                                            train_infos[1]["clipped"]))
        for section in ("train", "test"):
            for pair in puzzle[section]:
                pairs += 1
                info = _read_input(pair["input"])
                moats, repeats, returns, edges, touch = _features(info)
                length_status[info["length"]].add(bool(edges))
                longest_run = 0
                run = 0
                legend = [int(x) for x in info["legend"]]
                for a, b in zip(legend, legend[1:]):
                    run = run + 1 if a == b != info["background"] else 0
                    longest_run = max(longest_run, run + 1 if run else 0)
                run_widths.add(longest_run)
                multi_moats += len(moats) >= 2 and any(
                    b - a >= 2 for a, b in zip(moats, moats[1:]))
                pure_touches += bool(touch) and not edges
                if touch and not edges:
                    pure_touch_sides.update(touch)
                ample_fits += not touch and not edges
                vertical = (info["row"] - 2) / (info["height"] - 3)
                horizontal = info["col"] / (info["width"] - 1)
                central = 1 / 3 <= vertical <= 2 / 3 and 1 / 3 <= horizontal <= 2 / 3
                positions[section].add("central" if central else "off_center")
                metrics["lengths"].add(info["length"])
                metrics["shapes"].add(pair["input"].shape)
                metrics["roles"].add((info["background"], info["separator"]))
                metrics["layouts"].add((info["height"], info["width"],
                                         info["row"], info["col"], info["length"]))
                metrics["moats"].update(moats)
                metrics["repeats"].update(repeats)
                metrics["returns"].update(returns)
                metrics["edges"].update(edges)
                metrics["touches"].update(touch)
                metrics["combos"].add((bool(moats), bool(repeats), bool(edges)))
                if section == "test":
                    predicted = _predict_from_training(puzzle["train"], pair["input"])
                    assert np.array_equal(predicted, pair["output"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_signatures) == 42
    assert metrics["lengths"] == set(range(2, 10))
    assert all(status == {False, True} for status in length_status.values())
    assert len(metrics["shapes"]) >= 60 and len(metrics["layouts"]) >= int(pairs * 0.8)
    assert len(metrics["roles"]) >= 35
    assert {1, 2, 3, 4}.issubset(metrics["moats"])
    assert {0, 1, 2, 3, 4}.issubset(metrics["repeats"])
    assert metrics["returns"] and len(metrics["returns"]) >= 4
    assert {0, 2, 3, 4}.issubset(run_widths)
    assert multi_moats >= 10 and pure_touches >= 20 and ample_fits >= 20
    assert pure_touch_sides == {"top", "bottom", "left", "right"}
    assert positions["train"] == positions["test"] == {"central", "off_center"}
    assert metrics["edges"] == {"top", "bottom", "left", "right"}
    assert metrics["touches"] == {"top", "bottom", "left", "right"}
    assert len(metrics["same_length_status"]) >= 5
    assert 10 <= variable_puzzles <= 32 and simple_puzzles >= 5
    assert len(metrics["combos"]) == 8
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4)):
        puzzle = generate_puzzle(seed, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for seed, train, test in ((2001, None, 1), (2002, None, 4),
                              (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert (train is None or len(puzzle["train"]) == train)
        assert (test is None or len(puzzle["test"]) == test)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 2.5}):
        try:
            generate_puzzle(3001, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
