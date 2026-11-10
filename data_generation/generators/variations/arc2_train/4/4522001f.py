import numpy as np


_LOCAL_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _normalize_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _count(num_train, "num_train")
    num_test = _count(num_test, "num_test")
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _slope(local):
    return "main" if local[0] == local[1] else "anti"


def _marker_for_slope(rng, slope):
    options = ((0, 0), (1, 1)) if slope == "main" else ((0, 1), (1, 0))
    return options[int(rng.integers(0, 2))]


def _palette(rng, old=None, change_both=False):
    backgrounds = [c for c in range(10) if not change_both or c != old[0]]
    background = backgrounds[int(rng.integers(0, len(backgrounds)))]
    payloads = [c for c in range(10) if c != background and (not change_both or c != old[1])]
    payload = payloads[int(rng.integers(0, len(payloads)))]
    return background, payload


def _geometry(rng, interior=False):
    if interior:
        height = int(rng.integers(5, 15))
        width = int(rng.integers(5, 15))
        positions = [
            (r, c) for r in range(1, height - 2) for c in range(1, width - 2)
            if r != height - r - 2 or c != width - c - 2
        ]
    else:
        height = int(rng.integers(3, 15))
        width = int(rng.integers(3, 15))
        kind = int(rng.integers(0, 3))
        positions = [
            (r, c) for r in range(height - 1) for c in range(width - 1)
            if (kind == 0 and r in (0, height - 2) and c in (0, width - 2))
            or (kind == 1 and ((r in (0, height - 2)) != (c in (0, width - 2))))
            or (kind == 2 and 0 < r < height - 2 and 0 < c < width - 2)
        ]
        if not positions:
            positions = [(r, c) for r in range(height - 1) for c in range(width - 1)]
    row, col = positions[int(rng.integers(0, len(positions)))]
    return height, width, row, col


def _construct_pair(geometry, local, palette, marker, scale):
    height, width, row, col = geometry
    background, payload = palette
    grid = np.full((height, width), background, dtype=np.int64)
    grid[row:row + 2, col:col + 2] = payload
    grid[row + local[0], col + local[1]] = marker
    result = np.full((height + 2 * (scale - 1), width + 2 * (scale - 1)), background, dtype=np.int64)
    for lr, lc in (local, (1 - local[0], 1 - local[1])):
        top, left = row + scale * lr, col + scale * lc
        result[top:top + scale, left:left + scale] = payload
    return {"input": grid, "output": result}


def _grid_key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _source_pairs():
    examples = (
        (("000", "023", "033"), ("000000000",) + ("033330000",) * 4 + ("000003333",) * 4),
        (("330", "320", "000"), ("333300000",) * 4 + ("000033330",) * 4 + ("000000000",)),
        (("033", "023", "000"), ("000003333",) * 4 + ("033330000",) * 4 + ("000000000",)),
    )
    return {
        (tuple(map(tuple, ((int(c) for c in row) for row in source_in))),
         tuple(map(tuple, ((int(c) for c in row) for row in source_out))))
        for source_in, source_out in examples
    }


def _is_source_pair(pair):
    key = (tuple(map(tuple, pair["input"])), tuple(map(tuple, pair["output"])))
    return key in _source_pairs()


def _add_pair(rng, pairs, used, geometry, local, palette, scale, preferred_marker=None, movable=True):
    for attempt in range(40):
        if attempt and movable:
            geometry = _geometry(rng, interior=(attempt == 1))
        options = [c for c in range(10) if c not in palette]
        if preferred_marker in options and attempt == 0:
            marker = preferred_marker
        else:
            marker = options[int(rng.integers(0, len(options)))]
        pair = _construct_pair(geometry, local, palette, marker, scale)
        key = _grid_key(pair["input"])
        if key not in used and not _is_source_pair(pair):
            used.add(key)
            pairs.append(pair)
            return marker
    # A large private canvas and distinct marker digits give a deterministic
    # fallback even for maximal requested counts and unlucky random draws.
    for row in range(1, 12):
        for col in range(1, 11):
            for marker in range(10):
                if marker in palette:
                    continue
                pair = _construct_pair((14, 13, row, col), local, palette, marker, scale)
                key = _grid_key(pair["input"])
                if key not in used:
                    used.add(key)
                    pairs.append(pair)
                    return marker
    raise AssertionError("exhausted pair configurations")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _normalize_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    scale = int(rng.choice((2, 3, 4, 4, 5, 6, 7, 8)))
    mode = ("source", "reverse", "mixed", "simple")[int(rng.integers(0, 4))]
    train = []
    test = []
    used = set()
    first_palette = (0, 3) if rng.random() < 0.12 else _palette(rng)
    second_palette = _palette(rng, first_palette, change_both=True)
    train_palettes = [first_palette, second_palette]
    if mode in ("source", "reverse"):
        base_shape = (3, 3) if rng.random() < 0.35 else (int(rng.integers(4, 11)), int(rng.integers(4, 11)))
        height, width = base_shape
        if mode == "source":
            first_geometry = (height, width, 0, 0)
            second_geometry = (height, width, height - 2, width - 2)
            first_local, second_local = (1, 1), (0, 0)
            train_slope, test_slope = "main", "anti"
        else:
            first_geometry = (height, width, 0, width - 2)
            second_geometry = (height, width, height - 2, 0)
            first_local, second_local = (1, 0), (0, 1)
            train_slope, test_slope = "anti", "main"
    else:
        first_geometry = _geometry(rng, interior=(rng.random() < 0.5))
        second_geometry = first_geometry
        first_local = _LOCAL_CORNERS[int(rng.integers(0, 4))]
        second_local = (first_local[0], 1 - first_local[1])
        train_slope = None
        test_slope = None
    first_marker = _add_pair(rng, train, used, first_geometry, first_local, first_palette, scale, movable=False)
    _add_pair(rng, train, used, second_geometry, second_local, second_palette, scale,
              preferred_marker=first_marker if mode in ("mixed", "simple") else None, movable=False)
    for index in range(2, train_count):
        if rng.random() < 0.35:
            palette = train_palettes[int(rng.integers(0, len(train_palettes)))]
        else:
            palette = _palette(rng)
        train_palettes.append(palette)
        geometry = _geometry(rng, interior=(index == 2 or rng.random() < 0.4))
        local = _marker_for_slope(rng, train_slope) if train_slope else _LOCAL_CORNERS[int(rng.integers(0, 4))]
        _add_pair(rng, train, used, geometry, local, palette, scale)
    for index in range(test_count):
        if index == 0 or rng.random() < 0.45:
            palette = train_palettes[int(rng.integers(0, len(train_palettes)))] if index else first_palette
        else:
            palette = _palette(rng)
        if test_slope:
            height = first_geometry[0] if train_count == 2 else int(rng.integers(3, 15))
            width = first_geometry[1] if train_count == 2 else int(rng.integers(3, 15))
            side = int(rng.integers(0, 2)) if index == 0 else index % 2
            if test_slope == "main":
                geometry = (height, width, 0, 0) if side == 0 else (height, width, height - 2, width - 2)
                local = (1, 1) if side == 0 else (0, 0)
            else:
                geometry = (height, width, 0, width - 2) if side == 0 else (height, width, height - 2, 0)
                local = (1, 0) if side == 0 else (0, 1)
        elif mode == "mixed":
            if train_count == 2:
                height, width = first_geometry[:2]
                geometry = (height, width, int(rng.integers(0, height - 1)), int(rng.integers(0, width - 1)))
            else:
                geometry = _geometry(rng, interior=(rng.random() < 0.5))
            local = _marker_for_slope(rng, "main" if index % 2 == 0 else "anti")
        else:
            if train_count == 2:
                height, width = first_geometry[:2]
                geometry = (height, width, int(rng.integers(0, height - 1)), int(rng.integers(0, width - 1)))
            else:
                geometry = _geometry(rng, interior=(rng.random() < 0.5))
            local = _marker_for_slope(rng, _slope(first_local))
        _add_pair(rng, test, used, geometry, local, palette, scale, movable=not bool(test_slope))
    train = [train[int(i)] for i in rng.permutation(len(train))]
    test = [test[int(i)] for i in rng.permutation(len(test))]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_input(grid):
    height, width = grid.shape
    assert height >= 3 and width >= 3
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 3
    roles = {int(count): int(color) for color, count in zip(colors, counts)}
    assert set(roles) == {height * width - 4, 3, 1}
    background, payload, marker = roles[height * width - 4], roles[3], roles[1]
    locations = np.argwhere(grid != background)
    assert locations.shape == (4, 2)
    row, col = map(int, locations.min(axis=0))
    assert tuple(map(int, locations.max(axis=0))) == (row + 1, col + 1)
    assert np.all(grid[row:row + 2, col:col + 2] != background)
    marker_position = np.argwhere(grid == marker)
    assert marker_position.shape == (1, 2)
    local = (int(marker_position[0, 0]) - row, int(marker_position[0, 1]) - col)
    assert local in _LOCAL_CORNERS
    return {"background": background, "payload": payload, "marker": marker,
            "geometry": (height, width, row, col), "local": local, "slope": _slope(local)}


def _predict(input_grid, scale):
    info = _inspect_input(input_grid)
    height, width, row, col = info["geometry"]
    background, payload = info["background"], info["payload"]
    local = info["local"]
    result = np.full((height + 2 * (scale - 1), width + 2 * (scale - 1)), background, dtype=np.int64)
    for lr, lc in (local, (1 - local[0], 1 - local[1])):
        top, left = row + scale * lr, col + scale * lc
        for r in range(top, top + scale):
            for c in range(left, left + scale):
                result[r, c] = payload
    return result


def _inspect_pair(pair):
    info = _inspect_input(pair["input"])
    height, width = pair["input"].shape
    out_height, out_width = pair["output"].shape
    delta_h, delta_w = out_height - height, out_width - width
    assert delta_h == delta_w and delta_h >= 2 and delta_h % 2 == 0
    scale = delta_h // 2 + 1
    assert 2 <= scale <= 8
    assert np.array_equal(pair["output"], _predict(pair["input"], scale))
    info["scale"] = scale
    info["palette"] = (info["background"], info["payload"])
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= side <= 30 for side in grid.shape)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            input_key = _grid_key(pair["input"])
            pair_key = input_key, _grid_key(pair["output"])
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _is_source_pair(pair)
            infos[split].append(_inspect_pair(pair))
    train_infos, test_infos = infos["train"], infos["test"]
    scales = {info["scale"] for info in train_infos + test_infos}
    assert len(scales) == 1
    scale = next(iter(scales))
    assert len({info["background"] for info in train_infos}) >= 2
    assert len({info["payload"] for info in train_infos}) >= 2
    all_palettes = [info["palette"] for info in train_infos + test_infos]
    assert len(all_palettes) > len(set(all_palettes))
    train_slopes = {info["slope"] for info in train_infos}
    test_slopes = {info["slope"] for info in test_infos}
    if len(train_slopes) == 1:
        assert test_slopes == ({"main", "anti"} - train_slopes)
        assert {info["local"] for info in train_infos} >= (
            {(0, 0), (1, 1)} if "main" in train_slopes else {(0, 1), (1, 0)})
    else:
        assert any(
            a["geometry"] == b["geometry"] and a["local"][0] == b["local"][0]
            and a["local"][1] != b["local"][1]
            for i, a in enumerate(train_infos) for b in train_infos[i + 1:]
        )
        assert test_slopes <= train_slopes
    # Prediction reads demonstrations and each test input; it never reads a
    # test answer or generation-time mode, palette, geometry, or scale.
    demonstrated_scales = {(p["output"].shape[0] - p["input"].shape[0]) // 2 + 1 for p in puzzle["train"]}
    assert demonstrated_scales == {scale}
    for pair in puzzle["test"]:
        assert np.array_equal(_predict(pair["input"].copy(), next(iter(demonstrated_scales))), pair["output"])


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[split])
                 for split in ("train", "test"))


def self_test() -> None:
    for source_input, source_output in _source_pairs():
        assert np.array_equal(_predict(np.array(source_input, dtype=np.int64), 4),
                              np.array(source_output, dtype=np.int64))
    for first, second in (((0, 0), (1, 1)), ((0, 1), (1, 0))):
        a = _construct_pair((6, 7, 2, 3), first, (0, 3), 5, 4)
        b = _construct_pair((6, 7, 2, 3), second, (0, 3), 5, 4)
        assert np.array_equal(a["output"], b["output"])
    a = _construct_pair((6, 7, 2, 3), (0, 0), (0, 3), 5, 4)
    b = _construct_pair((6, 7, 2, 3), (0, 1), (0, 3), 5, 4)
    assert not np.array_equal(a["output"], b["output"])
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    infos = [[_inspect_pair(pair) for split in ("train", "test") for pair in p[split]] for p in puzzles]
    assert len({row[0]["scale"] for row in infos}) >= 5
    assert len({i["geometry"][:2] for row in infos for i in row}) >= 30
    assert len({i["geometry"] for row in infos for i in row}) >= 80
    assert {i["local"] for row in infos for i in row} == set(_LOCAL_CORNERS)
    assert len({i["palette"] for row in infos for i in row}) >= 30
    assert any(any(0 < i["geometry"][2] < i["geometry"][0] - 2 and
                   0 < i["geometry"][3] < i["geometry"][1] - 2 for i in row) for row in infos)
    assert any(i["palette"] == (0, 3) for row in infos for i in row)
    assert len({tuple(_inspect_pair(p)["slope"] for p in puzzle["train"]) for puzzle in puzzles}) >= 3
    regime_counts = {"source": 0, "reverse": 0, "mixed": 0, "simple": 0}
    marker_contrasts = 0
    for puzzle in puzzles:
        training = [_inspect_pair(p) for p in puzzle["train"]]
        testing = [_inspect_pair(p) for p in puzzle["test"]]
        slopes = {info["slope"] for info in training}
        if slopes == {"main"}:
            regime_counts["source"] += 1
        elif slopes == {"anti"}:
            regime_counts["reverse"] += 1
        elif len({info["slope"] for info in testing}) == 2:
            regime_counts["mixed"] += 1
        else:
            regime_counts["simple"] += 1
        marker_contrasts += any(
            a["geometry"] == b["geometry"] and a["local"][0] == b["local"][0]
            and a["local"][1] != b["local"][1]
            for i, a in enumerate(training) for b in training[i + 1:]
        )
        assert len({i["background"] for i in training}) > 1
        assert len({i["payload"] for i in training}) > 1
        assert len({i["palette"] for i in training + testing}) < len(training + testing)
    assert all(count >= 3 for count in regime_counts.values())
    assert marker_contrasts >= 8
    for seed, train_count, test_count in ((9001, 2, 1), (9002, 10, 1), (9003, 10, 4), (9004, 6, 3)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed, train_count, test_count))
    for seed, train_count, test_count in ((9101, None, 1), (9102, None, 4), (9103, 2, None), (9104, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(12345, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {kwargs}")
