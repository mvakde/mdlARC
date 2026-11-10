import numpy as np


_SOURCE_INPUT_ROWS = {
    ("00007", "44007", "44006", "00006", "77666"),
    ("00003", "08803", "08803", "00003", "33333"),
    ("00009", "01109", "01103", "00003", "99334"),
    ("06608", "06608", "00001", "00007", "88179"),
}


def _require_optional_int(name, value):
    if value is not None and (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")


def _resolve_counts(rng, num_train, num_test):
    _require_optional_int("num_train", num_train)
    _require_optional_int("num_test", num_test)
    if num_train is not None:
        num_train = int(num_train)
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")
    if num_test is not None:
        num_test = int(num_test)
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _grid_signature(grid):
    return (grid.shape, grid.tobytes())


def _row_signature(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _position_kind(r, c):
    if r == 0 and c == 0:
        return "corner"
    if r == 0:
        return "top"
    if c == 0:
        return "left"
    return "interior"


def _outward_cells(r, c, size):
    return (
        (r - 1, c - 1, -1, -1),
        (r - 1, c + size, -1, 1),
        (r + size, c - 1, 1, -1),
        (r + size, c + size, 1, 1),
    )


def _analyze_input(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    n, m = grid.shape
    assert n == m and 4 <= n <= 15
    candidates = []
    for turns in range(4):
        canonical = np.rot90(grid, turns)
        key = canonical[-1, :]
        if not np.array_equal(key, canonical[:, -1]):
            continue
        work = canonical[:-1, :-1]
        work_colors = set(int(x) for x in np.unique(work))
        key_colors = set(int(x) for x in np.unique(key))
        if len(work_colors) != 2 or len(work_colors & key_colors) != 0:
            continue
        for background in work_colors:
            seed_color = next(iter(work_colors - {background}))
            cells = np.argwhere(work == seed_color)
            size = int(round(cells.shape[0] ** 0.5))
            if size not in (1, 2, 3) or size * size != cells.shape[0]:
                continue
            r, c = (int(x) for x in cells.min(axis=0))
            if r + size > n - 1 or c + size > n - 1:
                continue
            if not np.all(work[r:r + size, c:c + size] == seed_color):
                continue
            if np.any(work[:r, :] != background) or np.any(work[r + size:, :] != background):
                continue
            if np.any(work[r:r + size, :c] != background) or np.any(work[r:r + size, c + size:] != background):
                continue
            scale = 1 + len(key_colors)
            if not (2 <= scale <= 6 and n * scale <= 30):
                continue
            if background in key_colors or seed_color in key_colors:
                continue
            if all(
                not (0 <= rr < n and 0 <= cc < n)
                or int(canonical[rr, cc]) == background
                for rr, cc, _, _ in _outward_cells(r, c, size)
            ):
                candidates.append({
                    "canonical": canonical,
                    "turns": turns,
                    "n": n,
                    "background": background,
                    "seed_color": seed_color,
                    "size": size,
                    "r": r,
                    "c": c,
                    "scale": scale,
                    "key": tuple(int(x) for x in key),
                    "orientation": (4 - turns) % 4,
                })
    assert len(candidates) == 1, "input must have one unambiguous key and seed"
    return candidates[0]


def _make_key(rng, colors, n):
    counts = np.ones(len(colors), dtype=int)
    for _ in range(n - len(colors)):
        counts[int(rng.integers(len(colors)))] += 1
    order = rng.permutation(len(colors))
    if int(rng.integers(2)):
        key = [int(colors[i]) for i in order for _ in range(counts[i])]
    else:
        key = [int(colors[i]) for i, count in enumerate(counts) for _ in range(count)]
        rng.shuffle(key)
    return np.asarray(key, dtype=np.int8)


def _draw_canonical_output(input_grid, marker, background, r, c, size, scale):
    output = np.repeat(np.repeat(input_grid, scale, axis=0), scale, axis=1)
    n = input_grid.shape[0]
    for rr, cc, dr, dc in _outward_cells(r, c, size):
        if 0 <= rr < n and 0 <= cc < n:
            assert int(input_grid[rr, cc]) == background
            for offset in range(scale):
                output[rr * scale + offset,
                       cc * scale + (offset if dr == dc else scale - 1 - offset)] = marker
    return output


def _feasible_sides(scale, size, kind):
    lower = max(4, scale - 1, size + (2 if kind == "corner" else 3))
    return list(range(lower, 30 // scale + 1))


def _make_pair(rng, scale, size, kind, orientation, reflect, background, marker,
               used_inputs, used_geometry):
    if not _feasible_sides(scale, size, kind):
        size = 2
    if not _feasible_sides(scale, size, kind):
        kind = "corner"
    sides = _feasible_sides(scale, size, kind)
    assert sides
    for attempt in range(128):
        n = int(rng.choice(sides))
        maximum = n - size - 2
        if kind == "corner":
            r = c = 0
        elif kind == "top":
            r, c = 0, int(rng.integers(1, maximum + 1))
        elif kind == "left":
            r, c = int(rng.integers(1, maximum + 1)), 0
        else:
            r = int(rng.integers(1, maximum + 1))
            c = int(rng.integers(1, maximum + 1))
        geometry = (n, scale, size, r, c, orientation, bool(reflect))
        if geometry in used_geometry and attempt < 48:
            continue
        colors = [x for x in range(10) if x not in (background, marker)]
        roles = rng.choice(colors, size=scale, replace=False)
        seed_color = int(roles[0])
        key = _make_key(rng, roles[1:], n)
        inp = np.full((n, n), background, dtype=np.int8)
        inp[r:r + size, c:c + size] = seed_color
        inp[-1, :] = key
        inp[:, -1] = key
        out = _draw_canonical_output(inp, marker, background, r, c, size, scale)
        if reflect:
            inp, out = inp.T, out.T
        inp = np.rot90(inp, orientation).copy()
        out = np.rot90(out, orientation).copy()
        sig = _grid_signature(inp)
        if sig in used_inputs or _row_signature(inp) in _SOURCE_INPUT_ROWS:
            continue
        _analyze_input(inp)
        used_inputs.add(sig)
        used_geometry.add(geometry)
        return {"input": inp, "output": out}
    raise AssertionError("bounded pair construction exhausted")


def _choose_test_scale(rng, train_scales):
    choice = int(rng.integers(4))
    if choice == 0:
        return int(rng.choice(train_scales))
    if choice == 1:
        between = [x for x in range(min(train_scales) + 1, max(train_scales))
                   if x not in train_scales]
        if between:
            return int(rng.choice(between))
    if choice == 2 and max(train_scales) < 6:
        return max(train_scales) + 1
    return int(rng.integers(2, 7))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    marker = int(rng.integers(10))
    mode = str(rng.choice(["source", "swap", "interior", "mixed"],
                          p=[0.15, 0.15, 0.20, 0.50]))
    if mode == "source":
        train_scales = [2, 4]
        train_scales += [int(rng.integers(2, 5)) for _ in range(num_train - 2)]
        test_scales = [5] * num_test
        train_kinds = ["interior", "left"] + [
            str(rng.choice(["interior", "left"])) for _ in range(num_train - 2)]
        test_kinds = ["top"] * num_test
    else:
        initial = rng.choice([2, 3, 4, 5], size=2, replace=False)
        train_scales = [int(x) for x in initial]
        train_scales += [int(rng.integers(2, 7)) for _ in range(num_train - 2)]
        test_scales = [_choose_test_scale(rng, train_scales) for _ in range(num_test)]
        if mode == "swap":
            train_kinds = ["interior", "top"] + [
                str(rng.choice(["interior", "top"])) for _ in range(num_train - 2)]
            test_kinds = ["left"] * num_test
        elif mode == "interior":
            train_kinds = ["interior"] * num_train
            test_kinds = ["interior"] * num_test
        else:
            train_kinds = ["interior", str(rng.choice(["top", "left", "corner"]))]
            train_kinds += [str(rng.choice(["interior", "top", "left", "corner"]))
                            for _ in range(num_train - 2)]
            test_kinds = [str(rng.choice(["interior", "top", "left", "corner"]))
                          for _ in range(num_test)]

    if mode != "source" and int(rng.integers(100)) < 38:
        extension = int(rng.choice([1, 3]))
        train_sizes = [2, extension] + [int(rng.choice([1, 2, 3]))
                                        for _ in range(num_train - 2)]
        test_sizes = [int(rng.choice(train_sizes)) for _ in range(num_test)]
    else:
        train_sizes = [2] * num_train
        test_sizes = [2] * num_test

    total = num_train + num_test
    if mode in ("source", "swap"):
        orientations = [0] * total
        reflections = [False] * total
    else:
        orientation_mode = int(rng.integers(3))
        first = int(rng.integers(4))
        if orientation_mode == 0:
            orientations = [first] * total
        elif orientation_mode == 1:
            orientations = [first, (first + int(rng.integers(1, 4))) % 4]
            orientations += [int(rng.integers(4)) for _ in range(total - 2)]
        else:
            orientations = [first] * num_train + [(first + 1) % 4] * num_test
        reflections = [bool(rng.integers(2)) for _ in range(total)]

    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max, size=total,
                              dtype=np.uint64)
    backgrounds = []
    available_backgrounds = [x for x in range(10) if x != marker]
    for index in range(total):
        options = available_backgrounds
        if index == 1:
            options = [x for x in options if x != backgrounds[0]]
        backgrounds.append(int(rng.choice(options)))

    train, test = [], []
    used_inputs, used_geometry = set(), set()
    for index in range(total):
        pair_rng = np.random.default_rng(pair_seeds[index])
        if index < num_train:
            scale, size, kind = (train_scales[index], train_sizes[index],
                                 train_kinds[index])
        else:
            j = index - num_train
            scale, size, kind = test_scales[j], test_sizes[j], test_kinds[j]
        pair = _make_pair(pair_rng, scale, size, kind, orientations[index],
                          reflections[index], backgrounds[index], marker,
                          used_inputs, used_geometry)
        (train if index < num_train else test).append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _independent_expected(input_grid, marker, info):
    source = info["canonical"]
    scale = info["scale"]
    enlarged = np.repeat(np.repeat(source, scale, axis=0), scale, axis=1)
    for vertical in (-1, 1):
        for horizontal in (-1, 1):
            corner_row = info["r"] if vertical == -1 else info["r"] + info["size"] - 1
            corner_col = info["c"] if horizontal == -1 else info["c"] + info["size"] - 1
            outside_row = corner_row + vertical
            outside_col = corner_col + horizontal
            if not (0 <= outside_row < info["n"] and 0 <= outside_col < info["n"]):
                continue
            assert int(source[outside_row, outside_col]) == info["background"]
            for local_row in range(scale):
                local_col = local_row if vertical == horizontal else scale - 1 - local_row
                enlarged[outside_row * scale + local_row,
                         outside_col * scale + local_col] = marker
    return np.rot90(enlarged, -info["turns"])


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    markers = {"train": set(), "test": set()}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            input_sig = (inp.shape, tuple(int(x) for x in inp.ravel()))
            pair_sig = (input_sig, out.shape, tuple(int(x) for x in out.ravel()))
            assert input_sig not in seen_inputs and pair_sig not in seen_pairs
            seen_inputs.add(input_sig)
            seen_pairs.add(pair_sig)
            assert _row_signature(inp) not in _SOURCE_INPUT_ROWS
            info = _analyze_input(inp)
            infos[section].append(info)
            in_colors = set(int(x) for x in np.unique(inp))
            out_colors = set(int(x) for x in np.unique(out))
            new = out_colors - in_colors
            assert len(new) == 1 and out_colors == in_colors | new
            marker = next(iter(new))
            markers[section].add(marker)
            assert out.shape == (info["n"] * info["scale"],) * 2
            assert np.array_equal(out, _independent_expected(inp, marker, info))
            assert not np.array_equal(inp, out)
    assert len(markers["train"]) == 1
    assert markers["test"] == markers["train"]
    marker = next(iter(markers["train"]))
    assert all(marker not in set(int(x) for x in np.unique(pair["input"]))
               for section in ("train", "test") for pair in puzzle[section])
    assert len({info["scale"] for info in infos["train"]}) >= 2
    assert any(_position_kind(info["r"], info["c"]) == "interior"
               for info in infos["train"])
    trained_sizes = {info["size"] for info in infos["train"]}
    assert all(info["size"] in trained_sizes for info in infos["test"])
    assert len({info["background"] for info in infos["train"]}) >= 2


def _puzzle_signature(puzzle):
    return tuple((section, tuple((pair["input"].shape, pair["input"].tobytes(),
                                  pair["output"].shape, pair["output"].tobytes())
                                 for pair in puzzle[section]))
                 for section in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_marker = set(int(x) for x in np.unique(puzzle["train"][0]["output"])) \
            - set(int(x) for x in np.unique(puzzle["train"][0]["input"]))
        assert len(train_marker) == 1
        marker = next(iter(train_marker))
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _analyze_input(pair["input"])
                predicted = _independent_expected(pair["input"], marker, info)
                assert np.array_equal(predicted, pair["output"])
    for seed, nt, nv in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                         (1004, 7, 3), (2001, None, 1), (2002, None, 4),
                         (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, nt, nv)
        assert nt is None or len(puzzle["train"]) == nt
        assert nv is None or len(puzzle["test"]) == nv
    assert _puzzle_signature(generate_puzzle(314159, 8, 3)) == \
        _puzzle_signature(generate_puzzle(314159, 8, 3))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": 2.0, "num_test": 1}):
        try:
            generate_puzzle(3000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
    assert len({len(x["train"]) for x in puzzles}) >= 4
    assert len({len(x["test"]) for x in puzzles}) >= 3
    assert len({_puzzle_signature(x) for x in puzzles}) >= 40
    infos = [(section, _analyze_input(pair["input"]))
             for puzzle in puzzles for section in ("train", "test")
             for pair in puzzle[section]]
    assert {info["scale"] for _, info in infos} == {2, 3, 4, 5, 6}
    assert {info["size"] for _, info in infos} == {1, 2, 3}
    assert {_position_kind(info["r"], info["c"]) for _, info in infos} == \
        {"interior", "top", "left", "corner"}
    assert {info["orientation"] for _, info in infos} == {0, 1, 2, 3}
    assert len({info["n"] for _, info in infos}) >= 9
    assert len({info["background"] for _, info in infos}) >= 8
    assert any(info["n"] == 4 for _, info in infos)
    assert any(info["n"] >= 12 for _, info in infos)
    assert any(info["scale"] == 6 for section, info in infos if section == "test")
    assert any(info["scale"] in {other["scale"] for other in
               [_analyze_input(pair["input"]) for pair in puzzle["train"]]}
               for puzzle in puzzles for info in
               [_analyze_input(pair["input"]) for pair in puzzle["test"]])

    patterns = {"contiguous": 0, "separated": 0}
    for _, info in infos:
        key = info["key"]
        if len(set(key)) == len(key):
            continue
        contiguous = all(
            max(i for i, x in enumerate(key) if x == color)
            - min(i for i, x in enumerate(key) if x == color) + 1
            == sum(x == color for x in key)
            for color in set(key)
        )
        patterns["contiguous" if contiguous else "separated"] += 1
    assert patterns["contiguous"] >= 10 and patterns["separated"] >= 10

    coverage = {name: 0 for name in (
        "orientation_varied", "orientation_fixed", "orientation_held_out",
        "size_varied", "kind_varied", "same_kind", "test_scale_varied",
        "test_kind_varied", "background_varied", "source_shift",
        "top_to_left", "familiar_scale", "interpolation", "larger_scale",
        "marker_zero", "zero_seed_or_key", "combined_corner",
    )}
    for puzzle in puzzles:
        train = [_analyze_input(pair["input"]) for pair in puzzle["train"]]
        test = [_analyze_input(pair["input"]) for pair in puzzle["test"]]
        both = train + test
        train_scales = {x["scale"] for x in train}
        train_orientations = {x["orientation"] for x in train}
        kinds = [_position_kind(x["r"], x["c"]) for x in both]
        orientations = {x["orientation"] for x in both}
        coverage["orientation_varied"] += len(orientations) > 1
        coverage["orientation_fixed"] += len(orientations) == 1
        coverage["orientation_held_out"] += any(
            x["orientation"] not in train_orientations for x in test)
        coverage["size_varied"] += len({x["size"] for x in train}) > 1
        coverage["kind_varied"] += len(set(kinds)) > 1
        coverage["same_kind"] += len(set(kinds)) == 1
        coverage["test_scale_varied"] += len({x["scale"] for x in test}) > 1
        coverage["test_kind_varied"] += len({
            _position_kind(x["r"], x["c"]) for x in test}) > 1
        coverage["background_varied"] += len({x["background"] for x in train}) > 1
        coverage["source_shift"] += {2, 4}.issubset(train_scales) and all(
            x["scale"] == 5 and _position_kind(x["r"], x["c"]) == "top"
            and x["orientation"] == 0 for x in test)
        coverage["top_to_left"] += any(
            _position_kind(x["r"], x["c"]) == "top" for x in train
        ) and all(_position_kind(x["r"], x["c"]) == "left" for x in test)
        coverage["familiar_scale"] += any(x["scale"] in train_scales for x in test)
        coverage["interpolation"] += any(
            min(train_scales) < x["scale"] < max(train_scales)
            and x["scale"] not in train_scales for x in test)
        coverage["larger_scale"] += any(
            x["scale"] > max(train_scales) for x in test)
        in_colors = set(int(x) for x in np.unique(puzzle["train"][0]["input"]))
        out_colors = set(int(x) for x in np.unique(puzzle["train"][0]["output"]))
        coverage["marker_zero"] += out_colors - in_colors == {0}
        coverage["zero_seed_or_key"] += any(
            x["background"] != 0 and (x["seed_color"] == 0 or 0 in x["key"])
            for x in both)
        coverage["combined_corner"] += any(
            x["background"] != 0 and x["orientation"] != 0
            and _position_kind(x["r"], x["c"]) == "corner" for x in both)
    minimums = {
        "orientation_varied": 10, "orientation_fixed": 10,
        "orientation_held_out": 3, "size_varied": 5, "kind_varied": 20,
        "same_kind": 3, "test_scale_varied": 10, "test_kind_varied": 5,
        "background_varied": 42, "source_shift": 3, "top_to_left": 3,
        "familiar_scale": 10, "interpolation": 3, "larger_scale": 10,
        "marker_zero": 1, "zero_seed_or_key": 10, "combined_corner": 5,
    }
    assert all(coverage[name] >= lower for name, lower in minimums.items()), coverage
