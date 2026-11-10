"""Unary run length and ordered palette ARC puzzle generator."""

import numpy as np


_SOURCE_SPECS = frozenset({
    (12, 8, 4, (2, 3), 0, 5),
    (7, 7, 3, (9, 8), 0, 5),
    (9, 7, 2, (9, 6, 7), 0, 5),
    (14, 6, 1, (2, 8, 4), 0, 5),
    (5, 5, 1, (3, 1), 0, 5),
    (9, 9, 2, (4, 8, 3), 0, 5),
})


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(rng, num_train, num_test):
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
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


def _orient(grid, code):
    turned = np.rot90(grid, code % 4)
    if code >= 4:
        turned = np.fliplr(turned)
    return turned.copy()


def _undo_orient(grid, code):
    if code >= 4:
        grid = np.fliplr(grid)
    return np.rot90(grid, -(code % 4)).copy()


def _palette(rng, size, background, marker):
    colors = [c for c in range(10) if c != background and c != marker]
    return tuple(int(c) for c in rng.choice(colors, size=size, replace=False))


def _roles(rng, core_palette, total, mode):
    pool = [c for c in range(10) if c not in core_palette]
    first = tuple(int(c) for c in rng.choice(pool, size=2, replace=False))
    result = [first]
    for index in range(1, total):
        if mode == "fixed":
            result.append(first)
            continue
        if mode == "background":
            backgrounds = [c for c in pool if c != first[1] and (index != 1 or c != first[0])]
            result.append((int(rng.choice(backgrounds)), first[1]))
        elif mode == "marker":
            markers = [c for c in pool if c != first[0] and (index != 1 or c != first[1])]
            result.append((first[0], int(rng.choice(markers))))
        else:
            candidates = [(b, m) for b in pool for m in pool
                          if b != m and (index != 1 or (b != first[0] and m != first[1]))]
            result.append(candidates[int(rng.integers(len(candidates)))])
    return result


def _height_options(k, size, regime, stop):
    cycle = k * size
    low = max(cycle + 1, 10 if regime == "target_test" else 5)
    high = 30 if regime == "target_test" else 26
    options = list(range(low, high + 1))
    if stop == "cycle":
        return [h for h in options if h % cycle == 0]
    if stop == "run":
        return [h for h in options if h % k == 0 and h % cycle != 0]
    if stop == "partial":
        return [h for h in options if h % k != 0]
    return options


def _build_pair(height, width, k, palette, background, marker, orientation):
    size = len(palette)
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    inp[:k, 0] = marker
    out[:k, 0] = marker
    for offset, color in enumerate(palette):
        inp[:, width - size + offset] = color
    out[:, width - size - 1] = [palette[(r // k) % size] for r in range(height)]
    return {"input": _orient(inp, orientation), "output": _orient(out, orientation)}


def _grid_key(grid):
    return (grid.shape, grid.astype(np.uint8, copy=False).tobytes())


def _choose_layout(rng, regime, k, palette, roles, orientation, stop,
                   preferred_width, used_inputs):
    size = len(palette)
    stops = [stop] + [x for x in ("cycle", "run", "partial", "any") if x != stop]
    options = next((_height_options(k, size, regime, s) for s in stops
                    if _height_options(k, size, regime, s)), [])
    assert options
    high_width = 18 if regime == "source_train" else 26
    widths = list(range(size + 3, high_width + 1))
    widths = [int(x) for x in rng.permutation(widths)]
    if preferred_width in widths:
        widths.remove(preferred_width)
        widths.insert(0, preferred_width)
    if regime == "target_test" and rng.random() < 0.5:
        long = [h for h in options if h >= 22]
        if long:
            options = long
    heights = [int(x) for x in rng.permutation(options)]
    for height in heights:
        for width in widths:
            pair = _build_pair(height, width, k, palette, roles[0], roles[1], orientation)
            source_spec = (height, width, k, palette, roles[0], roles[1])
            if orientation == 0 and source_spec in _SOURCE_SPECS:
                continue
            key = _grid_key(pair["input"])
            if key not in used_inputs:
                used_inputs.add(key)
                return pair, width
    raise AssertionError("no unique layout in the bounded construction space")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _counts(rng, num_train, num_test)
    total = num_train + num_test

    # The direction and role stability are puzzle-level conventions. In the
    # varying cases the training set explicitly demonstrates the variation.
    base_orientation = int(rng.integers(8))
    mixed = num_train >= 5 and bool(rng.integers(2))
    orientations = [base_orientation] * total
    if mixed:
        alternatives = [x for x in range(8) if x != base_orientation]
        orientations[4] = int(rng.choice(alternatives))
        for i in range(5, num_train):
            orientations[i] = int(rng.integers(8))
        shown = sorted(set(orientations[:num_train]))
        for i in range(num_train, total):
            orientations[i] = int(rng.choice(shown))

    base_k = int(rng.integers(1, 4))
    base_size = int(rng.integers(2, 4))
    core_palette = tuple(int(c) for c in rng.choice(10, size=base_size, replace=False))
    mode = ("fixed", "background", "marker", "both")[int(rng.integers(4))]
    roles = _roles(rng, core_palette, total, mode)

    specs = [(base_k, core_palette)]
    if num_train == 2 and bool(rng.integers(2)):
        other_size = int(rng.choice([n for n in (2, 3, 4) if n != base_size]))
        specs.append((base_k, _palette(rng, other_size, *roles[1])))
    else:
        other_k = int(rng.choice([k for k in range(1, 6) if k != base_k]))
        specs.append((other_k, core_palette))
    if num_train >= 3:
        other_size = int(rng.choice([n for n in (2, 3, 4) if n != base_size]))
        specs.append((base_k, _palette(rng, other_size, *roles[2])))
    if num_train >= 4:
        specs.append((base_k, tuple(reversed(core_palette))))
    while len(specs) < num_train:
        index = len(specs)
        size = int(rng.integers(2, 5))
        specs.append((int(rng.integers(1, 6)), _palette(rng, size, *roles[index])))

    observed_combos = [(k, len(p)) for k, p in specs]
    anchored_ks = sorted({k for k, _ in observed_combos[:min(3, num_train)]})
    anchored_sizes = sorted({n for _, n in observed_combos[:min(3, num_train)]})
    for index in range(num_test):
        role = roles[num_train + index]
        if num_train <= 3:
            k, size = observed_combos[int(rng.integers(len(observed_combos)))]
        else:
            k = int(rng.choice(anchored_ks))
            size = int(rng.choice(anchored_sizes))
        specs.append((k, _palette(rng, size, *role)))

    used_inputs = set()
    train, test = [], []
    first_width = int(rng.integers(8, 19))
    for index, (k, palette) in enumerate(specs):
        regime = "source_train" if index < num_train else "target_test"
        if index == 0:
            stop = "cycle"
        elif index == 1:
            stop = "partial" if k > 1 else "run"
        elif index == 2:
            stop = "run"
        else:
            stop = ("cycle", "run", "partial")[int(rng.integers(3))]
        preferred = first_width if index < 2 else None
        pair, width = _choose_layout(rng, regime, k, palette, roles[index],
                                     orientations[index], stop, preferred, used_inputs)
        if index == 0:
            first_width = width
        (train if regime == "source_train" else test).append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _canonical_input(inp, background):
    height, width = inp.shape
    palette_reversed = []
    column = width - 1
    while column >= 0:
        values = inp[:, column]
        if np.all(values == values[0]) and int(values[0]) != background:
            palette_reversed.append(int(values[0]))
            column -= 1
        else:
            break
    palette = tuple(reversed(palette_reversed))
    size = len(palette)
    assert 2 <= size <= 4 and len(set(palette)) == size
    assert width >= size + 3
    marker = int(inp[0, 0])
    assert marker != background and marker not in palette and background not in palette
    k = 0
    while k < height and int(inp[k, 0]) == marker:
        k += 1
    assert 1 <= k <= 5 and height >= k * size + 1
    expected = np.full(inp.shape, background, dtype=np.int8)
    expected[:k, 0] = marker
    for offset, color in enumerate(palette):
        expected[:, width - size + offset] = color
    assert np.array_equal(inp, expected)
    out = np.full(inp.shape, background, dtype=np.int8)
    out[:k, 0] = marker
    out[:, width - size - 1] = [palette[(r // k) % size] for r in range(height)]
    return {"height": height, "width": width, "k": k, "palette": palette,
            "size": size, "background": background, "marker": marker,
            "gap": width - size - 2, "stop": ("cycle" if height % (k * size) == 0
                                                else "run" if height % k == 0 else "partial")}, out


def _infer(inp):
    counts = np.bincount(inp.ravel().astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    assert int(np.count_nonzero(counts == counts[background])) == 1
    possibilities = []
    for code in range(8):
        canonical = _orient(inp, code)
        try:
            info, answer = _canonical_input(canonical, background)
        except (AssertionError, IndexError):
            continue
        info["orientation"] = code
        possibilities.append((info, _undo_orient(answer, code)))
    assert len(possibilities) == 1
    return possibilities[0]


def _prediction_from_training(train_pairs, test_input):
    for pair in train_pairs:
        _, predicted = _infer(pair["input"])
        assert np.array_equal(predicted, pair["output"])
    return _infer(test_input)[1]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    inputs, pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert grid.size and 0 <= int(grid.min()) <= int(grid.max()) <= 9
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            input_key, output_key = _grid_key(inp), _grid_key(out)
            assert input_key not in inputs and (input_key, output_key) not in pairs
            inputs.add(input_key)
            pairs.add((input_key, output_key))
            info, expected = _infer(inp)
            assert np.array_equal(out, expected)
            if split == "train":
                assert info["height"] <= 26 and info["width"] <= 18
            else:
                assert info["height"] >= 10 and info["width"] <= 26
            source_spec = (info["height"], info["width"], info["k"],
                           info["palette"], info["background"], info["marker"])
            assert not (info["orientation"] == 0 and source_spec in _SOURCE_SPECS)
            infos[split].append(info)

    train, test = infos["train"], infos["test"]
    assert train[0]["stop"] == "cycle"
    assert len({(x["k"], x["size"]) for x in train}) >= 2
    if ntrain == 2 and train[0]["k"] == train[1]["k"]:
        assert train[0]["size"] != train[1]["size"]
    else:
        assert train[0]["palette"] == train[1]["palette"]
        assert train[0]["k"] != train[1]["k"]
    if ntrain >= 3:
        assert train[0]["k"] == train[2]["k"] and train[0]["size"] != train[2]["size"]
    if ntrain >= 4:
        assert train[3]["k"] == train[0]["k"]
        assert train[3]["palette"] == tuple(reversed(train[0]["palette"]))
    for role in ("background", "marker", "orientation"):
        if len({x[role] for x in train}) == 1:
            assert all(x[role] == train[0][role] for x in test)
    seen_combos = {(x["k"], x["size"]) for x in train}
    seen_k = {x["k"] for x in train}
    seen_sizes = {x["size"] for x in train}
    for item in test:
        assert item["k"] in seen_k and item["size"] in seen_sizes
        if ntrain <= 3:
            assert (item["k"], item["size"]) in seen_combos
        assert item["orientation"] in {x["orientation"] for x in train}
    for pair in puzzle["test"]:
        assert np.array_equal(_prediction_from_training(puzzle["train"], pair["input"]),
                              pair["output"])


def _puzzle_key(puzzle):
    return tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    train_counts, test_counts = set(), set()
    puzzles, grids, shapes, layouts, palettes, roles = set(), set(), set(), set(), set(), set()
    ks, sizes, stops, orientations, gaps = set(), set(), set(), set(), set()
    test_combos, test_heights = set(), set()
    mixed_orientation = changed_roles = same_palette_contrast = size_contrast = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzles.add(_puzzle_key(puzzle))
        train_info = [_infer(p["input"])[0] for p in puzzle["train"]]
        all_info = train_info + [_infer(p["input"])[0] for p in puzzle["test"]]
        mixed_orientation += len({x["orientation"] for x in train_info}) > 1
        changed_roles += len({(x["background"], x["marker"]) for x in train_info}) > 1
        same_palette_contrast += (train_info[0]["palette"] == train_info[1]["palette"]
                                  and train_info[0]["k"] != train_info[1]["k"])
        size_contrast += any(x["k"] == train_info[0]["k"] and
                             x["size"] != train_info[0]["size"] for x in train_info[1:])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grids.add(_grid_key(pair["input"]))
        for x in all_info:
            ks.add(x["k"]); sizes.add(x["size"]); stops.add(x["stop"])
            orientations.add(x["orientation"]); gaps.add(x["gap"])
            palettes.add(x["palette"]); roles.add((x["background"], x["marker"]))
            shapes.add((x["height"], x["width"]))
            layouts.add((x["height"], x["width"], x["k"], x["size"],
                         x["orientation"], x["stop"]))
        for x in all_info[len(train_info):]:
            test_combos.add((x["k"], x["size"]))
            test_heights.add(x["height"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzles) == 42 and len(grids) >= 200
    assert len(shapes) >= 45 and len(layouts) >= 120 and len(palettes) >= 60
    assert len(roles) >= 20 and ks == {1, 2, 3, 4, 5} and sizes == {2, 3, 4}
    assert stops == {"cycle", "run", "partial"} and len(orientations) == 8
    assert len(gaps) >= 10 and len(test_combos) >= 8 and len(test_heights) >= 12
    assert mixed_orientation >= 3 and changed_roles >= 8
    assert same_palette_contrast >= 10 and size_contrast >= 10
    a, b = generate_puzzle(777, 6, 3), generate_puzzle(777, 6, 3)
    assert _puzzle_key(a) == _puzzle_key(b)
    for seed, nt, nv in ((100, 2, 1), (101, 10, 1), (102, 10, 4), (103, 6, 2)):
        puzzle = generate_puzzle(seed, nt, nv)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, nv)
        validate_generated_puzzle(puzzle)
    for seed, kwargs in ((200, {"num_test": 1}), (201, {"num_test": 4}),
                         (202, {"num_train": 2}), (203, {"num_train": 10})):
        puzzle = generate_puzzle(seed, **kwargs)
        validate_generated_puzzle(puzzle)
        for name, value in kwargs.items():
            assert len(puzzle["train" if name == "num_train" else "test"]) == value
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4},
                   {"num_train": 3.0, "num_test": 1}):
        try:
            generate_puzzle(300, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid count accepted: {kwargs}")
