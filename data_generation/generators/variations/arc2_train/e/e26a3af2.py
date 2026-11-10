"""Generate noisy full-span band reconstruction puzzles."""

import hashlib

import numpy as np


_SOURCE_HASHES = frozenset({
    "a3e4a2edfa9684a11c7a55a43939d159939a52f902a6918769d89c85623354d4",
    "631421a61c86050d401e1780d89a18bd3f584468b4a26d04bf976437ff2d7414",
    "417b8c4a32ba181d36c2b46559fffd089b42feee133f962eb18042e8a77adc46",
    "9ac9560bb40c7a006b7f1cdb1fb2099ad2878e94c0a0267e78879d0c15d0e447",
})


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _runs(values):
    colors, widths = [int(values[0])], [1]
    for value in values[1:]:
        value = int(value)
        if value == colors[-1]:
            widths[-1] += 1
        else:
            colors.append(value)
            widths.append(1)
    return tuple(colors), tuple(widths)


def _mode(line):
    counts = np.bincount(line, minlength=10)
    winner = int(np.argmax(counts))
    return winner if 2 * int(counts[winner]) > len(line) else None


def _infer(grid):
    candidates = []
    height, width = grid.shape
    for orientation in ("vertical", "horizontal"):
        lines = grid.T if orientation == "vertical" else grid
        modes = [_mode(line) for line in lines]
        if any(color is None for color in modes):
            continue
        colors, thicknesses = _runs(modes)
        if len(colors) < 2:
            continue
        vector = np.asarray(modes, dtype=np.int8)
        output = (np.repeat(vector[None, :], height, axis=0) if orientation == "vertical"
                  else np.repeat(vector[:, None], width, axis=1))
        candidates.append((output, orientation, colors, thicknesses))
    if len(candidates) != 1:
        raise AssertionError("band reconstruction is ambiguous")
    return candidates[0]


def _clean_info(grid):
    vertical = bool(np.all(grid == grid[:1, :]))
    horizontal = bool(np.all(grid == grid[:, :1]))
    assert vertical != horizontal
    orientation = "vertical" if vertical else "horizontal"
    colors, thicknesses = _runs(grid[0, :] if vertical else grid[:, 0])
    assert 2 <= len(colors) <= 8
    return orientation, colors, thicknesses


def _pair_hash(inp, out):
    height, width = inp.shape
    payload = bytes((height, width)) + inp.astype(np.uint8).tobytes() + out.astype(np.uint8).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _pair_info(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    predicted, orientation, colors, widths = _infer(inp)
    assert np.array_equal(predicted, out)
    assert (orientation, colors, widths) == _clean_info(out)
    changed = inp != out
    count = int(np.count_nonzero(changed))
    assert 0 < count <= int(np.ceil(inp.size * 0.25))
    assert _pair_hash(inp, out) not in _SOURCE_HASHES
    return {
        "orientation": orientation, "colors": colors, "widths": widths,
        "shape": inp.shape, "density": count / inp.size,
        "noise_only": frozenset(int(x) for x in np.unique(inp)) - set(colors),
        "max_row_errors": int(changed.sum(axis=1).max()),
        "max_col_errors": int(changed.sum(axis=0).max()),
    }


def _thicknesses(rng, bands):
    style = int(rng.choice(4, p=[0.48, 0.17, 0.22, 0.13]))
    if style == 3:
        each = int(rng.integers(1, 31 // bands + 1))
        return (each,) * bands
    total = int(rng.integers(max(8, bands + 1), 31))
    if style == 1 and total >= 2 * bands + 1:
        large = int(rng.integers(total // 2 + 1, total - bands + 2))
        rest = rng.multinomial(total - large - bands + 1, np.ones(bands - 1) / (bands - 1)) + 1
        values = [large] + [int(x) for x in rest]
    elif style == 2:
        rest = rng.multinomial(total - bands, np.ones(bands - 1) / (bands - 1)) + 1
        values = [1] + [int(x) for x in rest]
    else:
        weights = rng.gamma(float(rng.choice([0.5, 1.0, 2.0, 5.0])), 1, bands)
        values = [int(x) + 1 for x in rng.multinomial(total - bands, weights / weights.sum())]
    rng.shuffle(values)
    return tuple(values)


def _colors(rng, available, bands, forced=()):
    forced = [int(x) for x in forced]
    available = [int(x) for x in available]
    for _ in range(100):
        values = forced + [int(rng.choice(available)) for _ in range(bands - len(forced))]
        rng.shuffle(values)
        if all(a != b for a, b in zip(values, values[1:])):
            return np.asarray(values, dtype=np.int8)
    # Adjacent colors can always differ because at least two are available.
    values = forced[:]
    while len(values) < bands:
        choices = [x for x in available if not values or x != values[-1]]
        values.append(int(rng.choice(choices)))
    return np.asarray(values, dtype=np.int8)


def _noise(rng, clean, orientation, palette, density, required=()):
    height, width = clean.shape
    span = height if orientation == "vertical" else width
    line_count = width if orientation == "vertical" else height
    cap = (span - 1) // 2
    target = max(len(required), 1, int(round(clean.size * density)))
    target = min(target, line_count * cap)
    load = np.zeros(line_count, dtype=np.int16)
    positions = []
    for flat in rng.permutation(clean.size):
        row, col = divmod(int(flat), width)
        line = col if orientation == "vertical" else row
        if load[line] < cap:
            load[line] += 1
            positions.append((row, col))
            if len(positions) == target:
                break
    assert len(positions) == target
    noisy = clean.copy()
    clean_colors = set(int(x) for x in np.unique(clean))
    noise_only = [int(x) for x in palette if int(x) not in clean_colors]
    extra_count = int(rng.integers(0, len(noise_only) + 1))
    extra = [int(x) for x in rng.choice(noise_only, size=extra_count, replace=False)]
    assignments = [int(x) for x in required] + [x for x in extra if x not in required]
    replacement_palette = sorted(clean_colors | set(assignments))
    for index, (row, col) in enumerate(positions):
        original = int(clean[row, col])
        if index < len(assignments) and assignments[index] != original:
            replacement = assignments[index]
        else:
            choices = [x for x in replacement_palette if x != original]
            replacement = int(rng.choice(choices))
        noisy[row, col] = replacement
    return noisy


def _make_pair(rng, orientation, bands, palette, available, forced, density, required_noise=()):
    for _ in range(80):
        widths = _thicknesses(rng, bands)
        span = int(rng.integers(8, 31))
        colors = _colors(rng, available, bands, forced)
        vector = np.repeat(colors, widths)
        clean = (np.repeat(vector[None, :], span, axis=0) if orientation == "vertical"
                 else np.repeat(vector[:, None], span, axis=1)).astype(np.int8)
        pair = {"input": _noise(rng, clean, orientation, palette, density, required_noise),
                "output": clean}
        try:
            _pair_info(pair)
        except AssertionError:
            continue
        return pair
    raise RuntimeError("could not construct an unambiguous pair")


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    inputs, pairs = set(), set()
    train_orientations = set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for name in ("input", "output"):
                grid = pair[name]
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            inp, out = pair["input"], pair["output"]
            key = (inp.shape, inp.astype(np.uint8).tobytes())
            pair_key = _pair_hash(inp, out)
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            info = _pair_info(pair)
            if section == "train":
                train_orientations.add(info["orientation"])
    assert train_orientations == {"vertical", "horizontal"}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    # Color labels have no fixed meaning. The working palette is shared by
    # a puzzle; band and noise roles vary for each pair.
    palette = rng.permutation(np.arange(10, dtype=np.int8))[:int(rng.integers(6, 11))]
    promoted_count = int(rng.choice(3, p=[0.42, 0.34, 0.24]))
    promoted = tuple(int(x) for x in palette[:promoted_count])
    training_palette = palette[promoted_count:]
    expansion = bool(rng.random() < 0.30)
    orientations = ["vertical", "horizontal"] + [str(rng.choice(["vertical", "horizontal"])) for _ in range(train_count - 2)]
    rng.shuffle(orientations)
    puzzle = {"train": [], "test": []}
    seen = set()
    training_colors = set()
    for slot, orientation in enumerate(orientations):
        bands = int(rng.integers(2, 6 if expansion else 9))
        density = (float(rng.uniform(0.025, 0.08)) if slot == 0 else
                   float(rng.uniform(0.145, 0.205)) if slot == 1 else
                   float(rng.uniform(0.025, 0.215)))
        required = promoted if promoted and slot == 0 else ()
        for _ in range(80):
            pair = _make_pair(rng, orientation, bands, palette, training_palette, (), density, required)
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen:
                break
        else:
            raise RuntimeError("could not produce a unique training pair")
        puzzle["train"].append(pair)
        seen.add(key)
        training_colors.update(int(x) for x in np.unique(pair["output"]))
    for slot in range(test_count):
        orientation = str(rng.choice(["vertical", "horizontal"]))
        bands = int(rng.integers(6 if expansion and slot == 0 else 2, 9))
        density = float(rng.uniform(0.035, 0.215))
        if slot == 0:
            forced = promoted
        else:
            take = int(rng.integers(0, len(promoted) + 1))
            forced = tuple(int(x) for x in rng.choice(promoted, size=take, replace=False))
        test_palette = np.asarray(sorted(training_colors | set(forced)), dtype=np.int8)
        for _ in range(80):
            pair = _make_pair(rng, orientation, bands, palette, test_palette, forced, density)
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen:
                break
        else:
            raise RuntimeError("could not produce a unique test pair")
        puzzle["test"].append(pair)
        seen.add(key)
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    signatures, all_inputs, shapes, layouts, color_orders = set(), set(), set(), set(), set()
    densities, band_counts, test_orientations, novelties = [], set(), set(), set()
    one_cell = dominant = equal = high_line_noise = 0
    source_shapes = {(13, 14), (13, 17), (15, 14), (15, 15)}
    review_counts = {
        name: 0 for name in (
            "source_shape", "width_14_or_15", "odd_height", "odd_width",
            "row_7", "column_6", "noise_only_zero", "noise_only_several",
            "both_noise_types", "boundary_noise", "multiple_noisy_bands",
            "high_and_thin", "high_and_dominant", "test_adds_bands",
            "test_same_bands", "test_fewer_bands", "puzzle_shape_variety",
            "puzzle_palette_variety", "puzzle_density_variety",
        )
    }
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_pair_hash(q["input"], q["output"]) for s in ("train", "test") for q in puzzle[s]))
        training_colors = set(int(x) for q in puzzle["train"] for x in np.unique(q["output"]))
        novelties.add(max(len(set(int(x) for x in np.unique(q["output"])) - training_colors) for q in puzzle["test"]))
        training_bands = [len(_clean_info(pair["output"])[1]) for pair in puzzle["train"]]
        puzzle_shapes, puzzle_palettes, puzzle_densities = set(), set(), []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _pair_info(pair)
                inp, out = pair["input"], pair["output"]
                changed = inp != out
                input_palette = frozenset(int(x) for x in np.unique(inp))
                key = (pair["input"].shape, pair["input"].tobytes())
                assert key not in all_inputs
                all_inputs.add(key)
                shapes.add(info["shape"])
                layouts.add((info["orientation"], info["shape"], info["widths"]))
                color_orders.add((info["orientation"], info["colors"]))
                densities.append(info["density"])
                band_counts.add(len(info["colors"]))
                one_cell += min(info["widths"]) == 1
                dominant += max(info["widths"]) > sum(info["widths"]) / 2
                equal += len(set(info["widths"])) == 1
                high_line_noise += max(info["max_row_errors"], info["max_col_errors"]) >= 5
                puzzle_shapes.add(info["shape"])
                puzzle_palettes.add(input_palette)
                puzzle_densities.append(info["density"])
                review_counts["source_shape"] += info["shape"] in source_shapes
                review_counts["width_14_or_15"] += inp.shape[1] in (14, 15)
                review_counts["odd_height"] += inp.shape[0] % 2 == 1
                review_counts["odd_width"] += inp.shape[1] % 2 == 1
                review_counts["row_7"] += info["max_row_errors"] >= 7
                review_counts["column_6"] += info["max_col_errors"] >= 6
                review_counts["noise_only_zero"] += len(info["noise_only"]) == 0
                review_counts["noise_only_several"] += len(info["noise_only"]) >= 2
                changed_colors = set(int(x) for x in inp[changed])
                review_counts["both_noise_types"] += bool(changed_colors & set(info["colors"])) and bool(changed_colors - set(info["colors"]))
                boundaries = np.cumsum(info["widths"])[:-1]
                varying_positions = np.where(changed)[1 if info["orientation"] == "vertical" else 0]
                review_counts["boundary_noise"] += any(abs(int(position) - int(boundary)) <= 1 for position in varying_positions for boundary in boundaries)
                starts = np.cumsum((0,) + info["widths"])
                noisy_bands = sum(bool(np.any(changed[:, starts[i]:starts[i + 1]] if info["orientation"] == "vertical" else changed[starts[i]:starts[i + 1], :])) for i in range(len(info["widths"])))
                review_counts["multiple_noisy_bands"] += noisy_bands >= 2
                review_counts["high_and_thin"] += info["density"] >= .148 and min(info["widths"]) == 1
                review_counts["high_and_dominant"] += info["density"] >= .148 and max(info["widths"]) > sum(info["widths"]) / 2
                if section == "test":
                    test_orientations.add(info["orientation"])
                    review_counts["test_adds_bands"] += len(info["colors"]) > max(training_bands)
                    review_counts["test_same_bands"] += len(info["colors"]) in training_bands
                    review_counts["test_fewer_bands"] += len(info["colors"]) < max(training_bands)
        review_counts["puzzle_shape_variety"] += len(puzzle_shapes) > 1
        review_counts["puzzle_palette_variety"] += len(puzzle_palettes) > 1
        review_counts["puzzle_density_variety"] += max(puzzle_densities) - min(puzzle_densities) >= .05
    assert len(signatures) == 42 and len(shapes) >= 90 and len(layouts) >= 160
    assert len(color_orders) >= 140 and band_counts.issuperset({2, 3, 4, 5, 6, 7, 8})
    assert test_orientations == {"vertical", "horizontal"} and novelties.issuperset({0, 1, 2})
    assert min(densities) < 0.08 and max(densities) > 0.17
    assert min(one_cell, dominant, equal, high_line_noise) > 0
    assert all(count > 0 for count in review_counts.values())
    assert review_counts["puzzle_shape_variety"] >= 35
    assert review_counts["puzzle_palette_variety"] >= 35
    assert review_counts["puzzle_density_variety"] >= 35
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(9000 + i, train, test)
        assert (len(p["train"]), len(p["test"])) == (train, test)
    for i, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        validate_generated_puzzle(generate_puzzle(9100 + i, train, test))
    a, b, c = generate_puzzle(81234, 6, 2), generate_puzzle(81234, 6, 2), generate_puzzle(81235, 6, 2)
    signature = lambda p: tuple(_pair_hash(q["input"], q["output"]) for s in ("train", "test") for q in p[s])
    assert signature(a) == signature(b) and signature(a) != signature(c)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 2, "num_test": 3},
                   {"num_train": 3, "num_test": 4}, {"num_train": True}, {"num_train": 3.0}):
        try:
            generate_puzzle(9999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
