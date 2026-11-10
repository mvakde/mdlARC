"""Compositional generators for vertical mirror-pair highlighting."""

import numpy as np


_SOURCE_INPUT_ROWS = {
    ("0000050000", "0500000055", "0005555000", "0000550000",
     "0000500000", "0050000500", "0000050000", "0050555000",
     "0500550050", "5000550005"),
    ("0000000000", "0000550000", "0000550000", "0005555000",
     "0000500500", "5000000005", "0000555000", "0505555000",
     "0005500000", "0000000000"),
    ("0000000000", "0500550050", "0005005000", "0050000500",
     "0005050000", "0000050000", "0000550000", "5000550005",
     "0000000000", "0005000005"),
    ("0050005000", "0000005000", "0000500000", "0055555500",
     "0005000500", "0000550000", "0000000000", "0000000000",
     "0000000000", "0000000000"),
    ("0500000050", "0000500000", "0000500000", "0500000500",
     "0000550000", "0000050000", "0505555050", "0000550000",
     "0005555500", "0055555000"),
}


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _pair_counts(seed, num_train, num_test):
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = num_test + 1
        num_train = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        minimum = num_test + 1
        num_train = minimum + seed % (11 - minimum)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _is_source_input(grid):
    return grid.shape == (10, 10) and tuple(
        "".join(str(int(x)) for x in row) for row in grid
    ) in _SOURCE_INPUT_ROWS


def _grid_key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _output(grid, foreground, highlight):
    result = grid.copy()
    mask = (grid == foreground) & (np.fliplr(grid) == foreground)
    result[mask] = highlight
    return result


def _states_to_grid(states, background, foreground):
    height, half = states.shape
    grid = np.full((height, 2 * half), background, dtype=np.int8)
    for distance in range(half):
        state = states[:, distance]
        grid[(state == 1) | (state == 2), distance] = foreground
        grid[(state == 1) | (state == 3), 2 * half - 1 - distance] = foreground
    return grid


def _row_features(occupied):
    height, width = occupied.shape
    half = width // 2
    mirror = np.fliplr(occupied)
    matched = occupied & mirror
    unmatched = occupied & ~mirror
    mixed = np.any(matched, axis=1) & np.any(unmatched, axis=1)
    center_stem = any(bool(np.all(matched[r:r + 3, half - 1:half + 1]))
                      for r in range(height - 2))
    centered_bar = any(bool(np.all(matched[r, half - 3:half + 3]))
                       for r in range(height) if half >= 3)
    attached = False
    for row in range(height):
        for distance in range(half - 1):
            if matched[row, distance + 1] and unmatched[row, distance]:
                attached = True
    return {
        "matched": int(np.sum(matched)),
        "unmatched": int(np.sum(unmatched)),
        "mixed_rows": int(np.sum(mixed)),
        "center_stem": center_stem,
        "centered_bar": centered_bar,
        "attached": attached,
        "lower_density": float(np.mean(occupied[-min(6, height):])),
        "density": float(np.mean(occupied)),
        "pair_distances": int(np.sum(np.any(matched[:, :half], axis=0))),
        "outer_only_rows": int(sum(np.sum(row) == 2 and row[0] and row[-1]
                                   for row in occupied)),
        "center_only_rows": int(sum(np.sum(row) == 2 and row[half - 1] and row[half]
                                    for row in occupied)),
        "unmatched_left": bool(np.any(unmatched[:, :half])),
        "unmatched_right": bool(np.any(unmatched[:, half:])),
    }


def _draw_states(rng, height, half, mode, complexity):
    """State 0 is empty, 1 is paired, and 2/3 are one-sided marks."""
    states = np.zeros((height, half), dtype=np.int8)
    paired_rows = rng.random(height) < rng.uniform(0.40, 0.78)
    for row in range(height):
        if rng.random() < rng.uniform(0.03, 0.20):
            continue
        density = rng.uniform(0.08, 0.38) * rng.uniform(0.45, 1.4)
        if complexity == "dense":
            density *= rng.uniform(1.15, 1.7)
        if mode == "matched_only" or (mode == "segregated" and paired_rows[row]):
            states[row] = np.where(rng.random(half) < density, 1, 0)
        elif mode in ("segregated", "unmatched_only"):
            side = rng.integers(2, 4, size=half)
            states[row] = np.where(rng.random(half) < density, side, 0)
        else:
            paired_share = rng.uniform(0.35, 0.78)
            choices = rng.choice(4, size=half,
                                 p=[1 - density, density * paired_share,
                                    density * (1 - paired_share) / 2,
                                    density * (1 - paired_share) / 2])
            states[row] = choices

    # Shape operations are independently located; there is no reserved test block.
    shape_count = int(rng.integers(2, 6 if complexity == "dense" else 5))
    shape_types = ["side"] if mode == "unmatched_only" else ["stem", "bar", "isolated", "side", "block"]
    for kind in rng.choice(shape_types, size=shape_count, replace=True):
        if kind == "stem":
            length = int(rng.integers(2, min(6, height) + 1))
            start = int(rng.integers(height - length + 1))
            distance = half - 1 if rng.random() < 0.8 else int(rng.integers(half))
            if mode == "segregated":
                states[start:start + length][(states[start:start + length] >= 2)] = 0
            states[start:start + length, distance] = 1
        elif kind == "bar":
            row = int(rng.integers(height))
            length = int(rng.integers(2, min(half, 5) + 1))
            if mode == "segregated":
                states[row, states[row] >= 2] = 0
            states[row, half - length:half] = 1
            if mode == "mixed" and length < half and rng.random() < 0.65:
                states[row, half - length - 1] = int(rng.integers(2, 4))
        elif kind == "isolated":
            row = int(rng.integers(height))
            states[row] = 0
            states[row, int(rng.integers(half))] = 1
        elif kind == "side" and mode != "matched_only":
            row = int(rng.integers(height))
            if mode == "segregated":
                states[row, states[row] == 1] = 0
            distance = half - 1 if rng.random() < 0.5 else int(rng.integers(half))
            states[row, distance] = int(rng.integers(2, 4))
        elif kind == "block":
            length = int(rng.integers(2, min(4, height) + 1))
            start = int(rng.integers(height - length + 1))
            width = int(rng.integers(1, min(3, half) + 1))
            if mode == "segregated":
                states[start:start + length][(states[start:start + length] >= 2)] = 0
            states[start:start + length, half - width:half] = 1

    # Supply both treatments even when sparse sampling missed one of them.
    if mode != "unmatched_only" and not np.any(states == 1):
        states[int(rng.integers(height)), int(rng.integers(half))] = 1
    if mode != "matched_only" and not np.any(states >= 2):
        row = int(rng.integers(height))
        if mode == "segregated":
            states[row, states[row] == 1] = 0
        states[row, int(rng.integers(half))] = int(rng.integers(2, 4))
    return states


def _make_input(rng, background, foreground, regime, mode, evidence=False):
    for _ in range(160):
        height = int(rng.integers(8, 20) if evidence else rng.integers(6, 22))
        half = int(rng.integers(4, 12) if evidence else rng.integers(3, 14))
        complexity = "dense" if rng.random() < (0.55 if regime == "target_test" else 0.30) else "ordinary"
        states = _draw_states(rng, height, half, mode, complexity)
        grid = _states_to_grid(states, background, foreground)
        occupied = grid == foreground
        features = _row_features(occupied)
        if not 0.045 <= features["density"] <= 0.47:
            continue
        if mode != "unmatched_only" and features["matched"] < (6 if evidence else 2):
            continue
        if mode != "matched_only" and features["unmatched"] < (3 if evidence else 1):
            continue
        if evidence and not (features["mixed_rows"] and features["unmatched_left"]
                             and features["unmatched_right"] and features["pair_distances"] >= 3):
            continue
        if mode == "segregated" and features["mixed_rows"]:
            continue
        if mode == "matched_only" and features["unmatched"]:
            continue
        if mode == "unmatched_only" and features["matched"]:
            continue
        if _is_source_input(grid):
            continue
        return grid
    raise RuntimeError("could not construct a valid mirror pattern")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train, num_test = _pair_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)

    # The output color is shared. In some puzzles, training teaches two or three
    # input palettes; tests reuse only palettes already seen in training.
    colors = [int(x) for x in rng.permutation(10)]
    highlight = colors[0]
    palette_count = min(num_train, int(rng.integers(2, 4))) if rng.random() < 0.38 else 1
    available = colors[1:]
    palettes = [(available[2 * i], available[2 * i + 1]) for i in range(palette_count)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train":
                palette_index = index % palette_count
                mode = "mixed" if index == 0 else str(rng.choice(
                    ["mixed", "segregated", "matched_only", "unmatched_only"],
                    p=[0.43, 0.38, 0.12, 0.07]))
            else:
                palette_index = int(rng.integers(palette_count))
                mode = "mixed"
            background, foreground = palettes[palette_index]
            for _ in range(20):
                child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                grid = _make_input(child, background, foreground,
                                   "target_test" if split == "test" else "source_train",
                                   mode, evidence=(split == "train" and index == 0))
                key = _grid_key(grid)
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not construct a unique input")
            seen.add(key)
            puzzle[split].append({"input": grid, "output": _output(grid, foreground, highlight)})
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_conventions(training):
    palettes = set()
    highlights = set()
    matched_distances = set()
    saw_left = saw_right = saw_mixed = False
    for pair in training:
        inp, out = pair["input"], pair["output"]
        colors, counts = np.unique(inp, return_counts=True)
        assert len(colors) == 2
        background = int(colors[np.argmax(counts)])
        foreground = int(colors[1 - np.argmax(counts)])
        palettes.add((background, foreground))
        changed = inp != out
        if np.any(changed):
            assert np.all(inp[changed] == foreground)
            assert len(np.unique(out[changed])) == 1
            highlights.add(int(out[changed][0]))
        occ = inp == foreground
        features = _row_features(occ)
        saw_left |= features["unmatched_left"]
        saw_right |= features["unmatched_right"]
        saw_mixed |= bool(features["mixed_rows"])
        match = occ & np.fliplr(occ)
        matched_distances.update(int(c) for c in np.where(np.any(match[:, :inp.shape[1] // 2], axis=0))[0])
    assert len(highlights) == 1
    assert saw_left and saw_right and saw_mixed and len(matched_distances) >= 3
    return palettes, next(iter(highlights))


def _predict_from_demonstrations(training, test_input):
    """Use only visible demonstrations and the test input."""
    palettes, highlight = _training_conventions(training)
    colors, counts = np.unique(test_input, return_counts=True)
    assert len(colors) == 2
    background = int(colors[np.argmax(counts)])
    foreground = int(colors[1 - np.argmax(counts)])
    assert (background, foreground) in palettes
    height, width = test_input.shape
    prediction = test_input.copy()
    for row in range(height):
        for col in range(width):
            if test_input[row, col] == foreground and test_input[row, width - 1 - col] == foreground:
                prediction[row, col] = highlight
    return prediction


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    palettes, highlight = _training_conventions(puzzle["train"])
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert inp.shape == out.shape and inp.shape[1] % 2 == 0
            colors, counts = np.unique(inp, return_counts=True)
            assert len(colors) == 2
            background = int(colors[np.argmax(counts)])
            foreground = int(colors[1 - np.argmax(counts)])
            assert (background, foreground) in palettes
            assert highlight not in (background, foreground)
            occupied = inp == foreground
            features = _row_features(occupied)
            assert 0.045 <= features["density"] <= 0.47
            assert features["matched"] >= 2 or (split == "train" and features["matched"] == 0
                                                and features["unmatched"] >= 1)
            if split == "test":
                assert features["unmatched"] >= 1
            assert np.array_equal(out, _output(inp, foreground, highlight))
            assert (not np.array_equal(inp, out) or
                    (split == "train" and features["matched"] == 0))
            assert not _is_source_input(inp)
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            if split == "test":
                assert np.array_equal(_predict_from_demonstrations(puzzle["train"], inp), out)


def _equal_puzzles(left, right):
    return all(np.array_equal(a[role], b[role])
               for split in ("train", "test")
               for a, b in zip(left[split], right[split])
               for role in ("input", "output")) and all(
                   len(left[split]) == len(right[split]) for split in ("train", "test"))


def self_test() -> None:
    assert _equal_puzzles(generate_puzzle(314159), generate_puzzle(314159))
    assert not _equal_puzzles(generate_puzzle(314159), generate_puzzle(314160))
    for seed, train, test in ((31, 2, 1), (32, 10, 1), (33, 10, 4),
                              (34, None, 1), (35, None, 4),
                              (36, 2, None), (37, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (2, 4)):
        try:
            generate_puzzle(22, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")

    counts_train, counts_test, shapes, layouts, palettes = set(), set(), set(), set(), set()
    no_mixed = all_matched = all_unmatched = stems = bars = attached = sparse_tests = dense_trains = 0
    varied_shapes = varied_density = varied_palette = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts_train.add(len(puzzle["train"]))
        counts_test.add(len(puzzle["test"]))
        pair_shapes, pair_density, pair_palettes = set(), set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                colors, color_counts = np.unique(inp, return_counts=True)
                bg = int(colors[np.argmax(color_counts)])
                fg = int(colors[1 - np.argmax(color_counts)])
                pair_palettes.add((bg, fg))
                pair_shapes.add(inp.shape)
                occupied = inp == fg
                feat = _row_features(occupied)
                pair_density.add(round(feat["density"], 2))
                shapes.add(inp.shape)
                layouts.add((inp.shape, np.packbits(occupied).tobytes()))
                assert np.array_equal(out != inp, occupied & np.fliplr(occupied))
                assert np.all(out[inp == bg] == bg)
                no_mixed += split == "train" and feat["mixed_rows"] == 0
                all_matched += split == "train" and feat["unmatched"] == 0
                all_unmatched += split == "train" and feat["matched"] == 0
                stems += feat["center_stem"]
                bars += feat["centered_bar"]
                attached += feat["attached"]
                sparse_tests += split == "test" and feat["lower_density"] < 0.38
                dense_trains += split == "train" and feat["density"] > 0.25
        palettes.update(pair_palettes)
        varied_shapes += len(pair_shapes) > 1
        varied_density += len(pair_density) > 1
        varied_palette += len(pair_palettes) > 1
    assert len(counts_train) >= 4 and len(counts_test) >= 3
    assert len(shapes) >= 25 and len(layouts) >= 150 and len(palettes) >= 20
    assert no_mixed >= 15 and all_matched >= 2 and all_unmatched >= 1
    assert stems >= 15 and bars >= 15 and attached >= 15
    assert sparse_tests >= 10 and dense_trains >= 5
    assert varied_shapes >= 35 and varied_density >= 35 and varied_palette >= 5
