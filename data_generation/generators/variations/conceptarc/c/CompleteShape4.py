"""Aligned rectangle sides: replace the supplied sides with the missing sides."""

import numpy as np


# (height, width, orientation, foreground, start, end, first, second)
_SOURCE_SIGNATURES = {
    (10, 10, "horizontal", 4, 2, 7, 2, 8),
    (10, 10, "vertical", 6, 1, 6, 2, 8),
    (10, 10, "horizontal", 3, 1, 8, 1, 6),
    (12, 10, "vertical", 2, 0, 11, 2, 9),
    (6, 6, "vertical", 5, 1, 3, 1, 3),
}


def _resolve_counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError("Pair counts must be integers")
            if not low <= value <= high:
                raise ValueError("Pair count out of range")
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("Training count must exceed test count")

    if num_train is None:
        choices = np.array((2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4,
                            5, 5, 6, 7, 8, 9, 10))
        choices = choices[choices > (num_test if num_test is not None else 0)]
        num_train = int(choices[int(rng.integers(len(choices)))])
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return int(num_train), int(num_test)


def _draw_pair(height, width, orientation, background, color, start, length, first, second):
    inp = np.full((height, width), background, dtype=np.int8)
    out = np.full((height, width), background, dtype=np.int8)
    if orientation == "horizontal":
        inp[first, start:start + length] = color
        inp[second, start:start + length] = color
        out[first + 1:second, start] = color
        out[first + 1:second, start + length - 1] = color
    else:
        inp[start:start + length, first] = color
        inp[start:start + length, second] = color
        out[start, first + 1:second] = color
        out[start + length - 1, first + 1:second] = color
    return {"input": inp, "output": out}


def _choose_extent(rng, regime):
    quiet = rng.random() < (0.23 if regime == "source_train" else 0.10)
    if quiet:
        return 10, 10, True
    compact = rng.random() < (0.13 if regime == "source_train" else 0.18)
    if compact:
        return int(rng.integers(5, 8)), int(rng.integers(5, 8)), False
    if regime == "source_train":
        # Independent row and column draws permit squares and both aspect directions.
        height = int(rng.integers(5, 20))
        width = int(rng.integers(5, 20))
    else:
        height = int(rng.integers(5, 23))
        width = int(rng.integers(5, 23))
    return height, width, False


def _choose_geometry(rng, height, width, orientation, quiet, regime):
    along = width if orientation == "horizontal" else height
    across = height if orientation == "horizontal" else width
    lengths = range(4, min(7, along - 2) + 1) if quiet else range(3, along + 1)
    separations = range(3, across - 2) if quiet else range(2, across)
    possibilities = {
        "shorter": [], "equal": [], "longer": [],
    }
    for length in lengths:
        for separation in separations:
            if quiet and (along - length < 2 or across - separation < 3):
                continue
            new_length = separation - 1
            relation = "shorter" if new_length < length else "equal" if new_length == length else "longer"
            possibilities[relation].append((length, separation))
    labels = [label for label in ("shorter", "equal", "longer") if possibilities[label]]
    weights = np.array([{"shorter": 0.39, "equal": 0.25, "longer": 0.36}[label] for label in labels])
    weights /= weights.sum()
    gap_one = not quiet and rng.random() < (0.14 if regime == "source_train" else 0.19)
    label = "shorter" if gap_one else labels[int(rng.choice(len(labels), p=weights))]
    options = possibilities[label]
    if gap_one:
        options = [item for item in options if item[1] == 2]
    # Select a length band first so long segments are not swamped by the
    # larger number of short-length combinations.
    bands = (
        [item for item in options if item[0] <= 5],
        [item for item in options if 6 <= item[0] <= 9],
        [item for item in options if item[0] >= 10],
    )
    nonempty = [band for band in bands if band]
    band = nonempty[int(rng.integers(len(nonempty)))]
    length, separation = band[int(rng.integers(len(band)))]

    if quiet:
        start = int(rng.integers(1, along - length))
        first = int(rng.integers(1, across - separation - 1))
    else:
        start = int(rng.integers(along - length + 1))
        first = int(rng.integers(across - separation))
    return start, length, first, first + separation


def _sample_pair(rng, regime, orientation):
    height, width, quiet = _choose_extent(rng, regime)
    start, length, first, second = _choose_geometry(rng, height, width, orientation, quiet, regime)
    background = int(rng.integers(10))
    choices = [color for color in range(10) if color != background]
    color = choices[int(rng.integers(len(choices)))]
    return _draw_pair(height, width, orientation, background, color, start, length, first, second)


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 2
    return int(colors[int(np.argmax(counts))])


def _geometry(inp):
    background = _background(inp)
    positions = np.argwhere(inp != background)
    foreground = np.unique(inp[inp != background])
    assert len(foreground) == 1
    color = int(foreground[0])
    rows, cols = np.unique(positions[:, 0]), np.unique(positions[:, 1])
    height, width = inp.shape
    if len(rows) == 2 and len(cols) >= 3:
        orientation = "horizontal"
        start, end = int(cols[0]), int(cols[-1])
        first, second = int(rows[0]), int(rows[-1])
    elif len(cols) == 2 and len(rows) >= 3:
        orientation = "vertical"
        start, end = int(rows[0]), int(rows[-1])
        first, second = int(cols[0]), int(cols[-1])
    else:
        raise AssertionError("Input is not two aligned segments")
    assert second - first >= 2
    assert end - start + 1 >= 3
    expected = np.full(inp.shape, background, dtype=inp.dtype)
    if orientation == "horizontal":
        expected[first, start:end + 1] = color
        expected[second, start:end + 1] = color
    else:
        expected[start:end + 1, first] = color
        expected[start:end + 1, second] = color
    assert np.array_equal(inp, expected)
    return (height, width, orientation, background, color, start, end, first, second)


def _expected_output(inp, geometry):
    _, _, orientation, background, color, start, end, first, second = geometry
    out = np.full(inp.shape, background, dtype=inp.dtype)
    if orientation == "horizontal":
        out[first + 1:second, start] = color
        out[first + 1:second, end] = color
    else:
        out[start, first + 1:second] = color
        out[end, first + 1:second] = color
    return out


def _grid_key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _unique_pair(rng, seen_inputs, seen_layouts, regime, orientation, index):
    for _ in range(32):
        pair = _sample_pair(rng, regime, orientation)
        geometry = _geometry(pair["input"])
        layout = geometry[:3] + geometry[5:]
        if (_grid_key(pair["input"]) not in seen_inputs
                and layout not in seen_layouts
                and not (geometry[3] == 0 and geometry[:3] + geometry[4:] in _SOURCE_SIGNATURES)):
            seen_inputs.add(_grid_key(pair["input"]))
            seen_layouts.add(layout)
            return pair
    # Distinct 25-row canvases give a deterministic finite fallback.
    height, width = 25, 15 + index
    background, color = index % 10, (index + 1) % 10
    pair = _draw_pair(height, width, orientation, background, color, 1, 3, 1, 3)
    geometry = _geometry(pair["input"])
    layout = geometry[:3] + geometry[5:]
    assert _grid_key(pair["input"]) not in seen_inputs and layout not in seen_layouts
    seen_inputs.add(_grid_key(pair["input"]))
    seen_layouts.add(layout)
    return pair


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # The source has no hidden palette or direction convention. The only
    # puzzle-level choice is which demonstrated orientations will be tested.
    repertoire = int(rng.choice(3, p=(0.22, 0.22, 0.56)))
    if repertoire == 0:
        train_orientations = ["horizontal"] * train_count
    elif repertoire == 1:
        train_orientations = ["vertical"] * train_count
    else:
        train_orientations = ["horizontal", "vertical"]
        train_orientations += [
            ("horizontal", "vertical")[int(rng.integers(2))]
            for _ in range(train_count - 2)
        ]
        rng.shuffle(train_orientations)
    demonstrated = sorted(set(train_orientations))
    test_orientations = [demonstrated[int(rng.integers(len(demonstrated)))] for _ in range(test_count)]
    seen_inputs, seen_layouts = set(), set()
    train = [_unique_pair(rng, seen_inputs, seen_layouts, "source_train", orientation, i)
             for i, orientation in enumerate(train_orientations)]
    test = [_unique_pair(rng, seen_inputs, seen_layouts, "target_test", orientation, train_count + i)
            for i, orientation in enumerate(test_orientations)]
    return {"train": train, "test": test}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs, seen_layouts = set(), set(), set()
    orientations = {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            geometry = _geometry(inp)
            assert np.array_equal(out, _expected_output(inp, geometry))
            assert not np.array_equal(inp, out)
            assert geometry[3] == _background(out)
            assert not (geometry[3] == 0 and geometry[:3] + geometry[4:] in _SOURCE_SIGNATURES)
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            layout = geometry[:3] + geometry[5:]
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert layout not in seen_layouts
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            seen_layouts.add(layout)
            orientations[split].add(geometry[2])
    assert orientations["test"] <= orientations["train"]


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a[key], b[key])
        for a, b in zip(first[split], second[split]) for key in ("input", "output")
    ) for split in ("train", "test"))


def self_test() -> None:
    counts, sizes, layouts, colors, relations, orientation_profiles = set(), set(), set(), set(), set(), set()
    train_counts, test_counts = set(), set()
    axes = ("small", "rect", "gap1", "edge", "quiet", "asymmetric",
            "shorter", "equal", "longer", "large_rect_gap1_nonfull",
            "short_edge_large", "compact_wide_gap", "foreground_zero")
    evidence = {f"{split}_{axis}": 0 for split in ("train", "test") for axis in axes}
    evidence.update({name: 0 for name in (
        "reused_color", "changed_background", "all_horizontal", "all_vertical",
        "mixed_training", "mixed_test", "non_alternating_train", "short_demo",
        "train_dimensions_vary", "train_gap_varies", "repeated_gap1",
        "train_and_test_gap1", "color_both_orientations")})
    puzzle_keys = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        train_colors = set()
        test_colors = set()
        backgrounds = set()
        foreground_orientations = {}
        split_geometries = {"train": [], "test": []}
        key = []
        profile = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                height, width, orientation, bg, fg, start, end, first, second = _geometry(inp)
                assert np.array_equal(out, _expected_output(inp, _geometry(inp)))
                assert second - first >= 2 and end - start + 1 >= 3
                sizes.add((height, width))
                layouts.add((height, width, orientation, start, end, first, second))
                colors.add((bg, fg))
                backgrounds.add(bg)
                profile.append(orientation)
                foreground_orientations.setdefault(fg, set()).add(orientation)
                split_geometries[split].append((height, width, orientation, bg, fg,
                                                start, end, first, second))
                input_length, new_length = end - start + 1, second - first - 1
                relation = "shorter" if new_length < input_length else "equal" if new_length == input_length else "longer"
                relations.add(relation)
                evidence[f"{split}_{relation}"] += 1
                key.append((_grid_key(inp), _grid_key(out)))
                along = width if orientation == "horizontal" else height
                across = height if orientation == "horizontal" else width
                edge = start == 0 or end == along - 1 or first == 0 or second == across - 1
                evidence[f"{split}_small"] += max(height, width) <= 7
                evidence[f"{split}_rect"] += height != width
                evidence[f"{split}_gap1"] += second - first == 2
                evidence[f"{split}_edge"] += edge
                evidence[f"{split}_quiet"] += height == width == 10 and not edge and second - first >= 3
                evidence[f"{split}_asymmetric"] += (start != along - 1 - end or first != across - 1 - second)
                evidence[f"{split}_large_rect_gap1_nonfull"] += (max(height, width) >= 14 and height != width
                                                                  and second - first == 2 and input_length < along)
                evidence[f"{split}_short_edge_large"] += input_length <= 5 and max(height, width) >= 12 and edge
                evidence[f"{split}_compact_wide_gap"] += input_length <= 5 and second - first >= 4
                evidence[f"{split}_foreground_zero"] += fg == 0
                if split == "train":
                    train_colors.add(fg)
                else:
                    test_colors.add(fg)
        orientation_profiles.add(tuple(profile))
        train_geometries, test_geometries = split_geometries["train"], split_geometries["test"]
        train_orientations = [g[2] for g in train_geometries]
        test_orientations = [g[2] for g in test_geometries]
        all_orientations = set(train_orientations + test_orientations)
        evidence["all_horizontal"] += all_orientations == {"horizontal"}
        evidence["all_vertical"] += all_orientations == {"vertical"}
        evidence["mixed_training"] += len(set(train_orientations)) == 2
        evidence["mixed_test"] += len(set(test_orientations)) == 2
        evidence["non_alternating_train"] += any(a == b for a, b in zip(train_orientations, train_orientations[1:]))
        evidence["short_demo"] += len(puzzle["train"]) <= 3
        evidence["train_dimensions_vary"] += len({g[:2] for g in train_geometries}) > 1
        evidence["train_gap_varies"] += len({g[8] - g[7] for g in train_geometries}) > 1
        evidence["repeated_gap1"] += sum(g[8] - g[7] == 2 for g in train_geometries + test_geometries) >= 2
        evidence["train_and_test_gap1"] += (any(g[8] - g[7] == 2 for g in train_geometries)
                                            and any(g[8] - g[7] == 2 for g in test_geometries))
        evidence["color_both_orientations"] += any(len(value) == 2 for value in foreground_orientations.values())
        evidence["reused_color"] += bool(train_colors & test_colors)
        evidence["changed_background"] += len(backgrounds) > 1
        puzzle_keys.add(tuple(key))
    assert len(train_counts) >= 4 and len(test_counts) >= 3 and len(counts) >= 8
    assert len(puzzle_keys) >= 40 and len(sizes) >= 35 and len(layouts) >= 150
    assert len(colors) >= 35 and len(orientation_profiles) >= 8
    assert relations == {"shorter", "equal", "longer"}
    assert all(value > 0 for value in evidence.values())
    assert evidence["train_small"] >= 5 and evidence["test_small"] >= 5
    assert evidence["train_gap1"] >= 10 and evidence["test_gap1"] >= 5
    assert evidence["short_demo"] >= 8
    for requested_train, requested_test in ((2, 1), (10, 1), (10, 4), (5, 2),
                                            (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(501, requested_train, requested_test)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(501, requested_train, requested_test))
        assert requested_train is None or len(puzzle["train"]) == requested_train
        assert requested_test is None or len(puzzle["test"]) == requested_test
    for requested_train, requested_test in ((1, None), (11, None), (None, 0), (None, 5),
                                            (2, 2), (3, 4), (4, 4), (2.5, 1), (True, 1)):
        try:
            generate_puzzle(0, requested_train, requested_test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal pair counts were accepted")
