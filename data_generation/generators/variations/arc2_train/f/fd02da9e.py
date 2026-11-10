import numpy as np


UL, UR, LL, LR = range(4)
CORNERS = (UL, UR, LL, LR)
UPPER = (UL, UR)
LOWER = (LL, LR)
SOURCE_SPECS = (
    {LL: 1}, {UR: 8}, {LR: 0}, {UL: 9},
    {UL: 2, UR: 5, LL: 8, LR: 4},
)


def _corner_cell(corner, height, width):
    return (0 if corner in UPPER else height - 1,
            0 if corner in (UL, LL) else width - 1)


def _motif_cells(corner, height, width):
    if corner == UL:
        return ((1, 1), (1, 2), (2, 1), (2, 2))
    if corner == UR:
        return ((1, width - 3), (1, width - 2),
                (2, width - 3), (2, width - 2))
    if corner == LL:
        return ((height - 4, 2), (height - 3, 2), (height - 2, 3))
    if corner == LR:
        return ((height - 4, width - 3), (height - 3, width - 3),
                (height - 2, width - 4))
    raise ValueError("unknown corner")


def _layout_valid(corners, height, width):
    if not (5 <= height <= 30 and 5 <= width <= 30):
        return False
    marker_cells = {_corner_cell(c, height, width) for c in corners}
    occupied = set()
    for corner in corners:
        cells = _motif_cells(corner, height, width)
        if (any(not (0 <= r < height and 0 <= col < width)
                for r, col in cells)
                or occupied.intersection(cells)
                or marker_cells.intersection(cells)):
            return False
        occupied.update(cells)
    return True


def _make_input(height, width, background, markers):
    grid = np.full((height, width), background, dtype=np.int8)
    for corner, color in markers.items():
        grid[_corner_cell(corner, height, width)] = color
    return grid


def _extract_markers(grid, background):
    h, w = grid.shape
    return {c: int(grid[_corner_cell(c, h, w)]) for c in CORNERS
            if int(grid[_corner_cell(c, h, w)]) != background}


def _apply_rule(input_grid, background):
    h, w = input_grid.shape
    output = np.full((h, w), background, dtype=np.int8)
    for corner, color in _extract_markers(input_grid, background).items():
        for row, col in _motif_cells(corner, h, w):
            output[row, col] = color
    return output


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(v) for v in grid.flat))


def _source_input_keys():
    return {_grid_key(_make_input(8, 8, 7, spec)) for spec in SOURCE_SPECS}


def _checked_count(value, low, high, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train = _checked_count(num_train, 2, 10, "num_train")
    test = _checked_count(num_test, 1, 4, "num_test")
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _subsets(corners):
    corners = tuple(sorted(corners))
    return [tuple(corners[i] for i in range(len(corners)) if mask & (1 << i))
            for mask in range(1, 1 << len(corners))]


def _training_schedule(rng, count, classic, heldout):
    available = tuple(c for c in CORNERS if c != heldout)
    if classic:
        order = tuple(int(v) for v in rng.permutation(4))
        return [(order[i % 4],) for i in range(count)], available
    upper = int(rng.choice([c for c in available if c in UPPER]))
    lower = int(rng.choice([c for c in available if c in LOWER]))
    schedule = [(upper,), (lower,)]
    if count == 2:
        return schedule, (upper, lower)
    missing = [c for c in available if c not in (upper, lower)]
    addition = list(missing)
    for corner in (upper, lower):
        if int(rng.integers(0, 2)):
            addition.append(corner)
    if len(addition) == 1:
        addition.append(upper if int(rng.integers(0, 2)) else lower)
    schedule.append(tuple(sorted(addition)))
    candidates = [s for s in _subsets(available) if s not in schedule]
    rng.shuffle(candidates)
    while len(schedule) < count:
        multi = [s for s in candidates if len(s) > 1]
        pool = multi if multi and int(rng.integers(0, 5)) else candidates
        if pool:
            subset = pool[int(rng.integers(0, len(pool)))]
            candidates.remove(subset)
        else:
            choices = _subsets(available)
            subset = choices[int(rng.integers(0, len(choices)))]
        schedule.append(subset)
    return schedule, available


def _background(rng, base, varied, index):
    if varied and (index == 1 or int(rng.integers(0, 10)) < 4):
        return int(rng.choice([c for c in range(10) if c != base]))
    return base


def _colors(rng, corners, background, palette, repeat_chance):
    palette = [c for c in palette if c != background]
    if not palette:
        palette = [c for c in range(10) if c != background]
    if len(corners) >= 2 and int(rng.integers(0, 100)) < repeat_chance:
        colors = [int(rng.choice(palette))] * len(corners)
        if len(corners) > 2 and int(rng.integers(0, 2)):
            colors[int(rng.integers(0, len(corners)))] = int(rng.choice(palette))
    else:
        colors = [int(rng.choice(palette)) for _ in corners]
    return dict(zip(corners, colors))


def _dimensions(rng, corners):
    buckets = (
        ((5, 9), (5, 9)),
        ((10, 22), (5, 11)),
        ((5, 11), (10, 24)),
        ((7, 16), (8, 18)),
    )
    mode = int(rng.choice(4, p=[0.30, 0.18, 0.18, 0.34]))
    hr, wr = buckets[mode]
    valid = [(h, w) for h in range(hr[0], hr[1] + 1)
             for w in range(wr[0], wr[1] + 1)
             if _layout_valid(corners, h, w)]
    return valid[int(rng.integers(0, len(valid)))]


def _pair(rng, corners, background, palette, repeat_chance, used_inputs,
          used_specs, preferred=None):
    for attempt in range(120):
        choices = palette if attempt < 30 else tuple(range(10))
        markers = (dict(preferred) if attempt == 0 and preferred is not None
                   else _colors(rng, corners, background, choices, repeat_chance))
        spec = tuple(sorted(markers.items()))
        if spec in used_specs:
            continue
        h, w = _dimensions(rng, corners)
        input_grid = _make_input(h, w, background, markers)
        key = _grid_key(input_grid)
        if key in used_inputs:
            continue
        output = _apply_rule(input_grid, background)
        used_inputs.add(key)
        used_specs.add(spec)
        return {"input": input_grid, "output": output}
    raise RuntimeError("could not construct a unique corner pair")


def _inspect_input(grid):
    h, w = grid.shape
    background = int(grid[0, 1])
    markers = _extract_markers(grid, background)
    assert markers
    assert np.array_equal(grid, _make_input(h, w, background, markers))
    return background, markers


def _template(corner, output, background):
    h, w = output.shape
    return frozenset((int(r) if corner in UPPER else h - 1 - int(r),
                      int(c) if corner in (UL, LL) else w - 1 - int(c))
                     for r, c in np.argwhere(output != background))


def _predict_from_training(train, test_input):
    # Infer inward offsets from singleton examples of each vertical class.
    templates = {}
    for pair in train:
        background, markers = _inspect_input(pair["input"])
        if len(markers) == 1:
            corner = next(iter(markers))
            cls = corner in UPPER
            shape = _template(corner, pair["output"], background)
            if cls in templates:
                assert templates[cls] == shape
            templates[cls] = shape
    assert set(templates) == {True, False}
    background, markers = _inspect_input(test_input)
    h, w = test_input.shape
    prediction = np.full((h, w), background, dtype=np.int8)
    for corner, color in markers.items():
        for depth, inward in templates[corner in UPPER]:
            row = depth if corner in UPPER else h - 1 - depth
            col = inward if corner in (UL, LL) else w - 1 - inward
            prediction[row, col] = color
    return prediction


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    # Puzzle-level sampling preferences do not change the transformation.
    base = int(rng.integers(0, 10))
    varied_background = bool(rng.integers(0, 10) < 7)
    familiar = tuple(int(v) for v in rng.choice(
        10, size=int(rng.integers(3, 7)), replace=False))
    classic = train_count >= 4 and bool(rng.integers(0, 5) == 0)
    heldout = (int(rng.integers(0, 4))
               if train_count >= 3 and not classic and
               int(rng.integers(0, 4)) == 0 else None)
    schedule, trained = _training_schedule(rng, train_count, classic, heldout)
    used_inputs, used_specs = _source_input_keys(), set()
    train = []
    common_color = (int(rng.choice([c for c in familiar if c != base]))
                    if int(rng.integers(0, 2)) else None)
    for index, corners in enumerate(schedule):
        bg = _background(rng, base, varied_background, index)
        preferred = ({corners[0]: common_color}
                     if common_color is not None and index < 2 and
                     common_color != bg else None)
        train.append(_pair(rng, corners, bg, familiar, 32, used_inputs,
                           used_specs, preferred))
    seen = {color for pair in train
            for color in _inspect_input(pair["input"])[1].values()}
    test, test_subsets = [], set()
    for index in range(test_count):
        if classic and index == 0:
            corners = CORNERS
        else:
            candidates = _subsets(CORNERS if heldout is not None else trained)
            if heldout is not None and index == 0:
                candidates = [s for s in candidates if heldout in s]
            unused = [s for s in candidates if s not in test_subsets]
            corners = (unused or candidates)[int(rng.integers(0, len(unused or candidates)))]
        test_subsets.add(corners)
        bg = _background(rng, base, varied_background, train_count + index)
        mode = int(rng.integers(0, 4))
        novel = tuple(c for c in range(10) if c not in seen)
        if mode == 0:
            palette = tuple(seen)
        elif mode == 1:
            palette = novel
        elif mode == 2:
            palette = tuple(seen) + novel[:max(1, len(novel) // 2)]
        else:
            palette = tuple(range(10))
        test.append(_pair(rng, corners, bg, palette, 38, used_inputs, used_specs))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used_inputs, used_pairs, used_specs = set(), set(), set()
    source_inputs = _source_input_keys()
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape
            assert inp.shape[0] >= 5 and inp.shape[1] >= 5
            background, markers = _inspect_input(inp)
            assert _layout_valid(markers, *inp.shape)
            assert all(c != background for c in markers.values())
            assert np.array_equal(out, _apply_rule(inp, background))
            assert not np.array_equal(inp, out)
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            marker_spec = tuple(sorted(markers.items()))
            assert input_key not in source_inputs and input_key not in used_inputs
            assert pair_key not in used_pairs
            assert marker_spec not in used_specs
            used_inputs.add(input_key)
            used_pairs.add(pair_key)
            used_specs.add(marker_spec)
            records[split].append((background, markers))
    train_corners = {c for _, markers in records["train"] for c in markers}
    singles = {c in UPPER for _, markers in records["train"]
               if len(markers) == 1 for c in markers}
    assert singles == {True, False}
    for _, markers in records["test"]:
        for corner in markers:
            if corner not in train_corners:
                assert len(train_corners) == 3
                other = LOWER if corner in UPPER else UPPER
                assert set(other).issubset(train_corners)
    for pair in puzzle["test"]:
        assert np.array_equal(
            _predict_from_training(puzzle["train"], pair["input"]),
            pair["output"])


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({_puzzle_key(p) for p in puzzles}) >= 40
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    for seed, counts in ((500, (2, 1)), (501, (10, 1)), (502, (10, 4)),
                         (600, (None, 1)), (601, (None, 4)),
                         (602, (2, None)), (603, (10, None))):
        puzzle = generate_puzzle(seed, *counts)
        assert counts[0] is None or len(puzzle["train"]) == counts[0]
        assert counts[1] is None or len(puzzle["test"]) == counts[1]
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(700, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed, counts in ((31415, (None, None)), (27182, (7, 3)),
                         (16180, (None, 4))):
        assert _puzzle_key(generate_puzzle(seed, *counts)) == _puzzle_key(
            generate_puzzle(seed, *counts))
    backgrounds, shapes = set(), set()
    subsets = {"train": set(), "test": set()}
    counts = {"train": set(), "test": set()}
    background_changes = cross_color = heldouts = repeated_test = 0
    compact = aspect = classic = all_seen = zero_seen = combined = 0
    reuse_counts = set()
    for puzzle in puzzles:
        train = [_inspect_input(p["input"]) for p in puzzle["train"]]
        test = [_inspect_input(p["input"]) for p in puzzle["test"]]
        trained = {c for _, markers in train for c in markers}
        seen = {color for _, markers in train for color in markers.values()}
        classes = {}
        for _, markers in train:
            for corner, color in markers.items():
                classes.setdefault(color, set()).add(corner in UPPER)
        cross_color += any(len(v) == 2 for v in classes.values())
        background_changes += len({bg for bg, _ in train + test}) > 1
        combined += (len({bg for bg, _ in train + test}) > 1 and
                     any(len(m) > 1 for _, m in train) and
                     any(len(m) > 1 for _, m in test))
        classic += (all(len(m) == 1 for _, m in train) and
                    any(len(m) == 4 for _, m in test))
        for split, records in (("train", train), ("test", test)):
            for (bg, markers), pair in zip(records, puzzle[split]):
                backgrounds.add(bg)
                shape = pair["input"].shape
                shapes.add(shape)
                subsets[split].add(tuple(sorted(markers)))
                counts[split].add(len(markers))
                compact += shape[0] < 7 or shape[1] < 8
                aspect += max(shape) / min(shape) >= 2
                if split == "test":
                    heldouts += bool(set(markers) - trained)
                    repeated_test += len(set(markers.values())) < len(markers)
                    reused = sum(color in seen for color in markers.values())
                    reuse_counts.add(reused)
                    all_seen += reused == len(markers)
                    zero_seen += reused == 0
    assert len(backgrounds) >= 7 and len(shapes) >= 25
    assert counts["train"] == {1, 2, 3, 4}
    assert counts["test"] == {1, 2, 3, 4}
    assert len(subsets["train"]) >= 10 and len(subsets["test"]) >= 10
    assert {(UL, UR), (LL, LR)}.issubset(subsets["test"])
    assert background_changes >= 10 and cross_color >= 8
    assert heldouts >= 2 and repeated_test >= 2
    assert {0, 1, 2}.issubset(reuse_counts)
    assert all_seen >= 5 and zero_seen >= 5 and combined >= 5
    assert compact >= 10 and aspect >= 10 and classic >= 1
