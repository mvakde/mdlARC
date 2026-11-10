"""Generators for paired-border line completion puzzles."""

import hashlib

import numpy as np


_SOURCE_INPUT_HASHES = frozenset({
    "ec8a80375968ef212e639fbcb72c49492d44bf5d2b19f0df5eb2c27b66b25b7d",
    "52503af1ff99d03f11c55b0bdf2c2d3bef844e625d97a152d5e1f69d4da9ef5e",
    "94d4720fbf82559a8551bd60c231206af1b4276348bd0c4f43025782b4e9c04c",
    "c8a963d37f1091240520b14d0e286b97c2671138d16832f96db58da8ee08feab",
})
_SOURCE_SHAPES = ((13, 20), (11, 22), (15, 17), (16, 19))


def _integer_or_none(name, value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    num_train = _integer_or_none("num_train", num_train)
    num_test = _integer_or_none("num_test", num_test)
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        minimum = max(2, num_test + 1)
        num_train = minimum + seed % (11 - minimum)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _shuffle(rng, values):
    return [values[int(i)] for i in rng.permutation(len(values))]


def _sample_conventions(rng):
    colors = [int(x) for x in rng.permutation(10)]
    new_color, former_color, steady_background = colors[:3]
    steady = bool(rng.random() < 0.28)
    marker_pool = colors[3:]
    first_markers = marker_pool[:2]
    active = tuple(first_markers + [former_color] + marker_pool[2:4])
    return {
        "new": new_color,
        "former": former_color,
        "steady": steady,
        "background": steady_background,
        "marker_pool": marker_pool,
        "first_markers": first_markers,
        "active": active,
    }


def _sample_colors(rng, conventions, split, index, test_mode):
    if split == "train":
        if index < 2:
            marker = conventions["first_markers"][index]
        elif conventions["steady"]:
            marker = _choice(
                rng, tuple(c for c in conventions["active"]
                           if c != conventions["former"]))
        else:
            marker = _choice(rng, conventions["marker_pool"])
    elif test_mode == "new":
        marker = conventions["new"]
    elif test_mode == "former":
        marker = conventions["former"]
    else:
        marker = _choice(rng, conventions["first_markers"])

    if conventions["steady"]:
        background = conventions["background"]
        distractors = tuple(c for c in conventions["active"] if c != marker)
    else:
        excluded_backgrounds = {marker}
        if split == "train":
            excluded_backgrounds.update((conventions["new"], conventions["former"]))
        background = _choice(rng, [c for c in range(10) if c not in excluded_backgrounds])
        choices = [c for c in range(10)
                   if c not in (background, marker, conventions["new"] if split == "train" else -1)]
        if split == "train" and index == 0:
            choices.remove(conventions["former"])
            count = int(rng.integers(1, 5))
            distractors = (conventions["former"],) + tuple(
                _shuffle(rng, choices)[:count - 1])
        else:
            count = int(rng.integers(1, 5))
            distractors = tuple(_shuffle(rng, choices)[:count])
    return background, marker, distractors


def _sample_dimensions(rng, regime):
    if rng.random() < 0.13:
        return _choice(rng, _SOURCE_SHAPES)
    # Independently sample height and width: neither aspect nor test size encodes the rule.
    ranges = ((7, 12), (11, 20), (20, 30))
    weights = (0.26, 0.48, 0.26) if regime == "source_train" else (0.25, 0.39, 0.36)
    h_range = ranges[int(rng.choice(3, p=weights))]
    w_range = ranges[int(rng.choice(3, p=weights))]
    return (int(rng.integers(h_range[0], h_range[1] + 1)),
            int(rng.integers(w_range[0], w_range[1] + 1)))


def _sample_layout(rng, regime):
    labels = ("single_h", "single_v", "double_h", "double_v", "cross", "many")
    probabilities = ((0.12, 0.12, 0.18, 0.18, 0.23, 0.17)
                     if regime == "source_train" else
                     (0.10, 0.10, 0.14, 0.14, 0.22, 0.30))
    return str(rng.choice(labels, p=probabilities))


def _line_counts(rng, layout):
    if layout == "single_h":
        return 1, 0
    if layout == "single_v":
        return 0, 1
    if layout == "double_h":
        return 2, 0
    if layout == "double_v":
        return 0, 2
    if layout == "cross":
        return 1, 1
    if layout == "many":
        choices = ((3, 0), (0, 3), (2, 1), (1, 2), (2, 2),
                   (3, 1), (1, 3), (3, 2), (2, 3))
        return _choice(rng, choices)
    raise ValueError("unknown layout")


def _sample_pair(rng, regime, layout, colors, force_decoy, force_line_clutter):
    height, width = _sample_dimensions(rng, regime)
    background, marker, distractors = colors
    row_count, column_count = _line_counts(rng, layout)
    rows = tuple(sorted(int(x) for x in rng.choice(
        np.arange(1, height - 1), size=row_count, replace=False)))
    columns = tuple(sorted(int(x) for x in rng.choice(
        np.arange(1, width - 1), size=column_count, replace=False)))

    inp = np.full((height, width), background, dtype=np.int8)
    out = np.full((height, width), background, dtype=np.int8)
    line = np.zeros((height, width), dtype=bool)
    endpoint = np.zeros((height, width), dtype=bool)
    for row in rows:
        line[row, :] = True
        endpoint[row, (0, width - 1)] = True
        out[row, :] = marker
    for column in columns:
        line[:, column] = True
        endpoint[(0, height - 1), column] = True
        out[:, column] = marker
    inp[endpoint] = marker

    free = [(r, c) for r in range(height) for c in range(width)
            if not endpoint[r, c]]
    interior = [(r, c) for r in range(1, height - 1)
                for c in range(1, width - 1)]

    def put(position, color):
        r, c = position
        if not endpoint[r, c] and inp[r, c] == background:
            inp[r, c] = color

    # Every distractor has an interior occurrence; the genuine marker has only
    # paired endpoints. This makes even an aligned distractor pair decidable.
    for color, position in zip(distractors, _shuffle(rng, interior)):
        put(position, color)

    if force_decoy or rng.random() < (0.37 if regime == "source_train" else 0.47):
        decoy_rows = [r for r in range(1, height - 1) if r not in rows
                      and inp[r, 0] == background and inp[r, -1] == background]
        decoy_columns = [c for c in range(1, width - 1) if c not in columns
                         and inp[0, c] == background and inp[-1, c] == background]
        options = [("row", r) for r in decoy_rows] + [("column", c) for c in decoy_columns]
        if options:
            direction, location = _choice(rng, options)
            color = _choice(rng, distractors)
            if direction == "row":
                put((location, 0), color)
                put((location, width - 1), color)
            else:
                put((0, location), color)
                put((height - 1, location), color)

    if rows and columns and rng.random() < 0.30:
        put((_choice(rng, rows), _choice(rng, columns)), _choice(rng, distractors))

    if force_line_clutter or rng.random() < (0.70 if regime == "source_train" else 0.80):
        line_positions = [p for p in free if line[p] and inp[p] == background]
        for position in _shuffle(rng, line_positions)[:int(rng.integers(1, 5))]:
            put(position, _choice(rng, distractors))

    density = float(rng.uniform(0.025, 0.21))
    target = max(int(round(height * width * density)),
                 len(distractors) + 2)
    target = min(target, len(free))
    boundary_heavy = bool(rng.random() < 0.24)
    boundary_fraction = float(rng.uniform(0.27, 0.62) if boundary_heavy
                              else rng.uniform(0.05, 0.23))
    boundary_target = min(int(round(target * boundary_fraction)),
                          2 * height + 2 * width - 4 - int(endpoint.sum()))
    boundary = [p for p in free if p[0] in (0, height - 1) or
                p[1] in (0, width - 1)]

    def noise_count():
        return int(np.count_nonzero((inp != background) & (inp != marker)))

    def boundary_noise_count():
        return sum(inp[p] != background for p in boundary)

    for position in _shuffle(rng, boundary):
        if boundary_noise_count() >= boundary_target or noise_count() >= target:
            break
        put(position, _choice(rng, distractors))

    while noise_count() < target:
        available = [p for p in free if inp[p] == background]
        if not available:
            break
        if rng.random() < 0.48:
            occupied = [p for p in free if inp[p] not in (background, marker)]
            near = [p for p in available if any(
                abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1 for q in occupied)]
            if near:
                available = near
        put(_choice(rng, available), _choice(rng, distractors))
    return inp, out


def _input_key(grid):
    return tuple(grid.shape), grid.tobytes()


def _source_hash(grid):
    return hashlib.sha256(bytes(grid.shape) + grid.tobytes()).hexdigest()


def _paired_geometry(grid, color):
    h, w = grid.shape
    rows = tuple(r for r in range(1, h - 1)
                 if grid[r, 0] == color and grid[r, w - 1] == color)
    columns = tuple(c for c in range(1, w - 1)
                    if grid[0, c] == color and grid[h - 1, c] == color)
    endpoints = {(r, c) for r in rows for c in (0, w - 1)}
    endpoints.update((r, c) for c in columns for r in (0, h - 1))
    return rows, columns, endpoints


def _inspect_pair(inp, out):
    h, w = inp.shape
    counts = np.bincount(inp.ravel(), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > max(counts[c] for c in range(10) if c != background)
    others = [int(c) for c in np.unique(out) if c != background]
    assert len(others) == 1
    marker = others[0]
    rows, columns, endpoints = _paired_geometry(inp, marker)
    assert rows or columns
    actual = {tuple(int(v) for v in p) for p in np.argwhere(inp == marker)}
    assert actual == endpoints

    pure_candidates = []
    decoys = []
    for color in (int(c) for c in np.unique(inp) if c != background):
        cr, cc, ep = _paired_geometry(inp, color)
        if cr or cc:
            locations = {tuple(int(v) for v in p) for p in np.argwhere(inp == color)}
            if locations == ep:
                pure_candidates.append(color)
            else:
                decoys.append(color)
    assert pure_candidates == [marker]
    for color in (int(c) for c in np.unique(inp) if c not in (background, marker)):
        assert np.any(inp[1:-1, 1:-1] == color)
    expected = np.full((h, w), background, dtype=np.int8)
    for row in rows:
        expected[row, :] = marker
    for column in columns:
        expected[:, column] = marker
    assert np.array_equal(out, expected)
    assert not np.array_equal(inp, out)
    assert len(np.unique(inp)) >= 3
    return {
        "shape": (h, w), "background": background, "marker": marker,
        "rows": rows, "columns": columns, "decoy": bool(decoys),
        "line_clutter": int(sum(inp[r, c] not in (background, marker)
                                for r in range(h) for c in range(w)
                                if r in rows or c in columns)),
        "cross_clutter": any(inp[r, c] not in (background, marker)
                             for r in rows for c in columns),
        "distractors": {int(c) for c in np.unique(inp)
                        if c not in (background, marker)},
        "density": float(np.count_nonzero(
            (inp != background) & (inp != marker))) / (h * w),
    }


def _predict_from_training(train, test_input):
    # The output of each demonstration identifies the endpoint-only color.
    for pair in train:
        _inspect_pair(pair["input"], pair["output"])
    counts = np.bincount(test_input.ravel(), minlength=10)
    background = int(np.argmax(counts))
    candidates = []
    for color in (int(c) for c in np.unique(test_input) if c != background):
        rows, columns, endpoints = _paired_geometry(test_input, color)
        locations = {tuple(int(v) for v in p)
                     for p in np.argwhere(test_input == color)}
        if (rows or columns) and locations == endpoints:
            candidates.append((color, rows, columns))
    assert len(candidates) == 1
    marker, rows, columns = candidates[0]
    prediction = np.full(test_input.shape, background, dtype=np.int8)
    for row in rows:
        prediction[row, :] = marker
    for column in columns:
        prediction[:, column] = marker
    return prediction


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs = set()
    pairs = set()
    structures = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert grid.dtype.kind in "iu" and grid.dtype.kind != "b"
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            key = _input_key(inp)
            pair_key = (key, _input_key(out))
            assert key not in inputs and pair_key not in pairs
            assert _source_hash(inp) not in _SOURCE_INPUT_HASHES
            inputs.add(key)
            pairs.add(pair_key)
            analysis = _inspect_pair(inp, out)
            structure = (analysis["shape"], analysis["rows"], analysis["columns"])
            assert structure not in structures
            structures.add(structure)
            analyses[split].append(analysis)
    training = analyses["train"]
    assert any(a["rows"] and a["columns"] for a in training)
    assert any(len(a["rows"]) >= 2 or len(a["columns"]) >= 2 for a in training)
    assert any(a["decoy"] for a in training)
    assert any(a["line_clutter"] > 0 for a in training)
    assert len({a["marker"] for a in training}) >= 2
    for pair in puzzle["test"]:
        assert np.array_equal(
            _predict_from_training(puzzle["train"], pair["input"]), pair["output"])


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
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    conventions = _sample_conventions(rng)
    training_layouts = _shuffle(rng, ["cross", _choice(rng, ["double_h", "double_v"])])
    training_layouts += [_sample_layout(rng, "source_train") for _ in range(num_train - 2)]
    if conventions["steady"]:
        modes = _shuffle(rng, ["reused", "former"])
    else:
        modes = _shuffle(rng, ["new", "reused", "former"])
    puzzle = {"train": [], "test": []}
    used = set()
    used_structures = set()
    for split, count in (("train", num_train), ("test", num_test)):
        regime = "source_train" if split == "train" else "target_test"
        for index in range(count):
            layout = training_layouts[index] if split == "train" else _sample_layout(rng, regime)
            mode = None if split == "train" else modes[index % len(modes)]
            colors = _sample_colors(rng, conventions, split, index, mode)
            for _ in range(20):
                pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                inp, out = _sample_pair(
                    np.random.default_rng(pair_seed), regime, layout, colors,
                    split == "train" and index == 0,
                    split == "train" and index == 0)
                key = _input_key(inp)
                marker = colors[1]
                rows, columns, _ = _paired_geometry(inp, marker)
                structure = (inp.shape, rows, columns)
                if key not in used and structure not in used_structures:
                    used.add(key)
                    used_structures.add(structure)
                    puzzle[split].append({"input": inp, "output": out})
                    break
            else:
                raise RuntimeError("could not make a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(left, right):
    return all(np.array_equal(a[field], b[field])
               for split in ("train", "test")
               for a, b in zip(left[split], right[split])
               for field in ("input", "output")) and all(
                   len(left[split]) == len(right[split]) for split in ("train", "test"))


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(8675309, 7, 3),
                          generate_puzzle(8675309, 7, 3))
    assert not _puzzles_equal(generate_puzzle(1001, 4, 2),
                              generate_puzzle(1002, 4, 2))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (6, 3)):
        puzzle = generate_puzzle(5000 + train_count * 10 + test_count,
                                 train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        validate_generated_puzzle(puzzle)
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(6000 + (train_count or 0) + (test_count or 0),
                                 train_count, test_count)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(7000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")

    train_counts, test_counts = set(), set()
    shapes, layouts, backgrounds, markers = set(), set(), set(), set()
    decoys = clutter = crossings = dense = sparse = small = large = 0
    train_decoys = test_decoys = boundary_heavy = clustered = 0
    decoy_clutter = decoy_dense = 0
    wide = tall = square = 0
    within_layout = within_background = within_marker = within_palette = 0
    test_smaller = test_equal = test_larger = 0
    source_shape = 0
    test_layouts = set()
    marker_modes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pa = {split: [_inspect_pair(p["input"], p["output"])
                      for p in puzzle[split]] for split in ("train", "test")}
        all_a = pa["train"] + pa["test"]
        within_layout += len({(len(a["rows"]), len(a["columns"]))
                              for a in all_a}) > 1
        within_background += len({a["background"] for a in all_a}) > 1
        within_marker += len({a["marker"] for a in all_a}) > 1
        palettes = {
            tuple(int(c) for c in np.unique(pair["input"])
                  if c != analysis["background"])
            for split in ("train", "test")
            for pair, analysis in zip(puzzle[split], pa[split])
        }
        within_palette += len(palettes) > 1
        train_heights = [a["shape"][0] for a in pa["train"]]
        training_markers = {a["marker"] for a in pa["train"]}
        training_distractors = set().union(*(a["distractors"] for a in pa["train"]))
        for a in pa["test"]:
            test_smaller += a["shape"][0] < min(train_heights)
            test_equal += a["shape"][0] in train_heights
            test_larger += a["shape"][0] > max(train_heights)
            test_layouts.add((len(a["rows"]), len(a["columns"])))
            marker_modes.add("reused" if a["marker"] in training_markers
                             else "former" if a["marker"] in training_distractors
                             else "new")
        for split in ("train", "test"):
          for pair, a in zip(puzzle[split], pa[split]):
            shapes.add(a["shape"])
            layouts.add((len(a["rows"]), len(a["columns"])))
            backgrounds.add(a["background"])
            markers.add(a["marker"])
            decoys += a["decoy"]
            train_decoys += split == "train" and a["decoy"]
            test_decoys += split == "test" and a["decoy"]
            clutter += a["line_clutter"] > 0
            crossings += a["cross_clutter"]
            dense += a["density"] >= 0.14
            sparse += a["density"] <= 0.06
            h, w = a["shape"]
            small += h <= 12 and w <= 12
            large += h >= 20 and w >= 20
            wide += w / h >= 1.5
            tall += h / w >= 1.5
            square += 0.85 <= h / w <= 1.18
            source_shape += a["shape"] in _SOURCE_SHAPES
            decoy_clutter += a["decoy"] and a["line_clutter"] > 0
            decoy_dense += a["decoy"] and a["density"] >= 0.14
            inp = pair["input"]
            distractor = (inp != a["background"]) & (inp != a["marker"])
            boundary_count = (
                int(distractor[0, :].sum()) + int(distractor[-1, :].sum())
                + int(distractor[1:-1, 0].sum())
                + int(distractor[1:-1, -1].sum())
            )
            boundary_heavy += boundary_count / int(distractor.sum()) >= 0.30
            adjacent_count = (
                int(np.logical_and(distractor[:-1, :], distractor[1:, :]).sum())
                + int(np.logical_and(distractor[:, :-1], distractor[:, 1:]).sum())
            )
            clustered += adjacent_count >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 70 and len(layouts) >= 8
    assert len(backgrounds) >= 8 and len(markers) >= 8
    assert within_layout == 42 and within_marker == 42
    assert 15 <= within_background <= 35 and 15 <= within_palette <= 35
    assert marker_modes == {"new", "former", "reused"}
    assert len(test_layouts) >= 8
    assert decoys >= 25 and train_decoys >= 15 and test_decoys >= 8
    assert clutter >= 70 and crossings >= 8
    assert dense >= 25 and sparse >= 15 and small >= 10 and large >= 10
    assert wide >= 20 and tall >= 20 and square >= 20
    assert boundary_heavy >= 20 and clustered >= 30
    assert decoy_clutter >= 20 and decoy_dense >= 10
    assert test_smaller >= 4 and test_equal >= 2 and test_larger >= 4
    assert source_shape >= 10
