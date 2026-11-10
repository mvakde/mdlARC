import numpy as np


_SOURCE_INPUT_FINGERPRINTS = {
    (14, 14, (8, 8, 2, 2, 2, 2, 2, 6, 8, 8, 1, 1, 2, 2)),
    (13, 12, (2, 3, 2, 2, 3, 2, 1, 2, 2, 2, 2, 8, 8)),
    (14, 11, (3, 3, 8, 8, 8, 2, 8, 8, 3, 7, 7, 5, 8, 8)),
}


def _validate_requested_count(value, name, lower, upper):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not lower <= int(value) <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")


def _resolve_counts(rng, num_train, num_test):
    _validate_requested_count(num_train, "num_train", 2, 10)
    _validate_requested_count(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(
            rng.integers(max(2, test_count + 1), 11)
        )
    elif num_train is None:
        test_count = int(num_test)
        lowest_train = max(2, test_count + 1)
        if lowest_train > 10:
            raise ValueError(
                "no legal num_train is larger than num_test"
            )
        train_count = int(rng.integers(lowest_train, 11))
    elif num_test is None:
        train_count = int(num_train)
        highest_test = min(4, train_count - 1)
        if highest_test < 1:
            raise ValueError(
                "no legal num_test is smaller than num_train"
            )
        test_count = int(rng.integers(1, highest_test + 1))
    else:
        train_count = int(num_train)
        test_count = int(num_test)

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _choose(rng, values):
    values = list(values)
    return int(values[int(rng.integers(len(values)))])


def _sample_height(rng):
    # Typical stacks remain common, but every legal length has real mass.
    return int(rng.integers(5, 16) if rng.random() < 0.55 else rng.integers(2, 17))


def _sample_width(rng, height, excluded=()):
    mode = int(rng.integers(6))
    preferred = ([height], [1], range(24, 31), range(4, 25),
                 range(1, 31), range(1, 31))[mode]
    choices = [w for w in preferred if w not in excluded]
    return _choose(rng, choices or [w for w in range(1, 31) if w not in excluded])


def _sample_palette(rng, allowed, complexity):
    values = np.asarray(list(allowed), dtype=int)
    # Complexity changes weights, never the set of attainable palettes.
    weights = (1.0 / values, np.ones(len(values)), values.astype(float))[complexity]
    return int(rng.choice(values, p=weights / weights.sum()))


def _run_labels(rng, palette, count):
    """Cover all colors, then insert extra visits at arbitrary valid gaps."""
    labels = [int(c) for c in rng.permutation(palette)]
    if len(labels) == 1:
        assert count == 1
        return labels
    for _ in range(count - len(labels)):
        options = [(i, int(c)) for i in range(len(labels) + 1) for c in palette
                   if (i == 0 or labels[i - 1] != c)
                   and (i == len(labels) or labels[i] != c)]
        i, color = options[int(rng.integers(len(options)))]
        labels.insert(i, color)
    return labels


def _sample_row_colors(rng, height, palette_size, complexity,
                       center_mode=None, demonstrate_repeat=False):
    """Compose runs and thicknesses across the complete code, including center."""
    length = height - 1
    palette = rng.choice(10, size=palette_size, replace=False)
    if palette_size == 1:
        return np.full(height, palette[0], dtype=np.int8)

    broad_bands = (center_mode is None and palette_size == 2 and length >= 4
                   and rng.random() < 0.5)
    if broad_bands:
        center_mode = "free"
    if center_mode is None:
        center_mode = ("new", "return", "free")[int(rng.integers(3))]
    if center_mode == "return" and palette_size == length:
        center_mode = "new"

    minimum = palette_size + (center_mode == "return")
    maximum = length - int(demonstrate_repeat)
    if center_mode == "new" and palette_size == 2:
        maximum = 2  # A one-color prefix has exactly one run.
    assert minimum <= maximum
    style = int(rng.choice(3, p=((0.60, 0.15, 0.25),
                                 (0.25, 0.25, 0.50),
                                 (0.15, 0.40, 0.45))[complexity]))
    run_count = (minimum if style == 0 else maximum if style == 1
                 else int(rng.integers(minimum, maximum + 1)))
    if broad_bands:
        run_count = 2

    if center_mode == "new":
        labels = _run_labels(rng, palette[:-1], run_count - 1) + [int(palette[-1])]
    elif center_mode == "return":
        labels = _run_labels(rng, palette, run_count - 1)
        choices = [int(c) for c in palette if c != labels[-1]]
        center = labels[0] if labels[0] in choices and rng.random() < 0.5 else _choose(rng, choices)
        labels.append(center)
    else:
        labels = _run_labels(rng, palette, run_count)

    thickness = np.full(run_count, 2 if broad_bands else 1, dtype=int)
    extra = length - int(thickness.sum())
    # A 'new' center occurs for the first time specifically in the last two rows.
    eligible = np.arange(run_count - (center_mode == "new"))
    if extra:
        if center_mode != "new" and rng.random() < 0.22:
            center_extra = int(rng.integers(1, extra + 1))
            thickness[-1] += center_extra
            extra -= center_extra
        weights = rng.dirichlet(np.full(len(eligible), (0.45, 1.0, 2.0)[complexity]))
        thickness[eligible] += rng.multinomial(extra, weights)
    layers = np.repeat(np.asarray(labels, dtype=np.int8), thickness)
    return np.append(layers, layers[-1]).astype(np.int8)


def _row_fingerprint(row_colors, width):
    return len(row_colors), int(width), tuple(int(c) for c in row_colors)


def _grid_key(grid):
    return grid.shape, tuple(int(c) for c in grid.flat)


def _render_output(row_colors):
    # Preserve the reviewed generator's correct edge-distance construction.
    side = 2 * (len(row_colors) - 1)
    edge = np.minimum(np.arange(side), np.arange(side)[::-1])
    return np.asarray(row_colors[np.minimum(edge[:, None], edge[None, :])], dtype=np.int8)


def _border_wrap(code):
    """Different construction used for validation: grow outward from the center."""
    result = np.full((2, 2), int(code[-1]), dtype=np.int8)
    for color in code[-3::-1]:
        result = np.pad(result, 1, constant_values=int(color))
    return result


def _make_pair(rng, height, width, palette_size, complexity, seen,
               center_mode=None, demonstrate_repeat=False):
    code = _sample_row_colors(rng, height, palette_size, complexity,
                              center_mode, demonstrate_repeat)
    # Bounded uniqueness fallback: 30 widths suffice for <=14 puzzle pairs
    # plus three excluded source inputs, even for a fixed monochrome code.
    for candidate_width in [width] + [w for w in range(1, 31) if w != width]:
        fingerprint = _row_fingerprint(code, candidate_width)
        if fingerprint not in seen and fingerprint not in _SOURCE_INPUT_FINGERPRINTS:
            seen.add(fingerprint)
            return {"input": np.repeat(code[:, None], candidate_width, axis=1),
                    "output": _render_output(code)}
    raise AssertionError("width fallback exhausted")


def _relative_choices(observed, lower, upper, mode, used=()):
    """Shift preferences are independent of the rule, with feasible fallbacks."""
    observed = sorted(set(observed))
    if mode == 0:
        preferred = range(lower, min(observed))
    elif mode == 1:
        preferred = observed
    elif mode == 2:
        preferred = range(max(observed) + 1, upper + 1)
    else:
        preferred = range(lower, upper + 1)
    choices = [v for v in preferred if lower <= v <= upper and v not in used]
    return choices or [v for v in range(lower, upper + 1) if v not in used] or list(range(lower, upper + 1))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # Semantics are always horizontal/top-to-bottom with literal colors.
    # These puzzle-level parameters change distributions only; no hidden rule.
    complexity = int(rng.integers(3))
    height_regime, width_regime, palette_regime = [int(rng.integers(4)) for _ in range(3)]
    # Preserve the source's combined shift as a minority of puzzles; the
    # remaining puzzles choose the three axes independently.
    if rng.random() < 0.12:
        height_regime, width_regime, palette_regime = 1, 0, 2
    width_floor = int(rng.integers(3, 12)) if width_regime == 0 else 1
    width_ceiling = int(rng.integers(19, 29)) if width_regime == 2 else 30
    palette_ceiling = int(rng.integers(3, 8)) if palette_regime == 2 else 10
    seen = set()
    train = []
    for index in range(train_count):
        if index < 2:
            height = _choose(rng, [h for h in range(5, 16)
                                   if not train or h != train[0]["input"].shape[0]])
        else:
            height = _sample_height(rng)
        width = _sample_width(rng, height,
                              [train[0]["input"].shape[1]] if index == 1 else ())
        if not width_floor <= width <= width_ceiling:
            width = _choose(rng, [w for w in range(width_floor, width_ceiling + 1)
                                  if index != 1 or w != train[0]["input"].shape[1]])
        if index == 1 and train[0]["input"].shape[0] == train[0]["input"].shape[1] and width == height:
            width = _choose(rng, [w for w in range(width_floor, width_ceiling + 1)
                                  if w not in (height, train[0]["input"].shape[1])])
        if index < 2:
            # One novel center, one actual nonadjacent center return; both can
            # coexist with any palette, size and run style that fits.
            allowed = list(range(2, min(palette_ceiling, height - 2) + 1))
            if index == 1:
                first_palette = len(np.unique(train[0]["input"]))
                allowed = [k for k in allowed if k != first_palette]
            palette_size = _sample_palette(rng, allowed, complexity)
            mode = "new" if index == 0 else "return"
        else:
            minimum = 1 if height == 2 or rng.random() < 0.07 else 2
            palette_size = _sample_palette(rng, range(minimum, min(palette_ceiling, height - 1) + 1), complexity)
            mode = None
        train.append(_make_pair(rng, height, width, palette_size, complexity, seen,
                                mode, demonstrate_repeat=index == 0))
    # Presentation order does not identify the demonstration's construction role.
    rng.shuffle(train)

    train_heights = [p["input"].shape[0] for p in train]
    train_widths = [p["input"].shape[1] for p in train]
    train_palettes = [len(np.unique(p["input"])) for p in train]
    test = []
    used_heights, used_widths, used_palettes = [], [], []
    # In mixed regimes, rotate the order of familiar/smaller/larger preferences.
    offsets = [int(rng.integers(3)) for _ in range(3)]
    for index in range(test_count):
        modes = [regime if regime != 3 else (index + offset) % 3
                 for regime, offset in zip((height_regime, width_regime, palette_regime), offsets)]
        height = _choose(rng, _relative_choices(train_heights, 2, 16, modes[0], used_heights))
        width = _choose(rng, _relative_choices(train_widths, 1, 30, modes[1], used_widths))
        palette_size = _sample_palette(rng, _relative_choices(train_palettes, 1,
                                       min(10, height - 1), modes[2], used_palettes), complexity)
        test.append(_make_pair(rng, height, width, palette_size, complexity, seen))
        used_heights.append(height)
        used_widths.append(width)
        used_palettes.append(palette_size)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _assert_grid_is_arc_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert bool(np.all((0 <= grid) & (grid <= 9)))


def _assert_task_transformation(input_grid, output_grid):
    height, _ = input_grid.shape
    assert 2 <= height <= 16
    code = input_grid[:, 0]
    assert np.all(input_grid == code[:, None])
    assert code[-2] == code[-1]
    assert output_grid.shape == (2 * (height - 1),) * 2
    assert np.array_equal(output_grid, _border_wrap(code))
    assert set(np.unique(input_grid)) == set(np.unique(output_grid))


def _extract_code(grid, axis, reverse):
    code = grid[:, 0] if axis == 0 else grid[0, :]
    uniform = np.all(grid == (code[:, None] if axis == 0 else code[None, :]))
    if not uniform or not 2 <= len(code) <= 16:
        return None
    code = code[::-1] if reverse else code
    return code if code[-2] == code[-1] else None


def _learn_convention(train_pairs):
    # Infer from the returned training grids only, without seeds or metadata.
    matches = []
    for axis in (0, 1):
        for reverse in (False, True):
            for collapse in (False, True):
                matches_all = True
                for pair in train_pairs:
                    code = _extract_code(pair["input"], axis, reverse)
                    if code is None:
                        matches_all = False
                        break
                    if collapse:
                        runs = code[np.r_[True, code[1:] != code[:-1]]]
                        code = np.append(runs, runs[-1])
                    if not np.array_equal(_border_wrap(code), pair["output"]):
                        matches_all = False
                        break
                if matches_all:
                    matches.append((axis, reverse, collapse))
    assert matches == [(0, False, False)], "training does not distinguish the reading/thickness convention"
    return matches[0]


def _predict_from_training(train_pairs, test_input):
    axis, reverse, _ = _learn_convention(train_pairs)
    code = _extract_code(test_input, axis, reverse)
    assert code is not None
    return _border_wrap(code)


def _code_features(pair):
    grid = pair["input"]
    code = grid[:, 0]
    layers = code[:-1]
    starts = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
    runs = code[starts]
    thickness = np.diff(np.r_[starts, len(code)])
    palette = len(np.unique(code))
    normalized = {}
    signature = tuple(normalized.setdefault(int(c), len(normalized)) for c in layers)
    return {
        "height": len(code), "width": grid.shape[1], "palette": palette,
        "new_center": bool(code[-1] not in code[:-2]),
        "returned_center": bool(runs[-1] in runs[:-1]),
        "outer_equals_center": bool(code[0] == code[-1]),
        "internal_repeat": bool(np.any(layers[1:] == layers[:-1])),
        "nonadjacent_return": len(runs) > palette,
        "single_use_color": bool(np.any(np.unique(code, return_counts=True)[1] == 1)),
        "alternating_stretch": any(code[i] == code[i + 2] and code[i + 1] == code[i + 3]
                                   and code[i] != code[i + 1] for i in range(len(code) - 3)),
        "thin_and_thick": bool(np.any(thickness[:-1] == 1) and np.any(thickness[:-1] > 1)),
        "terminal_run": int(thickness[-1]),
        "solid_center_side": 2 * (int(thickness[-1]) - 1),
        "run_count": len(runs),
        "simple_unique_layers": palette == len(layers),
        "two_color_bands": palette == 2 and len(runs) == 2,
        "two_color_broad_bands": bool(palette == 2 and len(runs) == 2
                                       and thickness[0] >= 2 and thickness[-1] >= 3),
        "monochrome": palette == 1,
        "structure": (len(code), grid.shape[1], signature),
    }


def _assert_training_evidence(puzzle):
    features = [_code_features(p) for p in puzzle["train"]]
    assert len({f["height"] for f in features}) >= 2
    assert len({f["width"] for f in features}) >= 2
    assert any(f["height"] != f["width"] for f in features)
    assert len({f["palette"] for f in features}) >= 2
    assert any(f["new_center"] for f in features)
    assert any(f["returned_center"] for f in features)
    assert any(f["internal_repeat"] and not f["monochrome"] for f in features)
    assert sum(not f["monochrome"] for f in features) >= 2
    _learn_convention(puzzle["train"])


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                _assert_grid_is_arc_grid(grid)
            _assert_task_transformation(pair["input"], pair["output"])
            inp_key, out_key = _grid_key(pair["input"]), _grid_key(pair["output"])
            assert inp_key not in seen_inputs and (inp_key, out_key) not in seen_pairs
            seen_inputs.add(inp_key)
            seen_pairs.add((inp_key, out_key))
            assert _row_fingerprint(pair["input"][:, 0], pair["input"].shape[1]) not in _SOURCE_INPUT_FINGERPRINTS
            # The H=2,W=2 monochrome fixed point is a meaningful minimal case.
            if np.array_equal(pair["input"], pair["output"]):
                assert pair["input"].shape == (2, 2)
    _assert_training_evidence(puzzle)
    for pair in puzzle["test"]:
        predicted = _predict_from_training(puzzle["train"], pair["input"].copy())
        assert np.array_equal(predicted, pair["output"])


def _puzzle_signature(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def _puzzles_equal(first, second):
    return _puzzle_signature(first) == _puzzle_signature(second)


def self_test() -> None:
    features, puzzles = [], []
    train_counts, test_counts, signatures, colors = set(), set(), set(), set()
    relations = {name: set() for name in ("height", "width", "palette")}
    mixed_tests = {name: 0 for name in relations}
    source_shift_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_puzzle_signature(puzzle))
        training = [_code_features(p) for p in puzzle["train"]]
        testing = [_code_features(p) for p in puzzle["test"]]
        features.extend(training + testing)
        for name in relations:
            observed = [f[name] for f in training]
            for f in testing:
                relations[name].add("below" if f[name] < min(observed) else
                                    "above" if f[name] > max(observed) else
                                    "familiar" if f[name] in observed else "between")
            mixed_tests[name] += len({f[name] for f in testing}) > 1
        source_shift_pairs += sum(f["width"] < min(t["width"] for t in training)
                                  and f["palette"] > max(t["palette"] for t in training)
                                  for f in testing)
        for pair in puzzle["train"] + puzzle["test"]:
            colors.update(int(c) for c in np.unique(pair["input"]))
            assert np.array_equal(pair["output"], _border_wrap(pair["input"][:, 0]))
        # Only training labels and the queried input enter prediction.
        train_copy = [{k: v.copy() for k, v in p.items()} for p in puzzle["train"]]
        predictions = [_predict_from_training(train_copy, p["input"].copy()) for p in puzzle["test"]]
        assert all(np.array_equal(pred, pair["output"]) for pred, pair in zip(predictions, puzzle["test"]))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42
    assert len({f["structure"] for f in features}) >= 220
    assert {f["height"] for f in features} == set(range(2, 17))
    assert {f["width"] for f in features} == set(range(1, 31))
    assert {f["palette"] for f in features} == set(range(1, 11))
    assert colors == set(range(10))
    assert all({"below", "above", "familiar"} <= values for values in relations.values())
    assert all(count >= 15 for count in mixed_tests.values())
    assert source_shift_pairs >= 3
    assert sum(f["simple_unique_layers"] and f["palette"] >= 3 for f in features) >= 10
    assert sum(f["two_color_broad_bands"] for f in features) >= 5
    for name in ("new_center", "returned_center", "outer_equals_center", "internal_repeat",
                 "nonadjacent_return", "single_use_color", "alternating_stretch", "thin_and_thick",
                 "simple_unique_layers", "two_color_bands", "monochrome"):
        assert sum(f[name] for f in features) >= 8, name
    assert max(f["terminal_run"] for f in features if not f["monochrome"]) >= 7
    assert sum(f["height"] <= 5 and f["width"] >= 24 for f in features) >= 5
    assert sum(f["height"] >= 12 and f["width"] <= 3 for f in features) >= 5
    assert sum(f["height"] == f["width"] for f in features) >= 5
    assert sum(f["new_center"] and f["thin_and_thick"] and f["width"] != f["height"] for f in features) >= 10
    assert sum(f["returned_center"] and f["palette"] >= 6 and f["internal_repeat"] for f in features) >= 5

    assert _puzzles_equal(generate_puzzle(987654321, 7, 3), generate_puzzle(987654321, 7, 3))
    cases = [(a, b) for a in range(2, 11) for b in range(1, min(a, 5))]
    cases += [(None, 1), (None, 4), (2, None), (10, None)]
    for i, (train_count, test_count) in enumerate(cases):
        puzzle = generate_puzzle(2000 + i, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for a, b in [(1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4),
                 (True, 1), (2.5, 1), (3, False)]:
        try:
            generate_puzzle(5000, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")

    # Numeric fixtures cover formerly excluded structures, including all bounds.
    for code in ([8, 8], [1, 2, 3, 4, 5, 5], [1, 1, 2, 2, 2, 2],
                 [1, 2, 1, 1], [2, 3, 2, 2, 3, 2, 1, 2, 2, 2, 2, 8, 8],
                 list(range(10)) + [0] * 6, [1, 2] + [3] * 7):
        code = np.asarray(code, dtype=np.int8)
        output = _render_output(code)
        for width in (1, 17, 30):
            _assert_task_transformation(np.repeat(code[:, None], width, axis=1), output)
    assert np.all(_render_output(np.array([1, 2] + [3] * 7))[2:-2, 2:-2] == 3)
    assert _render_output(np.array([1, 2] + [3] * 7))[2:-2, 2:-2].shape == (12, 12)
    bad = {split: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][split]]
           for split in ("train", "test")}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        return
    raise AssertionError("validator accepted a corrupted frame")
