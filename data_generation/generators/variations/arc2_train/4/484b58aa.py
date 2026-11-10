"""Generate repairs of a periodically repeated grid with one erasure color."""

import hashlib

import numpy as np


_SOURCE_DIGESTS = {
    "049b66b34be1c1f424c04a0e6bc079f56007b591fce14fcd9c4c734c6fd526ff",
    "4d33a1e8935a1b4f2b7137fe07f0e307b3cd360b25a7330d8e1d615a9229a9e3",
    "557c58bb13bbbaba8a5e2e66ddd3c2f8299dac82e35a76c9591f173dfe998ce6",
    "418909f1dcfe6c62ccfe909a5f429ae2ee8b86dfafd51d02afd14a68a81fc284",
}
_PERIODS = np.arange(2, 12)
_STYLES = ("independent", "rotating", "interleaved", "separator")


def _validate_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    if int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(seed)


def _count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    train = _count(num_train, "num_train")
    test = _count(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
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


def _key(array):
    return (array.shape, array.tobytes())


def _puzzle_key(puzzle):
    return tuple(
        tuple((_key(pair["input"]), _key(pair["output"])) for pair in puzzle[split])
        for split in ("train", "test")
    )


def _source_digest(pair):
    inp = pair["input"]
    out = pair["output"]
    if inp.shape != (29, 29) or out.shape != (29, 29):
        return None
    return hashlib.sha256(inp.astype(np.uint8).tobytes() + out.astype(np.uint8).tobytes()).hexdigest()


def _exact_period(grid, axis):
    arr = grid if axis == 0 else grid.T
    for period in range(1, arr.shape[0]):
        if np.array_equal(arr[:-period], arr[period:]):
            return period
    return None


def _candidate_repairs(input_grid, erasure):
    """Find all fully supported completions, using either repeat direction."""
    answers = {}
    for axis in (0, 1):
        arr = input_grid if axis == 0 else input_grid.T
        length, breadth = arr.shape
        for period in range(2, length // 2 + 1):
            before = arr[:-period]
            after = arr[period:]
            common = (before != erasure) & (after != erasure)
            evidence = int(np.count_nonzero(common))
            if evidence < max(6, (length - period) * breadth // 5):
                continue
            if np.any(before[common] != after[common]):
                continue
            restored = arr.copy()
            for phase in range(period):
                group = arr[phase::period]
                visible = group != erasure
                if np.any(np.count_nonzero(visible, axis=0) == 0):
                    break
                values = np.where(visible, group, 10)
                lo = values.min(axis=0)
                hi = np.where(visible, group, -1).max(axis=0)
                if np.any(lo != hi):
                    break
                restored[phase::period] = np.broadcast_to(lo, group.shape)
            else:
                if _exact_period(restored, 0) != period:
                    continue
                result = restored if axis == 0 else restored.T.copy()
                key = _key(result)
                if key not in answers or period < answers[key][2]:
                    answers[key] = (result, axis, period)
    return list(answers.values())


def _horizontal_period(grid):
    return _exact_period(grid, 1)


def _supported_axes(input_grid, output_grid, erasure):
    axes = set()
    for axis in (0, 1):
        inp = input_grid if axis == 0 else input_grid.T
        out = output_grid if axis == 0 else output_grid.T
        period = _exact_period(out, 0)
        if period is None or not 2 <= period <= 11 or inp.shape[0] < 2 * period:
            continue
        top, bottom = inp[:-period], inp[period:]
        common = (top != erasure) & (bottom != erasure)
        if int(common.sum()) < max(6, (inp.shape[0] - period) * inp.shape[1] // 5):
            continue
        if np.any(top[common] != bottom[common]):
            continue
        if any(np.any(np.all(inp[phase::period] == erasure, axis=0))
               for phase in range(period)):
            continue
        axes.add(axis)
    return axes


def _has_interleaved_families(output_grid, axis, period):
    arr = output_grid if axis == 0 else output_grid.T
    width = _horizontal_period(arr) or arr.shape[1]
    rows = arr[:period, :width]

    def rotation_of(row, reference):
        return any(np.array_equal(row, np.roll(reference, shift))
                   for shift in range(width))

    for family_count in (2, 3, 4):
        if period < 2 * family_count:
            continue
        prototypes = rows[:family_count]
        if not all(rotation_of(rows[row], prototypes[row % family_count])
                   for row in range(period)):
            continue
        if any(not rotation_of(prototypes[family], prototypes[0])
               for family in range(1, family_count)):
            return True
    return False


def _mask_components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    boxes = []
    while remaining:
        stack = [remaining.pop()]
        points = []
        while stack:
            row, column = stack.pop()
            points.append((row, column))
            for neighbor in ((row - 1, column), (row + 1, column),
                             (row, column - 1), (row, column + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        rows = [row for row, _ in points]
        columns = [column for _, column in points]
        boxes.append((max(rows) - min(rows) + 1,
                      max(columns) - min(columns) + 1))
    return boxes


def _choose_dimensions(rng, period):
    # The first dimension carries the repeat cycle; orientation is applied later.
    mode = int(rng.integers(0, 5))
    if mode == 0:
        height = 2 * period
    elif mode == 1:
        choices = [k * period for k in range(2, 31 // period + 1)]
        height = int(rng.choice(choices))
    elif mode == 2:
        height = 29 if period <= 14 else 2 * period
    else:
        height = int(rng.integers(2 * period, 31))
    if rng.random() < 0.11:
        width = 29
    else:
        width = int(rng.integers(5, 31))
    return height, width


def _choose_word_width(rng, period, crop_width):
    mode = int(rng.integers(0, 5))
    if mode == 0:
        return int(rng.integers(2, min(8, crop_width) + 1))
    if mode == 1:
        return period
    if mode == 2:
        return 18
    if mode == 3:
        return int(rng.integers(8, 31))
    return int(rng.integers(crop_width, 37))


def _word(rng, palette, length):
    weights = rng.dirichlet(np.ones(len(palette)) * float(rng.uniform(0.7, 2.0)))
    return rng.choice(palette, size=length, replace=True, p=weights).astype(np.int8)


def _make_block(rng, period, word_width, palette, style):
    if style == "independent":
        rows = []
        for _ in range(period):
            if rng.random() < 0.45:
                short = int(rng.integers(2, min(word_width, 7) + 1))
            else:
                short = word_width
            base = _word(rng, palette, short)
            rows.append(base[np.arange(word_width) % short])
        return np.stack(rows)

    if style == "rotating":
        base = _word(rng, palette, word_width)
        if rng.random() < 0.5:
            step = int(rng.integers(1, word_width))
            shifts = np.arange(period) * step
        else:
            shifts = rng.integers(0, word_width, size=period)
        return np.stack([np.roll(base, int(s)) for s in shifts])

    if style == "interleaved":
        families = int(rng.integers(2, min(4, period) + 1))
        bases = [_word(rng, palette, word_width) for _ in range(families)]
        steps = rng.integers(1, word_width, size=families)
        return np.stack([
            np.roll(bases[row % families], int((row // families) * steps[row % families]))
            for row in range(period)
        ])

    # A short repeated separator interleaves the changing content rows.
    gap = int(rng.integers(2, min(3, period - 2) + 1))
    separator_length = int(rng.integers(1, min(3, word_width) + 1))
    separator = _word(rng, palette, separator_length)
    separator = separator[np.arange(word_width) % separator_length]
    base = _word(rng, palette, word_width)
    step = int(rng.integers(1, word_width))
    return np.stack([
        separator if row % gap == 1 else np.roll(base, row * step)
        for row in range(period)
    ])


def _make_clean(rng, period, height, width, palette, style):
    for _ in range(40):
        word_width = _choose_word_width(rng, period, width)
        if word_width < 2:
            continue
        block = _make_block(rng, period, word_width, palette, style)
        row_phase = int(rng.integers(0, period))
        column_phase = int(rng.integers(0, word_width))
        rows = (np.arange(height) + row_phase) % period
        columns = (np.arange(width) + column_phase) % word_width
        clean = block[np.ix_(rows, columns)].astype(np.int8)
        if _exact_period(clean, 0) != period:
            continue
        if len(np.unique(clean)) < 2:
            continue
        return clean
    raise AssertionError("could not make a full-period clean pattern")


def _region(rng, height, width, period):
    kind = int(rng.integers(0, 7))
    if kind == 0:  # long patch, sometimes spanning a complete row cycle
        rh = int(rng.integers(2, min(height, period + 8) + 1))
        rw = int(rng.integers(2, min(width, 11) + 1))
    elif kind == 1:  # thin horizontal strip
        rh = int(rng.integers(1, min(3, height) + 1))
        rw = int(rng.integers(3, min(width, 18) + 1)) if width >= 3 else width
    elif kind == 2:  # thin vertical strip
        rh = int(rng.integers(3, min(height, 18) + 1)) if height >= 3 else height
        rw = int(rng.integers(1, min(3, width) + 1))
    else:
        rh = int(rng.integers(2, min(height, max(3, period)) + 1))
        rw = int(rng.integers(2, min(width, 10) + 1))
    top = int(rng.integers(0, height - rh + 1))
    left = int(rng.integers(0, width - rw + 1))
    if rng.random() < 0.23:
        top = 0 if rng.random() < 0.5 else height - rh
    if rng.random() < 0.23:
        left = 0 if rng.random() < 0.5 else width - rw
    coords = []
    for r in range(rh):
        if kind == 3:
            lo, hi = min(r // 2, rw - 1), rw
        elif kind == 4:
            lo, hi = 0, max(1, rw - r // 2)
        elif kind == 5:
            inset = abs(2 * r - rh + 1) * rw // max(2, 2 * rh)
            lo, hi = inset, max(inset + 1, rw - inset)
        elif kind == 6:
            inset = int(rng.integers(0, min(3, rw)))
            lo, hi = inset, rw
        else:
            lo, hi = 0, rw
        coords.extend((top + r, left + c) for c in range(lo, hi))
    return coords


def _make_mask(rng, height, width, period, regime):
    mask = np.zeros((height, width), dtype=bool)
    copies = np.bincount(np.arange(height) % period, minlength=period)
    remaining = np.broadcast_to(copies[:, None], (period, width)).copy()
    limit = max(4, int(height * width * float(rng.uniform(0.14, 0.30))))
    regions = int(rng.integers(1, 5 if regime == "source_train" else 7))

    def add(row, column):
        if mask[row, column] or mask.sum() >= limit:
            return
        phase = row % period
        if remaining[phase, column] <= 1:
            return
        mask[row, column] = True
        remaining[phase, column] -= 1

    for _ in range(regions):
        for row, column in _region(rng, height, width, period):
            add(row, column)

    # Overlapping damage across corresponding copies makes one witness suffice.
    chance = 0.28 if regime == "source_train" else 0.56
    if rng.random() < chance:
        phase = int(rng.integers(0, period))
        left = int(rng.integers(0, width))
        span = int(rng.integers(1, min(6, width - left) + 1))
        for column in range(left, left + span):
            phase_rows = list(range(phase, height, period))
            survivor = int(rng.choice(phase_rows))
            for row in phase_rows:
                if row != survivor:
                    add(row, column)

    minimum = max(3, min(12, (height * width) // 30))
    if mask.sum() < minimum:
        for flat in rng.permutation(height * width):
            add(int(flat) // width, int(flat) % width)
            if mask.sum() >= minimum:
                break
    return mask


def _make_pair(rng, regime, erasure, period, orientation, style):
    for _ in range(80):
        height, width = _choose_dimensions(rng, period)
        colors = np.array([color for color in range(10) if color != erasure], dtype=np.int8)
        palette_size = int(rng.integers(2, 9))
        palette = rng.choice(colors, size=palette_size, replace=False)
        actual_style = style if period >= 5 else ("interleaved" if style == "separator" else style)
        clean = _make_clean(rng, period, height, width, palette, actual_style)
        mask = _make_mask(rng, height, width, period, regime)
        if not mask.any():
            continue
        inp = clean.copy()
        inp[mask] = erasure
        if orientation == 1:
            inp, clean = inp.T.copy(), clean.T.copy()
        repairs = _candidate_repairs(inp, erasure)
        if len(repairs) != 1 or not np.array_equal(repairs[0][0], clean):
            continue
        pair = {"input": inp, "output": clean}
        if _source_digest(pair) in _SOURCE_DIGESTS:
            continue
        return pair
    raise AssertionError("could not construct a uniquely repairable pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed_value = _validate_seed(seed)
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    erasure = int(rng.integers(0, 10))

    orientation_mode = int(rng.integers(0, 3))
    first_axis = int(rng.integers(0, 2))
    if orientation_mode == 0:  # one orientation throughout
        train_axes = [first_axis] * train_count
        test_axes = [first_axis] * test_count
    elif orientation_mode == 1:  # both orientations in the demonstrations
        train_axes = [(first_axis + i) % 2 for i in range(train_count)]
        test_axes = [int(rng.integers(0, 2)) for _ in range(test_count)]
    else:  # transpose generalization, with direction clear in the test input
        train_axes = [first_axis] * train_count
        test_axes = [1 - first_axis] * test_count

    period_order = list(rng.permutation(_PERIODS).astype(int))
    train_periods = period_order[:train_count]
    test_periods = []
    for _ in range(test_count):
        choice = int(rng.integers(0, 3))
        if choice == 0:
            period = int(rng.choice(train_periods))
        elif choice == 1:
            available = [p for p in _PERIODS if p not in train_periods]
            period = int(rng.choice(available if available else _PERIODS))
        else:
            larger = [p for p in _PERIODS if p > max(train_periods)]
            period = int(rng.choice(larger if larger else _PERIODS))
        test_periods.append(period)

    puzzle = {"train": [], "test": []}
    inputs = set()
    training_signatures = set()
    for split, regime, periods, axes in (
        ("train", "source_train", train_periods, train_axes),
        ("test", "target_test", test_periods, test_axes),
    ):
        for period, axis in zip(periods, axes):
            for _ in range(24):
                local_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                local_rng = np.random.default_rng(local_seed)
                style = _STYLES[int(local_rng.integers(0, len(_STYLES)))]
                pair = _make_pair(local_rng, regime, erasure, period, axis, style)
                key = _key(pair["input"])
                signature = tuple(
                    (supported_axis, _exact_period(pair["output"], supported_axis))
                    for supported_axis in sorted(
                        _supported_axes(pair["input"], pair["output"], erasure))
                )
                if key in inputs:
                    continue
                if split == "train" and len(puzzle["train"]) == 1 and signature in training_signatures:
                    continue
                inputs.add(key)
                puzzle[split].append(pair)
                if split == "train":
                    training_signatures.add(signature)
                break
            else:
                raise AssertionError("duplicate input construction")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    erasures = set()
    for pair in puzzle["train"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = pair["input"], pair["output"]
        assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
        assert inp.shape == out.shape
        changed = inp != out
        assert np.any(changed)
        values = np.unique(inp[changed])
        assert len(values) == 1
        erasures.add(int(values[0]))
    assert len(erasures) == 1
    erasure = next(iter(erasures))

    train_period_signatures = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert grid.dtype.kind in "iu"
                assert all(1 <= size <= 30 for size in grid.shape)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            assert inp.shape == out.shape
            assert inp.dtype == out.dtype
            input_key = _key(inp)
            pair_key = (input_key, _key(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _source_digest(pair) not in _SOURCE_DIGESTS
            mask = inp == erasure
            assert mask.any()
            assert np.array_equal(mask, inp != out)
            assert not np.any(out == erasure)
            assert np.array_equal(inp[~mask], out[~mask])
            assert len(np.unique(out)) >= 2
            repairs = _candidate_repairs(inp, erasure)
            assert len(repairs) == 1 and np.array_equal(repairs[0][0], out)
            axis, period = repairs[0][1:]
            assert 2 <= period <= 11
            assert inp.shape[axis] >= 2 * period
            if split == "train":
                train_period_signatures.add(
                    tuple((supported_axis, _exact_period(out, supported_axis))
                          for supported_axis in sorted(
                              _supported_axes(inp, out, erasure)))
                )
    assert len(train_period_signatures) >= 2


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzle_key(generate_puzzle(17)) == _puzzle_key(generate_puzzle(17))
    assert _puzzle_key(generate_puzzle(18, 6, 2)) == _puzzle_key(generate_puzzle(18, 6, 2))

    periods, horizontal_periods, shapes, palettes = set(), set(), set(), set()
    orientations, remainders, witness_counts = set(), set(), set()
    repeated_rows = 0
    exact_multiples = 0
    two_copy = 0
    top_damage = 0
    off_top_damage = 0
    square_29 = 0
    within_period_variation = 0
    within_axis_variation = 0
    differing_axis_periods = 0
    no_horizontal_period = 0
    small_grids = 0
    wide_aspect = 0
    one_component = 0
    many_components = 0
    thin_strips = 0
    cross_cycle_components = 0
    interior_only = 0
    edge_damage = 0
    tests_other_than_nine = 0
    puzzles_with_palette_variation = 0
    uniform_orientation_puzzles = 0
    mixed_orientation_puzzles = 0
    orientation_shift_puzzles = 0
    interleaved_family_pairs = 0
    familiar_period_tests = 0
    new_period_tests = 0
    longer_period_tests = 0
    all_inputs = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        inferred = []
        puzzle_palettes = set()
        training_axes = []
        testing_axes = []
        training_periods = set()
        for pair_index, pair in enumerate(puzzle["train"] + puzzle["test"]):
            inp, out = pair["input"], pair["output"]
            changed = inp != out
            erasure = int(np.unique(inp[changed])[0])
            answer = _candidate_repairs(inp.copy(), erasure)
            assert len(answer) == 1 and np.array_equal(answer[0][0], out)
            axis, period = answer[0][1:]
            inferred.append((axis, period))
            supported = _supported_axes(inp, out, erasure)
            assert supported
            (training_axes if pair_index < len(puzzle["train"]) else testing_axes).append(supported)
            supported_periods = {_exact_period(out, supported_axis)
                                 for supported_axis in supported}
            if pair_index < len(puzzle["train"]):
                training_periods.update(supported_periods)
            else:
                familiar_period_tests += bool(supported_periods & training_periods)
                new_period_tests += bool(supported_periods - training_periods)
                longer_period_tests += any(p > max(training_periods)
                                           for p in supported_periods)
            interleaved_family_pairs += any(
                _has_interleaved_families(
                    out, supported_axis, _exact_period(out, supported_axis))
                for supported_axis in supported
            )
            orientations.add(axis)
            periods.add(period)
            shapes.add(tuple(inp.shape))
            palettes.add(len(np.unique(out)))
            puzzle_palettes.add(len(np.unique(out)))
            hp = _horizontal_period(out)
            horizontal_periods.add(hp)
            vp = _exact_period(out, 0)
            differing_axis_periods += vp is not None and hp is not None and vp != hp
            no_horizontal_period += hp is None
            small_grids += min(inp.shape) <= 10
            wide_aspect += max(inp.shape) >= 2 * min(inp.shape)
            if inp.shape[axis] % period == 0:
                exact_multiples += 1
            remainders.add(inp.shape[axis] % period)
            if inp.shape[axis] == 2 * period:
                two_copy += 1
            if inp.shape == (29, 29):
                square_29 += 1
            if changed[0].any():
                top_damage += 1
            else:
                off_top_damage += 1
            components = _mask_components(changed)
            one_component += len(components) == 1
            many_components += len(components) >= 4
            thin_strips += any(min(h, w) <= 2 and max(h, w) >= 5
                               for h, w in components)
            cross_cycle_components += any((h if axis == 0 else w) > period
                                          for h, w in components)
            interior_only += not (changed[0].any() or changed[-1].any()
                                  or changed[:, 0].any() or changed[:, -1].any())
            edge_damage += changed[0].any() or changed[-1].any() or changed[:, 0].any() or changed[:, -1].any()
            if pair_index >= len(puzzle["train"]):
                tests_other_than_nine += period != 9
            block = out[:period] if axis == 0 else out[:, :period].T
            if len({row.tobytes() for row in block}) < period:
                repeated_rows += 1
            work_inp = inp if axis == 0 else inp.T
            work_mask = changed if axis == 0 else changed.T
            for row, column in np.argwhere(work_mask):
                phase_rows = range(int(row) % period, work_inp.shape[0], period)
                witnesses = sum(work_inp[r, column] != erasure for r in phase_rows)
                witness_counts.add(witnesses)
            key = _key(inp)
            assert key not in all_inputs
            all_inputs.add(key)
        within_period_variation += len({period for _, period in inferred}) > 1
        within_axis_variation += len({axis for axis, _ in inferred}) > 1
        puzzles_with_palette_variation += len(puzzle_palettes) > 1
        all_axes = training_axes + testing_axes
        uniform_orientation_puzzles += bool(set.intersection(*all_axes))
        mixed_orientation_puzzles += {0} in all_axes and {1} in all_axes
        orientation_shift_puzzles += (
            (all(axes == {0} for axes in training_axes) and {1} in testing_axes)
            or (all(axes == {1} for axes in training_axes) and {0} in testing_axes)
        )

    assert periods.issuperset({2, 3, 6, 7, 8, 9, 10, 11})
    assert len(shapes) >= 70 and len(palettes) >= 5
    assert orientations == {0, 1} and len(horizontal_periods) >= 8
    assert None in horizontal_periods and 18 in horizontal_periods
    assert differing_axis_periods >= 10 and no_horizontal_period >= 10
    assert repeated_rows >= 5 and exact_multiples >= 20 and two_copy >= 20
    assert 1 in witness_counts and 2 in witness_counts
    assert top_damage >= 10 and off_top_damage >= 10
    assert square_29 >= 1
    assert within_period_variation == 42 and within_axis_variation >= 8
    assert len(remainders) >= 7
    assert small_grids >= 10 and wide_aspect >= 10
    assert one_component >= 10 and many_components >= 10
    assert thin_strips >= 10 and cross_cycle_components >= 10
    assert interior_only >= 10 and edge_damage >= 10
    assert tests_other_than_nine >= 10
    assert puzzles_with_palette_variation >= 35
    assert uniform_orientation_puzzles >= 5
    assert mixed_orientation_puzzles >= 5
    assert orientation_shift_puzzles >= 1
    assert interleaved_family_pairs >= 5
    assert familiar_period_tests >= 5
    assert new_period_tests >= 5
    assert longer_period_tests >= 5

    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (1101, None, 1), (1102, None, 4),
                              (1103, 2, None), (1104, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1200, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
