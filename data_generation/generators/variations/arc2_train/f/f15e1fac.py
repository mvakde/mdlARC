import numpy as np


ORIENTATIONS = (
    ("top", "left"), ("top", "right"),
    ("bottom", "left"), ("bottom", "right"),
    ("left", "top"), ("left", "bottom"),
    ("right", "top"), ("right", "bottom"),
)
SOURCE_LAYOUTS = {
    ((14, 10), "top", "right", (2, 6), (3, 7, 11)),
    ((17, 12), "top", "left", (1, 5, 7, 9), (4, 10)),
    ((12, 12), "left", "bottom", (2, 6, 9), (4, 8)),
    ((12, 17), "right", "bottom", (2, 5, 8, 10), (5, 10, 14)),
}


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = 1 + int(seed) % 4
        low = max(2, test + 1)
        train = low + (int(seed) // 4) % (11 - low)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _delta(signal_side, control_side):
    if (signal_side, control_side) not in ORIENTATIONS:
        raise ValueError("the two occupied sides must be adjacent")
    if signal_side in ("top", "bottom"):
        return 1 if control_side == "left" else -1
    return 1 if control_side == "top" else -1


def _palette_plan(rng):
    colors = [int(x) for x in rng.permutation(10)]
    mode = ("fixed", "background", "signal", "control")[int(rng.integers(0, 4))]
    return {
        "mode": mode,
        "base": dict(zip(("background", "signal", "control"), colors[:3])),
        "alternates": colors,
    }


def _pair_palette(plan, pair_index):
    palette = plan["base"].copy()
    role = plan["mode"]
    if role != "fixed":
        fixed = {value for key, value in palette.items() if key != role}
        choices = [color for color in plan["alternates"] if color not in fixed]
        palette[role] = choices[pair_index % len(choices)]
    return palette


def _dimensions(rng, signal_side, signals, ticks, pattern, schedule):
    minimum_cross = max(6, signals + 2)
    if pattern == "comb" and signals >= 2:
        minimum_cross = max(minimum_cross, 2 * signals)
    if pattern == "grouped" and signals >= 3:
        minimum_cross = max(minimum_cross, signals + 4)
    allowed = None
    minimum_sweep = max(6, ticks + 4)
    if schedule == "equal":
        minimum_sweep = max(minimum_sweep, 2 * (ticks + 1))
        allowed = {x for x in range(6, 31) if x % (ticks + 1) == 0}
    aspect = ("square", "portrait", "landscape")[int(rng.integers(3))]
    size_bins = [[], [], []]
    for sweep in range(minimum_sweep, 31):
        if allowed is not None and sweep not in allowed:
            continue
        for cross in range(minimum_cross, 31):
            height, width = ((sweep, cross) if signal_side in ("top", "bottom")
                             else (cross, sweep))
            actual_aspect = ("square" if height == width else
                             "portrait" if height > width else "landscape")
            if actual_aspect == aspect:
                size = max(height, width)
                bin_index = 0 if size <= 10 else 1 if size <= 20 else 2
                size_bins[bin_index].append((height, width))
    available = [bucket for bucket in size_bins if bucket]
    if not available:
        raise ValueError("no fitting grid dimensions")
    bucket = available[int(rng.integers(len(available)))]
    return bucket[int(rng.integers(len(bucket)))]


def _tick_ranks(rng, sweep_length, count, style):
    low, high = 2, sweep_length - 2
    if style == "equal":
        step = sweep_length // (count + 1)
        ranks = [step * i for i in range(1, count + 1)]
    elif style == "burst":
        start = int(rng.integers(low, high - count + 2))
        ranks = list(range(start, start + count))
    elif style == "long":
        first_count = (count + 1) // 2
        ranks = list(range(low, low + first_count))
        ranks += list(range(high - (count - first_count) + 1, high + 1))
    elif style in ("early", "late"):
        anchor = low if style == "early" else high
        remaining = [x for x in range(low, high + 1) if x != anchor]
        ranks = [anchor] + [int(x) for x in rng.choice(
            remaining, size=count - 1, replace=False
        )]
    else:
        ranks = [int(x) for x in rng.choice(
            np.arange(low, high + 1), size=count, replace=False
        )]
    return tuple(sorted(ranks))


def _relative_patterns(rng, length, count, style):
    if count == 1:
        return [(0,)]
    if style == "run":
        return [tuple(range(count))]
    if style == "comb":
        max_gap = (length - 1) // (count - 1)
        return [tuple(i * gap for i in range(count))
                for gap in range(2, max_gap + 1)]
    if style == "grouped" and count >= 3:
        patterns = []
        for split in range(1, count):
            for gap in range(3, min(8, length - count + 2)):
                second = split - 1 + gap
                positions = tuple(range(split)) + tuple(
                    second + i for i in range(count - split)
                )
                if positions[-1] < length:
                    patterns.append(positions)
        return patterns
    patterns = set()
    for _ in range(36):
        positions = sorted(int(x) for x in rng.choice(
            np.arange(length), size=count, replace=False
        ))
        gaps = np.diff(positions)
        if len(set(int(x) for x in gaps)) >= 2 and not (
            sum(int(x) > 1 for x in gaps) == 1
            and all(int(x) in (1, max(gaps)) for x in gaps)
        ):
            patterns.add(tuple(x - positions[0] for x in positions))
    if not patterns:
        positions = list(range(count))
        positions[-1] += min(2, length - count)
        patterns.add(tuple(positions))
    return sorted(patterns)


def _signal_positions(rng, length, count, ticks, delta, style, clipping):
    candidates = {}
    for relative in _relative_patterns(rng, length, count, style):
        for offset in range(length - relative[-1]):
            positions = tuple(x + offset for x in relative)
            if not any(0 < x < length - 1 for x in positions):
                continue
            clipped = sum(not 0 <= x + ticks * delta < length for x in positions)
            candidates.setdefault(clipped, []).append(positions)
    if not candidates:
        raise ValueError("could not place signal markers")
    if clipping == "none":
        desired = [0]
    elif clipping == "all":
        desired = [count]
    else:
        desired = list(range(1, count))
        if count == 1:
            desired = [1]
    available = [number for number in desired if number in candidates]
    if not available:
        available = list(candidates)
    number = int(available[int(rng.integers(len(available)))])
    options = candidates[number]
    return options[int(rng.integers(len(options)))]


def _place_markers(height, width, signal_side, control_side, positions, ranks, palette):
    grid = np.full((height, width), palette["background"], dtype=np.int8)
    sweep_length = height if signal_side in ("top", "bottom") else width
    ticks = tuple(sweep_length - 1 - rank if signal_side in ("bottom", "right")
                  else rank for rank in ranks)
    signal = palette["signal"]
    control = palette["control"]
    if signal_side == "top":
        grid[0, list(positions)] = signal
    elif signal_side == "bottom":
        grid[-1, list(positions)] = signal
    elif signal_side == "left":
        grid[list(positions), 0] = signal
    else:
        grid[list(positions), -1] = signal
    if control_side == "top":
        grid[0, list(ticks)] = control
    elif control_side == "bottom":
        grid[-1, list(ticks)] = control
    elif control_side == "left":
        grid[list(ticks), 0] = control
    else:
        grid[list(ticks), -1] = control
    return grid


def _side_coordinates(grid, side, color):
    if side == "top":
        edge = grid[0, :]
    elif side == "bottom":
        edge = grid[-1, :]
    elif side == "left":
        edge = grid[:, 0]
    else:
        edge = grid[:, -1]
    return tuple(int(x) for x in np.flatnonzero(edge == color))


def _construct_output(input_grid, signal, control, signal_side, control_side):
    output = input_grid.copy()
    height, width = input_grid.shape
    delta = _delta(signal_side, control_side)
    active = np.asarray(_side_coordinates(input_grid, signal_side, signal), dtype=int)
    ticks = set(_side_coordinates(input_grid, control_side, control))
    if signal_side in ("top", "bottom"):
        sweep = range(height) if signal_side == "top" else range(height - 1, -1, -1)
        cross_length = width
        for row in sweep:
            if row in ticks:
                active = active + delta
                active = active[(active >= 0) & (active < cross_length)]
            output[row, active] = signal
    else:
        sweep = range(width) if signal_side == "left" else range(width - 1, -1, -1)
        cross_length = height
        for column in sweep:
            if column in ticks:
                active = active + delta
                active = active[(active >= 0) & (active < cross_length)]
            output[active, column] = signal
    output[input_grid == control] = control
    output[input_grid == signal] = signal
    return output


def _marker_side(grid, color):
    rows, columns = np.where(grid == color)
    assert len(rows)
    sides = []
    if np.all(rows == 0):
        sides.append("top")
    if np.all(rows == grid.shape[0] - 1):
        sides.append("bottom")
    if np.all(columns == 0):
        sides.append("left")
    if np.all(columns == grid.shape[1] - 1):
        sides.append("right")
    assert len(sides) == 1, "marker side is ambiguous"
    return sides[0]


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 3
    order = np.argsort(counts)
    assert counts[order[-1]] > counts[order[-2]], "background is ambiguous"
    return int(values[order[-1]])


def _infer_pair_palette(pair):
    inp, out = pair["input"], pair["output"]
    background = _background(inp)
    markers = [int(x) for x in np.unique(inp) if int(x) != background]
    assert set(int(x) for x in np.unique(out)) == set(int(x) for x in np.unique(inp))
    growing = [x for x in markers if np.count_nonzero(out == x) >
               np.count_nonzero(inp == x)]
    unchanged = [x for x in markers if np.count_nonzero(out == x) ==
                 np.count_nonzero(inp == x)]
    assert len(growing) == 1 and len(unchanged) == 1
    return {"background": background, "signal": growing[0], "control": unchanged[0]}


def _features(pair, palette):
    inp, out = pair["input"], pair["output"]
    height, width = inp.shape
    signal, control = palette["signal"], palette["control"]
    assert set(int(x) for x in np.unique(inp)) == set(palette.values())
    signal_side = _marker_side(inp, signal)
    control_side = _marker_side(inp, control)
    assert (signal_side, control_side) in ORIENTATIONS
    control_positions = _side_coordinates(inp, control_side, control)
    signal_positions = _side_coordinates(inp, signal_side, signal)
    sweep_length = height if signal_side in ("top", "bottom") else width
    cross_length = width if signal_side in ("top", "bottom") else height
    tick_ranks = tuple(sorted(sweep_length - 1 - x if signal_side in ("bottom", "right")
                              else x for x in control_positions))
    assert len(signal_positions) >= 1 and len(tick_ranks) >= 1
    assert any(0 < x < cross_length - 1 for x in signal_positions)
    assert all(2 <= x <= sweep_length - 2 for x in tick_ranks)
    assert np.array_equal(out, _construct_output(inp, signal, control,
                                                 signal_side, control_side))
    assert not np.array_equal(inp, out)
    clipped = sum(not 0 <= x + len(tick_ranks) * _delta(signal_side, control_side)
                  < cross_length for x in signal_positions)
    layout = (inp.shape, signal_side, control_side, signal_positions, tick_ranks)
    assert not (tuple(palette[x] for x in ("background", "signal", "control")) ==
                (0, 8, 2) and layout in SOURCE_LAYOUTS), "source pair copied"
    gaps = np.diff((0,) + tick_ranks + (sweep_length,))
    signal_gaps = np.diff(signal_positions)
    if len(signal_positions) == 1:
        pattern = "single"
    elif np.all(signal_gaps == 1):
        pattern = "run"
    elif np.all(signal_gaps == signal_gaps[0]):
        pattern = "comb"
    elif np.count_nonzero(signal_gaps > 1) == 1:
        pattern = "grouped"
    else:
        pattern = "irregular"
    return {
        "orientation": (signal_side, control_side),
        "shape": inp.shape,
        "signals": len(signal_positions),
        "ticks": len(tick_ranks),
        "clipped": clipped,
        "pattern": pattern,
        "equal_strips": len(set(int(x) for x in gaps)) == 1,
        "adjacent_ticks": any(b - a == 1 for a, b in zip(tick_ranks, tick_ranks[1:])),
        "long_quiet": int(max(gaps)) >= 8,
        "near_start": tick_ranks[0] == 2,
        "near_end": tick_ranks[-1] == sweep_length - 2,
    }


def _grid_key(grid):
    return (grid.shape, grid.astype(np.int8, copy=False).tobytes())


def _pair(rng, signal_side, control_side, signal_count, tick_count,
          palette, regime):
    patterns = ("run", "comb", "grouped", "irregular")
    pattern = patterns[int(rng.integers(len(patterns)))]
    if signal_count < 3 and pattern == "grouped":
        pattern = "comb" if signal_count == 2 else "run"
    if signal_count == 1:
        pattern = "single"
    schedules = ("equal", "random", "burst", "long", "early", "late")
    if regime == "target_test":
        schedules = ("equal", "random", "burst", "long", "early", "late",
                     "burst", "long")
    schedule = schedules[int(rng.integers(len(schedules)))]
    clipping = ("none", "partial", "all")[int(rng.integers(3))]
    height, width = _dimensions(rng, signal_side, signal_count,
                                tick_count, pattern, schedule)
    cross_length = width if signal_side in ("top", "bottom") else height
    sweep_length = height if signal_side in ("top", "bottom") else width
    ranks = _tick_ranks(rng, sweep_length, tick_count, schedule)
    positions = _signal_positions(rng, cross_length, signal_count,
                                  tick_count, _delta(signal_side, control_side),
                                  pattern, clipping)
    inp = _place_markers(height, width, signal_side, control_side,
                         positions, ranks, palette)
    out = _construct_output(inp, palette["signal"], palette["control"],
                            signal_side, control_side)
    return {"input": inp, "output": out}


def _training_orientations(rng, count):
    first = ORIENTATIONS[int(rng.integers(len(ORIENTATIONS)))]
    alternatives = [orientation for orientation in ORIENTATIONS
                    if orientation[0] != first[0]
                    and _delta(*orientation) != _delta(*first)]
    second = alternatives[int(rng.integers(len(alternatives)))]
    result = [first, second]
    unused = [orientation for orientation in ORIENTATIONS if orientation not in result]
    rng.shuffle(unused)
    while len(result) < min(count, 5):
        result.append(unused.pop())
    while len(result) < count:
        result.append(result[int(rng.integers(len(result)))])
    rng.shuffle(result)
    return result


def _test_orientation(rng, seed, index, training):
    def flipped(orientation):
        return next(candidate for candidate in ORIENTATIONS
                    if candidate[0] == orientation[0]
                    and candidate[1] != orientation[1])

    regime = (int(seed) % 3) if index == 0 else int(rng.integers(3))
    if regime == 0:
        return training[int(rng.integers(len(training)))]
    if regime == 1:
        choices = [orientation for orientation in ORIENTATIONS
                   if orientation not in training]
        if choices:
            return choices[int(rng.integers(len(choices)))]
    unpaired = [orientation for orientation in training
                if flipped(orientation) not in training]
    bases = unpaired if unpaired else training
    base = bases[int(rng.integers(len(bases)))]
    return flipped(base)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    plan = _palette_plan(rng)
    orientations = _training_orientations(rng, train_count)
    puzzle = {"train": [], "test": []}
    seen = set()
    train_signal_counts = []
    train_tick_counts = []
    for index in range(train_count + test_count):
        split = "train" if index < train_count else "test"
        local = index if split == "train" else index - train_count
        signal_side, control_side = (
            orientations[local] if split == "train" else
            _test_orientation(rng, seed, local, orientations)
        )
        if split == "train":
            if local == 0:
                signals = int(rng.integers(3, 6))
                ticks = int(rng.integers(2, 5))
            elif local == 1 and rng.random() < 0.65:
                signals = 1
                ticks = int(rng.integers(2, 6))
            else:
                signals = int(rng.integers(1, 9))
                ticks = int(rng.integers(1, 9))
            train_signal_counts.append(signals)
            train_tick_counts.append(ticks)
        else:
            if train_signal_counts and rng.random() < 0.45:
                signals = train_signal_counts[int(rng.integers(len(train_signal_counts)))]
            else:
                signals = int(rng.integers(1, 10))
            if train_tick_counts and rng.random() < 0.4:
                ticks = train_tick_counts[int(rng.integers(len(train_tick_counts)))]
            else:
                ticks = int(rng.integers(1, 9))
        palette = _pair_palette(plan, index)
        for _ in range(24):
            pair = _pair(rng, signal_side, control_side, signals, ticks,
                         palette, "source_train" if split == "train" else "target_test")
            key = _grid_key(pair["input"])
            if key not in seen:
                try:
                    _features(pair, palette)
                except AssertionError:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise ValueError("could not construct a distinct valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    palettes = {"train": [], "test": []}
    features = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert grid.dtype.kind in "iu"
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(np.min(grid)) <= int(np.max(grid)) <= 9
            assert pair["input"].shape == pair["output"].shape
            key = _grid_key(pair["input"])
            pair_key = (key, _grid_key(pair["output"]))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            palette = _infer_pair_palette(pair)
            palettes[split].append(palette)
            features[split].append(_features(pair, palette))
    train_palettes = palettes["train"]
    role_values = {role: {p[role] for p in train_palettes}
                   for role in ("background", "signal", "control")}
    assert sum(len(values) > 1 for values in role_values.values()) <= 1
    assert len(role_values["signal"]) == 1 or len(role_values["control"]) == 1
    orientations = [f["orientation"] for f in features["train"]]
    assert len(set(orientations)) >= 2
    assert {_delta(*orientation) for orientation in orientations} == {-1, 1}
    for pair, palette in zip(puzzle["test"], palettes["test"]):
        background = _background(pair["input"])
        markers = {int(x) for x in np.unique(pair["input"])} - {background}
        for role, values in role_values.items():
            if len(values) == 1:
                assert palette[role] == next(iter(values))
        if len(role_values["signal"]) == 1:
            signal = next(iter(role_values["signal"]))
            assert signal in markers
            control = next(iter(markers - {signal}))
        else:
            control = next(iter(role_values["control"]))
            assert control in markers
            signal = next(iter(markers - {control}))
        assert palette["signal"] == signal and palette["control"] == control
        assert background == palette["background"]


def _fingerprint(puzzle):
    return tuple((split, _grid_key(pair["input"]), _grid_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    train_counts, test_counts, fingerprints = set(), set(), set()
    first_orientations = set()
    orientations = {"train": set(), "test": set()}
    aspect_axes = {"train": set(), "test": set()}
    signal_counts = {"train": set(), "test": set()}
    tick_counts = {"train": set(), "test": set()}
    dimension_ranges = {
        "train": {"height": set(), "width": set()},
        "test": {"height": set(), "width": set()},
    }
    timing = {
        "train": {"equal": 0, "adjacent": 0, "quiet": 0, "early": 0, "late": 0},
        "test": {"equal": 0, "adjacent": 0, "quiet": 0, "early": 0, "late": 0},
    }
    shapes, patterns, clip_categories, pattern_clips = set(), set(), set(), set()
    clipped_counts, color_modes, regimes = set(), set(), set()
    within_aspects = within_signals = within_ticks = within_patterns = 0
    learned_singletons = 0
    one_tick_clipping = multi_tick_unclipped = multi_tick_empty = 0
    equal_strips = adjacent_ticks = long_quiet = near_start = near_end = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(_fingerprint(puzzle))
        train_palettes = [_infer_pair_palette(pair) for pair in puzzle["train"]]
        varying = [role for role in ("background", "signal", "control")
                   if len({palette[role] for palette in train_palettes}) > 1]
        color_modes.add(varying[0] if varying else "fixed")
        train_orientations = set()
        puzzle_aspects, puzzle_signals, puzzle_ticks, puzzle_patterns = (
            set(), set(), set(), set()
        )
        train_signals = set()
        for split in ("train", "test"):
            for pair_index, pair in enumerate(puzzle[split]):
                feature = _features(pair, _infer_pair_palette(pair))
                orientation = feature["orientation"]
                if split == "train" and pair_index == 0:
                    first_orientations.add(orientation)
                height, width = feature["shape"]
                aspect = ("square" if height == width else
                          "portrait" if height > width else "landscape")
                axis = ("vertical" if orientation[0] in ("top", "bottom")
                        else "horizontal")
                if split == "train":
                    train_orientations.add(orientation)
                    train_signals.add(feature["signals"])
                else:
                    regimes.add("shared" if orientation in train_orientations else
                                "flipped_control" if any(
                                    item[0] == orientation[0]
                                    for item in train_orientations
                                ) else "held_out")
                    learned_singletons += (feature["signals"] == 1 and
                                           1 in train_signals)
                orientations[split].add(orientation)
                aspect_axes[split].add((axis, aspect))
                signal_counts[split].add(feature["signals"])
                tick_counts[split].add(feature["ticks"])
                shapes.add(feature["shape"])
                dimension_ranges[split]["height"].add(height)
                dimension_ranges[split]["width"].add(width)
                puzzle_aspects.add(aspect)
                puzzle_signals.add(feature["signals"])
                puzzle_ticks.add(feature["ticks"])
                puzzle_patterns.add(feature["pattern"])
                patterns.add(feature["pattern"])
                clip_category = (
                    "none" if feature["clipped"] == 0 else
                    "all" if feature["clipped"] == feature["signals"] else "partial"
                )
                clip_categories.add(clip_category)
                pattern_clips.add((feature["pattern"], clip_category))
                clipped_counts.add(feature["clipped"])
                one_tick_clipping += feature["ticks"] == 1 and feature["clipped"] > 0
                multi_tick_unclipped += feature["ticks"] >= 2 and feature["clipped"] == 0
                multi_tick_empty += (feature["ticks"] >= 2 and
                                     feature["clipped"] == feature["signals"])
                for key, observed in (
                    ("equal", feature["equal_strips"]),
                    ("adjacent", feature["adjacent_ticks"]),
                    ("quiet", feature["long_quiet"]),
                    ("early", feature["near_start"]),
                    ("late", feature["near_end"]),
                ):
                    timing[split][key] += observed
                equal_strips += feature["equal_strips"]
                adjacent_ticks += feature["adjacent_ticks"]
                long_quiet += feature["long_quiet"]
                near_start += feature["near_start"]
                near_end += feature["near_end"]
        within_aspects += len(puzzle_aspects) >= 2
        within_signals += len(puzzle_signals) >= 2
        within_ticks += len(puzzle_ticks) >= 2
        within_patterns += len(puzzle_patterns) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) == 42
    assert len(first_orientations) >= 6
    assert len(orientations["train"]) == len(orientations["test"]) == 8
    assert len(aspect_axes["train"]) == len(aspect_axes["test"]) == 6
    assert len(shapes) >= 60
    for split in ("train", "test"):
        for axis in ("height", "width"):
            assert min(dimension_ranges[split][axis]) <= 8
            assert max(dimension_ranges[split][axis]) >= 28
        assert min(timing[split].values()) >= 3
    assert min(within_aspects, within_signals, within_ticks,
               within_patterns) >= 30
    assert len(signal_counts["train"]) >= 8 and len(signal_counts["test"]) >= 8
    assert len(tick_counts["train"]) >= 8 and len(tick_counts["test"]) >= 8
    assert 1 in signal_counts["test"] and 9 in signal_counts["test"]
    assert color_modes == {"fixed", "background", "signal", "control"}
    assert regimes == {"shared", "flipped_control", "held_out"}
    assert learned_singletons >= 3
    assert {"single", "run", "comb", "grouped", "irregular"} <= patterns
    assert clip_categories == {"none", "partial", "all"}
    assert {("comb", "none"), ("comb", "partial"),
            ("run", "partial"), ("grouped", "partial"),
            ("irregular", "partial"), ("single", "all")} <= pattern_clips
    assert one_tick_clipping >= 1 and multi_tick_unclipped >= 10
    assert multi_tick_empty >= 10
    assert len(clipped_counts) >= 4
    assert min(equal_strips, adjacent_ticks, long_quiet,
               near_start, near_end) >= 10
    for seed, train, test in (
        (8001, 2, 1), (8002, 10, 1), (8003, 10, 4),
        (8004, None, 1), (8005, None, 4),
        (8006, 2, None), (8007, 10, None),
    ):
        first = generate_puzzle(seed, train, test)
        second = generate_puzzle(seed, train, test)
        assert _fingerprint(first) == _fingerprint(second)
        if train is not None:
            assert len(first["train"]) == train
        if test is not None:
            assert len(first["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4)):
        try:
            generate_puzzle(9000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count request accepted")
