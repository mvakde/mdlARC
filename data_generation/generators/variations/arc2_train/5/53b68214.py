import numpy as np


_WIDTHS = (7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
_DRIFT_WIDTHS = (9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)

_SOURCE_TASK_PAIRS = (
    (
        (
            "0200000000",
            "0200000000",
            "2020000000",
            "0200000000",
            "0200000000",
            "2020000000",
            "0200000000",
            "0200000000",
        ),
        (
            "0200000000",
            "0200000000",
            "2020000000",
            "0200000000",
            "0200000000",
            "2020000000",
            "0200000000",
            "0200000000",
            "2020000000",
            "0200000000",
        ),
    ),
    (
        (
            "0030000000",
            "0030000000",
            "0030000000",
            "0030000000",
            "0030000000",
        ),
        (
            "0030000000",
            "0030000000",
            "0030000000",
            "0030000000",
            "0030000000",
            "0030000000",
            "0030000000",
            "0030000000",
            "0030000000",
            "0030000000",
        ),
    ),
    (
        (
            "1110000000",
            "0010000000",
            "0011100000",
            "0000100000",
            "0000111000",
            "0000001000",
        ),
        (
            "1110000000",
            "0010000000",
            "0011100000",
            "0000100000",
            "0000111000",
            "0000001000",
            "0000001110",
            "0000000010",
            "0000000011",
            "0000000000",
        ),
    ),
    (
        (
            "0888000000",
            "0808000000",
            "0888000000",
            "0808000000",
            "0888000000",
        ),
        (
            "0888000000",
            "0808000000",
            "0888000000",
            "0808000000",
            "0888000000",
            "0808000000",
            "0888000000",
            "0808000000",
            "0888000000",
            "0808000000",
        ),
    ),
    (
        (
            "0000600000",
            "0006000000",
            "0000600000",
            "0006000000",
            "0000600000",
            "0006000000",
            "0000600000",
            "0006000000",
        ),
        (
            "0000600000",
            "0006000000",
            "0000600000",
            "0006000000",
            "0000600000",
            "0006000000",
            "0000600000",
            "0006000000",
            "0000600000",
            "0006000000",
        ),
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_optional_count(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_optional_count(num_train, "num_train", 2, 10)
    num_test = _validate_optional_count(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        lower = num_test + 1
        span = 11 - lower
        num_train = lower + ((seed // 4 + 3 * seed) % span)
    elif num_train is None:
        lower = num_test + 1
        span = 11 - lower
        num_train = lower + seed % span
    elif num_test is None:
        upper = min(4, num_train - 1)
        num_test = 1 + seed % upper

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return int(num_train), int(num_test)


def _shift_row(row: np.ndarray, displacement: int, background: int) -> np.ndarray:
    width = row.shape[0]
    result = np.full(width, background, dtype=np.int8)
    if displacement == 0:
        result[:] = row
    elif displacement > 0:
        result[displacement:] = row[:width - displacement]
    else:
        distance = -displacement
        result[:width - distance] = row[distance:]
    return result


def _grid_signature(grid: np.ndarray) -> tuple:
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _pair_signature(pair: dict) -> tuple:
    return _grid_signature(pair["input"]), _grid_signature(pair["output"])


def _source_copy(pair: dict) -> bool:
    for input_rows, output_rows in _SOURCE_TASK_PAIRS:
        if pair["input"].shape != (len(input_rows), len(input_rows[0])):
            continue
        source_input = np.array([[int(c) for c in row] for row in input_rows], dtype=np.int8)
        source_output = np.array([[int(c) for c in row] for row in output_rows], dtype=np.int8)
        if np.array_equal(pair["input"], source_input) and np.array_equal(pair["output"], source_output):
            return True
    return False


def _colors(grid: np.ndarray) -> tuple[int, int]:
    values, counts = np.unique(grid, return_counts=True)
    _require(len(values) == 2, "input needs exactly two colors")
    _require(counts[0] != counts[1], "background must be identifiable")
    index = int(np.argmax(counts))
    return int(values[index]), int(values[1 - index])


def _infer_recurrence(grid: np.ndarray, background: int) -> tuple[int, int]:
    height, width = grid.shape
    for period in range(1, height // 2 + 1):
        matches = []
        for shift in range(-(width - 1), width):
            if not np.array_equal(_shift_row(grid[0], shift, background), grid[period]):
                continue
            if all(np.array_equal(_shift_row(grid[r - period], shift, background), grid[r])
                   for r in range(period + 1, height)):
                matches.append(shift)
        if matches:
            _require(len(matches) == 1, "ambiguous minimal displacement")
            return period, matches[0]
    raise AssertionError("no supported recurrence")


def _analyze_pair(pair: dict) -> dict:
    inp, out = pair["input"], pair["output"]
    height, width = inp.shape
    _require(height < width and out.shape == (width, width), "wrong square completion dimensions")
    _require(np.array_equal(inp, out[:height]), "input rows changed")
    background, foreground = _colors(inp)
    _require(set(np.unique(out)) == {background, foreground}, "color roles changed")
    period, shift = _infer_recurrence(inp, background)
    _require(height >= 2 * period + 1, "insufficient repeated phases")
    _require(all(np.any(inp[r] == foreground) for r in range(period)), "blank initial phase")
    expected = np.empty_like(out)
    expected[:height] = inp
    for r in range(height, width):
        expected[r] = _shift_row(expected[r - period], shift, background)
    _require(np.array_equal(out, expected), "incorrect continuation")
    if shift:
        _require(all(np.count_nonzero(inp[r] == foreground) == np.count_nonzero(inp[r + period] == foreground)
                     for r in range(period)), "every drifting phase needs an intact first repeat")
    return {"height": height, "width": width, "period": period, "shift": shift,
            "background": background, "foreground": foreground}


def _row_positions(rng: np.random.Generator, width: int, low: int, high: int,
                   count: int, force_high: bool = False) -> np.ndarray:
    # Samples bars, regular spacing, and scattered cells. The optional high
    # anchor makes the first clipping event occur at a chosen repetition.
    count = min(count, high - low + 1)
    if force_high:
        # Keep a low cell visible when the anchored edge cell clips.
        if count == 2:
            return np.array([low, high])
        middle = rng.choice(np.arange(low + 1, high), size=count - 2, replace=False)
        return np.sort(np.concatenate(([low], middle, [high])))
    if count == 1:
        return np.array([high if force_high else int(rng.integers(low, high + 1))])
    mode = int(rng.integers(0, 4))
    if mode == 0:
        end = high if force_high else int(rng.integers(low + count - 1, high + 1))
        return np.arange(end - count + 1, end + 1)
    if mode == 1:
        step = int(rng.integers(1, min(5, (high - low) // (count - 1)) + 1))
        end = high if force_high else int(rng.integers(low + step * (count - 1), high + 1))
        return end - step * np.arange(count - 1, -1, -1)
    pool = np.arange(low, high + (0 if force_high else 1))
    take = rng.choice(pool, size=count - int(force_high), replace=False)
    if force_high:
        take = np.append(take, high)
    return np.sort(take)


def _template_row(rng: np.random.Generator, width: int, background: int,
                  foreground: int, shift: int, high: int | None = None,
                  force_high: bool = False) -> np.ndarray:
    row = np.full(width, background, dtype=np.int8)
    if high is None:
        high = width - 1
    available = high + 1
    # A broad range includes singleton lines and dense bars on wide grids.
    if force_high:
        count = int(rng.integers(2, min(available, max(3, int(width * .43))) + 1))
    elif rng.random() < .18:
        count = 1
    else:
        count = int(rng.integers(2, min(available, max(3, int(width * .43))) + 1))
    positions = _row_positions(rng, width, 0, high, count, force_high)
    if shift < 0:
        positions = width - 1 - positions
    row[positions] = foreground
    return row


def _choose_geometry(rng: np.random.Generator, period: int, shift: int,
                     clip_mode: str) -> tuple[int, int, int]:
    for _ in range(60):
        width = int(rng.integers(max(9, 2 * period + 3), 31))
        # Both short and long prefixes occur; drifting inputs sometimes expose
        # a third repetition so that clipping itself can be seen.
        low_height = 2 * period + 1
        if clip_mode == "input_partial":
            low_height = 3 * period + 1
        if low_height >= width:
            continue
        if rng.random() < .5:
            height = int(rng.integers(low_height, min(width - 1, low_height + period + 2) + 1))
        else:
            height = int(rng.integers(low_height, width))
        if shift == 0 or clip_mode in ("input_partial", "blank_tail"):
            return width, height, shift
        repeats = (width - 1) // period
        if clip_mode == "no_clip" and abs(shift) * repeats <= width - 3:
            return width, height, shift
        if clip_mode in ("output_partial", "near_final"):
            input_repeats = (height - 1) // period
            if input_repeats < repeats and abs(shift) * input_repeats <= width - 3:
                return width, height, shift
    raise ValueError("infeasible geometry")


def _candidate_pair(rng: np.random.Generator, period: int, shift: int,
                    background: int, foreground: int, clip_mode: str) -> dict:
    width, height, shift = _choose_geometry(rng, period, shift, clip_mode)
    distance = abs(shift)
    templates = []
    if period == 2 and shift == 0 and rng.random() < .14:
        # The source tests include two one-cell phases at different columns.
        columns = rng.choice(width, size=2, replace=False)
        for column in columns:
            row = np.full(width, background, dtype=np.int8)
            row[int(column)] = foreground
            templates.append(row)
    for phase in range(period):
        if len(templates) == period:
            break
        repeats_in = (height - 1 - phase) // period
        repeats_out = (width - 1 - phase) // period
        high = None
        force = False
        if shift:
            if clip_mode == "no_clip":
                high = width - 1 - repeats_out * distance
            elif clip_mode == "input_partial" and phase == 0:
                clip_at = int(rng.integers(2, repeats_in + 1))
                high = int(rng.integers(max(1, width - clip_at * distance),
                                        min(width - 2, width - (clip_at - 1) * distance - 1) + 1))
                force = True
            elif clip_mode in ("output_partial", "near_final") and phase == 0:
                clip_at = repeats_out if clip_mode == "near_final" else int(rng.integers(repeats_in + 1, repeats_out + 1))
                high = int(rng.integers(max(1, width - clip_at * distance),
                                        min(width - 2, width - (clip_at - 1) * distance - 1) + 1))
                force = True
            elif clip_mode == "blank_tail":
                high = min(width - 1 - distance, max(2, width - 2 * distance))
            else:
                high = max(1, width - 1 - repeats_out * distance)
            if high < 1:
                raise ValueError("no room for drifting template")
        if period >= 3 and shift == 0 and phase > 0 and rng.random() < .23:
            templates.append(templates[-1].copy())
        else:
            templates.append(_template_row(rng, width, background, foreground, shift, high, force))
    full = np.full((width, width), background, dtype=np.int8)
    full[:period] = templates
    for r in range(period, width):
        full[r] = _shift_row(full[r - period], shift, background)
    return {"input": full[:height].copy(), "output": full}


def _make_pair(rng: np.random.Generator, period: int, shift: int,
               background: int, foreground: int, clip_mode: str) -> dict:
    for _ in range(120):
        try:
            pair = _candidate_pair(rng, period, shift, background, foreground, clip_mode)
            info = _analyze_pair(pair)
        except (AssertionError, ValueError):
            continue
        if info["period"] != period or info["shift"] != shift:
            continue
        if shift:
            inp, out = pair["input"], pair["output"]
            counts = np.count_nonzero(inp != background, axis=1)
            output_counts = np.count_nonzero(out != background, axis=1)
            partial_in = any(0 < counts[r + period] < counts[r]
                             for r in range(inp.shape[0] - period))
            partial_out = any(0 < output_counts[r + period] < output_counts[r]
                              for r in range(inp.shape[0] - period, out.shape[0] - period))
            if clip_mode == "input_partial" and not partial_in:
                continue
            if clip_mode in ("output_partial", "near_final") and (partial_in or not partial_out):
                continue
            if clip_mode == "no_clip" and np.any(output_counts == 0):
                continue
            if clip_mode == "blank_tail" and not np.any(output_counts[-period:] == 0):
                continue
        return pair
    raise RuntimeError(f"could not construct period={period}, shift={shift}, clip={clip_mode}")


def _sample_regime(rng: np.random.Generator, stationary: bool | None = None) -> tuple[int, int, str]:
    if stationary is None:
        stationary = bool(rng.random() < .52)
    period = int(rng.choice([1, 2, 2, 3, 3, 4]))
    if stationary:
        return period, 0, "stationary"
    clip_mode = str(rng.choice(["no_clip", "input_partial", "output_partial", "near_final", "blank_tail"],
                               p=[.27, .17, .30, .16, .10]))
    distance = int(rng.choice([1, 1, 2, 2, 3, 4, 5]))
    if clip_mode != "blank_tail" and distance > period:
        distance = int(rng.integers(1, period + 1))
    shift = distance if rng.random() < .5 else -distance
    if clip_mode == "no_clip" and distance >= period:
        clip_mode = "output_partial"
    if clip_mode == "near_final" and distance >= period:
        clip_mode = "output_partial"
    return period, shift, clip_mode


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # The only whole-puzzle choices are the mix of recurrence examples and a
    # palette style. Colors themselves have no learned mapping: each input
    # identifies its own background and foreground.
    shared_background = bool(rng.random() < .42)
    base_background = int(rng.integers(0, 10))
    reuse_foreground = bool(rng.random() < .5)
    favored_foreground = int(rng.choice([c for c in range(10) if c != base_background]))
    source_like = num_train >= 3 and bool(rng.random() < .18)
    if source_like:
        train_regimes = [(3, 0, "stationary"), (1, 0, "stationary"), (2, 2, "output_partial")]
        test_regimes = [(2, 0, "stationary")]
    else:
        mandatory_drift = _sample_regime(rng, False)
        if mandatory_drift[2] == "no_clip":
            mandatory_drift = (mandatory_drift[0], mandatory_drift[1], "output_partial")
        train_regimes = [_sample_regime(rng, True), mandatory_drift]
        test_regimes = []
    while len(train_regimes) < num_train:
        train_regimes.append(_sample_regime(rng))
    while len(test_regimes) < num_test:
        test_regimes.append(_sample_regime(rng))
    rng.shuffle(train_regimes)
    rng.shuffle(test_regimes)
    puzzle = {"train": [], "test": []}
    used = set()
    for section, regimes in (("train", train_regimes), ("test", test_regimes)):
        for period, shift, clip_mode in regimes:
            for _ in range(120):
                background = base_background if shared_background else int(rng.integers(0, 10))
                foreground = favored_foreground if reuse_foreground and favored_foreground != background and rng.random() < .7 else int(rng.choice([c for c in range(10) if c != background]))
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                try:
                    pair = _make_pair(pair_rng, period, shift, background, foreground, clip_mode)
                except RuntimeError:
                    continue
                key = _grid_signature(pair["input"])
                if key in used or _source_copy(pair):
                    continue
                used.add(key)
                puzzle[section].append(pair)
                break
            else:
                raise RuntimeError("failed to construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest, "illegal pair counts")
    inputs, pairs = set(), set()
    infos = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D NumPy array")
                _require(all(1 <= n <= 30 for n in grid.shape), "illegal grid size")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "illegal ARC color")
            key = _grid_signature(pair["input"])
            whole = _pair_signature(pair)
            _require(key not in inputs and whole not in pairs, "duplicate pair")
            _require(not _source_copy(pair), "source pair copied")
            inputs.add(key)
            pairs.add(whole)
            infos[section].append(_analyze_pair(pair))
    _require(any(i["shift"] == 0 for i in infos["train"]), "train needs stationary example")
    _require(any(i["shift"] != 0 for i in infos["train"]), "train needs drifting example")
    _require(any(i["shift"] != 0 and any(
        np.count_nonzero(pair["output"][r] != i["background"]) <
        np.count_nonzero(pair["output"][r - i["period"]] != i["background"])
        for r in range(i["period"], i["width"]))
        for pair, i in zip(puzzle["train"], infos["train"])),
        "training must demonstrate boundary clipping")
    # Both families are taught in training; every test input supplies its own
    # uniquely inferred minimal period and displacement.
    for info in infos["test"]:
        _require(info["period"] <= 4 and abs(info["shift"]) <= 5, "test regime outside demonstrated family")


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, widths, regimes, layouts = set(), set(), set(), set(), set()
    left, right, stationary, equal_counts, adjacent_equal = 0, 0, 0, 0, 0
    backgrounds_vary, palettes_overlap = 0, 0
    input_clip = output_clip = no_clip = final_clip = blank_tail = 0
    wide_dense = long_bar = large_shift = 0
    stationary_two_then_drift = test_one_or_three = mixed_tests = mixed_widths = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_infos = [_analyze_pair(p) for p in puzzle["train"]]
        all_pairs = puzzle["train"] + puzzle["test"]
        all_infos = train_infos + [_analyze_pair(p) for p in puzzle["test"]]
        assert any(i["shift"] == 0 for i in train_infos)
        assert any(i["shift"] != 0 for i in train_infos)
        backgrounds_vary += len({i["background"] for i in all_infos}) > 1
        palettes_overlap += bool({i["foreground"] for i in train_infos} &
                                 {i["foreground"] for i in all_infos[len(train_infos):]})
        test_infos = all_infos[len(train_infos):]
        stationary_two_then_drift += any(i["period"] == 2 and i["shift"] == 0 for i in train_infos) and any(i["shift"] for i in test_infos)
        test_one_or_three += any(i["period"] in (1, 3) for i in test_infos)
        mixed_tests += len({(i["period"], i["shift"]) for i in test_infos}) > 1
        mixed_widths += len({i["width"] for i in all_infos}) > 1
        for pair, info in zip(all_pairs, all_infos):
            widths.add(info["width"])
            regimes.add((info["period"], info["shift"]))
            layouts.add((info["period"], info["shift"], info["height"], info["width"],
                         tuple(tuple(np.flatnonzero(row != info["background"])) for row in pair["input"][:info["period"]])))
            left += info["shift"] < 0
            right += info["shift"] > 0
            stationary += info["shift"] == 0
            counts = [int(np.count_nonzero(row != info["background"])) for row in pair["input"][:info["period"]]]
            equal_counts += info["period"] > 1 and len(set(counts)) < len(counts)
            adjacent_equal += info["period"] >= 3 and any(np.array_equal(pair["input"][r], pair["input"][r + 1])
                                                             for r in range(info["period"] - 1))
            wide_dense += info["width"] >= 20 and max(counts) >= 6
            long_bar += any(any(np.all(row[k:k + 6] != info["background"])
                                for k in range(info["width"] - 5))
                            for row in pair["input"][:info["period"]])
            large_shift += abs(info["shift"]) >= 4
            if info["shift"]:
                out_counts = np.count_nonzero(pair["output"] != info["background"], axis=1)
                losses = [r for r in range(info["period"], info["width"])
                          if out_counts[r] < out_counts[r - info["period"]]]
                partial = [r for r in losses if out_counts[r] > 0]
                input_clip += any(r < info["height"] for r in partial)
                output_clip += any(r >= info["height"] for r in partial)
                no_clip += not losses
                final_clip += bool(partial and min(partial) >= info["width"] - info["period"])
                blank_tail += out_counts[-1] == 0
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(widths) >= 12 and 10 in widths and max(widths) >= 26
    assert len(layouts) >= 170 and len(regimes) >= 12
    assert min(left, right, stationary) >= 8
    assert equal_counts >= 8 and adjacent_equal >= 1
    assert backgrounds_vary >= 8 and palettes_overlap >= 4
    assert input_clip >= 5 and output_clip >= 5 and no_clip >= 3
    assert final_clip >= 3 and blank_tail >= 1
    assert wide_dense >= 20 and long_bar >= 10 and large_shift >= 2
    assert stationary_two_then_drift >= 5 and test_one_or_three >= 5
    assert mixed_tests >= 5 and mixed_widths >= 35
    for index, (a, b) in enumerate(((2, 1), (10, 1), (10, 4), (6, 2),
                                    (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + index, a, b)
        validate_generated_puzzle(puzzle)
        assert (a is None or len(puzzle["train"]) == a) and (b is None or len(puzzle["test"]) == b)
    assert _puzzles_equal(generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3))
    assert not _puzzles_equal(generate_puzzle(314159, 7, 3), generate_puzzle(314160, 7, 3))
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(4000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
