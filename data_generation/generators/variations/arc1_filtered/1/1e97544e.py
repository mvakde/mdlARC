"""Cyclic row pattern with a one-cell-per-row successor prefix and erasures."""

import hashlib
import json

import numpy as np


_SOURCE_PAIR_HASHES = frozenset({
    "dad870fac302f6272003982349acf19e3a877baa1c2a4ff7f1a643652d85ec92",
    "3cf3f070526024d097a2494e0e8b59b7706e78f994b2bab1525b99b3d9a2d25c",
    "c06c3f025c50d977e1129b632ad317b02783d34e55f68ec8d9d076b3e0fdf38b",
    "63fa21c96c65270662d880a20ffd213274699d4518266a071ecfcaf443204604",
})


def _source_hash(pair: dict) -> str:
    serial = {name: pair[name].tolist() for name in ("input", "output")}
    packed = json.dumps(serial, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode()).hexdigest()


def _validate_count(value, lower: int, upper: int) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not lower <= value <= upper
    ):
        raise ValueError(f"Count must be an integer from {lower} to {upper}")


def _lengths(rng: np.random.Generator, count: int) -> list[int]:
    # A source-like progressive set is one option, not the universal schedule.
    if count >= 3 and rng.random() < 0.32:
        values = [6, 7, 8]
        values += [int(rng.integers(4, 9)) for _ in range(count - 3)]
        if rng.random() < 0.35:
            rng.shuffle(values)
        return values
    maximum = int(rng.choice((6, 7, 8, 9), p=(0.17, 0.28, 0.43, 0.12)))
    minimum = int(rng.integers(4, maximum))
    values = [minimum, maximum]
    values += [int(rng.integers(4, maximum + 1)) for _ in range(count - 2)]
    rng.shuffle(values)
    return values


def _test_lengths(
    rng: np.random.Generator, train_lengths: list[int], count: int
) -> list[int]:
    maximum = max(train_lengths)
    values = []
    for _ in range(count):
        mode = int(rng.choice(3, p=(0.48, 0.32, 0.20)))
        if mode == 0 and maximum < 9:
            values.append(int(rng.integers(maximum + 1, min(9, maximum + 2) + 1)))
        elif mode == 2:
            values.append(int(rng.integers(4, maximum)))
        else:
            values.append(int(rng.choice(train_lengths)))
    return values


def _dimensions(rng: np.random.Generator, period: int) -> tuple[int, int]:
    minimum_width = max(14, 2 * period + 2)
    if rng.random() < 0.35:
        side = int(rng.integers(minimum_width, 31))
        return side, side
    height = int(rng.integers(12, 31))
    width = int(rng.integers(minimum_width, 31))
    if height == width:
        height = 12 if height > 20 else 30
    return height, width


def _fill_rule(cycle: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None]
    columns = np.arange(width)[None, :]
    indices = (columns + (columns < rows)) % len(cycle)
    return cycle[indices].astype(np.int8)


def _add_rectangle(
    mask: np.ndarray, row: int, col: int, height: int, width: int
) -> None:
    mask[row:min(mask.shape[0], row + height),
         col:min(mask.shape[1], col + width)] = True


def _visible_evidence(input_grid: np.ndarray, period: int, blank: int) -> bool:
    height, width = input_grid.shape
    if not np.any(input_grid[0, period:] != blank):
        return False
    if sum(
        np.any(input_grid[row, :row] != blank)
        and np.any(input_grid[row, row:] != blank)
        for row in range(1, min(height, width))
    ) < 3:
        return False
    return True


def _sample_mask(
    rng: np.random.Generator,
    shape: tuple[int, int],
    period: int,
    regime: str,
    top_erasure: bool,
) -> np.ndarray:
    height, width = shape
    for _ in range(100):
        mask = np.zeros(shape, dtype=bool)
        mode = int(rng.integers(0, 5))
        diagonal_clear = rng.random() < 0.28
        if regime == "source_train":
            fraction = float(rng.uniform(0.08, 0.25))
        else:
            fraction = float(rng.uniform(0.075, 0.30))
        target = max(8, int(round(height * width * fraction)))

        if mode == 0:  # several ordinary blocks
            for _ in range(int(rng.integers(2, 5))):
                _add_rectangle(mask, int(rng.integers(0, height)),
                               int(rng.integers(0, width)),
                               int(rng.integers(2, 7)), int(rng.integers(2, 8)))
        elif mode == 1:  # stepped left or right edge
            edge_left = bool(rng.integers(0, 2))
            start = int(rng.integers(1, height - 2))
            span = min(height - start, int(rng.integers(3, 9)))
            base = int(rng.integers(1, min(5, width // 3) + 1))
            for offset in range(span):
                reach = min(width - 2, base + (offset // 2))
                if edge_left:
                    mask[start + offset, :reach] = True
                else:
                    mask[start + offset, width - reach:] = True
        elif mode == 2:  # thin overlapping strips
            for _ in range(int(rng.integers(2, 5))):
                if rng.random() < 0.5:
                    _add_rectangle(mask, int(rng.integers(0, height)),
                                   int(rng.integers(0, width)),
                                   int(rng.integers(4, 11)), int(rng.integers(1, 4)))
                else:
                    _add_rectangle(mask, int(rng.integers(0, height)),
                                   int(rng.integers(0, width)),
                                   int(rng.integers(1, 4)), int(rng.integers(4, 11)))
        elif mode == 3:  # multiple compact, often separated patches
            for _ in range(int(rng.integers(4, 8))):
                _add_rectangle(mask, int(rng.integers(0, height)),
                               int(rng.integers(0, width)),
                               int(rng.integers(2, 5)), int(rng.integers(2, 5)))
        else:  # one larger area with smaller companions
            _add_rectangle(mask, int(rng.integers(0, height - 2)),
                           int(rng.integers(0, width - 2)),
                           int(rng.integers(4, 9)), int(rng.integers(4, 9)))

        if top_erasure:
            start = int(rng.integers(0, period))
            span = int(rng.integers(1, min(5, period - start) + 1))
            mask[0, start:start + span] = True
            if span == 1 and rng.random() < 0.30:
                mask[:int(rng.integers(2, min(6, height) + 1)), start] = True
        else:
            mask[0, :period] = False

        # A few area-aware companions keep the simple regimes readable while
        # the compact-patch regime can still make several separated holes.
        for _ in range(3 if mode == 3 else 4):
            remaining = target - int(mask.sum())
            if remaining <= 0:
                break
            side = max(2, min(9, int(np.sqrt(remaining))))
            patch_height = int(rng.integers(max(2, side - 1), min(11, side + 3)))
            patch_width = min(10, max(2, (remaining + patch_height - 1) // patch_height))
            row = int(rng.integers(1, max(2, height - patch_height + 1)))
            col = int(rng.integers(0, max(1, width - patch_width + 1)))
            _add_rectangle(mask, row, col, patch_height, patch_width)

        if diagonal_clear:
            np.fill_diagonal(mask, False)
        if not top_erasure:
            mask[0, :period] = False
        erased = int(mask.sum())
        if not (max(7, int(0.045 * height * width)) <= erased
                <= int(0.34 * height * width)):
            continue
        if top_erasure and not np.any(mask[0, :period]):
            continue
        if not top_erasure and np.any(mask[0, :period]):
            continue
        return mask
    raise RuntimeError("Could not make a bounded erasure mask")


def _infer_pair(
    input_grid: np.ndarray, blank: int
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Recover the cycle from visible cells, including later repetitions."""
    height, width = input_grid.shape
    colors = set(int(v) for v in input_grid.flat if v != blank)
    period = len(colors)
    assert 4 <= period <= 9
    assert width >= 2 * period + 2
    slots: list[int | None] = [None] * period
    observations = [0] * period
    for row in range(height):
        for col in range(width):
            value = int(input_grid[row, col])
            if value == blank:
                continue
            slot = (col + (col < row)) % period
            if slots[slot] is not None:
                assert slots[slot] == value
            slots[slot] = value
            observations[slot] += 1
    assert all(value is not None for value in slots)
    assert all(count >= 2 for count in observations)
    assert set(slots) == colors
    assert _visible_evidence(input_grid, period, blank)
    cycle = np.asarray(slots, dtype=np.int8)
    return _fill_rule(cycle, height, width), tuple(int(v) for v in cycle)


def _generate_pair(
    rng: np.random.Generator,
    order: tuple[int, ...],
    blank: int,
    period: int,
    regime: str,
    top_erasure: bool,
) -> dict:
    for _ in range(100):
        height, width = _dimensions(rng, period)
        phase = int(rng.integers(0, period))
        cycle = np.roll(np.asarray(order[:period], dtype=np.int8), -phase)
        output = _fill_rule(cycle, height, width)
        mask = _sample_mask(rng, (height, width), period, regime, top_erasure)
        input_grid = output.copy()
        input_grid[mask] = blank
        try:
            recovered, _ = _infer_pair(input_grid, blank)
        except AssertionError:
            continue
        if np.array_equal(recovered, output):
            return {"input": input_grid, "output": output}
    raise RuntimeError("Could not construct a recoverable pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    _validate_count(num_train, 2, 10)
    _validate_count(num_test, 1, 4)
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    rng = np.random.default_rng(seed)
    # The complete ordered palette is a puzzle-level convention. Every pair
    # takes a prefix of it, then supplies its own visible phase and period.
    if rng.random() < 0.12:
        blank = 0
        order = tuple(range(1, 10))
    else:
        blank = 0 if rng.random() < 0.50 else int(rng.integers(1, 10))
        palette = [color for color in range(10) if color != blank]
        order = tuple(int(v) for v in rng.permutation(palette))
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 0) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    train_lengths = _lengths(rng, num_train)
    test_lengths = _test_lengths(rng, train_lengths, num_test)
    top_first = bool(rng.integers(0, 2))
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    seen_outputs = set()
    seen_shapes = set()

    for section, lengths in (("train", train_lengths), ("test", test_lengths)):
        for index, period in enumerate(lengths):
            top_erasure = (
                (index % 2 == int(top_first)) if section == "train"
                else (rng.random() < 0.60)
            )
            for _ in range(120):
                pair_rng = np.random.default_rng(
                    int(rng.integers(0, np.iinfo(np.int64).max))
                )
                pair = _generate_pair(
                    pair_rng, order, blank, period,
                    "source_train" if section == "train" else "target_test",
                    top_erasure,
                )
                shape = pair["input"].shape
                input_key = (shape, pair["input"].tobytes())
                output_key = (shape, pair["output"].tobytes())
                if (input_key in seen_inputs or output_key in seen_outputs
                        or shape in seen_shapes or _source_hash(pair) in _SOURCE_PAIR_HASHES):
                    continue
                break
            else:
                raise RuntimeError("Could not construct a distinct puzzle pair")
            seen_inputs.add(input_key)
            seen_outputs.add(output_key)
            seen_shapes.add(shape)
            puzzle[section].append(pair)
    return puzzle


def _cyclic_equal(first: tuple[int, ...], second: tuple[int, ...]) -> bool:
    return len(first) == len(second) and any(
        first == second[offset:] + second[:offset]
        for offset in range(len(second))
    )


def _shared_palette_order(cycles: list[tuple[int, ...]]) -> bool:
    longest = max(cycles, key=len)
    shortest_set = set(min(cycles, key=len))
    for anchor in shortest_set:
        pivot = longest.index(anchor)
        order = longest[pivot:] + longest[:pivot]
        if all(
            set(cycle) == set(order[:len(cycle)])
            and _cyclic_equal(cycle, order[:len(cycle)])
            for cycle in cycles
        ):
            return True
    return False


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4
    assert train_count > test_count

    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        input_grid, output = pair["input"], pair["output"]
        for grid in (input_grid, output):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= extent <= 30 for extent in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((grid >= 0) & (grid <= 9))
        assert input_grid.shape == output.shape

    changed_train_colors = {
        int(value)
        for pair in puzzle["train"]
        for value in pair["input"][pair["input"] != pair["output"]]
    }
    assert len(changed_train_colors) == 1
    blank = changed_train_colors.pop()

    seen_inputs = set()
    seen_pairs = set()
    seen_outputs = set()
    seen_shapes = set()
    cycles = []
    train_periods = []
    test_records = []
    top_states = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            input_grid, output = pair["input"], pair["output"]
            shape = input_grid.shape
            input_key = (shape, input_grid.tobytes())
            output_key = (shape, output.tobytes())
            pair_key = (input_key, output_key)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert output_key not in seen_outputs
            assert shape not in seen_shapes
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            seen_outputs.add(output_key)
            seen_shapes.add(shape)
            assert _source_hash(pair) not in _SOURCE_PAIR_HASHES

            expected, cycle = _infer_pair(input_grid, blank)
            assert np.array_equal(expected, output)
            mask = input_grid == blank
            assert np.any(mask) and not np.any(output == blank)
            assert 0.045 <= float(mask.mean()) <= 0.34
            assert np.array_equal(input_grid[~mask], output[~mask])
            period = len(cycle)
            top_masked = bool(np.any(mask[0, :period]))
            cycles.append(cycle)
            if section == "train":
                train_periods.append(period)
                top_states.append(top_masked)
            else:
                test_records.append((period, top_masked, int(mask.sum()) / mask.size))

    assert len(set(train_periods)) >= 2
    assert len(set(top_states)) == 2
    assert _shared_palette_order(cycles)
    assert all(4 <= period <= 9 and 0.045 <= fraction <= 0.34
               for period, _, fraction in test_records)


def _signature(puzzle: dict) -> tuple:
    return tuple(
        (section, tuple(
            (pair["input"].shape, pair["input"].tobytes(),
             pair["output"].tobytes()) for pair in puzzle[section]
        )) for section in ("train", "test")
    )


def _component_count(mask: np.ndarray) -> int:
    remaining = set(map(tuple, np.argwhere(mask)))
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return components


def self_test() -> None:
    train_counts, test_counts = set(), set()
    periods, shapes, palette_sets, output_signatures = set(), set(), set(), set()
    top_hidden = top_clear = diagonal_clear = left_train = left_test = 0
    squares = rectangles = smaller_tests = extended_tests = 0
    train_areas, test_areas = set(), set()
    blank_colors = set()
    single_top = segment_top = separated_masks = 0
    combined_masks = test_top_clear = shorter_test_top_clear = 0
    lighter_test_puzzles = heavy_train_puzzles = repeated_period_puzzles = 0
    progressive_puzzles = canonical_six = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _signature(puzzle)
        assert signature == _signature(generate_puzzle(seed))
        signatures.add(signature)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        blank = int(puzzle["train"][0]["input"][
            puzzle["train"][0]["input"] != puzzle["train"][0]["output"]
        ][0])
        blank_colors.add(blank)
        max_train = max(len(set(int(v) for v in pair["output"].flat))
                        for pair in puzzle["train"])
        train_periods = [len(np.unique(pair["output"]))
                         for pair in puzzle["train"]]
        repeated_period_puzzles += int(len(set(train_periods)) < len(train_periods))
        progressive_puzzles += int(train_periods[:3] == [6, 7, 8])
        lighter_test_puzzles += int(
            min(np.mean(pair["input"] == blank) for pair in puzzle["test"])
            < max(np.mean(pair["input"] == blank) for pair in puzzle["train"])
        )
        heavy_train_puzzles += int(any(
            np.mean(pair["input"] == blank) >= 0.18
            for pair in puzzle["train"]
        ))
        for section in ("train", "test"):
            for pair in puzzle[section]:
                input_grid, output = pair["input"], pair["output"]
                expected, cycle = _infer_pair(input_grid, blank)
                assert np.array_equal(expected, output)
                period = len(cycle)
                # Check the output formula separately from visible-slot recovery.
                top_cycle = tuple(int(v) for v in output[0, :period])
                for row in range(output.shape[0]):
                    for col in range(output.shape[1]):
                        color = top_cycle[(col + (col < row)) % period]
                        assert int(output[row, col]) == color
                        assert int(input_grid[row, col]) in (blank, color)
                periods.add(period)
                shapes.add(output.shape)
                palette_sets.add(tuple(sorted(cycle)))
                output_signatures.add((output.shape, output.tobytes()))
                mask = input_grid == blank
                first_cycle_erased = int(mask[0, :period].sum())
                top_hidden += int(first_cycle_erased > 0)
                top_clear += int(first_cycle_erased == 0)
                single_top += int(first_cycle_erased == 1)
                segment_top += int(first_cycle_erased >= 2)
                separated_masks += int(_component_count(mask) >= 5)
                combined_masks += int(
                    first_cycle_erased > 0
                    and not np.any(np.diag(mask))
                    and np.any(mask[:, 0])
                )
                canonical_six += int(
                    period == 6
                    and _cyclic_equal(top_cycle, tuple(range(1, 7)))
                )
                diagonal_clear += int(not np.any(np.diag(mask)))
                squares += int(output.shape[0] == output.shape[1])
                rectangles += int(output.shape[0] != output.shape[1])
                if section == "train":
                    left_train += int(np.any(mask[:, 0]))
                    train_areas.add(int(mask.sum()))
                else:
                    left_test += int(np.any(mask[:, 0]))
                    test_top_clear += int(first_cycle_erased == 0)
                    shorter_test_top_clear += int(
                        first_cycle_erased == 0 and period <= max_train
                    )
                    test_areas.add(int(mask.sum()))
                    smaller_tests += int(period <= max_train)
                    extended_tests += int(period > max_train)
    assert len(signatures) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert periods == set(range(4, 10))
    assert len(shapes) >= 60 and len(palette_sets) >= 40
    assert len(blank_colors) >= 4
    assert len(output_signatures) >= 150
    assert min(top_hidden, top_clear, diagonal_clear, left_train, left_test) >= 8
    assert min(single_top, segment_top, separated_masks) >= 8
    assert combined_masks >= 2 and test_top_clear >= 8
    assert shorter_test_top_clear >= 2
    assert min(squares, rectangles, smaller_tests, extended_tests) >= 8
    assert len(train_areas) >= 20 and len(test_areas) >= 15
    assert lighter_test_puzzles >= 8 and heavy_train_puzzles >= 8
    assert repeated_period_puzzles >= 8
    assert progressive_puzzles >= 1 and canonical_six >= 1

    cases = ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
             (2, None), (10, None))
    for offset, (train, test) in enumerate(cases):
        puzzle = generate_puzzle(1000 + offset, train, test)
        validate_generated_puzzle(puzzle)
        assert _signature(puzzle) == _signature(
            generate_puzzle(1000 + offset, train, test))
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {(train, test)}")
