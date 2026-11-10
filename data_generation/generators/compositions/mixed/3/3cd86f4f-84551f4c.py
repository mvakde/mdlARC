"""Exact-contact rod toppling, followed by a bottom-anchored row shear.

Only NumPy and the standard library are used. Generation and self_test perform
no file I/O. Each call holds at most fourteen pairs; rejection attempts retain
only the current candidate. See task_spec.md for the complete family domain.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


TRIGGER = 1


def _integer(value, name, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its legal range")
    return value


def _grid(value):
    grid = np.asarray(value)
    if grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape):
        raise ValueError("grids must be two-dimensional with dimensions 1..30")
    if not np.issubdtype(grid.dtype, np.integer) or np.any((grid < 0) | (grid > 9)):
        raise ValueError("grid cells must be integers 0..9")
    return grid.astype(np.int8, copy=False)


def _parse(input_grid):
    """Read (column, height, color) rods using only the visible input."""
    grid = _grid(input_grid)
    height, width = grid.shape
    background = int(grid[0, 0])
    if background == TRIGGER or not np.all(grid[0] == background):
        raise ValueError("the top row must be blank, with background different from 1")
    if height + width - 1 > 30:
        raise ValueError("the widened output would exceed 30 columns")
    rods = []
    passive_colors = set()
    for column in range(width):
        rr = np.flatnonzero(grid[:, column] != background)
        if not len(rr):
            continue
        rod_height = len(rr)
        color = int(grid[rr[0], column])
        if rod_height < 2 or not np.array_equal(rr, np.arange(height - rod_height, height)):
            raise ValueError("each rod must have height >=2 and rest on the bottom row")
        if not np.all(grid[rr, column] == color):
            raise ValueError("rods must be monochromatic")
        if column + rod_height > width:
            raise ValueError("every rod's full landing interval must fit")
        if rods and column < rods[-1][0] + rods[-1][1]:
            raise ValueError("rods cannot lie inside another rod's landing interval")
        rods.append((column, rod_height, color))
        if color != TRIGGER:
            passive_colors.add(color)
    if not any(color == TRIGGER for _, _, color in rods) or len(passive_colors) > 1:
        raise ValueError("at least one trigger and at most one passive color are required")
    return grid, background, rods


def _active(rods):
    """Least closure of triggers under exact rightward endpoint contact."""
    at_column = {column: i for i, (column, _, _) in enumerate(rods)}
    active = {i for i, (_, _, color) in enumerate(rods) if color == TRIGGER}
    pending = list(active)
    while pending:
        i = pending.pop()
        column, rod_height, _ = rods[i]
        successor = at_column.get(column + rod_height)
        if successor is not None and successor not in active:
            active.add(successor)
            pending.append(successor)
    return active


def _topple(grid, background, rods, active):
    result = grid.copy()
    height = grid.shape[0]
    for i in active:
        column, rod_height, _ = rods[i]
        result[height - rod_height:, column] = background
    for i in active:
        column, rod_height, color = rods[i]
        result[-1, column:column + rod_height] = color
    return result


def _shear(grid, background):
    height, width = grid.shape
    result = np.full((height, width + height - 1), background, dtype=np.int8)
    for row in range(height):
        start = height - 1 - row
        result[row, start:start + width] = grid[row]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve any input in the specified family, without puzzle/seed context."""
    grid, background, rods = _parse(input_grid)
    return _shear(_topple(grid, background, rods, _active(rods)), background)


def _make_pair(height, width, rods, background):
    grid = np.full((height, width), background, dtype=np.int8)
    for column, rod_height, color in rods:
        grid[-rod_height:, column] = color
    return {"input": grid, "output": solve(grid)}


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _anchors(rng, background, passive):
    """Related hit/miss layouts: height alone changes the core contact graph.

    A leading passive and an independent final trigger also teach direction,
    selective preservation, multiple initiators, and nonzero row displacement.
    The canvas heights differ by one to expose bottom anchoring and full size.
    """
    for _ in range(128):
        short = int(rng.integers(2, 5))
        long = short + 1
        chain_count = int(rng.choice((3, 3, 4)))
        lead_height = int(rng.integers(2, 5))
        lead_column = int(rng.integers(0, 3))
        main_column = lead_column + lead_height + int(rng.integers(0, 3))
        downstream = []
        cursor = main_column + long
        for _ in range(chain_count - 1):
            rod_height = int(rng.integers(2, 5))
            downstream.append((cursor, rod_height, passive))
            cursor += rod_height
        tail_height = int(rng.integers(2, 5))
        tail = (cursor + int(rng.integers(1, 4)), tail_height, TRIGGER)
        maximum = max(long, lead_height, tail_height, *(h for _, h, _ in downstream))
        height = maximum + int(rng.integers(1, 4))
        width = tail[0] + tail_height + int(rng.integers(0, 4))
        if height + width > 30:
            continue
        pairs = []
        taller_canvas_for_hit = bool(rng.integers(0, 2))
        for hit in (False, True):
            rods = [(lead_column, lead_height, passive),
                    (main_column, long if hit else short, TRIGGER)] + downstream + [tail]
            canvas_height = height + int(hit == taller_canvas_for_hit)
            pairs.append(_make_pair(canvas_height, width, rods, background))
        rng.shuffle(pairs)
        return pairs
    raise RuntimeError("anchor sampling exhausted its bounded attempts")


def _longest_chain(rods):
    by_column = {column: i for i, (column, _, _) in enumerate(rods)}
    longest = 0
    for i, (_, _, color) in enumerate(rods):
        if color != TRIGGER:
            continue
        length = 0
        while i is not None:
            column, rod_height, _ = rods[i]
            length += 1
            i = by_column.get(column + rod_height)
        longest = max(longest, length)
    return longest


def _general_pair(rng, background, passive, training):
    for _ in range(256):
        count = int(rng.integers(1, 9))
        heights = [int(x) for x in rng.integers(2, 7, size=count)]
        colors = [TRIGGER if rng.random() < 0.4 else passive for _ in heights]
        colors[int(rng.integers(count))] = TRIGGER
        cursor = int(rng.integers(0, 5))
        rods = []
        for rod_height, color in zip(heights, colors):
            rods.append((cursor, rod_height, color))
            cursor += rod_height + int(rng.choice((0, 0, 0, 1, 1, 2, 3)))
        end = rods[-1][0] + rods[-1][1]
        height = max(heights) + int(rng.integers(1, 6))
        width = end + int(rng.integers(0, 5))
        if height + width - 1 > 30 or (training and _longest_chain(rods) > 3):
            continue
        return _make_pair(height, width, rods, background)
    raise RuntimeError("general sampling exhausted its bounded attempts")


def _shifted_pair(rng, background, passive, train_max_height, train_max_chain, tall):
    """Extend either recursive depth or the length of a surviving diagonal."""
    for _ in range(256):
        if tall:
            chain_count = int(rng.integers(2, 4))
            survivor_height = train_max_height + int(rng.integers(1, 3))
        else:
            chain_count = int(rng.integers(train_max_chain + 1, min(7, train_max_chain + 3) + 1))
            survivor_height = int(rng.integers(2, min(5, train_max_height) + 1))
        chain_heights = [int(x) for x in rng.integers(2, min(4, train_max_height) + 1,
                                                    size=chain_count)]
        chain = [(chain_heights[0], TRIGGER)] + [(h, passive) for h in chain_heights[1:]]
        components = [chain, [(survivor_height, passive)]]
        if rng.random() < 0.35:
            components.append([(int(rng.integers(2, 5)), TRIGGER)])
        rng.shuffle(components)
        rods = []
        cursor = int(rng.integers(0, 4))
        for component in components:
            for rod_height, color in component:
                rods.append((cursor, rod_height, color))
                cursor += rod_height
            cursor += int(rng.integers(1, 4))
        end = rods[-1][0] + rods[-1][1]
        height = max(h for _, h, _ in rods) + int(rng.integers(1, 5))
        width = end + int(rng.integers(0, 4))
        if height + width - 1 <= 30:
            return _make_pair(height, width, rods, background)
    raise RuntimeError("shift sampling exhausted its bounded attempts")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    # Color 1 is a fixed family convention, so solve never needs training data.
    background = int(rng.choice((0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9)))
    passive = int(rng.choice([c for c in range(10) if c not in (background, TRIGGER)]))
    puzzle = {"train": _anchors(rng, background, passive), "test": []}
    seen = {_key(pair["input"]) for pair in puzzle["train"]}

    def append_distinct(split, maker):
        for _ in range(64):
            pair = maker()
            signature = _key(pair["input"])
            if signature not in seen:
                seen.add(signature)
                puzzle[split].append(pair)
                return
        raise RuntimeError("distinct-pair sampling exhausted its bounded attempts")

    while len(puzzle["train"]) < num_train:
        append_distinct("train", lambda: _general_pair(rng, background, passive, True))
    training_rods = [_parse(pair["input"])[2] for pair in puzzle["train"]]
    max_height = max(h for rods in training_rods for _, h, _ in rods)
    max_chain = max(_longest_chain(rods) for rods in training_rods)
    for index in range(num_test):
        # The first test always makes a structural extrapolation. Other tests
        # can be simpler applications, including wholly active arrangements.
        mode = int(rng.integers(0, 2 if index == 0 else 3))
        if mode == 2:
            append_distinct("test", lambda: _general_pair(rng, background, passive, False))
        else:
            append_distinct("test", lambda: _shifted_pair(
                rng, background, passive, max_height, max_chain, mode == 1))
    validate_generated_puzzle(puzzle)
    return puzzle


def _coordinate_check(grid, output, background, rods):
    """Check via an independent forward scan and a final coordinate ledger.

    Nonoverlap makes only the immediately preceding rod able to activate a rod.
    This does not call solve, its closure, toppling, or row-copy implementation.
    """
    height, width = grid.shape
    assert output.shape == (height, width + height - 1)
    flags = []
    for i, (column, _, color) in enumerate(rods):
        flags.append(color == TRIGGER or bool(i and flags[i - 1] and
                     rods[i - 1][0] + rods[i - 1][1] == column))
    expected = np.full(output.shape, background, dtype=np.int8)
    for fallen, (column, rod_height, color) in zip(flags, rods):
        for offset in range(rod_height):
            row = height - 1 if fallen else height - 1 - offset
            expected[row, column + offset] = color
    assert np.array_equal(output, expected), "incorrect final rod coordinates"
    for color in set(int(c) for c in np.unique(grid)) - {background}:
        assert np.count_nonzero(grid == color) == np.count_nonzero(output == color)
    for row in range(height):
        start = height - 1 - row
        assert np.all(output[row, :start] == background)
        assert np.all(output[row, start + width:] == background)
    assert np.all(output[-1, width:] == background)
    return flags


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, palettes, train_shapes = set(), set(), set()
    train_rods, test_records = [], []
    active_passive = inactive_passive = recursive = one_cell_miss = multiple = False
    omit_a_changes = omit_b_changes = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, background, rods = _parse(pair["input"])
            output = _grid(pair["output"])
            signature = _key(grid)
            assert signature not in seen, "duplicate inputs"
            seen.add(signature)
            palettes.add(background)
            flags = _coordinate_check(grid, output, background, rods)
            assert np.array_equal(solve(grid), output)
            if split == "train":
                train_rods.append(rods)
                train_shapes.add(grid.shape)
                multiple |= sum(c == TRIGGER for _, _, c in rods) >= 2
                for i, (column, rod_height, color) in enumerate(rods):
                    active_passive |= color != TRIGGER and flags[i]
                    inactive_passive |= color != TRIGGER and not flags[i]
                    if i + 1 < len(rods):
                        nc, _, ncolor = rods[i + 1]
                        recursive |= flags[i] and color != TRIGGER and nc == column + rod_height
                        one_cell_miss |= (flags[i] and nc == column + rod_height + 1 and
                                          ncolor != TRIGGER and not flags[i + 1])
                # Give the no-A alternative the correct width, so failure must
                # involve visible cell placement rather than just dimensions.
                stage = _topple(grid, background, rods, _active(rods))
                no_a = np.full(output.shape, background, dtype=np.int8)
                no_a[:, :grid.shape[1]] = stage
                omit_a_changes |= not np.array_equal(no_a, output)
                omit_b_changes |= not np.array_equal(_shear(grid, background), output)
            else:
                test_records.append((rods, flags))
    assert len(palettes) == 1, "background must be consistent within a puzzle"
    background = next(iter(palettes))
    passive_colors = {int(c) for s in ("train", "test") for p in puzzle[s]
                      for c in np.unique(p["input"])} - {TRIGGER, background}
    assert len(passive_colors) == 1, "one shared passive role color per puzzle"
    assert len(train_shapes) >= 2
    assert active_passive and inactive_passive and recursive and one_cell_miss and multiple
    assert omit_a_changes and omit_b_changes
    # Related layouts must show a short trigger missing and a trigger exactly
    # one cell taller reaching the same passive at the same column.
    contrast = False
    for short_rods in train_rods:
        for i, (column, rod_height, color) in enumerate(short_rods[:-1]):
            successor = short_rods[i + 1]
            if color != TRIGGER or successor[2] == TRIGGER or successor[0] != column + rod_height + 1:
                continue
            contrast |= any((column, rod_height + 1, TRIGGER) in other and successor in other
                            for other in train_rods)
    assert contrast, "missing height-dependent hit/miss contrast"
    max_height = max(h for rods in train_rods for _, h, _ in rods)
    max_chain = max(_longest_chain(rods) for rods in train_rods)
    assert any(_longest_chain(rods) > max_chain or any(
        h > max_height and not fallen for (_, h, _), fallen in zip(rods, flags))
        for rods, flags in test_records), "missing structural train/test shift"


def _digest_puzzle(puzzle, normalized=False):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            background = int(pair["input"][0, 0])
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                if normalized:
                    grid = np.where(grid == background, 0, np.where(grid == TRIGGER, 1, 2)).astype(np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed coverage sweep; returns small metrics and writes nothing."""
    puzzles, normalized, shapes, layouts, palettes = set(), set(), set(), set(), set()
    heights, counts, headrooms, trigger_counts = set(), set(), set(), set()
    long_shifts = tall_shifts = wholly_active = leading_passive = into_trigger = 0
    pairs_checked = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_digest_puzzle(puzzle))
        normalized.add(_digest_puzzle(puzzle, True))
        train_max_height = max(h for pair in puzzle["train"] for _, h, _ in _parse(pair["input"])[2])
        train_max_chain = max(_longest_chain(_parse(pair["input"])[2]) for pair in puzzle["train"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, background, rods = _parse(pair["input"])
                flags = _coordinate_check(grid, pair["output"], background, rods)
                shapes.add(grid.shape)
                layouts.add(tuple((c - rods[0][0], h, color == TRIGGER) for c, h, color in rods))
                palettes.add((background, tuple(sorted({c for _, _, c in rods} - {TRIGGER}))))
                heights.update(h for _, h, _ in rods)
                counts.add(len(rods))
                headrooms.add(grid.shape[0] - max(h for _, h, _ in rods))
                trigger_counts.add(sum(color == TRIGGER for _, _, color in rods))
                wholly_active += all(flags)
                leading_passive += rods[0][2] != TRIGGER
                into_trigger += any(flags[i] and rods[i][2] != TRIGGER and
                                    rods[i + 1][2] == TRIGGER and
                                    rods[i][0] + rods[i][1] == rods[i + 1][0]
                                    for i in range(len(rods) - 1))
                pairs_checked += 1
                if split == "test":
                    long_shifts += _longest_chain(rods) > train_max_chain
                    tall_shifts += any(h > train_max_height and not f
                                       for (_, h, _), f in zip(rods, flags))
    assert len(puzzles) == len(normalized) == 42
    assert len(shapes) >= 40 and len(layouts) >= 100 and len(palettes) >= 20
    assert len(heights) >= 5 and len(counts) >= 5 and len(headrooms) >= 4
    assert len(trigger_counts) >= 3
    assert long_shifts >= 10 and tall_shifts >= 10
    assert wholly_active >= 5 and leading_passive >= 42 and into_trigger >= 1
    assert _digest_puzzle(generate_puzzle(0)) == _digest_puzzle(generate_puzzle(0))
    # All thirty legal explicit count combinations, streaming one at a time.
    legal_calls = 0
    for train_count in range(2, 11):
        for test_count in range(1, min(4, train_count - 1) + 1):
            puzzle = generate_puzzle(12345, train_count, test_count)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
            legal_calls += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Boundary and uncommon valid inputs outside the ordinary sampler ranges.
    fixtures = ((3, 28, [(0, 2, 1), (2, 2, 2), (5, 2, 2)], 0),
                (15, 16, [(0, 14, 0), (14, 2, 1)], 9),
                (16, 15, [(0, 15, 1)], 0),
                (29, 2, [(0, 2, 1)], 8),
                (5, 12, [(0, 3, 2), (3, 2, 1), (5, 4, 2), (9, 2, 1)], 0))
    for height, width, rods, background in fixtures:
        pair = _make_pair(height, width, rods, background)
        _coordinate_check(pair["input"], pair["output"], background, rods)
        original = pair["input"].copy()
        assert np.array_equal(solve(original.tolist()), pair["output"])
        assert np.array_equal(original, pair["input"])
    return {"seeds":42,"pairs_checked":pairs_checked,"distinct_puzzles":len(puzzles),
            "normalized_puzzles":len(normalized),"input_shapes":len(shapes),
            "normalized_rod_layouts":len(layouts),"palette_profiles":len(palettes),
            "rod_heights":sorted(heights),"rod_counts":sorted(counts),
            "headrooms":sorted(headrooms),"trigger_counts":sorted(trigger_counts),
            "long_cascade_tests":long_shifts,"taller_survivor_tests":tall_shifts,
            "wholly_active_pairs":wholly_active,"leading_passive_pairs":leading_passive,
            "passive_into_trigger_contacts":into_trigger,"legal_count_calls":legal_calls,
            "boundary_fixtures":len(fixtures),"independent_solver":False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
