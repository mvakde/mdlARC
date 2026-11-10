"""Generate aligned-bar comparison puzzles with locally readable color roles."""

import numpy as np


_SOURCE_INPUT_ROWS = frozenset({
    ("002000400", "000000400", "000000400", "000000000", "000000000",
     "555555555", "002000400", "002000400", "002000400", "002000400",
     "000000400"),
    ("002000400", "000000400", "000000000", "000000000", "555555555",
     "002000400", "002000400", "002000400", "002000400"),
    ("002000400", "002000400", "000000000", "000000000", "000000000",
     "555555555", "002000400", "002000400", "002000400", "002000000",
     "000000000"),
    ("002000400", "000000400", "000000000", "000000000", "555555555",
     "002000400", "002000400", "000000400", "000000400"),
    ("002000400", "002000400", "002000000", "000000000", "000000000",
     "555555555", "002000400", "002000400", "002000400", "002000400",
     "000000000"),
})

_MODES = (
    "zero_second", "zero_first", "ordinary", "shrink_wins",
    "growth_beats_shrink", "unchanged_loser", "zero_first_loser",
    "zero_second_loser", "free",
)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _choice(rng, values):
    assert values
    return values[int(rng.integers(len(values)))]


def _kind(pair):
    a, b = pair
    if a == 0:
        return "zero_first"
    if b == 0:
        return "zero_second"
    if b > a:
        return "grow"
    if b < a:
        return "shrink"
    return "same"


def _mode_ok(mode, winner, loser):
    wk, lk = _kind(winner), _kind(loser)
    if mode == "zero_second":
        return wk == "zero_second" and lk == "grow"
    if mode == "zero_first":
        return wk == "zero_first" and lk == "shrink"
    if mode == "ordinary":
        return wk == "grow" and lk == "grow"
    if mode == "shrink_wins":
        return wk == "shrink" and lk == "grow"
    if mode == "growth_beats_shrink":
        return wk == "grow" and lk == "shrink"
    if mode == "unchanged_loser":
        return wk in ("grow", "shrink") and lk == "same"
    if mode == "zero_first_loser":
        return wk in ("grow", "shrink") and lk == "zero_first"
    if mode == "zero_second_loser":
        return wk in ("grow", "shrink") and lk == "zero_second"
    return mode == "free"


def _sample_lengths(rng, first_depth, second_depth, mode, close):
    pairs = [(a, b) for a in range(first_depth + 1)
             for b in range(second_depth + 1) if a or b]
    options = []
    for winner in pairs:
        winning_diff = abs(winner[1] - winner[0])
        if winning_diff == 0:
            continue
        losers = [loser for loser in pairs
                  if (abs(loser[1] - loser[0]) == winning_diff - 1 if close
                      else abs(loser[1] - loser[0]) < winning_diff)
                  and _mode_ok(mode, winner, loser)]
        if losers:
            options.append((winner, losers))
    if not options and close:
        return _sample_lengths(rng, first_depth, second_depth, mode, False)
    assert options, (first_depth, second_depth, mode)
    winner, losers = _choice(rng, options)
    return winner, _choice(rng, losers)


def _sample_lanes(rng, cross_size):
    category = int(rng.integers(0, 10))
    if category < 3:
        edge = _choice(rng, [0, cross_size - 1])
        other = _choice(rng, [x for x in range(cross_size) if x != edge])
        return tuple(sorted((edge, other)))
    if category < 6:
        start = int(rng.integers(0, cross_size - 1))
        return start, start + 1
    choices = [(a, b) for a in range(cross_size) for b in range(a + 2, cross_size)]
    return _choice(rng, choices)


def _render_pair(rng, mode, orientation, colors, reverse_order, reflected, regime):
    if regime == "source_train":
        first_depth = int(rng.integers(3, 12))
        second_depth = int(rng.integers(3, 12))
        cross_size = int(rng.integers(5, 18))
    else:
        first_depth = int(rng.integers(3, 15))
        second_depth = int(rng.integers(3, 15))
        cross_size = int(rng.integers(5, 21))
    background, separator, signal_a, signal_b = colors
    winner, loser = _sample_lengths(
        rng, first_depth, second_depth, mode, bool(rng.integers(0, 3) == 0))
    winning_color = _choice(rng, [signal_a, signal_b])
    losing_color = signal_b if winning_color == signal_a else signal_a
    lengths = {winning_color: winner, losing_color: loser}
    lane_a, lane_b = _sample_lanes(rng, cross_size)
    ordered_colors = sorted((signal_a, signal_b), reverse=reverse_order)
    lanes = {ordered_colors[0]: lane_a, ordered_colors[1]: lane_b}
    if orientation == "horizontal":
        grid = np.full((first_depth + second_depth + 1, cross_size), background, dtype=np.int8)
        grid[first_depth, :] = separator
    else:
        grid = np.full((cross_size, first_depth + second_depth + 1), background, dtype=np.int8)
        grid[:, first_depth] = separator
    for color, (first, second) in lengths.items():
        if not reflected:
            first_start = 0
            second_start = first_depth + 1
        else:
            first_start = first_depth - first
            second_start = first_depth + 1 + second_depth - second
        lane = lanes[color]
        if orientation == "horizontal":
            grid[first_start:first_start + first, lane] = color
            grid[second_start:second_start + second, lane] = color
        else:
            grid[lane, first_start:first_start + first] = color
            grid[lane, second_start:second_start + second] = color
    return {"input": grid, "output": np.full((2, 2), winning_color, dtype=np.int8)}


def _fingerprint(array):
    normalized = np.asarray(array, dtype=np.int8)
    return normalized.shape, normalized.tobytes()


def _source_copy(grid):
    rows = tuple("".join(str(int(x)) for x in row) for row in grid)
    return rows in _SOURCE_INPUT_ROWS


def _analyze_input(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 4, "input requires background, separator, and two signals"
    assert np.count_nonzero(counts == np.max(counts)) == 1
    background = int(values[int(np.argmax(counts))])
    candidates = []
    for value in values:
        color = int(value)
        if color == background:
            continue
        candidates += [(color, "horizontal", int(i)) for i in np.flatnonzero(np.all(grid == color, axis=1))]
        candidates += [(color, "vertical", int(i)) for i in np.flatnonzero(np.all(grid == color, axis=0))]
    assert len(candidates) == 1, "one full-length separator is required"
    separator, orientation, index = candidates[0]
    first_depth = index
    second_depth = grid.shape[0 if orientation == "horizontal" else 1] - index - 1
    cross_size = grid.shape[1 if orientation == "horizontal" else 0]
    assert first_depth >= 3 and second_depth >= 3 and cross_size >= 5
    signals = tuple(sorted(int(v) for v in values if int(v) not in (background, separator)))
    lengths, lanes = {}, {}
    for color in signals:
        coords = np.argwhere(grid == color)
        assert len(coords) > 0
        lane_values = np.unique(coords[:, 1 if orientation == "horizontal" else 0])
        assert len(lane_values) == 1, "a signal must use one aligned lane"
        lane = int(lane_values[0])
        lanes[color] = lane
        positions = sorted(int(v) for v in coords[:, 0 if orientation == "horizontal" else 1])
        before = [p for p in positions if p < index]
        after = [p for p in positions if p > index]
        assert len(before) + len(after) == len(positions)
        lengths[color] = (len(before), len(after))
    assert len(set(lanes.values())) == 2
    anchors = []
    for reflected in (False, True):
        expected = np.full_like(grid, background)
        if orientation == "horizontal":
            expected[index, :] = separator
        else:
            expected[:, index] = separator
        for color, (first, second) in lengths.items():
            first_start = first_depth - first if reflected else 0
            second_start = index + 1 + second_depth - second if reflected else index + 1
            lane = lanes[color]
            if orientation == "horizontal":
                expected[first_start:first_start + first, lane] = color
                expected[second_start:second_start + second, lane] = color
            else:
                expected[lane, first_start:first_start + first] = color
                expected[lane, second_start:second_start + second] = color
        if np.array_equal(grid, expected):
            anchors.append(reflected)
    assert anchors, "bars must be contiguous and share an anchoring direction"
    return {"background": background, "separator": separator,
            "orientation": orientation, "first_depth": first_depth,
            "second_depth": second_depth, "cross_size": cross_size,
            "signals": signals, "lengths": lengths, "lanes": lanes,
            "anchors": tuple(anchors)}


def _check_pair_rule(input_grid, output_grid):
    info = _analyze_input(input_grid)
    a, b = info["signals"]
    difference = {color: abs(info["lengths"][color][1] - info["lengths"][color][0])
                  for color in (a, b)}
    assert difference[a] != difference[b], "discrepancies must have a unique winner"
    winner = a if difference[a] > difference[b] else b
    assert output_grid.shape == (2, 2) and np.all(output_grid == winner)
    info["winner"] = winner
    info["differences"] = difference
    info["margin"] = abs(difference[a] - difference[b])
    return info


def _pair_key(pair):
    return _fingerprint(pair["input"]), _fingerprint(pair["output"])


def _puzzle_key(puzzle):
    return tuple(tuple(_pair_key(p) for p in puzzle[split]) for split in ("train", "test"))


def _independent_winner(grid):
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    full_lines = []
    for value in colors:
        color = int(value)
        if color != background:
            full_lines.extend((color, 0, int(row)) for row in
                              np.flatnonzero(np.all(grid == color, axis=1)))
            full_lines.extend((color, 1, int(column)) for column in
                              np.flatnonzero(np.all(grid == color, axis=0)))
    assert len(full_lines) == 1
    separator, axis, index = full_lines[0]
    signals = [int(color) for color in colors if int(color) not in (background, separator)]
    assert len(signals) == 2
    discrepancies = {}
    for color in signals:
        positions = np.argwhere(grid == color)[:, axis]
        discrepancies[color] = abs(int(np.sum(positions < index)) -
                                   int(np.sum(positions > index)))
    assert discrepancies[signals[0]] != discrepancies[signals[1]]
    return max(signals, key=discrepancies.get)


def _make_pair(rng, mode, orientation, colors, reverse_order, reflected, regime, seen):
    for _ in range(200):
        pair = _render_pair(rng, mode, orientation, colors, reverse_order, reflected, regime)
        key = _fingerprint(pair["input"])
        if key in seen or _source_copy(pair["input"]):
            continue
        _check_pair_rule(pair["input"], pair["output"])
        seen.add(key)
        return pair
    raise AssertionError("could not construct a unique non-source pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    mixed = bool(rng.integers(0, 10) < 7)
    base_colors = tuple(int(x) for x in rng.choice(10, 4, replace=False))
    base_orientation = _choice(rng, ["horizontal", "vertical"])
    base_order = bool(rng.integers(0, 2))
    modes = list(_MODES[:num_train])
    rng.shuffle(modes)
    if len(modes) < num_train:
        modes += [_choice(rng, list(_MODES)) for _ in range(num_train - len(modes))]
    n = num_train + num_test
    orientations = [base_orientation] * n
    orders = [base_order] * n
    reflections = [False] * n
    palettes = [base_colors] * n
    if mixed:
        orientations = [_choice(rng, ["horizontal", "vertical"]) for _ in range(n)]
        orders = [bool(rng.integers(0, 2)) for _ in range(n)]
        reflections = [bool(rng.integers(0, 2)) for _ in range(n)]
        palettes = [tuple(int(x) for x in rng.choice(10, 4, replace=False)) for _ in range(n)]
        first, second = rng.choice(num_train, 2, replace=False)
        first, second = int(first), int(second)
        orientations[first], orientations[second] = "horizontal", "vertical"
        orders[first], orders[second] = False, True
        reflections[first], reflections[second] = False, True
        if palettes[first] == palettes[second]:
            palettes[second] = tuple(reversed(palettes[second]))
    puzzle = {"train": [], "test": []}
    seen = set()
    for index in range(n):
        split = "train" if index < num_train else "test"
        mode = modes[index] if split == "train" else _choice(
            rng, ["zero_second", "zero_first", "shrink_wins", "growth_beats_shrink",
                  "ordinary", "unchanged_loser", "zero_first_loser", "zero_second_loser", "free"])
        pair = _make_pair(rng, mode, orientations[index], palettes[index], orders[index],
                          reflections[index], "source_train" if split == "train" else "target_test", seen)
        puzzle[split].append(pair)
    # Both signal roles win in the demonstrations, without imposing an answer sequence.
    train_winners = [int(p["output"][0, 0]) for p in puzzle["train"]]
    if len(set(train_winners)) == 1:
        index = int(rng.integers(num_train))
        pair = puzzle["train"][index]
        seen.remove(_fingerprint(pair["input"]))
        signal_a, signal_b = _analyze_input(pair["input"])["signals"]
        original = pair["input"].copy()
        pair["input"][original == signal_a] = signal_b
        pair["input"][original == signal_b] = signal_a
        pair["output"][:] = signal_b if train_winners[index] == signal_a else signal_a
        assert _fingerprint(pair["input"]) not in seen
        seen.add(_fingerprint(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert grid.dtype.kind in "iu"
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            input_key = _fingerprint(pair["input"])
            pair_key = _pair_key(pair)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert not _source_copy(pair["input"])
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analyses[split].append(_check_pair_rule(pair["input"], pair["output"]))
    train = analyses["train"]
    assert len({a["winner"] for a in train}) >= 2
    assert any(_kind(a["lengths"][a["winner"]]) == "zero_second" and
               _kind(a["lengths"][next(c for c in a["signals"] if c != a["winner"])]) == "grow"
               for a in train)
    assert any(_kind(a["lengths"][a["winner"]]) == "zero_first" and
               _kind(a["lengths"][next(c for c in a["signals"] if c != a["winner"])]) == "shrink"
               for a in train)
    train_orientations = {a["orientation"] for a in train}
    for a in analyses["test"]:
        assert a["orientation"] in train_orientations
        assert any(set(a["anchors"]) & set(t["anchors"]) for t in train)
    if len({(a["background"], a["separator"], a["signals"]) for a in analyses["test"] + train}) > 1:
        assert len({(a["background"], a["separator"], a["signals"]) for a in train}) >= 2


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    features = {"palette": set(), "shape": set(), "layout": set(), "lengths": set(),
                "orientation": set(), "margin": set(), "winner_kind": set(),
                "lane_gap": set(), "depth_gap": set(), "anchor": set()}
    adjacent = edge = full_depth = unchanged = close = repeated_wins = 0
    mixed_orientations = mixed_palettes = mixed_orders = mixed_anchors = 0
    train_mode_counts = {mode: 0 for mode in _MODES}
    test_mode_counts = {mode: 0 for mode in _MODES}
    combined_vertical_reflected_edge = combined_adjacent_missing_unequal = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        analyses = [_check_pair_rule(p["input"], p["output"])
                    for split in ("train", "test") for p in puzzle[split]]
        palettes = {(a["background"], a["separator"], a["signals"]) for a in analyses}
        orders = {tuple(sorted(a["lanes"], key=a["lanes"].get)) == a["signals"] for a in analyses}
        if len({a["orientation"] for a in analyses}) > 1:
            mixed_orientations += 1
        if len(palettes) > 1:
            mixed_palettes += 1
        if len(orders) > 1:
            mixed_orders += 1
        if len({a["anchors"] for a in analyses}) > 1:
            mixed_anchors += 1
        winners = [a["winner"] for a in analyses]
        repeated_wins += any(winners[i] == winners[i + 1] for i in range(len(winners) - 1))
        for pair_index, (pair, a) in enumerate(zip(
                [p for split in ("train", "test") for p in puzzle[split]], analyses)):
            independent = _independent_winner(pair["input"])
            assert independent == a["winner"]
            assert np.array_equal(pair["output"], np.full((2, 2), independent))
            features["palette"].add((a["background"], a["separator"], a["signals"]))
            features["shape"].add((a["orientation"], a["first_depth"], a["second_depth"], a["cross_size"]))
            features["layout"].add((a["orientation"], a["cross_size"], tuple(sorted(a["lanes"].values()))))
            features["lengths"].add(tuple(sorted(a["lengths"].values())))
            features["orientation"].add(a["orientation"])
            features["margin"].add(a["margin"])
            features["winner_kind"].add(_kind(a["lengths"][a["winner"]]))
            features["lane_gap"].add(abs(a["lanes"][a["signals"][0]] - a["lanes"][a["signals"][1]]))
            features["depth_gap"].add(abs(a["first_depth"] - a["second_depth"]))
            features["anchor"].update(a["anchors"])
            lanes = list(a["lanes"].values())
            adjacent += abs(lanes[0] - lanes[1]) == 1
            edge += 0 in lanes or a["cross_size"] - 1 in lanes
            full_depth += any(x == depth for lengths in a["lengths"].values()
                              for x, depth in zip(lengths, (a["first_depth"], a["second_depth"])))
            unchanged += any(x == y for x, y in a["lengths"].values())
            close += a["margin"] == 1
            loser = next(c for c in a["signals"] if c != a["winner"])
            winner_lengths = a["lengths"][a["winner"]]
            loser_lengths = a["lengths"][loser]
            split_modes = train_mode_counts if pair_index < len(puzzle["train"]) else test_mode_counts
            for mode in _MODES:
                split_modes[mode] += _mode_ok(mode, winner_lengths, loser_lengths)
            combined_vertical_reflected_edge += (
                a["orientation"] == "vertical" and True in a["anchors"] and
                (0 in lanes or a["cross_size"] - 1 in lanes))
            combined_adjacent_missing_unequal += (
                abs(lanes[0] - lanes[1]) == 1 and
                a["first_depth"] != a["second_depth"] and
                (0 in winner_lengths or 0 in loser_lengths))
    assert len(features["palette"]) >= 40 and len(features["shape"]) >= 100
    assert len(features["layout"]) >= 60 and len(features["lengths"]) >= 100
    assert features["orientation"] == {"horizontal", "vertical"}
    assert len(features["margin"]) >= 3 and len(features["winner_kind"]) >= 4
    assert min(features["lane_gap"]) == 1 and max(features["lane_gap"]) >= 4
    assert max(features["depth_gap"]) >= 5 and features["anchor"] == {False, True}
    assert min(adjacent, edge, full_depth, unchanged, close) >= 10
    assert min(mixed_orientations, mixed_palettes, mixed_orders, mixed_anchors, repeated_wins) >= 10
    assert min(train_mode_counts.values()) >= 10
    assert min(test_mode_counts.values()) >= 5
    assert combined_vertical_reflected_edge >= 10
    assert combined_adjacent_missing_unequal >= 10
    assert _puzzle_key(generate_puzzle(314159)) == _puzzle_key(generate_puzzle(314159))
    assert _puzzle_key(generate_puzzle(4001, 6, 2)) != _puzzle_key(generate_puzzle(4002, 6, 2))
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (2001, None, 1), (2002, None, 4), (2003, 2, None),
                              (2004, 10, None), (271828, 7, 3)):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for request in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(9999, *request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {request}")
