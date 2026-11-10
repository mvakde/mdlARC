"""ARC marker transfer across one unchanged support cell to a receiver bar."""

import numpy as np


_SOURCE_PAIRS = (
    (("00000", "00000", "01010", "05050", "55555"),
     ("00000", "00000", "00000", "05050", "51515")),
    (("00000", "00000", "00100", "00500", "55555"),
     ("00000", "00000", "00000", "00500", "55155")),
    (("00000", "00000", "01001", "05005", "55555"),
     ("00000", "00000", "00000", "05005", "51551")),
    (("00000", "00000", "00101", "00505", "55555"),
     ("00000", "00000", "00000", "00505", "55151")),
)


def _checked_count(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        first = num_test + 1
        num_train = first + (seed // 4) % (11 - first)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _grid_key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _is_source_copy(inp, out):
    if inp.shape != (5, 5) or out.shape != (5, 5):
        return False
    rows = lambda grid: tuple("".join(str(int(v)) for v in row) for row in grid)
    return (rows(inp), rows(out)) in _SOURCE_PAIRS


def _sample_length(rng):
    band = int(rng.integers(0, 3))
    return int(rng.integers((4, 10, 19)[band], (10, 19, 31)[band]))


def _sample_positions(rng, length, count):
    """Choose independent points, a run, or a run mixed with other points."""
    mode = int(rng.integers(0, 3))
    if mode == 1 and count >= 2:
        start = int(rng.integers(0, length - count + 1))
        return tuple(range(start, start + count))
    if mode == 2 and count >= 3 and length >= count + 3:
        run = int(rng.integers(2, min(4, count - 1) + 1))
        start = int(rng.integers(0, length - run + 1))
        selected = set(range(start, start + run))
        separated = [p for p in range(length)
                     if p not in selected and (p < start - 1 or p > start + run)]
        if len(separated) >= count - run:
            rest = rng.choice(separated, size=count - run, replace=False)
            selected.update(int(p) for p in rest)
            return tuple(sorted(selected))
    selected = rng.choice(length, size=count, replace=False)
    return tuple(sorted(int(p) for p in selected))


def _build_pair(rng, orientation, count=None, multicolor=None,
                length=None, positions=None, forbidden_roles=None):
    """Construct one locally readable bar in downward canonical coordinates."""
    compact = False
    if length is None:
        compact = bool(rng.integers(0, 14) == 0)
        length = 5 if compact else _sample_length(rng)
    max_count = min(12, length - 2)
    if positions is not None:
        positions = tuple(sorted(positions))
        count = len(positions)
    elif count is None:
        count = int(rng.integers(1, max_count + 1))
    elif count == -1:
        count = int(rng.integers(2, max_count + 1))
    count = min(count, max_count)
    if positions is None:
        positions = _sample_positions(rng, length, count)
    if not (1 <= len(positions) <= max_count
            and len(set(positions)) == len(positions)
            and positions[0] >= 0 and positions[-1] < length):
        raise ValueError("invalid marker positions")

    palette = [int(v) for v in rng.permutation(10)]
    bg, structure = palette[:2]
    if forbidden_roles == (bg, structure):
        bg, structure = palette[1], palette[2]
    if multicolor is None:
        multicolor = count >= 2 and bool(rng.integers(0, 2))
    n_marker_colors = (int(rng.integers(2, min(4, count) + 1))
                       if multicolor and count >= 2 else 1)
    marker_palette = palette[2:2 + n_marker_colors]
    if bg in marker_palette or structure in marker_palette:
        marker_palette = [v for v in palette if v not in (bg, structure)][:n_marker_colors]
    marker_values = list(marker_palette)
    if count > n_marker_colors:
        marker_values.extend(int(v) for v in rng.choice(
            marker_palette, size=count - n_marker_colors, replace=True))
    marker_values = [marker_values[int(i)] for i in rng.permutation(count)]

    spare = 30 - length
    if compact:
        left = right = 0
        front = int(rng.integers(0, 3))
        back = 2 - front
    else:
        left = int(rng.integers(0, min(6, spare) + 1))
        right = int(rng.integers(0, min(6, spare - left) + 1))
        front = int(rng.integers(0, 7))
        back = int(rng.integers(0, 7))
    height, width = front + 3 + back, left + length + right
    source_row = front
    bar_row = front + 2
    inp = np.full((height, width), bg, dtype=np.int8)
    inp[bar_row, left:left + length] = structure
    for p, color in zip(positions, marker_values):
        column = left + p
        inp[source_row, column] = color
        inp[source_row + 1, column] = structure
    out = inp.copy()
    for p, color in zip(positions, marker_values):
        column = left + p
        out[source_row, column] = bg
        out[bar_row, column] = color
    return (np.rot90(inp, -orientation).copy(),
            np.rot90(out, -orientation).copy())


def _build_multi_pair(rng, orientation):
    """Repeat the same local receiver rule on separated finite bars."""
    number = int(rng.integers(2, 4))
    lengths = [int(rng.integers(5, 19)) for _ in range(number)]
    width = max(lengths) + int(rng.integers(2, 7))
    front = int(rng.integers(0, 4))
    back = int(rng.integers(0, 4))
    height = front + 4 * number - 1 + back
    palette = [int(v) for v in rng.permutation(10)]
    bg, structure = palette[:2]
    marker_palette = palette[2:2 + int(rng.integers(2, 5))]
    inp = np.full((height, width), bg, dtype=np.int8)
    out = inp.copy()
    for bar_index, length in enumerate(lengths):
        left = int(rng.integers(0, width - length + 1))
        source_row = front + 4 * bar_index
        bar_row = source_row + 2
        count = int(rng.integers(1, min(7, length - 2) + 1))
        positions = _sample_positions(rng, length, count)
        inp[bar_row, left:left + length] = structure
        out[bar_row, left:left + length] = structure
        for position in positions:
            col = left + position
            color = int(rng.choice(marker_palette))
            inp[source_row, col] = color
            inp[source_row + 1, col] = structure
            out[source_row + 1, col] = structure
            out[bar_row, col] = color
    return (np.rot90(inp, -orientation).copy(),
            np.rot90(out, -orientation).copy())


def _decode_input(inp):
    """Infer the complete transformation from the input's visible roles.

    Enumerating all four directions also detects any ambiguous layouts.
    No generator settings, test output, or training output enter this decoder.
    """
    if not isinstance(inp, np.ndarray) or inp.ndim != 2:
        return []
    found = []
    for orientation in range(4):
        grid = np.rot90(inp, orientation)
        height, width = grid.shape
        colors = [int(c) for c in np.unique(grid)]
        for bg in colors:
            for structure in colors:
                if bg == structure:
                    continue
                bars = []
                for bar_row in range(2, height):
                    row = grid[bar_row]
                    loc = np.flatnonzero(row == structure)
                    if len(loc) < 4 or loc[-1] - loc[0] + 1 != len(loc):
                        continue
                    left, right = int(loc[0]), int(loc[-1])
                    if np.any(row[:left] != bg) or np.any(row[right + 1:] != bg):
                        continue
                    source = grid[bar_row - 2]
                    marker_cols = np.flatnonzero(source != bg)
                    if not 1 <= len(marker_cols) <= len(loc) - 2:
                        continue
                    if marker_cols[0] < left or marker_cols[-1] > right:
                        continue
                    if np.any(source[marker_cols] == structure):
                        continue
                    expected_rows = np.full((3, width), bg, dtype=grid.dtype)
                    expected_rows[2, left:right + 1] = structure
                    expected_rows[1, marker_cols] = structure
                    expected_rows[0, marker_cols] = source[marker_cols]
                    if not np.array_equal(grid[bar_row - 2:bar_row + 1], expected_rows):
                        continue
                    bars.append({
                        "row": bar_row,
                        "left": left,
                        "right": right,
                        "positions": tuple(int(c - left) for c in marker_cols),
                        "colors": tuple(int(v) for v in source[marker_cols]),
                    })
                if not bars:
                    continue
                if any(bars[i + 1]["row"] - bars[i]["row"] < 4
                       for i in range(len(bars) - 1)):
                    continue
                expected = np.full(grid.shape, bg, dtype=grid.dtype)
                result = expected.copy()
                for bar in bars:
                    row, left, right = bar["row"], bar["left"], bar["right"]
                    cols = [left + p for p in bar["positions"]]
                    expected[row, left:right + 1] = structure
                    expected[row - 1, cols] = structure
                    expected[row - 2, cols] = bar["colors"]
                    result[row, left:right + 1] = structure
                    result[row - 1, cols] = structure
                    result[row, cols] = bar["colors"]
                if not np.array_equal(grid, expected):
                    continue
                first, last = bars[0], bars[-1]
                found.append({
                    "output": np.rot90(result, -orientation).copy(),
                    "orientation": orientation,
                    "background": bg,
                    "structure": structure,
                    "bars": tuple(bars),
                    "marker_count": sum(len(b["positions"]) for b in bars),
                    "marker_colors": tuple(v for b in bars for v in b["colors"]),
                    "length": first["right"] - first["left"] + 1,
                    "bar_start": first["left"],
                    "marker_positions": first["positions"],
                    "source_margin": first["row"] - 2,
                    "receiver_margin": height - last["row"] - 1,
                    "side_margins": (first["left"], width - first["right"] - 1),
                })
    return found


def _unique_analysis(inp):
    candidates = _decode_input(inp)
    if len(candidates) != 1:
        return None
    return candidates[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    input_keys, pair_keys = set(), set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            input_key, output_key = _grid_key(inp), _grid_key(out)
            assert input_key not in input_keys
            assert (input_key, output_key) not in pair_keys
            input_keys.add(input_key)
            pair_keys.add((input_key, output_key))
            assert not _is_source_copy(inp, out)
            analysis = _unique_analysis(inp)
            assert analysis is not None
            assert np.array_equal(analysis["output"], out)
            analyses[section].append(analysis)

    training = analyses["train"]
    assert any(a["marker_count"] == 1 for a in training)
    assert any(a["marker_count"] >= 2 for a in training)
    assert any(len(set(a["marker_colors"])) >= 2 for a in training)
    if any(len(a["bars"]) > 1 for a in analyses["test"]):
        assert any(len(a["bars"]) > 1 for a in training)
        demonstrated_multi_directions = {
            a["orientation"] for a in training if len(a["bars"]) > 1
        }
        assert all(a["orientation"] in demonstrated_multi_directions
                   for a in analyses["test"] if len(a["bars"]) > 1)
    # Direction is the only puzzle-level convention. Every tested direction
    # has an explicit training demonstration; colors are identified locally.
    trained_directions = {a["orientation"] for a in training}
    assert all(a["orientation"] in trained_directions for a in analyses["test"])
    # Pair colors must not become a memorized puzzle-wide lookup table.
    assert len({(a["background"], a["structure"]) for a in training}) >= 2


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)

    base_direction = int(rng.integers(0, 4))
    mixed = num_train >= 3 and bool(rng.integers(0, 2))
    train_directions = [base_direction] * num_train
    if mixed:
        train_directions[1] = (base_direction + int(rng.integers(1, 4))) % 4
        for index in range(2, num_train):
            train_directions[index] = int(rng.integers(0, 4))
    demonstrated = tuple(sorted(set(train_directions)))
    test_directions = [int(rng.choice(demonstrated)) for _ in range(num_test)]

    # A subset of fixed-direction puzzles explicitly recombine bar-relative
    # positions shown separately by the first two training pairs.
    multibar = num_train >= 3 and bool(rng.integers(0, 4) == 0)
    if multibar:
        test_directions[0] = train_directions[2]
    recombine = not mixed and not multibar and bool(rng.integers(0, 3) == 0)
    planned = {}
    if recombine:
        length = int(rng.integers(6, 19))
        a, b, c = (int(v) for v in rng.choice(length, size=3, replace=False))
        planned = {
            ("train", 0): (length, (a,)),
            ("train", 1): (length, tuple(sorted((b, c)))),
            ("test", 0): (length, tuple(sorted((a, b)))),
        }

    puzzle = {"train": [], "test": []}
    seen = set()
    first_roles = None
    for section, count, directions in (
        ("train", num_train, train_directions),
        ("test", num_test, test_directions),
    ):
        for index in range(count):
            pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            pair_rng = np.random.default_rng(pair_seed)
            forced_length, forced_positions = planned.get((section, index), (None, None))
            if forced_positions is not None:
                marker_count = len(forced_positions)
            elif section == "train" and index == 0:
                marker_count = 1
            elif section == "train" and index == 1:
                marker_count = -1
            elif section == "test" and index == 0 and bool(rng.integers(0, 3) == 0):
                marker_count = 1
            else:
                marker_count = None
            force_multi = True if section == "train" and index == 1 else None
            use_multiple = multibar and (
                (section == "train" and index == 2)
                or (section == "test" and index == 0)
                or (section == "train" and index > 2
                    and bool(rng.integers(0, 5) == 0))
            )
            for attempt in range(64):
                # The last attempt is a deterministic, spacious fallback.
                if attempt == 63:
                    forced_length = 19 + (index % 8)
                    forced_positions = (index % forced_length,)
                    marker_count = 1
                    force_multi = False
                if use_multiple:
                    inp, out = _build_multi_pair(pair_rng, directions[index])
                else:
                    inp, out = _build_pair(
                        pair_rng, directions[index], marker_count, force_multi,
                        forced_length, forced_positions,
                        first_roles if section == "train" and index == 1 else None,
                    )
                key = _grid_key(inp)
                analysis = _unique_analysis(inp)
                if (key not in seen and not _is_source_copy(inp, out)
                        and analysis is not None
                        and np.array_equal(analysis["output"], out)):
                    break
            else:
                raise RuntimeError("could not construct a unique readable pair")
            seen.add(key)
            if section == "train" and index == 0:
                first_roles = (analysis["background"], analysis["structure"])
            puzzle[section].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(
        np.array_equal(pa[side], pb[side])
        for section in ("train", "test")
        for pa, pb in zip(a[section], b[section])
        for side in ("input", "output")
    ) and all(len(a[s]) == len(b[s]) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    dimensions, layouts, roles, directions, marker_counts = set(), set(), set(), set(), set()
    has_adjacent = has_mixed_spacing = has_finite_bar = False
    has_multicolor = has_top_edge = has_long_bar = has_multibar = False
    has_30_axis = has_compact = has_within_roles = has_within_counts = False
    test_single = test_large = test_unseen = test_recombined = 0
    test_multibar = mixed_puzzles = multibar_train_puzzles = 0
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(tuple(_grid_key(p["input"])
                               for section in ("train", "test")
                               for p in puzzle[section]))
        train_analyses = [_unique_analysis(p["input"]) for p in puzzle["train"]]
        assert all(a is not None for a in train_analyses)
        test_analyses = [_unique_analysis(p["input"]) for p in puzzle["test"]]
        assert all(a is not None for a in test_analyses)
        trained_directions = {a["orientation"] for a in train_analyses}
        assert all(a["orientation"] in trained_directions for a in test_analyses)
        mixed_puzzles += len(trained_directions) > 1
        multibar_train_puzzles += any(len(a["bars"]) > 1 for a in train_analyses)
        trained_positions = {position for a in train_analyses
                             for bar in a["bars"] for position in bar["positions"]}
        trained_sets = [set(bar["positions"]) for a in train_analyses
                        for bar in a["bars"]]
        for a in test_analyses:
            positions = {position for bar in a["bars"]
                         for position in bar["positions"]}
            test_single += a["marker_count"] == 1
            test_large += a["marker_count"] >= 3
            test_unseen += bool(positions - trained_positions)
            test_multibar += len(a["bars"]) > 1
            test_recombined += (len(a["bars"]) == 1 and len(positions) >= 2
                                and positions <= trained_positions
                                and positions not in trained_sets
                                and sum(bool(positions & pattern)
                                        for pattern in trained_sets) >= 2)
        has_within_roles |= len({(a["background"], a["structure"])
                                 for a in train_analyses}) >= 2
        has_within_counts |= len({a["marker_count"]
                                  for a in train_analyses}) >= 2
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                a = _unique_analysis(inp)
                assert a is not None
                # Independent cell-difference check: each source clearing
                # pairs with a same-colored destination across one support.
                rotated_in = np.rot90(inp, a["orientation"])
                rotated_out = np.rot90(out, a["orientation"])
                sources = np.argwhere((rotated_in != rotated_out)
                                      & (rotated_out == a["background"]))
                targets = np.argwhere((rotated_in != rotated_out)
                                      & (rotated_in == a["structure"]))
                assert len(sources) == len(targets) == a["marker_count"]
                expected = rotated_in.copy()
                for row, col in sources:
                    assert rotated_in[row + 1, col] == a["structure"]
                    assert rotated_out[row + 1, col] == a["structure"]
                    assert rotated_in[row + 2, col] == a["structure"]
                    assert rotated_out[row + 2, col] == rotated_in[row, col]
                    expected[row, col] = a["background"]
                    expected[row + 2, col] = rotated_in[row, col]
                assert np.array_equal(expected, rotated_out)
                assert np.array_equal(a["output"], out)
                assert not _is_source_copy(inp, out)
                dimensions.add(inp.shape)
                roles.add((a["background"], a["structure"]))
                directions.add(a["orientation"])
                marker_counts.add(a["marker_count"])
                has_multibar |= len(a["bars"]) > 1
                for bar in a["bars"]:
                    positions = bar["positions"]
                    gaps = [positions[i + 1] - positions[i]
                            for i in range(len(positions) - 1)]
                    has_adjacent |= 1 in gaps
                    has_mixed_spacing |= 1 in gaps and any(g > 1 for g in gaps)
                    has_finite_bar |= bar["left"] > 0 or bar["right"] < rotated_in.shape[1] - 1
                has_multicolor |= len(set(a["marker_colors"])) >= 2
                has_top_edge |= a["source_margin"] == 0
                has_long_bar |= a["length"] >= 20
                has_30_axis |= 30 in inp.shape
                has_compact |= inp.shape == (5, 5)
                layouts.add((a["orientation"], inp.shape,
                             tuple((b["row"], b["left"], b["right"], b["positions"])
                                   for b in a["bars"])))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) == 42 and len(dimensions) >= 45
    assert len(layouts) >= 100 and len(roles) >= 25
    assert directions == {0, 1, 2, 3}
    assert {1, 2, 3, 4, 5}.issubset(marker_counts)
    assert (has_adjacent and has_mixed_spacing and has_finite_bar
            and has_multicolor and has_top_edge and has_long_bar
            and has_30_axis and has_compact and has_multibar and has_within_roles
            and has_within_counts)
    assert test_single >= 10 and test_large >= 30
    assert test_unseen >= 20 and test_recombined >= 10
    assert test_multibar >= 3 and multibar_train_puzzles >= 3
    assert mixed_puzzles >= 10
    for seed, nt, ne in ((100, 2, 1), (101, 10, 1), (102, 10, 4), (103, 6, 3),
                         (200, None, 1), (201, None, 4),
                         (202, 2, None), (203, 10, None)):
        p = generate_puzzle(seed, nt, ne)
        validate_generated_puzzle(p)
        if nt is not None:
            assert len(p["train"]) == nt
        if ne is not None:
            assert len(p["test"]) == ne
    assert _puzzles_equal(generate_puzzle(999, 7, 3), generate_puzzle(999, 7, 3))
    assert _puzzles_equal(generate_puzzle(17), generate_puzzle(17))
    assert not _puzzles_equal(generate_puzzle(1234), generate_puzzle(1235))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(300, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid counts accepted: {kwargs}")
