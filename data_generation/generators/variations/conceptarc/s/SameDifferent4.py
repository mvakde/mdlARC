"""Generate grids that keep the most frequent straight-line direction."""

import numpy as np


_DIRECTIONS = ("H", "V", "D", "A")
_SOURCE_INPUTS = (
    ("0000000040", "0000100040", "0001000040", "0010000040", "0100000040", "0000000040", "6000000000", "6002222200", "6000000000", "6000000000"),
    ("0000300000", "0003022220", "0030000000", "0300000090", "3000000090", "0000000090", "0000000090", "0777777090", "0000000000", "0000000000"),
    ("0000003333", "0000100000", "0001000000", "0010000500", "0100005000", "1000050000", "0000500000", "0700000000", "0700000000", "0700000000"),
    ("0000000000", "0020000000", "0020000000", "0000000000", "0000700000", "0000700000", "0000700000", "0000700000", "0000000000", "0000000000"),
    ("3000400000", "3000400000", "3000400006", "3000000060", "3000000600", "3000000000", "3000000000", "3006666666", "3000000000", "3000000000"),
)
_SOURCE_KEYS = frozenset(tuple(int(c) for row in rows for c in row) for rows in _SOURCE_INPUTS)
_MODES = ((2, 0, 0), (2, 1, 0), (2, 1, 1), (3, 1, 1), (3, 2, 1))


def _valid_count(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _choose_counts(seed, rng, num_train, num_test):
    if num_train is not None:
        num_train = _valid_count(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _valid_count(num_test, "num_test", 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        legal = [(a, b) for a in range(2, 11) for b in range(1, 5) if a > b]
        return legal[seed % len(legal)]
    if num_train is None:
        return int(rng.integers(num_test + 1, 11)), num_test
    return num_train, int(rng.integers(1, min(4, num_train - 1) + 1))


def _steps(direction):
    return {"H": (0, 1), "V": (1, 0), "D": (1, -1), "A": (1, 1)}[direction]


def _orientation(cells):
    assert len(cells) >= 2
    points = set(cells)
    assert len(points) == len(cells)
    for direction in _DIRECTIONS:
        dr, dc = _steps(direction)
        for row, col in points:
            if {(row + dr * k, col + dc * k) for k in range(len(points))} == points:
                return direction
    raise AssertionError("A colored component is not a straight segment")


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    assert counts[index] > grid.size * 0.55
    return int(colors[index])


def _analyze_segments(grid):
    background = _background(grid)
    height, width = grid.shape
    seen = set()
    segments = []
    for row in range(height):
        for col in range(width):
            if int(grid[row, col]) == background or (row, col) in seen:
                continue
            color = int(grid[row, col])
            stack = [(row, col)]
            seen.add((row, col))
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for rr in range(max(0, r - 1), min(height, r + 2)):
                    for cc in range(max(0, c - 1), min(width, c + 2)):
                        if (rr, cc) not in seen and int(grid[rr, cc]) == color:
                            seen.add((rr, cc))
                            stack.append((rr, cc))
            segments.append({"cells": cells, "color": color, "orientation": _orientation(cells), "length": len(cells)})
    return segments


def _winner(segments):
    counts = {direction: sum(s["orientation"] == direction for s in segments) for direction in _DIRECTIONS}
    top = max(counts.values())
    assert top >= 2 and list(counts.values()).count(top) == 1
    return next(direction for direction in _DIRECTIONS if counts[direction] == top)


def _solve(grid):
    background = _background(grid)
    segments = _analyze_segments(grid)
    winner = _winner(segments)
    result = grid.copy()
    for segment in segments:
        if segment["orientation"] != winner:
            for row, col in segment["cells"]:
                result[row, col] = background
    return result


def _dimensions(rng):
    band = int(rng.choice(4, p=(0.22, 0.28, 0.30, 0.20)))
    if band == 0:
        return 10, 10
    if band == 1:
        return int(rng.integers(9, 15)), int(rng.integers(9, 15))
    if band == 2:
        return int(rng.integers(14, 22)), int(rng.integers(14, 22))
    return int(rng.integers(21, 30)), int(rng.integers(21, 30))


def _maximum_length(direction, height, width):
    if direction == "H":
        return width
    if direction == "V":
        return height
    return min(height, width)


def _choose_lengths(rng, directions, height, width, winner, pattern):
    lengths = []
    for direction in directions:
        limit = _maximum_length(direction, height, width)
        roll = rng.random()
        if roll < 0.12:
            length = limit
        elif roll < 0.24:
            length = max(2, limit - int(rng.integers(1, 3)))
        else:
            length = int(rng.integers(2, min(limit, max(5, int(limit * 0.75))) + 1))
        lengths.append(length)
    kept = [i for i, direction in enumerate(directions) if direction == winner]
    removed = [i for i, direction in enumerate(directions) if direction != winner]
    if removed and pattern in ("kept", "erased", "tie"):
        favored = kept if pattern == "kept" else removed
        other = removed if pattern == "kept" else kept
        favored_index = max(favored, key=lambda i: lengths[i])
        other_index = max(other, key=lambda i: lengths[i])
        if pattern == "tie":
            common = min(_maximum_length(directions[favored_index], height, width), _maximum_length(directions[other_index], height, width))
            common = min(common, max(3, lengths[favored_index], lengths[other_index]))
            lengths[favored_index] = common
            lengths[other_index] = common
            for i in range(len(lengths)):
                lengths[i] = min(lengths[i], common)
        else:
            ceiling = _maximum_length(directions[favored_index], height, width)
            if lengths[favored_index] <= lengths[other_index]:
                lengths[favored_index] = min(ceiling, lengths[other_index] + 1)
            for i in other:
                lengths[i] = min(lengths[i], lengths[favored_index] - 1)
            if max(lengths[i] for i in favored) <= max(lengths[i] for i in other):
                lengths[favored_index] = min(ceiling, max(lengths[i] for i in other) + 1)
    return lengths


def _choose_colors(rng, directions, winner, background, reserved, split, scheme, include_new):
    pool = [c for c in range(1, 10) if c != background and (split == "test" or c != reserved)]
    colors = [int(rng.choice(pool)) for _ in directions]
    if scheme == "unique":
        colors = [int(c) for c in rng.choice(pool, size=len(directions), replace=False)]
    kept = [i for i, d in enumerate(directions) if d == winner]
    removed = [i for i, d in enumerate(directions) if d != winner]
    if scheme == "kept_share" and len(kept) > 1:
        colors[kept[1]] = colors[kept[0]]
    if scheme == "cross_role" and removed:
        colors[removed[0]] = colors[kept[0]]
    if scheme == "distractors_share" and len(removed) > 1:
        first = removed[0]
        second = next((i for i in removed[1:] if directions[i] != directions[first]), removed[1])
        colors[second] = colors[first]
    if include_new:
        colors[int(rng.integers(0, len(colors)))] = reserved
    return colors


def _candidate_cells(rng, direction, length, height, width):
    dr, dc = _steps(direction)
    low_row = max(0, -(length - 1) * dr)
    high_row = min(height - 1, height - 1 - (length - 1) * dr)
    low_col = max(0, -(length - 1) * dc)
    high_col = min(width - 1, width - 1 - (length - 1) * dc)
    row = int(rng.integers(low_row, high_row + 1))
    col = int(rng.integers(low_col, high_col + 1))
    return [(row + dr * k, col + dc * k) for k in range(length)]


def _clear(cells, color, direction, placed):
    for previous in placed:
        for row, col in cells:
            for rr, cc in previous["cells"]:
                distance = max(abs(row - rr), abs(col - cc))
                if distance == 0:
                    return False
                if distance == 1:
                    if color == previous["color"] or direction == previous["orientation"]:
                        return False
                    if abs(row - rr) + abs(col - cc) == 1:
                        return False
    return True


def _render(rng, directions, winner, background, reserved, split, scheme, pattern, include_new):
    minimum_density = 0.05 if len(directions) == 2 else 0.08
    for _ in range(100):
        height, width = _dimensions(rng)
        lengths = _choose_lengths(rng, directions, height, width, winner, pattern)
        if not minimum_density <= sum(lengths) / (height * width) <= 0.40:
            continue
        colors = _choose_colors(rng, directions, winner, background, reserved, split, scheme, include_new)
        specs = list(zip(directions, lengths, colors))
        placed = []
        for index in rng.permutation(len(specs)):
            direction, length, color = specs[int(index)]
            for _ in range(170):
                cells = _candidate_cells(rng, direction, length, height, width)
                if _clear(cells, color, direction, placed):
                    placed.append({"cells": cells, "color": color, "orientation": direction})
                    break
            else:
                break
        if len(placed) == len(specs):
            grid = np.full((height, width), background, dtype=np.int8)
            for segment in placed:
                for row, col in segment["cells"]:
                    grid[row, col] = segment["color"]
            if len(_analyze_segments(grid)) == len(directions):
                return grid
    # A roomy deterministic construction is used only if dense random packing fails.
    height = width = 24
    for _ in range(400):
        lengths = _choose_lengths(rng, directions, height, width, winner, pattern)
        lengths = [min(length, 14) for length in lengths]
        if sum(lengths) / (height * width) < minimum_density:
            continue
        colors = _choose_colors(rng, directions, winner, background, reserved, split, scheme, include_new)
        placed = []
        for index in rng.permutation(len(directions)):
            i = int(index)
            for _ in range(200):
                cells = _candidate_cells(rng, directions[i], lengths[i], height, width)
                if _clear(cells, colors[i], directions[i], placed):
                    placed.append({"cells": cells, "color": colors[i], "orientation": directions[i]})
                    break
            else:
                break
        if len(placed) == len(directions):
            grid = np.full((height, width), background, dtype=np.int8)
            for segment in placed:
                for row, col in segment["cells"]:
                    grid[row, col] = segment["color"]
            return grid
    raise AssertionError("Could not place distinct lines")


def _pair(rng, winner, slope, mode, background, reserved, split, scheme, pattern, include_new=False):
    others = [direction for direction in ("H", "V", slope) if direction != winner]
    others = [others[i] for i in rng.permutation(2)]
    directions = [winner] * mode[0] + [others[0]] * mode[1] + [others[1]] * mode[2]
    directions = [directions[i] for i in rng.permutation(len(directions))]
    grid = _render(rng, directions, winner, background, reserved, split, scheme, pattern, include_new)
    return {"input": grid, "output": _solve(grid)}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _choose_counts(seed, rng, num_train, num_test)
    # Only the choice of diagonal slope is a puzzle-wide visual convention.
    slope = ("D", "A")[int(rng.integers(0, 2))]
    reserved = int(rng.integers(1, 10))
    directions = ["H", "V", slope]
    if rng.random() < 0.45:
        first_winners = ["H", "V"]
    else:
        first_winners = [directions[i] for i in rng.choice(3, size=2, replace=False)]
    if rng.random() < 0.5:
        first_winners.reverse()
    long_patterns = ["kept", "erased"]
    if rng.random() < 0.5:
        long_patterns.reverse()
    first_modes = [_MODES[4], _MODES[int(rng.choice((2, 3)))]]
    if rng.random() < 0.5:
        first_modes.reverse()
    nonzero_train_index = int(rng.integers(0, num_train))
    cross_role_train_index = int(rng.integers(0, 2))
    new_test_index = int(rng.integers(0, num_test))
    streams = np.random.SeedSequence(seed).spawn(num_train + num_test)
    train, test = [], []
    train_winners = []
    for index in range(num_train + num_test):
        local = np.random.default_rng(streams[index])
        split = "train" if index < num_train else "test"
        local_index = index if split == "train" else index - num_train
        if split == "train":
            winner = first_winners[local_index] if local_index < 2 else directions[int(local.integers(0, 3))]
            mode = first_modes[local_index] if local_index < 2 else _MODES[int(local.integers(0, len(_MODES)))]
            pattern = long_patterns[local_index] if local_index < 2 else ("kept", "erased", "tie", "free")[int(local.integers(0, 4))]
            scheme = ("cross_role" if local_index == cross_role_train_index else "unique") if local_index < 2 else ("unique", "random", "kept_share", "cross_role", "distractors_share")[int(local.integers(0, 5))]
            use_nonzero = local_index == nonzero_train_index or (local_index >= 2 and local.random() < 0.30)
            train_winners.append(winner)
        else:
            unseen = [direction for direction in directions if direction not in train_winners]
            winner = unseen[int(local.integers(0, len(unseen)))] if unseen and local.random() < 0.5 else directions[int(local.integers(0, 3))]
            mode = _MODES[int(local.choice(len(_MODES), p=(0.16, 0.12, 0.25, 0.20, 0.27)))]
            pattern = ("kept", "erased", "tie", "free")[int(local.integers(0, 4))]
            scheme = ("unique", "random", "kept_share", "cross_role", "distractors_share")[int(local.integers(0, 5))]
            use_nonzero = local.random() < 0.35
        background = int(local.choice([c for c in range(1, 10) if c != reserved])) if use_nonzero else 0
        pair = _pair(local, winner, slope, mode, background, reserved, split, scheme, pattern, split == "test" and local_index == new_test_index)
        (train if split == "train" else test).append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    return grid.shape, tuple(int(c) for c in grid.flat)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs = set()
    pairs = set()
    analyses = {"train": [], "test": []}
    slopes = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert grid.dtype.kind in "iu"
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.min(grid) >= 0 and np.max(grid) <= 9
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape
            input_key = _key(inp)
            pair_key = (input_key, _key(out))
            assert input_key not in inputs and pair_key not in pairs
            inputs.add(input_key)
            pairs.add(pair_key)
            assert not (inp.shape == (10, 10) and input_key[1] in _SOURCE_KEYS)
            background = _background(inp)
            assert np.count_nonzero(inp != background) <= inp.size * 0.45
            segments = _analyze_segments(inp)
            assert 2 <= len(segments) <= 6
            assert np.count_nonzero(inp != background) / inp.size >= (0.05 if len(segments) == 2 else 0.08)
            winner = _winner(segments)
            assert np.array_equal(out, _solve(inp))
            assert {int(x) for x in np.unique(out)} <= {int(x) for x in np.unique(inp)}
            assert np.all(out[inp == background] == background)
            for segment in segments:
                expected = segment["color"] if segment["orientation"] == winner else background
                assert all(int(out[row, col]) == expected for row, col in segment["cells"])
                if segment["orientation"] in ("D", "A"):
                    slopes.add(segment["orientation"])
            # Contacts are allowed only diagonally, in different colors and directions.
            for i, first in enumerate(segments):
                for second in segments[i + 1:]:
                    assert _clear(first["cells"], first["color"], first["orientation"], [second])
            analyses[split].append((segments, winner, background))
    assert len(slopes) == 1
    training = analyses["train"]
    assert {s["orientation"] for segments, _, _ in training for s in segments if s["orientation"] in ("D", "A")} == slopes
    assert len({training[0][1], training[1][1]}) == 2
    assert all(len(segments) >= 4 and len(segments) > sum(s["orientation"] == winner for s in segments) for segments, winner, _ in training[:2])
    assert any(_background(pair["input"]) != 0 for pair in puzzle["train"])
    assert any(_background(pair["input"]) == 0 for pair in puzzle["train"])
    assert any(len({s["color"] for s in segments}) < len(segments) and any(s["color"] == t["color"] and (s["orientation"] == winner) != (t["orientation"] == winner) for s in segments for t in segments) for segments, winner, _ in training[:2])
    assert any(sum(s["orientation"] == winner for s in segments) == 3 and len(segments) == 6 for segments, winner, _ in training[:2])
    length_relations = set()
    for segments, winner, _ in training[:2]:
        kept = max(s["length"] for s in segments if s["orientation"] == winner)
        erased = max(s["length"] for s in segments if s["orientation"] != winner)
        length_relations.add((kept > erased) - (kept < erased))
    assert length_relations == {-1, 1}
    train_colors = {int(c) for pair in puzzle["train"] for c in np.unique(pair["input"])}
    assert any(s["color"] not in train_colors for segments, _, _ in analyses["test"] for s in segments)


def _equal(left, right):
    return all(len(left[split]) == len(right[split]) and all(np.array_equal(a[name], b[name]) for a, b in zip(left[split], right[split]) for name in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    shapes, layouts, palettes, fingerprints = set(), set(), set(), set()
    train_counts, test_counts = set(), set()
    counts_by_split = {"train": set(), "test": set()}
    winners_by_split = {"train": set(), "test": set()}
    backgrounds = set()
    slopes = set()
    unchanged_slots = set()
    shared_distractor_slots = set()
    length_patterns = set()
    touching = 0
    full_spans = 0
    kept_share = 0
    cross_role = 0
    compact_nonzero = 0
    source_like_shifts = 0
    repeated_winners = 0
    within_shape_variation = 0
    within_palette_variation = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _equal(puzzle, generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(tuple(_key(pair["input"]) for split in ("train", "test") for pair in puzzle[split]))
        palettes.add(tuple(sorted({int(c) for pair in puzzle["train"] for c in np.unique(pair["input"])})))
        local_shapes = set()
        local_palettes = set()
        train_winners = set()
        test_winners = set()
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                inp = pair["input"]
                segments = _analyze_segments(inp)
                winner = _winner(segments)
                shapes.add(inp.shape)
                local_shapes.add(inp.shape)
                local_palettes.add(tuple(int(c) for c in np.unique(inp)))
                counts_by_split[split].add(len(segments))
                winners_by_split[split].add(winner)
                (train_winners if split == "train" else test_winners).add(winner)
                backgrounds.add(_background(inp))
                slopes.update(s["orientation"] for s in segments if s["orientation"] in ("D", "A"))
                layouts.add((inp.shape, tuple(sorted((s["orientation"], s["length"], min(r for r, _ in s["cells"]), min(c for _, c in s["cells"])) for s in segments))))
                assert np.array_equal(pair["output"], _solve(inp))
                if split == "test" and np.array_equal(inp, pair["output"]):
                    unchanged_slots.add(index)
                removed = [s for s in segments if s["orientation"] != winner]
                retained = [s for s in segments if s["orientation"] == winner]
                if len({s["color"] for s in retained}) < len(retained):
                    kept_share += 1
                if any(s["color"] == t["color"] for s in retained for t in removed):
                    cross_role += 1
                if inp.shape == (10, 10) and _background(inp) != 0:
                    compact_nonzero += 1
                if split == "test" and any(a["orientation"] != b["orientation"] and a["color"] == b["color"] for a in removed for b in removed):
                    shared_distractor_slots.add(index)
                if removed:
                    a = max(s["length"] for s in segments if s["orientation"] == winner)
                    b = max(s["length"] for s in removed)
                    length_patterns.add("kept" if a > b else "erased" if a < b else "tie")
                if any(s["length"] == _maximum_length(s["orientation"], *inp.shape) for s in segments):
                    full_spans += 1
                for i, s in enumerate(segments):
                    for t in segments[i + 1:]:
                        if s["color"] != t["color"] and s["orientation"] != t["orientation"] and any(abs(r - rr) == 1 and abs(c - cc) == 1 for r, c in s["cells"] for rr, cc in t["cells"]):
                            touching += 1
        within_shape_variation += len(local_shapes) > 1
        within_palette_variation += len(local_palettes) > 1
        source_like_shifts += train_winners == {"H", "V"} and bool(test_winners & {"D", "A"})
        repeated_winners += bool(train_winners & test_winners)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 40 and (10, 10) in shapes
    assert len(layouts) >= 200 and len(fingerprints) == 42
    assert len(palettes) >= 15
    assert {2, 3, 4, 5, 6} <= counts_by_split["test"]
    assert {2, 3, 4, 5, 6} <= counts_by_split["train"]
    assert {"H", "V", "D", "A"} <= winners_by_split["train"] and {"H", "V", "D", "A"} <= winners_by_split["test"]
    assert len(backgrounds) >= 5 and slopes == {"D", "A"}
    assert len(unchanged_slots) >= 2 and len(shared_distractor_slots) >= 2
    assert length_patterns == {"kept", "erased", "tie"}
    assert touching >= 10 and full_spans >= 10
    assert kept_share >= 10 and cross_role >= 42 and compact_nonzero >= 2
    assert source_like_shifts >= 3 and repeated_winners >= 20
    assert within_shape_variation >= 38 and within_palette_variation == 42
    for offset, (a, b) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(100 + offset, a, b)
        validate_generated_puzzle(puzzle)
        assert _equal(puzzle, generate_puzzle(100 + offset, a, b))
        if a is not None:
            assert len(puzzle["train"]) == a
        if b is not None:
            assert len(puzzle["test"]) == b
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(999, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal pair counts were accepted")
