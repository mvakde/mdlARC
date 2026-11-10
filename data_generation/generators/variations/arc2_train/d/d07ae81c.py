"""Generate rectilinear-band ARC puzzles with seed-driven diagonal traces."""

import hashlib

import numpy as np


# SHA-256 of the four source inputs, flattened as ARC color bytes. Shape is
# included in the comparison; source-sized but different grids remain legal.
_SOURCE_INPUTS = {
    ((12, 14), "032d2e779fe70de4e56b518b6e7fad29de4cea4399210c285dd41434184e7b10"),
    ((19, 12), "036ecf6ae40ec7fce0a678943cf93613465c21de42246202e6579938f9e7d0ec"),
    ((15, 15), "ae5ade649e423d5b5eba65d433b4e056c0b8af79cdcdd0a0f1889f247b27c6d3"),
    ((17, 19), "e64b00c14939105775bdd044d18928698ef33a469d2407de979aed6f4f962426"),
}


def _checked_count(value, name, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _groups(mask):
    result = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(mask) - 1):
            end = i if value and i == len(mask) - 1 else i - 1
            result.append((start, end))
            start = None
    return result


def _band_axis(rng, length, count):
    """Sample separated full-span bands, keeping the unbanded role identifiable."""
    result = np.zeros(length, dtype=bool)
    if count == 0:
        return result
    budget = (length - 1) // 2
    widths = [1] * count
    for _ in range(max(0, budget - count)):
        if rng.random() < 0.68:
            choices = [i for i, width in enumerate(widths) if width < 6]
            if choices:
                widths[int(rng.choice(choices))] += 1
    rng.shuffle(widths)
    extra = length - sum(widths) - (count - 1)
    # Distribute background among both edges and the mandatory inner gaps.
    cuts = sorted(int(x) for x in rng.integers(0, extra + 1, size=count))
    gaps = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, count)] + [extra - cuts[-1]]
    pos = gaps[0]
    for i, width in enumerate(widths):
        result[pos:pos + width] = True
        pos += width + (1 + gaps[i + 1] if i < count - 1 else 0)
    return result


def _sample_shape(rng, regime):
    if regime == "source_train":
        mode = int(rng.integers(3))
        if mode == 0:
            return int(rng.integers(10, 17)), int(rng.integers(10, 17))
        if mode == 1:
            return int(rng.integers(13, 21)), int(rng.integers(13, 22))
        return int(rng.integers(17, 25)), int(rng.integers(14, 25))
    mode = int(rng.integers(3))
    if mode == 0:
        return int(rng.integers(10, 17)), int(rng.integers(10, 18))
    if mode == 1:
        return int(rng.integers(15, 25)), int(rng.integers(14, 26))
    return int(rng.integers(21, 31)), int(rng.integers(21, 31))


def _sample_layout(rng, regime, index):
    height, width = _sample_shape(rng, regime)
    if regime == "source_train" and index == 0:
        orientation = "mixed"
    elif regime == "source_train" and index == 1:
        orientation = "horizontal" if rng.random() < 0.5 else "vertical"
    else:
        orientation = str(rng.choice(["horizontal", "vertical", "mixed"], p=[0.28, 0.28, 0.44]))
    row_count = int(rng.integers(1, 4)) if orientation != "vertical" else 0
    col_count = int(rng.integers(1, 4)) if orientation != "horizontal" else 0
    rows = _band_axis(rng, height, row_count)
    cols = _band_axis(rng, width, col_count)
    return rows[:, None] | cols[None, :]


def _seed_lines(seeds):
    return {r + c for r, c in seeds}, {r - c for r, c in seeds}


def _sample_seeds(rng, stripe_mask, motif, count):
    coords = [tuple(map(int, p)) for p in np.argwhere(np.ones(stripe_mask.shape, bool))]
    field = [p for p in coords if not stripe_mask[p]]
    stripe = [p for p in coords if stripe_mask[p]]
    if not field or not stripe:
        raise ValueError("both base roles are required")
    pools = (field, stripe)
    for _ in range(350):
        seeds = []
        if motif == "independent":
            seeds = [field[int(rng.integers(len(field)))], stripe[int(rng.integers(len(stripe)))]]
            a, b = seeds
            if abs(a[0] - b[0]) == abs(a[1] - b[1]):
                continue
        elif motif in ("cross_pair", "same_pair"):
            role = int(rng.integers(2))
            first = pools[role][int(rng.integers(len(pools[role])))]
            partner_role = 1 - role if motif == "cross_pair" else role
            candidates = [p for p in pools[partner_role] if p != first and abs(p[0] - first[0]) == abs(p[1] - first[1])]
            if not candidates:
                continue
            seeds = [first, candidates[int(rng.integers(len(candidates)))]]
            if motif == "same_pair":
                seeds.append(pools[1 - role][int(rng.integers(len(pools[1 - role])))])
        else:
            role = int(rng.integers(2))
            center = pools[role][int(rng.integers(len(pools[role])))]
            same = [p for p in pools[role] if p != center and
                    (p[0] - center[0]) == (p[1] - center[1])]
            opposite = [p for p in pools[1 - role] if p != center and
                        (p[0] - center[0]) == -(p[1] - center[1])]
            if not same or not opposite:
                continue
            a = same[int(rng.integers(len(same)))]
            if motif == "wedge":
                opposite = [p for p in opposite if p[0] == a[0]]
            else:
                opposite = [p for p in opposite if p[0] != a[0]]
            if not opposite:
                continue
            b = opposite[int(rng.integers(len(opposite)))]
            seeds = [center, a, b]
        if len(set(seeds)) != len(seeds):
            continue
        if not ({bool(stripe_mask[p]) for p in seeds} == {False, True}):
            continue
        while len(seeds) < count:
            role = int(rng.integers(2))
            pool = pools[role]
            choices = [p for p in pool if p not in seeds and
                       (p[0] + p[1] not in _seed_lines(seeds)[0] or
                        p[0] - p[1] not in _seed_lines(seeds)[1])]
            if not choices:
                break
            seeds.append(choices[int(rng.integers(len(choices)))])
        if len(seeds) == count:
            return seeds
    raise ValueError("could not place the requested seed motif")


def _diagonal_mask(shape, seeds):
    rows, cols = np.indices(shape)
    sums, differences = _seed_lines(seeds)
    mask = np.zeros(shape, dtype=bool)
    for line in sums:
        mask |= rows + cols == line
    for line in differences:
        mask |= rows - cols == line
    return mask


def _build_grids(stripe_mask, seeds, colors):
    field, stripe, field_trace, stripe_trace = colors
    base = np.full(stripe_mask.shape, field, dtype=np.int8)
    base[stripe_mask] = stripe
    input_grid = base.copy()
    for p in seeds:
        input_grid[p] = stripe_trace if stripe_mask[p] else field_trace
    output_grid = base.copy()
    diagonal = _diagonal_mask(base.shape, seeds)
    output_grid[diagonal & ~stripe_mask] = field_trace
    output_grid[diagonal & stripe_mask] = stripe_trace
    return input_grid, output_grid


def _source_copy(input_grid):
    digest = hashlib.sha256(np.asarray(input_grid, dtype=np.uint8).tobytes()).hexdigest()
    return (input_grid.shape, digest) in _SOURCE_INPUTS


def _analyze_input(input_grid):
    values, counts = np.unique(input_grid, return_counts=True)
    assert len(values) == 4
    order = sorted(range(4), key=lambda i: -int(counts[i]))
    assert min(int(counts[i]) for i in order[:2]) > max(int(counts[i]) for i in order[2:])
    bases = [int(values[i]) for i in order[:2]]
    traces = [int(values[i]) for i in order[2:]]
    candidates = []
    for field, stripe in ((bases[0], bases[1]), (bases[1], bases[0])):
        f = input_grid == field
        s = input_grid == stripe
        rows = np.sum(s, axis=1) > np.sum(f, axis=1)
        cols = np.sum(s, axis=0) > np.sum(f, axis=0)
        if not (np.any(rows) or np.any(cols)):
            continue
        if 2 * int(np.sum(rows)) >= len(rows) or 2 * int(np.sum(cols)) >= len(cols):
            continue
        row_groups, col_groups = _groups(rows), _groups(cols)
        if len(row_groups) > 3 or len(col_groups) > 3:
            continue
        if any(b - a + 1 > 6 for a, b in row_groups + col_groups):
            continue
        mask = rows[:, None] | cols[None, :]
        base = np.full(input_grid.shape, field, dtype=np.int8)
        base[mask] = stripe
        intact = f | s
        if not np.all(input_grid[intact] == base[intact]):
            continue
        mapping = {}
        for trace in traces:
            locations = input_grid == trace
            roles = np.unique(mask[locations])
            if len(roles) != 1:
                break
            mapping[bool(roles[0])] = trace
        if len(mapping) != 2:
            continue
        seeds = [tuple(map(int, p)) for p in np.argwhere((input_grid == traces[0]) | (input_grid == traces[1]))]
        candidates.append((mask, base, mapping, seeds, rows, cols, (field, stripe, mapping[False], mapping[True])))
    assert len(candidates) == 1
    return candidates[0]


def _pair_valid(input_grid, output_grid):
    analysis = _analyze_input(input_grid)
    mask, base, mapping, seeds, rows, cols, colors = analysis
    assert 2 <= len(seeds) <= 5
    assert {bool(mask[p]) for p in seeds} == {False, True}
    diagonal = _diagonal_mask(input_grid.shape, seeds)
    expected = base.copy()
    expected[diagonal & ~mask] = mapping[False]
    expected[diagonal & mask] = mapping[True]
    assert np.array_equal(output_grid, expected)
    assert not np.array_equal(input_grid, output_grid)
    assert set(int(x) for x in np.unique(output_grid)) == set(colors)
    for trace in mapping.values():
        assert np.any((output_grid == trace) & (input_grid != trace))
    return analysis


def _sample_colors(rng, previous=None, swap=False):
    if swap:
        return (previous[1], previous[0], previous[3], previous[2])
    if previous is not None and rng.random() < 0.45:
        colors = list(previous)
        rng.shuffle(colors)
        return tuple(colors)
    return tuple(int(x) for x in rng.permutation(10)[:4])


def _generate_pair(rng, regime, index, colors):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    motifs = ["independent", "cross_pair", "same_pair", "chain", "wedge"]
    for _ in range(160):
        mask = _sample_layout(rng, regime, index)
        motif = str(rng.choice(motifs, p=[0.19, 0.24, 0.18, 0.23, 0.16]))
        minimum = 3 if motif in ("same_pair", "chain", "wedge") else 2
        count = int(rng.integers(minimum, 6 if regime == "target_test" else 5))
        try:
            seeds = _sample_seeds(rng, mask, motif, count)
        except ValueError:
            continue
        input_grid, output_grid = _build_grids(mask, seeds, colors)
        if _source_copy(input_grid):
            continue
        try:
            _pair_valid(input_grid, output_grid)
        except AssertionError:
            continue
        return input_grid, output_grid
    raise RuntimeError("could not generate a recoverable pair")


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    # The only puzzle-wide convention is the fixed source rule: both complete
    # diagonals from original seeds, colored by the local underlying base role.
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max,
                              size=num_train + num_test, dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_colors = None
    for split, count, offset in (("train", num_train, 0), ("test", num_test, num_train)):
        regime = "source_train" if split == "train" else "target_test"
        for index in range(count):
            pair_rng = np.random.default_rng(int(pair_seeds[offset + index]))
            colors = _sample_colors(pair_rng, previous_colors, swap=(split == "train" and index == 1))
            for _ in range(10):
                input_grid, output_grid = _generate_pair(pair_rng, regime, index, colors)
                if _grid_key(input_grid) not in seen:
                    break
            else:
                raise RuntimeError("duplicate input")
            seen.add(_grid_key(input_grid))
            puzzle[split].append({"input": input_grid, "output": output_grid})
            previous_colors = colors
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    training_colors = []
    orientations = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = pair["input"], pair["output"]
            for grid in (a, b):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            assert a.shape == b.shape
            assert not _source_copy(a)
            key, pair_key = _grid_key(a), (_grid_key(a), _grid_key(b))
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            mask, base, mapping, seeds, rows, cols, colors = _pair_valid(a, b)
            assert len(_groups(rows)) <= 3 and len(_groups(cols)) <= 3
            if split == "train":
                training_colors.append(colors)
                orientations.append(("horizontal" if np.any(rows) else "") +
                                    ("vertical" if np.any(cols) else ""))
    assert training_colors[0] != training_colors[1]
    assert orientations[0] == "horizontalvertical"
    assert orientations[1] in ("horizontal", "vertical")


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, orientations, palettes, line_profiles, widths = set(), set(), set(), set(), set()
    all_inputs = set()
    smaller_tests = 0
    test_orientations, test_aspects, test_seed_counts = set(), set(), set()
    shared_slopes, shared_roles, junction_roles = set(), set(), set()
    varied_seed_count_puzzles = 0
    edge_bands = mixed_wide = source_sized = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_areas = [p["input"].size for p in puzzle["train"]]
        puzzle_seed_counts = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                mask, base, mapping, seeds, rows, cols, colors = _pair_valid(grid, pair["output"])
                assert not _source_copy(grid)
                shapes.add(grid.shape)
                all_inputs.add(_grid_key(grid))
                palettes.add(colors)
                orientation = ("horizontal" if np.any(rows) else "") + (
                    "vertical" if np.any(cols) else "")
                orientations.add(orientation)
                groups = _groups(rows) + _groups(cols)
                widths.update(b - a + 1 for a, b in groups)
                edge_bands += any(a == 0 or b == len(rows) - 1
                                  for a, b in _groups(rows)) or any(
                                      a == 0 or b == len(cols) - 1
                                      for a, b in _groups(cols))
                mixed_wide += bool(np.any(rows) and np.any(cols) and
                                   any(b - a + 1 >= 4 for a, b in groups))
                source_sized += grid.shape in {x[0] for x in _SOURCE_INPUTS}
                puzzle_seed_counts.add(len(seeds))
                line_profiles.add((len(seeds), len(_seed_lines(seeds)[0]),
                                   len(_seed_lines(seeds)[1])))
                edges = []
                for i, first in enumerate(seeds):
                    for j, second in enumerate(seeds[:i]):
                        dr, dc = first[0] - second[0], first[1] - second[1]
                        if abs(dr) == abs(dc):
                            slope = "plus" if dr == dc else "minus"
                            same_role = bool(mask[first]) == bool(mask[second])
                            edges.append((i, j, slope, same_role))
                            shared_slopes.add(slope)
                            shared_roles.add("same" if same_role else "cross")
                for i, seed_position in enumerate(seeds):
                    incident = [e for e in edges if i in e[:2]]
                    if {"plus", "minus"} <= {e[2] for e in incident} and (
                        {True, False} <= {e[3] for e in incident}
                    ):
                        junction_roles.add(bool(mask[seed_position]))
                if split == "test" and grid.size <= max(train_areas):
                    smaller_tests += 1
                if split == "test":
                    test_orientations.add(orientation)
                    test_seed_counts.add(len(seeds))
                    h, w = grid.shape
                    test_aspects.add("square" if h == w else
                                     "portrait" if h > w else "landscape")
        varied_seed_count_puzzles += len(puzzle_seed_counts) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 40 and len(all_inputs) >= 120
    assert len(palettes) >= 60 and len(line_profiles) >= 10
    assert orientations == {"horizontal", "vertical", "horizontalvertical"}
    assert {4, 5, 6}.issubset(widths) and smaller_tests > 0
    assert test_orientations == orientations
    assert test_aspects == {"square", "portrait", "landscape"}
    assert {2, 3, 4, 5}.issubset(test_seed_counts)
    assert shared_slopes == {"plus", "minus"}
    assert shared_roles == {"same", "cross"}
    assert junction_roles == {False, True}
    assert varied_seed_count_puzzles >= 25
    assert edge_bands > 0 and mixed_wide > 0 and source_sized > 0
    for seed, counts in enumerate(((2, 1), (10, 1), (10, 4), (6, 2)), start=100):
        p = generate_puzzle(seed, *counts)
        assert (len(p["train"]), len(p["test"])) == counts
        validate_generated_puzzle(p)
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(seed, *counts))
    for seed, counts in enumerate(((None, 1), (None, 4), (2, None), (10, None)), start=200):
        p = generate_puzzle(seed, *counts)
        validate_generated_puzzle(p)
        if counts[0] is not None:
            assert len(p["train"]) == counts[0]
        if counts[1] is not None:
            assert len(p["test"]) == counts[1]
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(999, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
