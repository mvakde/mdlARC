"""Generate component-area recoloring puzzles from the reviewed ARC family."""

import numpy as np


_SOURCE_ROWS = (
    ("220002", "220002", "000200", "020000", "000202", "022200"),
    ("000000000000", "222000200000", "022000220000", "022200220020",
     "000000000020", "000002000000", "020022000222", "000000000222",
     "002000020020", "000002200000"),
    ("000000000", "022000000", "002000220", "000002200", "000000000",
     "000200000", "022000000", "000000020", "000000020", "000200000"),
    ("00000000", "22000220", "02200220", "00000022", "00000000",
     "02220000", "00000200", "00220000", "20220022", "20000000"),
    ("00002220", "22000000", "02200220", "02000220", "00000000",
     "20220002"),
)
_SOURCE_INPUTS = tuple(np.array([[int(c) for c in row] for row in rows],
                                dtype=np.int8) for rows in _SOURCE_ROWS)
_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _validate_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4 + 3 * seed) % (11 - minimum)
    elif num_train is None:
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 3 + 5 * seed) % (11 - minimum)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _components(grid, color):
    remaining = set(map(tuple, np.argwhere(grid == color)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component = []
        while stack:
            cell = stack.pop()
            component.append(cell)
            for dr, dc in _DIRECTIONS:
                neighbor = (cell[0] + dr, cell[1] + dc)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        result.append(component)
    return result


def _recolor(grid, source, highlight):
    output = grid.copy()
    for component in _components(grid, source):
        if len(component) >= 4:
            for r, c in component:
                output[r, c] = highlight
    return output


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _orient(rng, cells):
    cells = list(cells)
    if int(rng.integers(2)):
        cells = [(r, -c) for r, c in cells]
    for _ in range(int(rng.integers(4))):
        cells = [(c, -r) for r, c in cells]
    return _normalize(cells)


def _shape(rng, area, style=None):
    if area <= 2:
        return _orient(rng, {(0, c) for c in range(area)})
    if style is None:
        if area == 3:
            style = "l3" if rng.random() < .55 else "line"
        elif area == 4:
            style = str(rng.choice(["square", "tee", "bend", "line", "organic"],
                                   p=[.34, .22, .16, .12, .16]))
        else:
            style = str(rng.choice(["compact", "organic", "line", "bend"],
                                   p=[.42, .35, .13, .10]))
    if style == "square":
        cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
    elif style == "l3":
        cells = {(0, 0), (1, 0), (1, 1)}
    elif style == "tee":
        cells = {(0, 0), (0, 1), (0, 2), (1, 1)}
    elif style == "line":
        cells = {(0, c) for c in range(area)}
    elif style == "bend":
        arm = max(2, area // 2 + 1)
        cells = {(0, c) for c in range(arm)}
        cells.update((r, 0) for r in range(area - arm + 1))
    else:
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                frontier.update((r + dr, c + dc) for dr, dc in _DIRECTIONS)
            frontier.difference_update(cells)
            ordered = sorted(frontier)
            if style == "compact":
                weights = np.array([1 + 5 * sum((r + dr, c + dc) in cells
                                                for dr, dc in _DIRECTIONS)
                                    for r, c in ordered], dtype=float)
                weights = weights ** 2
                choice = ordered[int(rng.choice(len(ordered), p=weights / weights.sum()))]
            else:
                choice = ordered[int(rng.integers(len(ordered)))]
            cells.add(choice)
    return _orient(rng, cells)


def _diagonal_contact(a, b):
    return any((r + dr, c + dc) in b for r, c in a for dr, dc in _DIAGONALS)


def _orthogonal_contact(a, b):
    return any((r + dr, c + dc) in b for r, c in a for dr, dc in _DIRECTIONS)


def _placements(shape, height, width):
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1
    if sh > height or sw > width:
        return []
    return [(r, c) for r in range(height - sh + 1)
            for c in range(width - sw + 1)]


def _place(rng, height, width, shapes, mode, diagnostic=False):
    """Pack separate 4-components; mode may require or forbid corner contact."""
    for _ in range(32):
        occupied = set()
        first = set()
        remaining = list(shapes)
        if diagnostic:
            tri = {(0, 0), (1, 0), (1, 1)}
            square = {(2, 2), (2, 3), (3, 2), (3, 3)}
            if int(rng.integers(2)):
                tri = {(r, -c) for r, c in tri}
                square = {(r, -c) for r, c in square}
            for _ in range(int(rng.integers(4))):
                tri = {(c, -r) for r, c in tri}
                square = {(c, -r) for r, c in square}
            union = tri | square
            mr = min(r for r, _ in union)
            mc = min(c for _, c in union)
            tri = {(r - mr, c - mc) for r, c in tri}
            square = {(r - mr, c - mc) for r, c in square}
            places = _placements(tri | square, height, width)
            if not places:
                return None
            top, left = places[int(rng.integers(len(places)))]
            occupied = {(r + top, c + left) for r, c in tri | square}
            first = occupied.copy()
        else:
            if mode == "diag":
                small = next((i for i, shape in enumerate(remaining)
                              if len(shape) <= 3), None)
                large = next((i for i, shape in enumerate(remaining)
                              if len(shape) >= 4), None)
                if small is None or large is None:
                    mode = "free"
                else:
                    selected = [remaining[large], remaining[small]]
                    remaining = [shape for i, shape in enumerate(remaining)
                                 if i not in (large, small)]
                    remaining = selected + sorted(remaining, key=len, reverse=True)
            if mode != "diag":
                remaining.sort(key=len, reverse=True)
        succeeded = True
        for index, shape in enumerate(remaining):
            positions = _placements(shape, height, width)
            if not positions:
                succeeded = False
                break
            order = rng.permutation(len(positions))
            placed = False
            for k in order:
                top, left = positions[int(k)]
                candidate = {(r + top, c + left) for r, c in shape}
                if candidate & occupied or _orthogonal_contact(candidate, occupied):
                    continue
                if mode == "gap" and _diagonal_contact(candidate, occupied):
                    continue
                if mode == "diag" and index == 1 and not _diagonal_contact(candidate, first):
                    continue
                occupied.update(candidate)
                if index == 0 and not diagnostic:
                    first = candidate
                placed = True
                break
            if not placed:
                succeeded = False
                break
        if succeeded:
            return occupied
    return None


def _dimensions(rng, regime, role):
    if role == "contrast":
        return int(rng.integers(9, 15)), int(rng.integers(10, 17))
    p = rng.random()
    if p < (.30 if regime == "target_test" else .22):
        options = ((6, 6), (6, 8), (7, 7), (7, 8), (8, 6))
        return options[int(rng.integers(len(options)))]
    if p < .78:
        return int(rng.integers(7, 12)), int(rng.integers(8, 15))
    return int(rng.integers(11, 17)), int(rng.integers(12, 19))


def _area(rng, regime, klass=None):
    if klass == "small":
        return int(rng.integers(1, 4))
    if klass == "large":
        return int(rng.integers(4, 17 if regime == "target_test" else 13))
    if rng.random() < .59:
        return int(rng.integers(1, 4))
    if rng.random() < .82:
        return int(rng.integers(4, 9))
    return int(rng.integers(9, 17 if regime == "target_test" else 13))


def _profile(rng, regime, role, height, width, borrowed=None):
    if borrowed is not None:
        return list(borrowed)
    capacity = height * width
    max_count = min(9, max(3, capacity // 9))
    if role == "diagnostic":
        areas = [3, 4]
        extra_count = int(rng.integers(0, min(max_count - 2, 5) + 1))
    elif role == "contrast":
        areas = [4, 4, int(rng.integers(5, 11))]
        extra_count = int(rng.integers(0, min(max_count - 3, 5) + 1))
    else:
        total_count = int(rng.integers(2, max_count + 1))
        class_draw = rng.random()
        if class_draw < .12:
            areas = [_area(rng, regime, "small") for _ in range(total_count)]
        elif class_draw < .25:
            areas = [_area(rng, regime, "large") for _ in range(total_count)]
        else:
            areas = [_area(rng, regime, "small"), _area(rng, regime, "large")]
            areas.extend(_area(rng, regime) for _ in range(total_count - 2))
        extra_count = 0
    areas.extend(_area(rng, regime) for _ in range(extra_count))
    return areas


def _grid_key(grid):
    return (tuple(grid.shape), grid.astype(np.int8, copy=False).tobytes())


def _matches_source_input(grid):
    return any(grid.shape == old.shape and np.array_equal(grid, old)
               for old in _SOURCE_INPUTS)


def _make_pair(rng, regime, role, palette, borrowed=None):
    background, source, highlight = palette
    for _ in range(90):
        height, width = _dimensions(rng, regime, role)
        areas = _profile(rng, regime, role, height, width, borrowed)
        if sum(areas) >= height * width * .47:
            continue
        if role == "diagnostic":
            shapes = [_shape(rng, area) for area in areas[2:]]
            mode = "free"
        elif role == "contrast":
            shapes = [_shape(rng, 4, "square"),
                      _shape(rng, 4, str(rng.choice(["line", "tee", "bend"]))) ]
            shapes.extend(_shape(rng, area) for area in areas[2:])
            mode = "gap"
        else:
            shapes = [_shape(rng, area) for area in areas]
            mode = str(rng.choice(["free", "gap", "diag"], p=[.33, .37, .30]))
        cells = _place(rng, height, width, shapes, mode, role == "diagnostic")
        if cells is None or len(cells) * 2 >= height * width:
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        for r, c in cells:
            grid[r, c] = source
        if _matches_source_input(grid):
            continue
        actual = sorted(len(part) for part in _components(grid, source))
        if actual != sorted(areas):
            continue
        return {"input": grid, "output": _recolor(grid, source, highlight)}
    raise RuntimeError("unable to construct valid component layout")


def _palette(rng, highlight, base, fixed, index):
    if fixed or index == 0:
        return (*base, highlight)
    available = [c for c in range(10) if c != highlight]
    for _ in range(20):
        colors = rng.choice(available, size=2, replace=False)
        candidate = (int(colors[0]), int(colors[1]))
        if index != 1 or candidate != base:
            return (*candidate, highlight)
    raise RuntimeError("unable to sample distinct palette")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _validate_seed(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = rng.choice(10, size=3, replace=False)
    base = (int(colors[0]), int(colors[1]))
    highlight = int(colors[2])
    fixed = bool(rng.random() < .48)
    diagnostic_index = int(rng.integers(2))
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=num_train + num_test, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen = set()
    profiles = []
    for index, pair_seed in enumerate(pair_seeds):
        pair_rng = np.random.default_rng(int(pair_seed))
        regime = "source_train" if index < num_train else "target_test"
        role = ("diagnostic" if index == diagnostic_index else
                "contrast" if index < 2 else "general")
        palette = _palette(pair_rng, highlight, base, fixed, index)
        borrowed = None
        if regime == "target_test" and pair_rng.random() < .28:
            borrowed = profiles[int(pair_rng.integers(len(profiles)))]
        for _ in range(8):
            pair = _make_pair(pair_rng, regime, role, palette, borrowed)
            key = _grid_key(pair["input"])
            if key not in seen:
                break
        else:
            raise RuntimeError("unable to construct a unique pair")
        seen.add(key)
        split = "train" if index < num_train else "test"
        puzzle[split].append(pair)
        if split == "train":
            profiles.append(tuple(sorted(len(p) for p in _components(pair["input"], palette[1]))))
    validate_generated_puzzle(puzzle)
    return puzzle


def _read_pair(pair):
    grid = pair["input"]
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 2
    bg_idx = int(np.argmax(counts))
    assert counts[bg_idx] > counts[1 - bg_idx]
    background = int(values[bg_idx])
    source = int(values[1 - bg_idx])
    return background, source, _components(grid, source)


def _diagnostic(components):
    small = [set(c) for c in components if len(c) == 3
             and max(r for r, _ in c) - min(r for r, _ in c) == 1
             and max(col for _, col in c) - min(col for _, col in c) == 1]
    large = [set(c) for c in components if len(c) == 4
             and _normalize(c) == ((0, 0), (0, 1), (1, 0), (1, 1))]
    return any(_diagonal_contact(a, b) for a in small for b in large)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen_inputs = set()
    seen_pairs = set()
    colors_used = set()
    highlight = None
    train_areas = set()
    area4_shapes = set()
    diagnostic = False
    changed_train = 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for arr in (grid, output):
                assert isinstance(arr, np.ndarray) and arr.ndim == 2
                assert np.issubdtype(arr.dtype, np.integer)
                assert 1 <= arr.shape[0] <= 30 and 1 <= arr.shape[1] <= 30
                assert np.all((0 <= arr) & (arr <= 9))
            assert grid.shape == output.shape
            assert 6 <= grid.shape[0] <= 16 and 6 <= grid.shape[1] <= 18
            key = _grid_key(grid)
            pair_key = (key, _grid_key(output))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert not _matches_source_input(grid)
            background, source, components = _read_pair(pair)
            colors_used.update((background, source))
            assert 2 <= len(components) <= 9
            assert sum(map(len, components)) * 2 < grid.size
            extras = set(map(int, np.unique(output))) - {background, source}
            if extras:
                assert len(extras) == 1
                this_highlight = next(iter(extras))
                if highlight is None:
                    highlight = this_highlight
                assert this_highlight == highlight
            if split == "train" and extras:
                changed_train += 1
            if split == "train":
                train_areas.update(map(len, components))
                diagnostic |= _diagnostic(components)
                area4_shapes.update(_normalize(c) for c in components if len(c) == 4)
            if highlight is not None:
                assert np.array_equal(output, _recolor(grid, source, highlight))
            else:
                assert not extras and all(len(c) <= 3 for c in components)
                assert np.array_equal(grid, output)
    assert highlight is not None and highlight not in colors_used
    assert changed_train >= 2
    assert {3, 4}.issubset(train_areas) and any(a >= 5 for a in train_areas)
    assert diagnostic
    assert len(area4_shapes) >= 2
    assert ((0, 0), (0, 1), (1, 0), (1, 1)) in area4_shapes
    # Recheck every pair after learning the highlight, including unchanged pairs.
    for pair in puzzle["train"] + puzzle["test"]:
        _, source, _ = _read_pair(pair)
        assert np.array_equal(pair["output"], _recolor(pair["input"], source, highlight))


def _fingerprint(puzzle):
    return tuple((split, tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                               for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_fingerprint(p) for p in puzzles}) >= 40
    assert _fingerprint(generate_puzzle(17)) == _fingerprint(generate_puzzle(17))
    assert _fingerprint(generate_puzzle(314159, 7, 3)) == _fingerprint(generate_puzzle(314159, 7, 3))
    for seed, tr, te in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                         (1004, None, 1), (1005, None, 4),
                         (1006, 2, None), (1007, 10, None)):
        puzzle = generate_puzzle(seed, tr, te)
        assert tr is None or len(puzzle["train"]) == tr
        assert te is None or len(puzzle["test"]) == te
        validate_generated_puzzle(puzzle)
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 3), (3, 4), (4, 4)):
        try:
            generate_puzzle(2026, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    dimensions, profiles, silhouettes, palettes = set(), set(), set(), set()
    count_values, larger, crowded, square4, diag_pairs, gap_pairs = set(), 0, 0, 0, 0, 0
    variable_palettes = 0
    single_class, test_profile_repeats = 0, 0
    compact_6x6 = compact_6x8 = compact_crowded = 0
    nine_objects = duplicate_small = duplicate_large = 0
    one_qualifying = several_qualifying = extrapolated_test = 0
    tee4 = bent4 = compact_large = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        puzzle_palettes = set()
        training_profiles = set()
        maximum_train_area = 0
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                bg, src, components = _read_pair(pair)
                assert np.array_equal(pair["output"], _recolor(grid, src,
                    next(int(x) for q in puzzle["train"] for x in np.unique(q["output"])
                         if x not in set(map(int, np.unique(q["input"]))))))
                puzzle_palettes.add((bg, src))
                dimensions.add(tuple(grid.shape))
                count_values.add(len(components))
                profile = tuple(sorted(map(len, components)))
                profiles.add(profile)
                if split == "train":
                    training_profiles.add(profile)
                    maximum_train_area = max(maximum_train_area, max(profile))
                elif profile in training_profiles:
                    test_profile_repeats += 1
                if split == "test":
                    extrapolated_test += max(profile) > maximum_train_area
                compact_6x6 += grid.shape == (6, 6)
                compact_6x8 += grid.shape == (6, 8)
                compact_crowded += (grid.shape in ((6, 6), (6, 8))
                                    and sum(profile) / grid.size > 1 / 3)
                nine_objects += len(components) == 9
                small_sizes = [size for size in profile if size < 4]
                large_sizes = [size for size in profile if size >= 4]
                duplicate_small += len(small_sizes) != len(set(small_sizes))
                duplicate_large += len(large_sizes) != len(set(large_sizes))
                one_qualifying += len(large_sizes) == 1
                several_qualifying += len(large_sizes) >= 2
                if all(len(c) <= 3 for c in components) or all(len(c) >= 4 for c in components):
                    single_class += 1
                crowded += sum(map(len, components)) / grid.size > 1 / 3
                larger += any(len(c) >= 8 for c in components)
                square4 += any(len(c) == 4 and _normalize(c) ==
                               ((0, 0), (0, 1), (1, 0), (1, 1)) for c in components)
                for component in components:
                    area = len(component)
                    shape = _normalize(component)
                    height = max(r for r, _ in shape) + 1
                    width = max(col for _, col in shape) + 1
                    if area == 4:
                        cells = set(component)
                        is_tee = any(sum((r + dr, col + dc) in cells
                                         for dr, dc in _DIRECTIONS) == 3
                                     for r, col in cells)
                        is_square = height == width == 2
                        is_line = min(height, width) == 1
                        tee4 += is_tee
                        bent4 += not (is_tee or is_square or is_line)
                    elif area >= 5:
                        compact_large += area / (height * width) >= .7
                touching = any(_diagonal_contact(set(a), set(b))
                               for i, a in enumerate(components) for b in components[i + 1:])
                diag_pairs += touching
                gap_pairs += not touching
                silhouettes.add(_grid_key((grid != bg).astype(np.int8)))
        variable_palettes += len(puzzle_palettes) > 1
        palettes.update(puzzle_palettes)
    assert len(dimensions) >= 28 and len(profiles) >= 55
    assert len(silhouettes) >= 150 and len(palettes) >= 25
    assert max(count_values) >= 9 and min(count_values) <= 2
    assert larger >= 30 and crowded >= 3 and square4 >= 35
    assert diag_pairs >= 30 and gap_pairs >= 30
    assert variable_palettes >= 10 and single_class >= 10
    assert test_profile_repeats >= 5
    assert compact_6x6 >= 3 and compact_6x8 >= 3 and compact_crowded >= 2
    assert nine_objects >= 2 and duplicate_small >= 10 and duplicate_large >= 10
    assert one_qualifying >= 20 and several_qualifying >= 20
    assert extrapolated_test >= 2
    assert tee4 >= 10 and bent4 >= 10 and compact_large >= 25
