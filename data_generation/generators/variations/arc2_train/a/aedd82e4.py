"""Generate puzzles that recolor four-neighbor singleton foreground cells."""

import numpy as np


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


_SOURCE_ROWS = (
    (("2200", "0200", "2202", "0000", "0222"),
     ("2200", "0200", "2201", "0000", "0222")),
    (("2220", "0200", "0002", "0200"),
     ("2220", "0200", "0001", "0100")),
    (("220", "202", "020"), ("220", "201", "010")),
    (("022", "022", "200"), ("022", "022", "100")),
    (("2202", "0200", "0020", "2000", "0022"),
     ("2201", "0200", "0010", "1000", "0022")),
)


def _rows_key(rows: tuple[str, ...]) -> tuple:
    return ((len(rows), len(rows[0])), tuple(int(c) for row in rows for c in row))


_SOURCE_PAIR_KEYS = frozenset(
    (_rows_key(before), _rows_key(after)) for before, after in _SOURCE_ROWS
)
_ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            seen[row, col] = True
            stack = [(row, col)]
            component = []
            while stack:
                point = stack.pop()
                component.append(point)
                for dr, dc in _ORTHOGONAL:
                    nr, nc = point[0] + dr, point[1] + dc
                    if (0 <= nr < height and 0 <= nc < width
                            and mask[nr, nc] and not seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            result.append(component)
    return result


def _isolated(mask: np.ndarray) -> np.ndarray:
    """Use immediate neighbors, independently of the component traversal."""
    neighbors = np.zeros_like(mask, dtype=bool)
    neighbors[1:] |= mask[:-1]
    neighbors[:-1] |= mask[1:]
    neighbors[:, 1:] |= mask[:, :-1]
    neighbors[:, :-1] |= mask[:, 1:]
    return mask & ~neighbors


def _apply_rule(grid: np.ndarray, foreground: int, highlight: int) -> np.ndarray:
    result = grid.copy()
    result[_isolated(grid == foreground)] = highlight
    return result


def _component_data(mask: np.ndarray) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
    parts = _components(mask)
    labels = np.full(mask.shape, -1, dtype=np.int16)
    for index, part in enumerate(parts):
        for row, col in part:
            labels[row, col] = index
    return parts, labels


def _contact_features(mask: np.ndarray) -> tuple[int, int, int]:
    """Singletons with corner contact, singleton pairs, and multiple corner neighbors."""
    parts, labels = _component_data(mask)
    singletons = [part[0] for part in parts if len(part) == 1]
    contact = pair_contact = many_corners = 0
    for row, col in singletons:
        other_labels = []
        corner_cells = 0
        for dr, dc in _DIAGONAL:
            nr, nc = row + dr, col + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                other_labels.append(int(labels[nr, nc]))
                corner_cells += 1
        if corner_cells:
            contact += 1
        if any(len(parts[index]) == 1 for index in other_labels):
            pair_contact += 1
        if sum(len(parts[index]) > 1 for index in other_labels) >= 2:
            many_corners += 1
    return contact, pair_contact, many_corners


def _cross_component_diagonal(mask: np.ndarray) -> bool:
    _, labels = _component_data(mask)
    height, width = mask.shape
    for row in range(height - 1):
        for col in range(width):
            if labels[row, col] < 0:
                continue
            for dc in (-1, 1):
                nc = col + dc
                if 0 <= nc < width and labels[row + 1, nc] >= 0:
                    if labels[row, col] != labels[row + 1, nc]:
                        return True
    return False


def _shape(rng: np.random.Generator, size: int) -> set[tuple[int, int]]:
    kind = int(rng.integers(5))
    if kind == 0:
        cells = {(0, col) for col in range(size)}  # long or short bar
    elif kind == 1 and size >= 3:
        first = int(rng.integers(2, size))
        cells = {(0, col) for col in range(first)}
        cells.update((row, 0) for row in range(size - first + 1))  # bend
    elif kind == 2 and size >= 5:
        trunk = max(3, size // 2)
        cells = {(row, 0) for row in range(trunk)}
        left_row = int(rng.integers(trunk))
        right_row = int(rng.integers(trunk))
        left_length = right_length = 0
        while len(cells) < size:
            if left_length <= right_length:
                left_length += 1
                cells.add((left_row, -left_length))
            else:
                right_length += 1
                cells.add((right_row, right_length))
    elif kind == 3 and size >= 6:
        block_height = max(2, int(np.sqrt(size)))
        block_width = (size + block_height - 1) // block_height
        cells = {(i // block_width, i % block_width) for i in range(size)}
    else:
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = sorted({
                (row + dr, col + dc)
                for row, col in cells for dr, dc in _ORTHOGONAL
                if (row + dr, col + dc) not in cells
            })
            cells.add(frontier[int(rng.integers(len(frontier)))])
    transform = int(rng.integers(8))
    mapped = []
    for row, col in cells:
        if transform & 1:
            row = -row
        if transform & 2:
            col = -col
        if transform & 4:
            row, col = col, row
        mapped.append((row, col))
    min_row = min(row for row, _ in mapped)
    min_col = min(col for _, col in mapped)
    return {(row - min_row, col - min_col) for row, col in mapped}


def _place(
    rng: np.random.Generator, occupied: set[tuple[int, int]],
    shape: set[tuple[int, int]], height: int, width: int, no_corners: bool,
) -> bool:
    shape_height = max(row for row, _ in shape) + 1
    shape_width = max(col for _, col in shape) + 1
    if shape_height > height or shape_width > width:
        return False
    positions = (height - shape_height + 1) * (width - shape_width + 1)
    directions = _ORTHOGONAL + (_DIAGONAL if no_corners else ())
    for index in rng.permutation(positions):
        top, left = divmod(int(index), width - shape_width + 1)
        placed = {(top + row, left + col) for row, col in shape}
        if placed & occupied:
            continue
        if any((row + dr, col + dc) in occupied
               for row, col in placed for dr, dc in directions):
            continue
        occupied.update(placed)
        return True
    return False


def _sparse_mask(
    rng: np.random.Generator, height: int, width: int,
    diagonal_goal: str,
) -> np.ndarray | None:
    area = height * width
    for _ in range(22):
        singleton_count = int(rng.integers(1, min(5, max(3, area // 6))))
        if diagonal_goal == "singleton_pair":
            singleton_count = max(2, singleton_count)
        multi_count = int(rng.integers(1, min(5, max(3, area // 9))))
        largest = min(24, max(4, area // 3))
        sizes = [int(rng.integers(2, largest + 1)) for _ in range(multi_count)]
        occupied: set[tuple[int, int]] = set()
        no_corners = diagonal_goal in ("none", "none_any")
        shapes = []
        for size in sorted(sizes, reverse=True):
            if diagonal_goal == "none_any":
                shape = ({(0, col) for col in range(size)}
                         if rng.random() < 0.5 else {(row, 0) for row in range(size)})
            else:
                shape = _shape(rng, size)
            shapes.append(shape)
        if not all(_place(rng, occupied, shape, height, width, no_corners)
                   for shape in shapes):
            continue
        placed_singles: set[tuple[int, int]] = set()
        success = True
        for index in range(singleton_count):
            candidates = []
            for row in range(height):
                for col in range(width):
                    if (row, col) in occupied:
                        continue
                    if any((row + dr, col + dc) in occupied for dr, dc in _ORTHOGONAL):
                        continue
                    corners = [(row + dr, col + dc) for dr, dc in _DIAGONAL
                               if (row + dr, col + dc) in occupied]
                    if diagonal_goal in ("none", "none_any") and corners:
                        continue
                    if diagonal_goal == "diagonal" and index == 0 and not corners:
                        continue
                    if diagonal_goal == "multi_corner" and index == 0:
                        if sum(point not in placed_singles for point in corners) < 2:
                            continue
                    if diagonal_goal == "singleton_pair" and index == 1:
                        if not any(point in placed_singles for point in corners):
                            continue
                    candidates.append((row, col))
            if not candidates:
                success = False
                break
            point = candidates[int(rng.integers(len(candidates)))]
            occupied.add(point)
            placed_singles.add(point)
        if not success:
            continue
        mask = np.zeros((height, width), dtype=bool)
        for row, col in occupied:
            mask[row, col] = True
        sizes_seen = sorted(len(part) for part in _components(mask))
        if sizes_seen == sorted([1] * singleton_count + sizes):
            return mask
    return None


def _dense_mask(
    rng: np.random.Generator, height: int, width: int, diagonal_goal: str,
) -> np.ndarray | None:
    for _ in range(100):
        fill = float(rng.uniform(0.54, 0.77))
        mask = rng.random((height, width)) < fill
        density = float(mask.mean())
        if not 0.5 < density <= 0.82 or np.count_nonzero(~mask) < 3:
            continue
        sizes = [len(part) for part in _components(mask)]
        if not 1 <= sizes.count(1) <= 5 or not any(size >= 2 for size in sizes):
            continue
        if sum(size >= 2 for size in sizes) > 5:
            continue
        contact, pair_contact, many_corners = _contact_features(mask)
        if diagonal_goal == "diagonal" and not contact:
            continue
        if diagonal_goal == "singleton_pair" and not pair_contact:
            continue
        if diagonal_goal == "multi_corner" and not many_corners:
            continue
        return mask
    return None


def _dimensions(rng: np.random.Generator, regime: str, dense: bool) -> tuple[int, int]:
    if dense:
        mode = int(rng.choice(3, p=(0.55, 0.31, 0.14)
                              if regime == "source_train" else (0.45, 0.35, 0.20)))
        low, high = ((3, 8), (6, 12), (11, 18))[mode]
    else:
        mode = int(rng.choice(3, p=(0.46, 0.36, 0.18)
                              if regime == "source_train" else (0.38, 0.37, 0.25)))
        low, high = ((3, 8), (6, 14), (11, 21))[mode]
    height = int(rng.integers(low, high))
    width = int(rng.integers(low, high))
    if not dense and rng.random() < 0.27:
        long_side = int(rng.integers(8, 21))
        short_side = int(rng.integers(3, 7))
        height, width = (long_side, short_side) if rng.random() < 0.5 else (short_side, long_side)
    return height, width


def _fallback_mask(dense: bool) -> np.ndarray:
    if dense:
        mask = np.zeros((4, 4), dtype=bool)
        mask[:3, :3] = True
        mask[3, 3] = True
    else:
        mask = np.zeros((5, 5), dtype=bool)
        mask[0, :2] = True
        mask[1, 1] = True
        mask[2, 2] = True
    return mask


def _generate_pair(
    rng: np.random.Generator, regime: str, colors: tuple[int, int, int],
    dense: bool, diagonal_goal: str,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(110):
        height, width = _dimensions(rng, regime, dense)
        mask = (_dense_mask(rng, height, width, diagonal_goal) if dense
                else _sparse_mask(rng, height, width, diagonal_goal))
        if mask is None:
            continue
        if not dense and mask.mean() >= 0.5:
            continue
        if dense and mask.mean() <= 0.5:
            continue
        grid = np.full(mask.shape, colors[0], dtype=np.int8)
        grid[mask] = colors[1]
        return grid, _apply_rule(grid, colors[1], colors[2])
    mask = _fallback_mask(dense)
    grid = np.full(mask.shape, colors[0], dtype=np.int8)
    grid[mask] = colors[1]
    return grid, _apply_rule(grid, colors[1], colors[2])


def _checked_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} is outside its legal range")
    return value


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        train, test = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
                       (7, 1), (8, 2), (9, 3), (10, 4))[seed % 9]
    elif train is None:
        minimum = max(2, test + 1)
        train = minimum + seed % (11 - minimum)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _infer_colors(train_pairs: list[dict]) -> tuple[int, int, int]:
    convention = None
    for pair in train_pairs:
        before, after = pair["input"], pair["output"]
        changed = before != after
        assert np.any(changed)
        old = set(int(x) for x in np.unique(before[changed]))
        new = set(int(x) for x in np.unique(after[changed]))
        input_colors = set(int(x) for x in np.unique(before))
        assert len(old) == len(new) == 1 and len(input_colors) == 2
        foreground = old.pop()
        highlight = new.pop()
        background = (input_colors - {foreground}).pop()
        triple = (background, foreground, highlight)
        assert len(set(triple)) == 3
        if convention is None:
            convention = triple
        else:
            assert triple == convention
    assert convention is not None
    return convention


def _diagonal_goal(rng: np.random.Generator, dense: bool) -> str:
    modes = ("none_any", "none", "diagonal", "singleton_pair", "multi_corner", "any")
    probabilities = ((0.12, 0.18, 0.25, 0.14, 0.11, 0.20) if not dense
                     else (0.0, 0.0, 0.42, 0.12, 0.16, 0.30))
    return str(rng.choice(modes, p=probabilities))


def generate_puzzle(
    seed: int, num_train: int | None = None, num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = tuple(int(value) for value in rng.choice(10, size=3, replace=False))

    density_flags = [bool(rng.random() < 0.23) for _ in range(train_count + test_count)]
    if any(density_flags[train_count:]) and not any(density_flags[:train_count]):
        density_flags[int(rng.integers(train_count))] = True
    witness_index = int(rng.integers(train_count))
    result = {"train": [], "test": []}
    used_inputs = set()
    for index in range(train_count + test_count):
        section = "train" if index < train_count else "test"
        regime = "source_train" if section == "train" else "target_test"
        dense = density_flags[index]
        goal = "diagonal" if index == witness_index else _diagonal_goal(rng, dense)
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        for _ in range(80):
            before, after = _generate_pair(pair_rng, regime, colors, dense, goal)
            key = _grid_key(before)
            if key in used_inputs or (key, _grid_key(after)) in _SOURCE_PAIR_KEYS:
                continue
            used_inputs.add(key)
            result[section].append({"input": before, "output": after})
            break
        else:
            raise RuntimeError("unable to construct a distinct pair")
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    pairs = puzzle["train"] + puzzle["test"]
    input_keys = set()
    pair_keys = set()
    for pair in pairs:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        before, after = pair["input"], pair["output"]
        for grid in (before, after):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.all((grid >= 0) & (grid <= 9))
        assert before.shape == after.shape
        key = _grid_key(before)
        pair_key = (key, _grid_key(after))
        assert key not in input_keys and pair_key not in pair_keys
        assert pair_key not in _SOURCE_PAIR_KEYS
        input_keys.add(key)
        pair_keys.add(pair_key)
    background, foreground, highlight = _infer_colors(puzzle["train"])
    for pair in pairs:
        before, after = pair["input"], pair["output"]
        assert set(int(x) for x in np.unique(before)) == {background, foreground}
        assert set(int(x) for x in np.unique(after)) == {background, foreground, highlight}
        foreground_mask = before == foreground
        components = _components(foreground_mask)
        assert any(len(part) == 1 for part in components)
        assert any(len(part) >= 2 for part in components)
        # Check the local rule cell by cell, using the color convention inferred
        # from training; no hidden generation metadata is consulted here.
        for row in range(before.shape[0]):
            for col in range(before.shape[1]):
                expected = int(before[row, col])
                if expected == foreground and not any(
                    0 <= row + dr < before.shape[0]
                    and 0 <= col + dc < before.shape[1]
                    and before[row + dr, col + dc] == foreground
                    for dr, dc in _ORTHOGONAL
                ):
                    expected = highlight
                assert int(after[row, col]) == expected
    assert any(_contact_features(pair["input"] == foreground)[0]
               for pair in puzzle["train"])
    if any(np.mean(pair["input"] == foreground) > 0.5 for pair in puzzle["test"]):
        assert any(np.mean(pair["input"] == foreground) > 0.5
                   for pair in puzzle["train"])


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple((section, _grid_key(pair["input"]), _grid_key(pair["output"]))
                 for section in ("train", "test") for pair in puzzle[section])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(puzzle) for puzzle in puzzles}) == 42
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    colors_seen = set()
    shapes_seen = set()
    profiles = set()
    layouts = set()
    dense_train = dense_test = no_corner_tests = no_diagonal_tests = 0
    corner_pairs = many_corners = large_components = elongated = 0
    smaller_tests = fewer_component_tests = larger_training = 0
    larger_tests = busier_tests = dense_and_diagonal = dense_large = 0
    bars = branches = thick_blocks = border_components = far_singletons = 0
    varying_area = varying_singletons = varying_multis = mixed_density = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        colors = _infer_colors(puzzle["train"])
        colors_seen.add(colors)
        train_areas = [pair["input"].size for pair in puzzle["train"]]
        train_component_counts = [len(_components(pair["input"] == colors[1]))
                                  for pair in puzzle["train"]]
        if max(train_areas) >= 70:
            larger_training += 1
        pair_areas = []
        pair_singletons = []
        pair_multis = []
        pair_densities = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                before, after = pair["input"], pair["output"]
                mask = before == colors[1]
                components = _components(mask)
                sizes = sorted(len(part) for part in components)
                assert np.array_equal(after, _apply_rule(before, colors[1], colors[2]))
                shapes_seen.add(before.shape)
                profiles.add((sizes.count(1), tuple(size for size in sizes if size > 1)))
                layouts.add((before.shape, tuple(int(x) for x in mask.flat)))
                contact, pairs, corners = _contact_features(mask)
                pair_areas.append(before.size)
                pair_singletons.append(sizes.count(1))
                pair_multis.append(len(sizes) - sizes.count(1))
                pair_densities.append(float(mask.mean()))
                corner_pairs += bool(pairs)
                many_corners += bool(corners)
                large_components += max(sizes) >= 8
                elongated += max(before.shape) >= 2 * min(before.shape)
                far_singletons += contact < sizes.count(1)
                dense_and_diagonal += mask.mean() > 0.5 and contact > 0
                dense_large += mask.mean() > 0.5 and max(before.shape) >= 12
                for component in components:
                    if len(component) < 2:
                        continue
                    cells = set(component)
                    rows = [row for row, _ in component]
                    cols = [col for _, col in component]
                    box_height = max(rows) - min(rows) + 1
                    box_width = max(cols) - min(cols) + 1
                    bars += len(cells) >= 6 and (box_height == 1 or box_width == 1)
                    thick_blocks += (len(cells) >= 6 and box_height >= 2
                                     and box_width >= 2
                                     and len(cells) * 4 >= box_height * box_width * 3)
                    branches += any(sum((row + dr, col + dc) in cells
                                        for dr, dc in _ORTHOGONAL) >= 3
                                    for row, col in cells)
                    border_components += any(
                        row in (0, before.shape[0] - 1)
                        or col in (0, before.shape[1] - 1)
                        for row, col in cells
                    )
                if section == "train":
                    dense_train += mask.mean() > 0.5
                else:
                    dense_test += mask.mean() > 0.5
                    no_corner_tests += not _cross_component_diagonal(mask)
                    no_diagonal_tests += not (
                        (mask[:-1, :-1] & mask[1:, 1:]).any()
                        or (mask[:-1, 1:] & mask[1:, :-1]).any()
                    )
                    smaller_tests += before.size <= max(train_areas)
                    fewer_component_tests += len(sizes) <= max(train_component_counts)
                    larger_tests += before.size > max(train_areas)
                    busier_tests += len(sizes) > max(train_component_counts)
        varying_area += len(set(pair_areas)) > 1
        varying_singletons += len(set(pair_singletons)) > 1
        varying_multis += len(set(pair_multis)) > 1
        mixed_density += min(pair_densities) < 0.5 < max(pair_densities)
    assert len(colors_seen) >= 25 and len(shapes_seen) >= 35
    assert len(profiles) >= 25 and len(layouts) >= 150
    assert dense_train >= 8 and dense_test >= 3
    assert no_corner_tests >= 2 and corner_pairs >= 5 and many_corners >= 2
    assert no_diagonal_tests >= 4
    assert large_components >= 20 and elongated >= 10
    assert smaller_tests >= 8 and fewer_component_tests >= 8 and larger_training >= 5
    assert larger_tests >= 2 and busier_tests >= 2
    assert dense_and_diagonal >= 8 and dense_large >= 2
    assert bars >= 5 and branches >= 15 and thick_blocks >= 5
    assert border_components >= 30 and far_singletons >= 20
    assert varying_area >= 35 and varying_singletons >= 25
    assert varying_multis >= 20 and mixed_density >= 8
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        sample = generate_puzzle(8000 + train_count * 10 + test_count,
                                 train_count, test_count)
        assert (len(sample["train"]), len(sample["test"])) == (train_count, test_count)
    for arguments in ({"num_test": 1}, {"num_test": 4},
                      {"num_train": 2}, {"num_train": 10}):
        validate_generated_puzzle(generate_puzzle(9000, **arguments))
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(123, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
    explicit = generate_puzzle(314159, 7, 3)
    assert _puzzle_key(explicit) == _puzzle_key(generate_puzzle(314159, 7, 3))
