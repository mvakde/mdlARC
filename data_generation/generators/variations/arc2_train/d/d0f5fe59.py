import numpy as np


_SOURCE_INPUTS = (
    ("000000000000", "000800000000", "088800000000", "088000080000",
     "000008888000", "000008080000", "000000000000", "000000000000",
     "000880000000", "000880000000", "000000000000", "000000000000"),
    ("0000000000", "0088000000", "0088000000", "0888000000",
     "0080000000", "0000000000", "0000088000", "0000888000",
     "0000008000", "0000000000", "0008800000", "0888000000",
     "0008000880", "0000000880", "0000000000", "0000000000"),
    ("000000000000", "000000000000", "008800000000", "008880008000",
     "000880008800", "000000008000", "000000000000", "000000000000"),
    ("000000000000", "000000000800", "000008008800", "000888008800",
     "000088000000", "000000000000", "000000000000", "000000000000",
     "000000000000", "088000008800", "008800008880", "000800000000",
     "000000000000", "000000880000", "000000000000"),
)


def _checked_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _resolve_counts(seed, rng, num_train, num_test):
    if num_train is not None:
        num_train = _checked_integer(num_train, "num_train")
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")
    if num_test is not None:
        num_test = _checked_integer(num_test, "num_test")
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = seed % 4 + 1
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _infer_roles(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 2
    index = int(np.argmax(counts))
    assert int(counts[index]) > grid.size // 2
    return int(values[index]), int(values[1 - index])


def _components(grid, foreground):
    remaining = set(map(tuple, np.argwhere(grid == foreground)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        component = {first}
        stack = [first]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        result.append(frozenset(component))
    return result


def _answer(input_grid):
    background, foreground = _infer_roles(input_grid)
    count = len(_components(input_grid, foreground))
    answer = np.full((count, count), background, dtype=np.int8)
    np.fill_diagonal(answer, foreground)
    return answer


def _normalize(cells):
    top = min(row for row, _ in cells)
    left = min(col for _, col in cells)
    return tuple(sorted((row - top, col - left) for row, col in cells))


def _transform(rng, cells):
    cells = list(cells)
    if rng.random() < 0.5:
        cells = [(row, -col) for row, col in cells]
    for _ in range(int(rng.integers(0, 4))):
        cells = [(col, -row) for row, col in cells]
    return _normalize(cells)


def _grow_to_size(rng, cells, size, preserve_hole=False):
    cells = set(cells)
    while len(cells) < size:
        frontier = set()
        for row, col in cells:
            for candidate in ((row - 1, col), (row + 1, col),
                              (row, col - 1), (row, col + 1)):
                if candidate in cells:
                    continue
                if preserve_hole and candidate == (1, 1):
                    continue
                frontier.add(candidate)
        cells.add(sorted(frontier)[int(rng.integers(len(frontier)))])
    return cells


def _shape(rng, size, style=None):
    assert 1 <= size <= 24
    if size == 1:
        return ((0, 0),)
    if size == 2:
        return _transform(rng, ((0, 0), (0, 1)))
    if style is None:
        style = ("bar", "compact", "branch", "random", "ring", "bridge")[
            int(rng.integers(0, 6))]
    if style == "bar":
        cells = {(0, col) for col in range(size)}
    elif style == "compact":
        width = max(2, int(np.ceil(np.sqrt(size))))
        cells = {(index // width, index % width) for index in range(size)}
    elif style == "branch":
        arm = max(2, size // 3)
        cells = {(0, col) for col in range(arm)}
        cells.update((row, 0) for row in range(1, arm))
        cells = _grow_to_size(rng, cells, size)
    elif style == "ring" and size >= 8:
        cells = {(row, col) for row in range(3) for col in range(3)
                 if row in (0, 2) or col in (0, 2)}
        cells = _grow_to_size(rng, cells, size, preserve_hole=True)
    elif style == "bridge" and size >= 11:
        cells = {(row, col) for row in (0, 1) for col in (0, 1, 5, 6)}
        cells.update((0, col) for col in (2, 3, 4))
        cells = _grow_to_size(rng, cells, size)
    else:
        cells = _grow_to_size(rng, {(0, 0)}, size)
    return _transform(rng, cells)


def _interlocking_shapes(rng):
    # Two combs enter each other's gaps with a full background row between teeth.
    first = {(row, 0) for row in range(7)}
    first.update((row, col) for row in (1, 5) for col in (1, 2, 3))
    second = {(row, 6) for row in range(7)}
    second.update((3, col) for col in (3, 4, 5))
    union = first | second
    reflect = rng.random() < 0.5
    turns = int(rng.integers(0, 4))
    def move(cell):
        row, col = cell
        if reflect:
            col = -col
        for _ in range(turns):
            row, col = col, -row
        return row, col
    moved_first = {move(cell) for cell in first}
    moved_second = {move(cell) for cell in second}
    moved_union = {move(cell) for cell in union}
    top = min(row for row, _ in moved_union)
    left = min(col for _, col in moved_union)
    return (
        tuple(sorted((row - top, col - left) for row, col in moved_first)),
        tuple(sorted((row - top, col - left) for row, col in moved_second)),
    )


def _allocate_sizes(rng, count, total):
    total = max(count, total)
    sizes = [1] * count
    for _ in range(total - count):
        choices = [index for index, size in enumerate(sizes) if size < 24]
        if not choices:
            break
        sizes[choices[int(rng.integers(len(choices)))]] += 1
    rng.shuffle(sizes)
    return sizes


def _sample_dimensions(rng, shapes, count, regime):
    area = sum(len(shape) for shape in shapes)
    largest_height = max(max(row for row, _ in shape) + 1 for shape in shapes)
    largest_width = max(max(col for _, col in shape) + 1 for shape in shapes)
    bucket = int(rng.integers(0, 5))
    if bucket == 0:
        side = max(4, int(np.ceil(np.sqrt(area * 2.4))))
        height = side + int(rng.integers(0, 4))
        width = side + int(rng.integers(0, 4))
    elif bucket == 1:
        height = int(rng.integers(5, 12))
        width = int(rng.integers(14, 31))
    elif bucket == 2:
        height = int(rng.integers(14, 31))
        width = int(rng.integers(5, 12))
    elif bucket == 3:
        height = int(rng.integers(10, 26))
        width = int(rng.integers(10, 26))
    else:
        height = int(rng.integers(18, 31))
        width = int(rng.integers(18, 31))
    # Feasibility depends on geometry, never on the train/test split.
    height = min(30, max(height, largest_height, 3))
    width = min(30, max(width, largest_width, 3))
    if rng.random() < 0.10:
        if rng.random() < 0.5:
            height = 30
        else:
            width = 30
    return height, width


def _expanded(cells):
    return {(row + dr, col + dc) for row, col in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _place_shapes(rng, height, width, shapes, mode, background, foreground):
    occupied = set()
    forbidden = set()
    placements = []
    remaining = list(shapes)
    if mode == "interlock":
        combined = shapes[0] + shapes[1]
        shape_height = max(row for row, _ in combined) + 1
        shape_width = max(col for _, col in combined) + 1
        if shape_height > height or shape_width > width:
            return None
        top = int(rng.integers(0, height - shape_height + 1))
        left = int(rng.integers(0, width - shape_width + 1))
        for shape in shapes[:2]:
            cells = {(top + row, left + col) for row, col in shape}
            placements.append(cells)
            occupied.update(cells)
        forbidden = _expanded(occupied)
        remaining = remaining[2:]
    else:
        remaining.sort(key=len, reverse=True)

    for index, shape in enumerate(remaining):
        shape_height = max(row for row, _ in shape) + 1
        shape_width = max(col for _, col in shape) + 1
        if shape_height > height or shape_width > width:
            return None
        candidates = []
        for top in range(height - shape_height + 1):
            for left in range(width - shape_width + 1):
                if mode == "edge" and index == 0 and not (
                    top == 0 or left == 0 or
                    top + shape_height == height or left + shape_width == width
                ):
                    continue
                cells = {(top + row, left + col) for row, col in shape}
                if cells & forbidden:
                    continue
                candidates.append(cells)
        if not candidates:
            return None
        if mode == "tight" and occupied:
            # Minimum allowed separation is one visible background corridor.
            def gap(cells):
                return min(max(abs(row - other_row), abs(col - other_col))
                           for row, col in cells for other_row, other_col in occupied)
            scores = [gap(cells) for cells in candidates]
            best = min(scores)
            shortlist = [cells for cells, score in zip(candidates, scores)
                         if score <= best + 1]
            chosen = shortlist[int(rng.integers(len(shortlist)))]
        elif mode == "open" and occupied:
            def gap(cells):
                return min(max(abs(row - other_row), abs(col - other_col))
                           for row, col in cells for other_row, other_col in occupied)
            scores = [gap(cells) for cells in candidates]
            best = max(scores)
            shortlist = [cells for cells, score in zip(candidates, scores)
                         if score >= best - 1]
            chosen = shortlist[int(rng.integers(len(shortlist)))]
        else:
            chosen = candidates[int(rng.integers(len(candidates)))]
        placements.append(chosen)
        occupied.update(chosen)
        forbidden.update(_expanded(chosen))

    if len(occupied) * 2 >= height * width:
        return None
    grid = np.full((height, width), background, dtype=np.int8)
    for row, col in occupied:
        grid[row, col] = foreground
    if len(_components(grid, foreground)) != len(shapes):
        return None
    return grid


def _generate_input(rng, count, regime, background, foreground,
                    target_area=None, mode=None):
    modes = ("open", "random", "edge", "tight", "interlock")
    for attempt in range(120):
        selected_mode = mode if attempt < 36 and mode is not None else modes[
            int(rng.integers(len(modes)))]
        if count < 2 and selected_mode == "interlock":
            selected_mode = "edge"
        if selected_mode == "interlock":
            shapes = list(_interlocking_shapes(rng))
            for _ in range(count - 2):
                size = int(rng.integers(1, 13))
                shapes.append(_shape(rng, size))
        else:
            if target_area is None:
                sizes = [int(rng.choice((1, 2, 3, 4, 5, 6, 8, 11, 15, 20),
                                        p=(.10, .12, .10, .10, .12, .12,
                                           .12, .10, .08, .04)))
                         for _ in range(count)]
            else:
                sizes = _allocate_sizes(rng, count, target_area)
            shapes = [_shape(rng, size) for size in sizes]
        height, width = _sample_dimensions(rng, shapes, count, regime)
        grid = _place_shapes(rng, height, width, shapes, selected_mode,
                             background, foreground)
        if grid is not None:
            return grid
    # A roomy deterministic construction makes every legal count feasible.
    shapes = [_shape(rng, 1 + index % 2) for index in range(count)]
    grid = np.full((30, 30), background, dtype=np.int8)
    for index, shape in enumerate(shapes):
        top = 2 + 3 * index
        for row, col in shape:
            grid[top + row, 2 + col] = foreground
    assert len(_components(grid, foreground)) == count
    return grid


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(value) for value in grid.flat))


def _is_source_input(grid):
    rows = tuple("".join(str(int(value)) for value in row) for row in grid)
    return rows in _SOURCE_INPUTS


def _pair_plan(rng, train_count, test_count, seed):
    base_area = int(rng.integers(5, 18))
    training = [
        (2, base_area),
        (4, base_area + int(rng.integers(-1, 2))),
    ]
    if train_count >= 3:
        training.append((2, base_area + int(rng.integers(11, 17))))
    if train_count >= 4:
        training.append((3, base_area + int(rng.integers(-1, 2))))
    while len(training) < train_count:
        count = int(rng.integers(1, 8))
        training.append((count, None))

    demonstrated = [count for count, _ in training]
    test_counts = []
    if test_count == 1:
        if seed % 3 == 0:
            test_counts.append(demonstrated[int(rng.integers(len(demonstrated)))])
        else:
            test_counts.append(int(rng.integers(5, 10)))
    else:
        test_counts.append(demonstrated[int(rng.integers(len(demonstrated)))])
        test_counts.append(int(rng.integers(max(demonstrated) + 1, 10)))
        while len(test_counts) < test_count:
            choices = [value for value in range(1, 10) if value not in test_counts]
            test_counts.append(choices[int(rng.integers(len(choices)))])
        rng.shuffle(test_counts)
    return training, test_counts


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _checked_integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    training_plan, test_counts = _pair_plan(rng, train_count, test_count, seed)

    variable_palette = seed % 2 == 1
    fixed_colors = tuple(int(value) for value in rng.choice(10, size=2, replace=False))
    palettes = []
    for index in range(train_count + test_count):
        if variable_palette:
            colors = tuple(int(value) for value in rng.choice(10, size=2, replace=False))
            if index == 1 and colors == palettes[0]:
                colors = (colors[1], colors[0])
        else:
            colors = fixed_colors
        palettes.append(colors)

    puzzle = {"train": [], "test": []}
    seen = set()
    plans = [("source_train", count, area) for count, area in training_plan]
    plans.extend(("target_test", count, None) for count in test_counts)
    for index, (regime, count, area) in enumerate(plans):
        background, foreground = palettes[index]
        if index == 0:
            mode = ("edge", "tight", "open")[seed % 3]
        elif index == 2 and train_count >= 3 and seed % 4 == 0:
            mode = "interlock"
        else:
            mode = None
        for _ in range(40):
            child_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            input_grid = _generate_input(child_rng, count, regime, background,
                                         foreground, area, mode)
            key = _grid_key(input_grid)
            if key in seen or _is_source_input(input_grid):
                continue
            seen.add(key)
            pair = {"input": input_grid, "output": _answer(input_grid)}
            puzzle[regime == "target_test" and "test" or "train"].append(pair)
            break
        else:
            raise RuntimeError("could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    training_counts = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            input_grid, output_grid = pair["input"], pair["output"]
            key = _grid_key(input_grid)
            pair_key = (key, _grid_key(output_grid))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert not _is_source_input(input_grid)
            background, foreground = _infer_roles(input_grid)
            components = _components(input_grid, foreground)
            assert len(components) >= 1
            # Keeping separate objects out of diagonal contact avoids teaching
            # an unshown eight-versus-four-connectivity convention.
            owners = {cell: index for index, component in enumerate(components)
                      for cell in component}
            for (row, col), owner in owners.items():
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        neighbor = (row + dr, col + dc)
                        assert neighbor not in owners or owners[neighbor] == owner
            expected = np.full((len(components), len(components)),
                               background, dtype=np.int8)
            np.fill_diagonal(expected, foreground)
            assert np.array_equal(output_grid, expected)
            assert input_grid.shape != output_grid.shape or not np.array_equal(
                input_grid, output_grid)
            if split == "train":
                training_counts.add(len(components))
    assert len(training_counts) >= 2


def _independent_sizes(grid, foreground):
    mask = grid == foreground
    visited = np.zeros(mask.shape, dtype=bool)
    sizes = []
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if not mask[row, col] or visited[row, col]:
                continue
            visited[row, col] = True
            queue = [(row, col)]
            size = 0
            while queue:
                r, c = queue.pop(0)
                size += 1
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]:
                        if mask[rr, cc] and not visited[rr, cc]:
                            visited[rr, cc] = True
                            queue.append((rr, cc))
            sizes.append(size)
    return sorted(sizes)


def _component_features(component):
    rows = [row for row, _ in component]
    cols = [col for _, col in component]
    height = max(rows) - min(rows) + 1
    width = max(cols) - min(cols) + 1
    thick = any({(row, col), (row + 1, col), (row, col + 1),
                 (row + 1, col + 1)} <= component for row, col in component)
    branch = any(sum(neighbor in component for neighbor in
                     ((row - 1, col), (row + 1, col),
                      (row, col - 1), (row, col + 1))) >= 3
                 for row, col in component)
    hole = False
    for row in range(min(rows) + 1, max(rows)):
        for col in range(min(cols) + 1, max(cols)):
            if (row, col) in component:
                continue
            seen = {(row, col)}
            stack = [(row, col)]
            enclosed = True
            while stack:
                r, c = stack.pop()
                if r in (min(rows), max(rows)) or c in (min(cols), max(cols)):
                    enclosed = False
                    break
                for neighbor in ((r - 1, c), (r + 1, c),
                                 (r, c - 1), (r, c + 1)):
                    if neighbor not in component and neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            if enclosed:
                hole = True
                break
        if hole:
            break
    articulation = False
    if len(component) >= 3:
        for removed in component:
            remaining = component - {removed}
            start = next(iter(remaining))
            reached = {start}
            stack = [start]
            while stack:
                r, c = stack.pop()
                for neighbor in ((r - 1, c), (r + 1, c),
                                 (r, c - 1), (r, c + 1)):
                    if neighbor in remaining and neighbor not in reached:
                        reached.add(neighbor)
                        stack.append(neighbor)
            if len(reached) < len(remaining):
                articulation = True
                break
    return thick, branch, hole, max(height, width) / min(height, width) >= 3, articulation


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    observed = {"train_counts": set(), "test_counts": set(),
                "component_counts": {"train": set(), "test": set()},
                "palettes": set(), "dimensions": set(), "structures": set(),
                "tiny_train": 0, "tiny_test": 0, "large": 0,
                "edge_train": 0, "edge_test": 0, "tight": 0,
                "fixed_palettes": 0, "variable_palettes": 0,
                "same_count_area": 0, "different_count_area": 0,
                "multi_test_counts": 0, "demonstrated_test": 0,
                "extrapolated_test": 0, "novel_test_palette": 0,
                "singleton_train": 0, "domino_train": 0,
                "singleton_test": 0, "domino_test": 0,
                "zero_tiny_test": 0, "one_tiny_test": 0,
                "many_tiny_test": 0,
                "small_train": 0, "small_test": 0,
                "side_30_train": 0, "side_30_test": 0,
                "narrow_train": 0, "narrow_test": 0,
                "compact_train": 0, "compact_test": 0,
                "open": 0, "interlock": 0,
                "thick": 0, "branch": 0, "hole": 0,
                "long": 0, "thin_bridge": 0,
                "tiny_and_large": 0, "edge_and_tight": 0,
                "hole_and_edge": 0, "small_and_tight": 0}
    puzzles = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_key(puzzle))
        observed["train_counts"].add(len(puzzle["train"]))
        observed["test_counts"].add(len(puzzle["test"]))
        palettes = set()
        train_palettes = set()
        train_profiles = []
        test_counts = set()
        test_palettes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                values, counts = np.unique(inp, return_counts=True)
                background = int(values[int(np.argmax(counts))])
                foreground = int(values[int(np.argmin(counts))])
                sizes = _independent_sizes(inp, foreground)
                n = len(sizes)
                expected = np.full((n, n), background, dtype=int)
                for index in range(n):
                    expected[index, index] = foreground
                assert np.array_equal(out, expected)
                assert n >= 1 and sum(sizes) < inp.size // 2 + 1
                observed["component_counts"][split].add(n)
                observed["palettes"].add((background, foreground))
                observed["dimensions"].add(tuple(inp.shape))
                observed["structures"].add((tuple(inp.shape), n, tuple(sizes)))
                palettes.add((background, foreground))
                if min(sizes) <= 2:
                    observed["tiny_" + split] += 1
                observed["singleton_" + split] += int(1 in sizes)
                observed["domino_" + split] += int(2 in sizes)
                if split == "test":
                    tiny_count = sum(size <= 2 for size in sizes)
                    observed["zero_tiny_test"] += int(tiny_count == 0)
                    observed["one_tiny_test"] += int(tiny_count == 1)
                    observed["many_tiny_test"] += int(tiny_count >= 2)
                if max(sizes) >= 10:
                    observed["large"] += 1
                observed["tiny_and_large"] += int(min(sizes) <= 2 and
                                                   max(sizes) >= 10)
                edge = bool(np.any(inp[0] == foreground) or
                            np.any(inp[-1] == foreground) or
                            np.any(inp[:, 0] == foreground) or
                            np.any(inp[:, -1] == foreground))
                if edge:
                    observed["edge_" + split] += 1
                observed["small_" + split] += int(min(inp.shape) <= 7)
                observed["side_30_" + split] += int(max(inp.shape) == 30)
                ratio = max(inp.shape) / min(inp.shape)
                observed["narrow_" + split] += int(ratio >= 2)
                observed["compact_" + split] += int(ratio <= 1.25)
                comps = _components(inp, foreground)
                for comp in comps:
                    thick, branch, hole, long, articulation = _component_features(comp)
                    observed["thick"] += int(thick)
                    observed["branch"] += int(branch)
                    observed["hole"] += int(hole)
                    observed["hole_and_edge"] += int(hole and edge)
                    observed["long"] += int(long)
                    observed["thin_bridge"] += int(thick and articulation and
                                                   len(comp) >= 10)
                if len(comps) >= 2:
                    gaps = [
                        min(max(abs(r1 - r2), abs(c1 - c2))
                            for r1, c1 in first for r2, c2 in second)
                        for index, first in enumerate(comps)
                        for second in comps[index + 1:]
                    ]
                    observed["tight"] += int(min(gaps) == 2)
                    observed["open"] += int(min(gaps) >= 5)
                    observed["edge_and_tight"] += int(min(gaps) == 2 and edge)
                    observed["small_and_tight"] += int(min(gaps) == 2 and
                                                       min(inp.shape) <= 7)
                    for index, first in enumerate(comps):
                        for second in comps[index + 1:]:
                            a_rows = [r for r, _ in first]
                            a_cols = [c for _, c in first]
                            b_rows = [r for r, _ in second]
                            b_cols = [c for _, c in second]
                            overlapping_boxes = (
                                max(min(a_rows), min(b_rows)) <=
                                min(max(a_rows), max(b_rows)) and
                                max(min(a_cols), min(b_cols)) <=
                                min(max(a_cols), max(b_cols)))
                            if (overlapping_boxes and len(first) >= 8 and
                                    len(second) >= 8 and min(
                                        max(abs(r1 - r2), abs(c1 - c2))
                                        for r1, c1 in first
                                        for r2, c2 in second) == 2):
                                observed["interlock"] += 1
                if split == "train":
                    train_profiles.append((n, sum(sizes)))
                    train_palettes.add((background, foreground))
                else:
                    test_counts.add(n)
                    test_palettes.add((background, foreground))
        if len(palettes) == 1:
            observed["fixed_palettes"] += 1
        else:
            observed["variable_palettes"] += 1
            assert len({_infer_roles(p["input"]) for p in puzzle["train"]}) >= 2
        if len(test_counts) > 1:
            observed["multi_test_counts"] += 1
        observed["demonstrated_test"] += int(any(
            count in {profile[0] for profile in train_profiles}
            for count in test_counts))
        observed["extrapolated_test"] += int(any(
            count > max(profile[0] for profile in train_profiles)
            for count in test_counts))
        observed["novel_test_palette"] += int(bool(test_palettes - train_palettes))
        if any(a == b and abs(x - y) >= 6 for a, x in train_profiles
               for b, y in train_profiles):
            observed["same_count_area"] += 1
        if any(a != b and abs(x - y) <= 2 for a, x in train_profiles
               for b, y in train_profiles):
            observed["different_count_area"] += 1
    assert len(puzzles) == 42
    assert len(observed["train_counts"]) >= 4 and len(observed["test_counts"]) >= 3
    assert {1, 2, 4}.issubset(observed["component_counts"]["train"])
    assert {1, 2, 5, 6, 7, 8, 9}.issubset(observed["component_counts"]["test"])
    assert len(observed["palettes"]) >= 20
    assert len(observed["dimensions"]) >= 70
    assert len(observed["structures"]) >= 150
    for name in ("tiny_train", "tiny_test", "large", "edge_train", "edge_test",
                 "fixed_palettes", "variable_palettes", "same_count_area",
                 "different_count_area", "multi_test_counts",
                 "demonstrated_test", "extrapolated_test",
                 "novel_test_palette", "zero_tiny_test", "one_tiny_test",
                 "many_tiny_test", "singleton_train", "domino_train",
                 "singleton_test", "domino_test",
                 "small_train", "small_test", "side_30_train", "side_30_test",
                 "narrow_train", "narrow_test", "compact_train",
                 "compact_test", "tight", "open", "interlock", "thick",
                 "branch", "hole", "long", "thin_bridge", "tiny_and_large",
                 "edge_and_tight", "hole_and_edge", "small_and_tight"):
        assert observed[name] > 0, name
    for seed, train_count, test_count in ((101, 2, 1), (102, 10, 1),
                                          (103, 10, 4), (104, 5, 2)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for seed, train_count, test_count in ((201, None, 1), (202, None, 4),
                                          (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (3, 4),
                                    (2.5, 1), (3, True)):
        try:
            generate_puzzle(300, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
    assert _puzzle_key(generate_puzzle(77123, 8, 3)) == _puzzle_key(
        generate_puzzle(77123, 8, 3))
    assert _puzzle_key(generate_puzzle(77123, 8, 3)) != _puzzle_key(
        generate_puzzle(77124, 8, 3))
