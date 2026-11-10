import numpy as np


# Sparse descriptions of all original source-task pairs. Each grid
# specification is: (height, width, background, foreground_cells), where
# foreground_cells contains (row, column, color) triples.
_SOURCE_PAIR_SPECS = (
    (
        (10, 10, 8, ((4, 4, 7), (6, 8, 7))),
        (10, 10, 8, ((2, 4, 7), (4, 8, 7))),
    ),
    (
        (10, 10, 8, ((5, 7, 2), (9, 7, 2))),
        (10, 10, 8, ((5, 5, 2), (9, 5, 2))),
    ),
    (
        (10, 10, 8, ((4, 2, 9),)),
        (10, 10, 8, ((6, 2, 9),)),
    ),
    (
        (10, 10, 8, ((5, 7, 1), (9, 7, 1))),
        (10, 10, 8, ((5, 8, 1), (9, 8, 1))),
    ),
    (
        (
            16,
            16,
            8,
            (
                (1, 7, 9),
                (2, 2, 1),
                (3, 13, 2),
                (4, 8, 7),
                (7, 1, 9),
                (8, 8, 2),
                (10, 10, 1),
                (11, 2, 2),
                (12, 11, 9),
                (13, 5, 7),
            ),
        ),
        (
            16,
            16,
            8,
            (
                (2, 3, 1),
                (2, 8, 7),
                (3, 7, 9),
                (3, 11, 2),
                (8, 6, 2),
                (9, 1, 9),
                (10, 11, 1),
                (11, 0, 2),
                (11, 5, 7),
                (14, 11, 9),
            ),
        ),
    ),
)


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a non-negative integer")
    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed_value


def _normalize_count(
    value: int | None,
    name: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _resolve_counts(
    seed_value: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _normalize_count(num_train, "num_train", 2, 10)
    test_count = _normalize_count(num_test, "num_test", 1, 4)

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        # This schedule guarantees broad count diversity for consecutive seeds.
        test_count = 1 + seed_value % 4
        lowest_train = max(2, test_count + 1)
        number_of_choices = 11 - lowest_train
        train_count = (
            lowest_train
            + (seed_value // 4) % number_of_choices
        )
        return train_count, test_count

    if train_count is None:
        assert test_count is not None
        lowest_train = max(2, test_count + 1)
        number_of_choices = 11 - lowest_train
        train_count = (
            lowest_train + seed_value % number_of_choices
        )
        return train_count, test_count

    assert test_count is None
    largest_test = min(4, train_count - 1)
    if largest_test < 1:
        raise ValueError("no legal test count exists")
    test_count = 1 + seed_value % largest_test
    return train_count, test_count


def _sample_puzzle_parameters(rng: np.random.Generator, seed: int,
                              train_count: int) -> dict:
    role_count = int(rng.integers(1, 7))
    if seed % 9 == 0:
        role_count = min(role_count, 4, train_count)
    colors = [int(x) for x in rng.choice(10, size=role_count, replace=False)]
    vectors: list[tuple[int, int]] = []
    if seed % 9 == 0:
        original = [(-2, 0), (0, -2), (2, 0), (0, 1)]
        vectors = [original[int(i)] for i in rng.choice(4, size=min(role_count, 4), replace=False)]
    else:
        for _ in colors:
            distance = int(rng.integers(1, 5))
            if rng.random() < 0.55:
                axis = int(rng.integers(0, 2))
                sign = (-1, 1)[int(rng.integers(0, 2))]
                vector = (sign * distance, 0) if axis == 0 else (0, sign * distance)
            else:
                vector = (
                    (-1, 1)[int(rng.integers(0, 2))] * distance,
                    (-1, 1)[int(rng.integers(0, 2))] * int(rng.integers(1, 5)),
                )
            vectors.append(vector)
        if role_count >= 2 and seed % 5 == 2:
            first = vectors[0]
            if max(map(abs, first)) == 1:
                first = (2 if first[0] > 0 else -2 if first[0] < 0 else 0,
                         2 if first[1] > 0 else -2 if first[1] < 0 else 0)
                vectors[0] = first
            vectors[1] = (-first[0], -first[1])
        elif role_count >= 2 and seed % 7 == 3:
            vectors[1] = vectors[0]
    mapping = dict(zip(colors, vectors))
    backgrounds = [color for color in range(10) if color not in mapping]
    return {"mapping": mapping, "backgrounds": backgrounds, "shift": seed % 4}


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _pair_key(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return (_grid_key(input_grid), _grid_key(output_grid))


def _matches_sparse_spec(grid: np.ndarray, specification: tuple) -> bool:
    height, width, background, cells = specification
    if grid.shape != (height, width):
        return False
    if int(np.count_nonzero(grid != background)) != len(cells):
        return False
    return all(int(grid[row, column]) == color for row, column, color in cells)


def _is_original_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    return any(_matches_sparse_spec(input_grid, inp) and _matches_sparse_spec(output_grid, out)
               for inp, out in _SOURCE_PAIR_SPECS)


def _most_common_color(grid: np.ndarray) -> int:
    return int(np.argmax(np.bincount(grid.ravel().astype(np.int64), minlength=10)))


def _matching_vectors(source_coordinates: np.ndarray,
                      destination_coordinates: np.ndarray) -> list[tuple[int, int]]:
    if len(source_coordinates) == 0 or len(source_coordinates) != len(destination_coordinates):
        return []
    source = {(int(r), int(c)) for r, c in source_coordinates}
    destination = {(int(r), int(c)) for r, c in destination_coordinates}
    dr = min(r for r, _ in destination) - min(r for r, _ in source)
    dc = min(c for _, c in destination) - min(c for _, c in source)
    if (dr, dc) == (0, 0) or {(r + dr, c + dc) for r, c in source} != destination:
        return []
    return [(dr, dc)]


def _infer_training_conventions(puzzle: dict) -> dict[int, tuple[int, int]]:
    mapping: dict[int, tuple[int, int]] = {}
    backgrounds = set()
    for pair in puzzle["train"]:
        inp, out = pair["input"], pair["output"]
        background = _most_common_color(inp)
        backgrounds.add(background)
        assert _most_common_color(out) == background
        colors = {int(c) for c in np.unique(inp)} - {background}
        assert colors
        assert {int(c) for c in np.unique(out)} - {background} == colors
        for color in colors:
            matches = _matching_vectors(np.argwhere(inp == color), np.argwhere(out == color))
            assert len(matches) == 1
            vector = matches[0]
            if color in mapping:
                assert mapping[color] == vector
            mapping[color] = vector
    assert not backgrounds.intersection(mapping)
    return mapping


def _apply_translation(input_grid: np.ndarray, background: int,
                       mapping: dict[int, tuple[int, int]]) -> np.ndarray:
    output = np.full_like(input_grid, background)
    occupied = set()
    for rv, cv in np.argwhere(input_grid != background):
        row, column = int(rv), int(cv)
        color = int(input_grid[row, column])
        assert color in mapping
        dr, dc = mapping[color]
        dest = (row + dr, column + dc)
        assert 0 <= dest[0] < input_grid.shape[0]
        assert 0 <= dest[1] < input_grid.shape[1]
        assert dest not in occupied
        occupied.add(dest)
        output[dest] = color
    return output


def _draw_dimension(rng: np.random.Generator, low: int, high: int) -> int:
    if high == 30 and rng.random() < 0.14:
        return 30
    return int(rng.integers(low, high + 1))


def _sample_shape(rng: np.random.Generator, shift: int, is_test: bool,
                  role_count: int, train_shapes: list[tuple[int, int]]) -> tuple[int, int]:
    minimum = 8 + 2 * max(0, role_count - 4)
    if is_test and shift == 1:
        return train_shapes[int(rng.integers(0, len(train_shapes)))]
    if shift == 0:
        low, high = (17, 30) if is_test else (minimum, 16)
    elif shift == 3:
        low, high = (minimum, 16) if is_test else (17, 30)
    else:
        low, high = minimum, 30
    low = min(low, high)
    return (_draw_dimension(rng, low, high), _draw_dimension(rng, low, high))


def _plan_colors(rng: np.random.Generator, colors: list[int],
                 train_count: int, force_isolated: bool) -> list[list[int]]:
    if len(colors) <= train_count and (force_isolated or rng.random() < 0.28):
        plan = [[color] for color in rng.permutation(colors).tolist()]
        while len(plan) < train_count:
            size = 1 if force_isolated else int(rng.integers(1, min(4, len(colors)) + 1))
            plan.append([int(c) for c in rng.choice(colors, size=size, replace=False)])
    else:
        plan = []
        for _ in range(train_count):
            size = int(rng.integers(1, len(colors) + 1))
            plan.append([int(c) for c in rng.choice(colors, size=size, replace=False)])
        missing = [c for c in colors if not any(c in subset for subset in plan)]
        for index, color in enumerate(missing):
            plan[index % train_count].append(color)
    return [plan[int(i)] for i in rng.permutation(len(plan))]


def _token_colors(rng: np.random.Generator, subset: list[int],
                  shape: tuple[int, int], is_test: bool) -> list[int]:
    result = list(subset)
    area = shape[0] * shape[1]
    cap = min(18 if is_test else 14, max(len(subset), area // 9))
    if is_test and rng.random() < 0.35:
        extra = int(rng.integers(0, min(3, cap - len(result)) + 1))
    else:
        extra = int(rng.integers(0, cap - len(result) + 1))
    for _ in range(extra):
        result.append(subset[int(rng.integers(0, len(subset)))])
    return result


def _sample_positions(rng: np.random.Generator, shape: tuple[int, int],
                      token_colors: list[int], mapping: dict[int, tuple[int, int]],
                      encourage_vacated: bool) -> list[tuple[int, int, int]] | None:
    height, width = shape
    candidates = {}
    for color in set(token_colors):
        dr, dc = mapping[color]
        candidates[color] = [(r, c) for r in range(height) for c in range(width)
                             if 0 <= r + dr < height and 0 <= c + dc < width]
    for attempt in range(22):
        placed: list[tuple[int, int, int]] = []
        sources = set()
        destinations = set()
        remaining = list(token_colors)
        if encourage_vacated and len(remaining) >= 2:
            chain_color = next((color for color in remaining
                                if remaining.count(color) >= 2 and
                                max(map(abs, mapping[color])) >= 2), None)
            opposite = next(((a, b) for a in set(remaining) for b in set(remaining)
                             if a != b and mapping[a] == tuple(-v for v in mapping[b])
                             and max(map(abs, mapping[a])) >= 2), None)
            if opposite is not None or chain_color is not None:
                a, b = opposite if opposite is not None else (chain_color, chain_color)
                dr, dc = mapping[a]
                starts = [(r, c) for r, c in candidates[a]
                          if (r + dr, c + dc) in candidates[b]]
                if starts:
                    r, c = starts[int(rng.integers(0, len(starts)))]
                    r2, c2 = r + dr, c + dc
                    placed = [(r, c, a), (r2, c2, b)]
                    sources = {(r, c), (r2, c2)}
                    destinations = {(r2, c2), (r2 + mapping[b][0], c2 + mapping[b][1])}
                    remaining.remove(a)
                    remaining.remove(b)
        order = [remaining[int(i)] for i in rng.permutation(len(remaining))]
        succeeded = True
        for color in order:
            dr, dc = mapping[color]
            choices = candidates[color]
            permutation = rng.permutation(len(choices))
            picked = None
            for index in permutation:
                r, c = choices[int(index)]
                dst = (r + dr, c + dc)
                if (r, c) in sources or dst in destinations:
                    continue
                if any(max(abs(r - sr), abs(c - sc)) <= 1 for sr, sc in sources):
                    continue
                picked = (r, c, dst)
                break
            if picked is None:
                succeeded = False
                break
            r, c, dst = picked
            sources.add((r, c))
            destinations.add(dst)
            placed.append((r, c, color))
        if succeeded:
            return placed
    return None


def _build_pair(rng: np.random.Generator, shape: tuple[int, int], background: int,
                mapping: dict[int, tuple[int, int]], token_colors: list[int],
                encourage_vacated: bool) -> tuple[np.ndarray, np.ndarray] | None:
    positions = _sample_positions(rng, shape, token_colors, mapping, encourage_vacated)
    if positions is None:
        return None
    inp = np.full(shape, background, dtype=np.int8)
    for row, column, color in positions:
        inp[row, column] = color
    return inp, _apply_translation(inp, background, mapping)


def _fallback_pair(shape: tuple[int, int], background: int,
                   mapping: dict[int, tuple[int, int]], token_colors: list[int],
                   used_inputs: set, start: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    for offset in range(64):
        sources = set()
        destinations = set()
        placements = []
        for color in token_colors:
            dr, dc = mapping[color]
            for step in range(height * width):
                linear = (start + 17 * offset + step) % (height * width)
                row, column = divmod(linear, width)
                destination = (row + dr, column + dc)
                if not 0 <= destination[0] < height or not 0 <= destination[1] < width:
                    continue
                if destination in destinations:
                    continue
                if any(max(abs(row - r), abs(column - c)) <= 1 for r, c in sources):
                    continue
                placements.append((row, column, color))
                sources.add((row, column))
                destinations.add(destination)
                break
            else:
                break
        else:
            inp = np.full(shape, background, dtype=np.int8)
            for row, column, color in placements:
                inp[row, column] = color
            if _grid_key(inp) not in used_inputs:
                return inp, _apply_translation(inp, background, mapping)
    raise ValueError("unable to construct deterministic fallback pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed_value = _normalize_seed(seed)
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    params = _sample_puzzle_parameters(rng, seed_value, train_count)
    mapping = params["mapping"]
    colors = list(mapping)
    shift = params["shift"]
    backgrounds = params["backgrounds"]
    source_like = seed_value % 9 == 0 and len(colors) > 1
    train_plan = _plan_colors(rng, colors, train_count, source_like)
    train_pairs = []
    test_pairs = []
    train_shapes: list[tuple[int, int]] = []
    used_inputs = set()
    for section, count in (("train", train_count), ("test", test_count)):
        for pair_index in range(count):
            for attempt in range(64):
                if section == "train":
                    subset = train_plan[pair_index]
                else:
                    if source_like and pair_index == 0:
                        subset = colors[:]
                    elif len(colors) == 1 or rng.random() < 0.27:
                        subset = [colors[int(rng.integers(0, len(colors)))]]
                    elif rng.random() < 0.43:
                        subset = colors[:]
                    else:
                        size = int(rng.integers(1, len(colors) + 1))
                        subset = [int(c) for c in rng.choice(colors, size=size, replace=False)]
                shape = _sample_shape(rng, shift, section == "test", len(colors), train_shapes)
                if section == "train" and pair_index == 1 and shape == train_shapes[0]:
                    continue
                background = backgrounds[int(rng.integers(0, len(backgrounds)))]
                if section == "train" and pair_index == 1 and background == _most_common_color(train_pairs[0]["input"]):
                    background = backgrounds[(backgrounds.index(background) + 1) % len(backgrounds)]
                token_colors = _token_colors(rng, subset, shape, section == "test")
                if section == "train" and source_like:
                    token_colors = subset * int(rng.integers(1, 3))
                if section == "test" and source_like and pair_index == 0:
                    token_colors = subset * 2
                encourage = rng.random() < 0.27
                pair = _build_pair(rng, shape, background, mapping, token_colors, encourage)
                if pair is None:
                    continue
                inp, out = pair
                key = _grid_key(inp)
                if key in used_inputs or _is_original_source_pair(inp, out):
                    continue
                used_inputs.add(key)
                item = {"input": inp, "output": out}
                if section == "train":
                    train_pairs.append(item)
                    train_shapes.append(shape)
                else:
                    test_pairs.append(item)
                break
            else:
                shape = (30, 30)
                if section == "train" and pair_index == 1 and shape == train_shapes[0]:
                    shape = (29, 30)
                background = backgrounds[pair_index % len(backgrounds)]
                if section == "train" and pair_index == 1 and background == _most_common_color(train_pairs[0]["input"]):
                    background = backgrounds[(backgrounds.index(background) + 1) % len(backgrounds)]
                fallback_colors = colors if section == "test" else train_plan[pair_index]
                if section == "test" and source_like and pair_index == 0:
                    fallback_colors = fallback_colors * 2
                inp, out = _fallback_pair(shape, background, mapping, fallback_colors,
                                          used_inputs, train_count + pair_index)
                used_inputs.add(_grid_key(inp))
                item = {"input": inp, "output": out}
                if section == "train":
                    train_pairs.append(item)
                    train_shapes.append(shape)
                else:
                    test_pairs.append(item)
    puzzle = {"train": train_pairs, "test": test_pairs}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    all_pairs = puzzle["train"] + puzzle["test"]
    inputs = []
    pair_keys = []
    for pair in all_pairs:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = pair["input"], pair["output"]
        for grid in (inp, out):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert all(1 <= size <= 30 for size in grid.shape)
            assert np.all((0 <= grid) & (grid <= 9))
        assert inp.shape == out.shape and not np.array_equal(inp, out)
        assert not _is_original_source_pair(inp, out)
        background = _most_common_color(inp)
        assert _most_common_color(out) == background
        assert int(np.count_nonzero(inp == background)) > inp.size // 2
        assert int(np.count_nonzero(out == background)) > out.size // 2
        source = [tuple(map(int, pos)) for pos in np.argwhere(inp != background)]
        assert source
        assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1
                   for i, a in enumerate(source) for b in source[i + 1:])
        inputs.append(_grid_key(inp))
        pair_keys.append(_pair_key(inp, out))
    assert len(set(inputs)) == len(inputs)
    assert len(set(pair_keys)) == len(pair_keys)
    mapping = _infer_training_conventions(puzzle)
    assert 1 <= len(mapping) <= 6
    assert len({pair["input"].shape for pair in puzzle["train"]}) >= 2
    assert len({_most_common_color(pair["input"]) for pair in puzzle["train"]}) >= 2
    for pair in all_pairs:
        inp, out = pair["input"], pair["output"]
        background = _most_common_color(inp)
        assert background not in mapping
        colors = {int(c) for c in np.unique(inp)} - {background}
        assert colors and colors.issubset(mapping)
        assert {int(c) for c in np.unique(out)} - {background} == colors
        for color in colors:
            assert np.count_nonzero(inp == color) == np.count_nonzero(out == color)
        assert np.array_equal(_apply_translation(inp, background, mapping), out)


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple((section, tuple(_pair_key(p["input"], p["output"]) for p in puzzle[section]))
                 for section in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    maps = [_infer_training_conventions(p) for p in puzzles]
    assert len({tuple(sorted(m.items())) for m in maps}) >= 30
    assert {len(m) for m in maps} == set(range(1, 7))
    vectors = {v for m in maps for v in m.values()}
    assert len(vectors) >= 15
    assert any(dr and dc for dr, dc in vectors)
    assert any(len(set(m.values())) < len(m) for m in maps)
    train_shapes = {pair["input"].shape for p in puzzles for pair in p["train"]}
    test_shapes = {pair["input"].shape for p in puzzles for pair in p["test"]}
    assert len(train_shapes) >= 30 and len(test_shapes) >= 25
    assert any(30 in shape for shape in train_shapes | test_shapes)
    assert any(len({pair["input"].shape for pair in p["train"]}) > 1 for p in puzzles)
    assert any(len({_most_common_color(pair["input"]) for pair in p["train"]}) > 1 for p in puzzles)
    assert any(len(set(np.unique(pair["input"])) - {_most_common_color(pair["input"])}) > 1
               for p in puzzles for pair in p["train"])
    assert any(len(set(np.unique(pair["input"])) - {_most_common_color(pair["input"])}) == 1
               for p in puzzles for pair in p["test"])
    populations = {int(np.count_nonzero(pair["input"] != _most_common_color(pair["input"])))
                   for p in puzzles for pair in p["train"]}
    assert 1 in populations and max(populations) >= 7
    vacated = False
    adjacent_output = False
    for p in puzzles:
        for pair in p["train"] + p["test"]:
            inp, out = pair["input"], pair["output"]
            bg = _most_common_color(inp)
            src = {tuple(x) for x in np.argwhere(inp != bg)}
            dst = {tuple(x) for x in np.argwhere(out != bg)}
            vacated |= bool(src & dst)
            adjacent_output |= any(max(abs(a[0]-b[0]), abs(a[1]-b[1])) == 1
                                   for a in dst for b in dst if a != b)
    assert vacated and adjacent_output
    coverage = {name: 0 for name in (
        "source_like", "diagonal", "cardinal", "shared_vector", "same_direction",
        "more_roles", "same_size", "smaller_test", "larger_test",
        "small", "size_30", "sparse_large", "mixed_train", "mixed_changed_bg",
        "omitted_test_color", "single_color_test", "test_singleton",
        "train_repeat_four", "swap", "chain", "boundary",
    )}
    for puzzle, mapping in zip(puzzles, maps):
        vectors = list(mapping.values())
        coverage["diagonal"] += any(dr and dc for dr, dc in vectors)
        coverage["cardinal"] += all(dr == 0 or dc == 0 for dr, dc in vectors)
        coverage["shared_vector"] += len(set(vectors)) < len(vectors)
        coverage["more_roles"] += len(mapping) > len(puzzle["train"])
        coverage["same_direction"] += any(
            first != second and
            (np.sign(first[0]), np.sign(first[1])) ==
            (np.sign(second[0]), np.sign(second[1]))
            for first in vectors for second in vectors
        )
        train_shapes = {pair["input"].shape for pair in puzzle["train"]}
        first_background = _most_common_color(puzzle["train"][0]["input"])
        coverage["source_like"] += (
            len(mapping) > 1 and
            all(len(np.unique(pair["input"])) == 2 for pair in puzzle["train"]) and
            any(len(np.unique(pair["input"])) == len(mapping) + 1
                for pair in puzzle["test"])
        )
        for pair in puzzle["test"]:
            shape = pair["input"].shape
            coverage["same_size"] += shape in train_shapes
            coverage["smaller_test"] += any(
                shape[0] < train_shape[0] and shape[1] < train_shape[1]
                for train_shape in train_shapes
            )
            coverage["larger_test"] += all(
                shape[0] > train_shape[0] and shape[1] > train_shape[1]
                for train_shape in train_shapes
            )
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                background = _most_common_color(inp)
                colors = {int(c) for c in np.unique(inp)} - {background}
                population = int(np.count_nonzero(inp != background))
                coverage["small"] += max(inp.shape) <= 10
                coverage["size_30"] += 30 in inp.shape
                coverage["sparse_large"] += max(inp.shape) >= 20 and population <= 3
                coverage["boundary"] += any(
                    int(r) in (0, inp.shape[0] - 1) or
                    int(c) in (0, inp.shape[1] - 1)
                    for r, c in np.argwhere(out != background)
                )
                if section == "train":
                    coverage["mixed_train"] += len(colors) > 1
                    coverage["mixed_changed_bg"] += len(colors) > 1 and background != first_background
                    coverage["train_repeat_four"] += any(np.count_nonzero(inp == color) >= 4 for color in colors)
                else:
                    coverage["omitted_test_color"] += colors != set(mapping)
                    coverage["single_color_test"] += len(colors) == 1
                    coverage["test_singleton"] += any(np.count_nonzero(inp == color) == 1 for color in colors)
                source = {tuple(map(int, p)) for p in np.argwhere(inp != background)}
                edges = {}
                predicted = np.full_like(inp, background)
                for row, column in source:
                    dr, dc = mapping[int(inp[row, column])]
                    dest = (row + dr, column + dc)
                    assert predicted[dest] == background
                    predicted[dest] = inp[row, column]
                    edges[(row, column)] = dest
                assert np.array_equal(predicted, out)
                coverage["swap"] += any(
                    a != b and edges[a] == b and edges[b] == a
                    for a in edges for b in edges
                )
                coverage["chain"] += any(
                    a != b and edges[a] == b and edges[b] != a
                    for a in edges for b in edges
                )
    assert all(value > 0 for value in coverage.values()), coverage
    assert coverage["source_like"] >= 3
    assert coverage["mixed_train"] >= 30
    assert coverage["single_color_test"] >= 15
    assert coverage["size_30"] >= 10
    assert coverage["swap"] >= 1
    assert coverage["chain"] >= 10
    for seed, nt, ns in ((100, 2, 1), (101, 10, 1), (102, 10, 4),
                         (200, None, 1), (201, None, 4),
                         (202, 2, None), (203, 10, None)):
        p = generate_puzzle(seed, nt, ns)
        if nt is not None:
            assert len(p["train"]) == nt
        if ns is not None:
            assert len(p["test"]) == ns
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": 2.5, "num_test": 1},
                   {"num_train": 3, "num_test": True}):
        try:
            generate_puzzle(999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    assert _puzzle_key(generate_puzzle(314159, 7, 3)) == _puzzle_key(generate_puzzle(314159, 7, 3))
    assert _puzzle_key(generate_puzzle(700, 6, 2)) != _puzzle_key(generate_puzzle(701, 6, 2))
