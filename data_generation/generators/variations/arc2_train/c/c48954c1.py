import numpy as np


_ORIGINAL_INPUTS = (
    ((6, 1, 7), (1, 6, 7), (4, 7, 4)),
    ((7, 6, 7), (2, 7, 6), (1, 2, 7)),
    ((1, 9, 4), (9, 1, 6), (6, 9, 4)),
    ((8, 8, 6), (6, 3, 6), (6, 8, 8)),
)
_CATEGORIES = ("asymmetric", "horizontal", "vertical", "rotation", "both")
_TEXTURES = ("random", "sparse", "patch", "bands", "diagonal", "checker", "near", "multicolor")


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _mirror_mosaic(grid):
    horizontal = grid[:, ::-1]
    vertical = grid[::-1, :]
    rotation = grid[::-1, ::-1]
    upper = np.concatenate((rotation, vertical, rotation), axis=1)
    middle = np.concatenate((horizontal, grid, horizontal), axis=1)
    return np.concatenate((upper, middle, upper), axis=0).copy()


def _category(grid):
    h = np.array_equal(grid, grid[:, ::-1])
    v = np.array_equal(grid, grid[::-1, :])
    r = np.array_equal(grid, grid[::-1, ::-1])
    if h and v:
        return "both"
    if h:
        return "horizontal"
    if v:
        return "vertical"
    if r:
        return "rotation"
    return "asymmetric"


def _orbits(height, width, category):
    visited = set()
    result = []
    for row in range(height):
        for col in range(width):
            if (row, col) in visited:
                continue
            cells = {(row, col)}
            if category in ("horizontal", "both"):
                cells.add((row, width - 1 - col))
            if category in ("vertical", "both"):
                cells.add((height - 1 - row, col))
            if category in ("rotation", "both"):
                cells.add((height - 1 - row, width - 1 - col))
            ordered = tuple(sorted(cells))
            visited.update(ordered)
            result.append(ordered)
    return result


def _texture_labels(rng, height, width, orbits, palette_size, texture):
    count = len(orbits)
    if palette_size == 1:
        return np.zeros(count, dtype=np.int64)
    positions = [orbit[0] for orbit in orbits]
    if texture == "random":
        labels = rng.integers(palette_size, size=count)
    elif texture == "sparse":
        labels = np.zeros(count, dtype=np.int64)
        accents = min(count, max(palette_size - 1, int(np.ceil(count * rng.uniform(.12, .28)))))
        selected = rng.choice(count, size=accents, replace=False)
        for index, orbit_index in enumerate(selected):
            labels[orbit_index] = 1 + index % max(1, palette_size - 1)
    elif texture == "patch":
        r0 = int(rng.integers(height))
        r1 = int(rng.integers(r0 + 1, height + 1))
        c0 = int(rng.integers(width))
        c1 = int(rng.integers(c0 + 1, width + 1))
        labels = np.asarray([int(r0 <= r < r1 and c0 <= c < c1) for r, c in positions])
    elif texture == "bands":
        axis = int(rng.integers(2))
        thickness = int(rng.integers(1, 4))
        labels = np.asarray([(position[axis] // thickness) % palette_size for position in positions])
    elif texture == "diagonal":
        sign = 1 if int(rng.integers(2)) else -1
        offset = int(rng.integers(palette_size))
        labels = np.asarray([(r + sign * c + offset) % palette_size for r, c in positions])
    elif texture == "checker":
        labels = np.asarray([(r + c) % 2 for r, c in positions])
    elif texture == "near":
        offset = int(rng.integers(palette_size))
        labels = np.asarray([
            (min(r * width + c, height * width - 1 - (r * width + c)) + offset) % palette_size
            for r, c in positions
        ])
        chosen = int(rng.integers(count))
        labels[chosen] = (labels[chosen] + 1) % palette_size
    else:
        labels = np.arange(count, dtype=np.int64) % palette_size
        rng.shuffle(labels)

    missing = [value for value in range(palette_size) if value not in labels]
    for value in missing:
        counts = np.bincount(labels, minlength=palette_size)
        candidates = np.flatnonzero(counts[labels] > 1)
        labels[int(rng.choice(candidates))] = value
    return labels


def _sample_motif(rng, shape, category, texture, pool):
    height, width = shape
    orbits = _orbits(height, width, category)
    cap = min(10, len(pool), len(orbits))
    if cap == 1 or (category == "both" and rng.random() < .08):
        palette_size = 1
    elif texture == "multicolor" and cap >= 5:
        palette_size = int(rng.integers(5, cap + 1))
    elif rng.random() < .3:
        palette_size = 2
    else:
        palette_size = int(rng.integers(2, cap + 1))
    palette = np.asarray(rng.choice(pool, size=palette_size, replace=False), dtype=np.int8)
    for attempt in range(96):
        mode = texture if attempt < 48 else "random"
        labels = _texture_labels(rng, height, width, orbits, palette_size, mode)
        grid = np.empty(shape, dtype=np.int8)
        for orbit, label in zip(orbits, labels):
            for row, col in orbit:
                grid[row, col] = palette[label]
        if _category(grid) == category and len(np.unique(grid)) == palette_size:
            return grid
    raise RuntimeError("unable to construct requested symmetry")


def _sample_shape(rng):
    kind = int(rng.integers(3))
    first = int(rng.integers(2, 11))
    second = int(rng.integers(2, 11))
    if kind == 0:
        return (first, first)
    if first == second:
        second = 2 + (second - 1) % 9
    return (max(first, second), min(first, second)) if kind == 1 else (min(first, second), max(first, second))


def _different_shape(rng, baseline):
    for _ in range(12):
        shape = _sample_shape(rng)
        if shape != baseline:
            return shape
    return (baseline[0], 2 + (baseline[1] - 1) % 9)


def _validate_optional_count(value, name, lower, upper):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lower <= int(value) <= upper:
        raise ValueError(f"{name} must be an integer from {lower} through {upper}")


def _resolve_counts(rng, seed, num_train, num_test):
    _validate_optional_count(num_train, "num_train", 2, 10)
    _validate_optional_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        test_count = 1 + seed % 4
        train_count = test_count + 1 + (seed // 4) % (10 - test_count)
    elif num_train is None:
        test_count = int(num_test)
        train_count = int(rng.integers(test_count + 1, 11))
    elif num_test is None:
        train_count = int(num_train)
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    else:
        train_count, test_count = int(num_train), int(num_test)
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    return train_count, test_count


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(rng, int(seed), num_train, num_test)
    # Reflection placement is the only rule. A shared pool makes a coherent
    # puzzle while each motif's colors retain their literal values.
    pool = np.asarray(rng.permutation(10)[:int(rng.integers(5, 11))], dtype=np.int8)
    base_shape = _sample_shape(rng)
    mixed_sizes = rng.random() < .78
    source_style = rng.random() < .16
    streams = rng.integers(0, 2**63, size=train_count + test_count, dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    used = {_grid_key(np.asarray(rows, dtype=np.int8)) for rows in _ORIGINAL_INPUTS}
    cursor = 0
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(streams[cursor]), cursor]))
            cursor += 1
            if not mixed_sizes:
                shape = base_shape
            elif section == "train" and index == 0:
                shape = base_shape
            elif (section == "train" and index == 1) or (section == "test" and index == 0):
                shape = _different_shape(pair_rng, base_shape)
            else:
                shape = _sample_shape(pair_rng)
            if section == "train" and index == 0:
                category = "asymmetric"
            elif source_style:
                category = "asymmetric" if section == "train" else "rotation"
            else:
                weights = (.28, .18, .18, .20, .16) if section == "train" else (.32, .14, .14, .22, .18)
                category = str(pair_rng.choice(_CATEGORIES, p=weights))
            texture = str(pair_rng.choice(_TEXTURES))
            for attempt in range(128):
                motif = _sample_motif(pair_rng, shape, category, texture if attempt < 64 else "random", pool)
                key = _grid_key(motif)
                if key not in used:
                    used.add(key)
                    puzzle[section].append({"input": motif, "output": _mirror_mosaic(motif)})
                    break
            else:
                raise RuntimeError("unable to construct unique motif")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    source = {_grid_key(np.asarray(rows, dtype=np.int8)) for rows in _ORIGINAL_INPUTS}
    seen_inputs, seen_pairs = set(), set()
    demonstrations = 0
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            motif, result = pair["input"], pair["output"]
            for grid in (motif, result):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            assert 2 <= motif.shape[0] <= 10 and 2 <= motif.shape[1] <= 10
            assert result.shape == (3 * motif.shape[0], 3 * motif.shape[1])
            assert np.array_equal(result, _mirror_mosaic(motif))
            assert np.array_equal(np.unique(result), np.unique(motif))
            input_key = _grid_key(motif)
            pair_key = (input_key, _grid_key(result))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert input_key not in source
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            if section == "train" and _category(motif) == "asymmetric":
                demonstrations += 1
    assert _category(puzzle["train"][0]["input"]) == "asymmetric"
    assert demonstrations >= 1


def _reference_mosaic(grid):
    height, width = grid.shape
    output = np.empty((3 * height, 3 * width), dtype=grid.dtype)
    for row in range(output.shape[0]):
        tile_row, local_row = divmod(row, height)
        source_row = local_row if tile_row == 1 else height - 1 - local_row
        for col in range(output.shape[1]):
            tile_col, local_col = divmod(col, width)
            source_col = local_col if tile_col == 1 else width - 1 - local_col
            output[row, col] = grid[source_row, source_col]
    return output


def _puzzles_equal(left, right):
    return all(
        len(left[section]) == len(right[section]) and all(
            np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
            for a, b in zip(left[section], right[section])
        ) for section in ("train", "test")
    )


def _canonical_pattern(grid):
    labels = {}
    return (tuple(grid.shape), tuple(labels.setdefault(int(x), len(labels)) for x in grid.flat))


def self_test() -> None:
    puzzles = [generate_puzzle(2000 + index) for index in range(42)]
    assert _puzzles_equal(generate_puzzle(12345, 6, 2), generate_puzzle(12345, 6, 2))
    assert not _puzzles_equal(generate_puzzle(12345, 6, 2), generate_puzzle(12346, 6, 2))
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    all_motifs = []
    train_categories, test_categories = set(), set()
    mixed_shapes = symmetric_train = asymmetric_test = size_up = size_down = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        shapes = set()
        symmetric_train += any(_category(pair["input"]) != "asymmetric" for pair in puzzle["train"])
        asymmetric_test += any(_category(pair["input"]) == "asymmetric" for pair in puzzle["test"])
        first_area = puzzle["train"][0]["input"].size
        test_area = puzzle["test"][0]["input"].size
        size_up += test_area > first_area
        size_down += test_area < first_area
        for section, categories in (("train", train_categories), ("test", test_categories)):
            for pair in puzzle[section]:
                motif = pair["input"]
                assert np.array_equal(pair["output"], _reference_mosaic(motif))
                categories.add(_category(motif))
                shapes.add(motif.shape)
                all_motifs.append(motif)
        mixed_shapes += len(shapes) > 1
    assert train_categories == set(_CATEGORIES)
    assert test_categories == set(_CATEGORIES)
    assert mixed_shapes >= 20 and symmetric_train >= 12 and asymmetric_test >= 8
    assert size_up >= 5 and size_down >= 5
    assert len({grid.shape for grid in all_motifs}) >= 30
    assert any(grid.shape == (2, 2) for grid in all_motifs)
    assert len({_canonical_pattern(grid) for grid in all_motifs}) >= 100
    assert len({_grid_key(grid) for grid in all_motifs}) >= int(.9 * len(all_motifs))
    assert any(len(np.unique(grid)) == 2 for grid in all_motifs)
    assert any(len(np.unique(grid)) >= 6 for grid in all_motifs)
    assert any(max(np.unique(grid, return_counts=True)[1]) / grid.size >= .6 for grid in all_motifs)
    assert {("square" if h == w else "portrait" if h > w else "landscape")
            for h, w in (grid.shape for grid in all_motifs)} == {"square", "portrait", "landscape"}
    assert sum(len({pair["input"].shape for section in ("train", "test")
                    for pair in puzzle[section]}) == 1 for puzzle in puzzles) >= 3
    assert sum(len({len(np.unique(pair["input"])) for section in ("train", "test")
                    for pair in puzzle[section]}) > 1 for puzzle in puzzles) >= 30
    assert sum(
        all(_category(pair["input"]) == "asymmetric" for pair in puzzle["train"])
        and all(_category(pair["input"]) == "rotation" for pair in puzzle["test"])
        for puzzle in puzzles
    ) >= 2
    assert sum(len(np.unique(grid)) == 1 for grid in all_motifs) >= 1
    assert sum(len(np.unique(grid)) >= 6 and _category(grid) != "asymmetric"
               for grid in all_motifs) >= 10
    assert sum(grid.shape[0] != grid.shape[1] and len(np.unique(grid)) == 2
               for grid in all_motifs) >= 20
    assert sum(
        any(len(np.unique(grid[row])) == 1 for row in range(grid.shape[0]))
        or any(len(np.unique(grid[:, col])) == 1 for col in range(grid.shape[1]))
        for grid in all_motifs
    ) >= 30
    assert sum(
        any(np.all(grid[row:row + 2, col:col + 2] == grid[row, col])
            and grid[row, col] != np.unique(grid, return_counts=True)[0][
                np.argmax(np.unique(grid, return_counts=True)[1])]
            for row in range(grid.shape[0] - 1) for col in range(grid.shape[1] - 1))
        for grid in all_motifs
    ) >= 15
    assert sum(
        len(np.unique(grid)) == 2 and
        all((grid[row, col] == grid[0, 0]) == ((row + col) % 2 == 0)
            for row in range(grid.shape[0]) for col in range(grid.shape[1]))
        for grid in all_motifs
    ) >= 3
    assert sum(
        any(1 <= np.count_nonzero(grid != variant) <= 4
            for variant in (grid[:, ::-1], grid[::-1, :], grid[::-1, ::-1]))
        for grid in all_motifs
    ) >= 20
    for index, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4), (7, 3))):
        puzzle = generate_puzzle(3000 + index, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
    for index, (train_count, test_count) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(4000 + index, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(5000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid counts: {args}")
