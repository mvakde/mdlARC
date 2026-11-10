import numpy as np


# The six source inputs are excluded by content; the atlas makes their outputs
# deterministic, so this also excludes exact source pairs.
_SOURCE_INPUTS = {
    (4, 9, 1, 8, 8, 4, 1, 8, 4, 8, 8, 1, 1, 1, 1, 8),
    (6, 2, 6, 2, 6, 6, 5, 5, 1, 1, 1, 2, 5, 1, 2, 1),
    (6, 7, 7, 6, 7, 1, 6, 6, 9, 1, 6, 6, 9, 1, 6, 1),
    (1, 1, 2, 1, 6, 6, 7, 6, 7, 6, 2, 1, 1, 6, 2, 6),
    (4, 1, 9, 1, 1, 9, 1, 4, 9, 1, 4, 6, 4, 1, 6, 6),
    (4, 6, 4, 4, 4, 6, 4, 4, 7, 6, 7, 9, 9, 4, 9, 7),
}


def _key(grid: np.ndarray) -> tuple:
    return tuple(grid.shape), tuple(int(x) for x in grid.flat)


def _source_copy(grid: np.ndarray) -> bool:
    return grid.shape == (4, 4) and tuple(int(x) for x in grid.flat) in _SOURCE_INPUTS


def _atlas(tile: np.ndarray) -> np.ndarray:
    return np.block([
        [tile, np.rot90(tile, 1)],
        [np.rot90(tile, 2), np.rot90(tile, 3)],
    ]).copy()


def _distinct_rows(grid: np.ndarray) -> bool:
    return len({tuple(int(x) for x in row) for row in grid}) == len(grid)


def _asymmetric(grid: np.ndarray) -> bool:
    rotations = [_key(np.rot90(grid, turns)) for turns in range(4)]
    return len(set(rotations)) == 4 and _key(grid) != _key(np.fliplr(grid))


def _full_dihedral(grid: np.ndarray) -> bool:
    variants = [np.rot90(grid, turns) for turns in range(4)]
    reflected = np.fliplr(grid)
    variants.extend(np.rot90(reflected, turns) for turns in range(4))
    return len({_key(variant) for variant in variants}) == 8


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high
        ):
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    legal = [
        (train, test)
        for train in range(2, 11)
        for test in range(1, 5)
        if train > test
        and (num_train is None or train == int(num_train))
        and (num_test is None or test == int(num_test))
    ]
    if not legal:
        raise ValueError("num_train must exceed num_test")
    return legal[seed % len(legal)]


def _choose_sizes(
    rng: np.random.Generator, train_count: int, test_count: int,
) -> tuple[list[int], list[int]]:
    mode = int(rng.choice(4, p=[0.24, 0.13, 0.32, 0.31]))
    if mode == 0:
        return [4] * train_count, [4] * test_count
    if mode == 1:
        size = int(rng.integers(3, 16))
        return [size] * train_count, [size] * test_count

    # The first demonstration is always large enough to show orientation.
    train = [int(rng.integers(3, 10))]
    train.extend(int(rng.integers(2, 12)) for _ in range(train_count - 1))
    test = [int(rng.integers(1, 16)) for _ in range(test_count)]
    if mode == 2:
        # Leave room for one- and two-cell tests, including the fallback
        # when a required one-cell color was already used in this puzzle.
        train = [max(4, x) for x in train]
        test[0] = int(rng.integers(1, 4))
    else:
        # A test larger than all demonstrations is also possible.
        train = [min(x, 11) for x in train]
        test[0] = int(rng.integers(max(train) + 1, 16))
    return train, test


def _orbit_cells(size: int, mode: str) -> list[list[tuple[int, int]]]:
    seen = set()
    groups = []
    for row in range(size):
        for col in range(size):
            if (row, col) in seen:
                continue
            if mode == "half_turn":
                group = {(row, col), (size - 1 - row, size - 1 - col)}
            elif mode == "quarter_turn":
                group = {
                    (row, col), (size - 1 - col, row),
                    (size - 1 - row, size - 1 - col), (col, size - 1 - row),
                }
            else:
                group = {(row, col), (row, size - 1 - col)}
            seen.update(group)
            groups.append(sorted(group))
    return groups


def _labels(
    rng: np.random.Generator, units: int, colors: int, concentration: str,
) -> np.ndarray:
    assert 1 <= colors <= units
    if colors == 1:
        return np.zeros(units, dtype=np.int8)
    if concentration == "dominant":
        dominant = float(rng.uniform(0.72, 0.94))
        probabilities = np.full(colors, (1 - dominant) / (colors - 1))
        probabilities[0] = dominant
    else:
        probabilities = rng.dirichlet(np.full(colors, 1.5))
    values = rng.choice(colors, size=units, p=probabilities).astype(np.int8)
    positions = rng.choice(units, size=colors, replace=False)
    values[positions] = rng.permutation(colors)
    return values


def _capacity(size: int, style: str, block: int) -> int:
    if style == "stripes":
        return size
    if style == "checker":
        return 1 if size == 1 else 2
    if style == "blocks":
        return ((size + block - 1) // block) ** 2
    if style == "frames":
        return (size + 1) // 2
    if style == "sparse":
        return min(size * size, max(2, size * size // 5 + 1))
    if style in ("half_turn", "quarter_turn", "reflection"):
        return len(_orbit_cells(size, style))
    if style == "repeat_rows":
        return (size - 1) * size
    if style == "repeat_columns":
        return size * (size - 1)
    if style == "repeat_both":
        return (size - 1) ** 2
    return size * size


def _sample_color_count(
    rng: np.random.Generator, capacity: int, informative: bool,
) -> int:
    high = min(10, capacity)
    if informative:
        return int(rng.integers(2, min(6, high) + 1))
    if high == 1:
        return 1
    weights = np.array([0.05, 0.15, 0.16, 0.22, 0.09, 0.08, 0.07, 0.06, 0.06, 0.06])
    weights = weights[:high] / weights[:high].sum()
    return int(rng.choice(np.arange(1, high + 1), p=weights))


def _pattern(
    rng: np.random.Generator, size: int, colors: int,
    style: str, block: int, concentration: str,
) -> np.ndarray:
    if style == "stripes":
        codes = _labels(rng, size, colors, concentration)
        if bool(rng.integers(0, 2)):
            return np.repeat(codes[:, None], size, axis=1)
        return np.repeat(codes[None, :], size, axis=0)
    if style == "checker":
        if colors == 1:
            return np.zeros((size, size), dtype=np.int8)
        rows, cols = np.indices((size, size))
        return ((rows + cols) % 2).astype(np.int8)
    if style == "blocks":
        side = (size + block - 1) // block
        coarse = _labels(rng, side * side, colors, concentration).reshape(side, side)
        return np.repeat(np.repeat(coarse, block, axis=0), block, axis=1)[:size, :size]
    if style == "frames":
        rows, cols = np.indices((size, size))
        depth = np.minimum.reduce((rows, cols, size - 1 - rows, size - 1 - cols))
        codes = rng.permutation(colors).astype(np.int8)
        return codes[depth % colors]
    if style == "sparse":
        tile = np.zeros((size, size), dtype=np.int8)
        if colors > 1:
            maximum = min(size * size - 1, max(colors - 1, size * size // 5))
            count = int(rng.integers(colors - 1, maximum + 1))
            positions = rng.choice(size * size, size=count, replace=False)
            marks = np.ones(count, dtype=np.int8)
            if colors > 2:
                marks = rng.integers(1, colors - 1, size=count, dtype=np.int8)
                marks[:colors - 2] = np.arange(1, colors - 1, dtype=np.int8)
            marks[-1] = colors - 1
            tile.flat[positions] = marks
        return tile
    if style in ("half_turn", "quarter_turn", "reflection"):
        groups = _orbit_cells(size, style)
        codes = _labels(rng, len(groups), colors, concentration)
        tile = np.empty((size, size), dtype=np.int8)
        for group, code in zip(groups, codes):
            for row, col in group:
                tile[row, col] = code
        return tile
    if style in ("repeat_rows", "repeat_columns", "repeat_both"):
        height = size - (style != "repeat_columns")
        width = size - (style != "repeat_rows")
        base = _labels(rng, height * width, colors, concentration).reshape(height, width)
        tile = np.empty((size, size), dtype=np.int8)
        tile[:height, :width] = base
        if height < size:
            source = int(rng.integers(0, height))
            tile[height:, :width] = base[source]
        if width < size:
            source = int(rng.integers(0, width))
            tile[:, width:] = tile[:, source:source + 1]
        return tile
    # Irregular cells can be either balanced or strongly concentrated.
    return _labels(rng, size * size, colors, concentration).reshape(size, size)


def _choose_palette(
    rng: np.random.Generator, count: int, anchor: int,
    anchor_policy: str, previous: list[tuple[int, ...]],
    excluded_colors: set[int],
) -> tuple[int, ...]:
    allowed = [
        color for color in range(10)
        if color not in excluded_colors
        and (anchor_policy != "exclude" or color != anchor)
    ]
    required = {anchor} if anchor_policy == "include" else set()
    eligible = [
        palette for palette in previous
        if len(palette) == count and required.issubset(palette)
        and set(palette).issubset(allowed)
    ]
    if eligible and rng.random() < 0.23:
        return eligible[int(rng.integers(0, len(eligible)))]
    chosen = set(required)
    if previous and rng.random() < 0.54:
        old = previous[int(rng.integers(0, len(previous)))]
        overlap = [color for color in old if color in allowed and color not in chosen]
        rng.shuffle(overlap)
        take = int(rng.integers(1, min(len(overlap), count - len(chosen)) + 1)) if overlap and count > len(chosen) else 0
        chosen.update(overlap[:take])
    available = [color for color in allowed if color not in chosen]
    if count > len(chosen):
        chosen.update(int(x) for x in rng.choice(available, size=count - len(chosen), replace=False))
    palette = list(chosen)
    rng.shuffle(palette)
    return tuple(palette)


def _make_pair(
    rng: np.random.Generator, size: int, anchor: int,
    anchor_policy: str, previous_palettes: list[tuple[int, ...]],
    used: set, requirement: str, excluded_colors: set[int],
) -> dict:
    styles = (
        "irregular", "dominant", "stripes", "checker", "blocks", "frames",
        "sparse", "half_turn", "quarter_turn", "reflection",
        "repeat_rows", "repeat_columns", "repeat_both",
    )
    for attempt in range(192):
        if attempt == 96 and size == 1:
            # A one-cell tile has only ten possible inputs, or one when an
            # anchor is required. Grow it if those inputs were already used.
            size = 2
        if requirement == "informative" or requirement == "distinct":
            style = "irregular"
        elif requirement == "repeated":
            style = "repeat_both" if size > 2 and rng.random() < 0.45 else "repeat_rows"
        else:
            style = styles[int(rng.integers(0, len(styles)))]
        if size == 1 and style in ("repeat_rows", "repeat_columns", "repeat_both"):
            style = "irregular"
        block = int(rng.integers(2, min(5, size + 1))) if size >= 2 else 1
        capacity = _capacity(size, style, block)
        if anchor_policy == "exclude":
            capacity = min(capacity, 9)
        capacity = min(capacity, 10 - len(excluded_colors))
        informative = requirement in ("informative", "distinct")
        colors = _sample_color_count(rng, capacity, informative)
        concentration = "dominant" if style == "dominant" or (not informative and rng.random() < 0.17) else "balanced"
        indices = _pattern(rng, size, colors, style, block, concentration)
        if requirement == "informative" and not (
            _full_dihedral(indices) and _distinct_rows(indices)
            and _distinct_rows(indices.T)
        ):
            continue
        if requirement == "distinct" and not (
            _distinct_rows(indices) and _distinct_rows(indices.T)
        ):
            continue
        if requirement == "repeated" and _distinct_rows(indices):
            continue
        palette = _choose_palette(
            rng, colors, anchor, anchor_policy, previous_palettes, excluded_colors,
        )
        tile = np.asarray(palette, dtype=np.int8)[indices]
        key = _key(tile)
        if key in used or _source_copy(tile):
            continue
        used.add(key)
        previous_palettes.append(palette)
        return {"input": tile, "output": _atlas(tile)}
    raise RuntimeError("could not construct a unique tile")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    anchor = int(rng.integers(0, 10))
    # Stratify the two independent shift axes across consecutive seeds.
    # Every 16 seeds contain all palette/structure combinations.
    palette_mode = seed % 4
    structure_mode = (seed // 4) % 4
    train_sizes, test_sizes = _choose_sizes(rng, train_count, test_count)
    if structure_mode == 0:
        train_sizes = [max(2, x) for x in train_sizes]
        test_sizes = [max(2, x) for x in test_sizes]
    if structure_mode == 1:
        train_sizes[1] = max(2, train_sizes[1])
        test_sizes = [max(2, x) for x in test_sizes]
    train_policy = ("include", "exclude", "include", "free")[palette_mode]
    test_policy = ("exclude", "include", "include", "free")[palette_mode]

    puzzle = {"train": [], "test": []}
    palettes = []
    used = set()
    child_seeds = rng.integers(0, np.iinfo(np.int64).max, size=train_count + test_count)
    for index, size in enumerate(train_sizes):
        requirement = "informative" if index == 0 else (
            "distinct" if structure_mode == 0 else
            "repeated" if structure_mode == 1 and index == 1 else "free"
        )
        pair_rng = np.random.default_rng(int(child_seeds[index]))
        excluded_colors = (
            set(palettes[0]) - {anchor}
            if palette_mode == 0 and index == 1 else set()
        )
        puzzle["train"].append(_make_pair(
            pair_rng, size, anchor, train_policy, palettes, used, requirement,
            excluded_colors,
        ))
    for index, size in enumerate(test_sizes):
        requirement = (
            "repeated" if structure_mode == 0 and index == 0 else
            "distinct" if structure_mode == 1 else "free"
        )
        pair_rng = np.random.default_rng(int(child_seeds[train_count + index]))
        puzzle["test"].append(_make_pair(
            pair_rng, size, anchor, test_policy, palettes, used, requirement, set(),
        ))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        tile, output = pair["input"], pair["output"]
        for grid in (tile, output):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
        assert tile.shape[0] == tile.shape[1]
        assert output.shape == (2 * tile.shape[0], 2 * tile.shape[1])
        assert not _source_copy(tile)
        assert _key(tile) not in seen
        seen.add(_key(tile))
        size = tile.shape[0]
        for quadrant, turns in (
            (output[:size, :size], 0),
            (output[:size, size:], 1),
            (output[size:, :size], 2),
            (output[size:, size:], 3),
        ):
            assert np.array_equal(quadrant, np.rot90(tile, turns))
        assert set(int(x) for x in np.unique(output)) == set(int(x) for x in np.unique(tile))
    assert any(_asymmetric(pair["input"]) for pair in train)
    # The first demonstration also has distinct rows and columns, so the
    # orientation of every quadrant is visible in a single pair.
    first = train[0]["input"]
    assert _full_dihedral(first) and _distinct_rows(first) and _distinct_rows(first.T)


def _infer_from_training(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    # Reconstruct every quadrant convention from displayed examples only.
    def variants(tile: np.ndarray) -> list[np.ndarray]:
        reflected = np.fliplr(tile)
        return [np.rot90(tile, turns) for turns in range(4)] + [
            np.rot90(reflected, turns) for turns in range(4)
        ]

    observed = []
    for pair in train:
        size = pair["input"].shape[0]
        output = pair["output"]
        observed.append((
            output[:size, :size], output[:size, size:],
            output[size:, :size], output[size:, size:],
        ))
    test_variants = variants(test_input)
    quadrants = []
    for position in range(4):
        candidates = [
            index for index in range(8)
            if all(
                np.array_equal(variants(pair["input"])[index], quadrants[position])
                for pair, quadrants in zip(train, observed)
            )
        ]
        assert len(candidates) == 1
        quadrants.append(test_variants[candidates[0]])
    return np.block([
        [quadrants[0], quadrants[1]],
        [quadrants[2], quadrants[3]],
    ])


def _same_puzzle(first: dict, second: dict) -> bool:
    return all(
        np.array_equal(left[key], right[key])
        for split in ("train", "test")
        for left, right in zip(first[split], second[split])
        for key in ("input", "output")
    ) and all(len(first[split]) == len(second[split]) for split in ("train", "test"))


def _canonical(tile: np.ndarray) -> tuple:
    mapping = {}
    values = []
    for raw in tile.flat:
        value = int(raw)
        if value not in mapping:
            mapping[value] = len(mapping)
        values.append(mapping[value])
    return tuple(tile.shape), tuple(values)


def _visible_styles(tile: np.ndarray) -> set[str]:
    size = tile.shape[0]
    colors = len(np.unique(tile))
    result = set()
    if colors > 1 and (
        all(len(set(row)) == 1 for row in tile)
        or all(len(set(column)) == 1 for column in tile.T)
    ):
        result.add("stripes")
    if size >= 2 and colors == 2 and all(
        (tile[row, col] == tile[0, 0]) == ((row + col) % 2 == 0)
        for row in range(size) for col in range(size)
    ):
        result.add("checker")
    if size >= 5 and colors > 1:
        if all(
            tile[row, col] == tile[
                min(row, col, size - 1 - row, size - 1 - col),
                min(row, col, size - 1 - row, size - 1 - col),
            ]
            for row in range(size) for col in range(size)
        ):
            result.add("frames")
        if any(
            all(
                np.all(
                    tile[row:min(row + block, size), col:min(col + block, size)]
                    == tile[row, col]
                )
                for row in range(0, size, block)
                for col in range(0, size, block)
            )
            for block in (2, 3, 4)
        ):
            result.add("blocks")
    return result


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    layouts, palettes, sizes, color_counts = set(), set(), set(), set()
    mixed_sizes = small_test = large_test = repeated_train = distinct_test = 0
    singletons = dominated = half_symmetric = quarter_symmetric = 0
    large_singletons = large_dominated = large_rich = combined_rich = 0
    shared_test_color = no_shared_test_color = palette_reuse = 0
    source_like = reverse_shift = no_shift = 0
    style_counts = {name: 0 for name in ("stripes", "checker", "frames", "blocks")}
    palette_overlap_counts = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train, test = puzzle["train"], puzzle["test"]
        for pair in test:
            assert np.array_equal(
                _infer_from_training(train, pair["input"]), pair["output"]
            )
        train_sizes = [pair["input"].shape[0] for pair in train]
        test_sizes = [pair["input"].shape[0] for pair in test]
        mixed_sizes += len(set(train_sizes + test_sizes)) > 1
        small_test += min(test_sizes) < min(train_sizes)
        large_test += max(test_sizes) > max(train_sizes)
        repeated_train += any(not _distinct_rows(pair["input"]) for pair in train[1:])
        distinct_test += any(_distinct_rows(pair["input"]) for pair in test)
        common = set.intersection(*(set(int(x) for x in np.unique(pair["input"])) for pair in train))
        all_train_distinct = all(
            _distinct_rows(pair["input"]) and _distinct_rows(pair["input"].T)
            for pair in train
        )
        any_test_repeated = any(
            not _distinct_rows(pair["input"]) or not _distinct_rows(pair["input"].T)
            for pair in test
        )
        all_test_distinct = all(
            _distinct_rows(pair["input"]) and _distinct_rows(pair["input"].T)
            for pair in test
        )
        test_retains = bool(common) and any(
            bool(common.intersection(np.unique(pair["input"]))) for pair in test
        )
        test_omits = bool(common) and all(
            not common.intersection(np.unique(pair["input"])) for pair in test
        )
        source_like += all_train_distinct and any_test_repeated and test_omits
        reverse_shift += any(
            not _distinct_rows(pair["input"]) or not _distinct_rows(pair["input"].T)
            for pair in train[1:]
        ) and all_test_distinct
        no_shift += any_test_repeated and test_retains and any(
            not _distinct_rows(pair["input"]) or not _distinct_rows(pair["input"].T)
            for pair in train[1:]
        )
        combined_rich += any(
            len(np.unique(pair["input"])) >= 6 for pair in test
        ) and (
            len(set(train_sizes + test_sizes)) > 1 or any_test_repeated
        )
        shared_test_color += bool(common) and any(
            bool(common.intersection(np.unique(pair["input"]))) for pair in test
        )
        no_shared_test_color += bool(common) and all(
            not common.intersection(np.unique(pair["input"])) for pair in test
        )
        palette_sets = [tuple(sorted(int(x) for x in np.unique(pair["input"]))) for pair in train + test]
        palette_reuse += len(set(palette_sets)) < len(palette_sets)
        for index, left in enumerate(palette_sets):
            for right in palette_sets[index + 1:]:
                palette_overlap_counts.add(len(set(left).intersection(right)))
        for pair in train + test:
            tile = pair["input"]
            size = tile.shape[0]
            counts = np.unique(tile, return_counts=True)[1]
            layouts.add(_canonical(tile))
            palettes.add(tuple(sorted(int(x) for x in np.unique(tile))))
            sizes.add(size)
            color_counts.add(len(counts))
            for style in _visible_styles(tile):
                style_counts[style] += 1
            singletons += bool(np.any(counts == 1))
            dominated += bool(counts.max() > tile.size / 2)
            if size >= 9:
                large_singletons += bool(np.any(counts == 1))
                large_dominated += bool(counts.max() > tile.size / 2)
                large_rich += len(counts) >= 7
            half_symmetric += bool(np.array_equal(tile, np.rot90(tile, 2)))
            quarter_symmetric += bool(np.array_equal(tile, np.rot90(tile, 1)))
            assert np.array_equal(pair["output"], _atlas(tile))
    assert len(layouts) >= 150 and len(palettes) >= 30
    assert len(sizes) >= 12 and len(color_counts) >= 8
    assert mixed_sizes >= 15 and small_test >= 8 and large_test >= 8
    assert repeated_train >= 10 and distinct_test >= 15
    assert singletons >= 10 and dominated >= 20
    assert large_singletons >= 3 and large_dominated >= 15 and large_rich >= 5
    assert half_symmetric >= 8 and quarter_symmetric >= 5
    assert shared_test_color >= 8 and no_shared_test_color >= 4 and palette_reuse >= 4
    assert {0, 1, 2, 3, 4}.issubset(palette_overlap_counts)
    assert combined_rich >= 3
    assert source_like >= 2 and reverse_shift >= 5 and no_shift >= 1
    assert {1, 2, 3, 4, 10}.issubset(color_counts)
    assert all(style_counts[name] >= minimum for name, minimum in (
        ("stripes", 5), ("checker", 5), ("frames", 3), ("blocks", 2),
    ))
    for offset, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(1000 + offset, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        validate_generated_puzzle(puzzle)
    for offset, (train_count, test_count) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + offset, train_count, test_count)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    same = generate_puzzle(314159, 7, 3)
    assert _same_puzzle(same, generate_puzzle(314159, 7, 3))
    assert not _same_puzzle(same, generate_puzzle(314160, 7, 3))
    for kwargs in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4},
    ):
        try:
            generate_puzzle(99, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid counts accepted: {kwargs}")
