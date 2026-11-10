"""Corner-source square contours, clipped by Manhattan ownership."""

import numpy as np


# Clockwise from the top left.
_CORNERS = (0, 1, 2, 3)
_SOURCE_INPUTS = {
    (13, (2, 0, 0, 4)),
    (12, (0, 3, 0, 8)),
    (7, (1, 2, 0, 8)),
    (10, (1, 2, 0, 0)),
    (17, (4, 0, 1, 8)),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _corner_position(size: int, corner: int) -> tuple[int, int]:
    return ((0, 0), (0, size - 1), (size - 1, size - 1),
            (size - 1, 0))[corner]


def _background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _sources(grid: np.ndarray, background: int) -> list[tuple[int, int, int]]:
    size = int(grid.shape[0])
    result = []
    for corner in _CORNERS:
        row, col = _corner_position(size, corner)
        color = int(grid[row, col])
        if color != background:
            result.append((corner, color, row * size + col))
    return result


def _output(input_grid: np.ndarray) -> np.ndarray:
    """Generate the contours in one array pass per visible source."""
    size = int(input_grid.shape[0])
    background = _background(input_grid)
    sources = _sources(input_grid, background)
    rows, cols = np.indices((size, size))
    distances = np.stack([
        abs(rows - (flat // size)) + abs(cols - (flat % size))
        for _, _, flat in sources
    ])
    minimum = distances.min(axis=0)
    unique = (distances == minimum).sum(axis=0) == 1
    winner = distances.argmin(axis=0)
    answer = np.full_like(input_grid, background)
    for index, (_, color, flat) in enumerate(sources):
        radius = np.maximum(abs(rows - (flat // size)),
                            abs(cols - (flat % size)))
        answer[unique & (winner == index) & (radius % 2 == 0)] = color
    return answer


def _check_output(input_grid: np.ndarray) -> np.ndarray:
    """Cellwise rule reconstruction, independent of the array construction."""
    size = int(input_grid.shape[0])
    background = _background(input_grid)
    sources = _sources(input_grid, background)
    answer = np.full_like(input_grid, background)
    for row in range(size):
        for col in range(size):
            best = size * 3
            owner = None
            tied = False
            for corner, color, _ in sources:
                source_row, source_col = _corner_position(size, corner)
                distance = abs(row - source_row) + abs(col - source_col)
                if distance < best:
                    best = distance
                    owner = (color, source_row, source_col)
                    tied = False
                elif distance == best:
                    tied = True
            if not tied and owner is not None:
                color, source_row, source_col = owner
                if max(abs(row - source_row), abs(col - source_col)) % 2 == 0:
                    answer[row, col] = color
    return answer


def _grid_key(grid: np.ndarray) -> tuple:
    return tuple(grid.shape), tuple(int(value) for value in grid.flat)


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def _source_signature(grid: np.ndarray) -> tuple:
    size = int(grid.shape[0])
    return size, tuple(int(grid[_corner_position(size, corner)])
                       for corner in _CORNERS)


def _count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


def _resolve_counts(seed: int, num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    train = _count(num_train, "num_train")
    test = _count(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be from 2 through 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be from 1 through 4")
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _masks(allowed: tuple[int, ...], minimum: int = 1) -> list[tuple[int, ...]]:
    return [tuple(corner for corner in allowed if bits & (1 << corner))
            for bits in range(1, 16)
            if minimum <= sum(bool(bits & (1 << corner)) for corner in allowed)
            and not any(bits & (1 << corner) for corner in _CORNERS
                        if corner not in allowed)]


def _adjacent(mask: tuple[int, ...]) -> bool:
    return len(mask) == 2 and ((mask[1] - mask[0]) % 4 in (1, 3))


def _choose_geometry(rng: np.random.Generator, masks: list[tuple[int, ...]],
                     sizes: list[int], used: set[tuple[int, tuple[int, ...]]],
                     prefer_new_size: bool) -> tuple[int, tuple[int, ...]]:
    choices = [(size, mask) for size in sizes for mask in masks
               if (size, mask) not in used]
    _require(bool(choices), "geometry choices exhausted")
    if prefer_new_size:
        old_sizes = {size for size, _ in used}
        fresh = [choice for choice in choices if choice[0] not in old_sizes]
        if fresh:
            choices = fresh
    selected = choices[int(rng.integers(len(choices)))]
    used.add(selected)
    return selected


def _make_input(size: int, mask: tuple[int, ...], background: int,
                colors: list[int]) -> np.ndarray:
    grid = np.full((size, size), background, dtype=np.int8)
    for corner, color in zip(mask, colors):
        grid[_corner_position(size, corner)] = color
    return grid


def _safe_input(size: int, mask: tuple[int, ...], background: int,
                colors: list[int], seen: set, allowed_sizes: list[int]) -> np.ndarray:
    # A rare exact source match is repaired by changing color roles first.
    # Exhaustive bounded fallback also handles a collision after that change.
    for candidate_size in [size] + [s for s in allowed_sizes if s != size]:
        rotated = list(colors)
        for _ in range(len(colors)):
            grid = _make_input(candidate_size, mask, background, rotated)
            if (_source_signature(grid) not in _SOURCE_INPUTS
                    and _grid_key(grid) not in seen):
                return grid
            rotated = rotated[1:] + rotated[:1]
    raise AssertionError("no unique non-source input available")


def _training_colors(rng: np.random.Generator, core: tuple[int, ...],
                     background: int, count: int, unseen: set[int],
                     force_zero: bool) -> list[int]:
    selected = [0] if force_zero else []
    priority = [color for color in core if color in unseen
                and color != background and color not in selected]
    rng.shuffle(priority)
    selected.extend(priority[:count - len(selected)])
    remaining = [color for color in core if color != background
                 and color not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[:count - len(selected)])
    _require(len(selected) == count, "insufficient training colors")
    rng.shuffle(selected)
    unseen.difference_update(selected)
    return selected


def _tie_witness(grid: np.ndarray) -> bool:
    size = int(grid.shape[0])
    sources = _sources(grid, _background(grid))
    for row in range(size):
        for col in range(size):
            distances = []
            for corner, _, _ in sources:
                sr, sc = _corner_position(size, corner)
                distances.append(abs(row - sr) + abs(col - sc))
            minimum = min(distances)
            if distances.count(minimum) > 1:
                for index, (corner, _, _) in enumerate(sources):
                    if distances[index] == minimum:
                        sr, sc = _corner_position(size, corner)
                        if max(abs(row - sr), abs(col - sc)) % 2 == 0:
                            return True
    return False


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed % (1 << 64))
    train_count, test_count = _resolve_counts(seed, num_train, num_test)

    # These are sampling preferences, not hidden rules. Every pair identifies
    # its own background and source colors in its input.
    core_size = int(rng.integers(5, 8))
    core = (0,) + tuple(int(x) for x in rng.choice(np.arange(1, 10),
                                                   size=core_size - 1,
                                                   replace=False))
    preferred_background = int(core[int(rng.integers(1, len(core)))])
    background_mode = seed % 3  # zero, one nonzero color, or varying colors
    reserve_corner = int(rng.integers(4)) if int(rng.integers(2)) else None
    allowed = tuple(c for c in _CORNERS if c != reserve_corner)
    used_geometry: set[tuple[int, tuple[int, ...]]] = set()

    adjacent = [mask for mask in _masks(allowed, 2)
                if _adjacent(mask)]
    anchor_mask = adjacent[int(rng.integers(len(adjacent)))]
    train_specs = [
        _choose_geometry(rng, [anchor_mask], list(range(7, 18, 2)),
                         used_geometry, False),
        _choose_geometry(rng, [mask for mask in _masks(allowed, 3)
                               if len(mask) == 3], list(range(8, 19)),
                         used_geometry, True),
    ]
    for _ in range(train_count - 2):
        counts = [1, 2, 3, 4] if reserve_corner is None else [1, 2, 3]
        weights = [0.08, 0.40, 0.32, 0.20] if reserve_corner is None else [0.09, 0.52, 0.39]
        source_count = counts[int(rng.choice(len(counts), p=weights))]
        mask_options = [mask for mask in _masks(allowed) if len(mask) == source_count]
        train_specs.append(_choose_geometry(rng, mask_options,
                                            list(range(7, 19)),
                                            used_geometry, True))
    rng.shuffle(train_specs)

    if background_mode == 0:
        train_backgrounds = [0] * train_count
    elif background_mode == 1:
        train_backgrounds = [preferred_background] * train_count
    else:
        train_backgrounds = [0, preferred_background]
        train_backgrounds.extend(
            0 if int(rng.integers(2)) else preferred_background
            for _ in range(train_count - 2)
        )
        rng.shuffle(train_backgrounds)

    unseen_core = set(core)
    zero_demonstrated = False
    train = []
    seen_inputs = set()
    for (size, mask), background in zip(train_specs, train_backgrounds):
        force_zero = background != 0 and not zero_demonstrated
        colors = _training_colors(rng, core, background, len(mask),
                                  unseen_core, force_zero)
        if 0 in colors:
            zero_demonstrated = True
        grid = _safe_input(size, mask, background, colors, seen_inputs,
                           [s for s in range(7, 19)
                            if (s, mask) not in used_geometry])
        used_geometry.add((int(grid.shape[0]), mask))
        seen_inputs.add(_grid_key(grid))
        train.append({"input": grid, "output": _output(grid)})

    shown_colors = {int(value) for pair in train for value in
                    np.unique(pair["input"])}
    train_sizes = [int(pair["input"].shape[0]) for pair in train]
    max_train = max(train_sizes)
    planned_novel_color = rng.random() < 0.34
    test = []
    for index in range(test_count):
        category = int(rng.integers(3))
        if category == 0:
            sizes = list(range(7, max_train))
        elif category == 1:
            sizes = sorted(set(train_sizes))
        else:
            sizes = list(range(max_train + 1, 31))
        if not sizes:
            sizes = sorted(set(train_sizes))

        trained_masks = [tuple(corner for corner, _, _ in
                               _sources(pair["input"], _background(pair["input"])))
                         for pair in train]
        force_new_corner = (index == 0 and reserve_corner is not None
                            and rng.random() < 0.65)
        if not force_new_corner and int(rng.integers(4)) == 0:
            mask_options = list(set(trained_masks))
        else:
            mask_options = _masks(_CORNERS, 2)
            if any(len(mask) == 1 for mask in trained_masks):
                mask_options += [mask for mask in _masks(_CORNERS)
                                 if len(mask) == 1]
            if force_new_corner:
                mask_options = [mask for mask in mask_options
                                if reserve_corner in mask]
            counts = sorted(set(len(mask) for mask in mask_options))
            weights = {1: 0.08, 2: 0.42, 3: 0.33, 4: 0.17}
            probabilities = np.array([weights[count] for count in counts],
                                     dtype=float)
            probabilities /= probabilities.sum()
            wanted = counts[int(rng.choice(len(counts), p=probabilities))]
            mask_options = [mask for mask in mask_options if len(mask) == wanted]
        size, mask = _choose_geometry(rng, mask_options, sizes,
                                      used_geometry, False)
        background = train_backgrounds[int(rng.integers(train_count))]
        familiar = [color for color in shown_colors if color != background
                    and (color != 0 or zero_demonstrated)]
        rng.shuffle(familiar)
        novel = [color for color in range(10) if color not in shown_colors]
        use_novel = bool(novel) and (index == 0 and planned_novel_color
                                    or rng.random() < 0.16)
        colors = [int(novel[int(rng.integers(len(novel)))])] if use_novel else []
        colors.extend(familiar[:len(mask) - len(colors)])
        if len(colors) < len(mask):
            extras = [color for color in core if color != background
                      and color not in colors]
            rng.shuffle(extras)
            colors.extend(extras[:len(mask) - len(colors)])
        rng.shuffle(colors)
        grid = _safe_input(size, mask, background, colors, seen_inputs,
                           [s for s in sizes + list(range(7, 31))
                            if (s, mask) not in used_geometry])
        used_geometry.add((int(grid.shape[0]), mask))
        seen_inputs.add(_grid_key(grid))
        test.append({"input": grid, "output": _output(grid)})

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair: dict, split: str, seen: set) -> dict:
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "invalid pair schema")
    for name in ("input", "output"):
        grid = pair[name]
        _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                 "grid must be a two-dimensional NumPy array")
        _require(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
        _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                 "grid dimensions out of range")
        _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                 "grid color out of range")
    grid = pair["input"]
    answer = pair["output"]
    _require(grid.shape == answer.shape and grid.shape[0] == grid.shape[1],
             "input and output must be equally sized squares")
    size = int(grid.shape[0])
    _require(7 <= size <= 30, "square too small for clear contours")
    background = _background(grid)
    _require(int(np.count_nonzero(grid == background)) >= size * size - 4,
             "background is not dominant")
    sources = _sources(grid, background)
    _require(1 <= len(sources) <= 4, "invalid source count")
    _require(len({color for _, color, _ in sources}) == len(sources),
             "source colors must be distinct")
    if split == "train":
        _require(7 <= size <= 18, "train size outside supported range")
    for row in range(size):
        for col in range(size):
            if row not in (0, size - 1) or col not in (0, size - 1):
                _require(int(grid[row, col]) == background,
                         "non-corner input mark")
    key = _grid_key(grid)
    _require(key not in seen, "duplicate input")
    seen.add(key)
    _require(_source_signature(grid) not in _SOURCE_INPUTS,
             "source task input copied")
    _require(np.array_equal(answer, _check_output(grid)),
             "output violates the corner contour rule")
    _require(not np.array_equal(grid, answer), "unchanged pair")
    _require(set(int(x) for x in np.unique(grid)) ==
             set(int(x) for x in np.unique(answer)), "palette changed")
    return {"size": size, "mask": tuple(c for c, _, _ in sources),
            "background": background, "colors": {c for _, c, _ in sources},
            "tie_witness": _tie_witness(grid)}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain train and test")
    _require(isinstance(puzzle["train"], list) and
             isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and
             train_count > test_count, "invalid pair counts")
    seen = set()
    train = [_inspect_pair(pair, "train", seen) for pair in puzzle["train"]]
    test = [_inspect_pair(pair, "test", seen) for pair in puzzle["test"]]
    _require(any(len(r["mask"]) == 2 and _adjacent(r["mask"]) and
                 r["size"] % 2 == 1 and r["tie_witness"] for r in train),
             "training needs a painted-radius tie witness")
    _require(any(len(r["mask"]) >= 3 for r in train),
             "training needs multisource boundary intersections")
    _require(any(len(r["mask"]) == 2 for r in train),
             "training needs a simple two-source comparison")
    train_geometry = {(r["size"], r["mask"]) for r in train}
    _require(len(train_geometry) == len(train),
             "recolored training geometry repeated")
    _require(all((r["size"], r["mask"]) not in train_geometry for r in test),
             "test repeats a demonstrated geometry")
    _require(len({(r["size"], r["mask"]) for r in test}) == len(test),
             "test geometry repeated")
    shown_backgrounds = {r["background"] for r in train}
    _require(all(r["background"] in shown_backgrounds for r in test),
             "test background absent from training")
    trained_zero_source = any(0 in r["colors"] for r in train)
    _require(all(0 not in r["colors"] or trained_zero_source for r in test),
             "zero source absent from training")
    _require(all(len(r["mask"]) != 1 or
                 any(len(demo["mask"]) == 1 for demo in train) for r in test),
             "single source absent from training")


def _expect_bad_count(train: int | None, test: int | None) -> None:
    try:
        generate_puzzle(909, train, test)
    except ValueError:
        return
    raise AssertionError("illegal count was accepted")


def self_test() -> None:
    train_counts, test_counts, puzzles, geometries = set(), set(), set(), set()
    backgrounds, train_masks, test_masks, size_relations = set(), set(), set(), set()
    zero_source = novel_test = familiar_test = repeated_mask = 0
    tied_test = untied_test = four_test = 0
    even_adjacent_test = odd_adjacent_test = opposite_test = single_test = 0
    unseen_corner_test = train_even_triple = train_odd_triple = 0
    fixed_zero = fixed_nonzero = varied_background = 0
    varying_parity = varying_test_counts = 0
    axis_combinations = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzles.add(_puzzle_key(puzzle))
        train_records = [_inspect_pair(p, "train", set()) for p in puzzle["train"]]
        test_records = [_inspect_pair(p, "test", set()) for p in puzzle["test"]]
        max_train = max(r["size"] for r in train_records)
        trained_colors = set().union(*(r["colors"] | {r["background"]}
                                       for r in train_records))
        trained_masks = {r["mask"] for r in train_records}
        trained_corners = set().union(*(set(r["mask"]) for r in train_records))
        train_backgrounds = {r["background"] for r in train_records}
        fixed_zero += train_backgrounds == {0}
        fixed_nonzero += len(train_backgrounds) == 1 and 0 not in train_backgrounds
        varied_background += len(train_backgrounds) > 1
        varying_parity += len({r["size"] % 2 for r in train_records}) > 1
        varying_test_counts += len({len(r["mask"]) for r in test_records}) > 1
        for r in train_records:
            train_masks.add(r["mask"])
            backgrounds.add(r["background"])
            zero_source += 0 in r["colors"]
            geometries.add(("train", r["size"], r["mask"]))
            train_even_triple += len(r["mask"]) == 3 and r["size"] % 2 == 0
            train_odd_triple += len(r["mask"]) == 3 and r["size"] % 2 == 1
            axis_combinations.add(("train", len(r["mask"]), r["size"] % 2,
                                   r["background"] != 0))
        for r in test_records:
            test_masks.add(r["mask"])
            backgrounds.add(r["background"])
            geometries.add(("test", r["size"], r["mask"]))
            size_relations.add("smaller" if r["size"] < max_train else
                               "same" if r["size"] == max_train else "larger")
            novel_test += bool(r["colors"] - trained_colors)
            familiar_test += r["colors"].issubset(trained_colors)
            repeated_mask += r["mask"] in trained_masks
            tied_test += r["tie_witness"]
            untied_test += not r["tie_witness"]
            four_test += len(r["mask"]) == 4
            single_test += len(r["mask"]) == 1
            even_adjacent_test += _adjacent(r["mask"]) and r["size"] % 2 == 0
            odd_adjacent_test += _adjacent(r["mask"]) and r["size"] % 2 == 1
            opposite_test += len(r["mask"]) == 2 and not _adjacent(r["mask"])
            unseen_corner_test += not set(r["mask"]).issubset(trained_corners)
            axis_combinations.add(("test", len(r["mask"]), r["size"] % 2,
                                   r["background"] != 0))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "pair counts lack diversity")
    _require(len(puzzles) == 42 and len(geometries) >= 100,
             "structural diversity too narrow")
    _require(len(train_masks) >= 9 and len(test_masks) >= 10,
             "corner subsets lack diversity")
    _require(size_relations == {"smaller", "same", "larger"},
             "test size shifts lack diversity")
    _require(len(backgrounds) >= 5 and zero_source >= 5,
             "background and zero-source roles lack diversity")
    _require(novel_test >= 5 and familiar_test >= 10,
             "test color roles lack diversity")
    _require(repeated_mask >= 5 and tied_test >= 5 and untied_test >= 5
             and four_test >= 2, "test layout modes lack diversity")
    _require(even_adjacent_test >= 3 and odd_adjacent_test >= 3
             and opposite_test >= 3 and single_test >= 1
             and unseen_corner_test >= 3, "corner boundary variants missing")
    _require(train_even_triple >= 5 and train_odd_triple >= 5
             and varying_parity >= 20, "training parity is too fixed")
    _require(fixed_zero >= 5 and fixed_nonzero >= 5
             and varied_background >= 5, "background modes are missing")
    _require(varying_test_counts >= 5 and len(axis_combinations) >= 24,
             "compatible variations do not combine")
    for seed, train, test in ((2001, 2, 1), (2002, 10, 1), (2003, 10, 4),
                              (3001, None, 1), (3002, None, 4),
                              (3003, 2, None), (3004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "train count not honored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "test count not honored")
    first = generate_puzzle(123456, 7, 3)
    second = generate_puzzle(123456, 7, 3)
    _require(_puzzle_key(first) == _puzzle_key(second), "nondeterministic puzzle")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (2, 3), (3, 4), (4, 4), (10, 10)):
        _expect_bad_count(train, test)
