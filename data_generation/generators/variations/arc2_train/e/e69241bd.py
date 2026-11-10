import numpy as np


_SOURCE_PAIRS = (
    (
        (
            "510505005",
            "500503505",
            "055050500",
            "005050050",
            "500500005",
            "055055050",
            "070500500",
            "005055005",
            "050000550",
        ),
        (
            "511535005",
            "511533505",
            "055053500",
            "005053350",
            "500533335",
            "755055350",
            "777500500",
            "775055005",
            "750000550",
        ),
    ),
    (
        (
            "506050050",
            "050555050",
            "500005580",
            "050505005",
            "055000505",
            "505555005",
            "500055050",
            "055500500",
            "005050050",
        ),
        (
            "566650058",
            "056555058",
            "566665588",
            "056565885",
            "055666585",
            "505555885",
            "500055850",
            "055500500",
            "005050050",
        ),
    ),
    (
        (
            "0050050",
            "5540055",
            "0005500",
            "0505050",
            "5050050",
            "0505085",
            "3055500",
        ),
        (
            "0054450",
            "5544455",
            "4445500",
            "4545850",
            "5058850",
            "3505885",
            "3355588",
        ),
    ),
    (
        (
            "000503050",
            "555000550",
            "085505050",
            "005050055",
            "505005000",
            "000500500",
            "550505005",
            "050050550",
            "005506005",
        ),
        (
            "000533350",
            "555333550",
            "885535050",
            "885050055",
            "585005000",
            "888500500",
            "558505005",
            "058856550",
            "005566665",
        ),
    ),
)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grid_fingerprint(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _pair_fingerprint(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return (_grid_fingerprint(input_grid), _grid_fingerprint(output_grid))


def _puzzle_fingerprint(puzzle: dict) -> tuple:
    return tuple(
        (
            section,
            tuple(
                _pair_fingerprint(pair["input"], pair["output"])
                for pair in puzzle[section]
            ),
        )
        for section in ("train", "test")
    )


def _same_as_rows(grid: np.ndarray, rows: tuple[str, ...]) -> bool:
    if grid.ndim != 2:
        return False
    if grid.shape != (len(rows), len(rows[0])):
        return False
    for row_index, row in enumerate(rows):
        for column_index, character in enumerate(row):
            if int(grid[row_index, column_index]) != int(character):
                return False
    return True


def _matches_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    for source_input, source_output in _SOURCE_PAIRS:
        if (
            _same_as_rows(input_grid, source_input)
            and _same_as_rows(output_grid, source_output)
        ):
            return True
    return False


def _connected_components(open_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = open_mask.shape
    visited = np.zeros((height, width), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for start_row in range(height):
        for start_column in range(width):
            if not open_mask[start_row, start_column]:
                continue
            if visited[start_row, start_column]:
                continue

            visited[start_row, start_column] = True
            stack = [(start_row, start_column)]
            component: list[tuple[int, int]] = []

            while stack:
                row, column = stack.pop()
                component.append((row, column))

                for delta_row, delta_column in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ):
                    next_row = row + delta_row
                    next_column = column + delta_column

                    if not (0 <= next_row < height):
                        continue
                    if not (0 <= next_column < width):
                        continue
                    if not open_mask[next_row, next_column]:
                        continue
                    if visited[next_row, next_column]:
                        continue

                    visited[next_row, next_column] = True
                    stack.append((next_row, next_column))

            components.append(component)

    return components


def _component_touches_boundary(
    component: list[tuple[int, int]],
    height: int,
    width: int,
) -> bool:
    return any(
        row == 0
        or row == height - 1
        or column == 0
        or column == width - 1
        for row, column in component
    )


def _has_diagonal_contrast(input_grid: np.ndarray, output_grid: np.ndarray, wall: int) -> bool:
    height, width = input_grid.shape
    for row in range(height - 1):
        for column in range(width - 1):
            for first, second in (
                ((row, column), (row + 1, column + 1)),
                ((row + 1, column), (row, column + 1)),
            ):
                if (input_grid[first] != wall and input_grid[second] != wall
                        and output_grid[first] != output_grid[second]):
                    return True
    return False


def _candidate_wall_mask(
    rng: np.random.Generator,
    height: int,
    width: int,
    regime: str,
) -> np.ndarray:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown pair regime")
    mode = int(rng.choice(4, p=(0.47, 0.27, 0.15, 0.11)))
    density = float(rng.uniform(0.32, 0.52))
    if mode == 1:
        density -= 0.08
    elif mode == 2:
        density -= 0.11
    elif mode == 3:
        density -= 0.13
    walls = rng.random((height, width)) < density

    if mode in (1, 2):
        count = int(rng.integers(1, max(3, height * width // 18)))
        for _ in range(count):
            row = int(rng.integers(0, height))
            column = int(rng.integers(0, width))
            direction = int(rng.integers(0, 4))
            length = int(rng.integers(2, min(6, max(height, width)) + 1))
            for _ in range(length):
                if not (0 <= row < height and 0 <= column < width):
                    break
                walls[row, column] = True
                if rng.random() < 0.23:
                    direction = (direction + int(rng.choice((-1, 1)))) % 4
                dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[direction]
                row += dr
                column += dc
    if mode == 2 and height >= 6 and width >= 6:
        top = int(rng.integers(1, height - 3))
        left = int(rng.integers(1, width - 3))
        bottom = min(height - 2, top + int(rng.integers(2, 5)))
        right = min(width - 2, left + int(rng.integers(2, 5)))
        walls[top, left:right + 1] = True
        walls[bottom, left:right + 1] = True
        walls[top:bottom + 1, left] = True
        walls[top:bottom + 1, right] = True
        if rng.random() < 0.55:
            walls[top, int(rng.integers(left + 1, right))] = False
    if mode == 3:
        if rng.random() < 0.5:
            walls[int(rng.integers(1, height - 1)), :] = True
        else:
            walls[:, int(rng.integers(1, width - 1))] = True
        if rng.random() < 0.45:
            if rng.random() < 0.5:
                walls[int(rng.integers(1, height - 1)), :] = True
            else:
                walls[:, int(rng.integers(1, width - 1))] = True
    return walls


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    puzzle_parameters: dict,
    height: int,
    width: int,
    seed_colors: tuple[int, ...],
    require_diagonal: bool,
    prefer_small: bool,
    prefer_interior: bool,
    require_unseeded: bool,
) -> tuple[np.ndarray, np.ndarray]:
    background = int(puzzle_parameters["background"])
    wall = int(puzzle_parameters["wall"])
    seed_count = len(seed_colors)

    for attempt in range(240):
        # Relax optional placement preferences in the final bounded phase.
        use_small = prefer_small and attempt < 160
        use_interior = prefer_interior and attempt < 160
        wall_mask = _candidate_wall_mask(
            rng,
            height,
            width,
            regime,
        )

        wall_fraction = float(np.count_nonzero(wall_mask)) / (height * width)
        if not 0.13 <= wall_fraction <= 0.64:
            continue

        components = _connected_components(~wall_mask)
        if len(components) < seed_count + int(require_unseeded):
            continue

        eligible = [
            index
            for index, component in enumerate(components)
            if len(component) >= 2
        ]
        if len(eligible) < seed_count:
            continue
        if use_small and not any(2 <= len(components[i]) <= 3 for i in eligible):
            continue
        if use_interior and not any(
            not _component_touches_boundary(components[i], height, width)
            for i in eligible
        ):
            continue

        selected: list[int] = []
        available = eligible.copy()
        if use_small:
            choices = [i for i in available if 2 <= len(components[i]) <= 3]
            first = choices[int(rng.integers(len(choices)))]
            selected.append(first)
            available.remove(first)
        if use_interior and not any(
            not _component_touches_boundary(components[i], height, width)
            for i in selected
        ):
            choices = [i for i in available if not _component_touches_boundary(components[i], height, width)]
            if not choices:
                continue
            first = choices[int(rng.integers(len(choices)))]
            selected.append(first)
            available.remove(first)
        while len(selected) < seed_count:
            if not available:
                break
            if rng.random() < 0.55:
                available.sort(key=lambda i: len(components[i]), reverse=True)
                pool = available[: min(3, len(available))]
            else:
                pool = available
            chosen = pool[int(rng.integers(len(pool)))]
            selected.append(chosen)
            available.remove(chosen)
        if len(selected) != seed_count:
            continue
        if regime == "target_test" and height * width >= 100:
            minimum_large_region = max(8, int(height * width * 0.06))
            if max(len(components[index]) for index in selected) < minimum_large_region:
                continue

        ordered_colors = [int(seed_colors[int(index)]) for index in rng.permutation(seed_count)]

        input_grid = np.full(
            (height, width),
            background,
            dtype=np.int8,
        )
        input_grid[wall_mask] = wall

        seed_assignments: list[tuple[int, int]] = []
        for component_index, color in zip(selected, ordered_colors):
            component = components[component_index]
            seed_position = int(rng.integers(0, len(component)))
            row, column = component[seed_position]
            input_grid[row, column] = color
            seed_assignments.append((component_index, color))

        output_grid = input_grid.copy()
        for component_index, color in seed_assignments:
            for row, column in components[component_index]:
                output_grid[row, column] = color

        if np.array_equal(input_grid, output_grid):
            continue
        if require_diagonal and not _has_diagonal_contrast(input_grid, output_grid, wall):
            continue

        return input_grid, output_grid

    raise RuntimeError("could not construct a valid maze pair")


def _normalize_optional_count(
    name: str,
    value: int | None,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")

    normalized = int(value)
    if not lower <= normalized <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return normalized


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _normalize_optional_count(
        "num_train",
        num_train,
        2,
        10,
    )
    test_count = _normalize_optional_count(
        "num_test",
        num_test,
        1,
        4,
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = 1 + (seed % 4)
        legal_train_count = 10 - test_count
        train_count = (
            test_count
            + 1
            + ((seed // 4) % legal_train_count)
        )
        return train_count, test_count

    if train_count is None:
        assert test_count is not None
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))
        return train_count, test_count

    maximum_test = min(4, train_count - 1)
    if maximum_test < 1:
        raise ValueError("no legal num_test exists for num_train")
    test_count = int(rng.integers(1, maximum_test + 1))
    return train_count, test_count


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    colors = [int(x) for x in rng.permutation(10)]
    palette_size = int(rng.integers(3, 7))
    return {
        "background": colors[0],
        "wall": colors[1],
        "seed_palette": tuple(colors[2:2 + palette_size]),
        "new_colors": tuple(colors[2 + palette_size:]),
    }


def _sample_dimensions(rng: np.random.Generator, band: int) -> tuple[int, int]:
    ranges = ((5, 9), (8, 14), (13, 23))
    low, high = ranges[band]
    height = int(rng.integers(low, high + 1))
    width = height if rng.random() < 0.25 else int(rng.integers(low, high + 1))
    return height, width


def _sample_seed_count(rng: np.random.Generator, area: int, index: int, regime: str) -> int:
    maximum = 2 if area < 40 else 3 if area < 75 else 4 if area < 130 else 5
    if regime == "source_train" and index == 0:
        return min(maximum, int(rng.integers(2, 4)))
    if regime == "source_train" and index == 1:
        return min(maximum, int(rng.integers(1, 4)))
    return int(rng.integers(1, maximum + 1))


def _sample_colors(
    rng: np.random.Generator,
    palette: tuple[int, ...],
    count: int,
    used_training: set[int],
    new_colors: tuple[int, ...],
    regime: str,
    index: int,
) -> tuple[int, ...]:
    if regime == "source_train" and index == 0:
        chosen = list(rng.choice(palette, size=count, replace=False))
    elif regime == "source_train" and index == 1 and count >= 2:
        fresh = [color for color in palette if color not in used_training]
        chosen = [fresh[int(rng.integers(len(fresh)))]] if fresh else []
        chosen.extend(int(x) for x in rng.choice(palette, size=count - len(chosen), replace=False))
    else:
        chosen = [int(x) for x in rng.choice(palette, size=min(count, len(palette)), replace=False)]
        while len(chosen) < count:
            chosen.append(int(palette[int(rng.integers(len(palette)))]))
    chosen = [int(x) for x in chosen]
    if regime == "target_test" and new_colors and rng.random() < 0.22:
        chosen[int(rng.integers(len(chosen)))] = int(new_colors[int(rng.integers(len(new_colors)))])
    elif count >= 3 and rng.random() < 0.20:
        chosen[-1] = chosen[0]
    return tuple(chosen)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")

    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError("seed must be a non-negative integer")

    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_counts(
        seed_value,
        rng,
        num_train,
        num_test,
    )

    puzzle_parameters = sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    seen_layouts: set[tuple] = set()
    used_training: set[int] = set()

    def make_unique_pair(
        regime: str,
        height: int,
        width: int,
        colors: tuple[int, ...],
        diagonal: bool,
        small: bool,
        interior: bool,
        unseeded: bool,
    ) -> dict:
        for _ in range(12):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            input_grid, output_grid = _generate_pair(
                pair_rng,
                regime,
                puzzle_parameters,
                height,
                width,
                colors,
                diagonal,
                small,
                interior,
                unseeded,
            )

            input_fingerprint = _grid_fingerprint(input_grid)
            pair_fingerprint = _pair_fingerprint(
                input_grid,
                output_grid,
            )
            layout_fingerprint = (
                tuple(int(x) for x in input_grid.shape),
                (
                    input_grid == int(puzzle_parameters["wall"])
                ).tobytes(),
            )

            if input_fingerprint in seen_inputs:
                continue
            if pair_fingerprint in seen_pairs:
                continue
            if layout_fingerprint in seen_layouts:
                continue
            if _matches_source_pair(input_grid, output_grid):
                continue

            seen_inputs.add(input_fingerprint)
            seen_pairs.add(pair_fingerprint)
            seen_layouts.add(layout_fingerprint)
            return {
                "input": input_grid,
                "output": output_grid,
            }

        raise RuntimeError("could not create a unique pair")

    for regime, count in (("source_train", train_count), ("target_test", test_count)):
        for index in range(count):
            if regime == "source_train" and index == 0:
                band = 0
            elif regime == "source_train" and index == 1:
                band = 1
            else:
                probabilities = (0.30, 0.48, 0.22) if regime == "source_train" else (0.16, 0.46, 0.38)
                band = int(rng.choice(3, p=probabilities))
            height, width = _sample_dimensions(rng, band)
            if regime == "source_train" and index == 1:
                first_shape = puzzle["train"][0]["input"].shape
                if (height, width) == first_shape:
                    width = 10 if width == 9 else width + 1
            seed_count = _sample_seed_count(rng, height * width, index, regime)
            colors = _sample_colors(
                rng,
                puzzle_parameters["seed_palette"],
                seed_count,
                used_training,
                puzzle_parameters["new_colors"],
                regime,
                index,
            )
            pair = make_unique_pair(
                regime, height, width, colors,
                regime == "source_train" and index == 0,
                (regime == "source_train" and index == 1 and seed_count <= 3 and rng.random() < 0.55)
                or (regime == "target_test" and seed_count >= 2 and height * width < 130
                    and rng.random() < 0.25),
                rng.random() < 0.28,
                regime == "source_train" and index == 0,
            )
            puzzle["train" if regime == "source_train" else "test"].append(pair)
            if regime == "source_train":
                used_training.update(colors)

    validate_generated_puzzle(puzzle)
    return puzzle


def _infer_training_conventions(puzzle: dict) -> tuple[int, int, frozenset[int]]:
    changed_inputs: set[int] = set()
    fill_colors: set[int] = set()
    input_colors: set[int] = set()
    for pair in puzzle["train"]:
        inp, out = pair["input"], pair["output"]
        changed = inp != out
        _check(bool(np.any(changed)), "each training pair must demonstrate a fill")
        changed_inputs.update(int(x) for x in np.unique(inp[changed]))
        fill_colors.update(int(x) for x in np.unique(out[changed]))
        input_colors.update(int(x) for x in np.unique(inp))
    _check(len(changed_inputs) == 1, "training must show a consistent background")
    background = next(iter(changed_inputs))
    _check(len(fill_colors) >= 2, "training must demonstrate copying at least two seed colors")
    wall_candidates = input_colors - {background} - fill_colors
    _check(len(wall_candidates) == 1, "training must identify exactly one wall color")
    wall = next(iter(wall_candidates))
    _check(all(np.any(pair["input"] == wall) for pair in puzzle["train"]),
           "wall must be visible in every training input")
    return background, wall, frozenset(fill_colors)


def _analyze_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    background: int,
    wall: int,
) -> dict:
    _check(bool(np.any(input_grid == background)), "input needs background")
    _check(bool(np.any(input_grid == wall)), "input needs walls")
    height, width = input_grid.shape
    seeds = {
        (int(row), int(column)): int(input_grid[row, column])
        for row, column in np.argwhere((input_grid != background) & (input_grid != wall))
    }
    _check(len(seeds) >= 1, "input needs at least one seed")
    components = _connected_components(input_grid != wall)
    expected = input_grid.copy()
    seeded_sizes: list[int] = []
    seeded_boundary = 0
    seeded_interior = 0
    unseeded = 0
    for component in components:
        present = [(position, seeds[position]) for position in component if position in seeds]
        _check(len(present) <= 1, "competing seeds in one open component")
        if present:
            color = present[0][1]
            _check(len(component) >= 2, "a seed must make a visible fill")
            seeded_sizes.append(len(component))
            boundary = _component_touches_boundary(component, height, width)
            seeded_boundary += int(boundary)
            seeded_interior += int(not boundary)
            for row, column in component:
                expected[row, column] = color
        else:
            unseeded += 1
            _check(all(input_grid[row, column] == background for row, column in component),
                   "unseeded component contains a non-background cell")
    _check(len(seeded_sizes) == len(seeds), "every seed needs its own component")
    _check(np.array_equal(expected, output_grid), "incorrect orthogonal flood fill")
    _check(not np.array_equal(input_grid, output_grid), "pair must visibly transform")
    wall_fraction = float(np.count_nonzero(input_grid == wall)) / input_grid.size
    _check(0.13 <= wall_fraction <= 0.64, "wall fraction outside supported range")
    wall_components = _connected_components(input_grid == wall)
    return {
        "seed_count": len(seeds),
        "seed_colors": frozenset(seeds.values()),
        "repeated_seed_color": len(set(seeds.values())) < len(seeds),
        "open_components": len(components),
        "wall_components": len(wall_components),
        "unseeded": unseeded,
        "seeded_sizes": tuple(seeded_sizes),
        "boundary_seeded": seeded_boundary,
        "interior_seeded": seeded_interior,
        "diagonal_contrast": _has_diagonal_contrast(input_grid, output_grid, wall),
        "full_wall_line": bool(np.any(np.all(input_grid == wall, axis=0))
                               or np.any(np.all(input_grid == wall, axis=1))),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "puzzle must have exactly train and test")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
           "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
           "illegal train/test counts")
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
                   "pair must have input and output")
            for grid in pair.values():
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2,
                       "grids must be two-dimensional NumPy arrays")
                _check(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                       "grid dimensions outside ARC bounds")
                _check(np.issubdtype(grid.dtype, np.integer), "grids must be integers")
                _check(0 <= int(np.min(grid)) and int(np.max(grid)) <= 9,
                       "colors outside ARC range")
            inp, out = pair["input"], pair["output"]
            _check(inp.shape == out.shape, "output shape must equal input shape")
            _check(not _matches_source_pair(inp, out), "source pair copied exactly")
            input_key = _grid_fingerprint(inp)
            pair_key = _pair_fingerprint(inp, out)
            _check(input_key not in seen_inputs and pair_key not in seen_pairs,
                   "duplicate input or pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
    background, wall, demonstrated = _infer_training_conventions(puzzle)
    train_info = [_analyze_pair(p["input"], p["output"], background, wall)
                  for p in puzzle["train"]]
    test_info = [_analyze_pair(p["input"], p["output"], background, wall)
                 for p in puzzle["test"]]
    _check(any(info["diagonal_contrast"] for info in train_info),
           "training must distinguish four- from eight-connectivity")
    _check(any(info["unseeded"] > 0 for info in train_info),
           "training must show an unseeded region staying background")
    _check(demonstrated == frozenset().union(*(i["seed_colors"] for i in train_info)),
           "every training seed color must visibly fill")
    _check(len({p["input"].shape for p in puzzle["train"]}) >= 2,
           "training needs different canvas shapes")
    layouts: set[tuple] = set()
    for pair in puzzle["train"] + puzzle["test"]:
        key = (pair["input"].shape, (pair["input"] == wall).tobytes())
        _check(key not in layouts, "wall layout repeated in puzzle")
        layouts.add(key)
    _check(all(info["seed_count"] >= 1 for info in test_info),
           "test needs a visible source")


def _puzzles_equal(first: dict, second: dict) -> bool:
    return _puzzle_fingerprint(first) == _puzzle_fingerprint(second)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _check(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack diversity")
    _check(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack diversity")
    for seed, train, test in ((8001, 2, 1), (8002, 10, 1), (8003, 10, 4),
                              (9001, None, 1), (9002, None, 4),
                              (9003, 2, None), (9004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _check(len(puzzle["train"]) == train, "explicit train count ignored")
        if test is not None:
            _check(len(puzzle["test"]) == test, "explicit test count ignored")
    first = generate_puzzle(7001, 5, 2)
    _check(_puzzles_equal(first, generate_puzzle(7001, 5, 2)), "not deterministic")
    _check(not _puzzles_equal(first, generate_puzzle(7002, 5, 2)),
           "different seeds produced same puzzle")
    for index, arguments in enumerate((
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4},
    )):
        try:
            generate_puzzle(10000 + index, **arguments)
        except ValueError:
            continue
        raise AssertionError("illegal count combination accepted")
    infos = []
    shapes = set()
    layouts = set()
    signatures = set()
    puzzle_signatures = set()
    role_signatures = set()
    train_infos = []
    test_infos = []
    new_literal_tests = 0
    varied_test_palettes = 0
    reused_training_combinations = 0
    varied_train_seed_counts = 0
    larger_test_canvases = 0
    comparable_test_canvases = 0
    for puzzle in puzzles:
        puzzle_signatures.add(_puzzle_fingerprint(puzzle))
        bg, wall, _ = _infer_training_conventions(puzzle)
        role_signatures.add((bg, wall))
        local_train = []
        local_test = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp = pair["input"]
                shapes.add(inp.shape)
                layouts.add((inp.shape, (inp == wall).tobytes()))
                info = _analyze_pair(inp, pair["output"], bg, wall)
                infos.append(info)
                if section == "train":
                    train_infos.append(info)
                    local_train.append(info)
                else:
                    test_infos.append(info)
                    local_test.append(info)
                signatures.add((inp.shape, info["seed_count"], info["open_components"],
                                info["wall_components"], info["unseeded"],
                                tuple(sorted(info["seeded_sizes"]))))
                _check(inp.dtype == np.int8 and pair["output"].dtype == np.int8,
                       "expected compact integer grids")
        train_colors = frozenset().union(*(i["seed_colors"] for i in local_train))
        training_combinations = {i["seed_colors"] for i in local_train}
        new_literal_tests += sum(bool(i["seed_colors"] - train_colors) for i in local_test)
        varied_test_palettes += int(len({i["seed_colors"] for i in local_test}) > 1)
        reused_training_combinations += int(any(
            i["seed_colors"] in training_combinations for i in local_test))
        varied_train_seed_counts += int(len({i["seed_count"] for i in local_train}) > 1)
        train_max_area = max(p["input"].size for p in puzzle["train"])
        test_areas = [p["input"].size for p in puzzle["test"]]
        larger_test_canvases += int(max(test_areas) > train_max_area)
        comparable_test_canvases += int(min(test_areas) <= train_max_area)
    _check(len(puzzle_signatures) == 42, "puzzles repeat across seeds")
    _check(len(role_signatures) >= 25, "background and wall roles lack variation")
    _check(len(shapes) >= 20 and any(h != w for h, w in shapes),
           "canvas size diversity is too narrow")
    _check(min(min(h, w) for h, w in shapes) <= 5
           and max(max(h, w) for h, w in shapes) >= 20,
           "small and large canvases are both required")
    _check(len(layouts) >= int(0.95 * len(infos)), "wall layouts repeat too often")
    _check(len(signatures) >= 70, "component structure lacks diversity")
    _check({1, 2, 3, 4}.issubset({i["seed_count"] for i in infos}),
           "seed count variation is missing")
    _check(varied_train_seed_counts >= 25, "seed counts hardly vary within puzzles")
    _check(new_literal_tests >= 8, "unseen literal test seeds are missing")
    _check(varied_test_palettes >= 8, "multiple tests rarely vary their palettes")
    _check(reused_training_combinations >= 8,
           "tests should sometimes reuse a familiar seed combination")
    _check(larger_test_canvases >= 8 and comparable_test_canvases >= 8,
           "test difficulty regimes lack variation")
    _check(any(i["unseeded"] == 0 for i in infos)
           and any(i["unseeded"] >= 3 for i in infos),
           "unseeded region counts lack variation")
    _check(any(i["open_components"] >= 8 for i in infos)
           and any(i["wall_components"] >= 11 for i in infos),
           "irregular component layouts are missing")
    _check(any(2 <= size <= 3 for i in infos for size in i["seeded_sizes"]),
           "small seeded regions are missing")
    _check(any(size >= 30 for i in infos for size in i["seeded_sizes"]),
           "substantial seeded regions are missing")
    _check(any(i["interior_seeded"] for i in infos), "interior seeded regions missing")
    _check(any(i["boundary_seeded"] for i in infos), "boundary seeded regions missing")
    _check(any(i["full_wall_line"] for i in infos)
           and any(not i["full_wall_line"] for i in infos),
           "straight-cut and fragmented walls must both occur")
    _check(any(i["repeated_seed_color"] for i in infos),
           "repeated seed colors are missing")
    _check(sum(i["diagonal_contrast"] for i in train_infos) >= len(train_infos) // 2,
           "training seldom exposes diagonal contrast")
    _check(sum(i["diagonal_contrast"] for i in test_infos) >= len(test_infos) // 2,
           "tests seldom exercise diagonal contrast")
