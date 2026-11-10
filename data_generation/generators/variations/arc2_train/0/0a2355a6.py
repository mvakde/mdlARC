import numpy as np


_SOURCE_INPUT_ROWS = (
    (
        "00000000088800", "00888880080800", "00800080088800",
        "00888880080800", "00000000088800", "00000000080800",
        "00000000088800", "00000888000000", "00088808000000",
        "00080808000000", "00088808008880", "00000888008080",
        "00000000008880", "00000000000000",
    ),
    (
        "00000000000", "00888880000", "00808080000", "00888880000",
        "00808000888", "00808000808", "00888000888", "00000000000",
        "88800000000", "80800088880", "88888080080", "80008080080",
        "88888088880", "00000000000",
    ),
    (
        "00888000000000", "00808888888800", "00888080080800",
        "00008888888800", "00000000000000", "00000000000000",
        "08888000088880", "08008000080080", "08888000080080",
        "08008000888880", "08888000808000", "00000000888800",
        "00888000800800", "00808000888800", "00888000000000",
    ),
    (
        "00000008888", "00000008008", "00888008888", "00808000000",
        "00888088800", "00000080800", "00000088800", "00000080800",
        "00000088800",
    ),
    (
        "00000000000000000", "00888800000008880", "00800800888008080",
        "00888888808008880", "00008080808008080", "00008888888008080",
        "00000000000088880", "00000000000080080", "00000000000088880",
        "00888888000000000", "00800008000000000", "00800008008888888",
        "00888888008008008", "00000000008008008", "00000000008888888",
    ),
)

_SOURCE_INPUT_KEYS = {
    (len(rows), len(rows[0]), "".join(rows)) for rows in _SOURCE_INPUT_ROWS
}

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _check_count(name, value, lower, upper):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not lower <= int(value) <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")


def _resolve_counts(seed, num_train, num_test):
    _check_count("num_train", num_train, 2, 10)
    _check_count("num_test", num_test, 1, 4)
    seed_value = int(seed)

    if num_train is None and num_test is None:
        test_count = 1 + seed_value % 4
        train_count = 2 + seed_value % 9
        if train_count <= test_count:
            train_count = test_count + 1
    elif num_train is None:
        test_count = int(num_test)
        low = max(2, test_count + 1)
        train_count = low + seed_value % (11 - low)
    elif num_test is None:
        train_count = int(num_train)
        test_count = 1 + seed_value % min(4, train_count - 1)
    else:
        train_count = int(num_train)
        test_count = int(num_test)

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")
    return train_count, test_count


def _sample_puzzle_parameters(rng):
    if rng.random() < 0.3:
        labels = [int(x) for x in rng.choice(
            [1, 2, 3, 4, 5, 6, 7, 9], size=4, replace=False,
        )]
    else:
        labels = [int(x) for x in rng.permutation(10)[:4]]
    return {
        "labels": {
            hole_count: labels[hole_count - 1]
            for hole_count in range(1, 5)
        },
        "pair_colors": [color for color in range(10) if color not in labels],
        "simple_tendency": float(rng.beta(1.2, 1.2)),
    }


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    height, width = mask.shape

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue

            stack = [(row, col)]
            seen[row, col] = True
            component = []

            while stack:
                rr, cc = stack.pop()
                component.append((rr, cc))

                for dr, dc in _DIRS:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            result.append(component)

    return result


def _hole_areas(mask):
    wall = np.pad(
        np.asarray(mask, dtype=bool),
        1,
        constant_values=False,
    )
    height, width = wall.shape

    exterior = np.zeros(wall.shape, dtype=bool)
    stack = [(0, 0)]
    exterior[0, 0] = True

    while stack:
        row, col = stack.pop()

        for dr, dc in _DIRS:
            nr, nc = row + dr, col + dc
            if (
                0 <= nr < height
                and 0 <= nc < width
                and not wall[nr, nc]
                and not exterior[nr, nc]
            ):
                exterior[nr, nc] = True
                stack.append((nr, nc))

    visited = exterior.copy()
    areas = []

    for row in range(height):
        for col in range(width):
            if wall[row, col] or visited[row, col]:
                continue

            stack = [(row, col)]
            visited[row, col] = True
            area = 0

            while stack:
                rr, cc = stack.pop()
                area += 1

                for dr, dc in _DIRS:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not wall[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            areas.append(area)

    return sorted(areas)


def _normalize_cells(cells):
    min_row = min(row for row, _ in cells)
    min_col = min(col for _, col in cells)

    return tuple(
        sorted(
            (row - min_row, col - min_col)
            for row, col in cells
        )
    )


def _mask_from_cells(cells):
    normalized = _normalize_cells(cells)
    height = max(row for row, _ in normalized) + 1
    width = max(col for _, col in normalized) + 1
    mask = np.zeros((height, width), dtype=bool)
    for row, col in normalized:
        mask[row, col] = True
    return mask


def _ring(rng, count, style, simple):
    if style == "large_single":
        hole_height, hole_width = 2, 2
    elif style in ("small_single", "four_unit"):
        hole_height, hole_width = 1, 1
    elif simple and rng.random() < 0.8:
        hole_height, hole_width = 1, 1
    else:
        upper = 7 if count == 1 else 4 if count == 2 else 3
        hole_height = int(rng.integers(1, upper))
        hole_width = int(rng.integers(1, upper))
    thickness = 2 if style is None and not simple and rng.random() < 0.12 else 1
    ring = np.ones((hole_height + 2 * thickness,
                    hole_width + 2 * thickness), dtype=bool)
    ring[thickness:thickness + hole_height,
         thickness:thickness + hole_width] = False
    if (style is None and not simple and hole_height >= 2 and hole_width >= 2
            and rng.random() < 0.25):
        corner = (int(rng.choice([0, hole_height - 1])),
                  int(rng.choice([0, hole_width - 1])))
        ring[thickness + corner[0], thickness + corner[1]] = True
    return ring


def _transformed_mask(mask, rng):
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    return np.rot90(mask, int(rng.integers(0, 4)))


def _add_exterior_tabs(mask, rng):
    cells = set(zip(*np.nonzero(mask)))
    height, width = mask.shape
    for _ in range(int(rng.integers(0, 4))):
        side = int(rng.integers(0, 4))
        if side == 0:
            origins = [(0, col) for col in range(width) if mask[0, col]]
        elif side == 1:
            origins = [(height - 1, col) for col in range(width)
                       if mask[height - 1, col]]
        elif side == 2:
            origins = [(row, 0) for row in range(height) if mask[row, 0]]
        else:
            origins = [(row, width - 1) for row in range(height)
                       if mask[row, width - 1]]
        if not origins:
            continue
        row, col = origins[int(rng.integers(0, len(origins)))]
        direction = _DIRS[side]
        for length in range(1, int(rng.integers(2, 5))):
            cells.add((row + direction[0] * length,
                       col + direction[1] * length))
    return _mask_from_cells(cells)


def _make_object(rng, hole_count, style=None, simple=False):
    for _ in range(70):
        compact = hole_count == 4 and rng.random() < 0.7
        first = _ring(rng, hole_count, style, simple)
        walls = set(zip(*np.nonzero(first)))
        holes = set(zip(*np.nonzero(~first)))
        chambers = [(0, 0, *first.shape)]
        success = True
        for chamber_index in range(1, hole_count):
            attached = False
            for _ in range(35):
                ring = _ring(rng, hole_count, style, simple or compact)
                ring_height, ring_width = ring.shape
                if compact:
                    parent_index, side = ((0, 0), (0, 2), (1, 2))[
                        chamber_index - 1]
                else:
                    parent_index = int(rng.integers(0, len(chambers)))
                    side = int(rng.integers(0, 4))
                parent_row, parent_col, parent_height, parent_width = (
                    chambers[parent_index]
                )
                gap = 0 if simple or compact else int(rng.choice([0, 0, 0, 1, 2]))
                if side < 2:
                    offset = (0 if compact else int(rng.integers(
                        -ring_height + 2, parent_height - 1)))
                    top = parent_row + offset
                    left = (parent_col + parent_width - 1 + gap
                            if side == 0 else parent_col - ring_width + 1 - gap)
                    shared_row = int(rng.integers(
                        max(parent_row, top),
                        min(parent_row + parent_height, top + ring_height),
                    ))
                    bridge = {(shared_row, col) for col in range(
                        min(parent_col + parent_width - 1, left + ring_width - 1),
                        max(parent_col, left) + 1,
                    )} if gap > 0 else set()
                else:
                    offset = (0 if compact else int(rng.integers(
                        -ring_width + 2, parent_width - 1)))
                    left = parent_col + offset
                    top = (parent_row + parent_height - 1 + gap
                           if side == 2 else parent_row - ring_height + 1 - gap)
                    shared_col = int(rng.integers(
                        max(parent_col, left),
                        min(parent_col + parent_width, left + ring_width),
                    ))
                    bridge = {(row, shared_col) for row in range(
                        min(parent_row + parent_height - 1, top + ring_height - 1),
                        max(parent_row, top) + 1,
                    )} if gap > 0 else set()
                ring_walls = {(top + row, left + col)
                              for row, col in zip(*np.nonzero(ring))}
                ring_holes = {(top + row, left + col)
                              for row, col in zip(*np.nonzero(~ring))}
                if ring_holes & (walls | holes) or (ring_walls | bridge) & holes:
                    continue
                proposed = walls | ring_walls | bridge
                if len(_components(_mask_from_cells(proposed))) != 1:
                    continue
                if len(_hole_areas(_mask_from_cells(proposed))) != chamber_index + 1:
                    continue
                walls = proposed
                holes |= ring_holes
                chambers.append((top, left, ring_height, ring_width))
                attached = True
                break
            if not attached:
                success = False
                break
        if not success:
            continue
        mask = _mask_from_cells(walls)
        if not simple or rng.random() < 0.1:
            mask = _add_exterior_tabs(mask, rng)
        mask = _transformed_mask(mask, rng)
        if (max(mask.shape) <= 20 and len(_components(mask)) == 1
                and len(_hole_areas(mask)) == hole_count):
            return mask
    raise RuntimeError("could not construct an object with the requested topology")


def _pack_masks(rng, masks, height, width):
    if any(mask.shape[0] > height or mask.shape[1] > width
           for mask in masks):
        return None
    order = sorted(range(len(masks)), key=lambda index: masks[index].sum(),
                   reverse=True)
    for _ in range(10):
        occupied = np.zeros((height, width), dtype=bool)
        chosen = {}
        for index in order:
            mask = masks[index]
            mask_height, mask_width = mask.shape
            positions = [(row, col)
                         for row in range(height - mask_height + 1)
                         for col in range(width - mask_width + 1)]
            rng.shuffle(positions)
            blocked = occupied.copy()
            blocked[1:] |= occupied[:-1]
            blocked[:-1] |= occupied[1:]
            blocked[:, 1:] |= occupied[:, :-1]
            blocked[:, :-1] |= occupied[:, 1:]
            for row, col in positions:
                if any(
                    (row >= other_row and col >= other_col and
                     row + mask_height <= other_row + masks[other_index].shape[0]
                     and col + mask_width <= other_col + masks[other_index].shape[1])
                    or (other_row >= row and other_col >= col and
                        other_row + masks[other_index].shape[0] <= row + mask_height
                        and other_col + masks[other_index].shape[1] <= col + mask_width)
                    for other_index, (other_row, other_col) in chosen.items()
                ):
                    continue
                if np.any(blocked[row:row + mask_height,
                                  col:col + mask_width] & mask):
                    continue
                occupied[row:row + mask_height,
                         col:col + mask_width] |= mask
                chosen[index] = (row, col)
                break
            else:
                break
        if len(chosen) == len(masks):
            return [chosen[index] for index in range(len(masks))]
    return None


def _generate_pair(rng, regime, parameters, plan, palette,
                   preferred_shape=None):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown pair regime")
    background, input_ink = palette
    for _ in range(35):
        shuffled = list(plan)
        rng.shuffle(shuffled)
        simple = rng.random() < parameters["simple_tendency"]
        masks = [_make_object(rng, count, style, simple) for count, style in shuffled]
        total_area = sum(mask.size for mask in masks)
        aspect = float(np.exp(rng.uniform(-0.7, 0.7)))
        density = float(rng.uniform(1.4, 2.7))
        target_area = total_area * density
        height = max(max(mask.shape[0] for mask in masks),
                     int(np.sqrt(target_area / aspect)))
        width = max(max(mask.shape[1] for mask in masks),
                    int(np.sqrt(target_area * aspect)))
        if preferred_shape is not None:
            height = max(max(mask.shape[0] for mask in masks),
                         preferred_shape[0])
            width = max(max(mask.shape[1] for mask in masks),
                        preferred_shape[1])
        if rng.random() < 0.07:
            if rng.random() < 0.5:
                height = 30
                width = max(width, 14)
            else:
                width = 30
                height = max(height, 14)
        height, width = min(30, height), min(30, width)
        for _ in range(6):
            placements = _pack_masks(rng, masks, height, width)
            if placements is not None:
                break
            height = min(30, height + 2)
            width = min(30, width + 2)
        if placements is None:
            continue
        input_grid = np.full((height, width), background, dtype=np.int8)
        output_grid = input_grid.copy()
        for mask, (row, col), (count, _) in zip(masks, placements, shuffled):
            mask_height, mask_width = mask.shape
            input_grid[row:row + mask_height,
                       col:col + mask_width][mask] = input_ink
            output_grid[row:row + mask_height,
                        col:col + mask_width][mask] = parameters["labels"][count]
        if int(np.count_nonzero(input_grid == background)) <= int(
                np.count_nonzero(input_grid == input_ink)):
            continue
        return input_grid, output_grid
    raise RuntimeError(f"could not lay out a valid {regime} pair")


def _grid_value_key(grid):
    return (
        grid.shape,
        tuple(int(x) for x in grid.ravel()),
    )


def _source_key(grid):
    return (
        grid.shape[0],
        grid.shape[1],
        "".join(str(int(x)) for x in grid.ravel()),
    )


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    height, width = input_grid.shape
    input_colors, frequencies = np.unique(input_grid, return_counts=True)
    assert len(input_colors) == 2
    assert frequencies[0] != frequencies[1]
    background = int(input_colors[int(np.argmax(frequencies))])
    input_ink = int(input_colors[int(np.argmin(frequencies))])
    assert input_ink != background

    foreground = input_grid == input_ink

    assert np.array_equal(
        output_grid != background,
        foreground,
    )
    assert np.all(output_grid[~foreground] == background)

    component_data = []

    for component in _components(foreground):
        rows = [point[0] for point in component]
        cols = [point[1] for point in component]
        top, bottom = min(rows), max(rows)
        left, right = min(cols), max(cols)

        local_mask = np.zeros(
            (bottom - top + 1, right - left + 1),
            dtype=bool,
        )

        for row, col in component:
            local_mask[row - top, col - left] = True

        hole_areas = _hole_areas(local_mask)
        assert 1 <= len(hole_areas) <= 4
        labels = np.unique([
            output_grid[row, col]
            for row, col in component
        ])
        assert labels.size == 1

        component_data.append({
            "holes": len(hole_areas),
            "hole_areas": tuple(hole_areas),
            "label": int(labels[0]),
            "bbox": (
                top,
                left,
                bottom - top + 1,
                right - left + 1,
            ),
            "mask_key": (
                local_mask.shape,
                tuple(
                    bool(x)
                    for x in local_mask.ravel()
                ),
            ),
        })

    assert component_data

    return {
        "background": background,
        "input_ink": input_ink,
        "components": component_data,
        "height": height,
        "width": width,
    }


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(
        seed,
        num_train,
        num_test,
    )
    parameters = _sample_puzzle_parameters(rng)
    count_choices = np.array([1, 2, 3, 4, 5, 6])
    probabilities = np.array([0.12, 0.23, 0.28, 0.22, 0.11, 0.04])
    object_counts = [int(rng.choice(count_choices, p=probabilities))
                     for _ in range(train_count)]
    while sum(object_counts) < 5:
        object_counts[int(rng.integers(0, train_count))] += 1
    train_plans = [[(int(rng.integers(1, 5)), None)
                    for _ in range(count)] for count in object_counts]
    slots = [(pair_index, slot) for pair_index, count in
             enumerate(object_counts) for slot in range(count)]
    rng.shuffle(slots)
    demonstrations = [(1, "large_single"), (1, "small_single"),
                      (2, None), (3, None), (4, "four_unit")]
    rng.shuffle(demonstrations)
    for (pair_index, slot), demonstration in zip(slots, demonstrations):
        train_plans[pair_index][slot] = demonstration

    pair_seeds = rng.integers(
        0,
        np.iinfo(np.uint64).max,
        size=train_count + test_count,
        dtype=np.uint64,
    )

    puzzle = {
        "train": [],
        "test": [],
    }
    seen_inputs = set()
    previous_palette = None

    for pair_index in range(train_count + test_count):
        pair_rng = np.random.default_rng(
            int(pair_seeds[pair_index])
        )

        if pair_index < train_count:
            split = "train"
            regime = "source_train"
            plan = train_plans[pair_index]
            preferred_shape = None
        else:
            split = "test"
            regime = "target_test"
            count = int(rng.choice(count_choices, p=probabilities))
            plan = [(int(pair_rng.integers(1, 5)), None)
                    for _ in range(count)]
            reference_shape = puzzle["train"][int(pair_rng.integers(
                0, train_count))]["input"].shape
            factor = float(pair_rng.choice([0.7, 0.85, 1.0, 1.0,
                                            1.15, 1.3]))
            preferred_shape = (max(3, min(30, round(reference_shape[0] * factor))),
                               max(3, min(30, round(reference_shape[1] * factor))))

        available = parameters["pair_colors"]
        palettes = [(background, ink) for background in available
                    for ink in available if background != ink]
        if pair_index == 0 and (0, 8) in palettes and rng.random() < 0.5:
            palette = (0, 8)
        else:
            possibilities = [candidate for candidate in palettes
                             if previous_palette is None or
                             (candidate[0] != previous_palette[0] and
                              candidate[1] != previous_palette[1])]
            palette = possibilities[int(rng.integers(0, len(possibilities)))]
        previous_palette = palette

        for _ in range(30):
            input_grid, output_grid = _generate_pair(
                pair_rng,
                regime,
                parameters,
                plan,
                palette,
                preferred_shape,
            )
            key = _grid_value_key(input_grid)

            if key not in seen_inputs:
                seen_inputs.add(key)
                puzzle[split].append({
                    "input": input_grid,
                    "output": output_grid,
                })
                break
        else:
            raise RuntimeError(
                "could not generate a unique input grid"
            )

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    input_keys = set()
    pair_keys = set()
    analyses = {
        "train": [],
        "test": [],
    }

    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict)
            assert set(pair.keys()) == {"input", "output"}

            for name in ("input", "output"):
                grid = pair[name]

                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))

            input_key = _grid_value_key(pair["input"])
            pair_key = (
                input_key,
                _grid_value_key(pair["output"]),
            )

            assert input_key not in input_keys
            assert pair_key not in pair_keys

            input_keys.add(input_key)
            pair_keys.add(pair_key)

            assert (
                _source_key(pair["input"])
                not in _SOURCE_INPUT_KEYS
            )

            analyses[split].append(
                _analyze_pair(
                    pair["input"],
                    pair["output"],
                )
            )

    learned_mapping = {}
    train_areas = {count: [] for count in range(1, 5)}
    backgrounds = set()
    inks = set()

    for analysis in analyses["train"]:
        backgrounds.add(analysis["background"])
        inks.add(analysis["input_ink"])

        for component in analysis["components"]:
            holes = component["holes"]
            label = component["label"]
            train_areas[holes].append(component["hole_areas"])

            if holes in learned_mapping:
                assert learned_mapping[holes] == label
            else:
                learned_mapping[holes] = label

    assert set(learned_mapping) == {1, 2, 3, 4}
    assert (1,) in train_areas[1]
    assert (4,) in train_areas[1]
    assert (1, 1, 1, 1) in train_areas[4]
    assert len(set(learned_mapping.values())) == 4
    assert len(backgrounds) >= 2
    assert len(inks) >= 2

    for split in ("train", "test"):
        for analysis in analyses[split]:
            assert analysis["background"] not in learned_mapping.values()
            assert analysis["input_ink"] not in learned_mapping.values()
            for component in analysis["components"]:
                assert component["label"] == learned_mapping[component["holes"]]


def _puzzles_equal(first, second):
    if set(first) != set(second):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for pair_a, pair_b in zip(
            first[split],
            second[split],
        ):
            if not np.array_equal(
                pair_a["input"],
                pair_b["input"],
            ):
                return False

            if not np.array_equal(
                pair_a["output"],
                pair_b["output"],
            ):
                return False

    return True


def _predict_from_demonstrations(train_pairs, test_input):
    mapping = {}
    for pair in train_pairs:
        analysis = _analyze_pair(pair["input"], pair["output"])
        for component in analysis["components"]:
            count = component["holes"]
            label = component["label"]
            if count in mapping:
                assert mapping[count] == label
            mapping[count] = label
    colors, frequencies = np.unique(test_input, return_counts=True)
    assert len(colors) == 2
    background = int(colors[int(np.argmax(frequencies))])
    foreground = test_input != background
    predicted = np.full(test_input.shape, background, dtype=np.int8)
    for component in _components(foreground):
        top = min(row for row, _ in component)
        left = min(col for _, col in component)
        bottom = max(row for row, _ in component)
        right = max(col for _, col in component)
        mask = np.zeros((bottom - top + 1, right - left + 1), dtype=bool)
        for row, col in component:
            mask[row - top, col - left] = True
        label = mapping[len(_hole_areas(mask))]
        for row, col in component:
            predicted[row, col] = label
    return predicted


def self_test() -> None:
    automatic = generate_puzzle(314159)
    validate_generated_puzzle(automatic)

    explicit = generate_puzzle(
        271828,
        num_train=6,
        num_test=3,
    )
    validate_generated_puzzle(explicit)

    assert len(explicit["train"]) == 6
    assert len(explicit["test"]) == 3

    assert _puzzles_equal(
        explicit,
        generate_puzzle(271828, 6, 3),
    )

    assert not _puzzles_equal(
        generate_puzzle(9001, 5, 2),
        generate_puzzle(9002, 5, 2),
    )

    for train_count, test_count in (
        (2, 1),
        (10, 1),
        (10, 4),
    ):
        puzzle = generate_puzzle(
            1000 + train_count * 10 + test_count,
            train_count,
            test_count,
        )
        validate_generated_puzzle(puzzle)

        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count

    partial_cases = (
        (1201, None, 1),
        (1202, None, 4),
        (1203, 2, None),
        (1204, 10, None),
    )

    for seed, train_count, test_count in partial_cases:
        puzzle = generate_puzzle(
            seed,
            train_count,
            test_count,
        )
        validate_generated_puzzle(puzzle)

        if train_count is not None:
            assert len(puzzle["train"]) == train_count

        if test_count is not None:
            assert len(puzzle["test"]) == test_count

    illegal_cases = (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4},
        {"num_train": 3.0, "num_test": 1},
    )

    for arguments in illegal_cases:
        try:
            generate_puzzle(77, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"illegal counts were accepted: {arguments}"
            )

    train_counts = set()
    test_counts = set()
    palette_codebooks = set()
    dimensions = set()
    input_fingerprints = set()
    layout_fingerprints = set()
    mask_fingerprints = set()
    total_pairs = 0
    review_metrics = {name: 0 for name in (
        "train_all_four", "train_repeat_high", "test_repeat_high",
        "test_subset", "edge_object", "overlapping_bounds",
        "unequal_cavities", "hole_area_eight", "thirty_dimension",
        "small_canvas", "larger_test", "smaller_test", "similar_test",
        "four_not_widest", "ordinary_palette",
    )}
    train_object_counts = set()
    test_object_counts = set()

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)

        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))

        first_analysis = _analyze_pair(
            puzzle["train"][0]["input"],
            puzzle["train"][0]["output"],
        )

        mapping = {}

        for pair in puzzle["train"]:
            analysis = _analyze_pair(
                pair["input"],
                pair["output"],
            )

            for component in analysis["components"]:
                mapping[component["holes"]] = component["label"]

        palette_codebooks.add((
            first_analysis["background"],
            first_analysis["input_ink"],
            tuple(
                mapping[count]
                for count in range(1, 5)
            ),
        ))
        train_analyses = [_analyze_pair(pair["input"], pair["output"])
                          for pair in puzzle["train"]]
        test_analyses = [_analyze_pair(pair["input"], pair["output"])
                         for pair in puzzle["test"]]
        train_areas = [analysis["height"] * analysis["width"]
                       for analysis in train_analyses]
        test_areas = [analysis["height"] * analysis["width"]
                      for analysis in test_analyses]
        review_metrics["larger_test"] += any(area > max(train_areas)
                                              for area in test_areas)
        review_metrics["smaller_test"] += any(area < min(train_areas)
                                               for area in test_areas)
        review_metrics["similar_test"] += any(
            abs(area - train_area) <= train_area * 0.1
            for area in test_areas for train_area in train_areas
        )
        for pair in puzzle["test"]:
            assert np.array_equal(
                _predict_from_demonstrations(puzzle["train"], pair["input"].copy()),
                pair["output"],
            )

        for split in ("train", "test"):
            for pair in puzzle[split]:
                total_pairs += 1

                analysis = _analyze_pair(
                    pair["input"],
                    pair["output"],
                )
                components = analysis["components"]
                counts = [component["holes"] for component in components]
                if split == "train":
                    train_object_counts.add(len(components))
                    review_metrics["train_all_four"] += set(counts) == {1, 2, 3, 4}
                    review_metrics["train_repeat_high"] += (
                        counts.count(3) >= 2 or counts.count(4) >= 2
                    )
                else:
                    test_object_counts.add(len(components))
                    review_metrics["test_subset"] += len(set(counts)) < 4
                    review_metrics["test_repeat_high"] += (
                        counts.count(3) >= 2 or counts.count(4) >= 2
                    )
                    if 4 in counts:
                        widest = max(component["bbox"][3]
                                     for component in components)
                        review_metrics["four_not_widest"] += all(
                            component["bbox"][3] < widest
                            for component in components if component["holes"] == 4
                        )
                review_metrics["ordinary_palette"] += (
                    analysis["background"], analysis["input_ink"]
                ) == (0, 8)
                review_metrics["thirty_dimension"] += (
                    30 in pair["input"].shape
                )
                review_metrics["small_canvas"] += (
                    analysis["height"] <= 10 and analysis["width"] <= 10
                )
                review_metrics["edge_object"] += any(
                    component["bbox"][0] == 0 or component["bbox"][1] == 0
                    or component["bbox"][0] + component["bbox"][2] == analysis["height"]
                    or component["bbox"][1] + component["bbox"][3] == analysis["width"]
                    for component in components
                )
                review_metrics["overlapping_bounds"] += any(
                    not (first["bbox"][0] + first["bbox"][2] <= second["bbox"][0]
                         or second["bbox"][0] + second["bbox"][2] <= first["bbox"][0]
                         or first["bbox"][1] + first["bbox"][3] <= second["bbox"][1]
                         or second["bbox"][1] + second["bbox"][3] <= first["bbox"][1])
                    for index, first in enumerate(components)
                    for second in components[index + 1:]
                )
                for component in components:
                    mask_fingerprints.add(component["mask_key"])
                    review_metrics["unequal_cavities"] += (
                        len(set(component["hole_areas"])) > 1
                    )
                    review_metrics["hole_area_eight"] += (
                        max(component["hole_areas"]) >= 8
                    )

                dimensions.add(pair["input"].shape)
                input_fingerprints.add(
                    _grid_value_key(pair["input"])
                )

                layout_fingerprints.add((
                    pair["input"].shape,
                    tuple(sorted(
                        (
                            component["holes"],
                            component["bbox"],
                            component["mask_key"],
                        )
                        for component
                        in analysis["components"]
                    )),
                ))

                assert (
                    _source_key(pair["input"])
                    not in _SOURCE_INPUT_KEYS
                )
                assert not np.array_equal(
                    pair["input"],
                    pair["output"],
                )

    assert len(train_counts) >= 4
    assert len(test_counts) >= 3
    assert len(palette_codebooks) >= 30
    assert len(dimensions) >= 25
    assert len(mask_fingerprints) >= 300
    assert train_object_counts == {1, 2, 3, 4, 5, 6}
    assert test_object_counts == {1, 2, 3, 4, 5, 6}
    for name in review_metrics:
        assert review_metrics[name] > 0, name
    assert (
        len(input_fingerprints)
        >= int(total_pairs * 0.98)
    )
    assert (
        len(layout_fingerprints)
        >= int(total_pairs * 0.90)
    )
