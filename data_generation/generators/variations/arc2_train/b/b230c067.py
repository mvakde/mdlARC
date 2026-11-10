import numpy as np
from collections import Counter


_SOURCE_PAIR_SIGNATURES = {
    (
        "0000000000/0000008888/0888800088/0008800088/0008880000/"
        "0000000000/0008888000/0000088000/0000088800/0000000000",
        "0000000000/0000002222/0111100022/0001100022/0001110000/"
        "0000000000/0001111000/0000011000/0000011100/0000000000",
    ),
    (
        "0000000000/0000000888/0888800808/0800800888/0888800000/"
        "0000000000/0000000000/0000088880/0000080080/0000088880",
        "0000000000/0000000222/0111100202/0100100222/0111100000/"
        "0000000000/0000000000/0000011110/0000010010/0000011110",
    ),
    (
        "0000000000/0888000000/0808000000/0000000000/0000000000/"
        "8800000000/8808880000/0008080000/0000000000/0000000000",
        "0000000000/0111000000/0101000000/0000000000/0000000000/"
        "2200000000/2201110000/0001010000/0000000000/0000000000",
    ),
    (
        "0000000000/0880008880/0080000080/0800000800/0888008888/"
        "0000000000/0008880000/0000080000/0000800000/0008888000",
        "0000000000/0220001110/0020000010/0200000100/0222001111/"
        "0000000000/0001110000/0000010000/0000100000/0001111000",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _seed_value(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a non-negative integer")
    value = int(seed)
    if value < 0:
        raise ValueError("seed must be a non-negative integer")
    return value


def _optional_count(
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
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _optional_count(num_train, "num_train", 2, 10)
    test_count = _optional_count(num_test, "num_test", 1, 4)

    if train_count is None and test_count is None:
        # This schedule guarantees broad count variation for consecutive seeds.
        test_count = 1 + seed % 4
        lowest_train = max(2, test_count + 1)
        train_count = lowest_train + (
            (seed // 4) % (11 - lowest_train)
        )
    elif train_count is None:
        lowest_train = max(2, test_count + 1)
        train_count = lowest_train + seed % (11 - lowest_train)
    elif test_count is None:
        largest_test = min(4, train_count - 1)
        test_count = 1 + seed % largest_test

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    colors = [int(x) for x in rng.choice(10, size=2, replace=False)]
    return {
        "repeated": colors[0],
        "unique": colors[1],
    }


def _key_from_mask(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    occupied = np.argwhere(mask != 0)
    if occupied.size == 0:
        raise ValueError("a shape mask cannot be empty")

    r0, c0 = occupied.min(axis=0)
    r1, c1 = occupied.max(axis=0)
    cropped = (mask[r0 : r1 + 1, c0 : c1 + 1] != 0).astype(
        np.int8
    )
    return tuple(
        tuple(int(value) for value in row)
        for row in cropped.tolist()
    )


def _mask_from_key(
    key: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    return np.asarray(key, dtype=np.int8)


def _rotated_key(
    key: tuple[tuple[int, ...], ...],
    quarter_turns: int,
) -> tuple[tuple[int, ...], ...]:
    rotated = np.rot90(
        _mask_from_key(key),
        k=quarter_turns,
    )
    return _key_from_mask(rotated)


def _nontrivial_rotations(
    key: tuple[tuple[int, ...], ...],
) -> list[tuple[tuple[int, ...], ...]]:
    variants = []
    for turns in (1, 2, 3):
        rotated = _rotated_key(key, turns)
        if rotated != key and rotated not in variants:
            variants.append(rotated)
    return variants


def _shape_key_from_cells(
    cells: set[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)

    mask = np.zeros(
        (max_r - min_r + 1, max_c - min_c + 1),
        dtype=np.int8,
    )
    for r, c in cells:
        mask[r - min_r, c - min_c] = 1
    return _key_from_mask(mask)


def _random_shape_key(
    rng: np.random.Generator,
    forbidden: set[tuple[tuple[int, ...], ...]] | None = None,
    require_rotation_change: bool = False,
    compact: bool = False,
) -> tuple[tuple[int, ...], ...]:
    forbidden_keys = set() if forbidden is None else set(forbidden)
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    for _ in range(500):
        family = int(rng.integers(4))
        if family == 0:
            length = int(rng.integers(3, 7 if compact else 10))
            mask = np.ones((1, length), dtype=np.int8)
            if bool(rng.integers(2)):
                mask = mask.T
            key = _key_from_mask(mask)
        elif family == 1:
            height = int(rng.integers(3, 5 if compact else 7))
            width = int(rng.integers(3, 5 if compact else 7))
            mask = np.ones((height, width), dtype=np.int8)
            mask[1:-1, 1:-1] = 0
            key = _key_from_mask(mask)
        else:
            target_size = int(rng.integers(3, 8 if compact else 16))
            cells = {(0, 0)}
            for _ in range(250):
                if len(cells) == target_size:
                    break
                ordered_cells = sorted(cells)
                base_r, base_c = ordered_cells[int(rng.integers(len(ordered_cells)))]
                dr, dc = directions[int(rng.integers(4))]
                cells.add((base_r + dr, base_c + dc))
            if len(cells) != target_size:
                continue
            key = _shape_key_from_cells(cells)
        height = len(key)
        width = len(key[0])
        if height > (5 if compact else 8) or width > (5 if compact else 8):
            continue
        if key in forbidden_keys:
            continue
        if require_rotation_change and not _nontrivial_rotations(key):
            continue
        return key

    raise RuntimeError("unable to construct a suitable connected shape")


def _same_box_perturbations(key: tuple) -> list[tuple]:
    mask = _mask_from_key(key)
    height, width = mask.shape
    results = []
    occupied = [tuple(map(int, p)) for p in np.argwhere(mask)]
    empty = [tuple(map(int, p)) for p in np.argwhere(mask == 0)]
    for remove in occupied:
        for add in empty:
            changed = mask.copy()
            changed[remove] = 0
            changed[add] = 1
            if _key_from_mask(changed) != tuple(tuple(int(v) for v in row) for row in changed):
                continue
            cells = _mask_cells(changed)
            seen = {next(iter(cells))}
            stack = list(seen)
            while stack:
                r, c = stack.pop()
                for neighbor in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    if neighbor in cells and neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            if seen == cells:
                candidate = _key_from_mask(changed)
                if candidate != key and candidate not in results:
                    results.append(candidate)
    return results


def _near_match(key: tuple, kind: str, rng: np.random.Generator) -> tuple | None:
    if kind == "rotation":
        choices = _nontrivial_rotations(key)
    elif kind == "reflection":
        mask = _mask_from_key(key)
        choices = [_key_from_mask(np.fliplr(mask)), _key_from_mask(np.flipud(mask))]
        choices = [item for item in choices if item != key]
    else:
        choices = _same_box_perturbations(key)
        mask = _mask_from_key(key)
        dihedral = {_rotated_key(key, turns) for turns in (1, 2, 3)}
        flipped = np.fliplr(mask)
        dihedral.update(_key_from_mask(np.rot90(flipped, turns))
                        for turns in (0, 1, 2, 3))
        choices = [item for item in choices if item not in dihedral]
    if not choices:
        return None
    return choices[int(rng.integers(len(choices)))]


def _mask_cells(mask: np.ndarray) -> set[tuple[int, int]]:
    return {
        (int(r), int(c))
        for r, c in np.argwhere(mask != 0)
    }


def _compatible_placement(
    candidate: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    forbid_diagonal_contact: bool,
) -> bool:
    if candidate & occupied:
        return False

    for r, c in candidate:
        if (
            (r - 1, c) in occupied
            or (r + 1, c) in occupied
            or (r, c - 1) in occupied
            or (r, c + 1) in occupied
        ):
            return False

        if forbid_diagonal_contact and (
            (r - 1, c - 1) in occupied
            or (r - 1, c + 1) in occupied
            or (r + 1, c - 1) in occupied
            or (r + 1, c + 1) in occupied
        ):
            return False

    return True


def _frequency_pattern(rng: np.random.Generator, count: int) -> list[int]:
    for _ in range(100):
        groups = int(rng.integers(1, min(3, (count - 1) // 2) + 1))
        singles = int(rng.integers(1, min(3, count - 2 * groups) + 1))
        remainder = count - 2 * groups - singles
        if remainder > 2 * groups:
            continue
        sizes = [2] * groups
        while remainder:
            choices = [i for i, size in enumerate(sizes) if size < 4]
            index = choices[int(rng.integers(len(choices)))]
            sizes[index] += 1
            remainder -= 1
        return sizes + [1] * singles
    raise RuntimeError("unable to choose object frequencies")


def _shape_set(
    rng: np.random.Generator, pattern: list[int], near_kind: str | None,
    compact: bool,
) -> list[tuple]:
    for _ in range(100):
        first = _random_shape_key(rng, compact=compact)
        near = _near_match(first, near_kind, rng) if near_kind else None
        if near_kind and near is None:
            continue
        keys = [first]
        used = {first}
        if near is not None:
            used.add(near)
        for _ in pattern[1:]:
            key = _random_shape_key(rng, forbidden=used, compact=compact)
            used.add(key)
            keys.append(key)
        if near is not None:
            singleton_index = next(i for i, size in enumerate(pattern) if size == 1)
            keys[singleton_index] = near
        result = [key for key, size in zip(keys, pattern) for _ in range(size)]
        return [result[int(i)] for i in rng.permutation(len(result))]
    raise RuntimeError("unable to construct distinct shape classes")


def _diagonal_neighbors(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells
            for dr, dc in ((-1,-1),(-1,1),(1,-1),(1,1))}


def _place_varied(
    rng: np.random.Generator, masks: list[np.ndarray], contact_target: int,
    compact: bool,
) -> tuple[int, int, list[set[tuple[int, int]]]]:
    count = len(masks)
    max_h = max(mask.shape[0] for mask in masks)
    max_w = max(mask.shape[1] for mask in masks)
    area = sum(int(mask.sum()) for mask in masks)
    order = list(range(count))
    contact_children = set(int(v) for v in rng.choice(
        np.arange(1, count), size=min(contact_target, count - 1), replace=False
    )) if contact_target else set()
    parents = {child: int(rng.integers(child)) for child in contact_children}

    for attempt in range(240):
        if compact and attempt < 140:
            low_h, high_h = max(9, max_h), 14
            low_w, high_w = max(9, max_w), 14
        elif attempt < 190:
            low_h, high_h = max(12, max_h), 24
            low_w, high_w = max(12, max_w), 24
        else:
            low_h, high_h = max(18, max_h), 30
            low_w, high_w = max(18, max_w), 30
        if not compact and attempt < 140:
            style = int(rng.integers(3))
            if style == 0:
                low_h, high_h = max(12, max_h), 23
                low_w, high_w = max(12, max_w), 23
            elif style == 1:
                low_h, high_h = max(17, max_h), 30
                low_w, high_w = max(17, max_w), 30
            else:
                low_h, high_h = max(10, max_h), 19
                low_w, high_w = max(10, max_w), 19
        height = int(rng.integers(low_h, high_h + 1))
        width = int(rng.integers(low_w, high_w + 1))
        if height * width < area * (2.4 if compact else 2.1):
            continue
        placements: list[set[tuple[int, int]]] = []
        occupied: set[tuple[int, int]] = set()
        failed = False
        for index in order:
            local = _mask_cells(masks[index])
            mask_h, mask_w = masks[index].shape
            positions = []
            if index in parents:
                parent = placements[parents[index]]
                offsets = {(pr + dr - lr, pc + dc - lc)
                           for pr, pc in parent for lr, lc in local
                           for dr, dc in ((-1,-1),(-1,1),(1,-1),(1,1))}
                for top, left in offsets:
                    if 0 <= top <= height - mask_h and 0 <= left <= width - mask_w:
                        candidate = {(r + top, c + left) for r, c in local}
                        if _compatible_placement(candidate, occupied, False) and \
                                candidate & _diagonal_neighbors(parent):
                            positions.append(candidate)
                if positions:
                    chosen = positions[int(rng.integers(len(positions)))]
                else:
                    failed = True
                    break
            else:
                chosen = None
                for _ in range(300):
                    top = int(rng.integers(height - mask_h + 1))
                    left = int(rng.integers(width - mask_w + 1))
                    candidate = {(r + top, c + left) for r, c in local}
                    if _compatible_placement(candidate, occupied, True):
                        chosen = candidate
                        break
                if chosen is None:
                    failed = True
                    break
            placements.append(chosen)
            occupied.update(chosen)
        if not failed:
            return height, width, placements
    raise RuntimeError("unable to place all objects")


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    puzzle_parameters: dict,
    object_count: int,
    near_kind: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError(f"unknown regime: {regime}")
    compact = object_count <= 5 and int(rng.integers(3)) != 0
    pattern = _frequency_pattern(rng, object_count)
    shape_keys = _shape_set(rng, pattern, near_kind, compact)
    masks = [_mask_from_key(key) for key in shape_keys]
    contact_target = int(rng.choice([0, 1, 2, 3], p=[.27, .28, .28, .17]))
    height, width, placements = _place_varied(
        rng, masks, contact_target, compact,
    )
    available = [color for color in range(10) if color not in (
        puzzle_parameters["repeated"], puzzle_parameters["unique"]
    )]
    background, foreground = [int(value) for value in rng.choice(
        available, size=2, replace=False
    )]
    input_grid = np.full((height, width), background, dtype=np.int8)
    output_grid = np.full((height, width), background, dtype=np.int8)
    frequencies = Counter(shape_keys)
    for key, cells in zip(shape_keys, placements):
        output_color = (puzzle_parameters["repeated"] if frequencies[key] >= 2
                        else puzzle_parameters["unique"])
        for r, c in cells:
            input_grid[r, c] = foreground
            output_grid[r, c] = output_color
    return input_grid, output_grid


def _grid_value_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(int(value) for value in grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _row_signature(grid: np.ndarray) -> str:
    return "/".join(
        "".join(str(int(value)) for value in row)
        for row in grid.tolist()
    )


def _pair_source_signature(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple[str, str]:
    return _row_signature(input_grid), _row_signature(output_grid)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed_number = _seed_value(seed)
    train_count, test_count = _resolve_counts(
        seed_number,
        num_train,
        num_test,
    )

    rng = np.random.default_rng(seed_number)
    puzzle_parameters = _sample_puzzle_parameters(rng)

    train_sizes = [int(rng.integers(3, 10)) for _ in range(train_count)]
    if len(set(train_sizes)) == 1:
        train_sizes[-1] = 3 + (train_sizes[-1] - 2) % 7
    test_sizes = [int(rng.integers(3, 11)) for _ in range(test_count)]
    orientation_index = int(rng.integers(train_count))
    perturb_index = (orientation_index + int(rng.integers(1, train_count))) % train_count
    train_near = [rng.choice([None, "rotation", "reflection", "perturb"])
                  for _ in range(train_count)]
    train_near[orientation_index] = "rotation"
    train_near[perturb_index] = "perturb"
    test_near = [rng.choice([None, "rotation", "reflection", "perturb"])
                 for _ in range(test_count)]

    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    seen_pairs = set()

    jobs = [
        ("train", "source_train", train_sizes[index], train_near[index])
        for index in range(train_count)
    ] + [
        ("test", "target_test", test_sizes[index], test_near[index])
        for index in range(test_count)
    ]

    for split, regime, object_count, near_kind in jobs:
        accepted_pair = None

        for _ in range(300):
            child_seed = int(
                rng.integers(0, 2**63 - 1, dtype=np.int64)
            )
            pair_rng = np.random.default_rng(child_seed)

            try:
                input_grid, output_grid = _generate_pair(
                    pair_rng,
                    regime,
                    puzzle_parameters,
                    object_count,
                    near_kind,
                )
            except RuntimeError:
                continue

            input_signature = _grid_value_signature(input_grid)
            pair_signature = (
                input_signature,
                _grid_value_signature(output_grid),
            )

            if input_signature in seen_inputs:
                continue
            if pair_signature in seen_pairs:
                continue
            if (
                _pair_source_signature(input_grid, output_grid)
                in _SOURCE_PAIR_SIGNATURES
            ):
                continue
            if len(seen_inputs) == len(jobs) - 1 and all(
                _background_color(pair["input"]) == _background_color(input_grid)
                for pair in puzzle["train"] + puzzle["test"]
            ):
                continue

            accepted_pair = {
                "input": input_grid,
                "output": output_grid,
            }
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
            break

        if accepted_pair is None:
            raise RuntimeError("unable to generate a distinct valid pair")

        puzzle[split].append(accepted_pair)

    validate_generated_puzzle(puzzle)
    return puzzle


def _extract_components(
    grid: np.ndarray,
    background: int,
) -> list[set[tuple[int, int]]]:
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    components = []

    for start_r in range(height):
        for start_c in range(width):
            if visited[start_r, start_c]:
                continue
            if int(grid[start_r, start_c]) == background:
                continue

            component = set()
            stack = [(start_r, start_c)]
            visited[start_r, start_c] = True

            while stack:
                r, c = stack.pop()
                component.add((r, c))

                for nr, nc in (
                    (r - 1, c),
                    (r + 1, c),
                    (r, c - 1),
                    (r, c + 1),
                ):
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if visited[nr, nc]:
                        continue
                    if int(grid[nr, nc]) == background:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            components.append(component)

    return components


def _has_diagonal_contact(
    components: list[set[tuple[int, int]]],
) -> bool:
    owner = {}
    for index, component in enumerate(components):
        for cell in component:
            owner[cell] = index

    for (r, c), index in owner.items():
        for neighbor in (
            (r - 1, c - 1),
            (r - 1, c + 1),
            (r + 1, c - 1),
            (r + 1, c + 1),
        ):
            if neighbor in owner and owner[neighbor] != index:
                return True

    return False


def _contact_edges(components: list[set[tuple[int, int]]]) -> set[tuple[int, int]]:
    owner = {cell: index for index, component in enumerate(components)
             for cell in component}
    edges = set()
    for (r, c), first in owner.items():
        for neighbor in ((r-1,c-1),(r-1,c+1),(r+1,c-1),(r+1,c+1)):
            second = owner.get(neighbor)
            if second is not None and first != second:
                edges.add(tuple(sorted((first, second))))
    return edges


def _contact_count(components: list[set[tuple[int, int]]]) -> int:
    return len(_contact_edges(components))


def _has_rotated_singleton(
    frequencies: Counter,
) -> bool:
    for key, count in frequencies.items():
        if count < 2:
            continue
        for turns in (1, 2, 3):
            rotated = _rotated_key(key, turns)
            if rotated != key and frequencies.get(rotated, 0) == 1:
                return True
    return False


def _has_reflected_singleton(frequencies: Counter) -> bool:
    for key, count in frequencies.items():
        if count < 2:
            continue
        flipped = np.fliplr(_mask_from_key(key))
        for turns in (0, 1, 2, 3):
            reflected = _key_from_mask(np.rot90(flipped, turns))
            if reflected != key and frequencies.get(reflected, 0) == 1:
                return True
    return False


def _has_perturbed_singleton(frequencies: Counter) -> bool:
    for repeated, count in frequencies.items():
        if count < 2:
            continue
        variants = {repeated, *(_rotated_key(repeated, turns) for turns in (1, 2, 3))}
        mask = _mask_from_key(repeated)
        flipped = np.fliplr(mask)
        variants.update(_key_from_mask(np.rot90(flipped, turns))
                        for turns in (0, 1, 2, 3))
        for candidate, other_count in frequencies.items():
            if other_count != 1 or candidate in variants:
                continue
            if (len(candidate), len(candidate[0]), sum(map(sum, candidate))) == (
                len(repeated), len(repeated[0]), sum(map(sum, repeated))
            ):
                return True
    return False


def _inspect_pair(
    pair: dict,
) -> dict:
    input_grid = pair["input"]
    output_grid = pair["output"]

    _require(
        input_grid.shape == output_grid.shape,
        "input and output dimensions must match",
    )

    values, counts = np.unique(input_grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    _require(int(counts.max()) > input_grid.size // 2,
             "background must be the majority input color")

    input_colors = {int(value) for value in np.unique(input_grid)}
    _require(
        len(input_colors) == 2 and background in input_colors,
        "each input must contain exactly background and foreground",
    )
    foreground = next(
        color for color in input_colors if color != background
    )

    components = _extract_components(input_grid, background)
    shape_keys = [
        _shape_key_from_cells(component)
        for component in components
    ]
    frequencies = Counter(shape_keys)

    repeated_components = [
        component
        for component, key in zip(components, shape_keys)
        if frequencies[key] >= 2
    ]
    singleton_components = [
        component
        for component, key in zip(components, shape_keys)
        if frequencies[key] == 1
    ]

    _require(
        repeated_components and singleton_components,
        "each pair must contain both repeated and unique objects",
    )

    repeated_values = {
        int(output_grid[r, c])
        for component in repeated_components
        for r, c in component
    }
    unique_values = {
        int(output_grid[r, c])
        for component in singleton_components
        for r, c in component
    }

    _require(
        len(repeated_values) == 1,
        "all repeated-object cells must use one output color",
    )
    _require(
        len(unique_values) == 1,
        "all singleton-object cells must use one output color",
    )
    _require(
        np.all(output_grid[input_grid == background] == background),
        "background cells must be preserved",
    )

    repeated_color = next(iter(repeated_values))
    unique_color = next(iter(unique_values))

    _require(
        len(
            {
                background,
                foreground,
                repeated_color,
                unique_color,
            }
        )
        == 4,
        "the four semantic color roles must be distinct",
    )
    _require(
        {int(value) for value in np.unique(output_grid)}
        == {background, repeated_color, unique_color},
        "the output must contain exactly its three semantic colors",
    )

    return {
        "background": background,
        "foreground": foreground,
        "repeated_color": repeated_color,
        "unique_color": unique_color,
        "components": components,
        "shape_keys": shape_keys,
        "frequencies": frequencies,
        "height": input_grid.shape[0],
        "width": input_grid.shape[1],
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict), "puzzle must be a dictionary")
    _require(
        set(puzzle.keys()) == {"train", "test"},
        "puzzle must contain exactly train and test",
    )
    _require(
        isinstance(puzzle["train"], list),
        "train must be a list",
    )
    _require(
        isinstance(puzzle["test"], list),
        "test must be a list",
    )

    train_count = len(puzzle["train"])
    test_count = len(puzzle["test"])
    _require(2 <= train_count <= 10, "illegal training-pair count")
    _require(1 <= test_count <= 4, "illegal test-pair count")
    _require(
        train_count > test_count,
        "training-pair count must exceed test-pair count",
    )

    all_pairs = puzzle["train"] + puzzle["test"]
    input_signatures = set()
    pair_signatures = set()

    for pair in all_pairs:
        _require(isinstance(pair, dict), "every pair must be a dict")
        _require(
            set(pair.keys()) == {"input", "output"},
            "every pair must contain exactly input and output",
        )

        for name in ("input", "output"):
            grid = pair[name]
            _require(
                isinstance(grid, np.ndarray),
                f"{name} must be a NumPy array",
            )
            _require(grid.ndim == 2, f"{name} must be two-dimensional")
            height, width = grid.shape
            _require(
                1 <= height <= 30 and 1 <= width <= 30,
                f"{name} has illegal dimensions",
            )
            _require(
                np.issubdtype(grid.dtype, np.integer),
                f"{name} must use an integer dtype",
            )
            _require(
                int(grid.min()) >= 0 and int(grid.max()) <= 9,
                f"{name} contains a color outside 0 through 9",
            )

        input_signature = _grid_value_signature(pair["input"])
        pair_signature = (
            input_signature,
            _grid_value_signature(pair["output"]),
        )
        _require(
            input_signature not in input_signatures,
            "duplicate input grid found",
        )
        _require(
            pair_signature not in pair_signatures,
            "duplicate input/output pair found",
        )
        input_signatures.add(input_signature)
        pair_signatures.add(pair_signature)

        _require(
            _pair_source_signature(pair["input"], pair["output"])
            not in _SOURCE_PAIR_SIGNATURES,
            "a generated pair exactly duplicates a source pair",
        )

    conventions = None
    training_has_orientation_demo = False
    training_has_perturb_demo = False
    train_object_counts = set()
    backgrounds = set()

    for split_name in ("train", "test"):
        for pair in puzzle[split_name]:
            info = _inspect_pair(pair)
            frequencies = info["frequencies"]
            components = info["components"]
            count = len(components)
            _require(3 <= count <= (9 if split_name == "train" else 10),
                     "object count outside the sampled regime")
            backgrounds.add(info["background"])

            if split_name == "train":
                train_object_counts.add(count)
                training_has_orientation_demo |= (
                    _has_rotated_singleton(frequencies)
                )
                training_has_perturb_demo |= _has_perturbed_singleton(frequencies)

            current_conventions = (
                info["repeated_color"],
                info["unique_color"],
            )
            if conventions is None:
                conventions = current_conventions
            else:
                _require(
                    current_conventions == conventions,
                    "puzzle-level color conventions changed between pairs",
                )

            background = info["background"]
            repeated_color, unique_color = conventions
            expected = np.full(
                pair["input"].shape,
                background,
                dtype=pair["output"].dtype,
            )

            for component, key in zip(
                info["components"],
                info["shape_keys"],
            ):
                color = (
                    repeated_color
                    if frequencies[key] >= 2
                    else unique_color
                )
                for r, c in component:
                    expected[r, c] = color

            _require(
                np.array_equal(expected, pair["output"]),
                "output does not match exact shape-frequency recoloring",
            )

    _require(
        training_has_orientation_demo,
        "training must demonstrate orientation-sensitive matching",
    )
    _require(training_has_perturb_demo,
             "training must demonstrate a same-size non-symmetric near-match")
    _require(len(train_object_counts) >= 2,
             "training must vary object count")
    _require(len(backgrounds) >= 2,
             "pair backgrounds must vary within a puzzle")


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first.keys()) != set(second.keys()):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for first_pair, second_pair in zip(
            first[split],
            second[split],
        ):
            if not np.array_equal(
                first_pair["input"],
                second_pair["input"],
            ):
                return False
            if not np.array_equal(
                first_pair["output"],
                second_pair["output"],
            ):
                return False
    return True


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(
        tuple(
            (
                _grid_value_signature(pair["input"]),
                _grid_value_signature(pair["output"]),
            )
            for pair in puzzle[split]
        )
        for split in ("train", "test")
    )


def _color_signature(pair: dict) -> tuple[int, int]:
    info = _inspect_pair(pair)
    return (
        info["repeated_color"],
        info["unique_color"],
    )


def _layout_signature(input_grid: np.ndarray) -> tuple:
    background = _background_color(input_grid)
    components = _extract_components(input_grid, background)
    descriptions = []

    for component in components:
        min_r = min(r for r, _ in component)
        max_r = max(r for r, _ in component)
        min_c = min(c for _, c in component)
        max_c = max(c for _, c in component)
        descriptions.append(
            (
                min_r,
                min_c,
                max_r - min_r + 1,
                max_c - min_c + 1,
                len(component),
            )
        )

    return input_grid.shape, tuple(sorted(descriptions))


def self_test() -> None:
    # Automatic counts over 42 fixed seeds.
    corpus = [generate_puzzle(seed) for seed in range(42)]

    train_counts = set()
    test_counts = set()
    puzzle_signatures = set()
    color_signatures = set()
    dimensions = set()
    layout_signatures = set()
    shape_vocabulary = set()
    train_object_counts = set()
    test_object_counts = set()
    train_contacts = set()
    test_contacts = set()
    frequency_patterns = set()
    compact_pairs = 0
    border_pairs = 0
    bars = 0
    frames = 0
    large_shapes = 0
    lower_count_tests = 0
    test_without_rotation = 0
    higher_count_tests = 0
    reflected_train = 0
    reflected_test = 0
    contact_roles = Counter()
    contact_chains = 0
    dense_pairs = 0
    sparse_pairs = 0
    varied_foreground_puzzles = 0

    for puzzle in corpus:
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_signatures.add(_puzzle_signature(puzzle))

        pair_color_signatures = {
            _color_signature(pair)
            for pair in puzzle["train"] + puzzle["test"]
        }
        _require(
            len(pair_color_signatures) == 1,
            "one puzzle used inconsistent color conventions",
        )
        color_signatures.update(pair_color_signatures)

        training_infos = [
            _inspect_pair(pair) for pair in puzzle["train"]
        ]
        _require(len({_background_color(pair["input"]) for pair in
                     puzzle["train"] + puzzle["test"]}) >= 2,
                 "pair backgrounds did not vary")
        varied_foreground_puzzles += len({
            _inspect_pair(pair)["foreground"] for pair in
            puzzle["train"] + puzzle["test"]}) >= 2
        _require(len({len(info["components"]) for info in training_infos}) >= 2,
                 "training object counts did not vary")
        _require(
            any(
                _has_rotated_singleton(info["frequencies"])
                for info in training_infos
            ),
            "training did not expose orientation sensitivity",
        )
        _require(any(_has_perturbed_singleton(info["frequencies"])
                     for info in training_infos),
                 "training did not expose same-box arrangement differences")

        for info in training_infos:
            train_object_counts.add(len(info["components"]))
            train_contacts.add(_contact_count(info["components"]))
            reflected_train += _has_reflected_singleton(info["frequencies"])

        for pair in puzzle["test"]:
            info = _inspect_pair(pair)
            test_object_counts.add(len(info["components"]))
            test_contacts.add(_contact_count(info["components"]))
            lower_count_tests += len(info["components"]) <= max(
                len(train_info["components"]) for train_info in training_infos)
            higher_count_tests += len(info["components"]) > max(
                len(train_info["components"]) for train_info in training_infos)
            test_without_rotation += not _has_rotated_singleton(info["frequencies"])
            reflected_test += _has_reflected_singleton(info["frequencies"])

        for pair in puzzle["train"] + puzzle["test"]:
            input_grid = pair["input"]
            output_grid = pair["output"]
            dimensions.add(input_grid.shape)
            layout_signatures.add(_layout_signature(input_grid))

            background = _background_color(input_grid)
            info = _inspect_pair(pair)
            frequency_patterns.add(tuple(sorted(info["frequencies"].values())))
            compact_pairs += input_grid.shape[0] <= 13 and input_grid.shape[1] <= 13
            dense_pairs += float(np.mean(input_grid != background)) >= .14
            sparse_pairs += float(np.mean(input_grid != background)) <= .08
            border_pairs += bool(np.any(input_grid[0] != background) or
                                 np.any(input_grid[-1] != background) or
                                 np.any(input_grid[:, 0] != background) or
                                 np.any(input_grid[:, -1] != background))
            edges = _contact_edges(info["components"])
            roles = [info["frequencies"][key] >= 2 for key in info["shape_keys"]]
            for first, second in edges:
                contact_roles[(roles[first], roles[second]) if roles[first] == roles[second]
                              else (False, True)] += 1
            contact_chains += max((sum(index in edge for edge in edges)
                                   for index in range(len(roles))), default=0) >= 2
            for component in _extract_components(
                input_grid,
                background,
            ):
                key = _shape_key_from_cells(component)
                shape_vocabulary.add(key)
                height, width = len(key), len(key[0])
                bars += height == 1 or width == 1
                frames += height >= 3 and width >= 3 and all(
                    key[r][c] == int(r in (0, height - 1) or c in (0, width - 1))
                    for r in range(height) for c in range(width))
                large_shapes += len(component) >= 10

            _require(
                _pair_source_signature(input_grid, output_grid)
                not in _SOURCE_PAIR_SIGNATURES,
                "source pair was copied",
            )
            _require(
                input_grid.shape == output_grid.shape,
                "dimensions were not preserved",
            )
            _require(
                np.issubdtype(input_grid.dtype, np.integer)
                and np.issubdtype(output_grid.dtype, np.integer),
                "non-integer grid found",
            )
            _require(
                0 <= int(input_grid.min()) <= int(input_grid.max()) <= 9,
                "input color range failed",
            )
            _require(
                0 <= int(output_grid.min()) <= int(output_grid.max()) <= 9,
                "output color range failed",
            )

    _require(
        len(train_counts) >= 4,
        "automatic generation produced too few train counts",
    )
    _require(
        len(test_counts) >= 3,
        "automatic generation produced too few test counts",
    )
    _require(
        len(puzzle_signatures) >= 40,
        "different seeds produced insufficient puzzle diversity",
    )
    _require(
        len(color_signatures) >= 25,
        "puzzle-level color assignments lack diversity",
    )
    _require(
        len(dimensions) >= 30,
        "grid dimensions lack diversity",
    )
    _require(
        len(layout_signatures) >= 150,
        "object layouts lack diversity",
    )
    _require(
        len(shape_vocabulary) >= 30,
        "shape vocabulary lacks diversity",
    )
    _require(len(train_object_counts) >= 6 and len(test_object_counts) >= 7,
             "object count diversity is too narrow")
    _require(len(frequency_patterns) >= 12,
             "frequency patterns are too narrow")
    _require(0 in train_contacts and 0 in test_contacts and
             max(train_contacts) >= 2 and max(test_contacts) >= 2,
             "corner contact diversity is too narrow")
    _require(compact_pairs >= 8 and border_pairs >= 8,
             "compact or border-touching coverage is too narrow")
    _require(bars >= 8 and frames >= 8 and large_shapes >= 8,
             "mask family coverage is too narrow")
    _require(lower_count_tests >= 10 and test_without_rotation >= 10,
             "test regime diversity is too narrow")
    _require(higher_count_tests >= 5 and reflected_train >= 5 and reflected_test >= 5,
             "harder tests or reflection contrasts are too rare")
    _require(contact_roles[(True, True)] >= 10 and
             contact_roles[(False, False)] >= 5 and
             contact_roles[(False, True)] >= 10 and contact_chains >= 10,
             "corner contact relationships are too narrow")
    _require(dense_pairs >= 30 and sparse_pairs >= 10 and
             varied_foreground_puzzles >= 35,
             "density or input palette diversity is too narrow")

    # Explicit requested counts.
    explicit_cases = (
        (1001, 3, 1),
        (1002, 6, 2),
        (1003, 9, 4),
    )
    for seed, train_count, test_count in explicit_cases:
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        _require(
            len(puzzle["train"]) == train_count
            and len(puzzle["test"]) == test_count,
            "explicit pair counts were not honored",
        )

    # Whole-puzzle determinism.
    deterministic_first = generate_puzzle(314159, 7, 3)
    deterministic_second = generate_puzzle(314159, 7, 3)
    _require(
        _puzzles_equal(deterministic_first, deterministic_second),
        "generation is not deterministic",
    )

    # Legal boundary cases.
    for seed, train_count, test_count in (
        (2001, 2, 1),
        (2002, 10, 1),
        (2003, 10, 4),
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        _require(
            len(puzzle["train"]) == train_count
            and len(puzzle["test"]) == test_count,
            "legal boundary count failed",
        )

    # Partial count specifications.
    partial_cases = (
        (3001, None, 1),
        (3002, None, 4),
        (3003, 2, None),
        (3004, 10, None),
    )
    for seed, requested_train, requested_test in partial_cases:
        puzzle = generate_puzzle(
            seed,
            num_train=requested_train,
            num_test=requested_test,
        )
        validate_generated_puzzle(puzzle)
        if requested_train is not None:
            _require(
                len(puzzle["train"]) == requested_train,
                "partial num_train was not honored",
            )
        if requested_test is not None:
            _require(
                len(puzzle["test"]) == requested_test,
                "partial num_test was not honored",
            )

    # Illegal counts, including num_train <= num_test.
    illegal_cases = (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4},
    )
    for kwargs in illegal_cases:
        raised = False
        try:
            generate_puzzle(4001, **kwargs)
        except ValueError:
            raised = True
        _require(raised, f"illegal counts were accepted: {kwargs}")
