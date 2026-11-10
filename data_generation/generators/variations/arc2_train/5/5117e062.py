import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _array_key(array: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in array.shape),
        tuple(int(x) for x in array.reshape(-1)),
    )


def _pair_key(pair: dict) -> tuple:
    return (_array_key(pair["input"]), _array_key(pair["output"]))


def _four_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False

    start = (int(coords[0, 0]), int(coords[0, 1]))
    seen = {start}
    stack = [start]

    while stack:
        row, col = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            if (
                0 <= nr < mask.shape[0]
                and 0 <= nc < mask.shape[1]
                and bool(mask[nr, nc])
                and (nr, nc) not in seen
            ):
                seen.add((nr, nc))
                stack.append((nr, nc))

    return len(seen) == int(mask.sum())


def _shape_key(mask: np.ndarray) -> bytes:
    return np.asarray(mask, dtype=np.bool_).tobytes()


def _is_asymmetric(mask: np.ndarray) -> bool:
    forms = set()
    for rotation in range(4):
        rotated = np.rot90(mask, rotation)
        forms.add(rotated.tobytes())
        forms.add(np.fliplr(rotated).tobytes())
    return len(forms) == 8


def _build_mask_banks() -> tuple[list[np.ndarray], list[np.ndarray]]:
    all_masks = []
    selected_masks = []

    for bits in range(1 << 9):
        mask = np.array(
            [(bits >> index) & 1 for index in range(9)],
            dtype=np.bool_,
        ).reshape(3, 3)

        occupied = int(mask.sum())
        if not 4 <= occupied <= 8:
            continue
        if not (
            mask[0, :].any()
            and mask[2, :].any()
            and mask[:, 0].any()
            and mask[:, 2].any()
        ):
            continue
        if not _four_connected(mask):
            continue

        all_masks.append(mask)
        if bool(mask[1, 1]):
            selected_masks.append(mask)

    if len(selected_masks) < 14:
        raise RuntimeError("Insufficient selected motif vocabulary")

    return all_masks, selected_masks


_ALL_MASKS, _SELECTED_MASKS = _build_mask_banks()
_SYMMETRIC_SELECTED = [mask for mask in _SELECTED_MASKS if not _is_asymmetric(mask)]
_ASYMMETRIC_SELECTED = [mask for mask in _SELECTED_MASKS if _is_asymmetric(mask)]


def _text_grid(rows: list[str]) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = (
    (
        _text_grid(
            [
                "0000000020000",
                "0000000222000",
                "0000000020000",
                "0000000000000",
                "0003300000000",
                "0038000000000",
                "0003300000000",
                "0000000000000",
                "0000000101000",
                "0000000111000",
                "0000000111000",
                "0000000000000",
                "0000000000000",
            ]
        ),
        _text_grid(["033", "330", "033"]),
    ),
    (
        _text_grid(
            [
                "0030000000400",
                "0333000004840",
                "0330000000400",
                "0000000000000",
                "0000200000000",
                "0000222000000",
                "0000020000000",
                "0000000000000",
                "0000000000000",
                "0000000006600",
                "0000000060600",
                "0000000066600",
                "0000000000000",
            ]
        ),
        _text_grid(["040", "444", "040"]),
    ),
    (
        _text_grid(
            [
                "0000000000000",
                "0000000000000",
                "0222000000000",
                "0080000030300",
                "0220000033300",
                "0000000000300",
                "0000000000000",
                "0000000000000",
                "0001010000000",
                "0001110000000",
                "0000100000000",
                "0000000000000",
                "0000000000000",
            ]
        ),
        _text_grid(["222", "020", "220"]),
    ),
    (
        _text_grid(
            [
                "0000000000000",
                "0000011100000",
                "0000001100000",
                "0000000100000",
                "0000000000770",
                "0000000007870",
                "0200000000700",
                "0220000000000",
                "0022000000000",
                "0000000333000",
                "0000000030000",
                "0000000333000",
                "0000000000000",
            ]
        ),
        _text_grid(["077", "777", "070"]),
    ),
)


def _is_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    for source_input, source_output in _SOURCE_PAIRS:
        if (
            input_grid.shape == source_input.shape
            and output_grid.shape == source_output.shape
            and np.array_equal(input_grid, source_input)
            and np.array_equal(output_grid, source_output)
        ):
            return True
    return False


def _coerce_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")

    result = int(seed)
    if result < 0:
        raise ValueError("seed must be a nonnegative integer")
    return result


def _coerce_count(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _coerce_count("num_train", num_train)
    test_count = _coerce_count("num_test", num_test)

    if train_count is not None and not 2 <= train_count <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test_count is not None and not 1 <= test_count <= 4:
        raise ValueError("num_test must be between 1 and 4")

    if train_count is None and test_count is None:
        test_count = 1 + (seed % 4)
        minimum_train = max(2, test_count + 1)
        span = 11 - minimum_train
        train_count = minimum_train + ((seed // 4) % span)
    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal test count for the requested train count")
        test_count = int(rng.integers(1, maximum_test + 1))

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    colors = tuple(int(x) for x in rng.permutation(10))
    return {
        "familiar": colors[:4],
        "novel": colors[4],
        "role_colors": colors[5:],
        "variable_roles": bool(rng.integers(0, 2)),
        "first_test_novel": bool(rng.integers(0, 2)),
    }


def _compatible_windows(first: tuple[int, int], second: tuple[int, int]) -> bool:
    r1, c1 = first
    r2, c2 = second
    return r1 + 3 < r2 or r2 + 3 < r1 or c1 + 3 < c2 or c2 + 3 < c1


def _canvas_and_positions(
    rng: np.random.Generator,
    count: int,
    used_layouts: set[tuple],
    different_size: tuple[int, int] | None,
) -> tuple[int, int, tuple[tuple[int, int], ...], tuple]:
    for attempt in range(80):
        band = int(rng.integers(0, 4))
        if band == 0:
            height = width = 13
        elif band == 1:
            height = int(rng.integers(8, 14))
            width = int(rng.integers(8, 14))
        elif band == 2:
            height = int(rng.integers(12, 21))
            width = int(rng.integers(12, 21))
        else:
            height = int(rng.integers(18, 31))
            width = int(rng.integers(18, 31))
        if different_size == (height, width):
            continue
        if len(range(0, height - 2, 4)) * len(range(0, width - 2, 4)) < count:
            continue

        candidates = [(r, c) for r in range(height - 2) for c in range(width - 2)]
        for _ in range(16):
            chosen = []
            for raw in rng.permutation(len(candidates)):
                position = candidates[int(raw)]
                if all(_compatible_windows(position, prior) for prior in chosen):
                    chosen.append(position)
                    if len(chosen) == count:
                        break
            if len(chosen) == count:
                signature = (height, width, tuple(sorted(chosen)))
                if signature not in used_layouts:
                    return height, width, tuple(chosen), signature

        for row_offset in rng.permutation(4):
            for col_offset in rng.permutation(4):
                slots = [
                    (r, c)
                    for r in range(int(row_offset), height - 2, 4)
                    for c in range(int(col_offset), width - 2, 4)
                ]
                if len(slots) >= count:
                    indices = rng.permutation(len(slots))[:count]
                    chosen = tuple(slots[int(i)] for i in indices)
                    signature = (height, width, tuple(sorted(chosen)))
                    if signature not in used_layouts:
                        return height, width, chosen, signature
    raise RuntimeError("could not place separated 3 by 3 motifs")


def _pair_roles(parameters: dict, rng: np.random.Generator, split: str, index: int) -> tuple[int, int]:
    roles = parameters["role_colors"]
    if not parameters["variable_roles"]:
        return roles[0], roles[1]
    if split == "train" and index < 2:
        return (roles[0], roles[2]) if index == 0 else (roles[1], roles[3])
    indices = rng.choice(5, size=2, replace=False)
    return roles[int(indices[0])], roles[int(indices[1])]


def _sample_selected_mask(
    rng: np.random.Generator,
    symmetric: bool,
    history: list[np.ndarray],
    prefer_reuse: bool,
) -> np.ndarray:
    matching = [m for m in history if bool(m[1, 1]) and (_is_asymmetric(m) != symmetric)]
    if matching and (prefer_reuse or rng.random() < 0.27):
        return matching[int(rng.integers(len(matching)))].copy()
    if symmetric and rng.random() < 0.30:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool_)
    bank = _SYMMETRIC_SELECTED if symmetric else _ASYMMETRIC_SELECTED
    return bank[int(rng.integers(len(bank)))].copy()


_HOLE_MASKS = [mask for mask in _ALL_MASKS if not bool(mask[1, 1])]


def _generate_pair(
    rng: np.random.Generator,
    split: str,
    index: int,
    parameters: dict,
    count: int,
    selected_color: int,
    repeated_color: bool,
    symmetric: bool,
    history: list[np.ndarray],
    prefer_reuse: bool,
    used_layouts: set[tuple],
    different_size: tuple[int, int] | None,
) -> tuple[dict, tuple, list[np.ndarray]]:
    height, width, positions, layout = _canvas_and_positions(
        rng, count, used_layouts, different_size
    )
    background, marker = _pair_roles(parameters, rng, split, index)
    selected_mask = _sample_selected_mask(rng, symmetric, history, prefer_reuse)
    masks = [selected_mask]
    for distractor_index in range(count - 1):
        if split == "train" and index == 0 and distractor_index == 0:
            bank = _HOLE_MASKS
        elif rng.random() < 0.26:
            bank = _HOLE_MASKS
        elif rng.random() < 0.16:
            masks.append(selected_mask.copy())
            continue
        else:
            bank = _ALL_MASKS
        masks.append(bank[int(rng.integers(len(bank)))].copy())

    familiar = parameters["familiar"]
    other_colors = [x for x in familiar if x != selected_color]
    colors = [selected_color]
    for distractor_index in range(count - 1):
        if repeated_color and distractor_index == 0:
            colors.append(selected_color)
        else:
            colors.append(other_colors[int(rng.integers(len(other_colors)))])

    grid = np.full((height, width), background, dtype=np.int8)
    object_order = rng.permutation(count)
    selected_position = None
    for position, raw_object_index in zip(positions, object_order):
        object_index = int(raw_object_index)
        row, col = position
        patch = grid[row : row + 3, col : col + 3]
        patch[masks[object_index]] = colors[object_index]
        if object_index == 0:
            selected_position = position
    if selected_position is None:
        raise RuntimeError("selected motif was not placed")
    row, col = selected_position
    grid[row + 1, col + 1] = marker
    output = grid[row : row + 3, col : col + 3].copy()
    output[1, 1] = selected_color
    return {"input": grid, "output": output}, layout, masks


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed_value = _coerce_seed(seed)
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_counts(seed_value, rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    used_layouts = set()
    history = []
    first_train_size = None
    first_count = int(rng.integers(3, 5))

    for split, split_count in (("train", train_count), ("test", test_count)):
        for index in range(split_count):
            if split == "train":
                if index == 0:
                    count = first_count
                elif index == 1:
                    options = [n for n in range(2, 6) if n != first_count]
                    count = options[int(rng.integers(len(options)))]
                else:
                    count = int(rng.integers(2, 6))
                selected_color = parameters["familiar"][
                    index if index < 2 else int(rng.integers(4))
                ]
                repeated_color = index == 0 or bool(rng.integers(0, 2))
                symmetric = index == 0 or (index > 1 and bool(rng.integers(0, 2)))
                prefer_reuse = index > 1 and bool(rng.integers(0, 3) == 0)
            else:
                count = int(rng.integers(2, 6))
                novel = parameters["first_test_novel"] if index == 0 else bool(rng.integers(0, 2))
                selected_color = parameters["novel"] if novel else parameters["familiar"][int(rng.integers(4))]
                repeated_color = bool(rng.integers(0, 2))
                symmetric = bool(rng.integers(0, 2))
                prefer_reuse = bool(rng.integers(0, 2))

            for _ in range(40):
                pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(pair_seed)
                pair, layout, masks = _generate_pair(
                    pair_rng, split, index, parameters, count, selected_color,
                    repeated_color, symmetric, history, prefer_reuse,
                    used_layouts, first_train_size if split == "train" and index == 1 else None,
                )
                key = _array_key(pair["input"])
                if key in seen_inputs or _is_source_pair(pair["input"], pair["output"]):
                    continue
                puzzle[split].append(pair)
                seen_inputs.add(key)
                used_layouts.add(layout)
                history.extend(masks)
                if split == "train":
                    if index == 0:
                        first_train_size = pair["input"].shape
                break
            else:
                raise RuntimeError("could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(binary: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = set()
    result = []
    for raw_row, raw_col in np.argwhere(binary):
        start = (int(raw_row), int(raw_col))
        if start in seen:
            continue
        component = []
        stack = [start]
        seen.add(start)
        while stack:
            row, col = stack.pop()
            component.append((row, col))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                adjacent = (row + dr, col + dc)
                if (0 <= adjacent[0] < binary.shape[0]
                    and 0 <= adjacent[1] < binary.shape[1]
                    and binary[adjacent]
                    and adjacent not in seen):
                    seen.add(adjacent)
                    stack.append(adjacent)
        result.append(component)
    return result


def _inspect_pair(pair: dict) -> dict:
    grid, output = pair["input"], pair["output"]
    _require(8 <= grid.shape[0] <= 30 and 8 <= grid.shape[1] <= 30,
             "input canvas is outside the supported range")
    _require(output.shape == (3, 3), "output must be a 3 by 3 motif")
    counts = np.bincount(grid.astype(np.int64).ravel(), minlength=10)
    background = int(np.argmax(counts))
    singletons = np.flatnonzero(counts == 1)
    _require(len(singletons) == 1, "marker must be the sole singleton color")
    marker = int(singletons[0])
    _require(background != marker, "marker and background must differ")
    marker_position = tuple(int(x) for x in np.argwhere(grid == marker)[0])
    components = _components(grid != background)
    _require(2 <= len(components) <= 5, "candidate count must be 2 to 5")
    windows = []
    masks = []
    colors = []
    selected = None
    for component in components:
        rows = [x[0] for x in component]
        cols = [x[1] for x in component]
        r0, c0 = min(rows), min(cols)
        _require(max(rows) - r0 == 2 and max(cols) - c0 == 2,
                 "motif must have tight 3 by 3 bounds")
        mask = np.zeros((3, 3), dtype=np.bool_)
        for row, col in component:
            mask[row - r0, col - c0] = True
        _require(4 <= len(component) <= 8, "motif occupancy must be 4 to 8")
        _require(mask[0].any() and mask[2].any() and mask[:, 0].any() and mask[:, 2].any(),
                 "motif bounds are not tight")
        has_marker = marker_position in component
        carriers = {int(grid[row, col]) for row, col in component
                    if (row, col) != marker_position}
        _require(len(carriers) == 1, "each motif needs one carrier color")
        carrier = next(iter(carriers))
        _require(carrier not in (background, marker), "carrier color has another role")
        if has_marker:
            _require(selected is None, "more than one marked motif")
            _require(marker_position == (r0 + 1, c0 + 1),
                     "selected marker must replace the occupied center")
            _require(mask[1, 1], "selected center is unoccupied")
            _require(any(grid[marker_position[0] + dr, marker_position[1] + dc] == carrier
                         for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))),
                     "marker must touch its carrier")
            expected = grid[r0 : r0 + 3, c0 : c0 + 3].copy()
            expected[1, 1] = carrier
            _require(np.array_equal(output, expected), "output is not the repaired crop")
            selected = len(windows)
        windows.append((r0, c0))
        masks.append(mask)
        colors.append(carrier)
    _require(selected is not None, "marked motif is missing")
    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            _require(_compatible_windows(windows[i], windows[j]),
                     "motif windows must be separated by background")
    _require(not np.any(output == marker), "marker must not survive in output")
    return {
        "background": background, "marker": marker,
        "count": len(windows), "size": grid.shape,
        "layout": (grid.shape, tuple(sorted(windows))),
        "selected_color": colors[selected],
        "selected_mask": _shape_key(masks[selected]),
        "symmetric": not _is_asymmetric(masks[selected]),
        "hole_distractor": any(not bool(m[1, 1]) for i, m in enumerate(masks) if i != selected),
        "same_color_distractor": any(color == colors[selected] for i, color in enumerate(colors) if i != selected),
        "all_masks": tuple(_shape_key(m) for m in masks),
        "selected_position": windows[selected],
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal pair counts")
    seen_inputs = set()
    seen_pairs = set()
    for pair in puzzle["train"] + puzzle["test"]:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                 "each pair needs exactly input and output")
        for name in ("input", "output"):
            grid = pair[name]
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                     "grids must be two-dimensional NumPy arrays")
            _require(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
            _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                     "grid dimensions exceed ARC limits")
            _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                     "grid color exceeds ARC limits")
        input_key = _array_key(pair["input"])
        pair_key = _pair_key(pair)
        _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                 "duplicate input or pair")
        _require(not _is_source_pair(pair["input"], pair["output"]),
                 "generated pair duplicates a source pair")
        seen_inputs.add(input_key)
        seen_pairs.add(pair_key)
    training = [_inspect_pair(pair) for pair in puzzle["train"]]
    testing = [_inspect_pair(pair) for pair in puzzle["test"]]
    all_metadata = training + testing
    train_backgrounds = {m["background"] for m in training}
    train_markers = {m["marker"] for m in training}
    if len(train_backgrounds) == 1 and len(train_markers) == 1:
        expected_role = (next(iter(train_backgrounds)), next(iter(train_markers)))
        _require(all((m["background"], m["marker"]) == expected_role
                     for m in testing),
                 "test changes a role color not demonstrated in training")
    else:
        _require(len(train_backgrounds) >= 2 and len(train_markers) >= 2,
                 "training must demonstrate both changing local color roles")
    layouts = [m["layout"] for m in all_metadata]
    _require(len(layouts) == len(set(layouts)), "candidate layouts must be distinct")
    _require(len({m["count"] for m in training}) >= 2,
             "training must vary candidate counts")
    _require(len({m["size"] for m in training}) >= 2,
             "training must vary canvas size")
    _require(len({m["selected_color"] for m in training}) >= 2,
             "training must vary selected carrier colors")
    _require(any(m["symmetric"] for m in training)
             and any(not m["symmetric"] for m in training),
             "training must show symmetric and asymmetric selections")
    _require(any(m["hole_distractor"] for m in training),
             "training must include a center-hole distractor")
    _require(any(m["same_color_distractor"] for m in training),
             "training must show that marker wins over matching color")
    for metadata in all_metadata:
        _require(metadata["count"] >= 2, "each pair needs a distractor")


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(
        len(first[split]) == len(second[split])
        and all(np.array_equal(a[name], b[name])
                for a, b in zip(first[split], second[split])
                for name in ("input", "output"))
        for split in ("train", "test")
    )


def _predict_from_training(train_pairs: list[dict], test_input: np.ndarray) -> np.ndarray:
    def restored_crop(input_grid: np.ndarray) -> np.ndarray:
        counts = np.bincount(input_grid.astype(np.int64).ravel(), minlength=10)
        background = int(np.argmax(counts))
        singleton_colors = np.flatnonzero(counts == 1)
        _require(len(singleton_colors) == 1, "prediction needs one marker")
        marker = int(singleton_colors[0])
        row, col = (int(x) for x in np.argwhere(input_grid == marker)[0])
        crop = input_grid[row - 1 : row + 2, col - 1 : col + 2].copy()
        carriers = {int(x) for x in np.unique(crop)} - {background, marker}
        _require(len(carriers) == 1, "prediction needs one local carrier")
        crop[1, 1] = carriers.pop()
        return crop

    for pair in train_pairs:
        _require(np.array_equal(restored_crop(pair["input"].copy()),
                                pair["output"].copy()),
                 "training does not establish marker repair and cropping")
    return restored_crop(test_input.copy())


def self_test() -> None:
    _require(_puzzles_equal(generate_puzzle(31415), generate_puzzle(31415)),
             "automatic generation is not deterministic")
    _require(_puzzles_equal(generate_puzzle(27182, 7, 3), generate_puzzle(27182, 7, 3)),
             "explicit generation is not deterministic")
    _require(not _puzzles_equal(generate_puzzle(27182, 7, 3), generate_puzzle(27183, 7, 3)),
             "different seeds produced the same puzzle")
    for seed, nt, ns in ((4001, 2, 1), (4002, 10, 1), (4003, 10, 4),
                         (5001, None, 1), (5002, None, 4),
                         (5003, 2, None), (5004, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            _require(len(puzzle["train"]) == nt, "train count not honored")
        if ns is not None:
            _require(len(puzzle["test"]) == ns, "test count not honored")
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 3), (3, 4), (4, 4)):
        try:
            generate_puzzle(6001, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts did not raise ValueError")

    train_counts, test_counts, sizes, layouts, selected_masks = set(), set(), set(), set(), set()
    backgrounds, markers, object_counts, selected_colors = set(), set(), set(), set()
    symmetric_splits = {"train": 0, "test": 0}
    asymmetric_splits = {"train": 0, "test": 0}
    hole_splits = {"train": 0, "test": 0}
    repeated_color_splits = {"train": 0, "test": 0}
    novel_tests = familiar_tests = reused_test_masks = 0
    variable_role_puzzles = stable_role_puzzles = 0
    source_like = 0
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for test_pair in puzzle["test"]:
            prediction = _predict_from_training(puzzle["train"], test_pair["input"])
            _require(np.array_equal(prediction, test_pair["output"]),
                     "test output is not recoverable from training and input")
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(tuple(_pair_key(p) for split in ("train", "test") for p in puzzle[split]))
        train = [_inspect_pair(p) for p in puzzle["train"]]
        test = [_inspect_pair(p) for p in puzzle["test"]]
        train_colors = {m["selected_color"] for m in train}
        train_all_masks = {mask for m in train for mask in m["all_masks"]}
        roles = {(m["background"], m["marker"]) for m in train + test}
        variable_role_puzzles += len(roles) > 1
        stable_role_puzzles += len(roles) == 1
        for m in test:
            novel_tests += m["selected_color"] not in {int(v) for p in puzzle["train"] for v in np.unique(p["input"])}
            familiar_tests += m["selected_color"] in train_colors
            reused_test_masks += m["selected_mask"] in train_all_masks
        for split, metadata in (("train", train), ("test", test)):
            for m in metadata:
                sizes.add(m["size"])
                layouts.add(m["layout"])
                selected_masks.add(m["selected_mask"])
                backgrounds.add(m["background"])
                markers.add(m["marker"])
                object_counts.add(m["count"])
                selected_colors.add(m["selected_color"])
                symmetric_splits[split] += m["symmetric"]
                asymmetric_splits[split] += not m["symmetric"]
                hole_splits[split] += m["hole_distractor"]
                repeated_color_splits[split] += m["same_color_distractor"]
                source_like += m["size"] == (13, 13)
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack diversity")
    _require(len(fingerprints) == 42, "complete puzzles lack diversity")
    _require(len(sizes) >= 30 and source_like >= 10,
             "canvas sizes or source-like cases lack diversity")
    _require(len(layouts) >= 150 and len(selected_masks) >= 35,
             "layouts or selected shapes lack diversity")
    _require(object_counts == {2, 3, 4, 5}, "candidate counts lack diversity")
    _require(len(backgrounds) >= 7 and len(markers) >= 7 and len(selected_colors) >= 7,
             "semantic color assignments lack diversity")
    _require(all(symmetric_splits.values()) and all(asymmetric_splits.values()),
             "both splits need symmetric and asymmetric selections")
    _require(all(hole_splits.values()) and all(repeated_color_splits.values()),
             "both splits need center holes and matching-color distractors")
    _require(novel_tests and familiar_tests and reused_test_masks,
             "test shifts and shape reuse lack coverage")
    _require(variable_role_puzzles and stable_role_puzzles,
             "both palette regimes need coverage")
