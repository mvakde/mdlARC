import numpy as np


_SOURCE_FRAME = (2, 10, 3, 11)
_SOURCE_SEEDS = {(8, 5), (6, 7), (3, 9), (5, 4)}
_SOURCE_PALETTE = (0, 3, 4, 2)
_PLACEMENTS = ("center", "intermediate", "side", "corner")


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _validated_count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _resolve_counts(
    rng: np.random.Generator,
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validated_count("num_train", num_train, 2, 10)
    test_count = _validated_count("num_test", num_test, 1, 4)
    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        train_count = test_count + 1 + (seed // 4) % (10 - test_count)
    elif train_count is None:
        train_count = int(rng.integers(max(2, test_count + 1), 11))
    elif test_count is None:
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    return train_count, test_count


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    paints = [int(color) for color in rng.choice(10, size=2, replace=False)]
    return {"primary": paints[0], "secondary": paints[1]}


def _sample_visible_colors(
    rng: np.random.Generator,
    parameters: dict,
    previous: tuple[int, int] | None = None,
) -> tuple[int, int]:
    available = [
        color for color in range(10)
        if color not in (parameters["primary"], parameters["secondary"])
    ]
    backgrounds = [color for color in available if previous is None or color != previous[0]]
    background = int(rng.choice(backgrounds))
    structurals = [
        color for color in available
        if color != background and (previous is None or color != previous[1])
    ]
    structural = int(rng.choice(structurals))
    return background, structural


def _sample_interior_side(rng: np.random.Generator) -> int:
    band = int(rng.choice(3, p=[0.22, 0.43, 0.35]))
    bounds = ((3, 7), (7, 14), (14, 27))[band]
    return int(rng.integers(*bounds))


def _sample_geometry(
    rng: np.random.Generator,
    small_center: bool = False,
    center_placement: bool = False,
) -> dict:
    if small_center and rng.random() < 0.15:
        interior_height = interior_width = 3
    elif center_placement and rng.random() < 0.2:
        interior_height = interior_width = int(rng.integers(5, 9))
    else:
        interior_height = _sample_interior_side(rng)
        interior_width = _sample_interior_side(rng)

    def margins(interior_side: int) -> tuple[int, int]:
        maximum_total = min(12, 28 - interior_side)
        total = int(rng.integers(2, maximum_total + 1))
        before = int(rng.integers(1, total))
        return before, total - before

    top, bottom_margin = margins(interior_height)
    left, right_margin = margins(interior_width)
    return {
        "height": top + interior_height + 2 + bottom_margin,
        "width": left + interior_width + 2 + right_margin,
        "top": top,
        "bottom": top + interior_height + 1,
        "left": left,
        "right": left + interior_width + 1,
    }


def _seed_candidates(geometry: dict, placement: str) -> list[tuple[int, int]]:
    height = geometry["bottom"] - geometry["top"] - 1
    width = geometry["right"] - geometry["left"] - 1
    center_rows = {(height - 1) // 2, height // 2}
    center_cols = {(width - 1) // 2, width // 2}
    candidates = []
    for row in range(height):
        for col in range(width):
            edge_count = int(row in (0, height - 1)) + int(col in (0, width - 1))
            if placement == "corner" and edge_count == 2:
                candidates.append((row, col))
            elif placement == "side" and edge_count == 1:
                candidates.append((row, col))
            elif placement == "center" and row in center_rows and col in center_cols:
                candidates.append((row, col))
            elif (
                placement == "intermediate"
                and edge_count == 0
                and (row not in center_rows or col not in center_cols)
            ):
                candidates.append((row, col))
    return candidates


def _render_output(
    input_grid: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    seed: tuple[int, int],
    primary: int,
    secondary: int,
) -> np.ndarray:
    output = input_grid.copy()
    for row in range(top + 1, bottom):
        for col in range(left + 1, right):
            distance = max(abs(row - seed[0]), abs(col - seed[1]))
            if distance:
                output[row, col] = primary if distance % 2 else secondary
    return output


def _generate_pair(
    rng: np.random.Generator,
    parameters: dict,
    placement: str,
    visible_colors: tuple[int, int],
    witness: bool,
    small_center: bool,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        geometry = _sample_geometry(rng, small_center, placement == "center")
        candidates = _seed_candidates(geometry, placement)
        if not candidates:
            continue
        candidate = candidates[int(rng.integers(len(candidates)))]
        seed = (geometry["top"] + 1 + candidate[0], geometry["left"] + 1 + candidate[1])
        maximum_distance = max(
            abs(seed[0] - (geometry["top"] + 1)),
            abs(seed[0] - (geometry["bottom"] - 1)),
            abs(seed[1] - (geometry["left"] + 1)),
            abs(seed[1] - (geometry["right"] - 1)),
        )
        if not witness or maximum_distance >= 4:
            break
    else:
        raise RuntimeError("could not sample a valid framed chamber")

    background, structural = visible_colors
    input_grid = np.full(
        (geometry["height"], geometry["width"]), background, dtype=np.int8
    )
    top, bottom = geometry["top"], geometry["bottom"]
    left, right = geometry["left"], geometry["right"]
    input_grid[top, left:right + 1] = structural
    input_grid[bottom, left:right + 1] = structural
    input_grid[top:bottom + 1, left] = structural
    input_grid[top:bottom + 1, right] = structural
    input_grid[seed] = structural
    output_grid = _render_output(
        input_grid, top, bottom, left, right, seed,
        parameters["primary"], parameters["secondary"],
    )
    return input_grid, output_grid


def _array_key(array: np.ndarray) -> tuple:
    normalized = np.ascontiguousarray(array.astype(np.int16, copy=False))
    return tuple(int(value) for value in array.shape), normalized.tobytes()


def _analyze_input(input_grid: np.ndarray) -> dict:
    _require(isinstance(input_grid, np.ndarray) and input_grid.ndim == 2, "input must be a 2-D array")
    height, width = input_grid.shape
    background = int(input_grid[0, 0])
    colors = [int(color) for color in np.unique(input_grid)]
    _require(len(colors) == 2 and background in colors, "input must have two visible roles")
    structural = next(color for color in colors if color != background)
    cells = np.argwhere(input_grid == structural)
    top, bottom = int(cells[:, 0].min()), int(cells[:, 0].max())
    left, right = int(cells[:, 1].min()), int(cells[:, 1].max())
    _require(1 <= top < bottom < height - 1, "frame needs upper and lower exterior")
    _require(1 <= left < right < width - 1, "frame needs left and right exterior")
    _require(bottom - top - 1 >= 3 and right - left - 1 >= 3, "chamber is too small")
    frame = np.zeros(input_grid.shape, dtype=bool)
    frame[top, left:right + 1] = True
    frame[bottom, left:right + 1] = True
    frame[top:bottom + 1, left] = True
    frame[top:bottom + 1, right] = True
    _require(bool(np.all(input_grid[frame] == structural)), "frame is not closed")
    interior = input_grid[top + 1:bottom, left + 1:right]
    seed_cells = np.argwhere(interior == structural)
    _require(len(seed_cells) == 1, "exactly one interior seed is required")
    seed = (top + 1 + int(seed_cells[0, 0]), left + 1 + int(seed_cells[0, 1]))
    allowed = frame.copy()
    allowed[seed] = True
    _require(bool(np.all(input_grid[~allowed] == background)), "non-frame cells must be background")
    return {
        "background": background,
        "structural": structural,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "seed": seed,
        "interior_height": bottom - top - 1,
        "interior_width": right - left - 1,
    }


def _placement_of(info: dict) -> str:
    height, width = info["interior_height"], info["interior_width"]
    row = info["seed"][0] - info["top"] - 1
    col = info["seed"][1] - info["left"] - 1
    edge_count = int(row in (0, height - 1)) + int(col in (0, width - 1))
    if edge_count == 2:
        return "corner"
    if edge_count == 1:
        return "side"
    if row in {(height - 1) // 2, height // 2} and col in {(width - 1) // 2, width // 2}:
        return "center"
    return "intermediate"


def _distance_colors(output_grid: np.ndarray, info: dict, distance: int) -> set[int]:
    values = set()
    for row in range(info["top"] + 1, info["bottom"]):
        for col in range(info["left"] + 1, info["right"]):
            if max(abs(row - info["seed"][0]), abs(col - info["seed"][1])) == distance:
                values.add(int(output_grid[row, col]))
    return values


def _infer_paints(train_pairs: list) -> tuple[int, int]:
    info = _analyze_input(train_pairs[0]["input"])
    output = train_pairs[0]["output"]
    primary_values = _distance_colors(output, info, 1)
    secondary_values = _distance_colors(output, info, 2)
    _require(len(primary_values) == len(secondary_values) == 1, "training must show both paints")
    primary = next(iter(primary_values))
    secondary = next(iter(secondary_values))
    _require(primary != secondary, "shell paints must differ")
    return primary, secondary


def _is_original_source_pair(input_grid: np.ndarray, info: dict, paints: tuple[int, int]) -> bool:
    return (
        input_grid.shape == (15, 15)
        and (info["top"], info["bottom"], info["left"], info["right"]) == _SOURCE_FRAME
        and info["seed"] in _SOURCE_SEEDS
        and (info["background"], info["structural"], *paints) == _SOURCE_PALETTE
    )


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "pair lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "pair counts")
    seen_inputs = set()
    seen_pairs = set()
    paints = None
    training_modes = set()
    training_backgrounds = set()
    training_structurals = set()
    witness_found = False

    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, f"{label} keys")
            for role in ("input", "output"):
                grid = pair[role]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, f"{label} {role} type")
                _require(all(1 <= side <= 30 for side in grid.shape), f"{label} {role} dimensions")
                _require(np.issubdtype(grid.dtype, np.integer), f"{label} {role} dtype")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), f"{label} {role} colors")
            input_grid, output_grid = pair["input"], pair["output"]
            _require(input_grid.shape == output_grid.shape, f"{label} dimensions changed")
            _require(input_grid.dtype == output_grid.dtype, f"{label} dtype changed")
            input_key = _array_key(input_grid)
            pair_key = input_key, _array_key(output_grid)
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, f"{label} duplicate")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

            info = _analyze_input(input_grid)
            distance_one = _distance_colors(output_grid, info, 1)
            distance_two = _distance_colors(output_grid, info, 2)
            if paints is None:
                _require(split == "train" and index == 0, "missing training example")
                paints = _infer_paints(puzzle["train"])
            _require(distance_one == {paints[0]}, f"{label} primary paint")
            if distance_two:
                _require(distance_two == {paints[1]}, f"{label} secondary paint")
            _require(
                len({info["background"], info["structural"], *paints}) == 4,
                f"{label} color roles overlap",
            )
            expected = _render_output(
                input_grid, info["top"], info["bottom"], info["left"],
                info["right"], info["seed"], *paints,
            )
            _require(np.array_equal(output_grid, expected), f"{label} shell rule")
            _require(not np.array_equal(input_grid, output_grid), f"{label} unchanged")
            _require(not _is_original_source_pair(input_grid, info, paints), f"{label} source copy")

            if split == "train":
                training_modes.add(_placement_of(info))
                training_backgrounds.add(info["background"])
                training_structurals.add(info["structural"])
                maximum_distance = max(
                    abs(info["seed"][0] - info["top"] - 1),
                    abs(info["seed"][0] - info["bottom"] + 1),
                    abs(info["seed"][1] - info["left"] - 1),
                    abs(info["seed"][1] - info["right"] + 1),
                )
                if maximum_distance >= 4:
                    witness_found = True
                    for distance in range(1, 5):
                        _require(
                            _distance_colors(output_grid, info, distance)
                            == {paints[0] if distance % 2 else paints[1]},
                            f"{label} parity evidence",
                        )

    _require(len(training_modes) >= 2, "training placement diversity")
    _require(len(training_backgrounds) >= 2, "training background variation")
    _require(len(training_structurals) >= 2, "training structural variation")
    _require(witness_found, "training must show shells through distance four")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed_value = int(seed)
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_counts(rng, seed_value, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    first_modes = [str(mode) for mode in rng.choice(_PLACEMENTS, size=2, replace=False)]
    training_modes = first_modes + [str(rng.choice(_PLACEMENTS)) for _ in range(train_count - 2)]
    test_modes = [
        str(rng.choice(_PLACEMENTS, p=[0.2, 0.2, 0.32, 0.28]))
        for _ in range(test_count)
    ]
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    first_visible = None

    for split, modes in (("train", training_modes), ("test", test_modes)):
        for index, placement in enumerate(modes):
            previous = first_visible if split == "train" and index == 1 else None
            visible = _sample_visible_colors(rng, parameters, previous)
            if first_visible is None:
                first_visible = visible
            for _ in range(100):
                input_grid, output_grid = _generate_pair(
                    rng, parameters, placement, visible,
                    witness=split == "train" and index == 0,
                    small_center=split == "test" and placement == "center",
                )
                input_key = _array_key(input_grid)
                info = _analyze_input(input_grid)
                if input_key not in seen_inputs and not _is_original_source_pair(
                    input_grid, info, (parameters["primary"], parameters["secondary"])
                ):
                    seen_inputs.add(input_key)
                    puzzle[split].append({"input": input_grid, "output": output_grid})
                    break
            else:
                raise RuntimeError("could not construct a unique non-source pair")

    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        tuple((_array_key(pair["input"]), _array_key(pair["output"])) for pair in puzzle[split])
        for split in ("train", "test")
    )


def self_test() -> None:
    train_counts = set()
    test_counts = set()
    puzzle_keys = set()
    interior_shapes = set()
    canvas_shapes = set()
    layouts = set()
    train_modes = set()
    test_modes = set()
    observed_sides = set()
    even_sides = set()
    odd_sides = set()
    narrow = 0
    square = 0
    near_thirty = 0
    novel_fifteen = 0
    within_palette_variation = 0
    changed_placement = 0
    shared_placement = 0
    small_missing_paint = 0

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(_puzzle_key(puzzle))
        paints = _infer_paints(puzzle["train"])
        local_visible = set()
        local_train_modes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_input(pair["input"])
                mode = _placement_of(info)
                local_visible.add((info["background"], info["structural"]))
                interior = (info["interior_height"], info["interior_width"])
                interior_shapes.add(interior)
                canvas_shapes.add(pair["input"].shape)
                layouts.add((pair["input"].shape, info["top"], info["left"], interior, info["seed"]))
                for side in interior:
                    (even_sides if side % 2 == 0 else odd_sides).add(side)
                ratio = max(interior) / min(interior)
                narrow += ratio >= 2
                square += ratio <= 1.25
                near_thirty += max(pair["input"].shape) >= 29
                novel_fifteen += pair["input"].shape == (15, 15)
                if split == "train":
                    local_train_modes.add(mode)
                    train_modes.add(mode)
                else:
                    test_modes.add(mode)
                    changed_placement += mode not in local_train_modes
                    shared_placement += mode in local_train_modes
                    small_missing_paint += paints[1] not in set(int(v) for v in np.unique(pair["output"]))
                predicted = _render_output(
                    pair["input"], info["top"], info["bottom"], info["left"],
                    info["right"], info["seed"], *paints,
                )
                _require(np.array_equal(predicted, pair["output"]), "self-test prediction")
                if mode == "side":
                    row = info["seed"][0] - info["top"] - 1
                    col = info["seed"][1] - info["left"] - 1
                    if row == 0:
                        observed_sides.add("top")
                    elif row == info["interior_height"] - 1:
                        observed_sides.add("bottom")
                    elif col == 0:
                        observed_sides.add("left")
                    else:
                        observed_sides.add("right")
        within_palette_variation += len(local_visible) >= 2

    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic count diversity")
    _require(len(puzzle_keys) >= 40, "cross-seed puzzle diversity")
    _require(len(interior_shapes) >= 40 and len(canvas_shapes) >= 40, "dimension diversity")
    _require(len(layouts) >= 100, "layout diversity")
    _require(train_modes == set(_PLACEMENTS) and test_modes == set(_PLACEMENTS), "placement coverage")
    _require(observed_sides == {"top", "bottom", "left", "right"}, "side coverage")
    _require(len(even_sides) >= 5 and len(odd_sides) >= 5, "even and odd sides")
    _require(narrow >= 12 and square >= 12, "aspect ratio coverage")
    _require(near_thirty >= 2 and novel_fifteen >= 1, "large and source-sized canvases")
    _require(within_palette_variation == 42, "pair-level recoloring")
    _require(changed_placement >= 5 and shared_placement >= 5, "train/test placement relationships")
    _require(small_missing_paint >= 1, "small test with clipped paint")

    _require(_puzzle_key(generate_puzzle(17)) == _puzzle_key(generate_puzzle(17)), "determinism")
    _require(_puzzle_key(generate_puzzle(987654, 7, 3)) == _puzzle_key(generate_puzzle(987654, 7, 3)), "explicit determinism")
    for requested_train, requested_test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1000 + (requested_train or 0) * 10 + (requested_test or 0), requested_train, requested_test)
        validate_generated_puzzle(puzzle)
        _require(requested_train is None or len(puzzle["train"]) == requested_train, "train count request")
        _require(requested_test is None or len(puzzle["test"]) == requested_test, "test count request")
    for kwargs in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
        {"num_train": 2.0}, {"num_test": True},
    ):
        try:
            generate_puzzle(3001, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {kwargs}")
