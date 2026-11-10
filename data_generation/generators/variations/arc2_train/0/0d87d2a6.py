import hashlib

import numpy as np


_SOURCE_DIGESTS = {
    "7f013e4a3972cedcee546ca07433142c6f5ec9383d1524ce89bb97ffa01c41d4",
    "1bfb8ccd3c07824f6caaf2f0c41c94b21c2d99ce6ffe7108a17f0e9ed997b4a9",
    "203f823886bbfa3a7e2b5b8538e09e6b98cfcb1f76e18fc1734541e4fe74c09c",
    "bdb05aaf143c7ab83369e8c20907727175e33a4e25990f403cb2c05c607ac111",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _count(value: int | None, name: str, lower: int, upper: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not lower <= int(value) <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return int(value)


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train_count = _count(num_train, "num_train", 2, 10)
    test_count = _count(num_test, "num_test", 1, 4)
    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        train_count = max(2, test_count + 1) + (seed // 4) % (10 - max(2, test_count + 1) + 1)
    elif train_count is None:
        train_count = max(2, test_count + 1) + (seed // 4) % (10 - max(2, test_count + 1) + 1)
    elif test_count is None:
        test_count = 1 + seed % min(4, train_count - 1)
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    return train_count, test_count


def _sample_lines(rng: np.random.Generator, length: int, count: int) -> tuple[int, ...]:
    if not count:
        return ()
    choices = list(range(1, length - 1))
    rng.shuffle(choices)
    selected = []
    for coordinate in choices:
        if all(abs(coordinate - other) >= 2 for other in selected):
            selected.append(coordinate)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise RuntimeError("insufficient line positions")
    return tuple(sorted(selected))


def _shape(rng: np.random.Generator, height: int, width: int, kind: str) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    if kind == "rectangle":
        return mask
    if kind == "L":
        mask[:] = False
        thickness = 1 if min(height, width) < 5 else int(rng.integers(1, 3))
        mask[-thickness:, :] = True
        mask[:, :thickness] = True
    elif kind == "step":
        mask[:] = False
        middle = height // 2
        mask[:middle + 1, :max(2, width // 2)] = True
        mask[middle:, width // 2:] = True
    elif kind == "cross":
        mask[:] = False
        mask[:, width // 2] = True
        mask[height // 2, :] = True
    elif kind == "hollow":
        mask[1:-1, 1:-1] = False
    if bool(rng.integers(0, 2)):
        mask = np.flip(mask, axis=1)
    if bool(rng.integers(0, 2)):
        mask = np.flip(mask, axis=0)
    return mask


def _contact(cells: list[tuple[int, int]], horizontal: tuple[int, ...], vertical: tuple[int, ...]) -> str:
    if any(row in horizontal or col in vertical for row, col in cells):
        return "cross"
    if any(any(abs(row - line) == 1 for line in horizontal) or
           any(abs(col - line) == 1 for line in vertical) for row, col in cells):
        return "edge"
    return "untouched"


def _candidate(
    rng: np.random.Generator, height: int, width: int, desired_shape: str | None,
) -> tuple[list[tuple[int, int]], str]:
    if desired_shape is None:
        kind = str(rng.choice(("rectangle", "rectangle", "rectangle", "L", "step", "cross", "hollow")))
    else:
        kind = desired_shape
    minimum = 4 if kind == "hollow" else 3 if kind != "rectangle" else 1
    maximum_height = min(height - 1, 10)
    maximum_width = min(width - 1, 12)
    if kind == "rectangle":
        shape_height = int(rng.choice((1, 2, 2, 3, 3, 4, 5, 6, 7, 8)))
        shape_width = int(rng.choice((1, 2, 2, 3, 3, 4, 5, 6, 7, 9, 11)))
    else:
        shape_height = int(rng.integers(minimum, maximum_height + 1))
        shape_width = int(rng.integers(minimum, maximum_width + 1))
    shape_height = min(shape_height, maximum_height)
    shape_width = min(shape_width, maximum_width)
    if bool(rng.integers(0, 5) == 0):
        side = int(rng.integers(0, 4))
        row0 = 0 if side == 0 else height - shape_height if side == 1 else int(rng.integers(height - shape_height + 1))
        col0 = 0 if side == 2 else width - shape_width if side == 3 else int(rng.integers(width - shape_width + 1))
    else:
        row0 = int(rng.integers(height - shape_height + 1))
        col0 = int(rng.integers(width - shape_width + 1))
    mask = _shape(rng, shape_height, shape_width, kind)
    cells = [(row0 + int(row), col0 + int(col)) for row, col in np.argwhere(mask)]
    return cells, kind


def _place_objects(
    rng: np.random.Generator, height: int, width: int,
    horizontal: tuple[int, ...], vertical: tuple[int, ...],
    modes: list[str], shapes_enabled: bool, require_l: bool,
) -> list[dict]:
    occupied = np.zeros((height, width), dtype=bool)
    objects = []
    for index, mode in enumerate(modes):
        found = False
        for attempt in range(1200):
            desired_shape = "L" if index == 0 and require_l else None
            if not shapes_enabled and desired_shape is None:
                desired_shape = "rectangle"
            cells, kind = _candidate(rng, height, width, desired_shape)
            if _contact(cells, horizontal, vertical) != mode:
                continue
            if not any(0 < row < height - 1 and 0 < col < width - 1 for row, col in cells):
                continue
            if any((row in (0, height - 1) and col in vertical) or
                   (col in (0, width - 1) and row in horizontal) for row, col in cells):
                continue
            if any(occupied[max(0, row - 1):min(height, row + 2),
                            max(0, col - 1):min(width, col + 2)].any() for row, col in cells):
                continue
            for row, col in cells:
                occupied[row, col] = True
            objects.append({"cells": cells, "kind": kind, "contact": mode})
            found = True
            break
        if not found:
            raise RuntimeError("could not place separated objects")
    return objects


def _pair_dimensions(rng: np.random.Generator, regime: str, index: int, train_sizes: list[tuple[int, int]]) -> tuple[int, int]:
    if regime == "source_train" and index == 0:
        return int(rng.integers(14, 23)), int(rng.integers(15, 25))
    if regime == "target_test" and train_sizes and rng.random() < 0.35:
        return min(30, max(height for height, _ in train_sizes) + int(rng.integers(1, 7))), min(30, max(width for _, width in train_sizes) + int(rng.integers(1, 7)))
    ranges = ((10, 16), (16, 24), (24, 31))
    height_range = ranges[int(rng.choice(3, p=(0.23, 0.43, 0.34)))]
    width_range = ranges[int(rng.choice(3, p=(0.23, 0.43, 0.34)))]
    return int(rng.integers(*height_range)), int(rng.integers(*width_range))


def _line_counts(rng: np.random.Generator, policy: str, regime: str, index: int) -> tuple[int, int]:
    if regime == "source_train" and index == 0:
        return (0, 1) if policy != "horizontal" else (1, 0)
    if regime == "source_train" and index == 1 and policy == "mixed":
        return (1, 0) if bool(rng.integers(0, 2)) else (1, 1)
    total = int(rng.choice((1, 1, 2, 2, 3, 3, 4, 5)))
    if policy == "vertical":
        return 0, total
    if policy == "horizontal":
        return total, 0
    horizontal_count = int(rng.integers(0, total + 1))
    return horizontal_count, total - horizontal_count


def _modes(rng: np.random.Generator, count: int, regime: str, index: int) -> list[str]:
    if regime == "source_train" and index == 0:
        return ["edge", "untouched", "cross"] + [str(rng.choice(("cross", "edge", "untouched"))) for _ in range(count - 3)]
    profile = str(rng.choice(("mixed", "mixed", "cross", "edge", "untouched", "sparse")))
    if profile == "mixed":
        return [str(rng.choice(("cross", "edge", "untouched"))) for _ in range(count)]
    if profile == "sparse":
        return ["edge" if index == 0 else "untouched" for index in range(count)]
    return [profile for _ in range(count)]


def _generate_pair(
    rng: np.random.Generator, regime: str, index: int, policy: str,
    fixed_palette: tuple[int, int, int] | None, shapes_enabled: bool,
    multicolor: bool, train_sizes: list[tuple[int, int]],
) -> dict:
    for _ in range(65):
        height, width = _pair_dimensions(rng, regime, index, train_sizes)
        horizontal_count, vertical_count = _line_counts(rng, policy, regime, index)
        try:
            horizontal = _sample_lines(rng, height, horizontal_count)
            vertical = _sample_lines(rng, width, vertical_count)
        except RuntimeError:
            continue
        count_limit = max(2, min(12, height * width // 38))
        object_count = int(rng.integers(3 if regime == "source_train" and index == 0 else 2, count_limit + 1))
        modes = _modes(rng, object_count, regime, index)
        try:
            objects = _place_objects(
                rng, height, width, horizontal, vertical, modes, shapes_enabled,
                regime == "source_train" and index == 0 and shapes_enabled,
            )
        except RuntimeError:
            continue
        if fixed_palette is None:
            background, payload_base, signal = (int(color) for color in rng.choice(10, 3, replace=False))
        else:
            background, payload_base, signal = fixed_palette
        payload_colors = [payload_base]
        if multicolor and object_count >= 3:
            available = [color for color in range(10) if color not in (background, signal, payload_base)]
            rng.shuffle(available)
            payload_colors += available[:int(rng.integers(1, 3))]
        input_grid = np.full((height, width), background, dtype=np.int8)
        for object_index, obj in enumerate(objects):
            color_index = object_index % len(payload_colors)
            if regime == "source_train" and index == 0 and object_index == 2 and len(payload_colors) == 2:
                color_index = 1
            color = payload_colors[color_index]
            for row, col in obj["cells"]:
                input_grid[row, col] = color
        if int(np.sum(input_grid == background)) <= int(np.sum(input_grid != background)):
            continue
        for row in horizontal:
            input_grid[row, 0] = signal
            input_grid[row, -1] = signal
        for col in vertical:
            input_grid[0, col] = signal
            input_grid[-1, col] = signal
        if int(np.sum(input_grid == background)) <= int(np.sum(input_grid != background)):
            continue
        output_grid = input_grid.copy()
        for row in horizontal:
            output_grid[row, :] = signal
        for col in vertical:
            output_grid[:, col] = signal
        for obj in objects:
            if obj["contact"] != "untouched":
                for row, col in obj["cells"]:
                    output_grid[row, col] = signal
        if np.array_equal(input_grid, output_grid):
            continue
        return {"input": input_grid, "output": output_grid}
    raise RuntimeError("could not generate a pair")


def _signature(grid: np.ndarray) -> tuple:
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed_value = int(seed) % (1 << 64)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    rng = np.random.default_rng(seed_value)
    policy = str(rng.choice(("mixed", "mixed", "mixed", "vertical", "horizontal")))
    shapes_enabled = bool(rng.integers(0, 4) != 0)
    multicolor = bool(rng.integers(0, 3) != 0)
    fixed_palette = None
    if bool(rng.integers(0, 4) == 0):
        fixed_palette = tuple(int(color) for color in rng.choice(10, 3, replace=False))
        multicolor = False
    puzzle = {"train": [], "test": []}
    seen = set()
    train_sizes = []
    for regime, section, count in (("source_train", "train", train_count), ("target_test", "test", test_count)):
        for index in range(count):
            for _ in range(25):
                pair_rng = np.random.default_rng(int(rng.integers(0, 1 << 63)))
                try:
                    pair = _generate_pair(pair_rng, regime, index, policy, fixed_palette, shapes_enabled, multicolor, train_sizes)
                except RuntimeError:
                    continue
                signature = _signature(pair["input"])
                if signature in seen:
                    continue
                seen.add(signature)
                puzzle[section].append(pair)
                if section == "train":
                    train_sizes.append(pair["input"].shape)
                break
            else:
                raise RuntimeError("could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    result = []
    for position in np.argwhere(mask):
        start = tuple(int(value) for value in position)
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        cells = []
        while stack:
            row, col = stack.pop()
            cells.append((row, col))
            for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if 0 <= next_row < height and 0 <= next_col < width and mask[next_row, next_col] and not visited[next_row, next_col]:
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
        result.append(cells)
    return result


def _parse(grid: np.ndarray) -> dict:
    height, width = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    _require(int(np.max(counts)) > int(np.sum(counts) - np.max(counts)), "background must be dominant")
    signal_candidates = []
    for value in colors:
        color = int(value)
        if color == background:
            continue
        cells = {(int(row), int(col)) for row, col in np.argwhere(grid == color)}
        if not all(row in (0, height - 1) or col in (0, width - 1) for row, col in cells):
            continue
        horizontal = tuple(sorted(row for row in range(1, height - 1) if (row, 0) in cells and (row, width - 1) in cells))
        vertical = tuple(sorted(col for col in range(1, width - 1) if (0, col) in cells and (height - 1, col) in cells))
        expected = {(row, col) for row in horizontal for col in (0, width - 1)} | {(row, col) for col in vertical for row in (0, height - 1)}
        if expected == cells and expected:
            signal_candidates.append((color, horizontal, vertical))
    _require(len(signal_candidates) == 1, "signal markers must be uniquely paired")
    signal, horizontal, vertical = signal_candidates[0]
    payload_mask = (grid != background) & (grid != signal)
    objects = _components(payload_mask)
    _require(len(objects) >= 2, "at least two distinct payload objects required")
    labels = np.full(grid.shape, -1, dtype=np.int16)
    for object_index, cells in enumerate(objects):
        for row, col in cells:
            labels[row, col] = object_index
    for cells in objects:
        _require(len({int(grid[row, col]) for row, col in cells}) == 1, "objects must be monochrome")
        _require(any(0 < row < height - 1 and 0 < col < width - 1 for row, col in cells), "boundary object must extend inward")
        _require(all(labels[next_row, next_col] in (-1, labels[row, col]) for row, col in cells for next_row, next_col in ((row - 1, col - 1), (row - 1, col + 1), (row + 1, col - 1), (row + 1, col + 1)) if 0 <= next_row < height and 0 <= next_col < width), "separate objects need a diagonal gap")
    return {"background": background, "signal": signal, "horizontal": horizontal, "vertical": vertical, "objects": objects}


def _expected(grid: np.ndarray, parsed: dict) -> np.ndarray:
    result = grid.copy()
    signal = parsed["signal"]
    for row in parsed["horizontal"]:
        result[row, :] = signal
    for col in parsed["vertical"]:
        result[:, col] = signal
    for cells in parsed["objects"]:
        if _contact(cells, parsed["horizontal"], parsed["vertical"]) != "untouched":
            for row, col in cells:
                result[row, col] = signal
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    train = puzzle["train"]
    test = puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "pair lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "pair counts")
    seen = set()
    parsed_pairs = {"train": [], "test": []}
    for section, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid array")
                _require(np.issubdtype(grid.dtype, np.integer), "integer grid")
                _require(all(1 <= length <= 30 for length in grid.shape), "grid dimensions")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), "ARC colors")
            _require(input_grid.shape == output_grid.shape, "same dimensions")
            _require(_signature(input_grid) not in seen, "unique inputs")
            seen.add(_signature(input_grid))
            digest = hashlib.sha256(np.asarray(input_grid, dtype=np.int8).tobytes() + np.asarray(output_grid, dtype=np.int8).tobytes()).hexdigest()
            _require(digest not in _SOURCE_DIGESTS, "no source copies")
            parsed = _parse(input_grid)
            _require(np.array_equal(_expected(input_grid, parsed), output_grid), "line and whole-object contact rule")
            _require(not np.array_equal(input_grid, output_grid), "line completion must change grid")
            parsed_pairs[section].append(parsed)
    training = parsed_pairs["train"]
    _require(any(_contact(cells, parsed["horizontal"], parsed["vertical"]) == "edge" for parsed in training for cells in parsed["objects"]), "train edge witness")
    _require(any(_contact(cells, parsed["horizontal"], parsed["vertical"]) == "untouched" for parsed in training for cells in parsed["objects"]), "train untouched witness")
    train_orientations = {orientation for parsed in training for orientation, lines in (("horizontal", parsed["horizontal"]), ("vertical", parsed["vertical"])) if lines}
    test_orientations = {orientation for parsed in parsed_pairs["test"] for orientation, lines in (("horizontal", parsed["horizontal"]), ("vertical", parsed["vertical"])) if lines}
    _require(test_orientations <= train_orientations, "train demonstrates test line orientations")
    test_nonrect = any(len(cells) < (max(row for row, _ in cells) - min(row for row, _ in cells) + 1) * (max(col for _, col in cells) - min(col for _, col in cells) + 1) for parsed in parsed_pairs["test"] for cells in parsed["objects"])
    if test_nonrect:
        _require(any(len(cells) < (max(row for row, _ in cells) - min(row for row, _ in cells) + 1) * (max(col for _, col in cells) - min(col for _, col in cells) + 1) and _contact(cells, parsed["horizontal"], parsed["vertical"]) != "untouched" for parsed in training for cells in parsed["objects"]), "train demonstrates nonrectangular recoloring")
    if any(len({int(pair["input"][row, col]) for cells in parsed["objects"] for row, col in cells}) > 1 for pair, parsed in zip(test, parsed_pairs["test"])):
        _require(any(len({int(pair["input"][row, col]) for cells in parsed["objects"] if _contact(cells, parsed["horizontal"], parsed["vertical"]) != "untouched" for row, col in cells}) > 1 for pair, parsed in zip(train, training)), "train recolors multiple payload colors")


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple((_signature(pair["input"]), _signature(pair["output"])) for section in ("train", "test") for pair in puzzle[section])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_signature(puzzle) for puzzle in puzzles}) == 42, "cross-seed diversity")
    _require(len({len(puzzle["train"]) for puzzle in puzzles}) >= 4, "automatic train counts")
    _require(len({len(puzzle["test"]) for puzzle in puzzles}) >= 3, "automatic test counts")
    dimensions = set()
    orientations = set()
    palettes = set()
    boundary_sides = set()
    nonrect = multicolor = large_object = thin_object = False
    contacts = set()
    line_totals = set()
    object_totals = set()
    spacings = set()
    outcomes = set()
    test_scales = set()
    near_boundary_line = empty_and_busy = combined = False
    varied_line_layouts = varied_object_counts = varied_palettes = stable_palettes = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        pair_layouts = set()
        pair_object_counts = set()
        pair_palettes = set()
        train_areas = [int(np.prod(pair["input"].shape)) for pair in puzzle["train"]]
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid = pair["input"]
                parsed = _parse(grid)
                _require(np.array_equal(_expected(grid, parsed), pair["output"]), "independent output construction")
                dimensions.add(grid.shape)
                horizontal = parsed["horizontal"]
                vertical = parsed["vertical"]
                objects = parsed["objects"]
                orientations.add((bool(horizontal), bool(vertical)))
                palettes.add((parsed["background"], parsed["signal"]))
                line_totals.add(len(horizontal) + len(vertical))
                object_totals.add(len(objects))
                pair_layouts.add((len(horizontal), len(vertical)))
                pair_object_counts.add(len(objects))
                pair_palettes.add(tuple(int(color) for color in np.unique(grid)))
                if section == "test":
                    area = int(np.prod(grid.shape))
                    test_scales.add("larger" if area > max(train_areas) else "smaller" if area < min(train_areas) else "comparable")
                for positions, length in ((horizontal, grid.shape[0]), (vertical, grid.shape[1])):
                    near_boundary_line |= any(min(position, length - 1 - position) <= 2 for position in positions)
                    spacings.update(second - first for first, second in zip(positions, positions[1:]))
                kinds = [_contact(cells, horizontal, vertical) for cells in objects]
                contacts.update(kinds)
                if all(kind == "untouched" for kind in kinds):
                    outcomes.add("none")
                if all(kind != "untouched" for kind in kinds):
                    outcomes.add("all")
                for kind in contacts:
                    if kinds.count(kind) > len(kinds) / 2:
                        outcomes.add(kind + "_dominated")
                line_contacts = [sum(any(abs((row if orientation == "horizontal" else col) - position) <= 1 for row, col in cells) for cells in objects) for orientation, positions in (("horizontal", horizontal), ("vertical", vertical)) for position in positions]
                empty_and_busy |= 0 in line_contacts and any(value >= 2 for value in line_contacts)
                pair_nonrect = False
                pair_boundary = False
                for cells in objects:
                    rows = [row for row, _ in cells]
                    cols = [col for _, col in cells]
                    object_height = max(rows) - min(rows) + 1
                    object_width = max(cols) - min(cols) + 1
                    pair_nonrect |= len(cells) < object_height * object_width
                    large_object |= max(object_height, object_width) >= 9
                    thin_object |= min(object_height, object_width) == 1
                    for side, touches in (("top", 0 in rows), ("bottom", grid.shape[0] - 1 in rows), ("left", 0 in cols), ("right", grid.shape[1] - 1 in cols)):
                        if touches:
                            boundary_sides.add(side)
                            pair_boundary = True
                nonrect |= pair_nonrect
                pair_multicolor = len({int(grid[row, col]) for cells in objects for row, col in cells}) > 1
                multicolor |= pair_multicolor
                combined |= pair_nonrect and pair_boundary and pair_multicolor and bool(horizontal) and bool(vertical)
        varied_line_layouts += len(pair_layouts) > 1
        varied_object_counts += len(pair_object_counts) > 1
        varied_palettes += len(pair_palettes) > 1
        stable_palettes += len(pair_palettes) == 1
    _require(len(dimensions) >= 45 and len(palettes) >= 20, "size and color diversity")
    _require({(True, False), (False, True), (True, True)} <= orientations, "orientation diversity")
    _require(contacts == {"cross", "edge", "untouched"}, "contact diversity")
    _require(boundary_sides == {"top", "bottom", "left", "right"}, "boundary objects on all sides")
    _require(nonrect and multicolor and large_object and thin_object, "shape and color extensions")
    _require({1, 2, 3, 4, 5} <= line_totals and len(object_totals) >= 8, "line and object count diversity")
    _require({2, 3, 6} <= spacings and near_boundary_line, "line spacing diversity")
    _require({"none", "all", "cross_dominated", "edge_dominated", "untouched_dominated"} <= outcomes, "contact outcome diversity")
    _require(test_scales == {"smaller", "comparable", "larger"}, "test scale diversity")
    _require(empty_and_busy and combined, "combined review axes")
    _require(varied_line_layouts >= 30 and varied_object_counts >= 30, "within-puzzle geometry variation")
    _require(varied_palettes >= 20 and stable_palettes >= 1, "local and stable palette puzzles")
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(137, train_count, test_count))
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(137, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first = generate_puzzle(191, 4, 2)
    second = generate_puzzle(191, 4, 2)
    _require(_puzzle_signature(first) == _puzzle_signature(second), "determinism")
