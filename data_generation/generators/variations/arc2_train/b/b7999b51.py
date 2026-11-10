import numpy as np


_SOURCE_PAIRS = {
    (
        "000000000000000/002222222000000/002222222000000/"
        "002223333330000/000000000000000/000111110000000/"
        "000111110000000/000000000000000/000000000000000/"
        "000000000000000",
        "213/210/200",
    ),
    (
        "000000000000000000/000000000000000000/001111100000000000/"
        "001222222222222000/001111100000000000/000000000000000000/"
        "000088888880000000/000088888880000000/000088777777770000/"
        "000088777777770000/000088777777770000/000000777777770000/"
        "000000000000000000/000000000000000000/000000000000000000/"
        "000000000000000000/000000000000000000/000000000000000000/"
        "000000000000000000",
        "8712/8710/8710/8700/8000",
    ),
    (
        "000000000000000/088880000000000/088880000000000/"
        "088880000000000/000000000000000/004444440000000/"
        "033334440000000/004444440000000/004444440000000/"
        "000000000000000",
        "483/480/480/400",
    ),
    (
        "000000000000000000/000000000000000000/022222200111100000/"
        "022233333111100000/022233333300000000/000033333300000000/"
        "066666688888000000/066666633300000000/066666600000000000/"
        "066666600000000000/066666600000000000/066666600000000000",
        "63218/63210/63200/63000/63000/60000",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _resolve_counts(rng, num_train, num_test):
    for name, value, lower, upper in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer or None")
        if not lower <= int(value) <= upper:
            raise ValueError(f"{name} must be between {lower} and {upper}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid_key(grid):
    return tuple(grid.shape), tuple(int(value) for value in grid.flat)


def _grid_text(grid):
    return "/".join("".join(str(int(value)) for value in row) for row in grid)


def _background(grid):
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    value = int(border[0])
    _require(bool(np.all(border == value)), "background border is not uniform")
    colors, counts = np.unique(grid, return_counts=True)
    _require(value == int(colors[int(np.argmax(counts))]), "background is not the majority")
    return value


def _color_information(grid, background):
    information = {}
    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color == background:
            continue
        rows, columns = np.where(grid == color)
        information[color] = (
            int(rows.min()), int(columns.min()),
            int(rows.max()) + 1, int(columns.max()) + 1,
        )
    return information


def _chart(grid, background):
    information = _color_information(grid, background)
    heights = {color: box[2] - box[0] for color, box in information.items()}
    _require(len(heights) >= 2, "scene needs at least two rectangles")
    _require(len(set(heights.values())) == len(heights), "rectangle heights must differ")
    ordered = sorted(heights, key=lambda color: -heights[color])
    output = np.full((max(heights.values()), len(ordered)), background, dtype=np.int8)
    for column, color in enumerate(ordered):
        output[:heights[color], column] = color
    return output


def _layering_is_valid(grid, background, information):
    edges = {color: set() for color in information}
    for color, (top, left, bottom, right) in information.items():
        window = grid[top:bottom, left:right]
        if np.any(window == background):
            return False
        edges[color].update(int(other) for other in np.unique(window) if int(other) != color)
    indegree = {color: 0 for color in information}
    for covered in edges:
        for covering in edges[covered]:
            indegree[covering] += 1
    ready = sorted(color for color in indegree if indegree[color] == 0)
    order = []
    while ready:
        color = ready.pop(0)
        order.append(color)
        for covering in sorted(edges[color]):
            indegree[covering] -= 1
            if indegree[covering] == 0:
                ready.append(covering)
                ready.sort()
    if len(order) != len(information):
        return False
    reconstructed = np.full(grid.shape, background, dtype=np.int8)
    for color in order:
        top, left, bottom, right = information[color]
        reconstructed[top:bottom, left:right] = color
    return bool(np.array_equal(reconstructed, grid))


def _overlap_features(information):
    boxes = list(information.values())
    areas = []
    for index, first in enumerate(boxes):
        for second in boxes[:index]:
            depth = min(first[2], second[2]) - max(first[0], second[0])
            width = min(first[3], second[3]) - max(first[1], second[1])
            if depth > 0 and width > 0:
                areas.append(depth * width)
    return areas


def _hidden_rows(grid, information):
    return sum(
        not np.any(grid[row] == color)
        for color, box in information.items()
        for row in range(box[0] + 1, box[2] - 1)
    )


def _component_count(grid, background):
    foreground = grid != background
    seen = np.zeros(grid.shape, dtype=bool)
    count = 0
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            if not foreground[row, column] or seen[row, column]:
                continue
            count += 1
            stack = [(row, column)]
            seen[row, column] = True
            while stack:
                current_row, current_column = stack.pop()
                for next_row, next_column in (
                    (current_row - 1, current_column),
                    (current_row + 1, current_column),
                    (current_row, current_column - 1),
                    (current_row, current_column + 1),
                ):
                    if (
                        0 <= next_row < grid.shape[0]
                        and 0 <= next_column < grid.shape[1]
                        and foreground[next_row, next_column]
                        and not seen[next_row, next_column]
                    ):
                        seen[next_row, next_column] = True
                        stack.append((next_row, next_column))
    return count


def _review_features(grid, background, information):
    heights = {color: box[2] - box[0] for color, box in information.items()}
    widths = {color: box[3] - box[1] for color, box in information.items()}
    tallest = max(heights, key=heights.get)
    shortest = min(heights, key=heights.get)
    tallest_box = information[tallest]
    horizontal_heights = [
        heights[color] for color in sorted(information, key=lambda color: information[color][1])
    ]
    features = {
        "interior": False,
        "near_hidden": False,
        "joint_hidden": False,
        "tallest_split": False,
        "nonshort_coverer": False,
        "tall_coverer": False,
        "square": any(widths[color] == heights[color] for color in information),
        "wide": any(widths[color] > heights[color] for color in information),
        "separate": _component_count(grid, background) > 1,
        "tallest_outside_group": not any(
            min(tallest_box[2], other[2]) > max(tallest_box[0], other[0])
            and min(tallest_box[3], other[3]) > max(tallest_box[1], other[1])
            for color, other in information.items() if color != tallest
        ),
        "rank_not_spatial": (
            len(horizontal_heights) >= 3
            and horizontal_heights != sorted(horizontal_heights)
            and horizontal_heights != sorted(horizontal_heights, reverse=True)
        ),
        "narrow_tallest_wide_shortest": widths[tallest] <= 2 and widths[shortest] >= 7,
        "vertically_stacked": any(
            (first[2] <= second[0] or second[2] <= first[0])
            and first[1] < second[3] and second[1] < first[3]
            for first in information.values() for second in information.values()
            if first != second
        ),
    }
    for color, (top, left, bottom, right) in information.items():
        for row in range(top + 1, bottom - 1):
            strip = grid[row, left:right]
            covered = int(np.count_nonzero(strip != color))
            if covered:
                features["interior"] = True
            if covered >= 0.75 * (right - left):
                features["near_hidden"] = True
            if np.any(strip == color):
                continue
            covering = set(int(value) for value in strip) - {background, color}
            features["joint_hidden"] |= len(covering) >= 2
            features["tallest_split"] |= heights[color] == max(heights.values())
            features["nonshort_coverer"] |= any(
                heights[other] > min(heights.values()) for other in covering
            )
            features["tall_coverer"] |= any(heights[other] > 1 for other in covering)
    return features


def _width(rng, height):
    if height <= 2 and rng.random() < 0.4:
        return int(rng.integers(7, 15))
    return int(rng.choice((1, 2, 3, 4, 5, 6, 7, 8, 10, 12)))


def _sample_profile(rng, regime):
    if regime == "source_train":
        count = int(rng.integers(2, 7))
        maximum = int(rng.integers(max(3, count), 10))
    elif rng.random() < 0.36:
        count = int(rng.integers(2, 5))
        maximum = int(rng.integers(max(3, count), 7))
    else:
        count = int(rng.integers(4, 9))
        maximum = int(rng.integers(max(5, count), 12))
    remaining = rng.choice(np.arange(1, maximum), size=count - 1, replace=False)
    heights = [maximum] + [int(value) for value in remaining]
    rng.shuffle(heights)
    return heights


def _primary_indices(rng, heights, mode):
    indices = list(range(len(heights)))
    rng.shuffle(indices)
    if mode == "joint_split":
        for victim in indices:
            candidates = [index for index in indices if index != victim and heights[index] <= heights[victim] - 2]
            if heights[victim] >= 5 and len(candidates) >= 2:
                return victim, candidates[:2]
    if mode in ("split", "interior"):
        for victim in indices:
            candidates = [index for index in indices if index != victim and heights[index] <= heights[victim] - 2]
            if heights[victim] >= 3 and candidates:
                return victim, candidates[:1]
    return indices[0], indices[1:2]


def _relative_position(rng, anchor, height, width, style):
    top, left, anchor_height, anchor_width = anchor
    if style == "edge":
        horizontal = int(rng.integers(1, min(3, width, anchor_width) + 1))
        if rng.random() < 0.5:
            return top + anchor_height - 1, left + anchor_width - horizontal
        return top - height + 1, left + anchor_width - horizontal
    if style == "broad":
        vertical = min(height, anchor_height)
        horizontal = min(width, anchor_width)
        return (
            top + int(rng.integers(-height + max(1, vertical // 2), anchor_height - max(1, vertical // 2) + 1)),
            left + int(rng.integers(-width + max(1, horizontal // 2), anchor_width - max(1, horizontal // 2) + 1)),
        )
    return (
        top + int(rng.integers(-height + 1, anchor_height)),
        left + int(rng.integers(-width + 1, anchor_width)),
    )


def _construct_scene(rng, background, regime, heights):
    count = len(heights)
    colors = [int(value) for value in rng.choice(
        [color for color in range(10) if color != background], size=count, replace=False
    )]
    modes = ("edge", "interior", "broad", "split", "joint_split")
    weights = (0.21, 0.26, 0.22, 0.19, 0.12) if regime == "source_train" else (0.12, 0.20, 0.24, 0.26, 0.18)
    mode = str(rng.choice(modes, p=weights))
    victim, covering = _primary_indices(rng, heights, mode)
    if mode == "joint_split" and len(covering) != 2:
        mode = "broad"
    if mode in ("split", "interior") and heights[covering[0]] > heights[victim] - 2:
        mode = "broad"
    widths = [_width(rng, height) for height in heights]
    if mode in ("split", "joint_split"):
        widths[victim] = int(rng.integers(3, 9))
    rectangles = [None] * count
    rectangles[victim] = (0, 0, heights[victim], widths[victim])
    order = [victim]
    if mode == "joint_split":
        first, second = covering
        row = int(rng.integers(1, heights[victim] - max(heights[first], heights[second]) + 1))
        partition = int(rng.integers(1, widths[victim]))
        first_left = -int(rng.integers(0, 3))
        widths[first] = partition - first_left
        widths[second] = widths[victim] - partition + int(rng.integers(0, 3))
        rectangles[first] = (row, first_left, heights[first], widths[first])
        rectangles[second] = (row, partition, heights[second], widths[second])
        order.extend((first, second))
    else:
        cover = covering[0]
        if mode == "split":
            widths[cover] = widths[victim] + int(rng.integers(0, 4))
            top = int(rng.integers(1, heights[victim] - heights[cover]))
            left = -int(rng.integers(0, widths[cover] - widths[victim] + 1))
        elif mode == "interior":
            top = int(rng.integers(1, heights[victim] - heights[cover]))
            left = int(rng.integers(-widths[cover] + 1, widths[victim]))
        else:
            top, left = _relative_position(
                rng, rectangles[victim], heights[cover], widths[cover], mode
            )
        rectangles[cover] = (top, left, heights[cover], widths[cover])
        order.append(cover)
    extras = [index for index in range(count) if index not in order]
    rng.shuffle(extras)
    for index in extras:
        if rng.random() < (0.72 if regime == "target_test" else 0.55):
            anchor_index = int(rng.choice(order))
            style = str(rng.choice(("edge", "broad", "random")))
            top, left = _relative_position(
                rng, rectangles[anchor_index], heights[index], widths[index], style
            )
        else:
            top = int(rng.integers(-8, 9))
            left = int(rng.integers(-8, 9))
        rectangles[index] = (top, left, heights[index], widths[index])
        order.append(index)
    top_bound = min(rectangle[0] for rectangle in rectangles)
    left_bound = min(rectangle[1] for rectangle in rectangles)
    bottom_bound = max(rectangle[0] + rectangle[2] for rectangle in rectangles)
    right_bound = max(rectangle[1] + rectangle[3] for rectangle in rectangles)
    if bottom_bound - top_bound > 28 or right_bound - left_bound > 28:
        return None
    roomy = rng.random() < 0.48
    padding = int(rng.integers(3, 11)) if roomy else int(rng.integers(1, 3))
    top_padding = min(padding, max(1, (30 - (bottom_bound - top_bound)) // 2))
    left_padding = min(padding, max(1, (30 - (right_bound - left_bound)) // 2))
    bottom_padding = min(int(rng.integers(1, padding + 1)), 30 - (bottom_bound - top_bound) - top_padding)
    right_padding = min(int(rng.integers(1, padding + 1)), 30 - (right_bound - left_bound) - left_padding)
    if min(bottom_padding, right_padding) < 1:
        return None
    input_grid = np.full(
        (bottom_bound - top_bound + top_padding + bottom_padding,
         right_bound - left_bound + left_padding + right_padding),
        background, dtype=np.int8
    )
    shifted = {}
    for index, (top, left, height, width) in enumerate(rectangles):
        shifted[index] = (
            top - top_bound + top_padding, left - left_bound + left_padding,
            height, width,
        )
    for index in order:
        top, left, height, width = shifted[index]
        input_grid[top:top + height, left:left + width] = colors[index]
    information = _color_information(input_grid, background)
    if len(information) != count:
        return None
    for index, (top, left, height, width) in shifted.items():
        if information[colors[index]] != (top, left, top + height, left + width):
            return None
    if not _layering_is_valid(input_grid, background, information):
        return None
    if not _overlap_features(information):
        return None
    if mode in ("split", "joint_split") and _hidden_rows(input_grid, information) == 0:
        return None
    try:
        if _background(input_grid) != background:
            return None
        output_grid = _chart(input_grid, background)
    except AssertionError:
        return None
    return {"input": input_grid, "output": output_grid}


def _fallback_scene(rng, background, require_split=False):
    colors = [int(value) for value in rng.choice(
        [color for color in range(10) if color != background], size=2, replace=False
    )]
    first_height = int(rng.integers(5, 9)) if require_split else int(rng.integers(4, 9))
    second_height = 1 if require_split else 2
    first_width = int(rng.integers(4, 8))
    grid = np.full((first_height + second_height + 3, first_width + 7), background, dtype=np.int8)
    grid[1:1 + first_height, 1:1 + first_width] = colors[0]
    if require_split:
        grid[2:3, 1:1 + first_width] = colors[1]
    else:
        grid[2:4, first_width - 1:first_width + 3] = colors[1]
    return {"input": grid, "output": _chart(grid, background)}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    background = 0 if rng.random() < 0.76 else int(rng.integers(1, 10))
    source_progression = rng.random() < 0.28
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count, regime in (
        ("train", train_count, "source_train"),
        ("test", test_count, "target_test"),
    ):
        for pair_index in range(count):
            pair = None
            for _ in range(3):
                heights = _sample_profile(rng, regime)
                for _ in range(120):
                    candidate = _construct_scene(rng, background, regime, heights)
                    if candidate is None:
                        continue
                    information = _color_information(candidate["input"], background)
                    if split == "train" and pair_index == 0:
                        if not _review_features(candidate["input"], background, information)["interior"]:
                            continue
                        if max(_overlap_features(information)) < 4:
                            continue
                    if source_progression:
                        has_gap = _hidden_rows(candidate["input"], information) > 0
                        if (split == "train" and has_gap) or (
                            split == "test" and pair_index == 0 and not has_gap
                        ):
                            continue
                    key = _grid_key(candidate["input"])
                    if key in seen or (
                        _grid_text(candidate["input"]), _grid_text(candidate["output"])
                    ) in _SOURCE_PAIRS:
                        continue
                    pair = candidate
                    break
                if pair is not None:
                    break
            if pair is None:
                for _ in range(20):
                    candidate = _fallback_scene(
                        rng, background,
                        require_split=source_progression and split == "test" and pair_index == 0,
                    )
                    key = _grid_key(candidate["input"])
                    if key not in seen:
                        pair = candidate
                        break
            if pair is None:
                raise RuntimeError("could not construct a unique rectangle scene")
            seen.add(_grid_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
        _require(all(isinstance(puzzle[split], list) for split in ("train", "test")), "splits must be lists")
        train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
        _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "invalid pair counts")
        backgrounds = set()
        seen_inputs = set()
        seen_pairs = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
                for grid in pair.values():
                    _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D array")
                    _require(all(1 <= side <= 30 for side in grid.shape), "invalid grid dimensions")
                    _require(np.issubdtype(grid.dtype, np.integer), "noninteger grid")
                    _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, "invalid ARC color")
                input_grid, output_grid = pair["input"], pair["output"]
                input_key = _grid_key(input_grid)
                pair_key = (input_key, _grid_key(output_grid))
                _require(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate generated pair")
                seen_inputs.add(input_key)
                seen_pairs.add(pair_key)
                _require((_grid_text(input_grid), _grid_text(output_grid)) not in _SOURCE_PAIRS, "source pair copied")
                background = _background(input_grid)
                backgrounds.add(background)
                information = _color_information(input_grid, background)
                _require(2 <= len(information) <= 8, "invalid rectangle count")
                _require(_layering_is_valid(input_grid, background, information), "scene is not recoverable layered rectangles")
                _require(bool(_overlap_features(information)), "scene lacks occlusion")
                expected = _chart(input_grid, background)
                _require(np.array_equal(output_grid, expected), "incorrect unary height chart")
                _require(not np.array_equal(input_grid, output_grid), "input equals output")
        _require(len(backgrounds) == 1, "background convention changes within puzzle")
        learned_background = _background(puzzle["train"][0]["input"])
        first_training_grid = puzzle["train"][0]["input"]
        first_training_information = _color_information(first_training_grid, learned_background)
        _require(
            _review_features(first_training_grid, learned_background, first_training_information)["interior"]
            and max(_overlap_features(first_training_information)) >= 4,
            "training does not demonstrate an interior crossing",
        )
        for pair in puzzle["train"]:
            _require(_background(pair["input"]) == learned_background, "training background disagrees")
            _require(np.array_equal(_chart(pair["input"], learned_background), pair["output"]), "training does not teach chart")
        for pair in puzzle["test"]:
            _require(_background(pair["input"]) == learned_background, "test background not taught")
            predicted = _chart(pair["input"], learned_background)
            _require(np.array_equal(predicted, pair["output"]), "test not predictable from training convention and input")
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"malformed generated puzzle: {exc}") from exc


def _puzzle_key(puzzle):
    return tuple(
        (split, tuple((_grid_key(pair["input"]), _grid_key(pair["output"])) for pair in puzzle[split]))
        for split in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(puzzle["train"]) for puzzle in puzzles}) >= 4, "too few train counts")
    _require(len({len(puzzle["test"]) for puzzle in puzzles}) >= 3, "too few test counts")
    _require(len({_puzzle_key(puzzle) for puzzle in puzzles}) == 42, "cross-seed puzzles repeat")
    _require(_puzzle_key(generate_puzzle(12345, 7, 3)) == _puzzle_key(generate_puzzle(12345, 7, 3)), "nondeterministic")
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (6, 2)):
        puzzle = generate_puzzle(1000 + train_count * 10 + test_count, train_count, test_count)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count), "explicit counts ignored")
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2000 + (train_count or 0) * 10 + (test_count or 0), train_count, test_count)
        _require(train_count is None or len(puzzle["train"]) == train_count, "partial train count ignored")
        _require(test_count is None or len(puzzle["test"]) == test_count, "partial test count ignored")
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(999, train_count, test_count)
        except ValueError:
            continue
        raise AssertionError("invalid count accepted")
    features = []
    within_shapes = 0
    within_profiles = 0
    within_counts = 0
    within_components = 0
    reserved_test_splits = 0
    train_feature_counts = {name: 0 for name in (
        "interior", "near_hidden", "joint_hidden", "tallest_split",
        "nonshort_coverer", "tall_coverer", "square", "wide", "separate",
        "tallest_outside_group", "rank_not_spatial",
        "narrow_tallest_wide_shortest", "vertically_stacked",
    )}
    test_feature_counts = dict(train_feature_counts)
    demonstrated_count_tests = 0
    extrapolated_count_tests = 0
    demonstrated_height_tests = 0
    extrapolated_height_tests = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        shapes = set()
        profiles = set()
        object_counts = set()
        component_counts = set()
        split_counts = {"train": [], "test": []}
        height_maxima = {"train": [], "test": []}
        split_presence = {"train": False, "test": False}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                background = _background(grid)
                information = _color_information(grid, background)
                profile = tuple(sorted((box[2] - box[0] for box in information.values()), reverse=True))
                shapes.add(grid.shape)
                profiles.add(profile)
                object_counts.add(len(information))
                component_counts.add(_component_count(grid, background))
                split_counts[split].append(len(information))
                height_maxima[split].append(profile[0])
                review = _review_features(grid, background, information)
                split_presence[split] |= _hidden_rows(grid, information) > 0
                target_counts = train_feature_counts if split == "train" else test_feature_counts
                for name, present in review.items():
                    target_counts[name] += bool(present)
                areas = _overlap_features(information)
                widths = [box[3] - box[1] for box in information.values()]
                features.append((
                    split, grid.shape, len(information), profile, background,
                    max(areas), len(areas), _hidden_rows(grid, information),
                    max(widths), min(widths),
                ))
        within_shapes += len(shapes) > 1
        within_profiles += len(profiles) > 1
        within_counts += len(object_counts) > 1
        within_components += len(component_counts) > 1
        demonstrated_count_tests += max(split_counts["test"]) <= max(split_counts["train"])
        extrapolated_count_tests += max(split_counts["test"]) > max(split_counts["train"])
        demonstrated_height_tests += max(height_maxima["test"]) <= max(height_maxima["train"])
        extrapolated_height_tests += max(height_maxima["test"]) > max(height_maxima["train"])
        reserved_test_splits += not split_presence["train"] and split_presence["test"]
    _require(within_shapes >= 35 and within_profiles >= 35, "weak within-puzzle variation")
    _require(within_counts >= 30 and within_components >= 20, "weak within-puzzle arrangement variation")
    _require(len({item[1] for item in features}) >= 70, "weak canvas diversity")
    _require(len({item[3] for item in features}) >= 65, "weak height-profile diversity")
    _require(len({item[2] for item in features}) >= 6, "weak rectangle-count diversity")
    _require(len({item[4] for item in features}) >= 3, "weak background diversity")
    _require(any(item[5] >= 6 for item in features), "no broad overlap")
    _require(any(item[6] >= 3 for item in features), "no multi-overlap scene")
    _require(any(item[7] > 0 and item[0] == "train" for item in features), "no training split")
    _require(any(item[7] > 0 and item[0] == "test" for item in features), "no test split")
    _require(any(item[8] >= 10 for item in features), "no long bar")
    _require(any(item[9] <= 2 for item in features), "no narrow rectangle")
    _require(any(item[3][-1] > 1 for item in features), "all profiles end at height one")
    _require(any(item[3][0] - item[3][1] >= 2 for item in features), "all tallest heights consecutive")
    _require(any(item[1][0] <= 12 or item[1][1] <= 12 for item in features), "no tight canvas")
    _require(any(item[1][0] >= 25 or item[1][1] >= 25 for item in features), "no roomy canvas")
    _require(all(train_feature_counts[name] > 0 for name in train_feature_counts), "missing training review feature")
    _require(all(test_feature_counts[name] > 0 for name in test_feature_counts), "missing test review feature")
    _require(min(
        demonstrated_count_tests, extrapolated_count_tests,
        demonstrated_height_tests, extrapolated_height_tests,
    ) >= 3, "train/test regimes do not both overlap and extrapolate")
    _require(reserved_test_splits >= 3, "no source-like withheld split progression")
