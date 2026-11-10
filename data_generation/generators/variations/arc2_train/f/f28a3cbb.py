import numpy as np


_CARDINAL = "cardinal"
_DIAGONAL = "diagonal"

_SHAPES = {
    1: (
        ((0, 0),),
    ),
    2: (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ),
    3: (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ),
    4: (
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 2)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 1), (2, 1)),
    ),
}


def _array_from_rows(rows):
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = (
    (
        _array_from_rows((
            "999666694",
            "999666446",
            "999669666",
            "666666666",
            "669666666",
            "696666666",
            "966666444",
            "664666444",
            "666466444",
        )),
        _array_from_rows((
            "999966666",
            "999666666",
            "999966666",
            "999666666",
            "666666666",
            "666666444",
            "666666444",
            "666664444",
            "666664444",
        )),
    ),
    (
        _array_from_rows((
            "222666662",
            "222666662",
            "222666666",
            "666626666",
            "626666656",
            "666656666",
            "666666555",
            "566666555",
            "662666555",
        )),
        _array_from_rows((
            "222266666",
            "222266666",
            "222266666",
            "622666666",
            "666666666",
            "666666656",
            "666665555",
            "666665555",
            "666666555",
        )),
    ),
    (
        _array_from_rows((
            "333666366",
            "333636666",
            "333666666",
            "666666888",
            "666666666",
            "636686666",
            "666666888",
            "686666888",
            "666866888",
        )),
        _array_from_rows((
            "333366666",
            "333366666",
            "333666666",
            "636666666",
            "666666666",
            "666666888",
            "666668888",
            "666668888",
            "666668888",
        )),
    ),
)


def _corner_box(height, width, corner):
    """
    Corner numbering:
        0 = top-left
        1 = top-right
        2 = bottom-left
        3 = bottom-right
    """
    top = corner in (0, 1)
    left = corner in (0, 2)

    row_start = 0 if top else height - 3
    col_start = 0 if left else width - 3

    return (
        row_start,
        row_start + 2,
        col_start,
        col_start + 2,
    )


def _box_cells(box):
    row_start, row_end, col_start, col_end = box
    return frozenset(
        (row, col)
        for row in range(row_start, row_end + 1)
        for col in range(col_start, col_end + 1)
    )


def _touches(cells, anchor):
    for row, col in cells:
        if (
            (row - 1, col) in anchor
            or (row + 1, col) in anchor
            or (row, col - 1) in anchor
            or (row, col + 1) in anchor
        ):
            return True
    return False


def _infer_direction(cells, anchor_box):
    """
    Infer the approach direction from the fragment's spatial sector.

    A fragment must lie wholly above, within, or below the anchor's row
    interval, and wholly left of, within, or right of its column interval.
    The resulting non-central sector uniquely determines one of eight
    cardinal or diagonal directions toward the anchor.
    """
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    row_start, row_end, col_start, col_end = anchor_box

    if max(rows) < row_start:
        row_sector = -1
    elif min(rows) > row_end:
        row_sector = 1
    elif min(rows) >= row_start and max(rows) <= row_end:
        row_sector = 0
    else:
        return None

    if max(cols) < col_start:
        col_sector = -1
    elif min(cols) > col_end:
        col_sector = 1
    elif min(cols) >= col_start and max(cols) <= col_end:
        col_sector = 0
    else:
        return None

    if row_sector == 0 and col_sector == 0:
        return None

    return -row_sector, -col_sector


def _dock_component(
    cells,
    direction,
    own_anchor,
    all_anchors,
    height,
    width,
):
    """
    Move a fragment until its first edge contact with its own anchor.

    Returns:
        output_cells, path, swept_cells

    Returns None if the fragment leaves the grid, collides with an anchor,
    or reaches the other anchor system.
    """
    current = frozenset(cells)
    other_anchors = all_anchors - own_anchor

    if (
        current & all_anchors
        or _touches(current, own_anchor)
        or _touches(current, other_anchors)
    ):
        return None

    path = [current]
    delta_row, delta_col = direction

    for _ in range(height + width + 3):
        moved = frozenset(
            (row + delta_row, col + delta_col)
            for row, col in current
        )

        if any(
            row < 0
            or row >= height
            or col < 0
            or col >= width
            for row, col in moved
        ):
            return None

        if moved & all_anchors:
            return None

        if _touches(moved, other_anchors):
            return None

        path.append(moved)

        if _touches(moved, own_anchor):
            swept_cells = frozenset().union(*path)
            return moved, tuple(path), swept_cells

        current = moved

    return None


def _build_libraries(height, width, anchor_boxes):
    """
    Enumerate all valid fragment placements for both anchor systems.
    """
    anchors = [_box_cells(box) for box in anchor_boxes]
    all_anchors = anchors[0] | anchors[1]
    libraries = [[], []]

    for color_index in range(2):
        own_anchor = anchors[color_index]
        own_box = anchor_boxes[color_index]

        for size in (1, 2, 3, 4):
            for shape in _SHAPES[size]:
                shape_height = max(row for row, _ in shape) + 1
                shape_width = max(col for _, col in shape) + 1

                for row_start in range(height - shape_height + 1):
                    for col_start in range(width - shape_width + 1):
                        input_cells = frozenset(
                            (row_start + row, col_start + col)
                            for row, col in shape
                        )

                        direction = _infer_direction(
                            input_cells,
                            own_box,
                        )
                        if direction is None:
                            continue

                        docking = _dock_component(
                            input_cells,
                            direction,
                            own_anchor,
                            all_anchors,
                            height,
                            width,
                        )
                        if docking is None:
                            continue

                        output_cells, path, swept_cells = docking
                        direction_class = (
                            _CARDINAL
                            if 0 in direction
                            else _DIAGONAL
                        )

                        libraries[color_index].append({
                            "color_index": color_index,
                            "size": size,
                            "shape": shape,
                            "kind": _shape_kind(shape),
                            "direction": direction,
                            "direction_class": direction_class,
                            "input_cells": input_cells,
                            "output_cells": output_cells,
                            "path": path,
                            "sweep": swept_cells,
                            "steps": len(path) - 1,
                        })

    return libraries


def _shape_kind(shape):
    rows = {row for row, _ in shape}
    cols = {col for _, col in shape}
    if len(shape) == 1:
        return "singleton"
    return "bar" if len(rows) == 1 or len(cols) == 1 else "bent"


def _validate_count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    if num_train is not None:
        num_train = _validate_count(num_train, 2, 10, "num_train")
    if num_test is not None:
        num_test = _validate_count(num_test, 1, 4, "num_test")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
    elif num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _random_anchor_boxes(rng, height, width, sizes):
    mode = int(rng.choice(3, p=(0.38, 0.36, 0.26)))
    if mode == 0:
        corner = int(rng.integers(0, 4))
        corners = (corner, 3 - corner)
        boxes = []
        for size, place in zip(sizes, corners):
            row = 0 if place < 2 else height - size
            col = 0 if place % 2 == 0 else width - size
            boxes.append((row, row + size - 1, col, col + size - 1))
        return tuple(boxes)

    positions = []
    for size in sizes:
        possible = []
        for row in range(height - size + 1):
            for col in range(width - size + 1):
                edge = row in (0, height - size) or col in (0, width - size)
                corner = row in (0, height - size) and col in (0, width - size)
                if mode == 1 and edge and not corner:
                    possible.append((row, col))
                elif mode == 2 and not edge:
                    possible.append((row, col))
        positions.append(possible)
    if not positions[0]:
        positions[0] = [(0, 0)]
    first = positions[0][int(rng.integers(len(positions[0])))]
    size0 = sizes[0]
    first_center = (first[0] + (size0 - 1) / 2, first[1] + (size0 - 1) / 2)
    second_options = []
    size1 = sizes[1]
    for row in range(height - size1 + 1):
        for col in range(width - size1 + 1):
            second_center = (row + (size1 - 1) / 2, col + (size1 - 1) / 2)
            distance = sum((a - b) ** 2 for a, b in zip(first_center, second_center))
            if distance >= 32:
                second_options.append((row, col))
    if not second_options:
        second_options = [(height - size1, width - size1)]
    second = second_options[int(rng.integers(len(second_options)))]
    return (
        (first[0], first[0] + size0 - 1, first[1], first[1] + size0 - 1),
        (second[0], second[0] + size1 - 1, second[1], second[1] + size1 - 1),
    )


def _sample_pair_parameters(rng, shared_background):
    if rng.random() < 0.20:
        height, width = 9, 9
    else:
        height = int(rng.integers(8, 16))
        width = int(rng.integers(8, 16))
    limit = 3 if min(height, width) < 10 else 4
    sizes = tuple(int(rng.choice(tuple(range(2, limit + 1)), p=(0.17, 0.83) if limit == 3 else (0.13, 0.72, 0.15))) for _ in range(2))
    if shared_background is None:
        palette = rng.choice(10, size=3, replace=False)
        background = int(palette[0])
        payload = (int(palette[1]), int(palette[2]))
    else:
        background = shared_background
        palette = rng.choice([color for color in range(10) if color != background], size=2, replace=False)
        payload = (int(palette[0]), int(palette[1]))
    return {
        "height": height, "width": width,
        "background": background, "payload_colors": payload,
        "anchor_boxes": _random_anchor_boxes(rng, height, width, sizes),
    }


def _sample_specs(rng, regime, allowed):
    simple = rng.random() < (0.22 if regime == "source_train" else 0.25)
    tokens = ((1, "singleton"), (2, "bar"), (3, "bar"),
              (3, "bent"), (4, "bar"), (4, "bent"))
    weights = np.array((0.34, 0.29, 0.14, 0.13, 0.05, 0.05)
                       if regime == "source_train" else
                       (0.28, 0.26, 0.18, 0.16, 0.06, 0.06))
    if allowed is not None:
        weights = np.array([weight if kind == "singleton" or kind in allowed else 0
                            for weight, (_, kind) in zip(weights, tokens)])
        weights /= weights.sum()
    specs = []
    for color_index in range(2):
        count = int(rng.integers(2, 4 if regime == "source_train" else 5))
        for _ in range(count):
            if simple:
                size, kind = 1, "singleton"
            else:
                size, kind = tokens[int(rng.choice(len(tokens), p=weights))]
            specs.append({"color_index": color_index, "size": size,
                          "kind": kind,
                          "direction_class": _DIAGONAL if rng.random() < 0.42 else _CARDINAL,
                          "min_steps": 1 if rng.random() < 0.70 else 2})
    return specs


def _select_fragments(rng, specs, libraries):
    options = []
    for spec in specs:
        eligible = [candidate for candidate in libraries[spec["color_index"]]
                    if candidate["size"] == spec["size"]
                    and candidate["kind"] == spec["kind"]
                    and candidate["direction_class"] == spec["direction_class"]
                    and candidate["steps"] >= spec["min_steps"]]
        if not eligible:
            return None
        order = rng.permutation(len(eligible))
        options.append([eligible[int(index)] for index in order])
    spec_order = sorted(range(len(specs)), key=lambda index: len(options[index]))
    selected = [None] * len(specs)
    used_input, used_output = set(), set()
    input_by_color = [set(), set()]
    nodes = 0

    def search(depth):
        nonlocal nodes
        if depth == len(spec_order):
            return True
        index = spec_order[depth]
        color = specs[index]["color_index"]
        for candidate in options[index]:
            nodes += 1
            if nodes > 45000:
                return False
            input_cells = candidate["input_cells"]
            output_cells = candidate["output_cells"]
            if input_cells & used_input or output_cells & used_output:
                continue
            if any((row + dr, col + dc) in input_by_color[color]
                   for row, col in input_cells
                   for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                continue
            selected[index] = candidate
            used_input.update(input_cells)
            used_output.update(output_cells)
            input_by_color[color].update(input_cells)
            if search(depth + 1):
                return True
            input_by_color[color].difference_update(input_cells)
            used_output.difference_update(output_cells)
            used_input.difference_update(input_cells)
            selected[index] = None
        return False

    return selected if search(0) else None


def _render_pair(parameters, selected):
    shape = (parameters["height"], parameters["width"])
    background = parameters["background"]
    input_grid = np.full(shape, background, dtype=np.int8)
    output_grid = np.full(shape, background, dtype=np.int8)
    for index, box in enumerate(parameters["anchor_boxes"]):
        color = parameters["payload_colors"][index]
        for row, col in _box_cells(box):
            input_grid[row, col] = color
            output_grid[row, col] = color
    for fragment in selected:
        color = parameters["payload_colors"][fragment["color_index"]]
        for row, col in fragment["input_cells"]:
            input_grid[row, col] = color
        for row, col in fragment["output_cells"]:
            output_grid[row, col] = color
    return input_grid, output_grid


def _is_source_duplicate(input_grid, output_grid):
    return any(np.array_equal(input_grid, old_input) and
               np.array_equal(output_grid, old_output)
               for old_input, old_output in _SOURCE_PAIRS)


def _generate_pair(rng, regime, shared_background, allowed):
    for _ in range(90):
        parameters = _sample_pair_parameters(rng, shared_background)
        libraries = _build_libraries(parameters["height"], parameters["width"],
                                    parameters["anchor_boxes"])
        for _ in range(12):
            specs = _sample_specs(rng, regime, allowed)
            selected = _select_fragments(rng, specs, libraries)
            if selected is None:
                continue
            inp, out = _render_pair(parameters, selected)
            colors, counts = np.unique(inp, return_counts=True)
            if int(colors[np.argmax(counts)]) != parameters["background"]:
                continue
            if np.array_equal(inp, out) or _is_source_duplicate(inp, out):
                continue
            return {"input": inp, "output": out}
    raise ValueError("unable to construct a valid pair")


def _grid_key(grid):
    return grid.shape, grid.tobytes(order="C")


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _four_components(cells):
    remaining = set(cells)
    components = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = {start}
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(frozenset(component))
    return components


def _find_anchor(grid, color):
    height, width = grid.shape
    for size in (4, 3, 2):
        found = []
        for row in range(height - size + 1):
            for col in range(width - size + 1):
                if np.all(grid[row:row + size, col:col + size] == color):
                    found.append((row, row + size - 1, col, col + size - 1))
        if found:
            assert len(found) == 1
            return found[0]
    raise AssertionError("missing solid anchor")


def _shape_signature(cells):
    top = min(row for row, _ in cells)
    left = min(col for _, col in cells)
    return tuple(sorted((row - top, col - left) for row, col in cells))


def _analyze_pair(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    height, width = inp.shape
    colors, counts = np.unique(inp, return_counts=True)
    assert len(colors) == 3
    background = int(colors[np.argmax(counts)])
    payload = tuple(sorted(int(color) for color in colors if color != background))
    assert set(int(color) for color in np.unique(out)) == set(int(c) for c in colors)
    assert all(np.count_nonzero(inp == color) == np.count_nonzero(out == color)
               for color in colors)
    boxes = {color: _find_anchor(inp, color) for color in payload}
    anchors = {color: _box_cells(boxes[color]) for color in payload}
    all_anchors = anchors[payload[0]] | anchors[payload[1]]
    assert not (anchors[payload[0]] & anchors[payload[1]])
    expected = np.full(inp.shape, background, dtype=np.int8)
    for color in payload:
        for row, col in anchors[color]:
            assert out[row, col] == color
            expected[row, col] = color
    classes, kinds, sizes, directions, loads = set(), set(), set(), set(), []
    sweeps = []
    for color in payload:
        satellites = {(int(row), int(col)) for row, col in np.argwhere(inp == color)
                      if (int(row), int(col)) not in anchors[color]}
        components = _four_components(satellites)
        assert components
        load = 0
        for component in components:
            assert 1 <= len(component) <= 4
            shape = _shape_signature(component)
            assert shape in _SHAPES[len(component)]
            direction = _infer_direction(component, boxes[color])
            assert direction is not None
            docking = _dock_component(component, direction, anchors[color],
                                      all_anchors, height, width)
            assert docking is not None
            cells, _, sweep = docking
            sweeps.append(sweep)
            for row, col in cells:
                assert expected[row, col] == background
                expected[row, col] = color
            classes.add(_CARDINAL if 0 in direction else _DIAGONAL)
            kinds.add(_shape_kind(shape))
            sizes.add(len(component))
            directions.add(direction)
            load += len(component)
        loads.append(load)
    assert np.array_equal(expected, out)
    assert not np.array_equal(inp, out)
    crossing = any(bool(first & second) for index, first in enumerate(sweeps)
                   for second in sweeps[index + 1:])
    positions = tuple((box[0], box[2], box[1] - box[0] + 1)
                      for box in (boxes[color] for color in payload))
    return {"background": background, "payload": payload,
            "shape": inp.shape, "boxes": boxes, "positions": positions,
            "classes": classes, "kinds": kinds, "sizes": sizes,
            "directions": directions, "loads": tuple(loads),
            "crossing": crossing}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            key = _pair_key(pair)
            assert key[0] not in seen_inputs and key not in seen_pairs
            seen_inputs.add(key[0])
            seen_pairs.add(key)
            assert not _is_source_duplicate(pair["input"], pair["output"])
            info[split].append(_analyze_pair(pair))
    training_classes = set().union(*(item["classes"] for item in info["train"]))
    training_kinds = set().union(*(item["kinds"] for item in info["train"]))
    training_max_size = max(max(item["sizes"]) for item in info["train"])
    assert {_CARDINAL, _DIAGONAL} <= training_classes
    assert {"singleton", "bar"} <= training_kinds
    for item in info["test"]:
        assert item["classes"] <= training_classes
        assert item["kinds"] <= training_kinds
        assert max(item["sizes"]) <= min(4, training_max_size + 1)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    for _ in range(45):
        shared_background = int(rng.integers(0, 10)) if rng.random() < 0.50 else None
        puzzle = {"train": [], "test": []}
        seen = set()
        try:
            for _ in range(num_train):
                for _ in range(8):
                    pair = _generate_pair(rng, "source_train", shared_background, None)
                    if _grid_key(pair["input"]) not in seen:
                        break
                else:
                    raise ValueError("duplicate training input")
                seen.add(_grid_key(pair["input"]))
                puzzle["train"].append(pair)
            train_info = [_analyze_pair(pair) for pair in puzzle["train"]]
            classes = set().union(*(item["classes"] for item in train_info))
            kinds = set().union(*(item["kinds"] for item in train_info))
            max_size = max(max(item["sizes"]) for item in train_info)
            if not {_CARDINAL, _DIAGONAL} <= classes or not {"singleton", "bar"} <= kinds:
                continue
            for _ in range(num_test):
                for _ in range(12):
                    pair = _generate_pair(rng, "target_test", shared_background, kinds)
                    test_info = _analyze_pair(pair)
                    if (max(test_info["sizes"]) <= min(4, max_size + 1)
                            and _grid_key(pair["input"]) not in seen):
                        break
                else:
                    raise ValueError("unable to select distinct test")
                seen.add(_grid_key(pair["input"]))
                puzzle["test"].append(pair)
            validate_generated_puzzle(puzzle)
            return puzzle
        except (ValueError, AssertionError):
            continue
    raise ValueError("unable to construct a complete puzzle")


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a["input"], b["input"]) and
                   np.array_equal(a["output"], b["output"])
                   for a, b in zip(first[split], second[split]))
               for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    dimensions, palettes, layouts = set(), set(), set()
    kinds, sizes, directions = set(), set(), set()
    varied_palettes = varied_positions = crossings = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        all_info = [_analyze_pair(pair) for split in ("train", "test")
                    for pair in puzzle[split]]
        varied_palettes += len({(item["background"], item["payload"])
                                for item in all_info}) > 1
        varied_positions += len({item["positions"] for item in all_info}) > 1
        crossings += any(item["crossing"] for item in all_info)
        for item in all_info:
            dimensions.add(item["shape"])
            palettes.add((item["background"], item["payload"]))
            layouts.add((item["shape"], item["positions"],
                         tuple(sorted(item["sizes"])), item["loads"]))
            kinds.update(item["kinds"])
            sizes.update(item["sizes"])
            directions.update(item["directions"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 12 and len(palettes) >= 30 and len(layouts) >= 100
    assert kinds == {"singleton", "bar", "bent"}
    assert sizes == {1, 2, 3, 4} and len(directions) == 8
    assert varied_palettes >= 30 and varied_positions >= 30 and crossings >= 1
    assert _puzzles_equal(generate_puzzle(314159, 6, 3),
                          generate_puzzle(314159, 6, 3))
    assert not _puzzles_equal(generate_puzzle(2718, 5, 2),
                              generate_puzzle(2719, 5, 2))
    for index, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(8000 + index, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        validate_generated_puzzle(puzzle)
    for index, (train_count, test_count) in enumerate(((None, 1), (None, 4),
                                                        (2, None), (10, None))):
        puzzle = generate_puzzle(9000 + index, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(123, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
    # This source example has intersecting travel sweeps and distinct landings.
    source_crossing = _analyze_pair({"input": _SOURCE_PAIRS[1][0],
                                     "output": _SOURCE_PAIRS[1][1]})
    assert source_crossing["crossing"]
