import numpy as np


_EDGES = ("top", "bottom", "left", "right")

_SOURCE_PAIRS = (
    (
        (
            "20033004400",
            "20000500000",
            "20000500660",
            "20000008000",
            "20770008000",
        ),
        (
            "23344000000",
            "25000000000",
            "25660000000",
            "20080000000",
            "27780000000",
        ),
    ),
    (
        (
            "000088",
            "300400",
            "300400",
            "000006",
            "110006",
            "000000",
            "005500",
            "000000",
            "000000",
            "222222",
        ),
        (
            "000000",
            "000000",
            "000000",
            "000000",
            "000000",
            "000000",
            "300488",
            "300406",
            "115506",
            "222222",
        ),
    ),
    (
        (
            "0440000002",
            "0005500602",
            "0000000602",
            "0900880002",
            "0900000002",
        ),
        (
            "0000000442",
            "0000005562",
            "0000000062",
            "0000009882",
            "0000009002",
        ),
    ),
    (
        (
            "2222222222",
            "0033000600",
            "0000070600",
            "0880070000",
            "0000000066",
            "0000330000",
            "0090000040",
            "0090000040",
            "3000011000",
            "3000000550",
        ),
        (
            "2222222222",
            "3033070666",
            "3880070640",
            "0090330040",
            "0090011550",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
)


def _validate_requested_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _validate_requested_count(
        num_train, "num_train", 2, 10
    )
    num_test = _validate_requested_count(
        num_test, "num_test", 1, 4
    )

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        largest_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, largest_test + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _wall_cells(height, width, edge):
    if edge == "top":
        return frozenset((0, col) for col in range(width))
    if edge == "bottom":
        return frozenset((height - 1, col) for col in range(width))
    if edge == "left":
        return frozenset((row, 0) for row in range(height))
    if edge == "right":
        return frozenset((row, width - 1) for row in range(height))
    raise AssertionError(f"unknown edge: {edge}")


def _shift_cells(cells, edge):
    if edge == "top":
        dr, dc = -1, 0
    elif edge == "bottom":
        dr, dc = 1, 0
    elif edge == "left":
        dr, dc = 0, -1
    elif edge == "right":
        dr, dc = 0, 1
    else:
        raise AssertionError(f"unknown edge: {edge}")

    return frozenset((row + dr, col + dc) for row, col in cells)


def _front_distance(cells, height, width, edge):
    if edge == "top":
        return min(row for row, _ in cells)
    if edge == "bottom":
        return min(height - 1 - row for row, _ in cells)
    if edge == "left":
        return min(col for _, col in cells)
    if edge == "right":
        return min(width - 1 - col for _, col in cells)
    raise AssertionError(f"unknown edge: {edge}")


def _settle_objects(objects, height, width, edge):
    occupied = set(_wall_cells(height, width, edge))
    settled = [None] * len(objects)

    order = sorted(
        range(len(objects)),
        key=lambda index: (
            _front_distance(
                objects[index], height, width, edge
            ),
            index,
        ),
    )

    for index in order:
        current = frozenset(objects[index])
        steps = 0

        while True:
            candidate = _shift_cells(current, edge)
            inside = all(
                0 <= row < height and 0 <= col < width
                for row, col in candidate
            )
            if not inside or any(cell in occupied for cell in candidate):
                break
            current = candidate
            steps += 1
            assert steps <= max(height, width)

        assert not any(cell in occupied for cell in current)
        settled[index] = current
        occupied.update(current)

    assert all(cells is not None for cells in settled)
    return settled, frozenset(occupied)


def _analyze_input(grid):
    height, width = grid.shape
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    background = int(np.argmax(counts))

    other_counts = [
        int(counts[color])
        for color in range(10)
        if color != background
    ]
    assert int(counts[background]) > max(other_counts), (
        "background must be the unique most frequent color"
    )

    edge_lines = (
        ("top", grid[0, :]),
        ("bottom", grid[-1, :]),
        ("left", grid[:, 0]),
        ("right", grid[:, -1]),
    )
    candidates = []
    for edge, line in edge_lines:
        color = int(line[0])
        if color != background and np.all(line == color):
            candidates.append((edge, color))

    assert len(candidates) == 1, (
        "input must have exactly one non-background full boundary line"
    )
    edge, wall = candidates[0]

    wall_positions = _wall_cells(height, width, edge)
    assert all(int(grid[row, col]) == wall for row, col in wall_positions)
    assert int(np.count_nonzero(grid == wall)) == len(wall_positions), (
        "wall color may occur only on the collecting boundary"
    )

    objects = []
    for color_value in np.unique(grid):
        color = int(color_value)
        if color in (background, wall):
            continue

        mask = grid == color
        visited = np.zeros(grid.shape, dtype=bool)

        for start_row, start_col in zip(*np.where(mask)):
            start_row = int(start_row)
            start_col = int(start_col)
            if visited[start_row, start_col]:
                continue

            component = []
            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True

            while stack:
                row, col = stack.pop()
                component.append((row, col))

                for nr, nc in (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                ):
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and mask[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            assert len(component) == 2, (
                "each monochromatic payload component must have two cells"
            )
            first, second = component
            assert (
                abs(first[0] - second[0])
                + abs(first[1] - second[1])
                == 1
            ), "payload components must be axis-aligned dominoes"

            objects.append(
                {
                    "color": color,
                    "cells": frozenset(component),
                }
            )

    assert objects, "at least one domino is required"

    return {
        "background": background,
        "wall": wall,
        "edge": edge,
        "objects": objects,
    }


def _solve_grid(input_grid):
    analysis = _analyze_input(input_grid)
    height, width = input_grid.shape
    original = [obj["cells"] for obj in analysis["objects"]]

    settled, _ = _settle_objects(
        original, height, width, analysis["edge"]
    )

    output = np.full(
        input_grid.shape,
        analysis["background"],
        dtype=input_grid.dtype,
    )

    for row, col in _wall_cells(height, width, analysis["edge"]):
        output[row, col] = analysis["wall"]

    for obj, cells in zip(analysis["objects"], settled):
        for row, col in cells:
            assert int(output[row, col]) == analysis["background"]
            output[row, col] = obj["color"]

    return output


def _packing_metrics(original, settled, occupied, height, width, edge):
    horizontal = sum(
        len({row for row, _ in cells}) == 1 for cells in original
    )
    vertical = len(original) - horizontal

    moved = sum(
        frozenset(before) != frozenset(after)
        for before, after in zip(original, settled)
    )

    fronts = [
        _front_distance(cells, height, width, edge)
        for cells in settled
    ]
    blocked_away_from_wall = sum(front > 1 for front in fronts)
    max_front = max(fronts)

    asymmetric_stops = 0
    for cells in settled:
        is_horizontal = len({row for row, _ in cells}) == 1
        perpendicular = (
            is_horizontal if edge in ("top", "bottom") else not is_horizontal
        )
        if not perpendicular:
            continue

        external_occupied = set(occupied).difference(cells)
        candidate = _shift_cells(cells, edge)
        blocking_cells = {
            cell
            for cell in candidate
            if (
                cell in external_occupied
                or not (
                    0 <= cell[0] < height
                    and 0 <= cell[1] < width
                )
            )
        }
        if len(blocking_cells) == 1:
            asymmetric_stops += 1

    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "moved": moved,
        "blocked": blocked_away_from_wall,
        "max_front": max_front,
        "asymmetric": asymmetric_stops,
    }


_FAVORED_SHAPES = ((5, 10), (10, 6), (10, 10), (6, 11), (11, 5))
_STYLES = ("random", "stationary", "independent", "parallel_h",
           "parallel_v", "flat", "dense")


def _sample_dimensions(rng, count, style):
    for _ in range(80):
        if rng.random() < 0.28:
            depth, span = _FAVORED_SHAPES[int(rng.integers(len(_FAVORED_SHAPES)))]
        else:
            depth = int(rng.integers(5, 18))
            span = int(rng.integers(5, 18))
        if style == "teaching" and (depth < 6 or span < 8):
            continue
        if style == "independent" and span < count:
            continue
        # Keep the uniform background identifiable even with one active color.
        background_area = depth * span - span - 2 * count
        if background_area <= max(span, 2 * count):
            continue
        return depth, span
    raise RuntimeError("could not sample feasible dimensions")


def _random_geometry(rng, height, width, count, style):
    occupied = set()
    objects = []
    reserved_columns = set()

    if style == "teaching":
        col = int(rng.integers(0, width - 2))
        mirror = bool(rng.integers(0, 2))
        support_col = col + (1 if mirror else 0)
        mover_col = col if mirror else col + 1
        mover_row = int(rng.integers(3, height))
        objects = [
            frozenset(((1, support_col), (1, support_col + 1))),
            frozenset(((mover_row, mover_col), (mover_row, mover_col + 1))),
        ]
        occupied.update(objects[0])
        occupied.update(objects[1])
        reserved_columns = {col, col + 1, col + 2}

    if style == "independent":
        available = set(range(width))
        for index in range(count):
            choices = []
            for col in sorted(available):
                choices.append((False, col))
                if col + 1 in available:
                    choices.append((True, col))
            if not choices:
                return None
            # Leave enough lanes for the remaining pieces.
            choices = [choice for choice in choices
                       if len(available) - (2 if choice[0] else 1)
                       >= count - index - 1]
            if not choices:
                return None
            horizontal, col = choices[int(rng.integers(len(choices)))]
            high = height if horizontal else height - 1
            row = int(rng.integers(1, high))
            cells = frozenset(((row, col), (row, col + 1))) if horizontal else \
                frozenset(((row, col), (row + 1, col)))
            objects.append(cells)
            available.remove(col)
            if horizontal:
                available.remove(col + 1)
        return objects

    for _ in range(len(objects), count):
        placed = False
        for _ in range(250):
            if style == "parallel_h":
                horizontal = True
            elif style == "parallel_v":
                horizontal = False
            else:
                horizontal = bool(rng.integers(0, 2))
            high = height if horizontal else height - 1
            if style == "stationary" and not objects:
                row = 1
            else:
                row = int(rng.integers(1, high))
            col = int(rng.integers(0, width - (1 if horizontal else 0)))
            cells = frozenset(((row, col), (row, col + 1))) if horizontal else \
                frozenset(((row, col), (row + 1, col)))
            if any(cell in occupied or cell[1] in reserved_columns
                   for cell in cells):
                continue
            objects.append(cells)
            occupied.update(cells)
            placed = True
            break
        if not placed:
            return None
    return objects


def _sample_count(rng, style):
    bounds = {
        "teaching": (5, 9), "random": (2, 12),
        "stationary": (3, 11), "independent": (2, 6),
        "parallel_h": (3, 11), "parallel_v": (3, 11),
        "flat": (3, 9), "dense": (8, 14),
    }
    low, high = bounds[style]
    return int(rng.integers(low, high + 1))


def _geometry_matches(style, objects, height, width):
    settled, occupied = _settle_objects(objects, height, width, "top")
    metrics = _packing_metrics(objects, settled, occupied, height, width, "top")
    count = len(objects)
    stationary = count - metrics["moved"]
    if style == "teaching":
        vertical_mover = any(
            len({row for row, _ in before}) == 2 and before != after
            for before, after in zip(objects, settled)
        )
        return (stationary >= 1 and metrics["moved"] >= 2
                and metrics["asymmetric"] >= 1 and vertical_mover)
    if style == "stationary":
        return stationary >= 1 and metrics["moved"] >= 1
    if style == "independent":
        return metrics["moved"] >= 1 and metrics["blocked"] == 0
    if style in ("parallel_h", "parallel_v"):
        return metrics["moved"] >= 1
    if style == "flat":
        return metrics["moved"] >= 1 and metrics["blocked"] >= 1 and metrics["asymmetric"] == 0
    if style == "dense":
        return metrics["moved"] >= 2 and metrics["blocked"] >= 3 and metrics["max_front"] >= 3
    return metrics["moved"] >= 1


def _sample_geometry(rng, style):
    for _ in range(350):
        count = _sample_count(rng, style)
        height, width = _sample_dimensions(rng, count, style)
        objects = _random_geometry(rng, height, width, count, style)
        if objects is not None and _geometry_matches(style, objects, height, width):
            return height, width, objects
    raise RuntimeError(f"could not construct {style} layout")


def _objects_are_adjacent(first, second):
    return any(
        abs(r1 - r2) + abs(c1 - c2) == 1
        for r1, c1 in first
        for r2, c2 in second
    )


def _assign_colors(rng, objects, palette, active_count):
    if active_count > len(objects) or active_count > len(palette):
        return None
    adjacent = [
        {other for other in range(len(objects)) if other != index
         and _objects_are_adjacent(objects[index], objects[other])}
        for index in range(len(objects))
    ]
    for _ in range(30):
        colors = [int(x) for x in rng.choice(palette, active_count, replace=False)]
        order = [int(x) for x in rng.permutation(len(objects))]
        assignment = [None] * len(objects)
        for index, color in zip(order[:active_count], colors):
            assignment[index] = color
        for index in order[active_count:]:
            legal = [color for color in colors
                     if all(assignment[other] != color for other in adjacent[index])]
            if not legal:
                break
            assignment[index] = legal[int(rng.integers(len(legal)))]
        if all(color is not None for color in assignment):
            return assignment
    return None


def _render_top_grid(
    height,
    width,
    background,
    wall,
    objects,
    colors,
):
    grid = np.full((height, width), background, dtype=np.int8)
    grid[0, :] = wall

    for cells, color in zip(objects, colors):
        for row, col in cells:
            assert int(grid[row, col]) == background
            grid[row, col] = color

    return grid


def _from_top_orientation(grid, edge):
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.rot90(grid, 2).copy()
    if edge == "left":
        return np.rot90(grid, 1).copy()
    if edge == "right":
        return np.rot90(grid, 3).copy()
    raise AssertionError(f"unknown edge: {edge}")


def _generate_pair(rng, style, edge, fixed_roles):
    for _ in range(100):
        height, width, objects = _sample_geometry(rng, style)
        if fixed_roles is None:
            role_order = [int(x) for x in rng.permutation(10)]
            background, wall = role_order[:2]
        else:
            background, wall = fixed_roles
        palette = tuple(color for color in range(10) if color not in (background, wall))
        upper = min(len(palette), len(objects))
        if style == "independent" and rng.random() < 0.28:
            active_count = 1
        else:
            active_count = int(rng.integers(2, upper + 1)) if upper >= 2 else 1
        colors = _assign_colors(rng, objects, palette, active_count)
        if colors is None:
            continue
        canonical_input = _render_top_grid(height, width, background, wall, objects, colors)
        input_grid = _from_top_orientation(canonical_input, edge)
        try:
            output_grid = _solve_grid(input_grid)
        except AssertionError:
            continue
        if not np.array_equal(input_grid, output_grid):
            return input_grid, output_grid
    raise RuntimeError(f"could not color {style} layout")


def _grid_signature(grid):
    normalized = np.asarray(grid, dtype=np.int8)
    return tuple(grid.shape), normalized.tobytes()


def _grid_as_rows(grid):
    return tuple(
        "".join(chr(48 + int(value)) for value in row)
        for row in grid
    )


def _is_source_pair(input_grid, output_grid):
    input_rows = _grid_as_rows(input_grid)
    output_rows = _grid_as_rows(output_grid)
    return any(
        input_rows == source_input
        and output_rows == source_output
        for source_input, source_output in _SOURCE_PAIRS
    )


def _edge_plan(rng, num_train, num_test):
    mode = int(rng.integers(0, 5))
    if mode == 0 and num_train >= 3:
        train = ["left", "bottom", "right"]
        train += [train[int(rng.integers(3))] for _ in range(num_train - 3)]
        return train, ["top"] * num_test
    if mode == 1 or (mode == 0 and num_train < 3):
        held = _EDGES[int(rng.integers(4))]
        available = [edge for edge in _EDGES if edge != held]
        train = [available[int(x)] for x in rng.permutation(3)[:min(3, num_train)]]
        train += [available[int(rng.integers(3))] for _ in range(num_train - len(train))]
        return train, [held] * num_test
    if mode == 2:
        available = [str(x) for x in rng.choice(_EDGES, 2, replace=False)]
        train = available[:min(2, num_train)]
        train += [available[int(rng.integers(2))] for _ in range(num_train - len(train))]
        test = [train[int(rng.integers(len(train)))] for _ in range(num_test)]
        return train, test
    if mode == 3:
        edge = _EDGES[int(rng.integers(4))]
        return [edge] * num_train, [edge] * num_test
    first_two = [str(x) for x in rng.choice(_EDGES, 2, replace=False)]
    train = first_two + [_EDGES[int(rng.integers(4))] for _ in range(num_train - 2)]
    test = [_EDGES[int(rng.integers(4))] for _ in range(num_test)]
    return train, test


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    train_edges, test_edges = _edge_plan(rng, num_train, num_test)

    # The only learned rule is geometric. Some puzzles use stable literal
    # colors; others demonstrate that roles are local to each input.
    if rng.random() < 0.5:
        role_order = [int(x) for x in rng.permutation(10)]
        fixed_roles = tuple(role_order[:2])
    else:
        fixed_roles = None

    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for split, count, edges in (
        ("train", num_train, train_edges),
        ("test", num_test, test_edges),
    ):
        for index in range(count):
            pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            pair_rng = np.random.default_rng(pair_seed)
            if split == "train" and index == 0:
                style = "teaching"
            else:
                style_options = _STYLES if split == "train" else _STYLES + ("dense", "random")
                style = style_options[int(pair_rng.integers(len(style_options)))]
            for _ in range(30):
                input_grid, output_grid = _generate_pair(pair_rng, style, edges[index], fixed_roles)
                signature = _grid_signature(input_grid)
                if signature not in seen_inputs and not _is_source_pair(input_grid, output_grid):
                    break
            else:
                raise RuntimeError("could not generate a unique pair")
            seen_inputs.add(signature)
            puzzle[split].append({"input": input_grid, "output": output_grid})

    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_metrics(pair):
    analysis = _analyze_input(pair["input"])
    objects = [obj["cells"] for obj in analysis["objects"]]
    height, width = pair["input"].shape
    settled, occupied = _settle_objects(objects, height, width, analysis["edge"])
    metrics = _packing_metrics(objects, settled, occupied, height, width, analysis["edge"])
    metrics["count"] = len(objects)
    metrics["stationary"] = len(objects) - metrics["moved"]
    metrics["horizontal_moved"] = sum(
        len({row for row, _ in before}) == 1 and before != after
        for before, after in zip(objects, settled)
    )
    metrics["vertical_moved"] = sum(
        len({row for row, _ in before}) == 2 and before != after
        for before, after in zip(objects, settled)
    )
    metrics["active_colors"] = len({obj["color"] for obj in analysis["objects"]})
    metrics["edge"] = analysis["edge"]
    metrics["background"] = analysis["background"]
    metrics["wall"] = analysis["wall"]
    return metrics


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_metrics = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = (_grid_signature(inp), _grid_signature(out))
            assert key[0] not in seen_inputs and key not in seen_pairs
            seen_inputs.add(key[0])
            seen_pairs.add(key)
            assert not _is_source_pair(inp, out)
            analysis = _analyze_input(inp)
            assert np.array_equal(out, _solve_grid(inp)), "wrong rigid-body settling"
            assert int(np.count_nonzero(out == analysis["wall"])) == len(
                _wall_cells(*inp.shape, analysis["edge"])
            )
            if split == "train":
                train_metrics.append(_pair_metrics(pair))

    # A teaching demonstration contains both moving orientations, a
    # stationary support, and a one-cell contact stop. No hidden palette or
    # direction mapping is needed: each input carries its own colored edge.
    assert any(m["stationary"] >= 1 and m["moved"] >= 2
               and m["asymmetric"] >= 1 and m["horizontal_moved"] >= 1
               and m["vertical_moved"] >= 1 for m in train_metrics)
    train_edges = {m["edge"] for m in train_metrics}
    test_edges = {_analyze_input(pair["input"])["edge"] for pair in puzzle["test"]}
    assert not (test_edges - train_edges) or len(train_edges) >= 2


def _puzzles_equal(first, second):
    if set(first.keys()) != {"train", "test"}:
        return False
    if set(second.keys()) != {"train", "test"}:
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for first_pair, second_pair in zip(
            first[split], second[split]
        ):
            if not np.array_equal(
                first_pair["input"], second_pair["input"]
            ):
                return False
            if not np.array_equal(
                first_pair["output"], second_pair["output"]
            ):
                return False
    return True


def _puzzle_signature(puzzle):
    return tuple(
        (
            split,
            _grid_signature(pair["input"]),
            _grid_signature(pair["output"]),
        )
        for split in ("train", "test")
        for pair in puzzle[split]
    )


def _stepwise_prediction(input_grid):
    analysis = _analyze_input(input_grid)
    height, width = input_grid.shape
    pieces = [set(obj["cells"]) for obj in analysis["objects"]]
    wall = set(_wall_cells(height, width, analysis["edge"]))
    for _ in range(max(height, width) * len(pieces)):
        moved = False
        order = sorted(range(len(pieces)), key=lambda i: _front_distance(
            pieces[i], height, width, analysis["edge"]))
        for index in order:
            obstacles = wall.union(*(pieces[j] for j in range(len(pieces)) if j != index))
            candidate = _shift_cells(pieces[index], analysis["edge"])
            if all(0 <= row < height and 0 <= col < width and (row, col) not in obstacles
                   for row, col in candidate):
                pieces[index] = set(candidate)
                moved = True
        if not moved:
            break
    else:
        raise AssertionError("stepwise settling did not terminate")
    result = np.full(input_grid.shape, analysis["background"], dtype=input_grid.dtype)
    for row, col in wall:
        result[row, col] = analysis["wall"]
    for obj, cells in zip(analysis["objects"], pieces):
        for row, col in cells:
            assert result[row, col] == analysis["background"]
            result[row, col] = obj["color"]
    return result


def self_test() -> None:
    puzzles = []
    seen = {"train": [], "test": []}
    train_counts, test_counts = set(), set()
    roles, shapes, layouts = set(), set(), set()
    fixed_role_puzzles = local_role_puzzles = 0
    all_same = held_out = repeated = original_shift = 0
    within_count_changes = within_edge_changes = 0
    for seed in range(1000, 1042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_edges, test_edges, puzzle_roles, counts = [], [], set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                analysis = _analyze_input(inp)
                metrics = _pair_metrics(pair)
                seen[split].append((inp.shape, metrics))
                counts.add(metrics["count"])
                puzzle_roles.add((metrics["background"], metrics["wall"]))
                roles.add((metrics["background"], metrics["wall"]))
                shapes.add(inp.shape)
                layouts.add((inp.shape, metrics["edge"], (inp != metrics["background"]).tobytes()))
                assert np.array_equal(_stepwise_prediction(inp), pair["output"])
                assert not _is_source_pair(inp, pair["output"])
                (train_edges if split == "train" else test_edges).append(metrics["edge"])
        fixed_role_puzzles += len(puzzle_roles) == 1
        local_role_puzzles += len(puzzle_roles) > 1
        within_count_changes += len(counts) > 1
        within_edge_changes += len(set(train_edges + test_edges)) > 1
        all_same += len(set(train_edges + test_edges)) == 1
        held_out += bool(set(test_edges) - set(train_edges))
        repeated += bool(set(test_edges) & set(train_edges))
        original_shift += set(train_edges) == {"left", "bottom", "right"} and set(test_edges) == {"top"}

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(roles) >= 15 and len(shapes) >= 30
    assert len(layouts) >= 0.85 * sum(len(p["train"]) + len(p["test"]) for p in puzzles)
    assert fixed_role_puzzles >= 5 and local_role_puzzles >= 5
    assert all_same >= 2 and held_out >= 5 and repeated >= 5 and original_shift >= 1
    assert within_count_changes >= 25 and within_edge_changes >= 20
    for split in ("train", "test"):
        records = seen[split]
        assert {m["edge"] for _, m in records} == set(_EDGES)
        assert min(m["count"] for _, m in records) <= 3
        assert max(m["count"] for _, m in records) >= 11
        assert any(m["stationary"] > 0 and m["moved"] > 0 for _, m in records)
        assert any(m["stationary"] == 0 for _, m in records)
        assert any(m["blocked"] == 0 for _, m in records)
        assert any(m["blocked"] >= 3 for _, m in records)
        assert any(m["horizontal"] == 0 for _, m in records)
        assert any(m["vertical"] == 0 for _, m in records)
        assert any(m["asymmetric"] > 0 for _, m in records)
        assert any(m["asymmetric"] == 0 for _, m in records)
        assert any(m["active_colors"] < m["count"] for _, m in records)
        assert any(m["active_colors"] == m["count"] for _, m in records)
        assert min(min(shape) for shape, _ in records) <= 5
        assert max(max(shape) for shape, _ in records) >= 15

    assert _puzzles_equal(puzzles[17], generate_puzzle(1017))
    assert _puzzles_equal(generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3))
    assert len({_puzzle_signature(generate_puzzle(seed, 4, 2)) for seed in range(700, 708)}) >= 7
    for offset, (num_train, num_test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(2000 + offset, num_train, num_test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (num_train, num_test)
    for offset, (num_train, num_test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(3000 + offset, num_train, num_test)
        assert num_train is None or len(puzzle["train"]) == num_train
        assert num_test is None or len(puzzle["test"]) == num_test
    for num_train, num_test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(4000, num_train, num_test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
