"""Marked pixel triangles whose markers color rays beyond their apexes."""

import numpy as np


DIRECTIONS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

# Exact descriptors of the three source demonstrations and the source test.
SOURCE = {
    ((12, 12), "up", 4, 1, 5, 0, 8, 3, (5, 6)),
    ((10, 15), "right", 3, 1, 9, 0, 2, 1, (4, 5)),
    ((15, 12), "down", 3, 1, 10, 0, 3, 2, (4, 4)),
    ((16, 11), "up", 4, 1, 11, 0, 4, 8, (11, 4)),
}


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _counts(seed, num_train, num_test):
    if num_train is not None:
        if not _integer(num_train) or not 2 <= num_train <= 10:
            raise ValueError("num_train must be an integer from 2 to 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _integer(num_test) or not 1 <= num_test <= 4:
            raise ValueError("num_test must be an integer from 1 to 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _signature(grid):
    return (grid.shape, np.asarray(grid, dtype=np.int8).tobytes())


def _components(points):
    remaining = set(points)
    found = []
    while remaining:
        start = remaining.pop()
        component = {start}
        frontier = [start]
        while frontier:
            row, col = frontier.pop()
            for neighbor in (
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
        found.append(component)
    return found


def _cells(apex, direction, depth, taper):
    dr, dc = DIRECTIONS[direction]
    pr, pc = -dc, dr
    result = set()
    for layer in range(depth):
        center = (apex[0] - layer * dr, apex[1] - layer * dc)
        for offset in range(-taper * layer, taper * layer + 1):
            result.add((center[0] + offset * pr, center[1] + offset * pc))
    return result


def _ray(apex, direction, shape):
    dr, dc = DIRECTIONS[direction]
    result = []
    row, col = apex
    while 0 <= row + dr < shape[0] and 0 <= col + dc < shape[1]:
        row += dr
        col += dc
        result.append((row, col))
    return result


def _analyze_input(grid):
    height, width = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    assert int(np.max(counts)) > height * width // 2
    foreground = {
        (int(row), int(col))
        for row, col in np.argwhere(grid != background)
    }
    components = _components(foreground)
    assert 1 <= len(components) <= 3
    objects = []
    for component in components:
        values, frequencies = np.unique(
            [grid[position] for position in component], return_counts=True
        )
        assert len(values) == 2
        markers = [int(value) for value, count in zip(values, frequencies) if count == 1]
        assert len(markers) == 1
        marker = markers[0]
        body = next(int(value) for value in values if int(value) != marker)
        rows = [row for row, _ in component]
        cols = [col for _, col in component]
        bounds = (min(rows), max(rows), min(cols), max(cols))
        candidates = []
        for direction in DIRECTIONS:
            vertical = direction in ("up", "down")
            depth = (bounds[1] - bounds[0] + 1) if vertical else (bounds[3] - bounds[2] + 1)
            span = (bounds[3] - bounds[2] + 1) if vertical else (bounds[1] - bounds[0] + 1)
            if depth < 2 or span < 3 or (span - 1) % (2 * (depth - 1)):
                continue
            taper = (span - 1) // (2 * (depth - 1))
            if taper not in (1, 2):
                continue
            center = (bounds[2] + bounds[3]) // 2 if vertical else (bounds[0] + bounds[1]) // 2
            apex = {
                "up": (bounds[0], center),
                "down": (bounds[1], center),
                "left": (center, bounds[2]),
                "right": (center, bounds[3]),
            }[direction]
            if _cells(apex, direction, depth, taper) != component:
                continue
            dr, dc = DIRECTIONS[direction]
            base = (apex[0] - (depth - 1) * dr, apex[1] - (depth - 1) * dc)
            if int(grid[base]) != marker:
                continue
            if any(int(grid[position]) != body for position in component if position != base):
                continue
            ray = _ray(apex, direction, grid.shape)
            if not ray or any(int(grid[position]) != background for position in ray):
                continue
            candidates.append({
                "direction": direction, "depth": depth, "taper": taper,
                "apex": apex, "base": base, "ray": ray, "body": body,
                "marker": marker, "cells": component, "bounds": bounds,
            })
        assert len(candidates) == 1
        objects.append(candidates[0])
    rays = [point for obj in objects for point in obj["ray"]]
    assert len(rays) == len(set(rays))
    return background, objects


def _solve_input(grid):
    _, objects = _analyze_input(grid)
    output = grid.copy()
    for obj in objects:
        for point in obj["ray"]:
            output[point] = obj["marker"]
    return output


def _fingerprint(shape, background, obj):
    return (
        shape, obj["direction"], obj["depth"], obj["taper"],
        len(obj["ray"]), background, obj["body"], obj["marker"], obj["apex"],
    )


def _size(rng, multiple):
    if multiple:
        return int(rng.integers(20, 31)), int(rng.integers(20, 31))
    height = int(rng.integers(7, 31))
    width = int(rng.integers(7, 31))
    if int(rng.integers(0, 7)) == 0:
        if int(rng.integers(0, 2)):
            height = 30
        else:
            width = 30
    return height, width


def _palette(rng, previous=None, force_new_marker=None):
    if previous is not None and int(rng.integers(0, 5)) == 0:
        bg, body, marker = previous
        if int(rng.integers(0, 2)):
            body, marker = marker, body
        if marker != force_new_marker:
            return bg, body, marker
    for _ in range(20):
        colors = [int(x) for x in rng.choice(10, size=3, replace=False)]
        if colors[2] != force_new_marker:
            return tuple(colors)
    raise AssertionError("palette sampling failed")


def _placement(rng, shape, direction, taper, multiple, depth_choices, long_ray=False):
    height, width = shape
    axis = height if direction in ("up", "down") else width
    perpendicular = width if direction in ("up", "down") else height
    maximum_depth = min(10 if not multiple else 5, axis - 1, 1 + (perpendicular - 1) // (2 * taper))
    if maximum_depth < 2:
        return None
    feasible_depths = [depth for depth in depth_choices if depth <= maximum_depth]
    if not feasible_depths:
        return None
    depth = feasible_depths[int(rng.integers(0, len(feasible_depths)))]
    breadth = 1 + 2 * taper * (depth - 1)
    maximum_ray = axis - depth
    if long_ray:
        ray_length = int(rng.integers(max(1, maximum_ray * 2 // 3), maximum_ray + 1))
    else:
        kind = int(rng.integers(0, 8))
        ray_length = 1 if kind == 0 else (
            maximum_ray if kind == 1 else int(rng.integers(1, maximum_ray + 1))
        )
    side = int(rng.integers(0, perpendicular - breadth + 1))
    center = side + taper * (depth - 1)
    apex = {
        "up": (ray_length, center),
        "down": (height - 1 - ray_length, center),
        "left": (center, ray_length),
        "right": (center, width - 1 - ray_length),
    }[direction]
    cells = _cells(apex, direction, depth, taper)
    return {"direction": direction, "depth": depth, "taper": taper,
            "apex": apex, "cells": cells, "ray": _ray(apex, direction, shape)}


def _compatible(placement, existing):
    cells = placement["cells"]
    ray = set(placement["ray"])
    for other in existing:
        other_cells = other["cells"]
        if cells & (other_cells | set(other["ray"])) or ray & (other_cells | set(other["ray"])):
            return False
        if any(
            (row + dr, col + dc) in other_cells
            for row, col in cells for dr, dc in DIRECTIONS.values()
        ):
            return False
    return True


def _pair(rng, multiple, taper_required, preferred_direction, previous_palette,
          force_new_marker, long_ray, depth_choices, same_direction=False):
    for canvas_attempt in range(32):
        shape = _size(rng, multiple)
        placements = []
        for object_index in range(2 if multiple else 1):
            placed = False
            for attempt in range(400):
                direction = (
                    preferred_direction if object_index == 0 or same_direction
                    else tuple(DIRECTIONS)[int(rng.integers(0, 4))]
                )
                taper = 2 if taper_required and object_index == 0 else (
                    2 if int(rng.integers(0, 5)) == 0 and taper_required else 1
                )
                placement = _placement(rng, shape, direction, taper, multiple,
                                       depth_choices, long_ray and object_index == 0)
                if placement is not None and _compatible(placement, placements):
                    placements.append(placement)
                    placed = True
                    break
            if not placed:
                break
        if len(placements) != (2 if multiple else 1):
            continue
        palette = _palette(rng, previous_palette, force_new_marker)
        background, first_body, first_marker = palette
        grid = np.full(shape, background, dtype=np.int8)
        output = grid.copy()
        for index, placement in enumerate(placements):
            if index == 0:
                body, marker = first_body, first_marker
            else:
                if int(rng.integers(0, 3)) == 0:
                    body, marker = first_body, first_marker
                else:
                    choices = [color for color in range(10) if color != background]
                    selected = rng.choice(choices, size=2, replace=False)
                    body, marker = int(selected[0]), int(selected[1])
            for point in placement["cells"]:
                grid[point] = body
            dr, dc = DIRECTIONS[placement["direction"]]
            apex = placement["apex"]
            base = (apex[0] - (placement["depth"] - 1) * dr,
                    apex[1] - (placement["depth"] - 1) * dc)
            grid[base] = marker
        if int(np.count_nonzero(grid == background)) <= grid.size // 2:
            continue
        output[:] = grid
        for placement in placements:
            marker = int(grid[(
                placement["apex"][0] - (placement["depth"] - 1) * DIRECTIONS[placement["direction"]][0],
                placement["apex"][1] - (placement["depth"] - 1) * DIRECTIONS[placement["direction"]][1],
            )])
            for point in placement["ray"]:
                output[point] = marker
        pair = {"input": grid, "output": output}
        try:
            background_check, objects = _analyze_input(grid)
            assert background_check == background
            assert np.array_equal(_solve_input(grid), output)
            assert len(objects) != 1 or _fingerprint(shape, background, objects[0]) not in SOURCE
        except AssertionError:
            continue
        return pair, palette
    raise AssertionError("could not place marked triangles")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if not _integer(seed):
        raise ValueError("seed must be an integer")
    seed_value = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _counts(seed_value, num_train, num_test)

    # These switches specify which additional visible motifs training teaches.
    multiple_taught = int(rng.integers(0, 5)) < 2
    wide_taper_taught = int(rng.integers(0, 5)) < 2
    depth_mode = int(rng.integers(0, 6))
    if depth_mode == 0:
        depth_options = [int(rng.integers(2, 6))]
    elif depth_mode in (1, 2):
        maximum = 5 if multiple_taught else 7
        depth_options = sorted(int(x) for x in rng.choice(
            np.arange(2, maximum + 1), size=2, replace=False
        ))
    else:
        depth_options = list(range(2, 11))
    common_direction = int(rng.integers(0, 4)) == 0
    if depth_mode == 0 and common_direction and not (
        multiple_taught or wide_taper_taught
    ):
        wide_taper_taught = True
    anchor_direction = tuple(DIRECTIONS)[int(rng.integers(0, 4))]
    seen = set()
    train = []
    test = []
    previous_palette = None
    first_marker = None
    first_depth = None
    train_directions = []

    for index in range(train_count):
        if common_direction:
            direction = anchor_direction
        elif index == 0:
            direction = anchor_direction
        elif index == 1:
            direction = tuple(d for d in DIRECTIONS if d != anchor_direction)[int(rng.integers(0, 3))]
        else:
            direction = tuple(DIRECTIONS)[int(rng.integers(0, 4))]
        train_directions.append(direction)
        multiple = multiple_taught if index == 0 else multiple_taught and int(rng.integers(0, 3)) == 0
        taper = wide_taper_taught if index == 0 else wide_taper_taught and int(rng.integers(0, 4)) == 0
        if index == 1 and depth_mode == 0 and common_direction:
            multiple = False
            taper = False
        pair_depths = (
            [depth_options[index]] if depth_mode in (1, 2) and index < 2
            else depth_options
        )
        if index == 1 and depth_mode >= 3 and common_direction:
            pair_depths = [depth for depth in depth_options if depth != first_depth]
        for _ in range(16):
            pair, palette = _pair(
                rng, multiple, taper, direction, previous_palette,
                first_marker if index == 1 else None, False, pair_depths,
                common_direction,
            )
            signature = _signature(pair["input"])
            if signature not in seen:
                break
        else:
            raise AssertionError("duplicate training inputs")
        seen.add(signature)
        train.append(pair)
        previous_palette = palette
        if index == 0:
            first_marker = palette[2]
            first_depth = min(
                obj["depth"] for obj in _analyze_input(pair["input"])[1]
            )

    for index in range(test_count):
        direction = tuple(DIRECTIONS)[int(rng.integers(0, 4))]
        if int(rng.integers(0, 4)) == 0:
            unseen = [d for d in DIRECTIONS if d not in train_directions]
            if unseen:
                direction = unseen[int(rng.integers(0, len(unseen)))]
        multiple = multiple_taught and int(rng.integers(0, 2)) == 0
        taper = wide_taper_taught and int(rng.integers(0, 3)) == 0
        long_ray = int(rng.integers(0, 4)) == 0
        for _ in range(16):
            pair, palette = _pair(
                rng, multiple, taper, direction, previous_palette, None,
                long_ray, depth_options,
            )
            signature = _signature(pair["input"])
            if signature not in seen:
                break
        else:
            raise AssertionError("duplicate test inputs")
        seen.add(signature)
        test.append(pair)
        previous_palette = palette

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    information = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.min(grid) >= 0 and np.max(grid) <= 9
            assert inp.shape == out.shape
            key = _signature(inp)
            pair_key = (key, _signature(out))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            background, objects = _analyze_input(inp)
            assert np.array_equal(_solve_input(inp), out)
            assert np.count_nonzero(inp != out) == sum(len(obj["ray"]) for obj in objects)
            assert len(objects) != 1 or _fingerprint(inp.shape, background, objects[0]) not in SOURCE
            information[split].append(objects)
    train_objects = [obj for group in information["train"] for obj in group]
    test_objects = [obj for group in information["test"] for obj in group]
    assert len({obj["marker"] for obj in train_objects}) >= 2
    train_geometries = {
        tuple(sorted(
            (obj["direction"], obj["depth"], obj["taper"])
            for obj in group
        ))
        for group in information["train"]
    }
    assert len(train_geometries) >= 2
    if any(len(group) > 1 for group in information["test"]):
        assert any(len(group) > 1 for group in information["train"])
    if any(obj["taper"] == 2 for obj in test_objects):
        assert any(obj["taper"] == 2 for obj in train_objects)
    assert all(len(group) in (1, 2) for group in information["train"] + information["test"])


def _puzzles_equal(a, b):
    return all(
        len(a[split]) == len(b[split]) and all(
            np.array_equal(x["input"], y["input"]) and
            np.array_equal(x["output"], y["output"])
            for x, y in zip(a[split], b[split])
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzles_equal(puzzles[17], generate_puzzle(17))
    assert _puzzles_equal(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3))
    signatures = set()
    directions = set()
    depths = set()
    tapers = set()
    ray_lengths = set()
    shapes = set()
    palettes = set()
    multi_train = multi_test = 0
    edge_touch = 0
    shorter_test = 0
    depth_regimes = [0, 0, 0]
    varied_background = common_direction = held_out_direction = 0
    role_exchange = longer_test = large_canvas = 0
    shared_multi_palette = mixed_multi_direction = combined_extension = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_rays = []
        test_rays = []
        puzzle_depths = set()
        section_objects = {"train": [], "test": []}
        backgrounds = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                signatures.add(_signature(inp))
                shapes.add(inp.shape)
                background, objects = _analyze_input(inp)
                backgrounds.add(background)
                section_objects[split].extend(objects)
                large_canvas += int(30 in inp.shape)
                if len(objects) == 2:
                    if split == "train":
                        multi_train += 1
                    else:
                        multi_test += 1
                    shared_multi_palette += int(
                        len({(obj["body"], obj["marker"]) for obj in objects}) == 1
                    )
                    mixed_multi_direction += int(
                        len({obj["direction"] for obj in objects}) == 2
                    )
                    combined_extension += int(any(
                        obj["taper"] == 2 and len(obj["ray"]) >= 8
                        for obj in objects
                    ))
                for obj in objects:
                    directions.add((split, obj["direction"]))
                    depths.add(obj["depth"])
                    puzzle_depths.add(obj["depth"])
                    tapers.add(obj["taper"])
                    ray_lengths.add(len(obj["ray"]))
                    palettes.add((background, obj["body"], obj["marker"]))
                    bounds = obj["bounds"]
                    if bounds[0] == 0 or bounds[1] == inp.shape[0] - 1 or bounds[2] == 0 or bounds[3] == inp.shape[1] - 1:
                        edge_touch += 1
                    (train_rays if split == "train" else test_rays).append(len(obj["ray"]))
        if min(test_rays) <= max(train_rays):
            shorter_test += 1
        longer_test += int(max(test_rays) > max(train_rays))
        varied_background += int(len(backgrounds) > 1)
        train_directions = {
            obj["direction"] for obj in section_objects["train"]
        }
        common_direction += int(len(train_directions) == 1)
        held_out_direction += int(any(
            obj["direction"] not in train_directions
            for obj in section_objects["test"]
        ))
        role_exchange += int(any(
            first["body"] == second["marker"]
            and first["marker"] == second["body"]
            for first in section_objects["train"]
            for second in section_objects["test"]
        ))
        depth_regimes[min(3, len(puzzle_depths)) - 1] += 1
    assert len(signatures) >= 150 and len(shapes) >= 50
    assert len(palettes) >= 60 and len(depths) >= 6
    assert len(tapers) == 2 and 1 in ray_lengths and max(ray_lengths) >= 15
    assert all((split, direction) in directions for split in ("train", "test") for direction in DIRECTIONS)
    assert multi_train >= 5 and multi_test >= 2 and edge_touch >= 10
    assert shorter_test >= 20
    assert all(count >= 3 for count in depth_regimes)
    assert varied_background >= 30 and common_direction >= 1
    assert held_out_direction >= 5 and role_exchange >= 3
    assert longer_test >= 5 and large_canvas >= 15
    assert shared_multi_palette >= 3 and mixed_multi_direction >= 5
    assert combined_extension >= 2
    damaged = {
        "train": list(puzzles[0]["train"]),
        "test": list(puzzles[0]["test"]),
    }
    first_test = damaged["test"][0]
    wrong_output = first_test["output"].copy()
    changed = np.argwhere(first_test["input"] != wrong_output)
    point = tuple(int(x) for x in changed[0])
    wrong_output[point] = first_test["input"][point]
    damaged["test"][0] = {
        "input": first_test["input"], "output": wrong_output,
    }
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong ray output accepted")
    for seed, train_count, test_count in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert len(puzzle["train"]) == train_count and len(puzzle["test"]) == test_count
    for seed, train_count, test_count in ((2001, None, 1), (2002, None, 4), (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for arguments in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4}, {"num_train": 2.0},
        {"num_test": True},
    ):
        try:
            generate_puzzle(4001, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
