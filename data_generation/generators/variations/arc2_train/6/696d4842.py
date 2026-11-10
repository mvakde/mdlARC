"""Generate independent path-and-beacon ARC transformations."""

import hashlib

import numpy as np


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_SOURCE_DIGESTS = {
    "de770ddab22688c1b10ce5c47e3e9178ce65b7e3ca54830bf07b1c4532f265c3",
    "de18e896b3f65817bf14cf48078c03a1f574c7415ea02982876f494f98907d5f",
    "10a615285b2548c4fcf6b757ec1c227fd47dc6dc05f806fcbcb9b0434b2adf10",
    "c305bd14156490f505aa254e874be625fc97c418015534a0b9915fdfd11c8223",
}


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _resolve_counts(seed, num_train, num_test, rng):
    if num_train is not None:
        if not _is_int(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _is_int(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _neighbors(cell):
    row, col = cell
    return ((row - 1, col), (row, col + 1), (row + 1, col), (row, col - 1))


def _inside(cell, shape):
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _analyze_input(grid):
    """Recover all roles and ordered routes using the visible input alone."""
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > grid.size // 2
    nonbackground = {tuple(map(int, cell)) for cell in np.argwhere(grid != background)}
    components = []
    while nonbackground:
        start = next(iter(nonbackground))
        nonbackground.remove(start)
        component = {start}
        stack = [start]
        while stack:
            for neighbor in _neighbors(stack.pop()):
                if neighbor in nonbackground:
                    nonbackground.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    paths = [component for component in components if len(component) > 1]
    beacons = [next(iter(component)) for component in components if len(component) == 1]
    assert 1 <= len(paths) <= 3 and len(paths) == len(beacons)
    role_colors = [int(grid[cell]) for cell in beacons]
    records = []
    for cells in paths:
        colors = {int(grid[cell]) for cell in cells}
        assert len(colors) == 1
        body_color = colors.pop()
        role_colors.append(body_color)
        degrees = {cell: sum(neighbor in cells for neighbor in _neighbors(cell)) for cell in cells}
        ends = [cell for cell, degree in degrees.items() if degree == 1]
        assert len(ends) == 2 and all(degree in (1, 2) for degree in degrees.values())
        records.append((cells, body_color, ends))
    assert len(set(role_colors)) == 2 * len(paths)
    found = []
    used = set()
    for cells, body_color, ends in records:
        candidates = []
        for endpoint in ends:
            inward = next(neighbor for neighbor in _neighbors(endpoint) if neighbor in cells)
            outward = (endpoint[0] - inward[0], endpoint[1] - inward[1])
            for beacon in beacons:
                offset = (beacon[0] - endpoint[0], beacon[1] - endpoint[1])
                distance = offset[0] * outward[0] + offset[1] * outward[1]
                if distance < 2 or offset != (distance * outward[0], distance * outward[1]):
                    continue
                if all(grid[endpoint[0] + step * outward[0], endpoint[1] + step * outward[1]] == background
                       for step in range(1, distance)):
                    candidates.append((endpoint, outward, beacon, distance - 1))
        assert len(candidates) == 1
        endpoint, outward, beacon, gap = candidates[0]
        assert beacon not in used
        used.add(beacon)
        other = ends[0] if ends[1] == endpoint else ends[1]
        route = [other]
        previous = None
        while route[-1] != endpoint:
            choices = [n for n in _neighbors(route[-1]) if n in cells and n != previous]
            assert len(choices) == 1
            previous, current = route[-1], choices[0]
            route.append(current)
            assert len(route) <= len(cells)
        assert len(route) == len(cells) and len(route) >= gap
        moves = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(route, route[1:])]
        bends = sum(a != b for a, b in zip(moves, moves[1:]))
        recolored_moves = moves[:max(0, gap - 1)]
        crossed_bends = sum(a != b for a, b in zip(recolored_moves, recolored_moves[1:]))
        found.append({
            "body": body_color, "beacon_color": int(grid[beacon]), "beacon": beacon,
            "route": route, "extension_endpoint": endpoint, "outward": outward,
            "gap": gap, "bends": bends, "crossed_bends": crossed_bends,
            "direction": _DIRECTIONS.index(outward),
        })
    assert len(used) == len(beacons)
    return background, sorted(found, key=lambda item: item["body"])


def _derive_output(grid, analysis=None):
    if analysis is None:
        analysis = _analyze_input(grid)
    _, objects = analysis
    output = grid.copy()
    for item in objects:
        row, col = item["extension_endpoint"]
        dr, dc = item["outward"]
        for step in range(1, item["gap"] + 1):
            output[row + step * dr, col + step * dc] = item["body"]
        for cell in item["route"][:item["gap"]]:
            output[cell] = item["beacon_color"]
    return output


def _choose_shape(rng, object_count):
    special = int(rng.integers(10))
    if special == 0:
        return 20, 20
    if special == 1:
        return 30, 30
    minimum = {1: 12, 2: 16, 3: 22}[object_count]
    height = int(rng.integers(minimum, 31))
    if special in (2, 3):
        width = height
    elif special in (4, 5):
        width = int(rng.integers(minimum, 31))
        if abs(width - height) < 5:
            width = 30 if height <= 21 else minimum
    else:
        width = int(rng.integers(minimum, 31))
    return height, width


def _choose_object_count(rng, regime, pair_index):
    if regime == "source_train" and pair_index == 0:
        return 2
    if regime == "target_test":
        return int(rng.choice((1, 2, 3), p=(0.12, 0.54, 0.34)))
    return int(rng.choice((1, 2, 3), p=(0.20, 0.60, 0.20)))


def _relative_shape_once(rng, outward_index, gap, bend_count, target_crossings):
    """Construct a self-avoiding polyline with the requested recolored prefix."""
    if target_crossings > bend_count or target_crossings > max(0, gap - 2):
        return None
    movement_count = gap - 1
    lengths = [int(rng.integers(1, 9)) for _ in range(bend_count + 1)]
    if bend_count and int(rng.integers(5)) == 0:
        lengths[int(rng.integers(len(lengths)))] = int(rng.integers(9, 14))
    if target_crossings:
        cuts = sorted(int(x) for x in rng.choice(np.arange(1, movement_count), size=target_crossings, replace=False))
        previous = 0
        for index, cut in enumerate(cuts):
            lengths[-index - 1] = cut - previous
            previous = cut
        lengths[-target_crossings - 1] = movement_count - previous + int(rng.integers(0, 4))
    else:
        lengths[-1] = max(1, movement_count) + int(rng.integers(0, 5))
    direction = (outward_index + 2) % 4
    directions = [direction]
    for _ in range(bend_count):
        direction = (direction + int(rng.choice((-1, 1)))) % 4
        directions.append(direction)
    route = [(0, 0)]
    cells = {(0, 0)}
    for direction, length in zip(directions, lengths):
        dr, dc = _DIRECTIONS[direction]
        for _ in range(length):
            new = (route[-1][0] + dr, route[-1][1] + dc)
            if new in cells or sum(neighbor in cells for neighbor in _neighbors(new)) != 1:
                return None
            route.append(new)
            cells.add(new)
    if len(route) < gap:
        return None
    dr, dc = _DIRECTIONS[outward_index]
    gap_cells = [(step * dr, step * dc) for step in range(1, gap + 1)]
    beacon = ((gap + 1) * dr, (gap + 1) * dc)
    if beacon in cells or any(cell in cells for cell in gap_cells):
        return None
    if any(neighbor in cells for neighbor in _neighbors(beacon)):
        return None
    for index, cell in enumerate(gap_cells):
        neighbors = {neighbor for neighbor in _neighbors(cell) if neighbor in cells}
        if neighbors != ({(0, 0)} if index == 0 else set()):
            return None
    return {"route": route, "gap_cells": gap_cells, "beacon": beacon,
            "footprint": cells | set(gap_cells) | {beacon}}


def _place_shape(rng, relative, shape, occupied):
    footprint = relative["footprint"]
    rows = [cell[0] for cell in footprint]
    cols = [cell[1] for cell in footprint]
    row_low, row_high = -min(rows), shape[0] - 1 - max(rows)
    col_low, col_high = -min(cols), shape[1] - 1 - max(cols)
    if row_low > row_high or col_low > col_high:
        return None
    shifts = [(r, c) for r in range(row_low, row_high + 1)
              for c in range(col_low, col_high + 1)]
    rng.shuffle(shifts)
    if int(rng.integers(3)) == 0:
        shifts.sort(key=lambda pair: pair[0] not in (row_low, row_high)
                    and pair[1] not in (col_low, col_high))
    blocked = set(occupied)
    for cell in occupied:
        blocked.update(_neighbors(cell))
    for row_shift, col_shift in shifts:
        moved = {(r + row_shift, c + col_shift) for r, c in footprint}
        if moved & blocked:
            continue
        def shift(cell):
            return cell[0] + row_shift, cell[1] + col_shift
        return {"route": [shift(c) for c in relative["route"]],
                "gap_cells": [shift(c) for c in relative["gap_cells"]],
                "beacon": shift(relative["beacon"]), "footprint": moved}
    return None


def _generate_pair(rng, regime, pair_index):
    assert regime in ("source_train", "target_test")
    for _ in range(140):
        object_count = _choose_object_count(rng, regime, pair_index)
        shape = _choose_shape(rng, object_count)
        background = 0 if int(rng.integers(5)) else int(rng.integers(1, 10))
        palette = list(map(int, rng.choice([c for c in range(10) if c != background],
                                           size=2 * object_count, replace=False)))
        occupied = set()
        objects = []
        for index in range(object_count):
            placed = None
            for _ in range(70):
                gap = int(rng.integers(1, 9 if regime == "source_train" else 12))
                if regime == "source_train" and pair_index == 0 and index == 0:
                    gap = max(3, gap)
                bends = int(rng.choice((0, 1, 2, 3, 4, 5), p=(.10, .22, .28, .22, .13, .05)))
                if regime == "source_train" and pair_index == 0 and index == 0:
                    bends = max(1, bends)
                feasible = min(2, bends, gap - 2)
                crossing = int(rng.choice(np.arange(max(0, feasible) + 1)))
                if regime == "source_train" and pair_index == 0 and index == 0:
                    crossing = 1
                direction = int(rng.integers(4))
                relative = _relative_shape_once(rng, direction, gap, bends, crossing)
                if relative is None:
                    continue
                placed = _place_shape(rng, relative, shape, occupied)
                if placed is not None:
                    break
            if placed is None:
                break
            placed["body"] = palette[2 * index]
            placed["beacon_color"] = palette[2 * index + 1]
            occupied.update(placed["footprint"])
            objects.append(placed)
        if len(objects) != object_count:
            continue
        grid = np.full(shape, background, dtype=np.int8)
        for item in objects:
            for cell in item["route"]:
                grid[cell] = item["body"]
            grid[item["beacon"]] = item["beacon_color"]
        try:
            analysis = _analyze_input(grid)
        except AssertionError:
            continue
        output = _derive_output(grid, analysis)
        return grid, output
    raise RuntimeError("could not construct an unambiguous path puzzle")


def _grid_key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _pair_digest(input_grid, output_grid):
    return hashlib.sha256(np.asarray(input_grid, dtype=np.int8).tobytes()
                          + np.asarray(output_grid, dtype=np.int8).tobytes()).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_grid, output_grid = pair["input"], pair["output"]
            assert input_grid.shape == output_grid.shape
            key = _grid_key(input_grid)
            pair_key = key, _grid_key(output_grid)
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert _pair_digest(input_grid, output_grid) not in _SOURCE_DIGESTS
            background, objects = _analyze_input(input_grid)
            assert np.array_equal(_derive_output(input_grid, (background, objects)), output_grid)
            assert not np.array_equal(input_grid, output_grid)
            assert set(np.unique(input_grid)) == set(np.unique(output_grid))
            assert all(item["gap"] >= 1 and item["gap"] <= len(item["route"])
                       for item in objects)
            assert all(item["gap"] <= (8 if split == "train" else 11)
                       and 0 <= item["bends"] <= 5 for item in objects)
    assert len(_analyze_input(puzzle["train"][0]["input"])[1]) == 2
    first_objects = _analyze_input(puzzle["train"][0]["input"])[1]
    assert any(item["gap"] > 1 and item["crossed_bends"] >= 1
               for item in first_objects)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not _is_int(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test, rng)
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, split, count in (("source_train", "train", num_train),
                                 ("target_test", "test", num_test)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            for _ in range(12):
                input_grid, output_grid = _generate_pair(pair_rng, regime, index)
                if _grid_key(input_grid) not in seen:
                    break
            else:
                raise RuntimeError("could not generate a unique pair")
            seen.add(_grid_key(input_grid))
            puzzle[split].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(left, right):
    return all(len(left[split]) == len(right[split]) and
               all(np.array_equal(a[key], b[key])
                   for a, b in zip(left[split], right[split]) for key in ("input", "output"))
               for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, palettes, layouts, counts, gaps, bends, crossings, backgrounds = (set() for _ in range(8))
    same_direction = equal_gaps = edge_objects = short_crossing = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                background, objects = _analyze_input(grid)
                assert np.array_equal(_derive_output(grid), pair["output"])
                shapes.add(grid.shape)
                palettes.add(tuple(sorted(int(c) for c in np.unique(grid) if c != background)))
                backgrounds.add(background)
                counts.add(len(objects))
                gaps.update(item["gap"] for item in objects)
                bends.update(item["bends"] for item in objects)
                crossings.update(item["crossed_bends"] for item in objects)
                layouts.add((grid.shape, tuple(sorted((item["direction"], item["gap"],
                            item["bends"], len(item["route"]), item["extension_endpoint"])
                            for item in objects))))
                if len(objects) >= 2:
                    same_direction += len({item["direction"] for item in objects}) == 1
                    equal_gaps += len({item["gap"] for item in objects}) == 1
                edge_objects += any(any(r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1
                                        for r, c in item["route"] + [item["beacon"]]) for item in objects)
                short_crossing += any(item["crossed_bends"] and item["gap"] <= 5 for item in objects)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 35 and any(h == w for h, w in shapes)
    assert any(h == 30 or w == 30 for h, w in shapes)
    assert len(palettes) >= 30 and len(backgrounds) >= 2
    assert counts == {1, 2, 3}
    assert {1, 2, 3, 4, 5, 6, 7, 8}.issubset(gaps) and max(gaps) >= 9
    assert {0, 1, 2, 3, 4}.issubset(bends)
    assert {0, 1, 2}.issubset(crossings)
    assert len(layouts) >= 130
    assert same_direction and equal_gaps and edge_objects and short_crossing
    first = generate_puzzle(314159, 6, 3)
    assert _puzzles_equal(first, generate_puzzle(314159, 6, 3))
    assert not _puzzles_equal(first, generate_puzzle(314160, 6, 3))
    assert _puzzles_equal(generate_puzzle(271828), generate_puzzle(271828))
    for seed, train, test in ((500, 2, 1), (501, 10, 1), (502, 10, 4),
                              (600, None, 1), (601, None, 4),
                              (602, 2, None), (603, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(700, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
