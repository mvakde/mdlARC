"""Generate capped connectors from matching rectangles to hollow marker ends."""

import hashlib

import numpy as np


_DIRECTIONS = {"L": (0, -1), "R": (0, 1), "U": (-1, 0), "D": (1, 0)}
_NEIGHBORS = tuple(_DIRECTIONS.values())
_SOURCE_PAIR_DIGESTS = {
    "1a8b1ceeaa5a2cd10bdb413577e657a7b1b808de9e9c33c8c8fa0ff7a949d915",
    "efed8261b9cfcab080ffd30650dd160446ff85f072c136b8c83dde4620a45dcc",
    "75c2ec4f77b06a097141cf2baa1ae6b205e1af8eea86969aa0bd1d68d593db2f",
    "689e2ad2b0bd1c0987b9c311ea4540dc354bfd94c90793fcef6b290feb622533",
}


def _resolve_counts(rng, num_train, num_test):
    def clean(value, name, low, high):
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer or None")
        value = int(value)
        if not low <= value <= high:
            raise ValueError(f"{name} is outside its legal range")
        return value

    train = clean(num_train, "num_train", 2, 10)
    test = clean(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _grid_key(grid):
    return grid.shape, np.asarray(grid, dtype=np.uint8).tobytes()


def _pair_digest(input_grid, output_grid):
    data = b"".join(bytes(grid.shape) + np.asarray(grid, dtype=np.uint8).tobytes()
                    for grid in (input_grid, output_grid))
    return hashlib.sha256(data).hexdigest()


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) >= 2
    largest = int(np.argmax(counts))
    assert np.count_nonzero(counts == counts[largest]) == 1
    return int(colors[largest])


def _components(grid, color):
    remaining = {tuple(map(int, point)) for point in np.argwhere(grid == color)}
    components = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            row, col = stack.pop()
            for dr, dc in _NEIGHBORS:
                neighbor = (row + dr, col + dc)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _rectangle_cells(rect):
    top, left, bottom, right = rect
    return {(r, c) for r in range(top, bottom + 1) for c in range(left, right + 1)}


def _marker_direction(rect, marker):
    top, left, bottom, right = rect
    row, col = marker
    if top < row < bottom:
        if col < left - 2:
            return "L", left - col
        if col > right + 2:
            return "R", col - right
    if left < col < right:
        if row < top - 2:
            return "U", top - row
        if row > bottom + 2:
            return "D", row - bottom
    return None


def _action_geometry(rect, marker, direction, distance):
    top, left, bottom, right = rect
    row, col = marker
    dr, dc = _DIRECTIONS[direction]
    edge = {"L": (row, left), "R": (row, right),
            "U": (top, col), "D": (bottom, col)}[direction]
    transverse = (1, 0) if direction in ("L", "R") else (0, 1)
    tr, tc = transverse
    assert marker == (edge[0] + distance * dr, edge[1] + distance * dc)
    cap = (edge[0] + dr, edge[1] + dc)
    paint = {(cap[0] + k * tr, cap[1] + k * tc) for k in (-1, 0, 1)}
    paint.update((edge[0] + k * dr, edge[1] + k * dc)
                 for k in range(1, distance))
    paint.update(((row + tr, col + tc), (row - tr, col - tc),
                  (row + dr, col + dc)))
    return paint


def _analyze_input(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    height, width = grid.shape
    background = _background(grid)
    rectangles = []
    markers = []
    for color in np.unique(grid):
        color = int(color)
        if color == background:
            continue
        for component in _components(grid, color):
            if len(component) == 1:
                markers.append((color, next(iter(component))))
                continue
            rows = [point[0] for point in component]
            cols = [point[1] for point in component]
            rect = (min(rows), min(cols), max(rows), max(cols))
            assert rect[2] > rect[0] and rect[3] > rect[1]
            assert component == _rectangle_cells(rect)
            rectangles.append({"color": color, "rect": rect, "actions": []})
    assert rectangles and markers
    paint_union = set()
    centers = {point for _, point in markers}
    assert len(centers) == len(markers)
    for color, marker in markers:
        row, col = marker
        assert 0 < row < height - 1 and 0 < col < width - 1
        assert all(int(grid[row + dr, col + dc]) == background
                   for dr, dc in _NEIGHBORS)
        candidates = []
        for obj in rectangles:
            if obj["color"] == color:
                relation = _marker_direction(obj["rect"], marker)
                if relation is not None:
                    candidates.append((obj, relation))
        assert len(candidates) == 1
        obj, (direction, distance) = candidates[0]
        paint = _action_geometry(obj["rect"], marker, direction, distance)
        assert paint_union.isdisjoint(paint) and centers.isdisjoint(paint)
        for pr, pc in paint:
            assert 0 <= pr < height and 0 <= pc < width
            assert int(grid[pr, pc]) == background
        paint_union.update(paint)
        obj["actions"].append({"marker": marker, "direction": direction,
                               "distance": distance, "paint": paint})
    assert all(obj["actions"] for obj in rectangles)
    return {"background": background, "objects": rectangles}


def _solve_grid(grid):
    info = _analyze_input(grid)
    output = grid.copy()
    for obj in info["objects"]:
        for action in obj["actions"]:
            output[action["marker"]] = info["background"]
            for point in action["paint"]:
                output[point] = obj["color"]
    return output


def _check_output_invariants(input_grid, output_grid, info):
    background = info["background"]
    changed_locations = set()
    for obj in info["objects"]:
        color = obj["color"]
        top, left, bottom, right = obj["rect"]
        assert np.all(output_grid[top:bottom + 1, left:right + 1] == color)
        for action in obj["actions"]:
            row, col = action["marker"]
            direction, distance = action["direction"], action["distance"]
            dr, dc = _DIRECTIONS[direction]
            tr, tc = ((1, 0) if dr == 0 else (0, 1))
            edge_row, edge_col = row - distance * dr, col - distance * dc
            assert int(output_grid[row, col]) == background
            changed_locations.add((row, col))
            for step in range(1, distance):
                point = (edge_row + step * dr, edge_col + step * dc)
                assert int(output_grid[point]) == color
                changed_locations.add(point)
            for side in (-1, 1):
                cap = (edge_row + dr + side * tr, edge_col + dc + side * tc)
                rim = (row + side * tr, col + side * tc)
                assert int(output_grid[cap]) == color
                assert int(output_grid[rim]) == color
                changed_locations.update((cap, rim))
            outward = (row + dr, col + dc)
            assert int(output_grid[outward]) == color
            changed_locations.add(outward)
    unchanged = np.ones(input_grid.shape, dtype=bool)
    for point in changed_locations:
        unchanged[point] = False
    assert np.array_equal(input_grid[unchanged], output_grid[unchanged])


def _sets_touch(first, second):
    if first & second:
        return True
    return any((row + dr, col + dc) in second
               for row, col in first for dr, dc in _NEIGHBORS)


def _sample_candidate(rng, height, width, color, sides):
    minimum_height = max(2, 3 * max(sides.count("L"), sides.count("R")))
    minimum_width = max(2, 3 * max(sides.count("U"), sides.count("D")))
    top_margin = 4 if "U" in sides else 1
    bottom_margin = 4 if "D" in sides else 1
    left_margin = 4 if "L" in sides else 1
    right_margin = 4 if "R" in sides else 1
    height_max = min(12, height - top_margin - bottom_margin)
    width_max = min(12, width - left_margin - right_margin)
    if minimum_height > height_max or minimum_width > width_max:
        return None
    rect_height = int(rng.integers(minimum_height, height_max + 1))
    rect_width = int(rng.integers(minimum_width, width_max + 1))
    top = int(rng.integers(top_margin, height - bottom_margin - rect_height + 1))
    left = int(rng.integers(left_margin, width - right_margin - rect_width + 1))
    bottom, right = top + rect_height - 1, left + rect_width - 1
    rect = (top, left, bottom, right)
    rect_cells = _rectangle_cells(rect)
    centers, paint_union = set(), set()
    for side in rng.permutation(sides):
        if side in ("L", "R"):
            alignments = list(range(top + 1, bottom))
            room = left if side == "L" else width - 1 - right
        else:
            alignments = list(range(left + 1, right))
            room = top if side == "U" else height - 1 - bottom
        max_distance = min(10, room - 1)
        if max_distance < 3:
            return None
        accepted = False
        for alignment in rng.permutation(alignments):
            for distance in rng.permutation(np.arange(3, max_distance + 1)):
                alignment, distance = int(alignment), int(distance)
                marker = ((alignment, left - distance) if side == "L" else
                          (alignment, right + distance) if side == "R" else
                          (top - distance, alignment) if side == "U" else
                          (bottom + distance, alignment))
                paint = _action_geometry(rect, marker, side, distance)
                if (marker in paint_union or marker in centers or
                        not paint_union.isdisjoint(paint) or
                        not centers.isdisjoint(paint) or
                        paint & rect_cells):
                    continue
                if any((marker[0] + dr, marker[1] + dc) in centers
                       for dr, dc in _NEIGHBORS):
                    continue
                if any(not (0 <= r < height and 0 <= c < width) for r, c in paint):
                    continue
                centers.add(marker)
                paint_union.update(paint)
                accepted = True
                break
            if accepted:
                break
        if not accepted:
            return None
    return {"color": color, "rect": rect, "markers": centers,
            "input": rect_cells | centers, "paint": paint_union}


def _plan_sides(rng, object_count, mandatory, allowed):
    groups = [[] for _ in range(object_count)]
    for index, side in enumerate(rng.permutation(mandatory)):
        groups[index % object_count].append(str(side))
    for group in groups:
        if not group:
            group.append(str(rng.choice(allowed)))
    motif = int(rng.integers(0, 8))
    if motif in (0, 1, 2):
        choice = int(rng.integers(0, object_count))
        if len(groups[choice]) < 3:
            groups[choice].append(str(rng.choice(groups[choice])))
    if motif in (0, 3) and len(allowed) == 4:
        eligible = [i for i, group in enumerate(groups) if len(group) < 3]
        if eligible:
            choice = int(rng.choice(eligible))
            other_axis = [s for s in allowed if
                          (s in ("L", "R")) != (groups[choice][0] in ("L", "R"))]
            groups[choice].append(str(rng.choice(other_axis)))
    base = sum(map(len, groups))
    extra = int(rng.choice((0, 1, 2, 3), p=(0.46, 0.30, 0.17, 0.07)))
    target = min(6, base + extra)
    while sum(map(len, groups)) < target:
        eligible = [i for i, group in enumerate(groups) if len(group) < 3]
        if not eligible:
            break
        choice = int(rng.choice(eligible))
        groups[choice].append(str(rng.choice(allowed)))
    return groups


def _place_scene(rng, height, width, background, colors, side_groups):
    order = sorted(range(len(colors)), key=lambda i: -len(side_groups[i]))
    for _ in range(55):
        placed = []
        input_cells, paint_cells = set(), set()
        for index in order:
            accepted = None
            for _ in range(75):
                candidate = _sample_candidate(rng, height, width,
                                              colors[index], side_groups[index])
                if candidate is None:
                    continue
                if (_sets_touch(candidate["input"], input_cells) or
                        candidate["paint"] & (input_cells | paint_cells) or
                        candidate["input"] & paint_cells):
                    continue
                accepted = candidate
                break
            if accepted is None:
                break
            placed.append(accepted)
            input_cells.update(accepted["input"])
            paint_cells.update(accepted["paint"])
        if len(placed) != len(colors):
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        for obj in placed:
            top, left, bottom, right = obj["rect"]
            grid[top:bottom + 1, left:right + 1] = obj["color"]
            for marker in obj["markers"]:
                grid[marker] = obj["color"]
        try:
            _analyze_input(grid)
        except AssertionError:
            continue
        return grid
    return None


def _pair_palette(rng, object_count, background=None, required_color=None):
    if background is None:
        background = int(rng.integers(0, 10))
    assert required_color is None or required_color != background
    distinct = (1 if object_count == 1 else
                int(rng.choice((1, 2), p=(0.35, 0.65))) if object_count == 2 else
                int(rng.choice((1, 2, 3), p=(0.16, 0.38, 0.46))))
    choices = [color for color in range(10) if color != background and color != required_color]
    chosen = ([required_color] if required_color is not None else [])
    chosen += [int(c) for c in rng.choice(choices, size=max(0, distinct - len(chosen)),
                                        replace=False)]
    colors = [chosen[i % len(chosen)] for i in range(object_count)]
    rng.shuffle(colors)
    return background, colors


def _generate_pair(rng, regime, mandatory, allowed, background=None, required_color=None):
    for attempt in range(45):
        object_count = int(rng.choice((1, 2, 3), p=(0.29, 0.43, 0.28)
                                      if regime == "source_train" else
                                      (0.25, 0.35, 0.40)))
        if attempt >= 20:
            object_count = min(object_count, 2)
        side_groups = _plan_sides(rng, object_count, mandatory, allowed)
        minimum = (13, 17, 22)[object_count - 1]
        if rng.random() < 0.12 and object_count == 1:
            height, width = ((13, 16), (22, 16), (30, 30))[int(rng.integers(0, 3))]
        else:
            height = int(rng.integers(minimum, 31))
            width = int(rng.integers(minimum, 31))
        if attempt >= 28:
            height = width = 30
        bg, colors = _pair_palette(rng, object_count, background, required_color)
        grid = _place_scene(rng, height, width, bg, colors, side_groups)
        if grid is not None:
            return {"input": grid, "output": _solve_grid(grid)}
    raise RuntimeError("could not construct a valid scene")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    focus = int(rng.integers(0, 3))
    allowed = ("L", "R") if focus == 0 else (("U", "D") if focus == 1 else
                                                 ("L", "R", "U", "D"))
    order = [str(s) for s in rng.permutation(allowed)]
    if len(order) == 4:
        required_train = [order[:2], order[2:]]
    else:
        required_train = [[order[0]], [order[1]]]
    seeds = [int(s) for s in rng.integers(0, np.iinfo(np.int64).max,
                                          size=train_count + test_count)]
    puzzle = {"train": [], "test": []}
    seen = set()
    first_background = int(rng.integers(0, 10))
    second_background = int(rng.choice([c for c in range(10)
                                        if c != first_background]))
    for split, count, offset in (("train", train_count, 0),
                                 ("test", test_count, train_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(seeds[offset + index])
            mandatory = (required_train[index] if split == "train" and index < 2
                         else [])
            background = (first_background if split == "train" and index == 0 else
                          second_background if split == "train" and index == 1 else
                          None)
            required_color = (first_background if split == "train" and index == 1
                              else None)
            for _ in range(6):
                pair = _generate_pair(pair_rng, "source_train" if split == "train"
                                      else "target_test", mandatory, allowed,
                                      background, required_color)
                key = _grid_key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not generate a unique input")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    directions = {"train": set(), "test": set()}
    backgrounds = []
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert input_grid.shape == output_grid.shape
            assert not np.array_equal(input_grid, output_grid)
            input_key = _grid_key(input_grid)
            pair_key = (input_key, _grid_key(output_grid))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _pair_digest(input_grid, output_grid) not in _SOURCE_PAIR_DIGESTS
            info = _analyze_input(input_grid)
            assert np.array_equal(output_grid, _solve_grid(input_grid))
            _check_output_invariants(input_grid, output_grid, info)
            assert set(map(int, np.unique(input_grid))) == set(map(int, np.unique(output_grid)))
            backgrounds.append(info["background"])
            if split == "train" and index == 1:
                assert info["background"] != backgrounds[0]
                assert backgrounds[0] in {obj["color"] for obj in info["objects"]}
            for obj in info["objects"]:
                for action in obj["actions"]:
                    directions[split].add(action["direction"])
    assert directions["test"] <= directions["train"]
    assert len(set(backgrounds)) >= 2


def _puzzles_equal(first, second):
    return (all(len(first[split]) == len(second[split]) and
                all(np.array_equal(a[k], b[k]) for a, b in zip(first[split], second[split])
                    for k in ("input", "output")) for split in ("train", "test")))


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(73), generate_puzzle(73))
    assert _puzzles_equal(generate_puzzle(901, 7, 3), generate_puzzle(901, 7, 3))
    assert not _puzzles_equal(generate_puzzle(902, 7, 3), generate_puzzle(903, 7, 3))
    for train, test in ((2, 1), (10, 1), (10, 4),
                        (None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(2000 + (train or 0) * 10 +
                                                  (test or 0), train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (4, 4)):
        try:
            generate_puzzle(4000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")

    train_counts, test_counts, shapes, signatures = set(), set(), set(), set()
    features = {name: 0 for name in ("repeat_side", "triple_object", "perpendicular",
                                     "same_color", "square", "tall_vertical",
                                     "wide_horizontal", "single_test", "mixed_train",
                                     "simpler_test", "harder_test", "same_axis", "long_train",
                                     "long_test", "long_rectangle")}
    within = {name: 0 for name in ("objects", "markers", "colors", "directions")}
    for seed in range(1000, 1042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_complexity, test_complexity = [], []
        train_directions = set()
        test_directions = set()
        scene_values = {name: set() for name in within}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                info = _analyze_input(grid)
                shapes.add(grid.shape)
                signature = []
                scene_directions = set()
                marker_count = 0
                colors = []
                for obj in info["objects"]:
                    colors.append(obj["color"])
                    rect = obj["rect"]
                    rh, rw = rect[2] - rect[0] + 1, rect[3] - rect[1] + 1
                    sides = [a["direction"] for a in obj["actions"]]
                    marker_count += len(sides)
                    scene_directions.update(sides)
                    if len(sides) != len(set(sides)):
                        features["repeat_side"] += 1
                    if len(sides) >= 3:
                        features["triple_object"] += 1
                    if set(sides) & {"L", "R"} and set(sides) & {"U", "D"}:
                        features["perpendicular"] += 1
                    if rh == rw:
                        features["square"] += 1
                    if rh > rw and set(sides) & {"U", "D"}:
                        features["tall_vertical"] += 1
                    if rw > rh and set(sides) & {"L", "R"}:
                        features["wide_horizontal"] += 1
                    if max(rh, rw) >= 11:
                        features["long_rectangle"] += 1
                    for action in obj["actions"]:
                        if action["distance"] >= 9:
                            features["long_train" if split == "train" else
                                     "long_test"] += 1
                    signature.append((rh, rw, tuple(sorted(sides))))
                if len(colors) > len(set(colors)):
                    features["same_color"] += 1
                scene_values["objects"].add(len(info["objects"]))
                scene_values["markers"].add(marker_count)
                scene_values["colors"].add(len(set(colors)))
                scene_values["directions"].add(tuple(sorted(scene_directions)))
                if split == "test" and len(info["objects"]) == 1:
                    features["single_test"] += 1
                if split == "train" and scene_directions & {"L", "R"} and scene_directions & {"U", "D"}:
                    features["mixed_train"] += 1
                (train_complexity if split == "train" else test_complexity).append(
                    (len(info["objects"]), marker_count))
                (train_directions if split == "train" else test_directions).update(scene_directions)
                signatures.add((grid.shape, tuple(sorted(signature))))
        if (train_directions <= {"L", "R"} and test_directions <= {"L", "R"}) or (
                train_directions <= {"U", "D"} and test_directions <= {"U", "D"}):
            features["same_axis"] += 1
        if min(test_complexity) < max(train_complexity):
            features["simpler_test"] += 1
        if max(test_complexity) > min(train_complexity):
            features["harder_test"] += 1
        for name, values in scene_values.items():
            if len(values) > 1:
                within[name] += 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 45 and len(signatures) >= 100
    assert all(value > 0 for value in features.values()), features
    assert all(value >= 25 for value in within.values()), within
