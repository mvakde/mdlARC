import numpy as np


_RIGHT = (0, 1)
_LEFT = (0, -1)
_DOWN = (1, 0)
_UP = (-1, 0)
_ZERO = (0, 0)
_DIRECTIONS = (_RIGHT, _LEFT, _DOWN, _UP)

# Rejecting an original input is stronger than merely rejecting the corresponding
# original solved pair.
_SOURCE_INPUT_ROWS = (
    (
        "88888888",
        "18886888",
        "81194388",
        "18882888",
        "88888888",
        "88878881",
        "88546118",
        "88808881",
    ),
    (
        "88888",
        "88888",
        "87888",
        "64588",
        "83888",
        "81888",
        "81888",
        "18188",
    ),
    (
        "8888181881818",
        "8888818888188",
        "8888818888188",
        "8588898888288",
        "7408743880438",
        "8688858888688",
        "8188888888888",
        "8188888888888",
        "1818888888888",
    ),
    (
        "18880888888888888",
        "81134788888888888",
        "18885888888888888",
        "88888888888888888",
        "88888888888878881",
        "88888888888549118",
        "88888888888868881",
        "88888888888888888",
        "88888888888888888",
        "18889888888888888",
        "81104588888888888",
        "18886888888888888",
        "88888888888888888",
        "18885888888888888",
        "81174388888888888",
        "18880888888888888",
        "88888888888888888",
        "88888888888888888",
    ),
)


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _add(a: tuple[int, int],
         b: tuple[int, int],
         scale: int = 1) -> tuple[int, int]:
    return (a[0] + scale * b[0], a[1] + scale * b[1])


def _scale(a: tuple[int, int], factor: int) -> tuple[int, int]:
    return (factor * a[0], factor * a[1])


def _perpendicular(direction: tuple[int, int]) -> tuple[int, int]:
    return (-direction[1], direction[0])


def _inside(position: tuple[int, int], height: int, width: int) -> bool:
    return (
        0 <= position[0] < height
        and 0 <= position[1] < width
    )


def _pointer_positions(
    base: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    perpendicular = _perpendicular(direction)
    return (
        _add(base, perpendicular),
        _add(base, perpendicular, -1),
        _add(base, direction),
        _add(base, direction, 2),
    )


def _cross_offsets(
    direction: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    perpendicular = _perpendicular(direction)
    return (
        _scale(direction, -1),
        _ZERO,
        direction,
        perpendicular,
        _scale(perpendicular, -1),
    )


def _destination_base(
    base: tuple[int, int],
    direction: tuple[int, int],
    height: int,
    width: int,
) -> tuple[int, int]:
    if direction == _RIGHT:
        return (base[0], width - 6)
    if direction == _LEFT:
        return (base[0], 5)
    if direction == _DOWN:
        return (height - 6, base[1])
    if direction == _UP:
        return (5, base[1])
    raise AssertionError("invalid direction")


def _candidate_bases(
    height: int,
    width: int,
    direction: tuple[int, int],
):
    if direction == _RIGHT:
        for row in range(1, height - 1):
            yield (row, 0)
    elif direction == _LEFT:
        for row in range(1, height - 1):
            yield (row, width - 1)
    elif direction == _DOWN:
        for col in range(1, width - 1):
            yield (0, col)
    elif direction == _UP:
        for col in range(1, width - 1):
            yield (height - 1, col)


def _parse_objects(grid, background, pointer_color):
    height, width = grid.shape
    found = []
    for direction in _DIRECTIONS:
        for base in _candidate_bases(height, width, direction):
            if int(grid[base]) != background:
                continue
            pointer = _pointer_positions(base, direction)
            center = _add(base, direction, 4)
            offsets = _cross_offsets(direction)
            cross_cells = tuple(_add(center, offset) for offset in offsets)
            cells = pointer + cross_cells
            if not all(_inside(pos, height, width) for pos in cells):
                continue
            if not all(int(grid[pos]) == pointer_color for pos in pointer):
                continue
            values = tuple(int(grid[pos]) for pos in cross_cells)
            if any(value in (background, pointer_color) for value in values):
                continue
            found.append({"base": base, "direction": direction,
                          "cross": dict(zip(offsets, values)), "cells": frozenset(cells)})
    if not found:
        return None
    used = set()
    for obj in found:
        if used.intersection(obj["cells"]):
            return None
        used.update(obj["cells"])
    nonbackground = {tuple(map(int, pos)) for pos in np.argwhere(grid != background)}
    return found if used == nonbackground else None


def _dominant_color(grid):
    counts = np.bincount(grid.astype(np.int64, copy=False).ravel(), minlength=10)
    maxima = np.flatnonzero(counts == counts.max())
    _ensure(len(maxima) == 1, "background is ambiguous")
    return int(maxima[0])


def _infer_objects(grid):
    background = _dominant_color(grid)
    choices = []
    for color in range(10):
        if color == background:
            continue
        objects = _parse_objects(grid, background, color)
        if objects is not None:
            choices.append((color, objects))
    _ensure(len(choices) == 1, "fork color or objects are ambiguous")
    return background, choices[0][0], choices[0][1]


def _expected_output(input_grid, background, pointer_color, objects):
    height, width = input_grid.shape
    output = np.full((height, width), background, dtype=input_grid.dtype)
    for obj in objects:
        direction = obj["direction"]
        base = _destination_base(obj["base"], direction, height, width)
        pointer = _pointer_positions(base, direction)
        center = _add(base, direction, 4)
        positions = [(pos, pointer_color) for pos in pointer]
        positions += [(_add(center, offset, -1), value)
                      for offset, value in obj["cross"].items()]
        for pos, value in positions:
            _ensure(_inside(pos, height, width), "destination leaves grid")
            _ensure(int(output[pos]) == background, "destination collision")
            output[pos] = value
    return output


def _grid_value_key(grid):
    return (tuple(grid.shape), np.asarray(grid, dtype=np.uint8).tobytes())


def _puzzle_signature(puzzle):
    return tuple((split, tuple((_grid_value_key(pair["input"]),
                                _grid_value_key(pair["output"]))
                               for pair in puzzle[split]))
                 for split in ("train", "test"))


def _matches_source_input(grid):
    return any(grid.shape == (len(rows), len(rows[0])) and
               all(''.join(str(int(value)) for value in grid[i]) == row
                   for i, row in enumerate(rows))
               for rows in _SOURCE_INPUT_ROWS)


def _validate_optional_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    bounds = (2, 10) if name == "num_train" else (1, 4)
    if not bounds[0] <= value <= bounds[1]:
        raise ValueError(f"{name} outside permitted range")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train = _validate_optional_count(num_train, "num_train")
    test = _validate_optional_count(num_test, "num_test")
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _sample_parameters(rng):
    colors = [int(x) for x in rng.permutation(10)]
    arm_count = int(rng.integers(2, 8))
    arms = tuple(colors[:arm_count])
    nonarms = colors[arm_count:]
    role_colors = tuple(nonarms[:3])
    variable_roles = bool(rng.integers(0, 3) == 0)
    novel = arm_count >= 3 and bool(rng.integers(0, 3) == 0)
    return {"arms": arms, "roles": role_colors, "variable": variable_roles,
            "novel": novel}


def _roles(parameters, index):
    colors = parameters["roles"]
    shift = index % 3 if parameters["variable"] else 0
    return tuple(colors[(shift + j) % 3] for j in range(3))


def _sample_lanes(rng, transverse_length, count):
    minimum = 4 * count - 1
    _ensure(transverse_length >= minimum, "lanes do not fit")
    extras = rng.multinomial(transverse_length - minimum,
                             np.full(count + 1, 1 / (count + 1)))
    lanes = [1 + int(extras[0])]
    for i in range(1, count):
        lanes.append(lanes[-1] + 4 + int(extras[i]))
    return lanes


def _arm_values(rng, palette, strong=False, novel=None):
    palette = list(palette)
    if novel is not None:
        picks = [novel, int(rng.choice(palette)), int(rng.choice(palette)),
                 int(rng.choice(palette))]
    else:
        size = int(rng.integers(2, min(4, len(palette)) + 1))
        subset = [int(x) for x in rng.choice(palette, size=size, replace=False)]
        picks = [int(rng.choice(subset)) for _ in range(4)]
    if strong:
        a, b = [int(x) for x in rng.choice(palette, size=2, replace=False)]
        picks = [a, b, b, a]
    elif picks[0] == picks[1] and picks[2] == picks[3]:
        # Every object must make the cross turn visibly meaningful.
        picks[1] = next(value for value in palette if value != picks[0])
    return picks


def _spec(base, direction, center_color, arms):
    perpendicular = _perpendicular(direction)
    offsets = (_scale(direction, -1), direction, perpendicular,
               _scale(perpendicular, -1))
    cross = {_ZERO: center_color}
    cross.update(zip(offsets, arms))
    return {"base": base, "direction": direction, "cross": cross}


def _positions(spec, height, width, destination=False):
    base = (_destination_base(spec["base"], spec["direction"], height, width)
            if destination else spec["base"])
    center = _add(base, spec["direction"], 4)
    return set(_pointer_positions(base, spec["direction"])) | {
        _add(center, offset) for offset in spec["cross"]}


def _separated(specs, height, width):
    for destination in (False, True):
        groups = [_positions(spec, height, width, destination) for spec in specs]
        if any(not all(_inside(pos, height, width) for pos in group)
               for group in groups):
            return False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if any(max(abs(a[0]-b[0]), abs(a[1]-b[1])) <= 1
                       for a in groups[i] for b in groups[j]):
                    return False
    return True


def _make_pair(rng, directions, parameters, role_index, regime,
               strong=False, novel=False):
    mixed = any(d[0] == 0 for d in directions) and any(d[1] == 0 for d in directions)
    count = len(directions)
    if mixed:
        height = int(rng.integers(16, 23 if regime == "source_train" else 31))
        width = int(rng.integers(16, 23 if regime == "source_train" else 31))
    else:
        horizontal = directions[0][0] == 0
        if regime == "source_train":
            active = int(rng.integers(7, 15))
            transverse = int(rng.integers(max(5, 4*count-1),
                                          min(24, max(5, 4*count-1)+11)+1))
        else:
            active = (int(rng.integers(7, 15)) if rng.random() < .35
                      else int(rng.integers(15, 31)))
            transverse = int(rng.integers(max(5, 4*count-1), 31))
        height, width = ((transverse, active) if horizontal
                         else (active, transverse))
    background, pointer, center = _roles(parameters, role_index)
    palette = parameters["arms"]
    if regime == "source_train" and parameters["novel"]:
        palette = palette[:-1]
    specs = []
    if not mixed:
        lanes = _sample_lanes(rng, height if directions[0][0] == 0 else width, count)
        for lane, direction in zip(lanes, directions):
            base = ((lane, 0 if direction == _RIGHT else width-1)
                    if direction[0] == 0 else
                    (0 if direction == _DOWN else height-1, lane))
            arms = _arm_values(rng, palette, strong and not specs,
                               parameters["arms"][-1] if novel and not specs else None)
            specs.append(_spec(base, direction, center, arms))
    else:
        _ensure(count == 2, "mixed pairs have two objects")
        horizontal = next(d for d in directions if d[0] == 0)
        vertical = next(d for d in directions if d[1] == 0)
        # Put the two axes in opposite corners at both endpoints.
        row = 2 if vertical == _DOWN else height-3
        col = width-3 if horizontal == _RIGHT else 2
        bases = ((row, 0 if horizontal == _RIGHT else width-1),
                 (0 if vertical == _DOWN else height-1, col))
        for direction, base in zip((horizontal, vertical), bases):
            arms = _arm_values(rng, palette, strong and not specs,
                               parameters["arms"][-1] if novel and not specs else None)
            specs.append(_spec(base, direction, center, arms))
    _ensure(_separated(specs, height, width), "objects are not separated")
    return _render_specs(height, width, specs, background, pointer)


def _render_specs(height, width, specs, background, pointer):
    inp = np.full((height, width), background, dtype=np.int8)
    out = np.full((height, width), background, dtype=np.int8)
    for spec in specs:
        direction = spec["direction"]
        for destination, grid in ((False, inp), (True, out)):
            base = (_destination_base(spec["base"], direction, height, width)
                    if destination else spec["base"])
            center = _add(base, direction, 4)
            for pos in _pointer_positions(base, direction):
                _ensure(int(grid[pos]) == background, "pointer collision")
                grid[pos] = pointer
            for offset, value in spec["cross"].items():
                pos = _add(center, offset, -1 if destination else 1)
                _ensure(int(grid[pos]) == background, "cross collision")
                grid[pos] = value
    return inp, out


def _training_plan(rng, count):
    axes = [( _RIGHT, _LEFT), (_DOWN, _UP)]
    if rng.random() < .5:
        axes.reverse()
    first_axis, second_axis = axes
    mode = int(rng.integers(0, 4))
    if mode == 0:
        first = (first_axis[int(rng.integers(0, 2))],)
    elif mode == 1:
        first = (first_axis[int(rng.integers(0, 2))],) * 2
    elif mode == 2:
        first = first_axis
    else:
        first = (first_axis[0], first_axis[1], first_axis[int(rng.integers(0, 2))])
    second = second_axis if rng.random() < .7 else (second_axis[int(rng.integers(0, 2))],)
    plan = [first, second]
    if count >= 3:
        seen = set(first) | set(second)
        missing = [d for d in _DIRECTIONS if d not in seen]
        if missing:
            if len(missing) == 2 and missing[0][0] != missing[1][0]:
                plan.append(tuple(missing))
            else:
                plan.append(tuple(missing))
        else:
            plan.append(tuple(first_axis))
    while len(plan) < count:
        if rng.random() < .18:
            plan.append((first_axis[int(rng.integers(0, 2))],
                         second_axis[int(rng.integers(0, 2))]))
        else:
            axis = axes[int(rng.integers(0, 2))]
            size = int(rng.integers(1, 4))
            plan.append(tuple(axis[int(rng.integers(0, 2))] for _ in range(size)))
    return plan


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    parameters = _sample_parameters(rng)
    plan = _training_plan(rng, train_count)
    puzzle = {"train": [], "test": []}
    used = set()
    for i, directions in enumerate(plan):
        for attempt in range(100):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            inp, out = _make_pair(pair_rng, directions, parameters, i,
                                  "source_train", strong=i == 0)
            key = _grid_value_key(inp)
            if key not in used and not _matches_source_input(inp):
                break
        else:
            raise RuntimeError("could not construct distinct training pair")
        used.add(key)
        puzzle["train"].append({"input": inp, "output": out})
    observed = set(d for group in plan for d in group)
    mixed_train = any(any(d[0] == 0 for d in group) and
                      any(d[1] == 0 for d in group) for group in plan)
    training_role_count = min(3, train_count) if parameters["variable"] else 1
    for i in range(test_count):
        for attempt in range(100):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            if mixed_train and pair_rng.random() < .2:
                directions = (tuple(d for d in observed if d[0] == 0)[0],
                              tuple(d for d in observed if d[1] == 0)[0])
            else:
                axis = int(pair_rng.integers(0, 2))
                available = [d for d in observed if (d[0] == 0) == (axis == 0)]
                count = int(pair_rng.integers(1, 5))
                directions = tuple(available[int(pair_rng.integers(0, len(available)))]
                                   for _ in range(count))
            role_index = int(pair_rng.integers(0, training_role_count))
            inp, out = _make_pair(pair_rng, directions, parameters, role_index,
                                  "target_test", novel=parameters["novel"] and i == 0)
            key = _grid_value_key(inp)
            if key not in used and not _matches_source_input(inp):
                break
        else:
            raise RuntimeError("could not construct distinct test pair")
        used.add(key)
        puzzle["test"].append({"input": inp, "output": out})
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _ensure(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
            "puzzle schema")
    _ensure(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
            "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _ensure(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    used = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _ensure(isinstance(pair, dict) and set(pair) == {"input", "output"},
                    "pair schema")
            for grid in pair.values():
                _ensure(isinstance(grid, np.ndarray) and grid.ndim == 2,
                        "grid must be 2D NumPy array")
                _ensure(all(1 <= x <= 30 for x in grid.shape), "grid dimensions")
                _ensure(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                _ensure(int(grid.min()) >= 0 and int(grid.max()) <= 9, "ARC colors")
            inp, out = pair["input"], pair["output"]
            _ensure(inp.shape == out.shape and not np.array_equal(inp, out),
                    "grid dimensions or unchanged output")
            key = _grid_value_key(inp)
            _ensure(key not in used and not _matches_source_input(inp),
                    "duplicate or source input")
            used.add(key)
            background, pointer, objects = _infer_objects(inp)
            _ensure(_dominant_color(out) == background, "output background")
            _ensure(np.array_equal(out, _expected_output(inp, background, pointer, objects)),
                    "wrong border slide or cross turn")
            specs = [{"base": obj["base"], "direction": obj["direction"],
                      "cross": obj["cross"]} for obj in objects]
            _ensure(_separated(specs, *inp.shape), "object footprints collide")
            _ensure(1 <= len(objects) <= (3 if split == "train" else 4),
                    "object count")
            roles = (background, pointer, objects[0]["cross"][_ZERO])
            _ensure(all(obj["cross"][_ZERO] == roles[2] for obj in objects),
                    "centers differ inside pair")
            for obj in objects:
                vals = obj["cross"]
                d, p = obj["direction"], _perpendicular(obj["direction"])
                _ensure(vals[d] != vals[_scale(d,-1)] or
                        vals[p] != vals[_scale(p,-1)], "invisible cross turn")
            analyses[split].append((inp.shape, objects, roles))
    trained_directions = {obj["direction"] for _, objects, _ in analyses["train"]
                          for obj in objects}
    _ensure(any(d[0] == 0 for d in trained_directions) and
            any(d[1] == 0 for d in trained_directions), "training lacks an axis")
    trained_roles = {roles for _, _, roles in analyses["train"]}
    trained_mixed = any({d[0] == 0 for d in (obj["direction"] for obj in objects)}
                        == {True, False} for _, objects, _ in analyses["train"])
    strong_demo = any((obj["cross"][obj["direction"]] !=
                       obj["cross"][_scale(obj["direction"], -1)] and
                       obj["cross"][_perpendicular(obj["direction"])] !=
                       obj["cross"][_scale(_perpendicular(obj["direction"]), -1)])
                      for _, objects, _ in analyses["train"] for obj in objects)
    _ensure(strong_demo, "training does not demonstrate both swaps")
    for shape, objects, _ in analyses["train"]:
        mixed = len({obj["direction"][0] == 0 for obj in objects}) > 1
        if not mixed:
            active = shape[1] if objects[0]["direction"][0] == 0 else shape[0]
            _ensure(7 <= active <= 14, "training travel range")
    for _, objects, roles in analyses["test"]:
        _ensure(roles in trained_roles, "test role assignment unseen in training")
        _ensure(all(obj["direction"] in trained_directions for obj in objects),
                "test direction unseen in training")
        mixed = len({obj["direction"][0] == 0 for obj in objects}) > 1
        _ensure(not mixed or trained_mixed, "mixed test lacks demonstration")


def self_test() -> None:
    train_counts, test_counts, shapes, layouts, role_sets = set(), set(), set(), set(), set()
    features = {"singleton_first": 0, "same_first": 0, "vertical_test": 0,
                "four_horizontal_long": 0, "four_vertical_long": 0,
                "short_test": 0, "test_counts": set(), "same_test": 0,
                "repeated": 0, "unchanged_pair": 0, "novel_color": 0,
                "varying_roles": 0, "mixed_test": 0, "mixed_train": 0,
                "small_test": 0, "within_palette_variation": 0,
                "palettes": set(), "travel": set(), "axes": set(),
                "role_palette_combinations": set()}
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        first = _infer_objects(puzzle["train"][0]["input"])[2]
        features["singleton_first"] += len(first) == 1
        features["same_first"] += len(first) > 1 and len({o["direction"] for o in first}) == 1
        train_colors = set()
        train_roles = set()
        train_palettes = set()
        max_train_height = max(pair["input"].shape[0] for pair in puzzle["train"])
        max_train_width = max(pair["input"].shape[1] for pair in puzzle["train"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                shape = inp.shape
                bg, pointer, objects = _infer_objects(inp)
                shapes.add(shape)
                layouts.add((shape, tuple(sorted((o["base"], o["direction"]) for o in objects))))
                role = (bg, pointer, objects[0]["cross"][_ZERO])
                role_sets.add(role)
                if split == "train":
                    train_roles.add(role)
                else:
                    count = len(objects)
                    features["test_counts"].add(count)
                    features["vertical_test"] += all(o["direction"][1] == 0 for o in objects)
                    features["mixed_test"] += len({o["direction"][0] == 0 for o in objects}) > 1
                    features["same_test"] += len({o["direction"] for o in objects}) == 1
                    features["small_test"] += shape[0] <= max_train_height and shape[1] <= max_train_width
                    active = shape[1] if objects[0]["direction"][0] == 0 else shape[0]
                    features["short_test"] += active <= 14
                    features["four_horizontal_long"] += count == 4 and all(o["direction"][0] == 0 for o in objects) and shape[1] >= 16
                    features["four_vertical_long"] += count == 4 and all(o["direction"][1] == 0 for o in objects) and shape[0] >= 16
                arm_colors = set()
                for obj in objects:
                    values = obj["cross"]
                    d, p = obj["direction"], _perpendicular(obj["direction"])
                    object_arms = {value for offset, value in values.items() if offset != _ZERO}
                    arm_colors.update(object_arms)
                    features["repeated"] += len(object_arms) < 4
                    features["unchanged_pair"] += (values[d] == values[_scale(d,-1)] or
                                                   values[p] == values[_scale(p,-1)])
                features["palettes"].add(len(arm_colors))
                features["axes"].add((split, "mixed" if len({o["direction"][0] == 0 for o in objects}) > 1
                                      else "horizontal" if objects[0]["direction"][0] == 0 else "vertical"))
                features["role_palette_combinations"].add((role, len(arm_colors), len(objects)))
                for obj in objects:
                    features["travel"].add((shape[1] if obj["direction"][0] == 0 else shape[0]) - 6)
                if split == "train":
                    train_colors.update(arm_colors)
                    train_palettes.add(len(arm_colors))
                else:
                    features["novel_color"] += bool(arm_colors - train_colors)
        features["varying_roles"] += len(train_roles) > 1
        features["within_palette_variation"] += len(train_palettes) > 1
        features["mixed_train"] += any(
            len({o["direction"][0] == 0 for o in _infer_objects(pair["input"])[2]}) > 1
            for pair in puzzle["train"])
    _ensure(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity")
    _ensure(len(signatures) >= 40 and len(shapes) >= 30 and len(layouts) >= 100,
            "structural diversity")
    _ensure(len(role_sets) >= 20, "role diversity")
    for name in ("singleton_first", "same_first", "vertical_test", "four_horizontal_long",
                 "four_vertical_long", "short_test", "same_test", "repeated",
                 "unchanged_pair", "novel_color", "varying_roles", "mixed_train",
                 "mixed_test", "small_test", "within_palette_variation"):
        _ensure(features[name] > 0, f"missing diversity: {name}")
    _ensure(features["test_counts"] == {1,2,3,4}, "test object count diversity")
    _ensure(len(features["palettes"]) >= 3, "payload palette diversity")
    _ensure(len(features["travel"]) >= 15, "travel distance diversity")
    _ensure(len(features["role_palette_combinations"]) >= 40,
            "role, payload, and count combinations lack diversity")
    _ensure({("train", "horizontal"), ("train", "vertical"),
             ("test", "horizontal"), ("test", "vertical")}.issubset(features["axes"]),
            "axis diversity")
    for i, (nt, ns) in enumerate(((2,1), (10,1), (10,4), (6,3),
                                   (None,1), (None,4), (2,None), (10,None))):
        puzzle = generate_puzzle(1000+i, nt, ns)
        validate_generated_puzzle(puzzle)
        _ensure(nt is None or len(puzzle["train"]) == nt, "train count request")
        _ensure(ns is None or len(puzzle["test"]) == ns, "test count request")
    _ensure(_puzzle_signature(generate_puzzle(314159, 7, 3)) ==
            _puzzle_signature(generate_puzzle(314159, 7, 3)), "determinism")
    for i, kwargs in enumerate(({ "num_train": 1}, {"num_train":11},
                                 {"num_test":0}, {"num_test":5},
                                 {"num_train":2, "num_test":2},
                                 {"num_train":3, "num_test":4})):
        try:
            generate_puzzle(9000+i, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal count: {kwargs}")
