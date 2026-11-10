"""Compositional border-seeded, color-commanded route puzzles."""

import hashlib
import numpy as np


_SOURCE_DIGESTS = {
    "a2ca89dc70aa794b048ec121a20fbabac0a9511d540cefde266a4173a917aca0",
    "883af97043235691c583e2b7e785103037ce578f525859f53b03a107023ee2d8",
    "811d33b931eb6a153f7744979401f52de24d923ac60190d4eb5634020b5183ab",
    "4f83c68c71ee9944ac3c35bc724c8354382fb14273da11d5b610c05a34c34595",
    "334ae1d1a4536a3092feeb2f4eb2c5e0c71316d811f6ac690c52c4eea2e7e3b8",
}
_DIRECTIONS = ("down", "right", "up", "left")


def _require(value, message):
    if not value:
        raise AssertionError(message)


def _count(value, name, minimum, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _counts(seed, num_train, num_test):
    train = None if num_train is None else _count(num_train, "num_train", 2, 10)
    test = None if num_test is None else _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        low = max(2, test + 1)
        train = low + seed % (11 - low)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _dimensions(orientation, forward_size, transverse_size):
    if orientation in ("down", "up"):
        return forward_size, transverse_size
    return transverse_size, forward_size


def _coord(orientation, forward, transverse, shape):
    height, width = shape
    if orientation == "down":
        return forward, transverse
    if orientation == "up":
        return height - 1 - forward, transverse
    if orientation == "right":
        return transverse, forward
    return transverse, width - 1 - forward


def _axis_sizes(orientation, shape):
    if orientation in ("down", "up"):
        return shape
    return shape[1], shape[0]


def _seed_orientation(grid, route):
    spots = np.argwhere(grid == route)
    if spots.shape != (1, 2):
        raise ValueError("input needs exactly one route seed")
    row, col = (int(v) for v in spots[0])
    height, width = grid.shape
    if row == 0 and 0 < col < width - 1:
        return "down", col
    if row == height - 1 and 0 < col < width - 1:
        return "up", col
    if col == 0 and 0 < row < height - 1:
        return "right", row
    if col == width - 1 and 0 < row < height - 1:
        return "left", row
    raise ValueError("route seed must be on a noncorner edge")


def _left_sign(orientation):
    return 1 if orientation in ("down", "left") else -1


def _trace(grid, background, route, left, right):
    orientation, lane = _seed_orientation(grid, route)
    forward_size, transverse_size = _axis_sizes(orientation, grid.shape)
    output = grid.copy()
    collisions = []
    sign_for_left = _left_sign(orientation)
    for forward in range(1, forward_size):
        point = _coord(orientation, forward, lane, grid.shape)
        cell = int(grid[point])
        if cell == background:
            _require(int(output[point]) == background, "path revisits a cell")
            output[point] = route
            continue
        if cell not in (left, right):
            raise ValueError("route encountered a non-command cell")
        low = high = lane
        while low > 0 and int(grid[_coord(orientation, forward, low - 1, grid.shape)]) == cell:
            low -= 1
        while high + 1 < transverse_size and int(grid[_coord(orientation, forward, high + 1, grid.shape)]) == cell:
            high += 1
        sign = sign_for_left if cell == left else -sign_for_left
        target = high + 1 if sign > 0 else low - 1
        if not 0 <= target < transverse_size:
            raise ValueError("commanded bypass leaves the canvas")
        for cross_lane in range(lane + sign, target + sign, sign):
            detour = _coord(orientation, forward - 1, cross_lane, grid.shape)
            if int(grid[detour]) != background or int(output[detour]) != background:
                raise ValueError("near-side detour is obstructed")
            output[detour] = route
        crossing = _coord(orientation, forward, target, grid.shape)
        if int(grid[crossing]) != background or int(output[crossing]) != background:
            raise ValueError("outer crossing is obstructed")
        output[crossing] = route
        collisions.append((forward, lane, target, low, high, cell))
        lane = target
    return output, orientation, collisions


def _positions(rng, forward_size, count):
    base = 2 + 2 * np.arange(count)
    slack = forward_size - 2 - int(base[-1])
    if slack < 0:
        raise ValueError("collision count does not fit")
    offsets = np.sort(rng.integers(0, slack + 1, size=count))
    return [int(a + b) for a, b in zip(base, offsets)]


def _role_options(lane, role, transverse_size, orientation):
    sign = _left_sign(orientation) if role == 0 else -_left_sign(orientation)
    options = []
    for length in range(2, 6):
        for low in range(0, transverse_size - length + 1):
            high = low + length - 1
            if not low <= lane <= high:
                continue
            target = high + 1 if sign > 0 else low - 1
            if 0 <= target < transverse_size:
                options.append((low, high, target))
    return options


def _pick_option(rng, options, transverse_size, sweep):
    weights = []
    for low, high, target in options:
        length = high - low + 1
        weight = {2: 1.4, 3: 3.0, 4: 1.7, 5: 0.8}[length]
        if length == 2 and (low == 0 or high == transverse_size - 1):
            weight *= 3.0
        if sweep:
            weight *= 1 + abs(target - (low + high) / 2)
        weights.append(weight)
    weights = np.asarray(weights, dtype=float)
    return options[int(rng.choice(len(options), p=weights / weights.sum()))]


def _main_bars(rng, orientation, transverse_size, positions, train_anchor):
    edge_route = rng.random() < 0.18
    if edge_route:
        at_low = bool(rng.integers(0, 2))
        lane = 1 if at_low else transverse_size - 2
        first_sign = 1 if at_low else -1
        first_role = 0 if first_sign == _left_sign(orientation) else 1
    else:
        lane = int(rng.integers(2, transverse_size - 2))
        first_role = int(rng.integers(0, 2))
    start_lane = lane
    roles = [first_role]
    if train_anchor:
        roles.append(1 - first_role)
    while len(roles) < len(positions):
        roles.append(int(rng.integers(0, 2)))
    bars = []
    sweep = rng.random() < 0.4
    for index, (forward, role) in enumerate(zip(positions, roles)):
        if edge_route and index == 0:
            low, high = (0, 1) if start_lane == 1 else (transverse_size - 2, transverse_size - 1)
            sign = _left_sign(orientation) if role == 0 else -_left_sign(orientation)
            target = high + 1 if sign > 0 else low - 1
        else:
            options = _role_options(lane, role, transverse_size, orientation)
            if not options:
                if train_anchor and index < 2:
                    raise ValueError("anchor command cannot be placed")
                role = 1 - role
                options = _role_options(lane, role, transverse_size, orientation)
            low, high, target = _pick_option(rng, options, transverse_size, sweep)
        bars.append((forward, low, high, role))
        lane = target
    return start_lane, bars


def _draw(seed_lane, bars, orientation, shape, colors):
    background, route, left, right = colors
    grid = np.full(shape, background, dtype=np.int8)
    grid[_coord(orientation, 0, seed_lane, shape)] = route
    for forward, low, high, role in bars:
        for lane in range(low, high + 1):
            point = _coord(orientation, forward, lane, shape)
            if int(grid[point]) != background:
                raise ValueError("bars overlap or touch")
            grid[point] = (left, right)[role]
    return grid


def _bars_from_input(grid, orientation, command_colors):
    forward_size, transverse_size = _axis_sizes(orientation, grid.shape)
    bars = []
    for forward in range(forward_size):
        lane = 0
        while lane < transverse_size:
            cell = int(grid[_coord(orientation, forward, lane, grid.shape)])
            if cell not in command_colors:
                lane += 1
                continue
            low = lane
            while lane + 1 < transverse_size and int(grid[_coord(orientation, forward, lane + 1, grid.shape)]) in command_colors:
                lane += 1
            high = lane
            _require(2 <= high - low + 1 <= 5, "bar length must be 2 through 5")
            _require(all(int(grid[_coord(orientation, forward, t, grid.shape)]) == cell for t in range(low, high + 1)), "bar must have one color")
            _require(2 <= forward <= forward_size - 2, "bar forward position is illegal")
            bars.append((forward, low, high, cell))
            lane += 1
    _require(bool(bars), "input needs a command bar")
    return bars


def _add_distractors(rng, grid, orientation, colors, main_trace, desired):
    background, route, left, right = colors
    forward_size, transverse_size = _axis_sizes(orientation, grid.shape)
    main_levels = {entry[0] for entry in main_trace}
    placed = 0
    isolated = 0
    for _ in range(desired):
        accepted = False
        for _ in range(90):
            make_isolated = rng.random() < 0.38
            if make_isolated:
                levels = [f for f in range(2, forward_size - 1) if f not in main_levels and all(abs(f - m) >= 2 for m in main_levels)]
            else:
                levels = list(main_levels)
            if not levels:
                continue
            forward = int(rng.choice(levels))
            length = int(rng.choice((2, 2, 3, 3, 3, 4, 5)))
            if length >= transverse_size:
                continue
            if rng.random() < 0.25:
                low = 0 if rng.random() < 0.5 else transverse_size - length
            else:
                low = int(rng.integers(0, transverse_size - length + 1))
            high = low + length - 1
            # A blank cell separates distinct bars at the same level.
            if any(int(grid[_coord(orientation, forward, t, grid.shape)]) != background for t in range(max(0, low - 1), min(transverse_size, high + 2))):
                continue
            role = int(rng.integers(0, 2))
            trial = grid.copy()
            for lane in range(low, high + 1):
                trial[_coord(orientation, forward, lane, grid.shape)] = (left, right)[role]
            try:
                _, _, trace = _trace(trial, background, route, left, right)
                _bars_from_input(trial, orientation, {left, right})
            except (ValueError, AssertionError):
                continue
            if trace != main_trace:
                continue
            grid = trial
            placed += 1
            isolated += int(make_isolated)
            accepted = True
            break
        if not accepted:
            break
    return grid, placed, isolated


def _pair_colors(rng, commands):
    available = [color for color in range(10) if color not in commands]
    background, route = (int(x) for x in rng.choice(available, size=2, replace=False))
    return background, route, commands[0], commands[1]


def _generate_pair(rng, regime, commands, orientation, train_anchor):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    # Both regimes overlap. Tests visit long, compact, and sparse configurations.
    mode = rng.random()
    if mode < 0.26:
        forward_size = int(rng.integers(18, 25))
        transverse_size = int(rng.integers(10, 14))
    elif mode < 0.45:
        forward_size = int(rng.integers(26, 31))
        transverse_size = int(rng.integers(10, 17))
    else:
        forward_size = int(rng.integers(12, 31))
        transverse_size = int(rng.integers(9, 24))
    shape = _dimensions(orientation, forward_size, transverse_size)
    maximum = min(9, (forward_size - 2) // 2)
    draw = rng.random()
    if draw < 0.28 and not train_anchor:
        collision_count = int(rng.integers(1, min(3, maximum) + 1))
    elif draw < 0.63:
        lo, hi = min(3, maximum), min(6, maximum)
        collision_count = int(rng.integers(lo, hi + 1))
    else:
        collision_count = int(rng.integers(max(2, maximum - 2), maximum + 1))
    collision_count = max(2 if train_anchor else 1, collision_count)
    positions = _positions(rng, forward_size, collision_count)
    start_lane, bars = _main_bars(rng, orientation, transverse_size, positions, train_anchor)
    colors = _pair_colors(rng, commands)
    grid = _draw(start_lane, bars, orientation, shape, colors)
    output, _, main_trace = _trace(grid, *colors)
    _require(len(main_trace) == collision_count, "main bars were not all encountered")
    if train_anchor:
        _require({entry[-1] for entry in main_trace} == set(commands), "anchor needs both turns")
    # Some pairs are clean; others contain several close alternatives and off-level bars.
    density = rng.random()
    desired = 0 if density < 0.21 else int(rng.integers(1, (2 if density < 0.60 else 7) + 1))
    grid, _, _ = _add_distractors(rng, grid, orientation, colors, main_trace, desired)
    output, _, trace = _trace(grid, *colors)
    _require(trace == main_trace, "distractor changed the route")
    return {"input": grid, "output": output}


def _pair_roles(pair):
    grid = pair["input"]
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    _require(int(np.count_nonzero(grid == background)) > grid.size // 2, "background must dominate")
    candidates = []
    for color, count in zip(colors, counts):
        if int(color) == background or int(count) != 1:
            continue
        try:
            orientation, _ = _seed_orientation(grid, int(color))
        except ValueError:
            continue
        candidates.append((int(color), orientation))
    _require(len(candidates) == 1, "border seed must be uniquely identifiable")
    route, orientation = candidates[0]
    commands = {int(c) for c in colors} - {background, route}
    _require(1 <= len(commands) <= 2, "pair needs one or two command colors")
    return background, route, orientation, commands


def _infer_commands(puzzle):
    commands = set()
    for pair in puzzle["train"]:
        commands.update(_pair_roles(pair)[3])
    _require(len(commands) == 2, "training must show both command colors")
    first, second = sorted(commands)
    valid = []
    for left, right in ((first, second), (second, first)):
        matches = True
        for pair in puzzle["train"]:
            background, route, _, _ = _pair_roles(pair)
            try:
                output, _, _ = _trace(pair["input"], background, route, left, right)
            except (ValueError, AssertionError):
                matches = False
                break
            if not np.array_equal(output, pair["output"]):
                matches = False
                break
        if matches:
            valid.append((left, right))
    _require(len(valid) == 1, "training does not identify relative command meanings")
    return valid[0]


def _pair_digest(pair):
    grid = pair["input"].astype(np.uint8)
    output = pair["output"].astype(np.uint8)
    return hashlib.sha256(bytes(grid.shape) + grid.tobytes() + output.tobytes()).hexdigest()


def _pair_key(pair):
    grid = pair["input"]
    return grid.shape, grid.tobytes(), pair["output"].tobytes()


def _inspect_pair(pair, commands):
    grid, output = pair["input"], pair["output"]
    background, route, orientation, present = _pair_roles(pair)
    _require(present <= set(commands), "test has an unlearned command color")
    bars = _bars_from_input(grid, orientation, set(commands))
    predicted, _, trace = _trace(grid, background, route, *commands)
    _require(np.array_equal(predicted, output), "output disagrees with the input-only trace")
    _require(bool(trace), "route must meet a command")
    changed = grid != output
    _require(bool(np.any(changed)), "route must extend")
    _require(bool(np.all(grid[changed] == background)), "only background may change")
    _require(bool(np.all(output[changed] == route)), "traversal must use route color")
    _require(np.array_equal(grid[grid != background], output[grid != background]), "original cells changed")
    levels = {bar[0] for bar in bars}
    collisions = {entry[0] for entry in trace}
    counts = {level: sum(bar[0] == level for bar in bars) for level in levels}
    return {
        "orientation": orientation,
        "bars": bars,
        "trace": trace,
        "isolated": len(levels - collisions),
        "multi": sum(value > 1 for value in counts.values()),
        "clipped": sum(bar[2] - bar[1] + 1 == 2 and (bar[1] == 0 or bar[2] == _axis_sizes(orientation, grid.shape)[1] - 1) for bar in bars),
        "near_exit": any(entry[0] == _axis_sizes(orientation, grid.shape)[0] - 2 for entry in trace),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "pair counts")
    input_keys = set()
    pair_keys = set()
    for pair in puzzle["train"] + puzzle["test"]:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
        for name in ("input", "output"):
            grid = pair[name]
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid array and rank")
            _require(all(1 <= x <= 30 for x in grid.shape), "grid dimensions")
            _require(np.issubdtype(grid.dtype, np.integer), "grid integer dtype")
            _require(bool(np.all((grid >= 0) & (grid <= 9))), "grid colors")
        _require(pair["input"].shape == pair["output"].shape, "shape changes")
        key = _pair_key(pair)
        input_key = key[:2]
        _require(input_key not in input_keys and key not in pair_keys, "duplicate pair")
        input_keys.add(input_key)
        pair_keys.add(key)
        _require(_pair_digest(pair) not in _SOURCE_DIGESTS, "exact source pair")
    commands = _infer_commands(puzzle)
    details = [_inspect_pair(pair, commands) for pair in puzzle["train"] + puzzle["test"]]
    for index in (0, 1):
        _require({entry[-1] for entry in details[index]["trace"]} == set(commands), "anchor pair must show both turns")
    first, second = (details[index]["orientation"] for index in (0, 1))
    _require((first in ("down", "up")) != (second in ("down", "up")), "anchor orientations must be perpendicular")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    commands = tuple(int(v) for v in rng.choice(10, size=2, replace=False))
    first = int(rng.integers(0, 4))
    second = (first + int(rng.choice((1, 3)))) % 4
    train_orientations = [_DIRECTIONS[first], _DIRECTIONS[second]]
    train_orientations.extend(_DIRECTIONS[int(rng.integers(0, 4))] for _ in range(train_count - 2))
    test_orientations = []
    for _ in range(test_count):
        if rng.random() < 0.55:
            test_orientations.append(train_orientations[int(rng.integers(0, len(train_orientations)))])
        else:
            test_orientations.append(_DIRECTIONS[int(rng.integers(0, 4))])
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, orientations in (("train", train_orientations), ("test", test_orientations)):
        for index, orientation in enumerate(orientations):
            for _ in range(100):
                child_rng = np.random.default_rng(int(rng.integers(0, 2**63, dtype=np.int64)))
                try:
                    pair = _generate_pair(child_rng, "source_train" if split == "train" else "target_test", commands, orientation, split == "train" and index < 2)
                except (ValueError, AssertionError):
                    continue
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen and _pair_digest(pair) not in _SOURCE_DIGESTS:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("could not construct a unique valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal_puzzles(a, b):
    return all(len(a[split]) == len(b[split]) and all(np.array_equal(x[name], y[name]) for x, y in zip(a[split], b[split]) for name in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    stats = {name: set() for name in ("train_count", "test_count", "orientation", "shape", "commands", "layout", "length", "collision", "background", "route")}
    totals = {name: 0 for name in ("compact", "wide", "isolated", "multi", "clipped", "encountered_clipped", "near_exit", "one_color", "long", "sparse", "same_orientation", "mixed_test", "pair_color_change", "busy_right_train", "simple_down_test", "up_test", "left_test", "close_alternative", "half_width_sweep", "within_shape_change", "within_collision_change", "test_not_always_harder")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        stats["train_count"].add(len(puzzle["train"]))
        stats["test_count"].add(len(puzzle["test"]))
        commands = _infer_commands(puzzle)
        stats["commands"].add(commands)
        train_orientations = []
        test_orientations = []
        pair_colors = set()
        puzzle_shapes = set()
        puzzle_collision_counts = set()
        train_collision_counts = []
        test_collision_counts = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                details = _inspect_pair(pair, commands)
                orientation = details["orientation"]
                stats["orientation"].add(orientation)
                stats["shape"].add(pair["input"].shape)
                stats["collision"].add(len(details["trace"]))
                puzzle_shapes.add(pair["input"].shape)
                puzzle_collision_counts.add(len(details["trace"]))
                background, route, _, present = _pair_roles(pair)
                stats["background"].add(background)
                stats["route"].add(route)
                pair_colors.add((background, route))
                normalized = np.full(10, 0, dtype=np.int8)
                normalized[route] = 1
                normalized[commands[0]] = 2
                normalized[commands[1]] = 3
                stats["layout"].add((pair["input"].shape, normalized[pair["input"]].tobytes()))
                for bar in details["bars"]:
                    stats["length"].add(bar[2] - bar[1] + 1)
                forward_size, transverse_size = _axis_sizes(orientation, pair["input"].shape)
                totals["compact"] += int(forward_size >= 18 and transverse_size <= 13 and len(details["trace"]) >= 6)
                totals["wide"] += int(forward_size >= 26)
                totals["isolated"] += int(details["isolated"] > 0)
                totals["multi"] += int(details["multi"] > 0)
                totals["clipped"] += int(details["clipped"] > 0)
                totals["encountered_clipped"] += int(any(high - low + 1 == 2 and (low == 0 or high == transverse_size - 1) for _, _, _, low, high, _ in details["trace"]))
                totals["near_exit"] += int(details["near_exit"])
                totals["one_color"] += int(len(present) == 1)
                totals["long"] += int(len(details["trace"]) >= 7)
                totals["sparse"] += int(len(details["trace"]) <= 3 and len(details["bars"]) <= 4)
                totals["close_alternative"] += int(any(a[0] == b[0] and a[2] + 2 == b[1] for a in details["bars"] for b in details["bars"] if a != b))
                visited_lanes = [entry[1] for entry in details["trace"]] + [details["trace"][-1][2]]
                totals["half_width_sweep"] += int(max(visited_lanes) - min(visited_lanes) >= transverse_size / 2)
                if split == "train":
                    train_orientations.append(orientation)
                    train_collision_counts.append(len(details["trace"]))
                    totals["busy_right_train"] += int(orientation == "right" and len(details["trace"]) >= 6 and details["multi"] > 0)
                else:
                    test_orientations.append(orientation)
                    test_collision_counts.append(len(details["trace"]))
                    totals["simple_down_test"] += int(orientation == "down" and len(details["trace"]) <= 4)
                    totals["up_test"] += int(orientation == "up")
                    totals["left_test"] += int(orientation == "left")
        totals["same_orientation"] += int(any(o in train_orientations for o in test_orientations))
        totals["mixed_test"] += int(len(set(test_orientations)) > 1)
        totals["pair_color_change"] += int(len(pair_colors) > 1)
        totals["within_shape_change"] += int(len(puzzle_shapes) > 1)
        totals["within_collision_change"] += int(len(puzzle_collision_counts) > 1)
        totals["test_not_always_harder"] += int(min(test_collision_counts) <= max(train_collision_counts))
    _require(len(stats["train_count"]) >= 4 and len(stats["test_count"]) >= 3, "count diversity")
    _require(len(stats["orientation"]) == 4 and len(stats["shape"]) >= 80, "orientation or shape diversity")
    _require(len(stats["commands"]) >= 20 and len(stats["layout"]) >= 200, "semantic or layout diversity")
    _require(stats["length"] == {2, 3, 4, 5}, "bar length diversity")
    _require({1, 2, 7, 8} <= stats["collision"], "collision count diversity")
    _require(all(value > 0 for value in totals.values()), "review variation missing: " + str(totals))
    a = generate_puzzle(314159, 7, 3)
    b = generate_puzzle(314159, 7, 3)
    _require(_equal_puzzles(a, b), "nondeterministic puzzle")
    for seed, train, test in ((500, 2, 1), (501, 10, 1), (502, 10, 4), (600, None, 1), (601, None, 4), (602, 2, None), (603, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        _require((train is None or len(puzzle["train"]) == train) and (test is None or len(puzzle["test"]) == test), "count request ignored")
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(700, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
