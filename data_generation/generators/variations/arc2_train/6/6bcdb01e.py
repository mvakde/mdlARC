import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _in_bounds(position: tuple[int, int], height: int, width: int) -> bool:
    row, col = position
    return 0 <= row < height and 0 <= col < width


def _on_boundary(position: tuple[int, int], height: int, width: int) -> bool:
    row, col = position
    return row == 0 or row == height - 1 or col == 0 or col == width - 1


def _array_key(array: np.ndarray) -> tuple:
    return array.shape, array.tobytes()


def _pair_key(pair: dict) -> tuple:
    return _array_key(pair["input"]), _array_key(pair["output"])


def _grid_from_strings(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[int(value) for value in row] for row in rows], dtype=np.int8)


_SOURCE_PAIRS = (
    (
        _grid_from_strings(
            (
                "8778787",
                "7887778",
                "7777777",
                "7777778",
                "7777787",
                "3377877",
                "7778777",
            )
        ),
        _grid_from_strings(
            (
                "8778787",
                "7883338",
                "7773737",
                "3333338",
                "7773787",
                "3333877",
                "7778777",
            )
        ),
    ),
    (
        _grid_from_strings(
            (
                "78787",
                "87778",
                "77777",
                "73778",
                "73787",
            )
        ),
        _grid_from_strings(
            (
                "78787",
                "83338",
                "73737",
                "33338",
                "73787",
            )
        ),
    ),
    (
        _grid_from_strings(
            (
                "77877777",
                "78777733",
                "77778787",
                "78787778",
                "87787777",
                "77877777",
                "87777877",
                "78778777",
            )
        ),
        _grid_from_strings(
            (
                "77877777",
                "78333333",
                "77378787",
                "78383338",
                "83383737",
                "73873737",
                "83333837",
                "78778737",
            )
        ),
    ),
)


def _is_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    for source_input, source_output in _SOURCE_PAIRS:
        if (
            input_grid.shape == source_input.shape
            and output_grid.shape == source_output.shape
            and np.array_equal(input_grid, source_input)
            and np.array_equal(output_grid, source_output)
        ):
            return True
    return False


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_SIDE_DIRECTIONS = {"top": (1, 0), "right": (0, -1), "bottom": (-1, 0), "left": (0, 1)}
_DIRECTION_NAMES = {(-1, 0): "U", (0, 1): "R", (1, 0): "D", (0, -1): "L"}


def _matching_exists(positions: frozenset[tuple[int, int]]) -> bool:
    if not positions:
        return True
    first = min(positions)
    row, col = first
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        partner = (row + dr, col + dc)
        if partner in positions and _matching_exists(positions - {first, partner}):
            return True
    return False


def _seed_for_color(grid: np.ndarray, color: int) -> tuple | None:
    cells = [tuple(map(int, point)) for point in np.argwhere(grid == color)]
    if not 2 <= len(cells) <= 4:
        return None
    height, width = grid.shape
    boundary_cells = [p for p in cells if _on_boundary(p, height, width)]
    if len(boundary_cells) != 1:
        return None
    start = boundary_cells[0]
    sides = []
    if start[0] == 0:
        sides.append("top")
    if start[0] == height - 1:
        sides.append("bottom")
    if start[1] == 0:
        sides.append("left")
    if start[1] == width - 1:
        sides.append("right")
    if len(sides) != 1:
        return None
    side = sides[0]
    direction = _SIDE_DIRECTIONS[side]
    expected = tuple((start[0] + i * direction[0], start[1] + i * direction[1]) for i in range(len(cells)))
    if set(cells) != set(expected) or _on_boundary(expected[-1], height, width):
        return None
    return side, direction, expected


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    _require(isinstance(input_grid, np.ndarray) and isinstance(output_grid, np.ndarray), "Grids must be arrays")
    _require(input_grid.ndim == output_grid.ndim == 2, "Grids must be two-dimensional")
    _require(input_grid.shape == output_grid.shape, "Dimensions must be preserved")
    height, width = input_grid.shape
    _require(1 <= height <= 30 and 1 <= width <= 30, "Grid dimensions out of range")
    for grid in (input_grid, output_grid):
        _require(np.issubdtype(grid.dtype, np.integer), "Grid dtype must be integer")
        _require(bool(np.all((0 <= grid) & (grid <= 9))), "Grid colors out of range")
    colors, counts = np.unique(input_grid, return_counts=True)
    _require(len(colors) == 3, "Each input must use three colors")
    count = {int(c): int(n) for c, n in zip(colors, counts)}
    seed_candidates = [(int(c), _seed_for_color(input_grid, int(c))) for c in colors]
    seed_candidates = [(c, info) for c, info in seed_candidates if info is not None]
    _require(len(seed_candidates) == 1, "Seed role is not uniquely visible")
    trail_color, (seed_side, direction, seed_cells) = seed_candidates[0]
    other = [int(c) for c in colors if int(c) != trail_color]
    field_color = max(other, key=lambda c: count[c])
    marker_color = next(c for c in other if c != field_color)
    _require(count[field_color] > count[marker_color], "Field must be dominant")
    _require(count[marker_color] >= 2 and count[marker_color] % 2 == 0, "Marker count invalid")
    markers = frozenset(tuple(map(int, p)) for p in np.argwhere(input_grid == marker_color))
    _require(_matching_exists(markers), "Markers cannot form diagonal pairs")
    touching_markers = sum(
        sum((row + dr, col + dc) in markers for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1))) > 1
        for row, col in markers
    )

    current = seed_cells[-1]
    route = list(seed_cells)
    seen_states = set()
    senses = []
    sequence = [_DIRECTION_NAMES[direction]]
    leg_lengths = []
    leg_length = 0
    while True:
        state = (*current, *direction)
        _require(state not in seen_states, "Route enters a state loop")
        seen_states.add(state)
        _require(len(seen_states) <= 4 * height * width, "Route exceeds state space")
        ahead = (current[0] + direction[0], current[1] + direction[1])
        if not _in_bounds(ahead, height, width):
            if ahead[0] < 0:
                exit_side = "top"
            elif ahead[0] >= height:
                exit_side = "bottom"
            elif ahead[1] < 0:
                exit_side = "left"
            else:
                exit_side = "right"
            break
        if ahead in markers:
            left = (-direction[1], direction[0])
            right = (direction[1], -direction[0])
            left_cell = (current[0] + left[0], current[1] + left[1])
            right_cell = (current[0] + right[0], current[1] + right[1])
            _require(_in_bounds(left_cell, height, width) and _in_bounds(right_cell, height, width), "Turn at an ambiguous boundary")
            blocked_left = left_cell in markers
            blocked_right = right_cell in markers
            _require(blocked_left != blocked_right, "Forward marker lacks one unique turn")
            direction = right if blocked_left else left
            senses.append("R" if blocked_left else "L")
            sequence.append(_DIRECTION_NAMES[direction])
            leg_lengths.append(leg_length)
            leg_length = 0
            continue
        _require(int(input_grid[ahead]) in (field_color, trail_color), "Route enters an obstacle")
        current = ahead
        route.append(current)
        leg_length += 1
    leg_lengths.append(leg_length)
    expected = input_grid.copy()
    for cell in route:
        expected[cell] = trail_color
    _require(np.array_equal(output_grid, expected), "Output differs from deterministic trace")
    changed = int(np.count_nonzero(input_grid != output_grid))
    _require(changed > 0, "Trail must extend")
    return {
        "roles": (field_color, marker_color, trail_color),
        "seed_side": seed_side, "seed_length": len(seed_cells), "exit_side": exit_side,
        "turn_count": len(senses), "turn_senses": tuple(senses),
        "direction_sequence": "".join(sequence),
        "leg_lengths": tuple(leg_lengths), "touching_markers": touching_markers,
        "marker_count": len(markers), "inert_pairs": len(markers) // 2 - len(senses),
        "route": tuple(route), "route_cells": frozenset(route),
        "self_crossing": len(route) != len(set(route)), "changed_count": changed,
    }


def _boundary_start(rng: np.random.Generator, height: int, width: int, side: str) -> tuple[int, int]:
    if side == "top":
        return 0, int(rng.integers(1, width - 1))
    if side == "bottom":
        return height - 1, int(rng.integers(1, width - 1))
    if side == "left":
        return int(rng.integers(1, height - 1)), 0
    return int(rng.integers(1, height - 1)), width - 1


def _turn(direction: tuple[int, int], sense: str) -> tuple[int, int]:
    return (-direction[1], direction[0]) if sense == "L" else (direction[1], -direction[0])


def _plan_once(
    rng: np.random.Generator, height: int, width: int, side: str,
    prefix: int, senses: tuple[str, ...], colors: tuple[int, int, int],
    node_budget: int = 2500,
) -> tuple[np.ndarray, np.ndarray] | None:
    start = _boundary_start(rng, height, width, side)
    direction = _SIDE_DIRECTIONS[side]
    seed = tuple((start[0] + i * direction[0], start[1] + i * direction[1]) for i in range(prefix))
    if any(not _in_bounds(p, height, width) for p in seed) or _on_boundary(seed[-1], height, width):
        return None
    field, marker, trail = colors
    nodes = 0

    def search(index, current, heading, route, obstacles):
        nonlocal nodes
        nodes += 1
        if nodes > node_budget:
            return None
        if index == len(senses):
            final = list(route)
            position = current
            while True:
                position = (position[0] + heading[0], position[1] + heading[1])
                if not _in_bounds(position, height, width):
                    break
                if position in obstacles:
                    return None
                final.append(position)
                if len(final) > 4 * height * width:
                    return None
            grid = np.full((height, width), field, dtype=np.int8)
            for p in obstacles:
                grid[p] = marker
            for p in seed:
                grid[p] = trail
            output = grid.copy()
            for p in final:
                output[p] = trail
            try:
                a = _analyze_pair(grid, output)
            except AssertionError:
                return None
            if a["turn_senses"] != senses or a["seed_length"] != prefix:
                return None
            return grid, output
        new_heading = _turn(heading, senses[index])
        options = []
        for distance in range(1, max(height, width)):
            target = (current[0] + distance * heading[0], current[1] + distance * heading[1])
            forward = (target[0] + heading[0], target[1] + heading[1])
            side_marker = (target[0] - new_heading[0], target[1] - new_heading[1])
            exit_cell = (target[0] + new_heading[0], target[1] + new_heading[1])
            if not all(_in_bounds(p, height, width) for p in (target, forward, side_marker, exit_cell)):
                continue
            segment = [(current[0] + step * heading[0], current[1] + step * heading[1]) for step in range(1, distance + 1)]
            pair = {forward, side_marker}
            if (pair & (set(route) | set(segment) | obstacles)) or any(p in obstacles for p in segment) or exit_cell in obstacles or exit_cell in pair:
                continue
            options.append((target, segment, pair))
        if not options:
            return None
        # Prefer short legs often, while retaining long spans and one-cell elbows.
        order = rng.permutation(len(options)).tolist()
        if rng.random() < 0.4:
            order.sort(key=lambda i: len(options[i][1]))
        for option_index in order:
            target, segment, pair = options[option_index]
            result = search(index + 1, target, new_heading, route + segment, obstacles | pair)
            if result is not None:
                return result
        return None

    return search(0, seed[-1], direction, list(seed), set())


def _add_inert_pairs(rng, pair, colors, desired):
    grid, output = pair
    field, marker, _trail = colors
    base = _analyze_pair(grid, output)
    height, width = grid.shape
    candidates = []
    for row in range(height - 1):
        for col in range(width - 1):
            candidates.extend((((row, col), (row + 1, col + 1)), ((row + 1, col), (row, col + 1))))
    for _ in range(desired):
        added = False
        for i in rng.permutation(len(candidates))[:min(140, len(candidates))]:
            first, second = candidates[int(i)]
            if any(int(grid[p]) != field or int(output[p]) != field for p in (first, second)):
                continue
            trial_in, trial_out = grid.copy(), output.copy()
            for p in (first, second):
                trial_in[p] = marker
                trial_out[p] = marker
            try:
                result = _analyze_pair(trial_in, trial_out)
            except AssertionError:
                continue
            if result["route"] == base["route"] and result["turn_senses"] == base["turn_senses"]:
                grid, output, base = trial_in, trial_out, result
                added = True
                break
        if not added:
            break
    return grid, output


def _generate_pair(rng, colors, side, prefix, turns, both_senses=False, inert_target=0):
    for attempt in range(80):
        # Sizes are sampled independently of the train/test label. Small source-scale
        # grids and large or narrow canvases all occur in both regimes.
        scale = int(rng.choice(3, p=(0.40, 0.38, 0.22)))
        if scale == 0:
            height, width = (int(rng.integers(5, 10)), int(rng.integers(5, 10)))
        elif scale == 1:
            height, width = (int(rng.integers(8, 17)), int(rng.integers(8, 17)))
        else:
            height, width = (int(rng.integers(10, 31)), int(rng.integers(10, 31)))
        if rng.random() < 0.25:
            if rng.random() < 0.5:
                height = int(rng.integers(5, 9))
            else:
                width = int(rng.integers(5, 9))
        if min(height, width) <= prefix:
            continue
        if turns > (height + width) // 2:
            continue
        senses = ["L" if rng.random() < 0.5 else "R" for _ in range(turns)]
        if both_senses and turns >= 2:
            senses[0], senses[1] = "L", "R"
            rng.shuffle(senses)
        pair = _plan_once(rng, height, width, side, prefix, tuple(senses), colors)
        if pair is None:
            continue
        pair = _add_inert_pairs(rng, pair, colors, inert_target)
        return pair
    # Deterministic broad-canvas fallback for unusually constrained random plans.
    for height, width in ((14, 17), (20, 23), (27, 29)):
        for _ in range(40):
            senses = ["L" if rng.random() < 0.5 else "R" for _ in range(turns)]
            if both_senses and turns >= 2:
                senses[0], senses[1] = "L", "R"
                rng.shuffle(senses)
            pair = _plan_once(rng, height, width, side, prefix, tuple(senses), colors, 4000)
            if pair is not None:
                return _add_inert_pairs(rng, pair, colors, inert_target)
    raise ValueError("Unable to construct an unambiguous marker-guided route")


def _validate_optional_count(value: int | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if name == "num_train" and not 2 <= int(value) <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if name == "num_test" and not 1 <= int(value) <= 4:
        raise ValueError("num_test must be between 1 and 4")


def _resolve_counts(seed, num_train, num_test):
    _validate_optional_count(num_train, "num_train")
    _validate_optional_count(num_test, "num_test")
    train = None if num_train is None else int(num_train)
    test = None if num_test is None else int(num_test)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if not (2 <= train <= 10 and 1 <= test <= 4 and train > test):
        raise ValueError("Counts must satisfy 2 <= train <= 10, 1 <= test <= 4, and train > test")
    return train, test


def _colors(rng):
    return tuple(int(x) for x in rng.choice(10, size=3, replace=False))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    base_colors = _colors(rng)
    variable_palette = bool(rng.integers(0, 2))
    training_sides = list(rng.choice(tuple(_SIDE_DIRECTIONS), size=2, replace=False))
    training_prefixes = [2, int(rng.choice((2, 2, 2, 3, 4)))]
    pairs = {"train": [], "test": []}
    seen = set()
    train_roles = []
    train_turns = []
    for regime, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if regime == "train":
                side = training_sides[index] if index < 2 else str(rng.choice(tuple(_SIDE_DIRECTIONS)))
                prefix = training_prefixes[index] if index < 2 else int(rng.choice((2, 2, 2, 3, 4)))
                turns = int(rng.integers(2, 5)) if index == 0 else int(rng.integers(1, 7))
                if index == 1 and turns > 4 and rng.random() < 0.6:
                    turns = int(rng.integers(1, 3))
                colors = base_colors if not variable_palette or index == 0 else _colors(rng)
                if variable_palette and index == 1:
                    while colors == base_colors:
                        colors = _colors(rng)
            else:
                side = str(rng.choice(training_sides if rng.random() < 0.5 else tuple(_SIDE_DIRECTIONS)))
                prefix = int(rng.choice(tuple(set(training_prefixes + [a["seed_length"] for a in train_turns]))))
                turns = int(rng.integers(1, 8))
                if rng.random() < 0.25:
                    turns = max(1, max(a["turn_count"] for a in train_turns) - int(rng.integers(1, 3)))
                colors = base_colors if not variable_palette else _colors(rng)
            inert = int(rng.choice((0, 0, 1, 1, 2, 3)))
            for retry in range(16):
                child = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                try:
                    input_grid, output_grid = _generate_pair(child, colors, side, prefix, turns, both_senses=(regime == "train" and index == 0), inert_target=inert)
                except ValueError:
                    continue
                key = _array_key(input_grid)
                if key in seen or _is_source_pair(input_grid, output_grid):
                    continue
                pair = {"input": input_grid, "output": output_grid}
                analysis = _analyze_pair(input_grid, output_grid)
                if regime == "test" and any(
                    analysis["direction_sequence"] == _analyze_pair(p["input"], p["output"])["direction_sequence"]
                    for p in pairs["test"]
                ):
                    turns = int(rng.integers(1, 8))
                    continue
                pairs[regime].append(pair)
                seen.add(key)
                if regime == "train":
                    train_turns.append(analysis)
                    train_roles.append(analysis["roles"])
                    training_prefixes.append(prefix)
                break
            else:
                raise ValueError("Unable to generate distinct puzzle pair")
    validate_generated_puzzle(pairs)
    return pairs


def _normalized_shape(pair):
    roles = _analyze_pair(pair["input"], pair["output"])["roles"]
    normalized = np.zeros(pair["input"].shape, dtype=np.int8)
    normalized[pair["input"] == roles[1]] = 1
    normalized[pair["input"] == roles[2]] = 2
    return normalized.shape, normalized.tobytes()


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "Regimes must be lists")
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test, "Invalid pair counts")
    seen_input, seen_pair = set(), set()
    analyses = {"train": [], "test": []}
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            a = _analyze_pair(pair["input"], pair["output"])
            analyses[regime].append(a)
            in_key, pair_key = _array_key(pair["input"]), _pair_key(pair)
            _require(in_key not in seen_input and pair_key not in seen_pair, "Duplicate pair")
            seen_input.add(in_key)
            seen_pair.add(pair_key)
            _require(not _is_source_pair(pair["input"], pair["output"]), "Source pair copied")
            _require(a["inert_pairs"] >= 0, "More turns than marker pairs")
    training = analyses["train"]
    tests = analyses["test"]
    _require(len({a["direction_sequence"] for a in tests}) == len(tests), "Tests repeat a route structure")
    _require(len({a["seed_side"] for a in training}) >= 2, "Training must show two entrance sides")
    _require({"L", "R"} <= set(s for a in training for s in a["turn_senses"]), "Training must show both local turns")
    _require(2 in {a["seed_length"] for a in training}, "Training must show two-cell seed")
    trained_prefixes = {a["seed_length"] for a in training}
    _require(all(a["seed_length"] in trained_prefixes for a in tests), "Test prefix not demonstrated")
    roles = [a["roles"] for a in training]
    if len(set(roles)) == 1:
        _require(all(a["roles"] == roles[0] for a in tests), "Undemonstrated palette change")
    # Each local role is visible: the seed is a unique boundary prefix and the field dominates.
    for a in training + tests:
        _require(a["changed_count"] > 0, "Unchanged output")
        _require(a["turn_count"] >= 1, "Route has no marker-guided turn")


def _puzzles_equal(first, second):
    if any(len(first[r]) != len(second[r]) for r in ("train", "test")):
        return False
    return all(np.array_equal(p["input"], q["input"]) and np.array_equal(p["output"], q["output"])
               for r in ("train", "test") for p, q in zip(first[r], second[r]))


def self_test() -> None:
    for seed, train, test in ((12345, None, None), (54321, 6, 3), (7101, 2, 1), (7102, 10, 1), (7103, 10, 4),
                              (7201, None, 1), (7202, None, 4), (7203, 2, None), (7204, 10, None)):
        first = generate_puzzle(seed, train, test)
        second = generate_puzzle(seed, train, test)
        validate_generated_puzzle(first)
        _require(_puzzles_equal(first, second), "Generation is not deterministic")
        if train is not None:
            _require(len(first["train"]) == train, "Explicit train count ignored")
        if test is not None:
            _require(len(first["test"]) == test, "Explicit test count ignored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(7300, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    puzzles = [generate_puzzle(seed) for seed in range(10000, 10042)]
    _require(not _puzzles_equal(puzzles[0], puzzles[1]), "Different seeds duplicated puzzle")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "Train counts lack diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "Test counts lack diversity")
    pairs = [pair for puzzle in puzzles for pair in puzzle["train"] + puzzle["test"]]
    analyses = [_analyze_pair(p["input"], p["output"]) for p in pairs]
    by_regime = {
        regime: [_analyze_pair(pair["input"], pair["output"])
                 for puzzle in puzzles for pair in puzzle[regime]]
        for regime in ("train", "test")
    }
    _require(len({_normalized_shape(p) for p in pairs}) >= 120, "Layouts lack diversity")
    _require(len({p["input"].shape for p in pairs}) >= 30, "Grid sizes lack diversity")
    _require(len({a["direction_sequence"] for a in analyses}) >= 25, "Routes lack diversity")
    _require({"top", "bottom", "left", "right"} <= {a["seed_side"] for a in analyses}, "Entrance sides lack diversity")
    _require({"top", "bottom", "left", "right"} <= {a["exit_side"] for a in analyses}, "Exit sides lack diversity")
    _require({2, 3, 4} <= {a["seed_length"] for a in analyses}, "Seed prefixes lack diversity")
    _require(any(a["self_crossing"] for a in analyses) and any(not a["self_crossing"] for a in analyses), "Crossing lacks diversity")
    _require(any(a["inert_pairs"] == 0 for a in analyses) and any(a["inert_pairs"] >= 2 for a in analyses), "Inert marker counts lack diversity")
    for regime, group in by_regime.items():
        _require(any(a["self_crossing"] for a in group) and any(not a["self_crossing"] for a in group), f"{regime} crossing diversity absent")
        _require(any(a["inert_pairs"] >= 2 for a in group), f"{regime} multi-distractor layouts absent")
        _require(any(a["touching_markers"] > 0 for a in group), f"{regime} adjacent elbows absent")
        _require(any(1 in a["leg_lengths"][1:-1] for a in group), f"{regime} one-cell legs absent")
        _require(any(a["turn_count"] <= 2 for a in group), f"{regime} short routes absent")
        _require(any(a["turn_count"] >= 5 for a in group), f"{regime} long routes absent")
        _require(any(a["self_crossing"] and a["inert_pairs"] > 0 and a["seed_length"] > 2 for a in group), f"{regime} combined variations absent")
    _require(any(len({_analyze_pair(pair["input"], pair["output"])["roles"]
                      for regime in ("train", "test") for pair in puzzle[regime]}) > 1
                 for puzzle in puzzles), "Pair-level palette changes absent")
    _require(any(any(_analyze_pair(pair["input"], pair["output"])["seed_side"] not in
                         {_analyze_pair(train["input"], train["output"])["seed_side"] for train in puzzle["train"]}
                         for pair in puzzle["test"]) for puzzle in puzzles), "Novel test entrances absent")
