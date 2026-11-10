import hashlib

import numpy as np


_ROLE_NAMES = (
    "background",
    "source",
    "obstacle",
    "path",
    "incoming",
    "outgoing",
    "contact",
)

_SOURCE_PAIR_DIGESTS = {
    "042e9bfbb75dffd9ad3176628df9b194ba970e8ad76730cdaed73527149612b1",
    "e5ef031cc3f23502a68f32a8fc49f0362e0fc8432c43e00a8ce06cc6adef4e85",
    "ac577545790cd5fbf01ba8d7c470a45a046e3de9296def3fe89290b8b2201d9d",
    "0267b1e6cbf0e204dd1c6e4dedc3bd1c7150d66d41d5551b954362247d0f1c79",
    "8b89a29a8fa37f3b390e7a02a6c347f4c9bd60c9a3a07ba03b1a07ae2cbd6a65",
}


class _SceneError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _scene_require(condition, message):
    if not condition:
        raise _SceneError(message)


def _grid_key(grid):
    return (tuple(grid.shape), grid.dtype.str, grid.tobytes())


def _pair_key(pair):
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _source_digest(pair):
    a = np.asarray(pair["input"], dtype=np.uint8)
    b = np.asarray(pair["output"], dtype=np.uint8)
    return hashlib.sha256(a.tobytes() + b.tobytes()).hexdigest()


def _validate_supplied_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed_value, num_train, num_test):
    train_count = _validate_supplied_count(
        "num_train", num_train, 2, 10
    )
    test_count = _validate_supplied_count(
        "num_test", num_test, 1, 4
    )

    if train_count is None and test_count is None:
        # This schedule guarantees broad count diversity for consecutive seeds.
        test_count = 1 + seed_value % 4
        minimum_train = max(2, test_count + 1)
        span = 11 - minimum_train
        train_count = (
            minimum_train
            + (seed_value + seed_value // 4) % span
        )
    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal test count for num_train")
        test_count = int(rng.integers(1, maximum_test + 1))

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _sample_puzzle_parameters(rng):
    palette = [int(x) for x in rng.permutation(10)[:7]]
    params = dict(zip(_ROLE_NAMES, palette))
    params["side"] = int(rng.choice((-1, 1)))
    params["forward"] = -1
    return params


def _write_claimed(output, owners, row, col, value, route_index):
    key = (row, col)
    if key in owners and owners[key] != route_index:
        raise _SceneError("two routes require the same output cell")
    owners[key] = route_index
    output[row, col] = value


def _derive_output(input_grid, roles):
    background = int(roles["background"])
    source = int(roles["source"])
    obstacle = int(roles["obstacle"])
    path = int(roles["path"])
    incoming = int(roles["incoming"])
    outgoing = int(roles["outgoing"])
    contact = int(roles["contact"])
    side = int(roles["side"])

    _scene_require(side in (-1, 1), "invalid lateral direction")
    _scene_require(input_grid.ndim == 2, "input is not two-dimensional")

    height, width = input_grid.shape
    allowed_input = {background, source, obstacle}
    actual_input = set(int(x) for x in np.unique(input_grid))
    _scene_require(
        actual_input == allowed_input,
        "input must contain exactly background, source, and obstacle colors",
    )

    source_positions = np.argwhere(input_grid == source)
    _scene_require(len(source_positions) > 0, "input has no source")
    _scene_require(
        np.all(source_positions[:, 0] == height - 1),
        "source markers must occur only on the bottom boundary",
    )

    source_columns = [
        int(x) for x in np.flatnonzero(input_grid[-1] == source)
    ]
    output = input_grid.copy()
    owners = {}
    route_info = []

    for route_index, source_col in enumerate(source_columns):
        row = height - 1
        col = source_col
        info = {
            "source_col": source_col,
            "successful_bypasses": 0,
            "bypass_rows": [],
            "shifts": [],
            "trapped": False,
            "reached_boundary": False,
            "trap_obstacle_row": None,
            "terminal": None,
        }

        while row > 0:
            next_row = row - 1
            next_value = int(input_grid[next_row, col])

            if next_value == background:
                row = next_row
                value = source if row == 0 else path
                _write_claimed(
                    output, owners, row, col, value, route_index
                )
                if row == 0:
                    info["reached_boundary"] = True
                    info["terminal"] = (row, col)
                    break
                continue

            _scene_require(
                next_value == obstacle,
                "a route encountered an unsupported input color",
            )

            far_col = col
            while (
                0 <= far_col + side < width
                and int(input_grid[next_row, far_col + side]) == obstacle
            ):
                far_col += side

            new_col = far_col + side
            corridor = []
            cursor = col + side
            while 0 <= cursor < width:
                corridor.append(cursor)
                if cursor == new_col:
                    break
                cursor += side

            corridor_exists = (
                0 <= new_col < width
                and len(corridor) > 0
                and corridor[-1] == new_col
            )
            corridor_clear = corridor_exists and all(
                int(input_grid[row, corridor_col]) == background
                for corridor_col in corridor
            )

            if not corridor_clear:
                _write_claimed(
                    output, owners, row, col, source, route_index
                )
                info["trapped"] = True
                info["trap_obstacle_row"] = next_row
                info["terminal"] = (row, col)
                break

            _scene_require(
                row < height - 1,
                "a successful bypass may not overwrite its source",
            )

            _write_claimed(
                output, owners, next_row, col, contact, route_index
            )
            _write_claimed(
                output, owners, row, col, incoming, route_index
            )
            for corridor_col in corridor[:-1]:
                _write_claimed(
                    output,
                    owners,
                    row,
                    corridor_col,
                    path,
                    route_index,
                )
            _write_claimed(
                output,
                owners,
                row,
                corridor[-1],
                outgoing,
                route_index,
            )

            info["successful_bypasses"] += 1
            info["bypass_rows"].append(next_row)
            info["shifts"].append(abs(new_col - col))
            col = new_col

        route_info.append(info)

    return output, route_info


def _sample_shape(rng, source_count):
    height = int(rng.integers(9, 31))
    width = int(rng.integers({1: 8, 2: 12, 3: 18}[source_count], 31))
    return height, width


def _lanes(rng, width, count):
    extras = rng.multinomial(width - 5 * count, np.full(count, 1 / count))
    bounds = []
    left = 0
    for extra in extras:
        right = left + 4 + int(extra)
        bounds.append((left, right))
        left = right + 1
    return bounds


def _event_rows(rng, height, count):
    if count == 0:
        return []
    available = height - 2  # rows zero through height minus three
    base = np.sort(rng.choice(available - count + 1, size=count, replace=False))
    return [int(x) for x in (base + np.arange(count))[::-1]]


def _choose_plan(rng, lane_width, height, kind):
    limit = min(6, (height - 3) // 2, lane_width - 2)
    if kind == "simple_trap":
        return "trap", 0
    if kind == "teach_reach":
        return "reach", int(rng.integers(1, min(limit, 3) + 1))
    if kind == "teach_trap":
        return "trap", int(rng.integers(0, min(limit, 3) + 1))
    status = "trap" if rng.random() < (0.43 if kind == "challenge" else 0.4) else "reach"
    if kind == "challenge":
        if rng.random() < 0.18:
            return status, 0
        return status, int(rng.integers(max(1, limit // 2), limit + 1))
    return status, int(rng.integers(0, limit + 1))


def _place_route(grid, rng, bg, source, obstacle, bounds, plan, route_index, route_count, force_edge=False):
    height, width = grid.shape
    left, right = bounds
    status, successes = plan
    edge_trap = status == "trap" and route_index == route_count - 1 and (force_edge or rng.random() < 0.25)
    reserve = successes + int(status == "trap" and not edge_trap)
    if right - left < reserve:
        raise _SceneError("lane too narrow")
    start = int(rng.integers(left, right - reserve + 1))
    grid[-1, start] = source
    events = _event_rows(rng, height, successes + int(status == "trap"))
    col = start
    for index in range(successes):
        row = events[index]
        later = successes - index - 1 + int(status == "trap" and not edge_trap)
        maximum = min(7, right - col - later)
        if maximum < 1:
            raise _SceneError("no room for bypass")
        shift = int(rng.integers(1, maximum + 1))
        extension = int(rng.integers(0, min(6, col - left) + 1))
        if not np.all(grid[row, col - extension:col + shift] == bg):
            raise _SceneError("overlapping obstacle runs")
        grid[row, col - extension:col + shift] = obstacle
        col += shift
    if status == "trap":
        row = events[-1]
        if edge_trap:
            if not np.all(grid[row, col:width] == bg):
                raise _SceneError("edge run overlaps")
            grid[row, col:width] = obstacle
        else:
            shift = int(rng.integers(1, min(6, right - col) + 1))
            extension = int(rng.integers(0, min(5, col - left) + 1))
            if not np.all(grid[row, col - extension:col + shift] == bg):
                raise _SceneError("trap run overlaps")
            grid[row, col - extension:col + shift] = obstacle
            blocker_col = col + int(rng.integers(1, shift + 1))
            if grid[row + 1, blocker_col] != bg:
                raise _SceneError("blocker overlaps")
            grid[row + 1, blocker_col] = obstacle
    return start


def _add_distractors(grid, rng, bg, obstacle, protected, density):
    height, width = grid.shape
    goal = max(1, int(round(density * height * width)))
    attempts = 0
    while np.count_nonzero(grid == obstacle) < goal and attempts < height * width * 12:
        attempts += 1
        row = int(rng.integers(0, height))
        col = int(rng.integers(0, width))
        length = int(rng.integers(1, 6))
        if rng.random() < 0.58:
            cells = [(row, c) for c in range(col, min(width, col + length))]
        else:
            cells = [(r, col) for r in range(row, min(height, row + length))]
        if all(grid[r, c] == bg and not protected[r, c] for r, c in cells):
            for r, c in cells:
                grid[r, c] = obstacle
    if np.count_nonzero(grid == obstacle) < goal:
        choices = np.argwhere((grid == bg) & ~protected)
        for i in rng.permutation(len(choices)):
            row, col = choices[int(i)]
            grid[row, col] = obstacle
            if np.count_nonzero(grid == obstacle) >= goal:
                break


def _generate_pair_once(rng, regime, params, kind, source_count, density):
    if regime == "source_train":
        _require(kind in ("teach_reach", "teach_trap", "free", "simple_trap"),
                 "unsupported training regime")
    elif regime == "target_test":
        _require(kind in ("challenge", "simple_trap"),
                 "unsupported test regime")
    else:
        raise ValueError("unknown pair regime")
    height, width = _sample_shape(rng, source_count)
    if kind == "simple_trap":
        height = int(rng.integers(9, 17))
        width = int(rng.integers(8, 14))
    bg = int(params["background"])
    source = int(params["source"])
    obstacle = int(params["obstacle"])
    grid = np.full((height, width), bg, dtype=np.int8)
    bounds = _lanes(rng, width, source_count)
    plans = [_choose_plan(rng, right - left + 1, height, kind if i == 0 else "free")
             for i, (left, right) in enumerate(bounds)]
    for i, (lane, plan) in enumerate(zip(bounds, plans)):
        _place_route(grid, rng, bg, source, obstacle, lane, plan, i, source_count,
                     force_edge=kind == "simple_trap")
    right_params = dict(params, side=1)
    baseline, routes = _derive_output(grid, right_params)
    if any((r["trapped"], r["successful_bypasses"]) != (plan[0] == "trap", plan[1])
           for r, plan in zip(routes, plans)):
        raise _SceneError("engineered route changed")
    protected = baseline != grid
    _add_distractors(grid, rng, bg, obstacle, protected, density)
    if not np.any(grid[:-1] == obstacle):
        choices = np.argwhere((grid[:-1] == bg) & (~protected[:-1]))
        if len(choices) == 0:
            raise _SceneError("no interior obstacle location")
        row, col = choices[int(rng.integers(0, len(choices)))]
        grid[int(row), int(col)] = obstacle
    output, final_routes = _derive_output(grid, right_params)
    if any((r["trapped"], r["successful_bypasses"]) != (plan[0] == "trap", plan[1])
           for r, plan in zip(final_routes, plans)):
        raise _SceneError("distractors altered route outcomes")
    if (not np.array_equal(output[protected], baseline[protected])
            or not np.array_equal(output != grid, protected)):
        raise _SceneError("distractors changed route")
    if np.array_equal(grid, output):
        raise _SceneError("unchanged scene")
    if params["side"] == -1:
        grid = grid[:, ::-1].copy()
        output = output[:, ::-1].copy()
    return {"input": grid, "output": output}


def _generate_pair(rng, regime, params, kind, source_count, density):
    failures = {}
    for _ in range(90):
        try:
            return _generate_pair_once(rng, regime, params, kind, source_count, density)
        except _SceneError as exc:
            failures[str(exc)] = failures.get(str(exc), 0) + 1
    raise RuntimeError(f"could not construct a route scene: {failures}")


def _permutations(values):
    if not values:
        yield ()
        return
    for index, value in enumerate(values):
        remainder = values[:index] + values[index + 1 :]
        for tail in _permutations(remainder):
            yield (value,) + tail


def _local_roles(input_grid, convention):
    colors = set(int(x) for x in np.unique(input_grid))
    _require(len(colors) == 3, "input must have three semantic colors")
    counts = {color: int(np.count_nonzero(input_grid == color)) for color in colors}
    background = max(colors, key=lambda color: counts[color])
    _require(sum(count == counts[background] for count in counts.values()) == 1,
             "ambiguous background")
    source = int(convention["source"])
    _require(source in colors and source != background, "source color changed")
    positions = np.argwhere(input_grid == source)
    _require(len(positions) > 0 and np.all(positions[:, 0] == input_grid.shape[0] - 1),
             "source must occupy the launch boundary")
    obstacle = next(iter(colors - {background, source}))
    _require(np.any(input_grid[:-1] == obstacle), "obstacle role lacks interior evidence")
    return dict(convention, background=background, obstacle=obstacle)


def _infer_convention_from_training(training_pairs):
    _require(len(training_pairs) >= 2, "too few training pairs")
    source_colors = set()
    input_colors = set()
    first_background = None
    first_obstacle = None
    for pair in training_pairs:
        a = pair["input"]
        colors = set(int(x) for x in np.unique(a))
        _require(len(colors) == 3, "training input roles")
        input_colors.update(colors)
        counts = {color: int(np.count_nonzero(a == color)) for color in colors}
        background = max(colors, key=lambda color: counts[color])
        _require(sum(count == counts[background] for count in counts.values()) == 1,
                 "ambiguous background")
        candidates = [color for color in colors - {background}
                      if np.all(np.argwhere(a == color)[:, 0] == a.shape[0] - 1)]
        _require(len(candidates) == 1, "ambiguous boundary source")
        source_colors.add(candidates[0])
        obstacle = next(iter(colors - {background, candidates[0]}))
        _require(np.any(a[:-1] == obstacle), "obstacle role lacks interior evidence")
        if first_background is None:
            first_background, first_obstacle = background, obstacle
    _require(len(source_colors) == 1, "source color varies across training pairs")
    source = next(iter(source_colors))
    output_colors = set().union(*(set(int(x) for x in np.unique(p["output"]))
                                  for p in training_pairs))
    new_colors = sorted(output_colors - input_colors)
    _require(len(new_colors) == 4, "training must reveal four marker roles")
    matches = []
    for side in (-1, 1):
        for permutation in _permutations(new_colors):
            candidate = {"background": first_background, "source": source,
                         "obstacle": first_obstacle, "path": permutation[0],
                         "incoming": permutation[1], "outgoing": permutation[2],
                         "contact": permutation[3], "side": side, "forward": -1}
            valid = True
            for pair in training_pairs:
                local = _local_roles(pair["input"], candidate)
                try:
                    predicted, _ = _derive_output(pair["input"], local)
                except _SceneError:
                    valid = False
                    break
                if not np.array_equal(predicted, pair["output"]):
                    valid = False
                    break
            if valid:
                matches.append(candidate)
    _require(len(matches) == 1, "training pairs do not identify one convention")
    return matches[0]


def _has_horizontal_barrier(input_grid, output_grid, roles):
    obstacle = int(roles["obstacle"])
    contact = int(roles["contact"])
    height, width = input_grid.shape

    contacts = np.argwhere(
        (input_grid == obstacle)
        & (output_grid == contact)
    )
    for row, col in contacts:
        row = int(row)
        col = int(col)
        if col > 0 and int(input_grid[row, col - 1]) == obstacle:
            return True
        if (
            col + 1 < width
            and int(input_grid[row, col + 1]) == obstacle
        ):
            return True
    return False


def _check_regimes(puzzle, roles, metadata):
    training_routes = [route for routes in metadata["train"] for route in routes]
    _require(any(route["successful_bypasses"] for route in training_routes),
             "training lacks a successful detour")
    _require(any(route["trapped"] for route in training_routes),
             "training lacks a blocked detour")
    _require(any(route["reached_boundary"] for route in training_routes),
             "training lacks a boundary endpoint")
    _require(any(route["terminal"] is not None for route in training_routes),
             "training does not reveal the terminal color")
    for split in ("train", "test"):
        for routes in metadata[split]:
            _require(1 <= len(routes) <= 3, "source count outside supported range")


def _check_independent_routes(input_grid, output_grid, roles):
    source = int(roles["source"])
    background = int(roles["background"])
    starts = np.flatnonzero(input_grid[-1] == source)
    combined = np.zeros(input_grid.shape, dtype=bool)
    for start in starts:
        single = input_grid.copy()
        for other in starts:
            if other != start:
                single[-1, other] = background
        isolated_output, _ = _derive_output(single, roles)
        changed = isolated_output != single
        _require(not np.any(changed & combined), "source routes overlap")
        _require(np.array_equal(isolated_output[changed], output_grid[changed]),
                 "source routes depend on drawing order")
        combined |= changed
    _require(np.array_equal(combined, output_grid != input_grid),
             "combined paths differ from independent routes")


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict), "puzzle must be a dictionary")
    _require(
        set(puzzle.keys()) == {"train", "test"},
        "puzzle must contain exactly train and test",
    )
    _require(
        isinstance(puzzle["train"], list)
        and isinstance(puzzle["test"], list),
        "train and test must be lists",
    )

    train_count = len(puzzle["train"])
    test_count = len(puzzle["test"])
    _require(2 <= train_count <= 10, "illegal training count")
    _require(1 <= test_count <= 4, "illegal test count")
    _require(
        train_count > test_count,
        "training count must exceed test count",
    )

    input_keys = set()
    pair_keys = set()

    for split_name in ("train", "test"):
        for pair in puzzle[split_name]:
            _require(
                isinstance(pair, dict),
                "each pair must be a dictionary",
            )
            _require(
                set(pair.keys()) == {"input", "output"},
                "pair must contain exactly input and output",
            )

            for grid_name in ("input", "output"):
                grid = pair[grid_name]
                _require(
                    isinstance(grid, np.ndarray),
                    "every grid must be a NumPy array",
                )
                _require(grid.ndim == 2, "grid must be two-dimensional")
                height, width = grid.shape
                _require(
                    1 <= height <= 30 and 1 <= width <= 30,
                    "grid dimensions are outside ARC limits",
                )
                _require(
                    np.issubdtype(grid.dtype, np.integer),
                    "grid dtype must be integral",
                )
                _require(
                    int(grid.min()) >= 0 and int(grid.max()) <= 9,
                    "grid colors must be in the range 0 through 9",
                )

            _require(
                pair["input"].shape == pair["output"].shape,
                "input and output dimensions must be preserved",
            )
            _require(
                _source_digest(pair) not in _SOURCE_PAIR_DIGESTS,
                "generated pair copies a source pair",
            )
            _require(
                not np.array_equal(pair["input"], pair["output"]),
                "generated pair is degenerate",
            )

            input_key = _grid_key(pair["input"])
            pair_key = _pair_key(pair)
            _require(
                input_key not in input_keys,
                "duplicate input grid within puzzle",
            )
            _require(
                pair_key not in pair_keys,
                "duplicate input/output pair within puzzle",
            )
            input_keys.add(input_key)
            pair_keys.add(pair_key)

    roles = _infer_convention_from_training(puzzle["train"])
    all_role_colors = {
        int(roles[name]) for name in _ROLE_NAMES
    }
    _require(
        len(all_role_colors) == 7,
        "semantic color roles must be distinct",
    )

    metadata = {"train": [], "test": []}

    for split_name in ("train", "test"):
        for pair in puzzle[split_name]:
            input_grid = pair["input"]
            output_grid = pair["output"]
            local_roles = _local_roles(input_grid, roles)

            _require(
                set(int(x) for x in np.unique(input_grid))
                == {
                    int(local_roles["background"]),
                    int(local_roles["source"]),
                    int(local_roles["obstacle"]),
                },
                "input palette is inconsistent with puzzle roles",
            )
            _require(
                set(int(x) for x in np.unique(output_grid))
                .issubset({int(local_roles[name]) for name in _ROLE_NAMES}),
                "output contains an inconsistent color",
            )

            source_mask = input_grid == int(roles["source"])
            _require(
                np.all(output_grid[source_mask] == int(roles["source"])),
                "original source markers were not preserved",
            )

            try:
                expected_output, route_info = _derive_output(
                    input_grid, local_roles
                )
            except _SceneError as exc:
                raise AssertionError(str(exc))

            _require(
                np.array_equal(expected_output, output_grid),
                "output does not follow the path-and-bypass rule",
            )
            _check_independent_routes(input_grid, output_grid, local_roles)
            metadata[split_name].append(route_info)

    _check_regimes(puzzle, roles, metadata)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_counts(rng, seed_value, num_train, num_test)
    roles = _sample_puzzle_parameters(rng)
    fixed_colors = {roles[name] for name in ("source", "path", "incoming", "outgoing", "contact")}
    structural_colors = [color for color in range(10) if color not in fixed_colors]
    teaching = ["teach_reach", "teach_trap"] + ["free"] * (train_count - 2)
    rng.shuffle(teaching)
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    for split, kinds in (("train", teaching), ("test", ["challenge"] * test_count)):
        for kind in kinds:
            for _ in range(40):
                child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair_kind = "simple_trap" if kind in ("free", "challenge") and child.random() < 0.09 else kind
                source_count = 1 if pair_kind == "simple_trap" else int(child.integers(1, 4))
                density = 0.003 if pair_kind == "simple_trap" else float(child.uniform(0.012, 0.16))
                pair_colors = child.choice(structural_colors, size=2, replace=False)
                pair_roles = dict(roles, background=int(pair_colors[0]),
                                  obstacle=int(pair_colors[1]))
                pair = _generate_pair(child, "source_train" if split == "train" else "target_test",
                                      pair_roles, pair_kind, source_count, density)
                key = _grid_key(pair["input"])
                if key not in used_inputs:
                    used_inputs.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    if set(first.keys()) != {"train", "test"}:
        return False
    if set(second.keys()) != {"train", "test"}:
        return False

    for split_name in ("train", "test"):
        if len(first[split_name]) != len(second[split_name]):
            return False
        for first_pair, second_pair in zip(
            first[split_name], second[split_name]
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


def self_test() -> None:
    train_counts, test_counts, palettes, pair_palettes, shapes, layouts = set(), set(), set(), set(), set(), set()
    source_counts, sides, heights, widths = set(), set(), set(), set()
    observed = {name: 0 for name in (
        "top_contact", "long_shift", "many_bypasses", "zero_bypass",
        "immediate_trap", "clear_reach", "high_trap", "edge_trap",
        "source_row_obstacle", "compact", "train_sparse", "train_dense",
        "test_sparse", "test_dense", "train_three", "test_one", "test_two",
        "within_source", "within_density", "within_palette", "mixed_dense_trap_long",
        "close_sources", "unequal_sources", "left_extension", "connected_contact",
        "close_encounters", "long_straight", "compact_simple_trap",
    )}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        roles = _infer_convention_from_training(puzzle["train"])
        palettes.add(tuple(roles[name] for name in _ROLE_NAMES))
        sides.add(roles["side"])
        train_sources, train_densities, puzzle_palettes = [], [], set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                local = _local_roles(a, roles)
                expected, routes = _derive_output(a, local)
                _require(np.array_equal(expected, b), "self-test reconstruction failed")
                height, width = a.shape
                density = float(np.count_nonzero(a == local["obstacle"])) / (height * width)
                shapes.add(a.shape)
                heights.add(height)
                widths.add(width)
                source_counts.add(len(routes))
                layouts.add((a.shape, (a == local["obstacle"]).tobytes()))
                palette = tuple(sorted(int(x) for x in np.unique(a)))
                pair_palettes.add(palette)
                puzzle_palettes.add(palette)
                if split == "train":
                    train_sources.append(len(routes))
                    train_densities.append(density)
                observed[split + "_sparse"] += density < 0.05
                observed[split + "_dense"] += density >= 0.08
                observed["train_three"] += split == "train" and len(routes) == 3
                observed["test_one"] += split == "test" and len(routes) == 1
                observed["test_two"] += split == "test" and len(routes) == 2
                observed["compact"] += height <= 17 or width <= 12
                observed["source_row_obstacle"] += np.any(a[-1] == local["obstacle"])
                observed["top_contact"] += any(any(row < height // 3 for row in r["bypass_rows"]) for r in routes)
                observed["long_shift"] += any(any(shift >= 4 for shift in r["shifts"]) for r in routes)
                observed["many_bypasses"] += any(r["successful_bypasses"] >= 3 for r in routes)
                observed["zero_bypass"] += any(r["successful_bypasses"] == 0 for r in routes)
                observed["immediate_trap"] += any(r["trapped"] and r["successful_bypasses"] == 0 for r in routes)
                observed["clear_reach"] += any(r["reached_boundary"] and r["successful_bypasses"] == 0 for r in routes)
                observed["high_trap"] += any(r["trapped"] and r["terminal"][0] < height // 3 for r in routes)
                edge_col = 0 if roles["side"] == -1 else width - 1
                observed["edge_trap"] += any(r["trapped"] and a[r["trap_obstacle_row"], edge_col] == local["obstacle"] for r in routes)
                observed["mixed_dense_trap_long"] += density >= 0.08 and any(r["trapped"] for r in routes) and any(shift >= 4 for r in routes for shift in r["shifts"])
                starts = sorted(r["source_col"] for r in routes)
                gaps = [right - left for left, right in zip(starts, starts[1:])]
                observed["close_sources"] += bool(gaps) and min(gaps) <= 4
                observed["unequal_sources"] += len(gaps) > 1 and len(set(gaps)) > 1
                contacts = sorted(row for r in routes for row in r["bypass_rows"])
                observed["close_encounters"] += any(b - a <= 3 for a, b in zip(contacts, contacts[1:]))
                observed["compact_simple_trap"] += (height <= 17 or width <= 12) and int(np.count_nonzero(a == local["obstacle"])) <= 8 and any(r["trapped"] for r in routes)
                for route in routes:
                    col = route["source_col"]
                    previous_row = height - 1
                    for row, shift in zip(route["bypass_rows"], route["shifts"]):
                        observed["long_straight"] += previous_row - row >= 7
                        cursor = col - roles["side"]
                        extension = 0
                        while 0 <= cursor < width and a[row, cursor] == local["obstacle"]:
                            extension += 1
                            cursor -= roles["side"]
                        observed["left_extension"] += extension >= 2
                        observed["connected_contact"] += any(
                            0 <= row + delta < height and a[row + delta, col] == local["obstacle"]
                            for delta in (-1, 1))
                        previous_row = row
                        col += roles["side"] * shift
        observed["within_source"] += len(set(train_sources)) > 1
        observed["within_density"] += max(train_densities) - min(train_densities) >= 0.04
        observed["within_palette"] += len(puzzle_palettes) > 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic count diversity")
    _require(len(palettes) >= 35 and sides == {-1, 1}, "palette or side diversity")
    _require(len(pair_palettes) >= 50, "pair palette diversity")
    _require(len(shapes) >= 30 and len(layouts) >= 100, "structural diversity")
    _require(len(heights) >= 15 and len(widths) >= 15, "dimension diversity")
    _require(source_counts == {1, 2, 3}, "source-count diversity")
    minimums = {
        "top_contact": 15, "long_shift": 15, "many_bypasses": 15,
        "zero_bypass": 15, "immediate_trap": 10, "clear_reach": 10,
        "high_trap": 10, "edge_trap": 10, "source_row_obstacle": 10,
        "compact": 10, "train_sparse": 10, "train_dense": 20,
        "test_sparse": 3, "test_dense": 15, "train_three": 10,
        "test_one": 5, "test_two": 5, "within_source": 25,
        "within_density": 25, "within_palette": 35, "mixed_dense_trap_long": 5,
        "close_sources": 10, "unequal_sources": 20, "left_extension": 20,
        "connected_contact": 5, "close_encounters": 20,
        "long_straight": 20, "compact_simple_trap": 5,
    }
    for name, minimum in minimums.items():
        _require(observed[name] >= minimum, f"insufficient {name} diversity")
    first = generate_puzzle(314159, num_train=4, num_test=2)
    second = generate_puzzle(314159, num_train=4, num_test=2)
    _require(_puzzles_equal(first, second), "whole-puzzle generation is not deterministic")
    _require(not _puzzles_equal(first, generate_puzzle(314160, 4, 2)), "different seeds repeat")
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (2001, None, 1), (2002, None, 4),
                              (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, num_train=train, num_test=test)
        _require((train is None or len(puzzle["train"]) == train) and
                 (test is None or len(puzzle["test"]) == test), "requested counts ignored")
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(9999, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {kwargs}")
