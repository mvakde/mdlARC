import hashlib

import numpy as np


_SOURCE_PAIR_DIGESTS = frozenset({
    "6c0f2f44fcab1de300bd5d5d5268b391cd467c57632c633be3531fd22daec0fa",
    "82af83da22dd262cb4e5cb9d6f43ae5f08646a27d60bec398dbf237e093ddee0",
    "21740bfcdf2708e3e2c61480f8225a3d89bc36844b971308281720f81ca1b363",
    "bd6428e56390a123e9fe3b7f48ac239e13a3e24344b4545b193921d4e4ea02c1",
})
_SIDES = frozenset({"left", "right", "above", "below"})


def _components(
    mask: np.ndarray,
    eight_connected: bool = False,
) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)

    if eight_connected:
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),             (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        ]
    else:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    result: list[list[tuple[int, int]]] = []

    for row, column in np.argwhere(mask):
        row, column = int(row), int(column)
        if seen[row, column]:
            continue

        seen[row, column] = True
        stack = [(row, column)]
        component: list[tuple[int, int]] = []

        while stack:
            current_row, current_column = stack.pop()
            component.append((current_row, current_column))

            for delta_row, delta_column in directions:
                next_row = current_row + delta_row
                next_column = current_column + delta_column

                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and mask[next_row, next_column]
                    and not seen[next_row, next_column]
                ):
                    seen[next_row, next_column] = True
                    stack.append((next_row, next_column))

        result.append(component)

    return result


def _extract_anchors(
    grid: np.ndarray,
    anchor_color: int,
) -> list[tuple[int, int]]:
    components = _components(grid == anchor_color)
    assert components

    anchors: list[tuple[int, int]] = []

    for component in components:
        assert len(component) == 4

        rows = sorted({row for row, _ in component})
        columns = sorted({column for _, column in component})

        assert len(rows) == 2 and rows[1] == rows[0] + 1
        assert len(columns) == 2 and columns[1] == columns[0] + 1
        assert set(component) == {
            (row, column)
            for row in rows
            for column in columns
        }

        anchors.append((rows[0], columns[0]))

    anchors.sort()
    return anchors


def _is_anchor_color(grid: np.ndarray, color: int) -> bool:
    try:
        anchors = _extract_anchors(grid, color)
    except AssertionError:
        return False

    return 4 * len(anchors) == int(np.count_nonzero(grid == color))


def _is_marker_color(grid: np.ndarray, color: int) -> bool:
    # Source markers may touch diagonally and still move independently.
    components = _components(grid == color)
    return bool(components) and all(
        len(component) == 1 for component in components
    )


def _infer_roles(grid: np.ndarray) -> tuple[int, int, int]:
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 3

    largest = np.flatnonzero(counts == counts.max())
    assert len(largest) == 1

    background = int(values[int(largest[0])])
    other_colors = [
        int(value)
        for value in values
        if int(value) != background
    ]

    anchor_options = [
        color
        for color in other_colors
        if _is_anchor_color(grid, color)
    ]
    marker_options = [
        color
        for color in other_colors
        if _is_marker_color(grid, color)
    ]

    assert len(anchor_options) == 1
    assert len(marker_options) == 1
    assert anchor_options[0] != marker_options[0]

    return background, anchor_options[0], marker_options[0]


def _candidate_moves(
    grid: np.ndarray,
    background: int,
    anchors: list[tuple[int, int]],
    marker_position: tuple[int, int],
) -> list[dict]:
    row, column = marker_position
    candidates: list[dict] = []

    for anchor_index, (anchor_row, anchor_column) in enumerate(anchors):
        if anchor_row <= row <= anchor_row + 1:
            if (
                column < anchor_column
                and np.all(
                    grid[row, column + 1:anchor_column] == background
                )
            ):
                destination = (row, anchor_column - 1)
                candidates.append({
                    "anchor_index": anchor_index,
                    "destination": destination,
                    "distance": destination[1] - column,
                    "axis": "horizontal",
                    "side": "left",
                })

            elif (
                column > anchor_column + 1
                and np.all(
                    grid[row, anchor_column + 2:column] == background
                )
            ):
                destination = (row, anchor_column + 2)
                candidates.append({
                    "anchor_index": anchor_index,
                    "destination": destination,
                    "distance": column - destination[1],
                    "axis": "horizontal",
                    "side": "right",
                })

        if anchor_column <= column <= anchor_column + 1:
            if (
                row < anchor_row
                and np.all(
                    grid[row + 1:anchor_row, column] == background
                )
            ):
                destination = (anchor_row - 1, column)
                candidates.append({
                    "anchor_index": anchor_index,
                    "destination": destination,
                    "distance": destination[0] - row,
                    "axis": "vertical",
                    "side": "above",
                })

            elif (
                row > anchor_row + 1
                and np.all(
                    grid[anchor_row + 2:row, column] == background
                )
            ):
                destination = (anchor_row + 2, column)
                candidates.append({
                    "anchor_index": anchor_index,
                    "destination": destination,
                    "distance": row - destination[0],
                    "axis": "vertical",
                    "side": "below",
                })

    return candidates


def _solve_grid(
    grid: np.ndarray,
    background: int,
    anchor_color: int,
    marker_color: int,
) -> tuple[np.ndarray, dict]:
    anchors = _extract_anchors(grid, anchor_color)
    marker_positions = [
        tuple(map(int, position))
        for position in np.argwhere(grid == marker_color)
    ]

    assert marker_positions
    assert _is_marker_color(grid, marker_color)

    moves: list[dict] = []
    inert: list[dict] = []

    for position in marker_positions:
        candidates = _candidate_moves(
            grid,
            background,
            anchors,
            position,
        )

        if candidates:
            minimum = min(
                candidate["distance"]
                for candidate in candidates
            )
            winners = [
                candidate
                for candidate in candidates
                if candidate["distance"] == minimum
            ]
        else:
            winners = []

        if len(winners) == 1 and winners[0]["distance"] > 0:
            chosen = dict(winners[0])
            chosen["source"] = position
            chosen["candidate_count"] = len(candidates)
            chosen["candidate_axes"] = tuple(
                candidate["axis"] for candidate in candidates
            )
            chosen["candidate_distances"] = tuple(sorted(
                candidate["distance"]
                for candidate in candidates
            ))
            moves.append(chosen)
        else:
            inert.append({
                "position": position,
                "candidate_count": len(candidates),
                "minimum_tie_count": len(winners),
            })

    destinations = [move["destination"] for move in moves]

    assert len(destinations) == len(set(destinations))
    assert all(
        grid[destination] == background
        for destination in destinations
    )

    output = grid.copy()

    for move in moves:
        output[move["source"]] = background

    for move in moves:
        output[move["destination"]] = marker_color

    facts = {
        "anchors": anchors,
        "marker_count": len(marker_positions),
        "moves": moves,
        "inert": inert,
    }
    return output, facts



def _validate_requested_count(name, value, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(seed, num_train, num_test):
    _validate_requested_count("num_train", num_train, 2, 10)
    _validate_requested_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        test_count = 1 + seed % 4
        low = test_count + 1
        train_count = low + (seed // 4) % (11 - low)
    elif num_train is None:
        test_count = int(num_test)
        low = test_count + 1
        train_count = low + seed % (11 - low)
    elif num_test is None:
        train_count = int(num_train)
        test_count = 1 + seed % min(4, train_count - 1)
    else:
        train_count, test_count = int(num_train), int(num_test)
    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")
    return train_count, test_count


def _pair_digest(input_grid, output_grid):
    a = np.asarray(input_grid, dtype=np.int8)
    b = np.asarray(output_grid, dtype=np.int8)
    payload = bytes(a.shape) + a.tobytes() + b.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _sample_anchors(rng, height, width, count):
    anchors = []
    for _ in range(count):
        for _ in range(160):
            if anchors and rng.random() < 0.32:
                old_row, old_column = anchors[int(rng.integers(len(anchors)))]
                if rng.random() < 0.5:
                    candidate = (old_row, int(rng.integers(width - 1)))
                else:
                    candidate = (int(rng.integers(height - 1)), old_column)
            else:
                candidate = (int(rng.integers(height - 1)), int(rng.integers(width - 1)))
            # Keep a full blank-cell moat between separate blocks.
            if all(abs(candidate[0] - r) >= 4 or abs(candidate[1] - c) >= 4
                   for r, c in anchors):
                anchors.append(candidate)
                break
        else:
            return None
    return sorted(anchors)


def _base_grid(height, width, anchors):
    grid = np.zeros((height, width), dtype=np.int8)
    for row, column in anchors:
        grid[row:row + 2, column:column + 2] = 1
    return grid


def _candidate_pool(grid, anchors):
    height, width = grid.shape
    positions = set()
    for row, column in anchors:
        for r in (row, row + 1):
            positions.update((r, c) for c in range(width))
        for c in (column, column + 1):
            positions.update((r, c) for r in range(height))
    pool = []
    for position in sorted(positions):
        if grid[position] != 0:
            continue
        candidates = _candidate_moves(grid, 0, anchors, position)
        if not candidates:
            continue
        minimum = min(item["distance"] for item in candidates)
        winners = [item for item in candidates if item["distance"] == minimum]
        if len(winners) != 1 or minimum == 0:
            continue
        item = dict(winners[0])
        item["source"] = position
        item["axes"] = frozenset(x["axis"] for x in candidates)
        item["candidate_count"] = len(candidates)
        item["margin"] = min((x["distance"] - minimum for x in candidates
                              if x is not winners[0]), default=99)
        pool.append(item)
    return pool


def _can_add_active(grid, anchors, chosen, item):
    row, column = item["source"]
    if grid[row, column] != 0:
        return False
    for other_row, other_column in np.argwhere(grid == 2):
        if abs(row - int(other_row)) + abs(column - int(other_column)) == 1:
            return False
    trial = grid.copy()
    trial[row, column] = 2
    try:
        _, facts = _solve_grid(trial, 0, 1, 2)
    except AssertionError:
        return False
    moves = {move["source"]: move for move in facts["moves"]}
    if any(source not in moves or moves[source]["destination"] != destination
           for source, destination in chosen.items()):
        return False
    return position_in_moves(item, moves)


def position_in_moves(item, moves):
    source = item["source"]
    return source in moves and moves[source]["destination"] == item["destination"]


def _add_from_pool(rng, grid, anchors, pool, chosen, predicate):
    for index in rng.permutation(len(pool)):
        item = pool[int(index)]
        if not predicate(item):
            continue
        if _can_add_active(grid, anchors, chosen, item):
            source = item["source"]
            grid[source] = 2
            chosen[source] = item["destination"]
            return True
    return False


def _add_stationary_markers(rng, grid, anchors, count, try_diagonal):
    height, width = grid.shape
    anchor_rows = {r + d for r, _ in anchors for d in (0, 1)}
    anchor_columns = {c + d for _, c in anchors for d in (0, 1)}
    added = 0
    diagonal_added = False
    for _ in range(count):
        current = [tuple(map(int, p)) for p in np.argwhere(grid == 2)]
        choices = []
        for row in range(height):
            for column in range(width):
                if grid[row, column] != 0:
                    continue
                if row in anchor_rows or column in anchor_columns:
                    continue
                if any(abs(row - r) + abs(column - c) == 1 for r, c in current):
                    continue
                diagonal = any(abs(row - r) == 1 and abs(column - c) == 1
                               for r, c in current)
                interior = 0 < row < height - 1 and 0 < column < width - 1
                weight = 5 if interior else 1
                if try_diagonal and not diagonal_added and diagonal:
                    weight *= 30
                choices.extend([(row, column, diagonal)] * weight)
        if not choices:
            break
        row, column, diagonal = choices[int(rng.integers(len(choices)))]
        grid[row, column] = 2
        added += 1
        diagonal_added |= diagonal
    return added, diagonal_added


def _fallback_input(rng, mode):
    # Vetted source-distinct scenes keep construction total if a random
    # search ever exhausts its bounded budget.
    scenes = {
        "challenge": (
            (23, 24),
            [(3, 10), (5, 17), (11, 11), (11, 21), (15, 10)],
            [(4, 13), (8, 22), (10, 18), (12, 17), (14, 12),
             (15, 3), (15, 22), (16, 1), (16, 19), (17, 10)],
        ),
        "simple": (
            (17, 10),
            [(1, 1), (9, 5), (15, 1)],
            [(6, 1), (11, 1)],
        ),
        "standard": (
            (22, 17),
            [(17, 0)],
            [(1, 1), (9, 0), (17, 14), (18, 8), (20, 0), (21, 1)],
        ),
    }
    shape, anchors, markers = scenes[mode]
    grid = _base_grid(*shape, anchors)
    for position in markers:
        grid[position] = 2
    if rng.integers(2):
        grid = grid.T
    if rng.integers(2):
        grid = np.flip(grid, axis=0)
    if rng.integers(2):
        grid = np.flip(grid, axis=1)
    return grid.copy()


def _build_input(rng, mode):
    for _ in range(100):
        if mode == "challenge":
            height = int(rng.integers(23, 31))
            width = int(rng.integers(23, 31))
            anchor_count = int(rng.integers(4, 6))
            target_active = int(rng.integers(8, 13))
        elif mode == "simple":
            height = int(rng.integers(10, 20))
            width = int(rng.integers(10, 20))
            anchor_count = int(rng.integers(1, 4))
            target_active = int(rng.integers(2, 6))
        else:
            height = int(rng.integers(15, 31))
            width = int(rng.integers(15, 31))
            anchor_count = int(rng.integers(1, 6))
            target_active = int(rng.integers(3, 11))
        anchors = _sample_anchors(rng, height, width, anchor_count)
        if anchors is None:
            continue
        grid = _base_grid(height, width, anchors)
        pool = _candidate_pool(grid, anchors)
        if len(pool) < target_active:
            continue
        chosen = {}
        if mode == "challenge":
            required = (
                lambda x: x["axes"] == {"horizontal", "vertical"}
                          and x["axis"] == "horizontal" and x["distance"] >= 2,
                lambda x: x["axes"] == {"horizontal", "vertical"}
                          and x["axis"] == "vertical" and x["distance"] >= 2,
                lambda x: x["side"] == "left",
                lambda x: x["side"] == "right",
                lambda x: x["side"] == "above",
                lambda x: x["side"] == "below",
                lambda x: x["distance"] == 1,
                lambda x: x["distance"] >= 6,
            )
            if not all(_add_from_pool(rng, grid, anchors, pool, chosen, condition)
                       for condition in required):
                continue
        else:
            if not _add_from_pool(rng, grid, anchors, pool, chosen,
                                  lambda x: x["distance"] >= 2):
                continue
        for _ in range(target_active - len(chosen)):
            if not _add_from_pool(rng, grid, anchors, pool, chosen,
                                  lambda x: True):
                break
        if len(chosen) < (7 if mode == "challenge" else 2):
            continue
        # Adjacent markers have a visible zero-travel answer. Their placement
        # is optional and never allowed to alter an existing move.
        if rng.random() < 0.32:
            adjacent = []
            for position in sorted({x["destination"] for x in pool}):
                if grid[position] != 0:
                    continue
                if any(abs(position[0] - int(r)) + abs(position[1] - int(c)) == 1
                       for r, c in np.argwhere(grid == 2)):
                    continue
                trial = grid.copy()
                trial[position] = 2
                try:
                    _, trial_facts = _solve_grid(trial, 0, 1, 2)
                except AssertionError:
                    continue
                trial_moves = {m["source"]: m["destination"]
                               for m in trial_facts["moves"]}
                if position not in trial_moves and all(trial_moves.get(s) == d
                                                        for s, d in chosen.items()):
                    adjacent.append(position)
            if adjacent:
                grid[adjacent[int(rng.integers(len(adjacent)))]] = 2
        stationary_count = int(rng.integers(0, 6 if mode != "simple" else 4))
        _add_stationary_markers(rng, grid, anchors, stationary_count,
                                try_diagonal=rng.random() < 0.55)
        try:
            _, facts = _solve_grid(grid, 0, 1, 2)
        except AssertionError:
            continue
        if mode == "challenge":
            moves = facts["moves"]
            sides = {m["side"] for m in moves}
            cross_h = any(m["axis"] == "horizontal" and
                          set(m["candidate_axes"]) == {"horizontal", "vertical"}
                          for m in moves)
            cross_v = any(m["axis"] == "vertical" and
                          set(m["candidate_axes"]) == {"horizontal", "vertical"}
                          for m in moves)
            if not (sides == _SIDES and cross_h and cross_v and
                    any(m["distance"] == 1 for m in moves) and
                    any(m["distance"] >= 6 for m in moves)):
                continue
        return grid
    return _fallback_input(rng, mode)


def _recolor(grid, roles):
    return np.asarray(roles, dtype=np.int8)[grid]


def _pair_key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    base_roles = tuple(map(int, rng.choice(10, size=3, replace=False)))
    variable_colors = bool(rng.integers(0, 2))
    training_modes = ["standard"] * train_count
    challenge_index = int(rng.integers(train_count))
    training_modes[challenge_index] = "challenge"
    small_index = (challenge_index + 1 + int(rng.integers(train_count - 1))) % train_count
    training_modes[small_index] = "simple"
    for index in range(train_count):
        if training_modes[index] == "standard":
            training_modes[index] = "simple" if rng.random() < 0.28 else "standard"
    test_modes = [rng.choice(["simple", "standard", "challenge"], p=[0.22, 0.37, 0.41])
                  for _ in range(test_count)]
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=train_count + test_count, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_variable_roles = None
    cursor = 0
    for section, modes in (("train", training_modes), ("test", test_modes)):
        for mode in modes:
            pair_rng = np.random.default_rng(int(pair_seeds[cursor]))
            cursor += 1
            for _ in range(12):
                geometry = _build_input(pair_rng, mode)
                if variable_colors:
                    roles = tuple(map(int, pair_rng.choice(10, size=3, replace=False)))
                    if first_variable_roles is None:
                        first_variable_roles = roles
                    elif section == "train" and len(puzzle["train"]) == 1:
                        while roles == first_variable_roles:
                            roles = tuple(map(int, pair_rng.choice(10, size=3, replace=False)))
                else:
                    roles = base_roles
                input_grid = _recolor(geometry, roles)
                output_grid, _ = _solve_grid(input_grid, *roles)
                key = _pair_key(input_grid)
                if key in seen or _pair_digest(input_grid, output_grid) in _SOURCE_PAIR_DIGESTS:
                    continue
                seen.add(key)
                puzzle[section].append({"input": input_grid, "output": output_grid})
                break
            else:
                raise RuntimeError("could not produce distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_facts(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    for grid in (pair["input"], pair["output"]):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert np.issubdtype(grid.dtype, np.integer)
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.all((0 <= grid) & (grid <= 9))
    input_grid, output_grid = pair["input"], pair["output"]
    assert input_grid.shape == output_grid.shape
    roles = _infer_roles(input_grid)
    expected, facts = _solve_grid(input_grid, *roles)
    assert np.array_equal(output_grid, expected)
    assert facts["moves"]
    assert np.array_equal(input_grid == roles[1], output_grid == roles[1])
    assert np.count_nonzero(input_grid == roles[2]) == np.count_nonzero(output_grid == roles[2])
    assert np.count_nonzero(input_grid != output_grid) == 2 * len(facts["moves"])
    assert _pair_digest(input_grid, output_grid) not in _SOURCE_PAIR_DIGESTS
    for move in facts["moves"]:
        assert move["distance"] > 0
        assert move["candidate_distances"].count(move["distance"]) == 1
        assert input_grid[move["destination"]] == roles[0]
        assert output_grid[move["source"]] == roles[0]
        assert output_grid[move["destination"]] == roles[2]
    return roles, facts


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    palettes = []
    training_moves = []
    training_facts = []
    testing_facts = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            roles, facts = _pair_facts(pair)
            key = _pair_key(pair["input"])
            assert key not in seen
            seen.add(key)
            palettes.append(roles)
            if section == "train":
                training_moves.extend(facts["moves"])
                training_facts.append((pair, facts))
            else:
                testing_facts.append((pair, facts))
    training_roles = palettes[:len(puzzle["train"])]
    test_roles = palettes[len(puzzle["train"]):]
    if len(set(training_roles)) == 1:
        assert all(roles == training_roles[0] for roles in test_roles)
    assert any(23 <= p["input"].shape[0] <= 30 and
               23 <= p["input"].shape[1] <= 30 and
               len(f["anchors"]) >= 4 for p, f in training_facts)
    assert any(10 <= p["input"].shape[0] <= 19 and
               10 <= p["input"].shape[1] <= 19 for p, _ in training_facts)
    assert {m["side"] for m in training_moves} == _SIDES
    assert any(m["distance"] == 1 for m in training_moves)
    assert any(m["distance"] >= 6 for m in training_moves)
    for axis in ("horizontal", "vertical"):
        assert any(m["axis"] == axis and
                   set(m["candidate_axes"]) == {"horizontal", "vertical"}
                   for m in training_moves)
    for pair, facts in testing_facts:
        assert 1 <= len(facts["anchors"]) <= 5
        assert all(m["side"] in {x["side"] for x in training_moves}
                   for m in facts["moves"])


def _structural_signature(pair):
    roles, facts = _pair_facts(pair)
    return (pair["input"].shape, tuple(facts["anchors"]),
            tuple(sorted((m["source"], m["destination"]) for m in facts["moves"])))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    palettes = set()
    shapes = set()
    layouts = set()
    anchor_counts = set()
    marker_counts = set()
    cross_h = cross_v = diagonal = long_moves = zero_travel = 0
    test_without_edge = 0
    for puzzle in puzzles:
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid = pair["input"]
                roles, facts = _pair_facts(pair)
                palettes.add(roles)
                shapes.add(grid.shape)
                layouts.add(_structural_signature(pair))
                anchor_counts.add(len(facts["anchors"]))
                marker_counts.add(facts["marker_count"])
                long_moves += any(m["distance"] >= 7 for m in facts["moves"])
                cross_h += any(m["axis"] == "horizontal" and
                               set(m["candidate_axes"]) == {"horizontal", "vertical"}
                               for m in facts["moves"])
                cross_v += any(m["axis"] == "vertical" and
                               set(m["candidate_axes"]) == {"horizontal", "vertical"}
                               for m in facts["moves"])
                positions = [tuple(map(int, p)) for p in np.argwhere(grid == roles[2])]
                diagonal += any(abs(a[0]-b[0]) == 1 and abs(a[1]-b[1]) == 1
                                for i, a in enumerate(positions) for b in positions[i+1:])
                zero_travel += any(i["candidate_count"] > 0 for i in facts["inert"])
                if section == "test":
                    test_without_edge += not any(r in (0, grid.shape[0]-1) or
                                                 c in (0, grid.shape[1]-1)
                                                 for r, c in positions)
    assert len(palettes) >= 20 and len(shapes) >= 35 and len(layouts) >= 100
    assert anchor_counts.issuperset({1, 2, 3, 4, 5})
    assert len(marker_counts) >= 10
    assert cross_h >= 20 and cross_v >= 20
    assert diagonal >= 5 and long_moves >= 20 and zero_travel >= 5
    assert test_without_edge >= 5
    a = generate_puzzle(1234, 7, 3)
    b = generate_puzzle(1234, 7, 3)
    for section in ("train", "test"):
        for pa, pb in zip(a[section], b[section]):
            assert np.array_equal(pa["input"], pb["input"])
            assert np.array_equal(pa["output"], pb["output"])
    assert _pair_key(a["train"][0]["input"]) != _pair_key(
        generate_puzzle(1235, 7, 3)["train"][0]["input"])
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(2100 + train_count + test_count, train_count, test_count)
        assert len(puzzle["train"]) == train_count and len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for seed, train_count, test_count in ((3001, None, 1), (3002, None, 4),
                                          (3003, 2, None), (3004, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(4000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal request accepted: {kwargs}")
