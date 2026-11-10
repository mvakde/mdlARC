import numpy as np


_DIRECTIONS = ("top", "bottom", "left", "right")

# Exact source inputs. If an input differs, its complete input/output pair
# necessarily differs from every source pair.
_SOURCE_INPUT_ROWS = (
    (
        "008008000800",
        "008008000800",
        "008008000800",
        "208000000800",
        "008008000800",
        "000008000800",
        "008008000000",
        "008008000800",
        "000008000000",
        "008008000800",
        "008008000800",
    ),
    (
        "00000002000000000",
        "00000000000000000",
        "88888088808888888",
        "00000000000000000",
        "88088888888808888",
        "00000000000000000",
        "00000000000000000",
        "88888088888888888",
        "00000000000000000",
        "00000000000000000",
        "00000000000000000",
    ),
    (
        "0080080008000",
        "0000080008000",
        "0080080000000",
        "0000080008000",
        "0080080008002",
        "0080080000000",
        "0080080008000",
        "0080080008000",
        "0000000008000",
        "0080080008000",
    ),
    (
        "00000020000000",
        "00000000000000",
        "88880888888888",
        "00000000000000",
        "88088880888088",
        "00000000000000",
        "00000000000000",
        "88088888808888",
        "00000000000000",
        "00000000000000",
        "88880880888088",
        "00000000000000",
        "00000000000000",
        "00000000000000",
        "00000000000000",
    ),
)


def _content_key(grid: np.ndarray) -> tuple:
    return tuple(grid.shape), grid.tobytes()


def _is_source_input(grid: np.ndarray) -> bool:
    rows = tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )
    return rows in _SOURCE_INPUT_ROWS


def _to_canonical(grid: np.ndarray, direction: str) -> np.ndarray:
    """Put the seed on the top edge, with propagation increasing by row."""
    if direction == "top":
        return grid.copy()
    if direction == "bottom":
        return grid[::-1, :].copy()
    if direction == "left":
        return grid.T.copy()

    assert direction == "right"
    return grid[:, ::-1].T.copy()


def _from_canonical(grid: np.ndarray, direction: str) -> np.ndarray:
    """Undo _to_canonical."""
    if direction == "top":
        return grid.copy()
    if direction == "bottom":
        return grid[::-1, :].copy()
    if direction == "left":
        return grid.T.copy()

    assert direction == "right"
    return grid.T[:, ::-1].copy()


def _inspect_input(grid: np.ndarray) -> dict:
    """
    Infer all semantic roles and structural properties from an input.

    The returned canonical grid has depth on axis zero and the seed at
    canonical depth zero.
    """
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2

    values = [int(value) for value in np.unique(grid)]
    assert len(values) == 3

    # The signal is the only color occurring once at a non-corner boundary
    # position.
    signal_candidates = []
    for color in values:
        locations = np.argwhere(grid == color)
        if len(locations) != 1:
            continue

        row, column = (int(value) for value in locations[0])
        boundary_sides = (
            int(row == 0)
            + int(row == grid.shape[0] - 1)
            + int(column == 0)
            + int(column == grid.shape[1] - 1)
        )
        if boundary_sides == 1:
            signal_candidates.append((color, row, column))

    assert len(signal_candidates) == 1
    signal, seed_row, seed_column = signal_candidates[0]

    if seed_row == 0:
        direction = "top"
    elif seed_row == grid.shape[0] - 1:
        direction = "bottom"
    elif seed_column == 0:
        direction = "left"
    else:
        direction = "right"

    canonical = _to_canonical(grid, direction)
    depth, lateral = canonical.shape

    seed_locations = np.argwhere(canonical == signal)
    assert seed_locations.shape == (1, 2)
    assert int(seed_locations[0, 0]) == 0

    seed_lateral = int(seed_locations[0, 1])
    assert 0 < seed_lateral < lateral - 1

    # The barrier color is confined to dense, parallel canonical rows.
    barrier_candidates = []
    for color in values:
        if color == signal:
            continue

        locations = np.argwhere(canonical == color)
        occupied_depths = np.unique(locations[:, 0])

        if not 2 <= len(occupied_depths) <= 6:
            continue
        if np.any(occupied_depths < 1):
            continue
        if np.any(occupied_depths > depth - 2):
            continue

        valid = True
        for barrier_depth in occupied_depths:
            row = canonical[int(barrier_depth)]
            if row[0] != color or row[-1] != color:
                valid = False
                break
            if int(np.count_nonzero(row == color)) < (lateral + 1) // 2:
                valid = False
                break

        if valid:
            barrier_candidates.append(
                (
                    color,
                    tuple(int(value) for value in occupied_depths),
                )
            )

    assert len(barrier_candidates) == 1
    barrier, barriers = barrier_candidates[0]

    background_values = set(values) - {signal, barrier}
    assert len(background_values) == 1
    background = int(next(iter(background_values)))

    assert barriers[0] >= 2
    assert barriers[-1] <= depth - 2
    assert all(
        later - earlier >= 2
        for earlier, later in zip(barriers, barriers[1:])
    )

    aperture_counts = []
    for barrier_depth in barriers:
        row = canonical[barrier_depth]

        assert np.all((row == barrier) | (row == background))

        apertures = np.flatnonzero(row == background)
        assert len(apertures) >= 1
        assert np.all((apertures > 0) & (apertures < lateral - 1))
        assert len(apertures) <= (lateral - 2) // 2

        aperture_counts.append(int(len(apertures)))

    nonbarrier_depths = np.ones(depth, dtype=bool)
    nonbarrier_depths[list(barriers)] = False

    assert not np.any(canonical[nonbarrier_depths] == barrier)
    assert int(np.count_nonzero(canonical == signal)) == 1

    return {
        "roles": (background, barrier, signal),
        "direction": direction,
        "canonical": canonical,
        "depth": depth,
        "lateral": lateral,
        "seed_lateral": seed_lateral,
        "barriers": barriers,
        "aperture_counts": tuple(aperture_counts),
    }


def _trace_canonical(
    canonical: np.ndarray,
    roles: tuple[int, int, int],
    barriers: tuple[int, ...],
    seed_lateral: int,
) -> tuple[np.ndarray, dict]:
    """
    Apply the signal rule in canonical top-to-bottom coordinates.

    Signal branches are represented by their lateral positions after the
    most recently crossed barrier.
    """
    background, barrier, signal = roles
    depth, lateral = canonical.shape

    output = canonical.copy()
    active = {int(seed_lateral)}
    previous_depth = 0

    metrics = {
        "direct_crossings": 0,
        "collisions": 0,
        "redirected_crossings": 0,
        "boundary_terminations": 0,
        "lateral_steps": 0,
        "mergers": 0,
        "active_counts": [],
    }

    for barrier_depth in barriers:
        outgoing = set()

        for position in sorted(active):
            # Continue straight up to the near side of this barrier.
            output[
                previous_depth + 1:barrier_depth,
                position,
            ] = signal

            # An aligned aperture permits direct passage.
            if canonical[barrier_depth, position] == background:
                output[barrier_depth, position] = signal

                if position in outgoing:
                    metrics["mergers"] += 1
                outgoing.add(position)
                metrics["direct_crossings"] += 1
                continue

            # A solid cell redirects the signal in both lateral directions
            # along the barrier's near side.
            assert canonical[barrier_depth, position] == barrier
            metrics["collisions"] += 1
            near_side = barrier_depth - 1

            for delta in (-1, 1):
                lateral_position = position + delta
                found_aperture = False

                while 0 <= lateral_position < lateral:
                    output[near_side, lateral_position] = signal
                    metrics["lateral_steps"] += 1

                    if (
                        canonical[barrier_depth, lateral_position]
                        == background
                    ):
                        output[
                            barrier_depth,
                            lateral_position,
                        ] = signal

                        if lateral_position in outgoing:
                            metrics["mergers"] += 1
                        outgoing.add(lateral_position)

                        metrics["redirected_crossings"] += 1
                        found_aperture = True
                        break

                    lateral_position += delta

                if not found_aperture:
                    metrics["boundary_terminations"] += 1

        active = outgoing
        previous_depth = barrier_depth
        metrics["active_counts"].append(len(active))

    # Any branches surviving the final barrier run to the far boundary.
    for position in active:
        output[previous_depth + 1:, position] = signal

    return output, metrics


def _solve_input(
    grid: np.ndarray,
) -> tuple[np.ndarray, dict, dict]:
    info = _inspect_input(grid)

    canonical_output, metrics = _trace_canonical(
        info["canonical"],
        info["roles"],
        info["barriers"],
        info["seed_lateral"],
    )

    output = _from_canonical(
        canonical_output,
        info["direction"],
    )
    return output, info, metrics


def _sample_barrier_depths(
    rng: np.random.Generator, depth: int, count: int,
) -> tuple[int, ...]:
    """Sample separated rows without rejection."""
    reduced = np.sort(rng.choice(np.arange(2, depth - count), size=count, replace=False))
    return tuple(int(value) + index for index, value in enumerate(reduced))


def _advance_positions(active: set[int], holes: set[int], lateral: int) -> set[int]:
    outgoing = set()
    for position in active:
        if position in holes:
            outgoing.add(position)
            continue
        for delta in (-1, 1):
            candidate = position + delta
            while 0 <= candidate < lateral:
                if candidate in holes:
                    outgoing.add(candidate)
                    break
                candidate += delta
    return outgoing


def _random_holes(
    rng: np.random.Generator, active: set[int], lateral: int,
) -> set[int]:
    interior = np.arange(1, lateral - 1)
    maximum = min(5, (lateral - 2) // 2)
    style = int(rng.integers(0, 5))
    if style == 0:
        # An aligned hole keeps at least one existing branch straight.
        chosen = int(rng.choice(sorted(active)))
        return {chosen}
    if style == 1:
        # A single offset hole forces all branches through a bottleneck.
        return {int(rng.choice(interior))}
    if style == 2:
        # Openings by the side edges make long sideways traces possible.
        return {1 if int(rng.integers(0, 2)) == 0 else lateral - 2}
    if style == 3:
        count = int(rng.integers(2, maximum + 1))
    else:
        count = int(rng.integers(1, maximum + 1))
    return {int(value) for value in rng.choice(interior, size=count, replace=False)}


def _make_canonical(
    rng: np.random.Generator,
    roles: tuple[int, int, int],
    depth: int,
    lateral: int,
    barriers: tuple[int, ...],
    profile: str,
) -> np.ndarray:
    background, barrier, signal = roles
    seed = int(rng.integers(3, lateral - 3))
    canonical = np.full((depth, lateral), background, dtype=np.int8)
    canonical[0, seed] = signal
    active = {seed}
    first_left = first_right = 0

    for index, wall_depth in enumerate(barriers):
        if profile == "direct":
            holes = {seed}
        elif profile in ("single", "edge"):
            if profile == "edge" and index == 0:
                holes = {1 if int(rng.integers(0, 2)) == 0 else lateral - 2}
            else:
                current = next(iter(active))
                choices = [position for position in range(1, lateral - 1) if position != current]
                holes = {int(rng.choice(choices))}
        elif profile in ("complex", "late_bottleneck", "free", "branching") and index == 0:
            left_span = int(rng.integers(2, min(5, seed - 1) + 1))
            right_span = int(rng.integers(2, min(5, lateral - 2 - seed) + 1))
            first_left, first_right = seed - left_span, seed + right_span
            holes = {first_left, first_right}
        elif profile == "complex" and index == 1:
            # Both branches from wall one collide and merge at this hole.
            holes = {seed}
        elif profile == "complex" and index == 2:
            # An aligned passage is demonstrated in the same training pair.
            holes = {seed}
        elif profile == "late_bottleneck" and index == len(barriers) - 1:
            holes = {int(rng.integers(first_left + 1, first_right))}
        elif profile == "branching":
            for _ in range(40):
                candidate = {int(value) for value in rng.choice(
                    np.arange(1, lateral - 1), size=2, replace=False,
                )}
                if len(_advance_positions(active, candidate, lateral)) >= 2:
                    holes = candidate
                    break
            else:
                holes = set(sorted(active)[:2])
        else:
            holes = _random_holes(rng, active, lateral)

        canonical[wall_depth, :] = barrier
        canonical[wall_depth, sorted(holes)] = background
        active = _advance_positions(active, holes, lateral)
        assert active

    return canonical


def _sample_layer_count(rng: np.random.Generator, regime: str, profile: str) -> int:
    counts = np.array([2, 3, 4, 5, 6])
    if regime == "source_train":
        probabilities = np.array([0.17, 0.37, 0.27, 0.14, 0.05])
    else:
        probabilities = np.array([0.20, 0.24, 0.25, 0.20, 0.11])
    if profile == "complex":
        probabilities[0] = 0
    if profile == "late_bottleneck":
        probabilities[0] = 0
    probabilities /= probabilities.sum()
    return int(rng.choice(counts, p=probabilities))


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    roles: tuple[int, int, int],
    direction: str,
    profile: str,
) -> dict:
    layer_count = _sample_layer_count(rng, regime, profile)
    minimum_depth = max(8, 2 * layer_count + 2)
    if regime == "target_test" and rng.random() < 0.40:
        depth = int(rng.integers(max(18, minimum_depth), 31))
    else:
        upper = 25 if regime == "source_train" else 26
        depth = int(rng.integers(minimum_depth, upper + 1))
    lateral = int(rng.integers(9, 30))
    barriers = _sample_barrier_depths(rng, depth, layer_count)
    canonical = _make_canonical(rng, roles, depth, lateral, barriers, profile)
    seed_lateral = int(np.flatnonzero(canonical[0] == roles[2])[0])
    output, _ = _trace_canonical(canonical, roles, barriers, seed_lateral)
    return {
        "input": _from_canonical(canonical, direction),
        "output": _from_canonical(output, direction),
    }


def _normalize_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _choose_counts(
    rng: np.random.Generator, num_train: int | None, num_test: int | None,
) -> tuple[int, int]:
    weights = np.array([23, 23, 20, 12, 8, 6, 4, 3, 1], dtype=float)
    possible_train = np.arange(2, 11)
    if num_train is None:
        eligible = possible_train if num_test is None else possible_train[possible_train > num_test]
        eligible_weights = weights[eligible - 2]
        num_train = int(rng.choice(eligible, p=eligible_weights / eligible_weights.sum()))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _choose_directions(
    rng: np.random.Generator, num_train: int, num_test: int,
) -> tuple[list[str], list[str]]:
    mode = int(rng.choice(4, p=[0.20, 0.20, 0.25, 0.35]))
    anchor = _DIRECTIONS[int(rng.integers(0, 4))]
    opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    if mode == 0:
        return [anchor] * num_train, [anchor] * num_test
    if mode == 1:
        return [anchor] * num_train, [opposite[anchor]] * num_test
    if mode == 2:
        available = [direction for direction in _DIRECTIONS if direction != anchor]
        train = [available[int(rng.integers(0, 3))]]
        other_axis = [direction for direction in available if
                      (direction in ("top", "bottom")) != (train[0] in ("top", "bottom"))]
        train.append(other_axis[int(rng.integers(0, len(other_axis)))])
        while len(train) < num_train:
            train.append(available[int(rng.integers(0, 3))])
        test = [anchor]
        while len(test) < num_test:
            choices = [direction for direction in _DIRECTIONS if direction != test[0]]
            test.append(choices[int(rng.integers(0, 3))])
        return train, test
    train = [_DIRECTIONS[int(rng.integers(0, 4))] for _ in range(num_train)]
    test = [_DIRECTIONS[int(rng.integers(0, 4))] for _ in range(num_test)]
    if num_test > 1 and test[1] == test[0]:
        choices = [direction for direction in _DIRECTIONS if direction != test[0]]
        test[1] = choices[int(rng.integers(0, 3))]
    return train, test


def _choose_palettes(rng: np.random.Generator, count: int) -> list[tuple[int, int, int]]:
    base = tuple(int(value) for value in rng.choice(10, size=3, replace=False))
    mode = int(rng.choice(3, p=[0.45, 0.35, 0.20]))
    if mode == 0:
        return [base] * count
    if mode == 1:
        palettes = [base, (base[1], base[2], base[0])]
        for _ in range(count - 2):
            palettes.append(tuple(int(value) for value in rng.permutation(base)))
        return palettes
    palettes = [base]
    for _ in range(count - 1):
        candidate = tuple(int(value) for value in rng.choice(10, size=3, replace=False))
        if candidate == palettes[-1]:
            candidate = (candidate[1], candidate[2], candidate[0])
        palettes.append(candidate)
    return palettes


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    num_train = _normalize_count(num_train, "num_train", 2, 10)
    num_test = _normalize_count(num_test, "num_test", 1, 4)
    rng = np.random.default_rng(seed)
    num_train, num_test = _choose_counts(rng, num_train, num_test)

    train_directions, test_directions = _choose_directions(rng, num_train, num_test)
    palettes = _choose_palettes(rng, num_train + num_test)
    train_profiles = ["complex", "direct" if rng.random() < 0.43 else
                      ("single" if rng.random() < 0.58 else "edge")]
    extra_profiles = ("complex", "free", "late_bottleneck", "branching", "single", "edge", "direct")
    extra_weights = (0.17, 0.19, 0.17, 0.17, 0.14, 0.09, 0.07)
    for _ in range(num_train - 2):
        train_profiles.append(str(rng.choice(extra_profiles, p=extra_weights)))
    rng.shuffle(train_profiles)

    puzzle = {"train": [], "test": []}
    used_inputs = set()
    for section, count, directions in (("train", num_train, train_directions),
                                        ("test", num_test, test_directions)):
        for index in range(count):
            if section == "train":
                profile = train_profiles[index]
            else:
                profile = str(rng.choice(
                    ("complex", "late_bottleneck", "free", "branching", "single", "edge", "direct"),
                    p=(0.11, 0.25, 0.15, 0.31, 0.08, 0.06, 0.04),
                ))
            roles = palettes[index if section == "train" else num_train + index]
            for _ in range(12):
                child_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _generate_pair(child_rng, "source_train" if section == "train" else "target_test",
                                      roles, directions[index], profile)
                key = _content_key(pair["input"])
                if key not in used_inputs and not _is_source_input(pair["input"]):
                    used_inputs.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise ValueError("could not generate a unique pair")

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    training_metrics = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            input_key = _content_key(inp)
            pair_key = (input_key, _content_key(out))
            assert input_key not in inputs and pair_key not in pairs
            inputs.add(input_key)
            pairs.add(pair_key)
            assert not _is_source_input(inp)
            expected, info, metrics = _solve_input(inp)
            assert np.array_equal(out, expected)
            assert 2 <= len(info["barriers"]) <= 6
            assert 8 <= info["depth"] <= 30 and 9 <= info["lateral"] <= 29
            assert metrics["active_counts"][-1] >= 1
            if section == "train":
                training_metrics.append(metrics)
    # One demonstration shows a collision, two-sided routing, a merger,
    # a terminated branch, and a direct crossing. Another is simpler.
    assert any(m["collisions"] >= 2 and m["mergers"] >= 1 and
               m["boundary_terminations"] >= 1 and m["direct_crossings"] >= 1
               for m in training_metrics)
    assert any(m["collisions"] == 0 or
               (max(m["active_counts"]) == 1 and m["collisions"] >= 1)
               for m in training_metrics)


def _puzzles_equal(first: dict, second: dict) -> bool:
    for section in ("train", "test"):
        if len(first[section]) != len(second[section]):
            return False
        for a, b in zip(first[section], second[section]):
            if not np.array_equal(a["input"], b["input"]):
                return False
            if not np.array_equal(a["output"], b["output"]):
                return False
    return True


def _independent_output(grid: np.ndarray) -> np.ndarray:
    """Cellwise forward scan used only to cross-check the layer tracer."""
    info = _inspect_input(grid)
    source = info["canonical"]
    background, _, signal = info["roles"]
    result = source.copy()
    active = {info["seed_lateral"]}
    walls = set(info["barriers"])
    for depth in range(1, info["depth"]):
        if depth not in walls:
            for lateral in active:
                result[depth, lateral] = signal
            continue
        next_active = set()
        for lateral in active:
            if source[depth, lateral] == background:
                result[depth, lateral] = signal
                next_active.add(lateral)
                continue
            for step in (-1, 1):
                position = lateral + step
                while 0 <= position < info["lateral"]:
                    result[depth - 1, position] = signal
                    if source[depth, position] == background:
                        result[depth, position] = signal
                        next_active.add(position)
                        break
                    position += step
        active = next_active
    return _from_canonical(result, info["direction"])


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(314159), generate_puzzle(314159))
    assert _puzzles_equal(generate_puzzle(271828, 6, 3), generate_puzzle(271828, 6, 3))
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4),
                                    (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + index, *counts)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
        validate_generated_puzzle(puzzle)
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(99, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")

    train_counts, test_counts, shapes, layouts, palettes = set(), set(), set(), set(), set()
    directions, layer_counts, depths = set(), set(), set()
    simple_direct = single_reroute = bottlenecks = fewer_test_layers = 0
    same_axis = novel_direction = changed_palette = mixed_test_sides = 0
    compact = final_multi = small_depth = deep_depth = near_edge = wide_gap = 0
    stable_palette = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        compact += len(puzzle["train"]) <= 4
        infos = {"train": [], "test": []}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                expected, info, metrics = _solve_input(pair["input"])
                assert np.array_equal(expected, pair["output"])
                assert np.array_equal(_independent_output(pair["input"]), pair["output"])
                infos[section].append((info, metrics))
                shapes.add(tuple(pair["input"].shape))
                palettes.add(info["roles"])
                directions.add(info["direction"])
                layer_counts.add(len(info["barriers"]))
                depths.add(info["depth"])
                small_depth += 11 <= info["depth"] <= 15
                deep_depth += info["depth"] >= 25
                near_edge += any(info["canonical"][d, 1] == info["roles"][0] or
                                 info["canonical"][d, -2] == info["roles"][0]
                                 for d in info["barriers"])
                wide_gap += max(np.diff((0,) + info["barriers"] + (info["depth"] - 1,))) >= 5
                layouts.add((tuple(pair["input"].shape), info["direction"],
                             info["seed_lateral"], info["barriers"],
                             tuple(tuple(np.flatnonzero(info["canonical"][d] == info["roles"][0]))
                                   for d in info["barriers"])))
                simple_direct += int(section == "train" and metrics["collisions"] == 0)
                single_reroute += int(section == "train" and metrics["collisions"] > 0 and
                                      max(metrics["active_counts"]) == 1)
                bottlenecks += int(section == "test" and len(metrics["active_counts"]) >= 2 and
                                   max(metrics["active_counts"][:-1]) >= 2 and
                                   metrics["active_counts"][-1] == 1)
                final_multi += int(section == "test" and metrics["active_counts"][-1] >= 2)
        train_dirs = {info["direction"] for info, _ in infos["train"]}
        test_dirs = {info["direction"] for info, _ in infos["test"]}
        same_axis += int(len(train_dirs | test_dirs) == 1)
        novel_direction += int(bool(test_dirs - train_dirs))
        mixed_test_sides += int(len(test_dirs) > 1)
        changed_palette += int(len({info["roles"] for section in infos.values() for info, _ in section}) > 1)
        stable_palette += int(len({info["roles"] for section in infos.values() for info, _ in section}) == 1)
        fewer_test_layers += int(min(len(info["barriers"]) for info, _ in infos["test"]) <=
                                 max(len(info["barriers"]) for info, _ in infos["train"]))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 55 and len(layouts) >= 170 and len(palettes) >= 30
    assert directions == set(_DIRECTIONS) and layer_counts == set(range(2, 7))
    assert len(depths) >= 17
    assert simple_direct >= 5 and single_reroute >= 10 and bottlenecks >= 5
    assert same_axis >= 2 and novel_direction >= 5 and changed_palette >= 10
    assert mixed_test_sides >= 2 and fewer_test_layers >= 15
    assert compact >= 10 and final_multi >= 5 and small_depth >= 10 and deep_depth >= 5
    assert near_edge >= 20 and wide_gap >= 20 and stable_palette >= 5
    assert not _puzzles_equal(generate_puzzle(0), generate_puzzle(1))
