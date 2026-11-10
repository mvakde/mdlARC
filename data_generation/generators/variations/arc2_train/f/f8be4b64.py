import numpy as np


def _checked_count(value, name, low, high):
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


def _resolve_counts(rng, num_train, num_test):
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        maximum_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, maximum_test + 1))

    return num_train, num_test


def _sample_puzzle_parameters(rng):
    colors = [int(x) for x in rng.permutation(10)]
    return {
        "background": colors[0],
        "structure": colors[1],
        "stable_roles": bool(rng.integers(2)),
    }


def _pair_roles(rng, parameters, changed=False):
    if parameters["stable_roles"] or not changed:
        background = parameters["background"]
        structure = parameters["structure"]
    else:
        colors = [int(x) for x in rng.permutation(10)]
        background, structure = colors[:2]
        if changed and (background, structure) == (
            parameters["background"], parameters["structure"]
        ):
            background, structure = structure, background
    pool = tuple(
        int(x) for x in rng.permutation(
            [c for c in range(10) if c not in (background, structure)]
        )
    )
    return background, structure, pool


def _footprint(center):
    r, c = center
    return {
        (r, c),
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    }


def _arm_positions(centers):
    arms = set()
    for r, c in centers:
        arms.update(
            (
                (r - 1, c),
                (r + 1, c),
                (r, c - 1),
                (r, c + 1),
            )
        )
    return arms


def _geometric_centers(arms, height, width):
    centers = set()

    for r in range(1, height - 1):
        for c in range(1, width - 1):
            neighbors = (
                (r - 1, c),
                (r + 1, c),
                (r, c - 1),
                (r, c + 1),
            )
            if all(position in arms for position in neighbors):
                centers.add((r, c))

    return centers


def _can_add_center(candidate, centers):
    r, c = candidate

    # This prevents unspecified conflicts between rays of the same orientation.
    if any(r == old_r or c == old_c for old_r, old_c in centers):
        return False

    candidate_footprint = _footprint(candidate)
    return all(
        candidate_footprint.isdisjoint(_footprint(old))
        for old in centers
    )


def _place_centers(rng, height, width, count, pattern):
    centers = []

    if pattern == "double_clip":
        if count < 3 or height < 10 or width < 12:
            return None
        r = int(rng.integers(2, height - 2))
        c = int(rng.integers(5, width - 5))
        left = int(rng.integers(1, c - 3))
        right = int(rng.integers(c + 4, width - 1))
        centers = [(r, c), (r - 1, left), (r + 1, right)]

    elif pattern == "block":
        # Adjacent rows create two mutually blocking horizontal rays.
        # Adjacent columns create two mutually blocking vertical rays.
        if int(rng.integers(2)) == 0:
            low_r = int(rng.integers(1, height - 2))
            rows = [low_r, low_r + 1]

            for _ in range(100):
                c1 = int(rng.integers(1, width - 1))
                c2 = int(rng.integers(1, width - 1))
                if abs(c1 - c2) >= 5:
                    centers = [(rows[0], c1), (rows[1], c2)]
                    break
        else:
            low_c = int(rng.integers(1, width - 2))
            columns = [low_c, low_c + 1]

            for _ in range(100):
                r1 = int(rng.integers(1, height - 1))
                r2 = int(rng.integers(1, height - 1))
                if abs(r1 - r2) >= 5:
                    centers = [(r1, columns[0]), (r2, columns[1])]
                    break

        if len(centers) != 2:
            return None

        if int(rng.integers(2)):
            centers.reverse()

    candidates = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
    ]

    for index in rng.permutation(len(candidates)):
        if len(centers) == count:
            break

        candidate = candidates[int(index)]
        if _can_add_center(candidate, centers):
            centers.append(candidate)

    if len(centers) != count:
        return None

    arms = _arm_positions(centers)

    # No arm may be shared.
    if len(arms) != 4 * count:
        return None

    # Reject layouts whose structural cells accidentally imply extra crosses.
    if _geometric_centers(arms, height, width) != set(centers):
        return None

    return centers


def _build_input(
    height,
    width,
    centers,
    signals,
    background,
    structure,
):
    grid = np.full((height, width), background, dtype=np.int8)

    for r, c in _arm_positions(centers):
        grid[r, c] = structure

    for (r, c), signal in zip(centers, signals):
        grid[r, c] = signal

    return grid


def _trace_transform(grid, background, structure, centers):
    output = grid.copy()
    height, width = grid.shape

    structural = {
        tuple(position)
        for position in np.argwhere(grid == structure)
    }
    center_positions = {(r, c) for r, c, _ in centers}
    protected = structural | center_positions

    horizontal = {}
    vertical = {}
    blockers = []

    # Horizontal phase.
    for index, (r, c, signal) in enumerate(centers):
        for dr, dc in ((0, -1), (0, 1)):
            rr = r + 2 * dr
            cc = c + 2 * dc

            while 0 <= rr < height and 0 <= cc < width:
                position = (rr, cc)

                if position in structural:
                    blockers.append(
                        (index, dr, dc, position, signal)
                    )
                    break

                if position not in protected:
                    output[position] = signal
                    horizontal[position] = (index, signal)

                rr += dr
                cc += dc

    # Vertical phase. These writes take precedence at intersections.
    for index, (r, c, signal) in enumerate(centers):
        for dr, dc in ((-1, 0), (1, 0)):
            rr = r + 2 * dr
            cc = c + 2 * dc

            while 0 <= rr < height and 0 <= cc < width:
                position = (rr, cc)

                if position in structural:
                    blockers.append(
                        (index, dr, dc, position, signal)
                    )
                    break

                if position not in protected:
                    output[position] = signal
                    vertical[position] = (index, signal)

                rr += dr
                cc += dc

    # Explicit restoration makes protection independent of implementation
    # details in the ray loops.
    for r, c in protected:
        output[r, c] = grid[r, c]

    trace = {
        "horizontal": horizontal,
        "vertical": vertical,
        "blockers": blockers,
        "protected": protected,
    }
    return output, trace


def _analyze_pair(grid, background, structure, centers):
    output, trace = _trace_transform(
        grid,
        background,
        structure,
        centers,
    )

    intersections = (
        set(trace["horizontal"]) & set(trace["vertical"])
    )

    different_precedence = 0
    colored_precedence = 0
    background_precedence = 0

    for position in intersections:
        horizontal_signal = trace["horizontal"][position][1]
        vertical_signal = trace["vertical"][position][1]

        if horizontal_signal != vertical_signal:
            different_precedence += 1

            if (
                horizontal_signal != background
                and vertical_signal != background
            ):
                colored_precedence += 1

            if (
                horizontal_signal != background
                and vertical_signal == background
            ):
                background_precedence += 1

    blocked_colored = 0
    blocked_colored_witness = 0
    blocked_directions = {}
    height, width = grid.shape

    for index, dr, dc, blocker, signal in trace["blockers"]:
        if signal == background:
            continue

        blocked_colored += 1
        blocked_directions.setdefault(index, set()).add((dr, dc))
        rr = blocker[0] + dr
        cc = blocker[1] + dc
        witnessed = False

        # A background cell beyond the blocker that stays background visibly
        # distinguishes stopping from passing through the structural arm.
        while 0 <= rr < height and 0 <= cc < width:
            position = (rr, cc)

            if (
                position not in trace["protected"]
                and grid[position] == background
                and output[position] == background
            ):
                witnessed = True
                break

            rr += dr
            cc += dc

        if witnessed:
            blocked_colored_witness += 1

    stats = {
        "intersection_count": len(intersections),
        "different_precedence": different_precedence,
        "colored_precedence": colored_precedence,
        "background_precedence": background_precedence,
        "blocked_colored": blocked_colored,
        "blocked_colored_witness": blocked_colored_witness,
        "double_clip": any(
            {(0, -1), (0, 1)} <= directions
            or {(-1, 0), (1, 0)} <= directions
            for directions in blocked_directions.values()
        ),
        "one_sided_stop": sum(
            ((0, -1) in directions) != ((0, 1) in directions)
            for directions in blocked_directions.values()
        ) + sum(
            ((-1, 0) in directions) != ((1, 0) in directions)
            for directions in blocked_directions.values()
        ),
        "changed": int(np.count_nonzero(output != grid)),
    }
    return output, stats


def _sample_dimensions(rng, count):
    lower = 10 if count <= 4 else 15
    size_band = int(rng.integers(3))
    if size_band == 0 and lower == 10:
        low, high = 10, 15
    elif size_band == 1:
        low, high = max(lower, 14), 24
    else:
        low, high = max(lower, 23), 31
    height = int(rng.integers(low, high))
    if rng.integers(4) == 0:
        width = height
    elif size_band == 0 and lower == 10 and rng.integers(3) != 0:
        width = int(rng.integers(10, 15))
    else:
        width = int(rng.integers(lower, 31))
    return height, width


def _signals_for_kind(rng, kind, count, background, pool, known_colors):
    usable = [color for color in known_colors if color in pool]
    if kind == "colored_block_demo":
        signals = [pool[0], pool[1]]
    elif kind == "background_demo":
        signals = [pool[0], background]
    elif kind == "simple":
        return [pool[0]]
    elif kind == "double_clip":
        signals = [pool[0], pool[1], pool[2]]
        if rng.integers(2):
            signals[2] = background
        while len(signals) < count:
            signals.append(pool[int(rng.integers(len(pool)))])
        return [int(signal) for signal in signals]
    else:
        candidate_pool = usable if kind == "familiar" and usable else pool
        signals = [candidate_pool[int(rng.integers(len(candidate_pool)))]]
        if count >= 2:
            signals.append(candidate_pool[int(rng.integers(len(candidate_pool)))])

    while len(signals) < count:
        choices = usable if kind == "familiar" and usable else pool
        signals.append(choices[int(rng.integers(len(choices)))])

    if kind == "multi_background":
        signals[-1] = background
        signals[-2] = background
    elif kind in ("background_demo", "mixed"):
        if kind == "mixed" and rng.integers(2):
            signals[-1] = background
    elif kind == "repeat":
        signals[-1] = signals[0]
    elif kind == "extrapolate" and count >= 5:
        signals[:4] = list(pool[:4])
        if rng.integers(2):
            signals[-1] = background

    rng.shuffle(signals)
    return [int(signal) for signal in signals]


def _generate_pair(
    rng, regime, parameters, kind, known_colors=(), changed_roles=False,
    familiar_pair=None,
):
    if familiar_pair is None:
        background, structure, pool = _pair_roles(rng, parameters, changed_roles)
    else:
        background, structure, old_signals = familiar_pair
        pool = tuple(
            int(x) for x in rng.permutation(
                [c for c in range(10) if c not in (background, structure)]
            )
        )
    if kind == "colored_block_demo":
        cross_count, pattern = int(rng.integers(3, 5)), "block"
    elif kind == "background_demo":
        cross_count, pattern = int(rng.integers(2, 5)), "free"
    elif kind == "simple":
        cross_count, pattern = 1, "free"
    elif kind == "familiar" and familiar_pair is not None:
        cross_count, pattern = len(old_signals), "free"
    elif kind == "double_clip":
        upper = 5 if regime == "source_train" else 6
        cross_count, pattern = int(rng.integers(3, upper)), "double_clip"
    elif kind == "extrapolate":
        cross_count, pattern = int(rng.integers(5, 8)), "block"
    elif kind == "repeat":
        upper = 5 if regime == "source_train" else 6
        cross_count, pattern = int(rng.integers(3, upper)), "free"
    elif kind == "multi_background":
        upper = 5 if regime == "source_train" else 6
        cross_count, pattern = int(rng.integers(3, upper)), "free"
    else:
        upper = 5 if regime == "source_train" else 6
        cross_count = int(rng.integers(2, upper))
        pattern = "block" if rng.integers(3) == 0 else "free"

    if familiar_pair is None:
        signals = _signals_for_kind(
            rng, kind, cross_count, background, pool, known_colors
        )
    else:
        signals = list(old_signals)
        rng.shuffle(signals)
    for _ in range(2500):
        height, width = _sample_dimensions(rng, cross_count)
        centers = _place_centers(rng, height, width, cross_count, pattern)
        if centers is None:
            continue
        grid = _build_input(
            height, width, centers, signals, background, structure
        )
        described_centers = [
            (r, c, int(signal)) for (r, c), signal in zip(centers, signals)
        ]
        output, stats = _analyze_pair(
            grid, background, structure, described_centers
        )
        good = stats["changed"] > 0
        if kind == "colored_block_demo":
            good = good and stats["colored_precedence"] >= 1 and stats["blocked_colored_witness"] >= 1
        elif kind == "background_demo":
            good = good and stats["background_precedence"] >= 1
        elif kind == "double_clip":
            good = good and stats["double_clip"] and stats["blocked_colored_witness"] >= 1
        elif kind == "multi_background":
            good = good and stats["background_precedence"] >= 2
        if good:
            return grid, output
    raise RuntimeError("failed to construct a valid pair")


def _grid_key(grid):
    compact = np.asarray(grid, dtype=np.int8)
    return compact.shape, compact.tobytes()


def _infer_crosses(grid):
    values = np.asarray(grid, dtype=np.int64)
    counts = np.bincount(values.ravel(), minlength=10)
    background = int(np.argmax(counts))
    height, width = grid.shape
    role_candidates = []

    for structure in (
        int(value)
        for value in np.unique(grid)
        if int(value) != background
    ):
        structural = {
            tuple(position)
            for position in np.argwhere(grid == structure)
        }
        centers = _geometric_centers(
            structural,
            height,
            width,
        )

        if len(centers) < 1:
            continue
        if len(structural) != 4 * len(centers):
            continue
        if _arm_positions(centers) != structural:
            continue
        if centers & structural:
            continue

        allowed_positions = centers | structural
        unrelated_nonbackground = any(
            (r, c) not in allowed_positions
            and int(grid[r, c]) != background
            for r in range(height)
            for c in range(width)
        )
        if unrelated_nonbackground:
            continue

        if any(
            int(grid[r, c]) == structure
            for r, c in centers
        ):
            continue

        described_centers = tuple(
            sorted(
                (r, c, int(grid[r, c]))
                for r, c in centers
            )
        )
        role_candidates.append(
            (structure, described_centers)
        )

    assert len(role_candidates) == 1, (
        "input must have one unambiguous structural color"
    )

    structure, centers = role_candidates[0]
    assert counts[background] > counts[structure], (
        "background must be identifiable as the dominant input color"
    )

    return background, structure, centers


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30
    assert 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((grid >= 0) & (grid <= 9))


_SOURCE_INPUTS = {
    ((18, 18), ((2, 3, 0), (7, 13, 5), (11, 7, 6), (15, 14, 4))),
    ((10, 10), ((2, 2, 5), (3, 8, 6), (6, 5, 8))),
    ((13, 13), ((3, 3, 2), (9, 8, 0))),
    ((30, 30), ((3, 5, 2), (4, 20, 6), (18, 13, 0), (21, 22, 8))),
    ((19, 19), ((3, 6, 8), (4, 13, 4), (10, 8, 0),
                (12, 14, 1), (14, 3, 2))),
}


def _source_copy(grid, background, structure, centers):
    return (
        background == 0
        and structure == 3
        and (grid.shape, tuple(sorted(centers))) in _SOURCE_INPUTS
    )


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    seen_inputs = set()
    seen_pairs = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            _validate_grid(grid)
            _validate_grid(output)
            assert grid.shape == output.shape
            assert min(grid.shape) >= 10
            input_key = _grid_key(grid)
            pair_key = (input_key, _grid_key(output))
            assert input_key not in seen_inputs
            assert pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

            background, structure, centers = _infer_crosses(grid)
            assert not _source_copy(grid, background, structure, centers)
            assert len({r for r, _, _ in centers}) == len(centers)
            assert len({c for _, c, _ in centers}) == len(centers)
            for i, center in enumerate(centers):
                for other in centers[i + 1:]:
                    assert _footprint(center[:2]).isdisjoint(
                        _footprint(other[:2])
                    )
            expected, stats = _analyze_pair(
                grid, background, structure, centers
            )
            assert np.array_equal(output, expected)
            assert stats["changed"] > 0
            assert np.array_equal(
                output[grid == structure],
                grid[grid == structure],
            )
            assert all(
                int(output[r, c]) == signal for r, c, signal in centers
            )
            background_centers = sum(
                signal == background for _, _, signal in centers
            )
            if split == "train":
                assert 1 <= len(centers) <= 4
                assert len(centers) > 1 or background_centers == 0
            else:
                assert 2 <= len(centers) <= 7
            analyses[split].append({
                "background": background,
                "structure": structure,
                "centers": centers,
                "stats": stats,
                "background_centers": background_centers,
            })

    training = analyses["train"]
    assert any(
        item["stats"]["colored_precedence"] > 0 for item in training
    )
    assert any(
        item["stats"]["background_precedence"] > 0 for item in training
    )
    assert any(
        item["stats"]["blocked_colored_witness"] > 0 for item in training
    )
    assert any(item["background_centers"] > 0 for item in training)
    assert len({
        signal
        for item in training
        for _, _, signal in item["centers"]
        if signal != item["background"]
    }) >= 2
    if any(item["stats"]["double_clip"] for item in analyses["test"]):
        assert any(item["stats"]["double_clip"] for item in training)
    assert len({
        len(item["centers"])
        for split in ("train", "test")
        for item in analyses[split]
    }) > 1


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    known_colors = set()
    training_examples = []
    training_has_double_clip = False

    train_kinds = ["colored_block_demo", "background_demo"]
    extra_kinds = (
        "simple", "double_clip", "mixed", "repeat",
        "multi_background", "colored_block_demo",
    )
    for _ in range(train_count - 2):
        train_kinds.append(extra_kinds[int(rng.integers(len(extra_kinds)))])
    rng.shuffle(train_kinds)

    test_kinds = ("familiar", "repeat", "multi_background", "extrapolate", "mixed")
    if "double_clip" in train_kinds:
        test_kinds += ("double_clip",)

    for split, kinds in (
        ("train", train_kinds),
        ("test", [
            test_kinds[int(rng.integers(len(test_kinds)))]
            for _ in range(test_count)
        ]),
    ):
        for index, kind in enumerate(kinds):
            if (
                split == "test" and index == 0
                and len({len(example[2]) for example in training_examples}) == 1
            ):
                kind = "extrapolate"
            familiar_pair = None
            if split == "test" and kind == "familiar":
                candidates = [
                    example for example in training_examples
                    if 2 <= len(example[2]) <= 5
                ]
                familiar_pair = candidates[int(rng.integers(len(candidates)))]
            for _ in range(60):
                pair_rng = np.random.default_rng(
                    int(rng.integers(0, np.iinfo(np.int64).max))
                )
                grid, output = _generate_pair(
                    pair_rng,
                    "source_train" if split == "train" else "target_test",
                    parameters,
                    kind,
                    known_colors,
                    changed_roles=not (split == "train" and index == 0),
                    familiar_pair=familiar_pair,
                )
                key = _grid_key(grid)
                background, structure, centers = _infer_crosses(grid)
                if key in used_inputs or _source_copy(
                    grid, background, structure, centers
                ):
                    continue
                if split == "test":
                    _, stats = _analyze_pair(
                        grid, background, structure, centers
                    )
                    if stats["double_clip"] and not training_has_double_clip:
                        continue
                used_inputs.add(key)
                puzzle[split].append({"input": grid, "output": output})
                if split == "train":
                    training_has_double_clip |= _analyze_pair(
                        grid, background, structure, centers
                    )[1]["double_clip"]
                    training_examples.append(
                        (background, structure, tuple(
                            signal for _, _, signal in centers
                        ))
                    )
                    known_colors.update(
                        signal
                        for _, _, signal in centers
                        if signal != background
                    )
                break
            else:
                raise RuntimeError("failed to generate a distinct pair")

    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    if set(first) != set(second):
        return False
    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for first_pair, second_pair in zip(first[split], second[split]):
            if not np.array_equal(first_pair["input"], second_pair["input"]):
                return False
            if not np.array_equal(first_pair["output"], second_pair["output"]):
                return False
    return True


def _independent_projection(grid, background, structure, centers):
    arms = {tuple(position) for position in np.argwhere(grid == structure)}
    protected = arms | {(r, c) for r, c, _ in centers}
    result = grid.copy()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if (r, c) in protected:
                continue
            horizontal = None
            vertical = None
            for source_r, source_c, signal in centers:
                if r == source_r and abs(c - source_c) >= 2:
                    segment = (
                        range(source_c + 2, c + 1)
                        if c > source_c else range(c, source_c - 1)
                    )
                    if not any((r, x) in arms for x in segment):
                        horizontal = signal
                if c == source_c and abs(r - source_r) >= 2:
                    segment = (
                        range(source_r + 2, r + 1)
                        if r > source_r else range(r, source_r - 1)
                    )
                    if not any((x, c) in arms for x in segment):
                        vertical = signal
            if horizontal is not None:
                result[r, c] = horizontal
            if vertical is not None:
                result[r, c] = vertical
    return result


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, layouts, palettes = set(), set(), set()
    train_object_counts, test_object_counts = set(), set()
    squares = {"train": 0, "test": 0}
    small_tests = large_trains = 0
    colored_only_tests = multiple_background_tests = 0
    repeated_test_signals = overlap_tests = extrapolated_tests = 0
    varied_roles = stable_roles = double_clips = simple_mixed = 0
    combined_cases = familiar_tests = double_background = one_sided = 0
    multiple_background_train = 0
    count_varied_puzzles = background_varied_puzzles = 0
    literal_role_reassignments = new_test_signal_colors = 0

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_roles = set()
        train_colors = set()
        train_counts_here = set()
        train_palettes_here = set()
        roles_by_literal = {color: set() for color in range(10)}
        count_values = set()
        background_values = set()
        for pair in puzzle["train"]:
            grid = pair["input"]
            background, structure, centers = _infer_crosses(grid)
            train_roles.add((background, structure))
            train_counts_here.add(len(centers))
            train_palettes_here.add((
                background, structure,
                frozenset(signal for _, _, signal in centers if signal != background),
            ))
            train_colors.update(
                signal for _, _, signal in centers if signal != background
            )
            roles_by_literal[background].add("background")
            roles_by_literal[structure].add("structure")
            for _, _, signal in centers:
                if signal != background:
                    roles_by_literal[signal].add("signal")
            if len(centers) == 2 and sum(
                signal == background for _, _, signal in centers
            ) == 1:
                simple_mixed += 1
            multiple_background_train += sum(
                signal == background for _, _, signal in centers
            ) >= 2
        if len(train_roles) > 1:
            varied_roles += 1
        else:
            stable_roles += 1
        literal_role_reassignments += any(
            len(roles) > 1 for roles in roles_by_literal.values()
        )

        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                background, structure, centers = _infer_crosses(grid)
                expected, stats = _analyze_pair(
                    grid, background, structure, centers
                )
                assert np.array_equal(expected, pair["output"])
                assert np.array_equal(
                    _independent_projection(
                        grid, background, structure, centers
                    ),
                    pair["output"],
                )
                assert not _source_copy(
                    grid, background, structure, centers
                )
                shapes.add(grid.shape)
                count_values.add(len(centers))
                background_values.add(sum(
                    signal == background for _, _, signal in centers
                ))
                layouts.add((grid.shape, tuple((r, c) for r, c, _ in centers)))
                palettes.add((background, structure))
                squares[split] += grid.shape[0] == grid.shape[1]
                double_clips += bool(stats["double_clip"])
                one_sided += stats["one_sided_stop"] > 0
                double_background += bool(stats["double_clip"]) and any(
                    signal == background for _, _, signal in centers
                )
                if split == "train":
                    train_object_counts.add(len(centers))
                    large_trains += max(grid.shape) >= 27
                else:
                    test_object_counts.add(len(centers))
                    small_tests += max(grid.shape) <= 14
                    signals = [v for _, _, v in centers]
                    background_count = signals.count(background)
                    colored_only_tests += background_count == 0
                    multiple_background_tests += background_count >= 2
                    colored = [v for v in signals if v != background]
                    repeated_test_signals += len(set(colored)) < len(colored)
                    new_test_signal_colors += bool(
                        set(colored) - train_colors
                    )
                    overlap_tests += len(centers) in train_counts_here
                    familiar_tests += (
                        len(centers) in train_counts_here
                        and (
                            background, structure, frozenset(colored)
                        ) in train_palettes_here
                    )
                    extrapolated_tests += len(centers) > max(train_counts_here)
                    combined_cases += (
                        background_count >= 1
                        and stats["blocked_colored_witness"] >= 1
                        and stats["different_precedence"] >= 1
                    )
        count_varied_puzzles += len(count_values) > 1
        background_varied_puzzles += len(background_values) > 1

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 60 and len(layouts) >= 200
    assert len(palettes) >= 20
    assert {1, 2, 3, 4}.issubset(train_object_counts)
    assert {2, 3, 4, 5, 6, 7}.issubset(
        train_object_counts | test_object_counts
    )
    assert squares["train"] >= 5 and squares["test"] >= 2
    assert small_tests >= 1 and large_trains >= 5
    assert colored_only_tests >= 5
    assert multiple_background_tests >= 1
    assert repeated_test_signals >= 5
    assert overlap_tests >= 5 and extrapolated_tests >= 5
    assert varied_roles >= 5 and stable_roles >= 5
    assert count_varied_puzzles == 42
    assert background_varied_puzzles == 42
    assert literal_role_reassignments >= 5
    assert new_test_signal_colors >= 5
    assert double_clips >= 3 and simple_mixed >= 1
    assert one_sided >= 20 and multiple_background_train >= 1
    assert familiar_tests >= 5 and double_background >= 1
    assert combined_cases >= 5

    first = generate_puzzle(314159, 6, 3)
    assert _puzzles_equal(first, generate_puzzle(314159, 6, 3))
    assert not _puzzles_equal(first, generate_puzzle(314160, 6, 3))

    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(
            7000 + 10 * train_count + test_count,
            train_count,
            test_count,
        )
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)

    for index, arguments in enumerate((
        {"num_train": None, "num_test": 1},
        {"num_train": None, "num_test": 4},
        {"num_train": 2, "num_test": None},
        {"num_train": 10, "num_test": None},
    )):
        puzzle = generate_puzzle(8000 + index, **arguments)
        validate_generated_puzzle(puzzle)
        for argument, split in (("num_train", "train"), ("num_test", "test")):
            if arguments[argument] is not None:
                assert len(puzzle[split]) == arguments[argument]

    for arguments in (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": True, "num_test": 1},
    ):
        try:
            generate_puzzle(999, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
