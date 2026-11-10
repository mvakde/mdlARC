import numpy as np


_DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


def _components(mask):
    h, w = mask.shape
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []

    while unseen:
        start = unseen.pop()
        stack = [start]
        component = {start}

        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (r + dr, c + dc)
                if (
                    0 <= neighbor[0] < h
                    and 0 <= neighbor[1] < w
                    and neighbor in unseen
                ):
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)

        result.append(component)

    return result


def _is_rectangle(component):
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
    }
    return component == expected


def _infer_roles(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 3
    assert 0 not in values

    background = int(values[int(np.argmax(counts))])
    other_colors = [int(v) for v in values if int(v) != background]

    rectangular_colors = []
    for color in other_colors:
        comps = _components(grid == color)
        if comps and all(_is_rectangle(comp) for comp in comps):
            rectangular_colors.append(color)

    assert len(rectangular_colors) == 1
    token = rectangular_colors[0]
    structure = (
        other_colors[0]
        if other_colors[1] == token
        else other_colors[1]
    )
    return background, structure, token


def _parse_receiver_component(component):
    parses = []

    for direction, (dr, dc) in _DIRS.items():
        outward_r, outward_c = -dr, -dc
        perpendicular_r, perpendicular_c = -outward_c, outward_r

        raw_k = [
            r * outward_r + c * outward_c
            for r, c in component
        ]
        raw_q = [
            r * perpendicular_r + c * perpendicular_c
            for r, c in component
        ]
        k_min = min(raw_k)
        q_min = min(raw_q)

        local = {
            (
                r * outward_r + c * outward_c - k_min,
                r * perpendicular_r + c * perpendicular_c - q_min,
            )
            for r, c in component
        }

        for span in range(1, 4):
            for clearance in range(1, 3):
                width = span + 2 * clearance + 4
                center0 = clearance + 2

                for guard_len in range(4, 7):
                    for prong_len in range(1, guard_len - 1):
                        expected = {(0, q) for q in range(width)}
                        expected |= {
                            (k, q)
                            for k in range(guard_len)
                            for q in (1, width - 2)
                        }
                        expected |= {
                            (k, q)
                            for k in range(prong_len)
                            for q in range(center0, center0 + span)
                        }

                        if local == expected:
                            parses.append({
                                "direction": direction,
                                "out": (outward_r, outward_c),
                                "perp": (
                                    perpendicular_r,
                                    perpendicular_c,
                                ),
                                "k_min": k_min,
                                "q_min": q_min,
                                "span": span,
                                "clearance": clearance,
                                "guard_len": guard_len,
                                "prong_len": prong_len,
                                "center0": center0,
                                "cells": component,
                            })

    assert len(parses) == 1
    return parses[0]


def _global_cell(receiver, k, q):
    outward_r, outward_c = receiver["out"]
    perpendicular_r, perpendicular_c = receiver["perp"]

    raw_k = receiver["k_min"] + k
    raw_q = receiver["q_min"] + q

    return (
        raw_k * outward_r + raw_q * perpendicular_r,
        raw_k * outward_c + raw_q * perpendicular_c,
    )


def _candidate_for(receiver, token_component, grid, background):
    outward_r, outward_c = receiver["out"]
    perpendicular_r, perpendicular_c = receiver["perp"]

    local = {
        (
            r * outward_r + c * outward_c - receiver["k_min"],
            r * perpendicular_r
            + c * perpendicular_c
            - receiver["q_min"],
        )
        for r, c in token_component
    }

    ks = sorted({k for k, _ in local})
    qs = sorted({q for _, q in local})
    required_qs = list(range(
        receiver["center0"],
        receiver["center0"] + receiver["span"],
    ))

    if not ks or qs != required_qs:
        return None

    if ks != list(range(ks[0], ks[-1] + 1)):
        return None

    if local != {(k, q) for k in ks for q in qs}:
        return None

    length = len(ks)
    if not 1 <= length <= 3:
        return None

    if ks[0] < receiver["guard_len"] + 1:
        return None

    if (
        receiver["prong_len"] + length - 1
        >= receiver["guard_len"]
    ):
        return None

    h, w = grid.shape
    for k in range(receiver["prong_len"], ks[-1] + 1):
        for q in qs:
            r, c = _global_cell(receiver, k, q)

            if not (0 <= r < h and 0 <= c < w):
                return None

            if (r, c) in token_component:
                continue

            if int(grid[r, c]) != background:
                return None

    return {
        "cells": token_component,
        "k0": ks[0],
        "k1": ks[-1],
        "length": length,
    }


def _analyze_grid(grid):
    background, structure, token = _infer_roles(grid)

    receiver_components = _components(grid == structure)
    receivers = [
        _parse_receiver_component(component)
        for component in receiver_components
    ]

    token_components = _components(grid == token)
    assert token_components
    assert all(_is_rectangle(comp) for comp in token_components)

    matches = []
    used_tokens = set()

    for receiver_index, receiver in enumerate(receivers):
        candidates = []

        for token_index, component in enumerate(token_components):
            candidate = _candidate_for(
                receiver,
                component,
                grid,
                background,
            )
            if candidate is not None:
                candidates.append((token_index, candidate))

        assert len(candidates) <= 1

        if candidates:
            token_index, candidate = candidates[0]
            assert token_index not in used_tokens
            used_tokens.add(token_index)

            matches.append({
                "receiver_index": receiver_index,
                "token_index": token_index,
                "receiver": receiver,
                "candidate": candidate,
            })

    return {
        "background": background,
        "structure": structure,
        "token": token,
        "receivers": receivers,
        "token_components": token_components,
        "matches": matches,
        "unmatched_token_count": (
            len(token_components) - len(used_tokens)
        ),
    }


def _solve_grid(grid):
    analysis = _analyze_grid(grid)
    background = analysis["background"]
    token = analysis["token"]

    output = grid.copy()
    output[grid == token] = background

    for match in analysis["matches"]:
        receiver = match["receiver"]
        candidate = match["candidate"]

        q0 = receiver["center0"]
        q1 = q0 + receiver["span"]
        token_length = candidate["length"]

        trail_start = receiver["prong_len"] + token_length

        for k in range(trail_start, candidate["k1"] + 1):
            for q in range(q0, q1):
                r, c = _global_cell(receiver, k, q)
                assert int(output[r, c]) == background
                output[r, c] = 0

        for k in range(
            receiver["prong_len"],
            receiver["prong_len"] + token_length,
        ):
            for q in range(q0, q1):
                r, c = _global_cell(receiver, k, q)
                assert int(output[r, c]) == background
                output[r, c] = token

    return output, analysis


def _expand(cells, h, w):
    expanded = set()

    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))

    return expanded


def _receiver_template(
    direction,
    span,
    token_length,
    active,
    clearance,
    guard_len,
    prong_len,
    travel_gap,
):
    dr, dc = _DIRS[direction]
    outward_r, outward_c = -dr, -dc
    perpendicular_r, perpendicular_c = -outward_c, outward_r

    width = span + 2 * clearance + 4
    center0 = clearance + 2

    structure_local = {(0, q) for q in range(width)}
    structure_local |= {
        (k, q)
        for k in range(guard_len)
        for q in (1, width - 2)
    }
    structure_local |= {
        (k, q)
        for k in range(prong_len)
        for q in range(center0, center0 + span)
    }

    original_local = set()
    route_local = set()

    if active:
        original_start = guard_len + travel_gap

        original_local = {
            (k, q)
            for k in range(
                original_start,
                original_start + token_length,
            )
            for q in range(center0, center0 + span)
        }

        route_local = {
            (k, q)
            for k in range(
                prong_len,
                original_start + token_length,
            )
            for q in range(center0, center0 + span)
        }

    def convert(local_cells):
        return {
            (
                k * outward_r + q * perpendicular_r,
                k * outward_c + q * perpendicular_c,
            )
            for k, q in local_cells
        }

    structure_raw = convert(structure_local)
    original_raw = convert(original_local)
    route_raw = convert(route_local)
    reserve_raw = structure_raw | route_raw

    min_r = min(r for r, _ in reserve_raw)
    min_c = min(c for _, c in reserve_raw)

    def normalize(cells):
        return {
            (r - min_r, c - min_c)
            for r, c in cells
        }

    structure_cells = normalize(structure_raw)
    token_cells = normalize(original_raw)
    reserve_cells = normalize(reserve_raw)

    return {
        "direction": direction,
        "span": span,
        "token_length": token_length,
        "active": active,
        "clearance": clearance,
        "guard_len": guard_len,
        "prong_len": prong_len,
        "structure": structure_cells,
        "token": token_cells,
        "reserve": reserve_cells,
        "height": max(r for r, _ in reserve_cells) + 1,
        "width": max(c for _, c in reserve_cells) + 1,
    }


def _train_specs(index, rng):
    familiar = ("L", "U", "D")

    if index == 0:
        return [
            {
                "direction": "L",
                "span": 1,
                "token_length": 1,
                "active": True,
            },
            {
                "direction": "U",
                "span": 2,
                "token_length": 2,
                "active": True,
            },
        ]

    if index == 1:
        return [
            {
                "direction": "D",
                "span": 3,
                "token_length": int(rng.integers(1, 3)),
                "active": True,
            },
            {
                "direction": familiar[
                    int(rng.integers(0, len(familiar)))
                ],
                "span": int(rng.integers(1, 4)),
                "token_length": int(rng.integers(1, 3)),
                "active": False,
            },
        ]

    specs = []
    for _ in range(2):
        specs.append({
            "direction": familiar[
                int(rng.integers(0, len(familiar)))
            ],
            "span": int(rng.integers(1, 4)),
            "token_length": int(rng.integers(1, 3)),
            "active": bool(rng.integers(0, 2)),
        })

    if not any(spec["active"] for spec in specs):
        specs[int(rng.integers(0, 2))]["active"] = True

    return specs


def _test_specs(rng):
    return [
        {
            "direction": "R",
            "span": 3,
            "token_length": int(rng.integers(1, 3)),
            "active": True,
        },
        {
            "direction": "U",
            "span": int(rng.integers(1, 4)),
            "token_length": int(rng.integers(1, 3)),
            "active": True,
        },
        {
            "direction": "D",
            "span": int(rng.integers(1, 4)),
            "token_length": int(rng.integers(1, 3)),
            "active": True,
        },
    ]


def _shift(cells, row_shift, col_shift):
    return {
        (r + row_shift, c + col_shift)
        for r, c in cells
    }


def _generate_pair(rng, regime, index, puzzle_parameters):
    if regime == "source_train":
        h, w = 19, 22
        specs = _train_specs(index, rng)
        extra_distractors = int(rng.integers(1, 3))
    else:
        h = int(rng.integers(25, 31))
        w = int(rng.integers(26, 31))
        specs = _test_specs(rng)
        extra_distractors = int(rng.integers(3, 6))

    background = puzzle_parameters["background"]
    structure_color = puzzle_parameters["structure"]
    token_color = puzzle_parameters["token"]

    corners = (
        (0, 0),
        (0, w - 1),
        (h - 1, 0),
        (h - 1, w - 1),
    )

    for _layout_attempt in range(500):
        corner = corners[int(rng.integers(0, 4))]
        token_objects = [{corner}]
        blocked = _expand({corner}, h, w)

        geometries = []
        geometry_failed = False

        for spec in specs:
            clearance = int(rng.integers(1, 3))
            guard_limit = 6 if regime == "source_train" else 7
            guard_len = int(rng.integers(4, guard_limit))
            token_length = spec["token_length"]

            max_prong = guard_len - token_length
            if max_prong < 1:
                geometry_failed = True
                break

            prong_len = int(rng.integers(1, max_prong + 1))
            travel_gap = int(rng.integers(1, 5))

            geometry = _receiver_template(
                spec["direction"],
                spec["span"],
                token_length,
                spec["active"],
                clearance,
                guard_len,
                prong_len,
                travel_gap,
            )

            if (
                geometry["height"] > h
                or geometry["width"] > w
            ):
                geometry_failed = True
                break

            geometries.append(geometry)

        if geometry_failed:
            continue

        geometries.sort(
            key=lambda item: item["height"] * item["width"],
            reverse=True,
        )

        placed_structures = []
        expected_matches = []
        placement_failed = False

        for geometry in geometries:
            valid_positions = []

            for row_shift in range(
                h - geometry["height"] + 1
            ):
                for col_shift in range(
                    w - geometry["width"] + 1
                ):
                    reserve = _shift(
                        geometry["reserve"],
                        row_shift,
                        col_shift,
                    )

                    if reserve.isdisjoint(blocked):
                        valid_positions.append(
                            (row_shift, col_shift)
                        )

            if not valid_positions:
                placement_failed = True
                break

            row_shift, col_shift = valid_positions[
                int(rng.integers(0, len(valid_positions)))
            ]

            reserve = _shift(
                geometry["reserve"],
                row_shift,
                col_shift,
            )
            structure_cells = _shift(
                geometry["structure"],
                row_shift,
                col_shift,
            )
            token_cells = _shift(
                geometry["token"],
                row_shift,
                col_shift,
            )

            placed_structures.append(structure_cells)

            if geometry["active"]:
                token_objects.append(token_cells)
                expected_matches.append((
                    geometry["direction"],
                    geometry["span"],
                    geometry["token_length"],
                ))

            blocked.update(_expand(reserve, h, w))

        if placement_failed:
            continue

        distractor_failed = False
        sizes = (
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (1, 3),
            (3, 1),
        )

        for _ in range(extra_distractors):
            dh, dw = sizes[
                int(rng.integers(0, len(sizes)))
            ]
            valid_positions = []

            for r0 in range(h - dh + 1):
                for c0 in range(w - dw + 1):
                    cells = {
                        (r, c)
                        for r in range(r0, r0 + dh)
                        for c in range(c0, c0 + dw)
                    }

                    if cells.isdisjoint(blocked):
                        valid_positions.append((r0, c0))

            if not valid_positions:
                distractor_failed = True
                break

            r0, c0 = valid_positions[
                int(rng.integers(0, len(valid_positions)))
            ]
            cells = {
                (r, c)
                for r in range(r0, r0 + dh)
                for c in range(c0, c0 + dw)
            }

            token_objects.append(cells)
            blocked.update(_expand(cells, h, w))

        if distractor_failed:
            continue

        grid = np.full(
            (h, w),
            background,
            dtype=np.int8,
        )

        for component in placed_structures:
            for r, c in component:
                grid[r, c] = structure_color

        for component in token_objects:
            for r, c in component:
                grid[r, c] = token_color

        try:
            output, analysis = _solve_grid(grid)
        except AssertionError:
            continue

        actual_matches = sorted(
            (
                match["receiver"]["direction"],
                match["receiver"]["span"],
                match["candidate"]["length"],
            )
            for match in analysis["matches"]
        )

        if actual_matches != sorted(expected_matches):
            continue

        if len(analysis["receivers"]) != len(specs):
            continue

        if analysis["unmatched_token_count"] < 1:
            continue

        if int(output[corner]) != background:
            continue

        return grid, output

    raise RuntimeError(
        "could not construct an unambiguous pair"
    )


def _valid_count(value, low, high):
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and low <= int(value) <= high
    )


def _resolve_counts(seed_value, num_train, num_test):
    if (
        num_train is not None
        and not _valid_count(num_train, 2, 10)
    ):
        raise ValueError(
            "num_train must be between 2 and 10"
        )

    if (
        num_test is not None
        and not _valid_count(num_test, 1, 4)
    ):
        raise ValueError(
            "num_test must be between 1 and 4"
        )

    if num_train is not None:
        num_train = int(num_train)

    if num_test is not None:
        num_test = int(num_test)

    if num_train is None and num_test is None:
        num_test = 1 + seed_value % 4
        minimum_train = max(2, num_test + 1)
        num_train = minimum_train + (
            (seed_value // 4) % (11 - minimum_train)
        )

    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = minimum_train + (
            seed_value % (11 - minimum_train)
        )

    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = 1 + seed_value % maximum_test

    if num_train <= num_test:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return num_train, num_test


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if (
        not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
    ):
        raise ValueError("seed must be an integer")

    seed_value = int(seed) % (1 << 64)
    num_train, num_test = _resolve_counts(
        seed_value,
        num_train,
        num_test,
    )
    rng = np.random.default_rng(seed_value)

    palette = rng.choice(
        np.arange(1, 10),
        size=3,
        replace=False,
    )
    puzzle_parameters = {
        "background": int(palette[0]),
        "structure": int(palette[1]),
        "token": int(palette[2]),
    }

    child_seeds = rng.integers(
        0,
        (1 << 63) - 1,
        size=num_train + num_test,
        dtype=np.int64,
    )

    puzzle = {
        "train": [],
        "test": [],
    }
    seen_inputs = set()

    regimes = (
        ("source_train", num_train, 0),
        ("target_test", num_test, num_train),
    )

    for regime, count, offset in regimes:
        destination = (
            "train"
            if regime == "source_train"
            else "test"
        )

        for index in range(count):
            pair_rng = np.random.default_rng(
                int(child_seeds[offset + index])
            )

            for _duplicate_attempt in range(20):
                input_grid, output_grid = _generate_pair(
                    pair_rng,
                    regime,
                    index,
                    puzzle_parameters,
                )
                signature = (
                    input_grid.shape,
                    input_grid.tobytes(),
                )

                if signature not in seen_inputs:
                    seen_inputs.add(signature)
                    puzzle[destination].append({
                        "input": input_grid,
                        "output": output_grid,
                    })
                    break
            else:
                raise RuntimeError(
                    "could not generate a distinct pair"
                )

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    input_signatures = set()
    pair_signatures = set()
    palette = None
    train_analyses = []
    test_analyses = []

    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict)
            assert set(pair) == {"input", "output"}

            input_grid = pair["input"]
            output_grid = pair["output"]

            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.issubdtype(
                    grid.dtype,
                    np.integer,
                )
                assert np.all((0 <= grid) & (grid <= 9))

            assert input_grid.shape == output_grid.shape

            input_signature = (
                input_grid.shape,
                input_grid.tobytes(),
            )
            pair_signature = (
                input_grid.shape,
                input_grid.tobytes(),
                output_grid.shape,
                output_grid.tobytes(),
            )

            assert input_signature not in input_signatures
            assert pair_signature not in pair_signatures
            input_signatures.add(input_signature)
            pair_signatures.add(pair_signature)

            expected_output, analysis = _solve_grid(
                input_grid
            )
            assert np.array_equal(
                output_grid,
                expected_output,
            )

            roles = (
                analysis["background"],
                analysis["structure"],
                analysis["token"],
            )
            if palette is None:
                palette = roles
            assert roles == palette

            assert 0 not in np.unique(input_grid)
            assert set(map(
                int,
                np.unique(output_grid),
            )) == set(roles) | {0}

            assert len(analysis["matches"]) >= 1
            assert analysis["unmatched_token_count"] >= 1
            assert not np.array_equal(
                input_grid,
                output_grid,
            )

            background, _, token = roles
            corners = (
                (0, 0),
                (0, input_grid.shape[1] - 1),
                (input_grid.shape[0] - 1, 0),
                (
                    input_grid.shape[0] - 1,
                    input_grid.shape[1] - 1,
                ),
            )

            # Every supplied source-task grid has background
            # in all four corners. This required distractor
            # proves that no generated pair is an exact copy.
            assert any(
                int(input_grid[r, c]) == token
                and int(output_grid[r, c]) == background
                for r, c in corners
            )

            if split == "train":
                assert input_grid.shape == (19, 22)
                assert len(analysis["receivers"]) == 2
                assert {
                    receiver["direction"]
                    for receiver in analysis["receivers"]
                } <= {"L", "U", "D"}
                train_analyses.append(analysis)

            else:
                assert 25 <= input_grid.shape[0] <= 30
                assert 26 <= input_grid.shape[1] <= 30
                assert len(analysis["receivers"]) == 3
                assert {
                    receiver["direction"]
                    for receiver in analysis["receivers"]
                } == {"R", "U", "D"}
                assert len(analysis["matches"]) == 3
                assert (
                    analysis["unmatched_token_count"] >= 4
                )

                rightward = [
                    receiver
                    for receiver in analysis["receivers"]
                    if receiver["direction"] == "R"
                ]
                assert len(rightward) == 1
                assert rightward[0]["span"] == 3

                test_analyses.append(analysis)

    train_directions = {
        receiver["direction"]
        for analysis in train_analyses
        for receiver in analysis["receivers"]
    }
    assert train_directions == {"L", "U", "D"}

    active_train_matches = [
        match
        for analysis in train_analyses
        for match in analysis["matches"]
    ]

    assert {
        match["receiver"]["span"]
        for match in active_train_matches
    } == {1, 2, 3}

    assert {
        match["candidate"]["length"]
        for match in active_train_matches
    } == {1, 2}

    assert any(
        len(analysis["matches"]) == 2
        for analysis in train_analyses
    )
    assert any(
        len(analysis["matches"]) < 2
        for analysis in train_analyses
    )

    # Together these establish recoverability of the
    # puzzle-level palette and transformation convention.
    assert palette is not None
    assert len(set(palette)) == 3
    assert 0 not in palette
    assert all(
        len(analysis["matches"]) >= 1
        for analysis in train_analyses
    )
    assert all(
        analysis["unmatched_token_count"] >= 1
        for analysis in train_analyses
    )
    assert test_analyses


def _puzzles_equal(first, second):
    if set(first) != set(second):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for pair_a, pair_b in zip(
            first[split],
            second[split],
        ):
            if set(pair_a) != set(pair_b):
                return False

            if not np.array_equal(
                pair_a["input"],
                pair_b["input"],
            ):
                return False

            if not np.array_equal(
                pair_a["output"],
                pair_b["output"],
            ):
                return False

    return True


def self_test() -> None:
    train_counts = set()
    test_counts = set()
    palettes = set()
    shapes = set()
    puzzle_signatures = set()
    layout_signatures = set()
    geometry_signatures = set()

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)

        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))

        puzzle_signatures.add(tuple(
            (
                pair["input"].shape,
                pair["input"].tobytes(),
            )
            for split in ("train", "test")
            for pair in puzzle[split]
        ))

        first_analysis = _analyze_grid(
            puzzle["train"][0]["input"]
        )
        palettes.add((
            first_analysis["background"],
            first_analysis["structure"],
            first_analysis["token"],
        ))

        for split in ("train", "test"):
            for pair in puzzle[split]:
                input_grid = pair["input"]
                analysis = _analyze_grid(input_grid)
                shapes.add(input_grid.shape)

                receivers = tuple(sorted(
                    (
                        receiver["direction"],
                        receiver["span"],
                        receiver["clearance"],
                        receiver["guard_len"],
                        receiver["prong_len"],
                        tuple(sorted(receiver["cells"])),
                    )
                    for receiver in analysis["receivers"]
                ))
                layout_signatures.add((
                    input_grid.shape,
                    receivers,
                ))

                geometry_signatures.add(tuple(sorted(
                    (
                        receiver["direction"],
                        receiver["span"],
                        receiver["clearance"],
                        receiver["guard_len"],
                        receiver["prong_len"],
                    )
                    for receiver in analysis["receivers"]
                )))

    assert len(train_counts) >= 4
    assert len(test_counts) >= 3
    assert len(puzzle_signatures) >= 40
    assert len(palettes) >= 20
    assert len(shapes) >= 10
    assert len(layout_signatures) >= 250
    assert len(geometry_signatures) >= 40

    deterministic_a = generate_puzzle(
        123456,
        6,
        3,
    )
    deterministic_b = generate_puzzle(
        123456,
        6,
        3,
    )
    assert _puzzles_equal(
        deterministic_a,
        deterministic_b,
    )

    assert not _puzzles_equal(
        generate_puzzle(123456, 5, 2),
        generate_puzzle(123457, 5, 2),
    )

    for num_train, num_test in (
        (2, 1),
        (10, 1),
        (10, 4),
    ):
        puzzle = generate_puzzle(
            9000 + num_train * 10 + num_test,
            num_train,
            num_test,
        )
        assert len(puzzle["train"]) == num_train
        assert len(puzzle["test"]) == num_test
        validate_generated_puzzle(puzzle)

    partial_cases = (
        (None, 1),
        (None, 4),
        (2, None),
        (10, None),
    )

    for index, (num_train, num_test) in enumerate(
        partial_cases
    ):
        puzzle = generate_puzzle(
            12000 + index,
            num_train,
            num_test,
        )

        if num_train is not None:
            assert len(puzzle["train"]) == num_train

        if num_test is not None:
            assert len(puzzle["test"]) == num_test

        validate_generated_puzzle(puzzle)

    illegal_cases = (
        (1, None),
        (11, None),
        (None, 0),
        (None, 5),
        (2, 2),
        (3, 3),
        (4, 4),
        (2, 4),
    )

    for num_train, num_test in illegal_cases:
        try:
            generate_puzzle(
                7,
                num_train,
                num_test,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "illegal counts did not raise ValueError"
            )
