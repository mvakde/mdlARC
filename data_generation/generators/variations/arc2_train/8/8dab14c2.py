"""Generate completed one-cell slice shifts on a filled L."""

import hashlib

import numpy as np


_ORIENTATIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_SOURCE_FINGERPRINTS = {
    "3e7c3f31d891bd85eff3e431147c23acbef3cb845fcb97cd159ed152c9a114be",
    "311b51d5c87334295b8c5d9da9dbbf710d777ada0f9932b9bd3bc28993cc1791",
    "b0b5a1445d44aa399d591c1cbb2faa77c8e1b34f919c98e58b8f5b883d46aa",
    "ebb243606193c9e4c52128dc24f1ae695389e624ee0e4f9054afaedde8627059",
    "d0ca1fd7750c2f794bd4acf902ecbf7bb0c085d054804cbb93c516ee828449fa",
    "97e741ff43edf1f7293dc393a9edc11ec6732f98b53dbb29830c6e94af300acd",
    "75a1f614d3cfccc97144ab22861609a0031bcdeae8828b176d1db2b3165f896c",
    "7a8588a9ad1b4086290d2440627d01d5fc39f7899e8f503a0b8e050346b513a0",
}


def _counts(rng, num_train, num_test):
    for value, low, high, name in (
        (num_train, 2, 10, "num_train"),
        (num_test, 1, 4, "num_test"),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high
        ):
            raise ValueError(name)
    if num_test is None and num_train is None:
        num_test = int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, int(num_test) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _info(height, width, rlo, rhi, clo, chi, orientation, hh, vw):
    dr, dc = _ORIENTATIONS[orientation]
    hr0 = rlo if dr == 1 else rhi - hh + 1
    hr1 = hr0 + hh - 1
    vc0 = clo if dc == 1 else chi - vw + 1
    vc1 = vc0 + vw - 1
    mask = np.zeros((height, width), dtype=bool)
    mask[rlo:rhi + 1, vc0:vc1 + 1] = True
    mask[hr0:hr1 + 1, clo:chi + 1] = True
    return {
        "height": height, "width": width, "rlo": rlo, "rhi": rhi,
        "clo": clo, "chi": chi, "orientation": orientation,
        "hh": hh, "vw": vw, "hr0": hr0, "hr1": hr1,
        "vc0": vc0, "vc1": vc1, "mask": mask,
    }


def _arm_key(info):
    return (
        info["orientation"], info["rhi"] - info["rlo"] + 1,
        info["chi"] - info["clo"] + 1, info["hh"], info["vw"],
    )


def _grid_key(grid):
    return (grid.shape, grid.astype(np.int8, copy=False).tobytes())


def _puzzle_key(puzzle):
    return tuple(
        tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[split])
        for split in ("train", "test")
    )


def _fingerprint(input_grid, output_grid):
    return hashlib.sha256(
        bytes(input_grid.shape)
        + input_grid.astype(np.int8, copy=False).tobytes()
        + output_grid.astype(np.int8, copy=False).tobytes()
    ).hexdigest()


def _all_moves(info):
    mask = info["mask"]
    height, width = mask.shape
    moves = []
    for row in range(info["rlo"], info["rhi"] + 1):
        cols = np.flatnonzero(mask[row])
        left, right = int(cols[0]), int(cols[-1])
        if left >= 1:
            moves.append((0, row, -1, row * width + right, row * width + left - 1))
        if right + 1 < width:
            moves.append((0, row, 1, row * width + left, row * width + right + 1))
    for col in range(info["clo"], info["chi"] + 1):
        rows = np.flatnonzero(mask[:, col])
        top, bottom = int(rows[0]), int(rows[-1])
        if top >= 1:
            moves.append((1, col, -1, bottom * width + col, (top - 1) * width + col))
        if bottom + 1 < height:
            moves.append((1, col, 1, top * width + col, (bottom + 1) * width + col))
    return moves


def _compatible(move, chosen):
    return all(
        (move[0], move[1]) != (other[0], other[1])
        and move[3] not in (other[3], other[4])
        and move[4] not in (other[3], other[4])
        for other in chosen
    )


def _core_visible(info, moves):
    rows = {m[1] for m in moves if m[0] == 0}
    cols = {m[1] for m in moves if m[0] == 1}
    return (
        any(r not in rows for r in range(info["hr0"], info["hr1"] + 1))
        and any(c not in cols for c in range(info["vc0"], info["vc1"] + 1))
    )


def _colorize(mask, background, foreground):
    result = np.full(mask.shape, background, dtype=np.int8)
    result[mask] = foreground
    return result


def _candidate_infos(input_mask):
    height, width = input_mask.shape
    positions = np.argwhere(input_mask)
    if not len(positions):
        return
    top, left = positions.min(axis=0)
    bottom, right = positions.max(axis=0)
    longest_row = int(np.count_nonzero(input_mask, axis=1).max())
    longest_col = int(np.count_nonzero(input_mask, axis=0).max())
    area = int(input_mask.sum())

    # A one-cell protrusion can expand the observed box, but a single endpoint
    # deletion cannot erase an entire arm boundary (arm thickness is >= 3).
    for rlo in (int(top), int(top) + 1):
        if not 1 <= rlo <= height - 2:
            continue
        for rhi in (int(bottom) - 1, int(bottom)):
            bh = rhi - rlo + 1
            if bh < 7 or abs(bh - longest_col) > 3:
                continue
            for clo in (int(left), int(left) + 1):
                if not 1 <= clo <= width - 2:
                    continue
                for chi in (int(right) - 1, int(right)):
                    bw = chi - clo + 1
                    if bw < 7 or abs(bw - longest_row) > 3:
                        continue
                    for hh in range(3, min(7, bh - 3) + 1):
                        for vw in range(3, min(7, bw - 3) + 1):
                            nominal_area = bh * vw + bw * hh - hh * vw
                            if abs(nominal_area - area) > 6:
                                continue
                            if 2 * (nominal_area + 6) >= height * width:
                                continue
                            for orientation in range(4):
                                info = _info(
                                    height, width, rlo, rhi, clo, chi,
                                    orientation, hh, vw,
                                )
                                differences = np.flatnonzero(
                                    info["mask"].ravel() != input_mask.ravel()
                                )
                                if 2 <= len(differences) <= 6:
                                    yield info, differences


def _assignments(info, input_mask, differences):
    difference_set = set(map(int, differences))
    by_delete, by_add = {}, {}
    for move in _all_moves(info):
        by_delete.setdefault(move[3], []).append(move)
        by_add.setdefault(move[4], []).append(move)
    baseline = info["mask"].ravel()
    observed = input_mask.ravel()
    clue_options = []
    for raw in differences:
        clue = int(raw)
        if baseline[clue] and not observed[clue]:
            options = [(m, 0) for m in by_delete.get(clue, ()) if m[4] not in difference_set]
        elif not baseline[clue] and observed[clue]:
            options = [(m, 1) for m in by_add.get(clue, ()) if m[3] not in difference_set]
        else:
            return
        if not options:
            return
        clue_options.append(options)
    clue_options.sort(key=len)

    chosen = []
    used_cells = set()
    used_slices = set()

    def search(index):
        if index == len(clue_options):
            if _core_visible(info, [m for m, _ in chosen]):
                yield tuple(chosen)
            return
        for move, half in clue_options[index]:
            slice_key = (move[0], move[1])
            if slice_key in used_slices or move[3] in used_cells or move[4] in used_cells:
                continue
            used_slices.add(slice_key)
            used_cells.update((move[3], move[4]))
            chosen.append((move, half))
            yield from search(index + 1)
            chosen.pop()
            used_cells.difference_update((move[3], move[4]))
            used_slices.remove(slice_key)

    yield from search(0)


def _infer(input_grid, limit=2):
    if not isinstance(input_grid, np.ndarray) or input_grid.ndim != 2:
        return []
    values, counts = np.unique(input_grid, return_counts=True)
    if len(values) != 2 or counts[0] == counts[1]:
        return []
    foreground = int(values[int(np.argmin(counts))])
    background = int(values[int(np.argmax(counts))])
    input_mask = input_grid == foreground
    solutions = []
    outputs_seen = set()
    for info, differences in _candidate_infos(input_mask):
        for assignment in _assignments(info, input_mask, differences):
            output_mask = info["mask"].copy()
            for move, _ in assignment:
                output_mask.flat[move[3]] = False
                output_mask.flat[move[4]] = True
            output = _colorize(output_mask, background, foreground)
            key = _grid_key(output)
            if key in outputs_seen:
                continue
            outputs_seen.add(key)
            solutions.append({
                "info": info, "moves": assignment, "output": output,
                "background": background, "foreground": foreground,
            })
            if len(solutions) >= limit:
                return solutions
    return solutions


def _sample_geometry(rng, regime, orientation):
    for _ in range(100):
        if regime == "source_train":
            height = int(rng.integers(12, 27))
            width = int(rng.integers(12, 27))
        else:
            height = int(rng.integers(13, 31))
            width = int(rng.integers(13, 31))
        bh = int(rng.integers(7, height - 1))
        bw = int(rng.integers(7, width - 1))
        hh = int(rng.integers(3, min(7, bh - 3) + 1))
        vw = int(rng.integers(3, min(7, bw - 3) + 1))
        rlo = int(rng.integers(1, height - bh))
        clo = int(rng.integers(1, width - bw))
        info = _info(
            height, width, rlo, rlo + bh - 1, clo, clo + bw - 1,
            orientation, hh, vw,
        )
        if 2 * (int(info["mask"].sum()) + 6) < height * width:
            return info
    return None


def _choose_moves(rng, info, regime):
    all_moves = _all_moves(info)
    # Both splits contain sparse and dense examples. Tests tilt slightly toward
    # interactions, but neither count nor adjacency is a solver convention.
    counts = (2, 3, 4, 5, 6)
    weights = (4, 4, 3, 2, 2) if regime == "source_train" else (3, 4, 4, 3, 2)
    number = int(rng.choice(counts, p=np.array(weights) / sum(weights)))
    axis_mode = int(rng.integers(0, 3))  # row, column, or both
    half_mode = int(rng.integers(0, 3))  # dent, protrusion, or both
    cluster = bool(rng.integers(0, 2))

    for _ in range(100):
        chosen = []
        candidates = list(all_moves)
        rng.shuffle(candidates)
        for _position in range(number):
            available = [
                m for m in candidates if _compatible(m, chosen)
                and (axis_mode == 2 or m[0] == axis_mode)
            ]
            if not available:
                break
            if chosen:
                weights = [
                    5 if cluster and any(
                        m[0] == n[0] and abs(m[1] - n[1]) == 1 for n in chosen
                    ) else 1 for m in available
                ]
                selected = available[int(rng.choice(len(available), p=np.array(weights) / sum(weights)))]
            else:
                selected = available[int(rng.integers(0, len(available)))]
            chosen.append(selected)
        if len(chosen) != number or not _core_visible(info, chosen):
            continue
        if axis_mode == 2 and {m[0] for m in chosen} != {0, 1}:
            continue
        halves = []
        for _ in chosen:
            halves.append(half_mode if half_mode < 2 else int(rng.integers(0, 2)))
        if half_mode == 2 and set(halves) != {0, 1}:
            continue
        return tuple(zip(chosen, halves))
    return None


def _construct(info, assignments, background, foreground):
    input_mask = info["mask"].copy()
    output_mask = info["mask"].copy()
    for move, half in assignments:
        input_mask.flat[move[3] if half == 0 else move[4]] = bool(half)
        output_mask.flat[move[3]] = False
        output_mask.flat[move[4]] = True
    return (
        _colorize(input_mask, background, foreground),
        _colorize(output_mask, background, foreground),
    )


def _palette_plan(rng, total):
    pairs = [(bg, fg) for bg in range(10) for fg in range(10) if bg != fg]
    base = pairs[int(rng.integers(0, len(pairs)))]
    mode = int(rng.integers(0, 3))
    if mode == 0:
        return [base] * total
    if mode == 1:
        choices = [base, (base[1], base[0])]
    else:
        rng.shuffle(pairs)
        choices = [base] + [p for p in pairs if p != base][:3]
    result = [choices[i % len(choices)] for i in range(total)]
    rng.shuffle(result)
    return result


def _orientation_plan(rng, num_train, num_test):
    mode = int(rng.integers(0, 4))
    if mode == 0 and num_train >= 4:
        train = list(map(int, rng.permutation(4)))
        train += [int(rng.integers(0, 4)) for _ in range(num_train - 4)]
    elif mode == 1:
        single = int(rng.integers(0, 4))
        train = [single] * num_train
    else:
        subset = list(map(int, rng.choice(4, size=int(rng.integers(2, 4)), replace=False)))
        train = [subset[int(rng.integers(0, len(subset)))] for _ in range(num_train)]
    rng.shuffle(train)
    if mode == 3:
        novel = [o for o in range(4) if o not in train]
        test = [int(novel[int(rng.integers(0, len(novel)))])] if novel else []
    else:
        test = []
    test += [int(rng.integers(0, 4)) for _ in range(num_test - len(test))]
    return train, test


def _generate_pair(rng, regime, orientation, palette, used_inputs, used_arms):
    background, foreground = palette
    for _ in range(400):
        info = _sample_geometry(rng, regime, orientation)
        if info is None or _arm_key(info) in used_arms:
            continue
        assignments = _choose_moves(rng, info, regime)
        if assignments is None:
            continue
        input_grid, output_grid = _construct(info, assignments, background, foreground)
        if _grid_key(input_grid) in used_inputs:
            continue
        if _fingerprint(input_grid, output_grid) in _SOURCE_FINGERPRINTS:
            continue
        solutions = _infer(input_grid)
        if len(solutions) != 1 or not np.array_equal(solutions[0]["output"], output_grid):
            continue
        return {"input": input_grid, "output": output_grid}, solutions[0]
    raise ValueError("could not generate an unambiguous L pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    for _puzzle_attempt in range(20):
        orientations_train, orientations_test = _orientation_plan(rng, num_train, num_test)
        palettes = _palette_plan(rng, num_train + num_test)
        puzzle = {"train": [], "test": []}
        used_inputs, used_arms = set(), set()
        solutions = {"train": [], "test": []}
        for split, regime, orientations, offset in (
            ("train", "source_train", orientations_train, 0),
            ("test", "target_test", orientations_test, num_train),
        ):
            for index, orientation in enumerate(orientations):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair, solution = _generate_pair(
                    pair_rng, regime, orientation, palettes[offset + index],
                    used_inputs, used_arms,
                )
                puzzle[split].append(pair)
                solutions[split].append(solution)
                used_inputs.add(_grid_key(pair["input"]))
                used_arms.add(_arm_key(solution["info"]))
        train_axes = {m[0][0] for s in solutions["train"] for m in s["moves"]}
        train_halves = {m[1] for s in solutions["train"] for m in s["moves"]}
        if train_axes == {0, 1} and train_halves == {0, 1}:
            return puzzle
    raise ValueError("could not cover both clue conventions in training")


def _check_solution(input_grid, output_grid, solution):
    info = solution["info"]
    assignments = solution["moves"]
    assert 2 <= len(assignments) <= 6
    assert _core_visible(info, [m for m, _ in assignments])
    used = []
    for move, half in assignments:
        assert half in (0, 1)
        assert _compatible(move, used)
        assert info["mask"].flat[move[3]]
        assert not info["mask"].flat[move[4]]
        used.append(move)
    expected_input, expected_output = _construct(
        info, assignments, solution["background"], solution["foreground"],
    )
    assert np.array_equal(input_grid, expected_input)
    assert np.array_equal(output_grid, expected_output)
    assert np.count_nonzero(input_grid != output_grid) == len(assignments)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    inputs, pairs, arm_keys = set(), set(), set()
    solutions = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = pair["input"], pair["output"]
            for grid in (a, b):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert a.shape == b.shape and len(np.unique(a)) == 2
            assert set(np.unique(a)) == set(np.unique(b))
            key = _grid_key(a)
            pair_key = (key, _grid_key(b))
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            assert _fingerprint(a, b) not in _SOURCE_FINGERPRINTS
            inferred = _infer(a)
            assert len(inferred) == 1
            solution = inferred[0]
            assert np.array_equal(solution["output"], b)
            _check_solution(a, b, solution)
            arm_key = _arm_key(solution["info"])
            assert arm_key not in arm_keys
            arm_keys.add(arm_key)
            solutions[split].append(solution)
    train_axes = {m[0][0] for s in solutions["train"] for m in s["moves"]}
    train_halves = {m[1] for s in solutions["train"] for m in s["moves"]}
    test_axes = {m[0][0] for s in solutions["test"] for m in s["moves"]}
    test_halves = {m[1] for s in solutions["test"] for m in s["moves"]}
    assert train_axes == {0, 1} and train_halves == {0, 1}
    assert test_axes <= train_axes and test_halves <= train_halves


def self_test() -> None:
    puzzles = []
    train_counts, test_counts, dimensions, arm_keys, layouts = set(), set(), set(), set(), set()
    palettes, orientations, train_edits, test_edits = set(), set(), set(), set()
    axis_patterns, half_patterns, adjacent = set(), set(), set()
    directions, cluster_zones, thicknesses, margins = set(), set(), set(), set()
    proportions, combined = set(), set()
    mixed_palette_puzzles = 0
    single_palette_puzzles = 0
    swapped_palette_puzzles = 0
    homogeneous_orientation_puzzles = 0
    all_orientation_puzzles = 0
    novel_orientation_puzzles = 0
    varying_dimensions_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        per_palettes = set()
        per_dimensions = set()
        train_orientations = set()
        test_orientations = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                solution = _infer(pair["input"])[0]
                info = solution["info"]
                moves = [m for m, _ in solution["moves"]]
                halfs = [h for _, h in solution["moves"]]
                dimensions.add(pair["input"].shape)
                per_dimensions.add(pair["input"].shape)
                arm_keys.add(_arm_key(info))
                layouts.add((
                    info["height"], info["width"], info["rlo"], info["clo"],
                    *_arm_key(info),
                ))
                roles = (solution["background"], solution["foreground"])
                palettes.add(roles)
                per_palettes.add(roles)
                orientations.add(info["orientation"])
                (train_orientations if split == "train" else test_orientations).add(info["orientation"])
                (train_edits if split == "train" else test_edits).add(len(moves))
                axis_patterns.add(tuple(sorted({m[0] for m in moves})))
                half_patterns.add(tuple(sorted(set(halfs))))
                directions.update((m[0], m[2]) for m in moves)
                thicknesses.update((info["hh"], info["vw"]))
                margins.update((
                    info["rlo"], info["clo"],
                    info["height"] - info["rhi"] - 1,
                    info["width"] - info["chi"] - 1,
                ))
                bh = info["rhi"] - info["rlo"] + 1
                bw = info["chi"] - info["clo"] + 1
                ratio = max(bh, bw) / min(bh, bw)
                proportions.add("elongated" if ratio >= 1.6 else "compact" if ratio <= 1.2 else "middle")
                zones = set()
                for position, first in enumerate(moves):
                    for second in moves[position + 1:]:
                        if first[0] != second[0] or abs(first[1] - second[1]) != 1:
                            continue
                        if first[0] == 0:
                            elbow = info["hr0"] <= first[1] <= info["hr1"] and info["hr0"] <= second[1] <= info["hr1"]
                        else:
                            elbow = info["vc0"] <= first[1] <= info["vc1"] and info["vc0"] <= second[1] <= info["vc1"]
                        zones.add("elbow" if elbow else "distant")
                cluster_zones.add(tuple(sorted(zones)))
                if max(pair["input"].shape) >= 24 and {m[0] for m in moves} == {0, 1} and len(moves) >= 4:
                    combined.add((split, "large_mixed_dense"))
                if max(info["hh"], info["vw"]) >= 6 and zones:
                    combined.add((split, "thick_clustered"))
                adjacent.add(any(
                    m[0] == n[0] and abs(m[1] - n[1]) == 1
                    for i, m in enumerate(moves) for n in moves[i + 1:]
                ))
        mixed_palette_puzzles += len(per_palettes) > 1
        single_palette_puzzles += len(per_palettes) == 1
        swapped_palette_puzzles += any((fg, bg) in per_palettes for bg, fg in per_palettes)
        homogeneous_orientation_puzzles += len(train_orientations) == 1
        all_orientation_puzzles += len(train_orientations) == 4
        novel_orientation_puzzles += bool(test_orientations - train_orientations)
        varying_dimensions_puzzles += len(per_dimensions) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) >= 40
    assert len(dimensions) >= 30 and len(arm_keys) >= 80 and len(layouts) >= 150
    assert len(palettes) >= 15 and orientations == {0, 1, 2, 3}
    assert train_edits == {2, 3, 4, 5, 6} and test_edits == {2, 3, 4, 5, 6}
    assert axis_patterns == {(0,), (1,), (0, 1)}
    assert half_patterns == {(0,), (1,), (0, 1)}
    assert adjacent == {False, True}
    assert directions == {(0, -1), (0, 1), (1, -1), (1, 1)}
    assert cluster_zones >= {(), ("elbow",), ("distant",)}
    assert thicknesses >= {3, 4, 5, 6, 7}
    assert margins >= {1, 2, 3, 4}
    assert proportions == {"compact", "middle", "elongated"}
    assert combined >= {
        ("train", "large_mixed_dense"), ("test", "large_mixed_dense"),
        ("train", "thick_clustered"), ("test", "thick_clustered"),
    }
    assert mixed_palette_puzzles >= 8 and novel_orientation_puzzles >= 3
    assert single_palette_puzzles >= 3 and swapped_palette_puzzles >= 3
    assert homogeneous_orientation_puzzles >= 3 and all_orientation_puzzles >= 3
    assert varying_dimensions_puzzles >= 35
    for offset, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3))):
        p = generate_puzzle(10000 + offset, nt, ns)
        validate_generated_puzzle(p)
        assert (len(p["train"]), len(p["test"])) == (nt, ns)
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(10000 + offset, nt, ns))
    for offset, (nt, ns) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(11000 + offset, nt, ns)
        validate_generated_puzzle(p)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
    for offset, (nt, ns) in enumerate((
        (1, None), (11, None), (None, 0), (None, 5),
        (2, 2), (2, 3), (4, 4),
    )):
        try:
            generate_puzzle(12000 + offset, nt, ns)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
