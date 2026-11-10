import numpy as np


_SOURCE_INPUT_SIGNATURES = {
    "\n".join((
        "055400000005555", "004005545005055", "505000445055505",
        "500505540005505", "550505000550050", "050050550055500",
        "550550050000000", "554050500505055", "545405504545050",
        "555005500405055", "000500000000505", "050050500550500",
        "000005055000005", "050055550505000",
    )),
    "\n".join((
        "0500550005505005", "5500005050055500", "0045050550045550",
        "0554005005004005", "0045005050000050", "0005055550505505",
        "5050045455550500", "0000554550000005", "0055005055055450",
        "0050050005505545", "5004000050000555", "5545450555000055",
        "5005550545405000", "5005500504000550", "5555500055505500",
        "5555000050555505", "5050000050505500", "5000050055550005",
    )),
    "\n".join((
        "005000000500500", "050005550550555", "005000550500555",
        "555450004450505", "054500000505005", "505050000005505",
        "000505000000055", "550055050550000", "505005554500050",
        "045500545555000", "054505550000505", "005000550050500",
        "005000000050555", "000555050055050",
    )),
    "\n".join((
        "0505505550005005050", "0005000555000055000",
        "0550005055000050050", "0055055005505050500",
        "0050040505050005505", "0050454550500054555",
        "5000040055000054450", "5055550500050504500",
        "5500505055055555000", "5500555050050050000",
        "0055000054505500000", "5554050554450055550",
        "0045005504505005555", "0000500500005050055",
        "5505000555550500000", "5500050000000505005",
        "5050550550000550000", "0550000005555500000",
    )),
    "\n".join((
        "0050055050505545005", "0550545550550450550",
        "5550554055055550505", "5540000500050500055",
        "5455550500550550500", "5505000550545005055",
        "0555505555445055000", "5005555000050550050",
        "0500000500500050505", "0000500500500500550",
        "0000550505050400050", "5545555500504540505",
        "5554000555055405555", "0045000005050555555",
        "5050050450055555005", "5505505540055505055",
        "5550555405505500550", "5555000500505050500",
        "0505550055500555555",
    )),
}


def _grid_signature(grid):
    return "\n".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _array_key(grid):
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _footprint(height, width, row, col):
    cells = [(row, col)]
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = row + dr
        cc = col + dc
        if 0 <= rr < height and 0 <= cc < width:
            cells.append((rr, cc))
    return tuple(cells)


def _find_cross_solutions(grid, filler, clue, limit=2, respect_background=True):
    height, width = grid.shape
    clue_positions = [
        tuple(int(value) for value in position)
        for position in np.argwhere(grid == clue)
    ]
    clue_index = {
        position: index
        for index, position in enumerate(clue_positions)
    }

    candidates = []
    for row in range(height):
        for col in range(width):
            footprint = _footprint(height, width, row, col)
            values = [int(grid[r, c]) for r, c in footprint]

            if respect_background and any(
                value not in (filler, clue) for value in values
            ):
                continue

            clue_cells = [
                position
                for position in footprint
                if int(grid[position]) == clue
            ]
            filler_count = sum(
                int(grid[position]) == filler
                for position in footprint
            )

            if len(clue_cells) < 2 or filler_count < 1:
                continue

            clue_mask = 0
            cell_mask = 0
            for position in footprint:
                linear_index = position[0] * width + position[1]
                cell_mask |= 1 << linear_index
                if int(grid[position]) == clue:
                    clue_mask |= 1 << clue_index[position]

            candidates.append({
                "center": (row, col),
                "footprint": footprint,
                "clue_mask": clue_mask,
                "cell_mask": cell_mask,
                "clue_count": len(clue_cells),
                "filler_count": filler_count,
                "center_is_clue": int(grid[row, col]) == clue,
            })

    option_lists = [[] for _ in clue_positions]
    for candidate_index, candidate in enumerate(candidates):
        mask = candidate["clue_mask"]
        for clue_index_value in range(len(clue_positions)):
            if mask & (1 << clue_index_value):
                option_lists[clue_index_value].append(candidate_index)

    full_mask = (1 << len(clue_positions)) - 1
    solutions = []

    def search(covered_mask, occupied_mask, selected):
        if len(solutions) >= limit:
            return

        if covered_mask == full_mask:
            solution = tuple(sorted(selected))
            if solution not in solutions:
                solutions.append(solution)
            return

        best_options = None
        for clue_index_value in range(len(clue_positions)):
            if covered_mask & (1 << clue_index_value):
                continue

            compatible = []
            for candidate_index in option_lists[clue_index_value]:
                candidate = candidates[candidate_index]
                if candidate["clue_mask"] & covered_mask:
                    continue
                if candidate["cell_mask"] & occupied_mask:
                    continue
                compatible.append(candidate_index)

            if not compatible:
                return

            if best_options is None or len(compatible) < len(best_options):
                best_options = compatible
                if len(best_options) == 1:
                    break

        for candidate_index in best_options:
            candidate = candidates[candidate_index]
            search(
                covered_mask | candidate["clue_mask"],
                occupied_mask | candidate["cell_mask"],
                selected + (candidate_index,),
            )
            if len(solutions) >= limit:
                return

    if clue_positions:
        search(0, 0, ())

    return solutions, candidates


def _visible_interpretations(grid, limit=2):
    """Infer roles and footprints from the input alone."""
    colors = [int(x) for x in np.unique(grid)]
    if len(colors) != 3:
        return []
    interpretations = []
    for clue in colors:
        for filler in colors:
            if filler == clue:
                continue
            background = next(x for x in colors if x != clue and x != filler)
            solutions, candidates = _find_cross_solutions(grid, filler, clue, limit=limit)
            for solution in solutions:
                interpretations.append((
                    (background, filler, clue),
                    [candidates[index] for index in solution],
                ))
                if len(interpretations) >= limit:
                    return interpretations
    return interpretations


def _boundary_sides(center, height, width):
    row, col = center
    sides = []
    if row == 0:
        sides.append("top")
    if row == height - 1:
        sides.append("bottom")
    if col == 0:
        sides.append("left")
    if col == width - 1:
        sides.append("right")
    return tuple(sides)


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    input_colors = [int(x) for x in np.unique(input_grid)]
    output_colors = [int(x) for x in np.unique(output_grid)]
    assert len(input_colors) == 3 and len(output_colors) == 4
    changed = input_grid != output_grid
    assert np.any(changed)
    from_colors = np.unique(input_grid[changed])
    to_colors = np.unique(output_grid[changed])
    assert len(from_colors) == len(to_colors) == 1
    filler, completion = int(from_colors[0]), int(to_colors[0])
    assert completion not in input_colors
    assert set(output_colors) == set(input_colors) | {completion}
    interpretations = _visible_interpretations(input_grid)
    assert len(interpretations) == 1
    (background, inferred_filler, clue), crosses = interpretations[0]
    assert inferred_filler == filler
    expected = input_grid.copy()
    occupied = set()
    for cross in crosses:
        assert 2 <= cross["clue_count"] <= 4
        assert cross["filler_count"] >= 1
        for cell in cross["footprint"]:
            assert cell not in occupied
            occupied.add(cell)
            if int(input_grid[cell]) == filler:
                expected[cell] = completion
    assert np.array_equal(expected, output_grid)
    assert sum(x["clue_count"] for x in crosses) == int(np.count_nonzero(input_grid == clue))
    outside_background = sum(int(input_grid[r, c] == background) for r in range(input_grid.shape[0]) for c in range(input_grid.shape[1]) if (r, c) not in occupied)
    outside_filler = sum(int(input_grid[r, c] == filler) for r in range(input_grid.shape[0]) for c in range(input_grid.shape[1]) if (r, c) not in occupied)
    assert outside_background >= 2 and outside_filler >= 2
    return {
        "roles": (background, filler, clue, completion),
        "shape": input_grid.shape,
        "crosses": crosses,
        "density": outside_filler / (outside_filler + outside_background),
    }


def _sample_puzzle_parameters(rng):
    colors = [int(x) for x in rng.choice(10, size=4, replace=False)]
    return {
        "completion": colors[3],
        "stable_roles": rng.random() < 0.52,
        "fixed_roles": tuple(colors[:3]),
        "no_clipping": rng.random() < 0.18,
        "two_clue_puzzle": rng.random() < 0.15,
    }


def _sample_roles(rng, parameters):
    if parameters["stable_roles"]:
        return parameters["fixed_roles"]
    available = [x for x in range(10) if x != parameters["completion"]]
    return tuple(int(x) for x in rng.choice(available, size=3, replace=False))


def _sample_shape(rng, regime):
    if regime == "source_train":
        band = int(rng.choice(3, p=(0.25, 0.51, 0.24)))
    else:
        band = int(rng.choice(3, p=(0.16, 0.39, 0.45)))
    bounds = ((6, 11), (12, 20), (21, 30))[band]
    height = int(rng.integers(bounds[0], bounds[1] + 1))
    width = int(rng.integers(bounds[0], bounds[1] + 1))
    # Independently extend one dimension; this creates both wide and tall grids.
    if rng.random() < 0.31:
        if rng.random() < 0.5:
            height = int(rng.integers(6, 31))
        else:
            width = int(rng.integers(6, 31))
    if rng.random() < 0.16:
        width = height
    return height, width


def _place_centers(rng, height, width, count, clip_count, allowed_sides, force_side, allow_corners):
    all_cells = [(r, c) for r in range(height) for c in range(width)]
    boundary = [p for p in all_cells if _boundary_sides(p, height, width)]
    boundary = [p for p in boundary if set(_boundary_sides(p, height, width)) <= allowed_sides and (allow_corners or len(_boundary_sides(p, height, width)) == 1)]
    interior = [p for p in all_cells if not _boundary_sides(p, height, width)]
    chosen, occupied = [], set()

    def take(pool):
        for index in rng.permutation(len(pool)):
            center = pool[int(index)]
            footprint = set(_footprint(height, width, *center))
            if not footprint & occupied:
                chosen.append(center)
                occupied.update(footprint)
                return True
        return False

    if force_side is not None:
        if not take([p for p in boundary if force_side in _boundary_sides(p, height, width)]):
            return None
    while len(chosen) < clip_count:
        if not take(boundary):
            return None
    while len(chosen) < count:
        if not take(interior):
            return None
    return chosen


def _generate_pair(rng, regime, parameters, allowed_sides, force_side=None, allow_corners=True):
    completion = parameters["completion"]
    for _ in range(450):
        height, width = _sample_shape(rng, regime)
        area = height * width
        max_count = min(12, max(1, area // 24))
        low_count = max(1, max_count // 3) if regime == "target_test" and rng.random() < 0.70 else 1
        count = int(rng.integers(low_count, max_count + 1))
        max_clip = min(count, 3)
        if not allowed_sides:
            clip_count = 0
        else:
            clip_count = min(max_clip, int(rng.choice(4, p=(0.39, 0.35, 0.19, 0.07))))
        if force_side is not None:
            clip_count = max(1, clip_count)
        centers = _place_centers(rng, height, width, count, clip_count, allowed_sides, force_side, allow_corners)
        if centers is None:
            continue
        background, filler, clue = _sample_roles(rng, parameters)
        density = float(rng.uniform(0.17, 0.74))
        grid = np.full((height, width), background, dtype=np.int8)
        grid[rng.random((height, width)) < density] = filler
        footprints = []
        two_clue_pair = parameters["two_clue_puzzle"] or rng.random() < 0.12
        for center in centers:
            footprint = _footprint(height, width, *center)
            footprints.append(footprint)
            for position in footprint:
                grid[position] = filler
            maximum = min(4, len(footprint) - 1)
            if maximum == 2 or two_clue_pair:
                clue_count = 2
            else:
                counts = list(range(2, maximum + 1))
                weights = np.array([0.48, 0.32, 0.20][:len(counts)], dtype=float)
                weights /= weights.sum()
                clue_count = int(rng.choice(counts, p=weights))
            for index in rng.choice(len(footprint), size=clue_count, replace=False):
                grid[footprint[int(index)]] = clue
        output = grid.copy()
        for footprint in footprints:
            for position in footprint:
                if int(grid[position]) == filler:
                    output[position] = completion
        if _grid_signature(grid) in _SOURCE_INPUT_SIGNATURES:
            continue
        try:
            info = _analyze_pair(grid, output)
        except AssertionError:
            continue
        if info["roles"] != (background, filler, clue, completion):
            continue
        if {cross["center"] for cross in info["crosses"]} != set(centers):
            continue
        return grid, output
    raise RuntimeError("could not construct an unambiguous pair")


def _coerce_count(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    if num_train is not None:
        num_train = _coerce_count("num_train", num_train)
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")
    if num_test is not None:
        num_test = _coerce_count("num_test", num_test)
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed * 7 + (seed // 4) * 3) % (11 - low)
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = low + (seed * 11 + num_test * 3) % (11 - low)
    elif num_test is None:
        high = min(4, num_train - 1)
        num_test = 1 + (seed * 5 + num_train) % high
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    seen = set()
    training_sides = set()
    training_corner = False
    first_side = None if parameters["no_clipping"] else ("top", "bottom", "left", "right")[int(rng.integers(0, 4))]
    for section, count, regime in (("train", train_count, "source_train"), ("test", test_count, "target_test")):
        for pair_index in range(count):
            for _ in range(80):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                sides = (set() if parameters["no_clipping"] else set(("top", "bottom", "left", "right"))) if section == "train" else training_sides
                force_side = first_side if section == "train" and pair_index == 0 else None
                allow_corners = section == "train" or training_corner
                input_grid, output_grid = _generate_pair(pair_rng, regime, parameters, sides, force_side, allow_corners)
                key = _array_key(input_grid)
                if key in seen:
                    continue
                seen.add(key)
                puzzle[section].append({"input": input_grid, "output": output_grid})
                if section == "train":
                    info = _analyze_pair(input_grid, output_grid)
                    for cross in info["crosses"]:
                        labels = _boundary_sides(cross["center"], *info["shape"])
                        training_sides.update(labels)
                        training_corner |= len(labels) == 2
                break
            else:
                raise RuntimeError("could not construct distinct puzzle pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(np.min(grid)) and int(np.max(grid)) <= 9
            key = _array_key(pair["input"])
            pair_key = (key, _array_key(pair["output"]))
            assert key not in seen_inputs and pair_key not in seen_pairs
            assert _grid_signature(pair["input"]) not in _SOURCE_INPUT_SIGNATURES
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            info = _analyze_pair(pair["input"], pair["output"])
            assert 6 <= info["shape"][0] <= 30 and 6 <= info["shape"][1] <= 30
            assert 1 <= len(info["crosses"]) <= 12
            infos[section].append(info)
    completions = {info["roles"][3] for values in infos.values() for info in values}
    assert len(completions) == 1
    training_sides = set()
    training_corner = False
    for info in infos["train"]:
        for cross in info["crosses"]:
            sides = _boundary_sides(cross["center"], *info["shape"])
            training_sides.update(sides)
            training_corner |= len(sides) == 2
    for info in infos["test"]:
        for cross in info["crosses"]:
            sides = _boundary_sides(cross["center"], *info["shape"])
            assert set(sides) <= training_sides
            if len(sides) == 2:
                assert training_corner


def _puzzles_equal(first, second):
    return all(np.array_equal(p[x], q[x]) for section in ("train", "test") for p, q in zip(first[section], second[section]) for x in ("input", "output")) and all(len(first[s]) == len(second[s]) for s in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(9173, 6, 2)
    b = generate_puzzle(9173, 6, 2)
    assert _puzzles_equal(a, b)
    assert not _puzzles_equal(a, generate_puzzle(9174, 6, 2))
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(200 + index, *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    train_counts, test_counts, shapes, layouts, roles = set(), set(), set(), set(), set()
    sides, clue_counts, mask_types, close_pairs = set(), set(), set(), 0
    small, large, variable_roles, stable_roles, noise_needed = 0, 0, 0, 0, 0
    corners, unclipped_puzzles, homogeneous_puzzles, mixed_puzzles = 0, 0, 0, 0
    smaller_tests, sparser_tests, varying_counts, combined_pairs = 0, 0, 0, 0
    densities = []
    total_inputs, inputs = 0, set()
    for seed in range(1000, 1042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        palette_set = set()
        all_info = []
        section_info = {"train": [], "test": []}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid = pair["input"]
                info = _analyze_pair(grid, pair["output"])
                all_info.append(info)
                section_info[section].append(info)
                densities.append(info["density"])
                roles.add(info["roles"])
                palette_set.add(info["roles"][:3])
                shapes.add(info["shape"])
                small += max(info["shape"]) <= 11
                large += max(info["shape"]) >= 25
                key = _array_key(grid)
                assert key not in inputs
                inputs.add(key)
                total_inputs += 1
                centers = [x["center"] for x in info["crosses"]]
                layouts.add((info["shape"], tuple(sorted(centers))))
                close_pairs += any(abs(a[0]-b[0])+abs(a[1]-b[1]) < 5 for i,a in enumerate(centers) for b in centers[i+1:])
                pair_sides = set()
                for cross in info["crosses"]:
                    clue_counts.add(cross["clue_count"])
                    labels = _boundary_sides(cross["center"], *info["shape"])
                    sides.update(labels)
                    pair_sides.update(labels)
                    corners += len(labels) == 2
                    clue_cells = {p for p in cross["footprint"] if int(grid[p]) == info["roles"][2]}
                    if len(clue_cells) == 2:
                        if cross["center"] in clue_cells:
                            mask_types.add("center_arm")
                        else:
                            arms = list(clue_cells)
                            mask_types.add("opposite_arms" if arms[0][0] == arms[1][0] or arms[0][1] == arms[1][1] else "adjacent_arms")
                    elif len(clue_cells) == 3:
                        mask_types.add("center_two_arms" if cross["center"] in clue_cells else "three_arms")
                    else:
                        mask_types.add("four_clues")
                combined_pairs += len(pair_sides) >= 1 and len(info["crosses"]) >= 4 and len({x["clue_count"] for x in info["crosses"]}) >= 2 and info["shape"][0] != info["shape"][1]
                alternatives, _ = _find_cross_solutions(grid, info["roles"][1], info["roles"][2], respect_background=False)
                noise_needed += len(alternatives) > 1
        variable_roles += len(palette_set) > 1
        stable_roles += len(palette_set) == 1
        varying_counts += len({len(info["crosses"]) for info in all_info}) > 1
        homogeneous_puzzles += all(all(x["clue_count"] == 2 for x in info["crosses"]) for info in all_info)
        mixed_puzzles += len({x["clue_count"] for info in all_info for x in info["crosses"]}) > 1
        unclipped_puzzles += all(not _boundary_sides(x["center"], *info["shape"]) for info in all_info for x in info["crosses"])
        smaller_tests += any(np.prod(t["shape"]) < np.prod(tr["shape"]) for t in section_info["test"] for tr in section_info["train"])
        sparser_tests += any(len(t["crosses"]) < len(tr["crosses"]) for t in section_info["test"] for tr in section_info["train"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 60 and len(layouts) >= 100 and len(roles) >= 30
    assert inputs and len(inputs) == total_inputs
    assert small > 0 and large > 0 and variable_roles > 0 and stable_roles > 0
    assert sides == {"top", "bottom", "left", "right"}
    assert clue_counts == {2, 3, 4}
    assert mask_types == {"center_arm", "opposite_arms", "adjacent_arms", "center_two_arms", "three_arms", "four_clues"}
    assert close_pairs > 0 and noise_needed > 0
    assert corners > 0 and unclipped_puzzles > 0
    assert homogeneous_puzzles > 0 and mixed_puzzles > 0
    assert smaller_tests > 0 and sparser_tests > 0 and varying_counts > 0
    assert combined_pairs > 0 and min(densities) < 0.25 and max(densities) > 0.65
