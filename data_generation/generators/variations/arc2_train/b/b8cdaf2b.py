import numpy as np


_SOURCE_PAIRS = (
    (("00000", "00000", "00000", "06660", "61116"),
     ("00000", "00000", "10001", "06660", "61116")),
    (("00000", "00000", "00000", "00800", "88388"),
     ("00000", "30003", "03030", "00800", "88388")),
    (("0000000", "0000000", "0000000", "0000000", "0000000", "0022200", "2244422"),
     ("0000000", "0000000", "0000000", "4000004", "0400040", "0022200", "2244422")),
    (("000", "020", "242"), ("404", "020", "242")),
    (("000000000", "000000000", "000000000", "000000000", "000000000",
      "000000000", "000000000", "000888000", "888222888"),
     ("000000000", "000000000", "000000000", "000000000", "200000002",
      "020000020", "002000200", "000888000", "888222888")),
)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(rng, seed, num_train, num_test):
    for value, name, minimum, maximum in (
        (num_train, "num_train", 2, 10),
        (num_test, "num_test", 1, 4),
    ):
        if value is not None and (not _integer(value) or not minimum <= value <= maximum):
            raise ValueError(f"{name} must be an integer from {minimum} through {maximum}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _base_pair(background, scaffold, payload, flank, center, height):
    width = 2 * flank + center
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[-1, :flank] = scaffold
    input_grid[-1, flank:flank + center] = payload
    input_grid[-1, flank + center:] = scaffold
    input_grid[-2, flank:flank + center] = scaffold
    output_grid = input_grid.copy()
    for distance in range(1, flank + 1):
        row = height - 2 - distance
        output_grid[row, flank - distance] = payload
        output_grid[row, flank + center - 1 + distance] = payload
    return input_grid, output_grid


def _construct_pair(background, scaffold, payload, flank, center, height, orientation):
    input_grid, output_grid = _base_pair(background, scaffold, payload, flank, center, height)
    return np.rot90(input_grid, orientation).copy(), np.rot90(output_grid, orientation).copy()


def _grid_key(grid):
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _source_copy(input_grid, output_grid):
    key = (_grid_key(input_grid), _grid_key(output_grid))
    for source_input, source_output in _SOURCE_PAIRS:
        source_key = (
            ((len(source_input), len(source_input[0])), tuple(int(c) for row in source_input for c in row)),
            ((len(source_output), len(source_output[0])), tuple(int(c) for row in source_output for c in row)),
        )
        if key == source_key:
            return True
    return False


def _parse_input(input_grid):
    candidates = []
    for turn in range(4):
        canonical = np.rot90(input_grid, turn)
        height, width = canonical.shape
        if height < 3 or width < 3:
            continue
        background = int(canonical[0, 0])
        scaffold = int(canonical[-1, 0])
        if scaffold == background or int(canonical[-1, -1]) != scaffold:
            continue
        flank = 0
        while flank < width and int(canonical[-1, flank]) == scaffold:
            flank += 1
        center = width - 2 * flank
        if flank < 1 or center < 1 or height < flank + 2:
            continue
        payload = int(canonical[-1, flank])
        if payload in (background, scaffold):
            continue
        expected_input, expected_output = _base_pair(
            background, scaffold, payload, flank, center, height
        )
        if np.array_equal(canonical, expected_input):
            candidates.append({
                "background": background,
                "scaffold": scaffold,
                "payload": payload,
                "flank": flank,
                "center": center,
                "height": height,
                "width": width,
                "unused": height - flank - 2,
                "orientation": (-turn) % 4,
                "predicted": np.rot90(expected_output, -turn).copy(),
            })
    assert len(candidates) == 1
    return candidates[0]


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    information = _parse_input(input_grid)
    assert np.array_equal(output_grid, information["predicted"])
    assert not np.array_equal(input_grid, output_grid)
    return information


def _sample_layouts(rng, seed, train_count, test_count):
    small_flank = int(rng.integers(1, 3))
    large_flank = int(rng.choice([value for value in range(2, 7) if value != small_flank]))
    small_center = int(rng.choice((1, 3)))
    large_center = int(rng.choice([value for value in range(2, 9) if value != small_center]))
    corners = [
        (small_flank, small_center),
        (large_flank, large_center),
        (small_flank, large_center),
        (large_flank, small_center),
    ]
    if train_count == 2:
        train_layouts = corners[:2]
    elif train_count == 3:
        train_layouts = corners[:2] + [corners[int(rng.integers(2, 4))]]
    else:
        train_layouts = corners.copy()
    train_catalog = [(flank, center) for flank in range(1, 8)
                     for center in range(1, 10) if 2 * flank + center <= 25]
    rng.shuffle(train_catalog)
    for layout in train_catalog:
        if len(train_layouts) >= train_count:
            break
        if layout not in train_layouts:
            train_layouts.append(layout)
    test_catalog = [(flank, center) for flank in range(1, 11)
                    for center in range(1, 11) if 2 * flank + center <= 30]
    rng.shuffle(test_catalog)
    test_layouts = []
    preferred = (
        lambda flank, center: flank <= 2 and center == 1,
        lambda flank, center: center == 1,
        lambda flank, center: center >= 4 and center % 2 == 0,
        lambda flank, center: flank >= 5,
    )[seed % 4]
    for layout in test_catalog:
        if layout not in train_layouts and preferred(*layout):
            test_layouts.append(layout)
            break
    for layout in test_catalog:
        if len(test_layouts) >= test_count:
            break
        if layout not in train_layouts and layout not in test_layouts:
            test_layouts.append(layout)
    assert len(train_layouts) == train_count and len(test_layouts) == test_count
    return train_layouts, test_layouts


def _choose_palette(rng, background, history, category):
    available = [color for color in range(10) if color != background]
    previous_scaffolds = {scaffold for _, scaffold, _ in history}
    previous_payloads = {payload for _, _, payload in history}
    used = previous_scaffolds | previous_payloads
    if category == "fresh":
        fresh = [color for color in available if color not in used]
        if len(fresh) >= 2:
            chosen = rng.choice(fresh, size=2, replace=False)
            return int(chosen[0]), int(chosen[1])
    if category in ("retain", "cross") and history:
        scaffold_pool = previous_scaffolds if category == "retain" else previous_payloads
        payload_pool = previous_payloads if category == "retain" else previous_scaffolds
        scaffold_pool = sorted(scaffold_pool - {background})
        payload_pool = sorted(payload_pool - {background})
        if scaffold_pool:
            scaffold = int(rng.choice(scaffold_pool))
            alternatives = [color for color in payload_pool if color != scaffold]
            if alternatives:
                return scaffold, int(rng.choice(alternatives))
            return scaffold, int(rng.choice([color for color in available if color != scaffold]))
    chosen = rng.choice(available, size=2, replace=False)
    return int(chosen[0]), int(chosen[1])


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    varying_background = bool(rng.integers(0, 2))
    mixed_orientation = bool(rng.integers(0, 2))
    first_background = int(rng.integers(0, 10))
    train_layouts, test_layouts = _sample_layouts(rng, seed, train_count, test_count)
    train_orientations = [0] * train_count
    if mixed_orientation:
        train_orientations[1] = int(rng.integers(1, 4))
        for index in range(2, train_count):
            train_orientations[index] = int(rng.integers(0, 4))
    demonstrated_orientations = sorted(set(train_orientations))
    test_orientations = [int(rng.choice(demonstrated_orientations)) for _ in range(test_count)]
    categories = list(rng.permutation(["fresh", "retain", "cross", "independent"]))
    puzzle = {"train": [], "test": []}
    history = []
    used_inputs = set()
    used_active = set()
    layouts = train_layouts + test_layouts
    orientations = train_orientations + test_orientations
    for index, ((flank, center), orientation) in enumerate(zip(layouts, orientations)):
        if index == 0 or not varying_background:
            background = first_background
        elif index == 1:
            background = int(rng.choice([color for color in range(10) if color != first_background]))
        else:
            background = int(rng.integers(0, 10))
        category = "fresh" if index == 1 else categories[index % 4]
        scaffold, payload = _choose_palette(rng, background, history, category)
        max_unused = min(18, 30 - flank - 2)
        if index == 0 or bool(rng.integers(0, 2)):
            unused = int(rng.integers(0, min(6, max_unused) + 1))
        else:
            unused = int(rng.integers(min(7, max_unused), max_unused + 1))
        if index == train_count and seed % 3 == 0:
            unused = 0
        if index == train_count and seed % 3 == 1:
            unused = min(max_unused, int(rng.integers(5, max_unused + 1))) if max_unused >= 5 else max_unused
        height = flank + 2 + unused
        for offset in range(9):
            candidate_scaffold = (scaffold + offset) % 10
            if candidate_scaffold == background:
                continue
            candidate_payload = payload
            if candidate_payload in (background, candidate_scaffold):
                candidate_payload = next(color for color in range(10)
                                         if color not in (background, candidate_scaffold))
            active = (background, candidate_scaffold, candidate_payload, flank, center, orientation)
            input_grid, output_grid = _construct_pair(
                background, candidate_scaffold, candidate_payload,
                flank, center, height, orientation
            )
            if active not in used_active and _grid_key(input_grid) not in used_inputs and not _source_copy(input_grid, output_grid):
                break
        else:
            raise AssertionError("no distinct non-source pair")
        split = "train" if index < train_count else "test"
        puzzle[split].append({"input": input_grid, "output": output_grid})
        used_inputs.add(_grid_key(input_grid))
        used_active.add(active)
        history.append((background, candidate_scaffold, candidate_payload))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    inputs, pairs, active = set(), set(), set()
    information = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= extent <= 30 for extent in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            assert input_key not in inputs and pair_key not in pairs
            assert not _source_copy(pair["input"], pair["output"])
            info = _analyze_pair(pair["input"], pair["output"])
            signature = tuple(info[key] for key in
                              ("background", "scaffold", "payload", "flank", "center", "orientation"))
            assert signature not in active
            inputs.add(input_key)
            pairs.add(pair_key)
            active.add(signature)
            information[split].append(info)
    train = information["train"]
    test = information["test"]
    assert len({item["flank"] for item in train}) >= 2
    assert len({item["center"] for item in train}) >= 2
    assert any(item["flank"] <= 2 and item["center"] in (1, 3) for item in train)
    if train_count >= 4:
        observed = {(item["flank"], item["center"]) for item in train}
        assert any((first_flank, first_center) in observed
                   and (first_flank, second_center) in observed
                   and (second_flank, first_center) in observed
                   and (second_flank, second_center) in observed
                   for first_flank in {item["flank"] for item in train}
                   for second_flank in {item["flank"] for item in train} if first_flank != second_flank
                   for first_center in {item["center"] for item in train}
                   for second_center in {item["center"] for item in train} if first_center != second_center)
    train_orientations = {item["orientation"] for item in train}
    assert 0 in train_orientations
    assert all(item["orientation"] in train_orientations for item in test)
    train_layouts = {(item["flank"], item["center"]) for item in train}
    assert all((item["flank"], item["center"]) not in train_layouts for item in test)


def _fingerprint(puzzle):
    return tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def _predict_from_training(training_pairs, test_input):
    demonstrations = [_analyze_pair(pair["input"].copy(), pair["output"].copy())
                      for pair in training_pairs]
    assert len({item["flank"] for item in demonstrations}) >= 2
    assert len({item["center"] for item in demonstrations}) >= 2
    test_information = _parse_input(test_input.copy())
    assert test_information["orientation"] in {item["orientation"] for item in demonstrations}
    return test_information["predicted"]


def self_test() -> None:
    train_counts, test_counts, fingerprints = set(), set(), set()
    shapes, layouts, colors, orientations, backgrounds, unused = set(), set(), set(), set(), set(), set()
    short_tests = one_center_tests = even_center_tests = wide_center_tests = tight_tests = 0
    background_mixed = rotated_mixed = many_colors = 0
    source_scale = retained_roles = cross_roles = fresh_colors = exact_reversals = 0
    short_one_tight = rotated_even = total_tests = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(_fingerprint(puzzle))
        infos = {split: [_analyze_pair(pair["input"], pair["output"]) for pair in puzzle[split]]
                 for split in ("train", "test")}
        for pair in puzzle["test"]:
            prediction = _predict_from_training(puzzle["train"], pair["input"])
            assert np.array_equal(prediction, pair["output"])
        all_infos = infos["train"] + infos["test"]
        background_mixed += len({item["background"] for item in all_infos}) > 1
        rotated_mixed += len({item["orientation"] for item in all_infos}) > 1
        many_colors += len({color for item in all_infos for color in
                            (item["scaffold"], item["payload"])}) > 4
        source_scale += any(item["flank"] <= 2 and item["center"] in (1, 3)
                            and item["height"] <= 9 and item["width"] <= 9
                            for item in infos["train"])
        training_scaffolds = {item["scaffold"] for item in infos["train"]}
        training_payloads = {item["payload"] for item in infos["train"]}
        training_assignments = {(item["scaffold"], item["payload"])
                                for item in infos["train"]}
        for split in ("train", "test"):
            for item, pair in zip(infos[split], puzzle[split]):
                shapes.add(pair["input"].shape)
                layouts.add((split, item["flank"], item["center"], item["orientation"], item["unused"]))
                colors.add((item["background"], item["scaffold"], item["payload"]))
                orientations.add(item["orientation"])
                backgrounds.add(item["background"])
                unused.add(item["unused"])
                if split == "test":
                    total_tests += 1
                    short_tests += item["flank"] <= 2
                    one_center_tests += item["center"] == 1
                    even_center_tests += item["center"] % 2 == 0
                    wide_center_tests += item["center"] > 3
                    tight_tests += item["unused"] == 0
                    retained_roles += (item["scaffold"] in training_scaffolds
                                       or item["payload"] in training_payloads)
                    cross_roles += (item["scaffold"] in training_payloads
                                    or item["payload"] in training_scaffolds)
                    fresh_colors += (item["scaffold"] not in training_scaffolds | training_payloads
                                     or item["payload"] not in training_scaffolds | training_payloads)
                    exact_reversals += (item["payload"], item["scaffold"]) in training_assignments
                    short_one_tight += (item["flank"] <= 2 and item["center"] == 1
                                        and item["unused"] == 0)
                    rotated_even += item["orientation"] != 0 and item["center"] % 2 == 0
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) >= 40 and len(shapes) >= 35 and len(layouts) >= 100
    assert len(colors) >= 80 and len(backgrounds) >= 8 and orientations == {0, 1, 2, 3}
    assert len(unused) >= 12 and background_mixed >= 10 and rotated_mixed >= 10 and many_colors >= 10
    assert min(short_tests, one_center_tests, even_center_tests, wide_center_tests, tight_tests) >= 3
    assert source_scale >= 20 and min(retained_roles, cross_roles, fresh_colors) >= 10
    assert exact_reversals < total_tests // 2 and short_one_tight >= 1 and rotated_even >= 3
    assert _fingerprint(generate_puzzle(987654, 7, 3)) == _fingerprint(generate_puzzle(987654, 7, 3))
    assert _fingerprint(generate_puzzle(987654, 7, 3)) != _fingerprint(generate_puzzle(987655, 7, 3))
    for requested_train, requested_test in ((2, 1), (10, 1), (10, 4), (7, 3),
                                             (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1000 + (requested_train or 0) * 10 + (requested_test or 0),
                                 requested_train, requested_test)
        validate_generated_puzzle(puzzle)
        assert requested_train is None or len(puzzle["train"]) == requested_train
        assert requested_test is None or len(puzzle["test"]) == requested_test
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(123, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {arguments}")
