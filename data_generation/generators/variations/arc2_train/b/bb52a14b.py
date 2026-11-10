import hashlib

import numpy as np


_SOURCE_DIGESTS = {
    "d8c6b4ffa32ed399932b2c72ccc8da537bbefc5f47cdd0b181aa8a6fdae6c752",
    "09ea1e00e18a53b0ed19c8c87abc5aa82df862e9a5edce2da3ed00c10e0b5c3f",
    "6f34d33fd05e70b6c960c7bd24f4d36f0f0886ef75016fe0f3cb1b6cc9824df4",
    "d4ff0fdcf852b552ee3c048a311dc1c87f88d11820d0854f5c6745a257483d9e",
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _count(value, name, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lower <= int(value) <= upper:
        raise ValueError(f"{name} must be in {lower}..{upper}")
    return int(value)


def _counts(seed, rng, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _roles(rng):
    if rng.random() < 0.07:
        return tuple(int(color) for color in rng.permutation([0, 1, 4, 8]))
    return tuple(int(color) for color in rng.choice(10, size=4, replace=False))


def _template(rng, roles):
    _, completion, clue_a, clue_b = roles
    if rng.random() < 0.27:
        while True:
            choices = rng.choice([completion, clue_a, clue_b], size=6, p=[0.54, 0.23, 0.23])
            tile = np.column_stack((choices[:3], choices[3:], choices[:3]))
            missing = int(np.count_nonzero(tile == completion))
            if 2 <= missing <= 6 and {clue_a, clue_b} <= set(tile.flat):
                return tile.astype(np.int8)
    missing = int(rng.choice([2, 3, 4, 5, 6], p=[0.10, 0.15, 0.25, 0.25, 0.25]))
    positions = rng.permutation(9)
    first_count = int(rng.integers(1, 9 - missing))
    flat = np.empty(9, dtype=np.int8)
    flat[positions[:missing]] = completion
    flat[positions[missing:missing + first_count]] = clue_a
    flat[positions[missing + first_count:]] = clue_b
    return flat.reshape(3, 3)


def _separate(first, second):
    return (abs(first[0] - second[0]) >= 4
            or abs(first[1] - second[1]) >= 4)


def _layout(rng, height, width, copy_count, regime):
    positions = [(row, col) for row in range(height - 2) for col in range(width - 2)]
    mode = int(rng.integers(5))
    chosen = []
    if mode == 4 and copy_count >= 2 and regime == "target_test":
        right = [(row, width - 3) for row in range(height // 2, height - 2)]
        chosen.append(right[int(rng.integers(len(right)))])
        stagger = [(row, col) for row, col in positions
                   if row >= height // 2 and abs(row - chosen[0][0]) <= 1
                   and all(_separate((row, col), old) for old in chosen)]
        if stagger:
            chosen.append(stagger[int(rng.integers(len(stagger)))])
    if not chosen:
        edges = [(row, col) for row, col in positions
                 if row in (0, height - 3) or col in (0, width - 3)]
        corners = [(row, col) for row, col in edges
                   if row in (0, height - 3) and col in (0, width - 3)]
        pool = corners if mode == 0 else edges if mode == 1 else positions
        chosen.append(pool[int(rng.integers(len(pool)))])
    while len(chosen) < copy_count:
        candidates = [position for position in positions
                      if all(_separate(position, old) for old in chosen)]
        if mode == 2 and len(chosen) == 1:
            aligned = [position for position in candidates if position[0] == chosen[0][0]]
            candidates = aligned or candidates
        if mode == 3 and len(chosen) == 1:
            staggered = [position for position in candidates
                         if abs(position[0] - chosen[0][0]) == 1]
            candidates = staggered or candidates
        if not candidates:
            return None
        chosen.append(candidates[int(rng.integers(len(candidates)))])
    references = [position for position in positions
                  if all(_separate(position, old) for old in chosen)]
    if not references:
        return None
    reference = references[int(rng.integers(len(references)))]
    return reference, chosen


def _distractors(rng, grid, reserved, boxes, roles, template, regime):
    background, _, clue_a, clue_b = roles
    height, width = grid.shape
    balance = float(rng.uniform(0.15, 0.85))

    def color():
        return clue_a if rng.random() < balance else clue_b

    for row, col in boxes:
        ring = [(near_row, near_col)
                for near_row in range(max(0, row - 1), min(height, row + 4))
                for near_col in range(max(0, col - 1), min(width, col + 4))
                if not reserved[near_row, near_col]]
        if ring:
            for index in rng.choice(len(ring), size=min(len(ring), int(rng.integers(1, 4))), replace=False):
                near_row, near_col = ring[int(index)]
                grid[near_row, near_col] = color()

    clue_cells = np.argwhere(template != roles[1])
    for fragment_index in range(int(rng.integers(1, 4))):
        candidates = [(row, col) for row in range(height - 2)
                      for col in range(width - 2)
                      if not np.any(reserved[row:row + 3, col:col + 3])
                      and np.all(grid[row:row + 3, col:col + 3] == background)]
        if fragment_index == 0:
            nearby = [position for position in candidates
                      if any(abs(position[0] - box[0]) + abs(position[1] - box[1]) <= 7
                             for box in boxes)]
            candidates = nearby or candidates
        else:
            distant = [position for position in candidates
                       if all(abs(position[0] - box[0]) + abs(position[1] - box[1]) >= 7
                              for box in boxes)]
            candidates = distant or candidates
        if candidates:
            row, col = candidates[int(rng.integers(len(candidates)))]
            length = int(rng.integers(2, min(len(clue_cells), 5) + 1))
            for index in rng.choice(len(clue_cells), size=length, replace=False):
                local_row, local_col = clue_cells[int(index)]
                grid[row + local_row, col + local_col] = template[local_row, local_col]

    for _ in range(int(rng.integers(1, 5))):
        length = int(rng.integers(2, 5))
        row = int(rng.integers(height))
        col = int(rng.integers(width))
        direction = int(rng.integers(2))
        for offset in range(length):
            target_row = row + (offset if direction else 0)
            target_col = col + (offset if not direction else 0)
            if target_row < height and target_col < width and not reserved[target_row, target_col]:
                grid[target_row, target_col] = color()

    density = float(rng.uniform(0.045, 0.14 if regime == "source_train" else 0.18))
    target = int(round(density * height * width))
    available = np.argwhere((grid == background) & ~reserved)
    rng.shuffle(available)
    for row, col in available[:max(0, target - int(np.count_nonzero((grid != background) & ~reserved)))]:
        grid[row, col] = color()


def _matches(grid, template, background, completion):
    missing = template == completion
    matches = []
    for row in range(grid.shape[0] - 2):
        for col in range(grid.shape[1] - 2):
            window = grid[row:row + 3, col:col + 3]
            if (np.all(window[missing] == background)
                    and np.array_equal(window[~missing], template[~missing])):
                matches.append((row, col))
    return tuple(matches)


def _inspect(pair):
    source = pair["input"]
    target = pair["output"]
    _require(source.shape == target.shape, "dimensions changed")
    colors, counts = np.unique(source, return_counts=True)
    _require(len(colors) == 4, "four visible roles required")
    background = int(colors[np.argmax(counts)])
    _require(np.count_nonzero(counts == max(counts)) == 1 and max(counts) > source.size * 0.60,
             "background must dominate uniquely")
    full = [(row, col) for row in range(source.shape[0] - 2)
            for col in range(source.shape[1] - 2)
            if np.all(source[row:row + 3, col:col + 3] != background)]
    _require(len(full) == 1, "unique complete 3x3 reference required")
    reference = full[0]
    row, col = reference
    template = source[row:row + 3, col:col + 3].copy()
    confined = [int(color) for color in colors if color != background
                and np.all((source == color) <= _box_mask(source.shape, reference))]
    _require(len(confined) == 1, "completion color must occur only in reference")
    completion = confined[0]
    clues = tuple(sorted(int(color) for color in colors if color not in (background, completion)))
    _require(set(template.flat) == {completion, *clues}, "reference requires both clue colors")
    _require(2 <= np.count_nonzero(template == completion) <= 6, "unsupported completion count")
    actual = _matches(source, template, background, completion)
    _require(1 <= len(actual) <= 3, "must have one to three exact masked copies")
    _require(all(_separate(reference, position) for position in actual), "reference overlaps a copy")
    _require(all(_separate(first, second) for index, first in enumerate(actual)
                 for second in actual[index + 1:]), "masked copies overlap")
    expected = source.copy()
    for copy_row, copy_col in actual:
        window = expected[copy_row:copy_row + 3, copy_col:copy_col + 3]
        window[template == completion] = completion
    _require(np.array_equal(expected, target), "output differs from exact template completion")
    return {"background": background, "completion": completion, "clues": clues,
            "reference": reference, "template": template, "matches": actual}


def _box_mask(shape, position):
    mask = np.zeros(shape, dtype=bool)
    row, col = position
    mask[row:row + 3, col:col + 3] = True
    return mask


def _pair_digest(pair):
    source = pair["input"].astype(np.uint8)
    target = pair["output"].astype(np.uint8)
    return hashlib.sha256(bytes(source.shape) + source.tobytes() + target.tobytes()).hexdigest()


def _key(grid):
    return grid.shape, grid.tobytes()


def _semantic(info):
    background, completion = info["background"], info["completion"]
    first, second = info["clues"]
    return tuple({background: 0, completion: 3, first: 1, second: 2}[int(value)]
                 for value in info["template"].flat)


def _make_pair(rng, roles, count, regime, used_templates):
    for _ in range(1200):
        if rng.random() < 0.16:
            height, width = 22, 22
        else:
            height = int(rng.integers(14, 31))
            width = int(rng.integers(14, 31))
        template = _template(rng, roles)
        layout = _layout(rng, height, width, count, regime)
        if layout is None:
            continue
        reference, copies = layout
        background, completion, _, _ = roles
        source = np.full((height, width), background, dtype=np.int8)
        row, col = reference
        source[row:row + 3, col:col + 3] = template
        reserved = _box_mask(source.shape, reference)
        for copy_row, copy_col in copies:
            window = source[copy_row:copy_row + 3, copy_col:copy_col + 3]
            window[template != completion] = template[template != completion]
            reserved |= _box_mask(source.shape, (copy_row, copy_col))
        _distractors(rng, source, reserved, [reference] + copies, roles, template, regime)
        target = source.copy()
        for copy_row, copy_col in copies:
            window = target[copy_row:copy_row + 3, copy_col:copy_col + 3]
            window[template == completion] = completion
        pair = {"input": source, "output": target}
        try:
            info = _inspect(pair)
        except AssertionError:
            continue
        signature = _semantic(info)
        if set(info["matches"]) == set(copies) and signature not in used_templates:
            if _pair_digest(pair) not in _SOURCE_DIGESTS:
                return pair, signature
    raise RuntimeError("unable to construct an unambiguous pair")


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[section], list) for section in ("train", "test")), "pairs must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "invalid pair counts")
    inputs, pairs, templates = set(), set(), set()
    inspections = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grids must be 2D arrays")
                _require(np.issubdtype(grid.dtype, np.integer), "grids must contain integers")
                _require(all(1 <= size <= 30 for size in grid.shape), "grid size outside ARC bounds")
                _require(np.all((grid >= 0) & (grid <= 9)), "invalid ARC color")
            input_key = _key(pair["input"])
            pair_key = input_key, _key(pair["output"])
            _require(input_key not in inputs and pair_key not in pairs, "duplicate grid or pair")
            inputs.add(input_key)
            pairs.add(pair_key)
            _require(_pair_digest(pair) not in _SOURCE_DIGESTS, "copied source pair")
            info = _inspect(pair)
            signature = _semantic(info)
            _require(signature not in templates, "repeated semantic template")
            templates.add(signature)
            inspections[section].append(info)
    _require({len(info["matches"]) for info in inspections["train"]} >= {1, 2},
             "train must teach one and multiple copy completion")
    _require(all(info["template"].shape == (3, 3) for info in inspections["test"]),
             "test footprint differs from training")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed_code = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed_code)
    train_count, test_count = _counts(seed_code, rng, num_train, num_test)
    consistent = bool(rng.integers(2))
    common_roles = _roles(rng)
    train_copies = [1, 2] + [int(rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20]))
                              for _ in range(train_count - 2)]
    rng.shuffle(train_copies)
    puzzle = {"train": [], "test": []}
    used_templates, used_inputs = set(), set()
    for section, counts in (("train", train_copies),
                            ("test", [int(rng.choice([1, 2, 3], p=[0.20, 0.45, 0.35]))
                                      for _ in range(test_count)])):
        for copy_count in counts:
            for _ in range(100):
                roles = common_roles if consistent else _roles(rng)
                child = np.random.default_rng(int(rng.integers(0, 1 << 63)))
                regime = "source_train" if section == "train" else "target_test"
                pair, signature = _make_pair(child, roles, copy_count, regime, used_templates)
                if _key(pair["input"]) not in used_inputs:
                    puzzle[section].append(pair)
                    used_templates.add(signature)
                    used_inputs.add(_key(pair["input"]))
                    break
            else:
                raise RuntimeError("unable to generate distinct puzzle pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _same(first, second):
    return all(len(first[section]) == len(second[section])
               and all(np.array_equal(pair_a[side], pair_b[side])
                       for pair_a, pair_b in zip(first[section], second[section])
                       for side in ("input", "output"))
               for section in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _require(len({len(puzzle["train"]) for puzzle in puzzles}) >= 4, "train counts collapsed")
    _require(len({len(puzzle["test"]) for puzzle in puzzles}) >= 3, "test counts collapsed")
    _require(len({tuple(_key(pair["input"]) for section in ("train", "test")
                        for pair in puzzle[section]) for puzzle in puzzles}) >= 40,
             "cross-seed grids collapsed")
    _require(_same(generate_puzzle(314, 6, 3), generate_puzzle(314, 6, 3)), "nondeterministic")
    _require(not _same(puzzles[0], puzzles[1]), "different seeds match")
    dimensions, layouts, signatures, colors, copy_counts, missing_counts = set(), set(), set(), set(), set(), set()
    symmetric = small = elongated = same_row = staggered = boundary = interior = 0
    mixed_palette = consistent_palette = test_upper = sparse = clustered = 0
    source_palette = source_sized = corner = source_like_test = 0
    mixed_training = role_swaps = low_clutter = high_clutter = 0
    fragment_near = fragment_far = combined = 0
    for puzzle in puzzles:
        roles = []
        training_roles = []
        puzzle_combinations = set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _inspect(pair)
                source, template, matches = pair["input"], info["template"], info["matches"]
                background, completion = info["background"], info["completion"]
                _require(_matches(source, template, background, completion) == matches,
                         "independent exact match scan failed")
                _require(_pair_digest(pair) not in _SOURCE_DIGESTS, "source pair copied")
                roles.append((background, completion, *info["clues"]))
                if section == "train":
                    training_roles.append(roles[-1])
                dimensions.add(source.shape)
                layouts.add((source.shape, info["reference"], matches))
                signatures.add(_semantic(info))
                colors.add(tuple(sorted(np.unique(source))))
                copy_counts.add(len(matches))
                missing_counts.add(int(np.count_nonzero(template == completion)))
                symmetric += int(np.array_equal(template, np.fliplr(template))
                                 or np.array_equal(template, np.flipud(template)))
                small += int(min(source.shape) < 18)
                elongated += int(abs(source.shape[0] - source.shape[1]) >= 8)
                same_row += int(any(first[0] == second[0] for index, first in enumerate(matches)
                                    for second in matches[index + 1:]))
                staggered += int(any(abs(first[0] - second[0]) == 1 for index, first in enumerate(matches)
                                     for second in matches[index + 1:]))
                boundary += int(any(row in (0, source.shape[0] - 3) or col in (0, source.shape[1] - 3)
                                    for row, col in matches))
                corner += int(any(row in (0, source.shape[0] - 3)
                                  and col in (0, source.shape[1] - 3) for row, col in matches))
                interior += int(any(0 < row < source.shape[0] - 3 and 0 < col < source.shape[1] - 3
                                    for row, col in matches))
                source_sized += int(source.shape == (22, 22))
                source_palette += int(set(np.unique(source)) == {0, 1, 4, 8})
                test_upper += int(section == "test" and any(row < source.shape[0] // 2
                                                              for row, _ in matches))
                source_like_test += int(section == "test" and len(matches) >= 2
                                        and all(row >= source.shape[0] // 2 for row, _ in matches)
                                        and any(col == source.shape[1] - 3 for _, col in matches)
                                        and any(abs(first[0] - second[0]) == 1
                                                for index, first in enumerate(matches)
                                                for second in matches[index + 1:]))
                sparse += int(np.count_nonzero(template == completion) >= 5)
                mask = _box_mask(source.shape, info["reference"])
                for position in matches:
                    mask |= _box_mask(source.shape, position)
                clutter = (source != background) & ~mask
                clustered += int(np.any(clutter[:, :-2] & clutter[:, 1:-1] & clutter[:, 2:])
                                 or np.any(clutter[:-2] & clutter[1:-1] & clutter[2:]))
                clutter_fraction = np.count_nonzero(clutter) / source.size
                low_clutter += int(clutter_fraction < 0.08)
                high_clutter += int(clutter_fraction > 0.11)
                candidate_fragments = []
                for row in range(source.shape[0] - 2):
                    for col in range(source.shape[1] - 2):
                        if np.any(mask[row:row + 3, col:col + 3]):
                            continue
                        window = source[row:row + 3, col:col + 3]
                        correct = np.count_nonzero((window == template) & (template != completion))
                        if 2 <= correct < np.count_nonzero(template != completion):
                            candidate_fragments.append((row, col))
                fragment_near += int(any(
                    any(abs(row - tile_row) + abs(col - tile_col) <= 7
                        for tile_row, tile_col in (info["reference"], *matches))
                    for row, col in candidate_fragments))
                fragment_far += int(any(
                    all(abs(row - tile_row) + abs(col - tile_col) >= 7
                        for tile_row, tile_col in (info["reference"], *matches))
                    for row, col in candidate_fragments))
                puzzle_combinations.add((section, len(matches),
                                         int(np.count_nonzero(template == completion)),
                                         min(source.shape) < 18))
        if len(set(roles)) > 1:
            mixed_palette += 1
        else:
            consistent_palette += 1
        mixed_training += int(len(set(training_roles)) > 1)
        role_swaps += int(any(train_role[1] == test_role[0]
                              and train_role[0] != test_role[0]
                              for train_role in training_roles
                              for test_role in roles[len(training_roles):]))
        combined += int(len(puzzle_combinations) >= 3)
        _require({1, 2} <= {len(_inspect(pair)["matches"]) for pair in puzzle["train"]},
                 "training copy-count coverage failed")
    _require(len(dimensions) >= 55 and len(layouts) >= 160 and len(signatures) >= 100,
             "structural diversity collapsed")
    _require(len(colors) >= 25 and copy_counts == {1, 2, 3} and missing_counts == {2, 3, 4, 5, 6},
             "role or count diversity collapsed")
    _require(all(value > 0 for value in (symmetric, small, elongated, same_row, staggered,
                                        boundary, interior, mixed_palette, consistent_palette,
                                        test_upper, sparse, clustered, source_palette, source_sized,
                                        corner, source_like_test, mixed_training, role_swaps,
                                        low_clutter, high_clutter, fragment_near, fragment_far)),
             "review axis collapsed")
    _require(combined >= 35, "variation axes do not combine within puzzles")
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                           (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + index, train, test)
        validate_generated_puzzle(puzzle)
        _require((train is None or len(puzzle["train"]) == train)
                 and (test is None or len(puzzle["test"]) == test), "explicit count ignored")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(3000, **kwargs)
        except ValueError:
            continue
        raise AssertionError("invalid counts accepted")
