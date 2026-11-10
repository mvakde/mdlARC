import numpy as np


# Original input descriptors, including the row-8, column-7 seed in train 1.
_SOURCE_INPUT_DESCRIPTORS = frozenset({
    (12, 10, ((3, 2, 8), (4, 6, 4), (8, 7, 5))),
    (14, 15, ((4, 9, 3), (11, 5, 5))),
    (11, 8, ((4, 4, 9),)),
    (12, 12, ((2, 3, 8), (10, 9, 3))),
    (11, 16, ((1, 13, 3), (2, 4, 8), (6, 5, 6), (7, 11, 5))),
})
_ORIENTATIONS = (0, 1, 2, 3)  # top-left, top-right, bottom-left, bottom-right


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _checked_count(value, name, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return result


def _resolve_counts(rng, seed, num_train, num_test):
    train_count = _checked_count(num_train, "num_train", 2, 10)
    test_count = _checked_count(num_test, "num_test", 1, 4)
    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        train_count = test_count + 1 + (seed // 4) % (10 - test_count)
    elif train_count is None:
        train_count = int(rng.integers(test_count + 1, 11))
    elif test_count is None:
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    return train_count, test_count


def _orientation(row, col, height, width):
    _require(row != height - 1 - row and col != width - 1 - col,
             "seed has an equal-distance boundary tie")
    return (0 if row < height - 1 - row else 2) + (0 if col < width - 1 - col else 1)


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    _require(int(counts[index]) > grid.size // 2, "background is not dominant")
    return int(values[index])


def _mask(height, width, row, col):
    orientation = _orientation(row, col, height, width)
    mask = np.zeros((height, width), dtype=bool)
    if orientation in (0, 1):
        mask[:row + 1, col] = True
    else:
        mask[row:, col] = True
    if orientation in (0, 2):
        mask[row, :col + 1] = True
    else:
        mask[row, col:] = True
    return mask


def _derive_output_and_info(input_grid):
    height, width = input_grid.shape
    background = _background(input_grid)
    locations = np.argwhere(input_grid != background)
    _require(1 <= len(locations) <= 6, "input needs one to six isolated seeds")
    seeds = [(int(r), int(c), int(input_grid[r, c])) for r, c in locations]
    colors = [color for _, _, color in seeds]
    _require(len(colors) == len(set(colors)), "seed colors in a pair must be distinct")
    for i, (r, c, _) in enumerate(seeds):
        _require(0 < r < height - 1 and 0 < c < width - 1,
                 "seed must be interior")
        for r2, c2, _ in seeds[i + 1:]:
            _require(max(abs(r - r2), abs(c - c2)) > 1,
                     "input seeds must be isolated")
    output = np.full_like(input_grid, background)
    occupied = np.zeros((height, width), dtype=bool)
    orientations = []
    arm_lengths = []
    for row, col, color in seeds:
        orientation = _orientation(row, col, height, width)
        mask = _mask(height, width, row, col)
        _require(not np.any(occupied & mask), "colored output arms overlap")
        occupied |= mask
        output[mask] = color
        orientations.append(orientation)
        arm_lengths.append((row if orientation < 2 else height - 1 - row,
                            col if orientation % 2 == 0 else width - 1 - col))
    _require(not np.array_equal(input_grid, output), "output must change input")
    return output, {"background": background, "seeds": tuple(seeds),
                    "colors": tuple(colors), "orientations": tuple(orientations),
                    "arm_lengths": tuple(arm_lengths), "seed_count": len(seeds)}


def _input_descriptor(grid):
    background = _background(grid)
    seeds = tuple((int(r), int(c), int(grid[r, c]))
                  for r, c in np.argwhere(grid != background))
    return (int(grid.shape[0]), int(grid.shape[1]), seeds) if background == 0 else None


def _grid_key(grid):
    return (grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes())


def _train_counts(rng, count):
    if count == 2:
        patterns = ((1, 3), (2, 2), (2, 3), (3, 2), (3, 3))
        return list(patterns[int(rng.integers(len(patterns)))])
    result = [int(x) for x in rng.integers(1, 4, size=count)]
    while sum(result) < 5:
        candidates = [i for i, value in enumerate(result) if value < 3]
        index = candidates[int(rng.integers(len(candidates)))]
        result[index] += 1
    return result


def _train_orientations(rng, counts):
    flat = [int(x) for x in rng.permutation(4)]
    flat.extend(int(x) for x in rng.integers(0, 4, size=sum(counts) - 4))
    rng.shuffle(flat)
    chunks = []
    cursor = 0
    for count in counts:
        chunks.append(flat[cursor:cursor + count])
        cursor += count
    return chunks


def _train_colors(rng, orientations, backgrounds, palette):
    # Random assignment is constrained by observable coverage and pairwise clarity.
    for _ in range(2048):
        chunks = []
        for chunk, background in zip(orientations, backgrounds):
            available = [color for color in palette if color != background]
            if len(available) < len(chunk):
                break
            chunks.append([int(x) for x in rng.choice(available, len(chunk), replace=False)])
        if len(chunks) != len(orientations):
            continue
        associations = {color: set() for color in palette}
        for o_chunk, c_chunk in zip(orientations, chunks):
            for orientation, color in zip(o_chunk, c_chunk):
                associations[color].add(orientation)
        if all(associations.values()) and any(len(x) >= 2 for x in associations.values()):
            return chunks
    raise ValueError("unable to assign a clear training palette")


def _candidate_positions(rng, height, width, orientation):
    positions = []
    for row in range(1, height - 1):
        if row == height - 1 - row:
            continue
        for col in range(1, width - 1):
            if col != width - 1 - col and _orientation(row, col, height, width) == orientation:
                positions.append((row, col))
    rng.shuffle(positions)
    return positions


def _positions(rng, height, width, orientations):
    occupied = np.zeros((height, width), dtype=bool)
    positions = []
    for orientation in orientations:
        candidates = _candidate_positions(rng, height, width, orientation)
        chosen = None
        for row, col in candidates:
            if any(max(abs(row - r), abs(col - c)) <= 1 for r, c in positions):
                continue
            mask = _mask(height, width, row, col)
            if not np.any(occupied & mask):
                chosen = (row, col, mask)
                break
        if chosen is None:
            return None
        row, col, mask = chosen
        positions.append((row, col))
        occupied |= mask
    return positions


def _dimensions(rng, regime, mode, count, train_shapes):
    minimum = 7 if count <= 2 else (9 if count <= 4 else 14)
    if regime == "source_train":
        return (int(rng.integers(minimum, 24)), int(rng.integers(minimum, 24)))
    if mode == "wide_shift":
        low = max(minimum, max(w for _, w in train_shapes) + 1)
        width = int(rng.integers(low, 30))
        height = int(rng.integers(minimum, width)) if width > minimum else minimum
        return height, width
    if mode == "tall":
        width = int(rng.integers(minimum, 25))
        height = int(rng.integers(width + 1, 31))
        return height, width
    if mode == "square":
        side = int(rng.integers(minimum, 28))
        return side, side
    if mode == "same_shape":
        eligible = [(h, w) for h, w in train_shapes
                    if h >= minimum and w >= minimum]
        if eligible:
            return eligible[int(rng.integers(len(eligible)))]
    if mode == "compact":
        eligible = [(h, w) for h, w in train_shapes
                    if h >= minimum and w >= minimum]
        if eligible:
            h, w = eligible[int(rng.integers(len(eligible)))]
            return (int(rng.integers(minimum, h + 1)),
                    int(rng.integers(minimum, w + 1)))
    return (int(rng.integers(minimum, 29)), int(rng.integers(minimum, 29)))


def _generate_pair(rng, regime, orientations, colors, background, mode,
                   train_shapes, seen_inputs):
    _require(len(orientations) == len(colors), "seed assignments do not match")
    _require(len(set(colors)) == len(colors) and background not in colors,
             "pair colors must be distinct from one another and the background")
    for _ in range(180):
        height, width = _dimensions(rng, regime, mode, len(colors), train_shapes)
        positions = _positions(rng, height, width, orientations)
        if positions is None:
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        for (row, col), color in zip(positions, colors):
            grid[row, col] = color
        key = _grid_key(grid)
        if key in seen_inputs or _input_descriptor(grid) in _SOURCE_INPUT_DESCRIPTORS:
            continue
        try:
            output, _ = _derive_output_and_info(grid)
        except AssertionError:
            continue
        seen_inputs.add(key)
        return {"input": grid, "output": output}
    # A 30x30 diagonal placement fits every permitted multiset of six
    # orientations, including six seeds in the same quadrant.
    for shift in (0, 1, 2):
        used = [0, 0, 0, 0]
        grid = np.full((30, 30), background, dtype=np.int8)
        for orientation, color in zip(orientations, colors):
            depth = 1 + shift + 2 * used[orientation]
            used[orientation] += 1
            row = depth if orientation < 2 else 29 - depth
            col = depth if orientation % 2 == 0 else 29 - depth
            grid[row, col] = color
        key = _grid_key(grid)
        if key not in seen_inputs:
            output, _ = _derive_output_and_info(grid)
            seen_inputs.add(key)
            return {"input": grid, "output": output}
    raise ValueError("unable to construct a valid distinct pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    counts = _train_counts(rng, train_count)
    orientations = _train_orientations(rng, counts)
    palette_min = max(counts)
    palette_max = min(6, sum(counts) - 1)
    palette_size = int(rng.integers(palette_min, palette_max + 1))
    if palette_size >= 4 and rng.random() < 0.3:
        palette = (0,) + tuple(int(x) for x in rng.choice(
            np.arange(1, 10), palette_size - 1, replace=False))
    else:
        palette = tuple(int(x) for x in rng.choice(
            np.arange(1, 10), palette_size, replace=False))
    nonzero_backgrounds = [x for x in range(1, 10) if x not in palette]
    backgrounds = [0, int(rng.choice(nonzero_backgrounds))]
    backgrounds.extend(int(rng.choice([0] + nonzero_backgrounds))
                       for _ in range(train_count - 2))
    rng.shuffle(backgrounds)
    color_chunks = _train_colors(rng, orientations, backgrounds, palette)
    seen_inputs = set()
    train = []
    for o_chunk, c_chunk, background in zip(orientations, color_chunks, backgrounds):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        train.append(_generate_pair(pair_rng, "source_train", o_chunk, c_chunk,
                                    background, "free", (), seen_inputs))
    train_shapes = [pair["input"].shape for pair in train]
    test = []
    modes = ("wide_shift", "tall", "square", "same_shape", "compact", "free")
    for _ in range(test_count):
        test_background = int(rng.choice([0] + nonzero_backgrounds))
        novel_candidates = [x for x in range(10)
                            if x not in palette and x != test_background]
        novel = int(rng.choice(novel_candidates)) if novel_candidates and rng.random() < 0.5 else None
        available = [x for x in palette if x != test_background]
        if novel is not None:
            available.append(novel)
        max_count = min(6, len(available))
        test_seed_count = int(rng.integers(1, max_count + 1))
        if test_seed_count >= 4 and rng.random() < 0.55:
            o_chunk = [int(x) for x in rng.permutation(4)]
            o_chunk.extend(int(x) for x in rng.integers(0, 4, size=test_seed_count - 4))
        else:
            o_chunk = [int(x) for x in rng.integers(0, 4, size=test_seed_count)]
            if test_seed_count in (2, 3, 4) and rng.random() < 0.48:
                o_chunk[1] = o_chunk[0]
        c_chunk = [int(x) for x in rng.choice(available, test_seed_count, replace=False)]
        if novel is not None and novel not in c_chunk and rng.random() < 0.75:
            c_chunk[-1] = novel
        mode = modes[int(rng.integers(len(modes)))]
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        test.append(_generate_pair(pair_rng, "target_test", o_chunk, c_chunk,
                                   test_background, mode, train_shapes, seen_inputs))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid, label):
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
             f"{label} must be a two-dimensional NumPy array")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} must have integer cells")
    _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
             f"{label} has invalid dimensions")
    _require(np.all((grid >= 0) & (grid <= 9)), f"{label} has invalid ARC colors")


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    train_infos, test_infos = [], []
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label} must contain exactly input and output")
            input_grid, output_grid = pair["input"], pair["output"]
            _validate_grid(input_grid, f"{label}.input")
            _validate_grid(output_grid, f"{label}.output")
            _require(input_grid.shape == output_grid.shape, f"{label} changed size")
            input_key = _grid_key(input_grid)
            pair_key = (input_key, _grid_key(output_grid))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate pair or input")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _require(_input_descriptor(input_grid) not in _SOURCE_INPUT_DESCRIPTORS,
                     f"{label} copies a source input")
            expected, info = _derive_output_and_info(input_grid)
            _require(np.array_equal(output_grid, expected),
                     f"{label} violates the nearest-edge L transformation")
            (train_infos if split == "train" else test_infos).append(info)
    _require(all(1 <= info["seed_count"] <= 3 for info in train_infos),
             "training seed counts must be one to three")
    _require(set(o for info in train_infos for o in info["orientations"]) == set(_ORIENTATIONS),
             "training must show all four nearest-edge directions")
    _require({info["background"] for info in train_infos} - {0} and
             0 in {info["background"] for info in train_infos},
             "training must demonstrate both zero and nonzero backgrounds")
    associations = {}
    for info in train_infos:
        for color, orientation in zip(info["colors"], info["orientations"]):
            associations.setdefault(color, set()).add(orientation)
    _require(2 <= len(associations) <= 6 and any(len(x) >= 2 for x in associations.values()),
             "training must separate color identity from direction")
    _require(all(1 <= info["seed_count"] <= 6 for info in test_infos),
             "test seed count is outside the family range")
    for info in test_infos:
        _require(len(set(info["colors"]) - set(associations)) <= 1,
                 "test adds more than one unseen foreground color")


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[k], b[k]) for a, b in zip(first[split], second[split])
                   for k in ("input", "output")) for split in ("train", "test"))


def _predict_from_training(train_pairs, test_input):
    # Read the demonstrations before applying the inferred color-preserving rule.
    demonstrated = set()
    for pair in train_pairs:
        grid = pair["input"]
        background = _background(grid)
        for row, col in np.argwhere(grid != background):
            row, col = int(row), int(col)
            demonstrated.add(_orientation(row, col, *grid.shape))
            _require(pair["output"][row, col] == grid[row, col],
                     "a training output erased its seed")
    _require(demonstrated == set(_ORIENTATIONS),
             "training lacks a nearest-edge direction")
    background = _background(test_input)
    prediction = np.full_like(test_input, background)
    height, width = test_input.shape
    for row, col in np.argwhere(test_input != background):
        row, col = int(row), int(col)
        color = test_input[row, col]
        endpoint_row = 0 if row < height - 1 - row else height - 1
        endpoint_col = 0 if col < width - 1 - col else width - 1
        prediction[min(row, endpoint_row):max(row, endpoint_row) + 1, col] = color
        prediction[row, min(col, endpoint_col):max(col, endpoint_col) + 1] = color
    return prediction


def self_test() -> None:
    source_first_input = np.zeros((12, 10), dtype=np.int8)
    for row, col, color in ((3, 2, 8), (4, 6, 4), (8, 7, 5)):
        source_first_input[row, col] = color
    _require(_input_descriptor(source_first_input) in _SOURCE_INPUT_DESCRIPTORS,
             "original first input is not excluded")
    automatic = generate_puzzle(1234)
    validate_generated_puzzle(automatic)
    explicit = generate_puzzle(5678, 7, 3)
    _require(_puzzles_equal(explicit, generate_puzzle(5678, 7, 3)),
             "fixed seed and counts must be deterministic")
    _require(not _puzzles_equal(generate_puzzle(8001, 6, 2),
                                generate_puzzle(8002, 6, 2)),
             "different seeds should differ")
    for seed, train_count, test_count in ((9001, 2, 1), (9002, 10, 1),
                                          (9003, 10, 4), (9101, None, 1),
                                          (9102, None, 4), (9103, 2, None),
                                          (9104, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            _require(len(puzzle["train"]) == train_count, "training count mismatch")
        if test_count is not None:
            _require(len(puzzle["test"]) == test_count, "test count mismatch")
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}, {"num_train": 4.0},
                      {"num_test": 1.0}):
        try:
            generate_puzzle(9200, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {arguments}")
    train_counts, test_counts, palettes, shapes, layouts = set(), set(), set(), set(), set()
    backgrounds, test_seed_counts, train_patterns = set(), set(), set()
    novel_count = same_quadrant = short_arms = zero_seeds = mixed_arms = 0
    same_shape_test = smaller_test = wide_shift_test = large_test = 0
    within_background = within_seed_count = 0
    aspect = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info = [_derive_output_and_info(pair["input"])[1] for pair in puzzle["train"]]
        within_background += len({info["background"] for info in train_info}) > 1
        within_seed_count += len({info["seed_count"] for info in train_info}) > 1
        palette = set(c for info in train_info for c in info["colors"])
        palettes.add(tuple(sorted(palette)))
        train_patterns.add(tuple(info["seed_count"] for info in train_info))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                expected, info = _derive_output_and_info(grid)
                _require(np.array_equal(expected, pair["output"]), "self-test rule mismatch")
                backgrounds.add(info["background"])
                shapes.add(grid.shape)
                layouts.add((grid.shape, tuple((r, c) for r, c, _ in info["seeds"])))
                aspect.add("square" if grid.shape[0] == grid.shape[1]
                           else "wide" if grid.shape[1] > grid.shape[0] else "tall")
                short_arms += sum(1 for v, h in info["arm_lengths"] if v == 1 or h == 1)
                mixed_arms += sum(1 for v, h in info["arm_lengths"]
                                  if min(v, h) == 1 and max(v, h) >= 4)
                zero_seeds += sum(c == 0 for c in info["colors"])
                if split == "test":
                    _require(np.array_equal(
                        _predict_from_training(puzzle["train"], grid), pair["output"]),
                        "training-only prediction disagrees with test output")
                    test_seed_counts.add(info["seed_count"])
                    novel_count += bool(set(info["colors"]) - palette)
                    same_quadrant += len(set(info["orientations"])) < info["seed_count"]
                    same_shape_test += grid.shape in [p["input"].shape
                                                      for p in puzzle["train"]]
                    smaller_test += any(grid.shape[0] < p["input"].shape[0]
                                        and grid.shape[1] < p["input"].shape[1]
                                        for p in puzzle["train"])
                    wide_shift_test += grid.shape[1] > grid.shape[0] and grid.shape[1] > max(
                        p["input"].shape[1] for p in puzzle["train"])
                    large_test += info["seed_count"] >= 5
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity is weak")
    _require(len(palettes) >= 12 and len(shapes) >= 40 and len(layouts) >= 100,
             "palette, size, or layout diversity is weak")
    _require(len(train_patterns) >= 12 and len(test_seed_counts) >= 5,
             "train/test object count diversity is weak")
    _require(len(backgrounds) >= 5 and zero_seeds > 0 and short_arms > 0
             and mixed_arms > 0 and within_background == 42
             and within_seed_count >= 30,
             "background or arm-length diversity is weak")
    _require(novel_count > 0 and same_quadrant > 0 and len(aspect) == 3,
             "test colors, orientations, or aspect ratios lack diversity")
    _require(same_shape_test > 0 and smaller_test > 0 and wide_shift_test > 0
             and large_test > 0, "test size and count regimes lack diversity")
