"""Diagonal count-up ARC puzzle generator."""

import numpy as np


OPPOSITE = (3, 2, 1, 0)


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def checked(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer from {low} through {high}")
    return int(value)


def counts(rng, seed, num_train, num_test):
    train = checked(num_train, "num_train", 2, 10)
    test = checked(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        return minimum + (seed // 4) % (11 - minimum), test
    if train is None:
        return int(rng.integers(test + 1, 11)), test
    return train, int(rng.integers(1, min(4, train - 1) + 1))


def end(stage):
    return (stage + 1) * (stage + 2) // 2


def corner_spec(size, corner):
    return ((0, 0, 1, 1), (0, size - 1, 1, -1),
            (size - 1, 0, -1, 1), (size - 1, size - 1, -1, -1))[corner]


def axis(corner):
    return min(corner, OPPOSITE[corner])


def segment(size, background, foreground, corner, start, length):
    grid = np.full((size, size), background, dtype=np.int8)
    row, col, dr, dc = corner_spec(size, corner)
    positions = np.arange(start, start + length)
    grid[row + dr * positions, col + dc * positions] = foreground
    return grid


def pair_for(size, stage, palette, corner):
    need(1 <= stage <= 6 and end(stage) <= size <= 30 and palette[0] != palette[1], "invalid pair parameters")
    b, f = palette
    return {
        "input": segment(size, b, f, corner, stage * (stage - 1) // 2, stage),
        "output": segment(size, b, f, corner, stage * (stage + 1) // 2, stage + 1),
    }


def grid_key(grid):
    array = np.asarray(grid, dtype=np.int64)
    return tuple(array.shape), array.tobytes()


def source_copy(size, stage, palette, corner):
    return size == 16 and stage in (1, 2, 3, 4) and palette == (7, 2) and corner == 2


def palette_from(rng):
    values = rng.choice(10, 2, replace=False)
    return int(values[0]), int(values[1])


def sample_size(rng, stage):
    options = [size for size in range(end(stage), 31) if size != stage * stage]
    draw = int(rng.integers(0, 5))
    if draw == 0:
        return options[0]
    if draw == 1:
        return options[-1]
    return int(options[int(rng.integers(0, len(options)))])


def fresh_pair(stage, size, palette, corner, used, locked=False):
    """Use another palette if a local configuration repeats; never retry at random."""
    palettes = [palette]
    if not locked:
        palettes += [(b, f) for b in range(10) for f in range(10)
                     if b != f and (b, f) != palette]
    for b, f in palettes:
        if source_copy(size, stage, (b, f), corner):
            continue
        result = pair_for(size, stage, (b, f), corner)
        key = grid_key(result["input"])
        if key not in used:
            used.add(key)
            return result
    if locked:
        raise AssertionError("locked progression has a duplicate or source copy")
    for other_size in range(end(stage), 31):
        for other_corner in range(4):
            for b, f in palettes:
                if source_copy(other_size, stage, (b, f), other_corner):
                    continue
                result = pair_for(other_size, stage, (b, f), other_corner)
                key = grid_key(result["input"])
                if key not in used:
                    used.add(key)
                    return result
    raise AssertionError("no unused pair exists")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = counts(rng, seed, num_train, num_test)

    # A centered input needs a demonstrated direction on its diagonal.
    anchor = int(rng.integers(0, 4))
    base_palette = palette_from(rng)
    style = int(rng.choice(4, p=[0.20, 0.27, 0.25, 0.28]))
    if train_count == 3 and test_count == 1:
        style = 0
    local_recolor = style == 3 or rng.random() < 0.65
    shared = style in (0, 1)
    target = 4 if style == 0 else int(rng.integers(3, 7))
    max_core = min(train_count, target - 1, 4)
    core_length = 3 if style == 0 and max_core >= 3 else int(rng.integers(2, max_core + 1))
    core_stages = list(range(target - core_length, target))
    extra_stages = [int(rng.integers(1, target)) for _ in range(train_count - core_length)]

    if style == 0:
        shared_size = 16
    elif shared:
        center = target in (4, 5) and rng.random() < (0.55 if target == 5 else 0.32)
        shared_size = target * target if center else sample_size(rng, target)
    else:
        shared_size = None
    centered_primary = ((shared and shared_size == target * target) or
                        (not shared and target in (4, 5) and
                         rng.random() < (0.60 if target == 5 else 0.40)))
    if shared and base_palette == (7, 2) and anchor == 2 and shared_size == 16:
        base_palette = (7, 3)

    used = set()
    train = []
    for index, stage in enumerate(core_stages):
        if shared:
            size, palette, corner = shared_size, base_palette, anchor
        else:
            size = sample_size(rng, stage)
            palette = (base_palette if not local_recolor or index == 0 or
                       rng.random() < 0.40 else palette_from(rng))
            corner = anchor
            if style == 3 and not centered_primary and index > 0 and rng.random() < 0.50:
                corner = int(rng.integers(0, 4))
        train.append(fresh_pair(stage, size, palette, corner, used, locked=shared))

    for stage in extra_stages:
        size = shared_size if shared else sample_size(rng, stage)
        palette = (base_palette if not local_recolor or
                   (style in (0, 1) and rng.random() < 0.70) else palette_from(rng))
        corner = anchor
        if style == 3 and rng.random() < 0.75:
            choices = list(range(4))
            if centered_primary:
                choices.remove(OPPOSITE[anchor])
            corner = int(choices[int(rng.integers(0, len(choices)))])
        train.append(fresh_pair(stage, size, palette, corner, used))
    train = [train[int(i)] for i in rng.permutation(len(train))]

    test = []
    for index in range(test_count):
        if index == 0:
            stage = target
        elif style == 3 and rng.random() < 0.75:
            stage = core_stages[int(rng.integers(0, len(core_stages)))]
        else:
            stage = target if rng.random() < 0.30 else int(rng.integers(1, target))
        if index == 0 and shared:
            size, palette, corner = shared_size, base_palette, anchor
        else:
            size = (shared_size if shared else
                    target * target if index == 0 and centered_primary else sample_size(rng, stage))
            palette = (base_palette if not local_recolor or
                       (style in (0, 1) and rng.random() < 0.70) else palette_from(rng))
            corner = anchor
            if index > 0 and style == 3 and rng.random() < 0.75:
                opposite_axis = [c for c in range(4) if axis(c) != axis(anchor)]
                corner = int(opposite_axis[int(rng.integers(0, 2))])
        test.append(fresh_pair(stage, size, palette, corner, used, locked=index == 0 and shared))
    test = [test[int(i)] for i in rng.permutation(len(test))]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_grid(grid, label):
    need(isinstance(grid, np.ndarray), f"{label} is not an ndarray")
    need(grid.ndim == 2 and all(1 <= dim <= 30 for dim in grid.shape), f"{label} has bad dimensions")
    need(np.issubdtype(grid.dtype, np.integer), f"{label} is not integer")
    need(int(grid.min()) >= 0 and int(grid.max()) <= 9, f"{label} has bad colors")


def input_candidates(grid):
    need(grid.shape[0] == grid.shape[1], "canvas must be square")
    values, counts = np.unique(grid, return_counts=True)
    need(len(values) == 2, "input needs exactly two colors")
    background = int(values[int(np.argmax(counts))])
    foreground = int(values[int(np.argmin(counts))])
    stage = int(np.min(counts))
    size = int(grid.shape[0])
    need(1 <= stage <= 6 and end(stage) <= size, "invalid segment length or size")
    start = stage * (stage - 1) // 2
    return [{"size": size, "stage": stage, "background": background,
             "foreground": foreground, "corner": corner}
            for corner in range(4)
            if np.array_equal(grid, segment(size, background, foreground, corner, start, stage))]


def expected_output(info):
    stage = info["stage"]
    return segment(info["size"], info["background"], info["foreground"],
                   info["corner"], stage * (stage + 1) // 2, stage + 1)


def analyze_pair(inp, out):
    matches = [info for info in input_candidates(inp) if np.array_equal(out, expected_output(info))]
    need(len(matches) == 1, "pair does not show the next complete diagonal block")
    return matches[0]


def source_pair(inp, out):
    if inp.shape != (16, 16) or out.shape != (16, 16):
        return False
    for stage in (1, 2, 3, 4):
        original = pair_for(16, stage, (7, 2), 2)
        if np.array_equal(inp, original["input"]) and np.array_equal(out, original["output"]):
            return True
    return False


def predict_from_training(train_infos, test_input):
    candidates = input_candidates(test_input)
    need(len(candidates) in (1, 2), "test input is not a triangular block")
    if len(candidates) == 1:
        return expected_output(candidates[0])
    same_axis = {info["corner"] for info in train_infos
                 if axis(info["corner"]) == axis(candidates[0]["corner"])}
    need(len(same_axis) == 1, "training does not establish centered direction")
    selected = [info for info in candidates if info["corner"] in same_axis]
    need(len(selected) == 1, "centered input contradicts training")
    return expected_output(selected[0])


def validate_generated_puzzle(puzzle: dict) -> None:
    need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    need(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    need(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4 and
         len(puzzle["train"]) > len(puzzle["test"]), "bad pair counts")
    used_inputs, used_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            need(isinstance(pair, dict) and set(pair) == {"input", "output"}, f"{label} malformed")
            inp, out = pair["input"], pair["output"]
            validate_grid(inp, label + ".input")
            validate_grid(out, label + ".output")
            need(inp.shape == out.shape and not np.array_equal(inp, out), f"{label} changes dimensions or is unchanged")
            input_key = grid_key(inp)
            pair_key = (input_key, grid_key(out))
            need(input_key not in used_inputs and pair_key not in used_pairs, f"{label} duplicates a pair")
            need(not source_pair(inp, out), f"{label} copies the original")
            used_inputs.add(input_key)
            used_pairs.add(pair_key)
            infos[split].append(analyze_pair(inp, out))
    train_stages = {info["stage"] for info in infos["train"]}
    test_stages = {info["stage"] for info in infos["test"]}
    need(any(n + 1 in train_stages for n in train_stages), "training lacks successive lengths")
    need(max(train_stages) + 1 in test_stages, "test lacks the next length")
    need(all(n <= max(train_stages) + 1 for n in test_stages), "test skips a length")
    for pair in puzzle["test"]:
        prediction = predict_from_training(infos["train"], pair["input"].copy())
        need(np.array_equal(prediction, pair["output"]), "test is not inferable from training")


def puzzle_key(puzzle):
    return tuple(tuple((grid_key(pair["input"]), grid_key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    need(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack variety")
    need(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack variety")
    need(len({puzzle_key(p) for p in puzzles}) >= 40, "puzzles lack variety")
    sizes, corners, palettes, layouts = set(), set(), set(), set()
    train_stages, test_stages = set(), set()
    centered_tests = shared_chains = mixed_palettes = mixed_corners = familiar_tests = 0
    centered_four = centered_five = source_like = late_demo = 0
    full_shared = varied_sizes = familiar_new_canvas = familiar_new_axis = 0
    common_palettes = combined_variation = exact_fit = roomy = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        infos = {split: [analyze_pair(p["input"], p["output"]) for p in puzzle[split]]
                 for split in ("train", "test")}
        all_infos = infos["train"] + infos["test"]
        layouts.add(tuple((i["size"], i["stage"], i["corner"]) for i in all_infos))
        sizes.update(i["size"] for i in all_infos)
        corners.update(i["corner"] for i in all_infos)
        palettes.update((i["background"], i["foreground"]) for i in all_infos)
        train_stages.update(i["stage"] for i in infos["train"])
        test_stages.update(i["stage"] for i in infos["test"])
        centered_tests += any(i["size"] == i["stage"] ** 2 for i in infos["test"])
        centered_four += any(i["stage"] == 4 and i["size"] == 16 for i in infos["test"])
        centered_five += any(i["stage"] == 5 and i["size"] == 25 for i in infos["test"])
        source_like += (len(infos["train"]) == 3 and len(infos["test"]) == 1 and
                        sorted(i["stage"] for i in infos["train"]) == [1, 2, 3] and
                        infos["test"][0]["stage"] == 4 and all(i["size"] == 16 for i in all_infos))
        late_demo += (any(i["stage"] == 5 for i in infos["train"]) and
                      any(i["stage"] == 6 for i in infos["test"]))
        full_shared += len({i["size"] for i in all_infos}) == 1
        varied_sizes += len({i["size"] for i in all_infos}) > 1
        mixed_palettes += len({(i["background"], i["foreground"]) for i in all_infos}) > 1
        mixed_corners += len({i["corner"] for i in all_infos}) > 1
        common_palettes += len({(i["background"], i["foreground"]) for i in all_infos}) == 1
        combined_variation += (len({i["size"] for i in all_infos}) > 1 and
                               len({i["corner"] for i in all_infos}) > 1 and
                               len({(i["background"], i["foreground"]) for i in all_infos}) > 1)
        exact_fit += any(i["size"] == end(i["stage"]) for i in all_infos)
        roomy += any(i["size"] - end(i["stage"]) >= 10 for i in all_infos)
        familiar_tests += any(i["stage"] in {j["stage"] for j in infos["train"]} for i in infos["test"])
        familiar_new_canvas += any(
            i["stage"] in {j["stage"] for j in infos["train"]} and
            i["size"] not in {j["size"] for j in infos["train"] if j["stage"] == i["stage"]}
            for i in infos["test"])
        familiar_new_axis += any(
            i["stage"] in {j["stage"] for j in infos["train"]} and
            axis(i["corner"]) not in {axis(j["corner"]) for j in infos["train"] if j["stage"] == i["stage"]}
            for i in infos["test"])
        shared_chains += any(np.array_equal(a["output"], b["input"])
                             for a in puzzle["train"] for b in puzzle["train"] if a is not b)
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], infos[split]):
                foreground = info["foreground"]
                input_cells = {tuple(map(int, cell)) for cell in np.argwhere(pair["input"] == foreground)}
                output_cells = {tuple(map(int, cell)) for cell in np.argwhere(pair["output"] == foreground)}
                need(len(input_cells) == info["stage"] and len(output_cells) == info["stage"] + 1,
                     "foreground counts are wrong")
                _, _, dr, dc = corner_spec(info["size"], info["corner"])
                last = max(input_cells, key=lambda cell: dr * cell[0] + dc * cell[1])
                expected_cells = {(last[0] + dr * step, last[1] + dc * step)
                                  for step in range(1, info["stage"] + 2)}
                need(output_cells == expected_cells, "output does not follow the leading endpoint")
    need(len(layouts) >= 30 and len(sizes) >= 20 and len(corners) == 4, "structural diversity is insufficient")
    need(len(palettes) >= 25, "palette diversity is insufficient")
    need({1, 2, 3, 4, 5}.issubset(train_stages) and {3, 4, 5, 6}.issubset(test_stages), "stage coverage is incomplete")
    need(centered_tests >= 2 and shared_chains >= 5, "centered tests or linked chains are missing")
    need(mixed_palettes >= 10 and mixed_corners >= 5 and familiar_tests >= 5, "local variation is insufficient")
    need(centered_four >= 2 and centered_five >= 1 and source_like >= 1,
         "source-like and centered cases are missing")
    need(late_demo >= 2 and familiar_new_canvas >= 4 and familiar_new_axis >= 2,
         "later demonstrations or familiar-stage shifts are missing")
    need(full_shared >= 5 and varied_sizes >= 10 and common_palettes >= 2 and combined_variation >= 4,
         "shared and mixed puzzle regimes are missing")
    need(exact_fit >= 5 and roomy >= 5, "exact-fit or roomy canvases are missing")
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4), (1004, 6, 2)):
        puzzle = generate_puzzle(seed, train, test)
        need((len(puzzle["train"]), len(puzzle["test"])) == (train, test), "explicit counts ignored")
    for seed, options in ((1101, {"num_train": None, "num_test": 1}),
                          (1102, {"num_train": None, "num_test": 4}),
                          (1103, {"num_train": 2, "num_test": None}),
                          (1104, {"num_train": 10, "num_test": None})):
        puzzle = generate_puzzle(seed, **options)
        for name, value in options.items():
            if value is not None:
                need(len(puzzle["train" if name == "num_train" else "test"]) == value, "partial count ignored")
    need(puzzle_key(generate_puzzle(314159, 7, 3)) == puzzle_key(generate_puzzle(314159, 7, 3)), "not deterministic")
    for options in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                    {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(12345, **options)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {options}")
