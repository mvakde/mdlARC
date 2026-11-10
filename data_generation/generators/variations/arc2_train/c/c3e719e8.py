"""Generate modal-color mask-and-stamp ARC puzzles."""

import numpy as np


_SHAPES = (
    (2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (2, 5), (5, 2),
    (3, 3), (3, 4), (4, 3), (3, 5), (5, 3), (4, 4), (4, 5),
    (5, 4), (5, 5),
)
_SOURCE_INPUTS = {
    (3, 3, (8, 6, 8, 3, 3, 8, 8, 8, 8)),
    (3, 3, (3, 8, 7, 9, 3, 8, 7, 9, 3)),
    (3, 3, (6, 9, 9, 4, 6, 8, 9, 9, 8)),
    (3, 3, (1, 1, 7, 7, 4, 1, 5, 1, 7)),
}
_AUTOMATIC_COUNTS = (
    (2, 1), (3, 1), (4, 2), (5, 1), (5, 2), (6, 3),
    (7, 1), (7, 2), (8, 3), (9, 4), (10, 1), (10, 4),
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _key(grid):
    return (grid.shape[0], grid.shape[1], tuple(int(x) for x in grid.flat))


def _optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _counts(seed, rng, num_train, num_test):
    train = _optional_count(num_train, "num_train", 2, 10)
    test = _optional_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        return _AUTOMATIC_COUNTS[seed % len(_AUTOMATIC_COUNTS)]
    if train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _shapes(rng, train_count, test_count):
    profile = float(rng.random())
    total = train_count + test_count
    if profile < 0.20:
        return [(3, 3)] * total
    if profile < 0.42:
        shape = _SHAPES[int(rng.integers(len(_SHAPES)))]
        return [shape] * total
    result = []
    for index in range(total):
        if index >= train_count and rng.random() < 0.12:
            shape = (5, 5)
        elif rng.random() < 0.30:
            shape = (3, 3)
        else:
            shape = _SHAPES[int(rng.integers(len(_SHAPES)))]
        if index == 1 and shape == result[0]:
            alternatives = [item for item in _SHAPES if item != shape]
            shape = alternatives[int(rng.integers(len(alternatives)))]
        result.append(shape)
    return result


def _feasible_counts(shape, palette_size):
    cells = shape[0] * shape[1]
    return [k for k in range(2, cells - 1)
            if cells - k <= (palette_size - 1) * (k - 1)]


def _categories(shape, palette_size):
    height, width = shape
    counts = _feasible_counts(shape, palette_size)
    result = ["random", "cluster", "symmetric", "diagonal", "irregular"]
    if any(k <= max(2, height * width // 3) for k in counts):
        result.append("sparse")
    if any(k >= (height * width + 1) // 2 for k in counts):
        result.append("dense")
    if any(k <= height * width - min(height, width) for k in counts):
        result.append("gapped")
    if max(shape) <= 4 and height * width >= 6 and 4 in counts:
        result.append("four_cover")
    return result


def _random_mask(rng, shape, count):
    mask = np.zeros(shape, dtype=bool)
    mask.flat[rng.choice(mask.size, size=count, replace=False)] = True
    return mask


def _symmetric_mask(rng, shape, counts):
    height, width = shape
    for _ in range(36):
        axis = int(rng.integers(3))
        orbits = []
        unseen = set(range(height * width))
        while unseen:
            first = min(unseen)
            row, col = divmod(first, width)
            if axis == 0:
                mate = row * width + width - 1 - col
            elif axis == 1:
                mate = (height - 1 - row) * width + col
            else:
                mate = (height - 1 - row) * width + width - 1 - col
            orbit = {first, mate}
            unseen -= orbit
            orbits.append(orbit)
        mask = np.zeros(shape, dtype=bool)
        for i in rng.permutation(len(orbits)):
            if rng.random() < 0.56:
                for position in orbits[int(i)]:
                    mask.flat[position] = True
        if int(mask.sum()) in counts:
            return mask
    return _random_mask(rng, shape, int(rng.choice(counts)))


def _sample_mask(rng, shape, palette_size, category):
    height, width = shape
    counts = _feasible_counts(shape, palette_size)
    if category == "sparse":
        counts = [k for k in counts if k <= max(2, height * width // 3)]
    elif category == "dense":
        counts = [k for k in counts if k >= (height * width + 1) // 2]
    elif category == "gapped":
        counts = [k for k in counts if k <= height * width - min(shape)]
    elif category == "four_cover":
        counts = [4]
    _require(bool(counts), "requested mask category has no feasible count")
    if category == "symmetric":
        return _symmetric_mask(rng, shape, counts)
    if category == "four_cover":
        for _ in range(60):
            rows = list(range(height)) + list(rng.choice(height, size=4 - height))
            cols = list(range(width)) + list(rng.choice(width, size=4 - width))
            rng.shuffle(rows)
            rng.shuffle(cols)
            positions = set(zip(rows, cols))
            if len(positions) == 4:
                mask = np.zeros(shape, dtype=bool)
                for row, col in positions:
                    mask[row, col] = True
                return mask
    count = int(rng.choice(counts))
    if category == "gapped":
        if rng.random() < 0.5:
            row = int(rng.integers(height))
            available = [r * width + c for r in range(height) for c in range(width) if r != row]
        else:
            col = int(rng.integers(width))
            available = [r * width + c for r in range(height) for c in range(width) if c != col]
        if len(available) < count:
            return _sample_mask(rng, shape, palette_size, "random")
        mask = np.zeros(shape, dtype=bool)
        mask.flat[rng.choice(available, size=count, replace=False)] = True
        return mask
    if category == "cluster":
        mask = np.zeros(shape, dtype=bool)
        mask.flat[int(rng.integers(mask.size))] = True
        while int(mask.sum()) < count:
            frontier = set()
            for row, col in np.argwhere(mask):
                for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        frontier.add((nr, nc))
            candidates = sorted(frontier)
            row, col = candidates[int(rng.integers(len(candidates)))]
            mask[row, col] = True
        return mask
    if category == "diagonal":
        mask = np.zeros(shape, dtype=bool)
        reverse = bool(rng.integers(2))
        for row in range(min(shape)):
            col = width - 1 - row if reverse else row
            mask[row, col] = True
        if int(mask.sum()) not in counts:
            desired = min((k for k in counts if k >= int(mask.sum())), default=counts[-1])
            available = np.flatnonzero(~mask.ravel())
            mask.flat[rng.choice(available, size=desired - int(mask.sum()), replace=False)] = True
        return mask
    for _ in range(24):
        mask = _random_mask(rng, shape, count)
        if category != "irregular" or not (
            np.array_equal(mask, np.flipud(mask)) or
            np.array_equal(mask, np.fliplr(mask)) or
            np.array_equal(mask, np.rot90(mask, 2))
        ):
            return mask
    return mask


def _make_stamp(rng, mask, trigger, colors, required_filler=None):
    inactive = int(mask.size - mask.sum())
    capacity = int(mask.sum()) - 1
    fillers = [int(color) for color in colors if int(color) != trigger]
    minimum = max(2, (inactive + capacity - 1) // capacity)
    maximum = min(len(fillers), inactive)
    _require(minimum <= maximum, "insufficient filler colors for unique mode")
    style = int(rng.choice(4, p=[0.37, 0.24, 0.23, 0.16]))
    if style == 0 or style == 2:
        palette_size = minimum
    elif style == 1:
        palette_size = maximum
    else:
        palette_size = int(rng.integers(minimum, maximum + 1))
    palette = []
    if required_filler is not None and required_filler in fillers:
        palette.append(int(required_filler))
    remaining = [color for color in fillers if color not in palette]
    palette.extend(int(x) for x in rng.choice(remaining, size=palette_size - len(palette), replace=False))
    amounts = {color: 1 for color in palette}
    if style == 2 and capacity > 1 and inactive - palette_size >= capacity - 1:
        amounts[palette[0]] = capacity
    slots = [color for color in palette for _ in range(capacity - amounts[color])]
    additional = inactive - sum(amounts.values())
    if additional:
        for color in rng.choice(slots, size=additional, replace=False):
            amounts[int(color)] += 1
    values = [color for color in palette for _ in range(amounts[color])]
    if style in (1, 3):
        rng.shuffle(values)
    else:
        rng.shuffle(palette)
        values = [color for color in palette for _ in range(amounts[color])]
    stamp = np.full(mask.shape, trigger, dtype=np.int8)
    stamp.flat[np.flatnonzero(~mask.ravel())] = values
    return stamp


def _output(stamp, mask, background):
    height, width = stamp.shape
    result = np.full((height * height, width * width), background, dtype=np.int8)
    for row, col in np.argwhere(mask):
        result[row * height:(row + 1) * height, col * width:(col + 1) * width] = stamp
    return result


def _pair(rng, shape, background, colors, trigger, previous_masks, required_filler):
    possible = _categories(shape, len(colors))
    for attempt in range(80):
        category = possible[int(rng.integers(len(possible)))]
        reusable = [mask for mask in previous_masks if mask.shape == shape and
                    int(mask.sum()) in _feasible_counts(shape, len(colors))]
        if reusable and rng.random() < 0.17:
            mask = reusable[int(rng.integers(len(reusable)))].copy()
        else:
            mask = _sample_mask(rng, shape, len(colors), category)
        stamp = _make_stamp(rng, mask, trigger, colors, required_filler)
        if _key(stamp) not in _SOURCE_INPUTS:
            return stamp, _output(stamp, mask, background), mask
    raise AssertionError("could not construct a novel input")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    shapes = _shapes(rng, train_count, test_count)
    background = int(rng.integers(10))
    available = [color for color in rng.permutation(10) if color != background]
    bank_size = int(rng.integers(5, 9))
    train_bank = [int(color) for color in available[:bank_size]]
    all_colors = [int(color) for color in available]
    first_trigger, second_trigger = train_bank[:2]
    train_triggers = [first_trigger, second_trigger]
    train_triggers.extend(int(rng.choice(train_bank)) for _ in range(train_count - 2))
    puzzle = {"train": [], "test": []}
    masks = []
    for index in range(train_count):
        required = second_trigger if index == 0 and rng.random() < 0.60 else None
        stamp, output, mask = _pair(rng, shapes[index], background, train_bank,
                                    train_triggers[index], masks, required)
        puzzle["train"].append({"input": stamp, "output": output})
        masks.append(mask)
    seen_train = {int(x) for pair in puzzle["train"] for x in pair["input"].flat}
    train_trigger_set = set(train_triggers)
    former_fillers = sorted(seen_train - train_trigger_set)
    new_colors = sorted(set(all_colors) - seen_train)
    for index in range(test_count):
        mode = int(rng.integers(3))
        if mode == 0 or (mode == 2 and not new_colors):
            trigger = int(rng.choice(train_triggers))
        elif mode == 1 and former_fillers:
            trigger = int(rng.choice(former_fillers))
        elif new_colors:
            trigger = int(rng.choice(new_colors))
        else:
            trigger = int(rng.choice(train_bank))
        required = None
        other_new = [color for color in new_colors if color != trigger]
        if trigger in new_colors and other_new and rng.random() < 0.75:
            required = int(rng.choice(other_new))
        stamp, output, mask = _pair(rng, shapes[train_count + index], background,
                                    all_colors, trigger, masks, required)
        puzzle["test"].append({"input": stamp, "output": output})
        masks.append(mask)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    stamp, output = pair["input"], pair["output"]
    height, width = stamp.shape
    colors, counts = np.unique(stamp, return_counts=True)
    _require(len(colors) >= 3, "stamp needs multiple colors")
    _require(int((counts == counts.max()).sum()) == 1, "modal color must be unique")
    trigger = int(colors[int(np.argmax(counts))])
    mask = stamp == trigger
    _require(2 <= int(mask.sum()) <= stamp.size - 2, "mask must show copies and blanks")
    _require(output.shape == (height * height, width * width), "output size mismatch")
    blanks = []
    for row in range(height):
        for col in range(width):
            block = output[row * height:(row + 1) * height,
                           col * width:(col + 1) * width]
            if mask[row, col]:
                _require(np.array_equal(block, stamp), "copy block differs from full stamp")
            else:
                _require(bool(np.all(block == block[0, 0])), "inactive block is not uniform")
                blanks.append(int(block[0, 0]))
    _require(bool(blanks) and len(set(blanks)) == 1, "inactive blocks disagree")
    background = blanks[0]
    _require(background not in colors, "background occurs in stamp")
    return trigger, mask, background


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    inputs = set()
    pairs = set()
    backgrounds = set()
    train_triggers = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D array")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
                _require(all(1 <= side <= 30 for side in grid.shape), "grid exceeds ARC bounds")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, "invalid ARC color")
            stamp, output = pair["input"], pair["output"]
            input_key = _key(stamp)
            pair_key = (input_key, _key(output))
            _require(input_key not in inputs and pair_key not in pairs, "duplicate pair")
            _require(input_key not in _SOURCE_INPUTS, "source pair copied")
            inputs.add(input_key)
            pairs.add(pair_key)
            _require(stamp.shape in _SHAPES, "unsupported stamp shape")
            trigger, _, background = _analyze_pair(pair)
            backgrounds.add(background)
            if split == "train":
                train_triggers.add(trigger)
    _require(len(backgrounds) == 1, "neutral background changes within puzzle")
    _require(len(train_triggers) >= 2, "training does not vary modal color")


def _equal(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[role], b[role]) for a, b in zip(first[split], second[split])
                   for role in ("input", "output")) for split in ("train", "test"))


def _predict_from_training(train, test_input):
    neutral_colors = set()
    for pair in train:
        stamp, output = pair["input"], pair["output"]
        height, width = stamp.shape
        colors, counts = np.unique(stamp, return_counts=True)
        modal = int(colors[int(np.argmax(counts))])
        for row in range(height):
            for col in range(width):
                if stamp[row, col] != modal:
                    neutral_colors.add(int(output[row * height, col * width]))
    _require(len(neutral_colors) == 1, "training does not identify one neutral color")
    background = neutral_colors.pop()
    height, width = test_input.shape
    colors, counts = np.unique(test_input, return_counts=True)
    modal = int(colors[int(np.argmax(counts))])
    prediction = np.full((height * height, width * width), background, dtype=np.int8)
    for row in range(height):
        for col in range(width):
            if test_input[row, col] == modal:
                prediction[row * height:(row + 1) * height,
                           col * width:(col + 1) * width] = test_input
    return prediction


def self_test() -> None:
    validate_generated_puzzle(generate_puzzle(12345))
    for seed, train, test in ((54321, 7, 3), (61001, 2, 1), (61002, 10, 1),
                              (61003, 10, 4), (62001, None, 1),
                              (62002, None, 4), (62003, 2, None),
                              (62004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "explicit train count ignored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "explicit test count ignored")
    one = generate_puzzle(918273, 6, 2)
    _require(_equal(one, generate_puzzle(918273, 6, 2)), "generation is nondeterministic")
    _require(not _equal(one, generate_puzzle(918274, 6, 2)), "adjacent seeds coincide")
    for index, options in enumerate(({"num_train": 1}, {"num_train": 11},
                                     {"num_test": 0}, {"num_test": 5},
                                     {"num_train": 2, "num_test": 2},
                                     {"num_train": 3, "num_test": 4})):
        try:
            generate_puzzle(63000 + index, **options)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corpus = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in corpus}
    test_counts = {len(p["test"]) for p in corpus}
    shapes = set()
    backgrounds = set()
    copy_counts = set()
    mask_keys = set()
    stamp_keys = set()
    mixed = 0
    four_train = four_test = 0
    repeated_test_trigger = novel_test_trigger = 0
    role_switches = reused_masks = multiple_new_colors = 0
    split_features = {"train": {name: 0 for name in ("sparse", "dense", "symmetric", "gap",
                                                    "diagonal", "compact", "high", "near")},
                      "test": {name: 0 for name in ("sparse", "dense", "symmetric", "gap",
                                                   "diagonal", "compact", "high", "near")}}
    for puzzle in corpus:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["test"]:
            _require(np.array_equal(_predict_from_training(puzzle["train"], pair["input"]),
                                    pair["output"]), "training does not determine test output")
        training_shapes = {pair["input"].shape for pair in puzzle["train"]}
        mixed += len(training_shapes) > 1
        train_roles = {_analyze_pair(pair)[0] for pair in puzzle["train"]}
        train_colors = {int(x) for pair in puzzle["train"] for x in pair["input"].flat}
        prior_fillers = set()
        seen_masks = {}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                trigger, mask, background = _analyze_pair(pair)
                stamp = pair["input"]
                colors, frequencies = np.unique(stamp, return_counts=True)
                mask_key = _key(mask)
                if mask_key in seen_masks and seen_masks[mask_key] != _key(stamp):
                    reused_masks += 1
                seen_masks[mask_key] = _key(stamp)
                backgrounds.add(background)
                shapes.add(stamp.shape)
                copy_counts.add(int(mask.sum()))
                mask_keys.add(mask_key)
                stamp_keys.add(_key(stamp))
                features = split_features[split]
                features["sparse"] += int(mask.sum()) <= max(2, stamp.size // 3)
                features["dense"] += int(mask.sum()) >= (stamp.size + 1) // 2
                features["symmetric"] += any(np.array_equal(mask, flip) for flip in
                                             (np.flipud(mask), np.fliplr(mask), np.rot90(mask, 2)))
                features["gap"] += not mask.any(axis=0).all() or not mask.any(axis=1).all()
                features["diagonal"] += int(mask.sum()) == min(stamp.shape) and any(
                    all(bool(mask[row, row if not reverse else stamp.shape[1] - 1 - row])
                        for row in range(min(stamp.shape))) for reverse in (False, True))
                features["compact"] += len(colors) <= 4
                features["high"] += len(colors) >= 6
                features["near"] += any(int(frequency) == int(mask.sum()) - 1 and
                                        frequency > 1 for color, frequency in zip(colors, frequencies)
                                        if color != trigger)
                if int(mask.sum()) == 4 and mask.any(axis=0).all() and mask.any(axis=1).all():
                    if split == "train":
                        four_train += 1
                    else:
                        four_test += 1
                if split == "test":
                    repeated_test_trigger += trigger in train_roles
                    novel_test_trigger += trigger not in train_colors
                    multiple_new_colors += trigger not in train_colors and len(set(map(int, colors)) - train_colors) >= 2
                role_switches += trigger in prior_fillers
                prior_fillers.update(int(color) for color in colors if color != trigger)
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity too low")
    _require((2, 2) in shapes and (3, 3) in shapes and (5, 5) in shapes and
             len(shapes) >= 10, "shape diversity too low")
    _require(mixed >= 12, "too few mixed-shape puzzles")
    _require(len(backgrounds) >= 5, "neutral colors do not vary")
    _require(len(copy_counts) >= 9 and len(mask_keys) >= 90, "mask diversity too low")
    _require(len(stamp_keys) >= 200, "stamp diversity too low")
    _require(four_train >= 2 and four_test >= 1, "four-copy coverage absent from a split")
    _require(repeated_test_trigger >= 3 and novel_test_trigger >= 2,
             "test-trigger roles insufficiently varied")
    _require(role_switches >= 3 and reused_masks >= 1 and multiple_new_colors >= 1,
             "color-role or repeated-mask examples are missing")
    _require(all(all(value > 0 for value in features.values()) for features in split_features.values()),
             "a requested mask or stamp style is absent from a split")
