"""Generate ARC puzzles that mark complete monochromatic rows."""

import numpy as np


_SOURCE_INPUTS = (
    (2, 9, 2, 4, 4, 4, 9, 9, 9),
    (7, 3, 3, 6, 6, 6, 3, 7, 7),
    (2, 2, 4, 2, 2, 4, 1, 1, 1),
    (4, 4, 4, 2, 3, 2, 2, 3, 3),
    (4, 4, 4, 3, 2, 3, 8, 8, 8),
)
_SOURCE_KEYS = frozenset(((3, 3), values) for values in _SOURCE_INPUTS)
_AUTO_COUNTS = ((2, 1), (3, 2), (4, 3), (5, 4), (6, 1),
                (7, 2), (8, 3), (9, 4), (10, 1))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(name, value, minimum, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _counts(seed, num_train, num_test):
    train = None if num_train is None else _integer("num_train", num_train, 2, 10)
    test = None if num_test is None else _integer("num_test", num_test, 1, 4)
    if train is None and test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if train is None:
        train = max(2, test + 1) + seed % (11 - max(2, test + 1))
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _key(grid):
    return (tuple(int(x) for x in grid.shape),
            tuple(int(x) for x in grid.flat))


def _mask(grid):
    return np.all(grid == grid[:, :1], axis=1)


def _output(grid, positive, negative):
    colors = np.where(_mask(grid), positive, negative).astype(np.int8)
    return np.repeat(colors[:, None], grid.shape[1], axis=1)


def _shape(rng, mode):
    if mode == "source_shift":
        return (3, 3)
    if mode == "small":
        return (int(rng.integers(3, 6)), int(rng.integers(3, 6)))
    if mode == "medium":
        return (int(rng.integers(3, 10)), int(rng.integers(3, 11)))
    height = 30 if rng.random() < 0.08 else int(rng.integers(3, 17))
    width = 30 if rng.random() < 0.08 else int(rng.integers(3, 17))
    return height, width


def _random_mask(rng, height, category="mixed", excluded=()):
    excluded = set(excluded)
    if category in ("all_positive", "all_negative"):
        sole = tuple([category == "all_positive"] * height)
        if sole in excluded:
            category = "mixed"
    for _ in range(80):
        if category == "all_positive":
            mask = np.ones(height, dtype=bool)
        elif category == "all_negative":
            mask = np.zeros(height, dtype=bool)
        else:
            chance = float(rng.choice((0.25, 0.4, 0.6, 0.75)))
            mask = rng.random(height) < chance
            if not np.any(mask) or not np.any(~mask):
                continue
            required_negatives = {"two_negatives": 2, "three_negatives": 3}.get(category, 0)
            if required_negatives and np.count_nonzero(~mask) < required_negatives:
                continue
        if tuple(bool(x) for x in mask) not in excluded:
            return mask
    # At most 30 single-positive masks suffice for the four requested tests.
    if category in ("mixed", "two_negatives", "three_negatives"):
        for position in range(height):
            mask = np.zeros(height, dtype=bool)
            mask[position] = True
            if tuple(bool(x) for x in mask) not in excluded:
                return mask
    raise RuntimeError("no available row mask")


def _mixed_row(rng, palette, width, style=None):
    colors = rng.choice(palette, size=2, replace=False)
    a, b = (int(colors[0]), int(colors[1]))
    if style is None:
        style = str(rng.choice(("exception", "alternating", "runs", "balanced", "random")))
    if style in ("first", "middle", "last", "exception"):
        row = np.full(width, a, dtype=np.int8)
        if style == "first":
            position = 0
        elif style == "last":
            position = width - 1
        else:
            position = int(rng.integers(1, width - 1))
        row[position] = b
    elif style == "alternating":
        row = np.array([a if i % 2 == 0 else b for i in range(width)], dtype=np.int8)
    elif style == "runs":
        cut = int(rng.integers(1, width))
        row = np.array([a] * cut + [b] * (width - cut), dtype=np.int8)
    elif style == "balanced":
        row = np.array([a] * (width // 2) + [b] * (width - width // 2), dtype=np.int8)
        rng.shuffle(row)
    else:
        row = np.asarray(rng.choice(palette, size=width), dtype=np.int8)
        if np.all(row == row[0]):
            row[-1] = b if int(row[0]) == a else a
    return row


def _palette(rng, mode, split, previous, positive, negative, mask):
    if mode != "source_shift" and previous and rng.random() < 0.28:
        return previous[int(rng.integers(len(previous)))].copy()
    if mode == "source_shift":
        size = 3 if split == "train" else 4
        if bool(np.all(mask)):
            size = min(size, len(mask))
    elif mode == "small":
        size = int(rng.integers(2, 5))
    elif mode == "medium":
        size = int(rng.integers(2, 8))
    else:
        size = int(rng.integers(2, 11))
    if rng.random() < 0.42:
        included = int(rng.choice((positive, negative)))
        rest = rng.choice([x for x in range(10) if x != included],
                          size=size - 1, replace=False)
        return np.array([included, *rest], dtype=np.int8)
    return np.asarray(rng.choice(10, size=size, replace=False), dtype=np.int8)


def _input(rng, shape, mask, palette, forced, used, exact_palette):
    height, width = shape
    for _ in range(300):
        grid = np.empty(shape, dtype=np.int8)
        positive_colors = []
        mixed_rows = []
        for i in range(height):
            if mask[i]:
                if positive_colors and rng.random() < 0.35:
                    color = int(rng.choice(positive_colors))
                else:
                    color = int(rng.choice(palette))
                grid[i] = color
                positive_colors.append(color)
            elif i in forced:
                grid[i] = _mixed_row(rng, palette, width, forced[i])
                mixed_rows.append(grid[i].copy())
            elif mixed_rows and rng.random() < 0.14:
                grid[i] = mixed_rows[int(rng.integers(len(mixed_rows)))]
            else:
                grid[i] = _mixed_row(rng, palette, width)
                mixed_rows.append(grid[i].copy())
        key = _key(grid)
        if key in used or key in _SOURCE_KEYS:
            continue
        if exact_palette and np.unique(grid).size != len(palette):
            continue
        return grid
    raise RuntimeError("could not construct a fresh input")


def _assign_training_contrasts(rng, masks):
    first_slots = [(0, row_index) for row_index, positive in enumerate(masks[0])
                   if not positive]
    slots = first_slots if len(first_slots) >= 3 else first_slots + [
        (1, row_index) for row_index, positive in enumerate(masks[1])
        if not positive]
    _require(len(slots) >= 3, "training lacks room for equality contrasts")
    rng.shuffle(slots)
    styles = ["first", "middle", "last"]
    rng.shuffle(styles)
    forced = [dict() for _ in masks]
    for (pair_index, row_index), style in zip(slots, styles):
        forced[pair_index][row_index] = style
    return forced


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _integer("seed", seed, 0, 2**63 - 1)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    if rng.random() < 0.12:
        positive, negative = 5, 0
    else:
        positive, negative = (int(x) for x in rng.choice(10, size=2, replace=False))
    mode = str(rng.choice(("source_shift", "small", "medium", "rich"),
                          p=(0.16, 0.22, 0.34, 0.28)))
    shapes = [_shape(rng, mode) for _ in range(train_count + test_count)]
    if mode != "source_shift":
        widest_test = max(width for _, width in shapes[train_count:])
        first_width = max(shapes[0][1], widest_test)
        first_height = max(shapes[0][0], 4) if first_width >= 6 else shapes[0][0]
        shapes[0] = (first_height, first_width)
        if len(set(shapes)) == 1:
            height, width = shapes[-1]
            shapes[-1] = (height, 3 if width != 3 else 4)

    masks = []
    if mode == "source_shift":
        adjacent = np.array(rng.choice(((1, 1, 0), (0, 1, 1))), dtype=bool)
        masks.extend((adjacent, ~adjacent))
    else:
        first_category = "three_negatives" if shapes[0][1] >= 6 else "two_negatives"
        masks.append(_random_mask(rng, shapes[0][0], first_category))
        masks.append(_random_mask(rng, shapes[1][0], "mixed"))
    for index in range(2, train_count):
        category = str(rng.choice(("mixed", "all_positive", "all_negative"),
                                  p=(0.86, 0.07, 0.07)))
        masks.append(_random_mask(rng, shapes[index][0], category))
    used_test_masks = set()
    for index in range(test_count):
        if mode == "source_shift" and index == 0:
            mask = np.array((1, 0, 1), dtype=bool)
        else:
            category = str(rng.choice(("mixed", "all_positive", "all_negative"),
                                      p=(0.76, 0.12, 0.12)))
            mask = _random_mask(rng, shapes[train_count + index][0],
                                category, used_test_masks)
        masks.append(mask)
        used_test_masks.add(tuple(bool(x) for x in mask))

    forced = _assign_training_contrasts(rng, masks[:train_count])
    puzzle = {"train": [], "test": []}
    used = set()
    previous_palettes = []
    for index, (shape, mask) in enumerate(zip(shapes, masks)):
        split = "train" if index < train_count else "test"
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        for _ in range(50):
            palette = _palette(pair_rng, mode, split, previous_palettes,
                               positive, negative, mask)
            try:
                input_grid = _input(pair_rng, shape, mask, palette,
                                    forced[index] if split == "train" else {},
                                    used, mode == "source_shift")
            except RuntimeError:
                continue
            output_grid = _output(input_grid, positive, negative)
            if not np.array_equal(input_grid, output_grid):
                break
        else:
            raise RuntimeError("could not construct a nontrivial pair")
        puzzle[split].append({"input": input_grid, "output": output_grid})
        used.add(_key(input_grid))
        previous_palettes.append(palette)
    validate_generated_puzzle(puzzle)
    return puzzle


def _exception_styles(row):
    values, counts = np.unique(row, return_counts=True)
    if len(values) != 2 or sorted(counts.tolist()) != [1, len(row) - 1]:
        return set()
    exceptional = values[np.where(counts == 1)[0][0]]
    position = int(np.flatnonzero(row == exceptional)[0])
    if position == 0:
        return {"first"}
    if position == len(row) - 1:
        return {"last"}
    return {"middle"}


def _learn_colors(train_pairs):
    roles = {True: set(), False: set()}
    for pair in train_pairs:
        grid, output = pair["input"], pair["output"]
        for i, positive in enumerate(_mask(grid)):
            _require(np.all(output[i] == output[i, 0]), "output row is not solid")
            roles[bool(positive)].add(int(output[i, 0]))
    _require(len(roles[True]) == len(roles[False]) == 1,
             "training must expose one color for each row class")
    positive, negative = next(iter(roles[True])), next(iter(roles[False]))
    _require(positive != negative, "row classes need distinct output colors")
    return positive, negative


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    used = set()
    styles = set()
    wide_training_styles = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs exactly input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         f"{name} must be a 2D NumPy array")
                _require(grid.dtype.kind in "iu" and 1 <= grid.shape[0] <= 30
                         and 1 <= grid.shape[1] <= 30, f"{name} type or shape invalid")
                _require(0 <= int(grid.min()) and int(grid.max()) <= 9,
                         f"{name} has an illegal color")
            grid, output = pair["input"], pair["output"]
            _require(grid.shape == output.shape and grid.shape[0] >= 3
                     and grid.shape[1] >= 3, "row-classification shape invalid")
            _require(not np.array_equal(grid, output), "input equals output")
            key = _key(grid)
            _require(key not in used and key not in _SOURCE_KEYS,
                     "duplicate or source input")
            used.add(key)
            if split == "train":
                pair_styles = set()
                for row, positive in zip(grid, _mask(grid)):
                    if not positive:
                        row_styles = _exception_styles(row)
                        styles.update(row_styles)
                        pair_styles.update(row_styles)
                wide_training_styles.append((grid.shape[1], pair_styles))
    positive, negative = _learn_colors(puzzle["train"])
    _require(styles == {"first", "middle", "last"},
             "training lacks first/interior/last mismatch contrasts")
    widest_test = max(pair["input"].shape[1] for pair in puzzle["test"])
    if widest_test >= 6:
        _require(any(width >= widest_test and
                     pair_styles == {"first", "middle", "last"}
                     for width, pair_styles in wide_training_styles),
                 "wide tests lack equally wide training contrasts")
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(np.array_equal(pair["output"],
                                    _output(pair["input"], positive, negative)),
                     "pair violates the learned row rule")
    test_masks = [tuple(bool(x) for x in _mask(pair["input"]))
                  for pair in puzzle["test"]]
    _require(len(test_masks) == len(set(test_masks)),
             "multiple tests must have distinct answer masks")


def _same_puzzle(left, right):
    return all(len(left[split]) == len(right[split]) and
               all(np.array_equal(a[name], b[name])
                   for a, b in zip(left[split], right[split])
                   for name in ("input", "output"))
               for split in ("train", "test"))


def _predict_from_demonstrations(train_pairs, test_input):
    positive, negative = _learn_colors(train_pairs)
    result = np.empty(test_input.shape, dtype=np.int8)
    for i, row in enumerate(test_input):
        result[i] = positive if np.all(row == row[0]) else negative
    return result


def self_test() -> None:
    for seed in (7, 73):
        _require(_same_puzzle(generate_puzzle(seed), generate_puzzle(seed)),
                 "automatic generation is not deterministic")
        _require(_same_puzzle(generate_puzzle(seed, 7, 3),
                             generate_puzzle(seed, 7, 3)),
                 "explicit generation is not deterministic")
    cases = ((2, 1), (10, 1), (10, 4), (6, 2),
             (None, 1), (None, 4), (2, None), (10, None))
    for seed, (train, test) in enumerate(cases, start=100):
        puzzle = generate_puzzle(seed, train, test)
        _require(train is None or len(puzzle["train"]) == train,
                 "requested training count changed")
        _require(test is None or len(puzzle["test"]) == test,
                 "requested test count changed")
        validate_generated_puzzle(puzzle)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(201, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")

    corpus = [generate_puzzle(seed) for seed in range(42)]
    train_counts, test_counts = set(), set()
    shapes, palettes, masks, test_masks = set(), set(), set(), set()
    positive_colors, negative_colors, fingerprints = set(), set(), set()
    familiar_color_pairs = 0
    overlap = two_color = rich_palette = all_positive = all_negative = 0
    varied_shapes = varied_test_masks = 0
    for puzzle in corpus:
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        positive, negative = _learn_colors(puzzle["train"])
        familiar_color_pairs += (positive, negative) == (5, 0)
        positive_colors.add(positive)
        negative_colors.add(negative)
        pair_shapes = set()
        fingerprint = []
        local_test_masks = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                fingerprint.append((_key(grid), _key(output)))
                pair_shapes.add(grid.shape)
                shapes.add(grid.shape)
                palette = tuple(int(x) for x in np.unique(grid))
                palettes.add(palette)
                overlap += bool(positive in palette or negative in palette)
                two_color += len(palette) == 2
                rich_palette += len(palette) >= 5
                row_mask = tuple(bool(x) for x in _mask(grid))
                masks.add(row_mask)
                if split == "test":
                    test_masks.add(row_mask)
                    local_test_masks.add(row_mask)
                    all_positive += all(row_mask)
                    all_negative += not any(row_mask)
                    _require(np.array_equal(
                        output, _predict_from_demonstrations(puzzle["train"], grid)),
                        "test not determined by training and test input")
        fingerprints.add(tuple(fingerprint))
        varied_shapes += len(pair_shapes) > 1
        varied_test_masks += len(local_test_masks) > 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack diversity")
    _require(len(fingerprints) == 42 and len(shapes) >= 20,
             "puzzles or shapes lack diversity")
    _require(varied_shapes >= 24 and varied_test_masks >= 20,
             "within-puzzle shapes or test masks lack diversity")
    _require(len(palettes) >= 35 and two_color >= 12 and rich_palette >= 35,
             "palettes lack size or membership diversity")
    _require(len(positive_colors) >= 8 and len(negative_colors) >= 8
             and overlap >= 50, "output roles or input overlap lack diversity")
    _require(familiar_color_pairs >= 1,
             "42-seed corpus lacks the source output-color convention")
    _require(len(test_masks) >= 25 and all_positive >= 2 and all_negative >= 2,
             "test row decisions lack diversity")
